"""One trial: train one architecture on one database, write a flat metrics.json.

That contract (parameters in, `metrics.json` out) is all the Ubunye Research
Engine needs, so every arm of every claim in this study is this script under
different flags.

    python scripts/run_experiment.py --arch esn --dataset wsd --seed 1 --out RUNS/wsd_esn_s1

Everything runs on the GPU: the Chan-Vese sequences are generated there, the
whole database stays resident in VRAM, batching and augmentation are CUDA
kernels, and the reservoirs use sparse CUDA matrix multiplies. There is no CPU
dataloader in the loop.

Defaults reproduce Mashinini (2022) section 3.6 (binary cross entropy, SGD with
momentum 0.9, lr 0.1 stepped five times, batch 32, window 1, fully connected
head), so any departure from the thesis is visible as a flag on the command line.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from levelset.data import build_cache
from levelset.gpu_data import GPUSequences, to_gpu_splits
from levelset.metrics import Counts
from levelset.models import (RidgeReadoutModel, build, firing_rate, ridge_readout_gpu,
                             trainable_parameters)

ARCHES = ["esn", "lsm", "rnn", "lstm", "gru", "3dcnn", "noise", "copy"]


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=ARCHES)
    p.add_argument("--dataset", default="wsd", choices=["wsd", "bsd", "cifar10", "cifar100"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default="T:/datasets/levelset")
    p.add_argument("--cache-dir", default="T:/datasets/levelset/cache")

    # Task. window=1 and head=fc are the thesis; the rest is this study.
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--head", default="fc", choices=["fc", "spatial"])
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--rollout", type=int, default=25)
    p.add_argument("--horizon", type=int, default=1,
                   help="predict M_{t+horizon}; the thesis is 1, which copying nearly solves")
    p.add_argument("--eval-horizons", default="1,5,10,25",
                   help="extra horizons scored at test time, comma separated")

    # Optimisation (thesis section 3.6.2).
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--augment", type=int, default=1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--train-batches", type=int, default=0, help="0 means the whole split")

    # Architecture.
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--fc", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # Reservoir, shared by the echo state network and the liquid.
    p.add_argument("--spectral-radius", type=float, default=1.0)
    p.add_argument("--leak", type=float, default=0.0713)
    p.add_argument("--sparsity", type=float, default=0.8)
    p.add_argument("--input-scaling", type=float, default=1.0)
    p.add_argument("--dense-reservoir", type=int, default=0)
    p.add_argument("--readout", default="sgd", choices=["sgd", "ridge"])
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--freeze-encoder", type=int, default=0)

    # Liquid state machine only.
    p.add_argument("--lsm-beta", type=float, default=0.9, help="membrane decay")
    p.add_argument("--lsm-alpha", type=float, default=0.8, help="synaptic trace decay")
    p.add_argument("--lsm-threshold", type=float, default=0.5)
    p.add_argument("--lsm-substeps", type=int, default=4)
    p.add_argument("--lsm-input", default="current", choices=["current", "rate"])

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--amp", type=int, default=1)
    return p.parse_args()


def evaluate(model, seq: GPUSequences, threshold: float, amp: bool) -> dict[str, float]:
    model.eval()
    counts = Counts()
    loss_sum, batches = 0.0, 0
    with torch.no_grad():
        for x, y in seq.epoch(shuffle=False):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
            logits = logits.float()
            loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(logits, y))
            batches += 1
            counts.update(logits, y, threshold)
            # x[:, -1, 1] is the last mask the model was shown.
            counts.update_change(logits, y, x[:, -1, 1], threshold)
    out = counts.result()
    out["loss"] = loss_sum / max(batches, 1)
    return out


def evaluate_rollout(model, split, window: int, steps: int, mean: float, std: float,
                     threshold: float, amp: bool, limit: int = 40) -> dict[str, float]:
    """Free running: the model is fed its own prediction and has to keep going.

    One step ahead is nearly solved by copying the input, so it cannot tell a
    model that learned the evolution from one that learned to stand still. Under
    a rollout, standing still stops being free.
    """
    model.eval()
    counts = Counts()
    images = split.images[:limit]
    masks = split.masks[:limit]
    if images.numel() == 0:
        return {k: float("nan") for k in ("iou", "f1", "precision", "recall", "boundary_f1", "accuracy")}
    with torch.no_grad():
        frames = masks[:, :window].float()
        for step in range(steps):
            b, w = frames.shape[0], frames.shape[1]
            image = images.unsqueeze(1).expand(b, w, 64, 64)
            x = torch.stack([(image - mean) / std, frames], dim=2)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
            logits = logits.float()
            target = masks[:, window + step].float()
            counts.update(logits, target, threshold)
            counts.update_change(logits, target, masks[:, window - 1].float(), threshold)
            pred = (torch.sigmoid(logits) > threshold).float()
            frames = torch.cat([frames[:, 1:], pred.unsqueeze(1)], dim=1)
    return counts.result()


def main() -> None:
    a = parse()
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(a.amp) and device.type == "cuda"
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    cache = build_cache(a.dataset, Path(a.data_root), Path(a.cache_dir), a.iterations)
    splits = to_gpu_splits(cache, device, seed=42)
    mean = float(splits["train"].images.mean())
    std = float(splits["train"].images.std().clamp_min(1e-6))

    seqs = {
        name: GPUSequences(split, window=a.window, horizon=a.horizon, batch_size=a.batch_size,
                           augment=bool(a.augment) and name == "train",
                           mean=mean, std=std, seed=a.seed, drop_last=(name == "train"))
        for name, split in splits.items()
    }

    kwargs: dict = dict(hidden=a.hidden, fc=a.fc, dropout=a.dropout, head=a.head)
    if a.arch == "esn":
        kwargs["esn"] = dict(spectral_radius=a.spectral_radius, leak=a.leak, sparsity=a.sparsity,
                             input_scaling=a.input_scaling, seed=a.seed,
                             sparse=not bool(a.dense_reservoir))
    if a.arch == "lsm":
        kwargs["lsm"] = dict(spectral_radius=a.spectral_radius, sparsity=a.sparsity,
                             input_scaling=a.input_scaling, beta=a.lsm_beta, alpha=a.lsm_alpha,
                             threshold=a.lsm_threshold, substeps=a.lsm_substeps,
                             input_mode=a.lsm_input, seed=a.seed,
                             sparse=not bool(a.dense_reservoir))
    if a.arch in ("noise", "copy"):
        kwargs = {}
    if a.arch == "3dcnn":
        kwargs = dict(fc=a.fc, dropout=a.dropout, head=a.head)

    model = build(a.arch, **kwargs).to(device)
    if a.freeze_encoder and hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if a.optimizer == "sgd":
        opt = torch.optim.SGD(trainable, lr=a.lr, momentum=a.momentum,
                              weight_decay=a.weight_decay, fused=device.type == "cuda")
    else:
        opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.weight_decay,
                                fused=device.type == "cuda")
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(a.epochs // 5, 1), gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    by_gradient = a.arch not in ("noise", "copy") and not (a.arch in ("esn", "lsm") and a.readout == "ridge")
    best, best_epoch, epochs_ran, since_best = {"iou": -1.0}, -1, 0, 0
    history = []
    fit_seconds = 0.0

    if by_gradient:
        t0 = time.time()
        for epoch in range(a.epochs):
            epochs_ran = epoch + 1
            model.train()
            for x, y in seqs["train"].epoch(shuffle=True, limit=a.train_batches):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(x)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                scaler.step(opt)
                scaler.update()
            sched.step()
            val = evaluate(model, seqs["val"], a.threshold, amp)
            history.append({"epoch": epoch, **{k: round(v, 6) for k, v in val.items()}})
            if val["iou"] > best["iou"]:
                best, best_epoch, since_best = val, epoch, 0
                torch.save(model.state_dict(), out_dir / "best.pt")
            else:
                since_best += 1
                if since_best >= a.patience:
                    break
        fit_seconds = time.time() - t0
        if (out_dir / "best.pt").exists():
            model.load_state_dict(torch.load(out_dir / "best.pt", weights_only=True, map_location=device))
        eval_model = model
    elif a.arch in ("esn", "lsm") and a.readout == "ridge":
        # No backpropagation anywhere: fixed encoder, fixed reservoir, one solve.
        t0 = time.time()
        weights = ridge_readout_gpu(model, seqs["train"], ridge=a.ridge, amp=amp,
                                    limit=a.train_batches)
        fit_seconds = time.time() - t0
        eval_model = RidgeReadoutModel(model, weights).to(device)
        val = evaluate(eval_model, seqs["val"], a.threshold, amp)
        best, best_epoch, epochs_ran = val, 0, 1
        history.append({"epoch": 0, **{k: round(v, 6) for k, v in val.items()}})
        torch.save({"weights": weights.cpu()}, out_dir / "readout.pt")
    else:
        eval_model = model

    if hasattr(model, "cell") and hasattr(model.cell, "reset_telemetry"):
        model.cell.reset_telemetry()
    test = evaluate(eval_model, seqs["test"], a.threshold, amp)
    rate = firing_rate(model)

    # The same model scored at several horizons, so one run says how quickly a
    # method degrades as the question gets harder.
    horizons = {}
    for k in [int(v) for v in a.eval_horizons.split(",") if v]:
        seq_k = GPUSequences(splits["test"], window=a.window, horizon=k, batch_size=a.batch_size,
                             augment=False, mean=mean, std=std, seed=a.seed)
        r = evaluate(eval_model, seq_k, a.threshold, amp)
        horizons[f"h{k}_iou"] = round(r["iou"], 6)
        horizons[f"h{k}_change_iou"] = round(r["change_iou"], 6)
    roll = evaluate_rollout(eval_model, splits["test"], a.window, a.rollout, mean, std,
                            a.threshold, amp)

    metrics = {
        "iou": round(test["iou"], 6),
        "f1": round(test["f1"], 6),
        "precision": round(test["precision"], 6),
        "recall": round(test["recall"], 6),
        "boundary_f1": round(test["boundary_f1"], 6),
        "accuracy": round(test["accuracy"], 6),
        "test_loss": round(test["loss"], 6),
        "rollout_iou": round(roll["iou"], 6),
        "rollout_f1": round(roll["f1"], 6),
        "rollout_boundary_f1": round(roll["boundary_f1"], 6),
        "rollout_change_iou": round(roll["change_iou"], 6),
        "change_iou": round(test["change_iou"], 6),
        "change_f1": round(test["change_f1"], 6),
        "horizon": a.horizon,
        **horizons,
        "val_iou": round(best["iou"], 6),
        "arch": a.arch, "dataset": a.dataset, "seed": a.seed, "window": a.window,
        "head": a.head, "readout": a.readout if a.arch in ("esn", "lsm") else "sgd",
        "epochs_ran": epochs_ran, "best_epoch": best_epoch,
        "trainable_params": trainable_parameters(model),
        "fit_seconds": round(fit_seconds, 2),
        "total_seconds": round(time.time() - started, 1),
        "firing_rate": round(rate, 6),
        "train_examples": seqs["train"].size, "test_examples": seqs["test"].size,
        "optimizer": a.optimizer, "lr": a.lr, "hidden": a.hidden,
        "spectral_radius": a.spectral_radius if a.arch in ("esn", "lsm") else None,
        "leak": a.leak if a.arch == "esn" else None,
        "sparsity": a.sparsity if a.arch in ("esn", "lsm") else None,
        "input_scaling": a.input_scaling if a.arch in ("esn", "lsm") else None,
        "lsm_beta": a.lsm_beta if a.arch == "lsm" else None,
        "lsm_alpha": a.lsm_alpha if a.arch == "lsm" else None,
        "lsm_substeps": a.lsm_substeps if a.arch == "lsm" else None,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "commit": _git_commit(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "args.json").write_text(json.dumps(vars(a), indent=2))
    print(json.dumps({k: metrics[k] for k in
                      ("arch", "head", "window", "horizon", "readout", "seed", "iou",
                       "change_iou", "rollout_iou", "rollout_change_iou", "fit_seconds")}))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[1],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
