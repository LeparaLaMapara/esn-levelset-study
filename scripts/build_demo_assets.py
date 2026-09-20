"""Bundle what the deployed demo needs: sample sequences, checkpoints, results.

    python scripts/build_demo_assets.py --block G --dataset wsd

Writes into demo/:
  assets/samples.npz    a handful of test images and their Chan-Vese sequences,
                        chosen as the ones whose level set actually moves
  assets/results.json   the headline table, generated from the runs
  models/*.pt           the trained checkpoints the demo loads
  models/manifest.json  how to rebuild each model and what to call it

Only test split images are bundled, and only models that have a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from levelset.data import build_cache          # noqa: E402
from levelset.gpu_data import to_gpu_splits    # noqa: E402

LABEL = {"gru": "CGRU (trained)", "lstm": "CLSTM (trained)", "rnn": "CRNN (trained)",
         "3dcnn": "3DCNN (trained)", "esn": "CESN (reservoir, untrained recurrence)",
         "lsm": "CLSM (spiking reservoir, untrained)", "copy": "Copy previous (control)"}
NOTE = {"esn": "recurrent weights fixed at random", "lsm": "spiking, recurrent weights fixed",
        "copy": "predicts no change at all"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="T:/levelset-runs")
    ap.add_argument("--block", default="G")
    ap.add_argument("--dataset", default="wsd")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--archs", default="copy,gru,lstm,3dcnn,esn,lsm")
    a = ap.parse_args()

    runs = Path(a.runs)
    demo = ROOT / "demo"
    (demo / "assets").mkdir(parents=True, exist_ok=True)
    (demo / "models").mkdir(parents=True, exist_ok=True)

    cache = build_cache(a.dataset, Path("T:/datasets/levelset"), Path("T:/datasets/levelset/cache"), 100)
    device = torch.device("cpu")
    splits = to_gpu_splits(cache, device, seed=42)
    test, train = splits["test"], splits["train"]

    masks = test.masks.float()
    movement = (masks[:, -1] != masks[:, 0]).float().mean((1, 2))
    pick = torch.argsort(movement, descending=True)[: a.samples]
    np.savez_compressed(
        demo / "assets" / "samples.npz",
        images=test.images[pick].numpy().astype("float32"),
        masks=test.masks[pick].numpy().astype("uint8"),
    )

    manifest: dict = {"models": {}, "window": None, "horizon": None,
                      "mean": float(train.images.mean()), "std": float(train.images.std())}
    for arch in a.archs.split(","):
        d = runs / f"{a.block}_{a.dataset}_{arch}_s1"
        args_f, ckpt = d / "args.json", d / "best.pt"
        if not args_f.exists():
            continue
        args = json.loads(args_f.read_text())
        manifest["window"] = args["window"]
        manifest["horizon"] = args["horizon"]
        kwargs: dict = dict(hidden=args["hidden"], fc=args["fc"], dropout=args["dropout"], head=args["head"])
        if arch == "esn":
            kwargs["esn"] = dict(spectral_radius=args["spectral_radius"], leak=args["leak"],
                                 sparsity=args["sparsity"], input_scaling=args["input_scaling"],
                                 seed=args["seed"], sparse=False)  # dense on CPU: no cuSPARSE there
        if arch == "lsm":
            kwargs["lsm"] = dict(spectral_radius=args["spectral_radius"], sparsity=args["sparsity"],
                                 input_scaling=args["input_scaling"], beta=args["lsm_beta"],
                                 alpha=args["lsm_alpha"], threshold=args["lsm_threshold"],
                                 substeps=args["lsm_substeps"], input_mode=args["lsm_input"],
                                 seed=args["seed"], sparse=False)
        if arch in ("copy", "noise"):
            kwargs = {}
        if arch == "3dcnn":
            kwargs = dict(fc=args["fc"], dropout=args["dropout"], head=args["head"])

        if ckpt.exists():
            shutil.copy(ckpt, demo / "models" / f"{arch}.pt")
        elif arch == "copy":
            torch.save(torch.nn.Module().state_dict(), demo / "models" / "copy.pt")
        else:
            continue
        manifest["models"][arch] = {
            "arch": arch, "file": f"{arch}.pt", "kwargs": kwargs,
            "label": LABEL.get(arch, arch), "note": NOTE.get(arch, ""),
        }

    (demo / "models" / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # The headline table, measured.
    rows, cols = [], ["model", "IoU", "change IoU", "rollout IoU", "fit seconds"]
    for arch, spec in manifest["models"].items():
        f = runs / f"{a.block}_{a.dataset}_{arch}_s1" / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        rows.append([spec["label"], f"{m['iou']:.3f}", f"{m['change_iou']:.3f}",
                     f"{m['rollout_iou']:.3f}", f"{m['fit_seconds']:.0f}"])
    (demo / "assets" / "results.json").write_text(json.dumps({
        "caption": (f"{a.dataset.upper()} test split, predicting ten Chan-Vese iterations ahead, "
                    f"seed 1. Change IoU scores only the pixels that move."),
        "columns": cols, "table": rows,
    }, indent=2))
    print(f"bundled {len(manifest['models'])} models and {len(pick)} samples")


if __name__ == "__main__":
    main()
