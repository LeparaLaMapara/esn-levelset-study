"""Figures for the paper and the demo: predictions side by side, and the curves.

    python scripts/figures.py --runs T:/levelset-runs --out figures

Produces
  qualitative_<dataset>.png   input, Chan-Vese target, and each model's
                              prediction on the same test images, one row per
                              image, one column per model, with the change
                              region outlined so a reader can see what the
                              metric is scoring.
  rollout_<dataset>.png       accuracy against rollout step, which is where
                              copying stops working.
  horizons_<dataset>.png      accuracy against prediction horizon.
  sweep_esn.png               the reservoir hyper-parameter grid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from levelset.data import build_cache
from levelset.gpu_data import GPUSequences, to_gpu_splits
from levelset.models import RidgeReadoutModel, build

PANEL = ["copy", "gru", "lstm", "3dcnn", "esn", "lsm"]
LABEL = {"copy": "Copy", "gru": "CGRU", "lstm": "CLSTM", "rnn": "CRNN",
         "3dcnn": "3DCNN", "esn": "CESN", "lsm": "CLSM"}
# Paper palette: one colour per family, colourblind safe.
COLOUR = {"copy": "#777777", "noise": "#bbbbbb", "gru": "#0072B2", "lstm": "#56B4E9",
          "rnn": "#009E73", "3dcnn": "#E69F00", "esn": "#D55E00", "lsm": "#CC79A7"}


def load_model(run_dir: Path, device):
    args = json.loads((run_dir / "args.json").read_text())
    kwargs = dict(hidden=args["hidden"], fc=args["fc"], dropout=args["dropout"], head=args["head"])
    arch = args["arch"]
    if arch == "esn":
        kwargs["esn"] = dict(spectral_radius=args["spectral_radius"], leak=args["leak"],
                             sparsity=args["sparsity"], input_scaling=args["input_scaling"],
                             seed=args["seed"], sparse=not args["dense_reservoir"])
    if arch == "lsm":
        kwargs["lsm"] = dict(spectral_radius=args["spectral_radius"], sparsity=args["sparsity"],
                             input_scaling=args["input_scaling"], beta=args["lsm_beta"],
                             alpha=args["lsm_alpha"], threshold=args["lsm_threshold"],
                             substeps=args["lsm_substeps"], input_mode=args["lsm_input"],
                             seed=args["seed"], sparse=not args["dense_reservoir"])
    if arch in ("noise", "copy"):
        kwargs = {}
    if arch == "3dcnn":
        kwargs = dict(fc=args["fc"], dropout=args["dropout"], head=args["head"])
    model = build(arch, **kwargs).to(device)
    if (run_dir / "best.pt").exists():
        model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True, map_location=device))
    elif (run_dir / "readout.pt").exists():
        w = torch.load(run_dir / "readout.pt", weights_only=True)["weights"].to(device)
        model = RidgeReadoutModel(model, w).to(device)
    return model.eval(), args


def qualitative(runs: Path, out: Path, dataset: str, block: str, device, rows: int = 4):
    available = {}
    for arch in PANEL:
        d = runs / f"{block}_{dataset}_{arch}_s1"
        if (d / "metrics.json").exists():
            available[arch] = d
    if not available:
        return
    any_args = json.loads((next(iter(available.values())) / "args.json").read_text())
    window, horizon = any_args["window"], any_args["horizon"]

    cache = build_cache(dataset, Path("T:/datasets/levelset"), Path("T:/datasets/levelset/cache"), 100)
    splits = to_gpu_splits(cache, device, seed=42)
    split = splits["test"]
    mean = float(splits["train"].images.mean())
    std = float(splits["train"].images.std().clamp_min(1e-6))

    # A fixed, interesting set: the test images whose level set moves most.
    masks = split.masks.float()
    movement = (masks[:, window + horizon - 1] != masks[:, window - 1]).float().mean((1, 2))
    pick = torch.argsort(movement, descending=True)[:rows]

    images = split.images[pick]
    frames = masks[pick][:, :window]
    target = masks[pick][:, window + horizon - 1]
    prev = masks[pick][:, window - 1]

    x = torch.stack([(images.unsqueeze(1).expand_as(frames) - mean) / std, frames], dim=2)

    preds = {}
    for arch, d in available.items():
        model, _ = load_model(d, device)
        with torch.no_grad():
            preds[arch] = (torch.sigmoid(model(x).float()) > 0.5).float().cpu()

    n_cols = 3 + len(preds)
    fig, axes = plt.subplots(rows, n_cols, figsize=(1.7 * n_cols, 1.8 * rows))
    for r in range(rows):
        panels = [(images[r].cpu(), "Image", "gray"),
                  (prev[r].cpu(), f"Mask t", "gray"),
                  (target[r].cpu(), f"Chan-Vese t+{horizon}", "gray")]
        for arch in preds:
            panels.append((preds[arch][r], LABEL.get(arch, arch), "gray"))
        for c, (img, title, cmap) in enumerate(panels):
            ax = axes[r, c] if rows > 1 else axes[c]
            ax.imshow(img.numpy(), cmap=cmap, vmin=0, vmax=1)
            # Outline where the truth changed: that is what change IoU scores.
            change = (target[r] != prev[r]).cpu().numpy()
            if c >= 2:
                ax.contour(change, levels=[0.5], colors="#D55E00", linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
    fig.suptitle(f"{dataset.upper()}: predicting the level set {horizon} step(s) ahead "
                 f"(orange outlines the pixels that actually change)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"qualitative_{dataset}_{block}.png", dpi=170)
    plt.close(fig)


def rollout_curve(runs: Path, out: Path, dataset: str, block: str, device, steps: int = 25):
    cache = build_cache(dataset, Path("T:/datasets/levelset"), Path("T:/datasets/levelset/cache"), 100)
    splits = to_gpu_splits(cache, device, seed=42)
    split = splits["test"]
    mean = float(splits["train"].images.mean())
    std = float(splits["train"].images.std().clamp_min(1e-6))

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for arch in PANEL:
        d = runs / f"{block}_{dataset}_{arch}_s1"
        if not (d / "metrics.json").exists():
            continue
        model, args = load_model(d, device)
        window = args["window"]
        images = split.images[:40]
        masks = split.masks[:40].float()
        frames = masks[:, :window]
        ys = []
        with torch.no_grad():
            for step in range(steps):
                b, w = frames.shape[0], frames.shape[1]
                x = torch.stack([(images.unsqueeze(1).expand(b, w, 64, 64) - mean) / std, frames], dim=2)
                logits = model(x).float()
                pred = (torch.sigmoid(logits) > 0.5).float()
                truth = masks[:, window + step]
                inter = (pred * truth).sum((1, 2))
                union = ((pred + truth) > 0).float().sum((1, 2)).clamp_min(1)
                ys.append(float((inter / union).mean()))
                frames = torch.cat([frames[:, 1:], pred.unsqueeze(1)], dim=1)
        ax.plot(range(1, steps + 1), ys, label=LABEL.get(arch, arch), color=COLOUR.get(arch))
    ax.set_xlabel("free running step"); ax.set_ylabel("IoU against Chan-Vese")
    ax.set_title(f"{dataset.upper()}: error compounding under rollout", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / f"rollout_{dataset}_{block}.png", dpi=170); plt.close(fig)


def horizon_curve(runs: Path, out: Path, dataset: str, block: str):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    drew = False
    for arch in PANEL:
        d = runs / f"{block}_{dataset}_{arch}_s1"
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        hs = sorted(int(k[1:].split("_")[0]) for k in m if k.startswith("h") and k.endswith("_change_iou"))
        ys = [m[f"h{h}_change_iou"] for h in hs]
        if hs:
            ax.plot(hs, ys, marker="o", label=LABEL.get(arch, arch), color=COLOUR.get(arch))
            drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("prediction horizon (Chan-Vese iterations)")
    ax.set_ylabel("change region IoU")
    ax.set_title(f"{dataset.upper()}: accuracy on the pixels that move", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / f"horizons_{dataset}_{block}.png", dpi=170); plt.close(fig)


def sweep_heatmap(runs: Path, out: Path):
    rows = []
    for f in runs.glob("E_wsd_esn_*/metrics.json"):
        rows.append(json.loads(f.read_text()))
    if not rows:
        return
    radii = sorted({r["spectral_radius"] for r in rows})
    leaks = sorted({r["leak"] for r in rows})
    grid = np.full((len(leaks), len(radii)), np.nan)
    for r in rows:
        grid[leaks.index(r["leak"]), radii.index(r["spectral_radius"])] = r["change_iou"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(radii)), [str(v) for v in radii])
    ax.set_yticks(range(len(leaks)), [str(v) for v in leaks])
    ax.set_xlabel("spectral radius"); ax.set_ylabel("leak rate")
    ax.set_title("CESN: change region IoU over the reservoir grid", fontsize=10)
    for i in range(len(leaks)):
        for j in range(len(radii)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if grid[i, j] < np.nanmax(grid) * 0.7 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig(out / "sweep_esn.png", dpi=170); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="T:/levelset-runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--blocks", default="A,C")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for block in a.blocks.split(","):
        for ds in ["wsd", "bsd"]:
            qualitative(runs, out, ds, block, device)
            rollout_curve(runs, out, ds, block, device)
            horizon_curve(runs, out, ds, block)
    sweep_heatmap(runs, out)
    print("figures in", out.resolve())


if __name__ == "__main__":
    main()
