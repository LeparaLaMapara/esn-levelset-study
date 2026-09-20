"""Aggregate every run into the tables the paper reports.

Reads each metrics.json under the runs directory, groups by block, dataset and
architecture, and reports mean and standard deviation across seeds. Nothing is
typed by hand: the paper's numbers are generated from here, and a missing run is
a visible gap rather than a silently shorter average.

    python scripts/analyze.py                      # markdown tables to stdout
    python scripts/analyze.py --json results.json  # machine readable too
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

RUNS = Path("T:/levelset-runs")

BLOCK_TITLE = {
    "A": "Block A. Replication: fully connected head, window 1, one step ahead",
    "B": "Block B. Spatial head, otherwise the same task",
    "C": "Block C. Window 4, ten steps ahead, spatial head",
    "D": "Block D. Closed form ridge readout, no backpropagation",
    "E": "Block E. Reservoir hyper-parameter sweep (AdamW: under the thesis recipe the encoder collapses and the liquid never spikes)",
    "F": "Block F. The thesis head with AdamW: does the recipe or the head explain block A?",
    "G": "Block G. Main comparison: spatial head, window 4, ten steps ahead, AdamW",
    "H": "Block H. Block G restricted to the active phase of the evolution",
}

METRICS = ["iou", "change_iou", "rollout_iou", "rollout_change_iou", "boundary_f1",
           "firing_rate", "fit_seconds"]

ARCH_ORDER = ["copy", "noise", "gru", "lstm", "rnn", "3dcnn", "esn", "lsm"]
ARCH_LABEL = {
    "copy": "Copy previous (control)", "noise": "White noise (control)",
    "gru": "CGRU", "lstm": "CLSTM", "rnn": "CRNN", "3dcnn": "3DCNN",
    "esn": "CESN (echo state)", "lsm": "CLSM (liquid state)",
}


def load(runs: Path) -> list[dict]:
    out = []
    for f in sorted(runs.glob("*/metrics.json")):
        try:
            m = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        m["run"] = f.parent.name
        m["block"] = f.parent.name.split("_")[0]
        out.append(m)
    return out


def agg(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return stats.mean(values), (stats.stdev(values) if len(values) > 1 else 0.0)


def table(rows: list[dict], dataset: str) -> list[str]:
    # Group by architecture AND readout: a reservoir fitted by ridge regression
    # is a different method from the same reservoir trained by gradient descent,
    # and averaging them together hides both.
    by_arch = defaultdict(list)
    for r in rows:
        if r.get("dataset") == dataset:
            tag = r["arch"] + ("" if r.get("readout", "sgd") == "sgd" else " + ridge")
            by_arch[tag].append(r)
    if not by_arch:
        return []
    lines = [
        f"\n**{dataset.upper()}**\n",
        "| model | IoU | change IoU | rollout IoU | rollout change IoU | boundary F1 | firing | fit s | seeds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arch in ARCH_ORDER + [f"{a} + ridge" for a in ("esn", "lsm")]:
        rs = by_arch.get(arch)
        if not rs:
            continue
        cells = []
        for m in METRICS:
            mean, sd = agg([r[m] for r in rs if r.get(m) is not None])
            if m == "fit_seconds":
                cells.append(f"{mean:.0f}")
            elif m == "firing_rate":
                cells.append("--" if mean == 0 else f"{mean:.2f}")
            else:
                cells.append(f"{mean:.3f} ± {sd:.3f}")
        label = ARCH_LABEL.get(arch.replace(" + ridge", ""), arch)
        if "ridge" in arch:
            label += ", ridge readout"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {len(rs)} |")
    return lines


def sweep_table(rows: list[dict]) -> list[str]:
    esn = [r for r in rows if r["arch"] == "esn"]
    lsm = [r for r in rows if r["arch"] == "lsm"]
    lines = []
    if esn:
        lines += ["\n**Echo state reservoir: spectral radius and leak rate**\n",
                  "| spectral radius | leak | IoU | change IoU | rollout change IoU |", "|---|---|---|---|---|"]
        for r in sorted(esn, key=lambda r: (r["spectral_radius"], r["leak"])):
            lines.append(f"| {r['spectral_radius']} | {r['leak']} | {r['iou']:.3f} | "
                         f"{r['change_iou']:.3f} | {r['rollout_change_iou']:.3f} |")
    if lsm:
        lines += ["\n**Liquid reservoir: trace decay and firing threshold**\n",
                  "| trace decay | threshold | IoU | change IoU | firing rate |", "|---|---|---|---|---|"]
        for r in sorted(lsm, key=lambda r: (r["lsm_alpha"], r.get("lsm_threshold") or 0)):
            lines.append(f"| {r['lsm_alpha']} | {r.get('lsm_threshold', '')} | {r['iou']:.3f} | "
                         f"{r['change_iou']:.3f} | {r.get('firing_rate', 0):.3f} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    rows = load(Path(a.runs))
    print(f"# Results\n\n{len(rows)} runs read from {a.runs}.")

    for block in ["A", "F", "B", "C", "G", "H", "D"]:
        rs = [r for r in rows if r["block"] == block]
        if not rs:
            continue
        print(f"\n## {BLOCK_TITLE[block]}")
        for ds in ["wsd", "bsd"]:
            print("\n".join(table(rs, ds)))

    sweep = [r for r in rows if r["block"] == "E"]
    if sweep:
        print(f"\n## {BLOCK_TITLE['E']}")
        print("\n".join(sweep_table(sweep)))

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
