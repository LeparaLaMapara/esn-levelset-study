"""Generate the paper's numbers from the runs, so none of them can be typed.

Writes papers/numbers.tex, a list of \\newcommand definitions like

    \\newcommand{\\GwsdesnchangeIoU}{0.365}

and papers/tables.tex, the result tables themselves. main.tex uses those macros
and inputs those tables, so a figure in the text is always the measured one and
a missing run breaks the build instead of quietly shortening an average.

    python scripts/make_paper_numbers.py
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARCH_ORDER = ["copy", "noise", "gru", "lstm", "rnn", "3dcnn", "esn", "lsm"]
ARCH_TEX = {
    "copy": "Copy previous \\emph{(control)}", "noise": "White noise \\emph{(control)}",
    "gru": "CGRU", "lstm": "CLSTM", "rnn": "CRNN", "3dcnn": "3DCNN",
    "esn": "CESN \\emph{(reservoir)}", "lsm": "CLSM \\emph{(spiking reservoir)}",
}
BLOCK_CAPTION = {
    "A": ("blockA", "The published configuration reproduced: fully connected head, window one, "
                    "one step ahead, SGD at learning rate 0.1 with weight decay 0.1. Every model "
                    "collapses to a constant output, and none approaches the copy control."),
    "F": ("blockF", "The same head and task with AdamW instead. The recipe, not the head alone, "
                    "explains the collapse in Table~\\ref{tab:blockA}."),
    "G": ("blockG", "The corrected setting: spatial decoder head, window four, ten iterations "
                    "ahead, AdamW. Change IoU scores only the pixels whose label differs between "
                    "the last input mask and the target."),
}
METRIC_KEYS = [("iou", "IoU"), ("change_iou", "change IoU"), ("rollout_iou", "rollout IoU"),
               ("rollout_change_iou", "rollout change IoU"), ("boundary_f1", "boundary F1"),
               ("fit_seconds", "fit (s)")]


def tex_key(*parts: str) -> str:
    """A LaTeX safe macro name: letters only."""
    raw = "".join(str(p) for p in parts)
    out = []
    for ch in raw:
        if ch.isalpha():
            out.append(ch)
        elif ch.isdigit():
            out.append("ZOTTFFSSEN"[int(ch)])
    return "".join(out)


def load(runs: Path) -> list[dict]:
    rows = []
    for f in sorted(runs.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        m["run"] = f.parent.name
        m["block"] = f.parent.name.split("_")[0]
        rows.append(m)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="T:/levelset-runs")
    a = ap.parse_args()
    rows = load(Path(a.runs))
    papers = ROOT / "papers"
    papers.mkdir(exist_ok=True)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["block"], r.get("dataset"), r["arch"], r.get("readout", "sgd"))
        grouped[key].append(r)

    macros = []
    for (block, ds, arch, readout), rs in sorted(grouped.items()):
        for metric, _ in METRIC_KEYS:
            vals = [r[metric] for r in rs if r.get(metric) is not None]
            if not vals:
                continue
            mean = stats.mean(vals)
            sd = stats.stdev(vals) if len(vals) > 1 else 0.0
            suffix = "" if readout == "sgd" else "ridge"
            name = tex_key(block, ds, arch, suffix, metric)
            macros.append(f"\\newcommand{{\\{name}}}{{{mean:.3f}}}")
            macros.append(f"\\newcommand{{\\{name}sd}}{{{sd:.3f}}}")
        macros.append(f"\\newcommand{{\\{tex_key(block, ds, arch, '' if readout=='sgd' else 'ridge', 'seeds')}}}{{{len(rs)}}}")

    macros.append(f"\\newcommand{{\\totalruns}}{{{len(rows)}}}")

    # Derived statistics the argument leans on, computed here rather than by
    # hand: how much the reservoir sweep moves the result, against how much the
    # random seed moves it for one fixed configuration.
    def runs_matching(prefix: str, metric: str = "change_iou") -> list[float]:
        return [r[metric] for r in rows
                if r["run"].startswith(prefix) and r.get(metric) is not None]

    seed_rep = [r["change_iou"] for r in rows
                if r["run"].startswith("G_wsd_esn_s") and r.get("readout", "sgd") == "sgd"]
    for name, vals in [("esnsweep", runs_matching("E_wsd_esn")),
                       ("lsmsweep", runs_matching("E_wsd_lsm")),
                       ("seedrep", seed_rep)]:
        if len(vals) > 1:
            macros.append(f"\\newcommand{{\\{name}min}}{{{min(vals):.3f}}}")
            macros.append(f"\\newcommand{{\\{name}max}}{{{max(vals):.3f}}}")
            macros.append(f"\\newcommand{{\\{name}sd}}{{{stats.stdev(vals):.3f}}}")
            macros.append(f"\\newcommand{{\\{name}cells}}{{{len(vals)}}}")

    firing = runs_matching("E_wsd_lsm", "firing_rate")
    if firing:
        macros.append(f"\\newcommand{{\\lsmfiringmin}}{{{min(firing):.2f}}}")
        macros.append(f"\\newcommand{{\\lsmfiringmax}}{{{max(firing):.2f}}}")
    (papers / "numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    tables = []
    for block in ["A", "F", "G"]:
        label, caption = BLOCK_CAPTION[block]
        for ds in ["wsd", "bsd"]:
            rs = [r for r in rows if r["block"] == block and r.get("dataset") == ds]
            if not rs:
                continue
            by_arch = defaultdict(list)
            for r in rs:
                tag = r["arch"] + ("" if r.get("readout", "sgd") == "sgd" else " (ridge)")
                by_arch[tag].append(r)
            tables.append("\\begin{table}[t]\n\\centering\\small")
            tables.append("\\begin{tabular}{lrrrrrr}\n\\toprule")
            tables.append("model & " + " & ".join(n for _, n in METRIC_KEYS) + " \\\\\n\\midrule")
            for arch in ARCH_ORDER + [f"{a} (ridge)" for a in ("esn", "lsm")]:
                rs2 = by_arch.get(arch)
                if not rs2:
                    continue
                cells = []
                for metric, _ in METRIC_KEYS:
                    vals = [r[metric] for r in rs2 if r.get(metric) is not None]
                    if not vals:
                        cells.append("--")
                    elif metric == "fit_seconds":
                        cells.append(f"{stats.mean(vals):.0f}")
                    else:
                        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
                        cells.append(f"{stats.mean(vals):.3f} {{\\scriptsize$\\pm$ {sd:.3f}}}")
                name = ARCH_TEX.get(arch.replace(" (ridge)", ""), arch)
                if "ridge" in arch:
                    name += ", ridge readout"
                tables.append(f"{name} & " + " & ".join(cells) + " \\\\")
            tables.append("\\bottomrule\n\\end{tabular}")
            tables.append(f"\\caption{{{ds.upper()}. {caption}}}")
            tables.append(f"\\label{{tab:{label}{'' if ds == 'wsd' else 'bsd'}}}")
            tables.append("\\end{table}\n")

    (papers / "tables.tex").write_text("\n".join(tables) + "\n", encoding="utf-8")
    print(f"wrote papers/numbers.tex ({len(macros)} macros) and papers/tables.tex from {len(rows)} runs")


if __name__ == "__main__":
    main()
