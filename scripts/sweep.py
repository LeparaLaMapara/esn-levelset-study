"""The experiment matrix. Idempotent, resumable, one process, one GPU.

Each trial is `run_experiment.py` writing its own metrics.json, so a trial that
already has one is skipped and a crash costs only the trial that was running.

    python scripts/sweep.py --block all           # everything
    python scripts/sweep.py --block A --dry-run   # what would run

Blocks, each of which answers one question:

  A  replication      the thesis exactly: fully connected head, window 1,
                      one step ahead. Does the published ordering reappear?
  B  head             the same task with a spatial decoder. Was the ordering a
                      property of the recurrent cells or of the readout?
  C  task             window 4, ten steps ahead, spatial head. What happens when
                      the question stops being solvable by copying?
  D  readout          reservoirs with a closed form ridge readout and no
                      backpropagation anywhere. How much accuracy does the
                      cheap training actually cost?
  E  reservoir sweep  the hyper-parameters the thesis blamed for the echo state
                      network's failure, swept directly.
  F  optimiser       the thesis head with a modern optimiser. Block A collapses
                      to a constant output under the published recipe (SGD, lr
                      0.1, weight decay 0.1), so F separates "the recipe cannot
                      train this" from "the head cannot express this".
  G  main comparison the study's own setting: spatial head, window 4, ten steps
                      ahead, AdamW. This is the table the conclusions rest on.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "T:/venvs/ubunye-research-engine/Scripts/python.exe"
RUNS = Path("T:/levelset-runs")

SEEDS = [1, 2, 3]
DATASETS = ["wsd", "bsd"]


def trials(block: str) -> list[dict]:
    out: list[dict] = []

    if block in ("A", "all"):
        for ds, seed, arch in itertools.product(DATASETS, SEEDS,
                                                ["esn", "rnn", "lstm", "gru", "3dcnn", "noise", "copy"]):
            out.append(dict(name=f"A_{ds}_{arch}_s{seed}", arch=arch, dataset=ds, seed=seed,
                            head="fc", window=1, horizon=1, epochs=6))

    if block in ("B", "all"):
        for ds, seed, arch in itertools.product(DATASETS, SEEDS,
                                                ["esn", "lsm", "rnn", "lstm", "gru", "3dcnn"]):
            out.append(dict(name=f"B_{ds}_{arch}_s{seed}", arch=arch, dataset=ds, seed=seed,
                            head="spatial", window=1, horizon=1, epochs=6))

    if block in ("C", "all"):
        for ds, seed, arch in itertools.product(DATASETS, SEEDS,
                                                ["esn", "lsm", "rnn", "lstm", "gru", "3dcnn", "copy"]):
            out.append(dict(name=f"C_{ds}_{arch}_s{seed}", arch=arch, dataset=ds, seed=seed,
                            head="spatial", window=4, horizon=10, epochs=6))

    if block in ("D", "all"):
        for ds, seed, arch in itertools.product(DATASETS, SEEDS, ["esn", "lsm"]):
            out.append(dict(name=f"D_{ds}_{arch}_ridge_s{seed}", arch=arch, dataset=ds, seed=seed,
                            head="fc", window=4, horizon=10, readout="ridge", freeze_encoder=1,
                            epochs=1))

    if block in ("F", "all"):
        for ds, seed, arch in itertools.product(DATASETS, SEEDS, ["esn", "lsm", "gru", "lstm", "3dcnn"]):
            out.append(dict(name=f"F_{ds}_{arch}_s{seed}", arch=arch, dataset=ds, seed=seed,
                            head="fc", window=1, horizon=1, epochs=8,
                            optimizer="adamw", lr=1e-3, weight_decay=0.01))

    if block in ("G", "all"):
        for ds, seed, arch in itertools.product(DATASETS, SEEDS,
                                                ["esn", "lsm", "rnn", "lstm", "gru", "3dcnn", "copy", "noise"]):
            out.append(dict(name=f"G_{ds}_{arch}_s{seed}", arch=arch, dataset=ds, seed=seed,
                            head="spatial", window=4, horizon=10, epochs=10,
                            optimizer="adamw", lr=1e-3, weight_decay=0.01))
        # The cheap training regime, at the same task, for the cost comparison.
        for ds, seed, arch in itertools.product(DATASETS, SEEDS, ["esn", "lsm"]):
            out.append(dict(name=f"G_{ds}_{arch}_ridge_s{seed}", arch=arch, dataset=ds, seed=seed,
                            head="fc", window=4, horizon=10, readout="ridge", freeze_encoder=1,
                            epochs=1))

    if block in ("E", "all"):
        for rho, leak in itertools.product([0.5, 0.9, 1.1, 1.5], [0.0078125, 0.0713, 0.5, 1.0]):
            out.append(dict(name=f"E_wsd_esn_r{rho}_l{leak}", arch="esn", dataset="wsd", seed=1,
                            head="spatial", window=4, horizon=10, epochs=4,
                            spectral_radius=rho, leak=leak))
        for alpha, thr in itertools.product([0.5, 0.8, 0.95], [0.3, 0.5, 0.8]):
            out.append(dict(name=f"E_wsd_lsm_a{alpha}_t{thr}", arch="lsm", dataset="wsd", seed=1,
                            head="spatial", window=4, horizon=10, epochs=4,
                            lsm_alpha=alpha, lsm_threshold=thr))
    return out


FLAGS = {
    "arch": "--arch", "dataset": "--dataset", "seed": "--seed", "head": "--head",
    "window": "--window", "horizon": "--horizon", "epochs": "--epochs",
    "readout": "--readout", "freeze_encoder": "--freeze-encoder",
    "spectral_radius": "--spectral-radius", "leak": "--leak",
    "lsm_alpha": "--lsm-alpha", "lsm_threshold": "--lsm-threshold",
    "optimizer": "--optimizer", "lr": "--lr", "weight_decay": "--weight-decay",
}


def command(trial: dict) -> list[str]:
    cmd = [PY, str(ROOT / "scripts" / "run_experiment.py"), "--out", str(RUNS / trial["name"])]
    for key, flag in FLAGS.items():
        if key in trial:
            cmd += [flag, str(trial[key])]
    cmd += ["--patience", "2"]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="all", choices=["A", "B", "C", "D", "E", "F", "G", "all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--redo", action="store_true", help="ignore existing metrics.json")
    a = ap.parse_args()

    todo = trials(a.block)
    print(f"{len(todo)} trials in block {a.block}")
    started = time.time()
    done, skipped, failed = 0, 0, []

    for i, trial in enumerate(todo, 1):
        out = RUNS / trial["name"]
        if (out / "metrics.json").exists() and not a.redo:
            skipped += 1
            continue
        cmd = command(trial)
        if a.dry_run:
            print(" ".join(cmd))
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            failed.append(trial["name"])
            print(f"[{i}/{len(todo)}] FAILED {trial['name']}: {proc.stderr.strip().splitlines()[-1][:160]}")
        else:
            done += 1
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
            print(f"[{i}/{len(todo)}] {trial['name']} ({time.time() - t0:.0f}s) {line}")
        sys.stdout.flush()

    print(f"\ndone {done}, skipped {skipped}, failed {len(failed)} in {(time.time() - started) / 60:.1f} min")
    if failed:
        print("failed:", ", ".join(failed))
        (RUNS / "failed.json").write_text(json.dumps(failed, indent=2))


if __name__ == "__main__":
    main()
