# Reservoirs were not the problem

A replication and correction of Mashinini (2022), *Learning Level Set Method by
Echo State Network for Image Segmentation* (MSc, University of the
Witwatersrand), plus the arms it was missing.

**Live demo:** https://levelset-reservoirs.onrender.com — solves the Chan-Vese
level set on request and runs the trained models beside it.

## The short version

The dissertation asked whether an echo state network can learn the evolution of
a variational level set as well as a trained recurrent network, found that it
could not, and blamed the reservoir's leaking rate and spectral radius. Three
things about that comparison turn out to explain the result better.

| | finding |
|---|---|
| A no-op wins | Predicting that the mask does not change scores **0.993 IoU**, above every number the dissertation reported. |
| The recipe cannot train | Under its stated optimiser (SGD, lr 0.1, weight decay 0.1) **every architecture collapses to a constant 0.5 output**, so the reported differences between cells are differences between degenerate models. |
| The head, not the cell | Reconstructing a 64x64 mask from a globally pooled vector is the bottleneck. With a spatial decoder the same cells go from 0.42 to 0.96 IoU. |

Once the task is made hard enough to discriminate (ten iterations ahead, scored
only on pixels that move), the ordering does not reverse, it **disappears**:

| model | IoU | change IoU | fit time |
|---|---|---|---|
| Copy previous *(control)* | 0.925 | 0.000 | 0 s |
| CGRU | 0.964 | 0.523 | 53 s |
| CLSTM | 0.963 | 0.531 | 41 s |
| CRNN | 0.963 | 0.549 | 33 s |
| **CESN** *(untrained recurrence)* | 0.962 | 0.515 | 40 s |
| **CLSM** *(spiking, untrained)* | 0.962 | 0.511 | 50 s |
| CESN, closed form ridge readout | 0.646 | 0.305 | **1 s** |

WSD, three seeds. An untrained random reservoir matches networks trained end to
end. Fitted the way reservoirs are meant to be fitted (one linear solve, no
backpropagation) it is more than an order of magnitude cheaper and clearly
worse, which is the honest form of the reservoir argument.

## What is new here, beyond the replication

* **A liquid state machine**, which the dissertation reviews and never
  implements: leaky integrate and fire neurons, sparse fixed recurrence, 20%
  inhibitory, state read as a filtered spike train. Measured firing rate is
  reported, because a spiking model's argument is cost.
* **Change region metrics**: overlap computed only where the level set actually
  moves. Global IoU on a slow PDE is dominated by static background.
* **Horizons and rollouts**: predicting 1, 5, 10 and 25 iterations ahead, and
  free running for 25 steps so errors compound.
* **A closed form ridge readout** whose sufficient statistics are accumulated
  in one pass (and would all-reduce unchanged across ranks).
* **Controls the original lacked**: copy previous, alongside its white noise.
* **Active phase sampling**: the front is static for tens of iterations, then
  collapses; uniform sampling trains mostly on stillness.

## Running it

```bash
pip install -r requirements.txt            # torch with CUDA, numpy, pillow, matplotlib
python scripts/run_experiment.py --arch esn --dataset wsd --seed 1 \
    --head spatial --window 4 --horizon 10 --optimizer adamw --lr 1e-3 \
    --out RUNS/demo                        # one trial -> metrics.json
python scripts/sweep.py --block all        # the whole matrix, idempotent
python scripts/analyze.py                  # tables
python scripts/figures.py                  # figures
python scripts/make_paper_numbers.py       # papers/numbers.tex, papers/tables.tex
```

Everything runs on the GPU: the Chan-Vese sequences are generated there, the
database stays resident in VRAM, batching and augmentation are CUDA kernels,
and the reservoirs use sparse CSR matrix multiplication. A full WSD trial takes
under a minute on an RTX 2070 SUPER.

Datasets download themselves into `T:/datasets/levelset` (override with
`--data-root`): Weizmann Segmentation Database (one and two object), BSDS500,
and CIFAR-10/100 if you want them.

## Layout

```
levelset/chanvese.py   the Chan-Vese generator, thesis parameters, semi-implicit
levelset/gpu_data.py   GPU resident sequences, windows, horizons, augmentation
levelset/models.py     encoder, cells (RNN/LSTM/GRU/ESN), heads, controls, ridge
levelset/spiking.py    the liquid state machine
levelset/metrics.py    IoU, F1, boundary F1, change region variants
scripts/               one trial, the sweep, analysis, figures, paper numbers
demo/                  the deployed service
papers/main.tex        the write-up; every number comes from numbers.tex
```

## Honesty notes

* The generator is validated against the Weizmann human segmentations: the
  converged level set reaches about 0.87 IoU against them.
* Numbers in the paper are generated from the run directories. None is typed.
* One consumer GPU, 64x64 inputs, two of the original's four databases. The
  CIFAR scale experiments are not reproduced.
* The reservoir hyper-parameter sweep is one seed per cell and is reported as
  inconclusive, not as a tuning law.
