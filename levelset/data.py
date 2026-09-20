"""Datasets: the thesis's four databases, and the VLS sequences built from them.

Pipeline, once per database (cached to disk):

    images -> grayscale, 64x64  ->  Chan-Vese evolution (100 iterations)
           -> masks M_1..M_100  ->  cache file with images, masks, ground truth

A training example follows section 3.2.4: the input is a two channel image
[I, M_t] and the target is M_{t+1}. `window` generalises this to the last
`window` masks, which is the one knob the replication holds at 1 (the thesis
sets n = 1) and the improvement arm raises.

Splits are by IMAGE, not by example. The thesis split examples at random, which
puts 100 frames of the same image on both sides of the split; splitting by image
is the harder and more honest test, and is recorded as a deviation in README.md.
"""
from __future__ import annotations

import io
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .chanvese import evolve

SIZE = 64  # thesis section 3.2.4: inputs and targets are resized to 64 x 64


def _to_gray_64(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _binary_64(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32)
    return (arr > 127).astype(np.uint8)


def load_wsd(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Weizmann Segmentation Database, one object and two object sets (200 images).

    Ground truth is the first human segmentation for each image; the database
    ships three per image.
    """
    images, truths = [], []
    for name in ("wsd_1obj.zip", "wsd_2obj.zip"):
        path = root / name
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as z:
            members = z.namelist()
            sources = sorted(m for m in members if "/src_bw/" in m and m.endswith(".png"))
            for src in sources:
                stem = src.rsplit("/", 1)[-1][:-4]
                segs = sorted(m for m in members if f"/{stem}/human_seg/" in m and m.endswith(".png"))
                if not segs:
                    continue
                images.append(_to_gray_64(z.read(src)))
                truths.append(_binary_64(z.read(segs[0])))
    if not images:
        raise FileNotFoundError(f"no Weizmann zips under {root}")
    return np.stack(images), np.stack(truths)


def load_bsd(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """BSDS500 images (500). No binary object truth ships with it.

    Its segmentations are .mat boundary maps, so the ground truth channel is
    left empty and BSD is scored against the VLS target only, which is what the
    learning task itself is.
    """
    path = root / "bsds500.tgz"
    images = []
    with tarfile.open(path) as t:
        members = [m for m in t.getmembers() if m.name.endswith(".jpg") and "/images/" in m.name]
        for m in sorted(members, key=lambda m: m.name):
            f = t.extractfile(m)
            if f is not None:
                images.append(_to_gray_64(f.read()))
    if not images:
        raise FileNotFoundError(f"no BSDS images in {path}")
    arr = np.stack(images)
    return arr, np.zeros((len(arr), SIZE, SIZE), dtype=np.uint8)


def load_cifar(root: Path, which: str = "cifar10", limit: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """CIFAR-10 or CIFAR-100 images, grayscaled and upsampled to 64 x 64.

    The thesis generated 5 to 6 million VLS examples here and capped them at
    200,000. `limit` images x 100 frames is this study's cap; it is a deviation,
    recorded in README.md.
    """
    import pickle

    path = root / ("cifar-10-python.tar.gz" if which == "cifar10" else "cifar-100-python.tar.gz")
    arrays = []
    with tarfile.open(path) as t:
        names = [m.name for m in t.getmembers()]
        batches = [n for n in names if "data_batch" in n or n.endswith("/train")]
        for name in sorted(batches):
            f = t.extractfile(name)
            if f is None:
                continue
            d = pickle.load(io.BytesIO(f.read()), encoding="bytes")
            arrays.append(d[b"data"])
            if sum(a.shape[0] for a in arrays) >= limit:
                break
    raw = np.concatenate(arrays)[:limit]
    rgb = raw.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    gray = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2])
    t_gray = torch.from_numpy(gray).unsqueeze(1)
    up = torch.nn.functional.interpolate(t_gray, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    arr = up.squeeze(1).numpy()
    return arr, np.zeros((len(arr), SIZE, SIZE), dtype=np.uint8)


LOADERS = {
    "wsd": lambda root: load_wsd(root),
    "bsd": lambda root: load_bsd(root),
    "cifar10": lambda root: load_cifar(root, "cifar10"),
    "cifar100": lambda root: load_cifar(root, "cifar100"),
}


def build_cache(dataset: str, root: Path, cache_dir: Path, iterations: int = 100, device: str = "cuda") -> Path:
    """Evolve the VLS once per database and store the result."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{dataset}_{iterations}.pt"
    if out.exists():
        return out
    images, truths = LOADERS[dataset](root)
    dev = device if torch.cuda.is_available() else "cpu"
    batch = torch.from_numpy(images).to(dev)
    masks = []
    for i in range(0, len(batch), 64):  # chunked so VRAM holds regardless of size
        masks.append(evolve(batch[i : i + 64], iterations=iterations).cpu())
    torch.save(
        {
            "images": torch.from_numpy(images),
            "masks": torch.cat(masks),
            "truth": torch.from_numpy(truths),
            "iterations": iterations,
        },
        out,
    )
    return out


@dataclass
class Split:
    images: torch.Tensor  # (N, 64, 64) float
    masks: torch.Tensor   # (N, T, 64, 64) uint8
    truth: torch.Tensor   # (N, 64, 64) uint8, zeros where the database has none


def load_splits(cache_file: Path, seed: int = 42) -> dict[str, Split]:
    """70 / 10 / 20 by image, as in the thesis, but grouped so no image leaks."""
    blob = torch.load(cache_file, weights_only=True)
    n = len(blob["images"])
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train, n_val = int(0.7 * n), int(0.1 * n)
    bounds = {"train": order[:n_train], "val": order[n_train : n_train + n_val], "test": order[n_train + n_val :]}
    return {
        name: Split(blob["images"][idx], blob["masks"][idx], blob["truth"][idx])
        for name, idx in bounds.items()
    }


class VLSSequences(torch.utils.data.Dataset):
    """(window of [I, M] frames) -> M_{t+1}.

    window = 1 is the thesis's setting: the model sees one previous frame. The
    recurrent layer then runs over a sequence of length one, which is the
    condition under which the thesis found the echo state network could not
    compete. Raising `window` is the study's first hypothesis.
    """

    def __init__(self, split: Split, window: int = 1, augment: bool = False, seed: int = 0):
        self.split = split
        self.window = window
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        t = split.masks.shape[1]
        # A sample is (image index, t) where t is the frame being predicted.
        self.index = [(i, t0) for i in range(len(split.images)) for t0 in range(window, t)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        i, t0 = self.index[k]
        image = self.split.images[i]
        frames = self.split.masks[i, t0 - self.window : t0].float()
        target = self.split.masks[i, t0].float()

        x = torch.stack([image.expand_as(frames), frames], dim=1)  # (window, 2, 64, 64)

        if self.augment:
            if self.rng.random() < 0.5:
                x, target = torch.flip(x, dims=[-1]), torch.flip(target, dims=[-1])
            if self.rng.random() < 0.5:
                x, target = torch.flip(x, dims=[-2]), torch.flip(target, dims=[-2])
            k90 = int(self.rng.integers(0, 4))
            if k90:
                x, target = torch.rot90(x, k90, dims=[-2, -1]), torch.rot90(target, k90, dims=[-2, -1])
            # Thesis section 3.6.3: Gaussian noise with variance 0.01 on the image channel.
            x[:, 0] = x[:, 0] + torch.randn_like(x[:, 0]) * 0.1

        return x, target


def standardise(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Normalise the image channel only; the mask channel is already binary."""
    x = x.clone()
    x[:, :, 0] = (x[:, :, 0] - mean) / std
    return x


def channel_stats(split: Split) -> tuple[float, float]:
    img = split.images.float()
    return float(img.mean()), float(img.std().clamp_min(1e-6))
