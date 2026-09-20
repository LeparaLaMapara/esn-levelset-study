"""GPU resident data. No CPU dataloader, no worker processes, no host copies.

The whole of a database's VLS sequences fits in VRAM (WSD is 82 MB of masks,
BSD 205 MB), so the tensors are uploaded once and every batch is an index into
them. Shuffling, batching, the two channel assembly, augmentation and
standardisation are all CUDA kernels. On this machine that removed the input
pipeline as a bottleneck entirely: the earlier CPU DataLoader path spent more
time collating 64x64 crops than the GPU spent training on them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GPUSplit:
    images: torch.Tensor   # (N, 64, 64) float32 on device
    masks: torch.Tensor    # (N, T, 64, 64) uint8 on device
    truth: torch.Tensor    # (N, 64, 64) uint8 on device


class GPUSequences:
    """Batches of (window of [I, M] frames) -> M_{t+1}, assembled on the GPU.

    window = 1 reproduces the thesis's n = 1. Every sample is (image i, frame t),
    and the sampler draws from the flattened (i, t) index on the device.
    """

    def __init__(
        self,
        split: GPUSplit,
        window: int = 1,
        horizon: int = 1,
        batch_size: int = 32,
        augment: bool = False,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.split = split
        self.window = window
        self.horizon = horizon
        self.batch_size = batch_size
        self.augment = augment
        self.mean, self.std = mean, std
        self.drop_last = drop_last
        self.device = split.images.device
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        n, t = split.masks.shape[0], split.masks.shape[1]
        # A sample ends its input window at t-1 and is scored at t+horizon-1.
        last = t - horizon + 1
        idx_i = torch.arange(n, device=self.device).repeat_interleave(last - window)
        idx_t = torch.arange(window, last, device=self.device).repeat(n)
        self.index_i, self.index_t = idx_i, idx_t
        self.size = idx_i.numel()

    def __len__(self) -> int:
        full = self.size // self.batch_size
        return full if self.drop_last else full + int(self.size % self.batch_size > 0)

    def _gather(self, i: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, w = i.numel(), self.window
        # (B, W) frame indices ending at t-1, and the target at t.
        offsets = torch.arange(-w, 0, device=self.device).view(1, w)
        frames = self.split.masks[i.view(b, 1), (t.view(b, 1) + offsets)].float()   # (B, W, 64, 64)
        target = self.split.masks[i, t + self.horizon - 1].float()
        image = self.split.images[i].unsqueeze(1).expand(b, w, 64, 64)
        x = torch.stack([image, frames], dim=2)                                     # (B, W, 2, 64, 64)
        return x, target

    def _augment(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        flip_h = torch.rand(b, device=self.device, generator=self.gen) < 0.5
        flip_v = torch.rand(b, device=self.device, generator=self.gen) < 0.5
        x = torch.where(flip_h.view(b, 1, 1, 1, 1), x.flip(-1), x)
        y = torch.where(flip_h.view(b, 1, 1), y.flip(-1), y)
        x = torch.where(flip_v.view(b, 1, 1, 1, 1), x.flip(-2), x)
        y = torch.where(flip_v.view(b, 1, 1), y.flip(-2), y)
        # Rotations in multiples of 90 degrees, done as one grouped operation per k.
        k = torch.randint(0, 4, (1,), device=self.device, generator=self.gen).item()
        if k:
            x, y = torch.rot90(x, k, dims=(-2, -1)), torch.rot90(y, k, dims=(-2, -1))
        # Thesis section 3.6.3: Gaussian noise, variance 0.01, image channel only.
        noise = torch.randn(x[:, :, 0].shape, device=self.device, generator=self.gen) * 0.1
        x[:, :, 0] = x[:, :, 0] + noise
        return x, y

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        x[:, :, 0] = (x[:, :, 0] - self.mean) / self.std
        return x

    def epoch(self, shuffle: bool = True, limit: int = 0):
        order = (torch.randperm(self.size, device=self.device, generator=self.gen)
                 if shuffle else torch.arange(self.size, device=self.device))
        batches = len(self)
        for b in range(batches):
            if limit and b >= limit:
                break
            sel = order[b * self.batch_size : (b + 1) * self.batch_size]
            if self.drop_last and sel.numel() < self.batch_size:
                break
            x, y = self._gather(self.index_i[sel], self.index_t[sel])
            if self.augment:
                x, y = self._augment(x, y)
            # The last input mask travels with the batch: every change aware
            # metric needs to know what the model was given, not just the truth.
            yield self.standardise(x), y


def to_gpu_splits(cache_file, device: torch.device, seed: int = 42) -> dict[str, GPUSplit]:
    """Load the cache and put every split on the device, once."""
    blob = torch.load(cache_file, weights_only=True)
    n = len(blob["images"])
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train, n_val = int(0.7 * n), int(0.1 * n)
    bounds = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    return {
        name: GPUSplit(
            blob["images"][idx].to(device, non_blocking=True).float(),
            blob["masks"][idx].to(device, non_blocking=True),
            blob["truth"][idx].to(device, non_blocking=True),
        )
        for name, idx in bounds.items()
    }
