"""Architectures: the thesis's five, the controls it was missing, and the fixes.

The thesis (Mashinini 2022, section 3.2.5) put a convolutional encoder in front
of a recurrent cell (ESN, RNN, LSTM, GRU) or a 3D CNN, pooled to a vector, and
reconstructed a 64x64 mask from a fully connected layer. Every arm here can run
in that configuration (`head="fc"`), which is what makes the replication a
replication.

What is added, and why:

* `CopyPrevious`  the control the thesis lacked. Predicting "the mask does not
  change" scores about 0.99 IoU on this task, above every model the thesis
  reported. Without this baseline in the table, a model at 0.64 IoU looks like
  a result rather than a failure.
* `head="spatial"`  a decoder that keeps the spatial layout and takes a skip
  connection from the encoder, instead of squeezing 64x64 through a global
  vector. This tests whether the thesis measured echo state networks or measured
  its own bottleneck.
* `EchoStateCell` on CUDA with a sparse reservoir, and an optional closed form
  ridge readout (`fit_readout`) whose sufficient statistics are accumulated
  across ranks. The thesis trained the readout with SGD, which is not how an
  echo state network is normally trained and gives up its main advantage.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spiking import LIFReservoir, SpikeEncoder


class ConvEncoder(nn.Module):
    """Four 2D convolutional blocks (thesis section 3.2.5).

    Returns both the pooled vector the thesis used and the last feature map,
    so the spatial head can use what the fully connected head throws away.
    """

    def __init__(self, in_channels: int = 2, dropout: float = 0.1, width: float = 1.0):
        super().__init__()
        c1, c2, c3 = int(32 * width), int(64 * width), int(128 * width)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 7, stride=2, padding=3), nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(c1, c2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
        )                                        # 64 -> 32
        self.block2 = nn.Sequential(
            nn.MaxPool2d(2),                     # 32 -> 16
            nn.Conv2d(c2, c3, 7, stride=2, padding=3), nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(c3, c3, 1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )                                        # 16 -> 8
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.channels = c3
        self.skip_channels = c2
        self.out_features = c3 * 2 * 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.block1(x)
        f = self.block2(s)
        return self.pool(f).flatten(1), f, s


class EchoStateCell(nn.Module):
    """A leaky integrator reservoir: fixed weights, trained readout elsewhere.

        h_t = (1 - a) h_{t-1} + a tanh(W_in u_t + W h_{t-1})

    The recurrent matrix is stored sparse (CSR) and multiplied on the GPU, which
    is what makes a 4096 unit reservoir at 80 percent sparsity cheap. Nothing
    here is a Parameter: the reservoir never learns, which is the point.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        spectral_radius: float = 0.9,
        leak: float = 0.0713,
        sparsity: float = 0.8,
        input_scaling: float = 1.0,
        seed: int = 0,
        sparse: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.leak = leak
        self.sparse = sparse
        g = torch.Generator().manual_seed(seed)

        w = torch.empty(hidden_size, hidden_size).uniform_(-1.0, 1.0, generator=g)
        keep = torch.rand(hidden_size, hidden_size, generator=g) >= sparsity
        w = w * keep
        # Power iteration for the spectral radius: an eigendecomposition of a
        # 4096 square matrix would dominate setup time for no extra accuracy.
        v = torch.randn(hidden_size, 1, generator=g)
        for _ in range(60):
            v = w @ v
            n = v.norm().clamp_min(1e-9)
            v = v / n
        radius = (w @ v).norm() / v.norm().clamp_min(1e-9)
        w = w * (spectral_radius / radius.clamp_min(1e-9))

        w_in = torch.empty(hidden_size, input_size).uniform_(-1.0, 1.0, generator=g) * input_scaling

        if sparse:
            self.register_buffer("w_sparse", w.to_sparse_csr(), persistent=False)
            self.register_buffer("w", torch.zeros(1), persistent=False)
        else:
            self.register_buffer("w", w)
        self.register_buffer("w_in", w_in)

    def _recurrent(self, h: torch.Tensor) -> torch.Tensor:
        if self.sparse:
            # torch.sparse CSR expects (out, in) @ (in, batch), and cuSPARSE has
            # no half precision SpMM on this architecture, so the reservoir step
            # is forced to float32 inside an autocast region.
            with torch.autocast("cuda", enabled=False):
                out = torch.sparse.mm(self.w_sparse, h.float().t()).t()
            return out.to(h.dtype)
        return F.linear(h, self.w)

    def forward(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        pre = F.linear(u, self.w_in) + self._recurrent(h)
        return (1 - self.leak) * h + self.leak * torch.tanh(pre)


class SpatialHead(nn.Module):
    """Decode the recurrent state back to 64x64, keeping the layout.

    The state is projected to an 8x8 grid, concatenated with the encoder's own
    feature map for the most recent frame (the skip), and upsampled. The thesis
    instead flattened to a vector and predicted 4096 independent logits.
    """

    def __init__(self, hidden: int, feat_channels: int, skip_channels: int, base: int = 64, dropout: float = 0.1):
        super().__init__()
        self.project = nn.Linear(hidden, base * 8 * 8)
        self.base = base
        self.up1 = nn.Sequential(
            nn.Conv2d(base + feat_channels, base, 3, padding=1), nn.BatchNorm2d(base),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )                                                   # 8
        self.up2 = nn.Sequential(
            nn.Conv2d(base + skip_channels, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
        )                                                   # 32 after two upsamples
        self.out = nn.Conv2d(base + 1, 1, 3, padding=1)

    def forward(self, h: torch.Tensor, feat: torch.Tensor, skip: torch.Tensor, last_mask: torch.Tensor) -> torch.Tensor:
        b = h.shape[0]
        x = self.project(h).reshape(b, self.base, 8, 8)
        x = self.up1(torch.cat([x, feat], dim=1))
        x = F.interpolate(x, scale_factor=4, mode="nearest")          # 8 -> 32
        x = self.up2(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")          # 32 -> 64
        # The previous mask as an explicit input to the final layer: the task is
        # to predict its change, so the model should not have to rebuild it.
        return self.out(torch.cat([x, last_mask.unsqueeze(1)], dim=1)).squeeze(1)


class ConvRecurrent(nn.Module):
    """Encoder -> recurrent cell -> head. One class, four cells, two heads."""

    def __init__(
        self,
        cell: str = "gru",
        hidden: int = 512,
        fc: int = 1024,
        dropout: float = 0.1,
        width: float = 1.0,
        esn: dict | None = None,
        lsm: dict | None = None,
        head: str = "fc",
        out_size: int = 64,
    ):
        super().__init__()
        self.kind = cell
        self.head_kind = head
        self.encoder = ConvEncoder(2, dropout, width)
        self.hidden = hidden
        f = self.encoder.out_features

        if cell == "esn":
            self.cell = EchoStateCell(f, hidden, **(esn or {}))
        elif cell == "lsm":
            opts = dict(lsm or {})
            opts.pop("substeps", None)  # consumed by this module, not by the liquid
            self.encode_spikes = SpikeEncoder(opts.pop("input_mode", "current"), opts.pop("gain", 1.0))
            self.cell = LIFReservoir(f, hidden, **opts)
        elif cell == "rnn":
            self.cell = nn.RNNCell(f, hidden)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(f, hidden)
        elif cell == "gru":
            self.cell = nn.GRUCell(f, hidden)
        else:
            raise ValueError(f"unknown cell {cell}")

        if head == "fc":
            self.head = nn.Sequential(
                nn.Linear(hidden, fc), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(fc, out_size * out_size),
            )
        else:
            self.head = SpatialHead(hidden, self.encoder.channels, self.encoder.skip_channels, dropout=dropout)
        self.out_size = out_size
        self.lsm_substeps = int((lsm or {}).get("substeps", 4)) if cell == "lsm" else 1

    def states(self, x: torch.Tensor):
        """Run the sequence. Returns the final state and the last frame's maps."""
        b, t = x.shape[:2]
        pooled, feat, skip = self.encoder(x.reshape(b * t, *x.shape[2:]))
        pooled = pooled.reshape(b, t, -1)
        h = pooled.new_zeros(b, self.hidden)
        c = pooled.new_zeros(b, self.hidden)
        state = None
        for step in range(t):
            u = pooled[:, step]
            if self.kind == "lstm":
                h, c = self.cell(u, (h, c))
            elif self.kind == "lsm":
                # The liquid runs on its own clock: several membrane steps per
                # frame, so a single frame can still produce recurrent dynamics.
                drive = self.encode_spikes(u)
                for _ in range(self.lsm_substeps):
                    state = self.cell(drive, state)
                h = state[2]  # the filtered spike trace is the liquid state
            else:
                h = self.cell(u, h)
        last = slice(t - 1, None, t)  # the maps belonging to the final frame
        feat = feat.reshape(b, t, *feat.shape[1:])[:, -1]
        skip = skip.reshape(b, t, *skip.shape[1:])[:, -1]
        return h, feat, skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, feat, skip = self.states(x)
        if self.head_kind == "fc":
            return self.head(h).reshape(-1, self.out_size, self.out_size)
        return self.head(h, feat, skip, x[:, -1, 1])


class Conv3DNet(nn.Module):
    """Two 3D convolutional blocks over time (thesis section 3.2.5)."""

    def __init__(self, fc: int = 1024, dropout: float = 0.1, head: str = "fc", out_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2, 32, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.Dropout3d(dropout),
            nn.Conv3d(32, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.Dropout3d(dropout),
        )
        self.head_kind = head
        self.out_size = out_size
        if head == "fc":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 2, 2)), nn.Flatten(), nn.Linear(64 * 4, fc),
                nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(fc, out_size * out_size),
            )
        else:
            self.decode = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )
            self.out = nn.Conv2d(64 + 1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.net(x.permute(0, 2, 1, 3, 4))
        if self.head_kind == "fc":
            return self.head(v).reshape(-1, self.out_size, self.out_size)
        z = v.mean(dim=2)                                    # collapse time
        z = F.interpolate(self.decode(z), size=(self.out_size, self.out_size), mode="nearest")
        return self.out(torch.cat([z, x[:, -1, 1:2]], dim=1)).squeeze(1)


class WhiteNoise(nn.Module):
    """The thesis's control: predictions drawn at random."""

    def __init__(self, out_size: int = 64, **_):
        super().__init__()
        self.out_size = out_size
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.shape[0], self.out_size, self.out_size, device=x.device) + self.dummy


class CopyPrevious(nn.Module):
    """The control the thesis was missing: predict that nothing changes.

    The level set moves about half a percent of pixels per iteration, so this
    scores near 0.99 IoU. Any model below it has not learned the task, whatever
    its number looks like in isolation.
    """

    def __init__(self, out_size: int = 64, logit: float = 8.0, **_):
        super().__init__()
        self.out_size = out_size
        self.logit = logit
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, -1, 1] * 2.0 - 1.0) * self.logit + self.dummy


def build(arch: str, **kwargs) -> nn.Module:
    if arch == "3dcnn":
        return Conv3DNet(fc=kwargs.get("fc", 1024), dropout=kwargs.get("dropout", 0.1),
                         head=kwargs.get("head", "fc"))
    if arch == "noise":
        return WhiteNoise()
    if arch == "copy":
        return CopyPrevious()
    return ConvRecurrent(cell=arch, **kwargs)


def firing_rate(model: nn.Module) -> float:
    """Mean spikes per neuron per step, zero for anything that does not spike."""
    cell = getattr(model, "cell", None)
    return cell.mean_firing_rate() if isinstance(cell, LIFReservoir) else 0.0


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def ridge_readout_gpu(model, seq, ridge: float = 1e-2, amp: bool = True, limit: int = 0) -> torch.Tensor:
    """Solve the readout in closed form: W = (H'H + lambda I)^-1 H'Y.

    This is how a reservoir is normally trained, and it is what makes one cheap:
    no backpropagation through time, one pass over the data, one linear solve.
    The thesis trained its readout with stochastic gradient descent instead,
    which spends the cost of a trained recurrent network and keeps the accuracy
    of an untrained one.

    Only sufficient statistics are accumulated (H'H is hidden x hidden, H'Y is
    hidden x 4096), so memory does not grow with the dataset, and the same
    accumulate then solve structure is what would be all-reduced across ranks
    on a multi GPU machine.
    """
    model.eval()
    device = next(model.parameters()).device
    n = model.hidden + 1
    targets = model.out_size * model.out_size
    xtx = torch.zeros(n, n, device=device, dtype=torch.float64)
    xty = torch.zeros(n, targets, device=device, dtype=torch.float64)

    for step, (x, y) in enumerate(seq.epoch(shuffle=False)):
        if limit and step >= limit:
            break
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            h, _, _ = model.states(x)
        h = h.float()
        h = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device)], dim=1).double()
        xtx += h.t() @ h
        xty += h.t() @ y.reshape(y.shape[0], -1).double()

    xtx += ridge * torch.eye(n, device=device, dtype=torch.float64)
    return torch.linalg.solve(xtx, xty).float()


class RidgeReadoutModel(nn.Module):
    """A trained reservoir body with a closed form linear readout on top."""

    def __init__(self, body: ConvRecurrent, weights: torch.Tensor):
        super().__init__()
        self.body = body
        self.register_buffer("weights", weights)
        self.out_size = body.out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _, _ = self.body.states(x)
        h = torch.cat([h.float(), torch.ones(h.shape[0], 1, device=h.device)], dim=1)
        probs = (h @ self.weights).reshape(-1, self.out_size, self.out_size)
        # The solve fits probabilities; the trainer speaks logits.
        return torch.log(probs.clamp(1e-4, 1 - 1e-4) / (1 - probs.clamp(1e-4, 1 - 1e-4)))
