"""A Liquid State Machine: the spiking reservoir the thesis reviewed but never built.

Mashinini (2022) section 2.2.4 describes liquid state machines as the other
branch of reservoir computing and then compares only the echo state network
against trained recurrent networks. This implements the missing arm.

The liquid is a population of leaky integrate and fire neurons with fixed,
sparse, random recurrent weights, 80 percent excitatory and 20 percent
inhibitory (the usual cortical ratio used in the Maass formulation). It is
never trained. What the readout sees is not the spikes themselves but an
exponentially filtered spike trace, which is the standard "liquid state".

Everything runs as dense or sparse CUDA ops in float16 or float32, and the
spike count per input is recorded, because a spiking model's argument is cost:
if it needs as much compute as a GRU it has no advantage to trade.

Differences from the echo state network, which is the comparison the study is
for:
  ESN: continuous state, tanh, leaky integration, state IS the representation.
  LSM: binary events, threshold and reset, representation is a filtered trace,
       so the memory is in the synaptic filter rather than in the activation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LIFReservoir(nn.Module):
    """Fixed random spiking reservoir with an exponentially filtered readout state.

        I_t   = W_in u_t + W s_{t-1}
        v_t   = beta v_{t-1} (1 - s_{t-1}) + (1 - beta) I_t
        s_t   = 1[v_t > theta]
        x_t   = alpha x_{t-1} + s_t          (the liquid state the readout reads)

    `beta` is the membrane decay, `alpha` the synaptic trace decay, `theta` the
    firing threshold. Reset is by subtraction of the membrane on spike, which
    keeps the dynamics bounded without a hard clamp.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        spectral_radius: float = 0.9,
        sparsity: float = 0.8,
        input_scaling: float = 1.0,
        beta: float = 0.9,
        alpha: float = 0.8,
        threshold: float = 0.5,
        inhibitory_fraction: float = 0.2,
        seed: int = 0,
        sparse: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.beta, self.alpha, self.threshold = beta, alpha, threshold
        self.sparse = sparse
        g = torch.Generator().manual_seed(seed)

        w = torch.rand(hidden_size, hidden_size, generator=g)          # non negative weights
        keep = torch.rand(hidden_size, hidden_size, generator=g) >= sparsity
        w = w * keep
        # Dale's law, loosely: a fixed fraction of neurons is inhibitory, and a
        # neuron's sign applies to every synapse it makes.
        n_inh = int(inhibitory_fraction * hidden_size)
        sign = torch.ones(hidden_size)
        sign[torch.randperm(hidden_size, generator=g)[:n_inh]] = -1.0
        w = w * sign.view(1, -1)

        v = torch.randn(hidden_size, 1, generator=g)
        for _ in range(60):
            v = w @ v
            v = v / v.norm().clamp_min(1e-9)
        radius = (w @ v).norm() / v.norm().clamp_min(1e-9)
        w = w * (spectral_radius / radius.clamp_min(1e-9))

        w_in = torch.empty(hidden_size, input_size).uniform_(-1.0, 1.0, generator=g) * input_scaling

        if sparse:
            self.register_buffer("w_sparse", w.to_sparse_csr(), persistent=False)
        else:
            self.register_buffer("w", w)
        self.register_buffer("w_in", w_in)
        self.spike_count = 0.0
        self.step_count = 0

    def _recurrent(self, s: torch.Tensor) -> torch.Tensor:
        if self.sparse:
            return torch.sparse.mm(self.w_sparse, s.t()).t()
        return F.linear(s, self.w)

    def forward(self, u: torch.Tensor, state: tuple | None = None):
        """One step. state = (v, s, x); returns the new state."""
        b = u.shape[0]
        if state is None:
            z = u.new_zeros(b, self.hidden_size)
            v, s, x = z, z.clone(), z.clone()
        else:
            v, s, x = state
        current = F.linear(u, self.w_in) + self._recurrent(s)
        v = self.beta * v * (1.0 - s) + (1.0 - self.beta) * current
        s = (v > self.threshold).to(v.dtype)
        x = self.alpha * x + s
        # Cheap telemetry: mean spikes per neuron per step, for the cost table.
        self.spike_count += float(s.detach().float().mean())
        self.step_count += 1
        return v, s, x

    def mean_firing_rate(self) -> float:
        return self.spike_count / max(self.step_count, 1)

    def reset_telemetry(self) -> None:
        self.spike_count, self.step_count = 0.0, 0


class SpikeEncoder(nn.Module):
    """Turn continuous convolutional features into input current.

    Two options, both standard: pass the features through as analogue current
    (the usual choice when a liquid is driven by real valued sensors), or
    Bernoulli rate code them, which makes the input genuinely spiking. Rate
    coding is the more faithful liquid state machine and the noisier one, so
    the mode is a flag and the paper reports both.
    """

    def __init__(self, mode: str = "current", gain: float = 1.0):
        super().__init__()
        self.mode, self.gain = mode, gain

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self.mode == "current":
            return u * self.gain
        p = torch.sigmoid(u * self.gain)
        return torch.bernoulli(p)
