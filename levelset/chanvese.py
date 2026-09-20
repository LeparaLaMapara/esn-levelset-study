"""Chan-Vese variational level set (VLS) evolution, the data generator.

This reproduces the sequence generator of Mashinini (2022), "Learning Level Set
Method by Echo State Network for Image Segmentation" (MSc, Wits), section 3.2.3:
a grayscale image and a checkerboard initialisation are evolved by the Chan-Vese
piecewise constant model for 100 iterations, and every iterate is kept, so one
image yields 100 masks and a training example is (I, M_t) -> M_{t+1}.

Parameters are the thesis's: lambda1 = lambda2 = 1, mu = 0.2, dt = 0.5, a 3x3
stencil for the spatial derivatives, 100 iterations, checkerboard start.

The scheme is the semi-implicit one used by the reference implementations
(Chan and Vese 2001; scikit-image's `chan_vese`), vectorised over a batch in
PyTorch so a database evolves on the GPU in seconds. An explicit update was
tried first and is why this note exists: with a hard sign checkerboard it either
freezes (the fitting term is far weaker than curvature on [0, 1] images) or lets
|phi| grow until the regularised delta suppresses all motion. The semi-implicit
denominator is what keeps the front moving for a hundred iterations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# The thesis's settings (section 3.2.3). A change here is a deviation from the
# replication and belongs in a run config, not in this file.
THESIS = dict(lambda1=1.0, lambda2=1.0, mu=0.2, dt=0.5, iterations=100, eps=1.0)

ETA = 1e-8


def checkerboard(height: int, width: int, square: int = 5, device=None) -> torch.Tensor:
    """phi_0: a smooth checkerboard, sin(pi y / s) * sin(pi x / s).

    Smooth, not thresholded: the magnitude carries the gradient information the
    curvature term needs. Its zero level set is the initial contour, which is
    the "binary step function" the thesis describes once it is thresholded.
    """
    ys = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, -1)
    return torch.sin(torch.pi * ys / square) * torch.sin(torch.pi * xs / square)


def _delta(phi: torch.Tensor, eps: float) -> torch.Tensor:
    """Regularised Dirac delta, eps / (pi (eps^2 + phi^2))."""
    return eps / (torch.pi * (eps * eps + phi * phi))


def _pad(phi: torch.Tensor) -> torch.Tensor:
    return F.pad(phi, (1, 1, 1, 1), mode="replicate")


def evolve(
    images: torch.Tensor,
    iterations: int = THESIS["iterations"],
    mu: float = THESIS["mu"],
    dt: float = THESIS["dt"],
    lambda1: float = THESIS["lambda1"],
    lambda2: float = THESIS["lambda2"],
    eps: float = THESIS["eps"],
    square: int = 5,
) -> torch.Tensor:
    """Evolve the VLS on a batch of grayscale images in [0, 1].

    images: (B, H, W). returns (B, iterations, H, W) uint8 masks M_1..M_T.
    """
    if images.dim() != 3:
        raise ValueError(f"expected (B, H, W), got {tuple(images.shape)}")
    b, h, w = images.shape
    img = images.float()

    phi = checkerboard(h, w, square, images.device).expand(b, h, w).clone()
    masks = torch.empty((b, iterations, h, w), dtype=torch.uint8, device=images.device)

    for step in range(iterations):
        p = _pad(phi)
        # One sided differences, as in the semi-implicit Chan-Vese discretisation.
        phixp = p[:, 1:-1, 2:] - p[:, 1:-1, 1:-1]
        phixn = p[:, 1:-1, 1:-1] - p[:, 1:-1, :-2]
        phix0 = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / 2.0
        phiyp = p[:, 2:, 1:-1] - p[:, 1:-1, 1:-1]
        phiyn = p[:, 1:-1, 1:-1] - p[:, :-2, 1:-1]
        phiy0 = (p[:, 2:, 1:-1] - p[:, :-2, 1:-1]) / 2.0

        c1 = 1.0 / torch.sqrt(ETA + phixp**2 + phiy0**2)
        c2 = 1.0 / torch.sqrt(ETA + phixn**2 + phiy0**2)
        c3 = 1.0 / torch.sqrt(ETA + phix0**2 + phiyp**2)
        c4 = 1.0 / torch.sqrt(ETA + phix0**2 + phiyn**2)

        inside = (phi > 0).float()
        outside = 1.0 - inside
        n_in = inside.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        n_out = outside.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean_in = (img * inside).sum(dim=(1, 2), keepdim=True) / n_in
        mean_out = (img * outside).sum(dim=(1, 2), keepdim=True) / n_out

        delta = _delta(phi, eps)
        fitting = -lambda1 * (img - mean_in) ** 2 + lambda2 * (img - mean_out) ** 2
        smoothing = (
            c1 * p[:, 1:-1, 2:] + c2 * p[:, 1:-1, :-2] + c3 * p[:, 2:, 1:-1] + c4 * p[:, :-2, 1:-1]
        )

        phi = (phi + dt * delta * (mu * smoothing + fitting)) / (
            1.0 + mu * dt * delta * (c1 + c2 + c3 + c4)
        )
        masks[:, step] = (phi > 0).to(torch.uint8)

    return masks
