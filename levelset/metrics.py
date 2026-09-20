"""The thesis's metrics (section 3.4), computed over the whole evaluation set.

Precision, recall, F1 and IoU on the foreground class, at the thesis's 0.5
threshold. Counts are accumulated over batches and the metric is computed once
at the end, which is the micro average; averaging per batch would weight a
half empty last batch like a full one.

`boundary_f1` is an addition, not a replication: the thesis observed that the
models "localize the objects in the images, but fail to capture the boundaries"
(section 5.2) without measuring it. This measures it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class Counts:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0
    b_tp: float = 0.0
    b_fp: float = 0.0
    b_fn: float = 0.0
    c_tp: float = 0.0
    c_fp: float = 0.0
    c_fn: float = 0.0
    c_n: float = 0.0
    n: int = 0
    extra: dict = field(default_factory=dict)

    def update_change(self, logits: torch.Tensor, target: torch.Tensor, previous: torch.Tensor,
                      threshold: float = 0.5) -> None:
        """Score only where the level set actually moved.

        The Chan-Vese front displaces about half a percent of pixels per
        iteration, so a global IoU is dominated by static background and a model
        that predicts no change at all scores about 0.99. Restricting to the
        symmetric difference between the last input mask and the target is what
        separates learning the evolution from learning to stand still. This is a
        correction to the thesis's evaluation, not part of its replication.
        """
        pred = (torch.sigmoid(logits) > threshold).float()
        t = target.float()
        region = (t != previous.float()).float()
        self.c_tp += float((pred * t * region).sum())
        self.c_fp += float((pred * (1 - t) * region).sum())
        self.c_fn += float(((1 - pred) * t * region).sum())
        self.c_n += float(region.sum())

    def update(self, logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> None:
        pred = (torch.sigmoid(logits) > threshold).float()
        t = target.float()
        self.tp += float((pred * t).sum())
        self.fp += float((pred * (1 - t)).sum())
        self.fn += float(((1 - pred) * t).sum())
        self.tn += float(((1 - pred) * (1 - t)).sum())
        self.n += int(t.shape[0])

        pb, tb = _boundary(pred), _boundary(t)
        # A boundary pixel counts as matched if a true boundary lies within one
        # pixel, the usual tolerance for boundary F scores.
        tb_d = F.max_pool2d(tb.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        pb_d = F.max_pool2d(pb.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        self.b_tp += float((pb * tb_d).sum())
        self.b_fp += float((pb * (1 - tb_d)).sum())
        self.b_fn += float((tb * (1 - pb_d)).sum())

    def result(self) -> dict[str, float]:
        eps = 1e-9
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        b_p = self.b_tp / (self.b_tp + self.b_fp + eps)
        b_r = self.b_tp / (self.b_tp + self.b_fn + eps)
        c_iou = self.c_tp / (self.c_tp + self.c_fp + self.c_fn + eps)
        c_p = self.c_tp / (self.c_tp + self.c_fp + eps)
        c_r = self.c_tp / (self.c_tp + self.c_fn + eps)
        return {
            "change_iou": c_iou,
            "change_f1": 2 * c_p * c_r / (c_p + c_r + eps),
            "change_recall": c_r,
            "change_pixels": self.c_n,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps),
            "boundary_f1": 2 * b_p * b_r / (b_p + b_r + eps),
        }


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    """One pixel wide boundary of a binary mask: dilation minus erosion."""
    m = mask.unsqueeze(1)
    dil = F.max_pool2d(m, 3, stride=1, padding=1)
    ero = -F.max_pool2d(-m, 3, stride=1, padding=1)
    return (dil - ero).squeeze(1).clamp(0, 1)
