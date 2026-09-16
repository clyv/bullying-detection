"""Loss functions for the aggression classifier.

Cross-entropy is what drove this project's headline bug: on real corridor footage
the model returned P(aggressive) = 1.000 for every pair, which is not a threshold
problem but a calibration one — CE has no term that penalises confidence, so it
pushes logits apart until the softmax saturates and every downstream decision
(ranking incidents, choosing an operating point, reporting a false-alarm rate)
loses the information it needs.

Focal loss (Lin et al.; calibration analysis by Mukhoti et al., NeurIPS 2020) adds
a ``(1 - p)^gamma`` modulator that down-weights already-correct examples. Mukhoti et
al. show this acts as an implicit entropy regulariser: CE-trained models need
temperature ~2.5-2.8 to calibrate, focal-trained ones about 1.1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights and label smoothing.

    ``gamma=0`` degenerates to (weighted, smoothed) cross-entropy, which makes the
    ablation against the old behaviour a one-line config change.
    """

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        log_true = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Per-sample CE first, so class weights and smoothing are applied exactly
        # once and the focal modulator scales the already-weighted term.
        # (F.nll_loss has no label_smoothing argument — only F.cross_entropy does,
        # and that would re-apply log_softmax — so smoothing is done here.)
        loss = -log_true
        if self.label_smoothing > 0:
            uniform = -log_probs.mean(dim=-1)
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * uniform

        weights = None
        if self.weight is not None:
            weights = self.weight.to(logits.device).gather(0, targets)
            loss = loss * weights

        if self.gamma > 0:
            loss = loss * (1.0 - log_true.exp()).pow(self.gamma)

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        # Weighted mean normalises by the weight sum, matching CrossEntropyLoss.
        if weights is not None:
            return loss.sum() / weights.sum().clamp(min=1e-8)
        return loss.mean()


def class_weights_from_counts(counts, device=None, cap=10.0):
    """Inverse-frequency weights, normalised to mean 1 and clipped.

    The pooled corpus is not balanced (NTU contributes far more neutral clips than
    anything contributes aggressive ones), and an unweighted loss lets the model buy
    accuracy by leaning on that prior — which is exactly the shortcut that collapses
    under leave-one-dataset-out. The cap stops a nearly-empty class from producing a
    weight so large it destabilises training.
    """
    counts = torch.as_tensor(counts, dtype=torch.float32, device=device)
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights.clamp(max=cap)
    return weights / weights.mean()


def mixup_batch(tensors, targets, num_classes, alpha=0.2, generator=None):
    """Convex-combine pairs of clips and their one-hot labels.

    Returns ``(mixed_tensors, soft_targets)``. Mixup is an unusually good fit for
    skeletons: interpolating two normalized pose sequences stays on the manifold of
    plausible motion far better than interpolating two RGB frames does, and the soft
    targets are themselves a confidence regulariser alongside focal loss.
    """
    if alpha <= 0:
        return tensors, F.one_hot(targets, num_classes).float()
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    lam = max(lam, 1.0 - lam)  # keep the dominant sample dominant
    index = torch.randperm(tensors.size(0), device=tensors.device, generator=generator)
    mixed = lam * tensors + (1.0 - lam) * tensors[index]
    one_hot = F.one_hot(targets, num_classes).float()
    soft = lam * one_hot + (1.0 - lam) * one_hot[index]
    return mixed, soft


def soft_target_cross_entropy(logits, soft_targets):
    """Cross-entropy against a soft (non-one-hot) target distribution, for mixup."""
    return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
