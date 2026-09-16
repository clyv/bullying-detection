"""Confidence calibration and principled abstention.

Two things this project needed and did not have:

**Temperature scaling.** One scalar T, fitted on the held-out validation split,
divides the logits before softmax. It cannot change any prediction (dividing by a
positive constant preserves the argmax) so accuracy is untouched, but it turns a
saturated score back into a probability. Expected Calibration Error (ECE) measures
how far off the uncalibrated model was.

**Conformal abstention.** ``min_pair_height=120`` was found by trial on one video.
Split-conformal prediction replaces it with a threshold calibrated on held-out data
that carries a distribution-free guarantee: at level ``alpha``, the prediction set
contains the true class at least ``1 - alpha`` of the time. When the set contains
both classes the model abstains — the same behaviour, but now with a coverage
number attached to it rather than a magic constant.

Pure numpy except for the temperature fit, so it stays testable without a GPU.
"""

from __future__ import annotations

import itertools

import numpy as np


def softmax(logits, temperature=1.0):
    """Row-wise softmax of ``logits / temperature``, numerically stabilised."""
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(probs, targets, num_bins=15):
    """ECE: average gap between confidence and accuracy, over equal-width bins.

    0 is perfectly calibrated. A model that says 99% and is right 80% of the time
    scores ~0.19 — the number this project's saturated classifier needed.
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets)
    if len(targets) == 0:
        return 0.0
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == targets).astype(np.float64)

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for low, high in itertools.pairwise(edges):
        # Lower-exclusive bins, with the first bin closed so conf == 0 is counted.
        in_bin = (confidence > low) & (confidence <= high)
        if low == 0.0:
            in_bin |= confidence == 0.0
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        ece += weight * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


MIN_CALIBRATION_SAMPLES = 16
TEMPERATURE_BOUNDS = (0.05, 10.0)


def fit_temperature(logits, targets, max_iter=200, lr=0.05, min_samples=MIN_CALIBRATION_SAMPLES):
    """Fit the scalar temperature minimising NLL on a held-out split.

    Optimised in log-space so T stays positive without a constrained optimiser.

    Falls back to ``T = 1.0`` (the identity) when the calibration set cannot support
    a fit: too few samples, only one class present, or a perfectly separable set —
    in all three the NLL is minimised as T runs off to 0 or infinity and LBFGS
    returns inf or NaN. Small leave-one-dataset-out folds hit this routinely, and a
    NaN temperature silently poisons every downstream probability.
    """
    import torch

    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    if len(targets) < min_samples or len(np.unique(targets)) < 2:
        return 1.0

    logit_tensor = torch.tensor(logits, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(0) = 1
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logit_tensor / log_t.exp(), target_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.exp().item())
    if not np.isfinite(temperature):
        return 1.0
    return float(np.clip(temperature, *TEMPERATURE_BOUNDS))


def conformal_threshold(probs, targets, alpha=0.1):
    """Split-conformal threshold on the true class's softmax score.

    The returned ``q`` is the ``alpha`` empirical quantile (with the standard finite-
    sample correction) of the probability the model assigned to the correct class on
    calibration data. At prediction time, every class scoring at least ``q`` enters
    the prediction set, which then covers the truth with probability >= 1 - alpha.
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets)
    n = len(targets)
    if n == 0:
        return 0.0
    true_scores = probs[np.arange(n), targets]
    # Conservative rank: ceil((n+1) * alpha) / n, clipped into [0, 1].
    level = np.clip(np.ceil((n + 1) * alpha) / n, 0.0, 1.0)
    return float(np.quantile(true_scores, level, method="lower"))


def prediction_sets(probs, threshold):
    """Boolean (n, num_classes) mask of classes admitted at ``threshold``."""
    return np.asarray(probs, dtype=np.float64) >= threshold


def abstention_report(probs, targets, threshold):
    """Coverage, abstention rate, and selective accuracy at a conformal threshold.

    ``selective_accuracy`` is the number that matters operationally: how often the
    system is right *on the windows it chose to answer*. Abstaining is not a failure
    here — it is the designed behaviour on skeletons too small or too occluded to
    judge, and it is what the pixel-height gate was approximating.
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets)
    if len(targets) == 0:
        return {"coverage": 0.0, "abstention_rate": 0.0, "selective_accuracy": 0.0, "n": 0}

    sets = prediction_sets(probs, threshold)
    set_sizes = sets.sum(axis=1)
    covered = sets[np.arange(len(targets)), targets]
    # A set with exactly one class is a decision; empty or multi-class is an abstain.
    decided = set_sizes == 1
    predicted = probs.argmax(axis=1)
    selective = (predicted[decided] == targets[decided]).mean() if decided.any() else 0.0
    return {
        "coverage": float(covered.mean()),
        "abstention_rate": float(1.0 - decided.mean()),
        "selective_accuracy": float(selective),
        "n": len(targets),
    }


def format_calibration(temperature, ece_before, ece_after, report=None):
    """Human-readable calibration summary for evaluation output."""
    note = ""
    if temperature >= TEMPERATURE_BOUNDS[1] - 1e-6:
        # NLL is minimised as T -> infinity exactly when the logits carry no usable
        # ranking, so a pegged temperature means the model is near-uninformative on
        # this split — not that calibration succeeded.
        note = "  <- at bound: logits carry little signal on this split"
    elif temperature <= TEMPERATURE_BOUNDS[0] + 1e-6:
        note = "  <- at bound: validation split is (near) perfectly separable"
    lines = [
        "Calibration:",
        f"  temperature      T={temperature:.3f}{note}",
        f"  ECE before       {ece_before:.4f}",
        f"  ECE after        {ece_after:.4f}",
    ]
    if report:
        lines += [
            f"  coverage         {report['coverage'] * 100:.1f}%",
            f"  abstention rate  {report['abstention_rate'] * 100:.1f}%",
            f"  selective acc.   {report['selective_accuracy'] * 100:.2f}%",
        ]
    return "\n".join(lines)
