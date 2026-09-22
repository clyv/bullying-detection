"""Trivial baselines that any learned model has to beat to mean something.

The motion-energy baseline is one number per clip — how fast the most agitated
person moves, in body-heights per frame — and one threshold: faster than the
threshold means "aggressive". It encodes the hypothesis "aggression is just fast
movement", which is exactly the shortcut a skeleton model can fall into.

Reporting it beside the leave-one-dataset-out results is what showed that the
AGCN beat this single feature by only ~4.7 points across datasets, and was
indistinguishable from it on NTU (whose actors mime violence slowly) and on the
real-CCTV folds. A model gain that doesn't widen that gap is not a gain in
understanding aggression.

Pure numpy, so it runs in CI and costs seconds rather than a GPU run.
"""

from __future__ import annotations

import numpy as np

from src.datasets.unified_loader import coerce_persons

MIN_JOINTS = 4  # joints visible in both frames before a person's speed counts
MIN_HEIGHT = 8.0  # px floor on body height, so a 2px sliver can't produce a huge speed


def clip_motion_energy(kp, scores, max_persons=2):
    """Median over frames of the most-agitated person's speed, in body-heights/frame.

    ``kp`` (T, M, V, 2) pixel keypoints and ``scores`` (T, M, V) confidences, the raw
    cache format. Height-normalized, so a far-away person moving their own height
    per second scores the same as a near one — the comparison is between motions,
    not between camera distances.
    """
    kp, scores = coerce_persons(np.asarray(kp), np.asarray(scores), max_persons)
    if kp.shape[0] < 2:
        return 0.0
    now, prev = scores[1:] > 0, scores[:-1] > 0
    both = now & prev  # (T-1, M, V) joints visible across the step
    count = both.sum(axis=-1)  # (T-1, M)

    disp = np.where(both, np.linalg.norm(kp[1:] - kp[:-1], axis=-1), 0.0)
    mean_disp = disp.sum(axis=-1) / np.maximum(count, 1)

    ys = kp[1:, ..., 1]
    tallest = np.where(now, ys, -np.inf).max(axis=-1)
    lowest = np.where(now, ys, np.inf).min(axis=-1)
    height = np.where(now.any(axis=-1), tallest - lowest, 0.0)
    height = np.maximum(height, MIN_HEIGHT)

    speed = np.where(count >= MIN_JOINTS, mean_disp / height, 0.0)
    return float(np.median(speed.max(axis=1)))


def fit_energy_threshold(energies, labels):
    """Threshold maximising accuracy of ``energy > t -> aggressive`` (exact, O(n log n)).

    One direction only — faster means aggressive — because that direction *is*
    the hypothesis being benchmarked. Returns ``-inf`` when calling everything
    aggressive is optimal.
    """
    energies = np.asarray(energies, dtype=np.float64)
    labels = np.asarray(labels)
    if len(energies) == 0:
        return 0.0
    order = np.argsort(energies, kind="stable")
    sorted_e, sorted_y = energies[order], labels[order]
    # Splitting before position k predicts the first k clips neutral, the rest aggressive.
    neutral_below = np.concatenate([[0], np.cumsum(sorted_y == 0)])
    aggressive_above = (sorted_y == 1).sum() - np.concatenate([[0], np.cumsum(sorted_y == 1)])
    correct = (neutral_below + aggressive_above).astype(np.float64)
    # A threshold can only fall *between* distinct values: splitting inside a run of
    # ties (e.g. every motionless clip at exactly 0.0) scores an accuracy no real
    # threshold can reach, since "energy > t" sends all the ties the same way.
    inside_ties = np.zeros(len(correct), dtype=bool)
    inside_ties[1:-1] = sorted_e[:-1] == sorted_e[1:]
    correct[inside_ties] = -1.0
    k = int(np.argmax(correct))
    return -np.inf if k == 0 else float(sorted_e[k - 1])


def threshold_accuracy(energies, labels, threshold):
    """Accuracy of predicting aggressive whenever energy exceeds ``threshold``."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    return float(((np.asarray(energies) > threshold) == (labels == 1)).mean())
