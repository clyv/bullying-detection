"""Skeleton augmentation aimed at the degradations this pipeline actually hits.

Generic augmentation would be worth a point or two. These transforms are chosen
against the diagnosed failure modes instead:

* **joint dropout** — YOLO-Pose returns missing keypoints at range. Training on
  complete skeletons and inferring on gappy ones is a distribution shift the model
  has never seen. This is the single most relevant transform here.
* **scale jitter** — residual scale variation that per-clip normalization does not
  fully remove (different aspect ratios, partial bodies, cropped feet).
* **temporal crop + resample** — datasets disagree about where an action starts and
  ends. A model tuned to one corpus's trimming convention transfers badly; random
  re-cropping breaks that dependence.
* **horizontal flip** — with the COCO left/right joint indices swapped, otherwise
  the model learns a spurious handedness prior from whichever way the actors in a
  given corpus happened to face.
* **person swap** — aggression between two people is symmetric under relabelling
  which skeleton is person 0, so this is a free, exactly label-preserving transform.

Split by where they must run relative to per-clip normalization: structural
transforms change the visible-joint statistics normalization is computed from, so
they run first; geometric transforms run after, where they survive it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO-17 left/right joint index pairs, swapped on horizontal flip.
COCO_FLIP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


@dataclass
class AugmentConfig:
    """Augmentation strengths. All probabilities; zero disables the transform."""

    joint_dropout: float = 0.1  # per-joint probability of being marked invisible
    person_dropout: float = 0.0  # probability of blanking one whole person
    coord_noise: float = 0.02  # gaussian sigma, in units of skeleton extent
    flip_prob: float = 0.5
    person_swap_prob: float = 0.5
    temporal_crop: float = 0.2  # max fraction of the clip trimmed off each end
    scale_jitter: float = 0.2  # +/- fraction on the normalized scale
    rotate_degrees: float = 15.0
    shear: float = 0.1

    @classmethod
    def from_dict(cls, raw):
        """Build from a config mapping; ``None`` or empty disables augmentation."""
        if not raw:
            return None
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})


def apply_structural(kp, scores, rng, cfg: AugmentConfig):
    """Transforms that must run *before* normalization, on (T, M, V, 2) + (T, M, V).

    Returns new arrays; the caller's originals (memory-mapped from .npz) are never
    modified in place.
    """
    kp = np.array(kp, dtype=np.float32, copy=True)
    scores = np.array(scores, dtype=np.float32, copy=True)

    if cfg.temporal_crop > 0 and kp.shape[0] > 4:
        kp, scores = _temporal_crop(kp, scores, rng, cfg.temporal_crop)

    if cfg.flip_prob > 0 and rng.random() < cfg.flip_prob:
        kp, scores = _horizontal_flip(kp, scores)

    if cfg.person_swap_prob > 0 and kp.shape[1] > 1 and rng.random() < cfg.person_swap_prob:
        order = rng.permutation(kp.shape[1])
        kp, scores = kp[:, order], scores[:, order]

    if cfg.coord_noise > 0:
        extent = _extent(kp, scores)
        kp += rng.normal(0.0, cfg.coord_noise * extent, size=kp.shape).astype(np.float32)

    # Dropout last: it zeroes confidences, and the transforms above should see the
    # skeleton as the extractor produced it.
    if cfg.person_dropout > 0 and kp.shape[1] > 1:
        for person in range(kp.shape[1]):
            if rng.random() < cfg.person_dropout:
                scores[:, person] = 0.0
                kp[:, person] = 0.0
    if cfg.joint_dropout > 0:
        drop = rng.random(scores.shape) < cfg.joint_dropout
        scores[drop] = 0.0
        kp[drop] = 0.0

    return kp, scores


def apply_geometric(kp, rng, cfg: AugmentConfig):
    """Transforms that run *after* normalization, on (T, M, V, 2).

    Rotation and shear survive centre+scalar-scale normalization unchanged, and
    scale jitter deliberately reintroduces the scale variation normalization
    removed, so the model cannot assume its input is perfectly normalized — which
    on real footage with partially-visible bodies it is not.
    """
    kp = np.array(kp, dtype=np.float32, copy=True)
    matrix = np.eye(2, dtype=np.float32)

    if cfg.rotate_degrees > 0:
        theta = np.deg2rad(rng.uniform(-cfg.rotate_degrees, cfg.rotate_degrees))
        cos, sin = np.cos(theta), np.sin(theta)
        matrix = matrix @ np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    if cfg.shear > 0:
        s = rng.uniform(-cfg.shear, cfg.shear)
        matrix = matrix @ np.array([[1.0, s], [0.0, 1.0]], dtype=np.float32)
    if cfg.scale_jitter > 0:
        matrix = matrix * float(rng.uniform(1.0 - cfg.scale_jitter, 1.0 + cfg.scale_jitter))

    # Missing joints sit at exactly (0, 0) and must stay there — a linear map keeps
    # the origin fixed, so no explicit masking is needed.
    return kp @ matrix.T


def _temporal_crop(kp, scores, rng, max_fraction):
    """Take a random contiguous sub-window; length normalization happens downstream."""
    total = kp.shape[0]
    start = int(rng.integers(0, max(1, int(total * max_fraction))))
    end = total - int(rng.integers(0, max(1, int(total * max_fraction))))
    if end - start < 4:  # keep enough frames to be a clip at all
        return kp, scores
    return kp[start:end], scores[start:end]


def _horizontal_flip(kp, scores):
    """Mirror x about the clip's own centre and swap left/right joint indices."""
    visible = scores > 0
    if visible.any():
        centre = kp[..., 0][visible].mean()
        kp[..., 0] = np.where(visible, 2.0 * centre - kp[..., 0], 0.0)
    for left, right in COCO_FLIP_PAIRS:
        kp[:, :, [left, right]] = kp[:, :, [right, left]]
        scores[:, :, [left, right]] = scores[:, :, [right, left]]
    return kp, scores


def _extent(kp, scores):
    """Rough skeleton size in pixels, so noise scales with the subject not the frame."""
    visible = scores > 0
    if not visible.any():
        return 1.0
    coords = kp[visible]
    spread = float(coords.max(axis=0).max() - coords.min(axis=0).min())
    return max(spread, 1.0)
