import copy
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.augment import apply_geometric, apply_structural
from src.datasets.streams import build_stream
from src.datasets.taxonomy import binary_label

# UT-Interaction class order (Phase 1 baseline, num_classes: 6 in baseline.yaml).
CLASS_KEYWORDS = {
    "handshake": 0,
    "hug": 1,
    "kick": 2,
    "point": 3,
    "punch": 4,
    "push": 5,
}


def normalize_skeleton(kp, scores, legacy=False):
    """Center + scale a clip's keypoints into a camera-resolution-invariant space.

    Using only visible joints (score > 0), subtract the per-axis mean and divide
    by the RMS distance from that mean (a single scalar, so the x/y aspect and the
    geometry between the two people are preserved). Missing joints stay at 0. This
    removes the absolute pixel position/scale that otherwise lets the model shortcut
    on a dataset's coordinate range instead of the motion itself.

    ``legacy=True`` reproduces the original scale, which took the std of x and y
    *pooled*, i.e. about their combined mean rather than each axis's own. That made
    the scale grow with |mean_x - mean_y| — where in frame people stood — so it
    leaked frame position, and through it dataset identity (NTU clips landed at
    ~0.53x the intended scale, Bullying10K at ~0.86x). Kept only so checkpoints
    trained before the fix can run on the inputs they were trained on; select it
    with ``normalize: legacy`` in the config.
    """
    mask = scores > 0  # (T, M, 17)
    if not mask.any():
        return kp
    valid = kp[mask]  # (K, 2)
    mean = valid.mean(axis=0)
    if legacy:
        scale = float(valid.std())
    else:
        scale = float(np.sqrt(((valid - mean) ** 2).mean()))
    normed = (kp - mean) / (scale + 1e-6)
    return np.where(mask[..., None], normed, 0.0).astype(np.float32)


def coerce_persons(kp, scores, max_persons):
    """Force the person axis (M) to exactly ``max_persons``.

    Different caches were pose-extracted with different person counts (2 for the
    lab datasets, up to 8 for the crowd-capable surveillance ones), but the model
    is a fixed two-person graph. When there are too many people the most-visible
    ``max_persons`` (by total joint confidence) are kept; too few are zero-padded.
    """
    m = kp.shape[1]
    if max_persons is None or m == max_persons:
        return kp, scores
    if m > max_persons:
        visibility = scores.sum(axis=(0, 2))  # (M,) total confidence per person
        keep = np.sort(np.argsort(visibility)[::-1][:max_persons])
        return kp[:, keep], scores[:, keep]
    pad_kp = np.zeros((kp.shape[0], max_persons, kp.shape[2], kp.shape[3]), dtype=kp.dtype)
    pad_sc = np.zeros((scores.shape[0], max_persons, scores.shape[2]), dtype=scores.dtype)
    pad_kp[:, :m], pad_sc[:, :m] = kp, scores
    return pad_kp, pad_sc


def features_to_tensor(
    kp,
    scores,
    target_frames,
    normalize=False,
    max_persons=None,
    augment=None,
    stream="joint",
    rng=None,
):
    """(T, M, 17, 2) keypoints + (T, M, 17) scores -> ST-GCN tensor (C=3, T, V, M).

    Temporally pads (edge) or truncates to ``target_frames`` and stacks the
    per-joint confidence on as a third channel. When ``normalize`` is set, the
    skeleton is centered+scaled per clip (see normalize_skeleton) so different
    camera resolutions land in a common space. ``max_persons`` coerces the person
    axis to a fixed size so caches with different M can be batched together.

    ``augment`` (an AugmentConfig, training splits only) runs structural transforms
    before normalization — they change the visible-joint statistics normalization
    is computed from — and geometric ones after, where they survive it. ``stream``
    selects joint / bone / joint_motion / bone_motion (see datasets/streams.py).
    """
    kp, scores = coerce_persons(kp, scores, max_persons)
    if augment is not None:
        rng = rng if rng is not None else np.random.default_rng()
        kp, scores = apply_structural(kp, scores, rng, augment)
    if normalize:
        # ``normalize`` is True/False from most configs, or the string "legacy" to
        # reproduce the pre-fix scale for checkpoints trained on it.
        kp = normalize_skeleton(kp, scores, legacy=normalize == "legacy")
    if augment is not None:
        kp = apply_geometric(kp, rng, augment)
    T = kp.shape[0]
    if T < target_frames:
        kp = np.pad(kp, ((0, target_frames - T), (0, 0), (0, 0), (0, 0)), mode="edge")
        scores = np.pad(scores, ((0, target_frames - T), (0, 0), (0, 0)), mode="edge")
    elif T > target_frames:
        kp = kp[:target_frames]
        scores = scores[:target_frames]
    features = np.concatenate([kp, np.expand_dims(scores, axis=-1)], axis=-1)
    tensor = torch.tensor(features, dtype=torch.float32).permute(3, 0, 2, 1)
    return build_stream(tensor, stream)


def split_indices(n, seed=42, val_frac=0.15, test_frac=0.15):
    """Deterministic train/val/test index split of ``n`` items (seeded permutation).

    Returns (train_idx, val_idx, test_idx) numpy arrays. The same seed yields the
    same partition, so training and evaluation agree on the held-out test set.
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    return perm[n_test + n_val :], perm[n_test : n_test + n_val], perm[:n_test]


def label_from_filename(filename):
    """Derive a class index from a clip filename.

    Tries, in order: a known class keyword anywhere in the name, a trailing
    integer (e.g. ``clip_03.npz`` -> 3), then 0. Replace with a dataset-specific
    label source (e.g. NTU action id) as more datasets join the unified set.
    """
    stem = os.path.basename(filename).replace(".npz", "").lower()
    for keyword, idx in CLASS_KEYWORDS.items():
        if keyword in stem:
            return idx
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return 0


class _StreamingDataset(Dataset):
    """Shared augmentation/stream plumbing for the two pose-cache datasets."""

    augment_seed = None

    def with_augment(self, augment, seed=None):
        """Shallow clone that augments, sharing this instance's scanned file list.

        Train and val are ``Subset``s over the *same* index space, so they must come
        from datasets that enumerate samples identically — and re-scanning 29k .npz
        files just to turn augmentation on would be wasteful. Cloning keeps one scan
        and lets only the training subset be augmented.

        ``seed`` makes the transform deterministic per sample index, which training
        does not want (every epoch should differ) but the robustness sweep requires.
        """
        clone = copy.copy(self)
        clone.augment = augment
        clone.augment_seed = seed
        return clone

    def with_stream(self, stream):
        """Shallow clone reading a different input stream (joint / bone / motion)."""
        clone = copy.copy(self)
        clone.stream = stream
        return clone

    def _rng(self, idx):
        """Per-sample generator; seeded only when reproducibility was asked for."""
        if self.augment_seed is None:
            return None
        return np.random.default_rng(self.augment_seed + idx)


class UnifiedSkeletonDataset(_StreamingDataset):
    """Loads unified .npz frame structures according to baseline.yaml parameters."""

    def __init__(
        self,
        data_dir,
        target_frames=64,
        normalize=False,
        max_persons=None,
        augment=None,
        stream="joint",
    ):
        self.data_dir = data_dir
        self.target_frames = target_frames
        self.normalize = normalize
        self.max_persons = max_persons
        self.augment = augment
        self.stream = stream
        if os.path.exists(data_dir):
            # Sorted for a reproducible order, so split_indices gives the same
            # train/val/test partition across runs and between train & evaluate.
            self.file_list = sorted(
                os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")
            )
        else:
            self.file_list = []

    def __len__(self):
        return len(self.file_list)

    def labels(self):
        """Every clip's label, read without decoding keypoints (cached after first call).

        Used for class-balanced loss weighting, which needs the label distribution of
        the training split before training starts.
        """
        if getattr(self, "_labels", None) is None:
            found = []
            for path in self.file_list:
                with np.load(path) as data:
                    found.append(
                        int(data["label"]) if "label" in data else label_from_filename(path)
                    )
            self._labels = found
        return self._labels

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)

        # Prefer an explicit label written by a dataset converter (e.g.
        # bullying10k_poses.py); fall back to parsing it from the filename.
        if "label" in data:
            label = int(data["label"])
        else:
            label = label_from_filename(file_path)

        tensor_data = features_to_tensor(
            data["keypoints"],
            data["scores"],
            self.target_frames,
            self.normalize,
            self.max_persons,
            self.augment,
            self.stream,
            self._rng(idx),
        )
        return tensor_data, torch.tensor(label, dtype=torch.long)


class MultiDatasetSkeletonDataset(_StreamingDataset):
    """Pools several pose caches under the binary aggressive/neutral label space.

    ``specs`` is a list of (dataset_name, cache_dir) pairs. Each sample's native
    annotation is collapsed via taxonomy.binary_label; samples whose aggression
    can't be determined are skipped. ``self.datasets`` is the per-sample source
    dataset, used for cross-dataset evaluation and per-dataset ablations.
    """

    def __init__(
        self,
        specs,
        target_frames=64,
        normalize=False,
        max_persons=None,
        augment=None,
        stream="joint",
    ):
        self.target_frames = target_frames
        self.normalize = normalize
        self.max_persons = max_persons
        self.augment = augment
        self.stream = stream
        self.samples = []  # (path, dataset_name, binary_label)
        for name, cache in specs:
            if not os.path.isdir(cache):
                continue
            for fname in sorted(os.listdir(cache)):
                if not fname.endswith(".npz"):
                    continue
                path = os.path.join(cache, fname)
                with np.load(path, allow_pickle=True) as data:
                    label = binary_label(data, path, name)
                if label is not None:
                    self.samples.append((path, name, label))
        self.datasets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, _, label = self.samples[idx]
        with np.load(path) as data:
            tensor_data = features_to_tensor(
                data["keypoints"],
                data["scores"],
                self.target_frames,
                self.normalize,
                self.max_persons,
                self.augment,
                self.stream,
                self._rng(idx),
            )
        return tensor_data, torch.tensor(label, dtype=torch.long)
