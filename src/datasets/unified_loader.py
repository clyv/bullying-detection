import os

import numpy as np
import torch
from torch.utils.data import Dataset

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


def features_to_tensor(kp, scores, target_frames):
    """(T, M, 17, 2) keypoints + (T, M, 17) scores -> ST-GCN tensor (C=3, T, V, M).

    Temporally pads (edge) or truncates to ``target_frames`` and stacks the
    per-joint confidence on as a third channel.
    """
    T = kp.shape[0]
    if T < target_frames:
        kp = np.pad(kp, ((0, target_frames - T), (0, 0), (0, 0), (0, 0)), mode="edge")
        scores = np.pad(scores, ((0, target_frames - T), (0, 0), (0, 0)), mode="edge")
    elif T > target_frames:
        kp = kp[:target_frames]
        scores = scores[:target_frames]
    features = np.concatenate([kp, np.expand_dims(scores, axis=-1)], axis=-1)
    return torch.tensor(features, dtype=torch.float32).permute(3, 0, 2, 1)


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


class UnifiedSkeletonDataset(Dataset):
    """Loads unified .npz frame structures according to baseline.yaml parameters."""

    def __init__(self, data_dir, target_frames=64):  # Updated to match your num_frames: 64
        self.data_dir = data_dir
        self.target_frames = target_frames
        if os.path.exists(data_dir):
            self.file_list = [
                os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")
            ]
        else:
            self.file_list = []

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)

        # Prefer an explicit label written by a dataset converter (e.g.
        # bullying10k_poses.py); fall back to parsing it from the filename.
        if "label" in data:
            label = int(data["label"])
        else:
            label = label_from_filename(file_path)

        tensor_data = features_to_tensor(data["keypoints"], data["scores"], self.target_frames)
        return tensor_data, torch.tensor(label, dtype=torch.long)


class MultiDatasetSkeletonDataset(Dataset):
    """Pools several pose caches under the binary aggressive/neutral label space.

    ``specs`` is a list of (dataset_name, cache_dir) pairs. Each sample's native
    annotation is collapsed via taxonomy.binary_label; samples whose aggression
    can't be determined are skipped. ``self.datasets`` is the per-sample source
    dataset, used for cross-dataset evaluation and per-dataset ablations.
    """

    def __init__(self, specs, target_frames=64):
        self.target_frames = target_frames
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
            tensor_data = features_to_tensor(data["keypoints"], data["scores"], self.target_frames)
        return tensor_data, torch.tensor(label, dtype=torch.long)
