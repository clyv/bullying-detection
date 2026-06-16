import os

import numpy as np
import torch
from torch.utils.data import Dataset

# UT-Interaction class order (Phase 1 baseline, num_classes: 6 in baseline.yaml).
CLASS_KEYWORDS = {
    "handshake": 0,
    "hug": 1,
    "kick": 2,
    "point": 3,
    "punch": 4,
    "push": 5,
}


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

        kp = data["keypoints"]  # Shape: (T, M, 17, 2)
        scores = data["scores"]  # Shape: (T, M, 17)

        label = label_from_filename(file_path)

        T, M, V, _ = kp.shape

        # Temporal Uniform Padding / Truncation to your exact window size (64)
        if T < self.target_frames:
            kp = np.pad(kp, ((0, self.target_frames - T), (0, 0), (0, 0), (0, 0)), mode="edge")
            scores = np.pad(scores, ((0, self.target_frames - T), (0, 0), (0, 0)), mode="edge")
        elif T > self.target_frames:
            kp = kp[: self.target_frames]
            scores = scores[: self.target_frames]

        # Concatenate X, Y coordinates with the per-joint confidence scores -> Channel dimension = 3
        scores_expanded = np.expand_dims(scores, axis=-1)
        features = np.concatenate([kp, scores_expanded], axis=-1)

        # Reshape to standard ST-GCN dimension protocol: (Channels, Timesteps, Vertices, Monsters/Actors)
        tensor_data = torch.tensor(features, dtype=torch.float32).permute(3, 0, 2, 1)

        return tensor_data, torch.tensor(label, dtype=torch.long)
