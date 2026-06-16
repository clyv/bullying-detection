import os
import numpy as np
import torch
from torch.utils.data import Dataset


class UnifiedSkeletonDataset(Dataset):
    """Loads unified .npz frame structures according to baseline.yaml parameters."""

    def __init__(
        self, data_dir, target_frames=64
    ):  # Updated to match your num_frames: 64
        self.data_dir = data_dir
        self.target_frames = target_frames
        if os.path.exists(data_dir):
            self.file_list = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npz")
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

        # Parse the true multi-class label from your data or file naming conventions
        # For a 6-class setup (0 to 5), map filenames or label indexes appropriately
        # Placeholder: extracting trailing integer or defaulting to class 0
        try:
            label = int(os.path.basename(file_path).split("_")[-1].replace(".npz", ""))
        except ValueError:
            label = 0

        T, M, V, _ = kp.shape

        # Temporal Uniform Padding / Truncation to your exact window size (64)
        if T < self.target_frames:
            kp = np.pad(
                kp, ((0, self.target_frames - T), (0, 0), (0, 0), (0, 0)), mode="edge"
            )
            scores = np.pad(
                scores, ((0, self.target_frames - T), (0, 0), (0, 0)), mode="edge"
            )
        elif T > self.target_frames:
            kp = kp[: self.target_frames]
            scores = scores[: self.target_frames]

        # Concatenate X, Y coordinates with the per-joint confidence scores -> Channel dimension = 3
        scores_expanded = np.expand_dims(scores, axis=-1)
        features = np.concatenate([kp, scores_expanded], axis=-1)

        # Reshape to standard ST-GCN dimension protocol: (Channels, Timesteps, Vertices, Monsters/Actors)
        tensor_data = torch.tensor(features, dtype=torch.float32).permute(3, 0, 2, 1)

        return tensor_data, torch.tensor(label, dtype=torch.long)