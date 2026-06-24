import numpy as np
import torch

from src.datasets.unified_loader import (
    UnifiedSkeletonDataset,
    normalize_skeleton,
    split_indices,
)
from src.models.graph import Graph
from src.models.stgcn import STGCNBaseline


def test_normalize_skeleton_is_resolution_invariant():
    # Same pose at two camera scales should normalize to (almost) the same thing.
    rng = np.random.default_rng(0)
    base = rng.random((8, 2, 17, 2)).astype("float32")
    scores = np.ones((8, 2, 17), dtype="float32")
    small = normalize_skeleton(base * 50 + 10, scores)  # e.g. 346x260-ish
    large = normalize_skeleton(base * 1000 + 500, scores)  # e.g. 1920x1080-ish
    assert np.allclose(small, large, atol=1e-3)
    # centered (visible-joint mean ~ 0) and scaled (std ~ 1)
    assert abs(float(small.mean())) < 1e-4
    assert abs(float(small.std()) - 1.0) < 1e-2


def test_normalize_skeleton_keeps_missing_joints_zero():
    kp = np.ones((4, 2, 17, 2), dtype="float32")
    scores = np.ones((4, 2, 17), dtype="float32")
    scores[:, :, 5] = 0  # joint 5 missing everywhere
    out = normalize_skeleton(kp, scores)
    assert (out[:, :, 5] == 0).all()


def test_split_indices_disjoint_complete_and_deterministic():
    train, val, test = split_indices(100, seed=42, val_frac=0.15, test_frac=0.15)
    assert len(test) == 15 and len(val) == 15 and len(train) == 70
    union = set(train) | set(val) | set(test)
    assert union == set(range(100))  # complete partition
    assert len(union) == 100  # disjoint (no overlap)
    # same seed -> identical split; different seed -> different test set
    train2, _, test2 = split_indices(100, seed=42)
    assert (test == test2).all() and (train == train2).all()
    _, _, test3 = split_indices(100, seed=7)
    assert set(test3) != set(test)


def test_graph_adjacency_shape():
    graph = Graph(strategy="spatial")
    assert graph.A.shape == (3, 17, 17)
    assert int(graph.A[0].sum()) == 17  # Self-loops identity check


def test_model_forward_pass_dimensions():
    # Shape protocol: (Batch_Size, Channels, Timesteps, Vertices/Joints, Monsters/Actors)
    mock_input = torch.randn(2, 3, 100, 17, 2)
    model = STGCNBaseline(in_channels=3, num_classes=2)
    model.eval()
    with torch.no_grad():
        output = model(mock_input)
    assert output.shape == (2, 2)  # (Batch_size, Num_classes)


def test_loader_padding_and_truncation(tmp_path):
    # Construct short sequence .npz mock file
    short_kp = np.random.randn(20, 2, 17, 2)
    short_scores = np.random.rand(20, 2, 17)
    file_path = tmp_path / "mock_agg_01.npz"
    np.savez(file_path, keypoints=short_kp, scores=short_scores)

    dataset = UnifiedSkeletonDataset(data_dir=str(tmp_path), target_frames=150)
    assert len(dataset) == 1

    tensor_data, label = dataset[0]
    assert tensor_data.shape == (3, 150, 17, 2)  # (C, T, V, M)
    assert label.item() == 1  # Derived correctly from file signature name


def test_loader_prefers_stored_label(tmp_path):
    # A converter-written 'label' field overrides filename parsing.
    file_path = tmp_path / "anything.npz"
    np.savez(
        file_path,
        keypoints=np.zeros((10, 2, 17, 2)),
        scores=np.zeros((10, 2, 17)),
        label=7,
    )
    dataset = UnifiedSkeletonDataset(data_dir=str(tmp_path), target_frames=16)
    _, label = dataset[0]
    assert label.item() == 7
