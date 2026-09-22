import numpy as np
import torch

from src.datasets.unified_loader import (
    UnifiedSkeletonDataset,
    coerce_persons,
    features_to_tensor,
    normalize_skeleton,
    split_indices,
)
from src.models.graph import Graph
from src.models.stgcn import STGCNBaseline


def test_coerce_persons_truncates_to_most_visible():
    kp = np.zeros((5, 8, 17, 2), dtype="float32")
    scores = np.zeros((5, 8, 17), dtype="float32")
    scores[:, 3] = 1.0  # person 3 fully visible
    scores[:, 6] = 0.5  # person 6 half visible
    # everyone else invisible
    kp[:, 3] = 7.0
    kp[:, 6] = 9.0
    out_kp, _out_sc = coerce_persons(kp, scores, max_persons=2)
    assert out_kp.shape == (5, 2, 17, 2)
    # the two most-visible people (3 then 6), kept in sorted index order
    assert (out_kp[:, 0] == 7.0).all() and (out_kp[:, 1] == 9.0).all()


def test_coerce_persons_pads_when_too_few():
    kp = np.ones((4, 1, 17, 2), dtype="float32")
    scores = np.ones((4, 1, 17), dtype="float32")
    out_kp, out_sc = coerce_persons(kp, scores, max_persons=2)
    assert out_kp.shape == (4, 2, 17, 2)
    assert (out_sc[:, 1] == 0).all()  # padded slot is empty


def test_features_to_tensor_coerces_mismatched_person_counts():
    # an 8-person clip and a 2-person clip must produce stackable tensors
    rng = np.random.default_rng(0)
    big = features_to_tensor(
        rng.random((30, 8, 17, 2)).astype("float32"),
        rng.random((30, 8, 17)).astype("float32"),
        64,
        max_persons=2,
    )
    small = features_to_tensor(
        rng.random((30, 2, 17, 2)).astype("float32"),
        rng.random((30, 2, 17)).astype("float32"),
        64,
        max_persons=2,
    )
    assert big.shape == small.shape == (3, 64, 17, 2)


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


def test_normalize_skeleton_is_invariant_to_frame_position():
    # Regression: the test above shifts x and y by the *same* offset, the one case
    # where pooling both axes into one std happens to be right. A person at
    # x~1500, y~300 is not that case — the old scale grew with |mean_x - mean_y|,
    # leaking where in frame people stood (and so which dataset a clip came from).
    rng = np.random.default_rng(1)
    pose = rng.uniform(0, 100, size=(8, 2, 17, 2)).astype("float32")
    scores = np.ones((8, 2, 17), dtype="float32")
    centre = normalize_skeleton(pose + np.array([500.0, 500.0], dtype="float32"), scores)
    corner = normalize_skeleton(pose + np.array([1500.0, 300.0], dtype="float32"), scores)
    assert np.allclose(centre, corner, atol=1e-3)
    # Unit RMS about the per-axis mean, wherever the subject stands.
    rms = float(np.sqrt((corner.reshape(-1, 2) ** 2).mean()))
    assert abs(rms - 1.0) < 1e-3


def test_normalize_skeleton_legacy_reproduces_the_old_scale():
    # Checkpoints trained before the fix must still get the inputs they learned on.
    rng = np.random.default_rng(2)
    kp = rng.uniform(0, 100, size=(4, 2, 17, 2)).astype("float32") + np.array(
        [1500.0, 300.0], dtype="float32"
    )
    scores = np.ones((4, 2, 17), dtype="float32")
    valid = kp.reshape(-1, 2)
    expected = (kp - valid.mean(axis=0)) / (valid.std() + 1e-6)
    assert np.allclose(normalize_skeleton(kp, scores, legacy=True), expected, atol=1e-4)
    # And the legacy scale really is the position-dependent one.
    assert not np.allclose(normalize_skeleton(kp, scores), expected, atol=1e-2)


def test_features_to_tensor_accepts_legacy_normalize_mode():
    rng = np.random.default_rng(3)
    kp = rng.uniform(0, 100, size=(20, 2, 17, 2)).astype("float32") + 800.0
    scores = np.ones((20, 2, 17), dtype="float32")
    fixed = features_to_tensor(kp, scores, 16, normalize=True)
    legacy = features_to_tensor(kp, scores, 16, normalize="legacy")
    assert fixed.shape == legacy.shape == (3, 16, 17, 2)


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
