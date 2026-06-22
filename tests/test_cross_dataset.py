"""Phase 4: multi-dataset binary loader + cross-dataset evaluation."""

import numpy as np

from src.datasets.taxonomy import AGGRESSIVE, NEUTRAL
from src.datasets.unified_loader import MultiDatasetSkeletonDataset


def _write_clip(path, aggressive=None, label_name=None):
    fields = dict(
        keypoints=np.random.randn(30, 2, 17, 2).astype("float32"),
        scores=np.random.rand(30, 2, 17).astype("float32"),
    )
    if aggressive is not None:
        fields["aggressive"] = aggressive
    if label_name is not None:
        fields["label_name"] = label_name
    np.savez(path, **fields)


def test_multidataset_binary_labels_and_tags(tmp_path):
    b10k = tmp_path / "b10k"
    b10k.mkdir()
    _write_clip(b10k / "punching_01.npz", aggressive=True)
    _write_clip(b10k / "walking_02.npz", aggressive=False)

    ut = tmp_path / "ut"
    ut.mkdir()
    _write_clip(ut / "seq_kick_01.npz")  # inferred aggressive from filename
    _write_clip(ut / "seq_hug_02.npz")  # inferred neutral
    _write_clip(ut / "no_known_class.npz")  # undeterminable -> skipped

    ds = MultiDatasetSkeletonDataset(
        [("bullying10k", str(b10k)), ("ut_interaction", str(ut))], target_frames=16
    )
    assert len(ds) == 4  # the unrecognised UT clip is dropped
    assert ds.datasets.count("bullying10k") == 2
    assert ds.datasets.count("ut_interaction") == 2

    labels = sorted(int(ds[i][1]) for i in range(len(ds)))
    assert labels == [NEUTRAL, NEUTRAL, AGGRESSIVE, AGGRESSIVE]

    tensor, _ = ds[0]
    assert tensor.shape == (3, 16, 17, 2)


def test_multidataset_skips_missing_cache(tmp_path):
    ds = MultiDatasetSkeletonDataset([("ntu", str(tmp_path / "does_not_exist"))], target_frames=8)
    assert len(ds) == 0


def test_cross_dataset_end_to_end(tmp_path):
    """Pooled train + leave-one-out run to completion on tiny synthetic caches."""
    rng = np.random.default_rng(0)

    def make_cache(name, aggressive_flags):
        d = tmp_path / name
        d.mkdir()
        for i, agg in enumerate(aggressive_flags):
            np.savez(
                d / f"{name}_{i:02d}.npz",
                keypoints=rng.standard_normal((25, 2, 17, 2)).astype("float32"),
                scores=rng.random((25, 2, 17)).astype("float32"),
                aggressive=bool(agg),
            )
        return str(d)

    cfg = {
        "data": {
            "num_frames": 32,
            "max_persons": 2,
            "datasets": [
                {"name": "ds_a", "cache": make_cache("ds_a", [True, False, True, False])},
                {"name": "ds_b", "cache": make_cache("ds_b", [True, False, True, False])},
            ],
        },
        "model": {"in_channels": 3},
        "training": {"batch_size": 2, "epochs": 1, "lr": 0.01, "weight_decay": 0.0001},
    }

    import torch

    from src.evaluation.cross_dataset import leave_one_out, pooled_evaluation

    device = torch.device("cpu")
    pooled = pooled_evaluation(cfg, device)
    assert pooled is not None
    acc, cm = pooled
    assert cm.shape == (2, 2)

    loo = leave_one_out(cfg, device)
    assert set(loo) == {"ds_a", "ds_b"}
    for _, (acc, cm) in loo.items():
        assert cm.shape == (2, 2)
        assert cm.sum() == 4  # all four held-out clips classified
