import numpy as np

from src.evaluation.evaluate import (
    accuracy,
    confusion_matrix,
    format_report,
    latest_checkpoint,
    per_class_precision_recall,
)


def test_format_report_uses_supplied_class_names():
    cm = confusion_matrix(np.array([0, 1]), np.array([0, 1]), num_classes=2)
    report = format_report(cm, 1.0, class_names=["slapping", "walking"])
    assert "slapping" in report and "walking" in report
    assert "Accuracy: 100.00%" in report


def test_accuracy():
    assert accuracy(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1])) == 0.75
    assert accuracy(np.array([]), np.array([])) == 0.0


def test_confusion_matrix_shape_and_counts():
    preds = np.array([0, 1, 1, 2, 2, 2])
    targets = np.array([0, 1, 2, 2, 2, 0])
    cm = confusion_matrix(preds, targets, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm.sum() == 6
    assert cm[0, 0] == 1  # one true-0 predicted 0
    assert cm[2, 2] == 2  # two true-2 predicted 2
    assert cm[2, 1] == 1  # one true-2 predicted 1


def test_confusion_matrix_ignores_out_of_range():
    cm = confusion_matrix(np.array([5]), np.array([0]), num_classes=3)
    assert cm.sum() == 0


def test_per_class_precision_recall():
    # class 0: 2 correct; class 1: 1 correct, 1 predicted-as-0
    cm = np.array([[2, 0], [1, 1]])
    precision, recall = per_class_precision_recall(cm)
    assert np.isclose(precision[0], 2 / 3)  # 2 true-0 of 3 predicted-0
    assert np.isclose(recall[0], 1.0)  # both true-0 found
    assert np.isclose(recall[1], 0.5)  # 1 of 2 true-1 found


def test_per_class_handles_empty_classes():
    precision, recall = per_class_precision_recall(np.zeros((3, 3), dtype=int))
    assert (precision == 0).all() and (recall == 0).all()


def test_latest_checkpoint(tmp_path):
    assert latest_checkpoint(str(tmp_path)) is None
    import time

    older = tmp_path / "epoch_10.pt"
    older.write_bytes(b"x")
    time.sleep(0.01)
    newer = tmp_path / "epoch_20.pt"
    newer.write_bytes(b"x")
    assert latest_checkpoint(str(tmp_path)) == str(newer)


def test_evaluate_end_to_end(tmp_path):
    """Train-free smoke: save a checkpoint, then evaluate it on synthetic clips."""
    import yaml

    torch = __import__("torch")
    from src.models.stgcn import STGCNBaseline

    cache = tmp_path / "poses"
    cache.mkdir()
    rng = np.random.default_rng(0)
    classes = ["handshake", "hug", "kick", "point", "punch", "push"]
    for i in range(12):
        T = int(rng.integers(40, 80))
        np.savez(
            cache / f"clip{i:02d}_{classes[i % 6]}.npz",
            keypoints=rng.standard_normal((T, 2, 17, 2)).astype("float32"),
            scores=rng.random((T, 2, 17)).astype("float32"),
        )

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    model = STGCNBaseline(in_channels=3, num_classes=6, num_persons=2)
    ckpt = ckpt_dir / "stgcn_baseline_epoch_1.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)

    cfg = {
        "data": {"pose_cache": str(cache), "num_frames": 64, "max_persons": 2},
        "model": {"num_classes": 6, "in_channels": 3},
        "training": {"batch_size": 4},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from src.evaluation.evaluate import evaluate

    result = evaluate(config_path=str(cfg_path), checkpoint=str(ckpt))
    assert result is not None
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["confusion_matrix"].shape == (6, 6)
    assert result["confusion_matrix"].sum() == 12  # every clip classified
