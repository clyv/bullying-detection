import numpy as np
import pytest

from src.preprocessing.bullying10k_poses import (
    AGGRESSIVE,
    BULLYING10K_CLASSES,
    coco_to_unified,
    convert_file,
    label_for,
)


def test_class_list():
    assert len(BULLYING10K_CLASSES) == 10
    assert len(set(BULLYING10K_CLASSES)) == 10
    assert AGGRESSIVE == set(BULLYING10K_CLASSES[:6])


def test_label_for_keywords_and_aliases():
    assert label_for("clip_punching_001") == BULLYING10K_CLASSES.index("punching")
    assert label_for("S1_slap_02") == BULLYING10K_CLASSES.index("slapping")  # alias
    assert label_for("hair_grab_07") == BULLYING10K_CLASSES.index("hair_grab")
    assert label_for("finger_guessing_03") == BULLYING10K_CLASSES.index("finger_guessing")
    assert label_for("totally_unknown_action") is None


def test_coco_to_unified_shapes_and_visibility():
    T, M = 5, 2
    kp = np.zeros((T, M, 17, 3), dtype=np.float32)
    kp[..., 0] = 100.0  # x
    kp[..., 1] = 50.0  # y
    kp[..., 2] = 2.0  # fully visible
    kp[:, :, 0, 2] = 0.0  # joint 0 unlabelled
    coords, scores = coco_to_unified(kp, max_persons=2)
    assert coords.shape == (T, M, 17, 2)
    assert scores.shape == (T, M, 17)
    assert (scores[:, :, 1:] == 1.0).all()  # v=2 -> 1.0
    assert (scores[:, :, 0] == 0.0).all()  # unlabelled -> 0
    assert (coords[:, :, 0] == 0.0).all()  # unlabelled coords zeroed
    assert (coords[:, :, 1, 0] == 100.0).all()


def test_coco_to_unified_accepts_flattened():
    kp = np.ones((4, 2, 51), dtype=np.float32)
    coords, scores = coco_to_unified(kp)
    assert coords.shape == (4, 2, 17, 2)
    assert np.isclose(scores[0, 0, 0], 0.5)  # v=1 -> 0.5


def test_coco_to_unified_pads_and_truncates_persons():
    coords, scores = coco_to_unified(np.ones((3, 4, 17, 3), dtype=np.float32), max_persons=2)
    assert coords.shape == (3, 2, 17, 2)  # 4 people -> truncated to 2

    coords, scores = coco_to_unified(np.ones((3, 1, 17, 3), dtype=np.float32), max_persons=2)
    assert coords.shape == (3, 2, 17, 2)
    assert (scores[:, 1] == 0.0).all()  # second slot padded empty


def test_coco_to_unified_rejects_bad_shape():
    with pytest.raises(ValueError):
        coco_to_unified(np.ones((3, 2, 17, 2)))


def test_convert_file_writes_unified_npz_with_label(tmp_path):
    src = tmp_path / "subject1_kicking_05.npy"
    np.save(src, np.ones((6, 2, 17, 3), dtype=np.float32))
    out = convert_file(src, tmp_path / "out")
    with np.load(out, allow_pickle=True) as data:
        assert data["keypoints"].shape == (6, 2, 17, 2)
        assert data["scores"].shape == (6, 2, 17)
        assert int(data["label"]) == BULLYING10K_CLASSES.index("kicking")
        assert str(data["label_name"]) == "kicking"
        assert bool(data["aggressive"]) is True
