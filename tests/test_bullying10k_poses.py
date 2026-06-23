import json

import numpy as np
import pytest

from src.preprocessing.bullying10k_poses import (
    AGGRESSIVE,
    BULLYING10K_CLASSES,
    coco_to_unified,
    convert_file,
    convert_keypoints_json,
    keypoints_to_coco17,
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


def test_keypoints_to_coco17_takes_first_17_of_halpe26():
    halpe = list(range(26 * 3))  # 78 values
    coco = keypoints_to_coco17(halpe)
    assert coco.shape == (17, 3)
    assert coco[0].tolist() == [0.0, 1.0, 2.0]  # first joint preserved
    assert coco[16].tolist() == [48.0, 49.0, 50.0]  # 17th joint = vals 48-50


def _coco_json(tmp_path):
    """Two clips (punching=aggressive, handshake=neutral), 2 frames, 2 people each, Halpe-26."""
    images, annotations = [], []
    img_id = ann_id = 0
    for clip in ("punching/punching_a/dvSave-1", "handshake/handshake_b/dvSave-2"):
        for frame in range(2):
            images.append(
                {"file_name": f"{clip}/{frame}.png", "height": 260, "width": 346, "id": img_id}
            )
            for person in range(2):
                kp = [10 + person, 20, 2] * 26  # 26 Halpe joints, visible
                annotations.append(
                    {
                        "image_id": img_id,
                        "category_id": 1,
                        "keypoints": kp,
                        "score": 1.0 - person * 0.1,
                    }
                )
                ann_id += 1
            img_id += 1
    path = tmp_path / "train_keypoints.json"
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "person"}],
            }
        )
    )
    return path


def test_convert_keypoints_json_groups_clips_and_labels(tmp_path):
    json_path = _coco_json(tmp_path)
    out = tmp_path / "out"
    n = convert_keypoints_json(json_path, out, max_persons=2)
    assert n == 2

    files = sorted(out.glob("*.npz"))
    assert len(files) == 2
    by_label = {}
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            assert d["keypoints"].shape == (2, 2, 17, 2)  # 2 frames, 2 persons
            assert d["scores"].shape == (2, 2, 17)
            by_label[str(d["label_name"])] = bool(d["aggressive"])
    assert by_label == {"punching": True, "handshake": False}


def test_convert_keypoints_json_limit(tmp_path):
    json_path = _coco_json(tmp_path)
    assert convert_keypoints_json(json_path, tmp_path / "out", limit=1) == 1
