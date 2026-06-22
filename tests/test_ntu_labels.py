"""Phase 3: NTU converter writes a unified label/label_name/aggressive flag."""

import numpy as np

from src.preprocessing.ntu_skeleton import (
    ACTION_TO_LABEL,
    ACTION_TO_NAME,
    AGGRESSIVE,
    NTU_RELEVANT,
    RELEVANT,
    convert_file,
)


def test_label_mapping_is_contiguous_and_complete():
    assert len(NTU_RELEVANT) == 18
    assert set(ACTION_TO_LABEL) == RELEVANT
    assert sorted(ACTION_TO_LABEL.values()) == list(range(18))
    # aggressive actions are ordered first -> labels 0..7
    assert all(ACTION_TO_LABEL[a] < 8 for a in AGGRESSIVE)
    assert ACTION_TO_NAME[50] == "punch_slap"


def _write_skeleton(path, action_id):
    joint = "0.5 0.3 2.0 100 100 960.0 540.0 1 0 0 0 2"
    body = "71 0 1 1 1 1 0 0.0 0.0 2\n25\n" + "\n".join([joint] * 25)
    path.write_text("2\n" + ("1\n" + body + "\n") * 2)


def test_convert_file_writes_aggressive_label(tmp_path):
    src = tmp_path / "S001C001P001R001A050.skeleton"  # punch/slap -> aggressive
    _write_skeleton(src, 50)
    out = convert_file(src, tmp_path / "out", mode="color")
    with np.load(out, allow_pickle=True) as data:
        assert data["keypoints"].shape[1:] == (2, 17, 2)
        assert int(data["action"]) == 50
        assert int(data["label"]) == ACTION_TO_LABEL[50]
        assert str(data["label_name"]) == "punch_slap"
        assert bool(data["aggressive"]) is True


def test_convert_file_writes_neutral_label(tmp_path):
    src = tmp_path / "S001C001P001R001A058.skeleton"  # handshake -> neutral
    _write_skeleton(src, 58)
    out = convert_file(src, tmp_path / "out", mode="color")
    with np.load(out, allow_pickle=True) as data:
        assert int(data["label"]) == ACTION_TO_LABEL[58]
        assert bool(data["aggressive"]) is False
