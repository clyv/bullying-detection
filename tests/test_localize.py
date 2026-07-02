"""Phase 5: sliding-window temporal localization."""

import numpy as np

from src.evaluation.localize import (
    candidate_pairs,
    detection_matches,
    find_incidents,
    localize_stream,
    temporal_iou,
    window_starts,
)


def test_window_starts_basic():
    assert window_starts(100, 64, 16) == [0, 16, 32, 36]  # last flushes right to 100-64
    assert window_starts(64, 64, 16) == [0]
    assert window_starts(40, 64, 16) == [0]  # shorter than a window


def test_window_starts_exact_multiple():
    starts = window_starts(128, 64, 64)
    assert starts == [0, 64]  # no duplicate flush-right window


def test_find_incidents_merges_contiguous_windows():
    starts = [0, 16, 48]
    scores = [0.9, 0.8, 0.7]  # [0,32), [16,48), [48,80) — all overlapping/contiguous
    incidents = find_incidents(scores, starts, window=32, threshold=0.5, max_gap=0)
    assert incidents == [(0, 80, 0.9)]  # one continuous incident, peak score kept


def test_find_incidents_separates_on_real_gap():
    starts = [0, 16, 32, 48, 64]
    scores = [0.9, 0.1, 0.1, 0.1, 0.8]  # aggressive, quiet stretch, aggressive again
    incidents = find_incidents(scores, starts, window=32, threshold=0.5, max_gap=0)
    assert incidents == [(0, 32, 0.9), (64, 96, 0.8)]


def test_find_incidents_threshold_and_gap():
    starts = [0, 40]
    scores = [0.9, 0.9]
    # windows [0,32) and [40,72): gap of 8 frames -> merged only if max_gap >= 8
    assert find_incidents(scores, starts, 32, 0.5, max_gap=0) == [(0, 32, 0.9), (40, 72, 0.9)]
    assert find_incidents(scores, starts, 32, 0.5, max_gap=8) == [(0, 72, 0.9)]


def test_find_incidents_none_above_threshold():
    assert find_incidents([0.1, 0.2], [0, 16], 32, 0.5) == []


def test_temporal_iou():
    assert temporal_iou((0, 10), (0, 10)) == 1.0
    assert temporal_iou((0, 10), (10, 20)) == 0.0
    assert temporal_iou((0, 10), (5, 15)) == 5 / 15  # inter 5, union 15
    assert temporal_iou((0, 0), (0, 0)) == 0.0


def test_detection_matches():
    gt = [(0, 50), (100, 150)]
    pred = [(5, 55), (200, 220)]  # first overlaps gt[0] well, second matches nothing
    assert detection_matches(pred, gt, iou_threshold=0.5) == 1
    assert detection_matches([], gt) == 0


def test_candidate_pairs_keeps_nearby_people_in_crowd():
    T, M = 8, 4
    keypoints = np.zeros((T, M, 17, 2), dtype="float32")
    scores = np.ones((T, M, 17), dtype="float32")
    keypoints[:, 0, :, :] = [10, 10]
    keypoints[:, 1, :, :] = [16, 12]
    keypoints[:, 2, :, :] = [300, 300]
    keypoints[:, 3, :, :] = [306, 302]
    keypoints[:, :, :, 1] += np.linspace(0, 40, 17)

    pairs = candidate_pairs(keypoints, scores, max_pairs=4)

    assert (0, 1) in pairs
    assert (2, 3) in pairs
    assert (0, 2) not in pairs


def test_candidate_pairs_nearest_neighbours_only():
    from src.evaluation.localize import candidate_pairs

    T, M = 8, 4
    kp = np.zeros((T, M, 17, 2), dtype="float32")
    sc = np.ones((T, M, 17), dtype="float32")
    kp[:, :, :, 1] = np.linspace(0, 60, 17)  # ~60px tall skeletons
    kp[:, 0, :, 0] += 100.0
    kp[:, 1, :, 0] += 140.0  # close to person 0 -> pair
    kp[:, 2, :, 0] += 900.0  # far from everyone -> no pair
    sc[:, 3] = 0.0  # person 3 never visible -> ignored

    pairs = candidate_pairs(kp, sc)
    assert (0, 1) in pairs
    assert all(2 not in p and 3 not in p for p in pairs)


def test_candidate_pairs_empty_when_alone():
    from src.evaluation.localize import candidate_pairs

    kp = np.zeros((5, 3, 17, 2), dtype="float32")
    sc = np.zeros((5, 3, 17), dtype="float32")
    sc[:, 0] = 1.0  # only one visible person
    assert candidate_pairs(kp, sc) == []


def test_score_stream_pairs_end_to_end():
    import torch

    from src.evaluation.localize import score_stream_pairs
    from src.models.stgcn import STGCNBaseline

    rng = np.random.default_rng(0)
    T, M = 120, 5
    kp = rng.standard_normal((T, M, 17, 2)).astype("float32") * 20
    kp[:, :, :, 0] += np.array([100, 130, 400, 430, 800])[None, :, None]
    sc = rng.random((T, M, 17)).astype("float32")

    model = STGCNBaseline(in_channels=3, num_classes=2, num_persons=2)
    probs, starts, best_pairs = score_stream_pairs(
        model, kp, sc, torch.device("cpu"), window=64, stride=32
    )
    assert len(probs) == len(starts) == len(best_pairs)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(bp is None or (len(bp) == 2 and bp[0] < bp[1]) for bp in best_pairs)


def test_localize_stream_end_to_end():
    import torch

    from src.models.stgcn import STGCNBaseline

    model = STGCNBaseline(in_channels=3, num_classes=2, num_persons=2)
    rng = np.random.default_rng(0)
    T = 200
    keypoints = rng.standard_normal((T, 2, 17, 2)).astype("float32")
    scores = rng.random((T, 2, 17)).astype("float32")

    incidents = localize_stream(
        model, keypoints, scores, torch.device("cpu"), window=64, stride=32, threshold=0.5
    )
    assert isinstance(incidents, list)
    for start, end, score in incidents:
        assert 0 <= start < end <= T
        assert 0.5 <= score <= 1.0
