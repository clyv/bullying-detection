import numpy as np

from src.evaluation.localize import find_incidents
from src.evaluation.visualize import frame_scores_from_windows


def test_frame_scores_single_window_broadcasts():
    fs = frame_scores_from_windows([0.8], starts=[0], window=4, num_frames=4)
    assert fs.shape == (4,)
    assert np.allclose(fs, 0.8)


def test_frame_scores_averages_overlapping_windows():
    # windows [0,4) score 1.0 and [2,6) score 0.0 overlap on frames 2,3
    fs = frame_scores_from_windows([1.0, 0.0], starts=[0, 2], window=4, num_frames=6)
    assert np.allclose(fs[:2], 1.0)  # only first window
    assert np.allclose(fs[2:4], 0.5)  # averaged overlap
    assert np.allclose(fs[4:], 0.0)  # only second window


def test_frame_scores_uncovered_frames_are_zero():
    fs = frame_scores_from_windows([1.0], starts=[0], window=2, num_frames=5)
    assert np.allclose(fs[:2], 1.0)
    assert np.allclose(fs[2:], 0.0)  # no window covers these -> 0, not div-by-zero


def test_per_frame_incidents_via_find_incidents():
    # a per-frame curve thresholded into incidents (window=1 trick)
    fs = np.array([0.1, 0.9, 0.95, 0.2, 0.8, 0.85])
    incidents = find_incidents(fs, range(len(fs)), window=1, threshold=0.5, max_gap=0)
    assert incidents == [(1, 3, 0.95), (4, 6, 0.85)]
