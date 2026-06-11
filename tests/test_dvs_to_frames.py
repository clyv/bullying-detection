import numpy as np
import pytest

from src.preprocessing.dvs_to_frames import (
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    accumulate,
    load_events,
    render,
)


def synthetic_events(n=1000, duration_us=100_000.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, duration_us, n))
    x = rng.uniform(0, SENSOR_WIDTH, n)
    y = rng.uniform(0, SENSOR_HEIGHT, n)
    p = rng.choice([-1.0, 1.0], n)
    return np.stack([t, x, y, p], axis=1)


def test_accumulate_conserves_events():
    events = synthetic_events()
    frames = accumulate(events, window_ms=33.0)
    assert frames.shape[1:] == (2, SENSOR_HEIGHT, SENSOR_WIDTH)
    assert frames.sum() == len(events)


def test_accumulate_window_count():
    frames = accumulate(synthetic_events(duration_us=100_000.0), window_ms=25.0)
    assert frames.shape[0] == 4  # 100 ms of events / 25 ms windows


def test_polarity_channels():
    events = np.array(
        [
            [0.0, 10, 10, 1.0],
            [1.0, 10, 10, -1.0],
            [2.0, 10, 10, 0.0],  # zero polarity counts as OFF
        ]
    )
    frames = accumulate(events, window_ms=33.0)
    assert frames.shape[0] == 1
    assert frames[0, 0, 10, 10] == 1.0  # ON channel
    assert frames[0, 1, 10, 10] == 2.0  # OFF channel


def test_out_of_bounds_coordinates_are_clipped():
    events = np.array([[0.0, 9999.0, -5.0, 1.0]])
    frames = accumulate(events)
    assert frames.sum() == 1
    assert frames[0, 0, 0, SENSOR_WIDTH - 1] == 1.0


def test_empty_events():
    frames = accumulate(np.zeros((0, 4)))
    assert frames.shape == (0, 2, SENSOR_HEIGHT, SENSOR_WIDTH)
    assert render(frames).shape == (0, SENSOR_HEIGHT, SENSOR_WIDTH, 3)


def test_render_uint8_rgb():
    images = render(accumulate(synthetic_events()))
    assert images.dtype == np.uint8
    assert images.shape[-1] == 3
    assert images[..., 1].sum() == 0  # green channel unused (ON=red, OFF=blue)
    assert images.max() > 0


def test_load_events_structured_array(tmp_path):
    arr = np.zeros(5, dtype=[("t", "<f8"), ("x", "<i4"), ("y", "<i4"), ("p", "<i1")])
    arr["t"] = np.arange(5)
    arr["x"] = 1
    arr["y"] = 2
    arr["p"] = 1
    path = tmp_path / "events.npy"
    np.save(path, arr)
    events = load_events(path)
    assert events.shape == (5, 4)
    assert events.dtype == np.float64
    assert (events[:, 1] == 1).all()


def test_load_events_plain_array(tmp_path):
    path = tmp_path / "events.npy"
    np.save(path, synthetic_events(n=10))
    assert load_events(path).shape == (10, 4)


def test_load_events_rejects_bad_shape(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((5, 3)))
    with pytest.raises(ValueError):
        load_events(path)
