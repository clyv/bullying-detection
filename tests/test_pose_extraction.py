import numpy as np

from src.preprocessing.pose_extraction import assign_slots, extract_clip


def det(x, y):
    return np.full((17, 2), [x, y], dtype=np.float32)


def test_first_frame_fills_free_slots_in_order():
    last = np.full((2, 2), np.nan)
    assert assign_slots([det(10, 10), det(50, 50)], last) == [0, 1]


def test_keeps_identity_when_detection_order_swaps():
    last = np.array([[100.0, 100.0], [500.0, 100.0]])
    assert assign_slots([det(495, 105), det(105, 95)], last) == [1, 0]


def test_new_person_takes_free_slot():
    last = np.array([[100.0, 100.0], [np.nan, np.nan]])
    # detection 1 is near the known slot 0; detection 0 is a newcomer
    assert assign_slots([det(300, 300), det(102, 99)], last) == [1, 0]


def test_single_detection_keeps_nearest_slot():
    last = np.array([[100.0, 100.0], [500.0, 100.0]])
    assert assign_slots([det(480, 110)], last) == [1]


# --- extract_clip with a fake ultralytics model (no torch needed) ---


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = FakeTensor(xy)
        self.conf = FakeTensor(conf)

    def __len__(self):
        return len(self.xy.array)


class FakeBoxes:
    def __init__(self, conf):
        self.conf = FakeTensor(conf)


class FakeResult:
    def __init__(self, xy, kp_conf, box_conf):
        self.keypoints = FakeKeypoints(xy, kp_conf)
        self.boxes = FakeBoxes(box_conf)


class EmptyResult:
    keypoints = None
    boxes = None


class FakeModel:
    def __init__(self, results):
        self.results = results

    def predict(self, **kwargs):
        yield from self.results


def test_extract_clip_keeps_slots_consistent_across_swap():
    person_a = det(100, 100)
    person_b = det(500, 100)
    conf = np.ones((17,), dtype=np.float32)
    results = [
        FakeResult(np.stack([person_a, person_b]), np.stack([conf, conf]), np.array([0.9, 0.8])),
        # detection order swapped in the second frame
        FakeResult(np.stack([person_b, person_a]), np.stack([conf, conf]), np.array([0.9, 0.8])),
    ]
    keypoints, scores = extract_clip(FakeModel(results), source="fake", max_persons=2)
    assert keypoints.shape == (2, 2, 17, 2)
    assert scores.shape == (2, 2, 17)
    assert np.allclose(keypoints[0, 0], person_a) and np.allclose(keypoints[1, 0], person_a)
    assert np.allclose(keypoints[0, 1], person_b) and np.allclose(keypoints[1, 1], person_b)


def test_extract_clip_caps_at_max_persons():
    people = np.stack([det(100, 100), det(300, 100), det(500, 100)])
    conf = np.ones((3, 17), dtype=np.float32)
    results = [FakeResult(people, conf, np.array([0.5, 0.9, 0.7]))]
    keypoints, scores = extract_clip(FakeModel(results), source="fake", max_persons=2)
    assert keypoints.shape == (1, 2, 17, 2)
    # the two highest-confidence detections survive (indices 1 and 2)
    kept_x = sorted(keypoints[0, :, 0, 0].tolist())
    assert kept_x == [300.0, 500.0]
    assert scores.max() == 1.0


def test_extract_clip_handles_empty_frames():
    keypoints, scores = extract_clip(FakeModel([EmptyResult()]), source="fake", max_persons=2)
    assert keypoints.shape == (1, 2, 17, 2)
    assert keypoints.sum() == 0
    assert scores.sum() == 0


def test_extract_clip_no_frames():
    keypoints, scores = extract_clip(FakeModel([]), source="fake", max_persons=2)
    assert keypoints.shape == (0, 2, 17, 2)
    assert scores.shape == (0, 2, 17)
