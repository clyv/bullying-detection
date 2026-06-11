import numpy as np

from src.preprocessing.ntu_skeleton import (
    AGGRESSIVE,
    NEUTRAL,
    RELEVANT,
    SUBTLE,
    parse_name,
    parse_skeleton_file,
    to_sequence,
)


def joint_line(x, y, z, cx, cy):
    # columns: x y z depthX depthY colorX colorY orientW orientX orientY orientZ trackingState
    return f"{x} {y} {z} 100 100 {cx} {cy} 1 0 0 0 2"


def skeleton_text(frame_bodies):
    """frame_bodies: per-frame lists of (body_id, x, y, z, color_x, color_y) tuples."""
    lines = [str(len(frame_bodies))]
    for bodies in frame_bodies:
        lines.append(str(len(bodies)))
        for body_id, x, y, z, cx, cy in bodies:
            lines.append(f"{body_id} 0 1 1 1 1 0 0.0 0.0 2")
            lines.append("25")
            lines.extend([joint_line(x, y, z, cx, cy)] * 25)
    return "\n".join(lines) + "\n"


def write_skeleton(tmp_path, name, frame_bodies):
    path = tmp_path / f"{name}.skeleton"
    path.write_text(skeleton_text(frame_bodies))
    return path


def test_parse_name():
    meta = parse_name("S001C002P003R002A050")
    assert meta == {"setup": 1, "camera": 2, "performer": 3, "replication": 2, "action": 50}


def test_class_subsets():
    assert AGGRESSIVE & SUBTLE == set()
    assert AGGRESSIVE & NEUTRAL == set()
    assert SUBTLE & NEUTRAL == set()
    assert RELEVANT == AGGRESSIVE | SUBTLE | NEUTRAL


def test_parse_and_sequence_color_mode(tmp_path):
    path = write_skeleton(
        tmp_path, "S001C001P001R001A050", [[("71", 0.5, 0.3, 2.0, 960.0, 540.0)]] * 3
    )
    frames = parse_skeleton_file(path)
    assert len(frames) == 3
    assert frames[0][0]["joints"].shape == (25, 12)

    keypoints, scores = to_sequence(frames)
    assert keypoints.shape == (3, 2, 17, 2)
    assert scores.shape == (3, 2, 17)
    assert np.allclose(keypoints[:, 0], [960.0, 540.0])
    assert scores[:, 0, 0].all()  # nose tracked
    assert not scores[:, 0, 1:5].any()  # eyes/ears zeroed — no Kinect equivalent
    assert not scores[:, 1].any()  # second person slot empty


def test_project_mode_optical_axis_hits_principal_point(tmp_path):
    path = write_skeleton(tmp_path, "S001C001P001R001A055", [[("71", 0.0, 0.0, 2.0, 0.0, 0.0)]])
    keypoints, _ = to_sequence(
        parse_skeleton_file(path), mode="project", fx=1000.0, fy=1000.0, cx=960.0, cy=540.0
    )
    assert np.allclose(keypoints[0, 0, 0], [960.0, 540.0])


def test_project_mode_y_up_maps_to_image_v_down(tmp_path):
    # a joint above the optical axis (y > 0) must land above the principal point (v < cy)
    path = write_skeleton(tmp_path, "S001C001P001R001A055", [[("71", 0.0, 0.5, 2.0, 0.0, 0.0)]])
    keypoints, _ = to_sequence(
        parse_skeleton_file(path), mode="project", fx=1000.0, fy=1000.0, cx=960.0, cy=540.0
    )
    assert keypoints[0, 0, 0, 1] < 540.0


def test_ghost_body_filtering(tmp_path):
    # a moving body must outrank a static ghost when only one slot is kept
    frame_bodies = [
        [
            ("ghost", 0.0, 0.0, 2.0, 100.0, 100.0),
            ("real", 0.1 * t, 0.0, 2.0, 500.0 + 50.0 * t, 300.0),
        ]
        for t in range(4)
    ]
    path = write_skeleton(tmp_path, "S001C001P001R001A051", frame_bodies)
    keypoints, scores = to_sequence(parse_skeleton_file(path), max_bodies=1)
    assert keypoints.shape == (4, 1, 17, 2)
    assert keypoints[0, 0, 0, 0] == 500.0  # the mover, not the ghost
    assert scores[:, 0, 0].all()
