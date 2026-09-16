import numpy as np
import torch

from src.datasets.streams import COCO_PARENTS, STREAMS, build_stream, to_bone, to_motion


def _clip(t=8, v=17, m=2):
    torch.manual_seed(0)
    features = torch.randn(3, t, v, m)
    features[2] = torch.rand(t, v, m)  # confidence channel in [0, 1]
    return features


def test_bone_is_offset_from_parent():
    features = _clip()
    bone = to_bone(features)
    # Joint 9 (left wrist) hangs off joint 7 (left elbow).
    expected = features[:2, :, 9] - features[:2, :, 7]
    assert torch.allclose(bone[:2, :, 9], expected)
    # The root's bone is the zero vector by construction (parent of 0 is 0).
    assert torch.allclose(bone[:2, :, 0], torch.zeros_like(bone[:2, :, 0]))


def test_bone_confidence_is_the_weaker_endpoint():
    features = _clip()
    features[2, :, 7] = 0.2  # elbow uncertain
    features[2, :, 9] = 0.9  # wrist confident
    bone = to_bone(features)
    # A bone hanging off an unreliable joint must not inherit the child's score.
    assert torch.allclose(bone[2, :, 9], torch.full_like(bone[2, :, 9], 0.2))


def test_bone_is_translation_invariant():
    # The whole reason bone streams help here: shifting the camera must not change them.
    features = _clip()
    shifted = features.clone()
    shifted[:2] += 37.0
    assert torch.allclose(to_bone(features)[:2], to_bone(shifted)[:2], atol=1e-5)


def test_motion_is_the_temporal_difference():
    features = _clip(t=5)
    motion = to_motion(features)
    assert torch.allclose(motion[:2, 0], features[:2, 1] - features[:2, 0])
    # Final frame has no successor, so it is zeroed to keep the tensor length.
    assert torch.allclose(motion[:2, -1], torch.zeros_like(motion[:2, -1]))
    # Confidence is carried through, not differenced.
    assert torch.allclose(motion[2], features[2])


def test_every_stream_keeps_the_tensor_contract():
    features = _clip()
    for stream in STREAMS:
        out = build_stream(features, stream)
        assert out.shape == features.shape, stream


def test_parents_form_a_tree_rooted_at_the_nose():
    assert len(COCO_PARENTS) == 17
    assert COCO_PARENTS[0] == 0  # root is its own parent
    # Every non-root joint reaches the root by following parents (no cycles).
    for joint in range(1, 17):
        seen, node = set(), joint
        while node != 0:
            assert node not in seen, f"cycle through joint {joint}"
            seen.add(node)
            node = COCO_PARENTS[node]


def test_unknown_stream_is_rejected():
    try:
        build_stream(_clip(), "velocity")
    except ValueError as err:
        assert "velocity" in str(err)
    else:
        raise AssertionError("expected ValueError for an unknown stream")


def test_numpy_free_of_nans():
    features = _clip()
    for stream in STREAMS:
        assert not np.isnan(build_stream(features, stream).numpy()).any()
