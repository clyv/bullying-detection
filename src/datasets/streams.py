"""Joint / bone / motion stream derivation for skeleton models.

Standard practice in skeleton action recognition is to train one model per input
representation and sum their softmax scores. It is worth a couple of points on any
benchmark, but it matters more than usual here: bone vectors are differences between
adjacent joints and motion is a temporal difference, so both are **translation
invariant and far less sensitive to absolute pixel scale** than raw coordinates.
That is precisely the axis on which ~80px corridor skeletons differ from ~174px
lab-footage skeletons, i.e. the measured cause of this project's CCTV failure.

All four streams keep the tensor contract the model already expects:
``(C=3, T, V, M)`` with channels (x, y, confidence).
"""

from __future__ import annotations

import torch

STREAMS = ("joint", "bone", "joint_motion", "bone_motion")

# COCO-17 kinematic tree: parent[j] is the joint that j hangs off.
# The nose is the root and its bone is the zero vector by construction.
COCO_PARENTS = (
    0,  # 0  nose (root)
    0,  # 1  left eye
    0,  # 2  right eye
    1,  # 3  left ear
    2,  # 4  right ear
    0,  # 5  left shoulder
    0,  # 6  right shoulder
    5,  # 7  left elbow
    6,  # 8  right elbow
    7,  # 9  left wrist
    8,  # 10 right wrist
    5,  # 11 left hip
    6,  # 12 right hip
    11,  # 13 left knee
    12,  # 14 right knee
    13,  # 15 left ankle
    14,  # 16 right ankle
)


def to_bone(features: torch.Tensor, parents=COCO_PARENTS) -> torch.Tensor:
    """(C, T, V, M) joint coordinates -> bone vectors along the kinematic tree.

    Each joint's xy becomes its offset from its parent. The confidence channel
    becomes ``min(conf_joint, conf_parent)`` — a bone is only as trustworthy as
    its least-certain endpoint, so a bone hanging off a hallucinated joint is
    correctly marked low-confidence rather than inheriting the child's score.
    """
    parent_index = torch.as_tensor(parents, dtype=torch.long, device=features.device)
    gathered = features.index_select(2, parent_index)
    bone = features.clone()
    bone[:2] = features[:2] - gathered[:2]
    bone[2] = torch.minimum(features[2], gathered[2])
    return bone


def to_motion(features: torch.Tensor) -> torch.Tensor:
    """(C, T, V, M) -> per-frame temporal difference of the xy channels.

    ``motion[:, t] = features[:, t + 1] - features[:, t]``, with the final frame
    zeroed so the tensor keeps its length. The confidence channel is carried
    through unchanged rather than differenced — a *change* in confidence is not a
    meaningful quantity, but knowing whether the joint was visible still is.
    """
    motion = torch.zeros_like(features)
    motion[:2, :-1] = features[:2, 1:] - features[:2, :-1]
    motion[2] = features[2]
    return motion


def build_stream(features: torch.Tensor, stream: str = "joint") -> torch.Tensor:
    """Derive one of ``STREAMS`` from the joint-coordinate tensor."""
    if stream == "joint":
        return features
    if stream == "bone":
        return to_bone(features)
    if stream == "joint_motion":
        return to_motion(features)
    if stream == "bone_motion":
        return to_motion(to_bone(features))
    raise ValueError(f"unknown stream {stream!r}; expected one of {STREAMS}")
