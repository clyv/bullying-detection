"""Shared aggressive-vs-neutral label space for cross-dataset (Phase 4) work.

Each source dataset has its own action taxonomy that doesn't line up with the
others, so the unified model collapses them onto the binary axis the project is
actually about: aggressive (physical/confrontational) vs. neutral interaction.

The mapping is deliberately explicit and auditable here. Some assignments are
judgement calls — NTU's "subtle" bullying-relevant actions (point, follow,
whisper, shake fist) are treated as aggressive because they are exactly the kind
of confrontational behaviour the system is meant to flag; revisit if you want a
stricter physical-only definition.
"""

from __future__ import annotations

import os

NEUTRAL, AGGRESSIVE = 0, 1
BINARY_NAMES = ["neutral", "aggressive"]

# UT-Interaction (pose_extraction.py output carries no label, so infer by name).
UT_AGGRESSIVE = {"kick", "punch", "push"}
UT_NEUTRAL = {"handshake", "hug", "point"}
# UT clips are usually named numerically (e.g. "0_11_4"), where the trailing
# field is the class id: 0 handshake, 1 hug, 2 kick, 3 point, 4 punch, 5 push.
UT_AGGRESSIVE_IDX = {2, 4, 5}
UT_NEUTRAL_IDX = {0, 1, 3}


def ut_interaction_aggressive(name: str) -> int | None:
    """Binary label for a UT-Interaction clip filename, or None if unrecognised."""
    stem = os.path.basename(name).lower().replace(".npz", "")
    if any(k in stem for k in UT_AGGRESSIVE):
        return AGGRESSIVE
    if any(k in stem for k in UT_NEUTRAL):
        return NEUTRAL
    # numeric fallback: trailing field is the UT class id (0-5)
    try:
        idx = int(stem.split("_")[-1])
    except ValueError:
        return None
    if idx in UT_AGGRESSIVE_IDX:
        return AGGRESSIVE
    if idx in UT_NEUTRAL_IDX:
        return NEUTRAL
    return None


def binary_label(data, source_path: str = "", dataset: str = "") -> int | None:
    """Collapse a sample's native annotation to NEUTRAL/AGGRESSIVE.

    ``data`` is a loaded .npz (mapping-like). Bullying10K and NTU converters
    already store an explicit ``aggressive`` flag; UT-Interaction is inferred
    from the filename. Returns None when aggression can't be determined.
    """
    if "aggressive" in data:
        return AGGRESSIVE if bool(data["aggressive"]) else NEUTRAL
    if dataset == "ut_interaction" or not dataset:
        return ut_interaction_aggressive(source_path)
    return None
