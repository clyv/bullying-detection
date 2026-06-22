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

NEUTRAL, AGGRESSIVE = 0, 1
BINARY_NAMES = ["neutral", "aggressive"]

# UT-Interaction (pose_extraction.py output carries no label, so infer by name).
UT_AGGRESSIVE = {"kick", "punch", "push"}
UT_NEUTRAL = {"handshake", "hug", "point"}


def ut_interaction_aggressive(name: str) -> int | None:
    """Binary label for a UT-Interaction clip filename, or None if unrecognised."""
    stem = name.lower()
    if any(k in stem for k in UT_AGGRESSIVE):
        return AGGRESSIVE
    if any(k in stem for k in UT_NEUTRAL):
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
