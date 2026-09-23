"""Tests for the batch ledger, the aggregate and the annotation stub.

Pose extraction needs a real video and a GPU, so these drive the parts that turn
scores into claims: whether an unlabelled video is kept out of the statistics,
whether a lead time is only credited before onset, and whether a stub is marked
as unreviewed.
"""

import json

import numpy as np
import pytest

from src.early_warning.escalation_policy import PolicyConfig
from src.early_warning.run_batch import (
    aggregate,
    assault_intervals,
    case_template,
    find_videos,
    shortlist,
)

FPS = 10.0
CFG = PolicyConfig()


def _scores(tmp_path, video_id, series):
    d = tmp_path / "scores"
    d.mkdir(exist_ok=True)
    arr = np.asarray(series, dtype=np.float32)
    np.savez_compressed(
        d / f"{video_id}.npz", hazard_5s=arr, hazard_10s=arr, quality=np.ones_like(arr), fps=FPS
    )


def _row(video_id, **kw):
    base = dict(video_id=video_id, usable=True, fps=FPS, mean_quality=1.0, label_source="fight")
    return base | kw


# ------------------------------------------------------------- annotations


def test_missing_annotation_is_none_not_an_empty_list(tmp_path):
    from pathlib import Path

    video = Path("nowhere/clip.mp4")
    assert assault_intervals(video, None) is None
    assert assault_intervals(video, tmp_path) is None  # directory exists, file does not

    (tmp_path / "clip.csv").write_text("0\n0\n0\n")
    assert assault_intervals(video, tmp_path) == []  # annotated, and it says nothing happened


def test_annotation_becomes_inclusive_frame_spans(tmp_path):
    from pathlib import Path

    (tmp_path / "clip.csv").write_text("\n".join("0011100"))
    assert assault_intervals(Path("x/clip.mp4"), tmp_path) == [(2, 4)]


def test_find_videos_ignores_non_video_files(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.avi").touch()
    assert [p.name for p in find_videos(tmp_path)] == ["a.mp4", "b.avi"]


# --------------------------------------------------------------- aggregate


def test_unlabelled_videos_are_excluded_from_every_statistic(tmp_path):
    # The bug this guards: a video with no annotation counted as a clean
    # negative turns each of its real interactions into a false alarm.
    quiet = np.zeros(200)
    loud = np.concatenate([np.zeros(100), np.ones(100)])
    for vid, series in (("pos", loud), ("neg", quiet), ("unknown", loud)):
        _scores(tmp_path, vid, series)
    rows = [
        _row("pos", has_assault=True, onset_s=15.0),
        _row("neg", has_assault=False),
        _row("unknown", has_assault=None),
    ]

    out = aggregate(rows, tmp_path, budget_per_hour=0.15, cfg=CFG)
    assert (out["with_assault"], out["without_assault"], out["unlabelled"]) == (1, 1, 1)
    assert out["false_warn_per_hour"] == 0.0  # the unlabelled loud video is not counted


def test_aggregate_without_any_labels_reports_nothing_but_the_count(tmp_path):
    _scores(tmp_path, "a", np.ones(50))
    out = aggregate([_row("a", has_assault=None)], tmp_path, 0.15, CFG)
    assert out == {"videos": 1, "unlabelled": 1}


def test_an_alarm_after_onset_earns_no_lead_time(tmp_path):
    # Firing at the first blow is detection, not anticipation.
    late = np.concatenate([np.zeros(100), np.ones(100)])
    _scores(tmp_path, "late", late)
    _scores(tmp_path, "quiet", np.zeros(200))
    rows = [_row("late", has_assault=True, onset_s=10.0), _row("quiet", has_assault=False)]

    out = aggregate(rows, tmp_path, 0.15, CFG)
    assert out["anticipated"] == 0
    assert out["warn_lead_s"]["n"] == 0


def test_an_alarm_before_onset_earns_its_lead(tmp_path):
    early = np.concatenate([np.zeros(50), np.ones(150)])  # crosses at 5.0s
    _scores(tmp_path, "early", early)
    _scores(tmp_path, "quiet", np.zeros(200))
    rows = [_row("early", has_assault=True, onset_s=15.0), _row("quiet", has_assault=False)]

    out = aggregate(rows, tmp_path, 0.15, CFG)
    assert out["anticipated"] == 1
    assert out["warn_lead_s"]["median"] == pytest.approx(10.0, abs=0.2)


# --------------------------------------------------------- review artefacts


def test_shortlist_ranks_by_lead_and_skips_videos_that_never_fired():
    rows = [
        _row("short", onset_s=20.0, warn_lead_s=1.5, reasons_at_alarm={"contact": 1.0}),
        _row("long", onset_s=30.0, warn_lead_s=8.0, reasons_at_alarm={}),
        _row("silent", onset_s=10.0, warn_lead_s=None),
    ]
    got = shortlist(rows)
    assert [r["video_id"] for r in got] == ["long", "short"]
    assert got[0]["watch_at_s"] == 22.0  # onset 30 minus an 8s lead


def test_case_template_is_marked_unreviewed_and_flags_its_guesses(tmp_path):
    row = _row(
        "clip",
        steps=400,
        onset_s=20.0,
        warn_lead_s=6.0,
        assault_spans_s=[[20.0, 25.0]],
        reasons_at_alarm={"target enclosed by others": 0.8},
    )
    (tmp_path / "ledger.jsonl").write_text(json.dumps(row) + "\n")

    stub = case_template("clip", tmp_path)
    assert stub["phase_annotated"] is False  # never ground truth until a human says so
    assert stub["assault"] == [[200, 250]]
    assert [p[2] for p in stub["phases"]] == [1, 2]  # precursor then build-up
    assert stub["phases"][1] == [140, 200, 2]  # build-up spans the alarm to the onset
    assert stub["_guessed"]["warn_lead_s"] == 6.0


def test_case_template_refuses_an_unknown_video(tmp_path):
    (tmp_path / "ledger.jsonl").write_text(json.dumps(_row("other")) + "\n")
    with pytest.raises(KeyError):
        case_template("clip", tmp_path)
