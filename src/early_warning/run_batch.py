"""Score a folder of videos and keep a ledger of what the early warning claimed.

Two modes, matching the two halves of the job:

**Batch.** Extract tracked poses and hazard series for every video under a
directory, append one row per video to a JSONL ledger, and print the aggregate:
lead-time distribution on videos that contain an assault, false alarms per hour
on the ones that don't, and anticipation recall at an alarm budget. Resumable —
a video whose pose file already exists is skipped, so an interrupted run costs
nothing.

**Case file.** Turn one video into a pre-filled annotation stub, with the
system's own claim written into it, for a human to correct. Corrected stubs are
exactly the `VideoEvents` the hazard model trains on, so reviewing hits and
building the training set are the same activity.

    python -m src.early_warning.run_batch --videos data/ubi_fights/UBI_FIGHTS/videos \
        --annotations data/ubi_fights/UBI_FIGHTS/annotation --out outputs/ew_batch --limit 40

    python -m src.early_warning.run_batch --out outputs/ew_batch --case F_0_1_0_0_0

A single video that fires before an assault proves nothing on its own: with a
hazard this correlated with "two people are close", some of them are bound to
land. Read the aggregate, and read the negatives first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.early_warning.anticipation_labels import intervals_from_mask
from src.early_warning.baseline import HeuristicHazard
from src.early_warning.escalation_policy import PolicyConfig, top_reasons
from src.early_warning.metrics import (
    anticipation_at_budget,
    false_alarms_per_hour,
    first_alarm_frame,
    lead_time_distribution,
    lead_times,
)
from src.early_warning.social_features import FeatureConfig, compute, observation_quality

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")


def find_videos(root: str | Path, pattern: str = "*") -> list[Path]:
    """Every video under `root`, sorted, deepest paths included."""
    root = Path(root)
    hits = [p for p in sorted(root.rglob(pattern)) if p.suffix.lower() in VIDEO_SUFFIXES]
    return hits


def assault_intervals(
    video: Path, annotation_dir: str | Path | None
) -> list[tuple[int, int]] | None:
    """Inclusive source-frame assault spans from a matching per-frame CSV.

    Returns None when no annotation exists for this video, and [] when the
    annotation exists and says no assault. The difference decides whether a
    quiet score is evidence of a clean negative or of nothing at all; collapsing
    the two silently turns unlabelled footage into false alarms.
    """
    if annotation_dir is None:
        return None
    csv = Path(annotation_dir) / f"{video.stem}.csv"
    if not csv.exists():
        return None
    mask = np.loadtxt(csv, delimiter=",", ndmin=1).ravel() > 0.5
    return intervals_from_mask(mask)


def score_video(
    video: Path,
    out_dir: Path,
    annotation_dir: str | Path | None,
    process_fps: float,
    max_people: int,
    cfg: PolicyConfig,
) -> dict:
    """Extract (or reuse) poses, score them, and return one ledger row."""
    from src.early_warning.tracked_pose import extract

    poses_path = out_dir / "poses" / f"{video.stem}.npz"
    if not poses_path.exists():
        poses_path.parent.mkdir(parents=True, exist_ok=True)
        data = extract(str(video), max_people=max_people, process_fps=process_fps)
        np.savez_compressed(poses_path, **data)

    with np.load(poses_path) as d:
        kp, sc = d["keypoints"], d["scores"]
        fps = float(d["fps"])
        source_fps = float(d["source_fps"]) if "source_fps" in d else fps
        n_tracks = int(d["n_tracks"]) if "n_tracks" in d else -1

    row = dict(
        video=str(video),
        video_id=video.stem,
        label_source=video.parent.name,
        fps=fps,
        source_fps=source_fps,
        steps=int(len(kp)),
        duration_s=round(len(kp) / fps, 1) if fps else 0.0,
        n_tracks=n_tracks,
    )
    if len(kp) == 0:
        return row | dict(usable=False, reason="no frames decoded")

    bundle = compute(kp, sc, FeatureConfig(fps=fps))
    quality = observation_quality(bundle.geometry)
    hazard = HeuristicHazard(fps=fps)(bundle.signals)
    h5, h10 = hazard["hazard_5s"], hazard["hazard_10s"]

    # Source-frame annotations to replay steps.
    scale = fps / source_fps if source_fps else 1.0
    labelled = assault_intervals(video, annotation_dir)
    spans = [
        (a, b)
        for a, b in (
            (int(a * scale), min(int(b * scale), len(kp) - 1)) for a, b in (labelled or [])
        )
        if a < len(kp)
    ]
    onset = spans[0][0] if spans else None

    np.savez_compressed(
        out_dir / "scores" / f"{video.stem}.npz",
        hazard_5s=h5.astype(np.float32),
        hazard_10s=h10.astype(np.float32),
        quality=quality.astype(np.float32),
        fps=np.float32(fps),
    )

    row |= dict(
        usable=True,
        mean_quality=round(float(quality.mean()), 3),
        peak_5s=round(float(h5.max()), 3),
        peak_10s=round(float(h10.max()), 3),
        # Share of the clip already above WARN. A long lead time on a video whose
        # hazard is high throughout is not anticipation, it is a light left on,
        # and the peak alone cannot tell the two apart.
        frac_above_warn=round(float((h5 >= cfg.warn_enter).mean()), 3),
        # None = nobody has said what is in this video. Only True/False count.
        has_assault=None if labelled is None else onset is not None,
        assault_spans_s=[[round(a / fps, 1), round(b / fps, 1)] for a, b in spans],
    )

    if onset is None:
        # A video with no annotated assault is where false alarms are counted.
        row["first_warn_s"] = _crossing_s(h5, cfg.warn_enter, fps)
        row["first_watch_s"] = _crossing_s(h10, cfg.watch_enter, fps)
        return row

    row["onset_s"] = round(onset / fps, 1)
    for key, series, thr in (
        ("watch_lead_s", h10, cfg.watch_enter),
        ("warn_lead_s", h5, cfg.warn_enter),
    ):
        hit = first_alarm_frame(series, thr, end=onset)
        row[key] = None if hit is None else round((onset - hit) / fps, 1)
    # What the system would have told a member of staff, at the moment it first spoke.
    trigger = first_alarm_frame(h10, cfg.watch_enter, end=onset)
    row["reasons_at_alarm"] = top_reasons(bundle.signals, trigger) if trigger is not None else {}
    return row


def _median_frac(rows: list[dict]) -> float | None:
    """Median share of a clip spent above WARN, over the rows that recorded it."""
    vals = [r["frac_above_warn"] for r in rows if r.get("frac_above_warn") is not None]
    return round(float(np.median(vals)), 3) if vals else None


def _crossing_s(series, threshold: float, fps: float) -> float | None:
    hit = first_alarm_frame(series, threshold)
    return None if hit is None else round(hit / fps, 1)


def aggregate(rows: list[dict], out_dir: Path, budget_per_hour: float, cfg: PolicyConfig) -> dict:
    """Lead times, false alarms and anticipation-at-budget over the whole batch."""
    usable = [r for r in rows if r.get("usable")]
    pos = [r for r in usable if r.get("has_assault") is True]
    neg = [r for r in usable if r.get("has_assault") is False]
    unlabelled = len(usable) - len(pos) - len(neg)
    if not pos and not neg:
        return {"videos": len(usable), "unlabelled": unlabelled}

    def series(row, key):
        with np.load(out_dir / "scores" / f"{row['video_id']}.npz") as d:
            return d[key]

    fps = usable[0]["fps"]
    pos_scores = [series(r, "hazard_5s") for r in pos]
    pos_onsets = [int(round(r["onset_s"] * fps)) for r in pos]
    neg_scores = [series(r, "hazard_5s") for r in neg]

    leads = lead_times(pos_scores, pos_onsets, cfg.warn_enter, fps) if pos else np.array([])
    out = dict(
        videos=len(usable),
        with_assault=len(pos),
        without_assault=len(neg),
        unlabelled=unlabelled,
        # len() over neg_scores (the arrays), not neg (the ledger rows) -- the
        # latter silently measures how many columns a row has.
        negative_hours=round(sum(len(s) for s in neg_scores) / fps / 3600.0, 3),
        # The smallest rate this much footage could even resolve. A budget of
        # 0.15/h cannot be verified on ten minutes of video, and a threshold
        # chosen against it is fitted to noise.
        resolvable_per_hour=round(1.0 / (sum(len(s) for s in neg_scores) / fps / 3600.0), 1)
        if neg_scores
        else None,
        warn_lead_s=lead_time_distribution(leads),
        anticipated=int((leads >= 1.0).sum()),
        # If these are near 1, the lead times above are an artefact of a hazard
        # that is always on, and no threshold will separate anything.
        # .get: a ledger written by an older version has no such column, and a
        # missing diagnostic should not take the whole summary down with it.
        median_frac_above_warn=dict(
            with_assault=_median_frac(pos),
            without_assault=_median_frac(neg),
        ),
        false_warn_per_hour=round(false_alarms_per_hour(neg_scores, cfg.warn_enter, fps), 2)
        if neg
        else None,
        false_watch_per_hour=round(false_alarms_per_hour(neg_scores, cfg.watch_enter, fps), 2)
        if neg
        else None,
    )
    if pos and neg:
        out["at_budget"] = anticipation_at_budget(
            pos_scores, pos_onsets, neg_scores, fps, budget_per_hour
        )
    return out


def shortlist(rows: list[dict], k: int = 15) -> list[dict]:
    """Assaults with the longest genuine pre-onset WARN — the ones worth watching.

    This is a review queue, not a result. Every row here still has to be watched
    to decide whether the alarm tracked a build-up or merely noticed that two
    people were standing close together.
    """
    hits = [r for r in rows if r.get("warn_lead_s")]
    hits.sort(key=lambda r: r["warn_lead_s"], reverse=True)
    return [
        dict(
            video_id=r["video_id"],
            lead_s=r["warn_lead_s"],
            onset_s=r["onset_s"],
            watch_at_s=round(r["onset_s"] - r["warn_lead_s"], 1),
            # Close to 1 means the alarm covered most of the clip; treat the lead
            # as unearned and check the footage before believing it.
            lead_fraction=round(r["warn_lead_s"] / r["duration_s"], 2)
            if r.get("duration_s")
            else None,
            frac_above_warn=r.get("frac_above_warn"),
            quality=r["mean_quality"],
            reasons=r.get("reasons_at_alarm", {}),
        )
        for r in hits[:k]
    ]


def case_template(video_id: str, out_dir: Path) -> dict:
    """A phase-annotation stub pre-filled with what the system claimed.

    The point is to make disagreement cheap: the reviewer edits the numbers the
    system produced rather than starting from an empty file, and the corrected
    result loads straight back through `anticipation_labels.load_phase_annotation`.
    Every field the system guessed is marked, so a stub nobody has corrected
    cannot be mistaken for ground truth.
    """
    row = next(
        (
            json.loads(line)
            for line in (out_dir / "ledger.jsonl").read_text().splitlines()
            if json.loads(line)["video_id"] == video_id
        ),
        None,
    )
    if row is None:
        raise KeyError(f"{video_id} is not in {out_dir / 'ledger.jsonl'}")

    fps, onset = row["fps"], row.get("onset_s")
    lead = row.get("warn_lead_s")
    phases = []
    if onset is not None and lead:
        alarm = onset - lead
        phases = [
            [int(max(0, alarm - 5) * fps), int(alarm * fps), 1],  # precursor, GUESSED
            [int(alarm * fps), int(onset * fps), 2],  # build-up, GUESSED
        ]
    return {
        "video_id": video_id,
        "dataset": row["label_source"],
        "fps": fps,
        "n_frames": row["steps"],
        "assault": [[int(a * fps), int(b * fps)] for a, b in row.get("assault_spans_s", [])],
        "friendly": [],
        "phases": phases,
        "phase_annotated": False,  # flip to true ONLY after a human has checked
        "_guessed": {
            "source": "HeuristicHazard, uncalibrated",
            "warn_lead_s": lead,
            "reasons": row.get("reasons_at_alarm", {}),
            "review": "correct the phase boundaries, then set phase_annotated true",
        },
    }


def run(
    videos_dir: str,
    out_dir: str,
    annotation_dir: str | None = None,
    pattern: str = "*",
    limit: int | None = None,
    process_fps: float = 10.0,
    max_people: int = 12,
    budget_per_hour: float = 0.15,
    cfg: PolicyConfig | None = None,
) -> dict:
    """Score every matching video and write ledger.jsonl + summary.json."""
    cfg = cfg or PolicyConfig()
    out = Path(out_dir)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    found = find_videos(videos_dir, pattern)[: limit or None]
    if not found:
        raise FileNotFoundError(f"no videos matching {pattern!r} under {videos_dir}")

    ledger, rows = out / "ledger.jsonl", []
    done = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            rows.append(row)
            done.add(row["video_id"])

    with ledger.open("a") as f:
        for i, video in enumerate(found, 1):
            if video.stem in done:
                print(f"[{i}/{len(found)}] {video.stem}: already in ledger, skipping", flush=True)
                continue
            try:
                row = score_video(video, out, annotation_dir, process_fps, max_people, cfg)
            except Exception as exc:  # one bad file must not lose the batch
                row = dict(video=str(video), video_id=video.stem, usable=False, reason=str(exc))
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            note = (
                f"lead {row['warn_lead_s']}s"
                if row.get("warn_lead_s")
                else ("no pre-onset WARN" if row.get("has_assault") else "")
            )
            print(
                f"[{i}/{len(found)}] {video.stem}: {row.get('duration_s', 0)}s "
                f"peak {row.get('peak_5s', 0)} {note}",
                flush=True,
            )

    summary = aggregate(rows, out, budget_per_hour, cfg)
    summary["shortlist"] = shortlist(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--videos", help="directory to search for videos")
    parser.add_argument("--out", required=True, help="output directory for poses/scores/ledger")
    parser.add_argument("--annotations", default=None, help="directory of per-frame label CSVs")
    parser.add_argument("--pattern", default="*", help="glob, e.g. 'F_*' for UBI-Fights fights")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--process-fps", type=float, default=10.0)
    parser.add_argument("--max-people", type=int, default=12)
    parser.add_argument("--budget", type=float, default=0.15, help="false WARN per camera-hour")
    parser.add_argument("--case", default=None, help="write an annotation stub for this video id")
    args = parser.parse_args()

    out = Path(args.out)
    if args.case:
        stub = case_template(args.case, out)
        path = out / "cases" / f"{args.case}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stub, indent=2))
        print(f"wrote {path}\nEdit the phase boundaries, then set phase_annotated true.")
        return

    if not args.videos:
        parser.error("--videos is required unless --case is given")
    summary = run(
        args.videos,
        args.out,
        args.annotations,
        args.pattern,
        args.limit,
        args.process_fps,
        args.max_people,
        args.budget,
    )

    print(f"\n{summary['videos']} usable videos")
    if summary.get("unlabelled"):
        print(
            f"  {summary['unlabelled']} had no annotation and are excluded from every "
            f"number below — unlabelled is not the same as quiet"
        )
    if summary.get("with_assault"):
        dist = summary["warn_lead_s"]
        print(
            f"  {summary['anticipated']}/{summary['with_assault']} assaults had a WARN "
            f"at least 1s early; lead p25/median/p75 "
            f"{dist['p25']:.1f}/{dist['median']:.1f}/{dist['p75']:.1f}s (max {dist['max']:.1f}s)"
        )
    sat = summary.get("median_frac_above_warn") or {}
    if sat.get("with_assault") is not None:
        print(
            f"  median share of each clip already above WARN: "
            f"{sat['with_assault']:.0%} of assault videos, "
            f"{sat['without_assault']:.0%} of quiet ones"
            if sat.get("without_assault") is not None
            else f"  median share already above WARN: {sat['with_assault']:.0%}"
        )
    if summary.get("false_warn_per_hour") is not None:
        print(
            f"  on {summary['negative_hours']}h with no assault: "
            f"{summary['false_warn_per_hour']}/h false WARN, "
            f"{summary['false_watch_per_hour']}/h false WATCH"
        )
    if "at_budget" in summary:
        b = summary["at_budget"]
        print(
            f"  at a {b['false_alarms_per_hour']}/h budget (threshold {b['threshold']:.2f}): "
            f"{b['recall']:.0%} anticipated, median lead {b['median_lead_s']:.1f}s"
        )
        floor = summary.get("resolvable_per_hour")
        if floor and floor > 0.15:
            print(
                f"  [!] {summary['negative_hours']}h of quiet footage can only resolve rates "
                f"down to {floor}/h, so the budget threshold above is fitted to noise.\n"
                f"      Score more negatives before believing any of it."
            )
    print(f"\n  review queue: {len(summary['shortlist'])} in {Path(args.out) / 'summary.json'}")
    for item in summary["shortlist"][:5]:
        print(
            f"    {item['video_id']:<20} WARN at {item['watch_at_s']}s, "
            f"onset {item['onset_s']}s (+{item['lead_s']}s)  {', '.join(item['reasons']) or '-'}"
        )


if __name__ == "__main__":
    main()
