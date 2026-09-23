"""Check the social signals against UT-Interaction's own ground truth.

Build-order step 1: before any model is trained, the untrained signals should
already peak at the punch, kick and push moments a human labelled. If they do
not, the features are wrong and nothing downstream matters.

UT-Interaction is the only corpus here that gives frame-accurate boundaries for
*six* interaction types in one continuous take, three aggressive and three not.
That makes it the one place where "does the hazard rise before a punch?" can be
separated from "does the hazard rise whenever two people stand close together?",
which is the failure mode a fight-only dataset cannot reveal.

    python -m src.early_warning.tracked_pose --video data/ut_interaction/seq1.avi \
        --output outputs/ut_seq1_tracked.npz
    python -m src.early_warning.validate_ut --poses outputs/ut_seq1_tracked.npz --sequence seq1
"""

from __future__ import annotations

import argparse

import numpy as np

from src.early_warning.baseline import HeuristicHazard, geometry_hazard
from src.early_warning.escalation_policy import PolicyConfig
from src.early_warning.metrics import alarm_episodes, first_alarm_frame
from src.early_warning.social_features import FeatureConfig, compute, observation_quality

CLASSES = {0: "handshake", 1: "hug", 2: "kick", 3: "point", 4: "punch", 5: "push"}
AGGRESSIVE = frozenset({2, 4, 5})  # kick, punch, push
LABELS_XLS = "data/ut_interaction/ut-interaction_labels_110912.xls"
SOURCE_FPS = 30.0  # the spreadsheet numbers frames in the original 30 fps video


def load_events(sequence: str, labels_path: str = LABELS_XLS) -> list[tuple[int, str, int, int]]:
    """(class id, name, start frame, end frame) for one sequence, in time order.

    The sheet holds a main block and then an `others:` block listing extra
    interactions performed by a second pair in the background of the same takes.
    Both are real events in the video, so both are kept and the second pair is
    marked — a signal that fires on the foreground pair while the labelled
    interaction is happening behind them is not a hit.
    """
    import pandas as pd  # optional: only this validation needs a spreadsheet reader

    sheet = pd.read_excel(labels_path, header=None).iloc[2:, [1, 2, 3, 4]]
    sheet.columns = ["seq", "cls", "start", "end"]
    marker = sheet.index[sheet["seq"].astype(str).str.startswith("others")]
    second_pair = int(marker[0]) if len(marker) else len(sheet) + 2

    rows = sheet[sheet["seq"] == sequence]
    events = []
    for index, row in rows.iterrows():
        cls = int(row["cls"])
        name = CLASSES[cls] + ("" if index < second_pair else " (2nd pair)")
        events.append((cls, name, int(row["start"]), int(row["end"])))
    return sorted(events, key=lambda e: e[2])


def evaluate(
    poses_path: str,
    events: list[tuple[int, str, int, int]],
    cfg: PolicyConfig | None = None,
    run_up_s: float = 5.0,
) -> dict:
    """Score a tracked-pose file and compare the hazard against the labels."""
    cfg = cfg or PolicyConfig()
    with np.load(poses_path) as data:
        kp, sc = data["keypoints"], data["scores"]
        fps = float(data["fps"]) if "fps" in data else 10.0
    step = SOURCE_FPS / fps

    bundle = compute(kp, sc, FeatureConfig(fps=fps))
    quality = observation_quality(bundle.geometry)
    hazard = HeuristicHazard(fps=fps)(bundle.signals)
    n = len(quality)

    windows = [
        (cls, name, min(int(a / step), n - 1), min(int(b / step), n - 1))
        for cls, name, a, b in events
    ]

    rows = []
    for cls, name, a, b in windows:
        h5 = float(hazard["hazard_5s"][a : b + 1].max())
        h10 = float(hazard["hazard_10s"][a : b + 1].max())
        tier = "WARN" if h5 >= cfg.warn_enter else ("WATCH" if h10 >= cfg.watch_enter else "CALM")
        rows.append(
            {
                "cls": cls,
                "name": name,
                "aggressive": cls in AGGRESSIVE,
                "start_s": round(a / fps, 1),
                "end_s": round(b / fps, 1),
                "peak_5s": round(h5, 3),
                "peak_10s": round(h10, 3),
                "quality": round(float(quality[a : b + 1].mean()), 2),
                "tier": tier,
            }
        )

    # Background: everything outside an interaction and its run-up. A hazard that
    # is high here is a false alarm however well it scores on the events.
    covered = np.zeros(n, dtype=bool)
    for _, _, a, b in windows:
        covered[max(0, a - int(run_up_s * fps)) : b + 1] = True
    bg = ~covered
    masked = np.where(bg, hazard["hazard_10s"], 0.0)
    episodes = alarm_episodes(masked, cfg.watch_enter, fps, min_gap_s=5.0)
    bg_hours = bg.sum() / fps / 3600.0

    # Lead time is only credited for an alarm raised after the previous
    # interaction ended; an alarm still running from the last event is not
    # anticipation of this one.
    leads = []
    for cls, name, a, _ in windows:
        if cls not in AGGRESSIVE:
            continue
        prev_end = max([e for _, _, _, e in windows if e < a] + [0])
        entry = {"name": name, "onset_s": round(a / fps, 1)}
        for key, series, thr in (
            ("watch_lead_s", hazard["hazard_10s"], cfg.watch_enter),
            ("warn_lead_s", hazard["hazard_5s"], cfg.warn_enter),
        ):
            hit = first_alarm_frame(series[prev_end:a], thr)
            entry[key] = None if hit is None else round((a - prev_end - hit) / fps, 1)
        leads.append(entry)

    agg = np.array([r["peak_10s"] for r in rows if r["aggressive"]])
    ben = np.array([r["peak_10s"] for r in rows if not r["aggressive"]])
    base = geometry_hazard(bundle.signals, fps)

    return {
        "source": poses_path,
        "fps": fps,
        "steps": n,
        "duration_s": round(n / fps, 1),
        "mean_quality": round(float(quality.mean()), 2),
        "events": rows,
        "separation": round(float(agg.mean() - ben.mean()), 3) if agg.size and ben.size else 0.0,
        "geometry_separation": round(
            float(
                np.mean([base[a : b + 1].max() for c, _, a, b in windows if c in AGGRESSIVE])
                - np.mean([base[a : b + 1].max() for c, _, a, b in windows if c not in AGGRESSIVE])
            ),
            3,
        )
        if agg.size and ben.size
        else 0.0,
        "leads": leads,
        "background_s": round(float(bg.sum() / fps), 1),
        "background_peak": round(float(hazard["hazard_10s"][bg].max()) if bg.any() else 0.0, 3),
        "false_watch_per_hour": round(len(episodes) / bg_hours, 1) if bg_hours > 0 else 0.0,
    }


def report(result: dict) -> str:
    """The table worth pasting into the design doc."""
    # Parenthesised: inside a list, two adjacent strings concatenate silently, so
    # a dropped comma would merge two rows instead of raising anything.
    lines = [
        (
            f"{result['source']}: {result['steps']} steps @ {result['fps']:.0f} fps "
            f"= {result['duration_s']}s, mean pose quality {result['mean_quality']}"
        ),
        "",
        f"{'interaction':<20}{'window':>15}{'peak 5s':>10}{'peak 10s':>10}{'qual':>7}  tier",
        "-" * 68,
    ]
    for r in result["events"]:
        mark = "*" if r["aggressive"] else " "
        lines.append(
            f"{mark}{r['name']:<19}{r['start_s']:6.1f}-{r['end_s']:5.1f}s"
            f"{r['peak_5s']:10.3f}{r['peak_10s']:10.3f}{r['quality']:7.2f}  {r['tier']}"
        )
    lines += [
        "",
        (
            f"aggressive minus benign peak: {result['separation']:+.3f} "
            f"(coarse-geometry baseline {result['geometry_separation']:+.3f})"
        ),
        (
            f"background {result['background_s']}s: peak {result['background_peak']:.3f}, "
            f"{result['false_watch_per_hour']} false WATCH/camera-hour"
        ),
        "",
        "lead time before each aggressive onset:",
    ]
    for lead in result["leads"]:
        watch = "--" if lead["watch_lead_s"] is None else f"{lead['watch_lead_s']:.1f}s"
        warn = "--" if lead["warn_lead_s"] is None else f"{lead['warn_lead_s']:.1f}s"
        lines.append(
            f"  {lead['name']:<18} onset {lead['onset_s']:5.1f}s   WATCH {watch:>6}   WARN {warn:>6}"
        )
    lines += [
        "",
        "[note] A high peak on a handshake or a hug means the hazard is measuring",
        "       proximity, not aggression. Read the benign rows before the starred ones.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--poses", required=True, help="tracked .npz from tracked_pose.py")
    parser.add_argument("--sequence", default="seq1", help="which UT-Interaction sequence")
    parser.add_argument("--labels", default=LABELS_XLS)
    parser.add_argument("--json", default=None, help="also write the full result as JSON")
    args = parser.parse_args()

    result = evaluate(args.poses, load_events(args.sequence, args.labels))
    print(report(result))
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
