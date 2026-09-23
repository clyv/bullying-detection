"""Replay one clip through the whole early-warning stack and print a timeline.

    tracked pose .npz -> social signals -> hazard -> escalation policy -> timeline

This is the end-to-end entry point, and the thing to run first on real footage:
it needs no labels and no trained model, because the default hazard is the
untrained heuristic. What it produces is a per-timestep record of tier, hazard
and the reasons behind them, which is exactly what the annotation pass and the
threshold calibration both need.

Usage:
    python -m src.early_warning.tracked_pose --video hall.mp4 --output hall.npz
    python -m src.early_warning.replay --poses hall.npz --output hall_timeline.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.early_warning.baseline import HeuristicHazard, geometry_hazard
from src.early_warning.escalation_policy import (
    EscalationPolicy,
    PolicyConfig,
    Signals,
    Tier,
    top_reasons,
)
from src.early_warning.social_features import FeatureConfig, compute, observation_quality


def score_timeline(
    signals: dict,
    quality: np.ndarray,
    fps: float,
    hazard: dict,
    p_assault: np.ndarray | None = None,
    cfg: PolicyConfig | None = None,
) -> list[dict]:
    """Run the policy over pre-computed hazard series and return one record per step."""
    n = len(quality)
    p_assault = np.zeros(n) if p_assault is None else np.asarray(p_assault)
    policy = EscalationPolicy(cfg)
    timeline = []
    for i in range(n):
        s = Signals(
            t_s=i / fps,
            hazard_5s=float(hazard["hazard_5s"][i]),
            hazard_10s=float(hazard["hazard_10s"][i]),
            p_assault=float(p_assault[i]),
            p_precursor=float(hazard["precursor"][i]),
            p_buildup=float(hazard["buildup"][i]),
            quality=float(quality[i]),
            reasons=top_reasons(signals, i),
        )
        alert = policy.update(s)
        timeline.append(
            {
                "t_s": round(s.t_s, 3),
                "tier": policy.tier.name,
                "hazard_5s": round(s.hazard_5s, 4),
                "hazard_10s": round(s.hazard_10s, 4),
                "quality": round(s.quality, 3),
                "changed": alert is not None,
                "notify": bool(alert and alert.notify),
                "gated": bool(alert and alert.gated),
                "reasons": s.reasons,
            }
        )
    return timeline


def summarise(timeline: list[dict]) -> dict:
    """Tier durations and the transitions worth reading in the console."""
    if not timeline:
        return {"steps": 0}
    changes = [r for r in timeline if r["changed"]]
    held = {name: 0 for name in Tier.__members__}
    for r in timeline:
        held[r["tier"]] += 1
    step_s = timeline[1]["t_s"] - timeline[0]["t_s"] if len(timeline) > 1 else 0.0
    return {
        "steps": len(timeline),
        "duration_s": round(timeline[-1]["t_s"], 2),
        "seconds_in_tier": {k: round(v * step_s, 2) for k, v in held.items() if v},
        "transitions": [
            {"t_s": r["t_s"], "tier": r["tier"], "notify": r["notify"], "reasons": r["reasons"]}
            for r in changes
        ],
        "notifications": sum(r["notify"] for r in timeline),
    }


def run(
    poses_path: str,
    output: str | None = None,
    fps: float | None = None,
    scorer: str = "heuristic",
    near_radius: float = 1.5,
) -> dict:
    """Score one tracked-pose file and return {'timeline', 'summary'}."""
    with np.load(poses_path) as data:
        kp, sc = data["keypoints"], data["scores"]
        file_fps = float(data["fps"]) if "fps" in data else 10.0
    fps = float(fps or file_fps)
    if len(kp) == 0:
        raise ValueError(f"{poses_path} holds no frames")

    cfg = FeatureConfig(fps=fps, near_radius=near_radius)
    bundle = compute(kp, sc, cfg)
    quality = observation_quality(bundle.geometry)

    if scorer == "geometry":
        base = geometry_hazard(bundle.signals, fps)
        hazard = {"hazard_5s": base, "hazard_10s": base, "precursor": base, "buildup": base * 0.0}
    else:
        hazard = HeuristicHazard(fps=fps)(bundle.signals)

    timeline = score_timeline(bundle.signals, quality, fps, hazard)
    result = {"timeline": timeline, "summary": summarise(timeline), "source": poses_path}
    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--poses", required=True, help="tracked .npz from tracked_pose.py")
    parser.add_argument("--output", default=None, help="write the full timeline as JSON")
    parser.add_argument("--fps", type=float, default=None, help="override the stored fps")
    parser.add_argument("--scorer", default="heuristic", choices=("heuristic", "geometry"))
    args = parser.parse_args()

    result = run(args.poses, args.output, args.fps, args.scorer)
    summary = result["summary"]
    print(f"{args.poses}: {summary['steps']} steps, {summary.get('duration_s', 0)}s")
    print(f"  time per tier: {summary.get('seconds_in_tier', {})}")
    for change in summary.get("transitions", []):
        flag = " NOTIFY" if change["notify"] else ""
        reasons = ", ".join(change["reasons"]) or "-"
        print(f"  {change['t_s']:7.2f}s  {change['tier']:<8}{flag}  {reasons}")
    if not summary.get("transitions"):
        print("  stayed CALM throughout")
    print(
        "\n[note] Scores are uncalibrated. Set thresholds from this camera's own normal"
        "\n       footage with metrics.threshold_for_budget before alerting anyone."
    )


if __name__ == "__main__":
    main()
