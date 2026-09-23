"""Escalation early warning: anticipate an assault instead of recognising one.

Separate from the rest of `src/`, which detects an assault once it is happening.
Nothing here imports from the detection pipeline except the AGCN checkpoint loader,
and nothing there imports from here, so the two can be evaluated independently.

    social_features      interpretable person / pair / scene signals from tracked pose
    anticipation_labels  time-to-onset labels with right-censoring
    baseline             untrained heuristic hazard + trained window baselines
    hazard_model         EscalationNet (relational attention + causal GRU)
    escalation_policy    tiered alerts with dwell, hysteresis and a quality gate
    metrics              lead time, false alarms per hour, anticipation at a budget
    tracked_pose         multi-person pose extraction with stable track ids
    windows              leakage-safe window datasets
    replay               end-to-end CLI over one clip

See docs/EARLY_WARNING_DESIGN.md. Status: proposal, not validated on real data.
"""
