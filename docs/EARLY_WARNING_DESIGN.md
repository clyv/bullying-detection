# Escalation Early Warning: Design Spec

Status: proposal, not yet validated on real data. Every value marked *starting point* is to be revised once results exist.

This module lives in `src/early_warning/` and is deliberately **separate from the detection pipeline**. Nothing in `src/datasets`, `src/models`, `src/training` or `src/evaluation` imports it, and it imports nothing from them. The two can be evaluated, broken and replaced independently.

## 1. Goal

The existing system recognises an assault once it is happening. This module estimates, continuously and per camera, how likely an assault is to start in the next few seconds, and why. It is decision support for supervising staff, never an automated judgement about a student.

| Phase | Definition | Observable cues |
|---|---|---|
| 0 Calm | No hostile interaction between specific people | Normal movement; conversation groups with an empty centre |
| 1 Precursor | Visible hostility between specific people, no physical pressure yet | Squaring up face to face at close range; hard gesturing while standing still; pointing or raised arms toward someone; one person advancing while the other gives ground |
| 2 Build-up | Spatial or physical pressure on a target | Encirclement; being outnumbered; retreating then stalling against a wall or corner; a first shove or grab; outsiders converging |
| 3 Assault | First strike, kick, slap, strangle or hair grab | Existing AGCN detector |

Outputs per camera and timestep: the hazard `P(assault onset within 2 / 5 / 10 s)`, phase probabilities, per-person target and aggressor scores, the top contributing signals as human-readable reasons, and an alert tier.

## 2. What the literature actually supports

No published system anticipates bullying through staged phases. The components come from separate fields, and two findings constrain the design before any code is written.

**Most confrontations never become violent.** Collins' micro-sociology makes confrontational tension/fear the central mechanism: most confrontations abort. Levine, Taylor and Best coded 42 CCTV incidents (312 people) and found third parties were more often conciliatory than escalating, increasingly so as the group grew. A precursor detector is therefore *not* an alarm, and the tier system below is built around that.

**Reported "early violence detection" numbers are partly an artifact.** Ganesh (2026) holds tracker, head and supervision fixed and finds (a) no pose representation beat coarse bounding-box geometry, and (b) scoring only pre-onset frames retained 39–91% of above-chance separation, traced to dataset provenance cues (title cards, watermarks) absent from the surveillance-sourced normal class. Two consequences: `baseline.geometry_hazard` exists as the baseline everything must beat, and precursor and normal clips must share source and camera characteristics or the evaluation is measuring the dataset.

| Source | Adopt | Avoid |
|---|---|---|
| Traffic-accident anticipation (DSA, UString, DRIVE) | TTA / mTTA metrics; exponential anticipation loss; AP always reported beside lead time | TTA alone, which rewards firing on everything |
| Clinical aggression-onset forecasting (autism, Hawkes point processes) | Discrete-time hazard framing; `P(onset ≤ t)` alerting; self-exciting term after a shove | Assuming physiological lead times transfer to video |
| AHB-F / D3P (WACV 2025/26) | Observe-only-normal evaluation protocol | Treating generic anomaly forecasting as bullying-specific |
| Aggression escalation corpora (Lefter et al.) | Operator-defined tiers: medium = increase attention, high = act | Assuming unimodal fusion is straightforward |
| F-formation detection (Kendon's o-space) | Inverted: an *occupied* o-space is encirclement | — |
| Motorola threat-proximity patents | Closing distance and approach vectors as features | Lone-victim framing; watchlists |
| CN117952808A (campus anti-bullying patent) | Gathering in a known blind spot as a context trigger | Face watchlists and voice-emotion: fragile and banned in EU education |
| Criminology CCTV studies (Weenink; Liebst; "Circles of Peace") | Asymmetry, outnumbering, retreat; de-escalation is the norm | Assuming every conflict escalates; bystander rings are often peacemaking |
| Police pre-attack indicators | Hypotheses to test; the rule that cues come in clusters | Practitioner lore treated as validated, especially on children |

**The horizons are assumptions.** No peer-reviewed source gives a seconds-level distribution of precursor→assault lead time, and existing frame-level methods reach only 0–1 s of advance warning. If re-annotated footage shows intervals are usually under 2 s, collapse to a short "imminent" head plus a slower context signal, and say so.

## 3. Design principles

**Predict time to event, not a clip label.** Clip-level violent/non-violent labels put pre-fight footage in the non-violent class, teaching models to ignore precursors. A discrete-time hazard model trained on onset times asks the right question and handles quiet videos through right-censoring.

**Interpretable signals first, deep learning second.** Phase 4 showed the AGCN beats a motion-energy threshold by only ~5 points across datasets. Hand-built signals grounded in escalation research give an explainable baseline with little data; a learned model is kept only if it wins leave-one-dataset-out.

**Many people, stable identities.** Encirclement, outnumbering and convergence are invisible with two skeletons.

**The alarm budget is the metric.** A threshold of 0.5 means nothing operationally. Thresholds come from a false-alarm budget on that camera's own normal footage.

**Abstain on bad evidence.** Tiny or occluded skeletons produce confident nonsense — the failure the detection side already hit. `observation_quality` gates the anticipation tiers at WATCH below a quality floor.

**Humans decide.** Alerts mean "go and supervise", always carry a reason, and never trigger consequences automatically. Outputs describe observable behaviour (distance, orientation, contact), never inferred emotion.

## 4. Architecture

```text
video
  │
  ▼
L0  tracked_pose.py ── up to 12 people, stable track ids, 10 fps
  │
  ├──► L1  social_features.py (person / pair / scene, interpretable)
  │          ├──► baseline.geometry_hazard    coarse geometry only
  │          ├──► baseline.HeuristicHazard    untrained, runs today
  │          ├──► baseline.WindowHazard       gradient boosting on window stats
  │          └──► hazard_model.EscalationNet  relational attention + causal GRU
  │
  └──► AGCN assault detector on the highest-risk pair
                                     │
                                     ▼
              L3  escalation_policy.py (tiers, dwell, hysteresis, quality gate)
                                     ▼
              L4  replay.py → timeline JSON → triage UI
```

| Module | Role |
|---|---|
| `tracked_pose.py` | Ultralytics tracking (ByteTrack/BoT-SORT) → `keypoints`, `scores`, `track_ids`. `SlotAssigner` keeps one person in one array slot, with a grace period so brief occlusion doesn't reshuffle everyone. |
| `social_features.py` | Person, pair, target and scene signals. Distances in body heights, speeds in body heights per second, so a feature means the same thing near and far, child and adult. |
| `anticipation_labels.py` | Time-to-onset labels with right-censoring, an onset-jitter band at weight 0, and no hazard supervision inside an assault. |
| `windows.py` | Pools windows across videos and splits by video or dataset — never by window, since overlapping windows from one incident would leak. |
| `baseline.py` | The three non-neural rungs, plus `self_exciting` for the post-shove trace. |
| `hazard_model.py` | EscalationNet and the survival/anticipation losses. |
| `escalation_policy.py` | CALM → WATCH → WARN → INCIDENT with dwell, hysteresis, cooldown and the quality gate. |
| `metrics.py` | Lead time, alarm episodes, false alarms per hour, anticipation at a budget, lead-time percentiles. |
| `replay.py` | End-to-end CLI over one clip. |

## 5. Data and labels

Hazard labels need only the assault onset time, which is objective. Phase labels are subjective and optional.

| Tier | Source | What it provides | Status |
|---|---|---|---|
| A | UT-Interaction `seq1`–`seq20` + labels spreadsheet | Onsets for kick, punch, push, with handshake, hug, point as hard negatives after a similar approach | Have |
| A | UBI-Fights (80 h, 1,000 videos, 216 with fights, frame-level) | Real run-ups; ~780 normal videos for false-alarm rates | Have as clips; the **full videos** and frame labels are needed |
| B | NTU-CCTV-Fights (1,000 videos, frame-level) | More onsets, surveillance and phone cameras | Request |
| B | BEHAVE (approach, chase, follow, fight) | Group build-up primitives | Download |
| C | Own phase annotations, 150–300 videos | Precursor/build-up intervals; **de-escalated conflicts** | To create |
| D | Staged recordings with consenting adults | Encirclement and cornering, plus their innocuous look-alikes | Optional |

Annotation schema, one JSON per video, read by `anticipation_labels.load_phase_annotation`:

```json
{
  "video_id": "ubi_F_0042",
  "dataset": "ubi_fights",
  "fps": 30,
  "n_frames": 5400,
  "assault": [[1830, 2410]],
  "friendly": [],
  "phases": [[1200, 1519, 1], [1520, 1829, 2]],
  "phase_annotated": true,
  "flags": {"encircle": true, "corner": false, "shove": true, "gathering": true, "deescalated": false},
  "annotator": "cj"
}
```

**Guideline.** The precursor starts at the first visible hostile interaction between the people who later fight. The build-up starts at the first spatial or physical escalation. The assault onset is the first strike, using the dataset's own annotation where one exists. Episodes with a precursor or build-up but **no** assault are annotated too and flagged `deescalated`; they are the most valuable negatives in the whole set.

**Agreement.** Double-annotate 20% and report Cohen's kappa on frame-level phase plus boundary disagreement in seconds. Below roughly 0.6 (*starting point*), don't train the phase head. The hazard model does not depend on it.

**A bug this design depends on fixing.** `src/preprocessing/ubi_fights.py` samples "neutral" chunks from everything outside fight spans — including the seconds immediately before each fight. The build-up is currently being fed to the detector labelled neutral. Add a guard band there, and extract **whole videos** here so the run-up survives with its timing.

## 6. Evaluation protocol

**Headline: anticipation recall at a false-alarm budget.** Choose the lowest threshold whose false WARN episodes on normal footage stay within budget, then report the share of assaults with a WARN-level crossing at least 1 s before onset, plus the median lead (`metrics.anticipation_at_budget`). Budget *starting point*: at most one false WARN per camera per school day (~0.15/camera-hour); WATCH up to 2/camera-hour.

**Secondary.** Video-level AP, mTTA, TTA at 80% recall, always with AP beside TTA. Hazard calibration via expected calibration error, reusing `src/evaluation/calibrate.py`.

**Report the anticipatable fraction separately.** Some assaults have no visible build-up. Lead time is a distribution (`metrics.lead_time_distribution`), never a single promise.

**Splits.** By video within a dataset, and leave-one-dataset-out across corpora.

**Baselines every model must beat.** `geometry_hazard` (coarse geometry, per Ganesh), motion energy (`kinetic`), closest-pair distance, largest group size, and the AGCN assault probability used as an anticipation score — the last checks that we are anticipating rather than catching the first blow sooner.

**Ablations.** Drop each family in turn: facing, distance and closing, gesture, enclosure and outnumbering, cornering, contact, convergence.

**Scenario checks.** Approach-then-handshake against approach-then-punch on UT-Interaction; annotated de-escalated conflicts counted separately as justified alerts rather than false alarms.

**Stage 0 success criterion (*starting point*).** In the leave-one-dataset-out mean, anticipation recall at the WARN budget beats the best simple baseline by at least 10 points, with a median lead of at least 1 s. If it doesn't, find which signals fail before building the neural model.

## 7. Privacy, ethics and law

Process skeletons, not faces; run pose extraction on-site; never add identity recognition. Track ids live for seconds within one camera and are discarded. Keep raw video only as long as review or annotation requires, with access logging.

Two prohibitions shape the design (not legal advice; verify current guidance):

- **EU AI Act Article 5(1)(f):** inferring emotions in education is prohibited, in force since 2 February 2025. Outputs are framed as observable behaviour — distance, orientation, contact — never as anger or intent. That framing is the compliance argument, so keep it in the code, the UI and any paper.
- **Article 5(1)(d):** predicting the risk of a person committing a criminal offence **based solely on profiling or personality traits** is prohibited. The system therefore scores *situations*, never students: no per-student risk score, no identity persisted across sessions, no watchlist — which is exactly what the CN117952808A patent proposes and exactly what not to copy.

Check for bias across children's body proportions, dense crowds, wheelchair users and other atypical skeletons, and cultural differences in gesture and personal distance. Tell students and parents the system exists and what it does.

## 8. Build order

| # | Task | Acceptance |
|---|---|---|
| 1 | ~~Run `tracked_pose.py` + `replay.py` on UT-Interaction `seq1`~~ **done — see §8.1** | Signals plotted over the clip peak at its known punch/kick/push moments; ID-switch count logged |
| 2 | UBI-Fights whole-video extraction and loader | Per-dataset counts of event and censored windows; five videos spot-checked by eye |
| 3 | Baseline table | `geometry_hazard`, motion energy, min pair distance, largest group, AGCN-as-anticipation, scored with `metrics.py` |
| 4 | `WindowHazard` leave-one-dataset-out | Beats every baseline by the Stage 0 criterion, or the signals get fixed first |
| 5 | Policy replay over whole videos | Timelines for `seq1` and five UBI-Fights videos; thresholds set from the budget |
| 6 | Annotation pass (150–300 videos) | Kappa reported; de-escalated episodes included |
| 7 | `EscalationNet` | Beats Stage 0 leave-one-dataset-out, or is dropped |
| 8 | Shadow mode | Two to four weeks, no alerts, per-camera thresholds from that camera's own footage |

### 8.1 First run on real footage — UT-Interaction seq1

68 s, 677 steps at 10 fps, mean pose quality 0.88, 3 track ids for 2 actors and
one passer-by (no ID churn). Reproduce with
`python -m src.early_warning.validate_ut --poses outputs/ut_seq1_tracked.npz`.

| interaction | window | peak 5 s | peak 10 s | tier |
|---|---|---|---|---|
| **punch** | 25.4–28.0 s | 0.340 | 0.589 | WATCH |
| **kick** | 30.6–33.6 s | 0.339 | 0.677 | WATCH |
| hug | 35.4–40.2 s | 0.522 | 0.697 | WATCH |
| point (2nd pair) | 41.6–45.3 s | 0.000 | 0.000 | CALM |
| point | 43.7–46.9 s | 0.000 | 0.000 | CALM |
| handshake | 46.9–52.0 s | 0.345 | 0.690 | WATCH |
| **push** | 53.1–55.8 s | 0.654 | 0.868 | WARN |

The pipeline runs end to end on real video, and all three assaults score above
CALM. Everything else here is a failure, and each one is specific:

1. **The hazard is a proximity detector.** A hug (0.697) and a handshake (0.690)
   both outscore a punch (0.589). Aggressive minus benign is +0.365, against
   +0.247 for the coarse-geometry baseline — a +0.118 margin for the entire
   social-feature stack over "how far apart are they". That is the Ganesh result
   reproduced in miniature, and it fails the Stage 0 criterion in §6.
2. **Pointing scores exactly zero, twice.** The phase-1 precursor — hostile
   gesturing at a distance — is invisible, because `gesture_close_still` and
   `arm_raise_close` are gated on close range. A gesture term that only counts
   when the pair is already adjacent cannot be a precursor feature.
3. **Only one assault had a pre-onset alarm.** The punch got a fresh WATCH 10.4 s
   early, which is most likely the actors walking into frame, not build-up. The
   kick and the push had none: the hazard fell below threshold in the gap between
   interactions and did not recover until contact. No WARN ever preceded an onset.
4. **The false-alarm rate is far outside budget.** One WATCH episode in 32 s of
   walking extrapolates to ~112/camera-hour against a 2/camera-hour target. One
   sequence is far too small to estimate this, but the direction is not in doubt.

The top signals separating assault windows from background are `largest_group`,
`n_people` and `max_mutual_facing` — that is, "two people are near each other and
looking at one another", which is true of every interaction in the corpus. Fix
the gesture gating and the facing/closing asymmetry before step 3, not after.

## 9. Risks and open questions

Tracking in dense crowds may break identities exactly when build-up happens, so measure ID switches. Image-plane geometry distorts depth, so encirclement along the camera axis is under-detected until per-camera floor calibration exists; heading estimation from 2D skeletons is the weakest link and everything geometric depends on it. Footage of conflicts that fizzle out is scarce, which makes false-alarm estimates optimistic. Phase boundaries are subjective. Most training footage shows adults, not children. And lead times may simply be short: some attacks have no visible build-up at all.
