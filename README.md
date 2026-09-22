# School-Safe Vision 🎥🛡️

[![CI](https://github.com/clyv/bullying-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/clyv/bullying-detection/actions/workflows/ci.yml)

A computer vision research project exploring whether bullying, harassment,
and physical aggression can be detected from school surveillance cameras
using visual signals alone — no audio, no identity recognition.

## The Approach

Rather than relying on raw video appearance, this project unifies three
very different datasets into a single **2D pose-skeleton representation**,
allowing a model to learn aggressive interaction dynamics that transfer
across camera types and environments:

| Dataset | Modality | Contributes |
|---|---|---|
| Bullying10K | DVS event camera | Physical bullying actions, privacy-preserving |
| NTU RGB+D 120 | Kinect skeletons | Aggressive vs. neutral two-person interactions (point, push, follow, grab, whisper) |
| UT-Interaction | RGB video | Outdoor surveillance-style confrontations |

```
Bullying10K (DVS events)  → accumulate to pseudo-frames → pose extraction ─┐
NTU RGB+D 120 (3D skel)   → project 3D → 2D ───────────────────────────────┤
UT-Interaction (RGB)      → pose extraction (YOLO-Pose) ───────────────────┼→ unified 2D skeleton sequences
                                                                           │
School CCTV (deployment)  → pose extraction (same extractor) ──────────────┘
                                        ↓
                    Skeleton-based classifier (ST-GCN / 2s-AGCN)
                                        ↓
              Classes: aggressive / bullying / neutral interactions
```

RWF-2000 (real CCTV violence clips) is kept in the stack as an optional
fourth source — the only genuinely messy real-world footage.

## What This Is (and Isn't)

✅ A feasibility study for pose-based aggression detection on CCTV

✅ Privacy-conscious by design — skeletons, not faces

❌ Not a production system — no claims of detecting verbal-only abuse
   or social exclusion, which are invisible to cameras

## Repository Layout

```
├── src/
│   ├── preprocessing/
│   │   ├── dvs_to_frames.py    # Bullying10K event accumulation → pseudo-frames
│   │   ├── ntu_skeleton.py     # parse NTU .skeleton files, 3D → 2D projection
│   │   └── pose_extraction.py  # YOLO-Pose wrapper for RGB / pseudo-frame sources
│   ├── datasets/               # PyTorch Dataset per source + unified loader
│   ├── models/                 # ST-GCN / baseline implementations
│   ├── training/
│   └── evaluation/
├── data/                       # never committed — see data/README.md
├── notebooks/                  # per-dataset EDA
├── configs/
└── docs/
```

## Getting Started

Requires **Python 3.12**.

```
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on Linux)
pip install -r requirements.txt
```

> **GPU note:** `requirements.txt` pins CUDA 12.8 (`cu128`) PyTorch builds,
> required for RTX 50-series (Blackwell / sm_120) GPUs. On CPU-only or
> older-GPU machines, install the matching plain builds instead.

Fetch datasets following [data/README.md](data/README.md), then run the
preprocessing for whichever sources you have:

```
# UT-Interaction (RGB) — YOLO-Pose
python -m src.preprocessing.pose_extraction   --input data/ut_interaction --output outputs/ut_poses
# NTU RGB+D 120 (3D skeletons) — projected to 2D
python -m src.preprocessing.ntu_skeleton      --input data/ntu/skeletons  --output outputs/ntu_poses --classes relevant
# Bullying10K (DVS) — Route B: convert the provided COCO pose labels directly
python -m src.preprocessing.bullying10k_poses --input data/bullying10k    --output outputs/b10k_poses
# Bullying10K — Route A: accumulate events to pseudo-frames, then extract poses
python -m src.preprocessing.dvs_to_frames     --input data/bullying10k    --output outputs/b10k_frames --png
python -m src.preprocessing.pose_extraction   --input outputs/b10k_frames --output outputs/b10k_poses
```

Every route converges on the same `.npz` format: `keypoints (T, M, 17, 2)`
COCO-order pixel coordinates plus `scores (T, M, 17)` confidences (dataset
converters also write an integer `label`).

Train and evaluate on a single dataset, configured through a YAML file
([configs/baseline.yaml](configs/baseline.yaml) for UT-Interaction,
[configs/bullying10k.yaml](configs/bullying10k.yaml) for Bullying10K,
[configs/ntu.yaml](configs/ntu.yaml) for NTU):

```
python -m src.training.train      --config configs/bullying10k.yaml   # checkpoints to outputs/checkpoints/
python -m src.evaluation.evaluate --config configs/bullying10k.yaml   # accuracy, calibration, abstention
```

### Model and training stack

| Component | Default | Alternative |
|---|---|---|
| Backbone ([factory.py](src/models/factory.py)) | `agcn` — learnable graph topology + multi-scale temporal conv | `stgcn` reproduces the original Phase 1–4 numbers |
| Input stream ([streams.py](src/datasets/streams.py)) | `joint` | `bone`, `joint_motion`, `bone_motion` — train all four and ensemble |
| Loss ([losses.py](src/training/losses.py)) | focal (γ=2) + class-balanced weights + label smoothing | `focal_gamma: 0` falls back to cross-entropy |
| Augmentation ([augment.py](src/datasets/augment.py)) | joint dropout, temporal crop, flip, person swap, scale jitter | remove the `augment:` block to disable |

The four streams are trained separately and fused by summing their *calibrated*
softmax scores. Bone and motion streams are translation-invariant and largely
scale-invariant by construction, so they disagree with the joint stream exactly
where the joint stream is being fooled by absolute pixel geometry:

```
for %s in (joint bone joint_motion bone_motion) do python -m src.training.train --config configs/unified.yaml --stream %s
python -m src.evaluation.ensemble --config configs/unified.yaml
```

### Reading the numbers honestly

`evaluate.py` reports more than accuracy, because accuracy alone hid this
project's real failure (a model returning P(aggressive) = 1.000 on footage it
was getting wrong):

- **ECE before/after temperature scaling** — how far the confidence was from the
  truth. A scalar temperature fitted on validation cannot change any prediction,
  only make the score mean something.
- **Conformal abstention** — coverage, abstention rate, and *selective accuracy*
  (how often the system is right on the windows it chose to answer) at a
  threshold with a distribution-free guarantee, replacing a hand-tuned pixel gate.

For the **unified model** ([configs/unified.yaml](configs/unified.yaml)), every
dataset's native classes are collapsed to a binary *aggressive vs. neutral* space
([src/datasets/taxonomy.py](src/datasets/taxonomy.py)):

```
python -m src.evaluation.cross_dataset --config configs/unified.yaml
```

This prints two things. **Read the second one.** Pooled accuracy on a random split
is an upper bound inflated by corpus identity — when several corpora with distinct
capture rigs are mixed and split at random, the cheapest route to a high score is
to recognise which corpus a clip came from and apply that corpus's class prior.
The **leave-one-dataset-out** table trains on every corpus but one and tests on the
one held out; its mean is the honest generalization estimate.

Two further guards against a flattering headline:

- **By-source breakdown.** The pooled report splits accuracy by source corpus. One
  run's 88% pooled figure turned out to be 95.8% on Bullying10K and 87.1% on NTU,
  but only 55–62% on the real-CCTV corpora it had trained on. The two lab datasets
  are over 90% of the pool.
- **Motion-energy baseline.** Each leave-one-dataset-out fold also scores a single
  threshold on how fast the most agitated person moves
  ([baselines.py](src/evaluation/baselines.py)), fitted on the training corpora.
  The `gain` column is what the model knows beyond "someone is moving fast". It was
  +5.3 points on average, near zero on NTU (whose actors mime violence slowly) and
  UBI-Fights, and *negative* on fight-surv. The threshold alone scores 40.1% on
  held-out Bullying10K — the same figure the original ST-GCN collapsed to there,
  which suggests that collapse was this shortcut.

| Held out (AGCN, joint stream) | Model | Motion energy | Gain |
|---|---|---|---|
| UT-Interaction | 77.5% | 68.3% | +9.2 |
| Bullying10K | 62.5% | 40.1% | +22.4 |
| NTU RGB+D | 54.9% | 51.9% | +3.0 |
| UBI-Fights | 55.2% | 53.8% | +1.4 |
| fight-surv | 55.7% | 65.0% | −9.3 |
| **Mean** | **61.1%** | **55.8%** | **+5.3** |

Pooled test accuracy for the same model is 88.2%.

A degradation benchmark converts "it doesn't work on real CCTV" into a curve, by
corrupting the held-out split one axis at a time (joint dropout, coordinate noise,
lost participant, scale error) and reporting accuracy against each:

```
python -m src.evaluation.robustness --config configs/unified.yaml
```

Finally, **temporal localization** answers *when* an incident occurs in a
continuous stream: it slides the trained binary model over an untrimmed pose
sequence and merges aggressive windows into incident intervals (frame ranges +
timestamps) to flag for human review.

```
python -m src.evaluation.localize --stream outputs/cctv_poses/clip.npz \
    --checkpoint outputs/checkpoints/phase4_unified/agcn_best.pt --config configs/unified.yaml
```

The preprocessing, metrics, and temporal-localization logic is unit-tested
(`pip install pytest ruff && pytest`); the model, training, single-/cross-dataset
evaluation, and stream-localization paths are covered too. The same checks run in
CI on every push.

## Roadmap

- [x] **Phase 1 — Baseline:** pose-extraction pipeline (YOLO-Pose) + ST-GCN baseline (training & evaluation) on UT-Interaction / RWF-2000
- [x] **Phase 2 — Bullying10K:** DVS events → pseudo-frames → poses, or the dataset's provided COCO pose labels → unified `.npz`
- [x] **Phase 3 — NTU mutual actions:** relevant-class subset, 3D → 2D projection, unified labels, added to the training set
- [x] **Phase 4 — Unified model:** binary aggressive-vs-neutral space, cross-dataset evaluation, per-dataset (leave-one-out) ablations, confusion analysis
- [x] **Phase 5 (stretch):** temporal localization — sliding-window scoring + incident-interval merging to flag *when* in a stream aggression occurs (school-proxy testing still pending suitable footage)
- [x] **Phase 6 — Generalization & calibration:** adaptive-topology backbone, four-stream ensembling, degradation-targeted augmentation, focal loss + temperature scaling, conformal abstention, and a robustness sweep — aimed at the gap between pooled accuracy and leave-one-dataset-out accuracy

## Limitations

What this system fundamentally cannot see:

- **Verbal-only abuse** delivered with neutral body language — there is no audio, by design
- **Social exclusion / relational bullying** and **cyberbullying** — not visually observable
- **Domain gap:** every training dataset uses adult actors in non-school settings;
  children's body proportions and movement dynamics will degrade performance.
  This gap is documented, not solved.
- **Camera dependence:** pose extraction degrades with distance, angle, and
  resolution, so real-world quality hinges on camera placement

Any deployment of a system like this should only **flag incidents for human
review** — it must never make autonomous accusations.
