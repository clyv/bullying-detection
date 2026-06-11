# School-Safe Vision 🎥🛡️

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
python -m src.preprocessing.pose_extraction --input data/ut_interaction --output outputs/ut_poses
python -m src.preprocessing.dvs_to_frames   --input data/bullying10k   --output outputs/b10k_frames --png
python -m src.preprocessing.ntu_skeleton    --input data/ntu/skeletons --output outputs/ntu_poses --classes relevant
```

All three converge on the same `.npz` format: `keypoints (T, M, 17, 2)`
COCO-order pixel coordinates plus `scores (T, M, 17)` confidences.

## Roadmap

- [ ] **Phase 1 — Baseline:** pose-extraction pipeline (YOLO-Pose) + ST-GCN baseline on UT-Interaction / RWF-2000
- [ ] **Phase 2 — Bullying10K:** DVS events → pseudo-frames → poses, or the dataset's provided COCO pose labels
- [ ] **Phase 3 — NTU mutual actions:** relevant-class subset, 3D → 2D projection, added to the unified training set
- [ ] **Phase 4 — Unified model:** cross-dataset evaluation, per-dataset ablations, aggressive-vs-neutral confusion analysis
- [ ] **Phase 5 (stretch):** temporal localization — *when* in a stream an incident occurs

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
