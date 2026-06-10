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

## What This Is (and Isn't)

✅ A feasibility study for pose-based aggression detection on CCTV
✅ Privacy-conscious by design — skeletons, not faces
❌ Not a production system — no claims of detecting verbal-only abuse
   or social exclusion, which are invisible to cameras
