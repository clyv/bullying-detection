# Data

**No dataset files are committed to this repository.** NTU RGB+D's release
agreement forbids redistribution, and the other sources are large. Download
each dataset below into this directory — everything under `data/` except this
file is gitignored.

## Expected layout

```
data/
├── bullying10k/          # .npy event files, organised per action class
├── ntu/
│   └── skeletons/        # S001C001P001R001A001.skeleton ... (NTU60 + NTU120)
├── ut_interaction/       # segmented .avi clips (Set 1 + Set 2)
└── rwf2000/              # optional: train/ and val/ with Fight / NonFight
```

## 1. Bullying10K — DVS event camera, physical bullying

- Home: https://www.brain-cog.network/dataset/Bullying10k/
- Code/docs: https://github.com/Brain-Cog-Lab/Bullying10k
- ~10,000 segments, ~255 GB total of `.npy` event files (timestamp, x, y, polarity),
  recorded with a DVS346 sensor (346 x 260). You can download per-action subsets
  rather than the whole corpus.
- Actions: slapping, punching, kicking, strangling, hair grabbing, pushing,
  plus non-violent controls (walking, greeting, handshake, finger-guessing).
- Also ships COCO-format pose labels — usable directly in the unified skeleton
  pipeline as an alternative to extracting poses from pseudo-frames.
- License: CC BY 4.0 — citable and referenceable, but still not committed here.

## 2. NTU RGB+D 120 — 3D skeletons only

- Request access (signed release agreement required):
  https://rose1.ntu.edu.sg/dataset/actionRecognition/
- Download **only the 3D skeleton modality**:
  - "NTU RGB+D" skeletons (~5.8 GB)
  - "NTU RGB+D 120" additional skeletons (~4.5 GB)
- Do **not** download RGB / depth / IR — hundreds of GB and unused here.
- Unzip all `.skeleton` files into `data/ntu/skeletons/`.
- Grab the missing-skeleton lists from https://github.com/shahroudy/NTURGB-D
  (some samples have corrupt or absent tracking and should be skipped).
- This repo only converts the mutual-action subset relevant to aggression
  detection — see the class lists in `src/preprocessing/ntu_skeleton.py`.
- **This data cannot be redistributed in any form.**

## 3. UT-Interaction — surveillance-style RGB confrontations

- Home: https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html
  (Ryoo & Aggarwal, SDHA 2010 challenge; mirrors exist if the site is down)
- 120 segmented videos across 6 classes: shake-hands, point, hug, push, kick, punch.
- Two sets: parking lot (Set 1) and lawn (Set 2), fixed outdoor cameras —
  the closest match to real CCTV framing of the three core datasets.

## 4. RWF-2000 — real-world CCTV violence (optional)

- https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
- 2,000 five-second clips (violent / non-violent) sourced from real surveillance
  footage — the only genuinely messy real-world data in the stack.
- Access requires submitting the request form linked in that repo.
