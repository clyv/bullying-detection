# Early warning: running it, and what to do with a hit

The design doc says what the system is. This says what to do on a Tuesday.

It assumes you have read §8.1 of [EARLY_WARNING_DESIGN.md](EARLY_WARNING_DESIGN.md):
on UT-Interaction seq1 the hazard scored a handshake above a punch. Everything
below is written on the assumption that the current scores are **mostly a
proximity detector**, and the job is to find out how much more than that they are.

## 1. One video

```
python -m src.early_warning.tracked_pose --video <video> --output outputs/<name>.npz
python -m src.early_warning.replay       --poses outputs/<name>.npz
```

Roughly one minute of compute per minute of footage. `--process-fps 5` halves
that; the social signals do not need 30 fps. `--max-people 12` is the slot
count — raise it for a crowded corridor, and check `n_tracks` in the output,
because identity churn is the failure mode that quietly destroys every
interaction feature.

## 2. A folder of videos

```
python -m src.early_warning.run_batch \
    --videos data/ubi_fights/UBI_FIGHTS/videos \
    --annotations data/ubi_fights/UBI_FIGHTS/annotation \
    --out outputs/ew_batch --pattern "F_*" --limit 40
```

Writes `outputs/ew_batch/`:

| path | what it holds |
|---|---|
| `poses/<id>.npz` | tracked skeletons, reused on the next run |
| `scores/<id>.npz` | `hazard_5s`, `hazard_10s`, `quality` per step |
| `ledger.jsonl` | one row per video: peaks, onset, lead times, reasons at the alarm |
| `summary.json` | the aggregate, plus a review queue |
| `cases/<id>.json` | annotation stubs you have started correcting |

The run is **resumable** — a video whose pose file exists is skipped, so an
interrupted batch costs nothing. Run it once per `--pattern` to cover both
fight and normal footage; rows append to the same ledger.

`--annotations` is optional. Without it, every video is `has_assault: null`
and is excluded from all statistics, because *unlabelled* and *quiet* are
different claims. You still get the ledger and the timelines.

## 3. Reading the result

A real run over 8 UBI-Fights fight videos and 8 normal ones:

```
16 usable videos
  6/8 assaults had a WARN at least 1s early; lead p25/median/p75 5.1/24.4/51.3s (max 94.6s)
  median share of each clip already above WARN: 27% of assault videos, 0% of quiet ones
  on 0.089h with no assault: 0.0/h false WARN, 0.0/h false WATCH
  at a 0.0/h budget (threshold 0.23): 100% anticipated, median lead 12.9s
  [!] 0.089h of quiet footage can only resolve rates down to 11.2/h, so the budget
      threshold above is fitted to noise. Score more negatives before believing any of it.
```

This is much better than UT-Interaction suggested, and still not a result. Five
of the eight normal clips peaked at 0.0 — the hazard is genuinely silent on
ordinary CCTV, which seq1's staged squaring-up could never have shown. But 5.3
minutes of quiet footage is consistent with a false-alarm rate anywhere up to
~11/hour, so "zero false alarms" means "we did not look for long enough".

Read it in this order, and stop at the first line that fails:

1. **False alarms per hour.** If this is above the budget, nothing else on the
   page means anything. The budget starting point is one false WARN per camera
   per school day, about 0.15/hour. A system at 50/hour is not an early warning,
   it is a light that is always on.
2. **Anticipation at budget.** The share of assaults flagged at least 1 s early
   *at the threshold that keeps false alarms inside the budget*. This is the
   headline. A high recall at an unaffordable threshold is not a result.
3. **The lead-time distribution**, never its mean. Some assaults have no visible
   build-up and will always score zero.
4. **Long leads are a red flag, not a prize.** The 94.6-second lead above came
   from a 144-second video whose alarm was on from 17 s — that is a light left
   on, not a prediction. The ledger reports `frac_above_warn` per video and
   `lead_fraction` in the review queue for exactly this; a lead that is a large
   share of the clip is unearned. The median fight clip sat above WARN for 27%
   of its length, so most of these leads are partly that artefact.

## 4. When it fires before an assault

**Do not treat a single hit as evidence.** With a score this correlated with
"two people are near each other", some clips will fire early by luck. One hit is
a lead for review; the aggregate in §3 is the result.

Take the review queue from `summary.json`, then for each candidate:

```
python -m src.early_warning.run_batch --out outputs/ew_batch --case F_0_1_0_0_0
```

That writes `cases/F_0_1_0_0_0.json`, pre-filled with the system's own claim —
assault spans from the annotation, guessed precursor and build-up boundaries,
the reasons it gave, and `phase_annotated: false`.

The guesses are often visibly wrong, which is the point. One real stub proposed
a build-up running from 6 s to 64 s; nothing resembling build-up lasts a minute.
Deleting that is five seconds of work, and much faster than judging an empty file.

**Now watch the video**, starting ten seconds before `watch_at_s`, and answer
one question: *would a person walking past have seen this coming, from the
behaviour the system named?* Then edit the stub:

- Correct `phases` to where the precursor and build-up actually start. If there
  was no visible build-up, delete the phases and leave the list empty — that is
  a real and useful answer.
- Add any non-assault confrontation to `friendly`, especially **arguments that
  fizzled out**. These are the most valuable footage you have, because they are
  the difference between a false alarm and a justified alert, and no public
  dataset contains them.
- Set `phase_annotated: true` **only once a human has checked it.** An
  uncorrected stub must never enter the training set.

Record the verdict too, in your own words, in the `_guessed.review` field:
whether the alarm tracked genuine build-up, tracked mere proximity, or was
coincidence. Coincidences are data. A folder of 30 honestly-labelled cases where
20 are "proximity" tells you more than 200 unexamined hits.

Corrected stubs load straight back:

```python
from src.early_warning.anticipation_labels import load_phase_annotation
ev = load_phase_annotation("outputs/ew_batch/cases/F_0_1_0_0_0.json")
```

## 5. What comes next

In order, and each one gated on the last:

1. **Fix the two defects seq1 exposed** before scaling anything up. Gesture
   features are gated on close range, so pointing scores zero and phase 1 is
   invisible. And the aggressive-minus-benign margin over coarse geometry was
   +0.118, which fails the Stage 0 criterion in §6 of the design doc.
2. **Run the full UBI-Fights batch** — 216 fight and 784 normal videos — to get
   a false-alarm rate with a real denominator. This is the number that decides
   whether the idea is viable at all. Expect it to take several hours.
3. **Train `WindowHazard`** leave-one-dataset-out and check it beats
   `geometry_hazard`, motion energy, min pair distance and largest group size.
   If it does not beat coarse geometry, the features are the problem, not the model.
4. **Annotate 150–300 videos** using §4, with a second annotator on a subset so
   you can report kappa. Phase boundaries are subjective and a single annotator
   cannot show otherwise.
5. Only then `EscalationNet`, and only then shadow mode.

## 6. What not to do with a hit

- **Do not tune the thresholds to make a case look good.** Thresholds come from
  a false-alarm budget on the camera's own normal footage
  (`metrics.threshold_for_budget`), fitted before you look at the assaults.
- **Do not report a lead time without the false-alarm rate beside it.** A model
  that fires constantly has excellent lead times.
- **Do not describe a hit in terms of intent.** "Three people closed to within a
  metre and squared up" is observable and is the compliance argument for EU AI
  Act Art. 5(1)(f). "They looked angry" is an emotion inference in an
  educational setting, and is prohibited.
- **Do not build a per-student record**, however tempting a longitudinal case
  file looks. Art. 5(1)(d) prohibits predicting offending from profiling. The
  system scores situations; track ids die with the clip.
