# K-KUNO E290 model card

## Summary

E290 is an equal-logit ensemble of two R(2+1)D-34 video classifiers trained on
fixed cross-subject folds of the CUHK-X Small Model Track training set. The
classifiers consume 16 frames with four channels: Depth_Color RGB and IR
grayscale. YOLO11n finds a person window from eight endpoint-uniform IR probe
frames. The union of those boxes is enlarged 1.4x, made square in pixel space,
and fixed for the whole clip. Inference averages original and horizontal-flip
logits.

The two classifier states are stored with signed per-output-channel int5 and
int6 weights and dequantized for floating-point inference. Together with the
YOLO checkpoint, learned model assets occupy 93,688,142 bytes, below the Small
Track's 100,000,000-byte limit.

## Evidence

- Public leaderboard: `0.71144 = 143/201` on 2026-07-30.
- Kaggle submission ref: `55124182`.
- Parent subject-fold validation accuracies before lower-bit packing:
  `0.71560` and `0.71870`.
- Final test predictions cover 39 of 40 classes.
- No test labels, manual test annotations, metadata leakage, or pseudo-labels
  were used.

The public notebook is first run privately and its 405 ordered predictions are
compared with the immutable E290 submission. A matching output inherits the
already measured leaderboard result; it must not be submitted unchanged again.

## Intended use

This release is for participants in the CUHK-X Small Model Track and for
non-commercial research permitted by the CUHK-X competition terms. It is an
inference baseline and research handoff, not a general-purpose activity model.
Performance outside the official sensors, subjects, rooms, and action taxonomy
is unknown.

## Data

No CUHK-X competition data or derived frame cache is included. The notebook
reads the official competition attachment at runtime. Users must obtain and use
that data through Kaggle under the organizer's terms.

## Reproduction

Run the public Kaggle notebook with the competition source and the companion
public model dataset attached. Kaggle provides the submission template but the
sensor archives are distributed separately. Accept the organizer-linked gated
Hugging Face dataset, add `HF_TOKEN` as a Kaggle secret, and the notebook will
download the official test ZIP at runtime without printing the token. The
release verification itself uses an equivalent private cache that is never
published. The exact architecture source is included and pinned. The notebook
writes only `submission.csv` and a small aggregate `run_report.json`; the
archive and frame caches remain in `/kaggle/temp`.

## Limitations

- The leaderboard has only 201 public rows, so a single score has substantial
  sampling uncertainty.
- Cross-subject folds are more relevant than random clip splits, but the four
  hidden subjects can still differ materially from every local fold.
- The person crop can fail when IR is unreadable or the person detector misses;
  those clips fall back to the full frame.
- Depth_Color is already colorized depth, not ordinary RGB.
