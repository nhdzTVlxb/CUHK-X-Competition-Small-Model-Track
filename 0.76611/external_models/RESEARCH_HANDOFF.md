# Research handoff: where to go after LB 0.711

## What held up

1. **Evaluate by subject.** Random clip splits reward subject and room
   shortcuts. Fixed subject folds are the useful local promotion signal.
2. **Spend pixels on the actor.** A fixed per-clip YOLO window was the strongest
   visual preprocessing lever. A per-frame moving crop would erase motion.
3. **Pretraining matters more than fashionable heads.** The IG-65M to
   Kinetics-400 R(2+1)D-34 backbone remained hard to beat on only ~3k training
   clips.
4. **Diversity can fit under 100 MB.** Packing two folds at int5/int6 retained
   enough accuracy for an equal-logit ensemble while leaving room for YOLO.
5. **Kaggle is calibration, not training data.** The public board has 201 rows;
   many one- or two-row changes are noise. Use a result to choose a new
   hypothesis, not to hand-fit hidden labels.

## What did not clearly break the ceiling

- Saturated blend-weight sweeps and nearby bit-width variants.
- Alternative endpoint/half-bin temporal views: one half-bin variant tied the
  same `0.71144` rather than exceeding it.
- Small motion-linear corrections and repeated post-hoc class adjustments.
- Local gains that appeared only on random clip splits.

A failed implementation rejects those exact bytes and regime, not the whole
idea family.

## Best next shots

1. Add a genuinely different modality with honest subject-CV evidence, such as
   skeleton or radar, then distill or pack only if the joint model still fits.
2. Train crop robustness directly: occasional full-frame and crop-jitter
   augmentation, while keeping the deterministic test crop fixed.
3. Improve temporal diversity with a trained multi-view objective instead of a
   post-hoc sampling tweak.
4. Use per-subject and per-class OOF errors to select a complementary second
   model family, rather than another near-identical R(2+1)D fold.
5. Preserve an untouched design decision before checking any sealed aggregate;
   hidden-set feedback is too scarce to support broad tuning.

The notebook is intentionally simple to fork: preprocessing, unpacking, model
construction, inference, and schema checks are visible in order.
