# Licenses and attribution

## CUHK-X competition data and fine-tuned checkpoint

The notebook does not redistribute CUHK-X data. Participants obtain the data
from the competition page and remain bound by the CUHK-X competition rules and
CUHK-X License v2.0, including its non-commercial restriction and attribution
requirements. The fine-tuned E290 checkpoint is shared only for competition
participation and non-commercial research permitted by those terms.

Please credit the CUHK-X dataset paper and the CUHK AIoT Lab as requested by the
organizers. Competition page:
https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track

## YOLO11n

`yolo11n.pt` is an Ultralytics asset and is distributed under the Ultralytics
AGPL-3.0 terms. See https://www.ultralytics.com/license and
https://github.com/ultralytics/ultralytics.

## IG65M R(2+1)D architecture and initialization lineage

The architecture source comes from `moabitcoin/ig65m-pytorch`, pinned to commit
`fc749e2ee354c3e4ddbb144cf511bb868b008f61`, under the MIT License. The trained
classifiers descend from the public IG-65M to Kinetics-400 initialization.
Repository: https://github.com/moabitcoin/ig65m-pytorch

## Notebook code

Original K-KUNO notebook and inference code are released under Apache-2.0.
Third-party components retain their own licenses.

## Prior public baseline

The YOLO person-crop starting point was the public notebook by Kaggle user
`welshonionman`, `[LB0.667] baseline with YOLO person crop`. K-KUNO rebuilt the
pipeline with fixed subject folds, two-fold lower-bit packing, exact
reproduction checks, and the research summarized in this release.
