Reproduction bundle for CUHK-X Competition Small Model Track

Included:
- scripts/*.py
- src/**/*.py
- external_models/*.pt and supporting docs

Main run used for the 0.77611 submission:
python scripts\finetune_yolo_r2p1d_date_aug.py --cache-dir cache\yolo_r2p1d_v1 --checkpoint external_models\ensemble_packed.pt --model-index 0 --mode head --epochs 6 --lr 0.0002 --seed 9301 --split-mode date --val-dates 2025-06-12,2025-06-13 --balanced --output-prefix finetune_yolo_public_domain_head_m0_balanced_s9301_e6_lr2e4

Fusion and submission steps:
python scripts\build_anchor_gated_candidates.py --tag bestanchor_pubm01_balanced_s9301_v3
python scripts\build_score_posterior_candidates.py --tag bestanchor_pubm01_balanced_s9301_v3 --support-pool outputs\anchor_gated_candidate_pool_bestanchor_pubm01_balanced_s9301_v3.csv
python scripts\build_bestanchor_consensus_candidates.py --tag bestanchor_pubm01_balanced_s9301_v3

Final submitted file:
outputs\submission_score_posterior_top40_bestanchor_pubm01_balanced_s9301_v3.csv
