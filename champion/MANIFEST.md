# FROZEN CHAMPION: submission_v14b_sig

- **Leaderboard accuracy:** 0.9438309 (rank 1, Nov 9 2026, about 01:25 local). Previous v14a_sig: 0.9435555.
- **File:** `CHAMPION_submission_v14b_sig.csv` (39,933 rows, columns ID, default; predicted default rate 0.2531).
- **Out-of-fold accuracy of the blend:** 0.94612 (10-fold; +184 net rows over the v12 blend, McNemar p < 0.0001).

## Recipe
1. Features: `features.build_v12` + `local_k_features(Ws=(40,), share=0.08)` + `gap_norm_features(Ws=(60,150))` (76 columns). Script: `work/exp43_build_v14.py . ref`.
2. Models (10-fold, saved stage outputs `work/v14_*`): LightGBM x3 seeds (params in `lgb_best.json`), CatBoost seeds 42 and 7 (params in `cat_best.json`), XGBoost x3 seeds.
3. Blend and gate: `python work/exp44_blend_generic.py . v14b v14 oof_v12.npy` gives weights LGB 0.5 / Cat 0.4 / XGB 0.1, calibration flips skipped (EV < 1 row), signature override applied to 7 test rows.

## Integrity
SHA-256 checksums of the champion file, model outputs, parameters and code are in `SHA256SUMS.txt`.
The leaderboard-measured test default rate (0.24015) was used only in the calibration step, which applied no flips for this file.
Do not edit or overwrite anything listed in the checksum file.
