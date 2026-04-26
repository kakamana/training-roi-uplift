# Evaluation Plan — Training ROI Predictor

## 1. Held-out split
Stratified 70/30 train/test on `treatment` and `dept`.

## 2. Primary scorecard
| Model | Qini AUC | MAE vs true CATE | Top-30% uplift | Calibration slope |
|---|---|---|---|---|
| T-learner (baseline) | – | – | – | – |
| **X-learner (main)** | – | – | – | – |
| Causal Forest (stretch) | – | – | – | – |

## 3. Calibration
- Bin estimated CATE into deciles.
- For each decile: avg predicted CATE vs avg realised uplift in that bin.
- Slope of best-fit line should be near 1.

## 4. Policy simulation
- Sort holdout employees by predicted CATE (desc).
- For each $K$: avg realised uplift among the top-$K$ vs random-$K$.
- Plot Qini curve.

## 5. Subgroup heterogeneity
- Avg CATE by `dept`, `role_level` bucket, `tenure_yrs` bucket.
- Confirms whether the model surfaces meaningful heterogeneity (it should — synthetic data has it baked in).

## 6. Robustness
- Drop `pre_perf` (the strongest predictor) → recompute Qini AUC; degrades, confirming the feature carries signal.
- Inject 10% label noise → measure CATE shift.
- Trim propensity at [0.05, 0.95] vs no trim → Qini delta.

## 7. Bootstrap CI sanity
- For 100 random employees, compute 90% CI width.
- Confirm the *true CATE* lies inside the CI ~90% of the time (since we have ground truth in the synthetic panel).

## 8. Deployment readiness checklist
- [ ] Qini AUC >= 0.22 on holdout
- [ ] MAE on synthetic CATE < 0.20
- [ ] Top-30% policy uplift >= 1.8×
- [ ] CI coverage near nominal 90%
- [ ] /uplift returns CATE + CI
- [ ] UI shows population histogram + single-employee explainer
- [ ] Identifiability section reviewed by audit
