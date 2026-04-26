# Notebook 01 — EDA

>>> `from training_roi.data import make_panel; df = make_panel()`

## 1. Treatment balance
- Treated rate overall and by `dept`, `role_level`.
- Pre-treatment covariate distributions in treated vs control (covariate balance).

## 2. Outcome distributions
- `perf_uplift` histogram by treated vs control.
- Naive ATE (treated mean − control mean) — knowingly biased.

## 3. Propensity sanity
- Logistic on (pre_perf, tenure_yrs, role_level, dept) → AUC.
- Common-support check (overlap of propensity distributions).

## 4. Hypotheses
1. Lower `pre_perf` employees benefit more from skill trainings.
2. Higher `role_level` employees benefit more from leadership trainings.
3. T-learner under-shoots CATE in the smaller arm; X-learner corrects.
