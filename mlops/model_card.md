# Model Card — Training ROI Predictor (X-learner)

## Intended use
Decision-aid for L&D / HRBPs: per-employee CATE estimate + 90% bootstrap CI for a given training intervention. Inform nominations; never auto-decide.

## Training data
Synthetic 5,000-row panel with known ground-truth CATE (eval-only column). See `data/data_card.md`.

## Model family
- X-learner (Künzel et al., 2019)
- Stage-1 outcome models: GradientBoostingRegressor (n_estimators=200, max_depth=3)
- Stage-3 effect models: GradientBoostingRegressor (same hyperparams)
- Propensity: LogisticRegression with class_weight balanced, clipped to [0.05, 0.95]

## Metrics (target)
| Metric | Target |
|---|---|
| Qini AUC | >= 0.22 |
| MAE vs synthetic CATE | < 0.20 |
| Top-30% policy uplift | >= 1.8× |
| 90% CI coverage | ~ 90% |

## Limitations
- CATE identification rests on conditional ignorability — untestable from data alone.
- Bootstrap CI is approximate; production should use stratified resampling.
- Synthetic ground truth is only a sanity check; real deployments need a randomised pilot to validate.

## Ethical considerations
- No protected attributes in features.
- Disclaimer returned on every API response.
- Use is advisory.

## Retraining
- Monthly. Recheck calibration slope; refit if drifting > 0.1.
