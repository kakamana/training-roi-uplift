# Data Sources — H8 Training ROI Predictor

## Primary (synthetic)
| # | Source | Notes |
|---|---|---|
| 1 | `src/training_roi/data.py` | Deterministic, seed=42; ground-truth CATE function persisted with the panel |

## Real-world drop-ins
| Source | Use |
|---|---|
| Internal LMS + performance system join | (employee_id, training_id, treatment, pre_perf, post_perf, covariates) |
| World Bank / OECD adult-training panels | Observational benchmark for L&D effects |

## Schema for drop-in
`employee_id, training_id, treatment, pre_perf, post_perf, tenure_yrs, role_level, dept`.

`perf_uplift` is derived as `post_perf - pre_perf`.

## Ethics & identifiability note
For real data the `unconfoundedness` assumption requires a clear identification strategy — randomised pilot, instrumental variable, or a defended set of observed confounders. Document the strategy alongside the model card.
