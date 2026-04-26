# Data Card — H8 Training ROI Predictor

## Dataset composition

| Layer | Source | Shape | Purpose |
|---|---|---|---|
| Synthetic panel | `src/training_roi/data.py` | 5,000 × 9 | Per-employee+training record with known ground-truth CATE |

## Files
- `data/processed/training_outcomes.parquet`

## Fields
| Column | Type | Notes |
|---|---|---|
| employee_id | str | surrogate key |
| training_id | str | which training intervention (5 distinct) |
| treatment | int (0/1) | did the employee actually take the training |
| pre_perf | float | performance score before window |
| post_perf | float | performance score after window |
| perf_uplift | float | post − pre (the observed outcome) |
| tenure_yrs | float | years at company |
| role_level | int (1..5) | 1 junior … 5 executive |
| dept | str | Sales / R&D / Ops / HR / Finance |
| true_cate | float | (eval-only) ground-truth CATE from the generator |

## Known biases
- Synthetic — selection on observables: higher `role_level` and lower `pre_perf` are mildly more likely to be treated.
- True CATE is heterogeneous in `pre_perf` (lower performers benefit more for skill-based training; higher performers for leadership trainings).

## PII
None. Employee IDs are surrogate keys.

## Reproducing
```bash
python -m training_roi.data
```
Deterministic seed = 42.
