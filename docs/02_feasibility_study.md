# Feasibility Study — Training ROI Predictor

## 1. Data feasibility
- **Synthetic generator** with a *known ground-truth CATE function* — lets us evaluate against truth, not just observed outcomes. 5,000 rows: `employee_id, training_id, treatment (0/1), pre_perf, post_perf, tenure_yrs, role_level, dept, perf_uplift`.
- **Real-world drop-in:** any L&D ledger with pre/post performance scores + a treatment marker.

## 2. Technical feasibility
- **Algorithmic shortlist**
  - X-learner with two GradientBoostingRegressor base learners + LogisticRegression propensity (main).
  - T-learner (baseline).
  - Causal Forest (stretch — econml).
- **Compute:** 1 CPU; full pipeline < 30 seconds on 5k rows.
- **Serving:** persisted X-learner; bootstrap CI is computed at inference time (50 resamples by default).

## 3. Economic feasibility
| Line item | Monthly cost |
|---|---|
| 1× small container | ~$8 |
| Storage | ~$1 |
| **Total** | **~$9 / mo** |

**Value:** even a 10% lift in L&D-budget-to-uplift ratio at a mid-cap employer dwarfs the ops cost.

## 4. Operational feasibility
- Refit monthly. Bootstrap CI bounds intervention recommendations.
- Output is per-employee CATE + CI; HRBP makes the call.

## 5. Ethical / legal feasibility
- No protected attributes in the feature set (`tenure_yrs`, `role_level`, `dept`, `pre_perf`).
- Recommendation is advisory; never an automated decision.
- `dept` is a coarse organisational category, not a protected attribute, but is documented in the data card.

## 6. Recommendation
**Go.** Cheap, well-bounded methodology, identifiability assumptions explicitly written down. The honest framing about what causal estimates can and can't say is itself a portfolio differentiator.
