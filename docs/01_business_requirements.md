# Business Requirements — Training ROI Predictor

## 1. Problem Statement
Most L&D ROI dashboards show pre/post performance averages. They confuse two effects: who *was selected* for training (selection bias) and how much they *benefited* (treatment effect). Leadership wants a per-employee estimate of expected uplift from a given training intervention — and a population view to plan budget. The right framing is **causal uplift / CATE**, not regression on observed outcomes.

## 2. Stakeholders
| Role | Interest | Success criterion |
|---|---|---|
| CHRO | Allocate L&D budget where it earns the most uplift | Top-30% policy uplift gain >= 1.8× |
| HRBP | Pre-screen training nominations | CATE + CI per employee in API |
| Finance | Defend L&D spend | Calibrated CATE across bins |
| Audit / Ethics | No discriminatory targeting | No protected attributes in features |

## 3. Business Objectives
1. Per-employee **CATE estimate** for a chosen training intervention.
2. **90% bootstrap confidence interval** around CATE (so HRBPs see uncertainty).
3. **Qini AUC >= 0.22** on a holdout simulation with known ground-truth uplift.

## 4. KPIs
| KPI | Definition | Target |
|---|---|---|
| Qini AUC | Area under the Qini curve | >= 0.22 |
| MAE on synthetic CATE | Mean abs error vs ground truth | < 0.20 |
| Top-30% policy uplift | Avg CATE in top-30% / random 30% | >= 1.8× |
| Calibration slope | Linear slope of CATE vs binned realised uplift | 0.9 - 1.1 |

## 5. Scope
**In scope:** 5,000 synthetic employee+training records with known ground-truth CATE for evaluation; X-learner with GradientBoostingRegressor + LogisticRegression propensity.
**Out of scope:** dynamic treatment effects (sequence of trainings); spillover effects between employees.

## 6. Constraints / assumptions
- **SUTVA** (no interference between units; one version of treatment) — assumed; re-stated in `docs/03_methodology.md`.
- **Conditional ignorability** — assumed via the synthetic data design; in production this requires a documented identification strategy.
- **Common support** — propensity bounded away from 0 and 1; trim if needed.

## 7. Risks
| Risk | Mitigation |
|---|---|
| Misuse for nomination decisions | UI banner: advisory only |
| Propensity collapse (extreme imbalance) | Trim or use overlap weights |
| Mistaking correlation for uplift | X-learner + identifiability section in docs |
