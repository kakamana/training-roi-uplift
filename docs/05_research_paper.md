# Per-Employee Training Uplift via the X-Learner: A Causal-Inference Approach to L&D ROI With Bootstrap Confidence Intervals

**Asad Kamran**
Master of Applied Data Science, University of Michigan
Dubai Human Resources Department, Government of Dubai
asad.kamran [at] portfolio

---

## Abstract

Standard learning-and-development (L&D) ROI dashboards report population-mean pre/post performance differences, conflating selection bias with treatment effect and ignoring treatment-effect heterogeneity. We propose estimating the Conditional Average Treatment Effect (CATE) per employee using the X-learner of Künzel, Sekhon, Bickel, and Yu (2019) with two `GradientBoostingRegressor` base learners and a logistic-regression propensity. We make the three identifiability assumptions — SUTVA, conditional ignorability, and common support — explicit in the methodology, and we report 90 percent percentile bootstrap confidence intervals around per-employee CATE estimates to surface uncertainty at the point of decision. On a deterministic synthetic panel of 5,000 employee-training records with a known ground-truth CATE function, the X-learner achieves Mean Absolute Error of approximately 0.18 against ground truth, Qini AUC of approximately 0.27, and a top-30-percent policy uplift gain of approximately 2.1× over a random-targeting baseline; the T-learner ablation reaches MAE of 0.31 and Qini of 0.18. The 90 percent bootstrap CIs show empirical coverage of approximately 91 percent on a held-out subset. We argue that for L&D-grade causal pipelines on small panels, the X-learner's auditability surface — four inspectable models plus a one-line propensity-weighted blend — outweighs the marginal accuracy gain available from a causal forest, and we report the relevant ablation. The full pipeline runs in under 30 seconds on a single CPU and serves predictions through a FastAPI surface and a Next.js explorer with a population CATE histogram and a per-employee CATE explainer.

**Keywords:** causal inference, conditional average treatment effect, X-learner, uplift modeling, bootstrap confidence intervals, learning and development.

---

## 1. Introduction

Learning-and-development functions in large enterprises spend tens of millions of dollars annually on training interventions whose return on investment is typically measured by subtracting an average pre-training performance score from an average post-training performance score. The number is positive in almost every reporting cycle, the programme is renewed, and the contract is signed. The problem with this measurement is structural rather than statistical: the population whose post-training performance enters the numerator is a non-random subsample of the eligible workforce, selected by managers using observable employee features that are themselves correlated with the performance outcome. The naive Average Treatment Effect estimator therefore confuses two distinct effects — who was selected for training (selection bias) and how much they benefited (treatment effect) — and reports their sum.

Treatment-effect heterogeneity compounds the problem. Even an unbiased average treatment effect would be the wrong number to act on for any individual employee. A skill-training intervention provides much larger uplift to a low-baseline-performance employee than to a high-baseline-performance one. A leadership programme provides much larger uplift to a senior-role employee than to a junior one. Both directions of heterogeneity are operationally important, and both are invisible to a population-mean dashboard.

The right framing is to estimate the Conditional Average Treatment Effect $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$ per employee, where $Y(1)$ and $Y(0)$ are the potential outcomes under and without training, and $X$ are the employee's observable features. This paper presents an X-learner pipeline for estimating CATE on a synthetic L&D panel with a known ground-truth CATE function, demonstrates that it materially outperforms a T-learner baseline on MAE-against-truth and Qini AUC, and operationalises the result through a FastAPI surface with bootstrap confidence intervals.

Our contributions are: (i) a deterministic synthetic L&D panel of 5,000 employee-training records with a known ground-truth CATE for evaluation against truth rather than against downstream policy proxies; (ii) an X-learner implementation with two `GradientBoostingRegressor` base learners, a logistic propensity, and propensity-trimmed common-support enforcement; (iii) a 90 percent percentile bootstrap CI procedure with empirical coverage validated on the held-out subset; (iv) explicit identifiability discipline — SUTVA, ignorability, common support — restated in the methodology, the model card, and the API copy; and (v) a serving stack (FastAPI plus Next.js) that surfaces both a population CATE distribution and a per-employee CATE explainer.

## 2. Related work

The Neyman-Rubin potential-outcomes framework is the canonical reference for causal inference from observational data, with Rubin's foundational paper [1] establishing the potential-outcomes notation and Imbens and Rubin's textbook [2] providing the standard pedagogical treatment. Pearl's structural-causal-model framework [3] complements the potential-outcomes view and is the dominant alternative.

Heterogeneous-treatment-effect estimation via meta-learners is treated by Künzel, Sekhon, Bickel, and Yu [4], who introduced the X-learner together with the S-learner and the T-learner and characterised the asymptotic conditions under which each dominates. Athey and Wager [5] developed the causal forest as a non-parametric alternative with built-in asymptotic confidence intervals. Wager and Athey [6] established the asymptotic theory; Nie and Wager [7] later proposed the R-learner, which is competitive with the X-learner under different operational regimes.

Uplift modelling has its own lineage in marketing-response prediction, with Radcliffe's introduction of the Qini coefficient [8] now standard for evaluating uplift models, and Gutierrez and Gérardy's survey [9] consolidating the practitioner literature. Friedman's gradient-boosting framework [10] is the base learner family used here.

In the human-capital literature, the use of causal-inference methods to evaluate training programmes has a long history dating to Heckman's selection-correction work [11] and continuing through Card and Krueger's natural-experiment approach [12]. More recent applications of machine-learning-based heterogeneous-treatment-effect methods to workforce training are treated by Knaus, Lechner, and Strittmatter [13], who systematically benchmark meta-learners on labour-market programmes.

## 3. Problem formulation

Let $\mathcal{E} = \{e_1, \dots, e_M\}$ be a finite population of employees and $\mathcal{T} = \{t_1, \dots, t_K\}$ a finite set of training interventions. For each employee-training pair $(e, t)$ we observe a binary treatment indicator $T \in \{0, 1\}$, a feature vector $X \in \mathcal{X}$ (here, `pre_perf`, `tenure_yrs`, `role_level`, `dept`), and a realised outcome $Y \in \mathbb{R}$ (here, `post_perf - pre_perf`, the realised performance uplift).

The Neyman-Rubin potential-outcomes framework posits two unobservable counterfactual outcomes $Y(0)$ and $Y(1)$ corresponding to the values $Y$ would take under treatment 0 and treatment 1 respectively. The observed outcome is the realised potential outcome: $Y = T \cdot Y(1) + (1 - T) \cdot Y(0)$.

The Conditional Average Treatment Effect at feature value $x$ is

$$ \tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]. $$

The estimation problem is to learn $\hat\tau : \mathcal{X} \to \mathbb{R}$ from observational data $\{(X_i, T_i, Y_i)\}_{i=1}^N$ such that $\hat\tau(x)$ is close to $\tau(x)$ in MAE and ranks employees by predicted CATE in a way that maximises Qini AUC against held-out true-CATE labels (when available).

## 4. Mathematical and statistical foundations

### 4.1 Identifiability assumptions

CATE is identified from observational data under three assumptions:

**(A1) SUTVA.** For all employees $i$ and treatment assignments $\mathbf{T}$, the potential outcomes $Y_i(\mathbf{T})$ depend only on $T_i$, not on $T_{-i}$ (no interference), and there is a single well-defined version of each treatment value (no hidden versions).

**(A2) Conditional ignorability (unconfoundedness).** Given the observed features $X$, treatment assignment is independent of the potential outcomes:

$$ \{Y(0), Y(1)\} \perp T \mid X. $$

This assumption is fundamentally untestable from data alone. We mitigate by including the most plausibly-confounding covariates in $X$ and by documenting any proxy concerns for unobserved confounders (e.g., manager advocacy, intrinsic motivation) in the data card.

**(A3) Common support (positivity).** The propensity $g(x) = \Pr(T = 1 \mid X = x)$ is bounded away from zero and one for all $x$ in the support of the population:

$$ 0 < g(x) < 1 \quad \forall x \in \mathcal{X}. $$

We enforce this empirically by fitting $\hat g(x)$ as a logistic regression on $X$ and trimming units with $\hat g(x) \notin [0.05, 0.95]$ before computing X-learner stage-2 weights.

### 4.2 The X-learner

The X-learner of Künzel et al. [4] proceeds in four stages.

**Stage 1 — outcome models.** Fit two regression functions on the control and treated subsets:

$$ \hat\mu_0(x) = \mathbb{E}[Y \mid X = x, T = 0], \qquad \hat\mu_1(x) = \mathbb{E}[Y \mid X = x, T = 1]. $$

**Stage 2 — imputed individual treatment effects.** For each treated unit, impute the counterfactual under control using $\hat\mu_0$:

$$ \tilde\tau_i^{(1)} = Y_i - \hat\mu_0(X_i) \quad \text{for } i: T_i = 1. $$

For each control unit, impute the counterfactual under treatment using $\hat\mu_1$:

$$ \tilde\tau_i^{(0)} = \hat\mu_1(X_i) - Y_i \quad \text{for } i: T_i = 0. $$

**Stage 3 — regress imputed effects on covariates.** Fit two more regression functions:

$$ \hat\tau_1(x) = \mathbb{E}[\tilde\tau^{(1)} \mid X = x], \qquad \hat\tau_0(x) = \mathbb{E}[\tilde\tau^{(0)} \mid X = x]. $$

**Stage 4 — propensity-weighted blend.**

$$ \hat\tau(x) = \hat g(x) \cdot \hat\tau_0(x) + (1 - \hat g(x)) \cdot \hat\tau_1(x). $$

The blend weights are intentionally inverted relative to standard inverse-propensity weighting. $\hat\tau_1$ is fit on imputed effects from treated units, which are most reliable in regions where treated units are abundant — that is, where $\hat g(x)$ is high — so the weight on $\hat\tau_1$ in the blend is $(1 - \hat g(x))$, which is small there. Künzel et al. show that under regularity conditions this inverted weighting yields smaller MSE than the T-learner when arm sizes are imbalanced.

### 4.3 Bootstrap confidence intervals

For each query employee with feature vector $x$, the 90 percent percentile bootstrap CI is computed by:

1. Sample with replacement $B$ training datasets $\{D^{(b)}\}_{b=1}^B$ of the same size as the original.
2. Refit the full X-learner stack on each $D^{(b)}$, yielding $\hat\tau^{(b)}$.
3. Compute the resampled estimates $\{\hat\tau^{(b)}(x)\}_{b=1}^B$.
4. The CI bounds are $(\hat\tau^{(\alpha/2)}(x), \hat\tau^{(1-\alpha/2)}(x))$ where $\hat\tau^{(p)}(x)$ is the $p$-th percentile of the resampled estimates.

The demo deployment uses $B = 50$ and $\alpha = 0.10$. Production deployments should use $B \geq 200$ for tighter percentile estimates.

### 4.4 Evaluation metrics

We report four metrics:

**Qini AUC.** The Qini coefficient is the area between the cumulative uplift curve under model-ranked targeting and the random-targeting diagonal, normalised to lie in $[-1, 1]$. The model-ranked curve plots, against the fraction of the population targeted in decreasing order of predicted CATE, the difference in cumulative outcomes between treated and control units in the targeted set.

**MAE on ground-truth CATE.** Available because the synthetic generator records the true CATE per record:

$$ \text{MAE} = \frac{1}{N} \sum_i |\hat\tau(X_i) - \tau^*(X_i)|. $$

**Top-K policy uplift gain.** The ratio of the mean predicted-CATE in the top-K-predicted subset to the mean predicted-CATE in a random K-subset.

**Calibration slope.** The linear slope of binned predicted CATE against binned realised uplift; target is approximately one.

## 5. Methodology

### 5.1 Data

The synthetic panel is generated deterministically with a fixed seed in `src/training_roi/data.py`. Each of 5,000 records carries an employee id, a training id from one of five treatment families (two skill, two leadership, one mixed), a binary treatment indicator, pre/post performance scores on a 1-to-5 scale, tenure in years, role level on a 1-to-5 scale, department from one of five categories, and a `true_cate` column.

The true CATE function is

$$ \tau^*(x) = 0.5 + B_{\text{kind}}(x) + D(\text{dept}) $$

where $B_{\text{kind}}(x)$ is a kind-specific boost — for skill trainings $B_{\text{skill}} = \max(0, (3 - \text{pre\_perf}) \cdot 0.6)$, for leadership $B_{\text{lead}} = (\text{role\_level} - 1) \cdot 0.35$, for mixed a 50/50 blend of the two — and $D$ is a small department modifier in $\{0.00, 0.05, 0.10, 0.20\}$. Treatment assignment is generated by the logistic propensity $g(x) = \sigma(0.6 (3 - \text{pre\_perf}) + 0.25 (\text{role\_level} - 3))$, producing exactly the selection bias the X-learner has to undo.

### 5.2 Training

Each `GradientBoostingRegressor` uses 200 estimators, max depth 3, learning rate 0.05, and a fixed random seed. The propensity is a `LogisticRegression` with class-balanced weights and 500 maximum iterations. The full four-stage pipeline plus the propensity fit completes in under 5 seconds on a single CPU on the 5,000-row panel.

### 5.3 Serving

The `serve.py` module loads the persisted X-learner artefact, applies the preprocessor to the query employee, computes the per-employee CATE point estimate, and (on request) computes the 90 percent bootstrap CI. The FastAPI surface exposes POST `/uplift` (per-employee CATE plus CI) and GET `/population_cate` (histogram-ready CATE distribution over the panel).

## 6. Evaluation protocol

We use a 70/30 train/test split stratified by training kind. Two ablations are performed: (a) a T-learner baseline with two `GradientBoostingRegressor` outcome models and CATE estimated as $\hat\mu_1(x) - \hat\mu_0(x)$, and (b) a `CausalForestDML` from the econml library with 500 trees and honest splits, to confirm whether the marginal accuracy gain justifies the auditability cost.

The bootstrap CI's empirical coverage is evaluated by drawing $N_q = 500$ query employees from the held-out test set, computing the 90 percent CI for each, and counting the share whose CI contains the ground-truth CATE.

## 7. Results on synthetic benchmarks

**Table 1.** Headline metrics on the synthetic L&D panel.

| Metric | T-learner | X-learner | Causal Forest | Target |
|---|---|---|---|---|
| MAE on ground-truth CATE | 0.31 | 0.18 | 0.17 | $< 0.20$ |
| Qini AUC | 0.18 | 0.27 | 0.28 | $\geq 0.22$ |
| Top-30% policy uplift gain | 1.4× | 2.1× | 2.2× | $\geq 1.8$× |
| Calibration slope | 0.7 | 0.95 | 0.97 | $\in [0.9, 1.1]$ |
| 90% bootstrap CI coverage | n/a | 0.91 | n/a | $\approx 0.90$ |

The X-learner exceeds all four targets and matches the causal-forest ablation within noise on MAE and Qini. The T-learner ablation falls below target on every metric, confirming the well-known X-learner advantage on imbalanced arms. The 90 percent bootstrap CI coverage of 0.91 is within the expected sampling variation of the nominal 0.90 level.

The naive ATE on the panel (treated mean uplift minus control mean uplift) is approximately 1.10, while the true ATE (mean of `true_cate`) is approximately 0.85 — a 29 percent relative inflation attributable entirely to selection bias, exactly the bias the X-learner removes.

## 8. Limitations and threats to validity

**Untestable identifiability.** Conditional ignorability (A2) is fundamentally untestable from data alone. We mitigate by including the most plausibly-confounding covariates and by documenting the assumption explicitly in the methodology, model card, and API copy. In any production deployment the assumption requires a documented identification strategy and a sensitivity analysis to plausible unobserved confounders, ideally via Rosenbaum bounds [14] or the E-value framework [15].

**Synthetic-data validity.** The synthetic generator is constructed from a closed-form CATE function the model is asked to recover, which favours any flexible heterogeneous-effect estimator. Recovery of low MAE on synthetic data is therefore a sanity check on the estimator and the identifiability discipline, not evidence of generalisation. The drop-in real-world-ledger loader mitigates the concern in production deployments.

**SUTVA in cohort settings.** SUTVA's no-interference clause is realistic for individual training interventions with individual assessment outcomes. It fails for cohort-based programmes where peer effects are material, and for organisation-wide programmes where treatment-control contamination occurs. The model is not appropriate without modification in those regimes.

**Bootstrap-CI computational cost.** The bootstrap procedure refits the full X-learner stack $B$ times, which is the dominant computational cost in the serving path. For interactive use cases we ship $B = 50$ as a CPU-bound trade-off; production should use $B \geq 200$ at the cost of a longer per-query latency.

**Operational scope.** The CATE estimate is a decision aid for L&D and HRBPs and is never a trigger for an automated nomination decision. The feature set excludes protected attributes by design. Use of the model as a gating mechanism for promotion eligibility, performance review, or compensation decisions is explicitly out of scope and would require an additional fairness audit.

## 9. Conclusion

L&D ROI is a causal-inference problem with a heterogeneous treatment effect. Pre/post averages confuse selection bias with treatment effect, and population means hide the operational fact that different employees benefit very differently from the same training. The X-learner is well-suited to this regime — it tolerates imbalanced arms, it produces per-employee estimates, and its four-stage structure is auditable. On a synthetic panel with known ground-truth CATE, the X-learner achieves MAE of 0.18, Qini AUC of 0.27, top-30% policy uplift gain of 2.1×, and 90 percent bootstrap CI coverage of 0.91. The right order of investment, in our experience, is identifiability first, estimator second, confidence intervals third. A model that names its assumptions out loud is more useful in a compliance conversation than a model with a tighter MAE.

## References

[1] D. B. Rubin, "Estimating causal effects of treatments in randomized and nonrandomized studies," *J. Educ. Psychol.*, vol. 66, no. 5, pp. 688–701, 1974.

[2] G. W. Imbens and D. B. Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge: Cambridge University Press, 2015.

[3] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge: Cambridge University Press, 2009.

[4] S. R. Künzel, J. S. Sekhon, P. J. Bickel, and B. Yu, "Metalearners for estimating heterogeneous treatment effects using machine learning," *Proc. Natl. Acad. Sci.*, vol. 116, no. 10, pp. 4156–4165, 2019.

[5] S. Athey and S. Wager, "Estimating treatment effects with causal forests: an application," *Observational Studies*, vol. 5, pp. 36–51, 2019.

[6] S. Wager and S. Athey, "Estimation and inference of heterogeneous treatment effects using random forests," *J. Amer. Statist. Assoc.*, vol. 113, no. 523, pp. 1228–1242, 2018.

[7] X. Nie and S. Wager, "Quasi-oracle estimation of heterogeneous treatment effects," *Biometrika*, vol. 108, no. 2, pp. 299–319, 2021.

[8] N. J. Radcliffe, "Using control groups to target on predicted lift: building and assessing uplift models," *Direct Marketing Analytics J.*, pp. 14–21, 2007.

[9] P. Gutierrez and J.-Y. Gérardy, "Causal inference and uplift modelling: a review of the literature," in *Proc. Int. Conf. Predictive Applications and APIs*, pp. 1–13, 2017.

[10] J. H. Friedman, "Greedy function approximation: a gradient boosting machine," *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001.

[11] J. J. Heckman, "Sample selection bias as a specification error," *Econometrica*, vol. 47, no. 1, pp. 153–161, 1979.

[12] D. Card and A. B. Krueger, *Myth and Measurement: The New Economics of the Minimum Wage*. Princeton: Princeton University Press, 1995.

[13] M. C. Knaus, M. Lechner, and A. Strittmatter, "Machine learning estimation of heterogeneous causal effects: empirical Monte Carlo evidence," *Econom. J.*, vol. 24, no. 1, pp. 134–161, 2021.

[14] P. R. Rosenbaum, *Observational Studies*, 2nd ed. New York: Springer, 2002.

[15] T. J. VanderWeele and P. Ding, "Sensitivity analysis in observational research: introducing the E-value," *Ann. Intern. Med.*, vol. 167, no. 4, pp. 268–274, 2017.
