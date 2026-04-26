# Methodology — Training ROI Predictor (X-learner)

We estimate the **Conditional Average Treatment Effect** (CATE):

$$ \tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x] $$

where $Y(1)$ is the potential outcome under training and $Y(0)$ without. Our outcome is `post_perf - pre_perf` (the realised uplift); we model the *expected* uplift conditional on the employee's features.

We use the **X-learner** of Künzel et al. (2019), which is well-behaved when treated and control sample sizes are imbalanced — a typical L&D situation, since fewer employees are nominated to a given course than the universe of eligible employees.

---

## 1. Identifiability assumptions

The CATE is only identified from observational data under three assumptions:

### (i) SUTVA (Stable Unit Treatment Value)
- **No interference**: an employee's outcome under training is unaffected by which other employees are trained.
- **No hidden versions**: there's a single, well-defined version of "the training intervention" — no per-cohort heterogeneity in treatment quality.

### (ii) Conditional ignorability (unconfoundedness)
$$ \{Y(0), Y(1)\} \perp T \mid X $$

i.e. given the observed features $X$, treatment assignment is as good as random. This is the strongest assumption — it's untestable from data alone. We mitigate by:
- including the most plausibly-confounding covariates (`pre_perf`, `tenure_yrs`, `role_level`, `dept`),
- documenting any proxy concerns for unobserved confounders (motivation, manager advocacy) in the data card.

### (iii) Common support / positivity
$$ 0 < g(x) := \Pr(T = 1 \mid X = x) < 1 \quad \text{for all } x $$

We fit $g(x)$ as a logistic regression on $X$ and **trim** units with $g(x)$ outside $[0.05, 0.95]$ before computing the X-learner stage-2 weights.

---

## 2. The X-learner

**Stage 1 — outcome models on each arm:**

$$
\mu_0(x) = \mathbb{E}[Y \mid X = x, T = 0], \qquad
\mu_1(x) = \mathbb{E}[Y \mid X = x, T = 1].
$$

We fit two separate `GradientBoostingRegressor` instances on the control and treated subsets respectively.

**Stage 2 — imputed individual treatment effects on each arm:**

For treated units we use $\mu_0$ to fill in the counterfactual:

$$ \tilde\tau_i^{(1)} = Y_i - \mu_0(X_i) \qquad (\text{for } i: T_i = 1). $$

For control units we use $\mu_1$:

$$ \tilde\tau_i^{(0)} = \mu_1(X_i) - Y_i \qquad (\text{for } i: T_i = 0). $$

**Stage 3 — regress imputed effects on covariates:**

$$
\tau_1(x) = \mathbb{E}[\tilde\tau^{(1)} \mid X = x], \qquad
\tau_0(x) = \mathbb{E}[\tilde\tau^{(0)} \mid X = x],
$$

each fit with another `GradientBoostingRegressor`.

**Stage 4 — propensity-weighted blend:**

$$ \hat\tau(x) = g(x) \cdot \tau_0(x) + (1 - g(x)) \cdot \tau_1(x). $$

The blend weights $g(x)$ and $1 - g(x)$ are intentionally inverted relative to standard IPW: $\tau_0$ is more reliable in regions where treated units are abundant (so we trust $\tau_0$ more when $g(x)$ is high), and vice versa. Künzel et al. show this gives smaller MSE than the T-learner when arm sizes are imbalanced.

---

## 3. Confidence intervals — bootstrap

For the API we report a **percentile bootstrap 90% CI** for $\hat\tau(x)$:

1. Sample with replacement $B$ training datasets.
2. Refit the full X-learner stack on each.
3. Score the query employee $x$.
4. Take the 5th and 95th percentiles of the $B$ CATE estimates.

In the demo $B = 50$ (CPU-bound trade-off); production should use $B \geq 200$.

## 4. Evaluation
| Metric | Why |
|---|---|
| Qini AUC | Standard uplift ranking metric |
| MAE vs ground-truth CATE | Available because the synthetic generator knows the truth |
| Top-K policy uplift | Practical: how much extra uplift if we train top-K by CATE |
| Calibration slope | CATE bin vs realised uplift slope, target ~1 |

## 5. References
- Künzel, Sekhon, Bickel, Yu, *Metalearners for estimating heterogeneous treatment effects*, PNAS 2019.
- Pearl, *Causality*, 2009.
- Imbens & Rubin, *Causal Inference for Statistics, Social and Biomedical Sciences*, 2015.
- Athey & Wager, *Estimating Treatment Effects with Causal Forests*, 2019.
- Radcliffe, *Using control groups to target on predicted lift* (Qini), 2007.
