# Notebook 02 — Featurisation

>>> `from training_roi.features import build_design_matrix`

## 1. Numerics + one-hot dept
- StandardScaler on numerics
- OneHotEncoder on `dept`
- (intentionally exclude any protected attributes)

## 2. Propensity model
>>> `g = fit_propensity(X, T)` — LogisticRegression with class_weight balanced.

## 3. Common-support trim
>>> Trim 5th-95th percentile of propensity; report drop count.
