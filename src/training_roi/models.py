"""X-learner CATE model + persistence + bootstrap CI helper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .data import load_panel
from .features import build_preprocessor, feature_cols

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PROPENSITY_CLIP = (0.05, 0.95)


@dataclass
class XLearner:
    """X-learner pipeline (sklearn-compatible custom class).

    Stages:
      1. mu0, mu1 — outcome models on each arm.
      2. tau0, tau1 — regress imputed effects on covariates.
      3. propensity g — logistic; blend tau0 and tau1.
    """
    preproc: object
    mu0: GradientBoostingRegressor
    mu1: GradientBoostingRegressor
    tau0: GradientBoostingRegressor
    tau1: GradientBoostingRegressor
    propensity: LogisticRegression
    feature_cols: list[str]

    def predict_cate(self, df: pd.DataFrame) -> np.ndarray:
        X = self.preproc.transform(df[self.feature_cols])
        g = np.clip(
            self.propensity.predict_proba(X)[:, 1],
            PROPENSITY_CLIP[0],
            PROPENSITY_CLIP[1],
        )
        t1 = self.tau1.predict(X)
        t0 = self.tau0.predict(X)
        return g * t0 + (1.0 - g) * t1

    def predict_cate_with_ci(
        self, df: pd.DataFrame, train_df: pd.DataFrame, n_boot: int = 50, alpha: float = 0.10, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bootstrap percentile CI. Returns (point, lo, hi).

        Refits the X-learner stack `n_boot` times on resampled training data; scores `df` each time.
        """
        rng = np.random.default_rng(seed)
        point = self.predict_cate(df)
        boots = np.zeros((n_boot, len(df)), dtype=np.float64)
        n = len(train_df)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            tr = train_df.iloc[idx]
            xl_b = fit_x_learner(tr, base_random_state=seed + b + 1)
            boots[b] = xl_b.predict_cate(df)
        lo = np.percentile(boots, 100 * (alpha / 2), axis=0)
        hi = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)
        return point, lo, hi


def fit_x_learner(df: pd.DataFrame, base_random_state: int = 42) -> XLearner:
    cols = feature_cols(df)
    preproc = build_preprocessor().fit(df[cols])
    X = preproc.transform(df[cols])
    T = df["treatment"].to_numpy().astype(int)
    Y = df["perf_uplift"].to_numpy().astype(float)

    treated = T == 1
    control = ~treated

    mu0 = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=base_random_state,
    ).fit(X[control], Y[control])
    mu1 = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=base_random_state + 1,
    ).fit(X[treated], Y[treated])

    # Imputed individual treatment effects
    tau1_hat = Y[treated] - mu0.predict(X[treated])
    tau0_hat = mu1.predict(X[control]) - Y[control]

    tau1 = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=base_random_state + 2,
    ).fit(X[treated], tau1_hat)
    tau0 = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=base_random_state + 3,
    ).fit(X[control], tau0_hat)

    propensity = LogisticRegression(max_iter=500, class_weight="balanced").fit(X, T)

    return XLearner(
        preproc=preproc,
        mu0=mu0, mu1=mu1,
        tau0=tau0, tau1=tau1,
        propensity=propensity,
        feature_cols=cols,
    )


def save(obj, name: str = "x_learner.joblib") -> Path:
    path = MODEL_DIR / name
    joblib.dump(obj, path)
    return path


def load(name: str = "x_learner.joblib"):
    return joblib.load(MODEL_DIR / name)


def main():
    df = load_panel()
    xl = fit_x_learner(df)
    cate = xl.predict_cate(df)
    print(f"Fitted X-learner. Estimated mean CATE = {cate.mean():.3f}")
    if "true_cate" in df.columns:
        mae = float(np.mean(np.abs(cate - df["true_cate"].to_numpy())))
        print(f"MAE vs ground-truth CATE = {mae:.3f}")
    save(xl)
    print("Saved -> models/x_learner.joblib")


if __name__ == "__main__":
    main()
