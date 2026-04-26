"""Synthetic employee+training panel with a known ground-truth CATE.

The CATE function is a simple, interpretable mix:
    base = 0.5
    boost from skill-training: lower pre_perf -> larger uplift
    boost from leadership-training: higher role_level -> larger uplift
    dept moderator: small +/- effect
    noise on observed outcomes (but NOT on true_cate)

This lets us evaluate the model against the truth.

Run::

    python -m training_roi.data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED = DATA_DIR / "processed"

DEPTS = ("Sales", "R&D", "Ops", "HR", "Finance")
TRAINING_TYPES = (
    ("T-SKILL-001", "skill"),
    ("T-SKILL-002", "skill"),
    ("T-LEAD-001", "leadership"),
    ("T-LEAD-002", "leadership"),
    ("T-MIXED-001", "mixed"),
)
DEPT_MODIFIER = {"Sales": 0.10, "R&D": 0.20, "Ops": 0.05, "HR": 0.00, "Finance": 0.10}


def _true_cate(pre_perf: float, role_level: int, dept: str, training_kind: str) -> float:
    base = 0.5
    if training_kind == "skill":
        # lower pre_perf -> bigger uplift, capped
        boost = max(0.0, (3.0 - pre_perf) * 0.6)
    elif training_kind == "leadership":
        boost = (role_level - 1) * 0.35
    else:  # mixed
        boost = 0.5 * max(0.0, (3.0 - pre_perf) * 0.6) + 0.5 * (role_level - 1) * 0.35
    return float(base + boost + DEPT_MODIFIER.get(dept, 0.0))


def make_panel(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        emp = f"E-{i:05d}"
        training_id, kind = TRAINING_TYPES[int(rng.integers(0, len(TRAINING_TYPES)))]
        dept = str(rng.choice(DEPTS, p=[0.25, 0.22, 0.20, 0.15, 0.18]))
        role_level = int(rng.integers(1, 6))
        tenure = float(np.clip(rng.gamma(shape=2.0, scale=2.0), 0.1, 25.0))
        pre = float(np.clip(rng.normal(3.0, 0.8), 1.0, 5.0))

        # Selection on observables: lower pre_perf and higher role_level slightly more likely treated
        propensity = 1.0 / (1.0 + np.exp(-(0.6 * (3.0 - pre) + 0.25 * (role_level - 3))))
        treatment = int(rng.random() < propensity)

        true_tau = _true_cate(pre, role_level, dept, kind)
        # Observed post = pre + (treatment * tau) + noise + small natural drift
        natural = float(rng.normal(0.0, 0.25))
        observed_uplift = treatment * (true_tau + rng.normal(0.0, 0.4)) + natural
        post = pre + observed_uplift

        rows.append(dict(
            employee_id=emp,
            training_id=training_id,
            treatment=treatment,
            pre_perf=round(pre, 3),
            post_perf=round(post, 3),
            perf_uplift=round(post - pre, 3),
            tenure_yrs=round(tenure, 2),
            role_level=role_level,
            dept=dept,
            true_cate=round(true_tau, 3),
        ))
    return pd.DataFrame(rows)


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "training_outcomes.parquet")


def make_training_artifacts() -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = make_panel()
    out = PROCESSED / "training_outcomes.parquet"
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    df = make_training_artifacts()
    print(f"wrote {len(df):,} rows -> data/processed/training_outcomes.parquet")
    print("treatment rate:", round(df['treatment'].mean(), 3))
    print("naive ATE (treated mean - control mean uplift):",
          round(df.loc[df.treatment == 1, 'perf_uplift'].mean()
                - df.loc[df.treatment == 0, 'perf_uplift'].mean(), 3))
    print("true ATE (mean true_cate):", round(df['true_cate'].mean(), 3))
