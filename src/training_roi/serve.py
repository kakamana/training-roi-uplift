"""Inference: per-employee CATE + bootstrap CI."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import models
from .data import load_panel


@lru_cache(maxsize=1)
def _load():
    xl = models.load()
    train = load_panel()
    return xl, train


def estimate_cate(employee_features: dict, training_id: str, n_boot: int = 50) -> dict:
    """Estimate CATE for a single employee+training query, with 90% bootstrap CI."""
    xl, train = _load()
    payload = dict(employee_features)
    payload["training_id"] = training_id
    df = pd.DataFrame([payload])
    point, lo, hi = xl.predict_cate_with_ci(df, train_df=train, n_boot=n_boot, alpha=0.10)
    return dict(
        cate=float(point[0]),
        ci_lower=float(lo[0]),
        ci_upper=float(hi[0]),
        ci_level=0.90,
        n_boot=n_boot,
    )


def estimate_population_cate() -> list[float]:
    """Return predicted CATE over the training panel — feeds the UI histogram."""
    xl, train = _load()
    return xl.predict_cate(train).tolist()
