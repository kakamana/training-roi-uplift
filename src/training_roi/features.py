"""Design-matrix builder for X-learner."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["pre_perf", "tenure_yrs", "role_level"]
CATEGORICAL = ["dept", "training_id"]


def build_preprocessor() -> Pipeline:
    numeric = Pipeline(steps=[("scaler", StandardScaler())])
    categorical = Pipeline(
        steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ]
    )
    return Pipeline(steps=[("pre", pre)])


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in NUMERIC + CATEGORICAL if c in df.columns]
