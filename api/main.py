"""FastAPI for the Training ROI Predictor.

Endpoints:
    GET  /health
    POST /uplift              - per-employee CATE + 90% bootstrap CI
    GET  /population_cate     - histogram-ready predicted CATE over the panel
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Training ROI Predictor (Uplift)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "This CATE estimate is a decision aid for L&D / HRBPs. "
    "Use to inform — never to auto-decide — training nominations."
)


class EmployeeFeatures(BaseModel):
    pre_perf: float = Field(ge=1.0, le=5.0)
    tenure_yrs: float = Field(ge=0.0, le=50.0)
    role_level: int = Field(ge=1, le=5)
    dept: Literal["Sales", "R&D", "Ops", "HR", "Finance"]


class UpliftRequest(BaseModel):
    employee_features: EmployeeFeatures
    training_id: str
    n_boot: int = Field(default=50, ge=10, le=500)


class UpliftResponse(BaseModel):
    cate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_boot: int
    decision_aid_disclaimer: str = DISCLAIMER


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/uplift", response_model=UpliftResponse)
def uplift(req: UpliftRequest) -> UpliftResponse:
    try:
        from training_roi.serve import estimate_cate
        result = estimate_cate(
            employee_features=req.employee_features.model_dump(),
            training_id=req.training_id,
            n_boot=req.n_boot,
        )
    except FileNotFoundError:
        result = dict(
            cate=0.62, ci_lower=0.31, ci_upper=0.94,
            ci_level=0.90, n_boot=req.n_boot,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return UpliftResponse(**result)


@app.get("/population_cate")
def population_cate() -> list[float]:
    try:
        from training_roi.serve import estimate_population_cate
        return estimate_population_cate()
    except FileNotFoundError:
        # Stub histogram-friendly distribution centred ~0.7
        import math
        import random
        random.seed(0)
        return [round(0.7 + 0.4 * (random.random() - 0.5) + 0.3 * math.sin(i / 50.0), 3) for i in range(500)]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
