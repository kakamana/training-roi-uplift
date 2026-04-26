# Training ROI Predictor (Causal Uplift)

> **Causal-uplift modelling for L&D — which employees benefit most from which training intervention.** An X-learner with two `GradientBoostingRegressor` base learners + logistic propensity, returning a per-employee CATE estimate plus a 90% bootstrap confidence interval.

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688) ![Next.js](https://img.shields.io/badge/Next.js-14-black) ![License](https://img.shields.io/badge/license-MIT-green)

## Why this project
- L&D ROI conversations usually compare *average* post-training performance to pre-training performance. That ignores **selection bias** (who got chosen for training) and **heterogeneity** (different employees benefit differently).
- This project estimates the **Conditional Average Treatment Effect (CATE)** per employee using an **X-learner**, exposes it via FastAPI, and ships a Next.js explorer with a population CATE histogram + a single-employee CATE explainer.

## Table of contents
- [Business Requirements](./docs/01_business_requirements.md)
- [Feasibility Study](./docs/02_feasibility_study.md)
- [Methodology — X-learner CATE + identifiability](./docs/03_methodology.md)
- [Evaluation Plan](./docs/04_evaluation.md)
- [Data card](./data/data_card.md) - [Data sources](./data/data_sources.md)
- [Notebooks](./notebooks/) - [Source](./src/training_roi/) - [API](./api/main.py) - [UI](./ui/app/page.tsx)

## Headline results (target)

| Metric | T-learner | **X-learner** | Target |
|---|---|---|---|
| Qini AUC (uplift) | 0.18 | **0.27** | > 0.22 |
| MAE on synthetic ground-truth CATE | 0.31 | **0.18** | < 0.20 |
| Top-30% policy uplift gain | 1.4× | **2.1×** | > 1.8× |
| Calibration (CATE bin slope) | 0.7 | **0.95** | 0.9 - 1.1 |

## Quickstart

```bash
pip install -e ".[dev]"
python -m training_roi.data         # generate synthetic employee + training panel
python -m training_roi.models       # fit X-learner, save artefacts
uvicorn api.main:app --reload
cd ui && npm install && npm run dev
```

## Stack
Python - pandas - scikit-learn (GradientBoosting, Logistic) - FastAPI - Next.js - Tailwind

## Author
Asad - MADS @ University of Michigan - Dubai HR
