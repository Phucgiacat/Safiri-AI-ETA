# Safiri AI — Shipment ETA Prediction & Delay Propagation System

> **AI Internship Take-Home Assignment**
> Predicting Shipment ETA and Understanding How Delays Cascade Through the Supply Chain

---

## Overview

In global logistics, delays are rarely isolated events. A late vessel departure in Shanghai can cause a missed berth window in Los Angeles, cascade into customs queue overflow, and ultimately delay final delivery by 2-3x the original disruption. This phenomenon — **delay propagation** — is one of the most challenging problems in supply chain management.

This project builds an AI system that:

1. **Predicts the final delivery delay** (regression) with high accuracy
2. **Classifies delay risk** (binary classification) for proactive alerting
3. **Explains predictions** using SHAP values — identifying which factors drive each delay
4. **Analyzes delay propagation** — quantifying how disruptions cascade stage-by-stage

## Project Architecture

```
Safiri-AI-ETA/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── report.pdf                    # Technical report (2-3 pages)
│
├── data/
│   ├── generate_shipment_data.py # Synthetic data generator
│   └── shipments.csv             # 250 shipment records
│
├── notebooks/
│   └── ETA_Prediction.py         # Main analysis notebook (Jupyter-compatible)
│
├── src/
│   ├── config.py                 # Centralized configuration & hyperparameters
│   ├── preprocessing.py          # Data cleaning & missing data imputation
│   ├── features.py               # Domain-driven feature engineering
│   ├── regression.py             # ETA delay regression models
│   ├── classification.py         # Delay risk classification models
│   ├── explain.py                # SHAP explainability & propagation analysis
│   ├── visualization.py          # Publication-quality plot generation
│   └── utils.py                  # Data loading, metrics, model persistence
│
├── outputs/
│   ├── models/                   # Serialized trained models (.joblib)
│   ├── figures/                  # EDA & model evaluation plots
│   ├── shap/                     # SHAP explanation plots
│   └── tables/                   # Metrics & analysis tables (.csv)
│
└── tests/
    └── test_features.py          # Unit tests for feature engineering
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Phucgiacat/Safiri-AI-ETA.git
cd Safiri-AI-ETA
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Set encoding for Windows
set PYTHONIOENCODING=utf-8

# Run the complete analysis as a Python script
python notebooks/ETA_Prediction.py

# Or open and run the Jupyter Notebook
# jupyter notebook notebooks/ETA_Prediction.ipynb
```

### Run Individual Components

```bash
# Preprocessing only
python -m src.preprocessing

# Feature engineering only
python -m src.features

# Train regression models
python -m src.regression

# Train classification models
python -m src.classification

# Generate SHAP explanations
python -m src.explain

# Generate EDA visualizations
python -m src.visualization
```

### Run Tests

```bash
python -m unittest tests.test_features -v
```

## Methodology

### Data

We use a **synthetic dataset of 250 shipment journeys**, each traversing 4 stages:

| Stage | Description | Real-World Analog |
|---|---|---|
| 1. Departure | Cargo leaves the origin warehouse | Vessel sailing from origin port |
| 2. Port Arrival | Arrival at transshipment hub | Vessel berthing at destination port |
| 3. Customs Clearance | Regulatory inspection & documentation | CBP/customs processing |
| 4. Final Delivery | Last-mile transport to consignee | Trucking to warehouse |

**Key data characteristics:**
- 10 trade lanes (8 sea, 2 air) spanning major global corridors
- Delays generated with **explicit cascading propagation**
- ~9% missing customs data (simulating real-world tracking gaps)
- External factors: port congestion index, weather severity

### Feature Engineering (35 Features)

Our features are organized into 7 domain-driven categories:

| Category | Count | Purpose |
|---|---|---|
| Stage Delays | 3 | Direct delay measurements at upstream stages |
| Cumulative Delays | 3 | Total accumulated delay through the pipeline |
| Delay Ratios & Propagation | 7 | Normalized delays + stage-to-stage dynamics |
| Route Characteristics | 5 | Trade lane risk profiles |
| External Factors | 5 | Congestion × weather interactions |
| Temporal Patterns | 4 | Seasonal surges (peak season, weekends) |
| Data Quality & Encoding | 8 | Missing data signals + categorical encoding |

### Models

**Regression** (predicting continuous delay in days):
- LightGBM — Primary model (gradient boosting)
- XGBoost — Secondary comparison
- Ridge Regression — Linear baseline

**Classification** (predicting delay risk > 1 day):
- LightGBM Classifier — Primary model
- Logistic Regression — Interpretable baseline

### Explainability

- **SHAP values** — Per-feature contribution to each prediction
- **Feature importance** — Global ranking of delay drivers
- **Individual explanations** — "Why is this specific shipment delayed?"
- **Delay propagation analysis** — Quantified stage-to-stage cascade ratios

## Key Results

### Regression Performance (5-Fold CV)

| Model | MAE (days) | RMSE (days) | R² |
|---|---|---|---|
| Ridge (Baseline) | 0.2658 | 0.3288 | 0.7483 |
| XGBoost | 0.2789 | 0.3521 | 0.7113 |
| LightGBM | 0.2916 | 0.3682 | 0.6843 |

### Classification Performance (5-Fold Stratified CV)

| Model | Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.812 | 0.8498 | 0.8675 |
| LightGBM Classifier | 0.788 | 0.8418 | 0.8613 |

### Key Insights

1. **Customs delay is the strongest predictor** of final delivery delay (r=0.87), confirming delay propagation
2. **Port congestion** is the #1 external risk factor, especially for sea routes
3. **Sea routes** are 1.5x more vulnerable to disruption amplification than air routes
4. **Linear baselines performed surprisingly well (Ridge won)**: With a small sample size (250 records), highly parameterized non-linear models (XGBoost/LightGBM) are prone to overfitting, allowing the Ridge regression baseline to capture the fundamentally linear delay generation process more robustly.
5. **Missing customs data** is itself informative — it signals operational visibility gaps

## Domain Context

This system is designed to reflect real-world logistics behavior:

- **Delay propagation** is modeled explicitly, not treated as noise
- **Features reflect how logistics professionals think** (port dwell time, congestion impact, peak season)
- **Explanations are actionable**: "Reroute through less congested port" rather than "feature_7 is important"
- **The model supports mid-journey ETA updates** — the most valuable prediction in supply chain operations

## Author

Built for Safiri AI — AI Internship Take-Home Assignment

## License

This project is submitted as part of a job application and is not licensed for redistribution.
