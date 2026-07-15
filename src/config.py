"""
config.py — Centralized Configuration for Safiri AI ETA Prediction System

All project-wide constants, paths, and hyperparameters are defined here
to ensure reproducibility and easy experimentation.
"""

from pathlib import Path

# ======================================================================
# Project Paths
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
SHAP_DIR = OUTPUT_DIR / "shap"
TABLES_DIR = OUTPUT_DIR / "tables"

# Ensure output directories exist
for d in [MODELS_DIR, FIGURES_DIR, SHAP_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ======================================================================
# Data Files
# ======================================================================
RAW_DATA_PATH = DATA_DIR / "shipments.csv"

# ======================================================================
# Random Seed (reproducibility)
# ======================================================================
RANDOM_SEED = 42

# ======================================================================
# Shipment Pipeline Configuration
# ======================================================================
# The 4 stages of a shipment journey, in chronological order.
# This mirrors real-world ocean/air freight workflows:
#   1. Origin Departure  — cargo leaves the shipper's warehouse
#   2. Port Arrival       — vessel/aircraft reaches the transshipment hub
#   3. Customs Clearance  — regulatory inspection & documentation
#   4. Final Delivery     — last-mile transport to consignee
STAGES = ["departure", "port", "customs", "final"]

# Scheduled & actual timestamp columns for each stage
SCHEDULED_COLS = [
    "scheduled_departure",
    "scheduled_port_arrival",
    "scheduled_customs_clearance",
    "scheduled_final_delivery",
]

ACTUAL_COLS = [
    "actual_departure",
    "actual_port_arrival",
    "actual_customs_clearance",
    "actual_final_delivery",
]

DELAY_COLS = [
    "delay_departure_days",
    "delay_port_days",
    "delay_customs_days",
    "delay_final_days",
]

# ======================================================================
# Feature Engineering
# ======================================================================
# External factors that affect shipment performance
EXTERNAL_FEATURES = ["congestion_index", "weather_index"]

# Categorical columns requiring encoding
CATEGORICAL_COLS = ["origin", "destination", "route_type"]

# Sea routes are inherently more susceptible to delays due to:
# - Port congestion (berth availability, crane scheduling)
# - Weather disruptions (typhoons, monsoons, rough seas)
# - Canal bottlenecks (Suez, Panama)
SEA_ROUTE_FACTOR = 1.5
AIR_ROUTE_FACTOR = 0.5

# Delay threshold for binary classification (in days)
# A shipment is considered "significantly delayed" if final delay > 1 day.
# In practice, this threshold varies by service level agreement (SLA).
DELAY_THRESHOLD = 1.0

# Peak shipping seasons (month numbers)
# Q4 (Oct-Dec): Pre-holiday inventory build-up
# Jan-Feb: Chinese New Year disruption
PEAK_SEASON_MONTHS = [1, 2, 10, 11, 12]

# ======================================================================
# Model Hyperparameters
# ======================================================================
# Cross-validation
CV_FOLDS = 5

# LightGBM — Regression
LGBM_REG_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# XGBoost — Regression
XGB_REG_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}

# LightGBM — Classification
LGBM_CLF_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# ======================================================================
# Visualization
# ======================================================================
FIGURE_DPI = 150
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
COLOR_PALETTE = "viridis"

# ======================================================================
# Known Trade Lanes (base transit days for reference)
# ======================================================================
TRADE_LANES = {
    ("Shanghai", "Los Angeles"): {"route_type": "sea", "base_days": 14},
    ("Shanghai", "Rotterdam"): {"route_type": "sea", "base_days": 30},
    ("Singapore", "Dubai"): {"route_type": "sea", "base_days": 10},
    ("Ho Chi Minh City", "Tokyo"): {"route_type": "sea", "base_days": 6},
    ("Hamburg", "New York"): {"route_type": "sea", "base_days": 9},
    ("Hong Kong", "Sydney"): {"route_type": "sea", "base_days": 8},
    ("Mumbai", "London"): {"route_type": "air", "base_days": 1},
    ("Bangkok", "Frankfurt"): {"route_type": "air", "base_days": 1},
    ("Ho Chi Minh City", "Los Angeles"): {"route_type": "sea", "base_days": 18},
    ("Busan", "Long Beach"): {"route_type": "sea", "base_days": 12},
}
