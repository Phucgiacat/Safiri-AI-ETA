"""
utils.py — Utility Functions for Safiri AI ETA Prediction System

Provides reusable helpers for data loading, model persistence,
metric computation, and logging throughout the pipeline.
"""

import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config import RAW_DATA_PATH, MODELS_DIR, TABLES_DIR, RANDOM_SEED


# ======================================================================
# Logging
# ======================================================================
def get_logger(name: str) -> logging.Logger:
    """Create a standardized logger for each module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ======================================================================
# Data Loading
# ======================================================================
def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw shipment dataset.

    In a production logistics system, data would be streamed from
    TMS (Transportation Management System) or ERP integrations.
    Here we load from the synthetic CSV.
    """
    logger = get_logger("utils.load_raw_data")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} shipment records from {path.name}")
    return df


# ======================================================================
# Model Persistence
# ======================================================================
def save_model(model, name: str) -> Path:
    """Save a trained model to the outputs/models/ directory."""
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    logger = get_logger("utils.save_model")
    logger.info(f"Model saved → {path}")
    return path


def load_model(name: str):
    """Load a trained model from the outputs/models/ directory."""
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


# ======================================================================
# Regression Metrics
# ======================================================================
def compute_regression_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """
    Compute standard regression metrics for ETA prediction.

    In logistics, MAE is often the most business-relevant metric because
    it directly translates to "average error in days" — a quantity that
    supply chain planners can act on.

    Returns dict with MAE, RMSE, R², and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error) — common in logistics KPIs
    # Guard against division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}MAE": round(mae, 4),
        f"{p}RMSE": round(rmse, 4),
        f"{p}R2": round(r2, 4),
        f"{p}MAPE_%": round(mape, 2),
    }


# ======================================================================
# Classification Metrics
# ======================================================================
def compute_classification_metrics(
    y_true, y_pred, y_prob=None, prefix: str = ""
) -> dict:
    """
    Compute classification metrics for delay risk prediction.

    In logistics operations:
    - Precision matters when triggering costly interventions (e.g., expediting)
    - Recall matters when missing a delay has high downstream impact
    - F1 balances both, suitable for general delay alerting
    """
    p = f"{prefix}_" if prefix else ""
    metrics = {
        f"{p}Accuracy": round(accuracy_score(y_true, y_pred), 4),
        f"{p}Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        f"{p}Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        f"{p}F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics[f"{p}AUC_ROC"] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


# ======================================================================
# Table Export
# ======================================================================
def save_metrics_table(metrics: dict, filename: str) -> Path:
    """Save metrics dictionary as a formatted CSV table."""
    df = pd.DataFrame([metrics])
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    logger = get_logger("utils.save_metrics_table")
    logger.info(f"Metrics table saved → {path}")
    return path


def save_dataframe_table(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame as CSV to the tables output directory."""
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    return path


# ======================================================================
# Reproducibility
# ======================================================================
def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
