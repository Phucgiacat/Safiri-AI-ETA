"""
regression.py — ETA Delay Regression Models

Predicts the continuous final delay (in days) for a shipment.

Model Strategy:
- LightGBM (primary): Fast, handles categoricals natively, excellent
  for tabular data with mixed feature types.
- XGBoost (secondary): Strong alternative, good for comparison.
- Ridge Regression (baseline): Linear baseline to demonstrate that
  non-linear models add value beyond simple linear relationships.

In logistics operations, the regression output directly answers:
"How many days late will this shipment arrive?"
This is actionable information for supply chain planners who need to
decide whether to trigger contingency plans (expedited shipping,
alternative routing, customer notifications).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

from src.config import (
    LGBM_REG_PARAMS,
    XGB_REG_PARAMS,
    CV_FOLDS,
    RANDOM_SEED,
)
from src.features import get_feature_columns
from src.utils import (
    get_logger,
    compute_regression_metrics,
    save_model,
    save_metrics_table,
    save_dataframe_table,
)

logger = get_logger("regression")


def prepare_regression_data(df: pd.DataFrame):
    """
    Prepare feature matrix X and target vector y for regression.

    Target: delay_final_days — the total delay at final delivery.

    Note: We use upstream stage delays as features, simulating a
    scenario where we know what happened at earlier stages and want
    to predict the final outcome. In practice, this represents a
    "mid-journey" ETA update — the most valuable prediction in
    logistics because it allows proactive intervention.
    """
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]

    X = df[available_cols].copy()
    y = df["delay_final_days"].copy()

    # Handle any remaining NaN values
    X = X.fillna(0)

    logger.info(f"Regression data: X={X.shape}, y={y.shape}")
    logger.info(f"Target stats: mean={y.mean():.3f}, std={y.std():.3f}")

    return X, y, available_cols


def train_lightgbm_regression(X, y):
    """Train LightGBM regression model with cross-validation."""
    logger.info("Training LightGBM Regressor...")

    model = lgb.LGBMRegressor(**LGBM_REG_PARAMS)

    # Cross-validated predictions for honest evaluation
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_pred_cv = cross_val_predict(model, X, y, cv=kf)

    # Train final model on all data
    model.fit(X, y)

    # Compute metrics on CV predictions
    metrics = compute_regression_metrics(y.values, y_pred_cv, prefix="LGBM")
    logger.info(f"LightGBM CV Results: {metrics}")

    return model, y_pred_cv, metrics


def train_xgboost_regression(X, y):
    """Train XGBoost regression model with cross-validation."""
    logger.info("Training XGBoost Regressor...")

    model = xgb.XGBRegressor(**XGB_REG_PARAMS)

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_pred_cv = cross_val_predict(model, X, y, cv=kf)

    model.fit(X, y)

    metrics = compute_regression_metrics(y.values, y_pred_cv, prefix="XGB")
    logger.info(f"XGBoost CV Results: {metrics}")

    return model, y_pred_cv, metrics


def train_ridge_baseline(X, y):
    """
    Train Ridge regression baseline.

    Purpose: Demonstrate that the relationship between stage delays
    and final delay is non-trivial. If a simple linear model performs
    comparably to gradient boosting, it suggests the relationship is
    mostly linear (which, given the synthetic data generation process,
    is partially true — but the interaction terms add non-linearity).
    """
    logger.info("Training Ridge Regression Baseline...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ])

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_pred_cv = cross_val_predict(model, X, y, cv=kf)

    model.fit(X, y)

    metrics = compute_regression_metrics(y.values, y_pred_cv, prefix="Ridge")
    logger.info(f"Ridge CV Results: {metrics}")

    return model, y_pred_cv, metrics


def run_regression_pipeline(df: pd.DataFrame) -> dict:
    """
    Execute the full regression training pipeline.

    Returns a dictionary with models, predictions, and metrics
    for all three model types.
    """
    logger.info("=" * 60)
    logger.info("Starting Regression Pipeline")
    logger.info("=" * 60)

    X, y, feature_cols = prepare_regression_data(df)

    # Train all models
    lgbm_model, lgbm_pred, lgbm_metrics = train_lightgbm_regression(X, y)
    xgb_model, xgb_pred, xgb_metrics = train_xgboost_regression(X, y)
    ridge_model, ridge_pred, ridge_metrics = train_ridge_baseline(X, y)

    # Save models
    save_model(lgbm_model, "lgbm_regression")
    save_model(xgb_model, "xgb_regression")
    save_model(ridge_model, "ridge_regression")

    # Combine metrics for comparison table
    all_metrics = {}
    all_metrics.update(lgbm_metrics)
    all_metrics.update(xgb_metrics)
    all_metrics.update(ridge_metrics)
    save_metrics_table(all_metrics, "regression_metrics.csv")

    # Create comparison DataFrame
    comparison = pd.DataFrame({
        "Model": ["LightGBM", "XGBoost", "Ridge (Baseline)"],
        "MAE (days)": [
            lgbm_metrics["LGBM_MAE"],
            xgb_metrics["XGB_MAE"],
            ridge_metrics["Ridge_MAE"],
        ],
        "RMSE (days)": [
            lgbm_metrics["LGBM_RMSE"],
            xgb_metrics["XGB_RMSE"],
            ridge_metrics["Ridge_RMSE"],
        ],
        "R²": [
            lgbm_metrics["LGBM_R2"],
            xgb_metrics["XGB_R2"],
            ridge_metrics["Ridge_R2"],
        ],
        "MAPE (%)": [
            lgbm_metrics["LGBM_MAPE_%"],
            xgb_metrics["XGB_MAPE_%"],
            ridge_metrics["Ridge_MAPE_%"],
        ],
    })
    save_dataframe_table(comparison, "regression_comparison.csv")
    logger.info(f"\nModel Comparison:\n{comparison.to_string(index=False)}")

    # Save predictions for analysis
    pred_df = pd.DataFrame({
        "actual": y,
        "lgbm_pred": lgbm_pred,
        "xgb_pred": xgb_pred,
        "ridge_pred": ridge_pred,
    })
    save_dataframe_table(pred_df, "regression_predictions.csv")

    # Determine best model
    maes = {
        "LightGBM": lgbm_metrics["LGBM_MAE"],
        "XGBoost": xgb_metrics["XGB_MAE"],
        "Ridge": ridge_metrics["Ridge_MAE"],
    }
    best_name = min(maes, key=maes.get)
    logger.info(f"\n★ Best model by MAE: {best_name} ({maes[best_name]:.4f} days)")

    results = {
        "models": {
            "lgbm": lgbm_model,
            "xgb": xgb_model,
            "ridge": ridge_model,
        },
        "predictions": {
            "lgbm": lgbm_pred,
            "xgb": xgb_pred,
            "ridge": ridge_pred,
        },
        "metrics": {
            "lgbm": lgbm_metrics,
            "xgb": xgb_metrics,
            "ridge": ridge_metrics,
        },
        "feature_cols": feature_cols,
        "X": X,
        "y": y,
        "best_model_name": best_name,
        "comparison": comparison,
    }

    logger.info("Regression pipeline complete")
    return results


if __name__ == "__main__":
    from src.utils import load_raw_data
    from src.preprocessing import run_preprocessing
    from src.features import build_features

    df = load_raw_data()
    df = run_preprocessing(df)
    df = build_features(df)
    results = run_regression_pipeline(df)

    print("\n" + "=" * 60)
    print("REGRESSION RESULTS")
    print("=" * 60)
    print(results["comparison"].to_string(index=False))
