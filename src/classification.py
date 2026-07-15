"""
classification.py — Delay Risk Classification Models

Predicts whether a shipment will experience a significant delay
(final delay > 1 day).

In logistics operations, this binary prediction serves a different
purpose than the regression model:

  Regression  → "How late will it be?"  (operational planning)
  Classification → "Will it be late?"   (risk alerting & triage)

Supply chain control towers use delay risk scores to prioritize
which shipments need immediate attention. A shipment with 90% delay
probability gets flagged for proactive intervention (rebooking,
customer notification, buffer stock activation).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from src.config import LGBM_CLF_PARAMS, CV_FOLDS, RANDOM_SEED
from src.features import get_feature_columns
from src.utils import (
    get_logger,
    compute_classification_metrics,
    save_model,
    save_metrics_table,
    save_dataframe_table,
)

logger = get_logger("classification")


def prepare_classification_data(df: pd.DataFrame):
    """
    Prepare feature matrix X and binary target y for classification.

    Target: is_delayed — 1 if delay_final_days > 1.0 day, else 0.

    The 1-day threshold reflects a common SLA boundary in freight:
    shipments within ±1 day of schedule are typically considered
    "on time" in the industry.
    """
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]

    X = df[available_cols].copy()
    y = df["is_delayed"].copy()

    X = X.fillna(0)

    n_delayed = y.sum()
    n_total = len(y)
    logger.info(f"Classification data: X={X.shape}, y={y.shape}")
    logger.info(
        f"Class distribution: delayed={n_delayed} ({n_delayed/n_total:.1%}), "
        f"on-time={n_total - n_delayed} ({(n_total - n_delayed)/n_total:.1%})"
    )

    return X, y, available_cols


def train_lightgbm_classifier(X, y):
    """Train LightGBM classifier with stratified cross-validation."""
    logger.info("Training LightGBM Classifier...")

    model = lgb.LGBMClassifier(**LGBM_CLF_PARAMS)

    # Stratified CV to preserve class distribution in each fold
    skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    y_pred_cv = cross_val_predict(model, X, y, cv=skf)
    y_prob_cv = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]

    model.fit(X, y)

    metrics = compute_classification_metrics(
        y.values, y_pred_cv, y_prob_cv, prefix="LGBM"
    )
    logger.info(f"LightGBM CV Results: {metrics}")

    return model, y_pred_cv, y_prob_cv, metrics


def train_logistic_baseline(X, y):
    """
    Train Logistic Regression baseline.

    Logistic Regression serves as an interpretable baseline:
    - Coefficients directly show feature importance direction
    - Provides calibrated probability estimates
    - Useful for comparison against more complex models
    """
    logger.info("Training Logistic Regression Baseline...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced",  # Handle class imbalance
        )),
    ])

    skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    y_pred_cv = cross_val_predict(model, X, y, cv=skf)
    y_prob_cv = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]

    model.fit(X, y)

    metrics = compute_classification_metrics(
        y.values, y_pred_cv, y_prob_cv, prefix="LR"
    )
    logger.info(f"Logistic Regression CV Results: {metrics}")

    return model, y_pred_cv, y_prob_cv, metrics


def run_classification_pipeline(df: pd.DataFrame) -> dict:
    """
    Execute the full classification training pipeline.

    Returns a dictionary with models, predictions, probabilities,
    and metrics for all model types.
    """
    logger.info("=" * 60)
    logger.info("Starting Classification Pipeline")
    logger.info("=" * 60)

    X, y, feature_cols = prepare_classification_data(df)

    # Train models
    lgbm_model, lgbm_pred, lgbm_prob, lgbm_metrics = train_lightgbm_classifier(X, y)
    lr_model, lr_pred, lr_prob, lr_metrics = train_logistic_baseline(X, y)

    # Save models
    save_model(lgbm_model, "lgbm_classification")
    save_model(lr_model, "logistic_classification")

    # Combine metrics
    all_metrics = {}
    all_metrics.update(lgbm_metrics)
    all_metrics.update(lr_metrics)
    save_metrics_table(all_metrics, "classification_metrics.csv")

    # Comparison table
    comparison = pd.DataFrame({
        "Model": ["LightGBM", "Logistic Regression (Baseline)"],
        "Accuracy": [lgbm_metrics["LGBM_Accuracy"], lr_metrics["LR_Accuracy"]],
        "Precision": [lgbm_metrics["LGBM_Precision"], lr_metrics["LR_Precision"]],
        "Recall": [lgbm_metrics["LGBM_Recall"], lr_metrics["LR_Recall"]],
        "F1 Score": [lgbm_metrics["LGBM_F1"], lr_metrics["LR_F1"]],
        "AUC-ROC": [lgbm_metrics["LGBM_AUC_ROC"], lr_metrics["LR_AUC_ROC"]],
    })
    save_dataframe_table(comparison, "classification_comparison.csv")
    logger.info(f"\nModel Comparison:\n{comparison.to_string(index=False)}")

    # Save predictions
    pred_df = pd.DataFrame({
        "actual": y,
        "lgbm_pred": lgbm_pred,
        "lgbm_prob": lgbm_prob,
        "lr_pred": lr_pred,
        "lr_prob": lr_prob,
    })
    save_dataframe_table(pred_df, "classification_predictions.csv")

    results = {
        "models": {"lgbm": lgbm_model, "lr": lr_model},
        "predictions": {"lgbm": lgbm_pred, "lr": lr_pred},
        "probabilities": {"lgbm": lgbm_prob, "lr": lr_prob},
        "metrics": {"lgbm": lgbm_metrics, "lr": lr_metrics},
        "feature_cols": feature_cols,
        "X": X,
        "y": y,
        "comparison": comparison,
    }

    logger.info("Classification pipeline complete")
    return results


if __name__ == "__main__":
    from src.utils import load_raw_data
    from src.preprocessing import run_preprocessing
    from src.features import build_features

    df = load_raw_data()
    df = run_preprocessing(df)
    df = build_features(df)
    results = run_classification_pipeline(df)

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(results["comparison"].to_string(index=False))
