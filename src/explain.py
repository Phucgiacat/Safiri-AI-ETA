"""
explain.py — Model Explainability & Delay Propagation Analysis

Provides interpretable explanations for model predictions using SHAP
(SHapley Additive exPlanations) and domain-specific delay analysis.

Why Explainability Matters in Logistics:
- Operations teams need to UNDERSTAND why a delay is predicted,
  not just that it is predicted.
- Actionable insights: "Port congestion at Shanghai is the primary
  driver" → reroute through Ningbo.
- Trust: Supply chain managers won't act on black-box predictions.
- Regulatory: Some trade compliance requires documented reasoning.

This module generates:
1. Global feature importance (what drives delays overall?)
2. SHAP summary plots (feature impact distribution)
3. Per-shipment explanations (why is THIS shipment delayed?)
4. Delay propagation analysis (how do delays cascade?)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.config import FIGURES_DIR, SHAP_DIR, TABLES_DIR, FIGURE_DPI
from src.utils import get_logger, save_dataframe_table

logger = get_logger("explain")


def compute_shap_values(model, X, model_name: str = "model"):
    """
    Compute SHAP values for the given model and features.

    SHAP values decompose each prediction into per-feature contributions,
    enabling us to say: "This shipment's predicted delay of 2.3 days is
    driven by +0.8 days from port congestion, +0.5 days from customs
    delay, and -0.2 days from favorable weather."
    """
    logger.info(f"Computing SHAP values for {model_name}...")

    # Use TreeExplainer for tree-based models (fast & exact)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:
        # Fallback to KernelExplainer for non-tree models
        logger.info("Using KernelExplainer (slower, model-agnostic)")
        background = shap.sample(X, min(50, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)

    logger.info(f"SHAP values computed: shape={np.array(shap_values).shape}")
    return shap_values, explainer


def plot_feature_importance(model, feature_cols, model_name: str = "LightGBM"):
    """
    Plot tree-based feature importance.

    This provides a quick overview of which features the model relies on
    most, complementing the more detailed SHAP analysis.
    """
    logger.info(f"Plotting feature importance for {model_name}...")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.warning(f"{model_name} does not support feature_importances_")
        return None

    # Sort by importance
    indices = np.argsort(importances)[::-1]
    top_n = min(20, len(indices))
    top_indices = indices[:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))

    ax.barh(
        range(top_n),
        importances[top_indices][::-1],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(
        [feature_cols[i] for i in top_indices][::-1], fontsize=9
    )
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Features — {model_name} Regression",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    path = FIGURES_DIR / f"feature_importance_{model_name.lower()}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance plot saved → {path}")

    # Save as table too
    importance_df = pd.DataFrame({
        "Feature": [feature_cols[i] for i in indices],
        "Importance": importances[indices],
        "Rank": range(1, len(indices) + 1),
    })
    save_dataframe_table(importance_df, f"feature_importance_{model_name.lower()}.csv")

    return importance_df


def plot_shap_summary(shap_values, X, model_name: str = "LightGBM"):
    """
    Generate SHAP summary plot (beeswarm).

    This is the most information-dense explainability visualization:
    - Each dot = one shipment
    - X-axis = SHAP value (impact on prediction)
    - Color = feature value (red=high, blue=low)

    Allows us to see patterns like:
    "High congestion_index (red dots) pushes delay predictions UP (right)"
    """
    logger.info(f"Generating SHAP summary plot for {model_name}...")

    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X,
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title(
        f"SHAP Feature Impact — {model_name}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    path = SHAP_DIR / f"shap_summary_{model_name.lower()}.png"
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP summary plot saved → {path}")


def plot_shap_bar(shap_values, X, model_name: str = "LightGBM"):
    """
    Generate SHAP bar plot showing mean absolute SHAP values.

    This answers: "On average, which features have the largest impact
    on delay predictions across all shipments?"
    """
    logger.info(f"Generating SHAP bar plot for {model_name}...")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title(
        f"Mean |SHAP Value| — {model_name}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    path = SHAP_DIR / f"shap_bar_{model_name.lower()}.png"
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP bar plot saved → {path}")


def explain_single_shipment(
    shap_values, X, shipment_idx: int, model_name: str = "LightGBM"
):
    """
    Generate a detailed explanation for a single shipment.

    This is the most operationally useful output: for a specific
    shipment, it tells the logistics team exactly WHY the model
    predicts a delay and which factors are responsible.

    Example output:
    "Shipment #42 — Predicted delay: 2.3 days
     Top contributors:
       +0.82 days: delay_customs_days = 1.61  (customs bottleneck)
       +0.45 days: congestion_impact = 0.89   (port congestion)
       -0.18 days: is_peak_season = 0         (off-peak, favorable)"
    """
    sv = shap_values[shipment_idx]
    features = X.iloc[shipment_idx]

    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(sv))[::-1]

    explanation = {
        "Rank": [],
        "Feature": [],
        "Value": [],
        "SHAP_Impact": [],
        "Direction": [],
    }

    for rank, idx in enumerate(indices[:10], 1):
        feat_name = X.columns[idx]
        feat_val = features.iloc[idx]
        shap_val = sv[idx]

        explanation["Rank"].append(rank)
        explanation["Feature"].append(feat_name)
        explanation["Value"].append(round(feat_val, 4))
        explanation["SHAP_Impact"].append(round(shap_val, 4))
        explanation["Direction"].append("↑ Increases delay" if shap_val > 0 else "↓ Reduces delay")

    explanation_df = pd.DataFrame(explanation)
    return explanation_df


def plot_shap_waterfall(
    shap_values, X, explainer, shipment_idx: int, model_name: str = "LightGBM"
):
    """
    Generate a waterfall plot for a single shipment explanation.

    The waterfall plot shows how each feature pushes the prediction
    from the base value (average prediction) to the final prediction
    for this specific shipment.
    """
    logger.info(f"Generating waterfall plot for shipment #{shipment_idx}...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create SHAP Explanation object
    if hasattr(explainer, "expected_value"):
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
    else:
        base_value = 0

    explanation = shap.Explanation(
        values=shap_values[shipment_idx],
        base_values=base_value,
        data=X.iloc[shipment_idx].values,
        feature_names=X.columns.tolist(),
    )

    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.title(
        f"Shipment #{shipment_idx} — Delay Prediction Breakdown",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    path = SHAP_DIR / f"shap_waterfall_shipment_{shipment_idx}.png"
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Waterfall plot saved → {path}")


def analyze_delay_propagation(df: pd.DataFrame):
    """
    Analyze how delays propagate through the shipment pipeline.

    This is the core domain insight that Safiri AI values:
    understanding the MECHANICS of delay cascading, not just
    predicting the outcome.

    Computes:
    1. Stage-to-stage correlation matrix (how correlated are delays?)
    2. Average propagation ratios (how much does each stage amplify?)
    3. Delay recovery analysis (do some routes absorb delays better?)
    """
    logger.info("Analyzing delay propagation patterns...")

    delay_cols = [
        "delay_departure_days",
        "delay_port_days",
        "delay_customs_days",
        "delay_final_days",
    ]

    # 1. Stage-to-stage delay correlation
    corr = df[delay_cols].corr()
    save_dataframe_table(corr, "delay_correlation_matrix.csv")

    # 2. Propagation ratios by route type
    propagation_analysis = []

    for route_type in df["route_type"].unique():
        subset = df[df["route_type"] == route_type]
        eps = 0.01

        prop_dep_to_port = (
            subset["delay_port_days"] / (subset["delay_departure_days"] + eps)
        ).median()

        prop_port_to_customs = (
            subset["delay_customs_days"] / (subset["delay_port_days"] + eps)
        ).median()

        prop_customs_to_final = (
            subset["delay_final_days"] / (subset["delay_customs_days"] + eps)
        ).median()

        propagation_analysis.append({
            "Route Type": route_type,
            "Departure → Port (median ratio)": round(prop_dep_to_port, 3),
            "Port → Customs (median ratio)": round(prop_port_to_customs, 3),
            "Customs → Final (median ratio)": round(prop_customs_to_final, 3),
            "Avg Final Delay (days)": round(subset["delay_final_days"].mean(), 3),
            "Delay Recovery Rate (%)": round(
                (subset["delay_final_days"] < subset["cumulative_delay_customs"]).mean() * 100
                if "cumulative_delay_customs" in subset.columns else 0,
                1,
            ),
        })

    prop_df = pd.DataFrame(propagation_analysis)
    save_dataframe_table(prop_df, "delay_propagation_analysis.csv")
    logger.info(f"Delay propagation analysis:\n{prop_df.to_string(index=False)}")

    # 3. Route-level delay analysis
    route_analysis = (
        df.groupby(["origin", "destination"])
        .agg({
            "delay_final_days": ["mean", "std", "max"],
            "is_delayed": "mean",
            "congestion_index": "mean",
        })
        .round(3)
    )
    route_analysis.columns = [
        "Avg Delay", "Delay Std", "Max Delay",
        "Delay Rate", "Avg Congestion",
    ]
    route_analysis = route_analysis.sort_values("Avg Delay", ascending=False)
    save_dataframe_table(
        route_analysis.reset_index(), "route_delay_analysis.csv"
    )

    return {
        "correlation": corr,
        "propagation": prop_df,
        "route_analysis": route_analysis,
    }


def run_explanation_pipeline(
    reg_results: dict, clf_results: dict, df: pd.DataFrame
) -> dict:
    """
    Execute the full explainability pipeline.

    Generates all explanations, SHAP plots, and propagation analysis.
    """
    logger.info("=" * 60)
    logger.info("Starting Explainability Pipeline")
    logger.info("=" * 60)

    X = reg_results["X"]
    feature_cols = reg_results["feature_cols"]

    # --- Regression Explainability ---
    lgbm_reg = reg_results["models"]["lgbm"]

    # Feature importance
    importance_df = plot_feature_importance(lgbm_reg, feature_cols, "LightGBM_Reg")

    # SHAP values
    shap_values, explainer = compute_shap_values(lgbm_reg, X, "LightGBM_Reg")
    plot_shap_summary(shap_values, X, "LightGBM_Reg")
    plot_shap_bar(shap_values, X, "LightGBM_Reg")

    # Individual shipment explanations (pick interesting cases)
    # Find: most delayed, least delayed, and median shipment
    y = reg_results["y"]
    most_delayed_idx = y.idxmax()
    least_delayed_idx = y.idxmin()
    median_idx = (y - y.median()).abs().idxmin()

    sample_indices = [most_delayed_idx, median_idx, least_delayed_idx]
    sample_labels = ["Most Delayed", "Median", "Least Delayed"]

    all_explanations = []
    for idx, label in zip(sample_indices, sample_labels):
        explanation = explain_single_shipment(shap_values, X, idx, "LightGBM_Reg")
        explanation["Shipment"] = label
        explanation["Shipment_ID"] = df.iloc[idx]["shipment_id"] if "shipment_id" in df.columns else idx
        all_explanations.append(explanation)

        # Waterfall plot for each
        plot_shap_waterfall(shap_values, X, explainer, idx, "LightGBM_Reg")

    explanations_df = pd.concat(all_explanations, ignore_index=True)
    save_dataframe_table(explanations_df, "sample_explanations.csv")

    # --- Classification Explainability ---
    lgbm_clf = clf_results["models"]["lgbm"]
    X_clf = clf_results["X"]

    shap_values_clf, explainer_clf = compute_shap_values(
        lgbm_clf, X_clf, "LightGBM_Clf"
    )

    # Handle multi-output SHAP for binary classification
    if isinstance(shap_values_clf, list):
        shap_values_clf_plot = shap_values_clf[1]  # Class 1 (delayed)
    else:
        shap_values_clf_plot = shap_values_clf

    plot_shap_summary(shap_values_clf_plot, X_clf, "LightGBM_Clf")
    plot_shap_bar(shap_values_clf_plot, X_clf, "LightGBM_Clf")

    # Feature importance for classifier
    plot_feature_importance(lgbm_clf, feature_cols, "LightGBM_Clf")

    # --- Delay Propagation Analysis ---
    propagation = analyze_delay_propagation(df)

    results = {
        "shap_values_reg": shap_values,
        "shap_values_clf": shap_values_clf_plot,
        "explainer_reg": explainer,
        "importance": importance_df,
        "explanations": explanations_df,
        "propagation": propagation,
    }

    logger.info("Explainability pipeline complete")
    return results


if __name__ == "__main__":
    from src.utils import load_raw_data
    from src.preprocessing import run_preprocessing
    from src.features import build_features
    from src.regression import run_regression_pipeline
    from src.classification import run_classification_pipeline

    df = load_raw_data()
    df = run_preprocessing(df)
    df = build_features(df)

    reg_results = run_regression_pipeline(df)
    clf_results = run_classification_pipeline(df)
    explain_results = run_explanation_pipeline(reg_results, clf_results, df)

    print("\n[OK] All explanations generated successfully")
