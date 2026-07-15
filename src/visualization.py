"""
visualization.py — Comprehensive Visualization Pipeline

Generates publication-quality plots for EDA, model evaluation,
and delay propagation analysis.

All visualizations use a consistent professional style designed
for both technical reports and stakeholder presentations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.config import (
    FIGURES_DIR,
    FIGURE_DPI,
    DELAY_COLS,
)
from src.utils import get_logger

logger = get_logger("visualization")

# ======================================================================
# Global Plot Style
# ======================================================================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#cccccc",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})

COLORS = {
    "primary": "#2563eb",
    "secondary": "#7c3aed",
    "success": "#059669",
    "warning": "#d97706",
    "danger": "#dc2626",
    "info": "#0891b2",
    "sea": "#2563eb",
    "air": "#d97706",
}

STAGE_COLORS = ["#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444"]
STAGE_NAMES = ["Departure", "Port Arrival", "Customs", "Final Delivery"]


# ======================================================================
# EDA Visualizations
# ======================================================================
def plot_delay_distributions(df: pd.DataFrame):
    """
    Plot the distribution of delays at each shipment stage.

    This visualization reveals:
    - Departure delays follow an exponential distribution (random events)
    - Port & customs delays are right-skewed (congestion effects)
    - Final delays show the combined cascading effect
    """
    logger.info("Plotting delay distributions...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Delay Distribution Across Shipment Stages",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    delay_cols = [
        "delay_departure_days",
        "delay_port_days",
        "delay_customs_days",
        "delay_final_days",
    ]

    for ax, col, color, name in zip(
        axes.flat, delay_cols, STAGE_COLORS, STAGE_NAMES
    ):
        data = df[col].dropna()
        ax.hist(
            data, bins=25, color=color, alpha=0.7,
            edgecolor="white", linewidth=0.5,
        )
        ax.axvline(
            data.mean(), color="red", linestyle="--",
            linewidth=1.5, label=f"Mean: {data.mean():.2f}d",
        )
        ax.axvline(
            data.median(), color="black", linestyle=":",
            linewidth=1.5, label=f"Median: {data.median():.2f}d",
        )
        ax.set_title(f"Stage: {name}", fontsize=12)
        ax.set_xlabel("Delay (days)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "delay_distributions.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Plot correlation heatmap between delay columns and external factors.

    Key insight: The strong correlation between consecutive stage delays
    demonstrates the delay propagation effect — the core phenomenon
    this project aims to model.
    """
    logger.info("Plotting correlation heatmap...")

    cols = [
        "delay_departure_days", "delay_port_days",
        "delay_customs_days", "delay_final_days",
        "congestion_index", "weather_index",
    ]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
    )
    ax.set_title(
        "Delay Correlation Matrix — Evidence of Cascading Delays",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    path = FIGURES_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_delay_by_route(df: pd.DataFrame):
    """
    Compare delay patterns across different trade lanes.

    This reveals which shipping corridors are most delay-prone,
    helping logistics teams optimize route selection.
    """
    logger.info("Plotting delay by route...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # By route type (sea vs air)
    ax1 = axes[0]
    route_data = df.groupby("route_type")["delay_final_days"].agg(
        ["mean", "std", "median"]
    ).reset_index()

    bars = ax1.bar(
        route_data["route_type"],
        route_data["mean"],
        yerr=route_data["std"],
        capsize=5,
        color=[COLORS["sea"], COLORS["air"]],
        alpha=0.8,
        edgecolor="white",
        linewidth=1.5,
    )
    ax1.set_title("Average Final Delay by Transport Mode", fontsize=12)
    ax1.set_ylabel("Final Delay (days)")
    ax1.set_xlabel("Transport Mode")

    for bar, val in zip(bars, route_data["mean"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}d",
            ha="center", va="bottom", fontweight="bold",
        )

    # By trade lane
    ax2 = axes[1]
    lane_data = (
        df.groupby(["origin", "destination"])["delay_final_days"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    lane_data["lane"] = lane_data["origin"] + " → " + lane_data["destination"]

    colors = plt.cm.RdYlGn_r(
        np.linspace(0.2, 0.8, len(lane_data))
    )
    ax2.barh(
        lane_data["lane"],
        lane_data["delay_final_days"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.set_title("Average Final Delay by Trade Lane", fontsize=12)
    ax2.set_xlabel("Final Delay (days)")

    for i, val in enumerate(lane_data["delay_final_days"]):
        ax2.text(val + 0.02, i, f"{val:.2f}d", va="center", fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "delay_by_route.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_congestion_vs_delay(df: pd.DataFrame):
    """
    Scatter plot showing the relationship between port congestion
    and final delivery delay, colored by route type.

    This directly demonstrates a key logistics insight: port congestion
    is a leading indicator of downstream delays, especially for sea routes.
    """
    logger.info("Plotting congestion vs delay...")

    fig, ax = plt.subplots(figsize=(10, 7))

    for rtype, color in [("sea", COLORS["sea"]), ("air", COLORS["air"])]:
        subset = df[df["route_type"] == rtype]
        ax.scatter(
            subset["congestion_index"],
            subset["delay_final_days"],
            c=color,
            alpha=0.6,
            s=50,
            label=f"{rtype.upper()} ({len(subset)} shipments)",
            edgecolors="white",
            linewidth=0.5,
        )

    # Trend line
    z = np.polyfit(df["congestion_index"], df["delay_final_days"], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, df["congestion_index"].max(), 100)
    ax.plot(x_trend, p(x_trend), "--", color="red", linewidth=2, alpha=0.7,
            label=f"Trend (slope={z[0]:.2f})")

    ax.set_xlabel("Port Congestion Index", fontsize=12)
    ax.set_ylabel("Final Delivery Delay (days)", fontsize=12)
    ax.set_title(
        "Port Congestion Impact on Final Delay",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "congestion_vs_delay.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_delay_cascade(df: pd.DataFrame):
    """
    Visualize how delays accumulate across stages (cascade plot).

    This is the signature visualization of the project: it shows
    the "delay snowball effect" where small initial delays compound
    into significant final delays.
    """
    logger.info("Plotting delay cascade...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Sample shipments for clarity
    n_samples = min(30, len(df))
    sample = df.nlargest(n_samples, "delay_final_days")

    stages = ["Departure", "Port Arrival", "Customs", "Final Delivery"]
    delay_cols = [
        "delay_departure_days", "delay_port_days",
        "delay_customs_days", "delay_final_days",
    ]

    for _, row in sample.iterrows():
        delays = [row[c] if pd.notna(row[c]) else 0 for c in delay_cols]
        alpha = 0.4 + 0.6 * (row["delay_final_days"] / sample["delay_final_days"].max())
        ax.plot(stages, delays, "-o", alpha=min(alpha, 1.0),
                linewidth=1.5, markersize=5)

    # Mean trajectory
    mean_delays = [df[c].mean() for c in delay_cols]
    ax.plot(
        stages, mean_delays, "-s",
        color="red", linewidth=3, markersize=10,
        label=f"Mean delay trajectory", zorder=10,
    )

    ax.set_ylabel("Delay (days)", fontsize=12)
    ax.set_xlabel("Shipment Stage", fontsize=12)
    ax.set_title(
        "Delay Cascade Across Shipment Stages\n"
        "(Each line = one shipment, Red = average)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = FIGURES_DIR / "delay_cascade.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ======================================================================
# Model Evaluation Visualizations
# ======================================================================
def plot_actual_vs_predicted(y_true, y_pred, model_name: str = "LightGBM"):
    """
    Scatter plot of actual vs. predicted delay values.

    Points on the diagonal line indicate perfect predictions.
    The spread around the diagonal shows prediction uncertainty.
    """
    logger.info(f"Plotting actual vs predicted for {model_name}...")

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(
        y_true, y_pred,
        alpha=0.5, s=40,
        c=COLORS["primary"],
        edgecolors="white",
        linewidth=0.5,
    )

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        "--", color="red", linewidth=2,
        label="Perfect Prediction",
    )

    # Error bands (±0.5 days)
    ax.fill_between(
        [min_val, max_val],
        [min_val - 0.5, max_val - 0.5],
        [min_val + 0.5, max_val + 0.5],
        alpha=0.1, color="green",
        label="±0.5 day tolerance",
    )

    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    ax.set_xlabel("Actual Delay (days)", fontsize=12)
    ax.set_ylabel("Predicted Delay (days)", fontsize=12)
    ax.set_title(
        f"Actual vs. Predicted Delay — {model_name}\n"
        f"MAE={mae:.3f} days  |  R²={r2:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = FIGURES_DIR / f"actual_vs_predicted_{model_name.lower()}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_residuals(y_true, y_pred, model_name: str = "LightGBM"):
    """
    Residual plot to diagnose prediction errors.

    Random scatter around zero = good model.
    Patterns = systematic bias (e.g., underpredicting high delays).
    """
    logger.info(f"Plotting residuals for {model_name}...")

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Residual scatter
    ax1 = axes[0]
    ax1.scatter(
        y_pred, residuals,
        alpha=0.5, s=40,
        c=COLORS["secondary"],
        edgecolors="white",
        linewidth=0.5,
    )
    ax1.axhline(y=0, color="red", linewidth=2, linestyle="--")
    ax1.set_xlabel("Predicted Delay (days)")
    ax1.set_ylabel("Residual (days)")
    ax1.set_title(f"Residual Plot — {model_name}")

    # Residual distribution
    ax2 = axes[1]
    ax2.hist(
        residuals, bins=25, color=COLORS["secondary"],
        alpha=0.7, edgecolor="white",
    )
    ax2.axvline(
        x=0, color="red", linewidth=2, linestyle="--",
    )
    ax2.axvline(
        x=residuals.mean(), color="black", linewidth=1.5,
        linestyle=":", label=f"Mean: {residuals.mean():.3f}d",
    )
    ax2.set_xlabel("Residual (days)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Residual Distribution — {model_name}")
    ax2.legend()

    plt.tight_layout()
    path = FIGURES_DIR / f"residuals_{model_name.lower()}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str = "LightGBM"):
    """
    Plot confusion matrix for classification results.

    In logistics context:
    - True Positive: Correctly predicted delay → proactive action taken
    - False Positive: False alarm → unnecessary cost
    - False Negative: Missed delay → customer dissatisfaction
    - True Negative: Correctly predicted on-time → no action needed
    """
    logger.info(f"Plotting confusion matrix for {model_name}...")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["On-Time", "Delayed"],
        yticklabels=["On-Time", "Delayed"],
        ax=ax,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {model_name}\n"
        f"(TP: Correct delay alert  |  FN: Missed delay)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    path = FIGURES_DIR / f"confusion_matrix_{model_name.lower()}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_roc_curve(y_true, y_prob, model_name: str = "LightGBM"):
    """
    Plot ROC curve for classification model.

    AUC-ROC measures the model's ability to distinguish between
    delayed and on-time shipments across all probability thresholds.
    """
    logger.info(f"Plotting ROC curve for {model_name}...")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(
        fpr, tpr,
        color=COLORS["primary"],
        linewidth=2.5,
        label=f"{model_name} (AUC = {roc_auc:.3f})",
    )
    ax.plot(
        [0, 1], [0, 1],
        "--", color="gray", linewidth=1.5,
        label="Random Classifier",
    )
    ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS["primary"])

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC Curve — {model_name} Delay Classifier",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")

    plt.tight_layout()
    path = FIGURES_DIR / f"roc_curve_{model_name.lower()}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ======================================================================
# Master Visualization Runner
# ======================================================================
def run_eda_visualizations(df: pd.DataFrame):
    """Generate all EDA visualizations."""
    logger.info("=" * 60)
    logger.info("Generating EDA Visualizations")
    logger.info("=" * 60)

    plot_delay_distributions(df)
    plot_correlation_heatmap(df)
    plot_delay_by_route(df)
    plot_congestion_vs_delay(df)
    plot_delay_cascade(df)

    logger.info("All EDA visualizations complete")


def run_model_visualizations(reg_results: dict, clf_results: dict):
    """Generate all model evaluation visualizations."""
    logger.info("=" * 60)
    logger.info("Generating Model Evaluation Visualizations")
    logger.info("=" * 60)

    y_reg = reg_results["y"]

    # Regression plots for each model
    for name, key in [("LightGBM", "lgbm"), ("XGBoost", "xgb"), ("Ridge", "ridge")]:
        pred = reg_results["predictions"][key]
        plot_actual_vs_predicted(y_reg, pred, name)
        plot_residuals(y_reg, pred, name)

    # Classification plots
    y_clf = clf_results["y"]
    for name, key in [("LightGBM_Clf", "lgbm"), ("LogisticReg", "lr")]:
        pred = clf_results["predictions"][key]
        prob = clf_results["probabilities"][key]
        plot_confusion_matrix(y_clf, pred, name)
        plot_roc_curve(y_clf, prob, name)

    logger.info("All model evaluation visualizations complete")


if __name__ == "__main__":
    from src.utils import load_raw_data
    from src.preprocessing import run_preprocessing
    from src.features import build_features

    df = load_raw_data()
    df = run_preprocessing(df)
    df = build_features(df)
    run_eda_visualizations(df)

    print("\n[OK] All EDA visualizations generated")
