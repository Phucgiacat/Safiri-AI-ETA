"""
features.py — Domain-Driven Feature Engineering for Shipment ETA Prediction

This module constructs features that reflect real-world logistics dynamics.
Every feature is motivated by supply chain domain knowledge, not arbitrary
statistical transformations.

Key Design Principles:
1. Features should capture DELAY PROPAGATION between stages
2. Features should reflect HOW logistics professionals think about delays
3. Features should be interpretable to operations teams

Domain Context:
In global freight forwarding, delays are rarely isolated events. A late
vessel departure often causes missed berth windows at the destination port,
which cascades into longer customs queues, ultimately delaying final
delivery far beyond the initial disruption. This "bullwhip effect" of
delays is what we aim to capture.
"""

import pandas as pd
import numpy as np

from src.config import (
    PEAK_SEASON_MONTHS,
    SEA_ROUTE_FACTOR,
    AIR_ROUTE_FACTOR,
    DELAY_COLS,
)
from src.utils import get_logger

logger = get_logger("features")


# ======================================================================
# Stage-Level Delay Features
# ======================================================================
def add_cumulative_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative delay at each stage of the shipment journey.

    In logistics, cumulative delay is a critical operational metric because
    it determines whether downstream schedules (e.g., warehouse receiving
    windows, trucking appointments) are still achievable.

    If cumulative delay exceeds the buffer time at any stage, the entire
    downstream plan must be re-optimized — a costly exercise.
    """
    df = df.copy()

    # Cumulative delay up to each stage
    df["cumulative_delay_dep"] = df["delay_departure_days"]
    df["cumulative_delay_port"] = (
        df["delay_departure_days"] + df["delay_port_days"]
    )
    df["cumulative_delay_customs"] = (
        df["delay_departure_days"]
        + df["delay_port_days"]
        + df["delay_customs_days"]
    )

    return df


def add_delay_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delay as a fraction of planned leg duration.

    A 1-day delay on a 14-day trans-Pacific voyage is minor (~7%).
    A 1-day delay on a 1-day air freight route is catastrophic (100%).

    These ratios normalize delays by transit time, making them comparable
    across different trade lanes — essential for a model that must
    generalize across routes.
    """
    df = df.copy()

    # Avoid division by zero for very short legs
    eps = 0.01

    if "planned_leg1_days" in df.columns:
        df["delay_ratio_leg1"] = (
            df["delay_departure_days"]
            / (df["planned_leg1_days"] + eps)
        )
        df["delay_ratio_leg2"] = (
            df["delay_port_days"]
            / (df["planned_leg2_days"] + eps)
        )
        df["delay_ratio_leg3"] = (
            df["delay_customs_days"]
            / (df["planned_leg3_days"] + eps)
        )

    return df


def add_delay_propagation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capture the dynamics of delay propagation between consecutive stages.

    Key features:
    - delay_acceleration: Is the delay growing or shrinking stage-over-stage?
      Positive = delay is worsening (supply chain is deteriorating)
      Negative = delay is recovering (buffers are absorbing the shock)

    - propagation_ratio: How much of the previous stage's delay carried
      over to the next stage? Values > 1.0 indicate delay amplification;
      values < 1.0 indicate the supply chain has built-in resilience.

    These features directly model the cascading delay phenomenon that
    Safiri AI cares about most.
    """
    df = df.copy()

    # Delay acceleration (change in delay between consecutive stages)
    df["delay_accel_dep_to_port"] = (
        df["delay_port_days"] - df["delay_departure_days"]
    )
    df["delay_accel_port_to_customs"] = (
        df["delay_customs_days"] - df["delay_port_days"]
    )
    df["delay_accel_customs_to_final"] = (
        df["delay_final_days"] - df["delay_customs_days"]
    )

    # Propagation ratio (how much delay is amplified between stages)
    eps = 0.01
    df["propagation_dep_to_port"] = (
        df["delay_port_days"] / (df["delay_departure_days"] + eps)
    )
    df["propagation_port_to_customs"] = (
        df["delay_customs_days"] / (df["delay_port_days"] + eps)
    )

    # Overall delay amplification factor
    df["total_delay_amplification"] = (
        df["delay_final_days"] / (df["delay_departure_days"] + eps)
    )

    return df


# ======================================================================
# Route & Congestion Features
# ======================================================================
def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features based on trade lane characteristics.

    In maritime logistics:
    - Sea routes through congested chokepoints (Suez Canal, Strait of
      Malacca, Panama Canal) have higher delay variance.
    - Trans-Pacific routes (Asia → US West Coast) are the world's busiest
      and most delay-prone trade lanes.
    - Air freight is faster but has less schedule variability.

    The route_risk_score combines route type with congestion to create
    a composite risk indicator.
    """
    df = df.copy()

    # Binary indicator for sea vs. air transport
    df["is_sea_route"] = (df["route_type"] == "sea").astype(int)

    # Route factor: sea routes have higher sensitivity to disruptions
    df["route_factor"] = df["is_sea_route"].map(
        {1: SEA_ROUTE_FACTOR, 0: AIR_ROUTE_FACTOR}
    )

    # Route risk score: combines route type with congestion
    df["route_risk_score"] = df["congestion_index"] * df["route_factor"]

    # Congestion impact weighted by route sensitivity
    df["congestion_impact"] = (
        df["congestion_index"] * df["route_factor"] * 1.5
    )

    return df


def add_external_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model interactions between external disruption factors.

    In reality, congestion and weather often compound each other:
    - Port congestion worsens during typhoon season (vessels queue up
      waiting for weather windows)
    - Heavy rain slows container handling operations at ports
    - Multiple disruptions simultaneously cause non-linear delay escalation

    These interaction terms capture effects that simple additive models miss.
    """
    df = df.copy()

    # Congestion × Weather interaction (compounding disruptions)
    df["congestion_weather_interaction"] = (
        df["congestion_index"] * df["weather_index"]
    )

    # Disruption severity index (composite external risk)
    df["disruption_severity"] = (
        0.6 * df["congestion_index"]
        + 0.3 * df["weather_index"]
        + 0.1 * df["congestion_weather_interaction"]
    )

    # High-risk indicator: congestion > 0.5 AND weather > 0.3
    df["high_disruption_flag"] = (
        (df["congestion_index"] > 0.5) & (df["weather_index"] > 0.3)
    ).astype(int)

    return df


# ======================================================================
# Temporal Features
# ======================================================================
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal patterns from departure timestamps.

    Shipping volumes — and therefore delays — follow strong seasonal
    patterns in global trade:

    - Peak Season (Oct–Dec): Retailers stock up for Black Friday,
      Christmas, and year-end sales. Ports operate at near-capacity,
      and vessel space is at a premium.
    - Chinese New Year (Jan–Feb): Factories in China shut down for
      2–4 weeks, creating a pre-CNY shipping surge followed by a lull.
    - Monsoon Season (Jun–Sep): Affects South/Southeast Asian ports
      with weather delays and reduced vessel speeds.
    - Mid-year (Jul–Aug): Generally lower volumes, fewer delays.
    """
    df = df.copy()

    if "scheduled_departure" in df.columns:
        dep = pd.to_datetime(df["scheduled_departure"])

        df["departure_month"] = dep.dt.month
        df["departure_day_of_week"] = dep.dt.dayofweek  # 0=Mon, 6=Sun
        df["departure_week_of_year"] = dep.dt.isocalendar().week.astype(int)

        # Peak season indicator
        df["is_peak_season"] = dep.dt.month.isin(PEAK_SEASON_MONTHS).astype(int)

        # Weekend departure (often indicates scheduling issues)
        df["is_weekend_departure"] = (dep.dt.dayofweek >= 5).astype(int)

    return df


# ======================================================================
# Missing Data Features
# ======================================================================
def add_data_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from data quality signals.

    In logistics, missing data is itself informative:
    - Missing customs timestamps may indicate expedited clearance
      (pre-cleared cargo) or, conversely, data system failures.
    - The pattern of missing data often correlates with certain
      trade lanes or carriers that have weaker digital infrastructure.

    Encoding missingness as a feature allows the model to learn
    these patterns rather than losing information through imputation.
    """
    df = df.copy()

    # customs_data_missing is already created in preprocessing
    # Add additional quality indicators if needed
    if "customs_data_missing" not in df.columns:
        df["customs_data_missing"] = 0

    return df


# ======================================================================
# Feature Assembly
# ======================================================================
def get_feature_columns() -> list:
    """
    Return the list of feature column names used for modeling.

    These are organized by category for interpretability:
    1. Stage delays (upstream stages only — no data leakage)
    2. Cumulative delays
    3. Delay ratios & propagation
    4. Route characteristics
    5. External factors & interactions
    6. Temporal features
    7. Data quality signals
    """
    return [
        # --- Stage-level delays (known at prediction time) ---
        "delay_departure_days",
        "delay_port_days",
        "delay_customs_days",
        # --- Cumulative delays ---
        "cumulative_delay_dep",
        "cumulative_delay_port",
        "cumulative_delay_customs",
        # --- Delay ratios ---
        "delay_ratio_leg1",
        "delay_ratio_leg2",
        "delay_ratio_leg3",
        # --- Delay propagation ---
        "delay_accel_dep_to_port",
        "delay_accel_port_to_customs",
        "propagation_dep_to_port",
        "propagation_port_to_customs",
        # --- Route features ---
        "is_sea_route",
        "route_factor",
        "route_risk_score",
        "congestion_impact",
        "base_transit_days",
        # --- External factors ---
        "congestion_index",
        "weather_index",
        "congestion_weather_interaction",
        "disruption_severity",
        "high_disruption_flag",
        # --- Temporal ---
        "departure_month",
        "departure_day_of_week",
        "is_peak_season",
        "is_weekend_departure",
        # --- Planned durations ---
        "planned_leg1_days",
        "planned_leg2_days",
        "planned_leg3_days",
        "total_planned_days",
        # --- Data quality ---
        "customs_data_missing",
        # --- Categorical (encoded) ---
        "origin_encoded",
        "destination_encoded",
        "route_type_encoded",
    ]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline.

    This pipeline transforms raw preprocessed shipment data into
    a rich feature set that captures:
    - How delays cascade through the supply chain
    - Route-specific risk profiles
    - Seasonal patterns in global trade
    - The compounding effect of multiple disruptions
    """
    logger.info("=" * 60)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 60)

    df = add_cumulative_delay_features(df)
    logger.info("[OK] Cumulative delay features")

    df = add_delay_ratio_features(df)
    logger.info("[OK] Delay ratio features")

    df = add_delay_propagation_features(df)
    logger.info("[OK] Delay propagation features")

    df = add_route_features(df)
    logger.info("[OK] Route features")

    df = add_external_interaction_features(df)
    logger.info("[OK] External interaction features")

    df = add_temporal_features(df)
    logger.info("[OK] Temporal features")

    df = add_data_quality_features(df)
    logger.info("[OK] Data quality features")

    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]

    logger.info(f"Features built: {len(available)}/{len(feature_cols)}")
    if missing:
        logger.warning(f"Missing features: {missing}")

    logger.info("Feature engineering complete")
    return df


if __name__ == "__main__":
    from src.utils import load_raw_data
    from src.preprocessing import run_preprocessing

    df = load_raw_data()
    df = run_preprocessing(df)
    df = build_features(df)

    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]

    print(f"\nTotal features: {len(available)}")
    print(f"\nFeature columns:")
    for i, col in enumerate(available, 1):
        print(f"  {i:2d}. {col}")
    print(f"\nSample feature values:")
    print(df[available].head(3).T)
