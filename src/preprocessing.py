"""
preprocessing.py — Data Preprocessing Pipeline

Handles data cleaning, type conversion, and missing data imputation
specific to shipment/logistics data.

In real-world logistics systems, data quality is a major challenge:
- Tracking events may be reported out of order
- Timestamps can be missing when IoT devices lose connectivity
- Customs data is frequently incomplete due to regulatory delays
- Different carriers use different time zone conventions

This module addresses these issues systematically.
"""

import pandas as pd
import numpy as np

from src.config import (
    SCHEDULED_COLS,
    ACTUAL_COLS,
    DELAY_COLS,
    CATEGORICAL_COLS,
    TRADE_LANES,
)
from src.utils import get_logger

logger = get_logger("preprocessing")


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all timestamp columns from strings to datetime objects.

    In production logistics systems, timestamps come from various sources
    (vessel AIS signals, port terminal operating systems, customs EDI)
    and may use different formats. Here we standardize them.
    """
    df = df.copy()
    timestamp_cols = SCHEDULED_COLS + ACTUAL_COLS

    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            n_null = df[col].isna().sum()
            if n_null > 0:
                logger.warning(f"{col}: {n_null} values could not be parsed")

    return df


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the shipment dataset.

    Missing customs clearance data is common in real logistics:
    - Some countries have slow digital customs reporting
    - Small shipments may bypass formal customs procedures
    - Data integration failures between customs and TMS systems

    Strategy:
    - For delay_customs_days: impute with the median delay for the same
      route corridor, preserving route-specific patterns.
    - For actual_customs_clearance timestamp: reconstruct from
      scheduled + imputed delay.
    - Flag all imputed records with a 'customs_data_missing' indicator.
    """
    df = df.copy()

    # Flag missing customs data BEFORE imputation
    df["customs_data_missing"] = df["delay_customs_days"].isna().astype(int)
    n_missing = df["customs_data_missing"].sum()
    logger.info(
        f"Customs data missing: {n_missing}/{len(df)} "
        f"({n_missing/len(df)*100:.1f}%)"
    )

    if n_missing > 0:
        # Impute delay_customs_days using route-corridor median
        # This is more realistic than global median because different
        # trade lanes have very different customs processing times
        # (e.g., US customs vs. Singapore customs)
        route_key = df["origin"] + " → " + df["destination"]
        route_medians = (
            df.groupby(route_key)["delay_customs_days"]
            .transform("median")
        )
        global_median = df["delay_customs_days"].median()

        # Use route median where available, fall back to global
        imputed_values = route_medians.fillna(global_median)
        df["delay_customs_days"] = df["delay_customs_days"].fillna(imputed_values)

        # Reconstruct actual_customs_clearance from scheduled + delay
        mask = df["actual_customs_clearance"].isna()
        if "scheduled_customs_clearance" in df.columns:
            df.loc[mask, "actual_customs_clearance"] = (
                df.loc[mask, "scheduled_customs_clearance"]
                + pd.to_timedelta(df.loc[mask, "delay_customs_days"], unit="D")
            )

        logger.info(
            f"Imputed {n_missing} customs delay values "
            f"(route-corridor median strategy)"
        )

    return df


def add_base_transit_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the base (planned) transit duration for each trade lane.

    Base transit time is a fundamental logistics concept — it represents
    the scheduled port-to-port sailing time under normal conditions.
    Deviations from base transit indicate operational disruptions.
    """
    df = df.copy()

    def lookup_base_days(row):
        key = (row["origin"], row["destination"])
        info = TRADE_LANES.get(key)
        return info["base_days"] if info else np.nan

    df["base_transit_days"] = df.apply(lookup_base_days, axis=1)
    return df


def compute_planned_leg_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the planned duration (in days) for each leg of the shipment.

    A shipment journey is divided into legs:
      Leg 1: Departure → Port Arrival    (ocean/air transit)
      Leg 2: Port Arrival → Customs      (port dwell + customs queue)
      Leg 3: Customs → Final Delivery    (last-mile delivery)

    These planned durations serve as reference points for measuring
    how much each stage deviates from plan.
    """
    df = df.copy()

    if all(col in df.columns for col in SCHEDULED_COLS):
        df["planned_leg1_days"] = (
            df["scheduled_port_arrival"] - df["scheduled_departure"]
        ).dt.total_seconds() / 86400

        df["planned_leg2_days"] = (
            df["scheduled_customs_clearance"] - df["scheduled_port_arrival"]
        ).dt.total_seconds() / 86400

        df["planned_leg3_days"] = (
            df["scheduled_final_delivery"] - df["scheduled_customs_clearance"]
        ).dt.total_seconds() / 86400

        df["total_planned_days"] = (
            df["scheduled_final_delivery"] - df["scheduled_departure"]
        ).dt.total_seconds() / 86400

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables for model consumption.

    Uses label encoding for tree-based models (LightGBM/XGBoost handle
    ordinal encoding natively). One-hot encoding is avoided due to the
    high cardinality of origin/destination pairs in real shipping data.
    """
    df = df.copy()

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[f"{col}_encoded"] = df[col].astype("category").cat.codes

    return df


def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline.

    Pipeline order matters:
    1. Parse timestamps first (needed for duration calculations)
    2. Handle missing data (uses route info for imputation)
    3. Add base transit info (lookup table)
    4. Compute planned leg durations (uses parsed timestamps)
    5. Encode categoricals (last, as it creates new columns)
    """
    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)

    df = parse_timestamps(df)
    df = handle_missing_data(df)
    df = add_base_transit_days(df)
    df = compute_planned_leg_durations(df)
    df = encode_categoricals(df)

    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    from src.utils import load_raw_data

    df = load_raw_data()
    df = run_preprocessing(df)
    print("\nPreprocessed columns:")
    print(df.columns.tolist())
    print(f"\nShape: {df.shape}")
    print(f"\nSample:\n{df.head(3).T}")
