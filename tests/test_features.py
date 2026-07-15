"""
test_features.py — Unit Tests for Feature Engineering Pipeline

Validates that all feature transformations produce expected outputs,
handle edge cases, and maintain data integrity.
"""

import sys
import os
import unittest

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (
    add_cumulative_delay_features,
    add_delay_ratio_features,
    add_delay_propagation_features,
    add_route_features,
    add_external_interaction_features,
    add_temporal_features,
    add_data_quality_features,
    get_feature_columns,
    build_features,
)
from src.preprocessing import run_preprocessing
from src.utils import load_raw_data


def create_sample_dataframe():
    """Create a minimal sample DataFrame for testing."""
    return pd.DataFrame({
        "shipment_id": [1, 2, 3],
        "origin": ["Shanghai", "Mumbai", "Singapore"],
        "destination": ["Los Angeles", "London", "Dubai"],
        "route_type": ["sea", "air", "sea"],
        "congestion_index": [0.3, 0.1, 0.8],
        "weather_index": [0.2, 0.05, 0.5],
        "delay_departure_days": [0.5, 0.1, 1.0],
        "delay_port_days": [1.0, 0.2, 2.0],
        "delay_customs_days": [0.8, 0.15, 1.5],
        "delay_final_days": [1.2, 0.3, 2.5],
        "is_delayed": [1, 0, 1],
        "scheduled_departure": ["2025-03-01", "2025-07-15", "2025-11-20"],
        "actual_departure": ["2025-03-01 12:00", "2025-07-15 02:00", "2025-11-21 00:00"],
        "scheduled_port_arrival": ["2025-03-08", "2025-07-16", "2025-11-25"],
        "actual_port_arrival": ["2025-03-09", "2025-07-16 05:00", "2025-11-27"],
        "scheduled_customs_clearance": ["2025-03-13", "2025-07-16 08:00", "2025-11-28"],
        "actual_customs_clearance": ["2025-03-14", "2025-07-16 12:00", "2025-11-30"],
        "scheduled_final_delivery": ["2025-03-15", "2025-07-16 12:00", "2025-11-30"],
        "actual_final_delivery": ["2025-03-16 05:00", "2025-07-16 19:00", "2025-12-02 12:00"],
        "customs_data_missing": [0, 0, 0],
        "base_transit_days": [14, 1, 10],
        "planned_leg1_days": [7.0, 1.0, 5.0],
        "planned_leg2_days": [5.0, 0.33, 3.0],
        "planned_leg3_days": [2.0, 0.17, 2.0],
        "total_planned_days": [14.0, 1.5, 10.0],
        "origin_encoded": [0, 1, 2],
        "destination_encoded": [0, 1, 2],
        "route_type_encoded": [1, 0, 1],
    })


class TestCumulativeDelayFeatures(unittest.TestCase):
    """Test cumulative delay computation."""

    def setUp(self):
        self.df = create_sample_dataframe()

    def test_cumulative_delay_values(self):
        result = add_cumulative_delay_features(self.df)

        # cumulative_delay_dep should equal departure delay
        np.testing.assert_array_almost_equal(
            result["cumulative_delay_dep"],
            self.df["delay_departure_days"],
        )

        # cumulative_delay_port = departure + port
        expected_port = (
            self.df["delay_departure_days"] + self.df["delay_port_days"]
        )
        np.testing.assert_array_almost_equal(
            result["cumulative_delay_port"], expected_port,
        )

        # cumulative_delay_customs = departure + port + customs
        expected_customs = (
            self.df["delay_departure_days"]
            + self.df["delay_port_days"]
            + self.df["delay_customs_days"]
        )
        np.testing.assert_array_almost_equal(
            result["cumulative_delay_customs"], expected_customs,
        )

    def test_no_original_columns_modified(self):
        result = add_cumulative_delay_features(self.df)
        pd.testing.assert_series_equal(
            result["delay_departure_days"],
            self.df["delay_departure_days"],
        )


class TestDelayPropagationFeatures(unittest.TestCase):
    """Test delay propagation feature computation."""

    def setUp(self):
        self.df = create_sample_dataframe()

    def test_delay_acceleration(self):
        result = add_delay_propagation_features(self.df)

        # delay_accel_dep_to_port = port_delay - departure_delay
        expected = self.df["delay_port_days"] - self.df["delay_departure_days"]
        np.testing.assert_array_almost_equal(
            result["delay_accel_dep_to_port"], expected,
        )

    def test_propagation_ratio(self):
        result = add_delay_propagation_features(self.df)
        eps = 0.01

        # propagation_dep_to_port = port / (departure + eps)
        expected = self.df["delay_port_days"] / (
            self.df["delay_departure_days"] + eps
        )
        np.testing.assert_array_almost_equal(
            result["propagation_dep_to_port"], expected,
        )


class TestRouteFeatures(unittest.TestCase):
    """Test route-based feature engineering."""

    def setUp(self):
        self.df = create_sample_dataframe()

    def test_sea_route_indicator(self):
        result = add_route_features(self.df)
        self.assertEqual(result.iloc[0]["is_sea_route"], 1)  # Shanghai-LA = sea
        self.assertEqual(result.iloc[1]["is_sea_route"], 0)  # Mumbai-London = air

    def test_route_factor(self):
        result = add_route_features(self.df)
        self.assertEqual(result.iloc[0]["route_factor"], 1.5)  # sea
        self.assertEqual(result.iloc[1]["route_factor"], 0.5)  # air


class TestExternalInteractions(unittest.TestCase):
    """Test external factor interaction features."""

    def setUp(self):
        self.df = create_sample_dataframe()

    def test_congestion_weather_interaction(self):
        result = add_external_interaction_features(self.df)
        expected = self.df["congestion_index"] * self.df["weather_index"]
        np.testing.assert_array_almost_equal(
            result["congestion_weather_interaction"], expected,
        )

    def test_high_disruption_flag(self):
        result = add_external_interaction_features(self.df)
        # Only shipment 3 has congestion=0.8>0.5 AND weather=0.5>0.3
        self.assertEqual(result.iloc[2]["high_disruption_flag"], 1)
        self.assertEqual(result.iloc[0]["high_disruption_flag"], 0)


class TestTemporalFeatures(unittest.TestCase):
    """Test temporal feature extraction."""

    def setUp(self):
        self.df = create_sample_dataframe()

    def test_peak_season_detection(self):
        result = add_temporal_features(self.df)
        # Nov (month=11) is peak season
        self.assertEqual(result.iloc[2]["is_peak_season"], 1)
        # March (month=3) is not peak
        self.assertEqual(result.iloc[0]["is_peak_season"], 0)

    def test_departure_month(self):
        result = add_temporal_features(self.df)
        self.assertEqual(result.iloc[0]["departure_month"], 3)
        self.assertEqual(result.iloc[1]["departure_month"], 7)


class TestFeatureColumns(unittest.TestCase):
    """Test that feature column list is consistent."""

    def test_feature_list_not_empty(self):
        cols = get_feature_columns()
        self.assertGreater(len(cols), 20)

    def test_no_duplicate_features(self):
        cols = get_feature_columns()
        self.assertEqual(len(cols), len(set(cols)))


class TestFullPipeline(unittest.TestCase):
    """Integration test: run the full pipeline on real data."""

    def test_full_pipeline_on_real_data(self):
        """Verify the complete pipeline runs without errors."""
        df = load_raw_data()
        df = run_preprocessing(df)
        df = build_features(df)

        feature_cols = get_feature_columns()
        available = [c for c in feature_cols if c in df.columns]

        # Should have at least 30 features
        self.assertGreaterEqual(len(available), 30)

        # No all-NaN columns in features
        for col in available:
            self.assertFalse(
                df[col].isna().all(),
                f"Feature {col} is entirely NaN",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
