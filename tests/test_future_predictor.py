"""Tests for dataset validation, chronological ML forecasting, and proactive alerts."""

from __future__ import annotations

import unittest

from src.future_predictor import FutureTrafficPredictor, MODELS_DIR
from src.validation import load_and_validate_gandhipuram_data


class TestFuturePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df, cls.report = load_and_validate_gandhipuram_data()
        cls.predictor = FutureTrafficPredictor(auto_train=True)

    def test_dataset_validation_strict_gandhipuram_csv(self) -> None:
        self.assertTrue(self.report.passed)
        self.assertEqual(self.report.raw_row_count, 240)
        self.assertEqual(self.report.empty_rows_removed, 22)
        self.assertEqual(self.report.valid_row_count, 218)
        self.assertEqual(self.report.junction_count, 6)
        self.assertEqual(set(self.report.valid_labels_found), {"Low", "Medium", "High"})
        self.assertEqual(self.report.duplicate_rows, 0)
        self.assertEqual(sum(self.report.missing_values_after_cleaning.values()), 0)

    def test_trained_models_and_chronological_metadata_exist(self) -> None:
        self.assertTrue((MODELS_DIR / "gandhipuram_classifier.pkl").exists())
        self.assertTrue((MODELS_DIR / "gandhipuram_reg_vehicle_count.pkl").exists())
        self.assertTrue((MODELS_DIR / "gandhipuram_reg_speed.pkl").exists())
        self.assertTrue((MODELS_DIR / "gandhipuram_reg_occupancy.pkl").exists())
        self.assertTrue((MODELS_DIR / "gandhipuram_training_metadata.json").exists())

        meta = self.predictor.metadata
        best_acc = meta["classifier_comparison"][0]["accuracy"]
        pers_acc = meta["persistence_baseline"]["accuracy"]
        self.assertGreater(best_acc, pers_acc)
        self.assertEqual(len(meta["chronological_eval_dates"]), 3)

    def test_all_six_junctions_multi_horizon_forecast(self) -> None:
        forecasts = self.predictor.forecast_multi_horizon(max_horizon=6)
        self.assertEqual(len(forecasts), 36)  # 6 junctions x 6 horizons
        junctions_seen = {f.junction for f in forecasts}
        self.assertEqual(len(junctions_seen), 6)
        for fc in forecasts:
            self.assertIn(fc.predicted_condition, {"Low", "Medium", "High"})
            self.assertGreater(fc.predicted_vehicle_count, 0.0)
            self.assertGreaterEqual(fc.predicted_speed_kmh, 5.0)
            self.assertGreaterEqual(fc.congestion_risk_score, 0.0)
            self.assertLessEqual(fc.congestion_risk_score, 100.0)
            self.assertGreater(fc.prediction_confidence, 0.0)
            self.assertTrue(fc.is_model_forecast)

    def test_forecast_does_not_simply_copy_latest_condition(self) -> None:
        # From 16-10-2026 12:30 (Medium), +5h reaches 17:30 evening peak (High)
        fcs_1230 = self.predictor.forecast_multi_horizon(max_horizon=6, base_timestamp="16-10-2026 12:30")
        h5_preds = [f for f in fcs_1230 if f.horizon_hours == 5]
        self.assertTrue(any(f.predicted_condition != f.current_condition for f in h5_preds))

        # From 16-10-2026 07:30 (High), +5h reaches 12:30 mid-day (Medium/Low)
        fcs_0730 = self.predictor.forecast_multi_horizon(max_horizon=6, base_timestamp="16-10-2026 07:30")
        h5_0730 = [f for f in fcs_0730 if f.horizon_hours == 5]
        self.assertTrue(any(f.predicted_condition != f.current_condition for f in h5_0730))

    def test_accident_scenario_and_proactive_alerts(self) -> None:
        normal_h2 = self.predictor.predict_horizon(horizon_hours=2, base_timestamp="16-10-2026 12:30")
        acc_h2 = self.predictor.predict_horizon(
            horizon_hours=2,
            base_timestamp="16-10-2026 12:30",
            accident_junctions=["Cross Cut Road"],
        )
        n_cc = next(f for f in normal_h2 if f.junction == "Cross Cut Road")
        a_cc = next(f for f in acc_h2 if f.junction == "Cross Cut Road")
        self.assertGreater(a_cc.congestion_risk_score, n_cc.congestion_risk_score)
        self.assertLess(a_cc.predicted_speed_kmh, n_cc.predicted_speed_kmh)
        self.assertEqual(a_cc.predicted_condition, "High")

        multi = self.predictor.forecast_multi_horizon(max_horizon=6, base_timestamp="16-10-2026 12:30")
        alert_bundle = self.predictor.generate_proactive_alerts(multi)
        self.assertEqual(len(alert_bundle["alerts"]), 6)
        self.assertIsNotNone(alert_bundle["earliest_congested_junction"])
        self.assertIsNotNone(alert_bundle["earliest_congestion_time"])


if __name__ == "__main__":
    unittest.main()

