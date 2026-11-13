"""Tests for operator-controlled emergency vehicle priority routing and signal simulation."""

from __future__ import annotations

import unittest

from src.emergency_routing import EMERGENCY_VEHICLE_TYPES, EmergencyPriorityRouter
from src.future_predictor import FutureTrafficPredictor
from src.future_routing import FutureRouteRecommender


class TestEmergencyRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = FutureTrafficPredictor(auto_train=True)
        cls.router = FutureRouteRecommender(cls.predictor)
        cls.emg_router = EmergencyPriorityRouter(cls.predictor, cls.router)

    def test_emergency_mode_on_vs_off_and_signal_simulation(self) -> None:
        off_res = self.emg_router.compute_emergency_route(
            origin="Dr Nanjappa Road",
            destination="Avarampalayam Road Approach",
            vehicle_type="Ambulance",
            emergency_mode_active=False,
            departure_horizon_hours=2,
            base_timestamp="16-10-2026 12:30",
            accident_junctions=["Cross Cut Road"],
        )
        on_res = self.emg_router.compute_emergency_route(
            origin="Dr Nanjappa Road",
            destination="Avarampalayam Road Approach",
            vehicle_type="Ambulance",
            emergency_mode_active=True,
            departure_horizon_hours=2,
            base_timestamp="16-10-2026 12:30",
            accident_junctions=["Cross Cut Road"],
        )

        self.assertFalse(off_res.emergency_mode_active)
        self.assertEqual(len(off_res.signal_priority_plans), 0)

        self.assertTrue(on_res.emergency_mode_active)
        self.assertNotIn("Cross Cut Road", on_res.emergency_route.path)
        self.assertLess(on_res.estimated_emergency_time_min, on_res.normal_travel_time_min)
        self.assertEqual(len(on_res.signal_priority_plans), len(on_res.emergency_route.path))
        for plan in on_res.signal_priority_plans:
            self.assertTrue(plan.priority_activated)
            self.assertTrue(plan.simulation_only)
            self.assertGreater(plan.proposed_green_s, plan.normal_green_s)

    def test_all_emergency_vehicle_types_supported(self) -> None:
        for vtype in EMERGENCY_VEHICLE_TYPES:
            res = self.emg_router.compute_emergency_route(
                origin="Gandhipuram Bus Stand Approach",
                destination="Avarampalayam Road Approach",
                vehicle_type=vtype,
                emergency_mode_active=True,
                departure_horizon_hours=4,
                base_timestamp="16-10-2026 12:30",
            )
            self.assertEqual(res.vehicle_type, vtype)
            self.assertGreater(len(res.emergency_route.path), 1)
            self.assertGreater(len(res.signal_priority_plans), 1)


if __name__ == "__main__":
    unittest.main()

