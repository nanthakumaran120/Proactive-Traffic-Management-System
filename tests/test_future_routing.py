"""Tests for NetworkX + Dijkstra future-time route recommendation."""

from __future__ import annotations

import unittest

import networkx as nx

from src.future_predictor import FutureTrafficPredictor
from src.future_routing import FutureRouteRecommender


class TestFutureRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = FutureTrafficPredictor(auto_train=True)
        cls.router = FutureRouteRecommender(cls.predictor)

    def test_schematic_graph_connectivity_and_coordinates(self) -> None:
        g = self.router.build_base_graph()
        self.assertEqual(g.number_of_nodes(), 6)
        self.assertTrue(nx.is_strongly_connected(g))
        for _, _, d in g.edges(data=True):
            self.assertGreater(d["distance_km"], 0.0)

    def test_departure_time_changes_future_route_calculation(self) -> None:
        res_h1 = self.router.recommend_route(
            origin="100 Feet Road Junction",
            destination="Cross Cut Road",
            departure_horizon_hours=1,
            base_timestamp="16-10-2026 12:30",
        )
        res_h5 = self.router.recommend_route(
            origin="100 Feet Road Junction",
            destination="Cross Cut Road",
            departure_horizon_hours=5,
            base_timestamp="16-10-2026 12:30",
        )
        # Travel time & cost must reflect future congestion shift, and path switches at peak
        self.assertNotEqual(
            res_h1.ai_recommended_route.estimated_travel_time_min,
            res_h5.ai_recommended_route.estimated_travel_time_min,
        )
        self.assertNotEqual(
            res_h1.ai_recommended_route.path,
            res_h5.ai_recommended_route.path,
        )

    def test_accident_reroutes_around_congested_junction(self) -> None:
        res = self.router.recommend_route(
            origin="Dr Nanjappa Road",
            destination="Avarampalayam Road Approach",
            departure_horizon_hours=2,
            base_timestamp="16-10-2026 12:30",
            accident_junctions=["Cross Cut Road", "Gandhipuram Bus Stand Approach"],
        )
        self.assertIn("Cross Cut Road", res.shortest_route.path)
        self.assertNotIn("Cross Cut Road", res.ai_recommended_route.path)
        self.assertLess(
            res.ai_recommended_route.estimated_travel_time_min,
            res.shortest_route.estimated_travel_time_min,
        )
        self.assertGreater(res.time_saved_pct, 0.0)
        self.assertEqual(len(res.comparison_table), 6)


if __name__ == "__main__":
    unittest.main()

