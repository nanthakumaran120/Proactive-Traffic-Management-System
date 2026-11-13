"""Tests for real OpenStreetMap road geometry, snapping, map figures, and offline fallbacks."""

from __future__ import annotations

import unittest

from src.emergency_routing import EmergencyPriorityRouter
from src.future_predictor import FutureTrafficPredictor
from src.future_routing import FutureRouteRecommender, haversine_km
from src.map_component import build_real_traffic_map_figure
from src.map_data import build_map_nodes, compute_center_and_zoom, extract_high_congestion_road_segments
from src.road_network import load_or_fetch_gandhipuram_osm_network
from src.route_visualization import (
    compute_unified_emergency_route,
    compute_unified_route_comparison,
)
from src.validation import get_junction_coordinates, load_and_validate_gandhipuram_data


class TestMapAndOSMIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = FutureTrafficPredictor(auto_train=True)
        cls.router = FutureRouteRecommender(cls.predictor)
        cls.emg_router = EmergencyPriorityRouter(cls.predictor, cls.router)
        cls.osm_net = load_or_fetch_gandhipuram_osm_network()
        cls.offline_net = load_or_fetch_gandhipuram_osm_network(force_offline=True)
        cls.df, _ = load_and_validate_gandhipuram_data()
        cls.coords_df = get_junction_coordinates(cls.df)

    def test_six_csv_junction_coordinates_preserved_and_snapped(self) -> None:
        self.assertEqual(len(self.coords_df), 6)
        self.assertTrue(self.osm_net.is_real_osm)
        self.assertGreater(self.osm_net.graph.number_of_nodes(), 100)
        self.assertGreater(len(self.osm_net.way_polylines), 50)

        for _, row in self.coords_df.iterrows():
            jname = str(row["Location"])
            lat = float(row["Latitude"])
            lon = float(row["Longitude"])
            self.assertEqual(self.osm_net.junction_coords[jname], (lat, lon))
            snap = self.osm_net.snapped_nodes[jname]
            self.assertLess(float(snap["snap_distance_m"]), 150.0)

    def test_mapped_route_follows_real_roads_and_not_straight_line(self) -> None:
        origin = "Dr Nanjappa Road"
        dest = "Avarampalayam Road Approach"
        bundle = compute_unified_route_comparison(
            router=self.router,
            road_network=self.osm_net,
            origin=origin,
            destination=dest,
            departure_horizon_hours=4,
            base_timestamp="16-10-2026 12:30",
        )
        self.assertTrue(bundle.uses_real_osm_roads)
        short_rt = bundle.shortest_route
        ai_rt = bundle.ai_recommended_route

        # Mapped street polyline has intermediate OSM curve/intersection coordinates
        self.assertGreater(len(short_rt.polyline_coords), len(short_rt.junction_path))
        self.assertGreater(len(ai_rt.polyline_coords), len(ai_rt.junction_path))
        self.assertTrue(ai_rt.is_real_osm_route)
        self.assertIn("OpenStreetMap", ai_rt.distance_source_label)

        # True mapped road distance must exceed straight-line distance
        straight_km = haversine_km(
            *self.osm_net.junction_coords[origin],
            *self.osm_net.junction_coords[dest],
        )
        self.assertGreater(short_rt.distance_km, straight_km)
        self.assertGreaterEqual(ai_rt.distance_km, straight_km)

    def test_offline_road_network_fallback_is_explicitly_labeled(self) -> None:
        self.assertFalse(self.offline_net.is_real_osm)
        bundle = compute_unified_route_comparison(
            router=self.router,
            road_network=self.offline_net,
            origin="Dr Nanjappa Road",
            destination="Avarampalayam Road Approach",
            departure_horizon_hours=2,
            base_timestamp="16-10-2026 12:30",
            accident_junctions=["Cross Cut Road", "Gandhipuram Bus Stand Approach"],
        )
        self.assertFalse(bundle.uses_real_osm_roads)
        self.assertFalse(bundle.ai_recommended_route.is_real_osm_route)
        self.assertIn("Schematic", bundle.ai_recommended_route.distance_source_label)
        self.assertNotIn("Cross Cut Road", bundle.ai_recommended_route.junction_path)

    def test_emergency_corridor_and_signal_preemption_on_osm_roads(self) -> None:
        eb = compute_unified_emergency_route(
            emergency_router=self.emg_router,
            road_network=self.osm_net,
            origin="Dr Nanjappa Road",
            destination="Avarampalayam Road Approach",
            vehicle_type="Ambulance",
            emergency_mode_active=True,
            departure_horizon_hours=2,
            base_timestamp="16-10-2026 12:30",
            accident_junctions=["Cross Cut Road"],
        )
        self.assertTrue(eb.emergency_mode_active)
        self.assertTrue(eb.uses_real_osm_roads)
        self.assertGreater(len(eb.emergency_route.polyline_coords), 4)
        self.assertLessEqual(eb.estimated_emergency_time_min, eb.normal_travel_time_min)
        self.assertGreaterEqual(len(eb.signal_priority_plans), 2)
        for plan in eb.signal_priority_plans:
            self.assertTrue(plan.simulation_only)
            self.assertTrue(plan.priority_activated)

    def test_future_horizon_updates_map_markers_and_tile_fallback_figure(self) -> None:
        snap_df = self.predictor.get_latest_common_snapshot("16-10-2026 12:30")
        fc_h1 = {f.junction: f for f in self.predictor.predict_horizon(1, base_timestamp="16-10-2026 12:30")}
        fc_h5 = {f.junction: f for f in self.predictor.predict_horizon(5, base_timestamp="16-10-2026 12:30")}

        nodes_h1 = build_map_nodes(snap_df, fc_h1, use_forecast_values=True, selected_junction="Cross Cut Road", road_network=self.osm_net)
        nodes_h5 = build_map_nodes(snap_df, fc_h5, use_forecast_values=True, selected_junction="Cross Cut Road", road_network=self.osm_net)
        self.assertNotEqual([n.display_condition for n in nodes_h1], [n.display_condition for n in nodes_h5])

        high_segs = extract_high_congestion_road_segments(self.osm_net, nodes_h5)
        self.assertGreater(len(high_segs), 0)

        # Online OpenStreetMap figure
        fig_osm = build_real_traffic_map_figure(
            nodes=nodes_h5,
            road_network=self.osm_net,
            map_style="open-street-map",
            theme_mode="light",
        )
        self.assertEqual(fig_osm.layout.map.style, "open-street-map")

        # Offline / Missing tile fallback ("white-bg") + schematic network
        fig_fallback = build_real_traffic_map_figure(
            nodes=nodes_h1,
            road_network=self.offline_net,
            map_style="white-bg",
            theme_mode="dark",
            focus_junction="Cross Cut Road",
        )
        self.assertEqual(fig_fallback.layout.map.style, "white-bg")

        _, _, z_focus = compute_center_and_zoom([(11.020062, 76.965988)])
        self.assertGreaterEqual(z_focus, 16.0)


if __name__ == "__main__":
    unittest.main()

