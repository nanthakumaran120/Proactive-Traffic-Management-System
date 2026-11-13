"""Module 3: Operator-Controlled Emergency Vehicle Priority Routing & Simulation.

IMPORTANT LIMITATION:
The Gandhipuram synthetic CSV does not contain emergency vehicle labels or camera imagery.
This module provides an honest operator-triggered emergency priority routing & signal
preemption simulation, NOT automatic visual detection of emergency vehicles.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import networkx as nx
import pandas as pd

from src.future_predictor import FutureTrafficPredictor, JunctionForecast
from src.future_routing import (
    FutureRouteRecommender,
    RouteSummary,
    _junction_delay_minutes,
)
from src.signal_controller import (
    SIMULATION_DISCLAIMER,
    SignalPlan,
    simulate_emergency_signal_priority,
)


EMERGENCY_VEHICLE_TYPES = ("Ambulance", "Fire Engine", "Police Vehicle")

EMERGENCY_LIMITATION_NOTE = (
    "Operator Simulation Notice: gandhipuram_coimbatore_traffic_synthetic.csv contains no "
    "emergency vehicle labels or camera streams. Emergency Priority Mode is an operator-triggered "
    "simulation using predicted junction congestion and schematic Dijkstra corridor weighting."
)

VEHICLE_PRIORITY_PROFILE = {
    "Ambulance": {"corridor_speed_boost": 1.18, "congestion_aversion": 2.4, "accident_aversion": 8.0},
    "Fire Engine": {"corridor_speed_boost": 1.12, "congestion_aversion": 2.8, "accident_aversion": 10.0},
    "Police Vehicle": {"corridor_speed_boost": 1.22, "congestion_aversion": 2.0, "accident_aversion": 7.0},
}


@dataclass
class EmergencyRoutingResult:
    """Complete output of an operator-triggered emergency routing & signal simulation."""

    emergency_mode_active: bool
    vehicle_type: str
    origin: str
    destination: str
    departure_horizon_hours: int
    departure_timestamp: str
    weather_scenario: str
    accident_junctions: list[str]
    priority_status: str
    normal_route: RouteSummary
    emergency_route: RouteSummary
    estimated_emergency_time_min: float
    normal_travel_time_min: float
    time_saved_min: float
    time_saved_pct: float
    congestion_avoidance_pct: float
    junctions_to_avoid: list[str]
    signal_priority_plans: list[SignalPlan]
    junction_forecasts: dict[str, dict[str, Any]]
    graph_nodes: list[dict[str, Any]]
    limitation_note: str = EMERGENCY_LIMITATION_NOTE
    simulation_disclaimer: str = SIMULATION_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmergencyPriorityRouter:
    """Computes emergency priority paths avoiding high congestion and accidents."""

    def __init__(
        self,
        predictor: FutureTrafficPredictor | None = None,
        route_recommender: FutureRouteRecommender | None = None,
    ) -> None:
        self.predictor = predictor if predictor is not None else FutureTrafficPredictor()
        self.route_recommender = (
            route_recommender if route_recommender is not None else FutureRouteRecommender(self.predictor)
        )

    def build_emergency_graph(
        self,
        forecasts_by_junction: dict[str, JunctionForecast],
        vehicle_type: str = "Ambulance",
        weather_scenario: str = "Clear",
    ) -> nx.DiGraph:
        """Build emergency cost graph strongly penalizing High congestion & accidents."""
        graph = self.route_recommender.build_weighted_graph(
            forecasts_by_junction, weather_scenario=weather_scenario
        )
        profile = VEHICLE_PRIORITY_PROFILE.get(vehicle_type, VEHICLE_PRIORITY_PROFILE["Ambulance"])
        speed_boost = profile["corridor_speed_boost"]
        cong_aversion = profile["congestion_aversion"]
        acc_aversion = profile["accident_aversion"]

        for u, v, data in graph.edges(data=True):
            fc_u = forecasts_by_junction[u]
            fc_v = forecasts_by_junction[v]
            dist_km = float(data["distance_km"])

            # Preempted emergency corridor eliminates routine red-light stops at non-accident nodes,
            # though dense traffic / accidents still constrain physical lane clearance.
            base_seg_speed = float(data["segment_speed_kmh"])
            if fc_v.accident_scenario == "Yes" or fc_u.accident_scenario == "Yes":
                eff_speed = max(5.0, base_seg_speed * 0.75)
                preempt_delay = 2.5
            elif fc_v.predicted_condition == "High":
                eff_speed = min(38.0, base_seg_speed * speed_boost)
                preempt_delay = 0.45
            else:
                eff_speed = min(42.0, base_seg_speed * speed_boost)
                preempt_delay = 0.10

            emg_time_min = (dist_km / eff_speed) * 60.0 + preempt_delay

            cond_factor = {"Low": 0.0, "Medium": 0.45, "High": cong_aversion}.get(
                fc_v.predicted_condition, 0.45
            )
            occ_factor = max(0.0, (fc_v.predicted_occupancy_pct - 25.0) / 60.0)
            acc_factor = acc_aversion if (fc_v.accident_scenario == "Yes" or fc_u.accident_scenario == "Yes") else 0.0

            emg_cost = emg_time_min * (1.0 + cond_factor + occ_factor + acc_factor)
            data["emergency_speed_kmh"] = round(eff_speed, 2)
            data["emergency_time_min"] = round(emg_time_min, 3)
            data["emergency_cost"] = round(emg_cost, 4)

        return graph

    @staticmethod
    def _evaluate_emergency_path(
        graph: nx.DiGraph,
        path: list[str],
        forecasts_by_junction: dict[str, JunctionForecast],
        vehicle_type: str,
    ) -> RouteSummary:
        total_dist = 0.0
        total_time = 0.0
        total_cost = 0.0
        weighted_speed_num = 0.0

        for i in range(len(path) - 1):
            edge = graph[path[i]][path[i + 1]]
            d = float(edge["distance_km"])
            total_dist += d
            total_time += float(edge["emergency_time_min"])
            total_cost += float(edge["emergency_cost"])
            weighted_speed_num += d * float(edge["emergency_speed_kmh"])

        avg_speed = weighted_speed_num / total_dist if total_dist > 0 else 20.0
        risks = [forecasts_by_junction[n].congestion_risk_score for n in path]
        avg_risk = float(sum(risks) / len(risks)) if risks else 0.0
        node_conds = {n: forecasts_by_junction[n].predicted_condition for n in path}
        acc_nodes = [n for n in path if forecasts_by_junction[n].accident_scenario == "Yes"]
        overall_cond = "High" if "High" in node_conds.values() else ("Medium" if "Medium" in node_conds.values() else "Low")

        return RouteSummary(
            name=f"Emergency Priority Route ({vehicle_type})",
            path=list(path),
            distance_km=round(total_dist, 3),
            estimated_travel_time_min=round(total_time, 2),
            predicted_congestion=overall_cond,
            average_predicted_speed_kmh=round(avg_speed, 2),
            average_risk_score=round(avg_risk, 1),
            total_ai_cost=round(total_cost, 3),
            junction_conditions=node_conds,
            accident_affected_nodes=acc_nodes,
        )

    def compute_emergency_route(
        self,
        origin: str,
        destination: str,
        vehicle_type: str = "Ambulance",
        emergency_mode_active: bool = True,
        departure_horizon_hours: int = 1,
        base_timestamp: pd.Timestamp | str | None = None,
        weather_scenario: str = "Clear",
        accident_junctions: list[str] | set[str] | None = None,
    ) -> EmergencyRoutingResult:
        """Compute emergency corridor route, avoided junctions, and green-signal simulation."""
        if origin not in self.route_recommender.node_coords:
            raise ValueError(f"Unknown origin junction: {origin}")
        if destination not in self.route_recommender.node_coords:
            raise ValueError(f"Unknown destination junction: {destination}")
        if origin == destination:
            raise ValueError("Origin and destination junctions must be different.")
        if vehicle_type not in EMERGENCY_VEHICLE_TYPES:
            raise ValueError(f"Unsupported emergency vehicle type: {vehicle_type}")

        forecasts = self.predictor.predict_horizon(
            horizon_hours=departure_horizon_hours,
            base_timestamp=base_timestamp,
            weather_override=weather_scenario,
            accident_junctions=accident_junctions,
        )
        fc_map = {fc.junction: fc for fc in forecasts}
        emg_graph = self.build_emergency_graph(
            fc_map, vehicle_type=vehicle_type, weather_scenario=weather_scenario
        )

        # Normal baseline (standard shortest-distance route under current predicted traffic)
        normal_path = nx.dijkstra_path(emg_graph, origin, destination, weight="distance_km")
        normal_summary = self.route_recommender.evaluate_path(
            emg_graph, normal_path, fc_map, "Normal Shortest Route"
        )

        if emergency_mode_active:
            emg_path = nx.dijkstra_path(emg_graph, origin, destination, weight="emergency_cost")
            emg_summary = self._evaluate_emergency_path(emg_graph, emg_path, fc_map, vehicle_type)
            signal_plans = simulate_emergency_signal_priority(
                emg_path,
                vehicle_type=vehicle_type,
                junction_states={k: v.to_dict() for k, v in fc_map.items()},
            )
            priority_status = f"ACTIVE — {vehicle_type.upper()} GREEN CORRIDOR SIMULATION"
        else:
            emg_path = list(normal_path)
            emg_summary = RouteSummary(
                name="Normal Route (Emergency Mode OFF)",
                path=normal_summary.path,
                distance_km=normal_summary.distance_km,
                estimated_travel_time_min=normal_summary.estimated_travel_time_min,
                predicted_congestion=normal_summary.predicted_congestion,
                average_predicted_speed_kmh=normal_summary.average_predicted_speed_kmh,
                average_risk_score=normal_summary.average_risk_score,
                total_ai_cost=normal_summary.total_ai_cost,
                junction_conditions=normal_summary.junction_conditions,
                accident_affected_nodes=normal_summary.accident_affected_nodes,
            )
            signal_plans = []
            priority_status = "STANDBY — Emergency Mode OFF (Normal Routing Active)"

        t_norm = normal_summary.estimated_travel_time_min
        t_emg = emg_summary.estimated_travel_time_min
        time_saved_min = round(max(0.0, t_norm - t_emg), 2)
        time_saved_pct = round(((t_norm - t_emg) / t_norm) * 100.0, 1) if (t_norm > 0 and t_emg < t_norm) else 0.0

        r_norm = normal_summary.average_risk_score
        r_emg = emg_summary.average_risk_score
        congestion_avoidance_pct = round(((r_norm - r_emg) / r_norm) * 100.0, 1) if (r_norm > 0 and r_emg < r_norm) else 0.0

        avoid_list = sorted(
            [
                junc
                for junc, fc in fc_map.items()
                if (fc.predicted_condition == "High" or fc.accident_scenario == "Yes")
                and junc not in {origin, destination}
            ]
        )

        graph_nodes = [
            {
                "junction": n,
                "latitude": float(emg_graph.nodes[n]["latitude"]),
                "longitude": float(emg_graph.nodes[n]["longitude"]),
                "predicted_condition": fc_map[n].predicted_condition,
                "risk_score": fc_map[n].congestion_risk_score,
                "predicted_speed_kmh": fc_map[n].predicted_speed_kmh,
                "accident_scenario": fc_map[n].accident_scenario,
            }
            for n in sorted(emg_graph.nodes)
        ]

        return EmergencyRoutingResult(
            emergency_mode_active=emergency_mode_active,
            vehicle_type=vehicle_type,
            origin=origin,
            destination=destination,
            departure_horizon_hours=departure_horizon_hours,
            departure_timestamp=forecasts[0].target_timestamp if forecasts else "",
            weather_scenario=weather_scenario,
            accident_junctions=sorted(list(accident_junctions or [])),
            priority_status=priority_status,
            normal_route=normal_summary,
            emergency_route=emg_summary,
            estimated_emergency_time_min=t_emg,
            normal_travel_time_min=t_norm,
            time_saved_min=time_saved_min,
            time_saved_pct=time_saved_pct,
            congestion_avoidance_pct=congestion_avoidance_pct,
            junctions_to_avoid=avoid_list,
            signal_priority_plans=signal_plans,
            junction_forecasts={k: v.to_dict() for k, v in fc_map.items()},
            graph_nodes=graph_nodes,
        )

