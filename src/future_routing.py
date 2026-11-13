"""Module 2: Future-Time Intelligent Route Recommendation (NetworkX + Dijkstra).

Uses ONLY junction names and coordinates from datasets/gandhipuram_coimbatore_traffic_synthetic.csv
over a clearly documented schematic corridor connectivity graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import networkx as nx
import pandas as pd

from src.future_predictor import FutureTrafficPredictor, JunctionForecast
from src.validation import get_junction_coordinates


GRAPH_LIMITATION_NOTE = (
    "Schematic Connectivity Notice: The uploaded CSV provides junction coordinates but not "
    "physical road polylines or turn restrictions. Edges represent a modular schematic "
    "corridor network connecting the 6 Gandhipuram junctions using Haversine distances."
)

# Schematic bi-directional corridor links connecting all 6 Gandhipuram junctions.
# Modular so real road segments/one-way restrictions can be substituted later.
SCHEMATIC_CORRIDORS: list[tuple[str, str, str]] = [
    ("Dr Nanjappa Road", "Gandhipuram Bus Stand Approach", "Dr Nanjappa Rd - Bus Stand Corridor"),
    ("Dr Nanjappa Road", "Cross Cut Road", "Dr Nanjappa Rd - Cross Cut West Link"),
    ("Dr Nanjappa Road", "100 Feet Road Junction", "Dr Nanjappa Rd - 100 Feet Rd Radial"),
    ("Gandhipuram Bus Stand Approach", "Cross Cut Road", "Bus Stand - Cross Cut Central Link"),
    ("Gandhipuram Bus Stand Approach", "Sathy Road Junction", "Bus Stand - Sathy Rd Diagonal"),
    ("Gandhipuram Bus Stand Approach", "100 Feet Road Junction", "Bus Stand - 100 Feet Rd Corridor"),
    ("Cross Cut Road", "Sathy Road Junction", "Cross Cut - Sathy Rd Crosstown"),
    ("Cross Cut Road", "Avarampalayam Road Approach", "Cross Cut - Avarampalayam North-West"),
    ("Sathy Road Junction", "100 Feet Road Junction", "Sathy Rd - 100 Feet Rd East Link"),
    ("Sathy Road Junction", "Avarampalayam Road Approach", "Sathy Rd - Avarampalayam North-East"),
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two WGS-84 coordinates in kilometres."""
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


@dataclass
class RouteSummary:
    """Evaluated metrics for a candidate path across the Gandhipuram graph."""

    name: str
    path: list[str]
    distance_km: float
    estimated_travel_time_min: float
    predicted_congestion: str
    average_predicted_speed_kmh: float
    average_risk_score: float
    total_ai_cost: float
    junction_conditions: dict[str, str]
    accident_affected_nodes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteComparisonResult:
    """Complete result for a future-time route recommendation query."""

    origin: str
    destination: str
    departure_horizon_hours: int
    departure_timestamp: str
    weather_scenario: str
    accident_junctions: list[str]
    shortest_route: RouteSummary
    ai_recommended_route: RouteSummary
    alternative_routes: list[RouteSummary]
    comparison_table: list[dict[str, Any]]
    time_saved_min: float
    time_saved_pct: float
    congestion_avoidance_pct: float
    junction_forecasts: dict[str, dict[str, Any]]
    graph_nodes: list[dict[str, Any]]
    graph_edges: list[dict[str, Any]]
    graph_limitation_note: str = GRAPH_LIMITATION_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _condition_from_risk(risk_score: float, junction_conditions: list[str]) -> str:
    if "High" in junction_conditions or risk_score >= 70.0:
        return "High"
    if "Medium" in junction_conditions or risk_score >= 38.0:
        return "Medium"
    return "Low"


def _junction_delay_minutes(condition: str, accident: str) -> float:
    base = {"Low": 0.25, "Medium": 0.70, "High": 1.65}.get(condition, 0.70)
    if accident == "Yes":
        base += 2.80
    return base


class FutureRouteRecommender:
    """NetworkX + Dijkstra route recommendation driven by future ML traffic forecasts."""

    def __init__(self, predictor: FutureTrafficPredictor | None = None) -> None:
        self.predictor = predictor if predictor is not None else FutureTrafficPredictor()
        self.coords_df = get_junction_coordinates(self.predictor.df)
        self.node_coords: dict[str, tuple[float, float]] = {
            str(r["Location"]): (float(r["Latitude"]), float(r["Longitude"]))
            for _, r in self.coords_df.iterrows()
        }

    def build_base_graph(self) -> nx.DiGraph:
        """Build the schematic directed graph with exact CSV coordinates and Haversine distances."""
        graph = nx.DiGraph()
        for loc, (lat, lon) in self.node_coords.items():
            graph.add_node(loc, latitude=lat, longitude=lon)

        for u, v, road_name in SCHEMATIC_CORRIDORS:
            if u not in self.node_coords or v not in self.node_coords:
                continue
            lat1, lon1 = self.node_coords[u]
            lat2, lon2 = self.node_coords[v]
            dist_km = round(haversine_km(lat1, lon1, lat2, lon2), 4)
            graph.add_edge(u, v, distance_km=dist_km, road_name=road_name)
            graph.add_edge(v, u, distance_km=dist_km, road_name=road_name)
        return graph

    def build_weighted_graph(
        self,
        forecasts_by_junction: dict[str, JunctionForecast],
        weather_scenario: str = "Clear",
    ) -> nx.DiGraph:
        """Annotate every edge with predicted segment speed, travel time, and AI routing cost."""
        graph = self.build_base_graph()
        for loc, fc in forecasts_by_junction.items():
            if loc in graph.nodes:
                graph.nodes[loc].update(fc.to_dict())

        weather_penalty = {"Clear": 0.0, "Cloudy": 0.05, "Light Rain": 0.18}.get(weather_scenario, 0.0)

        for u, v, data in graph.edges(data=True):
            fc_u = forecasts_by_junction[u]
            fc_v = forecasts_by_junction[v]
            dist_km = float(data["distance_km"])

            # Weight entry junction v slightly higher because vehicles queue entering v
            seg_speed = max(5.0, 0.35 * fc_u.predicted_speed_kmh + 0.65 * fc_v.predicted_speed_kmh)
            delay_min = _junction_delay_minutes(fc_v.predicted_condition, fc_v.accident_scenario)
            travel_time_min = (dist_km / seg_speed) * 60.0 + delay_min

            cond_penalty = {"Low": 0.0, "Medium": 0.30, "High": 1.45}.get(fc_v.predicted_condition, 0.30)
            occ_penalty = max(0.0, (fc_v.predicted_occupancy_pct - 35.0) / 100.0)
            risk_penalty = fc_v.congestion_risk_score / 80.0
            accident_penalty = 4.5 if (fc_v.accident_scenario == "Yes" or fc_u.accident_scenario == "Yes") else 0.0

            ai_cost = travel_time_min * (1.0 + cond_penalty + occ_penalty + risk_penalty + weather_penalty + accident_penalty)

            data["segment_speed_kmh"] = round(seg_speed, 2)
            data["travel_time_min"] = round(travel_time_min, 3)
            data["ai_cost"] = round(ai_cost, 4)
            data["target_condition"] = fc_v.predicted_condition
            data["target_risk_score"] = fc_v.congestion_risk_score
        return graph

    @staticmethod
    def evaluate_path(
        graph: nx.DiGraph,
        path: list[str],
        forecasts_by_junction: dict[str, JunctionForecast],
        name: str,
        cost_attr: str = "ai_cost",
    ) -> RouteSummary:
        """Compute distance, predicted travel time, speed, and risk metrics along a path."""
        total_dist = 0.0
        total_time = 0.0
        total_cost = 0.0
        weighted_speed_num = 0.0

        for i in range(len(path) - 1):
            edge = graph[path[i]][path[i + 1]]
            d = float(edge["distance_km"])
            total_dist += d
            total_time += float(edge["travel_time_min"])
            total_cost += float(edge.get(cost_attr, edge["travel_time_min"]))
            weighted_speed_num += d * float(edge["segment_speed_kmh"])

        avg_speed = (weighted_speed_num / total_dist) if total_dist > 0 else forecasts_by_junction[path[0]].predicted_speed_kmh
        risks = [forecasts_by_junction[n].congestion_risk_score for n in path]
        avg_risk = float(sum(risks) / len(risks)) if risks else 0.0
        node_conds = {n: forecasts_by_junction[n].predicted_condition for n in path}
        acc_nodes = [n for n in path if forecasts_by_junction[n].accident_scenario == "Yes"]
        overall_cond = _condition_from_risk(avg_risk, list(node_conds.values()))

        return RouteSummary(
            name=name,
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

    def recommend_route(
        self,
        origin: str,
        destination: str,
        departure_horizon_hours: int = 1,
        base_timestamp: pd.Timestamp | str | None = None,
        weather_scenario: str = "Clear",
        accident_junctions: list[str] | set[str] | None = None,
        top_k_alternatives: int = 3,
    ) -> RouteComparisonResult:
        """Compute Route 1 (Shortest Distance) vs Route 2 (Future AI Route) and alternatives."""
        if origin not in self.node_coords:
            raise ValueError(f"Unknown origin junction: {origin}")
        if destination not in self.node_coords:
            raise ValueError(f"Unknown destination junction: {destination}")
        if origin == destination:
            raise ValueError("Origin and destination junctions must be different.")

        forecasts = self.predictor.predict_horizon(
            horizon_hours=departure_horizon_hours,
            base_timestamp=base_timestamp,
            weather_override=weather_scenario,
            accident_junctions=accident_junctions,
        )
        fc_map = {fc.junction: fc for fc in forecasts}
        graph = self.build_weighted_graph(fc_map, weather_scenario=weather_scenario)

        # Route 1: Normal Shortest Route (pure distance_km Dijkstra)
        shortest_path = nx.dijkstra_path(graph, origin, destination, weight="distance_km")
        shortest_summary = self.evaluate_path(graph, shortest_path, fc_map, "Normal Shortest Route")

        # Route 2: Future AI Recommended Route (ai_cost Dijkstra)
        ai_path = nx.dijkstra_path(graph, origin, destination, weight="ai_cost")
        ai_summary = self.evaluate_path(graph, ai_path, fc_map, "Future AI Recommended Route")

        # Alternative candidate paths ordered by AI cost
        alternatives: list[RouteSummary] = []
        seen_paths = {tuple(ai_path)}
        for idx, cand_path in enumerate(nx.shortest_simple_paths(graph, origin, destination, weight="ai_cost")):
            if tuple(cand_path) not in seen_paths:
                seen_paths.add(tuple(cand_path))
                alternatives.append(
                    self.evaluate_path(graph, cand_path, fc_map, f"Alternative Route {len(alternatives) + 1}")
                )
            if len(alternatives) >= top_k_alternatives or idx > 15:
                break

        # Honest travel-time savings: strictly zero unless Route 2 is faster than Route 1
        t1 = shortest_summary.estimated_travel_time_min
        t2 = ai_summary.estimated_travel_time_min
        time_saved_min = round(max(0.0, t1 - t2), 2)
        time_saved_pct = round(((t1 - t2) / t1) * 100.0, 1) if (t1 > 0 and t2 < t1) else 0.0

        r1 = shortest_summary.average_risk_score
        r2 = ai_summary.average_risk_score
        congestion_avoidance_pct = round(((r1 - r2) / r1) * 100.0, 1) if (r1 > 0 and r2 < r1) else 0.0

        time_saved_display = f"{time_saved_pct:.1f}% ({time_saved_min:.2f} min)" if time_saved_pct > 0 else "0.0% (Same/Fastest)"
        comparison_table = [
            {
                "Metric": "Route path",
                "Shortest Route": " -> ".join(shortest_summary.path),
                "Future AI Route": " -> ".join(ai_summary.path),
            },
            {
                "Metric": "Distance",
                "Shortest Route": f"{shortest_summary.distance_km:.2f} km",
                "Future AI Route": f"{ai_summary.distance_km:.2f} km",
            },
            {
                "Metric": "Estimated travel time",
                "Shortest Route": f"{shortest_summary.estimated_travel_time_min:.2f} min",
                "Future AI Route": f"{ai_summary.estimated_travel_time_min:.2f} min",
            },
            {
                "Metric": "Predicted congestion",
                "Shortest Route": f"{shortest_summary.predicted_congestion} (Risk {shortest_summary.average_risk_score:.0f}/100)",
                "Future AI Route": f"{ai_summary.predicted_congestion} (Risk {ai_summary.average_risk_score:.0f}/100)",
            },
            {
                "Metric": "Average predicted speed",
                "Shortest Route": f"{shortest_summary.average_predicted_speed_kmh:.1f} km/h",
                "Future AI Route": f"{ai_summary.average_predicted_speed_kmh:.1f} km/h",
            },
            {
                "Metric": "Time saved",
                "Shortest Route": "—",
                "Future AI Route": time_saved_display,
            },
        ]

        graph_nodes = [
            {
                "junction": n,
                "latitude": float(graph.nodes[n]["latitude"]),
                "longitude": float(graph.nodes[n]["longitude"]),
                "predicted_condition": fc_map[n].predicted_condition,
                "risk_score": fc_map[n].congestion_risk_score,
                "predicted_speed_kmh": fc_map[n].predicted_speed_kmh,
                "predicted_vehicle_count": fc_map[n].predicted_vehicle_count,
                "predicted_occupancy_pct": fc_map[n].predicted_occupancy_pct,
                "accident_scenario": fc_map[n].accident_scenario,
            }
            for n in sorted(graph.nodes)
        ]

        seen_undir: set[tuple[str, str]] = set()
        graph_edges: list[dict[str, Any]] = []
        for u, v, d in graph.edges(data=True):
            pair = tuple(sorted((u, v)))
            if pair in seen_undir:
                continue
            seen_undir.add(pair)
            graph_edges.append(
                {
                    "u": u,
                    "v": v,
                    "road_name": d["road_name"],
                    "distance_km": d["distance_km"],
                    "segment_speed_kmh": d["segment_speed_kmh"],
                    "travel_time_min": d["travel_time_min"],
                }
            )

        departure_ts = forecasts[0].target_timestamp if forecasts else ""
        return RouteComparisonResult(
            origin=origin,
            destination=destination,
            departure_horizon_hours=departure_horizon_hours,
            departure_timestamp=departure_ts,
            weather_scenario=weather_scenario,
            accident_junctions=sorted(list(accident_junctions or [])),
            shortest_route=shortest_summary,
            ai_recommended_route=ai_summary,
            alternative_routes=alternatives,
            comparison_table=comparison_table,
            time_saved_min=time_saved_min,
            time_saved_pct=time_saved_pct,
            congestion_avoidance_pct=congestion_avoidance_pct,
            junction_forecasts={k: v.to_dict() for k, v in fc_map.items()},
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )

