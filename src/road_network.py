"""Real OpenStreetMap drivable road network loader, snapper, and Dijkstra router for Gandhipuram.

IMPORTANT DATA SEPARATION:
- OpenStreetMap is used strictly as geographic map and road-network reference data
  (street geometry, one-way connectivity, road names, and real road distances).
- Traffic counts, speeds, occupancy, congestion labels, and future predictions come
  exclusively from datasets/gandhipuram_coimbatore_traffic_synthetic.csv and trained ML models.
- If OpenStreetMap road data cannot be loaded from cache or the OSM API, the engine
  falls back cleanly to the modular 6-junction schematic corridor graph and labels it
  explicitly as a schematic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import networkx as nx
import requests

from src.future_predictor import JunctionForecast
from src.future_routing import GRAPH_LIMITATION_NOTE, SCHEMATIC_CORRIDORS, haversine_km
from src.validation import PROJECT_ROOT, get_junction_coordinates, load_and_validate_gandhipuram_data


DEFAULT_OSM_CACHE_PATH = PROJECT_ROOT / "data" / "gandhipuram_osm_road_network.json"
OSM_USER_AGENT = "GandhipuramSmartTrafficPlatform/2.0 (Educational-Research-UI)"

DRIVABLE_HIGHWAYS = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
        "living_street",
    }
)

SCHEMATIC_FALLBACK_NOTE = (
    "Schematic Route (OSM Unavailable): OpenStreetMap road geometry could not be loaded; "
    "displaying straight-line corridor estimates derived from the 6 CSV coordinates."
)


@dataclass
class MappedRoute:
    """Route evaluated over either the mapped OpenStreetMap road graph or schematic fallback."""

    name: str
    junction_path: list[str]
    polyline_coords: list[tuple[float, float]]
    distance_km: float
    estimated_travel_time_min: float
    predicted_congestion: str
    average_predicted_speed_kmh: float
    average_risk_score: float
    street_names: list[str] = field(default_factory=list)
    junction_conditions: dict[str, str] = field(default_factory=dict)
    accident_affected_nodes: list[str] = field(default_factory=list)
    edge_ids: list[tuple[str, str]] = field(default_factory=list)
    is_real_osm_route: bool = True
    distance_source_label: str = "OpenStreetMap Road Distance"
    osm_node_count: int = 0
    total_ai_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _edge_junction_association(
    mid_lat: float,
    mid_lon: float,
    junction_coords: dict[str, tuple[float, float]],
) -> tuple[str | None, float]:
    """Associate a road segment with the nearest CSV junction and distance in metres."""
    best_name: str | None = None
    best_dist_m = float("inf")
    for jname, (jlat, jlon) in junction_coords.items():
        d_m = haversine_km(mid_lat, mid_lon, jlat, jlon) * 1000.0
        if d_m < best_dist_m:
            best_dist_m = d_m
            best_name = jname
    return best_name, round(best_dist_m, 1)


def _overall_condition_from_nodes(
    junction_names: list[str],
    forecasts: dict[str, JunctionForecast],
    avg_risk: float,
) -> str:
    conds = [forecasts[j].predicted_condition for j in junction_names if j in forecasts]
    if "High" in conds or avg_risk >= 68.0:
        return "High"
    if "Medium" in conds or avg_risk >= 36.0:
        return "Medium"
    return "Low"


class OSMRoadNetwork:
    """Gandhipuram drivable OpenStreetMap road graph + clean schematic fallback."""

    def __init__(
        self,
        graph: nx.DiGraph,
        junction_coords: dict[str, tuple[float, float]],
        snapped_nodes: dict[str, dict[str, Any]],
        way_polylines: list[dict[str, Any]],
        is_real_osm: bool,
        source_label: str,
        bbox: tuple[float, float, float, float] | None = None,
        fetched_at: str | None = None,
        error_note: str | None = None,
    ) -> None:
        self.graph = graph
        self.junction_coords = dict(junction_coords)
        self.snapped_nodes = dict(snapped_nodes)
        self.way_polylines = list(way_polylines)
        self.is_real_osm = bool(is_real_osm)
        self.source_label = source_label
        self.bbox = bbox
        self.fetched_at = fetched_at
        self.error_note = error_note

    @classmethod
    def from_osm_xml(
        cls,
        xml_bytes: bytes,
        junction_coords: dict[str, tuple[float, float]],
        bbox: tuple[float, float, float, float],
        source_label: str = "OpenStreetMap Road Network (Live API)",
    ) -> "OSMRoadNetwork":
        """Parse OpenStreetMap XML ways/nodes into a strongly connected drivable graph."""
        root = ET.fromstring(xml_bytes)
        raw_nodes: dict[str, tuple[float, float]] = {}
        for el in root.findall("node"):
            raw_nodes[el.attrib["id"]] = (float(el.attrib["lat"]), float(el.attrib["lon"]))

        graph = nx.DiGraph()
        bridge_interior_nodes: set[str] = set()
        raw_ways: list[dict[str, Any]] = []

        for way in root.findall("way"):
            tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
            hw = tags.get("highway")
            if hw not in DRIVABLE_HIGHWAYS:
                continue
            if tags.get("access") in {"private", "no"}:
                continue
            nds = [nd.attrib["ref"] for nd in way.findall("nd") if nd.attrib["ref"] in raw_nodes]
            if len(nds) < 2:
                continue

            if tags.get("bridge") == "yes" and len(nds) > 2:
                bridge_interior_nodes.update(nds[1:-1])

            road_name = tags.get("name") or tags.get("ref") or hw.replace("_", " ").title()
            oneway_tag = tags.get("oneway", "no").lower()
            if tags.get("junction") == "roundabout" and "oneway" not in tags:
                oneway_tag = "yes"
            is_oneway = oneway_tag in {"yes", "true", "1", "-1"}

            coords_list = [raw_nodes[n] for n in nds]
            mid_pt = coords_list[len(coords_list) // 2]
            near_junc, near_dist_m = _edge_junction_association(mid_pt[0], mid_pt[1], junction_coords)
            raw_ways.append(
                {
                    "way_id": way.attrib.get("id", ""),
                    "road_name": road_name,
                    "highway": hw,
                    "oneway": is_oneway,
                    "node_ids": nds,
                    "nearest_junction": near_junc,
                    "nearest_junction_dist_m": near_dist_m,
                }
            )

            for u, v in zip(nds[:-1], nds[1:]):
                lat1, lon1 = raw_nodes[u]
                lat2, lon2 = raw_nodes[v]
                seg_dist = max(0.001, haversine_km(lat1, lon1, lat2, lon2))
                seg_near_j, seg_near_d = _edge_junction_association(
                    0.5 * (lat1 + lat2), 0.5 * (lon1 + lon2), junction_coords
                )
                graph.add_node(u, lat=lat1, lon=lon1)
                graph.add_node(v, lat=lat2, lon=lon2)
                edge_meta = {
                    "distance_km": round(seg_dist, 6),
                    "road_name": road_name,
                    "highway": hw,
                    "oneway": is_oneway,
                    "nearest_junction": seg_near_j,
                    "nearest_junction_dist_m": seg_near_d,
                }
                if oneway_tag == "-1":
                    graph.add_edge(v, u, **edge_meta)
                elif oneway_tag in {"yes", "true", "1"}:
                    graph.add_edge(u, v, **edge_meta)
                else:
                    graph.add_edge(u, v, **edge_meta)
                    graph.add_edge(v, u, **edge_meta)

        if graph.number_of_nodes() == 0:
            raise ValueError("No drivable OpenStreetMap roads found in Gandhipuram bounding box.")

        largest_scc = max(nx.strongly_connected_components(graph), key=len)
        scc_graph = graph.subgraph(largest_scc).copy()
        scc_set = set(scc_graph.nodes)

        snapped: dict[str, dict[str, Any]] = {}
        sorted_nodes = list(scc_graph.nodes)
        for jname, (jlat, jlon) in junction_coords.items():
            ranked = sorted(
                sorted_nodes,
                key=lambda nid: haversine_km(
                    jlat, jlon, float(scc_graph.nodes[nid]["lat"]), float(scc_graph.nodes[nid]["lon"])
                ),
            )
            chosen = ranked[0]
            # Prefer a street-level bidirectional node close to the junction rather than elevated bridges
            for cand in ranked[:20]:
                d_m = (
                    haversine_km(
                        jlat, jlon, float(scc_graph.nodes[cand]["lat"]), float(scc_graph.nodes[cand]["lon"])
                    )
                    * 1000.0
                )
                if (
                    d_m <= 125.0
                    and cand not in bridge_interior_nodes
                    and any(not scc_graph[cand][nbr]["oneway"] for nbr in scc_graph.successors(cand))
                    and any(not scc_graph[pred][cand]["oneway"] for pred in scc_graph.predecessors(cand))
                ):
                    chosen = cand
                    break

            slat = float(scc_graph.nodes[chosen]["lat"])
            slon = float(scc_graph.nodes[chosen]["lon"])
            snap_m = round(haversine_km(jlat, jlon, slat, slon) * 1000.0, 1)
            adj_roads = sorted(
                {
                    str(scc_graph[chosen][nbr]["road_name"])
                    for nbr in scc_graph.successors(chosen)
                    if scc_graph[chosen][nbr].get("road_name")
                }
            )
            snapped[jname] = {
                "node_id": chosen,
                "lat": slat,
                "lon": slon,
                "snap_distance_m": snap_m,
                "adjacent_roads": adj_roads,
            }

        way_polylines: list[dict[str, Any]] = []
        for w in raw_ways:
            kept_coords = [
                [round(float(scc_graph.nodes[nid]["lat"]), 6), round(float(scc_graph.nodes[nid]["lon"]), 6)]
                for nid in w["node_ids"]
                if nid in scc_set
            ]
            if len(kept_coords) >= 2:
                way_polylines.append(
                    {
                        "road_name": w["road_name"],
                        "highway": w["highway"],
                        "oneway": w["oneway"],
                        "coords": kept_coords,
                        "nearest_junction": w["nearest_junction"],
                        "nearest_junction_dist_m": w["nearest_junction_dist_m"],
                    }
                )

        fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return cls(
            graph=scc_graph,
            junction_coords=junction_coords,
            snapped_nodes=snapped,
            way_polylines=way_polylines,
            is_real_osm=True,
            source_label=source_label,
            bbox=bbox,
            fetched_at=fetched_at,
        )

    @classmethod
    def build_schematic_fallback(
        cls,
        junction_coords: dict[str, tuple[float, float]],
        reason: str = SCHEMATIC_FALLBACK_NOTE,
    ) -> "OSMRoadNetwork":
        """Build the explicit schematic 6-junction fallback graph when OSM is unavailable."""
        graph = nx.DiGraph()
        snapped: dict[str, dict[str, Any]] = {}
        way_polylines: list[dict[str, Any]] = []

        for jname, (lat, lon) in junction_coords.items():
            graph.add_node(jname, lat=lat, lon=lon)
            snapped[jname] = {
                "node_id": jname,
                "lat": lat,
                "lon": lon,
                "snap_distance_m": 0.0,
                "adjacent_roads": ["Schematic Corridor Link"],
            }

        for u, v, road_name in SCHEMATIC_CORRIDORS:
            if u not in junction_coords or v not in junction_coords:
                continue
            lat1, lon1 = junction_coords[u]
            lat2, lon2 = junction_coords[v]
            dist_km = round(haversine_km(lat1, lon1, lat2, lon2), 5)
            for src_j, dst_j in ((u, v), (v, u)):
                graph.add_edge(
                    src_j,
                    dst_j,
                    distance_km=dist_km,
                    road_name=f"{road_name} (Schematic)",
                    highway="schematic",
                    oneway=False,
                    nearest_junction=dst_j,
                    nearest_junction_dist_m=0.0,
                )
            way_polylines.append(
                {
                    "road_name": road_name,
                    "highway": "schematic",
                    "oneway": False,
                    "coords": [[lat1, lon1], [lat2, lon2]],
                    "nearest_junction": u,
                    "nearest_junction_dist_m": 0.0,
                }
            )

        return cls(
            graph=graph,
            junction_coords=junction_coords,
            snapped_nodes=snapped,
            way_polylines=way_polylines,
            is_real_osm=False,
            source_label="Schematic Corridor Fallback (OSM Unavailable)",
            error_note=reason,
        )

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize compactly for fast instant reload without repeated OSM API hits."""
        nodes_payload = {
            str(nid): [round(float(d["lat"]), 7), round(float(d["lon"]), 7)]
            for nid, d in self.graph.nodes(data=True)
        }
        edges_payload = [
            [
                str(u),
                str(v),
                round(float(d["distance_km"]), 6),
                str(d["road_name"]),
                str(d["highway"]),
                bool(d["oneway"]),
                d.get("nearest_junction"),
                float(d.get("nearest_junction_dist_m", 9999.0)),
            ]
            for u, v, d in self.graph.edges(data=True)
        ]
        return {
            "version": 1,
            "fetched_at": self.fetched_at,
            "bbox": list(self.bbox) if self.bbox else None,
            "junction_coords": {k: [v[0], v[1]] for k, v in self.junction_coords.items()},
            "snapped_nodes": self.snapped_nodes,
            "nodes": nodes_payload,
            "edges": edges_payload,
            "way_polylines": self.way_polylines,
        }

    @classmethod
    def from_cache_dict(cls, payload: dict[str, Any]) -> "OSMRoadNetwork":
        """Reconstruct OSMRoadNetwork from cached JSON."""
        graph = nx.DiGraph()
        for nid, coords in payload["nodes"].items():
            graph.add_node(nid, lat=float(coords[0]), lon=float(coords[1]))
        for e in payload["edges"]:
            u, v, dist_km, road_name, highway, oneway, near_j, near_d = e
            graph.add_edge(
                u,
                v,
                distance_km=float(dist_km),
                road_name=str(road_name),
                highway=str(highway),
                oneway=bool(oneway),
                nearest_junction=near_j,
                nearest_junction_dist_m=float(near_d),
            )
        bbox_list = payload.get("bbox")
        bbox = tuple(float(x) for x in bbox_list) if bbox_list and len(bbox_list) == 4 else None
        jcoords = {k: (float(v[0]), float(v[1])) for k, v in payload["junction_coords"].items()}
        return cls(
            graph=graph,
            junction_coords=jcoords,
            snapped_nodes=payload["snapped_nodes"],
            way_polylines=payload.get("way_polylines", []),
            is_real_osm=True,
            source_label="OpenStreetMap Road Network (Cached)",
            bbox=bbox,  # type: ignore[arg-type]
            fetched_at=payload.get("fetched_at"),
        )

    def _edge_weight_for_mode(
        self,
        u: str,
        v: str,
        data: dict[str, Any],
        forecasts: dict[str, JunctionForecast],
        mode: str,
        weather_scenario: str,
        vehicle_type: str,
        penalized_edges: set[tuple[str, str]] | None = None,
        avoid_junctions: set[str] | None = None,
    ) -> tuple[float, float, float]:
        """Return (optimization_cost, est_segment_time_min, eff_speed_kmh) strictly from CSV/ML forecasts."""
        dist_km = float(data["distance_km"])
        near_j = data.get("nearest_junction")
        near_d = float(data.get("nearest_junction_dist_m", 9999.0))
        weather_pen = {"Clear": 0.0, "Cloudy": 0.05, "Light Rain": 0.16}.get(weather_scenario, 0.0)

        # Edges inside a Gandhipuram junction's 450m influence zone take that junction's forecasted traffic;
        # connecting segments outside use the network mean forecasted speed.
        if near_j and near_j in forecasts and near_d <= 450.0:
            fc = forecasts[near_j]
            proximity = max(0.25, 1.0 - (near_d / 550.0))
            base_speed = max(6.0, fc.predicted_speed_kmh)
            cond = fc.predicted_condition
            occ = fc.predicted_occupancy_pct
            risk = fc.congestion_risk_score
            is_acc = fc.accident_scenario == "Yes"
        else:
            avg_spd = sum(f.predicted_speed_kmh for f in forecasts.values()) / max(1, len(forecasts))
            proximity = 0.25
            base_speed = max(12.0, avg_spd)
            cond = "Low"
            occ = 40.0
            risk = 25.0
            is_acc = False

        hw = str(data.get("highway", "unclassified"))
        # Subtle road-class preference for arterial roads vs narrow side lanes
        road_class_factor = {
            "trunk": 0.96,
            "primary": 0.98,
            "secondary": 1.00,
            "tertiary": 1.03,
            "unclassified": 1.06,
            "residential": 1.10,
            "living_street": 1.15,
        }.get(hw, 1.04)

        # Unified physical travel-time estimate for standard vehicles across Shortest & AI modes
        standard_speed = max(5.0, base_speed * (0.75 if is_acc else 1.0))
        standard_delay = (0.55 if is_acc else {"Low": 0.02, "Medium": 0.08, "High": 0.24}.get(cond, 0.08)) * proximity
        standard_time_min = (dist_km / standard_speed) * 60.0 + standard_delay

        if mode == "emergency":
            boost = {"Ambulance": 1.20, "Fire Engine": 1.14, "Police Vehicle": 1.24}.get(vehicle_type, 1.18)
            eff_speed = max(6.0, base_speed * 0.72) if is_acc else min(42.0, base_speed * boost)
            delay_min = (0.35 if is_acc else 0.015) * proximity
            cong_pen = {"Low": 0.0, "Medium": 0.55, "High": 2.6}.get(cond, 0.55) * proximity
            acc_pen = (9.5 if is_acc else 0.0) * proximity
            time_min = (dist_km / eff_speed) * 60.0 + delay_min
            cost = time_min * road_class_factor * (1.0 + cong_pen + acc_pen + weather_pen)
        elif mode == "shortest":
            eff_speed = standard_speed
            time_min = standard_time_min
            cost = dist_km * (1.0 + (0.01 if road_class_factor > 1.05 else 0.0))
        else:  # mode == "ai"
            eff_speed = standard_speed
            time_min = standard_time_min
            cond_pen = {"Low": 0.0, "Medium": 0.35, "High": 1.65}.get(cond, 0.35) * proximity
            occ_pen = max(0.0, (occ - 35.0) / 95.0) * proximity
            risk_pen = (risk / 75.0) * proximity
            acc_pen = (5.0 if is_acc else 0.0) * proximity
            cost = time_min * road_class_factor * (1.0 + cond_pen + occ_pen + risk_pen + acc_pen + weather_pen)

        if avoid_junctions and near_j in avoid_junctions and near_d <= 220.0:
            cost *= 8.0
        if penalized_edges and (u, v) in penalized_edges:
            cost *= 4.0
        return cost, time_min, eff_speed

    def _trace_node_path_details(
        self,
        origin_junction: str,
        dest_junction: str,
        node_path: list[str],
        forecasts: dict[str, JunctionForecast],
        name: str,
        mode: str,
        weather_scenario: str,
        vehicle_type: str,
    ) -> MappedRoute:
        total_dist = 0.0
        total_time = 0.0
        total_cost = 0.0
        weighted_speed = 0.0
        street_names: list[str] = []
        edge_ids: list[tuple[str, str]] = []

        # Track CSV junctions in traversal order near the mapped OSM path
        junc_order: list[str] = [origin_junction]
        polyline: list[tuple[float, float]] = [self.junction_coords[origin_junction]]

        for nid in node_path:
            nlat = float(self.graph.nodes[nid]["lat"])
            nlon = float(self.graph.nodes[nid]["lon"])
            if haversine_km(polyline[-1][0], polyline[-1][1], nlat, nlon) * 1000.0 > 0.5:
                polyline.append((nlat, nlon))
            for jname, (jlat, jlon) in self.junction_coords.items():
                if jname not in (origin_junction, dest_junction) and jname not in junc_order:
                    if haversine_km(nlat, nlon, jlat, jlon) * 1000.0 <= 230.0:
                        junc_order.append(jname)

        end_lat, end_lon = self.junction_coords[dest_junction]
        if haversine_km(polyline[-1][0], polyline[-1][1], end_lat, end_lon) * 1000.0 > 0.5:
            polyline.append((end_lat, end_lon))
        if dest_junction not in junc_order:
            junc_order.append(dest_junction)

        for u, v in zip(node_path[:-1], node_path[1:]):
            edata = self.graph[u][v]
            d_km = float(edata["distance_km"])
            c, t_min, spd = self._edge_weight_for_mode(
                u, v, edata, forecasts, mode=mode, weather_scenario=weather_scenario, vehicle_type=vehicle_type
            )
            total_dist += d_km
            total_time += t_min
            total_cost += c
            weighted_speed += d_km * spd
            rname = str(edata.get("road_name", "Street"))
            if not street_names or street_names[-1] != rname:
                street_names.append(rname)
            edge_ids.append((str(u), str(v)))

        avg_spd = (weighted_speed / total_dist) if total_dist > 0 else forecasts[origin_junction].predicted_speed_kmh
        node_risks = [forecasts[j].congestion_risk_score for j in junc_order if j in forecasts]
        avg_risk = float(sum(node_risks) / len(node_risks)) if node_risks else 0.0
        junc_conds = {j: forecasts[j].predicted_condition for j in junc_order if j in forecasts}
        acc_nodes = [j for j in junc_order if j in forecasts and forecasts[j].accident_scenario == "Yes"]
        overall_cond = _overall_condition_from_nodes(junc_order, forecasts, avg_risk)

        return MappedRoute(
            name=name,
            junction_path=junc_order,
            polyline_coords=polyline,
            distance_km=round(total_dist, 3),
            estimated_travel_time_min=round(total_time, 2),
            predicted_congestion=overall_cond,
            average_predicted_speed_kmh=round(avg_spd, 2),
            average_risk_score=round(avg_risk, 1),
            street_names=street_names,
            junction_conditions=junc_conds,
            accident_affected_nodes=acc_nodes,
            edge_ids=edge_ids,
            is_real_osm_route=self.is_real_osm,
            distance_source_label=(
                "OpenStreetMap Mapped Road Distance"
                if self.is_real_osm
                else "Schematic Route (OSM Unavailable — Straight-Line)"
            ),
            osm_node_count=len(node_path),
            total_ai_cost=round(total_cost, 3),
        )

    def find_route(
        self,
        origin_junction: str,
        destination_junction: str,
        forecasts_by_junction: dict[str, JunctionForecast],
        mode: str = "ai",
        name: str = "Route",
        weather_scenario: str = "Clear",
        vehicle_type: str = "Ambulance",
        penalized_edges: set[tuple[str, str]] | None = None,
        avoid_junctions: set[str] | None = None,
    ) -> MappedRoute:
        """Compute Dijkstra path over the road graph using forecasted traffic costs."""
        src_node = str(self.snapped_nodes[origin_junction]["node_id"])
        dst_node = str(self.snapped_nodes[destination_junction]["node_id"])

        def _weight(u: Any, v: Any, d: dict[str, Any]) -> float:
            cost, _, _ = self._edge_weight_for_mode(
                str(u),
                str(v),
                d,
                forecasts_by_junction,
                mode=mode,
                weather_scenario=weather_scenario,
                vehicle_type=vehicle_type,
                penalized_edges=penalized_edges,
                avoid_junctions=avoid_junctions,
            )
            return cost

        node_path = [str(n) for n in nx.dijkstra_path(self.graph, src_node, dst_node, weight=_weight)]
        return self._trace_node_path_details(
            origin_junction=origin_junction,
            dest_junction=destination_junction,
            node_path=node_path,
            forecasts=forecasts_by_junction,
            name=name,
            mode=mode,
            weather_scenario=weather_scenario,
            vehicle_type=vehicle_type,
        )

    def find_alternative_routes(
        self,
        origin_junction: str,
        destination_junction: str,
        forecasts_by_junction: dict[str, JunctionForecast],
        exclude_routes: list[MappedRoute],
        weather_scenario: str = "Clear",
        max_alternatives: int = 2,
    ) -> list[MappedRoute]:
        """Generate genuinely distinct mapped road alternatives via iterative edge penalization."""
        alternatives: list[MappedRoute] = []
        seen_signatures: set[tuple[str, ...]] = {
            tuple(r.street_names) + (str(round(r.distance_km, 2)),) for r in exclude_routes
        }
        penalized: set[tuple[str, str]] = set()
        for r in exclude_routes:
            # Penalize middle portion so alternative keeps valid snapped start/end approaches
            edges = r.edge_ids
            if len(edges) > 4:
                penalized.update(edges[2:-2])
            else:
                penalized.update(edges)

        for idx in range(max_alternatives * 3):
            if len(alternatives) >= max_alternatives:
                break
            cand = self.find_route(
                origin_junction=origin_junction,
                destination_junction=destination_junction,
                forecasts_by_junction=forecasts_by_junction,
                mode="ai",
                name=f"Alternative Route {len(alternatives) + 1}",
                weather_scenario=weather_scenario,
                penalized_edges=penalized,
            )
            sig = tuple(cand.street_names) + (str(round(cand.distance_km, 2)),)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                alternatives.append(cand)
            edges = cand.edge_ids
            if len(edges) > 4:
                penalized.update(edges[2:-2])
            else:
                penalized.update(edges)
            if idx == 0 and not penalized:
                break
        return alternatives


def load_or_fetch_gandhipuram_osm_network(
    force_refresh: bool = False,
    force_offline: bool = False,
    cache_path: Path | str | None = None,
    timeout_sec: int = 20,
) -> OSMRoadNetwork:
    """Load Gandhipuram OSM road network from cache or OpenStreetMap API with schematic fallback."""
    df, _ = load_and_validate_gandhipuram_data()
    coords_df = get_junction_coordinates(df)
    junction_coords = {
        str(r["Location"]): (float(r["Latitude"]), float(r["Longitude"]))
        for _, r in coords_df.iterrows()
    }

    if force_offline:
        return OSMRoadNetwork.build_schematic_fallback(
            junction_coords,
            reason="Offline Road Network Mode enabled: using schematic corridor fallback.",
        )

    c_path = Path(cache_path) if cache_path is not None else DEFAULT_OSM_CACHE_PATH
    if not force_refresh and c_path.exists():
        try:
            payload = json.loads(c_path.read_text(encoding="utf-8"))
            return OSMRoadNetwork.from_cache_dict(payload)
        except Exception:
            pass

    lats = [c[0] for c in junction_coords.values()]
    lons = [c[1] for c in junction_coords.values()]
    bbox = (
        round(min(lons) - 0.0045, 4),
        round(min(lats) - 0.0045, 4),
        round(max(lons) + 0.0045, 4),
        round(max(lats) + 0.0045, 4),
    )
    api_url = f"https://api.openstreetmap.org/api/0.6/map?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    try:
        resp = requests.get(api_url, timeout=timeout_sec, headers={"User-Agent": OSM_USER_AGENT})
        resp.raise_for_status()
        net = OSMRoadNetwork.from_osm_xml(resp.content, junction_coords=junction_coords, bbox=bbox)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        c_path.write_text(json.dumps(net.to_cache_dict()), encoding="utf-8")
        return net
    except Exception as exc:
        if c_path.exists():
            try:
                net = OSMRoadNetwork.from_cache_dict(json.loads(c_path.read_text(encoding="utf-8")))
                net.error_note = f"Live OSM refresh failed ({type(exc).__name__}); using cached OSM road network."
                return net
            except Exception:
                pass
        return OSMRoadNetwork.build_schematic_fallback(
            junction_coords,
            reason=f"{GRAPH_LIMITATION_NOTE} (OSM API unavailable: {type(exc).__name__})",
        )


def check_map_tile_connectivity(timeout_sec: float = 3.0) -> dict[str, Any]:
    """Probe OpenStreetMap & Carto raster tile reachability gracefully."""
    probes = {
        "osm": "https://tile.openstreetmap.org/15/23394/15482.png",
        "carto_dark": "https://a.basemaps.cartocdn.com/dark_all/15/23394/15482.png",
        "carto_light": "https://a.basemaps.cartocdn.com/light_all/15/23394/15482.png",
    }
    results: dict[str, Any] = {}
    headers = {"User-Agent": OSM_USER_AGENT}
    for key, url in probes.items():
        try:
            r = requests.get(url, timeout=timeout_sec, headers=headers)
            results[key] = (r.status_code == 200)
        except Exception:
            results[key] = False
    results["any_tiles_available"] = bool(results["osm"] or results["carto_dark"] or results["carto_light"])
    return results
