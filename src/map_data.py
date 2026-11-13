"""Map view-models, congestion color palette, road-segment filters, and zoom math."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import pandas as pd

from src.future_predictor import JunctionForecast
from src.road_network import OSMRoadNetwork


# Phase 2 & Phase 4 unified color palette
CONDITION_COLOR_MAP = {
    "Low": "#22C55E",      # Green
    "Medium": "#F59E0B",   # Orange
    "High": "#EF4444",     # Red
}

MAP_HIGHLIGHT_COLORS = {
    "selected": "#3B82F6",      # Blue
    "emergency": "#A855F7",     # Purple
    "shortest": "#3B82F6",      # Neutral Blue
    "ai_recommended": "#10B981",  # Bright Emerald / Cyan accent
    "alternative": "#64748B",   # Grey
    "high_road": "#EF4444",     # Red affected street segment
    "origin": "#2563EB",
    "destination": "#0F172A",
}

CONGESTION_LEGEND_ITEMS = [
    ("Low Traffic", "#22C55E"),
    ("Medium Traffic", "#F59E0B"),
    ("High Traffic", "#EF4444"),
    ("Selected / Origin", "#3B82F6"),
    ("AI Recommended Route", "#10B981"),
    ("Emergency Corridor", "#A855F7"),
]


@dataclass
class MapJunctionNode:
    """Normalized junction marker payload rendered on Admin and User street maps."""

    junction: str
    latitude: float
    longitude: float
    display_condition: str
    observed_condition: str
    predicted_condition: str
    vehicle_count: float
    speed_kmh: float
    occupancy_pct: float
    risk_score: float
    signal_state: str
    weather_condition: str
    accident_report: str
    observation_time: str
    target_time: str
    is_forecast: bool
    is_selected: bool = False
    is_origin: bool = False
    is_destination: bool = False
    is_alert_focused: bool = False
    snap_distance_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_map_nodes(
    snapshot_df: pd.DataFrame,
    forecasts_by_junction: dict[str, JunctionForecast] | None = None,
    use_forecast_values: bool = False,
    selected_junction: str | None = None,
    origin_junction: str | None = None,
    destination_junction: str | None = None,
    alert_focus_junction: str | None = None,
    road_network: OSMRoadNetwork | None = None,
) -> list[MapJunctionNode]:
    """Create unified marker view-models from CSV snapshot + optional ML future forecast."""
    nodes: list[MapJunctionNode] = []
    for _, row in snapshot_df.iterrows():
        jname = str(row["Location"])
        fc = forecasts_by_junction.get(jname) if forecasts_by_junction else None
        obs_cond = str(row["Traffic_Condition"])
        pred_cond = fc.predicted_condition if fc else obs_cond

        if use_forecast_values and fc is not None:
            disp_cond = fc.predicted_condition
            vcount = float(fc.predicted_vehicle_count)
            spd = float(fc.predicted_speed_kmh)
            occ = float(fc.predicted_occupancy_pct)
            risk = float(fc.congestion_risk_score)
            weather = str(fc.weather_scenario)
            acc = str(fc.accident_scenario)
            target_t = str(fc.target_timestamp)
            is_fc = True
        else:
            disp_cond = obs_cond
            vcount = float(row["Vehicle_Count"])
            spd = float(row["Traffic_Speed_kmh"])
            occ = float(row["Road_Occupancy_pct"])
            risk = float(fc.congestion_risk_score) if fc else {"Low": 20.0, "Medium": 52.0, "High": 86.0}.get(obs_cond, 50.0)
            weather = str(row["Weather_Condition"])
            acc = str(row["Accident_Report"])
            target_t = fc.target_timestamp if fc else str(row["Timestamp"])
            is_fc = False

        snap_m = 0.0
        if road_network is not None and jname in road_network.snapped_nodes:
            snap_m = float(road_network.snapped_nodes[jname].get("snap_distance_m", 0.0))

        nodes.append(
            MapJunctionNode(
                junction=jname,
                latitude=float(row["Latitude"]),
                longitude=float(row["Longitude"]),
                display_condition=disp_cond,
                observed_condition=obs_cond,
                predicted_condition=pred_cond,
                vehicle_count=round(vcount, 1),
                speed_kmh=round(spd, 1),
                occupancy_pct=round(occ, 1),
                risk_score=round(risk, 1),
                signal_state=str(row["Traffic_Light_State"]),
                weather_condition=weather,
                accident_report=acc,
                observation_time=str(row["Timestamp"]),
                target_time=target_t,
                is_forecast=is_fc,
                is_selected=(selected_junction == jname),
                is_origin=(origin_junction == jname),
                is_destination=(destination_junction == jname),
                is_alert_focused=(alert_focus_junction == jname),
                snap_distance_m=snap_m,
            )
        )
    return nodes


def extract_high_congestion_road_segments(
    road_network: OSMRoadNetwork | None,
    nodes: list[MapJunctionNode],
    max_junction_radius_m: float = 260.0,
) -> list[dict[str, Any]]:
    """Return real mapped OSM road polylines linked to High-congestion junctions only."""
    if road_network is None or not road_network.is_real_osm:
        return []
    high_junctions = {n.junction for n in nodes if n.display_condition == "High"}
    if not high_junctions:
        return []
    segments: list[dict[str, Any]] = []
    for way in road_network.way_polylines:
        nj = way.get("nearest_junction")
        dist_m = float(way.get("nearest_junction_dist_m", 9999.0))
        if nj in high_junctions and dist_m <= max_junction_radius_m:
            segments.append(
                {
                    "road_name": way.get("road_name", "Affected Corridor"),
                    "junction": nj,
                    "coords": way["coords"],
                }
            )
    return segments


def compute_center_and_zoom(
    coords: list[tuple[float, float]],
    default_zoom: float = 14.9,
    min_zoom: float = 13.8,
    max_zoom: float = 16.4,
) -> tuple[float, float, float]:
    """Compute balanced MapLibre (center_lat, center_lon, zoom) fitting ``coords`` cleanly."""
    if not coords:
        return 11.0196, 76.9682, default_zoom
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    c_lat = 0.5 * (min(lats) + max(lats))
    c_lon = 0.5 * (min(lons) + max(lons))
    if len(coords) == 1:
        return c_lat, c_lon, 16.2
    span = max(max(lats) - min(lats), max(lons) - min(lons), 1e-4)
    # Empirical MapLibre zoom fit for urban Coimbatore ~11deg N latitude
    zoom = 14.9 - math.log2(max(span / 0.0085, 0.25))
    zoom = max(min_zoom, min(max_zoom, round(zoom, 2)))
    return round(c_lat, 6), round(c_lon, 6), zoom

