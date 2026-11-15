"""Unified route orchestration combining CSV/ML predictions with OSM or schematic road graphs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from src.emergency_routing import (
    EMERGENCY_LIMITATION_NOTE,
    EmergencyPriorityRouter,
)
from src.future_predictor import JunctionForecast
from src.future_routing import (
    GRAPH_LIMITATION_NOTE,
    FutureRouteRecommender,
    RouteSummary,
)
from src.road_network import (
    MappedRoute,
    OSMRoadNetwork,
    SCHEMATIC_FALLBACK_NOTE,
)
from src.signal_controller import SIMULATION_DISCLAIMER, SignalPlan, simulate_emergency_signal_priority


@dataclass
class UnifiedRouteBundle:
    """Combined route comparison with mapped geometry, true distances, and honest labels."""

    origin: str
    destination: str
    departure_horizon_hours: int
    departure_timestamp: str
    weather_scenario: str
    accident_junctions: list[str]
    shortest_route: MappedRoute
    ai_recommended_route: MappedRoute
    alternative_routes: list[MappedRoute]
    all_route_options: list[MappedRoute]
    recommended_reason: str
    comparison_table: list[dict[str, Any]]
    time_saved_min: float
    time_saved_pct: float
    congestion_avoidance_pct: float
    uses_real_osm_roads: bool
    network_source_label: str
    limitation_note: str
    junction_forecasts: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedEmergencyBundle:
    """Emergency corridor simulation integrated with mapped OSM streets or schematic fallback."""

    emergency_mode_active: bool
    vehicle_type: str
    origin: str
    destination: str
    departure_horizon_hours: int
    departure_timestamp: str
    weather_scenario: str
    accident_junctions: list[str]
    priority_status: str
    normal_route: MappedRoute
    emergency_route: MappedRoute
    estimated_emergency_time_min: float
    normal_travel_time_min: float
    time_saved_min: float
    time_saved_pct: float
    congestion_avoidance_pct: float
    junctions_to_avoid: list[str]
    signal_priority_plans: list[SignalPlan]
    uses_real_osm_roads: bool
    network_source_label: str
    limitation_note: str
    simulation_disclaimer: str
    junction_forecasts: dict[str, dict[str, Any]]


def _schematic_summary_to_mapped(
    summary: RouteSummary,
    node_coords: dict[str, tuple[float, float]],
) -> MappedRoute:
    coords = [node_coords[j] for j in summary.path if j in node_coords]
    return MappedRoute(
        name=summary.name,
        junction_path=list(summary.path),
        polyline_coords=coords,
        distance_km=summary.distance_km,
        estimated_travel_time_min=summary.estimated_travel_time_min,
        predicted_congestion=summary.predicted_congestion,
        average_predicted_speed_kmh=summary.average_predicted_speed_kmh,
        average_risk_score=summary.average_risk_score,
        street_names=["Schematic Corridor Link (OSM Unavailable)"],
        junction_conditions=dict(summary.junction_conditions),
        accident_affected_nodes=list(summary.accident_affected_nodes),
        edge_ids=[],
        is_real_osm_route=False,
        distance_source_label="Schematic Straight-Line Corridor Distance (OSM Unavailable)",
        osm_node_count=len(coords),
        total_ai_cost=summary.total_ai_cost,
    )


def _build_route_recommendation_explanation(
    ai_route: MappedRoute,
    shortest_route: MappedRoute,
    time_saved_min: float,
    time_saved_pct: float,
    forecasts: dict[str, JunctionForecast],
    uses_real_osm: bool,
) -> str:
    road_tag = "mapped OpenStreetMap road geometry" if uses_real_osm else "schematic fallback corridor estimates"
    avoided_acc = [j for j in shortest_route.accident_affected_nodes if j not in ai_route.accident_affected_nodes]
    avoided_high = [
        j
        for j, c in shortest_route.junction_conditions.items()
        if c == "High" and j not in ai_route.junction_conditions
    ]

    if avoided_acc:
        return (
            f"Recommended using {road_tag}: avoids simulated incident corridor(s) "
            f"({', '.join(avoided_acc)}), saving {time_saved_min:.1f} min ({time_saved_pct:.1f}%) "
            f"at {ai_route.average_predicted_speed_kmh:.1f} km/h average predicted speed."
        )
    if avoided_high or ai_route.average_risk_score < shortest_route.average_risk_score:
        extra = (
            f" Saves an estimated {time_saved_min:.1f} min ({time_saved_pct:.1f}%)."
            if time_saved_min > 0
            else ""
        )
        return (
            f"Recommended using {road_tag}: lowers average route congestion risk from "
            f"{shortest_route.average_risk_score:.0f}/100 to {ai_route.average_risk_score:.0f}/100.{extra}"
        )
    if time_saved_min > 0:
        return (
            f"Recommended using {road_tag}: predicted to arrive {time_saved_min:.1f} min "
            f"({time_saved_pct:.1f}%) faster under projected speeds ({ai_route.average_predicted_speed_kmh:.1f} km/h)."
        )
    origin_fc = forecasts.get(ai_route.junction_path[0])
    period_note = (
        " at your selected departure time"
        if origin_fc is None
        else f" around {origin_fc.target_timestamp}"
    )
    return (
        f"Recommended using {road_tag}: the direct path offers the lowest expected delay "
        f"({ai_route.estimated_travel_time_min:.1f} min, {ai_route.predicted_congestion} congestion, "
        f"Risk {ai_route.average_risk_score:.0f}/100){period_note}."
    )


def compute_unified_route_comparison(
    router: FutureRouteRecommender,
    road_network: OSMRoadNetwork | None,
    origin: str,
    destination: str,
    departure_horizon_hours: int = 1,
    base_timestamp: pd.Timestamp | str | None = None,
    weather_scenario: str = "Clear",
    accident_junctions: list[str] | set[str] | None = None,
    top_k_alternatives: int = 2,
    departure_datetime: pd.Timestamp | str | None = None,
    forecasts_by_junction: dict[str, JunctionForecast] | None = None,
) -> UnifiedRouteBundle:
    """Compute Shortest, AI Recommended, and Alternative routes over real OSM roads or schematic fallback."""
    if origin == destination:
        raise ValueError("Origin and destination junctions must be different.")

    if forecasts_by_junction is None:
        forecasts = router.predictor.predict_horizon(
            horizon_hours=departure_horizon_hours,
            base_timestamp=base_timestamp,
            weather_override=weather_scenario,
            accident_junctions=accident_junctions,
        )
        fc_map = {fc.junction: fc for fc in forecasts}
    else:
        fc_map = forecasts_by_junction
        forecasts = list(fc_map.values())
    departure_ts = (
        pd.Timestamp(departure_datetime).strftime("%d-%m-%Y %H:%M")
        if departure_datetime is not None
        else (forecasts[0].target_timestamp if forecasts else "")
    )

    if road_network is not None and road_network.is_real_osm:
        shortest = road_network.find_route(
            origin_junction=origin,
            destination_junction=destination,
            forecasts_by_junction=fc_map,
            mode="shortest",
            name="Shortest Road Route",
            weather_scenario=weather_scenario,
        )
        ai_route = road_network.find_route(
            origin_junction=origin,
            destination_junction=destination,
            forecasts_by_junction=fc_map,
            mode="ai",
            name="Future AI Recommended Route",
            weather_scenario=weather_scenario,
        )
        alts = road_network.find_alternative_routes(
            origin_junction=origin,
            destination_junction=destination,
            forecasts_by_junction=fc_map,
            exclude_routes=[shortest, ai_route],
            weather_scenario=weather_scenario,
            max_alternatives=top_k_alternatives,
        )
        uses_real = True
        source_lbl = road_network.source_label
        lim_note = (
            "Road geometry and street distances use cached/live OpenStreetMap ways; "
            "all speeds, congestion risks, and incidents come strictly from the Gandhipuram CSV and trained ML models."
        )
    else:
        schem = router.recommend_route(
            origin=origin,
            destination=destination,
            departure_horizon_hours=departure_horizon_hours,
            base_timestamp=base_timestamp,
            weather_scenario=weather_scenario,
            accident_junctions=accident_junctions,
            top_k_alternatives=top_k_alternatives,
            forecasts_by_junction=fc_map,
        )
        shortest = _schematic_summary_to_mapped(schem.shortest_route, router.node_coords)
        ai_route = _schematic_summary_to_mapped(schem.ai_recommended_route, router.node_coords)
        alts = [_schematic_summary_to_mapped(a, router.node_coords) for a in schem.alternative_routes]
        uses_real = False
        source_lbl = road_network.source_label if road_network else "Schematic Corridor Fallback (OSM Unavailable)"
        lim_note = road_network.error_note if (road_network and road_network.error_note) else SCHEMATIC_FALLBACK_NOTE or GRAPH_LIMITATION_NOTE

    t1 = shortest.estimated_travel_time_min
    t2 = ai_route.estimated_travel_time_min
    time_saved_min = round(max(0.0, t1 - t2), 2)
    time_saved_pct = round(((t1 - t2) / t1) * 100.0, 1) if (t1 > 0 and t2 < t1) else 0.0

    r1 = shortest.average_risk_score
    r2 = ai_route.average_risk_score
    cong_avoid = round(((r1 - r2) / r1) * 100.0, 1) if (r1 > 0 and r2 < r1) else 0.0

    dist_metric_name = "Mapped Road Distance (OSM)" if uses_real else "Schematic Corridor Distance"
    time_saved_disp = f"{time_saved_pct:.1f}% ({time_saved_min:.2f} min)" if time_saved_pct > 0 else "0.0% (Optimal / Same ETA)"
    comparison_table = [
        {
            "Metric": "Junctions Along Route",
            "Shortest Route": " -> ".join(shortest.junction_path),
            "Future AI Recommended": " -> ".join(ai_route.junction_path),
        },
        {
            "Metric": "Streets / Corridors",
            "Shortest Route": " -> ".join(shortest.street_names[:5]) + (" ..." if len(shortest.street_names) > 5 else ""),
            "Future AI Recommended": " -> ".join(ai_route.street_names[:5]) + (" ..." if len(ai_route.street_names) > 5 else ""),
        },
        {
            "Metric": dist_metric_name,
            "Shortest Route": f"{shortest.distance_km:.2f} km",
            "Future AI Recommended": f"{ai_route.distance_km:.2f} km",
        },
        {
            "Metric": "Estimated Travel Time",
            "Shortest Route": f"{shortest.estimated_travel_time_min:.2f} min",
            "Future AI Recommended": f"{ai_route.estimated_travel_time_min:.2f} min",
        },
        {
            "Metric": "Predicted Congestion",
            "Shortest Route": f"{shortest.predicted_congestion} (Risk {shortest.average_risk_score:.0f}/100)",
            "Future AI Recommended": f"{ai_route.predicted_congestion} (Risk {ai_route.average_risk_score:.0f}/100)",
        },
        {
            "Metric": "Average Predicted Speed",
            "Shortest Route": f"{shortest.average_predicted_speed_kmh:.1f} km/h",
            "Future AI Recommended": f"{ai_route.average_predicted_speed_kmh:.1f} km/h",
        },
        {
            "Metric": "Time Saved",
            "Shortest Route": "—",
            "Future AI Recommended": time_saved_disp,
        },
    ]

    all_options = [ai_route]
    if shortest.street_names != ai_route.street_names or abs(shortest.distance_km - ai_route.distance_km) > 0.01:
        all_options.append(shortest)
    all_options.extend(alts)

    reason = _build_route_recommendation_explanation(
        ai_route=ai_route,
        shortest_route=shortest,
        time_saved_min=time_saved_min,
        time_saved_pct=time_saved_pct,
        forecasts=fc_map,
        uses_real_osm=uses_real,
    )

    return UnifiedRouteBundle(
        origin=origin,
        destination=destination,
        departure_horizon_hours=departure_horizon_hours,
        departure_timestamp=departure_ts,
        weather_scenario=weather_scenario,
        accident_junctions=sorted(list(accident_junctions or [])),
        shortest_route=shortest,
        ai_recommended_route=ai_route,
        alternative_routes=alts,
        all_route_options=all_options,
        recommended_reason=reason,
        comparison_table=comparison_table,
        time_saved_min=time_saved_min,
        time_saved_pct=time_saved_pct,
        congestion_avoidance_pct=cong_avoid,
        uses_real_osm_roads=uses_real,
        network_source_label=source_lbl,
        limitation_note=lim_note,
        junction_forecasts={k: v.to_dict() for k, v in fc_map.items()},
    )


def compute_unified_emergency_route(
    emergency_router: EmergencyPriorityRouter,
    road_network: OSMRoadNetwork | None,
    origin: str,
    destination: str,
    vehicle_type: str = "Ambulance",
    emergency_mode_active: bool = True,
    departure_horizon_hours: int = 2,
    base_timestamp: pd.Timestamp | str | None = None,
    weather_scenario: str = "Clear",
    accident_junctions: list[str] | set[str] | None = None,
) -> UnifiedEmergencyBundle:
    """Compute operator emergency corridor over mapped OSM roads or schematic fallback."""
    if origin == destination:
        raise ValueError("Origin and destination junctions must be different.")

    forecasts = emergency_router.predictor.predict_horizon(
        horizon_hours=departure_horizon_hours,
        base_timestamp=base_timestamp,
        weather_override=weather_scenario,
        accident_junctions=accident_junctions,
    )
    fc_map = {fc.junction: fc for fc in forecasts}
    departure_ts = forecasts[0].target_timestamp if forecasts else ""

    if road_network is not None and road_network.is_real_osm:
        normal_route = road_network.find_route(
            origin_junction=origin,
            destination_junction=destination,
            forecasts_by_junction=fc_map,
            mode="shortest",
            name="Normal Baseline Road Route",
            weather_scenario=weather_scenario,
        )
        if emergency_mode_active:
            avoid_set = {
                j
                for j, fc in fc_map.items()
                if j not in (origin, destination) and (fc.accident_scenario == "Yes" or fc.predicted_condition == "High")
            }
            emg_route = road_network.find_route(
                origin_junction=origin,
                destination_junction=destination,
                forecasts_by_junction=fc_map,
                mode="emergency",
                name=f"Emergency Priority Route ({vehicle_type})",
                weather_scenario=weather_scenario,
                vehicle_type=vehicle_type,
                avoid_junctions=avoid_set or None,
            )
            signal_plans = simulate_emergency_signal_priority(
                emg_route.junction_path,
                vehicle_type=vehicle_type,
                junction_states={k: v.to_dict() for k, v in fc_map.items()},
            )
            status_lbl = f"ACTIVE — {vehicle_type.upper()} GREEN CORRIDOR SIMULATION"
        else:
            emg_route = normal_route
            signal_plans = []
            status_lbl = "STANDBY — Emergency Mode OFF (Normal Routing Active)"

        t_norm = normal_route.estimated_travel_time_min
        t_emg = emg_route.estimated_travel_time_min
        time_saved_min = round(max(0.0, t_norm - t_emg), 2)
        time_saved_pct = round(((t_norm - t_emg) / t_norm) * 100.0, 1) if (t_norm > 0 and t_emg < t_norm) else 0.0
        r_norm = normal_route.average_risk_score
        r_emg = emg_route.average_risk_score
        cong_avoid = round(((r_norm - r_emg) / r_norm) * 100.0, 1) if (r_norm > 0 and r_emg < r_norm) else 0.0
        avoided = sorted(
            j
            for j, fc in fc_map.items()
            if j not in emg_route.junction_path and (fc.predicted_condition == "High" or fc.accident_scenario == "Yes")
        )
        return UnifiedEmergencyBundle(
            emergency_mode_active=emergency_mode_active,
            vehicle_type=vehicle_type,
            origin=origin,
            destination=destination,
            departure_horizon_hours=departure_horizon_hours,
            departure_timestamp=departure_ts,
            weather_scenario=weather_scenario,
            accident_junctions=sorted(list(accident_junctions or [])),
            priority_status=status_lbl,
            normal_route=normal_route,
            emergency_route=emg_route,
            estimated_emergency_time_min=t_emg,
            normal_travel_time_min=t_norm,
            time_saved_min=time_saved_min,
            time_saved_pct=time_saved_pct,
            congestion_avoidance_pct=cong_avoid,
            junctions_to_avoid=avoided,
            signal_priority_plans=signal_plans,
            uses_real_osm_roads=True,
            network_source_label=road_network.source_label,
            limitation_note=EMERGENCY_LIMITATION_NOTE,
            simulation_disclaimer=SIMULATION_DISCLAIMER,
            junction_forecasts={k: v.to_dict() for k, v in fc_map.items()},
        )

    # Clean schematic fallback
    coords = emergency_router.route_recommender.node_coords
    eres = emergency_router.compute_emergency_route(
        origin=origin,
        destination=destination,
        vehicle_type=vehicle_type,
        emergency_mode_active=emergency_mode_active,
        departure_horizon_hours=departure_horizon_hours,
        base_timestamp=base_timestamp,
        weather_scenario=weather_scenario,
        accident_junctions=accident_junctions,
    )
    return UnifiedEmergencyBundle(
        emergency_mode_active=eres.emergency_mode_active,
        vehicle_type=eres.vehicle_type,
        origin=eres.origin,
        destination=eres.destination,
        departure_horizon_hours=eres.departure_horizon_hours,
        departure_timestamp=eres.departure_timestamp,
        weather_scenario=eres.weather_scenario,
        accident_junctions=eres.accident_junctions,
        priority_status=eres.priority_status,
        normal_route=_schematic_summary_to_mapped(eres.normal_route, coords),
        emergency_route=_schematic_summary_to_mapped(eres.emergency_route, coords),
        estimated_emergency_time_min=eres.estimated_emergency_time_min,
        normal_travel_time_min=eres.normal_travel_time_min,
        time_saved_min=eres.time_saved_min,
        time_saved_pct=eres.time_saved_pct,
        congestion_avoidance_pct=eres.congestion_avoidance_pct,
        junctions_to_avoid=eres.junctions_to_avoid,
        signal_priority_plans=eres.signal_priority_plans,
        uses_real_osm_roads=False,
        network_source_label=road_network.source_label if road_network else "Schematic Corridor Fallback (OSM Unavailable)",
        limitation_note=f"{SCHEMATIC_FALLBACK_NOTE} {EMERGENCY_LIMITATION_NOTE}",
        simulation_disclaimer=SIMULATION_DISCLAIMER,
        junction_forecasts=eres.junction_forecasts,
    )

