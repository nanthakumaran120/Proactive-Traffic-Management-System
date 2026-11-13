"""Reusable interactive OpenStreetMap / MapLibre component shared across Admin and User UIs."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.map_data import (
    CONDITION_COLOR_MAP,
    MAP_HIGHLIGHT_COLORS,
    MapJunctionNode,
    compute_center_and_zoom,
    extract_high_congestion_road_segments,
)
from src.road_network import MappedRoute, OSMRoadNetwork


BASEMAP_OPTIONS_DARK = {
    "Dark Street Map (Carto)": "carto-darkmatter",
    "OpenStreetMap Streets": "open-street-map",
    "Clean Light Streets (Carto)": "carto-positron",
    "Offline Vector Canvas": "white-bg",
}

BASEMAP_OPTIONS_LIGHT = {
    "OpenStreetMap Streets": "open-street-map",
    "Clean Light Streets (Carto)": "carto-positron",
    "Dark Street Map (Carto)": "carto-darkmatter",
    "Offline Vector Canvas": "white-bg",
}


def _polylines_to_lat_lon(polylines: list[list[tuple[float, float]] | list[list[float]]]) -> tuple[list[float | None], list[float | None]]:
    lats: list[float | None] = []
    lons: list[float | None] = []
    for line in polylines:
        if len(line) < 2:
            continue
        for pt in line:
            lats.append(float(pt[0]))
            lons.append(float(pt[1]))
        lats.append(None)
        lons.append(None)
    return lats, lons


def _add_route_trace(
    fig: go.Figure,
    route: MappedRoute | None,
    label: str,
    color: str,
    width: float,
    opacity: float = 0.92,
    halo_color: str | None = None,
) -> None:
    if route is None or len(route.polyline_coords) < 2:
        return
    lats = [float(p[0]) for p in route.polyline_coords]
    lons = [float(p[1]) for p in route.polyline_coords]
    mode_tag = "OSM Mapped Road" if route.is_real_osm_route else "Schematic Fallback"
    streets_txt = " -> ".join(route.street_names[:5]) if route.street_names else "Direct Link"

    if halo_color:
        fig.add_trace(
            go.Scattermap(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=width + 4.0, color=halo_color),
                opacity=0.45,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scattermap(
            lat=lats,
            lon=lons,
            mode="lines",
            name=label,
            line=dict(width=width, color=color),
            opacity=opacity,
            hovertemplate=(
                f"<b>{label}</b> ({mode_tag})<br>"
                f"Path: {' -> '.join(route.junction_path)}<br>"
                f"Streets: {streets_txt}<br>"
                f"Distance: {route.distance_km:.2f} km | ETA: {route.estimated_travel_time_min:.1f} min<br>"
                f"Congestion: {route.predicted_congestion} (Risk {route.average_risk_score:.0f}/100)<extra></extra>"
            ),
        )
    )


def build_real_traffic_map_figure(
    nodes: list[MapJunctionNode],
    road_network: OSMRoadNetwork | None = None,
    shortest_route: MappedRoute | None = None,
    ai_route: MappedRoute | None = None,
    alternative_routes: list[MappedRoute] | None = None,
    emergency_route: MappedRoute | None = None,
    selected_alt_route: MappedRoute | None = None,
    map_style: str = "open-street-map",
    theme_mode: str = "light",
    title: str = "Gandhipuram Real-Street Traffic Map",
    height: int = 560,
    focus_junction: str | None = None,
    fit_to_routes: bool = True,
    show_road_background: bool = True,
) -> go.Figure:
    """Construct interactive Plotly MapLibre figure with OSM street geometry & markers."""
    fig = go.Figure()
    is_dark = theme_mode == "dark" or map_style == "carto-darkmatter"
    offline_canvas = map_style == "white-bg"
    paper_bg = "#0B1120" if theme_mode == "dark" else "#FFFFFF"
    text_color = "#F8FAFC" if (is_dark and not (theme_mode == "light" and not offline_canvas)) else "#0F172A"
    if theme_mode == "dark" and map_style == "open-street-map":
        text_color = "#0F172A"
    elif theme_mode == "dark" and map_style in {"carto-darkmatter", "white-bg"}:
        text_color = "#F8FAFC"

    # 1. Background Gandhipuram road ways (or explicit schematic fallback links when offline)
    if show_road_background and road_network is not None and road_network.way_polylines:
        bg_lines = [w["coords"] for w in road_network.way_polylines]
        bg_lats, bg_lons = _polylines_to_lat_lon(bg_lines)
        if road_network.is_real_osm:
            bg_color = "rgba(148, 163, 184, 0.35)" if is_dark else "rgba(71, 85, 105, 0.25)"
            bg_width = 1.6 if not offline_canvas else 2.0
            bg_name = "Gandhipuram Streets (OSM)"
        else:
            bg_color = "rgba(148, 163, 184, 0.55)"
            bg_width = 2.0
            bg_name = "Schematic Corridors (Fallback)"

        fig.add_trace(
            go.Scattermap(
                lat=bg_lats,
                lon=bg_lons,
                mode="lines",
                name=bg_name,
                line=dict(width=bg_width, color=bg_color),
                hoverinfo="skip",
            )
        )

    # 2. High-congestion affected road segments (strictly from mapped OSM ways near High junctions)
    high_segs = extract_high_congestion_road_segments(road_network, nodes)
    if high_segs:
        h_lats, h_lons = _polylines_to_lat_lon([s["coords"] for s in high_segs])
        fig.add_trace(
            go.Scattermap(
                lat=h_lats,
                lon=h_lons,
                mode="lines",
                name="High-Congestion Road Segment",
                line=dict(width=3.8, color=MAP_HIGHLIGHT_COLORS["high_road"]),
                opacity=0.65,
                hoverinfo="skip",
            )
        )

    # 3. Route overlays (grey alternatives -> neutral blue shortest -> emerald AI -> purple emergency)
    if alternative_routes:
        for idx, alt in enumerate(alternative_routes):
            is_chosen_alt = selected_alt_route is not None and alt.street_names == selected_alt_route.street_names and abs(alt.distance_km - selected_alt_route.distance_km) < 0.01
            if is_chosen_alt:
                continue
            _add_route_trace(
                fig,
                alt,
                label=alt.name or f"Alternative Route {idx + 1}",
                color=MAP_HIGHLIGHT_COLORS["alternative"],
                width=4.5,
                opacity=0.72,
            )

    _add_route_trace(
        fig,
        shortest_route,
        label="Shortest Route (Baseline)",
        color=MAP_HIGHLIGHT_COLORS["shortest"],
        width=5.5,
        opacity=0.88,
    )

    if selected_alt_route is not None:
        _add_route_trace(
            fig,
            selected_alt_route,
            label=f"Selected: {selected_alt_route.name}",
            color="#F59E0B",
            width=7.5,
            opacity=0.96,
            halo_color="#FDE68A",
        )

    _add_route_trace(
        fig,
        ai_route,
        label="AI Recommended Route",
        color=MAP_HIGHLIGHT_COLORS["ai_recommended"],
        width=7.5,
        opacity=0.98,
        halo_color="#A7F3D0",
    )

    _add_route_trace(
        fig,
        emergency_route,
        label="Emergency Priority Route",
        color=MAP_HIGHLIGHT_COLORS["emergency"],
        width=8.5,
        opacity=0.98,
        halo_color="#E9D5FF",
    )

    # 4. Highlight rings (Selected / Alert-Focused Blue halo, Origin/Destination pins, Accident ring)
    halo_nodes = [n for n in nodes if n.is_selected or n.is_alert_focused]
    if halo_nodes:
        fig.add_trace(
            go.Scattermap(
                lat=[n.latitude for n in halo_nodes],
                lon=[n.longitude for n in halo_nodes],
                mode="markers",
                name="Selected Junction",
                marker=dict(size=30, color=MAP_HIGHLIGHT_COLORS["selected"], opacity=0.45),
                hoverinfo="skip",
            )
        )

    acc_nodes = [n for n in nodes if n.accident_report == "Yes"]
    if acc_nodes:
        fig.add_trace(
            go.Scattermap(
                lat=[n.latitude for n in acc_nodes],
                lon=[n.longitude for n in acc_nodes],
                mode="markers",
                name="Accident / Incident Scenario",
                marker=dict(size=28, color="#EF4444", opacity=0.55),
                hoverinfo="skip",
            )
        )

    orig_nodes = [n for n in nodes if n.is_origin]
    if orig_nodes:
        fig.add_trace(
            go.Scattermap(
                lat=[n.latitude for n in orig_nodes],
                lon=[n.longitude for n in orig_nodes],
                mode="markers",
                name="Origin (Start)",
                marker=dict(size=24, color=MAP_HIGHLIGHT_COLORS["origin"], opacity=0.88),
                hoverinfo="skip",
            )
        )

    dest_nodes = [n for n in nodes if n.is_destination]
    if dest_nodes:
        fig.add_trace(
            go.Scattermap(
                lat=[n.latitude for n in dest_nodes],
                lon=[n.longitude for n in dest_nodes],
                mode="markers",
                name="Destination (End)",
                marker=dict(size=24, color="#111827" if not is_dark else "#F8FAFC", opacity=0.88),
                hoverinfo="skip",
            )
        )

    # 5. Clickable color-coded junction markers (one trace per congestion level for crisp legend)
    for cond, color in CONDITION_COLOR_MAP.items():
        group = [n for n in nodes if n.display_condition == cond]
        if not group:
            continue
        labels = []
        for n in group:
            short_n = n.junction.replace(" Approach", "").replace(" Junction", "")
            prefix = ""
            if n.is_origin:
                prefix = "[START] "
            elif n.is_destination:
                prefix = "[END] "
            elif n.is_selected:
                prefix = "★ "
            suffix = " [ACCIDENT]" if n.accident_report == "Yes" else ""
            labels.append(f"{prefix}{short_n}{suffix}")

        sizes = [20 if (n.is_selected or n.is_origin or n.is_destination or n.is_alert_focused) else 16 for n in group]
        fig.add_trace(
            go.Scattermap(
                lat=[n.latitude for n in group],
                lon=[n.longitude for n in group],
                mode="markers+text",
                name=f"{cond} Congestion",
                marker=dict(size=sizes, color=color, opacity=0.96),
                text=labels,
                textposition="top right",
                textfont=dict(size=11, color=text_color),
                customdata=[
                    [
                        "JUNCTION",
                        n.junction,
                        "Model Forecast" if n.is_forecast else "Dataset Observed",
                        n.display_condition,
                        n.speed_kmh,
                        n.vehicle_count,
                        n.occupancy_pct,
                        n.risk_score,
                        n.signal_state,
                        n.weather_condition,
                        n.accident_report,
                        n.target_time if n.is_forecast else n.observation_time,
                    ]
                    for n in group
                ],
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Source: %{customdata[2]} (%{customdata[11]})<br>"
                    "Condition: <b>%{customdata[3]}</b> (Risk: %{customdata[7]:.0f}/100)<br>"
                    "Avg Speed: %{customdata[4]:.1f} km/h | Vehicles: %{customdata[5]:.0f}<br>"
                    "Occupancy: %{customdata[6]:.1f}% | Signal: %{customdata[8]}<br>"
                    "Weather: %{customdata[9]} | Accident: %{customdata[10]}<extra></extra>"
                ),
            )
        )

    # 6. Compute smart viewport center & zoom (focus alert/junction -> fit active route -> full network)
    if focus_junction:
        focus_match = [n for n in nodes if n.junction == focus_junction]
        if focus_match:
            c_lat, c_lon, zoom = compute_center_and_zoom(
                [(focus_match[0].latitude, focus_match[0].longitude)]
            )
        else:
            c_lat, c_lon, zoom = compute_center_and_zoom([(n.latitude, n.longitude) for n in nodes])
    elif fit_to_routes and any(r is not None for r in (ai_route, shortest_route, emergency_route, selected_alt_route)):
        fit_pts: list[tuple[float, float]] = []
        for r in (ai_route, shortest_route, emergency_route, selected_alt_route):
            if r is not None:
                fit_pts.extend(r.polyline_coords)
        c_lat, c_lon, zoom = compute_center_and_zoom(fit_pts)
    else:
        c_lat, c_lon, zoom = compute_center_and_zoom([(n.latitude, n.longitude) for n in nodes])

    legend_bg = "rgba(15, 23, 42, 0.85)" if theme_mode == "dark" else "rgba(255, 255, 255, 0.90)"
    legend_border = "#334155" if theme_mode == "dark" else "#CBD5E1"
    legend_text = "#E2E8F0" if theme_mode == "dark" else "#0F172A"
    title_color = "#F8FAFC" if theme_mode == "dark" else "#0F172A"

    fig.update_layout(
        title=dict(text=title, font=dict(color=title_color, size=15)),
        map=dict(
            style=map_style,
            center=dict(lat=c_lat, lon=c_lon),
            zoom=zoom,
        ),
        clickmode="event+select",
        dragmode="pan",
        paper_bgcolor=paper_bg,
        plot_bgcolor=paper_bg,
        font=dict(color=legend_text, family="Inter, Segoe UI, sans-serif"),
        height=height,
        margin=dict(l=6, r=6, t=42, b=6),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor=legend_bg,
            bordercolor=legend_border,
            borderwidth=1,
            font=dict(size=10, color=legend_text),
        ),
    )
    return fig


def render_traffic_map(
    nodes: list[MapJunctionNode],
    road_network: OSMRoadNetwork | None = None,
    shortest_route: MappedRoute | None = None,
    ai_route: MappedRoute | None = None,
    alternative_routes: list[MappedRoute] | None = None,
    emergency_route: MappedRoute | None = None,
    selected_alt_route: MappedRoute | None = None,
    theme_mode: str = "light",
    title: str = "Gandhipuram Street Map",
    key_prefix: str = "map",
    focus_junction: str | None = None,
    fit_to_routes: bool = True,
    default_height: int = 540,
    tiles_available: bool = True,
    simulate_tile_failure: bool = False,
) -> str | None:
    """Render the reusable interactive map with basemap switcher, expand toggle, legend & click capture."""
    options = BASEMAP_OPTIONS_DARK if theme_mode == "dark" else BASEMAP_OPTIONS_LIGHT
    opt_labels = list(options.keys())

    c_layer, c_exp, c_status = st.columns([1.4, 0.9, 1.7])
    with c_layer:
        chosen_lbl = st.selectbox(
            "Map Layer",
            options=opt_labels,
            index=0,
            key=f"{key_prefix}_basemap",
            label_visibility="collapsed",
        )
    with c_exp:
        expanded = st.toggle("Expanded Map", value=False, key=f"{key_prefix}_expand")

    requested_style = options[chosen_lbl]
    tile_fallback_active = simulate_tile_failure or (not tiles_available and requested_style != "white-bg")
    effective_style = "white-bg" if tile_fallback_active else requested_style

    with c_status:
        road_tag = "OSM Mapped Roads" if (road_network and road_network.is_real_osm) else "Schematic Fallback"
        tile_tag = "Offline Vector Canvas" if effective_style == "white-bg" else chosen_lbl
        st.caption(f"Roads: **{road_tag}** &nbsp;|&nbsp; Basemap: **{tile_tag}**")

    if tile_fallback_active:
        st.warning(
            "Map Tile Fallback Active: External map tiles are currently unreachable or offline fallback is enabled. "
            "Rendering cached road geometry and Gandhipuram junction coordinates without crashing."
        )
    if road_network is not None and not road_network.is_real_osm:
        st.info(
            "Schematic Corridor Fallback Active: Real OpenStreetMap road geometry could not be retrieved; "
            "routes represent straight-line schematic connections."
        )

    map_height = 740 if expanded else default_height
    fig = build_real_traffic_map_figure(
        nodes=nodes,
        road_network=road_network,
        shortest_route=shortest_route,
        ai_route=ai_route,
        alternative_routes=alternative_routes,
        emergency_route=emergency_route,
        selected_alt_route=selected_alt_route,
        map_style=effective_style,
        theme_mode=theme_mode,
        title=title,
        height=map_height,
        focus_junction=focus_junction,
        fit_to_routes=fit_to_routes,
    )

    event: Any = st.plotly_chart(
        fig,
        key=f"{key_prefix}_plotly",
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        config={"scrollZoom": True, "displayModeBar": True},
    )

    # Compact accessible legend strip below the map
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:0.80rem;opacity:0.9;margin:4px 0 8px 0;">'
        '<span><span style="color:#22C55E;">●</span> Low</span>'
        '<span><span style="color:#F59E0B;">●</span> Medium</span>'
        '<span><span style="color:#EF4444;">●</span> High</span>'
        '<span><span style="color:#3B82F6;">●</span> Selected / Shortest</span>'
        '<span><span style="color:#10B981;">━</span> AI Recommended</span>'
        '<span><span style="color:#64748B;">━</span> Alternative</span>'
        '<span><span style="color:#A855F7;">━</span> Emergency Corridor</span>'
        '<span><span style="color:#94A3B8;">ⓘ</span> Tip: Click any junction marker to inspect details; use scroll wheel / drag to zoom & pan.</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Extract clicked junction safely from Plotly selection event
    try:
        points = event.selection.get("points", []) if event and hasattr(event, "selection") else []
        for pt in points:
            cd = pt.get("customdata")
            if isinstance(cd, (list, tuple)) and len(cd) >= 2 and cd[0] == "JUNCTION":
                return str(cd[1])
    except Exception:
        pass
    return None

