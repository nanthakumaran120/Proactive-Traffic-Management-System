"""Plotly visualization helpers for the Dark-Navy Smart Traffic Control Centre."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


CONDITION_COLORS = {
    "Low": "#22C55E",     # Green
    "Medium": "#F59E0B",  # Orange / Amber
    "High": "#EF4444",    # Red
}

DARK_BG = "#0B1120"
PANEL_BG = "#111827"
GRID_COLOR = "#1F2937"
TEXT_COLOR = "#E5E7EB"


def apply_control_center_theme(
    fig: go.Figure,
    title: str = "",
    x_title: str | int = "",
    y_title: str = "",
    height: int = 420,
    theme_mode: str = "dark",
    *args: Any,
    **kwargs: Any,
) -> go.Figure:
    """Apply consistent dark-navy or clean-light styling to Plotly charts."""
    if isinstance(x_title, int):
        height = x_title
        x_title = ""
    if "theme_mode" in kwargs:
        theme_mode = kwargs["theme_mode"]

    if theme_mode == "light":
        paper = "#FFFFFF"
        plot = "#F8FAFC"
        grid = "#E2E8F0"
        text = "#0F172A"
        leg_bg = "rgba(255, 255, 255, 0.92)"
        leg_border = "#CBD5E1"
    else:
        paper = DARK_BG
        plot = PANEL_BG
        grid = GRID_COLOR
        text = TEXT_COLOR
        leg_bg = "rgba(17, 24, 39, 0.85)"
        leg_border = "#334155"

    x_dict = dict(gridcolor=grid, zerolinecolor=grid)
    if x_title:
        x_dict["title"] = dict(text=str(x_title))
    y_dict = dict(gridcolor=grid, zerolinecolor=grid)
    if y_title:
        y_dict["title"] = dict(text=str(y_title))

    fig.update_layout(
        title=dict(text=title, font=dict(color=text, size=15)),
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        font=dict(color=text, family="Inter, Segoe UI, sans-serif"),
        height=height,
        margin=dict(l=40, r=25, t=48, b=40),
        legend=dict(
            bgcolor=leg_bg,
            bordercolor=leg_border,
            borderwidth=1,
            font=dict(size=11, color=text),
        ),
        xaxis=x_dict,
        yaxis=y_dict,
    )
    return fig


def _edge_trace(
    coords: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    name: str,
    color: str,
    width: float,
    dash: str = "solid",
) -> go.Scatter:
    lons: list[float | None] = []
    lats: list[float | None] = []
    for u, v in edges:
        if u in coords and v in coords:
            lat1, lon1 = coords[u]
            lat2, lon2 = coords[v]
            lons.extend([lon1, lon2, None])
            lats.extend([lat1, lat2, None])
    return go.Scatter(
        x=lons,
        y=lats,
        mode="lines",
        name=name,
        line=dict(color=color, width=width, dash=dash),
        hoverinfo="skip",
    )


def build_network_map_figure(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    shortest_path: list[str] | None = None,
    ai_path: list[str] | None = None,
    emergency_path: list[str] | None = None,
    use_osm_tiles: bool = False,
    title: str = "Gandhipuram Schematic Corridor Network",
) -> go.Figure:
    """Create an interactive geospatial corridor map or OSM overlay for the 6 junctions."""
    coords = {n["junction"]: (float(n["latitude"]), float(n["longitude"])) for n in nodes}

    if use_osm_tiles:
        fig = go.Figure()
        # Base schematic corridors
        base_lons, base_lats = [], []
        for e in edges:
            u, v = e["u"], e["v"]
            if u in coords and v in coords:
                base_lats.extend([coords[u][0], coords[v][0], None])
                base_lons.extend([coords[u][1], coords[v][1], None])
        fig.add_trace(
            go.Scattermap(
                lon=base_lons,
                lat=base_lats,
                mode="lines",
                name="Schematic Corridors",
                line=dict(width=2, color="#64748B"),
                hoverinfo="skip",
            )
        )

        def _add_path_map(path: list[str] | None, label: str, color: str, width: int) -> None:
            if not path or len(path) < 2:
                return
            fig.add_trace(
                go.Scattermap(
                    lon=[coords[p][1] for p in path],
                    lat=[coords[p][0] for p in path],
                    mode="lines",
                    name=label,
                    line=dict(width=width, color=color),
                )
            )

        _add_path_map(shortest_path, "Normal Shortest Route", "#94A3B8", 4)
        _add_path_map(ai_path, "Future AI Route", "#38BDF8", 6)
        _add_path_map(emergency_path, "Emergency Priority Route", "#F43F5E", 7)

        for cond, color in CONDITION_COLORS.items():
            sub = [n for n in nodes if n.get("predicted_condition", n.get("Traffic_Condition")) == cond]
            if not sub:
                continue
            fig.add_trace(
                go.Scattermap(
                    lon=[float(n["longitude"]) for n in sub],
                    lat=[float(n["latitude"]) for n in sub],
                    mode="markers+text",
                    name=f"{cond} Congestion",
                    marker=dict(size=16, color=color),
                    text=[n["junction"].replace(" Approach", "").replace(" Junction", "") for n in sub],
                    textposition="top right",
                    customdata=[
                        [
                            n["junction"],
                            n.get("predicted_condition", n.get("Traffic_Condition", "")),
                            n.get("predicted_speed_kmh", n.get("Traffic_Speed_kmh", 0.0)),
                            n.get("predicted_vehicle_count", n.get("Vehicle_Count", 0.0)),
                            n.get("risk_score", 0.0),
                        ]
                        for n in sub
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Condition: %{customdata[1]}<br>"
                        "Speed: %{customdata[2]:.1f} km/h<br>"
                        "Vehicles: %{customdata[3]:.0f}<br>"
                        "Risk Score: %{customdata[4]:.1f}/100<extra></extra>"
                    ),
                )
            )

        center_lat = sum(c[0] for c in coords.values()) / len(coords)
        center_lon = sum(c[1] for c in coords.values()) / len(coords)
        fig.update_layout(
            title=dict(text=title, font=dict(color="#F8FAFC", size=15)),
            map=dict(
                style="carto-darkmatter",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=14.3,
            ),
            paper_bgcolor=DARK_BG,
            font=dict(color=TEXT_COLOR),
            height=480,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        return fig

    # Reliable geospatial coordinate canvas (works online & offline)
    fig = go.Figure()
    all_pairs = [(e["u"], e["v"]) for e in edges]
    fig.add_trace(_edge_trace(coords, all_pairs, "Schematic Corridors", "#334155", 2.0, "dot"))

    # Edge distance labels at segment midpoints
    mid_lons, mid_lats, mid_txt = [], [], []
    for e in edges:
        u, v = e["u"], e["v"]
        if u in coords and v in coords:
            mid_lats.append((coords[u][0] + coords[v][0]) / 2.0)
            mid_lons.append((coords[u][1] + coords[v][1]) / 2.0)
            mid_txt.append(f"{e['distance_km']:.2f} km")
    fig.add_trace(
        go.Scatter(
            x=mid_lons,
            y=mid_lats,
            mode="text",
            text=mid_txt,
            textfont=dict(size=9, color="#94A3B8"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    if shortest_path and len(shortest_path) >= 2:
        pairs_s = list(zip(shortest_path[:-1], shortest_path[1:]))
        fig.add_trace(
            _edge_trace(coords, pairs_s, "Normal Shortest Route", "#CBD5E1", 4.0, "dash")
        )

    if ai_path and len(ai_path) >= 2:
        pairs_ai = list(zip(ai_path[:-1], ai_path[1:]))
        fig.add_trace(
            _edge_trace(coords, pairs_ai, "Future AI Route", "#38BDF8", 6.0, "solid")
        )

    if emergency_path and len(emergency_path) >= 2:
        pairs_emg = list(zip(emergency_path[:-1], emergency_path[1:]))
        fig.add_trace(
            _edge_trace(coords, pairs_emg, "Emergency Priority Route", "#F43F5E", 7.0, "solid")
        )

    for cond, color in CONDITION_COLORS.items():
        sub = [n for n in nodes if n.get("predicted_condition", n.get("Traffic_Condition")) == cond]
        if not sub:
            continue
        fig.add_trace(
            go.Scatter(
                x=[float(n["longitude"]) for n in sub],
                y=[float(n["latitude"]) for n in sub],
                mode="markers+text",
                name=f"{cond} Congestion",
                marker=dict(
                    size=18,
                    color=color,
                    line=dict(width=2.5, color="#F8FAFC"),
                ),
                text=[
                    f"<b>{n['junction']}</b>" + (" [ACCIDENT]" if n.get("accident_scenario") == "Yes" else "")
                    for n in sub
                ],
                textposition="top center",
                textfont=dict(size=10, color="#F8FAFC"),
                customdata=[
                    [
                        n["junction"],
                        n.get("predicted_condition", n.get("Traffic_Condition", "")),
                        n.get("predicted_speed_kmh", n.get("Traffic_Speed_kmh", 0.0)),
                        n.get("predicted_vehicle_count", n.get("Vehicle_Count", 0.0)),
                        n.get("risk_score", 0.0),
                    ]
                    for n in sub
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Condition: %{customdata[1]}<br>"
                    "Speed: %{customdata[2]:.1f} km/h<br>"
                    "Vehicles: %{customdata[3]:.0f}<br>"
                    "Risk Score: %{customdata[4]:.1f}/100<extra></extra>"
                ),
            )
        )

    apply_control_center_theme(fig, title=title, height=490)
    fig.update_xaxes(title="Longitude", tickformat=".4f", range=[76.9648, 76.9726])
    fig.update_yaxes(title="Latitude", tickformat=".4f", range=[11.0145, 11.0252])
    return fig


def build_forecast_line_chart(
    forecast_df: pd.DataFrame,
    y_col: str,
    y_title: str,
    title: str,
) -> go.Figure:
    """Multi-junction line chart across forecast horizons."""
    fig = go.Figure()
    for junc, grp in forecast_df.groupby("junction"):
        grp = grp.sort_values("horizon_hours")
        fig.add_trace(
            go.Scatter(
                x=grp["target_timestamp"].str[-5:] + " (+" + grp["horizon_hours"].astype(str) + "h)",
                y=grp[y_col],
                mode="lines+markers",
                name=junc,
                line=dict(width=2.5),
                marker=dict(size=8),
            )
        )
    apply_control_center_theme(fig, title=title, height=360)
    fig.update_xaxes(title="Forecast Horizon")
    fig.update_yaxes(title=y_title)
    return fig


def build_junction_comparison_bar(
    snap_df: pd.DataFrame,
    metric_col: str,
    cond_col: str,
    title: str,
    y_title: str,
) -> go.Figure:
    """Color-coded junction bar chart (Green=Low, Orange=Medium, Red=High)."""
    colors = [CONDITION_COLORS.get(str(c), "#38BDF8") for c in snap_df[cond_col]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=snap_df["junction"] if "junction" in snap_df.columns else snap_df["Location"],
                y=snap_df[metric_col],
                marker_color=colors,
                text=snap_df[metric_col].round(1),
                textposition="outside",
            )
        ]
    )
    apply_control_center_theme(fig, title=title, height=350)
    fig.update_xaxes(tickangle=-18)
    fig.update_yaxes(title=y_title)
    return fig


def build_confusion_matrix_figure(
    labels: list[str] | None = None,
    matrix: list[list[int]] | None = None,
) -> go.Figure:
    """Render confusion matrix heatmap."""
    actual_labels = labels or ["High", "Low", "Medium"]
    actual_matrix = matrix or [[75, 0, 1], [0, 7, 23], [2, 9, 115]]
    fig = go.Figure(
        data=go.Heatmap(
            z=actual_matrix,
            x=actual_labels,
            y=actual_labels,
            text=actual_matrix,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )
    apply_control_center_theme(fig, title="Chronological Evaluation Confusion Matrix", height=360)
    fig.update_xaxes(title="Predicted Traffic Condition")
    fig.update_yaxes(title="Actual Traffic Condition")
    return fig

