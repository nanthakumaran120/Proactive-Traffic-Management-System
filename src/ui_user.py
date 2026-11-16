"""Consumer-friendly Light Navigation & Route Finder for Normal Users.

Features 7 Canonical Navigation Sections + Logout:
1. Home (Map-first interface, 6 junction traffic indicators, emergency advisory, route finder)
2. Traffic Map (OpenStreetMap with 6 CSV junctions & interactive detail inspector)
3. Find Route (2-column balanced layout: AI-recommended route, alternatives, OSM road map)
4. Future Traffic (1–6h human-readable forecast explanations & trend chart)
5. Traffic Alerts (Proactive pre-congestion warnings & click-to-focus map)
6. Emergency Information (Public traffic advisory for active emergency corridors & safe detours)
7. Account (User profile details & password changes)
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.auth import USER_PAGES, USER_PAGE_ALIASES, AuthManager, UserPublic
from src.charts import CONDITION_COLORS, apply_control_center_theme
from src.emergency_event_manager import EmergencyEventManager, EventReviewStatus
from src.future_predictor import FutureTrafficPredictor, JunctionForecast
from src.future_routing import FutureRouteRecommender
from src.map_component import render_traffic_map
from src.map_data import build_map_nodes
from src.road_network import MappedRoute, OSMRoadNetwork
from src.route_visualization import UnifiedRouteBundle, compute_unified_route_comparison
from src.ui_theme import render_kpi_card, render_limitation_footer, status_pill_html


def _friendly_risk_label(score: float) -> str:
    if score >= 70.0:
        return f"High Risk ({score:.0f}/100)"
    if score >= 40.0:
        return f"Moderate Risk ({score:.0f}/100)"
    return f"Low Risk ({score:.0f}/100)"


def _human_forecast_text(fc: JunctionForecast) -> str:
    if fc.predicted_condition == "High":
        return (
            f"Heavy traffic is expected around {fc.junction} at {fc.target_timestamp}, "
            f"with slower average speeds near {fc.predicted_speed_kmh:.0f} km/h. "
            "Leaving earlier or taking an alternative route is advised."
        )
    if fc.predicted_condition == "Medium":
        return (
            f"Moderate traffic is expected around {fc.junction} at {fc.target_timestamp}, "
            f"moving near {fc.predicted_speed_kmh:.0f} km/h. Allow a few extra minutes."
        )
    return (
        f"Smooth traffic is expected around {fc.junction} at {fc.target_timestamp}, "
        f"with speeds near {fc.predicted_speed_kmh:.0f} km/h."
    )


def _apply_user_map_click(clicked: str | None, state_key: str, last_click_key: str) -> None:
    if clicked and clicked != st.session_state.get(last_click_key):
        st.session_state[last_click_key] = clicked
        st.session_state[state_key] = clicked
        st.rerun()


def _navigate_user_page() -> None:
    """Keep the public navigation selector and URL in sync."""
    selected = st.session_state.get("usr_nav_select", "Home")
    st.session_state["nav_page"] = selected
    st.query_params["page"] = selected
    st.session_state["_last_qp_page"] = selected


def _get_emergency_manager() -> EmergencyEventManager:
    if "emergency_event_manager" not in st.session_state:
        st.session_state["emergency_event_manager"] = EmergencyEventManager()
    return st.session_state["emergency_event_manager"]


def _render_user_junction_spotlight(jname: str, snap_df: pd.DataFrame) -> None:
    match = snap_df[snap_df["Location"] == jname]
    if match.empty:
        return
    row = match.iloc[0]
    cond = str(row["Traffic_Condition"])
    st.markdown(
        f"""
        <div class="detail-panel">
            <div style="font-size:0.72rem;color:#176B68;font-weight:700;letter-spacing:0.05em;">
                JUNCTION INSPECTOR
            </div>
            <div style="font-size:1.10rem;font-weight:700;color:#202A28;margin:3px 0 6px 0;">
                {jname}
            </div>
            <div style="margin-bottom:8px;">{status_pill_html(cond, friendly=True)}</div>
            <div style="font-size:0.86rem;color:#334155;line-height:1.65;">
                <b>Observed Vehicles:</b> {int(row['Vehicle_Count'])}<br>
                <b>Average Speed:</b> {float(row['Traffic_Speed_kmh']):.1f} km/h<br>
                <b>Road Occupancy:</b> {float(row['Road_Occupancy_pct']):.1f}%<br>
                <b>Signal State:</b> {row['Traffic_Light_State']}<br>
                <b>Last Update:</b> {row['Timestamp']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_user_forecast_spotlight(jname: str, forecast: JunctionForecast) -> None:
    """Render selected-departure metrics for a junction on the route page."""
    st.markdown(
        f"""
        <div class="detail-panel">
            <div style="font-size:0.72rem;color:#176B68;font-weight:700;letter-spacing:0.05em;">
                JUNCTION INSPECTOR · DEPARTURE PROFILE
            </div>
            <div style="font-size:1.10rem;font-weight:700;color:#202A28;margin:3px 0 6px 0;">
                {jname}
            </div>
            <div style="margin-bottom:8px;">{status_pill_html(forecast.predicted_condition, friendly=True)}</div>
            <div style="font-size:0.86rem;color:#334155;line-height:1.65;">
                <b>Departure:</b> {forecast.target_timestamp}<br>
                <b>Expected Vehicles:</b> {forecast.predicted_vehicle_count:.1f}<br>
                <b>Expected Speed:</b> {forecast.predicted_speed_kmh:.1f} km/h<br>
                <b>Road Occupancy:</b> {forecast.predicted_occupancy_pct:.1f}%<br>
                <b>Congestion Risk:</b> {_friendly_risk_label(forecast.congestion_risk_score)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_compact_route_card(
    rt: MappedRoute,
    is_rec: bool,
    dist_note: str,
    btn_key: str,
    selected_alt_key: str,
    baseline_time: float | None = None,
) -> None:
    badge_html = (
        '<span class="pill-rec">AI RECOMMENDED</span>'
        if is_rec
        else '<span style="font-size:0.72rem;color:#64748B;font-weight:700;">ALTERNATIVE OPTION</span>'
    )
    card_cls = "route-card route-card-rec" if is_rec else "route-card"
    route_title = "AI Recommended Route" if is_rec else rt.name
    streets = " ➔ ".join(rt.street_names[:3]) + (" ..." if len(rt.street_names) > 3 else "")
    diff_html = ""
    if not is_rec and baseline_time is not None:
        delta_m = rt.estimated_travel_time_min - baseline_time
        if delta_m > 0:
            diff_html = f"<div style='font-size:0.78rem;color:#DC2626;font-weight:600;'>+{delta_m:.1f} min vs recommended</div>"
        elif delta_m < 0:
            diff_html = f"<div style='font-size:0.78rem;color:#059669;font-weight:600;'>{delta_m:.1f} min faster</div>"

    st.markdown(
        f"""
        <div class="{card_cls}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <div>{badge_html}</div>
                <div><span style="font-size:0.68rem;color:#64748B;font-weight:700;margin-right:5px;">ROUTE</span>{status_pill_html(rt.predicted_congestion, friendly=False)}</div>
            </div>
            <div style="font-size:1.02rem;font-weight:700;color:#202A28;margin-bottom:4px;">{route_title}</div>
            <div style="font-size:0.86rem;color:#334155;line-height:1.6;">
                <b>Travel Time:</b> {rt.estimated_travel_time_min:.1f} min &nbsp;|&nbsp; <b>Distance:</b> {rt.distance_km:.2f} km<br>
                <b>Congestion Risk:</b> {_friendly_risk_label(rt.average_risk_score)}<br>
                <b>Route:</b> {' ➔ '.join(rt.junction_path)}<br>
                <span style="font-size:0.78rem;color:#64748B;">Via: {streets}</span>
            </div>
            {diff_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not is_rec:
        if st.button("View on Map", key=btn_key, width="stretch"):
            st.session_state[selected_alt_key] = rt
            st.rerun()
    else:
        if st.session_state.get(selected_alt_key) is not None:
            if st.button("Highlight Recommended Only", key=f"{btn_key}_clear", width="stretch"):
                st.session_state[selected_alt_key] = None
                st.rerun()


def render_user_dashboard(
    auth: AuthManager,
    auth_token: str,
    user: UserPublic,
    predictor: FutureTrafficPredictor,
    router: FutureRouteRecommender,
    road_network: OSMRoadNetwork,
    tiles_available: bool,
    simulate_tile_failure: bool,
    requested_page: str,
) -> None:
    """Render the Light Consumer Smart Navigation & Route Finder for Normal Users."""
    verified_user = auth.require_authenticated(auth_token)
    junctions = predictor.junctions
    snap_df = predictor.get_latest_common_snapshot()
    emg_mgr = _get_emergency_manager()

    canon_page = USER_PAGE_ALIASES.get(requested_page, requested_page)
    current_page = canon_page if canon_page in USER_PAGES else "Home"
    if st.session_state.get("usr_nav_select") != current_page:
        st.session_state["usr_nav_select"] = current_page

    # Compact public navigation, sized for desktop and narrow screens.
    with st.container(border=True):
        h_brand, h_nav, h_user = st.columns([1.2, 2.1, 1.0])
        with h_brand:
            st.markdown(
                "<div class='user-brand-title'>Gandhipuram Traffic</div>"
                "<div class='user-brand-location'>Coimbatore</div>",
                unsafe_allow_html=True,
            )
        with h_nav:
            st.selectbox(
                "Go to page",
                options=list(USER_PAGES),
                index=USER_PAGES.index(current_page),
                key="usr_nav_select",
                label_visibility="collapsed",
                on_change=_navigate_user_page,
            )

        with h_user:
            u_c1, u_c2 = st.columns([1.2, 0.9])
            with u_c1:
                st.markdown(
                    f"<div class='user-account-name' style='font-size:0.78rem;padding-top:6px;' title='{verified_user.username}'>"
                    f"<b>{verified_user.username}</b></div>",
                    unsafe_allow_html=True,
                )
            with u_c2:
                if st.button("Logout", key="btn_user_logout", type="secondary", width="stretch"):
                    auth.logout(auth_token)
                    for k in (
                        "auth_token",
                        "auth_role",
                        "nav_page",
                        "_last_qp_page",
                        "admin_sidebar_nav",
                        "user_sidebar_nav",
                        "usr_nav_select",
                    ):
                        st.session_state.pop(k, None)
                    st.query_params.clear()
                    st.rerun()
    # Active Emergency Public Advisory Banner
    public_adv = emg_mgr.get_public_traffic_advisory()
    if public_adv:
        st.warning(
            f"🚨 **PUBLIC TRAFFIC ADVISORY:** {public_adv}\n\n"
            f"Please yield right of way. Check the **Emergency Information** tab for detour recommendations.",
            icon="⚠️",
        )

    # =========================================================================
    # 1. HOME (Map-First Navigation Experience)
    # =========================================================================
    if current_page == "Home":
        col_left, col_map = st.columns([1.0, 1.85])
        with col_left:
            with st.container(border=True):
                st.markdown("#### Plan Your Trip")
                st.caption("Select origin, destination, and departure time across Gandhipuram's 6 monitored junctions.")
                h_orig = st.selectbox("Starting Location", options=junctions, index=3, key="usr_home_orig")
                h_dest = st.selectbox(
                    "Destination",
                    options=[j for j in junctions if j != h_orig],
                    index=0,
                    key="usr_home_dest",
                )
                h_dep = st.selectbox(
                    "Departure Time",
                    options=list(range(0, 7)),
                    format_func=lambda h: "Leave Now (Current Conditions)" if h == 0 else f"In {h} hour(s) (Forecasted)",
                    index=0,
                    key="usr_home_dep",
                )

                if st.button("FIND BEST ROUTE", type="primary", width="stretch", key="usr_home_find_route"):
                    st.session_state["usr_home_route_res"] = compute_unified_route_comparison(
                        router=router,
                        road_network=road_network,
                        origin=h_orig,
                        destination=h_dest,
                        departure_horizon_hours=h_dep,
                    )
                    st.session_state["usr_home_alt"] = None

            sel_home_j = st.selectbox(
                "Inspect Junction (or click map marker)",
                options=junctions,
                index=junctions.index(st.session_state["usr_home_sel_j"])
                if st.session_state.get("usr_home_sel_j") in junctions
                else 0,
                key="usr_home_junc_picker",
            )
            if sel_home_j != st.session_state.get("_usr_home_picker_prev"):
                st.session_state["_usr_home_picker_prev"] = sel_home_j
                st.session_state["usr_home_sel_j"] = sel_home_j
            active_home_j = st.session_state.get("usr_home_sel_j", sel_home_j)
            _render_user_junction_spotlight(active_home_j, snap_df)

        home_route_res: UnifiedRouteBundle | None = st.session_state.get("usr_home_route_res")
        if home_route_res is not None:
            fc_for_home = {
                f.junction: f
                for f in predictor.predict_horizon(horizon_hours=home_route_res.departure_horizon_hours)
            }
            home_nodes = build_map_nodes(
                snap_df,
                forecasts_by_junction=fc_for_home,
                use_forecast_values=(home_route_res.departure_horizon_hours > 0),
                selected_junction=active_home_j,
                origin_junction=home_route_res.origin,
                destination_junction=home_route_res.destination,
                road_network=road_network,
            )
            map_title = f"Gandhipuram Route Map ({home_route_res.departure_timestamp})"
        else:
            home_nodes = build_map_nodes(
                snap_df,
                selected_junction=active_home_j,
                road_network=road_network,
            )
            map_title = f"Gandhipuram Real-Street Traffic Map (Current Snapshot)"

        with col_map:
            clicked_home = render_traffic_map(
                nodes=home_nodes,
                road_network=road_network,
                shortest_route=home_route_res.shortest_route if home_route_res else None,
                ai_route=home_route_res.ai_recommended_route if home_route_res else None,
                alternative_routes=home_route_res.alternative_routes if home_route_res else None,
                selected_alt_route=st.session_state.get("usr_home_alt"),
                theme_mode="light",
                title=map_title,
                key_prefix="usr_home_map",
                default_height=580,
                fit_to_routes=home_route_res is not None,
                tiles_available=tiles_available,
                simulate_tile_failure=simulate_tile_failure,
            )
            _apply_user_map_click(clicked_home, "usr_home_sel_j", "_usr_home_last_click")

        if home_route_res is not None:
            st.markdown("#### Recommended & Alternative Routes")
            st.info(f"**Why this route is recommended:** {home_route_res.recommended_reason}")
            rec_rt = home_route_res.ai_recommended_route
            all_alts = [rec_rt] + (
                [home_route_res.shortest_route]
                if home_route_res.shortest_route.junction_path != rec_rt.junction_path
                else []
            ) + home_route_res.alternative_routes

            cols = st.columns(min(len(all_alts), 3))
            dist_note = "OSM Roads" if home_route_res.uses_real_osm_roads else "Fallback"
            for idx, alt in enumerate(all_alts):
                with cols[idx % len(cols)]:
                    _render_compact_route_card(
                        rt=alt,
                        is_rec=(alt == rec_rt),
                        dist_note=dist_note,
                        btn_key=f"usr_home_view_alt_{idx}",
                        selected_alt_key="usr_home_alt",
                        baseline_time=rec_rt.estimated_travel_time_min,
                    )

    # =========================================================================
    # 2. TRAFFIC MAP
    # =========================================================================
    elif current_page in ("Traffic Map", "Traffic"):
        st.markdown(
            """
            <div class="brand-banner">
                <div class="brand-title">Gandhipuram Live Traffic Map</div>
                <div class="brand-subtitle">
                    Explore real-time street conditions, vehicle volumes, and speed observations across Gandhipuram's 6 monitored junctions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        per_junc_df = predictor.get_latest_per_junction_snapshot()
        sel_tr_j = st.selectbox(
            "Selected Junction",
            options=junctions,
            index=junctions.index(st.session_state["usr_tr_sel_j"])
            if st.session_state.get("usr_tr_sel_j") in junctions
            else 0,
            key="usr_tr_picker",
        )
        if sel_tr_j != st.session_state.get("_usr_tr_picker_prev"):
            st.session_state["_usr_tr_picker_prev"] = sel_tr_j
            st.session_state["usr_tr_sel_j"] = sel_tr_j
        active_tr_j = st.session_state.get("usr_tr_sel_j", sel_tr_j)

        col_tm, col_td = st.columns([1.65, 1.0])
        with col_tm:
            tr_nodes = build_map_nodes(per_junc_df, selected_junction=active_tr_j, road_network=road_network)
            clicked_tr = render_traffic_map(
                nodes=tr_nodes,
                road_network=road_network,
                theme_mode="light",
                title="Gandhipuram Traffic Status Map",
                key_prefix="usr_tr_map",
                default_height=520,
                fit_to_routes=False,
                tiles_available=tiles_available,
                simulate_tile_failure=simulate_tile_failure,
            )
            _apply_user_map_click(clicked_tr, "usr_tr_sel_j", "_usr_tr_last_click")
        with col_td:
            _render_user_junction_spotlight(active_tr_j, per_junc_df)

        st.markdown("#### All Six Monitored Junctions")
        cols = st.columns(3)
        for idx, (_, row) in enumerate(per_junc_df.iterrows()):
            with cols[idx % 3]:
                cond = str(row["Traffic_Condition"])
                border_c = CONDITION_COLORS.get(cond, "#CBD5E1")
                st.markdown(
                    f'<div class="junction-card" style="border-top:3px solid {border_c};">'
                    f'<div style="font-size:1.02rem;font-weight:700;color:#202A28;margin-bottom:4px;">{row["Location"]}</div>'
                    f'<div style="margin-bottom:6px;">{status_pill_html(cond, friendly=True)}</div>'
                    f'<div style="font-size:0.86rem;color:#334155;line-height:1.6;">'
                    f'<b>Vehicles:</b> {int(row["Vehicle_Count"])} &nbsp;|&nbsp; '
                    f'<b>Speed:</b> {float(row["Traffic_Speed_kmh"]):.1f} km/h<br>'
                    f'<b>Occupancy:</b> {float(row["Road_Occupancy_pct"]):.1f}% &nbsp;|&nbsp; '
                    f'<b>Signal:</b> {row["Traffic_Light_State"]}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # =========================================================================
    # 3. FIND ROUTE (Balanced 2-Column Real-Map Layout)
    # =========================================================================
    elif current_page in ("Find Route", "Find Best Route"):
        st.markdown(
            """
            <div class="brand-banner">
                <div class="brand-title">Smart Route Recommendation & Navigation</div>
                <div class="brand-subtitle">
                    Compares direct shortest distance paths against AI traffic-aware routes that intelligently bypass congestion.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_panel, col_map = st.columns([1.0, 1.65])

        with col_panel:
            with st.container(border=True):
                st.markdown("##### Route Search")
                u_orig = st.selectbox("Starting Location", options=junctions, index=3, key="usr_rt_orig")
                u_dest = st.selectbox(
                    "Destination",
                    options=[j for j in junctions if j != u_orig],
                    index=0,
                    key="usr_rt_dest",
                )
                route_now = datetime.now().replace(second=0, microsecond=0)
                default_departure = route_now + timedelta(hours=1)
                u_departure_date = st.date_input(
                    "Departure Date",
                    value=default_departure.date(),
                    min_value=route_now.date(),
                    key="usr_rt_date",
                )
                u_departure_time = st.time_input(
                    "Departure Time",
                    value=default_departure.time(),
                    key="usr_rt_time",
                )
                departure_datetime = datetime.combine(u_departure_date, u_departure_time)
                departure_is_future = departure_datetime > datetime.now()
                if not departure_is_future:
                    st.info("Choose a future departure time to compute the route.")

                route_request_signature = (
                    u_orig,
                    u_dest,
                    departure_datetime.strftime("%d-%m-%Y %H:%M"),
                )
                compute_clicked = st.button(
                    "COMPUTE BEST ROUTE",
                    type="primary",
                    width="stretch",
                    key="btn_usr_find_route",
                    disabled=not departure_is_future,
                )
                if compute_clicked or (
                    "usr_route_res" not in st.session_state
                    and departure_is_future
                ):
                    with st.spinner("Calculating OpenStreetMap route options..."):
                        route_forecasts = predictor.estimate_datetime_profile(departure_datetime)
                        route_forecast_map = {fc.junction: fc for fc in route_forecasts}
                        bundle = compute_unified_route_comparison(
                            router=router,
                            road_network=road_network,
                            origin=u_orig,
                            destination=u_dest,
                            departure_horizon_hours=0,
                            departure_datetime=departure_datetime,
                            forecasts_by_junction=route_forecast_map,
                        )
                        st.session_state["usr_route_res"] = bundle
                        st.session_state["usr_route_res_signature"] = route_request_signature
                        st.session_state["usr_rt_alt_view"] = None

            res_bundle: UnifiedRouteBundle | None = st.session_state.get("usr_route_res")
            if st.session_state.get("usr_route_res_signature") != route_request_signature:
                res_bundle = None
                st.info("Route inputs changed. Compute the route again to update its results.")
            if res_bundle:
                st.caption(f"Departure: {res_bundle.departure_timestamp}")
                st.info(f"**Recommendation Rationale:** {res_bundle.recommended_reason}")
                rec_rt = res_bundle.ai_recommended_route
                dist_note = "OSM Roads" if res_bundle.uses_real_osm_roads else "Fallback"

                # AI Recommended Route
                _render_compact_route_card(
                    rt=rec_rt,
                    is_rec=True,
                    dist_note=dist_note,
                    btn_key="usr_rt_rec_card",
                    selected_alt_key="usr_rt_alt_view",
                )

                # Alternative Routes
                candidate_alts = []
                if res_bundle.shortest_route.junction_path != rec_rt.junction_path:
                    candidate_alts.append(res_bundle.shortest_route)
                candidate_alts.extend(res_bundle.alternative_routes)

                if candidate_alts:
                    st.markdown("###### Alternative Options")
                    for idx, alt in enumerate(candidate_alts):
                        _render_compact_route_card(
                            rt=alt,
                            is_rec=False,
                            dist_note=dist_note,
                            btn_key=f"usr_rt_view_alt_{idx}",
                            selected_alt_key="usr_rt_alt_view",
                            baseline_time=rec_rt.estimated_travel_time_min,
                        )

            # Selected Junction Detail below route cards
            sel_rt_j = st.selectbox(
                "Inspect Junction Details",
                options=junctions,
                index=junctions.index(st.session_state.get("usr_rt_sel_j", u_orig))
                if st.session_state.get("usr_rt_sel_j") in junctions
                else 0,
                key="usr_rt_junc_picker",
            )
            st.session_state["usr_rt_sel_j"] = sel_rt_j
            if departure_is_future:
                inspector_forecasts = predictor.estimate_datetime_profile(
                    departure_datetime,
                    junctions=[sel_rt_j],
                )
                if inspector_forecasts:
                    _render_user_forecast_spotlight(sel_rt_j, inspector_forecasts[0])
                else:
                    _render_user_junction_spotlight(sel_rt_j, snap_df)
            else:
                _render_user_junction_spotlight(sel_rt_j, snap_df)

        with col_map:
            if res_bundle:
                fc_rt = {
                    name: JunctionForecast(**forecast)
                    for name, forecast in res_bundle.junction_forecasts.items()
                }
                rt_nodes = build_map_nodes(
                    snap_df,
                    forecasts_by_junction=fc_rt,
                    use_forecast_values=True,
                    origin_junction=res_bundle.origin,
                    destination_junction=res_bundle.destination,
                    road_network=road_network,
                )
                clicked_map_j = render_traffic_map(
                    nodes=rt_nodes,
                    road_network=road_network,
                    shortest_route=res_bundle.shortest_route,
                    ai_route=res_bundle.ai_recommended_route,
                    alternative_routes=res_bundle.alternative_routes,
                    selected_alt_route=st.session_state.get("usr_rt_alt_view"),
                    theme_mode="light",
                    title=f"Route: {res_bundle.origin} ➔ {res_bundle.destination} (Departure {res_bundle.departure_timestamp})",
                    key_prefix="usr_rt_map",
                    default_height=600,
                    fit_to_routes=True,
                    tiles_available=tiles_available,
                    simulate_tile_failure=simulate_tile_failure,
                )
                _apply_user_map_click(clicked_map_j, "usr_rt_sel_j", "_usr_rt_last_click")

    # =========================================================================
    # 4. FUTURE TRAFFIC
    # =========================================================================
    elif current_page in ("Future Traffic", "Future"):
        st.markdown(
            """
            <div class="brand-banner">
                <div class="brand-title">Future Traffic Congestion Forecast</div>
                <div class="brand-subtitle">
                    Select a junction and future date and time to view its typical traffic profile.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        now = datetime.now().replace(second=0, microsecond=0)
        default_target = now + timedelta(hours=2)
        col_fc1, col_fc2, col_fc3 = st.columns([1.3, 1.0, 1.0])
        with col_fc1:
            sel_fc_j = st.selectbox("Select Junction", options=junctions, index=0, key="usr_fc_junc")
        with col_fc2:
            sel_fc_date = st.date_input(
                "Future Date",
                value=default_target.date(),
                min_value=now.date(),
                key="usr_fc_date",
            )
        with col_fc3:
            sel_fc_time = st.time_input(
                "Future Time",
                value=default_target.time(),
                key="usr_fc_time",
            )

        target_datetime = datetime.combine(sel_fc_date, sel_fc_time)
        if target_datetime <= datetime.now():
            st.info("Choose a future time to view its traffic profile.")
        else:
            fcs = predictor.estimate_datetime_profile(target_datetime)
            fc_map = {forecast.junction: forecast for forecast in fcs}
            target_fc = fc_map.get(sel_fc_j)

            if target_fc:
                st.markdown(
                    f"""
                    <div class="kpi-card" style="margin:12px 0;border-left:4px solid #176B68;">
                        <div style="font-size:1.05rem;font-weight:700;color:#202A28;margin-bottom:4px;">
                            Traffic Profile for {target_fc.junction} at {target_fc.target_timestamp}
                        </div>
                        <div style="margin-bottom:6px;">{status_pill_html(target_fc.predicted_condition, friendly=True)}</div>
                        <div style="font-size:0.92rem;color:#1E293B;line-height:1.6;">
                            {_human_forecast_text(target_fc)}
                        </div>
                        <div style="font-size:0.84rem;color:#475569;margin-top:6px;">
                            <b>Expected Vehicles:</b> {target_fc.predicted_vehicle_count:.1f} &nbsp;|&nbsp;
                            <b>Expected Speed:</b> {target_fc.predicted_speed_kmh:.1f} km/h &nbsp;|&nbsp;
                            <b>Road Occupancy:</b> {target_fc.predicted_occupancy_pct:.1f}% &nbsp;|&nbsp;
                            <b>Congestion Risk:</b> {_friendly_risk_label(target_fc.congestion_risk_score)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if target_fc.is_outside_observed_hours:
                    st.caption("This time is outside the dataset's observed hours; the nearest available time profile is used.")

            col_fcm, col_fct = st.columns([1.4, 1.0])
            with col_fcm:
                fc_nodes = build_map_nodes(
                    snap_df,
                    forecasts_by_junction=fc_map,
                    use_forecast_values=True,
                    selected_junction=sel_fc_j,
                    road_network=road_network,
                )
                render_traffic_map(
                    nodes=fc_nodes,
                    road_network=road_network,
                    theme_mode="light",
                    title=f"Estimated Traffic Across Gandhipuram ({target_datetime:%d-%m-%Y %H:%M})",
                    key_prefix="usr_future_map",
                    default_height=460,
                    tiles_available=tiles_available,
                    simulate_tile_failure=simulate_tile_failure,
                )
            with col_fct:
                junction_rows = predictor.df[predictor.df["Location"] == sel_fc_j]
                profile_minutes = sorted(
                    set(
                        (junction_rows["Timestamp_dt"].dt.hour * 60 + junction_rows["Timestamp_dt"].dt.minute)
                        .astype(int)
                        .tolist()
                    )
                )
                profile_forecasts = []
                for minute in profile_minutes:
                    profile_time = datetime.combine(
                        sel_fc_date,
                        time(hour=minute // 60, minute=minute % 60),
                    )
                    profile_forecasts.extend(
                        predictor.estimate_datetime_profile(profile_time, junctions=[sel_fc_j])
                    )

                fig_traj = go.Figure()
                fig_traj.add_trace(
                    go.Scatter(
                        x=[forecast.target_timestamp[-5:] for forecast in profile_forecasts],
                        y=[forecast.predicted_speed_kmh for forecast in profile_forecasts],
                        mode="lines+markers",
                        name="Typical Speed (km/h)",
                        line=dict(color="#176B68", width=2.5),
                    )
                )
                if target_fc:
                    fig_traj.add_trace(
                        go.Scatter(
                            x=[target_datetime.strftime("%H:%M")],
                            y=[target_fc.predicted_speed_kmh],
                            mode="markers",
                            name="Selected Date/Time",
                            marker=dict(color="#F59E0B", size=11, symbol="diamond"),
                        )
                    )
                apply_control_center_theme(
                    fig_traj,
                    f"{sel_fc_j} Typical Daily Speed Profile",
                    x_title="Time of Day",
                    y_title="Speed (km/h)",
                    theme_mode="light",
                )
                st.plotly_chart(fig_traj, width="stretch")

    # =========================================================================
    # 5. TRAFFIC ALERTS
    # =========================================================================
    elif current_page in ("Traffic Alerts", "Alerts"):
        st.markdown(
            """
            <div class="brand-banner">
                <div class="brand-title">Proactive Congestion Early Warnings</div>
                <div class="brand-subtitle">
                    Early alerts for junctions predicted to experience heavy congestion in the next 1–6 hours.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        all_fcs = predictor.forecast_multi_horizon(max_horizon=6)
        al_bundle = predictor.generate_proactive_alerts(all_fcs)
        alerts = al_bundle["alerts"]

        focus_j = st.session_state.get("usr_alert_focus_j")

        col_alm, col_alc = st.columns([1.3, 1.0])
        with col_alm:
            fc_h1 = {f.junction: f for f in all_fcs if f.horizon_hours == 1}
            al_nodes = build_map_nodes(
                snap_df,
                forecasts_by_junction=fc_h1,
                use_forecast_values=True,
                selected_junction=focus_j,
                road_network=road_network,
            )
            render_traffic_map(
                nodes=al_nodes,
                road_network=road_network,
                theme_mode="light",
                title=f"Alert Focus Map ({focus_j or 'Network Overview'})",
                key_prefix="usr_alerts_map",
                focus_junction=focus_j,
                default_height=500,
                tiles_available=tiles_available,
                simulate_tile_failure=simulate_tile_failure,
            )

        with col_alc:
            if not alerts:
                st.success("No heavy congestion alerts predicted for the next 6 hours. Smooth travel expected!")
            else:
                for idx, al in enumerate(alerts):
                    b_color = "#DC2626" if al.alert_level == "Critical" else "#D97706"
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style="border-left:4px solid {b_color};padding-left:8px;">
                                <div style="font-weight:700;font-size:1.02rem;color:#202A28;">{al.junction}</div>
                                <div style="font-size:0.86rem;color:#475569;margin:4px 0;">
                                    <b>Expected In:</b> {al.horizon_hours} hour(s) &nbsp;|&nbsp; <b>Risk:</b> {al.congestion_risk_score:.0f}/100
                                </div>
                                <div style="font-size:0.88rem;color:#334155;line-height:1.5;">
                                    {al.recommended_action}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Focus on Map", key=f"usr_alert_focus_{idx}"):
                            st.session_state["usr_alert_focus_j"] = al.junction
                            st.rerun()

    # =========================================================================
    # 6. EMERGENCY INFORMATION
    # =========================================================================
    elif current_page in ("Emergency Information", "Emergency Info"):
        st.markdown(
            """
            <div class="brand-banner">
                <div class="brand-title">Emergency Priority Corridor Information</div>
                <div class="brand-subtitle">
                    Live public advisories regarding active emergency vehicles, green corridors, and recommended detours.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_ev = emg_mgr.get_active_event()

        if active_ev is not None and active_ev.origin_junction and active_ev.destination_junction:
            st.error(
                f"### 🚨 Active Emergency Vehicle Priority Corridor\n\n"
                f"- **Emergency Vehicle:** {active_ev.vehicle_type}\n"
                f"- **Priority Route Corridor:** **{active_ev.origin_junction}** ➔ **{active_ev.destination_junction}**\n"
                f"- **Detection Time:** {active_ev.detected_time_str}\n"
                f"- **Signal Status:** Simulated green-wave priority active along corridor\n\n"
                f"**Public Advisory:** Motorists traveling along this corridor are urged to yield right-of-way "
                f"and use alternative streets to prevent bottleneck delays.",
                icon="🚨",
            )

            st.markdown("#### Recommended Detour Corridors")
            st.info(
                f"Travelers wishing to avoid **{active_ev.origin_junction} ➔ {active_ev.destination_junction}** "
                f"are advised to divert via Dr Nanjappa Road or Sathy Road bypass corridors."
            )

            col_em1, col_em2 = st.columns([1.6, 1.0])
            with col_em1:
                emg_fc = {f.junction: f for f in predictor.predict_horizon(1)}
                em_nodes = build_map_nodes(
                    snap_df,
                    forecasts_by_junction=emg_fc,
                    use_forecast_values=True,
                    origin_junction=active_ev.origin_junction,
                    destination_junction=active_ev.destination_junction,
                    road_network=road_network,
                )
                render_traffic_map(
                    nodes=em_nodes,
                    road_network=road_network,
                    theme_mode="light",
                    title=f"Active Emergency Corridor ({active_ev.vehicle_type})",
                    key_prefix="usr_emg_map",
                    default_height=480,
                    tiles_available=tiles_available,
                    simulate_tile_failure=simulate_tile_failure,
                )
            with col_em2:
                with st.container(border=True):
                    st.markdown("##### Driver Safety Instructions")
                    st.write("1. **Stay Alert:** Listen for emergency sirens and flashing lights.")
                    st.write("2. **Yield Right of Way:** Pull over safely to the left curb where possible.")
                    st.write("3. **Do Not Block Intersections:** Leave the junction clear even on green light.")
                    st.write("4. **Use Detours:** Consult the route finder for real-time congestion detours.")
        else:
            st.success(
                "### 🟢 Normal Traffic Conditions\n\n"
                "There are currently **no active emergency vehicle priority corridors** in effect. "
                "All traffic signals are operating on normal schedule."
            )
            st.info(
                "When an emergency vehicle (Ambulance, Fire Engine, Police) is verified by the Control Centre, "
                "an advisory with priority corridors and detour recommendations will automatically appear here."
            )

    # =========================================================================
    # 7. ACCOUNT
    # =========================================================================
    elif current_page in ("Account", "My Account"):
        st.markdown(
            """
            <div class="brand-banner">
                <div class="brand-title">My Account & Preferences</div>
                <div class="brand-subtitle">
                    Manage your credentials and view account permissions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            with st.container(border=True):
                st.markdown("#### Account Information")
                st.write(f"**Username / Email:** `{verified_user.username}`")
                st.write(f"**Role:** `Consumer / Normal User`")
                st.write(f"**Account Created:** {verified_user.created_at}")
                st.write(f"**Account Status:** {'Active' if verified_user.is_active else 'Inactive'}")

        with col_a2:
            with st.container(border=True):
                st.markdown("#### Change Password")
                with st.form("usr_change_password_form"):
                    c_old = st.text_input("Current Password", type="password", key="usr_pw_old")
                    c_new = st.text_input("New Password (min 8 chars)", type="password", key="usr_pw_new")
                    c_conf = st.text_input("Confirm New Password", type="password", key="usr_pw_conf")
                    change_password = st.form_submit_button(
                        "Update Password",
                        type="primary",
                        key="btn_usr_change_pw",
                        width="stretch",
                    )
                if change_password:
                    if not c_old or not c_new or not c_conf:
                        st.error("Fill in all three password fields, then try again.")
                    elif len(c_new) < 8:
                        st.error("Use at least 8 characters for the new password.")
                    elif c_new != c_conf:
                        st.error("New passwords do not match.")
                    else:
                        try:
                            auth.change_own_password(auth_token, c_old, c_new)
                            st.success("Password updated. Use the new password next time you sign in.")
                        except Exception as err:
                            st.error(f"Error updating password: {err}")

    render_limitation_footer()
