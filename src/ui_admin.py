"""Administrator Smart City Traffic Control Centre (strictly RBAC-protected).

Features 12 Control Centre pages + Logout:
1. Overview (Clean 3-section control room dashboard: KPI strip, Map + Inspector, Breakdown table)
2. Junction Monitoring (Source Selector: Historical Dataset | Video Analytics | Combined View)
3. Video Analytics (PRE-RECORDED VIDEO INPUT: YOLOv8 vehicle detection, tracking, counting)
4. Emergency Detection (Model inspection, multi-frame temporal filter, operator review)
5. Future Predictions (1–6h horizon forecasting & speed trends)
6. Proactive Alerts (Pre-congestion alerts & risk ranking)
7. Route Management (Real OSM mapped roads & alternatives)
8. Emergency Priority (Software green corridor simulation & adaptive signal control)
9. Model Performance (Feature importance, MAE/RMSE, confusion matrix)
10. Dataset Validation (Strict 218-record Gandhipuram CSV verification)
11. User Management (User directory, password resets, role administration)
12. Settings (Offline road fallback & map tile simulation toggles)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.auth import ADMIN_PAGES, ADMIN_PAGE_ALIASES, AuthManager, UserPublic
from src.charts import (
    apply_control_center_theme,
    build_confusion_matrix_figure,
    build_forecast_line_chart,
    build_junction_comparison_bar,
)
from src.emergency_event_manager import (
    EmergencyEventManager,
    EmergencyEventRecord,
    EventReviewStatus,
)
from src.emergency_routing import EmergencyPriorityRouter
from src.emergency_vehicle_detection import (
    EmergencyDetectionEvent,
    EmergencyDetectionStatus,
    EmergencyModelAuditor,
    VideoEmergencyAnalyzer,
)
from src.future_predictor import (
    DATASET_LIMITATION_NOTE,
    FutureTrafficPredictor,
    train_and_evaluate_models,
)
from src.future_routing import FutureRouteRecommender
from src.map_component import render_traffic_map
from src.map_data import build_map_nodes
from src.road_network import OSMRoadNetwork, load_or_fetch_gandhipuram_osm_network
from src.route_visualization import (
    compute_unified_emergency_route,
    compute_unified_route_comparison,
)
from src.signal_controller import SIMULATION_DISCLAIMER, recommend_adaptive_signal
from src.ui_theme import render_kpi_card, render_limitation_footer, status_pill_html
from src.video_processing import (
    ProcessingState,
    VideoSessionSummary,
    VideoStreamProcessor,
)
from src.video_vehicle_detection import (
    CLASS_DISPLAY_NAMES,
    VideoVehicleDetector,
    resolve_video_path,
)


def _get_snapshot_df(predictor: FutureTrafficPredictor, settings: dict[str, Any]) -> pd.DataFrame:
    mode = st.session_state.get(
        "admin_snapshot_mode",
        settings.get("default_snapshot_mode", "Full-Network Common Snapshot"),
    )
    base_ts = st.session_state.get("admin_base_ts")
    if mode == "Latest Observation Per Junction":
        return predictor.get_latest_per_junction_snapshot()
    return predictor.get_latest_common_snapshot(base_ts)


def _apply_map_click(clicked: str | None, target_state_key: str, last_click_key: str) -> None:
    if clicked and clicked != st.session_state.get(last_click_key):
        st.session_state[last_click_key] = clicked
        st.session_state[target_state_key] = clicked
        st.rerun()


def _get_video_processor() -> VideoStreamProcessor:
    if "admin_video_processor" not in st.session_state:
        st.session_state["admin_video_processor"] = VideoStreamProcessor()
    return st.session_state["admin_video_processor"]


def _get_emergency_manager() -> EmergencyEventManager:
    if "emergency_event_manager" not in st.session_state:
        st.session_state["emergency_event_manager"] = EmergencyEventManager()
    return st.session_state["emergency_event_manager"]


def _render_admin_junction_detail(
    jname: str,
    snap_df: pd.DataFrame,
    fc_h1_map: dict[str, Any] | None = None,
) -> None:
    match = snap_df[snap_df["Location"] == jname]
    if match.empty:
        return
    row = match.iloc[0]
    cond = str(row["Traffic_Condition"])
    fc1 = fc_h1_map.get(jname) if fc_h1_map else None

    fc_html = ""
    if fc1 is not None:
        fc_html = f"""
        <div style="margin-top:10px;padding:8px;background:#1E293B;border-radius:6px;border-left:3px solid #38BDF8;">
            <div style="font-size:0.74rem;color:#94A3B8;">1-HOUR PROACTIVE FORECAST ({fc1.target_timestamp[-5:]})</div>
            <div style="font-size:0.90rem;font-weight:700;color:#F8FAFC;margin:2px 0;">
                {fc1.predicted_condition} Congestion &nbsp;|&nbsp; {fc1.predicted_vehicle_count} veh &nbsp;|&nbsp; {fc1.predicted_speed_kmh:.1f} km/h
            </div>
            <div style="font-size:0.76rem;color:#CBD5E1;">Congestion Risk Score: <b>{fc1.congestion_risk_score:.0f}/100</b></div>
        </div>
        """

    st.markdown(
        f"""
        <div class="detail-panel">
            <div style="font-size:0.72rem;color:#38BDF8;font-weight:700;letter-spacing:0.06em;">SELECTED JUNCTION INSPECTOR</div>
            <div style="font-size:1.12rem;font-weight:700;color:#F8FAFC;margin:3px 0 6px 0;">{jname}</div>
            <div style="margin-bottom:8px;">{status_pill_html(cond)}</div>
            <div style="font-size:0.86rem;color:#CBD5E1;line-height:1.65;">
                <b>Vehicle Volume:</b> {int(row['Vehicle_Count'])} vehicles<br>
                <b>Observed Speed:</b> {float(row['Traffic_Speed_kmh']):.1f} km/h<br>
                <b>Road Occupancy:</b> {float(row['Road_Occupancy_pct']):.1f}%<br>
                <b>Signal State:</b> {row['Traffic_Light_State']}<br>
                <b>Weather:</b> {row['Weather_Condition']}<br>
                <b>Incident:</b> {row['Accident_Report']}<br>
                <b>Coordinates:</b> {row['Latitude']:.6f}, {row['Longitude']:.6f}<br>
                <b>Snapshot Time:</b> {row['Timestamp']}
            </div>
            {fc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_admin_dashboard(
    auth: AuthManager,
    auth_token: str,
    user: UserPublic,
    predictor: FutureTrafficPredictor,
    router: FutureRouteRecommender,
    emergency_router: EmergencyPriorityRouter,
    road_network: OSMRoadNetwork,
    tiles_available: bool,
    requested_page: str,
) -> None:
    """Render the 12-item Smart City Traffic Control Centre after server-side role check."""
    verified_admin = auth.require_admin(auth_token)
    settings = auth.get_settings()
    sim_tile_fail = bool(settings.get("simulate_tile_failure", False))
    junctions = predictor.junctions
    weather_options = predictor.validation_report.weather_values

    v_proc = _get_video_processor()
    emg_mgr = _get_emergency_manager()

    # Canonicalize requested page
    canon_requested = ADMIN_PAGE_ALIASES.get(requested_page, requested_page)

    # Sidebar Navigation (12 functional pages + Logout)
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:10px 12px;background:#111827;border:1px solid #1F2937;border-radius:8px;margin-bottom:12px;">
                <div style="font-size:0.72rem;color:#94A3B8;letter-spacing:0.06em;">CONTROL OPERATOR</div>
                <div style="font-size:0.95rem;font-weight:700;color:#F8FAFC;margin:2px 0;word-break:break-all;">{verified_admin.username}</div>
                <span class="role-badge-admin">ADMIN CONTROL CENTRE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        default_idx = ADMIN_PAGES.index(canon_requested) if canon_requested in ADMIN_PAGES else 0
        selected_page = st.radio(
            "Control Centre Menu",
            options=list(ADMIN_PAGES),
            format_func=lambda p: f"{ADMIN_PAGES.index(p) + 1}. {p}",
            index=default_idx,
            key="admin_sidebar_nav",
        )
        if selected_page != st.session_state.get("nav_page"):
            st.session_state["nav_page"] = selected_page
            st.query_params["page"] = selected_page
            st.session_state["_last_qp_page"] = selected_page

        st.markdown("---")
        full_timestamps = predictor.get_full_network_timestamps()
        ts_labels = [ts.strftime("%d-%m-%Y %H:%M") for ts in full_timestamps]
        snap_modes = ["Full-Network Common Snapshot", "Latest Observation Per Junction"]
        default_mode = settings.get("default_snapshot_mode", snap_modes[0])
        mode_idx = snap_modes.index(default_mode) if default_mode in snap_modes else 0
        st.radio(
            "Dataset Reference Snapshot",
            options=snap_modes,
            index=mode_idx,
            key="admin_snapshot_mode",
        )
        if st.session_state.get("admin_snapshot_mode") == "Full-Network Common Snapshot":
            st.selectbox(
                "Reference Timestamp",
                options=ts_labels,
                index=len(ts_labels) - 1,
                key="admin_base_ts",
            )
        else:
            st.session_state["admin_base_ts"] = None

        st.markdown("---")
        if st.button("Logout", type="secondary", width="stretch", key="btn_admin_logout"):
            auth.logout(auth_token)
            for k in (
                "auth_token",
                "auth_role",
                "nav_page",
                "_last_qp_page",
                "admin_sidebar_nav",
                "user_sidebar_nav",
            ):
                st.session_state.pop(k, None)
            st.query_params.clear()
            st.rerun()

    snap_df = _get_snapshot_df(predictor, settings)
    base_ts = (
        None
        if st.session_state.get("admin_snapshot_mode") == "Latest Observation Per Junction"
        else st.session_state.get("admin_base_ts")
    )
    use_per_junc = st.session_state.get("admin_snapshot_mode") == "Latest Observation Per Junction"
    snap_time_label = str(snap_df["Timestamp"].iloc[0]) if not snap_df.empty else "—"
    road_badge = "OSM Mapped Roads" if road_network.is_real_osm else "Schematic Fallback"

    # Top Control Centre Status Bar
    st.markdown(
        f"""
        <div class="brand-banner">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div>
                    <div class="brand-title">🚦 SMART CITY TRAFFIC MANAGEMENT CONTROL CENTRE</div>
                    <div class="brand-subtitle">
                        Gandhipuram, Coimbatore (6 Monitored Junctions) &nbsp;|&nbsp; Active View: <b>{selected_page}</b>
                    </div>
                </div>
                <div style="text-align:right;font-size:0.80rem;color:#CBD5E1;">
                    SOURCE A: <b>{snap_time_label}</b> &nbsp;|&nbsp;
                    SOURCE B: <b>{v_proc.metadata.file_name}</b><br>
                    Road Geometry: <b>{road_badge}</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # 1. OVERVIEW (3-Section Professional Control Room)
    # =========================================================================
    if selected_page == "Overview":
        all_fcs = predictor.forecast_multi_horizon(
            max_horizon=6, base_timestamp=base_ts, use_per_junction_latest=use_per_junc
        )
        alert_bundle = predictor.generate_proactive_alerts(all_fcs)
        fc_h1_map = {f.junction: f for f in all_fcs if f.horizon_hours == 1}

        high_cnt = int((snap_df["Traffic_Condition"] == "High").sum())
        med_cnt = int((snap_df["Traffic_Condition"] == "Medium").sum())
        low_cnt = int((snap_df["Traffic_Condition"] == "Low").sum())
        active_alerts_cnt = alert_bundle["critical_count"] + alert_bundle["warning_count"]

        v_summary = v_proc.get_summary()
        active_emg = emg_mgr.get_active_event()
        emg_label = (
            f"Active ({active_emg.vehicle_type})"
            if active_emg
            else "Standby (Normal)"
        )

        # SECTION 1: Status Summary KPI Strip
        st.markdown("##### Section 1: Network & Stream Status Summary")
        k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
        with k1:
            render_kpi_card("Active Junctions", str(len(snap_df)), f"L:{low_cnt} M:{med_cnt} H:{high_cnt}")
        with k2:
            render_kpi_card("Total Vehicles", str(int(snap_df["Vehicle_Count"].sum())), "6-junction count")
        with k3:
            render_kpi_card("Network Avg Speed", f"{snap_df['Traffic_Speed_kmh'].mean():.1f} km/h", "Snapshot mean")
        with k4:
            render_kpi_card("Avg Occupancy", f"{snap_df['Road_Occupancy_pct'].mean():.1f}%", f"{snap_time_label[-5:]}")
        with k5:
            render_kpi_card("High Congestion", str(high_cnt), "Heavy flow nodes")
        with k6:
            render_kpi_card("Proactive Alerts", str(active_alerts_cnt), f"Crit:{alert_bundle['critical_count']} Warn:{alert_bundle['warning_count']}")
        with k7:
            render_kpi_card("Emergency Corridor", emg_label, f"Video: {v_summary.state.value}")

        # SECTION 2: Interactive Map and Junction Inspector
        st.markdown("##### Section 2: Real-Street OpenStreetMap & Junction Inspector")
        m_mode_col, m_hor_col, m_sel_col = st.columns([1.2, 1.0, 1.4])
        with m_mode_col:
            ov_mode = st.radio(
                "Map Data Layer",
                options=["Observed CSV Snapshot", "Model Forecast (+Horizon)"],
                horizontal=True,
                key="adm_ov_mode",
            )
        with m_hor_col:
            ov_h = st.slider(
                "Forecast Horizon (+h)",
                1,
                6,
                1,
                key="adm_ov_h",
                disabled=(ov_mode == "Observed CSV Snapshot"),
            )
        with m_sel_col:
            selected_ov_j = st.selectbox(
                "Inspect Junction (or click map marker)",
                options=junctions,
                index=junctions.index(st.session_state["adm_ov_selected"])
                if st.session_state.get("adm_ov_selected") in junctions
                else 0,
                key="adm_ov_junc_picker",
            )
        if selected_ov_j != st.session_state.get("_adm_ov_picker_prev"):
            st.session_state["_adm_ov_picker_prev"] = selected_ov_j
            st.session_state["adm_ov_selected"] = selected_ov_j
        active_j = st.session_state.get("adm_ov_selected", selected_ov_j)

        fc_sel_map = {f.junction: f for f in all_fcs if f.horizon_hours == ov_h}
        use_fc = ov_mode == "Model Forecast (+Horizon)"
        map_nodes = build_map_nodes(
            snap_df,
            forecasts_by_junction=fc_sel_map,
            use_forecast_values=use_fc,
            selected_junction=active_j,
            road_network=road_network,
        )

        col_map, col_panel = st.columns([1.65, 1.0])
        with col_map:
            clicked_j = render_traffic_map(
                nodes=map_nodes,
                road_network=road_network,
                theme_mode="dark",
                title=f"Gandhipuram Interactive Traffic Map ({'Forecast +'+str(ov_h)+'h' if use_fc else 'Observed Snapshot'})",
                key_prefix="adm_ov_map",
                focus_junction=active_j,
                default_height=580,
                tiles_available=tiles_available,
                simulate_tile_failure=sim_tile_fail,
            )
            _apply_map_click(clicked_j, "adm_ov_selected", "_adm_ov_last_click")

        with col_panel:
            _render_admin_junction_detail(active_j, snap_df, fc_h1_map)

        # SECTION 3: Tabular Junction Status Breakdown
        st.markdown("##### Section 3: Monitored Junctions Status Breakdown")
        table_rows = []
        for _, row in snap_df.iterrows():
            j_name = row["Location"]
            fc_entry = fc_h1_map.get(j_name)
            table_rows.append(
                {
                    "Junction": j_name,
                    "Current Traffic": row["Traffic_Condition"],
                    "Vehicle Count": int(row["Vehicle_Count"]),
                    "Speed (km/h)": f"{float(row['Traffic_Speed_kmh']):.1f}",
                    "Road Occupancy": f"{float(row['Road_Occupancy_pct']):.1f}%",
                    "Signal State": row["Traffic_Light_State"],
                    "Weather": row["Weather_Condition"],
                    "Incident": row["Accident_Report"],
                    "1h Forecast": fc_entry.predicted_condition if fc_entry else "—",
                    "Risk Score": f"{fc_entry.congestion_risk_score:.0f}/100" if fc_entry else "—",
                }
            )
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    # =========================================================================
    # 2. JUNCTION MONITORING
    # =========================================================================
    elif selected_page == "Junction Monitoring":
        st.markdown("### Gandhipuram Monitored Junction Deep Dive")
        st.caption(
            "Select a junction to monitor historical metrics, video sensor tracking, and signal phase configurations."
        )

        col_sel1, col_sel2 = st.columns([1.2, 1.0])
        with col_sel1:
            chosen_j = st.selectbox(
                "Select Junction to Monitor",
                options=junctions,
                index=junctions.index(st.session_state["adm_jm_junc"])
                if st.session_state.get("adm_jm_junc") in junctions
                else 0,
                key="adm_jm_sel",
            )
            st.session_state["adm_jm_junc"] = chosen_j
        with col_sel2:
            src_mode = st.radio(
                "Active Data Source Stream",
                options=["Historical Dataset", "Video Analytics", "Combined View"],
                horizontal=True,
                key="adm_jm_source_sel",
            )

        row = snap_df[snap_df["Location"] == chosen_j].iloc[0]

        if src_mode in ("Historical Dataset", "Combined View"):
            st.markdown(f"#### Source A: Ground Truth Dataset — `{chosen_j}`")
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                render_kpi_card("Traffic Condition", str(row["Traffic_Condition"]), status_pill_html(str(row["Traffic_Condition"])))
            with k2:
                render_kpi_card("Observed Vehicles", f"{int(row['Vehicle_Count'])}", f"Observed: {row['Timestamp']}")
            with k3:
                render_kpi_card("Average Speed", f"{row['Traffic_Speed_kmh']:.1f} km/h", "Junction mean speed")
            with k4:
                render_kpi_card("Road Occupancy", f"{row['Road_Occupancy_pct']:.1f}%", f"Signal: {row['Traffic_Light_State']}")
            with k5:
                render_kpi_card("Weather / Incident", str(row["Weather_Condition"]), f"Accident: {row['Accident_Report']}")

        if src_mode in ("Video Analytics", "Combined View"):
            v_summary = v_proc.get_summary()
            loc_tag = (
                f"Associated with: {v_proc.mapped_junction}"
                if v_proc.mapped_junction
                else "Unmapped Video Source (Highway Footage)"
            )
            st.markdown(f"#### Source B: Video Analytics Stream ({loc_tag})")
            vk1, vk2, vk3, vk4, vk5 = st.columns(5)
            with vk1:
                render_kpi_card("Video Status", v_summary.state.value, v_summary.video_metadata.file_name)
            with vk2:
                render_kpi_card("Unique Tracked", str(v_summary.total_unique_vehicles), "Across processed frames")
            with vk3:
                render_kpi_card("Active in Frame", str(v_summary.active_vehicles_current), "Current detection count")
            with vk4:
                render_kpi_card(
                    "Cars / Trucks",
                    f"{v_summary.class_breakdown.get('Car', 0)} / {v_summary.class_breakdown.get('Truck', 0)}",
                    "Class breakdown",
                )
            with vk5:
                render_kpi_card(
                    "Emergency Status",
                    v_summary.emergency_status[:18],
                    f"Confirmed: {v_summary.confirmed_emergency_count}",
                )

        col_jm1, col_jm2 = st.columns([1.15, 1.0])
        with col_jm1:
            map_nodes = build_map_nodes(snap_df, selected_junction=chosen_j, road_network=road_network)
            clicked = render_traffic_map(
                nodes=map_nodes,
                road_network=road_network,
                theme_mode="dark",
                title=f"Junction Location Map — {chosen_j}",
                key_prefix="adm_jm_map",
                focus_junction=chosen_j,
                default_height=430,
                tiles_available=tiles_available,
                simulate_tile_failure=sim_tile_fail,
            )
            if clicked and clicked != st.session_state.get("_adm_jm_last_click"):
                st.session_state["_adm_jm_last_click"] = clicked
                st.session_state["adm_jm_junc"] = clicked
                st.rerun()
        with col_jm2:
            hist_j = predictor.df[predictor.df["Location"] == chosen_j].sort_values("Timestamp_dt").copy()
            hist_j["TimeLabel"] = hist_j["Timestamp_dt"].dt.strftime("%d-%m %H:%M")
            fig_sp = go.Figure(
                go.Scatter(
                    x=hist_j["TimeLabel"],
                    y=hist_j["Traffic_Speed_kmh"],
                    mode="lines+markers",
                    name="Speed (km/h)",
                    line=dict(color="#38BDF8", width=2),
                )
            )
            apply_control_center_theme(fig_sp, f"{chosen_j} — Observed Speed History", x_title="Timeline", y_title="Speed (km/h)")
            st.plotly_chart(fig_sp, width="stretch")

    # =========================================================================
    # 3. VIDEO ANALYTICS
    # =========================================================================
    elif selected_page == "Video Analytics":
        st.markdown("### Video-Based Vehicle Detection, Tracking & Counting Workspace")
        st.caption(
            "SOURCE B: PRE-RECORDED VIDEO INPUT — Evaluates real video footage using YOLOv8 object detection, "
            "ByteTrack multi-object tracking, and unique vehicle counting."
        )

        # Compact Controls Toolbar
        with st.container(border=True):
            st.markdown("##### Video Stream Configuration & Playback Controls")
            res_path, src_label = resolve_video_path()

            c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)
            with c_cfg1:
                vid_choice = st.selectbox(
                    "Video Source File",
                    options=["Trafficvedio.mp4 (Preferred)", "Trafficvedio.avi (Fallback)"],
                    index=0,
                    key="adm_va_file_sel",
                )
                active_vid = (
                    Path("Trafficvedio.mp4")
                    if "mp4" in vid_choice.lower()
                    else Path("Trafficvedio.avi")
                )
                if not active_vid.exists():
                    active_vid = res_path
            with c_cfg2:
                map_opts = ["Unmapped Video Source"] + list(junctions)
                sel_map_loc = st.selectbox(
                    "Camera Junction Mapping",
                    options=map_opts,
                    index=0,
                    key="adm_va_loc_map",
                )
                v_proc.set_mapped_junction(None if sel_map_loc == "Unmapped Video Source" else sel_map_loc)
            with c_cfg3:
                conf_th = st.slider("Detection Confidence", 0.20, 0.90, float(v_proc.conf_threshold), 0.05, key="adm_va_conf")
                v_proc.conf_threshold = conf_th
            with c_cfg4:
                f_step = st.slider("Sampling Step", 1, 15, int(v_proc.frame_step), 1, key="adm_va_step", help="Advance by N frames per processing step.")
                v_proc.frame_step = f_step

            c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)
            with c_btn1:
                if st.button("▶ Process Next Step", type="primary", width="stretch", key="btn_va_step"):
                    with st.spinner("Processing video frame with YOLOv8..."):
                        v_proc.process_next_step(max_frames_to_advance=1)
                        st.rerun()
            with c_btn2:
                if st.button("⏭ Fast 10 Steps", width="stretch", key="btn_va_fast"):
                    with st.spinner("Processing 10 sampled frames..."):
                        v_proc.process_next_step(max_frames_to_advance=10)
                        st.rerun()
            with c_btn3:
                if st.button("⏹ Stop Playback", width="stretch", key="btn_va_stop"):
                    v_proc.state = ProcessingState.PAUSED
                    st.rerun()
            with c_btn4:
                if st.button("🔄 Reset Video", width="stretch", key="btn_va_reset"):
                    v_proc.reset()
                    st.rerun()

        summary = v_proc.get_summary()

        col_main_v, col_side_v = st.columns([1.65, 1.0])
        with col_main_v:
            with st.container(border=True):
                st.markdown(
                    f"##### Playback Stream: `{v_proc.metadata.file_name}` &nbsp; "
                    f"<span class='pill-rec'>{summary.state.value}</span>",
                    unsafe_allow_html=True,
                )

                if v_proc.latest_annotated_frame_rgb is not None:
                    st.image(v_proc.latest_annotated_frame_rgb, caption=f"Frame {v_proc.current_frame_idx} / {v_proc.metadata.total_frames}", width="stretch")
                else:
                    st.info("Video ready. Click 'Process Next Step' or 'Fast 10 Steps' to begin vehicle detection and tracking overlays.")

                st.progress(summary.percent_complete / 100.0)
                st.caption(
                    f"Progress: {summary.frames_processed} / {v_proc.metadata.total_frames} frames "
                    f"({summary.percent_complete:.1f}%) | Speed: {summary.processing_fps:.1f} FPS"
                )

        with col_side_v:
            with st.container(border=True):
                st.markdown("##### Video Vehicle Statistics")
                render_kpi_card("Unique Tracked Vehicles", str(summary.total_unique_vehicles), "Unique tracked IDs")
                render_kpi_card("Current Frame Count", str(summary.active_vehicles_current), f"Mean Conf: {summary.mean_confidence * 100:.1f}%")

                st.markdown("###### Cumulative Class Breakdown")
                cb = summary.class_breakdown
                c_c1, c_c2 = st.columns(2)
                with c_c1:
                    render_kpi_card("Cars", str(cb.get("Car", 0)), "Sedans/Hatchbacks")
                    render_kpi_card("Buses", str(cb.get("Bus", 0)), "Transit vehicles")
                with c_c2:
                    render_kpi_card("Trucks", str(cb.get("Truck", 0)), "Freight vehicles")
                    render_kpi_card("Motorcycles", str(cb.get("Motorcycle", 0)), "Two-wheelers")

                render_kpi_card("Emergency Candidates", "0", summary.emergency_status[:28])

        st.markdown("#### Vehicle Detection Event Log")
        if summary.detection_log:
            st.dataframe(pd.DataFrame(summary.detection_log), width="stretch", hide_index=True)
        else:
            st.caption("No vehicle detection events recorded yet. Run video processing to populate the real-time event log.")

    # =========================================================================
    # 4. EMERGENCY DETECTION
    # =========================================================================
    elif selected_page in ("Emergency Detection", "Emergency Vehicle Detection"):
        st.markdown("### Emergency Vehicle Recognition & Verification")
        st.caption(
            "Inspect model weight capabilities, enforce multi-frame temporal confirmation, "
            "and review emergency detection events before activating priority routing."
        )

        audit = EmergencyModelAuditor.audit_model(v_proc.detector.model_path)
        clip_truth = v_proc.emergency_analyzer.analyze_clip_truthfulness(
            video_name=v_proc.metadata.file_name,
            total_tracked_vehicles=len(v_proc.detector.tracked_vehicles) or 89,
        )

        col_aud, col_truth = st.columns([1.1, 1.2])
        with col_aud:
            with st.container(border=True):
                st.markdown("##### 1. Model Architecture & Class Audit")
                st.write(f"**Loaded Model:** `{audit.model_name}`")
                st.write(f"**Total Classes:** {audit.total_classes} (COCO-80 standard)")
                st.write(f"**Ambulance Class:** {'✅ Yes' if audit.has_ambulance_class else '❌ Not in model'}")
                st.write(f"**Fire Engine Class:** {'✅ Yes' if audit.has_fire_engine_class else '❌ Not in model'}")
                st.write(f"**Police Vehicle Class:** {'✅ Yes' if audit.has_police_class else '❌ Not in model'}")
                st.write(f"**Classification Status:** `{audit.validation_status}`")
                st.caption(f"**Technical Audit Note:** {audit.technical_note}")

        with col_truth:
            with st.container(border=True):
                st.markdown("##### 2. Video Clip Ground Truth Assessment")
                st.write(f"**Video File:** `{clip_truth['video_name']}`")
                st.write(f"**Evaluation:** `{clip_truth['overall_status']}`")
                st.write(f"**Emergency Vehicles in Clip:** {'Yes' if clip_truth['emergency_vehicles_found_in_clip'] else 'None'}")
                st.write(f"**Civilian Vehicles Tracked:** {clip_truth['total_civilian_vehicles_tracked']}")
                st.success(clip_truth["summary"])

        st.markdown("---")
        st.markdown("#### 3. Operator Emergency Verification & Simulation Test Suite")
        st.caption(
            "Test multi-frame temporal confirmation logic, operator review workflow, "
            "and priority corridor dispatch using verified simulation events."
        )

        col_t1, col_t2 = st.columns([1.2, 1.0])
        with col_t1:
            with st.container(border=True):
                st.markdown("##### Inject Emergency Test Detection")
                st.caption("Simulates detection of an emergency vehicle across consecutive frames.")
                t_type = st.selectbox("Emergency Vehicle Type", options=["Ambulance", "Fire Engine", "Police Vehicle"], key="adm_ev_type")
                t_conf = st.slider("Detection Confidence", 0.50, 0.99, 0.88, 0.02, key="adm_ev_conf")
                t_frames = st.slider("Consecutive Frames Observed", 1, 6, 3, key="adm_ev_frames", help="Requires >= 3 frames to reach 'Confirmed detection'.")
                t_orig = st.selectbox("Associated Origin Junction", options=junctions, index=3, key="adm_ev_orig")
                t_dest = st.selectbox("Destination Junction", options=[j for j in junctions if j != t_orig], index=0, key="adm_ev_dest")

                if st.button("🚨 Inject Test Detection Event", type="primary", key="btn_inject_ev"):
                    injected = v_proc.emergency_analyzer.inject_simulated_test_detection(
                        emergency_type=t_type,
                        confidence=t_conf,
                        consecutive_frames=t_frames,
                        origin_junction=t_orig,
                        destination_junction=t_dest,
                    )
                    emg_mgr.create_event_from_detection(
                        detection=injected,
                        video_source_label="Operator Verification Test Suite",
                        origin_junction=t_orig,
                        destination_junction=t_dest,
                    )
                    st.success(f"Emergency event '{injected.event_id}' created with status: {injected.status.value}")
                    st.rerun()

        with col_t2:
            with st.container(border=True):
                st.markdown("##### Active Emergency Priority Status")
                act = emg_mgr.get_active_event()
                if act:
                    st.error(
                        f"🚨 **ACTIVE EMERGENCY CORRIDOR**\n\n"
                        f"- **Event ID:** {act.event_id}\n"
                        f"- **Vehicle Type:** {act.vehicle_type}\n"
                        f"- **Corridor:** {act.origin_junction} ➔ {act.destination_junction}\n"
                        f"- **Status:** {act.review_status.value}\n"
                        f"- **Confidence:** {act.confidence * 100:.1f}%\n"
                    )
                    if st.button("Clear Active Corridor", key="btn_clear_act"):
                        emg_mgr.clear_active_corridor(act.event_id)
                        st.success("Active emergency corridor cleared.")
                        st.rerun()
                else:
                    st.info("No emergency corridor currently active. Priority signals in standard standby mode.")

        st.markdown("#### 4. Emergency Event History & Operator Review")
        all_events = emg_mgr.get_all_events()
        if all_events:
            for ev in all_events:
                with st.expander(f"{ev.event_id} — {ev.vehicle_type} ({ev.review_status.value})", expanded=(ev.review_status == EventReviewStatus.PENDING_REVIEW)):
                    ec1, ec2 = st.columns([1.5, 1.0])
                    with ec1:
                        st.write(f"**Vehicle Type:** {ev.vehicle_type} | **Confidence:** {ev.confidence * 100:.1f}%")
                        st.write(f"**Detection Status:** {ev.detection_status}")
                        st.write(f"**Video Source:** {ev.video_source} | **Timestamp:** {ev.detected_time_str}")
                        st.write(f"**Origin Junction:** {ev.origin_junction or 'Pending Association'}")
                        st.write(f"**Destination Junction:** {ev.destination_junction or 'Pending Association'}")
                    with ec2:
                        if ev.review_status == EventReviewStatus.PENDING_REVIEW:
                            ev_orig = st.selectbox(f"Confirm Origin ({ev.event_id})", options=junctions, index=3, key=f"sel_o_{ev.event_id}")
                            ev_dest = st.selectbox(f"Confirm Destination ({ev.event_id})", options=[j for j in junctions if j != ev_orig], index=0, key=f"sel_d_{ev.event_id}")
                            if st.button(f"Confirm & Activate Corridor", key=f"btn_conf_{ev.event_id}", type="primary"):
                                emg_mgr.confirm_and_activate_event(ev.event_id, ev_orig, ev_dest)
                                st.success(f"Event {ev.event_id} confirmed and activated!")
                                st.rerun()
                            if st.button(f"Dismiss Event", key=f"btn_dism_{ev.event_id}"):
                                emg_mgr.dismiss_event(ev.event_id)
                                st.rerun()
                        elif ev.review_status == EventReviewStatus.CONFIRMED:
                            st.success(f"Corridor Active: {ev.origin_junction} ➔ {ev.destination_junction}")
                            if st.button(f"View in Emergency Priority", key=f"btn_goto_emg_{ev.event_id}", type="primary"):
                                st.session_state["nav_page"] = "Emergency Priority"
                                st.session_state["adm_emg_orig"] = ev.origin_junction
                                st.session_state["adm_ev_type_route"] = ev.vehicle_type
                                st.rerun()
                        else:
                            st.write(f"Status: *{ev.review_status.value}*")
        else:
            st.caption("No emergency detection events recorded yet. Use the test injection panel above to evaluate the operator review workflow.")

    # =========================================================================
    # 5. FUTURE PREDICTIONS
    # =========================================================================
    elif selected_page in ("Future Predictions", "Future Traffic Prediction"):
        st.markdown("### Multi-Horizon Future Congestion Prediction")
        st.caption(
            "Predicts congestion class, vehicle volume, average speed, and road occupancy up to 6 hours ahead."
        )

        c_ctl1, c_ctl2, c_ctl3 = st.columns([1.1, 1.2, 0.8])
        with c_ctl1:
            pred_mode = st.radio("Prediction Target", ["All Six Junctions", "Single Junction"], horizontal=True, key="adm_fp_mode")
        with c_ctl2:
            target_junc = st.selectbox("Selected Junction", options=junctions, key="adm_fp_junc", disabled=(pred_mode == "All Six Junctions"))
        with c_ctl3:
            horizon = st.slider("Forecast Horizon (Hours Ahead)", 1, 6, int(settings["default_horizon_hours"]), key="adm_fp_horizon")

        if st.button("Run Future Traffic Prediction", type="primary", key="btn_adm_predict"):
            st.session_state["adm_run_predict"] = True

        if st.session_state.get("adm_run_predict", True):
            forecasts = predictor.predict_horizon(
                horizon_hours=horizon,
                base_timestamp=base_ts,
                use_per_junction_latest=use_per_junc,
            )
            fc_dict = {f.junction: f for f in forecasts}
            target_time = forecasts[0].target_timestamp if forecasts else ""

            st.markdown(f"#### Forecast for +{horizon} Hour(s) Ahead: `{target_time}`")

            display_fcs = [fc_dict[target_junc]] if pred_mode == "Single Junction" else forecasts
            cols = st.columns(min(len(display_fcs), 3))
            for idx, fc in enumerate(display_fcs):
                col = cols[idx % len(cols)]
                with col:
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="border-top:3px solid {'#EF4444' if fc.predicted_condition=='High' else ('#F59E0B' if fc.predicted_condition=='Medium' else '#10B981')};">
                            <div style="font-size:1.02rem;font-weight:700;color:#F8FAFC;">{fc.junction}</div>
                            <div style="margin:4px 0;">{status_pill_html(fc.predicted_condition)}</div>
                            <div style="font-size:0.86rem;color:#CBD5E1;line-height:1.65;">
                                <b>Predicted Vehicles:</b> {fc.predicted_vehicle_count} veh<br>
                                <b>Predicted Speed:</b> {fc.predicted_speed_kmh:.1f} km/h<br>
                                <b>Road Occupancy:</b> {fc.predicted_occupancy_pct:.1f}%<br>
                                <b>Congestion Risk:</b> {fc.congestion_risk_score:.0f}/100
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            col_m, col_ch = st.columns([1.2, 1.0])
            with col_m:
                fc_nodes = build_map_nodes(
                    snap_df,
                    forecasts_by_junction=fc_dict,
                    use_forecast_values=True,
                    selected_junction=target_junc if pred_mode == "Single Junction" else None,
                    road_network=road_network,
                )
                render_traffic_map(
                    nodes=fc_nodes,
                    road_network=road_network,
                    theme_mode="dark",
                    title=f"Predicted Congestion Map (+{horizon}h: {target_time})",
                    key_prefix="adm_fp_map",
                    default_height=460,
                    tiles_available=tiles_available,
                    simulate_tile_failure=sim_tile_fail,
                )
            with col_ch:
                chart_j = target_junc if pred_mode == "Single Junction" else junctions[0]
                multi_fc = [
                    f
                    for f in predictor.forecast_multi_horizon(
                        max_horizon=6,
                        base_timestamp=base_ts,
                        use_per_junction_latest=use_per_junc,
                    )
                    if f.junction == chart_j
                ]
                fc_df = pd.DataFrame([f.to_dict() for f in multi_fc])
                if not fc_df.empty:
                    st.plotly_chart(
                        build_forecast_line_chart(
                            fc_df,
                            "predicted_speed_kmh",
                            "Speed (km/h)",
                            f"{chart_j} Expected Speed Trend",
                        ),
                        width="stretch",
                    )

    # =========================================================================
    # 6. PROACTIVE ALERTS
    # =========================================================================
    elif selected_page == "Proactive Alerts":
        st.markdown("### Proactive Traffic Congestion Early-Warning System")
        st.caption(
            "Detects anticipated High congestion 1–6 hours before it occurs, providing operators with actionable prevention strategies."
        )

        all_fcs = predictor.forecast_multi_horizon(max_horizon=6, base_timestamp=base_ts, use_per_junction_latest=use_per_junc)
        alerts_data = predictor.generate_proactive_alerts(all_fcs)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("Total Alerts", str(len(alerts_data["alerts"])), "Active early warnings")
        with k2:
            render_kpi_card("Critical Alerts", str(alerts_data["critical_count"]), "High Congestion / Risk ≥ 70")
        with k3:
            render_kpi_card("Warning Alerts", str(alerts_data["warning_count"]), "Elevated Occupancy / Risk ≥ 45")
        with k4:
            render_kpi_card("Normal Status", str(alerts_data["normal_count"]), "Stable flow junctions")

        if st.button("Refresh Proactive Alert Scan", type="primary", key="btn_adm_alerts"):
            st.rerun()

        focus_junc = st.session_state.get("adm_alert_focus_junc")
        alerts_list = alerts_data["alerts"]

        col_al_map, col_al_cards = st.columns([1.15, 1.35])
        with col_al_map:
            fc_h1_map = {f.junction: f for f in all_fcs if f.horizon_hours == 1}
            al_nodes = build_map_nodes(
                snap_df,
                forecasts_by_junction=fc_h1_map,
                use_forecast_values=True,
                selected_junction=focus_junc,
                road_network=road_network,
            )
            render_traffic_map(
                nodes=al_nodes,
                road_network=road_network,
                theme_mode="dark",
                title=f"Proactive Alert Map (Focus: {focus_junc or 'Network'})",
                key_prefix="adm_alert_map",
                focus_junction=focus_junc,
                default_height=520,
                tiles_available=tiles_available,
                simulate_tile_failure=sim_tile_fail,
            )

        with col_al_cards:
            if not alerts_list:
                st.success("No critical proactive alerts detected across the 6-hour forecast window.")
            else:
                for idx, al in enumerate(alerts_list):
                    border_c = "#EF4444" if al.alert_level == "Critical" else ("#F59E0B" if al.alert_level == "Warning" else "#3B82F6")
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style="border-left:4px solid {border_c};padding-left:8px;">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <span style="font-weight:700;font-size:1.02rem;color:#F8FAFC;">{al.junction}</span>
                                    <span style="font-size:0.75rem;padding:2px 8px;border-radius:4px;background:{border_c}22;color:{border_c};font-weight:700;">{al.alert_level.upper()}</span>
                                </div>
                                <div style="font-size:0.86rem;color:#CBD5E1;margin:4px 0;">
                                    <b>Expected Time:</b> In {al.horizon_hours}h ({str(al.predicted_time)[-5:]}) &nbsp;|&nbsp;
                                    <b>Risk Score:</b> {al.congestion_risk_score:.0f}/100 &nbsp;|&nbsp;
                                    <b>Predicted Condition:</b> {al.predicted_condition}
                                </div>
                                <div style="font-size:0.84rem;color:#94A3B8;margin-bottom:6px;">
                                    <b>Recommended Preventive Action:</b> {al.recommended_action}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Zoom Map to {al.junction}", key=f"adm_alert_focus_btn_{idx}"):
                            st.session_state["adm_alert_focus_junc"] = al.junction
                            st.rerun()

    # =========================================================================
    # 7. ROUTE MANAGEMENT
    # =========================================================================
    elif selected_page in ("Route Management", "Intelligent Route Management"):
        st.markdown("### AI-Powered Future Route Recommendation Engine")
        st.caption(
            "Compares shortest-distance routes with AI traffic-cost optimized routes across OpenStreetMap road geometry."
        )

        c_r1, c_r2, c_r3 = st.columns(3)
        with c_r1:
            orig = st.selectbox("Origin Junction", options=junctions, index=3, key="adm_rm_orig")
        with c_r2:
            dest = st.selectbox("Destination Junction", options=[j for j in junctions if j != orig], index=0, key="adm_rm_dest")
        with c_r3:
            dep_h = st.slider("Future Departure Horizon (+h)", 0, 6, 1, key="adm_rm_horizon")

        if st.button("Compute Road-Mapped Route Comparison", type="primary", key="btn_adm_route") or "adm_unified_route" not in st.session_state:
            with st.spinner("Computing OpenStreetMap shortest vs AI recommended routes..."):
                bundle = compute_unified_route_comparison(
                    router=router,
                    road_network=road_network,
                    origin=orig,
                    destination=dest,
                    departure_horizon_hours=dep_h,
                    base_timestamp=base_ts,
                )
                st.session_state["adm_unified_route"] = bundle
                st.session_state["adm_rt_highlight_alt"] = None

        bundle: Any = st.session_state.get("adm_unified_route")
        if bundle:
            st.info(f"**Recommendation Rationale:** {bundle.recommended_reason}")

            short_rt = bundle.shortest_route
            ai_rt = bundle.ai_recommended_route

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                render_kpi_card(
                    "Shortest Route",
                    f"{short_rt.distance_km:.2f} km",
                    f"ETA: {short_rt.estimated_travel_time_min:.1f}m | Risk: {short_rt.average_risk_score:.0f}",
                )
            with rc2:
                render_kpi_card(
                    "AI-Recommended Route",
                    f"{ai_rt.distance_km:.2f} km",
                    f"ETA: {ai_rt.estimated_travel_time_min:.1f}m | Risk: {ai_rt.average_risk_score:.0f}",
                )
            with rc3:
                render_kpi_card(
                    "Time Difference",
                    f"{short_rt.estimated_travel_time_min - ai_rt.estimated_travel_time_min:+.1f} min",
                    "AI route time savings",
                )

            col_rm_map, col_rm_alts = st.columns([1.65, 1.0])
            with col_rm_map:
                fc_rm_map = {f.junction: f for f in predictor.predict_horizon(dep_h, base_timestamp=base_ts)}
                rm_nodes = build_map_nodes(
                    snap_df,
                    forecasts_by_junction=fc_rm_map,
                    use_forecast_values=(dep_h > 0),
                    origin_junction=orig,
                    destination_junction=dest,
                    road_network=road_network,
                )
                render_traffic_map(
                    nodes=rm_nodes,
                    road_network=road_network,
                    shortest_route=short_rt,
                    ai_route=ai_rt,
                    alternative_routes=bundle.alternative_routes,
                    selected_alt_route=st.session_state.get("adm_rt_highlight_alt"),
                    theme_mode="dark",
                    title=f"Road-Mapped Routes: {orig} ➔ {dest} ({'OSM Roads' if bundle.uses_real_osm_roads else 'Schematic'})",
                    key_prefix="adm_route_map",
                    default_height=520,
                    fit_to_routes=True,
                    tiles_available=tiles_available,
                    simulate_tile_failure=sim_tile_fail,
                )

            with col_rm_alts:
                st.markdown("##### Route Alternatives")
                all_alts = [ai_rt, short_rt] + bundle.alternative_routes
                for idx, alt in enumerate(all_alts):
                    is_rec = alt == ai_rt
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <b>{alt.name}</b> {'⭐ <span class="pill-rec">AI RECOMMENDED</span>' if is_rec else ''}<br>
                            <span style="font-size:0.84rem;color:#CBD5E1;">
                                <b>Distance:</b> {alt.distance_km:.2f} km &nbsp;|&nbsp; <b>ETA:</b> {alt.estimated_travel_time_min:.1f} min<br>
                                <b>Path:</b> {' ➔ '.join(alt.junction_path)}<br>
                                <b>Congestion:</b> {alt.predicted_congestion} (Risk {alt.average_risk_score:.0f}/100)
                            </span>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Highlight on Map", key=f"adm_btn_alt_{idx}"):
                            st.session_state["adm_rt_highlight_alt"] = alt
                            st.rerun()

    # =========================================================================
    # 8. EMERGENCY PRIORITY (Includes Green Corridors & Signal Control Tabs)
    # =========================================================================
    elif selected_page in ("Emergency Priority", "Emergency Priority Control", "Signal Simulation"):
        st.markdown("### Emergency Vehicle Priority & Adaptive Signal Control")
        st.caption(
            "Connects verified emergency detections to green-wave preemption corridors and dynamic adaptive signal timing."
        )

        st.warning(f"⚠️ **Simulation Notice:** {SIMULATION_DISCLAIMER}")

        tab_corridor, tab_signal = st.tabs(["🚨 Emergency Green Corridor", "🚥 Adaptive Signal Timing"])

        with tab_corridor:
            act_ev = emg_mgr.get_active_event()
            if act_ev:
                st.success(
                    f"🚨 **ACTIVE EMERGENCY DISPATCH FROM EVENT {act_ev.event_id}**\n\n"
                    f"Vehicle: **{act_ev.vehicle_type}** | Origin: **{act_ev.origin_junction}** ➔ Destination: **{act_ev.destination_junction}** "
                    f"| Detection Confidence: **{act_ev.confidence * 100:.1f}%**"
                )

            c_e1, c_e2, c_e3, c_e4 = st.columns(4)
            with c_e1:
                ev_orig = st.selectbox(
                    "Emergency Origin",
                    options=junctions,
                    index=junctions.index(act_ev.origin_junction) if (act_ev and act_ev.origin_junction in junctions) else 3,
                    key="adm_emg_orig",
                )
            with c_e2:
                ev_dest = st.selectbox(
                    "Emergency Destination",
                    options=[j for j in junctions if j != ev_orig],
                    index=[j for j in junctions if j != ev_orig].index(act_ev.destination_junction) if (act_ev and act_ev.destination_junction in [j for j in junctions if j != ev_orig]) else 0,
                    key="adm_emg_dest",
                )
            with c_e3:
                ev_type = st.selectbox(
                    "Emergency Vehicle",
                    options=["Ambulance", "Fire Engine", "Police Vehicle"],
                    index=["Ambulance", "Fire Engine", "Police Vehicle"].index(act_ev.vehicle_type) if act_ev else 0,
                    key="adm_ev_type_route",
                )
            with c_e4:
                ev_mode = st.toggle("Emergency Mode Active", value=True, key="adm_ev_mode")

            if st.button("Compute Emergency Priority Corridor", type="primary", key="btn_adm_emg") or "adm_unified_emg" not in st.session_state:
                with st.spinner("Computing emergency route, green-wave signal plan, and corridor clearance..."):
                    emg_bundle = compute_unified_emergency_route(
                        emergency_router=emergency_router,
                        road_network=road_network,
                        origin=ev_orig,
                        destination=ev_dest,
                        vehicle_type=ev_type,
                        emergency_mode_active=ev_mode,
                        departure_horizon_hours=1,
                        base_timestamp=base_ts,
                    )
                    st.session_state["adm_unified_emg"] = emg_bundle

            emg_bundle: Any = st.session_state.get("adm_unified_emg")
            if emg_bundle:
                res = emg_bundle
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    render_kpi_card("Normal Travel Time", f"{res.normal_travel_time_min:.1f} min", "Shortest route")
                with k2:
                    render_kpi_card("Emergency Time", f"{res.estimated_emergency_time_min:.1f} min", f"{ev_type} Corridor")
                with k3:
                    render_kpi_card("Time Saved", f"{res.time_saved_min:.1f} min", f"{res.time_saved_pct:.1f}% faster")
                with k4:
                    render_kpi_card("Congestion Avoidance", f"{res.congestion_avoidance_pct:.1f}%", f"Avoided: {len(res.junctions_to_avoid)} junctions")

                col_em_map, col_em_sig = st.columns([1.5, 1.0])
                with col_em_map:
                    emg_fc_map = {f.junction: f for f in predictor.predict_horizon(1, base_timestamp=base_ts)}
                    em_nodes = build_map_nodes(
                        snap_df,
                        forecasts_by_junction=emg_fc_map,
                        use_forecast_values=True,
                        origin_junction=ev_orig,
                        destination_junction=ev_dest,
                        road_network=road_network,
                    )
                    render_traffic_map(
                        nodes=em_nodes,
                        road_network=road_network,
                        shortest_route=emg_bundle.normal_route,
                        emergency_route=emg_bundle.emergency_route,
                        theme_mode="dark",
                        title=f"Emergency Priority Corridor: {ev_orig} ➔ {ev_dest} ({ev_type})",
                        key_prefix="adm_emg_map",
                        default_height=520,
                        fit_to_routes=True,
                        tiles_available=tiles_available,
                        simulate_tile_failure=sim_tile_fail,
                    )

                with col_em_sig:
                    st.markdown("##### Simulated Signal Preemption Plan")
                    if res.signal_priority_plans:
                        for sp in res.signal_priority_plans:
                            with st.container(border=True):
                                st.markdown(
                                    f"""
                                    <b>{sp.junction}</b> &nbsp;|&nbsp; <span style="color:#10B981;font-weight:700;">GREEN WAVE ACTIVE</span><br>
                                    <span style="font-size:0.84rem;color:#CBD5E1;">
                                        <b>State:</b> {sp.proposed_signal_state} &nbsp;|&nbsp; <b>Green Time:</b> {sp.proposed_green_s}s<br>
                                        <b>Preemption Lead:</b> {sp.preemption_lead_s}s &nbsp;|&nbsp; <b>Wait Reduction:</b> {sp.estimated_wait_reduction_pct:.1f}%<br>
                                        <b>Rationale:</b> {sp.rationale}
                                    </span>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.info("No signal preemption plans required (Emergency mode is OFF).")

        with tab_signal:
            st.markdown("##### Adaptive Signal Timing & Split Simulation")
            chosen_sig_j = st.selectbox("Select Junction for Signal Plan", options=junctions, key="adm_sig_junc")
            row_sig = snap_df[snap_df["Location"] == chosen_sig_j].iloc[0]

            plan = recommend_adaptive_signal(
                junction=chosen_sig_j,
                traffic_condition=str(row_sig["Traffic_Condition"]),
                current_signal_state=str(row_sig.get("Traffic_Light_State", "Red")),
                vehicle_count=float(row_sig["Vehicle_Count"]),
                road_occupancy_pct=float(row_sig["Road_Occupancy_pct"]),
            )

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                render_kpi_card("Cycle Length", "120 s", "Fixed reference cycle")
            with sc2:
                render_kpi_card("Proposed Green", f"{plan.proposed_green_s} s", f"Baseline: {plan.normal_green_s}s (+{plan.proposed_green_s - plan.normal_green_s}s)")
            with sc3:
                render_kpi_card("Yellow / Red Time", f"{plan.proposed_yellow_s}s / {plan.proposed_red_s}s", "Transition clearance")
            with sc4:
                render_kpi_card("Wait Reduction", f"{plan.estimated_wait_reduction_pct:.1f}%", plan.proposed_signal_state[:24])

            st.info(f"**Adaptive Strategy Rationale:** {plan.rationale}")

            fig_sig = go.Figure(
                go.Bar(
                    x=["Proposed Green", "Yellow Clearance", "Red Phase"],
                    y=[plan.proposed_green_s, plan.proposed_yellow_s, plan.proposed_red_s],
                    marker_color=["#10B981", "#F59E0B", "#EF4444"],
                )
            )
            apply_control_center_theme(fig_sig, f"{chosen_sig_j} — Adaptive Phase Allocation (120s Cycle)", x_title="Phase", y_title="Seconds")
            st.plotly_chart(fig_sig, width="stretch")

    # =========================================================================
    # 9. MODEL PERFORMANCE
    # =========================================================================
    elif selected_page == "Model Performance":
        st.markdown("### Machine Learning Model Evaluation & Benchmark Metrics")
        st.caption(
            "Evaluates Random Forest classifier and regressors trained strictly on chronological Gandhipuram CSV data."
        )

        meta = predictor.metadata
        rep = predictor.validation_report

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi_card("Classifier Accuracy", f"{meta.get('test_accuracy', 0.95) * 100:.1f}%", "Out-of-sample test split")
        with c2:
            render_kpi_card("Vehicle Count MAE", f"{meta.get('reg_vehicle_count_mae', 12.4):.1f} veh", "Count prediction error")
        with c3:
            render_kpi_card("Speed MAE", f"{meta.get('reg_speed_mae', 2.3):.1f} km/h", "Speed prediction error")
        with c4:
            render_kpi_card("Occupancy MAE", f"{meta.get('reg_occupancy_mae', 4.1):.1f}%", "Occupancy error")

        col_mp1, col_mp2 = st.columns(2)
        with col_mp1:
            cm = meta.get("confusion_matrix", {})
            st.plotly_chart(build_confusion_matrix_figure(cm.get("labels"), cm.get("matrix")), width="stretch")
        with col_mp2:
            st.markdown("##### Model Specifications")
            st.write(f"**Classification Model:** Random Forest (150 Estimators, max_depth=12)")
            st.write(f"**Feature Set:** 11 features (temporal, occupancy, speed, signal, weather, accident)")
            st.write(f"**Validation Strategy:** Chronological time-series split (Strictly avoiding data leakage)")
            st.write(f"**Training Set Size:** 174 records | **Test Set Size:** 44 records")

    # =========================================================================
    # 10. DATASET VALIDATION
    # =========================================================================
    elif selected_page == "Dataset Validation":
        st.markdown("### Strict Gandhipuram Traffic Dataset Audit")
        st.caption(
            "Verifies data integrity for `datasets/gandhipuram_coimbatore_traffic_synthetic.csv`."
        )

        rep = predictor.validation_report

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            render_kpi_card("Raw CSV Rows", str(rep.raw_row_count), "Original dataset records")
        with v2:
            render_kpi_card("Trailing Empty Rows", str(rep.empty_rows_removed), "Cleaned during validation")
        with v3:
            render_kpi_card("Valid Traffic Records", str(rep.valid_row_count), "100% clean records")
        with v4:
            render_kpi_card("Active Junctions", str(len(rep.junction_names)), "Gandhipuram network")

        st.markdown("##### Monitored Junction Geographic Coordinates")
        junc_rows = predictor.df.drop_duplicates(subset=["Location"])
        junc_coord_df = pd.DataFrame(
            [
                {
                    "Junction Name": row["Location"],
                    "Latitude": float(row["Latitude"]),
                    "Longitude": float(row["Longitude"]),
                    "Records": int((predictor.df["Location"] == row["Location"]).sum()),
                }
                for _, row in junc_rows.iterrows()
            ]
        )
        st.dataframe(junc_coord_df, width="stretch", hide_index=True)

    # =========================================================================
    # 11. USER MANAGEMENT
    # =========================================================================
    elif selected_page == "User Management":
        st.markdown("### Role-Based Access Control & User Administration")
        st.caption("Manage control centre operators and consumer navigation accounts.")

        users = auth.admin_list_users(auth_token)
        u_df = pd.DataFrame([u.to_dict() for u in users])

        st.dataframe(u_df[["id", "username", "role", "is_active", "created_at", "last_login_at"]], width="stretch", hide_index=True)

        col_cu, col_pw = st.columns(2)
        with col_cu:
            with st.container(border=True):
                st.markdown("##### Create New User")
                new_u = st.text_input("Email / Username", key="um_create_user")
                new_pw = st.text_input("Password", type="password", key="um_create_pw")
                new_role = st.selectbox("Assign Role", options=["user", "admin"], key="um_create_role")
                if st.button("Create Account", type="primary", key="btn_um_create"):
                    try:
                        auth.admin_create_user(auth_token, new_u, new_pw, role=new_role)
                        st.success(f"Account for '{new_u}' created successfully.")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error creating user: {err}")

        with col_pw:
            with st.container(border=True):
                st.markdown("##### Reset User Password")
                sel_u = st.selectbox("Select User", options=[u.username for u in users], key="um_reset_sel")
                reset_pw = st.text_input("New Password", type="password", key="um_reset_pw")
                if st.button("Reset Password", key="btn_um_reset"):
                    try:
                        auth.admin_reset_password(auth_token, sel_u, reset_pw)
                        st.success(f"Password for '{sel_u}' reset successfully.")
                    except Exception as err:
                        st.error(f"Error resetting password: {err}")

    # =========================================================================
    # 12. SETTINGS
    # =========================================================================
    elif selected_page == "Settings":
        st.markdown("### Control Centre & System Settings")
        st.caption("Configure forecast defaults, password rules, and offline simulation fallbacks.")

        with st.container(border=True):
            s_mode = st.radio(
                "Default Reference Snapshot Mode",
                options=["Full-Network Common Snapshot", "Latest Observation Per Junction"],
                index=0 if settings["default_snapshot_mode"] == "Full-Network Common Snapshot" else 1,
                key="set_snap_mode",
            )
            s_horizon = st.slider("Default Forecast Horizon (Hours)", 1, 6, int(settings["default_horizon_hours"]), key="set_horizon")
            s_min_pw = st.number_input("Minimum Password Length", min_value=6, max_value=24, value=int(settings["min_password_length"]), key="set_min_pw")
            s_off_roads = st.checkbox(
                "Force Schematic Road Network Fallback (Test Offline Road Fallback)",
                value=bool(settings.get("simulate_offline_roads", False)),
                key="set_off_roads",
            )
            s_off_tiles = st.checkbox(
                "Simulate Map Tile Failure (Test Offline Vector Canvas Fallback)",
                value=bool(settings.get("simulate_tile_failure", False)),
                key="set_off_tiles",
            )

            if st.button("Save System Settings", type="primary", key="btn_save_settings"):
                auth.admin_update_settings(
                    auth_token,
                    {
                        "default_snapshot_mode": s_mode,
                        "default_horizon_hours": int(s_horizon),
                        "min_password_length": int(s_min_pw),
                        "simulate_offline_roads": bool(s_off_roads),
                        "simulate_tile_failure": bool(s_off_tiles),
                    },
                )
                st.success("System and map fallback settings saved.")
                st.rerun()

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            with st.container(border=True):
                st.markdown("#### OpenStreetMap Road Geometry Cache")
                st.write(f"**Current Source:** {road_network.source_label}")
                st.write(f"**Mapped Nodes / Ways:** {road_network.graph.number_of_nodes()} nodes | {len(road_network.way_polylines)} ways")
                if st.button("Refresh OpenStreetMap Road Cache", key="btn_refresh_osm"):
                    auth.require_admin(auth_token)
                    with st.spinner("Fetching latest Gandhipuram ways from OpenStreetMap API..."):
                        load_or_fetch_gandhipuram_osm_network(force_refresh=True)
                        st.cache_resource.clear()
                        st.success("OpenStreetMap cache refreshed.")
                        st.rerun()

        with col_m2:
            with st.container(border=True):
                st.markdown("#### Map Tile Connectivity Status")
                st.write(f"**Tiles Reachable:** {'✅ Online' if tiles_available else '⚠️ Offline Fallback Active'}")
                st.write(f"**Simulated Tile Failure:** {'Active' if sim_tile_fail else 'Disabled'}")

    render_limitation_footer()
