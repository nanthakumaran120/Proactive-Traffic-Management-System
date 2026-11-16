"""AI-Powered Proactive Traffic Management & Intelligent Route Recommendation Application."""

from __future__ import annotations

import os
import streamlit as st

from src.auth import ADMIN_PAGES, USER_PAGES, AuthManager, can_access_page
from src.emergency_routing import EmergencyPriorityRouter
from src.future_predictor import FutureTrafficPredictor
from src.future_routing import FutureRouteRecommender
from src.road_network import (
    OSMRoadNetwork,
    check_map_tile_connectivity,
    load_or_fetch_gandhipuram_osm_network,
)
from src.ui_admin import render_admin_dashboard
from src.ui_login import render_login_page
from src.ui_theme import inject_theme
from src.ui_user import render_user_dashboard


st.set_page_config(
    page_title="AI-Powered Proactive Traffic Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_traffic_engines() -> tuple[FutureTrafficPredictor, FutureRouteRecommender, EmergencyPriorityRouter]:
    predictor = FutureTrafficPredictor(auto_train=True)
    router = FutureRouteRecommender(predictor)
    emergency_router = EmergencyPriorityRouter(predictor, router)
    return predictor, router, emergency_router


@st.cache_resource
def load_road_network(force_offline: bool = False) -> OSMRoadNetwork:
    return load_or_fetch_gandhipuram_osm_network(force_offline=force_offline)


@st.cache_data(ttl=120)
def probe_tile_status() -> dict[str, bool]:
    return check_map_tile_connectivity(timeout_sec=2.5)


def get_auth_manager() -> AuthManager:
    db_override = os.environ.get("TRAFFIC_AUTH_DB_PATH")
    return AuthManager(db_path=db_override) if db_override else AuthManager()


def main() -> None:
    auth = get_auth_manager()

    # Automation hook for headless screenshot capture and verified local inspection
    dev_role = os.environ.get("TRAFFIC_DEV_ROLE")
    if dev_role and "auth_token" not in st.session_state:
        try:
            if dev_role == "admin":
                _, tok = auth.authenticate("nanthakumaran120@gmail.com", "123456789")
                st.session_state["auth_token"] = tok
                st.session_state["auth_role"] = "admin"
            elif dev_role == "user":
                _, tok = auth.authenticate("nantha@gmail.com", "123456789")
                st.session_state["auth_token"] = tok
                st.session_state["auth_role"] = "user"
        except Exception:
            pass

    token = st.session_state.get("auth_token")
    current_user = auth.verify_session(token)

    if current_user is None or token is None:
        inject_theme("dark")
        st.session_state.pop("auth_token", None)
        st.session_state.pop("auth_role", None)
        render_login_page(auth)
        return

    # Admin/login retain the dark control-centre theme; users inherit Streamlit's selected theme.
    inject_theme("streamlit" if current_user.role == "user" else "dark")

    # Prevent direct URL / session tampering from opening unauthorized pages
    qp_page = st.query_params.get("page")
    if isinstance(qp_page, list):
        qp_page = qp_page[0] if qp_page else None

    default_home = "Overview" if current_user.role == "admin" else "Home"
    last_qp = st.session_state.get("_last_qp_page")
    if qp_page and qp_page != last_qp:
        candidate_page = str(qp_page)
    else:
        candidate_page = str(st.session_state.get("nav_page") or qp_page or default_home)

    if not can_access_page(current_user.role, candidate_page):
        from src.auth import ADMIN_PAGE_ALIASES, USER_PAGE_ALIASES
        canon_admin = ADMIN_PAGE_ALIASES.get(candidate_page, candidate_page)
        canon_user = USER_PAGE_ALIASES.get(candidate_page, candidate_page)
        if (
            candidate_page in ADMIN_PAGES
            or canon_admin in ADMIN_PAGES
            or candidate_page in USER_PAGES
            or canon_user in USER_PAGES
        ):
            st.error(
                f"Access Denied: Your role ({current_user.role.upper()}) is not authorized to access '{candidate_page}'."
            )
        candidate_page = default_home
        st.session_state["nav_page"] = candidate_page
        st.session_state.pop("admin_sidebar_nav", None)
        st.query_params["page"] = candidate_page
    else:
        st.session_state["nav_page"] = candidate_page

    st.session_state["_last_qp_page"] = st.query_params.get("page")

    settings = auth.get_settings()
    predictor, router, emergency_router = load_traffic_engines()
    road_network = load_road_network(force_offline=bool(settings.get("simulate_offline_roads", False)))
    tile_status = probe_tile_status()
    tiles_available = bool(tile_status.get("any_tiles_available", True))

    if current_user.role == "admin":
        render_admin_dashboard(
            auth=auth,
            auth_token=token,
            user=current_user,
            predictor=predictor,
            router=router,
            emergency_router=emergency_router,
            road_network=road_network,
            tiles_available=tiles_available,
            requested_page=str(candidate_page),
        )
    else:
        render_user_dashboard(
            auth=auth,
            auth_token=token,
            user=current_user,
            predictor=predictor,
            router=router,
            road_network=road_network,
            tiles_available=tiles_available,
            simulate_tile_failure=bool(settings.get("simulate_tile_failure", False)),
            requested_page=str(candidate_page),
        )


if __name__ == "__main__":
    main()
