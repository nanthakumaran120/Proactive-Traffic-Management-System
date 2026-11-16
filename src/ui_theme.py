"""Dual-mode commercial enterprise design system.

- Dark Control Centre for Admin Control Room and Login
- Clean, high-contrast Light Navigation for Normal Users / Travelers
"""

from __future__ import annotations

import html
import streamlit as st


CONDITION_HEX = {
    "Low": "#10B981",
    "Medium": "#F59E0B",
    "High": "#EF4444",
}

CONDITION_FRIENDLY_LABEL = {
    "Low": "Low Traffic (Smooth Flow)",
    "Medium": "Moderate Traffic (Normal Flow)",
    "High": "Heavy Traffic (Congestion)",
}

DARK_CSS = """
<style>
    /* Full viewport utilization */
    .stApp {
        background-color: #0B1120;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2.0rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1800px !important;
        width: 100% !important;
    }

    /* Clean Enterprise Banners */
    .brand-banner {
        background: #111827;
        border: 1px solid #1F2937;
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 16px;
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 4px;
        line-height: 1.4;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 3px;
        letter-spacing: -0.01em;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 3px;
    }

    /* Junction & Inspector Panels */
    .junction-card {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .detail-panel {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-top: 3px solid #38BDF8;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    /* Status Badges */
    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-low { background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .pill-medium { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .pill-high { background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.35); }
    .pill-rec { background-color: #059669; color: #FFFFFF; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; }

    /* Alert Cards */
    .alert-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        background-color: #111827;
        border: 1px solid #1E293B;
    }
    .alert-critical { border-left: 4px solid #EF4444; }
    .alert-warning { border-left: 4px solid #F59E0B; }
    .alert-normal { border-left: 4px solid #10B981; }

    /* Button Polish */
    .stButton > button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.12s ease-in-out !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #1D4ED8 !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
        border: 1px solid #334155 !important;
    }

    /* Badges */
    .role-badge-admin {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .role-badge-user {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
    }
</style>
"""

LIGHT_CSS = """
<style>
    /* Full viewport utilization */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    header[data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.95);
    }
    .block-container {
        padding-top: 0.85rem !important;
        padding-bottom: 2.0rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1800px !important;
        width: 100% !important;
    }

    /* Top Navigation Header Bar */
    .nav-header-bar {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* High-contrast User Navigation Buttons */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background-color: #F8FAFC !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.75rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.12s ease-in-out !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background-color: #EDF2F7 !important;
        color: #0F172A !important;
        border-color: #94A3B8 !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #1D4ED8 !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.22) !important;
    }

    /* User Page Header Banner */
    .brand-banner {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .brand-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 3px;
        line-height: 1.4;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.30rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 3px;
        letter-spacing: -0.01em;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 3px;
    }

    /* Junction & Inspector Panels */
    .junction-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .detail-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #2563EB;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    /* Clean Route Cards */
    .route-card {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .route-card-rec {
        border: 1.5px solid #10B981;
        border-left: 5px solid #10B981;
        background-color: #F0FDF4;
    }

    /* Status Badges */
    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-low { background-color: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
    .pill-medium { background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
    .pill-high { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    .pill-rec { background-color: #10B981; color: #FFFFFF; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; }

    /* Alert Cards */
    .alert-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .alert-critical { border-left: 4px solid #EF4444; }
    .alert-warning { border-left: 4px solid #F59E0B; }
    .alert-normal { border-left: 4px solid #10B981; }
</style>
"""


USER_CSS = """
<style>
    .stApp {
        background-color: var(--background-color, #FFFFFF);
        color: var(--text-color, #0F172A);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    section[data-testid="stSidebar"] { display: none; }
    header[data-testid="stHeader"] { background-color: var(--background-color, #FFFFFF); }
    .block-container {
        padding-top: 0.85rem !important;
        padding-bottom: 2rem !important;
        max-width: 1800px !important;
        width: 100% !important;
    }
    .nav-header-bar, .brand-banner, .kpi-card, .junction-card, .detail-panel, .alert-card, .route-card {
        background-color: var(--secondary-background-color, #F0F2F6);
        color: var(--text-color, #0F172A);
        border-color: var(--border-color, rgba(128, 128, 128, 0.3));
        border-radius: 8px;
    }
    .nav-header-bar {
        padding: 10px 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    .brand-banner { border-left: 4px solid #2563EB; padding: 12px 18px; margin-bottom: 14px; }
    .brand-title, .kpi-value { color: var(--text-color, #0F172A); }
    .brand-subtitle, .kpi-label, .kpi-sub { color: var(--text-color, #0F172A); opacity: 0.72; }
    .kpi-card, .junction-card, .detail-panel, .alert-card, .route-card {
        padding: 14px 16px;
        margin-bottom: 12px;
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.3));
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    .detail-panel { border-top: 3px solid #2563EB; }
    .route-card-rec {
        border: 1px solid #10B981;
        border-left: 5px solid #10B981;
        background-color: color-mix(in srgb, #10B981 10%, var(--secondary-background-color, #F0F2F6));
    }

    div[data-testid="stHorizontalBlock"] .stButton > button {
        background-color: var(--secondary-background-color, #F0F2F6) !important;
        color: var(--text-color, #0F172A) !important;
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.35)) !important;
        border-radius: 6px !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.55rem !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        border-color: #2563EB !important;
        filter: brightness(0.96);
    }
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] {
        background-color: var(--secondary-background-color, #F0F2F6) !important;
        border-color: var(--border-color, rgba(128, 128, 128, 0.4)) !important;
        color: var(--text-color, #0F172A) !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input { color: var(--text-color, #0F172A) !important; }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], li[role="option"] {
        background-color: var(--secondary-background-color, #F0F2F6) !important;
        color: var(--text-color, #0F172A) !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        filter: brightness(0.94);
    }
    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 650;
        color: var(--text-color, #0F172A);
        background-color: var(--secondary-background-color, #F0F2F6);
    }
    .pill-low { border: 1px solid #10B981; border-left: 3px solid #10B981; }
    .pill-medium { border: 1px solid #F59E0B; border-left: 3px solid #F59E0B; }
    .pill-high { border: 1px solid #EF4444; border-left: 3px solid #EF4444; }
    .pill-rec { background-color: #059669; color: #FFFFFF; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; }

    .metric-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
    .mini-metric { display: flex; flex-direction: column; gap: 2px; padding: 8px 10px; border-radius: 6px; background: var(--background-color, #FFFFFF); }
    .mini-metric span, .updated-at { font-size: 0.75rem; opacity: 0.72; }
    .mini-metric strong { font-size: 0.92rem; color: var(--text-color, #0F172A); }
    .updated-at { margin-top: 10px; }
    .trip-summary-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
    .trip-summary-item { display: flex; flex-direction: column; justify-content: center; gap: 3px; min-width: 0; padding: 10px 12px; border-radius: 7px; background: var(--background-color, #FFFFFF); }
    .trip-summary-item span, .route-card-stats span { font-size: 0.76rem; opacity: 0.72; }
    .trip-summary-item strong { font-size: clamp(1rem, 1.35vw, 1.3rem); line-height: 1.25; overflow-wrap: anywhere; }
    .trip-summary-risk-item { border-left: 3px solid #F59E0B; }

    .route-card { transition: transform 120ms ease, box-shadow 120ms ease; }
    .route-card:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12); }
    .route-card-rec { background: color-mix(in srgb, #10B981 9%, var(--secondary-background-color, #F0F2F6)); }
    .route-card-alt-selected { border-color: #2563EB !important; border-left: 5px solid #2563EB !important; }
    .route-card-header { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
    .route-card-status { display: flex; align-items: center; gap: 5px; }
    .route-card-status > span:first-child { font-size: 0.65rem; font-weight: 700; opacity: 0.68; letter-spacing: 0.03em; }
    .route-card-title { font-size: 1.05rem; line-height: 1.3; font-weight: 750; margin-bottom: 10px; }
    .route-card-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; margin-bottom: 10px; }
    .route-card-stats > div { display: flex; flex-direction: column; gap: 3px; min-width: 0; padding: 7px 8px; border-radius: 6px; background: var(--background-color, #FFFFFF); }
    .route-card-stats strong { font-size: 0.9rem; line-height: 1.25; overflow-wrap: anywhere; }
    .route-card-path { font-size: 0.83rem; line-height: 1.45; }
    .route-card-path > span { display: block; font-size: 0.7rem; opacity: 0.7; font-weight: 650; margin-bottom: 2px; }
    .route-card-via { font-size: 0.75rem; opacity: 0.72; margin-top: 5px; }
    .route-delta { font-size: 0.78rem; font-weight: 650; margin-top: 8px; }
    .route-delta-slower { color: #B91C1C; }
    .route-delta-faster { color: #047857; }

    .map-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin: 6px 0 10px; font-size: 0.78rem; }
    .map-legend > span { display: inline-flex; align-items: center; gap: 6px; }
    .legend-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .legend-low { background: #22C55E; }
    .legend-medium { background: #F59E0B; }
    .legend-high { background: #EF4444; }
    .legend-line { width: 18px; height: 3px; border-radius: 3px; display: inline-block; }
    .legend-recommended { background: #10B981; }
    .legend-alternative { background: #64748B; }
    .legend-emergency { background: #A855F7; }
    .map-tip { opacity: 0.72; }

    button:focus-visible, [role="button"]:focus-visible, [role="combobox"]:focus-visible, input:focus-visible {
        outline: 3px solid #2563EB !important;
        outline-offset: 2px !important;
    }
    @media (max-width: 760px) {
        .nav-header-bar { padding: 8px !important; }
        .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
        .metric-grid { gap: 6px; }
        .trip-summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
        .trip-summary-item { padding: 9px 10px; }
        .route-card-stats { gap: 5px; }
        .route-card-stats > div { padding: 7px 6px; }
        .map-legend { gap: 7px 10px; }
    }

    /* User-authored inline card colors follow the active Streamlit theme. */
    .stApp [style*="color:#0F172A"], .stApp [style*="color:#1E293B"],
    .stApp [style*="color:#334155"], .stApp [style*="color:#475569"],
    .stApp [style*="color:#64748B"], .stApp [style*="color:#94A3B8"] {
        color: var(--text-color, #0F172A) !important;
    }
</style>
"""


def inject_theme(mode: str = "dark") -> None:
    """Inject the admin/login dark theme or theme-aware user styles."""
    css = USER_CSS if mode == "streamlit" else LIGHT_CSS if mode == "light" else DARK_CSS
    shared_map_css = """<style>
        .map-legend { display:flex; flex-wrap:wrap; align-items:center; gap:8px 14px; margin:6px 0 10px; font-size:.78rem; }
        .map-legend > span { display:inline-flex; align-items:center; gap:6px; }
        .legend-dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
        .legend-low { background:#22C55E; } .legend-medium { background:#F59E0B; } .legend-high { background:#EF4444; }
        .legend-line { width:18px; height:3px; border-radius:3px; display:inline-block; }
        .legend-recommended { background:#10B981; } .legend-alternative { background:#64748B; } .legend-emergency { background:#A855F7; }
        .map-tip { opacity:.72; }
    </style>"""
    st.markdown(css + shared_map_css, unsafe_allow_html=True)


def status_pill_html(condition: str, friendly: bool = False) -> str:
    """Return an HTML badge colored Green=Low, Orange=Medium, Red=High."""
    clean = str(condition).title()
    cls = {"Low": "pill-low", "Medium": "pill-medium", "High": "pill-high"}.get(clean, "pill-medium")
    label = CONDITION_FRIENDLY_LABEL.get(clean, clean) if friendly else clean
    return f'<span class="status-pill {cls}">{html.escape(label)}</span>'


def render_kpi_card(label: str, value: str, sub_html: str = "") -> None:
    """Render a clean rounded KPI card respecting active light/dark theme CSS."""
    sub_block = f'<div class="kpi-sub">{sub_html}</div>' if sub_html else ""
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f"{sub_block}</div>",
        unsafe_allow_html=True,
    )


def render_limitation_footer() -> None:
    """Display the transparent dataset & simulation limitation notice."""
    st.caption(
        "Transparency Notice: Traffic metrics & ML forecasts derive strictly from the synthetic Gandhipuram CSV "
        "(218 valid rows, 12–16 Oct 2026, 07:30–20:00). OpenStreetMap provides street geometry/basemaps only; "
        "emergency priority & signal timings are software simulations."
    )
