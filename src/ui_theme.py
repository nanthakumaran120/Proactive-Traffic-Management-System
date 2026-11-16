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
    :root {
        --user-primary: #176B68;
        --user-primary-hover: #125754;
        --user-accent: #A96A12;
        --user-bg: #F5F4EF;
        --user-surface: #FFFFFF;
        --user-text: #202A28;
        --user-muted: #596762;
        --user-border: #D8DED9;
        --user-focus: #A96A12;
        --user-radius-control: 4px;
        --user-radius-panel: 6px;
    }
    /* Full viewport utilization */
    .stApp {
        background-color: var(--user-bg);
        color: var(--user-text);
        font-family: "Source Sans 3", "Segoe UI", Arial, sans-serif;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    header[data-testid="stHeader"] {
        background-color: rgba(245, 244, 239, 0.96);
    }
    .block-container {
        padding-top: 0.85rem !important;
        padding-bottom: 2.0rem !important;
        padding-left: clamp(1rem, 3vw, 2.5rem) !important;
        padding-right: clamp(1rem, 3vw, 2.5rem) !important;
        max-width: 1800px !important;
        width: 100% !important;
    }

    /* Top Navigation Header Bar */
    .stApp [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--user-surface);
        border-color: var(--user-border) !important;
        border-radius: var(--user-radius-panel) !important;
        box-shadow: none !important;
    }
    .user-brand-title {
        font-size: 1rem;
        line-height: 1.2;
        font-weight: 700;
        color: var(--user-text);
    }
    .user-brand-location {
        margin-top: 3px;
        font-size: 0.78rem;
        color: var(--user-muted);
    }
    .user-account-name {
        color: var(--user-text);
        text-align: right;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* High-contrast User Navigation Buttons */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background-color: var(--user-surface) !important;
        color: #34413D !important;
        border: 1px solid var(--user-border) !important;
        border-radius: var(--user-radius-control) !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.75rem !important;
        min-height: 44px !important;
        box-shadow: none !important;
        transition: background-color 0.12s ease-in-out, border-color 0.12s ease-in-out !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background-color: #EEF2EE !important;
        color: var(--user-text) !important;
        border-color: #87958F !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
        background-color: var(--user-primary) !important;
        color: var(--user-surface) !important;
        border-color: var(--user-primary) !important;
        box-shadow: none !important;
    }

    .stApp button:focus-visible, .stApp a:focus-visible,
    .stApp input:focus-visible, .stApp textarea:focus-visible,
    .stApp [role="combobox"]:focus-visible, .stApp [role="radio"]:focus-visible {
        outline: 2px solid var(--user-focus) !important;
        outline-offset: 2px !important;
        box-shadow: none !important;
    }
    .stApp input, .stApp textarea, .stApp [role="combobox"] {
        border-radius: var(--user-radius-control) !important;
    }
    .stApp [data-testid="stSelectbox"] [role="combobox"] {
        min-height: 44px;
    }
    .stApp input, .stApp textarea, .stApp button {
        scroll-margin: 12px;
    }
    @media (max-width: 760px) {
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 1.25rem !important;
        }
        .stApp [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            row-gap: 12px;
        }
        .stApp [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 280px;
            min-width: min(100%, 280px);
        }
    }

    /* User Page Header Banner */
    .brand-banner {
        background-color: var(--user-surface);
        border: 1px solid var(--user-border);
        border-left: 3px solid var(--user-primary);
        border-radius: var(--user-radius-panel);
        padding: 12px 18px;
        margin-bottom: 14px;
    }
    .brand-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: var(--user-text);
        letter-spacing: -0.01em;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.85rem;
        color: var(--user-muted);
        margin-top: 3px;
        line-height: 1.4;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: var(--user-surface);
        border: 1px solid var(--user-border);
        border-radius: var(--user-radius-panel);
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: none;
    }
    .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--user-muted);
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.30rem;
        font-weight: 700;
        color: var(--user-text);
        margin-top: 3px;
        letter-spacing: -0.01em;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: var(--user-muted);
        margin-top: 3px;
    }

    /* Junction & Inspector Panels */
    .junction-card {
        background-color: var(--user-surface);
        border: 1px solid var(--user-border);
        border-radius: var(--user-radius-panel);
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: none;
    }
    .detail-panel {
        background-color: var(--user-surface);
        border: 1px solid var(--user-border);
        border-top: 3px solid var(--user-primary);
        border-radius: var(--user-radius-panel);
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: none;
    }

    /* Clean Route Cards */
    .route-card {
        background-color: var(--user-surface);
        border: 1px solid var(--user-border);
        border-radius: var(--user-radius-panel);
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: none;
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
    .pill-rec { background-color: #10B981; color: var(--user-surface); padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; }

    /* Alert Cards */
    .alert-card {
        border-radius: var(--user-radius-panel);
        padding: 12px 16px;
        margin-bottom: 10px;
        background-color: var(--user-surface);
        border: 1px solid var(--user-border);
        box-shadow: none;
    }
    .alert-critical { border-left: 4px solid #EF4444; }
    .alert-warning { border-left: 4px solid #F59E0B; }
    .alert-normal { border-left: 4px solid #10B981; }
</style>
"""


def inject_theme(mode: str = "dark") -> None:
    """Inject consistent Dark Command Centre (Admin/Login) or Light Navigation (User) CSS."""
    st.markdown(LIGHT_CSS if mode == "light" else DARK_CSS, unsafe_allow_html=True)


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
