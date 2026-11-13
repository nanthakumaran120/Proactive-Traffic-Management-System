"""Login page and first-time administrator setup view."""

from __future__ import annotations

import streamlit as st

from src.auth import AuthManager, MIN_PASSWORD_LENGTH
from src.ui_theme import render_limitation_footer


def render_login_page(auth: AuthManager) -> None:
    """Render secure role-based login page (and initial admin setup if none exists)."""
    _, col_mid, _ = st.columns([1.0, 1.35, 1.0])

    with col_mid:
        st.markdown(
            """
            <div class="login-card" style="text-align:center;margin-bottom:18px;">
                <div style="font-size:2rem;margin-bottom:6px;">🚦</div>
                <div class="brand-title" style="font-size:1.55rem;">AI-POWERED PROACTIVE TRAFFIC MANAGEMENT</div>
                <div class="brand-subtitle" style="font-size:0.95rem;margin-top:6px;">
                    Smart Traffic Prediction and Intelligent Route Recommendation System
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        has_admin = auth.has_admin()

        if not has_admin:
            st.warning(
                "First-Time Setup: No administrator account exists yet in the local user database. "
                "Create the initial administrator account below (or set `TRAFFIC_ADMIN_USER` and "
                "`TRAFFIC_ADMIN_PASSWORD` environment variables)."
            )
            with st.container(border=True):
                st.subheader("Create Initial Administrator Account")
                setup_user = st.text_input("Administrator Email or Username", key="setup_admin_user")
                setup_show = st.checkbox("Show password", key="setup_show_pw")
                setup_pw = st.text_input(
                    "Administrator Password",
                    type="default" if setup_show else "password",
                    key="setup_admin_pw",
                    help=f"Minimum {MIN_PASSWORD_LENGTH} characters.",
                )
                setup_pw2 = st.text_input(
                    "Confirm Password",
                    type="default" if setup_show else "password",
                    key="setup_admin_pw2",
                )
                if st.button("Create Administrator Account", type="primary", width="stretch", key="btn_create_initial_admin"):
                    if setup_pw != setup_pw2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            created = auth.create_initial_admin(setup_user, setup_pw)
                            st.success(
                                f"Administrator account '{created.username}' created securely. You can now sign in below."
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

        with st.container(border=True):
            tab_signin, tab_register = st.tabs(["Sign In", "Register (Normal User)"])

            with tab_signin:
                st.subheader("Sign In")
                st.caption(
                    "Your assigned role (Admin or Normal User) is determined automatically from your account."
                )

                username = st.text_input(
                    "Email or Username",
                    key="login_username",
                    placeholder="name@example.com or username",
                )
                show_password = st.checkbox("Show password", key="login_show_password")
                password = st.text_input(
                    "Password",
                    type="default" if show_password else "password",
                    key="login_password",
                    placeholder="Enter your password",
                )
                remember_me = st.checkbox(
                    "Remember login session",
                    value=True,
                    key="login_remember_me",
                    help="Keeps your server session valid across extended active work (7 days vs 2 hours).",
                )

                login_btn = st.button("Login", type="primary", width="stretch", key="btn_login")
                if login_btn:
                    try:
                        user, token = auth.authenticate(username, password, remember_me=remember_me)
                        st.session_state["auth_token"] = token
                        st.session_state["auth_role"] = user.role
                        st.session_state["nav_page"] = "Overview" if user.role == "admin" else "Home"
                        st.query_params["page"] = st.session_state["nav_page"]
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            with tab_register:
                st.subheader("Create Normal User Account")
                st.caption(
                    "Any other email address registers strictly as a Normal User (Admin access cannot be selected)."
                )
                reg_email = st.text_input(
                    "Email Address",
                    key="reg_email",
                    placeholder="traveler@example.com",
                )
                reg_show_password = st.checkbox("Show password", key="reg_show_password")
                reg_password = st.text_input(
                    "Password",
                    type="default" if reg_show_password else "password",
                    key="reg_password",
                    help=f"Minimum {MIN_PASSWORD_LENGTH} characters.",
                )
                reg_password2 = st.text_input(
                    "Confirm Password",
                    type="default" if reg_show_password else "password",
                    key="reg_password2",
                )
                if st.button("Register & Sign In", type="primary", width="stretch", key="btn_register"):
                    if reg_password != reg_password2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            created = auth.register_user(reg_email, reg_password)
                            user, token = auth.authenticate(created.username, reg_password, remember_me=True)
                            st.session_state["auth_token"] = token
                            st.session_state["auth_role"] = user.role
                            st.session_state["nav_page"] = "Home"
                            st.query_params["page"] = "Home"
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

        render_limitation_footer()

