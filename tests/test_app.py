"""End-to-end Streamlit UI, Real-Map Dashboard, Video Analytics, Emergency Detection, RBAC, and Navigation Tests."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from streamlit.testing.v1 import AppTest

from src.auth import ADMIN_PAGES, USER_PAGES, AuthManager


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class TestRoleBasedStreamlitApp(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "ui_test.db"
        self.old_db_env = os.environ.get("TRAFFIC_AUTH_DB_PATH")
        os.environ["TRAFFIC_AUTH_DB_PATH"] = str(self.db_path)

        self.auth = AuthManager(db_path=self.db_path)
        self.auth.create_initial_admin("test_admin@example.com", "AdminPass!2026")
        _, tok = self.auth.authenticate("test_admin@example.com", "AdminPass!2026")
        self.auth.admin_create_user(tok, "test_user@example.com", "UserPass!2026", role="user")
        self.auth.logout(tok)

    def tearDown(self) -> None:
        if self.old_db_env is None:
            os.environ.pop("TRAFFIC_AUTH_DB_PATH", None)
        else:
            os.environ["TRAFFIC_AUTH_DB_PATH"] = self.old_db_env
        self.tmp_dir.cleanup()

    def test_invalid_login_admin_flow_and_all_13_admin_pages(self) -> None:
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Login startup exception: {at.exception}")

        # 1. Invalid login shows error
        at.text_input(key="login_username").set_value("test_admin@example.com").run()
        at.text_input(key="login_password").set_value("WrongPassword!").run()
        at.button(key="btn_login").click().run()
        self.assertGreater(len(at.error), 0)
        self.assertNotIn("auth_token", at.session_state)

        # 2. Valid Admin login opens Control Centre Overview
        at.text_input(key="login_password").set_value("AdminPass!2026").run()
        at.checkbox(key="login_remember_me").set_value(True).run()
        at.button(key="btn_login").click().run()
        self.assertEqual(len(at.exception), 0, f"Admin login exception: {at.exception}")
        self.assertEqual(at.session_state["auth_role"], "admin")
        self.assertEqual(at.session_state["nav_page"], "Overview")

        # 3. Traverse all Admin Control Centre pages + interactive controls
        route_page_name = "Route Management" if "Route Management" in ADMIN_PAGES else "Intelligent Route Management"
        for page in ADMIN_PAGES:
            at.radio(key="admin_sidebar_nav").set_value(page).run()
            self.assertEqual(len(at.exception), 0, f"Admin page '{page}' raised: {at.exception}")
            self.assertEqual(at.session_state["nav_page"], page)

            if page == "Junction Monitoring":
                at.radio(key="adm_jm_source_sel").set_value("Combined View").run()
                self.assertEqual(len(at.exception), 0)
            elif page == "Video Analytics":
                at.button(key="btn_va_step").click().run()
                self.assertEqual(len(at.exception), 0)
                self.assertIn("admin_video_processor", at.session_state)
            elif page in ("Emergency Detection", "Emergency Vehicle Detection"):
                at.button(key="btn_inject_ev").click().run()
                self.assertEqual(len(at.exception), 0)
                self.assertIn("emergency_event_manager", at.session_state)
            elif page in ("Future Predictions", "Future Traffic Prediction"):
                at.button(key="btn_adm_predict").click().run()
                self.assertEqual(len(at.exception), 0)
            elif page == "Proactive Alerts":
                at.button(key="btn_adm_alerts").click().run()
                self.assertEqual(len(at.exception), 0)
                if any(b.key == "adm_alert_focus_btn_0" for b in at.button):
                    at.button(key="adm_alert_focus_btn_0").click().run()
                    self.assertEqual(len(at.exception), 0)
            elif page in ("Route Management", "Intelligent Route Management"):
                at.button(key="btn_adm_route").click().run()
                self.assertEqual(len(at.exception), 0)
                self.assertIn("adm_unified_route", at.session_state)
            elif page in ("Emergency Priority", "Emergency Priority Control"):
                at.button(key="btn_adm_emg").click().run()
                self.assertEqual(len(at.exception), 0)
                self.assertTrue(at.session_state["adm_unified_emg"].emergency_mode_active)
            elif page == "User Management":
                at.text_input(key="um_create_user").set_value("new_traveler@example.com").run()
                at.text_input(key="um_create_pw").set_value("TravelPass!123").run()
                at.button(key="btn_um_create").click().run()
                self.assertEqual(len(at.exception), 0)
            elif page == "Settings":
                # Verify offline road & tile fallback simulation toggles work cleanly
                at.checkbox(key="set_off_roads").set_value(True).run()
                at.checkbox(key="set_off_tiles").set_value(True).run()
                at.button(key="btn_save_settings").click().run()
                self.assertEqual(len(at.exception), 0)
                at.radio(key="admin_sidebar_nav").set_value(route_page_name).run()
                self.assertEqual(len(at.exception), 0)
                at.button(key="btn_adm_route").click().run()
                self.assertEqual(len(at.exception), 0)
                self.assertFalse(at.session_state["adm_unified_route"].uses_real_osm_roads)
                # Restore normal live settings before proceeding
                at.radio(key="admin_sidebar_nav").set_value("Settings").run()
                at.checkbox(key="set_off_roads").set_value(False).run()
                at.checkbox(key="set_off_tiles").set_value(False).run()
                at.button(key="btn_save_settings").click().run()
                self.assertEqual(len(at.exception), 0)

        # 4. Admin Logout (14. Logout)
        at.button(key="btn_admin_logout").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertNotIn("auth_token", at.session_state)
        self.assertTrue(any(b.key == "btn_login" for b in at.button))

    def test_normal_user_map_first_flow_and_rbac_block(self) -> None:
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

        # Login as Normal User
        at.text_input(key="login_username").set_value("test_user@example.com").run()
        at.text_input(key="login_password").set_value("UserPass!2026").run()
        at.button(key="btn_login").click().run()
        self.assertEqual(len(at.exception), 0, f"User login exception: {at.exception}")
        self.assertEqual(at.session_state["auth_role"], "user")
        self.assertEqual(at.session_state["nav_page"], "Home")

        # Direct URL attempt to Admin page must be blocked cleanly
        at.query_params["page"] = "Signal Simulation"
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["nav_page"], "Home")
        self.assertTrue(any("Access Denied" in e.value for e in at.error))

        # Home page FIND BEST ROUTE computes mapped route right on Home screen
        at.button(key="usr_home_find_route").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertIn("usr_home_route_res", at.session_state)
        if any(b.key == "usr_home_view_alt_1" for b in at.button):
            at.button(key="usr_home_view_alt_1").click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertIn("usr_home_alt", at.session_state)
            self.assertIsNotNone(at.session_state["usr_home_alt"])
            if any(b.key == "usr_home_clear_alt" for b in at.button):
                at.button(key="usr_home_clear_alt").click().run()
                self.assertEqual(len(at.exception), 0)

        # Traverse every top-header page in USER_PAGES
        for page in USER_PAGES:
            at.button(key=f"usr_nav_{page}").click().run()
            self.assertEqual(len(at.exception), 0, f"User page '{page}' raised: {at.exception}")
            self.assertEqual(at.session_state["nav_page"], page)

            if page in ("Find Route", "Find Best Route"):
                at.button(key="btn_usr_find_route").click().run()
                self.assertEqual(len(at.exception), 0)
                self.assertIn("usr_route_res", at.session_state)
            elif page == "Future Traffic":
                at.button(key="btn_usr_check_future").click().run()
                self.assertEqual(len(at.exception), 0)
            elif page in ("Traffic Alerts", "Alerts"):
                if any(b.key == "usr_alert_focus_0" for b in at.button):
                    at.button(key="usr_alert_focus_0").click().run()
                    self.assertEqual(len(at.exception), 0)
            elif page in ("Emergency Information", "Emergency Info"):
                self.assertEqual(len(at.exception), 0)

        # Normal User Logout
        at.button(key="btn_user_logout").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertNotIn("auth_token", at.session_state)

    def test_other_email_self_registers_as_normal_user_only(self) -> None:
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

        # Duplicate admin email self-registration rejected
        at.text_input(key="reg_email").set_value("test_admin@example.com").run()
        at.text_input(key="reg_password").set_value("SomePass!2026").run()
        at.text_input(key="reg_password2").set_value("SomePass!2026").run()
        at.button(key="btn_register").click().run()
        self.assertGreater(len(at.error), 0)
        self.assertNotIn("auth_token", at.session_state)

        # Other valid email self-registers strictly as Normal User
        at.text_input(key="reg_email").set_value("commuter2@example.com").run()
        at.text_input(key="reg_password").set_value("CommutePass!2026").run()
        at.text_input(key="reg_password2").set_value("CommutePass!2026").run()
        at.button(key="btn_register").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["auth_role"], "user")
        self.assertEqual(at.session_state["nav_page"], "Home")


if __name__ == "__main__":
    unittest.main()
