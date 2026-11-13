"""Security, RBAC, password hashing, and session management tests."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from src.auth import (
    ADMIN_PAGES,
    USER_PAGES,
    AuthManager,
    can_access_page,
    hash_password,
    is_valid_email,
    verify_password,
)


class TestAuthAndRBAC(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_auth.db"
        self.auth = AuthManager(db_path=self.db_path)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_password_hashing_and_no_plaintext(self) -> None:
        raw_pw = "StrongPass!99"
        digest = hash_password(raw_pw)
        self.assertNotIn(raw_pw, digest)
        self.assertTrue(digest.startswith("pbkdf2_sha256$"))
        self.assertTrue(verify_password(raw_pw, digest))
        self.assertFalse(verify_password("WrongPass!99", digest))

    def test_admin_and_user_login_logout_and_session_persistence(self) -> None:
        self.assertFalse(self.auth.has_admin())
        admin = self.auth.create_initial_admin("root_admin", "AdminSecret!123")
        self.assertEqual(admin.role, "admin")
        self.assertTrue(self.auth.has_admin())

        # Second initial admin attempt must fail
        with self.assertRaises(PermissionError):
            self.auth.create_initial_admin("rogue_admin", "RoguePass!123")

        # Invalid password rejected cleanly
        with self.assertRaises(ValueError):
            self.auth.authenticate("root_admin", "WrongPassword123")

        # Valid admin login + session persistence across verify_session calls
        auth_admin, admin_tok = self.auth.authenticate("root_admin", "AdminSecret!123", remember_me=True)
        self.assertEqual(auth_admin.role, "admin")
        self.assertIsNotNone(self.auth.verify_session(admin_tok))

        # Admin creates normal user; user list must not expose password hashes
        created_user = self.auth.admin_create_user(admin_tok, "citizen1", "TravelerPass!12", role="user")
        self.assertEqual(created_user.role, "user")
        users = self.auth.admin_list_users(admin_tok)
        self.assertEqual(len(users), 2)
        for u in users:
            self.assertNotIn("password_hash", u.to_dict())

        # Normal user login & RBAC verification
        auth_usr, usr_tok = self.auth.authenticate("citizen1", "TravelerPass!12", remember_me=False)
        self.assertEqual(auth_usr.role, "user")
        self.assertIsNotNone(self.auth.verify_session(usr_tok))

        # Logout invalidates token immediately
        self.auth.logout(usr_tok)
        self.assertIsNone(self.auth.verify_session(usr_tok))

    def test_rbac_blocks_normal_user_from_admin_pages_and_actions(self) -> None:
        self.auth.create_initial_admin("admin1", "AdminPass!123")
        _, admin_tok = self.auth.authenticate("admin1", "AdminPass!123")
        self.auth.admin_create_user(admin_tok, "driver1", "DriverPass!123", role="user")
        _, usr_tok = self.auth.authenticate("driver1", "DriverPass!123")

        # Page access matrix
        for p in ADMIN_PAGES:
            self.assertTrue(can_access_page("admin", p))
            self.assertFalse(can_access_page("user", p))
        for p in USER_PAGES:
            self.assertTrue(can_access_page("user", p))

        # Backend admin mutations must raise PermissionError for normal user
        with self.assertRaises(PermissionError):
            self.auth.require_admin(usr_tok)
        with self.assertRaises(PermissionError):
            self.auth.admin_list_users(usr_tok)
        with self.assertRaises(PermissionError):
            self.auth.admin_create_user(usr_tok, "hacker", "HackerPass!12", role="admin")
        with self.assertRaises(PermissionError):
            self.auth.admin_change_role(usr_tok, "driver1", "admin")
        with self.assertRaises(PermissionError):
            self.auth.admin_reset_password(usr_tok, "admin1", "Hacked!12345")
        with self.assertRaises(PermissionError):
            self.auth.admin_update_settings(usr_tok, {"min_password_length": 4})

    def test_deactivation_revokes_session_and_protects_last_admin(self) -> None:
        self.auth.create_initial_admin("only_admin", "OnlyAdmin!123")
        _, admin_tok = self.auth.authenticate("only_admin", "OnlyAdmin!123")
        self.auth.admin_create_user(admin_tok, "commuter", "Commuter!123", role="user")
        _, comm_tok = self.auth.authenticate("commuter", "Commuter!123")

        # Cannot demote or deactivate sole admin
        with self.assertRaises(ValueError):
            self.auth.admin_change_role(admin_tok, "only_admin", "user")
        with self.assertRaises(ValueError):
            self.auth.admin_set_active(admin_tok, "only_admin", False)

        # Deactivating normal user revokes active session and blocks new login
        self.auth.admin_set_active(admin_tok, "commuter", False)
        self.assertIsNone(self.auth.verify_session(comm_tok))
        with self.assertRaises(PermissionError):
            self.auth.authenticate("commuter", "Commuter!123")

    def test_env_bootstrap_creates_initial_admin(self) -> None:
        env_db = Path(self.tmp_dir.name) / "env_auth.db"
        old_u = os.environ.get("TRAFFIC_ADMIN_USER")
        old_p = os.environ.get("TRAFFIC_ADMIN_PASSWORD")
        try:
            os.environ["TRAFFIC_ADMIN_USER"] = "env_admin"
            os.environ["TRAFFIC_ADMIN_PASSWORD"] = "EnvAdmin!2026"
            mgr = AuthManager(db_path=env_db)
            self.assertTrue(mgr.has_admin())
            user, tok = mgr.authenticate("env_admin", "EnvAdmin!2026")
            self.assertEqual(user.role, "admin")
            self.assertIsNotNone(mgr.verify_session(tok))
        finally:
            if old_u is None:
                os.environ.pop("TRAFFIC_ADMIN_USER", None)
            else:
                os.environ["TRAFFIC_ADMIN_USER"] = old_u
            if old_p is None:
                os.environ.pop("TRAFFIC_ADMIN_PASSWORD", None)
            else:
                os.environ["TRAFFIC_ADMIN_PASSWORD"] = old_p

    def test_email_admin_and_other_email_registers_as_user_only(self) -> None:
        self.assertTrue(is_valid_email("lead.admin@example.com"))
        self.assertFalse(is_valid_email("not-an-email"))
        self.assertFalse(is_valid_email("bad@domain"))

        # Configure email-based admin cleanly via set_admin_cli
        admin = self.auth.set_admin_cli("Lead.Admin@Example.com", "AdminPass!2026", replace_existing=True)
        self.assertEqual(admin.username, "lead.admin@example.com")
        self.assertEqual(admin.role, "admin")

        # Case-insensitive email authentication enters admin role
        signed_admin, _ = self.auth.authenticate("lead.admin@example.com", "AdminPass!2026")
        self.assertEqual(signed_admin.role, "admin")

        # Self-registration rejects existing admin email and invalid emails
        with self.assertRaises(ValueError):
            self.auth.register_user("lead.admin@example.com", "NewPass!2026")
        with self.assertRaises(ValueError):
            self.auth.register_user("invalid_email", "NewPass!2026")

        # Any other valid email registers strictly as Normal User
        normal = self.auth.register_user("Traveler99@Example.com", "Traveler!2026")
        self.assertEqual(normal.username, "traveler99@example.com")
        self.assertEqual(normal.role, "user")
        signed_user, _ = self.auth.authenticate("traveler99@example.com", "Traveler!2026")
        self.assertEqual(signed_user.role, "user")


if __name__ == "__main__":
    unittest.main()

