"""Secure role-based authentication, SQLite user store, and RBAC enforcement."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import getpass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import sys
from typing import Any, Iterator

from src.validation import PROJECT_ROOT


DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "traffic_app.db"
VALID_ROLES = ("admin", "user")
PBKDF2_ITERATIONS = 310_000
MIN_PASSWORD_LENGTH = 8
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def is_valid_email(value: str) -> bool:
    """Return True if ``value`` looks like a valid email address."""
    clean = value.strip()
    if len(clean) < 5 or len(clean) > 120 or " " in clean or ".." in clean:
        return False
    if not EMAIL_PATTERN.match(clean):
        return False
    local, domain = clean.split("@", 1)
    if (
        local.startswith(".")
        or local.endswith(".")
        or domain.startswith(".")
        or domain.endswith(".")
        or domain.startswith("-")
        or domain.endswith("-")
    ):
        return False
    return True

ADMIN_PAGES = (
    "Overview",
    "Junction Monitoring",
    "Video Analytics",
    "Emergency Detection",
    "Future Predictions",
    "Proactive Alerts",
    "Route Management",
    "Emergency Priority",
    "Model Performance",
    "Dataset Validation",
    "User Management",
    "Settings",
)

USER_PAGES = (
    "Home",
    "Traffic Map",
    "Find Route",
    "Future Traffic",
    "Traffic Alerts",
    "Emergency Information",
    "Account",
)

ADMIN_PAGE_ALIASES: dict[str, str] = {
    "Live Junction Monitoring": "Junction Monitoring",
    "Emergency Vehicle Detection": "Emergency Detection",
    "Future Traffic Prediction": "Future Predictions",
    "Intelligent Route Management": "Route Management",
    "Emergency Priority Control": "Emergency Priority",
    "Emergency Vehicle Priority": "Emergency Priority",
    "Signal Simulation": "Emergency Priority",
    "Adaptive Signal Control": "Emergency Priority",
}

USER_PAGE_ALIASES: dict[str, str] = {
    "Traffic": "Traffic Map",
    "Find Best Route": "Find Route",
    "Alerts": "Traffic Alerts",
    "My Account": "Account",
}

DEFAULT_APP_SETTINGS: dict[str, Any] = {
    "default_snapshot_mode": "Full-Network Common Snapshot",
    "default_horizon_hours": 4,
    "use_osm_dark_tiles": True,
    "min_password_length": MIN_PASSWORD_LENGTH,
    "simulate_offline_roads": False,
    "simulate_tile_failure": False,
}


@dataclass(frozen=True)
class UserPublic:
    """Public user projection — never contains password hashes or secrets."""

    id: int
    username: str
    role: str
    is_active: bool
    created_at: str
    last_login_at: str | None
    created_by: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(dt: datetime | None = None) -> str:
    return (dt or _utc_now()).strftime("%Y-%m-%d %H:%M:%S UTC")


def hash_password(password: str, min_length: int = MIN_PASSWORD_LENGTH) -> str:
    """Hash password with PBKDF2-HMAC-SHA256 and a random 16-byte salt."""
    if not isinstance(password, str) or len(password) < min_length:
        raise ValueError(f"Password must be at least {min_length} characters long.")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time verification against PBKDF2-HMAC-SHA256 stored hash."""
    if not password or not stored_hash:
        return False
    try:
        algo, iter_str, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iter_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, TypeError):
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def can_access_page(role: str, page_name: str) -> bool:
    """Return True strictly if ``role`` is authorized for ``page_name``."""
    canon_admin = ADMIN_PAGE_ALIASES.get(page_name, page_name)
    canon_user = USER_PAGE_ALIASES.get(page_name, page_name)

    if role == "admin":
        return canon_admin in ADMIN_PAGES or page_name in ADMIN_PAGES
    if role == "user":
        return canon_user in USER_PAGES or page_name in USER_PAGES
    return False


def get_allowed_pages(role: str) -> tuple[str, ...]:
    """Return ordered sidebar navigation pages permitted for ``role``."""
    if role == "admin":
        return ADMIN_PAGES
    if role == "user":
        return USER_PAGES
    return ()


class AuthManager:
    """SQLite-backed authentication, session management, settings, and RBAC."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self.bootstrap_admin_from_env()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT,
                    created_by TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    remember_me INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL
                );
                """
            )
            for key, val in DEFAULT_APP_SETTINGS.items():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO app_settings (key, value_json, updated_at, updated_by)
                    VALUES (?, ?, ?, ?)
                    """,
                    (key, json.dumps(val), _utc_iso(), "system"),
                )

    @staticmethod
    def _row_to_public(row: sqlite3.Row) -> UserPublic:
        return UserPublic(
            id=int(row["id"]),
            username=str(row["username"]),
            role=str(row["role"]),
            is_active=bool(row["is_active"]),
            created_at=str(row["created_at"]),
            last_login_at=str(row["last_login_at"]) if row["last_login_at"] else None,
            created_by=str(row["created_by"]),
        )

    @staticmethod
    def _validate_username(username: str) -> str:
        clean = username.strip()
        if "@" in clean:
            if not is_valid_email(clean):
                raise ValueError("Please enter a valid email address (for example: name@example.com).")
            return clean.lower()
        if len(clean) < 3 or len(clean) > 40 or not all(ch.isalnum() or ch in "_.-" for ch in clean):
            raise ValueError(
                "Enter a valid email address or 3-40 character username using letters, numbers, '.', '_', or '-'."
            )
        return clean

    def has_admin(self) -> bool:
        """Check whether at least one active admin exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin' AND is_active = 1"
            ).fetchone()
            return int(row["cnt"]) > 0

    def create_initial_admin(self, username: str, password: str) -> UserPublic:
        """Create the very first administrator account when none exists yet."""
        if self.has_admin():
            raise PermissionError(
                "An administrator account already exists. Use an existing admin session to manage accounts."
            )
        clean_user = self._validate_username(username)
        min_len = int(self.get_settings().get("min_password_length", MIN_PASSWORD_LENGTH))
        pw_hash = hash_password(password, min_length=min_len)
        now = _utc_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, role, is_active, created_at, created_by)
                VALUES (?, ?, 'admin', 1, ?, 'initial_setup')
                """,
                (clean_user, pw_hash, now),
            )
            row = conn.execute("SELECT * FROM users WHERE username = ?", (clean_user,)).fetchone()
        return self._row_to_public(row)

    def bootstrap_admin_from_env(self) -> UserPublic | None:
        """Create initial admin from TRAFFIC_ADMIN_USER and TRAFFIC_ADMIN_PASSWORD if unset."""
        env_user = os.environ.get("TRAFFIC_ADMIN_USER", "").strip()
        env_pw = os.environ.get("TRAFFIC_ADMIN_PASSWORD", "")
        if env_user and env_pw and not self.has_admin():
            return self.create_initial_admin(env_user, env_pw)
        return None

    def authenticate(
        self,
        username: str,
        password: str,
        remember_me: bool = False,
    ) -> tuple[UserPublic, str]:
        """Verify credentials, enforce active status, and issue a session token."""
        clean_user = username.strip()
        if not clean_user or not password:
            raise ValueError("Please enter both email/username and password.")

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (clean_user,)).fetchone()
            if row is None or not verify_password(password, str(row["password_hash"])):
                raise ValueError(
                    "Invalid email/username or password. New travelers can register any other email under the 'Register (Normal User)' tab."
                )
            if not bool(row["is_active"]):
                raise PermissionError("This account has been deactivated. Contact an administrator.")

            now = _utc_now()
            ttl_hours = 24 * 7 if remember_me else 2
            expires = now + timedelta(hours=ttl_hours)
            raw_token = secrets.token_urlsafe(32)

            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (_utc_iso(now), int(row["id"])))
            conn.execute(
                """
                INSERT INTO sessions (token_hash, user_id, remember_me, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (_hash_token(raw_token), int(row["id"]), 1 if remember_me else 0, _utc_iso(now), _utc_iso(expires)),
            )
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (int(row["id"]),)).fetchone()
        return self._row_to_public(updated), raw_token

    def verify_session(self, session_token: str | None) -> UserPublic | None:
        """Return active UserPublic for a valid non-expired session token, else None."""
        if not session_token:
            return None
        tok_hash = _hash_token(session_token)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.*, s.expires_at
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ?
                """,
                (tok_hash,),
            ).fetchone()
            if row is None:
                return None
            expires_dt = datetime.strptime(str(row["expires_at"]), "%Y-%m-%d %H:%M:%S UTC").replace(
                tzinfo=timezone.utc
            )
            if expires_dt <= _utc_now() or not bool(row["is_active"]):
                conn.execute("DELETE FROM sessions WHERE token_hash = ?", (tok_hash,))
                return None
            return self._row_to_public(row)

    def logout(self, session_token: str | None) -> None:
        """Invalidate a session token immediately."""
        if not session_token:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(session_token),))

    def require_authenticated(self, session_token: str | None) -> UserPublic:
        """Verify caller holds a valid active session."""
        user = self.verify_session(session_token)
        if user is None:
            raise PermissionError("Authentication required. Please log in.")
        return user

    def require_admin(self, session_token: str | None) -> UserPublic:
        """Verify caller holds an active administrator session."""
        user = self.require_authenticated(session_token)
        if user.role != "admin":
            raise PermissionError("Administrator privileges required for this action.")
        return user

    def admin_list_users(self, actor_token: str) -> list[UserPublic]:
        """List all registered users without exposing password hashes."""
        self.require_admin(actor_token)
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY role ASC, username ASC").fetchall()
        return [self._row_to_public(r) for r in rows]

    def admin_create_user(
        self,
        actor_token: str,
        username: str,
        password: str,
        role: str = "user",
    ) -> UserPublic:
        """Create a new user account (admin-only)."""
        admin = self.require_admin(actor_token)
        if role not in VALID_ROLES:
            raise ValueError(f"Role must be one of {VALID_ROLES}.")
        clean_user = self._validate_username(username)
        min_len = int(self.get_settings().get("min_password_length", MIN_PASSWORD_LENGTH))
        pw_hash = hash_password(password, min_length=min_len)
        now = _utc_iso()
        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM users WHERE username = ?", (clean_user,)).fetchone()
            if existing is not None:
                raise ValueError(f"Username '{clean_user}' already exists.")
            conn.execute(
                """
                INSERT INTO users (username, password_hash, role, is_active, created_at, created_by)
                VALUES (?, ?, ?, 1, ?, ?)
                """,
                (clean_user, pw_hash, role, now, admin.username),
            )
            row = conn.execute("SELECT * FROM users WHERE username = ?", (clean_user,)).fetchone()
        return self._row_to_public(row)

    def admin_change_role(
        self,
        actor_token: str,
        target_username: str,
        new_role: str,
    ) -> UserPublic:
        """Change a user's role while protecting the last active admin."""
        self.require_admin(actor_token)
        if new_role not in VALID_ROLES:
            raise ValueError(f"Role must be one of {VALID_ROLES}.")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (target_username,)).fetchone()
            if row is None:
                raise ValueError(f"User '{target_username}' not found.")
            if str(row["role"]) == "admin" and new_role != "admin":
                other_admins = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin' AND is_active = 1 AND id != ?",
                    (int(row["id"]),),
                ).fetchone()
                if int(other_admins["cnt"]) == 0:
                    raise ValueError("Cannot demote the last active administrator.")
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, int(row["id"])))
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (int(row["id"]),)).fetchone()
        return self._row_to_public(updated)

    def admin_set_active(
        self,
        actor_token: str,
        target_username: str,
        is_active: bool,
    ) -> UserPublic:
        """Activate or deactivate an account; invalidates active sessions on deactivation."""
        admin = self.require_admin(actor_token)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (target_username,)).fetchone()
            if row is None:
                raise ValueError(f"User '{target_username}' not found.")
            if not is_active and int(row["id"]) == admin.id:
                raise ValueError("You cannot deactivate your own active administrator session.")
            if not is_active and str(row["role"]) == "admin":
                other_admins = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin' AND is_active = 1 AND id != ?",
                    (int(row["id"]),),
                ).fetchone()
                if int(other_admins["cnt"]) == 0:
                    raise ValueError("Cannot deactivate the last active administrator.")
            conn.execute(
                "UPDATE users SET is_active = ? WHERE id = ?",
                (1 if is_active else 0, int(row["id"])),
            )
            if not is_active:
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (int(row["id"]),))
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (int(row["id"]),)).fetchone()
        return self._row_to_public(updated)

    def admin_reset_password(
        self,
        actor_token: str,
        target_username: str,
        new_password: str,
    ) -> None:
        """Securely reset a user's password and revoke existing sessions."""
        self.require_admin(actor_token)
        min_len = int(self.get_settings().get("min_password_length", MIN_PASSWORD_LENGTH))
        pw_hash = hash_password(new_password, min_length=min_len)
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM users WHERE username = ?", (target_username,)).fetchone()
            if row is None:
                raise ValueError(f"User '{target_username}' not found.")
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (pw_hash, int(row["id"])))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (int(row["id"]),))

    def change_own_password(
        self,
        actor_token: str,
        current_password: str,
        new_password: str,
    ) -> None:
        """Allow an authenticated user to change their own password."""
        user = self.require_authenticated(actor_token)
        min_len = int(self.get_settings().get("min_password_length", MIN_PASSWORD_LENGTH))
        new_hash = hash_password(new_password, min_length=min_len)
        with self._connect() as conn:
            row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user.id,)).fetchone()
            if row is None or not verify_password(current_password, str(row["password_hash"])):
                raise ValueError("Current password is incorrect.")
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user.id))

    def get_settings(self) -> dict[str, Any]:
        """Read persisted application configuration."""
        settings = dict(DEFAULT_APP_SETTINGS)
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value_json FROM app_settings").fetchall()
        for r in rows:
            try:
                settings[str(r["key"])] = json.loads(str(r["value_json"]))
            except json.JSONDecodeError:
                pass
        return settings

    def admin_update_settings(self, actor_token: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update application configuration (admin-only)."""
        admin = self.require_admin(actor_token)
        now = _utc_iso()
        with self._connect() as conn:
            for key, val in updates.items():
                if key not in DEFAULT_APP_SETTINGS:
                    continue
                conn.execute(
                    """
                    INSERT INTO app_settings (key, value_json, updated_at, updated_by)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json = excluded.value_json,
                        updated_at = excluded.updated_at,
                        updated_by = excluded.updated_by
                    """,
                    (key, json.dumps(val), now, admin.username),
                )
        return self.get_settings()

    def register_user(self, email: str, password: str) -> UserPublic:
        """Self-register a new account strictly as a Normal User (never admin)."""
        clean_email = email.strip()
        if not is_valid_email(clean_email):
            raise ValueError("Please enter a valid email address (for example: name@example.com).")
        clean_email = clean_email.lower()
        min_len = int(self.get_settings().get("min_password_length", MIN_PASSWORD_LENGTH))
        pw_hash = hash_password(password, min_length=min_len)
        now = _utc_iso()
        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM users WHERE username = ?", (clean_email,)).fetchone()
            if existing is not None:
                raise ValueError("An account with this email already exists. Please use Sign In instead.")
            conn.execute(
                """
                INSERT INTO users (username, password_hash, role, is_active, created_at, created_by)
                VALUES (?, ?, 'user', 1, ?, 'self_registration')
                """,
                (clean_email, pw_hash, now),
            )
            row = conn.execute("SELECT * FROM users WHERE username = ?", (clean_email,)).fetchone()
        return self._row_to_public(row)

    def set_admin_cli(
        self,
        username: str,
        password: str,
        replace_existing: bool = False,
    ) -> UserPublic:
        """Create or update an administrator account in the local database (CLI-only)."""
        clean_user = self._validate_username(username)
        min_len = int(self.get_settings().get("min_password_length", MIN_PASSWORD_LENGTH))
        pw_hash = hash_password(password, min_length=min_len)
        now = _utc_iso()
        with self._connect() as conn:
            existing = conn.execute("SELECT * FROM users WHERE username = ?", (clean_user,)).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO users (username, password_hash, role, is_active, created_at, created_by)
                    VALUES (?, ?, 'admin', 1, ?, 'cli_admin_setup')
                    """,
                    (clean_user, pw_hash, now),
                )
            else:
                conn.execute(
                    "UPDATE users SET password_hash = ?, role = 'admin', is_active = 1 WHERE id = ?",
                    (pw_hash, int(existing["id"])),
                )
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (int(existing["id"]),))
            target = conn.execute("SELECT * FROM users WHERE username = ?", (clean_user,)).fetchone()
            if replace_existing and target is not None:
                demoted = conn.execute(
                    "SELECT id FROM users WHERE role = 'admin' AND id != ?",
                    (int(target["id"]),),
                ).fetchall()
                for d_row in demoted:
                    conn.execute("UPDATE users SET role = 'user' WHERE id = ?", (int(d_row["id"]),))
                    conn.execute("DELETE FROM sessions WHERE user_id = ?", (int(d_row["id"]),))
            row = conn.execute("SELECT * FROM users WHERE username = ?", (clean_user,)).fetchone()
        return self._row_to_public(row)

    def create_user_cli(self, username: str, password: str, role: str = "user") -> UserPublic:
        """CLI helper for local development bootstrap (runs outside Streamlit UI)."""
        if role == "admin" and not self.has_admin():
            return self.create_initial_admin(username, password)
        if role not in VALID_ROLES:
            raise ValueError(f"Role must be one of {VALID_ROLES}.")
        clean_user = self._validate_username(username)
        min_len = int(self.get_settings().get("min_password_length", MIN_PASSWORD_LENGTH))
        pw_hash = hash_password(password, min_length=min_len)
        now = _utc_iso()
        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM users WHERE username = ?", (clean_user,)).fetchone()
            if existing is not None:
                raise ValueError(f"Username '{clean_user}' already exists.")
            conn.execute(
                """
                INSERT INTO users (username, password_hash, role, is_active, created_at, created_by)
                VALUES (?, ?, ?, 1, ?, 'cli_bootstrap')
                """,
                (clean_user, pw_hash, role, now),
            )
            row = conn.execute("SELECT * FROM users WHERE username = ?", (clean_user,)).fetchone()
        return self._row_to_public(row)


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="Traffic App Auth Administration CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    c_admin = sub.add_parser("create-admin", help="Create the initial administrator account")
    c_admin.add_argument("--username", required=True, help="Administrator email or username")
    c_admin.add_argument("--password", required=False, help="Password (prompted securely if omitted)")

    s_admin = sub.add_parser("set-admin", help="Create or update an administrator account cleanly")
    s_admin.add_argument("--username", required=True, help="Administrator email or username")
    s_admin.add_argument("--password", required=False, help="Password (prompted securely if omitted)")
    s_admin.add_argument(
        "--replace-existing",
        action="store_true",
        help="Demote other admin accounts to normal user role",
    )

    c_user = sub.add_parser("create-user", help="Create a normal user account")
    c_user.add_argument("--username", required=True, help="Normal user email or username")
    c_user.add_argument("--password", required=False, help="Password (prompted securely if omitted)")

    args = parser.parse_args()
    auth = AuthManager()
    pw = args.password or getpass.getpass("Enter password (min 8 chars): ")
    if args.command == "create-admin":
        created = auth.create_initial_admin(args.username, pw)
        print(f"Created initial admin account: {created.username} (role={created.role})")
    elif args.command == "set-admin":
        updated = auth.set_admin_cli(args.username, pw, replace_existing=bool(args.replace_existing))
        print(f"Configured admin account: {updated.username} (role={updated.role})")
    elif args.command == "create-user":
        created = auth.create_user_cli(args.username, pw, role="user")
        print(f"Created normal user account: {created.username} (role={created.role})")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _cli_main()

