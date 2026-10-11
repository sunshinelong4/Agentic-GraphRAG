"""Revocable bearer sessions and shared-knowledge-base RBAC.

Passwords use scrypt; only a SHA-256 digest of each random session token is stored.
SQLite WAL is sufficient for the single-host demo, not a multi-host identity store.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import Settings, get_settings


def password_hash(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
    return f"scrypt${salt}${digest}"


def password_matches(password: str, encoded: str) -> bool:
    try:
        algorithm, salt, _ = encoded.split("$")
        return algorithm == "scrypt" and hmac.compare_digest(password_hash(password, salt), encoded)
    except (ValueError, TypeError):
        return False


_DUMMY_PASSWORD_HASH = password_hash("unusable-dummy-password")


class AuthStore:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin','user')),
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    digest TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    expires_at REAL NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id)
                );
                CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_user(self, username: str, password: str, role: str = "user") -> dict:
        user = {"id": uuid.uuid4().hex, "username": username, "role": role, "active": True}
        encoded = password_hash(password)
        with self.connect() as db:
            db.execute("INSERT INTO users VALUES (?,?,?,?,1)", (user["id"], username, encoded, role))
        return user

    def login(self, username: str, password: str, ttl: int) -> tuple[str, dict] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        # Run a real hash for unknown users too, avoiding a cheap username timing oracle.
        encoded = row["password_hash"] if row else _DUMMY_PASSWORD_HASH
        valid = password_matches(password, encoded)
        if not row or not row["active"] or not valid:
            return None
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE expires_at<=?", (time.time(),))
            db.execute("INSERT INTO sessions VALUES (?,?,?)", (self.digest(token), row["id"], time.time() + ttl))
        return token, {key: row[key] for key in ("id", "username", "role", "active")}

    @staticmethod
    def digest(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def authenticate(self, token: str) -> dict | None:
        if not token or len(token) > 256:
            return None
        with self.connect() as db:
            row = db.execute("""SELECT u.id,u.username,u.role,u.active FROM sessions s
                JOIN users u ON u.id=s.user_id
                WHERE s.digest=? AND s.expires_at>? AND u.active=1""", (self.digest(token), time.time())).fetchone()
        return dict(row) if row else None

    def logout(self, token: str):
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE digest=?", (self.digest(token),))

    def users(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT id,username,role,active FROM users ORDER BY username")]


def get_auth_store(settings: Settings) -> AuthStore:
    # Cache initialization, not connections. Each thread owns its own short-lived connection.
    return _cached_store(settings.auth_db_path)


_cached_store = lru_cache(maxsize=8)(AuthStore)


router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=1, max_length=128)


class CreateUserRequest(LoginRequest):
    password: str = Field(min_length=12, max_length=128)
    role: str = Field(default="user", pattern=r"^(admin|user)$")


@router.post("/login")
def login(body: LoginRequest, settings: Settings = Depends(get_settings)):
    result = get_auth_store(settings).login(body.username, body.password, settings.auth_session_ttl_seconds)
    if result is None:
        raise HTTPException(401, "Invalid username or password", headers={"WWW-Authenticate": "Bearer"})
    token, user = result
    return {"access_token": token, "token_type": "bearer", "expires_in": settings.auth_session_ttl_seconds, "user": user}


@router.get("/me")
def me(request: Request):
    return request.state.user


@router.post("/logout", status_code=204)
def logout(request: Request, settings: Settings = Depends(get_settings)):
    token = request.headers.get("authorization", "").partition(" ")[2]
    get_auth_store(settings).logout(token)


@router.post("/users", status_code=201)
def create_user(body: CreateUserRequest, settings: Settings = Depends(get_settings)):
    try:
        return get_auth_store(settings).create_user(body.username, body.password, body.role)
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Username already exists")


@router.get("/users")
def users(settings: Settings = Depends(get_settings)):
    return get_auth_store(settings).users()


def main():
    parser = argparse.ArgumentParser(description="Create an account without exposing passwords in shell history")
    parser.add_argument("username")
    parser.add_argument("--role", choices=["admin", "user"], default="admin")
    args = parser.parse_args()
    password = getpass.getpass("Password (at least 12 characters): ")
    if password != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords do not match")
    body = CreateUserRequest(username=args.username, password=password, role=args.role)
    try:
        user = get_auth_store(get_settings()).create_user(body.username, body.password, body.role)
    except sqlite3.IntegrityError:
        raise SystemExit("Username already exists; no changes made")
    print(f"Created {user['role']} account: {user['username']}")


if __name__ == "__main__":
    main()
