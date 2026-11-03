"""SQLite storage for one department's term: cohorts, courses, people, assignments, logs."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Any

from asgn.load import Assignment, Cohort, Course

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS term (
    id INTEGER PRIMARY KEY CHECK (id = 1), name TEXT NOT NULL, start TEXT NOT NULL,
    days INTEGER NOT NULL, cap_hours REAL NOT NULL DEFAULT 3.0
);
CREATE TABLE IF NOT EXISTS cohorts (id TEXT PRIMARY KEY, name TEXT NOT NULL, size INTEGER NOT NULL,
    feed_token TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS courses (id TEXT PRIMARY KEY, code TEXT NOT NULL, name TEXT NOT NULL,
    level INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS course_cohorts (course_id TEXT NOT NULL REFERENCES courses(id),
    cohort_id TEXT NOT NULL REFERENCES cohorts(id), PRIMARY KEY (course_id, cohort_id));
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE COLLATE NOCASE, name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'instructor', 'student')),
    cohort_id TEXT REFERENCES cohorts(id), password_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS teaches (user_id INTEGER NOT NULL REFERENCES users(id),
    course_id TEXT NOT NULL REFERENCES courses(id), PRIMARY KEY (user_id, course_id));
CREATE TABLE IF NOT EXISTS assignments (
    id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id), title TEXT NOT NULL,
    kind TEXT NOT NULL, weight_pct REAL NOT NULL, size REAL NOT NULL,
    release INTEGER NOT NULL, due INTEGER NOT NULL CHECK (due >= release),
    est_p50 REAL NOT NULL, est_p80 REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'published')),
    window_lo INTEGER, window_hi INTEGER, preferred INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS time_logs (
    assignment_id TEXT NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id), hours REAL NOT NULL CHECK (hours > 0 AND hours <= 200),
    logged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (assignment_id, user_id)
);
CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
    expires TEXT NOT NULL);
"""


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${key.hex()}"


def check_password(pw: str, stored: str) -> bool:
    try:
        _, salt, key = stored.split("$")
    except ValueError:
        return False
    got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1, dklen=32)
    return hmac.compare_digest(got.hex(), key)


class Store:
    def __init__(self, path: str) -> None:
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.db.execute("BEGIN")
            try:
                yield self.db
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def q(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Row | None:
        rows = self.q(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ term

    def term(self) -> dict[str, Any]:
        row = self.one("SELECT * FROM term WHERE id = 1")
        if row is None:
            raise LookupError("no term set up; run `asgn demo` or `asgn init`")
        return dict(row)

    def date_of(self, day: int) -> date:
        return date.fromisoformat(self.term()["start"]) + timedelta(days=day)

    def day_of(self, d: date) -> int:
        return (d - date.fromisoformat(self.term()["start"])).days

    def cohorts(self) -> list[Cohort]:
        return [Cohort(r["id"], r["name"], r["size"]) for r in self.q("SELECT * FROM cohorts ORDER BY id")]

    def courses(self) -> dict[str, Course]:
        members: dict[str, list[str]] = {}
        for r in self.q("SELECT * FROM course_cohorts ORDER BY cohort_id"):
            members.setdefault(r["course_id"], []).append(r["cohort_id"])
        return {
            r["id"]: Course(r["id"], r["code"], r["name"], tuple(members.get(r["id"], [])))
            for r in self.q("SELECT * FROM courses ORDER BY code")
        }

    def assignments(self, status: str | None = None) -> list[Assignment]:
        rows = self.q(
            "SELECT * FROM assignments" + (" WHERE status = ?" if status else "") + " ORDER BY due, id",
            (status,) if status else (),
        )
        return [to_assignment(r) for r in rows]

    # ------------------------------------------------------------------ people

    def login(self, email: str, password: str) -> str | None:
        row = self.one("SELECT id, password_hash FROM users WHERE email = ?", (email.strip(),))
        if row is None or not check_password(password, row["password_hash"]):
            return None
        token = secrets.token_urlsafe(32)
        with self.tx() as db:
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, datetime('now', '+30 days'))",
                (hashlib.sha256(token.encode()).hexdigest(), row["id"]),
            )
        return token

    def user_for(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        row = self.one(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires > datetime('now')",
            (hashlib.sha256(token.encode()).hexdigest(),),
        )
        # iterating a Row yields values, so the keys are needed explicitly
        return None if row is None else {k: row[k] for k in row.keys() if k != "password_hash"}  # noqa: SIM118

    def logout(self, token: str) -> None:
        with self.tx() as db:
            db.execute("DELETE FROM sessions WHERE token_hash = ?", (hashlib.sha256(token.encode()).hexdigest(),))

    def add_user(
        self, email: str, name: str, role: str, password: str, cohort: str | None = None, teaches: tuple[str, ...] = ()
    ) -> int:
        with self.tx() as db:
            cur = db.execute(
                "INSERT INTO users (email, name, role, cohort_id, password_hash) VALUES (?, ?, ?, ?, ?)",
                (email, name, role, cohort, hash_password(password)),
            )
            uid = int(cur.lastrowid or 0)
            for c in teaches:
                db.execute("INSERT INTO teaches VALUES (?, ?)", (uid, c))
        return uid


def to_assignment(r: sqlite3.Row) -> Assignment:
    window = (r["window_lo"], r["window_hi"]) if r["window_lo"] is not None and r["window_hi"] is not None else None
    return Assignment(
        id=r["id"],
        course_id=r["course_id"],
        title=r["title"],
        release=r["release"],
        due=r["due"],
        hours=r["est_p50"],
        hours_p80=r["est_p80"],
        window=window,
        preferred=r["preferred"],
        kind=r["kind"],
        meta={"status": r["status"], "weight_pct": r["weight_pct"], "size": r["size"]},
    )
