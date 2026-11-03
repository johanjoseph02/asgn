"""What the web app asks for, computed from the store."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asgn.effort import CourseCalibration, EffortModel, Spec, synthetic_logs
from asgn.load import Assignment, Profile, daily_load, summarise
from asgn.schedule import Option, advise, apply, improve
from asgn.store import Store

MIN_REAL_LOGS = 300


@dataclass
class Estimator:
    model: EffortModel
    calibration: CourseCalibration
    source: str  # "this department's logs" or "built-in prior"
    logs: int


def build_estimator(store: Store) -> Estimator:
    """Train on the department's own logs when there are enough; otherwise on the
    synthetic prior, which is said plainly in the interface."""
    rows = store.q(
        "SELECT a.course_id, a.kind, a.weight_pct, a.size, c.level, a.due - a.release AS days, t.hours, "
        "a.id, a.est_p50 FROM time_logs t JOIN assignments a ON a.id = t.assignment_id "
        "JOIN courses c ON c.id = a.course_id"
    )
    if len(rows) >= MIN_REAL_LOGS:
        X = np.array([Spec(r["kind"], r["weight_pct"], r["size"], r["level"], r["days"]).vector() for r in rows])
        y = np.array([r["hours"] for r in rows])
        model, source = EffortModel().fit(X, y), "this department's logs"
    else:
        data = synthetic_logs(0)
        X = np.array([a.spec.vector() for a in data for _ in a.hours])
        y = np.array([h for a in data for h in a.hours])
        model, source = EffortModel().fit(X, y), "built-in prior (synthetic)"
    cal = CourseCalibration()
    by_assignment: dict[str, tuple[str, float, list[float]]] = {}
    for r in rows:
        by_assignment.setdefault(r["id"], (r["course_id"], r["est_p50"], []))[2].append(r["hours"])
    for course, est, hours in by_assignment.values():
        cal.observe(course, est, hours)
    return Estimator(model, cal, source, len(rows))


def estimate(est: Estimator, spec: Spec, course: str) -> tuple[float, float, float]:
    """Median and 80th-percentile hours, and the course factor applied."""
    lo, hi = est.model.predict([spec])
    f = est.calibration.factor(course)
    return round(float(lo[0]) * f, 1), round(float(hi[0]) * f, 1), round(f, 2)


def today_index(store: Store, today: date | None = None) -> int:
    t = store.term()
    return int(np.clip(store.day_of(today or date.today()), 0, t["days"] - 1))


def term_load(store: Store, include_drafts: bool = True) -> NDArray[np.float64]:
    t = store.term()
    items = store.assignments() if include_drafts else store.assignments("published")
    return daily_load(items, store.courses(), store.cohorts(), t["days"], Profile())


def advise_for(store: Store, new: Assignment, top: int = 5) -> tuple[list[Option], NDArray[np.float64]]:
    t = store.term()
    others = [a for a in store.assignments() if a.id != new.id]
    options = advise(new, others, store.courses(), store.cohorts(), t["days"], t["cap_hours"], top=top)
    return options, daily_load(others, store.courses(), store.cohorts(), t["days"], Profile())


def rebalance(store: Store, apply_changes: bool = False) -> dict[str, Any]:
    """Re-plan every draft assignment's due date inside its window; published ones stay."""
    t = store.term()
    courses, cohorts = store.courses(), store.cohorts()
    current = store.assignments()
    pinned = [replace(a, window=None) if a.meta.get("status") == "published" else a for a in current]
    planned = improve(pinned, courses, cohorts, t["days"], t["cap_hours"])
    before = summarise(daily_load(current, courses, cohorts, t["days"]), cohorts, t["cap_hours"])
    after = summarise(daily_load(planned, courses, cohorts, t["days"]), cohorts, t["cap_hours"])
    moves = [(a, p) for a, p in zip(current, planned, strict=True) if a.due != p.due]
    if apply_changes and moves:
        with store.tx() as db:
            for a, p in moves:
                db.execute(
                    "UPDATE assignments SET due = ?, release = ? WHERE id = ? AND status = 'draft'",
                    (p.due, p.release, a.id),
                )
    return {
        "before": before.as_dict(),
        "after": after.as_dict(),
        "moves": [
            {"id": a.id, "title": f"{courses[a.course_id].code} · {a.title}", "from": a.due, "to": p.due}
            for a, p in moves
        ],
    }


def new_id() -> str:
    return secrets.token_hex(6)


def to_moved(a: Assignment, due: int) -> Assignment:
    return apply({a.id: due}, [a])[0]
