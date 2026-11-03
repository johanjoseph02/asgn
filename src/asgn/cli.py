"""``asgn`` command line.

asgn demo                 a seeded term with accounts for every role, served on :8200
asgn serve --db asgn.db   the app on an existing database
asgn study                the scheduling and effort-model studies behind docs/
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

DEMO_PASSWORD = "asgn-demo-password"


def seed_demo(path: str, today: date | None = None, seed: int = 3) -> list[str]:
    """A term five weeks old: past work logged, upcoming work published, later work in draft."""
    from asgn.sim import make_term
    from asgn.store import Store

    today = today or date.today()
    store = Store(path)
    term = make_term(seed)
    start = today - timedelta(days=35)
    rng = np.random.default_rng(seed)
    logins = []
    with store.tx() as db:
        db.execute(
            "INSERT INTO term (id, name, start, days, cap_hours) VALUES (1, ?, ?, ?, 3.0)",
            (f"Term starting {start:%d %B %Y}", start.isoformat(), term.n_days),
        )
        for c in term.cohorts:
            db.execute("INSERT INTO cohorts VALUES (?, ?, ?, ?)", (c.id, c.name, c.size, secrets.token_urlsafe(12)))
        for cid, course in term.courses.items():
            level = int(cid[1]) if cid.startswith("C") else 3
            db.execute("INSERT INTO courses VALUES (?, ?, ?, ?)", (cid, course.code, course.name, level))
            for coh in course.cohorts:
                db.execute("INSERT INTO course_cohorts VALUES (?, ?)", (cid, coh))
    day = 35
    for a in term.assignments:
        status = "published" if a.due < day + 21 else "draft"
        lo, hi = a.window or (a.due, a.due)
        if status == "published":
            lo = hi = a.due
        with store.tx() as db:
            db.execute(
                "INSERT INTO assignments (id, course_id, title, kind, weight_pct, size, release, due, est_p50, "
                "est_p80, status, window_lo, window_hi, preferred) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    a.id,
                    a.course_id,
                    a.title,
                    a.kind,
                    float(rng.choice([5, 10, 15])),
                    float(rng.integers(3, 10)),
                    a.release,
                    a.due,
                    a.hours,
                    round(a.hours * 1.45, 1),
                    status,
                    max(lo, day + 1) if status == "draft" else lo,
                    hi,
                    a.preferred,
                ),
            )
    store.add_user("department@demo.asgn", "Head of department", "admin", DEMO_PASSWORD)
    logins.append("department  department@demo.asgn")
    courses = list(term.courses)
    for i in range(0, len(courses), 2):
        email = f"instructor{i // 2 + 1}@demo.asgn"
        store.add_user(
            email, f"Instructor {i // 2 + 1}", "instructor", DEMO_PASSWORD, teaches=tuple(courses[i : i + 2])
        )
        logins.append(f"instructor  {email}  ({', '.join(term.courses[c].code for c in courses[i : i + 2])})")
    student_ids = {}
    for c in term.cohorts:
        email = f"student.{c.id.lower()}@demo.asgn"
        student_ids[c.id] = store.add_user(email, f"Student, {c.name}", "student", DEMO_PASSWORD, cohort=c.id)
        logins.append(f"student     {email}  ({c.name})")
    # classmates' logs for work already done, so calibration has something to learn from
    with store.tx() as db:
        for a in term.assignments:
            if a.due >= day:
                continue
            for coh in term.courses[a.course_id].cohorts:
                for k in range(int(rng.integers(8, 20))):
                    uid = db.execute(
                        "INSERT INTO users (email, name, role, cohort_id, password_hash) "
                        "VALUES (?, ?, 'student', ?, '!') ON CONFLICT (email) DO UPDATE SET name = name "
                        "RETURNING id",
                        (f"classmate{k}.{coh.lower()}@demo.asgn", "Classmate", coh),
                    ).fetchone()[0]
                    hours = round(a.hours * 1.15 * math.exp(rng.normal(0, 0.45)), 1)
                    db.execute(
                        "INSERT OR IGNORE INTO time_logs (assignment_id, user_id, hours) VALUES (?, ?, ?)",
                        (a.id, uid, max(hours, 0.25)),
                    )
    return logins


def _demo(a: argparse.Namespace) -> int:
    import uvicorn

    from asgn.store import Store
    from asgn.web import create_app

    path = a.db or str(Path(tempfile.mkdtemp(prefix="asgn-demo-")) / "asgn.db")
    logins = seed_demo(path)
    print(f"\n  Asgn: http://127.0.0.1:{a.port}\n  Password for every account: {DEMO_PASSWORD}\n")
    for line in logins:
        print(f"  {line}")
    print("\n  Ctrl-C to stop.\n", flush=True)
    uvicorn.run(create_app(Store(path)), host="127.0.0.1", port=a.port, log_level="warning")
    return 0


def _serve(a: argparse.Namespace) -> int:
    import uvicorn

    from asgn.store import Store
    from asgn.web import create_app

    uvicorn.run(create_app(Store(a.db), secure_cookies=a.secure), host=a.host, port=a.port, log_level="info")
    return 0


def _study(a: argparse.Namespace) -> int:
    from asgn.effort import evaluate
    from asgn.sim import study

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    effort = {str(s): evaluate(s) for s in range(a.seeds)}
    (out / "effort.json").write_text(json.dumps(effort, indent=2) + "\n")
    print(
        json.dumps(
            {s: {k: round(v["mae_hours"], 2) for k, v in e["models"].items()} for s, e in effort.items()}, indent=1
        )
    )
    sched = study(range(a.seeds), slack=a.slack)
    (out / "scheduling.json").write_text(json.dumps(sched, indent=2) + "\n")
    for p, s in sched["summary"].items():
        print(
            f"{p:10s} "
            + "  ".join(
                f"{k} {v['mean']:.2f}±{v['sd']:.2f}"
                for k, v in s.items()
                if k in ("excess_hours", "worst_day", "mean_peak", "mean_shift_days", "seconds")
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="asgn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="seeded term and accounts, served locally")
    d.add_argument("--port", type=int, default=8200)
    d.add_argument("--db", default="")
    d.set_defaults(func=_demo)
    s = sub.add_parser("serve", help="serve an existing database")
    s.add_argument("--db", default="asgn.db")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8200)
    s.add_argument("--secure", action="store_true", help="mark cookies Secure (behind HTTPS)")
    s.set_defaults(func=_serve)
    t = sub.add_parser("study", help="the studies behind docs/")
    t.add_argument("--seeds", type=int, default=10)
    t.add_argument("--slack", type=int, default=4)
    t.add_argument("--out", default="results")
    t.set_defaults(func=_study)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
