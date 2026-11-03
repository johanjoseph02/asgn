from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from asgn.cli import DEMO_PASSWORD, seed_demo
from asgn.store import Store
from asgn.web import create_app

TODAY = date(2030, 10, 15)


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> tuple[TestClient, Store]:
    path = str(Path(tmp_path_factory.mktemp("asgn")) / "t.db")
    seed_demo(path, today=TODAY)
    store = Store(path)
    return TestClient(create_app(store, today=TODAY)), store


def signed_in(app: tuple[TestClient, Store], email: str) -> TestClient:
    client = TestClient(app[0].app)
    r = client.post("/login", data={"email": email, "password": DEMO_PASSWORD}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    return client


def test_sign_in_and_roles(app: tuple[TestClient, Store]) -> None:
    anon = TestClient(app[0].app)
    assert anon.get("/student", follow_redirects=False).headers["location"] == "/login"
    bad = anon.post("/login", data={"email": "department@demo.asgn", "password": "nope"}, follow_redirects=False)
    assert bad.headers["location"] == "/login?error=1"
    student = signed_in(app, "student.y2a@demo.asgn")
    assert student.get("/", follow_redirects=False).headers["location"] == "/student"
    assert student.get("/department").status_code == 403
    assert student.get("/instructor").status_code == 403
    r = student.post("/log", data={"assignment": "x", "hours": 2}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_student_sees_load_logs_hours_and_gets_a_feed(app: tuple[TestClient, Store]) -> None:
    client, store = app
    student = signed_in(app, "student.y2a@demo.asgn")
    page = student.get("/student")
    assert page.status_code == 200 and "<svg" in page.text and "Year 2, section A" in page.text
    past = store.one(
        "SELECT a.id FROM assignments a JOIN course_cohorts cc ON cc.course_id = a.course_id "
        "WHERE cc.cohort_id = 'Y2A' AND a.status = 'published' AND a.due < 35 ORDER BY a.due DESC"
    )
    assert past is not None
    assert (
        student.post("/log", data={"assignment": past["id"], "hours": 3.5}, follow_redirects=False).status_code == 303
    )
    assert student.post("/log", data={"assignment": past["id"], "hours": -1}).status_code == 422
    token = store.one("SELECT feed_token FROM cohorts WHERE id = 'Y2A'")["feed_token"]
    ics = client.get(f"/feed/Y2A/{token}.ics")
    assert ics.headers["content-type"].startswith("text/calendar") and "BEGIN:VEVENT" in ics.text
    assert client.get("/feed/Y2A/wrong.ics").status_code == 404


def test_instructor_plans_and_publishes(app: tuple[TestClient, Store]) -> None:
    _, store = app
    teacher = signed_in(app, "instructor3@demo.asgn")
    page = teacher.get(
        "/instructor",
        params={
            "course": "C20",
            "title": "Problem set 9",
            "kind": "problem_set",
            "size": 8,
            "weight": 5,
            "days": 10,
            "preferred": "2030-11-20",
            "flex": 3,
        },
    )
    assert page.status_code == 200 and "Due dates, best first" in page.text and "your choice" in page.text
    before = len(store.q("SELECT 1 FROM assignments WHERE course_id = 'C20'"))
    day = store.day_of(date(2030, 11, 21))
    r = teacher.post(
        "/instructor/assignments",
        data={
            "course": "C20",
            "title": "Problem set 9",
            "kind": "problem_set",
            "weight": 5,
            "size": 8,
            "days": 10,
            "due": day,
            "status": "draft",
            "lo": day - 3,
            "hi": day + 3,
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert len(store.q("SELECT 1 FROM assignments WHERE course_id = 'C20'")) == before + 1
    # someone else's course
    r = teacher.post(
        "/instructor/assignments",
        data={
            "course": "C10",
            "title": "Nope",
            "kind": "essay",
            "weight": 5,
            "size": 3,
            "days": 7,
            "due": day,
            "status": "published",
            "lo": day,
            "hi": day,
        },
    )
    assert r.status_code == 403


def test_department_previews_and_applies_a_rebalance(app: tuple[TestClient, Store]) -> None:
    _, store = app
    head = signed_in(app, "department@demo.asgn")
    published = {r["id"]: r["due"] for r in store.q("SELECT id, due FROM assignments WHERE status = 'published'")}
    page = head.get("/department", params={"preview": 1})
    assert page.status_code == 200 and "Rebalancing would move" in page.text
    assert head.post("/department/rebalance", follow_redirects=False).status_code == 303
    after = {r["id"]: r["due"] for r in store.q("SELECT id, due FROM assignments WHERE status = 'published'")}
    assert after == published  # published deadlines never move
    for r in store.q("SELECT due, window_lo, window_hi FROM assignments WHERE status = 'draft'"):
        assert r["window_lo"] <= r["due"] <= r["window_hi"]
    assert head.get("/api/load").json()["cap_hours"] == 3.0
