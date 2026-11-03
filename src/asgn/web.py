"""The web app: one view each for students, instructors and the department.

Server-rendered with plain HTML forms, so it works on any device and without
JavaScript. Sessions are HttpOnly SameSite=Lax cookies; form posts from other origins are
refused.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from asgn import charts
from asgn.effort import KINDS, Spec
from asgn.load import Assignment, daily_load
from asgn.service import (
    Estimator,
    advise_for,
    build_estimator,
    estimate,
    new_id,
    rebalance,
    term_load,
    today_index,
)
from asgn.store import Store

HERE = Path(__file__).parent
COOKIE = "asgn_session"
KIND_LABELS = {
    "problem_set": "Problem set",
    "lab_report": "Lab report",
    "essay": "Essay",
    "coding": "Coding",
    "reading_response": "Reading response",
    "project_milestone": "Project milestone",
}


def create_app(store: Store, *, secure_cookies: bool = False, today: date | None = None) -> FastAPI:
    app = FastAPI(title="Asgn", version="0.2.0")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(KIND_LABELS=KIND_LABELS, KINDS=KINDS)
    state: dict[str, Estimator] = {}

    def estimator() -> Estimator:
        if "e" not in state:
            state["e"] = build_estimator(store)
        return state["e"]

    def now_day() -> int:
        return today_index(store, today)

    def date_of(d: int) -> date:
        return store.date_of(d)

    def user(request: Request) -> dict[str, Any]:
        u = store.user_for(request.cookies.get(COOKIE))
        if u is None:
            raise HTTPException(303, headers={"Location": "/login"})
        return u

    def same_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "cross-site form post refused")

    def need(role: str) -> Any:
        def check(u: Annotated[dict[str, Any], Depends(user)]) -> dict[str, Any]:
            if u["role"] != role and u["role"] != "admin":
                raise HTTPException(403, "not available for your account")
            return u

        return check

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"term": store.term(), "date_of": date_of, **ctx})

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> Response:
        if exc.status_code == 303 and exc.headers:
            return RedirectResponse(exc.headers["Location"], status_code=303)
        return HTMLResponse(f"<p>{exc.detail}</p><p><a href='/'>Back</a></p>", status_code=exc.status_code)

    # ------------------------------------------------------------------ sign in

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, error: str = "") -> HTMLResponse:
        return page(request, "login.html", error=error)

    @app.post("/login", dependencies=[Depends(same_origin)])
    def login(email: Annotated[str, Form()], password: Annotated[str, Form()]) -> Response:
        token = store.login(email, password)
        if token is None:
            return RedirectResponse("/login?error=1", status_code=303)
        res = RedirectResponse("/", status_code=303)
        res.set_cookie(COOKIE, token, httponly=True, samesite="lax", secure=secure_cookies, max_age=30 * 86400)
        return res

    @app.post("/logout", dependencies=[Depends(same_origin)])
    def logout(request: Request) -> Response:
        if t := request.cookies.get(COOKIE):
            store.logout(t)
        res = RedirectResponse("/login", status_code=303)
        res.delete_cookie(COOKIE)
        return res

    @app.get("/")
    def home(u: Annotated[dict[str, Any], Depends(user)]) -> Response:
        return RedirectResponse(
            {"student": "/student", "instructor": "/instructor"}.get(u["role"], "/department"), status_code=303
        )

    # ------------------------------------------------------------------ students

    @app.get("/student", response_class=HTMLResponse)
    def student(request: Request, u: Annotated[dict[str, Any], Depends(user)]) -> HTMLResponse:
        cohort_id = u["cohort_id"] or store.cohorts()[0].id
        t = store.term()
        cohorts = store.cohorts()
        row = [c.id for c in cohorts].index(cohort_id)
        load = term_load(store, include_drafts=False)[row]
        day = now_day()
        first, last = day, min(day + 27, t["days"] - 1)
        mine = [
            r
            for r in store.q(
                "SELECT a.*, c.code, l.hours AS logged FROM assignments a JOIN courses c ON c.id = a.course_id "
                "JOIN course_cohorts cc ON cc.course_id = a.course_id AND cc.cohort_id = ? "
                "LEFT JOIN time_logs l ON l.assignment_id = a.id AND l.user_id = ? "
                "WHERE a.status = 'published' ORDER BY a.due",
                (cohort_id, u["id"]),
            )
        ]
        deadlines: dict[int, list[str]] = {}
        for a in mine:
            deadlines.setdefault(a["due"], []).append(f"{a['code']} {a['title']}")
        upcoming = [a for a in mine if a["due"] >= day][:12]
        to_log = [a for a in mine if a["due"] < day and a["logged"] is None and a["due"] >= day - 21]
        week = float(load[day : day + 7].sum())
        crunch = [d for d in range(first, last + 1) if load[d] > t["cap_hours"]]
        cohort = next(c for c in cohorts if c.id == cohort_id)
        feed = store.one("SELECT feed_token FROM cohorts WHERE id = ?", (cohort_id,))
        return page(
            request,
            "student.html",
            u=u,
            cohort=cohort,
            upcoming=upcoming,
            to_log=to_log,
            week=week,
            crunch=crunch,
            day=day,
            feed=f"/feed/{cohort_id}/{feed['feed_token'] if feed else ''}.ics",
            chart=charts.bars(load, first, last, t["cap_hours"], date_of, deadlines, day),
        )

    @app.post("/log", dependencies=[Depends(same_origin)])
    def log_hours(
        u: Annotated[dict[str, Any], Depends(user)],
        assignment: Annotated[str, Form()],
        hours: Annotated[float, Form(gt=0, le=200)],
    ) -> Response:
        ok = store.one(
            "SELECT 1 FROM assignments a JOIN course_cohorts cc ON cc.course_id = a.course_id "
            "WHERE a.id = ? AND cc.cohort_id = ?",
            (assignment, u["cohort_id"]),
        )
        if ok is None:
            raise HTTPException(404, "no such assignment for your cohort")
        with store.tx() as db:
            db.execute(
                "INSERT INTO time_logs (assignment_id, user_id, hours) VALUES (?, ?, ?) "
                "ON CONFLICT (assignment_id, user_id) DO UPDATE SET hours = excluded.hours",
                (assignment, u["id"], hours),
            )
        state.pop("e", None)  # recalibrate with the new evidence
        return RedirectResponse("/student", status_code=303)

    @app.get("/feed/{cohort_id}/{token}.ics")
    def feed(cohort_id: str, token: str) -> Response:
        row = store.one("SELECT feed_token, name FROM cohorts WHERE id = ?", (cohort_id,))
        if row is None or row["feed_token"] != token:
            raise HTTPException(404, "no such calendar")
        rows = store.q(
            "SELECT a.id, a.title, a.due, a.est_p50, a.est_p80, c.code FROM assignments a "
            "JOIN courses c ON c.id = a.course_id JOIN course_cohorts cc ON cc.course_id = a.course_id "
            "WHERE cc.cohort_id = ? AND a.status = 'published'",
            (cohort_id,),
        )
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//asgn//EN", f"X-WR-CALNAME:Deadlines, {row['name']}"]
        for a in rows:
            d = date_of(a["due"])
            lines += [
                "BEGIN:VEVENT",
                f"UID:{a['id']}@asgn",
                f"DTSTAMP:{date.today():%Y%m%d}T000000Z",
                f"DTSTART;VALUE=DATE:{d:%Y%m%d}",
                f"DTEND;VALUE=DATE:{d + timedelta(days=1):%Y%m%d}",
                f"SUMMARY:Due: {a['code']} {a['title']}",
                f"DESCRIPTION:Expected {a['est_p50']:.0f} to {a['est_p80']:.0f} hours",
                "END:VEVENT",
            ]
        lines.append("END:VCALENDAR")
        return Response("\r\n".join(lines) + "\r\n", media_type="text/calendar")

    # ------------------------------------------------------------------ instructors

    def my_courses(u: dict[str, Any]) -> list[Any]:
        if u["role"] == "admin":
            return store.q("SELECT * FROM courses ORDER BY code")
        return store.q(
            "SELECT c.* FROM courses c JOIN teaches t ON t.course_id = c.id WHERE t.user_id = ? ORDER BY c.code",
            (u["id"],),
        )

    @app.get("/instructor", response_class=HTMLResponse)
    def instructor(
        request: Request,
        u: Annotated[dict[str, Any], Depends(need("instructor"))],
        course: str = "",
        kind: str = "problem_set",
        weight: float = 5.0,
        size: float = 8.0,
        days: int = 10,
        preferred: str = "",
        flex: int = 4,
        title: str = "",
    ) -> HTMLResponse:
        courses = my_courses(u)
        if not courses:
            return page(request, "instructor.html", u=u, courses=[], course=None)
        c = next((x for x in courses if x["id"] == course), courses[0])
        t = store.term()
        day = now_day()
        items = [r for r in store.q("SELECT * FROM assignments WHERE course_id = ? ORDER BY due", (c["id"],))]
        plan: dict[str, Any] | None = None
        if preferred:
            pref = store.day_of(date.fromisoformat(preferred))
            if kind not in KINDS:
                raise HTTPException(422, "unknown kind of assignment")
            p50, p80, factor = estimate(estimator(), Spec(kind, weight, size, c["level"], days), c["id"])
            lo, hi = max(pref - flex, day + 1), min(pref + flex, t["days"] - 1)
            new = Assignment(
                new_id(),
                c["id"],
                title or "New assignment",
                pref - days,
                pref,
                p50,
                p80,
                window=(lo, hi),
                preferred=pref,
                kind=kind,
            )
            options, base = advise_for(store, new, top=hi - lo + 1)
            cohorts = store.cohorts()
            members = store.courses()[c["id"]].cohorts
            rows = [i for i, co in enumerate(cohorts) if co.id in members]
            plan = {
                "p50": p50,
                "p80": p80,
                "factor": factor,
                "options": options,
                "pref": pref,
                "best": options[0] if options else None,
                "spec": {"kind": kind, "weight": weight, "size": size, "days": days, "title": title},
                "chart": charts.heatmap(
                    base[rows],
                    [cohorts[i].name for i in rows],
                    max(lo - 10, 0),
                    min(hi + 10, t["days"] - 1),
                    t["cap_hours"],
                    date_of,
                    day,
                    16,
                ),
                "source": estimator().source,
            }
        return page(
            request,
            "instructor.html",
            u=u,
            courses=courses,
            course=c,
            items=items,
            plan=plan,
            day=day,
            form={
                "kind": kind,
                "weight": weight,
                "size": size,
                "days": days,
                "flex": flex,
                "title": title,
                "preferred": preferred or (date_of(min(day + 14, t["days"] - 1))).isoformat(),
            },
        )

    @app.post("/instructor/assignments", dependencies=[Depends(same_origin)])
    def create_assignment(
        u: Annotated[dict[str, Any], Depends(need("instructor"))],
        course: Annotated[str, Form()],
        title: Annotated[str, Form(min_length=2, max_length=120)],
        kind: Annotated[str, Form()],
        weight: Annotated[float, Form(ge=0, le=100)],
        size: Annotated[float, Form(gt=0, le=1000)],
        days: Annotated[int, Form(ge=1, le=60)],
        due: Annotated[int, Form()],
        status: Annotated[str, Form(pattern="^(draft|published)$")],
        lo: Annotated[int, Form()],
        hi: Annotated[int, Form()],
    ) -> Response:
        if course not in {c["id"] for c in my_courses(u)} or kind not in KINDS:
            raise HTTPException(403, "not your course")
        level = store.one("SELECT level FROM courses WHERE id = ?", (course,))
        p50, p80, _ = estimate(estimator(), Spec(kind, weight, size, level["level"] if level else 1, days), course)
        with store.tx() as db:
            db.execute(
                "INSERT INTO assignments (id, course_id, title, kind, weight_pct, size, release, due, est_p50, "
                "est_p80, status, window_lo, window_hi, preferred) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_id(), course, title, kind, weight, size, due - days, due, p50, p80, status, lo, hi, due),
            )
        return RedirectResponse(f"/instructor?course={course}", status_code=303)

    @app.post("/instructor/assignments/{aid}/publish", dependencies=[Depends(same_origin)])
    def publish(aid: str, u: Annotated[dict[str, Any], Depends(need("instructor"))]) -> Response:
        row = store.one("SELECT course_id FROM assignments WHERE id = ?", (aid,))
        if row is None or row["course_id"] not in {c["id"] for c in my_courses(u)}:
            raise HTTPException(404, "no such assignment")
        with store.tx() as db:
            db.execute("UPDATE assignments SET status = 'published' WHERE id = ?", (aid,))
        return RedirectResponse(f"/instructor?course={row['course_id']}", status_code=303)

    @app.post("/instructor/assignments/{aid}/delete", dependencies=[Depends(same_origin)])
    def delete(aid: str, u: Annotated[dict[str, Any], Depends(need("instructor"))]) -> Response:
        row = store.one("SELECT course_id, status FROM assignments WHERE id = ?", (aid,))
        if row is None or row["course_id"] not in {c["id"] for c in my_courses(u)}:
            raise HTTPException(404, "no such assignment")
        if row["status"] != "draft":
            raise HTTPException(409, "published assignments cannot be deleted here")
        with store.tx() as db:
            db.execute("DELETE FROM assignments WHERE id = ?", (aid,))
        return RedirectResponse(f"/instructor?course={row['course_id']}", status_code=303)

    # ------------------------------------------------------------------ department

    @app.get("/department", response_class=HTMLResponse)
    def department(
        request: Request, u: Annotated[dict[str, Any], Depends(need("admin"))], preview: int = 0
    ) -> HTMLResponse:
        t = store.term()
        cohorts = store.cohorts()
        load = term_load(store)
        day = now_day()
        weekly = [[float(load[r, s : s + 7].sum()) for s in range(0, t["days"], 7)] for r in range(len(cohorts))]
        crunch = sorted(
            (
                (float(load[r, d]), cohorts[r].name, d)
                for r in range(len(cohorts))
                for d in range(day, t["days"])
                if load[r, d] > t["cap_hours"]
            ),
            reverse=True,
        )[:12]
        drafts = store.q(
            "SELECT a.*, c.code FROM assignments a JOIN courses c ON c.id = a.course_id "
            "WHERE a.status = 'draft' ORDER BY a.due"
        )
        proposal = rebalance(store) if preview else None
        est = estimator()
        return page(
            request,
            "department.html",
            u=u,
            cohorts=cohorts,
            weekly=weekly,
            crunch=crunch,
            day=day,
            drafts=drafts,
            proposal=proposal,
            est=est,
            chart=charts.heatmap(load, [c.name for c in cohorts], 0, t["days"] - 1, t["cap_hours"], date_of, day, 9),
        )

    @app.post("/department/rebalance", dependencies=[Depends(same_origin)])
    def do_rebalance(u: Annotated[dict[str, Any], Depends(need("admin"))]) -> Response:
        rebalance(store, apply_changes=True)
        return RedirectResponse("/department", status_code=303)

    @app.get("/api/load")
    def api_load(u: Annotated[dict[str, Any], Depends(user)]) -> dict[str, Any]:
        t = store.term()
        load = daily_load(store.assignments("published"), store.courses(), store.cohorts(), t["days"])
        return {
            "start": t["start"],
            "cap_hours": t["cap_hours"],
            "cohorts": {c.id: [round(float(x), 2) for x in load[i]] for i, c in enumerate(store.cohorts())},
        }

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        store.term()
        return {"ok": True}

    return app
