from __future__ import annotations

import numpy as np
import pytest

from asgn.effort import CourseCalibration, EffortModel, Spec, evaluate, synthetic_logs
from asgn.load import Assignment, Cohort, Course, Profile, daily_load, summarise, work_profile
from asgn.schedule import advise, apply, improve, objective, plan
from asgn.sim import make_term, run_policy

COHORTS = [Cohort("A", "Cohort A", 50), Cohort("B", "Cohort B", 30)]
COURSES = {
    "X": Course("X", "X101", "X", ("A",)),
    "Y": Course("Y", "Y101", "Y", ("A", "B")),
    "Z": Course("Z", "Z101", "Z", ("B",)),
}


def a(
    id_: str, course: str, due: int, hours: float, window: tuple[int, int] | None = None, length: int = 7
) -> Assignment:
    return Assignment(id_, course, id_, due - length, due, hours, window=window, preferred=due)


def test_work_profile_sums_to_one_and_leans_late() -> None:
    start, w = work_profile(0, 9)
    assert start == 0 and w.sum() == pytest.approx(1)
    assert np.all(np.diff(w) > 0)
    start, w = work_profile(0, 30, Profile(horizon=5))
    assert start == 26 and len(w) == 5


def test_daily_load_lands_on_every_cohort_of_a_course() -> None:
    load = daily_load([a("y", "Y", 10, 6), a("x", "X", 10, 4)], COURSES, COHORTS, 20)
    assert load[0].sum() == pytest.approx(10)
    assert load[1].sum() == pytest.approx(6)
    assert load[:, 11:].sum() == 0
    s = summarise(load, COHORTS, cap=2)
    assert s.worst_day == pytest.approx(load.max())
    assert s.overloaded_student_days == int((load[0] > 2).sum()) * 50 + int((load[1] > 2).sum()) * 30


def test_advisor_steers_away_from_a_crowded_day() -> None:
    others = [a("x1", "X", 10, 8), a("z1", "Z", 10, 8)]
    new = a("y1", "Y", 10, 6, window=(6, 14))
    best = advise(new, others, COURSES, COHORTS, 30, cap=2)[0]
    assert best.due != 10
    assert (
        best.added_excess
        < advise(a("y1", "Y", 10, 6, window=(10, 10)), others, COURSES, COHORTS, 30, cap=2)[0].added_excess
    )


def test_advisor_prefers_the_instructors_date_when_it_costs_nothing() -> None:
    best = advise(a("y1", "Y", 10, 1, window=(6, 14)), [], COURSES, COHORTS, 30, cap=3)[0]
    assert best.due == 10 and best.shift == 0


def test_plan_is_exact_on_a_small_problem() -> None:
    items = [a(f"y{i}", "Y", 10, 5, window=(6, 16)) for i in range(3)] + [a("x", "X", 12, 6)]
    res = plan(items, COURSES, COHORTS, 30, cap=2)
    assert res.status == "optimal" and res.gap == pytest.approx(0, abs=1e-6)
    assert res.due["x"] == 12  # not movable
    assert len({res.due[f"y{i}"] for i in range(3)}) == 3  # spread out
    planned = apply(res.due, items)
    # local search can only match or trail the exact optimum
    heur = improve(items, COURSES, COHORTS, 30, cap=2)
    assert objective(planned, COURSES, COHORTS, 30, 2) <= objective(heur, COURSES, COHORTS, 30, 2) + 1e-6
    assert res.objective == pytest.approx(objective(planned, COURSES, COHORTS, 30, 2), rel=1e-6)


def test_apply_moves_release_with_due() -> None:
    moved = apply({"x": 15}, [a("x", "X", 10, 3, length=6)])[0]
    assert (moved.release, moved.due) == (9, 15)


def test_simulated_terms_and_policies() -> None:
    term = make_term(0)
    assert 80 < len(term.assignments) < 120
    assert all(x.window and x.window[0] <= x.due <= x.window[1] for x in term.assignments)
    pref = run_policy(term, "preferred", 0)
    planner = run_policy(term, "planner", 0)
    assert planner["excess_hours"] < pref["excess_hours"]
    assert planner["mean_shift_days"] <= 4  # never outside the window
    assert make_term(0).assignments == make_term(0).assignments


def test_effort_model_and_calibration() -> None:
    data = synthetic_logs(0, terms=1, courses=8, students=20)
    X = np.array([d.spec.vector() for d in data for _ in d.hours])
    y = np.array([h for d in data for h in d.hours])
    m = EffortModel().fit(X, y)
    lo, hi = m.predict([Spec("essay", 10, 5, 2, 10), Spec("reading_response", 2, 3, 1, 7)])
    assert np.all(hi >= lo) and lo[0] > lo[1]
    cal = CourseCalibration()
    assert cal.factor("K") == 1.0
    cal.observe("K", 2.0, [4.0, 4.0, 4.0])
    assert 1.0 < cal.factor("K") < 2.0  # moves towards the evidence, shrunk while evidence is thin
    with pytest.raises(ValueError):
        Spec("haiku", 1, 1, 1, 1).vector()


def test_effort_evaluation_beats_the_kind_average() -> None:
    res = evaluate(1)["models"]
    assert res["quantile model + course calibration"]["mae_hours"] < res["by kind (median of past logs)"]["mae_hours"]
