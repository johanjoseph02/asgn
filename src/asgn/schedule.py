"""Choosing due dates so coursework does not pile up.

Two tools share one objective: as few hours above the daily cap as possible across every
cohort (weighted by how many students are in it), then the lowest daily peaks, then the
smallest moves away from each instructor's preferred date.

* ``advise``: one instructor, one assignment. Every allowed due date is scored against
  the load already planned; the best few are returned with what each would do.
* ``plan``: every movable assignment in a term at once, as a mixed-integer linear program
  solved exactly with HiGHS (via ``scipy.optimize.milp``). Each assignment picks one due
  date inside its window; the load each choice adds to each cohort on each day is fixed,
  so the load is linear in the choices and the overload above the cap is linearised with
  one slack variable per cohort-day.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from asgn.load import DEFAULT_CAP, DEFAULT_PROFILE, Assignment, Cohort, Course, Profile, daily_load, work_profile

W_PEAK = 0.05  # per hour of each cohort's peak day, relative to one weighted excess hour
W_SHIFT = 0.01  # per day moved from the preferred date


@dataclass(frozen=True)
class Option:
    due: int
    added_excess: float  # weighted hours above the cap this choice adds
    peak: float  # the highest daily load among the cohorts this assignment lands on
    shift: int


def _contribution(
    a: Assignment,
    due: int,
    courses: Mapping[str, Course],
    cohort_index: Mapping[str, int],
    n_days: int,
    profile: Profile,
) -> list[tuple[int, int, float]]:
    """(cohort row, day, hours) triples that ``a`` adds if due on ``due``."""
    length = a.due - a.release
    start, share = work_profile(due - length, due, profile)
    out = []
    for cid in courses[a.course_id].cohorts:
        row = cohort_index.get(cid)
        if row is None:
            continue
        for k, h in enumerate(share):
            day = start + k
            if 0 <= day < n_days:
                out.append((row, day, a.hours * float(h)))
    return out


def advise(
    new: Assignment,
    others: Sequence[Assignment],
    courses: Mapping[str, Course],
    cohorts: Sequence[Cohort],
    n_days: int,
    cap: float = DEFAULT_CAP,
    profile: Profile = DEFAULT_PROFILE,
    top: int = 5,
) -> list[Option]:
    """Rank every allowed due date for ``new``. The release date moves with the due date,
    keeping the time students have."""
    base = daily_load(others, courses, cohorts, n_days, profile)
    index = {c.id: i for i, c in enumerate(cohorts)}
    size = np.array([c.size for c in cohorts], dtype=float)
    lo, hi = new.window if new.window else (new.due, new.due)
    preferred = new.preferred if new.preferred is not None else new.due
    rows = [index[c] for c in courses[new.course_id].cohorts if c in index]
    before = (np.clip(base - cap, 0, None) * size[:, None]).sum()
    options = []
    for d in range(max(lo, 0), min(hi, n_days - 1) + 1):
        load = base.copy()
        for r, day, h in _contribution(new, d, courses, index, n_days, profile):
            load[r, day] += h
        after = (np.clip(load - cap, 0, None) * size[:, None]).sum()
        peak = float(load[rows].max()) if rows else 0.0
        options.append(Option(d, float(after - before), peak, abs(d - preferred)))
    if not options:  # window entirely outside the term: nothing to choose
        return [Option(new.due, 0.0, float(base[rows].max()) if rows else 0.0, 0)]
    options.sort(key=lambda o: (round(o.added_excess, 3), round(o.peak, 3), o.shift, o.due))
    return options[:top]


@dataclass(frozen=True)
class PlanResult:
    due: dict[str, int]
    status: str
    objective: float
    seconds: float
    gap: float  # proven distance from the optimum, as a fraction of the objective (0 = optimal)
    bound: float  # no plan can have a lower objective than this


def plan(
    assignments: Sequence[Assignment],
    courses: Mapping[str, Course],
    cohorts: Sequence[Cohort],
    n_days: int,
    cap: float = DEFAULT_CAP,
    profile: Profile = DEFAULT_PROFILE,
    time_limit: float = 60.0,
) -> PlanResult:
    """Pick a due date for every movable assignment; fixed ones stay where they are."""
    import time

    t0 = time.perf_counter()
    index = {c.id: i for i, c in enumerate(cohorts)}
    n_c = len(cohorts)
    size = np.array([c.size for c in cohorts], dtype=float)
    fixed = [a for a in assignments if not a.movable()]
    movable = [a for a in assignments if a.movable()]
    base = daily_load(fixed, courses, cohorts, n_days, profile)

    # variables: x (one per assignment x candidate day), then o (n_c x n_days), then peak (n_c)
    choices: list[tuple[int, int]] = []  # (assignment index, due day)
    for i, a in enumerate(movable):
        assert a.window is not None
        for d in range(max(a.window[0], 0), min(a.window[1], n_days - 1) + 1):
            choices.append((i, d))
    nx = len(choices)
    o0 = nx
    p0 = o0 + n_c * n_days
    nvar = p0 + n_c

    cost = np.zeros(nvar)
    for k, (i, d) in enumerate(choices):
        a = movable[i]
        pref = a.preferred if a.preferred is not None else a.due
        cost[k] = W_SHIFT * abs(d - pref)
    cost[o0:p0] = np.repeat(size, n_days)
    cost[p0:] = W_PEAK * size

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    r = 0
    # each movable assignment takes exactly one due date
    for i in range(len(movable)):
        for k, (ii, _) in enumerate(choices):
            if ii == i:
                rows.append(r)
                cols.append(k)
                vals.append(1.0)
        lb.append(1.0)
        ub.append(1.0)
        r += 1
    # load[c, t] = base + sum x * contribution; then load - o <= cap and load - peak <= 0
    contrib: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for k, (i, d) in enumerate(choices):
        for row, day, h in _contribution(movable[i], d, courses, index, n_days, profile):
            contrib.setdefault((row, day), []).append((k, h))
    for (c, t), terms in contrib.items():
        for k, h in terms:  # overload
            rows.append(r)
            cols.append(k)
            vals.append(h)
        rows.append(r)
        cols.append(o0 + c * n_days + t)
        vals.append(-1.0)
        lb.append(-np.inf)
        ub.append(cap - base[c, t])
        r += 1
        for k, h in terms:  # peak
            rows.append(r)
            cols.append(k)
            vals.append(h)
        rows.append(r)
        cols.append(p0 + c)
        vals.append(-1.0)
        lb.append(-np.inf)
        ub.append(-base[c, t])
        r += 1
    # cohort-days no movable assignment touches still count towards the peak
    for c in range(n_c):
        rows.append(r)
        cols.append(p0 + c)
        vals.append(1.0)
        lb.append(float(base[c].max(initial=0.0)))
        ub.append(np.inf)
        r += 1

    A = coo_array((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    integrality = np.zeros(nvar)
    integrality[:nx] = 1
    bounds = Bounds(np.zeros(nvar), np.concatenate([np.ones(nx), np.full(nvar - nx, np.inf)]))
    res = milp(
        cost,
        constraints=LinearConstraint(A, lb, ub),
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit, "disp": False},
    )
    due: dict[str, int] = {a.id: a.due for a in fixed}
    if res.x is not None:
        x = res.x[:nx]
        for k, (i, d) in enumerate(choices):
            if x[k] > 0.5:
                due[movable[i].id] = d
    status = {0: "optimal", 1: "time limit", 2: "infeasible", 3: "unbounded"}.get(res.status, str(res.status))
    gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
    bound = float(getattr(res, "mip_dual_bound", float("nan")) or float("nan"))
    return PlanResult(
        due, status, float(res.fun) if res.fun is not None else float("nan"), time.perf_counter() - t0, gap, bound
    )


def objective(
    assignments: Sequence[Assignment],
    courses: Mapping[str, Course],
    cohorts: Sequence[Cohort],
    n_days: int,
    cap: float = DEFAULT_CAP,
    profile: Profile = DEFAULT_PROFILE,
) -> float:
    """The integer program's objective for a finished plan, so heuristics can be compared
    with its bound."""
    load = daily_load(assignments, courses, cohorts, n_days, profile)
    size = np.array([c.size for c in cohorts], dtype=float)
    excess = float((np.clip(load - cap, 0, None) * size[:, None]).sum())
    peaks = float((load.max(axis=1) * size).sum()) * W_PEAK
    shift = sum(abs(a.due - (a.preferred if a.preferred is not None else a.due)) for a in assignments) * W_SHIFT
    return excess + peaks + shift


def improve(
    assignments: Sequence[Assignment],
    courses: Mapping[str, Course],
    cohorts: Sequence[Cohort],
    n_days: int,
    cap: float = DEFAULT_CAP,
    profile: Profile = DEFAULT_PROFILE,
    passes: int = 20,
    seed: int = 0,
) -> list[Assignment]:
    """Local search: take each movable assignment out in turn and put it back on its best
    date given all the others, until a full pass changes nothing. Fast enough to run on
    every change in the web app; docs/scheduling.md compares it with the exact optimum."""
    rng = np.random.default_rng(seed)
    current = list(assignments)
    for _ in range(passes):
        changed = False
        for i in rng.permutation(len(current)):
            a = current[int(i)]
            if not a.movable():
                continue
            others = current[: int(i)] + current[int(i) + 1 :]
            best = advise(a, others, courses, cohorts, n_days, cap, profile, top=1)[0]
            if best.due != a.due:
                current[int(i)] = apply({a.id: best.due}, [a])[0]
                changed = True
        if not changed:
            break
    return current


def apply(due: Mapping[str, int], assignments: Sequence[Assignment]) -> list[Assignment]:
    """The same assignments with the given due dates; release dates move with them."""
    out = []
    for a in assignments:
        d = due.get(a.id, a.due)
        out.append(replace(a, release=d - (a.due - a.release), due=d))
    return out
