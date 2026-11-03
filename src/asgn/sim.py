"""Synthetic terms for comparing how deadlines get chosen.

A department with six cohorts (three year groups, two sections each) over a 15-week
term. Each cohort takes four core courses of its year plus electives shared across years,
so cohorts overlap. Each course sets five to seven assignments. Instructors, each on their
own, spread their assignments through the term, but four in ten deadlines drift towards
the same few moments: the week before mid-terms (week 7) and the last two weeks. That is
the pile-up the tool exists to see and fix.

Every assignment may move within a window around its preferred date (plus or minus
``slack`` days, never past the end of term), keeping its length (release to due).

Policies compared on identical terms:

* ``preferred``: everyone keeps their first choice. What happens today.
* ``advisor``: instructors plan one at a time, in random order; each takes the date the
  advisor ranks best given everything already planned. What a shared calendar with
  suggestions gives.
* ``optimal``: every movable assignment planned at once by the integer program.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from asgn.load import DEFAULT_CAP, DEFAULT_PROFILE, Assignment, Cohort, Course, Profile, daily_load, summarise
from asgn.schedule import advise, apply, improve, objective, plan

TERM_DAYS = 105
KINDS = ("problem_set", "lab_report", "essay", "coding", "reading_response", "project_milestone")
BASE_HOURS = {
    "problem_set": 4.0,
    "lab_report": 5.0,
    "essay": 6.0,
    "coding": 6.0,
    "reading_response": 1.5,
    "project_milestone": 8.0,
}


@dataclass
class Term:
    cohorts: list[Cohort]
    courses: dict[str, Course]
    assignments: list[Assignment]
    n_days: int = TERM_DAYS


def make_term(seed: int, slack: int = 4) -> Term:
    rng = np.random.default_rng(seed)
    cohorts = [Cohort(f"Y{y}{s}", f"Year {y}, section {s}", int(rng.integers(45, 70))) for y in (1, 2, 3) for s in "AB"]
    courses: dict[str, Course] = {}
    for y in (1, 2, 3):
        for k in range(4):
            cid = f"C{y}{k}"
            courses[cid] = Course(cid, f"CS{y}0{k}", f"Core course {y}.{k}", (f"Y{y}A", f"Y{y}B"))
    for k in range(4):  # electives for years 2 and 3, each taken by one or two cohorts
        members = tuple(
            sorted(rng.choice(["Y2A", "Y2B", "Y3A", "Y3B"], size=int(rng.integers(1, 3)), replace=False).tolist())
        )
        courses[f"E{k}"] = Course(f"E{k}", f"EL{k}", f"Elective {k}", members)

    pressure = (46, 48, 95, 98, 101)  # the week before mid-terms, and the end of term
    assignments = []
    for cid in courses:
        difficulty = float(np.exp(rng.normal(0, 0.25)))
        n = int(rng.integers(5, 8))
        for j in range(n):
            kind = str(rng.choice(KINDS))
            # spread through the term, on the instructor's teaching rhythm...
            pref = round((j + 0.75) * (TERM_DAYS - 7) / n + rng.integers(-4, 5))
            # ...but deadlines drift towards the same few weeks: the pile-up this tool is for
            if rng.random() < 0.4:
                target = min(pressure, key=lambda d: abs(d - pref))
                if abs(target - pref) <= 14:
                    pref = target + int(rng.integers(-1, 2))
            pref = int(np.clip(pref, 7, TERM_DAYS - 1))
            length = int(rng.integers(7, 15))
            hours = BASE_HOURS[kind] * difficulty * float(np.exp(rng.normal(0, 0.3)))
            lo, hi = max(pref - slack, 1), min(pref + slack, TERM_DAYS - 1)
            assignments.append(
                Assignment(
                    id=f"{cid}-{j}",
                    course_id=cid,
                    title=f"{kind.replace('_', ' ').capitalize()} {j + 1}",
                    release=pref - length,
                    due=pref,
                    hours=round(hours, 1),
                    window=(lo, hi),
                    preferred=pref,
                    kind=kind,
                )
            )
    return Term(cohorts, courses, assignments)


def run_policy(
    term: Term, policy: str, seed: int, cap: float = DEFAULT_CAP, profile: Profile = DEFAULT_PROFILE
) -> dict[str, Any]:
    t0 = time.perf_counter()
    extra: dict[str, float] = {}
    if policy == "preferred":
        final = list(term.assignments)
    elif policy == "advisor":
        rng = np.random.default_rng(seed + 1000)
        order = rng.permutation(len(term.assignments))
        planned: list[Assignment] = []
        for i in order:
            a = term.assignments[int(i)]
            best = advise(a, planned, term.courses, term.cohorts, term.n_days, cap, profile, top=1)[0]
            planned.append(apply({a.id: best.due}, [a])[0])
        final = planned
    elif policy == "planner":
        first = run_policy(term, "advisor", seed, cap, profile)["_final"]
        final = improve(first, term.courses, term.cohorts, term.n_days, cap, profile, seed=seed)
    elif policy == "optimal":
        res = plan(term.assignments, term.courses, term.cohorts, term.n_days, cap, profile)
        final = apply(res.due, term.assignments)
        extra = {"mip_gap": res.gap, "optimal": float(res.status == "optimal"), "bound": res.bound}
    else:
        raise ValueError(policy)
    load = daily_load(final, term.courses, term.cohorts, term.n_days, profile)
    shifts = [
        abs(f.due - (a.preferred if a.preferred is not None else a.due))
        for a, f in zip(sorted(term.assignments, key=lambda x: x.id), sorted(final, key=lambda x: x.id), strict=True)
    ]
    out = summarise(load, term.cohorts, cap).as_dict()
    out.update(
        {
            "mean_shift_days": float(np.mean(shifts)),
            "moved_share": float(np.mean([s > 0 for s in shifts])),
            "seconds": time.perf_counter() - t0,
            "objective": objective(final, term.courses, term.cohorts, term.n_days, cap, profile),
            **extra,
        }
    )
    out["_final"] = final  # type: ignore[assignment]
    return out


def study(seeds: range = range(10), slack: int = 4, log: bool = True) -> dict[str, Any]:
    policies = ("preferred", "advisor", "planner", "optimal")
    runs: dict[str, list[dict[str, Any]]] = {p: [] for p in policies}
    for s in seeds:
        term = make_term(s, slack)
        for p in policies:
            r = run_policy(term, p, s)
            r.pop("_final")
            runs[p].append(r)
        if log:
            print(f"seed {s}: " + ", ".join(f"{p} {runs[p][-1]['excess_hours']:.0f} h" for p in policies), flush=True)
    keys = [k for k in runs["optimal"][0] if all(k in r for rs in runs.values() for r in rs)]
    summary = {
        p: {k: {"mean": float(np.mean([r[k] for r in rs])), "sd": float(np.std([r[k] for r in rs]))} for k in keys}
        for p, rs in runs.items()
    }
    example = make_term(seeds[0], slack)
    return {
        "seeds": list(seeds),
        "slack_days": slack,
        "cap_hours": DEFAULT_CAP,
        "assignments_per_term": len(example.assignments),
        "summary": summary,
        "runs": runs,
    }


def with_slack(term: Term, slack: int) -> Term:
    out = []
    for a in term.assignments:
        pref = a.preferred if a.preferred is not None else a.due
        out.append(replace(a, window=(max(pref - slack, 1), min(pref + slack, term.n_days - 1))))
    return Term(term.cohorts, term.courses, out, term.n_days)
