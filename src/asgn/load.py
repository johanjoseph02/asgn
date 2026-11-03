"""Turning assignments into hours of work on each day, for each group of students.

Days are integers counted from the first day of term. A **cohort** is a group of students
who take the same set of courses (a section, a year group); every assignment in any of
its courses lands on it.

Students do not spread work evenly. ``work_profile`` puts an assignment's hours on the
days before its deadline with an exponential lean towards the deadline (time constant
``tau`` days), from its release date or ``horizon`` days before the deadline, whichever is
later. The shape is an assumption, stated here and configurable; what the tool reports is
the load *under this assumption*, and the scheduling results hold for any fixed shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

DEFAULT_CAP = 3.0  # hours of coursework per day beyond classes before a day counts as overloaded
DEFAULT_TAU = 2.5
DEFAULT_HORIZON = 10


@dataclass(frozen=True)
class Cohort:
    id: str
    name: str
    size: int


@dataclass(frozen=True)
class Course:
    id: str
    code: str
    name: str
    cohorts: tuple[str, ...]


@dataclass
class Assignment:
    id: str
    course_id: str
    title: str
    release: int
    due: int
    hours: float  # expected hours per student (the effort model's median)
    hours_p80: float = 0.0
    window: tuple[int, int] | None = None  # days the due date may move within, if planning
    preferred: int | None = None  # the instructor's first choice
    kind: str = "problem_set"
    meta: dict[str, object] = field(default_factory=dict)

    def movable(self) -> bool:
        return self.window is not None and self.window[0] < self.window[1]


@dataclass(frozen=True)
class Profile:
    tau: float = DEFAULT_TAU
    horizon: int = DEFAULT_HORIZON


DEFAULT_PROFILE = Profile()


def work_profile(release: int, due: int, profile: Profile = DEFAULT_PROFILE) -> tuple[int, NDArray[np.float64]]:
    """First day worked and the share of the effort done on each day up to ``due``."""
    start = max(release, due - profile.horizon + 1)
    if start > due:
        start = due
    days = np.arange(start, due + 1)
    w = np.exp(-(due - days) / profile.tau)
    return start, w / w.sum()


def daily_load(
    assignments: Iterable[Assignment],
    courses: Mapping[str, Course],
    cohorts: Sequence[Cohort],
    n_days: int,
    profile: Profile = DEFAULT_PROFILE,
) -> NDArray[np.float64]:
    """Hours per student per day, shape (cohorts, days)."""
    index = {c.id: i for i, c in enumerate(cohorts)}
    load = np.zeros((len(cohorts), n_days))
    for a in assignments:
        start, share = work_profile(a.release, a.due, profile)
        for cid in courses[a.course_id].cohorts:
            if cid not in index:
                continue
            lo, hi = max(start, 0), min(a.due + 1, n_days)
            if hi > lo:
                load[index[cid], lo:hi] += a.hours * share[lo - start : hi - start]
    return load


@dataclass(frozen=True)
class LoadSummary:
    overloaded_days: int  # cohort-days above the cap
    overloaded_student_days: int  # the same, weighted by cohort size
    excess_hours: float  # hours above the cap, summed over cohort-days
    worst_day: float  # highest daily load of any cohort
    mean_peak: float  # each cohort's highest day, averaged

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def summarise(load: NDArray[np.float64], cohorts: Sequence[Cohort], cap: float = DEFAULT_CAP) -> LoadSummary:
    over = load > cap + 1e-9
    sizes = np.array([c.size for c in cohorts])
    return LoadSummary(
        overloaded_days=int(over.sum()),
        overloaded_student_days=int((over.sum(axis=1) * sizes).sum()),
        excess_hours=float(np.clip(load - cap, 0, None).sum()),
        worst_day=float(load.max(initial=0.0)),
        mean_peak=float(load.max(axis=1).mean()) if len(load) else 0.0,
    )
