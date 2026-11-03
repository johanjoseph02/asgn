"""How long will this assignment take a student?

Instructors know how hard their own assignment is, but not how long it takes, and they
rarely know what else their students are doing that week. The model predicts the
**median** and the **80th percentile** of student hours from what an instructor knows when
setting the work (kind, weight in the final grade, size, course level, days given), then
corrects itself course by course as students log the hours they actually spent.

* ``EffortModel``: gradient-boosted quantile regression on individual students' logged
  hours (two models, quantiles 0.5 and 0.8).
* ``CourseCalibration``: after each assignment in a course closes, the ratio of what
  students logged to what was predicted updates a per-course factor (in log space, shrunk
  towards 1 until there is evidence). Courses differ in ways no feature captures.

The training data in this repository is synthetic (``synthetic_logs``), with its
generating assumptions written down; a real deployment trains on its own logs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor

KINDS = ("problem_set", "lab_report", "essay", "coding", "reading_response", "project_milestone")
FEATURES = (*(f"kind_{k}" for k in KINDS), "weight_pct", "size", "level", "days_given")


@dataclass(frozen=True)
class Spec:
    """What an instructor knows when setting an assignment."""

    kind: str
    weight_pct: float  # share of the final grade
    size: float  # questions, pages or tasks
    level: int  # 1 to 4: year of the course
    days_given: int

    def vector(self) -> list[float]:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        return [float(self.kind == k) for k in KINDS] + [
            self.weight_pct,
            self.size,
            float(self.level),
            float(self.days_given),
        ]


@dataclass
class EffortModel:
    p50: HistGradientBoostingRegressor | None = None
    p80: HistGradientBoostingRegressor | None = None

    def fit(self, X: NDArray[np.float64], hours: NDArray[np.float64], seed: int = 0) -> EffortModel:
        y = np.log(np.maximum(hours, 0.1))
        kw: dict[str, Any] = {"max_iter": 300, "learning_rate": 0.05, "min_samples_leaf": 40, "random_state": seed}
        self.p50 = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **kw).fit(X, y)
        self.p80 = HistGradientBoostingRegressor(loss="quantile", quantile=0.8, **kw).fit(X, y)
        return self

    def predict(self, specs: list[Spec]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert self.p50 is not None and self.p80 is not None, "fit first"
        X = np.array([s.vector() for s in specs])
        lo = np.exp(self.p50.predict(X))
        hi = np.maximum(np.exp(self.p80.predict(X)), lo)
        return lo, hi


@dataclass
class CourseCalibration:
    prior: float = 2.0  # pseudo-observations of "the model is right" per course
    log_ratios: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def factor(self, course: str) -> float:
        r = self.log_ratios.get(course, [])
        n = len(r)
        return math.exp(sum(r) / (n + self.prior)) if n else 1.0

    def observe(self, course: str, predicted_median: float, logged_hours: list[float]) -> None:
        if logged_hours and predicted_median > 0:
            self.log_ratios[course].append(math.log(float(np.median(logged_hours)) / predicted_median))


# ------------------------------------------------------------------------------ synthetic data

BASE = {
    "problem_set": 4.0,
    "lab_report": 5.0,
    "essay": 6.0,
    "coding": 6.0,
    "reading_response": 1.5,
    "project_milestone": 8.0,
}
REF_SIZE = {"problem_set": 8, "lab_report": 6, "essay": 5, "coding": 4, "reading_response": 3, "project_milestone": 4}


@dataclass(frozen=True)
class LoggedAssignment:
    term: int
    course: str
    spec: Spec
    hours: tuple[float, ...]  # one entry per student who logged


def synthetic_logs(seed: int = 0, terms: int = 3, courses: int = 30, students: int = 40) -> list[LoggedAssignment]:
    """Assumptions: hours grow sub-linearly with size (exponent 0.6), a little with grade
    weight and course level; every course has its own hidden difficulty (log-normal,
    sd 0.3) that no feature reveals; students differ by a log-normal factor (sd 0.45);
    about 60 % of students log their time."""
    rng = np.random.default_rng(seed)
    difficulty = {f"K{c:02d}": float(np.exp(rng.normal(0, 0.3))) for c in range(courses)}
    level = {c: int(rng.integers(1, 5)) for c in difficulty}
    out = []
    for t in range(terms):
        for c in difficulty:
            for _ in range(int(rng.integers(5, 8))):
                kind = str(rng.choice(KINDS))
                size = max(1, round(REF_SIZE[kind] * float(np.exp(rng.normal(0, 0.4)))))
                weight = float(rng.choice([2, 5, 5, 10, 10, 15, 20]))
                days = int(rng.integers(5, 15))
                spec = Spec(kind, weight, size, level[c], days)
                mean = (
                    BASE[kind]
                    * (size / REF_SIZE[kind]) ** 0.6
                    * (1 + weight / 100)
                    * (1 + 0.1 * (level[c] - 1))
                    * difficulty[c]
                )
                n = int(rng.binomial(students, 0.6))
                hours = tuple(round(float(mean * math.exp(rng.normal(0, 0.45))), 2) for _ in range(n))
                out.append(LoggedAssignment(t, c, spec, hours))
    return out


def evaluate(seed: int = 0) -> dict[str, Any]:
    """Train on the first two terms, then walk through the third term course by course,
    predicting each assignment before its logs arrive and calibrating afterwards."""
    data = synthetic_logs(seed)
    train = [a for a in data if a.term < 2]
    test = [a for a in data if a.term == 2]
    X = np.array([a.spec.vector() for a in train for _ in a.hours])
    y = np.array([h for a in train for h in a.hours])
    model = EffortModel().fit(X, y, seed)

    by_kind: dict[str, list[float]] = defaultdict(list)
    for a in train:
        by_kind[a.spec.kind].extend(a.hours)
    kind_p50 = {k: float(np.median(v)) for k, v in by_kind.items()}
    kind_p80 = {k: float(np.quantile(v, 0.8)) for k, v in by_kind.items()}

    p50, p80 = model.predict([a.spec for a in test])
    cal = CourseCalibration()
    rows: dict[str, list[tuple[float, float, list[float]]]] = defaultdict(list)
    for a, lo, hi in zip(test, p50, p80, strict=True):
        actual = list(a.hours)
        if not actual:
            continue
        f = cal.factor(a.course)
        rows["by kind (median of past logs)"].append((kind_p50[a.spec.kind], kind_p80[a.spec.kind], actual))
        rows["quantile model"].append((float(lo), float(hi), actual))
        rows["quantile model + course calibration"].append((float(lo) * f, float(hi) * f, actual))
        cal.observe(a.course, float(lo), actual)

    def score(r: list[tuple[float, float, list[float]]]) -> dict[str, float]:
        med_err = [abs(lo - float(np.median(act))) for lo, _, act in r]
        rel = [abs(math.log(lo / float(np.median(act)))) for lo, _, act in r]
        cover = [float(np.mean([h <= hi for h in act])) for _, hi, act in r]
        return {
            "assignments": len(r),
            "mae_hours": float(np.mean(med_err)),
            "median_abs_log_error": float(np.median(rel)),
            "p80_coverage": float(np.mean(cover)),
        }

    return {
        "train_assignments": len(train),
        "train_logs": len(y),
        "test_assignments": len(test),
        "models": {k: score(v) for k, v in rows.items()},
    }
