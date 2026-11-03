"""Server-rendered SVG: a load heatmap (cohorts by day) and a single cohort's load bars.

No JavaScript; every cell carries a <title> so the numbers are there on hover and to
screen readers, and the colour scale is explained beside each chart.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from html import escape

import numpy as np
from numpy.typing import NDArray


def colour(hours: float, cap: float) -> str:
    """Green-grey below half the cap, amber towards the cap, red above it."""
    if hours <= 0.05:
        return "var(--empty)"
    x = hours / cap
    if x <= 1:
        return f"color-mix(in oklab, var(--ok) {max(25, round(100 * (1 - x)))}%, var(--warn))"
    return f"color-mix(in oklab, var(--warn) {max(0, round(100 * (2 - x)))}%, var(--bad))"


def heatmap(
    load: NDArray[np.float64],
    labels: Sequence[str],
    first: int,
    last: int,
    cap: float,
    date_of: Callable[[int], date],
    today: int | None = None,
    cell: int = 11,
) -> str:
    days = range(first, last + 1)
    left, top = 110, 22
    w = left + cell * len(days) + 4
    h = top + (cell + 3) * len(labels) + 4
    parts = [f'<svg class="heat" viewBox="0 0 {w} {h}" role="img" aria-label="Daily coursework hours by cohort">']
    for i, d in enumerate(days):
        dt = date_of(d)
        if dt.weekday() == 0:
            x = left + i * cell
            parts.append(f'<text x="{x}" y="12" class="tick">{dt.strftime("%d %b")}</text>')
            parts.append(f'<line x1="{x}" x2="{x}" y1="15" y2="{h}" class="grid"/>')
    if today is not None and first <= today <= last:
        tx = left + (today - first) * cell + cell / 2
        parts.append(f'<line x1="{tx}" x2="{tx}" y1="15" y2="{h}" class="today"/>')
    for r, name in enumerate(labels):
        y = top + r * (cell + 3)
        parts.append(f'<text x="0" y="{y + cell - 2}" class="label">{escape(name)}</text>')
        for i, d in enumerate(days):
            v = float(load[r, d])
            parts.append(
                f'<rect x="{left + i * cell}" y="{y}" width="{cell - 1}" height="{cell}" rx="2" '
                f'fill="{colour(v, cap)}"><title>{escape(name)}, {date_of(d):%a %d %b}: {v:.1f} h</title></rect>'
            )
    parts.append("</svg>")
    return "".join(parts)


def bars(
    load: NDArray[np.float64],
    first: int,
    last: int,
    cap: float,
    date_of: Callable[[int], date],
    deadlines: dict[int, list[str]],
    today: int | None = None,
) -> str:
    days = list(range(first, last + 1))
    bw, gap, left, top, height = 18, 4, 30, 10, 140
    w = left + (bw + gap) * len(days)
    vmax = max(cap * 2, float(load[first : last + 1].max(initial=0.0)) * 1.1)

    def y(v: float) -> float:
        return top + height * (1 - v / vmax)

    parts = [
        f'<svg class="bars" viewBox="0 0 {w} {top + height + 34}" role="img" '
        f'aria-label="Expected coursework hours per day">'
    ]
    parts.append(f'<line x1="{left}" x2="{w}" y1="{y(cap)}" y2="{y(cap)}" class="cap"/>')
    parts.append(f'<text x="0" y="{y(cap) + 4}" class="tick">{cap:g} h</text>')
    for i, d in enumerate(days):
        v = float(load[d])
        x = left + i * (bw + gap)
        dt = date_of(d)
        due = deadlines.get(d, [])
        title = f"{dt:%a %d %b}: {v:.1f} h" + (f"; due: {', '.join(due)}" if due else "")
        parts.append(
            f'<rect x="{x}" y="{y(v)}" width="{bw}" height="{top + height - y(v)}" rx="3" '
            f'fill="{colour(v, cap)}"{" class=current" if d == today else ""}><title>{escape(title)}</title></rect>'
        )
        if due:
            parts.append(
                f'<circle cx="{x + bw / 2}" cy="{top + height + 8}" r="4" class="due">'
                f"<title>{escape(title)}</title></circle>"
            )
        if dt.weekday() == 0 or i == 0:
            parts.append(f'<text x="{x}" y="{top + height + 28}" class="tick">{dt:%d %b}</text>')
    parts.append("</svg>")
    return "".join(parts)
