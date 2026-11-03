# Study

Two questions, both answered on synthetic data whose every assumption is written in the
code: how well can the effort an assignment takes be predicted, and how much does
coordinating deadlines across courses reduce overload?

    uv run asgn study --seeds 10      # about 12 minutes; writes results/

## 1. How long will it take?

[`effort.py`](../src/asgn/effort.py). Synthetic time logs: 30 courses over three terms,
five to seven assignments per course per term, about 24 students logging time on each.
Hours grow with size (exponent 0.6), a little with grade weight and course level, and with
a hidden difficulty per course that no feature reveals; students differ from one another by a
log-normal factor (sd 0.45). Models train on the first two terms and are scored on the
third, course by course in date order: each assignment is predicted before its logs arrive,
then the course's calibration is updated.

| Model | Error in the median (hours) | Typical error, as a factor | Students within the 80th-percentile estimate |
|---|---:|---:|---:|
| Median hours for the kind of assignment | 2.21 | ×1.33 | 80 % |
| Quantile gradient boosting on what the instructor knows | 1.93 | ×1.27 | 75 % |
| **+ per-course calibration from logged time** | **1.60** | **×1.22** | 76 % |

Mean over 10 seeds. The typical error is the median of |log(predicted / actual)|, shown as a factor. Learning from each course's own logs cuts the error by about a quarter.
The 80th-percentile estimate is slightly too low (it covers 76 % of students rather than
80 %): the model has seen most courses' difficulty only through calibration, which adjusts
the median and the upper estimate by the same factor. A deployment should check this
coverage on its own logs and widen the upper estimate if it holds.

## 2. Does coordinating deadlines help?

[`sim.py`](../src/asgn/sim.py). A department of six cohorts (three year groups, two sections
each) over a 15-week term. Each cohort takes four core courses plus one or two electives
shared with other cohorts; about 93 assignments a term. Instructors spread their work
through the term, but four in ten deadlines drift towards the week before mid-terms or the
last two weeks. Every deadline may move up to 4 days either side of the instructor's choice,
keeping the time students get. A day is over the limit above 3 hours of coursework per
student beyond classes; students' effort leans towards the deadline
([`load.py`](../src/asgn/load.py)).

| Deadlines chosen by | Hours over the limit (cohort-days, summed) | Worst day (h) | Average cohort's worst day (h) | Average move (days) | Assignments moved | Time |
|---|---:|---:|---:|---:|---:|---:|
| each instructor alone | 171 ± 45 | 10.3 | 7.5 | 0 | 0 % | |
| the advisor, one instructor at a time | 70 ± 25 | 6.8 | 5.2 | 1.6 | 53 % | 0.04 s |
| **the planner, all at once** | **56 ± 20** | **6.6** | **5.0** | 1.9 | 68 % | 0.34 s |
| the integer program, 60 s time limit | 57 ± 21 | 6.4 | 5.0 | 1.9 | 73 % | 60 s |

Mean ± sd over 10 terms.

**What the numbers say**

1. **Seeing the load is most of the win.** If each instructor simply takes the advisor's
   suggestion when setting their own assignment, hours over the limit fall by 59 % and the
   worst day drops from about 10 hours to under 7. Nobody coordinates with anyone; the
   shared calendar does it for them.
2. **Planning everything together adds a fifth more**, down to a third of the original
   overload, moving the average deadline by under two days.
3. **The fast planner matches the exact method.** The integer program
   ([`schedule.py`](../src/asgn/schedule.py)) never proved optimality within 60 seconds on
   these terms (remaining gaps of 18 to 46 %). Measured on its own objective, the 0.3-second
   local search did better on 6 of the 10 terms and was within 8 % on the rest. The web app
   therefore uses the planner, and the integer program serves as the yardstick.
4. **Not every crunch can be fixed by moving dates.** When too much work is set close
   together, a 4-day window is not enough room. The remaining overload is a signal that
   the amount of work, not just its timing, needs a conversation.

## Limits

- Everything is synthetic. The structure (who takes what, how instructors pick dates,
  how students spread effort) is plausible and stated, not measured.
- The load model assumes every student starts and works the same way; real students differ,
  and the tool shows the typical student, not the one who starts the night before.
- Hours over the limit count every hour equally. Some departments would rather avoid
  exam weeks entirely, or protect weekends; those are extra constraints the integer
  program could take and the planner could weigh.
