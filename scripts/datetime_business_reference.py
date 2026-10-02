#!/usr/bin/env python3
"""Independent reference for `Std.Datetime.Business` (chelis#2860, [05-OP-73]).

Every answer here is computed from the definitions in [05-OP-73] by walking
days, never by the closed forms the Chelis module uses:

- A day of the horizon is a business day when the weekmask includes its
  weekday and it is not a holiday.
- A calendar answers only from the days of its horizon. A result exists when
  every choice of business days outside the horizon gives the same answer and
  that answer lies inside the horizon. The reference evaluates each query
  under three such choices (every outside day a business day, none, and the
  weekmask alone, which is NumPy's model) and reports `None` unless all three
  agree on an answer inside the horizon.

`numpy_disagreements` checks this reference against `numpy.is_busday`,
`numpy.busday_offset` and `numpy.busday_count` over a calendar, wherever the
reference defines an answer. NumPy's `busday_count` with `begin > end` counts
`(end, begin]` rather than `[end, begin)`, so reversed counts are compared
through the antisymmetry [05-OP-73] requires: `count(a, b) = -count(b, a)`.

`random_calendars` and `calendar_queries` generate the seeded corpus that
`datetime_business_differential.py` runs on `chelis eval` and compiled C and
that `test_datetime_business_reference.py` checks against NumPy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
import random

I64_MAX = 2**63 - 1
I64_MIN = -(2**63)
MIN_EPOCH_DAY = -4_371_587
MAX_EPOCH_DAY = 2_932_896
ROLLS = ("Unadjusted", "Following", "Preceding", "ModifiedFollowing", "ModifiedPreceding")
STARTS = ("RejectNonBusinessStart", "RollStartForward", "RollStartBackward")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def weekday(day: int) -> int:
    """0 for Monday through 6 for Sunday; 1970-01-01 (day 0) is a Thursday."""
    return (day + 3) % 7


@lru_cache(maxsize=None)
def civil_from_days(day: int) -> tuple[int, int, int]:
    """Proleptic Gregorian (year, month, day) of an epoch day, by whole 400-year cycles."""
    cycles, rest = divmod(day + 719_468, 146_097)
    year = 400 * cycles
    # Walk the cycle's years from 0000-03-01, then its months; the cycle has
    # 146 097 days, so this loop is bounded and needs no closed form.
    lengths = CYCLE_YEAR_LENGTHS
    offset = 0
    while rest >= lengths[offset]:
        rest -= lengths[offset]
        offset += 1
    year += offset
    month_lengths = [31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 29 if leap(year + 1) else 28]
    month = 0
    while rest >= month_lengths[month]:
        rest -= month_lengths[month]
        month += 1
    civil_month = month + 3 if month < 10 else month - 9
    return (year + 1 if civil_month <= 2 else year, civil_month, rest + 1)


def leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# Lengths of the 400 March-based years of a Gregorian cycle, from 0000-03-01.
CYCLE_YEAR_LENGTHS = tuple(366 if leap(y + 1) else 365 for y in range(400))


def month_of(day: int) -> tuple[int, int]:
    year, month, _ = civil_from_days(day)
    return (year, month)


@dataclass(frozen=True)
class Calendar:
    weekmask: tuple[bool, ...]
    holidays: tuple[int, ...]
    valid_from: int
    valid_until: int

    def inside(self, day: int) -> bool:
        return self.valid_from <= day <= self.valid_until

    @cached_property
    def holiday_set(self) -> frozenset[int]:
        return frozenset(self.holidays)

    def business(self, day: int) -> bool:
        return self.weekmask[weekday(day)] and day not in self.holiday_set

    @cached_property
    def business_list(self) -> tuple[int, ...]:
        return tuple(d for d in range(self.valid_from, self.valid_until + 1) if self.business(d))

    def business_days(self) -> tuple[int, ...]:
        return self.business_list


def make_calendar(weekmask: Sequence[bool], holidays: Iterable[int], valid_from: int, valid_until: int) -> Calendar | None:
    """`business_calendar`: `None` where it fails `domain`, else the normalized calendar."""
    holidays = list(holidays)
    if not any(weekmask) or valid_from > valid_until:
        return None
    if any(not valid_from <= h <= valid_until for h in holidays):
        return None
    kept = sorted({h for h in holidays if weekmask[weekday(h)]})
    return Calendar(tuple(weekmask), tuple(kept), valid_from, valid_until)


Outside = Callable[[Calendar, int], bool]
EXTENSIONS: tuple[Outside, ...] = (
    lambda cal, day: True,
    lambda cal, day: False,
    lambda cal, day: cal.weekmask[weekday(day)],
)
# Outside the horizon a walk stops after this many days: every extension above
# repeats weekly, so a business day exists within a week or never.
WALK = 8
NOWHERE = "nowhere"


def is_business(cal: Calendar, day: int, outside: Outside) -> bool:
    return cal.business(day) if cal.inside(day) else outside(cal, day)


def following(cal: Calendar, day: int, outside: Outside) -> int | str:
    probe = day
    while probe <= cal.valid_until + WALK:
        if is_business(cal, probe, outside):
            return probe
        probe += 1
    return NOWHERE


def preceding(cal: Calendar, day: int, outside: Outside) -> int | str:
    probe = day
    while probe >= cal.valid_from - WALK:
        if is_business(cal, probe, outside):
            return probe
        probe -= 1
    return NOWHERE


def roll_under(cal: Calendar, day: int, roll: str, outside: Outside) -> int | str:
    if roll == "Unadjusted":
        return day
    if roll == "Following":
        return following(cal, day, outside)
    if roll == "Preceding":
        return preceding(cal, day, outside)
    ahead, behind = following(cal, day, outside), preceding(cal, day, outside)
    if roll == "ModifiedFollowing":
        return ahead if ahead != NOWHERE and month_of(ahead) == month_of(day) else behind
    return behind if behind != NOWHERE and month_of(behind) == month_of(day) else ahead


def settled(cal: Calendar, answers: Sequence[int | str]) -> int | None:
    """The answer when every extension agrees on one inside the horizon."""
    first = answers[0]
    if all(a == first for a in answers) and first != NOWHERE and cal.inside(first):
        return first
    return None


def roll(cal: Calendar, day: int, roll_name: str) -> int | None:
    """`business_day_roll`: `None` where it fails `domain`."""
    if not cal.inside(day):
        return None
    return settled(cal, [roll_under(cal, day, roll_name, ext) for ext in EXTENSIONS])


def step_under(cal: Calendar, start: int, n: int, outside: Outside, budget: int) -> int | str:
    """`start` moved `n` business days; a walk that would leave the horizon is NOWHERE.

    Every step past a business day inside the horizon consumes one of the
    horizon's `budget` business days before the walk can leave it, so a
    larger `|n|` always leaves the horizon.
    """
    if abs(n) > budget:
        return NOWHERE
    probe, remaining, direction = start, abs(n), 1 if n > 0 else -1
    while remaining:
        probe += direction
        if not cal.inside(probe):
            return NOWHERE
        if is_business(cal, probe, outside):
            remaining -= 1
    return probe


def offset(cal: Calendar, day: int, n: int, start: str) -> int | None:
    """`business_day_offset`: `None` where it fails `domain`."""
    if not cal.inside(day):
        return None
    budget = len(cal.business_days())
    answers: list[int | str] = []
    for ext in EXTENSIONS:
        if is_business(cal, day, ext):
            begin: int | str = day
        elif start == "RejectNonBusinessStart":
            return None
        elif start == "RollStartForward":
            begin = following(cal, day, ext)
        else:
            begin = preceding(cal, day, ext)
        if begin == NOWHERE or not cal.inside(begin):
            answers.append(NOWHERE)
        else:
            answers.append(step_under(cal, begin, n, ext, budget))
    return settled(cal, answers)


def count(cal: Calendar, begin: int, end: int) -> int | None:
    """`business_day_count`: `None` where it fails `domain`."""
    if not (cal.valid_from <= begin <= cal.valid_until + 1 and cal.valid_from <= end <= cal.valid_until + 1):
        return None
    low, high = min(begin, end), max(begin, end)
    total = sum(1 for d in range(low, high) if cal.business(d))
    return total if begin <= end else -total


def combine(a: Calendar, b: Calendar, both: bool) -> Calendar | None:
    """`business_in_all` (both) or `business_in_any`: `None` where it fails `domain`."""
    valid_from, valid_until = max(a.valid_from, b.valid_from), min(a.valid_until, b.valid_until)
    if valid_from > valid_until:
        return None
    if both:
        weekmask = tuple(x and y for x, y in zip(a.weekmask, b.weekmask))
        open_day = lambda d: a.business(d) and b.business(d)
    else:
        weekmask = tuple(x or y for x, y in zip(a.weekmask, b.weekmask))
        open_day = lambda d: a.business(d) or b.business(d)
    if not any(weekmask):
        return None
    holidays = [d for d in range(valid_from, valid_until + 1) if weekmask[weekday(d)] and not open_day(d)]
    return Calendar(weekmask, tuple(holidays), valid_from, valid_until)


def first_failure(results: Sequence[int | None]) -> int | None:
    """The lowest index whose scalar twin fails, for the vectorized forms."""
    for index, result in enumerate(results):
        if result is None:
            return index
    return None


# ---------------------------------------------------------------------------
# NumPy comparison.

NUMPY_ROLLS = {"Following": "forward", "Preceding": "backward",
               "ModifiedFollowing": "modifiedfollowing", "ModifiedPreceding": "modifiedpreceding"}
NUMPY_STARTS = {"RejectNonBusinessStart": "raise", "RollStartForward": "forward", "RollStartBackward": "backward"}


def numpy_disagreements(np, cal: Calendar, days: Sequence[int], offsets: Sequence[int],
                        pairs: Sequence[tuple[int, int]], numpy_holidays: Sequence[int] | None = None) -> list[str]:
    """Every place the reference defines an answer that NumPy contradicts.

    NumPy's calendar takes `cal`'s holidays unless `numpy_holidays` names
    others, which lets a test show that a wrong reference is caught.
    """
    holidays = cal.holidays if numpy_holidays is None else numpy_holidays
    nc = np.busdaycalendar(weekmask=[int(x) for x in cal.weekmask],
                           holidays=np.array(holidays, dtype="int64").astype("datetime64[D]"))
    as_day = lambda d: np.datetime64(int(d), "D")
    as_int = lambda value: int(value.astype("int64"))
    problems = []
    for day in days:
        if cal.inside(day) and bool(np.is_busday(as_day(day), busdaycal=nc)) != cal.business(day):
            problems.append(f"is_busday {day}")
        for name, numpy_roll in NUMPY_ROLLS.items():
            ours = roll(cal, day, name)
            if ours is not None and as_int(np.busday_offset(as_day(day), 0, roll=numpy_roll, busdaycal=nc)) != ours:
                problems.append(f"roll {name} {day}: reference {ours}")
        for n in offsets:
            if abs(n) > 10**6:
                continue
            for start, numpy_roll in NUMPY_STARTS.items():
                ours = offset(cal, day, n, start)
                try:
                    theirs: int | None = as_int(np.busday_offset(as_day(day), n, roll=numpy_roll, busdaycal=nc))
                except ValueError:
                    theirs = None
                if ours is not None and theirs != ours:
                    problems.append(f"offset {start} {day} {n}: reference {ours}, numpy {theirs}")
                if start == "RejectNonBusinessStart" and cal.inside(day) and not cal.business(day) and theirs is not None:
                    problems.append(f"offset raise {day}: numpy accepted a non-business start")
    for begin, end in pairs:
        ours = count(cal, begin, end)
        if ours is None:
            continue
        low, high = min(begin, end), max(begin, end)
        theirs = int(np.busday_count(as_day(low), as_day(high), busdaycal=nc))
        if (theirs if begin <= end else -theirs) != ours:
            problems.append(f"count {begin} {end}: reference {ours}, numpy {theirs}")
    return problems


# ---------------------------------------------------------------------------
# Seeded corpus. Every generated input is a function of small integers that
# the Chelis programs recompute with the same linear congruential generator,
# so a program carries a few literals instead of long lists: long list
# literals overflow `chelis eval`'s stack (#906) and slow `chelis build`.


def lcg(x: int) -> int:
    return (x * 1_103_515_245 + 12_345) % 2_147_483_648


def mix(k: int, seed: int) -> int:
    """A pseudo-random value in 0..2^31 - 1 for `k + seed >= 0`."""
    return lcg(lcg(k + seed))


WEEKDAYS_ONLY = (True, True, True, True, True, False, False)


@dataclass(frozen=True)
class CalendarSpec:
    """Constructor arguments: literal holidays, or holidays drawn by `mix`."""

    weekmask: tuple[bool, ...]
    valid_from: int
    valid_until: int
    holidays: tuple[int, ...] = ()
    density: int | None = None  # holidays per thousand days, drawn by `mix`
    seed: int = 0

    def holiday_args(self) -> list[int]:
        """`holidays_of` in the generated programs: drawn days, then the first three again."""
        if self.density is None:
            return list(self.holidays)
        length = self.valid_until - self.valid_from + 1
        picked = [i for i in range(length) if mix(i, self.seed) % 1000 < self.density]
        return [self.valid_from + i for i in picked + picked[:3]]

    def calendar(self) -> Calendar:
        calendar = make_calendar(self.weekmask, self.holiday_args(), self.valid_from, self.valid_until)
        assert calendar is not None, self
        return calendar


def edge_calendars() -> list[CalendarSpec]:
    """Calendars at the range edges, at month edges, of one day, and with no business day."""
    start_2026 = 20_454  # 2026-01-01, a Thursday
    return [
        CalendarSpec(WEEKDAYS_ONLY, MIN_EPOCH_DAY, MIN_EPOCH_DAY + 120, (MIN_EPOCH_DAY, MIN_EPOCH_DAY + 3)),
        CalendarSpec(WEEKDAYS_ONLY, MAX_EPOCH_DAY - 120, MAX_EPOCH_DAY, (MAX_EPOCH_DAY, MAX_EPOCH_DAY - 7)),
        CalendarSpec((False,) * 5 + (True, True), MAX_EPOCH_DAY - 40, MAX_EPOCH_DAY),
        CalendarSpec(WEEKDAYS_ONLY, start_2026 + 2, start_2026 + 2),
        CalendarSpec(WEEKDAYS_ONLY, start_2026 + 2, start_2026 + 8, (start_2026 + 4,)),
        CalendarSpec((True,) + (False,) * 6, start_2026, start_2026 + 6, (start_2026 + 4,)),
        # May 2026 through its last day, and through Saturday the 23rd.
        CalendarSpec(WEEKDAYS_ONLY, 20_574, 20_604, (20_602,)),
        CalendarSpec(WEEKDAYS_ONLY, 20_574, 20_596),
        # From Sunday 2026-03-08, and from 2026-03-01.
        CalendarSpec(WEEKDAYS_ONLY, 20_520, 20_543),
        CalendarSpec(WEEKDAYS_ONLY, 20_513, 20_543, (20_514,)),
    ]


def build_calendars(seed: int, count: int) -> list[CalendarSpec]:
    """The edge calendars, then `count` seeded ones with drawn weekmasks and holidays."""
    rng = random.Random(seed)
    specs = edge_calendars()
    for index in range(count):
        while True:
            weekmask = tuple(rng.random() < 0.7 for _ in range(7))
            if any(weekmask):
                break
        valid_from = rng.randint(-200_000, 100_000)
        length = rng.choice([1, 7, 30, 31, 90, 365, 400, 800])
        density = rng.choice([0, 30, 100, 500])
        specs.append(CalendarSpec(weekmask, valid_from, valid_from + length - 1, density=density, seed=seed + 7 * index))
    return specs


OFFSETS_FIXED = (0, 1, -1, 3, -3, 17, -17)


@dataclass(frozen=True)
class Queries:
    """One calendar's queries, as the generated programs recompute them."""

    days: tuple[int, ...]
    offsets: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]


def query_days(cal: Calendar, seed: int, limit: int) -> list[int]:
    """`limit` drawn days of the horizon, its edges, and the days just outside it."""
    length = cal.valid_until - cal.valid_from + 1
    drawn = [cal.valid_from + mix(k, seed) % length for k in range(limit)]
    edges = [d for d in (cal.valid_from, cal.valid_until, cal.valid_from + 1, cal.valid_until - 1) if cal.inside(d)]
    outside = [d for d in (cal.valid_from - 1, cal.valid_until + 1) if MIN_EPOCH_DAY <= d <= MAX_EPOCH_DAY]
    return drawn + edges + outside


def calendar_queries(seed: int, cal: Calendar, limit: int = 32, pair_count: int = 30) -> Queries:
    days = query_days(cal, seed, limit)
    budget = len(cal.business_days())
    offsets = OFFSETS_FIXED + (budget, -budget, budget - 1, 1 - budget, I64_MAX, I64_MIN)
    ends = days + [d for d in (cal.valid_until + 2,) if d <= MAX_EPOCH_DAY]
    pairs = [(ends[mix(k, seed + 1) % len(ends)], ends[mix(k, seed + 2) % len(ends)]) for k in range(pair_count)]
    fixed = [(cal.valid_from, cal.valid_until + 1), (cal.valid_until + 1, cal.valid_from)]
    pairs += [(a, b) for a, b in fixed if b <= MAX_EPOCH_DAY and a <= MAX_EPOCH_DAY]
    return Queries(tuple(days), tuple(offsets), tuple(pairs))
