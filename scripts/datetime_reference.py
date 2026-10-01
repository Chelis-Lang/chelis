#!/usr/bin/env python3
"""Independent reference semantics for `Std.Datetime` stage S1 (chelis#2859).

This module is the oracle the differential harness compares `chelis eval` and
compiled C against. It is written from the algorithms' published definitions
and from the design of record, `spec/design/std_datetime.md` (§4 range, §5
failure grammar, §6 to §8 API, semantics and text profile), never by
translating Chelis source. Python integers are unbounded, so every value here
is the exact mathematical answer; a result that leaves i64 or the supported
range is reported the way §5 says the module must report it.

Values use the module's canonical representation (§6):

- Date: epoch day (days since 1970-01-01);
- Time: nanosecond of day;
- DateTime: (epoch day, nanosecond of day);
- Instant and Duration: (second, nanosecond) with nanosecond in [0, 10^9);
- Offset: seconds;
- OffsetDateTime: (Instant, Offset);
- Period: (months, days);
- Weekday: its ISO number, Monday 1 through Sunday 7.

Failures raise `DatetimeError`, whose `message` follows the §5 grammar
`<function>: <kind>: <detail>`. The design pins `<function>` and `<kind>`; it
pins `<detail>` only by one example, so the differential compares the first two
exactly and requires a nonempty detail.

Run `.venv/bin/python scripts/datetime_reference.py --self-check` for the
exhaustive internal cross-checks (every day of the range, every year's Easter
by two algorithms, and `datetime.date` over years 1 to 9999), and
`--dateutil` (with `python-dateutil` available, for example through
`uv run --python 3.11 --no-project --with python-dateutil`) to compare both
Easter computations with `dateutil.easter`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as pydt
import random
import sys

try:  # optional cross-check only; the reference itself is standard library
    from dateutil import easter as dateutil_easter
    from dateutil.relativedelta import relativedelta
except ImportError:
    dateutil_easter = None
    relativedelta = None

# §4 supported range.
MIN_YEAR = -9999
MAX_YEAR = 9999
I64_MIN = -(2**63)
I64_MAX = 2**63 - 1
NANOS_PER_SECOND = 10**9
SECONDS_PER_DAY = 86_400
NANOS_PER_DAY = SECONDS_PER_DAY * NANOS_PER_SECOND
MIN_EPOCH_DAY = -4_371_587
MAX_EPOCH_DAY = 2_932_896
CIVIL_MIN_SECOND = MIN_EPOCH_DAY * SECONDS_PER_DAY
CIVIL_MAX_SECOND = MAX_EPOCH_DAY * SECONDS_PER_DAY + SECONDS_PER_DAY - 1
OFFSET_MAX_SECONDS = 86_399
INSTANT_MIN_SECOND = CIVIL_MIN_SECOND + OFFSET_MAX_SECONDS
INSTANT_MAX_SECOND = CIVIL_MAX_SECOND - OFFSET_MAX_SECONDS
UNIX_EPOCH_ORDINAL = 719_163  # datetime.date(1970, 1, 1).toordinal()

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DAY_OVERFLOW = ("ClampToMonthEnd", "RejectInvalidDay")
TIME_ROUNDING = ("RoundTowardPast", "RoundTowardFuture", "RoundTowardZero", "RoundNearestTiesEven")
TIME_UNIT_NANOS = {
    "Hours": 3_600 * NANOS_PER_SECOND,
    "Minutes": 60 * NANOS_PER_SECOND,
    "Seconds": NANOS_PER_SECOND,
    "Milliseconds": 1_000_000,
    "Microseconds": 1_000,
    "Nanoseconds": 1,
}
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@dataclass(frozen=True)
class DatetimeError(Exception):
    """A §5 failure: `<function>: <kind>: <detail>`."""

    function: str
    kind: str
    detail: str

    def __post_init__(self) -> None:
        if self.kind not in ("domain", "overflow"):
            raise ValueError(f"unknown failure kind {self.kind!r}")

    @property
    def message(self) -> str:
        return f"{self.function}: {self.kind}: {self.detail}"

    def __str__(self) -> str:
        return self.message


def domain(function: str, detail: str) -> DatetimeError:
    return DatetimeError(function, "domain", detail)


def overflow(function: str, detail: str) -> DatetimeError:
    return DatetimeError(function, "overflow", detail)


def fits_i64(value: int) -> bool:
    return I64_MIN <= value <= I64_MAX


def in_year_range(year: int) -> bool:
    return MIN_YEAR <= year <= MAX_YEAR


def in_day_range(epoch_day: int) -> bool:
    return MIN_EPOCH_DAY <= epoch_day <= MAX_EPOCH_DAY


# ---------------------------------------------------------------------------
# §8.1 calendar queries: total over every integer year.


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_year(year: int) -> int:
    return 366 if is_leap_year(year) else 365


def days_in_month(year: int, month: int, function: str = "days_in_month") -> int:
    if not 1 <= month <= 12:
        raise domain(function, f"month {month} is outside 1..12")
    if month == 2 and is_leap_year(year):
        return 29
    return MONTH_DAYS[month - 1]


def weekday_iso_number(weekday: int) -> int:
    return weekday


def weekday_from_iso_number(number: int, function: str = "weekday_from_iso_number") -> int:
    if not 1 <= number <= 7:
        raise domain(function, f"ISO weekday number {number} is outside 1..7")
    return number


def weekday_name(weekday: int) -> str:
    return WEEKDAYS[weekday - 1].lower()


# ---------------------------------------------------------------------------
# Civil <-> epoch day: Howard Hinnant, "chrono-Compatible Low-Level Date
# Algorithms", days_from_civil and civil_from_days, with Python's floor
# division standing in for the paper's era adjustment.


def days_from_civil(year: int, month: int, day: int) -> int:
    y = year - (1 if month <= 2 else 0)
    era = y // 400
    yoe = y - era * 400
    mp = month - 3 if month > 2 else month + 9
    doy = (153 * mp + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146_097 + doe - 719_468


def civil_from_days(epoch_day: int) -> tuple[int, int, int]:
    z = epoch_day + 719_468
    era = z // 146_097
    doe = z - era * 146_097
    yoe = (doe - doe // 1_460 + doe // 36_524 - doe // 146_096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    day = doy - (153 * mp + 2) // 5 + 1
    month = mp + 3 if mp < 10 else mp - 9
    year = yoe + era * 400 + (1 if month <= 2 else 0)
    return year, month, day


def leap_years_through(year: int) -> int:
    """Leap years in (0, year], negative for year < 0; exact for every integer."""
    return year // 4 - year // 100 + year // 400


def days_from_civil_by_counting(year: int, month: int, day: int) -> int:
    """A second, independent derivation: whole years, then months, then days."""
    days = 365 * (year - 1970) + leap_years_through(year - 1) - leap_years_through(1969)
    days += sum(MONTH_DAYS[: month - 1])
    if month > 2 and is_leap_year(year):
        days += 1
    return days + day - 1


def weekday_of(epoch_day: int) -> int:
    """ISO weekday number; 1970-01-01 is a Thursday (4)."""
    return (epoch_day + 3) % 7 + 1


def day_of_year(epoch_day: int) -> int:
    year, _, _ = civil_from_days(epoch_day)
    return epoch_day - days_from_civil(year, 1, 1) + 1


def iso_week(epoch_day: int) -> tuple[int, int]:
    """ISO 8601 week-year and week: the week belongs to the year of its Thursday."""
    thursday = epoch_day + 4 - weekday_of(epoch_day)
    year, _, _ = civil_from_days(thursday)
    return year, (thursday - days_from_civil(year, 1, 1)) // 7 + 1


def iso_weeks_in_year(iso_year: int) -> int:
    """28 December always lies in the last ISO week of its year."""
    return iso_week(days_from_civil(iso_year, 12, 28))[1]


# ---------------------------------------------------------------------------
# §8.2 Date.


def check_date_fields(function: str, year: int, month: int, day: int) -> None:
    if not in_year_range(year):
        raise domain(function, f"year {year} is outside {MIN_YEAR}..{MAX_YEAR}")
    if not 1 <= month <= 12:
        raise domain(function, f"month {month} is outside 1..12")
    last = days_in_month(year, month)
    if not 1 <= day <= last:
        raise domain(function, f"day {day} is outside 1..{last} for {year_text(year)}-{month:02d}")


def date(year: int, month: int, day: int, function: str = "date") -> int:
    check_date_fields(function, year, month, day)
    return days_from_civil(year, month, day)


def date_fields(epoch_day: int) -> tuple[int, int, int]:
    return civil_from_days(epoch_day)


def date_from_epoch_day(epoch_day: int, function: str = "date_from_epoch_day") -> int:
    if not in_day_range(epoch_day):
        raise domain(function, f"epoch day {epoch_day} is outside {MIN_EPOCH_DAY}..{MAX_EPOCH_DAY}")
    return epoch_day


def date_from_iso_week(iso_year: int, week: int, weekday: int, function: str = "date_from_iso_week") -> int:
    # Only week-years -10000..10000 can name a day of the range; checking the
    # year first keeps the week count's own arithmetic bounded.
    if not MIN_YEAR - 1 <= iso_year <= MAX_YEAR + 1:
        raise domain(function, f"ISO week-year {iso_year} has no day in the supported range")
    weeks = iso_weeks_in_year(iso_year)
    if not 1 <= week <= weeks:
        raise domain(function, f"week {week} is outside 1..{weeks} for ISO week-year {iso_year}")
    jan4 = days_from_civil(iso_year, 1, 4)
    result = jan4 - (weekday_of(jan4) - 1) + 7 * (week - 1) + (weekday - 1)
    if not in_day_range(result):
        raise domain(function, f"ISO week {iso_year}-W{week:02d}-{weekday} is outside the supported range")
    return result


def date_add_days(epoch_day: int, days: int, function: str = "date_add_days") -> int:
    result = epoch_day + days
    if not in_day_range(result):
        raise overflow(function, f"{date_to_string(epoch_day)} plus {days} days leaves the supported range")
    return result


def date_days_until(a: int, b: int) -> int:
    return b - a


def date_add_months(epoch_day: int, months: int, policy: str, function: str = "date_add_months") -> int:
    if policy not in DAY_OVERFLOW:
        raise ValueError(policy)
    year, month, day = civil_from_days(epoch_day)
    total = 12 * year + (month - 1) + months
    target_year, target_month0 = divmod(total, 12)
    target_month = target_month0 + 1
    if not in_year_range(target_year):
        raise overflow(function, f"{date_to_string(epoch_day)} plus {months} months leaves the supported range")
    last = days_in_month(target_year, target_month)
    if day > last:
        if policy == "RejectInvalidDay":
            raise domain(
                function,
                f"day {day} does not exist in {year_text(target_year)}-{target_month:02d}",
            )
        day = last
    return days_from_civil(target_year, target_month, day)


def date_add_period(epoch_day: int, period_value: tuple[int, int], policy: str, function: str = "date_add_period") -> int:
    months, days = period_value
    return date_add_days(date_add_months(epoch_day, months, policy, function), days, function)


def date_period_until(a: int, b: int) -> tuple[int, int]:
    """§8.2: the single-sign period of largest-magnitude months that maps a to b."""
    ya, ma, _ = civil_from_days(a)
    yb, mb, _ = civil_from_days(b)
    months = 12 * (yb - ya) + (mb - ma)
    if a <= b:
        while date_add_months(a, months, "ClampToMonthEnd") > b:
            months -= 1
    else:
        while date_add_months(a, months, "ClampToMonthEnd") < b:
            months += 1
    return months, b - date_add_months(a, months, "ClampToMonthEnd")


# ---------------------------------------------------------------------------
# §8.3 holiday-rule helpers.


def check_year_month(function: str, year: int, month: int) -> None:
    if not 1 <= month <= 12:
        raise domain(function, f"month {month} is outside 1..12")
    if not in_year_range(year):
        raise domain(function, f"year {year} is outside {MIN_YEAR}..{MAX_YEAR}")


def nth_weekday_in_month(year: int, month: int, weekday: int, n: int, function: str = "nth_weekday_in_month") -> int | None:
    if n <= 0:
        raise domain(function, f"occurrence {n} is not positive")
    check_year_month(function, year, month)
    first = days_from_civil(year, month, 1)
    candidate = first + (weekday - weekday_of(first)) % 7 + 7 * (n - 1)
    if candidate > first + days_in_month(year, month) - 1:
        return None
    return candidate


def last_weekday_in_month(year: int, month: int, weekday: int, function: str = "last_weekday_in_month") -> int:
    check_year_month(function, year, month)
    last = days_from_civil(year, month, days_in_month(year, month))
    return last - (weekday_of(last) - weekday) % 7


def weekday_on_or_after(epoch_day: int, weekday: int, function: str = "weekday_on_or_after") -> int:
    result = epoch_day + (weekday - weekday_of(epoch_day)) % 7
    if not in_day_range(result):
        raise overflow(function, f"the next {weekday_name(weekday)} after {date_to_string(epoch_day)} leaves the supported range")
    return result


def weekday_on_or_before(epoch_day: int, weekday: int, function: str = "weekday_on_or_before") -> int:
    result = epoch_day - (weekday_of(epoch_day) - weekday) % 7
    if not in_day_range(result):
        raise overflow(function, f"the previous {weekday_name(weekday)} before {date_to_string(epoch_day)} leaves the supported range")
    return result


def easter_gregorian_meeus(year: int) -> tuple[int, int]:
    """Anonymous Gregorian algorithm (Meeus, Astronomical Algorithms, ch. 8)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day0 = divmod(h + l - 7 * m + 114, 31)
    return month, day0 + 1


def easter_gregorian_epact(year: int) -> tuple[int, int]:
    """Lilius and Clavius by golden number and epact (Knuth, TAOCP 1.3.2 ex. 14)."""
    golden = year % 19 + 1
    century = year // 100 + 1
    solar = 3 * century // 4 - 12
    lunar = (8 * century + 5) // 25 - 5
    sunday = 5 * year // 4 - solar - 10
    epact = (11 * golden + 20 + lunar - solar) % 30
    if (epact == 25 and golden > 11) or epact == 24:
        epact += 1
    full_moon = 44 - epact
    if full_moon < 21:
        full_moon += 30
    full_moon += 7 - (sunday + full_moon) % 7
    return (4, full_moon - 31) if full_moon > 31 else (3, full_moon)


def julian_to_epoch_day(year: int, month: int, day: int) -> int:
    """Julian calendar date to epoch day through the Julian day number."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - 32_083
    return jdn - 2_440_588


def julian_to_epoch_day_by_counting(year: int, month: int, day: int) -> int:
    """A second derivation: Julian leap years are exactly the multiples of 4."""
    days = 365 * year + (year - 1) // 4 + 1  # days before Julian 1 January of `year`, from 0-01-01
    days += sum(MONTH_DAYS[: month - 1])
    if month > 2 and year % 4 == 0:
        days += 1
    days += day - 1
    # Julian 0-01-01 is proleptic Gregorian -0001-12-30.
    return days + days_from_civil(-1, 12, 30)


def easter_julian_meeus(year: int) -> tuple[int, int]:
    """Julian computus (Meeus): a Julian calendar month and day."""
    a = year % 4
    b = year % 7
    c = year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month, day0 = divmod(d + e + 114, 31)
    return month, day0 + 1


def easter_orthodox_by_full_moon(year: int) -> int:
    """Julian paschal full moon, then the first Sunday strictly after it."""
    full_moon = julian_to_epoch_day_by_counting(year, 3, 21) + (19 * (year % 19) + 15) % 30
    return full_moon + 7 - (weekday_of(full_moon) % 7)


def easter_sunday_gregorian(year: int, function: str = "easter_sunday_gregorian") -> int:
    if not in_year_range(year):
        raise domain(function, f"year {year} is outside {MIN_YEAR}..{MAX_YEAR}")
    month, day = easter_gregorian_meeus(year)
    return days_from_civil(year, month, day)


def easter_sunday_orthodox(year: int, function: str = "easter_sunday_orthodox") -> int:
    if not in_year_range(year):
        raise domain(function, f"year {year} is outside {MIN_YEAR}..{MAX_YEAR}")
    month, day = easter_julian_meeus(year)
    result = julian_to_epoch_day(year, month, day)
    if not in_day_range(result):
        raise overflow(function, f"Orthodox Easter of {year} leaves the supported range")
    return result


# ---------------------------------------------------------------------------
# Rounding of an exact rational quotient (TimeRounding).


def round_quotient(numerator: int, denominator: int, mode: str) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    floor, remainder = divmod(numerator, denominator)
    if remainder == 0:
        return floor
    if mode == "RoundTowardPast":
        return floor
    if mode == "RoundTowardFuture":
        return floor + 1
    if mode == "RoundTowardZero":
        return floor + 1 if numerator < 0 else floor
    if mode == "RoundNearestTiesEven":
        twice = 2 * remainder
        if twice < denominator:
            return floor
        if twice > denominator:
            return floor + 1
        return floor if floor % 2 == 0 else floor + 1
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# §8.6 Duration (second, nanosecond) and Period (months, days).


def split_nanos(total: int) -> tuple[int, int]:
    return divmod(total, NANOS_PER_SECOND)


def nanos_of(value: tuple[int, int]) -> int:
    return value[0] * NANOS_PER_SECOND + value[1]


def make_duration(total_nanos: int, function: str) -> tuple[int, int]:
    second, nano = split_nanos(total_nanos)
    if not fits_i64(second):
        raise overflow(function, f"a duration of {total_nanos} ns does not fit")
    return second, nano


def duration(second: int, nanosecond: int, function: str = "duration") -> tuple[int, int]:
    return make_duration(second * NANOS_PER_SECOND + nanosecond, function)


def duration_from_count(count: int, unit: str, function: str = "duration_from_count") -> tuple[int, int]:
    return make_duration(count * TIME_UNIT_NANOS[unit], function)


def count_in_unit(total_nanos: int, unit: str, rounding: str, function: str) -> int:
    result = round_quotient(total_nanos, TIME_UNIT_NANOS[unit], rounding)
    if not fits_i64(result):
        raise overflow(function, f"{total_nanos} ns is {result} {unit.lower()}, which does not fit in i64")
    return result


def duration_to_count(d: tuple[int, int], unit: str, rounding: str, function: str = "duration_to_count") -> int:
    return count_in_unit(nanos_of(d), unit, rounding, function)


def try_duration_to_count(d: tuple[int, int], unit: str, rounding: str) -> int | None:
    try:
        return duration_to_count(d, unit, rounding)
    except DatetimeError:
        return None


def same_sign_parts(d: tuple[int, int]) -> tuple[int, int]:
    """§8.6: a whole part and a nanosecond fraction that share the duration's sign."""
    second, nano = d
    if second >= 0 or nano == 0:
        return second, nano
    return second + 1, nano - NANOS_PER_SECOND


def duration_to_seconds_f64(d: tuple[int, int]) -> float:
    """The fixed composition of §8.6; Python's int->float and float ops round to nearest even."""
    whole, fraction = same_sign_parts(d)
    return float(whole) + float(fraction) / 1e9


def duration_add(a: tuple[int, int], b: tuple[int, int], function: str = "duration_add") -> tuple[int, int]:
    return make_duration(nanos_of(a) + nanos_of(b), function)


def duration_sub(a: tuple[int, int], b: tuple[int, int], function: str = "duration_sub") -> tuple[int, int]:
    return make_duration(nanos_of(a) - nanos_of(b), function)


def duration_negate(d: tuple[int, int], function: str = "duration_negate") -> tuple[int, int]:
    return make_duration(-nanos_of(d), function)


def duration_mul(d: tuple[int, int], k: int, function: str = "duration_mul") -> tuple[int, int]:
    return make_duration(nanos_of(d) * k, function)


def period(months: int, days: int, function: str = "period") -> tuple[int, int]:
    if (months > 0 and days < 0) or (months < 0 and days > 0):
        raise domain(function, f"{months} months and {days} days have mixed signs")
    return months, days


def period_negate(p: tuple[int, int], function: str = "period_negate") -> tuple[int, int]:
    months, days = -p[0], -p[1]
    if not (fits_i64(months) and fits_i64(days)):
        raise overflow(function, f"the negation of {period_to_string(p)} does not fit in i64")
    return months, days


def period_mul(p: tuple[int, int], k: int, function: str = "period_mul") -> tuple[int, int]:
    months, days = p[0] * k, p[1] * k
    if not (fits_i64(months) and fits_i64(days)):
        raise overflow(function, f"{period_to_string(p)} times {k} does not fit in i64")
    return months, days


# ---------------------------------------------------------------------------
# §8.4 Time and DateTime.


def time(hour: int, minute: int, second: int, nanosecond: int, function: str = "time") -> int:
    if not 0 <= hour <= 23:
        raise domain(function, f"hour {hour} is outside 0..23")
    if not 0 <= minute <= 59:
        raise domain(function, f"minute {minute} is outside 0..59")
    if not 0 <= second <= 59:
        raise domain(function, f"second {second} is outside 0..59")
    if not 0 <= nanosecond < NANOS_PER_SECOND:
        raise domain(function, f"nanosecond {nanosecond} is outside 0..999999999")
    return ((hour * 60 + minute) * 60 + second) * NANOS_PER_SECOND + nanosecond


def time_fields(nanosecond_of_day: int) -> tuple[int, int, int, int]:
    seconds, nano = divmod(nanosecond_of_day, NANOS_PER_SECOND)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return hour, minute, second, nano


def time_from_nanosecond_of_day(n: int, function: str = "time_from_nanosecond_of_day") -> int:
    if not 0 <= n < NANOS_PER_DAY:
        raise domain(function, f"nanosecond of day {n} is outside 0..{NANOS_PER_DAY - 1}")
    return n


def time_add_duration(t: int, d: tuple[int, int]) -> tuple[int, int]:
    return divmod(t + nanos_of(d), NANOS_PER_DAY)


def time_until(a: int, b: int) -> tuple[int, int]:
    return split_nanos(b - a)


def civil_nanos(dt: tuple[int, int]) -> int:
    return dt[0] * NANOS_PER_DAY + dt[1]


def datetime_from_civil_nanos(total: int, function: str) -> tuple[int, int]:
    epoch_day, nod = divmod(total, NANOS_PER_DAY)
    if not in_day_range(epoch_day):
        raise overflow(function, "the result leaves the supported civil range")
    return epoch_day, nod


def datetime_add_duration(dt: tuple[int, int], d: tuple[int, int], function: str = "datetime_add_duration") -> tuple[int, int]:
    return datetime_from_civil_nanos(civil_nanos(dt) + nanos_of(d), function)


def datetime_add_period(dt: tuple[int, int], p: tuple[int, int], policy: str, function: str = "datetime_add_period") -> tuple[int, int]:
    return date_add_period(dt[0], p, policy, function), dt[1]


def datetime_until(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return split_nanos(civil_nanos(b) - civil_nanos(a))


# ---------------------------------------------------------------------------
# §8.5 Offset, Instant, OffsetDateTime.


def offset_from_seconds(seconds: int, function: str = "offset_from_seconds") -> int:
    if not -OFFSET_MAX_SECONDS <= seconds <= OFFSET_MAX_SECONDS:
        raise domain(function, f"offset {seconds} s is outside -86399..86399")
    return seconds


def in_instant_range(second: int) -> bool:
    return INSTANT_MIN_SECOND <= second <= INSTANT_MAX_SECOND


def instant_from_unix(second: int, nanosecond: int, function: str = "instant_from_unix") -> tuple[int, int]:
    if not 0 <= nanosecond < NANOS_PER_SECOND:
        raise domain(function, f"nanosecond {nanosecond} is outside 0..999999999")
    if not in_instant_range(second):
        raise domain(function, f"unix second {second} is outside {INSTANT_MIN_SECOND}..{INSTANT_MAX_SECOND}")
    return second, nanosecond


def instant_from_unix_count(count: int, unit: str, function: str = "instant_from_unix_count") -> tuple[int, int]:
    second, nano = split_nanos(count * TIME_UNIT_NANOS[unit])
    if not in_instant_range(second):
        raise domain(function, f"{count} {unit.lower()} since the unix epoch is outside the instant range")
    return second, nano


def instant_to_unix_count(i: tuple[int, int], unit: str, rounding: str, function: str = "instant_to_unix_count") -> int:
    return count_in_unit(nanos_of(i), unit, rounding, function)


def try_instant_to_unix_count(i: tuple[int, int], unit: str, rounding: str) -> int | None:
    try:
        return instant_to_unix_count(i, unit, rounding)
    except DatetimeError:
        return None


def instant_from_nanos(total: int, function: str) -> tuple[int, int]:
    second, nano = split_nanos(total)
    if not in_instant_range(second):
        raise overflow(function, "the result leaves the instant range")
    return second, nano


def instant_add_duration(i: tuple[int, int], d: tuple[int, int], function: str = "instant_add_duration") -> tuple[int, int]:
    return instant_from_nanos(nanos_of(i) + nanos_of(d), function)


def instant_until(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return split_nanos(nanos_of(b) - nanos_of(a))


def instant_round_to(i: tuple[int, int], increment: tuple[int, int], rounding: str, function: str = "instant_round_to") -> tuple[int, int]:
    step = nanos_of(increment)
    if step <= 0 or NANOS_PER_DAY % step != 0:
        raise domain(function, f"increment {duration_to_string(increment)} is not a positive divisor of 86400 s")
    return instant_from_nanos(round_quotient(nanos_of(i), step, rounding) * step, function)


def instant_to_datetime_at(i: tuple[int, int], offset_seconds: int) -> tuple[int, int]:
    epoch_day, second_of_day = divmod(i[0] + offset_seconds, SECONDS_PER_DAY)
    return epoch_day, second_of_day * NANOS_PER_SECOND + i[1]


def datetime_to_instant_at(dt: tuple[int, int], offset_seconds: int, function: str = "datetime_to_instant_at") -> tuple[int, int]:
    return instant_from_nanos(civil_nanos(dt) - offset_seconds * NANOS_PER_SECOND, function)


# ---------------------------------------------------------------------------
# §8.8 text profile: canonical formatting.


def year_text(year: int) -> str:
    return f"{year:04d}" if year >= 0 else f"-{-year:06d}"


def fraction_text(nanosecond: int) -> str:
    return "" if nanosecond == 0 else "." + f"{nanosecond:09d}".rstrip("0")


def date_to_string(epoch_day: int) -> str:
    year, month, day = civil_from_days(epoch_day)
    return f"{year_text(year)}-{month:02d}-{day:02d}"


def time_to_string(t: int) -> str:
    hour, minute, second, nano = time_fields(t)
    return f"{hour:02d}:{minute:02d}:{second:02d}{fraction_text(nano)}"


def datetime_to_string(dt: tuple[int, int]) -> str:
    return f"{date_to_string(dt[0])}T{time_to_string(dt[1])}"


def offset_to_string(seconds: int) -> str:
    if seconds == 0:
        return "Z"
    sign = "+" if seconds > 0 else "-"
    magnitude = abs(seconds)
    text = f"{sign}{magnitude // 3600:02d}:{magnitude // 60 % 60:02d}"
    return text + (f":{magnitude % 60:02d}" if magnitude % 60 else "")


def instant_to_string(i: tuple[int, int]) -> str:
    return datetime_to_string(instant_to_datetime_at(i, 0)) + "Z"


def offset_datetime_to_string(odt: tuple[tuple[int, int], int]) -> str:
    instant, offset_seconds = odt
    return datetime_to_string(instant_to_datetime_at(instant, offset_seconds)) + offset_to_string(offset_seconds)


def duration_to_string(d: tuple[int, int]) -> str:
    total = nanos_of(d)
    sign = "-" if total < 0 else ""
    whole, nano = divmod(abs(total), NANOS_PER_SECOND)
    return f"{sign}PT{whole}{fraction_text(nano)}S"


def period_to_string(p: tuple[int, int]) -> str:
    months, days = p
    if months == 0 and days == 0:
        return "P0D"
    sign = "-" if months < 0 or days < 0 else ""
    text = f"{sign}P"
    if months:
        text += f"{abs(months)}M"
    if days:
        text += f"{abs(days)}D"
    return text


# ---------------------------------------------------------------------------
# §8.8 text profile: parsing. Literals are case-sensitive except where the
# grammar lists both cases ("T"/"t", "Z"/"z"); digits are ASCII only.

DIGITS = "0123456789"


class _Scanner:
    def __init__(self, function: str, text: str) -> None:
        self.function = function
        self.text = text
        self.pos = 0

    def fail(self, reason: str) -> DatetimeError:
        return domain(self.function, f"{reason} in {self.text!r}")

    def at_end(self) -> bool:
        return self.pos == len(self.text)

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def take(self, literal: str) -> bool:
        if self.text.startswith(literal, self.pos):
            self.pos += len(literal)
            return True
        return False

    def expect(self, literal: str) -> None:
        if not self.take(literal):
            raise self.fail(f"expected {literal!r} at offset {self.pos}")

    def digits(self, exact: int | None = None, at_most: int | None = None) -> str:
        start = self.pos
        while self.peek() != "" and self.peek() in DIGITS:
            self.pos += 1
        found = self.text[start : self.pos]
        if not found:
            raise self.fail(f"expected a digit at offset {start}")
        if exact is not None and len(found) != exact:
            raise self.fail(f"expected {exact} digits at offset {start}")
        if at_most is not None and len(found) > at_most:
            raise self.fail(f"more than {at_most} fractional digits at offset {start}")
        return found

    def finish(self) -> None:
        if not self.at_end():
            raise self.fail(f"unexpected text at offset {self.pos}")


def _scan_date(s: _Scanner) -> int:
    if s.peek() in ("+", "-"):
        negative = s.peek() == "-"
        s.pos += 1
        digits = s.digits(exact=6)
        if negative and digits == "000000":
            raise s.fail("year -000000 is not valid")
        year = -int(digits) if negative else int(digits)
    else:
        year = int(s.digits(exact=4))
    s.expect("-")
    month = int(s.digits(exact=2))
    s.expect("-")
    day = int(s.digits(exact=2))
    try:
        check_date_fields(s.function, year, month, day)
    except DatetimeError as error:
        raise s.fail(error.detail) from None
    return days_from_civil(year, month, day)


def _scan_time(s: _Scanner) -> int:
    hour = int(s.digits(exact=2))
    s.expect(":")
    minute = int(s.digits(exact=2))
    s.expect(":")
    second = int(s.digits(exact=2))
    nano = 0
    if s.take("."):
        fraction = s.digits(at_most=9)
        nano = int(fraction.ljust(9, "0"))
    try:
        return time(hour, minute, second, nano, s.function)
    except DatetimeError as error:
        raise s.fail(error.detail) from None


def _scan_datetime(s: _Scanner) -> tuple[int, int]:
    epoch_day = _scan_date(s)
    if not (s.take("T") or s.take("t") or s.take(" ")):
        raise s.fail(f"expected 'T', 't' or ' ' at offset {s.pos}")
    return epoch_day, _scan_time(s)


def _scan_offset(s: _Scanner) -> int:
    if s.take("Z") or s.take("z"):
        return 0
    sign = s.peek()
    if sign not in ("+", "-"):
        raise s.fail(f"expected an offset at offset {s.pos}")
    s.pos += 1
    hour = int(s.digits(exact=2))
    s.expect(":")
    minute = int(s.digits(exact=2))
    second = 0
    if s.take(":"):
        second = int(s.digits(exact=2))
    if hour > 23 or minute > 59 or second > 59:
        raise s.fail(f"offset field out of range ({hour:02d}:{minute:02d}:{second:02d})")
    magnitude = hour * 3600 + minute * 60 + second
    return -magnitude if sign == "-" else magnitude


def parse_date(text: str, function: str = "parse_date") -> int:
    s = _Scanner(function, text)
    epoch_day = _scan_date(s)
    s.finish()
    return epoch_day


def parse_time(text: str, function: str = "parse_time") -> int:
    s = _Scanner(function, text)
    t = _scan_time(s)
    s.finish()
    return t


def parse_datetime(text: str, function: str = "parse_datetime") -> tuple[int, int]:
    s = _Scanner(function, text)
    dt = _scan_datetime(s)
    s.finish()
    return dt


def parse_offset(text: str, function: str = "parse_offset") -> int:
    s = _Scanner(function, text)
    offset_seconds = _scan_offset(s)
    s.finish()
    return offset_seconds


def parse_offset_datetime(text: str, function: str = "parse_offset_datetime") -> tuple[tuple[int, int], int]:
    s = _Scanner(function, text)
    dt = _scan_datetime(s)
    offset_seconds = _scan_offset(s)
    s.finish()
    second, nano = split_nanos(civil_nanos(dt) - offset_seconds * NANOS_PER_SECOND)
    if not in_instant_range(second):
        raise s.fail("the instant is outside the supported range")
    return (second, nano), offset_seconds


def parse_instant(text: str, function: str = "parse_instant") -> tuple[int, int]:
    return parse_offset_datetime(text, function)[0]


def _scan_designated(s: _Scanner, designators: str) -> dict[str, str]:
    """Components in grammar order; each is digits followed by its designator."""
    found: dict[str, str] = {}
    order = list(designators)
    while not s.at_end():
        start = s.pos
        digits = s.digits()
        fraction = None
        if "S" in order and s.peek() == ".":
            s.pos += 1
            fraction = s.digits(at_most=9)
        designator = s.peek()
        if designator not in order:
            raise s.fail(f"expected one of {''.join(order)!r} at offset {s.pos}")
        if fraction is not None and designator != "S":
            raise s.fail(f"a fraction is allowed only on seconds at offset {start}")
        order = order[order.index(designator) + 1 :]
        s.pos += 1
        found[designator] = digits if fraction is None else f"{digits}.{fraction}"
    if not found:
        raise s.fail("at least one component is required")
    return found


def parse_duration(text: str, function: str = "parse_duration") -> tuple[int, int]:
    s = _Scanner(function, text)
    negative = s.take("-")
    s.expect("PT")
    parts = _scan_designated(s, "HMS")
    whole, _, fraction = parts.get("S", "0").partition(".")
    total = (int(parts.get("H", "0")) * 3600 + int(parts.get("M", "0")) * 60 + int(whole)) * NANOS_PER_SECOND
    total += int(fraction.ljust(9, "0")) if fraction else 0
    if negative:
        total = -total
    second, nano = split_nanos(total)
    if not fits_i64(second):
        raise s.fail("the duration does not fit")
    return second, nano


def parse_period(text: str, function: str = "parse_period") -> tuple[int, int]:
    s = _Scanner(function, text)
    negative = s.take("-")
    s.expect("P")
    parts = _scan_designated(s, "YMWD")
    months = int(parts.get("Y", "0")) * 12 + int(parts.get("M", "0"))
    days = int(parts.get("W", "0")) * 7 + int(parts.get("D", "0"))
    if negative:
        months, days = -months, -days
    if not (fits_i64(months) and fits_i64(days)):
        raise s.fail("the period does not fit in i64")
    return months, days


PARSERS = {
    "date": (parse_date, date_to_string),
    "time": (parse_time, time_to_string),
    "datetime": (parse_datetime, datetime_to_string),
    "offset": (parse_offset, offset_to_string),
    "instant": (parse_instant, instant_to_string),
    "offset_datetime": (parse_offset_datetime, offset_datetime_to_string),
    "duration": (parse_duration, duration_to_string),
    "period": (parse_period, period_to_string),
}


def canonical(kind: str, text: str) -> str:
    """`format(parse(text))`, the canonical spelling §17 compares against."""
    parse, render = PARSERS[kind]
    return render(parse(text))


# ---------------------------------------------------------------------------
# Self-checks: independent derivations agree with the functions above.


def check_range_constants() -> None:
    assert days_from_civil(MIN_YEAR, 1, 1) == MIN_EPOCH_DAY
    assert days_from_civil(MAX_YEAR, 12, 31) == MAX_EPOCH_DAY
    assert MAX_EPOCH_DAY - MIN_EPOCH_DAY + 1 == 7_304_484
    assert CIVIL_MIN_SECOND == -377_705_116_800
    assert CIVIL_MAX_SECOND == 253_402_300_799
    assert INSTANT_MIN_SECOND == -377_705_030_401
    assert INSTANT_MAX_SECOND == 253_402_214_400
    assert INSTANT_MAX_SECOND - INSTANT_MIN_SECOND == 631_107_244_801 < 2**40


def walk_every_day(report=print) -> int:
    """Walk the whole range by the successor function and check every day."""
    year, month, day = MIN_YEAR, 1, 1
    count = 0
    weekday = weekday_of(MIN_EPOCH_DAY)
    for epoch_day in range(MIN_EPOCH_DAY, MAX_EPOCH_DAY + 1):
        if civil_from_days(epoch_day) != (year, month, day):
            raise AssertionError(f"civil_from_days({epoch_day}) != {(year, month, day)}")
        if days_from_civil(year, month, day) != epoch_day:
            raise AssertionError(f"days_from_civil{(year, month, day)} != {epoch_day}")
        if weekday_of(epoch_day) != weekday:
            raise AssertionError(f"weekday of {epoch_day}")
        iso_year, week = iso_week(epoch_day)
        if date_from_iso_week(iso_year, week, weekday) != epoch_day:
            raise AssertionError(f"ISO week round trip of {epoch_day}")
        if 1 <= year <= 9999:
            native = pydt.date(year, month, day)
            if native.toordinal() - UNIX_EPOCH_ORDINAL != epoch_day:
                raise AssertionError(f"toordinal disagrees at {epoch_day}")
            if tuple(native.isocalendar()) != (iso_year, week, weekday):
                raise AssertionError(f"isocalendar disagrees at {epoch_day}")
            if native.timetuple().tm_yday != day_of_year(epoch_day):
                raise AssertionError(f"day of year disagrees at {epoch_day}")
        count += 1
        weekday = weekday % 7 + 1
        day += 1
        if day > days_in_month(year, month):
            day, month = 1, month + 1
            if month > 12:
                month, year = 1, year + 1
                if year % 2000 == 0:
                    report(f"walked through {year - 1}")
    return count


def check_every_year() -> None:
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        if days_from_civil_by_counting(year, 1, 1) != days_from_civil(year, 1, 1):
            raise AssertionError(f"counting derivation disagrees for {year}")
        if easter_gregorian_meeus(year) != easter_gregorian_epact(year):
            raise AssertionError(f"Gregorian Easter algorithms disagree for {year}")
        if julian_to_epoch_day(year, 3, 21) != julian_to_epoch_day_by_counting(year, 3, 21):
            raise AssertionError(f"Julian conversions disagree for {year}")
        if easter_sunday_orthodox(year) != easter_orthodox_by_full_moon(year):
            raise AssertionError(f"Orthodox Easter algorithms disagree for {year}")


def compare_with_dateutil() -> str:
    """dateutil's Western and Julian methods for 1..9999; Orthodox where dateutil documents it.

    `dateutil.easter` documents its methods as valid for 1583 to 4099. Its
    Western (Oudin) and Julian methods agree with the computations here over
    every year 1 to 9999, so both are compared there. Its Orthodox method
    converts Julian to Gregorian with a correction that is exact only inside
    the documented span, so it is compared only there.
    """
    if dateutil_easter is None:
        raise SystemExit("--dateutil needs python-dateutil; run under uv with --with python-dateutil")
    for year in range(1, MAX_YEAR + 1):
        western = dateutil_easter.easter(year, dateutil_easter.EASTER_WESTERN)
        if western.toordinal() - UNIX_EPOCH_ORDINAL != easter_sunday_gregorian(year):
            raise AssertionError(f"dateutil Western Easter disagrees for {year}: {western}")
        julian = dateutil_easter.easter(year, dateutil_easter.EASTER_JULIAN)
        if (julian.month, julian.day) != easter_julian_meeus(year):
            raise AssertionError(f"dateutil Julian Easter disagrees for {year}: {julian}")
    for year in range(1583, 4100):
        orthodox = dateutil_easter.easter(year, dateutil_easter.EASTER_ORTHODOX)
        if orthodox.toordinal() - UNIX_EPOCH_ORDINAL != easter_sunday_orthodox(year):
            raise AssertionError(f"dateutil Orthodox Easter disagrees for {year}: {orthodox}")
    return "Western and Julian 1..9999, Orthodox 1583..4099"


def compare_months_with_dateutil(seed: int = 2859) -> int:
    """ClampToMonthEnd against `relativedelta(months=n)`, which clamps the same way."""
    if relativedelta is None:
        raise SystemExit("--dateutil needs python-dateutil; run under uv with --with python-dateutil")
    rng = random.Random(seed)
    low, high = days_from_civil(1, 1, 1), days_from_civil(MAX_YEAR, 12, 31)
    checked = 0
    for _ in range(50_000):
        start = rng.randint(low, high)
        year, month, day = civil_from_days(start)
        months = rng.randint(-24, 24) if rng.random() < 0.5 else rng.randint(-12 * 9000, 12 * 9000)
        target_year = (12 * year + month - 1 + months) // 12
        if not 1 <= target_year <= MAX_YEAR:
            continue
        native = pydt.date(year, month, day) + relativedelta(months=months)
        if native.toordinal() - UNIX_EPOCH_ORDINAL != date_add_months(start, months, "ClampToMonthEnd"):
            raise AssertionError(f"relativedelta disagrees for {date_to_string(start)} + {months} months")
        checked += 1
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-check", action="store_true", help="walk every day and every year of the range")
    parser.add_argument("--dateutil", action="store_true", help="compare Easter and month arithmetic with dateutil")
    args = parser.parse_args(argv)
    if not (args.self_check or args.dateutil):
        parser.error("choose --self-check and/or --dateutil")
    check_range_constants()
    if args.self_check:
        check_every_year()
        days = walk_every_day()
        print(f"DATETIME REFERENCE SELF-CHECK: PASS ({days} days, {MAX_YEAR - MIN_YEAR + 1} years)")
    if args.dateutil:
        print(f"DATETIME REFERENCE DATEUTIL EASTER: PASS ({compare_with_dateutil()})")
        print(f"DATETIME REFERENCE DATEUTIL MONTHS: PASS ({compare_months_with_dateutil()} seeded additions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
