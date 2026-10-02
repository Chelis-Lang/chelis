#!/usr/bin/env python3
"""Unit tests for the Std.Datetime reference and differential harness (chelis#2859, chelis#2861).

The reference is the oracle the compiler is held to, so these tests hold the
reference to independent evidence: Python's `datetime`, a second derivation of
each algorithm, published Easter dates, and the defining properties the design
of record (`spec/design/std_datetime.md`) states. Each accepting check has a
rejecting partner. The exhaustive walk over all 7 304 484 days is the manual
`--self-check`; these tests sample it.
"""

from __future__ import annotations

import datetime as pydt
from fractions import Fraction
import random
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datetime_differential as harness  # noqa: E402
import datetime_reference as ref  # noqa: E402
from datetime_reference import DatetimeError, I64_MAX, I64_MIN  # noqa: E402

NANO = ref.NANOS_PER_SECOND
MESSAGE = re.compile(r"^[a-z0-9_]+: (domain|overflow): \S.*$")

# Western and Orthodox Easter Sundays (Gregorian dates), 2000 to 2030, as
# published in the standard Easter tables.
EASTER_2000_2030 = {
    2000: ((4, 23), (4, 30)), 2001: ((4, 15), (4, 15)), 2002: ((3, 31), (5, 5)), 2003: ((4, 20), (4, 27)),
    2004: ((4, 11), (4, 11)), 2005: ((3, 27), (5, 1)), 2006: ((4, 16), (4, 23)), 2007: ((4, 8), (4, 8)),
    2008: ((3, 23), (4, 27)), 2009: ((4, 12), (4, 19)), 2010: ((4, 4), (4, 4)), 2011: ((4, 24), (4, 24)),
    2012: ((4, 8), (4, 15)), 2013: ((3, 31), (5, 5)), 2014: ((4, 20), (4, 20)), 2015: ((4, 5), (4, 12)),
    2016: ((3, 27), (5, 1)), 2017: ((4, 16), (4, 16)), 2018: ((4, 1), (4, 8)), 2019: ((4, 21), (4, 28)),
    2020: ((4, 12), (4, 19)), 2021: ((4, 4), (5, 2)), 2022: ((4, 17), (4, 24)), 2023: ((4, 9), (4, 16)),
    2024: ((3, 31), (5, 5)), 2025: ((4, 20), (4, 20)), 2026: ((4, 5), (4, 12)), 2027: ((3, 28), (5, 2)),
    2028: ((4, 16), (4, 16)), 2029: ((4, 1), (4, 8)), 2030: ((4, 21), (4, 28)),
}


def ymd(year: int, month: int, day: int) -> int:
    return ref.days_from_civil(year, month, day)


def failure(compute) -> DatetimeError:
    try:
        compute()
    except DatetimeError as error:
        return error
    raise AssertionError("expected a DatetimeError")


class RangeConstants(unittest.TestCase):
    def test_design_bounds_are_derived_not_asserted(self) -> None:
        ref.check_range_constants()
        self.assertEqual(sum(ref.days_in_year(y) for y in range(ref.MIN_YEAR, ref.MAX_YEAR + 1)), 7_304_484)
        self.assertEqual(ref.date_to_string(ref.MIN_EPOCH_DAY), "-009999-01-01")
        self.assertEqual(ref.date_to_string(ref.MAX_EPOCH_DAY), "9999-12-31")

    def test_the_bounds_are_tight(self) -> None:
        self.assertEqual(ref.civil_from_days(ref.MIN_EPOCH_DAY - 1), (-10_000, 12, 31))
        self.assertEqual(ref.civil_from_days(ref.MAX_EPOCH_DAY + 1), (10_000, 1, 1))
        self.assertTrue(ref.in_instant_range(ref.INSTANT_MIN_SECOND))
        self.assertFalse(ref.in_instant_range(ref.INSTANT_MIN_SECOND - 1))
        self.assertTrue(ref.in_instant_range(ref.INSTANT_MAX_SECOND))
        self.assertFalse(ref.in_instant_range(ref.INSTANT_MAX_SECOND + 1))


class CivilConversion(unittest.TestCase):
    def sample(self) -> list[int]:
        rng = random.Random(7)
        days = list(range(ymd(1900, 1, 1), ymd(2100, 12, 31) + 1, 13))
        days += [rng.randint(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY) for _ in range(20_000)]
        days += list(range(ref.MIN_EPOCH_DAY, ref.MIN_EPOCH_DAY + 800)) + list(range(ref.MAX_EPOCH_DAY - 800, ref.MAX_EPOCH_DAY + 1))
        return days

    def test_hinnant_agrees_with_toordinal_and_a_counting_derivation(self) -> None:
        for n in self.sample():
            year, month, day = ref.civil_from_days(n)
            self.assertEqual(ref.days_from_civil(year, month, day), n)
            self.assertEqual(ref.days_from_civil_by_counting(year, month, day), n)
            if 1 <= year <= 9999:
                native = pydt.date(year, month, day)
                self.assertEqual(native.toordinal() - ref.UNIX_EPOCH_ORDINAL, n)
                self.assertEqual(native.isoweekday(), ref.weekday_of(n))
                self.assertEqual(tuple(native.isocalendar()[:2]), ref.iso_week(n))
                self.assertEqual(native.timetuple().tm_yday, ref.day_of_year(n))

    def test_year_length_summation_for_every_year(self) -> None:
        # Independent of Hinnant's era arithmetic: only the leap rule and addition.
        for year, epoch_day in ref.jan_first_by_year_lengths().items():
            self.assertEqual(ref.days_from_civil(year, 1, 1), epoch_day, year)
            self.assertEqual(ref.civil_from_days(epoch_day - 1), (year - 1, 12, 31), year)
        self.assertEqual(ref.jan_first_by_year_lengths()[ref.MIN_YEAR], ref.MIN_EPOCH_DAY)

    def test_successor_walk_across_year_zero(self) -> None:
        n = ymd(-2, 1, 1)
        expected = (-2, 1, 1)
        while expected != (2, 1, 1):
            self.assertEqual(ref.civil_from_days(n), expected)
            year, month, day = expected
            day += 1
            if day > ref.days_in_month(year, month):
                day, month = 1, month + 1
                if month > 12:
                    month, year = 1, year + 1
            expected, n = (year, month, day), n + 1

    def test_known_days(self) -> None:
        self.assertEqual(ymd(1970, 1, 1), 0)
        self.assertEqual(ref.weekday_of(0), 4)  # Thursday
        self.assertEqual(ref.weekday_of(ymd(2000, 1, 1)), 6)  # Saturday
        self.assertEqual(ymd(0, 3, 1) - ymd(0, 2, 28), 2)  # year 0 is a leap year
        self.assertEqual(ymd(-1, 3, 1) - ymd(-1, 2, 28), 1)

    def test_iso_week_year_boundaries(self) -> None:
        self.assertEqual(ref.iso_week(ymd(2004, 12, 31)), (2004, 53))
        self.assertEqual(ref.iso_week(ymd(2005, 1, 2)), (2004, 53))
        self.assertEqual(ref.iso_week(ymd(2005, 1, 3)), (2005, 1))
        self.assertEqual(ref.iso_week(ymd(2008, 12, 29)), (2009, 1))
        # -9999-01-01 is a Monday and 9999-12-31 a Friday, so no day of the
        # range has ISO week-year -10000 or 10000.
        self.assertEqual(ref.iso_week(ref.MIN_EPOCH_DAY), (-9_999, 1))
        self.assertEqual(ref.iso_week(ref.MAX_EPOCH_DAY), (9_999, 52))

    def test_iso_week_inverse_and_its_rejections(self) -> None:
        for n in self.sample()[:5000]:
            iso_year, week = ref.iso_week(n)
            self.assertEqual(ref.date_from_iso_week(iso_year, week, ref.weekday_of(n)), n)
        self.assertEqual(ref.iso_weeks_in_year(2020), 53)
        self.assertEqual(ref.iso_weeks_in_year(2021), 52)
        for args in ((2021, 53, 1), (2021, 0, 1), (10_001, 1, 1), (I64_MIN, 1, 1)):
            self.assertEqual(failure(lambda: ref.date_from_iso_week(*args)).kind, "domain")
        # A week that exists but runs past the range fails `domain`, not `overflow`.
        self.assertEqual(ref.date_from_iso_week(9_999, 52, 5), ref.MAX_EPOCH_DAY)
        self.assertEqual(failure(lambda: ref.date_from_iso_week(9_999, 52, 6)).kind, "domain")
        self.assertEqual(ref.date_from_iso_week(-9_999, 1, 1), ref.MIN_EPOCH_DAY)
        self.assertEqual(failure(lambda: ref.date_from_iso_week(-10_000, 52, 7)).kind, "domain")


class CalendarQueries(unittest.TestCase):
    def test_leap_rule_including_negative_years(self) -> None:
        for year, leap in ((2000, True), (1900, False), (2024, True), (2023, False), (0, True), (-4, True),
                           (-100, False), (-400, True), (-1, False), (I64_MIN, True), (I64_MAX, False)):
            self.assertEqual(ref.is_leap_year(year), leap, year)
            self.assertEqual(ref.days_in_year(year), 366 if leap else 365)

    def test_days_in_month_and_rejections(self) -> None:
        self.assertEqual([ref.days_in_month(2024, m) for m in range(1, 13)], [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        self.assertEqual(ref.days_in_month(2023, 2), 28)
        self.assertEqual(ref.days_in_month(I64_MAX, 2), 28)
        for month in (0, 13, I64_MIN):
            error = failure(lambda: ref.days_in_month(2024, month))
            self.assertEqual((error.function, error.kind), ("days_in_month", "domain"))

    def test_weekday_numbers(self) -> None:
        self.assertEqual([ref.weekday_name(w) for w in range(1, 8)][0], "monday")
        self.assertEqual(ref.weekday_from_iso_number(7), 7)
        for number in (0, 8, I64_MIN):
            self.assertEqual(failure(lambda: ref.weekday_from_iso_number(number)).kind, "domain")


class DateConstruction(unittest.TestCase):
    def test_design_example_message(self) -> None:
        self.assertEqual(failure(lambda: ref.date(2024, 2, 30)).message, "date: domain: day 30 is outside 1..29 for 2024-02")

    def test_fields_and_range(self) -> None:
        self.assertEqual(ref.date(ref.MIN_YEAR, 1, 1), ref.MIN_EPOCH_DAY)
        for args in ((2023, 2, 29), (10_000, 1, 1), (-10_000, 12, 31), (2024, 13, 1), (2024, 1, 0), (I64_MAX, 1, 1)):
            self.assertEqual(failure(lambda: ref.date(*args)).kind, "domain")
        self.assertEqual(ref.date_from_epoch_day(ref.MAX_EPOCH_DAY), ref.MAX_EPOCH_DAY)
        self.assertEqual(failure(lambda: ref.date_from_epoch_day(ref.MAX_EPOCH_DAY + 1)).kind, "domain")


class DateArithmetic(unittest.TestCase):
    def test_add_days_checks_range_for_every_i64_step(self) -> None:
        self.assertEqual(ref.date_add_days(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY - ref.MIN_EPOCH_DAY), ref.MAX_EPOCH_DAY)
        for start, step in ((ref.MAX_EPOCH_DAY, 1), (ref.MIN_EPOCH_DAY, -1), (0, I64_MAX), (0, I64_MIN)):
            error = failure(lambda: ref.date_add_days(start, step))
            self.assertEqual((error.function, error.kind), ("date_add_days", "overflow"))

    def test_month_arithmetic_under_both_policies(self) -> None:
        jan31 = ymd(2024, 1, 31)
        self.assertEqual(ref.date_add_months(jan31, 1, "ClampToMonthEnd"), ymd(2024, 2, 29))
        self.assertEqual(ref.date_add_months(ymd(2023, 1, 31), 1, "ClampToMonthEnd"), ymd(2023, 2, 28))
        self.assertEqual(ref.date_add_months(jan31, 2, "RejectInvalidDay"), ymd(2024, 3, 31))
        self.assertEqual(ref.date_add_months(jan31, -13, "RejectInvalidDay"), ymd(2022, 12, 31))
        rejected = failure(lambda: ref.date_add_months(jan31, 1, "RejectInvalidDay"))
        self.assertEqual((rejected.function, rejected.kind), ("date_add_months", "domain"))
        self.assertEqual(ref.date_add_months(ymd(2024, 2, 29), 12, "ClampToMonthEnd"), ymd(2025, 2, 28))
        # Year and month move on the total month count, across year zero.
        self.assertEqual(ref.date_add_months(ymd(0, 1, 15), -1, "RejectInvalidDay"), ymd(-1, 12, 15))

    def test_month_overflow_wins_over_the_day_policy(self) -> None:
        for start, months in ((ymd(9999, 12, 31), 1), (ymd(-9999, 1, 31), -1), (0, I64_MAX), (0, I64_MIN)):
            for policy in ref.DAY_OVERFLOW:
                self.assertEqual(failure(lambda: ref.date_add_months(start, months, policy)).kind, "overflow")
        self.assertEqual(ref.date_add_months(ymd(9999, 11, 30), 1, "RejectInvalidDay"), ref.MAX_EPOCH_DAY - 1)

    def test_add_period_is_months_then_days(self) -> None:
        self.assertEqual(ref.date_add_period(ymd(2024, 1, 31), (1, 1), "ClampToMonthEnd"), ymd(2024, 3, 1))
        error = failure(lambda: ref.date_add_period(ymd(2024, 1, 31), (1, 1), "RejectInvalidDay"))
        self.assertEqual((error.function, error.kind), ("date_add_period", "domain"))
        error = failure(lambda: ref.date_add_period(ref.MAX_EPOCH_DAY, (0, 1), "ClampToMonthEnd"))
        self.assertEqual((error.function, error.kind), ("date_add_period", "overflow"))

    def test_period_until_meets_its_defining_property(self) -> None:
        rng = random.Random(11)
        pairs = [(rng.randint(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY), rng.randint(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY)) for _ in range(3000)]
        pairs += [(a, a + rng.randint(-100, 100)) for a in (rng.randint(-1000, 1000) for _ in range(3000))]
        pairs += [(ymd(2024, 3, 31), ymd(2024, 2, 29)), (ymd(2024, 1, 31), ymd(2024, 2, 29)), (5, 5)]
        for a, b in pairs:
            months, days = ref.date_period_until(a, b)
            self.assertFalse(months > 0 and days < 0 or months < 0 and days > 0, (a, b))
            self.assertEqual(ref.date_add_period(a, (months, days), "ClampToMonthEnd"), b)
            # Maximality: one more month in the period's direction overshoots b.
            step = 1 if a <= b else -1
            try:
                beyond = ref.date_add_months(a, months + step, "ClampToMonthEnd")
            except DatetimeError:
                continue
            self.assertTrue(beyond > b if a <= b else beyond < b, (a, b))

    def test_mirrored_period_is_not_the_negation_of_the_forward_one(self) -> None:
        a, b = ymd(2024, 3, 31), ymd(2024, 2, 29)
        self.assertEqual(ref.date_period_until(a, b), (-1, 0))
        forward = ref.date_period_until(b, a)
        self.assertEqual(forward, (1, 2))
        # Negating the forward period breaks the round trip, which is why the
        # reference does not define the backward case that way.
        self.assertNotEqual(ref.date_add_period(a, (-forward[0], -forward[1]), "ClampToMonthEnd"), b)


class HolidayHelpers(unittest.TestCase):
    def test_easter_tables(self) -> None:
        for year, (western, orthodox) in EASTER_2000_2030.items():
            self.assertEqual(ref.easter_sunday_gregorian(year), ymd(year, *western), year)
            self.assertEqual(ref.easter_sunday_orthodox(year), ymd(year, *orthodox), year)
            self.assertEqual(ref.weekday_of(ref.easter_sunday_orthodox(year)), 7)

    def test_two_algorithms_agree_over_every_year(self) -> None:
        for year in range(ref.MIN_YEAR, ref.MAX_YEAR + 1):
            self.assertEqual(ref.easter_gregorian_meeus(year), ref.easter_gregorian_epact(year), year)
            self.assertEqual(ref.easter_sunday_orthodox(year), ref.easter_orthodox_by_full_moon(year), year)

    def test_easter_rejections(self) -> None:
        for year in (-10_000, 10_000, I64_MIN, I64_MAX):
            self.assertEqual(failure(lambda: ref.easter_sunday_gregorian(year)).kind, "domain")
            self.assertEqual(failure(lambda: ref.easter_sunday_orthodox(year)).kind, "domain")

    def test_nth_and_last_weekday(self) -> None:
        self.assertEqual(ref.nth_weekday_in_month(2024, 11, 4, 4), ymd(2024, 11, 28))  # US Thanksgiving
        self.assertIsNone(ref.nth_weekday_in_month(2024, 2, 1, 5))
        self.assertEqual(ref.nth_weekday_in_month(2024, 2, 4, 5), ymd(2024, 2, 29))
        self.assertIsNone(ref.nth_weekday_in_month(2024, 2, 4, I64_MAX))
        self.assertEqual(ref.last_weekday_in_month(2024, 5, 1), ymd(2024, 5, 27))  # US Memorial Day
        for args in ((2024, 11, 4, 0), (2024, 13, 4, 1), (10_000, 1, 4, 1)):
            self.assertEqual(failure(lambda: ref.nth_weekday_in_month(*args)).kind, "domain")

    def test_weekday_rolls_and_edges(self) -> None:
        self.assertEqual(ref.weekday_on_or_after(0, 4), 0)
        self.assertEqual(ref.weekday_on_or_after(0, 5), 1)
        self.assertEqual(ref.weekday_on_or_before(0, 3), -1)
        last_weekday = ref.weekday_of(ref.MAX_EPOCH_DAY)
        after = last_weekday % 7 + 1
        self.assertEqual(failure(lambda: ref.weekday_on_or_after(ref.MAX_EPOCH_DAY, after)).kind, "overflow")
        first_weekday = ref.weekday_of(ref.MIN_EPOCH_DAY)
        before = (first_weekday - 2) % 7 + 1
        self.assertEqual(failure(lambda: ref.weekday_on_or_before(ref.MIN_EPOCH_DAY, before)).kind, "overflow")


class Rounding(unittest.TestCase):
    # Values v = numerator/denominator: exact ties (2.5, -2.5, 3.5, -3.5, 0.5,
    # -0.5), inexact non-ties (2.7, -2.7) and an exact multiple (2).
    VALUES = ((5, 2), (-5, 2), (7, 2), (-7, 2), (1, 2), (-1, 2), (27, 10), (-27, 10), (4, 2))
    TABLE = {
        "RoundTowardNegative": (2, -3, 3, -4, 0, -1, 2, -3, 2),
        "RoundTowardPositive": (3, -2, 4, -3, 1, 0, 3, -2, 2),
        "RoundTowardZero": (2, -2, 3, -3, 0, 0, 2, -2, 2),
        "RoundAwayFromZero": (3, -3, 4, -4, 1, -1, 3, -3, 2),
        "RoundTiesToEven": (2, -2, 4, -4, 0, 0, 3, -3, 2),
        "RoundTiesToAway": (3, -3, 4, -4, 1, -1, 3, -3, 2),
    }

    def test_six_total_modes(self) -> None:
        self.assertEqual(tuple(self.TABLE), ref.ROUNDING_TOTAL)
        for mode, expected in self.TABLE.items():
            self.assertEqual(tuple(ref.round_quotient(n, d, mode, "f") for n, d in self.VALUES), expected, mode)

    def test_reject_inexact_accepts_only_multiples(self) -> None:
        self.assertEqual(ref.round_quotient(4, 2, "RejectInexact", "f"), 2)
        self.assertEqual(ref.round_quotient(-6, 3, "RejectInexact", "f"), -2)
        for n, d in self.VALUES[:-1]:
            error = failure(lambda: ref.round_quotient(n, d, "RejectInexact", "instant_round_to"))
            self.assertEqual((error.function, error.kind), ("instant_round_to", "domain"))

    def test_reject_inexact_through_the_api(self) -> None:
        self.assertEqual(ref.duration_to_count((3, 0), "Milliseconds", "RejectInexact"), 3000)
        error = failure(lambda: ref.duration_to_count((0, 1), "Microseconds", "RejectInexact"))
        self.assertEqual((error.function, error.kind), ("duration_to_count", "domain"))
        self.assertIsNone(ref.try_duration_to_count((0, 1), "Microseconds", "RejectInexact"))
        self.assertIsNone(ref.try_instant_to_unix_count((-1, 999_999_999), "Seconds", "RejectInexact"))
        self.assertEqual(ref.instant_round_to((7_200, 0), (3_600, 0), "RejectInexact"), (7_200, 0))
        self.assertEqual(failure(lambda: ref.instant_round_to((7_201, 0), (3_600, 0), "RejectInexact")).kind, "domain")


class DurationsAndPeriods(unittest.TestCase):
    def test_euclidean_normalization(self) -> None:
        self.assertEqual(ref.duration(-1, -500_000_000), (-2, 500_000_000))
        self.assertEqual(ref.duration(0, -1), (-1, 999_999_999))
        self.assertEqual(ref.duration(I64_MIN, 0), (I64_MIN, 0))
        # The constructor's arguments denote no duration: `domain` by §5's coverage rule.
        self.assertEqual(failure(lambda: ref.duration(I64_MAX, 1_000_000_000)).kind, "domain")
        self.assertEqual(failure(lambda: ref.duration(I64_MIN, -1)).kind, "domain")

    def test_negate_fails_only_at_the_minimum(self) -> None:
        self.assertEqual(ref.duration_negate((I64_MIN, 1)), (I64_MAX, 999_999_999))
        self.assertEqual(failure(lambda: ref.duration_negate((I64_MIN, 0))).kind, "overflow")

    def test_mul_fails_only_for_unrepresentable_results(self) -> None:
        self.assertEqual(ref.duration_mul((0, 1), I64_MAX), (9_223_372_036, 854_775_807))
        self.assertEqual(ref.duration_mul((-1, 999_999_999), I64_MIN), (9_223_372_036, 854_775_808))
        self.assertEqual(failure(lambda: ref.duration_mul((I64_MAX, 0), 2)).kind, "overflow")

    def test_counts_and_their_try_forms(self) -> None:
        self.assertEqual(ref.duration_to_count((-2, 500_000_000), "Seconds", "RoundTowardZero"), -1)
        self.assertEqual(ref.duration_to_count((-2, 500_000_000), "Seconds", "RoundTowardNegative"), -2)
        self.assertEqual(failure(lambda: ref.duration_to_count((I64_MAX, 0), "Nanoseconds", "RoundTowardZero")).kind, "overflow")
        self.assertIsNone(ref.try_duration_to_count((I64_MAX, 0), "Nanoseconds", "RoundTowardZero"))
        self.assertEqual(ref.try_duration_to_count((1, 0), "Milliseconds", "RoundTowardZero"), 1000)
        self.assertEqual(failure(lambda: ref.duration_from_count(I64_MAX, "Hours")).kind, "domain")
        # Arithmetic on durations is `overflow`.
        self.assertEqual(failure(lambda: ref.duration_add((I64_MAX, 0), (1, 0))).kind, "overflow")

    def test_seconds_f64_is_within_one_ulp_when_the_whole_part_is_below_2_53(self) -> None:
        rng = random.Random(3)
        cases = [(-1, 999_999_999), (-1, 999_999_000), (-1, 0), (-2, 1), (0, 1), (-(2**53) + 1, 1)]
        cases += [(rng.randint(-(2**53) + 1, 2**53 - 1), rng.randrange(ref.NANOS_PER_SECOND)) for _ in range(5000)]
        cases += [(-1, rng.randrange(1, ref.NANOS_PER_SECOND)) for _ in range(2000)]
        for d in cases:
            got = ref.duration_to_seconds_f64(d)
            exact = Fraction(d[0]) + Fraction(d[1], ref.NANOS_PER_SECOND)
            # One unit in the last place of a nonzero result is at most |exact| * 2^-52.
            self.assertLessEqual(abs(Fraction(got) - exact), abs(exact) * Fraction(1, 2**52), d)
        self.assertEqual(ref.duration_to_seconds_f64((-1, 999_999_999)), -1e-9)

    def test_the_euclidean_composition_would_cancel(self) -> None:
        # The rejected composition f64(second) + f64(nanosecond)/1e9 loses the
        # value of -1 ns to cancellation; the same-sign split keeps it exact.
        naive = float(-1) + float(999_999_999) / 1e9
        self.assertGreater(abs(Fraction(naive) - Fraction(-1, 10**9)), Fraction(1, 10**9) * Fraction(1, 2**52))
        self.assertEqual(ref.same_sign_parts((-1, 999_999_999)), (0, -1))
        self.assertEqual(ref.same_sign_parts((-3, 0)), (-3, 0))

    def test_period_signs_and_overflow(self) -> None:
        self.assertEqual(ref.period(0, -1), (0, -1))
        for args in ((1, -1), (-1, 1), (I64_MIN, 1)):
            self.assertEqual(failure(lambda: ref.period(*args)).kind, "domain")
        self.assertEqual(failure(lambda: ref.period_negate((I64_MIN, 0))).kind, "overflow")
        self.assertEqual(failure(lambda: ref.period_negate((0, I64_MIN))).kind, "overflow")
        self.assertEqual(ref.period_negate((I64_MAX, I64_MAX)), (-I64_MAX, -I64_MAX))
        self.assertEqual(failure(lambda: ref.period_mul((2, 0), I64_MAX)).kind, "overflow")


class InstantsAndOffsets(unittest.TestCase):
    def test_offset_bounds(self) -> None:
        self.assertEqual(ref.offset_from_seconds(-86_399), -86_399)
        for seconds in (86_400, -86_400, I64_MIN):
            self.assertEqual(failure(lambda: ref.offset_from_seconds(seconds)).kind, "domain")

    def test_every_instant_has_a_civil_reading_under_every_offset(self) -> None:
        for second in (ref.INSTANT_MIN_SECOND, ref.INSTANT_MAX_SECOND):
            for offset in (-86_399, 0, 86_399):
                epoch_day, _ = ref.instant_to_datetime_at((second, 999_999_999), offset)
                self.assertTrue(ref.in_day_range(epoch_day))

    def test_instant_construction(self) -> None:
        self.assertEqual(ref.instant_from_unix(ref.INSTANT_MAX_SECOND, 999_999_999), (ref.INSTANT_MAX_SECOND, 999_999_999))
        for args in ((0, 1_000_000_000), (0, -1), (ref.INSTANT_MAX_SECOND + 1, 0), (ref.INSTANT_MIN_SECOND - 1, 0)):
            self.assertEqual(failure(lambda: ref.instant_from_unix(*args)).kind, "domain")
        self.assertEqual(ref.instant_from_unix_count(-1, "Milliseconds"), (-1, 999_000_000))
        self.assertEqual(failure(lambda: ref.instant_from_unix_count(I64_MAX, "Hours")).kind, "domain")

    def test_nanosecond_counts_fit_only_near_1970(self) -> None:
        self.assertEqual(ref.instant_to_unix_count((9_223_372_036, 854_775_807), "Nanoseconds", "RoundTowardZero"), I64_MAX)
        self.assertEqual(failure(lambda: ref.instant_to_unix_count((9_223_372_036, 854_775_808), "Nanoseconds", "RoundTowardZero")).kind, "overflow")
        self.assertIsNone(ref.try_instant_to_unix_count((ref.INSTANT_MAX_SECOND, 0), "Nanoseconds", "RoundTowardZero"))
        self.assertEqual(ref.instant_to_unix_count((ref.INSTANT_MIN_SECOND, 0), "Microseconds", "RoundTowardZero"), ref.INSTANT_MIN_SECOND * 10**6)

    def test_round_to(self) -> None:
        self.assertEqual(ref.instant_round_to((-1, 500_000_000), (1, 0), "RoundTowardZero"), (0, 0))
        self.assertEqual(ref.instant_round_to((-1, 500_000_000), (1, 0), "RoundTowardNegative"), (-1, 0))
        self.assertEqual(ref.instant_round_to((-1, 500_000_000), (1, 0), "RoundTiesToEven"), (0, 0))
        self.assertEqual(ref.instant_round_to((-1, 500_000_000), (1, 0), "RoundTiesToAway"), (-1, 0))
        self.assertEqual(ref.instant_round_to((-1, 500_000_000), (1, 0), "RoundAwayFromZero"), (-1, 0))
        self.assertEqual(ref.instant_round_to((86_399, 0), (1, 500_000_000), "RoundTowardPositive"), (86_400, 0))
        for increment in ((7, 0), (0, 0), (-1, 0), (172_800, 0)):
            self.assertEqual(failure(lambda: ref.instant_round_to((0, 0), increment, "RoundTowardZero")).kind, "domain")
        self.assertEqual(failure(lambda: ref.instant_round_to((ref.INSTANT_MAX_SECOND, 1), (86_400, 0), "RoundTowardPositive")).kind, "overflow")

    def test_datetime_to_instant_edges(self) -> None:
        self.assertEqual(ref.datetime_to_instant_at((ref.MIN_EPOCH_DAY, 0), -86_399), (ref.INSTANT_MIN_SECOND, 0))
        self.assertEqual(failure(lambda: ref.datetime_to_instant_at((ref.MIN_EPOCH_DAY, 0), 0)).kind, "overflow")


class TextProfile(unittest.TestCase):
    def test_canonical_formatting(self) -> None:
        self.assertEqual(ref.year_text(0), "0000")
        self.assertEqual(ref.date_to_string(ymd(-1, 12, 31)), "-000001-12-31")
        self.assertEqual(ref.time_to_string(ref.time(1, 2, 3, 450_000_000)), "01:02:03.45")
        self.assertEqual(ref.time_to_string(ref.time(1, 2, 3, 0)), "01:02:03")
        self.assertEqual(ref.offset_to_string(0), "Z")
        self.assertEqual(ref.offset_to_string(-1), "-00:00:01")
        self.assertEqual(ref.offset_to_string(19_800), "+05:30")
        self.assertEqual(ref.duration_to_string((3_661, 0)), "PT3661S")
        self.assertEqual(ref.duration_to_string((-1, 500_000_000)), "-PT0.5S")
        self.assertEqual(ref.duration_to_string((0, 0)), "PT0S")
        self.assertEqual(ref.duration_to_string((I64_MIN, 0)), "-PT9223372036854775808S")
        self.assertEqual(ref.period_to_string((14, 3)), "P14M3D")
        self.assertEqual(ref.period_to_string((0, 0)), "P0D")
        self.assertEqual(ref.period_to_string((-14, -3)), "-P14M3D")
        self.assertEqual(ref.offset_datetime_to_string(((1_727_789_400, 0), -14_400)), "2024-10-01T09:30:00-04:00")

    def test_valid_corpus_round_trips(self) -> None:
        for kind, texts in harness.VALID_TEXT.items():
            parse, render = ref.PARSERS[kind]
            for text in texts:
                value = parse(text)
                canonical = render(value)
                self.assertEqual(parse(canonical), value, (kind, text))
                self.assertEqual(render(parse(canonical)), canonical, (kind, text))

    def test_invalid_corpus_is_rejected_with_domain(self) -> None:
        for kind, texts in harness.INVALID_TEXT.items():
            parse, _ = ref.PARSERS[kind]
            for text in texts:
                error = failure(lambda: parse(text))
                self.assertEqual((error.function, error.kind), (f"parse_{kind}", "domain"), (kind, text))
                self.assertRegex(error.message, MESSAGE)

    def test_specific_profile_decisions(self) -> None:
        self.assertEqual(ref.parse_date("+000000-01-01"), ymd(0, 1, 1))
        self.assertEqual(ref.parse_offset("-00:00"), 0)
        self.assertEqual(ref.parse_offset_datetime("2026-10-01T13:30:00Z")[1], 0)
        self.assertEqual(ref.parse_period("P1Y2W"), (12, 14))
        self.assertEqual(ref.parse_period("-P768614336404564650Y8M"), (I64_MIN, 0))
        self.assertEqual(ref.parse_duration("-PT2562047788015215H30M8S"), (I64_MIN, 0))
        self.assertEqual(ref.parse_instant("2026-10-01T09:30:00-04:00"), (1_790_861_400, 0))
        self.assertEqual(ref.canonical("time", "12:00:00.500"), "12:00:00.5")


class Columns(unittest.TestCase):
    """§10: a column callable is its scalar twin at every index."""

    def test_fields_and_construction_follow_the_scalar_twins(self) -> None:
        days = [ref.MIN_EPOCH_DAY, -1, 19_782, ref.MAX_EPOCH_DAY]
        self.assertEqual(ref.dates_year(days), [-9999, 1969, 2024, 9999])
        self.assertEqual(ref.dates_weekday_iso_number(days), [1, 3, 4, 5])
        self.assertEqual(ref.dates_day_of_year(days), [1, 365, 60, 365])
        self.assertEqual(ref.dates_from_ymd([2024, 1969], [2, 12], [29, 31]), [19_782, -1])
        self.assertEqual(ref.dates_to_strings([-735_525]), ["-000044-03-15"])

    def test_the_lowest_failing_index_names_the_failure(self) -> None:
        error = failure(lambda: ref.dates_from_ymd([2024, 2023, 2024], [1, 2, 13], [1, 29, 1]))
        self.assertEqual((error.function, error.kind), ("dates_from_ymd", "domain"))
        self.assertTrue(error.detail.startswith("element 1: "), error.detail)
        error = failure(lambda: ref.dates_add_months([ref.date(9999, 12, 1), ref.date(2024, 1, 31)], [1, 1], "RejectInvalidDay"))
        self.assertEqual((error.kind, error.detail[:11]), ("overflow", "element 0: "))
        error = failure(lambda: ref.dates_add_months([ref.date(2024, 1, 31), ref.date(9999, 12, 1)], [1, 1], "RejectInvalidDay"))
        self.assertEqual((error.kind, error.detail[:11]), ("domain", "element 0: "))
        self.assertRegex(error.message, MESSAGE)

    def test_masked_forms_never_fail_on_elements(self) -> None:
        self.assertEqual(ref.try_dates_from_ymd([2024, 2023], [2, 2], [29, 29]), ([19_782, 0], [True, False]))
        self.assertEqual(ref.try_parse_dates(["2024-02-29", "2024-02-30"]), ([19_782, 0], [True, False]))
        self.assertEqual(ref.try_durations([I64_MAX, -1], [NANO, -500_000_000]), ([(0, 0), (-2, 500_000_000)], [False, True]))
        with self.assertRaises(DatetimeError):
            ref.durations([I64_MAX], [NANO])

    def test_columns_of_different_lengths_fail_domain_first(self) -> None:
        error = failure(lambda: ref.dates_add_days([0, 0], [1, 2, I64_MAX]))
        self.assertEqual((error.function, error.kind, error.detail), ("dates_add_days", "domain", "arguments have 2 and 3 elements"))
        self.assertEqual(ref.dates_add_days([0, 0], [1, 2]), [1, 2])

    def test_rounding_checks_the_increment_before_any_element(self) -> None:
        error = failure(lambda: ref.instants_round_to([], (7, 0), "RoundTiesToEven"))
        self.assertEqual((error.function, error.kind), ("instants_round_to", "domain"))
        self.assertEqual(ref.instants_round_to([], (900, 0), "RejectInexact"), [])
        error = failure(lambda: ref.instants_round_to([(0, 0), (1, 0)], (2, 0), "RejectInexact"))
        self.assertTrue(error.detail.startswith("element 1: "), error.detail)

    def test_counts_round_or_reject_per_element(self) -> None:
        column = [(-2, 500_000_000), (2, 500_000_000)]
        self.assertEqual(ref.instants_to_unix_count(column, "Seconds", "RoundTiesToEven"), [-2, 2])
        self.assertEqual(ref.instants_to_unix_count(column, "Seconds", "RoundTiesToAway"), [-2, 3])
        error = failure(lambda: ref.instants_to_unix_count([(0, 0), (ref.INSTANT_MAX_SECOND, 0)], "Nanoseconds", "RejectInexact"))
        self.assertEqual((error.kind, error.detail[:11]), ("overflow", "element 1: "))
        self.assertEqual(ref.instants_from_unix_count([-1_500], "Milliseconds"), [(-2, 500_000_000)])
        self.assertRaises(DatetimeError, ref.instants_from_unix_count, [I64_MAX], "Hours")

    def test_the_time_axis_is_the_scalar_composition(self) -> None:
        origin = (0, 1)
        axis = ref.instants_seconds_since_f64([(0, 0), (1_700_000_000, 500_000_000)], origin)
        self.assertEqual(axis, [-1e-9, 1_700_000_000.5])
        self.assertEqual(axis[0], ref.duration_to_seconds_f64(ref.instant_until(origin, (0, 0))))
        self.assertNotEqual(axis[0], 0.0)

    def test_order_and_local_dates(self) -> None:
        a, b = [(7, 1), (7, 0)], [(7, 0), (7, 0)]
        self.assertEqual(ref.instants_order("instants_lt", a, b), [False, False])
        self.assertEqual(ref.instants_order("instants_lte", a, b), [False, True])
        self.assertEqual(ref.instants_to_dates_at([(-1, 0)], 0), [-1])
        self.assertEqual(ref.instants_to_dates_at([(-1, 0)], 1), [0])


class FailureGrammar(unittest.TestCase):
    def test_every_reference_failure_in_the_corpus_follows_section_5(self) -> None:
        corpus = harness.build_ci_corpus()
        self.assertGreater(len(corpus.failures), 100)
        for failure_case in corpus.failures:
            self.assertRegex(f"{failure_case.function}: {failure_case.kind}: x", MESSAGE)
            self.assertIn(failure_case.function, harness.FUNCTIONS + harness.COLUMN_FUNCTIONS, failure_case)

    def test_kinds_are_closed(self) -> None:
        with self.assertRaises(ValueError):
            DatetimeError("date", "trap", "x")


class HarnessLogic(unittest.TestCase):
    def test_literals_cover_i64(self) -> None:
        self.assertEqual(harness.lit(I64_MIN), "(-9223372036854775807i64 - 1i64)")
        self.assertEqual(harness.lit(-5), "(-5i64)")
        with self.assertRaises(ValueError):
            harness.lit(I64_MAX + 1)

    def test_render_matches_lane_printing(self) -> None:
        self.assertEqual(harness.render([(True, (1, -2))]), "[(true, (1, -2))]")
        self.assertEqual(harness.render(["2024-01-01"]), "[2024-01-01]")

    def test_value_comparison_accepts_only_exact_output(self) -> None:
        value = harness.Value("v", "x", [1, 2, 3])
        self.assertIsNone(harness.compare_value(value, "[[1, 2, 3]]"))
        self.assertIn("element 1", harness.compare_value(value, "[[1, 9, 3]]") or "")
        self.assertIsNotNone(harness.compare_value(value, "[[1, 2]]"))
        self.assertEqual(harness.compare_value(value, None), "no output line")
        lazy = harness.Value("b", "x", lambda: [4])
        self.assertIsNone(harness.compare_value(lazy, "[[4]]"))

    def test_float_comparison_is_bitwise(self) -> None:
        value = harness.Value("v", "x", 0.1, float_bits=True)
        self.assertIsNone(harness.compare_value(value, "[0.1]"))
        self.assertIsNotNone(harness.compare_value(value, "[0.10000000000000002]"))
        self.assertIsNotNone(harness.compare_value(harness.Value("v", "x", 0.0, float_bits=True), "[-0.0]"))

    def test_failure_message_extraction(self) -> None:
        result = harness.LaneResult("eval", 1, "", "error: date: domain: day 30 is outside 1..29 for 2024-02\n", "eval")
        self.assertEqual(harness.failure_message(result), "date: domain: day 30 is outside 1..29 for 2024-02")
        trap = harness.LaneResult("c", 1, "", "integer overflow in add\n", "run")
        self.assertIsNone(harness.failure_message(trap))

    def test_failure_checks_reject_the_wrong_function_kind_or_lane_split(self) -> None:
        program = harness.Program("f", "", failure=harness.Failure("f0000_x", "e", "date", "domain"))

        def check(eval_err: str, c_err: str) -> list[str]:
            report = harness.Report()
            harness.check_program(program, [harness.LaneResult("eval", 1, "", eval_err, "eval"),
                                            harness.LaneResult("c", 1, "", c_err, "run")], report)
            return report.problems

        self.assertEqual(check("error: date: domain: d", "date: domain: d"), [])
        self.assertTrue(check("error: date: overflow: d", "date: overflow: d"))
        self.assertTrue(check("error: try_date: domain: d", "try_date: domain: d"))
        self.assertTrue(check("error: date: domain: d", "date: domain: e"))
        self.assertTrue(check("error: date: domain: ", "date: domain: "))
        report = harness.Report()
        harness.check_program(program, [harness.LaneResult("eval", 0, "f0000_x = [1]", "", "eval")], report)
        self.assertTrue(report.problems)

    def test_representative_failures_keep_named_cases_and_sample_the_rest(self) -> None:
        failures = [harness.Failure(f"f{i:04d}_add_days", "e", "date_add_days", "overflow") for i in range(10)]
        failures += [harness.Failure(f"f{i + 10:04d}_named_reject", "e", "parse_date", "domain") for i in range(5)]
        chosen = harness.representative_failures(failures, 2)
        self.assertEqual([f.name for f in chosen if "add_days" in f.name], ["f0000_add_days", "f0009_add_days"])
        self.assertEqual(len([f for f in chosen if "named" in f.name]), 5)

    def test_stray_output_lines_are_disagreements(self) -> None:
        printed, stray = harness.parse_bindings("v = [1]\nfirst_year = -9999\nnot a binding\n")
        self.assertEqual(printed, {"v": "[1]", "first_year": "-9999"})
        self.assertEqual(stray, ["not a binding"])
        program = harness.Program("p", "", (harness.Value("v", "x", 1),))
        report = harness.Report()
        harness.check_program(program, [harness.LaneResult("c", 0, "v = [1]\nfirst_year = -9999\n", "", "run")], report)
        self.assertEqual(len(report.problems), 1)
        self.assertIn("first_year", report.problems[0])
        clean = harness.Report()
        harness.check_program(program, [harness.LaneResult("c", 0, "v = [1]\n", "", "run")], clean)
        self.assertEqual(clean.problems, [])

    def test_generators_mirror_the_prelude_and_stay_in_range(self) -> None:
        prelude = harness.prelude()
        self.assertIn("mod(x * 1103515245i64 + 12345i64, 2147483648i64)", prelude)
        self.assertIn(f"lcg(lcg(k + {harness.SEED}i64))", prelude)
        self.assertNotIn("@", prelude)
        for k in range(0, 500_000, 997):
            self.assertTrue(ref.in_day_range(harness.sample_day(k)))
            self.assertTrue(ref.MIN_YEAR < harness.sample_year(k) < ref.MAX_YEAR)
            near = harness.near_day(k)
            self.assertTrue(ref.in_day_range(near - 800) and ref.in_day_range(near + 800))
            # The products inside lcg stay below 2^63 for every reachable input.
            self.assertLess((harness.lcg(k + harness.SEED)) * 1103515245 + 12345, 2**63)

    def test_every_grid_is_trap_free_under_the_reference(self) -> None:
        # A grid binding has no way to report a failure, so the reference must
        # compute every row of every grid without raising.
        corpus = harness.build_ci_corpus()
        for grid in corpus.bulk:
            if grid.lanes == harness.C_ONLY:
                continue  # the 1900-2100 sweep is checked by the C-only test below
            self.assertIsInstance(grid.resolve(), list, grid.name)

    def test_day_coverage_counts_a_union(self) -> None:
        coverage = harness.DayCoverage()
        coverage.add_range(0, 10)
        coverage.add_range(5, 15)
        coverage.update([3, 20, 21])
        self.assertEqual(len(coverage), 17)
        self.assertIn(20, coverage)
        self.assertNotIn(16, coverage)

    def test_lanes_and_day_coverage(self) -> None:
        corpus = harness.build_ci_corpus()
        lanes = {grid.name.split("_", 1)[1]: grid.lanes for grid in corpus.bulk}
        self.assertEqual(lanes["days_1900_2100"], harness.C_ONLY)
        self.assertEqual(lanes["days_first_400"], harness.BOTH)
        self.assertTrue(corpus.days.covers(harness.FIRST_1900, harness.LAST_2100 + 1))
        self.assertFalse(corpus.days.covers(harness.FIRST_1900 - 1, harness.LAST_2100 + 1))
        self.assertIn(ref.MIN_EPOCH_DAY, corpus.days)
        self.assertIn(ref.MAX_EPOCH_DAY, corpus.days)
        exhaustive = harness.build_exhaustive_corpus()
        self.assertEqual(len(exhaustive.days), ref.MAX_EPOCH_DAY - ref.MIN_EPOCH_DAY + 1)
        self.assertTrue(any(grid.lanes == harness.EVAL_ONLY for grid in exhaustive.bulk))

    def test_grid_source_nests_one_loop_per_axis(self) -> None:
        corpus = harness.Corpus()
        corpus.grid("g", [harness.range_axis("a", 0, 2), harness.int_axis("b", [5, 6])], "[a + b]", lambda a, b: [a + b])
        grid = corpus.bulk[0]
        self.assertEqual(grid.expr, "flat_map(fn (a: i64) -> flat_map(fn (b: i64) -> [a + b], [5i64, 6i64]), range(0i64, 2i64))")
        self.assertEqual(grid.resolve(), [5, 6, 6, 7])
        corpus.grid("s", [harness.text_axis("t", ["x", 'q"'])], "f(t)", lambda t: t.upper(), strings=True)
        self.assertEqual(corpus.bulk[1].expr, 'map(fn (t: string) -> f(t), ["x", "q\\""])')
        self.assertEqual(corpus.bulk[1].resolve(), ["X", 'Q"'])

    def test_ci_corpus_covers_the_brief(self) -> None:
        corpus = harness.build_ci_corpus()
        names = " ".join(grid.name for grid in corpus.bulk)
        for required in ("days_1900_2100", "days_first_400", "days_last_400", "days_sampled", "every_year", "period_until",
                         "add_months", "add_period", "nth_weekday", "last_weekday", "round_to", "try_to_count",
                         "try_duration_to_count", "text_parse_", "text_reject_", "text_canonical_", "text_reparse_"):
            self.assertIn(required, names)
        exprs = harness.prelude() + " ".join(v.expr for v in corpus.bulk + corpus.values) + " ".join(f.expr for f in corpus.failures)
        for constructor in ref.DAY_OVERFLOW + ref.ROUNDING + tuple(ref.TIME_UNIT_NANOS) + ref.WEEKDAYS:
            self.assertIn(constructor, exprs)
        missing = [function for function in harness.FUNCTIONS if not re.search(rf"\b{function}\(", exprs)]
        self.assertEqual(missing, [])
        column_exprs = harness.prelude() + " ".join(v.expr for v in corpus.columns) + " ".join(f.expr for f in corpus.failures)
        missing = [function for function in harness.COLUMN_FUNCTIONS if not re.search(rf"\b{function}\(", column_exprs)]
        self.assertEqual(missing, [])
        column_kinds = {(f.function, f.kind) for f in corpus.failures if f.name.startswith("cf")}
        for kind in (("dates_add_months", "domain"), ("dates_add_months", "overflow"), ("instants_to_unix_count", "domain"),
                     ("instants_to_unix_count", "overflow"), ("instants_round_to", "domain"), ("instants_round_to", "overflow"),
                     ("dates_days_until", "domain"), ("instants_until", "domain")):
            self.assertIn(kind, column_kinds)
        kinds = {(f.function, f.kind) for f in corpus.failures}
        self.assertIn(("try_date_add_months", "overflow"), kinds)
        self.assertIn(("date_add_months", "domain"), kinds)
        self.assertIn(("parse_time", "domain"), kinds)

if __name__ == "__main__":
    unittest.main()
