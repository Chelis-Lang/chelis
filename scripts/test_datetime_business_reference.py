"""Tests for the Std.Datetime.Business reference (chelis#2860).

The reference is checked three ways: against hand-derived facts, positive and
negative; against Python's `datetime` for the civil calendar it walks; and
against NumPy's `is_busday`, `busday_offset` and `busday_count` over the seeded
corpus the differential harness runs on both lanes, every roll and start
included.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import random
import sys
import unittest

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datetime_business_differential as differential  # noqa: E402
import datetime_business_reference as ref  # noqa: E402

JAN_1_2026 = 20_454  # a Thursday


def year_2026() -> ref.Calendar:
    calendar = ref.make_calendar(ref.WEEKDAYS_ONLY, [JAN_1_2026, JAN_1_2026 + 18, JAN_1_2026 + 3], JAN_1_2026, JAN_1_2026 + 364)
    assert calendar is not None
    return calendar


class CivilCalendarTests(unittest.TestCase):
    def test_civil_from_days_matches_datetime(self) -> None:
        rng = random.Random(2860)
        days = [0, -1, -719_162, ref.MAX_EPOCH_DAY] + [rng.randint(-719_162, ref.MAX_EPOCH_DAY) for _ in range(2000)]
        for day in days:
            civil = datetime.date.fromordinal(day + 719_163)
            self.assertEqual(ref.civil_from_days(day), (civil.year, civil.month, civil.day), day)

    def test_civil_from_days_reaches_the_range_start(self) -> None:
        self.assertEqual(ref.civil_from_days(ref.MIN_EPOCH_DAY), (-9999, 1, 1))
        self.assertEqual(ref.civil_from_days(-719_529), (-1, 12, 31))
        self.assertNotEqual(ref.civil_from_days(-719_528), (-1, 12, 31))


class ConstructionTests(unittest.TestCase):
    def test_normalization_drops_repeats_and_off_weekmask_holidays(self) -> None:
        calendar = year_2026()
        self.assertEqual(calendar.holidays, (JAN_1_2026, JAN_1_2026 + 18))

    def test_invalid_inputs_have_no_calendar(self) -> None:
        self.assertIsNone(ref.make_calendar((False,) * 7, [], 0, 1))
        self.assertIsNone(ref.make_calendar(ref.WEEKDAYS_ONLY, [], 1, 0))
        self.assertIsNone(ref.make_calendar(ref.WEEKDAYS_ONLY, [2], 0, 1))


class QueryTests(unittest.TestCase):
    def test_rolls(self) -> None:
        calendar = year_2026()
        saturday = JAN_1_2026 + 2
        self.assertEqual(ref.roll(calendar, saturday, "Following"), saturday + 2)
        self.assertEqual(ref.roll(calendar, saturday, "Preceding"), saturday - 1)
        may_30 = JAN_1_2026 + 149
        self.assertEqual(ref.roll(calendar, may_30, "ModifiedFollowing"), may_30 - 1)

    def test_rolls_without_an_answer_inside_the_horizon(self) -> None:
        calendar = year_2026()
        self.assertIsNone(ref.roll(calendar, JAN_1_2026, "Preceding"))
        self.assertIsNone(ref.roll(calendar, JAN_1_2026 - 1, "Unadjusted"))
        mid_month = ref.make_calendar(ref.WEEKDAYS_ONLY, [], 20_574, 20_596)
        month_end = ref.make_calendar(ref.WEEKDAYS_ONLY, [], 20_574, 20_604)
        self.assertIsNone(ref.roll(mid_month, 20_596, "ModifiedFollowing"))
        self.assertEqual(ref.roll(month_end, 20_603, "ModifiedFollowing"), 20_602)

    def test_offsets_roll_first(self) -> None:
        calendar = year_2026()
        saturday = JAN_1_2026 + 2
        self.assertEqual(ref.offset(calendar, saturday, 2, "RollStartForward"), saturday + 4)
        self.assertEqual(ref.offset(calendar, saturday, 2, "RollStartBackward"), saturday + 3)
        self.assertEqual(ref.offset(calendar, saturday, 0, "RollStartForward"), saturday + 2)

    def test_offsets_without_an_answer(self) -> None:
        calendar = year_2026()
        self.assertIsNone(ref.offset(calendar, JAN_1_2026 + 2, 1, "RejectNonBusinessStart"))
        self.assertIsNone(ref.offset(calendar, JAN_1_2026 + 364, 1, "RejectNonBusinessStart"))
        self.assertIsNone(ref.offset(calendar, JAN_1_2026 + 1, ref.I64_MAX, "RejectNonBusinessStart"))

    def test_counts_are_half_open_and_antisymmetric(self) -> None:
        calendar = year_2026()
        self.assertEqual(ref.count(calendar, JAN_1_2026, JAN_1_2026 + 31), 20)
        self.assertEqual(ref.count(calendar, JAN_1_2026 + 31, JAN_1_2026), -20)
        self.assertEqual(ref.count(calendar, JAN_1_2026, JAN_1_2026 + 365), 259)

    def test_counts_outside_the_horizon(self) -> None:
        calendar = year_2026()
        self.assertIsNone(ref.count(calendar, JAN_1_2026, JAN_1_2026 + 366))
        self.assertIsNone(ref.count(calendar, JAN_1_2026 - 1, JAN_1_2026))

    def test_combinations(self) -> None:
        calendar = year_2026()
        weekend = ref.make_calendar((False,) * 5 + (True, True), [], JAN_1_2026, JAN_1_2026 + 364)
        either = ref.combine(calendar, weekend, both=False)
        self.assertEqual(either.weekmask, (True,) * 7)
        self.assertEqual(either.holidays, (JAN_1_2026, JAN_1_2026 + 18))
        self.assertIsNone(ref.combine(calendar, weekend, both=True))
        later = ref.make_calendar(ref.WEEKDAYS_ONLY, [], JAN_1_2026 + 365, JAN_1_2026 + 400)
        self.assertIsNone(ref.combine(calendar, later, both=False))


class NumpyDifferentialTests(unittest.TestCase):
    """The reference agrees with NumPy wherever it defines an answer."""

    def test_reference_agrees_with_numpy_over_the_corpus(self) -> None:
        specs = ref.build_calendars(differential.SEED, 24)
        for index, spec in enumerate(specs):
            calendar = spec.calendar()
            queries = ref.calendar_queries(differential.SEED + 11 * index, calendar)
            with self.subTest(calendar=index):
                self.assertEqual(ref.numpy_disagreements(numpy, calendar, queries.days, queries.offsets, queries.pairs), [])

    def test_numpy_comparison_catches_a_wrong_reference(self) -> None:
        calendar = year_2026()
        queries = ref.calendar_queries(differential.SEED, calendar)
        days, offsets, pairs = list(queries.days) + [JAN_1_2026 + 17, JAN_1_2026 + 18], queries.offsets, queries.pairs
        self.assertEqual(ref.numpy_disagreements(numpy, calendar, days, offsets, pairs), [])
        forgetful = ref.make_calendar(ref.WEEKDAYS_ONLY, [JAN_1_2026], JAN_1_2026, JAN_1_2026 + 364)
        self.assertNotEqual(
            ref.numpy_disagreements(numpy, forgetful, days, offsets, pairs, numpy_holidays=calendar.holidays), [])

    def test_numpy_reversed_count_is_not_antisymmetric(self) -> None:
        """Why reversed counts are compared through antisymmetry."""
        monday, saturday = numpy.datetime64("2026-01-05"), numpy.datetime64("2026-01-03")
        self.assertEqual(int(numpy.busday_count(monday, saturday)), -1)
        self.assertEqual(int(numpy.busday_count(saturday, monday)), 0)


class HarnessTests(unittest.TestCase):
    def test_list_expressions_chunk_long_literals(self) -> None:
        self.assertEqual(differential.list_lit([]), "take([0i64], 0i64)")
        self.assertEqual(differential.list_lit([1, -2]), "[1i64, (-2i64)]")
        self.assertIn("concat(", differential.list_lit(list(range(100))))
        self.assertEqual(differential.lit(ref.I64_MIN), "sub(-9223372036854775807i64, 1i64)")

    def test_failure_messages_parse_only_domain_failures(self) -> None:
        result = differential.LaneResult("eval", 1, "", "error: business_day_roll: domain: x", "eval")
        self.assertEqual(differential.failure_message(result), "business_day_roll: domain: x")
        trap = differential.LaneResult("eval", 1, "", "error: Overflow: add", "eval")
        self.assertIsNone(differential.failure_message(trap))

    def test_expected_rows_cover_every_query(self) -> None:
        spec = ref.edge_calendars()[4]
        program = differential.calendar_program(0, spec, "c")
        queries = ref.calendar_queries(differential.SEED, spec.calendar(), differential.C_DAYS, differential.C_PAIRS)
        rows = program.expected["observed_000"].strip("[]").split(", ")
        self.assertEqual(len(rows), len(queries.days) * (1 + len(ref.ROLLS) + 3 * len(queries.offsets)) + len(queries.pairs))

    def test_drawn_holidays_follow_the_generator(self) -> None:
        spec = ref.CalendarSpec(ref.WEEKDAYS_ONLY, 100, 199, density=500, seed=7)
        drawn = spec.holiday_args()
        picked = [i for i in range(100) if ref.mix(i, 7) % 1000 < 500]
        self.assertEqual(drawn, [100 + i for i in picked + picked[:3]])
        self.assertEqual(ref.lcg(0), 12_345)
        self.assertEqual(ref.lcg(1), (1_103_515_245 + 12_345) % 2**31)


if __name__ == "__main__":
    unittest.main()
