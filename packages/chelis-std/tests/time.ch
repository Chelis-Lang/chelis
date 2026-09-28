module Std.Tests.Time
import Std.Test (assert_eq, assert_true, fail)
import Std.Time (date, try_date, duration, is_leap_year, add_days, days_between, date_lt, date_gt, date_to_string, parse_date, day_of_week_name, day_of_year)
def test_is_leap_year_2024() -> unit ! { Test } = assert_eq(is_leap_year(cast(2024, i64)), true, "2024 leap (div 4 not 100)")
def test_is_leap_year_2023() -> unit ! { Test } = assert_eq(is_leap_year(cast(2023, i64)), false, "2023 not leap (not div 4)")
def test_is_leap_year_2000() -> unit ! { Test } = assert_eq(is_leap_year(cast(2000, i64)), true, "2000 leap (div 400)")
def test_is_leap_year_1900() -> unit ! { Test } = assert_eq(is_leap_year(cast(1900, i64)), false, "1900 not leap (div 100 not 400)")
def test_add_days_non_leap_year() -> unit ! { Test } = {
  start = date(cast(2026, i64), cast(1, i64), cast(1, i64))
  rolled = add_days(start, cast(365, i64))
  expected = date(cast(2027, i64), cast(1, i64), cast(1, i64))
  assert_eq(date_to_string(rolled), date_to_string(expected), "2026-01-01 + 365 days = 2027-01-01")
}
def test_add_days_leap_year() -> unit ! { Test } = {
  start = date(cast(2024, i64), cast(1, i64), cast(1, i64))
  rolled = add_days(start, cast(366, i64))
  expected = date(cast(2025, i64), cast(1, i64), cast(1, i64))
  assert_eq(date_to_string(rolled), date_to_string(expected), "2024-01-01 + 366 days = 2025-01-01")
}
def test_days_between_inverse() -> unit ! { Test } = {
  start = date(cast(2026, i64), cast(1, i64), cast(1, i64))
  later = add_days(start, cast(100, i64))
  _ = assert_eq(days_between(start, later), cast(100, i64), "+100 days inverse")
  earlier = add_days(start, cast(-50, i64))
  assert_eq(days_between(start, earlier), cast(-50, i64), "-50 days inverse")
}
def test_date_ordering() -> unit ! { Test } = {
  d1 = date(cast(2026, i64), cast(1, i64), cast(1, i64))
  d2 = date(cast(2026, i64), cast(1, i64), cast(2, i64))
  _ = assert_eq(date_lt(d1, d2), true, "d1 < d2")
  _ = assert_eq(date_gt(d2, d1), true, "d2 > d1")
  _ = assert_eq(date_lt(d2, d1), false, "not d2 < d1")
  assert_eq(date_gt(d1, d2), false, "not d1 > d2")
}
def test_try_date_rejects_feb_30_non_leap() -> unit ! { Test } =
  match try_date(cast(2026, i64), cast(2, i64), cast(30, i64)) with {
    | Some(_) => fail("2026-02-30 should not parse")
    | None => assert_true(true, "2026-02-30 rejected")
  }
def test_try_date_accepts_leap_day() -> unit ! { Test } =
  match try_date(cast(2024, i64), cast(2, i64), cast(29, i64)) with {
    | Some(_) => assert_true(true, "2024-02-29 accepted")
    | None => fail("2024-02-29 should be valid leap day")
  }
def test_try_date_rejects_feb_30_leap_year() -> unit ! { Test } =
  match try_date(cast(2024, i64), cast(2, i64), cast(30, i64)) with {
    | Some(_) => fail("2024-02-30 never exists")
    | None => assert_true(true, "2024-02-30 rejected even in leap year")
  }
def test_iso_round_trip() -> unit ! { Test } = {
  d = date(cast(2026, i64), cast(4, i64), cast(25, i64))
  s = date_to_string(d)
  _ = assert_eq(s, "2026-04-25", "ISO format shape")
  match parse_date(s) with {
    | Some(parsed) => assert_eq(date_to_string(parsed), s, "round-trip preserves date")
    | None => fail("parse_date(date_to_string(d)) was None")
  }
}
def test_day_of_week_known() -> unit ! { Test } = {
  jan1 = date(cast(2024, i64), cast(1, i64), cast(1, i64))
  jan2 = date(cast(2024, i64), cast(1, i64), cast(2, i64))
  _ = assert_eq(day_of_week_name(jan1), "monday", "2024-01-01 Monday")
  assert_eq(day_of_week_name(jan2), "tuesday", "2024-01-02 Tuesday")
}
def test_day_of_year_leap_end() -> unit ! { Test } = {
  leap_end = date(cast(2024, i64), cast(12, i64), cast(31, i64))
  _ = assert_eq(day_of_year(leap_end), cast(366, i64), "2024-12-31 is day 366")
  non_leap_end = date(cast(2023, i64), cast(12, i64), cast(31, i64))
  assert_eq(day_of_year(non_leap_end), cast(365, i64), "2023-12-31 is day 365")
}
def test_duration_construction() -> unit ! { Test } = {
  d = duration(cast(1, i64), cast(2, i64), cast(3, i64), cast(4, i64))
  _ = assert_eq(d.days, cast(1, i64), "days")
  _ = assert_eq(d.hours, cast(2, i64), "hours")
  _ = assert_eq(d.minutes, cast(3, i64), "minutes")
  assert_eq(d.seconds, cast(4, i64), "seconds")
}
