module Std.Tests.Datetime
import Std.Datetime (Weekday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, DayOverflow, ClampToMonthEnd, RejectInvalidDay, TimeUnit, Hours, Minutes, Seconds, Milliseconds, Microseconds, Nanoseconds, Date, Time, DateTime, Instant, Offset, OffsetDateTime, Duration, Period, Dates, Instants, is_leap_year, days_in_year, days_in_month, weekday_iso_number, weekday_from_iso_number, try_weekday_from_iso_number, weekday_name, date, try_date, date_year, date_month, date_day, date_epoch_day, date_from_epoch_day, try_date_from_epoch_day, date_weekday, date_day_of_year, date_iso_week, date_from_iso_week, try_date_from_iso_week, date_add_days, date_days_until, date_add_months, try_date_add_months, date_add_period, try_date_add_period, date_period_until, date_lt, date_lte, date_gt, date_gte, date_to_string, parse_date, try_parse_date, nth_weekday_in_month, last_weekday_in_month, weekday_on_or_after, weekday_on_or_before, easter_sunday_gregorian, easter_sunday_orthodox, time, try_time, time_hour, time_minute, time_second, time_nanosecond, time_nanosecond_of_day, time_from_nanosecond_of_day, try_time_from_nanosecond_of_day, time_add_duration, time_until, time_lt, time_lte, time_gt, time_gte, time_to_string, parse_time, try_parse_time, datetime, datetime_date, datetime_time, datetime_add_duration, datetime_add_period, try_datetime_add_period, datetime_until, datetime_lt, datetime_lte, datetime_gt, datetime_gte, datetime_to_string, parse_datetime, try_parse_datetime, offset_from_seconds, try_offset_from_seconds, offset_seconds, offset_to_string, parse_offset, try_parse_offset, instant_from_unix, try_instant_from_unix, instant_unix_second, instant_nanosecond, instant_from_unix_count, try_instant_from_unix_count, instant_to_unix_count, try_instant_to_unix_count, instant_add_duration, instant_until, instant_round_to, instant_lt, instant_lte, instant_gt, instant_gte, instant_to_datetime_at, datetime_to_instant_at, instant_to_string, parse_instant, try_parse_instant, offset_datetime, offset_datetime_instant, offset_datetime_offset, offset_datetime_local, offset_datetime_to_string, parse_offset_datetime, try_parse_offset_datetime, duration, duration_from_count, duration_second, duration_nanosecond, duration_to_count, try_duration_to_count, duration_to_seconds_f64, duration_add, duration_sub, duration_negate, duration_mul, duration_lt, duration_lte, duration_gt, duration_gte, duration_to_string, parse_duration, try_parse_duration, period, try_period, period_months, period_days, period_negate, period_mul, period_to_string, parse_period, try_parse_period, dates_from_epoch_days, try_dates_from_epoch_days, dates_epoch_days, instants_from_unix, try_instants_from_unix, instants_unix_seconds, instants_nanoseconds)
import Std.Rounding (RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)
import Std.Test (assert_eq, assert_true, assert_false, assert_eq_tensor)
-- Expected values come from Python's datetime (years 1..9999) and an
-- independent Hinnant reference (years -9999..0); see [05-OP-73].
def i64_minimum() -> i64 = sub(-9223372036854775807i64, 1i64)
def is_none_date(value: Option[Date]) -> bool =
  match value with {
    | Some(_) => false
    | None => true
  }
def date_text(value: Option[Date]) -> string =
  match value with {
    | Some(d) => date_to_string(d)
    | None => "none"
  }
def test_leap_years() -> unit ! { Test } = {
  _ = assert_true(is_leap_year(2024i64), "2024 is a leap year")
  _ = assert_true(is_leap_year(2000i64), "2000 is a leap year")
  _ = assert_true(is_leap_year(0i64), "year 0 is a leap year")
  _ = assert_true(is_leap_year(-4i64), "year -4 is a leap year")
  _ = assert_eq(days_in_year(2024i64), 366i64, "2024 has 366 days")
  assert_true(is_leap_year(i64_minimum()), "the i64 minimum is divisible by 400")
}
def test_non_leap_years() -> unit ! { Test } = {
  _ = assert_false(is_leap_year(1900i64), "1900 is not a leap year")
  _ = assert_false(is_leap_year(2023i64), "2023 is not a leap year")
  _ = assert_false(is_leap_year(-1i64), "year -1 is not a leap year")
  _ = assert_eq(days_in_year(2023i64), 365i64, "2023 has 365 days")
  assert_false(is_leap_year(9223372036854775807i64), "the i64 maximum is odd")
}
def test_days_in_month() -> unit ! { Test } = {
  _ = assert_eq(days_in_month(2024i64, 2i64), 29i64, "February 2024")
  _ = assert_eq(days_in_month(2023i64, 2i64), 28i64, "February 2023")
  _ = assert_eq(days_in_month(1900i64, 2i64), 28i64, "February 1900")
  _ = assert_eq(days_in_month(2026i64, 4i64), 30i64, "April")
  assert_eq(days_in_month(2026i64, 12i64), 31i64, "December")
}
def test_weekday_numbers_round_trip() -> unit ! { Test } = {
  _ = assert_eq(weekday_iso_number(Monday), 1i64, "Monday is 1")
  _ = assert_eq(weekday_iso_number(Sunday), 7i64, "Sunday is 7")
  _ = assert_eq(weekday_from_iso_number(4i64), Thursday, "4 is Thursday")
  _ = assert_eq(weekday_name(Wednesday), "wednesday", "lowercase ASCII name")
  assert_eq(try_weekday_from_iso_number(7i64), Some(Sunday), "try form of 7")
}
def test_weekday_number_rejects_out_of_range() -> unit ! { Test } = {
  _ = assert_eq(try_weekday_from_iso_number(0i64), None, "0 is not a weekday number")
  assert_eq(try_weekday_from_iso_number(8i64), None, "8 is not a weekday number")
}
def test_date_fields_round_trip() -> unit ! { Test } = {
  d = date(2024i64, 2i64, 29i64)
  _ = assert_eq(date_epoch_day(d), 19782i64, "2024-02-29 epoch day")
  _ = assert_eq(date_year(d), 2024i64, "year")
  _ = assert_eq(date_month(d), 2i64, "month")
  _ = assert_eq(date_day(d), 29i64, "day")
  _ = assert_eq(date_epoch_day(date(1970i64, 1i64, 1i64)), 0i64, "the epoch")
  _ = assert_eq(date_epoch_day(date(1900i64, 3i64, 1i64)), -25508i64, "1900-03-01")
  _ = assert_eq(date_epoch_day(date(0i64, 1i64, 1i64)), -719528i64, "0000-01-01")
  assert_eq(date_epoch_day(date(-1i64, 12i64, 31i64)), -719529i64, "-0001-12-31")
}
def test_date_range_edges() -> unit ! { Test } = {
  _ = assert_eq(date_epoch_day(date(-9999i64, 1i64, 1i64)), -4371587i64, "first supported day")
  _ = assert_eq(date_epoch_day(date(9999i64, 12i64, 31i64)), 2932896i64, "last supported day")
  _ = assert_eq(date_to_string(date_from_epoch_day(-4371587i64)), "-009999-01-01", "first day text")
  assert_eq(date_to_string(date_from_epoch_day(2932896i64)), "9999-12-31", "last day text")
}
def test_try_date_rejects_invalid_fields() -> unit ! { Test } = {
  _ = assert_true(is_none_date(try_date(2023i64, 2i64, 29i64)), "no February 29 in 2023")
  _ = assert_true(is_none_date(try_date(2024i64, 13i64, 1i64)), "no month 13")
  _ = assert_true(is_none_date(try_date(2024i64, 4i64, 31i64)), "no April 31")
  _ = assert_true(is_none_date(try_date(10000i64, 1i64, 1i64)), "year 10000 is outside the range")
  _ = assert_true(is_none_date(try_date(-10000i64, 12i64, 31i64)), "year -10000 is outside the range")
  _ = assert_true(is_none_date(try_date(2024i64, 1i64, 0i64)), "no day 0")
  assert_eq(date_text(try_date(2024i64, 2i64, 29i64)), "2024-02-29", "a valid try_date")
}
def test_epoch_day_bounds() -> unit ! { Test } = {
  _ = assert_true(is_none_date(try_date_from_epoch_day(2932897i64)), "one day past the end")
  _ = assert_true(is_none_date(try_date_from_epoch_day(-4371588i64)), "one day before the start")
  assert_eq(date_text(try_date_from_epoch_day(0i64)), "1970-01-01", "the epoch")
}
def test_weekday_and_day_of_year() -> unit ! { Test } = {
  _ = assert_eq(date_weekday(date(1970i64, 1i64, 1i64)), Thursday, "1970-01-01 is a Thursday")
  _ = assert_eq(date_weekday(date(2026i64, 10i64, 1i64)), Thursday, "2026-10-01 is a Thursday")
  _ = assert_eq(date_weekday(date(2021i64, 1i64, 3i64)), Sunday, "2021-01-03 is a Sunday")
  _ = assert_eq(date_weekday(date(-1i64, 12i64, 31i64)), Friday, "-0001-12-31 is a Friday")
  _ = assert_eq(date_day_of_year(date(2020i64, 12i64, 31i64)), 366i64, "last day of a leap year")
  assert_eq(date_day_of_year(date(2026i64, 10i64, 1i64)), 274i64, "2026-10-01 is day 274")
}
def test_iso_weeks() -> unit ! { Test } = {
  _ = assert_eq(date_iso_week(date(2026i64, 10i64, 1i64)), (2026i64, 40i64), "2026-W40")
  _ = assert_eq(date_iso_week(date(2021i64, 1i64, 3i64)), (2020i64, 53i64), "early January in the previous week-year")
  _ = assert_eq(date_iso_week(date(2008i64, 12i64, 29i64)), (2009i64, 1i64), "late December in the next week-year")
  _ = assert_eq(date_iso_week(date(9999i64, 12i64, 31i64)), (9999i64, 52i64), "last supported day")
  _ = assert_eq(date_iso_week(date(-9999i64, 1i64, 1i64)), (-9999i64, 1i64), "first supported day is a Monday in week 1")
  _ = assert_eq(date_to_string(date_from_iso_week(2026i64, 53i64, Sunday)), "2027-01-03", "2026 has 53 ISO weeks")
  _ = assert_eq(date_to_string(date_from_iso_week(2020i64, 53i64, Friday)), "2021-01-01", "2020-W53-5")
  assert_eq(date_to_string(date_from_iso_week(2009i64, 1i64, Monday)), "2008-12-29", "2009-W01-1")
}
def test_iso_week_rejects_missing_weeks() -> unit ! { Test } = {
  _ = assert_true(is_none_date(try_date_from_iso_week(2025i64, 53i64, Monday)), "2025 has 52 ISO weeks")
  _ = assert_true(is_none_date(try_date_from_iso_week(2026i64, 0i64, Monday)), "week 0 does not exist")
  _ = assert_true(is_none_date(try_date_from_iso_week(10000i64, 2i64, Monday)), "past the range")
  assert_true(is_none_date(try_date_from_iso_week(-10000i64, 1i64, Monday)), "before the range")
}
def test_date_add_days() -> unit ! { Test } = {
  _ = assert_eq(date_to_string(date_add_days(date(2024i64, 2i64, 28i64), 1i64)), "2024-02-29", "leap day")
  _ = assert_eq(date_to_string(date_add_days(date(2024i64, 3i64, 1i64), -1i64)), "2024-02-29", "negative days")
  _ = assert_eq(date_days_until(date(1970i64, 1i64, 1i64), date(2024i64, 2i64, 29i64)), 19782i64, "days until")
  _ = assert_eq(date_days_until(date(9999i64, 12i64, 31i64), date(-9999i64, 1i64, 1i64)), -7304483i64, "days until across the range")
  assert_eq(date_to_string(date_add_days(date(-9999i64, 1i64, 1i64), 7304483i64)), "9999-12-31", "whole range")
}
def test_date_add_months_policies() -> unit ! { Test } = {
  jan31 = date(2024i64, 1i64, 31i64)
  _ = assert_eq(date_to_string(date_add_months(jan31, 1i64, ClampToMonthEnd)), "2024-02-29", "clamp to month end")
  _ = assert_eq(date_to_string(date_add_months(jan31, 13i64, ClampToMonthEnd)), "2025-02-28", "clamp in a common year")
  _ = assert_eq(date_to_string(date_add_months(jan31, -1i64, RejectInvalidDay)), "2023-12-31", "an existing day never rejects")
  _ = assert_eq(date_to_string(date_add_months(date(2024i64, 3i64, 15i64), -15i64, RejectInvalidDay)), "2022-12-15", "negative months")
  assert_eq(date_to_string(date_add_months(date(-1i64, 12i64, 15i64), 1i64, RejectInvalidDay)), "0000-01-15", "across year 0")
}
def test_date_add_months_reject() -> unit ! { Test } = {
  _ = assert_true(is_none_date(try_date_add_months(date(2024i64, 1i64, 31i64), 1i64, RejectInvalidDay)), "no February 31")
  assert_eq(date_text(try_date_add_months(date(2024i64, 1i64, 31i64), 1i64, ClampToMonthEnd)), "2024-02-29", "clamp form succeeds")
}
def test_date_add_period() -> unit ! { Test } = {
  _ = assert_eq(date_to_string(date_add_period(date(2024i64, 1i64, 31i64), period(1i64, 1i64), ClampToMonthEnd)), "2024-03-01", "months then days")
  _ = assert_eq(date_to_string(date_add_period(date(2024i64, 3i64, 1i64), period(-1i64, -1i64), ClampToMonthEnd)), "2024-01-31", "negative period")
  assert_true(is_none_date(try_date_add_period(date(2024i64, 1i64, 31i64), period(1i64, 1i64), RejectInvalidDay)), "reject fires on the month step")
}
def test_date_period_until() -> unit ! { Test } = {
  _ = assert_eq(period_to_string(date_period_until(date(2024i64, 1i64, 31i64), date(2024i64, 3i64, 1i64))), "P1M1D", "January 31 to March 1")
  _ = assert_eq(period_to_string(date_period_until(date(2024i64, 1i64, 31i64), date(2024i64, 2i64, 29i64))), "P1M", "clamped month")
  _ = assert_eq(period_to_string(date_period_until(date(2024i64, 3i64, 1i64), date(2024i64, 1i64, 31i64))), "-P1M1D", "mirrored")
  _ = assert_eq(period_to_string(date_period_until(date(2024i64, 3i64, 31i64), date(2024i64, 2i64, 29i64))), "-P1M", "mirrored onto a clamped month end")
  _ = assert_eq(period_to_string(date_period_until(date(2024i64, 3i64, 31i64), date(2024i64, 2i64, 27i64))), "-P1M2D", "mirrored with days")
  _ = assert_eq(period_to_string(date_period_until(date(2024i64, 5i64, 5i64), date(2024i64, 5i64, 5i64))), "P0D", "same day")
  a = date(2024i64, 3i64, 31i64)
  b = date(2023i64, 2i64, 28i64)
  assert_eq(date_add_period(a, date_period_until(a, b), ClampToMonthEnd), b, "round trip")
}
def test_date_equality() -> unit ! { Test } = {
  _ = assert_true(eq(date(2024i64, 2i64, 29i64), parse_date("2024-02-29")), "equal dates are eq")
  _ = assert_false(eq(date(2024i64, 2i64, 29i64), date(2024i64, 3i64, 1i64)), "different dates are not eq")
  assert_true(neq(date(2024i64, 2i64, 29i64), date(2024i64, 3i64, 1i64)), "different dates are neq")
}
def test_date_comparisons() -> unit ! { Test } = {
  a = date(2024i64, 1i64, 1i64)
  b = date(2024i64, 1i64, 2i64)
  _ = assert_true(date_lt(a, b), "lt")
  _ = assert_true(date_lte(a, a), "lte reflexive")
  _ = assert_true(date_gt(b, a), "gt")
  _ = assert_true(date_gte(b, b), "gte reflexive")
  _ = assert_false(date_lt(b, a), "lt is strict")
  assert_false(date_gt(a, a), "gt is strict")
}
def test_date_text_round_trip() -> unit ! { Test } = {
  _ = assert_eq(date_to_string(date(44i64, 3i64, 15i64)), "0044-03-15", "four-digit padding")
  _ = assert_eq(date_to_string(date(-44i64, 3i64, 15i64)), "-000044-03-15", "negative year")
  _ = assert_eq(parse_date("-000044-03-15"), date(-44i64, 3i64, 15i64), "parse a negative year")
  _ = assert_eq(parse_date("+002024-02-29"), date(2024i64, 2i64, 29i64), "parse a signed six-digit year")
  _ = assert_eq(date_to_string(parse_date("0000-01-01")), "0000-01-01", "year 0")
  assert_eq(date_text(try_parse_date("9999-12-31")), "9999-12-31", "last day")
}
def test_date_text_rejects() -> unit ! { Test } = {
  _ = assert_true(is_none_date(try_parse_date("2024-1-01")), "one-digit month")
  _ = assert_true(is_none_date(try_parse_date("-000000-01-01")), "negative zero year")
  _ = assert_true(is_none_date(try_parse_date("+010000-01-01")), "year 10000")
  _ = assert_true(is_none_date(try_parse_date("2024-02-30")), "February 30")
  _ = assert_true(is_none_date(try_parse_date("2024-02-01 ")), "trailing text")
  _ = assert_true(is_none_date(try_parse_date("")), "empty text")
  assert_true(is_none_date(try_parse_date("20240201")), "basic format is not in the profile")
}
def test_holiday_helpers() -> unit ! { Test } = {
  _ = assert_eq(date_text(nth_weekday_in_month(2026i64, 11i64, Thursday, 4i64)), "2026-11-26", "US Thanksgiving 2026")
  _ = assert_eq(date_text(nth_weekday_in_month(2026i64, 9i64, Monday, 1i64)), "2026-09-07", "US Labor Day 2026")
  _ = assert_eq(date_to_string(last_weekday_in_month(2026i64, 5i64, Monday)), "2026-05-25", "US Memorial Day 2026")
  _ = assert_eq(date_to_string(weekday_on_or_after(date(2026i64, 10i64, 1i64), Thursday)), "2026-10-01", "on or after includes the day")
  _ = assert_eq(date_to_string(weekday_on_or_after(date(2026i64, 10i64, 1i64), Wednesday)), "2026-10-07", "on or after")
  assert_eq(date_to_string(weekday_on_or_before(date(2026i64, 10i64, 1i64), Friday)), "2026-09-25", "on or before")
}
def test_nth_weekday_absent() -> unit ! { Test } = {
  _ = assert_true(is_none_date(nth_weekday_in_month(2026i64, 2i64, Monday, 5i64)), "February 2026 has four Mondays")
  assert_true(is_none_date(nth_weekday_in_month(2026i64, 2i64, Monday, 9223372036854775807i64)), "a huge occurrence is absent, not an overflow")
}
def test_easter() -> unit ! { Test } = {
  _ = assert_eq(date_to_string(easter_sunday_gregorian(2024i64)), "2024-03-31", "Easter 2024")
  _ = assert_eq(date_to_string(easter_sunday_gregorian(2025i64)), "2025-04-20", "Easter 2025")
  _ = assert_eq(date_to_string(easter_sunday_gregorian(2000i64)), "2000-04-23", "Easter 2000")
  _ = assert_eq(date_to_string(easter_sunday_gregorian(-9999i64)), "-009999-04-22", "Easter -9999")
  _ = assert_eq(date_to_string(easter_sunday_orthodox(2024i64)), "2024-05-05", "Orthodox Easter 2024")
  _ = assert_eq(date_to_string(easter_sunday_orthodox(2023i64)), "2023-04-16", "Orthodox Easter 2023")
  _ = assert_eq(date_to_string(easter_sunday_orthodox(9999i64)), "9999-06-27", "Orthodox Easter 9999")
  assert_eq(date_epoch_day(easter_sunday_orthodox(-9999i64)), -4371567i64, "Orthodox Easter -9999")
}
def test_time_fields() -> unit ! { Test } = {
  t = time(23i64, 59i64, 58i64, 123456789i64)
  _ = assert_eq(time_hour(t), 23i64, "hour")
  _ = assert_eq(time_minute(t), 59i64, "minute")
  _ = assert_eq(time_second(t), 58i64, "second")
  _ = assert_eq(time_nanosecond(t), 123456789i64, "nanosecond")
  _ = assert_eq(time_nanosecond_of_day(t), 86398123456789i64, "nanosecond of day")
  _ = assert_eq(time_from_nanosecond_of_day(86398123456789i64), t, "round trip")
  assert_eq(time_to_string(t), "23:59:58.123456789", "text")
}
def test_time_rejects_fields() -> unit ! { Test } = {
  _ = assert_eq(try_time(24i64, 0i64, 0i64, 0i64), None, "no 24:00")
  _ = assert_eq(try_time(23i64, 59i64, 60i64, 0i64), None, "no leap second")
  _ = assert_eq(try_time(0i64, 0i64, 0i64, 1000000000i64), None, "nanosecond range")
  _ = assert_eq(try_time_from_nanosecond_of_day(86400000000000i64), None, "a full day is not a time")
  assert_eq(try_time_from_nanosecond_of_day(-1i64), None, "negative nanosecond of day")
}
def test_time_add_duration_carries_days() -> unit ! { Test } = {
  (days, t) = time_add_duration(time(23i64, 0i64, 0i64, 0i64), duration(7200i64, 0i64))
  _ = assert_eq(days, 1i64, "one day carried")
  _ = assert_eq(time_to_string(t), "01:00:00", "wrapped time")
  (back_days, back) = time_add_duration(time(0i64, 0i64, 0i64, 0i64), duration(-1i64, 500000000i64))
  _ = assert_eq(back_days, -1i64, "one day borrowed")
  _ = assert_eq(time_to_string(back), "23:59:59.5", "half a second before midnight")
  (max_days, _wrapped) = time_add_duration(time(0i64, 0i64, 0i64, 0i64), duration(9223372036854775807i64, 999999999i64))
  assert_eq(max_days, 106751991167300i64, "the largest duration carries without overflow")
}
def test_time_until_and_compare() -> unit ! { Test } = {
  a = time(10i64, 0i64, 0i64, 0i64)
  b = time(9i64, 59i64, 59i64, 500000000i64)
  _ = assert_eq(duration_to_string(time_until(a, b)), "-PT0.5S", "negative within a day")
  _ = assert_true(time_lt(b, a), "lt")
  _ = assert_true(time_lte(a, a), "lte")
  _ = assert_true(time_gt(a, b), "gt")
  _ = assert_true(time_gte(b, b), "gte")
  assert_false(time_lt(a, b), "lt strict")
}
def test_time_text() -> unit ! { Test } = {
  _ = assert_eq(parse_time("09:30:00"), time(9i64, 30i64, 0i64, 0i64), "whole seconds")
  _ = assert_eq(parse_time("09:30:00.5"), time(9i64, 30i64, 0i64, 500000000i64), "one fractional digit")
  _ = assert_eq(time_to_string(parse_time("09:30:00.000000001")), "09:30:00.000000001", "nine fractional digits")
  assert_eq(time_to_string(time(0i64, 0i64, 0i64, 120000000i64)), "00:00:00.12", "trailing zeros removed")
}
def test_time_text_rejects() -> unit ! { Test } = {
  _ = assert_eq(try_parse_time("24:00:00"), None, "hour 24")
  _ = assert_eq(try_parse_time("23:59:60"), None, "second 60")
  _ = assert_eq(try_parse_time("09:30:00.1234567890"), None, "ten fractional digits")
  _ = assert_eq(try_parse_time("09:30:00."), None, "empty fraction")
  assert_eq(try_parse_time("9:30:00"), None, "one-digit hour")
}
def test_datetime_arithmetic() -> unit ! { Test } = {
  dt = datetime(date(2024i64, 2i64, 28i64), time(23i64, 30i64, 0i64, 0i64))
  _ = assert_eq(datetime_to_string(datetime_add_duration(dt, duration(3600i64, 0i64))), "2024-02-29T00:30:00", "crosses midnight")
  _ = assert_eq(datetime_to_string(datetime_add_duration(dt, duration(-86400i64, 0i64))), "2024-02-27T23:30:00", "one civil day back")
  _ = assert_eq(datetime_to_string(datetime_add_period(dt, period(12i64, 1i64), ClampToMonthEnd)), "2025-03-01T23:30:00", "period changes only the date")
  _ = assert_eq(datetime_date(dt), date(2024i64, 2i64, 28i64), "date part")
  _ = assert_eq(datetime_time(dt), time(23i64, 30i64, 0i64, 0i64), "time part")
  first = datetime(date(-9999i64, 1i64, 1i64), time(0i64, 0i64, 0i64, 0i64))
  last = datetime(date(9999i64, 12i64, 31i64), time(23i64, 59i64, 59i64, 999999999i64))
  assert_eq(duration_to_string(datetime_until(first, last)), "PT631107417599.999999999S", "the whole civil range")
}
def test_datetime_period_reject() -> unit ! { Test } = {
  dt = datetime(date(2024i64, 1i64, 31i64), time(12i64, 0i64, 0i64, 0i64))
  _ = assert_eq(try_datetime_add_period(dt, period(1i64, 0i64), RejectInvalidDay), None, "reject fires")
  _ = assert_true(datetime_lt(dt, datetime_add_duration(dt, duration(0i64, 1i64))), "lt by a nanosecond")
  _ = assert_true(datetime_lte(dt, dt), "lte")
  _ = assert_false(datetime_gt(dt, dt), "gt strict")
  assert_true(datetime_gte(dt, dt), "gte")
}
def test_datetime_text() -> unit ! { Test } = {
  _ = assert_eq(datetime_to_string(parse_datetime("2026-10-01t09:30:00")), "2026-10-01T09:30:00", "lowercase separator")
  _ = assert_eq(datetime_to_string(parse_datetime("2026-10-01 09:30:00.25")), "2026-10-01T09:30:00.25", "space separator")
  _ = assert_eq(try_parse_datetime("2026-10-01T09:30"), None, "seconds are required")
  _ = assert_eq(try_parse_datetime("2026-02-30T09:30:00"), None, "invalid day")
  assert_eq(try_parse_datetime("2026-10-01T09:30:00Z"), None, "an offset is not part of a civil datetime")
}
def test_offsets() -> unit ! { Test } = {
  _ = assert_eq(offset_to_string(offset_from_seconds(0i64)), "Z", "zero is Z")
  _ = assert_eq(offset_to_string(offset_from_seconds(-14400i64)), "-04:00", "negative offset")
  _ = assert_eq(offset_to_string(offset_from_seconds(86399i64)), "+23:59:59", "seconds shown when nonzero")
  _ = assert_eq(offset_seconds(parse_offset("-00:00")), 0i64, "negative zero is zero")
  _ = assert_eq(offset_seconds(parse_offset("z")), 0i64, "lowercase z")
  assert_eq(offset_seconds(parse_offset("+05:30")), 19800i64, "India")
}
def test_offsets_reject() -> unit ! { Test } = {
  _ = assert_eq(try_offset_from_seconds(86400i64), None, "a full day")
  _ = assert_eq(try_offset_from_seconds(-86400i64), None, "a full day back")
  _ = assert_eq(try_parse_offset("+24:00"), None, "hour 24")
  _ = assert_eq(try_parse_offset("+05:60"), None, "minute 60")
  assert_eq(try_parse_offset("+0530"), None, "basic format")
}
def test_instants() -> unit ! { Test } = {
  i = instant_from_unix(1727775000i64, 5i64)
  _ = assert_eq(instant_unix_second(i), 1727775000i64, "second")
  _ = assert_eq(instant_nanosecond(i), 5i64, "nanosecond")
  _ = assert_eq(instant_to_string(i), "2024-10-01T09:30:00.000000005Z", "text in UTC")
  _ = assert_eq(instant_to_string(instant_from_unix(-377705030401i64, 0i64)), "-009999-01-01T23:59:59Z", "first instant")
  _ = assert_eq(instant_to_string(instant_from_unix(253402214400i64, 999999999i64)), "9999-12-31T00:00:00.999999999Z", "last instant")
  assert_eq(parse_instant("2024-10-01T05:30:00.000000005-04:00"), i, "parse converts the offset exactly")
}
def test_instants_reject() -> unit ! { Test } = {
  _ = assert_eq(try_instant_from_unix(0i64, 1000000000i64), None, "nanosecond range")
  _ = assert_eq(try_instant_from_unix(0i64, -1i64), None, "negative nanosecond")
  _ = assert_eq(try_instant_from_unix(253402214401i64, 0i64), None, "past the last instant")
  _ = assert_eq(try_instant_from_unix(-377705030402i64, 0i64), None, "before the first instant")
  _ = assert_eq(try_parse_instant("9999-12-31T23:59:59-00:01"), None, "parsed instant past the range")
  assert_eq(try_parse_instant("2024-10-01T05:30:00"), None, "an instant needs an offset")
}
def test_unix_counts() -> unit ! { Test } = {
  i = instant_from_unix(-2i64, 500000000i64)
  _ = assert_eq(instant_to_unix_count(i, Seconds, RoundTowardNegative), -2i64, "floor")
  _ = assert_eq(instant_to_unix_count(i, Seconds, RoundTowardPositive), -1i64, "ceiling")
  _ = assert_eq(instant_to_unix_count(i, Seconds, RoundTowardZero), -1i64, "toward zero")
  _ = assert_eq(instant_to_unix_count(i, Seconds, RoundTiesToEven), -2i64, "tie to even")
  _ = assert_eq(instant_to_unix_count(i, Milliseconds, RoundTowardNegative), -1500i64, "milliseconds")
  _ = assert_eq(instant_to_unix_count(instant_from_unix(5400i64, 0i64), Hours, RoundTiesToEven), 2i64, "1.5 hours ties to 2")
  _ = assert_eq(instant_to_unix_count(instant_from_unix(-9223372037i64, 145224192i64), Nanoseconds, RoundTowardNegative), i64_minimum(), "the i64 minimum in nanoseconds")
  _ = assert_eq(instant_from_unix_count(-1500i64, Milliseconds), i, "milliseconds back")
  assert_eq(instant_from_unix_count(2i64, Hours), instant_from_unix(7200i64, 0i64), "hours")
}
def test_rounding_modes() -> unit ! { Test } = {
  minus = instant_from_unix(-3i64, 500000000i64)
  plus = instant_from_unix(2i64, 500000000i64)
  _ = assert_eq(instant_to_unix_count(minus, Seconds, RoundAwayFromZero), -3i64, "away from zero below the epoch")
  _ = assert_eq(instant_to_unix_count(plus, Seconds, RoundAwayFromZero), 3i64, "away from zero above the epoch")
  _ = assert_eq(instant_to_unix_count(minus, Seconds, RoundTiesToAway), -3i64, "tie away below the epoch")
  _ = assert_eq(instant_to_unix_count(plus, Seconds, RoundTiesToAway), 3i64, "tie away above the epoch")
  _ = assert_eq(instant_to_unix_count(plus, Seconds, RoundTiesToEven), 2i64, "tie to even")
  _ = assert_eq(instant_to_unix_count(instant_from_unix(-2i64, 0i64), Seconds, RejectInexact), -2i64, "an exact count passes")
  _ = assert_eq(duration_to_count(duration(-90i64, 0i64), Minutes, RoundAwayFromZero), -2i64, "duration away from zero")
  _ = assert_eq(duration_to_count(duration(120i64, 0i64), Minutes, RejectInexact), 2i64, "exact minutes")
  _ = assert_eq(instant_unix_second(instant_round_to(instant_from_unix(120i64, 0i64), duration(60i64, 0i64), RejectInexact)), 120i64, "an exact multiple is kept")
  assert_eq(instant_unix_second(instant_round_to(instant_from_unix(-90i64, 0i64), duration(60i64, 0i64), RoundTiesToAway)), -120i64, "an instant tie away from the epoch")
}
def test_rounding_reject_inexact() -> unit ! { Test } = {
  _ = assert_eq(try_instant_to_unix_count(instant_from_unix(0i64, 1i64), Seconds, RejectInexact), None, "an inexact instant count")
  _ = assert_eq(try_duration_to_count(duration(90i64, 0i64), Minutes, RejectInexact), None, "an inexact duration count")
  assert_eq(try_duration_to_count(duration(0i64, 1000000i64), Milliseconds, RejectInexact), Some(1i64), "an exact millisecond")
}
def test_unix_counts_reject() -> unit ! { Test } = {
  _ = assert_eq(try_instant_to_unix_count(instant_from_unix(-9223372037i64, 145224191i64), Nanoseconds, RoundTowardNegative), None, "one nanosecond below the i64 minimum")
  _ = assert_eq(try_instant_to_unix_count(instant_from_unix(253402214400i64, 0i64), Nanoseconds, RoundTowardNegative), None, "the last instant in nanoseconds")
  _ = assert_eq(try_instant_from_unix_count(9223372036854775807i64, Seconds), None, "too many seconds")
  _ = assert_eq(try_instant_from_unix_count(9223372036854775807i64, Hours), None, "too many hours")
  assert_eq(try_instant_from_unix_count(9223372036854775807i64, Nanoseconds), Some(instant_from_unix(9223372036i64, 854775807i64)), "the largest nanosecond count is in range")
}
def test_instant_arithmetic() -> unit ! { Test } = {
  a = instant_from_unix(0i64, 999999999i64)
  b = instant_add_duration(a, duration(0i64, 1i64))
  _ = assert_eq(b, instant_from_unix(1i64, 0i64), "carry into the second")
  _ = assert_eq(instant_until(b, a), duration(0i64, -1i64), "negative until")
  _ = assert_true(instant_lt(a, b), "lt")
  _ = assert_true(instant_lte(a, a), "lte")
  _ = assert_true(instant_gt(b, a), "gt")
  _ = assert_true(instant_gte(b, b), "gte")
  first = instant_from_unix(-377705030401i64, 0i64)
  last = instant_from_unix(253402214400i64, 0i64)
  assert_eq(duration_second(instant_until(first, last)), 631107244801i64, "largest instant difference")
}
def test_instant_rounding() -> unit ! { Test } = {
  i = instant_from_unix(-90i64, 0i64)
  minute = duration(60i64, 0i64)
  _ = assert_eq(instant_unix_second(instant_round_to(i, minute, RoundTowardNegative)), -120i64, "floor")
  _ = assert_eq(instant_unix_second(instant_round_to(i, minute, RoundTowardPositive)), -60i64, "ceiling")
  _ = assert_eq(instant_unix_second(instant_round_to(i, minute, RoundTowardZero)), -60i64, "toward the epoch")
  _ = assert_eq(instant_unix_second(instant_round_to(i, minute, RoundTiesToEven)), -120i64, "tie to the even multiple")
  _ = assert_eq(instant_unix_second(instant_round_to(instant_from_unix(30i64, 0i64), minute, RoundTiesToEven)), 0i64, "tie to zero")
  assert_eq(instant_to_string(instant_round_to(instant_from_unix(100000i64, 1i64), duration(86400i64, 0i64), RoundTowardPositive)), "1970-01-03T00:00:00Z", "whole days")
}
def test_civil_readings() -> unit ! { Test } = {
  i = instant_from_unix(0i64, 0i64)
  new_york = offset_from_seconds(-14400i64)
  local = instant_to_datetime_at(i, new_york)
  _ = assert_eq(datetime_to_string(local), "1969-12-31T20:00:00", "local reading")
  _ = assert_eq(datetime_to_instant_at(local, new_york), i, "back to the instant")
  _ = assert_eq(datetime_to_string(instant_to_datetime_at(instant_from_unix(-377705030401i64, 0i64), offset_from_seconds(-86399i64))), "-009999-01-01T00:00:00", "the first instant reads at the far offset")
  assert_eq(datetime_to_string(instant_to_datetime_at(instant_from_unix(253402214400i64, 999999999i64), offset_from_seconds(86399i64))), "9999-12-31T23:59:59.999999999", "the last instant reads at the far offset")
}
def test_offset_datetimes() -> unit ! { Test } = {
  odt = parse_offset_datetime("2026-10-01T09:30:00-04:00")
  _ = assert_eq(offset_datetime_to_string(odt), "2026-10-01T09:30:00-04:00", "the written offset survives")
  _ = assert_eq(instant_to_string(offset_datetime_instant(odt)), "2026-10-01T13:30:00Z", "instant")
  _ = assert_eq(offset_seconds(offset_datetime_offset(odt)), -14400i64, "offset")
  _ = assert_eq(datetime_to_string(offset_datetime_local(odt)), "2026-10-01T09:30:00", "local reading")
  _ = assert_eq(offset_datetime_to_string(parse_offset_datetime("2026-10-01T13:30:00Z")), "2026-10-01T13:30:00Z", "Z gives offset zero")
  assert_eq(offset_datetime(offset_datetime_instant(odt), offset_datetime_offset(odt)), odt, "construction round trip")
}
def test_offset_datetimes_reject() -> unit ! { Test } = {
  _ = assert_eq(try_parse_offset_datetime("2026-10-01T09:30:00"), None, "offset required")
  _ = assert_eq(try_parse_offset_datetime("2026-10-01T09:30:00+24:00"), None, "offset hour 24")
  assert_eq(try_parse_offset_datetime("2026-10-01T09:30:60Z"), None, "second 60")
}
def test_durations() -> unit ! { Test } = {
  _ = assert_eq(duration(-1i64, -500000000i64), duration(-2i64, 500000000i64), "Euclidean normalization")
  _ = assert_eq(duration_second(duration(0i64, -1i64)), -1i64, "negative nanosecond borrows a second")
  _ = assert_eq(duration_nanosecond(duration(0i64, -1i64)), 999999999i64, "nonnegative nanosecond field")
  _ = assert_eq(duration_from_count(-1500i64, Milliseconds), duration(-2i64, 500000000i64), "from milliseconds")
  _ = assert_eq(duration_from_count(3i64, Hours), duration(10800i64, 0i64), "from hours")
  _ = assert_eq(duration_to_count(duration(90i64, 0i64), Minutes, RoundTiesToEven), 2i64, "1.5 minutes ties to 2")
  _ = assert_eq(duration_to_count(duration(-90i64, 0i64), Minutes, RoundTowardZero), -1i64, "toward zero")
  _ = assert_eq(duration_to_count(duration(9223372036854775807i64, 0i64), Seconds, RoundTowardPositive), 9223372036854775807i64, "largest whole second")
  _ = assert_eq(duration_to_seconds_f64(duration(0i64, -1i64)), -1e-9f64, "one negative nanosecond keeps its sign")
  assert_eq(duration_to_seconds_f64(duration(-2i64, 500000000i64)), -1.5f64, "float boundary")
}
def test_durations_reject() -> unit ! { Test } = {
  _ = assert_eq(try_duration_to_count(duration(9223372036854775807i64, 1i64), Seconds, RoundTowardPositive), None, "rounding past the i64 maximum")
  _ = assert_eq(try_duration_to_count(duration(9223372037i64, 0i64), Nanoseconds, RoundTowardNegative), None, "too many nanoseconds")
  assert_eq(try_duration_to_count(duration(9223372036i64, 854775807i64), Nanoseconds, RoundTowardNegative), Some(9223372036854775807i64), "the largest nanosecond count")
}
def test_duration_arithmetic() -> unit ! { Test } = {
  lowest = duration(i64_minimum(), 0i64)
  _ = assert_eq(duration_add(duration(1i64, 600000000i64), duration(2i64, 600000000i64)), duration(4i64, 200000000i64), "add with carry")
  _ = assert_eq(duration_add(duration(i64_minimum(), 500000000i64), duration(-1i64, 500000000i64)), lowest, "add reaching the minimum")
  _ = assert_eq(duration_sub(duration(-1i64, 0i64), lowest), duration(9223372036854775807i64, 0i64), "sub reaching the maximum")
  _ = assert_eq(duration_negate(duration(-1i64, 500000000i64)), duration(0i64, 500000000i64), "negate a fraction")
  _ = assert_eq(duration_negate(duration(9223372036854775807i64, 0i64)), duration(-9223372036854775807i64, 0i64), "negate the maximum")
  _ = assert_eq(duration_mul(duration(1i64, 500000000i64), 3i64), duration(4i64, 500000000i64), "multiply")
  _ = assert_eq(duration_mul(duration(-2i64, 1i64), 4611686018427387905i64), duration(-9223372032243089792i64, 427387905i64), "a whole-second product past the minimum with a representable result")
  _ = assert_eq(duration_mul(duration(-1i64, 999999999i64), i64_minimum()), duration(9223372036i64, 854775808i64), "multiply by the minimum")
  _ = assert_true(duration_lt(duration(-1i64, 999999999i64), duration(0i64, 0i64)), "lt")
  _ = assert_true(duration_lte(lowest, lowest), "lte")
  _ = assert_true(duration_gt(duration(0i64, 1i64), duration(0i64, 0i64)), "gt")
  assert_true(duration_gte(lowest, lowest), "gte")
}
def test_duration_text() -> unit ! { Test } = {
  _ = assert_eq(duration_to_string(duration(3661i64, 0i64)), "PT3661S", "total seconds")
  _ = assert_eq(duration_to_string(duration(-1i64, 500000000i64)), "-PT0.5S", "negative fraction")
  _ = assert_eq(duration_to_string(duration(0i64, 0i64)), "PT0S", "zero")
  _ = assert_eq(duration_to_string(duration(i64_minimum(), 0i64)), "-PT9223372036854775808S", "the minimum")
  _ = assert_eq(parse_duration("PT1H1M1S"), duration(3661i64, 0i64), "all components")
  _ = assert_eq(parse_duration("-PT1.5S"), duration(-2i64, 500000000i64), "negative fraction")
  _ = assert_eq(parse_duration("-PT9223372036854775808S"), duration(i64_minimum(), 0i64), "the minimum parses")
  _ = assert_eq(parse_duration("-PT9223372036854775807.5S"), duration(i64_minimum(), 500000000i64), "the minimum with a fraction")
  _ = assert_eq(parse_duration("PT0002M"), duration(120i64, 0i64), "leading zeros")
  assert_eq(parse_duration("-PT0S"), duration(0i64, 0i64), "negative zero")
}
def test_duration_text_rejects() -> unit ! { Test } = {
  _ = assert_eq(try_parse_duration("PT"), None, "no component")
  _ = assert_eq(try_parse_duration("P1D"), None, "no day unit")
  _ = assert_eq(try_parse_duration("PT1M1H"), None, "components out of order")
  _ = assert_eq(try_parse_duration("PT1.5M"), None, "fraction only on seconds")
  _ = assert_eq(try_parse_duration("PT1.0000000001S"), None, "ten fractional digits")
  _ = assert_eq(try_parse_duration("pt1s"), None, "designators are uppercase")
  _ = assert_eq(try_parse_duration("PT9223372036854775808S"), None, "a value past the duration range")
  _ = assert_eq(try_parse_duration("PT2562047788015216H"), None, "components fit but the whole does not")
  assert_eq(try_parse_duration("PT-1S"), None, "no inner sign")
}
def test_periods() -> unit ! { Test } = {
  _ = assert_eq(period_months(period(14i64, 3i64)), 14i64, "months")
  _ = assert_eq(period_days(period(14i64, 3i64)), 3i64, "days")
  _ = assert_eq(period_to_string(period(14i64, 3i64)), "P14M3D", "text")
  _ = assert_eq(period_to_string(period(0i64, 0i64)), "P0D", "zero")
  _ = assert_eq(period_to_string(period(-12i64, 0i64)), "-P12M", "negative")
  _ = assert_eq(period_negate(period(1i64, 2i64)), period(-1i64, -2i64), "negate")
  _ = assert_eq(period_mul(period(1i64, 2i64), -3i64), period(-3i64, -6i64), "multiply")
  _ = assert_eq(parse_period("P1Y2M3W4D"), period(14i64, 25i64), "years and weeks")
  _ = assert_eq(parse_period("-P1Y"), period(-12i64, 0i64), "negative year")
  _ = assert_eq(parse_period("-P9223372036854775808M"), period(sub(-9223372036854775807i64, 1i64), 0i64), "the most negative months")
  assert_eq(parse_period("P12M"), parse_period("P1Y"), "a year is twelve months")
}
def test_periods_reject() -> unit ! { Test } = {
  _ = assert_eq(try_period(1i64, -1i64), None, "mixed signs")
  _ = assert_eq(try_period(-1i64, 1i64), None, "mixed signs mirrored")
  _ = assert_eq(try_parse_period("P"), None, "no component")
  _ = assert_eq(try_parse_period("PT1H"), None, "no time part")
  _ = assert_eq(try_parse_period("P1D1M"), None, "components out of order")
  _ = assert_eq(try_parse_period("P9223372036854775808D"), None, "a value past the period range")
  _ = assert_eq(try_parse_period("P768614336404564650Y8M"), None, "years and months that do not fit together")
  assert_eq(try_parse_period("P1M-1D"), None, "no inner sign")
}
def test_date_columns() -> unit ! { Test } = {
  ds = dates_from_epoch_days(to_tensor([0i64, 19782i64, -4371587i64]))
  _ = assert_eq_tensor(dates_epoch_days(ds), to_tensor([0i64, 19782i64, -4371587i64]), "epoch days round trip")
  (masked, valid) = try_dates_from_epoch_days(to_tensor([1i64, 2932897i64, -4371588i64, 2932896i64]))
  _ = assert_eq_tensor(valid, to_tensor([true, false, false, true]), "validity mask")
  assert_eq_tensor(dates_epoch_days(masked), to_tensor([1i64, 0i64, 0i64, 2932896i64]), "invalid positions hold the epoch")
}
def test_instant_columns() -> unit ! { Test } = {
  column = instants_from_unix(to_tensor([0i64, -1i64]), to_tensor([5i64, 999999999i64]))
  _ = assert_eq_tensor(instants_unix_seconds(column), to_tensor([0i64, -1i64]), "seconds")
  _ = assert_eq_tensor(instants_nanoseconds(column), to_tensor([5i64, 999999999i64]), "nanoseconds")
  (masked, valid) = try_instants_from_unix(to_tensor([0i64, 253402214401i64, 3i64]), to_tensor([1000000000i64, 0i64, 7i64]))
  _ = assert_eq_tensor(valid, to_tensor([false, false, true]), "validity mask")
  assert_eq_tensor(instants_nanoseconds(masked), to_tensor([0i64, 0i64, 7i64]), "invalid positions hold the epoch")
}
