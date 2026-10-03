module Std.Tests.Datetime.Columns
import Std.Datetime (ClampToMonthEnd, RejectInvalidDay, Hours, Minutes, Seconds, Milliseconds, Microseconds, Nanoseconds, Dates, Instants, Duration, date, instant_from_unix, offset_from_seconds, duration, dates_from_epoch_days, dates_epoch_days, instants_from_unix, instants_unix_seconds, instants_nanoseconds)
import Std.Datetime.Columns (Durations, durations, try_durations, durations_seconds, durations_nanoseconds, dates_year, dates_month, dates_day, dates_weekday_iso_number, dates_day_of_year, dates_from_ymd, try_dates_from_ymd, dates_add_days, dates_add_months, dates_days_until, dates_lt, dates_lte, dates_gt, dates_gte, try_parse_dates, dates_to_strings, instants_from_unix_count, instants_to_unix_count, instants_add_duration, instants_until, instants_round_to, instants_to_dates_at, instants_seconds_since_f64, instants_lt, instants_lte, instants_gt, instants_gte)
import Std.Rounding (RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)
import Std.Test (assert_eq, assert_eq_tensor)
-- Expected values come from Python's datetime and the independent reference
-- in scripts/datetime_reference.py; see [05-OP-73]. Failing calls are pinned
-- with their exact messages by crates/chelis-cli/tests/std_datetime_columns.rs.
-- 2024-02-29, 1969-12-31, -0044-03-15, 9999-12-31, -9999-01-01.
def sample_days() -> tensor[5, i64] = to_tensor([19782i64, -1i64, -735525i64, 2932896i64, -4371587i64])
def samples() -> Dates[5] = dates_from_epoch_days(sample_days())
def stamps() -> Instants[4] = instants_from_unix(to_tensor([0i64, -1i64, 1700000000i64, 253402214400i64]), to_tensor([0i64, 999999999i64, 500000000i64, 999999999i64]))
def test_date_fields() -> unit ! { Test } = {
  _ = assert_eq_tensor(dates_year(samples()), to_tensor([2024i64, 1969i64, -44i64, 9999i64, -9999i64]), "years")
  _ = assert_eq_tensor(dates_month(samples()), to_tensor([2i64, 12i64, 3i64, 12i64, 1i64]), "months")
  _ = assert_eq_tensor(dates_day(samples()), to_tensor([29i64, 31i64, 15i64, 31i64, 1i64]), "days")
  _ = assert_eq_tensor(dates_weekday_iso_number(samples()), to_tensor([4i64, 3i64, 4i64, 5i64, 1i64]), "ISO weekday numbers")
  assert_eq_tensor(dates_day_of_year(samples()), to_tensor([60i64, 365i64, 75i64, 365i64, 1i64]), "days of the year")
}
def test_date_fields_of_an_empty_column() -> unit ! { Test } = {
  empty = dates_from_epoch_days(to_tensor(skip([0i64], 1i64)))
  _ = assert_eq(len(to_list(dates_year(empty))), 0i64, "no years")
  assert_eq(dates_to_strings(dates_from_epoch_days(to_tensor(skip([0i64], 1i64)))), skip(["x"], 1i64), "no text")
}
def test_dates_from_fields() -> unit ! { Test } = {
  ds = dates_from_ymd(to_tensor([2024i64, 1969i64, -44i64, 9999i64, -9999i64]), to_tensor([2i64, 12i64, 3i64, 12i64, 1i64]), to_tensor([29i64, 31i64, 15i64, 31i64, 1i64]))
  assert_eq_tensor(dates_epoch_days(ds), sample_days(), "fields to epoch days")
}
def test_dates_from_fields_masks_invalid_fields() -> unit ! { Test } = {
  (masked, valid) = try_dates_from_ymd(to_tensor([2024i64, 2023i64, 10000i64, 2024i64, 2024i64, -9223372036854775807i64]), to_tensor([2i64, 2i64, 1i64, 13i64, 4i64, 1i64]), to_tensor([29i64, 29i64, 1i64, 1i64, 31i64, 1i64]))
  _ = assert_eq_tensor(valid, to_tensor([true, false, false, false, false, false]), "only the first date exists")
  assert_eq_tensor(dates_epoch_days(masked), to_tensor([19782i64, 0i64, 0i64, 0i64, 0i64, 0i64]), "invalid positions hold the epoch")
}
def test_dates_add_days() -> unit ! { Test } = {
  moved = dates_add_days(samples(), to_tensor([1i64, 1i64, -1i64, 0i64, 7304483i64]))
  assert_eq(dates_to_strings(moved), ["2024-03-01", "1970-01-01", "-000044-03-14", "9999-12-31", "9999-12-31"], "days added")
}
def test_dates_add_days_reaches_both_edges() -> unit ! { Test } = {
  moved = dates_add_days(dates_from_epoch_days(to_tensor([2932895i64, -4371586i64])), to_tensor([1i64, -1i64]))
  assert_eq_tensor(dates_epoch_days(moved), to_tensor([2932896i64, -4371587i64]), "the last and first days, not past them")
}
def test_dates_add_months() -> unit ! { Test } = {
  starts = dates_from_epoch_days(to_tensor([19753i64, 19813i64, 0i64]))
  _ = assert_eq(dates_to_strings(dates_add_months(starts, to_tensor([1i64, -1i64, -1i64]), ClampToMonthEnd)), ["2024-02-29", "2024-02-29", "1969-12-01"], "clamped to the month's end")
  assert_eq(dates_to_strings(dates_add_months(dates_from_epoch_days(to_tensor([19753i64, 19813i64])), to_tensor([2i64, 12i64]), RejectInvalidDay)), ["2024-03-31", "2025-03-31"], "existing days are kept")
}
def test_dates_add_months_policies_differ_only_at_missing_days() -> unit ! { Test } = {
  clamped = dates_add_months(dates_from_epoch_days(to_tensor([19753i64, 19754i64])), to_tensor([1i64, 1i64]), ClampToMonthEnd)
  assert_eq(dates_to_strings(clamped), ["2024-02-29", "2024-03-01"], "a missing day clamps and an existing day stays")
}
def test_dates_days_until_and_order() -> unit ! { Test } = {
  later = dates_from_epoch_days(to_tensor([19782i64, 0i64, -735525i64, 2932895i64, 2932896i64]))
  _ = assert_eq_tensor(dates_days_until(samples(), later), to_tensor([0i64, 1i64, 0i64, -1i64, 7304483i64]), "days until")
  _ = assert_eq_tensor(dates_lt(samples(), dates_from_epoch_days(to_tensor([19782i64, 0i64, -735525i64, 2932895i64, 2932896i64]))), to_tensor([false, true, false, false, true]), "before")
  assert_eq_tensor(dates_lte(samples(), dates_from_epoch_days(to_tensor([19782i64, 0i64, -735525i64, 2932895i64, 2932896i64]))), to_tensor([true, true, true, false, true]), "on or before")
}
def test_dates_order_mirrors() -> unit ! { Test } = {
  _ = assert_eq_tensor(dates_gt(samples(), dates_from_epoch_days(to_tensor([19782i64, 0i64, -735525i64, 2932895i64, 2932896i64]))), to_tensor([false, false, false, true, false]), "after")
  assert_eq_tensor(dates_gte(samples(), dates_from_epoch_days(to_tensor([19782i64, 0i64, -735525i64, 2932895i64, 2932896i64]))), to_tensor([true, false, true, true, false]), "on or after")
}
def test_dates_text() -> unit ! { Test } = {
  _ = assert_eq(dates_to_strings(samples()), ["2024-02-29", "1969-12-31", "-000044-03-15", "9999-12-31", "-009999-01-01"], "canonical text")
  (parsed, valid) = try_parse_dates(["2024-02-29", "-000044-03-15", "9999-12-31"])
  _ = assert_eq_tensor(valid, to_tensor([true, true, true]), "every text parses")
  assert_eq_tensor(dates_epoch_days(parsed), to_tensor([19782i64, -735525i64, 2932896i64]), "parsed days")
}
def test_dates_text_masks_text_outside_the_profile() -> unit ! { Test } = {
  (parsed, valid) = try_parse_dates(["2023-02-29", "2024-2-29", "10000-01-01", "-000000-01-01", "2024-02-29T00:00:00", "", "2024-02-29"])
  _ = assert_eq_tensor(valid, to_tensor([false, false, false, false, false, false, true]), "only canonical dates parse")
  assert_eq_tensor(dates_epoch_days(parsed), to_tensor([0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 19782i64]), "invalid positions hold the epoch")
}
def test_durations() -> unit ! { Test } = {
  ds = durations(to_tensor([-1i64, 1i64, 0i64]), to_tensor([-500000000i64, 1500000000i64, 0i64]))
  _ = assert_eq_tensor(durations_seconds(ds), to_tensor([-2i64, 2i64, 0i64]), "normalized seconds")
  _ = assert_eq_tensor(durations_nanoseconds(durations(to_tensor([-1i64, 1i64, 0i64]), to_tensor([-500000000i64, 1500000000i64, 0i64]))), to_tensor([500000000i64, 500000000i64, 0i64]), "normalized nanoseconds")
  edge = durations(to_tensor([-9223372036854775807i64]), to_tensor([-1000000000i64]))
  assert_eq_tensor(durations_seconds(edge), to_tensor([sub(-9223372036854775807i64, 1i64)]), "the most negative second")
}
def test_durations_mask_values_outside_the_type() -> unit ! { Test } = {
  (masked, valid) = try_durations(to_tensor([9223372036854775807i64, 5i64, sub(-9223372036854775807i64, 1i64)]), to_tensor([1000000000i64, -1i64, -1i64]))
  _ = assert_eq_tensor(valid, to_tensor([false, true, false]), "past either end of i64 seconds")
  assert_eq_tensor(durations_seconds(masked), to_tensor([0i64, 4i64, 0i64]), "invalid positions hold zero")
}
def test_instant_counts() -> unit ! { Test } = {
  from_ms = instants_from_unix_count(to_tensor([-1500i64, 1i64]), Milliseconds)
  _ = assert_eq_tensor(instants_unix_seconds(from_ms), to_tensor([-2i64, 0i64]), "millisecond seconds")
  _ = assert_eq_tensor(instants_nanoseconds(instants_from_unix_count(to_tensor([-1500i64, 1i64]), Milliseconds)), to_tensor([500000000i64, 1000000i64]), "millisecond fractions")
  _ = assert_eq_tensor(instants_unix_seconds(instants_from_unix_count(to_tensor([2i64, -1i64]), Hours)), to_tensor([7200i64, -3600i64]), "hours")
  _ = assert_eq_tensor(instants_to_unix_count(stamps(), Seconds, RoundTiesToEven), to_tensor([0i64, 0i64, 1700000000i64, 253402214401i64]), "seconds, ties to even")
  _ = assert_eq_tensor(instants_to_unix_count(stamps(), Minutes, RoundTowardNegative), to_tensor([0i64, -1i64, 28333333i64, 4223370240i64]), "minutes toward negative")
  _ = assert_eq_tensor(instants_to_unix_count(stamps(), Microseconds, RoundTowardPositive), to_tensor([0i64, 0i64, 1700000000500000i64, 253402214401000000i64]), "microseconds toward positive")
  assert_eq_tensor(instants_to_unix_count(instants_from_unix(to_tensor([-1i64, 1i64]), to_tensor([999999999i64, 0i64])), Nanoseconds, RejectInexact), to_tensor([-1i64, 1000000000i64]), "exact nanoseconds")
}
def test_instant_count_rounding_modes_differ() -> unit ! { Test } = {
  halves = instants_from_unix(to_tensor([-2i64, 2i64]), to_tensor([500000000i64, 500000000i64]))
  _ = assert_eq_tensor(instants_to_unix_count(halves, Seconds, RoundTowardZero), to_tensor([-1i64, 2i64]), "toward zero")
  _ = assert_eq_tensor(instants_to_unix_count(instants_from_unix(to_tensor([-2i64, 2i64]), to_tensor([500000000i64, 500000000i64])), Seconds, RoundAwayFromZero), to_tensor([-2i64, 3i64]), "away from zero")
  assert_eq_tensor(instants_to_unix_count(instants_from_unix(to_tensor([-2i64, 2i64]), to_tensor([500000000i64, 500000000i64])), Seconds, RoundTiesToAway), to_tensor([-2i64, 3i64]), "ties away")
}
def test_instant_arithmetic() -> unit ! { Test } = {
  shifted = instants_add_duration(stamps(), durations(to_tensor([1i64, 0i64, -1700000000i64, 0i64]), to_tensor([0i64, 1i64, 500000000i64, 0i64])))
  _ = assert_eq_tensor(instants_unix_seconds(shifted), to_tensor([1i64, 0i64, 1i64, 253402214400i64]), "seconds after adding")
  gaps = instants_until(instants_from_unix(to_tensor([0i64, 0i64, 0i64, 0i64]), to_tensor([1i64, 1i64, 1i64, 1i64])), stamps())
  _ = assert_eq_tensor(durations_seconds(gaps), to_tensor([-1i64, -1i64, 1700000000i64, 253402214400i64]), "gap seconds")
  assert_eq_tensor(durations_nanoseconds(instants_until(instants_from_unix(to_tensor([0i64, 0i64, 0i64, 0i64]), to_tensor([1i64, 1i64, 1i64, 1i64])), stamps())), to_tensor([999999999i64, 999999998i64, 499999999i64, 999999998i64]), "gap nanoseconds")
}
def test_instant_arithmetic_round_trips() -> unit ! { Test } = {
  origin = instants_from_unix(to_tensor([5i64, -5i64]), to_tensor([7i64, 999999999i64]))
  later = instants_from_unix(to_tensor([-3i64, 9i64]), to_tensor([0i64, 1i64]))
  back = instants_add_duration(origin, instants_until(instants_from_unix(to_tensor([5i64, -5i64]), to_tensor([7i64, 999999999i64])), later))
  assert_eq_tensor(instants_nanoseconds(back), to_tensor([0i64, 1i64]), "adding the gap returns to the later instant")
}
def test_instant_rounding_and_dates() -> unit ! { Test } = {
  quarter = instants_round_to(instants_from_unix(to_tensor([1700000123i64, -1i64]), to_tensor([456i64, 0i64])), duration(900i64, 0i64), RoundTiesToEven)
  _ = assert_eq_tensor(instants_unix_seconds(quarter), to_tensor([1700000100i64, 0i64]), "quarter-hour buckets")
  local = instants_to_dates_at(instants_from_unix(to_tensor([1727775000i64, -1i64]), to_tensor([5i64, 0i64])), offset_from_seconds(19800i64))
  _ = assert_eq_tensor(dates_epoch_days(local), to_tensor([19997i64, 0i64]), "dates at +05:30")
  assert_eq_tensor(dates_epoch_days(instants_to_dates_at(instants_from_unix(to_tensor([-1i64]), to_tensor([0i64])), offset_from_seconds(0i64))), to_tensor([-1i64]), "the last second of 1969 at UTC")
}
def test_instant_rounding_keeps_multiples() -> unit ! { Test } = {
  exact = instants_round_to(instants_from_unix(to_tensor([900i64, -900i64]), to_tensor([0i64, 0i64])), duration(900i64, 0i64), RejectInexact)
  assert_eq_tensor(instants_unix_seconds(exact), to_tensor([900i64, -900i64]), "multiples are already rounded")
}
def test_seconds_since() -> unit ! { Test } = {
  axis = instants_seconds_since_f64(stamps(), instant_from_unix(0i64, 1i64))
  assert_eq(to_list(axis), [-1e-9f64, -2e-9f64, 1700000000.5f64, 253402214401.0f64], "seconds since the origin")
}
def test_seconds_since_keeps_small_negative_gaps() -> unit ! { Test } = {
  axis = instants_seconds_since_f64(instants_from_unix(to_tensor([-1i64, 0i64]), to_tensor([999999000i64, 0i64])), instant_from_unix(0i64, 0i64))
  assert_eq(to_list(axis), [-1e-6f64, 0.0f64], "a microsecond before the origin, and the origin")
}
def test_instant_order() -> unit ! { Test } = {
  _ = assert_eq_tensor(instants_lt(stamps(), instants_from_unix(to_tensor([0i64, 0i64, 1700000000i64, 0i64]), to_tensor([0i64, 0i64, 500000001i64, 0i64]))), to_tensor([false, true, true, false]), "before")
  _ = assert_eq_tensor(instants_lte(stamps(), instants_from_unix(to_tensor([0i64, 0i64, 1700000000i64, 0i64]), to_tensor([0i64, 0i64, 500000001i64, 0i64]))), to_tensor([true, true, true, false]), "on or before")
  _ = assert_eq_tensor(instants_gt(stamps(), instants_from_unix(to_tensor([0i64, 0i64, 1700000000i64, 0i64]), to_tensor([0i64, 0i64, 500000001i64, 0i64]))), to_tensor([false, false, false, true]), "after")
  assert_eq_tensor(instants_gte(stamps(), instants_from_unix(to_tensor([0i64, 0i64, 1700000000i64, 0i64]), to_tensor([0i64, 0i64, 500000001i64, 0i64]))), to_tensor([true, false, false, true]), "on or after")
}
def test_instant_order_reads_nanoseconds() -> unit ! { Test } = {
  first = instants_from_unix(to_tensor([7i64]), to_tensor([1i64]))
  assert_eq_tensor(instants_lt(first, instants_from_unix(to_tensor([7i64]), to_tensor([0i64]))), to_tensor([false]), "the same second with a later nanosecond is not before")
}
