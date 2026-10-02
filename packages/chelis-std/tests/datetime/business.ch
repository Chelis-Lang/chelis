module Std.Tests.Datetime.Business
import Std.Datetime (Date, Dates, parse_date, date_to_string, date_epoch_day, dates_from_epoch_days, dates_epoch_days)
import Std.Datetime.Business (Weekmask, BusinessCalendar, BusinessDayRoll, Unadjusted, Following, Preceding, ModifiedFollowing, ModifiedPreceding, NonBusinessStart, RejectNonBusinessStart, RollStartForward, RollStartBackward, business_calendar, try_business_calendar, business_calendar_weekmask, business_calendar_holidays, business_calendar_valid_from, business_calendar_valid_until, is_business_day, try_is_business_day, business_day_roll, try_business_day_roll, business_day_offset, try_business_day_offset, business_day_count, try_business_day_count, business_in_all, try_business_in_all, business_in_any, try_business_in_any, dates_is_business_day, dates_business_day_roll, dates_business_day_offset, dates_business_day_count)
import Std.Test (assert_eq, assert_true, assert_false, assert_eq_tensor)
-- Expected values come from NumPy's busday functions over the same weekmask
-- and holidays; see [05-OP-73].
def weekdays() -> Weekmask = Weekmask { monday: true, tuesday: true, wednesday: true, thursday: true, friday: true, saturday: false, sunday: false }
def six_days() -> Weekmask = Weekmask { monday: true, tuesday: true, wednesday: true, thursday: true, friday: true, saturday: true, sunday: false }
def weekend() -> Weekmask = Weekmask { monday: false, tuesday: false, wednesday: false, thursday: false, friday: false, saturday: true, sunday: true }
def no_days() -> Weekmask = Weekmask { monday: false, tuesday: false, wednesday: false, thursday: false, friday: false, saturday: false, sunday: false }
def day(text: string) -> Date = parse_date(text)
-- Holidays out of order, with a duplicate and a Saturday that normalization drops.
def year_2026() -> BusinessCalendar = business_calendar(weekdays(), [day("2026-12-25"), day("2026-01-01"), day("2026-07-04"), day("2026-01-19"), day("2026-05-25"), day("2026-07-03"), day("2026-01-01")], day("2026-01-01"), day("2026-12-31"))
def texts(ds: List[Date]) -> List[string] = map(fn (d: Date) -> date_to_string(d), ds)
def option_text(value: Option[Date]) -> string =
  match value with {
    | Some(d) => date_to_string(d)
    | None => "none"
  }
def column[n](ds: List[Date]) -> Dates[n] = dates_from_epoch_days(to_tensor(map(fn (d: Date) -> date_epoch_day(d), ds)))
def test_calendar_normalizes_holidays() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_eq(texts(business_calendar_holidays(cal)), ["2026-01-01", "2026-01-19", "2026-05-25", "2026-07-03", "2026-12-25"], "sorted, unique, weekend holiday dropped")
  _ = assert_eq(business_calendar_weekmask(cal), weekdays(), "weekmask")
  _ = assert_eq(date_to_string(business_calendar_valid_from(cal)), "2026-01-01", "horizon start")
  assert_eq(date_to_string(business_calendar_valid_until(cal)), "2026-12-31", "horizon end")
}
def test_calendar_rejects_invalid_inputs() -> unit ! { Test } = {
  _ = assert_eq(try_business_calendar(no_days(), [], day("2026-01-01"), day("2026-12-31")), None, "a weekmask with no business day")
  _ = assert_eq(try_business_calendar(weekdays(), [], day("2026-12-31"), day("2026-01-01")), None, "an inverted horizon")
  _ = assert_eq(try_business_calendar(weekdays(), [day("2027-01-01")], day("2026-01-01"), day("2026-12-31")), None, "a holiday after the horizon")
  assert_eq(try_business_calendar(weekdays(), [day("2025-12-31")], day("2026-01-01"), day("2026-12-31")), None, "a holiday before the horizon")
}
def test_calendar_equality_is_normalized() -> unit ! { Test } = {
  same = business_calendar(weekdays(), [day("2026-01-19"), day("2026-07-03"), day("2026-01-01"), day("2026-05-25"), day("2026-12-25")], day("2026-01-01"), day("2026-12-31"))
  _ = assert_true(eq(year_2026(), same), "the same holidays in another order")
  assert_true(eq(try_business_calendar(weekdays(), [day("2026-01-01")], day("2026-01-01"), day("2026-01-01")), Some(business_calendar(weekdays(), [day("2026-01-01"), day("2026-01-01")], day("2026-01-01"), day("2026-01-01")))), "a duplicate holiday")
}
def test_calendar_equality_distinguishes_calendars() -> unit ! { Test } = {
  fewer = business_calendar(weekdays(), [day("2026-01-01")], day("2026-01-01"), day("2026-12-31"))
  _ = assert_false(eq(year_2026(), fewer), "different holidays")
  assert_true(neq(fewer, business_calendar(weekdays(), [day("2026-01-01")], day("2026-01-01"), day("2026-12-30"))), "different horizons")
}
def test_is_business_day() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_true(is_business_day(cal, day("2026-01-02")), "a Friday")
  _ = assert_true(is_business_day(cal, day("2026-12-31")), "the last day of the horizon")
  assert_eq(try_is_business_day(cal, day("2026-01-05")), Some(true), "try form inside the horizon")
}
def test_is_business_day_rejects() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_false(is_business_day(cal, day("2026-01-03")), "a Saturday")
  _ = assert_false(is_business_day(cal, day("2026-01-19")), "a holiday")
  _ = assert_eq(try_is_business_day(cal, day("2027-01-01")), None, "after the horizon")
  assert_eq(try_is_business_day(cal, day("2025-12-31")), None, "before the horizon")
}
def test_roll_following_and_preceding() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_eq(date_to_string(business_day_roll(cal, day("2026-01-03"), Following)), "2026-01-05", "Saturday to Monday")
  _ = assert_eq(date_to_string(business_day_roll(cal, day("2026-01-03"), Preceding)), "2026-01-02", "Saturday to Friday")
  _ = assert_eq(date_to_string(business_day_roll(cal, day("2026-01-01"), Following)), "2026-01-02", "past a holiday")
  _ = assert_eq(date_to_string(business_day_roll(cal, day("2026-01-02"), Preceding)), "2026-01-02", "a business day stays")
  assert_eq(date_to_string(business_day_roll(cal, day("2026-01-03"), Unadjusted)), "2026-01-03", "unadjusted")
}
def test_roll_rejects_outside_the_horizon() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_eq(option_text(try_business_day_roll(cal, day("2026-01-01"), Preceding)), "none", "no business day before 2026-01-01 inside the horizon")
  _ = assert_eq(option_text(try_business_day_roll(cal, day("2027-01-01"), Unadjusted)), "none", "unadjusted outside the horizon")
  short = business_calendar(weekdays(), [], day("2026-12-01"), day("2026-12-26"))
  assert_eq(option_text(try_business_day_roll(short, day("2026-12-26"), Following)), "none", "the following day lies after the horizon")
}
def test_roll_modified_conventions() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_eq(date_to_string(business_day_roll(cal, day("2026-05-30"), ModifiedFollowing)), "2026-05-29", "following would leave May")
  _ = assert_eq(date_to_string(business_day_roll(cal, day("2026-01-03"), ModifiedFollowing)), "2026-01-05", "following stays in January")
  _ = assert_eq(date_to_string(business_day_roll(cal, day("2026-03-01"), ModifiedPreceding)), "2026-03-02", "preceding would leave March")
  month_end = business_calendar(weekdays(), [], day("2026-05-01"), day("2026-05-31"))
  assert_eq(date_to_string(business_day_roll(month_end, day("2026-05-30"), ModifiedFollowing)), "2026-05-29", "a horizon ending with its month decides the roll")
}
def test_roll_modified_conventions_reject_undetermined_answers() -> unit ! { Test } = {
  mid_month = business_calendar(weekdays(), [], day("2026-05-01"), day("2026-05-23"))
  _ = assert_eq(option_text(try_business_day_roll(mid_month, day("2026-05-23"), ModifiedFollowing)), "none", "the following day could fall in May after the horizon")
  late_start = business_calendar(weekdays(), [], day("2026-03-08"), day("2026-03-31"))
  assert_eq(option_text(try_business_day_roll(late_start, day("2026-03-08"), ModifiedPreceding)), "none", "the preceding day could fall in March before the horizon")
}
def test_offset_rolls_then_moves() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_eq(date_to_string(business_day_offset(cal, day("2026-01-03"), 2i64, RollStartForward)), "2026-01-07", "Saturday forward plus two")
  _ = assert_eq(date_to_string(business_day_offset(cal, day("2026-01-03"), 2i64, RollStartBackward)), "2026-01-06", "Saturday backward plus two")
  _ = assert_eq(date_to_string(business_day_offset(cal, day("2026-01-03"), 0i64, RollStartForward)), "2026-01-05", "zero returns the rolled start")
  _ = assert_eq(date_to_string(business_day_offset(cal, day("2026-01-20"), -1i64, RejectNonBusinessStart)), "2026-01-16", "back over a holiday")
  assert_eq(date_to_string(business_day_offset(cal, day("2026-12-30"), 1i64, RejectNonBusinessStart)), "2026-12-31", "to the horizon end")
}
def test_offset_rejects() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_eq(option_text(try_business_day_offset(cal, day("2026-01-03"), 1i64, RejectNonBusinessStart)), "none", "a non-business start under RejectNonBusinessStart")
  _ = assert_eq(option_text(try_business_day_offset(cal, day("2026-12-31"), 1i64, RejectNonBusinessStart)), "none", "past the horizon end")
  _ = assert_eq(option_text(try_business_day_offset(cal, day("2026-01-02"), -1i64, RejectNonBusinessStart)), "none", "before the horizon start")
  _ = assert_eq(option_text(try_business_day_offset(cal, day("2026-01-02"), 9223372036854775807i64, RejectNonBusinessStart)), "none", "the largest offset")
  assert_eq(option_text(try_business_day_offset(cal, day("2027-01-04"), 0i64, RollStartForward)), "none", "a start outside the horizon")
}
def test_count_half_open_and_antisymmetric() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_eq(business_day_count(cal, day("2026-01-01"), day("2026-02-01")), 20i64, "January")
  _ = assert_eq(business_day_count(cal, day("2026-02-01"), day("2026-01-01")), -20i64, "January reversed")
  _ = assert_eq(business_day_count(cal, day("2026-01-02"), day("2026-01-02")), 0i64, "an empty range")
  assert_eq(try_business_day_count(cal, day("2026-01-01"), day("2027-01-01")), Some(256i64), "the end may be the day after the horizon")
}
def test_count_rejects() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_eq(try_business_day_count(cal, day("2026-01-01"), day("2027-01-02")), None, "an end two days after the horizon")
  _ = assert_eq(try_business_day_count(cal, day("2025-12-31"), day("2026-01-05")), None, "a begin before the horizon")
  assert_eq(try_business_day_count(cal, day("2027-01-01"), day("2026-01-05")), Some(-255i64), "a reversed range from the day after the horizon")
}
def test_business_in_all_and_any() -> unit ! { Test } = {
  shifted = business_calendar(six_days(), [day("2026-07-03"), day("2026-07-04"), day("2026-08-03")], day("2026-06-01"), day("2027-03-31"))
  both = business_in_all(year_2026(), shifted)
  _ = assert_true(eq(both, business_calendar(weekdays(), [day("2026-07-03"), day("2026-08-03"), day("2026-12-25")], day("2026-06-01"), day("2026-12-31"))), "weekmasks intersect and holidays unite")
  either = business_in_any(year_2026(), shifted)
  _ = assert_true(eq(either, business_calendar(six_days(), [day("2026-07-03"), day("2026-07-04")], day("2026-06-01"), day("2026-12-31"))), "weekmasks unite and shared closures remain")
  _ = assert_true(is_business_day(either, day("2026-12-25")), "open in the second calendar")
  assert_false(is_business_day(both, day("2026-08-03")), "closed in the second calendar")
}
def test_business_in_all_and_any_reject() -> unit ! { Test } = {
  later = business_calendar(weekdays(), [], day("2027-01-01"), day("2027-12-31"))
  _ = assert_eq(try_business_in_all(year_2026(), later), None, "disjoint horizons")
  _ = assert_eq(try_business_in_any(year_2026(), later), None, "disjoint horizons")
  saturdays = business_calendar(weekend(), [], day("2026-01-01"), day("2026-12-31"))
  _ = assert_eq(try_business_in_all(year_2026(), saturdays), None, "no weekday in both weekmasks")
  assert_true(is_business_day(business_in_any(year_2026(), saturdays), day("2026-01-03")), "either weekmask suffices")
}
def test_vectorized_forms_agree_with_scalar_forms() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_eq_tensor(dates_is_business_day(cal, column([day("2026-01-02"), day("2026-01-03"), day("2026-01-19")])), to_tensor([true, false, false]), "flags")
  _ = assert_eq_tensor(dates_epoch_days(dates_business_day_roll(cal, column([day("2026-01-03"), day("2026-05-30"), day("2026-03-01")]), ModifiedFollowing)), to_tensor([date_epoch_day(day("2026-01-05")), date_epoch_day(day("2026-05-29")), date_epoch_day(day("2026-03-02"))]), "modified following")
  _ = assert_eq_tensor(dates_epoch_days(dates_business_day_offset(cal, column([day("2026-01-03"), day("2026-01-20")]), to_tensor([2i64, -1i64]), RollStartForward)), to_tensor([date_epoch_day(day("2026-01-07")), date_epoch_day(day("2026-01-16"))]), "offsets")
  assert_eq_tensor(dates_business_day_count(cal, column([day("2026-01-01"), day("2026-02-01")]), column([day("2026-02-01"), day("2026-01-01")])), to_tensor([20i64, -20i64]), "counts")
}
def test_vectorized_forms_report_non_business_days() -> unit ! { Test } = {
  cal = year_2026()
  _ = assert_eq_tensor(dates_is_business_day(cal, column([day("2026-12-25"), day("2026-07-04")])), to_tensor([false, false]), "a holiday and a Saturday")
  _ = assert_eq_tensor(dates_epoch_days(dates_business_day_roll(cal, column([day("2026-12-26")]), Preceding)), to_tensor([date_epoch_day(day("2026-12-24"))]), "back over a weekend and a holiday")
  assert_eq_tensor(dates_business_day_count(cal, column([day("2026-01-03")]), column([day("2026-01-05")])), to_tensor([0i64]), "a weekend counts nothing")
}
