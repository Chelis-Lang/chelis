module Std.Time
export (DayOfWeek, Date, Duration, date, try_date, duration, is_leap_year, add_days, sub_days, days_between, date_lt, date_lte, date_gt, date_gte, date_to_string, parse_date, day_of_week, day_of_week_name, day_of_year)
type DayOfWeek =
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
type Date =
  | Date { year: i64, month: i64, day: i64 }
type Duration =
  | Duration { days: i64, hours: i64, minutes: i64, seconds: i64 }
def date(year: i64, month: i64, day: i64) -> Date =
  match try_date(year, month, day) with {
    | Some(value) => value
    | None => fail("date: invalid calendar date")
  }
def try_date(year: i64, month: i64, day: i64) -> Option[Date] = if gt(cast(1, i64), month) then None else if gt(month, cast(12, i64)) then None else if gt(cast(1, i64), day) then None else if gt(day, days_in_month(year, month)) then None else Some(Date { year, month, day })
def duration(days: i64, hours: i64, minutes: i64, seconds: i64) -> Duration = Duration { days, hours, minutes, seconds }
def is_leap_year(year: i64) -> bool = or(and(eq(mod(year, cast(4, i64)), cast(0, i64)), neq(mod(year, cast(100, i64)), cast(0, i64))), eq(mod(year, cast(400, i64)), cast(0, i64)))
def add_days(value: Date, delta: i64) -> Date = date_from_ordinal(add(date_to_ordinal(value), delta))
def sub_days(value: Date, delta: i64) -> Date = add_days(value, sub(cast(0, i64), delta))
def days_between(lhs: Date, rhs: Date) -> i64 = sub(date_to_ordinal(rhs), date_to_ordinal(lhs))
def date_lt(lhs: Date, rhs: Date) -> bool = lt(date_to_ordinal(lhs), date_to_ordinal(rhs))
def date_lte(lhs: Date, rhs: Date) -> bool = lte(date_to_ordinal(lhs), date_to_ordinal(rhs))
def date_gt(lhs: Date, rhs: Date) -> bool = gt(date_to_ordinal(lhs), date_to_ordinal(rhs))
def date_gte(lhs: Date, rhs: Date) -> bool = gte(date_to_ordinal(lhs), date_to_ordinal(rhs))
def date_to_string(value: Date) -> string = string_concat(pad_left(value.year, cast(4, i64)), string_concat("-", string_concat(pad_left(value.month, cast(2, i64)), string_concat("-", pad_left(value.day, cast(2, i64))))))
def parse_date(text: string) -> Option[Date] =
  if neq(string_len(text), cast(10, i64)) then None else if neq(char_at(text, cast(4, i64)), "-") then None else if neq(char_at(text, cast(7, i64)), "-") then None else match to_int(string_slice(text, cast(0, i64), cast(4, i64))) with {
    | Some(year) => match to_int(string_slice(text, cast(5, i64), cast(2, i64))) with {
    | Some(month) => match to_int(string_slice(text, cast(8, i64), cast(2, i64))) with {
    | Some(day) => try_date(year, month, day)
    | None => None
  }
    | None => None
  }
    | None => None
  }
def day_of_week(value: Date) -> DayOfWeek = {
  raw = mod(add(date_to_ordinal(value), cast(3, i64)), cast(7, i64))
  idx = if gt(cast(0, i64), raw) then add(raw, cast(7, i64)) else raw
  match idx with {
    | 0 => Monday
    | 1 => Tuesday
    | 2 => Wednesday
    | 3 => Thursday
    | 4 => Friday
    | 5 => Saturday
    | _ => Sunday
  }
}
def day_of_week_name(value: Date) -> string =
  match day_of_week(value) with {
    | Monday => "monday"
    | Tuesday => "tuesday"
    | Wednesday => "wednesday"
    | Thursday => "thursday"
    | Friday => "friday"
    | Saturday => "saturday"
    | Sunday => "sunday"
  }
def day_of_year(value: Date) -> i64 = add(days_before_month(value.year, value.month), value.day)
def days_in_month(year: i64, month: i64) -> i64 =
  match month with {
    | 1 => cast(31, i64)
    | 2 => if is_leap_year(year) then cast(29, i64) else cast(28, i64)
    | 3 => cast(31, i64)
    | 4 => cast(30, i64)
    | 5 => cast(31, i64)
    | 6 => cast(30, i64)
    | 7 => cast(31, i64)
    | 8 => cast(31, i64)
    | 9 => cast(30, i64)
    | 10 => cast(31, i64)
    | 11 => cast(30, i64)
    | _ => cast(31, i64)
  }
def days_before_month(year: i64, month: i64) -> i64 = days_before_month_loop(year, cast(1, i64), month, cast(0, i64))
def days_before_month_loop(year: i64, cursor: i64, limit: i64, acc: i64) -> i64 = if gte(cursor, limit) then acc else days_before_month_loop(year, add(cursor, cast(1, i64)), limit, add(acc, days_in_month(year, cursor)))
def date_to_ordinal(value: Date) -> i64 = add(days_before_year(value.year), sub(day_of_year(value), cast(1, i64)))
def days_before_year(year: i64) -> i64 = if eq(year, cast(1970, i64)) then cast(0, i64) else if gt(year, cast(1970, i64)) then days_before_year_forward(cast(1970, i64), year, cast(0, i64)) else sub(cast(0, i64), days_before_year_forward(year, cast(1970, i64), cast(0, i64)))
def days_before_year_forward(cursor: i64, limit: i64, acc: i64) -> i64 = if gte(cursor, limit) then acc else days_before_year_forward(add(cursor, cast(1, i64)), limit, add(acc, year_days(cursor)))
def year_days(year: i64) -> i64 = if is_leap_year(year) then cast(366, i64) else cast(365, i64)
def date_from_ordinal(days: i64) -> Date = if gte(days, cast(0, i64)) then date_from_ordinal_forward(cast(1970, i64), days) else date_from_ordinal_backward(cast(1969, i64), days)
def date_from_ordinal_forward(year: i64, remaining: i64) -> Date = {
  span = year_days(year)
  if gt(span, remaining) then date_from_year_offset(year, remaining) else date_from_ordinal_forward(add(year, cast(1, i64)), sub(remaining, span))
}
def date_from_ordinal_backward(year: i64, remaining: i64) -> Date = {
  span = year_days(year)
  shifted = add(remaining, span)
  if gte(shifted, cast(0, i64)) then date_from_year_offset(year, shifted) else date_from_ordinal_backward(sub(year, cast(1, i64)), shifted)
}
def date_from_year_offset(year: i64, offset: i64) -> Date = date_from_year_offset_loop(year, cast(1, i64), offset)
def date_from_year_offset_loop(year: i64, month: i64, offset: i64) -> Date = {
  span = days_in_month(year, month)
  if gt(span, offset) then Date { year, month, day: add(offset, cast(1, i64)) } else date_from_year_offset_loop(year, add(month, cast(1, i64)), sub(offset, span))
}
def pad_left(value: i64, width: i64) -> string = {
  text = to_string(value)
  if gte(string_len(text), width) then text else string_concat("0", pad_left(value, sub(width, cast(1, i64))))
}
def char_at(text: string, idx: i64) -> string = string_slice(text, idx, cast(1, i64))
