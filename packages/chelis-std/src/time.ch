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
  | Date { year: int64, month: int64, day: int64 }
type Duration =
  | Duration { days: int64, hours: int64, minutes: int64, seconds: int64 }
def date(year: int64, month: int64, day: int64) -> Date = { match try_date(year, month, day) with {
  | Some(value) => value
  | None => fail("date: invalid calendar date")
} }
def try_date(year: int64, month: int64, day: int64) -> Option[Date] = { if gt(cast(1, int64), month) then None else if gt(month, cast(12, int64)) then None else if gt(cast(1, int64), day) then None else if gt(day, days_in_month(year, month)) then None else Some(Date { year: year, month: month, day: day }) }
def duration(days: int64, hours: int64, minutes: int64, seconds: int64) -> Duration = Duration { days: days, hours: hours, minutes: minutes, seconds: seconds }
def is_leap_year(year: int64) -> bool = or(and(eq(mod(year, cast(4, int64)), cast(0, int64)), neq(mod(year, cast(100, int64)), cast(0, int64))), eq(mod(year, cast(400, int64)), cast(0, int64)))
def add_days(value: Date, delta: int64) -> Date = date_from_ordinal(add(date_to_ordinal(value), delta))
def sub_days(value: Date, delta: int64) -> Date = add_days(value, sub(cast(0, int64), delta))
def days_between(lhs: Date, rhs: Date) -> int64 = sub(date_to_ordinal(rhs), date_to_ordinal(lhs))
def date_lt(lhs: Date, rhs: Date) -> bool = lt(date_to_ordinal(lhs), date_to_ordinal(rhs))
def date_lte(lhs: Date, rhs: Date) -> bool = lte(date_to_ordinal(lhs), date_to_ordinal(rhs))
def date_gt(lhs: Date, rhs: Date) -> bool = gt(date_to_ordinal(lhs), date_to_ordinal(rhs))
def date_gte(lhs: Date, rhs: Date) -> bool = gte(date_to_ordinal(lhs), date_to_ordinal(rhs))
def date_to_string(value: Date) -> string = string_concat(pad_left(value.year, cast(4, int64)), string_concat("-", string_concat(pad_left(value.month, cast(2, int64)), string_concat("-", pad_left(value.day, cast(2, int64))))))
def parse_date(text: string) -> Option[Date] = { if neq(string_len(text), cast(10, int64)) then None else if neq(char_at(text, cast(4, int64)), "-") then None else if neq(char_at(text, cast(7, int64)), "-") then None else match to_int(string_slice(text, cast(0, int64), cast(4, int64))) with {
  | Some(year) => match to_int(string_slice(text, cast(5, int64), cast(2, int64))) with {
  | Some(month) => match to_int(string_slice(text, cast(8, int64), cast(2, int64))) with {
  | Some(day) => try_date(year, month, day)
  | None => None
}
  | None => None
}
  | None => None
} }
def day_of_week(value: Date) -> DayOfWeek = {
  raw = mod(add(date_to_ordinal(value), cast(3, int64)), cast(7, int64))
  idx = if gt(cast(0, int64), raw) then add(raw, cast(7, int64)) else raw
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
def day_of_week_name(value: Date) -> string = { match day_of_week(value) with {
  | Monday => "monday"
  | Tuesday => "tuesday"
  | Wednesday => "wednesday"
  | Thursday => "thursday"
  | Friday => "friday"
  | Saturday => "saturday"
  | Sunday => "sunday"
} }
def day_of_year(value: Date) -> int64 = add(days_before_month(value.year, value.month), value.day)
def days_in_month(year: int64, month: int64) -> int64 = { match month with {
  | 1 => cast(31, int64)
  | 2 => if is_leap_year(year) then cast(29, int64) else cast(28, int64)
  | 3 => cast(31, int64)
  | 4 => cast(30, int64)
  | 5 => cast(31, int64)
  | 6 => cast(30, int64)
  | 7 => cast(31, int64)
  | 8 => cast(31, int64)
  | 9 => cast(30, int64)
  | 10 => cast(31, int64)
  | 11 => cast(30, int64)
  | _ => cast(31, int64)
} }
def days_before_month(year: int64, month: int64) -> int64 = { days_before_month_loop(year, cast(1, int64), month, cast(0, int64)) }
def days_before_month_loop(year: int64, cursor: int64, limit: int64, acc: int64) -> int64 = { if gte(cursor, limit) then acc else days_before_month_loop(year, add(cursor, cast(1, int64)), limit, add(acc, days_in_month(year, cursor))) }
def date_to_ordinal(value: Date) -> int64 = { add(days_before_year(value.year), sub(day_of_year(value), cast(1, int64))) }
def days_before_year(year: int64) -> int64 = { if eq(year, cast(1970, int64)) then cast(0, int64) else if gt(year, cast(1970, int64)) then days_before_year_forward(cast(1970, int64), year, cast(0, int64)) else sub(cast(0, int64), days_before_year_forward(year, cast(1970, int64), cast(0, int64))) }
def days_before_year_forward(cursor: int64, limit: int64, acc: int64) -> int64 = { if gte(cursor, limit) then acc else days_before_year_forward(add(cursor, cast(1, int64)), limit, add(acc, year_days(cursor))) }
def year_days(year: int64) -> int64 = { if is_leap_year(year) then cast(366, int64) else cast(365, int64) }
def date_from_ordinal(days: int64) -> Date = { if gte(days, cast(0, int64)) then date_from_ordinal_forward(cast(1970, int64), days) else date_from_ordinal_backward(cast(1969, int64), days) }
def date_from_ordinal_forward(year: int64, remaining: int64) -> Date = {
  span = year_days(year)
  if gt(span, remaining) then date_from_year_offset(year, remaining) else date_from_ordinal_forward(add(year, cast(1, int64)), sub(remaining, span))
}
def date_from_ordinal_backward(year: int64, remaining: int64) -> Date = {
  span = year_days(year)
  shifted = add(remaining, span)
  if gte(shifted, cast(0, int64)) then date_from_year_offset(year, shifted) else date_from_ordinal_backward(sub(year, cast(1, int64)), shifted)
}
def date_from_year_offset(year: int64, offset: int64) -> Date = { date_from_year_offset_loop(year, cast(1, int64), offset) }
def date_from_year_offset_loop(year: int64, month: int64, offset: int64) -> Date = {
  span = days_in_month(year, month)
  if gt(span, offset) then Date { year: year, month: month, day: add(offset, cast(1, int64)) } else date_from_year_offset_loop(year, add(month, cast(1, int64)), sub(offset, span))
}
def pad_left(value: int64, width: int64) -> string = {
  text = to_string(value)
  if gte(string_len(text), width) then text else string_concat("0", pad_left(value, sub(width, cast(1, int64))))
}
def char_at(text: string, idx: int64) -> string = { string_slice(text, idx, cast(1, int64)) }
