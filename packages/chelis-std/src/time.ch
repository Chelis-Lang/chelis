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
def date(year: i64, month: i64, day: i64) -> Date = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def try_date(year: i64, month: i64, day: i64) -> Option[Date] = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def duration(days: i64, hours: i64, minutes: i64, seconds: i64) -> Duration = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def is_leap_year(year: i64) -> bool = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def add_days(value: Date, delta: i64) -> Date = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def sub_days(value: Date, delta: i64) -> Date = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def days_between(lhs: Date, rhs: Date) -> i64 = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def date_lt(lhs: Date, rhs: Date) -> bool = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def date_lte(lhs: Date, rhs: Date) -> bool = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def date_gt(lhs: Date, rhs: Date) -> bool = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def date_gte(lhs: Date, rhs: Date) -> bool = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def date_to_string(value: Date) -> string = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def parse_date(text: string) -> Option[Date] = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def day_of_week(value: Date) -> DayOfWeek = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def day_of_week_name(value: Date) -> string = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
def day_of_year(value: Date) -> i64 = fail("Std.Time is unavailable: exact Gregorian and duration arithmetic is not implemented (#2779)")
