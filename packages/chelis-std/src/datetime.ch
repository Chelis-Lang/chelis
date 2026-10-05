module Std.Datetime
export (Weekday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, DayOverflow, ClampToMonthEnd, RejectInvalidDay, TimeUnit, Hours, Minutes, Seconds, Milliseconds, Microseconds, Nanoseconds, Date, Time, DateTime, Instant, Offset, OffsetDateTime, Duration, Period, Dates, Instants, is_leap_year, days_in_year, days_in_month, weekday_iso_number, weekday_from_iso_number, try_weekday_from_iso_number, weekday_name, date, try_date, date_year, date_month, date_day, date_epoch_day, date_from_epoch_day, try_date_from_epoch_day, date_weekday, date_day_of_year, date_iso_week, date_from_iso_week, try_date_from_iso_week, date_add_days, date_days_until, date_add_months, try_date_add_months, date_add_period, try_date_add_period, date_period_until, date_lt, date_lte, date_gt, date_gte, date_to_string, parse_date, try_parse_date, nth_weekday_in_month, last_weekday_in_month, weekday_on_or_after, weekday_on_or_before, easter_sunday_gregorian, easter_sunday_orthodox, time, try_time, time_hour, time_minute, time_second, time_nanosecond, time_nanosecond_of_day, time_from_nanosecond_of_day, try_time_from_nanosecond_of_day, time_add_duration, time_until, time_lt, time_lte, time_gt, time_gte, time_to_string, parse_time, try_parse_time, datetime, datetime_date, datetime_time, datetime_add_duration, datetime_add_period, try_datetime_add_period, datetime_until, datetime_lt, datetime_lte, datetime_gt, datetime_gte, datetime_to_string, parse_datetime, try_parse_datetime, offset_from_seconds, try_offset_from_seconds, offset_seconds, offset_to_string, parse_offset, try_parse_offset, instant_from_unix, try_instant_from_unix, instant_unix_second, instant_nanosecond, instant_from_unix_count, try_instant_from_unix_count, instant_to_unix_count, try_instant_to_unix_count, instant_add_duration, instant_until, instant_round_to, instant_lt, instant_lte, instant_gt, instant_gte, instant_to_datetime_at, datetime_to_instant_at, instant_to_string, parse_instant, try_parse_instant, offset_datetime, offset_datetime_instant, offset_datetime_offset, offset_datetime_local, offset_datetime_to_string, parse_offset_datetime, try_parse_offset_datetime, duration, duration_from_count, duration_second, duration_nanosecond, duration_to_count, try_duration_to_count, duration_to_seconds_f64, duration_add, duration_sub, duration_negate, duration_mul, duration_lt, duration_lte, duration_gt, duration_gte, duration_to_string, parse_duration, try_parse_duration, period, try_period, period_months, period_days, period_negate, period_mul, period_to_string, parse_period, try_parse_period, dates_from_epoch_days, try_dates_from_epoch_days, dates_epoch_days, instants_from_unix, try_instants_from_unix, instants_unix_seconds, instants_nanoseconds)
import Std.Rounding (Rounding, RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)
-- Std.Datetime: civil dates and times, instants, durations, periods, fixed
-- offsets, and their text forms, governed by [05-OP-73]. Every value type is
-- opaque, so the exported validating producers below are the only way to
-- obtain one. Every range check runs before the arithmetic it protects, so no
-- primitive numeric trap escapes a call; failures use `fail` with the message
-- grammar `<function>: <kind>: <detail>`.
type Weekday =
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
type DayOverflow =
  | ClampToMonthEnd
  | RejectInvalidDay
type TimeUnit =
  | Hours
  | Minutes
  | Seconds
  | Milliseconds
  | Microseconds
  | Nanoseconds
@opaque
type Date =
  | Date { epoch_day: i64 }
@opaque
type Time =
  | Time { nanosecond_of_day: i64 }
@opaque
type DateTime =
  | DateTime { epoch_day: i64, nanosecond_of_day: i64 }
@opaque
type Instant =
  | Instant { unix_second: i64, nanosecond: i64 }
@opaque
type Offset =
  | Offset { seconds: i64 }
@opaque
type OffsetDateTime =
  | OffsetDateTime { instant: Instant, offset: Offset }
@opaque
type Duration =
  | Duration { second: i64, nanosecond: i64 }
@opaque
type Period =
  | Period { months: i64, days: i64 }
@opaque
type Dates[n] =
  | Dates { epoch_days: tensor[n, i64] }
@opaque
type Instants[n] =
  | Instants { unix_seconds: tensor[n, i64], nanoseconds: tensor[n, i64] }
-- Range constants. Years run over -9999..9999 and epoch days over the days
-- of those years. Total months are 12 * year + month - 1. Unix seconds cover
-- the civil range shrunk by the largest offset at each end, so every instant
-- has a civil reading under every offset; offsets lie strictly within one
-- day.
def min_year() -> i64 = -9999i64
def max_year() -> i64 = 9999i64
def min_epoch_day() -> i64 = -4371587i64
def max_epoch_day() -> i64 = 2932896i64
def min_total_month() -> i64 = -119988i64
def max_total_month() -> i64 = 119999i64
def min_unix_second() -> i64 = -377705030401i64
def max_unix_second() -> i64 = 253402214400i64
def max_offset_seconds() -> i64 = 86399i64
def nanos_per_second() -> i64 = 1000000000i64
def seconds_per_day() -> i64 = 86400i64
-- Checked-arithmetic predicates: each answers whether the exact result fits in
-- i64 without performing the operation that could trap.
def euclid_rem(x: i64, y: i64) -> i64 = {
  r = mod(x, y)
  if lt(r, 0i64) then add(r, y) else r
}
def sum_fits(x: i64, y: i64) -> bool = if gte(y, 0i64) then lte(x, sub(9223372036854775807i64, y)) else gte(x, sub(sub(-9223372036854775807i64, 1i64), y))
def difference_fits(x: i64, y: i64) -> bool = if gte(y, 0i64) then gte(x, add(sub(-9223372036854775807i64, 1i64), y)) else lte(x, add(9223372036854775807i64, y))
def product_fits(x: i64, y: i64) -> bool = if (x |> eq(0i64) |> or(eq(y, 0i64))) then true else if gt(x, 0i64) then if gt(y, 0i64) then lte(x, floor_div(9223372036854775807i64, y)) else if eq(y, -1i64) then true else lte(x, floor_div(sub(-9223372036854775807i64, 1i64), y)) else if gt(y, 0i64) then gte(x, floor_div(add(sub(-9223372036854775807i64, 1i64), sub(y, 1i64)), y)) else gte(x, neg(floor_div(neg(9223372036854775807i64), y)))
-- `scaled_sum(x, g, s)` is `x * g + s` for `g > 0` and `0 <= s <= g`, or None
-- when that exact value does not fit in i64.
def scaled_sum(x: i64, g: i64, s: i64) -> Option[i64] =
  if gte(x, 0i64) then if lte(x, floor_div(sub(9223372036854775807i64, s), g)) then Some(add(mul(x, g), s)) else None else {
    raised = add(x, 1i64)
    gap = sub(g, s)
    if gte(raised, floor_div(add(add(sub(-9223372036854775807i64, 1i64), gap), sub(g, 1i64)), g)) then Some(sub(mul(raised, g), gap)) else None
  }
-- Text assembly.
def joined(parts: List[string]) -> string = fold(fn (acc: string, part: string) -> string_concat(acc, part), "", parts)
def domain_failure(function: string, detail: string) -> string = joined([function, ": domain: ", detail])
def overflow_failure(function: string, detail: string) -> string = joined([function, ": overflow: ", detail])
def quoted(text: string) -> string = joined(["\"", text, "\""])
def zero_padded(value: i64, width: i64) -> string = {
  digits = to_string(value)
  missing = sub(width, string_len(digits))
  if gt(missing, 0i64) then ("000000000" |> string_slice(0i64, missing) |> string_concat(digits)) else digits
}
-- The decimal digits of |value|, exact for every i64 including the minimum.
def magnitude_text(value: i64) -> string = {
  digits = to_string(value)
  if lt(value, 0i64) then string_slice(digits, 1i64, digits |> string_len |> sub(1i64)) else digits
}
def year_text(year: i64) -> string = if (year |> gte(0i64) |> and(lte(year, 9999i64))) then zero_padded(year, 4i64) else if (year |> lt(0i64) |> and(gte(year, -999999i64))) then string_concat("-", year |> neg |> zero_padded(6i64)) else to_string(year)
def year_month_text(year: i64, month: i64) -> string = joined([year_text(year), "-", zero_padded(month, 2i64)])
def fraction_text(nanosecond: i64) -> string =
  if eq(nanosecond, 0i64) then "" else {
    digits = zero_padded(nanosecond, 9i64)
    kept = fold(fn (acc: i64, idx: i64) -> if (digits |> string_slice(idx, 1i64) |> neq("0")) then add(idx, 1i64) else acc, 0i64, range(0i64, 9i64))
    string_concat(".", string_slice(digits, 0i64, kept))
  }
def range_text(low: i64, high: i64) -> string = joined([to_string(low), "..", to_string(high)])
def outside_text(name: string, value: i64, low: i64, high: i64) -> string = joined([name, " ", to_string(value), " is outside ", range_text(low, high)])
def in_span(value: i64, low: i64, high: i64) -> bool = value |> gte(low) |> and(lte(value, high))
-- Calendar algorithms: Hinnant's days_from_civil and civil_from_days with
-- floor division and Euclidean remainders. Over the supported range every
-- intermediate stays below 2^33.
def days_from_civil(year: i64, month: i64, day: i64) -> i64 = {
  shifted = if lte(month, 2i64) then sub(year, 1i64) else year
  era = floor_div(shifted, 400i64)
  yoe = sub(shifted, mul(era, 400i64))
  mp = if gt(month, 2i64) then sub(month, 3i64) else add(month, 9i64)
  doy = add(floor_div(add(mul(153i64, mp), 2i64), 5i64), sub(day, 1i64))
  doe = add(sub(add(mul(yoe, 365i64), floor_div(yoe, 4i64)), floor_div(yoe, 100i64)), doy)
  sub(add(mul(era, 146097i64), doe), 719468i64)
}
def civil_from_days(epoch_day: i64) -> (i64, i64, i64) = {
  shifted = add(epoch_day, 719468i64)
  era = floor_div(shifted, 146097i64)
  doe = sub(shifted, mul(era, 146097i64))
  yoe = floor_div(sub(add(sub(doe, floor_div(doe, 1460i64)), floor_div(doe, 36524i64)), floor_div(doe, 146096i64)), 365i64)
  doy = sub(doe, sub(add(mul(365i64, yoe), floor_div(yoe, 4i64)), floor_div(yoe, 100i64)))
  mp = floor_div(add(mul(5i64, doy), 2i64), 153i64)
  day = doy |> sub(floor_div(add(mul(153i64, mp), 2i64), 5i64)) |> add(1i64)
  month = if lt(mp, 10i64) then add(mp, 3i64) else sub(mp, 9i64)
  year = add(yoe, mul(era, 400i64))
  (if lte(month, 2i64) then add(year, 1i64) else year, month, day)
}
def month_length(year: i64, month: i64) -> i64 = if eq(month, 2i64) then if is_leap_year(year) then 29i64 else 28i64 else if or(or(eq(month, 4i64), eq(month, 6i64)), or(eq(month, 9i64), eq(month, 11i64))) then 30i64 else 31i64
def epoch_day_text(epoch_day: i64) -> string = {
  (year, month, day) = civil_from_days(epoch_day)
  joined([year_text(year), "-", zero_padded(month, 2i64), "-", zero_padded(day, 2i64)])
}
-- The first problem with a (year, month, day) triple, if any.
def ymd_problem(year: i64, month: i64, day: i64) -> Option[string] = if (year |> in_span(min_year(), max_year()) |> not) then ("year" |> outside_text(year, min_year(), max_year()) |> Some) else if (month |> in_span(1i64, 12i64) |> not) then ("month" |> outside_text(month, 1i64, 12i64) |> Some) else if (day |> in_span(1i64, month_length(year, month)) |> not) then Some(joined([outside_text("day", day, 1i64, month_length(year, month)), " for ", year_month_text(year, month)])) else None
def year_problem(year: i64) -> Option[string] = if in_span(year, min_year(), max_year()) then None else ("year" |> outside_text(year, min_year(), max_year()) |> Some)
def year_month_problem(year: i64, month: i64) -> Option[string] = if (year |> in_span(min_year(), max_year()) |> not) then ("year" |> outside_text(year, min_year(), max_year()) |> Some) else if (month |> in_span(1i64, 12i64) |> not) then ("month" |> outside_text(month, 1i64, 12i64) |> Some) else None
def epoch_day_problem(epoch_day: i64) -> Option[string] = if in_span(epoch_day, min_epoch_day(), max_epoch_day()) then None else Some(outside_text("epoch day", epoch_day, min_epoch_day(), max_epoch_day()))
-- Calendar queries, total over every i64 year.
def is_leap_year(year: i64) -> bool = and(eq(mod(year, 4i64), 0i64), or(neq(mod(year, 100i64), 0i64), eq(mod(year, 400i64), 0i64)))
def days_in_year(year: i64) -> i64 = if is_leap_year(year) then 366i64 else 365i64
def days_in_month(year: i64, month: i64) -> i64 = if in_span(month, 1i64, 12i64) then month_length(year, month) else fail(domain_failure("days_in_month", outside_text("month", month, 1i64, 12i64)))
def weekday_index(w: Weekday) -> i64 =
  match w with {
    | Monday => 0i64
    | Tuesday => 1i64
    | Wednesday => 2i64
    | Thursday => 3i64
    | Friday => 4i64
    | Saturday => 5i64
    | Sunday => 6i64
  }
def weekday_at_index(idx: i64) -> Weekday = if eq(idx, 0i64) then Monday else if eq(idx, 1i64) then Tuesday else if eq(idx, 2i64) then Wednesday else if eq(idx, 3i64) then Thursday else if eq(idx, 4i64) then Friday else if eq(idx, 5i64) then Saturday else Sunday
def weekday_iso_number(w: Weekday) -> i64 = w |> weekday_index |> add(1i64)
def weekday_from_iso_number(n: i64) -> Weekday = if in_span(n, 1i64, 7i64) then (n |> sub(1i64) |> weekday_at_index) else fail(domain_failure("weekday_from_iso_number", outside_text("weekday number", n, 1i64, 7i64)))
def try_weekday_from_iso_number(n: i64) -> Option[Weekday] = if in_span(n, 1i64, 7i64) then Some(weekday_at_index(sub(n, 1i64))) else None
def weekday_name(w: Weekday) -> string =
  match w with {
    | Monday => "monday"
    | Tuesday => "tuesday"
    | Wednesday => "wednesday"
    | Thursday => "thursday"
    | Friday => "friday"
    | Saturday => "saturday"
    | Sunday => "sunday"
  }
def epoch_weekday_index(epoch_day: i64) -> i64 = epoch_day |> add(3i64) |> euclid_rem(7i64)
-- Date construction and access.
def date(year: i64, month: i64, day: i64) -> Date =
  match ymd_problem(year, month, day) with {
    | Some(detail) => "date" |> domain_failure(detail) |> fail
    | None => Date { epoch_day: days_from_civil(year, month, day) }
  }
def try_date(year: i64, month: i64, day: i64) -> Option[Date] =
  match ymd_problem(year, month, day) with {
    | Some(_) => None
    | None => Some(Date { epoch_day: days_from_civil(year, month, day) })
  }
def date_year(d: Date) -> i64 = civil_from_days(d.epoch_day).0
def date_month(d: Date) -> i64 = civil_from_days(d.epoch_day).1
def date_day(d: Date) -> i64 = civil_from_days(d.epoch_day).2
def date_epoch_day(d: Date) -> i64 = d.epoch_day
def date_from_epoch_day(n: i64) -> Date =
  match epoch_day_problem(n) with {
    | Some(detail) => "date_from_epoch_day" |> domain_failure(detail) |> fail
    | None => Date { epoch_day: n }
  }
def try_date_from_epoch_day(n: i64) -> Option[Date] = if in_span(n, min_epoch_day(), max_epoch_day()) then Some(Date { epoch_day: n }) else None
def date_weekday(d: Date) -> Weekday = (d.epoch_day) |> epoch_weekday_index |> weekday_at_index
def day_of_year_of(epoch_day: i64) -> i64 = add(sub(epoch_day, days_from_civil(civil_from_days(epoch_day).0, 1i64, 1i64)), 1i64)
def date_day_of_year(d: Date) -> i64 = day_of_year_of(d.epoch_day)
-- ISO 8601 weeks: a week belongs to the week-year of its Thursday.
def ordinal_week_of(epoch_day: i64) -> (i64, i64) = {
  thursday = epoch_day |> sub(epoch_weekday_index(epoch_day)) |> add(3i64)
  (civil_from_days(thursday).0, add(floor_div(sub(day_of_year_of(thursday), 1i64), 7i64), 1i64))
}
def date_iso_week(d: Date) -> (i64, i64) = ordinal_week_of(d.epoch_day)
def monday_of_week_one(iso_year: i64) -> i64 = {
  january_fourth = days_from_civil(iso_year, 1i64, 4i64)
  sub(january_fourth, epoch_weekday_index(january_fourth))
}
def weeks_in_week_year(iso_year: i64) -> i64 = floor_div(sub(monday_of_week_one(add(iso_year, 1i64)), monday_of_week_one(iso_year)), 7i64)
def ordinal_week_text(iso_year: i64, week: i64, w: Weekday) -> string = joined([year_text(iso_year), "-W", zero_padded(week, 2i64), "-", w |> weekday_iso_number |> to_string])
def ordinal_week_problem(iso_year: i64, week: i64, w: Weekday) -> Option[string] = if not(in_span(iso_year, min_year(), max_year())) then Some(outside_text("ISO week-year", iso_year, min_year(), max_year())) else if not(in_span(week, 1i64, weeks_in_week_year(iso_year))) then Some(joined([outside_text("week", week, 1i64, weeks_in_week_year(iso_year)), " for ISO week-year ", year_text(iso_year)])) else if not(in_span(epoch_day_of_week_date(iso_year, week, w), min_epoch_day(), max_epoch_day())) then Some(joined(["ISO week date ", ordinal_week_text(iso_year, week, w), " is outside the supported date range"])) else None
def epoch_day_of_week_date(iso_year: i64, week: i64, w: Weekday) -> i64 = add(add(monday_of_week_one(iso_year), mul(sub(week, 1i64), 7i64)), weekday_index(w))
def date_from_iso_week(iso_year: i64, week: i64, weekday: Weekday) -> Date =
  match ordinal_week_problem(iso_year, week, weekday) with {
    | Some(detail) => "date_from_iso_week" |> domain_failure(detail) |> fail
    | None => Date { epoch_day: epoch_day_of_week_date(iso_year, week, weekday) }
  }
def try_date_from_iso_week(iso_year: i64, week: i64, weekday: Weekday) -> Option[Date] =
  match ordinal_week_problem(iso_year, week, weekday) with {
    | Some(_) => None
    | None => Some(Date { epoch_day: epoch_day_of_week_date(iso_year, week, weekday) })
  }
-- Date arithmetic. Each range check runs before the addition it protects.
def shifted_epoch_day(function: string, epoch_day: i64, n: i64) -> i64 = if or(gt(n, sub(max_epoch_day(), epoch_day)), lt(n, sub(min_epoch_day(), epoch_day))) then fail(overflow_failure(function, joined([epoch_day_text(epoch_day), " plus ", to_string(n), " days is outside the supported date range"]))) else add(epoch_day, n)
def date_add_days(d: Date, n: i64) -> Date = Date { epoch_day: shifted_epoch_day("date_add_days", d.epoch_day, n) }
def date_days_until(a: Date, b: Date) -> i64 = sub(b.epoch_day, a.epoch_day)
def total_month_of(epoch_day: i64) -> i64 = {
  (year, month, _) = civil_from_days(epoch_day)
  year |> mul(12i64) |> add(sub(month, 1i64))
}
-- The epoch day `n` months after `epoch_day` with the day clamped to the
-- target month's end; the caller has already range-checked the target.
def clamped_month_shift(epoch_day: i64, n: i64) -> i64 = {
  day = civil_from_days(epoch_day).2
  target = epoch_day |> total_month_of |> add(n)
  year = floor_div(target, 12i64)
  month = target |> euclid_rem(12i64) |> add(1i64)
  days_from_civil(year, month, if lte(day, month_length(year, month)) then day else month_length(year, month))
}
def month_shift_fits(epoch_day: i64, n: i64) -> bool = {
  current = total_month_of(epoch_day)
  and(lte(n, sub(max_total_month(), current)), gte(n, sub(min_total_month(), current)))
}
-- None when `RejectInvalidDay` fires; an out-of-range target fails `overflow`
-- under `function`'s name.
def month_shift(function: string, epoch_day: i64, n: i64, overflow: DayOverflow) -> Option[i64] =
  if (epoch_day |> month_shift_fits(n) |> not) then fail(overflow_failure(function, joined([epoch_day_text(epoch_day), " plus ", to_string(n), " months is outside the supported date range"]))) else {
    day = civil_from_days(epoch_day).2
    target = epoch_day |> total_month_of |> add(n)
    year = floor_div(target, 12i64)
    month = target |> euclid_rem(12i64) |> add(1i64)
    if lte(day, month_length(year, month)) then (year |> days_from_civil(month, day) |> Some) else match overflow with {
      | ClampToMonthEnd => year |> days_from_civil(month, month_length(year, month)) |> Some
      | RejectInvalidDay => None
    }
  }
def month_shift_rejection(epoch_day: i64, n: i64) -> string = {
  day = civil_from_days(epoch_day).2
  target = epoch_day |> total_month_of |> add(n)
  year = floor_div(target, 12i64)
  month = target |> euclid_rem(12i64) |> add(1i64)
  joined([outside_text("day", day, 1i64, month_length(year, month)), " for ", year_month_text(year, month)])
}
def date_add_months(d: Date, n: i64, overflow: DayOverflow) -> Date =
  match month_shift("date_add_months", d.epoch_day, n, overflow) with {
    | Some(epoch_day) => Date { epoch_day }
    | None => fail(domain_failure("date_add_months", month_shift_rejection(d.epoch_day, n)))
  }
def try_date_add_months(d: Date, n: i64, overflow: DayOverflow) -> Option[Date] =
  match month_shift("try_date_add_months", d.epoch_day, n, overflow) with {
    | Some(epoch_day) => Some(Date { epoch_day })
    | None => None
  }
-- Months first, then days: the order every surveyed library uses.
def period_shift(function: string, epoch_day: i64, p: Period, overflow: DayOverflow) -> Option[i64] =
  match month_shift(function, epoch_day, p.months, overflow) with {
    | Some(moved) => function |> shifted_epoch_day(moved, p.days) |> Some
    | None => None
  }
def date_add_period(d: Date, p: Period, overflow: DayOverflow) -> Date =
  match period_shift("date_add_period", d.epoch_day, p, overflow) with {
    | Some(epoch_day) => Date { epoch_day }
    | None => fail(domain_failure("date_add_period", month_shift_rejection(d.epoch_day, p.months)))
  }
def try_date_add_period(d: Date, p: Period, overflow: DayOverflow) -> Option[Date] =
  match period_shift("try_date_add_period", d.epoch_day, p, overflow) with {
    | Some(epoch_day) => Some(Date { epoch_day })
    | None => None
  }
-- The single-sign period whose months have the largest magnitude such that
-- adding it to `a` with `ClampToMonthEnd` gives `b`.
def date_period_until(a: Date, b: Date) -> Period = {
  (by, bm, bd) = civil_from_days(b.epoch_day)
  ad = civil_from_days(a.epoch_day).2
  landing = if lte(ad, month_length(by, bm)) then ad else month_length(by, bm)
  if lte(a.epoch_day, b.epoch_day) then {
    whole = (b.epoch_day) |> total_month_of |> sub(total_month_of(a.epoch_day))
    months = if lte(landing, bd) then whole else sub(whole, 1i64)
    Period { months, days: sub(b.epoch_day, clamped_month_shift(a.epoch_day, months)) }
  } else {
    whole = (a.epoch_day) |> total_month_of |> sub(total_month_of(b.epoch_day))
    months = if gte(landing, bd) then whole else sub(whole, 1i64)
    Period { months: neg(months), days: sub(b.epoch_day, clamped_month_shift(a.epoch_day, neg(months))) }
  }
}
def date_lt(a: Date, b: Date) -> bool = lt(a.epoch_day, b.epoch_day)
def date_lte(a: Date, b: Date) -> bool = lte(a.epoch_day, b.epoch_day)
def date_gt(a: Date, b: Date) -> bool = gt(a.epoch_day, b.epoch_day)
def date_gte(a: Date, b: Date) -> bool = gte(a.epoch_day, b.epoch_day)
def date_to_string(d: Date) -> string = epoch_day_text(d.epoch_day)
-- Holiday-rule helpers: calendar computations that holiday data composes.
def nth_weekday_in_month(year: i64, month: i64, w: Weekday, n: i64) -> Option[Date] =
  match year_month_problem(year, month) with {
    | Some(detail) => "nth_weekday_in_month" |> domain_failure(detail) |> fail
    | None => if lte(n, 0i64) then fail(domain_failure("nth_weekday_in_month", joined(["occurrence ", to_string(n), " is not positive"]))) else if gt(n, 5i64) then None else {
    first = days_from_civil(year, month, 1i64)
    day = add(add(1i64, euclid_rem(sub(weekday_index(w), epoch_weekday_index(first)), 7i64)), mul(sub(n, 1i64), 7i64))
    if lte(day, month_length(year, month)) then Some(Date { epoch_day: days_from_civil(year, month, day) }) else None
  }
  }
def last_weekday_in_month(year: i64, month: i64, w: Weekday) -> Date =
  match year_month_problem(year, month) with {
    | Some(detail) => "last_weekday_in_month" |> domain_failure(detail) |> fail
    | None => {
    last = days_from_civil(year, month, month_length(year, month))
    Date { epoch_day: sub(last, euclid_rem(sub(epoch_weekday_index(last), weekday_index(w)), 7i64)) }
  }
  }
def weekday_on_or_after(d: Date, w: Weekday) -> Date = Date { epoch_day: shifted_epoch_day("weekday_on_or_after", d.epoch_day, euclid_rem(sub(weekday_index(w), epoch_weekday_index(d.epoch_day)), 7i64)) }
def weekday_on_or_before(d: Date, w: Weekday) -> Date = Date { epoch_day: shifted_epoch_day("weekday_on_or_before", d.epoch_day, neg(euclid_rem(sub(epoch_weekday_index(d.epoch_day), weekday_index(w)), 7i64))) }
def checked_easter(function: string, epoch_day: i64) -> Date =
  match epoch_day_problem(epoch_day) with {
    | Some(detail) => function |> overflow_failure(detail) |> fail
    | None => Date { epoch_day }
  }
-- The anonymous Gregorian computus in Meeus's form.
def gregorian_easter_epoch_day(year: i64) -> i64 = {
  a = euclid_rem(year, 19i64)
  b = floor_div(year, 100i64)
  c = euclid_rem(year, 100i64)
  d = floor_div(b, 4i64)
  e = euclid_rem(b, 4i64)
  f = b |> add(8i64) |> floor_div(25i64)
  g = floor_div(add(sub(b, f), 1i64), 3i64)
  h = euclid_rem(add(sub(sub(add(mul(19i64, a), b), d), g), 15i64), 30i64)
  i = floor_div(c, 4i64)
  k = euclid_rem(c, 4i64)
  l = euclid_rem(sub(sub(add(add(32i64, mul(2i64, e)), mul(2i64, i)), h), k), 7i64)
  m = floor_div(add(add(a, mul(11i64, h)), mul(22i64, l)), 451i64)
  s = add(sub(add(h, l), mul(7i64, m)), 114i64)
  days_from_civil(year, floor_div(s, 31i64), s |> euclid_rem(31i64) |> add(1i64))
}
-- The Julian computus in Meeus's form, then the Julian calendar date's
-- proleptic Gregorian epoch day by way of its Julian day number.
def orthodox_easter_epoch_day(year: i64) -> i64 = {
  a = euclid_rem(year, 4i64)
  b = euclid_rem(year, 7i64)
  c = euclid_rem(year, 19i64)
  d = euclid_rem(add(mul(19i64, c), 15i64), 30i64)
  e = euclid_rem(sub(add(add(mul(2i64, a), mul(4i64, b)), 34i64), d), 7i64)
  s = d |> add(e) |> add(114i64)
  month = floor_div(s, 31i64)
  day = s |> euclid_rem(31i64) |> add(1i64)
  shift = 14i64 |> sub(month) |> floor_div(12i64)
  y = year |> add(4800i64) |> sub(shift)
  mp = month |> add(mul(12i64, shift)) |> sub(3i64)
  julian_day = sub(add(add(add(day, floor_div(add(mul(153i64, mp), 2i64), 5i64)), mul(365i64, y)), floor_div(y, 4i64)), 32083i64)
  sub(julian_day, 2440588i64)
}
def easter_sunday_gregorian(year: i64) -> Date =
  match year_problem(year) with {
    | Some(detail) => "easter_sunday_gregorian" |> domain_failure(detail) |> fail
    | None => checked_easter("easter_sunday_gregorian", gregorian_easter_epoch_day(year))
  }
def easter_sunday_orthodox(year: i64) -> Date =
  match year_problem(year) with {
    | Some(detail) => "easter_sunday_orthodox" |> domain_failure(detail) |> fail
    | None => checked_easter("easter_sunday_orthodox", orthodox_easter_epoch_day(year))
  }
-- Time of day.
def clock_problem(hour: i64, minute: i64, second: i64, nanosecond: i64) -> Option[string] = if (hour |> in_span(0i64, 23i64) |> not) then ("hour" |> outside_text(hour, 0i64, 23i64) |> Some) else if (minute |> in_span(0i64, 59i64) |> not) then ("minute" |> outside_text(minute, 0i64, 59i64) |> Some) else if (second |> in_span(0i64, 59i64) |> not) then ("second" |> outside_text(second, 0i64, 59i64) |> Some) else if (nanosecond |> in_span(0i64, 999999999i64) |> not) then ("nanosecond" |> outside_text(nanosecond, 0i64, 999999999i64) |> Some) else None
def clock_nanos(hour: i64, minute: i64, second: i64, nanosecond: i64) -> i64 = add(mul(add(mul(add(mul(hour, 60i64), minute), 60i64), second), nanos_per_second()), nanosecond)
def time(hour: i64, minute: i64, second: i64, nanosecond: i64) -> Time =
  match clock_problem(hour, minute, second, nanosecond) with {
    | Some(detail) => "time" |> domain_failure(detail) |> fail
    | None => Time { nanosecond_of_day: clock_nanos(hour, minute, second, nanosecond) }
  }
def try_time(hour: i64, minute: i64, second: i64, nanosecond: i64) -> Option[Time] =
  match clock_problem(hour, minute, second, nanosecond) with {
    | Some(_) => None
    | None => Some(Time { nanosecond_of_day: clock_nanos(hour, minute, second, nanosecond) })
  }
def time_hour(t: Time) -> i64 = floor_div(t.nanosecond_of_day, 3600000000000i64)
def time_minute(t: Time) -> i64 = (t.nanosecond_of_day) |> floor_div(60000000000i64) |> euclid_rem(60i64)
def time_second(t: Time) -> i64 = (t.nanosecond_of_day) |> floor_div(nanos_per_second()) |> euclid_rem(60i64)
def time_nanosecond(t: Time) -> i64 = euclid_rem(t.nanosecond_of_day, nanos_per_second())
def time_nanosecond_of_day(t: Time) -> i64 = t.nanosecond_of_day
def time_from_nanosecond_of_day(n: i64) -> Time = if in_span(n, 0i64, sub(86400000000000i64, 1i64)) then Time { nanosecond_of_day: n } else fail(domain_failure("time_from_nanosecond_of_day", outside_text("nanosecond of day", n, 0i64, sub(86400000000000i64, 1i64))))
def try_time_from_nanosecond_of_day(n: i64) -> Option[Time] = if in_span(n, 0i64, sub(86400000000000i64, 1i64)) then Some(Time { nanosecond_of_day: n }) else None
-- Adds a duration to (second of day, nanosecond) and returns the whole days
-- carried with the new second of day and nanosecond. Never overflows.
def clock_shift(second_of_day: i64, nanosecond: i64, d: Duration) -> (i64, i64, i64) = {
  nanos = add(nanosecond, d.nanosecond)
  carry = if gte(nanos, nanos_per_second()) then 1i64 else 0i64
  seconds =
    second_of_day
    |> add(euclid_rem(d.second, seconds_per_day()))
    |> add(carry)
  (add(floor_div(d.second, seconds_per_day()), floor_div(seconds, seconds_per_day())), euclid_rem(seconds, seconds_per_day()), sub(nanos, mul(carry, nanos_per_second())))
}
def time_add_duration(t: Time, d: Duration) -> (i64, Time) = {
  (days, second_of_day, nanosecond) = clock_shift(floor_div(t.nanosecond_of_day, nanos_per_second()), euclid_rem(t.nanosecond_of_day, nanos_per_second()), d)
  (days, Time { nanosecond_of_day: second_of_day |> mul(nanos_per_second()) |> add(nanosecond) })
}
-- The normalized duration of a nanosecond count that is known to fit in i64.
def nanos_duration(nanos: i64) -> Duration = Duration { second: floor_div(nanos, nanos_per_second()), nanosecond: euclid_rem(nanos, nanos_per_second()) }
def time_until(a: Time, b: Time) -> Duration = (b.nanosecond_of_day) |> sub(a.nanosecond_of_day) |> nanos_duration
def time_lt(a: Time, b: Time) -> bool = lt(a.nanosecond_of_day, b.nanosecond_of_day)
def time_lte(a: Time, b: Time) -> bool = lte(a.nanosecond_of_day, b.nanosecond_of_day)
def time_gt(a: Time, b: Time) -> bool = gt(a.nanosecond_of_day, b.nanosecond_of_day)
def time_gte(a: Time, b: Time) -> bool = gte(a.nanosecond_of_day, b.nanosecond_of_day)
def clock_text(nanosecond_of_day: i64) -> string = {
  second_of_day = floor_div(nanosecond_of_day, nanos_per_second())
  joined([second_of_day |> floor_div(3600i64) |> zero_padded(2i64), ":", zero_padded(euclid_rem(floor_div(second_of_day, 60i64), 60i64), 2i64), ":", second_of_day |> euclid_rem(60i64) |> zero_padded(2i64), nanosecond_of_day |> euclid_rem(nanos_per_second()) |> fraction_text])
}
def time_to_string(t: Time) -> string = clock_text(t.nanosecond_of_day)
-- Civil date and time.
def datetime(d: Date, t: Time) -> DateTime = DateTime { epoch_day: d.epoch_day, nanosecond_of_day: t.nanosecond_of_day }
def datetime_date(dt: DateTime) -> Date = Date { epoch_day: dt.epoch_day }
def datetime_time(dt: DateTime) -> Time = Time { nanosecond_of_day: dt.nanosecond_of_day }
def duration_text(d: Duration) -> string = if lt(d.second, 0i64) then if eq(d.nanosecond, 0i64) then joined(["-PT", magnitude_text(d.second), "S"]) else joined(["-PT", (-1i64) |> sub(d.second) |> to_string, fraction_text(sub(nanos_per_second(), d.nanosecond)), "S"]) else joined(["PT", to_string(d.second), fraction_text(d.nanosecond), "S"])
def civil_text(epoch_day: i64, nanosecond_of_day: i64) -> string = joined([epoch_day_text(epoch_day), "T", clock_text(nanosecond_of_day)])
def datetime_add_duration(dt: DateTime, d: Duration) -> DateTime = {
  (days, second_of_day, nanosecond) = clock_shift(floor_div(dt.nanosecond_of_day, nanos_per_second()), euclid_rem(dt.nanosecond_of_day, nanos_per_second()), d)
  if or(gt(days, sub(max_epoch_day(), dt.epoch_day)), lt(days, sub(min_epoch_day(), dt.epoch_day))) then fail(overflow_failure("datetime_add_duration", joined([civil_text(dt.epoch_day, dt.nanosecond_of_day), " plus ", duration_text(d), " is outside the supported range"]))) else DateTime { epoch_day: add(dt.epoch_day, days), nanosecond_of_day: second_of_day |> mul(nanos_per_second()) |> add(nanosecond) }
}
def datetime_add_period(dt: DateTime, p: Period, overflow: DayOverflow) -> DateTime =
  match period_shift("datetime_add_period", dt.epoch_day, p, overflow) with {
    | Some(epoch_day) => DateTime { epoch_day, nanosecond_of_day: dt.nanosecond_of_day }
    | None => fail(domain_failure("datetime_add_period", month_shift_rejection(dt.epoch_day, p.months)))
  }
def try_datetime_add_period(dt: DateTime, p: Period, overflow: DayOverflow) -> Option[DateTime] =
  match period_shift("try_datetime_add_period", dt.epoch_day, p, overflow) with {
    | Some(epoch_day) => Some(DateTime { epoch_day, nanosecond_of_day: dt.nanosecond_of_day })
    | None => None
  }
-- The exact duration between two (second, nanosecond) readings whose second
-- difference is known to fit in i64.
def elapsed_between(a_second: i64, a_nanosecond: i64, b_second: i64, b_nanosecond: i64) -> Duration = {
  nanos = sub(b_nanosecond, a_nanosecond)
  Duration { second: b_second |> sub(a_second) |> add(floor_div(nanos, nanos_per_second())), nanosecond: euclid_rem(nanos, nanos_per_second()) }
}
def civil_second(epoch_day: i64, nanosecond_of_day: i64) -> i64 = add(mul(epoch_day, seconds_per_day()), floor_div(nanosecond_of_day, nanos_per_second()))
def datetime_until(a: DateTime, b: DateTime) -> Duration = elapsed_between(civil_second(a.epoch_day, a.nanosecond_of_day), euclid_rem(a.nanosecond_of_day, nanos_per_second()), civil_second(b.epoch_day, b.nanosecond_of_day), euclid_rem(b.nanosecond_of_day, nanos_per_second()))
def reading_before(a_major: i64, a_minor: i64, b_major: i64, b_minor: i64) -> bool = a_major |> lt(b_major) |> or(and(eq(a_major, b_major), lt(a_minor, b_minor)))
def datetime_lt(a: DateTime, b: DateTime) -> bool = reading_before(a.epoch_day, a.nanosecond_of_day, b.epoch_day, b.nanosecond_of_day)
def datetime_lte(a: DateTime, b: DateTime) -> bool = not(reading_before(b.epoch_day, b.nanosecond_of_day, a.epoch_day, a.nanosecond_of_day))
def datetime_gt(a: DateTime, b: DateTime) -> bool = reading_before(b.epoch_day, b.nanosecond_of_day, a.epoch_day, a.nanosecond_of_day)
def datetime_gte(a: DateTime, b: DateTime) -> bool = not(reading_before(a.epoch_day, a.nanosecond_of_day, b.epoch_day, b.nanosecond_of_day))
def datetime_to_string(dt: DateTime) -> string = civil_text(dt.epoch_day, dt.nanosecond_of_day)
-- Fixed offsets.
def offset_problem(seconds: i64) -> Option[string] = if in_span(seconds, neg(max_offset_seconds()), max_offset_seconds()) then None else Some(joined(["offset ", to_string(seconds), " s is outside ", range_text(neg(max_offset_seconds()), max_offset_seconds())]))
def offset_from_seconds(s: i64) -> Offset =
  match offset_problem(s) with {
    | Some(detail) => "offset_from_seconds" |> domain_failure(detail) |> fail
    | None => Offset { seconds: s }
  }
def try_offset_from_seconds(s: i64) -> Option[Offset] =
  match offset_problem(s) with {
    | Some(_) => None
    | None => Some(Offset { seconds: s })
  }
def offset_seconds(o: Offset) -> i64 = o.seconds
def offset_text(seconds: i64) -> string =
  if eq(seconds, 0i64) then "Z" else {
    size = if lt(seconds, 0i64) then neg(seconds) else seconds
    joined([if lt(seconds, 0i64) then "-" else "+", size |> floor_div(3600i64) |> zero_padded(2i64), ":", zero_padded(euclid_rem(floor_div(size, 60i64), 60i64), 2i64), if (size |> euclid_rem(60i64) |> eq(0i64)) then "" else string_concat(":", size |> euclid_rem(60i64) |> zero_padded(2i64))])
  }
def offset_to_string(o: Offset) -> string = offset_text(o.seconds)
-- Instants on the POSIX timescale.
def unix_second_problem(second: i64) -> Option[string] = if in_span(second, min_unix_second(), max_unix_second()) then None else Some(outside_text("unix second", second, min_unix_second(), max_unix_second()))
def instant_problem(second: i64, nanosecond: i64) -> Option[string] = if (nanosecond |> in_span(0i64, 999999999i64) |> not) then ("nanosecond" |> outside_text(nanosecond, 0i64, 999999999i64) |> Some) else unix_second_problem(second)
def instant_from_unix(second: i64, nanosecond: i64) -> Instant =
  match instant_problem(second, nanosecond) with {
    | Some(detail) => "instant_from_unix" |> domain_failure(detail) |> fail
    | None => Instant { unix_second: second, nanosecond }
  }
def try_instant_from_unix(second: i64, nanosecond: i64) -> Option[Instant] =
  match instant_problem(second, nanosecond) with {
    | Some(_) => None
    | None => Some(Instant { unix_second: second, nanosecond })
  }
def instant_unix_second(i: Instant) -> i64 = i.unix_second
def instant_nanosecond(i: Instant) -> i64 = i.nanosecond
-- Units: Hours and Minutes are whole multiples of a second; the others are
-- whole fractions of one.
def name_of_unit(unit: TimeUnit) -> string =
  match unit with {
    | Hours => "hours"
    | Minutes => "minutes"
    | Seconds => "seconds"
    | Milliseconds => "milliseconds"
    | Microseconds => "microseconds"
    | Nanoseconds => "nanoseconds"
  }
def seconds_per_unit(unit: TimeUnit) -> i64 =
  match unit with {
    | Hours => 3600i64
    | Minutes => 60i64
    | _ => 0i64
  }
def units_per_second(unit: TimeUnit) -> i64 =
  match unit with {
    | Milliseconds => 1000i64
    | Microseconds => 1000000i64
    | Nanoseconds => nanos_per_second()
    | _ => 1i64
  }
-- The exact (second, nanosecond) of `count` units, or None when the second
-- does not fit in i64.
def count_parts(count: i64, unit: TimeUnit) -> Option[(i64, i64)] = {
  multiple = seconds_per_unit(unit)
  if gt(multiple, 0i64) then if product_fits(count, multiple) then Some((mul(count, multiple), 0i64)) else None else {
    fraction = units_per_second(unit)
    Some((floor_div(count, fraction), count |> euclid_rem(fraction) |> mul(floor_div(nanos_per_second(), fraction))))
  }
}
-- 0 or 1: whether rounding moves a value up from its floor, given the
-- remainder `rem` over `denom`, whether the floor is even, and whether the
-- value is negative; None when `RejectInexact` meets a value that is not a
-- whole multiple.
def rounding_step(rounding: Rounding, rem: i64, denom: i64, floor_even: bool, negative: bool) -> Option[i64] =
  if eq(rem, 0i64) then Some(0i64) else match rounding with {
    | RoundTowardNegative => Some(0i64)
    | RoundTowardPositive => Some(1i64)
    | RoundTowardZero => Some(if negative then 1i64 else 0i64)
    | RoundAwayFromZero => Some(if negative then 0i64 else 1i64)
    | RoundTiesToEven => Some(nearest_step(rem, denom, if floor_even then 0i64 else 1i64))
    | RoundTiesToAway => Some(nearest_step(rem, denom, if negative then 0i64 else 1i64))
    | RejectInexact => None
  }
-- The step to the nearer of floor and floor + 1, or `tie` at the midpoint.
def nearest_step(rem: i64, denom: i64, tie: i64) -> i64 = if lt(mul(rem, 2i64), denom) then 0i64 else if gt(mul(rem, 2i64), denom) then 1i64 else tie
def is_even(value: i64) -> bool = value |> euclid_rem(2i64) |> eq(0i64)
-- The rounded count of `unit` in the exact value `second + nanosecond / 1e9`
-- with an empty status, or status "inexact" when `RejectInexact` meets a value
-- that is not a whole count, or "range" when the count does not fit in i64.
def parts_count(second: i64, nanosecond: i64, unit: TimeUnit, rounding: Rounding) -> (string, i64) = {
  multiple = seconds_per_unit(unit)
  if gt(multiple, 0i64) then {
    whole = floor_div(second, multiple)
    rem = add(mul(euclid_rem(second, multiple), nanos_per_second()), nanosecond)
    match rounding_step(rounding, rem, mul(multiple, nanos_per_second()), is_even(whole), lt(second, 0i64)) with {
      | Some(step) => ("", add(whole, step))
      | None => ("inexact", 0i64)
    }
  } else {
    fraction = units_per_second(unit)
    per = floor_div(nanos_per_second(), fraction)
    base = floor_div(nanosecond, per)
    floor_even = if is_even(fraction) then is_even(base) else is_even(add(euclid_rem(second, 2i64), base))
    match rounding_step(rounding, euclid_rem(nanosecond, per), per, floor_even, lt(second, 0i64)) with {
      | Some(step) => match scaled_sum(second, fraction, add(base, step)) with {
      | Some(count) => ("", count)
      | None => ("range", 0i64)
    }
      | None => ("inexact", 0i64)
    }
  }
}
def count_text(count: i64, unit: TimeUnit) -> string = joined([to_string(count), " ", name_of_unit(unit)])
def instant_count_problem(count: i64, unit: TimeUnit) -> Option[string] =
  match count_parts(count, unit) with {
    | Some(parts) => match unix_second_problem(parts.0) with {
    | Some(_) => Some(joined([count_text(count, unit), " since the unix epoch is outside the supported instant range"]))
    | None => None
  }
    | None => Some(joined([count_text(count, unit), " since the unix epoch is outside the supported instant range"]))
  }
def instant_of_count(count: i64, unit: TimeUnit) -> Instant =
  match count_parts(count, unit) with {
    | Some(parts) => Instant { unix_second: parts.0, nanosecond: parts.1 }
    | None => Instant { unix_second: 0i64, nanosecond: 0i64 }
  }
def instant_from_unix_count(count: i64, unit: TimeUnit) -> Instant =
  match instant_count_problem(count, unit) with {
    | Some(detail) => "instant_from_unix_count" |> domain_failure(detail) |> fail
    | None => instant_of_count(count, unit)
  }
def try_instant_from_unix_count(count: i64, unit: TimeUnit) -> Option[Instant] =
  match instant_count_problem(count, unit) with {
    | Some(_) => None
    | None => count |> instant_of_count(unit) |> Some
  }
def instant_text(second: i64, nanosecond: i64) -> string = joined([civil_text(floor_div(second, seconds_per_day()), add(mul(euclid_rem(second, seconds_per_day()), nanos_per_second()), nanosecond)), "Z"])
def instant_to_unix_count(i: Instant, unit: TimeUnit, rounding: Rounding) -> i64 = {
  (status, count) = parts_count(i.unix_second, i.nanosecond, unit, rounding)
  if eq(status, "") then count else if eq(status, "inexact") then fail(domain_failure("instant_to_unix_count", joined([instant_text(i.unix_second, i.nanosecond), " is not a whole number of ", name_of_unit(unit)]))) else fail(overflow_failure("instant_to_unix_count", joined([instant_text(i.unix_second, i.nanosecond), " in ", name_of_unit(unit), " does not fit in i64"])))
}
def try_instant_to_unix_count(i: Instant, unit: TimeUnit, rounding: Rounding) -> Option[i64] = {
  (status, count) = parts_count(i.unix_second, i.nanosecond, unit, rounding)
  if eq(status, "") then Some(count) else None
}
def instant_add_duration(i: Instant, d: Duration) -> Instant = {
  nanos = add(i.nanosecond, d.nanosecond)
  carry = if gte(nanos, nanos_per_second()) then 1i64 else 0i64
  base = add(i.unix_second, carry)
  if or(gt(d.second, sub(max_unix_second(), base)), lt(d.second, sub(min_unix_second(), base))) then fail(overflow_failure("instant_add_duration", joined([instant_text(i.unix_second, i.nanosecond), " plus ", duration_text(d), " is outside the supported instant range"]))) else Instant { unix_second: add(base, d.second), nanosecond: sub(nanos, mul(carry, nanos_per_second())) }
}
def instant_until(a: Instant, b: Instant) -> Duration = elapsed_between(a.unix_second, a.nanosecond, b.unix_second, b.nanosecond)
def increment_problem(increment: Duration) -> Option[string] = if or(lt(increment.second, 0i64), and(eq(increment.second, 0i64), eq(increment.nanosecond, 0i64))) then (["increment ", duration_text(increment), " is not positive"] |> joined |> Some) else if gt(increment.second, seconds_per_day()) then Some(joined(["increment ", duration_text(increment), " does not divide one day"])) else if neq(mod(86400000000000i64, (increment.second) |> mul(nanos_per_second()) |> add(increment.nanosecond)), 0i64) then Some(joined(["increment ", duration_text(increment), " does not divide one day"])) else None
-- Rounds to a multiple of `increment` counted from the unix epoch. The
-- increment divides one day, so the multiples align with UTC days and the
-- rounding runs within one day's nanoseconds.
def instant_round_to(i: Instant, increment: Duration, rounding: Rounding) -> Instant =
  match increment_problem(increment) with {
    | Some(detail) => fail(domain_failure("instant_round_to", detail))
    | None => {
    step = add(mul(increment.second, nanos_per_second()), increment.nanosecond)
    day = floor_div(i.unix_second, seconds_per_day())
    within = add(mul(euclid_rem(i.unix_second, seconds_per_day()), nanos_per_second()), i.nanosecond)
    whole = floor_div(within, step)
    per_day = floor_div(86400000000000i64, step)
    floor_even = is_even(add(mul(euclid_rem(day, 2i64), euclid_rem(per_day, 2i64)), euclid_rem(whole, 2i64)))
    match rounding_step(rounding, sub(within, mul(whole, step)), step, floor_even, lt(i.unix_second, 0i64)) with {
      | Some(up) => {
      rounded = mul(add(whole, up), step)
      second = add(mul(day, seconds_per_day()), floor_div(rounded, nanos_per_second()))
      if in_span(second, min_unix_second(), max_unix_second()) then Instant { unix_second: second, nanosecond: euclid_rem(rounded, nanos_per_second()) } else fail(overflow_failure("instant_round_to", joined(["rounding ", instant_text(i.unix_second, i.nanosecond), " to a multiple of ", duration_text(increment), " leaves the supported instant range"])))
    }
      | None => fail(domain_failure("instant_round_to", joined([instant_text(i.unix_second, i.nanosecond), " is not a multiple of ", duration_text(increment)])))
    }
  }
  }
def instant_lt(a: Instant, b: Instant) -> bool = reading_before(a.unix_second, a.nanosecond, b.unix_second, b.nanosecond)
def instant_lte(a: Instant, b: Instant) -> bool = not(reading_before(b.unix_second, b.nanosecond, a.unix_second, a.nanosecond))
def instant_gt(a: Instant, b: Instant) -> bool = reading_before(b.unix_second, b.nanosecond, a.unix_second, a.nanosecond)
def instant_gte(a: Instant, b: Instant) -> bool = not(reading_before(a.unix_second, a.nanosecond, b.unix_second, b.nanosecond))
def civil_reading(second: i64, nanosecond: i64, offset: i64) -> DateTime = {
  local = add(second, offset)
  DateTime { epoch_day: floor_div(local, seconds_per_day()), nanosecond_of_day: add(mul(euclid_rem(local, seconds_per_day()), nanos_per_second()), nanosecond) }
}
def instant_to_datetime_at(i: Instant, o: Offset) -> DateTime = civil_reading(i.unix_second, i.nanosecond, o.seconds)
def datetime_to_instant_at(dt: DateTime, o: Offset) -> Instant = {
  second = (dt.epoch_day) |> civil_second(dt.nanosecond_of_day) |> sub(o.seconds)
  if in_span(second, min_unix_second(), max_unix_second()) then Instant { unix_second: second, nanosecond: euclid_rem(dt.nanosecond_of_day, nanos_per_second()) } else fail(overflow_failure("datetime_to_instant_at", joined([civil_text(dt.epoch_day, dt.nanosecond_of_day), " at offset ", offset_text(o.seconds), " is outside the supported instant range"])))
}
def instant_to_string(i: Instant) -> string = instant_text(i.unix_second, i.nanosecond)
-- Instants with the offset they were written in.
def offset_datetime(i: Instant, o: Offset) -> OffsetDateTime = OffsetDateTime { instant: i, offset: o }
def offset_datetime_instant(odt: OffsetDateTime) -> Instant = odt.instant
def offset_datetime_offset(odt: OffsetDateTime) -> Offset = odt.offset
def offset_datetime_local(odt: OffsetDateTime) -> DateTime = civil_reading(odt.instant.unix_second, odt.instant.nanosecond, odt.offset.seconds)
def offset_datetime_to_string(odt: OffsetDateTime) -> string = {
  local = civil_reading(odt.instant.unix_second, odt.instant.nanosecond, odt.offset.seconds)
  string_concat(civil_text(local.epoch_day, local.nanosecond_of_day), offset_text(odt.offset.seconds))
}
-- Exact durations, Euclidean-normalized.
def duration(second: i64, nanosecond: i64) -> Duration = {
  carry = floor_div(nanosecond, nanos_per_second())
  if sum_fits(second, carry) then Duration { second: add(second, carry), nanosecond: euclid_rem(nanosecond, nanos_per_second()) } else fail(domain_failure("duration", joined(["second ", to_string(second), " with nanosecond ", to_string(nanosecond), " is outside the duration range"])))
}
def duration_from_count(count: i64, unit: TimeUnit) -> Duration =
  match count_parts(count, unit) with {
    | Some(parts) => Duration { second: parts.0, nanosecond: parts.1 }
    | None => fail(domain_failure("duration_from_count", joined([count_text(count, unit), " is outside the duration range"])))
  }
def duration_second(d: Duration) -> i64 = d.second
def duration_nanosecond(d: Duration) -> i64 = d.nanosecond
def duration_to_count(d: Duration, unit: TimeUnit, rounding: Rounding) -> i64 = {
  (status, count) = parts_count(d.second, d.nanosecond, unit, rounding)
  if eq(status, "") then count else if eq(status, "inexact") then fail(domain_failure("duration_to_count", joined([duration_text(d), " is not a whole number of ", name_of_unit(unit)]))) else fail(overflow_failure("duration_to_count", joined([duration_text(d), " in ", name_of_unit(unit), " does not fit in i64"])))
}
def try_duration_to_count(d: Duration, unit: TimeUnit, rounding: Rounding) -> Option[i64] = {
  (status, count) = parts_count(d.second, d.nanosecond, unit, rounding)
  if eq(status, "") then Some(count) else None
}
-- The whole part and fraction share a sign, so a small negative duration's
-- terms do not cancel.
def duration_to_seconds_f64(d: Duration) -> f64 = {
  same_sign = or(gte(d.second, 0i64), eq(d.nanosecond, 0i64))
  whole = if same_sign then d.second else add(d.second, 1i64)
  fraction = if same_sign then d.nanosecond else sub(d.nanosecond, nanos_per_second())
  add(cast(whole, f64), div(cast(fraction, f64), 1000000000.0f64))
}
-- `second_a + second_b + carry` for carry 0 or 1, or None when it does not fit.
def seconds_sum(second_a: i64, second_b: i64, carry: i64) -> Option[i64] = if eq(carry, 0i64) then if sum_fits(second_a, second_b) then (second_a |> add(second_b) |> Some) else None else if lt(second_a, 9223372036854775807i64) then if (second_a |> add(1i64) |> sum_fits(second_b)) then Some(add(add(second_a, 1i64), second_b)) else None else if lt(second_b, 9223372036854775807i64) then if sum_fits(second_a, add(second_b, 1i64)) then (second_a |> add(add(second_b, 1i64)) |> Some) else None else None
-- `second_a - second_b - debit` for debit 0 or 1, or None when it does not fit.
def seconds_difference(second_a: i64, second_b: i64, debit: i64) -> Option[i64] = if eq(debit, 0i64) then if difference_fits(second_a, second_b) then (second_a |> sub(second_b) |> Some) else None else if gt(second_a, sub(-9223372036854775807i64, 1i64)) then if (second_a |> sub(1i64) |> difference_fits(second_b)) then Some(sub(sub(second_a, 1i64), second_b)) else None else if lt(second_b, 9223372036854775807i64) then if difference_fits(second_a, add(second_b, 1i64)) then (second_a |> sub(add(second_b, 1i64)) |> Some) else None else None
def duration_add(a: Duration, b: Duration) -> Duration = {
  nanos = add(a.nanosecond, b.nanosecond)
  carry = if gte(nanos, nanos_per_second()) then 1i64 else 0i64
  match seconds_sum(a.second, b.second, carry) with {
    | Some(second) => Duration { second, nanosecond: sub(nanos, mul(carry, nanos_per_second())) }
    | None => fail(overflow_failure("duration_add", joined([duration_text(a), " plus ", duration_text(b), " does not fit in the duration range"])))
  }
}
def duration_sub(a: Duration, b: Duration) -> Duration = {
  nanos = sub(a.nanosecond, b.nanosecond)
  debit = if lt(nanos, 0i64) then 1i64 else 0i64
  match seconds_difference(a.second, b.second, debit) with {
    | Some(second) => Duration { second, nanosecond: add(nanos, mul(debit, nanos_per_second())) }
    | None => fail(overflow_failure("duration_sub", joined([duration_text(a), " minus ", duration_text(b), " does not fit in the duration range"])))
  }
}
def duration_negate(d: Duration) -> Duration = if eq(d.nanosecond, 0i64) then if eq(d.second, sub(-9223372036854775807i64, 1i64)) then fail(overflow_failure("duration_negate", joined(["the negation of ", duration_text(d), " does not fit in the duration range"]))) else Duration { second: neg(d.second), nanosecond: 0i64 } else Duration { second: sub(-1i64, d.second), nanosecond: sub(nanos_per_second(), d.nanosecond) }
-- Exact product. The value splits into a whole part `whole` rounded toward
-- zero and a fraction `numer / 1e9` of the same sign, so `whole * k` is no
-- larger in magnitude than the product; `numer * k` is split through
-- `k = high * 1e9 + low` so no partial product leaves i64.
def duration_mul(d: Duration, k: i64) -> Duration = {
  whole = if ((d.second) |> lt(0i64) |> and(gt(d.nanosecond, 0i64))) then add(d.second, 1i64) else d.second
  numer = if ((d.second) |> lt(0i64) |> and(gt(d.nanosecond, 0i64))) then sub(d.nanosecond, nanos_per_second()) else d.nanosecond
  high = floor_div(k, nanos_per_second())
  low = euclid_rem(k, nanos_per_second())
  partial = mul(numer, low)
  fraction_seconds =
    numer
    |> mul(high)
    |> add(floor_div(partial, nanos_per_second()))
  if product_fits(whole, k) then if (whole |> mul(k) |> sum_fits(fraction_seconds)) then Duration { second: whole |> mul(k) |> add(fraction_seconds), nanosecond: euclid_rem(partial, nanos_per_second()) } else (d |> duration_mul_failure(k) |> fail) else (d |> duration_mul_failure(k) |> fail)
}
def duration_mul_failure(d: Duration, k: i64) -> string = overflow_failure("duration_mul", joined([duration_text(d), " times ", to_string(k), " does not fit in the duration range"]))
def duration_lt(a: Duration, b: Duration) -> bool = reading_before(a.second, a.nanosecond, b.second, b.nanosecond)
def duration_lte(a: Duration, b: Duration) -> bool = (b.second) |> reading_before(b.nanosecond, a.second, a.nanosecond) |> not
def duration_gt(a: Duration, b: Duration) -> bool = reading_before(b.second, b.nanosecond, a.second, a.nanosecond)
def duration_gte(a: Duration, b: Duration) -> bool = (a.second) |> reading_before(a.nanosecond, b.second, b.nanosecond) |> not
def duration_to_string(d: Duration) -> string = duration_text(d)
-- Calendar periods: months and days, never of mixed sign.
def mixed_signs(months: i64, days: i64) -> bool = or(and(gt(months, 0i64), lt(days, 0i64)), and(lt(months, 0i64), gt(days, 0i64)))
def period(months: i64, days: i64) -> Period = if mixed_signs(months, days) then fail(domain_failure("period", joined(["months ", to_string(months), " and days ", to_string(days), " have mixed signs"]))) else Period { months, days }
def try_period(months: i64, days: i64) -> Option[Period] = if mixed_signs(months, days) then None else Some(Period { months, days })
def period_months(p: Period) -> i64 = p.months
def period_days(p: Period) -> i64 = p.days
def period_text(months: i64, days: i64) -> string = {
  body = if (months |> eq(0i64) |> and(eq(days, 0i64))) then "0D" else string_concat(if eq(months, 0i64) then "" else (months |> magnitude_text |> string_concat("M")), if eq(days, 0i64) then "" else (days |> magnitude_text |> string_concat("D")))
  string_concat(if (months |> lt(0i64) |> or(lt(days, 0i64))) then "-P" else "P", body)
}
def period_negate(p: Period) -> Period =
  if ((p.months)
  |> eq(sub(-9223372036854775807i64, 1i64))
  |> or(eq(p.days, sub(-9223372036854775807i64, 1i64)))) then fail(overflow_failure("period_negate", joined(["the negation of ", period_text(p.months, p.days), " does not fit in i64"]))) else Period { months: neg(p.months), days: neg(p.days) }
def period_mul(p: Period, k: i64) -> Period = if ((p.months) |> product_fits(k) |> and(product_fits(p.days, k))) then Period { months: mul(p.months, k), days: mul(p.days, k) } else fail(overflow_failure("period_mul", joined([period_text(p.months, p.days), " times ", to_string(k), " does not fit in i64"])))
def period_to_string(p: Period) -> string = period_text(p.months, p.days)
-- Text profile parsing. Every index is checked against the text's length
-- before a slice, and digit runs are folded, never recursed over.
def char_at(text: string, idx: i64) -> string = if (idx |> gte(0i64) |> and(lt(idx, string_len(text)))) then string_slice(text, idx, 1i64) else ""
def digit_at(text: string, idx: i64) -> i64 = {
  ch = char_at(text, idx)
  if eq(ch, "") then -1i64 else {
    code = char_code(ch)
    if in_span(code, 48i64, 57i64) then sub(code, 48i64) else -1i64
  }
}
-- The value of exactly `count <= 9` digits at `start`, or -1.
def fixed_digits(text: string, start: i64, count: i64) -> i64 = fold(fn (acc: i64, k: i64) -> if lt(acc, 0i64) then acc else if (text |> digit_at(add(start, k)) |> lt(0i64)) then -1i64 else (acc |> mul(10i64) |> add(digit_at(text, add(start, k)))), 0i64, range(0i64, count))
-- The index of the first non-digit at or after `start`.
def digit_run_end(text: string, start: i64) -> i64 = fold(fn (acc: i64, idx: i64) -> if (acc |> eq(idx) |> and(gte(digit_at(text, idx), 0i64))) then add(idx, 1i64) else acc, start, range(start, string_len(text)))
-- The negated value of the digits in [start, end), accumulated toward the
-- i64 minimum so that magnitude 2^63 is exact; None when it does not fit.
-- The fold carries (value, still fits) rather than an Option so the compiled
-- lane resolves its accumulator type (chelis#2599).
def negated_digits(text: string, start: i64, end: i64) -> Option[i64] = {
  folded = fold(fn (acc: (i64, bool), idx: i64) -> if not(acc.1) then acc else if gte(acc.0, floor_div(add(add(sub(-9223372036854775807i64, 1i64), digit_at(text, idx)), 9i64), 10i64)) then (sub(mul(acc.0, 10i64), digit_at(text, idx)), true) else (0i64, false), (0i64, true), range(start, end))
  if folded.1 then Some(folded.0) else None
}
-- Syntax of `year "-" 2DIGIT "-" 2DIGIT` at `start`: (year, month, day, end).
def year_syntax(text: string, start: i64) -> Option[(i64, i64)] = {
  sign = char_at(text, start)
  if (sign |> eq("+") |> or(eq(sign, "-"))) then {
    size = fixed_digits(text, add(start, 1i64), 6i64)
    if (size |> lt(0i64) |> or(and(eq(sign, "-"), eq(size, 0i64)))) then None else Some((if eq(sign, "-") then neg(size) else size, add(start, 7i64)))
  } else {
    value = fixed_digits(text, start, 4i64)
    if lt(value, 0i64) then None else Some((value, add(start, 4i64)))
  }
}
def date_syntax(text: string, start: i64) -> Option[(i64, i64, i64, i64)] =
  match year_syntax(text, start) with {
    | Some(found) => {
    (year, at) = found
    month = fixed_digits(text, add(at, 1i64), 2i64)
    day = fixed_digits(text, add(at, 4i64), 2i64)
    if and(and(eq(char_at(text, at), "-"), text |> char_at(add(at, 3i64)) |> eq("-")), month |> gte(0i64) |> and(gte(day, 0i64))) then Some((year, month, day, add(at, 6i64))) else None
  }
    | None => None
  }
-- Syntax of `2DIGIT ":" 2DIGIT ":" 2DIGIT [ "." 1*9DIGIT ]` at `start`:
-- (hour, minute, second, nanosecond, end).
def clock_syntax(text: string, start: i64) -> Option[(i64, i64, i64, i64, i64)] = {
  hour = fixed_digits(text, start, 2i64)
  minute = fixed_digits(text, add(start, 3i64), 2i64)
  second = fixed_digits(text, add(start, 6i64), 2i64)
  if and(and(and(gte(hour, 0i64), gte(minute, 0i64)), gte(second, 0i64)), and(eq(char_at(text, add(start, 2i64)), ":"), text |> char_at(add(start, 5i64)) |> eq(":"))) then if (text |> char_at(add(start, 8i64)) |> eq(".")) then {
    digits_end = digit_run_end(text, add(start, 9i64))
    count = sub(digits_end, add(start, 9i64))
    if in_span(count, 1i64, 9i64) then Some((hour, minute, second, mul(fixed_digits(text, add(start, 9i64), count), fixed_digits("1000000000", 0i64, sub(10i64, count))), digits_end)) else None
  } else Some((hour, minute, second, 0i64, add(start, 8i64))) else None
}
-- Syntax of an offset at `start`: (sign, hour, minute, second, end).
def offset_syntax(text: string, start: i64) -> Option[(i64, i64, i64, i64, i64)] = {
  sign = char_at(text, start)
  if (sign |> eq("Z") |> or(eq(sign, "z"))) then Some((1i64, 0i64, 0i64, 0i64, add(start, 1i64))) else if (sign |> eq("+") |> or(eq(sign, "-"))) then {
    hour = fixed_digits(text, add(start, 1i64), 2i64)
    minute = fixed_digits(text, add(start, 4i64), 2i64)
    if and(and(gte(hour, 0i64), gte(minute, 0i64)), eq(char_at(text, add(start, 3i64)), ":")) then if (text |> char_at(add(start, 6i64)) |> eq(":")) then {
      second = fixed_digits(text, add(start, 7i64), 2i64)
      if gte(second, 0i64) then Some((if eq(sign, "-") then -1i64 else 1i64, hour, minute, second, add(start, 9i64))) else None
    } else Some((if eq(sign, "-") then -1i64 else 1i64, hour, minute, 0i64, add(start, 6i64))) else None
  } else None
}
def offset_field_problem(hour: i64, minute: i64, second: i64) -> Option[string] = if (hour |> in_span(0i64, 23i64) |> not) then ("offset hour" |> outside_text(hour, 0i64, 23i64) |> Some) else if (minute |> in_span(0i64, 59i64) |> not) then ("offset minute" |> outside_text(minute, 0i64, 59i64) |> Some) else if (second |> in_span(0i64, 59i64) |> not) then ("offset second" |> outside_text(second, 0i64, 59i64) |> Some) else None
def syntax_detail(text: string, form: string) -> string = joined([quoted(text), " is not ", form, " in the text profile"])
def is_separator(text: string, idx: i64) -> bool = or(or(eq(char_at(text, idx), "T"), text |> char_at(idx) |> eq("t")), text |> char_at(idx) |> eq(" "))
-- Each `*_reading` parses a whole text into (problem, value fields): the
-- problem is empty exactly when the text is valid. Field-range problems are
-- reported after the grammar matches.
def date_reading(text: string) -> (string, i64) =
  match date_syntax(text, 0i64) with {
    | Some(found) => {
    (year, month, day, end) = found
    if neq(end, string_len(text)) then (syntax_detail(text, "a date"), 0i64) else match ymd_problem(year, month, day) with {
      | Some(detail) => (detail, 0i64)
      | None => ("", days_from_civil(year, month, day))
    }
  }
    | None => (syntax_detail(text, "a date"), 0i64)
  }
def parse_date(text: string) -> Date = {
  (problem, epoch_day) = date_reading(text)
  if eq(problem, "") then Date { epoch_day } else ("parse_date" |> domain_failure(problem) |> fail)
}
def try_parse_date(text: string) -> Option[Date] = {
  (problem, epoch_day) = date_reading(text)
  if eq(problem, "") then Some(Date { epoch_day }) else None
}
def clock_reading(text: string) -> (string, i64) =
  match clock_syntax(text, 0i64) with {
    | Some(found) => {
    (hour, minute, second, nanosecond, end) = found
    if neq(end, string_len(text)) then (syntax_detail(text, "a time"), 0i64) else match clock_problem(hour, minute, second, nanosecond) with {
      | Some(detail) => (detail, 0i64)
      | None => ("", clock_nanos(hour, minute, second, nanosecond))
    }
  }
    | None => (syntax_detail(text, "a time"), 0i64)
  }
def parse_time(text: string) -> Time = {
  (problem, nanosecond_of_day) = clock_reading(text)
  if eq(problem, "") then Time { nanosecond_of_day } else ("parse_time" |> domain_failure(problem) |> fail)
}
def try_parse_time(text: string) -> Option[Time] = {
  (problem, nanosecond_of_day) = clock_reading(text)
  if eq(problem, "") then Some(Time { nanosecond_of_day }) else None
}
-- A civil datetime at the start of `text`: (grammar matched, field problem,
-- epoch day, nanosecond of day, end). The field problem is empty exactly when
-- every field is in range.
def civil_reading_at(text: string) -> (bool, string, i64, i64, i64) =
  match date_syntax(text, 0i64) with {
    | Some(found_date) => {
    (year, month, day, at) = found_date
    if (text |> is_separator(at) |> not) then (false, "", 0i64, 0i64, 0i64) else match clock_syntax(text, add(at, 1i64)) with {
      | Some(found_time) => {
      (hour, minute, second, nanosecond, end) = found_time
      match ymd_problem(year, month, day) with {
        | Some(detail) => (true, detail, 0i64, 0i64, end)
        | None => match clock_problem(hour, minute, second, nanosecond) with {
        | Some(detail) => (true, detail, 0i64, 0i64, end)
        | None => (true, "", days_from_civil(year, month, day), clock_nanos(hour, minute, second, nanosecond), end)
      }
      }
    }
      | None => (false, "", 0i64, 0i64, 0i64)
    }
  }
    | None => (false, "", 0i64, 0i64, 0i64)
  }
def datetime_reading(text: string) -> (string, i64, i64) = {
  (matched, problem, epoch_day, nanosecond_of_day, end) = civil_reading_at(text)
  if (matched |> not |> or(neq(end, string_len(text)))) then (syntax_detail(text, "a datetime"), 0i64, 0i64) else (problem, epoch_day, nanosecond_of_day)
}
def parse_datetime(text: string) -> DateTime = {
  (problem, epoch_day, nanosecond_of_day) = datetime_reading(text)
  if eq(problem, "") then DateTime { epoch_day, nanosecond_of_day } else ("parse_datetime" |> domain_failure(problem) |> fail)
}
def try_parse_datetime(text: string) -> Option[DateTime] = {
  (problem, epoch_day, nanosecond_of_day) = datetime_reading(text)
  if eq(problem, "") then Some(DateTime { epoch_day, nanosecond_of_day }) else None
}
-- A whole offset text: (problem, seconds).
def offset_reading(text: string) -> (string, i64) =
  match offset_syntax(text, 0i64) with {
    | Some(found) => {
    (sign, hour, minute, second, end) = found
    if neq(end, string_len(text)) then (syntax_detail(text, "an offset"), 0i64) else match offset_field_problem(hour, minute, second) with {
      | Some(detail) => (detail, 0i64)
      | None => ("", mul(sign, add(add(mul(hour, 3600i64), mul(minute, 60i64)), second)))
    }
  }
    | None => (syntax_detail(text, "an offset"), 0i64)
  }
def parse_offset(text: string) -> Offset = {
  (problem, seconds) = offset_reading(text)
  if eq(problem, "") then Offset { seconds } else ("parse_offset" |> domain_failure(problem) |> fail)
}
def try_parse_offset(text: string) -> Option[Offset] = {
  (problem, seconds) = offset_reading(text)
  if eq(problem, "") then Some(Offset { seconds }) else None
}
-- `datetime offset`: (problem, unix second, nanosecond, offset seconds). A
-- text outside the grammar is reported first, then the datetime's fields,
-- then the offset's fields, then the instant range.
def instant_reading(text: string, form: string) -> (string, i64, i64, i64) = {
  (matched, problem, epoch_day, nanosecond_of_day, end) = civil_reading_at(text)
  if not(matched) then (syntax_detail(text, form), 0i64, 0i64, 0i64) else match offset_syntax(text, end) with {
    | Some(found) => {
    (sign, hour, minute, second, offset_end) = found
    if neq(offset_end, string_len(text)) then (syntax_detail(text, form), 0i64, 0i64, 0i64) else if neq(problem, "") then (problem, 0i64, 0i64, 0i64) else match offset_field_problem(hour, minute, second) with {
      | Some(detail) => (detail, 0i64, 0i64, 0i64)
      | None => {
      seconds = mul(sign, add(add(mul(hour, 3600i64), mul(minute, 60i64)), second))
      unix_second = epoch_day |> civil_second(nanosecond_of_day) |> sub(seconds)
      match unix_second_problem(unix_second) with {
        | Some(detail) => (detail, 0i64, 0i64, 0i64)
        | None => ("", unix_second, euclid_rem(nanosecond_of_day, nanos_per_second()), seconds)
      }
    }
    }
  }
    | None => (syntax_detail(text, form), 0i64, 0i64, 0i64)
  }
}
def parse_instant(text: string) -> Instant = {
  (problem, second, nanosecond, _) = instant_reading(text, "an instant")
  if eq(problem, "") then Instant { unix_second: second, nanosecond } else ("parse_instant" |> domain_failure(problem) |> fail)
}
def try_parse_instant(text: string) -> Option[Instant] = {
  (problem, second, nanosecond, _) = instant_reading(text, "an instant")
  if eq(problem, "") then Some(Instant { unix_second: second, nanosecond }) else None
}
def parse_offset_datetime(text: string) -> OffsetDateTime = {
  (problem, second, nanosecond, seconds) = instant_reading(text, "an offset datetime")
  if eq(problem, "") then OffsetDateTime { instant: Instant { unix_second: second, nanosecond }, offset: Offset { seconds } } else ("parse_offset_datetime" |> domain_failure(problem) |> fail)
}
def try_parse_offset_datetime(text: string) -> Option[OffsetDateTime] = {
  (problem, second, nanosecond, seconds) = instant_reading(text, "an offset datetime")
  if eq(problem, "") then Some(OffsetDateTime { instant: Instant { unix_second: second, nanosecond }, offset: Offset { seconds } }) else None
}
-- A designated component `1*DIGIT <letter>` at `start`: (digits start, digits
-- end, next index), or None when it is absent.
def component_at(text: string, start: i64, letter: string) -> Option[(i64, i64, i64)] = {
  end = digit_run_end(text, start)
  if (end |> gt(start) |> and(eq(char_at(text, end), letter))) then Some((start, end, add(end, 1i64))) else None
}
-- The negated value of an optional component, or Some(0) when absent; None on
-- i64 overflow.
def component_value(text: string, found: Option[(i64, i64, i64)]) -> Option[i64] =
  match found with {
    | Some(span) => negated_digits(text, span.0, span.1)
    | None => Some(0i64)
  }
def component_next(found: Option[(i64, i64, i64)], start: i64) -> i64 =
  match found with {
    | Some(span) => span.2
    | None => start
  }
def is_present(found: Option[(i64, i64, i64)]) -> bool =
  match found with {
    | Some(_) => true
    | None => false
  }
-- `scale * a + b` for nonpositive operands, or None when it leaves i64.
def negated_combination(a: Option[i64], scale: i64, b: Option[i64]) -> Option[i64] =
  match a with {
    | Some(x) => match b with {
    | Some(y) => if product_fits(x, scale) then if (x |> mul(scale) |> sum_fits(y)) then Some(add(mul(x, scale), y)) else None else None
    | None => None
  }
    | None => None
  }
-- (status, second, nanosecond): status is "" when valid, "syntax" for text
-- outside the grammar, and "range" for a value outside the duration range.
def duration_reading(text: string) -> (string, i64, i64) = {
  negative = text |> char_at(0i64) |> eq("-")
  start = if negative then 1i64 else 0i64
  if not(and(eq(char_at(text, start), "P"), text |> char_at(add(start, 1i64)) |> eq("T"))) then ("syntax", 0i64, 0i64) else {
    hours = component_at(text, add(start, 2i64), "H")
    after_hours = component_next(hours, add(start, 2i64))
    minutes = component_at(text, after_hours, "M")
    after_minutes = component_next(minutes, after_hours)
    seconds_end = digit_run_end(text, after_minutes)
    has_fraction =
      seconds_end
      |> gt(after_minutes)
      |> and(eq(char_at(text, seconds_end), "."))
    fraction_end = if has_fraction then digit_run_end(text, add(seconds_end, 1i64)) else seconds_end
    fraction_count = if has_fraction then sub(fraction_end, add(seconds_end, 1i64)) else 0i64
    has_seconds =
      seconds_end
      |> gt(after_minutes)
      |> and(eq(char_at(text, fraction_end), "S"))
    end = if has_seconds then add(fraction_end, 1i64) else after_minutes
    shape_ok = and(and(neq(end, add(start, 2i64)), eq(end, string_len(text))), or(not(has_fraction), and(has_seconds, in_span(fraction_count, 1i64, 9i64))))
    if not(shape_ok) then ("syntax", 0i64, 0i64) else {
      whole_seconds = if has_seconds then negated_digits(text, after_minutes, seconds_end) else Some(0i64)
      total = negated_combination(negated_combination(component_value(text, hours), 60i64, component_value(text, minutes)), 60i64, whole_seconds)
      fraction = if and(has_fraction, gt(fraction_count, 0i64)) then mul(fixed_digits(text, add(seconds_end, 1i64), fraction_count), fixed_digits("1000000000", 0i64, sub(10i64, fraction_count))) else 0i64
      match total with {
        | Some(negated) => if negative then if eq(fraction, 0i64) then ("", negated, 0i64) else if gt(negated, sub(-9223372036854775807i64, 1i64)) then ("", sub(negated, 1i64), sub(nanos_per_second(), fraction)) else ("range", 0i64, 0i64) else if gt(negated, sub(-9223372036854775807i64, 1i64)) then ("", neg(negated), fraction) else ("range", 0i64, 0i64)
        | None => ("range", 0i64, 0i64)
      }
    }
  }
}
def parse_duration(text: string) -> Duration = {
  (status, second, nanosecond) = duration_reading(text)
  if eq(status, "") then Duration { second, nanosecond } else fail(domain_failure("parse_duration", text_problem(text, status, "a duration", "duration")))
}
def try_parse_duration(text: string) -> Option[Duration] = {
  (status, second, nanosecond) = duration_reading(text)
  if eq(status, "") then Some(Duration { second, nanosecond }) else None
}
-- (status, months, days), as for `duration_reading`.
def period_reading(text: string) -> (string, i64, i64) = {
  negative = text |> char_at(0i64) |> eq("-")
  start = if negative then 1i64 else 0i64
  if not(eq(char_at(text, start), "P")) then ("syntax", 0i64, 0i64) else {
    years = component_at(text, add(start, 1i64), "Y")
    after_years = component_next(years, add(start, 1i64))
    months = component_at(text, after_years, "M")
    after_months = component_next(months, after_years)
    weeks = component_at(text, after_months, "W")
    after_weeks = component_next(weeks, after_months)
    days = component_at(text, after_weeks, "D")
    end = component_next(days, after_weeks)
    if (end |> eq(add(start, 1i64)) |> or(neq(end, string_len(text)))) then ("syntax", 0i64, 0i64) else {
      total_months = negated_combination(component_value(text, years), 12i64, component_value(text, months))
      total_days = negated_combination(component_value(text, weeks), 7i64, component_value(text, days))
      match total_months with {
        | Some(month_count) => match total_days with {
        | Some(day_count) => if negative then ("", month_count, day_count) else if (month_count
      |> gt(sub(-9223372036854775807i64, 1i64))
      |> and(gt(day_count, sub(-9223372036854775807i64, 1i64)))) then ("", neg(month_count), neg(day_count)) else ("range", 0i64, 0i64)
        | None => ("range", 0i64, 0i64)
      }
        | None => ("range", 0i64, 0i64)
      }
    }
  }
}
def parse_period(text: string) -> Period = {
  (status, months, days) = period_reading(text)
  if eq(status, "") then Period { months, days } else fail(domain_failure("parse_period", text_problem(text, status, "a period", "period")))
}
def try_parse_period(text: string) -> Option[Period] = {
  (status, months, days) = period_reading(text)
  if eq(status, "") then Some(Period { months, days }) else None
}
-- The detail for a duration or period text that denotes no value: outside the
-- grammar, or a whole value outside the type's range.
def text_problem(text: string, status: string, form: string, type_name: string) -> string = if eq(status, "range") then joined([quoted(text), " is outside the ", type_name, " range"]) else syntax_detail(text, form)
-- Columns. A constructor consumes the tensors it stores; an accessor consumes
-- the column and returns its storage.
def filled[n](like: &tensor[n, i64], value: i64) -> tensor[n, i64] = value |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))
-- The index of the first false element, or -1.
def first_false(flags: List[bool]) -> i64 = fold(fn (acc: (i64, i64), flag: bool) -> if gte(acc.1, 0i64) then acc else if flag then (add(acc.0, 1i64), -1i64) else (add(acc.0, 1i64), acc.0), (0i64, -1i64), flags).1
def epoch_day_mask[n](t: &tensor[n, i64]) -> tensor[n, bool] = and(gte(t, filled(t, min_epoch_day())), lte(t, filled(t, max_epoch_day())))
def dates_from_epoch_days[n](t: tensor[n, i64]) -> Dates[n] = {
  bad = first_false(to_list(epoch_day_mask(t)))
  if gte(bad, 0i64) then fail(domain_failure("dates_from_epoch_days", joined(["element ", to_string(bad), ": ", outside_text("epoch day", t |> to_list |> index(bad), min_epoch_day(), max_epoch_day())]))) else Dates { epoch_days: t }
}
def try_dates_from_epoch_days[n](t: tensor[n, i64]) -> (Dates[n], tensor[n, bool]) = {
  mask = epoch_day_mask(t)
  (Dates { epoch_days: where(mask, t, filled(t, 0i64)) }, mask)
}
def dates_epoch_days[n](ds: Dates[n]) -> tensor[n, i64] = ds.epoch_days
def instant_mask[n](seconds: &tensor[n, i64], nanoseconds: &tensor[n, i64]) -> tensor[n, bool] = and(and(gte(seconds, filled(seconds, min_unix_second())), lte(seconds, filled(seconds, max_unix_second()))), and(gte(nanoseconds, filled(nanoseconds, 0i64)), lte(nanoseconds, filled(nanoseconds, 999999999i64))))
def instants_from_unix[n](seconds: tensor[n, i64], nanoseconds: tensor[n, i64]) -> Instants[n] = {
  bad = first_false(to_list(instant_mask(seconds, nanoseconds)))
  if gte(bad, 0i64) then fail(domain_failure("instants_from_unix", joined(["element ", to_string(bad), ": ", element_instant_problem(index(to_list(seconds), bad), index(to_list(nanoseconds), bad))]))) else Instants { unix_seconds: seconds, nanoseconds }
}
def element_instant_problem(second: i64, nanosecond: i64) -> string =
  match instant_problem(second, nanosecond) with {
    | Some(detail) => detail
    | None => ""
  }
def try_instants_from_unix[n](seconds: tensor[n, i64], nanoseconds: tensor[n, i64]) -> (Instants[n], tensor[n, bool]) = {
  mask = instant_mask(seconds, nanoseconds)
  (Instants { unix_seconds: where(mask, seconds, filled(seconds, 0i64)), nanoseconds: where(mask, nanoseconds, filled(nanoseconds, 0i64)) }, mask)
}
def instants_unix_seconds[n](column: Instants[n]) -> tensor[n, i64] = column.unix_seconds
def instants_nanoseconds[n](column: Instants[n]) -> tensor[n, i64] = column.nanoseconds
