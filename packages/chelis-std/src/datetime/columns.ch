module Std.Datetime.Columns
export (Durations, durations, try_durations, durations_seconds, durations_nanoseconds, dates_year, dates_month, dates_day, dates_weekday_iso_number, dates_day_of_year, dates_from_ymd, try_dates_from_ymd, dates_add_days, dates_add_months, dates_days_until, dates_lt, dates_lte, dates_gt, dates_gte, try_parse_dates, dates_to_strings, instants_from_unix_count, instants_to_unix_count, instants_add_duration, instants_until, instants_round_to, instants_to_dates_at, instants_seconds_since_f64, instants_lt, instants_lte, instants_gt, instants_gte)
import Std.Datetime (DayOverflow, ClampToMonthEnd, RejectInvalidDay, TimeUnit, Hours, Minutes, Seconds, Milliseconds, Microseconds, Nanoseconds, Date, Instant, Offset, Duration, Dates, Instants, days_in_month, date, date_epoch_day, date_from_epoch_day, date_to_string, try_parse_date, instant_from_unix, instant_unix_second, instant_nanosecond, instant_to_string, offset_seconds, duration, duration_second, duration_nanosecond, duration_to_string, dates_from_epoch_days, dates_epoch_days, instants_from_unix, instants_unix_seconds, instants_nanoseconds)
import Std.Rounding (Rounding, RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)
-- Std.Datetime.Columns: vectorized forms of Std.Datetime over `Dates[n]`
-- and `Instants[n]`, and the `Durations[n]` column, governed by [05-OP-73].
-- Every kernel composes i64 tensor primitives, and every value a kernel
-- passes to a trapping primitive is in range or replaced first, so no
-- primitive numeric trap escapes a call. A call fails as a whole, naming
-- the lowest failing element. The masked `try_` forms instead mark an
-- element that denotes no value in their mask; they fail only when their
-- tensor arguments differ in length. Columns are consumed; tensor arguments
-- that are only read are borrowed.
@opaque
type Durations[n] =
  | Durations { seconds: tensor[n, i64], nanoseconds: tensor[n, i64] }
-- Range constants are written inline: epoch days
-- -4371587..2932896, total months -119988..119999, unix seconds
-- -377705030401..253402214400, 1000000000 nanoseconds per second, and 86400
-- seconds per day.
-- Text assembly.
def joined(parts: List[string]) -> string = fold(fn (acc: string, part: string) -> string_concat(acc, part), "", parts)
def element_failure(function: string, kind: string, position: i64, detail: string) -> string = joined([function, ": ", kind, ": element ", to_string(position), ": ", detail])
def outside_text(name: string, value: i64, low: i64, high: i64) -> string = joined([name, " ", to_string(value), " is outside ", to_string(low), "..", to_string(high)])
def in_span(value: i64, low: i64, high: i64) -> bool = value |> gte(low) |> and(lte(value, high))
def day_text(epoch_day: i64) -> string = epoch_day |> date_from_epoch_day |> date_to_string
def year_month_text(year: i64, month: i64) -> string = {
  text = date_to_string(date(year, month, 1i64))
  string_slice(text, 0i64, text |> string_len |> sub(3i64))
}
def instant_text(second: i64, nanosecond: i64) -> string = second |> instant_from_unix(nanosecond) |> instant_to_string
def duration_text(second: i64, nanosecond: i64) -> string = second |> duration(nanosecond) |> duration_to_string
-- A column argument whose length differs from another argument's fails
-- `domain` before any element is read. Tensor arguments that share a
-- dimension are already equal in length: spec/04 §4.7's entry guards check
-- them before the body runs.
def length_failure[n, m](function: string, first: &tensor[n, i64], second: &tensor[m, i64]) -> string = joined([function, ": domain: arguments have ", to_string(shape(first, 0i32)), " and ", to_string(shape(second, 0i32)), " elements"])
def lengths_differ[n, m](first: &tensor[n, i64], second: &tensor[m, i64]) -> bool = neq(shape(first, 0i32), shape(second, 0i32))
-- Element access for failure details, and the index of the first false
-- flag, or -1.
def element_at[n](t: &tensor[n, i64], position: i64) -> i64 = t |> to_list |> index(position)
def first_false(flags: List[bool]) -> i64 = fold(fn (acc: (i64, i64), flag: bool) -> if gte(acc.1, 0i64) then acc else if flag then (add(acc.0, 1i64), -1i64) else (add(acc.0, 1i64), acc.0), (0i64, -1i64), flags).1
-- Constant columns and Euclidean arithmetic with a scalar operand.
def filled[n](like: &tensor[n, i64], value: i64) -> tensor[n, i64] = value |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))
def flags_filled[n](like: &tensor[n, i64], value: bool) -> tensor[n, bool] = value |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))
def plus[n](t: &tensor[n, i64], value: i64) -> tensor[n, i64] = add(t, filled(t, value))
def times[n](t: &tensor[n, i64], value: i64) -> tensor[n, i64] = mul(t, filled(t, value))
def floored[n](t: &tensor[n, i64], value: i64) -> tensor[n, i64] = floor_div(t, filled(t, value))
-- The Euclidean remainder by a positive `value`. A truncated quotient times
-- `value` never exceeds `t` in magnitude, so no step overflows at any `t`.
def euclid_mod[n](t: &tensor[n, i64], value: i64) -> tensor[n, i64] = {
  r = sub(t, times(trunc_div(t, filled(t, value)), value))
  where(lt(r, filled(r, 0i64)), plus(r, value), r)
}
def within[n](t: &tensor[n, i64], low: i64, high: i64) -> tensor[n, bool] = and(gte(t, filled(t, low)), lte(t, filled(t, high)))
def ones_where[n](flags: &tensor[n, bool], like: &tensor[n, i64]) -> tensor[n, i64] = where(flags, filled(like, 1i64), filled(like, 0i64))
def divisible[n](t: &tensor[n, i64], value: i64) -> tensor[n, bool] = t |> euclid_mod(value) |> eq(filled(t, 0i64))
def evens[n](t: &tensor[n, i64]) -> tensor[n, bool] = t |> euclid_mod(2i64) |> eq(filled(t, 0i64))
-- Hinnant's days_from_civil and civil_from_days over columns, with floor
-- division and Euclidean remainders, for fields already in range.
def epoch_day_columns[n](year: &tensor[n, i64], month: &tensor[n, i64], day: &tensor[n, i64]) -> tensor[n, i64] = {
  early = lte(month, filled(month, 2i64))
  shifted = where(early, plus(year, -1i64), year)
  era = floored(shifted, 400i64)
  yoe = sub(shifted, times(era, 400i64))
  mp = where(early, plus(month, 9i64), plus(month, -3i64))
  doy = add(floored(plus(times(mp, 153i64), 2i64), 5i64), plus(day, -1i64))
  doe = add(sub(add(times(yoe, 365i64), floored(yoe, 4i64)), floored(yoe, 100i64)), doy)
  plus(add(times(era, 146097i64), doe), -719468i64)
}
def civil_columns[n](epoch_days: &tensor[n, i64]) -> (tensor[n, i64], tensor[n, i64], tensor[n, i64]) = {
  shifted = plus(epoch_days, 719468i64)
  era = floored(shifted, 146097i64)
  doe = sub(shifted, times(era, 146097i64))
  yoe = floored(sub(add(sub(doe, floored(doe, 1460i64)), floored(doe, 36524i64)), floored(doe, 146096i64)), 365i64)
  doy = sub(doe, sub(add(times(yoe, 365i64), floored(yoe, 4i64)), floored(yoe, 100i64)))
  mp = floored(plus(times(doy, 5i64), 2i64), 153i64)
  day = plus(sub(doy, floored(plus(times(mp, 153i64), 2i64), 5i64)), 1i64)
  month = where(lt(mp, filled(mp, 10i64)), plus(mp, 3i64), plus(mp, -9i64))
  year = add(add(yoe, times(era, 400i64)), ones_where(lte(month, filled(month, 2i64)), month))
  (year, month, day)
}
-- The length of each month; any year and any month 1..12.
def month_lengths[n](year: &tensor[n, i64], month: &tensor[n, i64]) -> tensor[n, i64] = {
  leap = and(divisible(year, 4i64), or(not(divisible(year, 100i64)), divisible(year, 400i64)))
  short = or(or(eq(month, filled(month, 4i64)), eq(month, filled(month, 6i64))), or(eq(month, filled(month, 9i64)), eq(month, filled(month, 11i64))))
  where(eq(month, filled(month, 2i64)), where(leap, filled(month, 29i64), filled(month, 28i64)), where(short, filled(month, 30i64), filled(month, 31i64)))
}
-- Durations: Euclidean-normalized like `Duration`.
def duration_mask[n](seconds: &tensor[n, i64], nanoseconds: &tensor[n, i64]) -> tensor[n, bool] = {
  carry = floored(nanoseconds, 1000000000i64)
  raising = gte(carry, filled(carry, 0i64))
  upper = sub(filled(carry, 9223372036854775807i64), where(raising, carry, filled(carry, 0i64)))
  lower = sub(filled(carry, sub(-9223372036854775807i64, 1i64)), where(raising, filled(carry, 0i64), carry))
  where(raising, lte(seconds, upper), gte(seconds, lower))
}
def normalized[n](seconds: &tensor[n, i64], nanoseconds: &tensor[n, i64]) -> Durations[n] = Durations { seconds: add(seconds, floored(nanoseconds, 1000000000i64)), nanoseconds: euclid_mod(nanoseconds, 1000000000i64) }
def durations[n](seconds: &tensor[n, i64], nanoseconds: &tensor[n, i64]) -> Durations[n] = {
  bad = first_false(to_list(duration_mask(seconds, nanoseconds)))
  if gte(bad, 0i64) then fail(element_failure("durations", "domain", bad, joined(["second ", to_string(element_at(seconds, bad)), " with nanosecond ", to_string(element_at(nanoseconds, bad)), " is outside the duration range"]))) else normalized(seconds, nanoseconds)
}
def try_durations[n](seconds: &tensor[n, i64], nanoseconds: &tensor[n, i64]) -> (Durations[n], tensor[n, bool]) = {
  valid = duration_mask(seconds, nanoseconds)
  (normalized(where(valid, seconds, filled(seconds, 0i64)), where(valid, nanoseconds, filled(nanoseconds, 0i64))), valid)
}
def duration_parts[n](column: Durations[n]) -> (tensor[n, i64], tensor[n, i64]) =
  match column with {
    | Durations { seconds, nanoseconds } => (seconds, nanoseconds)
  }
def durations_seconds[n](column: Durations[n]) -> tensor[n, i64] = column.seconds
def durations_nanoseconds[n](column: Durations[n]) -> tensor[n, i64] = column.nanoseconds
-- Date fields.
def dates_year[n](ds: Dates[n]) -> tensor[n, i64] = {
  (year, _, _) = civil_columns(dates_epoch_days(ds))
  year
}
def dates_month[n](ds: Dates[n]) -> tensor[n, i64] = {
  (_, month, _) = civil_columns(dates_epoch_days(ds))
  month
}
def dates_day[n](ds: Dates[n]) -> tensor[n, i64] = {
  (_, _, day) = civil_columns(dates_epoch_days(ds))
  day
}
def dates_weekday_iso_number[n](ds: Dates[n]) -> tensor[n, i64] =
  ds
  |> dates_epoch_days
  |> plus(3i64)
  |> euclid_mod(7i64)
  |> plus(1i64)
def dates_day_of_year[n](ds: Dates[n]) -> tensor[n, i64] = {
  days = dates_epoch_days(ds)
  (year, _, _) = civil_columns(days)
  days
  |> sub(epoch_day_columns(year, filled(year, 1i64), filled(year, 1i64)))
  |> plus(1i64)
}
-- Dates from fields.
def field_mask[n](year: &tensor[n, i64], month: &tensor[n, i64], day: &tensor[n, i64]) -> tensor[n, bool] = {
  fields = and(within(year, -9999i64, 9999i64), within(month, 1i64, 12i64))
  and(fields, and(gte(day, filled(day, 1i64)), lte(day, month_lengths(year, month))))
}
def field_detail(year: i64, month: i64, day: i64) -> string = if (year |> in_span(-9999i64, 9999i64) |> not) then outside_text("year", year, -9999i64, 9999i64) else if (month |> in_span(1i64, 12i64) |> not) then outside_text("month", month, 1i64, 12i64) else joined([outside_text("day", day, 1i64, days_in_month(year, month)), " for ", year_month_text(year, month)])
def dates_from_ymd[n](year: &tensor[n, i64], month: &tensor[n, i64], day: &tensor[n, i64]) -> Dates[n] = {
  bad = first_false(to_list(field_mask(year, month, day)))
  if gte(bad, 0i64) then fail(element_failure("dates_from_ymd", "domain", bad, field_detail(element_at(year, bad), element_at(month, bad), element_at(day, bad)))) else dates_from_epoch_days(epoch_day_columns(year, month, day))
}
def try_dates_from_ymd[n](year: &tensor[n, i64], month: &tensor[n, i64], day: &tensor[n, i64]) -> (Dates[n], tensor[n, bool]) = {
  valid = field_mask(year, month, day)
  (dates_from_epoch_days(epoch_day_columns(where(valid, year, filled(year, 1970i64)), where(valid, month, filled(month, 1i64)), where(valid, day, filled(day, 1i64)))), valid)
}
-- Date arithmetic.
def dates_add_days[n](ds: Dates[n], days: &tensor[n, i64]) -> Dates[n] = {
  start = dates_epoch_days(ds)
  if lengths_differ(start, days) then fail(length_failure("dates_add_days", start, days)) else shifted_days(start, days)
}
def shifted_days[n](start: &tensor[n, i64], days: &tensor[n, i64]) -> Dates[n] = {
  fits = and(lte(days, sub(filled(start, 2932896i64), start)), gte(days, sub(filled(start, -4371587i64), start)))
  bad = first_false(to_list(fits))
  if gte(bad, 0i64) then fail(element_failure("dates_add_days", "overflow", bad, joined([day_text(element_at(start, bad)), " plus ", to_string(element_at(days, bad)), " days is outside the supported date range"]))) else dates_from_epoch_days(add(start, days))
}
def dates_add_months[n](ds: Dates[n], months: &tensor[n, i64], overflow: DayOverflow) -> Dates[n] = {
  start = dates_epoch_days(ds)
  if lengths_differ(start, months) then fail(length_failure("dates_add_months", start, months)) else shifted_months(start, months, overflow)
}
def shifted_months[n](start: &tensor[n, i64], months: &tensor[n, i64], overflow: DayOverflow) -> Dates[n] = {
  (year, month, day) = civil_columns(start)
  total = add(times(year, 12i64), plus(month, -1i64))
  fits = and(lte(months, sub(filled(total, 119999i64), total)), gte(months, sub(filled(total, -119988i64), total)))
  target = add(total, where(fits, months, filled(total, 0i64)))
  target_year = floored(target, 12i64)
  target_month = target |> euclid_mod(12i64) |> plus(1i64)
  length = month_lengths(target_year, target_month)
  keeps = lte(day, length)
  rejecting = match overflow with {
    | ClampToMonthEnd => false
    | RejectInvalidDay => true
  }
  admitted = and(fits, or(keeps, flags_filled(day, not(rejecting))))
  bad = first_false(to_list(admitted))
  if gte(bad, 0i64) then if (fits |> to_list |> index(bad)) then fail(element_failure("dates_add_months", "domain", bad, joined([outside_text("day", element_at(day, bad), 1i64, element_at(length, bad)), " for ", year_month_text(element_at(target_year, bad), element_at(target_month, bad))]))) else fail(element_failure("dates_add_months", "overflow", bad, joined([day_text(element_at(start, bad)), " plus ", to_string(element_at(months, bad)), " months is outside the supported date range"]))) else dates_from_epoch_days(epoch_day_columns(target_year, target_month, where(keeps, day, length)))
}
def dates_days_until[n](a: Dates[n], b: Dates[n]) -> tensor[n, i64] = {
  start = dates_epoch_days(a)
  end = dates_epoch_days(b)
  if lengths_differ(start, end) then fail(length_failure("dates_days_until", start, end)) else sub(end, start)
}
-- Date order is epoch-day order.
def dates_lt[n](a: Dates[n], b: Dates[n]) -> tensor[n, bool] = {
  first = dates_epoch_days(a)
  second = dates_epoch_days(b)
  if lengths_differ(first, second) then fail(length_failure("dates_lt", first, second)) else lt(first, second)
}
def dates_lte[n](a: Dates[n], b: Dates[n]) -> tensor[n, bool] = {
  first = dates_epoch_days(a)
  second = dates_epoch_days(b)
  if lengths_differ(first, second) then fail(length_failure("dates_lte", first, second)) else lte(first, second)
}
def dates_gt[n](a: Dates[n], b: Dates[n]) -> tensor[n, bool] = {
  first = dates_epoch_days(a)
  second = dates_epoch_days(b)
  if lengths_differ(first, second) then fail(length_failure("dates_gt", first, second)) else gt(first, second)
}
def dates_gte[n](a: Dates[n], b: Dates[n]) -> tensor[n, bool] = {
  first = dates_epoch_days(a)
  second = dates_epoch_days(b)
  if lengths_differ(first, second) then fail(length_failure("dates_gte", first, second)) else gte(first, second)
}
-- Date text.
def parsed_day(parsed: Option[Date]) -> i64 =
  match parsed with {
    | Some(d) => date_epoch_day(d)
    | None => 0i64
  }
def parsed_flag(parsed: Option[Date]) -> bool =
  match parsed with {
    | Some(_) => true
    | None => false
  }
def try_parse_dates[n](texts: List[string]) -> (Dates[n], tensor[n, bool]) = {
  parsed = map(fn (text: string) -> try_parse_date(text), texts)
  (map(fn (p: Option[Date]) -> parsed_day(p), parsed)
  |> to_tensor
  |> dates_from_epoch_days, map(fn (p: Option[Date]) -> parsed_flag(p), parsed) |> to_tensor)
}
def dates_to_strings[n](ds: Dates[n]) -> List[string] = map(fn (day: i64) -> day_text(day), to_list(dates_epoch_days(ds)))
-- Instant columns. `parts` reads both storage tensors of a column.
def parts[n](column: Instants[n]) -> (tensor[n, i64], tensor[n, i64]) = (instants_unix_seconds(column), instants_nanoseconds(column))
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
    | Nanoseconds => 1000000000i64
    | _ => 1i64
  }
def instants_from_unix_count[n](counts: &tensor[n, i64], unit: TimeUnit) -> Instants[n] = {
  multiple = seconds_per_unit(unit)
  fraction = units_per_second(unit)
  valid = if gt(multiple, 0i64) then within(counts, neg(floor_div(377705030401i64, multiple)), floor_div(253402214400i64, multiple)) else (counts |> floored(fraction) |> within(-377705030401i64, 253402214400i64))
  bad = first_false(to_list(valid))
  if gte(bad, 0i64) then fail(element_failure("instants_from_unix_count", "domain", bad, joined([to_string(element_at(counts, bad)), " ", name_of_unit(unit), " since the unix epoch is outside the supported instant range"]))) else if gt(multiple, 0i64) then instants_from_unix(times(counts, multiple), filled(counts, 0i64)) else instants_from_unix(floored(counts, fraction), counts |> euclid_mod(fraction) |> times(floor_div(1000000000i64, fraction)))
}
-- 0 or 1 per element: whether `rounding` moves a value up from its floor,
-- given the remainder `rem` over the quantum `denom`, whether the floor is
-- even, and whether the value is negative. `RejectInexact` moves nothing;
-- `exact_or_admitted` says where it fires.
def rounding_steps[n](rounding: Rounding, rem: &tensor[n, i64], denom: i64, floor_even: &tensor[n, bool], negative: &tensor[n, bool]) -> tensor[n, i64] = {
  steps = match rounding with {
    | RoundTowardNegative => filled(rem, 0i64)
    | RoundTowardPositive => filled(rem, 1i64)
    | RoundTowardZero => ones_where(negative, rem)
    | RoundAwayFromZero => ones_where(not(negative), rem)
    | RoundTiesToEven => nearest_steps(rem, denom, not(floor_even))
    | RoundTiesToAway => nearest_steps(rem, denom, not(negative))
    | RejectInexact => filled(rem, 0i64)
  }
  where(eq(rem, filled(rem, 0i64)), filled(rem, 0i64), steps)
}
def nearest_steps[n](rem: &tensor[n, i64], denom: i64, tie_up: &tensor[n, bool]) -> tensor[n, i64] = {
  twice = times(rem, 2i64)
  where(lt(twice, filled(rem, denom)), filled(rem, 0i64), where(gt(twice, filled(rem, denom)), filled(rem, 1i64), ones_where(tie_up, rem)))
}
def exact_or_admitted[n](rounding: Rounding, rem: &tensor[n, i64]) -> tensor[n, bool] =
  match rounding with {
    | RejectInexact => eq(rem, filled(rem, 0i64))
    | _ => flags_filled(rem, true)
  }
-- `x * g + s` per element for `g > 0` and `0 <= s <= g`, and whether it fits
-- in i64; a value that does not fit is replaced by zero.
def scaled_sums[n](x: &tensor[n, i64], g: i64, s: &tensor[n, i64]) -> (tensor[n, i64], tensor[n, bool]) = {
  nonnegative = gte(x, filled(x, 0i64))
  gap = sub(filled(s, g), s)
  fits = where(nonnegative, lte(x, floored(sub(filled(s, 9223372036854775807i64), s), g)), gte(plus(x, 1i64), floored(plus(gap, add(sub(-9223372036854775807i64, 1i64), sub(g, 1i64))), g)))
  upper = add(times(where(and(fits, nonnegative), x, filled(x, 0i64)), g), s)
  lower = sub(times(plus(where(and(fits, not(nonnegative)), x, filled(x, -1i64)), 1i64), g), gap)
  (where(fits, where(nonnegative, upper, lower), filled(x, 0i64)), fits)
}
-- The rounded count of `unit` in each `second + nanosecond / 1e9`, whether
-- the rounding policy admits it, and whether it fits in i64.
def unit_counts[n](seconds: &tensor[n, i64], nanoseconds: &tensor[n, i64], unit: TimeUnit, rounding: Rounding) -> (tensor[n, i64], tensor[n, bool], tensor[n, bool]) = {
  multiple = seconds_per_unit(unit)
  negative = lt(seconds, filled(seconds, 0i64))
  if gt(multiple, 0i64) then {
    whole = floored(seconds, multiple)
    rem = add(times(euclid_mod(seconds, multiple), 1000000000i64), nanoseconds)
    (add(whole, rounding_steps(rounding, rem, mul(multiple, 1000000000i64), evens(whole), negative)), exact_or_admitted(rounding, rem), flags_filled(seconds, true))
  } else {
    fraction = units_per_second(unit)
    per = floor_div(1000000000i64, fraction)
    base = floored(nanoseconds, per)
    floor_even = if eq(mod(fraction, 2i64), 0i64) then evens(base) else evens(add(euclid_mod(seconds, 2i64), base))
    rem = euclid_mod(nanoseconds, per)
    (counts, fits) = scaled_sums(seconds, fraction, add(base, rounding_steps(rounding, rem, per, floor_even, negative)))
    (counts, exact_or_admitted(rounding, rem), fits)
  }
}
def instants_to_unix_count[n](is: Instants[n], unit: TimeUnit, rounding: Rounding) -> tensor[n, i64] = {
  (seconds, nanoseconds) = parts(is)
  (counts, admitted, fits) = unit_counts(seconds, nanoseconds, unit, rounding)
  bad = first_false(to_list(and(admitted, fits)))
  if gte(bad, 0i64) then if (admitted |> to_list |> index(bad)) then fail(element_failure("instants_to_unix_count", "overflow", bad, joined([instant_text(element_at(seconds, bad), element_at(nanoseconds, bad)), " in ", name_of_unit(unit), " does not fit in i64"]))) else fail(element_failure("instants_to_unix_count", "domain", bad, joined([instant_text(element_at(seconds, bad), element_at(nanoseconds, bad)), " is not a whole number of ", name_of_unit(unit)]))) else counts
}
def instants_add_duration[n](is: Instants[n], ds: Durations[n]) -> Instants[n] = {
  (seconds, nanoseconds) = parts(is)
  (shift, shift_nanos) = duration_parts(ds)
  if lengths_differ(seconds, shift) then fail(length_failure("instants_add_duration", seconds, shift)) else shifted_instants(seconds, nanoseconds, shift, shift_nanos)
}
def shifted_instants[n](seconds: &tensor[n, i64], nanoseconds: &tensor[n, i64], shift: &tensor[n, i64], shift_nanos: &tensor[n, i64]) -> Instants[n] = {
  nanos = add(nanoseconds, shift_nanos)
  carry = ones_where(gte(nanos, filled(nanos, 1000000000i64)), nanos)
  base = add(seconds, carry)
  fits = and(lte(shift, sub(filled(base, 253402214400i64), base)), gte(shift, sub(filled(base, -377705030401i64), base)))
  bad = first_false(to_list(fits))
  if gte(bad, 0i64) then fail(element_failure("instants_add_duration", "overflow", bad, joined([instant_text(element_at(seconds, bad), element_at(nanoseconds, bad)), " plus ", duration_text(element_at(shift, bad), element_at(shift_nanos, bad)), " is outside the supported instant range"]))) else instants_from_unix(add(base, shift), sub(nanos, times(carry, 1000000000i64)))
}
def instants_until[n](a: Instants[n], b: Instants[n]) -> Durations[n] = {
  (a_seconds, a_nanoseconds) = parts(a)
  (b_seconds, b_nanoseconds) = parts(b)
  if lengths_differ(a_seconds, b_seconds) then fail(length_failure("instants_until", a_seconds, b_seconds)) else {
    nanos = sub(b_nanoseconds, a_nanoseconds)
    Durations { seconds: add(sub(b_seconds, a_seconds), floored(nanos, 1000000000i64)), nanoseconds: euclid_mod(nanos, 1000000000i64) }
  }
}
def increment_problem(increment: Duration) -> Option[string] = {
  second = duration_second(increment)
  nanosecond = duration_nanosecond(increment)
  if or(lt(second, 0i64), and(eq(second, 0i64), eq(nanosecond, 0i64))) then (["increment ", duration_to_string(increment), " is not positive"]
  |> joined
  |> Some) else if gt(second, 86400i64) then Some(joined(["increment ", duration_to_string(increment), " does not divide one day"])) else if neq(mod(86400000000000i64, second |> mul(1000000000i64) |> add(nanosecond)), 0i64) then Some(joined(["increment ", duration_to_string(increment), " does not divide one day"])) else None
}
-- Rounds within each UTC day, as `instant_round_to` does.
def instants_round_to[n](is: Instants[n], increment: Duration, rounding: Rounding) -> Instants[n] =
  match increment_problem(increment) with {
    | Some(detail) => fail(joined(["instants_round_to: domain: ", detail]))
    | None => {
    step =
      increment
      |> duration_second
      |> mul(1000000000i64)
      |> add(duration_nanosecond(increment))
    (seconds, nanoseconds) = parts(is)
    day = floored(seconds, 86400i64)
    within_day = add(times(euclid_mod(seconds, 86400i64), 1000000000i64), nanoseconds)
    whole = floored(within_day, step)
    per_day = floor_div(86400000000000i64, step)
    floor_even = evens(add(times(euclid_mod(day, 2i64), mod(per_day, 2i64)), euclid_mod(whole, 2i64)))
    rem = sub(within_day, times(whole, step))
    rounded = times(add(whole, rounding_steps(rounding, rem, step, floor_even, lt(seconds, filled(seconds, 0i64)))), step)
    second = add(times(day, 86400i64), floored(rounded, 1000000000i64))
    admitted = exact_or_admitted(rounding, rem)
    fits = within(second, -377705030401i64, 253402214400i64)
    bad = first_false(to_list(and(admitted, fits)))
    if gte(bad, 0i64) then if (admitted |> to_list |> index(bad)) then fail(element_failure("instants_round_to", "overflow", bad, joined(["rounding ", instant_text(element_at(seconds, bad), element_at(nanoseconds, bad)), " to a multiple of ", duration_to_string(increment), " leaves the supported instant range"]))) else fail(element_failure("instants_round_to", "domain", bad, joined([instant_text(element_at(seconds, bad), element_at(nanoseconds, bad)), " is not a multiple of ", duration_to_string(increment)]))) else instants_from_unix(second, euclid_mod(rounded, 1000000000i64))
  }
  }
def instants_to_dates_at[n](is: Instants[n], o: Offset) -> Dates[n] =
  is
  |> instants_unix_seconds
  |> plus(offset_seconds(o))
  |> floored(86400i64)
  |> dates_from_epoch_days
-- `duration_to_seconds_f64` of each element's difference from `origin`: the
-- whole part and fraction share a sign, and each step rounds to nearest-even.
def instants_seconds_since_f64[n](is: Instants[n], origin: Instant) -> tensor[n, f64] = {
  (seconds, nanoseconds) = parts(is)
  nanos = sub(nanoseconds, filled(nanoseconds, instant_nanosecond(origin)))
  second = add(sub(seconds, filled(seconds, instant_unix_second(origin))), floored(nanos, 1000000000i64))
  nanosecond = euclid_mod(nanos, 1000000000i64)
  same_sign = or(gte(second, filled(second, 0i64)), eq(nanosecond, filled(nanosecond, 0i64)))
  whole = where(same_sign, second, plus(second, 1i64))
  fraction = where(same_sign, nanosecond, plus(nanosecond, -1000000000i64))
  per_second =
    1000000000.0f64
    |> scalar_to_tensor
    |> insert(0i32, shape(fraction, 0i32))
  add(cast(whole, f64), div(cast(fraction, f64), per_second))
}
-- Instant order: unix second, then nanosecond. `earlier` is `a < b`; the
-- other orders swap or negate it.
def earlier[n](a_seconds: &tensor[n, i64], a_nanoseconds: &tensor[n, i64], b_seconds: &tensor[n, i64], b_nanoseconds: &tensor[n, i64]) -> tensor[n, bool] = or(lt(a_seconds, b_seconds), and(eq(a_seconds, b_seconds), lt(a_nanoseconds, b_nanoseconds)))
def instants_ordered[n](function: string, a: Instants[n], b: Instants[n], swapped: bool, negated: bool) -> tensor[n, bool] = {
  (a_seconds, a_nanoseconds) = parts(a)
  (b_seconds, b_nanoseconds) = parts(b)
  if lengths_differ(a_seconds, b_seconds) then fail(length_failure(function, a_seconds, b_seconds)) else {
    before = if swapped then earlier(b_seconds, b_nanoseconds, a_seconds, a_nanoseconds) else earlier(a_seconds, a_nanoseconds, b_seconds, b_nanoseconds)
    if negated then not(before) else before
  }
}
def instants_lt[n](a: Instants[n], b: Instants[n]) -> tensor[n, bool] = instants_ordered("instants_lt", a, b, false, false)
def instants_lte[n](a: Instants[n], b: Instants[n]) -> tensor[n, bool] = instants_ordered("instants_lte", a, b, true, true)
def instants_gt[n](a: Instants[n], b: Instants[n]) -> tensor[n, bool] = instants_ordered("instants_gt", a, b, true, false)
def instants_gte[n](a: Instants[n], b: Instants[n]) -> tensor[n, bool] = instants_ordered("instants_gte", a, b, false, true)
