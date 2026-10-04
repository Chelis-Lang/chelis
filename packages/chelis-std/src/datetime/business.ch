module Std.Datetime.Business
export (Weekmask, BusinessCalendar, BusinessDayRoll, Unadjusted, Following, Preceding, ModifiedFollowing, ModifiedPreceding, NonBusinessStart, RejectNonBusinessStart, RollStartForward, RollStartBackward, business_calendar, try_business_calendar, business_calendar_weekmask, business_calendar_holidays, business_calendar_valid_from, business_calendar_valid_until, is_business_day, try_is_business_day, business_day_roll, try_business_day_roll, business_day_offset, try_business_day_offset, business_day_count, try_business_day_count, business_in_all, try_business_in_all, business_in_any, try_business_in_any, dates_is_business_day, dates_business_day_roll, dates_business_day_offset, dates_business_day_count)
import Std.Datetime (Date, Dates, date_epoch_day, date_from_epoch_day, date_to_string, dates_epoch_days, dates_from_epoch_days)
-- Std.Datetime.Business: business-day calendars over Std.Datetime dates,
-- governed by [05-OP-73]. A calendar answers only inside its horizon, and
-- every failure is a `domain` failure with the message grammar
-- `<function>: domain: <detail>`. The module reaches Std.Datetime through
-- its public callables alone.
type Weekmask =
  | Weekmask { monday: bool, tuesday: bool, wednesday: bool, thursday: bool, friday: bool, saturday: bool, sunday: bool }
type BusinessDayRoll =
  | Unadjusted
  | Following
  | Preceding
  | ModifiedFollowing
  | ModifiedPreceding
type NonBusinessStart =
  | RejectNonBusinessStart
  | RollStartForward
  | RollStartBackward
-- The weekmask, the holiday epoch days (sorted, unique, inside the horizon,
-- and each on a business weekday), and both horizon ends as epoch days. The
-- constructors normalize, so equal calendars have equal representations.
-- Queries count weekmask days in closed form and binary-search the
-- holidays, so the representation is O(h) for h holidays whatever the
-- horizon's length; spec/design/std_datetime.md §9 records the measured
-- costs.
@opaque
type BusinessCalendar =
  | BusinessCalendar { weekmask: Weekmask, holidays: List[i64], valid_from: i64, valid_until: i64 }
-- Text assembly.
def joined(parts: List[string]) -> string = fold(fn (acc: string, part: string) -> string_concat(acc, part), "", parts)
def domain_failure(function: string, detail: string) -> string = joined([function, ": domain: ", detail])
def element_detail(position: i64, detail: string) -> string = joined(["element ", to_string(position), ": ", detail])
def day_text(epoch_day: i64) -> string = epoch_day |> date_from_epoch_day |> date_to_string
def horizon_text(cal: BusinessCalendar) -> string = joined([day_text(cal.valid_from), "..", day_text(cal.valid_until)])
def outside_horizon(cal: BusinessCalendar, epoch_day: i64) -> string = joined([day_text(epoch_day), " is outside the horizon ", horizon_text(cal)])
def in_horizon(cal: BusinessCalendar, epoch_day: i64) -> bool = epoch_day |> gte(cal.valid_from) |> and(lte(epoch_day, cal.valid_until))
def euclid_rem(x: i64, y: i64) -> i64 = {
  r = mod(x, y)
  if lt(r, 0i64) then add(r, y) else r
}
-- Weekmask arithmetic. Weekday index 0 is Monday; 1970-01-01 is a Thursday.
-- A week table holds a weekmask's business weekdays per week, the business
-- weekdays before each weekday index 0 through 7, and the weekday index of
-- each business weekday in order. `mask_rank(week, x)` counts the weekmask
-- days from the Monday 1969-12-29 up to `x`, exclusive and negative before
-- it, so a difference of two ranks counts the weekmask days between them;
-- `mask_select` inverts it on weekmask days. Both are closed forms.
def weekmask_flags(w: Weekmask) -> List[bool] = [w.monday, w.tuesday, w.wednesday, w.thursday, w.friday, w.saturday, w.sunday]
def set_flags(flags: List[bool]) -> i64 = fold(fn (acc: i64, flag: bool) -> if flag then add(acc, 1i64) else acc, 0i64, flags)
def weekday_index(epoch_day: i64) -> i64 = epoch_day |> add(3i64) |> euclid_rem(7i64)
def weekmask_has(flags: List[bool], epoch_day: i64) -> bool = flags |> index(weekday_index(epoch_day))
def week_table(w: Weekmask) -> (i64, List[i64], List[i64]) = {
  flags = weekmask_flags(w)
  counts = map(fn (r: i64) -> set_flags(take(flags, r)), range(0i64, 8i64))
  (index(counts, 7i64), counts, filter(fn (i: i64) -> index(flags, i), range(0i64, 7i64)))
}
def mask_rank(week: (i64, List[i64], List[i64]), epoch_day: i64) -> i64 = {
  shifted = add(epoch_day, 3i64)
  shifted
  |> floor_div(7i64)
  |> mul(week.0)
  |> add(index(week.1, euclid_rem(shifted, 7i64)))
}
def mask_select(week: (i64, List[i64], List[i64]), mask_index: i64) -> i64 =
  mask_index
  |> floor_div(week.0)
  |> mul(7i64)
  |> add(index(week.2, euclid_rem(mask_index, week.0)))
  |> sub(3i64)
-- Holiday search. Each halving of a search interval of `count` positions
-- leaves at most half of it, so `search_steps(count)` halvings close it; a
-- horizon has at most 7304484 < 2^23 days, and so as many holidays.
def search_steps(count: i64) -> i64 = if lt(count, 16i64) then 5i64 else if lt(count, 1024i64) then 11i64 else if lt(count, 65536i64) then 17i64 else 23i64
-- The number of holidays before `epoch_day`.
def holidays_before(holidays: List[i64], epoch_day: i64) -> i64 = {
  bounds = fold(fn (acc: (i64, i64), step: i64) -> if lt(acc.0, acc.1) then {
    middle = floor_div(add(acc.0, acc.1), 2i64)
    if lt(index(holidays, middle), epoch_day) then (add(middle, 1i64), acc.1) else (acc.0, middle)
  } else acc, (0i64, len(holidays)), range(0i64, search_steps(len(holidays))))
  bounds.0
}
-- Business-day ranks. `business_before(cal, x)` is the number of business
-- days in `[valid_from, x)` for `x` in `valid_from..valid_until + 1`.
def business_before(cal: BusinessCalendar, epoch_day: i64) -> i64 = {
  week = week_table(cal.weekmask)
  week
  |> mask_rank(epoch_day)
  |> sub(mask_rank(week, cal.valid_from))
  |> sub(holidays_before(cal.holidays, epoch_day))
}
def business_total(cal: BusinessCalendar) -> i64 = business_before(cal, add(cal.valid_until, 1i64))
-- The business day with `ordinal` business days of the horizon before it,
-- for `ordinal` below `business_total(cal)`. With `j` holidays before the
-- answer, the answer is the weekmask day `mask_select(base + ordinal + j)`;
-- `j` is the least count whose next holiday lies after that day, a
-- predicate that stays true once it holds, so a binary search finds it.
def business_select(cal: BusinessCalendar, ordinal: i64) -> i64 = {
  week = week_table(cal.weekmask)
  base = week |> mask_rank(cal.valid_from) |> add(ordinal)
  holidays = cal.holidays
  bounds = fold(fn (acc: (i64, i64), step: i64) -> if lt(acc.0, acc.1) then {
    middle = floor_div(add(acc.0, acc.1), 2i64)
    if (week |> mask_select(add(base, middle)) |> lt(index(holidays, middle))) then (acc.0, middle) else (add(middle, 1i64), acc.1)
  } else acc, (0i64, len(holidays)), range(0i64, search_steps(len(holidays))))
  mask_select(week, add(base, bounds.0))
}
-- Whether a day of the horizon is a business day.
def is_business_at(cal: BusinessCalendar, epoch_day: i64) -> bool =
  if (cal.weekmask |> weekmask_flags |> weekmask_has(epoch_day)) then {
    position = holidays_before(cal.holidays, epoch_day)
    if lt(position, len(cal.holidays)) then neq(index(cal.holidays, position), epoch_day) else true
  } else false
-- Month identity: twelve times the March-based year plus the month counted
-- from March, from Hinnant's civil_from_days. Two days share a civil month
-- exactly when their keys are equal, and keys increase with the month.
def month_key(epoch_day: i64) -> i64 = {
  shifted = add(epoch_day, 719468i64)
  era = floor_div(shifted, 146097i64)
  doe = sub(shifted, mul(era, 146097i64))
  yoe = floor_div(sub(add(sub(doe, floor_div(doe, 1460i64)), floor_div(doe, 36524i64)), floor_div(doe, 146096i64)), 365i64)
  doy = sub(doe, sub(add(mul(365i64, yoe), floor_div(yoe, 4i64)), floor_div(yoe, 100i64)))
  yoe
  |> add(mul(era, 400i64))
  |> mul(12i64)
  |> add(floor_div(add(mul(5i64, doy), 2i64), 153i64))
}
-- Rolls. Each returns ("", day) or (detail, 0) when the answer is not
-- determined by the days inside the horizon or lies outside it.
def following_of(cal: BusinessCalendar, epoch_day: i64) -> (string, i64) = {
  ordinal = business_before(cal, epoch_day)
  if lt(ordinal, business_total(cal)) then ("", business_select(cal, ordinal)) else (joined(["no business day on or after ", day_text(epoch_day), " lies inside the horizon ", horizon_text(cal)]), 0i64)
}
def preceding_of(cal: BusinessCalendar, epoch_day: i64) -> (string, i64) = {
  ordinal = cal |> business_before(add(epoch_day, 1i64)) |> sub(1i64)
  if gte(ordinal, 0i64) then ("", business_select(cal, ordinal)) else (joined(["no business day on or before ", day_text(epoch_day), " lies inside the horizon ", horizon_text(cal)]), 0i64)
}
-- When no business day follows inside the horizon, the following business
-- day still lies in a later month than `epoch_day` if the day after the
-- horizon does, and the modified roll is then the preceding day.
def modified_following_of(cal: BusinessCalendar, epoch_day: i64) -> (string, i64) = {
  ahead = following_of(cal, epoch_day)
  month = month_key(epoch_day)
  if eq(ahead.0, "") then if eq(month_key(ahead.1), month) then ahead else preceding_of(cal, epoch_day) else if gt(month_key(add(cal.valid_until, 1i64)), month) then preceding_of(cal, epoch_day) else ahead
}
def modified_preceding_of(cal: BusinessCalendar, epoch_day: i64) -> (string, i64) = {
  behind = preceding_of(cal, epoch_day)
  month = month_key(epoch_day)
  if eq(behind.0, "") then if eq(month_key(behind.1), month) then behind else following_of(cal, epoch_day) else if lt(month_key(sub(cal.valid_from, 1i64)), month) then following_of(cal, epoch_day) else behind
}
def roll_of(cal: BusinessCalendar, epoch_day: i64, roll: BusinessDayRoll) -> (string, i64) =
  if in_horizon(cal, epoch_day) then match roll with {
    | Unadjusted => ("", epoch_day)
    | Following => following_of(cal, epoch_day)
    | Preceding => preceding_of(cal, epoch_day)
    | ModifiedFollowing => modified_following_of(cal, epoch_day)
    | ModifiedPreceding => modified_preceding_of(cal, epoch_day)
  } else (outside_horizon(cal, epoch_day), 0i64)
-- Offsets roll a non-business start first, then move whole business days.
def started_at(cal: BusinessCalendar, epoch_day: i64, start: NonBusinessStart) -> (string, i64) =
  if is_business_at(cal, epoch_day) then ("", epoch_day) else match start with {
    | RejectNonBusinessStart => (joined([day_text(epoch_day), " is not a business day"]), 0i64)
    | RollStartForward => following_of(cal, epoch_day)
    | RollStartBackward => preceding_of(cal, epoch_day)
  }
def offset_of(cal: BusinessCalendar, epoch_day: i64, n: i64, start: NonBusinessStart) -> (string, i64) =
  if in_horizon(cal, epoch_day) then {
    started = started_at(cal, epoch_day, start)
    if neq(started.0, "") then started else {
      ordinal = business_before(cal, started.1)
      last = sub(business_total(cal), 1i64)
      if or(gt(n, sub(last, ordinal)), lt(n, neg(ordinal))) then (joined([day_text(started.1), " plus ", to_string(n), " business days is outside the horizon ", horizon_text(cal)]), 0i64) else ("", business_select(cal, add(ordinal, n)))
    }
  } else (outside_horizon(cal, epoch_day), 0i64)
-- Counts take ends in `valid_from..valid_until + 1`.
def countable(cal: BusinessCalendar, epoch_day: i64) -> bool =
  epoch_day
  |> gte(cal.valid_from)
  |> and(lte(epoch_day, add(cal.valid_until, 1i64)))
def uncountable(cal: BusinessCalendar, role: string, epoch_day: i64) -> string = joined([role, " ", day_text(epoch_day), " is outside the horizon ", horizon_text(cal), " and is not the day after it"])
def count_of(cal: BusinessCalendar, begin: i64, end: i64) -> (string, i64) = if countable(cal, begin) then if countable(cal, end) then ("", sub(business_before(cal, end), business_before(cal, begin))) else (uncountable(cal, "end", end), 0i64) else (uncountable(cal, "begin", begin), 0i64)
-- Holiday normalization through the tensor sort: each day of `days` in
-- ascending order, paired with the day before it in that order (the first
-- with a smaller day).
def sorted_pairs(days: List[i64]) -> List[(i64, i64)] =
  if eq(len(days), 0i64) then [] else {
    ordered = to_list(sort(to_tensor(days), 0i32).0)
    first = index(ordered, 0i64)
    zip(ordered, take(concat([sub(first, 1i64)], ordered), len(ordered)))
  }
def sorted_unique(days: List[i64]) -> List[i64] = map(fn (pair: (i64, i64)) -> pair.0, filter(fn (pair: (i64, i64)) -> neq(pair.0, pair.1), sorted_pairs(days)))
-- The days that occur twice in `days`, which holds each day at most twice.
def repeated_days(days: List[i64]) -> List[i64] = map(fn (pair: (i64, i64)) -> pair.0, filter(fn (pair: (i64, i64)) -> eq(pair.0, pair.1), sorted_pairs(days)))
def normalized_holidays(weekmask: Weekmask, days: List[i64], valid_from: i64, valid_until: i64) -> List[i64] = {
  flags = weekmask_flags(weekmask)
  sorted_unique(filter(fn (day: i64) -> if (day |> gte(valid_from) |> and(lte(day, valid_until))) then weekmask_has(flags, day) else false, days))
}
-- Construction.
def first_outside(days: List[i64], valid_from: i64, valid_until: i64) -> i64 = fold(fn (acc: (i64, i64), day: i64) -> if gte(acc.1, 0i64) then acc else if (day |> gte(valid_from) |> and(lte(day, valid_until))) then (add(acc.0, 1i64), -1i64) else (add(acc.0, 1i64), acc.0), (0i64, -1i64), days).1
def calendar_problem(weekmask: Weekmask, days: List[i64], valid_from: i64, valid_until: i64) -> string =
  if eq(set_flags(weekmask_flags(weekmask)), 0i64) then "the weekmask has no business day" else if gt(valid_from, valid_until) then joined(["valid_from ", day_text(valid_from), " is after valid_until ", day_text(valid_until)]) else {
    bad = first_outside(days, valid_from, valid_until)
    if gte(bad, 0i64) then joined(["holiday ", day_text(index(days, bad)), " is outside the horizon ", day_text(valid_from), "..", day_text(valid_until)]) else ""
  }
def normalized_calendar(weekmask: Weekmask, days: List[i64], valid_from: i64, valid_until: i64) -> BusinessCalendar = BusinessCalendar { weekmask, holidays: normalized_holidays(weekmask, days, valid_from, valid_until), valid_from, valid_until }
def business_calendar(weekmask: Weekmask, holidays: List[Date], valid_from: Date, valid_until: Date) -> BusinessCalendar = {
  days = map(fn (d: Date) -> date_epoch_day(d), holidays)
  problem = calendar_problem(weekmask, days, date_epoch_day(valid_from), date_epoch_day(valid_until))
  if eq(problem, "") then normalized_calendar(weekmask, days, date_epoch_day(valid_from), date_epoch_day(valid_until)) else ("business_calendar" |> domain_failure(problem) |> fail)
}
def try_business_calendar(weekmask: Weekmask, holidays: List[Date], valid_from: Date, valid_until: Date) -> Option[BusinessCalendar] = {
  days = map(fn (d: Date) -> date_epoch_day(d), holidays)
  problem = calendar_problem(weekmask, days, date_epoch_day(valid_from), date_epoch_day(valid_until))
  if eq(problem, "") then Some(normalized_calendar(weekmask, days, date_epoch_day(valid_from), date_epoch_day(valid_until))) else None
}
def business_calendar_weekmask(cal: BusinessCalendar) -> Weekmask = cal.weekmask
def business_calendar_holidays(cal: BusinessCalendar) -> List[Date] = map(fn (day: i64) -> date_from_epoch_day(day), cal.holidays)
def business_calendar_valid_from(cal: BusinessCalendar) -> Date = date_from_epoch_day(cal.valid_from)
def business_calendar_valid_until(cal: BusinessCalendar) -> Date = date_from_epoch_day(cal.valid_until)
-- Scalar queries.
def is_business_day(cal: BusinessCalendar, d: Date) -> bool = {
  day = date_epoch_day(d)
  if in_horizon(cal, day) then is_business_at(cal, day) else ("is_business_day" |> domain_failure(outside_horizon(cal, day)) |> fail)
}
def try_is_business_day(cal: BusinessCalendar, d: Date) -> Option[bool] = {
  day = date_epoch_day(d)
  if in_horizon(cal, day) then Some(is_business_at(cal, day)) else None
}
def business_day_roll(cal: BusinessCalendar, d: Date, roll: BusinessDayRoll) -> Date = {
  (problem, day) = roll_of(cal, date_epoch_day(d), roll)
  if eq(problem, "") then date_from_epoch_day(day) else ("business_day_roll" |> domain_failure(problem) |> fail)
}
def try_business_day_roll(cal: BusinessCalendar, d: Date, roll: BusinessDayRoll) -> Option[Date] = {
  (problem, day) = roll_of(cal, date_epoch_day(d), roll)
  if eq(problem, "") then Some(date_from_epoch_day(day)) else None
}
def business_day_offset(cal: BusinessCalendar, d: Date, n: i64, start: NonBusinessStart) -> Date = {
  (problem, day) = offset_of(cal, date_epoch_day(d), n, start)
  if eq(problem, "") then date_from_epoch_day(day) else ("business_day_offset" |> domain_failure(problem) |> fail)
}
def try_business_day_offset(cal: BusinessCalendar, d: Date, n: i64, start: NonBusinessStart) -> Option[Date] = {
  (problem, day) = offset_of(cal, date_epoch_day(d), n, start)
  if eq(problem, "") then Some(date_from_epoch_day(day)) else None
}
def business_day_count(cal: BusinessCalendar, begin: Date, end: Date) -> i64 = {
  (problem, total) = count_of(cal, date_epoch_day(begin), date_epoch_day(end))
  if eq(problem, "") then total else ("business_day_count" |> domain_failure(problem) |> fail)
}
def try_business_day_count(cal: BusinessCalendar, begin: Date, end: Date) -> Option[i64] = {
  (problem, total) = count_of(cal, date_epoch_day(begin), date_epoch_day(end))
  if eq(problem, "") then Some(total) else None
}
-- Combination. The result's horizon is the intersection of the two.
def later_day(x: i64, y: i64) -> i64 = if gt(x, y) then x else y
def earlier_day(x: i64, y: i64) -> i64 = if lt(x, y) then x else y
def horizons_problem(a: BusinessCalendar, b: BusinessCalendar) -> string = if lt(earlier_day(a.valid_until, b.valid_until), later_day(a.valid_from, b.valid_from)) then joined(["the horizons ", horizon_text(a), " and ", horizon_text(b), " do not intersect"]) else ""
def weekmask_both(a: Weekmask, b: Weekmask) -> Weekmask = Weekmask { monday: and(a.monday, b.monday), tuesday: and(a.tuesday, b.tuesday), wednesday: and(a.wednesday, b.wednesday), thursday: and(a.thursday, b.thursday), friday: and(a.friday, b.friday), saturday: and(a.saturday, b.saturday), sunday: and(a.sunday, b.sunday) }
def weekmask_either(a: Weekmask, b: Weekmask) -> Weekmask = Weekmask { monday: or(a.monday, b.monday), tuesday: or(a.tuesday, b.tuesday), wednesday: or(a.wednesday, b.wednesday), thursday: or(a.thursday, b.thursday), friday: or(a.friday, b.friday), saturday: or(a.saturday, b.saturday), sunday: or(a.sunday, b.sunday) }
def all_problem(a: BusinessCalendar, b: BusinessCalendar) -> string = {
  disjoint = horizons_problem(a, b)
  if neq(disjoint, "") then disjoint else if eq(set_flags(weekmask_flags(weekmask_both(a.weekmask, b.weekmask))), 0i64) then "no weekday is a business day in both weekmasks" else ""
}
-- A day is a business day of the result when it is one in both calendars:
-- the weekmasks intersect and the holidays unite.
def all_calendar(a: BusinessCalendar, b: BusinessCalendar) -> BusinessCalendar = normalized_calendar(weekmask_both(a.weekmask, b.weekmask), concat(a.holidays, b.holidays), later_day(a.valid_from, b.valid_from), earlier_day(a.valid_until, b.valid_until))
-- A day is a business day of the result when it is one in either calendar:
-- the weekmasks unite, and a holiday of one calendar remains a holiday when
-- the other calendar's weekmask excludes its weekday or the other calendar
-- observes it too.
def any_calendar(a: BusinessCalendar, b: BusinessCalendar) -> BusinessCalendar = {
  valid_from = later_day(a.valid_from, b.valid_from)
  valid_until = earlier_day(a.valid_until, b.valid_until)
  a_flags = weekmask_flags(a.weekmask)
  b_flags = weekmask_flags(b.weekmask)
  a_days = filter(fn (day: i64) -> (day |> gte(valid_from) |> and(lte(day, valid_until))), a.holidays)
  b_days = filter(fn (day: i64) -> (day |> gte(valid_from) |> and(lte(day, valid_until))), b.holidays)
  a_only = filter(fn (day: i64) -> (b_flags |> weekmask_has(day) |> not), a_days)
  b_only = filter(fn (day: i64) -> (a_flags |> weekmask_has(day) |> not), b_days)
  holidays =
    a_only
    |> concat(b_only)
    |> concat(repeated_days(concat(a_days, b_days)))
    |> sorted_unique
  BusinessCalendar { weekmask: weekmask_either(a.weekmask, b.weekmask), holidays, valid_from, valid_until }
}
def business_in_all(a: BusinessCalendar, b: BusinessCalendar) -> BusinessCalendar = {
  problem = all_problem(a, b)
  if eq(problem, "") then all_calendar(a, b) else ("business_in_all" |> domain_failure(problem) |> fail)
}
def try_business_in_all(a: BusinessCalendar, b: BusinessCalendar) -> Option[BusinessCalendar] = if (a |> all_problem(b) |> eq("")) then Some(all_calendar(a, b)) else None
def business_in_any(a: BusinessCalendar, b: BusinessCalendar) -> BusinessCalendar = {
  problem = horizons_problem(a, b)
  if eq(problem, "") then any_calendar(a, b) else ("business_in_any" |> domain_failure(problem) |> fail)
}
def try_business_in_any(a: BusinessCalendar, b: BusinessCalendar) -> Option[BusinessCalendar] = if (a |> horizons_problem(b) |> eq("")) then Some(any_calendar(a, b)) else None
-- Vectorized forms. Each is a tensor kernel over the column: weekmask days
-- are ranked and selected in closed form through the week table, and every
-- holiday count is a prefix count over the merged sort of the queries and
-- the holidays, so a call costs O((n + h) log(n + h)) for n elements and h
-- holidays and nothing the length of the horizon. A call fails with the
-- detail its scalar twin gives for the lowest element where the kernel finds
-- no answer, and each kernel finds no answer exactly where the twin fails.
-- Every value a kernel passes to a trapping primitive is in range or
-- replaced first, so no primitive numeric trap escapes a call.
-- Constant columns and arithmetic with a scalar operand.
def filled[n](like: &tensor[n, i64], value: i64) -> tensor[n, i64] = value |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))
def flags_filled[n](like: &tensor[n, i64], value: bool) -> tensor[n, bool] = value |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))
def plus[n](t: &tensor[n, i64], value: i64) -> tensor[n, i64] = add(t, filled(t, value))
def times[n](t: &tensor[n, i64], value: i64) -> tensor[n, i64] = mul(t, filled(t, value))
def floored[n](t: &tensor[n, i64], value: i64) -> tensor[n, i64] = floor_div(t, filled(t, value))
-- The Euclidean remainder by a positive `value`.
def euclid_rems[n](t: &tensor[n, i64], value: i64) -> tensor[n, i64] = {
  r = sub(t, times(trunc_div(t, filled(t, value)), value))
  where(lt(r, filled(r, 0i64)), plus(r, value), r)
}
def within[n](t: &tensor[n, i64], low: i64, high: i64) -> tensor[n, bool] = and(gte(t, filled(t, low)), lte(t, filled(t, high)))
-- The positions 0, 1, ... of a column.
def positions_of[n](like: &tensor[n, i64]) -> tensor[n, i64] =
  like
  |> filled(1i64)
  |> cumsum(0i32)
  |> plus(-1i64)
-- For each query, the number of entries of `entries` below it. The queries
-- and entries are sorted together, keyed `2q - 1` and `2e` so that an entry
-- precedes a query exactly when it is smaller and no key of one kind equals
-- a key of the other; the running count of entries along the sorted order,
-- read at each query, is its answer. Equal queries see the same entries
-- before them, so the counts do not depend on how the sort orders equal
-- keys.
def count_below[n, h](entries: &tensor[h, i64], queries: &tensor[n, i64]) -> tensor[n, i64] = {
  size = shape(queries, 0i32)
  slots = positions_of(queries)
  others = plus(positions_of(entries), size)
  empty =
    0i64
    |> scalar_to_tensor
    |> insert(0i32, add(size, shape(entries, 0i32)))
  keys =
    empty
    |> copy
    |> scatter_replace(copy(slots), plus(times(queries, 2i64), -1i64), 0i32)
    |> scatter_replace(copy(others), times(entries, 2i64), 0i32)
  marks = empty |> copy |> scatter_replace(others, filled(entries, 1i64), 0i32)
  (ordered, order) = sort(keys, 0i32)
  counts = marks |> gather(copy(order), 0i32) |> cumsum(0i32)
  empty |> scatter_replace(order, counts, 0i32) |> gather(slots, 0i32)
}
-- `mask_rank`, `mask_select` and `weekmask_has` over columns.
def mask_ranks[n](week: (i64, List[i64], List[i64]), days: &tensor[n, i64]) -> tensor[n, i64] = {
  shifted = plus(days, 3i64)
  add(times(floored(shifted, 7i64), week.0), gather(to_tensor(week.1), euclid_rems(shifted, 7i64), 0i32))
}
def mask_selects[n](week: (i64, List[i64], List[i64]), indices: &tensor[n, i64]) -> tensor[n, i64] =
  indices
  |> floored(week.0)
  |> times(7i64)
  |> add(gather(to_tensor(week.2), euclid_rems(indices, week.0), 0i32))
  |> plus(-3i64)
def weekmask_hits[n](flags: List[bool], days: &tensor[n, i64]) -> tensor[n, bool] = gather(to_tensor(flags), euclid_rems(plus(days, 3i64), 7i64), 0i32)
-- `business_before`, `is_business_at` and `business_select` over columns.
def business_befores[n](cal: BusinessCalendar, days: &tensor[n, i64]) -> tensor[n, i64] = {
  week = week_table(cal.weekmask)
  week
  |> mask_ranks(days)
  |> plus(neg(mask_rank(week, cal.valid_from)))
  |> sub(count_below(to_tensor(cal.holidays), days))
}
def business_ats[n](cal: BusinessCalendar, days: &tensor[n, i64]) -> tensor[n, bool] = {
  holidays = to_tensor(cal.holidays)
  and(weekmask_hits(weekmask_flags(cal.weekmask), days), eq(count_below(holidays, days), count_below(holidays, plus(days, 1i64))))
}
-- With M_k the weekmask rank of the k-th holiday, the holidays a selection
-- skips are those with M_k - k at most the target rank: the target is the
-- (target - base)-th weekmask day from the horizon start that is no holiday.
def business_selects[n](cal: BusinessCalendar, ordinals: &tensor[n, i64]) -> tensor[n, i64] = {
  week = week_table(cal.weekmask)
  holidays = to_tensor(cal.holidays)
  gaps = sub(mask_ranks(week, holidays), positions_of(holidays))
  targets = plus(ordinals, mask_rank(week, cal.valid_from))
  mask_selects(week, add(targets, count_below(gaps, plus(targets, 1i64))))
}
-- Each kernel below returns whether it found an answer, and the answer, for
-- days inside the horizon.
def followings[n](cal: BusinessCalendar, days: &tensor[n, i64]) -> (tensor[n, bool], tensor[n, i64]) = {
  ordinals = business_befores(cal, days)
  found = lt(ordinals, filled(ordinals, business_total(cal)))
  (copy(found), business_selects(cal, where(found, ordinals, filled(ordinals, 0i64))))
}
def precedings[n](cal: BusinessCalendar, days: &tensor[n, i64]) -> (tensor[n, bool], tensor[n, i64]) = {
  ordinals = cal |> business_befores(plus(days, 1i64)) |> plus(-1i64)
  found = gte(ordinals, filled(ordinals, 0i64))
  (copy(found), business_selects(cal, where(found, ordinals, filled(ordinals, 0i64))))
}
def month_keys[n](days: &tensor[n, i64]) -> tensor[n, i64] = {
  shifted = plus(days, 719468i64)
  era = floored(shifted, 146097i64)
  doe = sub(shifted, times(era, 146097i64))
  yoe = floored(sub(add(sub(doe, floored(doe, 1460i64)), floored(doe, 36524i64)), floored(doe, 146096i64)), 365i64)
  doy = sub(doe, sub(add(times(yoe, 365i64), floored(yoe, 4i64)), floored(yoe, 100i64)))
  yoe
  |> add(times(era, 400i64))
  |> times(12i64)
  |> add(floored(plus(times(doy, 5i64), 2i64), 153i64))
}
-- As `modified_following_of`: the following day when it stays in the month,
-- and otherwise the preceding day, including when no business day follows
-- inside the horizon but the day after it already lies in a later month.
def modified_followings[n](cal: BusinessCalendar, days: &tensor[n, i64]) -> (tensor[n, bool], tensor[n, i64]) = {
  (ahead_found, ahead) = followings(cal, days)
  (behind_found, behind) = precedings(cal, days)
  month = month_keys(days)
  beyond = gt(filled(month, month_key(add(cal.valid_until, 1i64))), month)
  keep = where(copy(ahead_found), eq(month_keys(ahead), month), not(beyond))
  (where(copy(keep), ahead_found, behind_found), where(keep, ahead, behind))
}
def modified_precedings[n](cal: BusinessCalendar, days: &tensor[n, i64]) -> (tensor[n, bool], tensor[n, i64]) = {
  (ahead_found, ahead) = followings(cal, days)
  (behind_found, behind) = precedings(cal, days)
  month = month_keys(days)
  before = lt(filled(month, month_key(sub(cal.valid_from, 1i64))), month)
  keep = where(copy(behind_found), eq(month_keys(behind), month), not(before))
  (where(copy(keep), behind_found, ahead_found), where(keep, behind, ahead))
}
def rolls[n](cal: BusinessCalendar, days: &tensor[n, i64], roll: BusinessDayRoll) -> (tensor[n, bool], tensor[n, i64]) =
  match roll with {
    | Unadjusted => (flags_filled(days, true), copy(days))
    | Following => followings(cal, days)
    | Preceding => precedings(cal, days)
    | ModifiedFollowing => modified_followings(cal, days)
    | ModifiedPreceding => modified_precedings(cal, days)
  }
-- As `offset_of` for starts inside the horizon. An offset that leaves the
-- horizon is replaced by zero before any arithmetic reads it.
def offsets_from[n](cal: BusinessCalendar, days: &tensor[n, i64], offsets: &tensor[n, i64], start: NonBusinessStart) -> (tensor[n, bool], tensor[n, i64]) = {
  at = business_ats(cal, days)
  (rolled_found, rolled) = match start with {
    | RejectNonBusinessStart => (copy(at), copy(days))
    | RollStartForward => followings(cal, days)
    | RollStartBackward => precedings(cal, days)
  }
  started = where(copy(at), copy(days), rolled)
  ordinals = business_befores(cal, started)
  last = sub(business_total(cal), 1i64)
  moved = and(lte(offsets, sub(filled(ordinals, last), ordinals)), gte(offsets, neg(ordinals)))
  found = and(or(at, rolled_found), moved)
  zeros = filled(ordinals, 0i64)
  targets = add(where(copy(found), ordinals, copy(zeros)), where(copy(found), copy(offsets), zeros))
  (found, business_selects(cal, targets))
}
-- Element access for failure details, and the index of the first false
-- flag, or -1.
def element_at[n](t: &tensor[n, i64], position: i64) -> i64 = t |> to_list |> index(position)
def first_false(flags: List[bool]) -> i64 = fold(fn (acc: (i64, i64), flag: bool) -> if gte(acc.1, 0i64) then acc else if flag then (add(acc.0, 1i64), -1i64) else (add(acc.0, 1i64), acc.0), (0i64, -1i64), flags).1
def element_failure(function: string, position: i64, detail: string) -> string = function |> domain_failure(element_detail(position, detail))
-- A column argument whose length differs from another argument's fails
-- before any element is read ([05-OP-73]).
def length_failure[n, m](function: string, first: &tensor[n, i64], second: &tensor[m, i64]) -> string =
  function
  |> domain_failure(joined(["arguments have ", to_string(shape(first, 0i32)), " and ", to_string(shape(second, 0i32)), " elements"]))
def lengths_differ[n, m](first: &tensor[n, i64], second: &tensor[m, i64]) -> bool = neq(shape(first, 0i32), shape(second, 0i32))
-- Days outside the horizon are replaced by its first day before a kernel
-- reads them; their elements fail.
def clamped[n](cal: BusinessCalendar, days: &tensor[n, i64]) -> tensor[n, i64] = where(within(days, cal.valid_from, cal.valid_until), copy(days), filled(days, cal.valid_from))
def dates_is_business_day[n](cal: BusinessCalendar, ds: Dates[n]) -> tensor[n, bool] = {
  days = dates_epoch_days(ds)
  bad =
    days
    |> within(cal.valid_from, cal.valid_until)
    |> to_list
    |> first_false
  if gte(bad, 0i64) then ("dates_is_business_day"
  |> element_failure(bad, outside_horizon(cal, element_at(days, bad)))
  |> fail) else business_ats(cal, days)
}
def dates_business_day_roll[n](cal: BusinessCalendar, ds: Dates[n], roll: BusinessDayRoll) -> Dates[n] = {
  days = dates_epoch_days(ds)
  (found, rolled) = rolls(cal, clamped(cal, days), roll)
  bad =
    found
    |> and(within(days, cal.valid_from, cal.valid_until))
    |> to_list
    |> first_false
  if gte(bad, 0i64) then ("dates_business_day_roll"
  |> element_failure(bad, roll_of(cal, element_at(days, bad), roll).0)
  |> fail) else dates_from_epoch_days(rolled)
}
def dates_business_day_offset[n](cal: BusinessCalendar, ds: Dates[n], offsets: &tensor[n, i64], start: NonBusinessStart) -> Dates[n] = {
  days = dates_epoch_days(ds)
  if lengths_differ(days, offsets) then fail(length_failure("dates_business_day_offset", days, offsets)) else {
    (found, moved) = offsets_from(cal, clamped(cal, days), offsets, start)
    bad =
      found
      |> and(within(days, cal.valid_from, cal.valid_until))
      |> to_list
      |> first_false
    if gte(bad, 0i64) then ("dates_business_day_offset"
    |> element_failure(bad, offset_of(cal, element_at(days, bad), element_at(offsets, bad), start).0)
    |> fail) else dates_from_epoch_days(moved)
  }
}
def dates_business_day_count[n](cal: BusinessCalendar, begins: Dates[n], ends: Dates[n]) -> tensor[n, i64] = {
  firsts = dates_epoch_days(begins)
  lasts = dates_epoch_days(ends)
  if lengths_differ(firsts, lasts) then fail(length_failure("dates_business_day_count", firsts, lasts)) else {
    beyond = add(cal.valid_until, 1i64)
    bad =
      firsts
      |> within(cal.valid_from, beyond)
      |> and(within(lasts, cal.valid_from, beyond))
      |> to_list
      |> first_false
    if gte(bad, 0i64) then ("dates_business_day_count"
    |> element_failure(bad, count_of(cal, element_at(firsts, bad), element_at(lasts, bad)).0)
    |> fail) else sub(business_befores(cal, lasts), business_befores(cal, firsts))
  }
}
