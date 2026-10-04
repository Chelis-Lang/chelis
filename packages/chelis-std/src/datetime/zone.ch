module Std.Datetime.Zone
export (Disambiguation, EarlierInstant, LaterInstant, CompatibleInstant, RejectNonUniqueLocal, OffsetConflict, UseWrittenOffset, UseZoneRules, RejectOffsetMismatch, TimeZone, Zoned, ZonedText, time_zone_from_tzif, try_time_zone_from_tzif, time_zone_fixed, time_zone_utc, time_zone_name, time_zone_offset_at, try_time_zone_offset_at, zoned, try_zoned, zoned_from_local, try_zoned_from_local, zoned_instant, zoned_zone, zoned_local, zoned_offset, zoned_add_duration, zoned_add_period, zoned_to_string, parse_zoned_text, try_parse_zoned_text, zoned_from_text, try_zoned_from_text)
import Std.Datetime (DayOverflow, DateTime, Instant, Offset, Duration, Period, date, date_epoch_day, date_from_epoch_day, date_year, date_month, date_day, date_weekday, weekday_iso_number, is_leap_year, days_in_month, date_add_days, try_date_add_months, date_to_string, datetime, datetime_date, datetime_time, time_nanosecond_of_day, time_from_nanosecond_of_day, datetime_to_string, try_parse_datetime, offset_from_seconds, offset_seconds, offset_to_string, try_parse_offset, instant_from_unix, instant_unix_second, instant_nanosecond, instant_to_datetime_at, instant_to_string, duration_second, duration_nanosecond, duration_to_string, period_months, period_days, period_to_string)
-- Std.Datetime.Zone: time zone rules as values and zone-aware conversion,
-- governed by [05-OP-73]. A `TimeZone` is built from TZif bytes (RFC 9636,
-- versions 2 to 4) or from a fixed offset; nothing reads the host's time
-- zone database. `TimeZone` and `Zoned` are opaque, so the validating
-- producers below are their only construction path. Failures use `fail` with
-- the message grammar `<function>: <kind>: <detail>`, and every range check
-- runs before the arithmetic it protects.
type Disambiguation =
  | EarlierInstant
  | LaterInstant
  | CompatibleInstant
  | RejectNonUniqueLocal
type OffsetConflict =
  | UseWrittenOffset
  | UseZoneRules
  | RejectOffsetMismatch
-- A zone is its name, the offset before its first transition (TZif time type
-- 0), its strictly increasing transitions as (unix second, new offset), and
-- its footer rule, absent when the footer is empty. The footer rule is the
-- standard offset and, when the zone observes daylight saving time, the
-- daylight offset with the start and end rules. A rule is (form, a, b, c,
-- time of day in seconds): form 0 is `Jn` (a = n), form 1 is the zero-based
-- `n` (a = n), and form 2 is `Mm.w.d` (a = m, b = w, c = d). Offsets are
-- seconds east of UTC.
@opaque
type TimeZone =
  | TimeZone { name: string, initial_offset: i64, transitions: List[(i64, i64)], footer: Option[(i64, Option[(i64, (i64, i64, i64, i64, i64), (i64, i64, i64, i64, i64))])] }
@opaque
type Zoned =
  | Zoned { instant: Instant, zone: TimeZone }
-- Zoned text as written: its date and time, its offset, absent for `Z`, `z`
-- and a negative zero offset, which RFC 9557 reads as a UTC time with an
-- unknown local offset (the date and time are then UTC), the annotation's
-- zone name, and whether the annotation is critical.
type ZonedText =
  | ZonedText { written: DateTime, offset: Option[Offset], zone_name: string, critical: bool }
-- Range constants, written inline:
-- - unix seconds -377705030401..253402214400 (the instant range);
-- - offsets within -86399..86399 seconds;
-- - 12622780800 seconds in 400 Gregorian years, the period of every footer
--   rule, so a rule is evaluated on a reduced second in years 1970..2369.
def euclid_rem(x: i64, y: i64) -> i64 = {
  r = mod(x, y)
  if lt(r, 0i64) then add(r, y) else r
}
def in_span(value: i64, low: i64, high: i64) -> bool = value |> gte(low) |> and(lte(value, high))
def larger(a: i64, b: i64) -> i64 = if gt(a, b) then a else b
def smaller(a: i64, b: i64) -> i64 = if lt(a, b) then a else b
def joined(parts: List[string]) -> string = fold(fn (acc: string, part: string) -> string_concat(acc, part), "", parts)
def domain_failure(function: string, detail: string) -> string = joined([function, ": domain: ", detail])
def overflow_failure(function: string, detail: string) -> string = joined([function, ": overflow: ", detail])
def quoted(text: string) -> string = joined(["\"", text, "\""])
def contains_value(values: List[i64], value: i64) -> bool = fold(fn (acc: bool, item: i64) -> or(acc, eq(item, value)), false, values)
def with_value(values: List[i64], value: i64) -> List[i64] = if contains_value(values, value) then values else append(values, value)
-- Offset text in RFC 9557's zoned form: offset zero is `+00:00`, because `Z`
-- there means that the local offset is unknown.
def zoned_offset_text(seconds: i64) -> string = if eq(seconds, 0i64) then "+00:00" else (seconds |> offset_from_seconds |> offset_to_string)
def instant_text(second: i64, nanosecond: i64) -> string = instant_to_string(instant_from_unix(second, nanosecond))
def civil_text(civil: i64, nanosecond: i64) -> string = datetime_to_string(datetime(date_from_epoch_day(floor_div(civil, 86400i64)), time_from_nanosecond_of_day(add(mul(euclid_rem(civil, 86400i64), 1000000000i64), nanosecond))))
-- RFC 9557 `time-zone-name`: parts separated by "/", each starting with a
-- letter, "." or "_", continuing with those, digits, "-" or "+", and never
-- "." or "..".
def is_letter(code: i64) -> bool = or(in_span(code, 65i64, 90i64), in_span(code, 97i64, 122i64))
def is_digit(code: i64) -> bool = in_span(code, 48i64, 57i64)
def is_name_initial(code: i64) -> bool =
  code
  |> is_letter
  |> or(eq(code, 46i64))
  |> or(eq(code, 95i64))
def is_name_char(code: i64) -> bool =
  code
  |> is_name_initial
  |> or(is_digit(code))
  |> or(eq(code, 45i64))
  |> or(eq(code, 43i64))
def part_ends_well(length: i64, dots: bool) -> bool = and(gt(length, 0i64), or(not(dots), gt(length, 2i64)))
def is_zone_name(name: string) -> bool = {
  -- (valid so far, current part length, current part is all dots)
  scanned = fold(fn (acc: (bool, i64, bool), idx: i64) -> {
    code = char_code(string_slice(name, idx, 1i64))
    if not(acc.0) then acc else if eq(code, 47i64) then (part_ends_well(acc.1, acc.2), 0i64, true) else if eq(acc.1, 0i64) then (is_name_initial(code), 1i64, eq(code, 46i64)) else (is_name_char(code), add(acc.1, 1i64), and(acc.2, eq(code, 46i64)))
  }, (true, 0i64, true), range(0i64, string_len(name)))
  and(scanned.0, part_ends_well(scanned.1, scanned.2))
}
-- TZif bytes. Every reader below runs after its bounds and the 0..255 range
-- of every byte have been checked.
def unsigned_at(bytes: List[i64], start: i64, count: i64) -> i64 = fold(fn (acc: i64, idx: i64) -> (acc |> mul(256i64) |> add(index(bytes, idx))), 0i64, range(start, add(start, count)))
-- Two's-complement big-endian, for `count <= 8` bytes; every partial value
-- lies strictly inside i64.
def signed_at(bytes: List[i64], start: i64, count: i64) -> i64 = {
  lead = index(bytes, start)
  fold(fn (acc: i64, idx: i64) -> (acc |> mul(256i64) |> add(index(bytes, idx))), if gte(lead, 128i64) then sub(lead, 256i64) else lead, range(add(start, 1i64), add(start, count)))
}
-- The index of the first element outside 0..255, or -1.
def first_bad_byte(bytes: List[i64]) -> i64 = fold(fn (acc: (i64, i64), value: i64) -> if gte(acc.1, 0i64) then acc else if in_span(value, 0i64, 255i64) then (add(acc.0, 1i64), -1i64) else (add(acc.0, 1i64), acc.0), (0i64, -1i64), bytes).1
def has_magic(bytes: List[i64], at: i64) -> bool =
  bytes
  |> index(at)
  |> eq(84i64)
  |> and(eq(index(bytes, add(at, 1i64)), 90i64))
  |> and(eq(index(bytes, add(at, 2i64)), 105i64))
  |> and(eq(index(bytes, add(at, 3i64)), 102i64))
-- A header's six counts at `at`: (isutcnt, isstdcnt, leapcnt, timecnt,
-- typecnt, charcnt).
def header_counts(bytes: List[i64], at: i64) -> (i64, i64, i64, i64, i64, i64) = (unsigned_at(bytes, add(at, 20i64), 4i64), unsigned_at(bytes, add(at, 24i64), 4i64), unsigned_at(bytes, add(at, 28i64), 4i64), unsigned_at(bytes, add(at, 32i64), 4i64), unsigned_at(bytes, add(at, 36i64), 4i64), unsigned_at(bytes, add(at, 40i64), 4i64))
def version_text(version: i64) -> string = if in_span(version, 32i64, 126i64) then quoted(char_from_code(version)) else joined(["byte ", to_string(version)])
-- The layout of a TZif file: (problem, version, start of the version 2+ data
-- block, timecnt, typecnt, charcnt, isstdcnt, isutcnt). The problem is empty
-- exactly when the headers are well formed and the file is long enough for
-- the data block and a footer's two newlines.
def layout_of_tzif(bytes: List[i64]) -> (string, i64, i64, i64, i64, i64, i64, i64) = {
  size = len(bytes)
  if lt(size, 44i64) then (joined(["the file is ", to_string(size), " bytes, too short for a TZif header"]), 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else if (bytes |> has_magic(0i64) |> not) then ("the file does not start with \"TZif\"", 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else {
    version = index(bytes, 4i64)
    if eq(version, 0i64) then ("version 1 files have no 64-bit data", 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else if (version |> in_span(50i64, 52i64) |> not) then (joined(["version ", version_text(version), " is not \"2\", \"3\" or \"4\""]), 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else {
      (isut1, isstd1, leap1, time1, type1, char1) = header_counts(bytes, 0i64)
      second =
        44i64
        |> add(mul(time1, 5i64))
        |> add(mul(type1, 6i64))
        |> add(char1)
        |> add(mul(leap1, 8i64))
        |> add(isstd1)
        |> add(isut1)
      if gt(add(second, 44i64), size) then ("the file ends inside the version 2+ header", 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else if (bytes |> has_magic(second) |> not) then ("the version 2+ header does not start with \"TZif\"", 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else if (bytes |> index(add(second, 4i64)) |> neq(version)) then (joined(["the version 2+ header's version ", version_text(index(bytes, add(second, 4i64))), " differs from ", version_text(version)]), 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else {
        (isut, isstd, leap, timecnt, typecnt, charcnt) = header_counts(bytes, second)
        data = add(second, 44i64)
        footer =
          data
          |> add(mul(timecnt, 9i64))
          |> add(mul(typecnt, 6i64))
          |> add(charcnt)
          |> add(mul(leap, 12i64))
          |> add(isstd)
          |> add(isut)
        if eq(typecnt, 0i64) then ("typecnt is 0", 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else if eq(charcnt, 0i64) then ("charcnt is 0", 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else if (isstd |> neq(0i64) |> and(neq(isstd, typecnt))) then (joined(["isstdcnt ", to_string(isstd), " is neither 0 nor typecnt ", to_string(typecnt)]), 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else if (isut |> neq(0i64) |> and(neq(isut, typecnt))) then (joined(["isutcnt ", to_string(isut), " is neither 0 nor typecnt ", to_string(typecnt)]), 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else if neq(leap, 0i64) then (joined(["leapcnt is ", to_string(leap), ", and leap-second records are not on the POSIX timescale"]), 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else if gt(add(footer, 2i64), size) then ("the file ends inside the version 2+ data block or its footer", 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64) else ("", version, data, timecnt, typecnt, charcnt, isstd, isut)
      }
    }
  }
}
-- The first problem among the local time types, if any.
def type_problem(bytes: List[i64], types: i64, typecnt: i64, charcnt: i64) -> string =
  fold(fn (acc: string, k: i64) -> if neq(acc, "") then acc else {
    at = add(types, mul(k, 6i64))
    utoff = signed_at(bytes, at, 4i64)
    if (utoff |> in_span(-86399i64, 86399i64) |> not) then joined(["time type ", to_string(k), " has offset ", to_string(utoff), " s, outside -86399..86399"]) else if (bytes
    |> index(add(at, 4i64))
    |> in_span(0i64, 1i64)
    |> not) then joined(["time type ", to_string(k), " has DST flag ", to_string(index(bytes, add(at, 4i64))), ", not 0 or 1"]) else if (bytes
    |> index(add(at, 5i64))
    |> lt(charcnt)
    |> not) then joined(["time type ", to_string(k), " has designation index ", to_string(index(bytes, add(at, 5i64))), ", outside 0..", to_string(sub(charcnt, 1i64))]) else ""
  }, "", range(0i64, typecnt))
-- The first problem among the transitions: a type index out of range, or a
-- time not after its predecessor.
def transition_problem(bytes: List[i64], data: i64, timecnt: i64, typecnt: i64) -> string =
  fold(fn (acc: string, k: i64) -> if neq(acc, "") then acc else if (bytes
  |> index(data |> add(mul(timecnt, 8i64)) |> add(k))
  |> lt(typecnt)
  |> not) then joined(["transition ", to_string(k), " has time type ", to_string(index(bytes, data |> add(mul(timecnt, 8i64)) |> add(k))), ", outside 0..", to_string(sub(typecnt, 1i64))]) else if gt(k, 0i64) then if lte(signed_at(bytes, data |> add(mul(k, 8i64)), 8i64), signed_at(bytes, data |> add(mul(sub(k, 1i64), 8i64)), 8i64)) then joined(["transition ", to_string(k), " at unix second ", to_string(signed_at(bytes, data |> add(mul(k, 8i64)), 8i64)), " is not after transition ", to_string(sub(k, 1i64)), " at unix second ", to_string(signed_at(bytes, data |> add(mul(sub(k, 1i64), 8i64)), 8i64))]) else "" else "", "", range(0i64, timecnt))
-- The first problem among the standard/wall and UT/local indicators: each is
-- 0 or 1, and a UT indicator of 1 needs a standard indicator of 1.
def indicator_problem(bytes: List[i64], isstd_at: i64, isstd: i64, isut: i64, typecnt: i64) -> string =
  fold(fn (acc: string, k: i64) -> if neq(acc, "") then acc else {
    standard = if eq(isstd, 0i64) then 0i64 else index(bytes, add(isstd_at, k))
    universal = if eq(isut, 0i64) then 0i64 else (bytes |> index(isstd_at |> add(isstd) |> add(k)))
    if (standard |> in_span(0i64, 1i64) |> not) then joined(["standard/wall indicator ", to_string(k), " is ", to_string(standard), ", not 0 or 1"]) else if (universal |> in_span(0i64, 1i64) |> not) then joined(["UT/local indicator ", to_string(k), " is ", to_string(universal), ", not 0 or 1"]) else if (universal |> eq(1i64) |> and(eq(standard, 0i64))) then joined(["UT/local indicator ", to_string(k), " is 1 but its standard/wall indicator is 0"]) else ""
  }, "", range(0i64, typecnt))
-- The footer's characters: (problem, codes). The footer is a newline, a TZ
-- string of printable ASCII, and a newline that ends the file.
def footer_codes(bytes: List[i64], at: i64) -> (string, List[i64]) = {
  size = len(bytes)
  if (bytes |> index(at) |> neq(10i64)) then ("the footer does not start with a newline", []) else if (bytes |> index(sub(size, 1i64)) |> neq(10i64)) then ("the file does not end with the footer's closing newline", []) else {
    codes = bytes |> skip(add(at, 1i64)) |> take(size |> sub(at) |> sub(2i64))
    bad = fold(fn (acc: (i64, i64), code: i64) -> if gte(acc.1, 0i64) then acc else if in_span(code, 32i64, 126i64) then (add(acc.0, 1i64), -1i64) else (add(acc.0, 1i64), acc.0), (0i64, -1i64), codes).1
    if gte(bad, 0i64) then (joined(["footer byte ", to_string(bad), " is ", to_string(index(codes, bad)), ", not printable ASCII"]), []) else ("", codes)
  }
}
def codes_text(codes: List[i64]) -> string = fold(fn (acc: string, code: i64) -> string_concat(acc, char_from_code(code)), "", codes)
-- POSIX TZ strings. Each reader takes the footer's codes and an index and
-- returns what it read with the index after it, or None.
def code_at(codes: List[i64], idx: i64) -> i64 = if (idx |> gte(0i64) |> and(lt(idx, len(codes)))) then index(codes, idx) else -1i64
def digit_run_end(codes: List[i64], start: i64) -> i64 = fold(fn (acc: i64, idx: i64) -> if (acc |> eq(idx) |> and(is_digit(code_at(codes, idx)))) then add(idx, 1i64) else acc, start, range(start, len(codes)))
def letter_run_end(codes: List[i64], start: i64) -> i64 = fold(fn (acc: i64, idx: i64) -> if (acc |> eq(idx) |> and(is_letter(code_at(codes, idx)))) then add(idx, 1i64) else acc, start, range(start, len(codes)))
-- The value of the digits in [start, end), or -1 for a run longer than three
-- digits, which no field of a TZ string has, so a long run cannot overflow.
def digits_value(codes: List[i64], start: i64, end: i64) -> i64 = if gt(sub(end, start), 3i64) then -1i64 else fold(fn (acc: i64, idx: i64) -> (acc |> mul(10i64) |> add(sub(index(codes, idx), 48i64))), 0i64, range(start, end))
-- A designation: three or more letters, or "<" three or more letters, digits,
-- "+" or "-", and ">". Returns the index after it, or -1.
def designation_end(codes: List[i64], start: i64) -> i64 =
  if eq(code_at(codes, start), 60i64) then {
    close = fold(fn (acc: i64, idx: i64) -> if gte(acc, 0i64) then acc else if eq(code_at(codes, idx), 62i64) then idx else acc, -1i64, range(add(start, 1i64), len(codes)))
    inner_ok = fold(fn (acc: bool, idx: i64) -> and(acc, code_at(codes, idx)
    |> is_letter
    |> or(is_digit(code_at(codes, idx)))
    |> or(eq(code_at(codes, idx), 43i64))
    |> or(eq(code_at(codes, idx), 45i64))), true, range(add(start, 1i64), larger(close, add(start, 1i64))))
    if (close |> gte(add(start, 4i64)) |> and(inner_ok)) then add(close, 1i64) else -1i64
  } else {
    end = letter_run_end(codes, start)
    if gte(sub(end, start), 3i64) then end else -1i64
  }
-- `hh[:mm[:ss]]` with one to `hour_digits` hour digits and two-digit minutes
-- and seconds in 0..59: (seconds, hours, next).
def clock_at(codes: List[i64], start: i64, hour_digits: i64) -> Option[(i64, i64, i64)] = {
  hour_end = digit_run_end(codes, start)
  if (hour_end
  |> sub(start)
  |> in_span(1i64, hour_digits)
  |> not) then None else {
    hours = digits_value(codes, start, hour_end)
    if (codes |> code_at(hour_end) |> neq(58i64)) then Some((mul(hours, 3600i64), hours, hour_end)) else {
      minute_end = digit_run_end(codes, add(hour_end, 1i64))
      minutes = digits_value(codes, add(hour_end, 1i64), minute_end)
      if (minute_end
      |> sub(hour_end)
      |> neq(3i64)
      |> or(gt(minutes, 59i64))) then None else if (codes |> code_at(minute_end) |> neq(58i64)) then Some((add(mul(hours, 3600i64), mul(minutes, 60i64)), hours, minute_end)) else {
        second_end = digit_run_end(codes, add(minute_end, 1i64))
        seconds = digits_value(codes, add(minute_end, 1i64), second_end)
        if (second_end
        |> sub(minute_end)
        |> neq(3i64)
        |> or(gt(seconds, 59i64))) then None else Some((hours
        |> mul(3600i64)
        |> add(mul(minutes, 60i64))
        |> add(seconds), hours, second_end))
      }
    }
  }
}
-- A POSIX offset `[+|-]hh[:mm[:ss]]`, positive west of Greenwich: (seconds
-- east of UTC, next).
def posix_offset_at(codes: List[i64], start: i64) -> Option[(i64, i64)] = {
  lead = code_at(codes, start)
  signed = or(eq(lead, 43i64), eq(lead, 45i64))
  match clock_at(codes, if signed then add(start, 1i64) else start, 2i64) with {
    | Some(found) => Some((if eq(lead, 45i64) then found.0 else neg(found.0), found.2))
    | None => None
  }
}
-- A transition time `[+|-]hh[:mm[:ss]]` with hours up to 167 (RFC 9636
-- §3.3.2): (seconds, uses the version 3 extension, next).
def transition_time_at(codes: List[i64], start: i64) -> Option[(i64, bool, i64)] = {
  lead = code_at(codes, start)
  signed = or(eq(lead, 43i64), eq(lead, 45i64))
  match clock_at(codes, if signed then add(start, 1i64) else start, 3i64) with {
    | Some(found) => if gt(found.1, 167i64) then None else Some((if eq(lead, 45i64) then neg(found.0) else found.0, or(signed, gt(found.1, 24i64)), found.2))
    | None => None
  }
}
-- A rule date and optional time: ((form, a, b, c, time), extended, next).
def date_rule_at(codes: List[i64], start: i64) -> Option[((i64, i64, i64, i64, i64), bool, i64)] = {
  lead = code_at(codes, start)
  day = if eq(lead, 74i64) then {
    end = digit_run_end(codes, add(start, 1i64))
    n = digits_value(codes, add(start, 1i64), end)
    if (end
    |> sub(start)
    |> in_span(2i64, 4i64)
    |> and(in_span(n, 1i64, 365i64))) then Some(((0i64, n, 0i64, 0i64), end)) else None
  } else if eq(lead, 77i64) then {
    month_end = digit_run_end(codes, add(start, 1i64))
    month = digits_value(codes, add(start, 1i64), month_end)
    week = sub(code_at(codes, add(month_end, 1i64)), 48i64)
    weekday = sub(code_at(codes, add(month_end, 3i64)), 48i64)
    shape =
      month_end
      |> sub(start)
      |> in_span(2i64, 3i64)
      |> and(eq(code_at(codes, month_end), 46i64))
      |> and(eq(code_at(codes, add(month_end, 2i64)), 46i64))
      |> and(is_digit(code_at(codes, add(month_end, 1i64))))
      |> and(is_digit(code_at(codes, add(month_end, 3i64))))
    if (shape
    |> and(in_span(month, 1i64, 12i64))
    |> and(in_span(week, 1i64, 5i64))
    |> and(in_span(weekday, 0i64, 6i64))) then Some(((2i64, month, week, weekday), add(month_end, 4i64))) else None
  } else {
    end = digit_run_end(codes, start)
    n = digits_value(codes, start, end)
    if (end
    |> sub(start)
    |> in_span(1i64, 3i64)
    |> and(in_span(n, 0i64, 365i64))) then Some(((1i64, n, 0i64, 0i64), end)) else None
  }
  match day with {
    | Some(found) => {
    ((form, a, b, c), at) = found
    if (codes |> code_at(at) |> neq(47i64)) then Some(((form, a, b, c, 7200i64), false, at)) else match transition_time_at(codes, add(at, 1i64)) with {
      | Some(time) => Some(((form, a, b, c, time.0), time.1, time.2))
      | None => None
    }
  }
    | None => None
  }
}
def offset_bound_problem(text: string, seconds: i64) -> string = if in_span(seconds, -86399i64, 86399i64) then "" else joined(["footer ", quoted(text), " has offset ", to_string(seconds), " s, outside -86399..86399"])
-- The footer's rule: (problem, rule), where the rule is absent for an empty
-- TZ string and the problem is empty exactly when the string is valid.
def footer_reading(codes: List[i64], version: i64) -> (string, Option[(i64, Option[(i64, (i64, i64, i64, i64, i64), (i64, i64, i64, i64, i64))])]) = {
  text = codes_text(codes)
  syntax = joined(["footer ", quoted(text), " is not a POSIX TZ string"])
  if eq(len(codes), 0i64) then ("", None) else {
    std_end = designation_end(codes, 0i64)
    match posix_offset_at(codes, std_end) with {
      | Some(found_std) => {
      (standard, after_std) = found_std
      if eq(after_std, len(codes)) then if (text |> offset_bound_problem(standard) |> eq("")) then ("", Some((standard, None))) else (offset_bound_problem(text, standard), None) else {
        dst_end = designation_end(codes, after_std)
        lead = code_at(codes, dst_end)
        explicit =
          lead
          |> is_digit
          |> or(eq(lead, 43i64))
          |> or(eq(lead, 45i64))
        dst_offset = if explicit then posix_offset_at(codes, dst_end) else Some((add(standard, 3600i64), dst_end))
        match dst_offset with {
          | Some(found_dst) => {
          (daylight, after_dst) = found_dst
          if lt(dst_end, 0i64) then (syntax, None) else if eq(after_dst, len(codes)) then (joined(["footer ", quoted(text), " names daylight saving time without a rule"]), None) else if (codes |> code_at(after_dst) |> neq(44i64)) then (syntax, None) else match date_rule_at(codes, add(after_dst, 1i64)) with {
            | Some(found_start) => {
            (start_rule, start_extended, after_start) = found_start
            if (codes |> code_at(after_start) |> neq(44i64)) then (syntax, None) else match date_rule_at(codes, add(after_start, 1i64)) with {
              | Some(found_end) => {
              (end_rule, end_extended, after_end) = found_end
              if neq(after_end, len(codes)) then (syntax, None) else if (version |> eq(50i64) |> and(or(start_extended, end_extended))) then (joined(["footer ", quoted(text), " uses the version 3 transition-time extension in a version 2 file"]), None) else if (text |> offset_bound_problem(standard) |> neq("")) then (offset_bound_problem(text, standard), None) else if (text |> offset_bound_problem(daylight) |> neq("")) then (offset_bound_problem(text, daylight), None) else ("", Some((standard, Some((daylight, start_rule, end_rule)))))
            }
              | None => (syntax, None)
            }
          }
            | None => (syntax, None)
          }
        }
          | None => (syntax, None)
        }
      }
    }
      | None => (syntax, None)
    }
  }
}
-- Footer rules. A rule's transitions repeat every 400 Gregorian years
-- (146097 days, a whole number of weeks), so each evaluation reduces its
-- second into [0, 12622780800) and works in years 1968..2371, where the
-- `Std.Datetime` producers below never fail.
def epoch_day_of_rule(rule: (i64, i64, i64, i64, i64), year: i64) -> i64 = {
  january = date_epoch_day(date(year, 1i64, 1i64))
  if eq(rule.0, 0i64) then (january
  |> add(sub(rule.1, 1i64))
  |> add(if (year |> is_leap_year |> and(gte(rule.1, 60i64))) then 1i64 else 0i64)) else if eq(rule.0, 1i64) then add(january, rule.1) else {
    first = date(year, rule.1, 1i64)
    lead =
      first
      |> date_weekday
      |> weekday_iso_number
      |> euclid_rem(7i64)
    day =
      1i64
      |> add(euclid_rem(sub(rule.3, lead), 7i64))
      |> add(mul(sub(rule.2, 1i64), 7i64))
    first
    |> date_epoch_day
    |> add(if gt(day, days_in_month(year, rule.1)) then sub(day, 8i64) else sub(day, 1i64))
  }
}
-- The instant a rule names in `year`, its time of day read at `offset`.
def instant_of_rule(rule: (i64, i64, i64, i64, i64), year: i64, offset: i64) -> i64 =
  rule
  |> epoch_day_of_rule(year)
  |> mul(86400i64)
  |> add(rule.4)
  |> sub(offset)
def reduced_year(reduced: i64) -> i64 =
  reduced
  |> floor_div(86400i64)
  |> date_from_epoch_day
  |> date_year
-- The offset a footer gives at unix second `second`: the latest daylight
-- start or end at or before it, a start winning a tie, so that a rule that
-- starts each year where the previous year's ends keeps daylight time all
-- year (RFC 9636 §3.3.1).
def footer_offset(footer: (i64, Option[(i64, (i64, i64, i64, i64, i64), (i64, i64, i64, i64, i64))]), second: i64) -> i64 =
  match footer.1 with {
    | Some(daylight) => {
    (dst, start, end) = daylight
    reduced = euclid_rem(second, 12622780800i64)
    year = reduced_year(reduced)
    latest = fold(fn (acc: (i64, i64), y: i64) -> {
      ending = instant_of_rule(end, y, dst)
      starting = instant_of_rule(start, y, footer.0)
      after_end = if (ending |> lte(reduced) |> and(gt(ending, acc.0))) then (ending, 0i64) else acc
      if (starting |> lte(reduced) |> and(gte(starting, after_end.0))) then (starting, 1i64) else after_end
    }, (-9223372036854775807i64, 0i64), range(sub(year, 2i64), add(year, 2i64)))
    if eq(latest.1, 1i64) then dst else footer.0
  }
    | None => footer.0
  }
-- The footer's daylight starts and ends in (low, high], for `high - low`
-- within a few days.
def footer_changes(footer: (i64, Option[(i64, (i64, i64, i64, i64, i64), (i64, i64, i64, i64, i64))]), low: i64, high: i64) -> List[i64] =
  match footer.1 with {
    | Some(daylight) => {
    (dst, start, end) = daylight
    reduced = euclid_rem(low, 12622780800i64)
    base = sub(low, reduced)
    top = add(reduced, sub(high, low))
    year = reduced_year(reduced)
    fold(fn (acc: List[i64], y: i64) -> {
      ending = instant_of_rule(end, y, dst)
      starting = instant_of_rule(start, y, footer.0)
      with_end = if in_span(ending, add(reduced, 1i64), top) then with_value(acc, add(base, ending)) else acc
      if in_span(starting, add(reduced, 1i64), top) then with_value(with_end, add(base, starting)) else with_end
    }, [], range(sub(year, 2i64), add(year, 3i64)))
  }
    | None => []
  }
-- Zone queries. A zone covers every instant unless its footer is empty and it
-- has transitions; then it covers only the instants before its last one.
def transitions_through(transitions: List[(i64, i64)], second: i64) -> i64 =
  fold(fn (acc: (i64, i64), step: i64) -> if gte(acc.0, acc.1) then acc else {
    middle = floor_div(add(acc.0, acc.1), 2i64)
    if lte(index(transitions, middle).0, second) then (add(middle, 1i64), acc.1) else (acc.0, middle)
  }, (0i64, len(transitions)), range(0i64, 33i64)).0
-- (covered, offset at unix second `second`).
def offset_in_force(tz: TimeZone, second: i64) -> (bool, i64) = {
  count = len(tz.transitions)
  passed = transitions_through(tz.transitions, second)
  if lt(passed, count) then (true, if eq(passed, 0i64) then tz.initial_offset else index(tz.transitions, sub(passed, 1i64)).1) else match tz.footer with {
    | Some(footer) => (true, footer_offset(footer, second))
    | None => if eq(count, 0i64) then (true, tz.initial_offset) else (false, 0i64)
  }
}
-- The first uncovered unix second, or None when the zone covers every instant.
def coverage_end(tz: TimeZone) -> Option[i64] =
  match tz.footer with {
    | Some(_) => None
    | None => if eq(len(tz.transitions), 0i64) then None else Some(index(tz.transitions, sub(len(tz.transitions), 1i64)).0)
  }
def uncovered_text(tz: TimeZone, second: i64, nanosecond: i64) -> string = joined([instant_text(second, nanosecond), " is at or after unix second ", to_string(index(tz.transitions, sub(len(tz.transitions), 1i64)).0), ", the last transition of ", quoted(tz.name), ", whose footer is empty"])
-- The zone's offset changes in (low, high]: its transitions there and, from
-- its last transition on, its footer's.
def changes_between(tz: TimeZone, low: i64, high: i64) -> List[i64] = {
  count = len(tz.transitions)
  listed = map(fn (k: i64) -> index(tz.transitions, k).0, range(transitions_through(tz.transitions, low), transitions_through(tz.transitions, high)))
  match tz.footer with {
    | Some(footer) => {
    floor = if eq(count, 0i64) then low else larger(low, index(tz.transitions, sub(count, 1i64)).0)
    if lt(floor, high) then concat(listed, footer_changes(footer, floor, high)) else listed
  }
    | None => listed
  }
}
-- TZif parsing.
def tzif_reading(name: string, bytes: List[i64]) -> (string, TimeZone) = {
  placeholder = TimeZone { name, initial_offset: 0i64, transitions: [], footer: Some((0i64, None)) }
  bad = first_bad_byte(bytes)
  if (name |> is_zone_name |> not) then (joined(["name ", quoted(name), " is not an RFC 9557 time zone name"]), placeholder) else if gte(bad, 0i64) then (joined([quoted(name), ": byte ", to_string(bad), " is ", to_string(index(bytes, bad)), ", outside 0..255"]), placeholder) else {
    (layout_problem, version, data, timecnt, typecnt, charcnt, isstd, isut) = layout_of_tzif(bytes)
    types = add(data, mul(timecnt, 9i64))
    isstd_at = types |> add(mul(typecnt, 6i64)) |> add(charcnt)
    footer_at = isstd_at |> add(isstd) |> add(isut)
    problem = if neq(layout_problem, "") then layout_problem else {
      typed = type_problem(bytes, types, typecnt, charcnt)
      if neq(typed, "") then typed else {
        ordered = transition_problem(bytes, data, timecnt, typecnt)
        if neq(ordered, "") then ordered else indicator_problem(bytes, isstd_at, isstd, isut, typecnt)
      }
    }
    if neq(problem, "") then (joined([quoted(name), ": ", problem]), placeholder) else {
      (framing, codes) = footer_codes(bytes, footer_at)
      (rule_problem, footer) = if eq(framing, "") then footer_reading(codes, version) else (framing, None)
      offsets = map(fn (k: i64) -> signed_at(bytes, add(types, mul(k, 6i64)), 4i64), range(0i64, typecnt))
      transitions = map(fn (k: i64) -> (signed_at(bytes, add(data, mul(k, 8i64)), 8i64), index(offsets, bytes |> index(data |> add(mul(timecnt, 8i64)) |> add(k)))), range(0i64, timecnt))
      if neq(rule_problem, "") then (joined([quoted(name), ": ", rule_problem]), placeholder) else match footer with {
        | Some(rule) => if eq(timecnt, 0i64) then ("", TimeZone { name, initial_offset: index(offsets, 0i64), transitions, footer }) else {
        (last_second, last_offset) = index(transitions, sub(timecnt, 1i64))
        if (rule |> footer_offset(last_second) |> eq(last_offset)) then ("", TimeZone { name, initial_offset: index(offsets, 0i64), transitions, footer }) else (joined([quoted(name), ": footer ", quoted(codes_text(codes)), " gives offset ", to_string(footer_offset(rule, last_second)), " s at the last transition, unix second ", to_string(last_second), ", which gives ", to_string(last_offset), " s"]), placeholder)
      }
        | None => ("", TimeZone { name, initial_offset: index(offsets, 0i64), transitions, footer })
      }
    }
  }
}
def time_zone_from_tzif(name: string, bytes: List[i64]) -> TimeZone = {
  (problem, tz) = tzif_reading(name, bytes)
  if eq(problem, "") then tz else ("time_zone_from_tzif" |> domain_failure(problem) |> fail)
}
def try_time_zone_from_tzif(name: string, bytes: List[i64]) -> Option[TimeZone] = {
  (problem, tz) = tzif_reading(name, bytes)
  if eq(problem, "") then Some(tz) else None
}
def fixed_name(seconds: i64) -> string = if eq(seconds, 0i64) then "+00:00" else (seconds |> offset_from_seconds |> offset_to_string)
def time_zone_fixed(o: Offset) -> TimeZone = TimeZone { name: o |> offset_seconds |> fixed_name, initial_offset: offset_seconds(o), transitions: [], footer: Some((offset_seconds(o), None)) }
def time_zone_utc() -> TimeZone = TimeZone { name: "UTC", initial_offset: 0i64, transitions: [], footer: Some((0i64, None)) }
def time_zone_name(tz: TimeZone) -> string = tz.name
def time_zone_offset_at(tz: TimeZone, i: Instant) -> Offset = {
  (covered, seconds) = offset_in_force(tz, instant_unix_second(i))
  if covered then offset_from_seconds(seconds) else ("time_zone_offset_at"
  |> domain_failure(uncovered_text(tz, instant_unix_second(i), instant_nanosecond(i)))
  |> fail)
}
def try_time_zone_offset_at(tz: TimeZone, i: Instant) -> Option[Offset] = {
  (covered, seconds) = offset_in_force(tz, instant_unix_second(i))
  if covered then Some(offset_from_seconds(seconds)) else None
}
-- Zoned values.
def zoned(i: Instant, tz: TimeZone) -> Zoned =
  if offset_in_force(tz, instant_unix_second(i)).0 then Zoned { instant: i, zone: tz } else ("zoned"
  |> domain_failure(uncovered_text(tz, instant_unix_second(i), instant_nanosecond(i)))
  |> fail)
def try_zoned(i: Instant, tz: TimeZone) -> Option[Zoned] = if offset_in_force(tz, instant_unix_second(i)).0 then Some(Zoned { instant: i, zone: tz }) else None
def zoned_instant(z: Zoned) -> Instant = z.instant
def zoned_zone(z: Zoned) -> TimeZone = z.zone
def zoned_offset(z: Zoned) -> Offset = offset_from_seconds(offset_in_force(z.zone, instant_unix_second(z.instant)).1)
def zoned_local(z: Zoned) -> DateTime = instant_to_datetime_at(z.instant, zoned_offset(z))
-- Resolving a local reading. `civil` is the local reading's seconds since
-- 1970-01-01T00:00:00 read at offset zero. The trials are the offsets in
-- force within one day of it, the candidates are the trials `o` whose
-- instant `civil - o` has offset `o`, and a gap is found through the change
-- whose skipped local readings contain it. The result is (kind, detail, unix
-- second): kind "" on success, else "domain" or "overflow".
def local_resolution(tz: TimeZone, civil: i64, nanosecond: i64, disambiguation: Disambiguation) -> (string, string, i64) = {
  low = larger(sub(civil, 86399i64), -377705030401i64)
  high = smaller(add(civil, 86399i64), 253402214400i64)
  covered_high = match coverage_end(tz) with {
    | Some(end) => smaller(high, sub(end, 1i64))
    | None => high
  }
  local = civil_text(civil, nanosecond)
  trials = if lte(low, covered_high) then fold(fn (acc: List[i64], change: i64) -> with_value(acc, offset_in_force(tz, change).1), [offset_in_force(tz, low).1], changes_between(tz, low, covered_high)) else []
  -- The first trial whose instant leaves the range, if any.
  outside = fold(fn (acc: Option[i64], o: i64) -> match acc with {
    | Some(_) => acc
    | None => if (civil |> sub(o) |> in_span(-377705030401i64, 253402214400i64)) then None else Some(o)
  }, None, trials)
  match outside with {
    | Some(o) => ("overflow", joined([local, " at offset ", zoned_offset_text(o), " is outside the supported instant range"]), 0i64)
    | None => if lt(covered_high, high) then ("domain", joined([local, " needs offsets at or after unix second ", to_string(index(tz.transitions, sub(len(tz.transitions), 1i64)).0), ", the last transition of ", quoted(tz.name), ", whose footer is empty"]), 0i64) else {
    candidates = filter(fn (o: i64) -> eq(offset_in_force(tz, sub(civil, o)).1, o), trials)
    if gt(len(candidates), 0i64) then {
      earliest = fold(fn (acc: i64, o: i64) -> smaller(acc, sub(civil, o)), 9223372036854775807i64, candidates)
      latest = fold(fn (acc: i64, o: i64) -> larger(acc, sub(civil, o)), -9223372036854775807i64, candidates)
      if eq(len(candidates), 1i64) then ("", "", earliest) else match disambiguation with {
        | EarlierInstant => ("", "", earliest)
        | LaterInstant => ("", "", latest)
        | CompatibleInstant => ("", "", earliest)
        | RejectNonUniqueLocal => ("domain", joined([local, " occurs ", to_string(len(candidates)), " times in ", quoted(tz.name), ", at offsets ", joined_with(map(fn (o: i64) -> zoned_offset_text(o), candidates), ", ")]), 0i64)
      }
    } else {
      gaps = filter(fn (change: i64) -> and(lte(add(change, offset_in_force(tz, sub(change, 1i64)).1), civil), lt(civil, add(change, offset_in_force(tz, change).1))), changes_between(tz, low, high))
      if neq(len(gaps), 1i64) then ("domain", joined([local, " lies in ", to_string(len(gaps)), " gaps of ", quoted(tz.name)]), 0i64) else {
        change = index(gaps, 0i64)
        before = offset_in_force(tz, sub(change, 1i64)).1
        after = offset_in_force(tz, change).1
        match disambiguation with {
          | EarlierInstant => ("", "", sub(civil, after))
          | LaterInstant => ("", "", sub(civil, before))
          | CompatibleInstant => ("", "", sub(civil, before))
          | RejectNonUniqueLocal => ("domain", joined([local, " does not exist in ", quoted(tz.name), ": the transition at unix second ", to_string(change), " skips it"]), 0i64)
        }
      }
    }
  }
  }
}
def joined_with(parts: List[string], separator: string) -> string = fold(fn (acc: (string, bool), part: string) -> (if acc.1 then joined([acc.0, separator, part]) else part, true), ("", false), parts).0
def local_civil(dt: DateTime) -> (i64, i64) = {
  nanos = time_nanosecond_of_day(datetime_time(dt))
  (dt
  |> datetime_date
  |> date_epoch_day
  |> mul(86400i64)
  |> add(floor_div(nanos, 1000000000i64)), euclid_rem(nanos, 1000000000i64))
}
def resolved(function: string, tz: TimeZone, civil: i64, nanosecond: i64, disambiguation: Disambiguation) -> Option[Zoned] = {
  (kind, detail, second) = local_resolution(tz, civil, nanosecond, disambiguation)
  if eq(kind, "") then Some(Zoned { instant: instant_from_unix(second, nanosecond), zone: tz }) else if eq(kind, "domain") then None else (function |> overflow_failure(detail) |> fail)
}
def zoned_from_local(dt: DateTime, tz: TimeZone, disambiguation: Disambiguation) -> Zoned = {
  (civil, nanosecond) = local_civil(dt)
  (kind, detail, second) = local_resolution(tz, civil, nanosecond, disambiguation)
  if eq(kind, "") then Zoned { instant: instant_from_unix(second, nanosecond), zone: tz } else if eq(kind, "domain") then ("zoned_from_local" |> domain_failure(detail) |> fail) else ("zoned_from_local" |> overflow_failure(detail) |> fail)
}
def try_zoned_from_local(dt: DateTime, tz: TimeZone, disambiguation: Disambiguation) -> Option[Zoned] = {
  (civil, nanosecond) = local_civil(dt)
  resolved("try_zoned_from_local", tz, civil, nanosecond, disambiguation)
}
-- Arithmetic. A duration moves along the instant line; a period moves the
-- local date on the wall clock and then re-resolves the local reading.
def zoned_add_duration(z: Zoned, d: Duration) -> Zoned = {
  second = instant_unix_second(z.instant)
  nanos = add(instant_nanosecond(z.instant), duration_nanosecond(d))
  carry = if gte(nanos, 1000000000i64) then 1i64 else 0i64
  base = add(second, carry)
  if or(gt(duration_second(d), sub(253402214400i64, base)), lt(duration_second(d), sub(-377705030401i64, base))) then fail(overflow_failure("zoned_add_duration", joined([instant_to_string(z.instant), " plus ", duration_to_string(d), " is outside the supported instant range"]))) else {
    moved = add(base, duration_second(d))
    nanosecond = sub(nanos, mul(carry, 1000000000i64))
    if offset_in_force(z.zone, moved).0 then Zoned { instant: instant_from_unix(moved, nanosecond), zone: z.zone } else ("zoned_add_duration"
    |> domain_failure(uncovered_text(z.zone, moved, nanosecond))
    |> fail)
  }
}
-- A zero period returns `z` itself under every policy: re-resolving its
-- wall reading would move the later occurrence of a fold, or fail near the
-- end of a zone's coverage, for an addition that changes nothing.
def zoned_add_period(z: Zoned, p: Period, overflow: DayOverflow, disambiguation: Disambiguation) -> Zoned =
  if (p
  |> period_months
  |> eq(0i64)
  |> and(eq(period_days(p), 0i64))) then z else wall_clock_period(z, p, overflow, disambiguation)
-- `p` added to the wall reading of `z`, re-resolved under `disambiguation`.
def wall_clock_period(z: Zoned, p: Period, overflow: DayOverflow, disambiguation: Disambiguation) -> Zoned = {
  local = zoned_local(z)
  d = datetime_date(local)
  current =
    d
    |> date_year
    |> mul(12i64)
    |> add(sub(date_month(d), 1i64))
  months = period_months(p)
  if or(gt(months, sub(119999i64, current)), lt(months, sub(-119988i64, current))) then fail(overflow_failure("zoned_add_period", joined([datetime_to_string(local), " plus ", period_to_string(p), " is outside the supported date range"]))) else match try_date_add_months(d, months, overflow) with {
    | Some(moved) => {
    days = period_days(p)
    if or(gt(days, sub(2932896i64, date_epoch_day(moved))), lt(days, sub(-4371587i64, date_epoch_day(moved)))) then fail(overflow_failure("zoned_add_period", joined([datetime_to_string(local), " plus ", period_to_string(p), " is outside the supported date range"]))) else {
      (civil, nanosecond) = local_civil(datetime(date_add_days(moved, days), datetime_time(local)))
      (kind, detail, second) = local_resolution(z.zone, civil, nanosecond, disambiguation)
      if eq(kind, "") then Zoned { instant: instant_from_unix(second, nanosecond), zone: z.zone } else if eq(kind, "domain") then ("zoned_add_period" |> domain_failure(detail) |> fail) else ("zoned_add_period" |> overflow_failure(detail) |> fail)
    }
  }
    | None => {
    target = add(current, months)
    year = floor_div(target, 12i64)
    month = target |> euclid_rem(12i64) |> add(1i64)
    month_text =
      date(year, month, 1i64)
      |> date_to_string
      |> string_slice(0i64, sub(string_len(date_to_string(date(year, month, 1i64))), 3i64))
    fail(domain_failure("zoned_add_period", joined(["day ", to_string(date_day(d)), " is outside 1..", to_string(days_in_month(year, month)), " for ", month_text])))
  }
  }
}
-- Text: RFC 9557 with the text profile's datetime and offset forms.
def zoned_to_string(z: Zoned) -> string = joined([datetime_to_string(zoned_local(z)), zoned_offset_text(offset_in_force(z.zone, instant_unix_second(z.instant)).1), "[", z.zone.name, "]"])
def char_at(text: string, idx: i64) -> string = if (idx |> gte(0i64) |> and(lt(idx, string_len(text)))) then string_slice(text, idx, 1i64) else ""
-- The index of the first `target` at or after `start`, or -1.
def find_from(text: string, target: string, start: i64) -> i64 = fold(fn (acc: i64, idx: i64) -> if gte(acc, 0i64) then acc else if (text |> char_at(idx) |> eq(target)) then idx else acc, -1i64, range(start, string_len(text)))
-- The index of the first character after `start` that can begin an offset.
def offset_start(text: string, start: i64) -> i64 =
  fold(fn (acc: i64, idx: i64) -> if gte(acc, 0i64) then acc else if (text
  |> char_at(idx)
  |> eq("Z")
  |> or(eq(char_at(text, idx), "z"))
  |> or(eq(char_at(text, idx), "+"))
  |> or(eq(char_at(text, idx), "-"))) then idx else acc, -1i64, range(start, string_len(text)))
def separator_index(text: string) -> i64 =
  fold(fn (acc: i64, idx: i64) -> if gte(acc, 0i64) then acc else if (text
  |> char_at(idx)
  |> eq("T")
  |> or(eq(char_at(text, idx), "t"))
  |> or(eq(char_at(text, idx), " "))) then idx else acc, -1i64, range(0i64, string_len(text)))
def is_lower_key_initial(code: i64) -> bool = or(in_span(code, 97i64, 122i64), eq(code, 95i64))
def is_key(key: string) -> bool =
  and(gt(string_len(key), 0i64), fold(fn (acc: bool, idx: i64) -> {
    code = char_code(string_slice(key, idx, 1i64))
    and(acc, if eq(idx, 0i64) then is_lower_key_initial(code) else (code
    |> is_lower_key_initial
    |> or(is_digit(code))
    |> or(eq(code, 45i64))))
  }, true, range(0i64, string_len(key))))
-- RFC 9557 `suffix-values`: alphanumeric runs joined by single hyphens.
def is_suffix_values(values: string) -> bool = {
  scanned = fold(fn (acc: (bool, i64), idx: i64) -> {
    code = char_code(string_slice(values, idx, 1i64))
    if not(acc.0) then acc else if eq(code, 45i64) then (gt(acc.1, 0i64), 0i64) else (code |> is_letter |> or(is_digit(code)), add(acc.1, 1i64))
  }, (true, 0i64), range(0i64, string_len(values)))
  and(scanned.0, gt(scanned.1, 0i64))
}
-- The problem with the annotations from `start` to the end: (problem, zone
-- name, critical). The first annotation names the zone; each later one is a
-- `key=value` tag.
-- RFC 9557 §3.3 over the whole sequence of suffix tags, each (key, value,
-- critical), grouped by key: a critical tag must name the recognized key
-- `u-ca`; a key whose tags give more than one value fails when any of them is
-- critical; otherwise each key's first tag decides, so the first `u-ca` must
-- name a supported calendar and every later tag is ignored. The result is
-- the problem, or "" when the tags are accepted.
def tags_problem(text: string, tags: List[(string, string, bool)]) -> string = {
  unknown = filter(fn (tag: (string, string, bool)) -> and(tag.2, neq(tag.0, "u-ca")), tags)
  conflicted = filter(fn (tag: (string, string, bool)) -> key_conflicts(tags, tag.0), tags)
  calendars = filter(fn (tag: (string, string, bool)) -> eq(tag.0, "u-ca"), tags)
  if gt(len(unknown), 0i64) then joined([quoted(text), " has the unknown critical annotation ", quoted(joined([index(unknown, 0i64).0, "=", index(unknown, 0i64).1]))]) else if gt(len(conflicted), 0i64) then joined([quoted(text), " gives the critical key ", quoted(index(conflicted, 0i64).0), " more than one value"]) else if eq(len(calendars), 0i64) then "" else if (index(calendars, 0i64).1
  |> eq("iso8601")
  |> or(eq(index(calendars, 0i64).1, "gregory"))) then "" else joined([quoted(text), " names the calendar ", quoted(index(calendars, 0i64).1), ", not iso8601 or gregory"])
}
-- Whether the tags with `key` give more than one value while one of them is
-- critical.
def key_conflicts(tags: List[(string, string, bool)], key: string) -> bool = {
  same = filter(fn (tag: (string, string, bool)) -> eq(tag.0, key), tags)
  values_differ = fold(fn (acc: bool, tag: (string, string, bool)) -> or(acc, neq(tag.1, index(same, 0i64).1)), false, same)
  any_critical = fold(fn (acc: bool, tag: (string, string, bool)) -> or(acc, tag.2), false, same)
  and(values_differ, any_critical)
}
def annotation_reading(text: string, start: i64) -> (string, string, bool) = {
  size = string_len(text)
  -- (problem, next index, zone name, critical, annotations seen, suffix tags
  -- as (key, value, critical)), judged as a whole after the scan
  scanned = fold(fn (acc: (string, i64, string, bool, i64, List[(string, string, bool)]), step: i64) -> if neq(acc.0, "") then acc else if gte(acc.1, size) then acc else if (text |> char_at(acc.1) |> neq("[")) then (joined([quoted(text), " has text after its annotations"]), acc.1, acc.2, acc.3, acc.4, acc.5) else {
    close = find_from(text, "]", acc.1)
    if lt(close, 0i64) then (joined([quoted(text), " has an unclosed annotation"]), acc.1, acc.2, acc.3, acc.4, acc.5) else {
      critical = text |> char_at(add(acc.1, 1i64)) |> eq("!")
      body_start = if critical then add(acc.1, 2i64) else add(acc.1, 1i64)
      body = string_slice(text, body_start, sub(close, body_start))
      equals = find_from(body, "=", 0i64)
      if eq(acc.4, 0i64) then if gte(equals, 0i64) then (joined([quoted(text), " has no time zone annotation"]), acc.1, acc.2, acc.3, acc.4, acc.5) else if zone_annotation_ok(body) then ("", add(close, 1i64), body, critical, 1i64, acc.5) else (joined([quoted(text), " has an invalid time zone annotation ", quoted(body)]), acc.1, acc.2, acc.3, acc.4, acc.5) else if lt(equals, 0i64) then (joined([quoted(text), " has a second time zone annotation"]), acc.1, acc.2, acc.3, acc.4, acc.5) else {
        key = string_slice(body, 0i64, equals)
        value = string_slice(body, add(equals, 1i64), sub(sub(string_len(body), equals), 1i64))
        if (key
        |> is_key
        |> and(is_suffix_values(value))
        |> not) then (joined([quoted(text), " has an invalid annotation ", quoted(body)]), acc.1, acc.2, acc.3, acc.4, acc.5) else ("", add(close, 1i64), acc.2, acc.3, add(acc.4, 1i64), append(acc.5, (key, value, critical)))
      }
    }
  }, ("", start, "", false, 0i64, []), range(0i64, size))
  (problem, next, zone_name, critical, seen, tags) = scanned
  if neq(problem, "") then (problem, "", false) else if lt(next, size) then (joined([quoted(text), " has text after its annotations"]), "", false) else if eq(seen, 0i64) then (joined([quoted(text), " has no time zone annotation"]), "", false) else {
    judged = tags_problem(text, tags)
    if eq(judged, "") then ("", zone_name, critical) else (judged, "", false)
  }
}
-- A zone annotation is a time zone name or a numeric offset `±HH:MM[:SS]`.
def zone_annotation_ok(body: string) -> bool =
  if (body
  |> char_at(0i64)
  |> eq("+")
  |> or(eq(char_at(body, 0i64), "-"))) then match try_parse_offset(body) with {
    | Some(_) => true
    | None => false
  } else is_zone_name(body)
def zoned_text_reading(text: string) -> (string, ZonedText) = {
  placeholder = ZonedText { written: datetime(date(1970i64, 1i64, 1i64), time_from_nanosecond_of_day(0i64)), offset: None, zone_name: "UTC", critical: false }
  bracket = find_from(text, "[", 0i64)
  separator = separator_index(text)
  if lt(bracket, 0i64) then (joined([quoted(text), " has no time zone annotation"]), placeholder) else {
    head_end = bracket
    start = if gte(separator, 0i64) then offset_start(text, add(separator, 1i64)) else -1i64
    if (start |> lt(0i64) |> or(gt(start, head_end))) then (joined([quoted(text), " is not a zoned datetime in the text profile"]), placeholder) else {
      local_text = string_slice(text, 0i64, start)
      offset_text = string_slice(text, start, sub(head_end, start))
      match try_parse_datetime(local_text) with {
        | Some(local) => match try_parse_offset(offset_text) with {
        | Some(offset) => {
        (problem, zone_name, critical) = annotation_reading(text, bracket)
        unknown =
          offset_text
          |> eq("Z")
          |> or(eq(offset_text, "z"))
          |> or(offset
        |> offset_seconds
        |> eq(0i64)
        |> and(eq(string_slice(offset_text, 0i64, 1i64), "-")))
        if eq(problem, "") then ("", ZonedText { written: local, offset: if unknown then None else Some(offset), zone_name, critical }) else (problem, placeholder)
      }
        | None => (joined([quoted(offset_text), " in ", quoted(text), " is not an offset in the text profile"]), placeholder)
      }
        | None => (joined([quoted(local_text), " in ", quoted(text), " is not a datetime in the text profile"]), placeholder)
      }
    }
  }
}
def parse_zoned_text(text: string) -> ZonedText = {
  (problem, zt) = zoned_text_reading(text)
  if eq(problem, "") then zt else ("parse_zoned_text" |> domain_failure(problem) |> fail)
}
def try_parse_zoned_text(text: string) -> Option[ZonedText] = {
  (problem, zt) = zoned_text_reading(text)
  if eq(problem, "") then Some(zt) else None
}
-- (kind, detail, unix second) for a zoned text resolved against `tz`. An
-- absent offset names the UTC instant of the written date and time under
-- every policy.
def text_resolution(zt: ZonedText, tz: TimeZone, conflict: OffsetConflict) -> (string, string, i64) = {
  (civil, nanosecond) = local_civil(zt.written)
  match zt.offset with {
    | None => if (civil |> in_span(-377705030401i64, 253402214400i64) |> not) then ("overflow", joined([datetime_to_string(zt.written), "Z is outside the supported instant range"]), 0i64) else if offset_in_force(tz, civil).0 then ("", "", civil) else ("domain", uncovered_text(tz, civil, nanosecond), 0i64)
    | Some(o) => written_offset_resolution(zt, tz, conflict, civil, nanosecond, offset_seconds(o))
  }
}
-- (kind, detail, unix second) for a zoned text whose offset `written` is
-- known, resolved under `conflict`.
def written_offset_resolution(zt: ZonedText, tz: TimeZone, conflict: OffsetConflict, civil: i64, nanosecond: i64, written: i64) -> (string, string, i64) = {
  trial = sub(civil, written)
  shown = joined([datetime_to_string(zt.written), zoned_offset_text(written)])
  match conflict with {
    | UseZoneRules => {
    (kind, detail, second) = local_resolution(tz, civil, nanosecond, RejectNonUniqueLocal)
    (kind, detail, second)
  }
    | UseWrittenOffset => if (trial |> in_span(-377705030401i64, 253402214400i64) |> not) then ("overflow", joined([shown, " is outside the supported instant range"]), 0i64) else {
    (covered, actual) = offset_in_force(tz, trial)
    if not(covered) then ("domain", uncovered_text(tz, trial, nanosecond), 0i64) else if (zt.critical |> and(neq(actual, written))) then ("domain", joined([shown, " has a critical zone annotation, but ", quoted(tz.name), " has offset ", zoned_offset_text(actual), " there"]), 0i64) else ("", "", trial)
  }
    | RejectOffsetMismatch => {
    (kind, detail, _) = local_resolution(tz, civil, nanosecond, EarlierInstant)
    if eq(kind, "overflow") then (kind, detail, 0i64) else if (trial |> in_span(-377705030401i64, 253402214400i64) |> not) then ("overflow", joined([shown, " is outside the supported instant range"]), 0i64) else {
      (covered, actual) = offset_in_force(tz, trial)
      if (eq(kind, "domain") |> or(not(covered))) then ("domain", if eq(kind, "domain") then detail else uncovered_text(tz, trial, nanosecond), 0i64) else if neq(actual, written) then ("domain", joined([shown, " does not match ", quoted(tz.name), ", which has offset ", zoned_offset_text(actual), " there"]), 0i64) else ("", "", trial)
    }
  }
  }
}
def zoned_from_text(zt: ZonedText, tz: TimeZone, conflict: OffsetConflict) -> Zoned = {
  (kind, detail, second) = text_resolution(zt, tz, conflict)
  if eq(kind, "") then Zoned { instant: instant_from_unix(second, local_civil(zt.written).1), zone: tz } else if eq(kind, "domain") then ("zoned_from_text" |> domain_failure(detail) |> fail) else ("zoned_from_text" |> overflow_failure(detail) |> fail)
}
def try_zoned_from_text(zt: ZonedText, tz: TimeZone, conflict: OffsetConflict) -> Option[Zoned] = {
  (kind, detail, second) = text_resolution(zt, tz, conflict)
  if eq(kind, "") then Some(Zoned { instant: instant_from_unix(second, local_civil(zt.written).1), zone: tz }) else if eq(kind, "domain") then None else ("try_zoned_from_text" |> overflow_failure(detail) |> fail)
}
