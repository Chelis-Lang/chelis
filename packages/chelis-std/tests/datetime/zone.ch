module Std.Tests.Datetime.Zone
import Std.Datetime (DateTime, ClampToMonthEnd, RejectInvalidDay, date, time, datetime, datetime_to_string, instant_from_unix, instant_unix_second, offset_from_seconds, offset_seconds, period, duration)
import Std.Datetime.Zone (Disambiguation, EarlierInstant, LaterInstant, CompatibleInstant, RejectNonUniqueLocal, OffsetConflict, UseWrittenOffset, UseZoneRules, RejectOffsetMismatch, TimeZone, Zoned, ZonedText, time_zone_from_tzif, try_time_zone_from_tzif, time_zone_fixed, time_zone_utc, time_zone_name, time_zone_offset_at, try_time_zone_offset_at, zoned, try_zoned, zoned_from_local, try_zoned_from_local, zoned_instant, zoned_zone, zoned_local, zoned_offset, zoned_add_duration, zoned_add_period, zoned_to_string, parse_zoned_text, try_parse_zoned_text, zoned_from_text, try_zoned_from_text)
import Std.Test (assert_eq, assert_true, assert_false)
-- Expected instants come from Python's datetime and zoneinfo; see
-- [05-OP-73]. TZif files are built below with a slim version 1 block, one
-- designation, and the given transitions, offsets, and footer.
def byte_of(value: i64, shift: i64) -> i64 = {
  r = mod(floor_div(value, fold(fn (acc: i64, k: i64) -> mul(acc, 256i64), 1i64, range(0i64, shift))), 256i64)
  if lt(r, 0i64) then add(r, 256i64) else r
}
def big_endian(value: i64, count: i64) -> List[i64] = map(fn (k: i64) -> byte_of(value, sub(sub(count, 1i64), k)), range(0i64, count))
def codes_of(text: string) -> List[i64] = map(fn (idx: i64) -> char_code(string_slice(text, idx, 1i64)), range(0i64, string_len(text)))
def flat_bytes(parts: List[List[i64]]) -> List[i64] =
  fold(fn (acc: List[i64], part: List[i64]) -> concat(acc, part), [0i64], parts)
  |> skip(1i64)
def header_bytes(version: i64, counts: List[i64]) -> List[i64] = flat_bytes([[84i64, 90i64, 105i64, 102i64, version], map(fn (k: i64) -> 0i64, range(0i64, 15i64)), flat_bytes(map(fn (c: i64) -> big_endian(c, 4i64), counts))])
def tzif_file(version: i64, times: List[i64], indices: List[i64], offsets: List[i64], footer: string) -> List[i64] = flat_bytes([header_bytes(version, [0i64, 0i64, 0i64, 0i64, 1i64, 1i64]), [0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64], header_bytes(version, [0i64, 0i64, 0i64, len(times), len(offsets), 4i64]), flat_bytes(map(fn (t: i64) -> big_endian(t, 8i64), times)), indices, flat_bytes(map(fn (o: i64) -> concat(big_endian(o, 4i64), [0i64, 0i64]), offsets)), [76i64, 77i64, 84i64, 0i64, 10i64], codes_of(footer), [10i64]])
def new_york() -> TimeZone = time_zone_from_tzif("America/New_York", tzif_file(50i64, [1173596400i64], [1i64], [-18000i64, -14400i64], "EST5EDT,M3.2.0,M11.1.0"))
def lord_howe() -> TimeZone = time_zone_from_tzif("Australia/Lord_Howe", tzif_file(50i64, [], [], [37800i64], "<+1030>-10:30<+11>-11,M10.1.0,M4.1.0"))
def short_lived() -> TimeZone = time_zone_from_tzif("Etc/Short_Lived", tzif_file(50i64, [1000000000i64], [1i64], [0i64, 3600i64], ""))
def offset_seconds_at(tz: TimeZone, second: i64) -> i64 = offset_seconds(time_zone_offset_at(tz, instant_from_unix(second, 0i64)))
def local_reading(year: i64, month: i64, day: i64, hour: i64, minute: i64) -> DateTime = datetime(date(year, month, day), time(hour, minute, 0i64, 0i64))
def resolved_second(dt: DateTime, tz: TimeZone, policy: Disambiguation) -> Option[i64] =
  match try_zoned_from_local(dt, tz, policy) with {
    | Some(z) => Some(instant_unix_second(zoned_instant(z)))
    | None => None
  }
def second_from_text(text: string, tz: TimeZone, conflict: OffsetConflict) -> Option[i64] =
  match try_zoned_from_text(parse_zoned_text(text), tz, conflict) with {
    | Some(z) => Some(instant_unix_second(zoned_instant(z)))
    | None => None
  }
def test_utc_from_tzif_is_time_zone_utc() -> unit ! { Test } = {
  _ = assert_eq(time_zone_from_tzif("UTC", tzif_file(50i64, [], [], [0i64], "UTC0")), time_zone_utc(), "a TZif UTC equals time_zone_utc")
  _ = assert_eq(time_zone_name(time_zone_utc()), "UTC", "the UTC name")
  assert_eq(offset_seconds_at(time_zone_utc(), 253402214400i64), 0i64, "UTC at the end of the range")
}
def test_utc_differs_from_other_fixed_zones() -> unit ! { Test } = {
  _ = assert_false(eq(time_zone_fixed(offset_from_seconds(0i64)), time_zone_utc()), "+00:00 and UTC are different zones")
  assert_false(eq(time_zone_from_tzif("UTC", tzif_file(50i64, [], [], [3600i64], "UTC-1")), time_zone_utc()), "another offset is another zone")
}
def test_fixed_zone_names() -> unit ! { Test } = {
  _ = assert_eq(time_zone_name(time_zone_fixed(offset_from_seconds(19800i64))), "+05:30", "east")
  _ = assert_eq(time_zone_name(time_zone_fixed(offset_from_seconds(-14400i64))), "-04:00", "west")
  _ = assert_eq(time_zone_name(time_zone_fixed(offset_from_seconds(0i64))), "+00:00", "zero")
  assert_eq(time_zone_name(time_zone_fixed(offset_from_seconds(19815i64))), "+05:30:15", "seconds")
}
def test_type_zero_applies_before_the_first_transition() -> unit ! { Test } = {
  _ = assert_eq(offset_seconds_at(new_york(), 1173596399i64), -18000i64, "the second before the transition")
  assert_eq(offset_seconds_at(new_york(), -377705030401i64), -18000i64, "the start of the range")
}
def test_the_transition_takes_effect_at_its_second() -> unit ! { Test } = {
  _ = assert_eq(offset_seconds_at(new_york(), 1173596400i64), -14400i64, "at the transition")
  assert_false(eq(offset_seconds_at(new_york(), 1173596399i64), offset_seconds_at(new_york(), 1173596400i64)), "the offset changes there")
}
def test_footer_rule_northern_daylight_time() -> unit ! { Test } = {
  _ = assert_eq(offset_seconds_at(new_york(), 1772953199i64), -18000i64, "before the 2026 start")
  _ = assert_eq(offset_seconds_at(new_york(), 1772953200i64), -14400i64, "at the 2026 start")
  _ = assert_eq(offset_seconds_at(new_york(), 1793512799i64), -14400i64, "before the 2026 end")
  _ = assert_eq(offset_seconds_at(new_york(), 1793512800i64), -18000i64, "at the 2026 end")
  assert_eq(offset_seconds_at(new_york(), 253402214400i64), -18000i64, "December 9999")
}
def test_footer_rule_southern_daylight_time() -> unit ! { Test } = {
  _ = assert_eq(offset_seconds_at(lord_howe(), 1775314799i64), 39600i64, "before the April end")
  _ = assert_eq(offset_seconds_at(lord_howe(), 1775314800i64), 37800i64, "at the April end")
  _ = assert_eq(offset_seconds_at(lord_howe(), 1791041399i64), 37800i64, "before the October start")
  assert_eq(offset_seconds_at(lord_howe(), 1791041400i64), 39600i64, "at the October start")
}
def test_julian_days_skip_february_29() -> unit ! { Test } = {
  julian = time_zone_from_tzif("Etc/Julian", tzif_file(50i64, [], [], [0i64], "STD0DST,J60/0,J300/0"))
  _ = assert_eq(offset_seconds_at(julian, 1709208000i64), 0i64, "J60 is 1 March, so 29 February 2024 is standard")
  assert_eq(offset_seconds_at(julian, 1709294400i64), 3600i64, "1 March 2024 is daylight")
}
def test_zero_based_days_count_february_29() -> unit ! { Test } = {
  zero_based = time_zone_from_tzif("Etc/Zero_Based", tzif_file(50i64, [], [], [0i64], "STD0DST,59/0,299/0"))
  _ = assert_eq(offset_seconds_at(zero_based, 1709208000i64), 3600i64, "day 59 of 2024 is 29 February")
  assert_eq(offset_seconds_at(zero_based, 1709121600i64), 0i64, "28 February 2024 is standard")
}
def test_version_3_transition_hours_past_24() -> unit ! { Test } = {
  jerusalem = time_zone_from_tzif("Asia/Jerusalem", tzif_file(51i64, [], [], [7200i64], "IST-2IDT,M3.4.4/26,M10.5.0"))
  _ = assert_eq(offset_seconds_at(jerusalem, 1774569599i64), 7200i64, "before 26:00 on 26 March 2026")
  assert_eq(offset_seconds_at(jerusalem, 1774569600i64), 10800i64, "at 26:00 on 26 March 2026")
}
def test_version_2_rejects_the_hour_extension() -> unit ! { Test } = assert_eq(try_time_zone_from_tzif("Asia/Jerusalem", tzif_file(50i64, [], [], [7200i64], "IST-2IDT,M3.4.4/26,M10.5.0")), None, "26:00 needs version 3")
def test_daylight_time_all_year() -> unit ! { Test } = {
  all_year = time_zone_from_tzif("Etc/All_Year", tzif_file(51i64, [], [], [-14400i64], "EST5EDT,0/0,J365/25"))
  _ = assert_eq(offset_seconds_at(all_year, 1767225600i64), -14400i64, "1 January 2026")
  _ = assert_eq(offset_seconds_at(all_year, 1798761599i64), -14400i64, "31 December 2026")
  assert_eq(offset_seconds_at(all_year, 1798776000i64), -14400i64, "the new year 2027")
}
def test_daylight_time_ending_before_the_new_year() -> unit ! { Test } = {
  almost = time_zone_from_tzif("Etc/Almost", tzif_file(51i64, [], [], [-14400i64], "EST5EDT,0/0,J365/23"))
  _ = assert_eq(offset_seconds_at(almost, 1798776000i64), -18000i64, "standard for two hours of 1 January 2027")
  assert_eq(offset_seconds_at(almost, 1798779600i64), -14400i64, "daylight again from 00:00 standard")
}
def test_an_empty_footer_covers_up_to_the_last_transition() -> unit ! { Test } = {
  _ = assert_eq(offset_seconds_at(short_lived(), 999999999i64), 0i64, "before the last transition")
  assert_true(try_zoned(instant_from_unix(999999999i64, 999999999i64), short_lived())
  |> eq(None)
  |> not, "the last nanosecond before it")
}
def test_an_empty_footer_leaves_the_rest_uncovered() -> unit ! { Test } = {
  _ = assert_eq(try_time_zone_offset_at(short_lived(), instant_from_unix(1000000000i64, 0i64)), None, "at the last transition")
  _ = assert_eq(try_zoned(instant_from_unix(2000000000i64, 0i64), short_lived()), None, "after it")
  assert_eq(try_zoned_from_local(local_reading(2001i64, 9i64, 8i64, 12i64, 0i64), short_lived(), EarlierInstant), None, "within a day of it")
}
def test_well_formed_files_parse() -> unit ! { Test } =
  assert_true(try_time_zone_from_tzif("Etc/Valid", tzif_file(50i64, [1000000000i64], [1i64], [0i64, 3600i64], "STD-1"))
  |> eq(None)
  |> not, "a valid file")
def test_malformed_files_are_rejected() -> unit ! { Test } = {
  valid = tzif_file(50i64, [1000000000i64], [1i64], [0i64, 3600i64], "STD-1")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Bad", concat([84i64, 90i64, 105i64, 70i64], skip(valid, 4i64))), None, "bad magic")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Bad", take(valid, 60i64)), None, "truncated")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Bad", tzif_file(0i64, [], [], [0i64], "")), None, "version 1")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Bad", tzif_file(53i64, [], [], [0i64], "UTC0")), None, "version 5")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Bad", concat(valid, [0i64])), None, "a byte after the footer")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Bad", concat(take(valid, 10i64), concat([256i64], skip(valid, 11i64)))), None, "a byte outside 0..255")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Bad", tzif_file(50i64, [], [], [], "UTC0")), None, "no time types")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Bad", tzif_file(50i64, [1000000000i64], [2i64], [0i64, 3600i64], "STD-1")), None, "a type index out of range")
  assert_eq(try_time_zone_from_tzif("not a name", valid), None, "a name outside RFC 9557")
}
def test_offsets_are_bounded() -> unit ! { Test } = {
  _ = assert_true(try_time_zone_from_tzif("Etc/Edge", tzif_file(50i64, [], [], [86399i64], ""))
  |> eq(None)
  |> not, "23:59:59 east")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Edge", tzif_file(50i64, [], [], [86400i64], "")), None, "a whole day east")
  assert_eq(try_time_zone_from_tzif("Etc/Edge", tzif_file(50i64, [], [], [0i64], "STD24")), None, "a footer offset of 24 hours")
}
def test_transitions_strictly_increase() -> unit ! { Test } = {
  _ = assert_true(try_time_zone_from_tzif("Etc/Two", tzif_file(50i64, [900000000i64, 1000000000i64], [1i64, 0i64], [0i64, 3600i64], "STD0"))
  |> eq(None)
  |> not, "increasing")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Two", tzif_file(50i64, [1000000000i64, 1000000000i64], [1i64, 0i64], [0i64, 3600i64], "STD0")), None, "equal")
  assert_eq(try_time_zone_from_tzif("Etc/Two", tzif_file(50i64, [1000000000i64, 900000000i64], [1i64, 0i64], [0i64, 3600i64], "STD0")), None, "decreasing")
}
def test_the_footer_matches_the_last_transition() -> unit ! { Test } = {
  _ = assert_true(try_time_zone_from_tzif("Etc/Match", tzif_file(50i64, [1000000000i64], [1i64], [0i64, 3600i64], "STD-1"))
  |> eq(None)
  |> not, "consistent")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Match", tzif_file(50i64, [1000000000i64], [1i64], [0i64, 3600i64], "STD0")), None, "inconsistent")
  _ = assert_eq(try_time_zone_from_tzif("Etc/Match", tzif_file(50i64, [], [], [0i64], "STD0DST")), None, "daylight time without a rule")
  assert_eq(try_time_zone_from_tzif("Etc/Match", tzif_file(50i64, [], [], [0i64], "ST0")), None, "a two-letter designation")
}
def test_a_unique_local_reading_ignores_the_policy() -> unit ! { Test } = {
  noon = local_reading(2026i64, 7i64, 1i64, 12i64, 0i64)
  _ = assert_eq(resolved_second(noon, new_york(), EarlierInstant), Some(1782921600i64), "earlier")
  _ = assert_eq(resolved_second(noon, new_york(), LaterInstant), Some(1782921600i64), "later")
  assert_eq(resolved_second(noon, new_york(), RejectNonUniqueLocal), Some(1782921600i64), "reject")
}
def kiritimati() -> TimeZone = time_zone_from_tzif("Pacific/Kiritimati", tzif_file(50i64, [-2177415040i64, 307622400i64, 788868000i64], [1i64, 2i64, 3i64], [-37760i64, -38400i64, -36000i64, 50400i64], "<+14>-14"))
def test_offsets_far_from_the_reading_make_no_trial() -> unit ! { Test } = {
  _ = assert_eq(resolved_second(local_reading(9999i64, 12i64, 30i64, 14i64, 0i64), kiritimati(), EarlierInstant), Some(253402128000i64), "-10:40 ended in 1979, so it makes no trial near year 10000")
  assert_eq(resolved_second(local_reading(1994i64, 12i64, 31i64, 12i64, 0i64), kiritimati(), LaterInstant), Some(788911200i64), "the skipped day of 1994 resolves after the gap")
}
def test_the_skipped_day_has_no_unique_reading() -> unit ! { Test } = assert_eq(resolved_second(local_reading(1994i64, 12i64, 31i64, 12i64, 0i64), kiritimati(), RejectNonUniqueLocal), None, "31 December 1994 does not exist in Kiritimati")
def test_a_gap_under_each_policy() -> unit ! { Test } = {
  skipped = local_reading(2026i64, 3i64, 8i64, 2i64, 30i64)
  _ = assert_eq(resolved_second(skipped, new_york(), EarlierInstant), Some(1772951400i64), "before the gap, read at -04:00")
  _ = assert_eq(resolved_second(skipped, new_york(), LaterInstant), Some(1772955000i64), "after the gap, read at -05:00")
  assert_eq(resolved_second(skipped, new_york(), CompatibleInstant), Some(1772955000i64), "compatible is later in a gap")
}
def test_a_gap_rejects_a_non_unique_local_reading() -> unit ! { Test } = assert_eq(resolved_second(local_reading(2026i64, 3i64, 8i64, 2i64, 30i64), new_york(), RejectNonUniqueLocal), None, "reject in a gap")
def test_a_fold_under_each_policy() -> unit ! { Test } = {
  repeated = local_reading(2026i64, 11i64, 1i64, 1i64, 30i64)
  _ = assert_eq(resolved_second(repeated, new_york(), EarlierInstant), Some(1793511000i64), "the first 01:30, at -04:00")
  _ = assert_eq(resolved_second(repeated, new_york(), LaterInstant), Some(1793514600i64), "the second 01:30, at -05:00")
  assert_eq(resolved_second(repeated, new_york(), CompatibleInstant), Some(1793511000i64), "compatible is earlier in a fold")
}
def test_a_fold_rejects_a_non_unique_local_reading() -> unit ! { Test } = assert_eq(resolved_second(local_reading(2026i64, 11i64, 1i64, 1i64, 30i64), new_york(), RejectNonUniqueLocal), None, "reject in a fold")
def test_zoned_accessors() -> unit ! { Test } = {
  z = zoned(instant_from_unix(1793511000i64, 0i64), new_york())
  _ = assert_eq(datetime_to_string(zoned_local(z)), "2026-11-01T01:30:00", "the local reading")
  _ = assert_eq(offset_seconds(zoned_offset(z)), -14400i64, "the offset")
  assert_eq(zoned_zone(z), new_york(), "the zone")
}
def test_a_duration_moves_on_the_instant_line() -> unit ! { Test } = {
  before = zoned_from_local(local_reading(2026i64, 3i64, 8i64, 1i64, 30i64), new_york(), RejectNonUniqueLocal)
  _ = assert_eq(zoned_to_string(zoned_add_duration(before, duration(3600i64, 0i64))), "2026-03-08T03:30:00-04:00[America/New_York]", "one hour crosses the gap")
  assert_false(eq(zoned_to_string(zoned_add_duration(before, duration(3600i64, 0i64))), zoned_to_string(zoned_add_period(before, period(0i64, 1i64), ClampToMonthEnd, CompatibleInstant))), "an hour is not a day")
}
def test_a_period_moves_on_the_wall_clock() -> unit ! { Test } = {
  before = zoned_from_local(local_reading(2026i64, 3i64, 7i64, 2i64, 30i64), new_york(), RejectNonUniqueLocal)
  _ = assert_eq(zoned_to_string(zoned_add_period(before, period(0i64, 1i64), ClampToMonthEnd, CompatibleInstant)), "2026-03-08T03:30:00-04:00[America/New_York]", "into the gap, compatible")
  _ = assert_eq(zoned_to_string(zoned_add_period(before, period(0i64, 1i64), ClampToMonthEnd, EarlierInstant)), "2026-03-08T01:30:00-05:00[America/New_York]", "into the gap, earlier")
  assert_eq(zoned_to_string(zoned_add_period(before, period(1i64, 0i64), RejectInvalidDay, RejectNonUniqueLocal)), "2026-04-07T02:30:00-04:00[America/New_York]", "a month later")
}
def test_text_round_trips() -> unit ! { Test } = {
  z = zoned(instant_from_unix(1793514600i64, 0i64), new_york())
  _ = assert_eq(zoned_to_string(z), "2026-11-01T01:30:00-05:00[America/New_York]", "RFC 9557 text")
  _ = assert_eq(zoned_from_text(parse_zoned_text(zoned_to_string(z)), new_york(), RejectOffsetMismatch), z, "the round trip")
  assert_eq(zoned_to_string(zoned(instant_from_unix(0i64, 0i64), time_zone_utc())), "1970-01-01T00:00:00+00:00[UTC]", "offset zero is +00:00")
}
def test_parse_zoned_text_fields() -> unit ! { Test } = {
  zt = parse_zoned_text("2026-11-01T01:30:00.5-04:00[!America/New_York][u-ca=iso8601][_x=elective]")
  _ = assert_eq(datetime_to_string(zt.local), "2026-11-01T01:30:00.5", "the local reading")
  _ = assert_eq(offset_seconds(zt.offset), -14400i64, "the offset")
  _ = assert_eq(zt.zone_name, "America/New_York", "the zone name")
  _ = assert_true(zt.critical, "critical")
  _ = assert_false(parse_zoned_text("2026-11-01T01:30:00Z[UTC][u-ca=gregory]").critical, "elective")
  assert_eq(parse_zoned_text("2026-11-01T01:30:00+05:30[+05:30]").zone_name, "+05:30", "a numeric annotation")
}
def test_parse_zoned_text_rejects() -> unit ! { Test } = {
  _ = assert_eq(try_parse_zoned_text("2026-11-01T01:30:00-04:00"), None, "no annotation")
  _ = assert_eq(try_parse_zoned_text("2026-11-01T01:30:00-04:00[America/New_York][u-ca=hebrew]"), None, "another calendar")
  _ = assert_eq(try_parse_zoned_text("2026-11-01T01:30:00-04:00[America/New_York][!x-y=z]"), None, "an unknown critical tag")
  _ = assert_eq(try_parse_zoned_text("2026-11-01T01:30:00-04:00[America/../York]"), None, "a dot-dot part")
  _ = assert_eq(try_parse_zoned_text("2026-11-01T01:30:00-04:00[Z]x"), None, "text after the annotations")
  assert_eq(try_parse_zoned_text("2026-02-30T01:30:00-04:00[America/New_York]"), None, "an invalid date")
}
def test_written_offsets_select_a_fold_occurrence() -> unit ! { Test } = {
  _ = assert_eq(second_from_text("2026-11-01T01:30:00-04:00[America/New_York]", new_york(), RejectOffsetMismatch), Some(1793511000i64), "-04:00 is the first 01:30")
  _ = assert_eq(second_from_text("2026-11-01T01:30:00-05:00[America/New_York]", new_york(), RejectOffsetMismatch), Some(1793514600i64), "-05:00 is the second 01:30")
  assert_eq(second_from_text("2026-11-01T01:30:00-05:00[America/New_York]", new_york(), UseWrittenOffset), Some(1793514600i64), "the written offset")
}
def test_written_offsets_that_do_not_match() -> unit ! { Test } = {
  _ = assert_eq(second_from_text("2026-07-01T12:00:00-05:00[America/New_York]", new_york(), RejectOffsetMismatch), None, "a mismatch is rejected")
  _ = assert_eq(second_from_text("2026-07-01T12:00:00-05:00[America/New_York]", new_york(), UseWrittenOffset), Some(1782925200i64), "an elective annotation keeps the written instant")
  _ = assert_eq(second_from_text("2026-07-01T12:00:00-05:00[!America/New_York]", new_york(), UseWrittenOffset), None, "a critical annotation fails")
  _ = assert_eq(second_from_text("2026-07-01T12:00:00-05:00[America/New_York]", new_york(), UseZoneRules), Some(1782921600i64), "the zone's rules")
  assert_eq(second_from_text("2026-11-01T01:30:00-05:00[America/New_York]", new_york(), UseZoneRules), None, "the zone's rules in a fold")
}
