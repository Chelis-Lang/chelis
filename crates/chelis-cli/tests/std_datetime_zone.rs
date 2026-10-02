//! chelis#2862: `Std.Datetime.Zone` under [05-OP-73].
//!
//! Every failure is a `fail` whose message is exactly
//! `<function>: <kind>: <detail>`, and no primitive numeric trap escapes a
//! call made with extreme arguments. Each suite runs as one `chelis test`
//! invocation over a generated fixture whose every test makes one call; the
//! runner reports each call's failure message on its FAIL line. The calls
//! read the checked-in TZif fixtures under `tests/fixtures/tzif/` with
//! `read_bytes`. Opaque construction outside the module is checked through
//! `chelis check`, and the differential against Python's `zoneinfo` on both
//! lanes runs `scripts/datetime_zone_differential.py`.

#[path = "common/mod.rs"]
mod common;

#[path = "../../../tests/support/managed_python.rs"]
mod managed_python;

use assert_cmd::Command;
use chelis_backend_c::toolchain::{
    CodegenRequirements, strict_reference_toolchain, test_toolchain,
};
use common::{make_app, write_file};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const IMPORTS: &str = "import Std.Datetime (DateTime, ClampToMonthEnd, RejectInvalidDay, date, time, datetime, instant_from_unix, offset_from_seconds, period, duration)\nimport Std.Datetime.Zone (TimeZone, Zoned, ZonedText, EarlierInstant, LaterInstant, CompatibleInstant, RejectNonUniqueLocal, UseWrittenOffset, UseZoneRules, RejectOffsetMismatch, time_zone_from_tzif, try_time_zone_from_tzif, time_zone_fixed, time_zone_utc, time_zone_offset_at, try_time_zone_offset_at, zoned, try_zoned, zoned_from_local, try_zoned_from_local, zoned_add_duration, zoned_add_period, zoned_to_string, parse_zoned_text, try_parse_zoned_text, zoned_from_text, try_zoned_from_text)\nimport Std.Test (assert_eq)";

/// Helpers for the generated fixtures: zones read from the checked-in TZif
/// files (`@TZIF@` is the fixture directory), a TZif writer for files no
/// fixture holds, and local readings.
const PRELUDE: &str = r#"def i64_minimum() -> i64 = sub(-9223372036854775807i64, 1i64)
def zone_file(path: string) -> List[i64] ! { IO } = read_bytes(string_concat("@TZIF@/", path))
def new_york() -> TimeZone ! { IO } = time_zone_from_tzif("America/New_York", zone_file("zones/America/New_York.tzif"))
def kiritimati() -> TimeZone ! { IO } = time_zone_from_tzif("Pacific/Kiritimati", zone_file("zones/Pacific/Kiritimati.tzif"))
def short_lived() -> TimeZone ! { IO } = time_zone_from_tzif("Etc/Short_Lived", zone_file("synthetic/empty_footer.tzif"))
def london() -> TimeZone ! { IO } = time_zone_from_tzif("Europe/London", zone_file("zones/Europe/London.tzif"))
def byte_of(value: i64, shift: i64) -> i64 = {
  r = mod(floor_div(value, fold(fn (acc: i64, k: i64) -> mul(acc, 256i64), 1i64, range(0i64, shift))), 256i64)
  if lt(r, 0i64) then add(r, 256i64) else r
}
def big_endian(value: i64, count: i64) -> List[i64] = map(fn (k: i64) -> byte_of(value, sub(sub(count, 1i64), k)), range(0i64, count))
def codes_of(text: string) -> List[i64] = map(fn (idx: i64) -> char_code(string_slice(text, idx, 1i64)), range(0i64, string_len(text)))
def flat_bytes(parts: List[List[i64]]) -> List[i64] = fold(fn (acc: List[i64], part: List[i64]) -> concat(acc, part), [0i64], parts) |> skip(1i64)
def header_bytes(version: i64, counts: List[i64]) -> List[i64] = flat_bytes([[84i64, 90i64, 105i64, 102i64, version], map(fn (k: i64) -> 0i64, range(0i64, 15i64)), flat_bytes(map(fn (c: i64) -> big_endian(c, 4i64), counts))])
def tzif_file(version: i64, times: List[i64], indices: List[i64], offsets: List[i64], footer: string) -> List[i64] = flat_bytes([header_bytes(version, [0i64, 0i64, 0i64, 0i64, 1i64, 1i64]), [0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64], header_bytes(version, [0i64, 0i64, 0i64, len(times), len(offsets), 4i64]), flat_bytes(map(fn (t: i64) -> big_endian(t, 8i64), times)), indices, flat_bytes(map(fn (o: i64) -> concat(big_endian(o, 4i64), [0i64, 0i64]), offsets)), [76i64, 77i64, 84i64, 0i64, 10i64], codes_of(footer), [10i64]])
def two_gaps() -> TimeZone = time_zone_from_tzif("Etc/Two_Gaps", tzif_file(50i64, [1000000000i64, 1000000600i64, 1000001200i64], [1i64, 2i64, 1i64], [0i64, 10800i64, -3600i64], "TGA-3"))
def local(year: i64, month: i64, day: i64, hour: i64, minute: i64, second: i64) -> DateTime = datetime(date(year, month, day), time(hour, minute, second, 0i64))
def accepted(text: string) -> bool =
  match try_parse_zoned_text(text) with {
    | Some(_) => true
    | None => false
  }
"#;

/// (test name, expression, exact failure message).
const EXACT_FAILURES: &[(&str, &str, &str)] = &[
    (
        "tzif_bad_magic",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/bad_magic.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": the file does not start with "TZif""#,
    ),
    (
        "tzif_version_1",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/version_1.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": version 1 files have no 64-bit data"#,
    ),
    (
        "tzif_version_5",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/version_5.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": version "5" is not "2", "3" or "4""#,
    ),
    (
        "tzif_truncated",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/truncated.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": the file ends inside the version 2+ header"#,
    ),
    (
        "tzif_zero_typecnt",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/zero_typecnt.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": typecnt is 0"#,
    ),
    (
        "tzif_non_increasing",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/non_increasing.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": transition 1 at unix second 1000000000 is not after transition 0 at unix second 1000000000"#,
    ),
    (
        "tzif_decreasing",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/decreasing.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": transition 1 at unix second 900000000 is not after transition 0 at unix second 1000000000"#,
    ),
    (
        "tzif_offset_bound",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/offset_out_of_bounds.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": time type 1 has offset 86400 s, outside -86399..86399"#,
    ),
    (
        "tzif_leap_seconds",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/leap_seconds.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": leapcnt is 1, and leap-second records are not on the POSIX timescale"#,
    ),
    (
        "tzif_inconsistent_footer",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/inconsistent_footer.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": footer "STD0" gives offset 0 s at the last transition, unix second 1000000000, which gives 3600 s"#,
    ),
    (
        "tzif_dst_without_rule",
        r#"time_zone_from_tzif("Etc/Bad", zone_file("synthetic/dst_without_rule.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": footer "STD-1DST" names daylight saving time without a rule"#,
    ),
    (
        "tzif_extension_in_v2",
        r#"time_zone_from_tzif("Asia/Jerusalem", zone_file("synthetic/jerusalem_in_v2.tzif"))"#,
        r#"time_zone_from_tzif: domain: "Asia/Jerusalem": footer "IST-2IDT,M3.4.4/26,M10.5.0" uses the version 3 transition-time extension in a version 2 file"#,
    ),
    (
        "tzif_name",
        r#"time_zone_from_tzif("New York", zone_file("zones/America/New_York.tzif"))"#,
        r#"time_zone_from_tzif: domain: name "New York" is not an RFC 9557 time zone name"#,
    ),
    (
        "tzif_byte",
        r#"time_zone_from_tzif("Etc/Bad", [84i64, 90i64, 300i64])"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": byte 2 is 300, outside 0..255"#,
    ),
    (
        "tzif_short",
        r#"time_zone_from_tzif("Etc/Bad", [84i64, 90i64, 105i64, 102i64])"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": the file is 4 bytes, too short for a TZif header"#,
    ),
    (
        "tzif_type_index",
        r#"time_zone_from_tzif("Etc/Bad", tzif_file(50i64, [1000000000i64], [2i64], [0i64, 3600i64], "STD-1"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": transition 0 has time type 2, outside 0..1"#,
    ),
    (
        "tzif_footer_offset",
        r#"time_zone_from_tzif("Etc/Bad", tzif_file(50i64, [], [], [0i64], "STD24"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": footer "STD24" has offset -86400 s, outside -86399..86399"#,
    ),
    (
        "tzif_footer_syntax",
        r#"time_zone_from_tzif("Etc/Bad", tzif_file(50i64, [], [], [0i64], "ST0"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": footer "ST0" is not a POSIX TZ string"#,
    ),
    (
        "tzif_footer_long_digits",
        r#"time_zone_from_tzif("Etc/Bad", tzif_file(51i64, [], [], [0i64], "STD0DST,M3.2.0/1:99999999999999999999,M11.1.0"))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": footer "STD0DST,M3.2.0/1:99999999999999999999,M11.1.0" is not a POSIX TZ string"#,
    ),
    (
        "tzif_trailing_byte",
        r#"time_zone_from_tzif("Etc/Bad", concat(zone_file("zones/UTC.tzif"), [0i64]))"#,
        r#"time_zone_from_tzif: domain: "Etc/Bad": the file does not end with the footer's closing newline"#,
    ),
    (
        "offset_at_uncovered",
        "time_zone_offset_at(short_lived(), instant_from_unix(1000000000i64, 0i64))",
        r#"time_zone_offset_at: domain: 2001-09-09T01:46:40Z is at or after unix second 1000000000, the last transition of "Etc/Short_Lived", whose footer is empty"#,
    ),
    (
        "zoned_uncovered",
        "zoned(instant_from_unix(2000000000i64, 5i64), short_lived())",
        r#"zoned: domain: 2033-05-18T03:33:20.000000005Z is at or after unix second 1000000000, the last transition of "Etc/Short_Lived", whose footer is empty"#,
    ),
    (
        "from_local_end",
        "zoned_from_local(local(9999i64, 12i64, 31i64, 23i64, 59i64, 59i64), time_zone_utc(), EarlierInstant)",
        "zoned_from_local: overflow: 9999-12-31T23:59:59 at offset +00:00 is outside the supported instant range",
    ),
    (
        "try_from_local_end",
        "try_zoned_from_local(local(9999i64, 12i64, 31i64, 23i64, 59i64, 59i64), time_zone_utc(), EarlierInstant)",
        "try_zoned_from_local: overflow: 9999-12-31T23:59:59 at offset +00:00 is outside the supported instant range",
    ),
    (
        "from_local_start",
        "zoned_from_local(local(-9999i64, 1i64, 1i64, 0i64, 0i64, 0i64), time_zone_fixed(offset_from_seconds(3600i64)), LaterInstant)",
        "zoned_from_local: overflow: -009999-01-01T00:00:00 at offset +01:00 is outside the supported instant range",
    ),
    (
        "from_local_kiritimati_end",
        "zoned_from_local(local(9999i64, 12i64, 31i64, 23i64, 59i64, 59i64), kiritimati(), EarlierInstant)",
        "zoned_from_local: overflow: 9999-12-31T23:59:59 at offset +14:00 is outside the supported instant range",
    ),
    (
        "from_local_gap",
        "zoned_from_local(local(2026i64, 3i64, 8i64, 2i64, 30i64, 0i64), new_york(), RejectNonUniqueLocal)",
        r#"zoned_from_local: domain: 2026-03-08T02:30:00 does not exist in "America/New_York": the transition at unix second 1772953200 skips it"#,
    ),
    (
        "from_local_fold",
        "zoned_from_local(local(2026i64, 11i64, 1i64, 1i64, 30i64, 0i64), new_york(), RejectNonUniqueLocal)",
        r#"zoned_from_local: domain: 2026-11-01T01:30:00 occurs 2 times in "America/New_York", at offsets -04:00, -05:00"#,
    ),
    (
        "from_local_kiritimati_day",
        "zoned_from_local(local(1994i64, 12i64, 31i64, 12i64, 0i64, 0i64), kiritimati(), RejectNonUniqueLocal)",
        r#"zoned_from_local: domain: 1994-12-31T12:00:00 does not exist in "Pacific/Kiritimati": the transition at unix second 788868000 skips it"#,
    ),
    (
        "from_local_uncovered",
        "zoned_from_local(local(2001i64, 9i64, 8i64, 12i64, 0i64, 0i64), short_lived(), EarlierInstant)",
        r#"zoned_from_local: domain: 2001-09-08T12:00:00 needs offsets at or after unix second 1000000000, the last transition of "Etc/Short_Lived", whose footer is empty"#,
    ),
    (
        "from_local_uncovered_far",
        "zoned_from_local(local(2030i64, 1i64, 1i64, 0i64, 0i64, 0i64), short_lived(), EarlierInstant)",
        r#"zoned_from_local: domain: 2030-01-01T00:00:00 needs offsets at or after unix second 1000000000, the last transition of "Etc/Short_Lived", whose footer is empty"#,
    ),
    (
        "from_local_two_gaps",
        "zoned_from_local(local(2001i64, 9i64, 9i64, 2i64, 46i64, 40i64), two_gaps(), LaterInstant)",
        r#"zoned_from_local: domain: 2001-09-09T02:46:40 lies in 2 gaps of "Etc/Two_Gaps""#,
    ),
    (
        "add_duration_end",
        "zoned_add_duration(zoned(instant_from_unix(253402214400i64, 0i64), time_zone_utc()), duration(1i64, 0i64))",
        "zoned_add_duration: overflow: 9999-12-31T00:00:00Z plus PT1S is outside the supported instant range",
    ),
    (
        "add_duration_max",
        "zoned_add_duration(zoned(instant_from_unix(0i64, 999999999i64), time_zone_utc()), duration(9223372036854775807i64, 999999999i64))",
        "zoned_add_duration: overflow: 1970-01-01T00:00:00.999999999Z plus PT9223372036854775807.999999999S is outside the supported instant range",
    ),
    (
        "add_duration_uncovered",
        "zoned_add_duration(zoned(instant_from_unix(999999999i64, 0i64), short_lived()), duration(1i64, 0i64))",
        r#"zoned_add_duration: domain: 2001-09-09T01:46:40Z is at or after unix second 1000000000, the last transition of "Etc/Short_Lived", whose footer is empty"#,
    ),
    (
        "add_period_months",
        "zoned_add_period(zoned(instant_from_unix(0i64, 0i64), time_zone_utc()), period(i64_minimum(), 0i64), ClampToMonthEnd, EarlierInstant)",
        "zoned_add_period: overflow: 1970-01-01T00:00:00 plus -P9223372036854775808M is outside the supported date range",
    ),
    (
        "add_period_days",
        "zoned_add_period(zoned(instant_from_unix(0i64, 0i64), time_zone_utc()), period(0i64, 9223372036854775807i64), ClampToMonthEnd, EarlierInstant)",
        "zoned_add_period: overflow: 1970-01-01T00:00:00 plus P9223372036854775807D is outside the supported date range",
    ),
    (
        "add_period_invalid_day",
        "zoned_add_period(zoned_from_local(local(2026i64, 1i64, 31i64, 12i64, 0i64, 0i64), new_york(), RejectNonUniqueLocal), period(1i64, 0i64), RejectInvalidDay, EarlierInstant)",
        "zoned_add_period: domain: day 31 is outside 1..28 for 2026-02",
    ),
    (
        "add_period_gap",
        "zoned_add_period(zoned_from_local(local(2026i64, 3i64, 7i64, 2i64, 30i64, 0i64), new_york(), RejectNonUniqueLocal), period(0i64, 1i64), ClampToMonthEnd, RejectNonUniqueLocal)",
        r#"zoned_add_period: domain: 2026-03-08T02:30:00 does not exist in "America/New_York": the transition at unix second 1772953200 skips it"#,
    ),
    (
        "add_period_into_fold_rejected",
        "zoned_add_period(zoned(instant_from_unix(1793424600i64, 0i64), new_york()), period(0i64, 1i64), ClampToMonthEnd, RejectNonUniqueLocal)",
        r#"zoned_add_period: domain: 2026-11-01T01:30:00 occurs 2 times in "America/New_York", at offsets -04:00, -05:00"#,
    ),
    (
        "add_period_at_coverage_end",
        "zoned_add_period(zoned(instant_from_unix(999999999i64, 0i64), short_lived()), period(0i64, 1i64), ClampToMonthEnd, EarlierInstant)",
        r#"zoned_add_period: domain: 2001-09-10T01:46:39 needs offsets at or after unix second 1000000000, the last transition of "Etc/Short_Lived", whose footer is empty"#,
    ),
    (
        "add_period_end",
        "zoned_add_period(zoned_from_local(local(9999i64, 12i64, 30i64, 23i64, 59i64, 59i64), time_zone_utc(), EarlierInstant), period(0i64, 1i64), ClampToMonthEnd, EarlierInstant)",
        "zoned_add_period: overflow: 9999-12-31T23:59:59 at offset +00:00 is outside the supported instant range",
    ),
    (
        "parse_no_annotation",
        r#"parse_zoned_text("2026-11-01T01:30:00-04:00")"#,
        r#"parse_zoned_text: domain: "2026-11-01T01:30:00-04:00" has no time zone annotation"#,
    ),
    (
        "parse_calendar",
        r#"parse_zoned_text("2026-11-01T01:30:00-04:00[America/New_York][u-ca=hebrew]")"#,
        r#"parse_zoned_text: domain: "2026-11-01T01:30:00-04:00[America/New_York][u-ca=hebrew]" names the calendar "hebrew", not iso8601 or gregory"#,
    ),
    (
        "parse_critical_tag",
        r#"parse_zoned_text("2026-11-01T01:30:00-04:00[America/New_York][!x-y=z]")"#,
        r#"parse_zoned_text: domain: "2026-11-01T01:30:00-04:00[America/New_York][!x-y=z]" has the unknown critical annotation "x-y=z""#,
    ),
    (
        "parse_datetime",
        r#"parse_zoned_text("2026-02-30T01:30:00-04:00[America/New_York]")"#,
        r#"parse_zoned_text: domain: "2026-02-30T01:30:00" in "2026-02-30T01:30:00-04:00[America/New_York]" is not a datetime in the text profile"#,
    ),
    (
        "parse_offset",
        r#"parse_zoned_text("2026-11-01T01:30:00-24:00[America/New_York]")"#,
        r#"parse_zoned_text: domain: "-24:00" in "2026-11-01T01:30:00-24:00[America/New_York]" is not an offset in the text profile"#,
    ),
    (
        "parse_zone_name",
        r#"parse_zoned_text("2026-11-01T01:30:00-04:00[America/../York]")"#,
        r#"parse_zoned_text: domain: "2026-11-01T01:30:00-04:00[America/../York]" has an invalid time zone annotation "America/../York""#,
    ),
    (
        "parse_second_zone",
        r#"parse_zoned_text("2026-11-01T01:30:00-04:00[America/New_York][UTC]")"#,
        r#"parse_zoned_text: domain: "2026-11-01T01:30:00-04:00[America/New_York][UTC]" has a second time zone annotation"#,
    ),
    (
        "parse_trailing",
        r#"parse_zoned_text("2026-11-01T01:30:00-04:00[America/New_York]x")"#,
        r#"parse_zoned_text: domain: "2026-11-01T01:30:00-04:00[America/New_York]x" has text after its annotations"#,
    ),
    (
        "parse_critical_repeat_of_unknown_key",
        r#"parse_zoned_text("2026-01-01T00:00:00Z[UTC][foo=a][!foo=a]")"#,
        r#"parse_zoned_text: domain: "2026-01-01T00:00:00Z[UTC][foo=a][!foo=a]" has the unknown critical annotation "foo=a""#,
    ),
    (
        "parse_critical_duplicate_key",
        r#"parse_zoned_text("2022-07-08T00:14:07Z[UTC][!u-ca=iso8601][u-ca=gregory]")"#,
        r#"parse_zoned_text: domain: "2022-07-08T00:14:07Z[UTC][!u-ca=iso8601][u-ca=gregory]" gives the critical key "u-ca" more than one value"#,
    ),
    (
        "parse_critical_duplicate_key_second",
        r#"parse_zoned_text("2022-07-08T00:14:07Z[UTC][u-ca=iso8601][!u-ca=gregory]")"#,
        r#"parse_zoned_text: domain: "2022-07-08T00:14:07Z[UTC][u-ca=iso8601][!u-ca=gregory]" gives the critical key "u-ca" more than one value"#,
    ),
    (
        "parse_critical_after_two_values",
        r#"parse_zoned_text("2022-07-08T00:14:07Z[UTC][u-ca=iso8601][u-ca=gregory][!u-ca=iso8601]")"#,
        r#"parse_zoned_text: domain: "2022-07-08T00:14:07Z[UTC][u-ca=iso8601][u-ca=gregory][!u-ca=iso8601]" gives the critical key "u-ca" more than one value"#,
    ),
    (
        "parse_critical_between_two_values",
        r#"parse_zoned_text("2022-07-08T00:14:07Z[UTC][u-ca=iso8601][!u-ca=iso8601][u-ca=gregory]")"#,
        r#"parse_zoned_text: domain: "2022-07-08T00:14:07Z[UTC][u-ca=iso8601][!u-ca=iso8601][u-ca=gregory]" gives the critical key "u-ca" more than one value"#,
    ),
    (
        "parse_unclosed",
        r#"parse_zoned_text("2026-11-01T01:30:00-04:00[America/New_York")"#,
        r#"parse_zoned_text: domain: "2026-11-01T01:30:00-04:00[America/New_York" has an unclosed annotation"#,
    ),
    (
        "from_text_written_end",
        r#"zoned_from_text(parse_zoned_text("9999-12-31T23:59:59-05:00[UTC]"), time_zone_utc(), UseWrittenOffset)"#,
        "zoned_from_text: overflow: 9999-12-31T23:59:59-05:00 is outside the supported instant range",
    ),
    (
        "try_from_text_written_end",
        r#"try_zoned_from_text(parse_zoned_text("9999-12-31T23:59:59-05:00[UTC]"), time_zone_utc(), UseWrittenOffset)"#,
        "try_zoned_from_text: overflow: 9999-12-31T23:59:59-05:00 is outside the supported instant range",
    ),
    (
        "from_text_critical",
        r#"zoned_from_text(parse_zoned_text("2026-07-01T12:00:00-05:00[!America/New_York]"), new_york(), UseWrittenOffset)"#,
        r#"zoned_from_text: domain: 2026-07-01T12:00:00-05:00 has a critical zone annotation, but "America/New_York" has offset -04:00 there"#,
    ),
    (
        "from_text_plus_zero_critical",
        r#"zoned_from_text(parse_zoned_text("2022-07-08T00:14:07+00:00[!Europe/London]"), london(), UseWrittenOffset)"#,
        r#"zoned_from_text: domain: 2022-07-08T00:14:07+00:00 has a critical zone annotation, but "Europe/London" has offset +01:00 there"#,
    ),
    (
        "from_text_unknown_offset_end",
        r#"zoned_from_text(parse_zoned_text("9999-12-31T23:59:59Z[UTC]"), time_zone_utc(), UseZoneRules)"#,
        "zoned_from_text: overflow: 9999-12-31T23:59:59Z is outside the supported instant range",
    ),
    (
        "try_from_text_unknown_offset_end",
        r#"try_zoned_from_text(parse_zoned_text("9999-12-31T23:59:59-00:00[UTC]"), time_zone_utc(), RejectOffsetMismatch)"#,
        "try_zoned_from_text: overflow: 9999-12-31T23:59:59Z is outside the supported instant range",
    ),
    (
        "from_text_unknown_offset_uncovered",
        r#"zoned_from_text(parse_zoned_text("2030-01-01T00:00:00z[!Etc/Short_Lived]"), short_lived(), RejectOffsetMismatch)"#,
        r#"zoned_from_text: domain: 2030-01-01T00:00:00Z is at or after unix second 1000000000, the last transition of "Etc/Short_Lived", whose footer is empty"#,
    ),
    (
        "from_text_zone_rules_fold",
        r#"zoned_from_text(parse_zoned_text("2026-11-01T01:30:00-05:00[America/New_York]"), new_york(), UseZoneRules)"#,
        r#"zoned_from_text: domain: 2026-11-01T01:30:00 occurs 2 times in "America/New_York", at offsets -04:00, -05:00"#,
    ),
    (
        "from_text_mismatch",
        r#"zoned_from_text(parse_zoned_text("2026-07-01T12:00:00-05:00[America/New_York]"), new_york(), RejectOffsetMismatch)"#,
        r#"zoned_from_text: domain: 2026-07-01T12:00:00-05:00 does not match "America/New_York", which has offset -04:00 there"#,
    ),
    (
        "from_text_uncovered",
        r#"zoned_from_text(parse_zoned_text("2030-01-01T00:00:00+01:00[Etc/Short_Lived]"), short_lived(), UseWrittenOffset)"#,
        r#"zoned_from_text: domain: 2029-12-31T23:00:00Z is at or after unix second 1000000000, the last transition of "Etc/Short_Lived", whose footer is empty"#,
    ),
];

/// (test name, `assert_eq` call that must pass). RFC 9557 Figure 2 writes
/// `Z`, a UTC time with an unknown local offset, so every policy gives its
/// UTC instant, 00:14:07Z, whose London reading is 01:14:07+01:00. Figure 1
/// writes `+00:00`, a known offset that London does not use there; it differs
/// from Figure 2 under `UseZoneRules` and fails under a critical annotation
/// (`from_text_plus_zero_critical` above).
const EXACT_RESULTS: &[(&str, &str)] = &[
    (
        "figure_2_elective_written",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07Z[Europe/London]"), london(), UseWrittenOffset)), "2022-07-08T01:14:07+01:00[Europe/London]", "figure 2")"#,
    ),
    (
        "figure_2_elective_zone_rules",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07Z[Europe/London]"), london(), UseZoneRules)), "2022-07-08T01:14:07+01:00[Europe/London]", "figure 2")"#,
    ),
    (
        "figure_2_elective_reject_mismatch",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07Z[Europe/London]"), london(), RejectOffsetMismatch)), "2022-07-08T01:14:07+01:00[Europe/London]", "figure 2")"#,
    ),
    (
        "figure_2_critical_written",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07Z[!Europe/London]"), london(), UseWrittenOffset)), "2022-07-08T01:14:07+01:00[Europe/London]", "figure 2")"#,
    ),
    (
        "figure_2_critical_zone_rules",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07Z[!Europe/London]"), london(), UseZoneRules)), "2022-07-08T01:14:07+01:00[Europe/London]", "figure 2")"#,
    ),
    (
        "figure_2_critical_reject_mismatch",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07Z[!Europe/London]"), london(), RejectOffsetMismatch)), "2022-07-08T01:14:07+01:00[Europe/London]", "figure 2")"#,
    ),
    (
        "lower_z_written",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07z[!Europe/London]"), london(), UseWrittenOffset)), "2022-07-08T01:14:07+01:00[Europe/London]", "z")"#,
    ),
    (
        "lower_z_zone_rules",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07z[Europe/London]"), london(), UseZoneRules)), "2022-07-08T01:14:07+01:00[Europe/London]", "z")"#,
    ),
    (
        "lower_z_reject_mismatch",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07z[Europe/London]"), london(), RejectOffsetMismatch)), "2022-07-08T01:14:07+01:00[Europe/London]", "z")"#,
    ),
    (
        "minus_zero_written",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07-00:00[!Europe/London]"), london(), UseWrittenOffset)), "2022-07-08T01:14:07+01:00[Europe/London]", "-00:00")"#,
    ),
    (
        "minus_zero_zone_rules",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07-00:00[Europe/London]"), london(), UseZoneRules)), "2022-07-08T01:14:07+01:00[Europe/London]", "-00:00")"#,
    ),
    (
        "minus_zero_reject_mismatch",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07-00:00[Europe/London]"), london(), RejectOffsetMismatch)), "2022-07-08T01:14:07+01:00[Europe/London]", "-00:00")"#,
    ),
    (
        "unknown_offset_is_absent",
        r#"assert_eq(parse_zoned_text("2022-07-08T00:14:07Z[Europe/London]").offset, None, "Z")"#,
    ),
    (
        "minus_zero_is_absent",
        r#"assert_eq(parse_zoned_text("2022-07-08T00:14:07-00:00[Europe/London]").offset, None, "-00:00")"#,
    ),
    (
        "plus_zero_is_present",
        r#"assert_eq(parse_zoned_text("2022-07-08T00:14:07+00:00[Europe/London]").offset, Some(offset_from_seconds(0i64)), "+00:00")"#,
    ),
    (
        "figure_1_elective_written",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07+00:00[Europe/London]"), london(), UseWrittenOffset)), "2022-07-08T01:14:07+01:00[Europe/London]", "figure 1 keeps the written instant")"#,
    ),
    (
        "figure_1_elective_zone_rules",
        r#"assert_eq(zoned_to_string(zoned_from_text(parse_zoned_text("2022-07-08T00:14:07+00:00[Europe/London]"), london(), UseZoneRules)), "2022-07-08T00:14:07+01:00[Europe/London]", "figure 1 reads 00:14:07 on London's clock")"#,
    ),
    (
        "elective_duplicate_keys",
        r#"assert_eq(parse_zoned_text("2022-07-08T00:14:07Z[UTC][u-ca=iso8601][u-ca=gregory]").zone_name, "UTC", "elective duplicates")"#,
    ),
    (
        "elective_duplicate_first_decides",
        r#"assert_eq(parse_zoned_text("2022-07-08T00:14:07Z[UTC][u-ca=iso8601][u-ca=hebrew]").zone_name, "UTC", "the first tag decides")"#,
    ),
    (
        "elective_repeat_of_unknown_key",
        r#"assert_eq(parse_zoned_text("2026-01-01T00:00:00Z[UTC][foo=a][foo=a]").zone_name, "UTC", "an elective repeat is ignored")"#,
    ),
    (
        "zero_period_in_fold_earlierinstant",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(1793514600i64, 0i64), new_york()), period(0i64, 0i64), ClampToMonthEnd, EarlierInstant), zoned(instant_from_unix(1793514600i64, 0i64), new_york()), "P0D keeps the later 01:30")"#,
    ),
    (
        "zero_period_in_fold_laterinstant",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(1793514600i64, 0i64), new_york()), period(0i64, 0i64), ClampToMonthEnd, LaterInstant), zoned(instant_from_unix(1793514600i64, 0i64), new_york()), "P0D keeps the later 01:30")"#,
    ),
    (
        "zero_period_in_fold_compatibleinstant",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(1793514600i64, 0i64), new_york()), period(0i64, 0i64), ClampToMonthEnd, CompatibleInstant), zoned(instant_from_unix(1793514600i64, 0i64), new_york()), "P0D keeps the later 01:30")"#,
    ),
    (
        "zero_period_in_fold_rejectnonuniquelocal",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(1793514600i64, 0i64), new_york()), period(0i64, 0i64), ClampToMonthEnd, RejectNonUniqueLocal), zoned(instant_from_unix(1793514600i64, 0i64), new_york()), "P0D keeps the later 01:30")"#,
    ),
    (
        "zero_period_reject_invalid_day",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(1793514600i64, 0i64), new_york()), period(0i64, 0i64), RejectInvalidDay, RejectNonUniqueLocal), zoned(instant_from_unix(1793514600i64, 0i64), new_york()), "P0D under RejectInvalidDay")"#,
    ),
    (
        "zero_period_at_range_end",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(253402214400i64, 999999999i64), kiritimati()), period(0i64, 0i64), ClampToMonthEnd, RejectNonUniqueLocal), zoned(instant_from_unix(253402214400i64, 999999999i64), kiritimati()), "P0D at the last instant")"#,
    ),
    (
        "zero_period_at_range_start",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(-377705030401i64, 0i64), new_york()), period(0i64, 0i64), ClampToMonthEnd, RejectNonUniqueLocal), zoned(instant_from_unix(-377705030401i64, 0i64), new_york()), "P0D at the first instant")"#,
    ),
    (
        "zero_period_at_coverage_end",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(999999999i64, 0i64), short_lived()), period(0i64, 0i64), ClampToMonthEnd, EarlierInstant), zoned(instant_from_unix(999999999i64, 0i64), short_lived()), "P0D just before an empty footer")"#,
    ),
    (
        "one_day_into_fold_earlierinstant",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(1793424600i64, 0i64), new_york()), period(0i64, 1i64), ClampToMonthEnd, EarlierInstant), zoned(instant_from_unix(1793511000i64, 0i64), new_york()), "P1D re-resolves in the fold")"#,
    ),
    (
        "one_day_into_fold_laterinstant",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(1793424600i64, 0i64), new_york()), period(0i64, 1i64), ClampToMonthEnd, LaterInstant), zoned(instant_from_unix(1793514600i64, 0i64), new_york()), "P1D re-resolves in the fold")"#,
    ),
    (
        "one_day_into_fold_compatibleinstant",
        r#"assert_eq(zoned_add_period(zoned(instant_from_unix(1793424600i64, 0i64), new_york()), period(0i64, 1i64), ClampToMonthEnd, CompatibleInstant), zoned(instant_from_unix(1793511000i64, 0i64), new_york()), "P1D re-resolves in the fold")"#,
    ),
    (
        "one_day_into_gap_earlierinstant",
        r#"assert_eq(zoned_to_string(zoned_add_period(zoned(instant_from_unix(1772868600i64, 0i64), new_york()), period(0i64, 1i64), ClampToMonthEnd, EarlierInstant)), "2026-03-08T01:30:00-05:00[America/New_York]", "P1D into the gap")"#,
    ),
    (
        "one_day_into_gap_laterinstant",
        r#"assert_eq(zoned_to_string(zoned_add_period(zoned(instant_from_unix(1772868600i64, 0i64), new_york()), period(0i64, 1i64), ClampToMonthEnd, LaterInstant)), "2026-03-08T03:30:00-04:00[America/New_York]", "P1D into the gap")"#,
    ),
    (
        "one_day_into_gap_compatibleinstant",
        r#"assert_eq(zoned_to_string(zoned_add_period(zoned(instant_from_unix(1772868600i64, 0i64), new_york()), period(0i64, 1i64), ClampToMonthEnd, CompatibleInstant)), "2026-03-08T03:30:00-04:00[America/New_York]", "P1D into the gap")"#,
    ),
    (
        "elective_after_two_values",
        r#"assert_eq(parse_zoned_text("2022-07-08T00:14:07Z[UTC][u-ca=iso8601][u-ca=gregory][u-ca=iso8601]").zone_name, "UTC", "elective tags only")"#,
    ),
    (
        "critical_between_one_value",
        r#"assert_eq(parse_zoned_text("2022-07-08T00:14:07Z[UTC][u-ca=iso8601][!u-ca=iso8601][u-ca=iso8601]").zone_name, "UTC", "one value")"#,
    ),
    (
        "critical_duplicate_same_value",
        r#"assert_eq(parse_zoned_text("2022-07-08T00:14:07Z[UTC][!u-ca=iso8601][!u-ca=iso8601]").zone_name, "UTC", "one value")"#,
    ),
];

const MAX: &str = "9223372036854775807i64";
const MIN: &str = "i64_minimum()";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tzif")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is two levels below the repo root")
        .to_path_buf()
}

fn run_chelis(app_pkg: &Path, reef_home: &Path, args: &[&str]) -> (bool, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args(args)
        .output()
        .expect("chelis runs");
    (
        output.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

/// Expressions per `chelis test` run. Every run has its own suite timeout,
/// and each case reads its zone's TZif bytes again, so one run over a whole
/// sweep can outgrow that timeout on a slow runner.
const CASES_PER_SUITE: usize = 40;

/// Seconds a quiet machine takes, with the debug build, to compile one
/// generated suite file before its first test runs: the prelude, the imports
/// and the bundled std. A one-case run under a fresh reef home measures 8 s
/// (3 s once the reef cache is warm).
const LOCAL_COMPILE_SECS: f64 = 8.0;

/// Every `chelis test` limit below is this many times the run's local
/// quiet-machine time. CI has run these suites up to 5.7 times slower than
/// a quiet local machine (the extreme sweep: 791 s there, 139 s here), so 10
/// keeps the limits clear of CI with room to spare.
const LIMIT_MARGIN: f64 = 10.0;

/// The `chelis test` limits for one run of `cases` cases that each take
/// `case_secs` locally: `(--timeout, --suite-timeout)`, in seconds.
///
/// The run takes `LOCAL_COMPILE_SECS + cases × case_secs` locally, and the
/// suite timeout is `LIMIT_MARGIN` times that. Each test may use the whole
/// suite budget, which is at least `LIMIT_MARGIN` times any one case's local
/// time. `chelis test` kills a file worker after `timeout × (cases + 1) + 10`
/// seconds, which therefore exceeds the suite timeout as well.
fn run_limits(cases: usize, case_secs: f64) -> (u64, u64) {
    let local = LOCAL_COMPILE_SECS + case_secs * cases as f64;
    let suite = (LIMIT_MARGIN * local).ceil() as u64;
    (suite, suite)
}

/// Runs `chelis test` over fixtures with one test per expression, at most
/// [`CASES_PER_SUITE`] to a run, under [`run_limits`] for cases that take
/// `case_secs` each locally, and returns each test's outcome: `None` for PASS,
/// `Some(message)` for FAIL.
fn run_expression_suite(
    dir_name: &str,
    expressions: &[(String, String)],
    case_secs: f64,
) -> BTreeMap<String, Option<String>> {
    let (_dir, reef_home, app_pkg) = make_app(dir_name);
    let prelude = PRELUDE.replace("@TZIF@", &fixture_dir().to_string_lossy());
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\nanchor = 0i64\n",
    );
    let mut outcomes = BTreeMap::new();
    for (index, chunk) in expressions.chunks(CASES_PER_SUITE).enumerate() {
        let mut source = format!("module Demo.Tests.Zone\n{IMPORTS}\n{prelude}");
        for (name, expression) in chunk {
            source.push_str(&format!(
                "def test_{name}() -> unit ! {{ Test, IO }} = {{\n  _value = {expression}\n  ()\n}}\n"
            ));
        }
        // One fixture file at a time, so no run sees another's module.
        let path = app_pkg.join(format!("tests/zone_cases_{index:03}.ch"));
        write_file(&path, &source);
        let (timeout, suite_timeout) = run_limits(chunk.len(), case_secs);
        let started = std::time::Instant::now();
        let (_, rendered) = run_chelis(
            &app_pkg,
            &reef_home,
            &[
                "test",
                "--batch-mode",
                "file",
                "--timeout",
                &timeout.to_string(),
                "--suite-timeout",
                &suite_timeout.to_string(),
                path.to_str().unwrap(),
            ],
        );
        eprintln!(
            "{dir_name} run {index}: {} cases in {:.1} s (limits: {timeout} s per test, {suite_timeout} s per suite)",
            chunk.len(),
            started.elapsed().as_secs_f64()
        );
        std::fs::remove_file(&path).expect("remove the finished fixture");
        let mut reported = 0usize;
        for line in rendered.lines() {
            let Some(rest) = line.trim_start().strip_prefix("test_") else {
                continue;
            };
            let Some((name, verdict)) = rest.split_once(' ') else {
                continue;
            };
            let verdict = verdict.trim_start_matches(['.', ' ']);
            if verdict == "PASS" {
                outcomes.insert(name.to_string(), None);
                reported += 1;
            } else if let Some(message) = verdict
                .strip_prefix("FAIL (")
                .and_then(|tail| tail.strip_suffix(')'))
            {
                outcomes.insert(name.to_string(), Some(message.to_string()));
                reported += 1;
            }
        }
        assert_eq!(
            reported,
            chunk.len(),
            "every generated test must report exactly one verdict:\n{rendered}"
        );
    }
    assert_eq!(
        outcomes.len(),
        expressions.len(),
        "expression names must be distinct"
    );
    outcomes
}

#[test]
fn std_datetime_zone_failures_report_their_exact_message() {
    let expressions: Vec<(String, String)> = EXACT_FAILURES
        .iter()
        .map(|(name, expression, _)| (name.to_string(), expression.to_string()))
        .collect();
    // Locally the slowest run, 40 cases under a fresh reef home, takes 17.2 s:
    // (17.2 - 8) / 40 = 0.23 s a case. The limit for 40 cases is
    // 10 × (8 + 40 × 0.3) = 200 s.
    let outcomes = run_expression_suite("datetime-zone-failures-2862", &expressions, 0.3);
    for (name, expression, expected) in EXACT_FAILURES {
        assert_eq!(
            outcomes.get(*name),
            Some(&Some(expected.to_string())),
            "{name}: `{expression}` must fail with exactly `{expected}`"
        );
    }
}

#[test]
fn std_datetime_zone_text_results_are_exact() {
    let expressions: Vec<(String, String)> = EXACT_RESULTS
        .iter()
        .map(|(name, expression)| (name.to_string(), expression.to_string()))
        .collect();
    // Locally the one run of 23 cases took 24.9 s: (24.9 - 8) / 23 = 0.73 s a
    // case. With 37 cases the limit is 10 × (8 + 37 × 0.8) = 376 s.
    let outcomes = run_expression_suite("datetime-zone-results-2862", &expressions, 0.8);
    for (name, expression) in EXACT_RESULTS {
        assert_eq!(
            outcomes.get(*name),
            Some(&None),
            "{name}: `{expression}` must pass"
        );
    }
}

/// A suffix tag: (key, value, critical).
type Tag = (&'static str, &'static str, bool);

/// RFC 9557 §3.3 for the module's one recognized key, `u-ca`, written
/// independently of the parser: every critical tag names a recognized key; a
/// key that two tags give different values is erroneous when either is
/// critical; otherwise the first tag with a key decides, and the first `u-ca`
/// must name a calendar the module supports.
fn suffix_tags_accepted(tags: &[Tag]) -> bool {
    if tags
        .iter()
        .any(|(key, _, critical)| *critical && *key != "u-ca")
    {
        return false;
    }
    for (index, (key, value, critical)) in tags.iter().enumerate() {
        for (other_key, other_value, other_critical) in &tags[index + 1..] {
            if key == other_key && value != other_value && (*critical || *other_critical) {
                return false;
            }
        }
    }
    match tags.iter().find(|(key, _, _)| *key == "u-ca") {
        Some((_, value, _)) => matches!(*value, "iso8601" | "gregory"),
        None => true,
    }
}

/// Suffix tag sequences checked per `chelis test` case. Each case checks a
/// batch in one fold and reports the texts it judged wrongly.
const SUFFIX_TAGS_PER_CASE: usize = 30;

/// Every sequence of one to three suffix tags drawn from `foo=a`,
/// `u-ca=iso8601` and `u-ca=hebrew`, each elective or critical, after a zone
/// annotation, against [`suffix_tags_accepted`]: 6 + 36 + 216 = 258 texts.
///
/// Why this set and these lengths suffice: RFC 9557 §3.3 judges each key by
/// its tags as a group, through three facts: whether a critical tag names a
/// key the module does not recognize, whether the key's tags give more than
/// one value while one of them is critical, and the key's first value. `foo`
/// is the unrecognized key, and one value of it is enough, because any
/// critical `foo` tag already fails. `u-ca` with a supported and an
/// unsupported calendar gives it two values and both first-value outcomes.
/// A comparison that is not over the whole group (with the first tag only,
/// or with the previous tag only) disagrees with the rule first at three tags
/// of one key, as in `[u-ca=iso8601][u-ca=hebrew][!u-ca=iso8601]`, and every
/// combination of the three facts, with the other key's tag before, between
/// or after, occurs within three tags.
#[test]
fn std_datetime_zone_suffix_tags_follow_rfc_9557() {
    let tags: Vec<Tag> = vec![
        ("foo", "a", false),
        ("foo", "a", true),
        ("u-ca", "iso8601", false),
        ("u-ca", "iso8601", true),
        ("u-ca", "hebrew", false),
        ("u-ca", "hebrew", true),
    ];
    let mut sequences: Vec<Vec<Tag>> = vec![Vec::new()];
    let mut frontier: Vec<Vec<Tag>> = vec![Vec::new()];
    for _ in 0..3 {
        frontier = frontier
            .iter()
            .flat_map(|prefix| {
                tags.iter().map(move |tag| {
                    let mut longer = prefix.clone();
                    longer.push(*tag);
                    longer
                })
            })
            .collect();
        sequences.extend(frontier.iter().cloned());
    }
    sequences.remove(0);
    assert_eq!(sequences.len(), 6 + 36 + 216);
    let cases: Vec<(String, bool)> = sequences
        .iter()
        .map(|sequence| {
            let suffix: String = sequence
                .iter()
                .map(|(key, value, critical)| {
                    format!("[{}{key}={value}]", if *critical { "!" } else { "" })
                })
                .collect();
            (
                format!("2022-07-08T00:14:07Z[UTC]{suffix}"),
                suffix_tags_accepted(sequence),
            )
        })
        .collect();
    assert!(cases.iter().any(|case| case.1) && cases.iter().any(|case| !case.1));
    let expressions: Vec<(String, String)> = cases
        .chunks(SUFFIX_TAGS_PER_CASE)
        .enumerate()
        .map(|(index, chunk)| {
            let list: Vec<String> = chunk
                .iter()
                .map(|(text, expected)| format!("(\"{text}\", {expected})"))
                .collect();
            (
                format!("tags_{index:03}"),
                format!(
                    "assert_eq(fold(fn (acc: string, case: (string, bool)) -> if eq(accepted(case.0), case.1) then acc else acc |> string_concat(\" \") |> string_concat(case.0), \"\", [{}]), \"\", \"texts judged against RFC 9557\")",
                    list.join(", ")
                ),
            )
        })
        .collect();
    // Locally the one run, 9 cases of up to 30 texts, takes 24.2 s:
    // (24.2 - 8) / 9 = 1.8 s a case. The limit is 10 × (8 + 9 × 1.9) = 251 s.
    let outcomes = run_expression_suite("datetime-zone-suffix-tags-2862", &expressions, 1.9);
    let wrong: Vec<String> = expressions
        .iter()
        .filter_map(|(name, _)| match outcomes.get(name) {
            Some(None) => None,
            Some(Some(message)) => Some(format!("{name}: {message}")),
            None => Some(format!("{name}: no verdict")),
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "suffix tags against RFC 9557 §3.3:\n{}",
        wrong.join("\n")
    );
}

/// Every call below passes, or fails `domain` or `overflow` under its own
/// callable's name; a primitive trap would surface under another message.
fn extreme_cases() -> Vec<(&'static str, String)> {
    let zones = [
        "new_york()",
        "kiritimati()",
        "short_lived()",
        "time_zone_utc()",
        "time_zone_fixed(offset_from_seconds(86399i64))",
        "time_zone_fixed(offset_from_seconds(-86399i64))",
    ];
    let instants = [
        "instant_from_unix(-377705030401i64, 0i64)",
        "instant_from_unix(253402214400i64, 999999999i64)",
        "instant_from_unix(0i64, 0i64)",
    ];
    let locals = [
        "datetime(date(-9999i64, 1i64, 1i64), time(0i64, 0i64, 0i64, 0i64))".to_string(),
        "datetime(date(9999i64, 12i64, 31i64), time(23i64, 59i64, 59i64, 999999999i64))"
            .to_string(),
    ];
    let policies = [
        "EarlierInstant",
        "LaterInstant",
        "CompatibleInstant",
        "RejectNonUniqueLocal",
    ];
    let conflicts = ["UseWrittenOffset", "UseZoneRules", "RejectOffsetMismatch"];
    let durations = [
        format!("duration({MAX}, 999999999i64)"),
        format!("duration({MIN}, 0i64)"),
    ];
    let periods = [
        format!("period({MAX}, 0i64)"),
        format!("period({MIN}, 0i64)"),
        format!("period(0i64, {MAX})"),
        format!("period(0i64, {MIN})"),
    ];
    let mut cases: Vec<(&'static str, String)> = Vec::new();
    for zone in zones {
        for instant in instants {
            cases.push((
                "time_zone_offset_at",
                format!("time_zone_offset_at({zone}, {instant})"),
            ));
            cases.push(("zoned", format!("zoned({instant}, {zone})")));
        }
        for local in &locals {
            for policy in policies {
                cases.push((
                    "zoned_from_local",
                    format!("zoned_from_local({local}, {zone}, {policy})"),
                ));
            }
            for offset in [
                "Some(offset_from_seconds(86399i64))",
                "Some(offset_from_seconds(-86399i64))",
                "None",
            ] {
                for conflict in conflicts {
                    cases.push((
                        "zoned_from_text",
                        format!("zoned_from_text(ZonedText {{ written: {local}, offset: {offset}, zone_name: \"X\", critical: true }}, {zone}, {conflict})"),
                    ));
                }
            }
        }
        let start = format!("zoned(instant_from_unix(0i64, 0i64), {zone})");
        for d in &durations {
            cases.push((
                "zoned_add_duration",
                format!("zoned_add_duration({start}, {d})"),
            ));
        }
        for p in &periods {
            for policy in policies {
                cases.push((
                    "zoned_add_period",
                    format!("zoned_add_period({start}, {p}, ClampToMonthEnd, {policy})"),
                ));
            }
        }
    }
    for bytes in [
        format!("[{MAX}]"),
        format!("[{MIN}]"),
        "map(fn (k: i64) -> 255i64, range(0i64, 200i64))".to_string(),
        "concat([84i64, 90i64, 105i64, 102i64, 50i64], map(fn (k: i64) -> 255i64, range(0i64, 120i64)))".to_string(),
        "concat(header_bytes(50i64, [0i64, 0i64, 0i64, 0i64, 1i64, 1i64]), concat([0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64], header_bytes(50i64, [4294967295i64, 4294967295i64, 0i64, 4294967295i64, 4294967295i64, 4294967295i64])))".to_string(),
        "tzif_file(51i64, [], [], [0i64], \"STD0DST,M3.2.0/167,M11.1.0/-167\")".to_string(),
        "tzif_file(51i64, [], [], [0i64], \"<+999>-99:59:59<-999>99:59:59,J365/-167:59:59,0/167:59:59\")".to_string(),
        format!("tzif_file(50i64, [{MIN}, {MAX}], [0i64, 0i64], [0i64], \"STD0DST,M3.2.0,M11.1.0\")"),
        format!("tzif_file(50i64, [{MAX}], [0i64], [86399i64], \"STD-23:59:59\")"),
    ] {
        cases.push((
            "time_zone_from_tzif",
            format!("time_zone_from_tzif(\"Etc/Extreme\", {bytes})"),
        ));
    }
    for text in [
        "",
        "[",
        "]]",
        "9999-12-31T23:59:59-23:59[X]",
        "-009999-01-01T00:00:00+23:59:59[X]",
        "+999999-01-01T00:00:00Z[X]",
        "2026-01-01T00:00:00Z[!]",
        "2026-01-01T00:00:00Z[X][=]",
        "2026-01-01T00:00:00Z[X][u-ca=]",
        "2026-01-01T00:00:00Z[+99:99]",
    ] {
        cases.push(("parse_zoned_text", format!("parse_zoned_text(\"{text}\")")));
    }
    cases
}

#[test]
fn std_datetime_zone_extreme_arguments_raise_no_primitive_trap() {
    let cases = extreme_cases();
    let expressions: Vec<(String, String)> = cases
        .iter()
        .enumerate()
        .map(|(index, (_, expression))| (format!("case_{index:04}"), expression.clone()))
        .collect();
    // Locally the slowest run, the first 40 cases (New York and Kiritimati,
    // under a fresh reef home), takes 65.1 s: (65.1 - 8) / 40 = 1.43 s a case.
    // The limit for 40 cases is 10 × (8 + 40 × 1.5) = 680 s.
    let outcomes = run_expression_suite("datetime-zone-extremes-2862", &expressions, 1.5);
    let mut failures = 0usize;
    for (index, (function, expression)) in cases.iter().enumerate() {
        let outcome = &outcomes[&format!("case_{index:04}")];
        if let Some(message) = outcome {
            failures += 1;
            assert!(
                message.starts_with(&format!("{function}: domain: "))
                    || message.starts_with(&format!("{function}: overflow: ")),
                "`{expression}` escaped the failure contract with `{message}`"
            );
        }
    }
    assert!(
        failures > 0 && failures < cases.len(),
        "the sweep must exercise both accepted and rejected extremes ({failures} of {})",
        cases.len()
    );
}

/// [05-OP-73]: `TimeZone` and `Zoned` are opaque, so constructing or
/// inspecting them outside `Std.Datetime.Zone` is an `OpaqueTypeViolation`;
/// `ZonedText` is a plain record that any module constructs and reads.
#[test]
fn std_datetime_zone_opaque_types_reject_outside_construction() {
    let (_dir, reef_home, app_pkg) = make_app("datetime-zone-opaque-2862");
    let header = "module Demo.Main\nimport Std.Datetime (datetime, date, time, instant_from_unix, offset_from_seconds)\nimport Std.Datetime.Zone (TimeZone, Zoned, ZonedText, time_zone_utc, parse_zoned_text)\n";
    let path = app_pkg.join("src/main.ch");
    write_file(
        &path,
        &format!(
            "{header}plain = ZonedText {{ written: datetime(date(2026i64, 1i64, 1i64), time(0i64, 0i64, 0i64, 0i64)), offset: Some(offset_from_seconds(0i64)), zone_name: \"UTC\", critical: false }}\nname = parse_zoned_text(\"2026-01-01T00:00:00Z[UTC]\").zone_name\n"
        ),
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        success,
        "the plain ZonedText record must check:\n{rendered}"
    );

    write_file(
        &path,
        &format!(
            "{header}forged_zone = TimeZone {{ name: \"UTC\", initial_offset: 0i64, transitions: [], footer: None }}\nforged_zoned = Zoned {{ instant: instant_from_unix(0i64, 0i64), zone: time_zone_utc() }}\npeeked = time_zone_utc().name\n"
        ),
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        !success,
        "forging opaque zone values must not check:\n{rendered}"
    );
    for type_name in ["TimeZone", "Zoned"] {
        assert!(
            rendered.contains(&format!(
                "record construction of opaque type `{type_name}` outside its defining module"
            )),
            "`{type_name}` construction was not rejected as opaque:\n{rendered}"
        );
    }
    assert!(
        rendered.contains("field access") && rendered.contains("opaque type `TimeZone`"),
        "field access on an opaque TimeZone was not rejected:\n{rendered}"
    );
}

fn toolchain_json() -> String {
    let requirements = CodegenRequirements::default();
    let toolchain = strict_reference_toolchain(test_toolchain(requirements).compiler, requirements);
    serde_json::json!({
        "compiler": toolchain.compiler,
        "compile_flags": toolchain.compile_flags,
        "link_flags": toolchain.link_flags,
    })
    .to_string()
}

/// Runs the differential on `lanes` (`eval` or `c`): every fixture zone's
/// offsets, every gap and fold under each `Disambiguation`, and the RFC 9557
/// text, against Python's `zoneinfo` reading the same file. Each lane is its
/// own test so that neither comes near nextest's per-test kill on CI
/// (`ci-full`: 3 000 s). Together they took 1 948 s there before the
/// `utc_texts` row, which made them about half as slow again. On a quiet
/// local machine the eval lane takes 263 s and the C lane 189 s; at CI's
/// observed 5.7 times that is about 1 500 s and 1 080 s. The two full lanes
/// therefore run nightly, and the `canary` profile runs both lanes on every
/// pull request: a zone with both a gap and a fold, and `UTC`.
fn run_zone_differential(profile: &str, lanes: &str, zones: &str) {
    let python =
        managed_python::managed_python(&repo_root()).unwrap_or_else(|error| panic!("{error}"));
    let chelis = assert_cmd::cargo_bin!("chelis").to_path_buf();
    let mut command = std::process::Command::new(python);
    command
        .arg(repo_root().join("scripts/datetime_zone_differential.py"))
        .arg("--chelis")
        .arg(&chelis)
        .arg("--reef-home")
        .arg(&common::SHARED_REEF.reef_home)
        .args(["--profile", profile, "--lanes", lanes]);
    if lanes.contains('c') {
        command.args(["--toolchain-json", &toolchain_json()]);
    }
    let started = std::time::Instant::now();
    let output = command
        .output()
        .expect("spawn the zone differential harness");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!(
        "zone differential ({profile}) on {lanes}: {:.1} s",
        started.elapsed().as_secs_f64()
    );
    assert!(
        output.status.success()
            && stdout
                .lines()
                .any(|line| line.starts_with("STD DATETIME ZONE ORACLE: PASS")),
        "Std.Datetime.Zone differential failed ({}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains(zones) && stdout.contains(&format!("lanes {})", lanes.replace(',', "+"))),
        "{stdout}"
    );
}

#[test]
fn std_datetime_zone_agrees_with_zoneinfo_on_eval() {
    run_zone_differential("full", "eval", "7 zones");
}

#[test]
fn std_datetime_zone_agrees_with_zoneinfo_on_c() {
    run_zone_differential("full", "c", "7 zones");
}

#[test]
fn std_datetime_zone_canary_agrees_with_zoneinfo_on_eval_and_c() {
    run_zone_differential("canary", "eval,c", "2 zones");
}
