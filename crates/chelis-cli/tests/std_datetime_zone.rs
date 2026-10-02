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

const IMPORTS: &str = "import Std.Datetime (DateTime, ClampToMonthEnd, RejectInvalidDay, date, time, datetime, instant_from_unix, offset_from_seconds, period, duration)\nimport Std.Datetime.Zone (TimeZone, Zoned, ZonedText, EarlierInstant, LaterInstant, CompatibleInstant, RejectNonUniqueLocal, UseWrittenOffset, UseZoneRules, RejectOffsetMismatch, time_zone_from_tzif, try_time_zone_from_tzif, time_zone_fixed, time_zone_utc, time_zone_offset_at, try_time_zone_offset_at, zoned, try_zoned, zoned_from_local, try_zoned_from_local, zoned_add_duration, zoned_add_period, parse_zoned_text, try_parse_zoned_text, zoned_from_text, try_zoned_from_text)";

/// Helpers for the generated fixtures: zones read from the checked-in TZif
/// files (`@TZIF@` is the fixture directory), a TZif writer for files no
/// fixture holds, and local readings.
const PRELUDE: &str = r#"def i64_minimum() -> i64 = sub(-9223372036854775807i64, 1i64)
def zone_file(path: string) -> List[i64] ! { IO } = read_bytes(string_concat("@TZIF@/", path))
def new_york() -> TimeZone ! { IO } = time_zone_from_tzif("America/New_York", zone_file("zones/America/New_York.tzif"))
def kiritimati() -> TimeZone ! { IO } = time_zone_from_tzif("Pacific/Kiritimati", zone_file("zones/Pacific/Kiritimati.tzif"))
def short_lived() -> TimeZone ! { IO } = time_zone_from_tzif("Etc/Short_Lived", zone_file("synthetic/empty_footer.tzif"))
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
        r#"zoned_from_local: domain: 2001-09-08T12:00:00 is within one day of unix second 1000000000, the last transition of "Etc/Short_Lived", whose footer is empty"#,
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

/// Runs one `chelis test` over a fixture with one test per expression and
/// returns each test's outcome: `None` for PASS, `Some(message)` for FAIL.
fn run_expression_suite(
    dir_name: &str,
    expressions: &[(String, String)],
) -> BTreeMap<String, Option<String>> {
    let (_dir, reef_home, app_pkg) = make_app(dir_name);
    let prelude = PRELUDE.replace("@TZIF@", &fixture_dir().to_string_lossy());
    let mut source = format!("module Demo.Tests.Zone\n{IMPORTS}\n{prelude}");
    for (name, expression) in expressions {
        source.push_str(&format!(
            "def test_{name}() -> unit ! {{ Test, IO }} = {{\n  _value = {expression}\n  ()\n}}\n"
        ));
    }
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\nanchor = 0i64\n",
    );
    let path = app_pkg.join("tests/zone_cases.ch");
    write_file(&path, &source);
    let (_, rendered) = run_chelis(
        &app_pkg,
        &reef_home,
        &["test", "--batch-mode", "file", path.to_str().unwrap()],
    );
    let mut outcomes = BTreeMap::new();
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
        } else if let Some(message) = verdict
            .strip_prefix("FAIL (")
            .and_then(|tail| tail.strip_suffix(')'))
        {
            outcomes.insert(name.to_string(), Some(message.to_string()));
        }
    }
    assert_eq!(
        outcomes.len(),
        expressions.len(),
        "every generated test must report exactly one verdict:\n{rendered}"
    );
    outcomes
}

#[test]
fn std_datetime_zone_failures_report_their_exact_message() {
    let expressions: Vec<(String, String)> = EXACT_FAILURES
        .iter()
        .map(|(name, expression, _)| (name.to_string(), expression.to_string()))
        .collect();
    let outcomes = run_expression_suite("datetime-zone-failures-2862", &expressions);
    for (name, expression, expected) in EXACT_FAILURES {
        assert_eq!(
            outcomes.get(*name),
            Some(&Some(expected.to_string())),
            "{name}: `{expression}` must fail with exactly `{expected}`"
        );
    }
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
            for offset in ["86399i64", "-86399i64"] {
                for conflict in conflicts {
                    cases.push((
                        "zoned_from_text",
                        format!("zoned_from_text(ZonedText {{ local: {local}, offset: offset_from_seconds({offset}), zone_name: \"X\", critical: true }}, {zone}, {conflict})"),
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
    let outcomes = run_expression_suite("datetime-zone-extremes-2862", &expressions);
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
            "{header}plain = ZonedText {{ local: datetime(date(2026i64, 1i64, 1i64), time(0i64, 0i64, 0i64, 0i64)), offset: offset_from_seconds(0i64), zone_name: \"UTC\", critical: false }}\nname = parse_zoned_text(\"2026-01-01T00:00:00Z[UTC]\").zone_name\n"
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

/// The differential: every fixture zone's offsets, every gap and fold under
/// each `Disambiguation`, and the RFC 9557 text, on `chelis eval` and on
/// compiled C, each against Python's `zoneinfo` reading the same file.
#[test]
fn std_datetime_zone_agrees_with_zoneinfo_on_eval_and_c() {
    let python =
        managed_python::managed_python(&repo_root()).unwrap_or_else(|error| panic!("{error}"));
    let chelis = assert_cmd::cargo_bin!("chelis").to_path_buf();
    let output = std::process::Command::new(python)
        .arg(repo_root().join("scripts/datetime_zone_differential.py"))
        .arg("--chelis")
        .arg(&chelis)
        .arg("--reef-home")
        .arg(&common::SHARED_REEF.reef_home)
        .args(["--toolchain-json", &toolchain_json(), "--lanes", "eval,c"])
        .output()
        .expect("spawn the zone differential harness");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            && stdout
                .lines()
                .any(|line| line.starts_with("STD DATETIME ZONE ORACLE: PASS")),
        "Std.Datetime.Zone differential failed ({}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("7 zones") && stdout.contains("lanes eval+c"),
        "{stdout}"
    );
}
