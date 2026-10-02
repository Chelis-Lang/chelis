//! chelis#2860: the `Std.Datetime.Business` failure contract of [05-OP-73].
//!
//! Every Business failure is a `fail` whose message is exactly
//! `<function>: domain: <detail>`: a calendar answers only from the days of its
//! horizon, and a query whose answer those days do not determine fails
//! `domain`. Each suite runs as one `chelis test` invocation over a generated
//! fixture whose every test calls one failing (or extreme) expression; the
//! runner reports each call's failure message on its FAIL line. Opaque
//! construction and field access outside the module are checked through
//! `chelis check`, against the plain `Weekmask` record that any module may
//! build.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{make_app, write_file};
use std::collections::BTreeMap;
use std::path::Path;

const IMPORTS: &str = "import Std.Datetime (Date, date, dates_from_epoch_days, dates_epoch_days)\nimport Std.Datetime.Business (Weekmask, BusinessCalendar, BusinessDayRoll, NonBusinessStart, Unadjusted, Following, Preceding, ModifiedFollowing, ModifiedPreceding, RejectNonBusinessStart, RollStartForward, RollStartBackward, business_calendar, try_business_calendar, is_business_day, try_is_business_day, business_day_roll, try_business_day_roll, business_day_offset, try_business_day_offset, business_day_count, try_business_day_count, business_in_all, try_business_in_all, business_in_any, try_business_in_any, dates_is_business_day, dates_business_day_roll, dates_business_day_offset, dates_business_day_count)\nimport Std.Test (assert_true)";

/// Calendars shared by every generated fixture: a 2026 Monday-to-Friday
/// calendar with two holidays, one ending on Saturday 2026-12-26, one starting
/// on Sunday 2026-03-08, a weekend calendar, and full-range calendars.
const PRELUDE: &str = "def i64_minimum() -> i64 = sub(-9223372036854775807i64, 1i64)
def present[a](value: Option[a]) -> i64 =
  match value with {
    | Some(_) => 1i64
    | None => 0i64
  }
def total(values: List[i64]) -> i64 = fold(fn (acc: i64, value: i64) -> add(acc, value), 0i64, values)
def weekdays() -> Weekmask = Weekmask { monday: true, tuesday: true, wednesday: true, thursday: true, friday: true, saturday: false, sunday: false }
def weekend() -> Weekmask = Weekmask { monday: false, tuesday: false, wednesday: false, thursday: false, friday: false, saturday: true, sunday: true }
def no_days() -> Weekmask = Weekmask { monday: false, tuesday: false, wednesday: false, thursday: false, friday: false, saturday: false, sunday: false }
def cal() -> BusinessCalendar = business_calendar(weekdays(), [date(2026i64, 1i64, 1i64), date(2026i64, 1i64, 19i64)], date(2026i64, 1i64, 1i64), date(2026i64, 12i64, 31i64))
def december() -> BusinessCalendar = business_calendar(weekdays(), [], date(2026i64, 12i64, 1i64), date(2026i64, 12i64, 26i64))
def march() -> BusinessCalendar = business_calendar(weekdays(), [], date(2026i64, 3i64, 8i64), date(2026i64, 3i64, 31i64))
def later() -> BusinessCalendar = business_calendar(weekdays(), [], date(2027i64, 1i64, 1i64), date(2027i64, 12i64, 31i64))
def weekends() -> BusinessCalendar = business_calendar(weekend(), [], date(2026i64, 1i64, 1i64), date(2026i64, 12i64, 31i64))
def everything() -> BusinessCalendar = business_calendar(weekdays(), [date(-9999i64, 1i64, 1i64), date(9999i64, 12i64, 31i64)], date(-9999i64, 1i64, 1i64), date(9999i64, 12i64, 31i64))
def everything_weekend() -> BusinessCalendar = business_calendar(weekend(), [], date(-9999i64, 1i64, 1i64), date(9999i64, 12i64, 31i64))
";

const HORIZON: &str = "the horizon 2026-01-01..2026-12-31";

/// (test name, expression, exact failure message).
fn exact_failures() -> Vec<(&'static str, &'static str, String)> {
    vec![
        (
            "calendar_no_business_day",
            "business_calendar(no_days(), [], date(2026i64, 1i64, 1i64), date(2026i64, 12i64, 31i64))",
            "business_calendar: domain: the weekmask has no business day".into(),
        ),
        (
            "calendar_inverted_horizon",
            "business_calendar(weekdays(), [], date(2026i64, 12i64, 31i64), date(2026i64, 1i64, 1i64))",
            "business_calendar: domain: valid_from 2026-12-31 is after valid_until 2026-01-01"
                .into(),
        ),
        (
            "calendar_holiday_outside",
            "business_calendar(weekdays(), [date(2026i64, 3i64, 2i64), date(2027i64, 1i64, 1i64), date(2025i64, 1i64, 1i64)], date(2026i64, 1i64, 1i64), date(2026i64, 12i64, 31i64))",
            format!("business_calendar: domain: holiday 2027-01-01 is outside {HORIZON}"),
        ),
        (
            "is_business_day_after",
            "is_business_day(cal(), date(2027i64, 1i64, 1i64))",
            format!("is_business_day: domain: 2027-01-01 is outside {HORIZON}"),
        ),
        (
            "roll_unadjusted_before",
            "business_day_roll(cal(), date(2025i64, 12i64, 31i64), Unadjusted)",
            format!("business_day_roll: domain: 2025-12-31 is outside {HORIZON}"),
        ),
        (
            "roll_preceding_undetermined",
            "business_day_roll(cal(), date(2026i64, 1i64, 1i64), Preceding)",
            format!(
                "business_day_roll: domain: no business day on or before 2026-01-01 lies inside {HORIZON}"
            ),
        ),
        (
            "roll_following_undetermined",
            "business_day_roll(december(), date(2026i64, 12i64, 26i64), Following)",
            "business_day_roll: domain: no business day on or after 2026-12-26 lies inside the horizon 2026-12-01..2026-12-26".into(),
        ),
        (
            "roll_modified_following_undetermined",
            "business_day_roll(december(), date(2026i64, 12i64, 26i64), ModifiedFollowing)",
            "business_day_roll: domain: no business day on or after 2026-12-26 lies inside the horizon 2026-12-01..2026-12-26".into(),
        ),
        (
            "roll_modified_preceding_undetermined",
            "business_day_roll(march(), date(2026i64, 3i64, 8i64), ModifiedPreceding)",
            "business_day_roll: domain: no business day on or before 2026-03-08 lies inside the horizon 2026-03-08..2026-03-31".into(),
        ),
        (
            "offset_reject_start",
            "business_day_offset(cal(), date(2026i64, 1i64, 3i64), 1i64, RejectNonBusinessStart)",
            "business_day_offset: domain: 2026-01-03 is not a business day".into(),
        ),
        (
            "offset_past_end",
            "business_day_offset(cal(), date(2026i64, 12i64, 31i64), 1i64, RejectNonBusinessStart)",
            format!(
                "business_day_offset: domain: 2026-12-31 plus 1 business days is outside {HORIZON}"
            ),
        ),
        (
            "offset_before_start",
            "business_day_offset(cal(), date(2026i64, 1i64, 3i64), -1i64, RollStartBackward)",
            format!(
                "business_day_offset: domain: 2026-01-02 plus -1 business days is outside {HORIZON}"
            ),
        ),
        (
            "offset_max",
            "business_day_offset(cal(), date(2026i64, 1i64, 2i64), 9223372036854775807i64, RejectNonBusinessStart)",
            format!(
                "business_day_offset: domain: 2026-01-02 plus 9223372036854775807 business days is outside {HORIZON}"
            ),
        ),
        (
            "offset_min",
            "business_day_offset(cal(), date(2026i64, 1i64, 2i64), i64_minimum(), RejectNonBusinessStart)",
            format!(
                "business_day_offset: domain: 2026-01-02 plus -9223372036854775808 business days is outside {HORIZON}"
            ),
        ),
        (
            "offset_start_outside",
            "business_day_offset(cal(), date(2027i64, 1i64, 4i64), 0i64, RollStartForward)",
            format!("business_day_offset: domain: 2027-01-04 is outside {HORIZON}"),
        ),
        (
            "offset_roll_undetermined",
            "business_day_offset(december(), date(2026i64, 12i64, 26i64), 0i64, RollStartForward)",
            "business_day_offset: domain: no business day on or after 2026-12-26 lies inside the horizon 2026-12-01..2026-12-26".into(),
        ),
        (
            "count_end_after",
            "business_day_count(cal(), date(2026i64, 1i64, 1i64), date(2027i64, 1i64, 2i64))",
            format!(
                "business_day_count: domain: end 2027-01-02 is outside {HORIZON} and is not the day after it"
            ),
        ),
        (
            "count_begin_before",
            "business_day_count(cal(), date(2025i64, 12i64, 31i64), date(2026i64, 1i64, 5i64))",
            format!(
                "business_day_count: domain: begin 2025-12-31 is outside {HORIZON} and is not the day after it"
            ),
        ),
        (
            "all_disjoint_horizons",
            "business_in_all(cal(), later())",
            "business_in_all: domain: the horizons 2026-01-01..2026-12-31 and 2027-01-01..2027-12-31 do not intersect".into(),
        ),
        (
            "all_disjoint_weekmasks",
            "business_in_all(cal(), weekends())",
            "business_in_all: domain: no weekday is a business day in both weekmasks".into(),
        ),
        (
            "any_disjoint_horizons",
            "business_in_any(later(), cal())",
            "business_in_any: domain: the horizons 2027-01-01..2027-12-31 and 2026-01-01..2026-12-31 do not intersect".into(),
        ),
        (
            "column_is_business_day",
            "dates_is_business_day(cal(), dates_from_epoch_days(to_tensor([20454i64, 20819i64, 20453i64])))",
            format!("dates_is_business_day: domain: element 1: 2027-01-01 is outside {HORIZON}"),
        ),
        (
            "column_roll",
            "dates_business_day_roll(cal(), dates_from_epoch_days(to_tensor([20456i64, 20454i64])), Preceding)",
            format!(
                "dates_business_day_roll: domain: element 1: no business day on or before 2026-01-01 lies inside {HORIZON}"
            ),
        ),
        (
            "column_offset",
            "dates_business_day_offset(cal(), dates_from_epoch_days(to_tensor([20455i64, 20456i64])), to_tensor([1i64, 1i64]), RejectNonBusinessStart)",
            "dates_business_day_offset: domain: element 1: 2026-01-03 is not a business day"
                .into(),
        ),
        (
            "column_count",
            "dates_business_day_count(cal(), dates_from_epoch_days(to_tensor([20454i64, 20454i64])), dates_from_epoch_days(to_tensor([20819i64, 20820i64])))",
            format!(
                "dates_business_day_count: domain: element 1: end 2027-01-02 is outside {HORIZON} and is not the day after it"
            ),
        ),
    ]
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
    let mut source = format!("module Demo.Tests.Business\n{IMPORTS}\n{PRELUDE}");
    for (name, expression) in expressions {
        source.push_str(&format!(
            "def test_{name}() -> unit ! {{ Test }} = {{\n  _value = {expression}\n  ()\n}}\n"
        ));
    }
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\nanchor = 0i64\n",
    );
    let path = app_pkg.join("tests/business_cases.ch");
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
fn std_datetime_business_failures_report_their_exact_message() {
    let cases = exact_failures();
    let expressions: Vec<(String, String)> = cases
        .iter()
        .map(|(name, expression, _)| (name.to_string(), expression.to_string()))
        .collect();
    let outcomes = run_expression_suite("business-failures-2860", &expressions);
    for (name, expression, expected) in &cases {
        assert_eq!(
            outcomes.get(*name),
            Some(&Some(expected.clone())),
            "{name}: `{expression}` must fail with exactly `{expected}`"
        );
    }
}

/// Extreme calls through the `try_` forms, which share their trapping twins'
/// checks, arithmetic, and detail text: each returns `Some` or `None`, and a
/// primitive trap anywhere would fail the test under a primitive's message
/// instead. Calendars at both range edges, with and without holidays at the
/// edges, meet the edge dates, i64 extremes, every roll and start, and every
/// pairing of calendars.
const EXTREME_SWEEP: &str = "{
  calendars = [cal(), everything(), everything_weekend(), weekends()]
  days = [date(-9999i64, 1i64, 1i64), date(-9999i64, 1i64, 2i64), date(9999i64, 12i64, 31i64), date(9999i64, 12i64, 30i64), date(2026i64, 1i64, 1i64), date(2026i64, 12i64, 31i64)]
  counts = [9223372036854775807i64, i64_minimum(), 0i64, 1i64, -1i64, 5000000i64, -5000000i64]
  rolls = [Unadjusted, Following, Preceding, ModifiedFollowing, ModifiedPreceding]
  starts = [RejectNonBusinessStart, RollStartForward, RollStartBackward]
  per_day = fn (c: BusinessCalendar, d: Date) -> add(add(present(try_is_business_day(c, d)), total(map(fn (r: BusinessDayRoll) -> present(try_business_day_roll(c, d, r)), rolls))), add(total(map(fn (n: i64) -> total(map(fn (s: NonBusinessStart) -> present(try_business_day_offset(c, d, n, s)), starts)), counts)), total(map(fn (e: Date) -> present(try_business_day_count(c, d, e)), days))))
  per_calendar = fn (c: BusinessCalendar) -> add(total(map(fn (d: Date) -> per_day(c, d), days)), total(map(fn (o: BusinessCalendar) -> add(present(try_business_in_all(c, o)), present(try_business_in_any(c, o))), calendars)))
  found = total(map(per_calendar, calendars))
  assert_true(and(gt(found, 0i64), lt(found, 824i64)), \"both accepted and rejected extremes\")
}";

#[test]
fn std_datetime_business_extreme_arguments_raise_no_primitive_trap() {
    let outcomes = run_expression_suite(
        "business-extremes-2860",
        &[("sweep".to_string(), EXTREME_SWEEP.to_string())],
    );
    assert_eq!(
        outcomes.get("sweep"),
        Some(&None),
        "824 extreme try_ calls must neither trap nor agree on presence"
    );
}

/// [05-OP-73]: `BusinessCalendar` is opaque, so construction and field access
/// outside `Std.Datetime.Business` are `OpaqueTypeViolation`s, while the plain
/// `Weekmask` record is built and read anywhere.
#[test]
fn std_datetime_business_calendar_rejects_outside_construction() {
    let (_dir, reef_home, app_pkg) = make_app("business-opaque-2860");
    let path = app_pkg.join("src/main.ch");
    let weekmask = "Weekmask { monday: true, tuesday: true, wednesday: true, thursday: true, friday: true, saturday: false, sunday: false }";
    write_file(
        &path,
        &format!(
            "module Demo.Main\nimport Std.Datetime (date)\nimport Std.Datetime.Business (Weekmask, BusinessCalendar, business_calendar)\nforged = BusinessCalendar {{ weekmask: {weekmask}, holidays: [], valid_from: 0i64, valid_until: 0i64 }}\npeeked = business_calendar({weekmask}, [], date(2026i64, 1i64, 1i64), date(2026i64, 1i64, 2i64)).holidays\n"
        ),
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        !success,
        "forging a business calendar must not check:\n{rendered}"
    );
    assert!(
        rendered.contains(
            "record construction of opaque type `BusinessCalendar` outside its defining module"
        ),
        "the forged calendar was not rejected as an opaque construction:\n{rendered}"
    );
    assert!(
        rendered.contains("field access") && rendered.contains("opaque type `BusinessCalendar`"),
        "field access on a calendar was not rejected:\n{rendered}"
    );

    write_file(
        &path,
        &format!(
            "module Demo.Main\nimport Std.Datetime (date)\nimport Std.Datetime.Business (Weekmask, business_calendar, business_calendar_weekmask)\nsaturday = {weekmask}.saturday\nkept = business_calendar_weekmask(business_calendar({weekmask}, [], date(2026i64, 1i64, 1i64), date(2026i64, 1i64, 2i64))).friday\n"
        ),
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        success,
        "the plain Weekmask record is built and read outside its module:\n{rendered}"
    );
}
