//! chelis#2861: the `Std.Datetime.Columns` failure contract of [05-OP-73].
//!
//! A column callable fails exactly where its scalar twin fails at some
//! element, with the twin's kind for the lowest such index and the twin's
//! detail prefixed `element k: `, under the column callable's name.
//! A column argument whose length differs from another argument's fails
//! `domain` before any element is read, and no primitive numeric trap
//! escapes a call made with extreme arguments. Each suite runs as one `chelis test`
//! invocation over a generated fixture whose every test evaluates one
//! expression; the runner reports each call's failure message on its FAIL
//! line. Construction of the opaque `Durations` outside its module is checked
//! through `chelis check`.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{make_app, write_file};
use std::collections::BTreeMap;
use std::path::Path;

const IMPORTS: &str = "import Std.Datetime (ClampToMonthEnd, RejectInvalidDay, Hours, Minutes, Seconds, Milliseconds, Microseconds, Nanoseconds, Dates, Instants, instant_from_unix, offset_from_seconds, duration, dates_from_epoch_days, instants_from_unix)\nimport Std.Datetime.Columns (Durations, durations, try_durations, durations_seconds, durations_nanoseconds, dates_year, dates_month, dates_day, dates_weekday_iso_number, dates_day_of_year, dates_from_ymd, try_dates_from_ymd, dates_add_days, dates_add_months, dates_days_until, dates_lt, dates_lte, dates_gt, dates_gte, try_parse_dates, dates_to_strings, instants_from_unix_count, instants_to_unix_count, instants_add_duration, instants_until, instants_round_to, instants_to_dates_at, instants_seconds_since_f64, instants_lt, instants_lte, instants_gt, instants_gte)\nimport Std.Rounding (RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)";

/// Columns whose length only the running program knows, so the checker
/// accepts a pair of different lengths and the call itself must refuse it.
const PRELUDE: &str = "def i64_minimum() -> i64 = sub(-9223372036854775807i64, 1i64)\ndef zeros[k](count: i64) -> tensor[k, i64] = to_tensor(map(fn (i: i64) -> mul(i, 0i64), range(0i64, count)))\ndef days[k](count: i64) -> Dates[k] = dates_from_epoch_days(zeros(count))\ndef stamps[k](count: i64) -> Instants[k] = instants_from_unix(zeros(count), zeros(count))\n";

/// (test name, expression, exact failure message).
const EXACT_FAILURES: &[(&str, &str, &str)] = &[
    (
        "durations_element",
        "durations(to_tensor([0i64, 9223372036854775807i64]), to_tensor([0i64, 1000000000i64]))",
        "durations: domain: element 1: second 9223372036854775807 with nanosecond 1000000000 is outside the duration range",
    ),
    (
        "dates_from_ymd_day",
        "dates_from_ymd(to_tensor([2024i64, 2024i64, 2023i64]), to_tensor([2i64, 2i64, 2i64]), to_tensor([29i64, 30i64, 29i64]))",
        "dates_from_ymd: domain: element 1: day 30 is outside 1..29 for 2024-02",
    ),
    (
        "dates_from_ymd_negative_year",
        "dates_from_ymd(to_tensor([-44i64]), to_tensor([2i64]), to_tensor([30i64]))",
        "dates_from_ymd: domain: element 0: day 30 is outside 1..29 for -000044-02",
    ),
    (
        "dates_from_ymd_year",
        "dates_from_ymd(to_tensor([2024i64, -10000i64]), to_tensor([1i64, 1i64]), to_tensor([1i64, 1i64]))",
        "dates_from_ymd: domain: element 1: year -10000 is outside -9999..9999",
    ),
    (
        "dates_from_ymd_month",
        "dates_from_ymd(to_tensor([2024i64]), to_tensor([0i64]), to_tensor([1i64]))",
        "dates_from_ymd: domain: element 0: month 0 is outside 1..12",
    ),
    (
        "dates_from_ymd_lowest_index",
        "dates_from_ymd(to_tensor([2024i64, 2023i64, 2023i64]), to_tensor([13i64, 2i64, 13i64]), to_tensor([1i64, 29i64, 1i64]))",
        "dates_from_ymd: domain: element 0: month 13 is outside 1..12",
    ),
    (
        "dates_add_days_end",
        "dates_add_days(dates_from_epoch_days(to_tensor([0i64, 2932896i64])), to_tensor([1i64, 1i64]))",
        "dates_add_days: overflow: element 1: 9999-12-31 plus 1 days is outside the supported date range",
    ),
    (
        "dates_add_days_min",
        "dates_add_days(dates_from_epoch_days(to_tensor([0i64])), to_tensor([i64_minimum()]))",
        "dates_add_days: overflow: element 0: 1970-01-01 plus -9223372036854775808 days is outside the supported date range",
    ),
    (
        "dates_add_days_lengths",
        "dates_add_days(days(2i64), zeros(3i64))",
        "dates_add_days: domain: arguments have 2 and 3 elements",
    ),
    (
        "dates_add_months_reject",
        "dates_add_months(dates_from_epoch_days(to_tensor([19754i64, 19753i64])), to_tensor([1i64, 1i64]), RejectInvalidDay)",
        "dates_add_months: domain: element 1: day 31 is outside 1..29 for 2024-02",
    ),
    (
        "dates_add_months_lowest_index_domain",
        "dates_add_months(dates_from_epoch_days(to_tensor([19753i64, 2932866i64])), to_tensor([1i64, 1i64]), RejectInvalidDay)",
        "dates_add_months: domain: element 0: day 31 is outside 1..29 for 2024-02",
    ),
    (
        "dates_add_months_lowest_index_overflow",
        "dates_add_months(dates_from_epoch_days(to_tensor([2932866i64, 19753i64])), to_tensor([1i64, 1i64]), RejectInvalidDay)",
        "dates_add_months: overflow: element 0: 9999-12-01 plus 1 months is outside the supported date range",
    ),
    (
        "dates_add_months_range_before_day",
        "dates_add_months(dates_from_epoch_days(to_tensor([19753i64])), to_tensor([9223372036854775807i64]), RejectInvalidDay)",
        "dates_add_months: overflow: element 0: 2024-01-31 plus 9223372036854775807 months is outside the supported date range",
    ),
    (
        "dates_add_months_lengths",
        "dates_add_months(days(1i64), zeros(2i64), ClampToMonthEnd)",
        "dates_add_months: domain: arguments have 1 and 2 elements",
    ),
    (
        "dates_days_until_lengths",
        "dates_days_until(days(2i64), days(3i64))",
        "dates_days_until: domain: arguments have 2 and 3 elements",
    ),
    (
        "dates_lt_lengths",
        "dates_lt(days(2i64), days(1i64))",
        "dates_lt: domain: arguments have 2 and 1 elements",
    ),
    (
        "dates_gte_lengths",
        "dates_gte(days(0i64), days(1i64))",
        "dates_gte: domain: arguments have 0 and 1 elements",
    ),
    (
        "instants_from_unix_count_hours",
        "instants_from_unix_count(to_tensor([0i64, 9223372036854775807i64]), Hours)",
        "instants_from_unix_count: domain: element 1: 9223372036854775807 hours since the unix epoch is outside the supported instant range",
    ),
    (
        "instants_from_unix_count_seconds",
        "instants_from_unix_count(to_tensor([-377705030402i64]), Seconds)",
        "instants_from_unix_count: domain: element 0: -377705030402 seconds since the unix epoch is outside the supported instant range",
    ),
    (
        "instants_to_unix_count_inexact",
        "instants_to_unix_count(instants_from_unix(to_tensor([0i64, 1i64]), to_tensor([0i64, 5i64])), Seconds, RejectInexact)",
        "instants_to_unix_count: domain: element 1: 1970-01-01T00:00:01.000000005Z is not a whole number of seconds",
    ),
    (
        "instants_to_unix_count_range",
        "instants_to_unix_count(instants_from_unix(to_tensor([253402214400i64]), to_tensor([0i64])), Nanoseconds, RoundTowardNegative)",
        "instants_to_unix_count: overflow: element 0: 9999-12-31T00:00:00Z in nanoseconds does not fit in i64",
    ),
    (
        "instants_to_unix_count_lowest_index",
        "instants_to_unix_count(instants_from_unix(to_tensor([0i64, -377705030401i64, 3600i64]), to_tensor([0i64, 0i64, 1i64])), Nanoseconds, RejectInexact)",
        "instants_to_unix_count: overflow: element 1: -009999-01-01T23:59:59Z in nanoseconds does not fit in i64",
    ),
    (
        "instants_add_duration_end",
        "instants_add_duration(instants_from_unix(to_tensor([0i64, 253402214400i64]), to_tensor([0i64, 999999999i64])), durations(to_tensor([0i64, 0i64]), to_tensor([0i64, 1i64])))",
        "instants_add_duration: overflow: element 1: 9999-12-31T00:00:00.999999999Z plus PT0.000000001S is outside the supported instant range",
    ),
    (
        "instants_add_duration_lengths",
        "instants_add_duration(stamps(1i64), durations(zeros(2i64), zeros(2i64)))",
        "instants_add_duration: domain: arguments have 1 and 2 elements",
    ),
    (
        "instants_until_lengths",
        "instants_until(stamps(3i64), stamps(2i64))",
        "instants_until: domain: arguments have 3 and 2 elements",
    ),
    (
        "instants_lt_lengths",
        "instants_lt(stamps(1i64), stamps(0i64))",
        "instants_lt: domain: arguments have 1 and 0 elements",
    ),
    (
        "instants_round_to_increment_empty_column",
        "instants_round_to(stamps(0i64), duration(0i64, 0i64), RoundTiesToEven)",
        "instants_round_to: domain: increment PT0S is not positive",
    ),
    (
        "instants_round_to_increment_divides",
        "instants_round_to(stamps(2i64), duration(7i64, 0i64), RoundTiesToEven)",
        "instants_round_to: domain: increment PT7S does not divide one day",
    ),
    (
        "instants_round_to_inexact",
        "instants_round_to(instants_from_unix(to_tensor([2i64, 1i64]), to_tensor([0i64, 0i64])), duration(2i64, 0i64), RejectInexact)",
        "instants_round_to: domain: element 1: 1970-01-01T00:00:01Z is not a multiple of PT2S",
    ),
    (
        "instants_round_to_range",
        "instants_round_to(instants_from_unix(to_tensor([253402214400i64]), to_tensor([999999999i64])), duration(1i64, 0i64), RoundTowardPositive)",
        "instants_round_to: overflow: element 0: rounding 9999-12-31T00:00:00.999999999Z to a multiple of PT1S leaves the supported instant range",
    ),
    (
        "instants_round_to_range_below",
        "instants_round_to(instants_from_unix(to_tensor([-377705030401i64]), to_tensor([0i64])), duration(2i64, 0i64), RoundTowardNegative)",
        "instants_round_to: overflow: element 0: rounding -009999-01-01T23:59:59Z to a multiple of PT2S leaves the supported instant range",
    ),
];

const MAX: &str = "9223372036854775807i64";
const MIN: &str = "i64_minimum()";

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
    let mut source = format!("module Demo.Tests.Columns\n{IMPORTS}\n{PRELUDE}");
    for (name, expression) in expressions {
        source.push_str(&format!(
            "def test_{name}() -> unit ! {{ Test }} = {{\n  _value = {expression}\n  ()\n}}\n"
        ));
    }
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\nanchor = 0i64\n",
    );
    let path = app_pkg.join("tests/columns_cases.ch");
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
fn std_datetime_columns_failures_report_their_exact_message() {
    let expressions: Vec<(String, String)> = EXACT_FAILURES
        .iter()
        .map(|(name, expression, _)| (name.to_string(), expression.to_string()))
        .collect();
    let outcomes = run_expression_suite("datetime-columns-failures-2861", &expressions);
    for (name, expression, expected) in EXACT_FAILURES {
        assert_eq!(
            outcomes.get(*name),
            Some(&Some(expected.to_string())),
            "{name}: `{expression}` must fail with exactly `{expected}`"
        );
    }
}

/// The negative twin of the exact-failure suite: the same callables on
/// admitted arguments of equal length, the masked forms on every argument
/// their twins reject, and the callables that never fail on the range edges,
/// all pass.
#[test]
fn std_datetime_columns_admitted_arguments_pass() {
    let admitted = [
        "durations(zeros(2i64), zeros(2i64))",
        "try_durations(to_tensor([9223372036854775807i64, 0i64]), to_tensor([1000000000i64, -1i64]))",
        "try_dates_from_ymd(to_tensor([10000i64, 2024i64]), to_tensor([0i64, 2i64]), to_tensor([32i64, 29i64]))",
        "try_parse_dates([\"\", \"2024-02-30\", \"x\"])",
        "dates_add_days(dates_from_epoch_days(to_tensor([2932895i64])), to_tensor([1i64]))",
        "dates_add_months(dates_from_epoch_days(to_tensor([19753i64])), to_tensor([1i64]), ClampToMonthEnd)",
        "dates_days_until(dates_from_epoch_days(to_tensor([-4371587i64])), dates_from_epoch_days(to_tensor([2932896i64])))",
        "dates_gte(days(0i64), days(0i64))",
        "instants_to_unix_count(instants_from_unix(to_tensor([253402214400i64]), to_tensor([999999000i64])), Microseconds, RejectInexact)",
        "instants_until(instants_from_unix(to_tensor([253402214400i64]), to_tensor([999999999i64])), instants_from_unix(to_tensor([-377705030401i64]), to_tensor([0i64])))",
        "instants_round_to(stamps(0i64), duration(86400i64, 0i64), RejectInexact)",
        "instants_to_dates_at(instants_from_unix(to_tensor([253402214400i64]), to_tensor([999999999i64])), offset_from_seconds(86399i64))",
        "instants_seconds_since_f64(instants_from_unix(to_tensor([253402214400i64]), to_tensor([999999999i64])), instant_from_unix(-377705030401i64, 0i64))",
    ];
    let expressions: Vec<(String, String)> = admitted
        .iter()
        .enumerate()
        .map(|(index, expression)| (format!("admitted_{index:02}"), expression.to_string()))
        .collect();
    let outcomes = run_expression_suite("datetime-columns-admitted-2861", &expressions);
    for (index, expression) in admitted.iter().enumerate() {
        assert_eq!(
            outcomes[&format!("admitted_{index:02}")],
            None,
            "`{expression}` must pass"
        );
    }
}

/// Every call below passes, or fails `domain` or `overflow` under its own
/// callable's name; a primitive trap would surface under another message.
fn extreme_cases() -> Vec<(&'static str, String)> {
    let ints = [MAX, MIN, "0i64"];
    let edge_days = "dates_from_epoch_days(to_tensor([-4371587i64, 2932896i64, 0i64]))";
    let edge_instants = "instants_from_unix(to_tensor([-377705030401i64, 253402214400i64, -1i64]), to_tensor([0i64, 999999999i64, 999999999i64]))";
    let units = [
        "Hours",
        "Minutes",
        "Seconds",
        "Milliseconds",
        "Microseconds",
        "Nanoseconds",
    ];
    let roundings = [
        "RoundTowardNegative",
        "RoundTowardPositive",
        "RoundTowardZero",
        "RoundAwayFromZero",
        "RoundTiesToEven",
        "RoundTiesToAway",
        "RejectInexact",
    ];
    let increments = [
        "duration(1i64, 0i64)".to_string(),
        "duration(0i64, 1i64)".to_string(),
        "duration(86400i64, 0i64)".to_string(),
        "duration(28800i64, 0i64)".to_string(),
        format!("duration({MAX}, 0i64)"),
        format!("duration({MIN}, 0i64)"),
    ];
    let mut cases: Vec<(&'static str, String)> = Vec::new();
    for a in ints {
        let column = format!("to_tensor([{a}, {a}, {a}])");
        cases.push(("durations", format!("durations({column}, {column})")));
        cases.push((
            "try_durations",
            format!("try_durations({column}, {column})"),
        ));
        cases.push((
            "dates_from_ymd",
            format!("dates_from_ymd({column}, {column}, {column})"),
        ));
        cases.push((
            "try_dates_from_ymd",
            format!("try_dates_from_ymd({column}, {column}, {column})"),
        ));
        cases.push((
            "dates_add_days",
            format!("dates_add_days({edge_days}, {column})"),
        ));
        for policy in ["ClampToMonthEnd", "RejectInvalidDay"] {
            cases.push((
                "dates_add_months",
                format!("dates_add_months({edge_days}, {column}, {policy})"),
            ));
        }
        for unit in units {
            cases.push((
                "instants_from_unix_count",
                format!("instants_from_unix_count({column}, {unit})"),
            ));
        }
        for b in ["0i64", "999999999i64"] {
            cases.push((
                "instants_add_duration",
                format!(
                    "instants_add_duration({edge_instants}, durations(to_tensor([{a}, {a}, {a}]), to_tensor([{b}, {b}, {b}])))"
                ),
            ));
        }
    }
    for unit in units {
        for rounding in roundings {
            cases.push((
                "instants_to_unix_count",
                format!("instants_to_unix_count({edge_instants}, {unit}, {rounding})"),
            ));
        }
    }
    for increment in &increments {
        for rounding in roundings {
            cases.push((
                "instants_round_to",
                format!("instants_round_to({edge_instants}, {increment}, {rounding})"),
            ));
        }
    }
    for origin in [
        "instant_from_unix(-377705030401i64, 0i64)",
        "instant_from_unix(253402214400i64, 999999999i64)",
    ] {
        cases.push((
            "instants_seconds_since_f64",
            format!("instants_seconds_since_f64({edge_instants}, {origin})"),
        ));
    }
    for offset in ["86399i64", "-86399i64"] {
        cases.push((
            "instants_to_dates_at",
            format!("instants_to_dates_at({edge_instants}, offset_from_seconds({offset}))"),
        ));
    }
    cases.push((
        "instants_until",
        format!("instants_until({edge_instants}, {edge_instants})"),
    ));
    cases.push((
        "dates_days_until",
        format!("dates_days_until({edge_days}, {edge_days})"),
    ));
    cases
}

#[test]
fn std_datetime_columns_extreme_arguments_raise_no_primitive_trap() {
    let cases = extreme_cases();
    let expressions: Vec<(String, String)> = cases
        .iter()
        .enumerate()
        .map(|(index, (_, expression))| (format!("case_{index:04}"), expression.clone()))
        .collect();
    let outcomes = run_expression_suite("datetime-columns-extremes-2861", &expressions);
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
            assert!(
                !function.starts_with("try_"),
                "the masked form `{expression}` failed with `{message}`"
            );
        }
    }
    assert!(
        failures > 0 && failures < cases.len(),
        "the sweep must exercise both accepted and rejected extremes ({failures} of {})",
        cases.len()
    );
}

/// [05-OP-73]: `Durations` is opaque, so construction and field access
/// outside `Std.Datetime.Columns` are `OpaqueTypeViolation`s, while its
/// exported producer and accessors check.
#[test]
fn std_datetime_columns_durations_reject_outside_construction() {
    let (_dir, reef_home, app_pkg) = make_app("datetime-columns-opaque-2861");
    let path = app_pkg.join("src/main.ch");
    write_file(
        &path,
        "module Demo.Main\nimport Std.Datetime.Columns (Durations, durations, durations_seconds)\nforged = Durations { seconds: to_tensor([0i64]), nanoseconds: to_tensor([0i64]) }\npeeked = durations(to_tensor([1i64]), to_tensor([0i64])).seconds\n",
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        !success,
        "forging an opaque Durations column must not check:\n{rendered}"
    );
    assert!(
        rendered
            .contains("record construction of opaque type `Durations` outside its defining module"),
        "the Durations literal was not rejected as an opaque construction:\n{rendered}"
    );
    assert!(
        rendered.contains("field access") && rendered.contains("opaque type `Durations`"),
        "field access on an opaque Durations was not rejected:\n{rendered}"
    );

    write_file(
        &path,
        "module Demo.Main\nimport Std.Datetime.Columns (Durations, durations, durations_seconds)\nseconds = durations_seconds(durations(to_tensor([1i64]), to_tensor([0i64])))\n",
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        success,
        "the exported producer and accessor must check:\n{rendered}"
    );
}
