use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, gcc_available, make_app, write_file};

#[test]
fn reef_std_decimal_calls_raise_issue_2778_error() {
    let (_dir, reef_home, app_pkg) = make_app("decimal-fence-2778");
    let value = "Decimal { coefficient: cast(1, i64), scale: cast(0, i64) }";
    let invalid = "Decimal { coefficient: cast(1, i64), scale: cast(-1, i64) }";
    let cases = [
        "round_half_up()".to_owned(),
        "round_half_even()".to_owned(),
        "round_down()".to_owned(),
        "round_up()".to_owned(),
        "decimal(\"9223372036854775807.0\")".to_owned(),
        "try_decimal(\"0.1\")".to_owned(),
        "decimal_from_int(cast(1, i64))".to_owned(),
        format!("decimal_add({value}, {value})"),
        format!("decimal_sub({value}, {value})"),
        format!("decimal_mul({value}, {value})"),
        format!("decimal_div({value}, {value}, cast(0, i64), round_half_even())"),
        format!("decimal_eq({value}, {value})"),
        format!("decimal_lt({value}, {value})"),
        format!("decimal_lte({value}, {value})"),
        format!("decimal_gt({value}, {value})"),
        format!("decimal_gte({value}, {value})"),
        format!("decimal_to_float({value})"),
        format!("decimal_to_string({invalid})"),
    ];
    let path = app_pkg.join("src/main.ch");
    for (index, expression) in cases.iter().enumerate() {
        let source = format!(
            "module Demo.Main\nimport Std.Decimal (Decimal, RoundingMode, round_half_up, round_half_even, round_down, round_up, decimal, try_decimal, decimal_from_int, decimal_add, decimal_sub, decimal_mul, decimal_div, decimal_eq, decimal_lt, decimal_lte, decimal_gt, decimal_gte, decimal_to_float, decimal_to_string)\nresult = {expression}\n"
        );
        write_file(&path, &source);
        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .current_dir(&app_pkg)
            .args(["eval", "--file", path.to_str().unwrap()])
            .output()
            .expect("eval output");
        let rendered = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !output.status.success()
                && rendered.contains("Std.Decimal is unavailable")
                && rendered.contains("#2778"),
            "case {index} ({expression}) escaped the Decimal fence: {rendered}"
        );
    }
}

#[test]
#[ignore = "manual gate: Phase 3i std package acceptance suite exceeds the default inner-loop budget"]
fn reef_std_decimal_module_checks_then_rejects_eval() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-decimal");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main
import Std.Decimal (decimal, decimal_add, decimal_div, decimal_eq, decimal_from_int, decimal_to_string, round_half_even, try_decimal)
exact = decimal_eq(decimal_add(decimal("0.1"), decimal("0.2")), decimal("0.3"))
banker = decimal_to_string(decimal_div(decimal_from_int(cast(5, i64)), decimal_from_int(cast(2, i64)), cast(0, i64), round_half_even()))
bad_decimal = match try_decimal("x.y") with {
  | Some(_) => "bad"
  | None => "invalid-decimal"
}
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Std.Decimal is unavailable"))
        .stderr(predicate::str::contains("#2778"));
}

/// A program over `Std.Datetime` whose every root is a canonical value.
const DATETIME_ACCEPTANCE_PROGRAM: &str = r#"module Demo.Main
import Std.Datetime (ClampToMonthEnd, Milliseconds, date, date_add_days, date_add_months, date_to_string, date_weekday, weekday_name, parse_instant, instant_to_unix_count, duration_to_string, instant_until, parse_period, period_to_string, date_period_until)
import Std.Rounding (RoundTowardNegative)
month_end = date_to_string(date_add_months(date(2024i64, 1i64, 31i64), 1i64, ClampToMonthEnd))
weekday = weekday_name(date_weekday(date(2026i64, 10i64, 1i64)))
millis = instant_to_unix_count(parse_instant("2026-10-01T09:30:00.25-04:00"), Milliseconds, RoundTowardNegative)
elapsed = duration_to_string(instant_until(parse_instant("2026-10-01T00:00:00Z"), parse_instant("2026-10-01T01:01:01.5Z")))
gap = period_to_string(date_period_until(date(2024i64, 1i64, 31i64), date(2025i64, 3i64, 1i64)))
year = period_to_string(parse_period("P1Y"))
same = eq(date(2024i64, 2i64, 29i64), date_add_days(date(2024i64, 2i64, 28i64), 1i64))
different = eq(date(2024i64, 2i64, 29i64), date(2024i64, 3i64, 1i64))
"#;

/// chelis#2859: `Std.Datetime` replaces `Std.Time`. A program importing it
/// type-checks and evaluates through the bundled standard library.
#[test]
fn reef_std_datetime_checks_and_evaluates() {
    let (_dir, reef_home, app_pkg) = make_app("datetime-acceptance-2859");
    let path = app_pkg.join("src/main.ch");
    write_file(&path, DATETIME_ACCEPTANCE_PROGRAM);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("month_end = 2024-02-29"))
        .stdout(predicate::str::contains("weekday = thursday"))
        .stdout(predicate::str::contains("millis = 1790861400250"))
        .stdout(predicate::str::contains("elapsed = PT3661.5S"))
        .stdout(predicate::str::contains("gap = P13M1D"))
        .stdout(predicate::str::contains("year = P12M"))
        .stdout(predicate::str::contains("same = true"))
        .stdout(predicate::str::contains("different = false"));
}

/// chelis#2859: the compiled C lane builds a program over `Std.Datetime` and
/// prints exactly what `chelis eval` prints: the program's own roots and
/// nothing from the imported modules.
#[test]
fn reef_std_datetime_compiled_lane_matches_eval() {
    if !gcc_available() {
        return;
    }
    let (_dir, reef_home, app_pkg) = make_app("datetime-lanes-2859");
    let path = app_pkg.join("src/main.ch");
    write_file(&path, DATETIME_ACCEPTANCE_PROGRAM);
    let evaluated = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval output");
    assert!(
        evaluated.status.success(),
        "eval must succeed: {evaluated:?}"
    );
    let evaluated = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    let roots = DATETIME_ACCEPTANCE_PROGRAM
        .lines()
        .filter(|line| line.contains(" = "))
        .count();
    assert_eq!(
        evaluated.trim().lines().count(),
        roots,
        "eval prints every root:\n{evaluated}"
    );
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    assert_eq!(
        evaluated.trim(),
        compiled.trim(),
        "the compiled lane must print exactly the eval lane's output"
    );
}

#[test]
#[ignore = "manual gate: Phase 3i std package acceptance suite exceeds the default inner-loop budget"]
fn reef_package_mode_preserves_split_map_and_runtime_reshape_typing() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-package-typing");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

source = pad_sequences_to([[1.0, 2.0], [3.0, 4.0]], cast(2, i64), 0.0)
rows = split(source, cast(0, i32), [cast(1, i64), cast(1, i64)])
flat_rows = map(
  fn (row: tensor[piece, seq, f32]) -> reshape(row, [cast(2, i64)]),
  rows
)
first = index(flat_rows, cast(0, i64))
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "first = tensor(shape=[2], data=[1.0, 2.0])",
        ));
}
