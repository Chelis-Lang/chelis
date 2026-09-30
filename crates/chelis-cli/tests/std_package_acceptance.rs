use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

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
fn reef_std_time_and_decimal_modules_check_then_reject_eval() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-time-decimal");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Time (
  Date,
  Duration,
  add_days,
  date,
  date_gte,
  date_lt,
  date_to_string,
  day_of_year,
  day_of_week_name,
  days_between,
  duration,
  is_leap_year,
  parse_date
)
import Std.Decimal (
  decimal,
  decimal_add,
  decimal_div,
  decimal_eq,
  decimal_from_int,
  decimal_to_string,
  round_half_even,
  try_decimal
)

next_day = date_to_string(add_days(date(cast(2024, i64), cast(2, i64), cast(28, i64)), cast(1, i64)))
weekday = day_of_week_name(date(cast(2024, i64), cast(2, i64), cast(26, i64)))
ordinal = day_of_year(date(cast(2024, i64), cast(12, i64), cast(31, i64)))
leap = is_leap_year(cast(2024, i64))
span = duration(cast(1, i64), cast(2, i64), cast(3, i64), cast(4, i64))
parsed_ok = match parse_date("2024-12-31") with {
  | Some(value) => date_to_string(value)
  | None => "invalid"
}
parsed = match parse_date("2024-02-30") with {
  | Some(value) => date_to_string(value)
  | None => "invalid"
}
cross_year_days = days_between(
  date(cast(2024, i64), cast(12, i64), cast(31, i64)),
  date(cast(2025, i64), cast(1, i64), cast(2, i64))
)
cross_year_lt = date_lt(
  date(cast(2024, i64), cast(12, i64), cast(31, i64)),
  date(cast(2025, i64), cast(1, i64), cast(2, i64))
)
cross_year_gte = date_gte(
  date(cast(2025, i64), cast(1, i64), cast(2, i64)),
  date(cast(2024, i64), cast(12, i64), cast(31, i64))
)
exact = decimal_eq(decimal_add(decimal("0.1"), decimal("0.2")), decimal("0.3"))
banker = decimal_to_string(
  decimal_div(decimal_from_int(cast(5, i64)), decimal_from_int(cast(2, i64)), cast(0, i64), round_half_even())
)
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
