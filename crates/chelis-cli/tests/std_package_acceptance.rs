use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

#[test]
#[ignore = "manual gate: Phase 3i std package acceptance suite exceeds the default inner-loop budget"]
fn reef_std_decimal_module_eval() {
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
        .success()
        .stdout(predicate::str::contains("exact = true"))
        .stdout(predicate::str::contains("banker = 2"))
        .stdout(predicate::str::contains("bad_decimal = invalid-decimal"));
}

#[test]
fn reef_std_time_calls_raise_issue_2779_error() {
    let (_dir, reef_home, app_pkg) = make_app("time-fence-2779");
    let date = "Date { year: cast(1970, i64), month: cast(1, i64), day: cast(1, i64) }";
    let invalid = "Date { year: cast(1970, i64), month: cast(13, i64), day: cast(1, i64) }";
    let cases = [
        "date(cast(1970, i64), cast(1, i64), cast(1, i64))".to_owned(),
        "try_date(cast(1970, i64), cast(1, i64), cast(1, i64))".to_owned(),
        "duration(cast(0, i64), cast(0, i64), cast(0, i64), cast(-1, i64))".to_owned(),
        "is_leap_year(cast(2024, i64))".to_owned(),
        format!("add_days({date}, cast(1, i64))"),
        format!("sub_days({date}, cast(-9223372036854775807, i64))"),
        format!("days_between({date}, {date})"),
        format!("date_lt({date}, {date})"),
        format!("date_lte({date}, {date})"),
        format!("date_gt({date}, {date})"),
        format!("date_gte({date}, {date})"),
        format!("date_to_string({invalid})"),
        "parse_date(\"+10000-01-01\")".to_owned(),
        format!("day_of_week({date})"),
        format!("day_of_week_name({date})"),
        format!("day_of_year({invalid})"),
    ];
    for (index, expression) in cases.iter().enumerate() {
        let source = format!(
            "module Demo.Main\nimport Std.Time (Date, date, try_date, duration, is_leap_year, add_days, sub_days, days_between, date_lt, date_lte, date_gt, date_gte, date_to_string, parse_date, day_of_week, day_of_week_name, day_of_year)\nresult = {expression}\n"
        );
        let path = app_pkg.join("src/main.ch");
        write_file(&path, &source);
        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .current_dir(&app_pkg)
            .args(["eval", "--file", path.to_str().unwrap()])
            .output()
            .expect("eval output");
        assert!(
            !output.status.success(),
            "case {index} silently succeeded: {expression}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let rendered = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            rendered.contains("Std.Time is unavailable") && rendered.contains("#2779"),
            "case {index} ({expression}) failed without the Time fence: {rendered}"
        );
    }
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
