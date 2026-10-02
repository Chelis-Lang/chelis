use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, gcc_available, make_app, write_file};

/// A program over every `Std.Decimal` callable whose every root is a
/// canonical value ([05-OP-76]).
const DECIMAL_ACCEPTANCE_PROGRAM: &str = r#"module Demo.Main
import Std.Decimal (Decimal, decimal, try_decimal, decimal_to_string, decimal_to_fixed_string, decimal_from_i64, decimal_to_i64, try_decimal_to_i64, decimal_from_f64, try_decimal_from_f64, decimal_to_f64, decimal_to_f32, decimal_scale, decimal_add, decimal_sub, decimal_mul, decimal_round, decimal_div, try_decimal_div, decimal_lt, decimal_lte, decimal_gt, decimal_gte)
import Std.Rounding (Rounding, RoundTowardZero, RoundTiesToEven, RejectInexact)
def shown(value: Option[Decimal]) -> string =
  match value with {
    | Some(d) => decimal_to_string(d)
    | None => "none"
  }
def shown_int(value: Option[i64]) -> string =
  match value with {
    | Some(v) => to_string(v)
    | None => "none"
  }
parsed = decimal_to_string(decimal("9223372036854775807.0"))
tried = shown(try_decimal("0.10"))
rejected = shown(try_decimal("+1"))
fixed = decimal_to_fixed_string(decimal("1.5"), 2i64)
from_int = decimal_to_string(decimal_from_i64(sub(-9223372036854775807i64, 1i64)))
narrowed = decimal_to_i64(decimal("2.5"), RoundTiesToEven)
try_narrowed = shown_int(try_decimal_to_i64(decimal("1e30"), RoundTowardZero))
ingested = decimal_to_string(decimal_from_f64(0.1f64, 2i64, RoundTiesToEven))
try_ingested = shown(try_decimal_from_f64(0.1f64, 2i64, RejectInexact))
double = decimal_to_f64(decimal("0.1"))
single = decimal_to_f32(decimal("0.1"))
scale = decimal_scale(decimal("1.50"))
total = decimal_to_string(decimal_add(decimal("0.1"), decimal("0.2")))
difference = decimal_to_string(decimal_sub(decimal("1.00"), decimal("0.90")))
product = decimal_to_string(decimal_mul(decimal("1.1"), decimal("1.1")))
rounded = decimal_to_string(decimal_round(decimal("2.675"), 2i64, RoundTiesToEven))
quotient = decimal_to_string(decimal_div(decimal("5"), decimal("2"), 0i64, RoundTiesToEven))
try_quotient = shown(try_decimal_div(decimal("1"), decimal("0"), 2i64, RoundTiesToEven))
below = decimal_lt(decimal("-0.5"), decimal("0"))
at_most = decimal_lte(decimal("1.50"), decimal("1.5"))
above = decimal_gt(decimal("1e-38"), decimal("0"))
at_least = decimal_gte(decimal("1"), decimal("1.01"))
"#;

/// chelis#2778: every `Std.Decimal` callable type-checks and evaluates
/// through the bundled standard library to its exact value.
#[test]
fn reef_std_decimal_callables_evaluate_exactly() {
    let (_dir, reef_home, app_pkg) = make_app("decimal-acceptance-2778");
    let path = app_pkg.join("src/main.ch");
    write_file(&path, DECIMAL_ACCEPTANCE_PROGRAM);
    let mut eval = Command::cargo_bin("chelis").expect("binary");
    let mut assertion = eval
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success();
    for line in [
        "parsed = 9223372036854775807",
        "tried = 0.1",
        "rejected = none",
        "fixed = 1.50",
        "from_int = -9223372036854775808",
        "narrowed = 2",
        "try_narrowed = none",
        "ingested = 0.1",
        "try_ingested = none",
        "double = 0.1",
        "single = 0.1",
        "scale = 1",
        "total = 0.3",
        "difference = 0.1",
        "product = 1.21",
        "rounded = 2.68",
        "quotient = 2",
        "try_quotient = none",
        "below = true",
        "at_most = true",
        "above = true",
        "at_least = false",
    ] {
        assertion = assertion.stdout(predicate::str::contains(format!("{line}\n")));
    }
    let _ = assertion;
}

#[test]
#[ignore = "manual gate: Phase 3i std package acceptance suite exceeds the default inner-loop budget"]
fn reef_std_decimal_module_checks_then_evaluates() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-decimal");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main
import Std.Decimal (decimal, decimal_add, decimal_div, decimal_from_i64, decimal_to_string, try_decimal)
import Std.Rounding (RoundTiesToEven)
exact = eq(decimal_to_string(decimal_add(decimal("0.1"), decimal("0.2"))), "0.3")
banker = decimal_to_string(decimal_div(decimal_from_i64(5i64), decimal_from_i64(2i64), 0i64, RoundTiesToEven))
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
