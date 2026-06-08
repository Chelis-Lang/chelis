use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

#[test]
#[ignore = "manual gate: Phase 3i std package acceptance suite exceeds the default inner-loop budget"]
fn reef_std_time_and_decimal_modules_eval() {
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

next_day = date_to_string(add_days(date(cast(2024, int64), cast(2, int64), cast(28, int64)), cast(1, int64)))
weekday = day_of_week_name(date(cast(2024, int64), cast(2, int64), cast(26, int64)))
ordinal = day_of_year(date(cast(2024, int64), cast(12, int64), cast(31, int64)))
leap = is_leap_year(cast(2024, int64))
span = duration(cast(1, int64), cast(2, int64), cast(3, int64), cast(4, int64))
parsed_ok = match parse_date("2024-12-31") with {
  | Some(value) => date_to_string(value)
  | None => "invalid"
}
parsed = match parse_date("2024-02-30") with {
  | Some(value) => date_to_string(value)
  | None => "invalid"
}
cross_year_days = days_between(
  date(cast(2024, int64), cast(12, int64), cast(31, int64)),
  date(cast(2025, int64), cast(1, int64), cast(2, int64))
)
cross_year_lt = date_lt(
  date(cast(2024, int64), cast(12, int64), cast(31, int64)),
  date(cast(2025, int64), cast(1, int64), cast(2, int64))
)
cross_year_gte = date_gte(
  date(cast(2025, int64), cast(1, int64), cast(2, int64)),
  date(cast(2024, int64), cast(12, int64), cast(31, int64))
)
exact = decimal_eq(decimal_add(decimal("0.1"), decimal("0.2")), decimal("0.3"))
banker = decimal_to_string(
  decimal_div(decimal_from_int(cast(5, int64)), decimal_from_int(cast(2, int64)), cast(0, int64), round_half_even())
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
        .success()
        .stdout(predicate::str::contains("next_day = 2024-02-29"))
        .stdout(predicate::str::contains("weekday = monday"))
        .stdout(predicate::str::contains("ordinal = 366"))
        .stdout(predicate::str::contains("leap = true"))
        .stdout(predicate::str::contains("span = Duration(1, 2, 3, 4)"))
        .stdout(predicate::str::contains("parsed_ok = 2024-12-31"))
        .stdout(predicate::str::contains("parsed = invalid"))
        .stdout(predicate::str::contains("cross_year_days = 2"))
        .stdout(predicate::str::contains("cross_year_lt = true"))
        .stdout(predicate::str::contains("cross_year_gte = true"))
        .stdout(predicate::str::contains("exact = true"))
        .stdout(predicate::str::contains("banker = 2"))
        .stdout(predicate::str::contains("bad_decimal = invalid-decimal"));
}

#[test]
#[ignore = "manual gate: Phase 3i std package acceptance suite exceeds the default inner-loop budget"]
fn reef_package_mode_preserves_split_map_and_runtime_reshape_typing() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-package-typing");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

source = pad_sequences_to([[1.0, 2.0], [3.0, 4.0]], cast(2, int64), 0.0)
rows = split(source, cast(0, int32), [cast(1, int64), cast(1, int64)])
flat_rows = map(
  fn (row: tensor[piece, seq, f32]) -> reshape(row, [cast(2, int64)]),
  rows
)
first = index(flat_rows, cast(0, int64))
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
