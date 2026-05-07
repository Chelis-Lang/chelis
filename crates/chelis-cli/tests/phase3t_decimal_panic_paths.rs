//! Phase 3t.A1 follow-up — Std.Decimal panic-path coverage (RT-A1W1 MEDIUM).
//!
//! `Std.Decimal` exposes two panic paths that cannot be covered from inside
//! `chelis test` because `fail` aborts the worker before any subsequent
//! assertion can run:
//!
//!   * `decimal("garbage")` — `decimal/1` calls `fail` for malformed input
//!     because the Option-returning variant is `try_decimal/1`. The non-try
//!     entry is the panicking convenience.
//!   * `decimal_div(_, decimal_from_int(0), _, _)` — `decimal_div/4` calls
//!     `fail("decimal_div: division by zero")` for a zero denominator.
//!     There is intentionally no `try_decimal_div`.
//!
//! Both paths are exercised here through `chelis eval --file`. Each test
//! stages chelis-std into a tempdir reef home, writes a `main.ch` whose
//! module-load triggers the panic path, and asserts:
//!
//!   * exit code != 0
//!   * stderr contains the branded fail message
//!
//! The pattern mirrors `phase3t_test_std.rs::assert_eval_fails_with`.

use assert_cmd::Command;
use predicates::prelude::*;
use std::path::Path;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

fn assert_eval_fails_with(reef_home: &Path, app_pkg: &Path, stderr_contains: &[&str]) {
    let mut cmd = Command::cargo_bin("chelis").expect("binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ]);
    let mut assertion = cmd.assert().failure();
    for fragment in stderr_contains {
        assertion = assertion.stderr(predicate::str::contains(*fragment));
    }
    let _ = assertion;
}

#[test]
#[ignore = "manual gate: Std.Decimal failure-path CLI acceptance exceeds the default inner-loop budget"]
fn decimal_of_garbage_string_calls_fail_with_branded_message() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-decimal-panic-garbage");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Decimal (decimal)

bad = decimal("not a number")
"#,
    );
    // The fail in Std.Decimal.decimal/1 reads:
    //   string_concat("decimal: invalid literal ", text)
    // so the branded prefix and the offending text both appear on stderr.
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["decimal: invalid literal", "not a number"],
    );
}

#[test]
#[ignore = "manual gate: Std.Decimal failure-path CLI acceptance exceeds the default inner-loop budget"]
fn decimal_div_by_zero_calls_fail_with_branded_message() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-decimal-panic-divzero");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Decimal (decimal_div, decimal_from_int, round_half_even)

quotient = decimal_div(decimal_from_int(cast(1, int64)), decimal_from_int(cast(0, int64)), cast(0, int64), round_half_even())
"#,
    );
    // Std.Decimal.decimal_div/4 fails with the literal:
    //   "decimal_div: division by zero"
    assert_eval_fails_with(&reef_home, &app_pkg, &["decimal_div: division by zero"]);
}
