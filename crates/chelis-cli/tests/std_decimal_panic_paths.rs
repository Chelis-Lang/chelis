//! Retained invalid-input probes for the temporary Std.Decimal fence (#2778).
//! The complete callable matrix lives in `std_package_acceptance`.

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
fn decimal_of_garbage_string_reports_fence() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-decimal-panic-garbage");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Decimal (decimal)

bad = decimal("not a number")
"#,
    );
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["Std.Decimal is unavailable", "#2778"],
    );
}

#[test]
fn decimal_div_by_zero_reports_fence() {
    let (_dir, reef_home, app_pkg) = make_app("phase3t-decimal-panic-divzero");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Decimal (decimal_div, decimal_from_int, round_half_even)

quotient = decimal_div(decimal_from_int(cast(1, i64)), decimal_from_int(cast(0, i64)), cast(0, i64), round_half_even())
"#,
    );
    assert_eval_fails_with(
        &reef_home,
        &app_pkg,
        &["Std.Decimal is unavailable", "#2778"],
    );
}
