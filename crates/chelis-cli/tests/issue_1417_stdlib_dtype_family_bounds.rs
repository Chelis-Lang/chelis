//! chelis#1417: `Std.Tensor.Construct.arange` and `linspace` enforce the
//! `[05-OP-35]` dtype domains their signatures declare.
//!
//! These exercise the published `chelis-std` package surface, so they prove
//! the bound survives the shell round trip (`TypeVariableDomain`) as well as
//! isolated checking. Each reproducer from the issue has its positive mirror
//! at every dtype the family admits.
//!
//! Manual gate. Run with:
//!
//! ```text
//! cargo nextest run -p chelis-cli --test issue_1417_stdlib_dtype_family_bounds -- --ignored
//! ```
//!
//! Expected success condition: every test passes.

use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

const FLOATS: &[&str] = &["f16", "bf16", "f32", "f64"];
const INTS: &[&str] = &["i8", "i16", "i32", "i64"];

fn eval(slug: &str, source: &str) -> assert_cmd::assert::Assert {
    let (_dir, reef_home, app_pkg) = make_app(slug);
    write_file(&app_pkg.join("src/main.ch"), source);
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
}

// === The issue's exact reproducers, which used to be accepted ===

#[test]
#[ignore = "manual gate: published-package acceptance exceeds the default inner-loop budget"]
fn arange_rejects_float_endpoints() {
    eval(
        "issue-1417-arange-float",
        r#"module Demo.Main

import Std.Tensor.Construct (arange)

values = arange(cast(0.0, f32), cast(2.0, f32))
"#,
    )
    .failure()
    .stderr(predicate::str::contains("Int"))
    .stderr(predicate::str::contains("f32"));
}

#[test]
#[ignore = "manual gate: published-package acceptance exceeds the default inner-loop budget"]
fn linspace_rejects_integer_endpoints() {
    eval(
        "issue-1417-linspace-int",
        r#"module Demo.Main

import Std.Tensor.Construct (linspace)

values = linspace(cast(0, i32), cast(2, i32), cast(3, i64))
"#,
    )
    .failure()
    .stderr(predicate::str::contains("Float"))
    .stderr(predicate::str::contains("i32"));
}

// === Every admitted dtype still works ===

#[test]
#[ignore = "manual gate: published-package acceptance exceeds the default inner-loop budget"]
fn arange_accepts_every_active_signed_integer() {
    for dtype in INTS {
        eval(
            &format!("issue-1417-arange-{dtype}"),
            &format!(
                r#"module Demo.Main

import Std.Tensor.Construct (arange)

values = arange(cast(0, {dtype}), cast(3, {dtype}))
"#
            ),
        )
        .success()
        .stdout(predicate::str::contains(
            "values = tensor(shape=[3], data=[0, 1, 2])",
        ));
    }
}

#[test]
#[ignore = "manual gate: published-package acceptance exceeds the default inner-loop budget"]
fn linspace_accepts_every_active_float() {
    for dtype in FLOATS {
        eval(
            &format!("issue-1417-linspace-{dtype}"),
            &format!(
                r#"module Demo.Main

import Std.Tensor.Construct (linspace)

values = linspace(cast(0.0, {dtype}), cast(1.0, {dtype}), cast(3, i64))
"#
            ),
        )
        .success()
        .stdout(predicate::str::contains(
            "values = tensor(shape=[3], data=[0.0, 0.5, 1.0])",
        ));
    }
}

// === The bound is on the exported scheme, not on the callee name ===

#[test]
#[ignore = "manual gate: published-package acceptance exceeds the default inner-loop budget"]
fn an_imported_bound_survives_a_local_wrapper() {
    eval(
        "issue-1417-arange-wrapper",
        r#"module Demo.Main

import Std.Tensor.Construct (arange)

def wrap[p: Int](start: p, stop: p) -> tensor[n, p] = arange(start, stop)

values = wrap(cast(0.0, f64), cast(2.0, f64))
"#,
    )
    .failure()
    .stderr(predicate::str::contains("Int"));
}
