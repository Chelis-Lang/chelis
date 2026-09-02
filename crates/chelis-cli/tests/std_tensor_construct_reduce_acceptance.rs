//! Phase 3j-pre Batch 2: Std.Tensor.Construct and Std.Tensor.Reduce
//! acceptance tests.
//!
//! Positive tests use hand-computed reference values and require exact
//! equality against the `eval` printer. Each positive test is paired
//! with a negative test that exercises an obvious failure mode.
//!
//! KNOWN RESIDUAL (chelis#1416): the public rank-polymorphic signatures for
//! `stack`, `squeeze`, and `unsqueeze` are outside the type system's decidable
//! fragment for concrete callers. The import-only test below keeps the exports
//! visible without claiming that concrete calls execute; `linspace` and
//! `arange` retain executable positive/negative coverage here.

use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

#[test]
#[ignore = "manual gate: Phase 3j-pre batch acceptance suite exceeds the default inner-loop budget"]
fn phase3j_pre_batch2_linspace_matches_reference_values() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-linspace");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Construct (linspace)

ls_5 = linspace(cast(0.0, f32), cast(1.0, f32), cast(5, int64))
ls_3 = linspace(cast(-1.0, f32), cast(1.0, f32), cast(3, int64))
ls_1 = linspace(cast(4.0, f32), cast(9.0, f32), cast(1, int64))
"#,
    );

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
            "ls_5 = tensor(shape=[5], data=[0.0, 0.25, 0.5, 0.75, 1.0])",
        ))
        .stdout(predicate::str::contains(
            "ls_3 = tensor(shape=[3], data=[-1.0, 0.0, 1.0])",
        ))
        // The valid count=1 boundary is a single-element tensor of `start`.
        .stdout(predicate::str::contains(
            "ls_1 = tensor(shape=[1], data=[4.0])",
        ));
}

#[test]
#[ignore = "manual gate: Phase 3j-pre batch acceptance suite exceeds the default inner-loop budget"]
fn phase3j_pre_batch2_linspace_rejects_non_scalar_start() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-linspace-bad");
    // Passing a tensor where a scalar f32 is required should fail check.
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Construct (linspace)

bad = linspace(to_tensor([cast(0.0, f32)]), cast(1.0, f32), cast(5, int64))
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        // Issue #207: type errors produce exit 2; assert on stdout content only.
        // Score must not be perfect when a scalar-vs-tensor mismatch is present.
        .stdout(predicate::str::contains("\"score\": 1").not());
}

#[test]
#[ignore = "manual gate: Phase 3j-pre batch acceptance suite exceeds the default inner-loop budget"]
fn phase3j_pre_batch2_linspace_rejects_count_below_one() {
    // chelis#1422. [05-OP-35]: "`linspace` requires finite endpoints and int64
    // `count >= 1`; count one returns `[start]`". Count zero is a runtime
    // Domain failure, not a value: the pre-fix `count <= 1` branch returned
    // `[start]` for zero and for every negative count.
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-linspace-count0");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Construct (linspace)

bad = linspace(cast(0.0, f32), cast(1.0, f32), cast(0, int64))
"#,
    );

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
        .stderr(predicate::str::contains("count must be at least 1"));
}

#[test]
#[ignore = "manual gate: Phase 3j-pre batch acceptance suite exceeds the default inner-loop budget"]
fn phase3j_pre_batch2_arange_matches_reference_values() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-arange");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Construct (arange)

ar_0_4 = arange(cast(0, int32), cast(4, int32))
ar_2_6 = arange(cast(2, int32), cast(6, int32))
"#,
    );

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
            "ar_0_4 = tensor(shape=[4], data=[0, 1, 2, 3])",
        ))
        .stdout(predicate::str::contains(
            "ar_2_6 = tensor(shape=[4], data=[2, 3, 4, 5])",
        ));
}

#[test]
#[ignore = "manual gate: Phase 3j-pre batch acceptance suite exceeds the default inner-loop budget"]
fn phase3j_pre_batch2_arange_rejects_mixed_bound_dtypes() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-arange-bad");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Construct (arange)

bad = arange(cast(0, int16), cast(4, int64))
"#,
    );

    // Repeated p_int positions must actualize to one dtype. chelis#1417 owns
    // the separate defect that the authored p_int domain still admits floats.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        // Issue #207: type errors produce exit 2; assert on stdout content only.
        .stdout(predicate::str::contains("\"score\": 1").not());
}

// chelis#333: the three `phase3j_pre_batch2_reduce_*` acceptance tests
// (min/prod, argmax/argmin, and the scalar-input rejection) were removed
// with the Std.Tensor.Reduce module. The four functions were bodyless sigs
// taking a runtime int32 axis that could not forward to the const-axis
// `*_reduce` builtins, so they never had a runtime implementation and these
// import-and-eval tests could not have passed. Reductions are exercised
// directly through the `*_reduce` builtins with a compile-time-constant axis
// (e.g. `min_reduce(x, cast(1, int32))`).

#[test]
#[ignore = "manual gate: Phase 3j-pre batch acceptance suite exceeds the default inner-loop budget"]
fn phase3j_pre_batch2_stack_squeeze_unsqueeze_publish_successfully() {
    // chelis#1416: the exact published signatures conflict with spec/04's
    // unitary rank-spread rules, so concrete calls reject before evaluation.
    // Until the numbered specs and implementation agree on a representable
    // contract, this test pins only publication and downstream importability.
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-stack-publish");
    // make_app already publishes; if it returned, publishing succeeded
    // and the wrappers type-checked cleanly enough to be shipped. This
    // second import-only check asserts the three exports are actually
    // importable from a downstream consumer, even though we cannot yet
    // evaluate them end-to-end due to the residuals documented above.
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Construct (stack, squeeze, unsqueeze)

touch = stack
touch2 = squeeze
touch3 = unsqueeze
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
}
