//! chelis#709 - `with seed` / `with device` bodies are not type-checked
//! (no `handle-effect` case in infer.rs), plus the two executed escalations
//! from the 2026-07-16 sweep and the #721 round-trip bug found alongside.
//!
//! ## The scoping, confirmed by execution
//!
//! A wrapper battery (locked below) shows the hole is EXACTLY `with seed` /
//! `with device` bodies: the same ill-typed expression is caught inside
//! let/if/match/lambda/pipe/tuple/list/grad/vmap/jit, and the seed/device
//! HANDLER expressions are checked too. #709's blast-radius claim holds.
//!
//! ## The escalations
//!
//! * The masked error reaches a RUNNABLE binary: `with seed(42) { cast(5,
//!   int64) }` from an `-> f32` fn passes check (score 1), builds, runs,
//!   and prints `5` in BOTH lanes. The DAG-gate rejection #709 called
//!   incidental does not fire for host-lane bodies.
//! * `lower_handle_effect`'s catch-all is live TODAY via `.dp`: rewriting
//!   `effect: random` to a bogus kind still checks at score 1 and builds a
//!   working binary - the unknown kind and its handler silently dropped
//!   (the #703 shape).
//!
//! ## chelis#721 (found while probing the controls)
//!
//! `chelis eval` cannot ingest the canonical Deep of a NULLARY fn:
//! `(fn {} (params {}) body)` evaluates to its body value, so
//! `surf -> deep -> eval` fails with `value is not callable` on the
//! simplest constant-returning def, while check scores 1 and the compiled
//! lane runs the same file. Unary defs round-trip fine (locked).

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const MASKED_ERROR: &str = "add(cast(1.0, f32), cast(2, int64))";

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `chelis check` score for a program written to a file with the given
/// extension (".ch" or ".dp").
fn check_score(program: &str, ext: &str) -> f64 {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("p{ext}"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("check must emit JSON: {e}"));
    parsed["score"].as_f64().expect("numeric score")
}

/// `chelis deep` output for a surf program.
fn deep_of(program: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", path.to_str().unwrap()])
        .output()
        .expect("chelis deep should run");
    assert!(out.status.success(), "chelis deep must succeed");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Run `chelis eval --file` on a program with the given extension.
fn eval_with_ext(program: &str, ext: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("p{ext}"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Build to C and (if it builds) link + run. Distinguishes build rejection
/// from a runnable binary, which is what the escalation rows assert on.
enum CLane {
    Rejected(String),
    Ran(String),
}

fn c_lane_outcome(program: &str, ext: &str, name: &str) -> CLane {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}{ext}"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    if !built.status.success() {
        return CLane::Rejected(String::from_utf8_lossy(&built.stderr).into_owned());
    }
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return CLane::Rejected(format!("emitted C failed to compile: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    CLane::Ran(String::from_utf8_lossy(&run.stdout).into_owned())
}

// ===========================================================================
// CONTROLS: the hole is exactly with seed / with device. Everything else
// catches the error, including the handler expressions.
// ===========================================================================

/// Every non-handle-effect wrapper catches the masked error. If any row here
/// starts scoring 1, a new #709-class hole has opened.
#[test]
fn wrapper_constructs_catch_the_masked_error() {
    let cases: &[(&str, String)] = &[
        ("bare", format!("def f() -> f32 = {MASKED_ERROR}\n")),
        (
            "let_body",
            format!("def f() -> f32 = let x = {MASKED_ERROR} in x\n"),
        ),
        (
            "if_then",
            format!("def f(c: bool) -> f32 = if c then {MASKED_ERROR} else 1.0\n"),
        ),
        (
            "lambda_body",
            format!("def f() -> f32 = (fn (v: f32) -> f32 = {MASKED_ERROR})(1.0)\n"),
        ),
        (
            "pipe_stage",
            format!("def f() -> f32 = 1.0 |> fn (v: f32) -> f32 = {MASKED_ERROR}\n"),
        ),
        (
            "tuple_elem",
            format!("def f() -> (f32, f32) = ({MASKED_ERROR}, 1.0)\n"),
        ),
        (
            "list_elem",
            format!("def f() -> [f32] = [{MASKED_ERROR}, 1.0]\n"),
        ),
        (
            "match_arm",
            format!("def f(c: bool) -> f32 = match c {{ true => {MASKED_ERROR}, false => 1.0 }}\n"),
        ),
        (
            "grad_callee",
            format!("def g(x: f32) -> f32 = {MASKED_ERROR}\ndef f(x: f32) -> f32 = grad(g)(x)\n"),
        ),
        (
            "vmap_lambda",
            format!(
                "def f(t: tensor[4, f32]) -> tensor[4, f32] = \
                 vmap(fn (v: f32) -> f32 = {MASKED_ERROR})(t)\n"
            ),
        ),
        (
            "jit_callee",
            format!("def g(x: f32) -> f32 = {MASKED_ERROR}\ndef f(x: f32) -> f32 = jit(g)(x)\n"),
        ),
    ];
    for (name, program) in cases {
        let score = check_score(program, ".ch");
        assert!(
            score < 1.0,
            "{name}: the masked error must be caught (score < 1), got score {score}"
        );
    }
}

/// The HANDLER expressions of with seed / with device ARE checked - the
/// hole is only the body. Bounds #709 from the other side.
#[test]
fn handler_expressions_are_checked() {
    let score = check_score(
        "def f() -> f32 = with seed(\"not a seed\") { 1.0 }\n",
        ".ch",
    );
    assert!(score < 1.0, "a string seed must be rejected, got {score}");
    let score = check_score("def f() -> f32 = with device(42) { 1.0 }\n", ".ch");
    assert!(score < 1.0, "an int device must be rejected, got {score}");
}

/// chelis#721 control: a UNARY def's canonical Deep round-trips through eval.
#[test]
fn unary_fn_deep_roundtrips_through_eval() {
    let dp = deep_of("def f(x: f32) -> f32 = add(x, 1.5)\nout = print(f(1.0))\n");
    let stdout = eval_with_ext(&dp, ".dp").expect("unary deep must evaluate");
    assert!(stdout.contains("2.5"), "got: {stdout}");
}

// ===========================================================================
// chelis#709 - the holes
// ===========================================================================

/// Observed today: score 1 with the error inside `with seed`.
#[test]
#[ignore = "chelis#709: infer.rs has no handle-effect case; the with seed body is not \
            checked and this program scores 1. Run with \
            `cargo test -p chelis-cli --test issue_709_handle_effect_and_dp_roundtrip -- --ignored`."]
fn with_seed_body_is_type_checked() {
    let score = check_score(
        &format!("def f() -> f32 = with seed(42) {{ {MASKED_ERROR} }}\n"),
        ".ch",
    );
    assert!(
        score < 1.0,
        "the ill-typed with seed body must be caught, got score {score}"
    );
}

/// Observed today: score 1 with the error inside `with device`.
#[test]
#[ignore = "chelis#709: the with device body is not checked; scores 1 today. Run with \
            `cargo test -p chelis-cli --test issue_709_handle_effect_and_dp_roundtrip -- --ignored`."]
fn with_device_body_is_type_checked() {
    let score = check_score(
        &format!("def f() -> f32 = with device(\"gpu:0\") {{ {MASKED_ERROR} }}\n"),
        ".ch",
    );
    assert!(
        score < 1.0,
        "the ill-typed with device body must be caught, got score {score}"
    );
}

/// Observed today: check scores 1, the build succeeds, and the binary runs,
/// printing an int64 `5` from a function declared `-> f32`. The escalation
/// row: the mask reaches a runnable binary through the host lane.
#[test]
#[ignore = "chelis#709: `with seed(42) { cast(5, int64) }` from an -> f32 fn compiles and \
            runs, printing 5 in both lanes; the build must reject it with a type \
            diagnostic. Run with \
            `cargo test -p chelis-cli --test issue_709_handle_effect_and_dp_roundtrip -- --ignored`."]
fn masked_return_type_violation_does_not_reach_a_binary() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "def f() -> f32 = with seed(42) { cast(5, int64) }\nout = print(f())\n";
    match c_lane_outcome(program, ".ch", "seed_masked") {
        CLane::Rejected(stderr) => assert!(
            stderr.contains("Type errors") || stderr.contains("type"),
            "the rejection must be a type diagnostic, got: {stderr}"
        ),
        CLane::Ran(stdout) => panic!(
            "an ill-typed program must not reach a runnable binary; it ran and printed: {stdout}"
        ),
    }
}

/// Observed today: a `.dp` whose handle-effect carries a BOGUS effect kind
/// checks at score 1 and builds a working binary; `lower_handle_effect`'s
/// catch-all lowers the body and silently drops the unknown kind and its
/// handler (the #703 shape, live today via the .dp path).
#[test]
#[ignore = "chelis#709: a handle-effect with effect kind `teleport` builds and runs; an \
            unknown effect kind must be rejected loudly. Run with \
            `cargo test -p chelis-cli --test issue_709_handle_effect_and_dp_roundtrip -- --ignored`."]
fn unknown_effect_kind_is_rejected() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let dp = deep_of("def f() -> f32 = with seed(42) { 2.5 }\nout = print(f())\n")
        .replace("effect: random", "effect: teleport");
    assert!(
        dp.contains("effect: teleport"),
        "probe fixture must rewrite"
    );
    match c_lane_outcome(&dp, ".dp", "bogus_effect") {
        CLane::Rejected(stderr) => assert!(
            !stderr.is_empty(),
            "rejection must carry a diagnostic naming the unknown effect kind"
        ),
        CLane::Ran(stdout) => panic!(
            "an unknown effect kind must not silently compile to a working \
             binary; it ran and printed: {stdout}"
        ),
    }
}

// ===========================================================================
// chelis#721 - nullary defs cannot round-trip surf -> deep -> eval
// ===========================================================================

/// Observed today: `error: value is not callable: Tensor(... 2.5 ...)` -
/// eval binds the nullary fn to its body VALUE, then `(app (var f))` calls
/// the scalar. check scores the same file 1 and the compiled lane runs it.
#[test]
#[ignore = "chelis#721: eval cannot ingest the canonical Deep of a nullary fn (`value is \
            not callable`); the same .dp checks at 1 and runs compiled. Run with \
            `cargo test -p chelis-cli --test issue_709_handle_effect_and_dp_roundtrip -- --ignored`."]
fn nullary_fn_deep_roundtrips_through_eval() {
    let dp = deep_of("def f() -> f32 = 2.5\nout = print(f())\n");
    let stdout = eval_with_ext(&dp, ".dp")
        .expect("chelis#721: the canonical Deep of a nullary fn must evaluate");
    assert!(stdout.contains("2.5"), "got: {stdout}");
}
