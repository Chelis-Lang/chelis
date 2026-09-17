//! chelis#2151: `cast` and `cast_trunc` from a SCALAR source typed by a
//! dtype-family-bounded binder.
//!
//! [05-OP-6] says `cast_trunc`'s semantics are identical on the scalar and
//! tensor surfaces, and [05-OP-63] gives `cast` the same scalar-or-tensor
//! shape. The tensor form over a bounded binder (`tensor[n, p]`) was already
//! accepted (#1564). The scalar form (`u: p`) was not: the checker suspended
//! the decision on the source variable (#1489's design for a variable that
//! will be bound later), but an authored binder is rigid and never is. The
//! pending gate then surfaced as the misleading
//! "cast requires tensor or prim type, got `p`".
//!
//! [04-DTYPE-2] restricts a family-bounded binder to active primitives. The
//! source is therefore always a scalar, the result is always the named
//! primitive, and `cast_trunc`'s float-source requirement follows from the
//! bound alone. Nothing is left to wait for.
//!
//! Negative parity: each rejection must name its real [05-OP-6] reason, not
//! the generic one. Every rejection below failed before the fix too, but with
//! that generic message, which is why the diagnostic is asserted. An
//! unbounded binder admits non-dtypes ([04-DTYPE-2]), so it stays rejected.
use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn run(command: &str, source: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("cast.ch");
    fs::write(&path, source).expect("source");
    let mut cli = Command::cargo_bin("chelis").expect("chelis");
    cli.env("CHELIS_STYLE_GATE_DISABLE", "1").arg(command);
    if command == "eval" {
        cli.arg("--file");
    }
    cli.arg(&path);
    if command == "build" {
        cli.args(["--target", "c", "--output"])
            .arg(dir.path().join("out"));
    }
    cli.output().expect(command)
}

fn program(body: &str) -> String {
    format!("module Bounded.Main\nexport (main)\n{body}\n")
}

/// Evaluate, build, link and run the same program; both lanes must agree
/// and produce `expected`.
fn assert_lanes_agree(body: &str, name: &str, expected: &[f64]) {
    let source = program(body);
    let interpreted = run("eval", &source);
    assert!(
        interpreted.status.success(),
        "eval rejected {name}: {}",
        String::from_utf8_lossy(&interpreted.stderr)
    );
    let interpreted = String::from_utf8(interpreted.stdout).expect("UTF-8");
    assert_eq!(
        common::parse_tensor_data(&interpreted, "main"),
        expected,
        "{name}"
    );
    let native = common::build_and_run(&source, name);
    assert_eq!(native.trim(), interpreted.trim(), "{name}: eval vs C");
}

#[test]
fn float_bounded_scalar_truncates_at_f32_and_f64() {
    assert_lanes_agree(
        "def t[p: Float](u: p) -> i64 = cast_trunc(u, i64)\n\
         def main() -> tensor[2, i64] = to_tensor([t(7.999999999f64), t(-2.5f64)])",
        "trunc_f64",
        &[7.0, -2.0],
    );
    assert_lanes_agree(
        "def t[p: Float](u: p) -> i64 = cast_trunc(u, i64)\n\
         def main() -> tensor[2, i64] = to_tensor([t(3.75f32), t(-0.5f32)])",
        "trunc_f32",
        &[3.0, 0.0],
    );
}

#[test]
fn one_program_truncates_distinct_instantiations_independently() {
    // 16777217 is exact in f64 but not in f32. If the two calls wrongly
    // shared one f32 instantiation, the f64 result could not be 16777217.
    assert_lanes_agree(
        "def t[p: Float](u: p) -> i64 = cast_trunc(u, i64)\n\
         def main() -> tensor[2, i64] = to_tensor([t(16777217.0f64), t(16777216.0f32)])",
        "trunc_distinct",
        &[16777217.0, 16777216.0],
    );
}

#[test]
fn bounded_scalar_casts_to_a_concrete_dtype() {
    assert_lanes_agree(
        "def w[p: Float](u: p) -> f64 = cast(u, f64)\n\
         def main() -> tensor[2, f64] = to_tensor([w(0.5f32), w(0.25f64)])",
        "float_to_f64",
        &[0.5, 0.25],
    );
    assert_lanes_agree(
        "def g[p: Int](k: p) -> f64 = cast(k, f64)\n\
         def main() -> tensor[2, f64] = to_tensor([g(7i32), g(-3i64)])",
        "int_to_f64",
        &[7.0, -3.0],
    );
    assert_lanes_agree(
        "def g[p: Numeric](k: p) -> f64 = cast(k, f64)\n\
         def main() -> tensor[2, f64] = to_tensor([g(7i32), g(2.5f32)])",
        "numeric_to_f64",
        &[7.0, 2.5],
    );
}

#[test]
fn a_bounded_scalar_computed_in_the_body_casts_too() {
    // The source is not the parameter itself but a value derived from it, so
    // the fix cannot be keyed on parameter identity.
    assert_lanes_agree(
        "def t[p: Float](u: p) -> i64 = cast_trunc(add(u, u), i64)\n\
         def main() -> tensor[1, i64] = to_tensor([t(1.75f64)])",
        "trunc_derived",
        &[3.0],
    );
}

#[test]
fn invalid_bounded_scalar_casts_name_their_real_reason_in_every_lane() {
    for (body, expected) in [
        (
            "def t[p: Int](u: p) -> i64 = cast_trunc(u, i64)\n\
             def main() -> tensor[1, i64] = to_tensor([t(7i32)])",
            "`cast_trunc` source",
        ),
        (
            "def t[p: Numeric](u: p) -> i64 = cast_trunc(u, i64)\n\
             def main() -> tensor[1, i64] = to_tensor([t(7.5f32)])",
            "`cast_trunc` source",
        ),
        (
            "def t[p: Float](u: p) -> f64 = cast_trunc(u, f64)\n\
             def main() -> tensor[1, f64] = to_tensor([t(7.5f32)])",
            "is not an integer dtype",
        ),
    ] {
        let source = program(body);
        for command in ["check", "eval", "build"] {
            let output = run(command, &source);
            let diagnostic = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            if command == "check" {
                let report: serde_json::Value =
                    serde_json::from_str(&diagnostic[diagnostic.find('{').expect("check JSON")..])
                        .expect("check JSON");
                assert!(
                    report["errors"].as_array().is_some_and(|e| !e.is_empty()),
                    "check accepted {body}"
                );
            } else {
                assert!(!output.status.success(), "{command} accepted {body}");
            }
            assert!(
                diagnostic.contains(expected),
                "{command} on {body}: expected {expected:?} in {diagnostic}"
            );
            assert!(
                !diagnostic.contains("cast requires tensor or prim type"),
                "{command} on {body} still reports the generic pending-gate message: {diagnostic}"
            );
        }
    }
}

#[test]
fn an_unbounded_scalar_binder_source_stays_rejected() {
    // [04-DTYPE-2]: an unbounded binder admits non-dtypes, so the scalar
    // result is not determined by the binder and the rejection is correct.
    let source = program(
        "def g[p](k: p) -> f64 = cast(k, f64)\n\
         def main() -> tensor[1, f64] = to_tensor([g(7.5f32)])",
    );
    for command in ["eval", "build"] {
        let output = run(command, &source);
        assert!(
            !output.status.success(),
            "{command} accepted an unbounded binder source"
        );
    }
}
