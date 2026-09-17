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
//! [04-DTYPE-2] restricts a family-bounded binder to active primitives, so the
//! source is always a scalar and the result is always the named primitive.
//!
//! The decision is made at the EVENT that makes it decidable, never on a
//! schedule (#1489). There are two such events:
//!
//! - the cast is inferred while its source already carries a declared bound;
//! - a variable with a suspended cast acquires a bound through a binding (it
//!   is identified with a bounded variable, or with a rigid binder).
//!
//! Whichever comes second settles the cast, so the verdict cannot depend on
//! which operand inference visits first. Only verdicts that no later binding
//! can change are taken there: an acceptance, or a rejection that reads only
//! the target. `cast_trunc` from an `Int` or `Numeric` bound would reject on
//! the source, so it stays suspended until the source binds.
//!
//! Negative parity: invalid casts are still rejected in every lane, whatever
//! the order. A target-only rejection names its real reason. A rigid
//! `Int`/`Numeric` `cast_trunc` source is still rejected, through the
//! suspended gate, with the same generic message as before this change. An
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

/// Run `check`, `eval` and `build` on `source`. Each must reject it, and the
/// combined output is returned for the caller's diagnostic assertions.
fn rejected_in_every_lane(source: &str, what: &str) -> Vec<(&'static str, String)> {
    ["check", "eval", "build"]
        .into_iter()
        .map(|command| {
            let output = run(command, source);
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
                    "check accepted {what}"
                );
            } else {
                assert!(!output.status.success(), "{command} accepted {what}");
            }
            (command, diagnostic)
        })
        .collect()
}

#[test]
fn target_only_invalid_bounded_scalar_casts_name_their_real_reason_in_every_lane() {
    // These read only the target, so they are decided at the cast and must
    // not fall back to the generic pending-gate message.
    for (body, expected) in [
        (
            "def t[p: Float](u: p) -> f64 = cast_trunc(u, f64)\n\
             def main() -> tensor[1, f64] = to_tensor([t(7.5f32)])",
            "is not an integer dtype",
        ),
        (
            "def t[p: Float](u: p) -> string = cast(u, string)\n\
             def main() -> string = t(1.5f32)",
            "cannot cast scalar to unsupported precision `string`",
        ),
    ] {
        for (command, diagnostic) in rejected_in_every_lane(&program(body), body) {
            assert!(
                diagnostic.contains(expected),
                "{command} on {body}: expected {expected:?} in {diagnostic}"
            );
            assert!(
                !diagnostic.contains("cast requires tensor or prim type"),
                "{command} on {body} reports the generic pending-gate message: {diagnostic}"
            );
        }
    }
}

#[test]
fn a_rigid_int_or_numeric_cast_trunc_source_is_still_rejected_in_every_lane() {
    // [05-OP-6]: `cast_trunc` needs a float source, and an `Int` or `Numeric`
    // bound admits an integer. The rejection stays with the suspended gate, so
    // only the rejection itself is asserted, not a new message.
    for body in [
        "def t[p: Int](u: p) -> i64 = cast_trunc(u, i64)\n\
         def main() -> tensor[1, i64] = to_tensor([t(7i32)])",
        "def t[p: Numeric](u: p) -> i64 = cast_trunc(u, i64)\n\
         def main() -> tensor[1, i64] = to_tensor([t(7.5f32)])",
    ] {
        rejected_in_every_lane(&program(body), body);
    }
}

const HIGHER_ORDER: &str = "def num_id[p: Numeric](u: p) -> p = u\n\
     def fid[p: Float](u: p) -> p = u\n\
     def apply_it[a, b](f: (a) -> b, x: a) -> b = f(x)\n";

#[test]
fn a_numeric_inference_variable_later_bound_to_a_float_truncates() {
    // The lambda parameter `x` is an inference variable. `num_id(x)` gives it a
    // `Numeric` bound before `apply_it` binds it to f32. Rejecting the
    // truncation at that point would refuse a valid f32 -> i64 truncation.
    assert_lanes_agree(
        &format!(
            "{HIGHER_ORDER}def main() -> tensor[1, i64] = \
             to_tensor([apply_it(fn (x) -> cast_trunc(num_id(x), i64), 2.5f32)])"
        ),
        "numeric_inference_later_float",
        &[2.0],
    );
}

#[test]
fn the_verdict_does_not_depend_on_which_operand_inference_visits_first() {
    // The same program with the two `add` operands swapped. One order bounds
    // `x` by `Numeric` first, the other by `Float`. Both must be accepted and
    // agree, or the verdict is following the inference schedule (#1489).
    for (name, operands) in [
        (
            "numeric_first",
            "cast_trunc(num_id(x), i64), cast_trunc(fid(x), i64)",
        ),
        (
            "float_first",
            "cast_trunc(fid(x), i64), cast_trunc(num_id(x), i64)",
        ),
    ] {
        assert_lanes_agree(
            &format!(
                "{HIGHER_ORDER}def main() -> tensor[1, i64] = \
                 to_tensor([apply_it(fn (x) -> add({operands}), 2.5f32)])"
            ),
            name,
            &[4.0],
        );
    }
}

#[test]
fn a_value_restricted_source_still_suspends_and_may_be_a_tensor() {
    // `sqrt(x)` gives `x` a float VALUE restriction, which admits a tensor.
    // Deciding it early as a scalar would reject this valid tensor cast.
    assert_lanes_agree(
        &format!(
            "{HIGHER_ORDER}def main() -> tensor[1, f64] = \
             apply_it(fn (x) -> cast(sqrt(x), f64), to_tensor([2.25f32]))"
        ),
        "value_restriction_tensor",
        &[1.5],
    );
}

#[test]
fn an_unbounded_scalar_binder_source_stays_rejected() {
    // [04-DTYPE-2]: an unbounded binder admits non-dtypes, so the scalar
    // result is not determined by the binder and the rejection is correct.
    let source = program(
        "def g[p](k: p) -> f64 = cast(k, f64)\n\
         def main() -> tensor[1, f64] = to_tensor([g(7.5f32)])",
    );
    rejected_in_every_lane(&source, "an unbounded binder source");
}

#[test]
fn a_rigid_float_binder_reached_through_an_inference_variable_decides_in_either_order() {
    // Round 2 of the #2154 review: `x` is an inference variable later
    // identified with the rigid `p`, which is never bound to a concrete type.
    // With `fid(x)` first the bound arrives before the cast; with the bare `x`
    // first it arrives after. Both orders must settle the same way.
    for (name, body) in [
        (
            "rigid_apply_bare_first",
            "apply_it(fn (x) -> add(cast_trunc(x, i64), cast_trunc(fid(x), i64)), u)",
        ),
        (
            "rigid_apply_fid_first",
            "apply_it(fn (x) -> add(cast_trunc(fid(x), i64), cast_trunc(x, i64)), u)",
        ),
        (
            "rigid_let_bare_first",
            "{\n  g = fn (x) -> add(cast_trunc(x, i64), cast_trunc(fid(x), i64))\n  g(u)\n}",
        ),
        (
            "rigid_let_fid_first",
            "{\n  g = fn (x) -> add(cast_trunc(fid(x), i64), cast_trunc(x, i64))\n  g(u)\n}",
        ),
    ] {
        assert_lanes_agree(
            &format!(
                "{HIGHER_ORDER}def t[p: Float](u: p) -> i64 = {body}\n\
                 def main() -> tensor[1, i64] = to_tensor([t(2.5f32)])"
            ),
            name,
            &[4.0],
        );
    }
}

#[test]
fn a_cast_and_a_truncation_sharing_a_rigid_bound_variable_agree_in_either_order() {
    for (name, operands) in [
        ("shared_cast_first", "cast(x, i64), cast_trunc(fid(x), i64)"),
        (
            "shared_trunc_first",
            "cast_trunc(fid(x), i64), cast(x, i64)",
        ),
    ] {
        assert_lanes_agree(
            &format!(
                "{HIGHER_ORDER}def t[p: Float](u: p) -> i64 = {{\n  g = fn (x) -> add({operands})\n  g(u)\n}}\n\
                 def main() -> tensor[1, i64] = to_tensor([t(2.0f32)])"
            ),
            name,
            &[4.0],
        );
    }
}

#[test]
fn a_bare_inference_variable_later_bound_to_a_rigid_float_binder_truncates() {
    // The cast is inferred before the variable meets `p` at all. `main` rejected
    // this; the binding that brings the bound now settles the cast.
    assert_lanes_agree(
        &format!(
            "{HIGHER_ORDER}def t[p: Float](u: p) -> i64 = apply_it(fn (x) -> cast_trunc(x, i64), u)\n\
             def main() -> tensor[1, i64] = to_tensor([t(2.5f32)])"
        ),
        "rigid_bare_apply",
        &[2.0],
    );
}

#[test]
fn an_integer_source_reached_by_either_order_is_rejected_in_every_lane() {
    // Deferring the source-dependent rejection must not turn it into an
    // acceptance: once the variable binds to an integer, every order rejects.
    for operands in [
        "cast_trunc(num_id(x), i64), cast(num_id(x), i64)",
        "cast(num_id(x), i64), cast_trunc(num_id(x), i64)",
    ] {
        let body = format!(
            "{HIGHER_ORDER}def main() -> tensor[1, i64] = \
             to_tensor([apply_it(fn (x) -> add({operands}), 3i32)])"
        );
        rejected_in_every_lane(&program(&body), &body);
    }
}
