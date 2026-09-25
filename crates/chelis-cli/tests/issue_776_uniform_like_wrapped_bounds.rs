//! chelis#776 — the C/GPU IR lowering silently dropped a `uniform_like`
//! range whose bounds were wrapped in a value-carrying node the static
//! extractor could not see through. `extract_f64_value` (lower.rs) matched
//! only a bare `Atom::Float/Int` or a `(lit …)` node, so a `cast(...)`- or
//! `neg`-wrapped bound fell to the caller's `unwrap_or(0.0)` / `unwrap_or(1.0)`
//! and the compiled lane sampled the raw [0,1) unit stream while `chelis eval`
//! honored the declared range. A whole-range silent cross-lane divergence
//! (strictly worse than #770's ≤1-ULP affine issue), the #703 silent-
//! substitution class.
//!
//! The fix (raise-or-prove, C1.4): the extractor now sees through the
//! statically-resolvable value-carrying wrappers — a `neg(...)` of an
//! extractable value and a `cast(..., <float prim>)` of an extractable value
//! (a float-target cast preserves the numeric value; the f32 sampler narrows
//! the same bits in every lane) — and any bound it still cannot fold is a
//! **loud fatal lowering error** at build, never a silent default.
//!
//! ## Oracles
//!
//! * **Declared range**: every compiled sample lies in the declared range and
//!   outside the [0,1) default, which directly locks the issue mechanism.
//! * **Cross-lane f32-bit parity** (needs gcc): every element bit-identical
//!   once the host f64 value is narrowed to f32 — the same comparator shape as
//!   `issue_770_uniform_like_affine_parity.rs` (`%.16g` round-trips an f32
//!   exactly), so a stdout digit-count difference between the lanes cannot
//!   masquerade as a value divergence and a silently skipped element cannot
//!   mask one (`parse_tensor_data` fails loudly on a missing/unparseable line).
//! * **Runtime bounds** (chelis#2411/#2413): [05-OP-8] bounds are ordinary
//!   scalar operands, so a runtime-computed bound checks clean, evaluates and
//!   compiles; both lanes sample the computed range bit for bit. The static
//!   gate that used to reject it, and PR #782's fatal lowering error behind
//!   it, are gone with the static-rate grammar.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{gcc_available, link_generated, parse_tensor_data, write_file};

/// An 8-element f32 template plus a uniform draw keyed by
/// `key_from_seed(seed)` over `[low, high)`, where `low`/`high` are arbitrary
/// Surf bound expressions.
fn program(low: &str, high: &str, seed: i64) -> String {
    format!(
        "template = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), \
         cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])\n\
         sampled = uniform_like(key_from_seed({seed}i64), copy(template), {low}, {high})\n"
    )
}

/// Run `chelis eval --file` and return the printed `sampled` tensor data.
fn eval_sampled(program: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "chelis eval failed: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    parse_tensor_data(&String::from_utf8_lossy(&out.stdout), "sampled")
}

/// Build the program to C, link it, run it, and return the printed `sampled`
/// tensor data.
fn c_sampled(program: &str, name: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    Command::cargo_bin("chelis")
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
        .assert()
        .success();
    let status = link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link of generated C failed: {status}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "compiled binary exited non-zero: {}",
        String::from_utf8_lossy(&run.stderr),
    );
    parse_tensor_data(&String::from_utf8_lossy(&run.stdout), "sampled")
}

/// Assert both lanes produce the same number of elements and that every
/// element is bit-identical once the host f64 value is narrowed to f32.
fn assert_f32_bit_parity(eval: &[f64], c: &[f64], context: &str) {
    assert_eq!(
        eval.len(),
        c.len(),
        "[{context}] element count mismatch: eval {} vs C {}",
        eval.len(),
        c.len(),
    );
    assert!(!eval.is_empty(), "[{context}] no elements parsed");
    for (i, (e, cv)) in eval.iter().zip(c.iter()).enumerate() {
        assert_eq!(
            (*e as f32).to_bits(),
            (*cv as f32).to_bits(),
            "[{context}] elem[{i}] f32 bits diverge: eval {} (bits {:#010x}) vs C {} (bits {:#010x})",
            e,
            (*e as f32).to_bits(),
            cv,
            (*cv as f32).to_bits(),
        );
    }
}

/// Every sample lies in `[low, high)`, and at least one lies outside the
/// [0,1) default range the pre-fix lowering silently substituted.
fn assert_declared_range(samples: &[f64], low: f64, high: f64, context: &str) {
    assert!(!samples.is_empty(), "[{context}] no elements parsed");
    for (index, sample) in samples.iter().enumerate() {
        assert!(
            (low..high).contains(sample),
            "[{context}] elem[{index}] = {sample} is outside [{low}, {high})"
        );
    }
    assert!(
        samples.iter().any(|sample| !(0.0..1.0).contains(sample)),
        "[{context}] every sample lies in the [0,1) default range"
    );
}

// -- cast-wrapped bounds -----------------------------------------------------

#[test]
fn cast_wrapped_bounds_sample_declared_range_not_unit_default() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    // `cast(2.0, f32)` / `cast(5.0, f32)`: pre-fix the compiled lane drew the
    // [0,1) default.
    let c = c_sampled(
        &program("cast(2.0, f32)", "cast(5.0, f32)", 42),
        "cast_bits",
    );
    assert_declared_range(&c, 2.0, 5.0, "cast [2,5)");
}

#[test]
fn cast_wrapped_bounds_cross_lane() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = program("cast(2.0, f32)", "cast(5.0, f32)", 42);
    let eval = eval_sampled(&src);
    let c = c_sampled(&src, "cast_2_5");
    assert_f32_bit_parity(&eval, &c, "cast [2,5)");
}

// -- negative-literal bounds -------------------------------------------------

#[test]
fn negative_literal_bounds_sample_declared_range_not_unit_default() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let c = c_sampled(&program("-3.0", "-1.0", 42), "neg_bits");
    assert_declared_range(&c, -3.0, -1.0, "neg [-3,-1)");
}

#[test]
fn negative_literal_bounds_cross_lane() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = program("-3.0", "-1.0", 42);
    let eval = eval_sampled(&src);
    let c = c_sampled(&src, "neg_3_1");
    assert_f32_bit_parity(&eval, &c, "neg [-3,-1)");
}

// -- mixed neg + cast --------------------------------------------------------

#[test]
fn mixed_neg_and_cast_bounds_cross_lane() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    // low is a unary-minus literal, high is a float-cast literal: both wrapper
    // forms exercised in one range.
    let src = program("-3.0", "cast(5.0, f32)", 42);
    let eval = eval_sampled(&src);
    let c = c_sampled(&src, "mixed_neg_cast");
    assert_f32_bit_parity(&eval, &c, "mixed [-3,5)");
}

// -- runtime-computed bounds ---------------------------------------------------
//
// chelis#2411/#2413: a runtime-computed bound is an ordinary [05-OP-8] scalar
// operand, validated by the draw at execution. It checks clean at top level
// and in a `def` body, and both lanes sample the computed range.

#[test]
fn runtime_computed_bound_samples_its_computed_range_in_both_lanes() {
    let src = program("cast(2.0, f32) + cast(1.0, f32)", "cast(5.0, f32)", 42);
    let eval = eval_sampled(&src);
    assert_declared_range(&eval, 3.0, 5.0, "eval runtime [3,5)");
    if !gcc_available() {
        eprintln!("skipping the C lane: no host C compiler");
        return;
    }
    let c = c_sampled(&src, "runtime_3_5");
    assert_f32_bit_parity(&eval, &c, "runtime [3,5)");
}

#[test]
fn def_form_runtime_bound_checks_clean() {
    let score = check_score(
        "def draw[n](k: key, lo: f32, hi: f32, t: &tensor[n, f32]) -> tensor[n, f32] = \
         uniform_like(k, t, lo, hi)\n",
    );
    assert!(
        (score - 1.0).abs() < 1e-9,
        "a def-form runtime bound must check clean, got {score}"
    );
}

// -- RT-782 keeper: pad-fill sibling fix locked at the emitted-C level --------
//
// The suite above proves the uniform_like emitted-bits fix but asserts pad's
// fix only through the in-crate `pad_fill()` DAG helper in lower.rs. RT-782
// confirmed by execution that on the parent (b469ace6) a cast-wrapped pad fill
// silently baked `chelis_f32_from_bits(0x00000000u)` (0.0f) into the generated
// C — a real shipped-binary miscompile of the same #703 class as the
// uniform_like bug — while the branch bakes the declared `0x40e00000u` (7.0f).

#[test]
fn pad_cast_wrapped_fill_emits_declared_bits_not_silent_zero() {
    // `cast(7.0, f32)` fill -> 7.0f == 0x40e00000. Pre-fix the generated C baked
    // the pad-with-zeros default 0x00000000 verbatim, dropping the user's fill.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pad_fill.ch");
    let out_dir = dir.path().join("pad_fill-out");
    write_file(
        &path,
        "def f(x: tensor[4, f32]) -> tensor[6, f32] = pad(&x, [[1i64, 1i64]], cast(7.0, f32))\n\
         out = f(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n",
    );
    Command::cargo_bin("chelis")
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
        .assert()
        .success();
    let src = std::fs::read_to_string(out_dir.join("pad_fill.c")).expect("generated C source");
    assert!(
        src.contains(
            "chelis_fill_scalar(t1_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F32, UINT32_C(0x40e00000)))"
        ),
        "emitted C must carry the declared 7.0f pad fill (0x40e00000):\n{src}"
    );
    assert!(
        !src.contains(
            "chelis_fill_scalar(t1_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F32, UINT32_C(0x00000000)))"
        ),
        "emitted C must NOT silently pad with the 0.0f default when a fill was given:\n{src}"
    );
}

#[test]
fn pad_runtime_fill_build_fails_loudly() {
    // A runtime pad fill is not statically foldable: the build must fail loudly
    // naming pad + chelis#776, never silently bake a 0.0f fill.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pad_rt.ch");
    let out_dir = dir.path().join("pad_rt-out");
    write_file(
        &path,
        "def f(x: tensor[4, f32], r: f32) -> tensor[6, f32] = pad(&x, [[1i64, 1i64]], r)\n\
         out = f(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]), cast(9.0, f32))\n",
    );
    let out = Command::cargo_bin("chelis")
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
    assert!(
        !out.status.success(),
        "build must fail loudly on a runtime pad fill, not silently compile"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("pad")
            && stderr.contains("statically-resolvable")
            && stderr.contains("chelis#776"),
        "build diagnostic must name pad, the static requirement, and chelis#776:\n{stderr}"
    );
}

// -- chelis#731 obligation: the checker sees a keyed draw's bounds -----------
//
// Before chelis#731 Phase 1 the uniform_like literal-bounds checker gate never
// fired inside a seeded handler body (the body was unchecked, chelis#709), so
// only the lowering enforced it. Explicit keys (chelis#2413) removed the
// handler; a draw's bounds are checked wherever the keyed draw is written.
// These two `chelis check`-level tests pin that a statically-resolvable bound
// and a runtime-computed bound of a keyed draw both check clean.

/// `chelis check` score for a `.ch` program.
fn check_score(program: &str) -> f64 {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("gate.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("check must emit JSON");
    parsed["score"].as_f64().expect("numeric score")
}

#[test]
fn keyed_draw_runtime_bound_checks_clean() {
    let score = check_score(&program("add(cast(2.0, f32), cast(1.0, f32))", "5.0", 42));
    assert!(
        (score - 1.0).abs() < 1e-9,
        "a runtime bound of a keyed draw must check clean, got {score}"
    );
}

#[test]
fn gate_accepts_static_bound_in_a_keyed_draw() {
    // Positive parity: a statically-resolvable bound (a negated literal) of a
    // keyed draw passes the gate and checks clean.
    let score = check_score(&program("-3.0", "-1.0", 42));
    assert!(
        (score - 1.0).abs() < 1e-9,
        "a statically-resolvable bound of a keyed draw must check clean, got {score}"
    );
}
