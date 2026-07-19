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
//! * **Emitted-bits** (gcc-free): the generated C passes the bounds as
//!   `chelis_f32_from_bits(0x…u)`. A correct build carries the real bounds'
//!   bit patterns and never the [0,1) default pair
//!   (`0x00000000` / `0x3f800000`). This directly locks the issue mechanism.
//! * **Cross-lane f32-bit parity** (needs gcc): every element bit-identical
//!   once the host f64 value is narrowed to f32 — the same comparator shape as
//!   `issue_770_uniform_like_affine_parity.rs` (`%.16g` round-trips an f32
//!   exactly), so a stdout digit-count difference between the lanes cannot
//!   masquerade as a value divergence and a silently skipped element cannot
//!   mask one (`parse_tensor_data` fails loudly on a missing/unparseable line).
//! * **Loud fallback**: a genuinely non-static bound (a runtime add) fails the
//!   `chelis build` loudly, naming chelis#776 — it never silently compiles to
//!   the [0,1) default and never laundered into a `/* unsupported builtin */`
//!   host stub. `chelis eval`, which interprets the range, stays correct.
//!
//! Note on the type checker: the `def f(...) = uniform_like(t, lo, hi)` form is
//! rejected at check time ("uniform_like currently requires literal low/high
//! bounds"), but that gate is bypassed for the top-level
//! `with seed { uniform_like(...) }` form used here — the only form that
//! supplies a seed — so the lowering guard is the actual enforcement, not mere
//! defense-in-depth. The def-form rejection is pinned separately below.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{gcc_available, link_generated, parse_tensor_data, write_file};

/// An 8-element f32 template plus a `with seed(seed)` uniform draw over
/// `[low, high)`, where `low`/`high` are arbitrary Surf bound expressions.
fn program(low: &str, high: &str, seed: u64) -> String {
    format!(
        "template = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), \
         cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])\n\
         sampled = with seed({seed}) {{ uniform_like(copy(template), {low}, {high}) }}\n"
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

/// Build the program to C and return the generated `<name>.c` source (does not
/// require a host C compiler).
fn generated_c_source(program: &str, name: &str) -> String {
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
    std::fs::read_to_string(out_dir.join(format!("{name}.c"))).expect("generated C source")
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

/// The [0,1) default bounds the pre-fix lowering silently substituted:
/// `low = 0.0f` (`0x00000000`), `high = 1.0f` (`0x3f800000`).
const UNIT_LOW_BITS: &str = "chelis_f32_from_bits(0x00000000u)";
const UNIT_HIGH_BITS: &str = "chelis_f32_from_bits(0x3f800000u)";

// -- cast-wrapped bounds -----------------------------------------------------

#[test]
fn cast_wrapped_bounds_emit_declared_range_not_unit_default() {
    // `cast(2.0, f32)` / `cast(5.0, f32)` -> low 2.0f (0x40000000),
    // high 5.0f (0x40a00000). Pre-fix the emitted call was the [0,1) default.
    let src = generated_c_source(
        &program("cast(2.0, f32)", "cast(5.0, f32)", 42),
        "cast_bits",
    );
    assert!(
        src.contains("chelis_f32_from_bits(0x40000000u)")
            && src.contains("chelis_f32_from_bits(0x40a00000u)"),
        "emitted C must carry the declared [2,5) bounds:\n{src}"
    );
    assert!(
        !src.contains(UNIT_HIGH_BITS),
        "emitted C must NOT fall back to the [0,1) default high bound (0x3f800000)"
    );
    assert!(
        !src.contains("unsupported builtin"),
        "emitted C must not launder uniform_like into an unsupported-builtin stub"
    );
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
fn negative_literal_bounds_emit_declared_range_not_unit_default() {
    // `-3.0` / `-1.0` -> low -3.0f (0xc0400000), high -1.0f (0xbf800000).
    let src = generated_c_source(&program("-3.0", "-1.0", 42), "neg_bits");
    assert!(
        src.contains("chelis_f32_from_bits(0xc0400000u)")
            && src.contains("chelis_f32_from_bits(0xbf800000u)"),
        "emitted C must carry the declared [-3,-1) bounds:\n{src}"
    );
    assert!(
        !src.contains(UNIT_LOW_BITS) && !src.contains(UNIT_HIGH_BITS),
        "emitted C must NOT fall back to the [0,1) default bounds"
    );
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

// -- loud fallback on a genuinely non-static bound ---------------------------

#[test]
fn runtime_computed_bound_build_fails_loudly() {
    // `cast(2.0, f32) + cast(1.0, f32)` is a runtime add the lowering cannot
    // fold to a compile-time constant. The build must fail LOUDLY, never emit
    // a [0,1) sampler and never a `/* unsupported builtin */` stub.
    let src = program("cast(2.0, f32) + cast(1.0, f32)", "cast(5.0, f32)", 42);
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("runtime.ch");
    let out_dir = dir.path().join("runtime-out");
    write_file(&path, &src);
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
        "build must fail loudly on a runtime uniform_like bound, not silently compile"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("uniform_like")
            && stderr.contains("statically-resolvable")
            && stderr.contains("chelis#776"),
        "build diagnostic must name uniform_like, the static requirement, and chelis#776:\n{stderr}"
    );
    // The failed build must not have written a C artifact that silently
    // samples the [0,1) default.
    let generated = out_dir.join("runtime.c");
    if let Ok(c) = std::fs::read_to_string(&generated) {
        assert!(
            !c.contains(UNIT_HIGH_BITS),
            "a failed build must not leave a [0,1)-default sampler artifact:\n{c}"
        );
    }
}

#[test]
fn runtime_computed_bound_eval_still_correct() {
    // The eval lane interprets the range, so the loud build error must not
    // regress it: `[2.0+1.0, 5.0)` = `[3, 5)`.
    let src = program("cast(2.0, f32) + cast(1.0, f32)", "cast(5.0, f32)", 42);
    let sampled = eval_sampled(&src);
    assert_eq!(sampled.len(), 8, "expected 8 samples");
    for (i, v) in sampled.iter().enumerate() {
        assert!(
            (3.0..5.0).contains(v),
            "eval elem[{i}] = {v} must lie in [3,5)"
        );
    }
}

// -- def-form checker rejection (a separate, earlier loud guard) -------------

#[test]
fn def_form_non_literal_bound_rejected_by_checker() {
    // In a `def` body the checker rejects a non-literal bound before lowering
    // ("uniform_like currently requires literal low/high bounds"). This pins
    // that this earlier guard stays loud; the lowering guard above covers the
    // `with seed { ... }` form the checker does not gate.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("def.ch");
    write_file(
        &path,
        "def draw(lo: f32, hi: f32, t: &tensor[n, f32]) -> tensor[n, f32] ! { Random } = \
         uniform_like(t, lo, hi)\n",
    );
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("uniform_like currently requires literal low/high bounds"),
        "def-form non-literal bound must be rejected by the checker:\n{combined}"
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
        "def f(x: tensor[4, f32]) -> tensor[6, f32] = pad(&x, [[1, 1]], cast(7.0, f32))\n\
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
        src.contains("0x40e00000u"),
        "emitted C must carry the declared 7.0f pad fill (0x40e00000):\n{src}"
    );
    assert!(
        !src.contains("fill_f32_bits(t1, 0x00000000u)"),
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
        "def f(x: tensor[4, f32], r: f32) -> tensor[6, f32] = pad(&x, [[1, 1]], r)\n\
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
