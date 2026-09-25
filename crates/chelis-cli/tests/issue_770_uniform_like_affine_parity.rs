//! chelis#770 — the host evaluator computed `uniform_like`'s affine
//! `low + (high - low) * unit` in f64, while the compiled C lane's
//! `chelis_uniform_sample_f32` does it in f32 (`return low + (high - low) *
//! (float)unit;` with f32 `low`/`high`, chelis-backend-c/src/emit.rs:164 and
//! host_emit.rs:413). For non-[0,1) ranges the f64 affine double-rounds and
//! drifts up to 1 ULP from the compiled lane. The decision (posted on the
//! issue) is **eval conforms to C**: the fix mirrors the C f32 sampler
//! op-for-op at both eval sites (chelis-compiler-api host_ops
//! `uniform_like_value` and chelis-ir `eval::uniform_like`).
//!
//! ## Oracle: f32-bit parity, not stdout byte-equality
//!
//! The oracle is **element-wise f32-bit equality** — the same comparator the
//! chelis#735 RNG sweep used, which is what surfaced this bug: parse each
//! lane's rendered `data=[...]` values, cast the host f64 element to f32, and
//! compare its bit pattern to the compiled lane's f32 element (also read back
//! through f64 and cast to f32; `%.16g` round-trips an f32 exactly).
//!
//! Stdout byte-equality is deliberately NOT the oracle: the two lanes print
//! floats through different algorithms — the host lane via Rust `{:?}`
//! (shortest round-trip, `render_value`) and the compiled lane via
//! `%.1f`/`%.16g` — so their raw strings can differ (17 vs 16 significant
//! digits) even when the underlying f32 values are bit-identical. That
//! text-level divergence is the float-printer work (chelis#732 / chelis#728),
//! not chelis#770's concern; the f32-bit level is the value-conformance
//! invariant this fix makes universal. The parser fails loudly on a missing
//! `sampled` line or an unparseable token (via `parse_tensor_data`), so a
//! silently skipped element cannot mask a divergence.
//!
//! ## Ranges under test
//!
//! * `[2, 5)` — the primary fix case. Pre-fix, host elem[4] cast to f32 was
//!   `0x404215a9` while C was `0x404215aa` (measured on origin/main before
//!   the fix, under the retired pre-[05-RNG-1] mixing); this asserts they now
//!   agree bit-for-bit at every element. Under explicit keys (chelis#2413)
//!   the draw keyed by `key_from_seed(30)` has an element, elem[6], where
//!   both an f64 affine and a two-rounding f32 affine land 1 ULP below the
//!   single-rounding FMA (`0x4095f8ce` against `0x4095f8cf`, from
//!   `common::key_ref`'s [05-OP-8] transcription and `key_ref.py`).
//! * `[0, 1)` — also a FIXED case, not a control: per this PR's Finding B the
//!   pre-fix host lane held the raw f64 `unit` while C held `(float)unit`, so
//!   the raw-f64 values differed; they only ever agreed at the f32-bit level
//!   (the #735 comparator's level), which this fix now makes the guaranteed
//!   invariant. Asserted with the same f32-bit oracle.
//! * Sanity control = different keys produce different draws
//!   (`different_keys_produce_different_draws`), so the oracle is not
//!   trivially always-equal.
//!
//! A negative range (`[-3, -1)`) is covered at the unit level in the two eval
//! crates' own tests rather than cross-lane here: the compiled C lane cannot
//! be driven with a bare negative low/high literal today —
//! `extract_f64_value` in the IR lowering (lower.rs:6950-6951) does not see
//! through the unary-minus (nor a `cast(...)`) form and silently defaults the
//! range to `[0, 1)`. That silent range collapse is a separate lowering gap,
//! filed as chelis#776; these tests use bare positive literals deliberately
//! to stay clear of it.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{gcc_available, link_generated, parse_tensor_data, write_file};

/// An 8-element f32 template plus a uniform draw keyed by `key_from_seed(seed)`
/// over `[low, high)`. Bare numeric literals (not `cast(...)`) so the compiled
/// lane's IR lowering plumbs the range through to the sampler (see the
/// module doc and chelis#776 for why the cast form is avoided).
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

/// Build the program to C, compile it with FP contraction DISABLED
/// (`-ffp-contract=off`), run it, and return the printed `sampled` data.
/// `link_generated` bakes the default flags (which imply `-ffp-contract=fast`
/// under `-march=native`); this mirrors that recipe but forces contraction
/// off, so the test can assert the sampler's explicit `fmaf` gives the SAME
/// bytes either way — the flag-independence / RNG-determinism lock (chelis#770).
fn c_sampled_no_fp_contract(program: &str, name: &str) -> Vec<f64> {
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
    let source = format!("{name}.c");
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: common::generated_source_needs_blas(&out_dir, &source),
        },
    );
    let mut cmd = std::process::Command::new(&toolchain.compiler);
    cmd.current_dir(&out_dir);
    cmd.arg("-O2");
    cmd.arg("-ffp-contract=off");
    cmd.args(&toolchain.compile_flags);
    cmd.arg(&source);
    cmd.args(["-L.", "-lchelis_runtime"]);
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", name]);
    let status = cmd.status().expect("host compiler should run");
    assert!(
        status.success(),
        "no-contract compile of generated C failed: {status}"
    );
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

#[test]
fn uniform_like_affine_parity_positive_range() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = program("2.0", "5.0", 30);
    let eval = eval_sampled(&src);
    let c = c_sampled(&src, "affine_2_5");
    assert_f32_bit_parity(&eval, &c, "[2,5)");
    // Nail the element where an f64 affine and a two-rounding f32 affine
    // both round to the adjacent f32 (0x4095f8ce). The bits are the
    // [05-RNG-2]/[05-OP-8] value for `key_from_seed(30)`, from
    // `common::key_ref` (and `key_ref.py`/`slice2_ref.py`).
    let reference = common::key_ref::uniform_f32(common::key_ref::key_from_seed(30), 8, 2.0, 5.0);
    assert_eq!(reference[6].to_bits(), 0x4095f8cf);
    assert_eq!(
        (eval[6] as f32).to_bits(),
        0x4095f8cf,
        "elem[6] must be the single-rounding f32 FMA value",
    );
    assert_f32_bit_parity(
        &reference.into_iter().map(f64::from).collect::<Vec<_>>(),
        &c,
        "reference vs C, [2,5)",
    );
}

/// Flag-independence / RNG-determinism lock: with the C sampler's explicit
/// `fmaf`, the compiled [2,5) output is bit-identical to eval even when FP
/// contraction is disabled. Pre-fix, `-ffp-contract=off` diverged at one
/// element; under explicit keys, `key_from_seed(30)`'s elem[6] is such an
/// element (a two-rounding affine gives 0x4095f8ce where the FMA gives
/// 0x4095f8cf), i.e. the same source would produce different "random" bytes
/// under different compile flags. This asserts that hole is closed.
#[test]
fn uniform_like_affine_parity_positive_range_no_fp_contract() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = program("2.0", "5.0", 30);
    let eval = eval_sampled(&src);
    let c = c_sampled_no_fp_contract(&src, "affine_2_5_noc");
    assert_f32_bit_parity(&eval, &c, "[2,5) -ffp-contract=off");
    // And the compiled output must be identical under both contraction modes.
    let c_default = c_sampled(&src, "affine_2_5_default");
    assert_eq!(
        c.iter().map(|v| (*v as f32).to_bits()).collect::<Vec<_>>(),
        c_default
            .iter()
            .map(|v| (*v as f32).to_bits())
            .collect::<Vec<_>>(),
        "C sampler output must be identical with and without FP contraction",
    );
}

#[test]
fn uniform_like_affine_parity_unit_range() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = program("0.0", "1.0", 42);
    let eval = eval_sampled(&src);
    let c = c_sampled(&src, "unit_0_1");
    // A FIXED case, not a control (Finding B): `[0,1)` only ever agreed at the
    // f32-bit level, which this fix makes the guaranteed invariant.
    assert_f32_bit_parity(&eval, &c, "[0,1)");
}

/// Sanity control so the f32-bit oracle is not trivially always-equal: two
/// different keys must produce different draws. Pure host lane (no C
/// compiler needed), so it also runs where the codegen tests skip.
#[test]
fn different_keys_produce_different_draws() {
    let a = eval_sampled(&program("2.0", "5.0", 42));
    let b = eval_sampled(&program("2.0", "5.0", 43));
    assert_eq!(a.len(), b.len(), "same shape for both keys");
    assert_ne!(a, b, "distinct keys must yield distinct uniform draws");
}
