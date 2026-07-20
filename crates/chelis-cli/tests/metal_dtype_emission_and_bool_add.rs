//! Sweep 4 of the numeric audit brief: the Metal backend's per-dtype
//! emission surface, plus chelis#726 (add on bool tensors).
//!
//! Metal came out of the sweep as the best-behaved backend, and this file
//! locks that: rank-1 kernels are HONESTLY TYPED per dtype (`long*` for
//! int64, `int*` for int32, `bool*` for bool, `half`/`bfloat` for f16/bf16;
//! the narrow-float rows live in narrow_dtype_matrix.rs), f64 is rejected
//! with a specific diagnostic, and rank-2+ falls back to a LOUD abort stub
//! that names itself. No F32 substitution anywhere (contrast chelis#689).
//!
//! The #699 Metal symptom is also settled here: an int64 `abs` emits
//! `// node 0 = Const 0` with a `(int64_t)0LL` fill and the input tensor
//! absent from the kernel signature - the zero arrives PRE-PLANTED from
//! `lower_transcendental`; there is no separate Metal bug.
//!
//! chelis#726: `add` on bool tensors - the checker accepts it, both host
//! lanes store the out-of-domain value 2 in a bool-typed tensor (prints
//! `2.0`, `to_list`s back as `true`), and Metal's honestly-typed `bool*`
//! kernel would compute 1. Three answers for `true + true`.
//!
//! All Metal probes are emission-only (`chelis build --target metal` never
//! invokes `xcrun metal`), so they run on any host.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// Build for --target metal; (success, stderr, concatenated text artifacts).
fn build_metal(program: &str, name: &str) -> (bool, String, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    let mut emitted = String::new();
    if out_dir.is_dir() {
        for entry in std::fs::read_dir(&out_dir).expect("read out dir") {
            let p = entry.expect("entry").path();
            if p.extension().is_some_and(|e| e == "mm" || e == "h")
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                emitted.push_str(&text);
            }
        }
    }
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        emitted,
    )
}

fn eval_first_line(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
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
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

// ===========================================================================
// Metal emission locks (all pass)
// ===========================================================================

/// Integer and bool kernels are typed to their dtype - the property #689's
/// HIP `elem_kind` fallback lacks.
#[test]
fn metal_kernels_are_honestly_typed_per_dtype() {
    let (ok, stderr, emitted) = build_metal(
        "def f(a: tensor[4, int64], b: tensor[4, int64]) -> tensor[4, int64] = add(a, b)\n",
        "metal_i64",
    );
    assert!(ok, "{stderr}");
    assert!(
        emitted.contains("device const long*") && emitted.contains("device long*"),
        "int64 kernels must be long-typed"
    );

    let (ok, stderr, emitted) = build_metal(
        "def f(a: tensor[4, int32], b: tensor[4, int32]) -> tensor[4, int32] = add(a, b)\n",
        "metal_i32",
    );
    assert!(ok, "{stderr}");
    assert!(
        emitted.contains("device const int*"),
        "int32 kernels must be int-typed"
    );
}

/// f64 is rejected with a diagnostic naming the dtype - loud, specific,
/// by design (Apple Silicon has no f64).
#[test]
fn metal_rejects_f64_with_a_specific_diagnostic() {
    let (ok, stderr, _) = build_metal(
        "def f(a: tensor[4, f64], b: tensor[4, f64]) -> tensor[4, f64] = add(a, b)\n",
        "metal_f64",
    );
    assert!(!ok, "Metal must reject f64");
    assert!(
        stderr.contains("rejects f64"),
        "the rejection must name f64; got: {stderr}"
    );
}

/// Rank-2+ falls back to an abort stub that NAMES ITSELF in the emitted
/// source - the loud fallback shape #703 asks for (contrast the silent
/// substitutions elsewhere in the audit).
#[test]
fn metal_rank2_fallback_is_a_named_abort_stub() {
    let (ok, stderr, emitted) = build_metal(
        "def f(a: tensor[2, 2, f32], b: tensor[2, 2, f32]) -> tensor[2, 2, f32] = add(a, b)\n",
        "metal_rank2",
    );
    assert!(ok, "{stderr}");
    assert!(
        emitted.contains("fallback stub") && emitted.contains("abort()"),
        "the rank-2 fallback must be a self-naming abort, not a silent stub"
    );
}

/// The #699 Metal symptom, REPLACED at chelis#730 Phase 1: the int64
/// `abs` def used to arrive with a pre-planted `Const 0` node from
/// `lower_transcendental` and Metal emitted a zero-filled buffer. The
/// placeholder now raises, so the build is REJECTED loudly (the DAG lane
/// refuses; the host-emission fallback's scalar arm refuses the tensor
/// operand) and no zero-filled emission exists to lock. Replace with a
/// correctness row when chelis#729 lands integer abs.
#[test]
fn metal_int64_abs_is_rejected_not_pre_planted_zero() {
    let (ok, stderr, emitted) = build_metal(
        "def f(a: tensor[4, int64]) -> tensor[4, int64] = abs(a)\n",
        "metal_i64_abs",
    );
    assert!(
        !ok,
        "an int64 abs def must be rejected, never emitted as a zero buffer"
    );
    assert!(
        stderr.contains("unsupported:"),
        "the rejection must carry the branded diagnostic; got: {stderr}"
    );
    assert!(
        !emitted.contains("node 0 = Const 0"),
        "no pre-planted zero emission may be left behind"
    );
}

// ===========================================================================
// chelis#726 - add on bool tensors
// ===========================================================================

/// Observed today: eval accepts and prints `data=[2.0, 1.0]` - the value 2
/// inside a bool-typed tensor (and `to_list` of the same tensor says
/// `[true, true]`). The correct behavior is a checker rejection; this row
/// asserts rejection-or-domain-consistency so it goes green on either a
/// checker fix or an authored bool-arithmetic semantics.
#[test]
#[ignore = "chelis#726: add on bool tensors stores 2 in a bool tensor (prints 2.0, to_lists \
            as true; Metal's typed kernel would compute 1). Must be rejected or made \
            domain-consistent. Run with \
            `cargo test -p chelis-cli --test metal_dtype_emission_and_bool_add -- --ignored`."]
fn bool_tensor_add_is_rejected_or_stays_in_domain() {
    let program = "module M.Main\n\
         def f(x: tensor[2, bool], y: tensor[2, bool]) -> tensor[2, bool] = add(x, y)\n\
         out = print(f(to_tensor([true, false]), to_tensor([true, true])))\n";
    match eval_first_line(program) {
        Err(stderr) => assert!(
            stderr.contains("bool") || stderr.contains("Type errors"),
            "a rejection must name the bool-arithmetic problem; got: {stderr}"
        ),
        Ok(line) => assert!(
            !line.contains("2.0"),
            "a bool tensor must never hold the value 2; got: {line}"
        ),
    }
}

/// Control: boolean ops on bool tensors are correct, and the counting idiom
/// (cast then sum) works - what the #726 rejection should point users to.
#[test]
fn bool_logic_and_counting_idiom_are_correct() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, bool], y: tensor[2, bool]) -> tensor[2, bool] = and(x, y)\n\
         out = print(f(to_tensor([true, false]), to_tensor([true, true])))\n",
    )
    .expect("eval");
    // chelis#732 P1 migration: bool tensor elements print true/false
    // ([05-OBS-2]) and rank-0 tensors render bare ([05-OBS-4]).
    assert_eq!(line, "tensor(shape=[2], data=[true, false])");

    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[3, bool]) -> tensor[int64] = sum(cast(x, int64), 0)\n\
         out = print(f(to_tensor([true, false, true])))\n",
    )
    .expect("eval");
    assert_eq!(line, "2");
}
