//! Sweep 4 of the numeric audit brief: the Metal backend's per-dtype
//! emission surface, plus chelis#726 (add on bool tensors).
//!
//! Metal came out of the sweep as the best-behaved backend, and this file
//! locks that: rank-1 kernels are HONESTLY TYPED per dtype (`long*` for
//! i64, `int*` for i32, `bool*` for bool, `half`/`bfloat` for f16/bf16;
//! the narrow-float rows live in narrow_dtype_matrix.rs), f64 is rejected
//! with a specific diagnostic, and unsupported rank-2+ lowering fails through
//! the typed codegen channel. No F32 substitution anywhere (contrast
//! chelis#689).
//!
//! The #699 Metal symptom is also settled here: an i64 `abs` no longer
//! lowers to a pre-planted `Const 0`. Until Phase 3 supplies a typed,
//! trapping Metal kernel, emission returns the branded typed unsupported
//! reason without writing an artifact.
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
        "def f(a: tensor[4, i64], b: tensor[4, i64]) -> tensor[4, i64] = add(a, b)\n",
        "metal_i64",
    );
    assert!(ok, "{stderr}");
    assert!(
        emitted.contains("device const long*") && emitted.contains("device long*"),
        "i64 kernels must be long-typed"
    );

    let (ok, stderr, emitted) = build_metal(
        "def f(a: tensor[4, i32], b: tensor[4, i32]) -> tensor[4, i32] = add(a, b)\n",
        "metal_i32",
    );
    assert!(ok, "{stderr}");
    assert!(
        emitted.contains("device const int*"),
        "i32 kernels must be int-typed"
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
        stderr.contains("f64 value"),
        "the rejection must name f64; got: {stderr}"
    );
    assert!(
        stderr.contains("deliberate [04-TGT-1]"),
        "the rejection must cite the Metal target contract; got: {stderr}"
    );
    assert!(
        stderr.contains("--target c") && stderr.contains("--target hip"),
        "the rejection must name targets that support f64; got: {stderr}"
    );
}

/// Rank-2+ fails through the typed codegen channel before any artifact exists.
#[test]
fn metal_rank2_is_a_typed_error_without_an_artifact() {
    let (ok, stderr, emitted) = build_metal(
        "def f(a: tensor[2, 2, f32], b: tensor[2, 2, f32]) -> tensor[2, 2, f32] = add(a, b)\n",
        "metal_rank2",
    );
    assert!(!ok, "rank-2 Metal must fail the build");
    assert!(
        stderr.contains("unsupported:") && stderr.contains("codegen:metal"),
        "the rank-2 rejection must use the typed codegen channel: {stderr}"
    );
    assert!(
        emitted.is_empty(),
        "a rejected build wrote an artifact: {emitted}"
    );
}

/// The #699 Metal symptom, REPLACED at chelis#730 Phase 1: the i64
/// `abs` def used to arrive with a pre-planted `Const 0` node from
/// `lower_transcendental` and Metal emitted a zero-filled buffer. Phase 2
/// now preserves a typed integer `Abs` node. Metal's public DAG emitter
/// rejects that node through the typed error channel without materializing
/// an abort artifact. Replace this with a
/// correctness row when the chelis#699 Phase 3 kernel lands.
#[test]
fn metal_int64_abs_is_a_typed_error_not_pre_planted_zero() {
    let (ok, stderr, emitted) = build_metal(
        "def f(a: tensor[4, i64]) -> tensor[4, i64] = abs(a)\n",
        "metal_i64_abs",
    );
    assert!(!ok, "the Metal build must reject integer abs");
    assert!(
        !emitted.contains("node 0 = Const 0"),
        "no pre-planted zero emission may be left behind"
    );
    assert!(
        stderr.contains("unsupported:")
            && stderr.contains(
                "integer abs code generation waits for the typed, trapping Phase 3 kernel"
            ),
        "integer abs must return its branded typed reason; got:\n{stderr}"
    );
    assert!(
        emitted.is_empty(),
        "a rejected build wrote an artifact: {emitted}"
    );
}

// ===========================================================================
// chelis#726 - add on bool tensors
// ===========================================================================

/// Historical observation: eval accepted and printed `data=[2.0, 1.0]` -
/// the value 2 inside a bool-typed tensor (while `to_list` of the same
/// tensor said `[true, true]`). UN-IGNORED at the chelis#729 rework: the
/// decided chelis#726 disposition landed as a check-time rejection at
/// the shared operand-dtype chokepoint (chelis#860), so the Err arm is
/// now the only reachable one and carries the capability citation.
#[test]
fn bool_tensor_add_is_rejected_or_stays_in_domain() {
    let program = "module M.Main\n\
         def f(x: tensor[2, bool], y: tensor[2, bool]) -> tensor[2, bool] = add(x, y)\n\
         out = print(f(to_tensor([true, false]), to_tensor([true, true])))\n";
    match eval_first_line(program) {
        Err(stderr) => assert!(
            stderr.contains("chelis#726") && stderr.contains("bool"),
            "the rejection must carry the chelis#726 capability citation; \
             got: {stderr}"
        ),
        Ok(line) => panic!(
            "add on bool tensors must be rejected by the checker \
             (chelis#726, decided); it evaluated and returned {line}"
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
         def f(x: tensor[3, bool]) -> tensor[i64] = sum(cast(x, i64), 0)\n\
         out = print(f(to_tensor([true, false, true])))\n",
    )
    .expect("eval");
    assert_eq!(line, "2");
}
