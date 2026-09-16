//! Narrow-float (f16/bf16) coverage across lanes (chelis#714, #716, #717).
//!
//! ## What this file is
//!
//! The bf16/f16 half of the numeric audit (the sweep brief in
//! `docs/investigations/numeric_audit_next_sweeps.md` listed narrow floats as
//! the largest never-swept gap). Three distinct bugs were found by execution,
//! plus a set of controls that bound them:
//!
//! | issue | one line |
//! |---|---|
//! | chelis#714 | f16/bf16 scalars had no C-host-lane representation: they typed as `HostType::Unknown`, arithmetic defaulted to `int64_t`, and `add(0.5f16, 0.25f16)` printed `0` |
//! | chelis#716 | the C host boundaries for bf16/f16 tensors were broken: `print` read the 2-byte buffers as f32, while `to_list`/`to_tensor` aborted |
//! | chelis#717 | the eval tensor lane did not round f16/bf16, so an f16 tensor could hold 2049.0 (not an f16 value) |
//!
//! ## The part that works, and must keep working
//!
//! * the **eval scalar lane** rounds f16/bf16 per-op correctly (IEEE
//!   sequential rounding, locked below);
//! * the **C DAG kernels** compute correctly-rounded f16/bf16 values (WS-1) -
//!   originally proven here by decoding the misprinted bytes; since
//!   chelis#732 Phase 2 the faithful print renders them directly
//!   (`c_print_of_f16_tensor_prints_f16_values`, un-ignored);
//! * **HIP rejects unsupported f16/bf16 elementwise `Add`** with a clean diagnostic and
//!   **Metal emits properly typed** `half`/`bfloat` kernels - the two
//!   backends that get it right;
//! * **f8e4m3 is rejected** by the checker in both lanes (spec §1.1.1).
//!
//! Precision boundaries: f16 mantissa 11 bits (first non-representable
//! integer 2049, max 65504); bf16 mantissa 8 bits (first non-representable
//! 257). Tiny thresholds, trivially reachable from ordinary ML code.
//!
//! Repaired Phase 3 rows are ordinary regression tests; no numeric value row
//! remains hidden behind `#[ignore]`.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

// ---------------------------------------------------------------------------
// Harness (same drivers as issue_703_silent_placeholders.rs: verbatim strings,
// never parsed through f64 except where a test explicitly decodes bits)
// ---------------------------------------------------------------------------

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build `program` to C, link, run. `Ok((emitted_c, stdout))` or stage error.
fn build_and_run_c(program: &str, name: &str) -> Result<(String, String), String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
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
        return Err(String::from_utf8_lossy(&built.stderr).into_owned());
    }
    let emitted = std::fs::read_to_string(out_dir.join(format!("{name}.c")))
        .map_err(|e| format!("read emitted C: {e}"))?;
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    if !run.status.success() {
        return Err(format!(
            "binary exited {}: {}",
            run.status,
            String::from_utf8_lossy(&run.stderr)
        ));
    }
    Ok((emitted, String::from_utf8_lossy(&run.stdout).into_owned()))
}

/// `chelis eval` a full program; first printed line or stderr.
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

/// `chelis eval` a bare printed expression.
fn eval_expr(expr: &str) -> Result<String, String> {
    eval_first_line(&format!("module M.Main\nout = print({expr})\n"))
}

/// Run `chelis build` for `target` and return (success, stderr, out_dir files
/// joined). Emission-only targets (hip/metal) never invoke a GPU toolchain.
fn build_target(program: &str, name: &str, target: &str) -> (bool, String, String) {
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
            target,
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    let mut emitted = String::new();
    if out_dir.is_dir() {
        for entry in std::fs::read_dir(&out_dir).expect("read out dir") {
            let p = entry.expect("entry").path();
            if p.is_file()
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

// ===========================================================================
// CONTROLS THAT PASS. The parts of the narrow-float surface that are correct.
// ===========================================================================

/// **The eval scalar lane rounds f16 per-op, correctly.** IEEE binary16:
/// 2048 + 1 = 2049 is unrepresentable and rounds ties-to-even DOWN to 2048,
/// so the chained add stays at 2048 instead of drifting to 2050. The f64
/// evaluator narrowing once per op is correctly rounded (f64 carries more
/// than 2p+2 bits for p=11). Any fix for chelis#714/#717 must not break this.
#[test]
fn eval_scalar_f16_rounds_per_op() {
    assert_eq!(
        eval_expr("add(cast(2048.0, f16), cast(1.0, f16))").unwrap(),
        // chelis#732 P1 migration: own-width Debug-grammar f16 scalar.
        "2048.0"
    );
    assert_eq!(
        eval_expr("add(add(cast(2048.0, f16), cast(1.0, f16)), cast(1.0, f16))").unwrap(),
        "2048.0",
        "sequential f16 rounding: each add must round before the next"
    );
    assert_eq!(
        // chelis#732 P1 migration: shortest digits AT F16 WIDTH replace the
        // f64-image digits (0.0099945068359375); parse-back at f16 width
        // yields the same stored bits (the exhaustive observation lock).
        eval_expr("mul(cast(0.1, f16), cast(0.1, f16))").unwrap(),
        "0.009995",
        "correctly rounded f16 product of f16(0.1) with itself"
    );
    assert_eq!(
        eval_expr("mul(cast(65504.0, f16), cast(2.0, f16))").unwrap(),
        "inf",
        "f16 overflow must saturate to infinity, not keep a wider value"
    );
    assert_eq!(eval_expr("cast(2049.0, f16)").unwrap(), "2048.0");
}

/// bf16 sibling: mantissa is 8 bits, first non-representable integer is 257.
#[test]
fn eval_scalar_bf16_rounds_per_op() {
    assert_eq!(
        eval_expr("add(cast(256.0, bf16), cast(1.0, bf16))").unwrap(),
        // chelis#732 P1 migration: own-width Debug-grammar bf16 scalar.
        "256.0"
    );
    assert_eq!(
        eval_expr("add(cast(0.5, bf16), cast(0.25, bf16))").unwrap(),
        "0.75"
    );
    assert_eq!(eval_expr("cast(257.0, bf16)").unwrap(), "256.0");
}

/// The remaining f16/bf16 scalar op surface, distilled from the probe
/// batteries (`docs/investigations/probes/bat_narrow*.py`) into one
/// table-driven cross-lane row set. Every expected value is the locked
/// eval answer (correct IEEE narrow-float semantics); the C lane must
/// match it. Before Phase 3, C returned int64_t-truncated or unrounded
/// values for every row (1.25 -> 1, 0.333.. -> 0, -1.5 -> -1,
/// sqrt -> 1, exp -> 2, inf -> 131008, cast(2049) -> 2049).
#[test]
fn f16_bf16_scalar_op_surface_agrees_across_lanes() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let rows: &[(&str, &str, &str)] = &[
        ("sub(cast(1.5, f16), cast(0.25, f16))", "f16", "1.25"),
        ("div(cast(1.0, f16), cast(3.0, f16))", "f16", "0.3333"),
        ("neg(cast(1.5, f16))", "f16", "-1.5"),
        ("sqrt(cast(2.0, f16))", "f16", "1.414"),
        ("exp(cast(1.0, f16))", "f16", "2.719"),
        ("mul(cast(0.1, f16), cast(0.1, f16))", "f16", "0.009995"),
        ("mul(cast(65504.0, f16), cast(2.0, f16))", "f16", "inf"),
        ("cast(2049.0, f16)", "f16", "2048.0"),
        // [04-NUM-14] + section 5.6 position 4: an unsuffixed literal
        // adopts the cast target and rounds there once.  These values sit
        // immediately beyond a reduced-float midpoint while still rounding
        // to that midpoint in f32, so an f64 -> f32 -> reduced path selects
        // the wrong neighbor.
        ("cast(52847.99970178839, f16)", "f16", "52830.0"),
        ("cast(1.0039062500000002, bf16)", "bf16", "1.01"),
        // Negative parity: an explicit f32 suffix really does bind the
        // source at f32 first.  Do not repair the unsuffixed case by erasing
        // that source-width distinction.
        ("cast(52847.99970178839f32, f16)", "f16", "52860.0"),
        ("cast(1.0039062500000002f32, bf16)", "bf16", "1.0"),
        ("cast(cast(2049.0, f16), f32)", "f32", "2048.0"),
        ("mul(cast(0.1, bf16), cast(0.1, bf16))", "bf16", "0.01"),
        ("cast(257.0, bf16)", "bf16", "256.0"),
        ("tan(cast(0.0, f16))", "f16", "0.0"),
        ("atan(cast(0.0, f16))", "f16", "0.0"),
        ("floor(cast(1.5, f16))", "f16", "1.0"),
        ("ceil(cast(1.5, f16))", "f16", "2.0"),
        ("round(cast(1.5, f16))", "f16", "2.0"),
        ("recip(cast(4.0, f16))", "f16", "0.25"),
        ("max_elem(cast(1.5, f16), cast(0.25, f16))", "f16", "1.5"),
        ("min_elem(cast(1.5, f16), cast(0.25, f16))", "f16", "0.25"),
        ("tan(cast(0.0, bf16))", "bf16", "0.0"),
        ("atan(cast(0.0, bf16))", "bf16", "0.0"),
        ("floor(cast(1.5, bf16))", "bf16", "1.0"),
        ("ceil(cast(1.5, bf16))", "bf16", "2.0"),
        ("round(cast(1.5, bf16))", "bf16", "2.0"),
        ("recip(cast(4.0, bf16))", "bf16", "0.25"),
        ("max_elem(cast(1.5, bf16), cast(0.25, bf16))", "bf16", "1.5"),
        (
            "min_elem(cast(1.5, bf16), cast(0.25, bf16))",
            "bf16",
            "0.25",
        ),
    ];
    for (i, (expr, ret_ty, expected)) in rows.iter().enumerate() {
        let program =
            format!("module M.Main\ndef run() -> {ret_ty} = {expr}\nout = print(run())\n");
        let eval_got = eval_first_line(&program).expect("eval");
        // chelis#729 Phase 0: both lanes' printed values must be members
        // of the declared dtype's value set before any comparison.
        common::assert_elements_in_domain(ret_ty, &eval_got, expr);
        assert_eq!(
            eval_got, *expected,
            "eval drifted for `{expr}`; update the row"
        );
        let (_, stdout) = build_and_run_c(&program, &format!("f16_surface_{i}"))
            .expect("C lane should build and run");
        let c_got = stdout.lines().next().unwrap_or("").trim().to_string();
        common::assert_elements_in_domain(ret_ty, &c_got, expr);
        assert_eq!(c_got, *expected, "LANE DIVERGENCE for `{expr}`");
    }
}

/// [04-NUM-14]: a non-literal f64 host value narrows directly to the declared
/// reduced width. Literal-only coverage cannot reach this host-cast branch.
#[test]
fn c_nonliteral_f64_to_reduced_float_rounds_once() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    for (dtype, value, expected, name, helper) in [
        (
            "f16",
            "52847.99970178839",
            "52830.0",
            "f64_param_f16",
            "chelis_host_f64_to_f16",
        ),
        (
            "bf16",
            "1.0039062500000002",
            "1.01",
            "f64_param_bf16",
            "chelis_host_f64_to_bf16",
        ),
    ] {
        let program = format!(
            "module M.Main\ndef narrow(x: f64) -> {dtype} = cast(x, {dtype})\n\
             out = print(narrow(cast({value}, f64)))\n"
        );
        assert_eq!(eval_first_line(&program).expect("eval"), expected);
        let (emitted, stdout) = build_and_run_c(&program, name).expect("compiled lane");
        assert!(
            emitted.contains(helper),
            "missing direct f64 narrowing helper"
        );
        assert_eq!(stdout.lines().next().unwrap_or("").trim(), expected);
    }
}

/// **The eval scalar comparison rounds casts before comparing.** cast(2049.0,
/// f16) is 2048, so lt(2048, 2049) at f16 is FALSE. The compiled lane gets
/// this wrong (chelis#714, wrong-branch row below).
#[test]
fn eval_scalar_f16_comparison_rounds_before_compare() {
    assert_eq!(
        eval_first_line(
            "module M.Main\n\
             def run() -> bool = lt(cast(2048.0, f16), cast(2049.0, f16))\n\
             out = print(run())\n",
        )
        .unwrap(),
        "false"
    );
}

/// **HIP rejects unsupported f16/bf16 elementwise `Add` with a clean
/// diagnostic** - the correct row-three behavior from chelis#703's response
/// table, and the control that bounds chelis#689 (whose i64 siblings DO slip
/// through to F32 kernels).
/// Emission-only: no hipcc needed.
#[test]
fn hip_rejects_unsupported_f16_bf16_elementwise_add_cleanly() {
    for ty in ["f16", "bf16"] {
        let program = format!(
            "def f(a: tensor[4, {ty}], b: tensor[4, {ty}]) -> tensor[4, {ty}] = add(a, b)\n"
        );
        let (ok, stderr, _) = build_target(&program, &format!("hip_{ty}_add"), "hip");
        assert!(!ok, "HIP must reject unsupported {ty} elementwise Add");
        assert!(
            stderr.contains("narrow-float compute")
                && stderr.contains(&format!("`{ty}`"))
                && stderr.contains("unimplemented chelis#729")
                && stderr.contains("this operation has no typed HIP narrow-float kernel")
                && stderr.contains("spec/04-type-system.md §1.1.3"),
            "the rejection must be the specific narrow-float diagnostic, got: {stderr}"
        );
        for stale in [
            "planned",
            "HIP tensor load/store",
            "`BlasMatmul`",
            "[05-OP-43] ReLU identities",
            "spec/04-type-system.md §5.7.1",
        ] {
            assert!(
                !stderr.contains(stale),
                "the rejection must not repeat the capability matrix (`{stale}`): {stderr}"
            );
        }
    }
}

/// **Metal emits properly typed `half` / `bfloat` kernels** for rank-1
/// bf16/f16 - no F32 substitution, no zero placeholder. Emission-only.
#[test]
fn metal_emits_typed_half_and_bfloat_kernels() {
    let (ok, stderr, emitted) = build_target(
        "def f(a: tensor[4, f16], b: tensor[4, f16]) -> tensor[4, f16] = add(a, b)\n",
        "metal_f16_add",
        "metal",
    );
    assert!(ok, "Metal must admit rank-1 f16: {stderr}");
    assert!(
        emitted.contains("device const half*"),
        "the f16 kernel must be `half`-typed, not float"
    );

    let (ok, stderr, emitted) = build_target(
        "def f(a: tensor[4, bf16], b: tensor[4, bf16]) -> tensor[4, bf16] = add(a, b)\n",
        "metal_bf16_add",
        "metal",
    );
    assert!(ok, "Metal must admit rank-1 bf16: {stderr}");
    assert!(
        emitted.contains("device const bfloat*"),
        "the bf16 kernel must be `bfloat`-typed, not float"
    );
}

/// **f8e4m3 stays rejected in both lanes** (spec/04-type-system.md §1.1.1).
/// The deferred dtype is the one narrow float that does NOT hit a fallback.
#[test]
fn f8e4m3_is_rejected_in_both_lanes() {
    let err = eval_expr("cast(1.0, f8e4m3)").expect_err("eval must reject f8e4m3");
    assert!(
        err.contains("UnsupportedTensorPrecision") || err.contains("f8e4m3"),
        "rejection must name the precision problem, got: {err}"
    );
    let (ok, stderr, _) = build_target(
        "def run() -> f8e4m3 = cast(1.0, f8e4m3)\nout = run()\n",
        "f8_build",
        "c",
    );
    assert!(
        !ok && (stderr.contains("UnsupportedTensorPrecision") || stderr.contains("f8e4m3")),
        "build must reject f8e4m3, got ok={ok}, stderr: {stderr}"
    );
}

// The interim lock formerly here
// (`c_f16_tensor_print_aborts_with_dtype_id_instead_of_misreading`, itself
// the chelis#730 re-authoring of the
// `c_dag_kernels_compute_correct_f16_bits_despite_print` byte-decode lock)
// is RETIRED per its own instructions at chelis#732 Phase 2: the generated
// print helper renders f16/bf16 tensors faithfully, so the
// abort-at-the-print contract it pinned no longer exists, and the f16
// kernel-bits VALUE evidence flows through the direct print row
// (`c_print_of_f16_tensor_prints_f16_values`, un-ignored below). The
// helper's `default:` arm still aborts with the raw dtype id, but only for
// ids the RuntimeDType vocabulary does not define, which no buildable
// program can produce.

// ===========================================================================
// chelis#714 - f16/bf16 scalars in the compiled C lane
// ===========================================================================

/// Before Phase 3, C printed `0` because the value went through `int64_t`.
#[test]
fn f16_scalar_fraction_survives_compilation() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout) = build_and_run_c(
        "def run() -> f16 = add(cast(0.5, f16), cast(0.25, f16))\nout = print(run())\n",
        "f16_frac",
    )
    .expect("build and run");
    assert!(
        stdout.lines().next().unwrap_or("").trim() == "0.75",
        "f16 0.5 + 0.25 must print 0.75; got: {stdout}"
    );
}

/// Before Phase 3, C printed `0` for bf16 as well.
#[test]
fn bf16_scalar_fraction_survives_compilation() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout) = build_and_run_c(
        "def run() -> bf16 = add(cast(0.5, bf16), cast(0.25, bf16))\nout = print(run())\n",
        "bf16_frac",
    )
    .expect("build and run");
    assert!(
        stdout.lines().next().unwrap_or("").trim() == "0.75",
        "bf16 0.5 + 0.25 must print 0.75; got: {stdout}"
    );
}

/// Before Phase 3, C printed `2049` because the host lane did not finalize
/// f16 operations; eval correctly printed 2048.
#[test]
fn f16_scalar_add_boundary_agrees_across_lanes() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout) = build_and_run_c(
        "def run() -> f16 = add(cast(2048.0, f16), cast(1.0, f16))\nout = print(run())\n",
        "f16_boundary",
    )
    .expect("build and run");
    // chelis#729 Phase 0: 2049 is not an f16 value; the domain checker
    // catches this mechanically before the exact compare does.
    common::assert_elements_in_domain(
        "f16",
        stdout.lines().next().unwrap_or("").trim(),
        "f16_boundary",
    );
    assert_eq!(
        stdout.lines().next().unwrap_or("").trim(),
        "2048.0",
        "f16 2048 + 1 must round ties-to-even to 2048; got: {stdout}"
    );
}

/// Before Phase 3, C printed `true`. cast(2049.0, f16) is 2048, so the
/// comparison must be false - the compiled lane picks the wrong branch at a
/// threshold of 2049 instead of #680's 2^53.
#[test]
fn f16_scalar_comparison_rounds_in_compiled_lane() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout) = build_and_run_c(
        "def run() -> bool = lt(cast(2048.0, f16), cast(2049.0, f16))\nout = print(run())\n",
        "f16_lt",
    )
    .expect("build and run");
    assert!(
        stdout.lines().next().unwrap_or("").trim() == "false",
        "cast(2049.0, f16) rounds to 2048; lt(2048, 2048) is false; got: {stdout}"
    );
}

/// Before Phase 3, the emitted C did not compile (`void* __result =
/// fabs(...)` - clang: assigning to 'void *' from incompatible type
/// 'double'). The one #714 symptom that is loud, though as a toolchain error
/// rather than a diagnostic.
#[test]
fn f16_scalar_abs_compiles_and_runs() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout) = build_and_run_c(
        "def run() -> f16 = abs(cast(-1.5, f16))\nout = print(run())\n",
        "f16_abs",
    )
    .expect("chelis#714: emitted C must compile");
    assert!(
        stdout.lines().next().unwrap_or("").trim() == "1.5",
        "abs(-1.5f16) must print 1.5; got: {stdout}"
    );
}

// ===========================================================================
// chelis#716 - the C host boundaries around correct f16/bf16 kernels
// ===========================================================================

/// Green since chelis#732 Phase 2 (un-ignored per the plan's B2.3): the
/// generated print helper decodes f16 storage and formats at f16 width.
/// (The pre-fix helper printed `data=[0.0004898309707641602, 0.0]` - the
/// correct f16 buffer read as f32; the chelis#730 interim aborted.)
#[test]
fn c_print_of_f16_tensor_prints_f16_values() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout) = build_and_run_c(
        "def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f16] = add(cast(x, f16), cast(y, f16))\n\
         out = print(f(to_tensor([2048.0, 0.5]), to_tensor([1.0, 0.25])))\n",
        "f16_print",
    )
    .expect("build and run");
    common::assert_elements_in_domain(
        "f16",
        stdout.lines().next().unwrap_or("").trim(),
        "f16_print",
    );
    assert!(
        stdout.contains("data=[2048.0, 0.75]"),
        "the print helper must decode f16 elements; got: {stdout}"
    );
}

/// Green since chelis#732 Phase 2 (un-ignored per the plan's B2.3):
/// `chelis_list_from_tensor` reads the 2-byte f16 storage exactly and the
/// host ABI carries `list[f16]` as the boxed-element state. (The pre-fix
/// runtime aborted `to_list expects a supported numeric or bool tensor
/// input` from a build that succeeded.)
#[test]
fn c_to_list_of_f16_tensor_works() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout) = build_and_run_c(
        "def f(x: tensor[2, f32]) -> tensor[2, f16] = cast(x, f16)\n\
         out = print(to_list(f(to_tensor([2049.0, 0.75]))))\n",
        "f16_to_list",
    )
    .expect("chelis#716: to_list of an f16 tensor must not abort");
    common::assert_elements_in_domain(
        "f16",
        stdout.lines().next().unwrap_or("").trim(),
        "f16_to_list",
    );
    assert!(
        stdout.contains("2048") && stdout.contains("0.75"),
        "to_list must yield the f16 values; got: {stdout}"
    );
}

/// Before Phase 3 (re-verified at chelis#732 Phase 2 round 1), the build
/// rejected this typed literal at C-host ABI selection. The exact narrow
/// scalar carrier now reaches the runtime destination-dtype constructor, so
/// this is an ordinary positive ingress lock rather than an ignored exit row.
#[test]
fn c_f16_tensor_literal_constructs() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout) = build_and_run_c(
        "def f() -> tensor[2, f16] = to_tensor([cast(2049.0, f16), cast(0.75, f16)])\n\
         out = print(f())\n",
        "f16_literal",
    )
    .expect("chelis#714/[#729] ingress: a typed f16 to_tensor literal must construct");
    common::assert_elements_in_domain(
        "f16",
        stdout.lines().next().unwrap_or("").trim(),
        "f16_literal",
    );
    assert!(
        stdout.contains("data=[2048.0, 0.75]"),
        "the constructed f16 literal must round and print; got: {stdout}"
    );
}

/// Phase 3 host-ingress parity: a narrow scalar converted to a rank-0 tensor
/// keeps its exact storage width. This covers the scalar-to-tensor edge
/// separately from list-literal ingress without introducing a movement-op
/// extent boundary owned by chelis#1112.
#[test]
fn c_narrow_scalar_to_tensor_preserves_declared_width() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    for (dtype, value, expected, name) in [
        ("f16", "0.3333", "0.3333", "f16_scalar_to_tensor"),
        ("bf16", "0.334", "0.334", "bf16_scalar_to_tensor"),
    ] {
        let program = format!(
            "module M.Main\ndef run() -> tensor[{dtype}] = \
             scalar_to_tensor(cast({value}, {dtype}))\n\
             out = print(run())\n"
        );
        let eval = eval_first_line(&program).expect("eval lane");
        let (_, stdout) = build_and_run_c(&program, name).expect("compiled lane");
        let compiled = stdout.lines().next().unwrap_or("").trim();
        assert_eq!(eval, expected, "{dtype} eval scalar_to_tensor drift");
        assert_eq!(compiled, eval, "{dtype} scalar_to_tensor lane divergence");
    }
}

// ===========================================================================
// chelis#717 (narrow-float rows) - the eval tensor lane never rounds f16/bf16
// ===========================================================================

/// Before typed eval storage landed, eval printed `data=[2049.0, 0.75]` -
/// 2049.0 does not exist
/// in f16. The scalar evaluator gets the same computation right (locked
/// above), and the C DAG kernels get it right (bit-locked above); only the
/// eval tensor lane skips the rounding.
#[test]
fn eval_tensor_f16_add_rounds_to_f16() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f16], y: tensor[2, f16]) -> tensor[2, f16] = add(x, y)\n\
         out = print(f(to_tensor([cast(2048.0, f16), cast(0.5, f16)]), \
         to_tensor([cast(1.0, f16), cast(0.25, f16)])))\n",
    )
    .expect("eval should run");
    // chelis#729 Phase 0: 2049.0 in an f16 buffer is the mechanical
    // detection of this cell, independent of the exact-string assert.
    common::assert_elements_in_domain("f16", &line, "eval_f16_tensor_add");
    assert!(
        line.contains("data=[2048.0, 0.75]"),
        "f16 tensor add must round per-op like the scalar lane; got: {line}"
    );
}

/// bf16 sibling. Before typed eval storage landed, it printed
/// `data=[257.0, 0.75]`; 257 is not a bf16
/// value (mantissa 8 bits).
#[test]
fn eval_tensor_bf16_add_rounds_to_bf16() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, bf16], y: tensor[2, bf16]) -> tensor[2, bf16] = add(x, y)\n\
         out = print(f(to_tensor([cast(256.0, bf16), cast(0.5, bf16)]), \
         to_tensor([cast(1.0, bf16), cast(0.25, bf16)])))\n",
    )
    .expect("eval should run");
    common::assert_elements_in_domain("bf16", &line, "eval_bf16_tensor_add");
    assert!(
        line.contains("data=[256.0, 0.75]"),
        "bf16 tensor add must round per-op like the scalar lane; got: {line}"
    );
}

/// Even a bare tensor cast only narrows to f32, never to f16
/// (`crates/chelis-ir/src/eval.rs:142-146` routes Bf16|F16 through the f32
/// arm). Before typed eval storage landed, 2049.0 survived a cast to f16.
#[test]
fn eval_tensor_cast_to_f16_rounds() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32]) -> tensor[2, f16] = cast(x, f16)\n\
         out = print(f(to_tensor([2049.0, 0.75])))\n",
    )
    .expect("eval should run");
    common::assert_elements_in_domain("f16", &line, "eval_f16_tensor_cast");
    assert!(
        line.contains("data=[2048.0, 0.75]"),
        "a cast to f16 must apply f16 rounding; got: {line}"
    );
}
