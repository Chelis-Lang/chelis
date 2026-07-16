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
//! | chelis#714 | f16/bf16 SCALARS have no C-host-lane representation: they type as `HostType::Unknown`, arithmetic defaults to `int64_t`, and `add(0.5f16, 0.25f16)` compiles to a binary that prints `0` |
//! | chelis#716 | the C host boundaries for bf16/f16 TENSORS are broken: `print` reads the 2-byte buffers as f32 (garbage), `to_list`/`to_tensor` abort at runtime |
//! | chelis#717 | the EVAL tensor lane never rounds f16/bf16, so an f16 tensor holds 2049.0 (not an f16 value) |
//!
//! ## The part that works, and must keep working
//!
//! * the **eval scalar lane** rounds f16/bf16 per-op correctly (IEEE
//!   sequential rounding, locked below);
//! * the **C DAG kernels** compute correctly-rounded f16/bf16 values (WS-1) -
//!   proven here by decoding the misprinted bytes, so the lock survives the
//!   broken print path;
//! * **HIP rejects** f16/bf16 compute ops with a clean diagnostic and
//!   **Metal emits properly typed** `half`/`bfloat` kernels - the two
//!   backends that get it right;
//! * **f8e4m3 is rejected** by the checker in both lanes (spec §1.1.1).
//!
//! Precision boundaries: f16 mantissa 11 bits (first non-representable
//! integer 2049, max 65504); bf16 mantissa 8 bits (first non-representable
//! 257). Tiny thresholds, trivially reachable from ordinary ML code.
//!
//! Tests that assert correct behavior and fail today are `#[ignore]`d with
//! their issue number, observed wrong value, and run command.

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
        "2048"
    );
    assert_eq!(
        eval_expr("add(add(cast(2048.0, f16), cast(1.0, f16)), cast(1.0, f16))").unwrap(),
        "2048",
        "sequential f16 rounding: each add must round before the next"
    );
    assert_eq!(
        eval_expr("mul(cast(0.1, f16), cast(0.1, f16))").unwrap(),
        "0.0099945068359375",
        "correctly rounded f16 product of f16(0.1) with itself"
    );
    assert_eq!(
        eval_expr("mul(cast(65504.0, f16), cast(2.0, f16))").unwrap(),
        "inf",
        "f16 overflow must saturate to infinity, not keep a wider value"
    );
    assert_eq!(eval_expr("cast(2049.0, f16)").unwrap(), "2048");
}

/// bf16 sibling: mantissa is 8 bits, first non-representable integer is 257.
#[test]
fn eval_scalar_bf16_rounds_per_op() {
    assert_eq!(
        eval_expr("add(cast(256.0, bf16), cast(1.0, bf16))").unwrap(),
        "256"
    );
    assert_eq!(
        eval_expr("add(cast(0.5, bf16), cast(0.25, bf16))").unwrap(),
        "0.75"
    );
    assert_eq!(eval_expr("cast(257.0, bf16)").unwrap(), "256");
}

/// The remaining f16/bf16 scalar op surface, distilled from the probe
/// batteries (`docs/investigations/probes/bat_narrow*.py`) into one
/// table-driven cross-lane row set. Every expected value is the locked
/// eval answer (correct IEEE narrow-float semantics); the C lane must
/// match it. Observed today: C returns int64_t-truncated or unrounded
/// values for every row (1.25 -> 1, 0.333.. -> 0, -1.5 -> -1,
/// sqrt -> 1, exp -> 2, inf -> 131008, cast(2049) -> 2049).
#[test]
#[ignore = "chelis#714: the full f16/bf16 scalar op surface diverges in the compiled lane \
            (int64_t storage, no narrow-float rounding). Each row's expected value is the \
            locked eval answer. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
fn f16_bf16_scalar_op_surface_agrees_across_lanes() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let rows: &[(&str, &str, &str)] = &[
        ("sub(cast(1.5, f16), cast(0.25, f16))", "f16", "1.25"),
        (
            "div(cast(1.0, f16), cast(3.0, f16))",
            "f16",
            "0.333251953125",
        ),
        ("neg(cast(1.5, f16))", "f16", "-1.5"),
        ("sqrt(cast(2.0, f16))", "f16", "1.4140625"),
        ("exp(cast(1.0, f16))", "f16", "2.71875"),
        (
            "mul(cast(0.1, f16), cast(0.1, f16))",
            "f16",
            "0.0099945068359375",
        ),
        ("mul(cast(65504.0, f16), cast(2.0, f16))", "f16", "inf"),
        ("cast(2049.0, f16)", "f16", "2048"),
        ("cast(cast(2049.0, f16), f32)", "f32", "2048"),
        (
            "mul(cast(0.1, bf16), cast(0.1, bf16))",
            "bf16",
            "0.010009765625",
        ),
        ("cast(257.0, bf16)", "bf16", "256"),
    ];
    for (i, (expr, ret_ty, expected)) in rows.iter().enumerate() {
        let program =
            format!("module M.Main\ndef run() -> {ret_ty} = {expr}\nout = print(run())\n");
        assert_eq!(
            eval_first_line(&program).expect("eval"),
            *expected,
            "eval drifted for `{expr}`; update the row"
        );
        let (_, stdout) = build_and_run_c(&program, &format!("f16_surface_{i}"))
            .expect("C lane should build and run");
        assert_eq!(
            stdout.lines().next().unwrap_or("").trim(),
            *expected,
            "LANE DIVERGENCE for `{expr}`"
        );
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

/// **HIP rejects f16/bf16 compute ops with a clean diagnostic** - the correct
/// row-three behavior from chelis#703's response table, and the control that
/// bounds chelis#689 (whose int64 siblings DO slip through to F32 kernels).
/// Emission-only: no hipcc needed.
#[test]
fn hip_rejects_f16_bf16_compute_ops_cleanly() {
    for ty in ["f16", "bf16"] {
        let program = format!(
            "def f(a: tensor[4, {ty}], b: tensor[4, {ty}]) -> tensor[4, {ty}] = add(a, b)\n"
        );
        let (ok, stderr, _) = build_target(&program, &format!("hip_{ty}_add"), "hip");
        assert!(!ok, "HIP must reject {ty} compute ops today");
        assert!(
            stderr.contains("admits") && stderr.contains("only on tensor load/store"),
            "the rejection must be the specific narrow-float diagnostic, got: {stderr}"
        );
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

/// **The C DAG kernels compute correct f16 bits** - locked by decoding the
/// bytes the broken print helper emits (chelis#716). The printed garbage
/// `0.0004898309707641602` IS the proof the kernels are right: its f32 bit
/// pattern is 0x3A006800, which is the two correct f16 results 0x6800
/// (= 2048.0, the correctly rounded f16 sum of 2048 + 1) and 0x3A00 (= 0.75)
/// concatenated little-endian.
///
/// Compares BIT PATTERNS, not decimal text (a grep for a decimal is how the
/// chelis#711 probe went wrong). When #716 is fixed this test fails loudly at
/// the misread assertion and should be replaced by the direct-print row below.
#[test]
fn c_dag_kernels_compute_correct_f16_bits_despite_print() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let (_, stdout) = build_and_run_c(
        "def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f16] = add(cast(x, f16), cast(y, f16))\n\
         out = print(f(to_tensor([2048.0, 0.5]), to_tensor([1.0, 0.25])))\n",
        "f16_kernel_bits",
    )
    .expect("build and run");
    let line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("tensor(shape="))
        .unwrap_or_else(|| panic!("no tensor line in:\n{stdout}"));
    let start = line.find("data=[").expect("data marker") + "data=[".len();
    let end = start + line[start..].find([',', ']']).expect("delimiter");
    let misread: f64 = line[start..end].trim().parse().expect("numeric payload");
    let bits = (misread as f32).to_bits();
    assert_eq!(
        bits, 0x3A00_6800,
        "the misprinted f32 must carry the two CORRECT f16 results \
         (0x6800 = 2048.0 correctly rounded, 0x3A00 = 0.75); if this fails \
         either the kernels regressed or chelis#716 was fixed - if the \
         latter, replace this lock with c_print_of_f16_tensor_prints_f16_values"
    );
}

// ===========================================================================
// chelis#714 - f16/bf16 scalars in the compiled C lane
// ===========================================================================

/// Observed today: C prints `0` (the value went through `int64_t`).
#[test]
#[ignore = "chelis#714: add(0.5f16, 0.25f16) compiles to a binary that prints 0 (int64_t \
            storage truncates the fraction); eval correctly prints 0.75. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
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

/// Observed today: C prints `0` for bf16 as well.
#[test]
#[ignore = "chelis#714: add(0.5bf16, 0.25bf16) compiles to a binary that prints 0; eval \
            correctly prints 0.75. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
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

/// Observed today: C prints `2049` (no f16 rounding anywhere in the host
/// lane); eval correctly prints 2048.
#[test]
#[ignore = "chelis#714: add(2048f16, 1f16) compiles to 2049 (not an f16 value); eval \
            correctly prints 2048. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
fn f16_scalar_add_boundary_agrees_across_lanes() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout) = build_and_run_c(
        "def run() -> f16 = add(cast(2048.0, f16), cast(1.0, f16))\nout = print(run())\n",
        "f16_boundary",
    )
    .expect("build and run");
    assert!(
        stdout.lines().next().unwrap_or("").trim() == "2048",
        "f16 2048 + 1 must round ties-to-even to 2048; got: {stdout}"
    );
}

/// Observed today: C prints `true`. cast(2049.0, f16) is 2048, so the
/// comparison must be false - the compiled lane picks the wrong branch at a
/// threshold of 2049 instead of #680's 2^53.
#[test]
#[ignore = "chelis#714: lt(2048f16, 2049f16) compiles to true (the casts never round, so \
            2048 < 2049); correct IEEE answer is false, and eval agrees. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
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

/// Observed today: the emitted C does not even compile (`void* __result =
/// fabs(...)` - clang: assigning to 'void *' from incompatible type
/// 'double'). The one #714 symptom that is loud, though as a toolchain error
/// rather than a diagnostic.
#[test]
#[ignore = "chelis#714: abs(cast(-1.5, f16)) emits C that fails to compile (void* __result \
            = fabs(...)). This test asserts it builds, runs, and prints 1.5. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
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

/// Observed today: prints `data=[0.0004898309707641602, 0.0]` - the correct
/// f16 buffer read as f32 by the print helper's `default:` arm
/// (host_emit.rs:512).
#[test]
#[ignore = "chelis#716: the emitted print helper reads f16 buffers as f32 and prints \
            0.0004898309707641602 instead of 2048.0, 0.75. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
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
    assert!(
        stdout.contains("data=[2048.0, 0.75]"),
        "the print helper must decode f16 elements; got: {stdout}"
    );
}

/// Observed today: runtime abort `to_list expects numeric or bool tensor
/// input` (chelis-runtime lib.rs:2296). Loud, but from a build that succeeded.
#[test]
#[ignore = "chelis#716: to_list of an f16 tensor aborts at runtime in the compiled lane; \
            eval prints [2048, 0.75]. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
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
    assert!(
        stdout.contains("2048") && stdout.contains("0.75"),
        "to_list must yield the f16 values; got: {stdout}"
    );
}

/// Observed today: runtime abort `to_tensor: unsupported destination dtype`
/// (chelis-runtime lib.rs:2182). The build gate admits the program
/// (cli_admits_bf16_f16_target_c.rs locks that), then the runtime cannot
/// construct the value the program starts from.
#[test]
#[ignore = "chelis#716: an f16 to_tensor literal aborts at runtime in the compiled lane; \
            eval evaluates it fine. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
fn c_f16_tensor_literal_constructs() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout) = build_and_run_c(
        "def f() -> tensor[2, f16] = to_tensor([cast(2049.0, f16), cast(0.75, f16)])\n\
         out = print(f())\n",
        "f16_literal",
    )
    .expect("chelis#716: an f16 tensor literal must be constructible at runtime");
    assert!(
        stdout.contains("data=[2048.0, 0.75]"),
        "the constructed f16 literal must round and print; got: {stdout}"
    );
}

// ===========================================================================
// chelis#717 (narrow-float rows) - the eval tensor lane never rounds f16/bf16
// ===========================================================================

/// Observed today: eval prints `data=[2049.0, 0.75]` - 2049.0 does not exist
/// in f16. The scalar evaluator gets the same computation right (locked
/// above), and the C DAG kernels get it right (bit-locked above); only the
/// eval tensor lane skips the rounding.
#[test]
#[ignore = "chelis#717: eval f16 tensor add produces 2049.0, which is not representable in \
            f16; correct is 2048.0. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
fn eval_tensor_f16_add_rounds_to_f16() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f16], y: tensor[2, f16]) -> tensor[2, f16] = add(x, y)\n\
         out = print(f(to_tensor([cast(2048.0, f16), cast(0.5, f16)]), \
         to_tensor([cast(1.0, f16), cast(0.25, f16)])))\n",
    )
    .expect("eval should run");
    assert!(
        line.contains("data=[2048.0, 0.75]"),
        "f16 tensor add must round per-op like the scalar lane; got: {line}"
    );
}

/// bf16 sibling. Observed today: `data=[257.0, 0.75]`; 257 is not a bf16
/// value (mantissa 8 bits).
#[test]
#[ignore = "chelis#717: eval bf16 tensor add produces 257.0, not representable in bf16; \
            correct is 256.0. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
fn eval_tensor_bf16_add_rounds_to_bf16() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, bf16], y: tensor[2, bf16]) -> tensor[2, bf16] = add(x, y)\n\
         out = print(f(to_tensor([cast(256.0, bf16), cast(0.5, bf16)]), \
         to_tensor([cast(1.0, bf16), cast(0.25, bf16)])))\n",
    )
    .expect("eval should run");
    assert!(
        line.contains("data=[256.0, 0.75]"),
        "bf16 tensor add must round per-op like the scalar lane; got: {line}"
    );
}

/// Even a bare tensor cast only narrows to f32, never to f16
/// (`crates/chelis-ir/src/eval.rs:142-146` routes Bf16|F16 through the f32
/// arm). Observed today: 2049.0 survives a cast to f16.
#[test]
#[ignore = "chelis#717: eval tensor cast(x, f16) leaves 2049.0 in the tensor (narrows only \
            to f32); correct is 2048.0. Run with \
            `cargo test -p chelis-cli --test narrow_dtype_matrix -- --ignored`."]
fn eval_tensor_cast_to_f16_rounds() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32]) -> tensor[2, f16] = cast(x, f16)\n\
         out = print(f(to_tensor([2049.0, 0.75])))\n",
    )
    .expect("eval should run");
    assert!(
        line.contains("data=[2048.0, 0.75]"),
        "a cast to f16 must apply f16 rounding; got: {line}"
    );
}
