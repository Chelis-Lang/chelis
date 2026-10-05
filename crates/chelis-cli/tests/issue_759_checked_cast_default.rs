//! chelis#759 / chelis#729 rework - the CHECKED cast is the default on
//! every eval surface (spec/04-type-system.md section 5.2, authored at
//! the PR #857 rework; `spec/design/dtype_semantics.md` section C3's
//! cast ladder).
//!
//! One authored rule per direction, identical on the scalar and tensor
//! surfaces:
//!
//! * float -> int: require an integral finite value, then the width check. Out of
//!   range TRAPS `overflow` (pre-rework the tensor surface SATURATED:
//!   `cast(300.0, i8)` was `127`).
//! * int -> narrower int: exact value or `overflow` trap (pre-rework
//!   the tensor surface WRAPPED: i32 `300 -> i8` was `44`).
//! * NaN / inf -> int: `domain` trap.
//! * anything -> bool: STRICT {0, 1} membership or `domain` trap
//!   (pre-rework the tensor surface encoded nonzero-to-1:
//!   `cast(2, bool)` was `true`). Scalar->bool now WORKS under the
//!   same rule (pre-rework it was a loud "unsupported cast" hole).
//! * any -> float: IEEE RNE finalize at the target width; overflow is
//!   the correctly signed infinity, never a trap ([04-NUM-2]).
//!
//! The named lossy/wrapping cast forms remain chelis#759's future surface.
//! The compiled C lane adopted the checked ladder in chelis#729 Phase 3;
//! the former red rows below are unconditional cross-lane regressions.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// Run a printed expression through `chelis eval --file`; return the
/// first printed line verbatim, or the stderr on failure.
fn eval_lane_str(expr: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    write_file(&path, &format!("module M.Main\nout = print({expr})\n"));
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

/// Eval a whole program (for `if`-shaped fold probes); first stdout
/// line on success, stderr on failure.
fn eval_program(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
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

/// The expression must trap with the branded numeric-trap diagnostic
/// carrying `kind` and `prim` (section C2 shape: `numeric trap: <kind>
/// in <op> at <prim>`).
fn assert_cast_traps(expr: &str, kind: &str, prim: &str, label: &str) {
    match eval_lane_str(expr) {
        Ok(v) => panic!(
            "{label}: `{expr}` must trap ({kind} at {prim}) under the checked \
             cast default (spec/04 section 5.2, chelis#759), but it succeeded \
             and returned {v}"
        ),
        Err(stderr) => {
            assert!(
                stderr.contains(&format!("numeric trap: {kind}")) && stderr.contains(prim),
                "{label}: `{expr}` failed but without the branded {kind}-at-{prim} \
                 diagnostic. Got: {stderr}"
            );
        }
    }
}

// ===========================================================================
// float -> int: integral-and-in-range or trap.
// ===========================================================================

#[test]
fn tensor_fractional_float_to_int_traps_domain() {
    assert_cast_traps(
        "cast(to_tensor([3.5, -3.5]), i8)",
        "domain",
        "i8",
        "tensor_fractional_f2i",
    );
}

/// [04-NUM-9]: the trap renders byte-identically in every lane, so the
/// trap line is the LAST line of the failure on eval and on compiled C, with
/// no lane-only advice after it. The named lossy forms (`cast_trunc`,
/// `cast_saturate`, `cast_wrap`) are documented in the book's cast section,
/// not taught from a trap.
#[test]
fn scalar_fractional_float_to_int_traps_domain() {
    for prim in ["i8", "i64"] {
        let trap = format!("numeric trap: domain in cast at {prim}");
        let stderr =
            eval_lane_str(&format!("cast(3.5, {prim})")).expect_err("fractional cast must trap");
        assert_eq!(
            stderr.lines().last(),
            Some(trap.as_str()),
            "eval ends on the trap line: {stderr}"
        );
        if c_toolchain_available() {
            let program = format!(
                "module M.Main\ndef f(x: f64) -> {prim} = cast(x, {prim})\nout = print(f(3.5f64))\n"
            );
            let (_, c_stderr, ok) =
                c_lane_run(&program, &format!("fractional_{prim}")).expect("C lane builds");
            assert!(!ok, "compiled fractional cast must trap: {c_stderr}");
            assert_eq!(
                c_stderr.lines().last(),
                Some(trap.as_str()),
                "compiled C ends on the same trap line: {c_stderr}"
            );
        }
        // The passing twin: an integral value casts exactly.
        assert_eq!(
            eval_lane_str(&format!("cast(3.0, {prim})")).expect("integral cast"),
            "3"
        );
    }
}
#[test]
fn integral_float_to_int_casts_exactly_on_both_eval_surfaces() {
    let tensor = eval_lane_str("cast(to_tensor([3.0, -3.0]), i8)").expect("integral tensor");
    assert_eq!(tensor, "tensor(shape=[2], data=[3, -3])");
    assert_eq!(
        eval_lane_str("cast(3.0, i8)").expect("integral scalar"),
        "3"
    );
}

#[test]
fn tensor_float_to_int_out_of_range_traps_overflow_not_saturate() {
    // Pre-rework: saturated to 127 on this surface.
    assert_cast_traps(
        "cast(to_tensor([300.0]), i8)",
        "overflow",
        "i8",
        "tensor_f2i_overflow",
    );
}

#[test]
fn scalar_float_to_int_out_of_range_traps_overflow() {
    assert_cast_traps("cast(300.0, i8)", "overflow", "i8", "scalar_f2i_overflow");
}

#[test]
fn tensor_non_finite_to_int_traps_domain() {
    // sqrt(-1.0) is NaN; div(1.0, 0.0) is +inf (IEEE, [04-NUM-2]).
    assert_cast_traps(
        "cast(sqrt(to_tensor([-1.0])), i32)",
        "domain",
        "i32",
        "tensor_nan_to_int",
    );
    assert_cast_traps(
        "cast(div(to_tensor([1.0]), to_tensor([0.0])), i32)",
        "domain",
        "i32",
        "tensor_inf_to_int",
    );
}

// ===========================================================================
// int -> narrower int: exact in range; overflow trap out of range (no wrap).
// ===========================================================================

#[test]
fn tensor_int_narrowing_in_range_is_exact() {
    let got = eval_lane_str("cast(to_tensor([127, -128, 0]), i8)").expect("in-range eval");
    assert_eq!(got, "tensor(shape=[3], data=[127, -128, 0])");
}

#[test]
fn tensor_int_narrowing_out_of_range_traps_overflow_not_wrap() {
    // Pre-rework: wrapped two's-complement to 44 on this surface.
    assert_cast_traps(
        "cast(to_tensor([300]), i8)",
        "overflow",
        "i8",
        "tensor_i2i_overflow",
    );
}

#[test]
fn scalar_int_narrowing_out_of_range_traps_overflow() {
    assert_cast_traps(
        "cast(cast(300, i32), i8)",
        "overflow",
        "i8",
        "scalar_i2i_overflow",
    );
}

// ===========================================================================
// -> bool: strict {0, 1} membership on both surfaces.
// ===========================================================================

#[test]
fn tensor_cast_to_bool_accepts_exact_zero_one_only() {
    let got = eval_lane_str("cast(to_tensor([1, 0]), bool)").expect("0/1 eval");
    assert_eq!(got, "tensor(shape=[2], data=[true, false])");
}

#[test]
fn tensor_cast_to_bool_rejects_nonzero_nonone_with_domain_trap() {
    // Pre-rework: nonzero encoded true on this surface (cast(2, bool) = true).
    assert_cast_traps(
        "cast(to_tensor([2]), bool)",
        "domain",
        "bool",
        "tensor_to_bool_strict",
    );
}

#[test]
fn scalar_cast_to_bool_works_under_the_strict_rule() {
    // Pre-rework the scalar surface had NO bool arm (loud "unsupported
    // cast"); the unified ladder gives it the same strict rule.
    assert_eq!(
        eval_lane_str("cast(1, bool)").expect("cast(1, bool)"),
        "true"
    );
    assert_eq!(
        eval_lane_str("cast(0, bool)").expect("cast(0, bool)"),
        "false"
    );
    assert_cast_traps("cast(2, bool)", "domain", "bool", "scalar_to_bool_strict");
    assert_cast_traps(
        "cast(0.5, bool)",
        "domain",
        "bool",
        "scalar_fractional_to_bool",
    );
}

// ===========================================================================
// -> float: finalize, never trap (negative parity for the trap rules).
// ===========================================================================

#[test]
fn casts_to_float_targets_finalize_and_never_trap() {
    // f16 overflow goes to inf per IEEE ([04-NUM-2]), not a trap.
    assert_eq!(
        eval_lane_str("cast(to_tensor([1000000.0]), f16)").expect("f16 overflow eval"),
        "tensor(shape=[1], data=[inf])"
    );
    // i64 above 2^53 to f64 is the lossy-by-design direction ([04-NUM-6]).
    assert_eq!(
        eval_lane_str("cast(9007199254740993i64, f64)").expect("i64->f64"),
        "9007199254740992.0"
    );
    // i32 above 2^24 similarly rounds at f32 width. This is a
    // by-design lossy direction, not a checked-cast trap.
    assert_eq!(
        eval_lane_str("cast(16777217i32, f32)").expect("i32->f32"),
        "16777216.0"
    );
}

#[test]
fn int64_to_bf16_rounds_once_at_the_target_width_on_both_eval_surfaces() {
    // This exact integer is one above a bf16 midpoint but first becomes the
    // midpoint if it is rounded through f64. Direct target-width RNE must
    // therefore choose the upper bf16 value ([04-NUM-14]).
    const ABOVE_MIDPOINT: &str = "4629700416936869889i64";
    assert_eq!(
        eval_lane_str(&format!("cast({ABOVE_MIDPOINT}, bf16)")).expect("scalar i64->bf16"),
        "4.65e18"
    );
    assert_eq!(
        eval_lane_str(&format!("cast(to_tensor([{ABOVE_MIDPOINT}]), bf16)"))
            .expect("tensor i64->bf16"),
        "tensor(shape=[1], data=[4.65e18])"
    );
}

// ===========================================================================
// The fold rule: a trapping cast DECLINES TO FOLD (section C2's decline
// clause); the condition falls to runtime and traps loudly there. A fold
// must never bake the trap away into a taken-or-deleted branch.
// ===========================================================================

#[test]
fn static_if_with_trapping_cast_condition_traps_at_runtime_not_folds() {
    let program = "def pick() -> f32 = if lt(cast(300.0, i8), 10i8) \
                   then 111.0 else 222.0\nout = print(pick())\n";
    match eval_program(program) {
        Ok(v) => panic!(
            "a compile-time out-of-range cast condition must not fold to a \
             branch; the checked ladder declines and the runtime trap fires. \
             Got a folded/evaluated answer: {v}"
        ),
        Err(stderr) => assert!(
            stderr.contains("numeric trap: overflow") && stderr.contains("i8"),
            "expected the branded overflow trap from the unfolded runtime \
             condition; got: {stderr}"
        ),
    }
}

// ===========================================================================
// Compiled C lane: chelis#729 Phase 3 checked-cast regressions. These retain
// the original assertions that were red before the compiled lane adopted the
// same checked ladder as eval (dtype_semantics.md section C3).
// ===========================================================================

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn c_lane_build(program: &str, name: &str) -> Result<(TempDir, std::path::PathBuf), String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let binary = out_dir.join(name);
    Ok((dir, binary))
}

/// Build + link + run a program through the C lane; return
/// `(stdout, stderr, ok)`.
fn c_lane_run(program: &str, name: &str) -> Result<(String, String, bool), String> {
    let (_dir, binary) = c_lane_build(program, name)?;
    let run = std::process::Command::new(binary)
        .output()
        .expect("compiled binary should run");
    Ok((
        String::from_utf8_lossy(&run.stdout).into_owned(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
        run.status.success(),
    ))
}

fn assert_c_trap_at_thread_counts(program: &str, expected: &str, name: &str) {
    let (_dir, binary) = c_lane_build(program, name).expect("C lane build");
    for threads in [1, 2, 4, 8] {
        let run = std::process::Command::new(&binary)
            .env("OMP_NUM_THREADS", threads.to_string())
            .output()
            .expect("compiled binary should run");
        let stdout = String::from_utf8_lossy(&run.stdout);
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            !run.status.success() && stderr.contains(expected),
            "{name} with OMP_NUM_THREADS={threads} must select `{expected}`; \
             status={} stdout={stdout} stderr={stderr}",
            run.status
        );
    }
}

const ACTIVE_CAST_PRIMS: [&str; 9] = [
    "f64", "f32", "f16", "bf16", "i8", "i16", "i32", "i64", "bool",
];

fn safe_scalar_at(source: &str) -> String {
    match source {
        "f64" | "f32" | "f16" | "bf16" => format!("cast(1.0, {source})"),
        "i8" | "i16" | "i32" | "i64" => format!("cast(1, {source})"),
        "bool" => "true".to_string(),
        other => panic!("matrix contains an unknown source dtype {other}"),
    }
}

fn checked_cast_product_program() -> String {
    let mut definitions = String::new();
    let mut outputs = String::new();
    for source in ACTIVE_CAST_PRIMS {
        for target in ACTIVE_CAST_PRIMS {
            let scalar = format!("checked_scalar_{source}_to_{target}");
            let tensor_dag = format!("checked_tensor_dag_{source}_to_{target}");
            let tensor_host = format!("checked_tensor_host_{source}_to_{target}");
            let source_value = safe_scalar_at(source);
            definitions.push_str(&format!(
                "def {scalar}(x: {source}) -> {target} = cast(x, {target})\n"
            ));
            definitions.push_str(&format!(
                "def {tensor_dag}(x: tensor[1, {source}]) -> tensor[1, {target}] = \
                 cast(x, {target})\n"
            ));
            definitions.push_str(&format!(
                "def {tensor_host}() -> tensor[1, {target}] = \
                 cast(to_tensor([{source_value}]), {target})\n"
            ));
            outputs.push_str(&format!("out_{scalar} = {scalar}({source_value})\n"));
            outputs.push_str(&format!(
                "out_{tensor_dag} = {tensor_dag}(reshape(to_tensor([{source_value}]), \
                 [cast(1, i64)]))\n"
            ));
            outputs.push_str(&format!("out_{tensor_host} = {tensor_host}()\n"));
        }
    }
    format!("{definitions}{outputs}")
}

fn assert_checked_cast_product_observations(stdout: &str, lane: &str) {
    let lines = stdout.lines().collect::<chelis_unord::UnordSet<_>>();
    for source in ACTIVE_CAST_PRIMS {
        for target in ACTIVE_CAST_PRIMS {
            let scalar_value = match target {
                "f64" | "f32" | "f16" | "bf16" => "1.0",
                "i8" | "i16" | "i32" | "i64" => "1",
                "bool" => "true",
                other => panic!("matrix contains an unknown target dtype {other}"),
            };
            let tensor_value = format!("tensor(shape=[1], data=[{scalar_value}])");
            for (surface, expected) in [
                ("scalar", scalar_value.to_string()),
                ("tensor_dag", tensor_value.clone()),
                ("tensor_host", tensor_value.clone()),
            ] {
                let root = format!("out_checked_{surface}_{source}_to_{target} = {expected}");
                assert!(
                    lines.contains(root.as_str()),
                    "{lane} product is missing exact observation `{root}`; matching root line: {:?}",
                    stdout.lines().find(|line| line
                        .contains(&format!("out_checked_{surface}_{source}_to_{target}")))
                );
            }
            let automatic = format!("checked_tensor_host_{source}_to_{target} = {tensor_value}");
            assert!(
                lines.contains(automatic.as_str()),
                "{lane} product is missing the automatic pure-nullary observation `{automatic}`"
            );
        }
    }
}

#[test]
fn generated_checked_cast_product_is_positive_on_every_active_pair_and_surface() {
    let program = checked_cast_product_program();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("checked_cast_product.ch");
    write_file(&path, &program);
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("checked-cast product eval should run");
    assert!(
        eval.status.success(),
        "the generated scalar/tensor eval product must succeed; stderr={} ",
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval_stdout = String::from_utf8_lossy(&eval.stdout);
    assert_eq!(
        eval_stdout.lines().count(),
        ACTIVE_CAST_PRIMS.len() * ACTIVE_CAST_PRIMS.len() * 4,
        "every generated eval cell must reach root observation"
    );
    assert_checked_cast_product_observations(&eval_stdout, "eval");

    let (stdout, stderr, ok) = c_lane_run(&program, "checked_cast_product").expect("C product");
    assert!(ok, "the generated C product must succeed; stderr={stderr}");
    assert_eq!(
        stdout.lines().count(),
        ACTIVE_CAST_PRIMS.len() * ACTIVE_CAST_PRIMS.len() * 4,
        "every generated C DAG/host cell must reach root observation"
    );
    assert_checked_cast_product_observations(&stdout, "compiled C");
}

#[test]
fn compiled_checked_casts_round_directly_at_reduced_float_width() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }

    let cases = [
        ("f64_bf16", "f64", "bf16", "1.0039062500000002f64", "1.01"),
        ("f64_f16", "f64", "f16", "52847.99970178839f64", "52830.0"),
        (
            "int64_bf16",
            "i64",
            "bf16",
            "4629700416936869889i64",
            "4.65e18",
        ),
    ];
    let mut definitions = String::new();
    let mut outputs = String::new();
    let mut expected = Vec::new();
    for (label, source, target, value, rendered) in cases {
        definitions.push_str(&format!(
            "def {label}_scalar(x: {source}) -> {target} = cast(x, {target})\n\
             def {label}_tensor_dag(x: tensor[1, {source}]) -> tensor[1, {target}] = \
             cast(x, {target})\n\
             def {label}_tensor_host() -> tensor[1, {target}] = \
             cast(to_tensor([{value}]), {target})\n"
        ));
        outputs.push_str(&format!(
            "out_{label}_scalar = {label}_scalar({value})\n\
             out_{label}_tensor_dag = {label}_tensor_dag(\
             reshape(to_tensor([{value}]), [cast(1, i64)]))\n\
             out_{label}_tensor_host = {label}_tensor_host()\n"
        ));
        expected.extend([
            format!("out_{label}_scalar = {rendered}"),
            format!("out_{label}_tensor_dag = tensor(shape=[1], data=[{rendered}])"),
            format!("out_{label}_tensor_host = tensor(shape=[1], data=[{rendered}])"),
        ]);
    }

    let program = format!("{definitions}{outputs}");
    let (stdout, stderr, ok) =
        c_lane_run(&program, "checked_cast_direct_rounding").expect("compiled rounding probe");
    assert!(
        ok,
        "direct target-width rounding probe must compile and run; stderr={stderr}"
    );
    let observed = stdout.lines().collect::<chelis_unord::UnordSet<_>>();
    for line in expected {
        assert!(
            observed.contains(line.as_str()),
            "compiled checked-cast output is missing `{line}`; matching output:\n{stdout}"
        );
    }
}

#[test]
fn c_tensor_float_to_int_out_of_range_traps_overflow() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "def f() -> tensor[1, i8] = cast(to_tensor([300.0]), i8)\n\
                   out = print(f())\n";
    let (stdout, stderr, ok) = c_lane_run(program, "c_checked_cast_f2i").expect("C lane");
    assert!(
        !ok && stderr.contains("overflow"),
        "compiled cast(300.0, i8) must trap with the branded overflow \
         diagnostic (chelis#729 Phase 3); got ok={ok} stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn c_tensor_fractional_float_to_int_traps_domain() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "def f() -> tensor[1, i8] = cast(to_tensor([3.5]), i8)\n\
                   out = print(f())\n";
    let (stdout, stderr, ok) =
        c_lane_run(program, "c_checked_cast_fractional_f2i").expect("C lane");
    assert!(
        !ok && stderr.contains("domain"),
        "compiled fractional float->int cast must Domain-trap (chelis#729 Phase 3); \
         got ok={ok} stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn c_tensor_int_narrowing_out_of_range_traps_overflow() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "def f() -> tensor[1, i8] = cast(to_tensor([300]), i8)\n\
                   out = print(f())\n";
    let (stdout, stderr, ok) = c_lane_run(program, "c_checked_cast_i2i").expect("C lane");
    assert!(
        !ok && stderr.contains("overflow"),
        "compiled i32->i8 out-of-range cast must trap (chelis#729 Phase 3); \
         got ok={ok} stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn mixed_offenders_select_the_lowest_flat_index_in_eval() {
    assert_cast_traps(
        "cast(to_tensor([300.0, sqrt(-1.0)]), i8)",
        "overflow",
        "i8",
        "eval_overflow_before_domain",
    );
    assert_cast_traps(
        "cast(to_tensor([sqrt(-1.0), 300.0]), i8)",
        "domain",
        "i8",
        "eval_domain_before_overflow",
    );
}

#[test]
fn c_dag_mixed_offenders_select_the_lowest_flat_index_at_multiple_thread_counts() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    for (values, expected, name) in [
        (
            "300.0, sqrt(-1.0)",
            "numeric trap: overflow in cast at i8",
            "c_dag_overflow_before_domain",
        ),
        (
            "sqrt(-1.0), 300.0",
            "numeric trap: domain in cast at i8",
            "c_dag_domain_before_overflow",
        ),
    ] {
        let program = format!(
            "def f(x: tensor[2, f32]) -> tensor[2, i8] = cast(x, i8)\n\
             out = print(f(reshape(to_tensor([{values}]), [cast(2, i64)])))\n"
        );
        assert_c_trap_at_thread_counts(&program, expected, name);
    }
}

#[test]
fn c_host_mixed_offenders_convert_instead_of_reinterpreting_and_select_lowest_index() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    for (values, expected, name) in [
        (
            "300.0, sqrt(-1.0)",
            "numeric trap: overflow in cast at i8",
            "c_host_overflow_before_domain",
        ),
        (
            "sqrt(-1.0), 300.0",
            "numeric trap: domain in cast at i8",
            "c_host_domain_before_overflow",
        ),
    ] {
        let program = format!(
            "def f() -> tensor[2, i8] = cast(to_tensor([{values}]), i8)\n\
             out = print(f())\n"
        );
        assert_c_trap_at_thread_counts(&program, expected, name);
    }
}
