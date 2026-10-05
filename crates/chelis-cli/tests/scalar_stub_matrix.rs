//! chelis#715 - eight scalar builtins formerly hit the C host emitter's
//! `/* unsupported builtin */ 0` substitution, including plain f32/f64:
//! `tan`, `atan`, `floor`, `ceil`, `round`, `recip`, `max_elem`, `min_elem`.
//!
//! Unlike chelis#704 (scalar activations, where eval also rejects), eval
//! computed all of these correctly, so nothing warned the user before the
//! compiled binary replaced `floor(1.5)` or `max_elem(lr, floor_val)` with
//! `0`. Same historical substitution site as #682/#704/#705; these ordinary
//! regressions preserve the confirmed blast radius.
//!
//! Also carried here: the chelis#719 regression locks. The C backend's
//! contiguous f32 tensor `sqrt` path used Accelerate's `vvsqrtf`, which is not
//! correctly rounded (IEEE-754 requires sqrt to be) and differed from the
//! strided path's `sqrtf` on the same values. The fix drops `vvsqrtf` for the
//! scalar `sqrtf` loop; these tests lock correct rounding and layout
//! independence in the compiled lane.
//!
//! The passing controls bound the historical stub list exactly: the already
//! working scalar ops stay locked in both lanes, the corresponding tensor
//! forms stay locked, and correct int-dtype rejections stay rejected.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use chelis_types::agreement::{AgreementOp, ArithmeticWidthStatus, compare_rendered_elements};
use chelis_types::types::Prim;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
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

/// Build + link + run; `Ok((emitted_c, first stdout line))`.
fn c_lane(program: &str, name: &str) -> Result<(String, String), String> {
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
    let emitted = std::fs::read_to_string(out_dir.join(format!("{name}.c")))
        .map_err(|e| format!("read emitted C: {e}"))?;
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    let line = String::from_utf8_lossy(&run.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    Ok((emitted, line))
}

fn assert_check_rejects(program: &str, op: &str, rejected_dtype: &str) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("reject_{op}_{rejected_dtype}.ch"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success() || !stdout.contains("\"score\": 1"),
        "`{op}` must reject scalar {rejected_dtype} at check time; \
         stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains(op) || stderr.contains(op) || stdout.contains("PrecisionMismatch"),
        "the rejection should identify `{op}` or its precision contract; \
         stdout={stdout} stderr={stderr}"
    );
}

fn scalar_program(op_expr: &str, ret_ty: &str) -> String {
    format!("module M.Main\ndef run() -> {ret_ty} = {op_expr}\nout = print(run())\n")
}

const STUB_MARKER: &str = "unsupported builtin";

/// One formerly broken row: eval computes `eval_expected`, the C lane must
/// agree on the VALUE and must not contain the stub marker. Before Phase 3,
/// C printed `0`. The observation contract requires the compiled lane to use
/// the same own-width rendering as eval, so the two expected strings are
/// byte-identical for every repaired row.
fn assert_scalar_parity(
    op_expr: &str,
    ret_ty: &str,
    eval_expected: &str,
    c_expected: &str,
    name: &str,
) {
    let program = scalar_program(op_expr, ret_ty);
    let eval_got = eval_first_line(&program)
        .unwrap_or_else(|e| panic!("{name}: eval failed for `{op_expr}`: {e}"));
    // chelis#729 Phase 0: printed values must be members of the declared
    // dtype's value set in both lanes before any parity compare.
    common::assert_elements_in_domain(ret_ty, &eval_got, name);
    assert_eq!(
        eval_got, eval_expected,
        "{name}: eval value drifted; update the row"
    );
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (emitted, c_got) = c_lane(&program, name).expect("C lane should build and run");
    assert!(
        !emitted.contains(STUB_MARKER),
        "{name}: the C emitter substituted a stub for `{op_expr}`"
    );
    common::assert_elements_in_domain(ret_ty, &c_got, name);
    assert_eq!(
        c_got, c_expected,
        "{name}: LANE DIVERGENCE for `{op_expr}`: eval={eval_got}, C={c_got}"
    );
}

fn scalar_number(line: &str, context: &str) -> f64 {
    line.parse::<f64>().unwrap_or_else(|error| {
        panic!("{context}: expected a scalar number, got `{line}`: {error}")
    })
}

/// [05-OP-43] requires reduced-float scalar ReLU to select the stored input
/// bits, rather than round-trip a selected NaN through f32. Keep this check
/// independent of the emitter's temporary names: recover the predicate
/// operand from the emitted selection and require the selected operand to be
/// that same raw value.
fn assert_reduced_relu_raw_selection(run_body: &str, dtype: &str, name: &str) {
    let decoder = format!("chelis_{dtype}_to_f32(");
    let selection = run_body
        .lines()
        .find(|line| line.contains(&decoder) && line.contains(") < 0) ? 0 : "))
        .unwrap_or_else(|| {
            panic!("{name}: no decoded-predicate/raw-value ReLU selection:\n{run_body}")
        });
    let (_, after_decoder) = selection
        .split_once(&decoder)
        .expect("selection line contains the decoder");
    let (predicate_arg, selected) = after_decoder
        .split_once(") < 0) ? 0 : ")
        .expect("selection line has the closed conditional spelling");
    let selected_arg = selected
        .strip_suffix(");")
        .unwrap_or_else(|| panic!("{name}: ReLU selection has an unexpected tail: {selection}"));
    assert_eq!(
        selected_arg, predicate_arg,
        "{name}: ReLU must select the same raw value used by its decoded predicate"
    );

    for forbidden in [
        format!("chelis_host_relu_{dtype}("),
        format!("chelis_host_finalize_{dtype}("),
        format!("chelis_f32_to_{dtype}("),
    ] {
        assert!(
            !run_body.contains(&forbidden),
            "{name}: reduced ReLU must not call `{forbidden}`:\n{run_body}"
        );
    }
}

/// The scalar activation surface decided on chelis#712: every active float
/// width is admitted, eval and compiled C both execute it, and each lane
/// follows the Tier-2 composition. The non-zero input makes the former C stub
/// observable for every operation (including silu/gelu, whose value at zero
/// would not distinguish a stub). The f64 input also distinguishes every f64
/// helper call from an f32 detour.
fn assert_activation_width_matrix(op: &str) {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let input = if dtype == "f64" {
            "1.0000000000000002"
        } else {
            "1.0"
        };
        let expr = format!("{op}(cast({input}, {dtype}))");
        let name = format!("scalar_{op}_{dtype}");
        let program = scalar_program(&expr, dtype);
        let eval = eval_first_line(&program)
            .unwrap_or_else(|error| panic!("{name}: eval rejected a supported scalar: {error}"));
        common::assert_elements_in_domain(dtype, &eval, &name);
        let eval_number = scalar_number(&eval, &name);
        let (emitted, compiled) = c_lane(&program, &name)
            .unwrap_or_else(|error| panic!("{name}: C lane failed: {error}"));
        assert!(
            !emitted.contains(STUB_MARKER),
            "{name}: emitted the historical unsupported-builtin stub"
        );
        // chelis#1820: located by NAME, not by the full signature. chelis#1799
        // added a `chelis_rng_state` parameter to every host body, and the old
        // `run__chelis_owned_body() {` needle then missed the definition and
        // failed before this row inspected any call site.
        let run_body = common::authored_host_body_definition(&emitted, "run");
        let reduced_relu = op == "relu" && matches!(dtype, "f16" | "bf16");
        if reduced_relu {
            assert_reduced_relu_raw_selection(run_body, dtype, &name);
        } else {
            let helper_call = format!("chelis_host_{op}_{dtype}(");
            assert!(
                run_body.contains(&helper_call),
                "{name}: no call site selects `{helper_call}`:\n{emitted}"
            );
        }
        for wrong_width in ["f16", "bf16", "f32", "f64"]
            .into_iter()
            .filter(|width| *width != dtype)
        {
            let wrong_call = format!("chelis_host_{op}_{wrong_width}(");
            assert!(
                !run_body.contains(&wrong_call),
                "{name}: a call site incorrectly selects `{wrong_call}`:\n{emitted}"
            );
        }
        common::assert_elements_in_domain(dtype, &compiled, &name);
        assert_eq!(
            compiled, eval,
            "{name}: scalar activation differs between eval and compiled C"
        );

        match op {
            "relu" => assert_eq!(
                eval_number,
                input.parse::<f64>().unwrap(),
                "{name}: relu of a positive input must be exact"
            ),
            "sigmoid" | "tanh" | "silu" => assert!(
                (0.7..0.8).contains(&eval_number),
                "{name}: {op}({input}) must lie in (0.7, 0.8), got {eval_number}"
            ),
            "gelu" => assert!(
                (0.8..0.9).contains(&eval_number),
                "{name}: gelu({input}) must lie in (0.8, 0.9), got {eval_number}"
            ),
            _ => unreachable!("activation matrix called for `{op}`"),
        }
    }
}

fn tier2_sigmoid_expr(x: &str, dtype: &str) -> String {
    format!("recip(add(cast(1.0, {dtype}), exp(neg({x}))))")
}

/// The spec/05 section 3.3 lowering of each activation. `tanh` is a primitive
/// with no lowering; every witness below is small enough that its correctly
/// rounded value is the input itself, so the expected expression is `x`.
fn tier2_activation_expr(op: &str, x: &str, dtype: &str) -> String {
    match op {
        "sigmoid" => tier2_sigmoid_expr(x, dtype),
        "tanh" => x.to_string(),
        "silu" => format!("mul({x}, {})", tier2_sigmoid_expr(x, dtype)),
        "gelu_tanh" => {
            let x_sq = format!("mul({x}, {x})");
            let x_cu = format!("mul({x_sq}, {x})");
            let k_x_cu = format!("mul(cast(0.044715, {dtype}), {x_cu})");
            let sum_inner = format!("add({x}, {k_x_cu})");
            let u = format!("mul(cast(0.7978845608028654, {dtype}), {sum_inner})");
            let two_u = format!("mul(cast(2.0, {dtype}), {u})");
            format!("mul({x}, {})", tier2_sigmoid_expr(&two_u, dtype))
        }
        _ => unreachable!("no Tier-2 activation expression for `{op}`"),
    }
}

#[test]
fn reduced_float_scalar_activations_match_tier2_node_finalization() {
    for (dtype, op, input) in [
        ("f16", "sigmoid", "0.0007328987121582031"),
        ("f16", "tanh", "5.960464477539063e-8"),
        ("f16", "silu", "2.9802322387695313e-7"),
        ("f16", "gelu_tanh", "2.9802322387695313e-7"),
        ("bf16", "sigmoid", "0.005889892578125"),
        ("bf16", "tanh", "9.183549615799121e-41"),
        ("bf16", "silu", "0.00555419921875"),
        ("bf16", "gelu_tanh", "0.0030975341796875"),
    ] {
        let x = format!("cast({input}, {dtype})");
        let activation = format!("{op}({x})");
        let tier2 = tier2_activation_expr(op, &x, dtype);
        let name = format!("{dtype}_{op}_tier2_finalization");
        let expected = eval_first_line(&scalar_program(&tier2, dtype))
            .unwrap_or_else(|error| panic!("{name}: Tier-2 expression failed: {error}"));
        let scalar = eval_first_line(&scalar_program(&activation, dtype))
            .unwrap_or_else(|error| panic!("{name}: scalar activation failed: {error}"));
        assert_eq!(
            scalar, expected,
            "{name}: scalar activation must equal its Tier-2 composition"
        );
        let (emitted, compiled) = c_lane(&scalar_program(&activation, dtype), &name)
            .unwrap_or_else(|error| panic!("{name}: C lane failed: {error}"));
        assert!(
            !emitted.contains(STUB_MARKER),
            "{name}: emitted the historical unsupported-builtin stub"
        );
        assert_eq!(
            compiled, expected,
            "{name}: compiled scalar activation must equal its Tier-2 composition"
        );
    }
}

/// [05-OBS-3] parity for an f64 transcendental, correctly rounded in both
/// lanes under [05-OP-46], so the shared comparator demands exact agreement;
/// the emitted-source assertion keeps a mutually wrong f32 implementation
/// from passing by byte agreement.
fn assert_f64_transcendental_parity(op_expr: &str, op_name: &str, name: &str) {
    let program = scalar_program(op_expr, "f64");
    let eval_got = eval_first_line(&program).unwrap_or_else(|e| panic!("{name}: eval failed: {e}"));
    common::assert_elements_in_domain("f64", &eval_got, name);
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (emitted, c_got) = c_lane(&program, name).expect("C lane should build and run");
    common::assert_elements_in_domain("f64", &c_got, name);
    assert!(
        !emitted.contains(&format!("{op_name}f(")),
        "{name}: an f64 operation must not route through the f32 libm entry"
    );
    compare_rendered_elements(
        AgreementOp::Exact,
        Prim::F64,
        ArithmeticWidthStatus::StoredAtArithmeticWidth,
        &eval_got,
        &c_got,
    )
    .unwrap_or_else(|error| panic!("{name}: {error}"));
}

// ===========================================================================
// chelis#715 - repaired rows (all printed 0 before the Phase 3 fix)
// ===========================================================================

#[test]
fn f32_scalar_tan_agrees_across_lanes() {
    assert_scalar_parity(
        "tan(cast(1.0, f32))",
        "f32",
        "1.5574077",
        "1.5574077",
        "f32_tan",
    );
}

#[test]
fn f32_scalar_atan_agrees_across_lanes() {
    assert_scalar_parity(
        "atan(cast(1.0, f32))",
        "f32",
        "0.7853982",
        "0.7853982",
        "f32_atan",
    );
}

#[test]
fn f32_scalar_floor_agrees_across_lanes() {
    assert_scalar_parity("floor(cast(1.5, f32))", "f32", "1.0", "1.0", "f32_floor");
}

#[test]
fn f32_scalar_ceil_agrees_across_lanes() {
    assert_scalar_parity("ceil(cast(1.5, f32))", "f32", "2.0", "2.0", "f32_ceil");
}

#[test]
fn f32_scalar_round_agrees_across_lanes() {
    // [05] §2.2 requires roundTiesToEven. Both signs distinguish the
    // C `rint{,f}` family from `round{,f}`, whose half ties go away from
    // zero. Cover both scalar host widths because each selects a distinct
    // libm entry point.
    for (dtype, input, expected, name) in [
        ("f32", "2.5", "2.0", "f32_round_positive_even_tie"),
        ("f32", "-2.5", "-2.0", "f32_round_negative_even_tie"),
        ("f64", "2.5", "2.0", "f64_round_positive_even_tie"),
        ("f64", "-2.5", "-2.0", "f64_round_negative_even_tie"),
    ] {
        assert_scalar_parity(
            &format!("round(cast({input}, {dtype}))"),
            dtype,
            expected,
            expected,
            name,
        );
    }
}

#[test]
fn f32_scalar_recip_agrees_across_lanes() {
    assert_scalar_parity("recip(cast(4.0, f32))", "f32", "0.25", "0.25", "f32_recip");
}

#[test]
fn f32_scalar_max_elem_agrees_across_lanes() {
    assert_scalar_parity(
        "max_elem(cast(1.5, f32), cast(0.25, f32))",
        "f32",
        "1.5",
        "1.5",
        "f32_max_elem",
    );
}

#[test]
fn f32_scalar_min_elem_agrees_across_lanes() {
    assert_scalar_parity(
        "min_elem(cast(1.5, f32), cast(0.25, f32))",
        "f32",
        "0.25",
        "0.25",
        "f32_min_elem",
    );
}

#[test]
fn f64_scalar_floor_agrees_across_lanes() {
    assert_scalar_parity("floor(cast(1.5, f64))", "f64", "1.0", "1.0", "f64_floor");
}

#[test]
fn i64_scalar_max_elem_agrees_across_lanes() {
    assert_scalar_parity(
        "max_elem(cast(7, i64), cast(3, i64))",
        "i64",
        "7",
        "7",
        "i64_max_elem",
    );
}

/// The bonus three-lane row (#712 shape): the checker accepts
/// `floor(cast(5, i64))` (score 1), eval REJECTS it at runtime (`float op
/// expects float arg`), and the compiled binary prints 0. floor is
/// well-defined on integers and the checker's own op list agrees (it is
/// absent from TRANSCENDENTAL_FLOAT_ONLY_OPS, per chelis#699), so the
/// correct behavior is identity.
#[test]
fn integer_scalar_floor_ceil_round_are_identity_in_all_lanes() {
    for dtype in ["i8", "i16", "i32", "i64"] {
        for (op, input) in [("floor", "5"), ("ceil", "-5"), ("round", "5")] {
            let name = format!("{dtype}_{op}_identity");
            let expected = input;
            assert_scalar_parity(
                &format!("{op}(cast({input}, {dtype}))"),
                dtype,
                expected,
                expected,
                &name,
            );
        }
    }
}

#[test]
fn scalar_activation_family_agrees_at_every_float_width() {
    for op in ["relu", "sigmoid", "tanh", "silu", "gelu"] {
        assert_activation_width_matrix(op);
    }
}

#[test]
fn scalar_transcendental_and_rounding_families_cover_reduced_float_widths() {
    for dtype in ["f16", "bf16"] {
        for (op, input) in [
            ("tan", "1.0"),
            ("atan", "1.0"),
            ("recip", "4.0"),
            ("floor", "1.5"),
            ("ceil", "1.5"),
            ("round", "2.5"),
        ] {
            let name = format!("{dtype}_{op}");
            let program = scalar_program(&format!("{op}(cast({input}, {dtype}))"), dtype);
            let eval = eval_first_line(&program)
                .unwrap_or_else(|error| panic!("{name}: eval failed: {error}"));
            let (emitted, compiled) = c_lane(&program, &name)
                .unwrap_or_else(|error| panic!("{name}: C lane failed: {error}"));
            assert!(!emitted.contains(STUB_MARKER), "{name}: emitted a stub");
            assert_eq!(compiled, eval, "{name}: reduced-float lane divergence");
        }
    }
}

#[test]
fn scalar_max_min_cover_all_admitted_widths() {
    for dtype in ["i8", "i16", "i32", "i64"] {
        for (op, expected) in [("max_elem", "7"), ("min_elem", "-3")] {
            let name = format!("{dtype}_{op}");
            assert_scalar_parity(
                &format!("{op}(cast(7, {dtype}), cast(-3, {dtype}))"),
                dtype,
                expected,
                expected,
                &name,
            );
        }
    }
    for dtype in ["f16", "bf16"] {
        for (op, expected) in [("max_elem", "1.5"), ("min_elem", "-0.25")] {
            let name = format!("{dtype}_{op}");
            assert_scalar_parity(
                &format!("{op}(cast(1.5, {dtype}), cast(-0.25, {dtype}))"),
                dtype,
                expected,
                expected,
                &name,
            );
        }
    }
}

#[test]
fn float_only_scalar_families_reject_integer_and_bool_at_check_time() {
    for op in [
        "relu", "sigmoid", "tanh", "silu", "gelu", "tan", "atan", "recip",
    ] {
        assert_check_rejects(
            &format!("module M.Main\ndef run() -> i64 = {op}(cast(1, i64))\nout = print(run())\n"),
            op,
            "i64",
        );
        assert_check_rejects(
            &format!("module M.Main\ndef run() -> bool = {op}(true)\nout = print(run())\n"),
            op,
            "bool",
        );
    }
}

/// The f64 rows of the stub family, distilled from the probe battery
/// (`docs/investigations/probes/bat_scalar_ops.py`) - chelis#715's title
/// says EVERY dtype, so the f64 half is asserted too, not just f32.
/// Before Phase 3, C printed 0 for all seven rows while eval was correct.
#[test]
fn f64_scalar_stub_family_agrees_across_lanes() {
    assert_f64_transcendental_parity("tan(cast(1.0, f64))", "tan", "f64_tan");
    assert_f64_transcendental_parity("atan(cast(1.0, f64))", "atan", "f64_atan");
    for (expr, eval_expected, c_expected, name) in [
        ("ceil(cast(1.5, f64))", "2.0", "2.0", "f64_ceil"),
        ("recip(cast(4.0, f64))", "0.25", "0.25", "f64_recip"),
        (
            "max_elem(cast(1.5, f64), cast(0.25, f64))",
            "1.5",
            "1.5",
            "f64_max_elem",
        ),
        (
            "min_elem(cast(1.5, f64), cast(0.25, f64))",
            "0.25",
            "0.25",
            "f64_min_elem",
        ),
    ] {
        assert_scalar_parity(expr, "f64", eval_expected, c_expected, name);
    }
}

/// The f64 working-op controls, mirroring the f32 set: bounds the f64 half
/// of #715 to exactly the seven formerly broken ops above.
#[test]
fn working_f64_scalar_ops_agree_across_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    for (op_expr, eval_expected, c_expected, name) in [
        ("abs(cast(-1.5, f64))", "1.5", "1.5", "ctl64_abs"),
        ("neg(cast(1.5, f64))", "-1.5", "-1.5", "ctl64_neg"),
        ("sqrt(cast(2.25, f64))", "1.5", "1.5", "ctl64_sqrt"),
        ("exp(cast(0.0, f64))", "1.0", "1.0", "ctl64_exp"),
        ("log(cast(1.0, f64))", "0.0", "0.0", "ctl64_log"),
        ("sin(cast(0.0, f64))", "0.0", "0.0", "ctl64_sin"),
        ("cos(cast(0.0, f64))", "1.0", "1.0", "ctl64_cos"),
        (
            "add(cast(1.5, f64), cast(0.25, f64))",
            "1.75",
            "1.75",
            "ctl64_add",
        ),
        (
            "sub(cast(1.5, f64), cast(0.25, f64))",
            "1.25",
            "1.25",
            "ctl64_sub",
        ),
        (
            "mul(cast(1.5, f64), cast(0.25, f64))",
            "0.375",
            "0.375",
            "ctl64_mul",
        ),
        (
            "div(cast(1.5, f64), cast(0.25, f64))",
            "6.0",
            "6.0",
            "ctl64_div",
        ),
    ] {
        assert_scalar_parity(op_expr, "f64", eval_expected, c_expected, name);
    }
}

// ===========================================================================
// chelis#719 - vvsqrtf on the contiguous f32 tensor path (macOS/Accelerate)
// ===========================================================================

/// IEEE-754 requires squareRoot to be correctly rounded. The contiguous
/// kernel path calls Accelerate's `vvsqrtf`, which returns 0x3F9CC470 for
/// sqrt(1.5) - one ulp below the correctly rounded 0x3F9CC471 that both
/// `sqrtf` (the strided path of the SAME kernel) and eval produce.
/// Bit-pattern comparison, not decimal text.
#[test]
fn c_f32_tensor_sqrt_is_correctly_rounded() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: the vvsqrtf path is emitted only where Accelerate exists");
        return;
    }
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, line) = c_lane(
        "module M.Main\n\
         def f(x: tensor[2, f32]) -> tensor[2, f32] = sqrt(x)\n\
         out = print(f(to_tensor([1.5, 3.0])))\n",
        "c_sqrt_rounding",
    )
    .expect("C lane should run");
    let start = line.find("data=[").expect("data marker") + "data=[".len();
    let end = start + line[start..].find(',').expect("comma");
    let value: f64 = line[start..end].trim().parse().expect("numeric");
    assert_eq!(
        (value as f32).to_bits(),
        0x3F9C_C471,
        "sqrt(1.5) must be correctly rounded (sqrtf/eval agree on 0x3F9CC471); \
         the vvsqrtf path returned bits {:#010X} ({value})",
        (value as f32).to_bits()
    );
}

/// Bits of the first element printed in a `data=[...]` line.
fn first_data_elem_bits(line: &str) -> u32 {
    let start = line.find("data=[").expect("data marker") + "data=[".len();
    let rest = &line[start..];
    let end = rest.find([',', ']']).expect("element terminator");
    let value: f64 = rest[..end].trim().parse().expect("numeric element");
    (value as f32).to_bits()
}

/// chelis#719 layout-independence lock. sqrt(1.5) must print byte-identical
/// bits whether its input is contiguous or a strided (permuted) view. Before
/// the fix the contiguous kernel takes Accelerate `vvsqrtf` (0x3F9CC470) while
/// a permuted view takes the strided `sqrtf` (0x3F9CC471), so the SAME value
/// gives two answers depending on memory layout. After the fix both paths use
/// the correctly-rounded `sqrtf` and agree.
#[test]
fn c_f32_tensor_sqrt_is_layout_independent() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    // Contiguous input: sqrt(1.5) is the first printed element.
    let (_, contig) = c_lane(
        "module M.Main\n\
         def f(x: tensor[2, f32]) -> tensor[2, f32] = sqrt(x)\n\
         out = print(f(to_tensor([1.5, 3.0])))\n",
        "c_sqrt_contig",
    )
    .expect("contiguous C lane should run");
    // Strided input: permute transposes a 2x2 into a non-contiguous view, so
    // sqrt sees strided data; the transposed [0][0] element is still 1.5.
    let (_, strided) = c_lane(
        "module M.Main\n\
         def f(x: tensor[2, 2, f32]) -> tensor[2, 2, f32] = sqrt(permute(x, 1, 0))\n\
         out = print(f(to_tensor([[1.5, 3.0], [9.0, 4.0]])))\n",
        "c_sqrt_strided",
    )
    .expect("strided C lane should run");
    let contig_bits = first_data_elem_bits(&contig);
    let strided_bits = first_data_elem_bits(&strided);
    assert_eq!(
        contig_bits, strided_bits,
        "sqrt(1.5) must not depend on memory layout: contiguous returned \
         {contig_bits:#010X}, strided (permuted view) returned {strided_bits:#010X}"
    );
    assert_eq!(
        contig_bits, 0x3F9C_C471,
        "the layout-independent value must be the correctly-rounded one \
         (0x3F9CC471), not both paths agreeing on the wrong bits; got {contig_bits:#010X}"
    );
}

/// f32 bits of every element in a `data=[...]` line. Comparing bits (not the
/// decimal text) keeps a lane-specific float-to-string rendering of the SAME
/// f32 value - e.g. eval `1.4142135381698608` vs C `1.414213538169861`, both
/// 0x3FB504F3 - from reading as a false divergence.
fn data_elem_bits(line: &str) -> Vec<u32> {
    let start = line.find("data=[").expect("data marker") + "data=[".len();
    let close = line[start..].find(']').expect("close bracket") + start;
    line[start..close]
        .split(',')
        .map(|tok| {
            let v: f64 = tok.trim().parse().expect("numeric element");
            (v as f32).to_bits()
        })
        .collect()
}

/// chelis#719 negative parity. Ordinary sqrt inputs - a perfect square
/// (4.0 -> 2.0 exact) and inexact roots (2.0, 0.5, 1.5) - must be bit-identical
/// in the eval and compiled-C lanes. This is the regression control: the fix
/// removes the vvsqrtf divergence at 1.5 while leaving every other value exactly
/// where eval computes it. Bit comparison, so lane-specific decimal formatting
/// of the same f32 is not mistaken for a numeric difference.
#[test]
fn c_f32_tensor_sqrt_ordinary_values_agree_across_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "module M.Main\n\
         def f(x: tensor[4, f32]) -> tensor[4, f32] = sqrt(x)\n\
         out = print(f(to_tensor([4.0, 2.0, 0.5, 1.5])))\n";
    let eval_line = eval_first_line(program).expect("eval should run");
    let (_, c_line) = c_lane(program, "c_sqrt_ordinary").expect("C lane should build and run");
    assert_eq!(
        data_elem_bits(&eval_line),
        data_elem_bits(&c_line),
        "sqrt lane divergence on ordinary values: eval={eval_line}, C={c_line}"
    );
    // Anchor the perfect-square element so the parity check cannot pass on two
    // lanes that are wrong in the same way: sqrt(4.0) is exactly 2.0.
    assert_eq!(
        first_data_elem_bits(&c_line),
        2.0f32.to_bits(),
        "sqrt(4.0) must be exactly 2.0; C lane printed {c_line}"
    );
}

// ===========================================================================
// CONTROLS: the working scalar surface, locked in both lanes.
// ===========================================================================

/// The scalar ops that DO work, at f32, with values whose printed form is
/// identical in both lanes. Bounds #715 to the eight formerly broken ops.
#[test]
fn working_f32_scalar_ops_agree_across_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    for (op_expr, eval_expected, c_expected, name) in [
        ("abs(cast(-1.5, f32))", "1.5", "1.5", "ctl_abs"),
        ("neg(cast(1.5, f32))", "-1.5", "-1.5", "ctl_neg"),
        ("sqrt(cast(2.25, f32))", "1.5", "1.5", "ctl_sqrt"),
        ("exp(cast(0.0, f32))", "1.0", "1.0", "ctl_exp"),
        ("log(cast(1.0, f32))", "0.0", "0.0", "ctl_log"),
        ("sin(cast(0.0, f32))", "0.0", "0.0", "ctl_sin"),
        ("cos(cast(0.0, f32))", "1.0", "1.0", "ctl_cos"),
        (
            "add(cast(1.5, f32), cast(0.25, f32))",
            "1.75",
            "1.75",
            "ctl_add",
        ),
        (
            "sub(cast(1.5, f32), cast(0.25, f32))",
            "1.25",
            "1.25",
            "ctl_sub",
        ),
        (
            "mul(cast(1.5, f32), cast(0.25, f32))",
            "0.375",
            "0.375",
            "ctl_mul",
        ),
        (
            "div(cast(1.5, f32), cast(0.25, f32))",
            "6.0",
            "6.0",
            "ctl_div",
        ),
    ] {
        assert_scalar_parity(op_expr, "f32", eval_expected, c_expected, name);
    }
}

/// i64 scalar `mod` / `floor_div` / `trunc_div` / `abs` / `neg` are correct
/// in both lanes: the #387 `checked_int_binop` family holds up in C. Bounds
/// the stub to the eight ops and keeps the int lane's working core locked.
#[test]
fn working_i64_scalar_ops_agree_across_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    for (op_expr, expected, name) in [
        ("mod(cast(7, i64), cast(3, i64))", "1", "ctl_i64_mod"),
        ("floor_div(cast(7, i64), cast(2, i64))", "3", "ctl_i64_fdiv"),
        (
            "trunc_div(cast(-7, i64), cast(2, i64))",
            "-3",
            "ctl_i64_tdiv",
        ),
        ("abs(cast(-5, i64))", "5", "ctl_i64_abs"),
        ("neg(cast(5, i64))", "-5", "ctl_i64_neg"),
    ] {
        // Integer scalar digits are grammar-stable: one expected string
        // serves both lanes.
        assert_scalar_parity(op_expr, "i64", expected, expected, name);
    }
}

/// The TENSOR forms of the formerly broken ops are correct in both lanes -
/// the stub was scalar-only, exactly like #704's tensor/scalar split. (floor and
/// max_elem chosen for exact printed values; tan/recip carry rounding
/// differences that belong to chelis#717/#719, not here.)
#[test]
fn tensor_forms_of_stubbed_ops_are_correct() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "module M.Main\n\
         def f(x: tensor[2, f32]) -> tensor[2, f32] = floor(x)\n\
         out = print(f(to_tensor([1.5, 0.25])))\n";
    let eval_line = eval_first_line(program).expect("eval");
    let (emitted, c_line) = c_lane(program, "ctl_tensor_floor").expect("C lane");
    assert!(!emitted.contains(STUB_MARKER));
    assert!(eval_line.contains("data=[1.0, 0.0]"), "got: {eval_line}");
    assert!(c_line.contains("data=[1.0, 0.0]"), "got: {c_line}");

    let program = "module M.Main\n\
         def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = max_elem(x, y)\n\
         out = print(f(to_tensor([1.5, 0.25]), to_tensor([0.5, 2.5])))\n";
    let eval_line = eval_first_line(program).expect("eval");
    let (emitted, c_line) = c_lane(program, "ctl_tensor_maxelem").expect("C lane");
    assert!(!emitted.contains(STUB_MARKER));
    assert!(eval_line.contains("data=[1.5, 2.5]"), "got: {eval_line}");
    assert!(c_line.contains("data=[1.5, 2.5]"), "got: {c_line}");
}

/// The int rejections that are CORRECT stay rejected: `div` and `recip` on
/// integer scalars fail the checker in both lanes (use floor_div/trunc_div).
/// Negative parity so a #715 fix does not over-accept.
#[test]
fn int_div_and_recip_stay_rejected_by_the_checker() {
    let err = eval_first_line(&scalar_program("div(cast(7, i64), cast(2, i64))", "i64"))
        .expect_err("int div must be rejected");
    assert!(
        err.contains("PrecisionMismatch") || err.contains("div on integer"),
        "got: {err}"
    );
    let err = eval_first_line(&scalar_program("recip(cast(4, i64))", "i64"))
        .expect_err("int recip must be rejected");
    assert!(
        err.contains("TypeMismatch") || err.contains("recip"),
        "got: {err}"
    );
}
