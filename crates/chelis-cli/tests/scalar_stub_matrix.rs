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

/// [05-OBS-3] parity for an f64 transcendental whose libm result may differ
/// by one ULP across supported platforms.  The shared closed-op comparator is
/// the only authority for the tolerance; the emitted-source assertion keeps
/// a mutually wrong f32 implementation from passing by byte agreement.
fn assert_f64_transcendental_parity(op_expr: &str, op: AgreementOp, name: &str) {
    let program = scalar_program(op_expr, "f64");
    let eval_got = eval_first_line(&program).unwrap_or_else(|e| panic!("{name}: eval failed: {e}"));
    common::assert_elements_in_domain("f64", &eval_got, name);
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (emitted, c_got) = c_lane(&program, name).expect("C lane should build and run");
    common::assert_elements_in_domain("f64", &c_got, name);
    assert!(
        !emitted.contains(&format!("{}f(", op.name())),
        "{name}: an f64 operation must not route through the f32 libm entry"
    );
    compare_rendered_elements(
        op,
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
        "max_elem(cast(7, int64), cast(3, int64))",
        "int64",
        "7",
        "7",
        "i64_max_elem",
    );
}

/// The bonus three-lane row (#712 shape): the checker accepts
/// `floor(cast(5, int64))` (score 1), eval REJECTS it at runtime (`float op
/// expects float arg`), and the compiled binary prints 0. floor is
/// well-defined on integers and the checker's own op list agrees (it is
/// absent from TRANSCENDENTAL_FLOAT_ONLY_OPS, per chelis#699), so the
/// correct behavior is identity.
#[test]
#[ignore = "chelis#715 Phase 4: the capability-table cell for integer scalar floor is \
            not ratified end-to-end; eval still rejects it. Run with \
            `cargo test -p chelis-cli --test scalar_stub_matrix -- --ignored`."]
fn i64_scalar_floor_is_identity_in_all_lanes() {
    assert_scalar_parity("floor(cast(5, int64))", "int64", "5", "5", "i64_floor");
}

/// The f64 rows of the stub family, distilled from the probe battery
/// (`docs/investigations/probes/bat_scalar_ops.py`) - chelis#715's title
/// says EVERY dtype, so the f64 half is asserted too, not just f32.
/// Before Phase 3, C printed 0 for all seven rows while eval was correct.
#[test]
fn f64_scalar_stub_family_agrees_across_lanes() {
    assert_f64_transcendental_parity("tan(cast(1.0, f64))", AgreementOp::Tan, "f64_tan");
    assert_f64_transcendental_parity("atan(cast(1.0, f64))", AgreementOp::Atan, "f64_atan");
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

/// int64 scalar `mod` / `floor_div` / `trunc_div` / `abs` / `neg` are correct
/// in both lanes: the #387 `checked_int_binop` family holds up in C. Bounds
/// the stub to the eight ops and keeps the int lane's working core locked.
#[test]
fn working_i64_scalar_ops_agree_across_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    for (op_expr, expected, name) in [
        ("mod(cast(7, int64), cast(3, int64))", "1", "ctl_i64_mod"),
        (
            "floor_div(cast(7, int64), cast(2, int64))",
            "3",
            "ctl_i64_fdiv",
        ),
        (
            "trunc_div(cast(-7, int64), cast(2, int64))",
            "-3",
            "ctl_i64_tdiv",
        ),
        ("abs(cast(-5, int64))", "5", "ctl_i64_abs"),
        ("neg(cast(5, int64))", "-5", "ctl_i64_neg"),
    ] {
        // Integer scalar digits are grammar-stable: one expected string
        // serves both lanes.
        assert_scalar_parity(op_expr, "int64", expected, expected, name);
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
    let err = eval_first_line(&scalar_program(
        "div(cast(7, int64), cast(2, int64))",
        "int64",
    ))
    .expect_err("int div must be rejected");
    assert!(
        err.contains("PrecisionMismatch") || err.contains("div on integer"),
        "got: {err}"
    );
    let err = eval_first_line(&scalar_program("recip(cast(4, int64))", "int64"))
        .expect_err("int recip must be rejected");
    assert!(
        err.contains("TypeMismatch") || err.contains("recip"),
        "got: {err}"
    );
}
