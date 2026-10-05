//! chelis#717 - the host evaluator's tensor lane formerly narrowed results to
//! the element dtype per-op inconsistently in three directions:
//!
//! 1. **f64 tensors are destroyed to f32** by every unary float op:
//!    `tensor_float_unop_f32` (`crates/chelis-compiler-api/src/runtime/
//!    host_ops.rs:518-544`) routes each element through `op(x as f32) as
//!    f64` regardless of the tensor's dtype. Its doc comment says it exists
//!    to stay byte-identical with the C backend's f32 helpers - the right
//!    goal for f32 tensors, silently destructive for f64 ones. The C lane
//!    emits genuine `double` math for f64 tensors, so the two lanes applied
//!    "match the other lane" in opposite directions.
//! 2. **f32 tensors skip narrowing on add/div/recip** (results are raw f64
//!    values not representable in f32), while tan/sqrt/sin do narrow.
//! 3. **f16/bf16 tensors are never rounded at all** (covered in
//!    narrow_dtype_matrix.rs; this file carries the f32/f64 rows).
//!
//! The compiled C lane supplied the correct controls for every cell, so each
//! historical row captured a cross-lane divergence in the direction
//! eval-wrong / C-right - the same direction as chelis#691, opposite to
//! chelis#680. All rows are now ordinary regression tests.
//!
//! Why nothing caught it: `eval_agreement.rs` compares lanes through f64
//! with a tolerance (chelis#687), and every f32/f64 row here is a relative
//! error of 1e-8 .. 1e-16 - exactly what a tolerance oracle ignores.
//!
//! The C-lane cells remain passing controls so a future change cannot regress
//! the correct lane while preserving evaluator parity.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
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

/// Build + link + run through the C lane; first stdout line verbatim.
fn c_first_line(program: &str, name: &str) -> Result<String, String> {
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
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    Ok(String::from_utf8_lossy(&run.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

const F64_UNOP: &str = "module M.Main\n\
     def f(x: tensor[2, f64]) -> tensor[2, f64] = OP(x)\n\
     out = print(f(to_tensor([cast(A, f64), cast(B, f64)])))\n";

fn f64_unop_program(op: &str, a: &str, b: &str) -> String {
    F64_UNOP.replace("OP", op).replace('A', a).replace('B', b)
}

// ===========================================================================
// CONTROLS: the C lane is exact in every broken cell, and the parts of eval
// that are correct must stay correct.
// ===========================================================================

/// The C lane computes f64 tensor transcendentals at genuine f64 precision.
/// These exact strings are what eval must ALSO produce once #717 is fixed.
#[test]
fn c_f64_tensor_unary_ops_are_f64_precise() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let tan_program = f64_unop_program("tan", "1.5", "3.0");
    let line = c_first_line(&tan_program, "c_f64_tan").expect("C lane should run");
    common::assert_elements_in_domain("f64", &line, "c_f64_tan");
    // f64 PRECISION is the claim: the #717 bug's f32-destroyed value is
    // ~5e-7 away from true tan. Both lanes compute the correctly rounded
    // result ([05-OP-46]), so they also agree byte for byte.
    let v = parse_data(&line)[0];
    let truth = 14.10141994717172_f64;
    assert!(
        (v - truth).abs() < 1e-12,
        "C f64 tan(1.5) must be f64-precise (within 1e-12 of {truth}); got {v} in: {line}"
    );
    let eval_line = eval_first_line(&tan_program).expect("eval tan should run");
    assert_eq!(
        eval_line, line,
        "f64 tan: eval and C must agree byte for byte"
    );

    let exp_program = f64_unop_program("exp", "2.0", "3.0");
    let c_exp_line = c_first_line(&exp_program, "c_f64_exp").expect("C exp should run");
    let eval_exp_line = eval_first_line(&exp_program).expect("eval exp should run");
    assert_eq!(
        eval_exp_line, c_exp_line,
        "f64 exp: eval and C must agree byte for byte"
    );

    let line = c_first_line(&f64_unop_program("sqrt", "2.0", "3.0"), "c_f64_sqrt")
        .expect("C lane should run");
    assert!(
        line.contains("1.414213562373095"),
        "C f64 sqrt(2) must be f64-precise; got: {line}"
    );
}

/// f64 tensor binary ops are exact in BOTH lanes (the binary path does not
/// take the f32 wrapper). Bounds #717 to the unary path on f64.
#[test]
fn f64_tensor_div_is_exact_in_both_lanes() {
    let program = "module M.Main\n\
         def f(x: tensor[2, f64], y: tensor[2, f64]) -> tensor[2, f64] = div(x, y)\n\
         out = print(f(to_tensor([cast(1.0, f64), cast(2.0, f64)]), \
         to_tensor([cast(3.0, f64), cast(3.0, f64)])))\n";
    let eval_line = eval_first_line(program).expect("eval should run");
    common::assert_elements_in_domain("f64", &eval_line, "f64_div eval");
    assert!(
        eval_line.contains("0.3333333333333333"),
        "eval f64 div must be exact f64; got: {eval_line}"
    );
    if c_toolchain_available() {
        let c_line = c_first_line(program, "f64_div_both").expect("C lane should run");
        common::assert_elements_in_domain("f64", &c_line, "f64_div C");
        assert!(
            c_line.contains("0.3333333333333333"),
            "C f64 div must be exact f64; got: {c_line}"
        );
    }
}

/// The C lane rounds f32 tensor arithmetic through f32, correctly.
/// f32(0.1) + f32(0.2) = 0.30000001192092896 exactly.
#[test]
fn c_f32_tensor_add_rounds_to_f32() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let line = c_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = add(x, y)\n\
         out = print(f(to_tensor([0.1, 1.0], f32), to_tensor([0.2, 2.0], f32)))\n",
        "c_f32_add",
    )
    .expect("C lane should run");
    common::assert_elements_in_domain("f32", &line, "c_f32_add");
    // chelis#732 Phase 2: the C lane renders at f32 width, and the
    // shortest string parsing back to the exact f32 sum
    // (0.30000001192092896) is "0.3" - same stored bits, own-width
    // digits ([05-OBS-2]; the pre-contract %.16g printed
    // "0.300000011920929").
    assert!(
        line.contains("data=[0.3, 3.0]"),
        "C f32 add(0.1, 0.2) must render the f32 sum at f32 width; got: {line}"
    );
}

/// Parse the flat `data=[..]` payload at f32 width (the eval f32 tensor
/// exit's own width per spec/05 section 8.1; parsing shortest-at-f32
/// digits as f64 would manufacture a non-f32 value).
fn parse_data_f32(line: &str) -> Vec<f32> {
    let start = line.find("data=[").expect("data marker") + "data=[".len();
    let end = start + line[start..].find(']').expect("closing bracket");
    line[start..end]
        .split(',')
        .map(|s| s.trim().parse::<f32>().expect("numeric"))
        .collect()
}

/// Parse the flat `data=[..]` payload of a printed tensor line.
fn parse_data(line: &str) -> Vec<f64> {
    let start = line.find("data=[").expect("data marker") + "data=[".len();
    let end = start + line[start..].find(']').expect("closing bracket");
    line[start..end]
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("numeric"))
        .collect()
}

/// **The other half of the inconsistency claim, locked**: eval's f32
/// tensor `tan` and `sqrt` DO narrow to f32, while `add`/`div`/`recip` do
/// not (the broken rows below). Distilled from
/// `docs/investigations/probes/bat_f32_tensor_round.py`.
///
/// The `tan` assertion is PROPERTY-BASED (every element must be exactly
/// f32-representable and near the true value), not an exact string:
/// `tanf` is not required to be correctly rounded and its result differs
/// by 1 ulp between platform libms (macOS `14.101419448852539` vs glibc
/// `14.101420402526855` - CI caught exactly this, and it is the
/// transcendental-variance class the chelis#732 tolerance table exists
/// for). `sqrt` IS required correctly rounded by IEEE-754, so its exact
/// strings are safe on every platform.
///
/// If this control ever fails, either the wrapper moved (re-check
/// #717's table) or the fix landed and the broken rows should be
/// flipping green with it.
#[test]
fn eval_f32_tensor_tan_and_sqrt_do_narrow_to_f32() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32]) -> tensor[2, f32] = tan(x)\n\
         out = print(f(to_tensor([1.5, 3.0], f32)))\n",
    )
    .expect("eval should run");
    // chelis#729 Phase 0: the domain checker states the same
    // f32-representability property mechanically; the hand-rolled loop
    // below stays as the original control (controls never move).
    common::assert_elements_in_domain("f32", &line, "eval_f32_tan");
    // chelis#729 Phase 1 width migration (spec/05 section 8.1's recorded
    // note: own-width tensor digits arrive when #729 repairs the storage,
    // which is this change): eval f32 tensor elements now render
    // shortest-round-trip AT F32 WIDTH, so the parse-back property check
    // reads them at f32 and the exact strings below carry f32 digits.
    let values = parse_data_f32(&line);
    for (v, truth) in values
        .iter()
        .zip([14.10141994717172_f64, -0.1425465430742778])
    {
        assert!(
            (f64::from(*v) - truth).abs() < 1e-4 * truth.abs().max(1.0),
            "tan value implausibly far from tan(x); got {v} in: {line}"
        );
    }
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32]) -> tensor[2, f32] = sqrt(x)\n\
         out = print(f(to_tensor([1.5, 3.0], f32)))\n",
    )
    .expect("eval should run");
    common::assert_elements_in_domain("f32", &line, "eval_f32_sqrt");
    assert!(
        line.contains("data=[1.2247449, 1.7320508]"),
        "eval f32 tensor sqrt narrows through f32 today (and is correctly \
         rounded on every platform per IEEE-754, unlike the C lane's \
         vvsqrtf - chelis#719); got: {line}"
    );
}

// ===========================================================================
// chelis#717 - f64 tensors destroyed to f32 by unary ops (eval)
// ===========================================================================

/// Before the typed kernel split, eval printed 14.101419448852539 - the
/// f32-precision tan - for an f64 tensor. This locks the repaired f64 path.
#[test]
fn eval_f64_tensor_tan_keeps_f64_precision() {
    let line = eval_first_line(&f64_unop_program("tan", "1.5", "3.0")).expect("eval should run");
    common::assert_elements_in_domain("f64", &line, "eval_f64_tan");
    // Property-based like the C control above, under the same Phase 1
    // implementation margin and for the same reason
    // (rt857 round-1 CI caught it): double tan is not required to be
    // correctly rounded and differs by 1 ulp between platform libms
    // (macOS 14.10141994717172 vs glibc 14.101419947171719). f64
    // PRECISION is the claim: the chelis#717 f32-destroyed value is
    // ~5e-7 away, while any reasonable libm is within ~1e-15.
    let v = parse_data(&line)[0];
    let truth = 14.10141994717172_f64;
    assert!(
        (v - truth).abs() < 1e-12,
        "f64 tensor tan must be f64-precise (within 1e-12 of {truth}); got {v} in: {line}"
    );
}

/// The pre-repair value was 1.4142135381698608 = f32(sqrt(2)).
#[test]
fn eval_f64_tensor_sqrt_keeps_f64_precision() {
    let line = eval_first_line(&f64_unop_program("sqrt", "2.0", "3.0")).expect("eval should run");
    assert!(
        line.contains("1.4142135623730951"),
        "f64 tensor sqrt must be f64-precise; got: {line}"
    );
}

/// The pre-repair value was 7.389056205749512 = f32(exp(2)).
#[test]
fn eval_f64_tensor_exp_keeps_f64_precision() {
    let line = eval_first_line(&f64_unop_program("exp", "2.0", "3.0")).expect("eval should run");
    common::assert_elements_in_domain("f64", &line, "eval_f64_exp");
    // Property-based under the Phase 1 implementation's 1e-12 separating
    // margin, for the same libm-variance reason as tan (double
    // exp is also not required correctly rounded); sqrt keeps its exact
    // string below because IEEE-754 requires sqrt correctly rounded.
    let v = parse_data(&line)[0];
    let truth = 7.38905609893065_f64;
    assert!(
        (v - truth).abs() < 1e-12,
        "f64 tensor exp must be f64-precise (within 1e-12 of {truth}); got {v} in: {line}"
    );
}

// ===========================================================================
// chelis#717 - f32 tensors: add/div/recip skip the f32 rounding (eval)
// ===========================================================================

/// Before the repair, eval printed 0.30000000447034836, a value that does not
/// exist in f32 (the unfinalized f64 sum of two f32 inputs).
#[test]
fn eval_f32_tensor_add_rounds_to_f32() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = add(x, y)\n\
         out = print(f(to_tensor([0.1, 1.0], f32), to_tensor([0.2, 2.0], f32)))\n",
    )
    .expect("eval should run");
    // chelis#729 Phase 0: 0.30000000447034836 is not an f32 value; the
    // domain checker is the mechanical form of this cell's claim.
    common::assert_elements_in_domain("f32", &line, "eval_f32_tensor_add");
    assert!(
        line.contains("data=[0.3, 3.0]"),
        "f32 tensor add must round its result to f32 (rendered at f32 \
         width per spec/05 section 8.1); got: {line}"
    );
}

/// The pre-repair value was 0.3333333333333333 (a raw f64 quotient).
#[test]
fn eval_f32_tensor_div_rounds_to_f32() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = div(x, y)\n\
         out = print(f(to_tensor([1.0, 2.0], f32), to_tensor([3.0, 3.0], f32)))\n",
    )
    .expect("eval should run");
    common::assert_elements_in_domain("f32", &line, "eval_f32_tensor_div");
    assert!(
        line.contains("data=[0.33333334, 0.6666667]"),
        "f32 tensor div must round its result to f32 (rendered at f32 \
         width per spec/05 section 8.1); got: {line}"
    );
}

/// The pre-repair value was 0.6666666666666666 (raw f64). This locks the
/// required f32 finalization after `recip`.
#[test]
fn eval_f32_tensor_recip_rounds_to_f32() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32]) -> tensor[2, f32] = recip(x)\n\
         out = print(f(to_tensor([1.5, 3.0], f32)))\n",
    )
    .expect("eval should run");
    common::assert_elements_in_domain("f32", &line, "eval_f32_tensor_recip");
    assert!(
        line.contains("data=[0.6666667, 0.33333334]"),
        "f32 tensor recip must round its result to f32 (rendered at f32 \
         width per spec/05 section 8.1); got: {line}"
    );
}

// ===========================================================================
// chelis#3041 - einsum accumulates at the default accumulator in both lanes
// ===========================================================================

/// `chelis_tensor_einsum` forms each product at the §5.7.1 default
/// accumulator (f32 for f16, bf16 and f32 operands) and sums them in the
/// runtime's balanced order; eval accumulated in f64 ([05-OP-51]: never an
/// unrequested f64 graph). Each witness rounds differently under the two:
/// f32 gave 16777220 in eval, bf16 0x3f81 and f16 0x3c01.
#[test]
fn einsum_accumulates_at_the_default_accumulator_in_eval_and_the_executable() {
    let source = "module Probe.Case\n\
         def wide() -> tensor[1, f32] = einsum(\"ij,j->i\", to_tensor([[16777216.0f32, 1.0f32, 1.0f32, 1.0f32]]), to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))\n\
         def brain() -> tensor[1, bf16] = einsum(\"ij,j->i\", to_tensor([[1.0bf16, 0.00390625bf16, 9.313225746154785e-10bf16]]), to_tensor([1.0bf16, 1.0bf16, 1.0bf16]))\n\
         def half() -> tensor[1, f16] = einsum(\"ij,j->i\", to_tensor([[1.0f16, 0.00048828125f16, 5.960464477539063e-8f16]]), to_tensor([1.0f16, 1.0f16, 1.0f16]))\n\
         a = print(wide())\n\
         b = print(brain())\n\
         c = print(half())\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(eval.status.success(), "{eval:?}");
    let eval = String::from_utf8(eval.stdout).expect("utf-8 stdout");
    let printed = eval.lines().take(3).collect::<Vec<_>>();
    assert_eq!(
        printed,
        [
            "tensor(shape=[1], data=[16777218.0])",
            "tensor(shape=[1], data=[1.0])",
            "tensor(shape=[1], data=[1.0])",
        ],
        "{eval}"
    );
    assert_eq!(common::build_and_run(source, "p"), eval, "{source}");
}

/// The `numeric trap:` line a failed run printed to stderr.
fn trap_line(output: &std::process::Output, lane: &str) -> String {
    assert!(!output.status.success(), "{lane} did not trap: {output:?}");
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .find(|line| line.starts_with("numeric trap:"))
        .unwrap_or_else(|| panic!("{lane} failed without a trap line: {output:?}"))
        .to_string()
}

/// An integer einsum that leaves the i32 accumulator traps as `einsum` in
/// both lanes, whether the balanced sum (`[[2147483647, 1]]·[1, 1]`) or a
/// product (`[[65536]]·[65536]`) overflows; eval once reported the `add` or
/// `mul` the contraction was formed from.
#[test]
fn integer_einsum_overflow_traps_as_einsum_in_eval_and_the_executable() {
    for (name, lhs, rhs) in [
        ("sum", "[[2147483647i32, 1i32]]", "[1i32, 1i32]"),
        ("product", "[[65536i32]]", "[65536i32]"),
    ] {
        let source = format!(
            "module Probe.Case\n\
             def main() -> tensor[1, i32] = einsum(\"ij,j->i\", to_tensor({lhs}), to_tensor({rhs}))\n"
        );
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(format!("{name}.ch"));
        let out_dir = dir.path().join(format!("{name}-out"));
        write_file(&path, &source);
        let eval = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file", path.to_str().unwrap()])
            .output()
            .expect("chelis eval should run");
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
        let run = std::process::Command::new(out_dir.join(name))
            .output()
            .expect("compiled binary should run");
        let expected = "numeric trap: overflow in einsum at i32";
        assert_eq!(trap_line(&run, "the executable"), expected, "{source}");
        assert_eq!(trap_line(&eval, "eval"), expected, "{source}");
    }
}
