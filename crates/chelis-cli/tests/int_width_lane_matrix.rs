//! chelis#718 - integer width semantics are INVERTED between lanes and
//! surfaces. What happens when an int8/int16/int32 result exceeds its width
//! depends on both which lane runs it and whether the value is a scalar or
//! a tensor:
//!
//! | surface  | `chelis eval`        | compiled C            |
//! |----------|----------------------|-----------------------|
//! | scalar   | wraps at width (-56) | escapes width (200)   |
//! | tensor   | escapes width (200.0)| wraps at width (-56)  |
//!
//! Each lane is width-correct exactly where the other is width-less, so no
//! overflowing narrow-int program agrees across lanes. This also corrects a
//! claim in chelis#695 ("the C backend currently wraps natively"): true only
//! for tensors; the scalar C lane computes at int64 width and returns values
//! that do not exist in the declared type (`neg(-128i8) = 128`).
//!
//! The decided contract (chelis#680/#695): overflow TRAPS with a branded
//! diagnostic, at every width, in both lanes. The `#[ignore]`d tests assert
//! that contract per cell so the fix cannot land on one surface only.
//! precision_matrix.rs already carries the eval-scalar trap rows; this file
//! adds the other three cells plus in-range controls.

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

/// Build + link + run; `Ok((first stdout line, full stderr))`. A non-zero
/// exit is NOT an error here - the trap rows check stderr on failure.
fn c_lane(program: &str, name: &str) -> Result<(String, String, bool), String> {
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
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    Ok((
        String::from_utf8_lossy(&run.stdout)
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
        run.status.success(),
    ))
}

const SCALAR_I8_OVERFLOW: &str = "module M.Main\n\
     def run() -> int8 = add(cast(100, int8), cast(100, int8))\n\
     out = print(run())\n";

const TENSOR_I8_OVERFLOW: &str = "module M.Main\n\
     def f(x: tensor[2, int8], y: tensor[2, int8]) -> tensor[2, int8] = add(x, y)\n\
     out = print(f(to_tensor([cast(100, int8), cast(1, int8)]), \
     to_tensor([cast(100, int8), cast(2, int8)])))\n";

// ===========================================================================
// chelis#718 - the three unlocked overflow cells (eval-scalar is in
// precision_matrix.rs). Each asserts the decided trap contract.
// ===========================================================================

/// Observed today: the compiled binary prints `200` - a value that does not
/// exist in int8. No wrap, no trap; the width is simply absent (the scalar
/// travels as `int64_t`, chelis#714's mechanism).
#[test]
#[ignore = "chelis#718: compiled C scalar int8 add(100, 100) prints 200 (no width); eval \
            wraps to -56; the contract says both must trap. Run with \
            `cargo test -p chelis-cli --test int_width_lane_matrix -- --ignored`."]
fn c_scalar_int8_add_overflow_traps() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (line, stderr, ok) = c_lane(SCALAR_I8_OVERFLOW, "c_i8_scalar_ovf").expect("C lane");
    if ok {
        // chelis#729 Phase 0: if the lane produced a value instead of
        // trapping, that value must at least be a member of int8 (today
        // it prints 200, which is the mechanical detection of #718).
        common::assert_elements_in_domain("int8", &line, "c_i8_scalar_ovf");
    }
    assert!(
        !ok && stderr.contains("overflow"),
        "int8 100 + 100 must trap with a branded overflow diagnostic; \
         got exit ok={ok}, stdout `{line}`, stderr `{stderr}`"
    );
}

/// Observed today: `128` from the compiled binary - not an int8 value.
/// (eval wraps to -128, which the contract also forbids, but at least stays
/// in range; see precision_matrix.rs for the eval rows.)
#[test]
#[ignore = "chelis#718: compiled C neg(cast(-128, int8)) prints 128, not an int8 value; \
            the contract says overflow must trap. Run with \
            `cargo test -p chelis-cli --test int_width_lane_matrix -- --ignored`."]
fn c_scalar_int8_neg_min_traps() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (line, stderr, ok) = c_lane(
        "module M.Main\ndef run() -> int8 = neg(cast(-128, int8))\nout = print(run())\n",
        "c_i8_neg_min",
    )
    .expect("C lane");
    if ok {
        common::assert_elements_in_domain("int8", &line, "c_i8_neg_min");
    }
    assert!(
        !ok && stderr.contains("overflow"),
        "neg(i8::MIN) must trap; got exit ok={ok}, stdout `{line}`, stderr `{stderr}`"
    );
}

/// Observed today: eval prints `data=[200.0, 3.0]` - no width applied
/// (f64 storage, chelis#684), and float-formatted integers to boot.
#[test]
#[ignore = "chelis#718: eval int8 TENSOR add(100, 100) prints 200.0 (no width; f64 \
            storage); compiled C wraps to -56; the contract says both must trap. Run with \
            `cargo test -p chelis-cli --test int_width_lane_matrix -- --ignored`."]
fn eval_tensor_int8_add_overflow_traps() {
    let result = eval_first_line(TENSOR_I8_OVERFLOW);
    match result {
        Ok(line) => panic!("int8 tensor overflow must trap, but eval printed: {line}"),
        Err(stderr) => assert!(
            stderr.contains("overflow"),
            "the trap must carry a branded overflow diagnostic; got: {stderr}"
        ),
    }
}

/// Observed today: the compiled binary prints `data=[-56, 3]` - a silent
/// two's-complement wrap in genuine int8_t buffers.
#[test]
#[ignore = "chelis#718: compiled C int8 TENSOR add(100, 100) silently wraps to -56; the \
            contract says overflow must trap, and eval disagrees (200.0) besides. Run with \
            `cargo test -p chelis-cli --test int_width_lane_matrix -- --ignored`."]
fn c_tensor_int8_add_overflow_traps() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (line, stderr, ok) = c_lane(TENSOR_I8_OVERFLOW, "c_i8_tensor_ovf").expect("C lane");
    assert!(
        !ok && stderr.contains("overflow"),
        "int8 tensor overflow must trap; got exit ok={ok}, stdout `{line}`, stderr `{stderr}`"
    );
}

/// The remaining scalar overflow cells from the probe battery
/// (`docs/investigations/probes/bat_narrow.py`), one row per width and op
/// shape. Observed today in compiled C: int8 mul prints 256, int16 add
/// prints 60000, int32 add prints 4000000000 - values that do not exist
/// in the declared types (the int64_t widening, chelis#714's mechanism).
/// eval wraps to 0 / -5536 / -294967296 respectively. The contract says
/// every cell traps.
#[test]
#[ignore = "chelis#718: compiled C narrow-int scalar overflow escapes the width at every \
            width (int8 mul 256, int16 add 60000, int32 add 4000000000); eval wraps; the \
            contract says both trap. Run with \
            `cargo test -p chelis-cli --test int_width_lane_matrix -- --ignored`."]
fn c_scalar_overflow_traps_at_every_width() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let rows: &[(&str, &str)] = &[
        ("mul(cast(16, int8), cast(16, int8))", "int8"),
        ("add(cast(30000, int16), cast(30000, int16))", "int16"),
        (
            "add(cast(2000000000, int32), cast(2000000000, int32))",
            "int32",
        ),
    ];
    for (i, (expr, ret_ty)) in rows.iter().enumerate() {
        let program =
            format!("module M.Main\ndef run() -> {ret_ty} = {expr}\nout = print(run())\n");
        let (line, stderr, ok) = c_lane(&program, &format!("c_ovf_{i}")).expect("C lane");
        if ok {
            common::assert_elements_in_domain(ret_ty, &line, expr);
        }
        assert!(
            !ok && stderr.contains("overflow"),
            "`{expr}` must trap with a branded overflow diagnostic; \
             got exit ok={ok}, stdout `{line}`, stderr `{stderr}`"
        );
    }
}

// ===========================================================================
// CONTROLS: in-range narrow-int arithmetic agrees across lanes and surfaces.
// A trap fix must not fire on values that fit.
// ===========================================================================

/// 126 + 1 = 127 fits int8 exactly; both lanes must print 127 and keep
/// printing it after the overflow fix lands.
#[test]
fn in_range_int8_scalar_add_agrees_across_lanes() {
    let program = "module M.Main\n\
         def run() -> int8 = add(cast(126, int8), cast(1, int8))\n\
         out = print(run())\n";
    let eval_line = eval_first_line(program).expect("eval");
    common::assert_elements_in_domain("int8", &eval_line, "i8_in_range eval");
    assert_eq!(eval_line, "127");
    if c_toolchain_available() {
        let (line, _, ok) = c_lane(program, "c_i8_in_range").expect("C lane");
        assert!(ok);
        common::assert_elements_in_domain("int8", &line, "i8_in_range C");
        assert_eq!(line, "127");
    }
}

/// In-range int16 tensor arithmetic agrees across lanes: both lanes hold
/// value-100 elements. Since chelis#732 Phase 1, eval prints integer
/// tensor elements as integers ([05-OBS-2]); the compiled lane keeps its
/// pre-contract float form until Phase 2.
#[test]
fn in_range_int16_tensor_add_agrees_across_lanes() {
    let program = "module M.Main\n\
         def f(x: tensor[2, int16], y: tensor[2, int16]) -> tensor[2, int16] = add(x, y)\n\
         out = print(f(to_tensor([cast(60, int16), cast(1, int16)]), \
         to_tensor([cast(40, int16), cast(2, int16)])))\n";
    let eval_line = eval_first_line(program).expect("eval");
    common::assert_elements_in_domain("int16", &eval_line, "i16_in_range eval");
    assert!(
        eval_line.contains("data=[100, 3]"),
        "eval int16 tensor add of in-range values; got: {eval_line}"
    );
    if c_toolchain_available() {
        let (line, _, ok) = c_lane(program, "c_i16_in_range").expect("C lane");
        assert!(ok);
        common::assert_elements_in_domain("int16", &line, "i16_in_range C");
        assert!(
            line.contains("data=[100.0, 3.0]") || line.contains("data=[100, 3]"),
            "C int16 tensor add of in-range values; got: {line}"
        );
    }
}
