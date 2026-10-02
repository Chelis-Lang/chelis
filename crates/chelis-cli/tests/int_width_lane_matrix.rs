//! chelis#718 regression matrix. Before #729 Phase 3, integer width semantics
//! were inverted between lanes and surfaces:
//!
//! | surface  | `chelis eval`        | compiled C            |
//! |----------|----------------------|-----------------------|
//! | scalar   | wraps at width (-56) | escapes width (200)   |
//! | tensor   | escapes width (200.0)| wraps at width (-56)  |
//!
//! Each lane was width-correct exactly where the other was width-less, so no
//! overflowing narrow-int program agreed across lanes. This also corrects a
//! historical claim in chelis#695 ("the C backend currently wraps natively"):
//! it was true only for tensors; the scalar C lane computed at i64 width and returned values
//! that do not exist in the declared type (`neg(-128i8) = 128`).
//!
//! The decided contract (chelis#680/#695): overflow TRAPS with a branded
//! diagnostic, at every width, in both lanes. The tests assert that contract
//! per cell so the fix cannot land on one surface only.
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
     def run() -> i8 = add(cast(100, i8), cast(100, i8))\n\
     out = print(run())\n";

const TENSOR_I8_OVERFLOW: &str = "module M.Main\n\
     def f(x: tensor[2, i8], y: tensor[2, i8]) -> tensor[2, i8] = add(x, y)\n\
     out = print(f(to_tensor([cast(100, i8), cast(1, i8)]), \
     to_tensor([cast(100, i8), cast(2, i8)])))\n";

// ===========================================================================
// chelis#718 - the three unlocked overflow cells (eval-scalar is in
// precision_matrix.rs). Each asserts the decided trap contract.
// ===========================================================================

/// Before Phase 3 the compiled binary printed `200` - a value that does not
/// exist in i8. No wrap, no trap; the width is simply absent (the scalar
/// travels as `int64_t`, chelis#714's mechanism).
#[test]
fn c_scalar_int8_add_overflow_traps() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (line, stderr, ok) = c_lane(SCALAR_I8_OVERFLOW, "c_i8_scalar_ovf").expect("C lane");
    if ok {
        // chelis#729 Phase 0: if the lane produced a value instead of
        // trapping, that value must at least be a member of i8 (the
        // historical value 200 is the mechanical detection of #718).
        common::assert_elements_in_domain("i8", &line, "c_i8_scalar_ovf");
    }
    assert!(
        !ok && stderr.contains("overflow"),
        "i8 100 + 100 must trap with a branded overflow diagnostic; \
         got exit ok={ok}, stdout `{line}`, stderr `{stderr}`"
    );
}

/// Before Phase 3 this produced `128` from the compiled binary - not an i8 value.
/// (eval wraps to -128, which the contract also forbids, but at least stays
/// in range; see precision_matrix.rs for the eval rows.)
#[test]
fn c_scalar_int8_neg_min_traps() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (line, stderr, ok) = c_lane(
        "module M.Main\ndef run() -> i8 = neg(cast(-128, i8))\nout = print(run())\n",
        "c_i8_neg_min",
    )
    .expect("C lane");
    if ok {
        common::assert_elements_in_domain("i8", &line, "c_i8_neg_min");
    }
    assert!(
        !ok && stderr.contains("overflow"),
        "neg(i8::MIN) must trap; got exit ok={ok}, stdout `{line}`, stderr `{stderr}`"
    );
}

/// Before the typed-storage repair eval printed `data=[200.0, 3.0]` - no width applied
/// (f64 storage, chelis#684), and float-formatted integers to boot.
#[test]
fn eval_tensor_int8_add_overflow_traps() {
    let result = eval_first_line(TENSOR_I8_OVERFLOW);
    match result {
        Ok(line) => panic!("i8 tensor overflow must trap, but eval printed: {line}"),
        Err(stderr) => assert!(
            stderr.contains("overflow"),
            "the trap must carry a branded overflow diagnostic; got: {stderr}"
        ),
    }
}

/// Before Phase 3 the compiled binary printed `data=[-56, 3]` - a silent
/// two's-complement wrap in genuine int8_t buffers.
#[test]
fn c_tensor_int8_add_overflow_traps() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (line, stderr, ok) = c_lane(TENSOR_I8_OVERFLOW, "c_i8_tensor_ovf").expect("C lane");
    assert!(
        !ok && stderr.contains("overflow"),
        "i8 tensor overflow must trap; got exit ok={ok}, stdout `{line}`, stderr `{stderr}`"
    );
}

/// The remaining scalar overflow cells from the probe battery
/// (`docs/investigations/probes/bat_narrow.py`), one row per width and op
/// shape. Before Phase 3, compiled C printed i8 mul as 256, i16 add as
/// 60000, and i32 add as 4000000000 - values that do not exist
/// in the declared types (the int64_t widening, chelis#714's mechanism).
/// eval wraps to 0 / -5536 / -294967296 respectively. The contract says
/// every cell traps.
#[test]
fn c_scalar_overflow_traps_at_every_width() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let rows: &[(&str, &str)] = &[
        ("mul(cast(16, i8), cast(16, i8))", "i8"),
        ("add(cast(30000, i16), cast(30000, i16))", "i16"),
        ("add(cast(2000000000, i32), cast(2000000000, i32))", "i32"),
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

/// 126 + 1 = 127 fits i8 exactly; both lanes must print 127 and keep
/// printing it after the overflow fix lands.
#[test]
fn in_range_int8_scalar_add_agrees_across_lanes() {
    let program = "module M.Main\n\
         def run() -> i8 = add(cast(126, i8), cast(1, i8))\n\
         out = print(run())\n";
    let eval_line = eval_first_line(program).expect("eval");
    common::assert_elements_in_domain("i8", &eval_line, "i8_in_range eval");
    assert_eq!(eval_line, "127");
    if c_toolchain_available() {
        let (line, _, ok) = c_lane(program, "c_i8_in_range").expect("C lane");
        assert!(ok);
        common::assert_elements_in_domain("i8", &line, "i8_in_range C");
        assert_eq!(line, "127");
    }
}

/// A function-typed i8 argument is a real callable value, not a nullable
/// numeric slot. The C lane may specialize this closed call or emit a callable
/// symbol, but emitting `0` for the argument is a forbidden §C6.3 callback
/// substitution and traps when `apply` invokes it.
#[test]
fn in_range_int8_inline_callback_executes_exactly() {
    if !c_toolchain_available() {
        return;
    }
    let program = "module M.Main\n\
         def apply(f: i8 -> i8, x: i8) -> i8 = f(x)\n\
         out = print(apply(fn (x: i8) -> add(x, cast(1, i8)), cast(6, i8)))\n";
    let (line, stderr, ok) = c_lane(program, "c_i8_inline_callback").expect("C callback lane");
    assert!(
        ok,
        "the emitted callback must be callable; stdout `{line}`, stderr `{stderr}`"
    );
    common::assert_elements_in_domain("i8", &line, "i8 inline callback C");
    assert_eq!(line, "7");
}

/// Positive parity for the ordinary named-function-pointer representation.
/// The anonymous-call specialization must not weaken the existing C ABI path
/// for a named callback.
#[test]
fn in_range_int8_named_callback_executes_exactly() {
    if !c_toolchain_available() {
        return;
    }
    let program = "module M.Main\n\
         def apply(f: i8 -> i8, x: i8) -> i8 = f(x)\n\
         def increment(x: i8) -> i8 = add(x, cast(1, i8))\n\
         out = print(apply(increment, cast(6, i8)))\n";
    let (line, stderr, ok) = c_lane(program, "c_i8_named_callback").expect("C callback lane");
    assert!(
        ok,
        "named callback must execute; stdout `{line}`, stderr `{stderr}`"
    );
    common::assert_elements_in_domain("i8", &line, "i8 named callback C");
    assert_eq!(line, "7");
}

/// Positive parity for the second pre-existing narrow scalar ABI. Callback
/// projection must preserve the complete signature rather than widening or
/// erasing it while closing first-class function values.
#[test]
fn in_range_int16_named_callback_executes_exactly() {
    if !c_toolchain_available() {
        return;
    }
    let program = "module M.Main\n\
         def apply(f: i16 -> i16, x: i16) -> i16 = f(x)\n\
         def increment(x: i16) -> i16 = add(x, cast(2, i16))\n\
         out = print(apply(increment, cast(300, i16)))\n";
    let (line, stderr, ok) = c_lane(program, "c_i16_named_callback").expect("C callback lane");
    assert!(
        ok,
        "the emitted i16 callback must be callable; stdout `{line}`, stderr `{stderr}`"
    );
    common::assert_elements_in_domain("i16", &line, "i16 named callback C");
    assert_eq!(line, "302");
}

/// A directly-constructed generic record must substitute its applied type
/// before field access. The host boundary may not expose the declaration's
/// `a` term as if it were a concrete field type.
#[test]
fn in_range_int8_direct_generic_record_access_executes_exactly() {
    if !c_toolchain_available() {
        return;
    }
    let program = "module M.Main\n\
         type ReviewBox[a] =\n\
           | ReviewBox { value: a }\n\
         def direct() -> i8 = (ReviewBox { value: cast(7, i8) }).value\n\
         out = print(direct())\n";
    let (line, stderr, ok) =
        c_lane(program, "c_i8_direct_generic_access").expect("C generic-record lane");
    assert!(
        ok,
        "direct generic field access must execute; stdout `{line}`, stderr `{stderr}`"
    );
    common::assert_elements_in_domain("i8", &line, "direct generic i8 access C");
    assert_eq!(line, "7");
}

/// Nested generic fields require recursive substitution at every declaration
/// boundary, including the final access through `ReviewBox[a]`.
#[test]
fn in_range_int8_nested_generic_record_access_executes_exactly() {
    if !c_toolchain_available() {
        return;
    }
    let program = "module M.Main\n\
         type ReviewBox[a] =\n\
           | ReviewBox { value: a }\n\
         type ReviewEnvelope[a] =\n\
           | ReviewEnvelope { inner: ReviewBox[a] }\n\
         def open(envelope: ReviewEnvelope[i8]) -> i8 = envelope.inner.value\n\
         out = print(open(ReviewEnvelope {\n\
           inner: ReviewBox { value: cast(7, i8) }\n\
         }))\n";
    let (line, stderr, ok) =
        c_lane(program, "c_i8_nested_generic_access").expect("C nested-generic lane");
    assert!(
        ok,
        "nested generic field access must execute; stdout `{line}`, stderr `{stderr}`"
    );
    common::assert_elements_in_domain("i8", &line, "nested generic i8 access C");
    assert_eq!(line, "7");
}

/// In-range i16 tensor arithmetic agrees across lanes: both lanes hold
/// value-100 elements. Since chelis#732 Phase 1, eval prints integer
/// tensor elements as integers ([05-OBS-2]); the compiled lane keeps its
/// pre-contract float form until Phase 2.
#[test]
fn in_range_int16_tensor_add_agrees_across_lanes() {
    let program = "module M.Main\n\
         def f(x: tensor[2, i16], y: tensor[2, i16]) -> tensor[2, i16] = add(x, y)\n\
         out = print(f(to_tensor([cast(60, i16), cast(1, i16)]), \
         to_tensor([cast(40, i16), cast(2, i16)])))\n";
    let eval_line = eval_first_line(program).expect("eval");
    common::assert_elements_in_domain("i16", &eval_line, "i16_in_range eval");
    assert!(
        eval_line.contains("data=[100, 3]"),
        "eval i16 tensor add of in-range values; got: {eval_line}"
    );
    if c_toolchain_available() {
        let (line, _, ok) = c_lane(program, "c_i16_in_range").expect("C lane");
        assert!(ok);
        common::assert_elements_in_domain("i16", &line, "i16_in_range C");
        assert!(
            line.contains("data=[100.0, 3.0]") || line.contains("data=[100, 3]"),
            "C i16 tensor add of in-range values; got: {line}"
        );
    }
}
