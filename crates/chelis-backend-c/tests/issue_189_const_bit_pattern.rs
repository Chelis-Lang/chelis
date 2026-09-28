//! Issue #189: scalar-f32 evaluator vs C backend divergence.
//!
//! The C backend `emit_const` F32 arm narrows the IR's f64 source value
//! to f32 (intentional) and then formats the resulting f32 via a `%.8`
//! format string (lossy). For values in the `1e-7` magnitude range
//! that second step drops most of the significant digits, so the
//! generated C reproduces a value up to ~3% off the closest-f32 value
//! the evaluator's `Vec<f64>` storage would yield after `as f32`. The
//! same lossy pattern lives on the F64 arm (`%.17` format, which
//! treats `.17` as decimal places after the point, not significant
//! digits, and zeroes out values below `1e-17`).
//!
//! Acceptance: the C backend emits bit-identical reproductions of the
//! narrowed f32 (and the source f64) by emitting their bit patterns
//! and bit-casting at runtime. The pre-fix emitter cannot satisfy
//! these tests because its format strings discard information; these
//! fixtures are written before the fix lands to lock the contract.

mod support;
use chelis_ir::dag::{Dag, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::emit_dag;

mod common;

fn scalar(p: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision: p,
    }
}

/// The reproducer from issue #189: `cast(0.000000123456789, f32)`.
/// Pre-fix the F32 arm emits `chelis_fill_f32(t0, 0.00000012f);` which
/// parses to f32 bits `0x3400d959` (about `1.2e-7`). The closest-f32
/// to the source value is `f32::from_bits(0x34048f8b)` (about
/// `1.2345679e-7`). Post-fix the emitter must produce the latter bit
/// pattern verbatim.
#[test]
fn issue_189_f32_const_emits_exact_bit_pattern() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    dag.add_node(
        decl,
        RiscOp::synth_const(scalar(Prim::F32).precision, 0.000000123456789_f64),
        vec![],
        scalar(Prim::F32),
        None,
    );
    let src = emit_dag(&dag, "test_fn").unwrap();
    let want_bits = (0.000000123456789_f64 as f32).to_bits();
    let needle = format!("0x{want_bits:08x}");
    assert!(
        src.contains(&needle),
        "F32 const must round-trip via exact bit pattern `{needle}`; emitted source:\n{src}"
    );
    // The exact tagged scalar keeps the bit pattern and dtype coupled
    // through the single public fill entry point.
    assert!(
        src.contains(
            "chelis_fill_scalar(t0_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F32,"
        ),
        "F32 const must dispatch through the exact tagged fill; emitted source:\n{src}"
    );
}

/// Same shape for f64. Pre-fix the F64 arm emits
/// `chelis_fill_f64(t0, 0.00000000000000000)` for values like `1e-300`
/// because `{:.17}` formats 17 digits after the decimal point. Post-fix
/// the bit pattern of the source f64 must appear verbatim.
#[test]
fn issue_189_f64_const_emits_exact_bit_pattern() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let v: f64 = 1.0e-300;
    dag.add_node(
        decl,
        RiscOp::synth_const(scalar(Prim::F64).precision, v),
        vec![],
        scalar(Prim::F64),
        None,
    );
    let src = emit_dag(&dag, "test_fn").unwrap();
    let want_bits = v.to_bits();
    let needle = format!("0x{want_bits:016x}");
    assert!(
        src.contains(&needle),
        "F64 const must round-trip via exact bit pattern `{needle}`; emitted source:\n{src}"
    );
    assert!(
        src.contains(
            "chelis_fill_scalar(t0_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F64,"
        ),
        "F64 const must dispatch through the exact tagged fill; emitted source:\n{src}"
    );
}

/// Negative-parity guard. The pre-fix lossy literals must never
/// re-emerge in F32 const emission. Stamp the canonical reproducer
/// from issue #189 plus a denormal and the smallest positive normal so
/// a future format-string refactor that re-introduces `%.8` is caught
/// here directly.
#[test]
fn issue_189_f32_const_does_not_use_lossy_format() {
    let values: &[f64] = &[
        0.000000123456789_f64,
        f32::MIN_POSITIVE as f64,
        // Smallest positive denormal.
        f32::from_bits(0x0000_0001).into(),
        0.1_f64,
    ];
    for &v in values {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
            RiscOp::synth_const(scalar(Prim::F32).precision, v),
            vec![],
            scalar(Prim::F32),
            None,
        );
        let src = emit_dag(&dag, "test_fn").unwrap();
        let v32 = v as f32;
        let want_bits = v32.to_bits();
        // The bit pattern must be present.
        assert!(
            src.contains(&format!("0x{want_bits:08x}")),
            "F32 const for v={v}: expected bit pattern `0x{want_bits:08x}`; \
             emitted source:\n{src}"
        );
    }
}

/// f32 denormal: the smallest positive subnormal f32
/// (`f32::from_bits(0x0000_0001)`, about `1.4e-45`). `{:.8}` flattens
/// this to `0.00000000` and parsing back yields zero. The bit-pattern
/// emitter must preserve the denormal pattern.
#[test]
fn issue_189_f32_const_smallest_denormal_round_trips() {
    let v = f32::from_bits(0x0000_0001);
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    dag.add_node(
        decl,
        RiscOp::synth_const(scalar(Prim::F32).precision, v as f64),
        vec![],
        scalar(Prim::F32),
        None,
    );
    let src = emit_dag(&dag, "test_fn").unwrap();
    assert!(
        src.contains("0x00000001"),
        "f32 denormal must round-trip via bit pattern `0x00000001`; emitted source:\n{src}"
    );
}

/// f64 small-magnitude sibling: `0.1f64` has bits `0x3fb999999999999a`.
/// `{:.17}` happens to round-trip 0.1 itself, but the closely related
/// `0.1_f64.next_up()` differs by one ULP and the lossy format may
/// collapse the pair. Pin both with bit-pattern verification.
#[test]
fn issue_189_f64_const_one_ulp_pair_round_trips() {
    let v1: f64 = 0.1;
    let v2: f64 = f64::from_bits(v1.to_bits() + 1);
    for v in [v1, v2] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
            RiscOp::synth_const(scalar(Prim::F64).precision, v),
            vec![],
            scalar(Prim::F64),
            None,
        );
        let src = emit_dag(&dag, "test_fn").unwrap();
        let bits = v.to_bits();
        assert!(
            src.contains(&format!("0x{bits:016x}")),
            "f64 one-ULP pair member v={v}: expected `0x{bits:016x}`; \
             emitted source:\n{src}"
        );
    }
}

// ---------------------------------------------------------------
// End-to-end byte-identical compile + run tests.
//
// These are the strongest evidence the bit-pattern emission is
// correct: emit C, gcc-compile, run, read the f32 / f64 value back as
// its u32 / u64 bit pattern, and require it match the source value's
// bit pattern exactly.
// ---------------------------------------------------------------

use std::fs;
use std::process::Command;
use support::codegen;

fn compile_and_run(test_name: &str, c_source: &str, harness: &str) -> Option<String> {
    let probe = common::probe_dir(&format!("issue189_{test_name}"));
    let dir = probe.path().to_path_buf();
    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), harness).unwrap();
    let staged = chelis_runtime_bundle::stage(&dir)
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    let bin = dir.join("test_bin");
    let runtime_lib = staged.archive;
    let compile = Command::new("gcc")
        .args([
            "-O2",
            "-std=c11",
            "-I",
            dir.to_str().unwrap(),
            dir.join("kernel.c").to_str().unwrap(),
            dir.join("main.c").to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            runtime_lib.to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-ldl",
        ])
        .output()
        .expect("failed to invoke gcc");
    if !compile.status.success() {
        eprintln!(
            "COMPILE FAILED [{test_name}]:\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );
        eprintln!("Kernel C:\n{c_source}");
        return None;
    }
    let run = Command::new(&bin).output().expect("failed to run binary");
    if !run.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&run.stdout).into_owned())
}

/// Acceptance oracle: emitted C, compiled by gcc, must produce an f32
/// value whose bit pattern matches `(value as f32).to_bits()` for the
/// issue #189 reproducer. Prior to the fix this test would print a
/// different hex (about 3% off).
#[test]
fn issue_189_f32_const_byte_identical_to_eval_under_gcc() {
    let v: f64 = 0.000000123456789;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    dag.add_node(
        decl,
        RiscOp::synth_const(
            TensorType {
                dims: vec![chelis_ir::dag::DimInfo::Lit(1)],
                precision: Prim::F32,
            }
            .precision,
            v,
        ),
        vec![],
        TensorType {
            dims: vec![chelis_ir::dag::DimInfo::Lit(1)],
            precision: Prim::F32,
        },
        None,
    );
    let result = codegen(&dag, "test_const_f32").unwrap();
    let src = &result.c_source;
    let want_bits = (v as f32).to_bits();
    let harness = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_runtime.h"

extern void test_const_f32(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main(void) {
    chelis_tensor* outputs[1] = { NULL };
    test_const_f32(NULL, 0, outputs, 1);
    chelis_read_view view = chelis_tensor_read_view(outputs[0]);
    float v;
    memcpy(&v, view.data, sizeof(float));
    uint32_t bits;
    memcpy(&bits, &v, sizeof(uint32_t));
    printf("0x%08x\n", bits);
    chelis_tensor_release(outputs[0]);
    return 0;
}
"#;
    let Some(output) = compile_and_run("f32_const_byte_id", src, harness) else {
        panic!("emitted C did not compile/run");
    };
    let expected = format!("0x{want_bits:08x}");
    assert!(
        output.trim() == expected,
        "f32 byte-identical mismatch: expected `{expected}`, got `{}`. Source:\n{src}",
        output.trim()
    );
}

/// f64 sibling: same acceptance check for `f64::to_bits()`. The
/// pre-fix emitter would format `1.0e-300` as `0.00000000000000000`,
/// the gcc-parsed literal would be zero, and the runtime would print
/// `0x0000000000000000`. Post-fix the runtime must print the source
/// f64 bit pattern verbatim.
#[test]
fn issue_189_f64_const_byte_identical_to_eval_under_gcc() {
    let v: f64 = 1.0e-300;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    dag.add_node(
        decl,
        RiscOp::synth_const(
            TensorType {
                dims: vec![chelis_ir::dag::DimInfo::Lit(1)],
                precision: Prim::F64,
            }
            .precision,
            v,
        ),
        vec![],
        TensorType {
            dims: vec![chelis_ir::dag::DimInfo::Lit(1)],
            precision: Prim::F64,
        },
        None,
    );
    let result = codegen(&dag, "test_const_f64").unwrap();
    let src = &result.c_source;
    let want_bits = v.to_bits();
    let harness = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "chelis_runtime.h"

extern void test_const_f64(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main(void) {
    chelis_tensor* outputs[1] = { NULL };
    test_const_f64(NULL, 0, outputs, 1);
    chelis_read_view view = chelis_tensor_read_view(outputs[0]);
    double v;
    memcpy(&v, view.data, sizeof(double));
    uint64_t bits;
    memcpy(&bits, &v, sizeof(uint64_t));
    printf("0x%016lx\n", (unsigned long)bits);
    chelis_tensor_release(outputs[0]);
    return 0;
}
"#;
    let Some(output) = compile_and_run("f64_const_byte_id", src, harness) else {
        panic!("emitted C did not compile/run");
    };
    let expected = format!("0x{want_bits:016x}");
    assert!(
        output.trim() == expected,
        "f64 byte-identical mismatch: expected `{expected}`, got `{}`. Source:\n{src}",
        output.trim()
    );
}
