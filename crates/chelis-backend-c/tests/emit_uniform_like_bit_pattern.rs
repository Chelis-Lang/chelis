//! Issue #248 (#189 follow-up): `emit_uniform_like` was the third
//! lossy `%.8`-format-string site in the C backend. The F32/F64 Const
//! and Pad sites landed in PR #243 (#189); this sibling on the
//! `chelis_uniform_sample_f32` call's `low` / `high` arguments stayed
//! unfixed and is closed here.
//!
//! Pre-fix the emitter wrote:
//!
//! ```text
//! t1->data[i] = chelis_uniform_sample_f32(t1_seed, (uint64_t)i,
//!                                         0.00000000f, 0.50000000f);
//! ```
//!
//! ...for `low = 1e-40`, narrowing the sub-normal-range value to f32
//! and then losing it entirely to the `%.8` format string (gcc parses
//! "0.00000000f" as 0.0f).
//!
//! Post-fix the emitter reconstructs each f32 argument from its exact
//! bit pattern via the `chelis_f32_from_bits` static-inline helper
//! declared in `chelis_runtime.h`, symmetric with PR #243's
//! `chelis_fill_f32_bits` mechanism.

mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::emit_dag;

mod common;

fn tensor(precision: Prim, size: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(size)],
        precision,
    }
}

fn build_uniform_like_dag(low: f64, high: f64, seed: u64) -> Dag {
    build_uniform_like_dag_for(Prim::F32, low, high, seed)
}

/// `uniform_like(key_from_seed(seed), template, low, high)`.
fn build_uniform_like_dag_for(precision: Prim, low: f64, high: f64, seed: u64) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let template = dag.add_node(
        decl,
        RiscOp::synth_const(precision, 0.0),
        vec![],
        tensor(precision, 4),
        None,
    );
    let rank0 = |precision| TensorType {
        dims: vec![],
        precision,
    };
    let low = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, low),
        vec![],
        rank0(Prim::F32),
        None,
    );
    let high = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, high),
        vec![],
        rank0(Prim::F32),
        None,
    );
    let seed = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::Int64, seed as f64),
        vec![],
        rank0(Prim::Int64),
        None,
    );
    let key = dag.add_node(
        decl,
        RiscOp::KeyFromSeed,
        vec![seed],
        rank0(Prim::Key),
        None,
    );
    let draw = dag.add_node(
        decl,
        RiscOp::UniformLike,
        vec![template, low, high, key],
        tensor(precision, 4),
        None,
    );
    dag.add_root(draw);
    dag
}

/// The output slot of the graph's draw.
fn draw_slot(dag: &Dag) -> String {
    format!("t{}_data", dag.roots()[0].0)
}

#[test]
fn issue_937_uniform_like_emits_dtype_specific_sampler_and_storage() {
    let f64_dag = build_uniform_like_dag_for(Prim::F64, 0.1, 0.9, 17);
    let slot = draw_slot(&f64_dag);
    let f64_src = emit_dag(&f64_dag, "uniform_f64").unwrap();
    assert!(f64_src.contains("static inline double chelis_uniform_sample_f64("));
    assert!(f64_src.contains(&format!(
        "((double*){slot})[i] = chelis_uniform_sample_f64("
    )));
    assert!(
        !f64_src
            .lines()
            .any(|line| line.contains(&format!("{slot})[i]")) && line.contains("sample_f32")),
        "f64 output must never widen an f32 sample:\n{f64_src}"
    );

    for (precision, conversion) in [
        (Prim::F16, "chelis_f32_to_f16"),
        (Prim::Bf16, "chelis_f32_to_bf16"),
    ] {
        let dag = build_uniform_like_dag_for(precision, 0.1, 0.9, 17);
        let slot = draw_slot(&dag);
        let src = emit_dag(&dag, "uniform_reduced").unwrap();
        assert!(src.contains("chelis_uniform_sample_f32("));
        assert!(src.contains(&format!("((uint16_t*){slot})[i] = {conversion}(")));
    }
}

/// Bit-pattern reproducer: the sub-normal-range `1e-40` reproducer
/// from the issue. The closest f32 to `1.0e-40_f64` is a denormal
/// (`f32::from_bits(0x000116c2)`), and the emitted C must reconstruct
/// that bit pattern verbatim via `chelis_f32_from_bits`.
#[test]
fn issue_248_uniform_like_low_arg_emits_exact_bit_pattern() {
    let low: f64 = 1.0e-40;
    let high: f64 = 0.5;
    let dag = build_uniform_like_dag(low, high, 42);
    let src = emit_dag(&dag, "test_fn").unwrap();

    let low_bits = (low as f32).to_bits();
    let high_bits = (high as f32).to_bits();
    let low_needle = format!("0x{low_bits:08x}");
    let high_needle = format!("0x{high_bits:08x}");

    assert!(
        src.contains(&low_needle),
        "uniform_like low arg must emit `{low_needle}`; emitted source:\n{src}"
    );
    assert!(
        src.contains(&high_needle),
        "uniform_like high arg must emit `{high_needle}`; emitted source:\n{src}"
    );
}

/// Negative-parity guard: the pre-fix `%.8f`-style decimal literal
/// must never re-emerge for ordinary or subnormal-range values. Stamps
/// the issue's reproducer plus a one-ULP-off ordinary value (`0.1`)
/// where `%.8` round-trips one ULP short of the source-narrowing.
#[test]
fn issue_248_uniform_like_does_not_use_lossy_format() {
    let cases: &[(f64, f64)] = &[
        (1.0e-40, 0.5),
        (0.1, 0.9),
        (f32::MIN_POSITIVE as f64, 1.0),
        // smallest positive f32 denormal
        (f32::from_bits(0x0000_0001) as f64, 1.0),
    ];
    for &(low, high) in cases {
        let dag = build_uniform_like_dag(low, high, 7);
        let src = emit_dag(&dag, "test_fn").unwrap();
        let low_bits = (low as f32).to_bits();
        let high_bits = (high as f32).to_bits();
        assert!(
            src.contains(&format!("0x{low_bits:08x}")),
            "uniform_like for low={low}: expected bit pattern `0x{low_bits:08x}`; emitted:\n{src}"
        );
        assert!(
            src.contains(&format!("0x{high_bits:08x}")),
            "uniform_like for high={high}: expected bit pattern `0x{high_bits:08x}`; emitted:\n{src}"
        );
        // Belt-and-braces: assert every call site of
        // `chelis_uniform_sample_f32` routes both scalar args through
        // `chelis_f32_from_bits`. Filter out the static-inline
        // definition line (the only non-call site that mentions the
        // symbol) by requiring the line to also reference a tensor
        // slot the call writes into. A future refactor that
        // hand-edits the `{:.8}f` literal back into the emit would
        // trip this on top of the bit-pattern assertions.
        for line in src.lines() {
            if line.contains("chelis_uniform_sample_f32(") && line.contains("->data[i]") {
                assert!(
                    line.contains("chelis_f32_from_bits"),
                    "uniform_like call line must route through `chelis_f32_from_bits`; got:\n  {line}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------
// End-to-end gcc-compile-and-run byte-identity test. Mirrors the
// issue_189_*_byte_identical_to_eval_under_gcc oracles: emit C,
// gcc-compile, run, and verify the sampler saw the byte-identical
// f32 narrowing of the source `low` / `high`. Linux-only because the
// emitted uniform_like kernel uses `-fopenmp`; macOS clang
// (masquerading as gcc) rejects that flag.
// ---------------------------------------------------------------

#[cfg(target_os = "linux")]
use std::fs;
#[cfg(target_os = "linux")]
use std::process::Command;
#[cfg(target_os = "linux")]
use support::codegen;

#[cfg(target_os = "linux")]
fn compile_and_run(test_name: &str, c_source: &str, harness: &str) -> Option<String> {
    let probe = common::probe_dir(&format!("issue248_{test_name}"));
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
            "-fopenmp",
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

/// Byte-identity acceptance oracle. Sets `low = high = 1e-40` so the
/// sampler returns exactly `low` for every element (with `high - low ==
/// 0` the lerp degenerates). Pre-fix that would print
/// `0x00000000` (the `%.8` literal collapsed to zero). Post-fix it
/// must print the f32 bit pattern of `(1.0e-40_f64 as f32)`
/// (`0x000116c2`, a denormal).
#[cfg(target_os = "linux")]
#[test]
fn issue_248_uniform_like_byte_identical_low_under_gcc() {
    let low: f64 = 1.0e-40;
    let high: f64 = 1.0e-40;
    let dag = build_uniform_like_dag(low, high, 0);
    let result = codegen(&dag, "test_uniform_like").unwrap();
    let src = &result.c_source;
    let want_bits = (low as f32).to_bits();
    let harness = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_runtime.h"

extern void test_uniform_like(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main(void) {
    chelis_tensor* outputs[1] = { NULL };
    test_uniform_like(NULL, 0, outputs, 1);
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
    let Some(output) = compile_and_run("uniform_like_low_byte_id", src, harness) else {
        panic!("emitted C did not compile/run");
    };
    let expected = format!("0x{want_bits:08x}");
    assert!(
        output.trim() == expected,
        "uniform_like low byte-identical mismatch: expected `{expected}`, got `{}`. Source:\n{src}",
        output.trim()
    );
}

/// chelis#937: f64 output executes the affine at f64 width. This Linux
/// compile-run lock reads the emitted buffer as f64 and compares raw bits
/// with the shared [05-OP-8] sampler, so widening f32 output cannot pass.
#[cfg(target_os = "linux")]
#[test]
fn issue_937_uniform_like_f64_matches_shared_sampler_under_gcc() {
    let dag = build_uniform_like_dag_for(Prim::F64, 2.0, 5.0, 42);
    let result = codegen(&dag, "test_uniform_like_f64").unwrap();
    let src = &result.c_source;
    let harness = r#"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "chelis_runtime.h"

extern void test_uniform_like_f64(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main(void) {
    chelis_tensor* outputs[1] = { NULL };
    test_uniform_like_f64(NULL, 0, outputs, 1);
    chelis_read_view view = chelis_tensor_read_view(outputs[0]);
    const double *data = (const double *)view.data;
    for (int64_t i = 0; i < view.count; i++) {
        uint64_t bits;
        memcpy(&bits, &data[i], sizeof(bits));
        printf(i == 0 ? "%016llx" : " %016llx", (unsigned long long)bits);
    }
    printf("\n");
    chelis_tensor_release(outputs[0]);
    return 0;
}
"#;
    let Some(output) = compile_and_run("uniform_like_f64", src, harness) else {
        panic!("emitted f64 C did not compile/run");
    };
    let bound = |value| chelis_types::scalar_from_f64("test", Prim::F32, value).unwrap();
    // The draw's key is `key_from_seed(42)`.
    let key = chelis_types::RandomKey::from_seed(
        chelis_types::scalar_from_i64("test", Prim::Int64, 42).unwrap(),
    )
    .unwrap();
    let sampled = chelis_types::PreparedUniformLike::new(Prim::F64, 4, bound(2.0), bound(5.0))
        .unwrap()
        .apply(key)
        .unwrap();
    let expected = (0..4)
        .map(|index| format!("{:016x}", sampled.scalar_at(index).as_f64_lossy().to_bits()))
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(output.trim(), expected, "emitted source:\n{src}");
}
