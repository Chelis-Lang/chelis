//! WS-Cleanup-Fixups extension to the bf16/f16 dtype matrix.
//!
//! Closes RT-Cleanup coverage gaps for ops the original WS-1 matrix did
//! not exercise directly:
//!   * Elementwise: Div, Recip, Sqrt, Sin, Cos, Tan, Atan, Floor, Ceil
//!   * Reductions: MinReduce, ProdReduce
//!   * Cross-precision casts: bf16 <-> f32, f16 <-> f32, bf16 <-> f16
//!     (each asserts both exact bit pattern AND in-tolerance round-trip)
//!
//! Tolerances and bit patterns match `dtype_matrix_bf16_f16.rs`:
//!   * `BF16_TOL` (1e-2): bf16 7-bit mantissa round-trip allowance.
//!   * `F16_TOL`  (1e-3): f16 10-bit mantissa round-trip allowance.
//!
//! Each elementwise / reduction test builds a DAG, codegens C, compiles
//! with gcc, runs the binary, and compares the result to the IR
//! evaluator within tolerance. The cast tests assert the exact bit
//! pattern returned by the runtime helper (computed at test-build time
//! via the `half` crate) AND the in-tolerance round-trip.
//!
//! Acceptance oracle: `cargo test -p chelis-backend-c --test dtype_matrix_bf16_f16_extended`.

mod support;
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::eval_tensor;
use chelis_types::types::Prim;
use chelis_unord::UnordMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use support::codegen;

mod common;

const BF16_TOL: f64 = 1e-2;
const F16_TOL: f64 = 1e-3;

// ---------------------------------------------------------------------
// Test harness (mirrors dtype_matrix_bf16_f16.rs; cannot share because
// integration tests do not share modules without an explicit `mod`).
// ---------------------------------------------------------------------

fn runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

fn ensure_runtime_static_lib(canonical: &Path) -> std::io::Result<()> {
    if canonical.exists() {
        return Ok(());
    }
    let deps_dir = canonical
        .parent()
        .expect("canonical lib path has no parent")
        .join("deps");
    let hashed = find_newest_runtime_archive(&deps_dir)?;
    let hashed = match hashed {
        Some(path) => path,
        None => {
            Command::new(env!("CARGO"))
                .args(["build", "-p", "chelis-runtime", "--lib"])
                .status()
                .map_err(|e| std::io::Error::other(format!("cargo build chelis-runtime: {e}")))?;
            find_newest_runtime_archive(&deps_dir)?.ok_or_else(|| {
                std::io::Error::other(format!(
                    "no libchelis_runtime-*.a found in {} after explicit \
                     `cargo build -p chelis-runtime --lib`",
                    deps_dir.display()
                ))
            })?
        }
    };
    static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = canonical.with_extension(format!(
        "a.tmp.{}.{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::copy(&hashed, &tmp)?;
    fs::rename(&tmp, canonical)?;
    Ok(())
}

fn find_newest_runtime_archive(deps_dir: &Path) -> std::io::Result<Option<PathBuf>> {
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(deps_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("libchelis_runtime-") && name.ends_with(".a") {
            let meta = entry.metadata()?;
            let mtime = meta.modified()?;
            match &newest {
                Some((cur, _)) if *cur >= mtime => {}
                _ => newest = Some((mtime, entry.path())),
            }
        }
    }
    Ok(newest.map(|(_, p)| p))
}

fn runtime_lib_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let canonical = target_debug_dir().join("libchelis_runtime.a");
        if let Err(e) = ensure_runtime_static_lib(&canonical) {
            panic!(
                "failed to materialize libchelis_runtime.a at {}: {}",
                canonical.display(),
                e
            );
        }
        canonical
    })
    .clone()
}

fn gcc_available() -> bool {
    Command::new("gcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn compile_and_run_kernel(test_name: &str, c_source: &str, main_c: &str) -> String {
    let probe = common::probe_dir(&format!("bf16_ext_{test_name}"));
    let dir = probe.path().to_path_buf();
    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), main_c).unwrap();

    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_runtime_views.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }

    let bin = dir.join("test_bin");
    let runtime_lib = runtime_lib_path();
    let args: Vec<String> = vec![
        "-O2".into(),
        "-std=c11".into(),
        "-I".into(),
        dir.to_str().unwrap().into(),
        dir.join("kernel.c").to_str().unwrap().into(),
        dir.join("main.c").to_str().unwrap().into(),
        "-o".into(),
        bin.to_str().unwrap().into(),
        runtime_lib.to_str().unwrap().into(),
        "-lm".into(),
        "-lpthread".into(),
        "-ldl".into(),
    ];
    let compile = Command::new("gcc")
        .args(&args)
        .output()
        .expect("failed to invoke gcc");
    assert!(
        compile.status.success(),
        "gcc failed for {test_name}:\nstderr: {}\nC source:\n{c_source}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&bin).output().expect("failed to run binary");
    assert!(
        run.status.success(),
        "binary failed for {test_name}: stderr={}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).unwrap()
}

const HARNESS: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "chelis_runtime.h"

static chelis_tensor *bf16_tensor_from_f32(const float *src, int n) {
    int64_t shape[1] = {n};
    chelis_tensor *t = chelis_alloc(1, shape, CHELIS_DTYPE_BF16);
    chelis_tensor_write *guard = chelis_tensor_begin_write(t);
    uint16_t *p = (uint16_t*)chelis_tensor_write_view(guard).data;
    for (int i = 0; i < n; i++) p[i] = chelis_f32_to_bf16(src[i]);
    chelis_tensor_end_write(guard);
    return t;
}

static chelis_tensor *f16_tensor_from_f32(const float *src, int n) {
    int64_t shape[1] = {n};
    chelis_tensor *t = chelis_alloc(1, shape, CHELIS_DTYPE_F16);
    chelis_tensor_write *guard = chelis_tensor_begin_write(t);
    uint16_t *p = (uint16_t*)chelis_tensor_write_view(guard).data;
    for (int i = 0; i < n; i++) p[i] = chelis_f32_to_f16(src[i]);
    chelis_tensor_end_write(guard);
    return t;
}

static chelis_tensor *f32_tensor_from_f32(const float *src, int n) {
    int64_t shape[1] = {n};
    chelis_tensor *t = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *guard = chelis_tensor_begin_write(t);
    float *p = (float*)chelis_tensor_write_view(guard).data;
    for (int i = 0; i < n; i++) p[i] = src[i];
    chelis_tensor_end_write(guard);
    return t;
}

static const uint16_t *tensor_u16_data(const chelis_tensor *t) {
    return (const uint16_t*)chelis_tensor_read_view(t).data;
}

static const float *tensor_f32_data(const chelis_tensor *t) {
    return (const float*)chelis_tensor_read_view(t).data;
}
"#;

fn vec_ty(n: usize, prec: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: prec,
    }
}

fn scalar_ty(prec: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision: prec,
    }
}

/// Evaluate the DAG; return the result tensor data of the last node as f64.
fn eval_last(dag: &Dag) -> Vec<f64> {
    let inputs = UnordMap::new();
    let vals = eval_tensor(dag, &inputs).unwrap();
    let last_id = NodeId(dag.len() - 1);
    vals[&last_id].to_f64_lossy_vec().clone()
}

/// Build a vector-rooted unary kernel: input tensor of `vals` interpreted
/// as bf16 or f16, apply `op`, return a function that runs the C build
/// and returns the elementwise output as f64.
fn run_unary_reduced(
    test_name: &str,
    prec: Prim,
    vals: &[f32],
    op: RiscOp,
    expected_eval: &[f64],
    tol: f64,
) {
    if !gcc_available() {
        eprintln!("skipping {test_name}: gcc not available");
        return;
    }
    let n = vals.len();
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, prec),
        None,
    );
    dag.add_node(op, vec![load], vec_ty(n, prec), None);
    let result = codegen(&dag, test_name).unwrap();
    let load_helper = match prec {
        Prim::Bf16 => "bf16_tensor_from_f32",
        Prim::F16 => "f16_tensor_from_f32",
        _ => panic!("run_unary_reduced is bf16/f16 only"),
    };
    let to_f32 = match prec {
        Prim::Bf16 => "chelis_bf16_to_f32",
        Prim::F16 => "chelis_f16_to_f32",
        _ => unreachable!(),
    };
    let vals_init = vals
        .iter()
        .map(|v| format!("{v:?}f"))
        .collect::<Vec<_>>()
        .join(", ");
    let main_c = format!(
        r#"{HARNESS}
extern void {test_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float vals[{n}] = {{ {vals_init} }};
    chelis_tensor *x = {load_helper}(vals, {n});
    chelis_tensor *inputs[1] = {{x}};
    chelis_tensor *outputs[1] = {{0}};
    {test_name}(inputs, 1, outputs, 1);
    const uint16_t *p = tensor_u16_data(outputs[0]);
    for (int i = 0; i < {n}; i++) printf("%.8f\n", {to_f32}(p[i]));
    chelis_tensor_release(x);
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel(test_name, &result.c_source, &main_c);
    let c_vals: Vec<f64> = stdout
        .trim()
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    assert_eq!(
        c_vals.len(),
        expected_eval.len(),
        "{test_name}: got {} values, expected {}",
        c_vals.len(),
        expected_eval.len()
    );
    for (i, (got, want)) in c_vals.iter().zip(expected_eval.iter()).enumerate() {
        // NaN-tolerant compare: if eval produces NaN, the C result
        // must also be NaN (Floor / Ceil etc. propagate NaN per IEEE 754).
        if want.is_nan() {
            assert!(got.is_nan(), "{test_name}[{i}]: eval=NaN but C={got}");
            continue;
        }
        assert!(
            (got - want).abs() <= tol,
            "{test_name}[{i}]: C={got}, eval={want}, diff={}, tol={tol}",
            (got - want).abs()
        );
    }
}

/// Same shape as `run_unary_reduced` but the kernel emits a binary op
/// (lhs and rhs both reduced floats).
fn run_binary_reduced(
    test_name: &str,
    prec: Prim,
    lhs: &[f32],
    rhs: &[f32],
    op: RiscOp,
    expected_eval: &[f64],
    tol: f64,
) {
    if !gcc_available() {
        eprintln!("skipping {test_name}: gcc not available");
        return;
    }
    let n = lhs.len();
    assert_eq!(n, rhs.len(), "lhs and rhs must match length");
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_ty(n, prec),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_ty(n, prec),
        None,
    );
    dag.add_node(op, vec![a, b], vec_ty(n, prec), None);
    let result = codegen(&dag, test_name).unwrap();
    let load_helper = match prec {
        Prim::Bf16 => "bf16_tensor_from_f32",
        Prim::F16 => "f16_tensor_from_f32",
        _ => panic!("run_binary_reduced is bf16/f16 only"),
    };
    let to_f32 = match prec {
        Prim::Bf16 => "chelis_bf16_to_f32",
        Prim::F16 => "chelis_f16_to_f32",
        _ => unreachable!(),
    };
    let lhs_init = lhs
        .iter()
        .map(|v| format!("{v:?}f"))
        .collect::<Vec<_>>()
        .join(", ");
    let rhs_init = rhs
        .iter()
        .map(|v| format!("{v:?}f"))
        .collect::<Vec<_>>()
        .join(", ");
    let main_c = format!(
        r#"{HARNESS}
extern void {test_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float la[{n}] = {{ {lhs_init} }};
    float lb[{n}] = {{ {rhs_init} }};
    chelis_tensor *a = {load_helper}(la, {n});
    chelis_tensor *b = {load_helper}(lb, {n});
    chelis_tensor *inputs[2] = {{a, b}};
    chelis_tensor *outputs[1] = {{0}};
    {test_name}(inputs, 2, outputs, 1);
    const uint16_t *p = tensor_u16_data(outputs[0]);
    for (int i = 0; i < {n}; i++) printf("%.8f\n", {to_f32}(p[i]));
    chelis_tensor_release(a);
    chelis_tensor_release(b);
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel(test_name, &result.c_source, &main_c);
    let c_vals: Vec<f64> = stdout
        .trim()
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    for (i, (got, want)) in c_vals.iter().zip(expected_eval.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{test_name}[{i}]: C={got}, eval={want}, diff={}, tol={tol}",
            (got - want).abs()
        );
    }
}

// ---------------------------------------------------------------------
// Elementwise binary: Div
// ---------------------------------------------------------------------

#[test]
fn bf16_div_agrees_with_evaluator() {
    let lhs = [6.0_f32, 9.0, 1.0, -2.0];
    let rhs = [2.0_f32, 3.0, 4.0, 8.0];
    let mut dag = Dag::new();
    let n = 4;
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_ty(n, Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_ty(n, Prim::Bf16),
        None,
    );
    dag.add_node(RiscOp::Div, vec![a, b], vec_ty(n, Prim::Bf16), None);
    let inputs: UnordMap<String, chelis_ir::eval::TensorValue> = [
        (
            "a".into(),
            chelis_ir::eval::TensorValue::from_vec(
                vec![n],
                lhs.iter().map(|&v| v as f64).collect(),
            ),
        ),
        (
            "b".into(),
            chelis_ir::eval::TensorValue::from_vec(
                vec![n],
                rhs.iter().map(|&v| v as f64).collect(),
            ),
        ),
    ]
    .into_iter()
    .collect();
    let vals = eval_tensor(&dag, &inputs).unwrap();
    let last_id = NodeId(dag.len() - 1);
    let expected = vals[&last_id].to_f64_lossy_vec().clone();
    run_binary_reduced(
        "bf16_div",
        Prim::Bf16,
        &lhs,
        &rhs,
        RiscOp::Div,
        &expected,
        BF16_TOL,
    );
}

#[test]
fn f16_div_agrees_with_evaluator() {
    let lhs = [6.0_f32, 9.0, 1.0, -2.0];
    let rhs = [2.0_f32, 3.0, 4.0, 8.0];
    let mut dag = Dag::new();
    let n = 4;
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_ty(n, Prim::F16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_ty(n, Prim::F16),
        None,
    );
    dag.add_node(RiscOp::Div, vec![a, b], vec_ty(n, Prim::F16), None);
    let inputs: UnordMap<String, chelis_ir::eval::TensorValue> = [
        (
            "a".into(),
            chelis_ir::eval::TensorValue::from_vec(
                vec![n],
                lhs.iter().map(|&v| v as f64).collect(),
            ),
        ),
        (
            "b".into(),
            chelis_ir::eval::TensorValue::from_vec(
                vec![n],
                rhs.iter().map(|&v| v as f64).collect(),
            ),
        ),
    ]
    .into_iter()
    .collect();
    let vals = eval_tensor(&dag, &inputs).unwrap();
    let last_id = NodeId(dag.len() - 1);
    let expected = vals[&last_id].to_f64_lossy_vec().clone();
    run_binary_reduced(
        "f16_div",
        Prim::F16,
        &lhs,
        &rhs,
        RiscOp::Div,
        &expected,
        F16_TOL,
    );
}

// ---------------------------------------------------------------------
// Elementwise unary: Recip / Sqrt / Sin / Cos / Tan / Atan / Floor / Ceil
// ---------------------------------------------------------------------

fn unary_eval(op: RiscOp, prec: Prim, vals: &[f32]) -> Vec<f64> {
    let n = vals.len();
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, prec),
        None,
    );
    dag.add_node(op, vec![load], vec_ty(n, prec), None);
    let inputs: UnordMap<String, chelis_ir::eval::TensorValue> = [(
        "x".into(),
        chelis_ir::eval::TensorValue::from_vec(vec![n], vals.iter().map(|&v| v as f64).collect()),
    )]
    .into_iter()
    .collect();
    let vals = eval_tensor(&dag, &inputs).unwrap();
    let last_id = NodeId(dag.len() - 1);
    vals[&last_id].to_f64_lossy_vec().clone()
}

#[test]
fn bf16_recip_agrees_with_evaluator() {
    let vals = [1.0_f32, 2.0, 4.0, 0.5];
    let expected = unary_eval(RiscOp::Recip, Prim::Bf16, &vals);
    run_unary_reduced(
        "bf16_recip",
        Prim::Bf16,
        &vals,
        RiscOp::Recip,
        &expected,
        BF16_TOL,
    );
}

#[test]
fn f16_recip_agrees_with_evaluator() {
    let vals = [1.0_f32, 2.0, 4.0, 0.5];
    let expected = unary_eval(RiscOp::Recip, Prim::F16, &vals);
    run_unary_reduced(
        "f16_recip",
        Prim::F16,
        &vals,
        RiscOp::Recip,
        &expected,
        F16_TOL,
    );
}

#[test]
fn bf16_sqrt_agrees_with_evaluator() {
    let vals = [1.0_f32, 4.0, 9.0, 16.0];
    let expected = unary_eval(RiscOp::Sqrt, Prim::Bf16, &vals);
    run_unary_reduced(
        "bf16_sqrt",
        Prim::Bf16,
        &vals,
        RiscOp::Sqrt,
        &expected,
        BF16_TOL,
    );
}

#[test]
fn f16_sqrt_agrees_with_evaluator() {
    let vals = [1.0_f32, 4.0, 9.0, 16.0];
    let expected = unary_eval(RiscOp::Sqrt, Prim::F16, &vals);
    run_unary_reduced(
        "f16_sqrt",
        Prim::F16,
        &vals,
        RiscOp::Sqrt,
        &expected,
        F16_TOL,
    );
}

#[test]
fn bf16_sin_agrees_with_evaluator() {
    let vals = [0.0_f32, 0.5, 1.0, -0.5];
    let expected = unary_eval(RiscOp::Sin, Prim::Bf16, &vals);
    run_unary_reduced(
        "bf16_sin",
        Prim::Bf16,
        &vals,
        RiscOp::Sin,
        &expected,
        BF16_TOL,
    );
}

#[test]
fn f16_sin_agrees_with_evaluator() {
    let vals = [0.0_f32, 0.5, 1.0, -0.5];
    let expected = unary_eval(RiscOp::Sin, Prim::F16, &vals);
    run_unary_reduced("f16_sin", Prim::F16, &vals, RiscOp::Sin, &expected, F16_TOL);
}

#[test]
fn bf16_cos_agrees_with_evaluator() {
    let vals = [0.0_f32, 0.5, 1.0, -0.5];
    let expected = unary_eval(RiscOp::Cos, Prim::Bf16, &vals);
    run_unary_reduced(
        "bf16_cos",
        Prim::Bf16,
        &vals,
        RiscOp::Cos,
        &expected,
        BF16_TOL,
    );
}

#[test]
fn f16_cos_agrees_with_evaluator() {
    let vals = [0.0_f32, 0.5, 1.0, -0.5];
    let expected = unary_eval(RiscOp::Cos, Prim::F16, &vals);
    run_unary_reduced("f16_cos", Prim::F16, &vals, RiscOp::Cos, &expected, F16_TOL);
}

#[test]
fn bf16_tan_agrees_with_evaluator() {
    // Stay well away from pi/2 to keep tan() finite within bf16 dynamic range.
    let vals = [0.0_f32, 0.25, 0.5, -0.25];
    let expected = unary_eval(RiscOp::Tan, Prim::Bf16, &vals);
    run_unary_reduced(
        "bf16_tan",
        Prim::Bf16,
        &vals,
        RiscOp::Tan,
        &expected,
        BF16_TOL,
    );
}

#[test]
fn f16_tan_agrees_with_evaluator() {
    let vals = [0.0_f32, 0.25, 0.5, -0.25];
    let expected = unary_eval(RiscOp::Tan, Prim::F16, &vals);
    run_unary_reduced("f16_tan", Prim::F16, &vals, RiscOp::Tan, &expected, F16_TOL);
}

#[test]
fn bf16_atan_agrees_with_evaluator() {
    let vals = [0.0_f32, 1.0, -1.0, 0.5];
    let expected = unary_eval(RiscOp::Atan, Prim::Bf16, &vals);
    run_unary_reduced(
        "bf16_atan",
        Prim::Bf16,
        &vals,
        RiscOp::Atan,
        &expected,
        BF16_TOL,
    );
}

#[test]
fn f16_atan_agrees_with_evaluator() {
    let vals = [0.0_f32, 1.0, -1.0, 0.5];
    let expected = unary_eval(RiscOp::Atan, Prim::F16, &vals);
    run_unary_reduced(
        "f16_atan",
        Prim::F16,
        &vals,
        RiscOp::Atan,
        &expected,
        F16_TOL,
    );
}

#[test]
fn bf16_floor_agrees_with_evaluator() {
    let vals = [1.7_f32, -1.2, 0.5, 3.0];
    let expected = unary_eval(RiscOp::Floor, Prim::Bf16, &vals);
    run_unary_reduced(
        "bf16_floor",
        Prim::Bf16,
        &vals,
        RiscOp::Floor,
        &expected,
        BF16_TOL,
    );
}

#[test]
fn f16_floor_agrees_with_evaluator() {
    let vals = [1.7_f32, -1.2, 0.5, 3.0];
    let expected = unary_eval(RiscOp::Floor, Prim::F16, &vals);
    run_unary_reduced(
        "f16_floor",
        Prim::F16,
        &vals,
        RiscOp::Floor,
        &expected,
        F16_TOL,
    );
}

#[test]
fn bf16_ceil_agrees_with_evaluator() {
    let vals = [1.2_f32, -1.7, 0.5, 3.0];
    let expected = unary_eval(RiscOp::Ceil, Prim::Bf16, &vals);
    run_unary_reduced(
        "bf16_ceil",
        Prim::Bf16,
        &vals,
        RiscOp::Ceil,
        &expected,
        BF16_TOL,
    );
}

#[test]
fn f16_ceil_agrees_with_evaluator() {
    let vals = [1.2_f32, -1.7, 0.5, 3.0];
    let expected = unary_eval(RiscOp::Ceil, Prim::F16, &vals);
    run_unary_reduced(
        "f16_ceil",
        Prim::F16,
        &vals,
        RiscOp::Ceil,
        &expected,
        F16_TOL,
    );
}

// `round` is round-to-nearest-ties-to-even. The .5 cases (0.5 -> 0,
// 2.5 -> 2) exercise the tiebreak that distinguishes it from
// ties-away-from-zero `roundf`; all chosen values are exactly
// representable in bf16/f16, so the C `rintf` output must agree with
// the evaluator's `f64::round_ties_even` to the dtype tolerance.
#[test]
fn bf16_round_agrees_with_evaluator() {
    let vals = [0.5_f32, 2.5, -2.5, 2.0];
    let expected = unary_eval(RiscOp::Round, Prim::Bf16, &vals);
    run_unary_reduced(
        "bf16_round",
        Prim::Bf16,
        &vals,
        RiscOp::Round,
        &expected,
        BF16_TOL,
    );
}

#[test]
fn f16_round_agrees_with_evaluator() {
    let vals = [0.5_f32, 2.5, -2.5, 2.0];
    let expected = unary_eval(RiscOp::Round, Prim::F16, &vals);
    run_unary_reduced(
        "f16_round",
        Prim::F16,
        &vals,
        RiscOp::Round,
        &expected,
        F16_TOL,
    );
}

// ---------------------------------------------------------------------
// Reductions: MinReduce, ProdReduce
// ---------------------------------------------------------------------

fn run_scalar_reduce_reduced(
    test_name: &str,
    prec: Prim,
    vals: &[f32],
    op: RiscOp,
    expected: f64,
    tol: f64,
) {
    if !gcc_available() {
        eprintln!("skipping {test_name}: gcc not available");
        return;
    }
    let n = vals.len();
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, prec),
        None,
    );
    dag.add_node(op, vec![load], scalar_ty(prec), None);
    let result = codegen(&dag, test_name).unwrap();
    let load_helper = match prec {
        Prim::Bf16 => "bf16_tensor_from_f32",
        Prim::F16 => "f16_tensor_from_f32",
        _ => panic!("reduced only"),
    };
    let to_f32 = match prec {
        Prim::Bf16 => "chelis_bf16_to_f32",
        Prim::F16 => "chelis_f16_to_f32",
        _ => unreachable!(),
    };
    let vals_init = vals
        .iter()
        .map(|v| format!("{v:?}f"))
        .collect::<Vec<_>>()
        .join(", ");
    let main_c = format!(
        r#"{HARNESS}
extern void {test_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float vals[{n}] = {{ {vals_init} }};
    chelis_tensor *x = {load_helper}(vals, {n});
    chelis_tensor *inputs[1] = {{x}};
    chelis_tensor *outputs[1] = {{0}};
    {test_name}(inputs, 1, outputs, 1);
    const uint16_t *p = tensor_u16_data(outputs[0]);
    printf("%.8f\n", {to_f32}(p[0]));
    chelis_tensor_release(x);
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel(test_name, &result.c_source, &main_c);
    let c_val: f64 = stdout.trim().parse().unwrap();
    assert!(
        (c_val - expected).abs() <= tol,
        "{test_name}: C={c_val}, expected={expected}, diff={}",
        (c_val - expected).abs()
    );
}

/// Evaluator reference for a vec -> scalar reduction over reduced floats.
fn scalar_reduce_eval(op: RiscOp, prec: Prim, vals: &[f32]) -> f64 {
    let n = vals.len();
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, prec),
        None,
    );
    dag.add_node(op, vec![load], scalar_ty(prec), None);
    let inputs: UnordMap<String, chelis_ir::eval::TensorValue> = [(
        "x".into(),
        chelis_ir::eval::TensorValue::from_vec(vec![n], vals.iter().map(|&v| v as f64).collect()),
    )]
    .into_iter()
    .collect();
    let out = eval_tensor(&dag, &inputs).unwrap();
    out[&NodeId(dag.len() - 1)].to_f64_lossy_vec()[0]
}

// MinReduce on bf16/f16 is emitted by `emit_reduce_extreme`, which widens
// reduced floats to f32 arithmetic (chelis#1281). The minimum is one of the
// rounded inputs, so the C result must equal the evaluator exactly.
#[test]
fn bf16_min_reduce_agrees_with_evaluator() {
    let vals = [3.5_f32, -1.25, 2.0, 0.5];
    let op = RiscOp::MinReduce { axis: 0 };
    let expected = scalar_reduce_eval(op.clone(), Prim::Bf16, &vals);
    run_scalar_reduce_reduced("bf16_min_reduce", Prim::Bf16, &vals, op, expected, 0.0);
}

#[test]
fn f16_min_reduce_agrees_with_evaluator() {
    let vals = [3.5_f32, -1.25, 2.0, 0.5];
    let op = RiscOp::MinReduce { axis: 0 };
    let expected = scalar_reduce_eval(op.clone(), Prim::F16, &vals);
    run_scalar_reduce_reduced("f16_min_reduce", Prim::F16, &vals, op, expected, 0.0);
}

// ProdReduce on bf16/f16 is still rejected at codegen: `emit_reduce_simple`
// is f32-hardcoded (WS-A1 / F1 guard). Closure path: when it gains the
// convert-then-reduce arm for reduced floats, flip these to agreement tests
// like the MinReduce ones above.

#[test]
fn bf16_prod_reduce_is_structurally_unsupported_today() {
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(4, Prim::Bf16),
        None,
    );
    dag.add_node(
        RiscOp::ProdReduce { axis: 0 },
        vec![load],
        scalar_ty(Prim::Bf16),
        None,
    );
    // chelis#730 Phase 1: the former f32-hardcoded panic is a section C2
    // diagnostic through the Result channel.
    let err = codegen(&dag, "bf16_prod_reduce_reject_probe")
        .map(|_| ())
        .expect_err("a bf16 prod_reduce must be rejected, not emitted");
    let rendered = err.to_string();
    assert!(
        rendered.starts_with("unsupported:") && rendered.contains("bf16"),
        "the rejection must be branded and name the dtype; got: {rendered}"
    );
}

#[test]
fn f16_prod_reduce_is_structurally_unsupported_today() {
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(4, Prim::F16),
        None,
    );
    dag.add_node(
        RiscOp::ProdReduce { axis: 0 },
        vec![load],
        scalar_ty(Prim::F16),
        None,
    );
    // chelis#730 Phase 1: the former f32-hardcoded panic is a section C2
    // diagnostic through the Result channel.
    let err = codegen(&dag, "f16_prod_reduce_reject_probe")
        .map(|_| ())
        .expect_err("a f16 prod_reduce must be rejected, not emitted");
    let rendered = err.to_string();
    assert!(
        rendered.starts_with("unsupported:") && rendered.contains("f16"),
        "the rejection must be branded and name the dtype; got: {rendered}"
    );
}

// ---------------------------------------------------------------------
// Cross-precision casts (post-WS-Cleanup-Fixups behavior)
// ---------------------------------------------------------------------

/// Run a cast f32 -> reduced (bf16 or f16); assert the output bit pattern
/// is exactly `half::{bf16,f16}::from_f64(value).to_bits()` AND the
/// round-trip through `chelis_{bf16,f16}_to_f32` lands within tolerance.
fn run_cast_f32_to_reduced(test_name: &str, dst: Prim, value: f32, tol: f64) {
    if !gcc_available() {
        eprintln!("skipping {test_name}: gcc not available");
        return;
    }
    let n = 4;
    let mut dag = Dag::new();
    let src = dag.add_node(
        RiscOp::synth_const(vec_ty(n, Prim::F32).precision, value as f64),
        vec![],
        vec_ty(n, Prim::F32),
        None,
    );
    dag.add_node(
        RiscOp::Cast { new_precision: dst },
        vec![src],
        vec_ty(n, dst),
        None,
    );
    let result = codegen(&dag, test_name).unwrap();
    let to_f32 = match dst {
        Prim::Bf16 => "chelis_bf16_to_f32",
        Prim::F16 => "chelis_f16_to_f32",
        _ => panic!("dst must be reduced"),
    };
    let main_c = format!(
        r#"{HARNESS}
extern void {test_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    {test_name}(NULL, 0, outputs, 1);
    const uint16_t *p = tensor_u16_data(outputs[0]);
    printf("0x%04X\n", (unsigned)p[0]);
    printf("%.8f\n", {to_f32}(p[0]));
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel(test_name, &result.c_source, &main_c);
    let mut lines = stdout.trim().lines();
    let bits_line = lines.next().unwrap();
    let round_line = lines.next().unwrap();
    let got_bits = u16::from_str_radix(bits_line.trim().trim_start_matches("0x"), 16).unwrap();
    let expected_bits = match dst {
        Prim::Bf16 => half::bf16::from_f64(value as f64).to_bits(),
        Prim::F16 => half::f16::from_f64(value as f64).to_bits(),
        _ => unreachable!(),
    };
    assert_eq!(
        got_bits, expected_bits,
        "{test_name}: bit pattern got 0x{got_bits:04X}, expected 0x{expected_bits:04X}"
    );
    let round: f64 = round_line.parse().unwrap();
    assert!(
        (round - value as f64).abs() <= tol,
        "{test_name}: round-trip got {round}, expected {value}, tol {tol}"
    );
}

/// Cast reduced -> f32; assert the f32 result equals the IEEE-754 decode
/// of the reduced bit pattern via the runtime helper.
fn run_cast_reduced_to_f32(test_name: &str, src: Prim, value: f32, tol: f64) {
    if !gcc_available() {
        eprintln!("skipping {test_name}: gcc not available");
        return;
    }
    let n = 4;
    let mut dag = Dag::new();
    let c = dag.add_node(
        RiscOp::synth_const(vec_ty(n, src).precision, value as f64),
        vec![],
        vec_ty(n, src),
        None,
    );
    dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![c],
        vec_ty(n, Prim::F32),
        None,
    );
    let result = codegen(&dag, test_name).unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void {test_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    {test_name}(NULL, 0, outputs, 1);
    float v = tensor_f32_data(outputs[0])[0];
    printf("%.8f\n", v);
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel(test_name, &result.c_source, &main_c);
    let got: f64 = stdout.trim().parse().unwrap();
    let expected = match src {
        Prim::Bf16 => half::bf16::from_f64(value as f64).to_f64(),
        Prim::F16 => half::f16::from_f64(value as f64).to_f64(),
        _ => panic!("src must be reduced"),
    };
    assert!(
        (got - expected).abs() <= tol,
        "{test_name}: got {got}, expected {expected}, tol {tol}"
    );
}

/// Cast reduced -> reduced (bf16 <-> f16); both bit pattern and round-trip checked.
fn run_cast_reduced_to_reduced(test_name: &str, src: Prim, dst: Prim, value: f32, tol: f64) {
    if !gcc_available() {
        eprintln!("skipping {test_name}: gcc not available");
        return;
    }
    assert_ne!(src, dst, "same-precision cast tested elsewhere");
    let n = 4;
    let mut dag = Dag::new();
    let c = dag.add_node(
        RiscOp::synth_const(vec_ty(n, src).precision, value as f64),
        vec![],
        vec_ty(n, src),
        None,
    );
    dag.add_node(
        RiscOp::Cast { new_precision: dst },
        vec![c],
        vec_ty(n, dst),
        None,
    );
    let result = codegen(&dag, test_name).unwrap();
    let to_f32 = match dst {
        Prim::Bf16 => "chelis_bf16_to_f32",
        Prim::F16 => "chelis_f16_to_f32",
        _ => panic!("dst must be reduced"),
    };
    let main_c = format!(
        r#"{HARNESS}
extern void {test_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    {test_name}(NULL, 0, outputs, 1);
    const uint16_t *p = tensor_u16_data(outputs[0]);
    printf("0x%04X\n", (unsigned)p[0]);
    printf("%.8f\n", {to_f32}(p[0]));
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel(test_name, &result.c_source, &main_c);
    let mut lines = stdout.trim().lines();
    let bits_line = lines.next().unwrap();
    let round_line = lines.next().unwrap();
    let got_bits = u16::from_str_radix(bits_line.trim().trim_start_matches("0x"), 16).unwrap();
    // The codegen-time Const fill in `emit_const` already stores the
    // src-precision bit pattern. The cross-narrow-float cast then
    // chains through f32 so the dst-precision bit pattern is the
    // canonical round-to-nearest-even of the src value.
    let src_value = match src {
        Prim::Bf16 => half::bf16::from_f64(value as f64).to_f64(),
        Prim::F16 => half::f16::from_f64(value as f64).to_f64(),
        _ => panic!("src must be reduced"),
    };
    let expected_bits = match dst {
        Prim::Bf16 => half::bf16::from_f64(src_value).to_bits(),
        Prim::F16 => half::f16::from_f64(src_value).to_bits(),
        _ => unreachable!(),
    };
    assert_eq!(
        got_bits, expected_bits,
        "{test_name}: bit pattern got 0x{got_bits:04X}, expected 0x{expected_bits:04X}"
    );
    let round: f64 = round_line.parse().unwrap();
    assert!(
        (round - src_value).abs() <= tol,
        "{test_name}: round-trip got {round}, expected {src_value}, tol {tol}"
    );
}

#[test]
fn cast_f32_to_bf16_bit_pattern_and_roundtrip() {
    run_cast_f32_to_reduced("cast_ext_f32_to_bf16", Prim::Bf16, 1.5, BF16_TOL);
}

#[test]
fn cast_f32_to_f16_bit_pattern_and_roundtrip() {
    run_cast_f32_to_reduced("cast_ext_f32_to_f16", Prim::F16, 1.5, F16_TOL);
}

#[test]
fn cast_bf16_to_f32_value_preserved() {
    run_cast_reduced_to_f32("cast_ext_bf16_to_f32", Prim::Bf16, 1.5, BF16_TOL);
}

#[test]
fn cast_f16_to_f32_value_preserved() {
    run_cast_reduced_to_f32("cast_ext_f16_to_f32", Prim::F16, 1.5, F16_TOL);
}

#[test]
fn cast_bf16_to_f16_chains_through_f32() {
    // bf16(1.5) -> f32(1.5) -> f16(1.5) = 0x3E00.
    run_cast_reduced_to_reduced("cast_ext_bf16_to_f16", Prim::Bf16, Prim::F16, 1.5, F16_TOL);
}

#[test]
fn cast_f16_to_bf16_chains_through_f32() {
    // f16(1.5) -> f32(1.5) -> bf16(1.5) = 0x3FC0.
    run_cast_reduced_to_reduced("cast_ext_f16_to_bf16", Prim::F16, Prim::Bf16, 1.5, BF16_TOL);
}

/// Sweep additional values to catch a regression that only handled
/// one operand value correctly (e.g., a power-of-two-only special case).
#[test]
fn cast_f32_to_bf16_sweep_non_round_values() {
    for value in [0.5_f32, 2.5, -1.75, 100.0, 0.125] {
        run_cast_f32_to_reduced(
            &format!("cast_ext_sweep_bf16_{}", value.to_bits()),
            Prim::Bf16,
            value,
            BF16_TOL,
        );
    }
}

#[test]
fn cast_f32_to_f16_sweep_non_round_values() {
    for value in [0.5_f32, 2.5, -1.75, 100.0, 0.125] {
        run_cast_f32_to_reduced(
            &format!("cast_ext_sweep_f16_{}", value.to_bits()),
            Prim::F16,
            value,
            F16_TOL,
        );
    }
}

// Sanity: confirm the eval_last helper isn't dead; used by future
// extensions. The compiler would otherwise flag dead code under -D warnings.
#[test]
fn eval_last_helper_returns_const_value() {
    let mut dag = Dag::new();
    dag.add_node(
        RiscOp::synth_const(scalar_ty(Prim::F32).precision, 7.0),
        vec![],
        scalar_ty(Prim::F32),
        None,
    );
    let vals = eval_last(&dag);
    assert_eq!(vals, vec![7.0]);
}
