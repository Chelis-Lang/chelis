//! WS-1 (dtype + Metal cleanup cycle): bf16 and f16 acceptance for the
//! C backend. Each test builds a DAG, codegens C, compiles with gcc,
//! runs the binary, and compares the result to the IR evaluator (or
//! to a hand-pinned bit pattern, for Const-rooted tests).
//!
//! Tolerances are pinned:
//!   * `BF16_TOL` (1e-2): bf16 has 7 mantissa bits; round-trip error
//!     on a single op stays below 1e-2 for the values these tests use.
//!   * `F16_TOL` (1e-3): f16 has 10 mantissa bits; tighter tolerance.
//!
//! Bit patterns are pinned where the test asserts exact storage:
//!   * bf16(1.5)  = 0x3FC0   * f16(1.5)  = 0x3E00
//!   * bf16(2.5)  = 0x4020   * f16(2.5)  = 0x4100
//!   * bf16(-1.0) = 0xBF80   * f16(-1.0) = 0xBC00
//!   * bf16(0.0)  = 0x0000   * f16(0.0)  = 0x0000
//!
//! Acceptance oracle: `cargo test -p chelis-backend-c --test dtype_matrix_bf16_f16`.

use chelis_backend_c::codegen;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::eval_tensor;
use chelis_types::types::Prim;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const BF16_TOL: f64 = 1e-2;
const F16_TOL: f64 = 1e-3;

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
    // Use a PID-suffixed tmp filename so concurrent test binaries (this
    // file and exec_compile.rs both call into this helper, and nextest
    // runs them in parallel) do not race on a shared tmp path. Each
    // process writes its own tmp and renames into the shared canonical
    // location; last writer wins, but the content is identical so the
    // race is harmless. Without the PID, two processes that interleave
    // `fs::copy` and `fs::rename` produce an ENOENT on the second
    // rename because the first rename moved the shared tmp away.
    static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = canonical.with_extension(format!(
        "a.tmp.{}.{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::copy(&hashed, &tmp)?;
    // The rename can still race with another process renaming its own
    // unique tmp into the same canonical path. On POSIX, rename onto an
    // existing file is atomic, so this is fine. If a peer beat us to
    // it, treat NotFound from a follow-up cleanup as benign.
    match fs::rename(&tmp, canonical) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && canonical.exists() => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn find_newest_runtime_archive(deps_dir: &Path) -> std::io::Result<Option<PathBuf>> {
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = match fs::read_dir(deps_dir) {
        Ok(it) => it,
        // Truly cold target dirs may not have `deps/` yet; let the
        // caller fall through to the explicit `cargo build` fallback.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    for entry in entries {
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

fn cblas_available() -> bool {
    // Probe by linking a trivial program that references `cblas_sgemm`.
    // If the OS does not ship libcblas the matmul tests early-return
    // gracefully (the agreement tests still cover this via the e2e
    // suite under a unified cblas guard).
    let dir = std::env::temp_dir().join("chelis_bf16_cblas_probe");
    let _ = fs::create_dir_all(&dir);
    let probe = dir.join("probe.c");
    fs::write(
        &probe,
        r#"
extern void cblas_sgemm();
int main(void) { (void)cblas_sgemm; return 0; }
"#,
    )
    .unwrap();
    let out = dir.join("probe_bin");
    Command::new("gcc")
        .args([
            probe.to_str().unwrap(),
            "-lcblas",
            "-o",
            out.to_str().unwrap(),
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build a C source + main harness, gcc-compile, run, return stdout.
/// Panics on compile or runtime failure with the generated C attached.
fn compile_and_run_kernel(
    test_name: &str,
    c_source: &str,
    main_c: &str,
    needs_cblas: bool,
) -> String {
    let dir = std::env::temp_dir().join(format!("chelis_bf16_{test_name}"));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), main_c).unwrap();

    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
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
    let mut args: Vec<String> = vec![
        "-O2".into(),
        "-std=c11".into(),
        "-I".into(),
        dir.to_str().unwrap().into(),
        dir.join("kernel.c").to_str().unwrap().into(),
        dir.join("main.c").to_str().unwrap().into(),
        "-o".into(),
        bin.to_str().unwrap().into(),
        runtime_lib.to_str().unwrap().into(),
    ];
    if needs_cblas {
        args.push("-lcblas".into());
    }
    args.push("-lm".into());
    args.push("-lpthread".into());
    args.push("-ldl".into());
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

/// Harness header shared across tests. Provides bf16/f16 helpers to
/// build operand tensors and read 16-bit elements back from the result.
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
    uint16_t *p = (uint16_t*)t->data;
    for (int i = 0; i < n; i++) p[i] = chelis_f32_to_bf16(src[i]);
    return t;
}

static chelis_tensor *f16_tensor_from_f32(const float *src, int n) {
    int64_t shape[1] = {n};
    chelis_tensor *t = chelis_alloc(1, shape, CHELIS_DTYPE_F16);
    uint16_t *p = (uint16_t*)t->data;
    for (int i = 0; i < n; i++) p[i] = chelis_f32_to_f16(src[i]);
    return t;
}

static chelis_tensor *bf16_matrix_from_f32(const float *src, int rows, int cols) {
    int64_t shape[2] = {rows, cols};
    chelis_tensor *t = chelis_alloc(2, shape, CHELIS_DTYPE_BF16);
    uint16_t *p = (uint16_t*)t->data;
    for (int i = 0; i < rows * cols; i++) p[i] = chelis_f32_to_bf16(src[i]);
    return t;
}

static chelis_tensor *f16_matrix_from_f32(const float *src, int rows, int cols) {
    int64_t shape[2] = {rows, cols};
    chelis_tensor *t = chelis_alloc(2, shape, CHELIS_DTYPE_F16);
    uint16_t *p = (uint16_t*)t->data;
    for (int i = 0; i < rows * cols; i++) p[i] = chelis_f32_to_f16(src[i]);
    return t;
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

fn mat_ty(rows: usize, cols: usize, prec: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: prec,
    }
}

// ---------------------------------------------------------------------
// Const-rooted bit-pattern tests
// ---------------------------------------------------------------------

#[test]
fn bf16_const_fill_produces_exact_bit_pattern() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // Each (value, expected bf16 bit pattern) pair the plan pins for
    // the Const-fill emit path.
    let cases: &[(f32, u16)] = &[
        (1.5_f32, 0x3FC0),
        (2.5_f32, 0x4020),
        (-1.0_f32, 0xBF80),
        (0.0_f32, 0x0000),
    ];
    for &(value, expected) in cases {
        let mut dag = Dag::new();
        let n = 4;
        dag.add_node(
            RiscOp::synth_const(vec_ty(n, Prim::Bf16).precision, value as f64),
            vec![],
            vec_ty(n, Prim::Bf16),
            None,
        );
        let result = codegen(&dag, "bf16_const_bits").unwrap();
        let main_c = format!(
            r#"{HARNESS}
extern void bf16_const_bits(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    bf16_const_bits(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    for (int i = 0; i < {n}; i++) printf("0x%04X\n", (unsigned)p[i]);
    chelis_free(outputs[0]);
    return 0;
}}
"#
        );
        let stdout = compile_and_run_kernel(
            &format!("bf16_const_{:08x}", value.to_bits()),
            &result.c_source,
            &main_c,
            false,
        );
        for line in stdout.lines() {
            let got = u16::from_str_radix(line.trim().trim_start_matches("0x"), 16).unwrap();
            assert_eq!(
                got, expected,
                "bf16({value}) expected bit pattern 0x{expected:04X}, got 0x{got:04X}"
            );
        }
    }
}

#[test]
fn f16_const_fill_produces_exact_bit_pattern() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let cases: &[(f32, u16)] = &[
        (1.5_f32, 0x3E00),
        (2.5_f32, 0x4100),
        (-1.0_f32, 0xBC00),
        (0.0_f32, 0x0000),
    ];
    for &(value, expected) in cases {
        let mut dag = Dag::new();
        let n = 4;
        dag.add_node(
            RiscOp::synth_const(vec_ty(n, Prim::F16).precision, value as f64),
            vec![],
            vec_ty(n, Prim::F16),
            None,
        );
        let result = codegen(&dag, "f16_const_bits").unwrap();
        let main_c = format!(
            r#"{HARNESS}
extern void f16_const_bits(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    f16_const_bits(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    for (int i = 0; i < {n}; i++) printf("0x%04X\n", (unsigned)p[i]);
    chelis_free(outputs[0]);
    return 0;
}}
"#
        );
        let stdout = compile_and_run_kernel(
            &format!("f16_const_{:08x}", value.to_bits()),
            &result.c_source,
            &main_c,
            false,
        );
        for line in stdout.lines() {
            let got = u16::from_str_radix(line.trim().trim_start_matches("0x"), 16).unwrap();
            assert_eq!(
                got, expected,
                "f16({value}) expected bit pattern 0x{expected:04X}, got 0x{got:04X}"
            );
        }
    }
}

// ---------------------------------------------------------------------
// Evaluator agreement: add, mul, reduce, matmul
// ---------------------------------------------------------------------

/// Read the last node's evaluator scalar result (only valid when the
/// trailing node has a scalar `data` slot, which holds true for the
/// reduction-rooted tests below).
fn eval_scalar(dag: &Dag) -> f64 {
    let inputs = HashMap::new();
    let vals = eval_tensor(dag, &inputs).unwrap();
    let last_id = chelis_ir::dag::NodeId(dag.len() - 1);
    vals[&last_id].element_f64_lossy(0)
}

#[test]
fn bf16_add_agrees_with_evaluator() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_ty(Prim::Bf16).precision, 1.5),
        vec![],
        scalar_ty(Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_ty(Prim::Bf16).precision, 2.5),
        vec![],
        scalar_ty(Prim::Bf16),
        None,
    );
    dag.add_node(RiscOp::Add, vec![a, b], scalar_ty(Prim::Bf16), None);
    let result = codegen(&dag, "bf16_add").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void bf16_add(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    bf16_add(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("%.8f\n", chelis_bf16_to_f32(p[0]));
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("bf16_add", &result.c_source, &main_c, false);
    let c_val: f64 = stdout.trim().parse().unwrap();
    let eval_val = eval_scalar(&dag);
    assert!(
        (c_val - eval_val).abs() <= BF16_TOL,
        "bf16 add: C={c_val}, eval={eval_val}, diff={}",
        (c_val - eval_val).abs()
    );
}

#[test]
fn f16_add_agrees_with_evaluator() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_ty(Prim::F16).precision, 1.5),
        vec![],
        scalar_ty(Prim::F16),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_ty(Prim::F16).precision, 2.5),
        vec![],
        scalar_ty(Prim::F16),
        None,
    );
    dag.add_node(RiscOp::Add, vec![a, b], scalar_ty(Prim::F16), None);
    let result = codegen(&dag, "f16_add").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void f16_add(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    f16_add(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("%.8f\n", chelis_f16_to_f32(p[0]));
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("f16_add", &result.c_source, &main_c, false);
    let c_val: f64 = stdout.trim().parse().unwrap();
    let eval_val = eval_scalar(&dag);
    assert!(
        (c_val - eval_val).abs() <= F16_TOL,
        "f16 add: C={c_val}, eval={eval_val}, diff={}",
        (c_val - eval_val).abs()
    );
}

#[test]
fn bf16_mul_agrees_with_evaluator() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_ty(Prim::Bf16).precision, 3.0),
        vec![],
        scalar_ty(Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_ty(Prim::Bf16).precision, 2.0),
        vec![],
        scalar_ty(Prim::Bf16),
        None,
    );
    dag.add_node(RiscOp::Mul, vec![a, b], scalar_ty(Prim::Bf16), None);
    let result = codegen(&dag, "bf16_mul").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void bf16_mul(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    bf16_mul(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("%.8f\n", chelis_bf16_to_f32(p[0]));
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("bf16_mul", &result.c_source, &main_c, false);
    let c_val: f64 = stdout.trim().parse().unwrap();
    let eval_val = eval_scalar(&dag);
    assert!(
        (c_val - eval_val).abs() <= BF16_TOL,
        "bf16 mul: C={c_val}, eval={eval_val}"
    );
}

#[test]
fn f16_mul_agrees_with_evaluator() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_ty(Prim::F16).precision, 3.0),
        vec![],
        scalar_ty(Prim::F16),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_ty(Prim::F16).precision, 2.0),
        vec![],
        scalar_ty(Prim::F16),
        None,
    );
    dag.add_node(RiscOp::Mul, vec![a, b], scalar_ty(Prim::F16), None);
    let result = codegen(&dag, "f16_mul").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void f16_mul(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    f16_mul(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("%.8f\n", chelis_f16_to_f32(p[0]));
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("f16_mul", &result.c_source, &main_c, false);
    let c_val: f64 = stdout.trim().parse().unwrap();
    let eval_val = eval_scalar(&dag);
    assert!(
        (c_val - eval_val).abs() <= F16_TOL,
        "f16 mul: C={c_val}, eval={eval_val}"
    );
}

/// §5.7.1 enforcement: 1024 elements of bf16(0.01) sum to 10.24 in
/// the spec's f32-accumulator path, but a naive bf16-direct accumulator
/// loses precision because each partial sum truncates back to 7
/// mantissa bits. This test pins both:
///   * the f32-accumulator path keeps the result within `BF16_TOL` of
///     10.24
///   * a hand-rolled naive bf16-direct accumulator diverges by more
///     than 0.1, confirming the f32 accumulator is doing real work
///     (not coincidentally producing the right answer).
#[test]
fn bf16_reduce_sum_uses_f32_accumulator_per_spec_5_7_1() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let n = 1024;
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, Prim::Bf16),
        None,
    );
    let sum_op = RiscOp::sum_default(0, Prim::Bf16).expect("sum constructs");
    dag.add_node(sum_op, vec![load], scalar_ty(Prim::F32), None);
    let result = codegen(&dag, "bf16_sum_1024").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void bf16_sum_1024(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float vals[{n}];
    for (int i = 0; i < {n}; i++) vals[i] = 0.01f;
    chelis_tensor *x = bf16_tensor_from_f32(vals, {n});
    chelis_tensor *inputs[1] = {{x}};
    chelis_tensor *outputs[1] = {{0}};
    bf16_sum_1024(inputs, 1, outputs, 1);
    float result = ((float*)outputs[0]->data)[0];
    printf("%.6f\n", result);
    // Hand-rolled naive bf16-direct accumulator: round to bf16 after
    // every add. This is the silent-corruption baseline; if the
    // backend's reduce_sum mis-routed to a bf16 accumulator the
    // backend result would match this naive number.
    uint16_t acc_bits = 0x0000;
    for (int i = 0; i < {n}; i++) {{
        float a = chelis_bf16_to_f32(acc_bits) + 0.01f;
        acc_bits = chelis_f32_to_bf16(a);
    }}
    printf("%.6f\n", chelis_bf16_to_f32(acc_bits));
    chelis_free(x);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("bf16_sum_1024", &result.c_source, &main_c, false);
    let mut lines = stdout.lines();
    let backend_result: f64 = lines.next().unwrap().trim().parse().unwrap();
    let naive_result: f64 = lines.next().unwrap().trim().parse().unwrap();
    let expected = 10.24_f64;
    assert!(
        (backend_result - expected).abs() <= BF16_TOL,
        "backend reduce_sum result {backend_result} not within {BF16_TOL} of expected {expected}"
    );
    assert!(
        (naive_result - expected).abs() > 0.1,
        "naive bf16-direct accumulator {naive_result} is too close to {expected}; the test \
         cannot prove the f32 accumulator path is doing real work"
    );
}

#[test]
fn f16_reduce_sum_uses_f32_accumulator_per_spec_5_7_1() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let n = 1024;
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, Prim::F16),
        None,
    );
    let sum_op = RiscOp::sum_default(0, Prim::F16).expect("sum constructs");
    dag.add_node(sum_op, vec![load], scalar_ty(Prim::F32), None);
    let result = codegen(&dag, "f16_sum_1024").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void f16_sum_1024(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float vals[{n}];
    for (int i = 0; i < {n}; i++) vals[i] = 0.01f;
    chelis_tensor *x = f16_tensor_from_f32(vals, {n});
    chelis_tensor *inputs[1] = {{x}};
    chelis_tensor *outputs[1] = {{0}};
    f16_sum_1024(inputs, 1, outputs, 1);
    printf("%.6f\n", ((float*)outputs[0]->data)[0]);
    uint16_t acc_bits = 0x0000;
    for (int i = 0; i < {n}; i++) {{
        float a = chelis_f16_to_f32(acc_bits) + 0.01f;
        acc_bits = chelis_f32_to_f16(a);
    }}
    printf("%.6f\n", chelis_f16_to_f32(acc_bits));
    chelis_free(x);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("f16_sum_1024", &result.c_source, &main_c, false);
    let mut lines = stdout.lines();
    let backend_result: f64 = lines.next().unwrap().trim().parse().unwrap();
    let naive_result: f64 = lines.next().unwrap().trim().parse().unwrap();
    let expected = 10.24_f64;
    assert!(
        (backend_result - expected).abs() <= F16_TOL * 100.0,
        "backend reduce_sum result {backend_result} not within tolerance of {expected}"
    );
    // f16 has finer mantissa than bf16, so the naive accumulator
    // diverges by less; pin the gap loosely (>0.005) so the
    // pass-or-fail distinguishes a real f32-accumulator path from a
    // mis-routed f16-direct one.
    assert!(
        (naive_result - expected).abs() > 0.005,
        "naive f16-direct accumulator {naive_result} should differ from {expected} by more \
         than 0.005; if not, the test cannot prove the f32 accumulator path is doing real work"
    );
}

#[test]
fn bf16_reduce_max_agrees_with_evaluator() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let n = 4;
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, Prim::Bf16),
        None,
    );
    dag.add_node(
        RiscOp::MaxReduce { axis: 0 },
        vec![load],
        scalar_ty(Prim::Bf16),
        None,
    );
    let result = codegen(&dag, "bf16_max").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void bf16_max(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float vals[{n}] = {{1.0f, 3.5f, 2.0f, -1.0f}};
    chelis_tensor *x = bf16_tensor_from_f32(vals, {n});
    chelis_tensor *inputs[1] = {{x}};
    chelis_tensor *outputs[1] = {{0}};
    bf16_max(inputs, 1, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("%.6f\n", chelis_bf16_to_f32(p[0]));
    chelis_free(x);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("bf16_max", &result.c_source, &main_c, false);
    let c_val: f64 = stdout.trim().parse().unwrap();
    let expected = 3.5_f64;
    assert!(
        (c_val - expected).abs() <= BF16_TOL,
        "bf16 reduce_max: got {c_val}, expected {expected}"
    );
}

#[test]
fn f16_reduce_max_agrees_with_evaluator() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let n = 4;
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, Prim::F16),
        None,
    );
    dag.add_node(
        RiscOp::MaxReduce { axis: 0 },
        vec![load],
        scalar_ty(Prim::F16),
        None,
    );
    let result = codegen(&dag, "f16_max").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void f16_max(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float vals[{n}] = {{1.0f, 3.5f, 2.0f, -1.0f}};
    chelis_tensor *x = f16_tensor_from_f32(vals, {n});
    chelis_tensor *inputs[1] = {{x}};
    chelis_tensor *outputs[1] = {{0}};
    f16_max(inputs, 1, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("%.6f\n", chelis_f16_to_f32(p[0]));
    chelis_free(x);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("f16_max", &result.c_source, &main_c, false);
    let c_val: f64 = stdout.trim().parse().unwrap();
    let expected = 3.5_f64;
    assert!(
        (c_val - expected).abs() <= F16_TOL,
        "f16 reduce_max: got {c_val}, expected {expected}"
    );
}

// ---------------------------------------------------------------------
// Matmul: convert-then-cblas_sgemm
// ---------------------------------------------------------------------

fn build_bf16_matmul_dag() -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_ty(2, 3, Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_ty(3, 4, Prim::Bf16),
        None,
    );
    let mm = RiscOp::matmul_default(
        vec![],
        chelis_ir::dag::DimExpr::Concrete(2),
        chelis_ir::dag::DimExpr::Concrete(4),
        chelis_ir::dag::DimExpr::Concrete(3),
        Prim::Bf16,
    )
    .expect("bf16 matmul constructs");
    dag.add_node(mm, vec![a, b], mat_ty(2, 4, Prim::Bf16), None);
    dag
}

fn build_f16_matmul_dag() -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_ty(2, 3, Prim::F16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_ty(3, 4, Prim::F16),
        None,
    );
    let mm = RiscOp::matmul_default(
        vec![],
        chelis_ir::dag::DimExpr::Concrete(2),
        chelis_ir::dag::DimExpr::Concrete(4),
        chelis_ir::dag::DimExpr::Concrete(3),
        Prim::F16,
    )
    .expect("f16 matmul constructs");
    dag.add_node(mm, vec![a, b], mat_ty(2, 4, Prim::F16), None);
    dag
}

/// Structural test: the emitted C for a bf16 matmul contains both
/// `chelis_bf16_to_f32` (the operand-side conversion call) and
/// `cblas_sgemm` (the BLAS dispatch). Runs unconditionally; does not
/// need gcc, libcblas, or any runtime. This is the lock for the
/// convert-then-sgemm routing decision.
#[test]
fn bf16_matmul_routes_through_convert_then_sgemm() {
    let dag = build_bf16_matmul_dag();
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let result = codegen(&specialized, "bf16_matmul_routing").unwrap();
    assert!(
        result.c_source.contains("chelis_bf16_to_f32"),
        "emitted C must convert bf16 operands to f32: {}",
        result.c_source
    );
    assert!(
        result.c_source.contains("cblas_sgemm"),
        "emitted C must dispatch cblas_sgemm: {}",
        result.c_source
    );
    assert!(
        result.c_source.contains("chelis_f32_to_bf16"),
        "emitted C must downcast f32 accumulator back to bf16 for the destination: {}",
        result.c_source
    );
}

#[test]
fn f16_matmul_routes_through_convert_then_sgemm() {
    let dag = build_f16_matmul_dag();
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let result = codegen(&specialized, "f16_matmul_routing").unwrap();
    assert!(
        result.c_source.contains("chelis_f16_to_f32"),
        "emitted C must convert f16 operands to f32: {}",
        result.c_source
    );
    assert!(
        result.c_source.contains("cblas_sgemm"),
        "emitted C must dispatch cblas_sgemm: {}",
        result.c_source
    );
    assert!(
        result.c_source.contains("chelis_f32_to_f16"),
        "emitted C must downcast f32 accumulator back to f16 for the destination: {}",
        result.c_source
    );
}

#[test]
fn bf16_matmul_agrees_with_evaluator() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    if !cblas_available() {
        eprintln!("skipping: libcblas not available");
        return;
    }
    let dag = build_bf16_matmul_dag();
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let result = codegen(&specialized, "bf16_matmul_exec").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void bf16_matmul_exec(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float a_vals[6] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
    float b_vals[12] = {{
        1.0f, 0.0f, 2.0f, 1.0f,
       -1.0f, 3.0f, 0.5f, 2.0f,
        4.0f, -2.0f, 1.0f, 0.0f
    }};
    chelis_tensor *a = bf16_matrix_from_f32(a_vals, 2, 3);
    chelis_tensor *b = bf16_matrix_from_f32(b_vals, 3, 4);
    chelis_tensor *inputs[2] = {{a, b}};
    chelis_tensor *outputs[1] = {{0}};
    bf16_matmul_exec(inputs, 2, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    for (int i = 0; i < 8; i++) {{
        if (i > 0) printf(" ");
        printf("%.4f", chelis_bf16_to_f32(p[i]));
    }}
    printf("\n");
    chelis_free(a);
    chelis_free(b);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("bf16_matmul", &result.c_source, &main_c, true);
    let got: Vec<f32> = stdout
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    // Reference computed in f32 directly: A(2x3) @ B(3x4).
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [
        1.0f32, 0.0, 2.0, 1.0, -1.0, 3.0, 0.5, 2.0, 4.0, -2.0, 1.0, 0.0,
    ];
    let mut expected = [0.0f32; 8];
    for i in 0..2 {
        for j in 0..4 {
            for k in 0..3 {
                expected[i * 4 + j] += a[i * 3 + k] * b[k * 4 + j];
            }
        }
    }
    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let rel = (g - e).abs() / (e.abs() + 1e-3);
        assert!(
            rel <= 1e-2,
            "bf16 matmul[{idx}]: got {g}, expected {e}, rel diff {rel}"
        );
    }
}

#[test]
fn f16_matmul_agrees_with_evaluator() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    if !cblas_available() {
        eprintln!("skipping: libcblas not available");
        return;
    }
    let dag = build_f16_matmul_dag();
    let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
    let result = codegen(&specialized, "f16_matmul_exec").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void f16_matmul_exec(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float a_vals[6] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
    float b_vals[12] = {{
        1.0f, 0.0f, 2.0f, 1.0f,
       -1.0f, 3.0f, 0.5f, 2.0f,
        4.0f, -2.0f, 1.0f, 0.0f
    }};
    chelis_tensor *a = f16_matrix_from_f32(a_vals, 2, 3);
    chelis_tensor *b = f16_matrix_from_f32(b_vals, 3, 4);
    chelis_tensor *inputs[2] = {{a, b}};
    chelis_tensor *outputs[1] = {{0}};
    f16_matmul_exec(inputs, 2, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    for (int i = 0; i < 8; i++) {{
        if (i > 0) printf(" ");
        printf("%.4f", chelis_f16_to_f32(p[i]));
    }}
    printf("\n");
    chelis_free(a);
    chelis_free(b);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("f16_matmul", &result.c_source, &main_c, true);
    let got: Vec<f32> = stdout
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [
        1.0f32, 0.0, 2.0, 1.0, -1.0, 3.0, 0.5, 2.0, 4.0, -2.0, 1.0, 0.0,
    ];
    let mut expected = [0.0f32; 8];
    for i in 0..2 {
        for j in 0..4 {
            for k in 0..3 {
                expected[i * 4 + j] += a[i * 3 + k] * b[k * 4 + j];
            }
        }
    }
    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let rel = (g - e).abs() / (e.abs() + 1e-3);
        assert!(
            rel <= 1e-3,
            "f16 matmul[{idx}]: got {g}, expected {e}, rel diff {rel}"
        );
    }
}
