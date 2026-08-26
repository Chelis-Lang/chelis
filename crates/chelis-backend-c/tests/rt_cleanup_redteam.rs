//! RT-Cleanup adversarial tests for WS-1 (C backend bf16/f16 admission)
//! and WS-2 (Metal emit_const fix).
//!
//! Categories (per the RT-Cleanup brief and plan §RT-Cleanup):
//!   1. Edge-value bit-pattern preservation (subnormals, +/-0, +/-inf,
//!      NaN, max-finite, smallest-normal) through the C-backend's
//!      reduced-float arithmetic surface.
//!   2. §5.7.1 enforcement at scale (4096-element bf16 sum where the
//!      f32-accumulator vs bf16-direct divergence is pronounced).
//!   3. Convert-then-sgemm wrapper correctness (operand-precision
//!      dispatch even when the output is f32; scratch buffer
//!      alloc/free shape; pattern-detected matmul routing).
//!   4. Additional pinned Const bit-pattern sweep (0.1, 0.01, pi).
//!   5. Metal Const-rooted DAGs (non-pinned values; uint16_t cast vs
//!      bfloat/half cast).
//!   6. Cross-backend agreement on a bf16 program.
//!   7. Sibling-sweep regression locks.
//!
//! Failing tests left enabled when they reflect spec-correct behavior
//! and the post-fix lock; tests that fail on current main due to a
//! BLOCKER are marked `#[ignore = "BLOCKER: ..."]` so the workspace
//! gate stays green for orchestrator merging and WS-Cleanup-Fixups can
//! re-enable them when fixed.

use chelis_backend_c::codegen;
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_types::types::Prim;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const BF16_TOL: f64 = 1e-2;
#[allow(dead_code)]
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

#[allow(dead_code)]
fn cblas_available() -> bool {
    let dir = std::env::temp_dir().join("chelis_rtcleanup_cblas_probe");
    let _ = fs::create_dir_all(&dir);
    let probe = dir.join("probe.c");
    fs::write(
        &probe,
        "extern void cblas_sgemm(); int main(void){(void)cblas_sgemm; return 0;}",
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

fn compile_and_run_kernel(
    test_name: &str,
    c_source: &str,
    main_c: &str,
    needs_cblas: bool,
) -> String {
    let dir = std::env::temp_dir().join(format!("chelis_rtcleanup_{test_name}"));
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

const HARNESS: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "chelis_runtime.h"

static chelis_tensor *bf16_tensor_from_bits(const uint16_t *bits, int n) {
    int64_t shape[1] = {n};
    chelis_tensor *t = chelis_alloc(1, shape, CHELIS_DTYPE_BF16);
    memcpy(t->data, bits, (size_t)n * sizeof(uint16_t));
    return t;
}

static chelis_tensor *f16_tensor_from_bits(const uint16_t *bits, int n) {
    int64_t shape[1] = {n};
    chelis_tensor *t = chelis_alloc(1, shape, CHELIS_DTYPE_F16);
    memcpy(t->data, bits, (size_t)n * sizeof(uint16_t));
    return t;
}

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
"#;

fn vec_ty(n: usize, prec: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: prec,
    }
}

fn mat_ty(rows: usize, cols: usize, prec: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: prec,
    }
}

// =====================================================================
// Category 1: Edge-value bit-pattern preservation
// =====================================================================
//
// For each edge value (subnormal, +/-0, +/-inf, NaN, max-finite,
// smallest-normal): load a bf16 / f16 tensor whose elements have the
// pinned 16-bit pattern, run a trivial op (abs, or add of zero), and
// assert IEEE 754 semantics are preserved.
//
// `abs` is chosen because it's a true unary op that goes through the
// emit_unary_reduced_f path (convert-to-f32 → fabsf → convert-back).
// Round-tripping through chelis_bf16_to_f32 / chelis_f32_to_bf16 must
// preserve subnormals, zero, infinity, and NaN-ness. For NaN the bit
// payload may change (NaN payload is not preserved), but the result
// must still be a NaN (mantissa nonzero, exp all-ones).

/// Run `abs` on a hand-crafted bf16 tensor and return the result bits.
fn run_bf16_abs_with_bits(bits: u16) -> u16 {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return bits;
    }
    let n = 4;
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, Prim::Bf16),
        None,
    );
    dag.add_node(RiscOp::Abs, vec![load], vec_ty(n, Prim::Bf16), None);
    let result = codegen(&dag, "bf16_abs_edge").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void bf16_abs_edge(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    uint16_t bits[{n}] = {{0x{bits:04X}, 0x{bits:04X}, 0x{bits:04X}, 0x{bits:04X}}};
    chelis_tensor *x = bf16_tensor_from_bits(bits, {n});
    chelis_tensor *inputs[1] = {{x}};
    chelis_tensor *outputs[1] = {{0}};
    bf16_abs_edge(inputs, 1, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("0x%04X\n", (unsigned)p[0]);
    chelis_free(x);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel(
        &format!("bf16_abs_{bits:04x}"),
        &result.c_source,
        &main_c,
        false,
    );
    u16::from_str_radix(stdout.trim().trim_start_matches("0x"), 16).unwrap()
}

fn run_f16_abs_with_bits(bits: u16) -> u16 {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return bits;
    }
    let n = 4;
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, Prim::F16),
        None,
    );
    dag.add_node(RiscOp::Abs, vec![load], vec_ty(n, Prim::F16), None);
    let result = codegen(&dag, "f16_abs_edge").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void f16_abs_edge(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    uint16_t bits[{n}] = {{0x{bits:04X}, 0x{bits:04X}, 0x{bits:04X}, 0x{bits:04X}}};
    chelis_tensor *x = f16_tensor_from_bits(bits, {n});
    chelis_tensor *inputs[1] = {{x}};
    chelis_tensor *outputs[1] = {{0}};
    f16_abs_edge(inputs, 1, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("0x%04X\n", (unsigned)p[0]);
    chelis_free(x);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel(
        &format!("f16_abs_{bits:04x}"),
        &result.c_source,
        &main_c,
        false,
    );
    u16::from_str_radix(stdout.trim().trim_start_matches("0x"), 16).unwrap()
}

#[test]
fn bf16_abs_preserves_positive_zero() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // +0 stays +0.
    assert_eq!(run_bf16_abs_with_bits(0x0000), 0x0000);
}

#[test]
fn bf16_abs_flips_negative_zero_to_positive_zero() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // fabsf(-0) = +0 per IEEE 754.
    assert_eq!(run_bf16_abs_with_bits(0x8000), 0x0000);
}

#[test]
fn bf16_abs_preserves_positive_infinity() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // |+inf| = +inf.
    assert_eq!(run_bf16_abs_with_bits(0x7F80), 0x7F80);
}

#[test]
fn bf16_abs_flips_negative_infinity_to_positive_infinity() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // |-inf| = +inf.
    assert_eq!(run_bf16_abs_with_bits(0xFF80), 0x7F80);
}

#[test]
fn bf16_abs_preserves_nan_ness() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // |NaN| is still NaN (exact payload not pinned, but must satisfy the
    // bf16 NaN encoding: exp = 0xFF, mantissa != 0).
    let result = run_bf16_abs_with_bits(0x7FC0);
    let exp = (result >> 7) & 0xFF;
    let mant = result & 0x7F;
    assert_eq!(exp, 0xFF, "abs(NaN) must keep exp=0xFF, got 0x{result:04X}");
    assert!(
        mant != 0,
        "abs(NaN) must keep mantissa nonzero, got 0x{result:04X}"
    );
}

#[test]
fn bf16_abs_preserves_max_finite_bit_pattern() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // |max-finite| = max-finite. Round-trip must NOT overflow to inf.
    assert_eq!(run_bf16_abs_with_bits(0x7F7F), 0x7F7F);
}

#[test]
fn bf16_abs_preserves_smallest_subnormal() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // |smallest positive subnormal| stays the same bit pattern.
    assert_eq!(run_bf16_abs_with_bits(0x0001), 0x0001);
}

#[test]
fn bf16_abs_preserves_smallest_normal() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    assert_eq!(run_bf16_abs_with_bits(0x0080), 0x0080);
}

#[test]
fn f16_abs_preserves_positive_zero() {
    if !gcc_available() {
        return;
    }
    assert_eq!(run_f16_abs_with_bits(0x0000), 0x0000);
}

#[test]
fn f16_abs_flips_negative_zero_to_positive_zero() {
    if !gcc_available() {
        return;
    }
    assert_eq!(run_f16_abs_with_bits(0x8000), 0x0000);
}

#[test]
fn f16_abs_preserves_positive_infinity() {
    if !gcc_available() {
        return;
    }
    assert_eq!(run_f16_abs_with_bits(0x7C00), 0x7C00);
}

#[test]
fn f16_abs_flips_negative_infinity_to_positive_infinity() {
    if !gcc_available() {
        return;
    }
    assert_eq!(run_f16_abs_with_bits(0xFC00), 0x7C00);
}

#[test]
fn f16_abs_preserves_nan_ness() {
    if !gcc_available() {
        return;
    }
    let result = run_f16_abs_with_bits(0x7E00);
    let exp = (result >> 10) & 0x1F;
    let mant = result & 0x3FF;
    assert_eq!(exp, 0x1F, "abs(NaN) must keep exp=0x1F, got 0x{result:04X}");
    assert!(
        mant != 0,
        "abs(NaN) must keep mantissa nonzero, got 0x{result:04X}"
    );
}

#[test]
fn f16_abs_preserves_max_finite_bit_pattern() {
    if !gcc_available() {
        return;
    }
    assert_eq!(run_f16_abs_with_bits(0x7BFF), 0x7BFF);
}

#[test]
fn f16_abs_preserves_smallest_subnormal() {
    if !gcc_available() {
        return;
    }
    assert_eq!(run_f16_abs_with_bits(0x0001), 0x0001);
}

#[test]
fn f16_abs_preserves_smallest_normal() {
    if !gcc_available() {
        return;
    }
    assert_eq!(run_f16_abs_with_bits(0x0400), 0x0400);
}

// =====================================================================
// Category 2: §5.7.1 enforcement at scale (4096 elements)
// =====================================================================

/// A 4096 x bf16(0.001) reduce_sum is dominated by the accumulator
/// precision: the mathematical sum 4.096 stays within BF16_TOL only if
/// the accumulator is f32. A naive bf16-direct accumulator loses the
/// per-add residuals (0.001 in bf16 is ~9.77e-4 after round-trip, and
/// each running-sum truncation drops mantissa bits) and diverges by
/// more than 0.2 on this scale. This is the larger-N companion to
/// WS-1's existing 1024-element 0.01 test.
#[test]
fn bf16_reduce_sum_4096_x_0_001_uses_f32_accumulator_per_spec_5_7_1() {
    if !gcc_available() {
        return;
    }
    let n = 4096;
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty(n, Prim::Bf16),
        None,
    );
    let sum_op = RiscOp::sum_default(0, Prim::Bf16).expect("sum constructs");
    dag.add_node(
        sum_op,
        vec![load],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let result = codegen(&dag, "bf16_sum_4096").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void bf16_sum_4096(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float vals[{n}];
    for (int i = 0; i < {n}; i++) vals[i] = 0.001f;
    chelis_tensor *x = bf16_tensor_from_f32(vals, {n});
    chelis_tensor *inputs[1] = {{x}};
    chelis_tensor *outputs[1] = {{0}};
    bf16_sum_4096(inputs, 1, outputs, 1);
    float backend = ((float*)outputs[0]->data)[0];
    /* Naive bf16-direct: every add rounds back to bf16. */
    uint16_t acc = 0x0000;
    uint16_t step = chelis_f32_to_bf16(0.001f);
    float step_f = chelis_bf16_to_f32(step);
    for (int i = 0; i < {n}; i++) {{
        acc = chelis_f32_to_bf16(chelis_bf16_to_f32(acc) + step_f);
    }}
    float naive = chelis_bf16_to_f32(acc);
    printf("%.8f\n%.8f\n", backend, naive);
    chelis_free(x);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("bf16_sum_4096", &result.c_source, &main_c, false);
    let mut lines = stdout.lines();
    let backend: f64 = lines.next().unwrap().trim().parse().unwrap();
    let naive: f64 = lines.next().unwrap().trim().parse().unwrap();
    // bf16(0.001) rounds to ~9.7656e-4. Expected sum is 4096 * 9.7656e-4
    // ~= 4.0; the tolerance widens slightly relative to BF16_TOL because
    // the per-element rounding error compounds.
    let expected = 4096.0_f64 * f64::from(half::bf16::from_f64(0.001).to_f32());
    assert!(
        (backend - expected).abs() <= 0.05,
        "backend reduce_sum {backend} not within 0.05 of expected {expected}; \
         the f32 accumulator path failed at N=4096"
    );
    assert!(
        (naive - expected).abs() > 0.2,
        "naive bf16-direct {naive} too close to expected {expected}; the \
         test cannot prove the f32 accumulator path is doing real work at N=4096"
    );
}

// =====================================================================
// Category 3: convert-then-sgemm routing on operand precision (not output)
// =====================================================================

/// The matmul-pattern detector emits BlasMatmul nodes whose output
/// precision is the accumulator precision (f32 for bf16/f16 operands),
/// not the operand precision. WS-1's emit_blas_matmul dispatches the
/// reduced-float path on `spec.operand_precision`, NOT on output type,
/// so a bf16-input/f32-output matmul still routes through the
/// convert-then-sgemm wrapper. Lock this behavior: building a
/// BlasMatmul with bf16 operands and an f32-typed output node must
/// still emit element-wise `chelis_bf16_to_f32` conversion in the generated C.
#[test]
fn bf16_matmul_with_f32_output_still_routes_through_convert_wrapper() {
    use chelis_ir::dag::DimExpr;
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
    // Construct a BlasMatmul directly with operand bf16 + f32
    // accumulator. We bypass matmul_default's result-type construction
    // and feed an f32 output type to mimic what the
    // matmul-pattern detector emits after AD lowering when the Sum's
    // accumulator-pinned output precision is f32.
    let mm = RiscOp::BlasMatmul {
        batch_dims: vec![],
        m: DimExpr::Concrete(2),
        n: DimExpr::Concrete(4),
        k: DimExpr::Concrete(3),
        accumulator: Prim::F32,
    };
    dag.add_node(mm, vec![a, b], mat_ty(2, 4, Prim::F32), None);
    let result = codegen(&dag, "bf16_mm_f32_out").unwrap();
    assert!(
        result.c_source.contains("chelis_bf16_to_f32"),
        "bf16-operand matmul with f32 output must convert operands through \
         chelis_bf16_to_f32, not read them as raw f32:\n{}",
        result.c_source
    );
    assert!(
        result.c_source.contains("cblas_sgemm"),
        "must dispatch cblas_sgemm:\n{}",
        result.c_source
    );
    // When output is f32, no chelis_f32_to_bf16 downcast is
    // emitted (cblas_sgemm writes directly into t{id}->data).
    assert!(
        !result.c_source.contains("chelis_f32_to_bf16"),
        "bf16-operand matmul with f32 output must NOT downcast back to bf16:\n{}",
        result.c_source
    );
}

#[test]
fn f16_matmul_with_f32_output_still_routes_through_convert_wrapper() {
    use chelis_ir::dag::DimExpr;
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
    let mm = RiscOp::BlasMatmul {
        batch_dims: vec![],
        m: DimExpr::Concrete(2),
        n: DimExpr::Concrete(4),
        k: DimExpr::Concrete(3),
        accumulator: Prim::F32,
    };
    dag.add_node(mm, vec![a, b], mat_ty(2, 4, Prim::F32), None);
    let result = codegen(&dag, "f16_mm_f32_out").unwrap();
    assert!(
        result.c_source.contains("chelis_f16_to_f32"),
        "f16-operand matmul with f32 output must convert operands:\n{}",
        result.c_source
    );
    assert!(
        !result.c_source.contains("chelis_f32_to_f16"),
        "must not downcast back to f16 when output is f32:\n{}",
        result.c_source
    );
}

/// The matmul wrapper allocates scratch buffers `t{id}_af`, `t{id}_bf`,
/// and (when output is reduced-float) `t{id}_cf`, and must free EVERY
/// scratch buffer it allocates. A mismatched count would leak per call.
/// Lock by counting `malloc(... * sizeof(float))` vs `free(t...)` lines
/// in the emitted source for both output-precision cases.
#[test]
fn bf16_matmul_wrapper_balances_scratch_alloc_and_free_when_output_is_bf16() {
    use chelis_ir::dag::DimExpr;
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
    let mm = RiscOp::BlasMatmul {
        batch_dims: vec![],
        m: DimExpr::Concrete(2),
        n: DimExpr::Concrete(4),
        k: DimExpr::Concrete(3),
        accumulator: Prim::F32,
    };
    dag.add_node(mm, vec![a, b], mat_ty(2, 4, Prim::Bf16), None);
    let result = codegen(&dag, "bf16_mm_alloc_free").unwrap();
    let src = &result.c_source;
    let mallocs = src.matches("malloc((size_t)").count();
    let frees_in_wrapper = src.matches("free(t").filter(|_| true).count();
    // The wrapper should allocate 3 scratch buffers (af, bf, cf) and
    // free 3. If the count diverges, the emitter has a leak.
    assert_eq!(
        mallocs, 3,
        "bf16 matmul wrapper must allocate exactly 3 scratch buffers (af, bf, cf):\n{src}"
    );
    // free(t{id}_af), free(t{id}_bf), free(t{id}_cf) plus the contiguity
    // chelis_free's (`if (t{id}_a != t{a}) chelis_free(t{id}_a);` and
    // similar for _b). Match the bare `free(t` and the wrapper-internal
    // pattern.
    let af_free = src.contains("free(t") && src.contains("_af);");
    let bf_free = src.contains("free(t") && src.contains("_bf);");
    let cf_free = src.contains("free(t") && src.contains("_cf);");
    assert!(
        af_free && bf_free && cf_free,
        "all three scratch buffers (af, bf, cf) must be freed:\n{src}"
    );
    assert!(
        frees_in_wrapper >= 3,
        "expected at least 3 `free(t` calls (af/bf/cf), got {frees_in_wrapper}:\n{src}"
    );
}

#[test]
fn bf16_matmul_wrapper_balances_scratch_alloc_and_free_when_output_is_f32() {
    use chelis_ir::dag::DimExpr;
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
    let mm = RiscOp::BlasMatmul {
        batch_dims: vec![],
        m: DimExpr::Concrete(2),
        n: DimExpr::Concrete(4),
        k: DimExpr::Concrete(3),
        accumulator: Prim::F32,
    };
    dag.add_node(mm, vec![a, b], mat_ty(2, 4, Prim::F32), None);
    let result = codegen(&dag, "bf16_mm_alloc_free_f32_out").unwrap();
    let src = &result.c_source;
    let mallocs = src.matches("malloc((size_t)").count();
    // When output is f32, only af and bf are scratch; cf is the result tensor itself.
    assert_eq!(
        mallocs, 2,
        "bf16 matmul wrapper with f32 output must allocate exactly 2 scratch buffers \
         (af, bf; cf=destination):\n{src}"
    );
    assert!(
        !src.contains("_cf"),
        "no cf scratch buffer should be allocated when output is f32:\n{src}"
    );
}

// =====================================================================
// Category 4: Additional Const-fill bit-pattern sweep
// =====================================================================

#[test]
fn bf16_const_fill_pinned_bit_patterns_for_0_1_0_01_pi() {
    if !gcc_available() {
        return;
    }
    // Compute expected bit patterns at test-build time via the `half`
    // crate so the test pins the post-round-trip value the codegen
    // uses; if codegen's `half::bf16::from_f64(...).to_bits()` ever
    // returns something different, this test trips.
    let cases: &[(f64, u16)] = &[
        (0.1_f64, half::bf16::from_f64(0.1).to_bits()),
        (0.01_f64, half::bf16::from_f64(0.01).to_bits()),
        (
            std::f64::consts::PI,
            half::bf16::from_f64(std::f64::consts::PI).to_bits(),
        ),
    ];
    for &(value, expected) in cases {
        let n = 4;
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::synth_const(Prim::Bf16, value),
            vec![],
            vec_ty(n, Prim::Bf16),
            None,
        );
        let result = codegen(&dag, "bf16_const_extra").unwrap();
        let main_c = format!(
            r#"{HARNESS}
extern void bf16_const_extra(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    bf16_const_extra(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("0x%04X\n", (unsigned)p[0]);
    chelis_free(outputs[0]);
    return 0;
}}
"#
        );
        let stdout = compile_and_run_kernel(
            &format!("bf16_const_{}", value.to_bits()),
            &result.c_source,
            &main_c,
            false,
        );
        let got = u16::from_str_radix(stdout.trim().trim_start_matches("0x"), 16).unwrap();
        assert_eq!(
            got, expected,
            "bf16({value}): expected 0x{expected:04X}, got 0x{got:04X}"
        );
    }
}

#[test]
fn f16_const_fill_pinned_bit_patterns_for_0_1_0_01_pi() {
    if !gcc_available() {
        return;
    }
    let cases: &[(f64, u16)] = &[
        (0.1_f64, half::f16::from_f64(0.1).to_bits()),
        (0.01_f64, half::f16::from_f64(0.01).to_bits()),
        (
            std::f64::consts::PI,
            half::f16::from_f64(std::f64::consts::PI).to_bits(),
        ),
    ];
    for &(value, expected) in cases {
        let n = 4;
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::synth_const(Prim::F16, value),
            vec![],
            vec_ty(n, Prim::F16),
            None,
        );
        let result = codegen(&dag, "f16_const_extra").unwrap();
        let main_c = format!(
            r#"{HARNESS}
extern void f16_const_extra(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    f16_const_extra(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("0x%04X\n", (unsigned)p[0]);
    chelis_free(outputs[0]);
    return 0;
}}
"#
        );
        let stdout = compile_and_run_kernel(
            &format!("f16_const_{}", value.to_bits()),
            &result.c_source,
            &main_c,
            false,
        );
        let got = u16::from_str_radix(stdout.trim().trim_start_matches("0x"), 16).unwrap();
        assert_eq!(
            got, expected,
            "f16({value}): expected 0x{expected:04X}, got 0x{got:04X}"
        );
    }
}

// =====================================================================
// Category 4b: Cast between bf16/f16 and f32 must call the conversion
// helpers (NOT the C integer cast `(uint16_t)v`, which truncates the
// float value to an integer).
// =====================================================================

/// Cast f32 → bf16 of a scalar value of 1.5 must produce the bf16 bit
/// pattern 0x3FC0 (the encoding of 1.5), NOT 0x0001 (the result of
/// `(uint16_t)1.5f`, which truncates the float to int).
///
/// Post WS-Cleanup-Fixups: `emit_cast` routes f32 → bf16 through
/// `chelis_f32_to_bf16` (and the symmetric helpers for the other three
/// reduced-float cast directions), so this test pins the spec-correct
/// IEEE 754 rounding instead of the previous integer-truncation bug
/// (`(uint16_t)1.5f` → 0x0001 instead of bf16(1.5)=0x3FC0).
#[test]
fn cast_f32_to_bf16_preserves_value_per_ieee_754() {
    if !gcc_available() {
        return;
    }
    let n = 4;
    let mut dag = Dag::new();
    let src = dag.add_node(
        RiscOp::synth_const(vec_ty(n, Prim::F32).precision, 1.5_f64),
        vec![],
        vec_ty(n, Prim::F32),
        None,
    );
    dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::Bf16,
        },
        vec![src],
        vec_ty(n, Prim::Bf16),
        None,
    );
    let result = codegen(&dag, "cast_f32_bf16").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void cast_f32_bf16(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    cast_f32_bf16(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("0x%04X\n", (unsigned)p[0]);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("cast_f32_bf16", &result.c_source, &main_c, false);
    let got = u16::from_str_radix(stdout.trim().trim_start_matches("0x"), 16).unwrap();
    assert_eq!(
        got, 0x3FC0,
        "cast(f32, bf16) of 1.5 must produce bf16(1.5) bit pattern 0x3FC0, got 0x{got:04X}; \
         a result of 0x0001 indicates the C integer cast `(uint16_t)1.5f` is being used \
         instead of `chelis_f32_to_bf16`"
    );
}

/// Cast bf16 → f32: the bf16 storage is `uint16_t` bit pattern 0x3FC0
/// (= 1.5). Post WS-Cleanup-Fixups, the cast routes through
/// `chelis_bf16_to_f32` so the result is f32(1.5) (= 0x3FC00000), not
/// the previous int-to-float widening of the uint16_t value (16320.0).
#[test]
fn cast_bf16_to_f32_preserves_value_per_ieee_754() {
    if !gcc_available() {
        return;
    }
    let n = 4;
    let mut dag = Dag::new();
    let src = dag.add_node(
        RiscOp::synth_const(vec_ty(n, Prim::Bf16).precision, 1.5_f64),
        vec![],
        vec_ty(n, Prim::Bf16),
        None,
    );
    dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![src],
        vec_ty(n, Prim::F32),
        None,
    );
    let result = codegen(&dag, "cast_bf16_f32").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void cast_bf16_f32(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    cast_bf16_f32(NULL, 0, outputs, 1);
    float v = ((float*)outputs[0]->data)[0];
    printf("%.8f\n", v);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("cast_bf16_f32", &result.c_source, &main_c, false);
    let got: f32 = stdout.trim().parse().unwrap();
    assert!(
        (got - 1.5_f32).abs() <= BF16_TOL as f32,
        "cast(bf16, f32) of bf16(1.5)=0x3FC0 must produce f32(1.5), got {got}; \
         a result near 16320.0 (= 0x3FC0 as int) indicates the C integer cast is being used \
         instead of `chelis_bf16_to_f32`"
    );
}

/// Post WS-Cleanup-Fixups: f32 → f16 cast routes through
/// `chelis_f32_to_f16`, pinning IEEE 754 binary16 round-to-nearest-even
/// instead of the previous integer-truncation bug.
#[test]
fn cast_f32_to_f16_preserves_value_per_ieee_754() {
    if !gcc_available() {
        return;
    }
    let n = 4;
    let mut dag = Dag::new();
    let src = dag.add_node(
        RiscOp::synth_const(vec_ty(n, Prim::F32).precision, 1.5_f64),
        vec![],
        vec_ty(n, Prim::F32),
        None,
    );
    dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F16,
        },
        vec![src],
        vec_ty(n, Prim::F16),
        None,
    );
    let result = codegen(&dag, "cast_f32_f16").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void cast_f32_f16(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    cast_f32_f16(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("0x%04X\n", (unsigned)p[0]);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("cast_f32_f16", &result.c_source, &main_c, false);
    let got = u16::from_str_radix(stdout.trim().trim_start_matches("0x"), 16).unwrap();
    assert_eq!(
        got, 0x3E00,
        "cast(f32, f16) of 1.5 must produce f16(1.5) bit pattern 0x3E00, got 0x{got:04X}"
    );
}

// =====================================================================
// Category 6: Cross-backend agreement (eval vs C) for a bf16 program
// =====================================================================

/// A small bf16 program (add then mul) executed via the C backend
/// produces values within BF16_TOL of the IR evaluator's output for
/// each element. The matmul-pattern path is not exercised here (Sum
/// is not in the chain); this exercises the bare elementwise stack.
#[test]
fn cross_backend_bf16_add_mul_chain_agrees_with_evaluator() {
    if !gcc_available() {
        return;
    }
    let n = 8;
    let mut dag = Dag::new();
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
    let add = dag.add_node(RiscOp::Add, vec![a, b], vec_ty(n, Prim::Bf16), None);
    let c = dag.add_node(
        RiscOp::Load { name: "c".into() },
        vec![],
        vec_ty(n, Prim::Bf16),
        None,
    );
    dag.add_node(RiscOp::Mul, vec![add, c], vec_ty(n, Prim::Bf16), None);

    // Evaluator
    let mut inputs = std::collections::HashMap::new();
    let a_eval = chelis_ir::eval::TensorValue::from_vec(
        vec![n],
        vec![0.5, 1.5, 2.5, -0.25, 1.0, 2.0, -1.0, 0.125],
    );
    let b_eval = chelis_ir::eval::TensorValue::from_vec(
        vec![n],
        vec![0.25, 0.5, -1.5, 0.75, 0.125, -0.25, 1.0, 2.0],
    );
    let c_eval = chelis_ir::eval::TensorValue::from_vec(
        vec![n],
        vec![1.0, -1.0, 0.5, 2.0, 0.5, 1.0, -1.0, 1.0],
    );
    inputs.insert("a".to_string(), a_eval.clone());
    inputs.insert("b".to_string(), b_eval.clone());
    inputs.insert("c".to_string(), c_eval.clone());
    let evals = chelis_ir::eval::eval_tensor(&dag, &inputs).unwrap();
    let last_id = NodeId(dag.len() - 1);
    let eval_out = evals[&last_id].to_f64_lossy_vec().clone();

    // C backend
    let result = codegen(&dag, "bf16_chain").unwrap();
    let main_c = format!(
        r#"{HARNESS}
extern void bf16_chain(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    float a_vals[{n}] = {{0.5f, 1.5f, 2.5f, -0.25f, 1.0f, 2.0f, -1.0f, 0.125f}};
    float b_vals[{n}] = {{0.25f, 0.5f, -1.5f, 0.75f, 0.125f, -0.25f, 1.0f, 2.0f}};
    float c_vals[{n}] = {{1.0f, -1.0f, 0.5f, 2.0f, 0.5f, 1.0f, -1.0f, 1.0f}};
    chelis_tensor *a = bf16_tensor_from_f32(a_vals, {n});
    chelis_tensor *b = bf16_tensor_from_f32(b_vals, {n});
    chelis_tensor *c = bf16_tensor_from_f32(c_vals, {n});
    chelis_tensor *inputs[3] = {{a, b, c}};
    chelis_tensor *outputs[1] = {{0}};
    bf16_chain(inputs, 3, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    for (int i = 0; i < {n}; i++) {{
        if (i) printf(" ");
        printf("%.8f", chelis_bf16_to_f32(p[i]));
    }}
    printf("\n");
    chelis_free(a);
    chelis_free(b);
    chelis_free(c);
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    let stdout = compile_and_run_kernel("bf16_chain", &result.c_source, &main_c, false);
    let c_out: Vec<f32> = stdout
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(c_out.len(), n);
    for (idx, (g, e)) in c_out.iter().zip(eval_out.iter()).enumerate() {
        let diff = (*g as f64 - e).abs();
        assert!(
            diff <= BF16_TOL,
            "bf16 chain elem {idx}: C={g}, eval={e}, diff={diff} > {BF16_TOL}"
        );
    }
}

// =====================================================================
// Category 7: Sibling-sweep regression locks
// =====================================================================
//
// These are static-source assertions over the chelis-backend-c source
// tree; they fail if a regression reintroduces a silent-default-arm
// `_ => "float"`, a `value as f32` silent truncation outside the
// approved helper, or a stray `panic!("... bf16 ...")` / `panic!("...
// f16 ...")` in production code.

fn read_emit_src() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/emit.rs");
    std::fs::read_to_string(path).expect("read src/emit.rs")
}

fn read_host_emit_src() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/host_emit.rs");
    std::fs::read_to_string(path).expect("read src/host_emit.rs")
}

#[test]
fn sibling_sweep_no_default_float_arm_in_backend_c() {
    let emit = read_emit_src();
    let host = read_host_emit_src();
    for (label, src) in [("emit.rs", &emit), ("host_emit.rs", &host)] {
        assert!(
            !src.contains(r#"_ => "float""#),
            "{label} must not contain the silent `_ => \"float\"` default arm; \
             WS-1 removed it. A regression has reintroduced it."
        );
    }
}

#[test]
fn sibling_sweep_no_value_as_f32_in_emit_outside_helper() {
    let emit = read_emit_src();
    // The intentional precision narrowing for F32 Const fill lives in
    // `f64_to_f32_truncate`. Strip that function body and assert no
    // other `value as f32` occurs.
    let stripped = emit.replace(
        "fn f64_to_f32_truncate(v: f64) -> f32 {\n        v as f32\n    }",
        "",
    );
    // The grep target is `value as f32`; the helper uses `v as f32`,
    // which is intentionally different. If a regression adds back a
    // silent `value as f32` truncation default arm, it shows up here.
    assert!(
        !stripped.contains("value as f32"),
        "emit.rs must not contain `value as f32` outside the f64_to_f32_truncate helper"
    );
}

#[test]
fn sibling_sweep_no_bf16_f16_panic_in_production_emit() {
    let emit = read_emit_src();
    let host = read_host_emit_src();
    // Strip the rejection diagnostic string that still mentions bf16/f16
    // (e.g., "C-backend BlasMatmul supports f32, f64, bf16, and f16").
    // That mention is INFORMATIONAL ("we admit these") and not a panic
    // on bf16 itself. Look only for `panic!(` calls that include
    // `bf16` or `f16` in the panic STRING in a way that implies the
    // panic happens BECAUSE of bf16/f16.
    for (label, src) in [("emit.rs", &emit), ("host_emit.rs", &host)] {
        // Crude but adequate: forbid lines of the form
        // `panic!("...bf16...")` where the panic happens because the
        // dtype is bf16. The diagnostic phrasings WS-1 introduced
        // refer to bf16 inclusively ("admits bf16/f16"), not
        // "reaches an unsupported bf16 arm".
        for line in src.lines() {
            let l = line.trim();
            if l.starts_with("panic!(")
                && (l.contains("unsupported bf16") || l.contains("unsupported f16"))
            {
                panic!(
                    "{label}: stray bf16/f16 unsupported-precision panic remains: `{l}`. \
                     WS-1 admitted both dtypes; the panic site is now stale."
                );
            }
        }
    }
}

/// Em-dash sibling sweep. Per CLAUDE.md §8.6 and the lint-rule history
/// (`feedback_em_dash_in_test_strings`), em dashes in user-facing
/// string literals trip the lint gate. Em dashes inside Rust comments
/// are harmless (they don't reach any user-facing surface), so this
/// sweep only flags em dashes inside `"..."` string literals in the
/// WS-1 + WS-2 touched files.
///
/// The check is intentionally conservative: a multi-line string with
/// an em dash counts as a hit. Doc comments (`///`, `//!`) and `//`
/// line comments are excluded.
#[test]
fn sibling_sweep_no_em_dash_in_string_literals_in_touched_crates() {
    let roots = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/src"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-backend-metal/src"),
    ];
    let mut hits: Vec<String> = Vec::new();
    for root in roots {
        walk(&root, &mut hits);
    }
    assert!(
        hits.is_empty(),
        "em-dash in string literal (CLAUDE.md §8.6): {:?}",
        hits
    );

    fn walk(p: &Path, out: &mut Vec<String>) {
        if !p.exists() {
            return;
        }
        if p.is_file() {
            if let Some(name) = p.file_name().and_then(|n| n.to_str())
                && (name.ends_with(".rs") || name.ends_with(".h"))
                && let Ok(s) = std::fs::read_to_string(p)
            {
                scan(p, &s, out);
            }
            return;
        }
        if let Ok(entries) = std::fs::read_dir(p) {
            for e in entries.flatten() {
                walk(&e.path(), out);
            }
        }
    }

    /// Strip Rust line comments and doc comments. For surviving
    /// content, flag em dashes inside `"..."` literals. This is a
    /// crude tokenizer (it does not understand raw strings perfectly)
    /// but it catches the common case the lint rule pins.
    fn scan(p: &Path, src: &str, out: &mut Vec<String>) {
        for (lineno, line) in src.lines().enumerate() {
            // Strip leading-whitespace `//` / `///` / `//!` line.
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            // Cut off trailing `// ...` comment at first occurrence.
            let code = match line.find("//") {
                Some(idx) => &line[..idx],
                None => line,
            };
            // Find every `"..."` span; check for em dash inside.
            let mut in_string = false;
            let mut buf = String::new();
            let mut prev = '\0';
            for ch in code.chars() {
                if ch == '"' && prev != '\\' {
                    if in_string {
                        if buf.contains('\u{2014}') {
                            out.push(format!("{}:{} {}", p.display(), lineno + 1, line));
                            break;
                        }
                        buf.clear();
                    }
                    in_string = !in_string;
                } else if in_string {
                    buf.push(ch);
                }
                prev = ch;
            }
        }
    }
}
