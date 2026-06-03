//! WS-A3: HIP backend bf16 / f16 matmul integration tests.
//!
//! Acceptance oracle for the WS-A3 brief. Each `#[ignore]`'d test
//! exercises the manual HIP gate (requires a HIP-capable GPU plus
//! `hipcc`/`hiprtc`/`libhipblas`):
//!
//!     cargo test -p chelis-backend-hip --test bf16_f16_matmul \
//!       -- --ignored --test-threads=1
//!
//! Test inventory:
//! - `bf16_matmul_with_default_f32_accumulator_within_tolerance`
//! - `f16_matmul_with_default_f32_accumulator_within_tolerance`
//! - `bf16_matmul_with_explicit_bf16_accumulator_is_rejected_by_ir`
//!   (does NOT require a GPU; runs unconditionally — verifies the
//!   spec §5.7.1 narrowness rule before any backend dispatch)
//! - `bf16_matmul_with_explicit_f64_accumulator_is_rejected_at_codegen`
//!   (does NOT require a GPU; verifies the WS-A3-scope diagnostic for
//!   the bf16+f64 path that overlaps WS-A2's `chelis_hipblas_dgemm`)
//!
//! The codegen-only tests do not need a GPU and are not `#[ignore]`d.

use chelis_backend_hip::codegen_hip;
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// -----------------------------------------------------------------------------
// Shape helpers
// -----------------------------------------------------------------------------

fn matrix(rows: usize, cols: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: p,
    }
}

// -----------------------------------------------------------------------------
// IR-level acceptance tests (no HIP execution required)
// -----------------------------------------------------------------------------

/// Spec §5.7.1: an explicit accumulator on a bf16 matmul that is
/// narrower than the f32 default (e.g. requesting `accumulator=bf16`)
/// is a type error at IR construction. Pinned at the IR layer because
/// the contract lives in `RiscOp::matmul_with_accumulator`, not in
/// the HIP dispatch.
#[test]
fn bf16_matmul_with_explicit_bf16_accumulator_is_rejected_by_ir() {
    let err = RiscOp::matmul_with_accumulator(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::Bf16,
        Prim::Bf16,
    )
    .expect_err("bf16 accumulator on bf16 operand violates §5.7.1 narrowness");
    assert!(
        err.contains("narrower than the spec/04-type-system.md §5.7.1 default"),
        "expected §5.7.1 narrowness diagnostic; got: {err}"
    );
    assert!(
        err.contains("bf16") && err.contains("f32"),
        "diagnostic must name both the requested accumulator and the default; got: {err}"
    );
}

/// Mirror of the bf16 narrowness rule for f16. The default for f16
/// operands is also f32 per §5.7.1.
#[test]
fn f16_matmul_with_explicit_f16_accumulator_is_rejected_by_ir() {
    let err = RiscOp::matmul_with_accumulator(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F16,
        Prim::F16,
    )
    .expect_err("f16 accumulator on f16 operand violates §5.7.1 narrowness");
    assert!(
        err.contains("narrower than the spec/04-type-system.md §5.7.1 default"),
        "expected §5.7.1 narrowness diagnostic; got: {err}"
    );
}

/// Spec §5.4: bf16 + f32 elementwise add is a type error (no implicit
/// promotion). The IR forbids constructing such a node at the
/// front-end / type-check layer; this codegen-side test pins that the
/// HIP backend never has to face a mixed-precision Add by exercising
/// `RiscOp::matmul_default` etc. on consistent dtypes.
///
/// At the DAG-construction level, a mixed-precision Add can be built
/// (DAG nodes don't enforce type unification — that's the type
/// checker's job upstream), but the IR validator's C2 add-precision
/// rule rejects it. Verify here.
#[test]
fn bf16_plus_f32_add_is_rejected_by_ir_validation() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        matrix(2, 3, Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        matrix(2, 3, Prim::F32),
        None,
    );
    // Output dtype is intentionally one of the operand dtypes; the
    // validator's job is to spot the input mismatch regardless.
    let _add = dag.add_node(RiscOp::Add, vec![a, b], matrix(2, 3, Prim::Bf16), None);

    let errors = chelis_ir::verify::verify(&dag);
    assert!(
        !errors.is_empty(),
        "IR validation must reject bf16 + f32 elementwise Add per spec §5.4 \
         (no implicit promotion). Validation returned: {errors:?}"
    );
}

/// WS-A3 backend-scope decision: bf16 / f16 with a wider-than-default
/// accumulator (e.g. f64 on bf16 operands) is admissible at the IR
/// level (the spec §5.7.1 narrowness rule passes), but the HIP
/// backend in this cycle does not provide the operand-promotion path
/// to `hipblasDgemm` — that wrapper is owned by WS-A2. Codegen panics
/// with a clean, escalation-ready diagnostic citing WS-A2 rather than
/// silently downgrading the user's accumulator request to f32.
///
/// This test pins the diagnostic shape so the WS-A2 lift breaks it
/// loudly when the wider-accumulator path lands.
#[test]
#[should_panic(expected = "operand promotion to `f64`")]
fn bf16_matmul_with_explicit_f64_accumulator_panics_at_codegen() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        matrix(2, 3, Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        matrix(3, 4, Prim::Bf16),
        None,
    );
    let matmul = RiscOp::matmul_with_accumulator(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::Bf16,
        Prim::F64,
    )
    .expect("bf16+f64 accumulator constructs (wider than default = §5.7.1 admissible)");
    let out = dag.add_node(matmul, vec![a, b], matrix(2, 4, Prim::Bf16), None);
    dag.add_root(out);

    // Should panic inside emit_blas_matmul with the WS-A2-routing
    // diagnostic. The codegen panic is the test oracle.
    let _ = codegen_hip(&dag, "ws_a3_bf16_f64_acc");
}

/// Codegen smoke: bf16 matmul with the spec-default f32 accumulator
/// produces a compilable C/HIP source that calls the new
/// `chelis_hipblas_bf16_gemm_f32_acc_row_major` wrapper and links
/// `-lhipblas`. Does not require a GPU.
#[test]
fn bf16_matmul_default_accumulator_emits_bf16_gemm_wrapper() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        matrix(2, 3, Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        matrix(3, 4, Prim::Bf16),
        None,
    );
    let matmul = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::Bf16,
    )
    .expect("bf16 matmul constructs with default accumulator");
    let out = dag.add_node(matmul, vec![a, b], matrix(2, 4, Prim::Bf16), None);
    dag.add_root(out);

    let result = codegen_hip(&dag, "ws_a3_bf16_default");
    assert!(
        result
            .c_source
            .contains("chelis_hipblas_bf16_gemm_f32_acc_row_major("),
        "bf16 + default f32 accumulator must dispatch to the bf16 GemmEx wrapper; \
         got source:\n{}",
        result.c_source
    );
    assert!(
        result.link_flags.iter().any(|f| f == "-lhipblas"),
        "bf16 matmul codegen must request `-lhipblas` link flag; got {:?}",
        result.link_flags,
    );
}

/// Mirror smoke test for f16: verifies the f16 GemmEx wrapper is
/// emitted when the operand precision is f16.
#[test]
fn f16_matmul_default_accumulator_emits_f16_gemm_wrapper() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        matrix(2, 3, Prim::F16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        matrix(3, 4, Prim::F16),
        None,
    );
    let matmul = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F16,
    )
    .expect("f16 matmul constructs with default accumulator");
    let out = dag.add_node(matmul, vec![a, b], matrix(2, 4, Prim::F16), None);
    dag.add_root(out);

    let result = codegen_hip(&dag, "ws_a3_f16_default");
    assert!(
        result
            .c_source
            .contains("chelis_hipblas_f16_gemm_f32_acc_row_major("),
        "f16 + default f32 accumulator must dispatch to the f16 GemmEx wrapper; \
         got source:\n{}",
        result.c_source
    );
    assert!(
        result.link_flags.iter().any(|f| f == "-lhipblas"),
        "f16 matmul codegen must request `-lhipblas` link flag; got {:?}",
        result.link_flags,
    );
}

/// Negative parity for the bf16 wrapper test: an f32 matmul does NOT
/// emit the bf16 wrapper. Locks the per-dtype dispatch table so a
/// dispatch-table refactor that accidentally aliases the wrappers
/// fails loudly.
#[test]
fn f32_matmul_does_not_emit_bf16_or_f16_gemm_wrapper() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        matrix(2, 3, Prim::F32),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        matrix(3, 4, Prim::F32),
        None,
    );
    let matmul = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F32,
    )
    .expect("f32 matmul constructs");
    let out = dag.add_node(matmul, vec![a, b], matrix(2, 4, Prim::F32), None);
    dag.add_root(out);

    let result = codegen_hip(&dag, "ws_a3_f32_baseline");
    assert!(
        !result
            .c_source
            .contains("chelis_hipblas_bf16_gemm_f32_acc_row_major("),
        "f32 matmul must NOT route through the bf16 wrapper"
    );
    assert!(
        !result
            .c_source
            .contains("chelis_hipblas_f16_gemm_f32_acc_row_major("),
        "f32 matmul must NOT route through the f16 wrapper"
    );
    assert!(
        result.c_source.contains("chelis_hipblas_sgemm_row_major("),
        "f32 matmul must dispatch to chelis_hipblas_sgemm_row_major"
    );
}

// -----------------------------------------------------------------------------
// HIP execution harness (mirrors crates/chelis-backend-hip/tests/gpu_correctness.rs
// but specialized for bf16 / f16 input/output bit patterns).
// -----------------------------------------------------------------------------

fn write_temp_file(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("write temp file");
    path
}

fn hip_runtime_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(chelis_backend_hip::runtime_dir())
}

fn cpu_runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn cpu_runtime_library_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../target/debug/deps"),
        manifest_dir.join("../../target/release/deps"),
    ];
    if let Ok(dir) = env::var("CHELIS_RUNTIME_DIR") {
        let candidate_dir = PathBuf::from(dir);
        if let Some(path) = fs::read_dir(&candidate_dir).ok().and_then(|entries| {
            entries.flatten().map(|entry| entry.path()).find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                    .unwrap_or(false)
            })
        }) {
            return path;
        }
    }
    for dir in candidates {
        if let Some(path) = fs::read_dir(&dir).ok().and_then(|entries| {
            entries.flatten().map(|entry| entry.path()).find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                    .unwrap_or(false)
            })
        }) {
            return path;
        }
    }
    panic!("could not locate libchelis_runtime.a for backend-hip manual tests");
}

fn copy_runtime_artifacts(dst: &Path) {
    let include = cpu_runtime_include_dir();
    for header in &[
        "chelis_runtime.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        write_temp_file(
            dst,
            header,
            &fs::read_to_string(include.join(header)).unwrap_or_else(|_| panic!("read {header}")),
        );
    }
    fs::copy(cpu_runtime_library_path(), dst.join("libchelis_runtime.a"))
        .expect("copy rust runtime library");
}

fn require_hipcc() {
    let output = Command::new("hipcc")
        .arg("--version")
        .output()
        .unwrap_or_else(|err| panic!("failed to probe hipcc: {err}"));
    assert!(
        output.status.success(),
        "hipcc is required for the manual HIP gate"
    );
}

/// Storage element width in bytes for the dtype constants emitted by
/// `dtype_macro`. Mirrors `chelis_gpu_dtype_size` in
/// `chelis_hip_runtime.h`.
fn dtype_bytes(p: Prim) -> usize {
    match p {
        Prim::F32 | Prim::Int32 | Prim::Bool => 4,
        Prim::F64 | Prim::Int64 => 8,
        Prim::Bf16 | Prim::F16 => 2,
        other => panic!("ws_a3 harness does not size dtype {}", other.name()),
    }
}

/// C macro name for a dtype, matching `HipEmitter::dtype_macro`.
fn dtype_macro(p: Prim) -> &'static str {
    match p {
        Prim::F32 => "CHELIS_F32",
        Prim::F64 => "CHELIS_F64",
        Prim::Int32 => "CHELIS_I32",
        Prim::Int64 => "CHELIS_I64",
        Prim::Bool => "CHELIS_BOOL",
        Prim::Bf16 => "CHELIS_BF16",
        Prim::F16 => "CHELIS_F16",
        other => panic!("ws_a3 harness has no dtype macro for {}", other.name()),
    }
}

/// Emit C source that builds a chelis_tensor in `prefix_input_storage[slot]`
/// of shape `shape`, dtype `dtype`, populated with `data` (interpreted as f32
/// values). For bf16 / f16 the f32 values are converted into the right
/// bit pattern before being written into the 2-byte storage slots.
fn emit_input_setup(
    prefix: &str,
    slot: usize,
    shape: &[usize],
    dtype: Prim,
    data: &[f32],
) -> Vec<String> {
    let ndim = shape.len().max(1);
    let dims: Vec<usize> = if shape.is_empty() {
        vec![1]
    } else {
        shape.to_vec()
    };
    let dim_str = dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut lines = vec![
        format!("    int {prefix}_shape_{slot}[{ndim}] = {{ {dim_str} }};"),
        format!(
            "    {prefix}_input_storage[{slot}] = chelis_alloc({ndim}, {prefix}_shape_{slot}, {dtype});",
            dtype = dtype_macro(dtype),
        ),
    ];
    let elem_bytes = dtype_bytes(dtype);
    let elem_count = dims.iter().product::<usize>();
    assert_eq!(
        data.len(),
        elem_count,
        "input data length {} does not match shape product {}",
        data.len(),
        elem_count,
    );
    match dtype {
        Prim::F32 => {
            for (idx, value) in data.iter().enumerate() {
                // Sibling of #250/#251/#252: reconstruct from the exact f32
                // bit pattern via `chelis_f32_from_bits` (declared in the
                // included `chelis_runtime.h`) rather than a lossy `{:.8}f`
                // decimal literal, matching the bf16/f16 arms below which
                // already emit exact `to_bits()` patterns.
                let bits = value.to_bits();
                lines.push(format!(
                    "    {prefix}_input_storage[{slot}]->data[{idx}] = chelis_f32_from_bits(0x{bits:08x}u);"
                ));
            }
        }
        Prim::Bf16 => {
            for (idx, value) in data.iter().enumerate() {
                let bits = half::bf16::from_f32(*value).to_bits();
                lines.push(format!(
                    "    ((uint16_t*){prefix}_input_storage[{slot}]->data)[{idx}] = (uint16_t)0x{bits:04x};"
                ));
            }
        }
        Prim::F16 => {
            for (idx, value) in data.iter().enumerate() {
                let bits = half::f16::from_f32(*value).to_bits();
                lines.push(format!(
                    "    ((uint16_t*){prefix}_input_storage[{slot}]->data)[{idx}] = (uint16_t)0x{bits:04x};"
                ));
            }
        }
        other => panic!("ws_a3 harness cannot fill input dtype {}", other.name()),
    }
    let _ = elem_bytes;
    lines
}

struct ExecCase {
    /// Inputs by (label, shape, data, dtype). Data is f32 for ergonomics; the
    /// harness reinterprets into the dtype's storage on emission.
    inputs: Vec<(String, Vec<usize>, Vec<f32>, Prim)>,
    /// Expected output dtype (so the harness knows how to read the bytes back).
    output_dtype: Prim,
}

fn build_main_cpp(
    func_name: &str,
    input_labels: &[String],
    n_out: usize,
    case: &ExecCase,
) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "    chelis_tensor *case0_input_storage[{}] = {{0}};",
        input_labels.len().max(1),
    ));
    lines.push("    chelis_tensor **case0_inputs = case0_input_storage;".to_string());
    for (slot, label) in input_labels.iter().enumerate() {
        let (_, shape, data, dtype) = case
            .inputs
            .iter()
            .find(|i| i.0 == *label)
            .unwrap_or_else(|| panic!("missing input '{label}'"));
        for line in emit_input_setup("case0", slot, shape, *dtype, data) {
            lines.push(line);
        }
    }
    lines.push(format!(
        "    chelis_tensor *case0_outputs[{n_out}] = {{0}};"
    ));
    lines.push(format!(
        "    {func_name}(case0_inputs, {}, case0_outputs, {n_out});",
        input_labels.len()
    ));
    lines.push("    for (int o = 0; o < ".to_string() + &n_out.to_string() + "; o++) {");
    lines.push("        for (int i = 0; i < case0_outputs[o]->size; i++) {".to_string());
    lines.push("            if (i > 0) printf(\" \");".to_string());
    let read_expr = match case.output_dtype {
        Prim::F32 => "case0_outputs[o]->data[i]".to_string(),
        Prim::Bf16 => {
            // Reinterpret 2-byte slot as bf16 → f32 by left-shifting the
            // bf16 bit pattern into the upper half of a uint32 and
            // bit-casting to float (the standard bf16→f32 rule).
            "({\
              uint16_t b = ((uint16_t*)case0_outputs[o]->data)[i]; \
              union { uint32_t u; float f; } cv; \
              cv.u = ((uint32_t)b) << 16; \
              cv.f; })"
                .to_string()
        }
        Prim::F16 => {
            // Use HIP's __half→float by including <hip/hip_fp16.h> at
            // file scope; here we read the bit pattern and call the
            // conversion via __half_as_float-compatible path.
            // Implementation: build a __half from the bit pattern and
            // convert with __half2float.
            "__half2float(*(const __half*)((const uint16_t*)case0_outputs[o]->data + i))"
                .to_string()
        }
        other => panic!("ws_a3 harness cannot read output dtype {}", other.name()),
    };
    lines.push(format!(
        "            printf(\"%.6f\", (float)({read_expr}));"
    ));
    lines.push("        }".to_string());
    lines.push("        printf(\"\\n\");".to_string());
    lines.push("        chelis_free(case0_outputs[o]);".to_string());
    lines.push("    }".to_string());
    for slot in 0..input_labels.len() {
        lines.push(format!("    chelis_free(case0_input_storage[{slot}]);"));
    }

    let f16_include = if matches!(case.output_dtype, Prim::F16) {
        "#include <hip/hip_fp16.h>\n"
    } else {
        ""
    };
    format!(
        r#"#include "chelis_runtime.h"
#include <stdint.h>
{f16_include}extern "C" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

int main(void) {{
{body}
    return 0;
}}
"#,
        body = lines.join("\n")
    )
}

fn compile_and_run(dag: &Dag, func_name: &str, case: &ExecCase) -> Vec<f32> {
    require_hipcc();
    let result = codegen_hip(dag, func_name);
    assert_eq!(
        result.output_labels.len(),
        1,
        "ws_a3 harness expects a single output"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let hip_rt = hip_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    copy_runtime_artifacts(tmp.path());
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    write_temp_file(
        tmp.path(),
        "main.cpp",
        &build_main_cpp(
            func_name,
            &result.input_labels,
            result.output_labels.len(),
            case,
        ),
    );

    let bin_path = tmp.path().join("ws_a3_bin");
    let mut compile_cmd = Command::new("hipcc");
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("main.cpp"));
    compile_cmd.arg(tmp.path().join("model.cpp"));
    compile_cmd.arg(format!("-L{}", tmp.path().display()));
    compile_cmd.arg("-lchelis_runtime");
    compile_cmd.arg("-lpthread");
    compile_cmd.arg("-ldl");
    compile_cmd.args(&result.link_flags);
    compile_cmd.arg("-o");
    compile_cmd.arg(&bin_path);
    let compile = compile_cmd.output().expect("run hipcc");
    assert!(
        compile.status.success(),
        "hipcc compile failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr),
    );

    let run = Command::new(&bin_path).output().expect("run gpu binary");
    assert!(
        run.status.success(),
        "gpu binary failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
    let stdout = String::from_utf8(run.stdout).expect("utf8 stdout");
    stdout
        .lines()
        .next()
        .expect("at least one output line")
        .split_whitespace()
        .map(|t| t.parse::<f32>().expect("parse output float"))
        .collect()
}

/// Compute the f32 reference for a 2D matmul using a naive
/// triple-nested loop. Mirrors the f32-everywhere reference the brief
/// names as the tolerance baseline.
fn naive_f32_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn assert_within_relative_tolerance(actual: &[f32], expected: &[f32], rel_tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: output length mismatch (actual={}, expected={})",
        actual.len(),
        expected.len(),
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = e.abs().max(1e-3);
        let diff = (a - e).abs();
        assert!(
            diff <= rel_tol * scale,
            "{label}: index {idx} diff={diff} > {rel_tol} * scale ({scale}); \
             actual={a}, expected={e}"
        );
    }
}

// -----------------------------------------------------------------------------
// HIP-execution acceptance tests (manual gate)
// -----------------------------------------------------------------------------

/// Brief acceptance: bf16 matmul (rank-2, default f32 accumulator)
/// produces values within bf16 tolerance (~1e-2 relative) of the f32
/// reference. The data range is small enough to stay inside the bf16
/// finite range; large values would expand the tolerance further.
///
/// Manual-gate limitation (2026-05-11, gfx1151 wheel SDK): the wheel
/// rocBLAS bundled with `_rocm_sdk_libraries_gfx1151` returns
/// `HIPBLAS_STATUS_NOT_SUPPORTED` (status=7) for bf16/f16 matmul on
/// this device, even though the codegen path itself is correct (the
/// generated source compiles, links against `-lhipblas`, and calls
/// `hipblasGemmEx` with the spec-compliant enum values). A ROCm stack
/// that ships bf16 kernel tunings for the gfx1151 lane will pass this
/// test as written. Documented as a WS-A3 escalation in the PR
/// description; codegen-side acceptance is exercised by the
/// non-`#[ignore]`d tests above.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU, hipcc, and libhipblas with \
            bf16-tuned rocBLAS kernels (gfx1151 wheel SDK does not ship them \
            as of 2026-05-11)"]
fn bf16_matmul_with_default_f32_accumulator_within_tolerance() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        matrix(2, 3, Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        matrix(3, 4, Prim::Bf16),
        None,
    );
    let matmul = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::Bf16,
    )
    .expect("bf16 matmul constructs");
    let out = dag.add_node(matmul, vec![a, b], matrix(2, 4, Prim::Bf16), None);
    dag.add_root(out);

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![1.0, 0.0, 2.0, 1.0, -1.0, 3.0, 0.5, 2.0, 4.0, -2.0, 1.0, 0.0];
    let case = ExecCase {
        inputs: vec![
            ("a".into(), vec![2, 3], a_data.clone(), Prim::Bf16),
            ("b".into(), vec![3, 4], b_data.clone(), Prim::Bf16),
        ],
        output_dtype: Prim::Bf16,
    };
    let actual = compile_and_run(&dag, "ws_a3_bf16_matmul_default", &case);
    let expected_f32 = naive_f32_matmul(&a_data, &b_data, 2, 4, 3);
    assert_within_relative_tolerance(&actual, &expected_f32, 1e-2, "bf16 matmul vs f32 reference");
}

/// Brief acceptance: f16 matmul matches f32 reference within f16
/// tolerance (~1e-3 relative). f16 has more mantissa bits than bf16 so
/// the tolerance is tighter.
///
/// Same manual-gate limitation as the bf16 sibling above; see that
/// test's `#[ignore]` reason for the gfx1151 wheel-SDK note.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU, hipcc, and libhipblas with \
            f16-tuned rocBLAS kernels (gfx1151 wheel SDK does not ship them \
            as of 2026-05-11)"]
fn f16_matmul_with_default_f32_accumulator_within_tolerance() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        matrix(2, 3, Prim::F16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        matrix(3, 4, Prim::F16),
        None,
    );
    let matmul = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F16,
    )
    .expect("f16 matmul constructs");
    let out = dag.add_node(matmul, vec![a, b], matrix(2, 4, Prim::F16), None);
    dag.add_root(out);

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![1.0, 0.0, 2.0, 1.0, -1.0, 3.0, 0.5, 2.0, 4.0, -2.0, 1.0, 0.0];
    let case = ExecCase {
        inputs: vec![
            ("a".into(), vec![2, 3], a_data.clone(), Prim::F16),
            ("b".into(), vec![3, 4], b_data.clone(), Prim::F16),
        ],
        output_dtype: Prim::F16,
    };
    let actual = compile_and_run(&dag, "ws_a3_f16_matmul_default", &case);
    let expected_f32 = naive_f32_matmul(&a_data, &b_data, 2, 4, 3);
    assert_within_relative_tolerance(&actual, &expected_f32, 1e-3, "f16 matmul vs f32 reference");
}
