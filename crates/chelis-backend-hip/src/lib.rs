//! HIP GPU code generation backend for the Chelis language.
//!
//! Generates C host code with embedded HIP kernel source strings.
//! At runtime, `hiprtc` JIT-compiles the kernels and dispatches them to GPU.

use std::collections::HashMap;

use chelis_ir::dag::DimExpr;

pub mod blas;
pub mod emit;
pub mod kernels;
pub mod launch;
pub mod memory;

/// Result of HIP code generation.
pub struct HipCodegenResult {
    /// Generated C host source (includes `#include "chelis_hip_runtime.h"` and kernel strings).
    pub c_source: String,
    /// Generated C header declaration for the function.
    pub h_header: String,
    /// Compiler flags required (e.g., passed to `hipcc`).
    pub compile_flags: Vec<String>,
    /// Linker flags required (e.g., `-lhiprtc`).
    pub link_flags: Vec<String>,
    /// Input slot labels in positional order.
    pub input_labels: Vec<String>,
    /// Output slot labels in positional order.
    pub output_labels: Vec<String>,
    /// Unresolved symbolic dimensions that the generated function binds from input metadata.
    pub symbolic_dims: Vec<String>,
    /// Human-readable peak device-memory formula from the slot plan plus inline staged-reduction scratch.
    pub peak_device_bytes_formula: String,
    /// Concrete peak device-memory estimate when every term is statically known.
    pub peak_device_bytes_estimate: Option<usize>,
    peak_device_bytes_terms: Vec<DimExpr>,
    peak_device_bytes_static_extra: usize,
}

impl HipCodegenResult {
    pub fn peak_device_bytes_at(&self, bindings: &HashMap<String, usize>) -> Result<usize, String> {
        self.peak_device_bytes_terms
            .iter()
            .try_fold(self.peak_device_bytes_static_extra, |acc, term| {
                Ok(acc + term.evaluate(bindings)?)
            })
    }
}

/// Return the path to the HIP runtime directory (relative to the crate root).
pub fn runtime_dir() -> &'static str {
    "runtime"
}

/// Generate HIP GPU source code from a RISC DAG.
///
/// The generated code follows the same ABI as the C backend:
/// ```c
/// void func_name(chelis_tensor **inputs, int n_in,
///                chelis_tensor **outputs, int n_out);
/// ```
///
/// Inputs arrive as host tensors, are transferred to GPU, processed via
/// HIP kernels, and results are transferred back to host tensors in outputs.
pub fn codegen_hip(dag: &chelis_ir::dag::Dag, func_name: &str) -> HipCodegenResult {
    let specialized = chelis_ir::specialize::specialize_for_blas(dag);
    let dag = &specialized;
    let (c_source, peak_device_bytes) = emit::HipEmitter::emit_dag(dag, func_name);
    let h_header = format!(
        "extern \"C\" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"
    );
    let input_labels = emit::HipEmitter::input_labels(dag);
    let output_labels = emit::HipEmitter::output_labels(dag);
    let symbolic_dims = chelis_ir::dag::symbolic_params(dag);
    let mut link_flags = vec!["-lhiprtc".to_string()];
    // WS-A3: bf16 / f16 matmul also routes through hipBLAS (via
    // `hipblasGemmEx`). Add `-lhipblas` whenever any hipblas wrapper
    // call appears in the generated source, not just the f32 ones.
    if c_source.contains("chelis_hipblas_sgemm_row_major(")
        || c_source.contains("chelis_hipblas_sgemm_batched_row_major(")
        || c_source.contains("chelis_hipblas_sgemm_strided_batched_row_major(")
        || c_source.contains("chelis_hipblas_dgemm_row_major(")
        || c_source.contains("chelis_hipblas_dgemm_batched_row_major(")
        || c_source.contains("chelis_hipblas_dgemm_strided_batched_row_major(")
        || c_source.contains("chelis_hipblas_bf16_gemm_f32_acc_row_major(")
        || c_source.contains("chelis_hipblas_f16_gemm_f32_acc_row_major(")
    {
        link_flags.push("-lhipblas".to_string());
    }
    HipCodegenResult {
        c_source,
        h_header,
        compile_flags: vec![],
        link_flags,
        input_labels,
        output_labels,
        symbolic_dims,
        peak_device_bytes_formula: peak_device_bytes.formula,
        peak_device_bytes_estimate: peak_device_bytes.estimate,
        peak_device_bytes_terms: peak_device_bytes.terms,
        peak_device_bytes_static_extra: peak_device_bytes.extra_bytes,
    }
}
