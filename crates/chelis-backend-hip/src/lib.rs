//! HIP GPU code generation backend for the Chelis language.
//!
//! Generates C host code with embedded HIP kernel source strings.
//! At runtime, `hiprtc` JIT-compiles the kernels and dispatches them to GPU.

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
    let c_source = emit::HipEmitter::emit_dag(dag, func_name);
    let h_header = format!(
        "void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"
    );
    let input_labels = emit::HipEmitter::input_labels(dag);
    let output_labels = emit::HipEmitter::output_labels(dag);
    HipCodegenResult {
        c_source,
        h_header,
        compile_flags: vec![],
        link_flags: vec!["-lhiprtc".to_string()],
        input_labels,
        output_labels,
    }
}
