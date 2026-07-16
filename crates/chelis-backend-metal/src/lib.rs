//! Metal GPU code generation backend for the Chelis language.
//!
//! Generates Objective-C++ host source (`.mm`) with embedded MSL kernel
//! source strings. At runtime, `[MTLDevice newLibraryWithSource:options:error:]`
//! JIT-compiles the kernels and dispatches them to the Apple-Silicon GPU.
//! This is the direct analog of how the HIP backend uses `hiprtc`.
//!
//! Architectural decision: the Metal backend is **pure string emission** in
//! Rust. There is no `metal-rs` / `objc` / `cocoa` crate dep. Apple-SDK
//! integration happens later when the user runs
//! `clang++ -fobjc-arc -framework Metal -framework Foundation` against the
//! emitted `.mm`. This keeps the crate platform-portable (it builds on
//! Linux unchanged) and mirrors HIP exactly.
//!
//! See `spec/design/chelis_metal_backend_plan.md` and
//! `spec/08-backends.md` §4 (Phase M).

use std::collections::HashMap;

use chelis_ir::dag::DimExpr;

pub mod blas;
pub mod dtype;
pub mod emit;
pub mod kernels;

/// Result of Metal code generation.
///
/// Field-for-field peer of `chelis_backend_hip::HipCodegenResult`. Renames:
/// `c_source` → `mm_source` because the generated host file is Objective-C++
/// (`.mm`), not C.
pub struct MetalCodegenResult {
    /// Generated Objective-C++ host source. Embeds MSL kernel strings as
    /// C++11 raw string literals and `#import "chelis_metal_runtime.h"`.
    pub mm_source: String,
    /// Generated header declaration for the function (`extern "C"`).
    pub h_header: String,
    /// Compiler flags required (passed to `clang++`).
    pub compile_flags: Vec<String>,
    /// Linker flags required: `-framework Metal -framework Foundation
    /// -framework MetalPerformanceShaders` (the third was added in WS-M1
    /// for the f32/f16 MPS matmul dispatch path).
    pub link_flags: Vec<String>,
    /// Input slot labels in positional order.
    pub input_labels: Vec<String>,
    /// Output slot labels in positional order.
    pub output_labels: Vec<String>,
    /// Unresolved symbolic dimensions that the generated function binds from
    /// input metadata at runtime.
    pub symbolic_dims: Vec<String>,
    /// Human-readable peak device-memory formula. On Apple Silicon (unified
    /// memory) this is also peak system RAM consumed for tensor storage.
    pub peak_device_bytes_formula: String,
    /// Concrete peak device-memory estimate when every term is statically known.
    pub peak_device_bytes_estimate: Option<usize>,
    /// Symbolic terms for builds with unbound dims. Empty in the M-phase
    /// today because the emitter rejects symbolic dims at the host site;
    /// kept as a vec for parity with `chelis_backend_hip::HipCodegenResult`
    /// so a future symbolic-dim phase can populate it without an API
    /// break.
    peak_device_bytes_terms: Vec<DimExpr>,
    /// Static byte count from fully-resolved buffer allocations.
    peak_device_bytes_static_extra: usize,
}

impl MetalCodegenResult {
    /// Resolve `peak_device_bytes_estimate` against runtime symbolic-dim
    /// bindings. The M-phase today emits no symbolic terms, so this
    /// reduces to `peak_device_bytes_static_extra` regardless of
    /// `bindings` content. When a future phase adds symbolic-dim support,
    /// `peak_device_bytes_terms` will carry the per-binding contribution
    /// and this fn becomes load-bearing.
    pub fn peak_device_bytes_at(&self, bindings: &HashMap<String, usize>) -> Result<usize, String> {
        self.peak_device_bytes_terms
            .iter()
            .try_fold(self.peak_device_bytes_static_extra, |acc, term| {
                Ok(acc + term.evaluate(bindings)?)
            })
    }
}

/// Return the path to the Metal runtime directory (relative to the crate root).
#[must_use]
pub fn runtime_dir() -> &'static str {
    "runtime"
}

/// Generate Metal GPU source code from a RISC DAG.
///
/// The generated code follows the same ABI as the C and HIP backends:
/// ```c
/// void func_name(chelis_tensor **inputs, int n_in,
///                chelis_tensor **outputs, int n_out);
/// ```
///
/// **M2 first cut:** real elementwise emission for same-shape contiguous
/// rank-1 tensors. Falls back to a stub-with-abort body when the M2
/// emitter rejects the DAG (e.g., reductions land in M4, matmul in M5,
/// broadcasts/strides incremental). The stub still links and emits the
/// correct ABI, so CLI/structural tests remain stable as the supported
/// surface grows.
#[must_use]
pub fn codegen_metal(dag: &chelis_ir::dag::Dag, func_name: &str) -> MetalCodegenResult {
    let input_labels = emit::input_labels(dag);
    let output_labels = emit::output_labels(dag);
    let symbolic_dims = chelis_ir::dag::symbolic_params(dag);

    let (mm_source, peak_device_bytes) = match emit::emit_dag(dag, func_name) {
        Ok(r) => (r.mm_source, r.peak_device_bytes),
        // Unsupported DAG shape: keep the stub so the build pipeline (CLI
        // dispatch, file emission, link recipe) stays consistent. Calling
        // the stub aborts at runtime, surfacing the unsupported case
        // rather than silently miscompiling. Peak bytes is 0 in this
        // case; the stub allocates nothing. The error reason is threaded
        // into the stub so the user sees a meaningful diagnostic
        // (e.g., the integer-matmul §5.7.2 hint) instead of a bare
        // "not yet implemented" string.
        Err(reason) => {
            let hint = emit::stub_reason_hint(dag, &reason);
            (emit::stub_mm_source_with_reason(func_name, &hint), 0)
        }
    };
    let h_header = format!(
        "extern \"C\" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"
    );

    // The formula string carries the unified-memory note; the CLI then
    // prefixes it with the user-facing "Apple Silicon: also peak system
    // RAM for tensor storage" explanation when printing the build recipe.
    let peak_device_bytes_formula = format!(
        "{peak_device_bytes} bytes (sum of per-op contiguous buffer allocations; \
         Apple Silicon unified memory means this is also peak system RAM for tensor storage)"
    );

    MetalCodegenResult {
        mm_source,
        h_header,
        compile_flags: vec!["-std=c++17".to_string(), "-fobjc-arc".to_string()],
        link_flags: vec![
            "-framework".to_string(),
            "Metal".to_string(),
            "-framework".to_string(),
            "Foundation".to_string(),
            // WS-M1: MPSMatrixMultiplication is the f32/f16 matmul
            // dispatch path; integer matmul is rejected at type-check
            // (spec/04-type-system.md §5.7.2) and bf16 falls back to
            // the tiled MSL kernel via `kernels::matmul_tiled_kernel`.
            "-framework".to_string(),
            "MetalPerformanceShaders".to_string(),
        ],
        input_labels,
        output_labels,
        symbolic_dims,
        peak_device_bytes_formula,
        peak_device_bytes_estimate: Some(peak_device_bytes),
        peak_device_bytes_terms: Vec::new(),
        peak_device_bytes_static_extra: peak_device_bytes,
    }
}
