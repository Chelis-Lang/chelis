//! Metal GPU code generation backend for the Chelis language.
//!
//! Generates Objective-C++ host source (`.mm`) with embedded MSL kernel
//! source strings. At runtime, `[MTLDevice newLibraryWithSource:options:error:]`
//! JIT-compiles the kernels and dispatches them to the Apple-Silicon GPU.
//! This is the direct analog of how the HIP backend uses `hiprtc`.
//!
//! Architectural decision: the Metal backend is **pure string emission** in
//! Rust. There is no `metal-rs` / `objc` / `cocoa` crate dep. Apple-SDK
//! integration happens when the CLI invokes `clang++` against the emitted `.mm`
//! (or downstream tooling compiles the `--emit-c` output). This keeps the crate platform-portable (it builds on
//! Linux unchanged) and mirrors HIP exactly.
//!
//! See `spec/design/chelis_metal_backend_plan.md` and
//! `spec/08-backends.md` §4.

/// Primitive types the Metal backend's tensor-DAG path can realize.
/// Declared from Metal hardware spec (Apple Silicon): no f64 support.
/// Declaration-only in CI (no Metal toolchain on Linux) — verified in the
/// field via [05-UNS-1] wiring when the backend rejection path fires.
pub const TENSOR_CAPABLE_PRIMS: &[chelis_types::types::Prim] = &[
    chelis_types::types::Prim::F32,
    chelis_types::types::Prim::Bool,
    chelis_types::types::Prim::Bf16,
    chelis_types::types::Prim::F16,
    chelis_types::types::Prim::Int32,
    chelis_types::types::Prim::Int64,
];

use chelis_unord::UnordMap;

use chelis_ir::dag::DimExpr;
use chelis_ir::dag::NodeId;
use chelis_ir::ownership::{VerifiedDagProgram, VerifiedDagView};
use chelis_types::unsupported::Unsupported;

pub mod blas;
pub mod dtype;
mod emit;
pub mod kernels;

/// Opaque identity for one physical Metal allocation.
///
/// The key cannot be constructed outside this crate. Equality is exposed so
/// structural tests can prove that two materialized nodes did not collapse
/// onto one allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetalAllocationId {
    key: usize,
}

#[derive(Debug)]
struct DistinctMetalAllocation {
    node: NodeId,
    id: MetalAllocationId,
}

/// Target projection whose representation cannot carry storage reuse.
///
/// [`plan_metal`] is the only construction path. The fields are private, the
/// type is neither `Clone` nor `Copy`, and it contains only the exact verified
/// DAG plus one distinct identity for every physical allocation the Metal
/// emitter will make.
pub struct MetalNeverReuse {
    program: VerifiedDagProgram,
    allocations: Vec<DistinctMetalAllocation>,
}

impl MetalNeverReuse {
    pub(crate) fn dag(&self) -> VerifiedDagView<'_> {
        self.program.emission()
    }

    pub fn allocation_for(&self, node: NodeId) -> Option<MetalAllocationId> {
        self.allocations
            .iter()
            .find(|allocation| allocation.node == node)
            .map(|allocation| allocation.id)
    }

    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }
}

/// Project a verified DAG onto Metal's closed no-reuse allocation plan.
///
/// Virtual nodes folded into a kernel are excluded. Every node that reaches
/// a physical `chelis_metal_alloc` site receives exactly one opaque identity;
/// emission checks the resulting bijection before returning an artifact.
/// The verified input is moved into the plan and cannot be reused to mint a
/// second target projection:
///
/// ```compile_fail
/// # use chelis_ir::ownership::VerifiedDagProgram;
/// fn reuse_verified(program: VerifiedDagProgram) {
///     let _plan = chelis_backend_metal::plan_metal(program);
///     let _other_projection = program.emission();
/// }
/// ```
pub fn plan_metal(program: VerifiedDagProgram) -> MetalNeverReuse {
    let allocations = emit::allocation_nodes(program.emission())
        .into_iter()
        .enumerate()
        .map(|(key, node)| DistinctMetalAllocation {
            node,
            id: MetalAllocationId { key },
        })
        .collect();
    MetalNeverReuse {
        program,
        allocations,
    }
}

/// Result of Metal code generation.
///
/// Field-for-field peer of `chelis_backend_hip::HipCodegenResult`. Renames:
/// `c_source` → `mm_source` because the generated host file is Objective-C++
/// (`.mm`), not C.
#[derive(Debug)]
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

/// One device translation unit used by a host-program wrapper.
pub struct MetalHostTensorHelperCodegen {
    pub name: String,
    pub result: MetalCodegenResult,
}

/// A scalar/container host wrapper plus every Count-bearing Metal helper it calls.
pub struct MetalHostProgramCodegenResult {
    pub host: chelis_backend_c::CodegenResult,
    pub device_helpers: Vec<MetalHostTensorHelperCodegen>,
}

impl MetalCodegenResult {
    /// Resolve `peak_device_bytes_estimate` against runtime symbolic-dim
    /// bindings. The M-phase today emits no symbolic terms, so this
    /// reduces to `peak_device_bytes_static_extra` regardless of
    /// `bindings` content. When a future phase adds symbolic-dim support,
    /// `peak_device_bytes_terms` will carry the per-binding contribution
    /// and this fn becomes load-bearing.
    pub fn peak_device_bytes_at(
        &self,
        bindings: &UnordMap<String, usize>,
    ) -> Result<usize, String> {
        self.peak_device_bytes_terms
            .iter()
            .try_fold(self.peak_device_bytes_static_extra, |acc, term| {
                Ok(acc + term.evaluate(bindings)?)
            })
    }
}

/// Return the path to the Metal runtime directory (relative to the crate root).
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
/// Unsupported DAGs return the typed [05-UNS] failure channel. They never
/// produce an Objective-C++ artifact containing a delayed abort stub.
///
/// A raw DAG cannot be passed directly to Metal:
///
/// ```compile_fail
/// # use chelis_ir::dag::Dag;
/// fn bypass(raw: &Dag) {
///     let _ = chelis_backend_metal::codegen_metal(raw, "unchecked");
/// }
/// ```
///
/// Neither can the verified DAG boundary used by C/HIP today:
///
/// ```compile_fail
/// # use chelis_ir::ownership::VerifiedDagProgram;
/// fn bypass(reuse_capable: &VerifiedDagProgram) {
///     let _ = chelis_backend_metal::codegen_metal(reuse_capable, "unchecked");
/// }
/// ```
///
/// A token-carrying target projection is likewise not substitutable for the
/// exact no-reuse type:
///
/// ```compile_fail
/// # use chelis_ir::ownership::VerifiedDagProgram;
/// struct ReuseCapable<'a> {
///     dag: &'a VerifiedDagProgram,
///     reusable_storage_token: &'a (),
/// }
/// fn bypass(reuse_capable: &ReuseCapable<'_>) {
///     let _ = chelis_backend_metal::codegen_metal(reuse_capable, "unchecked");
/// }
/// ```
///
/// The no-reuse plan itself has no public constructor, so a caller cannot
/// attach a reuse token or forge an incomplete allocation projection:
///
/// ```compile_fail
/// # use chelis_backend_metal::MetalNeverReuse;
/// # use chelis_ir::ownership::VerifiedDagProgram;
/// fn forge(program: VerifiedDagProgram) -> MetalNeverReuse {
///     MetalNeverReuse { program, allocations: Vec::new() }
/// }
/// ```
///
/// The plan is linear at the public emission edge: codegen consumes it, so it
/// cannot emit twice or be cloned into a second capability.
///
/// ```compile_fail
/// # use chelis_backend_metal::{codegen_metal, MetalNeverReuse};
/// fn emit_twice(plan: MetalNeverReuse) {
///     let _ = codegen_metal(plan, "first");
///     let _ = codegen_metal(plan, "second");
/// }
/// ```
///
/// ```compile_fail
/// # use chelis_backend_metal::MetalNeverReuse;
/// fn clone_plan(plan: MetalNeverReuse) {
///     let _second = plan.clone();
/// }
/// ```
pub fn codegen_metal(
    plan: MetalNeverReuse,
    func_name: &str,
) -> Result<MetalCodegenResult, Unsupported> {
    let dag = plan.dag();
    chelis_ir::dag::reject_device_transcendentals(dag.nodes(), "metal")?;
    let input_labels = emit::input_labels(dag);
    let output_labels = emit::output_labels(dag);
    let symbolic_dims = dag.symbolic_params();

    let emitted = emit::emit_verified_dag(&plan, func_name)?;
    let mm_source = emitted.mm_source;
    let peak_device_bytes = emitted.peak_device_bytes;
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

    Ok(MetalCodegenResult {
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
    })
}

/// Compile a verified host program whose Count-bearing tensor helpers run
/// on the device.
///
/// `helpers` is the wrapper's helper manifest, read off the concrete program
/// by [`chelis_backend_c::host_tensor_helper_codegen`] before payload
/// selection and ownership lowering. Every helper whose DAG contains `Count`
/// becomes its own Metal translation unit: the C wrapper declares it as an
/// external symbol, and this function lowers the helper's source DAG through
/// ownership lowering, verification, and the no-reuse plan a tensor entry
/// receives before emitting it with [`codegen_metal`]. Other tensor helpers
/// keep their C-host disposition. There is no C fallback for a Count helper
/// and no abort stub for an unsupported one: the selection predicate below is
/// the only place that decides, and the typed emitter error propagates.
pub fn codegen_metal_host_program(
    program: &chelis_ir::ownership::VerifiedHostProgram,
    func_name: &str,
    helpers: Vec<chelis_backend_c::HostTensorHelperCodegen>,
) -> Result<MetalHostProgramCodegenResult, Unsupported> {
    let count_helpers = helpers
        .into_iter()
        .filter(|helper| {
            helper
                .dag
                .nodes()
                .iter()
                .any(|node| matches!(node.op, chelis_ir::dag::RiscOp::Count { .. }))
        })
        .collect::<Vec<_>>();
    let external_names = count_helpers
        .iter()
        .map(|helper| helper.name.clone())
        .collect::<Vec<_>>();
    let host = chelis_backend_c::codegen_host_program_with_external_tensor_helpers(
        program,
        func_name,
        &external_names,
    )?;
    let device_helpers = count_helpers
        .into_iter()
        .map(|helper| {
            let verified = verified_host_helper_dag(&helper.name, helper.dag)?;
            Ok(MetalHostTensorHelperCodegen {
                result: codegen_metal(plan_metal(verified), &helper.name)?,
                name: helper.name,
            })
        })
        .collect::<Result<Vec<_>, Unsupported>>()?;
    Ok(MetalHostProgramCodegenResult {
        host,
        device_helpers,
    })
}

/// Lower one externalized helper DAG exactly as a Metal tensor entry is
/// lowered: ownership lowering, then verification.
fn verified_host_helper_dag(
    helper_name: &str,
    dag: chelis_ir::dag::Dag,
) -> Result<VerifiedDagProgram, Unsupported> {
    let unsupported = |error: chelis_ir::ownership::OwnershipError| {
        Unsupported::new(
            chelis_types::unsupported::UnsupportedKind::Construct(format!(
                "Metal device helper `{helper_name}` ownership lowering"
            )),
            error.to_string(),
            chelis_types::unsupported::Stage::Codegen("metal"),
            chelis_types::deliberate_rejection!(
                "[04-SHAPE-1]",
                "a device-emitted host tensor helper requires the same verified ownership plan as a tensor entry"
            ),
        )
    };
    chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(dag).map_err(unsupported)?,
    )
    .map_err(unsupported)
}

#[cfg(test)]
pub(crate) mod testing {
    use chelis_ir::ownership::{OwnershipError, VerifiedDagProgram};

    pub(crate) fn verified_dag(
        dag: &chelis_ir::dag::Dag,
    ) -> Result<VerifiedDagProgram, OwnershipError> {
        chelis_ir::ownership::verify_ownership(chelis_ir::ownership::lower_dag_ownership(
            dag.clone(),
        )?)
    }
}
