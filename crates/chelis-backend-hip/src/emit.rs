//! RISC DAG to HIP host code + kernel string emission.
//!
//! Generates C source that includes HIP runtime, embeds kernel source strings,
//! and walks the DAG in topological order launching kernels on GPU.

use std::collections::BTreeMap;

use chelis_ir::dag::{
    DagNode, DimExpr, DimInfo, ExtremaKind, ExtremaOperand, NodeId, RiscOp, RtDim, TensorType,
};
use chelis_ir::ownership::{
    HipStorageLane, StoragePlacement, VerifiedDagAction, VerifiedDagView, VerifiedStoragePlan,
};
use chelis_types::types::Prim;
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};
use chelis_types::{ElementRef, ScalarValue};

fn unsupported_verified_dag_action(node: NodeId, detail: &str) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::Op("Drop".to_string()),
        format!("verified DAG ownership action at node {}: {detail}", node.0),
        Stage::Codegen("hip"),
        chelis_types::deliberate_rejection!(
            "[04-TOT-2]",
            "verified ownership and the retained DAG payload must agree exactly; no backend-local ownership fallback is permitted"
        ),
    )
}

fn unsupported_storage_plan(error: chelis_ir::ownership::OwnershipError) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::Op("storage planning".to_string()),
        error.to_string(),
        Stage::Codegen("hip"),
        chelis_types::deliberate_rejection!(
            "[04-SHAPE-1]",
            "HIP storage placement requires the verified exact-capacity plan"
        ),
    )
}

/// chelis#616: the HIP device-kernel lane does not support runtime (node-valued)
/// movement bounds; `reject_unsupported_hip_ops` (compiler-api + CLI) rejects
/// them before codegen. This converter materializes the compile-time bound for
/// the literal launch emitters and panics on a node-valued bound as a defensive
/// backstop (only reachable if a path bypasses the reject seam).
fn hip_bound_to_usize(b: &RtDim) -> usize {
    match b {
        RtDim::Lit(n) => *n,
        RtDim::ToEnd => chelis_ir::dag::SHRINK_TO_END,
        RtDim::Node(_) => panic!(
            "HIP backend reached a node-valued (runtime) movement bound; \
             reject_unsupported_hip_ops must reject it before codegen (chelis#616)"
        ),
        RtDim::Sym(name) => panic!(
            "HIP backend reached a symbolic movement bound `{name}`; verify rejects \
             symbolic dims outside reshape targets (chelis#616)"
        ),
        RtDim::InputAxis { .. } => panic!(
            "HIP backend reached InputAxis on an owner other than Expand or Reshape; IR \
             verification must reject that owner before codegen"
        ),
    }
}

fn hip_pairs_to_usize(bounds: &[(RtDim, RtDim)]) -> Vec<(usize, usize)> {
    bounds
        .iter()
        .map(|(s, e)| (hip_bound_to_usize(s), hip_bound_to_usize(e)))
        .collect()
}

/// chelis#616: the HIP lane only supports Load-declared symbolic dims; an
/// op-declared dim implies a node-valued movement bound or runtime reshape
/// target, which `reject_unsupported_hip_ops` rejects before codegen. This
/// panic is a defensive backstop against a seam bypass.
fn require_load_source(occurrence: &chelis_ir::dag::SymbolicDimOccurrence) -> (&String, usize) {
    match &occurrence.source {
        chelis_ir::dag::SymbolicDimSource::Load { input_label, axis } => (input_label, *axis),
        chelis_ir::dag::SymbolicDimSource::OpDeclared { node, .. } => panic!(
            "HIP backend reached an op-declared runtime dim `{}` (declared by node {}); \
             reject_unsupported_hip_ops must reject it before codegen (chelis#616)",
            occurrence.name, node.0
        ),
    }
}

fn hip_strides_to_usize(strides: &[RtDim]) -> Vec<usize> {
    strides.iter().map(hip_bound_to_usize).collect()
}

use crate::blas;
use crate::fusion::{FusedReuseMechanics, HipFusedReuse};
use crate::kernels;
use crate::memory::{MemoryPlan, NodeMemoryKind};

pub(crate) struct PeakDeviceBytesBreakdown {
    pub formula: String,
    pub estimate: Option<usize>,
    pub terms: Vec<DimExpr>,
    pub extra_bytes: usize,
}

/// Emits C/HIP host source code from a RISC DAG.
pub struct HipEmitter {
    lines: Vec<String>,
    indent: usize,
    /// Collected kernel sources: (kernel_name, kernel_source_string).
    kernel_sources: Vec<(String, String)>,
    /// Planner-driven slot/wrapper ownership.
    plan: MemoryPlan,
    /// Linear reuse authorities minted by the shared planner. Both kernel
    /// signatures and host/device wrappers derive mechanics from this map.
    fused_reuse: BTreeMap<NodeId, HipFusedReuse>,
    /// FusedElem nodes inlined into a trailing reduction (no standalone emission).
    reduction_inlined: chelis_unord::UnordSet<usize>,
    /// Worst-case inline staged-reduction scratch requirement outside the slot plan.
    extra_peak_device_bytes_estimate: usize,
    /// Device entrypoints pre-allocate slot storage because borrowed input-backed views
    /// are not the first owners in the host memory plan.
    device_entrypoint_mode: bool,
    /// Shared specialization for every kernel and its launch arguments.
    kernel_rank: usize,
}

#[derive(Debug, Clone)]
#[expect(dead_code)]
struct OutputSpec {
    id: NodeId,
    label: String,
    is_store: bool,
}

struct MatmulEmitSpec {
    a: NodeId,
    b: NodeId,
    batch_dims: Vec<DimExpr>,
    m: DimExpr,
    n: DimExpr,
    k: DimExpr,
    /// Accumulator precision selected by the IR (`RiscOp::BlasMatmul`'s
    /// `accumulator` field, or the spec-default resolved from the
    /// detected matmul pattern's Sum-node accumulator). The HIP backend
    /// dispatches per `(operand_dtype, accumulator)` pair so the
    /// destructure-`..` F1 footgun (silent `hipblasSgemm` regardless
    /// of accumulator) cannot recur for bf16/f16 in this cycle.
    accumulator: Prim,
}

/// Selected matmul-dispatch wrapper. Determined by the
/// `(operand_dtype, accumulator)` pair, not by operand alone — see the
/// dispatch table in `emit_blas_matmul`. The wrapper names mirror the
/// inline functions in
/// `crates/chelis-backend-hip/runtime/chelis_hip_runtime.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatmulWrapper {
    /// f32 operands + f32 accumulator → `chelis_hipblas_sgemm_*`.
    Sgemm,
    /// f64 operands + f64 accumulator → `chelis_hipblas_dgemm_*` (WS-A2).
    Dgemm,
    /// bf16 operands + f32 accumulator → `chelis_hipblas_bf16_gemm_f32_acc_*`.
    Bf16GemmF32,
    /// f16 operands + f32 accumulator → `chelis_hipblas_f16_gemm_f32_acc_*`.
    F16GemmF32,
}

impl MatmulWrapper {
    fn row_major_call_name(self) -> &'static str {
        match self {
            MatmulWrapper::Sgemm => "chelis_hipblas_sgemm_row_major",
            MatmulWrapper::Dgemm => "chelis_hipblas_dgemm_row_major",
            MatmulWrapper::Bf16GemmF32 => "chelis_hipblas_bf16_gemm_f32_acc_row_major",
            MatmulWrapper::F16GemmF32 => "chelis_hipblas_f16_gemm_f32_acc_row_major",
        }
    }

    /// Strided-batched wrapper (rank > 2). Only Sgemm/Dgemm support
    /// strided-batched in this cycle; bf16/f16 strided-batched would
    /// require `hipblasGemmStridedBatchedEx` plumbing.
    fn strided_batched_row_major_call_name(self) -> Option<&'static str> {
        match self {
            MatmulWrapper::Sgemm => Some("chelis_hipblas_sgemm_strided_batched_row_major"),
            MatmulWrapper::Dgemm => Some("chelis_hipblas_dgemm_strided_batched_row_major"),
            MatmulWrapper::Bf16GemmF32 | MatmulWrapper::F16GemmF32 => None,
        }
    }

    /// Batched (non-strided) wrapper (rank > 2 fallback). Only
    /// Sgemm/Dgemm support batched in this cycle.
    fn batched_row_major_call_name(self) -> Option<&'static str> {
        match self {
            MatmulWrapper::Sgemm => Some("chelis_hipblas_sgemm_batched_row_major"),
            MatmulWrapper::Dgemm => Some("chelis_hipblas_dgemm_batched_row_major"),
            MatmulWrapper::Bf16GemmF32 | MatmulWrapper::F16GemmF32 => None,
        }
    }
}

impl HipEmitter {
    fn i64_c_literal(value: i64) -> String {
        if value == i64::MIN {
            "INT64_MIN".to_string()
        } else if value < 0 {
            format!("-INT64_C({})", value.unsigned_abs())
        } else {
            format!("INT64_C({value})")
        }
    }

    /// The current HIP unary kernel spells `fabsf` for every dtype. Keep
    /// integer `abs` out of that float-only template until Phase 3 supplies
    /// the typed, trapping backend kernel (chelis#699).
    fn reject_integer_abs(dag: VerifiedDagView<'_>) -> Result<(), Unsupported> {
        if let Some(node) = dag.first_integer_abs_node() {
            return Err(Unsupported::new(
                UnsupportedKind::Op("Abs".to_string()),
                format!("an integer tensor at HIP DAG node {}", node.0),
                Stage::Codegen("hip"),
                chelis_types::unimplemented_rejection!(
                    689,
                    "integer abs code generation waits for the typed, trapping Phase 3 \
                     kernel (chelis#699); use `chelis eval` for the Phase 2 reference lane"
                ),
            ));
        }
        Ok(())
    }

    fn reject_count(dag: VerifiedDagView<'_>) -> Result<(), Unsupported> {
        if let Some(node) = dag
            .nodes()
            .iter()
            .find(|node| matches!(node.op, RiscOp::Count { .. }))
        {
            return Err(Unsupported::new(
                UnsupportedKind::Op("count".to_string()),
                format!("the HIP kernel set (node {})", node.id.0),
                Stage::Codegen("hip"),
                chelis_types::unimplemented_rejection!(
                    1291,
                    "first-class count ships on eval and C-host/C-DAG in chelis#1287; \
                     chelis#1291 owns the dedicated HIP/Metal kernels"
                ),
            ));
        }
        Ok(())
    }

    /// Emit complete C/HIP source for a DAG as a function.
    pub(crate) fn emit_dag(
        mut storage_plan: VerifiedStoragePlan<HipStorageLane>,
        func_name: &str,
    ) -> Result<(String, PeakDeviceBytesBreakdown), Unsupported> {
        let dag = storage_plan.emission();
        Self::reject_integer_abs(dag)?;
        Self::reject_count(dag)?;
        // Device input ownership proves its admitted strides, not a row-major
        // layout. Only a planned, emitter-materialized slot authorizes GEMM.
        // Production preparation inserts Realize before ownership is verified.
        for node in dag.nodes() {
            let operands = if matches!(node.op, RiscOp::BlasMatmul { .. }) {
                Some([node.inputs[0], node.inputs[1]])
            } else if let Some(matmul) = blas::detect_matmul_pattern(dag, node.id)
                && Self::supports_static_hipblas_matmul(dag, &matmul, &node.output_type)
            {
                Some([matmul.a, matmul.b])
            } else {
                None
            };
            if let Some(operands) = operands {
                for operand in operands {
                    if !matches!(
                        storage_plan.placement(operand),
                        Some(StoragePlacement::OwnedSlot { .. })
                    ) {
                        return Err(unsupported_storage_plan(
                            chelis_ir::ownership::OwnershipError::LoweringInvariant {
                                unit: "hip-blas-layout".into(),
                                detail: format!(
                                    "BLAS operand {} needs prepare_dag_for_codegen before ownership lowering; borrowed or view storage does not prove contiguous materialization",
                                    operand.0
                                ),
                            },
                        ));
                    }
                }
            }
        }
        // F1 (WS-A0 RT-1 fixup, tactical) — lifted by WS-A2 (HIP f32/f64)
        // and WS-A3 (HIP bf16/f16).
        //
        // Original guard: every non-f32 BlasMatmul was rejected because
        // the HIP backend destructured BlasMatmul with `..` and silently
        // dispatched single-precision `hipblasSgemm` regardless of the
        // operand precision (silent precision loss for f64 source, undefined
        // behavior for narrower-than-f32 source). See
        // `crates/chelis-ir/src/verify.rs` for the canonical guard and
        // its `F1:` lift-target tag.
        //
        // The backend now reads `BlasMatmul.accumulator` (the F1
        // footgun-fix) and dispatches:
        //   - `hipblasSgemm` for f32 operands (WS-A2),
        //   - `hipblasDgemm` for f64 operands (WS-A2),
        //   - `hipblasGemmEx` with `HIPBLAS_COMPUTE_32F` for bf16/f16
        //     operands + spec-default f32 accumulator (WS-A3) per
        //     spec/04-type-system.md §5.7.1.
        //
        // Wider-than-default accumulators on bf16/f16 (e.g. f64
        // accumulator) require an operand-promotion path and are
        // rejected at codegen with a clean diagnostic in this cycle
        // (see emit_blas_matmul). i8 / i16 / i64 matmul are out of
        // scope for this guard per spec §5.7.2 (lifted in WS-A4).
        for node in dag.nodes() {
            if matches!(node.op, RiscOp::BlasMatmul { .. })
                && let Some(lhs) = dag.get(node.inputs[0])
            {
                let operand = lhs.output_type.precision;
                if !matches!(operand, Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16) {
                    panic!(
                        "F1: BlasMatmul on operand precision `{}` is not yet \
                         supported by the HIP backend; node {}. \
                         spec/04-type-system.md §5.7.1 documents the per-precision \
                         accumulator defaults; the HIP backend dispatches \
                         `hipblasSgemm`/`hipblasDgemm` for f32/f64 (WS-A2) and \
                         `hipblasGemmEx` for bf16/f16 (WS-A3). WS-A4 lifts the \
                         integer matmul arm per spec §5.7.2.",
                        operand.name(),
                        node.id.0,
                    );
                }
            }
        }

        let reduction_inlined = dag.reduction_inlined_fused_elems();
        let output_specs = Self::output_specs(dag);
        let plan = MemoryPlan::from_shared(&storage_plan);
        let node_ids = dag.nodes().iter().map(|node| node.id).collect::<Vec<_>>();
        let mut fused_reuse = BTreeMap::new();
        for node in node_ids {
            if let Some(token) = storage_plan
                .take_reuse_for(node)
                .map_err(unsupported_storage_plan)?
            {
                let has_later_owner = plan.slot_has_later_owner(node);
                fused_reuse.insert(node, HipFusedReuse::new(token, has_later_owner));
            }
        }
        let dag = storage_plan.emission();
        let mut e = HipEmitter {
            lines: Vec::new(),
            indent: 0,
            kernel_sources: Vec::new(),
            plan,
            fused_reuse,
            reduction_inlined: reduction_inlined
                .to_sorted()
                .into_iter()
                .map(|id| id.0)
                .collect(),
            extra_peak_device_bytes_estimate: 0,
            device_entrypoint_mode: false,
            kernel_rank: dag
                .nodes()
                .iter()
                .map(|node| node.output_type.dims.len())
                .max()
                .unwrap_or(0)
                .max(1),
        };

        // First pass: collect all needed kernel sources by walking the DAG.
        e.collect_kernels(dag)?;

        // Emit includes
        e.line("#include \"chelis_hip_runtime.h\"");
        e.line("");

        // Emit kernel source string declarations
        let ks: Vec<(String, String)> = e.kernel_sources.clone();
        for (name, source) in &ks {
            e.emit_kernel_string_decl(name, source);
        }
        if !ks.is_empty() {
            e.line("");
        }

        // Emit function signature (same ABI as C backend)
        let input_labels = Self::input_labels(dag);
        let input_slots = Self::input_slots(&input_labels);
        let expected_inputs = input_labels.len();
        let expected_outputs = output_specs.len();

        e.line(&format!(
            "extern \"C\" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out) {{"
        ));
        e.indent = 1;

        // Producer-supplied `func_name` flows into the format-string
        // context of the fprintf below; sanitize per
        // spec/upstream-bugs/producer-string-sanitization.md so a
        // forbidden byte cannot break the C string literal or be misread
        // as a `%`-specifier.
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);

        // Input/output count validation
        e.line(&format!("if (n_in != {expected_inputs}) {{"));
        e.indent += 1;
        e.line(&format!(
            "fprintf(stderr, \"{func_name_fmt}: expected %d inputs, got %d\\n\", {expected_inputs}, n_in);"
        ));
        e.line("abort();");
        e.indent -= 1;
        e.line("}");

        e.line(&format!("if (n_out != {expected_outputs}) {{"));
        e.indent += 1;
        e.line(&format!(
            "fprintf(stderr, \"{func_name_fmt}: expected %d outputs, got %d\\n\", {expected_outputs}, n_out);"
        ));
        e.line("abort();");
        e.indent -= 1;
        e.line("}");

        e.line("");

        e.emit_input_shape_preamble(dag, &input_slots, func_name, &output_specs);
        e.emit_reshape_count_preflight(dag);
        e.line("");

        // Each invocation owns modules in its current device context
        let kernel_names: Vec<String> = ks.iter().map(|(n, _)| n.clone()).collect();
        for name in &kernel_names {
            e.line(&format!(
                "hipModule_t mod_{name} = chelis_compile_kernel({name}_src, \"{name}\");"
            ));
        }
        if !kernel_names.is_empty() {
            e.line("");
        }

        // Walk DAG in topological order
        for node in dag.nodes() {
            // Skip FusedElem nodes inlined into a trailing reduction.
            if e.reduction_inlined.contains(&node.id.0) {
                continue;
            }
            if let RiscOp::Load { name } = &node.op {
                let input_idx = *input_slots
                    .get(name.as_str())
                    .unwrap_or_else(|| panic!("missing input slot for load '{name}'"));
                e.emit_span_comments(node);
                e.emit_load(node.id.0, input_idx, &node.output_type);
            } else {
                e.emit_span_comments(node);
                e.emit_node(node, dag)?;
            }
        }

        e.line("");

        // Copy outputs to host
        for (slot, output) in output_specs.iter().enumerate() {
            let id = output.id.0;
            let is_load = dag
                .get(output.id)
                .map(|n| matches!(n.op, RiscOp::Load { .. }))
                .unwrap_or(false);

            if is_load {
                // Root loads are borrowed inputs. Return an owned host tensor so
                // callers may free outputs without double-freeing their inputs.
                let load_name = match &dag.get(output.id).unwrap().op {
                    RiscOp::Load { name } => name.as_str().to_string(),
                    _ => unreachable!(),
                };
                let input_idx = input_slots
                    .get(&load_name)
                    .unwrap_or_else(|| panic!("missing input slot for load '{load_name}'"));
                e.line(&format!(
                    "outputs[{slot}] = chelis_contiguous(inputs[{input_idx}]);"
                ));
            } else {
                // Allocate host tensor and transfer from device
                let ty = &dag.get(output.id).unwrap().output_type;
                let ndim = Self::ndim(ty);
                let dtype = Self::dtype_macro(ty);
                let shape = if ndim == 0 {
                    "NULL".to_string()
                } else {
                    format!("chelis_output_shape_{slot}")
                };
                e.line(&format!(
                    "outputs[{slot}] = chelis_alloc({ndim}, {shape}, {dtype});"
                ));
                e.line(&format!("chelis_tensor_write *output_guard_{slot} = chelis_tensor_begin_write(outputs[{slot}]);"));
                e.line(&format!(
                    "chelis_device_tensor_copy_to_host(output_guard_{slot}, o_t{id});"
                ));
                e.line(&format!("chelis_tensor_end_write(output_guard_{slot});"));
            }
        }

        e.line("");

        // Cleanup: free GPU tensors (skip reduction-inlined FusedElems — never allocated)
        let dropped_sources = dag
            .actions()
            .filter_map(|action| match action {
                VerifiedDagAction::OwnedDrop { source, .. } => Some(source),
                _ => None,
            })
            .collect::<Vec<_>>();
        e.line("CHELIS_HIP_CHECK(hipDeviceSynchronize());");
        for name in &kernel_names {
            e.line(&format!("CHELIS_HIP_CHECK(hipModuleUnload(mod_{name}));"));
        }
        let cleanup = e.plan.emit_cleanup_with_drops(&dropped_sources);
        for line in cleanup {
            e.lines.push(line);
        }

        e.indent = 0;
        e.line("}");
        e.line("");
        e.emit_device_entrypoint(dag, func_name, &output_specs, &input_slots, &kernel_names)?;
        let mut formula = e.plan.peak_device_bytes_formula();
        if e.extra_peak_device_bytes_estimate > 0 {
            formula = if formula == "0" {
                e.extra_peak_device_bytes_estimate.to_string()
            } else {
                format!("{formula} + {}", e.extra_peak_device_bytes_estimate)
            };
        }
        let estimate = e
            .plan
            .peak_device_bytes_estimate()
            .map(|bytes| bytes + e.extra_peak_device_bytes_estimate);
        let breakdown = PeakDeviceBytesBreakdown {
            formula,
            estimate,
            terms: e.plan.peak_device_bytes_terms(),
            extra_bytes: e.extra_peak_device_bytes_estimate,
        };
        Ok((e.lines.join("\n"), breakdown))
    }

    fn emit_device_entrypoint(
        &mut self,
        dag: VerifiedDagView<'_>,
        func_name: &str,
        output_specs: &[OutputSpec],
        input_slots: &chelis_unord::UnordMap<String, usize>,
        kernel_names: &[String],
    ) -> Result<(), Unsupported> {
        let expected_inputs = input_slots.len();
        let expected_outputs = output_specs.len();

        self.line(&format!(
            "extern \"C\" void {func_name}_device(const chelis_device_tensor_owner *const *inputs, int32_t n_in, chelis_device_tensor_owner **outputs, int32_t n_out) {{"
        ));
        self.indent = 1;

        // Format-string-context sanitization for `func_name` per
        // spec/upstream-bugs/producer-string-sanitization.md.
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);

        self.line(&format!("if (n_in != {expected_inputs}) {{"));
        self.indent += 1;
        self.line(&format!(
            "fprintf(stderr, \"{func_name_fmt}_device: expected %d inputs, got %d\\n\", {expected_inputs}, n_in);"
        ));
        self.line("abort();");
        self.indent -= 1;
        self.line("}");

        self.line(&format!("if (n_out != {expected_outputs}) {{"));
        self.indent += 1;
        self.line(&format!(
            "fprintf(stderr, \"{func_name_fmt}_device: expected %d outputs, got %d\\n\", {expected_outputs}, n_out);"
        ));
        self.line("abort();");
        self.indent -= 1;
        self.line("}");
        self.line("");

        self.line("if ((n_in > 0 && inputs == NULL) || (n_out > 0 && outputs == NULL)) chelis_numeric_trap(\"numeric trap: domain in entry at int64\");");
        self.line("int current_device = 0;");
        self.line("CHELIS_HIP_CHECK(hipGetDevice(&current_device));");
        self.line("for (int32_t slot = 0; slot < n_in; ++slot) if (chelis_device_tensor_device(inputs[slot]) != current_device) chelis_numeric_trap(\"numeric trap: domain in device_entry at int64\");");
        self.emit_input_shape_preamble_device(dag, input_slots, func_name);
        self.emit_reshape_count_preflight(dag);
        self.line("");

        for name in kernel_names {
            self.line(&format!(
                "hipModule_t mod_{name} = chelis_compile_kernel({name}_src, \"{name}\");"
            ));
        }
        if !kernel_names.is_empty() {
            self.line("");
        }

        self.device_entrypoint_mode = true;
        self.emit_device_slot_allocations(dag);
        if !self.plan.slots().is_empty() {
            self.line("");
        }

        for node in dag.nodes() {
            if self.reduction_inlined.contains(&node.id.0) {
                continue;
            }
            if let RiscOp::Load { name } = &node.op {
                let input_idx = *input_slots
                    .get(name.as_str())
                    .unwrap_or_else(|| panic!("missing input slot for load '{name}'"));
                self.emit_span_comments(node);
                self.emit_load_device(node.id.0, input_idx, &node.output_type);
            } else {
                self.emit_span_comments(node);
                self.emit_node(node, dag)?;
            }
        }

        self.line("");
        for (slot, output) in output_specs.iter().enumerate() {
            let id = output.id.0;
            let line = match &dag.get(output.id).unwrap().op {
                RiscOp::Load { name } => {
                    let input_idx = input_slots
                        .get(name.as_str())
                        .unwrap_or_else(|| panic!("missing input slot for load '{name}'"));
                    format!("outputs[{slot}] = chelis_device_tensor_clone(inputs[{input_idx}]);")
                }
                _ => format!("outputs[{slot}] = chelis_device_tensor_clone(o_t{id});"),
            };
            self.line(&line);
        }

        self.line("");
        let dropped_sources = dag
            .actions()
            .filter_map(|action| match action {
                VerifiedDagAction::OwnedDrop { source, .. } => Some(source),
                _ => None,
            })
            .collect::<Vec<_>>();
        self.line("CHELIS_HIP_CHECK(hipDeviceSynchronize());");
        for name in kernel_names {
            self.line(&format!("CHELIS_HIP_CHECK(hipModuleUnload(mod_{name}));"));
        }
        let cleanup = self.plan.emit_cleanup_with_drops(&dropped_sources);
        for line in cleanup {
            self.lines.push(line);
        }
        self.device_entrypoint_mode = false;
        self.indent = 0;
        self.line("}");
        Ok(())
    }

    // ------------------------------------------------------------------
    // Kernel collection (first pass)
    // ------------------------------------------------------------------

    fn collect_kernels(&mut self, dag: VerifiedDagView<'_>) -> Result<(), Unsupported> {
        let mut seen = chelis_unord::UnordSet::new();
        for node in dag.nodes() {
            // Skip reduction-inlined FusedElem nodes (they become part of the
            // reduction kernel).
            if self.reduction_inlined.contains(&node.id.0) {
                continue;
            }
            match &node.op {
                // WS-A4: `accumulator` is read inside
                // `reduction_kernel_sources` (it picks the
                // dtype-specialized kernel template). The `..` here
                // would normally trip the destructure-`..` rule, but
                // the dispatch happens inside the called helper which
                // explicitly re-matches and binds the field.
                RiscOp::Sum { axis, .. } | RiscOp::MaxReduce { axis } => {
                    let sources = self.reduction_kernel_sources(node, dag, *axis)?;
                    for (name, source) in sources {
                        if seen.insert(name.clone()) {
                            // Reduction kernels named `kernel_fused_<sum|maxred>_<id>`
                            // are per-node (id encodes the originating DagNode),
                            // so prepend that node's spans inside the kernel
                            // source. Shared reduction kernels (e.g.
                            // `kernel_sum_ax0`) are launched from multiple
                            // nodes — host-side launch comments cover those.
                            let source = if Self::is_per_node_kernel_name(&name) {
                                Self::prepend_span_comments_to_kernel_source(node, source)
                            } else {
                                source
                            };
                            self.kernel_sources.push((name, source));
                        }
                    }
                    continue;
                }
                RiscOp::MinReduce { .. }
                | RiscOp::ProdReduce { .. }
                | RiscOp::Argmax { .. }
                | RiscOp::Argmin { .. } => {
                    // WS-A2 lift: these reductions now have HIP kernels
                    // for f32 and f64. The kernel sources are emitted
                    // here, then the standard `emit_node` dispatch in
                    // `emit_dag` launches them via the same machinery as
                    // the existing Sum/MaxReduce kernels.
                    let sources = self.extra_reduction_kernel_sources(node, dag)?;
                    for (name, source) in sources {
                        if seen.insert(name.clone()) {
                            self.kernel_sources.push((name, source));
                        }
                    }
                    continue;
                }
                _ => {}
            }
            if matches!(
                node.op,
                RiscOp::ScatterAdd { .. } | RiscOp::Scatter { .. } | RiscOp::ScatterElements { .. }
            ) {
                let name = format!("kernel_materialize_{}", node.id.0);
                let width = node
                    .output_type
                    .precision
                    .runtime_dtype()
                    .expect("verified representation")
                    .byte_width();
                self.kernel_sources.push((
                    name.clone(),
                    kernels::reshape_copy(self.kernel_rank, &name, width),
                ));
            }
            let name = self.kernel_name_for_op(&node.op, node, dag)?;
            if let Some(name) = name
                && seen.insert(name.clone())
            {
                let source = self.kernel_source_for_op(&name, &node.op, node, dag)?;
                // Per-node kernels (FusedElem `kernel_fused_<id>`, fused
                // reductions) get this node's spans embedded inside their
                // source string so the audit chain survives into the
                // runtime-compiled kernel. Shared kernels (kernel_neg,
                // kernel_add, …) are launched from multiple DAG nodes so
                // there is no single canonical span — host-side launch
                // comments are the audit anchor for those.
                let source = if Self::is_per_node_kernel_name(&name) {
                    Self::prepend_span_comments_to_kernel_source(node, source)
                } else {
                    source
                };
                self.kernel_sources.push((name, source));
            }
        }
        Ok(())
    }

    /// True when a kernel name is unique to a single DagNode (i.e. its
    /// suffix encodes a node id). Used to decide whether prepending span
    /// comments inside the kernel source is unambiguous.
    fn is_per_node_kernel_name(name: &str) -> bool {
        // FusedElem: kernel_fused_<id>
        // Fused reductions: kernel_fused_sum_<id>, kernel_fused_maxred_<id>
        name.starts_with("kernel_fused_")
    }

    fn input_types(dag: VerifiedDagView<'_>) -> chelis_unord::UnordMap<String, TensorType> {
        let mut seen = chelis_unord::UnordMap::<String, TensorType>::new();
        for node in dag.nodes() {
            if let RiscOp::Load { name } = &node.op {
                seen.entry(name.as_str().to_string())
                    .or_insert_with(|| node.output_type.clone());
            }
        }
        seen
    }

    fn emit_input_shape_preamble(
        &mut self,
        dag: VerifiedDagView<'_>,
        input_slots: &chelis_unord::UnordMap<String, usize>,
        func_name: &str,
        output_specs: &[OutputSpec],
    ) {
        // Iteration order over `input_types` (a UnordMap) must be
        // deterministic so the emitted host code is byte-identical
        // across runs. Sort by label; lookups are by name and emitted
        // lines are independent per label.
        // See spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md.
        self.line("if ((n_in > 0 && inputs == NULL) || (n_out > 0 && outputs == NULL)) chelis_numeric_trap(\"numeric trap: domain in entry at int64\");");
        let input_types = Self::input_types(dag);
        let sorted_labels = input_types.to_sorted();
        // Producer-supplied `func_name` flows into format-string context;
        // sanitize per spec/upstream-bugs/producer-string-sanitization.md.
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);
        for (label, _) in sorted_labels {
            let ty = &input_types[label];
            let slot = input_slots[label];
            // `label` is `LoadStoreName::as_str()` (validated); route
            // through the format-string sanitizer to lock the
            // architectural pattern.
            let label_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(label);
            self.line(&format!("if (inputs[{slot}] == NULL) {{"));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"{func_name_fmt}: input `{label_fmt}` at slot {slot} is NULL\\n\");"
            ));
            self.line("abort();");
            self.indent -= 1;
            self.line("}");
            self.line(&format!("if (chelis_tensor_dtype(inputs[{slot}]) != {}) chelis_numeric_trap(\"numeric trap: domain in load at int64\");", Self::dtype_macro(ty)));
            self.line(&format!(
                "if (chelis_tensor_rank(inputs[{slot}]) != {}) {{",
                Self::ndim(ty)
            ));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"{func_name_fmt}: input `{label_fmt}` expected rank {}, got %d\\n\", chelis_tensor_rank(inputs[{slot}]));",
                Self::ndim(ty)
            ));
            self.line("abort();");
            self.indent -= 1;
            self.line("}");
            for (axis, dim) in ty.dims.iter().enumerate() {
                if let Some(expected) = Self::known_dim_size(dim) {
                    self.line(&format!(
                        "if (chelis_tensor_shape(inputs[{slot}], {axis}) != {expected}) {{"
                    ));
                    self.indent += 1;
                    self.line(&format!(
                        "fprintf(stderr, \"{func_name_fmt}: input `{label_fmt}` axis {axis} expected {expected}, got %lld\\n\", (long long)chelis_tensor_shape(inputs[{slot}], {axis}));"
                    ));
                    self.line("abort();");
                    self.indent -= 1;
                    self.line("}");
                }
            }
        }

        for binding in dag.symbolic_bindings_interface() {
            let (canonical_label, canonical_axis) = require_load_source(&binding.canonical);
            let canonical_slot = input_slots[canonical_label];
            // `binding.name` flows into format-string context; sanitize.
            let binding_name_fmt =
                chelis_ir::span_sanitize::sanitize_for_format_string(&binding.name);
            self.line(&format!(
                "int64_t {} = chelis_tensor_shape(inputs[{canonical_slot}], {canonical_axis});",
                binding.name
            ));
            for occurrence in &binding.others {
                let (occ_label, occ_axis) = require_load_source(occurrence);
                let slot = input_slots[occ_label];
                let occ_label_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(occ_label);
                self.line(&format!(
                    "if (chelis_tensor_shape(inputs[{slot}], {occ_axis}) != {}) {{",
                    binding.name
                ));
                self.indent += 1;
                self.line(&format!(
                    "fprintf(stderr, \"{func_name_fmt}: symbolic dim `{binding_name_fmt}` mismatch: {occ_label_fmt}[{occ_axis}]=%lld but {binding_name_fmt}=%lld\\n\", (long long)chelis_tensor_shape(inputs[{slot}], {occ_axis}), (long long){});",
                    binding.name
                ));
                self.line("abort();");
                self.indent -= 1;
                self.line("}");
            }
        }

        // chelis#1277 S2b: the same-rank `expand`'s unit-extent claim, on the
        // host prologue. One derivation, three lanes: this reads the same
        // `derive_unit_extent_claims` and the same `member_load_axis` the C
        // emitter and the evaluator read.
        //
        // It renders [04-NUM-9] through `chelis_numeric_trap`, which
        // `chelis_hip_runtime.h` reaches by including `chelis_runtime.h`. The
        // `abort()` above is the LEGACY rendering for this lane's `Name`
        // bindings and is chelis#1112's to move; this guard does not adopt it,
        // because `spec/05-risc-primitives.md` section 2.4.1 sends a failed
        // claim to a `Domain` trap "placed and rendered per
        // `spec/04-type-system.md` section 4.7 and [04-NUM-9]".
        for (load, read_axis) in dag.entry_unit_extent_reads() {
            let Some(RiscOp::Load { name: label }) = dag.get(load).map(|node| &node.op) else {
                continue;
            };
            let Some(&slot) = input_slots.get(label.as_str()) else {
                continue;
            };
            let label_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(label.as_str());
            self.line(&format!(
                "if (chelis_tensor_shape(inputs[{slot}], {read_axis}) != 1) {{"
            ));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"extent `1`: claimed = 1, {label_fmt} axis {read_axis} = %lld\\n\", (long long)chelis_tensor_shape(inputs[{slot}], {read_axis}));"
            ));
            self.line("chelis_numeric_trap(\"numeric trap: domain in load at int64\");");
            self.indent -= 1;
            self.line("}");
        }

        // Host allocation consumes the published runtime ABI's int64_t shape
        // carrier. Keep these declarations in the already-classified shape
        // preamble; device allocations below deliberately retain int[].
        for (slot, output) in output_specs.iter().enumerate() {
            let node = dag
                .get(output.id)
                .expect("every output spec must reference a DAG node");
            if matches!(node.op, RiscOp::Load { .. }) {
                continue;
            }
            let dims: Vec<String> = node
                .output_type
                .dims
                .iter()
                .map(Self::emit_dim_info)
                .collect();
            let shape = if dims.is_empty() {
                "1".to_string()
            } else {
                dims.join(", ")
            };
            self.line(&format!(
                "int64_t chelis_output_shape_{slot}[{}] = {{ {shape} }};",
                Self::ndim(&node.output_type).max(1)
            ));
        }
    }

    fn emit_input_shape_preamble_device(
        &mut self,
        dag: VerifiedDagView<'_>,
        input_slots: &chelis_unord::UnordMap<String, usize>,
        func_name: &str,
    ) {
        // Iteration order over `input_types` (a UnordMap) must be
        // deterministic so the emitted device-side code is byte-
        // identical across runs. Sort by label; lookups are by name
        // and emitted lines are independent per label.
        // See spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md.
        self.line("if ((n_in > 0 && inputs == NULL) || (n_out > 0 && outputs == NULL)) chelis_numeric_trap(\"numeric trap: domain in entry at int64\");");
        let input_types = Self::input_types(dag);
        let sorted_labels = input_types.to_sorted();
        // Format-string-context sanitization for producer-supplied
        // strings per spec/upstream-bugs/producer-string-sanitization.md.
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);
        for (label, _) in sorted_labels {
            let ty = &input_types[label];
            let slot = input_slots[label];
            let label_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(label);
            self.line(&format!("if (inputs[{slot}] == NULL) {{"));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"{func_name_fmt}_device: input `{label_fmt}` at slot {slot} is NULL\\n\");"
            ));
            self.line("abort();");
            self.indent -= 1;
            self.line("}");
            self.line(&format!("const chelis_gpu_tensor *input_view_{slot} = chelis_device_tensor_view(inputs[{slot}]);"));
            self.line(&format!("if (input_view_{slot}->dtype != {}) chelis_numeric_trap(\"numeric trap: domain in load at int64\");", Self::dtype_macro(ty)));
            self.line(&format!(
                "if (input_view_{slot}->rank != {}) {{",
                Self::ndim(ty)
            ));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"{func_name_fmt}_device: input `{label_fmt}` expected rank {}, got %d\\n\", input_view_{slot}->rank);",
                Self::ndim(ty)
            ));
            self.line("abort();");
            self.indent -= 1;
            self.line("}");
            for (axis, dim) in ty.dims.iter().enumerate() {
                if let Some(expected) = Self::known_dim_size(dim) {
                    self.line(&format!(
                        "if (input_view_{slot}->shape[{axis}] != {expected}) {{"
                    ));
                    self.indent += 1;
                    self.line(&format!(
                        "fprintf(stderr, \"{func_name_fmt}_device: input `{label_fmt}` axis {axis} expected {expected}, got %lld\\n\", (long long)input_view_{slot}->shape[{axis}]);"
                    ));
                    self.line("abort();");
                    self.indent -= 1;
                    self.line("}");
                }
            }
        }

        for binding in dag.symbolic_bindings_interface() {
            let (canonical_label, canonical_axis) = require_load_source(&binding.canonical);
            let canonical_slot = input_slots[canonical_label];
            let binding_name_fmt =
                chelis_ir::span_sanitize::sanitize_for_format_string(&binding.name);
            self.line(&format!(
                "int64_t {} = input_view_{canonical_slot}->shape[{canonical_axis}];",
                binding.name
            ));
            for occurrence in &binding.others {
                let (occ_label, occ_axis) = require_load_source(occurrence);
                let slot = input_slots[occ_label];
                let occ_label_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(occ_label);
                self.line(&format!(
                    "if (input_view_{slot}->shape[{occ_axis}] != {}) {{",
                    binding.name
                ));
                self.indent += 1;
                self.line(&format!(
                    "fprintf(stderr, \"{func_name_fmt}_device: symbolic dim `{binding_name_fmt}` mismatch: {occ_label_fmt}[{occ_axis}]= %lld but {binding_name_fmt}=%lld\\n\", (long long)input_view_{slot}->shape[{occ_axis}], (long long){});",
                    binding.name
                ));
                self.line("abort();");
                self.indent -= 1;
                self.line("}");
            }
        }
    }

    fn reduction_kernel_sources(
        &self,
        node: &DagNode,
        dag: VerifiedDagView<'_>,
        axis: usize,
    ) -> Result<Vec<(String, String)>, Unsupported> {
        // WS-A4: bind `accumulator` instead of `..`. The kernel-name and
        // kernel-source paths agree on the (source, accumulator) tuple
        // so the i8/i16 → i32 promoted path produces a uniquely-named
        // kernel source instead of colliding with the f32 default.
        let (kind, accumulator) = match node.op {
            RiscOp::Sum { accumulator, .. } => (kernels::ReduceKind::Sum, Some(accumulator)),
            RiscOp::MaxReduce { .. } => (kernels::ReduceKind::Max, None),
            _ => unreachable!("reduction_kernel_sources called on non-reduction"),
        };
        let input_id = node.inputs[0];
        if self.reduction_inlined.contains(&input_id.0) {
            let fused_node = dag.get(input_id).unwrap();
            let elem = Self::elem_kind(&fused_node.output_type)?;
            let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
            let name = Self::fused_reduction_kernel_name(node.id.0, kind);
            let source =
                kernels::reduce_fused(self.kernel_rank, &name, axis, steps, n_ext, kind, elem);
            return Ok(vec![(name, source)]);
        }

        if matches!(kind, kernels::ReduceKind::Sum)
            && let Some(matmul) = blas::detect_matmul_pattern(dag, node.id)
            && Self::supports_static_hipblas_matmul(dag, &matmul, &node.output_type)
        {
            return Ok(Vec::new());
        }

        // For Sum, the result precision IS the accumulator (spec §5.7.1)
        // so we read the operand precision separately. For MaxReduce, the
        // accumulator and result both match the operand precision. The
        // f32/f64 paths use the WS-A2 dtype-parameterized
        // `reduction_kernel_name(kind, axis, ElemKind)` naming so the
        // pre-WS-A4 kernel-name convention is preserved; the WS-A4
        // i8/i16 → i32 promoted path uses `reduction_kernel_name_typed`
        // so its source/accumulator suffix encodes the promotion.
        let operand_ty = &dag.get(input_id).unwrap().output_type;
        let src_prec = operand_ty.precision;
        let acc_prec = node.output_type.precision;
        let (name, source) = match kind {
            kernels::ReduceKind::Sum => {
                let acc = accumulator.unwrap_or(acc_prec);
                if matches!(src_prec, Prim::Int8 | Prim::Int16) && acc == Prim::Int32 {
                    // WS-A4 i8/i16 → i32 promoted-accumulator path.
                    let name = Self::reduction_kernel_name_typed(kind, axis, src_prec, acc);
                    let source = kernels::reduce_sum_promoted(
                        self.kernel_rank,
                        &name,
                        axis,
                        Self::dtype_c_type(src_prec),
                        "int32_t",
                    );
                    (name, source)
                } else {
                    // WS-A2 f32/f64 path via ElemKind-driven naming +
                    // dispatch. Panics in `elem_kind` if a non-float
                    // dtype slips through here, which is the loud
                    // failure mode we want for unwired precisions.
                    let operand_kind = Self::elem_kind(operand_ty)?;
                    let acc_kind = Self::elem_kind(&node.output_type)?;
                    let name = Self::reduction_kernel_name(kind, axis, acc_kind);
                    let source =
                        kernels::reduce_sum(self.kernel_rank, &name, axis, operand_kind, acc_kind);
                    (name, source)
                }
            }
            kernels::ReduceKind::Max => {
                let acc_kind = Self::elem_kind(&node.output_type)?;
                let name = Self::reduction_kernel_name(kind, axis, acc_kind);
                let source = kernels::reduce_max(self.kernel_rank, &name, axis, acc_kind);
                (name, source)
            }
        };
        Ok(vec![(name, source)])
    }

    fn extra_reduction_kernel_sources(
        &self,
        node: &DagNode,
        dag: VerifiedDagView<'_>,
    ) -> Result<Vec<(String, String)>, Unsupported> {
        // Kernel sources for the four reductions previously deferred to
        // the C backend: Min / Prod / Argmax / Argmin. Argmax/Argmin emit
        // an i64 result tensor; Min/Prod emit an in-precision result.
        // For all four, kernel naming + body are driven by the OPERAND
        // precision, which lives on the input tensor (Argmax/Argmin's
        // output_type is `int64` and would otherwise tip elem_kind into
        // its panic arm).
        let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
        let elem = Self::elem_kind(input_ty)?;
        Ok(match &node.op {
            RiscOp::MinReduce { axis } => {
                let name = Self::extra_reduction_kernel_name("min", *axis, elem);
                let src = kernels::reduce_min(self.kernel_rank, &name, *axis, elem);
                vec![(name, src)]
            }
            RiscOp::ProdReduce { axis } => {
                let name = Self::extra_reduction_kernel_name("prod", *axis, elem);
                let src = kernels::reduce_prod(self.kernel_rank, &name, *axis, elem);
                vec![(name, src)]
            }
            RiscOp::Argmax { axis } => {
                let name = Self::extra_reduction_kernel_name("argmax", *axis, elem);
                let src = kernels::reduce_argmax(self.kernel_rank, &name, *axis, elem);
                vec![(name, src)]
            }
            RiscOp::Argmin { axis } => {
                let name = Self::extra_reduction_kernel_name("argmin", *axis, elem);
                let src = kernels::reduce_argmin(self.kernel_rank, &name, *axis, elem);
                vec![(name, src)]
            }
            _ => unreachable!("extra_reduction_kernel_sources expected Min/Prod/Argmax/Argmin"),
        })
    }

    fn kernel_name_for_op(
        &self,
        op: &RiscOp,
        node: &DagNode,
        dag: VerifiedDagView<'_>,
    ) -> Result<Option<String>, Unsupported> {
        // WS-A2 + WS-A4: kernel-name dispatch must agree with the
        // kernel-source emission in `kernel_source_for_op`. For binary
        // elementwise ops the operand precision is unambiguous
        // (verifier-enforced same-precision per spec §5.4) and drives
        // the kernel specialization. For unary ops and reductions the
        // f32/f64 split is owned by `ElemKind::suffix()`.
        let operand_prec = || dag.get(node.inputs[0]).unwrap().output_type.precision;
        let kind_for_node = |n: &DagNode| -> Result<kernels::ElemKind, Unsupported> {
            Self::elem_kind(&n.output_type)
        };
        Ok(match op {
            // WS-A4: Add / Mul use the dtype-suffixed convention so f32
            // stays unsuffixed (`kernel_add`) and non-f32 dtypes pick
            // up an explicit suffix (`kernel_add_f64`, `kernel_add_i8`).
            RiscOp::Add => Some(format!(
                "kernel_add{}",
                Self::dtype_kernel_suffix(operand_prec())
            )),
            RiscOp::Sub => Some(format!("kernel_sub_{}", kind_for_node(node)?.suffix())),
            RiscOp::Mul => Some(format!(
                "kernel_mul{}",
                Self::dtype_kernel_suffix(operand_prec())
            )),
            RiscOp::Div => Some(format!(
                "kernel_div{}",
                Self::dtype_kernel_suffix(operand_prec())
            )),
            // chelis#178: floor / truncating integer division. Both
            // dispatch on operand precision (the dtype suffix) so the
            // kernel name matches `kernel_source_for_op`.
            RiscOp::FloorDiv => Some(format!(
                "kernel_floor_div{}",
                Self::dtype_kernel_suffix(operand_prec())
            )),
            RiscOp::Mod => return Err(Self::remainder_unsupported(node)),
            RiscOp::TruncDiv => Some(format!(
                "kernel_trunc_div{}",
                Self::dtype_kernel_suffix(operand_prec())
            )),
            // WS-A2: float-only kernel templates remain `_<f32|f64>`-suffixed.
            RiscOp::MaxElem => Some(format!(
                "kernel_max_elem{}",
                Self::dtype_kernel_suffix(operand_prec())
            )),
            RiscOp::MinElem => Some(format!(
                "kernel_min_elem{}",
                Self::dtype_kernel_suffix(operand_prec())
            )),
            RiscOp::ExtremaAdjoint { kind, operand } => {
                let extrema = match kind {
                    ExtremaKind::Max => "max",
                    ExtremaKind::Min => "min",
                };
                let selected = match operand {
                    ExtremaOperand::Left => "left",
                    ExtremaOperand::Right => "right",
                };
                Some(format!(
                    "kernel_{extrema}_adjoint_{selected}_{}",
                    kind_for_node(node)?.suffix()
                ))
            }
            RiscOp::Relu => Some(format!(
                "kernel_relu{}",
                Self::dtype_kernel_suffix(operand_prec())
            )),
            RiscOp::ReluAdjoint => Some(format!(
                "kernel_relu_adjoint{}",
                Self::dtype_kernel_suffix(operand_prec())
            )),
            RiscOp::CmpLt => {
                // CmpLt has bool output but operand-precision storage;
                // dispatch on the operand precision so the kernel name
                // matches the kernel source emitted in
                // `kernel_source_for_op`.
                let operand_kind = Self::elem_kind(&dag.get(node.inputs[0]).unwrap().output_type)?;
                Some(format!("kernel_cmplt_{}", operand_kind.suffix()))
            }
            RiscOp::Neg => Some(format!("kernel_neg_{}", kind_for_node(node)?.suffix())),
            RiscOp::Recip => Some(format!("kernel_recip_{}", kind_for_node(node)?.suffix())),
            RiscOp::Exp => Some(format!("kernel_exp_{}", kind_for_node(node)?.suffix())),
            RiscOp::Log => Some(format!("kernel_log_{}", kind_for_node(node)?.suffix())),
            RiscOp::Sin => Some(format!("kernel_sin_{}", kind_for_node(node)?.suffix())),
            RiscOp::Sqrt => Some(format!("kernel_sqrt_{}", kind_for_node(node)?.suffix())),
            RiscOp::Cos => Some(format!("kernel_cos_{}", kind_for_node(node)?.suffix())),
            RiscOp::Tan => Some(format!("kernel_tan_{}", kind_for_node(node)?.suffix())),
            RiscOp::Atan => Some(format!("kernel_atan_{}", kind_for_node(node)?.suffix())),
            RiscOp::Abs => Some(format!("kernel_abs_{}", kind_for_node(node)?.suffix())),
            RiscOp::Floor => Some(format!("kernel_floor_{}", kind_for_node(node)?.suffix())),
            RiscOp::Ceil => Some(format!("kernel_ceil_{}", kind_for_node(node)?.suffix())),
            RiscOp::Round => Some(format!("kernel_round_{}", kind_for_node(node)?.suffix())),
            RiscOp::UniformLike { .. } => Some(format!(
                "kernel_uniform_like_{}",
                kind_for_node(node)?.suffix()
            )),
            RiscOp::Dropout { .. } | RiscOp::Drop => None,
            RiscOp::Copy => Some(Self::cast_kernel_name(node, dag)?),
            // WS-A4: bind `accumulator` instead of `..` per the
            // destructure-`..` memory rule. The kernel name encodes
            // both source and accumulator dtype when they differ
            // (i8/i16 → i32 path), so the kernel-source emission can
            // dispatch unambiguously from the name.
            RiscOp::Sum { axis, accumulator } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    Some(Self::fused_reduction_kernel_name(
                        node.id.0,
                        kernels::ReduceKind::Sum,
                    ))
                } else {
                    // Sum naming follows the same dispatch rule as
                    // `reduction_kernel_sources`: f32/f64 use the
                    // `ElemKind`-suffixed legacy name; i8/i16 → i32 uses
                    // the WS-A4 typed naming so the i8/i16 kernel is
                    // distinguishable from any future i32→i32 case.
                    if matches!(operand_prec(), Prim::Int8 | Prim::Int16)
                        && *accumulator == Prim::Int32
                    {
                        Some(Self::reduction_kernel_name_typed(
                            kernels::ReduceKind::Sum,
                            *axis,
                            operand_prec(),
                            *accumulator,
                        ))
                    } else {
                        Some(Self::reduction_kernel_name(
                            kernels::ReduceKind::Sum,
                            *axis,
                            Self::elem_kind(&node.output_type)?,
                        ))
                    }
                }
            }
            RiscOp::MaxReduce { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    Some(Self::fused_reduction_kernel_name(
                        node.id.0,
                        kernels::ReduceKind::Max,
                    ))
                } else {
                    Some(Self::reduction_kernel_name(
                        kernels::ReduceKind::Max,
                        *axis,
                        Self::elem_kind(&node.output_type)?,
                    ))
                }
            }
            RiscOp::MinReduce { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                Some(Self::extra_reduction_kernel_name(
                    "min",
                    *axis,
                    Self::elem_kind(input_ty)?,
                ))
            }
            RiscOp::ProdReduce { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                Some(Self::extra_reduction_kernel_name(
                    "prod",
                    *axis,
                    Self::elem_kind(input_ty)?,
                ))
            }
            RiscOp::Argmax { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                Some(Self::extra_reduction_kernel_name(
                    "argmax",
                    *axis,
                    Self::elem_kind(input_ty)?,
                ))
            }
            RiscOp::Argmin { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                Some(Self::extra_reduction_kernel_name(
                    "argmin",
                    *axis,
                    Self::elem_kind(input_ty)?,
                ))
            }
            // `reduce_window_*` HIP codegen is excluded by the
            // target contract ([05-RWIN-2] / spec §2.3.1). The
            // C backend is canonical; returning `None` here means no
            // kernel name is registered, and the launch-emit arm below
            // panics via `todo!` if a `ReduceWindow` node ever reaches
            // codegen on the HIP target.
            RiscOp::ReduceWindow { .. } => None,
            // `reduce_window_*` adjoint: excluded alongside the forward op
            // under [05-RWIN-2]; launch-emit panics via `todo!`.
            RiscOp::ReduceWindowGrad { .. } => None,
            RiscOp::OneHot { .. } => None,
            // The runtime `shape` value read is excluded by [05-SHAPE-1];
            // `reject_unsupported_hip_ops` rejects it cleanly before
            // codegen, so no kernel name is registered. The launch-emit arm
            // below is a defensive `todo!` if one ever reaches codegen.
            RiscOp::Shape { .. }
            | RiscOp::ExtentWitness { .. }
            | RiscOp::CheckedReshapeExtent { .. }
            | RiscOp::CheckedUnitAxis { .. } => None,
            RiscOp::Const { .. } => Some(format!("kernel_fill_{}", kind_for_node(node)?.suffix())),
            RiscOp::ConstTensor { .. } => {
                Some(format!("kernel_fill_{}", kind_for_node(node)?.suffix()))
            }
            RiscOp::Reshape { .. } => Some(format!(
                "kernel_reshape_{}",
                Self::dtype_macro(&node.output_type)
            )),
            RiscOp::Realize => Some(format!(
                "kernel_realize_{}",
                Self::dtype_macro(&node.output_type)
            )),
            RiscOp::Cast { .. } => Some(Self::cast_kernel_name(node, dag)?),
            // Both `reject_unsupported_hip_ops` copies (chelis-cli and
            // chelis-compiler-api) gate this out before codegen; the
            // emitter arms below are the backstop if a future caller
            // reaches the backend without passing a gate.
            RiscOp::CastTrunc { .. } => None,
            RiscOp::Count { .. } => None,
            // `pad` / `shrink` materialize a fresh buffer via a typed
            // per-output-element kernel (see `kernels::pad_typed` /
            // `kernels::shrink_typed`); the kernel name carries the output
            // dtype suffix so one kernel serves every pad/shrink node of
            // that dtype regardless of rank.
            RiscOp::Pad { .. } => Some(format!(
                "kernel_pad{}",
                Self::dtype_kernel_suffix(node.output_type.precision)
            )),
            RiscOp::Shrink { .. } => Some(format!(
                "kernel_shrink{}",
                Self::dtype_kernel_suffix(node.output_type.precision)
            )),
            // Pure-metadata movement ops and Load/Store are not kernels
            RiscOp::Load { .. }
            | RiscOp::Store { .. }
            | RiscOp::Permute { .. }
            | RiscOp::Expand { .. }
            | RiscOp::Stride { .. }
            | RiscOp::BlasMatmul { .. } => None,
            RiscOp::Gather { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                let elem = Self::elem_kind(&node.output_type)?;
                Some(match indices_ty.precision {
                    Prim::Int32 => format!("kernel_gather_i32_{}", elem.suffix()),
                    Prim::Int64 => format!("kernel_gather_i64_{}", elem.suffix()),
                    _ => "kernel_gather_invalid".into(),
                })
            }
            RiscOp::ScatterAdd { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                let elem = Self::elem_kind(&node.output_type)?;
                Some(match indices_ty.precision {
                    Prim::Int32 => format!("kernel_scatter_add_i32_{}", elem.suffix()),
                    Prim::Int64 => format!("kernel_scatter_add_i64_{}", elem.suffix()),
                    _ => "kernel_scatter_add_invalid".into(),
                })
            }
            RiscOp::Scatter { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                Some(match indices_ty.precision {
                    Prim::Int32 => "kernel_scatter_replace_i32".into(),
                    Prim::Int64 => "kernel_scatter_replace_i64".into(),
                    _ => "kernel_scatter_replace_invalid".into(),
                })
            }
            RiscOp::ScatterElements { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                Some(match indices_ty.precision {
                    Prim::Int32 => "kernel_scatter_elements_i32".into(),
                    Prim::Int64 => "kernel_scatter_elements_i64".into(),
                    _ => "kernel_scatter_elements_invalid".into(),
                })
            }
            RiscOp::FusedElem { .. } => Some(format!("kernel_fused_{}", node.id.0)),
        })
    }

    /// Build the cast kernel name. When src/dst precision agree, this is
    /// the in-precision identity kernel; when they differ, it's the
    /// cross-precision conversion kernel.
    fn cast_kernel_name(node: &DagNode, dag: VerifiedDagView<'_>) -> Result<String, Unsupported> {
        let (src_kind, dst_kind) = Self::cast_elem_kinds(node, dag)?;
        Ok(if src_kind == dst_kind {
            format!("kernel_cast_{}", dst_kind.suffix())
        } else {
            format!("kernel_cast_{}_to_{}", src_kind.suffix(), dst_kind.suffix())
        })
    }

    fn kernel_source_for_op(
        &self,
        name: &str,
        op: &RiscOp,
        node: &DagNode,
        dag: VerifiedDagView<'_>,
    ) -> Result<String, Unsupported> {
        // CmpLt's output type is `bool` (semantically) but the kernel
        // writes 1.0/0.0 of operand precision to the GPU buffer. Use the
        // operand precision for kernel emission; the rest of the
        // floating ops have output_type == operand_type so the more
        // common path uses output_type below. For Add/Mul/Sum the
        // output may be a non-float dtype (i32 acc for i8/i16 sum,
        // i8/i16 for narrow-int Add/Mul), so each of those arms
        // resolves the right template inline rather than touching the
        // float-only `elem_kind` shorthand.
        let elem_for_unary =
            || -> Result<kernels::ElemKind, Unsupported> { Self::elem_kind(&node.output_type) };
        let operand_prec = || dag.get(node.inputs[0]).unwrap().output_type.precision;
        Ok(match op {
            // WS-A4: Add / Mul dispatch on operand precision so each
            // dtype gets its own kernel source. f32/f64 route through
            // the WS-A2 `ElemKind` template (which now also handles
            // f64); i8/i16 route through the typed template.
            RiscOp::Add => {
                let prec = operand_prec();
                if matches!(prec, Prim::F32 | Prim::F64) {
                    kernels::binary_elementwise(
                        self.kernel_rank,
                        name,
                        "+",
                        Self::elem_kind(&dag.get(node.inputs[0]).unwrap().output_type)?,
                    )
                } else {
                    kernels::binary_elementwise_typed(
                        self.kernel_rank,
                        name,
                        "+",
                        Self::dtype_c_type(prec),
                    )
                }
            }
            RiscOp::Sub => kernels::binary_elementwise(
                self.kernel_rank,
                name,
                "-",
                Self::elem_kind(&dag.get(node.inputs[0]).unwrap().output_type)?,
            ),
            RiscOp::Mul => {
                let prec = operand_prec();
                if matches!(prec, Prim::F32 | Prim::F64) {
                    kernels::binary_elementwise(
                        self.kernel_rank,
                        name,
                        "*",
                        Self::elem_kind(&dag.get(node.inputs[0]).unwrap().output_type)?,
                    )
                } else {
                    kernels::binary_elementwise_typed(
                        self.kernel_rank,
                        name,
                        "*",
                        Self::dtype_c_type(prec),
                    )
                }
            }
            // IEEE elementwise division. The type checker rejects
            // integer operands at every entry point (the direct-call
            // arm in `validate_polymorphic_op_constraints` and the
            // polymorphic-wrapper arm in
            // `TRANSCENDENTAL_FLOAT_ONLY_OPS`), so only f32/f64 can
            // reach codegen here. A `debug_assert!` guards the
            // invariant; release-mode builds will still emit a
            // float kernel for whatever precision lands here.
            RiscOp::Div => {
                let prec = operand_prec();
                debug_assert!(
                    matches!(prec, Prim::F32 | Prim::F64),
                    "RiscOp::Div on non-float precision `{prec:?}` reached HIP \
                     codegen; the type checker should reject this at \
                     spec/04-type-system.md \u{00a7}5.4 before lowering"
                );
                kernels::binary_elementwise(
                    self.kernel_rank,
                    name,
                    "/",
                    Self::elem_kind(&dag.get(node.inputs[0]).unwrap().output_type)?,
                )
            }
            // chelis#178: floor division (round toward -inf). Integer
            // operands use the sign-corrected kernel; float operands use
            // `floorf(a / b)`.
            RiscOp::FloorDiv => {
                let prec = operand_prec();
                kernels::binary_floor_div_typed(
                    self.kernel_rank,
                    name,
                    Self::dtype_c_type(prec),
                    prec.is_integer(),
                )
            }
            // chelis#178: truncating (round-toward-zero) division. Integer
            // operands only — native `/` is exactly the C truncating
            // quotient, so it reuses the typed binary template.
            RiscOp::Mod => return Err(Self::remainder_unsupported(node)),
            RiscOp::TruncDiv => {
                let prec = operand_prec();
                debug_assert!(
                    prec.is_integer(),
                    "RiscOp::TruncDiv on non-integer precision `{prec:?}` reached HIP \
                     codegen; the type checker should reject this at \
                     spec/05-risc-primitives.md \u{00a7}2.1 before lowering"
                );
                kernels::binary_elementwise_typed(
                    self.kernel_rank,
                    name,
                    "/",
                    Self::dtype_c_type(prec),
                )
            }
            RiscOp::MaxElem | RiscOp::MinElem => {
                let precision = operand_prec();
                let is_max = matches!(op, RiscOp::MaxElem);
                if precision.is_integer() {
                    kernels::binary_extrema_integer(
                        self.kernel_rank,
                        name,
                        is_max,
                        Self::dtype_c_type(precision),
                    )
                } else {
                    kernels::binary_extrema(self.kernel_rank, name, is_max, elem_for_unary()?)
                }
            }
            RiscOp::ExtremaAdjoint { kind, operand } => kernels::extrema_adjoint(
                self.kernel_rank,
                name,
                matches!(kind, ExtremaKind::Max),
                matches!(operand, ExtremaOperand::Left),
                elem_for_unary()?,
            ),
            RiscOp::Relu => match operand_prec() {
                Prim::F16 => kernels::relu_reduced(self.kernel_rank, name, 0x7c00, 0x03ff),
                Prim::Bf16 => kernels::relu_reduced(self.kernel_rank, name, 0x7f80, 0x007f),
                _ => kernels::relu(self.kernel_rank, name, elem_for_unary()?),
            },
            RiscOp::ReluAdjoint => match operand_prec() {
                Prim::F16 => kernels::relu_adjoint_reduced(self.kernel_rank, name, 0x7c00, 0x03ff),
                Prim::Bf16 => kernels::relu_adjoint_reduced(self.kernel_rank, name, 0x7f80, 0x007f),
                _ => kernels::relu_adjoint(self.kernel_rank, name, elem_for_unary()?),
            },
            RiscOp::CmpLt => {
                let operand_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                Self::require_result_width_matches_operand(node, operand_ty)?;
                kernels::cmplt(self.kernel_rank, name, Self::elem_kind(operand_ty)?)
            }
            RiscOp::Neg => kernels::unary_prefix(self.kernel_rank, name, "-", elem_for_unary()?),
            // IEEE reciprocal kernel.
            RiscOp::Recip => kernels::unary_recip(self.kernel_rank, name, elem_for_unary()?),
            RiscOp::Exp => kernels::unary_func(self.kernel_rank, name, "expf", elem_for_unary()?),
            RiscOp::Log => kernels::unary_func(self.kernel_rank, name, "logf", elem_for_unary()?),
            RiscOp::Sin => kernels::unary_func(self.kernel_rank, name, "sinf", elem_for_unary()?),
            RiscOp::Sqrt => kernels::unary_func(self.kernel_rank, name, "sqrtf", elem_for_unary()?),
            RiscOp::Cos => kernels::unary_func(self.kernel_rank, name, "cosf", elem_for_unary()?),
            RiscOp::Tan => kernels::unary_func(self.kernel_rank, name, "tanf", elem_for_unary()?),
            RiscOp::Atan => kernels::unary_func(self.kernel_rank, name, "atanf", elem_for_unary()?),
            RiscOp::Abs => kernels::unary_func(self.kernel_rank, name, "fabsf", elem_for_unary()?),
            RiscOp::Floor => {
                kernels::unary_func(self.kernel_rank, name, "floorf", elem_for_unary()?)
            }
            RiscOp::Ceil => kernels::unary_func(self.kernel_rank, name, "ceilf", elem_for_unary()?),
            RiscOp::Round => {
                kernels::unary_func(self.kernel_rank, name, "rintf", elem_for_unary()?)
            }
            RiscOp::UniformLike { .. } => {
                kernels::uniform_like(self.kernel_rank, name, elem_for_unary()?)
            }
            // WS-A4: bind `accumulator` instead of `..`. The fused
            // reduction path is f32-only today (its source kernel
            // template doesn't carry a dtype suffix); the unfused path
            // dispatches on (source, accumulator) and uses the
            // promoted-accumulator template for the i8/i16 → i32
            // case.
            RiscOp::Sum { axis, accumulator } => {
                let input_id = node.inputs[0];
                let operand_kind = Self::elem_kind(&dag.get(input_id).unwrap().output_type)?;
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
                    kernels::reduce_fused(
                        self.kernel_rank,
                        name,
                        *axis,
                        steps,
                        n_ext,
                        kernels::ReduceKind::Sum,
                        operand_kind,
                    )
                } else {
                    let src = operand_prec();
                    if matches!(src, Prim::Int8 | Prim::Int16) && *accumulator == Prim::Int32 {
                        // WS-A4 i8/i16 → i32 promoted-accumulator path.
                        kernels::reduce_sum_promoted(
                            self.kernel_rank,
                            name,
                            *axis,
                            Self::dtype_c_type(src),
                            "int32_t",
                        )
                    } else {
                        // WS-A2 f32/f64 path. `operand_kind` and the
                        // accumulator `ElemKind` (derived from
                        // `node.output_type`) drive the dtype-parameterized
                        // template; non-float dtypes that aren't covered
                        // by the WS-A4 promoted path panic loudly inside
                        // `elem_kind` rather than silently downgrading.
                        let acc_kind = Self::elem_kind(&node.output_type)?;
                        kernels::reduce_sum(self.kernel_rank, name, *axis, operand_kind, acc_kind)
                    }
                }
            }
            RiscOp::MaxReduce { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
                    let inner_kind = Self::elem_kind(&fused_node.output_type)?;
                    kernels::reduce_fused(
                        self.kernel_rank,
                        name,
                        *axis,
                        steps,
                        n_ext,
                        kernels::ReduceKind::Max,
                        inner_kind,
                    )
                } else {
                    kernels::reduce_max(self.kernel_rank, name, *axis, elem_for_unary()?)
                }
            }
            RiscOp::MinReduce { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                kernels::reduce_min(self.kernel_rank, name, *axis, Self::elem_kind(input_ty)?)
            }
            RiscOp::ProdReduce { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                kernels::reduce_prod(self.kernel_rank, name, *axis, Self::elem_kind(input_ty)?)
            }
            RiscOp::Argmax { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                kernels::reduce_argmax(self.kernel_rank, name, *axis, Self::elem_kind(input_ty)?)
            }
            RiscOp::Argmin { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                kernels::reduce_argmin(self.kernel_rank, name, *axis, Self::elem_kind(input_ty)?)
            }
            RiscOp::Const { .. } => kernels::fill(self.kernel_rank, name, elem_for_unary()?),
            RiscOp::ConstTensor { .. } => kernels::fill(self.kernel_rank, name, elem_for_unary()?),
            RiscOp::Reshape { .. } | RiscOp::Realize => kernels::reshape_copy(
                self.kernel_rank,
                name,
                node.output_type
                    .precision
                    .runtime_dtype()
                    .expect("verified numeric representation")
                    .byte_width(),
            ),
            RiscOp::Cast { .. } => self.cast_kernel_source(name, node, dag)?,
            RiscOp::CastTrunc { .. } => {
                return Err(Self::cast_trunc_unsupported(node));
            }
            RiscOp::Copy => self.cast_kernel_source(name, node, dag)?,
            RiscOp::FusedElem { ops } => {
                let aliased_ext = self.fused_reuse.get(&node.id).map(|reuse| {
                    let reusable = reuse.mechanics(node.id).reusable_input;
                    node.inputs
                        .iter()
                        .position(|&input| input == reusable)
                        .expect("reusable input must appear in node inputs")
                });
                kernels::fused_elementwise(
                    self.kernel_rank,
                    name,
                    ops,
                    node.inputs.len(),
                    aliased_ext,
                    elem_for_unary()?,
                )
            }
            RiscOp::Gather { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                let elem = elem_for_unary()?;
                match indices_ty.precision {
                    Prim::Int32 => kernels::gather(self.kernel_rank, name, "int", elem),
                    Prim::Int64 => kernels::gather(self.kernel_rank, name, "long long", elem),
                    other => panic!(
                        "HIP backend sparse gather requires int32/int64 indices, got {}",
                        other.name()
                    ),
                }
            }
            RiscOp::ScatterAdd { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                let elem = elem_for_unary()?;
                match indices_ty.precision {
                    Prim::Int32 => kernels::scatter_add(self.kernel_rank, name, "int", elem),
                    Prim::Int64 => kernels::scatter_add(self.kernel_rank, name, "long long", elem),
                    other => panic!(
                        "HIP backend sparse scatter_add requires int32/int64 indices, got {}",
                        other.name()
                    ),
                }
            }
            RiscOp::Scatter { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                match indices_ty.precision {
                    Prim::Int32 => kernels::scatter_replace(self.kernel_rank, name, "int"),
                    Prim::Int64 => kernels::scatter_replace(self.kernel_rank, name, "long long"),
                    other => panic!(
                        "HIP backend sparse scatter_replace requires int32/int64 indices, got {}",
                        other.name()
                    ),
                }
            }
            RiscOp::ScatterElements { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                match indices_ty.precision {
                    Prim::Int32 => kernels::scatter_elements(self.kernel_rank, name, "int"),
                    Prim::Int64 => kernels::scatter_elements(self.kernel_rank, name, "long long"),
                    other => panic!(
                        "HIP backend sparse scatter_elements requires int32/int64 indices, got {}",
                        other.name()
                    ),
                }
            }
            // `pad` / `shrink` use the typed per-output-element kernels.
            // Dispatch on the output dtype (the input dtype always matches:
            // these ops do not change precision) so the full active
            // dtype set is covered, matching the C backend.
            RiscOp::Pad { .. } => kernels::pad_typed(
                self.kernel_rank,
                name,
                Self::dtype_c_type(node.output_type.precision),
            ),
            RiscOp::Shrink { .. } => kernels::shrink_typed(
                self.kernel_rank,
                name,
                Self::dtype_c_type(node.output_type.precision),
            ),
            _ => unreachable!("no kernel for op: {op:?}"),
        })
    }

    /// Cast / Realize / Copy kernel source: in-precision identity when
    /// src and dst kinds agree, cross-precision conversion otherwise.
    fn cast_kernel_source(
        &self,
        name: &str,
        node: &DagNode,
        dag: VerifiedDagView<'_>,
    ) -> Result<String, Unsupported> {
        let (src_kind, dst_kind) = Self::cast_elem_kinds(node, dag)?;
        Ok(if src_kind == dst_kind {
            kernels::cast(self.kernel_rank, name, dst_kind)
        } else {
            kernels::cast_convert(self.kernel_rank, name, src_kind, dst_kind)
        })
    }

    /// Extract the fused steps and number of external inputs from a FusedElem op.
    fn extract_fused_steps(op: &RiscOp) -> (&[chelis_ir::dag::FusedStep], usize) {
        match op {
            RiscOp::FusedElem { ops } => {
                // Count unique external inputs referenced by the steps
                let n_ext = ops
                    .iter()
                    .flat_map(|s| &s.input_indices)
                    .filter_map(|fi| match fi {
                        chelis_ir::dag::FusedInput::External(i) => Some(i),
                        _ => None,
                    })
                    .max()
                    .map(|m| m + 1)
                    .unwrap_or(0);
                (ops, n_ext)
            }
            _ => panic!("expected FusedElem op"),
        }
    }

    // ------------------------------------------------------------------
    // Kernel string declaration emission
    // ------------------------------------------------------------------

    fn emit_kernel_string_decl(&mut self, name: &str, source: &str) {
        // Emit as a const char* with escaped newlines
        let escaped = source
            .lines()
            .map(|l| {
                let escaped = l.replace('\\', "\\\\").replace('"', "\\\"");
                format!("    \"{escaped}\\n\"")
            })
            .collect::<Vec<_>>()
            .join("\n");
        self.line(&format!("const char *{name}_src =\n{escaped};"));
    }

    // ------------------------------------------------------------------
    // Node emission
    // ------------------------------------------------------------------

    fn emit_node(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) -> Result<(), Unsupported> {
        let id = node.id.0;
        // Resolve the precision-suffixed kernel name once, so the launch
        // shims agree with the kernel-source emitter on the symbol the
        // host references (e.g. `kernel_add_f32` vs `kernel_add_f64`).
        let resolved_kernel_name = || -> Result<String, Unsupported> {
            Ok(self
                .kernel_name_for_op(&node.op, node, dag)?
                .unwrap_or_else(|| panic!("op {:?} has no kernel name", node.op)))
        };
        match &node.op {
            RiscOp::Const { value } => {
                self.emit_const(id, value.as_f64_lossy(), &node.output_type)?
            }
            RiscOp::ConstTensor { data } => {
                self.emit_const_tensor(id, &data.to_f64_lossy_vec(), &node.output_type)?
            }
            RiscOp::Load { .. } => unreachable!("handled in emit_dag"),
            RiscOp::Add => self.emit_binary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Sub => self.emit_binary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Mul => self.emit_binary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Div => self.emit_binary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            // chelis#178: floor / truncating integer division launch like
            // any other binary elementwise kernel.
            RiscOp::Mod => return Err(Self::remainder_unsupported(node)),
            RiscOp::FloorDiv | RiscOp::TruncDiv => self.emit_binary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::MaxElem | RiscOp::MinElem => self.emit_binary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::ExtremaAdjoint { .. } => self.emit_ternary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Relu => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::ReluAdjoint => self.emit_binary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::CmpLt => self.emit_binary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Neg => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Recip => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Exp => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Log => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Sin => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Sqrt => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Cos => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Tan => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Atan => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Abs => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Floor => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Ceil => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Round => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::UniformLike { low, high, seed } => {
                self.emit_uniform_like_launch(id, *low, *high, *seed, &node.output_type)?
            }
            RiscOp::Dropout { .. } => {
                unreachable!("dropout should be rejected before HIP code generation")
            }
            RiscOp::Copy => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Drop => {
                let action = dag.action_for_node(node.id).ok_or_else(|| {
                    unsupported_verified_dag_action(node.id, "missing Drop action")
                })?;
                match action {
                    VerifiedDagAction::BorrowedDrop { node: drop, source }
                    | VerifiedDagAction::OwnedDrop { node: drop, source }
                        if drop == node.id && node.inputs.first() == Some(&source) =>
                    {
                        if matches!(action, VerifiedDagAction::OwnedDrop { .. }) {
                            self.line(&format!("chelis_device_tensor_release(o_t{});", source.0));
                        }
                    }
                    _ => {
                        return Err(unsupported_verified_dag_action(
                            node.id,
                            "Drop action does not name the exact typed payload source",
                        ));
                    }
                }
            }
            // WS-A4: bind `accumulator` instead of `..` and thread it
            // through to `emit_reduce_launch` so the launch-side kernel
            // name agrees with the kernel-source-side name (otherwise
            // hipModuleGetFunction fails to resolve the launched
            // symbol against the registered source).
            RiscOp::Sum { axis, accumulator } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    self.emit_fused_reduce_launch(id, *axis, &node.inputs, &node.output_type, dag);
                } else {
                    self.emit_reduce_launch(
                        id,
                        *axis,
                        &node.inputs,
                        &node.output_type,
                        dag,
                        kernels::ReduceKind::Sum,
                        Some(*accumulator),
                    )?;
                }
            }
            RiscOp::MaxReduce { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    self.emit_fused_reduce_launch(id, *axis, &node.inputs, &node.output_type, dag);
                } else {
                    self.emit_reduce_launch(
                        id,
                        *axis,
                        &node.inputs,
                        &node.output_type,
                        dag,
                        kernels::ReduceKind::Max,
                        None,
                    )?;
                }
            }
            RiscOp::MinReduce { axis }
            | RiscOp::ProdReduce { axis }
            | RiscOp::Argmax { axis }
            | RiscOp::Argmin { axis } => {
                // WS-A2 lift: these four reductions previously panicked
                // ("Phase 3j-pre ships C backend only"). The kernel
                // launch shape mirrors `emit_reduce_launch` but the
                // kernel source comes from `extra_reduction_kernel_sources`
                // collected by the first pass.
                self.emit_extra_reduce_launch(id, *axis, &node.inputs, &node.output_type, dag)?;
            }
            // `reduce_window_*` HIP codegen is excluded by the
            // target contract ([05-RWIN-2] / spec §2.3.1). C is
            // the canonical backend. A `reduce_window_*` node is rejected
            // before codegen with a clean `unsupported_feature` diagnostic by
            // `reject_unsupported_hip_ops` (compiler-api + CLI mirror); the
            // `todo!` below is a defensive backstop matching the Pad / Shrink
            // HIP stubs above, reached only if some path bypasses that guard.
            RiscOp::ReduceWindow { .. } => {
                todo!(
                    "reduce_window_* must have been rejected before HIP codegen by [05-RWIN-2]. Use the C backend."
                )
            }
            RiscOp::ReduceWindowGrad { .. } => {
                todo!(
                    "ReduceWindowGrad must have been rejected before HIP codegen by [05-RWIN-2]. Use the C backend."
                )
            }
            RiscOp::OneHot { .. } => {
                panic!(
                    "HIP backend: internal OneHot must be consumed by specialization before codegen"
                )
            }
            // The runtime `shape` value read is excluded by [05-SHAPE-1]
            // and rejected before codegen by
            // `reject_unsupported_hip_ops` (compiler-api + CLI mirror); this
            // `todo!` is a defensive backstop matching the ReduceWindow
            // stubs above, reached only if some path bypasses that guard.
            RiscOp::Shape { .. }
            | RiscOp::ExtentWitness { .. }
            | RiscOp::CheckedReshapeExtent { .. }
            | RiscOp::CheckedUnitAxis { .. } => {
                todo!(
                    "runtime `shape` value read must have been rejected before HIP codegen by [05-SHAPE-1]. Use `--target c`."
                )
            }
            RiscOp::Reshape { .. } => {
                self.emit_reshape(id, &node.inputs, &node.output_type);
            }
            RiscOp::Permute { axes } => {
                self.emit_permute(id, axes, &node.inputs, &node.output_type);
            }
            RiscOp::Expand { axis, size } => {
                self.emit_expand(id, *axis, size, &node.inputs, &node.output_type);
            }
            RiscOp::Pad { padding, fill } => {
                self.emit_pad_launch(
                    id,
                    &hip_pairs_to_usize(padding),
                    *fill,
                    &resolved_kernel_name()?,
                    &node.inputs,
                    &node.output_type,
                    dag,
                );
            }
            RiscOp::Shrink { bounds } => {
                self.emit_shrink_launch(
                    id,
                    &hip_pairs_to_usize(bounds),
                    &resolved_kernel_name()?,
                    &node.inputs,
                    &node.output_type,
                    dag,
                );
            }
            RiscOp::Stride { strides } => {
                self.emit_stride(
                    id,
                    &hip_strides_to_usize(strides),
                    &node.inputs,
                    &node.output_type,
                );
            }
            RiscOp::Realize => self.emit_logical_copy(
                id,
                &node.inputs,
                &node.output_type,
                &resolved_kernel_name()?,
            ),
            RiscOp::Cast { .. } => self.emit_unary_launch(
                id,
                &resolved_kernel_name()?,
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::CastTrunc { .. } => return Err(Self::cast_trunc_unsupported(node)),
            RiscOp::Count { .. } => {
                return Err(Unsupported::new(
                    UnsupportedKind::Op("count".to_string()),
                    format!("the HIP kernel set (node {})", node.id.0),
                    Stage::Codegen("hip"),
                    chelis_types::unimplemented_rejection!(
                        1291,
                        "chelis#1291 owns the dedicated HIP/Metal count kernels"
                    ),
                ));
            }
            RiscOp::Store { name } => {
                self.emit_store(id, name.as_str(), &node.inputs, &node.output_type)
            }
            RiscOp::FusedElem { ops } => {
                let kernel_name = format!("kernel_fused_{}", node.id.0);
                let in_place = self
                    .fused_reuse
                    .get(&node.id)
                    .map(|reuse| reuse.mechanics(node.id));
                self.emit_fused_launch(
                    node.id.0,
                    &kernel_name,
                    &node.inputs,
                    ops,
                    &node.output_type,
                    in_place,
                );
            }
            // F1 footgun fix (WS-A3): bind every BlasMatmul field
            // explicitly so a future field addition (e.g. an alpha/beta
            // scaling parameter) cannot be silently ignored by `..` the
            // way the original `accumulator` field was. The accumulator
            // is the precision the inner-product runs at, per
            // spec/04-type-system.md §5.7.1.
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
                accumulator,
            } => {
                self.emit_blas_matmul(
                    id,
                    &MatmulEmitSpec {
                        a: node.inputs[0],
                        b: node.inputs[1],
                        batch_dims: batch_dims.clone(),
                        m: m.clone(),
                        n: n.clone(),
                        k: k.clone(),
                        accumulator: *accumulator,
                    },
                    &node.output_type,
                    dag,
                );
            }
            RiscOp::Gather { axis } => {
                self.emit_gather_launch(id, *axis, &node.inputs, &node.output_type, dag)?
            }
            RiscOp::ScatterAdd { axis } => {
                self.emit_scatter_add_launch(id, *axis, &node.inputs, &node.output_type, dag)?
            }
            RiscOp::Scatter { axis } => {
                self.emit_scatter_replace_launch(id, *axis, &node.inputs, &node.output_type, dag)
            }
            RiscOp::ScatterElements { axis } => {
                self.emit_scatter_elements_launch(id, *axis, &node.inputs, &node.output_type, dag)
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Const (fill kernel)
    // ------------------------------------------------------------------

    fn slot_id_for_node(&self, id: usize) -> usize {
        match self.plan.node_kind(NodeId(id)) {
            NodeMemoryKind::UniqueInput { slot, .. } | NodeMemoryKind::SlotBacked { slot } => *slot,
            other => panic!("node {id} does not own slot-backed storage: {other:?}"),
        }
    }

    fn emit_reshape_count_preflight(&mut self, dag: VerifiedDagView<'_>) {
        for node in dag.nodes() {
            if !matches!(node.op, RiscOp::Reshape { .. }) {
                continue;
            }
            let source = dag.get(node.inputs[0]).expect("verified reshape source");
            let id = node.id.0;
            self.emit_metadata_plan(
                &format!("reshape_input{id}"),
                &source.output_type,
                None,
                "0",
            );
            self.emit_metadata_plan(&format!("reshape_output{id}"), &node.output_type, None, "0");
            self.line(&format!("if (chelis_metadata_plan_count(reshape_input{id}) != chelis_metadata_plan_count(reshape_output{id})) chelis_numeric_trap(\"numeric trap: domain in reshape at int64\");"));
            self.line(&format!(
                "chelis_metadata_plan_release(reshape_output{id});"
            ));
            self.line(&format!("chelis_metadata_plan_release(reshape_input{id});"));
        }
    }

    fn tagged_i64(expression: &str) -> String {
        format!("chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)({expression}))")
    }

    fn emit_metadata_plan(
        &mut self,
        name: &str,
        ty: &TensorType,
        strides: Option<&[String]>,
        capacity: &str,
    ) {
        let rank = ty.dims.len();
        let rank_scalar = Self::tagged_i64(&rank.to_string());
        let exemplar = format!("chelis_scalar_from_bits({}, 0)", Self::dtype_macro(ty));
        let shape = if rank == 0 {
            "NULL".to_string()
        } else {
            let values = ty
                .dims
                .iter()
                .map(|dim| Self::tagged_i64(&Self::emit_dim_info(dim)))
                .collect::<Vec<_>>()
                .join(", ");
            self.line(&format!(
                "const chelis_scalar {name}_shape[{rank}] = {{ {values} }};"
            ));
            format!("{name}_shape")
        };
        if let Some(strides) = strides {
            assert_eq!(
                strides.len(),
                rank,
                "a view requires every checked axis stride"
            );
            let stride_array = if rank == 0 {
                "NULL".to_string()
            } else {
                let values = strides
                    .iter()
                    .map(|stride| Self::tagged_i64(stride))
                    .collect::<Vec<_>>()
                    .join(", ");
                self.line(&format!(
                    "const chelis_scalar {name}_strides[{rank}] = {{ {values} }};"
                ));
                format!("{name}_strides")
            };
            let capacity = Self::tagged_i64(capacity);
            self.line(&format!("chelis_metadata_plan *{name} = chelis_metadata_plan_view({rank_scalar}, {shape}, {stride_array}, {exemplar}, {capacity});"));
        } else {
            self.line(&format!("chelis_metadata_plan *{name} = chelis_metadata_plan_new({rank_scalar}, {shape}, {exemplar});"));
        }
    }

    fn emit_owner_observation(&mut self, id: usize) {
        self.line(&format!(
            "const chelis_gpu_tensor *d_t{id} = chelis_device_tensor_view(o_t{id});"
        ));
        // Kernel launch argument storage is local; no cast removes packet constness.
        self.line(&format!("void *p_t{id} = d_t{id}->data;"));
    }

    fn emit_slot_allocation_if_needed(&mut self, id: usize, ty: &TensorType) {
        if self.device_entrypoint_mode {
            return;
        }
        let slot_id = self.slot_id_for_node(id);
        if self.plan.slot(slot_id).first_owner != NodeId(id) {
            return;
        }
        self.emit_metadata_plan(&format!("slot_plan{slot_id}"), ty, None, "0");
        self.line(&format!("chelis_device_tensor_owner *chelis_slot{slot_id} = chelis_device_tensor_alloc(slot_plan{slot_id});"));
    }

    fn emit_slot_wrapper(&mut self, id: usize, ty: &TensorType) {
        self.emit_slot_allocation_if_needed(id, ty);
        let slot = self.slot_id_for_node(id);
        let source = format!("chelis_device_tensor_view(chelis_slot{slot})");
        let capacity = format!("{source}->byte_capacity");
        self.emit_metadata_plan(&format!("plan_t{id}"), ty, None, &capacity);
        self.line(&format!("chelis_device_tensor_owner *o_t{id} = chelis_device_tensor_borrow(plan_t{id}, {source}->data, {});", Self::tagged_i64(&capacity)));
        self.emit_owner_observation(id);
    }

    fn emit_device_slot_allocations(&mut self, dag: VerifiedDagView<'_>) {
        let slots = self
            .plan
            .slots()
            .iter()
            .map(|slot| (slot.id, slot.first_owner))
            .collect::<Vec<_>>();
        for (slot, owner) in slots {
            if matches!(
                self.plan.node_kind(owner),
                NodeMemoryKind::UniqueInput { .. }
            ) {
                // Device inputs borrow caller storage; their host-mirror slot
                // is absent on this entry path and never authorizes reuse.
                self.line(&format!(
                    "chelis_device_tensor_owner *chelis_slot{slot} = NULL;"
                ));
                continue;
            }
            let ty = &dag
                .get(owner)
                .expect("verified slot first owner")
                .output_type;
            self.emit_metadata_plan(&format!("slot_plan{slot}"), ty, None, "0");
            self.line(&format!("chelis_device_tensor_owner *chelis_slot{slot} = chelis_device_tensor_alloc(slot_plan{slot});"));
        }
    }

    fn emit_alias_view(&mut self, id: usize, ty: &TensorType, source: &str) {
        let strides = (0..ty.dims.len())
            .map(|axis| format!("{source}->strides[{axis}]"))
            .collect::<Vec<_>>();
        self.emit_strided_view(id, ty, source, &strides);
    }

    fn emit_strided_view(&mut self, id: usize, ty: &TensorType, source: &str, strides: &[String]) {
        let capacity = format!("{source}->byte_capacity");
        self.emit_metadata_plan(&format!("plan_t{id}"), ty, Some(strides), &capacity);
        self.line(&format!("chelis_device_tensor_owner *o_t{id} = chelis_device_tensor_borrow(plan_t{id}, {source}->data, {});", Self::tagged_i64(&capacity)));
        self.emit_owner_observation(id);
    }

    fn emit_const(&mut self, id: usize, value: f64, ty: &TensorType) -> Result<(), Unsupported> {
        self.emit_slot_wrapper(id, ty);
        let elem = Self::elem_kind(ty)?;
        let kernel = format!("kernel_fill_{}", elem.suffix());
        self.line("{");
        self.indent += 1;
        match elem {
            kernels::ElemKind::F32 => {
                // Issue #250 (parallel #189): narrow the IR's f64 source to
                // f32 (storage width is f32) and reconstruct the fill value
                // from its exact bit pattern via the `chelis_f32_from_bits`
                // static inline helper (declared in the included
                // `chelis_runtime.h`). The pre-fix `{:.8}f` format string
                // printed decimal places after the point, not significant
                // digits, so small magnitudes drifted by ~3% or collapsed
                // to zero when the emitted HIP host code parsed the literal
                // back. Bit-pattern emission round-trips the closest-f32 to
                // the source value verbatim.
                let bits = (value as f32).to_bits();
                self.line(&format!(
                    "float fill_val = chelis_f32_from_bits(0x{bits:08x}u);"
                ));
            }
            kernels::ElemKind::F64 => {
                // Issue #250 sibling: emit the source f64's exact bit
                // pattern and reconstruct it via `chelis_f64_from_bits`
                // rather than a decimal format string. Symmetric with the
                // f32 arm above and with the C backend's #189 fix.
                let bits = value.to_bits();
                self.line(&format!(
                    "double fill_val = chelis_f64_from_bits(0x{bits:016x}uLL);"
                ));
            }
        }
        self.line(&format!("int64_t fill_size = d_t{id}->count;"));
        self.line(&format!(
            "void *fill_args[] = {{ &p_t{id}, &fill_val, &fill_size }};"
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel}"),
            &kernel,
            "fill_size / 256 + (fill_size % 256 != 0)",
            "256",
            "fill_args",
        );
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    /// Emit a multi-element constant tensor on HIP. Uploads data
    /// element-by-element via fill calls (same kernel as Const).
    fn emit_const_tensor(
        &mut self,
        id: usize,
        data: &[f64],
        ty: &TensorType,
    ) -> Result<(), Unsupported> {
        self.emit_slot_wrapper(id, ty);
        let elem = Self::elem_kind(ty)?;
        self.line("{");
        self.indent += 1;
        match elem {
            kernels::ElemKind::F32 => {
                // Upload the full data via a host-side memcpy into a
                // temporary then hipMemcpy to device. For simplicity,
                // reuse the fill kernel per-element is too slow; instead
                // build a host-side buffer and copy.
                self.line(&format!("int64_t fill_size = d_t{id}->count;"));
                self.line(&format!(
                    "float *__host_data = (float*)malloc({}u * sizeof(float));",
                    data.len()
                ));
                for (i, v) in data.iter().enumerate() {
                    let bits = (*v as f32).to_bits();
                    self.line(&format!(
                        "__host_data[{i}] = chelis_f32_from_bits(0x{bits:08x}u);"
                    ));
                }
                self.line(&format!(
                    "hipMemcpy(d_t{id}->data, __host_data, {}u * sizeof(float), hipMemcpyHostToDevice);",
                    data.len()
                ));
                self.line("free(__host_data);");
            }
            kernels::ElemKind::F64 => {
                self.line(&format!("int64_t fill_size = d_t{id}->count;"));
                self.line(&format!(
                    "double *__host_data = (double*)malloc({}u * sizeof(double));",
                    data.len()
                ));
                for (i, v) in data.iter().enumerate() {
                    let bits = v.to_bits();
                    self.line(&format!(
                        "__host_data[{i}] = chelis_f64_from_bits(0x{bits:016x}uLL);"
                    ));
                }
                self.line(&format!(
                    "hipMemcpy(d_t{id}->data, __host_data, {}u * sizeof(double), hipMemcpyHostToDevice);",
                    data.len()
                ));
                self.line("free(__host_data);");
            }
        }
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    // ------------------------------------------------------------------
    // Load
    // ------------------------------------------------------------------

    fn emit_load(&mut self, id: usize, input_idx: usize, ty: &TensorType) {
        match self.plan.node_kind(NodeId(id)) {
            NodeMemoryKind::UniqueInput { .. } => {
                self.emit_slot_wrapper(id, ty);
                let slot = self.slot_id_for_node(id);
                self.line(&format!(
                    "chelis_device_tensor_copy_from_host(chelis_slot{slot}, inputs[{input_idx}]);"
                ));
            }
            NodeMemoryKind::RepeatedLoadAlias { canonical_load } => {
                self.emit_alias_view(id, ty, &format!("d_t{}", canonical_load.0));
            }
            other => panic!("unexpected memory plan for load node {id}: {other:?}"),
        }
    }

    fn emit_load_device(&mut self, id: usize, input_idx: usize, ty: &TensorType) {
        match self.plan.node_kind(NodeId(id)) {
            NodeMemoryKind::UniqueInput { .. } => {
                self.emit_alias_view(id, ty, &format!("input_view_{input_idx}"));
            }
            NodeMemoryKind::RepeatedLoadAlias { canonical_load } => {
                self.emit_alias_view(id, ty, &format!("d_t{}", canonical_load.0));
            }
            other => panic!("unexpected memory plan for load node {id}: {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // Binary elementwise kernel launch
    // ------------------------------------------------------------------

    fn emit_binary_launch(
        &mut self,
        id: usize,
        kernel_name: &str,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int64_t t{id}_size = d_t{id}->count;"));
        // Stride params for a (8 ints)
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int64_t t{id}_a_ndim = d_t{a}->rank;"));
        self.line(&format!(
            "int64_t t{id}_a_size = (d_t{a}->byte_capacity / chelis_dtype_size(d_t{a}->dtype));"
        ));
        // Stride params for b (8 ints)
        self.emit_stride_vars(id, "b", b);
        self.line(&format!("int64_t t{id}_b_ndim = d_t{b}->rank;"));
        self.line(&format!(
            "int64_t t{id}_b_size = (d_t{b}->byte_capacity / chelis_dtype_size(d_t{b}->dtype));"
        ));
        // Shape params for output (8 ints)
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_out_ndim = d_t{id}->rank;"));
        // Build args array
        self.line(&format!(
            "void *args[] = {{ &p_t{a}, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             &p_t{b}, {b_stride_refs}, &t{id}_b_ndim, &t{id}_b_size, \
             &p_t{id}, {out_shape_refs}, &t{id}_out_ndim, &t{id}_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            b_stride_refs = self.stride_arg_refs(id, "b"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            kernel_name,
            &format!("t{id}_size / 256 + (t{id}_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    // ------------------------------------------------------------------
    // Ternary elementwise kernel launch
    // ------------------------------------------------------------------

    fn emit_ternary_launch(
        &mut self,
        id: usize,
        kernel_name: &str,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let g = inputs[2].0;
        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int64_t t{id}_size = d_t{id}->count;"));
        for (name, source) in [("a", a), ("b", b), ("g", g)] {
            self.emit_stride_vars(id, name, source);
            self.line(&format!("int64_t t{id}_{name}_ndim = d_t{source}->rank;"));
            self.line(&format!(
                "int64_t t{id}_{name}_size = (d_t{source}->byte_capacity / chelis_dtype_size(d_t{source}->dtype));"
            ));
        }
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_out_ndim = d_t{id}->rank;"));
        self.line(&format!(
            "void *args[] = {{ &p_t{a}, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             &p_t{b}, {b_stride_refs}, &t{id}_b_ndim, &t{id}_b_size, \
             &p_t{g}, {g_stride_refs}, &t{id}_g_ndim, &t{id}_g_size, \
             &p_t{id}, {out_shape_refs}, &t{id}_out_ndim, &t{id}_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            b_stride_refs = self.stride_arg_refs(id, "b"),
            g_stride_refs = self.stride_arg_refs(id, "g"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            kernel_name,
            &format!("t{id}_size / 256 + (t{id}_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    // ------------------------------------------------------------------
    // Unary elementwise kernel launch
    // ------------------------------------------------------------------

    fn emit_unary_launch(
        &mut self,
        id: usize,
        kernel_name: &str,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int64_t t{id}_size = d_t{id}->count;"));
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int64_t t{id}_a_ndim = d_t{a}->rank;"));
        self.line(&format!(
            "int64_t t{id}_a_size = (d_t{a}->byte_capacity / chelis_dtype_size(d_t{a}->dtype));"
        ));
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_out_ndim = d_t{id}->rank;"));
        self.line(&format!(
            "void *args[] = {{ &p_t{a}, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             &p_t{id}, {out_shape_refs}, &t{id}_out_ndim, &t{id}_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            kernel_name,
            &format!("t{id}_size / 256 + (t{id}_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    fn emit_uniform_like_launch(
        &mut self,
        id: usize,
        low: f64,
        high: f64,
        seed: u64,
        ty: &TensorType,
    ) -> Result<(), Unsupported> {
        self.emit_slot_wrapper(id, ty);
        let elem = Self::elem_kind(ty)?;
        let kernel = format!("kernel_uniform_like_{}", elem.suffix());
        self.line("{");
        self.indent += 1;
        // [05-OP-8]: bounds have f32 dtype. The f32 kernel consumes those
        // exact images; the f64 kernel widens the same images exactly and
        // executes the affine at f64 width.
        let low_bits = (low as f32).to_bits();
        let high_bits = (high as f32).to_bits();
        match elem {
            kernels::ElemKind::F32 => {
                self.line(&format!(
                    "float t{id}_low = chelis_f32_from_bits(0x{low_bits:08x}u);"
                ));
                self.line(&format!(
                    "float t{id}_high = chelis_f32_from_bits(0x{high_bits:08x}u);"
                ));
            }
            kernels::ElemKind::F64 => {
                let low_wide_bits = (low as f32 as f64).to_bits();
                let high_wide_bits = (high as f32 as f64).to_bits();
                self.line(&format!(
                    "double t{id}_low = chelis_f64_from_bits(0x{low_wide_bits:016x}uLL);"
                ));
                self.line(&format!(
                    "double t{id}_high = chelis_f64_from_bits(0x{high_wide_bits:016x}uLL);"
                ));
            }
        }
        self.line(&format!("unsigned long long t{id}_seed = {seed}ULL;"));
        self.line(&format!("int64_t t{id}_size = d_t{id}->count;"));
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_out_ndim = d_t{id}->rank;"));
        self.line(&format!(
            "void *args[] = {{ &t{id}_low, &t{id}_high, &t{id}_seed, &p_t{id}, {out_shape_refs}, &t{id}_out_ndim, &t{id}_size }};",
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel}"),
            &kernel,
            &format!("t{id}_size / 256 + (t{id}_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    fn emit_logical_metadata_args(&mut self, id: usize, prefix: &str, source: usize) -> String {
        self.emit_shape_vars(id, prefix, source);
        self.emit_stride_vars(id, prefix, source);
        self.line(&format!("int64_t t{id}_{prefix}_ndim = d_t{source}->rank;"));
        format!(
            "{}, {}, &t{id}_{prefix}_ndim",
            self.shape_arg_refs(id, prefix),
            self.stride_arg_refs(id, prefix)
        )
    }

    fn emit_materialize_into_slot(&mut self, id: usize, source: usize) {
        self.line(&format!("if (d_t{id}->count != d_t{source}->count) chelis_numeric_trap(\"numeric trap: domain in materialize at int64\");"));
        self.line("{");
        self.indent += 1;
        self.emit_stride_vars(id, "copy", source);
        self.emit_shape_vars(id, "copy", source);
        self.line(&format!("int64_t t{id}_copy_rank = d_t{source}->rank;"));
        self.line(&format!("int64_t t{id}_copy_count = d_t{id}->count;"));
        self.line(&format!("void *copy_args[] = {{ &p_t{source}, {}, {}, &t{id}_copy_rank, &p_t{id}, &t{id}_copy_count }};", self.stride_arg_refs(id, "copy"), self.shape_arg_refs(id, "copy")));
        let name = format!("kernel_materialize_{id}");
        self.emit_kernel_launch_expr(
            &format!("mod_{name}"),
            &name,
            &format!("t{id}_copy_count / 256 + (t{id}_copy_count % 256 != 0)"),
            "256",
            "copy_args",
        );
        self.indent -= 1;
        self.line("}");
    }

    fn emit_sparse_geometry(&mut self, id: usize, axis: usize, ty: &TensorType) {
        let plan = format!("sparse_geometry{id}");
        self.emit_metadata_plan(&plan, ty, None, "0");
        self.line(&format!(
            "int64_t t{id}_axis_size = chelis_metadata_plan_shape({plan})[{axis}];"
        ));
        self.line(&format!(
            "int64_t t{id}_after = chelis_metadata_plan_strides({plan})[{axis}];"
        ));
        // An empty source admits no logical read. Avoid dividing by its zero
        // extent/stride; the bounds check rejects any attempted index into it.
        self.line(&format!("int64_t t{id}_before = 0;"));
        self.line(&format!("if (chelis_metadata_plan_count({plan}) != 0) t{id}_before = chelis_metadata_plan_count({plan}) / t{id}_axis_size / t{id}_after;"));
        self.line(&format!("chelis_metadata_plan_release({plan});"));
    }

    fn emit_gather_launch(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) -> Result<(), Unsupported> {
        let values = inputs[0].0;
        let indices = inputs[1].0;
        let values_ty = &dag.get(inputs[0]).unwrap().output_type;
        let indices_ty = &dag.get(inputs[1]).unwrap().output_type;
        if values_ty.precision != ty.precision
            || !matches!(values_ty.precision, Prim::F32 | Prim::F64)
        {
            panic!(
                "HIP backend sparse gather requires matching f32 or f64 payload precision; \
                 values=`{}`, out=`{}`",
                values_ty.precision.name(),
                ty.precision.name(),
            );
        }
        if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
            panic!(
                "HIP backend sparse gather requires int32/int64 indices, got {}",
                indices_ty.precision.name()
            );
        }
        let elem = Self::elem_kind(ty)?;
        let kernel_name = match indices_ty.precision {
            Prim::Int32 => format!("kernel_gather_i32_{}", elem.suffix()),
            Prim::Int64 => format!("kernel_gather_i64_{}", elem.suffix()),
            _ => unreachable!(),
        };

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.emit_sparse_geometry(id, axis, values_ty);
        self.line(&format!("int64_t t{id}_index_count = d_t{indices}->count;"));
        self.line(&format!("int64_t t{id}_total = d_t{id}->count;"));
        let values_metadata = self.emit_logical_metadata_args(id, "values", values);
        let indices_metadata = self.emit_logical_metadata_args(id, "idx", indices);
        self.line(&format!(
            "void *args[] = {{ &p_t{values}, &p_t{indices}, &p_t{id}, &t{id}_before, &t{id}_axis_size, &t{id}_after, &t{id}_index_count, &t{id}_total, {values_metadata}, {indices_metadata} }};"
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            &kernel_name,
            &format!("t{id}_total / 256 + (t{id}_total % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    fn emit_scatter_add_launch(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) -> Result<(), Unsupported> {
        let target = inputs[0].0;
        let indices = inputs[1].0;
        let updates = inputs[2].0;
        let target_ty = &dag.get(inputs[0]).unwrap().output_type;
        let indices_ty = &dag.get(inputs[1]).unwrap().output_type;
        let updates_ty = &dag.get(inputs[2]).unwrap().output_type;
        if target_ty.precision != ty.precision
            || updates_ty.precision != ty.precision
            || !matches!(ty.precision, Prim::F32 | Prim::F64)
        {
            panic!(
                "HIP backend sparse scatter_add requires matching f32 or f64 payload precision; \
                 target=`{}`, updates=`{}`, out=`{}`",
                target_ty.precision.name(),
                updates_ty.precision.name(),
                ty.precision.name(),
            );
        }
        if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
            panic!(
                "HIP backend sparse scatter_add requires int32/int64 indices, got {}",
                indices_ty.precision.name()
            );
        }
        let elem = Self::elem_kind(ty)?;
        let kernel_name = match indices_ty.precision {
            Prim::Int32 => format!("kernel_scatter_add_i32_{}", elem.suffix()),
            Prim::Int64 => format!("kernel_scatter_add_i64_{}", elem.suffix()),
            _ => unreachable!(),
        };

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.emit_materialize_into_slot(id, target);
        self.emit_sparse_geometry(id, axis, target_ty);
        self.line(&format!("int64_t t{id}_index_count = d_t{indices}->count;"));
        self.line(&format!("int64_t t{id}_total = d_t{updates}->count;"));
        let indices_metadata = self.emit_logical_metadata_args(id, "idx", indices);
        let updates_metadata = self.emit_logical_metadata_args(id, "updates", updates);
        self.line(&format!(
            "void *args[] = {{ &p_t{indices}, &p_t{updates}, &p_t{id}, &t{id}_before, &t{id}_axis_size, &t{id}_after, &t{id}_index_count, &t{id}_total, {indices_metadata}, {updates_metadata} }};"
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            &kernel_name,
            &format!("t{id}_total / 256 + (t{id}_total % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    /// Launch the sparse replace-scatter (last-write-wins) kernel.
    ///
    /// Per `spec/05-risc-primitives.md` §3.5, the deterministic-order
    /// rule is updates-tensor row-major flat iteration. The kernel is
    /// launched as `<<<1, 1>>>` — a single thread serializes all
    /// writes so duplicate target indices resolve in the same order
    /// the IR evaluator and C backend use. This is intentionally low
    /// throughput; the AD policy for this op is `no_grad` and the
    /// design assumes scatter_replace is used in inference / data
    /// pipelines, not on a hot training path. A parallel
    /// implementation would have to preserve the same tie-breaking
    /// (max-flat-index wins per cell) — see `kernels::scatter_replace`.
    fn emit_scatter_replace_launch(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let target = inputs[0].0;
        let indices = inputs[1].0;
        let updates = inputs[2].0;
        let target_ty = &dag.get(inputs[0]).unwrap().output_type;
        let indices_ty = &dag.get(inputs[1]).unwrap().output_type;
        let updates_ty = &dag.get(inputs[2]).unwrap().output_type;
        if target_ty.precision != Prim::F32
            || updates_ty.precision != Prim::F32
            || ty.precision != Prim::F32
        {
            panic!("HIP backend sparse scatter_replace currently supports f32 payloads only");
        }
        if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
            panic!(
                "HIP backend sparse scatter_replace requires int32/int64 indices, got {}",
                indices_ty.precision.name()
            );
        }
        let kernel_name = match indices_ty.precision {
            Prim::Int32 => "kernel_scatter_replace_i32",
            Prim::Int64 => "kernel_scatter_replace_i64",
            _ => unreachable!(),
        };

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.emit_materialize_into_slot(id, target);
        self.emit_sparse_geometry(id, axis, target_ty);
        self.line(&format!("int64_t t{id}_index_count = d_t{indices}->count;"));
        self.line(&format!("int64_t t{id}_total = d_t{updates}->count;"));
        let indices_metadata = self.emit_logical_metadata_args(id, "idx", indices);
        let updates_metadata = self.emit_logical_metadata_args(id, "updates", updates);
        self.line(&format!(
            "void *args[] = {{ &p_t{indices}, &p_t{updates}, &p_t{id}, &t{id}_before, &t{id}_axis_size, &t{id}_after, &t{id}_index_count, &t{id}_total, {indices_metadata}, {updates_metadata} }};"
        ));
        // Single-thread serial launch preserves last-write-wins order.
        self.emit_kernel_launch_expr(&format!("mod_{kernel_name}"), kernel_name, "1", "1", "args");
        self.indent -= 1;
        self.line("}");
    }

    /// Launch for ONNX `ScatterElements` (spec §3.5.1). Passes the
    /// indices shape and the (data-shaped) output shape as `MAX_DIM`
    /// scalar ints each, plus `ndim`/`axis`/`axis_size`/`total`. Like
    /// `Scatter`, it launches a single-thread serial kernel to keep
    /// last-write-wins deterministic.
    fn emit_scatter_elements_launch(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let data = inputs[0].0;
        let indices = inputs[1].0;
        let updates = inputs[2].0;
        let data_ty = &dag.get(inputs[0]).unwrap().output_type;
        let indices_ty = &dag.get(inputs[1]).unwrap().output_type;
        let updates_ty = &dag.get(inputs[2]).unwrap().output_type;
        if data_ty.precision != Prim::F32
            || updates_ty.precision != Prim::F32
            || ty.precision != Prim::F32
        {
            panic!("HIP backend sparse scatter_elements currently supports f32 payloads only");
        }
        if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
            panic!(
                "HIP backend sparse scatter_elements requires int32/int64 indices, got {}",
                indices_ty.precision.name()
            );
        }
        let axis_size = Self::emit_dim_info(&data_ty.dims[axis]);
        let kernel_name = match indices_ty.precision {
            Prim::Int32 => "kernel_scatter_elements_i32",
            Prim::Int64 => "kernel_scatter_elements_i64",
            _ => unreachable!(),
        };

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        // Initialize the output from `data`; the kernel overwrites only
        // the scattered cells, so the remainder must equal `data`.
        self.emit_materialize_into_slot(id, data);
        self.emit_shape_vars(id, "idx", indices);
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_ndim = d_t{id}->rank;"));
        self.line(&format!("int64_t t{id}_axis = {axis};"));
        self.line(&format!("int64_t t{id}_axis_size = {axis_size};"));
        self.line(&format!("int64_t t{id}_total = d_t{updates}->count;"));
        let idx_sh_refs = self.shape_arg_refs(id, "idx");
        let out_sh_refs = self.shape_arg_refs(id, "out");
        self.emit_stride_vars(id, "idx", indices);
        self.emit_stride_vars(id, "updates", updates);
        self.emit_stride_vars(id, "out", id);
        let idx_stride_refs = self.stride_arg_refs(id, "idx");
        let update_stride_refs = self.stride_arg_refs(id, "updates");
        let out_stride_refs = self.stride_arg_refs(id, "out");
        self.line(&format!(
            "void *args[] = {{ &p_t{indices}, &p_t{updates}, &p_t{id}, {idx_sh_refs}, {out_sh_refs}, &t{id}_ndim, &t{id}_axis, &t{id}_axis_size, &t{id}_total, {idx_stride_refs}, {update_stride_refs}, {out_stride_refs} }};"
        ));
        // Single-thread serial launch preserves last-write-wins order.
        self.emit_kernel_launch_expr(&format!("mod_{kernel_name}"), kernel_name, "1", "1", "args");
        self.indent -= 1;
        self.line("}");
    }

    // ------------------------------------------------------------------
    // Fused elementwise kernel launch
    // ------------------------------------------------------------------

    fn emit_fused_launch(
        &mut self,
        id: usize,
        kernel_name: &str,
        inputs: &[NodeId],
        _ops: &[chelis_ir::dag::FusedStep],
        ty: &TensorType,
        in_place: Option<FusedReuseMechanics>,
    ) {
        if let Some(spec) = in_place {
            self.emit_fused_in_place_wrapper(id, ty, spec);
        } else {
            self.emit_slot_wrapper(id, ty);
        }
        self.line("{");
        self.indent += 1;
        self.line(&format!("int64_t t{id}_size = d_t{id}->count;"));

        // Emit stride vars for each external input
        for (i, inp) in inputs.iter().enumerate() {
            let pfx = format!("ext{i}");
            self.emit_stride_vars(id, &pfx, inp.0);
            self.line(&format!("int64_t t{id}_{pfx}_ndim = d_t{}->rank;", inp.0));
            self.line(&format!(
                "int64_t t{id}_{pfx}_size = d_t{src}->byte_capacity / chelis_dtype_size(d_t{src}->dtype);",
                src = inp.0
            ));
        }

        // Emit shape vars for output
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_out_ndim = d_t{id}->rank;"));

        // Build args array
        let mut arg_parts = Vec::new();
        for (i, inp) in inputs.iter().enumerate() {
            let pfx = format!("ext{i}");
            arg_parts.push(format!("&p_t{}", inp.0));
            arg_parts.push(self.stride_arg_refs(id, &pfx));
            arg_parts.push(format!("&t{id}_{pfx}_ndim"));
            arg_parts.push(format!("&t{id}_{pfx}_size"));
        }
        arg_parts.push(format!("&p_t{id}"));
        arg_parts.push(self.shape_arg_refs(id, "out"));
        arg_parts.push(format!("&t{id}_out_ndim"));
        arg_parts.push(format!("&t{id}_size"));

        self.line(&format!("void *args[] = {{ {} }};", arg_parts.join(", ")));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            kernel_name,
            &format!("t{id}_size / 256 + (t{id}_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    /// Emit the host-side wrapper for an in-place FusedElem launch on
    /// HIP. At runtime, if the reusable input's device storage is
    /// contiguous, the FusedElem output view is aliased onto the
    /// reusable input's device buffer (saving one slot's worth of GPU
    /// allocation + the eventual `hipFree`). Otherwise we fall back to
    /// the slot-backed view exactly as the non-in-place path would.
    ///
    /// Mirrors the C-side `emit_fused_in_place_wrapper`
    /// (`crates/chelis-backend-c/src/emit.rs::emit_fused_in_place_wrapper`)
    /// with one HIP-specific addition: in `device_entrypoint_mode` the
    /// slot declarations are pre-emitted by
    /// `emit_device_slot_allocations`, so the wrapper must not
    /// redeclare them. In host-entrypoint mode the C-side discipline
    /// applies: declare the slot variable on the first-owner path and
    /// defer allocation to inside the non-contiguous branch when the
    /// slot has no later owners.
    fn emit_fused_in_place_wrapper(
        &mut self,
        id: usize,
        ty: &TensorType,
        spec: FusedReuseMechanics,
    ) {
        let slot = self.slot_id_for_node(id);
        let first = self.plan.slot(slot).first_owner == NodeId(id);
        let reusable = spec.reusable_input.0;
        if !self.device_entrypoint_mode && first {
            if spec.slot_has_later_owner {
                self.emit_slot_allocation_if_needed(id, ty);
            } else {
                self.line(&format!(
                    "chelis_device_tensor_owner *chelis_slot{slot} = NULL;"
                ));
            }
        }
        self.emit_metadata_plan(&format!("plan_t{id}"), ty, None, "0");
        self.line(&format!(
            "bool contiguous_t{id} = d_t{reusable}->rank == chelis_metadata_plan_rank(plan_t{id});"
        ));
        self.line(&format!("for (int32_t axis = 0; contiguous_t{id} && axis < d_t{reusable}->rank; ++axis) contiguous_t{id} = d_t{reusable}->strides[axis] == chelis_metadata_plan_strides(plan_t{id})[axis];"));
        self.line(&format!("chelis_device_tensor_owner *o_t{id};"));
        self.line(&format!("if (contiguous_t{id}) {{"));
        self.indent += 1;
        self.line(&format!(
            "o_t{id} = chelis_device_tensor_borrow(plan_t{id}, d_t{reusable}->data, {});",
            Self::tagged_i64(&format!("d_t{reusable}->byte_capacity"))
        ));
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        if !self.device_entrypoint_mode && first && !spec.slot_has_later_owner {
            self.emit_metadata_plan(&format!("slot_plan{slot}"), ty, None, "0");
            self.line(&format!(
                "chelis_slot{slot} = chelis_device_tensor_alloc(slot_plan{slot});"
            ));
        }
        let source = format!("chelis_device_tensor_view(chelis_slot{slot})");
        self.line(&format!(
            "o_t{id} = chelis_device_tensor_borrow(plan_t{id}, {source}->data, {});",
            Self::tagged_i64(&format!("{source}->byte_capacity"))
        ));
        self.indent -= 1;
        self.line("}");
        self.emit_owner_observation(id);
    }

    // ------------------------------------------------------------------
    // Reduce kernel launch
    // ------------------------------------------------------------------

    // WS-A4: extra `accumulator` parameter is required so the launch
    // site computes the same dtype-suffixed kernel name the
    // source-emit side produces. The function already had 6 args
    // pre-WS-A4.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduce_launch(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
        kind: kernels::ReduceKind,
        // WS-A4: accumulator dtype for Sum reductions (None for Max
        // and other non-accumulator-carrying kinds). Used to compute
        // the dtype-suffixed kernel name so it matches the
        // kernel-source side.
        accumulator: Option<Prim>,
    ) -> Result<(), Unsupported> {
        let a = inputs[0].0;
        if matches!(kind, kernels::ReduceKind::Sum)
            && let Some(matmul) = blas::detect_matmul_pattern(dag, NodeId(id))
            && Self::supports_static_hipblas_matmul(dag, &matmul, ty)
        {
            // F1 footgun fix (WS-A3): the `Sum`-detected matmul carries
            // the operand precision in `ty`; resolve the spec-default
            // accumulator from it so the GEMM dispatch routes by the
            // correct (operand, accumulator) pair. The detected pattern
            // came from a Sum node, whose `accumulator` field is
            // already populated to the spec default during IR
            // construction; we use the same default here so detection
            // and lowering agree.
            let accumulator = RiscOp::default_matmul_accumulator(ty.precision)
                .expect("matmul detection only fires on operand types admitted by the matmul rule");
            self.emit_blas_matmul(
                id,
                &MatmulEmitSpec {
                    a: matmul.a,
                    b: matmul.b,
                    batch_dims: Vec::new(),
                    m: DimExpr::Concrete(matmul.m),
                    n: DimExpr::Concrete(matmul.n),
                    k: DimExpr::Concrete(matmul.k),
                    accumulator,
                },
                ty,
                dag,
            );
            return Ok(());
        }
        // WS-A2 + WS-A4: kernel-name dispatch must match the
        // kernel-source emission in `reduction_kernel_sources`. f32/f64
        // use the `ElemKind`-suffixed naming (so existing tests + ABI
        // stay byte-identical); the WS-A4 i8/i16 → i32 promoted path
        // uses the typed-name encoding so its kernel string is deduped
        // separately.
        let kernel_name = match (kind, accumulator) {
            (kernels::ReduceKind::Sum, Some(acc))
                if matches!(
                    dag.get(NodeId(a)).unwrap().output_type.precision,
                    Prim::Int8 | Prim::Int16
                ) && acc == Prim::Int32 =>
            {
                let src_prec = dag.get(NodeId(a)).unwrap().output_type.precision;
                Self::reduction_kernel_name_typed(kind, axis, src_prec, acc)
            }
            _ => Self::reduction_kernel_name(kind, axis, Self::elem_kind(ty)?),
        };

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int64_t t{id}_out_size = d_t{id}->count;"));
        self.line(&format!("int64_t t{id}_axis_size = d_t{a}->shape[{axis}];"));
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int64_t t{id}_a_ndim = d_t{a}->rank;"));
        self.line(&format!(
            "int64_t t{id}_a_size = (d_t{a}->byte_capacity / chelis_dtype_size(d_t{a}->dtype));"
        ));
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_out_ndim = d_t{id}->rank;"));
        self.line(&format!(
            "void *args[] = {{ &p_t{a}, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             &p_t{id}, {out_shape_refs}, &t{id}_out_ndim, &t{id}_out_size, &t{id}_axis_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            &kernel_name,
            &format!("t{id}_out_size / 256 + (t{id}_out_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    /// Launch shape mirrors `emit_reduce_launch` for the four reductions
    /// previously deferred to the C backend (Min / Prod / Argmax /
    /// Argmin). The kernel signature is identical: per-output thread,
    /// `axis_size` trailing parameter, in/out tensor pointers, strides
    /// and shapes packed into `args`.
    fn emit_extra_reduce_launch(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) -> Result<(), Unsupported> {
        let a = inputs[0].0;
        let input_ty = &dag.get(inputs[0]).unwrap().output_type;
        let node_op = &dag.get(NodeId(id)).unwrap().op;
        // All four extra reductions name kernels by the operand
        // precision: Min/Prod return the operand precision (so input ==
        // output), and Argmax/Argmin's output is i64 even though the
        // value being compared is the operand precision. Reading the
        // input precision uniformly avoids the elem_kind panic on the
        // i64 output of the arg-reductions.
        let elem_for_naming = Self::elem_kind(input_ty)?;
        let op_name = match node_op {
            RiscOp::MinReduce { .. } => "min",
            RiscOp::ProdReduce { .. } => "prod",
            RiscOp::Argmax { .. } => "argmax",
            RiscOp::Argmin { .. } => "argmin",
            _ => unreachable!("emit_extra_reduce_launch called on non-extra reduction"),
        };
        let kernel_name = Self::extra_reduction_kernel_name(op_name, axis, elem_for_naming);

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int64_t t{id}_out_size = d_t{id}->count;"));
        self.line(&format!("int64_t t{id}_axis_size = d_t{a}->shape[{axis}];"));
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int64_t t{id}_a_ndim = d_t{a}->rank;"));
        self.line(&format!(
            "int64_t t{id}_a_size = (d_t{a}->byte_capacity / chelis_dtype_size(d_t{a}->dtype));"
        ));
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_out_ndim = d_t{id}->rank;"));
        self.line(&format!(
            "void *args[] = {{ &p_t{a}, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             &p_t{id}, {out_shape_refs}, &t{id}_out_ndim, &t{id}_out_size, &t{id}_axis_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            &kernel_name,
            &format!("t{id}_out_size / 256 + (t{id}_out_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    // ------------------------------------------------------------------
    // Fused reduce kernel launch (elementwise chain inlined into reduction)
    // ------------------------------------------------------------------

    fn emit_fused_reduce_launch(
        &mut self,
        id: usize,
        axis: usize,
        reduction_inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let fused_node = dag.get(reduction_inputs[0]).unwrap();
        let ext_inputs = &fused_node.inputs;
        // WS-A4: bind `accumulator` so the fused-reduce path explicitly
        // names the field. The fused-reduce kernel template is f32-only
        // in this cycle (its source comes from `kernels::reduce_fused`
        // which still hardcodes `float`); when that template grows
        // dtype variants the `_acc` binding here is the hook that wires
        // accumulator into the fused-reduce kernel name.
        let (kind, _acc) = match dag.get(NodeId(id)).unwrap().op {
            RiscOp::Sum { accumulator, .. } => (kernels::ReduceKind::Sum, Some(accumulator)),
            RiscOp::MaxReduce { .. } => (kernels::ReduceKind::Max, None),
            _ => unreachable!("emit_fused_reduce_launch called on non-reduction"),
        };
        let kernel_name = Self::fused_reduction_kernel_name(id, kind);
        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int64_t t{id}_out_size = d_t{id}->count;"));
        self.line(&format!(
            "int64_t t{id}_axis_size = d_t{}->shape[{axis}];",
            reduction_inputs[0].0
        ));

        // Emit stride vars for each external input of the fused chain
        for (i, inp) in ext_inputs.iter().enumerate() {
            let pfx = format!("ext{i}");
            self.emit_stride_vars(id, &pfx, inp.0);
            self.line(&format!("int64_t t{id}_{pfx}_ndim = d_t{}->rank;", inp.0));
            self.line(&format!(
                "int64_t t{id}_{pfx}_size = d_t{src}->byte_capacity / chelis_dtype_size(d_t{src}->dtype);",
                src = inp.0
            ));
        }

        // Emit shape vars for output
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_out_ndim = d_t{id}->rank;"));

        // Build args array: ext inputs + output
        let mut arg_parts = Vec::new();
        for (i, inp) in ext_inputs.iter().enumerate() {
            let pfx = format!("ext{i}");
            arg_parts.push(format!("&p_t{}", inp.0));
            arg_parts.push(self.stride_arg_refs(id, &pfx));
            arg_parts.push(format!("&t{id}_{pfx}_ndim"));
            arg_parts.push(format!("&t{id}_{pfx}_size"));
        }
        arg_parts.push(format!("&p_t{id}"));
        arg_parts.push(self.shape_arg_refs(id, "out"));
        arg_parts.push(format!("&t{id}_out_ndim"));
        arg_parts.push(format!("&t{id}_out_size"));
        arg_parts.push(format!("&t{id}_axis_size"));

        self.line(&format!("void *args[] = {{ {} }};", arg_parts.join(", ")));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            &kernel_name,
            &format!("t{id}_out_size / 256 + (t{id}_out_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    #[allow(dead_code)]
    fn emit_blas_matmul(
        &mut self,
        id: usize,
        spec: &MatmulEmitSpec,
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        // WS-A2 / WS-A3: dispatch by `(operand, accumulator)` pair (see
        // the match below). The historical F32-only assertion that lived
        // here was lifted by WS-A2 (admits f64) and WS-A3 (admits
        // bf16/f16 via hipblasGemmEx); upstream guards in verify.rs and
        // validate_supported_precisions reject any unsupported
        // accumulator dtype before this function is called.
        let a = spec.a.0;
        let b = spec.b.0;
        let matrix_axis = ty.dims.len() - 2;
        let m_expr = format!("d_t{id}->shape[{matrix_axis}]");
        let n_expr = format!("d_t{id}->shape[{}]", matrix_axis + 1);
        let k_expr = format!("d_t{a}->shape[d_t{a}->rank - 1]");
        // F1 footgun fix (WS-A3): dispatch by `(operand, accumulator)`
        // pair, not by operand alone. The original implementation
        // unconditionally called `hipblasSgemm` regardless of the
        // accumulator field — which silently miscompiled non-f32
        // operands. The dispatch table below mirrors the wrapper
        // surface in `crates/chelis-backend-hip/runtime/chelis_hip_runtime.h`.
        let operand = ty.precision;
        let acc = spec.accumulator;
        let wrapper = match (operand, acc) {
            (Prim::F32, Prim::F32) => MatmulWrapper::Sgemm,
            (Prim::F64, Prim::F64) => MatmulWrapper::Dgemm,
            (Prim::Bf16, Prim::F32) => MatmulWrapper::Bf16GemmF32,
            (Prim::F16, Prim::F32) => MatmulWrapper::F16GemmF32,
            // Wider-than-default accumulator on bf16/f16 (per
            // spec/04-type-system.md §5.7.1 the request is admissible
            // because f64 is wider than the f32 default) requires an
            // operand-promotion compute path with explicit f64
            // intermediate buffers; that path is not yet wired even
            // after WS-A2 lands `chelis_hipblas_dgemm`. Reject at
            // codegen with a clean diagnostic rather than silently
            // downgrading the user's accumulator request to f32.
            (Prim::Bf16, _) | (Prim::F16, _) => panic!(
                "F1: HIP BlasMatmul on operand `{}` with accumulator `{}` requires a \
                 wider-than-default GEMM compute path (operand promotion to `{}`); \
                 the corresponding mixed-precision wrapper is not implemented in \
                 this cycle. Omit the accumulator parameter to accept the \
                 spec-default `f32` per spec/04-type-system.md §5.7.1.",
                operand.name(),
                acc.name(),
                acc.name(),
            ),
            (other, _) => panic!(
                "F1: HIP BlasMatmul on operand precision `{}` is not yet supported \
                 by the HIP backend (accumulator `{}`); spec/04-type-system.md \
                 §5.7.1 documents the per-precision dispatch. Reachable only via \
                 the F1 partial-lift escape; tighten `emit_dag`'s F1 guard.",
                other.name(),
                acc.name(),
            ),
        };
        self.emit_slot_wrapper(id, ty);

        // WS-A2: dispatch on operand precision. The IR `BlasMatmul` node
        // also carries a pinned `accumulator: Prim` field per WS-A0 spec;
        // for f32 and f64 the accumulator is operand-matching by default
        // (§5.7.1) and the host-side hipBLAS helpers compute
        // alpha/beta/A/B/C in the operand precision. The per-precision
        // accumulator is read here so that auditors can see that the
        // field is consumed, not destructured-and-ignored as it was
        // before WS-A0 RT-1 (the F1 footgun).
        let node = dag.get(NodeId(id)).expect("matmul node exists");
        let (operand_prec, declared_acc) = match &node.op {
            RiscOp::BlasMatmul { accumulator, .. } => (ty.precision, *accumulator),
            // emit_blas_matmul is also invoked from the matmul-pattern
            // detector (Sum-of-Mul-of-Expand-Expand) — that path lowered
            // through reduce_sum which carries its own accumulator. For
            // f32/f64 operands the spec default makes them coincide with
            // ty.precision, so we read it here too rather than re-deriving.
            _ => (ty.precision, ty.precision),
        };
        // Note: the `operand_prec`/`declared_acc` consistency check
        // proper has already been encoded in the (operand, acc) match
        // above that built `wrapper` — bf16/f16 with non-f32
        // accumulators panic'd there with a spec-shaped diagnostic.
        // The destructure here exists to keep the audit trail showing
        // that the BlasMatmul.accumulator field is read, not ignored
        // (the WS-A0 RT-1 F1 footgun was destructure-and-ignore).
        let _ = (operand_prec, declared_acc);

        self.line(&format!("if (d_t{id}->count != 0) {{"));
        self.indent += 1;
        if spec.batch_dims.is_empty() {
            self.line(&format!(
                "{call}(d_t{a}, d_t{b}, d_t{id}, {m_expr}, {n_expr}, {k_expr});",
                call = wrapper.row_major_call_name(),
            ));
        } else if Self::strided_batched_hipblas_plan(dag, spec, ty).is_some() {
            // Batched/strided-batched bf16/f16 wrappers are not yet
            // present (would need `hipblasGemmStridedBatchedEx`
            // plumbing). For this cycle, batched matmul is
            // f32 (Sgemm) or f64 (Dgemm) only.
            let call = wrapper.strided_batched_row_major_call_name().expect(
                "WS-A3: batched bf16/f16 matmul is not yet wired (would require \
                 `hipblasGemmStridedBatchedEx`); rank-2 only in this cycle",
            );
            let last_batch_axis = matrix_axis - 1;
            self.line(&format!(
                "int64_t t{id}_batch_count = d_t{id}->count / {m_expr} / {n_expr};"
            ));
            self.line(&format!("{call}(d_t{a}, d_t{b}, d_t{id}, {m_expr}, {n_expr}, {k_expr}, t{id}_batch_count, d_t{a}->strides[{last_batch_axis}], d_t{b}->strides[{last_batch_axis}], d_t{id}->strides[{last_batch_axis}]);"));
        } else {
            let call = wrapper.batched_row_major_call_name().expect(
                "WS-A3: batched bf16/f16 matmul is not yet wired (would require \
                 `hipblasGemmBatchedEx`); rank-2 only in this cycle",
            );
            self.line(&format!(
                "{call}(d_t{a}, d_t{b}, d_t{id}, {m_expr}, {n_expr}, {k_expr});"
            ));
        }
        self.indent -= 1;
        self.line("}");
    }

    // ------------------------------------------------------------------
    // Movement ops (host-side metadata, no kernel)
    // ------------------------------------------------------------------

    fn emit_reshape(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let kernel_name = format!("kernel_reshape_{}", Self::dtype_macro(ty));
        self.emit_logical_copy(id, inputs, ty, &kernel_name);
    }

    fn emit_logical_copy(
        &mut self,
        id: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        kernel_name: &str,
    ) {
        let a = inputs[0].0;
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("if (d_t{id}->count != d_t{a}->count) chelis_numeric_trap(\"numeric trap: domain in materialize at int64\");"));
        self.line("{");
        self.indent += 1;
        self.emit_stride_vars(id, "a", a);
        self.emit_shape_vars(id, "a", a);
        self.line(&format!("int64_t t{id}_a_ndim = d_t{a}->rank;"));
        self.line(&format!("int64_t t{id}_size = d_t{id}->count;"));
        self.line(&format!(
            "void *args[] = {{ &p_t{a}, {}, {}, &t{id}_a_ndim, &p_t{id}, &t{id}_size }};",
            self.stride_arg_refs(id, "a"),
            self.shape_arg_refs(id, "a")
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            kernel_name,
            &format!("t{id}_size / 256 + (t{id}_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    fn emit_permute(&mut self, id: usize, axes: &[usize], inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let strides = axes
            .iter()
            .map(|axis| format!("d_t{a}->strides[{axis}]"))
            .collect::<Vec<_>>();
        self.emit_strided_view(id, ty, &format!("d_t{a}"), &strides);
    }

    fn emit_expand(
        &mut self,
        id: usize,
        axis: usize,
        _size: &RtDim,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        let rank = ty.dims.len();
        let strides = (0..rank)
            .map(|d| {
                if d == axis {
                    "0".to_string()
                } else if d < axis {
                    format!("d_t{a}->strides[{d}]")
                } else {
                    format!("d_t{a}->strides[d_t{a}->rank == {rank} ? {d} : {}]", d - 1)
                }
            })
            .collect::<Vec<_>>();
        self.emit_strided_view(id, ty, &format!("d_t{a}"), &strides);
    }

    fn emit_stride(
        &mut self,
        id: usize,
        stride_factors: &[usize],
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        assert_eq!(stride_factors.len(), ty.dims.len());
        let strides = stride_factors.iter().enumerate().map(|(axis, factor)| {
            format!("chelis_int_checked_mul(d_t{a}->strides[{axis}], INT64_C({factor}), 64, \"numeric trap: overflow in stride at int64\")")
        }).collect::<Vec<_>>();
        self.emit_strided_view(id, ty, &format!("d_t{a}"), &strides);
    }

    /// Emit per-axis `int64_t t{node_id}_{prefix}{0..7} = <val>;` constants
    /// from a compile-time vector, zero-padding the unused trailing axes.
    /// Used by `pad`/`shrink` for the low-padding / start-offset vectors,
    /// which are pinned in the `RiscOp` (not read from runtime metadata).
    fn emit_axis_const_vars(&mut self, node_id: usize, prefix: &str, values: &[usize]) {
        for i in 0..self.kernel_rank {
            let v = values.get(i).copied().unwrap_or(0);
            self.line(&format!("int64_t t{node_id}_{prefix}{i} = {v};"));
        }
    }

    /// Generate `&t{node_id}_{prefix}0, ...` arg references for the
    /// per-axis constants emitted by [`Self::emit_axis_const_vars`].
    fn axis_const_arg_refs(&self, node_id: usize, prefix: &str) -> String {
        (0..self.kernel_rank)
            .map(|i| format!("&t{node_id}_{prefix}{i}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Launch the typed `pad` kernel. The output slot is materialized
    /// contiguous; one thread per output element scatters from the source
    /// (per-axis low offset) or writes the `fill` value when the output
    /// cell lies in the padded margin. Source shape comes from runtime
    /// metadata (`d_t{a}->shape`) so a strided/symbolic source still
    /// bounds-checks correctly; the low offsets are the compile-time
    /// `padding[d].0`.
    #[allow(clippy::too_many_arguments)]
    fn emit_pad_launch(
        &mut self,
        id: usize,
        padding: &[(usize, usize)],
        fill: ScalarValue,
        kernel_name: &str,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let prec = ty.precision;
        debug_assert_eq!(
            prec,
            dag.get(inputs[0]).unwrap().output_type.precision,
            "pad must preserve precision"
        );
        let lo: Vec<usize> = padding.iter().map(|&(before, _)| before).collect();
        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int64_t t{id}_size = d_t{id}->count;"));
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int64_t t{id}_a_ndim = d_t{a}->rank;"));
        self.line(&format!(
            "int64_t t{id}_a_size = (d_t{a}->byte_capacity / chelis_dtype_size(d_t{a}->dtype));"
        ));
        self.emit_axis_const_vars(id, "lo", &lo);
        // Source shape is read from runtime metadata so symbolic/strided
        // inputs bounds-check against their real extents.
        self.emit_shape_vars(id, "srcsh", a);
        self.emit_typed_scalar_local(&format!("t{id}_fill"), fill);
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_out_ndim = d_t{id}->rank;"));
        self.line(&format!(
            "void *args[] = {{ &p_t{a}, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             {lo_refs}, {srcsh_refs}, &t{id}_fill, \
             &p_t{id}, {out_shape_refs}, &t{id}_out_ndim, &t{id}_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            lo_refs = self.axis_const_arg_refs(id, "lo"),
            srcsh_refs = self.shape_arg_refs(id, "srcsh"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            kernel_name,
            &format!("t{id}_size / 256 + (t{id}_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    /// Launch the typed `shrink` kernel. One thread per (contiguous)
    /// output element reads the source at `out_index + start` where
    /// `start` is the compile-time `bounds[d].0`. The shrink output is
    /// always strictly inside the source, so no bounds margin exists.
    fn emit_shrink_launch(
        &mut self,
        id: usize,
        bounds: &[(usize, usize)],
        kernel_name: &str,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        debug_assert_eq!(
            ty.precision,
            dag.get(inputs[0]).unwrap().output_type.precision,
            "shrink must preserve precision"
        );
        let start: Vec<usize> = bounds.iter().map(|&(s, _)| s).collect();
        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int64_t t{id}_size = d_t{id}->count;"));
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int64_t t{id}_a_ndim = d_t{a}->rank;"));
        self.line(&format!(
            "int64_t t{id}_a_size = (d_t{a}->byte_capacity / chelis_dtype_size(d_t{a}->dtype));"
        ));
        self.emit_axis_const_vars(id, "start", &start);
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int64_t t{id}_out_ndim = d_t{id}->rank;"));
        self.line(&format!(
            "void *args[] = {{ &p_t{a}, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             {start_refs}, \
             &p_t{id}, {out_shape_refs}, &t{id}_out_ndim, &t{id}_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            start_refs = self.axis_const_arg_refs(id, "start"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            kernel_name,
            &format!("t{id}_size / 256 + (t{id}_size % 256 != 0)"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    /// Emit a typed scalar local from its finalized, dtype-tagged value.
    fn emit_typed_scalar_local(&mut self, name: &str, value: ScalarValue) {
        match value.element_ref() {
            ElementRef::F32(value) => {
                let bits = value.to_bits();
                self.line(&format!(
                    "float {name} = chelis_f32_from_bits(0x{bits:08x}u);"
                ));
            }
            ElementRef::F64(value) => {
                let bits = value.to_bits();
                self.line(&format!(
                    "double {name} = chelis_f64_from_bits(0x{bits:016x}uLL);"
                ));
            }
            ElementRef::F16(value) => self.line(&format!(
                "uint16_t {name} = UINT16_C(0x{:04x});",
                value.to_bits()
            )),
            ElementRef::Bf16(value) => self.line(&format!(
                "uint16_t {name} = UINT16_C(0x{:04x});",
                value.to_bits()
            )),
            ElementRef::I8(value) => self.line(&format!("int8_t {name} = INT8_C({value});")),
            ElementRef::I16(value) => self.line(&format!("int16_t {name} = INT16_C({value});")),
            ElementRef::I32(value) => self.line(&format!("int32_t {name} = INT32_C({value});")),
            ElementRef::I64(value) => {
                self.line(&format!("int64_t {name} = {};", Self::i64_c_literal(value)))
            }
            ElementRef::Bool(value) => self.line(&format!(
                "float {name} = {};",
                if value { "1.0f" } else { "0.0f" }
            )),
        }
    }

    fn emit_store(&mut self, id: usize, name: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        self.emit_alias_view(id, ty, &format!("d_t{a}"));
        // Producer-supplied Store name in a `/* ... */` block-comment
        // context. LoadStoreName grammar already excludes `*` and `/`,
        // so a `*/` cannot reach this format!() via the constructor;
        // the deserialize bypass remains a hypothetical second-layer
        // concern but is out of scope for the comment-control-byte
        // sanitizer (which targets the line-comment context). The
        // control-byte sanitizer below still applies as defense in
        // depth per spec/upstream-bugs/producer-string-sanitization.md.
        let safe_name = chelis_ir::span_sanitize::sanitize_for_comment(name);
        self.line(&format!("/* store: {safe_name} */"));
    }

    // ------------------------------------------------------------------
    // Stride/shape variable helpers
    // ------------------------------------------------------------------

    /// Emit `int64_t t{node_id}_{prefix}_s{0..7} = d_t{src_id}->strides[i];` for MAX_DIM dims.
    fn emit_stride_vars(&mut self, node_id: usize, prefix: &str, src_id: usize) {
        for i in 0..self.kernel_rank {
            self.line(&format!(
                "int64_t t{node_id}_{prefix}_s{i} = ({i} < d_t{src_id}->rank) ? d_t{src_id}->strides[{i}] : 0;"
            ));
        }
    }

    /// Emit `int64_t t{node_id}_{prefix}_sh{0..7} = d_t{src_id}->shape[i];` for MAX_DIM dims.
    fn emit_shape_vars(&mut self, node_id: usize, prefix: &str, src_id: usize) {
        for i in 0..self.kernel_rank {
            self.line(&format!(
                "int64_t t{node_id}_{prefix}_sh{i} = ({i} < d_t{src_id}->rank) ? d_t{src_id}->shape[{i}] : 0;"
            ));
        }
    }

    /// Generate `&t{node_id}_{prefix}_s0, &t{node_id}_{prefix}_s1, ...` for args array.
    fn stride_arg_refs(&self, node_id: usize, prefix: &str) -> String {
        (0..self.kernel_rank)
            .map(|i| format!("&t{node_id}_{prefix}_s{i}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Generate `&t{node_id}_{prefix}_sh0, &t{node_id}_{prefix}_sh1, ...` for args array.
    fn shape_arg_refs(&self, node_id: usize, prefix: &str) -> String {
        (0..self.kernel_rank)
            .map(|i| format!("&t{node_id}_{prefix}_sh{i}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn emit_kernel_launch_expr(
        &mut self,
        module_var: &str,
        kernel_name: &str,
        grid_expr: &str,
        block_expr: &str,
        args_var: &str,
    ) {
        self.line(&format!("chelis_prepare_kernel_launch({module_var});"));
        self.line(&format!(
            "chelis_launch_kernel({module_var}, \"{kernel_name}\", ({grid_expr}), ({block_expr}), {args_var});"
        ));
        self.line(&format!(
            "chelis_finalize_kernel_launch({module_var}, \"{kernel_name}\");"
        ));
    }

    fn reduction_kernel_name(
        kind: kernels::ReduceKind,
        axis: usize,
        elem: kernels::ElemKind,
    ) -> String {
        let op = match kind {
            kernels::ReduceKind::Sum => "sum",
            kernels::ReduceKind::Max => "maxred",
        };
        format!("kernel_{op}_ax{axis}_{}", elem.suffix())
    }

    /// Kernel-name helper for the four reductions the HIP backend
    /// previously deferred to the C backend (Min / Prod / Argmax /
    /// Argmin). `op` is the short name (e.g. "min", "argmax"); `elem` is
    /// the operand precision.
    fn extra_reduction_kernel_name(op: &str, axis: usize, elem: kernels::ElemKind) -> String {
        format!("kernel_{op}_ax{axis}_{}", elem.suffix())
    }

    /// WS-A4: dtype-aware variant of `reduction_kernel_name` for
    /// reductions that need to specialize on either source precision
    /// or accumulator precision. The unsuffixed name is preserved when
    /// `src == acc == f32` so the legacy emit and existing tests
    /// (`reduce_sum_kernel_has_axis_loop`, `gpu_correctness::*`) stay
    /// byte-identical to their pre-WS-A4 output. Non-f32 paths get a
    /// `_<src>_<acc>` suffix so the i8/i16 → i32 promoted-accumulator
    /// kernels are deduplicated separately from any future i32 → i32
    /// or f64 → f64 specialization.
    fn reduction_kernel_name_typed(
        kind: kernels::ReduceKind,
        axis: usize,
        src: Prim,
        acc: Prim,
    ) -> String {
        let op = match kind {
            kernels::ReduceKind::Sum => "sum",
            kernels::ReduceKind::Max => "maxred",
        };
        if src == Prim::F32 && acc == Prim::F32 {
            format!("kernel_{op}_ax{axis}")
        } else {
            format!(
                "kernel_{op}_ax{axis}{src_sfx}{acc_sfx}",
                src_sfx = Self::dtype_kernel_suffix(src),
                acc_sfx = Self::dtype_kernel_suffix(acc),
            )
        }
    }

    fn fused_reduction_kernel_name(id: usize, kind: kernels::ReduceKind) -> String {
        let op = match kind {
            kernels::ReduceKind::Sum => "fused_sum",
            kernels::ReduceKind::Max => "fused_maxred",
        };
        format!("kernel_{op}_{id}")
    }

    /// Predicate: is the matmul-pattern subgraph dispatchable via the
    /// statically-shaped hipBLAS GEMM helper? Both f32 and f64 operands
    /// are admitted (WS-A2). Operands and result must share precision —
    /// the WS-A0 accumulator parameter is checked at the call site.
    #[allow(dead_code)]
    fn supports_static_hipblas_matmul(
        dag: VerifiedDagView<'_>,
        info: &blas::MatmulInfo,
        ty: &TensorType,
    ) -> bool {
        // WS-A2 + WS-A3: f32 / f64 (Sgemm/Dgemm) and bf16 / f16
        // (`hipblasGemmEx` with `HIPBLAS_COMPUTE_32F` per
        // spec/04-type-system.md §5.7.1) all admitted by the static
        // hipBLAS GEMM helper. Wider-accumulator requests (bf16/f16
        // + f64 accumulator) hit the rejection in `emit_blas_matmul`,
        // not here, so the detection still admits them and the caller
        // gets a precise diagnostic instead of a silent fall-back
        // into the elementwise reduce path.
        let dtype_admits = matches!(ty.precision, Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16);
        dtype_admits
            && ty.dims.len() == 2
            && dag.get(info.a).map(|n| n.output_type.precision) == Some(ty.precision)
            && dag.get(info.b).map(|n| n.output_type.precision) == Some(ty.precision)
            && Self::node_is_statically_contiguous(dag, info.a)
            && Self::node_is_statically_contiguous(dag, info.b)
    }

    fn strided_batched_hipblas_plan(
        dag: VerifiedDagView<'_>,
        spec: &MatmulEmitSpec,
        ty: &TensorType,
    ) -> Option<()> {
        if spec.batch_dims.is_empty()
            || !matches!(ty.precision, Prim::F32 | Prim::F64)
            || !spec.batch_dims.iter().all(Self::is_simple_runtime_dim)
        {
            return None;
        }

        spec.m.as_concrete()?;
        spec.n.as_concrete()?;
        spec.k.as_concrete()?;
        let batch_rank = spec.batch_dims.len();
        if ty.dims.len() != batch_rank + 2 {
            return None;
        }

        let a_node = dag.get(spec.a)?;
        let b_node = dag.get(spec.b)?;
        let prec = ty.precision;
        if a_node.output_type.precision != prec
            || b_node.output_type.precision != prec
            || a_node.output_type.dims.len() != batch_rank + 2
            || b_node.output_type.dims.len() != batch_rank + 2
            || !Self::node_is_statically_contiguous(dag, spec.a)
            || !Self::node_is_statically_contiguous(dag, spec.b)
        {
            return None;
        }

        let expected_out = spec
            .batch_dims
            .iter()
            .cloned()
            .chain([spec.m.clone(), spec.n.clone()])
            .collect::<Vec<_>>();
        let expected_a = spec
            .batch_dims
            .iter()
            .cloned()
            .chain([spec.m.clone(), spec.k.clone()])
            .collect::<Vec<_>>();
        let expected_b = spec
            .batch_dims
            .iter()
            .cloned()
            .chain([spec.k.clone(), spec.n.clone()])
            .collect::<Vec<_>>();
        let out_dims = ty.dims.iter().map(DimExpr::from).collect::<Vec<_>>();
        let a_dims = a_node
            .output_type
            .dims
            .iter()
            .map(DimExpr::from)
            .collect::<Vec<_>>();
        let b_dims = b_node
            .output_type
            .dims
            .iter()
            .map(DimExpr::from)
            .collect::<Vec<_>>();
        if out_dims != expected_out || a_dims != expected_a || b_dims != expected_b {
            return None;
        }

        Some(())
    }

    fn is_simple_runtime_dim(expr: &DimExpr) -> bool {
        matches!(expr, DimExpr::Concrete(_) | DimExpr::Sym(_))
    }

    /// The [05-OP-6] rung has no guarded device kernel, so it never
    /// reaches codegen: `reject_unsupported_hip_ops` gates it first.
    /// These arms exist so a future HIP implementation has to remove
    /// this rejection deliberately rather than inherit `cast`'s
    /// unguarded conversion by accident.
    fn remainder_unsupported(node: &DagNode) -> Unsupported {
        Unsupported::new(
            UnsupportedKind::Op("mod".to_string()),
            format!("the HIP kernel set (node {})", node.id.0),
            Stage::Codegen("hip"),
            chelis_types::unimplemented_rejection!(
                1277,
                "checked integer remainder has no device kernel; the C host path preserves [05-OP-64] DivZero traps"
            ),
        )
    }

    fn cast_trunc_unsupported(node: &DagNode) -> Unsupported {
        Unsupported::new(
            UnsupportedKind::Op("cast_trunc".to_string()),
            format!("the HIP kernel set (node {})", node.id.0),
            Stage::Codegen("hip"),
            chelis_types::unimplemented_rejection!(
                759,
                "the HIP cast kernels emit an unguarded device-side conversion, \
                 so the [05-OP-6] Domain/Overflow traps have no device \
                 implementation; the C target is canonical for the named cast ladder"
            ),
        )
    }

    #[allow(dead_code)]
    fn node_is_statically_contiguous(dag: VerifiedDagView<'_>, id: NodeId) -> bool {
        match &dag.get(id).unwrap().op {
            RiscOp::Load { .. } => false,
            RiscOp::Const { .. }
            | RiscOp::ConstTensor { .. }
            | RiscOp::Add
            | RiscOp::Sub
            | RiscOp::Mul
            | RiscOp::Div
            | RiscOp::FloorDiv
            | RiscOp::TruncDiv
            | RiscOp::Mod
            | RiscOp::MaxElem
            | RiscOp::MinElem
            | RiscOp::ExtremaAdjoint { .. }
            | RiscOp::Relu
            | RiscOp::ReluAdjoint
            | RiscOp::CmpLt
            | RiscOp::Neg
            | RiscOp::Recip
            | RiscOp::Exp
            | RiscOp::Log
            | RiscOp::Sin
            | RiscOp::Sqrt
            | RiscOp::Cos
            | RiscOp::Tan
            | RiscOp::Atan
            | RiscOp::Abs
            | RiscOp::Floor
            | RiscOp::Ceil
            | RiscOp::Round
            | RiscOp::UniformLike { .. }
            | RiscOp::Dropout { .. }
            | RiscOp::Copy
            | RiscOp::Drop
            | RiscOp::Sum { .. }
            | RiscOp::MaxReduce { .. }
            | RiscOp::MinReduce { .. }
            | RiscOp::ProdReduce { .. }
            | RiscOp::ReduceWindow { .. }
            | RiscOp::ReduceWindowGrad { .. }
            | RiscOp::Argmax { .. }
            | RiscOp::Argmin { .. }
            | RiscOp::OneHot { .. }
            | RiscOp::Realize
            | RiscOp::Reshape { .. }
            | RiscOp::Cast { .. }
            | RiscOp::CastTrunc { .. }
            | RiscOp::FusedElem { .. }
            | RiscOp::BlasMatmul { .. }
            | RiscOp::Gather { .. }
            | RiscOp::ScatterAdd { .. }
            | RiscOp::Scatter { .. }
            | RiscOp::ScatterElements { .. }
            // `pad` / `shrink` now materialize a fresh dense contiguous
            // slot via their kernels (one thread per output element into
            // the contiguous output buffer), so the result is statically
            // contiguous like any other kernel output.
            | RiscOp::Pad { .. }
            | RiscOp::Shrink { .. }
            // `Shape` materializes a fresh rank-0 scalar (trivially
            // contiguous). It is HIP-rejected before codegen under
            // [05-SHAPE-1], so this arm is only for classification completeness.
            | RiscOp::Shape { .. } | RiscOp::ExtentWitness { .. } | RiscOp::CheckedReshapeExtent { .. } | RiscOp::CheckedUnitAxis { .. } => true,
            RiscOp::Count { .. } => true,
            RiscOp::Store { .. } => {
                Self::node_is_statically_contiguous(dag, dag.get(id).unwrap().inputs[0])
            }
            RiscOp::Permute { .. } | RiscOp::Expand { .. } | RiscOp::Stride { .. } => false,
        }
    }

    // ------------------------------------------------------------------
    // Utility methods (mirrored from C backend)
    // ------------------------------------------------------------------

    fn line(&mut self, s: &str) {
        let prefix = "    ".repeat(self.indent);
        self.lines.push(format!("{prefix}{s}"));
    }

    /// Emit `// span: <id>` host-side comment lines for a node's
    /// `span_id ∪ merged_spans`. Per `spec/design/chelis_span_survival.md`
    /// §2.4 (S4): canonical first, then `merged_spans` lex-sorted (deduped
    /// against `span_id`). No-op when both fields are empty.
    fn emit_span_comments(&mut self, node: &DagNode) {
        for line in Self::span_comment_block(node) {
            self.line(&line);
        }
    }

    /// Build the deduped, lex-sorted `// span:` comment block for a node.
    /// Returns a vector of comment strings (each one a single line, no
    /// indent prefix). Used both by host-side emission (via
    /// `emit_span_comments`) and by per-node device-kernel string
    /// emission (where the comments are prepended inside the embedded
    /// kernel source so they survive into the runtime-compiled HIP).
    fn span_comment_block(node: &DagNode) -> Vec<String> {
        let mut out = Vec::new();
        if node.span_id.is_none() && node.merged_spans.is_empty() {
            return out;
        }
        if let Some(canonical) = node.span_id.as_deref() {
            let safe = chelis_ir::span_sanitize::sanitize_for_comment(canonical);
            out.push(format!("// span: {safe}"));
        }
        let mut merged: Vec<&str> = node
            .merged_spans
            .iter()
            .map(String::as_str)
            .filter(|s| node.span_id.as_deref() != Some(*s))
            .collect();
        merged.sort();
        merged.dedup();
        for span in merged {
            let safe = chelis_ir::span_sanitize::sanitize_for_comment(span);
            out.push(format!("// span: {safe}"));
        }
        out
    }

    /// Prepend `// span:` comment lines (followed by a newline) to a
    /// kernel source string. Returns the augmented source. No-op when the
    /// node carries no spans. Used for per-node kernels (FusedElem,
    /// fused reductions) where the kernel string is unique to the
    /// originating DAG node.
    fn prepend_span_comments_to_kernel_source(node: &DagNode, source: String) -> String {
        let block = Self::span_comment_block(node);
        if block.is_empty() {
            return source;
        }
        let mut out = String::new();
        for line in block {
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str(&source);
        out
    }

    fn ndim(ty: &TensorType) -> usize {
        ty.dims.len()
    }

    fn known_dim_size(dim: &DimInfo) -> Option<usize> {
        match dim {
            DimInfo::Lit(n) => Some(*n),
            DimInfo::Named(_, Some(n)) => Some(*n),
            DimInfo::Named(_, None) => None,
        }
    }

    fn emit_dim_info(dim: &DimInfo) -> String {
        match dim {
            DimInfo::Lit(n) => n.to_string(),
            DimInfo::Named(_, Some(n)) => n.to_string(),
            DimInfo::Named(name, None) => name.clone(),
        }
    }

    fn dtype_macro(ty: &TensorType) -> &'static str {
        ty.precision
            .runtime_dtype()
            .unwrap_or_else(|error| panic!("HIP backend does not support this tensor: {error}"))
            .c_macro()
    }

    /// WS-A4: kernel-name suffix encoding the source-level dtype, used
    /// when generating dtype-specialized HIP kernels. `f32` keeps an
    /// empty suffix so the legacy kernel names (`kernel_add`,
    /// `kernel_sum_ax0`, …) survive byte-identical to their pre-WS-A4
    /// emit; non-f32 dtypes get `_<short>` (e.g. `_i8`, `_i16`,
    /// `_i32`). Pin the mapping in one place so the kernel-name path
    /// and the kernel-source path agree on what each suffix means.
    fn dtype_kernel_suffix(p: Prim) -> &'static str {
        match p {
            Prim::F32 => "",
            Prim::F64 => "_f64",
            Prim::F16 => "_f16",
            Prim::Bf16 => "_bf16",
            Prim::Bool => "_bool",
            Prim::Int8 => "_i8",
            Prim::Int16 => "_i16",
            Prim::Int32 => "_i32",
            Prim::Int64 => "_i64",
            other => panic!(
                "HIP kernel suffix not defined for `{}` (active dtype set per \
                 spec/04-type-system.md §1.1: f32/f64/bf16/f16/bool/int8/int16/int32/int64).",
                other.name()
            ),
        }
    }

    /// Reject a node whose result is stored at a different width than its
    /// operands.
    ///
    /// [`kernels::cmplt`] and its relatives declare **one** element type and
    /// use it for both the operand pointers and the result pointer, so the
    /// emitted kernel writes `result_count * operand_width` bytes into a
    /// buffer the runtime sized at `result_count * result_width`. That is
    /// only safe when the two widths agree.
    ///
    /// chelis#1360 is what it costs when they stop agreeing silently. `cmplt`
    /// dispatched on its operand dtype alone and never asked about its bool
    /// result, which was fine while the HIP runtime stored bool as a four-byte
    /// `1.0f`/`0.0f` payload. chelis#1308's tagged carrier made bool one byte,
    /// nothing here noticed, and `x < y` began overrunning its device
    /// allocation by `3N` bytes while decoding the low bytes of the float
    /// stream on readback.
    ///
    /// Both widths come from [`chelis_vocab::Repr`] rather than a local table,
    /// so this check cannot drift away from what the allocator actually does.
    fn require_result_width_matches_operand(
        node: &DagNode,
        operand_ty: &TensorType,
    ) -> Result<(), Unsupported> {
        let width = |ty: &TensorType| {
            ty.precision
                .runtime_dtype()
                .map(|dtype| dtype.byte_width())
                .ok()
        };
        let (Some(result_width), Some(operand_width)) =
            (width(&node.output_type), width(operand_ty))
        else {
            // A dtype with no runtime representation at all is the ordinary
            // unsupported-dtype path; let `elem_kind` name it.
            return Ok(());
        };
        if result_width == operand_width {
            return Ok(());
        }
        Err(Unsupported::new(
            UnsupportedKind::Dtype(node.output_type.precision.name().to_string()),
            "a HIP kernel template that stores its result at the operand width",
            Stage::Codegen("hip"),
            chelis_types::unimplemented_rejection!(
                1364,
                "this kernel writes its result at the operand's element width, and the \
                 result dtype is stored at a different width; emitting it would overrun \
                 the result allocation (chelis#1360). A typed kernel family for the \
                 result dtype is chelis#1364"
            ),
        ))
    }

    /// Resolve the two arithmetic families used by the materialization
    /// template shared by Cast, Copy, and Realize.
    ///
    /// A bool source or destination is not part of chelis#689's generic
    /// float-family fallback class. It needs the exact one-byte read/write and
    /// checked-cast behavior owned by chelis#1364. Keep that decision at this
    /// operation-aware seam so unrelated bool arithmetic still receives the
    /// generic no-typed-kernel rejection from [`Self::elem_kind`].
    fn cast_elem_kinds(
        node: &DagNode,
        dag: VerifiedDagView<'_>,
    ) -> Result<(kernels::ElemKind, kernels::ElemKind), Unsupported> {
        let src_ty = &dag.get(node.inputs[0]).unwrap().output_type;
        let operation = match &node.op {
            RiscOp::Cast { .. } => "cast",
            RiscOp::Copy => "copy",
            RiscOp::Realize => "realize",
            _ => unreachable!("cast_elem_kinds is only for Cast, Copy, and Realize"),
        };
        if src_ty.precision == Prim::Bool || node.output_type.precision == Prim::Bool {
            return Err(Unsupported::new(
                UnsupportedKind::Dtype(Prim::Bool.name().to_string()),
                format!("HIP {operation} materialization requiring a one-byte bool kernel family"),
                Stage::Codegen("hip"),
                chelis_types::unimplemented_rejection!(
                    1364,
                    "HIP has no Bool8 materialization family: casts to bool need a checked \
                     one-byte writer, casts from bool need a one-byte reader, and bool \
                     Copy/Realize need an exact one-byte identity kernel"
                ),
            ));
        }
        let dst_kind = Self::elem_kind(&node.output_type)?;
        let src_kind = Self::elem_kind(src_ty)?;
        Ok((src_kind, dst_kind))
    }

    /// Map a tensor's precision to a [`kernels::ElemKind`] for kernel
    /// emission. `ElemKind` names an f32/f64 arithmetic kernel family, so
    /// exactly f32 and f64 map; every other precision is a section C2
    /// diagnostic. bf16/f16 elementwise kernels are matmul-only via
    /// `hipblasGemmEx` in WS-A3, and i8/i16 route through the WS-A4 typed
    /// templates via [`Self::dtype_c_type`].
    ///
    /// chelis#730 Phase 1 (census row 5, chelis#689): the former `_ =>
    /// ElemKind::F32` wildcard silently dispatched f32 kernels over
    /// non-f32 buffers - runtime-confirmed corrupt on gfx1151 (int64
    /// `neg` read 8-byte lanes as 4-byte floats and left half the output
    /// buffer unwritten). The ops with typed WS-A4 templates
    /// (Add/Mul/Div/FloorDiv/TruncDiv, pad/shrink, i8/i16 sum) never call
    /// this shorthand. Exhaustive per section C4.1 - no wildcard arm.
    ///
    /// chelis#1360: `bool` used to map here too, because the HIP runtime
    /// stored bool tensors as 4-byte `1.0f`/`0.0f` payloads and the f32
    /// family happened to have the right width. chelis#1308 replaced that
    /// encoding with the tagged carrier's one-byte `Repr::Bool8`, and the
    /// arm became the same defect chelis#689 closed: `cmplt` and
    /// `Cast`-to-bool wrote `N * 4` bytes into an `N * 1` byte
    /// `hipMalloc`, and the one-byte readback decoded the low bytes of the
    /// float stream. bool now rejects, because HIP has no bool kernel
    /// family to dispatch to - restoring the f32-encoded payload would
    /// undo chelis#1308 and put this lane back into chelis#892's shape.
    /// A real one-byte bool family is chelis#1364; it owes a device-side
    /// checked cast, which this lane never had even at four bytes.
    fn elem_kind(ty: &TensorType) -> Result<kernels::ElemKind, Unsupported> {
        Ok(match ty.precision {
            Prim::F32 => kernels::ElemKind::F32,
            Prim::F64 => kernels::ElemKind::F64,
            Prim::Bool
            | Prim::F16
            | Prim::Bf16
            | Prim::F8e4m3
            | Prim::Int8
            | Prim::Int16
            | Prim::Int32
            | Prim::Int64
            | Prim::String => {
                return Err(Unsupported::new(
                    UnsupportedKind::Dtype(ty.precision.name().to_string()),
                    "a HIP kernel family with f32/f64 variants only",
                    Stage::Codegen("hip"),
                    chelis_types::unimplemented_rejection!(
                        689,
                        "this op has no typed HIP kernel for the operand dtype; the former \
                         silent F32 fallback emitted a corrupting kernel (chelis#689). \
                         Cast to f32/f64, or use the ops with typed templates \
                         (add/mul/div and the i8/i16 promoted sum)"
                    ),
                ));
            }
        })
    }

    /// WS-A4: C++ element-type spelling for an active-set dtype. Used
    /// by the typed kernel templates in `crate::kernels` so a single
    /// kernel template can emit `int8_t *a, …` etc. instead of
    /// duplicating per-dtype templates. Complements [`Self::elem_kind`]
    /// for the dtypes that don't (yet) have an `ElemKind` variant.
    fn dtype_c_type(p: Prim) -> &'static str {
        match p {
            Prim::F32 => "float",
            Prim::F64 => "double",
            /* chelis#1308's tagged carrier stores bool as `Repr::Bool8`:
             * exactly one byte, holding exactly 0 or 1 (`Bool8::get` is
             * `== 1`, and `Bool8::from_u8` rejects every other byte). Use
             * an exact-width unsigned type, never `bool`, whose width C++
             * does not fix. This said "float" until chelis#1360. */
            Prim::Bool => "unsigned char",
            Prim::Int8 => "int8_t",
            Prim::Int16 => "int16_t",
            Prim::Int32 => "int32_t",
            Prim::Int64 => "int64_t",
            Prim::F16 | Prim::Bf16 => "uint16_t",
            other => panic!(
                "HIP element type not defined for {} (active dtype set per spec/04-type-system.md §1.1)",
                other.name()
            ),
        }
    }

    // ------------------------------------------------------------------
    // Input/output label logic (identical to C backend)
    // ------------------------------------------------------------------

    pub fn input_labels(dag: VerifiedDagView<'_>) -> Vec<String> {
        let mut labels = Vec::new();
        let mut seen = chelis_unord::UnordSet::new();
        for node in dag.nodes() {
            if let RiscOp::Load { name } = &node.op
                && seen.insert(name.as_str().to_string())
            {
                labels.push(name.as_str().to_string());
            }
        }
        labels
    }

    pub fn output_labels(dag: VerifiedDagView<'_>) -> Vec<String> {
        Self::output_specs(dag)
            .into_iter()
            .map(|o| o.label)
            .collect()
    }

    fn output_specs(dag: VerifiedDagView<'_>) -> Vec<OutputSpec> {
        let mut specs = Vec::new();
        let mut seen = chelis_unord::UnordSet::new();

        for node in dag.nodes() {
            if let RiscOp::Store { name } = &node.op
                && seen.insert(node.id)
            {
                specs.push(OutputSpec {
                    id: node.id,
                    label: name.as_str().to_string(),
                    is_store: true,
                });
            }
        }

        let roots: Vec<NodeId> = if dag.roots().is_empty() {
            dag.nodes().last().map(|n| vec![n.id]).unwrap_or_default()
        } else {
            dag.roots().to_vec()
        };

        for (index, root_id) in roots.into_iter().enumerate() {
            if seen.insert(root_id) {
                specs.push(OutputSpec {
                    id: root_id,
                    label: format!("root{index}"),
                    is_store: false,
                });
            }
        }
        specs
    }

    fn input_slots(labels: &[String]) -> chelis_unord::UnordMap<String, usize> {
        labels
            .iter()
            .cloned()
            .enumerate()
            .map(|(slot, label)| (label, slot))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emit_test_dag(
        dag: &Dag,
        name: &str,
    ) -> Result<(String, PeakDeviceBytesBreakdown), Unsupported> {
        let verified = crate::testing::verified_dag(dag)
            .expect("HIP emitter unit-test DAG must verify ownership");
        let plan = chelis_ir::ownership::plan_hip_storage(verified)
            .expect("HIP emitter unit-test DAG must plan exact storage");
        HipEmitter::emit_dag(plan, name)
    }

    #[test]
    fn drop_releases_exact_device_descriptor_once() {
        let mut dag = Dag::new();
        let source = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Drop, vec![source], TensorType::scalar_f32(), None);

        let (source, _) = emit_test_dag(&dag, "verified_drop").unwrap();
        assert_eq!(
            source
                .matches("chelis_device_tensor_release(o_t0);")
                .count(),
            2,
            "host and device entrypoints each consume the exact verified descriptor once:\n{source}"
        );
    }

    #[test]
    fn borrowed_drop_is_a_logical_discard_without_a_device_release() {
        let mut dag = Dag::new();
        let borrowed = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Drop, vec![borrowed], TensorType::scalar_f32(), None);
        let output = dag.add_node(RiscOp::Copy, vec![borrowed], TensorType::scalar_f32(), None);
        dag.add_root(output);

        let (source, _) = emit_test_dag(&dag, "borrowed_drop").unwrap();
        assert_eq!(
            source
                .matches("chelis_device_tensor_release(o_t0);")
                .count(),
            2,
            "the borrowed descriptor may be cleaned up once per host/device entrypoint, but the logical Drop must not add a third release:\n{source}"
        );
    }

    /// Width of a C type spelling this backend is allowed to emit.
    ///
    /// The list is closed on purpose, and an unrecognised spelling is a test
    /// failure rather than a skipped row: an allowlist of "known widths" can
    /// never be complete, so a new spelling has to be classified here before
    /// it can reach a kernel. Same discipline the capacity census applies to
    /// C type words.
    fn c_type_width(spelling: &str) -> usize {
        match spelling {
            "unsigned char" | "int8_t" => 1,
            "uint16_t" | "int16_t" => 2,
            "float" | "int32_t" => 4,
            "double" | "int64_t" => 8,
            other => panic!(
                "unclassified HIP element type `{other}`: add it to this table with its \
                 exact width before emitting it (chelis#1360)"
            ),
        }
    }

    /// chelis#1360's tripwire.
    ///
    /// Every dtype the emitter is willing to name a device C type for must name
    /// one whose width equals the width the runtime allocates for it. The
    /// defect this catches is not a wrong line of code anywhere: it is two
    /// constants in different files disagreeing, which every individual
    /// emission site reads as correct.
    ///
    /// Before chelis#1308 this passed with `Prim::Bool => "float"`, because
    /// `Repr::Bool` was four bytes. chelis#1308 made it `Repr::Bool8`, and this
    /// assertion is what should have gone red that day.
    ///
    /// `Prim` is matched exhaustively so that adding a dtype stops this
    /// compiling until the new dtype is classified, rather than silently
    /// leaving it unchecked.
    #[test]
    fn every_emittable_dtype_c_type_has_the_runtime_storage_width() {
        for prim in [
            Prim::F32,
            Prim::F64,
            Prim::Bool,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::F16,
            Prim::Bf16,
        ] {
            // Exhaustive by construction: a new `Prim` variant makes this
            // match non-exhaustive and fails the build.
            match prim {
                Prim::F32
                | Prim::F64
                | Prim::Bool
                | Prim::Int8
                | Prim::Int16
                | Prim::Int32
                | Prim::Int64
                | Prim::F16
                | Prim::Bf16
                | Prim::F8e4m3
                | Prim::String => {}
            }
            let runtime_width = prim
                .runtime_dtype()
                .unwrap_or_else(|error| panic!("{} has no runtime dtype: {error}", prim.name()))
                .byte_width();
            let emitted = HipEmitter::dtype_c_type(prim);
            assert_eq!(
                c_type_width(emitted),
                runtime_width,
                "HIP emits `{emitted}` for `{}`, which the runtime stores at {runtime_width} \
                 byte(s); a kernel over that buffer would read or write the wrong span",
                prim.name()
            );
        }
    }

    /// The `ElemKind` arithmetic families carry the same obligation: an
    /// `ElemKind` reached from a dtype must spell a C type of that dtype's
    /// storage width, or the kernel walks the buffer at the wrong stride.
    #[test]
    fn every_elem_kind_family_matches_its_dtype_storage_width() {
        for prim in [Prim::F32, Prim::F64] {
            let ty = TensorType {
                dims: vec![DimInfo::Lit(1)],
                precision: prim,
            };
            let kind = HipEmitter::elem_kind(&ty).expect("f32/f64 have kernel families");
            let runtime_width = prim.runtime_dtype().expect("runtime dtype").byte_width();
            assert_eq!(
                c_type_width(kind.c_type()),
                runtime_width,
                "{}",
                prim.name()
            );
        }

        // bool has no arithmetic family. It mapped to `ElemKind::F32` until
        // chelis#1360, which is exactly the four-versus-one byte mismatch the
        // test above now forbids at the C-type level.
        let bool_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::Bool,
        };
        assert!(
            HipEmitter::elem_kind(&bool_ty).is_err(),
            "bool must not resolve to an f32/f64 arithmetic kernel family"
        );
    }

    #[test]
    fn issue_878_i64_pad_literal_spelling_is_portable_at_signed_min() {
        assert_eq!(HipEmitter::i64_c_literal(i64::MIN), "INT64_MIN");
        assert_eq!(
            HipEmitter::i64_c_literal(-9_007_199_254_740_993),
            "-INT64_C(9007199254740993)"
        );
    }
    use chelis_ir::dag::{Dag, FusedInput, FusedStep, FusedStepOp};

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn vec_i64(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::Int64,
        }
    }

    fn vec_f64(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F64,
        }
    }

    #[test]
    fn integer_abs_is_rejected_before_the_float_unary_template() {
        let ty = vec_i64(1);

        let mut direct = Dag::new();
        let x = direct.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let out = direct.add_node(RiscOp::Abs, vec![x], ty.clone(), None);
        direct.set_roots(vec![out]);
        let err = match emit_test_dag(&direct, "integer_abs") {
            Err(error) => error,
            Ok(_) => panic!("integer abs must not enter the HIP fabsf template"),
        };
        assert!(err.to_string().contains("unsupported: op `Abs`"));

        let mut fused = Dag::new();
        let x = fused.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let out = fused.add_node(
            RiscOp::FusedElem {
                ops: vec![FusedStep {
                    op: FusedStepOp::Abs,
                    input_indices: vec![FusedInput::External(0)],
                }],
            },
            vec![x],
            ty,
            None,
        );
        fused.set_roots(vec![out]);
        let err = match emit_test_dag(&fused, "fused_integer_abs") {
            Err(error) => error,
            Ok(_) => panic!("fused integer abs must not bypass the HIP guard"),
        };
        assert!(err.to_string().contains("unsupported: op `Abs`"));
    }

    fn mat_f32(rows: usize, cols: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
            precision: Prim::F32,
        }
    }

    fn fused_mul_reusable_input_dag() -> Dag {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let scale = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let ops = vec![FusedStep {
            op: FusedStepOp::Mul,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        let fused = dag.add_node(RiscOp::FusedElem { ops }, vec![x, scale], vec_f32(4), None);
        dag.set_reusable_input(fused, x);
        dag
    }

    fn fused_mul_program_owned_reusable_input_dag() -> Dag {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let owned = dag.add_node(RiscOp::Copy, vec![x], vec_f32(4), None);
        let scale = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let ops = vec![FusedStep {
            op: FusedStepOp::Mul,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        let fused = dag.add_node(
            RiscOp::FusedElem { ops },
            vec![owned, scale],
            vec_f32(4),
            None,
        );
        dag.set_reusable_input(fused, owned);
        dag.add_root(fused);
        dag
    }

    #[test]
    fn sparse_scatter_add_emits_hip_atomic_kernel() {
        let mut dag = Dag::new();
        let target = dag.add_node(
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            mat_f32(3, 2),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            vec_i64(4),
            None,
        );
        let updates = dag.add_node(
            RiscOp::Load {
                name: "updates".into(),
            },
            vec![],
            mat_f32(4, 2),
            None,
        );
        let out = dag.add_node(
            RiscOp::ScatterAdd { axis: 0 },
            vec![target, indices, updates],
            mat_f32(3, 2),
            None,
        );
        dag.add_root(out);

        let (hip, _) = emit_test_dag(&dag, "test_sparse").unwrap();

        assert!(hip.contains("kernel_scatter_add_i64"));
        assert!(hip.contains("const long long *indices"));
        assert!(hip.contains("atomicAdd(&out[dst], updates[chelis_logical_offset(i, updates_sh, updates_s, updates_ndim)]);"));
    }

    /// A program-owned `Copy` with a terminal fused use receives the shared
    /// reuse token. The kernel ships
    /// the in-place shape — `__restrict__` only on non-aliased
    /// externals (`ext1`), never on the aliased external (`ext0`) or
    /// `out`.
    #[test]
    fn fused_reusable_input_emits_hip_in_place_restrict_shape() {
        let dag = fused_mul_program_owned_reusable_input_dag();
        let (hip, _) = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(hip.contains("extern \\\"C\\\" __global__ void kernel_fused_3("));
        assert!(hip.contains("o_t3 = chelis_device_tensor_borrow(plan_t3, d_t1->data,"));
        // Aliased external (ext0 ↔ x) must NOT carry __restrict__.
        assert!(!hip.contains("const float *__restrict__ ext0"));
        // Output must NOT carry __restrict__ — it aliases ext0.
        assert!(!hip.contains("float *__restrict__ out"));
        // Non-aliased externals (ext1 here, the const scale) MUST
        // carry __restrict__ in the in-place shape.
        assert!(hip.contains("const float *__restrict__ ext1"));
    }

    /// chelis#1214: a reusable program input is
    /// caller-owned and must not supply the produced tensor's storage.
    #[test]
    fn fused_in_place_does_not_alias_a_caller_owned_input() {
        let dag = fused_mul_reusable_input_dag();
        let (hip, _) = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(
            !hip.contains("d_t0->data, (d_t0->byte_capacity / chelis_dtype_size(d_t0->dtype))"),
            "the fused output must not reuse caller-owned input bytes; got:\n{hip}"
        );
    }

    /// chelis#1214: metadata views preserve the
    /// caller-owned provenance of their source storage.
    #[test]
    fn fused_in_place_does_not_alias_a_view_of_a_caller_owned_input() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            mat_f32(2, 2),
            None,
        );
        let flat = dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(4)],
            },
            vec![x],
            vec_f32(4),
            None,
        );
        let scale = dag.add_node(
            RiscOp::synth_const(Prim::F32, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let fused = dag.add_node(
            RiscOp::FusedElem {
                ops: vec![FusedStep {
                    op: FusedStepOp::Mul,
                    input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
                }],
            },
            vec![flat, scale],
            vec_f32(4),
            None,
        );
        dag.set_reusable_input(fused, flat);
        let (hip, _) = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(
            !hip.contains("d_t1->data, (d_t1->byte_capacity / chelis_dtype_size(d_t1->dtype))"),
            "the fused output must not reuse a caller-owned view's bytes; got:\n{hip}"
        );
    }

    /// Negative: a FusedElem with no `reusable_input` set must still
    /// emit the legacy non-`__restrict__` kernel parameter list.
    /// The in-place shape is opt-in via the upstream linearity-marked
    /// hint, not the default.
    #[test]
    fn fused_without_reusable_input_keeps_non_in_place_kernel_shape() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let scale = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let ops = vec![FusedStep {
            op: FusedStepOp::Mul,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        let fused = dag.add_node(RiscOp::FusedElem { ops }, vec![x, scale], vec_f32(4), None);
        // No set_reusable_input call — the in-place gate must reject.
        dag.add_root(fused);

        let (hip, _) = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(hip.contains("extern \\\"C\\\" __global__ void kernel_fused_2("));
        assert!(hip.contains("const float *ext0"));
        assert!(hip.contains("const float *ext1"));
        assert!(hip.contains("float *out"));
        assert!(
            !hip.contains("__restrict__"),
            "no-reusable-input fused kernels must keep the legacy non-__restrict__ shape"
        );
    }

    /// Issue #250 (parallel #189): an F32 `Const` whose source value is a
    /// denormal must emit the exact f32 bit pattern through
    /// `chelis_f32_from_bits`, not a lossy `{:.8}f` decimal literal. The
    /// reproducer `1e-40` collapses to `0.0f` under `%.8`, so a
    /// round-trip through the emitted literal would lose the source value.
    #[test]
    fn issue_250_f32_const_emits_exact_bit_pattern() {
        // Denormal f32: `{:.8}` formats this as `0.00000000`, which
        // reparses to a different (zero) bit pattern.
        let value = 1e-40_f64;
        let mut dag = Dag::new();
        let c = dag.add_node(
            RiscOp::synth_const(Prim::F32, value),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_root(c);
        let (hip, _) = emit_test_dag(&dag, "test_fn").unwrap();

        let want_bits = (value as f32).to_bits();
        assert_ne!(
            want_bits, 0,
            "reproducer must be a nonzero denormal so `%.8` would lose it"
        );
        let needle = format!("chelis_f32_from_bits(0x{want_bits:08x}u)");
        assert!(
            hip.contains(&needle),
            "F32 const must emit exact bit pattern via chelis_f32_from_bits; \
             expected `{needle}` in:\n{hip}"
        );
        // Negative parity: the lossy decimal form must be gone.
        assert!(
            !hip.contains("float fill_val = 0.00000000f;"),
            "F32 const must not emit a lossy `{{:.8}}f` literal:\n{hip}"
        );
    }

    /// Issue #250 sibling: an F64 `Const` below `1e-17` must round-trip
    /// through `chelis_f64_from_bits` rather than a decimal literal.
    #[test]
    fn issue_250_f64_const_emits_exact_bit_pattern() {
        // 1.0 / 3.0 has no exact decimal form; pin the exact f64 bits.
        let value = 1.0_f64 / 3.0_f64;
        let mut dag = Dag::new();
        let c = dag.add_node(
            RiscOp::synth_const(Prim::F64, value),
            vec![],
            vec_f64(4),
            None,
        );
        dag.add_root(c);
        let (hip, _) = emit_test_dag(&dag, "test_fn").unwrap();

        let want_bits = value.to_bits();
        let needle = format!("chelis_f64_from_bits(0x{want_bits:016x}uLL)");
        assert!(
            hip.contains(&needle),
            "F64 const must emit exact bit pattern via chelis_f64_from_bits; \
             expected `{needle}` in:\n{hip}"
        );
    }

    /// Issue #251 (parallel #248): `uniform_like` `low` / `high` args must
    /// emit through `chelis_f32_from_bits`, not a lossy `{:.8}f` literal.
    /// The reproducer `1e-40` collapses to `0.0f` under `%.8`.
    #[test]
    fn issue_251_uniform_like_args_emit_exact_bit_pattern() {
        let low = 1e-40_f64; // denormal f32: lost by `%.8`
        let high = 1.0_f64 / 3.0_f64; // off-by-ULP under `%.8`
        let mut dag = Dag::new();
        let like = dag.add_node(
            RiscOp::Load {
                name: "like".into(),
            },
            vec![],
            vec_f32(8),
            None,
        );
        let u = dag.add_node(
            RiscOp::UniformLike { low, high, seed: 7 },
            vec![like],
            vec_f32(8),
            None,
        );
        dag.add_root(u);
        let (hip, _) = emit_test_dag(&dag, "test_fn").unwrap();

        let low_bits = (low as f32).to_bits();
        let high_bits = (high as f32).to_bits();
        assert_ne!(
            low_bits, 0,
            "reproducer `low` must be a nonzero denormal so `%.8` would lose it"
        );
        let low_needle = format!("chelis_f32_from_bits(0x{low_bits:08x}u)");
        let high_needle = format!("chelis_f32_from_bits(0x{high_bits:08x}u)");
        assert!(
            hip.contains(&low_needle),
            "uniform_like `low` must emit exact bit pattern; \
             expected `{low_needle}` in:\n{hip}"
        );
        assert!(
            hip.contains(&high_needle),
            "uniform_like `high` must emit exact bit pattern; \
             expected `{high_needle}` in:\n{hip}"
        );
        // Negative parity: no lossy decimal literal for the collapsed
        // denormal `low`.
        assert!(
            !hip.contains("float t1_low = 0.00000000f;"),
            "uniform_like must not emit a lossy `{{:.8}}f` literal:\n{hip}"
        );
    }

    #[test]
    fn issue_937_uniform_like_f64_uses_f64_sampler_and_bounds() {
        let low = 0.1_f64;
        let high = 0.9_f64;
        let mut dag = Dag::new();
        let like = dag.add_node(
            RiscOp::Load {
                name: "like".into(),
            },
            vec![],
            vec_f64(8),
            None,
        );
        let u = dag.add_node(
            RiscOp::UniformLike {
                low,
                high,
                seed: 17,
            },
            vec![like],
            vec_f64(8),
            None,
        );
        dag.add_root(u);
        let (hip, _) = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(hip.contains("__device__ double chelis_uniform_sample_f64("));
        assert!(hip.contains("double t1_low = chelis_f64_from_bits("));
        assert!(hip.contains("double t1_high = chelis_f64_from_bits("));
        assert!(hip.contains("double low, double high"));
        assert!(hip.contains("out[i] = chelis_uniform_sample_f64("));
        assert!(
            !hip.lines()
                .any(|line| line.contains("out[i]") && line.contains("sample_f32")),
            "f64 HIP output must not widen an f32 random sample:\n{hip}"
        );
    }
}
