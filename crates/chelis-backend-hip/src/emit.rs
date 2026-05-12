//! RISC DAG to HIP host code + kernel string emission.
//!
//! Generates C source that includes HIP runtime, embeds kernel source strings,
//! and walks the DAG in topological order launching kernels on GPU.

use chelis_ir::dag::{
    Dag, DagNode, DimExpr, DimInfo, NodeId, RiscOp, TensorType, symbolic_bindings,
};
use chelis_types::types::Prim;

use crate::blas;
use crate::fusion::{FusedInPlaceSpec, fused_in_place_spec};
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
    /// FusedElem nodes inlined into a trailing reduction (no standalone emission).
    reduction_inlined: std::collections::HashSet<usize>,
    /// Worst-case inline staged-reduction scratch requirement outside the slot plan.
    extra_peak_device_bytes_estimate: usize,
    /// Device entrypoints pre-allocate slot storage because borrowed input-backed views
    /// are not the first owners in the host memory plan.
    device_entrypoint_mode: bool,
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
}

struct StridedBatchedMatmulPlan {
    batch_count_expr: String,
    a_batch_stride: usize,
    b_batch_stride: usize,
    out_batch_stride: usize,
}

impl HipEmitter {
    /// Emit complete C/HIP source for a DAG as a function.
    pub(crate) fn emit_dag(dag: &Dag, func_name: &str) -> (String, PeakDeviceBytesBreakdown) {
        // F1 (WS-A2 partial lift): the HIP backend now plumbs the
        // BlasMatmul accumulator field through and dispatches to either
        // `hipblasSgemm` (f32) or `hipblasDgemm` (f64). bf16 / f16 / i8 /
        // i16 are still WS-A3 / WS-A4 territory and are rejected here.
        // The original tactical guard in crates/chelis-ir/src/verify.rs
        // also restricts to f32; that guard is C-backend-shaped (mentions
        // `cblas_sgemm`). The HIP-side check below is the operationally
        // important one when only this backend is exercised.
        for node in dag.nodes() {
            if matches!(node.op, RiscOp::BlasMatmul { .. })
                && let Some(lhs) = dag.get(node.inputs[0])
                && !matches!(lhs.output_type.precision, Prim::F32 | Prim::F64)
            {
                panic!(
                    "F1: HIP BlasMatmul currently supports only f32 / f64; \
                     node {} has operand precision `{}`. \
                     spec/04-type-system.md §5.7.1 documents the per-precision \
                     accumulator defaults; bf16/f16 dispatch lands in WS-A3 and \
                     i8/i16 are out of scope for matmul per §5.7.2.",
                    node.id.0,
                    lhs.output_type.precision.name(),
                );
            }
        }

        let reduction_inlined = chelis_ir::fuse::reduction_inlined_fused_elems(dag);
        let output_specs = Self::output_specs(dag);
        let output_ids: Vec<NodeId> = output_specs.iter().map(|o| o.id).collect();
        let plan = MemoryPlan::build(dag, &output_ids, &reduction_inlined);
        let mut e = HipEmitter {
            lines: Vec::new(),
            indent: 0,
            kernel_sources: Vec::new(),
            plan,
            reduction_inlined: reduction_inlined.iter().map(|id| id.0).collect(),
            extra_peak_device_bytes_estimate: 0,
            device_entrypoint_mode: false,
        };

        // First pass: collect all needed kernel sources by walking the DAG.
        e.collect_kernels(dag);

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

        e.emit_input_shape_preamble(dag, &input_slots, func_name);
        e.line("");

        // Emit static kernel module caches
        let kernel_names: Vec<String> = ks.iter().map(|(n, _)| n.clone()).collect();
        for name in &kernel_names {
            e.line(&format!("static hipModule_t mod_{name} = NULL;"));
            e.line(&format!(
                "if (!mod_{name}) mod_{name} = chelis_compile_kernel({name}_src, \"{name}\");"
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
                e.emit_node(node, dag);
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
                let shape = Self::shape_literal(ty);
                let dtype = Self::dtype_macro(ty);
                e.line(&format!(
                    "outputs[{slot}] = chelis_alloc({ndim}, {shape}, {dtype});"
                ));
                e.line(&format!("chelis_device_to_host(outputs[{slot}], d_t{id});"));
            }
        }

        e.line("");

        // Cleanup: free GPU tensors (skip reduction-inlined FusedElems — never allocated)
        let cleanup = e.plan.emit_cleanup();
        for line in cleanup {
            e.lines.push(line);
        }

        e.indent = 0;
        e.line("}");
        e.line("");
        e.emit_device_entrypoint(dag, func_name, &output_specs, &input_slots, &kernel_names);
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
        (e.lines.join("\n"), breakdown)
    }

    fn emit_device_entrypoint(
        &mut self,
        dag: &Dag,
        func_name: &str,
        output_specs: &[OutputSpec],
        input_slots: &std::collections::HashMap<String, usize>,
        kernel_names: &[String],
    ) {
        let expected_inputs = input_slots.len();
        let expected_outputs = output_specs.len();

        self.line(&format!(
            "extern \"C\" void {func_name}_device(chelis_gpu_tensor **inputs, int n_in, chelis_gpu_tensor **outputs, int n_out) {{"
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

        self.emit_input_shape_preamble_device(dag, input_slots, func_name);
        self.line("");

        for name in kernel_names {
            self.line(&format!("static hipModule_t mod_{name} = NULL;"));
            self.line(&format!(
                "if (!mod_{name}) mod_{name} = chelis_compile_kernel({name}_src, \"{name}\");"
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
                self.emit_node(node, dag);
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
                    format!("outputs[{slot}] = chelis_gpu_clone(inputs[{input_idx}]);")
                }
                _ => format!("outputs[{slot}] = chelis_gpu_clone(d_t{id});"),
            };
            self.line(&line);
        }

        self.line("");
        let cleanup = self.plan.emit_cleanup();
        for line in cleanup {
            self.lines.push(line);
        }
        self.device_entrypoint_mode = false;
        self.indent = 0;
        self.line("}");
    }

    // ------------------------------------------------------------------
    // Kernel collection (first pass)
    // ------------------------------------------------------------------

    fn collect_kernels(&mut self, dag: &Dag) {
        let mut seen = std::collections::HashSet::new();
        for node in dag.nodes() {
            // Skip reduction-inlined FusedElem nodes (they become part of the
            // reduction kernel).
            if self.reduction_inlined.contains(&node.id.0) {
                continue;
            }
            match &node.op {
                RiscOp::Sum { axis, .. } | RiscOp::MaxReduce { axis } => {
                    let sources = self.reduction_kernel_sources(node, dag, *axis);
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
                    let sources = self.extra_reduction_kernel_sources(node, dag);
                    for (name, source) in sources {
                        if seen.insert(name.clone()) {
                            self.kernel_sources.push((name, source));
                        }
                    }
                    continue;
                }
                _ => {}
            }
            let name = self.kernel_name_for_op(&node.op, node, dag);
            if let Some(name) = name
                && seen.insert(name.clone())
            {
                let source = self.kernel_source_for_op(&name, &node.op, node, dag);
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
    }

    /// True when a kernel name is unique to a single DagNode (i.e. its
    /// suffix encodes a node id). Used to decide whether prepending span
    /// comments inside the kernel source is unambiguous.
    fn is_per_node_kernel_name(name: &str) -> bool {
        // FusedElem: kernel_fused_<id>
        // Fused reductions: kernel_fused_sum_<id>, kernel_fused_maxred_<id>
        name.starts_with("kernel_fused_")
    }

    fn input_types(dag: &Dag) -> std::collections::HashMap<String, TensorType> {
        let mut seen = std::collections::HashMap::<String, TensorType>::new();
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
        dag: &Dag,
        input_slots: &std::collections::HashMap<String, usize>,
        func_name: &str,
    ) {
        // Iteration order over `input_types` (a HashMap) must be
        // deterministic so the emitted host code is byte-identical
        // across runs. Sort by label; lookups are by name and emitted
        // lines are independent per label.
        // See spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md.
        let input_types = Self::input_types(dag);
        let mut sorted_labels: Vec<&String> = input_types.keys().collect();
        sorted_labels.sort();
        // Producer-supplied `func_name` flows into format-string context;
        // sanitize per spec/upstream-bugs/producer-string-sanitization.md.
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);
        for label in sorted_labels {
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
            self.line(&format!(
                "if (inputs[{slot}]->ndim != {}) {{",
                Self::ndim(ty)
            ));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"{func_name_fmt}: input `{label_fmt}` expected rank {}, got %d\\n\", inputs[{slot}]->ndim);",
                Self::ndim(ty)
            ));
            self.line("abort();");
            self.indent -= 1;
            self.line("}");
            for (axis, dim) in ty.dims.iter().enumerate() {
                if let Some(expected) = Self::known_dim_size(dim) {
                    self.line(&format!(
                        "if (inputs[{slot}]->shape[{axis}] != {expected}) {{"
                    ));
                    self.indent += 1;
                    self.line(&format!(
                        "fprintf(stderr, \"{func_name_fmt}: input `{label_fmt}` axis {axis} expected {expected}, got %d\\n\", inputs[{slot}]->shape[{axis}]);"
                    ));
                    self.line("abort();");
                    self.indent -= 1;
                    self.line("}");
                }
            }
        }

        for binding in symbolic_bindings(dag) {
            let canonical_slot = input_slots[&binding.canonical.input_label];
            // `binding.name` flows into format-string context; sanitize.
            let binding_name_fmt =
                chelis_ir::span_sanitize::sanitize_for_format_string(&binding.name);
            self.line(&format!(
                "int {} = inputs[{canonical_slot}]->shape[{}];",
                binding.name, binding.canonical.axis
            ));
            for occurrence in binding.others {
                let slot = input_slots[&occurrence.input_label];
                let occ_label_fmt =
                    chelis_ir::span_sanitize::sanitize_for_format_string(&occurrence.input_label);
                self.line(&format!(
                    "if (inputs[{slot}]->shape[{}] != {}) {{",
                    occurrence.axis, binding.name
                ));
                self.indent += 1;
                self.line(&format!(
                    "fprintf(stderr, \"{func_name_fmt}: symbolic dim `{binding_name_fmt}` mismatch: {occ_label_fmt}[{}]=%d but {binding_name_fmt}=%d\\n\", inputs[{slot}]->shape[{}], {});",
                    occurrence.axis,
                    occurrence.axis,
                    binding.name
                ));
                self.line("abort();");
                self.indent -= 1;
                self.line("}");
            }
        }
    }

    fn emit_input_shape_preamble_device(
        &mut self,
        dag: &Dag,
        input_slots: &std::collections::HashMap<String, usize>,
        func_name: &str,
    ) {
        // Iteration order over `input_types` (a HashMap) must be
        // deterministic so the emitted device-side code is byte-
        // identical across runs. Sort by label; lookups are by name
        // and emitted lines are independent per label.
        // See spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md.
        let input_types = Self::input_types(dag);
        let mut sorted_labels: Vec<&String> = input_types.keys().collect();
        sorted_labels.sort();
        // Format-string-context sanitization for producer-supplied
        // strings per spec/upstream-bugs/producer-string-sanitization.md.
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);
        for label in sorted_labels {
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
            self.line(&format!(
                "if (inputs[{slot}]->ndim != {}) {{",
                Self::ndim(ty)
            ));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"{func_name_fmt}_device: input `{label_fmt}` expected rank {}, got %d\\n\", inputs[{slot}]->ndim);",
                Self::ndim(ty)
            ));
            self.line("abort();");
            self.indent -= 1;
            self.line("}");
            for (axis, dim) in ty.dims.iter().enumerate() {
                if let Some(expected) = Self::known_dim_size(dim) {
                    self.line(&format!(
                        "if (inputs[{slot}]->shape[{axis}] != {expected}) {{"
                    ));
                    self.indent += 1;
                    self.line(&format!(
                        "fprintf(stderr, \"{func_name_fmt}_device: input `{label_fmt}` axis {axis} expected {expected}, got %d\\n\", inputs[{slot}]->shape[{axis}]);"
                    ));
                    self.line("abort();");
                    self.indent -= 1;
                    self.line("}");
                }
            }
        }

        for binding in symbolic_bindings(dag) {
            let canonical_slot = input_slots[&binding.canonical.input_label];
            let binding_name_fmt =
                chelis_ir::span_sanitize::sanitize_for_format_string(&binding.name);
            self.line(&format!(
                "int {} = inputs[{canonical_slot}]->shape[{}];",
                binding.name, binding.canonical.axis
            ));
            for occurrence in binding.others {
                let slot = input_slots[&occurrence.input_label];
                let occ_label_fmt =
                    chelis_ir::span_sanitize::sanitize_for_format_string(&occurrence.input_label);
                self.line(&format!(
                    "if (inputs[{slot}]->shape[{}] != {}) {{",
                    occurrence.axis, binding.name
                ));
                self.indent += 1;
                self.line(&format!(
                    "fprintf(stderr, \"{func_name_fmt}_device: symbolic dim `{binding_name_fmt}` mismatch: {occ_label_fmt}[{}]=%d but {binding_name_fmt}=%d\\n\", inputs[{slot}]->shape[{}], {});",
                    occurrence.axis,
                    occurrence.axis,
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
        dag: &Dag,
        axis: usize,
    ) -> Vec<(String, String)> {
        let kind = match node.op {
            RiscOp::Sum { .. } => kernels::ReduceKind::Sum,
            RiscOp::MaxReduce { .. } => kernels::ReduceKind::Max,
            _ => unreachable!("reduction_kernel_sources called on non-reduction"),
        };
        let input_id = node.inputs[0];
        if self.reduction_inlined.contains(&input_id.0) {
            let fused_node = dag.get(input_id).unwrap();
            let elem = Self::elem_kind(&fused_node.output_type);
            let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
            let name = Self::fused_reduction_kernel_name(node.id.0, kind);
            let source = kernels::reduce_fused(&name, axis, steps, n_ext, kind, elem);
            return vec![(name, source)];
        }

        if matches!(kind, kernels::ReduceKind::Sum)
            && let Some(matmul) = blas::detect_matmul_pattern(dag, node.id)
            && Self::supports_static_hipblas_matmul(dag, &matmul, &node.output_type)
        {
            return Vec::new();
        }

        // For Sum, the result precision IS the accumulator (spec §5.7.1)
        // so we read the operand precision separately. For MaxReduce, the
        // accumulator and result both match the operand precision.
        let operand_ty = &dag.get(input_id).unwrap().output_type;
        let operand_kind = Self::elem_kind(operand_ty);
        let acc_kind = Self::elem_kind(&node.output_type);
        let name = Self::reduction_kernel_name(kind, axis, acc_kind);
        let source = match kind {
            kernels::ReduceKind::Sum => kernels::reduce_sum(&name, axis, operand_kind, acc_kind),
            kernels::ReduceKind::Max => kernels::reduce_max(&name, axis, acc_kind),
        };
        vec![(name, source)]
    }

    fn extra_reduction_kernel_sources(&self, node: &DagNode, dag: &Dag) -> Vec<(String, String)> {
        // Kernel sources for the four reductions previously deferred to
        // the C backend: Min / Prod / Argmax / Argmin. Argmax/Argmin emit
        // an i64 result tensor; Min/Prod emit an in-precision result.
        // For all four, kernel naming + body are driven by the OPERAND
        // precision, which lives on the input tensor (Argmax/Argmin's
        // output_type is `int64` and would otherwise tip elem_kind into
        // its panic arm).
        let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
        let elem = Self::elem_kind(input_ty);
        match &node.op {
            RiscOp::MinReduce { axis } => {
                let name = Self::extra_reduction_kernel_name("min", *axis, elem);
                let src = kernels::reduce_min(&name, *axis, elem);
                vec![(name, src)]
            }
            RiscOp::ProdReduce { axis } => {
                let name = Self::extra_reduction_kernel_name("prod", *axis, elem);
                let src = kernels::reduce_prod(&name, *axis, elem);
                vec![(name, src)]
            }
            RiscOp::Argmax { axis } => {
                let name = Self::extra_reduction_kernel_name("argmax", *axis, elem);
                let src = kernels::reduce_argmax(&name, *axis, elem);
                vec![(name, src)]
            }
            RiscOp::Argmin { axis } => {
                let name = Self::extra_reduction_kernel_name("argmin", *axis, elem);
                let src = kernels::reduce_argmin(&name, *axis, elem);
                vec![(name, src)]
            }
            _ => unreachable!("extra_reduction_kernel_sources expected Min/Prod/Argmax/Argmin"),
        }
    }

    fn kernel_name_for_op(&self, op: &RiscOp, node: &DagNode, dag: &Dag) -> Option<String> {
        let kind_for_node = |n: &DagNode| -> kernels::ElemKind { Self::elem_kind(&n.output_type) };
        match op {
            RiscOp::Add => Some(format!("kernel_add_{}", kind_for_node(node).suffix())),
            RiscOp::Mul => Some(format!("kernel_mul_{}", kind_for_node(node).suffix())),
            RiscOp::MaxElem => Some(format!("kernel_max_elem_{}", kind_for_node(node).suffix())),
            RiscOp::CmpLt => {
                // CmpLt has bool output but operand-precision storage;
                // dispatch on the operand precision so the kernel name
                // matches the kernel source emitted in
                // `kernel_source_for_op`.
                let operand_kind = Self::elem_kind(&dag.get(node.inputs[0]).unwrap().output_type);
                Some(format!("kernel_cmplt_{}", operand_kind.suffix()))
            }
            RiscOp::Neg => Some(format!("kernel_neg_{}", kind_for_node(node).suffix())),
            RiscOp::Exp => Some(format!("kernel_exp_{}", kind_for_node(node).suffix())),
            RiscOp::Log => Some(format!("kernel_log_{}", kind_for_node(node).suffix())),
            RiscOp::Sin => Some(format!("kernel_sin_{}", kind_for_node(node).suffix())),
            RiscOp::Sqrt => Some(format!("kernel_sqrt_{}", kind_for_node(node).suffix())),
            RiscOp::Cos => Some(format!("kernel_cos_{}", kind_for_node(node).suffix())),
            RiscOp::Tan => Some(format!("kernel_tan_{}", kind_for_node(node).suffix())),
            RiscOp::Atan => Some(format!("kernel_atan_{}", kind_for_node(node).suffix())),
            RiscOp::Abs => Some(format!("kernel_abs_{}", kind_for_node(node).suffix())),
            RiscOp::Floor => Some(format!("kernel_floor_{}", kind_for_node(node).suffix())),
            RiscOp::Ceil => Some(format!("kernel_ceil_{}", kind_for_node(node).suffix())),
            RiscOp::UniformLike { .. } => Some(format!(
                "kernel_uniform_like_{}",
                kind_for_node(node).suffix()
            )),
            RiscOp::Dropout { .. } | RiscOp::Drop => None,
            RiscOp::Copy => Some(Self::cast_kernel_name(node, dag)),
            RiscOp::Sum { axis, .. } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    Some(Self::fused_reduction_kernel_name(
                        node.id.0,
                        kernels::ReduceKind::Sum,
                    ))
                } else {
                    Some(Self::reduction_kernel_name(
                        kernels::ReduceKind::Sum,
                        *axis,
                        Self::elem_kind(&node.output_type),
                    ))
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
                        Self::elem_kind(&node.output_type),
                    ))
                }
            }
            RiscOp::MinReduce { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                Some(Self::extra_reduction_kernel_name(
                    "min",
                    *axis,
                    Self::elem_kind(input_ty),
                ))
            }
            RiscOp::ProdReduce { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                Some(Self::extra_reduction_kernel_name(
                    "prod",
                    *axis,
                    Self::elem_kind(input_ty),
                ))
            }
            RiscOp::Argmax { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                Some(Self::extra_reduction_kernel_name(
                    "argmax",
                    *axis,
                    Self::elem_kind(input_ty),
                ))
            }
            RiscOp::Argmin { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                Some(Self::extra_reduction_kernel_name(
                    "argmin",
                    *axis,
                    Self::elem_kind(input_ty),
                ))
            }
            RiscOp::OneHot { .. } => None,
            RiscOp::Const { .. } => Some(format!("kernel_fill_{}", kind_for_node(node).suffix())),
            RiscOp::Realize => Some(Self::cast_kernel_name(node, dag)),
            RiscOp::Cast { .. } => Some(Self::cast_kernel_name(node, dag)),
            // Movement ops and Load/Store are not kernels
            RiscOp::Load { .. }
            | RiscOp::Store { .. }
            | RiscOp::Reshape { .. }
            | RiscOp::Permute { .. }
            | RiscOp::Expand { .. }
            | RiscOp::Pad { .. }
            | RiscOp::Shrink { .. }
            | RiscOp::Stride { .. }
            | RiscOp::BlasMatmul { .. } => None,
            RiscOp::Gather { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                let elem = Self::elem_kind(&node.output_type);
                Some(match indices_ty.precision {
                    Prim::Int32 => format!("kernel_gather_i32_{}", elem.suffix()),
                    Prim::Int64 => format!("kernel_gather_i64_{}", elem.suffix()),
                    _ => "kernel_gather_invalid".into(),
                })
            }
            RiscOp::ScatterAdd { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                let elem = Self::elem_kind(&node.output_type);
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
            RiscOp::FusedElem { .. } => Some(format!("kernel_fused_{}", node.id.0)),
        }
    }

    /// Build the cast kernel name. When src/dst precision agree, this is
    /// the in-precision identity kernel; when they differ, it's the
    /// cross-precision conversion kernel.
    fn cast_kernel_name(node: &DagNode, dag: &Dag) -> String {
        let src_ty = &dag.get(node.inputs[0]).unwrap().output_type;
        let dst_kind = Self::elem_kind(&node.output_type);
        let src_kind = Self::elem_kind(src_ty);
        if src_kind == dst_kind {
            format!("kernel_cast_{}", dst_kind.suffix())
        } else {
            format!("kernel_cast_{}_to_{}", src_kind.suffix(), dst_kind.suffix())
        }
    }

    fn kernel_source_for_op(&self, name: &str, op: &RiscOp, node: &DagNode, dag: &Dag) -> String {
        // CmpLt's output type is `bool` (semantically) but the kernel
        // writes 1.0/0.0 of operand precision to the GPU buffer. Use the
        // operand precision for kernel emission; the rest of the
        // floating ops have output_type == operand_type so the more
        // common path uses output_type below.
        let elem = match op {
            RiscOp::CmpLt => Self::elem_kind(&dag.get(node.inputs[0]).unwrap().output_type),
            _ => Self::elem_kind(&node.output_type),
        };
        match op {
            RiscOp::Add => kernels::binary_elementwise(name, "+", elem),
            RiscOp::Mul => kernels::binary_elementwise(name, "*", elem),
            RiscOp::MaxElem => kernels::binary_func(name, "fmaxf", elem),
            RiscOp::CmpLt => kernels::cmplt(name, elem),
            RiscOp::Neg => kernels::unary_prefix(name, "-", elem),
            RiscOp::Exp => kernels::unary_func(name, "expf", elem),
            RiscOp::Log => kernels::unary_func(name, "logf", elem),
            RiscOp::Sin => kernels::unary_func(name, "sinf", elem),
            RiscOp::Sqrt => kernels::unary_func(name, "sqrtf", elem),
            RiscOp::Cos => kernels::unary_func(name, "cosf", elem),
            RiscOp::Tan => kernels::unary_func(name, "tanf", elem),
            RiscOp::Atan => kernels::unary_func(name, "atanf", elem),
            RiscOp::Abs => kernels::unary_func(name, "fabsf", elem),
            RiscOp::Floor => kernels::unary_func(name, "floorf", elem),
            RiscOp::Ceil => kernels::unary_func(name, "ceilf", elem),
            RiscOp::UniformLike { .. } => kernels::uniform_like(name, elem),
            RiscOp::Sum { axis, .. } => {
                let input_id = node.inputs[0];
                let operand_kind = Self::elem_kind(&dag.get(input_id).unwrap().output_type);
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
                    kernels::reduce_fused(
                        name,
                        *axis,
                        steps,
                        n_ext,
                        kernels::ReduceKind::Sum,
                        operand_kind,
                    )
                } else {
                    kernels::reduce_sum(name, *axis, operand_kind, elem)
                }
            }
            RiscOp::MaxReduce { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
                    let inner_kind = Self::elem_kind(&fused_node.output_type);
                    kernels::reduce_fused(
                        name,
                        *axis,
                        steps,
                        n_ext,
                        kernels::ReduceKind::Max,
                        inner_kind,
                    )
                } else {
                    kernels::reduce_max(name, *axis, elem)
                }
            }
            RiscOp::MinReduce { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                kernels::reduce_min(name, *axis, Self::elem_kind(input_ty))
            }
            RiscOp::ProdReduce { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                kernels::reduce_prod(name, *axis, Self::elem_kind(input_ty))
            }
            RiscOp::Argmax { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                kernels::reduce_argmax(name, *axis, Self::elem_kind(input_ty))
            }
            RiscOp::Argmin { axis } => {
                let input_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                kernels::reduce_argmin(name, *axis, Self::elem_kind(input_ty))
            }
            RiscOp::Const { .. } => kernels::fill(name, elem),
            RiscOp::Realize => Self::cast_kernel_source(name, node, dag),
            RiscOp::Cast { .. } => Self::cast_kernel_source(name, node, dag),
            RiscOp::Copy => Self::cast_kernel_source(name, node, dag),
            RiscOp::FusedElem { ops } => {
                let aliased_ext = fused_in_place_spec(node, dag).map(|reusable| {
                    node.inputs
                        .iter()
                        .position(|&input| input == reusable)
                        .expect("reusable input must appear in node inputs")
                });
                kernels::fused_elementwise(name, ops, node.inputs.len(), aliased_ext, elem)
            }
            RiscOp::Gather { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                match indices_ty.precision {
                    Prim::Int32 => kernels::gather(name, "int", elem),
                    Prim::Int64 => kernels::gather(name, "long long", elem),
                    other => panic!(
                        "HIP backend sparse gather requires int32/int64 indices, got {}",
                        other.name()
                    ),
                }
            }
            RiscOp::ScatterAdd { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                match indices_ty.precision {
                    Prim::Int32 => kernels::scatter_add(name, "int", elem),
                    Prim::Int64 => kernels::scatter_add(name, "long long", elem),
                    other => panic!(
                        "HIP backend sparse scatter_add requires int32/int64 indices, got {}",
                        other.name()
                    ),
                }
            }
            RiscOp::Scatter { .. } => {
                let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                match indices_ty.precision {
                    Prim::Int32 => kernels::scatter_replace(name, "int"),
                    Prim::Int64 => kernels::scatter_replace(name, "long long"),
                    other => panic!(
                        "HIP backend sparse scatter_replace requires int32/int64 indices, got {}",
                        other.name()
                    ),
                }
            }
            _ => unreachable!("no kernel for op: {op:?}"),
        }
    }

    /// Cast / Realize / Copy kernel source: in-precision identity when
    /// src and dst kinds agree, cross-precision conversion otherwise.
    fn cast_kernel_source(name: &str, node: &DagNode, dag: &Dag) -> String {
        let src_ty = &dag.get(node.inputs[0]).unwrap().output_type;
        let dst_kind = Self::elem_kind(&node.output_type);
        let src_kind = Self::elem_kind(src_ty);
        if src_kind == dst_kind {
            kernels::cast(name, dst_kind)
        } else {
            kernels::cast_convert(name, src_kind, dst_kind)
        }
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

    fn emit_node(&mut self, node: &DagNode, dag: &Dag) {
        let id = node.id.0;
        // Resolve the precision-suffixed kernel name once, so the launch
        // shims agree with the kernel-source emitter on the symbol the
        // host references (e.g. `kernel_add_f32` vs `kernel_add_f64`).
        let resolved_kernel_name = || -> String {
            self.kernel_name_for_op(&node.op, node, dag)
                .unwrap_or_else(|| panic!("op {:?} has no kernel name", node.op))
        };
        match &node.op {
            RiscOp::Const { value } => self.emit_const(id, *value, &node.output_type),
            RiscOp::Load { .. } => unreachable!("handled in emit_dag"),
            RiscOp::Add => self.emit_binary_launch(
                id,
                &resolved_kernel_name(),
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Mul => self.emit_binary_launch(
                id,
                &resolved_kernel_name(),
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::MaxElem => self.emit_binary_launch(
                id,
                &resolved_kernel_name(),
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::CmpLt => self.emit_binary_launch(
                id,
                &resolved_kernel_name(),
                &node.inputs,
                &node.output_type,
            ),
            RiscOp::Neg => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Exp => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Log => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Sin => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Sqrt => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Cos => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Tan => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Atan => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Abs => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Floor => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Ceil => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::UniformLike { low, high, seed } => {
                self.emit_uniform_like_launch(id, *low, *high, *seed, &node.output_type)
            }
            RiscOp::Dropout { .. } => {
                unreachable!("dropout should be rejected before HIP code generation")
            }
            RiscOp::Copy => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Drop => {}
            RiscOp::Sum { axis, .. } => {
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
                    );
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
                    );
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
                self.emit_extra_reduce_launch(id, *axis, &node.inputs, &node.output_type, dag);
            }
            RiscOp::OneHot { .. } => {
                panic!(
                    "HIP backend: internal OneHot must be consumed by specialization before codegen"
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
            RiscOp::Pad { .. } => {
                todo!("Pad on GPU requires a kernel. Deferred to Phase 1a iteration 2")
            }
            RiscOp::Shrink { .. } => {
                todo!("Shrink on GPU requires a kernel. Deferred to Phase 1a iteration 2")
            }
            RiscOp::Stride { strides } => {
                self.emit_stride(id, strides, &node.inputs, &node.output_type);
            }
            RiscOp::Realize => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Cast { .. } => {
                self.emit_unary_launch(id, &resolved_kernel_name(), &node.inputs, &node.output_type)
            }
            RiscOp::Store { name } => {
                self.emit_store(id, name.as_str(), &node.inputs, &node.output_type)
            }
            RiscOp::FusedElem { ops } => {
                let kernel_name = format!("kernel_fused_{}", node.id.0);
                let in_place =
                    fused_in_place_spec(node, dag).map(|reusable_input| FusedInPlaceSpec {
                        reusable_input,
                        slot_has_later_owner: self.slot_has_later_owner(id),
                    });
                self.emit_fused_launch(
                    node.id.0,
                    &kernel_name,
                    &node.inputs,
                    ops,
                    &node.output_type,
                    in_place,
                );
            }
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
                ..
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
                    },
                    &node.output_type,
                    dag,
                );
            }
            RiscOp::Gather { axis } => {
                self.emit_gather_launch(id, *axis, &node.inputs, &node.output_type, dag)
            }
            RiscOp::ScatterAdd { axis } => {
                self.emit_scatter_add_launch(id, *axis, &node.inputs, &node.output_type, dag)
            }
            RiscOp::Scatter { axis } => {
                self.emit_scatter_replace_launch(id, *axis, &node.inputs, &node.output_type, dag)
            }
        }
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

    /// True iff some node strictly after `id` in topological order also
    /// owns the slot that `id` owns. Mirrors the C-side
    /// `slot_has_later_owner` used by the in-place fused-elementwise
    /// wrapper to decide whether to defer slot allocation to the
    /// non-aliased fall-back branch.
    fn slot_has_later_owner(&self, id: usize) -> bool {
        let slot_id = self.slot_id_for_node(id);
        self.plan.iter_node_kinds().skip(id + 1).any(|kind| {
            matches!(
                kind,
                NodeMemoryKind::SlotBacked { slot } if *slot == slot_id
            )
        })
    }

    fn emit_slot_allocation_if_needed(&mut self, id: usize, ty: &TensorType) {
        if self.device_entrypoint_mode {
            return;
        }
        let slot_id = self.slot_id_for_node(id);
        let slot = self.plan.slot(slot_id);
        if slot.first_owner != NodeId(id) {
            return;
        }
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_gpu_tensor *chelis_slot{slot_id} = chelis_gpu_alloc({ndim}, {shape}, {dtype});"
        ));
    }

    fn emit_slot_wrapper(&mut self, id: usize, ty: &TensorType) {
        self.emit_slot_allocation_if_needed(id, ty);
        let slot_id = self.slot_id_for_node(id);
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc_view({ndim}, {shape}, {dtype}, chelis_slot{slot_id}->data, chelis_slot{slot_id}->storage_size);"
        ));
    }

    fn emit_device_slot_allocations(&mut self, dag: &Dag) {
        let declarations = self
            .plan
            .slots()
            .iter()
            .map(|slot| {
                let ty = &dag
                    .get(slot.first_owner)
                    .unwrap_or_else(|| panic!("missing first owner {}", slot.first_owner.0))
                    .output_type;
                let ndim = Self::ndim(ty);
                let shape = Self::shape_literal(ty);
                let dtype = Self::dtype_macro(ty);
                format!(
                    "chelis_gpu_tensor *chelis_slot{} = chelis_gpu_alloc({ndim}, {shape}, {dtype});",
                    slot.id
                )
            })
            .collect::<Vec<_>>();
        for declaration in declarations {
            self.line(&declaration);
        }
    }

    fn emit_alias_view(&mut self, id: usize, ty: &TensorType, data_expr: &str, storage_expr: &str) {
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc_view({ndim}, {shape}, {dtype}, {data_expr}, {storage_expr});"
        ));
    }

    fn emit_const(&mut self, id: usize, value: f64, ty: &TensorType) {
        self.emit_slot_wrapper(id, ty);
        let elem = Self::elem_kind(ty);
        let kernel = format!("kernel_fill_{}", elem.suffix());
        self.line("{");
        self.indent += 1;
        match elem {
            kernels::ElemKind::F32 => {
                self.line(&format!("float fill_val = {:.8}f;", value as f32));
            }
            kernels::ElemKind::F64 => {
                // emit a double literal (no `f` suffix); use 17 sig digits
                // per IEEE-754 round-trip.
                self.line(&format!("double fill_val = {value:.17e};"));
            }
        }
        self.line(&format!("int fill_size = d_t{id}->size;"));
        self.line(&format!(
            "void *fill_args[] = {{ &d_t{id}->data, &fill_val, &fill_size }};"
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel}"),
            &kernel,
            "(fill_size + 255) / 256",
            "256",
            "fill_args",
        );
        self.indent -= 1;
        self.line("}");
    }

    // ------------------------------------------------------------------
    // Load
    // ------------------------------------------------------------------

    fn emit_load(&mut self, id: usize, input_idx: usize, ty: &TensorType) {
        match self.plan.node_kind(NodeId(id)) {
            NodeMemoryKind::UniqueInput { .. } => {
                self.emit_slot_wrapper(id, ty);
                self.line(&format!(
                    "chelis_host_to_device(d_t{id}, inputs[{input_idx}]);"
                ));
            }
            NodeMemoryKind::RepeatedLoadAlias { canonical_load } => {
                self.emit_alias_view(
                    id,
                    ty,
                    &format!("d_t{}->data", canonical_load.0),
                    &format!("d_t{}->storage_size", canonical_load.0),
                );
            }
            other => panic!("unexpected memory plan for load node {id}: {other:?}"),
        }
    }

    fn emit_load_device(&mut self, id: usize, input_idx: usize, ty: &TensorType) {
        match self.plan.node_kind(NodeId(id)) {
            NodeMemoryKind::UniqueInput { .. } => {
                self.emit_alias_view(
                    id,
                    ty,
                    &format!("inputs[{input_idx}]->data"),
                    &format!("inputs[{input_idx}]->storage_size"),
                );
            }
            NodeMemoryKind::RepeatedLoadAlias { canonical_load } => {
                self.emit_alias_view(
                    id,
                    ty,
                    &format!("d_t{}->data", canonical_load.0),
                    &format!("d_t{}->storage_size", canonical_load.0),
                );
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
        self.line(&format!("int t{id}_size = d_t{id}->size;"));
        // Stride params for a (8 ints)
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int t{id}_a_ndim = d_t{a}->ndim;"));
        self.line(&format!("int t{id}_a_size = d_t{a}->storage_size;"));
        // Stride params for b (8 ints)
        self.emit_stride_vars(id, "b", b);
        self.line(&format!("int t{id}_b_ndim = d_t{b}->ndim;"));
        self.line(&format!("int t{id}_b_size = d_t{b}->storage_size;"));
        // Shape params for output (8 ints)
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int t{id}_out_ndim = d_t{id}->ndim;"));
        // Build args array
        self.line(&format!(
            "void *args[] = {{ &d_t{a}->data, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             &d_t{b}->data, {b_stride_refs}, &t{id}_b_ndim, &t{id}_b_size, \
             &d_t{id}->data, {out_shape_refs}, &t{id}_out_ndim, &t{id}_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            b_stride_refs = self.stride_arg_refs(id, "b"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            kernel_name,
            &format!("(t{id}_size + 255) / 256"),
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
        self.line(&format!("int t{id}_size = d_t{id}->size;"));
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int t{id}_a_ndim = d_t{a}->ndim;"));
        self.line(&format!("int t{id}_a_size = d_t{a}->storage_size;"));
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int t{id}_out_ndim = d_t{id}->ndim;"));
        self.line(&format!(
            "void *args[] = {{ &d_t{a}->data, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             &d_t{id}->data, {out_shape_refs}, &t{id}_out_ndim, &t{id}_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            kernel_name,
            &format!("(t{id}_size + 255) / 256"),
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
    ) {
        self.emit_slot_wrapper(id, ty);
        let elem = Self::elem_kind(ty);
        let kernel = format!("kernel_uniform_like_{}", elem.suffix());
        self.line("{");
        self.indent += 1;
        // The PRNG itself is f32; the f64 kernel widens at the final
        // store. Emit `low` / `high` as `float` regardless of `ty.precision`.
        self.line(&format!("float t{id}_low = {:.8}f;", low as f32));
        self.line(&format!("float t{id}_high = {:.8}f;", high as f32));
        self.line(&format!("unsigned long long t{id}_seed = {seed}ULL;"));
        self.line(&format!("int t{id}_size = d_t{id}->size;"));
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int t{id}_out_ndim = d_t{id}->ndim;"));
        self.line(&format!(
            "void *args[] = {{ &t{id}_low, &t{id}_high, &t{id}_seed, &d_t{id}->data, {out_shape_refs}, &t{id}_out_ndim, &t{id}_size }};",
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel}"),
            &kernel,
            &format!("(t{id}_size + 255) / 256"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    fn emit_gather_launch(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: &Dag,
    ) {
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
        let before = Self::dim_product_expr(&values_ty.dims[..axis]);
        let axis_size = Self::emit_dim_info(&values_ty.dims[axis]);
        let after = Self::dim_product_expr(&values_ty.dims[axis + 1..]);
        let elem = Self::elem_kind(ty);
        let kernel_name = match indices_ty.precision {
            Prim::Int32 => format!("kernel_gather_i32_{}", elem.suffix()),
            Prim::Int64 => format!("kernel_gather_i64_{}", elem.suffix()),
            _ => unreachable!(),
        };

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int t{id}_before = {before};"));
        self.line(&format!("int t{id}_axis_size = {axis_size};"));
        self.line(&format!("int t{id}_after = {after};"));
        self.line(&format!("int t{id}_index_count = d_t{indices}->size;"));
        self.line(&format!("int t{id}_total = d_t{id}->size;"));
        self.line(&format!(
            "void *args[] = {{ &d_t{values}->data, &d_t{indices}->data, &d_t{id}->data, &t{id}_before, &t{id}_axis_size, &t{id}_after, &t{id}_index_count, &t{id}_total }};"
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            &kernel_name,
            &format!("(t{id}_total + 255) / 256"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    fn emit_scatter_add_launch(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: &Dag,
    ) {
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
        let before = Self::dim_product_expr(&target_ty.dims[..axis]);
        let axis_size = Self::emit_dim_info(&target_ty.dims[axis]);
        let after = Self::dim_product_expr(&target_ty.dims[axis + 1..]);
        let elem = Self::elem_kind(ty);
        let kernel_name = match indices_ty.precision {
            Prim::Int32 => format!("kernel_scatter_add_i32_{}", elem.suffix()),
            Prim::Int64 => format!("kernel_scatter_add_i64_{}", elem.suffix()),
            _ => unreachable!(),
        };

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!(
            "CHELIS_HIP_CHECK(hipMemcpy(d_t{id}->data, d_t{target}->data, d_t{id}->size * chelis_gpu_dtype_size(d_t{id}->dtype), hipMemcpyDeviceToDevice));"
        ));
        self.line(&format!("int t{id}_before = {before};"));
        self.line(&format!("int t{id}_axis_size = {axis_size};"));
        self.line(&format!("int t{id}_after = {after};"));
        self.line(&format!("int t{id}_index_count = d_t{indices}->size;"));
        self.line(&format!("int t{id}_total = d_t{updates}->size;"));
        self.line(&format!(
            "void *args[] = {{ &d_t{indices}->data, &d_t{updates}->data, &d_t{id}->data, &t{id}_before, &t{id}_axis_size, &t{id}_after, &t{id}_index_count, &t{id}_total }};"
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            &kernel_name,
            &format!("(t{id}_total + 255) / 256"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
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
        dag: &Dag,
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
        let before = Self::dim_product_expr(&target_ty.dims[..axis]);
        let axis_size = Self::emit_dim_info(&target_ty.dims[axis]);
        let after = Self::dim_product_expr(&target_ty.dims[axis + 1..]);
        let kernel_name = match indices_ty.precision {
            Prim::Int32 => "kernel_scatter_replace_i32",
            Prim::Int64 => "kernel_scatter_replace_i64",
            _ => unreachable!(),
        };

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!(
            "CHELIS_HIP_CHECK(hipMemcpy(d_t{id}->data, d_t{target}->data, d_t{id}->size * chelis_gpu_dtype_size(d_t{id}->dtype), hipMemcpyDeviceToDevice));"
        ));
        self.line(&format!("int t{id}_before = {before};"));
        self.line(&format!("int t{id}_axis_size = {axis_size};"));
        self.line(&format!("int t{id}_after = {after};"));
        self.line(&format!("int t{id}_index_count = d_t{indices}->size;"));
        self.line(&format!("int t{id}_total = d_t{updates}->size;"));
        self.line(&format!(
            "void *args[] = {{ &d_t{indices}->data, &d_t{updates}->data, &d_t{id}->data, &t{id}_before, &t{id}_axis_size, &t{id}_after, &t{id}_index_count, &t{id}_total }};"
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
        in_place: Option<FusedInPlaceSpec>,
    ) {
        if let Some(spec) = in_place {
            self.emit_fused_in_place_wrapper(id, ty, spec);
        } else {
            self.emit_slot_wrapper(id, ty);
        }
        self.line("{");
        self.indent += 1;
        self.line(&format!("int t{id}_size = d_t{id}->size;"));

        // Emit stride vars for each external input
        for (i, inp) in inputs.iter().enumerate() {
            let pfx = format!("ext{i}");
            self.emit_stride_vars(id, &pfx, inp.0);
            self.line(&format!("int t{id}_{pfx}_ndim = d_t{}->ndim;", inp.0));
            self.line(&format!(
                "int t{id}_{pfx}_size = d_t{}->storage_size;",
                inp.0
            ));
        }

        // Emit shape vars for output
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int t{id}_out_ndim = d_t{id}->ndim;"));

        // Build args array
        let mut arg_parts = Vec::new();
        for (i, inp) in inputs.iter().enumerate() {
            let pfx = format!("ext{i}");
            arg_parts.push(format!("&d_t{}->data", inp.0));
            arg_parts.push(self.stride_arg_refs(id, &pfx));
            arg_parts.push(format!("&t{id}_{pfx}_ndim"));
            arg_parts.push(format!("&t{id}_{pfx}_size"));
        }
        arg_parts.push(format!("&d_t{id}->data"));
        arg_parts.push(self.shape_arg_refs(id, "out"));
        arg_parts.push(format!("&t{id}_out_ndim"));
        arg_parts.push(format!("&t{id}_size"));

        self.line(&format!("void *args[] = {{ {} }};", arg_parts.join(", ")));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            kernel_name,
            &format!("(t{id}_size + 255) / 256"),
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
    fn emit_fused_in_place_wrapper(&mut self, id: usize, ty: &TensorType, spec: FusedInPlaceSpec) {
        let slot_id = self.slot_id_for_node(id);
        let slot_is_first_owner = self.plan.slot(slot_id).first_owner == NodeId(id);
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        let reusable = spec.reusable_input.0;
        if !self.device_entrypoint_mode && slot_is_first_owner {
            self.line(&format!("chelis_gpu_tensor *chelis_slot{slot_id} = NULL;"));
            if spec.slot_has_later_owner {
                self.line(&format!(
                    "chelis_slot{slot_id} = chelis_gpu_alloc({ndim}, {shape}, {dtype});"
                ));
            }
        }
        self.line(&format!("chelis_gpu_tensor *d_t{id};"));
        self.line(&format!("if (chelis_gpu_is_contiguous(d_t{reusable})) {{"));
        self.indent += 1;
        self.line(&format!(
            "d_t{id} = chelis_gpu_alloc_view({ndim}, {shape}, {dtype}, d_t{reusable}->data, d_t{reusable}->storage_size);"
        ));
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        if !self.device_entrypoint_mode && slot_is_first_owner && !spec.slot_has_later_owner {
            self.line(&format!(
                "chelis_slot{slot_id} = chelis_gpu_alloc({ndim}, {shape}, {dtype});"
            ));
        }
        self.line(&format!(
            "d_t{id} = chelis_gpu_alloc_view({ndim}, {shape}, {dtype}, chelis_slot{slot_id}->data, chelis_slot{slot_id}->storage_size);"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ------------------------------------------------------------------
    // Reduce kernel launch
    // ------------------------------------------------------------------

    fn emit_reduce_launch(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: &Dag,
        kind: kernels::ReduceKind,
    ) {
        let a = inputs[0].0;
        if matches!(kind, kernels::ReduceKind::Sum)
            && let Some(matmul) = blas::detect_matmul_pattern(dag, NodeId(id))
            && Self::supports_static_hipblas_matmul(dag, &matmul, ty)
        {
            self.emit_blas_matmul(
                id,
                &MatmulEmitSpec {
                    a: matmul.a,
                    b: matmul.b,
                    batch_dims: Vec::new(),
                    m: DimExpr::Concrete(matmul.m),
                    n: DimExpr::Concrete(matmul.n),
                    k: DimExpr::Concrete(matmul.k),
                },
                ty,
                dag,
            );
            return;
        }
        let kernel_name = Self::reduction_kernel_name(kind, axis, Self::elem_kind(ty));

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int t{id}_out_size = d_t{id}->size;"));
        self.line(&format!("int t{id}_axis_size = d_t{a}->shape[{axis}];"));
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int t{id}_a_ndim = d_t{a}->ndim;"));
        self.line(&format!("int t{id}_a_size = d_t{a}->storage_size;"));
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int t{id}_out_ndim = d_t{id}->ndim;"));
        self.line(&format!(
            "void *args[] = {{ &d_t{a}->data, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             &d_t{id}->data, {out_shape_refs}, &t{id}_out_ndim, &t{id}_out_size, &t{id}_axis_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            &kernel_name,
            &format!("(t{id}_out_size + 255) / 256"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
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
        dag: &Dag,
    ) {
        let a = inputs[0].0;
        let input_ty = &dag.get(inputs[0]).unwrap().output_type;
        let node_op = &dag.get(NodeId(id)).unwrap().op;
        // All four extra reductions name kernels by the operand
        // precision: Min/Prod return the operand precision (so input ==
        // output), and Argmax/Argmin's output is i64 even though the
        // value being compared is the operand precision. Reading the
        // input precision uniformly avoids the elem_kind panic on the
        // i64 output of the arg-reductions.
        let elem_for_naming = Self::elem_kind(input_ty);
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
        self.line(&format!("int t{id}_out_size = d_t{id}->size;"));
        self.line(&format!("int t{id}_axis_size = d_t{a}->shape[{axis}];"));
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int t{id}_a_ndim = d_t{a}->ndim;"));
        self.line(&format!("int t{id}_a_size = d_t{a}->storage_size;"));
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int t{id}_out_ndim = d_t{id}->ndim;"));
        self.line(&format!(
            "void *args[] = {{ &d_t{a}->data, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             &d_t{id}->data, {out_shape_refs}, &t{id}_out_ndim, &t{id}_out_size, &t{id}_axis_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            &kernel_name,
            &format!("(t{id}_out_size + 255) / 256"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
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
        dag: &Dag,
    ) {
        let fused_node = dag.get(reduction_inputs[0]).unwrap();
        let ext_inputs = &fused_node.inputs;
        let kind = match dag.get(NodeId(id)).unwrap().op {
            RiscOp::Sum { .. } => kernels::ReduceKind::Sum,
            RiscOp::MaxReduce { .. } => kernels::ReduceKind::Max,
            _ => unreachable!("emit_fused_reduce_launch called on non-reduction"),
        };
        let kernel_name = Self::fused_reduction_kernel_name(id, kind);
        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int t{id}_out_size = d_t{id}->size;"));
        self.line(&format!(
            "int t{id}_axis_size = d_t{}->shape[{axis}];",
            reduction_inputs[0].0
        ));

        // Emit stride vars for each external input of the fused chain
        for (i, inp) in ext_inputs.iter().enumerate() {
            let pfx = format!("ext{i}");
            self.emit_stride_vars(id, &pfx, inp.0);
            self.line(&format!("int t{id}_{pfx}_ndim = d_t{}->ndim;", inp.0));
            self.line(&format!(
                "int t{id}_{pfx}_size = d_t{}->storage_size;",
                inp.0
            ));
        }

        // Emit shape vars for output
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int t{id}_out_ndim = d_t{id}->ndim;"));

        // Build args array: ext inputs + output
        let mut arg_parts = Vec::new();
        for (i, inp) in ext_inputs.iter().enumerate() {
            let pfx = format!("ext{i}");
            arg_parts.push(format!("&d_t{}->data", inp.0));
            arg_parts.push(self.stride_arg_refs(id, &pfx));
            arg_parts.push(format!("&t{id}_{pfx}_ndim"));
            arg_parts.push(format!("&t{id}_{pfx}_size"));
        }
        arg_parts.push(format!("&d_t{id}->data"));
        arg_parts.push(self.shape_arg_refs(id, "out"));
        arg_parts.push(format!("&t{id}_out_ndim"));
        arg_parts.push(format!("&t{id}_out_size"));
        arg_parts.push(format!("&t{id}_axis_size"));

        self.line(&format!("void *args[] = {{ {} }};", arg_parts.join(", ")));
        self.emit_kernel_launch_expr(
            &format!("mod_{kernel_name}"),
            &kernel_name,
            &format!("(t{id}_out_size + 255) / 256"),
            "256",
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    #[allow(dead_code)]
    fn emit_blas_matmul(&mut self, id: usize, spec: &MatmulEmitSpec, ty: &TensorType, dag: &Dag) {
        // Defense in depth: chelis_hipblas_sgemm_* wraps hipblasSgemm /
        // hipblasSgemmStridedBatched, both single-precision only. If a non-F32
        // BlasMatmul reaches here it indicates a missing precision filter
        // upstream (the canonical filter is at
        // chelis_ir::specialize::detect_matmul_pattern). Refuse to emit
        // rather than silently miscompile.
        assert_eq!(
            ty.precision,
            Prim::F32,
            "emit_blas_matmul received non-F32 output (precision={:?}) at node {id}; \
             chelis_hipblas_sgemm_* is single-precision only. The upstream specializer \
             in chelis_ir::specialize must keep non-F32 matmul subgraphs on the \
             generic expand+mul+sum path.",
            ty.precision,
        );
        let a = spec.a.0;
        let b = spec.b.0;
        let m_expr = Self::emit_dim_expr(&spec.m);
        let n_expr = Self::emit_dim_expr(&spec.n);
        let k_expr = Self::emit_dim_expr(&spec.k);
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
        if operand_prec != declared_acc {
            // Mixed-precision matmul (e.g. f64 operand with f32 acc) is
            // not in the spec default table; if it appears here, the IR
            // is doing something the WS-A2 backend has not modeled yet.
            // bf16/f16 with f32 accumulator is WS-A3 territory.
            panic!(
                "HIP backend matmul cannot dispatch operand precision `{}` \
                 with explicit accumulator `{}` at node {id}; \
                 spec/04-type-system.md §5.7.1 default for f32/f64 is \
                 operand-matching, and bf16/f16 mixed-precision lands in WS-A3.",
                operand_prec.name(),
                declared_acc.name(),
            );
        }

        let helper_prefix = match operand_prec {
            Prim::F32 => "chelis_hipblas_sgemm",
            Prim::F64 => "chelis_hipblas_dgemm",
            other => panic!(
                "HIP backend matmul for operand precision `{}` is not implemented",
                other.name()
            ),
        };

        if spec.batch_dims.is_empty() {
            self.line(&format!(
                "{helper_prefix}_row_major(d_t{a}, d_t{b}, d_t{id}, {m_expr}, {n_expr}, {k_expr});"
            ));
        } else if let Some(plan) = Self::strided_batched_hipblas_plan(dag, spec, ty) {
            self.line(&format!(
                "{helper_prefix}_strided_batched_row_major(d_t{a}, d_t{b}, d_t{id}, {m_expr}, {n_expr}, {k_expr}, {batch_count}, {a_stride}LL, {b_stride}LL, {out_stride}LL);",
                batch_count = plan.batch_count_expr,
                a_stride = plan.a_batch_stride,
                b_stride = plan.b_batch_stride,
                out_stride = plan.out_batch_stride,
            ));
        } else {
            self.line(&format!(
                "{helper_prefix}_batched_row_major(d_t{a}, d_t{b}, d_t{id}, {m_expr}, {n_expr}, {k_expr});"
            ));
        }
    }

    // ------------------------------------------------------------------
    // Movement ops (host-side metadata, no kernel)
    // ------------------------------------------------------------------

    fn emit_reshape(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        self.emit_alias_view(
            id,
            ty,
            &format!("d_t{a}->data"),
            &format!("d_t{a}->storage_size"),
        );
    }

    fn emit_permute(&mut self, id: usize, axes: &[usize], inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        self.emit_alias_view(
            id,
            ty,
            &format!("d_t{a}->data"),
            &format!("d_t{a}->storage_size"),
        );
        for (new_d, &old_d) in axes.iter().enumerate() {
            self.line(&format!(
                "d_t{id}->strides[{new_d}] = d_t{a}->strides[{old_d}];"
            ));
        }
    }

    fn emit_expand(
        &mut self,
        id: usize,
        axis: usize,
        _size: &DimExpr,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        self.emit_alias_view(
            id,
            ty,
            &format!("d_t{a}->data"),
            &format!("d_t{a}->storage_size"),
        );
        self.line(&format!("if (d_t{id}->ndim == d_t{a}->ndim) {{"));
        self.indent += 1;
        self.line(&format!(
            "for (int d = 0; d < d_t{a}->ndim; d++) d_t{id}->strides[d] = d_t{a}->strides[d];"
        ));
        self.line(&format!("d_t{id}->strides[{axis}] = 0;"));
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        for d in 0..axis {
            self.line(&format!("d_t{id}->strides[{d}] = d_t{a}->strides[{d}];"));
        }
        self.line(&format!("d_t{id}->strides[{axis}] = 0;"));
        self.line(&format!(
            "for (int d = {axis}; d < d_t{a}->ndim; d++) d_t{id}->strides[d+1] = d_t{a}->strides[d];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    fn emit_stride(
        &mut self,
        id: usize,
        stride_factors: &[usize],
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        self.emit_alias_view(
            id,
            ty,
            &format!("d_t{a}->data"),
            &format!("d_t{a}->storage_size"),
        );
        let max_dims = ty.dims.len().max(1);
        for (d, &s) in stride_factors.iter().enumerate() {
            if d >= max_dims {
                break;
            }
            self.line(&format!(
                "d_t{id}->strides[{d}] = d_t{a}->strides[{d}] * {s};"
            ));
        }
    }

    fn emit_store(&mut self, id: usize, name: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        self.emit_alias_view(
            id,
            ty,
            &format!("d_t{a}->data"),
            &format!("d_t{a}->storage_size"),
        );
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

    /// Emit `int t{node_id}_{prefix}_s{0..7} = d_t{src_id}->strides[i];` for MAX_DIM dims.
    fn emit_stride_vars(&mut self, node_id: usize, prefix: &str, src_id: usize) {
        for i in 0..kernels::MAX_DIM {
            self.line(&format!(
                "int t{node_id}_{prefix}_s{i} = ({i} < d_t{src_id}->ndim) ? d_t{src_id}->strides[{i}] : 0;"
            ));
        }
    }

    /// Emit `int t{node_id}_{prefix}_sh{0..7} = d_t{src_id}->shape[i];` for MAX_DIM dims.
    fn emit_shape_vars(&mut self, node_id: usize, prefix: &str, src_id: usize) {
        for i in 0..kernels::MAX_DIM {
            self.line(&format!(
                "int t{node_id}_{prefix}_sh{i} = ({i} < d_t{src_id}->ndim) ? d_t{src_id}->shape[{i}] : 0;"
            ));
        }
    }

    /// Generate `&t{node_id}_{prefix}_s0, &t{node_id}_{prefix}_s1, ...` for args array.
    fn stride_arg_refs(&self, node_id: usize, prefix: &str) -> String {
        (0..kernels::MAX_DIM)
            .map(|i| format!("&t{node_id}_{prefix}_s{i}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Generate `&t{node_id}_{prefix}_sh0, &t{node_id}_{prefix}_sh1, ...` for args array.
    fn shape_arg_refs(&self, node_id: usize, prefix: &str) -> String {
        (0..kernels::MAX_DIM)
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
            "chelis_launch_kernel({module_var}, \"{kernel_name}\", dim3({grid_expr}), dim3({block_expr}), {args_var});"
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
    fn supports_static_hipblas_matmul(dag: &Dag, info: &blas::MatmulInfo, ty: &TensorType) -> bool {
        if !matches!(ty.precision, Prim::F32 | Prim::F64) || ty.dims.len() != 2 {
            return false;
        }
        let prec = ty.precision;
        dag.get(info.a).map(|n| n.output_type.precision) == Some(prec)
            && dag.get(info.b).map(|n| n.output_type.precision) == Some(prec)
            && Self::node_is_statically_contiguous(dag, info.a)
            && Self::node_is_statically_contiguous(dag, info.b)
    }

    fn strided_batched_hipblas_plan(
        dag: &Dag,
        spec: &MatmulEmitSpec,
        ty: &TensorType,
    ) -> Option<StridedBatchedMatmulPlan> {
        if spec.batch_dims.is_empty()
            || !matches!(ty.precision, Prim::F32 | Prim::F64)
            || !spec.batch_dims.iter().all(Self::is_simple_runtime_dim)
        {
            return None;
        }

        let m = spec.m.as_concrete()?;
        let n = spec.n.as_concrete()?;
        let k = spec.k.as_concrete()?;
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

        Some(StridedBatchedMatmulPlan {
            batch_count_expr: Self::batch_count_expr(&spec.batch_dims),
            a_batch_stride: m * k,
            b_batch_stride: k * n,
            out_batch_stride: m * n,
        })
    }

    fn is_simple_runtime_dim(expr: &DimExpr) -> bool {
        matches!(expr, DimExpr::Concrete(_) | DimExpr::Sym(_))
    }

    fn batch_count_expr(batch_dims: &[DimExpr]) -> String {
        batch_dims
            .iter()
            .map(Self::emit_dim_expr)
            .reduce(|lhs, rhs| format!("({lhs} * {rhs})"))
            .unwrap_or_else(|| "1".to_string())
    }

    fn dim_product_expr(dims: &[DimInfo]) -> String {
        dims.iter()
            .map(Self::emit_dim_info)
            .reduce(|lhs, rhs| format!("({lhs} * {rhs})"))
            .unwrap_or_else(|| "1".to_string())
    }

    #[allow(dead_code)]
    fn supports_staged_scalar_reduction(
        node: &DagNode,
        input_node: &DagNode,
        axis: usize,
        dag: &Dag,
    ) -> bool {
        Self::total_size(&node.output_type) == 1
            && input_node.output_type.dims.len() <= 1
            && axis == 0
            && Self::node_is_statically_contiguous(dag, input_node.id)
    }

    #[allow(dead_code)]
    fn node_is_statically_contiguous(dag: &Dag, id: NodeId) -> bool {
        match &dag.get(id).unwrap().op {
            RiscOp::Load { .. }
            | RiscOp::Const { .. }
            | RiscOp::Add
            | RiscOp::Mul
            | RiscOp::MaxElem
            | RiscOp::CmpLt
            | RiscOp::Neg
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
            | RiscOp::UniformLike { .. }
            | RiscOp::Dropout { .. }
            | RiscOp::Copy
            | RiscOp::Drop
            | RiscOp::Sum { .. }
            | RiscOp::MaxReduce { .. }
            | RiscOp::MinReduce { .. }
            | RiscOp::ProdReduce { .. }
            | RiscOp::Argmax { .. }
            | RiscOp::Argmin { .. }
            | RiscOp::OneHot { .. }
            | RiscOp::Realize
            | RiscOp::Cast { .. }
            | RiscOp::FusedElem { .. }
            | RiscOp::BlasMatmul { .. }
            | RiscOp::Gather { .. }
            | RiscOp::ScatterAdd { .. }
            | RiscOp::Scatter { .. } => true,
            RiscOp::Reshape { .. } | RiscOp::Store { .. } => {
                Self::node_is_statically_contiguous(dag, dag.get(id).unwrap().inputs[0])
            }
            RiscOp::Permute { .. }
            | RiscOp::Expand { .. }
            | RiscOp::Stride { .. }
            | RiscOp::Pad { .. }
            | RiscOp::Shrink { .. } => false,
        }
    }

    #[allow(dead_code)]
    fn bytes_per_element(dtype: Prim) -> usize {
        match dtype {
            Prim::F32 | Prim::Bool | Prim::Int32 => 4,
            Prim::F64 | Prim::Int64 => 8,
            other => panic!(
                "unsupported HIP dtype in device-memory estimate: {}",
                other.name()
            ),
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

    fn shape_literal(ty: &TensorType) -> String {
        let dims: Vec<String> = ty.dims.iter().map(Self::emit_dim_info).collect();
        if dims.is_empty() {
            "(int[]){1}".to_string()
        } else {
            format!("(int[]){{ {} }}", dims.join(", "))
        }
    }

    fn ndim(ty: &TensorType) -> usize {
        if ty.dims.is_empty() { 1 } else { ty.dims.len() }
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

    fn emit_dim_expr(expr: &DimExpr) -> String {
        match expr {
            DimExpr::Concrete(n) => n.to_string(),
            DimExpr::Sym(name) => name.clone(),
            DimExpr::Mul(lhs, rhs) => {
                format!(
                    "({} * {})",
                    Self::emit_dim_expr(lhs),
                    Self::emit_dim_expr(rhs)
                )
            }
            DimExpr::Div(lhs, rhs) => {
                format!(
                    "({} / {})",
                    Self::emit_dim_expr(lhs),
                    Self::emit_dim_expr(rhs)
                )
            }
        }
    }

    #[allow(dead_code)]
    fn total_size(ty: &TensorType) -> usize {
        if ty.dims.is_empty() {
            1
        } else {
            ty.dims
                .iter()
                .map(|dim| {
                    Self::known_dim_size(dim)
                        .unwrap_or_else(|| panic!("unsized named dimension in static total_size"))
                })
                .product()
        }
    }

    fn dtype_macro(ty: &TensorType) -> &'static str {
        match ty.precision {
            Prim::F32 => "CHELIS_F32",
            Prim::F64 => "CHELIS_F64",
            Prim::Bool => "CHELIS_BOOL",
            Prim::Int32 => "CHELIS_I32",
            Prim::Int64 => "CHELIS_I64",
            other => panic!(
                "HIP backend supports f32/f64/bool/int32/int64 tensors today; \
                 got `{}`. bf16/f16 land in WS-A3; i8/i16 in WS-A4.",
                other.name()
            ),
        }
    }

    /// Map a tensor's precision to a [`kernels::ElemKind`] for kernel
    /// emission. The HIP runtime stores `bool` tensors as 4-byte values
    /// (per `chelis_gpu_dtype_size`), so kernels emit them under the
    /// `f32` path — the cmplt convention is `1.0f`/`0.0f` written into a
    /// `float *` GPU buffer, and Cast-to-bool likewise materializes a
    /// 4-byte payload. f64 gets its own variant. Other precisions fall
    /// outside the WS-A2 scope (bf16/f16 in WS-A3; i8/i16 in WS-A4) and
    /// panic so callers see the limit immediately.
    fn elem_kind(ty: &TensorType) -> kernels::ElemKind {
        match ty.precision {
            Prim::F32 | Prim::Bool => kernels::ElemKind::F32,
            Prim::F64 => kernels::ElemKind::F64,
            other => panic!(
                "HIP backend kernel emission expected a floating precision, got `{}`",
                other.name()
            ),
        }
    }

    // ------------------------------------------------------------------
    // Input/output label logic (identical to C backend)
    // ------------------------------------------------------------------

    pub fn input_labels(dag: &Dag) -> Vec<String> {
        let mut labels = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for node in dag.nodes() {
            if let RiscOp::Load { name } = &node.op
                && seen.insert(name.as_str().to_string())
            {
                labels.push(name.as_str().to_string());
            }
        }
        labels
    }

    pub fn output_labels(dag: &Dag) -> Vec<String> {
        Self::output_specs(dag)
            .into_iter()
            .map(|o| o.label)
            .collect()
    }

    fn output_specs(dag: &Dag) -> Vec<OutputSpec> {
        let mut specs = Vec::new();
        let mut seen = std::collections::HashSet::new();

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

    fn input_slots(labels: &[String]) -> std::collections::HashMap<String, usize> {
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
    use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};

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

    fn mat_f32(rows: usize, cols: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
            precision: Prim::F32,
        }
    }

    fn fused_mul_reusable_input_dag() -> Dag {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let scale = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4), None);
        let ops = vec![FusedStep {
            op: FusedStepOp::Mul,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        let fused = dag.add_node(RiscOp::FusedElem { ops }, vec![x, scale], vec_f32(4), None);
        dag.set_reusable_input(fused, x);
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

        let (hip, _) = HipEmitter::emit_dag(&dag, "test_sparse");

        assert!(hip.contains("kernel_scatter_add_i64"));
        assert!(hip.contains("const long long *indices"));
        assert!(hip.contains("atomicAdd(&out[dst], updates[i]);"));
    }

    /// Perf-F2(b) shipped: with a single-consumer reusable input
    /// (here `x`, the FusedElem's first external), the kernel ships
    /// the in-place shape — `__restrict__` only on non-aliased
    /// externals (`ext1`), never on the aliased external (`ext0`) or
    /// `out`.
    #[test]
    fn fused_reusable_input_emits_hip_in_place_restrict_shape() {
        let dag = fused_mul_reusable_input_dag();
        let (hip, _) = HipEmitter::emit_dag(&dag, "test_fn");

        assert!(hip.contains("extern \\\"C\\\" __global__ void kernel_fused_2("));
        // Aliased external (ext0 ↔ x) must NOT carry __restrict__.
        assert!(!hip.contains("const float *__restrict__ ext0"));
        // Output must NOT carry __restrict__ — it aliases ext0.
        assert!(!hip.contains("float *__restrict__ out"));
        // Non-aliased externals (ext1 here, the const scale) MUST
        // carry __restrict__ in the in-place shape.
        assert!(hip.contains("const float *__restrict__ ext1"));
    }

    /// Negative: a FusedElem with no `reusable_input` set must still
    /// emit the legacy non-`__restrict__` kernel parameter list.
    /// The in-place shape is opt-in via the upstream linearity-marked
    /// hint, not the default.
    #[test]
    fn fused_without_reusable_input_keeps_non_in_place_kernel_shape() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let scale = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4), None);
        let ops = vec![FusedStep {
            op: FusedStepOp::Mul,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        let fused = dag.add_node(RiscOp::FusedElem { ops }, vec![x, scale], vec_f32(4), None);
        // No set_reusable_input call — the in-place gate must reject.
        dag.add_root(fused);

        let (hip, _) = HipEmitter::emit_dag(&dag, "test_fn");

        assert!(hip.contains("extern \\\"C\\\" __global__ void kernel_fused_2("));
        assert!(hip.contains("const float *ext0"));
        assert!(hip.contains("const float *ext1"));
        assert!(hip.contains("float *out"));
        assert!(
            !hip.contains("__restrict__"),
            "no-reusable-input fused kernels must keep the legacy non-__restrict__ shape"
        );
    }
}
