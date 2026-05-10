//! RISC DAG to HIP host code + kernel string emission.
//!
//! Generates C source that includes HIP runtime, embeds kernel source strings,
//! and walks the DAG in topological order launching kernels on GPU.

use chelis_ir::dag::{
    Dag, DagNode, DimExpr, DimInfo, NodeId, RiscOp, TensorType, symbolic_bindings,
};
use chelis_types::types::Prim;

use crate::blas;
use crate::kernels;
use crate::launch;
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

impl HipEmitter {
    /// Emit complete C/HIP source for a DAG as a function.
    pub(crate) fn emit_dag(dag: &Dag, func_name: &str) -> (String, PeakDeviceBytesBreakdown) {
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
                RiscOp::Sum { axis } | RiscOp::MaxReduce { axis } => {
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
                    // Phase 3j-pre: these ops ship in the C backend only.
                    // HIP backend support is explicitly deferred; calling
                    // hip codegen on a DAG containing them should be a loud
                    // failure, not silent miscompilation.
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
            let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
            let name = Self::fused_reduction_kernel_name(node.id.0, kind);
            let source = kernels::reduce_fused(&name, axis, steps, n_ext, kind);
            return vec![(name, source)];
        }

        if matches!(kind, kernels::ReduceKind::Sum)
            && let Some(matmul) = blas::detect_matmul_pattern(dag, node.id)
            && Self::supports_static_hipblas_matmul(dag, &matmul, &node.output_type)
        {
            return Vec::new();
        }

        let name = Self::reduction_kernel_name(kind, axis);
        let source = match kind {
            kernels::ReduceKind::Sum => kernels::reduce_sum(&name, axis),
            kernels::ReduceKind::Max => kernels::reduce_max(&name, axis),
        };
        vec![(name, source)]
    }

    fn kernel_name_for_op(&self, op: &RiscOp, node: &DagNode, _dag: &Dag) -> Option<String> {
        match op {
            RiscOp::Add => Some("kernel_add".into()),
            RiscOp::Mul => Some("kernel_mul".into()),
            RiscOp::MaxElem => Some("kernel_max_elem".into()),
            RiscOp::CmpLt => Some("kernel_cmplt".into()),
            RiscOp::Neg => Some("kernel_neg".into()),
            RiscOp::Exp => Some("kernel_exp".into()),
            RiscOp::Log => Some("kernel_log".into()),
            RiscOp::Sin => Some("kernel_sin".into()),
            RiscOp::Sqrt => Some("kernel_sqrt".into()),
            RiscOp::Cos => Some("kernel_cos".into()),
            RiscOp::Tan => Some("kernel_tan".into()),
            RiscOp::Atan => Some("kernel_atan".into()),
            RiscOp::Abs => Some("kernel_abs".into()),
            RiscOp::Floor => Some("kernel_floor".into()),
            RiscOp::Ceil => Some("kernel_ceil".into()),
            RiscOp::UniformLike { .. } => Some("kernel_uniform_like".into()),
            RiscOp::Dropout { .. } | RiscOp::Drop => None,
            RiscOp::Copy => Some("kernel_cast".into()),
            RiscOp::Sum { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    Some(Self::fused_reduction_kernel_name(
                        node.id.0,
                        kernels::ReduceKind::Sum,
                    ))
                } else {
                    Some(Self::reduction_kernel_name(kernels::ReduceKind::Sum, *axis))
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
                    Some(Self::reduction_kernel_name(kernels::ReduceKind::Max, *axis))
                }
            }
            RiscOp::MinReduce { .. }
            | RiscOp::ProdReduce { .. }
            | RiscOp::Argmax { .. }
            | RiscOp::Argmin { .. } => None,
            RiscOp::Const { .. } => Some("kernel_fill".into()),
            RiscOp::Realize => Some("kernel_cast".into()),
            RiscOp::Cast { .. } => Some("kernel_cast".into()),
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
            RiscOp::FusedElem { .. } => Some(format!("kernel_fused_{}", node.id.0)),
        }
    }

    fn kernel_source_for_op(&self, name: &str, op: &RiscOp, node: &DagNode, dag: &Dag) -> String {
        match op {
            RiscOp::Add => kernels::binary_elementwise(name, "+"),
            RiscOp::Mul => kernels::binary_elementwise(name, "*"),
            RiscOp::MaxElem => kernels::binary_func(name, "fmaxf"),
            RiscOp::CmpLt => kernels::cmplt(name),
            RiscOp::Neg => kernels::unary_prefix(name, "-"),
            RiscOp::Exp => kernels::unary_func(name, "expf"),
            RiscOp::Log => kernels::unary_func(name, "logf"),
            RiscOp::Sin => kernels::unary_func(name, "sinf"),
            RiscOp::Sqrt => kernels::unary_func(name, "sqrtf"),
            RiscOp::Cos => kernels::unary_func(name, "cosf"),
            RiscOp::Tan => kernels::unary_func(name, "tanf"),
            RiscOp::Atan => kernels::unary_func(name, "atanf"),
            RiscOp::Abs => kernels::unary_func(name, "fabsf"),
            RiscOp::Floor => kernels::unary_func(name, "floorf"),
            RiscOp::Ceil => kernels::unary_func(name, "ceilf"),
            RiscOp::UniformLike { .. } => kernels::uniform_like(name),
            RiscOp::Sum { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
                    kernels::reduce_fused(name, *axis, steps, n_ext, kernels::ReduceKind::Sum)
                } else {
                    kernels::reduce_sum(name, *axis)
                }
            }
            RiscOp::MaxReduce { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
                    kernels::reduce_fused(name, *axis, steps, n_ext, kernels::ReduceKind::Max)
                } else {
                    kernels::reduce_max(name, *axis)
                }
            }
            RiscOp::Const { .. } => kernels::fill(name),
            RiscOp::Realize => kernels::cast(name),
            RiscOp::Cast { .. } => kernels::cast(name),
            RiscOp::Copy => kernels::cast(name),
            RiscOp::FusedElem { ops } => kernels::fused_elementwise(name, ops, node.inputs.len()),
            _ => unreachable!("no kernel for op: {op:?}"),
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
        match &node.op {
            RiscOp::Const { value } => self.emit_const(id, *value, &node.output_type),
            RiscOp::Load { .. } => unreachable!("handled in emit_dag"),
            RiscOp::Add => {
                self.emit_binary_launch(id, "kernel_add", &node.inputs, &node.output_type)
            }
            RiscOp::Mul => {
                self.emit_binary_launch(id, "kernel_mul", &node.inputs, &node.output_type)
            }
            RiscOp::MaxElem => {
                self.emit_binary_launch(id, "kernel_max_elem", &node.inputs, &node.output_type);
            }
            RiscOp::CmpLt => {
                self.emit_binary_launch(id, "kernel_cmplt", &node.inputs, &node.output_type);
            }
            RiscOp::Neg => {
                self.emit_unary_launch(id, "kernel_neg", &node.inputs, &node.output_type)
            }
            RiscOp::Exp => {
                self.emit_unary_launch(id, "kernel_exp", &node.inputs, &node.output_type)
            }
            RiscOp::Log => {
                self.emit_unary_launch(id, "kernel_log", &node.inputs, &node.output_type)
            }
            RiscOp::Sin => {
                self.emit_unary_launch(id, "kernel_sin", &node.inputs, &node.output_type)
            }
            RiscOp::Sqrt => {
                self.emit_unary_launch(id, "kernel_sqrt", &node.inputs, &node.output_type)
            }
            RiscOp::Cos => {
                self.emit_unary_launch(id, "kernel_cos", &node.inputs, &node.output_type)
            }
            RiscOp::Tan => {
                self.emit_unary_launch(id, "kernel_tan", &node.inputs, &node.output_type)
            }
            RiscOp::Atan => {
                self.emit_unary_launch(id, "kernel_atan", &node.inputs, &node.output_type)
            }
            RiscOp::Abs => {
                self.emit_unary_launch(id, "kernel_abs", &node.inputs, &node.output_type)
            }
            RiscOp::Floor => {
                self.emit_unary_launch(id, "kernel_floor", &node.inputs, &node.output_type)
            }
            RiscOp::Ceil => {
                self.emit_unary_launch(id, "kernel_ceil", &node.inputs, &node.output_type)
            }
            RiscOp::UniformLike { low, high, seed } => {
                self.emit_uniform_like_launch(id, *low, *high, *seed, &node.output_type)
            }
            RiscOp::Dropout { .. } => {
                unreachable!("dropout should be rejected before HIP code generation")
            }
            RiscOp::Copy => {
                self.emit_unary_launch(id, "kernel_cast", &node.inputs, &node.output_type);
            }
            RiscOp::Drop => {}
            RiscOp::Sum { axis } => {
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
            RiscOp::MinReduce { .. } => {
                panic!(
                    "HIP backend: MinReduce is not yet supported (Phase 3j-pre ships C backend only)"
                );
            }
            RiscOp::ProdReduce { .. } => {
                panic!(
                    "HIP backend: ProdReduce is not yet supported (Phase 3j-pre ships C backend only)"
                );
            }
            RiscOp::Argmax { .. } => {
                panic!(
                    "HIP backend: Argmax is not yet supported (Phase 3j-pre ships C backend only)"
                );
            }
            RiscOp::Argmin { .. } => {
                panic!(
                    "HIP backend: Argmin is not yet supported (Phase 3j-pre ships C backend only)"
                );
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
                todo!("Pad on GPU requires a kernel — deferred to Phase 1a iteration 2")
            }
            RiscOp::Shrink { .. } => {
                todo!("Shrink on GPU requires a kernel — deferred to Phase 1a iteration 2")
            }
            RiscOp::Stride { strides } => {
                self.emit_stride(id, strides, &node.inputs, &node.output_type);
            }
            RiscOp::Realize => {
                self.emit_unary_launch(id, "kernel_cast", &node.inputs, &node.output_type);
            }
            RiscOp::Cast { .. } => {
                self.emit_unary_launch(id, "kernel_cast", &node.inputs, &node.output_type);
            }
            RiscOp::Store { name } => {
                self.emit_store(id, name.as_str(), &node.inputs, &node.output_type)
            }
            RiscOp::FusedElem { ops } => {
                let kernel_name = format!("kernel_fused_{}", node.id.0);
                self.emit_fused_launch(
                    node.id.0,
                    &kernel_name,
                    &node.inputs,
                    ops,
                    &node.output_type,
                );
            }
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
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
                );
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
        self.line("{");
        self.indent += 1;
        self.line(&format!("float fill_val = {:.8}f;", value as f32));
        self.line(&format!("int fill_size = d_t{id}->size;"));
        self.line(&format!(
            "void *fill_args[] = {{ &d_t{id}->data, &fill_val, &fill_size }};"
        ));
        self.emit_kernel_launch_expr(
            "mod_kernel_fill",
            "kernel_fill",
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
        self.line("{");
        self.indent += 1;
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
            "mod_kernel_uniform_like",
            "kernel_uniform_like",
            &format!("(t{id}_size + 255) / 256"),
            "256",
            "args",
        );
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
    ) {
        self.emit_slot_wrapper(id, ty);
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
            );
            return;
        }
        let kernel_name = Self::reduction_kernel_name(kind, axis);

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
    fn emit_blas_matmul(&mut self, id: usize, spec: &MatmulEmitSpec, ty: &TensorType) {
        let a = spec.a.0;
        let b = spec.b.0;
        let m_expr = Self::emit_dim_expr(&spec.m);
        let n_expr = Self::emit_dim_expr(&spec.n);
        let k_expr = Self::emit_dim_expr(&spec.k);
        self.emit_slot_wrapper(id, ty);
        if spec.batch_dims.is_empty() {
            self.line(&format!(
                "chelis_hipblas_sgemm_row_major(d_t{a}, d_t{b}, d_t{id}, {m_expr}, {n_expr}, {k_expr});"
            ));
        } else {
            self.line(&format!(
                "chelis_hipblas_sgemm_batched_row_major(d_t{a}, d_t{b}, d_t{id}, {m_expr}, {n_expr}, {k_expr});"
            ));
        }
    }

    #[allow(dead_code)]
    fn emit_staged_scalar_reduce_launch(
        &mut self,
        id: usize,
        input_id: usize,
        input_node: &DagNode,
        ty: &TensorType,
        kind: kernels::ReduceKind,
    ) {
        let total_size = Self::total_size(&input_node.output_type);
        let block = launch::reduction_block_size(total_size);
        let counts = launch::staged_chain_partial_counts(total_size, block);
        let scratch_elems = launch::staged_chain_scratch_elements(total_size, block);
        self.extra_peak_device_bytes_estimate = self
            .extra_peak_device_bytes_estimate
            .max(scratch_elems * Self::bytes_per_element(ty.precision));
        let stage1_name = Self::scalar_stage1_kernel_name(kind, block);
        let stage_n_name = Self::scalar_stage_n_kernel_name(kind, block);

        self.emit_slot_wrapper(id, ty);
        self.line("{");
        self.indent += 1;
        self.line(&format!("int t{id}_stage0_total = {total_size};"));
        for (stage, &partials) in counts.iter().take_while(|&&c| c > 1).enumerate() {
            self.line(&format!("float *t{id}_partials{stage} = NULL;"));
            self.line(&format!(
                "CHELIS_HIP_CHECK(hipMalloc(&t{id}_partials{stage}, {partials} * sizeof(float)));"
            ));
        }
        if counts[0] > 1 {
            self.line(&format!(
                "void *stage0_args[] = {{ &d_t{input_id}->data, &t{id}_partials0, &t{id}_stage0_total }};"
            ));
            self.emit_kernel_launch(
                &format!("mod_{stage1_name}"),
                &stage1_name,
                counts[0],
                block,
                "stage0_args",
            );
            for stage in 1..counts.len() {
                let in_count = counts[stage - 1];
                let out_count = counts[stage];
                self.line(&format!("int t{id}_stage{stage}_total = {in_count};"));
                let in_ptr = if stage == 1 {
                    format!("t{id}_partials0")
                } else {
                    format!("t{id}_partials{}", stage - 1)
                };
                let out_ptr = if out_count == 1 {
                    format!("d_t{id}->data")
                } else {
                    format!("t{id}_partials{stage}")
                };
                self.line(&format!(
                    "void *stage{stage}_args[] = {{ &{in_ptr}, &{out_ptr}, &t{id}_stage{stage}_total }};"
                ));
                self.emit_kernel_launch(
                    &format!("mod_{stage_n_name}"),
                    &stage_n_name,
                    out_count,
                    block,
                    &format!("stage{stage}_args"),
                );
            }
            for stage in 0..counts.iter().take_while(|&&c| c > 1).count() {
                self.line(&format!(
                    "CHELIS_HIP_CHECK(hipFree(t{id}_partials{stage}));"
                ));
            }
        } else {
            self.line(&format!(
                "void *stage0_args[] = {{ &d_t{input_id}->data, &d_t{id}->data, &t{id}_stage0_total }};"
            ));
            self.emit_kernel_launch(
                &format!("mod_{stage1_name}"),
                &stage1_name,
                1,
                block,
                "stage0_args",
            );
        }
        self.indent -= 1;
        self.line("}");
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

    #[allow(dead_code)]
    fn emit_kernel_launch(
        &mut self,
        module_var: &str,
        kernel_name: &str,
        grid: usize,
        block: usize,
        args_var: &str,
    ) {
        self.emit_kernel_launch_expr(
            module_var,
            kernel_name,
            &grid.to_string(),
            &block.to_string(),
            args_var,
        );
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

    fn reduction_kernel_name(kind: kernels::ReduceKind, axis: usize) -> String {
        let op = match kind {
            kernels::ReduceKind::Sum => "sum",
            kernels::ReduceKind::Max => "maxred",
        };
        format!("kernel_{op}_ax{axis}")
    }

    fn fused_reduction_kernel_name(id: usize, kind: kernels::ReduceKind) -> String {
        let op = match kind {
            kernels::ReduceKind::Sum => "fused_sum",
            kernels::ReduceKind::Max => "fused_maxred",
        };
        format!("kernel_{op}_{id}")
    }

    #[allow(dead_code)]
    fn scalar_stage1_kernel_name(kind: kernels::ReduceKind, block: usize) -> String {
        let op = match kind {
            kernels::ReduceKind::Sum => "sum",
            kernels::ReduceKind::Max => "maxred",
        };
        format!("kernel_{op}_scalar_stage1_bs{block}")
    }

    #[allow(dead_code)]
    fn scalar_stage_n_kernel_name(kind: kernels::ReduceKind, block: usize) -> String {
        let op = match kind {
            kernels::ReduceKind::Sum => "sum",
            kernels::ReduceKind::Max => "maxred",
        };
        format!("kernel_{op}_scalar_stage_n_bs{block}")
    }

    #[allow(dead_code)]
    fn supports_static_hipblas_matmul(dag: &Dag, info: &blas::MatmulInfo, ty: &TensorType) -> bool {
        ty.precision == Prim::F32
            && ty.dims.len() == 2
            && dag.get(info.a).map(|n| n.output_type.precision) == Some(Prim::F32)
            && dag.get(info.b).map(|n| n.output_type.precision) == Some(Prim::F32)
            && Self::node_is_statically_contiguous(dag, info.a)
            && Self::node_is_statically_contiguous(dag, info.b)
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
            | RiscOp::Realize
            | RiscOp::Cast { .. }
            | RiscOp::FusedElem { .. }
            | RiscOp::BlasMatmul { .. } => true,
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
            Prim::F32 | Prim::Bool => 4,
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
            Prim::Bool => "CHELIS_BOOL",
            other => panic!(
                "Phase 1a HIP backend only supports f32/bool tensors, got {}",
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
