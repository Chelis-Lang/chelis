//! RISC DAG to HIP host code + kernel string emission.
//!
//! Generates C source that includes HIP runtime, embeds kernel source strings,
//! and walks the DAG in topological order launching kernels on GPU.

use chelis_ir::dag::{Dag, DagNode, DimInfo, NodeId, RiscOp, TensorType};
use chelis_types::types::Prim;

use crate::kernels;
use crate::launch;

/// Emits C/HIP host source code from a RISC DAG.
pub struct HipEmitter {
    lines: Vec<String>,
    indent: usize,
    /// Collected kernel sources: (kernel_name, kernel_source_string).
    kernel_sources: Vec<(String, String)>,
    /// Track which nodes are views (movement ops) vs allocations.
    views: std::collections::HashSet<usize>,
    /// FusedElem nodes inlined into a trailing reduction (no standalone emission).
    reduction_inlined: std::collections::HashSet<usize>,
}

#[derive(Debug, Clone)]
#[expect(dead_code)]
struct OutputSpec {
    id: NodeId,
    label: String,
    is_store: bool,
}

impl HipEmitter {
    /// Emit complete C/HIP source for a DAG as a function.
    pub fn emit_dag(dag: &Dag, func_name: &str) -> String {
        let reduction_inlined = chelis_ir::fuse::reduction_inlined_fused_elems(dag);
        let mut e = HipEmitter {
            lines: Vec::new(),
            indent: 0,
            kernel_sources: Vec::new(),
            views: std::collections::HashSet::new(),
            reduction_inlined: reduction_inlined.iter().map(|id| id.0).collect(),
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
        let output_specs = Self::output_specs(dag);
        let expected_inputs = input_labels.len();
        let expected_outputs = output_specs.len();

        e.line(&format!(
            "void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out) {{"
        ));
        e.indent = 1;

        // Input/output count validation
        e.line(&format!("if (n_in != {expected_inputs}) {{"));
        e.indent += 1;
        e.line(&format!(
            "fprintf(stderr, \"{func_name}: expected %d inputs, got %d\\n\", {expected_inputs}, n_in);"
        ));
        e.line("abort();");
        e.indent -= 1;
        e.line("}");

        e.line(&format!("if (n_out != {expected_outputs}) {{"));
        e.indent += 1;
        e.line(&format!(
            "fprintf(stderr, \"{func_name}: expected %d outputs, got %d\\n\", {expected_outputs}, n_out);"
        ));
        e.line("abort();");
        e.indent -= 1;
        e.line("}");

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
                    .get(name)
                    .unwrap_or_else(|| panic!("missing input slot for load '{name}'"));
                e.emit_load(node.id.0, input_idx);
            } else {
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
                // Load is already a host tensor — look up the correct input slot by name
                let load_name = match &dag.get(output.id).unwrap().op {
                    RiscOp::Load { name } => name.clone(),
                    _ => unreachable!(),
                };
                let input_idx = input_slots
                    .get(&load_name)
                    .unwrap_or_else(|| panic!("missing input slot for load '{load_name}'"));
                e.line(&format!("outputs[{slot}] = inputs[{input_idx}];"));
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
        let output_ids: Vec<NodeId> = output_specs.iter().map(|o| o.id).collect();
        let skip_ids: Vec<NodeId> = e.reduction_inlined.iter().map(|&id| NodeId(id)).collect();
        let cleanup = crate::memory::emit_cleanup_with_skip(dag, &output_ids, &skip_ids);
        for line in cleanup {
            e.lines.push(line);
        }

        e.indent = 0;
        e.line("}");
        e.lines.join("\n")
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
            let name = self.kernel_name_for_op(&node.op, node, dag);
            if let Some(name) = name
                && seen.insert(name.clone())
            {
                let source = self.kernel_source_for_op(&name, &node.op, node, dag);
                self.kernel_sources.push((name, source));
            }
        }
    }

    fn kernel_name_for_op(&self, op: &RiscOp, node: &DagNode, dag: &Dag) -> Option<String> {
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
            RiscOp::Sum { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    // Fused reduction: unique kernel per node ID
                    Some(format!("kernel_fused_sum_{}", node.id.0))
                } else {
                    let input_node = dag.get(input_id).unwrap();
                    let axis_size = Self::dim_size(&input_node.output_type.dims[*axis]);
                    Some(format!("kernel_sum_ax{axis}_sz{axis_size}"))
                }
            }
            RiscOp::MaxReduce { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    Some(format!("kernel_fused_maxred_{}", node.id.0))
                } else {
                    let input_node = dag.get(input_id).unwrap();
                    let axis_size = Self::dim_size(&input_node.output_type.dims[*axis]);
                    Some(format!("kernel_maxred_ax{axis}_sz{axis_size}"))
                }
            }
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
            | RiscOp::Stride { .. } => None,
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
            RiscOp::Sum { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
                    // axis_size comes from the FusedElem's output type (the reduction input shape)
                    let axis_size = Self::dim_size(&fused_node.output_type.dims[*axis]);
                    kernels::reduce_fused(
                        name,
                        *axis,
                        axis_size,
                        steps,
                        n_ext,
                        kernels::ReduceKind::Sum,
                    )
                } else {
                    let input_node = dag.get(input_id).unwrap();
                    let axis_size = Self::dim_size(&input_node.output_type.dims[*axis]);
                    kernels::reduce_sum(name, *axis, axis_size)
                }
            }
            RiscOp::MaxReduce { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    let (steps, n_ext) = Self::extract_fused_steps(&fused_node.op);
                    let axis_size = Self::dim_size(&fused_node.output_type.dims[*axis]);
                    kernels::reduce_fused(
                        name,
                        *axis,
                        axis_size,
                        steps,
                        n_ext,
                        kernels::ReduceKind::Max,
                    )
                } else {
                    let input_node = dag.get(input_id).unwrap();
                    let axis_size = Self::dim_size(&input_node.output_type.dims[*axis]);
                    kernels::reduce_max(name, *axis, axis_size)
                }
            }
            RiscOp::Const { .. } => kernels::fill(name),
            RiscOp::Realize => kernels::cast(name),
            RiscOp::Cast { .. } => kernels::cast(name),
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
            RiscOp::Sum { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    let kname = format!("kernel_fused_sum_{id}");
                    self.emit_fused_reduce_launch(
                        id,
                        &kname,
                        &fused_node.inputs.clone(),
                        &node.output_type,
                    );
                } else {
                    let input_node = dag.get(input_id).unwrap();
                    let axis_size = Self::dim_size(&input_node.output_type.dims[*axis]);
                    let kname = format!("kernel_sum_ax{axis}_sz{axis_size}");
                    self.emit_reduce_launch(id, &kname, &node.inputs, &node.output_type, dag);
                }
            }
            RiscOp::MaxReduce { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    let kname = format!("kernel_fused_maxred_{id}");
                    self.emit_fused_reduce_launch(
                        id,
                        &kname,
                        &fused_node.inputs.clone(),
                        &node.output_type,
                    );
                } else {
                    let input_node = dag.get(input_id).unwrap();
                    let axis_size = Self::dim_size(&input_node.output_type.dims[*axis]);
                    let kname = format!("kernel_maxred_ax{axis}_sz{axis_size}");
                    self.emit_reduce_launch(id, &kname, &node.inputs, &node.output_type, dag);
                }
            }
            RiscOp::Reshape { .. } => {
                self.emit_reshape(id, &node.inputs, &node.output_type);
            }
            RiscOp::Permute { axes } => {
                self.emit_permute(id, axes, &node.inputs, &node.output_type);
            }
            RiscOp::Expand { axis, size } => {
                self.emit_expand(id, *axis, *size, &node.inputs, &node.output_type);
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
            RiscOp::Store { name } => self.emit_store(id, name, &node.inputs),
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
        }
    }

    // ------------------------------------------------------------------
    // Const (fill kernel)
    // ------------------------------------------------------------------

    fn emit_const(&mut self, id: usize, value: f64, ty: &TensorType) {
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        let (grid, block) = launch::grid_1d(Self::total_size(ty));
        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("{");
        self.indent += 1;
        self.line(&format!("float fill_val = {:.8}f;", value as f32));
        self.line(&format!("int fill_size = d_t{id}->storage_size;"));
        self.line(&format!(
            "void *fill_args[] = {{ &d_t{id}->data, &fill_val, &fill_size }};"
        ));
        self.emit_kernel_launch("mod_kernel_fill", "kernel_fill", grid, block, "fill_args");
        self.indent -= 1;
        self.line("}");
    }

    // ------------------------------------------------------------------
    // Load
    // ------------------------------------------------------------------

    fn emit_load(&mut self, id: usize, input_idx: usize) {
        // Allocate GPU tensor from host input and transfer data
        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc(inputs[{input_idx}]->ndim, inputs[{input_idx}]->shape, inputs[{input_idx}]->dtype);"
        ));
        self.line(&format!(
            "chelis_host_to_device(d_t{id}, inputs[{input_idx}]);"
        ));
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
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        let size = Self::total_size(ty);
        let (grid, block) = launch::grid_1d(size);

        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("{");
        self.indent += 1;
        self.line(&format!("int t{id}_size = d_t{id}->storage_size;"));
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
        self.emit_kernel_launch(
            &format!("mod_{kernel_name}"),
            kernel_name,
            grid,
            block,
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
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        let size = Self::total_size(ty);
        let (grid, block) = launch::grid_1d(size);

        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("{");
        self.indent += 1;
        self.line(&format!("int t{id}_size = d_t{id}->storage_size;"));
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
        self.emit_kernel_launch(
            &format!("mod_{kernel_name}"),
            kernel_name,
            grid,
            block,
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
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        let size = Self::total_size(ty);
        let (grid, block) = launch::grid_1d(size);

        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("{");
        self.indent += 1;
        self.line(&format!("int t{id}_size = d_t{id}->storage_size;"));

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
        self.emit_kernel_launch(
            &format!("mod_{kernel_name}"),
            kernel_name,
            grid,
            block,
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
        kernel_name: &str,
        inputs: &[NodeId],
        ty: &TensorType,
        _dag: &Dag,
    ) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        let out_size = Self::total_size(ty);
        let (grid, block) = launch::grid_1d(out_size);

        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("{");
        self.indent += 1;
        self.line(&format!("int t{id}_out_size = d_t{id}->storage_size;"));
        self.emit_stride_vars(id, "a", a);
        self.line(&format!("int t{id}_a_ndim = d_t{a}->ndim;"));
        self.line(&format!("int t{id}_a_size = d_t{a}->storage_size;"));
        self.emit_shape_vars(id, "out", id);
        self.line(&format!("int t{id}_out_ndim = d_t{id}->ndim;"));
        self.line(&format!(
            "void *args[] = {{ &d_t{a}->data, {a_stride_refs}, &t{id}_a_ndim, &t{id}_a_size, \
             &d_t{id}->data, {out_shape_refs}, &t{id}_out_ndim, &t{id}_out_size }};",
            a_stride_refs = self.stride_arg_refs(id, "a"),
            out_shape_refs = self.shape_arg_refs(id, "out"),
        ));
        self.emit_kernel_launch(
            &format!("mod_{kernel_name}"),
            kernel_name,
            grid,
            block,
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
        kernel_name: &str,
        ext_inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        let out_size = Self::total_size(ty);
        let (grid, block) = launch::grid_1d(out_size);

        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("{");
        self.indent += 1;
        self.line(&format!("int t{id}_out_size = d_t{id}->storage_size;"));

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

        self.line(&format!("void *args[] = {{ {} }};", arg_parts.join(", ")));
        self.emit_kernel_launch(
            &format!("mod_{kernel_name}"),
            kernel_name,
            grid,
            block,
            "args",
        );
        self.indent -= 1;
        self.line("}");
    }

    // ------------------------------------------------------------------
    // Movement ops (host-side metadata, no kernel)
    // ------------------------------------------------------------------

    fn emit_reshape(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc_view({ndim}, {shape}, {dtype}, d_t{a}->data, d_t{a}->storage_size);"
        ));
        self.views.insert(id);
    }

    fn emit_permute(&mut self, id: usize, axes: &[usize], inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc_view({ndim}, {shape}, {dtype}, d_t{a}->data, d_t{a}->storage_size);"
        ));
        for (new_d, &old_d) in axes.iter().enumerate() {
            self.line(&format!(
                "d_t{id}->strides[{new_d}] = d_t{a}->strides[{old_d}];"
            ));
        }
        self.views.insert(id);
    }

    fn emit_expand(
        &mut self,
        id: usize,
        axis: usize,
        _size: usize,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc_view({ndim}, {shape}, {dtype}, d_t{a}->data, d_t{a}->storage_size);"
        ));
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
        self.views.insert(id);
    }

    fn emit_stride(
        &mut self,
        id: usize,
        stride_factors: &[usize],
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = chelis_gpu_alloc_view({ndim}, {shape}, {dtype}, d_t{a}->data, d_t{a}->storage_size);"
        ));
        let max_dims = ty.dims.len().max(1);
        for (d, &s) in stride_factors.iter().enumerate() {
            if d >= max_dims {
                break;
            }
            self.line(&format!(
                "d_t{id}->strides[{d}] = d_t{a}->strides[{d}] * {s};"
            ));
        }
        self.views.insert(id);
    }

    fn emit_store(&mut self, id: usize, name: &str, inputs: &[NodeId]) {
        let a = inputs[0].0;
        // Store just aliases the input — transfer to host happens in the output section
        self.line(&format!(
            "chelis_gpu_tensor *d_t{id} = d_t{a}; /* store: {name} */"
        ));
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

    fn emit_kernel_launch(
        &mut self,
        module_var: &str,
        kernel_name: &str,
        grid: usize,
        block: usize,
        args_var: &str,
    ) {
        self.line(&format!("chelis_prepare_kernel_launch({module_var});"));
        self.line(&format!(
            "chelis_launch_kernel({module_var}, \"{kernel_name}\", dim3({grid}), dim3({block}), {args_var});"
        ));
        self.line(&format!(
            "chelis_finalize_kernel_launch({module_var}, \"{kernel_name}\");"
        ));
    }

    // ------------------------------------------------------------------
    // Utility methods (mirrored from C backend)
    // ------------------------------------------------------------------

    fn line(&mut self, s: &str) {
        let prefix = "    ".repeat(self.indent);
        self.lines.push(format!("{prefix}{s}"));
    }

    fn shape_literal(ty: &TensorType) -> String {
        let dims: Vec<String> = ty
            .dims
            .iter()
            .map(|d| match d {
                DimInfo::Lit(n) => n.to_string(),
                DimInfo::Named(_, Some(n)) => n.to_string(),
                DimInfo::Named(name, None) => {
                    panic!("unsized named dimension '{name}' in HIP codegen")
                }
            })
            .collect();
        if dims.is_empty() {
            "(int[]){1}".to_string()
        } else {
            format!("(int[]){{ {} }}", dims.join(", "))
        }
    }

    fn ndim(ty: &TensorType) -> usize {
        if ty.dims.is_empty() { 1 } else { ty.dims.len() }
    }

    fn dim_size(dim: &DimInfo) -> usize {
        match dim {
            DimInfo::Lit(n) => *n,
            DimInfo::Named(_, Some(n)) => *n,
            DimInfo::Named(name, None) => {
                panic!("unsized named dimension '{name}' in HIP codegen")
            }
        }
    }

    fn total_size(ty: &TensorType) -> usize {
        if ty.dims.is_empty() {
            1
        } else {
            ty.dims.iter().map(Self::dim_size).product()
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
                && seen.insert(name.clone())
            {
                labels.push(name.clone());
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
                    label: name.clone(),
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
