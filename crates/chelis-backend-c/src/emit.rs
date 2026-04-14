//! RISC DAG to C source code emission.

use chelis_ir::dag::{
    Dag, DagNode, DimExpr, DimInfo, FusedInput, FusedStep, FusedStepOp, NodeId, RiscOp, TensorType,
    symbolic_bindings,
};
use chelis_types::types::Prim;

/// Emits C source code from a RISC DAG.
pub struct CEmitter {
    lines: Vec<String>,
    indent: usize,
    use_blas: bool,
    /// FusedElem nodes inlined into a trailing reduction (no standalone emission).
    reduction_inlined: std::collections::HashSet<usize>,
}

#[derive(Debug, Clone)]
struct OutputSpec {
    id: NodeId,
    label: String,
    is_store: bool,
}

impl CEmitter {
    /// Emit C source for an entire DAG as a function.
    pub fn emit_dag(dag: &Dag, func_name: &str) -> String {
        Self::emit_dag_with_options(dag, func_name, crate::CodegenOptions::default())
    }

    /// Emit C source for an entire DAG with explicit backend options.
    pub fn emit_dag_with_options(
        dag: &Dag,
        func_name: &str,
        options: crate::CodegenOptions,
    ) -> String {
        Self::validate_supported_precisions(dag);
        Self::validate_load_abi(dag);
        // Some Surf signatures surface anonymous (Named("", None)) axes into
        // the lowered DAG (e.g. a rank-1 tensor parameter whose dim has no
        // declared name). These would emit `int  = inputs[0]->shape[0];` and
        // `(int[]){ }` shape literals, neither of which compiles. Rewrite
        // empty dim names to a stable synthesized identifier before the
        // emitter walks the DAG.
        let dag_owned = Self::rename_anonymous_dims(dag);
        let dag = &dag_owned;

        let reduction_inlined = chelis_ir::fuse::reduction_inlined_fused_elems(dag);
        let mut e = CEmitter {
            lines: Vec::new(),
            indent: 0,
            use_blas: options.use_blas,
            reduction_inlined: reduction_inlined.iter().map(|id| id.0).collect(),
        };

        e.line("#include \"chelis_runtime.h\"");
        if e.use_blas {
            e.line("#include <cblas.h>");
        }
        e.line(
            "static inline float chelis_uniform_sample_f32(uint64_t seed, uint64_t index, float low, float high) {",
        );
        e.line("    uint64_t x = seed ^ (index * 0x9E3779B97F4A7C15ULL);");
        e.line("    x ^= x >> 30;");
        e.line("    x *= 0xBF58476D1CE4E5B9ULL;");
        e.line("    x ^= x >> 27;");
        e.line("    x *= 0x94D049BB133111EBULL;");
        e.line("    x ^= x >> 31;");
        e.line("    double unit = (double)(x >> 11) / (double)(1ULL << 53);");
        e.line("    return low + (high - low) * (float)unit;");
        e.line("}");
        e.line("");

        e.line(&format!(
            "void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out) {{"
        ));
        e.indent = 1;

        let input_labels = Self::input_labels(dag);
        let input_slots = Self::input_slots(&input_labels);
        let expected_inputs = input_labels.len();
        let output_specs = Self::output_specs(dag);
        let expected_outputs = output_specs.len();

        e.line(&format!("if (n_in != {expected_inputs}) {{"));
        e.indent += 1;
        e.line(&format!(
            "fprintf(stderr, \"{func_name}: expected %d inputs, got %d\\n\", {expected_inputs}, n_in);"
        ));
        e.line("abort();");
        e.indent -= 1;
        e.line("}");
        if expected_inputs > 0 {
            e.line("if (inputs == NULL) {");
            e.indent += 1;
            e.line(&format!(
                "fprintf(stderr, \"{func_name}: inputs array is NULL but %d inputs are required\\n\", {expected_inputs});"
            ));
            e.line("abort();");
            e.indent -= 1;
            e.line("}");
        }

        e.line(&format!("if (n_out != {expected_outputs}) {{"));
        e.indent += 1;
        e.line(&format!(
            "fprintf(stderr, \"{func_name}: expected %d outputs, got %d\\n\", {expected_outputs}, n_out);"
        ));
        e.line("abort();");
        e.indent -= 1;
        e.line("}");
        if expected_outputs > 0 {
            e.line("if (outputs == NULL) {");
            e.indent += 1;
            e.line(&format!(
                "fprintf(stderr, \"{func_name}: outputs array is NULL but %d outputs are required\\n\", {expected_outputs});"
            ));
            e.line("abort();");
            e.indent -= 1;
            e.line("}");
        }

        e.emit_input_shape_preamble(dag, &input_slots, func_name);

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

        // Copy outputs to contiguous buffers owned by the caller
        for (slot, output) in output_specs.iter().enumerate() {
            let is_load = dag
                .get(output.id)
                .map(|n| matches!(n.op, RiscOp::Load { .. }))
                .unwrap_or(false);
            if output.is_store {
                e.line(&format!("outputs[{slot}] = t{};", output.id.0));
            } else {
                e.line(&format!(
                    "outputs[{slot}] = chelis_contiguous(t{});",
                    output.id.0
                ));
                // Free the original if contiguous made a copy (but not Loads — they're borrowed)
                if !is_load {
                    e.line(&format!(
                        "if (outputs[{slot}] != t{id}) chelis_free(t{id});",
                        id = output.id.0
                    ));
                }
            }
        }

        let skip_ids: Vec<NodeId> = e.reduction_inlined.iter().map(|&id| NodeId(id)).collect();
        let cleanup = crate::memory::emit_cleanup_with_skip(
            dag,
            &output_specs
                .iter()
                .map(|output| output.id)
                .collect::<Vec<_>>(),
            &skip_ids,
        );
        for line in cleanup {
            e.lines.push(line);
        }

        e.indent = 0;
        e.line("}");
        e.lines.join("\n")
    }

    fn rename_anonymous_dims(dag: &Dag) -> Dag {
        use chelis_ir::dag::DimInfo;
        fn is_anon(name: &str) -> bool {
            name.is_empty() || name == "*"
        }
        fn rewrite_dim(dim: &DimInfo) -> DimInfo {
            match dim {
                DimInfo::Named(name, size) if is_anon(name) => {
                    DimInfo::Named("_anon_dim".to_string(), *size)
                }
                other => other.clone(),
            }
        }
        let mut out = dag.clone();
        // DAG exposes no `nodes_mut`; rewrite by round-tripping replace_node.
        let ids: Vec<_> = out.nodes().iter().map(|n| n.id).collect();
        for id in ids {
            if let Some(node) = out.get(id) {
                let needs = node
                    .output_type
                    .dims
                    .iter()
                    .any(|d| matches!(d, DimInfo::Named(name, _) if is_anon(name)));
                if !needs {
                    continue;
                }
                let mut new_ty = node.output_type.clone();
                new_ty.dims = new_ty.dims.iter().map(rewrite_dim).collect();
                let op = node.op.clone();
                let inputs = node.inputs.clone();
                out.replace_node(id, op, inputs, new_ty);
            }
        }
        out
    }

    fn emit_node(&mut self, node: &DagNode, dag: &Dag) {
        let id = node.id.0;
        match &node.op {
            RiscOp::Const { value } => self.emit_const(id, *value, &node.output_type),
            RiscOp::Load { .. } => unreachable!("handled in emit_dag"),
            RiscOp::Add => self.emit_binary(id, "+", &node.inputs, &node.output_type),
            RiscOp::Mul => self.emit_binary(id, "*", &node.inputs, &node.output_type),
            RiscOp::MaxElem => {
                self.emit_binary_func(id, "fmaxf", &node.inputs, &node.output_type);
            }
            RiscOp::CmpLt => self.emit_cmplt(id, &node.inputs, &node.output_type),
            RiscOp::Neg => self.emit_unary(id, "-", &node.inputs, &node.output_type),
            RiscOp::Exp => self.emit_unary_func(id, "expf", &node.inputs, &node.output_type),
            RiscOp::Log => self.emit_unary_func(id, "logf", &node.inputs, &node.output_type),
            RiscOp::Sin => self.emit_unary_func(id, "sinf", &node.inputs, &node.output_type),
            RiscOp::Sqrt => self.emit_unary_func(id, "sqrtf", &node.inputs, &node.output_type),
            RiscOp::UniformLike { low, high, seed } => {
                self.emit_uniform_like(id, *low, *high, *seed, &node.output_type)
            }
            RiscOp::Dropout { .. } => {
                unreachable!("dropout should be rejected before C code generation")
            }
            RiscOp::Sum { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    self.emit_fused_reduce(
                        id,
                        *axis,
                        &fused_node.inputs.clone(),
                        &fused_node.op.clone(),
                        &fused_node.output_type.clone(),
                        &node.output_type,
                        "sum",
                    );
                } else {
                    self.emit_reduce_sum(id, *axis, &node.inputs, &node.output_type, dag);
                }
            }
            RiscOp::MaxReduce { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    self.emit_fused_reduce(
                        id,
                        *axis,
                        &fused_node.inputs.clone(),
                        &fused_node.op.clone(),
                        &fused_node.output_type.clone(),
                        &node.output_type,
                        "max",
                    );
                } else {
                    self.emit_reduce_max(id, *axis, &node.inputs, &node.output_type, dag);
                }
            }
            RiscOp::MinReduce { axis } => {
                self.emit_reduce_simple(
                    id,
                    *axis,
                    &node.inputs,
                    &node.output_type,
                    dag,
                    "INFINITY",
                    "acc = fminf(acc, t{a}->data[src_idx]);",
                );
            }
            RiscOp::ProdReduce { axis } => {
                self.emit_reduce_simple(
                    id,
                    *axis,
                    &node.inputs,
                    &node.output_type,
                    dag,
                    "1.0f",
                    "acc *= t{a}->data[src_idx];",
                );
            }
            RiscOp::Argmax { axis } => {
                self.emit_reduce_argcmp(id, *axis, &node.inputs, &node.output_type, dag, true);
            }
            RiscOp::Argmin { axis } => {
                self.emit_reduce_argcmp(id, *axis, &node.inputs, &node.output_type, dag, false);
            }
            RiscOp::Reshape { .. } => {
                self.emit_reshape(id, &node.inputs, &node.output_type, dag);
            }
            RiscOp::Permute { axes } => {
                self.emit_permute(id, axes, &node.inputs, &node.output_type, dag);
            }
            RiscOp::Expand { axis, size } => {
                self.emit_expand(id, *axis, size, &node.inputs, &node.output_type, dag);
            }
            RiscOp::Pad { padding, fill } => {
                self.emit_pad(id, padding, *fill, &node.inputs, &node.output_type, dag);
            }
            RiscOp::Shrink { bounds } => {
                self.emit_shrink(id, bounds, &node.inputs, &node.output_type, dag);
            }
            RiscOp::Stride { strides } => {
                self.emit_stride(id, strides, &node.inputs, &node.output_type);
            }
            RiscOp::Realize => self.emit_realize(id, &node.inputs, &node.output_type),
            RiscOp::Cast { .. } => self.emit_cast(id, &node.inputs, &node.output_type),
            RiscOp::Store { name } => self.emit_store(id, name, &node.inputs),
            RiscOp::FusedElem { ops } => {
                self.emit_fused_elem(id, ops, &node.inputs, &node.output_type);
            }
        }
    }

    fn line(&mut self, s: &str) {
        let prefix = "    ".repeat(self.indent);
        self.lines.push(format!("{prefix}{s}"));
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
            dag.nodes()
                .last()
                .map(|node| vec![node.id])
                .unwrap_or_default()
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

    pub(crate) fn output_labels(dag: &Dag) -> Vec<String> {
        Self::output_specs(dag)
            .into_iter()
            .map(|output| output.label)
            .collect()
    }

    pub(crate) fn input_labels(dag: &Dag) -> Vec<String> {
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

    fn input_slots(labels: &[String]) -> std::collections::HashMap<String, usize> {
        labels
            .iter()
            .cloned()
            .enumerate()
            .map(|(slot, label)| (label, slot))
            .collect()
    }

    fn validate_supported_precisions(dag: &Dag) {
        for node in dag.nodes() {
            match node.output_type.precision {
                Prim::F32 | Prim::Bool => {}
                other => panic!(
                    "Phase 0f C backend only supports f32/bool tensors, found {} at node {}",
                    other.name(),
                    node.id.0
                ),
            }

            if let RiscOp::Cast { new_precision } = node.op
                && new_precision != Prim::F32
            {
                panic!(
                    "Phase 0f C backend only supports casts to f32, found cast to {} at node {}",
                    new_precision.name(),
                    node.id.0
                );
            }
        }
    }

    fn validate_load_abi(dag: &Dag) {
        let mut seen = std::collections::HashMap::<String, TensorType>::new();
        for node in dag.nodes() {
            if let RiscOp::Load { name } = &node.op {
                if let Some(prev_ty) = seen.get(name) {
                    assert_eq!(
                        prev_ty, &node.output_type,
                        "Load name '{}' used with inconsistent tensor types in C codegen",
                        name
                    );
                } else {
                    seen.insert(name.clone(), node.output_type.clone());
                }
            }
        }
    }

    fn input_types(dag: &Dag) -> std::collections::HashMap<String, TensorType> {
        let mut seen = std::collections::HashMap::<String, TensorType>::new();
        for node in dag.nodes() {
            if let RiscOp::Load { name } = &node.op {
                seen.entry(name.clone())
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
        let input_types = Self::input_types(dag);
        for (label, ty) in &input_types {
            let slot = input_slots[label];
            self.line(&format!("if (inputs[{slot}] == NULL) {{"));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"{func_name}: input `{label}` at slot {slot} is NULL\\n\");"
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
                "fprintf(stderr, \"{func_name}: input `{label}` expected rank {}, got %d\\n\", inputs[{slot}]->ndim);",
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
                        "fprintf(stderr, \"{func_name}: input `{label}` axis {axis} expected {expected}, got %d\\n\", inputs[{slot}]->shape[{axis}]);"
                    ));
                    self.line("abort();");
                    self.indent -= 1;
                    self.line("}");
                }
            }
        }

        for binding in symbolic_bindings(dag) {
            let canonical_slot = input_slots[&binding.canonical.input_label];
            self.line(&format!(
                "int {} = inputs[{canonical_slot}]->shape[{}];",
                binding.name, binding.canonical.axis
            ));
            for occurrence in binding.others {
                let slot = input_slots[&occurrence.input_label];
                self.line(&format!(
                    "if (inputs[{slot}]->shape[{}] != {}) {{",
                    occurrence.axis, binding.name
                ));
                self.indent += 1;
                self.line(&format!(
                    "fprintf(stderr, \"{func_name}: symbolic dim `{}` mismatch: {}[{}]=%d but {}=%d\\n\", inputs[{slot}]->shape[{}], {});",
                    binding.name,
                    occurrence.input_label,
                    occurrence.axis,
                    binding.name,
                    occurrence.axis,
                    binding.name
                ));
                self.line("abort();");
                self.indent -= 1;
                self.line("}");
            }
        }
    }

    fn shape_literal(ty: &TensorType) -> String {
        let dims: Vec<String> = ty
            .dims
            .iter()
            .map(|dim| Self::emit_dim_expr(&DimExpr::from(dim)))
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

    fn dtype_macro(ty: &TensorType) -> &'static str {
        match ty.precision {
            Prim::F32 => "CHELIS_F32",
            Prim::Bool => "CHELIS_BOOL",
            other => panic!(
                "Phase 0f C backend only supports f32/bool tensors, got {}",
                other.name()
            ),
        }
    }

    // ---- Const ----
    fn emit_const(&mut self, id: usize, value: f64, ty: &TensorType) {
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line(&format!("chelis_fill_f32(t{id}, {:.8}f);", value as f32));
    }

    // ---- Load ----
    fn emit_load(&mut self, id: usize, input_idx: usize) {
        self.line(&format!("chelis_tensor *t{id} = inputs[{input_idx}];"));
    }

    // ---- Binary elementwise ----
    fn emit_binary(&mut self, id: usize, op: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line("int indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}->shape, t{id}->ndim, indices);"
        ));
        self.line(&format!(
            "int idx_a = chelis_indices_to_flat(indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!(
            "int idx_b = chelis_indices_to_flat(indices, t{b}->strides, t{b}->ndim);"
        ));
        self.line(&format!(
            "t{id}->data[i] = t{a}->data[idx_a] {op} t{b}->data[idx_b];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Binary func (fmaxf etc.) ----
    fn emit_binary_func(&mut self, id: usize, func: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line("int indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}->shape, t{id}->ndim, indices);"
        ));
        self.line(&format!(
            "int idx_a = chelis_indices_to_flat(indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!(
            "int idx_b = chelis_indices_to_flat(indices, t{b}->strides, t{b}->ndim);"
        ));
        self.line(&format!(
            "t{id}->data[i] = {func}(t{a}->data[idx_a], t{b}->data[idx_b]);"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- CmpLt ----
    fn emit_cmplt(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line("int indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}->shape, t{id}->ndim, indices);"
        ));
        self.line(&format!(
            "int idx_a = chelis_indices_to_flat(indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!(
            "int idx_b = chelis_indices_to_flat(indices, t{b}->strides, t{b}->ndim);"
        ));
        self.line(&format!(
            "t{id}->data[i] = (t{a}->data[idx_a] < t{b}->data[idx_b]) ? 1.0f : 0.0f;"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Unary elementwise ----
    fn emit_unary(&mut self, id: usize, op: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line("int indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}->shape, t{id}->ndim, indices);"
        ));
        self.line(&format!(
            "int idx = chelis_indices_to_flat(indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!("t{id}->data[i] = {op}t{a}->data[idx];"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Unary func (expf, logf, sinf, sqrtf) ----
    fn emit_unary_func(&mut self, id: usize, func: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line("int indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}->shape, t{id}->ndim, indices);"
        ));
        self.line(&format!(
            "int idx = chelis_indices_to_flat(indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!("t{id}->data[i] = {func}(t{a}->data[idx]);"));
        self.indent -= 1;
        self.line("}");
    }

    fn emit_uniform_like(&mut self, id: usize, low: f64, high: f64, seed: u64, ty: &TensorType) {
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "t{id}->data[i] = chelis_uniform_sample_f32({seed}ULL, (uint64_t)i, {:.8}f, {:.8}f);",
            low as f32, high as f32
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Fused elementwise ----
    fn emit_fused_elem(
        &mut self,
        id: usize,
        ops: &[FusedStep],
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line("int indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}->shape, t{id}->ndim, indices);"
        ));

        // Compute strided index for each external input.
        for (ext_idx, ext_node) in inputs.iter().enumerate() {
            let ext_id = ext_node.0;
            self.line(&format!(
                "int idx_ext{ext_idx} = chelis_indices_to_flat(indices, t{ext_id}->strides, t{ext_id}->ndim);"
            ));
        }

        // Emit each fused step.
        let resolve = |fi: &FusedInput| -> String {
            match fi {
                FusedInput::External(i) => {
                    let ext_id = inputs[*i].0;
                    format!("t{ext_id}->data[idx_ext{i}]")
                }
                FusedInput::PreviousStep(j) => format!("v{j}"),
            }
        };

        for (s, step) in ops.iter().enumerate() {
            let expr = match &step.op {
                FusedStepOp::Add => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("{a} + {b}")
                }
                FusedStepOp::Mul => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("{a} * {b}")
                }
                FusedStepOp::MaxElem => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("fmaxf({a}, {b})")
                }
                FusedStepOp::CmpLt => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("({a} < {b}) ? 1.0f : 0.0f")
                }
                FusedStepOp::Neg => {
                    let a = resolve(&step.input_indices[0]);
                    format!("-{a}")
                }
                FusedStepOp::Exp => {
                    let a = resolve(&step.input_indices[0]);
                    format!("expf({a})")
                }
                FusedStepOp::Log => {
                    let a = resolve(&step.input_indices[0]);
                    format!("logf({a})")
                }
                FusedStepOp::Sin => {
                    let a = resolve(&step.input_indices[0]);
                    format!("sinf({a})")
                }
                FusedStepOp::Sqrt => {
                    let a = resolve(&step.input_indices[0]);
                    format!("sqrtf({a})")
                }
            };
            self.line(&format!("float v{s} = {expr};"));
        }

        // Store last step's result.
        let last = ops.len() - 1;
        self.line(&format!("t{id}->data[i] = v{last};"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- BLAS matmul ----
    fn emit_blas_matmul(&mut self, id: usize, info: &crate::blas::MatmulInfo, ty: &TensorType) {
        let a = info.a.0;
        let b = info.b.0;
        let m = info.m;
        let n = info.n;
        let k = info.k;
        let dtype = Self::dtype_macro(ty);
        self.line(&format!("chelis_tensor *t{id}_a = t{a};"));
        self.line(&format!("if (!chelis_is_contiguous(t{id}_a)) {{"));
        self.indent += 1;
        self.line(&format!("t{id}_a = chelis_contiguous(t{id}_a);"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_tensor *t{id}_b = t{b};"));
        self.line(&format!("if (!chelis_is_contiguous(t{id}_b)) {{"));
        self.indent += 1;
        self.line(&format!("t{id}_b = chelis_contiguous(t{id}_b);"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc(2, (int[]){{ {m}, {n} }}, {dtype});"
        ));
        self.line(&format!(
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, {m}, {n}, {k}, 1.0f, t{id}_a->data, {k}, t{id}_b->data, {n}, 0.0f, t{id}->data, {n});"
        ));
        self.line(&format!("if (t{id}_a != t{a}) chelis_free(t{id}_a);"));
        self.line(&format!("if (t{id}_b != t{b}) chelis_free(t{id}_b);"));
    }

    // ---- Reduce sum ----
    fn emit_reduce_sum(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: &Dag,
    ) {
        // Check for matmul pattern before generic reduction
        if self.use_blas
            && let Some(matmul) = crate::blas::detect_matmul_pattern(dag, NodeId(id))
        {
            self.emit_blas_matmul(id, &matmul, ty);
            return;
        }
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line(&format!("chelis_fill_f32(t{id}, 0.0f);"));
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int outer = 0; outer < t{id}->size; outer++) {{"
        ));
        self.indent += 1;
        self.line("float acc = 0.0f;");
        self.line("int out_indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}->shape, t{id}->ndim, out_indices);"
        ));
        self.line(&format!("for (int k = 0; k < {axis_size}; k++) {{"));
        self.indent += 1;
        self.line("int full_indices[CHELIS_MAX_DIM];");
        // Build full indices: insert k at the reduction axis
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}->ndim; d++) {{"));
        self.indent += 1;
        self.line(&format!("if (d == {axis}) {{"));
        self.indent += 1;
        self.line("full_indices[d] = k;");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("full_indices[d] = out_indices[out_d];");
        self.line("out_d++;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "int src_idx = chelis_indices_to_flat(full_indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!("acc += t{a}->data[src_idx];"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("t{id}->data[outer] = acc;"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Reduce max ----
    fn emit_reduce_max(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: &Dag,
    ) {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int outer = 0; outer < t{id}->size; outer++) {{"
        ));
        self.indent += 1;
        self.line("float acc = -INFINITY;");
        self.line("int out_indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}->shape, t{id}->ndim, out_indices);"
        ));
        self.line(&format!("for (int k = 0; k < {axis_size}; k++) {{"));
        self.indent += 1;
        self.line("int full_indices[CHELIS_MAX_DIM];");
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}->ndim; d++) {{"));
        self.indent += 1;
        self.line(&format!("if (d == {axis}) {{"));
        self.indent += 1;
        self.line("full_indices[d] = k;");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("full_indices[d] = out_indices[out_d];");
        self.line("out_d++;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "int src_idx = chelis_indices_to_flat(full_indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!("acc = fmaxf(acc, t{a}->data[src_idx]);"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("t{id}->data[outer] = acc;"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Generic scalar reduction (min / prod) ----
    //
    // `init` is the C literal for the accumulator's starting value, and
    // `update_tmpl` is the body of the inner loop with literal `{a}` tokens
    // for the input node id. Used by MinReduce and ProdReduce; the structure
    // mirrors `emit_reduce_max` exactly.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduce_simple(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: &Dag,
        init: &str,
        update_tmpl: &str,
    ) {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int outer = 0; outer < t{id}->size; outer++) {{"
        ));
        self.indent += 1;
        self.line(&format!("float acc = {init};"));
        self.line("int out_indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}->shape, t{id}->ndim, out_indices);"
        ));
        self.line(&format!("for (int k = 0; k < {axis_size}; k++) {{"));
        self.indent += 1;
        self.line("int full_indices[CHELIS_MAX_DIM];");
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}->ndim; d++) {{"));
        self.indent += 1;
        self.line(&format!("if (d == {axis}) {{"));
        self.indent += 1;
        self.line("full_indices[d] = k;");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("full_indices[d] = out_indices[out_d];");
        self.line("out_d++;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "int src_idx = chelis_indices_to_flat(full_indices, t{a}->strides, t{a}->ndim);"
        ));
        let update = update_tmpl.replace("{a}", &a.to_string());
        self.line(&update);
        self.indent -= 1;
        self.line("}");
        self.line(&format!("t{id}->data[outer] = acc;"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Argmax / Argmin ----
    //
    // Emits an index-tracking reduction. Output is an F32 tensor holding
    // integer-valued floats (e.g. 0.0, 1.0, 2.0); see the RiscOp::Argmax doc
    // comment in dag.rs for the rationale — the C runtime does not yet carry
    // Int64 tensors natively, so the IR carries F32 and downstream casts are
    // the caller's responsibility.
    fn emit_reduce_argcmp(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: &Dag,
        is_argmax: bool,
    ) {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        let init = if is_argmax { "-INFINITY" } else { "INFINITY" };
        let cmp = if is_argmax { ">" } else { "<" };
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int outer = 0; outer < t{id}->size; outer++) {{"
        ));
        self.indent += 1;
        self.line(&format!("float best_val = {init};"));
        self.line("int best_idx = -1;");
        self.line("int out_indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}->shape, t{id}->ndim, out_indices);"
        ));
        self.line(&format!("for (int k = 0; k < {axis_size}; k++) {{"));
        self.indent += 1;
        self.line("int full_indices[CHELIS_MAX_DIM];");
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}->ndim; d++) {{"));
        self.indent += 1;
        self.line(&format!("if (d == {axis}) {{"));
        self.indent += 1;
        self.line("full_indices[d] = k;");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("full_indices[d] = out_indices[out_d];");
        self.line("out_d++;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "int src_idx = chelis_indices_to_flat(full_indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!("float v = t{a}->data[src_idx];"));
        self.line(&format!("if (best_idx < 0 || v {cmp} best_val) {{"));
        self.indent += 1;
        self.line("best_val = v;");
        self.line("best_idx = k;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!("t{id}->data[outer] = (float)best_idx;"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Fused elementwise→reduction ----
    #[allow(clippy::too_many_arguments)]
    fn emit_fused_reduce(
        &mut self,
        id: usize,
        axis: usize,
        ext_inputs: &[NodeId],
        fused_op: &RiscOp,
        fused_input_type: &TensorType,
        out_ty: &TensorType,
        reduce_kind: &str,
    ) {
        let ops = match fused_op {
            RiscOp::FusedElem { ops } => ops,
            _ => panic!("expected FusedElem op"),
        };
        let axis_size = Self::emit_dim_info(&fused_input_type.dims[axis]);
        // ndim of the fused input (pre-reduction shape)
        let fused_ndim = fused_input_type.dims.len();

        let ndim = Self::ndim(out_ty);
        let shape = Self::shape_literal(out_ty);
        let dtype = Self::dtype_macro(out_ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        if reduce_kind == "sum" {
            self.line(&format!("chelis_fill_f32(t{id}, 0.0f);"));
        }
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int outer = 0; outer < t{id}->size; outer++) {{"
        ));
        self.indent += 1;
        let init = if reduce_kind == "sum" {
            "0.0f"
        } else {
            "-INFINITY"
        };
        self.line(&format!("float acc = {init};"));
        self.line("int out_indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}->shape, t{id}->ndim, out_indices);"
        ));
        self.line(&format!("for (int k = 0; k < {axis_size}; k++) {{"));
        self.indent += 1;
        self.line("int full_indices[CHELIS_MAX_DIM];");
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < {fused_ndim}; d++) {{"));
        self.indent += 1;
        self.line(&format!("if (d == {axis}) {{"));
        self.indent += 1;
        self.line("full_indices[d] = k;");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("full_indices[d] = out_indices[out_d];");
        self.line("out_d++;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");

        // Compute strided index for each external input using full_indices
        for (ext_idx, ext_node) in ext_inputs.iter().enumerate() {
            let ext_id = ext_node.0;
            self.line(&format!(
                "int idx_ext{ext_idx} = chelis_indices_to_flat(full_indices, t{ext_id}->strides, t{ext_id}->ndim);"
            ));
        }

        // Emit each fused step
        let resolve = |fi: &FusedInput| -> String {
            match fi {
                FusedInput::External(i) => {
                    let ext_id = ext_inputs[*i].0;
                    format!("t{ext_id}->data[idx_ext{i}]")
                }
                FusedInput::PreviousStep(j) => format!("v{j}"),
            }
        };

        for (s, step) in ops.iter().enumerate() {
            let expr = match &step.op {
                FusedStepOp::Add => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("{a} + {b}")
                }
                FusedStepOp::Mul => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("{a} * {b}")
                }
                FusedStepOp::MaxElem => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("fmaxf({a}, {b})")
                }
                FusedStepOp::CmpLt => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("({a} < {b}) ? 1.0f : 0.0f")
                }
                FusedStepOp::Neg => {
                    let a = resolve(&step.input_indices[0]);
                    format!("-{a}")
                }
                FusedStepOp::Exp => {
                    let a = resolve(&step.input_indices[0]);
                    format!("expf({a})")
                }
                FusedStepOp::Log => {
                    let a = resolve(&step.input_indices[0]);
                    format!("logf({a})")
                }
                FusedStepOp::Sin => {
                    let a = resolve(&step.input_indices[0]);
                    format!("sinf({a})")
                }
                FusedStepOp::Sqrt => {
                    let a = resolve(&step.input_indices[0]);
                    format!("sqrtf({a})")
                }
            };
            self.line(&format!("float v{s} = {expr};"));
        }

        // Accumulate the last step's result
        let last = ops.len() - 1;
        if reduce_kind == "sum" {
            self.line(&format!("acc += v{last};"));
        } else {
            self.line(&format!("acc = fmaxf(acc, v{last});"));
        }
        self.indent -= 1;
        self.line("}");
        self.line(&format!("t{id}->data[outer] = acc;"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Reshape ----
    fn emit_reshape(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType, _dag: &Dag) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!("chelis_tensor *t{id};"));
        self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
        self.indent += 1;
        self.line(&format!(
            "t{id} = chelis_alloc_view({ndim}, {shape}, {dtype}, t{a}->data);"
        ));
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line(&format!(
            "chelis_tensor *t{id}_src = chelis_contiguous(t{a});"
        ));
        self.line(&format!(
            "t{id} = chelis_alloc_view({ndim}, {shape}, {dtype}, t{id}_src->data);"
        ));
        self.line(&format!("t{id}->owns_data = 1;"));
        self.line(&format!("t{id}_src->owns_data = 0;"));
        self.line(&format!("chelis_free(t{id}_src);"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Permute ----
    fn emit_permute(
        &mut self,
        id: usize,
        axes: &[usize],
        inputs: &[NodeId],
        ty: &TensorType,
        _dag: &Dag,
    ) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc_view({ndim}, {shape}, {dtype}, t{a}->data);"
        ));
        // Set permuted strides
        for (new_d, &old_d) in axes.iter().enumerate() {
            self.line(&format!(
                "t{id}->strides[{new_d}] = t{a}->strides[{old_d}];"
            ));
        }
    }

    // ---- Expand ----
    fn emit_expand(
        &mut self,
        id: usize,
        axis: usize,
        _size: &DimExpr,
        inputs: &[NodeId],
        ty: &TensorType,
        _dag: &Dag,
    ) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc_view({ndim}, {shape}, {dtype}, t{a}->data);"
        ));
        self.line(&format!("if (t{id}->ndim == t{a}->ndim) {{"));
        self.indent += 1;
        self.line(&format!(
            "for (int d = 0; d < t{a}->ndim; d++) t{id}->strides[d] = t{a}->strides[d];"
        ));
        self.line(&format!("t{id}->strides[{axis}] = 0;"));
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        for d in 0..axis {
            self.line(&format!("t{id}->strides[{d}] = t{a}->strides[{d}];"));
        }
        self.line(&format!("t{id}->strides[{axis}] = 0;"));
        self.line(&format!(
            "for (int d = {axis}; d < t{a}->ndim; d++) t{id}->strides[d+1] = t{a}->strides[d];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Pad ----
    fn emit_pad(
        &mut self,
        id: usize,
        padding: &[(usize, usize)],
        fill: f64,
        inputs: &[NodeId],
        ty: &TensorType,
        _dag: &Dag,
    ) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line(&format!("chelis_fill_f32(t{id}, {:.8}f);", fill as f32));
        // Copy source data into the padded region
        self.line(&format!("for (int i = 0; i < t{a}->size; i++) {{"));
        self.indent += 1;
        self.line("int src_indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(i, t{a}->shape, t{a}->ndim, src_indices);"
        ));
        self.line("int dst_indices[CHELIS_MAX_DIM];");
        for (d, &(lo, _hi)) in padding.iter().enumerate() {
            self.line(&format!("dst_indices[{d}] = src_indices[{d}] + {lo};"));
        }
        self.line(&format!(
            "int dst_flat = chelis_indices_to_flat(dst_indices, t{id}->strides, t{id}->ndim);"
        ));
        self.line(&format!(
            "int src_flat = chelis_indices_to_flat(src_indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!("t{id}->data[dst_flat] = t{a}->data[src_flat];"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Shrink ----
    fn emit_shrink(
        &mut self,
        id: usize,
        bounds: &[(usize, usize)],
        inputs: &[NodeId],
        ty: &TensorType,
        _dag: &Dag,
    ) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line("int dst_indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}->shape, t{id}->ndim, dst_indices);"
        ));
        self.line("int src_indices[CHELIS_MAX_DIM];");
        for (d, &(lo, _hi)) in bounds.iter().enumerate() {
            self.line(&format!("src_indices[{d}] = dst_indices[{d}] + {lo};"));
        }
        self.line(&format!(
            "int src_flat = chelis_indices_to_flat(src_indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!("t{id}->data[i] = t{a}->data[src_flat];"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Stride ----
    fn emit_stride(&mut self, id: usize, strides: &[usize], inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc_view({ndim}, {shape}, {dtype}, t{a}->data);"
        ));
        let max_dims = ty.dims.len().max(1);
        for (d, &s) in strides.iter().enumerate() {
            if d >= max_dims {
                break;
            }
            self.line(&format!("t{id}->strides[{d}] = t{a}->strides[{d}] * {s};"));
        }
    }

    // ---- Realize ----
    fn emit_realize(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line("int indices[CHELIS_MAX_DIM];");
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}->shape, t{id}->ndim, indices);"
        ));
        self.line(&format!(
            "int idx = chelis_indices_to_flat(indices, t{a}->strides, t{a}->ndim);"
        ));
        self.line(&format!("t{id}->data[i] = t{a}->data[idx];"));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Cast ----
    fn emit_cast(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        // Phase 0f only supports casts to f32, validated before emission.
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.line(&format!(
            "memcpy(t{id}->data, t{a}->data, t{id}->size * sizeof(float));"
        ));
    }

    // ---- Store ----
    fn emit_store(&mut self, id: usize, name: &str, inputs: &[NodeId]) {
        let a = inputs[0].0;
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_contiguous(t{a}); /* store: {name} */"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
    use chelis_types::types::Prim;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn mat_f32(r: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
            precision: Prim::F32,
        }
    }

    #[test]
    fn const_emits_alloc_and_fill() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_alloc"));
        assert!(c.contains("chelis_fill_f32"));
        assert!(c.contains("3.0"));
    }

    #[test]
    fn add_emits_stride_aware_loop() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_flat_to_indices"));
        assert!(c.contains("chelis_indices_to_flat"));
        assert!(c.contains("+"));
    }

    #[test]
    fn neg_emits_unary_minus() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Neg, vec![a], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("-t0->data[idx]"));
    }

    #[test]
    fn exp_emits_expf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Exp, vec![a], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("expf("));
    }

    #[test]
    fn log_emits_logf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Log, vec![a], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("logf("));
    }

    #[test]
    fn sin_emits_sinf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Sin, vec![a], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("sinf("));
    }

    #[test]
    fn sqrt_emits_sqrtf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Sqrt, vec![a], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("sqrtf("));
    }

    #[test]
    fn cmplt_emits_ternary_float() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::CmpLt, vec![a, b], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("1.0f : 0.0f"));
    }

    #[test]
    fn omp_pragma_in_elementwise() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("#pragma omp parallel for"));
    }

    #[test]
    fn sum_emits_reduction_loop() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
        dag.add_node(RiscOp::Sum { axis: 0 }, vec![a], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("acc +="));
        assert!(c.contains("for (int k"));
    }

    #[test]
    fn max_reduce_emits_fmaxf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
        dag.add_node(RiscOp::MaxReduce { axis: 0 }, vec![a], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("fmaxf(acc"));
        assert!(c.contains("-INFINITY"));
    }

    #[test]
    fn mul_emits_star_op() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("*"));
    }

    #[test]
    fn max_elem_emits_fmaxf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::MaxElem, vec![a, b], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("fmaxf("));
    }

    #[test]
    fn reshape_emits_contiguous_check() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(6));
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            },
            vec![a],
            mat_f32(2, 3),
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_is_contiguous"));
        assert!(c.contains("chelis_contiguous"));
    }

    #[test]
    fn permute_emits_stride_swap() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
        dag.add_node(RiscOp::Permute { axes: vec![1, 0] }, vec![a], mat_f32(3, 2));
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("strides[0] = t0->strides[1]"));
        assert!(c.contains("strides[1] = t0->strides[0]"));
    }

    #[test]
    fn expand_sets_stride_zero() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(1));
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            vec_f32(4),
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("strides[0] = 0"));
    }

    #[test]
    fn store_is_alias() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_node(
            RiscOp::Store {
                name: "out".to_string(),
            },
            vec![a],
            scalar_f32(),
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_contiguous(t0); /* store: out */"));
    }

    #[test]
    fn cast_emits_memcpy() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_node(
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![a],
            scalar_f32(),
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("memcpy"));
    }

    #[test]
    fn realize_emits_materialization_loop() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(6));
        let s = dag.add_node(RiscOp::Stride { strides: vec![2] }, vec![x], vec_f32(3));
        dag.add_node(RiscOp::Realize, vec![s], vec_f32(3));

        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_tensor *t2 = chelis_alloc("));
        assert!(c.contains("chelis_indices_to_flat(indices, t1->strides, t1->ndim)"));
        assert!(!c.contains("chelis_alloc_view(1, (int[]){ 3 }, CHELIS_F32, t1->data)"));
    }

    #[test]
    fn load_emits_input_reference() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("inputs[0]"));
    }

    #[test]
    fn repeated_load_names_share_one_input_slot() {
        let mut dag = Dag::new();
        let x0 = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        let x1 = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        dag.add_node(RiscOp::Add, vec![x0, x1], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("if (n_in != 1)"));
        assert!(c.contains("chelis_tensor *t0 = inputs[0];"));
        assert!(c.contains("chelis_tensor *t1 = inputs[0];"));
    }

    #[test]
    fn input_labels_follow_first_load_occurrence() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load {
                name: "b".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        dag.add_node(
            RiscOp::Load {
                name: "a".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        dag.add_node(
            RiscOp::Load {
                name: "b".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        assert_eq!(CEmitter::input_labels(&dag), vec!["b", "a"]);
    }

    #[test]
    fn function_signature_correct() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "my_func");
        assert!(c.contains(
            "void my_func(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out)"
        ));
    }

    #[test]
    fn includes_runtime_header() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("#include \"chelis_runtime.h\""));
    }

    #[test]
    fn pad_emits_fill_and_copy() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(3));
        dag.add_node(
            RiscOp::Pad {
                padding: vec![(1, 1)],
                fill: 0.0,
            },
            vec![a],
            vec_f32(5),
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_fill_f32"));
        assert!(c.contains("dst_indices[0] = src_indices[0] + 1"));
    }

    #[test]
    fn shrink_emits_offset_copy() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(5));
        dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(1, 4)],
            },
            vec![a],
            vec_f32(3),
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("src_indices[0] = dst_indices[0] + 1"));
    }

    #[test]
    fn stride_emits_stride_multiply() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
        dag.add_node(RiscOp::Stride { strides: vec![2] }, vec![a], vec_f32(2));
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("strides[0] = t0->strides[0] * 2"));
    }

    #[test]
    fn add_then_mul_chains() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let d = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Mul, vec![c, d], scalar_f32());
        let code = CEmitter::emit_dag(&dag, "test_fn");
        // t2 is add result, t4 is mul result
        assert!(code.contains("t2->data"));
        assert!(code.contains("t4->data"));
    }

    #[test]
    fn vector_add_uses_correct_shape() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4));
        dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4));
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("(int[]){ 4 }"));
    }

    #[test]
    fn sum_then_neg_chains() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(3));
        let s = dag.add_node(RiscOp::Sum { axis: 0 }, vec![a], scalar_f32());
        dag.add_node(RiscOp::Neg, vec![s], scalar_f32());
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("acc +="));
        assert!(c.contains("-t1->data[idx]"));
    }

    #[test]
    fn load_is_borrowed_not_freed() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(!c.contains("chelis_free(t0);"));
        assert!(c.contains("outputs[0] = chelis_contiguous(t0);"));
    }

    #[test]
    fn roots_are_emitted_as_multiple_outputs() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_root(a);
        dag.add_root(b);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("if (n_out != 2)"));
        assert!(c.contains("outputs[0] = chelis_contiguous(t0);"));
        assert!(c.contains("outputs[1] = chelis_contiguous(t1);"));
    }

    #[test]
    fn bool_outputs_use_bool_dtype() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(
            RiscOp::CmpLt,
            vec![a, b],
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_alloc(1, (int[]){1}, CHELIS_BOOL);"));
    }

    #[test]
    fn matmul_pattern_emits_cblas_call() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let mul = dag.add_node(
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
        let c = CEmitter::emit_dag_with_options(
            &dag,
            "test_fn",
            crate::CodegenOptions { use_blas: true },
        );
        assert!(c.contains("cblas_sgemm("));
    }

    #[test]
    fn default_codegen_uses_generic_matmul_path() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let mul = dag.add_node(
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(!c.contains("cblas_sgemm("));
        assert!(c.contains("for (int k = 0; k < 3; k++) {"));
    }

    #[test]
    #[should_panic(expected = "Phase 0f C backend only supports f32/bool tensors")]
    fn unsupported_precision_panics() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Const { value: 1.0 },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::F64,
            },
        );
        let _ = CEmitter::emit_dag(&dag, "test_fn");
    }
}
