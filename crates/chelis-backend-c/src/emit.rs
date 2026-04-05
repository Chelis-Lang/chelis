//! RISC DAG to C source code emission.

use chelis_ir::dag::{Dag, DagNode, DimInfo, NodeId, RiscOp, TensorType};
use chelis_types::types::Prim;

/// Emits C source code from a RISC DAG.
pub struct CEmitter {
    lines: Vec<String>,
    indent: usize,
    use_blas: bool,
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

        let mut e = CEmitter {
            lines: Vec::new(),
            indent: 0,
            use_blas: options.use_blas,
        };

        e.line("#include \"chelis_runtime.h\"");
        if e.use_blas {
            e.line("#include <cblas.h>");
        }
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

        for node in dag.nodes() {
            if let RiscOp::Load { name } = &node.op {
                let input_idx = *input_slots
                    .get(name)
                    .unwrap_or_else(|| panic!("missing input slot for load '{name}'"));
                e.emit_load(node.id.0, input_idx);
            } else {
                e.emit_node(node, dag);
            }
        }

        for (slot, output) in output_specs.iter().enumerate() {
            if output.is_store {
                e.line(&format!("outputs[{slot}] = t{};", output.id.0));
            } else {
                e.line(&format!(
                    "outputs[{slot}] = chelis_contiguous(t{});",
                    output.id.0
                ));
            }
        }

        let cleanup = crate::memory::emit_cleanup(
            dag,
            &output_specs
                .iter()
                .map(|output| output.id)
                .collect::<Vec<_>>(),
        );
        for line in cleanup {
            e.lines.push(line);
        }

        e.indent = 0;
        e.line("}");
        e.lines.join("\n")
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
            RiscOp::Sum { axis } => {
                self.emit_reduce_sum(id, *axis, &node.inputs, &node.output_type, dag);
            }
            RiscOp::MaxReduce { axis } => {
                self.emit_reduce_max(id, *axis, &node.inputs, &node.output_type, dag);
            }
            RiscOp::Reshape { .. } => {
                self.emit_reshape(id, &node.inputs, &node.output_type, dag);
            }
            RiscOp::Permute { axes } => {
                self.emit_permute(id, axes, &node.inputs, &node.output_type, dag);
            }
            RiscOp::Expand { axis, size } => {
                self.emit_expand(id, *axis, *size, &node.inputs, &node.output_type, dag);
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
            RiscOp::Cast { .. } => self.emit_cast(id, &node.inputs, &node.output_type),
            RiscOp::Store { name } => self.emit_store(id, name, &node.inputs),
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

    fn shape_literal(ty: &TensorType) -> String {
        let dims: Vec<String> = ty
            .dims
            .iter()
            .map(|d| match d {
                DimInfo::Lit(n) => n.to_string(),
                DimInfo::Named(_, Some(n)) => n.to_string(),
                DimInfo::Named(name, None) => {
                    panic!("unsized named dimension '{name}' in C codegen")
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
                panic!("unsized named dimension '{name}' in C codegen")
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
        let axis_size = Self::dim_size(&input_node.output_type.dims[axis]);
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
        let axis_size = Self::dim_size(&input_node.output_type.dims[axis]);
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
        size: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        _dag: &Dag,
    ) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let _ = size; // used in shape already
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
        dag.add_node(RiscOp::Expand { axis: 0, size: 4 }, vec![a], vec_f32(4));
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
            RiscOp::Expand { axis: 2, size: 4 },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let eb = dag.add_node(
            RiscOp::Expand { axis: 0, size: 2 },
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
            RiscOp::Expand { axis: 2, size: 4 },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let eb = dag.add_node(
            RiscOp::Expand { axis: 0, size: 2 },
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
