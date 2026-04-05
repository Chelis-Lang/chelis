//! RISC DAG to C source code emission.

use chelis_ir::dag::{Dag, DagNode, DimInfo, NodeId, RiscOp, TensorType};

/// Emits C source code from a RISC DAG.
pub struct CEmitter {
    lines: Vec<String>,
    indent: usize,
}

impl CEmitter {
    /// Emit C source for an entire DAG as a function.
    pub fn emit_dag(dag: &Dag, func_name: &str) -> String {
        let mut e = CEmitter {
            lines: Vec::new(),
            indent: 0,
        };

        e.line("#include \"chelis_runtime.h\"");
        e.line("");

        e.line(&format!(
            "void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out) {{"
        ));
        e.indent = 1;

        // Track input counter for Load nodes
        let mut input_idx = 0usize;

        for node in dag.nodes() {
            if matches!(node.op, RiscOp::Load { .. }) {
                e.emit_load(node.id.0, &node.op, &node.output_type, input_idx);
                input_idx += 1;
            } else {
                e.emit_node(node, dag);
            }
        }

        // Return last node as output
        if let Some(last) = dag.nodes().last() {
            e.line(&format!("outputs[0] = t{};", last.id.0));

            // Free all intermediate tensors that are not the output
            let output_ids = vec![last.id];
            let cleanup = crate::memory::emit_cleanup(dag, &output_ids);
            for line in cleanup {
                e.lines.push(line);
            }
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

    // ---- Const ----
    fn emit_const(&mut self, id: usize, value: f64, ty: &TensorType) {
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
        ));
        self.line(&format!("chelis_fill_f32(t{id}, {:.8}f);", value as f32));
    }

    // ---- Load ----
    fn emit_load(&mut self, id: usize, _op: &RiscOp, _ty: &TensorType, input_idx: usize) {
        self.line(&format!("chelis_tensor *t{id} = inputs[{input_idx}];"));
    }

    // ---- Binary elementwise ----
    fn emit_binary(&mut self, id: usize, op: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
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
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
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
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
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
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
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
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
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

    // ---- BLAS matmul (naive fallback) ----
    fn emit_blas_matmul(&mut self, id: usize, info: &crate::blas::MatmulInfo, _dag: &Dag) {
        let a = info.a.0;
        let b = info.b.0;
        let m = info.m;
        let n = info.n;
        let k = info.k;
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc(2, (int[]){{ {m}, {n} }}, CHELIS_F32);"
        ));
        self.line(&format!("for (int i = 0; i < {m}; i++) {{"));
        self.indent += 1;
        self.line(&format!("for (int j = 0; j < {n}; j++) {{"));
        self.indent += 1;
        self.line("float acc = 0.0f;");
        self.line(&format!("for (int p = 0; p < {k}; p++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int a_idx = i * t{a}->strides[0] + p * t{a}->strides[1];"
        ));
        self.line(&format!(
            "int b_idx = p * t{b}->strides[0] + j * t{b}->strides[1];"
        ));
        self.line(&format!("acc += t{a}->data[a_idx] * t{b}->data[b_idx];"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("t{id}->data[i * {n} + j] = acc;"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
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
        if let Some(matmul) = crate::blas::detect_matmul_pattern(dag, NodeId(id)) {
            self.emit_blas_matmul(id, &matmul, dag);
            return;
        }
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::dim_size(&input_node.output_type.dims[axis]);
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
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
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
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
        // If the source might be non-contiguous, make it contiguous first
        self.line(&format!("chelis_tensor *t{id}_src = t{a};"));
        self.line(&format!("if (!chelis_is_contiguous(t{id}_src)) {{"));
        self.indent += 1;
        self.line(&format!("t{id}_src = chelis_contiguous(t{id}_src);"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
        ));
        self.line(&format!(
            "memcpy(t{id}->data, t{id}_src->data, t{id}->size * sizeof(float));"
        ));
        // Free the contiguous copy if one was made
        self.line(&format!("if (t{id}_src != t{a}) {{"));
        self.indent += 1;
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
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
        ));
        // Copy data pointer (share data, adjust strides)
        self.line(&format!("free(t{id}->data);"));
        self.line(&format!("t{id}->data = t{a}->data;"));
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
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
        ));
        self.line(&format!("free(t{id}->data);"));
        self.line(&format!("t{id}->data = t{a}->data;"));
        // Copy strides from source, inserting stride 0 at the expanded axis
        for d in 0..axis {
            self.line(&format!("t{id}->strides[{d}] = t{a}->strides[{d}];"));
        }
        self.line(&format!("t{id}->strides[{axis}] = 0;"));
        self.line(&format!(
            "for (int d = {axis}; d < t{a}->ndim; d++) t{id}->strides[d+1] = t{a}->strides[d];"
        ));
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
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
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
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
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
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
        ));
        self.line(&format!("free(t{id}->data);"));
        self.line(&format!("t{id}->data = t{a}->data;"));
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
        // Phase 0: everything is f32, so cast is identity copy
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, CHELIS_F32);"
        ));
        self.line(&format!(
            "memcpy(t{id}->data, t{a}->data, t{id}->size * sizeof(float));"
        ));
    }

    // ---- Store ----
    fn emit_store(&mut self, id: usize, name: &str, inputs: &[NodeId]) {
        let a = inputs[0].0;
        self.line(&format!("chelis_tensor *t{id} = t{a}; /* store: {name} */"));
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
        assert!(c.contains("store: out"));
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
}
