//! RISC DAG to C source code emission.

use chelis_ir::dag::{
    Dag, DagNode, DimExpr, DimInfo, FusedInput, FusedStep, FusedStepOp, NodeId, RiscOp, TensorType,
    symbolic_bindings,
};
use chelis_types::types::Prim;

use crate::memory::{MemoryPlan, NodeMemoryKind};

/// Emits C source code from a RISC DAG.
pub struct CEmitter {
    lines: Vec<String>,
    indent: usize,
    use_blas: bool,
    /// Which vectorized math library to target for fused-elem SIMD emission (Level 3b).
    math_lib: crate::MathLib,
    /// FusedElem nodes inlined into a trailing reduction (no standalone emission).
    reduction_inlined: std::collections::HashSet<usize>,
    /// Backing-slot plan for materialized C tensors.
    memory_plan: MemoryPlan,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone, Copy)]
struct FusedInPlaceSpec {
    reusable_input: NodeId,
    slot_has_later_owner: bool,
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
        Self::validate_sparse_contracts(dag);
        // Some Surf signatures surface anonymous (Named("", None)) axes into
        // the lowered DAG (e.g. a rank-1 tensor parameter whose dim has no
        // declared name). These would emit `int  = inputs[0]->shape[0];` and
        // `(int[]){ }` shape literals, neither of which compiles. Rewrite
        // empty dim names to a stable synthesized identifier before the
        // emitter walks the DAG.
        let dag_owned = Self::rename_anonymous_dims(dag);
        let dag = &dag_owned;

        let reduction_inlined = chelis_ir::fuse::reduction_inlined_fused_elems(dag);
        let math_lib = options
            .math_lib_override
            .unwrap_or_else(crate::MathLib::detect);
        let output_specs = Self::output_specs(dag);
        let output_ids = output_specs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>();
        let memory_plan = MemoryPlan::build(dag, &output_ids, &reduction_inlined);

        let mut e = CEmitter {
            lines: Vec::new(),
            indent: 0,
            use_blas: options.use_blas,
            math_lib,
            reduction_inlined: reduction_inlined.iter().map(|id| id.0).collect(),
            memory_plan,
        };

        e.line("#include \"chelis_runtime.h\"");
        e.line("#include <assert.h>");
        if e.use_blas {
            e.line("#include \"chelis_blas.h\"");
        }
        if e.math_lib != crate::MathLib::None {
            e.line("#include \"chelis_math.h\"");
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
        e.line("#ifndef CHELIS_EFFECTIVE_UNIFORM_SEED");
        e.line("#define CHELIS_EFFECTIVE_UNIFORM_SEED(seed) (seed)");
        e.line("#endif");
        e.line("");

        let linkage = if options.static_entry { "static " } else { "" };
        // Producer-supplied `func_name` (typically the source filename's
        // stem) flows into both an identifier context (the `void {name}(...)`
        // declarator) and a format-string context (the `fprintf(stderr,
        // "{name}: ...")` runtime-error reports). The format-string context
        // requires escaping `%`, `\\`, `"`, and control bytes per
        // spec/upstream-bugs/producer-string-sanitization.md. The identifier
        // context inherits whatever the upstream chooses; if `func_name`
        // contains non-identifier bytes the emitted C will fail to compile,
        // which is the desired outcome (loud failure, not silent injection).
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);
        e.line(&format!(
            "{linkage}void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out) {{"
        ));
        e.indent = 1;

        let input_labels = Self::input_labels(dag);
        let input_slots = Self::input_slots(&input_labels);
        let expected_inputs = input_labels.len();
        let expected_outputs = output_specs.len();

        e.line(&format!("if (n_in != {expected_inputs}) {{"));
        e.indent += 1;
        e.line(&format!(
            "fprintf(stderr, \"{func_name_fmt}: expected %d inputs, got %d\\n\", {expected_inputs}, n_in);"
        ));
        e.line("abort();");
        e.indent -= 1;
        e.line("}");
        if expected_inputs > 0 {
            e.line("if (inputs == NULL) {");
            e.indent += 1;
            e.line(&format!(
                "fprintf(stderr, \"{func_name_fmt}: inputs array is NULL but %d inputs are required\\n\", {expected_inputs});"
            ));
            e.line("abort();");
            e.indent -= 1;
            e.line("}");
        }

        e.line(&format!("if (n_out != {expected_outputs}) {{"));
        e.indent += 1;
        e.line(&format!(
            "fprintf(stderr, \"{func_name_fmt}: expected %d outputs, got %d\\n\", {expected_outputs}, n_out);"
        ));
        e.line("abort();");
        e.indent -= 1;
        e.line("}");
        if expected_outputs > 0 {
            e.line("if (outputs == NULL) {");
            e.indent += 1;
            e.line(&format!(
                "fprintf(stderr, \"{func_name_fmt}: outputs array is NULL but %d outputs are required\\n\", {expected_outputs});"
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
                    .get(name.as_str())
                    .unwrap_or_else(|| panic!("missing input slot for load '{name}'"));
                e.emit_span_comments(node);
                e.emit_load(node.id.0, input_idx);
            } else {
                e.emit_span_comments(node);
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

        let cleanup = e.memory_plan.emit_cleanup(&output_ids);
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
        fn rewrite_dim(id: NodeId, axis: usize, dim: &DimInfo) -> DimInfo {
            match dim {
                DimInfo::Named(name, size) if is_anon(name) => {
                    DimInfo::Named(format!("_anon_dim_{}_{}", id.0, axis), *size)
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
                if let RiscOp::Gather { axis } = &node.op
                    && node.inputs.len() == 2
                    && let (Some(values), Some(indices)) =
                        (out.get(node.inputs[0]), out.get(node.inputs[1]))
                    && *axis < values.output_type.dims.len()
                {
                    let mut dims = Vec::new();
                    dims.extend_from_slice(&values.output_type.dims[..*axis]);
                    dims.extend(indices.output_type.dims.iter().cloned());
                    dims.extend_from_slice(&values.output_type.dims[*axis + 1..]);
                    new_ty.dims = dims;
                } else if let Some(first_input) =
                    node.inputs.first().and_then(|input| out.get(*input))
                    && first_input.output_type.dims.len() == new_ty.dims.len()
                {
                    new_ty.dims = first_input.output_type.dims.clone();
                } else {
                    new_ty.dims = new_ty
                        .dims
                        .iter()
                        .enumerate()
                        .map(|(axis, dim)| rewrite_dim(id, axis, dim))
                        .collect();
                }
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
            RiscOp::Cos => self.emit_unary_func(id, "cosf", &node.inputs, &node.output_type),
            RiscOp::Tan => self.emit_unary_func(id, "tanf", &node.inputs, &node.output_type),
            RiscOp::Atan => self.emit_unary_func(id, "atanf", &node.inputs, &node.output_type),
            RiscOp::Abs => self.emit_unary_func(id, "fabsf", &node.inputs, &node.output_type),
            RiscOp::Floor => self.emit_unary_func(id, "floorf", &node.inputs, &node.output_type),
            RiscOp::Ceil => self.emit_unary_func(id, "ceilf", &node.inputs, &node.output_type),
            RiscOp::UniformLike { low, high, seed } => {
                self.emit_uniform_like(id, *low, *high, *seed, &node.output_type)
            }
            RiscOp::Dropout { .. } => {
                unreachable!("dropout should be rejected before C code generation")
            }
            RiscOp::Copy => self.emit_realize(id, &node.inputs, &node.output_type),
            RiscOp::Drop => {}
            RiscOp::Sum { axis, .. } => {
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
                    Some("chelis_min_f32"),
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
                    None,
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
            RiscOp::OneHot { .. } => {
                panic!(
                    "C backend: internal OneHot must be consumed by specialization before codegen"
                )
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
            RiscOp::Cast { .. } => self.emit_cast(id, &node.inputs, &node.output_type, dag),
            RiscOp::Store { name } => self.emit_store(id, name.as_str(), &node.inputs),
            RiscOp::FusedElem { ops } => {
                let in_place =
                    Self::fused_in_place_spec(node, dag).map(|reusable_input| FusedInPlaceSpec {
                        reusable_input,
                        slot_has_later_owner: self.slot_has_later_owner(id, dag),
                    });
                self.emit_fused_elem(id, ops, &node.inputs, &node.output_type, in_place);
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
                );
            }
            RiscOp::Gather { axis } => {
                self.emit_sparse_gather(id, *axis, &node.inputs, &node.output_type, dag);
            }
            RiscOp::ScatterAdd { axis } => {
                self.emit_sparse_scatter_add(id, *axis, &node.inputs, &node.output_type, dag);
            }
            RiscOp::Scatter { axis } => {
                self.emit_sparse_scatter_replace(id, *axis, &node.inputs, &node.output_type, dag);
            }
        }
    }

    fn line(&mut self, s: &str) {
        let prefix = "    ".repeat(self.indent);
        self.lines.push(format!("{prefix}{s}"));
    }

    /// Emit `// span: <id>` comment lines for a node's `span_id ∪ merged_spans`.
    ///
    /// Per `spec/design/chelis_span_survival.md` §2.4 (S4):
    ///   * canonical `span_id` first (if present),
    ///   * then `merged_spans` lex-sorted (deduped against `span_id`).
    ///
    /// `merged_spans` are already kept lex-sorted/deduped by the
    /// `chelis_ir::span_merge` helpers, so we sort defensively here.
    /// No-op when both fields are empty (the common case for hand-written
    /// Chelis or for span-free Deep input).
    fn emit_span_comments(&mut self, node: &DagNode) {
        if node.span_id.is_none() && node.merged_spans.is_empty() {
            return;
        }
        if let Some(canonical) = node.span_id.as_deref() {
            let safe = chelis_ir::span_sanitize::sanitize_for_comment(canonical);
            self.line(&format!("// span: {safe}"));
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
            self.line(&format!("// span: {safe}"));
        }
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
                && seen.insert(name.as_str().to_string())
            {
                labels.push(name.as_str().to_string());
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
                Prim::F32 | Prim::F64 | Prim::Bool | Prim::Int32 | Prim::Int64 => {}
                other => panic!(
                    "C backend does not yet support {} tensors, found at node {}",
                    other.name(),
                    node.id.0
                ),
            }

            if let RiscOp::Cast { new_precision } = node.op
                && !matches!(
                    new_precision,
                    Prim::F32 | Prim::F64 | Prim::Int32 | Prim::Int64
                )
            {
                panic!(
                    "C backend does not yet support casts to {}, found at node {}",
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
                if let Some(prev_ty) = seen.get(name.as_str()) {
                    assert_eq!(
                        prev_ty, &node.output_type,
                        "Load name '{}' used with inconsistent tensor types in C codegen",
                        name
                    );
                } else {
                    seen.insert(name.as_str().to_string(), node.output_type.clone());
                }
            }
        }
    }

    fn validate_sparse_contracts(dag: &Dag) {
        for node in dag.nodes() {
            match &node.op {
                RiscOp::Gather { .. } => {
                    if node.inputs.len() != 2 {
                        continue;
                    }
                    let values_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                    let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                    if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
                        panic!(
                            "C backend sparse gather requires int32/int64 indices, got {} at node {}",
                            indices_ty.precision.name(),
                            node.id.0
                        );
                    }
                    if node.output_type.precision != values_ty.precision {
                        panic!(
                            "C backend sparse gather output precision must match values at node {}",
                            node.id.0
                        );
                    }
                }
                RiscOp::ScatterAdd { .. } => {
                    if node.inputs.len() != 3 {
                        continue;
                    }
                    let target_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                    let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                    let updates_ty = &dag.get(node.inputs[2]).unwrap().output_type;
                    if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
                        panic!(
                            "C backend sparse scatter_add requires int32/int64 indices, got {} at node {}",
                            indices_ty.precision.name(),
                            node.id.0
                        );
                    }
                    if updates_ty.precision != target_ty.precision
                        || node.output_type.precision != target_ty.precision
                    {
                        panic!(
                            "C backend sparse scatter_add target, update, and output precision must match at node {}",
                            node.id.0
                        );
                    }
                }
                RiscOp::Scatter { .. } => {
                    if node.inputs.len() != 3 {
                        continue;
                    }
                    let target_ty = &dag.get(node.inputs[0]).unwrap().output_type;
                    let indices_ty = &dag.get(node.inputs[1]).unwrap().output_type;
                    let updates_ty = &dag.get(node.inputs[2]).unwrap().output_type;
                    if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
                        panic!(
                            "C backend sparse scatter_replace requires int32/int64 indices, got {} at node {}",
                            indices_ty.precision.name(),
                            node.id.0
                        );
                    }
                    if updates_ty.precision != target_ty.precision
                        || node.output_type.precision != target_ty.precision
                    {
                        panic!(
                            "C backend sparse scatter_replace target, update, and output precision must match at node {}",
                            node.id.0
                        );
                    }
                }
                _ => {}
            }
        }
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
        // deterministic so the emitted C is byte-identical across runs
        // for the same input. Sort by label; the lookup is by name and
        // the emitted lines are independent per label.
        // See spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md.
        let input_types = Self::input_types(dag);
        let mut sorted_labels: Vec<&String> = input_types.keys().collect();
        sorted_labels.sort();
        // Producer-supplied strings flowing into the fprintf format string
        // baked into a `"..."` C string literal. Sanitize once per emission
        // boundary per spec/upstream-bugs/producer-string-sanitization.md.
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);
        for label in sorted_labels {
            let ty = &input_types[label];
            let slot = input_slots[label];
            // `label` originates as `LoadStoreName::as_str()` (validated)
            // but we route through the format-string sanitizer to lock the
            // architectural pattern: every producer-supplied string into a
            // format-string context goes through `sanitize_for_format_string`.
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
            // `binding.name` flows into BOTH an identifier context (the
            // emitted `int {name} = ...;` declarator) and a format-string
            // context (the fprintf below). The identifier emission is
            // guarded by parser/IR construction; the format-string
            // emission needs `%`/`\\`/`"`/control sanitization here.
            // `occurrence.input_label` is a Load name (LoadStoreName-
            // validated) but we route both through the format-string
            // sanitizer to lock the architectural pattern.
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

    fn shape_literal(ty: &TensorType) -> String {
        let dims: Vec<String> = ty
            .dims
            .iter()
            .map(|dim| Self::emit_dim_expr(&DimExpr::from(dim)))
            .collect();
        if dims.is_empty() {
            "NULL".to_string()
        } else {
            format!("(int[]){{ {} }}", dims.join(", "))
        }
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
            Prim::F64 => "CHELIS_F64",
            Prim::Bool => "CHELIS_BOOL",
            Prim::Int32 => "CHELIS_I32",
            Prim::Int64 => "CHELIS_I64",
            other => panic!("C backend does not yet support {} tensors", other.name()),
        }
    }

    /// Returns the C element type for direct element access in generated loops.
    /// F32 and Bool use `float` (the native data pointer type).
    /// Int32 uses `int32_t` (reinterpret cast; sizeof matches float).
    /// Int64 uses `int64_t` (reinterpret cast; sizeof is 2x float, alloc adjusts).
    /// F64 uses `double` (reinterpret cast; sizeof is 2x float, alloc adjusts).
    ///
    /// **WS-A0 footgun fix.** This used to fall through to `"float"` for
    /// any unhandled `Prim`, which silently downgraded the wider
    /// active dtypes (f16, bf16, int8, int16) to single-precision in
    /// emitted C code. WS-A1 expands the C backend to handle those
    /// dtypes; until then this function panics with the unhandled
    /// variant so the silent downgrade cannot recur and so any new
    /// dtype landing later in the active set per
    /// `spec/04-type-system.md` §1.1 produces an explicit gap, not a
    /// quiet wrong answer.
    fn elem_type(ty: &TensorType) -> &'static str {
        match ty.precision {
            Prim::F32 | Prim::Bool => "float",
            Prim::F64 => "double",
            Prim::Int32 => "int32_t",
            Prim::Int64 => "int64_t",
            other => panic!(
                "C backend does not yet support `{}` tensors; the silent \
                 default-arm downgrade was removed by WS-A0 to surface \
                 missing dtype emit logic. WS-A1 widens this match.",
                other.name()
            ),
        }
    }

    fn elem_size_expr(ty: &TensorType) -> String {
        format!("sizeof({})", Self::elem_type(ty))
    }

    /// Returns true when the tensor's element type is `double`, requiring
    /// double-precision math helpers (`exp` vs `expf`) and `chelis_fill_f64`.
    fn is_f64(ty: &TensorType) -> bool {
        matches!(ty.precision, Prim::F64)
    }

    /// Double-precision equivalent of a single-precision C math symbol used
    /// by the emitter. Only the set we actually route through emit_unary_func
    /// and emit_binary_func is mapped here; unmapped symbols pass through
    /// unchanged so the emitter never silently renames an unexpected function.
    fn double_math_fn(scalar_f: &str) -> &str {
        match scalar_f {
            "expf" => "exp",
            "logf" => "log",
            "sinf" => "sin",
            "sqrtf" => "sqrt",
            "cosf" => "cos",
            "tanf" => "tan",
            "atanf" => "atan",
            "fabsf" => "fabs",
            "floorf" => "floor",
            "ceilf" => "ceil",
            "fmaxf" => "fmax",
            "fminf" => "fmin",
            other => other,
        }
    }

    fn slot_id_for_node(&self, id: usize) -> usize {
        match self.memory_plan.node_kind(NodeId(id)) {
            NodeMemoryKind::SlotBacked { slot } => *slot,
            other => panic!("node {id} does not own slot-backed storage: {other:?}"),
        }
    }

    fn emit_slot_allocation_if_needed(&mut self, id: usize, ty: &TensorType) {
        let slot_id = self.slot_id_for_node(id);
        let slot = self.memory_plan.slot(slot_id);
        if slot.first_owner != NodeId(id) {
            return;
        }
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *chelis_slot{slot_id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
    }

    fn emit_slot_wrapper(&mut self, id: usize, ty: &TensorType) {
        self.emit_slot_allocation_if_needed(id, ty);
        let slot_id = self.slot_id_for_node(id);
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc_view({ndim}, {shape}, {dtype}, chelis_slot{slot_id}->data);"
        ));
    }

    fn emit_fused_in_place_wrapper(&mut self, id: usize, ty: &TensorType, spec: FusedInPlaceSpec) {
        let slot_id = self.slot_id_for_node(id);
        let slot_is_first_owner = self.memory_plan.slot(slot_id).first_owner == NodeId(id);
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        if slot_is_first_owner {
            self.line(&format!("chelis_tensor *chelis_slot{slot_id} = NULL;"));
            if spec.slot_has_later_owner {
                self.line(&format!(
                    "chelis_slot{slot_id} = chelis_alloc({ndim}, {shape}, {dtype});"
                ));
            }
        }
        self.line(&format!("chelis_tensor *t{id};"));
        self.line(&format!(
            "if (chelis_is_contiguous(t{})) {{",
            spec.reusable_input.0
        ));
        self.indent += 1;
        self.line(&format!(
            "t{id} = chelis_alloc_view({ndim}, {shape}, {dtype}, t{}->data);",
            spec.reusable_input.0
        ));
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        if slot_is_first_owner && !spec.slot_has_later_owner {
            self.line(&format!(
                "chelis_slot{slot_id} = chelis_alloc({ndim}, {shape}, {dtype});"
            ));
        }
        self.line(&format!(
            "t{id} = chelis_alloc_view({ndim}, {shape}, {dtype}, chelis_slot{slot_id}->data);"
        ));
        self.indent -= 1;
        self.line("}");
    }

    fn slot_has_later_owner(&self, id: usize, dag: &Dag) -> bool {
        let slot_id = self.slot_id_for_node(id);
        dag.nodes()
            .iter()
            .skip(id + 1)
            .any(|node| matches!(self.memory_plan.node_kind(node.id), NodeMemoryKind::SlotBacked { slot } if *slot == slot_id))
    }

    fn fused_in_place_spec(node: &DagNode, dag: &Dag) -> Option<NodeId> {
        let reusable_input = node.reusable_input?;
        if !matches!(node.op, RiscOp::FusedElem { .. }) {
            return None;
        }
        if node
            .inputs
            .iter()
            .filter(|&&input| input == reusable_input)
            .count()
            != 1
        {
            return None;
        }
        let input_node = dag.get(reusable_input)?;
        // Perf-F2(c): the reusable input's output_type must be
        // binder-equivalent to the FusedElem's output_type, not just
        // PartialEq-equal. This admits scoped same-property `forall` /
        // binder-equivalent aliases (e.g. `Lit(4)` vs
        // `Named("seq", Some(4))` for the same scope), which the
        // upstream linearity analyzer already proved single-use.
        if !Self::binder_equivalent_tensor_type(&input_node.output_type, &node.output_type) {
            return None;
        }
        let consumer_count = dag
            .nodes()
            .iter()
            .flat_map(|candidate| candidate.inputs.iter())
            .filter(|&&input| input == reusable_input)
            .count()
            + dag
                .roots()
                .iter()
                .filter(|&&root| root == reusable_input)
                .count();
        if consumer_count != 1 {
            return None;
        }
        Some(reusable_input)
    }

    /// Perf-F2(c): conservative binder-equivalent equality for
    /// `TensorType` shape comparisons in the in-place fused-elementwise
    /// aliasing gate.
    ///
    /// Two tensor types are binder-equivalent iff:
    ///   * precisions match exactly,
    ///   * ranks match exactly,
    ///   * each pair of dim descriptors is binder-equivalent per
    ///     `binder_equivalent_dim_info` below.
    ///
    /// This is strictly weaker than `DimExprKey::normalized_key` (which
    /// alpha-renames symbolic dims by shape alone, an unsound expansion
    /// per the warning in `chelis_ir::dag::DimExprKey`'s rustdoc) and
    /// strictly stronger than ignoring binder names. It accepts only
    /// dim pairs whose binder name or known-size is provably consistent.
    fn binder_equivalent_tensor_type(a: &TensorType, b: &TensorType) -> bool {
        if a.precision != b.precision {
            return false;
        }
        if a.dims.len() != b.dims.len() {
            return false;
        }
        a.dims
            .iter()
            .zip(b.dims.iter())
            .all(|(da, db)| Self::binder_equivalent_dim_info(da, db))
    }

    /// Two `DimInfo`s are binder-equivalent under the same forall scope
    /// when their known-or-binder identity provably matches:
    ///   * `Lit(n)` ≡ `Lit(n)` — identical concrete sizes.
    ///   * `Named(n1, _)` ≡ `Named(n2, _)` — identical binder names
    ///     **and** consistent known sizes when both are known.
    ///   * `Lit(n)` ≡ `Named(_, Some(n))` and vice versa — a concrete
    ///     literal matches a named binder that has been resolved to the
    ///     same size (e.g. specialize lowering a `Named("seq", Some(4))`
    ///     to `Lit(4)` mid-pipeline still admits in-place aliasing).
    ///   * Everything else is rejected. `Lit` vs `Named(_, None)` is
    ///     intentionally rejected: a binder with unresolved size has no
    ///     evidence it matches a specific literal — `n` may differ.
    fn binder_equivalent_dim_info(a: &DimInfo, b: &DimInfo) -> bool {
        match (a, b) {
            (DimInfo::Lit(la), DimInfo::Lit(lb)) => la == lb,
            (DimInfo::Named(na, sa), DimInfo::Named(nb, sb)) => {
                if na != nb {
                    return false;
                }
                match (sa, sb) {
                    (Some(la), Some(lb)) => la == lb,
                    _ => true,
                }
            }
            (DimInfo::Lit(la), DimInfo::Named(_, Some(lb)))
            | (DimInfo::Named(_, Some(la)), DimInfo::Lit(lb)) => la == lb,
            _ => false,
        }
    }

    // ---- Const ----
    fn emit_const(&mut self, id: usize, value: f64, ty: &TensorType) {
        self.emit_slot_wrapper(id, ty);
        match ty.precision {
            Prim::Int64 => {
                self.line(&format!(
                    "chelis_fill_i64(t{id}, (int64_t){});",
                    value as i64
                ));
            }
            Prim::Int32 => {
                self.line(&format!(
                    "{{ int32_t *__p = (int32_t*)t{id}->data; for (int __i = 0; __i < t{id}->size; __i++) __p[__i] = (int32_t){}; }}",
                    value as i32
                ));
            }
            Prim::F64 => {
                self.line(&format!("chelis_fill_f64(t{id}, {value:.17});"));
            }
            _ => {
                self.line(&format!("chelis_fill_f32(t{id}, {:.8}f);", value as f32));
            }
        }
    }

    // ---- Load ----
    fn emit_load(&mut self, id: usize, input_idx: usize) {
        self.line(&format!("chelis_tensor *t{id} = inputs[{input_idx}];"));
    }

    // ---- Binary elementwise ----
    fn emit_binary(&mut self, id: usize, op: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let et = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && t{a}->size == t{id}->size && t{b}->size == t{id}->size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}->size == t{id}->size);"));
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}->data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}->data;"
        ));
        self.line(&format!(
            "const {et}* restrict __in_b_{id} = (const {et}*)t{b}->data;"
        ));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "__out_{id}[i] = __in_a_{id}[i] {op} __in_b_{id}[i];"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
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
            "(({et}*)t{id}->data)[i] = (({et}*)t{a}->data)[idx_a] {op} (({et}*)t{b}->data)[idx_b];"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- Binary func (fmaxf etc.) ----
    fn emit_binary_func(&mut self, id: usize, func: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let et = Self::elem_type(ty);
        let f = if Self::is_f64(ty) {
            Self::double_math_fn(func)
        } else {
            func
        };
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && t{a}->size == t{id}->size && t{b}->size == t{id}->size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}->size == t{id}->size);"));
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}->data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}->data;"
        ));
        self.line(&format!(
            "const {et}* restrict __in_b_{id} = (const {et}*)t{b}->data;"
        ));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "__out_{id}[i] = {f}(__in_a_{id}[i], __in_b_{id}[i]);"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
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
            "(({et}*)t{id}->data)[i] = {f}((({et}*)t{a}->data)[idx_a], (({et}*)t{b}->data)[idx_b]);"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- CmpLt ----
    fn emit_cmplt(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && t{a}->size == t{id}->size && t{b}->size == t{id}->size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}->size == t{id}->size);"));
        self.line(&format!("float* restrict __out_{id} = t{id}->data;"));
        self.line(&format!("const float* restrict __in_a_{id} = t{a}->data;"));
        self.line(&format!("const float* restrict __in_b_{id} = t{b}->data;"));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "__out_{id}[i] = (__in_a_{id}[i] < __in_b_{id}[i]) ? 1.0f : 0.0f;"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
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
        self.indent -= 1;
        self.line("}");
    }

    // ---- Unary elementwise ----
    fn emit_unary(&mut self, id: usize, op: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
        self.indent += 1;
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}->data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}->data;"
        ));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line(&format!("__out_{id}[i] = {op}__in_a_{id}[i];"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
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
        self.line(&format!(
            "(({et}*)t{id}->data)[i] = {op}(({et}*)t{a}->data)[idx];"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- Unary func (expf, logf, sinf, sqrtf) ----
    fn emit_unary_func(&mut self, id: usize, func: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        let is_f64 = Self::is_f64(ty);
        let f = if is_f64 {
            Self::double_math_fn(func)
        } else {
            func
        };
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
        self.indent += 1;
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}->data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}->data;"
        ));
        // SIMD/batched math paths currently only support f32. For f64 tensors
        // or when no SIMD library is selected, fall back to the scalar OMP SIMD
        // loop with the appropriate (double- or single-precision) math function.
        if !is_f64 && self.math_lib == crate::MathLib::VForce {
            if let Some(vf_fn) = Self::vforce_func(func) {
                // vForce whole-array batch API on macOS (Accelerate.framework).
                self.line("{");
                self.indent += 1;
                self.line(&format!("int __n_{id} = t{id}->size;"));
                self.line(&format!("{vf_fn}(__out_{id}, __in_a_{id}, &__n_{id});"));
                self.indent -= 1;
                self.line("}");
            } else {
                self.line("#pragma omp parallel for simd");
                self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
                self.indent += 1;
                self.line(&format!("__out_{id}[i] = {f}(__in_a_{id}[i]);"));
                self.indent -= 1;
                self.line("}");
            }
        } else if !is_f64 && self.math_lib == crate::MathLib::Sleef {
            if let Some(simd_macro) = Self::sleef_macro(func) {
                self.line("#ifdef CHELIS_HAS_SLEEF");
                self.line("{");
                self.indent += 1;
                self.line(&format!("int __i_{id} = 0;"));
                self.line(&format!(
                    "for (; __i_{id} + 8 <= t{id}->size; __i_{id} += 8) {{"
                ));
                self.indent += 1;
                self.line(&format!(
                    "__m256 __v_{id} = _mm256_loadu_ps(__in_a_{id} + __i_{id});"
                ));
                self.line(&format!("__m256 __r_{id} = {simd_macro}(__v_{id});"));
                self.line(&format!(
                    "_mm256_storeu_ps(__out_{id} + __i_{id}, __r_{id});"
                ));
                self.indent -= 1;
                self.line("}");
                self.line(&format!("for (; __i_{id} < t{id}->size; __i_{id}++) {{"));
                self.indent += 1;
                self.line(&format!(
                    "__out_{id}[__i_{id}] = {f}(__in_a_{id}[__i_{id}]);"
                ));
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
                self.line("#else");
                self.line("#pragma omp parallel for simd");
                self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
                self.indent += 1;
                self.line(&format!("__out_{id}[i] = {f}(__in_a_{id}[i]);"));
                self.indent -= 1;
                self.line("}");
                self.line("#endif");
            } else {
                self.line("#pragma omp parallel for simd");
                self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
                self.indent += 1;
                self.line(&format!("__out_{id}[i] = {f}(__in_a_{id}[i]);"));
                self.indent -= 1;
                self.line("}");
            }
        } else {
            self.line("#pragma omp parallel for simd");
            self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
            self.indent += 1;
            self.line(&format!("__out_{id}[i] = {f}(__in_a_{id}[i]);"));
            self.indent -= 1;
            self.line("}");
        }
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
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
        self.line(&format!(
            "(({et}*)t{id}->data)[i] = {f}((({et}*)t{a}->data)[idx]);"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// Map a scalar C math function name to its vForce batch equivalent.
    fn vforce_func(scalar_func: &str) -> Option<&'static str> {
        match scalar_func {
            "expf" => Some("vvexpf"),
            "logf" => Some("vvlogf"),
            "sinf" => Some("vvsinf"),
            "sqrtf" => Some("vvsqrtf"),
            _ => None,
        }
    }

    /// Map a scalar C math function name to its Sleef AVX2 8-wide macro.
    fn sleef_macro(scalar_func: &str) -> Option<&'static str> {
        match scalar_func {
            "expf" => Some("CHELIS_EXPF8"),
            "logf" => Some("CHELIS_LOGF8"),
            "sinf" => Some("CHELIS_SINF8"),
            "sqrtf" => Some("CHELIS_SQRTF8"),
            _ => None,
        }
    }

    fn emit_uniform_like(&mut self, id: usize, low: f64, high: f64, seed: u64, ty: &TensorType) {
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "uint64_t t{id}_seed = CHELIS_EFFECTIVE_UNIFORM_SEED({seed}ULL);"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "t{id}->data[i] = chelis_uniform_sample_f32(t{id}_seed, (uint64_t)i, {:.8}f, {:.8}f);",
            low as f32, high as f32
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Fused elementwise helpers ----

    /// Returns true if any step in a fused kernel performs a transcendental math op.
    fn has_math_ops(ops: &[FusedStep]) -> bool {
        ops.iter().any(|s| {
            matches!(
                s.op,
                FusedStepOp::Exp
                    | FusedStepOp::Log
                    | FusedStepOp::Sin
                    | FusedStepOp::Sqrt
                    | FusedStepOp::Cos
                    | FusedStepOp::Tan
                    | FusedStepOp::Atan
                    | FusedStepOp::Abs
                    | FusedStepOp::Floor
                    | FusedStepOp::Ceil
            )
        })
    }

    /// Emit one fused-step expression for the scalar fast/tail path.
    fn scalar_step_expr(
        op: &FusedStepOp,
        resolve: &dyn Fn(&FusedInput) -> String,
        inputs: &[FusedInput],
    ) -> String {
        match op {
            FusedStepOp::Add => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("{a} + {b}")
            }
            FusedStepOp::Mul => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("{a} * {b}")
            }
            FusedStepOp::MaxElem => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("fmaxf({a}, {b})")
            }
            FusedStepOp::CmpLt => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("({a} < {b}) ? 1.0f : 0.0f")
            }
            FusedStepOp::Neg => {
                let a = resolve(&inputs[0]);
                format!("-{a}")
            }
            FusedStepOp::Exp => {
                let a = resolve(&inputs[0]);
                format!("expf({a})")
            }
            FusedStepOp::Log => {
                let a = resolve(&inputs[0]);
                format!("logf({a})")
            }
            FusedStepOp::Sin => {
                let a = resolve(&inputs[0]);
                format!("sinf({a})")
            }
            FusedStepOp::Sqrt => {
                let a = resolve(&inputs[0]);
                format!("sqrtf({a})")
            }
            FusedStepOp::Cos => {
                let a = resolve(&inputs[0]);
                format!("cosf({a})")
            }
            FusedStepOp::Tan => {
                let a = resolve(&inputs[0]);
                format!("tanf({a})")
            }
            FusedStepOp::Atan => {
                let a = resolve(&inputs[0]);
                format!("atanf({a})")
            }
            FusedStepOp::Abs => {
                let a = resolve(&inputs[0]);
                format!("fabsf({a})")
            }
            FusedStepOp::Floor => {
                let a = resolve(&inputs[0]);
                format!("floorf({a})")
            }
            FusedStepOp::Ceil => {
                let a = resolve(&inputs[0]);
                format!("ceilf({a})")
            }
        }
    }

    /// Emit one fused-step expression for the AVX2 + Sleef path.
    fn simd_step_expr(
        op: &FusedStepOp,
        resolve: &dyn Fn(&FusedInput) -> String,
        inputs: &[FusedInput],
    ) -> String {
        match op {
            FusedStepOp::Add => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("_mm256_add_ps({a}, {b})")
            }
            FusedStepOp::Mul => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("_mm256_mul_ps({a}, {b})")
            }
            FusedStepOp::MaxElem => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("_mm256_max_ps({a}, {b})")
            }
            FusedStepOp::CmpLt => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!(
                    "_mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(1.0f), _mm256_cmp_ps({a}, {b}, _CMP_LT_OS))"
                )
            }
            FusedStepOp::Neg => {
                let a = resolve(&inputs[0]);
                format!("_mm256_sub_ps(_mm256_setzero_ps(), {a})")
            }
            FusedStepOp::Exp => {
                let a = resolve(&inputs[0]);
                format!("CHELIS_EXPF8({a})")
            }
            FusedStepOp::Log => {
                let a = resolve(&inputs[0]);
                format!("CHELIS_LOGF8({a})")
            }
            FusedStepOp::Sin => {
                let a = resolve(&inputs[0]);
                format!("CHELIS_SINF8({a})")
            }
            FusedStepOp::Sqrt => {
                let a = resolve(&inputs[0]);
                format!("CHELIS_SQRTF8({a})")
            }
            // New ops: no SIMD intrinsic yet, emit scalar broadcast via set1
            FusedStepOp::Cos => {
                let a = resolve(&inputs[0]);
                format!("_mm256_set1_ps(cosf(_mm256_cvtss_f32({a})))")
            }
            FusedStepOp::Tan => {
                let a = resolve(&inputs[0]);
                format!("_mm256_set1_ps(tanf(_mm256_cvtss_f32({a})))")
            }
            FusedStepOp::Atan => {
                let a = resolve(&inputs[0]);
                format!("_mm256_set1_ps(atanf(_mm256_cvtss_f32({a})))")
            }
            FusedStepOp::Abs => {
                let a = resolve(&inputs[0]);
                format!("_mm256_set1_ps(fabsf(_mm256_cvtss_f32({a})))")
            }
            FusedStepOp::Floor => {
                let a = resolve(&inputs[0]);
                format!("_mm256_set1_ps(floorf(_mm256_cvtss_f32({a})))")
            }
            FusedStepOp::Ceil => {
                let a = resolve(&inputs[0]);
                format!("_mm256_set1_ps(ceilf(_mm256_cvtss_f32({a})))")
            }
        }
    }

    // ---- Fused elementwise ----
    fn emit_fused_elem(
        &mut self,
        id: usize,
        ops: &[FusedStep],
        inputs: &[NodeId],
        ty: &TensorType,
        in_place: Option<FusedInPlaceSpec>,
    ) {
        if let Some(spec) = in_place {
            self.emit_fused_in_place_wrapper(id, ty, spec);
        } else {
            self.emit_slot_wrapper(id, ty);
        }

        // Build contiguity guard for all external inputs.
        let contiguity_cond: String = if inputs.is_empty() {
            "1".to_string()
        } else {
            inputs
                .iter()
                .map(|n| format!("chelis_is_contiguous(t{})", n.0))
                .collect::<Vec<_>>()
                .join(" && ")
        };
        self.line(&format!("if ({contiguity_cond}) {{"));
        self.indent += 1;

        // Declare restrict pointers for each external input (used by all fast paths).
        if in_place.is_some() {
            self.line(&format!("float* __out_{id} = t{id}->data;"));
        } else {
            self.line(&format!("float* restrict __out_{id} = t{id}->data;"));
        }
        for (ext_idx, ext_node) in inputs.iter().enumerate() {
            let ext_id = ext_node.0;
            if in_place.is_some_and(|spec| spec.reusable_input == *ext_node) {
                self.line(&format!(
                    "const float* __ext{ext_idx}_{id} = t{ext_id}->data;"
                ));
            } else {
                self.line(&format!(
                    "const float* restrict __ext{ext_idx}_{id} = t{ext_id}->data;"
                ));
            }
        }

        let use_sleef = self.math_lib == crate::MathLib::Sleef && Self::has_math_ops(ops);

        // Closures for resolving fused inputs in scalar (fast-path) context.
        let resolve_fast = |fi: &FusedInput| -> String {
            match fi {
                FusedInput::External(i) => format!("__in_ext{i}"),
                FusedInput::PreviousStep(j) => format!("v{j}"),
            }
        };

        // Closure for resolving fused inputs in AVX2 SIMD context.
        let resolve_simd = |fi: &FusedInput| -> String {
            match fi {
                FusedInput::External(i) => format!("__v_ext{i}"),
                FusedInput::PreviousStep(j) => format!("__v{j}"),
            }
        };

        let last = ops.len() - 1;

        if use_sleef {
            // --- Sleef AVX2 path ---
            // The 8-wide body is guarded by #ifdef CHELIS_HAS_SLEEF so the
            // generated C compiles without Sleef installed; the #else branch
            // falls back to the Level-1 scalar loop.
            self.line("#ifdef CHELIS_HAS_SLEEF");
            self.line("{");
            self.indent += 1;
            self.line("int __i = 0;");
            // 8-wide main loop
            self.line(&format!("for (; __i + 8 <= t{id}->size; __i += 8) {{"));
            self.indent += 1;
            // Load 8 floats from each external input.
            for (ext_idx, _) in inputs.iter().enumerate() {
                self.line(&format!(
                    "__m256 __v_ext{ext_idx} = _mm256_loadu_ps(__ext{ext_idx}_{id} + __i);"
                ));
            }
            // Emit each step with SIMD intrinsics.
            for (s, step) in ops.iter().enumerate() {
                let expr = Self::simd_step_expr(&step.op, &resolve_simd, &step.input_indices);
                self.line(&format!("__m256 __v{s} = {expr};"));
            }
            self.line(&format!("_mm256_storeu_ps(__out_{id} + __i, __v{last});"));
            self.indent -= 1;
            self.line("}");
            // Scalar tail loop for remaining elements (n % 8).
            self.line(&format!("for (; __i < t{id}->size; __i++) {{"));
            self.indent += 1;
            for (ext_idx, _) in inputs.iter().enumerate() {
                self.line(&format!(
                    "float __in_ext{ext_idx} = __ext{ext_idx}_{id}[__i];"
                ));
            }
            for (s, step) in ops.iter().enumerate() {
                let expr = Self::scalar_step_expr(&step.op, &resolve_fast, &step.input_indices);
                self.line(&format!("float v{s} = {expr};"));
            }
            self.line(&format!("__out_{id}[__i] = v{last};"));
            self.indent -= 1;
            self.line("}");
            self.indent -= 1;
            self.line("}");
            self.line("#else");
            // Fallback: Level-1 scalar OMP SIMD loop.
            self.line("#pragma omp parallel for simd");
            self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
            self.indent += 1;
            for (ext_idx, _) in inputs.iter().enumerate() {
                self.line(&format!(
                    "float __in_ext{ext_idx} = __ext{ext_idx}_{id}[i];"
                ));
            }
            for (s, step) in ops.iter().enumerate() {
                let expr = Self::scalar_step_expr(&step.op, &resolve_fast, &step.input_indices);
                self.line(&format!("float v{s} = {expr};"));
            }
            self.line(&format!("__out_{id}[i] = v{last};"));
            self.indent -= 1;
            self.line("}");
            self.line("#endif");
        } else {
            // --- Level-1 scalar OMP SIMD loop (default fast path) ---
            self.line("#pragma omp parallel for simd");
            self.line(&format!("for (int i = 0; i < t{id}->size; i++) {{"));
            self.indent += 1;
            for (ext_idx, _) in inputs.iter().enumerate() {
                self.line(&format!(
                    "float __in_ext{ext_idx} = __ext{ext_idx}_{id}[i];"
                ));
            }
            for (s, step) in ops.iter().enumerate() {
                let expr = Self::scalar_step_expr(&step.op, &resolve_fast, &step.input_indices);
                self.line(&format!("float v{s} = {expr};"));
            }
            self.line(&format!("__out_{id}[i] = v{last};"));
            self.indent -= 1;
            self.line("}");
        }

        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;

        // Slow path: existing index-conversion loop (handles non-contiguous strides).
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

        // Emit each fused step using slow-path indexed access.
        let resolve_slow = |fi: &FusedInput| -> String {
            match fi {
                FusedInput::External(i) => {
                    let ext_id = inputs[*i].0;
                    format!("t{ext_id}->data[idx_ext{i}]")
                }
                FusedInput::PreviousStep(j) => format!("v{j}"),
            }
        };

        for (s, step) in ops.iter().enumerate() {
            let expr = Self::scalar_step_expr(&step.op, &resolve_slow, &step.input_indices);
            self.line(&format!("float v{s} = {expr};"));
        }

        // Store last step's result.
        self.line(&format!("t{id}->data[i] = v{last};"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- BLAS matmul ----
    fn emit_blas_matmul(&mut self, id: usize, spec: &MatmulEmitSpec, ty: &TensorType) {
        // Defense in depth: cblas_sgemm is F32-only. If a non-F32 BlasMatmul
        // reaches here it indicates a missing precision filter upstream (the
        // canonical filter is at chelis_ir::specialize::detect_matmul_pattern).
        // Refuse to emit rather than silently miscompile.
        assert_eq!(
            ty.precision,
            Prim::F32,
            "emit_blas_matmul received non-F32 output (precision={:?}) at node {id}; \
             cblas_sgemm is single-precision only. The upstream specializer in \
             chelis_ir::specialize must keep non-F32 matmul subgraphs on the \
             generic expand+mul+sum path.",
            ty.precision,
        );
        let a = spec.a.0;
        let b = spec.b.0;
        let m_expr = Self::emit_dim_expr(&spec.m);
        let n_expr = Self::emit_dim_expr(&spec.n);
        let k_expr = Self::emit_dim_expr(&spec.k);
        self.line(&format!("chelis_tensor *t{id}_a = t{a};"));
        self.line(&format!(
            "if (!(t{id}_a->ndim >= 2 && t{id}_a->strides[t{id}_a->ndim - 1] == 1 && t{id}_a->strides[t{id}_a->ndim - 2] == {k_expr})) {{"
        ));
        self.indent += 1;
        self.line(&format!("t{id}_a = chelis_contiguous(t{id}_a);"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_tensor *t{id}_b = t{b};"));
        self.line(&format!(
            "if (!(t{id}_b->ndim >= 2 && t{id}_b->strides[t{id}_b->ndim - 1] == 1 && t{id}_b->strides[t{id}_b->ndim - 2] == {n_expr})) {{"
        ));
        self.indent += 1;
        self.line(&format!("t{id}_b = chelis_contiguous(t{id}_b);"));
        self.indent -= 1;
        self.line("}");
        self.emit_slot_wrapper(id, ty);
        if spec.batch_dims.is_empty() {
            self.line(&format!(
                "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, {m_expr}, {n_expr}, {k_expr}, 1.0f, t{id}_a->data, {k_expr}, t{id}_b->data, {n_expr}, 0.0f, t{id}->data, {n_expr});"
            ));
        } else {
            let batch_count = spec
                .batch_dims
                .iter()
                .map(Self::emit_dim_expr)
                .reduce(|lhs, rhs| format!("({lhs} * {rhs})"))
                .unwrap_or_else(|| "1".to_string());
            self.line(&format!("int t{id}_batch_count = {batch_count};"));
            self.line(&format!(
                "for (int t{id}_batch = 0; t{id}_batch < t{id}_batch_count; t{id}_batch++) {{"
            ));
            self.indent += 1;
            self.line(&format!("int t{id}_rem = t{id}_batch;"));
            self.line(&format!("int t{id}_a_offset = 0;"));
            self.line(&format!("int t{id}_b_offset = 0;"));
            self.line(&format!("int t{id}_out_offset = 0;"));
            for axis in (0..spec.batch_dims.len()).rev() {
                let dim_expr = Self::emit_dim_expr(&spec.batch_dims[axis]);
                self.line(&format!(
                    "int t{id}_coord_{axis} = t{id}_rem % ({dim_expr});"
                ));
                self.line(&format!("t{id}_rem /= ({dim_expr});"));
                self.line(&format!(
                    "t{id}_a_offset += t{id}_coord_{axis} * t{id}_a->strides[{axis}];"
                ));
                self.line(&format!(
                    "t{id}_b_offset += t{id}_coord_{axis} * t{id}_b->strides[{axis}];"
                ));
                self.line(&format!(
                    "t{id}_out_offset += t{id}_coord_{axis} * t{id}->strides[{axis}];"
                ));
            }
            self.line(&format!(
                "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, {m_expr}, {n_expr}, {k_expr}, 1.0f, t{id}_a->data + t{id}_a_offset, {k_expr}, t{id}_b->data + t{id}_b_offset, {n_expr}, 0.0f, t{id}->data + t{id}_out_offset, {n_expr});"
            ));
            self.indent -= 1;
            self.line("}");
        }
        self.line(&format!("if (t{id}_a != t{a}) chelis_free(t{id}_a);"));
        self.line(&format!("if (t{id}_b != t{b}) chelis_free(t{id}_b);"));
    }

    fn dim_product_expr(dims: &[DimInfo]) -> String {
        dims.iter()
            .map(Self::emit_dim_info)
            .reduce(|lhs, rhs| format!("({lhs} * {rhs})"))
            .unwrap_or_else(|| "1".to_string())
    }

    fn emit_sparse_gather(
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
        let value_et = Self::elem_type(values_ty);
        let index_et = Self::elem_type(indices_ty);
        let before = Self::dim_product_expr(&values_ty.dims[..axis]);
        let axis_size = Self::emit_dim_info(&values_ty.dims[axis]);
        let after = Self::dim_product_expr(&values_ty.dims[axis + 1..]);
        self.line(&format!(
            "chelis_tensor *t{id}_values = chelis_contiguous(t{values});"
        ));
        self.line(&format!(
            "chelis_tensor *t{id}_indices = chelis_contiguous(t{indices});"
        ));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "const {value_et} *t{id}_values_data = (const {value_et}*)t{id}_values->data;"
        ));
        self.line(&format!(
            "const {index_et} *t{id}_indices_data = (const {index_et}*)t{id}_indices->data;"
        ));
        self.line(&format!(
            "{value_et} *t{id}_out_data = ({value_et}*)t{id}->data;"
        ));
        self.line(&format!("int t{id}_before = {before};"));
        self.line(&format!("int t{id}_axis_size = {axis_size};"));
        self.line(&format!("int t{id}_after = {after};"));
        self.line(&format!("int t{id}_index_count = t{id}_indices->size;"));
        self.line(&format!(
            "for (int t{id}_b = 0; t{id}_b < t{id}_before; t{id}_b++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "for (int t{id}_i = 0; t{id}_i < t{id}_index_count; t{id}_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int t{id}_g = (t{id}_indices->dtype == CHELIS_I64) ? (int)((const int64_t*)t{id}_indices->data)[t{id}_i] : (int)t{id}_indices->data[t{id}_i];"
        ));
        self.line(&format!(
            "if (t{id}_g < 0 || t{id}_g >= t{id}_axis_size) abort();"
        ));
        self.line(&format!(
            "for (int t{id}_d = 0; t{id}_d < t{id}_after; t{id}_d++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int t{id}_out = ((t{id}_b * t{id}_index_count + t{id}_i) * t{id}_after) + t{id}_d;"
        ));
        self.line(&format!(
            "int t{id}_src = ((t{id}_b * t{id}_axis_size + t{id}_g) * t{id}_after) + t{id}_d;"
        ));
        self.line(&format!(
            "t{id}_out_data[t{id}_out] = t{id}_values_data[t{id}_src];"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "if (t{id}_values != t{values}) chelis_free(t{id}_values);"
        ));
        self.line(&format!(
            "if (t{id}_indices != t{indices}) chelis_free(t{id}_indices);"
        ));
    }

    fn emit_sparse_scatter_add(
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
        let target_et = Self::elem_type(target_ty);
        let index_et = Self::elem_type(indices_ty);
        let update_et = Self::elem_type(updates_ty);
        let target_elem_size = Self::elem_size_expr(target_ty);
        let before = Self::dim_product_expr(&target_ty.dims[..axis]);
        let axis_size = Self::emit_dim_info(&target_ty.dims[axis]);
        let after = Self::dim_product_expr(&target_ty.dims[axis + 1..]);
        self.line(&format!(
            "chelis_tensor *t{id}_target = chelis_contiguous(t{target});"
        ));
        self.line(&format!(
            "chelis_tensor *t{id}_indices = chelis_contiguous(t{indices});"
        ));
        self.line(&format!(
            "chelis_tensor *t{id}_updates = chelis_contiguous(t{updates});"
        ));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "const {index_et} *t{id}_indices_data = (const {index_et}*)t{id}_indices->data;"
        ));
        self.line(&format!(
            "const {update_et} *t{id}_updates_data = (const {update_et}*)t{id}_updates->data;"
        ));
        self.line(&format!(
            "{target_et} *t{id}_out_data = ({target_et}*)t{id}->data;"
        ));
        self.line(&format!(
            "memcpy(t{id}->data, t{id}_target->data, (size_t)t{id}->size * {target_elem_size});"
        ));
        self.line(&format!("int t{id}_before = {before};"));
        self.line(&format!("int t{id}_axis_size = {axis_size};"));
        self.line(&format!("int t{id}_after = {after};"));
        self.line(&format!("int t{id}_index_count = t{id}_indices->size;"));
        self.line(&format!(
            "for (int t{id}_b = 0; t{id}_b < t{id}_before; t{id}_b++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "for (int t{id}_i = 0; t{id}_i < t{id}_index_count; t{id}_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int t{id}_g = (t{id}_indices->dtype == CHELIS_I64) ? (int)((const int64_t*)t{id}_indices->data)[t{id}_i] : (int)t{id}_indices->data[t{id}_i];"
        ));
        self.line(&format!(
            "if (t{id}_g < 0 || t{id}_g >= t{id}_axis_size) abort();"
        ));
        self.line(&format!(
            "for (int t{id}_d = 0; t{id}_d < t{id}_after; t{id}_d++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int t{id}_src = ((t{id}_b * t{id}_index_count + t{id}_i) * t{id}_after) + t{id}_d;"
        ));
        self.line(&format!(
            "int t{id}_out = ((t{id}_b * t{id}_axis_size + t{id}_g) * t{id}_after) + t{id}_d;"
        ));
        self.line(&format!(
            "t{id}_out_data[t{id}_out] += t{id}_updates_data[t{id}_src];"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "if (t{id}_target != t{target}) chelis_free(t{id}_target);"
        ));
        self.line(&format!(
            "if (t{id}_indices != t{indices}) chelis_free(t{id}_indices);"
        ));
        self.line(&format!(
            "if (t{id}_updates != t{updates}) chelis_free(t{id}_updates);"
        ));
    }

    /// Emit a bounded sparse replace-scatter (last-write-wins) loop.
    ///
    /// The deterministic order matches `spec/05-risc-primitives.md` §3.5:
    /// updates-tensor row-major (C order) flat iteration. We iterate
    /// `(b, i, d)` in the same nesting as `emit_sparse_scatter_add` —
    /// that nest order traverses `updates` flat-index ascending, so
    /// the **last write wins** invariant matches the IR evaluator
    /// (`scatter_replace` in `chelis_ir::eval`). The loop is single-
    /// threaded: no `#pragma omp parallel for`. Adding parallelism
    /// would race on duplicate indices and break determinism, which
    /// is the whole reason this op rejects AD.
    fn emit_sparse_scatter_replace(
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
        let target_et = Self::elem_type(target_ty);
        let index_et = Self::elem_type(indices_ty);
        let update_et = Self::elem_type(updates_ty);
        let target_elem_size = Self::elem_size_expr(target_ty);
        let before = Self::dim_product_expr(&target_ty.dims[..axis]);
        let axis_size = Self::emit_dim_info(&target_ty.dims[axis]);
        let after = Self::dim_product_expr(&target_ty.dims[axis + 1..]);
        self.line(&format!(
            "chelis_tensor *t{id}_target = chelis_contiguous(t{target});"
        ));
        self.line(&format!(
            "chelis_tensor *t{id}_indices = chelis_contiguous(t{indices});"
        ));
        self.line(&format!(
            "chelis_tensor *t{id}_updates = chelis_contiguous(t{updates});"
        ));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "const {index_et} *t{id}_indices_data = (const {index_et}*)t{id}_indices->data;"
        ));
        self.line(&format!(
            "const {update_et} *t{id}_updates_data = (const {update_et}*)t{id}_updates->data;"
        ));
        self.line(&format!(
            "{target_et} *t{id}_out_data = ({target_et}*)t{id}->data;"
        ));
        self.line(&format!(
            "memcpy(t{id}->data, t{id}_target->data, (size_t)t{id}->size * {target_elem_size});"
        ));
        self.line(&format!("int t{id}_before = {before};"));
        self.line(&format!("int t{id}_axis_size = {axis_size};"));
        self.line(&format!("int t{id}_after = {after};"));
        self.line(&format!("int t{id}_index_count = t{id}_indices->size;"));
        // Single-threaded sequential loop: deterministic last-write-wins
        // requires that no two writes to the same target cell race. The
        // outer (b, i, d) iteration order is the canonical
        // updates-tensor row-major traversal.
        self.line(&format!(
            "for (int t{id}_b = 0; t{id}_b < t{id}_before; t{id}_b++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "for (int t{id}_i = 0; t{id}_i < t{id}_index_count; t{id}_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int t{id}_g = (t{id}_indices->dtype == CHELIS_I64) ? (int)((const int64_t*)t{id}_indices->data)[t{id}_i] : (int)t{id}_indices->data[t{id}_i];"
        ));
        self.line(&format!(
            "if (t{id}_g < 0 || t{id}_g >= t{id}_axis_size) abort();"
        ));
        self.line(&format!(
            "for (int t{id}_d = 0; t{id}_d < t{id}_after; t{id}_d++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int t{id}_src = ((t{id}_b * t{id}_index_count + t{id}_i) * t{id}_after) + t{id}_d;"
        ));
        self.line(&format!(
            "int t{id}_out = ((t{id}_b * t{id}_axis_size + t{id}_g) * t{id}_after) + t{id}_d;"
        ));
        // Last-write-wins assignment (NOT accumulation).
        self.line(&format!(
            "t{id}_out_data[t{id}_out] = t{id}_updates_data[t{id}_src];"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "if (t{id}_target != t{target}) chelis_free(t{id}_target);"
        ));
        self.line(&format!(
            "if (t{id}_indices != t{indices}) chelis_free(t{id}_indices);"
        ));
        self.line(&format!(
            "if (t{id}_updates != t{updates}) chelis_free(t{id}_updates);"
        ));
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
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        self.emit_slot_wrapper(id, ty);
        let output_is_scalar = ty.dims.is_empty();
        if output_is_scalar {
            self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
            self.indent += 1;
            self.line(&format!(
                "t{id}->data[0] = chelis_sum_f32(t{a}->data, t{a}->size);"
            ));
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
        }
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
        self.line(&format!(
            "for (int __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line("int full_indices[CHELIS_MAX_DIM];");
        // Build full indices: insert k at the reduction axis
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}->ndim; d++) {{"));
        self.indent += 1;
        self.line(&format!("if (d == {axis}) {{"));
        self.indent += 1;
        self.line("full_indices[d] = __reduce_i;");
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
        if output_is_scalar {
            self.indent -= 1;
            self.line("}");
        }
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
        self.emit_slot_wrapper(id, ty);
        let output_is_scalar = ty.dims.is_empty();
        if output_is_scalar {
            self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
            self.indent += 1;
            self.line(&format!(
                "t{id}->data[0] = chelis_max_f32(t{a}->data, t{a}->size);"
            ));
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
        }
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
        self.line(&format!(
            "for (int __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line("int full_indices[CHELIS_MAX_DIM];");
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}->ndim; d++) {{"));
        self.indent += 1;
        self.line(&format!("if (d == {axis}) {{"));
        self.indent += 1;
        self.line("full_indices[d] = __reduce_i;");
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
        if output_is_scalar {
            self.indent -= 1;
            self.line("}");
        }
    }

    // ---- Generic scalar reduction (min / prod) ----
    //
    // `init` is the C literal for the accumulator's starting value, and
    // `update_tmpl` is the body of the inner loop with literal `{a}` tokens
    // for the input node id. Used by MinReduce and ProdReduce; the structure
    // mirrors `emit_reduce_max` exactly.
    //
    // `simd_fn` is an optional SIMD helper name (e.g. "chelis_min_f32") used
    // when the output is a scalar and the input is contiguous.
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
        simd_fn: Option<&str>,
    ) {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        self.emit_slot_wrapper(id, ty);
        let output_is_scalar = ty.dims.is_empty();
        let use_simd = output_is_scalar && simd_fn.is_some();
        if use_simd {
            let fn_name = simd_fn.unwrap();
            self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
            self.indent += 1;
            self.line(&format!(
                "t{id}->data[0] = {fn_name}(t{a}->data, t{a}->size);"
            ));
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
        }
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
        self.line(&format!(
            "for (int __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line("int full_indices[CHELIS_MAX_DIM];");
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}->ndim; d++) {{"));
        self.indent += 1;
        self.line(&format!("if (d == {axis}) {{"));
        self.indent += 1;
        self.line("full_indices[d] = __reduce_i;");
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
        if use_simd {
            self.indent -= 1;
            self.line("}");
        }
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
        let init = if is_argmax { "-INFINITY" } else { "INFINITY" };
        let cmp = if is_argmax { ">" } else { "<" };
        let simd_fn = if is_argmax {
            "chelis_argmax_f32"
        } else {
            "chelis_argmin_f32"
        };
        self.emit_slot_wrapper(id, ty);
        let output_is_scalar = ty.dims.is_empty();
        if output_is_scalar {
            self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
            self.indent += 1;
            self.line(&format!(
                "t{id}->data[0] = (float){simd_fn}(t{a}->data, t{a}->size);"
            ));
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
        }
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
        self.line(&format!(
            "for (int __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line("int full_indices[CHELIS_MAX_DIM];");
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}->ndim; d++) {{"));
        self.indent += 1;
        self.line(&format!("if (d == {axis}) {{"));
        self.indent += 1;
        self.line("full_indices[d] = __reduce_i;");
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
        self.line("best_idx = __reduce_i;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!("t{id}->data[outer] = (float)best_idx;"));
        self.indent -= 1;
        self.line("}");
        if output_is_scalar {
            self.indent -= 1;
            self.line("}");
        }
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

        self.emit_slot_wrapper(id, out_ty);
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
        self.line(&format!(
            "for (int __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line("int full_indices[CHELIS_MAX_DIM];");
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < {fused_ndim}; d++) {{"));
        self.indent += 1;
        self.line(&format!("if (d == {axis}) {{"));
        self.indent += 1;
        self.line("full_indices[d] = __reduce_i;");
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
                FusedStepOp::Cos => {
                    let a = resolve(&step.input_indices[0]);
                    format!("cosf({a})")
                }
                FusedStepOp::Tan => {
                    let a = resolve(&step.input_indices[0]);
                    format!("tanf({a})")
                }
                FusedStepOp::Atan => {
                    let a = resolve(&step.input_indices[0]);
                    format!("atanf({a})")
                }
                FusedStepOp::Abs => {
                    let a = resolve(&step.input_indices[0]);
                    format!("fabsf({a})")
                }
                FusedStepOp::Floor => {
                    let a = resolve(&step.input_indices[0]);
                    format!("floorf({a})")
                }
                FusedStepOp::Ceil => {
                    let a = resolve(&step.input_indices[0]);
                    format!("ceilf({a})")
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
        let et = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        match ty.precision {
            Prim::Int64 => {
                self.line(&format!(
                    "chelis_fill_i64(t{id}, (int64_t){});",
                    fill as i64
                ));
            }
            Prim::Int32 => {
                self.line(&format!(
                    "{{ int32_t *__p = (int32_t*)t{id}->data; for (int __i = 0; __i < t{id}->size; __i++) __p[__i] = (int32_t){}; }}",
                    fill as i32
                ));
            }
            _ => {
                self.line(&format!("chelis_fill_f32(t{id}, {:.8}f);", fill as f32));
            }
        }
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
        self.line(&format!(
            "(({et}*)t{id}->data)[dst_flat] = (({et}*)t{a}->data)[src_flat];"
        ));
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
        let et = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
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
        self.line(&format!(
            "(({et}*)t{id}->data)[i] = (({et}*)t{a}->data)[src_flat];"
        ));
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
        let et = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
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
        self.line(&format!(
            "(({et}*)t{id}->data)[i] = (({et}*)t{a}->data)[idx];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Cast ----
    //
    // `cast` is a precision conversion, not a bit-reinterpret. The previous
    // implementation issued a `memcpy(dst, src, n * sizeof(float))` and was
    // wrong on every cross-precision arm (f32<->f64, f32<->int32,
    // int32<->int64, ...). See
    // `docs/investigations/cbackend_cast_memcpy_diagnosis.md`. The host
    // runtime parallel was fixed in PR #59
    // (`crates/chelis-compiler-api/src/runtime.rs::cast_tensor_value` /
    // `convert_scalar_data`); this site mirrors those semantics in emitted
    // C.
    //
    // Validated precision set (see the validator around line 558):
    //   F32 | F64 | Int32 | Int64. `bool` and reduced floats are not valid
    // cast targets and panic before reaching this site.
    fn emit_cast(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType, dag: &Dag) {
        let a = inputs[0].0;
        let src_ty = &dag
            .get(inputs[0])
            .expect("cast input must resolve in dag")
            .output_type;
        let src_et = Self::elem_type(src_ty);
        let dst_et = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        // Strided element-wise loop. The output is freshly allocated and
        // contiguous, so the destination index is the flat loop index. The
        // source may be non-contiguous; resolve its element via the
        // standard `chelis_flat_to_indices` + `chelis_indices_to_flat`
        // dance used by `emit_realize` and friends. A C-level primitive
        // cast `(dst_et)src` performs the precision conversion -- this is
        // the canonical C semantics for f32<->f64 rounding,
        // float->int truncate-toward-zero, and int->float widening, and
        // matches the runtime evaluator's `convert_scalar_data` semantics
        // on the validated precision set.
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
        self.line(&format!(
            "(({dst_et}*)t{id}->data)[i] = ({dst_et})(({src_et}*)t{a}->data)[idx];"
        ));
        self.indent -= 1;
        self.line("}");
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

    fn tensor_ty(dims: &[usize], precision: Prim) -> TensorType {
        TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision,
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
        dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_alloc"));
        assert!(c.contains("chelis_fill_f32"));
        assert!(c.contains("3.0"));
    }

    #[test]
    fn add_emits_stride_aware_loop() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_flat_to_indices"));
        assert!(c.contains("chelis_indices_to_flat"));
        assert!(c.contains("+"));
    }

    #[test]
    fn neg_emits_unary_minus() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Neg, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        // The slow (non-contiguous) path emits a typed pointer cast then negates.
        assert!(c.contains("((float*)t0->data)[idx]"));
    }

    #[test]
    fn exp_emits_expf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Exp, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("expf("));
    }

    #[test]
    fn log_emits_logf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Log, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("logf("));
    }

    #[test]
    fn sin_emits_sinf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Sin, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("sinf("));
    }

    #[test]
    fn sqrt_emits_sqrtf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Sqrt, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("sqrtf("));
    }

    #[test]
    fn cmplt_emits_ternary_float() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::CmpLt, vec![a, b], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("1.0f : 0.0f"));
    }

    #[test]
    fn omp_pragma_in_elementwise() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("#pragma omp parallel for"));
    }

    #[test]
    fn sum_emits_reduction_loop() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4), None);
        dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![a],
            scalar_f32(),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("acc +="));
        assert!(c.contains("for (int __reduce_i"));
    }

    #[test]
    fn copy_materializes_and_drop_emits_no_wrapper() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let copy = dag.add_node(RiscOp::Copy, vec![x], vec_f32(4), None);
        dag.add_node(RiscOp::Drop, vec![x], vec_f32(4), None);
        dag.add_root(copy);

        let c = CEmitter::emit_dag(&dag, "test_copy_drop");

        assert!(c.contains("chelis_tensor *t1"));
        assert!(c.contains("((float*)t1->data)[i] = ((float*)t0->data)[idx];"));
        assert!(
            !c.contains("t2"),
            "Drop should not emit a tensor wrapper or compute statement:\n{c}"
        );
    }

    #[test]
    fn max_reduce_emits_fmaxf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4), None);
        dag.add_node(RiscOp::MaxReduce { axis: 0 }, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("fmaxf(acc"));
        assert!(c.contains("-INFINITY"));
    }

    #[test]
    fn mul_emits_star_op() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("*"));
    }

    #[test]
    fn max_elem_emits_fmaxf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::MaxElem, vec![a, b], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("fmaxf("));
    }

    #[test]
    fn reshape_emits_contiguous_check() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(6), None);
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            },
            vec![a],
            mat_f32(2, 3),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_is_contiguous"));
        assert!(c.contains("chelis_contiguous"));
    }

    #[test]
    fn permute_emits_stride_swap() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3), None);
        dag.add_node(
            RiscOp::Permute { axes: vec![1, 0] },
            vec![a],
            mat_f32(3, 2),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("strides[0] = t0->strides[1]"));
        assert!(c.contains("strides[1] = t0->strides[0]"));
    }

    #[test]
    fn expand_sets_stride_zero() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(1), None);
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            vec_f32(4),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("strides[0] = 0"));
    }

    #[test]
    fn store_is_alias() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![a],
            scalar_f32(),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_contiguous(t0); /* store: out */"));
    }

    #[test]
    fn cast_emits_elementwise_conversion_loop() {
        // CBackend-CastMemcpy fix: cast no longer emits a bit-preserving
        // `memcpy`. The new shape is a strided element-wise loop with a
        // C-level primitive cast `(dst_et)src` performing the conversion.
        // See `docs/investigations/cbackend_cast_memcpy_diagnosis.md`.
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let dst_ty = TensorType {
            dims: vec![],
            precision: Prim::F64,
        };
        dag.add_node(
            RiscOp::Cast {
                new_precision: Prim::F64,
            },
            vec![a],
            dst_ty,
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(
            c.contains("chelis_flat_to_indices"),
            "cast must emit a strided element-wise loop, not memcpy; got:\n{c}"
        );
        assert!(
            c.contains("(double)((float*)"),
            "cast must emit a C-level primitive cast (dst_et)(src_et*)src; got:\n{c}"
        );
        assert!(
            !c.contains("memcpy(t1->data, t0->data"),
            "cast must not emit the legacy bit-preserving memcpy; got:\n{c}"
        );
    }

    #[test]
    fn realize_emits_materialization_loop() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(6), None);
        let s = dag.add_node(
            RiscOp::Stride { strides: vec![2] },
            vec![x],
            vec_f32(3),
            None,
        );
        dag.add_node(RiscOp::Realize, vec![s], vec_f32(3), None);

        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_tensor *chelis_slot0 = chelis_alloc("));
        assert!(c.contains("chelis_tensor *t2 = chelis_alloc_view("));
        assert!(c.contains("chelis_indices_to_flat(indices, t1->strides, t1->ndim)"));
        assert!(!c.contains("chelis_alloc_view(1, (int[]){ 3 }, CHELIS_F32, t1->data)"));
    }

    #[test]
    fn load_emits_input_reference() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("inputs[0]"));
    }

    #[test]
    fn repeated_load_names_share_one_input_slot() {
        let mut dag = Dag::new();
        let x0 = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let x1 = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Add, vec![x0, x1], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("if (n_in != 1)"));
        assert!(c.contains("chelis_tensor *t0 = inputs[0];"));
        assert!(c.contains("chelis_tensor *t1 = inputs[0];"));
    }

    #[test]
    fn input_labels_follow_first_load_occurrence() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            scalar_f32(),
            None,
        );
        assert_eq!(CEmitter::input_labels(&dag), vec!["b", "a"]);
    }

    #[test]
    fn function_signature_correct() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "my_func");
        assert!(c.contains(
            "void my_func(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out)"
        ));
    }

    #[test]
    fn includes_runtime_header() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("#include \"chelis_runtime.h\""));
    }

    #[test]
    fn pad_emits_fill_and_copy() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(3), None);
        dag.add_node(
            RiscOp::Pad {
                padding: vec![(1, 1)],
                fill: 0.0,
            },
            vec![a],
            vec_f32(5),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_fill_f32"));
        assert!(c.contains("dst_indices[0] = src_indices[0] + 1"));
    }

    #[test]
    fn shrink_emits_offset_copy() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(5), None);
        dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(1, 4)],
            },
            vec![a],
            vec_f32(3),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("src_indices[0] = dst_indices[0] + 1"));
    }

    #[test]
    fn stride_emits_stride_multiply() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4), None);
        dag.add_node(
            RiscOp::Stride { strides: vec![2] },
            vec![a],
            vec_f32(2),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("strides[0] = t0->strides[0] * 2"));
    }

    #[test]
    fn add_then_mul_chains() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let d = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Mul, vec![c, d], scalar_f32(), None);
        let code = CEmitter::emit_dag(&dag, "test_fn");
        // t2 is add result, t4 is mul result
        assert!(code.contains("t2->data"));
        assert!(code.contains("t4->data"));
    }

    #[test]
    fn vector_add_uses_correct_shape() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4), None);
        dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("(int[]){ 4 }"));
    }

    #[test]
    fn sum_then_neg_chains() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(3), None);
        let s = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![a],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Neg, vec![s], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("acc +="));
        // The slow (non-contiguous) path emits a typed pointer cast then negates.
        assert!(c.contains("((float*)t1->data)[idx]"));
    }

    #[test]
    fn load_is_borrowed_not_freed() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(!c.contains("chelis_free(t0);"));
        assert!(c.contains("outputs[0] = chelis_contiguous(t0);"));
    }

    #[test]
    fn roots_are_emitted_as_multiple_outputs() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
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
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_node(
            RiscOp::CmpLt,
            vec![a, b],
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("chelis_alloc(0, NULL, CHELIS_BOOL);"));
    }

    #[test]
    fn int64_const_does_not_panic() {
        // Regression: dtype_macro used to panic for int64 tensors.
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Const { value: 42.0 },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Int64,
            },
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(
            c.contains("CHELIS_I64"),
            "generated C must use CHELIS_I64 dtype macro"
        );
        assert!(
            c.contains("chelis_fill_i64"),
            "generated C must call chelis_fill_i64 for int64 const"
        );
    }

    #[test]
    fn f64_const_uses_chelis_fill_f64_and_dtype_macro() {
        // v0.2.3: f64 tensors are a first-class precision. The const path must
        // emit CHELIS_F64 and chelis_fill_f64 (not chelis_fill_f32, which would
        // silently downcast).
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Const { value: 1.5 },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F64,
            },
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(
            c.contains("CHELIS_F64"),
            "generated C must use CHELIS_F64 dtype macro:\n{c}"
        );
        assert!(
            c.contains("chelis_fill_f64"),
            "generated C must call chelis_fill_f64 for f64 const:\n{c}"
        );
        assert!(
            !c.contains("chelis_fill_f32(t0"),
            "generated C must not fall back to chelis_fill_f32 on an f64 tensor:\n{c}"
        );
    }

    #[test]
    fn f64_tensor_add_uses_double_pointer_cast() {
        // Regression: binary elementwise over f64 tensors must emit double*
        // typed pointer casts so gcc reads 8 bytes per element.
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F64,
        };
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], ty.clone(), None);
        dag.add_node(RiscOp::Add, vec![a, b], ty, None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("CHELIS_F64"), "generated C must use CHELIS_F64");
        assert!(
            c.contains("double"),
            "generated C must use double typed pointer casts:\n{c}"
        );
    }

    #[test]
    fn f64_tensor_exp_emits_exp_not_expf() {
        // exp(tensor[n, f64]) must route to the double-precision math symbol.
        // Emitting expf() would silently truncate to float and lose precision.
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F64,
        };
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty.clone(), None);
        dag.add_node(RiscOp::Exp, vec![a], ty, None);
        let c = CEmitter::emit_dag_with_options(
            &dag,
            "test_fn",
            crate::CodegenOptions {
                math_lib_override: Some(crate::MathLib::None),
                ..crate::CodegenOptions::default()
            },
        );
        assert!(
            c.contains("exp("),
            "f64 exp must emit exp(, not expf(:\n{c}"
        );
        assert!(
            !c.contains("expf("),
            "f64 exp must not emit the f32 expf( symbol:\n{c}"
        );
    }

    #[test]
    fn int64_add_does_not_panic() {
        // Regression: binary elementwise over int64 tensors used to panic.
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int64,
        };
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], ty.clone(), None);
        dag.add_node(RiscOp::Add, vec![a, b], ty, None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("CHELIS_I64"), "generated C must use CHELIS_I64");
        assert!(
            c.contains("int64_t"),
            "generated C must use int64_t typed pointer casts"
        );
    }

    #[test]
    fn matmul_pattern_emits_cblas_call() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3), None);
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4), None);
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
            None,
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
            None,
        );
        let mul = dag.add_node(
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat_f32(2, 4),
            None,
        );
        let result = crate::codegen_with_options(
            &dag,
            "test_fn",
            crate::CodegenOptions {
                use_blas: true,
                ..crate::CodegenOptions::default()
            },
        );
        assert!(result.c_source.contains("cblas_sgemm("));
    }

    #[test]
    fn default_codegen_uses_generic_matmul_path() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3), None);
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4), None);
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
            None,
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
            None,
        );
        let mul = dag.add_node(
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat_f32(2, 4),
            None,
        );
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(!c.contains("cblas_sgemm("));
        assert!(c.contains("for (int __reduce_i = 0; __reduce_i < 3; __reduce_i++) {"));
    }

    #[test]
    #[should_panic(expected = "C backend does not yet support bf16 tensors")]
    fn unsupported_precision_panics() {
        // v0.2.3: f64 is now supported, so this regression uses bf16 as
        // the unsupported-precision fixture. The backend must still panic
        // on any other reduced-precision float.
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Const { value: 1.0 },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Bf16,
            },
            None,
        );
        let _ = CEmitter::emit_dag(&dag, "test_fn");
    }

    // ---- SIMD Level 1b fast-path tests ----

    /// Binary add for two contiguous-capable inputs must emit the fast path
    /// containing restrict pointers, #pragma omp parallel for simd, and chelis_is_contiguous.
    #[test]
    fn binary_add_fast_path_emits_restrict_and_simd() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4), None);
        dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(
            c.contains("restrict"),
            "fast path should declare restrict pointers"
        );
        assert!(
            c.contains("#pragma omp parallel for simd"),
            "fast path should emit #pragma omp parallel for simd"
        );
        assert!(
            c.contains("chelis_is_contiguous"),
            "fast path should be guarded by chelis_is_contiguous"
        );
    }

    /// When inputs have non-unit strides (via Expand with stride 0),
    /// chelis_is_contiguous returns false at runtime, so the emitted C must
    /// include the slow-path chelis_flat_to_indices fallback code.
    #[test]
    fn binary_add_with_expanded_input_includes_slow_path() {
        let mut dag = Dag::new();
        // Build a broadcast-style input: Const [1] expanded to [4] via stride=0
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(1), None);
        let a_exp = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4), None);
        dag.add_node(RiscOp::Add, vec![a_exp, b], vec_f32(4), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        // The slow path (index-conversion fallback) must always be present in the
        // emitted C; at runtime, chelis_is_contiguous(t_expanded) == 0 directs
        // execution into this branch.
        assert!(
            c.contains("chelis_flat_to_indices"),
            "slow path must contain chelis_flat_to_indices for non-contiguous inputs"
        );
        // The fast-path guard is still emitted (as code text), but contains the
        // contiguity check, which will be false at runtime for the expanded input.
        assert!(
            c.contains("chelis_is_contiguous"),
            "contiguity guard must still appear in the emitted code"
        );
    }

    /// Unary neg fast path must emit restrict pointers and #pragma omp parallel for simd.
    #[test]
    fn unary_neg_fast_path_emits_restrict_and_simd() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], vec_f32(4), None);
        dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("restrict"), "unary fast path must use restrict");
        assert!(
            c.contains("#pragma omp parallel for simd"),
            "unary fast path must have #pragma omp parallel for simd"
        );
        assert!(
            c.contains("chelis_is_contiguous"),
            "unary fast path must be guarded by chelis_is_contiguous"
        );
    }

    /// Fused elementwise fast path must emit restrict pointers and #pragma omp parallel for simd.
    #[test]
    fn fused_elem_fast_path_emits_restrict_and_simd() {
        use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4), None);
        // Fused: add(a, b)
        let ops = vec![FusedStep {
            op: FusedStepOp::Add,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        dag.add_node(RiscOp::FusedElem { ops }, vec![a, b], vec_f32(4), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(
            c.contains("restrict"),
            "fused fast path must use restrict pointers"
        );
        assert!(
            c.contains("#pragma omp parallel for simd"),
            "fused fast path must have #pragma omp parallel for simd"
        );
        assert!(
            c.contains("chelis_is_contiguous"),
            "fused fast path must be guarded by chelis_is_contiguous"
        );
        // Slow path must also be present as a fallback
        assert!(
            c.contains("chelis_flat_to_indices"),
            "fused slow path must still be present"
        );
    }

    #[test]
    fn fused_elem_without_reusable_input_keeps_non_in_place_restrict_shape() {
        use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let scale = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4), None);
        let ops = vec![FusedStep {
            op: FusedStepOp::Mul,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        dag.add_node(RiscOp::FusedElem { ops }, vec![x, scale], vec_f32(4), None);

        let c = CEmitter::emit_dag(&dag, "test_fn");

        assert!(c.contains("float* restrict __out_2 = t2->data;"));
        assert!(c.contains("const float* restrict __ext0_2 = t0->data;"));
        assert!(c.contains("const float* restrict __ext1_2 = t1->data;"));
        assert!(
            !c.contains("chelis_alloc_view(1, (int[]){ 4 }, CHELIS_F32, t0->data);"),
            "C fused codegen must not claim in-place aliasing without reusable_input"
        );
    }

    #[test]
    fn sparse_gather_uses_typed_indices_and_payload_pointers() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Const { value: 0.0 },
            vec![],
            tensor_ty(&[3], Prim::Int32),
            None,
        );
        dag.add_node(
            RiscOp::Gather { axis: 0 },
            vec![values, indices],
            tensor_ty(&[3, 2], Prim::F64),
            None,
        );

        let c = CEmitter::emit_dag(&dag, "test_fn");

        assert!(c.contains("const double *t2_values_data = (const double*)t2_values->data;"));
        assert!(c.contains("const int32_t *t2_indices_data = (const int32_t*)t2_indices->data;"));
        assert!(c.contains("double *t2_out_data = (double*)t2->data;"));
        assert!(c.contains("t2_indices->dtype == CHELIS_I64"));
        assert!(c.contains("t2_out_data[t2_out] = t2_values_data[t2_src];"));
    }

    #[test]
    fn sparse_gather_embedding_probe_does_not_allocate_dense_one_hot_product() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[50000, 1024], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            tensor_ty(&[128], Prim::Int32),
            None,
        );
        dag.add_node(
            RiscOp::Gather { axis: 0 },
            vec![values, indices],
            tensor_ty(&[128, 1024], Prim::F32),
            None,
        );

        let c = CEmitter::emit_dag(&dag, "embedding_probe");

        assert!(c.contains("chelis_alloc(2, (int[]){ 128, 1024 }, CHELIS_F32);"));
        assert!(
            !c.contains("(int[]){ 128, 50000, 1024 }"),
            "sparse gather codegen must not allocate the dense [N,V,D] one-hot/product tensor"
        );
        assert!(
            !c.contains("128 * 50000 * 1024"),
            "sparse gather codegen must not compute dense embedding volume"
        );
        assert!(c.contains("t2_indices->dtype == CHELIS_I64"));
    }

    #[test]
    #[should_panic(expected = "C backend sparse gather requires int32/int64 indices")]
    fn sparse_gather_rejects_float_indices_at_emit_boundary() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], vec_f32(3), None);
        dag.add_node(
            RiscOp::Gather { axis: 0 },
            vec![values, indices],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );

        let _ = CEmitter::emit_dag(&dag, "test_fn");
    }

    #[test]
    fn sparse_scatter_add_uses_typed_indices_payload_and_copy_size() {
        let mut dag = Dag::new();
        let target = dag.add_node(
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Const { value: 0.0 },
            vec![],
            tensor_ty(&[3], Prim::Int64),
            None,
        );
        let updates = dag.add_node(
            RiscOp::Const { value: 1.0 },
            vec![],
            tensor_ty(&[3, 2], Prim::F64),
            None,
        );
        dag.add_node(
            RiscOp::ScatterAdd { axis: 0 },
            vec![target, indices, updates],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );

        let c = CEmitter::emit_dag(&dag, "test_fn");

        assert!(c.contains("const int64_t *t3_indices_data = (const int64_t*)t3_indices->data;"));
        assert!(c.contains("const double *t3_updates_data = (const double*)t3_updates->data;"));
        assert!(c.contains("double *t3_out_data = (double*)t3->data;"));
        assert!(
            c.contains("memcpy(t3->data, t3_target->data, (size_t)t3->size * sizeof(double));")
        );
        assert!(c.contains("t3_out_data[t3_out] += t3_updates_data[t3_src];"));
    }

    #[test]
    #[should_panic(expected = "C backend sparse scatter_add requires int32/int64 indices")]
    fn sparse_scatter_add_rejects_float_indices_at_emit_boundary() {
        let mut dag = Dag::new();
        let target = dag.add_node(
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], vec_f32(3), None);
        let updates = dag.add_node(
            RiscOp::Const { value: 1.0 },
            vec![],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );
        dag.add_node(
            RiscOp::ScatterAdd { axis: 0 },
            vec![target, indices, updates],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );

        let _ = CEmitter::emit_dag(&dag, "test_fn");
    }

    #[test]
    fn target_fused_in_place_restrict_shape_aliases_only_reusable_input() {
        use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let scale = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4), None);
        let ops = vec![FusedStep {
            op: FusedStepOp::Mul,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        let fused = dag.add_node(RiscOp::FusedElem { ops }, vec![x, scale], vec_f32(4), None);
        dag.set_reusable_input(fused, x);

        let c = CEmitter::emit_dag(&dag, "test_fn");

        assert!(c.contains("chelis_alloc_view(1, (int[]){ 4 }, CHELIS_F32, t0->data);"));
        assert!(!c.contains("float* restrict __out_2 = t2->data;"));
        assert!(!c.contains("const float* restrict __ext0_2 = t0->data;"));
        assert!(c.contains("float* __out_2 = t2->data;"));
        assert!(c.contains("const float* __ext0_2 = t0->data;"));
        assert!(c.contains("const float* restrict __ext1_2 = t1->data;"));
    }

    #[test]
    fn fused_reusable_input_with_multiple_consumers_does_not_alias() {
        use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let scale = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4), None);
        let ops = vec![FusedStep {
            op: FusedStepOp::Mul,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        let fused = dag.add_node(RiscOp::FusedElem { ops }, vec![x, scale], vec_f32(4), None);
        dag.set_reusable_input(fused, x);
        let other = dag.add_node(RiscOp::Neg, vec![x], vec_f32(4), None);
        dag.add_root(fused);
        dag.add_root(other);

        let c = CEmitter::emit_dag(&dag, "test_fn");

        assert!(
            !c.contains("chelis_alloc_view(1, (int[]){ 4 }, CHELIS_F32, t0->data);"),
            "multi-consumer reusable input must not be aliased in place"
        );
        assert!(c.contains("float* restrict __out_2 = t2->data;"));
        assert!(c.contains("const float* restrict __ext0_2 = t0->data;"));
        assert!(c.contains("const float* restrict __ext1_2 = t1->data;"));
    }

    // ---- New scalar builtin C emission tests ----

    #[test]
    fn cos_emits_cosf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Cos, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("cosf("), "expected cosf( in:\n{c}");
    }

    #[test]
    fn tan_emits_tanf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Tan, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("tanf("), "expected tanf( in:\n{c}");
    }

    #[test]
    fn atan_emits_atanf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Atan, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("atanf("), "expected atanf( in:\n{c}");
    }

    #[test]
    fn abs_emits_fabsf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: -2.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Abs, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("fabsf("), "expected fabsf( in:\n{c}");
    }

    #[test]
    fn floor_emits_floorf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.7 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Floor, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("floorf("), "expected floorf( in:\n{c}");
    }

    #[test]
    fn ceil_emits_ceilf() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.3 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Ceil, vec![a], scalar_f32(), None);
        let c = CEmitter::emit_dag(&dag, "test_fn");
        assert!(c.contains("ceilf("), "expected ceilf( in:\n{c}");
    }

    // ---- Numerical correctness: verify via constant folding in the evaluator ----
    // These tests confirm that the Rust-side evaluator and the IR pipeline agree
    // on the mathematical values. The C emission tests above cover the symbol name.

    #[test]
    fn cos_numerical_correctness() {
        // cos(π/3) ≈ 0.5 (within 1e-4)
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Const {
                value: std::f64::consts::PI / 3.0,
            },
            vec![],
            scalar_f32(),
            None,
        );
        let out = dag.add_node(RiscOp::Cos, vec![x], scalar_f32(), None);
        dag.add_root(out);
        let results = chelis_ir::eval::eval_scalar(&dag, &std::collections::HashMap::new());
        let val = results[&out] as f32;
        assert!(
            (val - 0.5_f32).abs() < 1e-4,
            "cos(π/3) should be ≈ 0.5, got {val}"
        );
    }

    #[test]
    fn abs_numerical_correctness() {
        // abs(-2.0) == 2.0
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: -2.0 }, vec![], scalar_f32(), None);
        let out = dag.add_node(RiscOp::Abs, vec![x], scalar_f32(), None);
        dag.add_root(out);
        let results = chelis_ir::eval::eval_scalar(&dag, &std::collections::HashMap::new());
        let val = results[&out] as f32;
        assert!(
            (val - 2.0_f32).abs() < 1e-4,
            "abs(-2.0) should be 2.0, got {val}"
        );
    }
}
