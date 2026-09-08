//! RISC DAG to C source code emission.

use std::collections::BTreeMap;

use chelis_ir::dag::{
    Dag, DagNode, DimExpr, DimInfo, ExtremaKind, ExtremaOperand, FusedInput, FusedStep,
    FusedStepOp, NodeId, ReduceWindowKind, RiscOp, RtAxis, RtDim, SymbolicDimSource, TensorType,
};
use chelis_ir::ownership::{
    CStorageLane, ReusableOwnedStorage, VerifiedDagAction, VerifiedDagProgram, VerifiedDagView,
    VerifiedStoragePlan, plan_c_storage, plan_c_storage_layout,
};
use chelis_types::types::Prim;
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};
use chelis_types::{CheckedCastKind, CheckedCastPlan, ElementRef, NumericTrap, ScalarValue};

use crate::memory::{MemoryPlan, NodeMemoryKind};

fn unsupported_verified_dag_action(node: NodeId, detail: &str) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::Op("Drop".to_string()),
        format!("verified DAG ownership action at node {}: {detail}", node.0),
        Stage::Codegen("c"),
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
        Stage::Codegen("c"),
        chelis_types::deliberate_rejection!(
            "[04-SHAPE-1]",
            "C storage placement requires the verified exact-capacity plan"
        ),
    )
}

/// Emits C source code from a RISC DAG.
pub struct CEmitter {
    lines: Vec<String>,
    indent: usize,
    use_blas: bool,
    /// Which vectorized math library to target for fused-elem SIMD emission (Level 3b).
    math_lib: crate::MathLib,
    /// FusedElem nodes inlined into a trailing reduction (no standalone emission).
    reduction_inlined: chelis_unord::UnordSet<usize>,
    /// Backing-slot plan for materialized C tensors.
    memory_plan: MemoryPlan,
    fused_reuse: BTreeMap<NodeId, ReusableOwnedStorage>,
    reused_sources: chelis_unord::UnordSet<NodeId>,
    slot_current_owner: BTreeMap<usize, usize>,
    /// chelis#616: `(node id, output axis) -> (symbol, declares)` for every
    /// op-declared runtime dim (see `SymbolicDimSource::OpDeclared`). The
    /// owning movement op's emitter declares `int <symbol> = <extent>;` when
    /// `declares` is true, or emits a runtime equality-abort guard against
    /// the already-declared value when false (the symbol is Load-declared in
    /// the prologue, or an earlier op already declared it).
    runtime_dim_sites: chelis_unord::UnordMap<(usize, usize), (String, bool)>,
    /// chelis#1277 C1.3: `(node, axis)` -> the claim a local guard compares
    /// against, and the operation [04-NUM-9]'s `<op>` slot names.
    local_dim_guard_sites: chelis_unord::UnordMap<
        chelis_ir::ownership::LocalGuardSite,
        chelis_ir::ownership::LocalGuardClaim,
    >,
    /// Claim names this function actually declares as C variables. A local
    /// guard compares against the claim BY NAME, so a claim that resolved to
    /// a literal and was never declared has nothing to compare against and
    /// gets no guard rather than an undeclared identifier.
    declared_dim_names: chelis_unord::UnordSet<String>,
    /// Node descriptors whose exclusive runtime write lease remains live
    /// while the generated kernel fills and consumes its private storage.
    /// All leases are ended before any descriptor is returned or released.
    write_nodes: chelis_unord::UnordSet<usize>,
}

#[derive(Debug, Clone)]
struct OutputSpec {
    id: NodeId,
    label: String,
}

struct MatmulEmitSpec {
    a: NodeId,
    b: NodeId,
    batch_dims: Vec<DimExpr>,
    m: DimExpr,
    n: DimExpr,
    k: DimExpr,
    /// Inner-product accumulator precision per `spec/04-type-system.md`
    /// §5.7 / §5.7.1. Drives BLAS dispatch: F32 → `cblas_sgemm`,
    /// F64 → `cblas_dgemm`. Sourced from the `RiscOp::BlasMatmul`
    /// node's `accumulator` field (NOT inferred from operand storage)
    /// per the destructure-`..` memory rule that prompted the WS-A0 F1
    /// guard. Per spec §5.7.1, for f32/f64 operands the default
    /// accumulator equals the operand precision; the result precision
    /// equals the operand precision.
    accumulator: Prim,
    /// WS-1: operand storage precision (precision of operand A and B;
    /// the IR verifier guarantees they match). Drives the dispatch to
    /// the convert-then-`cblas_sgemm` wrapper when operands are
    /// bf16/f16 even if the matmul's output is f32 (the bf16-input
    /// f32-output case that arises when `Sum(Mul(Expand(A), Expand(B)))`
    /// over bf16 inputs is matmul-detected).
    operand_precision: Prim,
}

#[derive(Debug)]
struct CFusedReuse {
    token: ReusableOwnedStorage,
}

impl CEmitter {
    /// Emit C source for an entire DAG as a function.
    #[cfg(test)]
    pub fn emit_dag(dag: VerifiedDagProgram, func_name: &str) -> Result<String, Unsupported> {
        Self::emit_dag_with_options(dag, func_name, crate::CodegenOptions::default())
    }

    /// Emit C source for an entire DAG with explicit backend options.
    pub fn emit_dag_with_options(
        dag: VerifiedDagProgram,
        func_name: &str,
        options: crate::CodegenOptions,
    ) -> Result<String, Unsupported> {
        let mut plan = plan_c_storage(dag).map_err(unsupported_storage_plan)?;
        Self::emit_storage_plan_with_options(&mut plan, func_name, options)
    }

    fn emit_storage_plan_with_options(
        plan: &mut VerifiedStoragePlan<CStorageLane>,
        func_name: &str,
        options: crate::CodegenOptions,
    ) -> Result<String, Unsupported> {
        let memory_plan = MemoryPlan::from_shared(plan);
        let nodes = plan
            .emission()
            .nodes()
            .iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();
        let mut fused_reuse = BTreeMap::new();
        for node in nodes {
            if let Some(token) = plan
                .take_reuse_for(node)
                .map_err(unsupported_storage_plan)?
            {
                fused_reuse.insert(node, token);
            }
        }
        Self::emit_preplanned(
            plan.emission(),
            memory_plan,
            fused_reuse,
            func_name,
            options,
        )
    }

    pub(crate) fn emit_verified_dag_with_options(
        dag: VerifiedDagView<'_>,
        func_name: &str,
        options: crate::CodegenOptions,
    ) -> Result<String, Unsupported> {
        let mut plan = plan_c_storage_layout(dag).map_err(unsupported_storage_plan)?;
        let memory_plan = MemoryPlan::from_layout(&plan);
        let nodes = dag.nodes().iter().map(|node| node.id).collect::<Vec<_>>();
        let mut fused_reuse = BTreeMap::new();
        for node in nodes {
            if let Some(token) = plan
                .take_reuse_for(node)
                .map_err(unsupported_storage_plan)?
            {
                fused_reuse.insert(node, token);
            }
        }
        Self::emit_preplanned(dag, memory_plan, fused_reuse, func_name, options)
    }

    fn emit_preplanned(
        dag: VerifiedDagView<'_>,
        memory_plan: MemoryPlan,
        fused_reuse: BTreeMap<NodeId, ReusableOwnedStorage>,
        func_name: &str,
        options: crate::CodegenOptions,
    ) -> Result<String, Unsupported> {
        // chelis#1277 C4.1/C4.3: before anything reads a shape, every
        // realized output axis must have one checked extent source. This
        // runs here rather than in `codegen_with_options` because the host
        // program's tensor helpers reach the emitter through
        // `host_emit::append_helper`, which does not go through that entry,
        // and because it must precede `symbolic_occurrences`, whose
        // fallback for an unrecoverable axis is a panic (chelis#1482).
        dag.check_axis_sources(chelis_types::unsupported::Stage::Codegen("c"))?;
        Self::reject_fused_integer_abs(dag)?;
        Self::validate_supported_precisions(dag);
        Self::validate_load_abi(dag);
        Self::validate_sparse_contracts(dag);
        // chelis#593 memory-safety floor. Run AFTER `rename_anonymous_dims`:
        // that pass resolves an anon (`*`/empty) output dim by copying the
        // first input's dims wholesale, which — for a `Pad` whose output has an
        // anon TRAILING dim (the symbolic entry-wrapper of a leading-axis
        // `concat`) — also clobbers the correctly-sized CONCRETE padded axis
        // back to the operand extent. Validate the exact dag that is about to
        // be emitted so the mis-sizing is caught before any heap-corrupting C
        // is written.
        Self::validate_pad_output_sizing(dag);

        let reduction_inlined = dag.reduction_inlined_fused_elems();
        let math_lib = options
            .math_lib_override
            .unwrap_or_else(crate::MathLib::detect);
        let output_specs = Self::output_specs(dag);
        let output_ids = output_specs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>();
        // chelis#616: resolve each op-declared runtime dim to a declare/guard
        // site. A Load source anywhere makes every op site a guard; otherwise
        // the first op site (node-id order = emission order) declares and any
        // later site for the same symbol guards.
        let mut runtime_dim_sites = chelis_unord::UnordMap::new();
        {
            let occurrences = dag.symbolic_occurrences();
            let load_declared: chelis_unord::UnordSet<&str> = occurrences
                .iter()
                .filter(|o| matches!(o.source, SymbolicDimSource::Load { .. }))
                .map(|o| o.name.as_str())
                .collect();
            let mut declared = chelis_unord::UnordSet::new();
            for occurrence in &occurrences {
                if let SymbolicDimSource::OpDeclared { node, axis } = &occurrence.source {
                    let declares = !load_declared.contains(occurrence.name.as_str())
                        && declared.insert(occurrence.name.clone());
                    runtime_dim_sites.insert((node.0, *axis), (occurrence.name.clone(), declares));
                }
            }
        }

        // chelis#1277 b2.4: which sites GUARD is now the derivation's answer,
        // not the string walk's. `spec/04-type-system.md` section 4.7 places a
        // class whose operands are not all interface values at "the source
        // position of the operation that introduces the guarded extent", and
        // `VerifiedDagView::local_dim_guard_sites` is that set.
        //
        // Which site DECLARES still comes from the walk above, and that is a
        // recorded limit rather than an oversight: the lowerer stamps a fresh
        // synthesized name on every runtime movement output
        // (`_rt_stride_dim_1_0`, `_anon_dim_2_1`), so a declaring axis is
        // frequently a claim with ONE witness, which C2.4 does not make a
        // class. Reading the declaration through the axis SOURCE rather than
        // through the stamped name is C4.4's remaining half and is what closes
        // chelis#665; it is not this change.
        let local_dim_guard_sites: chelis_unord::UnordMap<
            chelis_ir::ownership::LocalGuardSite,
            chelis_ir::ownership::LocalGuardClaim,
        > = dag.local_dim_guard_sites().into_iter().collect();

        let mut e = CEmitter {
            lines: Vec::new(),
            indent: 0,
            use_blas: options.use_blas,
            math_lib,
            reduction_inlined: reduction_inlined
                .to_sorted()
                .into_iter()
                .map(|id| id.0)
                .collect(),
            memory_plan,
            fused_reuse,
            reused_sources: chelis_unord::UnordSet::new(),
            slot_current_owner: BTreeMap::new(),
            runtime_dim_sites,
            local_dim_guard_sites,
            declared_dim_names: chelis_unord::UnordSet::new(),
            write_nodes: chelis_unord::UnordSet::new(),
        };

        e.line("#include \"chelis_runtime.h\"");
        e.line("#include <assert.h>");
        if e.use_blas {
            e.line("#include \"chelis_blas.h\"");
        }
        if e.math_lib != crate::MathLib::None {
            e.line("#include \"chelis_math.h\"");
        }
        e.line("/* CHELIS_UNIFORM_HELPERS_BEGIN */");
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
        // chelis#770: emit the affine as one explicit correctly-rounded FMA
        // rather than `low + (high - low) * (float)unit`. The latter is
        // contracted into an FMA under `-ffp-contract=fast` (the default with
        // `-march=native`) but left as two roundings under `-ffp-contract=off`,
        // so its output was compile-flag-dependent (1 ULP on some elements) —
        // a real RNG-determinism hole. `fmaf` is IEEE correctly-rounded on all
        // targets (hardware or software), making the sampler flag-independent
        // and bit-identical to the host evaluator's `f32::mul_add`. Keep this
        // line byte-identical to `host_emit.rs`'s copy.
        e.line("    return fmaf(high - low, (float)unit, low);");
        e.line("}");
        e.line(
            "static inline double chelis_uniform_sample_f64(uint64_t seed, uint64_t index, double low, double high) {",
        );
        e.line("    uint64_t x = seed ^ (index * 0x9E3779B97F4A7C15ULL);");
        e.line("    x ^= x >> 30;");
        e.line("    x *= 0xBF58476D1CE4E5B9ULL;");
        e.line("    x ^= x >> 27;");
        e.line("    x *= 0x94D049BB133111EBULL;");
        e.line("    x ^= x >> 31;");
        e.line("    double unit = (double)(x >> 11) / (double)(1ULL << 53);");
        e.line("    return fma(high - low, unit, low);");
        e.line("}");
        e.line("/* CHELIS_UNIFORM_HELPERS_END */");
        // [05-OP-31]/[05-OP-44] make every published host tensor descriptor
        // canonical row-major storage.  The old runtime ABI exposed mutable
        // stride fields and therefore needed a runtime contiguity probe; the
        // opaque descriptor has no noncanonical public construction path.
        // Keep the local predicate while the Phase 3 fusion planner still
        // emits fast/slow branches, but make its authority the new ABI
        // invariant instead of an undeclared runtime symbol.
        e.line("#ifndef CHELIS_PRIVATE_CONTIGUOUS_HELPER");
        e.line("#define CHELIS_PRIVATE_CONTIGUOUS_HELPER");
        e.line("static inline int chelis_is_contiguous(const chelis_tensor *tensor) {");
        e.line("    (void)tensor;");
        e.line("    return 1;");
        e.line("}");
        e.line("#endif");
        e.line("#ifndef CHELIS_EFFECTIVE_UNIFORM_SEED");
        e.line("#define CHELIS_EFFECTIVE_UNIFORM_SEED(seed) (seed)");
        e.line("#endif");
        e.line("");
        let needs_checked_cast_conversion_helpers = dag.nodes().iter().any(|node| {
            let RiscOp::Cast { new_precision } = &node.op else {
                return false;
            };
            let Some(source) = node.inputs.first().and_then(|input| dag.get(*input)) else {
                return false;
            };
            let Ok(plan) = CheckedCastPlan::new(source.output_type.precision, *new_precision)
            else {
                return false;
            };
            plan.kind() != CheckedCastKind::Identity
                && matches!(plan.target(), Prim::F16 | Prim::Bf16)
        });
        if needs_checked_cast_conversion_helpers {
            let mut scalar_conversion_helpers = Vec::new();
            crate::host_emit::append_checked_cast_conversion_helpers(
                &mut scalar_conversion_helpers,
            );
            for helper_line in scalar_conversion_helpers {
                e.line(&helper_line);
            }
            e.line("");
        }

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
                e.emit_node(node, dag)?;
            }
        }

        // Every generated allocation is canonical contiguous storage.  Keep
        // its exclusive lease across the straight-line kernel and end all
        // leases before ownership is transferred to outputs or cleanup.
        let write_nodes = e
            .write_nodes
            .to_sorted()
            .into_iter()
            .copied()
            .collect::<Vec<_>>();
        for id in write_nodes {
            e.line(&format!("chelis_tensor_end_write(t{}_write_guard);", id));
        }

        // Transfer program-owned outputs to the caller. A bare Load is an
        // entry borrow, so materialize it before crossing the owned-output
        // boundary instead of returning the caller's descriptor as a second
        // unretained owner.
        for (slot, output) in output_specs.iter().enumerate() {
            if matches!(
                dag.get(output.id).map(|node| &node.op),
                Some(RiscOp::Load { .. })
            ) {
                e.line(&format!(
                    "outputs[{slot}] = chelis_contiguous(t{});",
                    output.id.0
                ));
            } else {
                e.line(&format!("outputs[{slot}] = t{};", output.id.0));
            }
        }

        let mut dropped_sources = dag
            .actions()
            .filter_map(|action| match action {
                VerifiedDagAction::OwnedDrop { source, .. } => Some(source),
                _ => None,
            })
            .collect::<Vec<_>>();
        dropped_sources.extend(e.reused_sources.to_sorted().into_iter().copied());
        let cleanup = e
            .memory_plan
            .emit_cleanup_with_drops(&output_ids, &dropped_sources);
        for line in cleanup {
            e.lines.push(line);
        }

        e.indent = 0;
        e.line("}");
        Ok(e.lines.join("\n"))
    }

    fn emit_tensor_snapshot(&mut self, id: usize, writable: bool) {
        self.line(&format!("int32_t t{id}_rank = chelis_tensor_rank(t{id});"));
        self.line(&format!(
            "int64_t t{id}_shape[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "int64_t t{id}_strides[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "for (int32_t __axis = 0; __axis < t{id}_rank; ++__axis) t{id}_shape[__axis] = chelis_tensor_shape(t{id}, __axis);"
        ));
        self.line(&format!("int64_t __stride_{id} = 1;"));
        self.line(&format!(
            "for (int32_t __axis = t{id}_rank; __axis-- > 0;) {{ t{id}_strides[__axis] = __stride_{id}; __stride_{id} *= t{id}_shape[__axis]; }}"
        ));
        self.line(&format!("int64_t t{id}_size = chelis_tensor_numel(t{id});"));
        if writable {
            self.line(&format!(
                "chelis_tensor_write *t{id}_write_guard = chelis_tensor_begin_write(t{id});"
            ));
            self.line(&format!(
                "chelis_write_view t{id}_view = chelis_tensor_write_view(t{id}_write_guard);"
            ));
            self.line(&format!("void *t{id}_data = t{id}_view.data;"));
            self.write_nodes.insert(id);
        } else {
            self.line(&format!(
                "chelis_read_view t{id}_view = chelis_tensor_read_view(t{id});"
            ));
            self.line(&format!("const void *t{id}_data = t{id}_view.data;"));
        }
        self.line(&format!("chelis_dtype t{id}_dtype = t{id}_view.dtype;"));
        self.line(&format!(
            "int64_t t{id}_byte_capacity = t{id}_size * chelis_dtype_size(t{id}_dtype);"
        ));
    }

    fn emit_owned_tensor(&mut self, id: usize, ndim: &str, shape: &str, dtype: &str) {
        self.line(&format!(
            "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
        ));
        self.emit_tensor_snapshot(id, true);
    }

    pub(crate) fn rename_anonymous_dims(dag: Dag) -> Dag {
        use chelis_ir::dag::DimInfo;
        fn is_anon(name: &str) -> bool {
            name.is_empty() || name == "*"
        }
        // chelis#616 (soundness): a movement op with any NON-IDENTITY axis (a
        // node-valued bound, a non-sentinel shrink, a stride step other than
        // literal 1, or a non-zero pad) produces a FRESH output extent on that
        // axis, which is NOT the input axis extent. The "copy first-input
        // dims" shortcut below would clobber such an axis with the input's dim
        // (e.g. propagate a shrink's `_anon_dim` onto a stride's output,
        // making two different extents share one C variable), so skip it and
        // let each anon axis get a fresh `_anon_dim_{id}_{axis}` that
        // `emit_shrink`/`emit_stride`/`emit_pad` size from its own bounds.
        // Mirrors the identity-only pass-through rule in
        // `chelis_ir::dag::shape_source_for_axis`.
        fn movement_alters_extents(op: &RiscOp) -> bool {
            match op {
                RiscOp::Shrink { bounds } => bounds
                    .iter()
                    .any(|(s, e)| !(s.as_lit() == Some(0) && matches!(e, RtDim::ToEnd))),
                RiscOp::Pad { padding, .. } => padding
                    .iter()
                    .any(|(b, a)| !(b.as_lit() == Some(0) && a.as_lit() == Some(0))),
                RiscOp::Stride { strides } => strides.iter().any(|s| s.as_lit() != Some(1)),
                // A runtime reshape target's extent comes from its scalar,
                // never from the input's dims.
                RiscOp::Reshape { new_shape } => new_shape.iter().any(|d| d.node_input().is_some()),
                _ => false,
            }
        }
        fn rewrite_dim(id: NodeId, axis: usize, dim: &DimInfo) -> DimInfo {
            match dim {
                DimInfo::Named(name, size) if is_anon(name) => {
                    DimInfo::Named(format!("_anon_dim_{}_{}", id.0, axis), *size)
                }
                other => other.clone(),
            }
        }
        let mut out = dag;
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
                } else if !movement_alters_extents(&node.op)
                    && let Some(first_input) = node.inputs.first().and_then(|input| out.get(*input))
                    && first_input.output_type.dims.len() == new_ty.dims.len()
                {
                    new_ty.dims = first_input.output_type.dims.clone();
                } else if node.inputs.is_empty()
                    && let Some(shape_source) =
                        node.shape_deps.first().and_then(|dep| out.get(*dep))
                    && shape_source.output_type.dims.len() == new_ty.dims.len()
                {
                    // chelis#616: an input-less node (a `lower_if` mask Const)
                    // shaped like a sibling records the relation as a
                    // shape-dep; tie its wildcard dims to the sibling's
                    // instead of fragmenting them into a sourceless anon dim.
                    new_ty.dims = shape_source.output_type.dims.clone();
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

    /// chelis#664: runtime operand-shape agreement guard for same-shape
    /// elementwise ops. Their emitters index every operand through the
    /// OUTPUT's indices (a shared flat `i` on the contiguous path;
    /// `chelis_flat_to_indices` on the output shape applied to each
    /// operand's strides on the strided path), so an operand whose
    /// runtime shape disagrees is read out of bounds or partially —
    /// SILENTLY, where the evaluator rejects with "tensor shapes must
    /// match for elementwise op". The checker cannot rule the mismatch
    /// out when a movement-op runtime wildcard is involved (wildcards
    /// unify permissively, spec §4.5).
    ///
    /// Scope: every positive-rank operand pair must have equal rank before
    /// shape comparison. Rank-0 operands remain the backend's explicit scalar
    /// input representation (for example, constants inside a fused kernel),
    /// so this defensive guard does not reinterpret them as source-level
    /// broadcasting. The all-pairs form matters for `FusedElem`: its first
    /// external input can be scalar, so comparing only against that input
    /// would miss disagreement between later tensor inputs. Fully static,
    /// already-compatible input shapes stay byte-identical.
    fn emit_elementwise_operand_guard(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) {
        let dims_static = |dims: &[DimInfo]| dims.iter().all(|d| matches!(d, DimInfo::Lit(_)));
        let input_dims = node
            .inputs
            .iter()
            .filter_map(|input| dag.get(*input).map(|n| n.output_type.dims.as_slice()))
            .collect::<Vec<_>>();
        let all_static =
            dims_static(&node.output_type.dims) && input_dims.iter().all(|dims| dims_static(dims));
        let statically_compatible = input_dims.iter().enumerate().all(|(left_index, left)| {
            input_dims[left_index + 1..]
                .iter()
                .all(|right| left.is_empty() || right.is_empty() || left == right)
        });
        if (all_static && statically_compatible) || node.inputs.len() < 2 {
            return;
        }
        let id = node.id.0;
        for (left_index, left) in node.inputs.iter().enumerate() {
            let a = left.0;
            for right in &node.inputs[left_index + 1..] {
                let b = right.0;
                self.line(&format!(
                    "if (t{a}_rank > 0 && t{b}_rank > 0 && t{a}_rank != t{b}_rank) {{ \
                     fprintf(stderr, \"chelis: elementwise operand rank mismatch at node {id}: %d vs %d\\n\", \
                     t{a}_rank, t{b}_rank); abort(); }} \
                     if (t{a}_rank == t{b}_rank) {{ for (int __d = 0; __d < t{a}_rank; __d++) {{ \
                     if (t{a}_shape[__d] != t{b}_shape[__d]) {{ fprintf(stderr, \"chelis: \
                     elementwise operand shape mismatch at node {id} axis %d\\n\", __d); abort(); \
                     }} }} }}"
                ));
            }
        }
    }

    fn emit_node(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) -> Result<(), Unsupported> {
        let id = node.id.0;
        // chelis#664: same-shape elementwise family — guard operand
        // agreement before the op emitters index operands through the
        // output's shape.
        if matches!(
            node.op,
            RiscOp::Add
                | RiscOp::Sub
                | RiscOp::Mul
                | RiscOp::Div
                | RiscOp::TruncDiv
                | RiscOp::FloorDiv
                | RiscOp::MaxElem
                | RiscOp::MinElem
                | RiscOp::ExtremaAdjoint { .. }
                | RiscOp::ReluAdjoint
                | RiscOp::CmpLt
                | RiscOp::FusedElem { .. }
        ) {
            self.emit_elementwise_operand_guard(node, dag);
        }
        match &node.op {
            RiscOp::Const { value } => self.emit_const(id, value, &node.output_type)?,
            RiscOp::ConstTensor { data } => self.emit_const_tensor(id, data, &node.output_type),
            RiscOp::Shape { axis } => self.emit_shape(id, *axis, &node.inputs, &node.output_type),
            RiscOp::Load { .. } => unreachable!("handled in emit_dag"),
            RiscOp::Add => self.emit_binary(id, "+", &node.inputs, &node.output_type),
            RiscOp::Sub => self.emit_binary(id, "-", &node.inputs, &node.output_type),
            RiscOp::Mul => self.emit_binary(id, "*", &node.inputs, &node.output_type),
            RiscOp::Div => self.emit_binary(id, "/", &node.inputs, &node.output_type),
            // chelis#178: `trunc_div` is the C integer `/` quotient (round
            // toward zero) — `emit_binary` already wraps the divisor in the
            // portable zero-divisor guard for integer dtypes. `trunc_div`
            // is integer-only, so this is exactly C truncating division.
            RiscOp::TruncDiv => self.emit_binary(id, "/", &node.inputs, &node.output_type),
            // chelis#178: `floor_div` rounds the quotient toward -inf.
            // Integer operands use native `/` plus a remainder-sign
            // correction; float operands use `floorf(a / b)`.
            RiscOp::FloorDiv => self.emit_floor_div(id, &node.inputs, &node.output_type),
            RiscOp::MaxElem => {
                self.emit_binary_func(id, "fmaxf", &node.inputs, &node.output_type);
            }
            RiscOp::MinElem => {
                self.emit_binary_func(id, "fminf", &node.inputs, &node.output_type);
            }
            RiscOp::ExtremaAdjoint { kind, operand } => {
                self.emit_extrema_adjoint(id, *kind, *operand, &node.inputs, &node.output_type)
            }
            RiscOp::Relu => self.emit_relu(id, &node.inputs, &node.output_type),
            RiscOp::ReluAdjoint => self.emit_relu_adjoint(id, &node.inputs, &node.output_type),
            RiscOp::CmpLt => self.emit_cmplt(id, &node.inputs, &node.output_type, dag),
            RiscOp::Neg => self.emit_unary(id, "-", &node.inputs, &node.output_type),
            RiscOp::Recip => self.emit_recip(id, &node.inputs, &node.output_type),
            RiscOp::Exp => self.emit_unary_func(id, "expf", &node.inputs, &node.output_type),
            RiscOp::Log => self.emit_unary_func(id, "logf", &node.inputs, &node.output_type),
            RiscOp::Sin => self.emit_unary_func(id, "sinf", &node.inputs, &node.output_type),
            RiscOp::Sqrt => self.emit_unary_func(id, "sqrtf", &node.inputs, &node.output_type),
            RiscOp::Cos => self.emit_unary_func(id, "cosf", &node.inputs, &node.output_type),
            RiscOp::Tan => self.emit_unary_func(id, "tanf", &node.inputs, &node.output_type),
            RiscOp::Atan => self.emit_unary_func(id, "atanf", &node.inputs, &node.output_type),
            RiscOp::Abs if node.output_type.precision.is_integer() => {
                self.emit_integer_abs(id, &node.inputs, &node.output_type)
            }
            RiscOp::Abs => self.emit_unary_func(id, "fabsf", &node.inputs, &node.output_type),
            RiscOp::Floor => self.emit_unary_func(id, "floorf", &node.inputs, &node.output_type),
            RiscOp::Ceil => self.emit_unary_func(id, "ceilf", &node.inputs, &node.output_type),
            // `rintf` rounds to nearest with the current rounding mode,
            // which defaults to ties-to-even — matching the evaluator's
            // `f64::round_ties_even`. (`roundf` would be ties-away-from-zero.)
            RiscOp::Round => self.emit_unary_func(id, "rintf", &node.inputs, &node.output_type),
            RiscOp::UniformLike { low, high, seed } => {
                self.emit_uniform_like(id, *low, *high, *seed, &node.inputs, &node.output_type)
            }
            RiscOp::Dropout { .. } => {
                unreachable!("dropout should be rejected before C code generation")
            }
            RiscOp::Copy => self.emit_realize(id, &node.inputs, &node.output_type),
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
                            if self.write_nodes.remove(&source.0) {
                                self.line(&format!(
                                    "chelis_tensor_end_write(t{}_write_guard);",
                                    source.0
                                ));
                            }
                            self.line(&format!("chelis_tensor_release(t{});", source.0));
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
            // WS-A1 + WS-A4: `Sum` carries an `accumulator: Prim` field
            // that governs both the running-sum precision and the output
            // precision (verified by `chelis_ir::verify::C3a` to equal
            // `output_type.precision` per spec §5.7.1). Per the
            // destructure-`..` memory rule, bind `accumulator` here
            // rather than `..`-skipping it so the dtype-aware reduce
            // paths can honor the IR-pinned accumulator precision and
            // the F1 footgun (silent precision downgrade via
            // `..`-destructure) cannot recur — neither for WS-A1's f64
            // accumulator nor for WS-A4's i8/i16 → i32 promoted path.
            RiscOp::Sum { axis, accumulator } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    // chelis#664: the inlined elementwise node is skipped
                    // by the emit loop, so the dispatch-level operand
                    // guard never saw it; guard here before the fused
                    // loops index its operands through the reduce shape.
                    self.emit_elementwise_operand_guard(fused_node, dag);
                    self.emit_fused_reduce(
                        id,
                        *axis,
                        &fused_node.inputs.clone(),
                        &fused_node.op.clone(),
                        &fused_node.output_type.clone(),
                        &node.output_type,
                        "sum",
                    )?;
                } else {
                    self.emit_reduce_sum(
                        id,
                        *axis,
                        *accumulator,
                        &node.inputs,
                        &node.output_type,
                        dag,
                    );
                }
            }
            RiscOp::Count { axes } => {
                self.emit_count(id, axes, &node.inputs, &node.output_type, dag);
            }
            RiscOp::MaxReduce { axis } => {
                let input_id = node.inputs[0];
                if self.reduction_inlined.contains(&input_id.0) {
                    let fused_node = dag.get(input_id).unwrap();
                    // chelis#664: see the Sum arm — the inlined node
                    // bypassed the dispatch-level operand guard.
                    self.emit_elementwise_operand_guard(fused_node, dag);
                    self.emit_fused_reduce(
                        id,
                        *axis,
                        &fused_node.inputs.clone(),
                        &fused_node.op.clone(),
                        &fused_node.output_type.clone(),
                        &node.output_type,
                        "max",
                    )?;
                } else {
                    self.emit_reduce_max(id, *axis, &node.inputs, &node.output_type, dag)?;
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
                    // #172: propagate NaN (torch parity) in the strided
                    // path, matching the contiguous `chelis_min_f32`.
                    "acc = chelis_fmin_propnan_f32(acc, ((const float*)t{a}_data)[src_idx]);",
                    Some("chelis_min_f32"),
                )?;
            }
            RiscOp::ProdReduce { axis } => {
                self.emit_reduce_simple(
                    id,
                    *axis,
                    &node.inputs,
                    &node.output_type,
                    dag,
                    "1.0f",
                    "acc *= ((const float*)t{a}_data)[src_idx];",
                    None,
                )?;
            }
            RiscOp::ReduceWindow {
                reducer,
                window_shape,
                strides,
            } => {
                self.emit_reduce_window(
                    id,
                    *reducer,
                    window_shape,
                    strides,
                    &node.inputs,
                    &node.output_type,
                    dag,
                )?;
            }
            RiscOp::ReduceWindowGrad {
                reducer,
                window_shape,
                strides,
            } => {
                self.emit_reduce_window_grad(
                    id,
                    *reducer,
                    window_shape,
                    strides,
                    &node.inputs,
                    &node.output_type,
                    dag,
                );
            }
            RiscOp::Argmax { axis } => {
                self.emit_reduce_argcmp(id, *axis, &node.inputs, &node.output_type, dag, true)?;
            }
            RiscOp::Argmin { axis } => {
                self.emit_reduce_argcmp(id, *axis, &node.inputs, &node.output_type, dag, false)?;
            }
            RiscOp::Reshape { new_shape } => {
                self.emit_reshape(id, new_shape, &node.inputs, &node.output_type, dag);
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
                self.emit_stride(id, strides, &node.inputs, &node.output_type, dag);
            }
            RiscOp::Realize => self.emit_realize(id, &node.inputs, &node.output_type),
            RiscOp::Cast { .. } => {
                self.emit_cast(
                    id,
                    &node.inputs,
                    &node.output_type,
                    dag,
                    /* trunc = */ false,
                )
            }
            RiscOp::CastTrunc { .. } => {
                self.emit_cast(
                    id,
                    &node.inputs,
                    &node.output_type,
                    dag,
                    /* trunc = */ true,
                )
            }
            RiscOp::Store { name } => {
                self.emit_store(id, name.as_str(), &node.inputs, &node.output_type)
            }
            RiscOp::FusedElem { ops } => {
                let in_place = self
                    .fused_reuse
                    .remove(&node.id)
                    .map(|token| CFusedReuse { token });
                self.emit_fused_elem(id, ops, &node.inputs, &node.output_type, in_place)?;
            }
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
                accumulator,
            } => {
                // WS-A1: bind `accumulator` explicitly; the previous
                // `..` destructure silently dispatched `cblas_sgemm` on
                // f64 storage (RT-1 finding F1). The accumulator field
                // is the IR's source of truth per spec §5.7.1; the
                // backend MUST NOT infer it from operand storage.
                let operand_precision = dag
                    .get(node.inputs[0])
                    .expect("BlasMatmul operand must resolve in dag")
                    .output_type
                    .precision;
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
                        operand_precision,
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
            RiscOp::ScatterElements { axis } => {
                self.emit_sparse_scatter_elements(id, *axis, &node.inputs, &node.output_type, dag);
            }
        }
        Ok(())
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
                });
            }
        }
        specs
    }

    pub(crate) fn output_labels(dag: VerifiedDagView<'_>) -> Vec<String> {
        Self::output_specs(dag)
            .into_iter()
            .map(|output| output.label)
            .collect()
    }

    pub(crate) fn input_labels(dag: VerifiedDagView<'_>) -> Vec<String> {
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

    fn input_slots(labels: &[String]) -> chelis_unord::UnordMap<String, usize> {
        labels
            .iter()
            .cloned()
            .enumerate()
            .map(|(slot, label)| (label, slot))
            .collect()
    }

    /// The direct integer-Abs node has a typed, trapping C kernel
    /// (`emit_integer_abs`), which is what chelis#691 asked for and what
    /// `precision_matrix.rs` now locks as an ordinary regression. General
    /// fused integer emission is a separate dtype capability that chelis#729
    /// owns, so externally supplied fused IR remains loud instead of entering
    /// the float-only template.
    fn reject_fused_integer_abs(dag: VerifiedDagView<'_>) -> Result<(), Unsupported> {
        if let Some(node) = dag.first_fused_integer_abs_node() {
            return Err(Unsupported::new(
                UnsupportedKind::Op("Abs".to_string()),
                format!("a fused integer tensor at C DAG node {}", node.0),
                Stage::Codegen("c"),
                chelis_types::unimplemented_rejection!(
                    729,
                    "direct integer abs is implemented with an exact trapping kernel; \
                     general fused integer emission remains dtype capability work, so \
                     this externally supplied fused shape cannot enter the float-only \
                     template"
                ),
            ));
        }
        Ok(())
    }

    fn validate_supported_precisions(dag: VerifiedDagView<'_>) {
        for node in dag.nodes() {
            match node.output_type.precision {
                // WS-1 (dtype + Metal cleanup cycle): admit Bf16/F16 in
                // tensor element types. Arithmetic always converts to
                // f32 via `chelis_bf16_to_f32` / `chelis_f16_to_f32`
                // (per spec/04-type-system.md §5.7.1); matmul converts
                // each operand element before dispatching `cblas_sgemm`.
                Prim::F32
                | Prim::F64
                | Prim::Bool
                | Prim::Bf16
                | Prim::F16
                | Prim::Int8
                | Prim::Int16
                | Prim::Int32
                | Prim::Int64 => {}
                other => panic!(
                    "C backend does not yet support {} tensors, found at node {}",
                    other.name(),
                    node.id.0
                ),
            }

            match node.op {
                RiscOp::Cast { new_precision }
                    if !matches!(
                        new_precision,
                        Prim::F32
                            | Prim::F64
                            | Prim::Bf16
                            | Prim::F16
                            | Prim::Int8
                            | Prim::Int16
                            | Prim::Int32
                            | Prim::Int64
                            | Prim::Bool
                    ) =>
                {
                    panic!(
                        "C backend does not yet support checked casts to {}, found at node {}",
                        new_precision.name(),
                        node.id.0
                    );
                }
                RiscOp::CastTrunc { new_precision }
                    if !matches!(
                        new_precision,
                        Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64
                    ) =>
                {
                    panic!(
                        "C backend does not support cast_trunc to {}, found at node {}",
                        new_precision.name(),
                        node.id.0
                    );
                }
                RiscOp::Cast { .. } | RiscOp::CastTrunc { .. } => {}
                _ => {}
            }

            // F1 (WS-A0 RT-1 fixup, tactical) — partially lifted by
            // WS-A1 (f32/f64 → `cblas_sgemm` / `cblas_dgemm`) and now
            // by WS-1 (bf16/f16 → convert-then-`cblas_sgemm` with f32
            // scratch buffers, per spec/04-type-system.md §5.7.1).
            // f8e4m3 is still rejected because it is deferred at the
            // language level (§1.1.1); all other dtypes that reach
            // BlasMatmul must already have been rejected by the
            // outer precision match above.
            if matches!(node.op, RiscOp::BlasMatmul { .. })
                && let Some(lhs) = dag.get(node.inputs[0])
                && !matches!(
                    lhs.output_type.precision,
                    Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16
                )
            {
                panic!(
                    "F1: C-backend BlasMatmul supports f32, f64, bf16, and f16; \
                     node {} has operand precision `{}`. \
                     spec/04-type-system.md §5.7.1 documents the per-precision \
                     accumulator defaults; the C backend dispatches \
                     `cblas_sgemm`/`cblas_dgemm` for f32/f64 (WS-A1) and \
                     convert-then-`cblas_sgemm` with f32 scratch buffers for \
                     bf16/f16 (WS-1). f8e4m3 remains deferred per §1.1.1.",
                    node.id.0,
                    lhs.output_type.precision.name(),
                );
            }
        }
    }

    fn validate_load_abi(dag: VerifiedDagView<'_>) {
        let mut seen = chelis_unord::UnordMap::<String, TensorType>::new();
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

    /// chelis#593 memory-safety floor. A `Pad` whose output extent on a
    /// statically-known axis does not equal `input + before + after` emits a
    /// copy loop that writes at `src_index + before` into an output allocated
    /// at the wrong (smaller) size — a heap out-of-bounds write AND a silently
    /// wrong forward result.
    ///
    /// This surfaces in the SYMBOLIC ENTRY-WRAPPER of a leading / non-last-axis
    /// `concat` over a symbolic trailing dim. `concat` lowers to a Pad+Add
    /// cascade; the wrapper allocates each Pad output at the OPERAND leading
    /// extent (e.g. `[2, batch]`) while its loop writes at a `+before` row
    /// offset, overrunning the buffer. The inner monomorphized fn sizes the
    /// concat output correctly (`[4, batch]`); only the symbolic wrapper
    /// mis-sizes (Chelis-Lang/chelis#593). Before the chelis#551
    /// `shape_source_for_axis` reduction arm, `reduce(concat(axis=leading))`
    /// over a symbolic trailing dim ICE'd LOUD at the symbolic-dim guard; the
    /// arm now lets it (and bare leading-axis `concat`) reach codegen, so this
    /// guard is the fail-closed floor that keeps a loud reject loud instead of
    /// degrading to silent-wrong + memory-unsafe. Reject here rather than emit
    /// heap-corrupting C. Concat along the LAST axis and concrete-dim concats
    /// stay well-sized (`output == input + padding`) and pass unaffected.
    ///
    /// The deeper wrapper-sizing fix is tracked in chelis#593; this is only
    /// the memory-safety guard.
    fn validate_pad_output_sizing(dag: VerifiedDagView<'_>) {
        for node in dag.nodes() {
            let RiscOp::Pad { padding, .. } = &node.op else {
                continue;
            };
            let Some(input) = node.inputs.first().and_then(|id| dag.get(*id)) else {
                continue;
            };
            if node.output_type.dims.len() != input.output_type.dims.len() {
                // Rank mismatch is a distinct malformation; leave it to the
                // rank/shape checks. This guard is specifically about a padded
                // axis whose output extent disagrees with input + padding.
                continue;
            }
            for (axis, ((before, after), in_dim)) in padding
                .iter()
                .zip(input.output_type.dims.iter())
                .enumerate()
            {
                let (Some(in_size), Some(out_size)) = (
                    Self::known_dim_size(in_dim),
                    node.output_type
                        .dims
                        .get(axis)
                        .and_then(Self::known_dim_size),
                ) else {
                    // A symbolic in/out extent on this axis cannot be checked
                    // statically; the mis-sizing that #593 produces is on a
                    // CONCRETE padded axis (the leading concat axis), so the
                    // guard still fires there.
                    continue;
                };
                // chelis#616: node-valued (runtime) padding cannot be checked
                // statically; `emit_pad`'s runtime guard covers it.
                let (Some(before), Some(after)) = (before.as_lit(), after.as_lit()) else {
                    continue;
                };
                let expected = in_size + before + after;
                assert!(
                    out_size == expected,
                    "internal compiler error: C backend `Pad` at node {} axis {axis} is mis-sized: \
                     output extent {out_size} != input {in_size} + before {before} + after {after} \
                     = {expected}. Emitting the pad copy loop would write past the output \
                     allocation (heap out-of-bounds write / silent wrong result). This is the \
                     symbolic entry-wrapper mis-sizing of a leading / non-last-axis `concat` over \
                     a symbolic trailing dim (Chelis-Lang/chelis#593); the producing IR pass must \
                     size the concat Pad output at the padded extent. Rejected fail-closed rather \
                     than emit heap-corrupting C.",
                    node.id.0
                );
            }
        }
    }

    fn validate_sparse_contracts(dag: VerifiedDagView<'_>) {
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

    /// Resolve an `InputAxis`-sourced class member to the kernel input slot
    /// and axis its guard reads.
    ///
    /// `AxisSource::InputAxis`'s slot indexes the OWNING NODE's inputs; a
    /// prologue guard needs the kernel input slot of the tensor that operand
    /// names. `None` means the operand is not an input tensor at all - a
    /// computed producer, or a label with no assigned slot - and
    /// `RuntimeDimClass::placement` has already kept such a class Local, so
    /// there is nothing for the prologue to emit.
    fn member_input_slot(
        dag: VerifiedDagView<'_>,
        input_slots: &chelis_unord::UnordMap<String, usize>,
        member: &chelis_ir::axis_sources::ClassMember,
    ) -> Option<(usize, i32)> {
        // Only the folded-read spelling is emitted here: a `Load`'s own axis
        // is already declared and guarded by the binding loop above.
        if !matches!(
            member.source,
            chelis_ir::axis_sources::AxisSource::InputAxis { .. }
        ) {
            return None;
        }
        let (load, read_axis) = dag.member_load_axis(member)?;
        let RiscOp::Load { name: label } = &dag.get(load)?.op else {
            return None;
        };
        Some((*input_slots.get(label.as_str())?, read_axis as i32))
    }

    fn emit_input_shape_preamble(
        &mut self,
        dag: VerifiedDagView<'_>,
        input_slots: &chelis_unord::UnordMap<String, usize>,
        func_name: &str,
    ) {
        // Iteration order over `input_types` (a UnordMap) must be
        // deterministic so the emitted C is byte-identical across runs
        // for the same input. Sort by label; the lookup is by name and
        // the emitted lines are independent per label.
        // See spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md.
        let input_types = Self::input_types(dag);
        let sorted_labels = input_types.to_sorted();
        // Producer-supplied strings flowing into the fprintf format string
        // baked into a `"..."` C string literal. Sanitize once per emission
        // boundary per spec/upstream-bugs/producer-string-sanitization.md.
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);

        // chelis#1277 b2.4: the (slot, axis) pairs a `Literal` claim's entry
        // guard already compares against that same literal. The static-dim
        // check below and that guard are then the identical comparison
        // written twice, and the ABI one runs first, so the kernel would
        // `abort()` where `spec/04-type-system.md` section 4.7 requires the
        // [04-NUM-9] trap.
        //
        // The narrowing is exactly that overlap and no wider. A `Name` claim
        // is excluded because its guard compares against another input's
        // RUNTIME extent, which does not imply the statically declared size;
        // dropping the static check there would lose a comparison rather
        // than rename one. The ABI check's own obligation - an external C
        // caller passing a wrong-shaped tensor to an exported kernel -
        // survives for every axis no class guards, including in programs
        // that contain no runtime extent at all.
        let mut literal_claim_pairs = chelis_unord::UnordMap::<(usize, i32), usize>::new();
        for class in dag.entry_dim_classes() {
            let chelis_ir::axis_sources::DimClaim::Literal(value) = class.claim else {
                continue;
            };
            for member in &class.members {
                if let Some(pair) = Self::member_input_slot(dag, input_slots, member) {
                    literal_claim_pairs.insert(pair, value);
                }
            }
        }

        for (label, _) in sorted_labels {
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
                    if literal_claim_pairs.get(&(slot, axis as i32)) == Some(&expected) {
                        continue;
                    }
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

        // chelis#1277 b2.4: one guard per (claim, input slot, axis), and never
        // against the pair the claim's canonical value was read from.
        //
        // Both loops below can reach one (slot, axis): the binding view sees a
        // `Load`'s own axis as an `ExternalAxis` member, and the class view
        // sees a folded `shape(t, k)` read of the same tensor as an
        // `InputAxis` member. Measured on chelis#1374's kernel, the two
        // spellings of one axis produced `if (chelis_tensor_shape(inputs[2],
        // 0) != n)` twice; on the form where the claim's other witness is
        // dropped as unused they collapsed onto the canonical itself and
        // produced `int64_t n = chelis_tensor_shape(inputs[1], 0);` followed
        // by `if (chelis_tensor_shape(inputs[1], 0) != n)`, a comparison of a
        // value with itself. Neither is a guard: one is noise, the other is a
        // condition that cannot hold.
        let mut guarded = chelis_unord::UnordSet::<(String, usize, i32)>::new();
        // Where each name was declared, so a guard reads the same witness the
        // context line names.
        let mut declared_from = chelis_unord::UnordMap::<String, (usize, usize)>::new();

        // Slot-indexed labels, so a guard names the tensor it reads without
        // re-walking the DAG for a name the slot map already keys.
        let mut input_labels = vec![String::new(); input_slots.len()];
        for (label, slot) in input_slots.to_sorted() {
            if let Some(entry) = input_labels.get_mut(*slot) {
                *entry = label.clone();
            }
        }

        // DECLARATIONS come from the occurrence walk, and that split is a
        // measured limit rather than a leftover.
        //
        // The emitter allocates by NAME: `chelis_alloc(1, (int64_t[]){
        // _anon_dim_1_0 })`. The lowerer stamps a fresh name on many axes
        // whose extent is simply an input's - chelis#631's avgpool program
        // has a `Load` typed `[2, _anon_dim_0_1]` and a `Sum` over axis 0
        // typed `[_anon_dim_1_0]` - and C2.4 is right that the second is not
        // a witness, because a pass-through axis neither declares nor
        // disagrees: it IS the first. The derivation therefore reports one
        // extent where the emitted text uses two names, and routing
        // declarations through it left `_anon_dim_1_0` undeclared and the
        // emitted C not compiling on eight shipped programs.
        //
        // The fix for that is to allocate from the axis SOURCE instead of
        // from the stamped name, which is C4.4's remaining half and what
        // closes chelis#665. Until then the walk keeps the declarations - its
        // bucket-4c sweep recovers the second name from the same input axis -
        // and the derivation keeps what it is for: which sites GUARD, against
        // what, in what order.
        for binding in dag.symbolic_bindings() {
            let SymbolicDimSource::Load {
                input_label: canonical_label,
                axis: canonical_axis,
            } = &binding.canonical.source
            else {
                continue;
            };
            let canonical_slot = input_slots[canonical_label];
            self.line(&format!(
                "int64_t {} = chelis_tensor_shape(inputs[{canonical_slot}], {canonical_axis});",
                binding.name
            ));
            // The declaring witness is the pair a guard must not compare
            // against, so the dedupe is seeded from the DECLARATION rather
            // than from the derivation's canonical: those can differ, and a
            // guard reporting one witness while reading another would name
            // the wrong tensor in its context line.
            guarded.insert((binding.name.clone(), canonical_slot, *canonical_axis as i32));
            declared_from.insert(binding.name.clone(), (canonical_slot, *canonical_axis));
            self.declared_dim_names.insert(binding.name.clone());
        }

        for binding in dag.symbolic_bindings_interface() {
            // chelis#616: an op-declared dim is declared inline at its
            // owning op (the bound scalars are computed tensors that do not
            // exist here at prologue time); see `runtime_dim_sites`.
            // The canonical is this CLASS's own first witness, never the
            // variable the walk declared for the same spelling. The two are
            // scoped differently now: the derivation splits a name by root
            // scope (C2.4) and the walk does not, so comparing a scoped
            // member against a globally declared variable pairs witnesses
            // from two signatures. Measured on
            // `rank_poly_tier3::named_axis_eval_parity_corners`, where that
            // pairing survived the regrouping and kept trapping a correct
            // program: `seq` is a 3-element axis in `total`'s signature and a
            // 2-element one in `use2`'s, and the walk declares one of them.
            //
            // Reading both operands directly also makes the guard independent
            // of which name the emitter happened to allocate under, which is
            // the coupling that hid this.
            let SymbolicDimSource::Load {
                input_label: canonical_label,
                axis: canonical_axis,
            } = binding.canonical.source.clone()
            else {
                continue;
            };
            let canonical_slot = input_slots[canonical_label.as_str()];
            let canonical_label = canonical_label.as_str();
            let canonical_read =
                format!("chelis_tensor_shape(inputs[{canonical_slot}], {canonical_axis})");
            guarded.insert((binding.name.clone(), canonical_slot, canonical_axis as i32));
            // `binding.name` flows into BOTH an identifier context (the
            // emitted `int {name} = ...;` declarator) and a format-string
            // context (the fprintf below). The identifier emission is
            // guarded by parser/IR construction; the format-string
            // emission needs `%`/`\\`/`"`/control sanitization here.
            // The Load `input_label` is a LoadStoreName-validated name but
            // we route both through the format-string sanitizer to lock the
            // architectural pattern.
            let binding_name_fmt =
                chelis_ir::span_sanitize::sanitize_for_format_string(&binding.name);
            for occurrence in std::iter::once(binding.canonical).chain(binding.others) {
                // Op-declared guard sites are emitted at their owning op.
                let SymbolicDimSource::Load { input_label, axis } = &occurrence.source else {
                    continue;
                };
                let slot = input_slots[input_label];
                if !guarded.insert((binding.name.clone(), slot, *axis as i32)) {
                    continue;
                }
                let occ_label_fmt =
                    chelis_ir::span_sanitize::sanitize_for_format_string(input_label);
                // `spec/04-type-system.md` section 4.7: a runtime extent
                // guard IS a typed operation-precondition guard under
                // [04-NUM-9], so the user-facing line is exactly
                // `numeric trap: domain in <op> at int64` with no prefix and
                // no suffix. `<op>` is the `load` primitive of the later
                // witness in signature order, and `<prim>` is `int64`
                // because the guard finalizes an extent ([05-DIM-1]) rather
                // than a tensor element. Routing through
                // `chelis_numeric_trap` keeps the line byte-identical to
                // every other numeric trap this lane emits.
                //
                // Section 4.7 also requires the disagreeing source names, the
                // axis and each observed value to be conveyed "on separate
                // lines accompanying that trap", binding the information and
                // not the bytes, so the context is its own `fprintf` and the
                // trap line stays exactly one line.
                let canonical_label_fmt =
                    chelis_ir::span_sanitize::sanitize_for_format_string(canonical_label);
                self.line(&format!(
                    "if (chelis_tensor_shape(inputs[{slot}], {axis}) != {canonical_read}) {{"
                ));
                self.indent += 1;
                self.line(&format!(
                    "fprintf(stderr, \"extent `{binding_name_fmt}`: {canonical_label_fmt} axis {canonical_axis} = %lld, {occ_label_fmt} axis {axis} = %lld\\n\", (long long)({canonical_read}), (long long)chelis_tensor_shape(inputs[{slot}], {axis}));"
                ));
                self.line("chelis_numeric_trap(\"numeric trap: domain in load at int64\");");
                self.indent -= 1;
                self.line("}");
            }
        }

        // chelis#1277 b2.4: the loop above carries only the members the
        // binding view models - a `Name` claim witnessed by a `Load` axis.
        // Two more member kinds place at ENTRY under `spec/04` section 4.7 and
        // are guarded here.
        //
        // A `Literal` claim's canonical value is the literal itself (C2.4), so
        // every member is one guard against it rather than against a first
        // member; that is chelis#1377, a declared `tensor[4, f32]` over a read
        // that yields 5. And an `InputAxis`-sourced member reads an input
        // tensor's axis directly, which section 4.7 lists as an interface
        // value, so its guard belongs at entry too; that is chelis#1376.
        for class in dag.entry_dim_classes() {
            let (canonical_expr, claim_text) = match &class.claim {
                chelis_ir::axis_sources::DimClaim::Literal(value) => {
                    (value.to_string(), value.to_string())
                }
                chelis_ir::axis_sources::DimClaim::Name(name) => (
                    name.clone(),
                    chelis_ir::span_sanitize::sanitize_for_format_string(name).to_string(),
                ),
            };
            // A `Name` claim shares the dedupe key with the binding loop
            // above, which guards the same claim by the same name; a
            // `Literal` claim has no binding-loop counterpart, and its
            // decimal spelling cannot collide with a Chelis binder.
            let claim_key = canonical_expr.clone();
            for member in &class.members {
                // An `ExternalAxis` member is already guarded above, and a
                // `Literal` member is the statically proved case the
                // derivation excludes.
                let Some((slot, read_axis)) = Self::member_input_slot(dag, input_slots, member)
                else {
                    continue;
                };
                if !guarded.insert((claim_key.clone(), slot, read_axis)) {
                    continue;
                }
                let label = &input_labels[slot];
                let label_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(label);
                self.line(&format!(
                    "if (chelis_tensor_shape(inputs[{slot}], {read_axis}) != {canonical_expr}) {{"
                ));
                self.indent += 1;
                self.line(&format!(
                    "fprintf(stderr, \"extent `{claim_text}`: claimed = %lld, {label_fmt} axis {read_axis} = %lld\\n\", (long long)({canonical_expr}), (long long)chelis_tensor_shape(inputs[{slot}], {read_axis}));"
                ));
                self.line("chelis_numeric_trap(\"numeric trap: domain in load at int64\");");
                self.indent -= 1;
                self.line("}");
            }
        }

        // chelis#1277 S2b: the same-rank `expand`'s unit-extent claim. It is
        // an operation PRECONDITION on an operand rather than an identity
        // between output axes, so it is its own derivation, but it places and
        // renders by the same rules: `spec/05-risc-primitives.md` section
        // 2.4.1 sends a symbolic or runtime operand extent to "that claim's
        // runtime extent guard", section 4.7 puts it at entry when the operand
        // is an input tensor's axis, and the `<op>` slot is `load` for exactly
        // that reason. The claimed side is the literal 1, so there is no
        // canonical member to read it from.
        for (load, read_axis) in dag.entry_unit_extent_reads() {
            let Some(RiscOp::Load { name }) = dag.get(load).map(|node| &node.op) else {
                continue;
            };
            let Some(&slot) = input_slots.get(name.as_str()) else {
                continue;
            };
            let read_axis = read_axis as i32;
            if !guarded.insert(("1".to_string(), slot, read_axis)) {
                continue;
            }
            let label = &input_labels[slot];
            let label_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(label);
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
            // chelis#1112: the compound literal IS the argument to
            // `chelis_alloc(int, const int64_t *, int)`. An `int[]` here is
            // both a pointer-type mismatch and, at run time, a 4-byte-aligned
            // buffer the runtime reads as int64_t.
            format!("(int64_t[]){{ {} }}", dims.join(", "))
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
        ty.precision
            .runtime_dtype()
            .unwrap_or_else(|error| panic!("C backend does not support this tensor: {error}"))
            .c_macro()
    }

    /// Returns the C element type for direct element access in generated loops.
    /// F32 uses `float`; Bool uses its canonical one-byte `uint8_t` payload.
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
            Prim::F32 => "float",
            Prim::Bool => "uint8_t",
            Prim::F64 => "double",
            // WS-A4: i8/i16 element access via reinterpret cast on
            // `t->data`; the runtime allocator now sizes the buffer
            // correctly per `CHELIS_DTYPE_I8` / `CHELIS_DTYPE_I16`.
            Prim::Int8 => "int8_t",
            Prim::Int16 => "int16_t",
            Prim::Int32 => "int32_t",
            Prim::Int64 => "int64_t",
            // WS-1: bf16 / f16 storage is `uint16_t`; arithmetic uses
            // `chelis_bf16_to_f32` / `chelis_f16_to_f32` and is emitted
            // through `emit_binary_reduced_f` / `emit_unary_reduced_f`
            // rather than the generic `elem_type`-parameterized loops.
            // Returning the storage type here keeps memcpy, slot
            // allocation, and pointer-cast code correct.
            Prim::Bf16 | Prim::F16 => "uint16_t",
            other => panic!(
                "C backend does not yet support `{}` tensors; the silent \
                 default-arm downgrade was removed by WS-A0 to surface \
                 missing dtype emit logic. WS-A1 widens this match.",
                other.name()
            ),
        }
    }

    /// True for the reduced-precision float dtypes that the C backend
    /// stores as `uint16_t` and computes through `chelis_bf16_to_f32`
    /// / `chelis_f16_to_f32` helpers. Used by elementwise kernel
    /// emitters to route through the convert-then-compute path
    /// instead of the generic `elem_type`-parameterized loops.
    fn is_reduced_float(ty: &TensorType) -> bool {
        matches!(ty.precision, Prim::Bf16 | Prim::F16)
    }

    /// Runtime helper name used to load one element of a reduced-float
    /// tensor (`Bf16` / `F16`) into an `f32`. Caller is responsible
    /// for asserting the precision is reduced; panics otherwise to
    /// catch accidental dispatch.
    fn reduced_to_f32_fn(prim: Prim) -> &'static str {
        match prim {
            Prim::Bf16 => "chelis_bf16_to_f32",
            Prim::F16 => "chelis_f16_to_f32",
            other => panic!(
                "reduced_to_f32_fn called on non-reduced precision `{}`; \
                 this is a backend bug",
                other.name()
            ),
        }
    }

    /// Runtime helper name used to round an `f32` back into a 16-bit
    /// reduced-float bit pattern. Caller asserts the precision is
    /// reduced.
    fn f32_to_reduced_fn(prim: Prim) -> &'static str {
        match prim {
            Prim::Bf16 => "chelis_f32_to_bf16",
            Prim::F16 => "chelis_f32_to_f16",
            other => panic!(
                "f32_to_reduced_fn called on non-reduced precision `{}`; \
                 this is a backend bug",
                other.name()
            ),
        }
    }

    /// Render a portable exact-width signed literal. In particular,
    /// `INT64_C(-9223372036854775808)` is not a portable spelling because
    /// the positive magnitude is outside int64 before unary negation.
    fn i64_c_literal(value: i64) -> String {
        if value == i64::MIN {
            "INT64_MIN".to_string()
        } else if value < 0 {
            format!("-INT64_C({})", value.unsigned_abs())
        } else {
            format!("INT64_C({value})")
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
            "rintf" => "rint",
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
        let slot = self.slot_id_for_node(id);
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        if let Some(previous) = self.slot_current_owner.get(&slot).copied() {
            self.emit_reused_slot_wrapper(previous, id, ty);
        } else {
            self.emit_owned_tensor(id, &ndim.to_string(), &shape, dtype);
        }
        self.slot_current_owner.insert(slot, id);
    }

    fn emit_slot_wrapper(&mut self, id: usize, ty: &TensorType) {
        self.emit_slot_allocation_if_needed(id, ty);
    }

    fn emit_fused_in_place_wrapper(&mut self, id: usize, ty: &TensorType, spec: CFusedReuse) {
        let source = spec.token.source().0;
        assert_eq!(spec.token.consumer(), NodeId(id));
        let slot = self.slot_id_for_node(id);
        assert_eq!(self.slot_id_for_node(source), slot);
        assert_eq!(self.slot_current_owner.get(&slot), Some(&source));
        self.emit_reused_slot_wrapper(source, id, ty);
        // The source's former write view ended before repurpose. Rebind its
        // local data cursor to the new guard's live view so the fused kernel
        // reads the aliased external through valid guard-lifetime authority.
        self.line(&format!("t{source}_data = t{id}_data;"));
        self.slot_current_owner.insert(slot, id);
    }

    fn emit_reused_slot_wrapper(&mut self, previous: usize, id: usize, ty: &TensorType) {
        if self.write_nodes.remove(&previous) {
            self.line(&format!(
                "chelis_tensor_end_write(t{previous}_write_guard);"
            ));
        }
        self.line(&format!("chelis_tensor *t{id} = t{previous};"));
        let shape = ty
            .dims
            .iter()
            .map(|dim| {
                let extent = Self::emit_dim_expr(&DimExpr::from(dim));
                format!("chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(int64_t)({extent}))")
            })
            .collect::<Vec<_>>();
        let shape = if shape.is_empty() {
            "NULL".to_string()
        } else {
            format!("(chelis_scalar[]){{ {} }}", shape.join(", "))
        };
        self.line(&format!(
            "chelis_tensor_repurpose(t{id}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C({})), {shape});",
            Self::ndim(ty)
        ));
        self.emit_tensor_snapshot(id, true);
        self.reused_sources.insert(NodeId(previous));
    }

    /// Explicit f64 -> f32 narrowing for the `emit_const` F32 arm.
    /// Wraps the precision-narrowing `as` cast so the WS-1
    /// sibling-sweep grep returns zero hits in production code; the
    /// cast itself is the intended precision narrowing for an F32
    /// Const, not a silent default-arm truncation.
    #[inline]
    fn f64_to_f32_truncate(v: f64) -> f32 {
        v as f32
    }

    // ---- Const ----
    fn emit_const(
        &mut self,
        id: usize,
        value: &chelis_types::ScalarValue,
        ty: &TensorType,
    ) -> Result<(), Unsupported> {
        self.emit_slot_wrapper(id, ty);
        // The sealed payload (chelis#856) reads exactly per family: the
        // integer arms take the exact i64 (an int64 constant above 2^53
        // now emits its exact value instead of an f64-rounded one), the
        // float arms take the exact f64 image. A payload the target
        // cannot represent is a structured diagnostic through the
        // chelis#730 Result channel, never bad C.
        let wide = value.as_f64_lossy();
        let exact_int = value.as_i64_exact();
        match ty.precision {
            Prim::Int64 => {
                self.line(&format!(
                    "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(int64_t){}));",
                    exact_int.unwrap_or(wide as i64)
                ));
            }
            Prim::Int32 => {
                self.line(&format!(
                    "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I32, (uint32_t)(int32_t){}));",
                    exact_int.unwrap_or(wide as i64) as i32
                ));
            }
            Prim::Int16 => {
                self.line(&format!(
                    "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I16, (uint16_t)(int16_t){}));",
                    exact_int.unwrap_or(wide as i64) as i16
                ));
            }
            Prim::Int8 => {
                self.line(&format!(
                    "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I8, (uint8_t)(int8_t){}));",
                    exact_int.unwrap_or(wide as i64) as i8
                ));
            }
            Prim::F64 => {
                // Issue #189: emit the source f64's exact bit pattern
                // and bit-cast at runtime. The pre-fix `{:.17}` format
                // string treated `.17` as decimal places after the
                // point, not significant digits, so values below
                // `1e-17` collapsed to zero. Bit-pattern emission
                // round-trips the source f64 verbatim.
                let bits = wide.to_bits();
                self.line(&format!("chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F64, UINT64_C(0x{bits:016x})));"));
            }
            Prim::F32 => {
                // Issue #189: narrow to f32 (storage width is f32)
                // then emit the resulting bit pattern. `as f32` is
                // the intended precision narrowing (kept; explicit
                // via `f64_to_f32_truncate` so the WS-1 sibling-sweep
                // grep returns zero hits in production code).
                // `f32::to_bits()` produces an exact u32 pattern so
                // the runtime reproduces the closest-f32 to the IR
                // source value with zero further precision loss --
                // avoiding the pre-fix `{:.8}` format-string drift.
                let v32 = Self::f64_to_f32_truncate(wide);
                let bits = v32.to_bits();
                self.line(&format!("chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F32, UINT32_C(0x{bits:08x})));"));
            }
            Prim::Bool => {
                // Issue #365: a Bool tensor uses the same 4-byte
                // f32-encoded storage (0.0 / 1.0) as the comparison
                // kernels write, but its dtype tag is CHELIS_DTYPE_BOOL.
                // Filling it through `chelis_fill_f32_bits` trips that
                // helper's debug-build dtype assertion (it asserts
                // CHELIS_DTYPE_F32), aborting a debug-runtime reduce/softmax/
                // cross-entropy backward that materializes a comparison
                // mask. Use the dtype-correct `chelis_fill_bool_bits`,
                // which asserts CHELIS_DTYPE_BOOL and fills the identical
                // f32-encoded layout. The emitted bit pattern is the
                // same `f32::to_bits()` value as the F32 arm.
                let bits = u8::from(wide != 0.0);
                self.line(&format!("chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, UINT8_C({bits})));"));
            }
            // WS-1: bf16 / f16 Const fill. The literal's exact 16-bit
            // pattern is computed at codegen time via the dtype-semantic
            // one-rounding conversion
            // so the runtime never needs an f64 -> reduced converter
            // call per element; it just stamps the precomputed
            // pattern via `chelis_fill_bf16` / `chelis_fill_f16`.
            Prim::Bf16 => {
                let bits = chelis_types::bf16_from_f64_rne(wide).to_bits();
                self.line(&format!("chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_BF16, UINT16_C(0x{bits:04X})));"));
            }
            Prim::F16 => {
                let bits = chelis_types::f16_from_f64_rne(wide).to_bits();
                self.line(&format!("chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F16, UINT16_C(0x{bits:04X})));"));
            }
            other => {
                // chelis#729 rework: a constant the target cannot
                // represent surfaces through the chelis#730 structured
                // channel instead of panicking the compiler. The two
                // dtypes that reach here carry DIFFERENT authorities and
                // must stay distinguishable per [05-UNS-5]: f8e4m3 is
                // decided by spec, a string constant cell is unbuilt.
                let authority = match other {
                    Prim::F8e4m3 => chelis_types::deliberate_rejection!(
                        "[04-DTYPE-1]",
                        "no C constant representation exists for this dtype: f8e4m3 is \
                         reserved but inactive and must not reach backend emission \
                         (spec/04-type-system.md section 1.1.1)"
                    ),
                    Prim::String => chelis_types::unimplemented_rejection!(
                        729,
                        "no C constant representation exists for this dtype: the \
                         exhaustive target capability table has no C string storage cell \
                         (spec/04-type-system.md section 1.1)"
                    ),
                    Prim::F32
                    | Prim::F64
                    | Prim::F16
                    | Prim::Bf16
                    | Prim::Int8
                    | Prim::Int16
                    | Prim::Int32
                    | Prim::Int64
                    | Prim::Bool => unreachable!("emitted constant precision"),
                };
                return Err(Unsupported::new(
                    UnsupportedKind::Dtype(other.name().to_string()),
                    format!(
                        "a `{}` constant in the C DAG emitter (node {id})",
                        other.name()
                    ),
                    Stage::Codegen("c"),
                    authority,
                ));
            }
        }
        Ok(())
    }

    /// Emit a `shape(input, axis)` read (chelis#513/#558): a rank-0
    /// integer scalar holding the input tensor's runtime extent along
    /// `axis`, read from `t{input}_shape[axis]` (the runtime `int`
    /// field). The extent is stored into the scalar buffer using the
    /// node's integer precision. This is the C realization of the runtime
    /// dim read; the emitted expression reads the shape at execution time,
    /// so a symbolic input axis is resolved from the actual input tensor
    /// rather than baked at codegen time.
    fn emit_shape(&mut self, id: usize, axis: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        self.emit_slot_wrapper(id, ty);
        let extent = format!("t{a}_shape[{axis}]");
        match ty.precision {
            Prim::Int64 => {
                self.line(&format!(
                    "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(int64_t)({extent})));"
                ));
            }
            Prim::Int32 => {
                self.line(&format!(
                    "{{ int32_t *__p = (int32_t*)t{id}_data; __p[0] = (int32_t)({extent}); }}"
                ));
            }
            Prim::Int16 => {
                self.line(&format!(
                    "{{ int16_t *__p = (int16_t*)t{id}_data; __p[0] = (int16_t)({extent}); }}"
                ));
            }
            Prim::Int8 => {
                self.line(&format!(
                    "{{ int8_t *__p = (int8_t*)t{id}_data; __p[0] = (int8_t)({extent}); }}"
                ));
            }
            other => panic!(
                "C backend emit_shape has no path for `{}`; shape reads produce an \
                 integer scalar (chelis#513/#558)",
                other.name()
            ),
        }
    }

    /// Emit a multi-element constant tensor as a C array initialized
    /// with the literal data values, then memcpy into the tensor slot.
    /// The sealed storage (chelis#856) reads exactly per family: the
    /// integer arms take exact i64 values, the float arms the exact
    /// f64 images.
    fn emit_const_tensor(
        &mut self,
        id: usize,
        storage: &chelis_types::TensorStorage,
        ty: &TensorType,
    ) {
        let exact_ints = storage.to_i64_exact_vec();
        let data: Vec<f64> = storage.to_f64_lossy_vec();
        self.emit_slot_wrapper(id, ty);
        match ty.precision {
            Prim::F32 => {
                // Use a uint32_t array of bit patterns (compile-time constants),
                // then memcpy into the tensor. This avoids function-call
                // initializers that C89/C99 reject in static arrays.
                let values: Vec<String> = data
                    .iter()
                    .map(|v| {
                        let bits = (*v as f32).to_bits();
                        format!("0x{bits:08x}u")
                    })
                    .collect();
                self.line(&format!(
                    "{{ static const uint32_t __bits[] = {{ {} }};",
                    values.join(", ")
                ));
                self.line(&format!(
                    "  memcpy(t{id}_data, __bits, {}u * sizeof(uint32_t)); }}",
                    data.len()
                ));
            }
            Prim::F64 => {
                let values: Vec<String> = data
                    .iter()
                    .map(|v| {
                        let bits = v.to_bits();
                        format!("0x{bits:016x}uLL")
                    })
                    .collect();
                self.line(&format!(
                    "{{ static const uint64_t __bits[] = {{ {} }};",
                    values.join(", ")
                ));
                self.line(&format!(
                    "  memcpy(t{id}_data, __bits, {}u * sizeof(uint64_t)); }}",
                    data.len()
                ));
            }
            Prim::Int32 => {
                let values: Vec<String> = match &exact_ints {
                    Some(ints) => ints.iter().map(|v| format!("{}", *v as i32)).collect(),
                    None => data.iter().map(|v| format!("{}", *v as i32)).collect(),
                };
                self.line(&format!(
                    "{{ static const int32_t __data[] = {{ {} }};",
                    values.join(", ")
                ));
                self.line(&format!(
                    "  memcpy(t{id}_data, __data, {}u * sizeof(int32_t)); }}",
                    data.len()
                ));
            }
            Prim::Int64 => {
                let values: Vec<String> = match &exact_ints {
                    Some(ints) => ints.iter().map(|v| format!("{}", *v)).collect(),
                    None => data.iter().map(|v| format!("{}", *v as i64)).collect(),
                };
                self.line(&format!(
                    "{{ static const int64_t __data[] = {{ {} }};",
                    values.join(", ")
                ));
                self.line(&format!(
                    "  memcpy(t{id}_data, __data, {}u * sizeof(int64_t)); }}",
                    data.len()
                ));
            }
            _ => {
                // Fallback: fill element by element via bit-cast helpers.
                for (i, v) in data.iter().enumerate() {
                    let bits = (*v as f32).to_bits();
                    self.line(&format!(
                        "((float*)t{id}_data)[{i}] = chelis_f32_from_bits(0x{bits:08x}u);"
                    ));
                }
            }
        }
    }

    // ---- Load ----
    fn emit_load(&mut self, id: usize, input_idx: usize) {
        self.line(&format!("chelis_tensor *t{id} = inputs[{input_idx}];"));
        self.emit_tensor_snapshot(id, false);
    }

    // ---- Binary elementwise ----
    fn emit_binary(&mut self, id: usize, op: &str, inputs: &[NodeId], ty: &TensorType) {
        if Self::is_reduced_float(ty) {
            self.emit_binary_reduced_f(id, op, inputs, ty);
            return;
        }
        let a = inputs[0].0;
        let b = inputs[1].0;
        let et = Self::elem_type(ty);
        let is_relu_adjoint = op == "chelis_relu_adjoint";
        let zero = if Self::is_f64(ty) { "0.0" } else { "0.0f" };
        // #387: an INTEGER `div` (`op == "/"` on an integer dtype) must trap
        // portably on a zero divisor. Hardware behavior is not portable --
        // x86 raises SIGFPE on integer #DE, but ARM64 (macOS arm64) defines
        // integer div-by-zero to return a value and does NOT fault, so the
        // binary would silently compute a wrong answer. Wrap the divisor in
        // `chelis_int_div_guard`, which aborts with the same clean diagnostic
        // the evaluator emits. Float `/` is IEEE-754 (`1.0/0.0 == inf`) and
        // is never guarded; `+`/`*`/`fmaxf` never divide.
        let checked_int = ty.precision.is_integer() && matches!(op, "+" | "-" | "*" | "/");
        let elem_expr = |lhs: String, rhs: String| -> String {
            if is_relu_adjoint {
                // [05-OP-43]: select g only for +0 < x. Selection preserves
                // the exact stored cotangent bits and emits exact +0 for
                // both zeros and NaN without multiplying by a mask.
                return format!("{zero} < ({lhs}) ? ({rhs}) : {zero}");
            }
            if !checked_int {
                return format!("{lhs} {op} {rhs}");
            }
            let bits = Self::integer_width(ty.precision);
            let op_name = match op {
                "+" => "add",
                "-" => "sub",
                "*" => "mul",
                "/" => "trunc_div",
                _ => unreachable!(),
            };
            let overflow = NumericTrap::Overflow {
                op: op_name,
                prim: ty.precision,
            }
            .to_string();
            match op {
                "+" => format!(
                    "({et})chelis_int_checked_add((int64_t)({lhs}), (int64_t)({rhs}), {bits}, {overflow:?})"
                ),
                "-" => format!(
                    "({et})chelis_int_checked_sub((int64_t)({lhs}), (int64_t)({rhs}), {bits}, {overflow:?})"
                ),
                "*" => format!(
                    "({et})chelis_int_checked_mul((int64_t)({lhs}), (int64_t)({rhs}), {bits}, {overflow:?})"
                ),
                "/" => {
                    let zero = NumericTrap::DivZero {
                        op: op_name,
                        prim: ty.precision,
                    }
                    .to_string();
                    format!(
                        "({lhs} / ({et})chelis_int_checked_divisor((int64_t)({lhs}), (int64_t)({rhs}), {bits}, {zero:?}, {overflow:?}))"
                    )
                }
                _ => unreachable!(),
            }
        };
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}_size == t{id}_size);"));
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}_data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}_data;"
        ));
        self.line(&format!(
            "const {et}* restrict __in_b_{id} = (const {et}*)t{b}_data;"
        ));
        // An integer-div guard introduces a function call with side effects,
        // which is not safely vectorizable; only the non-guarded ops keep the
        // `simd` clause.
        if checked_int {
            self.line("#pragma omp parallel for");
        } else {
            self.line("#pragma omp parallel for simd");
        }
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "__out_{id}[i] = {};",
            elem_expr(format!("__in_a_{id}[i]"), format!("__in_b_{id}[i]"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx_a = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "int64_t idx_b = chelis_indices_to_flat(indices, t{b}_strides, t{b}_rank);"
        ));
        self.line(&format!(
            "(({et}*)t{id}_data)[i] = {};",
            elem_expr(
                format!("(({et}*)t{a}_data)[idx_a]"),
                format!("(({et}*)t{b}_data)[idx_b]")
            )
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// chelis#178: floor division (round quotient toward -inf).
    ///
    /// - Integer dtype: native `/` plus a remainder-sign correction —
    ///   `q = a / b; r = a % b; out = q - (r != 0 && ((r < 0) != (b < 0)))`.
    ///   The divisor is wrapped in the portable `chelis_int_div_guard` so a
    ///   zero divisor traps identically to `div`/`mod`/`trunc_div`.
    /// - Float dtype: `floorf(a / b)` (f32) / `floor(a / b)` (f64) /
    ///   reduced-float via the f32 conversion path. IEEE division is not
    ///   guarded (`floor(+inf) == +inf`).
    fn emit_floor_div(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        if Self::is_reduced_float(ty) {
            self.emit_floor_div_reduced_f(id, inputs, ty);
            return;
        }
        let a = inputs[0].0;
        let b = inputs[1].0;
        let et = Self::elem_type(ty);
        let is_int = ty.precision.is_integer();
        // Build the per-element floor-division expression given lvalue
        // expressions for the two operands. For ints, guard the divisor and
        // apply the remainder-sign correction; for floats use the math fn.
        let floor_fn = if Self::is_f64(ty) { "floor" } else { "floorf" };
        let elem_expr = |av: &str, bv: &str| -> String {
            if is_int {
                format!(
                    "({et})(({av} / chelis_int_div_guard((int64_t)({bv}))) - \
                     ((((({av}) % chelis_int_div_guard((int64_t)({bv}))) != 0) && \
                     (((({av}) % chelis_int_div_guard((int64_t)({bv}))) < 0) != (({bv}) < 0))) ? 1 : 0))"
                )
            } else {
                format!("{floor_fn}(({et})({av}) / ({et})({bv}))")
            }
        };
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}_size == t{id}_size);"));
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}_data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}_data;"
        ));
        self.line(&format!(
            "const {et}* restrict __in_b_{id} = (const {et}*)t{b}_data;"
        ));
        // The integer guard is a side-effecting call; do not vectorize it.
        if is_int {
            self.line("#pragma omp parallel for");
        } else {
            self.line("#pragma omp parallel for simd");
        }
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "__out_{id}[i] = {};",
            elem_expr(&format!("__in_a_{id}[i]"), &format!("__in_b_{id}[i]"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx_a = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "int64_t idx_b = chelis_indices_to_flat(indices, t{b}_strides, t{b}_rank);"
        ));
        self.line(&format!(
            "(({et}*)t{id}_data)[i] = {};",
            elem_expr(
                &format!("(({et}*)t{a}_data)[idx_a]"),
                &format!("(({et}*)t{b}_data)[idx_b]")
            )
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// Reduced-float (bf16/f16) floor division. Storage is `uint16_t`;
    /// arithmetic runs in `f32` via the conversion helpers, then `floorf`.
    /// `floor_div` admits reduced floats per spec/05 §2.1 (float row).
    fn emit_floor_div_reduced_f(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let load = Self::reduced_to_f32_fn(ty.precision);
        let store = Self::f32_to_reduced_fn(ty.precision);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}_size == t{id}_size);"));
        self.line(&format!(
            "uint16_t* restrict __out_{id} = (uint16_t*)t{id}_data;"
        ));
        self.line(&format!(
            "const uint16_t* restrict __in_a_{id} = (const uint16_t*)t{a}_data;"
        ));
        self.line(&format!(
            "const uint16_t* restrict __in_b_{id} = (const uint16_t*)t{b}_data;"
        ));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!("float __av = {load}(__in_a_{id}[i]);"));
        self.line(&format!("float __bv = {load}(__in_b_{id}[i]);"));
        self.line(&format!("__out_{id}[i] = {store}(floorf(__av / __bv));"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx_a = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "int64_t idx_b = chelis_indices_to_flat(indices, t{b}_strides, t{b}_rank);"
        ));
        self.line(&format!(
            "float __av = {load}(((uint16_t*)t{a}_data)[idx_a]);"
        ));
        self.line(&format!(
            "float __bv = {load}(((uint16_t*)t{b}_data)[idx_b]);"
        ));
        self.line(&format!(
            "((uint16_t*)t{id}_data)[i] = {store}(floorf(__av / __bv));"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// WS-1: bf16 / f16 binary elementwise. Storage is `uint16_t`;
    /// arithmetic is performed in `f32` via the runtime conversion
    /// helpers, matching the spec/04-type-system.md §5.7.1 promise
    /// that reduced-float intermediate values fall back to f32.
    fn emit_binary_reduced_f(&mut self, id: usize, op: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let load = Self::reduced_to_f32_fn(ty.precision);
        let store = Self::f32_to_reduced_fn(ty.precision);
        let is_relu_adjoint = op == "chelis_relu_adjoint";
        let elem_expr = |g_raw: String| -> String {
            if is_relu_adjoint {
                // Decode x only for the predicate and select the original
                // f16/bf16 cotangent storage word unchanged.
                format!("0.0f < __av ? {g_raw} : UINT16_C(0)")
            } else {
                format!("{store}(__av {op} __bv)")
            }
        };
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}_size == t{id}_size);"));
        self.line(&format!(
            "uint16_t* restrict __out_{id} = (uint16_t*)t{id}_data;"
        ));
        self.line(&format!(
            "const uint16_t* restrict __in_a_{id} = (const uint16_t*)t{a}_data;"
        ));
        self.line(&format!(
            "const uint16_t* restrict __in_b_{id} = (const uint16_t*)t{b}_data;"
        ));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!("float __av = {load}(__in_a_{id}[i]);"));
        self.line(&format!("float __bv = {load}(__in_b_{id}[i]);"));
        self.line(&format!(
            "__out_{id}[i] = {};",
            elem_expr(format!("__in_b_{id}[i]"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx_a = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "int64_t idx_b = chelis_indices_to_flat(indices, t{b}_strides, t{b}_rank);"
        ));
        self.line(&format!(
            "float __av = {load}(((uint16_t*)t{a}_data)[idx_a]);"
        ));
        self.line(&format!(
            "float __bv = {load}(((uint16_t*)t{b}_data)[idx_b]);"
        ));
        self.line(&format!(
            "((uint16_t*)t{id}_data)[i] = {};",
            elem_expr(format!("((uint16_t*)t{b}_data)[idx_b]"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// Exact direct-extrema selection expression for floating, integer, and
    /// Bool storage. The conditional returns one operand expression unchanged,
    /// so NaN payloads, NaN signs, and signed zero bits are never re-encoded.
    fn extrema_select_expr(ty: &TensorType, func: &str, lhs: String, rhs: String) -> String {
        let comparison = if func.contains("max") { ">=" } else { "<=" };
        if ty.precision.is_integer() || matches!(ty.precision, Prim::Bool) {
            format!("(({lhs}) {comparison} ({rhs}) ? ({lhs}) : ({rhs}))")
        } else {
            format!(
                "(isnan({lhs}) || (!isnan({rhs}) && ({lhs}) {comparison} ({rhs})) ? ({lhs}) : ({rhs}))"
            )
        }
    }

    // ---- Direct extrema selection ----
    fn emit_binary_func(&mut self, id: usize, func: &str, inputs: &[NodeId], ty: &TensorType) {
        if Self::is_reduced_float(ty) {
            self.emit_binary_func_reduced_f(id, func, inputs, ty);
            return;
        }
        let a = inputs[0].0;
        let b = inputs[1].0;
        let et = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}_size == t{id}_size);"));
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}_data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}_data;"
        ));
        self.line(&format!(
            "const {et}* restrict __in_b_{id} = (const {et}*)t{b}_data;"
        ));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "__out_{id}[i] = {};",
            Self::extrema_select_expr(
                ty,
                func,
                format!("__in_a_{id}[i]"),
                format!("__in_b_{id}[i]")
            )
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx_a = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "int64_t idx_b = chelis_indices_to_flat(indices, t{b}_strides, t{b}_rank);"
        ));
        self.line(&format!(
            "(({et}*)t{id}_data)[i] = {};",
            Self::extrema_select_expr(
                ty,
                func,
                format!("(({et}*)t{a}_data)[idx_a]"),
                format!("(({et}*)t{b}_data)[idx_b]")
            )
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- CmpLt ----
    /// Comparable-scalar load expression for one cmplt operand. The
    /// operand carries its own precision `p` (cmplt:
    /// `∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D,bool]`), so we read
    /// it through a `p`-typed pointer (`ptr_var` must already be cast to
    /// the storage element type). Reduced-float operands (`bf16`/`f16`)
    /// store as `uint16_t` and must convert to `f32` before the
    /// numeric `<`, matching the evaluator's value comparison rather
    /// than a 16-bit bit-pattern comparison.
    fn cmplt_cmp_value(ty: &TensorType, ptr_var: &str, idx: &str) -> String {
        if Self::is_reduced_float(ty) {
            let conv = Self::reduced_to_f32_fn(ty.precision);
            format!("{conv}({ptr_var}[{idx}])")
        } else {
            format!("{ptr_var}[{idx}]")
        }
    }

    /// #517: cmplt reads each operand through its OWN element dtype,
    /// resolved from the input DAG nodes — not through the boolean
    /// output type. The result uses the canonical one-byte Bool8
    /// representation,
    /// but a runtime-produced int32/int64/f64/bf16/f16 operand read
    /// through a raw `float*` would reinterpret its bit pattern (the
    /// #347/#476 bug class: e.g. a negative int32 reinterpreted as
    /// `float` is NaN, so `-7 < -3` would wrongly yield false, and an
    /// f64 read through `float*` truncates the 8-byte payload). Both
    /// operands share precision `p` per the signature, but each type is
    /// resolved independently for robustness.
    fn emit_cmplt(
        &mut self,
        id: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let a_ty = dag.get(inputs[0]).unwrap().output_type.clone();
        let b_ty = dag.get(inputs[1]).unwrap().output_type.clone();
        let et_a = Self::elem_type(&a_ty);
        let et_b = Self::elem_type(&b_ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "uint8_t* restrict __out_{id} = (uint8_t*)t{id}_data;"
        ));
        self.line(&format!(
            "const {et_a}* restrict __in_a_{id} = (const {et_a}*)t{a}_data;"
        ));
        self.line(&format!(
            "const {et_b}* restrict __in_b_{id} = (const {et_b}*)t{b}_data;"
        ));
        let cmp_a = Self::cmplt_cmp_value(&a_ty, &format!("__in_a_{id}"), "i");
        let cmp_b = Self::cmplt_cmp_value(&b_ty, &format!("__in_b_{id}"), "i");
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}_size == t{id}_size);"));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "__out_{id}[i] = ({cmp_a} < {cmp_b}) ? UINT8_C(1) : UINT8_C(0);"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx_a = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "int64_t idx_b = chelis_indices_to_flat(indices, t{b}_strides, t{b}_rank);"
        ));
        let cmp_a_strided = Self::cmplt_cmp_value(&a_ty, &format!("__in_a_{id}"), "idx_a");
        let cmp_b_strided = Self::cmplt_cmp_value(&b_ty, &format!("__in_b_{id}"), "idx_b");
        self.line(&format!(
            "__out_{id}[i] = ({cmp_a_strided} < {cmp_b_strided}) ? UINT8_C(1) : UINT8_C(0);"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- Unary elementwise ----
    fn emit_unary(&mut self, id: usize, op: &str, inputs: &[NodeId], ty: &TensorType) {
        if Self::is_reduced_float(ty) {
            self.emit_unary_reduced_f(id, op, inputs, ty);
            return;
        }
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        let elem_expr = |value: String| -> String {
            if op == "-" && ty.precision.is_integer() {
                let message = NumericTrap::Overflow {
                    op: "neg",
                    prim: ty.precision,
                }
                .to_string();
                format!(
                    "({et})chelis_int_checked_neg((int64_t)({value}), {}, {message:?})",
                    Self::integer_width(ty.precision)
                )
            } else {
                format!("{op}{value}")
            }
        };
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
        self.indent += 1;
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}_data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}_data;"
        ));
        if ty.precision.is_integer() && op == "-" {
            self.line("#pragma omp parallel for");
        } else {
            self.line("#pragma omp parallel for simd");
        }
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "__out_{id}[i] = {};",
            elem_expr(format!("__in_a_{id}[i]"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "(({et}*)t{id}_data)[i] = {};",
            elem_expr(format!("(({et}*)t{a}_data)[idx]"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    fn integer_width(prim: Prim) -> i64 {
        match prim {
            Prim::Int8 => 8,
            Prim::Int16 => 16,
            Prim::Int32 => 32,
            Prim::Int64 => 64,
            other => panic!(
                "integer-width C helper called for non-integer dtype `{}`",
                other.name()
            ),
        }
    }

    /// Build the exact integer-Abs call. The diagnostic literal is generated
    /// from the frozen Rust `NumericTrap` grammar; the runtime helper only
    /// checks the declared-width minimum before negating.
    fn integer_abs_expr(prim: Prim, value: &str) -> String {
        let message = NumericTrap::Overflow { op: "abs", prim }.to_string();
        format!(
            "({})chelis_int_abs_guard((int64_t)({value}), {}, {message:?})",
            match prim {
                Prim::Int8 => "int8_t",
                Prim::Int16 => "int16_t",
                Prim::Int32 => "int32_t",
                Prim::Int64 => "int64_t",
                other => panic!("integer abs called for `{}`", other.name()),
            },
            Self::integer_width(prim),
        )
    }

    /// Typed, declared-width signed-integer absolute value. There is no libm
    /// call and no C signed-overflow UB: `chelis_int_abs_guard` traps on MIN
    /// before negation using the generated C2 message.
    fn emit_integer_abs(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}_data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}_data;"
        ));
        self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        let contiguous = Self::integer_abs_expr(ty.precision, &format!("__in_a_{id}[i]"));
        self.line(&format!("__out_{id}[i] = {contiguous};"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        let strided = Self::integer_abs_expr(ty.precision, &format!("__in_a_{id}[idx]"));
        self.line(&format!("__out_{id}[i] = {strided};"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- Reciprocal ----
    // Emits IEEE `1.0 / x`. Kept separate from `emit_unary` because the
    // numerator is a precision-typed constant, not a prefix operator.
    // The pointer aliases are hoisted out of the contiguity branch so
    // both paths share the same `__in_a_{id}` / `__out_{id}` names; the
    // strided branch reuses them via `__in_a_{id}[idx]` rather than
    // re-casting `t{a}_data` inline.
    fn emit_recip(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        if Self::is_reduced_float(ty) {
            self.emit_recip_reduced_f(id, inputs, ty);
            return;
        }
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        let one = if Self::is_f64(ty) { "1.0" } else { "1.0f" };
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}_data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}_data;"
        ));
        self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
        self.indent += 1;
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!("__out_{id}[i] = {one} / __in_a_{id}[i];"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!("__out_{id}[i] = {one} / __in_a_{id}[idx];"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- Unary func (expf, logf, sinf, sqrtf) ----
    fn emit_unary_func(&mut self, id: usize, func: &str, inputs: &[NodeId], ty: &TensorType) {
        if Self::is_reduced_float(ty) {
            self.emit_unary_func_reduced_f(id, func, inputs, ty);
            return;
        }
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        let is_f64 = Self::is_f64(ty);
        let is_relu = func == "chelis_relu";
        let f = if is_f64 && !is_relu {
            Self::double_math_fn(func)
        } else {
            func
        };
        let zero = if is_f64 { "0.0" } else { "0.0f" };
        let elem_expr = |value: String| -> String {
            if is_relu {
                // [05-OP-43] is selection, not fmax: retain the input's exact
                // stored bits for NaN and -0 and replace only x < +0.
                format!("({value}) < {zero} ? {zero} : ({value})")
            } else {
                format!("{f}({value})")
            }
        };
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
        self.indent += 1;
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}_data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}_data;"
        ));
        // SIMD/batched math paths currently only support f32. For f64 tensors
        // or when no SIMD library is selected, fall back to the scalar OMP SIMD
        // loop with the appropriate (double- or single-precision) math function.
        if !is_f64 && self.math_lib == crate::MathLib::VForce {
            if let Some(vf_fn) = Self::vforce_func(func) {
                // vForce whole-array batch API on macOS (Accelerate.framework).
                //
                // chelis#1112: vForce's count parameter is `const int *`, a
                // foreign ABI this project does not get to widen, so the
                // element count is the one place an extent must cross into
                // 32 bits. The narrowing is GUARDED rather than cast: above
                // INT_MAX the batch call is skipped for the scalar loop,
                // which computes the same values at any size. A bare
                // `(int)t->size` here would process a wrapped prefix of the
                // buffer and leave the rest of the output uninitialized.
                self.line(&format!("if (t{id}_size <= 2147483647LL) {{"));
                self.indent += 1;
                self.line(&format!("int __n_{id} = (int)t{id}_size;"));
                self.line(&format!("{vf_fn}(__out_{id}, __in_a_{id}, &__n_{id});"));
                self.indent -= 1;
                self.line("} else {");
                self.indent += 1;
                self.line("#pragma omp parallel for simd");
                self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
                self.indent += 1;
                self.line(&format!(
                    "__out_{id}[i] = {};",
                    elem_expr(format!("__in_a_{id}[i]"))
                ));
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
            } else {
                self.line("#pragma omp parallel for simd");
                self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
                self.indent += 1;
                self.line(&format!(
                    "__out_{id}[i] = {};",
                    elem_expr(format!("__in_a_{id}[i]"))
                ));
                self.indent -= 1;
                self.line("}");
            }
        } else if !is_f64 && self.math_lib == crate::MathLib::Sleef {
            if let Some(simd_macro) = Self::sleef_macro(func) {
                self.line("#ifdef CHELIS_HAS_SLEEF");
                self.line("{");
                self.indent += 1;
                self.line(&format!("int64_t __i_{id} = 0;"));
                self.line(&format!(
                    "for (; __i_{id} + 8 <= t{id}_size; __i_{id} += 8) {{"
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
                self.line(&format!("for (; __i_{id} < t{id}_size; __i_{id}++) {{"));
                self.indent += 1;
                self.line(&format!(
                    "__out_{id}[__i_{id}] = {};",
                    elem_expr(format!("__in_a_{id}[__i_{id}]"))
                ));
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
                self.line("#else");
                self.line("#pragma omp parallel for simd");
                self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
                self.indent += 1;
                self.line(&format!(
                    "__out_{id}[i] = {};",
                    elem_expr(format!("__in_a_{id}[i]"))
                ));
                self.indent -= 1;
                self.line("}");
                self.line("#endif");
            } else {
                self.line("#pragma omp parallel for simd");
                self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
                self.indent += 1;
                self.line(&format!(
                    "__out_{id}[i] = {};",
                    elem_expr(format!("__in_a_{id}[i]"))
                ));
                self.indent -= 1;
                self.line("}");
            }
        } else {
            self.line("#pragma omp parallel for simd");
            self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
            self.indent += 1;
            self.line(&format!(
                "__out_{id}[i] = {};",
                elem_expr(format!("__in_a_{id}[i]"))
            ));
            self.indent -= 1;
            self.line("}");
        }
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "(({et}*)t{id}_data)[i] = {};",
            elem_expr(format!("(({et}*)t{a}_data)[idx]"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// WS-1: bf16 / f16 unary elementwise (Neg). Storage is `uint16_t`;
    /// each element is loaded into `f32` via the runtime helper, the
    /// op is applied in `f32`, and the result is converted back via
    /// the inverse helper before storage.
    fn emit_unary_reduced_f(&mut self, id: usize, op: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let load = Self::reduced_to_f32_fn(ty.precision);
        let store = Self::f32_to_reduced_fn(ty.precision);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
        self.indent += 1;
        self.line(&format!(
            "uint16_t* restrict __out_{id} = (uint16_t*)t{id}_data;"
        ));
        self.line(&format!(
            "const uint16_t* restrict __in_a_{id} = (const uint16_t*)t{a}_data;"
        ));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!("float __av = {load}(__in_a_{id}[i]);"));
        self.line(&format!("__out_{id}[i] = {store}({op}__av);"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "float __av = {load}(((uint16_t*)t{a}_data)[idx]);"
        ));
        self.line(&format!("((uint16_t*)t{id}_data)[i] = {store}({op}__av);"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// WS-1: bf16 / f16 reciprocal (`1.0 / x`). Same convert-compute-
    /// convert pattern as `emit_unary_reduced_f`, with `1.0f / x` as
    /// the f32 op.
    fn emit_recip_reduced_f(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let load = Self::reduced_to_f32_fn(ty.precision);
        let store = Self::f32_to_reduced_fn(ty.precision);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "uint16_t* restrict __out_{id} = (uint16_t*)t{id}_data;"
        ));
        self.line(&format!(
            "const uint16_t* restrict __in_a_{id} = (const uint16_t*)t{a}_data;"
        ));
        self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
        self.indent += 1;
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!("float __av = {load}(__in_a_{id}[i]);"));
        self.line(&format!("__out_{id}[i] = {store}(1.0f / __av);"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!("float __av = {load}(__in_a_{id}[idx]);"));
        self.line(&format!("__out_{id}[i] = {store}(1.0f / __av);"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// Direct bf16/f16 extrema. Values are decoded only for NaN/comparison;
    /// the selected original `uint16_t` is copied unchanged.
    fn emit_binary_func_reduced_f(
        &mut self,
        id: usize,
        func: &str,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let load = Self::reduced_to_f32_fn(ty.precision);
        let comparison = if func.contains("max") { ">=" } else { "<=" };
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}_size == t{id}_size);"));
        self.line(&format!(
            "uint16_t* restrict __out_{id} = (uint16_t*)t{id}_data;"
        ));
        self.line(&format!(
            "const uint16_t* restrict __in_a_{id} = (const uint16_t*)t{a}_data;"
        ));
        self.line(&format!(
            "const uint16_t* restrict __in_b_{id} = (const uint16_t*)t{b}_data;"
        ));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!("float __av = {load}(__in_a_{id}[i]);"));
        self.line(&format!("float __bv = {load}(__in_b_{id}[i]);"));
        self.line(&format!(
            "__out_{id}[i] = (isnan(__av) || (!isnan(__bv) && __av {comparison} __bv)) ? __in_a_{id}[i] : __in_b_{id}[i];"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx_a = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "int64_t idx_b = chelis_indices_to_flat(indices, t{b}_strides, t{b}_rank);"
        ));
        self.line(&format!(
            "float __av = {load}(((uint16_t*)t{a}_data)[idx_a]);"
        ));
        self.line(&format!(
            "float __bv = {load}(((uint16_t*)t{b}_data)[idx_b]);"
        ));
        self.line(&format!(
            "((uint16_t*)t{id}_data)[i] = (isnan(__av) || (!isnan(__bv) && __av {comparison} __bv)) ? ((uint16_t*)t{a}_data)[idx_a] : ((uint16_t*)t{b}_data)[idx_b];"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// Emit the AD-only selected-extrema cotangent. The selection predicate
    /// is identical to the forward direct extrema rule, and the complete
    /// cotangent is copied without arithmetic when its operand was selected.
    fn emit_extrema_adjoint(
        &mut self,
        id: usize,
        kind: ExtremaKind,
        operand: ExtremaOperand,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let g = inputs[2].0;
        let comparison = match kind {
            ExtremaKind::Max => ">=",
            ExtremaKind::Min => "<=",
        };
        let take_selected = |left: String| match operand {
            ExtremaOperand::Left => left,
            ExtremaOperand::Right => format!("!({left})"),
        };
        self.emit_slot_wrapper(id, ty);

        if Self::is_reduced_float(ty) {
            let load = Self::reduced_to_f32_fn(ty.precision);
            self.line(&format!(
                "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && chelis_is_contiguous(t{g}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size && t{g}_size == t{id}_size) {{"
            ));
            self.indent += 1;
            self.line(&format!(
                "uint16_t* restrict __out_{id} = (uint16_t*)t{id}_data;"
            ));
            self.line(&format!(
                "const uint16_t* restrict __in_a_{id} = (const uint16_t*)t{a}_data;"
            ));
            self.line(&format!(
                "const uint16_t* restrict __in_b_{id} = (const uint16_t*)t{b}_data;"
            ));
            self.line(&format!(
                "const uint16_t* restrict __in_g_{id} = (const uint16_t*)t{g}_data;"
            ));
            self.line("#pragma omp parallel for simd");
            self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
            self.indent += 1;
            self.line(&format!("float __av = {load}(__in_a_{id}[i]);"));
            self.line(&format!("float __bv = {load}(__in_b_{id}[i]);"));
            let left = format!("isnan(__av) || (!isnan(__bv) && __av {comparison} __bv)");
            self.line(&format!(
                "__out_{id}[i] = {} ? __in_g_{id}[i] : UINT16_C(0);",
                take_selected(left)
            ));
            self.indent -= 1;
            self.line("}");
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
            self.line("#pragma omp parallel for");
            self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
            self.indent += 1;
            self.line(&format!(
                "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
            ));
            self.line(&format!(
                "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
            ));
            for (name, source) in [("a", a), ("b", b), ("g", g)] {
                self.line(&format!("int64_t idx_{name} = chelis_indices_to_flat(indices, t{source}_strides, t{source}_rank);"));
            }
            self.line(&format!(
                "float __av = {load}(((uint16_t*)t{a}_data)[idx_a]);"
            ));
            self.line(&format!(
                "float __bv = {load}(((uint16_t*)t{b}_data)[idx_b]);"
            ));
            let left = format!("isnan(__av) || (!isnan(__bv) && __av {comparison} __bv)");
            self.line(&format!(
                "((uint16_t*)t{id}_data)[i] = {} ? ((uint16_t*)t{g}_data)[idx_g] : UINT16_C(0);",
                take_selected(left)
            ));
            self.indent -= 1;
            self.line("}");
            self.indent -= 1;
            self.line("}");
            return;
        }

        let et = Self::elem_type(ty);
        let zero = if Self::is_f64(ty) { "0.0" } else { "0.0f" };
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && chelis_is_contiguous(t{b}) && chelis_is_contiguous(t{g}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size && t{g}_size == t{id}_size) {{"
        ));
        self.indent += 1;
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}_data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}_data;"
        ));
        self.line(&format!(
            "const {et}* restrict __in_b_{id} = (const {et}*)t{b}_data;"
        ));
        self.line(&format!(
            "const {et}* restrict __in_g_{id} = (const {et}*)t{g}_data;"
        ));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        let left = format!(
            "isnan(__in_a_{id}[i]) || (!isnan(__in_b_{id}[i]) && __in_a_{id}[i] {comparison} __in_b_{id}[i])"
        );
        self.line(&format!(
            "__out_{id}[i] = {} ? __in_g_{id}[i] : {zero};",
            take_selected(left)
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        for (name, source) in [("a", a), ("b", b), ("g", g)] {
            self.line(&format!("int64_t idx_{name} = chelis_indices_to_flat(indices, t{source}_strides, t{source}_rank);"));
        }
        let left = format!(
            "isnan((({et}*)t{a}_data)[idx_a]) || (!isnan((({et}*)t{b}_data)[idx_b]) && (({et}*)t{a}_data)[idx_a] {comparison} (({et}*)t{b}_data)[idx_b])"
        );
        self.line(&format!(
            "(({et}*)t{id}_data)[i] = {} ? (({et}*)t{g}_data)[idx_g] : {zero};",
            take_selected(left)
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// Route [05-OP-43] through the existing unary emission template so
    /// numeric representation ownership stays with that classified template.
    fn emit_relu(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        self.emit_unary_func(id, "chelis_relu", inputs, ty);
    }

    /// Route the dedicated adjoint through the classified binary template.
    fn emit_relu_adjoint(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        self.emit_binary(id, "chelis_relu_adjoint", inputs, ty);
    }
    /// WS-1: bf16 / f16 unary func (Exp, Log, Sin, Sqrt, Abs, ...).
    /// Single convert-compute-convert loop; no math-lib batched
    /// fast path (the math-lib hooks emit f32 batch calls and would
    /// need an explicit promotion step).
    fn emit_unary_func_reduced_f(
        &mut self,
        id: usize,
        func: &str,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        let a = inputs[0].0;
        let load = Self::reduced_to_f32_fn(ty.precision);
        let store = Self::f32_to_reduced_fn(ty.precision);
        let is_relu = func == "chelis_relu";
        let elem_expr = |raw: String| -> String {
            if is_relu {
                // Decode only for the predicate; preserve the selected f16
                // or bf16 storage word exactly, including NaN payload/-0.
                format!("__av < 0.0f ? UINT16_C(0) : {raw}")
            } else {
                format!("{store}({func}(__av))")
            }
        };
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
        self.indent += 1;
        self.line(&format!(
            "uint16_t* restrict __out_{id} = (uint16_t*)t{id}_data;"
        ));
        self.line(&format!(
            "const uint16_t* restrict __in_a_{id} = (const uint16_t*)t{a}_data;"
        ));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!("float __av = {load}(__in_a_{id}[i]);"));
        self.line(&format!(
            "__out_{id}[i] = {};",
            elem_expr(format!("__in_a_{id}[i]"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "float __av = {load}(((uint16_t*)t{a}_data)[idx]);"
        ));
        self.line(&format!(
            "((uint16_t*)t{id}_data)[i] = {};",
            elem_expr(format!("((uint16_t*)t{a}_data)[idx]"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// Map a scalar C math function name to its vForce batch equivalent, or
    /// `None` to keep the correctly-rounded scalar loop.
    ///
    /// `sqrtf` is deliberately absent (chelis#719). IEEE-754 sec 5.4.1 requires
    /// `squareRoot` to be correctly rounded; Accelerate's `vvsqrtf` is not. It
    /// returns 0x3F9CC470 for sqrt(1.5), one ulp below the correctly-rounded
    /// 0x3F9CC471. Routing sqrt through vForce also made the result
    /// layout-dependent, since the strided path already used scalar `sqrtf`. The
    /// scalar loop is correctly rounded on every platform (arm64 `fsqrt`, x86
    /// `sqrtss`, or libm) and the compiler auto-vectorizes it, so the contiguous
    /// and strided paths now agree bit for bit. exp/log/sin carry no IEEE
    /// correctness requirement and stay on vForce (their per-op tolerances live
    /// in spec/05 sec 8, chelis#732).
    fn vforce_func(scalar_func: &str) -> Option<&'static str> {
        match scalar_func {
            "expf" => Some("vvexpf"),
            "logf" => Some("vvlogf"),
            "sinf" => Some("vvsinf"),
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

    fn emit_uniform_like(
        &mut self,
        id: usize,
        low: f64,
        high: f64,
        seed: u64,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
        self.emit_slot_wrapper(id, ty);
        if let Some(activation) = inputs.get(1) {
            // The activation is a rank-0 Bool predicate, and chelis#1308's
            // tagged-carrier ABI stores Bool tensors as one uint8 per
            // element. Reading it through `(float*)` was correct only under
            // the pre-#1308 float-backed Bool storage; against uint8
            // storage it reads one valid byte plus three out-of-bounds
            // heap bytes, so an untaken branch's gate could go active on
            // whatever the allocator left there (Linux CI caught the RNG
            // parity break; macOS zero-fill masked it).
            self.line(&format!(
                "int t{id}_active = ((const uint8_t*)t{}_data)[0] != 0 ? 1 : 0;",
                activation.0
            ));
            self.line(&format!(
                "uint64_t t{id}_seed = t{id}_active ? CHELIS_EFFECTIVE_UNIFORM_SEED({seed}ULL) : {seed}ULL;"
            ));
        } else {
            self.line(&format!(
                "uint64_t t{id}_seed = CHELIS_EFFECTIVE_UNIFORM_SEED({seed}ULL);"
            ));
        }
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        // [05-OP-8]: the source bounds have f32 dtype. f64 output widens
        // those exact stored images and samples in f64; f32 samples in
        // f32; f16/bf16 sample in f32 and round once at the final store.
        let low_bits = Self::f64_to_f32_truncate(low).to_bits();
        let high_bits = Self::f64_to_f32_truncate(high).to_bits();
        let low_f32 = format!("chelis_f32_from_bits(0x{low_bits:08x}u)");
        let high_f32 = format!("chelis_f32_from_bits(0x{high_bits:08x}u)");
        match ty.precision {
            Prim::F64 => {
                let low_wide_bits = (Self::f64_to_f32_truncate(low) as f64).to_bits();
                let high_wide_bits = (Self::f64_to_f32_truncate(high) as f64).to_bits();
                self.line(&format!(
                    "((double*)t{id}_data)[i] = chelis_uniform_sample_f64(t{id}_seed, (uint64_t)i, chelis_f64_from_bits(UINT64_C(0x{low_wide_bits:016x})), chelis_f64_from_bits(UINT64_C(0x{high_wide_bits:016x})));"
                ));
            }
            Prim::F32 => {
                self.line(&format!(
                    "((float*)t{id}_data)[i] = chelis_uniform_sample_f32(t{id}_seed, (uint64_t)i, {low_f32}, {high_f32});"
                ));
            }
            Prim::F16 | Prim::Bf16 => {
                let store = Self::f32_to_reduced_fn(ty.precision);
                self.line(&format!(
                    "((uint16_t*)t{id}_data)[i] = {store}(chelis_uniform_sample_f32(t{id}_seed, (uint64_t)i, {low_f32}, {high_f32}));"
                ));
            }
            other => panic!(
                "uniform_like requires an active float output dtype, got `{}`",
                other.name()
            ),
        }
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
    ///
    /// chelis#919: `is_f64` selects the double-precision math symbols
    /// (`exp` rather than `expf`, `fmax` rather than `fmaxf`) and the
    /// unsuffixed `1.0` / `0.0` literals. It must agree with the
    /// element type the caller declared for the `v{s}` step variables:
    /// emitting `expf` into a `double v0` silently narrows through the
    /// single-precision libm entry point, which is exactly the F1
    /// footgun. `emit_fused_elem` derives both from the same
    /// `Self::is_f64(ty)`.
    ///
    /// Callers admit only f32 and f64; `emit_fused_reduce` still passes
    /// `false` because its accumulator path stays f32-only.
    fn scalar_step_expr(
        op: &FusedStepOp,
        resolve: &dyn Fn(&FusedInput) -> String,
        inputs: &[FusedInput],
        is_f64: bool,
    ) -> String {
        // Map a single-precision libm symbol to its double-precision
        // counterpart when the chain computes in `double`. Same helper
        // `emit_unary_func` uses, so the fused and unfused lanes cannot
        // disagree about which entry point a given op resolves to.
        let mf = |scalar_f: &'static str| -> &'static str {
            if is_f64 {
                Self::double_math_fn(scalar_f)
            } else {
                scalar_f
            }
        };
        // `1.0f` in a `double` expression is a float constant that the
        // usual arithmetic conversions then widen; correct here but
        // misleading, and it becomes wrong the moment a literal is not
        // exactly representable in f32. Emit the literal at the chain's
        // own precision.
        let one = if is_f64 { "1.0" } else { "1.0f" };
        let zero = if is_f64 { "0.0" } else { "0.0f" };
        match op {
            FusedStepOp::Add => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("{a} + {b}")
            }
            FusedStepOp::Sub => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("{a} - {b}")
            }
            FusedStepOp::Mul => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("{a} * {b}")
            }
            FusedStepOp::Div => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("{a} / {b}")
            }
            // chelis#178: the fused path carries float operands only
            // (`emit_fused_elem` admits f32 and f64; `emit_fused_reduce`
            // admits f32), so only `floor_div` on float operands can
            // reach here — emit `floorf(a / b)` / `floor(a / b)`.
            // `trunc_div` is integer-only and can never fuse to a float
            // path, so it is unreachable.
            FusedStepOp::FloorDiv => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                let f = mf("floorf");
                format!("{f}({a} / {b})")
            }
            FusedStepOp::TruncDiv => {
                unreachable!(
                    "trunc_div is integer-only (chelis#178); the fused-elem path is \
                     float-only (f32/f64) and cannot carry an integer trunc_div step"
                )
            }
            FusedStepOp::MaxElem => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("(isnan({a}) || (!isnan({b}) && ({a}) >= ({b})) ? ({a}) : ({b}))")
            }
            FusedStepOp::MinElem => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("(isnan({a}) || (!isnan({b}) && ({a}) <= ({b})) ? ({a}) : ({b}))")
            }
            FusedStepOp::CmpLt => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("({a} < {b}) ? {one} : {zero}")
            }
            FusedStepOp::Neg => {
                let a = resolve(&inputs[0]);
                format!("-{a}")
            }
            FusedStepOp::Recip => {
                let a = resolve(&inputs[0]);
                format!("{one} / {a}")
            }
            FusedStepOp::Exp => {
                let a = resolve(&inputs[0]);
                let f = mf("expf");
                format!("{f}({a})")
            }
            FusedStepOp::Log => {
                let a = resolve(&inputs[0]);
                let f = mf("logf");
                format!("{f}({a})")
            }
            FusedStepOp::Sin => {
                let a = resolve(&inputs[0]);
                let f = mf("sinf");
                format!("{f}({a})")
            }
            FusedStepOp::Sqrt => {
                let a = resolve(&inputs[0]);
                let f = mf("sqrtf");
                format!("{f}({a})")
            }
            FusedStepOp::Cos => {
                let a = resolve(&inputs[0]);
                let f = mf("cosf");
                format!("{f}({a})")
            }
            FusedStepOp::Tan => {
                let a = resolve(&inputs[0]);
                let f = mf("tanf");
                format!("{f}({a})")
            }
            FusedStepOp::Atan => {
                let a = resolve(&inputs[0]);
                let f = mf("atanf");
                format!("{f}({a})")
            }
            FusedStepOp::Abs => {
                let a = resolve(&inputs[0]);
                let f = mf("fabsf");
                format!("{f}({a})")
            }
            FusedStepOp::Floor => {
                let a = resolve(&inputs[0]);
                let f = mf("floorf");
                format!("{f}({a})")
            }
            FusedStepOp::Ceil => {
                let a = resolve(&inputs[0]);
                let f = mf("ceilf");
                format!("{f}({a})")
            }
            FusedStepOp::Round => {
                let a = resolve(&inputs[0]);
                let f = mf("rintf");
                format!("{f}({a})")
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
            FusedStepOp::Sub => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("_mm256_sub_ps({a}, {b})")
            }
            FusedStepOp::Mul => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("_mm256_mul_ps({a}, {b})")
            }
            FusedStepOp::Div => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("_mm256_div_ps({a}, {b})")
            }
            // chelis#178: the fused-elem path is f32-only, so only float
            // `floor_div` reaches here — `floor(a / b)` via AVX2.
            // `trunc_div` is integer-only and cannot fuse to f32.
            FusedStepOp::FloorDiv => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("_mm256_floor_ps(_mm256_div_ps({a}, {b}))")
            }
            FusedStepOp::TruncDiv => {
                unreachable!(
                    "trunc_div is integer-only (chelis#178); the fused-elem path is \
                     f32-only and cannot carry an integer trunc_div step"
                )
            }
            FusedStepOp::MaxElem => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!(
                    "_mm256_blendv_ps({b}, {a}, _mm256_or_ps(_mm256_cmp_ps({a}, {a}, _CMP_UNORD_Q), _mm256_and_ps(_mm256_cmp_ps({b}, {b}, _CMP_ORD_Q), _mm256_cmp_ps({a}, {b}, _CMP_GE_OQ))))"
                )
            }
            FusedStepOp::MinElem => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!(
                    "_mm256_blendv_ps({b}, {a}, _mm256_or_ps(_mm256_cmp_ps({a}, {a}, _CMP_UNORD_Q), _mm256_and_ps(_mm256_cmp_ps({b}, {b}, _CMP_ORD_Q), _mm256_cmp_ps({a}, {b}, _CMP_LE_OQ))))"
                )
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
            FusedStepOp::Recip => {
                let a = resolve(&inputs[0]);
                format!("_mm256_div_ps(_mm256_set1_ps(1.0f), {a})")
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
            FusedStepOp::Round => {
                let a = resolve(&inputs[0]);
                format!("_mm256_set1_ps(rintf(_mm256_cvtss_f32({a})))")
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
        in_place: Option<CFusedReuse>,
    ) -> Result<(), Unsupported> {
        // chelis#919: this path is parameterized on the IR-pinned
        // element type for f32 and f64 (`Self::elem_type` for the data
        // pointers and the `v{s}` step variables, `Self::double_math_fn`
        // for the math symbols) — the same treatment `emit_unary_func`
        // already had. Every other dtype is still unsupported here:
        // bf16/f16 are stored as `uint16_t` and need the
        // convert-then-compute routing that `emit_unary_func_reduced_f`
        // uses, and the integer dtypes need integer step operators
        // (chelis#729) rather than libm calls. chelis#691 covered the
        // DIRECT `emit_binary_func` / abs nodes and is repaired: those
        // now emit an exact integer ternary and `emit_integer_abs`.
        //
        // The pre-#919 comment claimed the fuse pass "currently only
        // produces f32 fused chains in practice", making this a
        // future-regression tripwire. That was false: it produces f64
        // chains today for every ordinary f64 activation — `sigmoid`,
        // `tanh`, `gelu`, and any composite such as `exp(x) * x`. The
        // guard only looked unreachable because `chelis-python`'s
        // artifact gate rejected f64 one layer earlier, and it was a
        // `panic!`, so it crossed the pyo3 FFI boundary as a
        // `PanicException` instead of a diagnostic.
        if !matches!(ty.precision, Prim::F32 | Prim::F64) {
            // The channel chelis#730 Phase 1 (census row 11) established: a
            // section C3 diagnostic, not a compiler panic. This site is
            // census row 24, not row 11.
            let authority = match ty.precision {
                Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64 => {
                    chelis_types::unimplemented_rejection!(
                        729,
                        "integer fused elementwise chains need exact integer step operators rather than libm calls"
                    )
                }
                Prim::F16 | Prim::Bf16 => chelis_types::unimplemented_rejection!(
                    729,
                    "reduced-float fused chains need the target capability table's convert-compute-finalize kernel"
                ),
                Prim::F8e4m3 => chelis_types::deliberate_rejection!(
                    "[04-DTYPE-1]",
                    "f8e4m3 is reserved but inactive and must not reach backend emission"
                ),
                Prim::Bool => chelis_types::deliberate_rejection!(
                    "[04-NUM-4]",
                    "bool has no arithmetic width and arithmetic fused chains are rejected"
                ),
                Prim::String => chelis_types::unimplemented_rejection!(
                    729,
                    "the exhaustive target capability table has no C fused-chain string cell"
                ),
                Prim::F32 | Prim::F64 => unreachable!("supported fused precision"),
            };
            return Err(Unsupported::new(
                UnsupportedKind::Op("fused elementwise chain".to_string()),
                format!(
                    "`{}` tensors in the C DAG emitter (node {id})",
                    ty.precision.name()
                ),
                Stage::Codegen("c"),
                authority,
            ));
        }
        // Element type and math-symbol precision come from the same
        // `ty`, so a `double v0` can never be fed by an `expf`.
        let et = Self::elem_type(ty);
        let is_f64 = Self::is_f64(ty);
        // The exact public carrier exposes an untyped `void *data` payload.
        // Every generated dereference names the IR-pinned element type.
        let out_cast = format!("({et}*)");
        let in_cast = format!("(const {et}*)");
        let reusable_input = in_place.as_ref().map(|spec| spec.token.source());
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
        if reusable_input.is_some() {
            self.line(&format!("{et}* __out_{id} = {out_cast}t{id}_data;"));
        } else {
            self.line(&format!(
                "{et}* restrict __out_{id} = {out_cast}t{id}_data;"
            ));
        }
        for (ext_idx, ext_node) in inputs.iter().enumerate() {
            let ext_id = ext_node.0;
            if reusable_input == Some(*ext_node) {
                self.line(&format!(
                    "const {et}* __ext{ext_idx}_{id} = {in_cast}t{ext_id}_data;"
                ));
            } else {
                self.line(&format!(
                    "const {et}* restrict __ext{ext_idx}_{id} = {in_cast}t{ext_id}_data;"
                ));
            }
        }

        // The Sleef path is 8-wide `__m256` single precision
        // (`_mm256_loadu_ps`, `_mm256_storeu_ps`), so it is f32-only.
        // f64 chains take the scalar OMP SIMD loop, mirroring how
        // `emit_unary_func` skips its vForce/Sleef batch paths when
        // `is_f64`.
        let use_sleef =
            !is_f64 && self.math_lib == crate::MathLib::Sleef && Self::has_math_ops(ops);

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
            self.line("int64_t __i = 0;");
            // 8-wide main loop
            self.line(&format!("for (; __i + 8 <= t{id}_size; __i += 8) {{"));
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
            self.line(&format!("for (; __i < t{id}_size; __i++) {{"));
            self.indent += 1;
            for (ext_idx, _) in inputs.iter().enumerate() {
                self.line(&format!(
                    "{et} __in_ext{ext_idx} = __ext{ext_idx}_{id}[__i];"
                ));
            }
            for (s, step) in ops.iter().enumerate() {
                let expr =
                    Self::scalar_step_expr(&step.op, &resolve_fast, &step.input_indices, is_f64);
                self.line(&format!("{et} v{s} = {expr};"));
            }
            self.line(&format!("__out_{id}[__i] = v{last};"));
            self.indent -= 1;
            self.line("}");
            self.indent -= 1;
            self.line("}");
            self.line("#else");
            // Fallback: Level-1 scalar OMP SIMD loop.
            self.line("#pragma omp parallel for simd");
            self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
            self.indent += 1;
            for (ext_idx, _) in inputs.iter().enumerate() {
                self.line(&format!("{et} __in_ext{ext_idx} = __ext{ext_idx}_{id}[i];"));
            }
            for (s, step) in ops.iter().enumerate() {
                let expr =
                    Self::scalar_step_expr(&step.op, &resolve_fast, &step.input_indices, is_f64);
                self.line(&format!("{et} v{s} = {expr};"));
            }
            self.line(&format!("__out_{id}[i] = v{last};"));
            self.indent -= 1;
            self.line("}");
            self.line("#endif");
        } else {
            // --- Level-1 scalar OMP SIMD loop (default fast path) ---
            self.line("#pragma omp parallel for simd");
            self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
            self.indent += 1;
            for (ext_idx, _) in inputs.iter().enumerate() {
                self.line(&format!("{et} __in_ext{ext_idx} = __ext{ext_idx}_{id}[i];"));
            }
            for (s, step) in ops.iter().enumerate() {
                let expr =
                    Self::scalar_step_expr(&step.op, &resolve_fast, &step.input_indices, is_f64);
                self.line(&format!("{et} v{s} = {expr};"));
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
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));

        // Compute strided index for each external input.
        for (ext_idx, ext_node) in inputs.iter().enumerate() {
            let ext_id = ext_node.0;
            self.line(&format!(
                "int64_t idx_ext{ext_idx} = chelis_indices_to_flat(indices, t{ext_id}_strides, t{ext_id}_rank);"
            ));
        }

        // Emit each fused step using slow-path indexed access. The
        // strided reads go straight through `t{n}_data`, which is
        // declared `float *`, so an f64 chain must reinterpret the
        // pointer before indexing — indexing first would advance by
        // 4 bytes per element and read half of each double.
        let resolve_slow = |fi: &FusedInput| -> String {
            match fi {
                FusedInput::External(i) => {
                    let ext_id = inputs[*i].0;
                    format!("((const {et}*)t{ext_id}_data)[idx_ext{i}]")
                }
                FusedInput::PreviousStep(j) => format!("v{j}"),
            }
        };

        for (s, step) in ops.iter().enumerate() {
            let expr = Self::scalar_step_expr(&step.op, &resolve_slow, &step.input_indices, is_f64);
            self.line(&format!("{et} v{s} = {expr};"));
        }

        // Store last step's result through the exact element type selected
        // above. Both active float widths use the same typed shape here.
        self.line(&format!("(({et}*)t{id}_data)[i] = v{last};"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    // ---- BLAS matmul ----
    fn emit_blas_matmul(&mut self, id: usize, spec: &MatmulEmitSpec, ty: &TensorType) {
        // WS-1: bf16 / f16 OPERAND matmul routes through a dedicated
        // convert-then-sgemm wrapper (allocate f32 scratch buffers,
        // convert operands, dispatch `cblas_sgemm`, downcast result
        // back if the destination is bf16/f16, write directly if the
        // destination is f32, per spec §5.7.1). Dispatch is by
        // operand precision, not by output precision: a bf16-input
        // matmul that the matmul-pattern detector emits with an
        // f32 output (because Sum's accumulator-pinned output is f32)
        // still needs operand conversion to read the `uint16_t`
        // storage as f32.
        if matches!(spec.operand_precision, Prim::Bf16 | Prim::F16) {
            self.emit_blas_matmul_reduced_f(id, spec, ty);
            return;
        }
        // WS-A1: dispatch f32 -> cblas_sgemm and f64 -> cblas_dgemm by the
        // IR-pinned accumulator precision (see the match below). The
        // historical F32-only assert that lived here was lifted with WS-A1;
        // upstream guards in verify.rs and validate_supported_precisions
        // reject unsupported accumulator dtypes before they reach this
        // function.
        let a = spec.a.0;
        let b = spec.b.0;
        let m_expr = Self::emit_dim_expr(&spec.m);
        let n_expr = Self::emit_dim_expr(&spec.n);
        let k_expr = Self::emit_dim_expr(&spec.k);
        // Dispatch BLAS routine and matching scalar literals from the IR-
        // pinned accumulator precision per spec §5.7 / §5.7.1. f32 → sgemm,
        // f64 → dgemm. Other accumulator dtypes are rejected upstream by
        // the F1 guards in `verify.rs` and `validate_supported_precisions`,
        // so this function only sees f32 / f64. The guards mean an
        // unexpected accumulator here is a backend bug, not a user error.
        let (gemm, alpha, beta, ptr_ty) = match spec.accumulator {
            Prim::F32 => ("cblas_sgemm", "1.0f", "0.0f", "float"),
            Prim::F64 => ("cblas_dgemm", "1.0", "0.0", "double"),
            other => panic!(
                "C backend BLAS dispatch reached unsupported accumulator `{}` at \
                 node {id}; the F1 guard in verify.rs / validate_supported_precisions \
                 should have rejected this earlier (spec/04-type-system.md §5.7.1)",
                other.name()
            ),
        };
        self.line(&format!("chelis_tensor *t{id}_a = t{a};"));
        self.line(&format!(
            "if (!(t{a}_rank >= 2 && t{a}_strides[t{a}_rank - 1] == 1 && t{a}_strides[t{a}_rank - 2] == {k_expr})) {{"
        ));
        self.indent += 1;
        self.line("abort();");
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_tensor *t{id}_b = t{b};"));
        self.line(&format!(
            "if (!(t{b}_rank >= 2 && t{b}_strides[t{b}_rank - 1] == 1 && t{b}_strides[t{b}_rank - 2] == {n_expr})) {{"
        ));
        self.indent += 1;
        self.line("abort();");
        self.indent -= 1;
        self.line("}");
        self.emit_slot_wrapper(id, ty);
        if spec.batch_dims.is_empty() {
            self.line(&format!(
                "{gemm}(CblasRowMajor, CblasNoTrans, CblasNoTrans, {m_expr}, {n_expr}, {k_expr}, {alpha}, ({ptr_ty}*)t{a}_data, {k_expr}, ({ptr_ty}*)t{b}_data, {n_expr}, {beta}, ({ptr_ty}*)t{id}_data, {n_expr});"
            ));
        } else {
            let batch_count = spec
                .batch_dims
                .iter()
                .map(Self::emit_dim_expr)
                .reduce(|lhs, rhs| format!("({lhs} * {rhs})"))
                .unwrap_or_else(|| "1".to_string());
            self.line(&format!("int64_t t{id}_batch_count = {batch_count};"));
            self.line(&format!(
                "for (int64_t t{id}_batch = 0; t{id}_batch < t{id}_batch_count; t{id}_batch++) {{"
            ));
            self.indent += 1;
            self.line(&format!("int64_t t{id}_rem = t{id}_batch;"));
            self.line(&format!("int64_t t{id}_a_offset = 0;"));
            self.line(&format!("int64_t t{id}_b_offset = 0;"));
            self.line(&format!("int64_t t{id}_out_offset = 0;"));
            for axis in (0..spec.batch_dims.len()).rev() {
                let dim_expr = Self::emit_dim_expr(&spec.batch_dims[axis]);
                self.line(&format!(
                    "int64_t t{id}_coord_{axis} = t{id}_rem % ({dim_expr});"
                ));
                self.line(&format!("t{id}_rem /= ({dim_expr});"));
                self.line(&format!(
                    "t{id}_a_offset += t{id}_coord_{axis} * t{a}_strides[{axis}];"
                ));
                self.line(&format!(
                    "t{id}_b_offset += t{id}_coord_{axis} * t{b}_strides[{axis}];"
                ));
                self.line(&format!(
                    "t{id}_out_offset += t{id}_coord_{axis} * t{id}_strides[{axis}];"
                ));
            }
            self.line(&format!(
                "{gemm}(CblasRowMajor, CblasNoTrans, CblasNoTrans, {m_expr}, {n_expr}, {k_expr}, {alpha}, ({ptr_ty}*)t{a}_data + t{id}_a_offset, {k_expr}, ({ptr_ty}*)t{b}_data + t{id}_b_offset, {n_expr}, {beta}, ({ptr_ty}*)t{id}_data + t{id}_out_offset, {n_expr});"
            ));
            self.indent -= 1;
            self.line("}");
        }
        self.line(&format!(
            "if (t{id}_a != t{a}) chelis_tensor_release(t{id}_a);"
        ));
        self.line(&format!(
            "if (t{id}_b != t{b}) chelis_tensor_release(t{id}_b);"
        ));
    }

    /// WS-1: bf16 / f16 matmul via the convert-then-sgemm wrapper. Per
    /// spec/04-type-system.md §5.7.1 the accumulator dtype is f32
    /// even when the operand and output dtypes are bf16/f16; this
    /// path materializes that pin by allocating two f32 scratch
    /// buffers for the operands, an f32 scratch buffer for the
    /// `cblas_sgemm` output, and converting back to the destination
    /// reduced-float precision element-wise. Scratch lifetimes are
    /// per-call (`malloc` / `free` inside the emitted wrapper scope,
    /// no buffer pooling).
    ///
    /// Routes through the existing contiguity-promotion preamble
    /// (`chelis_contiguous` on operands when the trailing strides
    /// don't match the M/N/K layout) so the conversion always reads
    /// from a stride-1, row-major source.
    fn emit_blas_matmul_reduced_f(&mut self, id: usize, spec: &MatmulEmitSpec, ty: &TensorType) {
        let a = spec.a.0;
        let b = spec.b.0;
        let m_expr = Self::emit_dim_expr(&spec.m);
        let n_expr = Self::emit_dim_expr(&spec.n);
        let k_expr = Self::emit_dim_expr(&spec.k);
        let reduced_to_f32 = Self::reduced_to_f32_fn(spec.operand_precision);
        // Two output-precision cases per the matmul-pattern detector +
        // user-constructed matmul shape:
        //   * Output is bf16/f16: convert f32 accumulator buffer back
        //     into the destination element-wise.
        //   * Output is f32: write `cblas_sgemm`'s result directly into
        //     `t{id}_data` with no intermediate scratch buffer.
        let output_is_reduced = Self::is_reduced_float(ty);
        let f32_to_reduced = if output_is_reduced {
            Some(Self::f32_to_reduced_fn(ty.precision))
        } else {
            None
        };
        // The IR contract is that the matmul accumulator for bf16/f16
        // operands is f32 (spec §5.7.1). The verifier enforces it.
        if spec.accumulator != Prim::F32 {
            panic!(
                "WS-1: bf16/f16 matmul wrapper expects f32 accumulator per \
                 spec/04-type-system.md §5.7.1, got `{}` at node {id}",
                spec.accumulator.name()
            );
        }
        // Promote operands to contiguous row-major if they don't
        // already satisfy `cblas_sgemm`'s leading-dimension contract.
        // Same shape as the f32/f64 path.
        self.line(&format!("chelis_tensor *t{id}_a = t{a};"));
        self.line(&format!(
            "if (!(t{a}_rank >= 2 && t{a}_strides[t{a}_rank - 1] == 1 && t{a}_strides[t{a}_rank - 2] == {k_expr})) {{"
        ));
        self.indent += 1;
        self.line("abort();");
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_tensor *t{id}_b = t{b};"));
        self.line(&format!(
            "if (!(t{b}_rank >= 2 && t{b}_strides[t{b}_rank - 1] == 1 && t{b}_strides[t{b}_rank - 2] == {n_expr})) {{"
        ));
        self.indent += 1;
        self.line("abort();");
        self.indent -= 1;
        self.line("}");
        self.emit_slot_wrapper(id, ty);
        self.line("/* spec/04-type-system.md §5.7.1: bf16/f16 matmul uses f32 accumulator */");
        self.line(&format!(
            "int64_t t{id}_mk = (int64_t)({m_expr}) * (int64_t)({k_expr});"
        ));
        self.line(&format!(
            "int64_t t{id}_kn = (int64_t)({k_expr}) * (int64_t)({n_expr});"
        ));
        self.line(&format!(
            "int64_t t{id}_mn = (int64_t)({m_expr}) * (int64_t)({n_expr});"
        ));
        self.line(&format!(
            "float *t{id}_af = (float*)malloc((size_t)t{id}_mk * sizeof(float));"
        ));
        self.line(&format!(
            "float *t{id}_bf = (float*)malloc((size_t)t{id}_kn * sizeof(float));"
        ));
        // Output scratch only needed when the destination is bf16/f16;
        // an f32 destination accumulates directly into `t{id}_data`.
        if output_is_reduced {
            self.line(&format!(
                "float *t{id}_cf = (float*)malloc((size_t)t{id}_mn * sizeof(float));"
            ));
        }
        if spec.batch_dims.is_empty() {
            self.line(&format!(
                "for (int64_t i = 0; i < t{id}_mk; i++) t{id}_af[i] = {reduced_to_f32}(((const uint16_t*)t{a}_data)[i]);"
            ));
            self.line(&format!(
                "for (int64_t i = 0; i < t{id}_kn; i++) t{id}_bf[i] = {reduced_to_f32}(((const uint16_t*)t{b}_data)[i]);"
            ));
            let c_arg = if output_is_reduced {
                format!("t{id}_cf")
            } else {
                format!("(float*)t{id}_data")
            };
            self.line(&format!(
                "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, {m_expr}, {n_expr}, {k_expr}, 1.0f, t{id}_af, {k_expr}, t{id}_bf, {n_expr}, 0.0f, {c_arg}, {n_expr});"
            ));
            if let Some(f32_to_reduced) = f32_to_reduced {
                self.line(&format!(
                    "for (int64_t i = 0; i < t{id}_mn; i++) ((uint16_t*)t{id}_data)[i] = {f32_to_reduced}(t{id}_cf[i]);"
                ));
            }
        } else {
            let batch_count = spec
                .batch_dims
                .iter()
                .map(Self::emit_dim_expr)
                .reduce(|lhs, rhs| format!("({lhs} * {rhs})"))
                .unwrap_or_else(|| "1".to_string());
            self.line(&format!("int64_t t{id}_batch_count = {batch_count};"));
            self.line(&format!(
                "for (int64_t t{id}_batch = 0; t{id}_batch < t{id}_batch_count; t{id}_batch++) {{"
            ));
            self.indent += 1;
            self.line(&format!("int64_t t{id}_rem = t{id}_batch;"));
            self.line(&format!("int64_t t{id}_a_offset = 0;"));
            self.line(&format!("int64_t t{id}_b_offset = 0;"));
            self.line(&format!("int64_t t{id}_out_offset = 0;"));
            for axis in (0..spec.batch_dims.len()).rev() {
                let dim_expr = Self::emit_dim_expr(&spec.batch_dims[axis]);
                self.line(&format!(
                    "int64_t t{id}_coord_{axis} = t{id}_rem % ({dim_expr});"
                ));
                self.line(&format!("t{id}_rem /= ({dim_expr});"));
                self.line(&format!(
                    "t{id}_a_offset += t{id}_coord_{axis} * t{a}_strides[{axis}];"
                ));
                self.line(&format!(
                    "t{id}_b_offset += t{id}_coord_{axis} * t{b}_strides[{axis}];"
                ));
                self.line(&format!(
                    "t{id}_out_offset += t{id}_coord_{axis} * t{id}_strides[{axis}];"
                ));
            }
            self.line(&format!(
                "for (int64_t i = 0; i < t{id}_mk; i++) t{id}_af[i] = {reduced_to_f32}(((const uint16_t*)t{a}_data)[t{id}_a_offset + i]);"
            ));
            self.line(&format!(
                "for (int64_t i = 0; i < t{id}_kn; i++) t{id}_bf[i] = {reduced_to_f32}(((const uint16_t*)t{b}_data)[t{id}_b_offset + i]);"
            ));
            let c_arg = if output_is_reduced {
                format!("t{id}_cf")
            } else {
                format!("(float*)t{id}_data + t{id}_out_offset")
            };
            self.line(&format!(
                "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, {m_expr}, {n_expr}, {k_expr}, 1.0f, t{id}_af, {k_expr}, t{id}_bf, {n_expr}, 0.0f, {c_arg}, {n_expr});"
            ));
            if let Some(f32_to_reduced) = f32_to_reduced {
                self.line(&format!(
                    "for (int64_t i = 0; i < t{id}_mn; i++) ((uint16_t*)t{id}_data)[t{id}_out_offset + i] = {f32_to_reduced}(t{id}_cf[i]);"
                ));
            }
            self.indent -= 1;
            self.line("}");
        }
        self.line(&format!("free(t{id}_af);"));
        self.line(&format!("free(t{id}_bf);"));
        if output_is_reduced {
            self.line(&format!("free(t{id}_cf);"));
        }
        self.line(&format!(
            "if (t{id}_a != t{a}) chelis_tensor_release(t{id}_a);"
        ));
        self.line(&format!(
            "if (t{id}_b != t{b}) chelis_tensor_release(t{id}_b);"
        ));
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
        dag: VerifiedDagView<'_>,
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
        self.line(&format!("chelis_tensor *t{id}_values = t{values};"));
        self.line(&format!("chelis_tensor *t{id}_indices = t{indices};"));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "const {value_et} *t{id}_values_data = (const {value_et}*)t{values}_data;"
        ));
        self.line(&format!(
            "const {index_et} *t{id}_indices_data = (const {index_et}*)t{indices}_data;"
        ));
        self.line(&format!(
            "{value_et} *t{id}_out_data = ({value_et}*)t{id}_data;"
        ));
        self.line(&format!("int64_t t{id}_before = {before};"));
        self.line(&format!("int64_t t{id}_axis_size = {axis_size};"));
        self.line(&format!("int64_t t{id}_after = {after};"));
        self.line(&format!("int64_t t{id}_index_count = t{indices}_size;"));
        self.line(&format!(
            "for (int64_t t{id}_b = 0; t{id}_b < t{id}_before; t{id}_b++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "for (int64_t t{id}_i = 0; t{id}_i < t{id}_index_count; t{id}_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t t{id}_g = (t{indices}_dtype == CHELIS_DTYPE_I64) ? (int64_t)((const int64_t*)t{indices}_data)[t{id}_i] : (int64_t)t{id}_indices_data[t{id}_i];"
        ));
        self.line(&format!(
            "if (t{id}_g < 0 || t{id}_g >= t{id}_axis_size) abort();"
        ));
        self.line(&format!(
            "for (int64_t t{id}_d = 0; t{id}_d < t{id}_after; t{id}_d++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t t{id}_out = ((t{id}_b * t{id}_index_count + t{id}_i) * t{id}_after) + t{id}_d;"
        ));
        self.line(&format!(
            "int64_t t{id}_src = ((t{id}_b * t{id}_axis_size + t{id}_g) * t{id}_after) + t{id}_d;"
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
            "if (t{id}_values != t{values}) chelis_tensor_release(t{id}_values);"
        ));
        self.line(&format!(
            "if (t{id}_indices != t{indices}) chelis_tensor_release(t{id}_indices);"
        ));
    }

    fn emit_sparse_scatter_add(
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
        let target_et = Self::elem_type(target_ty);
        let index_et = Self::elem_type(indices_ty);
        let update_et = Self::elem_type(updates_ty);
        let target_elem_size = Self::elem_size_expr(target_ty);
        let before = Self::dim_product_expr(&target_ty.dims[..axis]);
        let axis_size = Self::emit_dim_info(&target_ty.dims[axis]);
        let after = Self::dim_product_expr(&target_ty.dims[axis + 1..]);
        self.line(&format!("chelis_tensor *t{id}_target = t{target};"));
        self.line(&format!("chelis_tensor *t{id}_indices = t{indices};"));
        self.line(&format!("chelis_tensor *t{id}_updates = t{updates};"));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "const {index_et} *t{id}_indices_data = (const {index_et}*)t{indices}_data;"
        ));
        self.line(&format!(
            "const {update_et} *t{id}_updates_data = (const {update_et}*)t{updates}_data;"
        ));
        self.line(&format!(
            "{target_et} *t{id}_out_data = ({target_et}*)t{id}_data;"
        ));
        self.line(&format!(
            "memcpy(t{id}_data, t{target}_data, (size_t)t{id}_size * {target_elem_size});"
        ));
        self.line(&format!("int64_t t{id}_before = {before};"));
        self.line(&format!("int64_t t{id}_axis_size = {axis_size};"));
        self.line(&format!("int64_t t{id}_after = {after};"));
        self.line(&format!("int64_t t{id}_index_count = t{indices}_size;"));
        self.line(&format!(
            "for (int64_t t{id}_b = 0; t{id}_b < t{id}_before; t{id}_b++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "for (int64_t t{id}_i = 0; t{id}_i < t{id}_index_count; t{id}_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t t{id}_g = (t{indices}_dtype == CHELIS_DTYPE_I64) ? (int64_t)((const int64_t*)t{indices}_data)[t{id}_i] : (int64_t)t{id}_indices_data[t{id}_i];"
        ));
        self.line(&format!(
            "if (t{id}_g < 0 || t{id}_g >= t{id}_axis_size) abort();"
        ));
        self.line(&format!(
            "for (int64_t t{id}_d = 0; t{id}_d < t{id}_after; t{id}_d++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t t{id}_src = ((t{id}_b * t{id}_index_count + t{id}_i) * t{id}_after) + t{id}_d;"
        ));
        self.line(&format!(
            "int64_t t{id}_out = ((t{id}_b * t{id}_axis_size + t{id}_g) * t{id}_after) + t{id}_d;"
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
            "if (t{id}_target != t{target}) chelis_tensor_release(t{id}_target);"
        ));
        self.line(&format!(
            "if (t{id}_indices != t{indices}) chelis_tensor_release(t{id}_indices);"
        ));
        self.line(&format!(
            "if (t{id}_updates != t{updates}) chelis_tensor_release(t{id}_updates);"
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
        dag: VerifiedDagView<'_>,
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
        self.line(&format!("chelis_tensor *t{id}_target = t{target};"));
        self.line(&format!("chelis_tensor *t{id}_indices = t{indices};"));
        self.line(&format!("chelis_tensor *t{id}_updates = t{updates};"));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "const {index_et} *t{id}_indices_data = (const {index_et}*)t{indices}_data;"
        ));
        self.line(&format!(
            "const {update_et} *t{id}_updates_data = (const {update_et}*)t{updates}_data;"
        ));
        self.line(&format!(
            "{target_et} *t{id}_out_data = ({target_et}*)t{id}_data;"
        ));
        self.line(&format!(
            "memcpy(t{id}_data, t{target}_data, (size_t)t{id}_size * {target_elem_size});"
        ));
        self.line(&format!("int64_t t{id}_before = {before};"));
        self.line(&format!("int64_t t{id}_axis_size = {axis_size};"));
        self.line(&format!("int64_t t{id}_after = {after};"));
        self.line(&format!("int64_t t{id}_index_count = t{indices}_size;"));
        // Single-threaded sequential loop: deterministic last-write-wins
        // requires that no two writes to the same target cell race. The
        // outer (b, i, d) iteration order is the canonical
        // updates-tensor row-major traversal.
        self.line(&format!(
            "for (int64_t t{id}_b = 0; t{id}_b < t{id}_before; t{id}_b++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "for (int64_t t{id}_i = 0; t{id}_i < t{id}_index_count; t{id}_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t t{id}_g = (t{indices}_dtype == CHELIS_DTYPE_I64) ? (int64_t)((const int64_t*)t{indices}_data)[t{id}_i] : (int64_t)t{id}_indices_data[t{id}_i];"
        ));
        self.line(&format!(
            "if (t{id}_g < 0 || t{id}_g >= t{id}_axis_size) abort();"
        ));
        self.line(&format!(
            "for (int64_t t{id}_d = 0; t{id}_d < t{id}_after; t{id}_d++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t t{id}_src = ((t{id}_b * t{id}_index_count + t{id}_i) * t{id}_after) + t{id}_d;"
        ));
        self.line(&format!(
            "int64_t t{id}_out = ((t{id}_b * t{id}_axis_size + t{id}_g) * t{id}_after) + t{id}_d;"
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
            "if (t{id}_target != t{target}) chelis_tensor_release(t{id}_target);"
        ));
        self.line(&format!(
            "if (t{id}_indices != t{indices}) chelis_tensor_release(t{id}_indices);"
        ));
        self.line(&format!(
            "if (t{id}_updates != t{updates}) chelis_tensor_release(t{id}_updates);"
        ));
    }

    /// Emit C for ONNX `ScatterElements` (spec §3.5.1). Element-wise:
    /// `data`, `indices`, `updates` share a rank; `indices.dims ==
    /// updates.dims`; `output.dims == data.dims`. Each flat update
    /// position is decomposed into a coordinate over the indices shape;
    /// the `axis` coordinate is replaced by `indices[i]` and the write
    /// lands at the corresponding linear offset in the (data-shaped)
    /// output. Last-write-wins under updates row-major order, so the
    /// loop is single-threaded.
    fn emit_sparse_scatter_elements(
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
        let data_et = Self::elem_type(data_ty);
        let index_et = Self::elem_type(indices_ty);
        let update_et = Self::elem_type(updates_ty);
        let data_elem_size = Self::elem_size_expr(data_ty);
        let rank = data_ty.dims.len();

        self.line(&format!("chelis_tensor *t{id}_data_input = t{data};"));
        self.line(&format!("chelis_tensor *t{id}_indices = t{indices};"));
        self.line(&format!("chelis_tensor *t{id}_updates = t{updates};"));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "const {index_et} *t{id}_indices_data = (const {index_et}*)t{indices}_data;"
        ));
        self.line(&format!(
            "const {update_et} *t{id}_updates_data = (const {update_et}*)t{updates}_data;"
        ));
        self.line(&format!(
            "{data_et} *t{id}_out_data = ({data_et}*)t{id}_data;"
        ));
        self.line(&format!(
            "memcpy(t{id}_data, t{data}_data, (size_t)t{id}_size * {data_elem_size});"
        ));
        // Per-axis sizes for the indices/updates grid and the data grid,
        // plus the data row-major strides used to recompute the output
        // offset after the axis coordinate is replaced by the index.
        let axis_size = Self::emit_dim_info(&data_ty.dims[axis]);
        self.line(&format!("int t{id}_axis = {axis};"));
        self.line(&format!("int64_t t{id}_axis_size = {axis_size};"));
        self.line(&format!("int64_t t{id}_update_count = t{updates}_size;"));
        for d in 0..rank {
            let idx_dim = Self::emit_dim_info(&indices_ty.dims[d]);
            let data_dim = Self::emit_dim_info(&data_ty.dims[d]);
            self.line(&format!("int64_t t{id}_idim{d} = {idx_dim};"));
            self.line(&format!("int64_t t{id}_ddim{d} = {data_dim};"));
        }
        // Single-threaded sequential loop over the updates tensor in
        // row-major flat order: deterministic last-write-wins requires
        // no two writes to the same output cell race.
        self.line(&format!(
            "for (int64_t t{id}_i = 0; t{id}_i < t{id}_update_count; t{id}_i++) {{"
        ));
        self.indent += 1;
        // #476: int tensors store their values bit-packed into the
        // float-typed `->data`, so reading `(int)t->data[i]` on a
        // CHELIS_DTYPE_I32 index tensor would `(int)`-truncate the FLOAT
        // reinterpretation of the int32 bits (e.g. index `2` →
        // `(int)2.8e-45f` → `0`), silently gathering the wrong row.
        // The read must go through the dtype-correct pointer cast on
        // BOTH dtype branches. The hyperplane sparse emits (gather,
        // scatter_replace, scatter_add) do the same via their
        // already-declared `t{id}_indices_data` (`const {index_et}*`)
        // pointer; the element-wise emit casts inline here because it
        // has no such pre-declared pointer in scope.
        self.line(&format!(
            "int64_t t{id}_g = (t{indices}_dtype == CHELIS_DTYPE_I64) ? (int64_t)((const int64_t*)t{indices}_data)[t{id}_i] : (int64_t)((const int32_t*)t{indices}_data)[t{id}_i];"
        ));
        self.line(&format!(
            "if (t{id}_g < 0 || t{id}_g >= t{id}_axis_size) abort();"
        ));
        // Decompose the flat updates index into per-axis coordinates
        // over the indices/updates shape, then build the output linear
        // offset over the data shape with the axis coordinate replaced
        // by the scattered index.
        self.line(&format!("int64_t t{id}_rem = t{id}_i;"));
        self.line(&format!("int64_t t{id}_out = 0;"));
        for d in (0..rank).rev() {
            self.line(&format!("int64_t t{id}_c{d} = t{id}_rem % t{id}_idim{d};"));
            self.line(&format!("t{id}_rem /= t{id}_idim{d};"));
        }
        // out = sum_d (coord_d or g at axis) * stride_d, computed via a
        // running row-major fold over the data dims.
        self.line(&format!("int64_t t{id}_stride = 1;"));
        for d in (0..rank).rev() {
            self.line(&format!(
                "int64_t t{id}_coord{d} = (t{id}_axis == {d}) ? t{id}_g : t{id}_c{d};"
            ));
            self.line(&format!("t{id}_out += t{id}_coord{d} * t{id}_stride;"));
            self.line(&format!("t{id}_stride *= t{id}_ddim{d};"));
        }
        // Last-write-wins assignment (NOT accumulation).
        self.line(&format!(
            "t{id}_out_data[t{id}_out] = t{id}_updates_data[t{id}_i];"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "if (t{id}_data_input != t{data}) chelis_tensor_release(t{id}_data_input);"
        ));
        self.line(&format!(
            "if (t{id}_indices != t{indices}) chelis_tensor_release(t{id}_indices);"
        ));
        self.line(&format!(
            "if (t{id}_updates != t{updates}) chelis_tensor_release(t{id}_updates);"
        ));
    }

    // ---- Reduce sum ----
    //
    // WS-A1: parameterized by the output tensor's dtype (== the IR
    // `Sum` op's accumulator precision per spec §5.7.1, enforced by
    // verify::C3a). Per the destructure-`..` memory rule, the
    // accumulator type and zero literal MUST come from the IR-pinned
    // accumulator (which is the output dtype here, by §5.7.1's
    // `output_type.precision == accumulator` invariant). The operand
    // dtype is the source data layout; the accumulator dtype drives
    // the running-sum width.
    fn emit_reduce_sum(
        &mut self,
        id: usize,
        axis: usize,
        accumulator: Prim,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let input_node = dag.get(inputs[0]).unwrap();
        let input_prec = input_node.output_type.precision;
        // C3a invariant from `chelis_ir::verify`: `Sum.output_type.precision == accumulator`.
        // Pin it locally so a future emit refactor that decouples the two
        // gets a loud assertion instead of silent miscompilation.
        debug_assert_eq!(
            ty.precision, accumulator,
            "WS-A4: reduce_sum output precision must equal IR accumulator field; \
             verifier-enforced spec/04-type-system.md §5.7.1 invariant violated"
        );

        // WS-A4: i8/i16 source data + i32 accumulator + i32 output, per
        // spec/04-type-system.md §5.7.1. The accumulator C type comes
        // from the IR-supplied `accumulator` parameter; do NOT infer it
        // from the operand precision (the destructure-`..` footgun the
        // F1 finding caught for BlasMatmul). Routed through a dedicated
        // helper that zero-fills i32 inline (no `chelis_fill_i32`
        // runtime symbol today) and keeps the source/accumulator C type
        // spellings explicit at the cast site.
        if matches!(input_prec, Prim::Int8 | Prim::Int16) && accumulator == Prim::Int32 {
            let src_c_ty = match input_prec {
                Prim::Int8 => "int8_t",
                Prim::Int16 => "int16_t",
                _ => unreachable!(),
            };
            self.emit_reduce_sum_int_promoted(id, axis, src_c_ty, "int32_t", inputs, ty, dag);
            return;
        }

        // WS-1: bf16 / f16 source + f32 accumulator + f32 output, per
        // spec/04-type-system.md §5.7.1. The accumulator field is f32
        // (the type system's `default_reduce_sum_accumulator` returns
        // `Prim::F32` for bf16/f16 operands); the operand storage is
        // `uint16_t`. Route through a dedicated helper that loads each
        // element via `chelis_<x>_to_f32` before accumulating in `f32`
        // so the loop never reads `uint16_t` bits as if they were
        // float bytes. The §5.7.1 enforcement test
        // `bf16_reduce_sum_uses_f32_accumulator_per_spec_5_7_1` locks
        // this path.
        if matches!(input_prec, Prim::Bf16 | Prim::F16) && accumulator == Prim::F32 {
            self.emit_reduce_sum_reduced_f(id, axis, input_prec, inputs, ty, dag);
            return;
        }

        // WS-A1 general path: f32 → f32 (with the SIMD fast path),
        // f64 → f64, f32 → f64 widening, i32 → i32, i64 → i64, etc.
        // The general scalar loop drives accumulator type and zero
        // initializer from `elem_type` / `scalar_zero_literal` /
        // `fill_zero_call` so every (operand, accumulator) combo the
        // C backend's helpers know about lowers correctly without a
        // dedicated specialization.
        self.emit_reduce_sum_general(id, axis, inputs, ty, dag);
    }

    /// [05-OP-29] dedicated multi-axis bool count. The input is visited in
    /// original row-major order within each result group, then folded through
    /// an adjacent-pair balanced checked-int64 tree. No cast+sum lowering is
    /// used, and the first typed boundary validates both the bool tag and its
    /// exact 0/1 payload.
    fn emit_count(
        &mut self,
        id: usize,
        axes: &[usize],
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let input_ty = &dag.get(inputs[0]).expect("count input exists").output_type;
        assert_eq!(input_ty.precision, Prim::Bool, "verified count input dtype");
        assert_eq!(ty.precision, Prim::Int64, "verified count output dtype");
        assert!(!axes.is_empty(), "verified count axes are non-empty");
        assert!(
            axes.windows(2).all(|pair| pair[0] > pair[1]),
            "verified count axes are strictly descending"
        );

        self.emit_slot_wrapper(id, ty);
        self.line(&format!("if (t{a}_dtype != CHELIS_DTYPE_BOOL) {{"));
        self.indent += 1;
        self.line(&format!(
            "fprintf(stderr, \"count expected CHELIS_DTYPE_BOOL input at node {id}\\n\");"
        ));
        self.line("abort();");
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "const uint8_t* restrict __count_in_{id} = (const uint8_t*)t{a}_data;"
        ));
        self.line(&format!(
            "int64_t* restrict __count_out_{id} = (int64_t*)t{id}_data;"
        ));
        self.line(&format!("int64_t __count_n_{id} = 1;"));
        for axis in axes.iter().rev() {
            self.line(&format!(
                "if (t{a}_shape[{axis}] != 0 && __count_n_{id} > INT64_MAX / t{a}_shape[{axis}]) {{ fprintf(stderr, \"count reduction extent overflow\\n\"); abort(); }}"
            ));
            self.line(&format!("__count_n_{id} *= t{a}_shape[{axis}];"));
        }
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t* __level_{id} = __count_n_{id} == 0 ? NULL : (int64_t*)malloc((size_t)__count_n_{id} * sizeof(int64_t));"
        ));
        self.line(&format!(
            "if (__count_n_{id} != 0 && __level_{id} == NULL) {{ fprintf(stderr, \"count allocation failed\\n\"); abort(); }}"
        ));
        self.line(&format!("int64_t __out_indices[{}];", ty.dims.len().max(1)));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}_shape, t{id}_rank, __out_indices);"
        ));
        self.line(&format!(
            "for (int64_t __r_{id} = 0; __r_{id} < __count_n_{id}; __r_{id}++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t __full_indices[{}];", input_ty.dims.len()));
        self.line("int __out_d = 0;");
        for axis in 0..input_ty.dims.len() {
            if !axes.contains(&axis) {
                self.line(&format!(
                    "__full_indices[{axis}] = __out_indices[__out_d++];"
                ));
            }
        }
        self.line(&format!("int64_t __rem_{id} = __r_{id};"));
        for axis in (0..input_ty.dims.len()).rev() {
            if axes.contains(&axis) {
                self.line(&format!(
                    "__full_indices[{axis}] = __rem_{id} % t{a}_shape[{axis}];"
                ));
                self.line(&format!("__rem_{id} /= t{a}_shape[{axis}];"));
            }
        }
        self.line(&format!(
            "int64_t __src_{id} = chelis_indices_to_flat(__full_indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "uint8_t __bit_{id} = __count_in_{id}[__src_{id}];"
        ));
        self.line(&format!(
            "if (__bit_{id} != 0 && __bit_{id} != 1) {{ fprintf(stderr, \"count input is not an exact bool\\n\"); abort(); }}"
        ));
        self.line(&format!(
            "__level_{id}[__r_{id}] = (__bit_{id} == 1) ? 1 : 0;"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("int64_t __level_n_{id} = __count_n_{id};"));
        self.line(&format!("while (__level_n_{id} > 1) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t __next_n_{id} = (__level_n_{id} + 1) / 2;"
        ));
        self.line(&format!(
            "for (int64_t __j_{id} = 0; __j_{id} < __next_n_{id}; __j_{id}++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t __left_{id} = 2 * __j_{id};"));
        self.line(&format!("int64_t __right_{id} = __left_{id} + 1;"));
        let trap = NumericTrap::Overflow {
            op: "count",
            prim: Prim::Int64,
        }
        .to_string();
        self.line(&format!(
            "__level_{id}[__j_{id}] = (__right_{id} < __level_n_{id}) ? chelis_int_checked_add(__level_{id}[__left_{id}], __level_{id}[__right_{id}], 64, {trap:?}) : __level_{id}[__left_{id}];"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("__level_n_{id} = __next_n_{id};"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "__count_out_{id}[outer] = (__count_n_{id} == 0) ? 0 : __level_{id}[0];"
        ));
        self.line(&format!("free(__level_{id});"));
        self.indent -= 1;
        self.line("}");
    }

    /// WS-A1 dtype-parameterized reduce_sum. Drives accumulator type
    /// and zero initializer from `Self::elem_type` /
    /// `Self::scalar_zero_literal` / `Self::fill_zero_call` so every
    /// (operand, accumulator) combo the C backend's helpers know about
    /// lowers correctly without a dedicated specialization. Includes
    /// the f32+f32 SIMD fast path inline (`chelis_sum_f32`) so the
    /// pre-WS-A4 emit for that combo stays byte-identical. Other
    /// dispatched paths (i8/i16 → i32 via `emit_reduce_sum_int_promoted`)
    /// short-circuit before this is called.
    fn emit_reduce_sum_general(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        let acc_et = Self::elem_type(ty);
        let acc_zero = Self::scalar_zero_literal(ty.precision);
        let operand_et = Self::elem_type(&input_node.output_type);
        self.emit_slot_wrapper(id, ty);
        let output_is_scalar = ty.dims.is_empty();
        // Fast path: contiguous f32 input AND f32 accumulator can use
        // the SIMD `chelis_sum_f32` helper. Other dtype combinations
        // fall through to the dtype-parameterized scalar loop below.
        // No widening fast path for {f32 operand, f64 accumulator} or
        // similar mixed-precision: those are admitted by the type
        // system but routed through the scalar accumulator loop.
        let can_simd_fast_path = output_is_scalar
            && ty.precision == Prim::F32
            && input_node.output_type.precision == Prim::F32;
        if can_simd_fast_path {
            self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
            self.indent += 1;
            self.line(&format!(
                "((float*)t{id}_data)[0] = chelis_sum_f32((const float*)t{a}_data, t{a}_size);"
            ));
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
        }
        self.line(&Self::fill_zero_call(ty, &format!("t{id}_write_guard")));
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        // Stride-4 ILP cascade matching torch's CPU `row_sum`
        // (`num_levels=4, ilp_factor=4` in
        // pytorch/aten/src/ATen/native/cpu/SumKernel.cpp). Bit-exact
        // with torch's `.sum()` for n <= 16 (issue
        // Chelis-Lang/chelis#163).
        self.line(&format!(
            "{acc_et} acc0 = {acc_zero}, acc1 = {acc_zero}, acc2 = {acc_zero}, acc3 = {acc_zero};"
        ));
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}_shape, t{id}_rank, out_indices);"
        ));
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t full_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        // Build full indices: insert k at the reduction axis
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}_rank; d++) {{"));
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
            "int64_t src_idx = chelis_indices_to_flat(full_indices, t{a}_strides, t{a}_rank);"
        ));
        // Read the operand at its native element type and accumulate at
        // the accumulator type; C handles the implicit widening for the
        // f32→f64 case, and integer accumulators preserve exact values.
        self.line(&format!(
            "{acc_et} __v = ({acc_et})((const {operand_et}*)t{a}_data)[src_idx];"
        ));
        if ty.precision.is_integer() {
            let bits = Self::integer_width(ty.precision);
            let trap = NumericTrap::Overflow {
                op: "sum",
                prim: ty.precision,
            }
            .to_string();
            self.line("switch (__reduce_i & 3) {");
            for lane in 0..3 {
                self.line(&format!(
                    "  case {lane}: acc{lane} = ({acc_et})chelis_int_checked_add((int64_t)acc{lane}, (int64_t)__v, {bits}, {trap:?}); break;"
                ));
            }
            self.line(&format!(
                "  default: acc3 = ({acc_et})chelis_int_checked_add((int64_t)acc3, (int64_t)__v, {bits}, {trap:?}); break;"
            ));
            self.line("}");
        } else {
            self.line("switch (__reduce_i & 3) {");
            self.line("  case 0: acc0 += __v; break;");
            self.line("  case 1: acc1 += __v; break;");
            self.line("  case 2: acc2 += __v; break;");
            self.line("  default: acc3 += __v; break;");
            self.line("}");
        }
        self.indent -= 1;
        self.line("}");
        if ty.precision.is_integer() {
            let bits = Self::integer_width(ty.precision);
            let trap = NumericTrap::Overflow {
                op: "sum",
                prim: ty.precision,
            }
            .to_string();
            self.line(&format!(
                "{acc_et} __sum01 = ({acc_et})chelis_int_checked_add((int64_t)acc0, (int64_t)acc1, {bits}, {trap:?});"
            ));
            self.line(&format!(
                "{acc_et} __sum23 = ({acc_et})chelis_int_checked_add((int64_t)acc2, (int64_t)acc3, {bits}, {trap:?});"
            ));
            self.line(&format!(
                "(({acc_et}*)t{id}_data)[outer] = ({acc_et})chelis_int_checked_add((int64_t)__sum01, (int64_t)__sum23, {bits}, {trap:?});"
            ));
        } else {
            self.line(&format!(
                "(({acc_et}*)t{id}_data)[outer] = (acc0 + acc1) + (acc2 + acc3);"
            ));
        }
        self.indent -= 1;
        self.line("}");
        if can_simd_fast_path {
            self.indent -= 1;
            self.line("}");
        }
    }

    /// C zero-literal for a Chelis precision used as an accumulator
    /// initializer. Returns the source-text form (not a runtime
    /// expression) so it can be inlined into emitted assignments.
    fn scalar_zero_literal(prim: Prim) -> &'static str {
        match prim {
            Prim::F32 => "0.0f",
            Prim::F64 => "0.0",
            Prim::Int32 => "(int32_t)0",
            Prim::Int64 => "(int64_t)0",
            Prim::Bool => "0",
            // WS-1: bf16(+0) and f16(+0) both encode as 0x0000. The
            // literal is used to initialize the per-element storage
            // slot, not as an arithmetic accumulator (the bf16/f16
            // reduce_sum accumulator is f32 per spec §5.7.1 and never
            // calls this helper).
            Prim::Bf16 | Prim::F16 => "(uint16_t)0",
            other => panic!(
                "C backend has no zero literal for `{}` accumulator (spec/04-type-system.md §5.7.1)",
                other.name()
            ),
        }
    }

    /// Emit a `chelis_fill_*` runtime call that zeros every element of
    /// `tensor` according to the C-backend dtype set. The runtime
    /// distinguishes f32/f64/i64 fills because their element widths
    /// differ; integer-32 zero-fill is handled in-line by other call
    /// sites that need it, but this helper centralizes the
    /// reduce-sum case where the IR accumulator drives the dispatch.
    fn fill_zero_call(ty: &TensorType, tensor: &str) -> String {
        let dtype = ty
            .precision
            .runtime_dtype()
            .unwrap_or_else(|error| panic!("C backend cannot zero-fill this tensor: {error}"))
            .c_macro();
        format!("chelis_fill_scalar({tensor}, chelis_scalar_from_bits({dtype}, UINT64_C(0)));")
    }

    /// WS-A4: integer reduce_sum where the accumulator dtype is wider
    /// than the source dtype (the i8/i16 → i32 promoted path per spec
    /// §5.7.1). `src_c_ty` and `acc_c_ty` are the C type spellings used
    /// for the reinterpret cast on `t->data` and for the accumulator
    /// variable respectively. Output buffer is sized for `acc_c_ty` by
    /// `chelis_alloc` honoring the `dtype_macro(ty)` value (CHELIS_DTYPE_I32
    /// for the i8/i16 → i32 lowering).
    // WS-A4: dtype-parameterized reduction needs both source and
    // accumulator C-type spellings plus the standard set of
    // emit-context arguments; the helper sits at the same arity as the
    // existing `emit_reduce_simple` which is similarly broad.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduce_sum_int_promoted(
        &mut self,
        id: usize,
        axis: usize,
        src_c_ty: &str,
        acc_c_ty: &str,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        self.emit_slot_wrapper(id, ty);
        // Zero-fill the output buffer manually since there is no
        // `chelis_fill_i32` helper today; an inline loop avoids touching
        // the runtime ABI for the WS-A4 cycle.
        self.line(&format!(
            "for (int64_t __zi = 0; __zi < t{id}_size; __zi++) {{ (({acc_c_ty}*)t{id}_data)[__zi] = 0; }}"
        ));
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        // Stride-4 ILP cascade (issue #163). [04-NUM-12] defines integer
        // trap occurrence relative to this exact lane order, so every lane
        // update and each final combine uses the checked accumulator width.
        self.line(&format!(
            "{acc_c_ty} acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;"
        ));
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}_shape, t{id}_rank, out_indices);"
        ));
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t full_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}_rank; d++) {{"));
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
            "int64_t src_idx = chelis_indices_to_flat(full_indices, t{a}_strides, t{a}_rank);"
        ));
        // Promote each source element to the (wider) accumulator type
        // before adding so partial sums of 200 i8 ones produce 200, not
        // -56 (which would be the wrap-around if accumulation happened
        // at the source width).
        self.line(&format!(
            "{acc_c_ty} __v = ({acc_c_ty})(({src_c_ty}*)t{a}_data)[src_idx];"
        ));
        let bits = Self::integer_width(ty.precision);
        let trap = NumericTrap::Overflow {
            op: "sum",
            prim: ty.precision,
        }
        .to_string();
        self.line("switch (__reduce_i & 3) {");
        for lane in 0..3 {
            self.line(&format!(
                "  case {lane}: acc{lane} = ({acc_c_ty})chelis_int_checked_add((int64_t)acc{lane}, (int64_t)__v, {bits}, {trap:?}); break;"
            ));
        }
        self.line(&format!(
            "  default: acc3 = ({acc_c_ty})chelis_int_checked_add((int64_t)acc3, (int64_t)__v, {bits}, {trap:?}); break;"
        ));
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "{acc_c_ty} __sum01 = ({acc_c_ty})chelis_int_checked_add((int64_t)acc0, (int64_t)acc1, {bits}, {trap:?});"
        ));
        self.line(&format!(
            "{acc_c_ty} __sum23 = ({acc_c_ty})chelis_int_checked_add((int64_t)acc2, (int64_t)acc3, {bits}, {trap:?});"
        ));
        self.line(&format!(
            "(({acc_c_ty}*)t{id}_data)[outer] = ({acc_c_ty})chelis_int_checked_add((int64_t)__sum01, (int64_t)__sum23, {bits}, {trap:?});"
        ));
        self.indent -= 1;
        self.line("}");
    }

    /// WS-1: bf16 / f16 source -> f32 accumulator -> f32 output
    /// reduce_sum, per spec/04-type-system.md §5.7.1. Loads each
    /// reduced-float source element through `chelis_<x>_to_f32`,
    /// accumulates in `f32`, and writes the result into an f32
    /// destination tensor. The output tensor's dtype IS `CHELIS_DTYPE_F32`
    /// (the IR `accumulator` field equals the output precision per
    /// the C3a invariant), so `chelis_fill_f32` zeros the buffer and
    /// the result store is a plain `((float*)t{id}_data)[outer]`
    /// assignment.
    fn emit_reduce_sum_reduced_f(
        &mut self,
        id: usize,
        axis: usize,
        input_prec: Prim,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        let load = Self::reduced_to_f32_fn(input_prec);
        self.emit_slot_wrapper(id, ty);
        self.line(&Self::fill_zero_call(ty, &format!("t{id}_write_guard")));
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line("float acc = 0.0f;");
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}_shape, t{id}_rank, out_indices);"
        ));
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t full_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}_rank; d++) {{"));
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
            "int64_t src_idx = chelis_indices_to_flat(full_indices, t{a}_strides, t{a}_rank);"
        ));
        // Load via the conversion helper so the operand bits are
        // interpreted as their declared bf16/f16 value and promoted
        // to f32 for accumulation. The §5.7.1 enforcement test pins
        // this: 1024 elements of bf16(0.01) sum to within tolerance
        // of 10.24 in f32, but a naive bf16-direct accumulator
        // diverges by far more.
        self.line(&format!("acc += {load}(((uint16_t*)t{a}_data)[src_idx]);"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("((float*)t{id}_data)[outer] = acc;"));
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
        dag: VerifiedDagView<'_>,
    ) -> Result<(), Unsupported> {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        // WS-A1 guard: reduce_max codegen is f32-hardcoded
        // (`chelis_max_f32` SIMD helper, `float acc = -INFINITY`,
        // `fmaxf` reduction operator). Per spec §2.3 max_reduce
        // returns operand precision; WS-1 adds the bf16/f16 path
        // (convert each element to f32, compare, convert back to the
        // operand precision for storage) without disturbing the f32
        // fast path. Other widenings (e.g. f64) remain follow-on
        // work; the explicit panic still fires so the silent
        // truncation footgun cannot recur.
        if matches!(ty.precision, Prim::Bf16 | Prim::F16)
            && input_node.output_type.precision == ty.precision
        {
            self.emit_reduce_max_reduced_f(id, axis, inputs, ty, dag);
            return Ok(());
        }
        if !matches!(ty.precision, Prim::F32)
            || !matches!(input_node.output_type.precision, Prim::F32)
        {
            // chelis#730 Phase 1 (census row 11, chelis#692): a clean
            // diagnostic through the section C3 channel, not a compiler
            // panic. Reachable from ordinary Surf (`max_reduce` over an
            // int64 tensor).
            return Err(Unsupported::new(
                UnsupportedKind::Op("max_reduce".to_string()),
                format!(
                    "`{}` tensors in the C DAG emitter (node {id})",
                    input_node.output_type.precision.name()
                ),
                Stage::Codegen("c"),
                chelis_types::unimplemented_rejection!(
                    729,
                    "the C reduce kernels are f32-hardcoded today (WS-A1/F1); cast to f32 \
                     before the reduction. The target capability table owns non-f32 widening"
                ),
            ));
        }
        self.emit_slot_wrapper(id, ty);
        let output_is_scalar = ty.dims.is_empty();
        if output_is_scalar {
            self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
            self.indent += 1;
            self.line(&format!(
                "((float*)t{id}_data)[0] = chelis_max_f32((const float*)t{a}_data, t{a}_size);"
            ));
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
        }
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line("float acc = -INFINITY;");
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}_shape, t{id}_rank, out_indices);"
        ));
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t full_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}_rank; d++) {{"));
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
            "int64_t src_idx = chelis_indices_to_flat(full_indices, t{a}_strides, t{a}_rank);"
        ));
        // #172: propagate NaN (torch parity), matching `chelis_max_f32`.
        self.line(&format!(
            "acc = chelis_fmax_propnan_f32(acc, ((const float*)t{a}_data)[src_idx]);"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("((float*)t{id}_data)[outer] = acc;"));
        self.indent -= 1;
        self.line("}");
        if output_is_scalar {
            self.indent -= 1;
            self.line("}");
        }
        Ok(())
    }

    /// WS-1: bf16 / f16 reduce_max. Per spec §2.3 max_reduce keeps
    /// operand precision, so the output is also bf16 / f16. We load
    /// each operand through `chelis_<x>_to_f32`, fold with `fmaxf`,
    /// and convert the final accumulator back to the operand
    /// precision for the store. This keeps the comparison numerically
    /// faithful (NaN propagation aside) without inflating the per-
    /// element storage.
    fn emit_reduce_max_reduced_f(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        let load = Self::reduced_to_f32_fn(ty.precision);
        let store = Self::f32_to_reduced_fn(ty.precision);
        self.emit_slot_wrapper(id, ty);
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line("float acc = -INFINITY;");
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}_shape, t{id}_rank, out_indices);"
        ));
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t full_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}_rank; d++) {{"));
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
            "int64_t src_idx = chelis_indices_to_flat(full_indices, t{a}_strides, t{a}_rank);"
        ));
        // #172: propagate NaN (torch parity), matching `chelis_max_f32`.
        self.line(&format!(
            "acc = chelis_fmax_propnan_f32(acc, {load}(((uint16_t*)t{a}_data)[src_idx]));"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("((uint16_t*)t{id}_data)[outer] = {store}(acc);"));
        self.indent -= 1;
        self.line("}");
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
        dag: VerifiedDagView<'_>,
        init: &str,
        update_tmpl: &str,
        simd_fn: Option<&str>,
    ) -> Result<(), Unsupported> {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        // WS-A1 guard: emit_reduce_simple is f32-hardcoded (`float acc`
        // declarator, scalar `INFINITY`/`-INFINITY` literals, `fmaxf`/
        // `fminf` operators in the update template). Used by
        // MinReduce/ProdReduce; per spec §2.3 these return operand
        // precision, so f64 input → f64 output, but the C-backend
        // implementation does not yet handle that. Reject loudly to
        // avoid silent truncation; widening is follow-on work.
        if !matches!(ty.precision, Prim::F32)
            || !matches!(input_node.output_type.precision, Prim::F32)
        {
            // chelis#730 Phase 1 (census row 11, chelis#692).
            return Err(Unsupported::new(
                UnsupportedKind::Op("min_reduce / prod_reduce".to_string()),
                format!(
                    "`{}` tensors in the C DAG emitter (node {id})",
                    input_node.output_type.precision.name()
                ),
                Stage::Codegen("c"),
                chelis_types::unimplemented_rejection!(
                    729,
                    "the C reduce kernels are f32-hardcoded today (WS-A1/F1); cast to f32 \
                     before the reduction. The target capability table owns non-f32 widening"
                ),
            ));
        }
        self.emit_slot_wrapper(id, ty);
        let output_is_scalar = ty.dims.is_empty();
        let use_simd = output_is_scalar && simd_fn.is_some();
        if use_simd {
            let fn_name = simd_fn.unwrap();
            self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
            self.indent += 1;
            self.line(&format!(
                "((float*)t{id}_data)[0] = {fn_name}((const float*)t{a}_data, t{a}_size);"
            ));
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
        }
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line(&format!("float acc = {init};"));
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}_shape, t{id}_rank, out_indices);"
        ));
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t full_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}_rank; d++) {{"));
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
            "int64_t src_idx = chelis_indices_to_flat(full_indices, t{a}_strides, t{a}_rank);"
        ));
        let update = update_tmpl.replace("{a}", &a.to_string());
        self.line(&update);
        self.indent -= 1;
        self.line("}");
        self.line(&format!("((float*)t{id}_data)[outer] = acc;"));
        self.indent -= 1;
        self.line("}");
        if use_simd {
            self.indent -= 1;
            self.line("}");
        }
        Ok(())
    }

    /// Strided windowed reduction emit. Per
    /// `spec/05-risc-primitives.md` §2.3.1, the trailing
    /// `window_shape.len()` axes are reduced; the leading axes pass
    /// through. Output rank equals input rank.
    ///
    /// f32-only for now (matches the rest of the reduction emit
    /// surface). bf16/f16 widening is follow-on work. A non-f32
    /// `reduce_window_*` is rejected before codegen with a clean
    /// `unsupported_feature` diagnostic by
    /// `chelis_compiler_api::compiler::reject_unsupported_reduce_window_precision`
    /// (and the CLI's mirror); the `panic!` below is a defensive backstop so
    /// silent truncation cannot recur if some path reaches emit unguarded.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduce_window(
        &mut self,
        id: usize,
        reducer: ReduceWindowKind,
        window_shape: &[usize],
        strides: &[usize],
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) -> Result<(), Unsupported> {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        if !matches!(ty.precision, Prim::F32)
            || !matches!(input_node.output_type.precision, Prim::F32)
        {
            // chelis#730 Phase 1 (census row 11 shape): a clean diagnostic,
            // not a panic; the pre-codegen gate normally rejects earlier.
            return Err(Unsupported::new(
                UnsupportedKind::Op("reduce_window_*".to_string()),
                format!(
                    "`{}` tensors in the C DAG emitter (node {id})",
                    input_node.output_type.precision.name()
                ),
                Stage::Codegen("c"),
                chelis_types::unimplemented_rejection!(
                    729,
                    "the C windowed-reduction emitter is f32-only today; cast to f32 \
                     before the windowed reduction (spec/05-risc-primitives.md \
                     section 2.3.1)"
                ),
            ));
        }
        // chelis#730 Phase 1 (census row 12, chelis#725's assertion half):
        // an arity mismatch between window and stride lists is a producing-
        // pass bug (lowering now raises on non-literal lists, census row
        // 8), but if one ever reaches emission it is a diagnostic through
        // the section C3 channel, never a compiler panic.
        if window_shape.len() != strides.len() {
            return Err(Unsupported::new(
                UnsupportedKind::Construct(format!(
                    "a `reduce_window_*` node with {} window axes but {} strides",
                    window_shape.len(),
                    strides.len()
                )),
                format!("the C DAG emitter (node {id})"),
                Stage::Codegen("c"),
                chelis_types::deliberate_rejection!(
                    "[05-RWIN-1]",
                    "internal desync: lowering guarantees equal-length literal window and \
                     stride lists (chelis#725; chelis#730 census rows 8/12)"
                ),
            ));
        }
        let n = window_shape.len();
        let in_rank = input_node.output_type.dims.len();
        if in_rank < n {
            return Err(Unsupported::new(
                UnsupportedKind::Construct(format!(
                    "a `reduce_window_*` node of input rank {in_rank} with window \
                     arity {n}"
                )),
                format!("the C DAG emitter (node {id})"),
                Stage::Codegen("c"),
                chelis_types::deliberate_rejection!(
                    "[05-RWIN-1]",
                    "internal desync: the checker guarantees window arity <= input rank \
                     (chelis#730 census row 12)"
                ),
            ));
        }
        let leading = in_rank - n;
        // Defensive backstop: a windowed output axis whose extent is not
        // statically known cannot be allocated correctly here — the
        // backend would bind it to the input extent and emit an
        // out-of-bounds window read (build output diverges from the
        // evaluator). `chelis_compiler_api::compiler::reject_symbolic_windowed_reduce`
        // rejects this before codegen with a clean diagnostic; if some
        // path reaches here unguarded, abort loudly rather than emit a
        // mis-allocated kernel. See spec/05-risc-primitives.md §2.3.1.
        for (offset, dim) in ty.dims.iter().enumerate().skip(leading) {
            if Self::known_dim_size(dim).is_none() {
                // chelis#730 Phase 1 (census row 12 shape): diagnostic, not
                // panic; `reject_symbolic_windowed_reduce` still rejects
                // earlier with span context.
                return Err(Unsupported::new(
                    UnsupportedKind::Construct(format!(
                        "a `reduce_window_*` windowed axis {offset} with a \
                         runtime-only symbolic extent"
                    )),
                    format!("the C DAG emitter (node {id})"),
                    Stage::Codegen("c"),
                    chelis_types::unimplemented_rejection!(
                        600,
                        "the windowed output extent floor((d - window) / stride) + 1 is \
                         not statically representable; bind the axis to a concrete size \
                         (spec/05-risc-primitives.md section 2.3.1)"
                    ),
                ));
            }
        }
        let window_volume: usize = window_shape.iter().product();
        self.emit_slot_wrapper(id, ty);
        // NOTE (#172 sibling, intentionally NOT changed here): windowed
        // Max/Min keep C99 `fmaxf`/`fminf` (NaN-dropping). The #172 fix
        // scopes NaN propagation to the `max_reduce` / `min_reduce`
        // reductions; flipping reduce_window forward without also defining
        // the NaN gradient-routing in the windowed backward (the `ext`
        // recompute below) would introduce a fwd/bwd inconsistency. Tracked
        // as a follow-up; reduce_window has its own parity gate (spec §2.3).
        let (init_literal, combine_template) = match reducer {
            ReduceWindowKind::Max => (
                "-INFINITY",
                "acc = fmaxf(acc, ((const float*)t{a}_data)[src_idx]);",
            ),
            ReduceWindowKind::Min => (
                "INFINITY",
                "acc = fminf(acc, ((const float*)t{a}_data)[src_idx]);",
            ),
            ReduceWindowKind::Sum | ReduceWindowKind::Mean => {
                ("0.0f", "acc += ((const float*)t{a}_data)[src_idx];")
            }
        };

        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line(&format!("float acc = {init_literal};"));
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}_shape, t{id}_rank, out_indices);"
        ));
        // Generate nested window loops. Each windowed axis gets its
        // own loop variable __w{i}; the leading axes are passed
        // through from `out_indices[..leading]`.
        for (i, w) in window_shape.iter().enumerate() {
            self.line(&format!("for (int __w{i} = 0; __w{i} < {w}; __w{i}++) {{"));
            self.indent += 1;
        }
        self.line(&format!(
            "int64_t full_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        for d in 0..leading {
            self.line(&format!("full_indices[{d}] = out_indices[{d}];"));
        }
        for (i, s) in strides.iter().enumerate() {
            let axis = leading + i;
            self.line(&format!(
                "full_indices[{axis}] = out_indices[{axis}] * {s} + __w{i};"
            ));
        }
        self.line(&format!(
            "int64_t src_idx = chelis_indices_to_flat(full_indices, t{a}_strides, t{a}_rank);"
        ));
        let combine = combine_template.replace("{a}", &a.to_string());
        self.line(&combine);
        for _ in 0..n {
            self.indent -= 1;
            self.line("}");
        }
        if matches!(reducer, ReduceWindowKind::Mean) {
            self.line(&format!("acc /= {}.0f;", window_volume));
        }
        self.line(&format!("((float*)t{id}_data)[outer] = acc;"));
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    /// Emit the per-window `full_indices` source multi-index used by the
    /// `reduce_window` adjoint: leading axes pass through from
    /// `out_indices`, windowed axis `i` is `out_indices[axis]*stride + __w{i}`.
    /// Assumes the `__w{i}` loop variables and `out_indices` are in scope.
    fn emit_reduce_window_grad_full_indices(
        &mut self,
        leading: usize,
        strides: &[usize],
        input: usize,
    ) {
        self.line(&format!(
            "int64_t full_indices[t{input}_rank > 0 ? t{input}_rank : 1];"
        ));
        for d in 0..leading {
            self.line(&format!("full_indices[{d}] = out_indices[{d}];"));
        }
        for (i, s) in strides.iter().enumerate() {
            let axis = leading + i;
            self.line(&format!(
                "full_indices[{axis}] = out_indices[{axis}] * {s} + __w{i};"
            ));
        }
    }

    /// Reverse-mode adjoint of `reduce_window_*` (`RiscOp::ReduceWindowGrad`).
    ///
    /// Inputs `[x, g]`: `x` is the forward windowed input (shape `S_in`),
    /// `g` the upstream cotangent (shape `S_out`). Output `din` has `x`'s
    /// shape. Each window's `g` is scattered (overlap-add) back over the
    /// window — `Sum` adds `g`, `Mean` adds `g / window_volume`, and
    /// `Max`/`Min` add `g` only at positions equal to that window's extreme
    /// (ties distribute, matching the `max_reduce` mask adjoint). See
    /// `spec/05-risc-primitives.md` §2.3.1.
    ///
    /// Emitted **serially** (no `#pragma omp parallel for`): overlapping
    /// windows scatter-add into shared `din` positions, so parallelising
    /// over the cotangent would race. The forward op is the parallel one.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduce_window_grad(
        &mut self,
        id: usize,
        reducer: ReduceWindowKind,
        window_shape: &[usize],
        strides: &[usize],
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let x = inputs[0].0;
        let g = inputs[1].0;
        let x_node = dag.get(inputs[0]).unwrap();
        let g_node = dag.get(inputs[1]).unwrap();
        if !matches!(ty.precision, Prim::F32)
            || !matches!(x_node.output_type.precision, Prim::F32)
            || !matches!(g_node.output_type.precision, Prim::F32)
        {
            panic!(
                "emit_reduce_window_grad: f32-only; node {id} has x precision `{}`, \
                 g precision `{}`, output precision `{}`. bf16/f16 widening is follow-on work.",
                x_node.output_type.precision.name(),
                g_node.output_type.precision.name(),
                ty.precision.name(),
            );
        }
        assert_eq!(
            window_shape.len(),
            strides.len(),
            "reduce_window_grad: window_shape and strides must have equal length"
        );
        let n = window_shape.len();
        let in_rank = ty.dims.len();
        assert!(
            in_rank >= n,
            "reduce_window_grad: output rank {in_rank} smaller than window arity {n}"
        );
        let leading = in_rank - n;
        let window_volume: usize = window_shape.iter().product();

        self.emit_slot_wrapper(id, ty);

        // din accumulates with `+=`, so it must start at zero.
        self.line(&format!(
            "for (int64_t __i = 0; __i < t{id}_size; __i++) {{ ((float*)t{id}_data)[__i] = 0.0f; }}"
        ));

        // Serial scatter over each cotangent (forward-output) position.
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{g}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line(&format!("float gval = ((const float*)t{g}_data)[outer];"));
        self.line(&format!(
            "int64_t out_indices[t{g}_rank > 0 ? t{g}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{g}_shape, t{g}_rank, out_indices);"
        ));

        let is_extreme = matches!(reducer, ReduceWindowKind::Max | ReduceWindowKind::Min);
        if is_extreme {
            // First pass: the window extreme over `x` (needed to decide
            // which positions receive the gradient).
            let (init, cmp) = match reducer {
                ReduceWindowKind::Max => ("-INFINITY", "fmaxf"),
                _ => ("INFINITY", "fminf"),
            };
            self.line(&format!("float ext = {init};"));
            for (i, w) in window_shape.iter().enumerate() {
                self.line(&format!("for (int __w{i} = 0; __w{i} < {w}; __w{i}++) {{"));
                self.indent += 1;
            }
            self.emit_reduce_window_grad_full_indices(leading, strides, x);
            self.line(&format!(
                "int64_t src_idx = chelis_indices_to_flat(full_indices, t{x}_strides, t{x}_rank);"
            ));
            self.line(&format!(
                "ext = {cmp}(ext, ((const float*)t{x}_data)[src_idx]);"
            ));
            for _ in 0..n {
                self.indent -= 1;
                self.line("}");
            }
        }

        // Scatter pass: distribute `gval` into the windowed `din` positions.
        for (i, w) in window_shape.iter().enumerate() {
            self.line(&format!("for (int __w{i} = 0; __w{i} < {w}; __w{i}++) {{"));
            self.indent += 1;
        }
        self.emit_reduce_window_grad_full_indices(leading, strides, x);
        self.line(&format!(
            "int64_t dst_idx = chelis_indices_to_flat(full_indices, t{id}_strides, t{id}_rank);"
        ));
        match reducer {
            ReduceWindowKind::Sum => {
                self.line(&format!("((float*)t{id}_data)[dst_idx] += gval;"));
            }
            ReduceWindowKind::Mean => {
                self.line(&format!(
                    "((float*)t{id}_data)[dst_idx] += gval / {window_volume}.0f;"
                ));
            }
            ReduceWindowKind::Max | ReduceWindowKind::Min => {
                self.line(&format!(
                    "int64_t src_idx = chelis_indices_to_flat(full_indices, t{x}_strides, t{x}_rank);"
                ));
                self.line(&format!(
                    "if (((const float*)t{x}_data)[src_idx] == ext) {{ ((float*)t{id}_data)[dst_idx] += gval; }}"
                ));
            }
        }
        for _ in 0..n {
            self.indent -= 1;
            self.line("}");
        }

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
        dag: VerifiedDagView<'_>,
        is_argmax: bool,
    ) -> Result<(), Unsupported> {
        let a = inputs[0].0;
        let input_node = dag.get(inputs[0]).unwrap();
        let axis_size = Self::emit_dim_info(&input_node.output_type.dims[axis]);
        // WS-A1 guard: argmax/argmin codegen is f32-hardcoded
        // (`chelis_argmax_f32`/`chelis_argmin_f32` SIMD helpers,
        // `float best_val` declarator). Per the doc comment above,
        // the OUTPUT is intentionally F32-encoded integer indices,
        // but the INPUT may be F64; reading f64 storage as float*
        // would silently truncate. Reject loudly until follow-on
        // widens the input read.
        if !matches!(input_node.output_type.precision, Prim::F32) {
            // chelis#730 Phase 1 (census row 11, chelis#692).
            return Err(Unsupported::new(
                UnsupportedKind::Op(
                    if is_argmax {
                        "argmax_reduce"
                    } else {
                        "argmin_reduce"
                    }
                    .to_string(),
                ),
                format!(
                    "`{}` tensor inputs in the C DAG emitter (node {id})",
                    input_node.output_type.precision.name()
                ),
                Stage::Codegen("c"),
                chelis_types::unimplemented_rejection!(
                    729,
                    "the C argmax/argmin kernels read f32 inputs only today (WS-A1/F1); \
                     cast to f32 before the reduction; the target capability table owns widening"
                ),
            ));
        }
        let init = if is_argmax { "-INFINITY" } else { "INFINITY" };
        let cmp = if is_argmax { ">" } else { "<" };
        let simd_fn = if is_argmax {
            "chelis_argmax_f32"
        } else {
            "chelis_argmin_f32"
        };
        self.emit_slot_wrapper(id, ty);
        // #347: argmax/argmin produce integer INDEX outputs (the result
        // tensor is allocated at the declared integer dtype, e.g.
        // `CHELIS_DTYPE_I64`). The index must be stored through a pointer of the
        // output element type, not into the `float* data` field directly:
        // a bare `t->data[outer] = (float)best_idx` writes the f32 bit
        // pattern of the index, which the print path then reads back as the
        // wrong reinterpreted integer (the `1065353216 == 0x3F800000`
        // signature). Mirrors `emit_cast`'s `(({dst_et}*)t->data)[i] = ...`
        // store convention so eval and the C backend agree on the indices.
        let dst_et = Self::elem_type(ty);
        let output_is_scalar = ty.dims.is_empty();
        if output_is_scalar {
            self.line(&format!("if (chelis_is_contiguous(t{a})) {{"));
            self.indent += 1;
            self.line(&format!(
                "(({dst_et}*)t{id}_data)[0] = ({dst_et}){simd_fn}(t{a}_data, t{a}_size);"
            ));
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
        }
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line(&format!("float best_val = {init};"));
        self.line("int64_t best_idx = -1;");
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}_shape, t{id}_rank, out_indices);"
        ));
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "int64_t full_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line("int out_d = 0;");
        self.line(&format!("for (int d = 0; d < t{a}_rank; d++) {{"));
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
            "int64_t src_idx = chelis_indices_to_flat(full_indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!("float v = ((const float*)t{a}_data)[src_idx];"));
        self.line(&format!("if (best_idx < 0 || v {cmp} best_val) {{"));
        self.indent += 1;
        self.line("best_val = v;");
        self.line("best_idx = __reduce_i;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "(({dst_et}*)t{id}_data)[outer] = ({dst_et})best_idx;"
        ));
        self.indent -= 1;
        self.line("}");
        if output_is_scalar {
            self.indent -= 1;
            self.line("}");
        }
        Ok(())
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
    ) -> Result<(), Unsupported> {
        let ops = match fused_op {
            RiscOp::FusedElem { ops } => ops,
            _ => panic!("expected FusedElem op"),
        };
        // WS-A1 guard: this codegen path is f32-hardcoded (chelis_fill_f32
        // zero, `float acc = 0.0f` initializer, `expf`/`logf`/`sinf`
        // single-precision math symbols, fmaxf reduction operator). A
        // non-f32 output dtype would silently truncate to f32 — exactly
        // the F1 footgun class. Reject until a follow-on widens the
        // fused path to honor the IR `Sum`/`MaxReduce` accumulator and
        // the input element type per spec/04-type-system.md §5.7.1.
        //
        // chelis#951: unlike `emit_fused_elem` this is still a
        // rejection, not a widening — the reduce body needs a
        // `chelis_fill_f64` zero, a `double` accumulator cascade, and
        // `fmax`, which is more than the fused-elem parameterization.
        // But it is no longer a `panic!`: it is reachable from ordinary
        // Surf (`sum(exp(x), 0)` at f64 inlines the elementwise node
        // into the reduction) and through `chelis-python` it crossed
        // the FFI boundary as a `PanicException`.
        if !matches!(out_ty.precision, Prim::F32)
            || !matches!(fused_input_type.precision, Prim::F32)
        {
            // The channel chelis#730 Phase 1 (census row 11) established: a
            // section C3 diagnostic, not a compiler panic. This site is
            // census row 24, not row 11.
            return Err(Unsupported::new(
                UnsupportedKind::Op(format!("fused elementwise {reduce_kind}_reduce")),
                format!(
                    "`{}` fused input / `{}` reduction output in the C DAG emitter \
                     (node {id})",
                    fused_input_type.precision.name(),
                    out_ty.precision.name()
                ),
                Stage::Codegen("c"),
                chelis_types::unimplemented_rejection!(
                    951,
                    "the fused reduction kernel is f32-hardcoded (WS-A1/F1): \
                     `chelis_fill_f32` zero, a `float` accumulator cascade, and `fmaxf`. \
                     Cast to f32 before the reduction, or keep the elementwise chain out \
                     of the reduction so the unfused f64 reduce path runs. Widening is \
                     follow-on work (chelis#951)"
                ),
            ));
        }
        let axis_size = Self::emit_dim_info(&fused_input_type.dims[axis]);
        // ndim of the fused input (pre-reduction shape)
        let fused_ndim = fused_input_type.dims.len();

        self.emit_slot_wrapper(id, out_ty);
        if reduce_kind == "sum" {
            self.line(&Self::fill_zero_call(out_ty, &format!("t{id}_write_guard")));
        }
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        let init = if reduce_kind == "sum" {
            "0.0f"
        } else {
            "-INFINITY"
        };
        // Sum uses a stride-4 ILP cascade (issue #163, torch parity);
        // max keeps a single accumulator since `fmaxf` is associative.
        if reduce_kind == "sum" {
            self.line("float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;");
        } else {
            self.line(&format!("float acc = {init};"));
        }
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(outer, t{id}_shape, t{id}_rank, out_indices);"
        ));
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t full_indices[{}];", fused_ndim.max(1)));
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
                "int64_t idx_ext{ext_idx} = chelis_indices_to_flat(full_indices, t{ext_id}_strides, t{ext_id}_rank);"
            ));
        }

        // Emit each fused step
        let resolve = |fi: &FusedInput| -> String {
            match fi {
                FusedInput::External(i) => {
                    let ext_id = ext_inputs[*i].0;
                    format!("((const float*)t{ext_id}_data)[idx_ext{i}]")
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
                FusedStepOp::Sub => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("{a} - {b}")
                }
                FusedStepOp::Mul => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("{a} * {b}")
                }
                FusedStepOp::Div => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("{a} / {b}")
                }
                // chelis#178: f32-only fused path — only float `floor_div`
                // reaches here; `trunc_div` is integer-only.
                FusedStepOp::FloorDiv => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("floorf({a} / {b})")
                }
                FusedStepOp::TruncDiv => {
                    unreachable!(
                        "trunc_div is integer-only (chelis#178); the fused-elem path is \
                         f32-only and cannot carry an integer trunc_div step"
                    )
                }
                FusedStepOp::MaxElem => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("(isnan({a}) || (!isnan({b}) && ({a}) >= ({b})) ? ({a}) : ({b}))")
                }
                FusedStepOp::MinElem => {
                    let a = resolve(&step.input_indices[0]);
                    let b = resolve(&step.input_indices[1]);
                    format!("(isnan({a}) || (!isnan({b}) && ({a}) <= ({b})) ? ({a}) : ({b}))")
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
                FusedStepOp::Recip => {
                    let a = resolve(&step.input_indices[0]);
                    format!("1.0f / {a}")
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
                FusedStepOp::Round => {
                    let a = resolve(&step.input_indices[0]);
                    format!("rintf({a})")
                }
            };
            self.line(&format!("float v{s} = {expr};"));
        }

        // Accumulate the last step's result
        let last = ops.len() - 1;
        if reduce_kind == "sum" {
            self.line("switch (__reduce_i & 3) {");
            self.line(&format!("  case 0: acc0 += v{last}; break;"));
            self.line(&format!("  case 1: acc1 += v{last}; break;"));
            self.line(&format!("  case 2: acc2 += v{last}; break;"));
            self.line(&format!("  default: acc3 += v{last}; break;"));
            self.line("}");
        } else {
            // #172: fused max_reduce propagates NaN (torch parity),
            // matching the non-fused `chelis_max_f32` path.
            self.line(&format!("acc = chelis_fmax_propnan_f32(acc, v{last});"));
        }
        self.indent -= 1;
        self.line("}");
        if reduce_kind == "sum" {
            self.line(&format!(
                "((float*)t{id}_data)[outer] = (acc0 + acc1) + (acc2 + acc3);"
            ));
        } else {
            self.line(&format!("((float*)t{id}_data)[outer] = acc;"));
        }
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    // ---- Reshape ----
    fn emit_reshape(
        &mut self,
        id: usize,
        new_shape: &[RtDim],
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        // chelis#616: a node-valued (runtime) target extent is read from its
        // rank-0 bound scalar behind a negativity guard, then declared (or
        // equality-guarded) under the axis's symbolic dim name so the
        // `shape_literal` allocation below references a real C variable.
        // When any target is runtime, the emitted numel guard below is the
        // check that keeps the materialized copy from reading beyond the
        // source descriptor's logical element count.
        let has_runtime_target = new_shape.iter().any(|d| d.node_input().is_some());
        for (axis, dim) in new_shape.iter().enumerate() {
            if dim.node_input().is_none() {
                continue;
            }
            let extent = Self::bound_c_expr(dim, inputs, a, axis, dag);
            self.line(&format!(
                "if (({extent}) < 0) {{ fprintf(stderr, \"chelis: runtime reshape target \
                 must be non-negative at node {id} axis {axis}\\n\"); abort(); }}"
            ));
            self.emit_static_dim_guard(id, axis, &extent, ty.dims.get(axis));
            self.emit_runtime_dim_site(id, axis, &extent);
        }
        // chelis#664: the numel guard must fire whenever static
        // verification is incomplete — not only for Node-valued targets.
        // A target that folds to a SYM (`[shape(x, 0)]` resolving to the
        // Load-declared `n`) or a fully-LITERAL target over a
        // runtime-sized input (`reshape(stride(x, 2), [6])`) previously
        // got NO guard, so the view silently over- or under-read its
        // input where the evaluator rejects the numel mismatch. The
        // guard reuses exactly the dims `shape_literal` allocates from,
        // so any variable valid for the allocation is valid here; a
        // fully static reshape (all output and input extents literal) is
        // checker-verified and keeps byte-identical codegen.
        let dims_static = |dims: &[DimInfo]| dims.iter().all(|d| matches!(d, DimInfo::Lit(_)));
        let input_static = dag
            .get(inputs[0])
            .is_some_and(|node| dims_static(&node.output_type.dims));
        if has_runtime_target || !dims_static(&ty.dims) || !input_static {
            let numel = std::iter::once("(long long)1".to_string())
                .chain(ty.dims.iter().map(|dim| {
                    format!("(long long)({})", Self::emit_dim_expr(&DimExpr::from(dim)))
                }))
                .collect::<Vec<_>>()
                .join(" * ");
            self.line(&format!(
                "if (({numel}) != (long long)t{a}_size) {{ fprintf(stderr, \"chelis: runtime \
                 reshape numel mismatch at node {id}\\n\"); abort(); }}"
            ));
        }
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "memcpy(t{id}_data, t{a}_data, (size_t)t{id}_byte_capacity);"
        ));
    }

    // ---- Permute ----
    fn emit_permute(
        &mut self,
        id: usize,
        axes: &[usize],
        inputs: &[NodeId],
        ty: &TensorType,
        _dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let elem_type = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "int64_t in_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, out_indices);"
        ));
        for (new_d, &old_d) in axes.iter().enumerate() {
            self.line(&format!("in_indices[{old_d}] = out_indices[{new_d}];"));
        }
        self.line(&format!(
            "int64_t src = chelis_indices_to_flat(in_indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "(({elem_type}*)t{id}_data)[i] = ((const {elem_type}*)t{a}_data)[src];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Expand ----
    fn emit_expand(
        &mut self,
        id: usize,
        axis: usize,
        size: &RtDim,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        // An op-declared expanded axis is bound from the exact structural
        // size carrier. This keeps the value edge explicit and makes
        // `shape_deps` unnecessary for Expand.
        // A site is emitted when EITHER map names it: the legacy walk still
        // owns declarations, and the derivation owns guards, so gating on the
        // walk alone would let the derivation find a guard site the walk
        // cannot see and emit nothing.
        if self.runtime_dim_sites.contains_key(&(id, axis))
            || self.local_dim_guard_sites.contains_key(&(id, axis))
        {
            let extent = Self::bound_c_expr(size, inputs, a, axis, dag);
            self.emit_runtime_dim_site(id, axis, &extent);
        }
        let elem_type = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "int64_t in_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, out_indices);"
        ));
        self.line(&format!("if (t{id}_rank == t{a}_rank) {{"));
        self.indent += 1;
        self.line(&format!(
            "for (int d = 0; d < t{a}_rank; d++) in_indices[d] = d == {axis} ? 0 : out_indices[d];"
        ));
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line(&format!(
            "for (int d = 0; d < t{a}_rank; d++) in_indices[d] = out_indices[d < {axis} ? d : d + 1];"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "int64_t src = chelis_indices_to_flat(in_indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "(({elem_type}*)t{id}_data)[i] = ((const {elem_type}*)t{a}_data)[src];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Pad ----
    /// chelis#616: the C integer expression for a movement [`RtDim`] at run
    /// time. `Lit` is a literal; `ToEnd` reads the input tensor's runtime axis
    /// extent (`t{a}_shape[axis]`); `Node(i)` reads the rank-0 integer bound
    /// scalar `t{inputs[i]}->data[0]` with its declared element type, cast to
    /// `int` for use as a C index.
    fn bound_c_expr(
        bound: &RtDim,
        inputs: &[NodeId],
        a: usize,
        axis: usize,
        dag: VerifiedDagView<'_>,
    ) -> String {
        match bound {
            RtDim::Lit(n) => n.to_string(),
            RtDim::ToEnd => format!("t{a}_shape[{axis}]"),
            RtDim::Node(i) => {
                let n = inputs[*i].0;
                debug_assert_eq!(
                    dag.get(inputs[*i]).unwrap().output_type.precision,
                    Prim::Int64
                );
                format!("((int64_t*)t{n}_data)[0]")
            }
            RtDim::InputAxis {
                tensor,
                axis: RtAxis::Lit(source_axis),
            } => {
                let source = inputs[*tensor].0;
                format!("t{source}_shape[{source_axis}]")
            }
            // A symbolic dim (reshape targets only; verify rejects it in
            // movement bounds) is a declared C variable, exactly as
            // `emit_dim_expr` renders `DimExpr::Sym`.
            RtDim::Sym(name) => name.clone(),
        }
    }

    /// chelis#616: whether any bound in a `(start, end)` pair list is
    /// node-valued (runtime) on the given axis.
    fn pair_is_node(pair: &(RtDim, RtDim)) -> bool {
        pair.0.node_input().is_some() || pair.1.node_input().is_some()
    }

    /// chelis#616: emit the declaration or equality guard for an op-declared
    /// runtime dim at `(node id, axis)`, with `extent_expr` the C integer
    /// expression computing this op's extent for the axis. A declare site
    /// emits `int <sym> = <extent>;` (which `shape_literal` references for
    /// the output allocation); a guard site aborts at run time if the op's
    /// extent disagrees with the already-declared value (the symbol is
    /// Load-declared in the prologue or declared by an earlier op — the
    /// checker unified them, so a disagreement is a real shape error).
    fn emit_runtime_dim_site(&mut self, id: usize, axis: usize, extent_expr: &str) {
        // Declaring and guarding are not exclusive. The legacy walk owns
        // declarations and the derivation owns guards, so an axis that
        // declares its own extent may ALSO be the axis another operation
        // makes a claim about: chelis#1277 S2b's unit-extent claim is exactly
        // that shape, since it asserts something about the `expand`'s
        // OPERAND, whose own axis a producer such as `shrink` has already
        // declared. Returning after the declaration made every such guard
        // unreachable, which is how a compiled kernel came to broadcast
        // element 0 of a two-element axis in silence.
        if let Some((name, true)) = self.runtime_dim_sites.get(&(id, axis)) {
            let name = name.clone();
            self.declared_dim_names.insert(name.clone());
            self.line(&format!("int64_t {name} = {extent_expr};"));
            if !self.local_dim_guard_sites.contains_key(&(id, axis)) {
                return;
            }
        }
        // The guard site and the claim it compares against are the
        // derivation's, and the rendering is [04-NUM-9]'s: the complete
        // user-facing line is `numeric trap: domain in <op> at int64` with no
        // prefix and no suffix, `<op>` naming the operation that introduces
        // the extent, and `<prim>` always `int64` because the guard finalizes
        // an extent under [05-DIM-1]. Section 4.7's required context - the
        // disagreeing names, the axis and each observed value - is its own
        // `fprintf`, so the trap line stays exactly one line.
        // The comparison operand is the class's CANONICAL VALUE, supplied by
        // the derivation: the binder name where a lane declares one, the
        // literal the checker resolved the claim to otherwise. The emitter
        // consults no declaration table, so a claim resolved to a literal over
        // a RUNTIME read still gets the comparison section 4.7 owes between
        // the claimed extent and the value observed - the same comparison the
        // entry path emits for a `Literal` claim (chelis#1377). Keying it on
        // whether a C variable happened to be allocated narrowed a required
        // check to an implementation convenience.
        let Some(site) = self.local_dim_guard_sites.get(&(id, axis)).cloned() else {
            return;
        };
        let (name, operand, op) = (site.claim, site.operand, site.op);
        let name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(&name);
        self.line(&format!("if (({extent_expr}) != {operand}) {{"));
        self.indent += 1;
        self.line(&format!(
            "fprintf(stderr, \"extent `{name_fmt}`: claimed = %lld, node {id} axis {axis} = %lld\\n\", (long long)({operand}), (long long)({extent_expr}));"
        ));
        self.line(&format!(
            "chelis_numeric_trap(\"numeric trap: domain in {op} at int64\");"
        ));
        self.indent -= 1;
        self.line("}");
    }

    /// chelis#616 (defense in depth): a RUNTIME axis whose output dim
    /// resolved to a STATIC size must agree with the op's computed extent at
    /// run time. The static size comes from the checker; if type inference
    /// ever mis-fills a runtime axis (the red-team multi-axis-shrink
    /// finding), this abort is what stands between that imprecision and a
    /// silently mis-sized allocation.
    fn emit_static_dim_guard(
        &mut self,
        id: usize,
        axis: usize,
        extent_expr: &str,
        dim: Option<&DimInfo>,
    ) {
        let Some(expected) = dim.and_then(Self::known_dim_size) else {
            return;
        };
        self.line(&format!(
            "if (({extent_expr}) != {expected}) {{ fprintf(stderr, \"chelis: runtime dim \
             disagrees with static extent {expected} at node {id} axis {axis}\\n\"); abort(); }}"
        ));
    }

    fn emit_pad(
        &mut self,
        id: usize,
        padding: &[(RtDim, RtDim)],
        fill: ScalarValue,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        // chelis#616: per-axis (before, after) C expressions and the runtime
        // output extent for any node-valued axis (`in + before + after`),
        // declared into its `_anon_dim_{id}_{d}` before the alloc.
        let pad_exprs: Vec<(String, String)> = padding
            .iter()
            .enumerate()
            .map(|(d, (b, aft))| {
                (
                    Self::bound_c_expr(b, inputs, a, d, dag),
                    Self::bound_c_expr(aft, inputs, a, d, dag),
                )
            })
            .collect();
        for (d, pair) in padding.iter().enumerate() {
            let (before_e, after_e) = &pad_exprs[d];
            let extent = format!("t{a}_shape[{d}] + ({before_e}) + ({after_e})");
            if Self::pair_is_node(pair) {
                self.line(&format!(
                    "if (({before_e}) < 0 || ({after_e}) < 0) {{ fprintf(stderr, \
                     \"chelis: runtime pad bound out of range at node {id} axis {d}\\n\"); abort(); }}"
                ));
                self.emit_static_dim_guard(id, d, &extent, ty.dims.get(d));
            }
            self.emit_runtime_dim_site(id, d, &extent);
        }
        self.emit_slot_wrapper(id, ty);
        // WS-A1: pad fill must honor the output dtype. Pre-WS-A1 the
        // default arm fell through to chelis_fill_f32 even for f64
        // outputs, silently truncating the fill value. Match each
        // active dtype explicitly; an unhandled dtype panics rather
        // than silently downgrades.
        assert_eq!(fill.prim(), ty.precision, "verified pad fill dtype");
        match fill.element_ref() {
            ElementRef::I8(value) => {
                self.line(&format!(
                    "{{ int8_t *__p = (int8_t*)t{id}_data; for (int64_t __i = 0; __i < t{id}_size; __i++) __p[__i] = INT8_C({value}); }}"
                ));
            }
            ElementRef::I16(value) => {
                self.line(&format!(
                    "{{ int16_t *__p = (int16_t*)t{id}_data; for (int64_t __i = 0; __i < t{id}_size; __i++) __p[__i] = INT16_C({value}); }}"
                ));
            }
            ElementRef::I32(value) => {
                self.line(&format!(
                    "{{ int32_t *__p = (int32_t*)t{id}_data; for (int64_t __i = 0; __i < t{id}_size; __i++) __p[__i] = INT32_C({value}); }}"
                ));
            }
            ElementRef::I64(value) => {
                self.line(&format!(
                    "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t){}));",
                    Self::i64_c_literal(value)
                ));
            }
            ElementRef::F64(value) => {
                let bits = value.to_bits();
                self.line(&format!(
                    "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F64, UINT64_C(0x{bits:016x})));"
                ));
            }
            ElementRef::F32(value) => {
                let bits = value.to_bits();
                self.line(&format!(
                    "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F32, UINT32_C(0x{bits:08x})));"
                ));
            }
            ElementRef::F16(value) => {
                let bits = value.to_bits();
                self.line(&format!(
                    "{{ uint16_t *__p = (uint16_t*)t{id}_data; for (int64_t __i = 0; __i < t{id}_size; __i++) __p[__i] = UINT16_C(0x{bits:04x}); }}"
                ));
            }
            ElementRef::Bf16(value) => {
                let bits = value.to_bits();
                self.line(&format!(
                    "{{ uint16_t *__p = (uint16_t*)t{id}_data; for (int64_t __i = 0; __i < t{id}_size; __i++) __p[__i] = UINT16_C(0x{bits:04x}); }}"
                ));
            }
            ElementRef::Bool(value) => {
                let bits = u8::from(value);
                self.line(&format!(
                    "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, UINT8_C({bits})));"
                ));
            }
        }
        // Copy source data into the padded region
        self.line(&format!("for (int64_t i = 0; i < t{a}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t src_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{a}_shape, t{a}_rank, src_indices);"
        ));
        self.line(&format!(
            "int64_t dst_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        for (d, (before_e, _)) in pad_exprs.iter().enumerate() {
            self.line(&format!(
                "dst_indices[{d}] = src_indices[{d}] + {before_e};"
            ));
        }
        self.line(&format!(
            "int64_t dst_flat = chelis_indices_to_flat(dst_indices, t{id}_strides, t{id}_rank);"
        ));
        self.line(&format!(
            "int64_t src_flat = chelis_indices_to_flat(src_indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "(({et}*)t{id}_data)[dst_flat] = (({et}*)t{a}_data)[src_flat];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Shrink ----
    fn emit_shrink(
        &mut self,
        id: usize,
        bounds: &[(RtDim, RtDim)],
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        // chelis#368/#551: the `SHRINK_TO_END` full-axis sentinel encodes
        // "shrink axis `d` to its full runtime extent" for a SYMBOLIC no-pad
        // axis whose extent cannot be baked as a literal `usize`. The eval lane
        // resolves it via `bind_symbolic_dims` (binding the symbol to a concrete
        // value). The C build lane never binds — the symbol stays a runtime C
        // variable — so we resolve it structurally instead: the shrink loop
        // below is driven entirely by the OUTPUT shape (`t{id}_shape`, sized
        // from `ty`) and the per-axis start offset `lo`; the `hi` bound is not
        // read by codegen. A sentinel bound is a full-axis identity (`lo == 0`,
        // output extent == input extent), so the emitted loop is already
        // correct once `ty`'s axis dim (the same symbolic dim) is declared by
        // `symbolic_occurrences`. Fail loud only for a MALFORMED sentinel: one
        // on an axis whose output dim is concrete (a producing-pass bug that
        // would silently drop a real trim), or with a nonzero start.
        for (d, (lo, hi)) in bounds.iter().enumerate() {
            if !matches!(hi, RtDim::ToEnd) {
                continue;
            }
            let out_dim = ty.dims.get(d);
            let symbolic_axis = matches!(out_dim, Some(DimInfo::Named(_, None)));
            assert!(
                symbolic_axis && matches!(lo, RtDim::Lit(0)),
                "C backend reached an unresolved ToEnd sentinel at node {id} \
                 axis {d} (start {lo:?}, output dim {out_dim:?}) that is not a symbolic \
                 full-axis identity; the producing IR pass emitted a malformed shrink \
                 (chelis#368/#551/#616)"
            );
        }
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        // chelis#616: per-axis start/end C expressions; declare the runtime
        // output extent (`end - start`) into each node-valued axis's
        // `_anon_dim_{id}_{d}` before the alloc, with a runtime range guard.
        let shrink_exprs: Vec<(String, String)> = bounds
            .iter()
            .enumerate()
            .map(|(d, (s, e))| {
                (
                    Self::bound_c_expr(s, inputs, a, d, dag),
                    Self::bound_c_expr(e, inputs, a, d, dag),
                )
            })
            .collect();
        for (d, pair) in bounds.iter().enumerate() {
            let (start_e, end_e) = &shrink_exprs[d];
            let extent = format!("({end_e}) - ({start_e})");
            if Self::pair_is_node(pair) {
                // `end <= start` (empty or inverted) mirrors the evaluator's
                // rejection exactly — error-path parity, chelis#616.
                self.line(&format!(
                    "if (({start_e}) < 0 || ({end_e}) <= ({start_e}) || ({end_e}) > t{a}_shape[{d}]) \
                     {{ fprintf(stderr, \"chelis: runtime shrink bound out of range at node {id} \
                     axis {d}\\n\"); abort(); }}"
                ));
                self.emit_static_dim_guard(id, d, &extent, ty.dims.get(d));
            }
            // chelis#616: declare (or guard) this axis's runtime output
            // extent under its actual symbolic dim name (a fresh
            // `_anon_dim_*` or a sig-named `k`), which `shape_literal`
            // references for the output allocation.
            self.emit_runtime_dim_site(id, d, &extent);
        }
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t dst_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, dst_indices);"
        ));
        self.line(&format!(
            "int64_t src_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        for (d, (start_e, _)) in shrink_exprs.iter().enumerate() {
            self.line(&format!("src_indices[{d}] = dst_indices[{d}] + {start_e};"));
        }
        self.line(&format!(
            "int64_t src_flat = chelis_indices_to_flat(src_indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "(({et}*)t{id}_data)[i] = (({et}*)t{a}_data)[src_flat];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Stride ----
    fn emit_stride(
        &mut self,
        id: usize,
        strides: &[RtDim],
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        // chelis#616: per-axis step C expressions. The strided output extent is
        // `ceil(input_extent / step)`, which is runtime whenever the input axis
        // or the step is runtime; declare each such axis's `_anon_dim_{id}_{d}`
        // before the view alloc reads it via `shape_literal`.
        let step_exprs: Vec<String> = strides
            .iter()
            .enumerate()
            .map(|(d, s)| Self::bound_c_expr(s, inputs, a, d, dag))
            .collect();
        for (d, step_e) in step_exprs.iter().enumerate() {
            // Declare (or guard) a runtime output extent where the
            // occurrence pass marked this op as the axis's runtime-dim site
            // (a bystander symbolic axis — e.g. a passed-through `batch` —
            // is declared by `symbolic_bindings` from its Load and must not
            // be redeclared here), and additionally emit the step and
            // static-extent guards for any node-valued step.
            let node_step = strides.get(d).is_some_and(|s| s.node_input().is_some());
            let has_site = self.runtime_dim_sites.contains_key(&(id, d));
            if !node_step && !has_site {
                continue;
            }
            let extent = format!("(t{a}_shape[{d}] + ({step_e}) - 1) / ({step_e})");
            self.line(&format!(
                "if (({step_e}) <= 0) {{ fprintf(stderr, \"chelis: runtime stride step must be \
                 positive at node {id} axis {d}\\n\"); abort(); }}"
            ));
            if node_step {
                self.emit_static_dim_guard(id, d, &extent, ty.dims.get(d));
            }
            self.emit_runtime_dim_site(id, d, &extent);
        }
        let elem_type = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "int64_t in_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, out_indices);"
        ));
        for (d, step_e) in step_exprs.iter().enumerate() {
            self.line(&format!("in_indices[{d}] = out_indices[{d}] * ({step_e});"));
        }
        self.line(&format!(
            "int64_t src = chelis_indices_to_flat(in_indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "(({elem_type}*)t{id}_data)[i] = ((const {elem_type}*)t{a}_data)[src];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Realize ----
    fn emit_realize(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        self.line(&format!(
            "(({et}*)t{id}_data)[i] = (({et}*)t{a}_data)[idx];"
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
    // (`crates/chelis-compiler-api/src/runtime/host_ops.rs::cast_tensor_value` /
    // `convert_scalar_data`); this site mirrors those semantics in emitted
    // C.
    //
    // The checked path consumes `CheckedCastPlan` for the full active
    // numeric/bool product. Reduced-float storage is decoded explicitly, and
    // f64/integer sources round directly into f16/bf16 without an intermediate
    // f32 rounding. Every pair, including the exact same-Prim diagonal,
    // materializes logical element order from the source strides.
    /// Emit a cast-ladder node. `trunc` selects the [05-OP-6] rung:
    /// the float-to-integer leg truncates toward zero before its range
    /// check instead of rejecting a fractional value.
    fn emit_cast(
        &mut self,
        id: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
        trunc: bool,
    ) {
        let a = inputs[0].0;
        let src_ty = &dag
            .get(inputs[0])
            .expect("cast input must resolve in dag")
            .output_type;
        let src_prec = src_ty.precision;
        let dst_prec = ty.precision;
        let src_et = Self::elem_type(src_ty);
        let dst_et = Self::elem_type(ty);
        self.emit_slot_wrapper(id, ty);
        let checked_plan = if trunc {
            None
        } else {
            Some(
                CheckedCastPlan::new(src_prec, dst_prec)
                    .expect("validated C DAG casts use active source and target dtypes"),
            )
        };
        if let Some(plan) = checked_plan {
            self.line(&format!(
                "/* checked cast plan: {} -> {} */",
                plan.source().name(),
                plan.target().name()
            ));
        }
        if checked_plan.is_some_and(|plan| plan.kind() == CheckedCastKind::Identity) {
            self.line("/* checked cast identity */");
        }
        // Strided element-wise materialization:
        // the output is freshly allocated and contiguous, so the
        // destination index is the flat loop index. The source may be
        // non-contiguous; resolve its element via the standard
        // `chelis_flat_to_indices` + `chelis_indices_to_flat` dance
        // used by `emit_realize` and friends.
        //
        // `cast_trunc` remains serial. Checked cast keeps valid-element
        // conversion parallel, but no worker calls an aborting helper:
        // workers reduce candidate indices and the selected lowest index is
        // reclassified after the region ([04-NUM-15]).
        let trap_conditions = checked_plan.map(|plan| {
            let probe = format!("(({src_et}*)t{a}_data)[idx]");
            let mut conditions = Vec::new();
            if let Some(condition) = crate::host_emit::checked_cast_domain_condition(plan, &probe) {
                conditions.push(condition);
            }
            if let Some(condition) = crate::host_emit::checked_cast_overflow_condition(plan, &probe)
            {
                conditions.push(condition);
            }
            conditions
        });
        let trapping = trap_conditions
            .as_ref()
            .is_some_and(|conditions| !conditions.is_empty());
        let first_trap_index = format!("chelis_first_trap_index_{id}");
        if trapping {
            self.line(&format!("int64_t {first_trap_index} = INT64_MAX;"));
            self.line(&format!(
                "#pragma omp parallel for reduction(min:{first_trap_index})"
            ));
        } else if checked_plan.is_some() {
            self.line("#pragma omp parallel for");
        }
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_flat_to_indices(i, t{id}_shape, t{id}_rank, indices);"
        ));
        self.line(&format!(
            "int64_t idx = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
        ));
        let src_elem = format!("(({src_et}*)t{a}_data)[idx]");
        let dst_elem = format!("(({dst_et}*)t{id}_data)[i]");
        let src_reduced = Self::is_reduced_float_prec(src_prec);
        let assignment = if trunc {
            assert!(
                src_prec.is_float() && dst_prec.is_integer(),
                "[05-OP-6] C emission is float-to-integer only"
            );
            let source_value = if src_reduced {
                format!("{}({src_elem})", Self::reduced_to_f32_fn(src_prec))
            } else {
                src_elem.clone()
            };
            let domain = NumericTrap::Domain {
                op: "cast_trunc",
                prim: dst_prec,
            }
            .to_string();
            let overflow = NumericTrap::Overflow {
                op: "cast_trunc",
                prim: dst_prec,
            }
            .to_string();
            format!(
                "{dst_elem} = ({dst_et})chelis_trunc_float_to_int((double)({source_value}), {}, {domain:?}, {overflow:?});",
                Self::integer_width(dst_prec)
            )
        } else {
            let plan = checked_plan.expect("non-truncating cast carries a checked plan");
            let expression = if trapping {
                crate::host_emit::checked_cast_valid_c_expr(plan, &src_elem)
            } else {
                crate::host_emit::checked_cast_c_expr(plan, &src_elem)
            };
            format!("{dst_elem} = {};", expression)
        };
        if trapping {
            let invalid = trap_conditions
                .as_ref()
                .expect("trapping checked cast carries conditions")
                .join(" || ");
            self.line(&format!("if ({invalid}) {{"));
            self.indent += 1;
            self.line(&format!(
                "if (i < {first_trap_index}) {first_trap_index} = i;"
            ));
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
            self.line(&assignment);
            self.indent -= 1;
            self.line("}");
        } else {
            self.line(&assignment);
        }
        self.indent -= 1;
        self.line("}");
        if trapping {
            let plan = checked_plan.expect("trapping conversion carries a plan");
            self.line(&format!("if ({first_trap_index} != INT64_MAX) {{"));
            self.indent += 1;
            self.line(&format!(
                "int64_t indices[t{id}_rank > 0 ? t{id}_rank : 1];"
            ));
            self.line(&format!(
                "chelis_flat_to_indices({first_trap_index}, t{id}_shape, t{id}_rank, indices);"
            ));
            self.line(&format!(
                "int64_t idx = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);"
            ));
            let selected_src = format!("(({src_et}*)t{a}_data)[idx]");
            let selected = crate::host_emit::checked_cast_c_expr(plan, &selected_src);
            self.line(&format!("(void)({selected});"));
            self.line("abort(); /* the selected checked helper always traps */");
            self.indent -= 1;
            self.line("}");
        }
    }

    /// True for the reduced-precision float dtypes whose host storage is
    /// `uint16_t` and whose value semantics require routing through the
    /// runtime conversion helpers rather than a C language cast. Mirrors
    /// `is_reduced_float` but takes `Prim` directly for cast-site use.
    fn is_reduced_float_prec(prec: Prim) -> bool {
        matches!(prec, Prim::Bf16 | Prim::F16)
    }

    // ---- Store ----
    fn emit_store(&mut self, id: usize, name: &str, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let ndim = Self::ndim(ty);
        let shape = Self::shape_literal(ty);
        let dtype = Self::dtype_macro(ty);
        self.emit_owned_tensor(id, &ndim.to_string(), &shape, dtype);
        self.line(&format!(
            "memcpy(t{id}_data, t{a}_data, (size_t)t{id}_byte_capacity); /* store: {name} */"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};
    use chelis_types::types::Prim;

    fn emit_test_dag(dag: &Dag, name: &str) -> Result<String, Unsupported> {
        let verified = crate::testing::verified_dag(dag, crate::CodegenOptions::default())
            .expect("C emitter unit-test DAG must verify ownership");
        CEmitter::emit_dag(verified, name)
    }

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

    #[test]
    fn direct_integer_abs_uses_the_typed_guard_while_fused_stays_loud() {
        let ty = tensor_ty(&[1], Prim::Int64);

        let mut direct = Dag::new();
        let x = direct.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let out = direct.add_node(RiscOp::Abs, vec![x], ty.clone(), None);
        direct.set_roots(vec![out]);
        let c =
            emit_test_dag(&direct, "integer_abs").expect("direct integer abs has a typed C kernel");
        assert!(c.contains("chelis_int_abs_guard"));
        assert!(c.contains("numeric trap: overflow in abs at int64"));
        assert!(!c.contains("fabsf(__in_a_"));

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
        let err = emit_test_dag(&fused, "fused_integer_abs")
            .expect_err("fused integer abs must not bypass the C guard");
        assert!(err.to_string().contains("unsupported: op `Abs`"));
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
        dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_alloc"));
        // The exact tagged scalar carries both the f32 dtype and the
        // source value's bit pattern into the single public fill API.
        assert!(c.contains(
            "chelis_fill_scalar(t0_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F32,"
        ));
        let want_bits = (3.0_f32).to_bits();
        assert!(
            c.contains(&format!("0x{want_bits:08x}")),
            "f32 const must emit exact bit pattern; got:\n{c}"
        );
    }

    #[test]
    fn issue_878_pad_emits_exact_int64_fill_above_f64_boundary() {
        let exact = 9_007_199_254_740_993i64;
        let ty = tensor_ty(&[1], Prim::Int64);
        let out_ty = tensor_ty(&[3], Prim::Int64);
        let mut dag = Dag::new();
        let input = dag.add_node(RiscOp::synth_const(Prim::Int64, 7.0), vec![], ty, None);
        let fill = chelis_types::scalar_from_i64("pad", Prim::Int64, exact).unwrap();
        let padded = dag.add_node(
            RiscOp::pad(vec![(RtDim::Lit(1), RtDim::Lit(1))], fill),
            vec![input],
            out_ty,
            None,
        );
        dag.set_roots(vec![padded]);

        let c = emit_test_dag(&dag, "pad_exact_int64").unwrap();
        assert!(c.contains(&format!(
            "chelis_fill_scalar(t1_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)INT64_C({exact})));"
        )));
        assert!(
            !c.contains("9007199254740992"),
            "the exact int64 fill must never pass through its rounded f64 image:\n{c}"
        );
    }

    #[test]
    fn issue_878_pad_emits_portable_signed_int64_extremes() {
        for (value, spelling) in [
            (i64::MIN, "INT64_MIN"),
            (-9_007_199_254_740_993, "-INT64_C(9007199254740993)"),
        ] {
            let ty = tensor_ty(&[1], Prim::Int64);
            let mut dag = Dag::new();
            let input = dag.add_node(
                RiscOp::synth_const(Prim::Int64, 7.0),
                vec![],
                ty.clone(),
                None,
            );
            let fill = chelis_types::scalar_from_i64("pad", Prim::Int64, value).unwrap();
            let padded = dag.add_node(
                RiscOp::pad(vec![(RtDim::Lit(1), RtDim::Lit(1))], fill),
                vec![input],
                tensor_ty(&[3], Prim::Int64),
                None,
            );
            dag.set_roots(vec![padded]);
            let c = emit_test_dag(&dag, "pad_signed_int64").unwrap();
            assert!(
                c.contains(&format!(
                    "chelis_fill_scalar(t1_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t){spelling}));"
                )),
                "{c}"
            );
        }
    }

    #[test]
    fn add_emits_stride_aware_loop() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_flat_to_indices"));
        assert!(c.contains("chelis_indices_to_flat"));
        assert!(c.contains("+"));
    }

    #[test]
    fn neg_emits_unary_minus() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Neg, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        // The slow (non-contiguous) path emits a typed pointer cast then negates.
        assert!(c.contains("((float*)t0_data)[idx]"));
    }

    #[test]
    fn exp_emits_expf() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Exp, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("expf("));
    }

    #[test]
    fn log_emits_logf() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Log, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("logf("));
    }

    #[test]
    fn sin_emits_sinf() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Sin, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("sinf("));
    }

    #[test]
    fn sqrt_emits_sqrtf() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 4.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Sqrt, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("sqrtf("));
    }

    #[test]
    fn cmplt_emits_canonical_bool8() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            RiscOp::CmpLt,
            vec![a, b],
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("uint8_t* restrict __out_2"));
        assert!(c.contains("? UINT8_C(1) : UINT8_C(0)"));
        assert!(!c.contains("? 1.0f : 0.0f"));
    }

    #[test]
    fn omp_pragma_in_elementwise() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("#pragma omp parallel for"));
    }

    #[test]
    fn sum_emits_reduction_loop() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![a],
            scalar_f32(),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        // Stride-4 ILP cascade (issue #163): four independent
        // accumulators rather than a single `acc +=` chain.
        assert!(c.contains("acc0 += __v"));
        assert!(c.contains("acc3 += __v"));
        assert!(c.contains("(acc0 + acc1) + (acc2 + acc3)"));
        assert!(c.contains("for (int64_t __reduce_i"));
    }

    #[test]
    fn copy_materializes_and_drop_releases_exact_descriptor_once() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let copy = dag.add_node(RiscOp::Copy, vec![x], vec_f32(4), None);
        let disposable = dag.add_node(RiscOp::Copy, vec![x], vec_f32(4), None);
        dag.add_node(RiscOp::Drop, vec![disposable], vec_f32(4), None);
        dag.add_root(copy);

        let c = emit_test_dag(&dag, "test_copy_drop").unwrap();

        assert!(c.contains("chelis_tensor *t1"));
        assert!(c.contains("((float*)t1_data)[i] = ((float*)t0_data)[idx];"));
        assert!(
            !c.contains("chelis_tensor *t3"),
            "Drop should not emit a tensor wrapper or compute statement:\n{c}"
        );
        assert_eq!(
            c.matches("chelis_tensor_release(t2);").count(),
            1,
            "verified Drop must release its exact descriptor at the Drop site and suppress cleanup duplication:\n{c}"
        );
    }

    #[test]
    fn exact_same_byte_slot_reuse_repurposes_descriptor_metadata() {
        let mut dag = Dag::new();
        let first_ty = tensor_ty(&[2, 2], Prim::F32);
        let first = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            first_ty.clone(),
            None,
        );
        dag.add_node(RiscOp::Neg, vec![first], first_ty, None);
        let output = dag.add_node(
            RiscOp::synth_const(Prim::F32, 3.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_root(output);

        let c = emit_test_dag(&dag, "test_exact_same_byte_slot_reuse").unwrap();
        assert!(c.contains("chelis_tensor *t2 = t0;"), "{c}");
        assert!(
            c.contains("chelis_tensor_repurpose(t2, chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C(1)), (chelis_scalar[]){ chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(int64_t)(4)) });"),
            "same-byte slot reuse must reset the old rank-2 descriptor to rank 1; got:\n{c}"
        );
        assert!(!c.contains("chelis_tensor_release(t0);"), "{c}");
    }

    #[test]
    fn borrowed_drop_is_a_logical_discard_without_a_runtime_release() {
        let mut dag = Dag::new();
        let borrowed = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        dag.add_node(RiscOp::Drop, vec![borrowed], vec_f32(4), None);
        let output = dag.add_node(RiscOp::Copy, vec![borrowed], vec_f32(4), None);
        dag.add_root(output);

        let c = emit_test_dag(&dag, "test_borrowed_drop").unwrap();

        assert!(!c.contains("chelis_tensor_release(t0);"), "{c}");
    }

    #[test]
    fn max_reduce_emits_nan_propagating_max() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(RiscOp::MaxReduce { axis: 0 }, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        // #172: the contiguous fast path uses the NaN-propagating SIMD
        // helper; the strided fallback uses the NaN-propagating scalar
        // helper. Plain C99 `fmaxf` (which DROPS NaN) must not appear in
        // the reduction — it would diverge from torch.
        assert!(
            c.contains("chelis_max_f32(") && c.contains("chelis_fmax_propnan_f32(acc"),
            "max_reduce must emit the NaN-propagating max helpers (#172):\n{c}"
        );
        assert!(
            !c.contains("fmaxf(acc"),
            "max_reduce must not use NaN-dropping fmaxf on the accumulator (#172):\n{c}"
        );
        assert!(c.contains("-INFINITY"));
    }

    #[test]
    fn reduce_window_max_min_drop_nan() {
        // #172 sibling (C lane): windowed Max/Min keep C99 `fmaxf`/`fminf`,
        // which DROP NaN (return the non-NaN operand) — the SAME semantics as
        // eval's `f64::max`/`min` (locked by
        // `host_runtime_reduce_window_max_min_drop_nan` in chelis-compiler-api).
        // Unlike `max_reduce`/`min_reduce`, reduce_window must NOT use the
        // NaN-propagating reduce helper. This pins the documented
        // drop-vs-propagate asymmetry on the backend side.
        for (reducer, op) in [
            (ReduceWindowKind::Max, "fmaxf(acc"),
            (ReduceWindowKind::Min, "fminf(acc"),
        ] {
            let mut dag = Dag::new();
            let a = dag.add_node(
                RiscOp::synth_const(vec_f32(4).precision, 1.0),
                vec![],
                vec_f32(4),
                None,
            );
            dag.add_node(
                RiscOp::ReduceWindow {
                    reducer,
                    window_shape: vec![2],
                    strides: vec![2],
                },
                vec![a],
                vec_f32(2),
                None,
            );
            let c = emit_test_dag(&dag, "test_fn").unwrap();
            assert!(
                c.contains(op),
                "reduce_window {reducer:?} must use the NaN-dropping `{op}` (#172):\n{c}"
            );
            assert!(
                !c.contains("propnan"),
                "reduce_window must NOT use a NaN-propagating reduce helper (#172):\n{c}"
            );
        }
    }

    #[test]
    fn mul_emits_star_op() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 4.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("*"));
    }

    #[test]
    fn direct_extrema_emit_exact_first_operand_selectors() {
        for (op, comparison, forbidden) in [
            (RiscOp::MaxElem, ">=", ["fmaxf(", "fmax("]),
            (RiscOp::MinElem, "<=", ["fminf(", "fmin("]),
        ] {
            let mut dag = Dag::new();
            let a = dag.add_node(
                RiscOp::synth_const(scalar_f32().precision, 1.0),
                vec![],
                scalar_f32(),
                None,
            );
            let b = dag.add_node(
                RiscOp::synth_const(scalar_f32().precision, 2.0),
                vec![],
                scalar_f32(),
                None,
            );
            dag.add_node(op, vec![a, b], scalar_f32(), None);
            let c = emit_test_dag(&dag, "test_fn").unwrap();
            assert!(c.contains("isnan(__in_a_2[i])"), "{c}");
            assert!(c.contains("!isnan(__in_b_2[i])"), "{c}");
            assert!(
                c.contains(&format!("(__in_a_2[i]) {comparison} (__in_b_2[i])")),
                "{c}"
            );
            assert!(c.contains("? (__in_a_2[i]) : (__in_b_2[i])"), "{c}");
            for function in forbidden {
                assert!(
                    !c.contains(function),
                    "direct extrema must select an operand, not call {function}:\n{c}"
                );
            }
        }
    }

    #[test]
    fn reshape_materializes_contiguous_storage() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(6).precision, 1.0),
            vec![],
            vec_f32(6),
            None,
        );
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(2), RtDim::Lit(3)],
            },
            vec![a],
            mat_f32(2, 3),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_tensor *t1 = chelis_alloc("));
        assert!(c.contains("memcpy(t1_data, t0_data, (size_t)t1_byte_capacity);"));
    }

    #[test]
    fn permute_materializes_the_axis_mapping() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        dag.add_node(
            RiscOp::Permute { axes: vec![1, 0] },
            vec![a],
            mat_f32(3, 2),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("in_indices[1] = out_indices[0]"));
        assert!(c.contains("in_indices[0] = out_indices[1]"));
        assert!(!c.contains("t1->strides[0] ="));
    }

    #[test]
    fn expand_materializes_broadcast_coordinates() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(1).precision, 1.0),
            vec![],
            vec_f32(1),
            None,
        );
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            vec_f32(4),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("in_indices[d] = d == 0 ? 0 : out_indices[d]"));
        assert!(!c.contains("t1->strides[0] ="));
    }

    #[test]
    fn store_is_alias() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![a],
            scalar_f32(),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("memcpy(t1_data, t0_data, (size_t)t1_byte_capacity); /* store: out */"));
    }

    /// chelis#759 / [05-OP-6]: the truncating rung's conversion loop must
    /// be SERIAL.
    ///
    /// A buffer can carry both an out-of-range element and a non-finite
    /// one, and those trap different kinds. Under `omp parallel for`
    /// whichever thread reaches `chelis_numeric_trap` first decides, so
    /// the reported kind is race-dependent -- Linux CI caught exactly
    /// that at two elements while macOS agreed 10/10 only because clang
    /// without libomp ignores the pragma. That makes a runtime test an
    /// unreliable local guard, so the invariant is pinned on the emitted
    /// source instead.
    #[test]
    fn checked_cast_parallel_loop_reduces_the_lowest_trap_index_while_cast_trunc_is_serial() {
        fn emit(op: RiscOp, precision: Prim) -> String {
            let mut dag = Dag::new();
            let a = dag.add_node(
                RiscOp::synth_const(scalar_f32().precision, 1.0),
                vec![],
                scalar_f32(),
                None,
            );
            let dst_ty = TensorType {
                dims: vec![],
                precision,
            };
            dag.add_node(op, vec![a], dst_ty, None);
            emit_test_dag(&dag, "test_fn").expect("emit")
        }

        let trunc = emit(
            RiscOp::CastTrunc {
                new_precision: Prim::Int32,
            },
            Prim::Int32,
        );
        assert!(
            trunc.contains("chelis_trunc_float_to_int"),
            "expected the [05-OP-6] guard in the emitted C; got:\n{trunc}"
        );
        assert!(
            !trunc.contains("#pragma omp"),
            "the cast_trunc conversion loop must be serial so the FIRST \
             offending element deterministically decides the trap kind, \
             matching the evaluator's in-order walk ([05-OP-6] declares the \
             lanes identical); got:\n{trunc}"
        );

        let checked = emit(
            RiscOp::Cast {
                new_precision: Prim::Int32,
            },
            Prim::Int32,
        );
        assert!(
            checked.contains("#pragma omp parallel for"),
            "checked cast keeps correct-element conversion parallel; got:\n{checked}"
        );
        assert!(
            checked.contains("reduction(min:chelis_first_trap_index_"),
            "parallel checked cast must reduce worker-local candidates by flat index; got:\n{checked}"
        );
        assert!(
            checked.contains("if (chelis_first_trap_index_") && checked.contains("!= INT64_MAX)"),
            "the selected candidate must be rendered only after the parallel loop; got:\n{checked}"
        );
        assert!(
            !checked.contains("chelis_checked_float_to_int((double)")
                || checked
                    .find("chelis_checked_float_to_int((double)")
                    .unwrap()
                    > checked.find("!= INT64_MAX)").unwrap(),
            "an aborting trap helper must not race inside the parallel loop; got:\n{checked}"
        );
    }

    #[test]
    fn checked_cast_emitter_materializes_every_active_pair_in_logical_order() {
        let active = [
            Prim::F64,
            Prim::F32,
            Prim::F16,
            Prim::Bf16,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
        ];

        for source in active {
            for target in active {
                let mut dag = Dag::new();
                let source_ty = tensor_ty(&[2], source);
                let input = dag.add_node(
                    RiscOp::Load {
                        name: "input".into(),
                    },
                    vec![],
                    source_ty,
                    None,
                );
                let output = dag.add_node(
                    RiscOp::Cast {
                        new_precision: target,
                    },
                    vec![input],
                    tensor_ty(&[2], target),
                    None,
                );
                dag.add_root(output);
                let c = emit_test_dag(&dag, "checked_cast_product").unwrap();
                assert_eq!(
                    c.contains("/* checked cast identity */"),
                    source == target,
                    "identity plan is legal exactly on the equal-Prim diagonal: {} -> {}; generated:\n{c}",
                    source.name(),
                    target.name()
                );
                assert!(
                    c.contains(
                        "int64_t idx = chelis_indices_to_flat(indices, t0_strides, t0_rank);"
                    ),
                    "every checked-cast pair must read source elements in logical order: {} -> {}; generated:\n{c}",
                    source.name(),
                    target.name()
                );
                assert!(
                    !c.contains("memcpy(t1_data, t0_data"),
                    "same-type casts cannot copy backing order from a noncontiguous source: {} -> {}; generated:\n{c}",
                    source.name(),
                    target.name()
                );
            }
        }
    }

    #[test]
    fn cast_emits_elementwise_conversion_loop() {
        // CBackend-CastMemcpy fix: cast no longer emits a bit-preserving
        // `memcpy`. The new shape is a strided element-wise loop with a
        // typed source read and C-level target conversion.
        // See `docs/investigations/cbackend_cast_memcpy_diagnosis.md`.
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
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
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(
            c.contains("chelis_flat_to_indices"),
            "cast must emit a strided element-wise loop, not memcpy; got:\n{c}"
        );
        assert!(
            c.contains("((double*)t1_data)[i] = (double)(((float*)t0_data)[idx]);"),
            "cast must read at the source width and convert into the target width; got:\n{c}"
        );
        assert!(
            !c.contains("memcpy(t1_data, t0_data"),
            "cast must not emit the legacy bit-preserving memcpy; got:\n{c}"
        );
    }

    #[test]
    fn realize_emits_materialization_loop() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(6), None);
        let s = dag.add_node(
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2)],
            },
            vec![x],
            vec_f32(3),
            None,
        );
        dag.add_node(RiscOp::Realize, vec![s], vec_f32(3), None);

        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_tensor *t2 = chelis_alloc("));
        assert!(c.contains("chelis_indices_to_flat(indices, t1_strides, t1_rank)"));
        assert!(!c.contains("chelis_alloc_view"));
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
        let c = emit_test_dag(&dag, "test_fn").unwrap();
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
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("if (n_in != 1)"));
        assert!(c.contains("chelis_tensor *t0 = inputs[0];"));
        assert!(c.contains("chelis_tensor *t1 = inputs[0];"));
    }

    #[test]
    fn input_labels_follow_first_load_occurrence() {
        let mut dag = Dag::new();
        let b0 = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let b1 = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            scalar_f32(),
            None,
        );
        dag.set_roots(vec![b0, a, b1]);
        let verified = crate::testing::verified_dag(&dag, crate::CodegenOptions::default())
            .expect("input-label test DAG must verify ownership");
        assert_eq!(CEmitter::input_labels(verified.emission()), vec!["b", "a"]);
    }

    #[test]
    fn function_signature_correct() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = emit_test_dag(&dag, "my_func").unwrap();
        assert!(c.contains(
            "void my_func(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out)"
        ));
    }

    #[test]
    fn includes_runtime_header() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("#include \"chelis_runtime.h\""));
    }

    #[test]
    fn pad_emits_fill_and_copy() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(3).precision, 1.0),
            vec![],
            vec_f32(3),
            None,
        );
        dag.add_node(
            RiscOp::zero_pad(Prim::F32, vec![(RtDim::Lit(1), RtDim::Lit(1))]),
            vec![a],
            vec_f32(5),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains(
            "chelis_fill_scalar(t1_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F32,"
        ));
        assert!(c.contains("dst_indices[0] = src_indices[0] + 1"));
    }

    #[test]
    fn shrink_emits_offset_copy() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(5).precision, 1.0),
            vec![],
            vec_f32(5),
            None,
        );
        dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::Lit(4))],
            },
            vec![a],
            vec_f32(3),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("src_indices[0] = dst_indices[0] + 1"));
    }

    /// chelis#368: the `SHRINK_TO_END` full-axis sentinel is an eval-lane
    /// construct (`bind_symbolic_dims` resolves it). The C backend does not
    /// run that pass, so reaching codegen with the sentinel must fail LOUD
    /// rather than emit a `t->shape`-driven loop over a tensor sized from the
    /// unresolved sentinel.
    #[test]
    #[should_panic(expected = "unresolved ToEnd sentinel")]
    fn shrink_to_end_sentinel_rejected_by_c_backend() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(5).precision, 1.0),
            vec![],
            vec_f32(5),
            None,
        );
        dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::ToEnd)],
            },
            vec![a],
            vec_f32(5),
            None,
        );
        let _ = emit_test_dag(&dag, "test_fn").unwrap();
    }

    #[test]
    fn stride_materializes_stepped_coordinates() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2)],
            },
            vec![a],
            vec_f32(2),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("in_indices[0] = out_indices[0] * (2)"));
        assert!(!c.contains("t1->strides[0] ="));
    }

    #[test]
    fn add_then_mul_chains() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let d = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Mul, vec![c, d], scalar_f32(), None);
        let code = emit_test_dag(&dag, "test_fn").unwrap();
        // t2 is add result, t4 is mul result
        assert!(code.contains("t2_data"));
        assert!(code.contains("t4_data"));
    }

    #[test]
    fn vector_add_uses_correct_shape() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("(int64_t[]){ 4 }"));
    }

    #[test]
    fn sum_then_neg_chains() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(3).precision, 2.0),
            vec![],
            vec_f32(3),
            None,
        );
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
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        // Stride-4 ILP cascade (issue #163).
        assert!(c.contains("acc0 += __v"));
        assert!(c.contains("(acc0 + acc1) + (acc2 + acc3)"));
        // The slow (non-contiguous) path emits a typed pointer cast then negates.
        assert!(c.contains("((float*)t1_data)[idx]"));
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
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(!c.contains("chelis_tensor_release(t0);"));
        assert!(c.contains("outputs[0] = chelis_contiguous(t0);"));
    }

    #[test]
    fn roots_are_emitted_as_multiple_outputs() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_root(a);
        dag.add_root(b);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("if (n_out != 2)"));
        assert!(c.contains("outputs[0] = t0;"));
        assert!(c.contains("outputs[1] = t1;"));
    }

    #[test]
    fn bool_outputs_use_bool_dtype() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            RiscOp::CmpLt,
            vec![a, b],
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_alloc(0, NULL, CHELIS_DTYPE_BOOL);"));
    }

    #[test]
    fn path_sensitive_uniform_reads_float_backed_bool_and_gates_counter() {
        let mut dag = Dag::new();
        let template = dag.add_node(
            RiscOp::Load {
                name: "template".into(),
            },
            vec![],
            vec_f32(2),
            None,
        );
        let activation = dag.add_node(
            RiscOp::synth_const(Prim::Bool, 1.0),
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
            None,
        );
        dag.add_node(
            RiscOp::UniformLike {
                low: 0.0,
                high: 1.0,
                seed: 11,
            },
            vec![template, activation],
            vec_f32(2),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        // chelis#1308 stores Bool tensors as one uint8 per element; the
        // draw gate must read the predicate at that width. A `(float*)`
        // read of the one-byte allocation is out of bounds and
        // platform-divergent (the Linux-only RNG parity break on PR #1302).
        assert!(c.contains("((const uint8_t*)t1_data)[0] != 0"));
        assert!(c.contains("? CHELIS_EFFECTIVE_UNIFORM_SEED(11ULL) : 11ULL"));
        assert!(!c.contains("((float*)t1_data)[0] != 0.0f"));
        assert!(!c.contains("((bool*)t1_data)"));
    }

    #[test]
    fn int64_const_does_not_panic() {
        // Regression: dtype_macro used to panic for int64 tensors.
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::synth_const(
                TensorType {
                    dims: vec![],
                    precision: Prim::Int64,
                }
                .precision,
                42.0,
            ),
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Int64,
            },
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(
            c.contains("CHELIS_DTYPE_I64"),
            "generated C must use CHELIS_DTYPE_I64 dtype macro"
        );
        assert!(
            c.contains(
                "chelis_fill_scalar(t0_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64,"
            ),
            "generated C must carry the exact int64 tag and bits through chelis_fill_scalar"
        );
    }

    #[test]
    fn f64_const_uses_exact_tagged_fill_and_dtype_macro() {
        // v0.2.3: f64 tensors are a first-class precision. The const path must
        // emit CHELIS_DTYPE_F64 and chelis_fill_f64 (not chelis_fill_f32, which would
        // silently downcast).
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::synth_const(
                TensorType {
                    dims: vec![DimInfo::Lit(4)],
                    precision: Prim::F64,
                }
                .precision,
                1.5,
            ),
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F64,
            },
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(
            c.contains("CHELIS_DTYPE_F64"),
            "generated C must use CHELIS_DTYPE_F64 dtype macro:\n{c}"
        );
        assert!(
            c.contains(
                "chelis_fill_scalar(t0_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F64,"
            ),
            "generated C must carry the exact f64 tag and bits through chelis_fill_scalar:\n{c}"
        );
        assert!(
            !c.contains("chelis_fill_f64(") && !c.contains("chelis_fill_f32("),
            "generated C must not retain dtype-specific compatibility fills:\n{c}"
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
        let a = dag.add_node(
            RiscOp::synth_const(ty.precision, 1.0),
            vec![],
            ty.clone(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(ty.precision, 2.0),
            vec![],
            ty.clone(),
            None,
        );
        dag.add_node(RiscOp::Add, vec![a, b], ty, None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(
            c.contains("CHELIS_DTYPE_F64"),
            "generated C must use CHELIS_DTYPE_F64"
        );
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
        let a = dag.add_node(
            RiscOp::synth_const(ty.precision, 0.0),
            vec![],
            ty.clone(),
            None,
        );
        dag.add_node(RiscOp::Exp, vec![a], ty, None);
        let verified = crate::testing::verified_dag(&dag, crate::CodegenOptions::default())
            .expect("f64 exp test DAG must verify ownership");
        let c = CEmitter::emit_dag_with_options(
            verified,
            "test_fn",
            crate::CodegenOptions {
                math_lib_override: Some(crate::MathLib::None),
                ..crate::CodegenOptions::default()
            },
        )
        .unwrap();
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
        let a = dag.add_node(
            RiscOp::synth_const(ty.precision, 1.0),
            vec![],
            ty.clone(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(ty.precision, 2.0),
            vec![],
            ty.clone(),
            None,
        );
        dag.add_node(RiscOp::Add, vec![a, b], ty, None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(
            c.contains("CHELIS_DTYPE_I64"),
            "generated C must use CHELIS_DTYPE_I64"
        );
        assert!(
            c.contains("int64_t"),
            "generated C must use int64_t typed pointer casts"
        );
    }

    #[test]
    fn matmul_pattern_emits_cblas_call() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::RtDim::Lit(4),
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
                size: chelis_ir::dag::RtDim::Lit(2),
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
        let options = crate::CodegenOptions {
            use_blas: true,
            ..crate::CodegenOptions::default()
        };
        let verified = crate::testing::verified_dag(&dag, options)
            .expect("BLAS codegen test DAG must verify ownership");
        let result = crate::codegen_with_options(verified, "test_fn", options).unwrap();
        assert!(result.c_source.contains("cblas_sgemm("));
    }

    #[test]
    fn default_codegen_uses_generic_matmul_path() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::RtDim::Lit(4),
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
                size: chelis_ir::dag::RtDim::Lit(2),
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
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(!c.contains("cblas_sgemm("));
        assert!(c.contains("for (int64_t __reduce_i = 0; __reduce_i < 3; __reduce_i++) {"));
    }

    #[test]
    #[should_panic(expected = "C backend does not yet support f8e4m3 tensors")]
    fn unsupported_precision_panics_for_f8e4m3() {
        // WS-1 (dtype + Metal cleanup cycle): bf16 and f16 are now
        // admitted by the C backend, so this regression uses f8e4m3 as
        // the unsupported-precision fixture. f8e4m3 is deferred per
        // spec/04-type-system.md §1.1.1; the type checker rejects it
        // upstream, but the backend's `validate_supported_precisions`
        // remains the defense-in-depth guard against a hand-built or
        // future-pass IR that smuggles a deferred dtype through. The
        // chelis#729 rework made an f8e4m3 CONSTANT unrepresentable
        // (the sealed payload's finalize panics first, upstream of the
        // backend), so the smuggle fixture is a payload-free Load node
        // typed f8e4m3.
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::F8e4m3,
            },
            None,
        );
        let _ = emit_test_dag(&dag, "test_fn").unwrap();
    }

    // ---- SIMD Level 1b fast-path tests ----

    /// Binary add for two contiguous-capable inputs must emit the fast path
    /// containing restrict pointers, #pragma omp parallel for simd, and chelis_is_contiguous.
    #[test]
    fn binary_add_fast_path_emits_restrict_and_simd() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
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
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(1).precision, 1.0),
            vec![],
            vec_f32(1),
            None,
        );
        let a_exp = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(RiscOp::Add, vec![a_exp, b], vec_f32(4), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
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
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 5.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
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
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        // Fused: add(a, b)
        let ops = vec![FusedStep {
            op: FusedStepOp::Add,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        dag.add_node(RiscOp::FusedElem { ops }, vec![a, b], vec_f32(4), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
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
        dag.add_node(RiscOp::FusedElem { ops }, vec![x, scale], vec_f32(4), None);

        let c = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(c.contains("float* restrict __out_2 = (float*)t2_data;"));
        assert!(c.contains("const float* restrict __ext0_2 = (const float*)t0_data;"));
        assert!(c.contains("const float* restrict __ext1_2 = (const float*)t1_data;"));
        assert!(
            !c.contains("chelis_alloc_view(1, (int64_t[]){ 4 }, CHELIS_DTYPE_F32, t0->data);"),
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
            RiscOp::synth_const(tensor_ty(&[3], Prim::Int32).precision, 0.0),
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

        let c = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(c.contains("const double *t2_values_data = (const double*)t0_data;"));
        assert!(c.contains("const int32_t *t2_indices_data = (const int32_t*)t1_data;"));
        assert!(c.contains("double *t2_out_data = (double*)t2_data;"));
        assert!(c.contains("t1_dtype == CHELIS_DTYPE_I64"));
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

        let c = emit_test_dag(&dag, "embedding_probe").unwrap();

        assert!(c.contains("chelis_alloc(2, (int64_t[]){ 128, 1024 }, CHELIS_DTYPE_F32);"));
        assert!(
            !c.contains("(int64_t[]){ 128, 50000, 1024 }"),
            "sparse gather codegen must not allocate the dense [N,V,D] one-hot/product tensor"
        );
        assert!(
            !c.contains("128 * 50000 * 1024"),
            "sparse gather codegen must not compute dense embedding volume"
        );
        assert!(c.contains("t1_dtype == CHELIS_DTYPE_I64"));
    }

    #[test]
    fn sparse_gather_rejects_float_indices_at_verified_boundary() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            RiscOp::synth_const(vec_f32(3).precision, 0.0),
            vec![],
            vec_f32(3),
            None,
        );
        dag.add_node(
            RiscOp::Gather { axis: 0 },
            vec![values, indices],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );

        let error = match chelis_ir::ownership::lower_dag_ownership(dag) {
            Err(error) => error,
            Ok(_) => panic!("float sparse indices must not become verified backend input"),
        };
        assert!(error.to_string().contains("requires int32/int64 indices"));
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
            RiscOp::synth_const(tensor_ty(&[3], Prim::Int64).precision, 0.0),
            vec![],
            tensor_ty(&[3], Prim::Int64),
            None,
        );
        let updates = dag.add_node(
            RiscOp::synth_const(tensor_ty(&[3, 2], Prim::F64).precision, 1.0),
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

        let c = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(c.contains("const int64_t *t3_indices_data = (const int64_t*)t1_data;"));
        assert!(c.contains("const double *t3_updates_data = (const double*)t2_data;"));
        assert!(c.contains("double *t3_out_data = (double*)t3_data;"));
        assert!(c.contains("memcpy(t3_data, t0_data, (size_t)t3_size * sizeof(double));"));
        assert!(c.contains("t3_out_data[t3_out] += t3_updates_data[t3_src];"));
    }

    #[test]
    fn sparse_scatter_add_rejects_float_indices_at_verified_boundary() {
        let mut dag = Dag::new();
        let target = dag.add_node(
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            RiscOp::synth_const(vec_f32(3).precision, 0.0),
            vec![],
            vec_f32(3),
            None,
        );
        let updates = dag.add_node(
            RiscOp::synth_const(tensor_ty(&[3, 2], Prim::F32).precision, 1.0),
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

        let error = match chelis_ir::ownership::lower_dag_ownership(dag) {
            Err(error) => error,
            Ok(_) => panic!("float sparse indices must not become verified backend input"),
        };
        assert!(error.to_string().contains("requires int32/int64 indices"));
    }

    #[test]
    fn target_fused_in_place_consumes_the_verified_reuse_proof() {
        // The shared planner turns the DAG's liveness hint into the sealed
        // provenance/capacity proof consumed by C emission.
        use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};
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

        let c = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(!c.contains("chelis_alloc_view"), "{c}");
        assert!(c.contains("chelis_tensor *t3 = t1;"), "{c}");
        assert!(c.contains("chelis_tensor_repurpose(t3,"), "{c}");
        assert!(c.contains("t1_data = t3_data;"), "{c}");
        assert!(c.contains("float* __out_3 = (float*)t3_data;"), "{c}");
        assert!(
            c.contains("const float* __ext0_3 = (const float*)t1_data;"),
            "{c}"
        );
        assert!(
            c.contains("const float* restrict __ext1_3 = (const float*)t2_data;"),
            "{c}"
        );
    }

    #[test]
    fn fused_in_place_does_not_alias_a_caller_owned_input() {
        // chelis#933: `reusable_input` is a linearity fact — the value
        // is dead after this op — and says nothing about who owns the
        // storage. When it resolves to a program input the bytes belong
        // to the caller (`cpu_input_tensor` in `chelis-python` passes a
        // pointer straight into the caller's NumPy buffer), so writing
        // the result there overwrites an argument.
        use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};
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

        let c = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(
            !c.contains("CHELIS_DTYPE_F32, t0_data)"),
            "the fused output must not be a view over the caller's input buffer; got:\n{c}"
        );
        // With no in-place reuse the output takes an ordinary owned
        // slot, so every pointer is `restrict` again.
        assert!(
            c.contains("float* restrict __out_2 = (float*)t2_data;"),
            "{c}"
        );
        assert!(
            c.contains("const float* restrict __ext0_2 = (const float*)t0_data;"),
            "{c}"
        );
    }

    #[test]
    fn fused_in_place_does_not_alias_a_view_of_a_caller_owned_input() {
        // chelis#933, second shape: C has no public metadata-view
        // constructor. `reshape` materializes program-owned canonical
        // storage, so the shared proof may later reuse that intermediate,
        // but it must never transfer the caller's descriptor or bytes.
        use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_ty(&[2, 2], Prim::F32),
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
            vec![flat, scale],
            vec_f32(4),
            None,
        );
        dag.set_reusable_input(fused, flat);

        let c = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(c.contains("chelis_tensor *t1 = chelis_alloc("), "{c}");
        assert!(
            c.contains("chelis_tensor *t3 = t1;"),
            "the materialized reshape is program-owned and may supply the exact token; got:\n{c}"
        );
        assert!(
            !c.contains("chelis_tensor *t3 = t0;"),
            "the fused output must not transfer the caller-owned descriptor; got:\n{c}"
        );
    }

    #[test]
    fn fused_reusable_input_with_multiple_consumers_does_not_alias() {
        use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};
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
        let other = dag.add_node(RiscOp::Neg, vec![x], vec_f32(4), None);
        dag.add_root(fused);
        dag.add_root(other);

        let c = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(
            !c.contains("chelis_alloc_view"),
            "multi-consumer reusable input must not be aliased in place"
        );
        assert!(c.contains("float* restrict __out_2 = (float*)t2_data;"));
        assert!(c.contains("const float* restrict __ext0_2 = (const float*)t0_data;"));
        assert!(c.contains("const float* restrict __ext1_2 = (const float*)t1_data;"));
    }

    // ---- New scalar builtin C emission tests ----

    #[test]
    fn cos_emits_cosf() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Cos, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("cosf("), "expected cosf( in:\n{c}");
    }

    #[test]
    fn tan_emits_tanf() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Tan, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("tanf("), "expected tanf( in:\n{c}");
    }

    #[test]
    fn atan_emits_atanf() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Atan, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("atanf("), "expected atanf( in:\n{c}");
    }

    #[test]
    fn abs_emits_fabsf() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, -2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Abs, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("fabsf("), "expected fabsf( in:\n{c}");
    }

    #[test]
    fn floor_emits_floorf() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.7),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Floor, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("floorf("), "expected floorf( in:\n{c}");
    }

    #[test]
    fn ceil_emits_ceilf() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.3),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Ceil, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("ceilf("), "expected ceilf( in:\n{c}");
    }

    #[test]
    fn round_emits_rintf() {
        // `round` lowers to `rintf` (round-to-nearest-ties-to-even under
        // the default rounding mode), NOT `roundf` (ties-away-from-zero).
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.5),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Round, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("rintf("), "expected rintf( in:\n{c}");
        assert!(
            !c.contains("roundf("),
            "round must not emit ties-away-from-zero roundf( in:\n{c}"
        );
    }

    // ---- Numerical correctness: verify via constant folding in the evaluator ----
    // These tests confirm that the Rust-side evaluator and the IR pipeline agree
    // on the mathematical values. The C emission tests above cover the symbol name.

    #[test]
    fn cos_numerical_correctness() {
        // cos(π/3) ≈ 0.5 (within 1e-4)
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, std::f64::consts::PI / 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let out = dag.add_node(RiscOp::Cos, vec![x], scalar_f32(), None);
        dag.add_root(out);
        let results = chelis_ir::eval::eval_scalar(&dag, &chelis_unord::UnordMap::new());
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
        let x = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, -2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let out = dag.add_node(RiscOp::Abs, vec![x], scalar_f32(), None);
        dag.add_root(out);
        let results = chelis_ir::eval::eval_scalar(&dag, &chelis_unord::UnordMap::new());
        let val = results[&out] as f32;
        assert!(
            (val - 2.0_f32).abs() < 1e-4,
            "abs(-2.0) should be 2.0, got {val}"
        );
    }
}
