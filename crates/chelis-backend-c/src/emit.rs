//! RISC DAG to C source code emission.

use std::collections::{BTreeMap, BTreeSet};

use chelis_ir::dag::{
    Dag, DagNode, DimInfo, ExtremaKind, ExtremaOperand, FusedInput, FusedStep, FusedStepOp, NodeId,
    ReduceWindowKind, RiscOp, RtAxis, RtDim, SymbolicDimSource, TensorType,
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
    /// Local claims in derivation/declaration order at each operation. Grouping
    /// by axis would reorder simultaneous failures when axes are permuted.
    local_dim_guard_sites:
        chelis_unord::UnordMap<usize, Vec<(usize, chelis_ir::ownership::LocalGuardClaim)>>,
    /// Claim names this function actually declares as C variables. Resolved
    /// claims compare against their numeric canonical value instead.
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

#[derive(Clone, Copy)]
enum SparseEmission {
    Gather,
    Add,
    Replace,
    Elements,
}

struct MatmulEmitSpec {
    a: NodeId,
    b: NodeId,
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
        // Two claim KINDS reach this list: the equality classes and, since
        // chelis#1277 S2b, the unit-extent claims (C2.9). They are keyed the
        // same way, on the axis whose extent the guard reads, so two sites can
        // land on one key.
        //
        // Every distinct claim on a key is emitted, and equal ones coalesce:
        //
        // - EQUAL sites coalesce. One locally computed unit axis feeding two
        //   `expand` nodes produces the same claim, canonical and operation
        //   twice, and one emitted guard satisfies both, so a second is
        //   redundant rather than lost.
        // - DISAGREEING sites are TWO OBLIGATIONS, and both are emitted. This
        //   replaces an `Unsupported` refusal. The refusal was argued from a
        //   pair that could not arise; it can. A `reshape` with a computed
        //   target, claimed by a signature and then broadcast by a same-rank
        //   `expand`, puts the class's `n` and the unit claim's `1` on the
        //   reshape's own axis, and refusing there refuses a program that
        //   checks clean. One comparison cannot discharge two claims, so the
        //   answer is two comparisons, not a rejection and not a silent
        //   replacement.
        //
        // `LocalGuardClaim` derives `PartialEq` over its fields, so "equal"
        // means agreeing in claim, canonical, operation AND read instruction,
        // which is exactly the set one comparison discharges.
        let mut local_dim_guard_sites: chelis_unord::UnordMap<
            usize,
            Vec<(usize, chelis_ir::ownership::LocalGuardClaim)>,
        > = chelis_unord::UnordMap::new();
        for ((node, axis), claim) in dag.local_dim_guard_sites() {
            let claims = local_dim_guard_sites.entry(node).or_default();
            let entry = (axis, claim);
            if !claims.contains(&entry) {
                claims.push(entry);
            }
        }

        let mut e = CEmitter {
            lines: Vec::new(),
            indent: 0,
            use_blas: dag
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. })),
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
            e.line(&Self::blas_integer_support());
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
        self.line(&format!(
            "for (int32_t __axis = 0; __axis < t{id}_rank; ++__axis) t{id}_strides[__axis] = chelis_tensor_stride(t{id}, __axis);"
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
            "int64_t t{id}_byte_capacity = chelis_tensor_byte_count(t{id});"
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
                if matches!(node.op, RiscOp::Expand { .. })
                    && !new_ty
                        .dims
                        .iter()
                        .any(|dim| matches!(dim, DimInfo::Named(name, _) if !is_anon(name)))
                {
                    // [05-MOV-1], #1619: the replaced/inserted axis reads
                    // the size carrier; kept axes read their own operand
                    // positions. Rank equality does not prove pass-through.
                    // Numeric result claims remain independent of their
                    // sources. Explicit named outputs stay on the existing
                    // path: preserving one without its unread signature
                    // witness can newly execute an unchecked wrong shape.
                    // B2b-1 owns that scoped claim-transport repair.
                    let sources = chelis_ir::output_axis_sources(&out, id);
                    for (axis, dim) in new_ty.dims.iter_mut().enumerate() {
                        if !matches!(dim, DimInfo::Named(name, _) if is_anon(name)) {
                            continue;
                        }
                        if let DimInfo::Named(_, Some(required)) = dim {
                            // Anonymous spelling supplies no binder, but a
                            // required number is still a literal claim.
                            *dim = DimInfo::Lit(*required);
                            continue;
                        }
                        use chelis_ir::AxisSource;
                        let observed = match sources.get(axis) {
                            Some(AxisSource::Literal { value }) => {
                                usize::try_from(*value).ok().map(DimInfo::Lit)
                            }
                            Some(AxisSource::InputAxis {
                                input,
                                axis: RtAxis::Lit(source_axis),
                            }) => node
                                .inputs
                                .get(*input)
                                .and_then(|input| out.get(*input))
                                .and_then(|input| {
                                    input
                                        .output_type
                                        .dims
                                        .get(usize::try_from(*source_axis).ok()?)
                                })
                                .cloned(),
                            // A scalar size is read at execution. Give
                            // that output its own symbol.
                            Some(
                                AxisSource::ScalarInput { .. }
                                | AxisSource::OpComputed { .. }
                                | AxisSource::ClassSupplied { .. }
                                | AxisSource::ExternalAxis { .. },
                            )
                            | None => None,
                        };
                        *dim = observed.unwrap_or_else(|| rewrite_dim(id, axis, dim));
                    }
                } else if let RiscOp::Gather { axis } = &node.op
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
            RiscOp::ConstTensor { data } => self.emit_const_tensor(id, data, &node.output_type)?,
            RiscOp::Shape { axis } => self.emit_shape(id, *axis, &node.inputs, &node.output_type),
            RiscOp::ExtentWitness {
                parameter,
                axis: RtAxis::Lit(axis),
                requirements,
            } => {
                let input = node.inputs[0].0;
                let parameter =
                    chelis_ir::span_sanitize::sanitize_for_format_string(parameter).to_string();
                for required in requirements {
                    let required = required.as_i64_exact().expect("verified int64 requirement");
                    self.line(&format!("if (t{input}_shape[{axis}] != {required}) {{"));
                    self.indent += 1;
                    self.line(&format!("fprintf(stderr, \"extent `{required}`: claimed = {required}, {parameter} axis {axis} = %lld\\n\", (long long)t{input}_shape[{axis}]);"));
                    self.line("chelis_numeric_trap(\"numeric trap: domain in load at int64\");");
                    self.indent -= 1;
                    self.line("}");
                }
                self.emit_shape(id, *axis as usize, &node.inputs, &node.output_type);
            }
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
                )?;
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
                batch_dims: _,
                m: _,
                n: _,
                k: _,
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
        let entry_guards = dag.entry_extent_guards();
        let mut literal_claim_pairs = chelis_unord::UnordMap::<(usize, i32), usize>::new();
        for guard in &entry_guards {
            if let chelis_ir::axis_sources::EntryExtentGuard::Literal {
                required,
                observed: (load, axis),
            } = guard
                && let RiscOp::Load { name } = &dag.get(*load).expect("entry input").op
            {
                literal_claim_pairs.insert((input_slots[name.as_str()], *axis as i32), *required);
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
            self.declared_dim_names.insert(binding.name.clone());
        }

        // Shared IR owns ordering and witness identity. Rendering never
        // re-groups checks by class, name or guard kind.
        for guard in entry_guards {
            use chelis_ir::axis_sources::EntryExtentGuard;
            let input_read = |(load, axis): (NodeId, usize)| {
                let RiscOp::Load { name } = &dag.get(load).expect("entry input").op else {
                    unreachable!("entry witness must be an input");
                };
                let slot = input_slots[name.as_str()];
                let label = chelis_ir::span_sanitize::sanitize_for_format_string(name.as_str());
                (
                    format!("chelis_tensor_shape(inputs[{slot}], {axis})"),
                    label.to_string(),
                    axis,
                )
            };
            let (left, right, diagnostic) = match guard {
                EntryExtentGuard::Named {
                    claim,
                    canonical,
                    observed,
                } => {
                    let (left, canonical_label, canonical_axis) = input_read(canonical);
                    let (right, label, axis) = input_read(observed);
                    let claim = chelis_ir::span_sanitize::sanitize_for_format_string(&claim);
                    let diagnostic = format!(
                        "fprintf(stderr, \"extent `{claim}`: {canonical_label} axis {canonical_axis} = %lld, {label} axis {axis} = %lld\\n\", (long long)({left}), (long long)({right}));"
                    );
                    (left, right, diagnostic)
                }
                EntryExtentGuard::Literal { required, observed } => {
                    let (right, label, axis) = input_read(observed);
                    let diagnostic = format!(
                        "fprintf(stderr, \"extent `{required}`: claimed = {required}, {label} axis {axis} = %lld\\n\", (long long)({right}));"
                    );
                    (required.to_string(), right, diagnostic)
                }
            };
            self.line(&format!("if ({right} != {left}) {{"));
            self.indent += 1;
            self.line(&diagnostic);
            self.line("chelis_numeric_trap(\"numeric trap: domain in load at int64\");");
            self.indent -= 1;
            self.line("}");
        }
    }

    fn shape_literal(ty: &TensorType) -> String {
        let dims: Vec<String> = ty.dims.iter().map(|dim| Self::emit_dim_info(dim)).collect();
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

    /// Save input projections before a storage slot can be repurposed. The
    /// returned condition belongs after allocation, where the output's checked
    /// count is available, and guards every direct-index optimized path.
    fn emit_elementwise_index_steps(
        &mut self,
        id: usize,
        inputs: &[NodeId],
        ty: &TensorType,
    ) -> String {
        let shape = Self::tagged_shape_literal(ty);
        let rank = Self::ndim(ty);
        let inputs = inputs.iter().map(|input| input.0).collect::<BTreeSet<_>>();
        let mut identity = Vec::new();
        for input in inputs {
            self.line(&format!(
                "const int64_t t{id}_input{input}_step = chelis_tensor_elementwise_index_step_for_shape(t{input}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {shape});"
            ));
            identity.push(format!("t{id}_input{input}_step == 1"));
        }
        if identity.is_empty() {
            "1".into()
        } else {
            format!("t{id}_size <= 1 || ({})", identity.join(" && "))
        }
    }

    fn tagged_shape_literal(ty: &TensorType) -> String {
        let shape = ty
            .dims
            .iter()
            .map(|dim| {
                let extent = Self::emit_dim_info(dim);
                // The constructor's uint64 bits parameter preserves an int64
                // extent's bits by C's defined modulo conversion, including
                // negative values that the metadata owner then rejects.
                format!("chelis_scalar_from_bits(CHELIS_DTYPE_I64, {extent})")
            })
            .collect::<Vec<_>>();
        if shape.is_empty() {
            "NULL".to_string()
        } else {
            format!("(chelis_scalar[]){{ {} }}", shape.join(", "))
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
            DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => n.to_string(),
            DimInfo::Named(name, None) => name.clone(),
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
        let shape = Self::tagged_shape_literal(ty);
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

    /// A literal's tagged count and every finalized scalar agree with its
    /// checked destination before storage submission. No raw payload copy or
    /// backend width fallback participates in this path.
    fn emit_const_tensor(
        &mut self,
        id: usize,
        storage: &chelis_types::TensorStorage,
        ty: &TensorType,
    ) -> Result<(), Unsupported> {
        let invalid = |detail: &str| {
            Unsupported::new(
                UnsupportedKind::Construct("an inconsistent finalized tensor literal".into()),
                format!("the C DAG emitter (node {id}): {detail}"),
                Stage::Codegen("c"),
                chelis_types::deliberate_rejection!(
                    "[05-OP-33]",
                    "literal count and dtype must agree with the checked result metadata"
                ),
            )
        };
        if storage.prim() != ty.precision {
            return Err(invalid("storage dtype differs from result dtype"));
        }
        let count =
            i64::try_from(storage.len()).map_err(|_| invalid("literal count exceeds int64"))?;
        let dtype = Self::dtype_macro(ty);
        let rank = Self::ndim(ty);
        let shape = Self::tagged_shape_literal(ty);
        self.line(&format!("chelis_tensor_check_literal(chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {shape}, chelis_scalar_from_bits({dtype}, UINT64_C(0)), chelis_scalar_from_bits(CHELIS_DTYPE_I64, {count}));"));
        self.emit_slot_wrapper(id, ty);
        let values = if count == 0 {
            "NULL".to_owned()
        } else {
            let images = (0..storage.len())
                .map(|index| {
                    let bits = match storage.scalar_at(index).element_ref() {
                        ElementRef::I8(n) => u64::from(n as u8),
                        ElementRef::I16(n) => u64::from(n as u16),
                        ElementRef::I32(n) => u64::from(n as u32),
                        ElementRef::I64(n) => n as u64,
                        ElementRef::F64(n) => n.to_bits(),
                        ElementRef::F32(n) => u64::from(n.to_bits()),
                        ElementRef::F16(n) => u64::from(n.to_bits()),
                        ElementRef::Bf16(n) => u64::from(n.to_bits()),
                        ElementRef::Bool(n) => u64::from(n),
                    };
                    format!(
                        "{{ .dtype = {dtype}, .reserved = {{0}}, .bits = UINT64_C(0x{bits:016x}) }}"
                    )
                })
                .collect::<Vec<_>>();
            self.line(&format!(
                "static const chelis_scalar t{id}_literal[] = {{ {} }};",
                images.join(", ")
            ));
            format!("t{id}_literal")
        };
        self.line(&format!("chelis_tensor_write_literal(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {count}), {values});"));
        Ok(())
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
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
        self.line(&format!("int64_t idx_a = i * t{id}_input{a}_step;"));
        self.line(&format!("int64_t idx_b = i * t{id}_input{b}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
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
        self.line(&format!("int64_t idx_a = i * t{id}_input{a}_step;"));
        self.line(&format!("int64_t idx_b = i * t{id}_input{b}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
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
        self.line(&format!("int64_t idx_a = i * t{id}_input{a}_step;"));
        self.line(&format!("int64_t idx_b = i * t{id}_input{b}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
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
        self.line(&format!("int64_t idx_a = i * t{id}_input{a}_step;"));
        self.line(&format!("int64_t idx_b = i * t{id}_input{b}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
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
        self.line(&format!("int64_t idx_a = i * t{id}_input{a}_step;"));
        self.line(&format!("int64_t idx_b = i * t{id}_input{b}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
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
            "if (chelis_is_contiguous(t{a}) && ({identity}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
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
        self.line(&format!("int64_t idx_a = i * t{id}_input{a}_step;"));
        self.line(&format!("int64_t idx_b = i * t{id}_input{b}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity})) {{"
        ));
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
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}_data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}_data;"
        ));
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity})) {{"
        ));
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
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("{et}* restrict __out_{id} = ({et}*)t{id}_data;"));
        self.line(&format!(
            "const {et}* restrict __in_a_{id} = (const {et}*)t{a}_data;"
        ));
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity})) {{"
        ));
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
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity})) {{"
        ));
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
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity})) {{"
        ));
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
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "uint16_t* restrict __out_{id} = (uint16_t*)t{id}_data;"
        ));
        self.line(&format!(
            "const uint16_t* restrict __in_a_{id} = (const uint16_t*)t{a}_data;"
        ));
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity})) {{"
        ));
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
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
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
        self.line(&format!("int64_t idx_a = i * t{id}_input{a}_step;"));
        self.line(&format!("int64_t idx_b = i * t{id}_input{b}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);

        if Self::is_reduced_float(ty) {
            let load = Self::reduced_to_f32_fn(ty.precision);
            self.line(&format!(
                "if (chelis_is_contiguous(t{a}) && ({identity}) && chelis_is_contiguous(t{b}) && chelis_is_contiguous(t{g}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size && t{g}_size == t{id}_size) {{"
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
            for (name, source) in [("a", a), ("b", b), ("g", g)] {
                self.line(&format!(
                    "int64_t idx_{name} = i * t{id}_input{source}_step;"
                ));
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
            "if (chelis_is_contiguous(t{a}) && ({identity}) && chelis_is_contiguous(t{b}) && chelis_is_contiguous(t{g}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size && t{g}_size == t{id}_size) {{"
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
        for (name, source) in [("a", a), ("b", b), ("g", g)] {
            self.line(&format!(
                "int64_t idx_{name} = i * t{id}_input{source}_step;"
            ));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity})) {{"
        ));
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
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
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
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
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
        self.line(&format!("if (({contiguity_cond}) && ({identity})) {{"));
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

        // Scalar-input path: use the saved checked projection for each input.
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;

        // Project each external input into the checked iteration domain.
        for (ext_idx, ext_node) in inputs.iter().enumerate() {
            let ext_id = ext_node.0;
            self.line(&format!(
                "int64_t idx_ext{ext_idx} = i * t{id}_input{ext_id}_step;"
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
    /// Bind the integer width to both actual vendor prototypes. A header whose
    /// published type and prototypes disagree fails compilation before any call.
    pub(crate) fn blas_integer_support() -> String {
        let f32_type = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::F32,
        });
        let f64_type = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::F64,
        });
        format!(
            r#"
#ifndef CHELIS_C_BLAS_CONTRACT
#define CHELIS_C_BLAS_CONTRACT
#if defined(__APPLE__)
typedef __LAPACK_int chelis_blas_integer;
typedef enum CBLAS_ORDER chelis_blas_layout;
#elif defined(OPENBLAS_VERSION)
typedef blasint chelis_blas_integer;
typedef enum CBLAS_ORDER chelis_blas_layout;
#elif defined(CBLAS_INT)
typedef CBLAS_INT chelis_blas_integer;
typedef CBLAS_LAYOUT chelis_blas_layout;
#else
#error "CBLAS header must declare its supported signed dimension type"
#endif
_Static_assert((chelis_blas_integer)-1 < 0, "CBLAS dimensions must be signed");
_Static_assert(sizeof(chelis_blas_integer) == 4 || sizeof(chelis_blas_integer) == 8, "unsupported CBLAS dimension width");
typedef void (*chelis_sgemm_signature)(chelis_blas_layout, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE, chelis_blas_integer, chelis_blas_integer, chelis_blas_integer, {f32_type}, const {f32_type}*, chelis_blas_integer, const {f32_type}*, chelis_blas_integer, {f32_type}, {f32_type}*, chelis_blas_integer);
typedef void (*chelis_dgemm_signature)(chelis_blas_layout, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE, chelis_blas_integer, chelis_blas_integer, chelis_blas_integer, {f64_type}, const {f64_type}*, chelis_blas_integer, const {f64_type}*, chelis_blas_integer, {f64_type}, {f64_type}*, chelis_blas_integer);
_Static_assert(_Generic(&cblas_sgemm, chelis_sgemm_signature: 1, default: 0), "CBLAS sgemm dimension contract mismatch");
_Static_assert(_Generic(&cblas_dgemm, chelis_dgemm_signature: 1, default: 0), "CBLAS dgemm dimension contract mismatch");
#define CHELIS_BLAS_MAXIMUM (sizeof(chelis_blas_integer) == 8 ? INT64_MAX : INT32_MAX)
#endif
"#
        )
    }

    fn emit_matmul_plan(&mut self, id: usize, spec: &MatmulEmitSpec, ty: &TensorType) {
        let a = spec.a.0;
        let b = spec.b.0;
        let dtype = Self::dtype_macro(ty);
        let index = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::Int64,
        });
        self.line(&format!("chelis_matmul_plan *t{id}_matmul = chelis_tensor_matmul_plan(t{a}, t{b}, chelis_scalar_from_bits({dtype}, UINT64_C(0)));"));
        let extents = (0..ty.dims.len()).map(|axis| (axis, format!("chelis_matmul_extent(t{id}_matmul, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {axis}))"))).collect::<Vec<_>>();
        self.emit_runtime_dim_sites(id, &extents);
        let rank = ty.dims.len();
        let shape = Self::tagged_shape_literal(ty);
        self.line(&format!("chelis_matmul_check_target(t{id}_matmul, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {shape});"));
        self.line(&format!("chelis_matmul_check_vendor(t{id}_matmul, chelis_scalar_from_bits(CHELIS_DTYPE_I64, CHELIS_BLAS_MAXIMUM));"));
        for (name, dimension) in [("m", "ROWS"), ("n", "COLUMNS"), ("k", "REDUCTION")] {
            self.line(&format!("{index} t{id}_{name} = chelis_matmul_dimension(t{id}_matmul, CHELIS_MATMUL_{dimension});"));
        }
        self.line(&format!(
            "{index} t{id}_batch_count = chelis_matmul_batch_count(t{id}_matmul);"
        ));
        if matches!(spec.operand_precision, Prim::F16 | Prim::Bf16) {
            for (name, part) in [("af", "LEFT"), ("bf", "RIGHT"), ("cf", "RESULT")] {
                if name == "cf" && !Self::is_reduced_float(ty) {
                    continue;
                }
                self.line(&format!("{index} t{id}_{name}_count = chelis_matmul_matrix_count(t{id}_matmul, CHELIS_MATMUL_{part});"));
                self.line(&format!("chelis_matmul_check_scratch(t{id}_matmul, CHELIS_MATMUL_{part}, chelis_scalar_from_bits(CHELIS_DTYPE_F32, UINT64_C(0)));"));
            }
        }
    }

    fn emit_blas_matmul(&mut self, id: usize, spec: &MatmulEmitSpec, ty: &TensorType) {
        if matches!(spec.operand_precision, Prim::Bf16 | Prim::F16) {
            self.emit_blas_matmul_reduced_f(id, spec, ty);
            return;
        }
        let a = spec.a.0;
        let b = spec.b.0;
        let (gemm, alpha, beta) = match spec.accumulator {
            Prim::F32 => ("cblas_sgemm", "1.0f", "0.0f"),
            Prim::F64 => ("cblas_dgemm", "1.0", "0.0"),
            other => panic!("unsupported verified BLAS accumulator {}", other.name()),
        };
        let element = Self::elem_type(ty);
        let index = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::Int64,
        });
        self.emit_matmul_plan(id, spec, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("if (t{id}_k == 0) {{"));
        self.indent += 1;
        self.line(&format!(
            "if (t{id}_byte_capacity != 0) memset(t{id}_data, 0, (size_t)t{id}_byte_capacity);"
        ));
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line(&format!(
            "for ({index} t{id}_batch = 0; t{id}_batch < t{id}_batch_count; ++t{id}_batch) {{"
        ));
        self.indent += 1;
        for (name, part) in [("a", "LEFT"), ("b", "RIGHT"), ("out", "RESULT")] {
            self.line(&format!("{index} t{id}_{name}_offset = chelis_matmul_index(t{id}_matmul, CHELIS_MATMUL_{part}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_batch), chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0));"));
        }
        self.line(&format!("{gemm}(CblasRowMajor, CblasNoTrans, CblasNoTrans, (chelis_blas_integer)t{id}_m, (chelis_blas_integer)t{id}_n, (chelis_blas_integer)t{id}_k, {alpha}, (const {element}*)t{a}_data + t{id}_a_offset, (chelis_blas_integer)t{id}_k, (const {element}*)t{b}_data + t{id}_b_offset, (chelis_blas_integer)t{id}_n, {beta}, ({element}*)t{id}_data + t{id}_out_offset, (chelis_blas_integer)t{id}_n);"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_matmul_plan_release(t{id}_matmul);"));
    }

    /// Keep the IR-pinned f32 accumulator and destination conversion. Scratch
    /// capacity belongs to checked metadata; storage belongs to runtime tensors.
    fn emit_blas_matmul_reduced_f(&mut self, id: usize, spec: &MatmulEmitSpec, ty: &TensorType) {
        assert_eq!(spec.accumulator, Prim::F32, "verified BLAS accumulator");
        let a = spec.a.0;
        let b = spec.b.0;
        let to_f32 = Self::reduced_to_f32_fn(spec.operand_precision);
        let output_reduced = Self::is_reduced_float(ty);
        let element = Self::elem_type(ty);
        let operand = Self::elem_type(&TensorType {
            dims: vec![],
            precision: spec.operand_precision,
        });
        let f32_type = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::F32,
        });
        let index = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::Int64,
        });
        self.emit_matmul_plan(id, spec, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("if (t{id}_k == 0) {{"));
        self.indent += 1;
        self.line(&format!(
            "if (t{id}_byte_capacity != 0) memset(t{id}_data, 0, (size_t)t{id}_byte_capacity);"
        ));
        self.indent -= 1;
        self.line(&format!("}} else if (t{id}_batch_count != 0) {{"));
        self.indent += 1;
        for name in ["af", "bf", "cf"] {
            if name == "cf" && !output_reduced {
                continue;
            }
            self.line(&format!("chelis_tensor *t{id}_{name}_scratch = chelis_alloc(1, &t{id}_{name}_count, CHELIS_DTYPE_F32);"));
            self.line(&format!("chelis_tensor_write *t{id}_{name}_guard = chelis_tensor_begin_write(t{id}_{name}_scratch);"));
            self.line(&format!("{f32_type} *t{id}_{name} = ({f32_type}*)chelis_tensor_write_view(t{id}_{name}_guard).data;"));
        }
        self.line(&format!(
            "for ({index} t{id}_batch = 0; t{id}_batch < t{id}_batch_count; ++t{id}_batch) {{"
        ));
        self.indent += 1;
        for (name, source, part) in [("af", a, "LEFT"), ("bf", b, "RIGHT")] {
            self.line(&format!(
                "for ({index} t{id}_i = 0; t{id}_i < t{id}_{name}_count; ++t{id}_i) {{"
            ));
            self.indent += 1;
            self.line(&format!("{index} t{id}_source = chelis_matmul_index(t{id}_matmul, CHELIS_MATMUL_{part}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_batch), chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_i));"));
            self.line(&format!("t{id}_{name}[t{id}_i] = {to_f32}(((const {operand}*)t{source}_data)[t{id}_source]);"));
            self.indent -= 1;
            self.line("}");
        }
        let output = if output_reduced {
            format!("t{id}_cf")
        } else {
            self.line(&format!("{index} t{id}_out_offset = chelis_matmul_index(t{id}_matmul, CHELIS_MATMUL_RESULT, chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_batch), chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0));"));
            format!("({f32_type}*)t{id}_data + t{id}_out_offset")
        };
        self.line(&format!("cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (chelis_blas_integer)t{id}_m, (chelis_blas_integer)t{id}_n, (chelis_blas_integer)t{id}_k, 1.0f, t{id}_af, (chelis_blas_integer)t{id}_k, t{id}_bf, (chelis_blas_integer)t{id}_n, 0.0f, {output}, (chelis_blas_integer)t{id}_n);"));
        if output_reduced {
            let from_f32 = Self::f32_to_reduced_fn(ty.precision);
            self.line(&format!(
                "for ({index} t{id}_i = 0; t{id}_i < t{id}_cf_count; ++t{id}_i) {{"
            ));
            self.indent += 1;
            self.line(&format!("{index} t{id}_destination = chelis_matmul_index(t{id}_matmul, CHELIS_MATMUL_RESULT, chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_batch), chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_i));"));
            self.line(&format!(
                "(({element}*)t{id}_data)[t{id}_destination] = {from_f32}(t{id}_cf[t{id}_i]);"
            ));
            self.indent -= 1;
            self.line("}");
        }
        self.indent -= 1;
        self.line("}");
        for name in ["af", "bf", "cf"] {
            if name == "cf" && !output_reduced {
                continue;
            }
            self.line(&format!("chelis_tensor_end_write(t{id}_{name}_guard);"));
            self.line(&format!("chelis_tensor_release(t{id}_{name}_scratch);"));
        }
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_matmul_plan_release(t{id}_matmul);"));
    }

    fn emit_sparse_gather(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        self.emit_sparse_checked(id, axis, inputs, ty, dag, SparseEmission::Gather);
    }

    fn emit_sparse_scatter_add(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        self.emit_sparse_checked(id, axis, inputs, ty, dag, SparseEmission::Add);
    }

    fn emit_sparse_scatter_replace(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        self.emit_sparse_checked(id, axis, inputs, ty, dag, SparseEmission::Replace);
    }

    fn emit_sparse_scatter_elements(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        self.emit_sparse_checked(id, axis, inputs, ty, dag, SparseEmission::Elements);
    }

    /// Validate the whole iteration shape before storage submission; each loop
    /// position then obtains both its index slot and base offset from that plan.
    fn emit_sparse_checked(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
        operation: SparseEmission,
    ) {
        let base = inputs[0].0;
        let indices = inputs[1].0;
        let (operation, update) = match operation {
            SparseEmission::Gather => ("CHELIS_SPARSE_GATHER", None),
            SparseEmission::Add => ("CHELIS_SPARSE_ADD", Some("+=")),
            SparseEmission::Replace => ("CHELIS_SPARSE_REPLACE", Some("=")),
            SparseEmission::Elements => ("CHELIS_SPARSE_ELEMENTS", Some("=")),
        };
        let updates = update.map(|_| inputs[2].0);
        let updates_arg = updates
            .map(|n| format!("t{n}"))
            .unwrap_or_else(|| "NULL".into());
        let index_type = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::Int64,
        });
        let index_element = Self::elem_type(&dag.get(inputs[1]).unwrap().output_type);
        let element = Self::elem_type(ty);
        self.line(&format!("chelis_sparse_plan *t{id}_sparse = chelis_tensor_sparse_plan(t{base}, t{indices}, {updates_arg}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {axis}), {operation});"));
        let extents = (0..ty.dims.len()).map(|axis| (axis, format!("chelis_sparse_extent(t{id}_sparse, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {axis}))"))).collect::<Vec<_>>();
        self.emit_runtime_dim_sites(id, &extents);
        let rank = ty.dims.len();
        let shape = Self::tagged_shape_literal(ty);
        self.line(&format!("chelis_sparse_check_target(t{id}_sparse, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {shape});"));
        self.line(&format!(
            "{index_type} t{id}_sparse_count = chelis_sparse_count(t{id}_sparse);"
        ));
        self.emit_slot_wrapper(id, ty);
        if updates.is_some() {
            self.line(&format!("if (t{id}_byte_capacity != 0 && t{id}_data != t{base}_data) memcpy(t{id}_data, t{base}_data, (size_t)t{id}_byte_capacity);"));
        }
        // Ascending iteration positions are the exact updates row-major order,
        // including duplicate destinations. Each scatter therefore stays serial.
        self.line(&format!(
            "for ({index_type} t{id}_i = 0; t{id}_i < t{id}_sparse_count; ++t{id}_i) {{"
        ));
        self.indent += 1;
        self.line(&format!("{index_type} t{id}_index_slot = chelis_sparse_index_slot(t{id}_sparse, chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_i));"));
        self.line(&format!("{index_type} t{id}_selected = ((const {index_element}*)t{indices}_data)[t{id}_index_slot];"));
        self.line(&format!("{index_type} t{id}_base_index = chelis_sparse_data_index(t{id}_sparse, chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_i), chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_selected));"));
        if let (Some(updates), Some(update)) = (updates, update) {
            self.line(&format!("(({element}*)t{id}_data)[t{id}_base_index] {update} ((const {element}*)t{updates}_data)[t{id}_i];"));
        } else {
            self.line(&format!("(({element}*)t{id}_data)[t{id}_i] = ((const {element}*)t{base}_data)[t{id}_base_index];"));
        }
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_sparse_plan_release(t{id}_sparse);"));
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
        // C3a invariant from `chelis_ir::verify`: `Sum.output_type.precision == accumulator`.
        // Pin it locally so a future emit refactor that decouples the two
        // gets a loud assertion instead of silent miscompilation.
        debug_assert_eq!(
            ty.precision, accumulator,
            "WS-A4: reduce_sum output precision must equal IR accumulator field; \
             verifier-enforced spec/04-type-system.md §5.7.1 invariant violated"
        );

        self.emit_reduce_sum_general(id, axis, inputs, ty, dag);
    }

    /// Snapshot one checked grouping before output shape claims or storage reuse.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduction_plan(
        &mut self,
        id: usize,
        input: Option<usize>,
        input_ty: &TensorType,
        axes: &[usize],
        output_ty: &TensorType,
        operation: &str,
        scratch: bool,
    ) {
        let axes_name = format!("t{id}_reduction_axes");
        self.emit_affine_bounds(
            &axes_name,
            &axes.iter().map(usize::to_string).collect::<Vec<_>>(),
        );
        let count = axes.len();
        let dtype = Self::dtype_macro(output_ty);
        let exemplar = format!("chelis_scalar_from_bits({dtype}, UINT64_C(0))");
        if let Some(input) = input {
            self.line(&format!("chelis_reduction_plan *t{id}_reduction = chelis_tensor_reduction_plan(t{input}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {count}), {axes_name}, {exemplar}, {operation});"));
        } else {
            let rank = input_ty.dims.len();
            let shape = Self::tagged_shape_literal(input_ty);
            self.line(&format!("chelis_reduction_plan *t{id}_reduction = chelis_shape_reduction_plan(chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {shape}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {count}), {axes_name}, {exemplar}, {operation});"));
        }
        let extents = (0..output_ty.dims.len()).map(|axis|
            (axis, format!("chelis_reduction_extent(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {axis}))"))
        ).collect::<Vec<_>>();
        self.emit_runtime_dim_sites(id, &extents);
        let rank = output_ty.dims.len();
        let shape = Self::tagged_shape_literal(output_ty);
        self.line(&format!("chelis_reduction_check_target(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {shape});"));
        let index_type = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::Int64,
        });
        self.line(&format!(
            "{index_type} t{id}_leaf_count = chelis_reduction_count(t{id}_reduction);"
        ));
        if scratch {
            self.line(&format!(
                "chelis_reduction_check_scratch(t{id}_reduction, {exemplar});"
            ));
        }
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

        self.emit_reduction_plan(id, Some(a), input_ty, axes, ty, "CHELIS_REDUCE_COUNT", true);
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
        self.line(&format!("int64_t __count_n_{id} = t{id}_leaf_count;"));
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line(&format!("chelis_tensor *__count_scratch_{id} = chelis_alloc(1, &__count_n_{id}, CHELIS_DTYPE_I64);"));
        self.line(&format!("chelis_tensor_write *__count_guard_{id} = chelis_tensor_begin_write(__count_scratch_{id});"));
        self.line(&format!(
            "int64_t *__level_{id} = (int64_t*)chelis_tensor_write_view(__count_guard_{id}).data;"
        ));
        self.line(&format!(
            "for (int64_t __r_{id} = 0; __r_{id} < __count_n_{id}; __r_{id}++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t __src_{id} = chelis_reduction_index(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)__r_{id}));"));
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
            "int64_t __next_n_{id} = __level_n_{id} / 2 + __level_n_{id} % 2;"
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
        self.line(&format!("chelis_tensor_end_write(__count_guard_{id});"));
        self.line(&format!("chelis_tensor_release(__count_scratch_{id});"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_reduction_plan_release(t{id}_reduction);"));
    }

    /// Allocate the leaves of [05-OP-30]'s adjacent-pair tree. An empty
    /// slice has no leaves: its identity is introduced only at the final store.
    fn emit_sum_level(&mut self, id: usize, axis_size: &str, precision: Prim) {
        let et = Self::elem_type(&TensorType {
            dims: vec![],
            precision,
        });
        let index_et = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::Int64,
        });
        self.line(&format!("{index_et} __sum_n_{id} = {axis_size};"));
        let dtype = Self::dtype_macro(&TensorType {
            dims: vec![],
            precision,
        });
        self.line(&format!(
            "chelis_tensor *__sum_scratch_{id} = chelis_alloc(1, &__sum_n_{id}, {dtype});"
        ));
        self.line(&format!(
            "chelis_tensor_write *__sum_guard_{id} = chelis_tensor_begin_write(__sum_scratch_{id});"
        ));
        self.line(&format!(
            "{et} *__sum_level_{id} = ({et}*)chelis_tensor_write_view(__sum_guard_{id}).data;"
        ));
    }

    /// Pair in positional order, round/check each actual addition at the
    /// accumulator width, and carry odd leaves without adding an identity.
    /// Fused and materialized Sum use this same emitted tree.
    fn emit_sum_fold(&mut self, id: usize, precision: Prim) {
        let et = Self::elem_type(&TensorType {
            dims: vec![],
            precision,
        });
        let index_et = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::Int64,
        });
        let zero = Self::scalar_zero_literal(precision);
        self.line(&format!("while (__sum_n_{id} > 1) {{"));
        self.indent += 1;
        self.line(&format!(
            "{index_et} __next_n_{id} = __sum_n_{id} / 2 + __sum_n_{id} % 2;"
        ));
        self.line(&format!(
            "for ({index_et} __j_{id} = 0; __j_{id} < __next_n_{id}; __j_{id}++) {{"
        ));
        self.indent += 1;
        self.line(&format!("{index_et} __left_{id} = 2 * __j_{id};"));
        self.line(&format!("{index_et} __right_{id} = __left_{id} + 1;"));
        let left = format!("__sum_level_{id}[__left_{id}]");
        let right = format!("__sum_level_{id}[__right_{id}]");
        let sum = if precision.is_integer() {
            let bits = Self::integer_width(precision);
            let trap = NumericTrap::Overflow {
                op: "sum",
                prim: precision,
            }
            .to_string();
            format!(
                "({et})chelis_int_checked_add(({index_et}){left}, ({index_et}){right}, {bits}, {trap:?})"
            )
        } else {
            format!("{left} + {right}")
        };
        self.line(&format!(
            "__sum_level_{id}[__j_{id}] = (__right_{id} < __sum_n_{id}) ? {sum} : {left};"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("__sum_n_{id} = __next_n_{id};"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!(
            "(({et}*)t{id}_data)[outer] = __sum_n_{id} ? __sum_level_{id}[0] : {zero};"
        ));
        self.line(&format!("chelis_tensor_end_write(__sum_guard_{id});"));
        self.line(&format!("chelis_tensor_release(__sum_scratch_{id});"));
    }

    /// Sum loads each source at its storage width, then finalizes every
    /// adjacent pair at the explicitly selected accumulator width.
    fn emit_reduce_sum_general(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let input_ty = &dag.get(inputs[0]).unwrap().output_type;
        let axis_size = format!("t{id}_leaf_count");
        let acc_et = Self::elem_type(ty);
        let operand_et = Self::elem_type(input_ty);
        self.emit_reduction_plan(
            id,
            Some(a),
            input_ty,
            &[axis],
            ty,
            "CHELIS_REDUCE_SUM",
            true,
        );
        self.emit_slot_wrapper(id, ty);
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.emit_sum_level(id, &axis_size, ty.precision);
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < __sum_n_{id}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src_idx = chelis_reduction_index(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)__reduce_i));"));
        let native = format!("((const {operand_et}*)t{a}_data)[src_idx]");
        let load = if matches!(input_ty.precision, Prim::Bf16 | Prim::F16) {
            format!("{}({native})", Self::reduced_to_f32_fn(input_ty.precision))
        } else {
            native
        };
        self.line(&format!(
            "__sum_level_{id}[__reduce_i] = ({acc_et})({load});"
        ));
        self.indent -= 1;
        self.line("}");
        self.emit_sum_fold(id, ty.precision);
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_reduction_plan_release(t{id}_reduction);"));
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
        let axis_size = format!("t{id}_leaf_count");
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
        self.emit_reduction_plan(
            id,
            Some(a),
            &input_node.output_type,
            &[axis],
            ty,
            "CHELIS_REDUCE_MAX",
            false,
        );
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
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src_idx = chelis_reduction_index(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)__reduce_i));"));
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
        self.line(&format!("chelis_reduction_plan_release(t{id}_reduction);"));
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
        let axis_size = format!("t{id}_leaf_count");
        let load = Self::reduced_to_f32_fn(ty.precision);
        let store = Self::f32_to_reduced_fn(ty.precision);
        self.emit_reduction_plan(
            id,
            Some(a),
            &input_node.output_type,
            &[axis],
            ty,
            "CHELIS_REDUCE_MAX",
            false,
        );
        self.emit_slot_wrapper(id, ty);
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line("float acc = -INFINITY;");
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src_idx = chelis_reduction_index(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)__reduce_i));"));
        // #172: propagate NaN (torch parity), matching `chelis_max_f32`.
        self.line(&format!(
            "acc = chelis_fmax_propnan_f32(acc, {load}(((uint16_t*)t{a}_data)[src_idx]));"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("((uint16_t*)t{id}_data)[outer] = {store}(acc);"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_reduction_plan_release(t{id}_reduction);"));
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
        let axis_size = format!("t{id}_leaf_count");
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
        self.emit_reduction_plan(
            id,
            Some(a),
            &input_node.output_type,
            &[axis],
            ty,
            if simd_fn == Some("chelis_min_f32") {
                "CHELIS_REDUCE_MIN"
            } else {
                "CHELIS_REDUCE_PROD"
            },
            false,
        );
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
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src_idx = chelis_reduction_index(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)__reduce_i));"));
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
        self.line(&format!("chelis_reduction_plan_release(t{id}_reduction);"));
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
    fn emit_window_plan(
        &mut self,
        id: usize,
        input: usize,
        window: &[usize],
        strides: &[usize],
        ty: &TensorType,
        operation: &str,
        side: &str,
    ) -> Result<(), Unsupported> {
        let array = |values: &[usize]| -> Result<String, Unsupported> {
            if values.is_empty() {
                return Ok("NULL".into());
            }
            let values = values
                .iter()
                .map(|&value| {
                    let value = i64::try_from(value).map_err(|_| {
                        Unsupported::new(
                            UnsupportedKind::Construct("a window parameter outside int64".into()),
                            format!("the C DAG emitter (node {id})"),
                            Stage::Codegen("c"),
                            chelis_types::deliberate_rejection!(
                                "[05-RWIN-1]",
                                "window extents and strides require positive int64 values"
                            ),
                        )
                    })?;
                    Ok(format!(
                        "chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C({value}))"
                    ))
                })
                .collect::<Result<Vec<_>, Unsupported>>()?;
            Ok(format!("(chelis_scalar[]){{{}}}", values.join(", ")))
        };
        let window = array(window)?;
        let count = strides.len();
        let strides = array(strides)?;
        self.line(&format!("chelis_window_plan *t{id}_window = chelis_tensor_window_plan(t{input}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {count}), {window}, {strides}, {operation});"));
        let extents = (0..ty.dims.len()).map(|axis| (axis, format!("chelis_window_extent(t{id}_window, {side}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {axis}))"))).collect::<Vec<_>>();
        self.emit_runtime_dim_sites(id, &extents);
        let shape = Self::tagged_shape_literal(ty);
        let rank = ty.dims.len();
        self.line(&format!("chelis_window_check_target(t{id}_window, {side}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {shape});"));
        let index = Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::Int64,
        });
        self.line(&format!(
            "{index} t{id}_window_count = chelis_window_count(t{id}_window);"
        ));
        Ok(())
    }

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
        let operation = match reducer {
            ReduceWindowKind::Sum => "CHELIS_WINDOW_SUM",
            ReduceWindowKind::Mean => "CHELIS_WINDOW_MEAN",
            ReduceWindowKind::Max => "CHELIS_WINDOW_MAX",
            ReduceWindowKind::Min => "CHELIS_WINDOW_MIN",
        };
        self.emit_window_plan(
            id,
            a,
            window_shape,
            strides,
            ty,
            operation,
            "CHELIS_WINDOW_RESULT",
        )?;
        self.emit_slot_wrapper(id, ty);
        // Arithmetic policy is unchanged here; #1298 owns canonical window
        // accumulation and extrema NaN/adjoint remediation.
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
            "for (int64_t leaf = 0; leaf < t{id}_window_count; leaf++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src_idx = chelis_window_index(t{id}_window, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)leaf));"));
        self.line(&combine_template.replace("{a}", &a.to_string()));
        self.indent -= 1;
        self.line("}");
        if matches!(reducer, ReduceWindowKind::Mean) {
            self.line(&format!("acc /= (float)t{id}_window_count;"));
        }
        self.line(&format!("((float*)t{id}_data)[outer] = acc;"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_window_plan_release(t{id}_window);"));
        Ok(())
    }

    /// Reverse-mode adjoint of `reduce_window_*` (`RiscOp::ReduceWindowGrad`).
    ///
    /// Inputs `[x, g]`: `x` is the forward windowed input (shape `S_in`),
    /// `g` the upstream cotangent (shape `S_out`). Output `din` has `x`'s
    /// shape. Each window's `g` is scattered (overlap-add) back over the
    /// window — `Sum` adds `g`, `Mean` adds `g / window_volume`, and
    /// `Max`/`Min` add `g` only at positions equal to that window's extreme
    /// (existing tie/NaN arithmetic remains tracked by #1298). Geometry follows
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
    ) -> Result<(), Unsupported> {
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
        self.emit_window_plan(
            id,
            x,
            window_shape,
            strides,
            ty,
            "CHELIS_WINDOW_GRAD",
            "CHELIS_WINDOW_SOURCE",
        )?;
        self.line(&format!(
            "chelis_window_check_tensor(t{id}_window, t{g}, CHELIS_WINDOW_RESULT);"
        ));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "for (int64_t i = 0; i < t{id}_size; i++) {{ ((float*)t{id}_data)[i] = 0.0f; }}"
        ));
        // Serial cotangent/leaf order preserves deterministic overlap addition.
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{g}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line(&format!("float gval = ((const float*)t{g}_data)[outer];"));
        if matches!(reducer, ReduceWindowKind::Max | ReduceWindowKind::Min) {
            let (init, cmp) = match reducer {
                ReduceWindowKind::Max => ("-INFINITY", "fmaxf"),
                _ => ("INFINITY", "fminf"),
            };
            self.line(&format!("float ext = {init};"));
            self.line(&format!(
                "for (int64_t leaf = 0; leaf < t{id}_window_count; leaf++) {{"
            ));
            self.indent += 1;
            self.line(&format!("int64_t src_idx = chelis_window_index(t{id}_window, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)leaf));"));
            self.line(&format!(
                "ext = {cmp}(ext, ((const float*)t{x}_data)[src_idx]);"
            ));
            self.indent -= 1;
            self.line("}");
        }
        self.line(&format!(
            "for (int64_t leaf = 0; leaf < t{id}_window_count; leaf++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t dst_idx = chelis_window_index(t{id}_window, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)leaf));"));
        match reducer {
            ReduceWindowKind::Sum => self.line(&format!("((float*)t{id}_data)[dst_idx] += gval;")),
            ReduceWindowKind::Mean => self.line(&format!("((float*)t{id}_data)[dst_idx] += gval / (float)t{id}_window_count;")),
            ReduceWindowKind::Max | ReduceWindowKind::Min => self.line(&format!("if (((const float*)t{x}_data)[dst_idx] == ext) {{ ((float*)t{id}_data)[dst_idx] += gval; }}")),
        }
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_window_plan_release(t{id}_window);"));
        Ok(())
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
        let axis_size = format!("t{id}_leaf_count");
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
        self.emit_reduction_plan(
            id,
            Some(a),
            &input_node.output_type,
            &[axis],
            ty,
            if is_argmax {
                "CHELIS_REDUCE_ARGMAX"
            } else {
                "CHELIS_REDUCE_ARGMIN"
            },
            false,
        );
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
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src_idx = chelis_reduction_index(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)__reduce_i));"));
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
        self.line(&format!("chelis_reduction_plan_release(t{id}_reduction);"));
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
        let axis_size = format!("t{id}_leaf_count");
        // ndim of the fused input (pre-reduction shape)

        self.emit_elementwise_index_steps(id, ext_inputs, fused_input_type);
        self.emit_reduction_plan(
            id,
            None,
            fused_input_type,
            &[axis],
            out_ty,
            if reduce_kind == "sum" {
                "CHELIS_REDUCE_SUM"
            } else {
                "CHELIS_REDUCE_MAX"
            },
            reduce_kind == "sum",
        );
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
        if reduce_kind == "sum" {
            self.emit_sum_level(id, &axis_size, out_ty.precision);
        } else {
            self.line(&format!("float acc = {init};"));
        }
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < {axis_size}; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t source_index = chelis_reduction_index(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)__reduce_i));"));
        for (ext_idx, ext_node) in ext_inputs.iter().enumerate() {
            let ext_id = ext_node.0;
            self.line(&format!(
                "int64_t idx_ext{ext_idx} = source_index * t{id}_input{ext_id}_step;"
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
            self.line(&format!("__sum_level_{id}[__reduce_i] = v{last};"));
        } else {
            // #172: fused max_reduce propagates NaN (torch parity),
            // matching the non-fused `chelis_max_f32` path.
            self.line(&format!("acc = chelis_fmax_propnan_f32(acc, v{last});"));
        }
        self.indent -= 1;
        self.line("}");
        if reduce_kind == "sum" {
            self.emit_sum_fold(id, out_ty.precision);
        } else {
            self.line(&format!("((float*)t{id}_data)[outer] = acc;"));
        }
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_reduction_plan_release(t{id}_reduction);"));
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
        // rank-0 bound scalar. Check claims in declaration order before the
        // legacy target validation and before allocation. Symbolic targets
        // also declare the variables used by `shape_literal` below.
        let extents: Vec<_> = new_shape
            .iter()
            .enumerate()
            .filter(|(_, dim)| matches!(dim, RtDim::Node(_) | RtDim::InputAxis { .. }))
            .map(|(axis, dim)| (axis, Self::bound_c_expr(dim, inputs, a, axis, dag)))
            .collect();
        self.emit_runtime_dim_sites(id, &extents);
        for (axis, extent) in &extents {
            self.line(&format!(
                "if (({extent}) < 0) {{ fprintf(stderr, \"chelis: runtime reshape target \
                 must be non-negative at node {id} axis {axis}\\n\"); abort(); }}"
            ));
            self.emit_static_dim_guard(id, *axis, extent, ty.dims.get(*axis));
        }
        // The runtime's checked metadata owner validates this exact target
        // before either allocation or capacity-proven repurpose can occur.
        self.line(&format!(
            "chelis_tensor_check_reshape(t{a}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C({})), {});",
            Self::ndim(ty),
            Self::tagged_shape_literal(ty),
        ));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "if (t{id}_byte_capacity != 0) memcpy(t{id}_data, t{a}_data, (size_t)t{id}_byte_capacity);"
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
        let axis_values = if axes.is_empty() {
            "0".to_string()
        } else {
            axes.iter()
                .map(|axis| format!("chelis_scalar_from_bits(CHELIS_DTYPE_I64, {axis})"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        self.line(&format!(
            "chelis_tensor_check_permute(t{a}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {}), {}, (chelis_scalar[]){{{axis_values}}});",
            Self::ndim(ty), Self::tagged_shape_literal(ty)
        ));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "chelis_scalar out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_scalar in_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_tensor_unravel_index(t{id}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i), out_indices);"
        ));
        for (new_d, &old_d) in axes.iter().enumerate() {
            self.line(&format!("in_indices[{old_d}] = out_indices[{new_d}];"));
        }
        self.line(&format!(
            "int64_t src = chelis_tensor_flat_index(t{a}, in_indices);"
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
            || self.local_dim_guard_sites.contains_key(&id)
        {
            let extent = Self::bound_c_expr(size, inputs, a, axis, dag);
            self.emit_runtime_dim_sites(id, &[(axis, extent)]);
        }
        let elem_type = Self::elem_type(ty);
        self.line(&format!(
            "chelis_tensor_check_expand(t{a}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {}), {}, {axis});",
            Self::ndim(ty), Self::tagged_shape_literal(ty)
        ));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "chelis_scalar out_indices[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_scalar in_indices[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line(&format!(
            "chelis_tensor_unravel_index(t{id}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i), out_indices);"
        ));
        self.line(&format!("if (t{id}_rank == t{a}_rank) {{"));
        self.indent += 1;
        self.line(&format!(
            "for (int d = 0; d < t{a}_rank; d++) in_indices[d] = d == {axis} ? chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0) : out_indices[d];"
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
            "int64_t src = chelis_tensor_flat_index(t{a}, in_indices);"
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
            // `emit_dim_info` renders a runtime-bound named dimension.
            RtDim::Sym(name) => name.clone(),
        }
    }

    /// Declare an op-owned runtime extent under its existing representation
    /// owner. Guard scheduling is separate from C variable declaration.
    fn emit_runtime_dim_site(&mut self, id: usize, axis: usize, extent_expr: &str) {
        if let Some((name, true)) = self.runtime_dim_sites.get(&(id, axis)) {
            let name = name.clone();
            self.declared_dim_names.insert(name.clone());
            self.line(&format!("int64_t {name} = {extent_expr};"));
        }
    }

    /// Declare supported runtime extents, then consume this operation's
    /// claims in declaration order. Supplying all axes together preserves
    /// that order even when the output permutes the signature's dimensions.
    fn emit_runtime_dim_sites(&mut self, id: usize, extents: &[(usize, String)]) {
        // Declaring and guarding are not exclusive. The legacy walk owns
        // declarations and the derivation owns guards, so an axis that
        // declares its own extent may ALSO be the axis another operation
        // makes a claim about: chelis#1277 S2b's unit-extent claim is exactly
        // that shape, since it asserts something about the `expand`'s
        // OPERAND, whose own axis a producer such as `shrink` has already
        // declared. Returning after the declaration made every such guard
        // unreachable, which is how a compiled kernel came to broadcast
        // element 0 of a two-element axis in silence.
        for (axis, extent_expr) in extents {
            self.emit_runtime_dim_site(id, *axis, extent_expr);
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
        let Some(sites) = self.local_dim_guard_sites.get(&id).cloned() else {
            return;
        };
        // Consume claims, not axes: multiple claims on one axis can be
        // interleaved with claims on another axis in declaration order.
        for (axis, site) in sites {
            // Only the extent forms supported by this movement consumer are
            // supplied here. Other local source kinds retain their existing
            // ownership in runtime_extents.md B2b-0b.
            let Some((_, extent_expr)) = extents.iter().find(|(a, _)| *a == axis) else {
                continue;
            };
            // The class's canonical value, rendered as a C expression: the
            // variable this function's prologue declared for the claim's
            // binder, or the size the checker resolved.
            let operand = site.canonical.to_string();
            let (name, op) = (site.claim, site.op);
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

    /// Materialize canonical bound scalars before allocation can repurpose a source.
    fn emit_affine_bounds(&mut self, name: &str, expressions: &[String]) {
        let values = expressions
            .iter()
            .map(|value| format!("chelis_scalar_from_bits(CHELIS_DTYPE_I64, ({value}))"))
            .collect::<Vec<_>>();
        let values = if values.is_empty() {
            "{0}".into()
        } else {
            values.join(", ")
        };
        self.line(&format!(
            "chelis_scalar {name}[{}] = {{ {values} }};",
            expressions.len().max(1)
        ));
    }

    /// Derive checked extents before consuming the existing ordered extent claims.
    fn emit_affine_shape(&mut self, id: usize, a: usize, ty: &TensorType, op: &str, bounds: &str) {
        let rank = ty.dims.len();
        self.line(&format!(
            "chelis_scalar t{id}_movement_shape[{}];",
            rank.max(1)
        ));
        self.line(&format!(
            "chelis_tensor_{op}_shape(t{a}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {bounds}, t{id}_movement_shape);"
        ));
        let extents = (0..rank)
            .map(|axis| (axis, format!("t{id}_movement_shape[{axis}].bits")))
            .collect::<Vec<_>>();
        self.emit_runtime_dim_sites(id, &extents);
        // Check every submitted axis, including static and symbolic bystanders.
        // Equal element counts alone cannot authorize a different movement shape.
        for (axis, dim) in ty.dims.iter().enumerate() {
            let expected = Self::emit_dim_info(dim);
            self.line(&format!(
                "if (t{id}_movement_shape[{axis}].bits != ({expected})) {{ fprintf(stderr, \"movement target mismatch at node {id} axis {axis}\\n\"); chelis_numeric_trap(\"numeric trap: domain in {op} at int64\"); }}"
            ));
        }
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
        let before = padding
            .iter()
            .enumerate()
            .map(|(axis, (n, _))| Self::bound_c_expr(n, inputs, a, axis, dag))
            .collect::<Vec<_>>();
        let after = padding
            .iter()
            .enumerate()
            .map(|(axis, (_, n))| Self::bound_c_expr(n, inputs, a, axis, dag))
            .collect::<Vec<_>>();
        self.emit_affine_bounds(&format!("t{id}_before"), &before);
        self.emit_affine_bounds(&format!("t{id}_after"), &after);
        self.emit_affine_bounds(&format!("t{id}_steps"), &vec!["1".into(); padding.len()]);
        self.emit_affine_shape(id, a, ty, "pad", &format!("t{id}_before, t{id}_after"));
        self.emit_slot_wrapper(id, ty);
        assert_eq!(fill.prim(), ty.precision, "verified pad fill dtype");
        let bits = match fill.element_ref() {
            ElementRef::I8(n) => u64::from(n as u8),
            ElementRef::I16(n) => u64::from(n as u16),
            ElementRef::I32(n) => u64::from(n as u32),
            ElementRef::I64(n) => n as u64,
            ElementRef::F64(n) => n.to_bits(),
            ElementRef::F32(n) => u64::from(n.to_bits()),
            ElementRef::F16(n) => u64::from(n.to_bits()),
            ElementRef::Bf16(n) => u64::from(n.to_bits()),
            ElementRef::Bool(n) => u64::from(n),
        };
        let dtype = ty
            .precision
            .runtime_dtype()
            .expect("verified pad dtype")
            .c_macro();
        if ty.precision == Prim::Int64 {
            let value = fill.as_i64_exact().expect("verified int64 pad fill");
            self.line(&format!("chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t){}));", Self::i64_c_literal(value)));
        } else {
            let literal = match ty.precision {
                Prim::F32 => format!("UINT32_C(0x{bits:08x})"),
                Prim::F64 => format!("UINT64_C(0x{bits:016x})"),
                _ => format!("UINT64_C({bits})"),
            };
            self.line(&format!("chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits({dtype}, {literal}));"));
        }
        self.line(&format!("for (int64_t i = 0; i < t{a}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "chelis_scalar coordinates[t{a}_rank > 0 ? t{a}_rank : 1];"
        ));
        self.line(&format!("chelis_tensor_unravel_index(t{a}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i), coordinates);"));
        self.line(&format!("int64_t dst = chelis_tensor_affine_index(t{id}, coordinates, t{id}_before, t{id}_steps);"));
        self.line(&format!(
            "(({et}*)t{id}_data)[dst] = ((const {et}*)t{a}_data)[i];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    fn emit_shrink(
        &mut self,
        id: usize,
        bounds: &[(RtDim, RtDim)],
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
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
        // Verified ToEnd is the identity end of a symbolic bystander; read the
        // saved input extent before any output allocation or repurpose.
        let start = bounds
            .iter()
            .enumerate()
            .map(|(axis, (n, _))| Self::bound_c_expr(n, inputs, a, axis, dag))
            .collect::<Vec<_>>();
        let end = bounds
            .iter()
            .enumerate()
            .map(|(axis, (_, n))| Self::bound_c_expr(n, inputs, a, axis, dag))
            .collect::<Vec<_>>();
        self.emit_affine_bounds(&format!("t{id}_start"), &start);
        self.emit_affine_bounds(&format!("t{id}_end"), &end);
        self.emit_affine_bounds(&format!("t{id}_steps"), &vec!["1".into(); bounds.len()]);
        self.emit_affine_shape(id, a, ty, "shrink", &format!("t{id}_start, t{id}_end"));
        // Preserve the existing runtime-bound empty-range rejection shared
        // with Eval. The metadata API also serves statically empty tensors;
        // this operation-level admission rule is separate from shape safety.
        for (axis, (start, end)) in bounds.iter().enumerate() {
            if start.node_input().is_some() || end.node_input().is_some() {
                self.line(&format!("if (t{id}_start[{axis}].bits == t{id}_end[{axis}].bits) {{ chelis_numeric_trap(\"numeric trap: domain in shrink at int64\"); }}"));
            }
        }
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "chelis_scalar coordinates[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!("chelis_tensor_unravel_index(t{id}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i), coordinates);"));
        self.line(&format!(
            "int64_t src = chelis_tensor_affine_index(t{a}, coordinates, t{id}_start, t{id}_steps);"
        ));
        self.line(&format!(
            "(({et}*)t{id}_data)[i] = ((const {et}*)t{a}_data)[src];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    fn emit_stride(
        &mut self,
        id: usize,
        strides: &[RtDim],
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        let steps = strides
            .iter()
            .enumerate()
            .map(|(axis, n)| Self::bound_c_expr(n, inputs, a, axis, dag))
            .collect::<Vec<_>>();
        self.emit_affine_bounds(&format!("t{id}_steps"), &steps);
        self.emit_affine_bounds(&format!("t{id}_offsets"), &vec!["0".into(); strides.len()]);
        self.emit_affine_shape(id, a, ty, "stride", &format!("t{id}_steps"));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "chelis_scalar coordinates[t{id}_rank > 0 ? t{id}_rank : 1];"
        ));
        self.line(&format!("chelis_tensor_unravel_index(t{id}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i), coordinates);"));
        self.line(&format!("int64_t src = chelis_tensor_affine_index(t{a}, coordinates, t{id}_offsets, t{id}_steps);"));
        self.line(&format!(
            "(({et}*)t{id}_data)[i] = ((const {et}*)t{a}_data)[src];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    // ---- Realize ----
    fn emit_realize(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
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
        self.emit_elementwise_index_steps(id, inputs, ty);
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
        // Checked scalar/identity projection is saved before destination allocation.
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
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
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
                "int64_t idx = {first_trap_index} * t{id}_input{a}_step;"
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
        assert!(c.contains("chelis_tensor_elementwise_index_step_for_shape"));
        assert!(c.contains("i * t2_input0_step"));
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
        assert!(c.contains("__sum_level_"));
        assert!(!c.contains("chelis_sum_f32("));
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
            c.contains("chelis_tensor_repurpose(t2, chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C(1)), (chelis_scalar[]){ chelis_scalar_from_bits(CHELIS_DTYPE_I64, 4) });"),
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
        assert!(c.contains(
            "in_indices[d] = d == 0 ? chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0) : out_indices[d]"
        ));
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
                    c.contains("int64_t idx = i * t1_input0_step;"),
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
            c.contains("chelis_tensor_elementwise_index_step_for_shape"),
            "cast must emit a checked element-wise loop; got:\n{c}"
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
        assert!(c.contains("i * t2_input1_step"));
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
        assert!(c.contains("chelis_tensor_affine_index(t1, coordinates, t1_before, t1_steps)"));
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
        assert!(c.contains("chelis_tensor_affine_index(t0, coordinates, t1_start, t1_steps)"));
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
        assert!(c.contains("chelis_tensor_affine_index(t0, coordinates, t1_offsets, t1_steps)"));
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
        assert!(c.contains("__sum_level_"));
        assert!(!c.contains("chelis_sum_f32("));
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
    fn matmul_pattern_retains_canonical_sum() {
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
            .expect("matmul codegen test DAG must verify ownership");
        let result = crate::codegen_with_options(verified, "test_fn", options).unwrap();
        assert!(!result.c_source.contains("cblas_sgemm("));
        assert!(result.c_source.contains("__sum_level_"));
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
        assert!(
            c.lines()
                .any(|line| line.contains("__sum_n_") && line.ends_with("_leaf_count;"))
        );
        assert!(c.contains("chelis_reduction_count("));
        assert!(c.contains("__sum_level_"));
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

    /// Materialized Expand output participates through checked identity indexing.
    #[test]
    fn binary_add_with_expanded_input_includes_slow_path() {
        let mut dag = Dag::new();
        // Expand materializes the repeated value into a contiguous [4] tensor.
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
        // The expanded operand still passes through checked shape validation.
        assert!(
            c.contains("chelis_tensor_elementwise_index_step_for_shape"),
            "elementwise inputs must have checked iteration projections"
        );
        // Contiguous identity inputs remain eligible for the optimized arm.
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
            c.contains("chelis_tensor_elementwise_index_step_for_shape"),
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

        assert!(c.contains("((const int32_t*)t1_data)[t2_index_slot]"));
        assert!(c.contains("chelis_sparse_data_index(t2_sparse,"));
        assert!(c.contains("((double*)t2_data)[t2_i] = ((const double*)t0_data)[t2_base_index];"));
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
        assert!(c.contains("chelis_sparse_index_slot(t2_sparse,"));
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

        assert!(c.contains("((const int64_t*)t1_data)[t3_index_slot]"));
        assert!(c.contains("memcpy(t3_data, t0_data, (size_t)t3_byte_capacity);"));
        assert!(c.contains("((double*)t3_data)[t3_base_index] += ((const double*)t2_data)[t3_i];"));
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
