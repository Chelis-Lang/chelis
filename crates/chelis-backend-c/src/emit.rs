//! RISC DAG to C source code emission.

use std::collections::{BTreeMap, BTreeSet};

use chelis_ir::dag::{
    ComparisonKind, Dag, DagNode, DimExpr, DimInfo, ExtremaKind, ExtremaOperand, FusedInput,
    FusedStep, FusedStepOp, LogicalKind, NodeId, ReduceWindowKind, RiscOp, RtAxis, RtDim,
    TensorType,
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
    /// chelis#1374: `(witness node, claim index)` pairs the entry-guard
    /// prologue already compares, so the witness does not emit a second
    /// comparison of the same two axes (spec/04 §4.7, "exactly once").
    entry_covered_claims: Vec<(NodeId, usize)>,
    /// [04-NUM-2]: the NaN finalization of the node being emitted, set once
    /// per node from `fp_env::risc_nan_finalization`. Every element value an
    /// op emitter produces passes through [`finalize_elem`] with it.
    nan_finalization: Option<crate::fp_env::NanFinalization>,
    /// Exact immutable input comparisons proven by the enclosing host entry.
    /// Only the private helper emitter accepts this projection; public DAG
    /// entry always supplies an empty set and retains every check.
    host_entry_coverage: Vec<chelis_ir::axis_sources::EntryExtentGuard>,
    use_blas: bool,
    /// FusedElem nodes inlined into a trailing reduction (no standalone emission).
    reduction_inlined: chelis_unord::UnordSet<usize>,
    /// Backing-slot plan for materialized C tensors.
    memory_plan: MemoryPlan,
    fused_reuse: BTreeMap<NodeId, ReusableOwnedStorage>,
    reused_sources: chelis_unord::UnordSet<NodeId>,
    slot_current_owner: BTreeMap<usize, usize>,
    /// chelis#1277 C4.4: `(node id, output axis) -> names declared there`,
    /// for every name whose extent SOURCE the function entry cannot supply.
    /// The owning operation's emitter declares the first from the extent
    /// expression it renders and each later one from the first, so two
    /// spellings of one extent land as `int64_t b = a;` rather than as a
    /// second read. Guard sites are a separate map: which sites GUARD is
    /// `local_dim_guard_sites`, and declaring does not exclude guarding.
    runtime_dim_sites: chelis_unord::UnordMap<(usize, usize), Vec<String>>,
    /// Local claims in derivation/declaration order at each operation. Grouping
    /// by axis would reorder simultaneous failures when axes are permuted.
    local_dim_guard_sites:
        chelis_unord::UnordMap<usize, Vec<(usize, chelis_ir::ownership::LocalGuardClaim)>>,
    /// Local obligations already emitted at an earlier ready point. A later
    /// realized-extent fallback may observe the same semantic claim through a
    /// different C expression, but one comparison has already discharged it.
    emitted_local_dim_guards: Vec<(usize, usize, chelis_ir::ownership::LocalGuardClaim)>,
    /// Per node id, whether it checks nothing where its activation is false
    /// ([`chelis_ir::dag::TrapSeeds::is_activation_gated`]), from one seed
    /// query over the graph.
    activation_gated: Vec<bool>,
    /// Per node id, whether its declared extent rests on a claim checked
    /// under its activation ([`chelis_ir::dag::TrapSeeds::is_claim_sized`]):
    /// where no row of its activation holds it is zero-filled rather than
    /// computed. A movement is not, since its own gate already zero-fills
    /// ([`CEmitter::emit_movement_copy`]).
    zero_filled_when_inactive: Vec<bool>,
    /// Per claim-sized node, the extent each claim states for one of its axes
    /// ([`chelis_ir::axis_sources::GuardActivation::sized_axis`]): what that
    /// axis declares where the claim is not checked.
    claimed_extents: BTreeMap<usize, Vec<(usize, chelis_ir::axis_sources::CanonicalExtent)>>,
    /// The claim-sized node whose operation is being emitted inside its
    /// activation's branch ([`CEmitter::open_inactive_zeros`]).
    inactive_zeros: Option<InactiveZeros>,
    /// The literal result claims each witness checks
    /// ([`chelis_ir::dag::TrapSeeds::literal_result_witness_requirements`]),
    /// derived once for the graph.
    literal_result_witness_requirements: BTreeMap<NodeId, Vec<chelis_types::ScalarValue>>,
    /// Claim names this function actually declares as C variables. Resolved
    /// claims compare against their numeric canonical value instead.
    declared_dim_names: chelis_unord::UnordSet<String>,
    inherited_result_sites: Vec<chelis_ir::axis_sources::ResultExtentSite>,
    inherited_result_rank: Option<usize>,
    /// Node descriptors whose exclusive runtime write lease remains live
    /// while the generated kernel fills and consumes its private storage.
    /// All leases are ended before any descriptor is returned or released.
    write_nodes: chelis_unord::UnordSet<usize>,
    /// The activation gate of the checking node being emitted, if it has an
    /// activation (spec/10 section 3.2): where it is false the node checks
    /// nothing, so each operand element it reads takes a value its checks
    /// accept ([`chelis_ir::dag::DagNode::inactive_operand`]).
    gate: Option<ActivationGate>,
}

/// A claim-sized node emitted inside its activation's branch
/// ([`CEmitter::open_inactive_zeros`]).
#[derive(Debug, Clone, Copy)]
struct InactiveZeros {
    node: usize,
    /// Where what both arms need goes: before the branch
    /// ([`CEmitter::before_inactive_zeros_branch`]).
    before_branch: usize,
    indent: usize,
    /// Whether its output was allocated before the branch.
    allocated: bool,
    /// How many runtime extent names were declared before the branch.
    declared_names: usize,
}

/// How the node being emitted reads its activation.
#[derive(Debug, Clone)]
struct ActivationGate {
    /// Whether any row is active, hoisted before the node's loops. A check
    /// of an extent or a bound, which every row of one tensor shares, runs
    /// exactly when it holds.
    any: String,
    /// The activation's node when it indexes the node's output rows (a
    /// per-row activation under `vmap`, of rank at most the node's): each
    /// element loop then runs row by row ([`CEmitter::open_element_loop`])
    /// and reads the row's activation once. `None` when every element reads
    /// [`Self::any`].
    rows: Option<usize>,
    /// Whether the element being computed is active, inside an element loop
    /// [`CEmitter::open_element_loop`] opened.
    element: String,
    /// The value each operand slot takes where the element's row is inactive,
    /// or `None` for a slot no check reads (a guarded abort's fallback).
    neutrals: Vec<Option<i64>>,
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

#[derive(Clone, Copy)]
struct SparseAxes {
    axis: usize,
    batch_rank: usize,
}

#[derive(Clone, Copy)]
enum UnaryEmission {
    Neg,
}

impl UnaryEmission {
    fn expression(self, value: &str) -> String {
        match self {
            Self::Neg => format!("-{value}"),
        }
    }
}

/// The C carrier of a random key ([05-RNG-1]'s draw keys and [05-RNG-2]'s
/// derived keys), of the seed and counter words a key is taken from, and of
/// the element index a key is read at. It is also the element type of a
/// `CHELIS_DTYPE_KEY` tensor, whose one 64-bit word per element
/// `unsigned long long` holds exactly on every supported target; every such
/// word converts exactly to and from the `uint64_t` parameters and results of
/// the runtime's key helpers.
const RANDOM_WORD_C_TYPE: &str = "unsigned long long";

/// Finalize one f32 or f64 element value of the node being emitted through
/// `fp_env::finalize_float`. Integer and reduced-float element types pass
/// through: integers have no NaN, and f16/bf16 narrow through storage helpers
/// that canonicalize.
fn finalize_elem(
    finalization: Option<crate::fp_env::NanFinalization>,
    expr: String,
    ty: &TensorType,
) -> String {
    match (finalization, ty.precision) {
        (Some(finalization), Prim::F32) => {
            crate::fp_env::finalize_float(&expr, false, finalization)
        }
        (Some(finalization), Prim::F64) => crate::fp_env::finalize_float(&expr, true, finalization),
        _ => expr,
    }
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

/// Private invocation evidence accompanies the exact verified helper graph.
/// Public tensor entries construct this without discharged host obligations.
struct InvocationEmission<'a> {
    private_invocation_context: bool,
    entry_coverage: &'a [chelis_ir::axis_sources::EntryExtentGuard],
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
            InvocationEmission {
                private_invocation_context: false,
                entry_coverage: &[],
            },
        )
    }

    /// `source_location` names the helper slot an observed invocation admits
    /// for a helper that draws.
    pub(crate) fn emit_verified_dag_with_options(
        dag: VerifiedDagView<'_>,
        func_name: &str,
        options: crate::CodegenOptions,
        entry_coverage: &[chelis_ir::axis_sources::EntryExtentGuard],
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
        Self::emit_preplanned(
            dag,
            memory_plan,
            fused_reuse,
            func_name,
            options,
            InvocationEmission {
                private_invocation_context: true,
                entry_coverage,
            },
        )
    }

    fn emit_preplanned(
        dag: VerifiedDagView<'_>,
        memory_plan: MemoryPlan,
        fused_reuse: BTreeMap<NodeId, ReusableOwnedStorage>,
        func_name: &str,
        options: crate::CodegenOptions,
        invocation: InvocationEmission<'_>,
    ) -> Result<String, Unsupported> {
        let InvocationEmission {
            private_invocation_context,
            entry_coverage,
        } = invocation;
        // chelis#1277 C4.1/C4.3: before anything reads a shape, every
        // realized output axis must have one checked extent source. This
        // runs here rather than in `codegen_with_options` because the host
        // program's tensor helpers reach the emitter through
        // `host_emit::append_helper`, which does not go through that entry,
        // and because the declaration derivation below reads those sources.
        dag.check_axis_sources(chelis_types::unsupported::Stage::Codegen("c"))?;
        // chelis#665: and every NAME this emitter will render as a C
        // identifier resolves to one place that assigns it. The check above
        // is per AXIS and this one is per NAME, which is the half the
        // occurrence walk's `panic!` covered; a receipt replaces the panic.
        dag.check_rendered_dim_origins(chelis_types::unsupported::Stage::Codegen("c"))?;
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
        let output_specs = Self::output_specs(dag);
        let output_ids = output_specs
            .iter()
            .map(|output| output.id)
            .collect::<Vec<_>>();
        // chelis#1277 C4.4: a name is declared where its extent SOURCE is
        // produced, not where a `Load` happens to carry a matching string.
        // `dim_extent_origins` answers that for every name the emitter
        // renders; an origin the entry can supply (a literal, or an input
        // tensor's axis) is declared in the prologue instead, so what lands
        // here is exactly the set the prologue cannot reach.
        //
        // Several names can land on one site, and that is the chelis#665
        // repair rather than an accident: `insert(stride(x, 2i64), 0i32,
        // shape(x, 0i32))` gives the stride's own axis and the insert's KEPT
        // axis two spellings of ONE extent, so the kept name is declared from
        // the first, `int64_t _anon_dim_2_1 = _anon_dim_1_0;`, at the stride.
        // `spec/design/runtime_extents.md` C4 calls this "an unchanged axis
        // forwards its exact input axis", and it is a DECLARATION rather than
        // a rename: overwriting the kept axis's spelling with the producer's
        // would erase a signature claim where the name is one, which is the
        // trap recorded at `rename_anonymous_dims` below for chelis#1619.
        let mut runtime_dim_sites: chelis_unord::UnordMap<(usize, usize), Vec<String>> =
            chelis_unord::UnordMap::new();
        for (name, origin) in dag.dim_extent_origins() {
            if let Some((site, axis)) = origin.local_site() {
                runtime_dim_sites
                    .entry((site.0, axis))
                    .or_default()
                    .push(name);
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
        let mut claimed_extents: BTreeMap<
            usize,
            Vec<(usize, chelis_ir::axis_sources::CanonicalExtent)>,
        > = BTreeMap::new();
        for ((node, axis), claim) in dag.local_dim_guard_sites().map_err(|reason| {
            Unsupported::new(
                UnsupportedKind::Construct("producer extent guard".into()),
                reason,
                Stage::Codegen("c"),
                chelis_types::deliberate_rejection!(
                    "[04-TOT-2]",
                    "a producer extent claim requires its exact checked output axis"
                ),
            )
        })? {
            if let Some(sized) = claim.activation.sized_axis() {
                claimed_extents
                    .entry(claim.activation.claimed().0)
                    .or_default()
                    .push((sized, claim.canonical.clone()));
            }
            let claims = local_dim_guard_sites.entry(node).or_default();
            let entry = (axis, claim);
            if !claims.contains(&entry) {
                claims.push(entry);
            }
        }

        let seeds = dag.trap_seeds();
        let mut e = CEmitter {
            lines: Vec::new(),
            indent: 0,
            entry_covered_claims: dag.entry_covered_witness_claims(),
            nan_finalization: None,
            host_entry_coverage: entry_coverage.to_vec(),
            use_blas: dag
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. })),
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
            emitted_local_dim_guards: Vec::new(),
            activation_gated: dag
                .nodes()
                .iter()
                .map(|node| seeds.is_activation_gated(node))
                .collect(),
            zero_filled_when_inactive: dag
                .nodes()
                .iter()
                .map(|node| {
                    node.owner.activation.is_some()
                        && seeds.is_claim_sized(node)
                        && node.runtime_check() != chelis_ir::dag::RuntimeCheck::MovementBounds
                })
                .collect(),
            claimed_extents,
            inactive_zeros: None,
            literal_result_witness_requirements: dag
                .nodes()
                .iter()
                .filter_map(|node| {
                    let requirements = seeds.literal_result_witness_requirements(node.id);
                    (!requirements.is_empty()).then(|| (node.id, requirements.to_vec()))
                })
                .collect(),
            declared_dim_names: chelis_unord::UnordSet::new(),
            inherited_result_sites: if private_invocation_context && dag.roots().len() == 1 {
                dag.result_extent_sites(dag.roots()[0])
            } else {
                Vec::new()
            },
            inherited_result_rank: dag
                .roots()
                .first()
                .and_then(|id| dag.get(*id))
                .map(|node| node.output_type.dims.len()),
            write_nodes: chelis_unord::UnordSet::new(),
            gate: None,
        };

        e.line("#include \"chelis_runtime.h\"");
        e.line("#include <assert.h>");
        if e.use_blas {
            e.line("#include \"chelis_blas.h\"");
            e.line(&Self::blas_integer_support());
        }
        // The C port of the random kernels' one boundary (chelis#2408):
        // `chelis_random_unit` is `[05-RNG-1]`'s unit value of the word under
        // a draw's key. The `[05-OP-8]` samplers take a draw key, never a
        // seed. `host_emit`'s `append_uniform_sample_helper` carries the same
        // lines, and a unit test there holds the two copies equal.
        e.line("/* CHELIS_UNIFORM_HELPERS_BEGIN */");
        e.line("static inline uint64_t chelis_random_mix(uint64_t value) {");
        e.line("    value += 0x9E3779B97F4A7C15ULL;");
        e.line("    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;");
        e.line("    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;");
        e.line("    return value ^ (value >> 31);");
        e.line("}");
        e.line("static inline double chelis_random_unit(uint64_t key, uint64_t index) {");
        e.line("    uint64_t element = chelis_random_mix(index);");
        e.line("    uint64_t word = chelis_random_mix(key ^ ((element << 41) | (element >> 23)));");
        e.line("    return (double)(word >> 11) / (double)(1ULL << 53);");
        e.line("}");
        // chelis#770: one explicit correctly-rounded FMA rather than
        // `low + (high - low) * (float)unit`, which `-ffp-contract` would
        // contract or not depending on flags. `fmaf` is IEEE correctly
        // rounded on every target and bit-identical to the evaluator's
        // `f32::mul_add`.
        e.line(
            "static inline float chelis_uniform_sample_f32(uint64_t key, uint64_t index, float low, float high) {",
        );
        e.line("    return fmaf(high - low, (float)chelis_random_unit(key, index), low);");
        e.line("}");
        e.line(
            "static inline double chelis_uniform_sample_f64(uint64_t key, uint64_t index, double low, double high) {",
        );
        e.line("    return fma(high - low, chelis_random_unit(key, index), low);");
        e.line("}");
        // [05-RNG-2]'s derive, only in a kernel that derives keys, so every
        // other kernel's source is unchanged. `host_emit` carries the same
        // lines after the samplers.
        if dag.nodes().iter().any(|node| {
            matches!(
                node.op,
                RiscOp::Split { .. } | RiscOp::FoldIn | RiscOp::SplitN { .. }
            )
        }) {
            e.line("static inline uint64_t chelis_key_derive(uint64_t key, uint64_t index) {");
            e.line("    uint64_t mixed = chelis_random_mix(index);");
            e.line("    return chelis_random_mix(key ^ ((mixed << 29) | (mixed >> 35)));");
            e.line("}");
        }
        e.line("/* CHELIS_UNIFORM_HELPERS_END */");
        for helper_line in crate::fp_env::helper_lines() {
            e.line(helper_line);
        }
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
        // requires escaping `%`, `\\`, `"`, and control bytes. The identifier
        // context inherits whatever the upstream chooses; if `func_name`
        // contains non-identifier bytes the emitted C will fail to compile,
        // which is the desired outcome (loud failure, not silent injection).
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);
        // Only host-owned tensor helpers receive the invocation context.
        // Standalone/public kernels keep the four-argument tensor ABI.
        let invocation_param = if private_invocation_context {
            ", const __chelis_host_result_claim *__chelis_caller_result_claims".to_string()
        } else {
            String::new()
        };
        let declaration = format!(
            "void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out{invocation_param});"
        );
        if !options.static_entry {
            e.line(&crate::generated_header::render_direct_export_begin(
                func_name,
                func_name,
                &declaration,
            ));
        }
        e.line(&format!(
            "{linkage}{} {{",
            declaration.trim_end_matches(';')
        ));
        e.indent = 1;
        if !options.static_entry {
            e.line(crate::fp_env::ENTRY);
        }

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
            e.emit_input_axis_result_guards(node, dag);
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
            let mut realized = e
                .inherited_result_sites
                .iter()
                .filter(|site| {
                    site.producer() == node.id
                        && matches!(
                            site.observation(),
                            chelis_ir::axis_sources::LocalGuardObservation::RealizedExtent
                        )
                })
                .map(|site| {
                    let chelis_ir::dag::RtAxis::Lit(axis) = site.producer_axis();
                    (
                        axis as usize,
                        format!("chelis_tensor_shape(t{}, {axis})", node.id.0),
                    )
                })
                .collect::<Vec<_>>();
            if let Some(sites) = e.local_dim_guard_sites.get(&node.id.0) {
                for (axis, claim) in sites {
                    if matches!(
                        claim.observed,
                        chelis_ir::axis_sources::LocalGuardObservation::RealizedExtent
                    ) && !realized.iter().any(|(existing, _)| existing == axis)
                    {
                        realized.push((
                            *axis,
                            format!("chelis_tensor_shape(t{}, {axis})", node.id.0),
                        ));
                    }
                }
            }
            e.emit_runtime_dim_sites(node.id.0, &realized);
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
        if !options.static_entry {
            e.line(crate::fp_env::EXIT);
        }

        e.indent = 0;
        e.line("}");
        if !options.static_entry {
            e.line(&crate::generated_header::render_direct_export_end(
                func_name,
            ));
        }
        // chelis#665: `declared_dim_names` used to be written and never read.
        // Make it the executable invariant it was always shaped like: every
        // name this emitter can render as a C identifier ends the function
        // with a declaration. The entry declarations and the per-operation
        // ones are two separate loops over two separate origin kinds, and
        // nothing else checked that between them they covered the set.
        //
        // Over-declaring is harmless (an unused `int64_t` is at worst a
        // warning); rendering an identifier nothing declares is C that does
        // not compile, which is the class the occurrence walk's `panic!` was
        // guarding and the reason a receipt has to take its place rather than
        // nothing taking it.
        //
        // Covering the set is not enough: a declaration the operation order
        // places after a reference is the same C that does not compile
        // (chelis#2883, where the only site that produces the extent runs
        // after an earlier operation's guard reads it). So the receipt also
        // covers a name the finished function reads before declaring. The
        // order comes from the marks the emitter put on each name where it
        // rendered or declared it, read in the finished function's order;
        // a spelling the emitter did not mark is never an extent read, and a
        // mark character the emitter did not write is refused.
        let rendered = dag.rendered_dim_names();
        let declared = &e.declared_dim_names;
        let settled = settle_extent_marks(&mut e.lines, |name| {
            declared.contains(name) || rendered.iter().any(|known| known == name)
        });
        let unwritten_mark = |defect: String| {
            chelis_types::unsupported::Unsupported::new(
                chelis_types::unsupported::UnsupportedKind::Construct(format!(
                    "extent mark the emitter did not write ({defect})"
                )),
                format!("emitted function `{func_name}`"),
                chelis_types::unsupported::Stage::Codegen("c"),
                chelis_types::unimplemented_rejection!(
                    1277,
                    "the emitter marks each extent read and declaration once; see runtime_extents.md C4"
                ),
            )
        };
        let read_first = settled.map_err(&unwritten_mark)?;
        let violation = rendered
            .into_iter()
            .find(|name| !e.declared_dim_names.contains(name))
            .map(|name| (name, "is rendered but never declared"))
            .or_else(|| read_first.map(|name| (name, "is rendered before it is declared")));
        if let Some((missing, how)) = violation {
            return Err(chelis_types::unsupported::Unsupported::new(
                chelis_types::unsupported::UnsupportedKind::Construct(format!(
                    "extent `{missing}` {how}"
                )),
                format!("emitted function `{func_name}`"),
                chelis_types::unsupported::Stage::Codegen("c"),
                chelis_types::unimplemented_rejection!(
                    1277,
                    "declare the name from its resolved extent origin; see runtime_extents.md C4.4"
                ),
            ));
        }
        let source = crate::fp_env::prune_unused_nan_helpers(&e.lines.join("\n"));
        if source.contains(EXTENT_MARKS) {
            return Err(unwritten_mark(
                "a mark character remains after settling".into(),
            ));
        }
        Ok(source)
    }

    fn emit_tensor_snapshot(&mut self, id: usize, writable: bool) {
        self.line(&format!("int32_t t{id}_rank = chelis_tensor_rank(t{id});"));
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

    /// Finish a generated tensor's exclusive write lease before a checked
    /// scalar read, then refresh its data pointer through the public read
    /// view for any later consumers.
    ///
    /// Straight-line kernels ordinarily retain write leases until cleanup.
    /// A path-local extent guard is different: its activation is a computed
    /// rank-zero Bool that must cross the exact tagged scalar carrier. The
    /// runtime correctly refuses `chelis_tensor_to_scalar` while the write
    /// lease is active, so discharge that producer lease at the first guard
    /// read rather than bypassing ownership through its raw data pointer.
    fn finish_tensor_write_for_checked_read(&mut self, id: usize) {
        if !self.write_nodes.remove(&id) {
            return;
        }
        self.line(&format!("chelis_tensor_end_write(t{id}_write_guard);"));
        self.line(&format!(
            "chelis_read_view t{id}_checked_read = chelis_tensor_read_view(t{id});"
        ));
        self.line(&format!("t{id}_data = (void*)t{id}_checked_read.data;"));
    }

    fn emit_owned_tensor(&mut self, id: usize, ndim: &str, shape: &str, dtype: &str) {
        self.inactive_zeros_allocation(id, |emitter| {
            emitter.line(&format!(
                "chelis_tensor *t{id} = chelis_alloc({ndim}, {shape}, {dtype});"
            ));
            emitter.emit_tensor_snapshot(id, true);
        });
    }

    /// chelis#1788 shape 2. Two roots in ONE emitted function whose signatures
    /// each spell the same dimension binder, independently.
    ///
    /// Declarations are keyed by NAME graph-wide, in `resolve_named_dim_origin`
    /// and in the prologue loop, so a merged kernel over
    /// `f(x: tensor[seq, f32])` and `g(y: tensor[batch, seq, f32])` declared
    /// `seq` once from the first input slot and sized the second root's work
    /// with the first root's extent. Nothing compared them, because the two
    /// witnesses are in different scopes and scoping correctly refuses to make
    /// them one class; the defect was that they were nevertheless one C
    /// variable.
    ///
    /// The repair gives each scope after the first its own identity,
    /// `<name>__s<k>`, so the prologue declares one variable per scope and each
    /// root reads its own input. It runs BEFORE `rename_anonymous_dims` so the
    /// anonymous pass propagates whatever identity a scope ended up with.
    ///
    /// Only names a `Load` axis DECLARES are renamed, and only in scopes that
    /// share no node with an earlier scope, because a shared node cannot carry
    /// two names for one axis. `node_scopes` is coarse for exactly that reason.
    ///
    /// Fail-closed rather than half-renamed: if any op still references the old
    /// name through a payload this pass does not rewrite, the whole pass is
    /// abandoned and the graph keeps the behaviour it had. A partially renamed
    /// graph would declare a symbol nothing reads and read a symbol nothing
    /// declares.
    pub(crate) fn rename_scoped_dims(dag: Dag) -> Dag {
        fn rename_dim_expr(expr: &DimExpr, from: &str, to: &str) -> DimExpr {
            match expr {
                DimExpr::Concrete(value) => DimExpr::Concrete(*value),
                DimExpr::Sym(name) if name == from => DimExpr::Sym(to.to_string()),
                DimExpr::Sym(name) => DimExpr::Sym(name.clone()),
                DimExpr::Mul(lhs, rhs) => DimExpr::Mul(
                    Box::new(rename_dim_expr(lhs, from, to)),
                    Box::new(rename_dim_expr(rhs, from, to)),
                ),
                DimExpr::Div(lhs, rhs) => DimExpr::Div(
                    Box::new(rename_dim_expr(lhs, from, to)),
                    Box::new(rename_dim_expr(rhs, from, to)),
                ),
            }
        }
        fn rename_rt_dim(dim: &RtDim, from: &str, to: &str) -> RtDim {
            match dim {
                RtDim::Sym(name) if name == from => RtDim::Sym(to.to_string()),
                other => other.clone(),
            }
        }
        fn rename_op(op: &RiscOp, from: &str, to: &str) -> RiscOp {
            let mut renamed = op.clone();
            match &mut renamed {
                RiscOp::Reshape { new_shape } => {
                    for dim in new_shape.iter_mut() {
                        *dim = rename_rt_dim(dim, from, to);
                    }
                }
                RiscOp::Expand { size, .. } => *size = rename_rt_dim(size, from, to),
                RiscOp::Shrink { bounds } => {
                    for (start, end) in bounds.iter_mut() {
                        *start = rename_rt_dim(start, from, to);
                        *end = rename_rt_dim(end, from, to);
                    }
                }
                RiscOp::Pad { padding, .. } => {
                    for (before, after) in padding.iter_mut() {
                        *before = rename_rt_dim(before, from, to);
                        *after = rename_rt_dim(after, from, to);
                    }
                }
                RiscOp::BlasMatmul {
                    batch_dims,
                    m,
                    n,
                    k,
                    ..
                } => {
                    for dim in batch_dims.iter_mut() {
                        *dim = rename_dim_expr(dim, from, to);
                    }
                    *m = rename_dim_expr(m, from, to);
                    *n = rename_dim_expr(n, from, to);
                    *k = rename_dim_expr(k, from, to);
                }
                _ => {}
            }
            renamed
        }

        let scopes = chelis_ir::node_scopes(&dag);
        // Which scopes declare each name through a `Load` axis, in the order
        // the scopes are numbered.
        let mut declared: Vec<(String, Vec<usize>)> = Vec::new();
        for node in dag.nodes() {
            if !matches!(node.op, RiscOp::Load { .. }) {
                continue;
            }
            let Some(Some(scope)) = scopes.get(node.id.0) else {
                continue;
            };
            for dim in &node.output_type.dims {
                let DimInfo::Named(name, None) = dim else {
                    continue;
                };
                if name.is_empty() || name == "*" {
                    continue;
                }
                match declared.iter_mut().find(|(seen, _)| seen == name) {
                    Some((_, list)) => {
                        if !list.contains(scope) {
                            list.push(*scope);
                        }
                    }
                    None => declared.push((name.clone(), vec![*scope])),
                }
            }
        }
        // `(scope, old, new)` for every scope after a name's first.
        //
        // The new identity must be FRESH. `<name>__s<k>` is a legal Chelis
        // dimension name, so a graph can already carry it, and renaming onto a
        // name another scope declares would recreate the collision this pass
        // exists to remove. `dimension_identity_names` is the enumerator that
        // answers which identities a DAG already carries, output axes and
        // op-internal payloads alike, so the suffix is bumped until it answers
        // no. Names minted here are reserved as they are chosen, because two
        // scopes of two different binders can otherwise pick the same one.
        let mut taken = chelis_ir::dag::dimension_identity_names(&dag);
        let mut renames: Vec<(usize, String, String)> = Vec::new();
        for (name, mut list) in declared {
            if list.len() < 2 {
                continue;
            }
            list.sort_unstable();
            for (index, scope) in list.into_iter().enumerate().skip(1) {
                let mut suffix = index;
                let mut fresh = format!("{name}__s{suffix}");
                while taken.contains(&fresh) {
                    suffix += 1;
                    fresh = format!("{name}__s{suffix}");
                }
                taken.insert(fresh.clone());
                renames.push((scope, name.clone(), fresh));
            }
        }
        if renames.is_empty() {
            return dag;
        }

        let mut out = dag.clone();
        let ids: Vec<NodeId> = out.nodes().iter().map(|node| node.id).collect();
        for id in ids {
            let Some(Some(scope)) = scopes.get(id.0).copied() else {
                continue;
            };
            let Some(node) = out.get(id) else {
                continue;
            };
            let mut op = node.op.clone();
            let inputs = node.inputs.clone();
            let mut ty = node.output_type.clone();
            let mut changed = false;
            for (renamed_scope, from, to) in &renames {
                if *renamed_scope != scope {
                    continue;
                }
                for dim in ty.dims.iter_mut() {
                    if let DimInfo::Named(name, size) = dim
                        && name == from
                    {
                        *dim = DimInfo::Named(to.clone(), *size);
                        changed = true;
                    }
                }
                let next = rename_op(&op, from, to);
                if next != op {
                    op = next;
                    changed = true;
                }
                if chelis_ir::dag::op_references_symbol(&op, from) {
                    // A payload this pass does not rewrite still names the old
                    // identity. Abandon rather than emit a half-renamed graph.
                    return dag;
                }
            }
            if changed {
                out.replace_node(id, op, inputs, ty);
            }
        }
        out
    }

    /// Give every anonymous extent its identity by
    /// [`chelis_ir::anonymous_dims::anonymous_axis_names`], in node order, so
    /// an axis that reads another node's reads it already named. The runtime
    /// `if` join reads the same rule to decide which arms share an extent.
    pub(crate) fn rename_anonymous_dims(dag: Dag) -> Dag {
        use chelis_ir::anonymous_dims::{
            AnonymousAxisName, anonymous_axis_names, fresh_anonymous_dim,
        };
        let mut out = dag;
        // DAG exposes no `nodes_mut`; rewrite by round-tripping replace_node.
        let ids: Vec<_> = out.nodes().iter().map(|n| n.id).collect();
        for id in ids {
            let Some(names) = anonymous_axis_names(&out, id) else {
                continue;
            };
            let node = out.get(id).expect("named node exists");
            let mut new_ty = node.output_type.clone();
            new_ty.dims = names
                .iter()
                .enumerate()
                .map(|(axis, name)| match *name {
                    AnonymousAxisName::Own => node.output_type.dims[axis].clone(),
                    AnonymousAxisName::Literal(extent) => chelis_ir::dag::DimInfo::Lit(extent),
                    AnonymousAxisName::Of {
                        node: source,
                        axis: read,
                    } => out
                        .get(source)
                        .expect("named source exists")
                        .output_type
                        .dims[read]
                        .clone(),
                    AnonymousAxisName::Fresh => {
                        fresh_anonymous_dim(id, axis, &node.output_type.dims[axis])
                    }
                })
                .collect();
            let op = node.op.clone();
            let inputs = node.inputs.clone();
            out.replace_node(id, op, inputs, new_ty);
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
        let Some(agreement) = dag.same_shape_result_agreement(node.id) else {
            return;
        };
        let input_dims = agreement
            .members()
            .iter()
            .map(|input| {
                dag.get(*input)
                    .expect("verified agreement member")
                    .output_type
                    .dims
                    .as_slice()
            })
            .collect::<Vec<_>>();
        let all_static =
            dims_static(&node.output_type.dims) && input_dims.iter().all(|dims| dims_static(dims));
        let statically_compatible = input_dims.iter().enumerate().all(|(left_index, left)| {
            input_dims[left_index + 1..]
                .iter()
                .all(|right| left.is_empty() || right.is_empty() || left == right)
        });
        if (all_static && statically_compatible) || agreement.members().len() < 2 {
            return;
        }
        let id = node.id.0;
        for (left_index, left) in agreement.members().iter().enumerate() {
            let a = left.0;
            for right in &agreement.members()[left_index + 1..] {
                let b = right.0;
                self.line(&format!(
                    "if (t{a}_rank != t{b}_rank) {{ \
                     fprintf(stderr, \"chelis: elementwise operand rank mismatch at node {id}: %d vs %d\\n\", \
                     t{a}_rank, t{b}_rank); abort(); }} \
                     if (t{a}_rank == t{b}_rank) {{ for (int __d = 0; __d < t{a}_rank; __d++) {{ \
                     if (chelis_tensor_shape(t{a}, __d) != chelis_tensor_shape(t{b}, __d)) {{ fprintf(stderr, \"chelis: \
                     elementwise operand shape mismatch at node {id} axis %d\\n\", __d); abort(); \
                     }} }} }}"
                ));
            }
        }
    }

    /// Hoist the activation of a node that checks nothing where it is false
    /// ([`chelis_ir::dag::TrapSeeds::is_activation_gated`], spec/10 section 3.2) before its
    /// loops and record how its loops read it. A rank-0 activation is read
    /// once. A per-row activation (a `vmap`ped `if`), shaped like the node's
    /// leading axes, is read once per row by each element loop
    /// ([`Self::open_element_loop`]); one the node's rank cannot index (a
    /// shared scalar under a per-row activation) decides as "some row is
    /// active". Every emitter of a gated class reads the gate: an operand
    /// value check through [`Self::gated`], an extent or bound check
    /// through [`Self::gated_check`].
    fn emit_activation_gate(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) {
        self.gate = None;
        if !self.activation_gated[node.id.0] {
            return;
        }
        let activation = node
            .owner
            .activation
            .expect("a gated node has an activation");
        let id = node.id.0;
        let act = activation.0;
        let bool_et = Self::prim_elem_type(Prim::Bool);
        let rank = dag
            .get(activation)
            .expect("verified activation")
            .output_type
            .dims
            .len();
        let any = format!("__act_{id}");
        let (rows, element) = if rank == 0 {
            self.line(&format!(
                "const int {any} = (((const {bool_et}*)t{act}_data)[0] != 0);"
            ));
            (None, any.clone())
        } else {
            self.line(&format!(
                "const int64_t __act_rows_{id} = chelis_tensor_numel(t{act});"
            ));
            self.line(&format!("int {any} = 0;"));
            self.line(&format!(
                "for (int64_t __r = 0; __r < __act_rows_{id}; ++__r) {any} |= (((const {bool_et}*)t{act}_data)[__r] != 0);"
            ));
            // The tensor whose leading axes are the activation's rows: a
            // guarded abort's condition, whose elements it checks, and every
            // other node's output.
            let indexed_rank = match &node.op {
                RiscOp::GuardedFail { .. } => dag
                    .get(node.inputs[0])
                    .expect("verified guarded_fail condition")
                    .output_type
                    .dims
                    .len(),
                _ => node.output_type.dims.len(),
            };
            if rank <= indexed_rank {
                (Some(act), format!("__act_row_{id}"))
            } else {
                (None, any.clone())
            }
        };
        self.gate = Some(ActivationGate {
            any,
            rows,
            element,
            neutrals: (0..node.inputs.len())
                .map(|slot| node.inactive_operand(slot))
                .collect(),
        });
    }

    /// Operand element `elem` of slot `slot` as the node being emitted reads
    /// it: unchanged without an activation gate, and otherwise the value its
    /// checks accept where the element's row is inactive. Read inside an
    /// element loop [`Self::open_element_loop`] opened.
    fn gated(&self, elem: String, slot: usize) -> String {
        match &self.gate {
            Some(gate) => match gate.neutrals.get(slot).copied().flatten() {
                Some(neutral) => format!("(({}) ? ({elem}) : {neutral})", gate.element),
                None => elem,
            },
            None => elem,
        }
    }

    /// Open the loop `for (int64_t {var} = 0; {var} < {size}; {var}++)`
    /// over the elements of the node being emitted, whose leading axes are
    /// its activation's rows. Under a per-row gate ([`ActivationGate::rows`])
    /// it runs row by row: the row's activation is read once, into
    /// `__act_row_{id}`, and an inner loop covers the row's elements, so an
    /// element reads its activation without a division or a load. `pragma`
    /// is the loop's OpenMP line; the row loop keeps it without `simd`.
    /// [`Self::close_element_loop`] closes what this opens.
    fn open_element_loop(&mut self, id: usize, var: &str, size: &str, pragma: Option<&str>) {
        match self.gate.as_ref().and_then(|gate| gate.rows) {
            Some(act) => {
                let bool_et = Self::prim_elem_type(Prim::Bool);
                if let Some(pragma) = pragma {
                    self.line(&pragma.replace(" simd", ""));
                }
                self.line(&format!(
                    "for (int64_t __row_{id} = 0; __row_{id} < __act_rows_{id}; __row_{id}++) {{"
                ));
                self.indent += 1;
                self.line(&format!(
                    "const int __act_row_{id} = (((const {bool_et}*)t{act}_data)[__row_{id}] != 0);"
                ));
                self.line(&format!(
                    "const int64_t __row_len_{id} = ({size}) / __act_rows_{id};"
                ));
                self.line(&format!(
                    "for (int64_t {var} = __row_{id} * __row_len_{id}; {var} < (__row_{id} + 1) * __row_len_{id}; {var}++) {{"
                ));
            }
            None => {
                if let Some(pragma) = pragma {
                    self.line(pragma);
                }
                self.line(&format!(
                    "for (int64_t {var} = 0; {var} < {size}; {var}++) {{"
                ));
            }
        }
        self.indent += 1;
    }

    /// Close a loop [`Self::open_element_loop`] opened.
    fn close_element_loop(&mut self) {
        self.indent -= 1;
        self.line("}");
        if self.gate.as_ref().is_some_and(|gate| gate.rows.is_some()) {
            self.indent -= 1;
            self.line("}");
        }
    }

    /// The condition under which the node being emitted checks an extent or
    /// a bound (`check`): unchanged without a gate, and otherwise only where
    /// some row of its activation holds, since every row of one tensor
    /// shares its extents (spec/10 section 3.2).
    fn gated_check(&self, check: &str) -> String {
        match &self.gate {
            Some(gate) => format!("{} && ({check})", gate.any),
            None => check.to_string(),
        }
    }

    fn emit_node(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) -> Result<(), Unsupported> {
        let id = node.id.0;
        let first_line = self.lines.len();
        self.nan_finalization = crate::fp_env::risc_nan_finalization(&node.op);
        self.emit_activation_gate(node, dag);
        // chelis#664/#1948: every semantic same-shape producer validates its
        // complete positive-rank operand relation before the op emitters index
        // operands through the output's shape. Nonmembers and rank-zero
        // results return immediately inside the shared derivation.
        let operand_guard = self.lines.len();
        self.emit_elementwise_operand_guard(node, dag);
        self.gate_claim_sized_operand_guard(id, operand_guard);
        // chelis#1948: operand agreement precedes the producer-owned result
        // claim, and both precede the operation's allocation or first access.
        self.emit_same_shape_result_guards(node);
        self.open_inactive_zeros(node);
        match &node.op {
            RiscOp::Softmax { .. } => {
                return Err(Unsupported::new(
                    UnsupportedKind::Op("softmax".into()),
                    "undecomposed softmax reached C emission",
                    Stage::Codegen("c"),
                    chelis_types::deliberate_rejection!(
                        "[05-OP-48]",
                        "prepare the retained softmax identity after AD and before ownership/codegen"
                    ),
                ));
            }
            RiscOp::Const { value } => self.emit_const(id, value, &node.output_type)?,
            RiscOp::ConstTensor { data } => self.emit_const_tensor(id, data, &node.output_type)?,
            RiscOp::Shape { axis } => self.emit_shape(id, *axis, &node.inputs, &node.output_type),
            RiscOp::ExtentWitness {
                site: chelis_ir::dag::ExtentWitnessSite::LiteralResultClaim,
                requirements,
                ..
            } => self.emit_const(id, &requirements[0], &node.output_type)?,
            RiscOp::ExtentWitness {
                site: chelis_ir::dag::ExtentWitnessSite::LocalAscriptionClaim { .. },
                requirements,
                ..
            } if !requirements.is_empty() => {
                self.emit_const(id, &requirements[0], &node.output_type)?
            }
            RiscOp::ExtentWitness {
                site,
                parameter,
                axis: RtAxis::Lit(axis),
                requirements,
                claims,
            } => {
                let operation = match site {
                    chelis_ir::dag::ExtentWitnessSite::Caller => "load",
                    chelis_ir::dag::ExtentWitnessSite::LocalExpand => "expand",
                    chelis_ir::dag::ExtentWitnessSite::ResultClaim { .. } => "shape",
                    chelis_ir::dag::ExtentWitnessSite::LocalAscriptionClaim { .. } => "shape",
                    chelis_ir::dag::ExtentWitnessSite::LiteralResultClaim => {
                        unreachable!("literal role handled above")
                    }
                };
                let input = node.inputs[0].0;
                let parameter = match site {
                    chelis_ir::dag::ExtentWitnessSite::Caller => {
                        chelis_ir::span_sanitize::sanitize_for_format_string(parameter).to_string()
                    }
                    chelis_ir::dag::ExtentWitnessSite::LocalExpand => format!("node {input}"),
                    chelis_ir::dag::ExtentWitnessSite::ResultClaim { .. } => parameter.clone(),
                    chelis_ir::dag::ExtentWitnessSite::LocalAscriptionClaim { .. } => {
                        parameter.clone()
                    }
                    chelis_ir::dag::ExtentWitnessSite::LiteralResultClaim => {
                        unreachable!("literal role handled above")
                    }
                };
                let literal = self
                    .literal_result_witness_requirements
                    .get(&node.id)
                    .cloned();
                for required in requirements.iter().chain(literal.iter().flatten()) {
                    let required = required.as_i64_exact().expect("verified i64 requirement");
                    let differs = self.gated_check(&format!(
                        "chelis_tensor_shape(t{input}, {axis}) != {required}"
                    ));
                    self.line(&format!("if ({differs}) {{"));
                    self.indent += 1;
                    self.line(&format!("fprintf(stderr, \"extent `{required}`: claimed = %lld, {parameter} axis {axis} = %lld\\n\", (long long){required}, (long long)chelis_tensor_shape(t{input}, {axis}));"));
                    self.line(&format!(
                        "chelis_numeric_trap(\"numeric trap: domain in {operation} at i64\");"
                    ));
                    self.indent -= 1;
                    self.line("}");
                }
                // chelis#1374/#1376: §4.7.2's named half. The requirement is
                // another witness of this activation, so both records read a
                // tensor's own shape metadata rather than a compile-time
                // number, and the declaring side leads as it does in the
                // `Load`-witnessed entry guards.
                for (index, (claim, edge)) in
                    claims.iter().zip(node.inputs.iter().skip(1)).enumerate()
                {
                    if self.entry_covered_claims.contains(&(node.id, index)) {
                        continue;
                    }
                    let required = dag.get(*edge).expect("verified extent claim edge");
                    let RiscOp::ExtentWitness {
                        parameter: required_parameter,
                        axis: RtAxis::Lit(required_axis),
                        ..
                    } = &required.op
                    else {
                        unreachable!("the verifier requires a witness edge per named claim")
                    };
                    let required_input = required.inputs[0].0;
                    let required_parameter =
                        chelis_ir::span_sanitize::sanitize_for_format_string(required_parameter)
                            .to_string();
                    let label = chelis_ir::span_sanitize::sanitize_for_format_string(&claim.claim)
                        .to_string();
                    let here = format!("{parameter} axis {axis} = %lld");
                    let there = format!("{required_parameter} axis {required_axis} = %lld");
                    let here_value = format!("(long long)chelis_tensor_shape(t{input}, {axis})");
                    let there_value = format!(
                        "(long long)chelis_tensor_shape(t{required_input}, {required_axis})"
                    );
                    let (first, first_value, second, second_value) = if claim.requirement_declares {
                        (there, there_value, here, here_value)
                    } else {
                        (here, here_value, there, there_value)
                    };
                    let differs = self.gated_check(&format!(
                        "chelis_tensor_shape(t{input}, {axis}) != chelis_tensor_shape(t{required_input}, {required_axis})"
                    ));
                    self.line(&format!("if ({differs}) {{"));
                    self.indent += 1;
                    self.line(&format!(
                        "fprintf(stderr, \"extent `{label}`: {first}, {second}\\n\", {first_value}, {second_value});"
                    ));
                    self.line(&format!(
                        "chelis_numeric_trap(\"numeric trap: domain in {operation} at i64\");"
                    ));
                    self.indent -= 1;
                    self.line("}");
                }
                self.emit_shape(id, *axis as usize, &node.inputs, &node.output_type);
            }
            RiscOp::CheckedReshapeExtent {
                claims,
                axis: RtAxis::Lit(axis),
            } => {
                let actual = node.inputs[0].0;
                for (claim, input) in claims.iter().zip(&node.inputs[1..]) {
                    let required = input.0;
                    let claim = chelis_ir::span_sanitize::sanitize_for_format_string(claim);
                    let differs = self.gated_check(&format!(
                        "((const int64_t*)t{actual}_data)[0] != ((const int64_t*)t{required}_data)[0]"
                    ));
                    self.line(&format!("if ({differs}) {{"));
                    self.indent += 1;
                    self.line(&format!("fprintf(stderr, \"extent `{claim}`: claimed = %lld, reshape axis {axis} = %lld\\n\", (long long)((const int64_t*)t{required}_data)[0], (long long)((const int64_t*)t{actual}_data)[0]);"));
                    self.line("chelis_numeric_trap(\"numeric trap: domain in reshape at i64\");");
                    self.indent -= 1;
                    self.line("}");
                }
                self.emit_realize(id, &node.inputs, &node.output_type);
            }
            RiscOp::CheckedUnitAxis { .. } => {
                self.emit_realize(id, &node.inputs, &node.output_type)
            }
            RiscOp::Load { .. } => unreachable!("handled in emit_dag"),
            RiscOp::Add => self.emit_binary(id, "+", &node.inputs, &node.output_type),
            RiscOp::Sub => self.emit_binary(id, "-", &node.inputs, &node.output_type),
            RiscOp::Mul => self.emit_binary(id, "*", &node.inputs, &node.output_type),
            RiscOp::Div => {
                if Self::is_lowered_mean_div(dag, node) {
                    self.emit_mean_nonempty_guard(id, &node.inputs, &node.output_type, dag);
                }
                self.emit_binary(id, "/", &node.inputs, &node.output_type);
            }
            // chelis#178: `trunc_div` is the C integer `/` quotient (round
            // toward zero) — `emit_binary` already wraps the divisor in the
            // portable zero-divisor guard for integer dtypes. `trunc_div`
            // is integer-only, so this is exactly C truncating division.
            RiscOp::TruncDiv => self.emit_binary(id, "/", &node.inputs, &node.output_type),
            RiscOp::Mod => self.emit_binary(id, "%", &node.inputs, &node.output_type),
            RiscOp::Bitwise(kind) => self.emit_binary(
                id,
                match kind {
                    chelis_types::BitwiseKind::And => "&",
                    chelis_types::BitwiseKind::Or => "|",
                    chelis_types::BitwiseKind::Xor => "^",
                    chelis_types::BitwiseKind::ShiftLeft => "shl",
                    chelis_types::BitwiseKind::ShiftRight => "shr",
                },
                &node.inputs,
                &node.output_type,
            ),
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
            RiscOp::Compare(kind) => {
                self.emit_compare(id, *kind, &node.inputs, &node.output_type, dag)
            }
            RiscOp::Logical(kind) => self.emit_logical(id, *kind, &node.inputs, &node.output_type),
            RiscOp::Where => self.emit_where(id, &node.inputs, &node.output_type),
            RiscOp::GuardedFail {
                message,
                trap_on_true,
            } => {
                self.emit_guarded_fail(id, &node.inputs, &node.output_type, message, *trap_on_true)
            }
            RiscOp::Neg => self.emit_unary(id, UnaryEmission::Neg, &node.inputs, &node.output_type),
            RiscOp::Recip => self.emit_recip(id, &node.inputs, &node.output_type),
            RiscOp::Exp => {
                self.emit_unary_func(id, "chelis_cr_expf", &node.inputs, &node.output_type)
            }
            RiscOp::Log => {
                self.emit_unary_func(id, "chelis_cr_logf", &node.inputs, &node.output_type)
            }
            RiscOp::Sin => {
                self.emit_unary_func(id, "chelis_cr_sinf", &node.inputs, &node.output_type)
            }
            RiscOp::Sqrt => self.emit_unary_func(id, "sqrtf", &node.inputs, &node.output_type),
            RiscOp::Cos => {
                self.emit_unary_func(id, "chelis_cr_cosf", &node.inputs, &node.output_type)
            }
            RiscOp::Tan => {
                self.emit_unary_func(id, "chelis_cr_tanf", &node.inputs, &node.output_type)
            }
            RiscOp::Atan => {
                self.emit_unary_func(id, "chelis_cr_atanf", &node.inputs, &node.output_type)
            }
            RiscOp::Tanh => {
                self.emit_unary_func(id, "chelis_cr_tanhf", &node.inputs, &node.output_type)
            }
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
            RiscOp::KeyFromSeed => self.emit_key_from_seed(node, dag),
            RiscOp::Split { branch } => self.emit_split_key(node, *branch, dag),
            RiscOp::FoldIn => self.emit_fold_in(node, dag),
            RiscOp::SplitN { count } => self.emit_split_keys(node, count),
            RiscOp::Iota => self.emit_iota(node),
            RiscOp::ListMapCapture { .. } => self.emit_list_map_capture(node),
            RiscOp::OrderedAdjointSum { groups } => self.emit_ordered_adjoint_sum(node, groups),
            RiscOp::KeySelect => self.emit_key_select(node, dag),
            RiscOp::Dropout => self.emit_keyed_dropout(node, dag),
            RiscOp::DropoutReplay => self.emit_keyed_dropout(node, dag),
            RiscOp::UniformLike => self.emit_keyed_uniform_like(node, dag),
            RiscOp::UniformBoundAdjoint { bound } => {
                self.emit_uniform_bound_adjoint(node, *bound, dag)
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
                self.emit_reduce_extreme(id, *axis, &node.inputs, &node.output_type, dag, false)?;
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
            RiscOp::Gather { axis, batch_rank } => {
                self.emit_sparse_checked(
                    id,
                    SparseAxes {
                        axis: *axis,
                        batch_rank: *batch_rank,
                    },
                    &node.inputs,
                    &node.output_type,
                    dag,
                    SparseEmission::Gather,
                );
            }
            RiscOp::ScatterAdd { axis, batch_rank } => {
                self.emit_sparse_checked(
                    id,
                    SparseAxes {
                        axis: *axis,
                        batch_rank: *batch_rank,
                    },
                    &node.inputs,
                    &node.output_type,
                    dag,
                    SparseEmission::Add,
                );
            }
            RiscOp::Scatter { axis, batch_rank } => {
                self.emit_sparse_checked(
                    id,
                    SparseAxes {
                        axis: *axis,
                        batch_rank: *batch_rank,
                    },
                    &node.inputs,
                    &node.output_type,
                    dag,
                    SparseEmission::Replace,
                );
            }
            RiscOp::ScatterElements { axis } => {
                self.emit_sparse_checked(
                    id,
                    SparseAxes {
                        axis: *axis,
                        batch_rank: 0,
                    },
                    &node.inputs,
                    &node.output_type,
                    dag,
                    SparseEmission::Elements,
                );
            }
        }
        self.assert_nan_finalized(node, dag, first_line);
        self.close_inactive_zeros(node)
    }

    /// [04-NUM-2]: a node whose classification is canonical and whose result is
    /// f32 or f64 produced its value through `fp_env::finalize_float`, so the C
    /// it emitted calls the canonicalizing helper. An emitter that writes its
    /// result another way (a BLAS call, a reduction's own store, a runtime
    /// helper) fails here, at every build that reaches it, instead of
    /// disagreeing with eval on a NaN's bits. An f16 or bf16 result computes at
    /// f32 and narrows through the canonicalizing storage helper
    /// ([04-NUM-8]), so its C calls `chelis_f32_to_f16` or `chelis_f32_to_bf16`;
    /// an emitter that does arithmetic on the `uint16_t` storage words fails
    /// here. A conversion from a non-float or from the same dtype produces no
    /// new NaN.
    fn assert_nan_finalized(&self, node: &DagNode, dag: VerifiedDagView<'_>, first_line: usize) {
        if self.nan_finalization != Some(crate::fp_env::NanFinalization::Canonical) {
            return;
        }
        // A float-to-half conversion narrows from f64 in one rounding through
        // the checked-cast helper; every other half result narrows from f32.
        let helpers: &[&str] = match node.output_type.precision {
            Prim::F32 => &[crate::fp_env::canonical_nan_helper(false)],
            Prim::F64 => &[crate::fp_env::canonical_nan_helper(true)],
            Prim::F16 => &["chelis_f32_to_f16", "chelis_host_f64_to_f16"],
            Prim::Bf16 => &["chelis_f32_to_bf16", "chelis_host_f64_to_bf16"],
            _ => return,
        };
        if let RiscOp::Cast { new_precision } = node.op {
            let source = dag
                .get(node.inputs[0])
                .expect("verified cast operand")
                .output_type
                .precision;
            if !source.is_float() || source == new_precision {
                return;
            }
        }
        assert!(
            self.lines[first_line..].iter().any(|line| helpers
                .iter()
                .any(|helper| line.contains(&format!("{helper}(")))),
            "C backend: node {} (`{}`) produces a {} result its classification \
             finalizes canonically ([04-NUM-2]), but its emitter wrote it through \
             none of `{}`, the finalizations its dtype computes through",
            node.id.0,
            chelis_ir::grad::risc_op_name(&node.op),
            node.output_type.precision.name(),
            helpers.join("`, `"),
        );
    }

    /// The condition under which a claim-sized node computes
    /// ([`chelis_ir::dag::TrapSeeds::is_claim_sized`]): some row of its
    /// activation holds. `None` for every other node.
    fn claim_sized_activity(&self, id: usize) -> Option<String> {
        if !self.zero_filled_when_inactive[id] {
            return None;
        }
        let gate = self
            .gate
            .as_ref()
            .expect("a claim-sized node under an activation is gated");
        Some(gate.any.clone())
    }

    /// Put the operand-agreement lines emitted from `start` under a
    /// claim-sized node's activation: the agreement is what its reads need,
    /// and where no row is active it reads no operand (spec/10 section 3.2),
    /// as the evaluator's zero value reads none.
    fn gate_claim_sized_operand_guard(&mut self, id: usize, start: usize) {
        let Some(active) = self.claim_sized_activity(id) else {
            return;
        };
        if self.lines.len() == start {
            return;
        }
        for line in &mut self.lines[start..] {
            line.insert_str(0, "    ");
        }
        let prefix = "    ".repeat(self.indent);
        self.lines
            .insert(start, format!("{prefix}if ({active}) {{"));
        self.line("}");
    }

    /// Open a claim-sized node's activation branch before its operation is
    /// emitted. Where no row of the activation holds, the node checks
    /// nothing and produces zeros of its declared type instead of reading an
    /// operand whose extent need not be the one it declares, as the evaluator's
    /// `inactive_unchecked_value` does and as a movement's
    /// [`Self::emit_movement_copy`] does for its bounds: the operation, every
    /// operand-shape check it makes and every guard it places run in the
    /// branch, its allocation before it ([`Self::before_inactive_zeros_branch`]), and
    /// [`Self::close_inactive_zeros`] zero-fills the other arm. A movement
    /// is not wrapped: its own gate already reads no bound and zero-fills.
    fn open_inactive_zeros(&mut self, node: &DagNode) {
        let Some(active) = self.claim_sized_activity(node.id.0) else {
            return;
        };
        self.inactive_zeros = Some(InactiveZeros {
            node: node.id.0,
            before_branch: self.lines.len(),
            indent: self.indent,
            allocated: false,
            declared_names: self.declared_dim_names.len(),
        });
        self.line(&format!("if ({active}) {{"));
        self.indent += 1;
    }

    /// Close the branch [`Self::open_inactive_zeros`] opened, zero-filling
    /// the arm where no row is active. An operation that allocates no output
    /// of its own, or that declares a runtime extent name from what it
    /// computes inside the branch, has nothing both arms can define, so it is
    /// refused.
    fn close_inactive_zeros(&mut self, node: &DagNode) -> Result<(), Unsupported> {
        let Some(zeros) = self.inactive_zeros.take() else {
            return Ok(());
        };
        self.indent -= 1;
        if !zeros.allocated || self.declared_dim_names.len() != zeros.declared_names {
            return Err(Unsupported::new(
                UnsupportedKind::Op(chelis_ir::grad::risc_op_name(&node.op).to_string()),
                format!(
                    "a claim-sized operation under an activation at C DAG node {} that {}",
                    node.id.0,
                    if zeros.allocated {
                        "declares a runtime extent itself"
                    } else {
                        "allocates no output of its own"
                    }
                ),
                Stage::Codegen("c"),
                chelis_types::unimplemented_rejection!(
                    2413,
                    "its untaken arm has no zero value both branches define (spec/10 section 3.2)"
                ),
            ));
        }
        self.line("} else {");
        self.indent += 1;
        self.emit_zero_fill(zeros.node);
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    /// Emit what both arms of `id`'s inactive-zeros branch need (`emit`): its
    /// output allocation, and runtime extent names it declares from its
    /// operands' metadata. For a claim-sized node emitted inside its
    /// activation's branch ([`Self::open_inactive_zeros`]) it is placed
    /// before that branch, so both arms define the one tensor every later
    /// reader sees; elsewhere it is emitted in place. The allocation's slot is
    /// never one of the node's operands' (only a fused kernel reuses an
    /// operand in place, and fusion keeps a gated node out), so allocating
    /// first leaves every operand read in the branch unchanged.
    fn before_inactive_zeros_branch(&mut self, id: usize, emit: impl FnOnce(&mut Self)) {
        let Some(InactiveZeros {
            before_branch,
            indent,
            ..
        }) = self.inactive_zeros.filter(|zeros| zeros.node == id)
        else {
            emit(self);
            return;
        };
        let branch = self.lines.split_off(before_branch);
        let inside = std::mem::replace(&mut self.indent, indent);
        let declared_before = self.declared_dim_names.len();
        emit(self);
        self.indent = inside;
        let before_branch = self.lines.len();
        self.lines.extend(branch);
        // Only the names declared here are declared before the branch; one
        // the branch itself declared stays inside it, and is refused.
        let hoisted_names = self.declared_dim_names.len() - declared_before;
        if let Some(zeros) = self.inactive_zeros.as_mut() {
            zeros.before_branch = before_branch;
            zeros.declared_names += hoisted_names;
        }
    }

    /// Allocate `id`'s output ([`Self::before_inactive_zeros_branch`]).
    fn inactive_zeros_allocation(&mut self, id: usize, allocate: impl FnOnce(&mut Self)) {
        self.before_inactive_zeros_branch(id, allocate);
        if let Some(zeros) = self
            .inactive_zeros
            .as_mut()
            .filter(|zeros| zeros.node == id)
        {
            zeros.allocated = true;
        }
    }

    /// Fill `t{id}` with zero bytes: the value of its declared type that an
    /// operation where no row of its activation holds produces (spec/10
    /// section 3.2), positive zeros and at `key` the key whose bits are
    /// zero, as the evaluator's `inactive_unchecked_value` produces.
    fn emit_zero_fill(&mut self, id: usize) {
        self.line(&format!(
            "if (t{id}_byte_capacity != 0) memset(t{id}_data, 0, (size_t)t{id}_byte_capacity);"
        ));
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

    pub(crate) fn output_types(dag: VerifiedDagView<'_>) -> Vec<TensorType> {
        Self::output_specs(dag)
            .into_iter()
            .map(|output| dag.get(output.id).expect("output node").output_type.clone())
            .collect()
    }

    pub(crate) fn output_labels(dag: VerifiedDagView<'_>) -> Vec<String> {
        Self::output_specs(dag)
            .into_iter()
            .map(|output| output.label)
            .collect()
    }

    pub(crate) fn input_labels(dag: VerifiedDagView<'_>) -> Vec<String> {
        chelis_ir::axis_sources::ordered_interface_loads(dag.nodes())
            .into_iter()
            .map(|node| match &node.op {
                RiscOp::Load { name } => name.as_str().to_string(),
                _ => unreachable!("ordered interface input is a Load"),
            })
            .collect()
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
                // A draw key is a C local of its draw node; every other key
                // is a `CHELIS_DTYPE_KEY` tensor.
                Prim::Key => {}
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
                            "C backend sparse gather requires i32/i64 indices, got {} at node {}",
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
                            "C backend sparse scatter_add requires i32/i64 indices, got {} at node {}",
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
                            "C backend sparse scatter_replace requires i32/i64 indices, got {} at node {}",
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

    fn emit_input_shape_preamble(
        &mut self,
        dag: VerifiedDagView<'_>,
        input_slots: &chelis_unord::UnordMap<String, usize>,
        func_name: &str,
    ) {
        // Producer-supplied strings flowing into the fprintf format string
        // baked into a `"..."` C string literal. Sanitize once per emission
        // boundary.
        let func_name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(func_name);

        // The shared IR plan follows ABI slots, interleaves each input's
        // metadata checks with its due extent witnesses, and excludes exact
        // duplicate literal comparisons. A second label/guard traversal
        // would give Eval and C different first failures (#2531).
        for step in dag.entry_validation_plan() {
            use chelis_ir::axis_sources::{EntryExtentGuard, EntryValidationStep};
            let input = |load: NodeId| {
                let node = dag.get(load).expect("entry input");
                let RiscOp::Load { name } = &node.op else {
                    unreachable!()
                };
                let slot = input_slots[name.as_str()];
                let label = chelis_ir::span_sanitize::sanitize_for_format_string(name.as_str());
                (node, slot, label)
            };
            match step {
                EntryValidationStep::DType { load } => {
                    let (node, slot, label) = input(load);
                    self.line(&format!("if (inputs[{slot}] == NULL) {{"));
                    self.indent += 1;
                    self.line(&format!("fprintf(stderr, \"{func_name_fmt}: input `{label}` at slot {slot} is NULL\\n\");"));
                    self.line("abort();");
                    self.indent -= 1;
                    self.line("}");
                    for line in Self::entry_dtype_guard(
                        &format!("inputs[{slot}]"),
                        &format!("{func_name_fmt}: input `{label}`"),
                        "",
                        &node.output_type,
                    ) {
                        self.line(&line);
                    }
                }
                EntryValidationStep::Rank { load } => {
                    let (node, slot, label) = input(load);
                    let rank = node.output_type.dims.len();
                    self.line(&format!(
                        "if (chelis_tensor_rank(inputs[{slot}]) != {rank}) {{"
                    ));
                    self.indent += 1;
                    self.line(&format!("fprintf(stderr, \"{func_name_fmt}: input `{label}` expected rank {rank}, got %d\\n\", chelis_tensor_rank(inputs[{slot}]));"));
                    self.line("chelis_numeric_trap(\"numeric trap: domain in load at i64\");");
                    self.indent -= 1;
                    self.line("}");
                }
                EntryValidationStep::LiteralAxis {
                    load,
                    axis,
                    required,
                } => {
                    if self.host_entry_coverage.iter().any(|guard| {
                        matches!(guard, EntryExtentGuard::Literal { required: covered, observed }
                            if *covered == required && *observed == (load, axis))
                    }) {
                        continue;
                    }
                    let (_, slot, label) = input(load);
                    self.line(&format!(
                        "if (chelis_tensor_shape(inputs[{slot}], {axis}) != {required}) {{"
                    ));
                    self.indent += 1;
                    self.line(&format!("fprintf(stderr, \"{func_name_fmt}: input `{label}` axis {axis} expected {required}, got %lld\\n\", (long long)chelis_tensor_shape(inputs[{slot}], {axis}));"));
                    self.line("chelis_numeric_trap(\"numeric trap: domain in load at i64\");");
                    self.indent -= 1;
                    self.line("}");
                }
                EntryValidationStep::Extent(guard) => {
                    if self.host_entry_coverage.contains(&guard) {
                        continue;
                    }
                    let input_read = |(load, axis): (NodeId, usize)| {
                        let (_, slot, label) = input(load);
                        (
                            format!("chelis_tensor_shape(inputs[{slot}], {axis})"),
                            label,
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
                            let claim =
                                chelis_ir::span_sanitize::sanitize_for_format_string(&claim);
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
                    self.line("chelis_numeric_trap(\"numeric trap: domain in load at i64\");");
                    self.indent -= 1;
                    self.line("}");
                }
            }
        }

        // DECLARATIONS come from the axis SOURCE (chelis#1277 C4.4), and
        // what the ENTRY can declare is one kind: an input tensor's axis,
        // read from shape metadata before any operation of the function
        // runs. `spec/04-type-system.md` section 4.7 orders entry work by
        // "the assigned ABI input slot", so these follow `input_slots` and
        // then the axis. Every declaration here dominates every reference, so
        // the order is determinism rather than correctness.
        //
        // A `Literal` origin is deliberately NOT declared here, and that is a
        // narrowing rather than an oversight. Declaring one would put a
        // second C element-type spelling in this function, which the runtime
        // representation inventory reads as a new seam under a new owner, and
        // growing that frozen foundation is a contract change rather than a
        // repair. Nothing measured needs it: chelis#1556, the shape that
        // would have used it, does not reproduce. A literal-origin name that
        // no operation site declares reaches the `declared_dim_names`
        // invariant below and is refused with a receipt, which is what the
        // occurrence walk did with a panic. Residual under chelis#1372.
        let mut entry_dim_declarations: Vec<((usize, usize), String)> = Vec::new();
        for (name, origin) in dag.dim_extent_origins() {
            let chelis_ir::axis_sources::ExtentOrigin::ExternalAxis { load, axis } = origin else {
                continue;
            };
            let Some(RiscOp::Load { name: label }) = dag.get(load).map(|node| &node.op) else {
                continue;
            };
            entry_dim_declarations.push(((input_slots[label.as_str()], axis), name));
        }
        entry_dim_declarations.sort_by_key(|entry| entry.0);
        for ((canonical_slot, canonical_axis), name) in entry_dim_declarations {
            self.line(&format!(
                "int64_t {} = chelis_tensor_shape(inputs[{canonical_slot}], {canonical_axis});",
                extent_declaration(&name)
            ));
            self.declared_dim_names.insert(name);
        }
    }

    fn shape_literal(ty: &TensorType) -> String {
        let dims: Vec<String> = ty.dims.iter().map(Self::emit_dim_info).collect();
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
                // The constructor's uint64 bits parameter preserves an i64
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
            DimInfo::Named(name, None) => extent_read(name),
        }
    }

    fn dtype_macro(ty: &TensorType) -> &'static str {
        ty.precision
            .runtime_dtype()
            .unwrap_or_else(|error| panic!("C backend does not support this tensor: {error}"))
            .c_macro()
    }

    /// [04-NUM-11]: an entry compares a supplied tensor's dtype tag with its
    /// declared dtype before any element is read, and a mismatch traps
    /// `Domain` in `load` at the declared dtype ([04-NUM-9]) after one context
    /// line naming the input and both dtypes. The storage is never read at the
    /// declared dtype. `input` is the sanitized format fragment that names the
    /// input and `input_args` the arguments it consumes, each preceded by a
    /// comma, or empty. The DAG entry, the host signature entry and the host
    /// entry's nested-value walk all render their guard through this one
    /// function.
    pub(crate) fn entry_dtype_guard(
        tensor: &str,
        input: &str,
        input_args: &str,
        ty: &TensorType,
    ) -> [String; 5] {
        let declared = ty.precision;
        // The supplied tag has passed the runtime's own validation, so it is
        // one of the runtime ABI's dtypes; each is spelled as its language
        // dtype, the spelling the declared side uses.
        let supplied = chelis_vocab::RuntimeDType::ALL
            .iter()
            .map(|dtype| {
                format!(
                    "__chelis_supplied_dtype == {} ? \"{}\" : ",
                    dtype.c_macro(),
                    Prim::from_runtime_dtype(*dtype).name()
                )
            })
            .collect::<String>();
        let trap = NumericTrap::Domain {
            op: "load",
            prim: declared,
        }
        .to_string();
        [
            format!(
                "if (chelis_tensor_read_view({tensor}).dtype != {}) {{",
                Self::dtype_macro(ty)
            ),
            format!(
                "    const chelis_dtype __chelis_supplied_dtype = chelis_tensor_read_view({tensor}).dtype;"
            ),
            format!(
                "    fprintf(stderr, \"{input} expected dtype {}, got %s\\n\"{input_args}, {supplied}\"an unregistered dtype\");",
                declared.name()
            ),
            format!("    chelis_numeric_trap({trap:?});"),
            "}".to_string(),
        ]
    }

    /// Returns the C element type for direct element access in generated loops.
    /// F32 uses `float`; Bool uses its canonical one-byte `uint8_t` payload.
    /// Int32 uses `int32_t` (reinterpret cast; sizeof matches float).
    /// Int64 uses `int64_t` (reinterpret cast; sizeof is 2x float, alloc adjusts).
    /// F64 uses `double` (reinterpret cast; sizeof is 2x float, alloc adjusts).
    ///
    /// **WS-A0 footgun fix.** This used to fall through to `"float"` for
    /// any unhandled `Prim`, which silently downgraded the wider
    /// active dtypes (f16, bf16, i8, i16) to single-precision in
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
            // A key tensor's element is one opaque random word.
            Prim::Key => RANDOM_WORD_C_TYPE,
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

    /// The C type of a rank-0 value of `prim`, from the element-type
    /// authority `elem_type`.
    pub(crate) fn prim_elem_type(prim: Prim) -> &'static str {
        Self::elem_type(&TensorType {
            dims: vec![],
            precision: prim,
        })
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
    /// the positive magnitude is outside i64 before unary negation.
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
    /// double-precision math helpers (`chelis_cr_exp` vs `chelis_cr_expf`) and
    /// `chelis_fill_f64`.
    fn is_f64(ty: &TensorType) -> bool {
        matches!(ty.precision, Prim::F64)
    }

    /// A libm transcendental name. Emission never names one: every
    /// transcendental is a carried `chelis_cr_*` kernel ([05-OP-46]).
    fn is_host_math_transcendental(func: &str) -> bool {
        matches!(
            func,
            "exp"
                | "expf"
                | "log"
                | "logf"
                | "sin"
                | "sinf"
                | "cos"
                | "cosf"
                | "tan"
                | "tanf"
                | "atan"
                | "atanf"
                | "tanh"
                | "tanhf"
                | "pow"
                | "powf"
                | "erf"
                | "erff"
        )
    }

    /// Double-precision equivalent of a single-precision C math symbol used
    /// by the emitter. Transcendentals name the correctly rounded kernels the
    /// generated unit carries ([05-OP-46]); `sqrt` and the rounding helpers
    /// are exact C operations. Only the set we actually route through emit_unary_func
    /// and emit_binary_func is mapped here; unmapped symbols pass through
    /// unchanged so the emitter never silently renames an unexpected function.
    fn double_math_fn(scalar_f: &str) -> &str {
        match scalar_f {
            "chelis_cr_expf" => "chelis_cr_exp",
            "chelis_cr_logf" => "chelis_cr_log",
            "chelis_cr_sinf" => "chelis_cr_sin",
            "sqrtf" => "sqrt",
            "chelis_cr_cosf" => "chelis_cr_cos",
            "chelis_cr_tanf" => "chelis_cr_tan",
            "chelis_cr_atanf" => "chelis_cr_atan",
            "chelis_cr_tanhf" => "chelis_cr_tanh",
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
        self.inactive_zeros_allocation(id, |emitter| {
            if emitter.write_nodes.remove(&previous) {
                emitter.line(&format!(
                    "chelis_tensor_end_write(t{previous}_write_guard);"
                ));
            }
            emitter.line(&format!("chelis_tensor *t{id} = t{previous};"));
            let shape = Self::tagged_shape_literal(ty);
            emitter.line(&format!(
                "chelis_tensor_repurpose(t{id}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C({})), {shape});",
                Self::ndim(ty)
            ));
            emitter.emit_tensor_snapshot(id, true);
        });
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
        // integer arms take the exact i64 (an i64 constant above 2^53
        // now emits its exact value instead of an f64-rounded one), the
        // float arms take the exact f64 image. A payload the target
        // cannot represent is a structured diagnostic through the
        // chelis#730 Result channel, never bad C.
        let wide = value.as_f64_lossy();
        let exact_int = value.as_i64_exact();
        match ty.precision {
            Prim::Int64 => {
                // The exact-width signed spelling the pad fill uses too:
                // i64::MIN's positive magnitude is no signed C literal, even
                // under unary minus (chelis#1859).
                self.line(&format!(
                    "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t){}));",
                    Self::i64_c_literal(exact_int.unwrap_or(wide as i64))
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
                    Prim::Key => chelis_types::deliberate_rejection!(
                        "[05-RNG-1]",
                        "a random key has no literal carrier; every key is produced by a draw node"
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
    /// `axis`, read from checked metadata before storage submission.
    /// The extent is stored into the scalar buffer using the
    /// node's integer precision. This is the C realization of the runtime
    /// dim read; the emitted expression reads the shape at execution time,
    /// so a symbolic input axis is resolved from the actual input tensor
    /// rather than baked at codegen time.
    fn emit_shape(&mut self, id: usize, axis: usize, inputs: &[NodeId], ty: &TensorType) {
        let a = inputs[0].0;
        // The destination may repurpose the last-use source at the same capacity.
        // Capture its logical extent while the source metadata still names it.
        let extent = format!("t{id}_shape_extent");
        self.line(&format!(
            "const int64_t {extent} = chelis_tensor_shape(t{a}, {axis});"
        ));
        self.emit_slot_wrapper(id, ty);
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
            i64::try_from(storage.len()).map_err(|_| invalid("literal count exceeds i64"))?;
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
        // `chelis_int_checked_divisor`, which aborts with the same [04-NUM-9]
        // trap the evaluator raises. Float `/` is IEEE-754 (`1.0/0.0 == inf`) and
        // is never guarded; `+`/`*`/`fmaxf` never divide.
        let checked_int =
            ty.precision.is_integer() && matches!(op, "+" | "-" | "*" | "/" | "%" | "shl" | "shr");
        let is_float = matches!(ty.precision, Prim::F32 | Prim::F64);
        let nan = self.nan_finalization;
        let elem_expr = |lhs: String, rhs: String| -> String {
            if matches!(op, "shl" | "shr") {
                let bits = Self::integer_width(ty.precision);
                return format!(
                    "({et})chelis_int_{op}((int64_t)({lhs}), (int64_t)({rhs}), {bits})"
                );
            }
            if is_relu_adjoint {
                // [05-OP-43]: select g only for +0 < x. Selection preserves
                // the exact stored cotangent bits and emits exact +0 for
                // both zeros and NaN without multiplying by a mask.
                return format!("{zero} < ({lhs}) ? ({rhs}) : {zero}");
            }
            if !checked_int {
                if is_float && matches!(op, "+" | "-" | "*" | "/") {
                    return finalize_elem(nan, format!("({lhs}) {op} ({rhs})"), ty);
                }
                // [05-OP-64] float `mod` is C `fmod`, which is exact
                // (chelis#626); only its NaN needs finalizing.
                if is_float && op == "%" {
                    let fmod = if Self::is_f64(ty) { "fmod" } else { "fmodf" };
                    return finalize_elem(nan, format!("{fmod}({lhs}, {rhs})"), ty);
                }
                return format!("{lhs} {op} {rhs}");
            }
            let bits = Self::integer_width(ty.precision);
            let op_name = match op {
                "+" => "add",
                "-" => "sub",
                "*" => "mul",
                "/" => "trunc_div",
                "%" => "mod",
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
                "/" | "%" => {
                    let zero = NumericTrap::DivZero {
                        op: op_name,
                        prim: ty.precision,
                    }
                    .to_string();
                    let expression = format!(
                        "({lhs} {op} ({et})chelis_int_checked_divisor((int64_t)({lhs}), (int64_t)({rhs}), {bits}, {zero:?}, {overflow:?}))"
                    );
                    if op == "%" {
                        // The exact remainder is zero even when MIN / -1 has
                        // no representable quotient. Do not evaluate that C %.
                        format!("(({rhs}) == -1 ? ({et})0 : {expression})")
                    } else {
                        expression
                    }
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
        // Shift errors include the offending count. Visit logical elements
        // in order so different negative counts cannot race the diagnostic.
        let serial_shift = matches!(op, "shl" | "shr");
        let pragma = if serial_shift {
            None
        } else if checked_int {
            Some("#pragma omp parallel for")
        } else {
            Some("#pragma omp parallel for simd")
        };
        self.open_element_loop(id, "i", &format!("t{id}_size"), pragma);
        let contiguous = elem_expr(
            self.gated(format!("__in_a_{id}[i]"), 0),
            self.gated(format!("__in_b_{id}[i]"), 1),
        );
        self.line(&format!("__out_{id}[i] = {contiguous};"));
        self.close_element_loop();
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.open_element_loop(
            id,
            "i",
            &format!("t{id}_size"),
            (!serial_shift).then_some("#pragma omp parallel for"),
        );
        self.line(&format!("int64_t idx_a = i * t{id}_input{a}_step;"));
        self.line(&format!("int64_t idx_b = i * t{id}_input{b}_step;"));
        let strided = elem_expr(
            self.gated(format!("(({et}*)t{a}_data)[idx_a]"), 0),
            self.gated(format!("(({et}*)t{b}_data)[idx_b]"), 1),
        );
        self.line(&format!("(({et}*)t{id}_data)[i] = {strided};"));
        self.close_element_loop();
        self.indent -= 1;
        self.line("}");
    }

    /// chelis#178: floor division (round quotient toward -inf).
    ///
    /// - Integer dtype: `chelis_int_checked_floor_div`, which checks the
    ///   divisor before any C `/` or `%` and then applies the remainder-sign
    ///   correction. A zero divisor and MIN / -1 raise the same [04-NUM-9]
    ///   traps the evaluator raises, rather than C undefined behavior.
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
        let nan = self.nan_finalization;
        let bits = if is_int {
            Self::integer_width(ty.precision)
        } else {
            0
        };
        let zero = NumericTrap::DivZero {
            op: "floor_div",
            prim: ty.precision,
        }
        .to_string();
        let overflow = NumericTrap::Overflow {
            op: "floor_div",
            prim: ty.precision,
        }
        .to_string();
        let elem_expr = |av: &str, bv: &str| -> String {
            if is_int {
                format!(
                    "({et})chelis_int_checked_floor_div((int64_t)({av}), (int64_t)({bv}), {bits}, {zero:?}, {overflow:?})"
                )
            } else {
                finalize_elem(nan, format!("{floor_fn}(({et})({av}) / ({et})({bv}))"), ty)
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
        let pragma = if is_int {
            "#pragma omp parallel for"
        } else {
            "#pragma omp parallel for simd"
        };
        self.open_element_loop(id, "i", &format!("t{id}_size"), Some(pragma));
        let contiguous = elem_expr(
            &self.gated(format!("__in_a_{id}[i]"), 0),
            &self.gated(format!("__in_b_{id}[i]"), 1),
        );
        self.line(&format!("__out_{id}[i] = {contiguous};"));
        self.close_element_loop();
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.open_element_loop(
            id,
            "i",
            &format!("t{id}_size"),
            Some("#pragma omp parallel for"),
        );
        self.line(&format!("int64_t idx_a = i * t{id}_input{a}_step;"));
        self.line(&format!("int64_t idx_b = i * t{id}_input{b}_step;"));
        let strided = elem_expr(
            &self.gated(format!("(({et}*)t{a}_data)[idx_a]"), 0),
            &self.gated(format!("(({et}*)t{b}_data)[idx_b]"), 1),
        );
        self.line(&format!("(({et}*)t{id}_data)[i] = {strided};"));
        self.close_element_loop();
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
                // The narrowing store finalizes any NaN to the dtype's
                // canonical quiet NaN ([04-NUM-2]). Float `mod` is the exact
                // `fmodf` at f32, so the one narrowing is exact (chelis#626).
                if op == "%" {
                    format!("{store}(fmodf(__av, __bv))")
                } else {
                    format!("{store}(__av {op} __bv)")
                }
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

    // ---- Typed comparisons ----
    /// Comparable-scalar load expression for one comparison operand. The
    /// operand carries its own precision `p`:
    /// `∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D,bool]`), so we read
    /// it through a `p`-typed pointer (`ptr_var` must already be cast to
    /// the storage element type). Reduced-float operands (`bf16`/`f16`)
    /// store as `uint16_t` and must convert to `f32` before the
    /// numeric `<`, matching the evaluator's value comparison rather
    /// than a 16-bit bit-pattern comparison.
    fn comparison_value(ty: &TensorType, ptr_var: &str, idx: &str) -> String {
        if Self::is_reduced_float(ty) {
            let conv = Self::reduced_to_f32_fn(ty.precision);
            format!("{conv}({ptr_var}[{idx}])")
        } else {
            format!("{ptr_var}[{idx}]")
        }
    }

    fn comparison_operator(kind: ComparisonKind) -> &'static str {
        match kind {
            ComparisonKind::CmpLt | ComparisonKind::Lt => "<",
            ComparisonKind::Eq => "==",
            ComparisonKind::Neq => "!=",
            ComparisonKind::Gt => ">",
            ComparisonKind::Gte => ">=",
            ComparisonKind::Lte => "<=",
        }
    }

    /// #517/#630/#666: comparisons read each operand through its OWN element dtype,
    /// resolved from the input DAG nodes — not through the boolean
    /// output type. The result uses the canonical one-byte Bool8
    /// representation,
    /// but a runtime-produced i32/i64/f64/bf16/f16 operand read
    /// through a raw `float*` would reinterpret its bit pattern (the
    /// #347/#476 bug class: e.g. a negative i32 reinterpreted as
    /// `float` is NaN, so `-7 < -3` would wrongly yield false, and an
    /// f64 read through `float*` truncates the 8-byte payload). Both
    /// operands share precision `p` per the signature, but each type is
    /// resolved independently for robustness.
    fn emit_compare(
        &mut self,
        id: usize,
        kind: ComparisonKind,
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
        let cmp_a = Self::comparison_value(&a_ty, &format!("__in_a_{id}"), "i");
        let cmp_b = Self::comparison_value(&b_ty, &format!("__in_b_{id}"), "i");
        let operator = Self::comparison_operator(kind);
        self.line(&format!(
            "if (chelis_is_contiguous(t{a}) && ({identity}) && chelis_is_contiguous(t{b}) && t{a}_size == t{id}_size && t{b}_size == t{id}_size) {{"
        ));
        self.indent += 1;
        self.line(&format!("assert(t{a}_size == t{id}_size);"));
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "__out_{id}[i] = ({cmp_a} {operator} {cmp_b}) ? UINT8_C(1) : UINT8_C(0);"
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
        let cmp_a_strided = Self::comparison_value(&a_ty, &format!("__in_a_{id}"), "idx_a");
        let cmp_b_strided = Self::comparison_value(&b_ty, &format!("__in_b_{id}"), "idx_b");
        self.line(&format!(
            "__out_{id}[i] = ({cmp_a_strided} {operator} {cmp_b_strided}) ? UINT8_C(1) : UINT8_C(0);"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- Typed logical operations ----
    fn emit_logical(&mut self, id: usize, kind: LogicalKind, inputs: &[NodeId], ty: &TensorType) {
        let left = inputs[0].0;
        let right = inputs.get(1).map(|input| input.0);
        let identity = self.emit_elementwise_index_steps(id, inputs, ty);
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "uint8_t* restrict __out_{id} = (uint8_t*)t{id}_data;"
        ));
        self.line(&format!(
            "const uint8_t* restrict __logical_left_{id} = (const uint8_t*)t{left}_data;"
        ));
        if let Some(right) = right {
            self.line(&format!(
                "const uint8_t* restrict __logical_right_{id} = (const uint8_t*)t{right}_data;"
            ));
        }
        let contiguous = if let Some(right) = right {
            format!(
                "chelis_is_contiguous(t{left}) && chelis_is_contiguous(t{right}) && \
                 ({identity}) && t{left}_size == t{id}_size && t{right}_size == t{id}_size"
            )
        } else {
            format!("chelis_is_contiguous(t{left}) && ({identity}) && t{left}_size == t{id}_size")
        };
        let expression = |left_index: &str, right_index: Option<&str>| match kind {
            LogicalKind::And => format!(
                "(__logical_left_{id}[{left_index}] != UINT8_C(0) && \
                 __logical_right_{id}[{}] != UINT8_C(0))",
                right_index.expect("and has a right input")
            ),
            LogicalKind::Or => format!(
                "(__logical_left_{id}[{left_index}] != UINT8_C(0) || \
                 __logical_right_{id}[{}] != UINT8_C(0))",
                right_index.expect("or has a right input")
            ),
            LogicalKind::Not => {
                format!("(__logical_left_{id}[{left_index}] == UINT8_C(0))")
            }
        };
        self.line(&format!("if ({contiguous}) {{"));
        self.indent += 1;
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "__out_{id}[i] = {} ? UINT8_C(1) : UINT8_C(0);",
            expression("i", right.map(|_| "i"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!("int64_t idx_left = i * t{id}_input{left}_step;"));
        if let Some(right) = right {
            self.line(&format!("int64_t idx_right = i * t{id}_input{right}_step;"));
        }
        self.line(&format!(
            "__out_{id}[i] = {} ? UINT8_C(1) : UINT8_C(0);",
            expression("idx_left", right.map(|_| "idx_right"))
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- Stored-bit conditional selection ----
    /// chelis#1464 / [05-OP-68]: emit the guard as a real abort, not a
    /// selected value. The check runs BEFORE the fallback is carried, so a
    /// taken abort never produces a result — which is what makes the
    /// compiled lane agree with the evaluator instead of both quietly
    /// returning a placeholder.
    ///
    /// The message is a compile-time part of the node's identity, so it is
    /// emitted as a byte literal through the same fixed-width octal escaper
    /// the host lane uses. That escaper is immune to the `\u{...}` class of
    /// defect because it never reproduces a source escape spelling.
    fn emit_guarded_fail(
        &mut self,
        id: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        message: &str,
        trap_on_true: bool,
    ) {
        let condition = inputs[0].0;
        let fires = if trap_on_true { "!=" } else { "==" };
        // A batched condition aborts when ANY mapped element fires; an
        // unbatched condition has exactly one element, so the same loop
        // serves both without a rank special case. Deliberately NOT an
        // OpenMP parallel loop: the first firing element must win
        // deterministically. Under an activation an element of an inactive
        // row reads the value that does not fire (spec/10 section 3.2).
        self.open_element_loop(id, "i", &format!("t{condition}_size"), None);
        let element = self.gated(format!("((const uint8_t*)t{condition}_data)[i]"), 0);
        self.line(&format!("if ({element} {fires} UINT8_C(0)) {{"));
        self.indent += 1;
        self.line(&format!(
            "chelis_fail(chelis_string_from_utf8((const uint8_t *){}, INT64_C({})));",
            crate::host_emit::c_utf8_byte_literal(message),
            message.len()
        ));
        self.indent -= 1;
        self.line("}");
        self.close_element_loop();
        // The guard did not fire, so the result is the fallback unchanged.
        self.emit_realize(id, &inputs[1..], ty);
    }

    /// [05-OP-53]: the condition's shape equals the shape of every branch it
    /// selects. A branch selected nowhere is neither read nor shape-checked,
    /// so a condition selecting one branch everywhere yields that branch; the
    /// condition itself is always checked, the evaluator's
    /// `where_elementwise` rule.
    fn emit_where(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
        let condition = inputs[0].0;
        let then_value = inputs[1].0;
        let else_value = inputs[2].0;
        let element_size = Self::elem_type(ty);
        let then_read = format!("__where_then_read_{id}");
        let else_read = format!("__where_else_read_{id}");
        let mixed = format!("__where_mixed_{id}");
        self.line(&format!("int {then_read} = 0;"));
        self.line(&format!("int {else_read} = 0;"));
        self.line(&format!(
            "for (int64_t i = 0; i < t{condition}_size && !({then_read} && {else_read}); i++) {{"
        ));
        self.indent += 1;
        self.line(&format!(
            "if (((const uint8_t*)t{condition}_data)[i] != UINT8_C(0)) {then_read} = 1; else {else_read} = 1;"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("const int {mixed} = {then_read} && {else_read};"));
        // Each distinct operand's step is taken, and its shape checked, only
        // under the roles that read it; the condition's role always does.
        let mut reads: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        for (node, read) in [
            (condition, "1".to_string()),
            (then_value, then_read.clone()),
            (else_value, else_read.clone()),
        ] {
            reads.entry(node).or_default().push(read);
        }
        let shape = Self::tagged_shape_literal(ty);
        let rank = Self::ndim(ty);
        let mut identity = Vec::new();
        let mut contiguity = Vec::new();
        for (node, roles) in &reads {
            let read = format!("({})", roles.join(" || "));
            self.line(&format!(
                "const int64_t t{id}_input{node}_step = {read} ? chelis_tensor_elementwise_index_step_for_shape(t{node}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {shape}) : 0;"
            ));
            identity.push(format!("(!{read} || t{id}_input{node}_step == 1)"));
            contiguity.push(format!(
                "(!{read} || (chelis_is_contiguous(t{node}) && t{node}_size == t{id}_size))"
            ));
        }
        let identity = format!("t{id}_size <= 1 || ({})", identity.join(" && "));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "uint8_t* restrict __where_out_{id} = (uint8_t*)t{id}_data;"
        ));
        self.line(&format!(
            "const uint8_t* restrict __where_condition_{id} = (const uint8_t*)t{condition}_data;"
        ));
        self.line(&format!(
            "const uint8_t* restrict __where_then_{id} = (const uint8_t*)t{then_value}_data;"
        ));
        self.line(&format!(
            "const uint8_t* restrict __where_else_{id} = (const uint8_t*)t{else_value}_data;"
        ));
        self.line(&format!(
            "const size_t __where_width_{id} = sizeof({element_size});"
        ));
        let contiguity_cond = contiguity.join(" && ");
        self.line(&format!("if (({contiguity_cond}) && ({identity})) {{"));
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "const uint8_t* selected = ({mixed} ? __where_condition_{id}[i] != UINT8_C(0) : {then_read}) \
             ? __where_then_{id} : __where_else_{id};"
        ));
        self.line(&format!(
            "memcpy(__where_out_{id} + (size_t)i * __where_width_{id}, \
             selected + (size_t)i * __where_width_{id}, __where_width_{id});"
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
            "int64_t idx_condition = i * t{id}_input{condition}_step;"
        ));
        self.line(&format!(
            "int64_t idx_then = i * t{id}_input{then_value}_step;"
        ));
        self.line(&format!(
            "int64_t idx_else = i * t{id}_input{else_value}_step;"
        ));
        self.line(&format!(
            "const uint8_t* selected = ({mixed} ? __where_condition_{id}[idx_condition] != UINT8_C(0) : {then_read}) \
             ? __where_then_{id} + (size_t)idx_then * __where_width_{id} \
             : __where_else_{id} + (size_t)idx_else * __where_width_{id};"
        ));
        self.line(&format!(
            "memcpy(__where_out_{id} + (size_t)i * __where_width_{id}, selected, __where_width_{id});"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- Unary elementwise ----
    fn emit_unary(&mut self, id: usize, op: UnaryEmission, inputs: &[NodeId], ty: &TensorType) {
        if Self::is_reduced_float(ty) {
            self.emit_unary_reduced_f(id, op, inputs, ty);
            return;
        }
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        let nan = self.nan_finalization;
        let elem_expr = |value: String| -> String {
            if matches!(op, UnaryEmission::Neg) && ty.precision.is_integer() {
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
                finalize_elem(nan, op.expression(&value), ty)
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
        let pragma = if ty.precision.is_integer() && matches!(op, UnaryEmission::Neg) {
            "#pragma omp parallel for"
        } else {
            "#pragma omp parallel for simd"
        };
        self.open_element_loop(id, "i", &format!("t{id}_size"), Some(pragma));
        let contiguous = elem_expr(self.gated(format!("__in_a_{id}[i]"), 0));
        self.line(&format!("__out_{id}[i] = {contiguous};"));
        self.close_element_loop();
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.open_element_loop(
            id,
            "i",
            &format!("t{id}_size"),
            Some("#pragma omp parallel for"),
        );
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
        let strided = elem_expr(self.gated(format!("(({et}*)t{a}_data)[idx]"), 0));
        self.line(&format!("(({et}*)t{id}_data)[i] = {strided};"));
        self.close_element_loop();
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
        self.open_element_loop(
            id,
            "i",
            &format!("t{id}_size"),
            Some("#pragma omp parallel for"),
        );
        let contiguous =
            Self::integer_abs_expr(ty.precision, &self.gated(format!("__in_a_{id}[i]"), 0));
        self.line(&format!("__out_{id}[i] = {contiguous};"));
        self.close_element_loop();
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.open_element_loop(
            id,
            "i",
            &format!("t{id}_size"),
            Some("#pragma omp parallel for"),
        );
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
        let strided =
            Self::integer_abs_expr(ty.precision, &self.gated(format!("__in_a_{id}[idx]"), 0));
        self.line(&format!("__out_{id}[i] = {strided};"));
        self.close_element_loop();
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
        let recip = finalize_elem(self.nan_finalization, format!("{one} / __in_a_{id}[i]"), ty);
        self.line(&format!("__out_{id}[i] = {recip};"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("#pragma omp parallel for");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
        let recip = finalize_elem(
            self.nan_finalization,
            format!("{one} / __in_a_{id}[idx]"),
            ty,
        );
        self.line(&format!("__out_{id}[i] = {recip};"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- Unary func (chelis_cr_expf, ..., sqrtf) ----
    fn emit_unary_func(&mut self, id: usize, func: &str, inputs: &[NodeId], ty: &TensorType) {
        assert!(
            !Self::is_host_math_transcendental(func),
            "`{func}` is a host math library transcendental; [05-OP-46] requires the \
             correctly rounded `chelis_cr_*` kernel (chelis#2957)"
        );
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
        let nan = self.nan_finalization;
        let elem_expr = |value: String| -> String {
            let raw = if is_relu {
                // [05-OP-43] is selection, not fmax: retain the input's exact
                // stored bits for NaN and -0 and replace only x < +0.
                format!("({value}) < {zero} ? {zero} : ({value})")
            } else {
                format!("{f}({value})")
            };
            finalize_elem(nan, raw, ty)
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
    fn emit_unary_reduced_f(
        &mut self,
        id: usize,
        op: UnaryEmission,
        inputs: &[NodeId],
        ty: &TensorType,
    ) {
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
        self.line(&format!(
            "__out_{id}[i] = {store}({});",
            op.expression("__av")
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
            "((uint16_t*)t{id}_data)[i] = {store}({});",
            op.expression("__av")
        ));
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

    /// A rank-0 float input at its exact arithmetic reading: f16 and bf16
    /// widen to `float`, f32 is `float`, f64 is `double`.
    fn rank0_float_expr(dag: VerifiedDagView<'_>, input: NodeId) -> String {
        let ty = &dag.get(input).expect("verified random control").output_type;
        let prim = ty.precision;
        let widen = match prim {
            Prim::F64 | Prim::F32 => None,
            Prim::F16 | Prim::Bf16 => Some(Self::reduced_to_f32_fn(prim)),
            other => panic!(
                "random control of dtype `{}` is not f16, bf16, f32 or f64",
                other.name()
            ),
        };
        let stored = format!("((const {}*)t{}_data)[0]", Self::elem_type(ty), input.0);
        match widen {
            Some(widen) => format!("{widen}({stored})"),
            None => stored,
        }
    }

    /// A random node's own activation ([`chelis_ir::dag::Owner::activation`])
    /// when rank 0, read as its stored Bool byte. No activation is active.
    fn rank0_bool_expr(input: Option<&NodeId>) -> String {
        match input {
            Some(input) => format!(
                "(((const {}*)t{}_data)[0] != 0)",
                Self::prim_elem_type(Prim::Bool),
                input.0
            ),
            None => "1".to_string(),
        }
    }

    /// [05-OP-37] over an operand rate and a key, and its pathwise replay
    /// over a cotangent: drop where the arithmetic-width unit is below the
    /// rate, else the finalized `div(x, sub(1p, rate))`.
    fn emit_keyed_dropout(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) {
        self.emit_draw_extent_guards(node, dag);
        self.emit_explicitly_keyed_dropout(node, dag);
    }

    /// [05-OP-8] over operand bounds and a key, with the same samplers the
    /// baked node uses.
    fn emit_keyed_uniform_like(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) {
        self.emit_draw_extent_guards(node, dag);
        self.emit_explicitly_keyed_uniform_like(node, dag);
    }

    /// [05-OP-8]'s bound adjoint: contributions `g_i * (1 - u_i)` or
    /// `g_i * u_i` in row-major order at the arithmetic width, combined by
    /// the canonical adjacent-pair tree and narrowed once to `p`.
    fn emit_uniform_bound_adjoint(
        &mut self,
        node: &DagNode,
        bound: chelis_ir::dag::UniformBound,
        dag: VerifiedDagView<'_>,
    ) {
        self.emit_draw_extent_guards(node, dag);
        self.emit_explicitly_keyed_bound_adjoint(node, bound, dag);
    }

    // ---- Explicit keys ([05-RNG-2], [05-OP-69..72]) ----

    /// Element `row` of a key tensor, or of a rank-0 key at row 0.
    fn key_word_expr(key: NodeId, row: &str) -> String {
        format!(
            "((const {}*)t{}_data)[{row}]",
            Self::prim_elem_type(Prim::Key),
            key.0
        )
    }

    /// The element row `b` of draw `id` reads from an operand shaped like
    /// its key's leading axes: `b / (rows / size)`, which is `b` itself for
    /// an operand of the key's own shape. A row exists only for a nonempty
    /// key batch, whose leading part is nonempty too.
    fn leading_row_expr(id: usize, input: NodeId) -> String {
        format!("b / (t{id}_rows / t{}_size)", input.0)
    }

    /// Draw `id`'s control at row `b`: its one element when rank 0, or the
    /// row's element of a control shaped like the key's leading axes, at its
    /// exact arithmetic reading.
    fn row_float_expr(dag: VerifiedDagView<'_>, input: NodeId, id: usize) -> String {
        let ty = &dag.get(input).expect("verified random control").output_type;
        if ty.dims.is_empty() {
            return Self::rank0_float_expr(dag, input);
        }
        let row = Self::leading_row_expr(id, input);
        let stored = format!("((const {}*)t{}_data)[{row}]", Self::elem_type(ty), input.0);
        match ty.precision {
            Prim::F64 | Prim::F32 => stored,
            prim @ (Prim::F16 | Prim::Bf16) => {
                format!("{}({stored})", Self::reduced_to_f32_fn(prim))
            }
            other => panic!(
                "random control of dtype `{}` is not f16, bf16, f32 or f64",
                other.name()
            ),
        }
    }

    /// Draw `id`'s activation at row `b`: absent is active, a rank-0 Bool is
    /// its byte, and a Bool shaped like the key's leading axes is the row's
    /// byte.
    fn row_bool_expr(dag: VerifiedDagView<'_>, input: Option<&NodeId>, id: usize) -> String {
        match input {
            Some(input)
                if !dag
                    .get(*input)
                    .expect("verified activation")
                    .output_type
                    .dims
                    .is_empty() =>
            {
                format!(
                    "(((const {}*)t{}_data)[{}] != 0)",
                    Self::prim_elem_type(Prim::Bool),
                    input.0,
                    Self::leading_row_expr(id, *input)
                )
            }
            other => Self::rank0_bool_expr(other),
        }
    }

    /// Check each extent `dims` declares against `observed(axis)`, a C
    /// expression, before the result exists: a mismatch reports the claim
    /// in the DAG evaluator's `check_declared_extents` form and traps
    /// `Domain` in `op` at i64.
    fn emit_declared_extent_guards(
        &mut self,
        op: &'static str,
        dims: &[DimInfo],
        observed: impl Fn(usize) -> String,
    ) {
        let trap = NumericTrap::Domain {
            op,
            prim: Prim::Int64,
        }
        .to_string();
        for (axis, dim) in dims.iter().enumerate() {
            let observed = observed(axis);
            let claimed = Self::emit_dim_info(dim);
            let claim = Self::extent_claim_label(dim);
            self.line(&format!("if (({claimed}) != ({observed})) {{"));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"extent `{claim}`: claimed = %lld, {op} axis {axis} = %lld\\n\", (long long)({claimed}), (long long)({observed}));"
            ));
            self.line(&format!("chelis_numeric_trap({trap:?});"));
            self.indent -= 1;
            self.line("}");
        }
    }

    /// How an extent report names a declared axis: its literal, or its name
    /// made safe for a format string.
    fn extent_claim_label(dim: &DimInfo) -> String {
        match dim {
            DimInfo::Lit(value) => value.to_string(),
            DimInfo::Named(name, _) => {
                chelis_ir::span_sanitize::sanitize_for_format_string(name).into_owned()
            }
        }
    }

    /// Check `input`, which the report calls `what` (its input slot, `input
    /// 1`, or the node's `activation`), against `reference`, whose declared
    /// axes are `reference_dims`, on its first `axes` runtime extents: a
    /// mismatch reports the reference's claim in the DAG evaluator's
    /// `check_operand_extents` form and traps `Domain` in `op` at i64.
    fn emit_operand_extent_guards(
        &mut self,
        op: &'static str,
        reference: NodeId,
        reference_dims: &[DimInfo],
        what: &str,
        input: NodeId,
        axes: usize,
    ) {
        let trap = NumericTrap::Domain {
            op,
            prim: Prim::Int64,
        }
        .to_string();
        for (axis, dim) in reference_dims.iter().enumerate().take(axes) {
            let claim = Self::extent_claim_label(dim);
            let claimed = format!("chelis_tensor_shape(t{}, {axis})", reference.0);
            let observed = format!("chelis_tensor_shape(t{}, {axis})", input.0);
            self.line(&format!("if (({claimed}) != ({observed})) {{"));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"extent `{claim}`: claimed = %lld, {op} {what} axis {axis} = %lld\\n\", (long long)({claimed}), (long long)({observed}));"
            ));
            self.line(&format!("chelis_numeric_trap({trap:?});"));
            self.indent -= 1;
            self.line("}");
        }
    }

    /// A key-operand random primitive's extents, checked before it
    /// allocates its result or reads an operand. First its key batch against
    /// its operands, in [`RiscOp::draw_batch_layout`]'s order and with the
    /// DAG evaluator's report (spec/10 §3.2, rule V5): a key batch's shape is
    /// its data's leading axes, and each per-row control, then the node's own
    /// activation, is a leading part of it. Then every extent the result's type declares,
    /// from which this lane allocates the result, against the data's (a bound
    /// adjoint's, against the key's leading axes), with the local extent
    /// guard's report; the evaluator builds each result from its data and
    /// reads no declared extent. So no row index, row length or loop bound
    /// below rests on an extent that nothing has checked against the
    /// operands.
    fn emit_draw_extent_guards(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) {
        let layout = node
            .op
            .draw_batch_layout()
            .expect("emit_draw_extent_guards emits only key-operand random primitives");
        let op = layout.op;
        let key = node.inputs[layout.key];
        let data = node.inputs[layout.data_input];
        let key_dims = &dag.get(key).expect("verified key").output_type.dims;
        let rank = |input: NodeId| {
            dag.get(input)
                .expect("verified random operand")
                .output_type
                .dims
                .len()
        };
        if !key_dims.is_empty() {
            let data_input = format!("input {}", layout.data_input);
            let operands =
                std::iter::once((data_input, data, key_dims.len()))
                    .chain(layout.per_row.iter().filter_map(|slot| {
                        node.inputs
                            .get(*slot)
                            .map(|input| (format!("input {slot}"), *input, rank(*input)))
                    }))
                    .chain(node.owner.activation.map(|activation| {
                        ("activation".to_string(), activation, rank(activation))
                    }));
            for (what, input, axes) in operands.collect::<Vec<_>>() {
                self.emit_operand_extent_guards(op, key, key_dims, &what, input, axes);
            }
        }
        let source = if matches!(node.op, RiscOp::UniformBoundAdjoint { .. }) {
            key
        } else {
            data
        };
        self.emit_declared_extent_guards(op, &node.output_type.dims, |axis| {
            format!("chelis_tensor_shape(t{}, {axis})", source.0)
        });
    }

    /// The row count and per-row element count of draw `id`'s data `data`
    /// under key `key`: one row of every element for a rank-0 key, and for a
    /// batch one row per key, of the data's runtime element count over the
    /// key's (spec/10 §3.2). [`Self::emit_draw_extent_guards`] has checked
    /// that the data's leading axes are the key's shape, so each row is
    /// whole. Row `b` holds flat elements `[b * row_len, (b + 1) * row_len)`
    /// and numbers them from zero.
    fn emit_draw_rows(&mut self, id: usize, key: NodeId, dag: VerifiedDagView<'_>, data: NodeId) {
        let index = Self::prim_elem_type(Prim::Int64);
        let batched = !dag
            .get(key)
            .expect("verified key")
            .output_type
            .dims
            .is_empty();
        let rows = if batched {
            format!("t{}_size", key.0)
        } else {
            "1".to_string()
        };
        self.line(&format!("{index} t{id}_rows = {rows};"));
        self.line(&format!(
            "{index} t{id}_row_len = t{id}_rows > 0 ? t{}_size / t{id}_rows : 0;",
            data.0
        ));
    }

    /// The row `b` and in-row index `e` of flat element `i` of a draw.
    fn emit_row_of_element(&mut self, id: usize) {
        let index = Self::prim_elem_type(Prim::Int64);
        self.line(&format!("{index} b = i / t{id}_row_len;"));
        self.line(&format!(
            "{RANDOM_WORD_C_TYPE} e = ({RANDOM_WORD_C_TYPE})(i - b * t{id}_row_len);"
        ));
    }

    /// [05-OP-37] under an explicit key: the draw validates its own rate for
    /// every active row, then drops where the arithmetic-width unit of
    /// `word(key[b], e)` is below row `b`'s rate.
    fn emit_explicitly_keyed_dropout(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) {
        let id = node.id.0;
        let ty = &node.output_type;
        let prim = ty.precision;
        let data = node.inputs[0].0;
        let key = node.inputs[2];
        let index = Self::prim_elem_type(Prim::Int64);
        let binary64 = Self::prim_elem_type(Prim::F64);
        let binary32 = Self::prim_elem_type(Prim::F32);
        let storage = Self::elem_type(ty);
        let rate = Self::row_float_expr(dag, node.inputs[1], id);
        let active = Self::row_bool_expr(dag, node.owner.activation.as_ref(), id);
        let trap = NumericTrap::Domain {
            op: "dropout",
            prim,
        }
        .to_string();
        self.emit_slot_wrapper(id, ty);
        self.emit_draw_rows(id, key, dag, node.inputs[0]);
        self.line(&format!("for ({index} b = 0; b < t{id}_rows; b++) {{"));
        self.indent += 1;
        self.line(&format!("if ({active}) {{"));
        self.indent += 1;
        self.line(&format!("{binary64} rate = ({binary64})({rate});"));
        self.line(&format!(
            "if (!(rate >= 0.0 && rate < 1.0)) chelis_numeric_trap({trap:?});"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("#pragma omp parallel for");
        self.line(&format!("for ({index} i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.emit_row_of_element(id);
        self.line(&format!(
            "if (!({active})) {{ (({storage}*)t{id}_data)[i] = ({storage})0; continue; }}"
        ));
        let unit = format!("chelis_random_unit({}, e)", Self::key_word_expr(key, "b"));
        match prim {
            Prim::F64 => {
                self.line(&format!("{binary64} rate = {rate};"));
                self.line(&format!(
                    "(({storage}*)t{id}_data)[i] = {unit} < rate ? 0.0 : {};",
                    finalize_elem(
                        self.nan_finalization,
                        format!("((const {storage}*)t{data}_data)[i] / (1.0 - rate)"),
                        &node.output_type,
                    )
                ));
            }
            Prim::F32 => {
                self.line(&format!("{binary32} rate = {rate};"));
                self.line(&format!(
                    "(({storage}*)t{id}_data)[i] = ({binary32}){unit} < rate ? 0.0f : {};",
                    finalize_elem(
                        self.nan_finalization,
                        format!("((const {storage}*)t{data}_data)[i] / (1.0f - rate)"),
                        &node.output_type,
                    )
                ));
            }
            Prim::F16 | Prim::Bf16 => {
                let widen = Self::reduced_to_f32_fn(prim);
                let narrow = Self::f32_to_reduced_fn(prim);
                self.line(&format!("{binary32} rate = {rate};"));
                self.line(&format!(
                    "{binary32} denom = {widen}({narrow}(1.0f - rate));"
                ));
                self.line(&format!(
                    "(({storage}*)t{id}_data)[i] = ({binary32}){unit} < rate ? {narrow}(0.0f) : {narrow}({widen}(((const {storage}*)t{data}_data)[i]) / denom);"
                ));
            }
            other => panic!(
                "dropout of dtype `{}` is not f16, bf16, f32 or f64",
                other.name()
            ),
        }
        self.indent -= 1;
        self.line("}");
    }

    /// [05-OP-8] under an explicit key: the draw validates its own bounds for
    /// every active row, then samples `word(key[b], e)` with row `b`'s bounds.
    fn emit_explicitly_keyed_uniform_like(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) {
        let id = node.id.0;
        let ty = &node.output_type;
        let prim = ty.precision;
        let key = node.inputs[3];
        let index = Self::prim_elem_type(Prim::Int64);
        let binary64 = Self::prim_elem_type(Prim::F64);
        let storage = Self::elem_type(ty);
        let low = Self::row_float_expr(dag, node.inputs[1], id);
        let high = Self::row_float_expr(dag, node.inputs[2], id);
        let active = Self::row_bool_expr(dag, node.owner.activation.as_ref(), id);
        let trap = NumericTrap::Domain {
            op: "uniform_like",
            prim,
        }
        .to_string();
        self.emit_slot_wrapper(id, ty);
        self.emit_draw_rows(id, key, dag, node.inputs[0]);
        self.line(&format!("for ({index} b = 0; b < t{id}_rows; b++) {{"));
        self.indent += 1;
        self.line(&format!("if ({active}) {{"));
        self.indent += 1;
        self.line(&format!(
            "{binary64} low = ({binary64})({low}), high = ({binary64})({high});"
        ));
        if prim == Prim::F64 {
            self.line(&format!("{binary64} span = high - low;"));
        } else {
            let binary32 = Self::prim_elem_type(Prim::F32);
            self.line(&format!(
                "{binary32} span = ({binary32})high - ({binary32})low;"
            ));
        }
        self.line(&format!(
            "if (!(isfinite(low) && isfinite(high) && low <= high && isfinite(span))) chelis_numeric_trap({trap:?});"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        let (wide, sampler) = if prim == Prim::F64 {
            (binary64, "chelis_uniform_sample_f64")
        } else {
            (Self::prim_elem_type(Prim::F32), "chelis_uniform_sample_f32")
        };
        self.line("#pragma omp parallel for");
        self.line(&format!("for ({index} i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.emit_row_of_element(id);
        self.line(&format!(
            "if (!({active})) {{ (({storage}*)t{id}_data)[i] = ({storage})0; continue; }}"
        ));
        let sample = format!(
            "{sampler}({}, e, ({wide})({low}), ({wide})({high}))",
            Self::key_word_expr(key, "b")
        );
        match prim {
            Prim::F64 | Prim::F32 => {
                self.line(&format!("(({storage}*)t{id}_data)[i] = {sample};"));
            }
            Prim::F16 | Prim::Bf16 => self.line(&format!(
                "(({storage}*)t{id}_data)[i] = {}({sample});",
                Self::f32_to_reduced_fn(prim)
            )),
            other => panic!(
                "uniform_like of dtype `{}` is not f16, bf16, f32 or f64",
                other.name()
            ),
        }
        self.indent -= 1;
        self.line("}");
    }

    /// [05-OP-8]'s bound adjoint under an explicit key. Row `b`'s leaves are
    /// `g_i * (1 - u_i)` or `g_i * u_i` with `u_i` from `word(key[b], e)`, and
    /// positive zero for an inactive row. The result is shaped like the key's
    /// leading axes, and each element folds the leaves of the rows that share
    /// it in one canonical adjacent-pair tree: a rank-0 result folds every
    /// leaf, and a result of the key's shape folds each row's.
    fn emit_explicitly_keyed_bound_adjoint(
        &mut self,
        node: &DagNode,
        bound: chelis_ir::dag::UniformBound,
        dag: VerifiedDagView<'_>,
    ) {
        let id = node.id.0;
        let ty = &node.output_type;
        let prim = ty.precision;
        let cotangent = node.inputs[1].0;
        let key = node.inputs[2];
        let active = Self::row_bool_expr(dag, node.owner.activation.as_ref(), id);
        let arithmetic_ty = TensorType {
            dims: vec![],
            precision: if prim == Prim::F64 {
                Prim::F64
            } else {
                Prim::F32
            },
        };
        let arithmetic = Self::elem_type(&arithmetic_ty);
        let arithmetic_dtype = Self::dtype_macro(&arithmetic_ty);
        let index = Self::prim_elem_type(Prim::Int64);
        let storage = Self::elem_type(ty);
        let load_g = match prim {
            Prim::F64 | Prim::F32 => format!("((const {storage}*)t{cotangent}_data)[i]"),
            Prim::F16 | Prim::Bf16 => format!(
                "{}(((const {storage}*)t{cotangent}_data)[i])",
                Self::reduced_to_f32_fn(prim)
            ),
            other => panic!(
                "uniform bound adjoint of dtype `{}` is not f16, bf16, f32 or f64",
                other.name()
            ),
        };
        let weight = match bound {
            chelis_ir::dag::UniformBound::Low => format!("(({arithmetic})1 - u)"),
            chelis_ir::dag::UniformBound::High => "u".to_string(),
        };
        let nan_finalization = self.nan_finalization;
        let store = |value: &str| match prim {
            Prim::F64 | Prim::F32 => finalize_elem(nan_finalization, value.to_string(), ty),
            _ => format!("{}({value})", Self::f32_to_reduced_fn(prim)),
        };
        let per_row = !ty.dims.is_empty();
        self.emit_slot_wrapper(id, ty);
        self.emit_draw_rows(id, key, dag, node.inputs[1]);
        // One output per group of rows sharing a bound element, or one for
        // the whole draw; each group's leaves are contiguous.
        let outputs = if per_row {
            format!("t{id}_size")
        } else {
            "1".to_string()
        };
        let segment = if per_row {
            format!("(t{cotangent}_size / t{id}_size)")
        } else {
            format!("t{cotangent}_size")
        };
        self.line(&format!("{index} t{id}_n = t{cotangent}_size;"));
        self.line(&format!(
            "for ({index} out = 0; out < {outputs}; out++) (({storage}*)t{id}_data)[out] = {};",
            store(&format!("({arithmetic})0"))
        ));
        self.line(&format!("if (t{id}_n > 0) {{"));
        self.indent += 1;
        self.line(&format!(
            "chelis_tensor *t{id}_leaves_tensor = chelis_alloc(1, &t{id}_n, {arithmetic_dtype});"
        ));
        self.line(&format!(
            "chelis_tensor_write *t{id}_leaves_guard = chelis_tensor_begin_write(t{id}_leaves_tensor);"
        ));
        self.line(&format!(
            "{arithmetic} *t{id}_leaves = ({arithmetic}*)chelis_tensor_write_view(t{id}_leaves_guard).data;"
        ));
        self.line(&format!("for ({index} i = 0; i < t{id}_n; i++) {{"));
        self.indent += 1;
        self.emit_row_of_element(id);
        self.line(&format!(
            "{arithmetic} u = ({arithmetic})chelis_random_unit({}, e);",
            Self::key_word_expr(key, "b")
        ));
        self.line(&format!(
            "t{id}_leaves[i] = ({active}) ? ({arithmetic})({load_g}) * {weight} : ({arithmetic})0;"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("for ({index} out = 0; out < {outputs}; out++) {{"));
        self.indent += 1;
        self.line(&format!(
            "{arithmetic} *seg = t{id}_leaves + out * {segment};"
        ));
        self.line(&format!("{index} n = {segment};"));
        let fold =
            Self::adjacent_pair_fold_lines("seg", "n", |left, right| format!("{left} + {right}"));
        self.emit_relative_lines(fold);
        self.line(&format!(
            "if (n > 0) (({storage}*)t{id}_data)[out] = {};",
            store("seg[0]")
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_tensor_end_write(t{id}_leaves_guard);"));
        self.line(&format!("chelis_tensor_release(t{id}_leaves_tensor);"));
        self.indent -= 1;
        self.line("}");
    }

    /// An elementwise key operation's extents ([05-OP-69], [05-OP-70],
    /// [05-OP-72]), checked before it allocates its result or reads an
    /// operand. The verifier gives the result and every operand one declared
    /// shape; this lane allocates the result from it and reads each operand
    /// at every flat index of the result. So first each later operand's
    /// runtime extents against the first's, with the DAG evaluator's report,
    /// then every extent the result declares against the first operand's,
    /// with the local extent guard's report; the evaluator builds the result
    /// from its operands and reads no declared extent. Every index below the
    /// result's size is then in bounds for every operand. A join's two
    /// activations are not operands, and a key operation's own activation
    /// changes no key, so this lane never reads it here.
    fn emit_key_operation_extent_guards(
        &mut self,
        op: &'static str,
        node: &DagNode,
        dag: VerifiedDagView<'_>,
    ) {
        let first = node.inputs[0];
        let dims = &dag
            .get(first)
            .expect("verified key operand")
            .output_type
            .dims;
        let operands = match node.op {
            RiscOp::KeySelect => 2,
            _ => node.inputs.len(),
        };
        for (slot, input) in node.inputs.iter().enumerate().take(operands).skip(1) {
            self.emit_operand_extent_guards(
                op,
                first,
                dims,
                &format!("input {slot}"),
                *input,
                dims.len(),
            );
        }
        self.emit_declared_extent_guards(op, &node.output_type.dims, |axis| {
            format!("chelis_tensor_shape(t{}, {axis})", first.0)
        });
    }

    /// [05-OP-69]: each key is its i64 seed's two's-complement bits.
    fn emit_key_from_seed(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) {
        let id = node.id.0;
        let seed = node.inputs[0].0;
        let word = Self::prim_elem_type(Prim::Key);
        let seed_ty = Self::prim_elem_type(Prim::Int64);
        let index = Self::prim_elem_type(Prim::Int64);
        self.emit_key_operation_extent_guards("key_from_seed", node, dag);
        self.emit_slot_wrapper(id, &node.output_type);
        self.line(&format!("for ({index} i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "(({word}*)t{id}_data)[i] = ({word})((const {seed_ty}*)t{seed}_data)[i];"
        ));
        self.indent -= 1;
        self.line("}");
    }

    /// One half of [05-OP-70]: `derive(k, 0)` or `derive(k, 1)` per element.
    fn emit_split_key(
        &mut self,
        node: &DagNode,
        branch: chelis_ir::dag::KeyBranch,
        dag: VerifiedDagView<'_>,
    ) {
        let id = node.id.0;
        let key = node.inputs[0];
        let word = Self::prim_elem_type(Prim::Key);
        let index = Self::prim_elem_type(Prim::Int64);
        let derive_index = match branch {
            chelis_ir::dag::KeyBranch::Left => 0,
            chelis_ir::dag::KeyBranch::Right => 1,
        };
        self.emit_key_operation_extent_guards("split_key", node, dag);
        self.emit_slot_wrapper(id, &node.output_type);
        self.line(&format!("for ({index} i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "(({word}*)t{id}_data)[i] = chelis_key_derive({}, {derive_index}ULL);",
            Self::key_word_expr(key, "i")
        ));
        self.indent -= 1;
        self.line("}");
    }

    /// [05-OP-72]: `derive(derive(k, 2), n)` per element, `n`'s i64 read as
    /// its two's-complement bits.
    fn emit_fold_in(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) {
        let id = node.id.0;
        let key = node.inputs[0];
        let n = node.inputs[1].0;
        let word = Self::prim_elem_type(Prim::Key);
        let index_ty = Self::prim_elem_type(Prim::Int64);
        self.emit_key_operation_extent_guards("fold_in", node, dag);
        self.emit_slot_wrapper(id, &node.output_type);
        self.line(&format!("for ({index_ty} i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "(({word}*)t{id}_data)[i] = chelis_key_derive(chelis_key_derive({}, 2ULL), ({word})((const {index_ty}*)t{n}_data)[i]);",
            Self::key_word_expr(key, "i")
        ));
        self.indent -= 1;
        self.line("}");
    }

    /// Rule S's join (spec/10 section 3.2): element `i` is the then key's
    /// where the then activation holds for its row, and the else key's
    /// elsewhere, as the DAG evaluator's `eval_key_select` selects. An
    /// activation is rank 0 or shaped like a leading part of the keys'
    /// shape; row `i` of `n` keys reads its element `i / (n / len)`. The
    /// else key's extents are checked against the then key's first, and the
    /// declared result's against the then key's.
    fn emit_key_select(&mut self, node: &DagNode, dag: VerifiedDagView<'_>) {
        let id = node.id.0;
        let (then_key, else_key, active) = (node.inputs[0], node.inputs[1], node.inputs[2]);
        let word = Self::prim_elem_type(Prim::Key);
        let index = Self::prim_elem_type(Prim::Int64);
        let byte = Self::prim_elem_type(Prim::Bool);
        let batched = !dag
            .get(active)
            .expect("verified join activation")
            .output_type
            .dims
            .is_empty();
        let row = if batched {
            format!("i / (t{id}_size / t{}_size)", active.0)
        } else {
            "0".to_string()
        };
        self.emit_key_operation_extent_guards("if", node, dag);
        self.emit_slot_wrapper(id, &node.output_type);
        self.line(&format!("for ({index} i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        self.line(&format!(
            "(({word}*)t{id}_data)[i] = ((const {byte}*)t{}_data)[{row}] != 0 ? {} : {};",
            active.0,
            Self::key_word_expr(then_key, "i"),
            Self::key_word_expr(else_key, "i")
        ));
        self.indent -= 1;
        self.line("}");
    }

    /// [05-OP-54]: exact i64 range, with its own realized count witness.
    fn emit_list_map_capture(&mut self, node: &DagNode) {
        let id = node.id.0;
        let source = node.inputs[0].0;
        let carrier = node.inputs[1].0;
        let count = format!("chelis_tensor_shape(t{carrier}, 0)");
        self.emit_runtime_dim_sites(id, &[(0, count.clone())]);
        self.emit_declared_extent_guards("map", &node.output_type.dims, |_| count.clone());
        self.emit_slot_wrapper(id, &node.output_type);
        let et = Self::elem_type(&node.output_type);
        self.line(&format!("for (int64_t i = 0; i < {count}; i++) (({et}*)t{id}_data)[i] = ((const {et}*)t{source}_data)[0];"));
    }

    fn emit_ordered_adjoint_sum(&mut self, node: &DagNode, groups: &[usize]) {
        let id = node.id.0;
        let et = Self::elem_type(&node.output_type);
        self.emit_slot_wrapper(id, &node.output_type);
        self.line(&format!("int64_t t{id}_contributions = 1;"));
        let mut offset = 0;
        for &width in groups {
            let first = node.inputs[offset].0;
            for input in &node.inputs[offset + 1..offset + width] {
                self.line(&format!("if (t{}_size != t{first}_size) {{ fprintf(stderr, \"ordered List cotangent columns have different lengths\\n\"); abort(); }}", input.0));
            }
            self.line(&format!("t{id}_contributions = chelis_int_checked_add(t{id}_contributions, chelis_int_checked_mul(t{first}_size, {width}, 64, \"ordered List cotangent count overflow\"), 64, \"ordered List cotangent count overflow\");"));
            offset += width;
        }
        self.emit_sum_level(
            id,
            &format!("t{id}_contributions"),
            node.output_type.precision,
        );
        self.line(&format!(
            "__sum_level_{id}[0] = {};",
            Self::scalar_zero_literal(node.output_type.precision)
        ));
        self.line(&format!("int64_t t{id}_leaf = 1;"));
        offset = 0;
        for &width in groups {
            let first = node.inputs[offset].0;
            self.line(&format!(
                "for (int64_t row = 0; row < t{first}_size; row++) {{"
            ));
            self.indent += 1;
            for input in &node.inputs[offset..offset + width] {
                self.line(&format!(
                    "__sum_level_{id}[t{id}_leaf++] = ((const {et}*)t{}_data)[row];",
                    input.0
                ));
            }
            self.indent -= 1;
            self.line("}");
            offset += width;
        }
        self.line("{");
        self.indent += 1;
        self.line("const int64_t outer = 0;");
        self.emit_sum_fold(id, node.output_type.precision);
        self.indent -= 1;
        self.line("}");
    }

    fn emit_iota(&mut self, node: &DagNode) {
        let id = node.id.0;
        let start = node.inputs[0].0;
        let end = node.inputs[1].0;
        let index = Self::prim_elem_type(Prim::Int64);
        let trap = NumericTrap::Overflow {
            op: "range",
            prim: Prim::Int64,
        }
        .to_string();
        self.line(&format!(
            "{index} t{id}_start = ((const {index}*)t{start}_data)[0];"
        ));
        self.line(&format!(
            "{index} t{id}_end = ((const {index}*)t{end}_data)[0];"
        ));
        if let Some(active) = node.owner.activation {
            let active = active.0;
            let byte = Self::prim_elem_type(Prim::Bool);
            self.line(&format!(
                "if (((const {byte}*)t{active}_data)[0] == 0) {{ t{id}_start = 0; t{id}_end = 0; }}"
            ));
        }
        self.line(&format!("{index} t{id}_count = t{id}_end <= t{id}_start ? 0 : chelis_int_checked_sub(t{id}_end, t{id}_start, 64, {trap:?});"));
        self.emit_runtime_dim_sites(id, &[(0, format!("t{id}_count"))]);
        self.emit_declared_extent_guards("range", &node.output_type.dims, |_| {
            format!("t{id}_count")
        });
        self.emit_slot_wrapper(id, &node.output_type);
        self.line(&format!(
            "for ({index} i = 0; i < t{id}_count; i++) (({index}*)t{id}_data)[i] = t{id}_start + i;"
        ));
    }

    /// [05-OP-71]: row `j` of key `i` is `derive(derive(k[i], 2), j)`, the
    /// count axis last. A negative runtime count traps before allocation.
    /// Where the split's activation holds in no row, it reads no count: the
    /// count axis takes its declared extent where a literal or an earlier
    /// binding fixes it, and zero where this split declares it, as the DAG
    /// evaluator's `inactive_split_count` does, so no check below can fail.
    fn emit_split_keys(&mut self, node: &DagNode, count: &RtDim) {
        let id = node.id.0;
        let key = node.inputs[0];
        let word = Self::prim_elem_type(Prim::Key);
        let index = Self::prim_elem_type(Prim::Int64);
        let last = node.output_type.dims.len() - 1;
        let extent = Self::bound_c_expr(count, &node.inputs, key.0, last);
        let trap = NumericTrap::Domain {
            op: "split_keys",
            prim: Prim::Int64,
        }
        .to_string();
        self.line(&format!("{index} t{id}_count = ({index})({extent});"));
        if let Some(active) = node.owner.activation {
            let inactive = if self.runtime_dim_sites.contains_key(&(id, last)) {
                "0".to_string()
            } else {
                Self::emit_dim_info(&node.output_type.dims[last])
            };
            let active = active.0;
            let byte = Self::prim_elem_type(Prim::Bool);
            self.line(&format!("int t{id}_live = 0;"));
            self.line(&format!(
                "for ({index} a = 0; a < t{active}_size && !t{id}_live; a++) t{id}_live = ((const {byte}*)t{active}_data)[a] != 0;"
            ));
            self.line(&format!(
                "if (!t{id}_live) t{id}_count = ({index})({inactive});"
            ));
        }
        if matches!(count, RtDim::Node(_)) {
            self.line(&format!(
                "if (t{id}_count < 0) chelis_numeric_trap({trap:?});"
            ));
        }
        if self.runtime_dim_sites.contains_key(&(id, last)) {
            self.emit_runtime_dim_sites(id, &[(last, format!("t{id}_count"))]);
        }
        // The key's extents and the count are the result's extents. Every
        // extent the result's type declares is a claim about them, checked
        // before the allocation, as `chelis_movement_check_target` checks an
        // expansion's target and as the DAG evaluator checks this node; the
        // report is the local extent guard's.
        self.emit_declared_extent_guards("split_keys", &node.output_type.dims, |axis| {
            if axis == last {
                format!("t{id}_count")
            } else {
                format!("chelis_tensor_shape(t{}, {axis})", key.0)
            }
        });
        self.emit_slot_wrapper(id, &node.output_type);
        self.line(&format!("for ({index} i = 0; i < t{}_size; i++) {{", key.0));
        self.indent += 1;
        self.line(&format!(
            "{word} parent = chelis_key_derive({}, 2ULL);",
            Self::key_word_expr(key, "i")
        ));
        self.line(&format!("for ({index} j = 0; j < t{id}_count; j++) {{"));
        self.indent += 1;
        self.line(&format!(
            "(({word}*)t{id}_data)[i * t{id}_count + j] = chelis_key_derive(parent, ({word})j);"
        ));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    // ---- Fused elementwise helpers ----

    /// Emit one fused-step expression for the scalar fast/tail path.
    ///
    /// chelis#919: `is_f64` selects the double-precision math symbols
    /// (`chelis_cr_exp` rather than `chelis_cr_expf`, `fmax` rather than
    /// `fmaxf`) and the unsuffixed `1.0` / `0.0` literals. It must agree with
    /// the element type the caller declared for the `v{s}` step variables:
    /// emitting `chelis_cr_expf` into a `double v0` silently narrows through
    /// the single-precision kernel, which is exactly the F1
    /// footgun. `emit_fused_elem` derives both from the same
    /// `Self::is_f64(ty)`.
    ///
    /// Callers admit only f32 and f64; `emit_fused_reduce` still passes
    /// `false` because its accumulator path stays f32-only.
    /// [04-NUM-2]: every arithmetic step finalizes its NaN to the canonical
    /// quiet NaN. Extrema are selection and keep their operand's bits.
    fn scalar_step_expr(
        op: &FusedStepOp,
        resolve: &dyn Fn(&FusedInput) -> String,
        inputs: &[FusedInput],
        is_f64: bool,
    ) -> String {
        let raw = Self::scalar_step_raw_expr(op, resolve, inputs, is_f64);
        crate::fp_env::finalize_float(&raw, is_f64, crate::fp_env::fused_step_nan_finalization(op))
    }

    fn scalar_step_raw_expr(
        op: &FusedStepOp,
        resolve: &dyn Fn(&FusedInput) -> String,
        inputs: &[FusedInput],
        is_f64: bool,
    ) -> String {
        // Map a single-precision math symbol to its double-precision
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
        match op {
            FusedStepOp::Add => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("{a} + {b}")
            }
            FusedStepOp::Sub => {
                let a = resolve(&inputs[0]);
                let b = resolve(&inputs[1]);
                format!("({a}) - ({b})")
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
                let f = mf("chelis_cr_expf");
                format!("{f}({a})")
            }
            FusedStepOp::Log => {
                let a = resolve(&inputs[0]);
                let f = mf("chelis_cr_logf");
                format!("{f}({a})")
            }
            FusedStepOp::Sin => {
                let a = resolve(&inputs[0]);
                let f = mf("chelis_cr_sinf");
                format!("{f}({a})")
            }
            FusedStepOp::Sqrt => {
                let a = resolve(&inputs[0]);
                let f = mf("sqrtf");
                format!("{f}({a})")
            }
            FusedStepOp::Cos => {
                let a = resolve(&inputs[0]);
                let f = mf("chelis_cr_cosf");
                format!("{f}({a})")
            }
            FusedStepOp::Tan => {
                let a = resolve(&inputs[0]);
                let f = mf("chelis_cr_tanf");
                format!("{f}({a})")
            }
            FusedStepOp::Atan => {
                let a = resolve(&inputs[0]);
                let f = mf("chelis_cr_atanf");
                format!("{f}({a})")
            }
            FusedStepOp::Tanh => {
                let a = resolve(&inputs[0]);
                let f = mf("chelis_cr_tanhf");
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
                Prim::Key => chelis_types::deliberate_rejection!(
                    "[05-RNG-1]",
                    "a random key has no arithmetic and never joins a fused chain"
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
        // `ty`, so a `double v0` can never be fed by a `chelis_cr_expf`.
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

        // Closures for resolving fused inputs in scalar (fast-path) context.
        let resolve_fast = |fi: &FusedInput| -> String {
            match fi {
                FusedInput::External(i) => format!("__in_ext{i}"),
                FusedInput::PreviousStep(j) => format!("v{j}"),
            }
        };

        let last = ops.len() - 1;

        // --- Level-1 scalar OMP SIMD loop (default fast path) ---
        self.line("#pragma omp parallel for simd");
        self.line(&format!("for (int64_t i = 0; i < t{id}_size; i++) {{"));
        self.indent += 1;
        for (ext_idx, _) in inputs.iter().enumerate() {
            self.line(&format!("{et} __in_ext{ext_idx} = __ext{ext_idx}_{id}[i];"));
        }
        for (s, step) in ops.iter().enumerate() {
            let expr = Self::scalar_step_expr(&step.op, &resolve_fast, &step.input_indices, is_f64);
            self.line(&format!("{et} v{s} = {expr};"));
        }
        self.line(&format!("__out_{id}[i] = v{last};"));
        self.indent -= 1;
        self.line("}");

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

    /// Finalize, in place, every element of node `id`'s f32 or f64 result that a
    /// vendor routine (BLAS) wrote directly, so its NaNs are canonical like every
    /// other emitter's ([04-NUM-2]). The result is the node's own contiguous
    /// allocation of `t{id}_size` elements.
    fn emit_finalize_written_result(&mut self, id: usize, ty: &TensorType) {
        let element = Self::elem_type(ty);
        let index = Self::prim_elem_type(Prim::Int64);
        let value = format!("(({element}*)t{id}_data)[t{id}_nan_i]");
        let finalized = finalize_elem(self.nan_finalization, value.clone(), ty);
        self.line(&format!(
            "for ({index} t{id}_nan_i = 0; t{id}_nan_i < t{id}_size; ++t{id}_nan_i) {value} = {finalized};"
        ));
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
        self.emit_finalize_written_result(id, ty);
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
        // An f32 result was written by `cblas_sgemm` directly; a reduced one
        // narrowed through its canonicalizing storage helper above.
        if !output_reduced {
            self.emit_finalize_written_result(id, ty);
        }
        self.line(&format!("chelis_matmul_plan_release(t{id}_matmul);"));
    }

    /// Validate the whole iteration shape before storage submission; each loop
    /// position then obtains both its index slot and base offset from that plan.
    fn emit_sparse_checked(
        &mut self,
        id: usize,
        axes: SparseAxes,
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
        self.line(&format!("chelis_sparse_plan *t{id}_sparse = chelis_tensor_sparse_plan(t{base}, t{indices}, {updates_arg}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {}), {operation} + 4 * {});", axes.axis, axes.batch_rank));
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
        let add = matches!(update, Some("+="));
        let tree = format!("t{id}_scatter");
        if add {
            let lines = Self::scatter_add_tree_open_lines(
                &tree,
                &format!("t{id}_sparse_count"),
                &format!("t{id}_size"),
                ty.precision,
            );
            self.emit_relative_lines(lines);
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
        if add {
            self.line(&Self::scatter_add_tree_record_line(
                &tree,
                &format!("t{id}_i"),
                &format!("t{id}_base_index"),
            ));
        } else if let (Some(updates), Some(update)) = (updates, update) {
            self.line(&format!("(({element}*)t{id}_data)[t{id}_base_index] {update} ((const {element}*)t{updates}_data)[t{id}_i];"));
        } else {
            self.line(&format!("(({element}*)t{id}_data)[t{id}_i] = ((const {element}*)t{base}_data)[t{id}_base_index];"));
        }
        self.indent -= 1;
        self.line("}");
        if let (true, Some(updates)) = (add, updates) {
            let lines = Self::scatter_add_tree_close_lines(
                &tree,
                &format!("t{id}_sparse_count"),
                &format!("t{id}_data"),
                &format!("t{id}_size"),
                &format!("t{updates}_data"),
                ty.precision,
                self.nan_finalization,
            );
            self.emit_relative_lines(lines);
        }
        self.line(&format!("chelis_sparse_plan_release(t{id}_sparse);"));
    }

    fn lowered_mean_sum_source(
        dag: VerifiedDagView<'_>,
        result: NodeId,
    ) -> Option<(usize, NodeId)> {
        let mut node = dag.get(result)?;
        if matches!(node.op, RiscOp::Cast { .. }) {
            node = dag.get(*node.inputs.first()?)?;
        }
        let RiscOp::Sum { axis, .. } = node.op else {
            return None;
        };
        Some((axis, *node.inputs.first()?))
    }

    /// Recognize exactly the Tier-2 [05-OP-11] mean graph. Shape-dependency
    /// edges distinguish it from an authored `div(sum(x), y)`.
    fn is_lowered_mean_div(dag: VerifiedDagView<'_>, node: &DagNode) -> bool {
        if !matches!(node.op, RiscOp::Div) || node.inputs.len() != 2 {
            return false;
        }
        let numerator = node.inputs[0];
        let Some((axis, source)) = Self::lowered_mean_sum_source(dag, numerator) else {
            return false;
        };
        let Some(divisor) = dag.get(node.inputs[1]) else {
            return false;
        };
        if matches!(divisor.op, RiscOp::Const { .. }) && divisor.shape_deps.contains(&numerator) {
            return true;
        }
        let Some((divisor_axis, divisor_source)) = Self::lowered_mean_sum_source(dag, divisor.id)
        else {
            return false;
        };
        divisor_axis == axis
            && dag.get(divisor_source).is_some_and(|ones| {
                matches!(ones.op, RiscOp::Const { .. }) && ones.shape_deps.contains(&source)
            })
    }

    /// Runtime-derived zero extents must trap as `mean`, before the separate
    /// division operation can observe a zero divisor and manufacture NaN.
    fn emit_mean_nonempty_guard(
        &mut self,
        id: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
    ) {
        let divisor = inputs[1].0;
        let divisor_ty = &dag
            .get(inputs[1])
            .expect("lowered mean divisor exists")
            .output_type;
        let storage_et = Self::elem_type(divisor_ty);
        let value = if matches!(divisor_ty.precision, Prim::F16 | Prim::Bf16) {
            format!(
                "{}(((const {storage_et}*)t{divisor}_data)[mean_i])",
                Self::reduced_to_f32_fn(divisor_ty.precision)
            )
        } else {
            format!("((const {storage_et}*)t{divisor}_data)[mean_i]")
        };
        let trap = NumericTrap::Domain {
            op: "mean",
            prim: ty.precision,
        }
        .to_string();
        self.line(&format!(
            "for (int64_t mean_i = 0; mean_i < t{divisor}_size; mean_i++) {{"
        ));
        self.indent += 1;
        let checked = match &self.gate {
            Some(gate) => format!("{} && {value} == 0", gate.any),
            None => format!("{value} == 0"),
        };
        self.line(&format!("if ({checked}) chelis_numeric_trap({trap:?});"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("/* mean nonempty guard for node {id} */"));
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
    /// an adjacent-pair balanced checked-i64 tree. No cast+sum lowering is
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
        let trap = NumericTrap::Overflow {
            op: "count",
            prim: Prim::Int64,
        }
        .to_string();
        let fold = Self::adjacent_pair_fold_lines(
            &format!("__level_{id}"),
            &format!("__level_n_{id}"),
            |left, right| format!("chelis_int_checked_add({left}, {right}, 64, {trap:?})"),
        );
        self.emit_relative_lines(fold);
        self.line(&format!(
            "__count_out_{id}[outer] = (__count_n_{id} == 0) ? 0 : __level_{id}[0];"
        ));
        self.line(&format!("chelis_tensor_end_write(__count_guard_{id});"));
        self.line(&format!("chelis_tensor_release(__count_scratch_{id});"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_reduction_plan_release(t{id}_reduction);"));
    }

    /// The C spelling of an `i64` index or count, from the element-type
    /// authority.
    fn index_c_type() -> &'static str {
        Self::elem_type(&TensorType {
            dims: vec![],
            precision: Prim::Int64,
        })
    }

    /// [05-OP-30]'s canonical adjacent-pair balanced tree over
    /// `level[0..count)`, folded in place into `level[0]`. Each pass combines
    /// positions `2j` and `2j + 1` in order and carries an odd last leaf
    /// unchanged; `combine` spells one addition of two element expressions.
    /// Every canonical fold the C backend emits goes through this loop, so
    /// the tree's order has one definition in the lane. `count` is an
    /// `int64_t` lvalue the fold consumes.
    pub(crate) fn adjacent_pair_fold_lines(
        level: &str,
        count: &str,
        combine: impl Fn(&str, &str) -> String,
    ) -> Vec<(usize, String)> {
        let index = Self::index_c_type();
        let sum = combine(
            &format!("{level}[__pair_left]"),
            &format!("{level}[__pair_right]"),
        );
        vec![
            (0, format!("while ({count} > 1) {{")),
            (
                1,
                format!("{index} __pair_next = {count} / 2 + {count} % 2;"),
            ),
            (
                1,
                format!("for ({index} __pair = 0; __pair < __pair_next; __pair++) {{"),
            ),
            (2, format!("{index} __pair_left = 2 * __pair;")),
            (2, format!("{index} __pair_right = __pair_left + 1;")),
            (
                2,
                format!(
                    "{level}[__pair] = (__pair_right < {count}) ? {sum} : {level}[__pair_left];"
                ),
            ),
            (1, "}".into()),
            (1, format!("{count} = __pair_next;")),
            (0, "}".into()),
        ]
    }

    /// Emit lines whose indentation is relative to the current depth.
    fn emit_relative_lines(&mut self, lines: Vec<(usize, String)>) {
        for (depth, line) in lines {
            self.indent += depth;
            self.line(&line);
            self.indent -= depth;
        }
    }

    /// The scratch of one [05-OP-33] scatter-add tree, in allocation order.
    fn scatter_add_tree_scratch(
        count: &str,
        out_count: &str,
        precision: Prim,
    ) -> [(&'static str, String, &'static str, &'static str); 4] {
        let index = Self::index_c_type();
        let index_dtype = Self::dtype_macro(&TensorType {
            dims: vec![],
            precision: Prim::Int64,
        });
        let scalar = TensorType {
            dims: vec![],
            precision,
        };
        [
            ("dest", count.to_string(), index, index_dtype),
            ("order", count.to_string(), index, index_dtype),
            ("end", out_count.to_string(), index, index_dtype),
            (
                "level",
                format!("{count} + 1"),
                Self::elem_type(&scalar),
                Self::dtype_macro(&scalar),
            ),
        ]
    }

    /// [05-OP-33] scatter-add, in three parts that bracket the caller's own
    /// checked sparse loop. This part allocates the scratch. In that loop,
    /// [`Self::scatter_add_tree_record_line`] records each update's
    /// destination; [`Self::scatter_add_tree_close_lines`] then folds into
    /// `out`, which already holds the base. Each destination's leaves are its
    /// base value followed by its targeting updates in increasing row-major
    /// update order (a stable counting sort of the recorded destinations),
    /// combined by [`Self::adjacent_pair_fold_lines`] at the operand
    /// arithmetic width: f16 and bf16 widen to f32 and narrow at every node
    /// ([04-NUM-8]), integers check overflow at every node. An untargeted
    /// destination keeps its base bits; a targeted one is finalized once.
    /// The DAG kernel (the gather adjoint) and the host sparse summary both
    /// emit these parts, so the lanes cannot fold differently.
    pub(crate) fn scatter_add_tree_open_lines(
        prefix: &str,
        count: &str,
        out_count: &str,
        precision: Prim,
    ) -> Vec<(usize, String)> {
        let index = Self::index_c_type();
        let mut lines = Vec::new();
        for (name, extent, et, dtype) in Self::scatter_add_tree_scratch(count, out_count, precision)
        {
            let var = format!("{prefix}_{name}");
            lines.extend([
                (0, format!("{index} {var}_n = {extent};")),
                (
                    0,
                    format!("chelis_tensor *{var}_tensor = chelis_alloc(1, &{var}_n, {dtype});"),
                ),
                (
                    0,
                    format!(
                        "chelis_tensor_write *{var}_guard = chelis_tensor_begin_write({var}_tensor);"
                    ),
                ),
                (
                    0,
                    format!("{et} *{var} = ({et}*)chelis_tensor_write_view({var}_guard).data;"),
                ),
            ]);
        }
        lines
    }

    /// Record that update `update` targets the checked data index
    /// `destination`; see [`Self::scatter_add_tree_open_lines`].
    pub(crate) fn scatter_add_tree_record_line(
        prefix: &str,
        update: &str,
        destination: &str,
    ) -> String {
        format!("{prefix}_dest[{update}] = {destination}; {prefix}_end[{destination}] += 1;")
    }

    /// Fold every recorded destination and release the scratch; see
    /// [`Self::scatter_add_tree_open_lines`].
    pub(crate) fn scatter_add_tree_close_lines(
        prefix: &str,
        count: &str,
        out: &str,
        out_count: &str,
        updates: &str,
        precision: Prim,
        finalization: Option<crate::fp_env::NanFinalization>,
    ) -> Vec<(usize, String)> {
        let scalar = TensorType {
            dims: vec![],
            precision,
        };
        let element = Self::elem_type(&scalar);
        let index = Self::index_c_type();
        let combine = |left: &str, right: &str| {
            if matches!(precision, Prim::F16 | Prim::Bf16) {
                let load = Self::reduced_to_f32_fn(precision);
                let store = Self::f32_to_reduced_fn(precision);
                format!("{store}({load}({left}) + {load}({right}))")
            } else if precision.is_integer() {
                let bits = Self::integer_width(precision);
                let trap = NumericTrap::Overflow {
                    op: "scatter",
                    prim: precision,
                }
                .to_string();
                format!(
                    "({element})chelis_int_checked_add(({index}){left}, ({index}){right}, {bits}, {trap:?})"
                )
            } else {
                format!("{left} + {right}")
            }
        };
        let level = format!("{prefix}_level");
        let total = if matches!(precision, Prim::F16 | Prim::Bf16) {
            format!(
                "{}({}({level}[0]))",
                Self::f32_to_reduced_fn(precision),
                Self::reduced_to_f32_fn(precision)
            )
        } else {
            finalize_elem(finalization, format!("{level}[0]"), &scalar)
        };
        let mut lines = vec![
            (
                0,
                format!(
                    "for ({index} __d = 1; __d < {out_count}; ++__d) {prefix}_end[__d] += {prefix}_end[__d - 1];"
                ),
            ),
            (
                0,
                format!(
                    "for ({index} __i = {count} - 1; __i >= 0; --__i) {prefix}_order[--{prefix}_end[{prefix}_dest[__i]]] = __i;"
                ),
            ),
            (
                0,
                format!("for ({index} __d = 0; __d < {out_count}; ++__d) {{"),
            ),
            (
                1,
                format!(
                    "{index} __first = {prefix}_end[__d], __last = __d + 1 < {out_count} ? {prefix}_end[__d + 1] : {count};"
                ),
            ),
            (1, "if (__first == __last) continue;".into()),
            (1, format!("{level}[0] = (({element}*){out})[__d];")),
            (
                1,
                format!(
                    "for ({index} __k = __first; __k < __last; ++__k) {level}[__k - __first + 1] = ((const {element}*){updates})[{prefix}_order[__k]];"
                ),
            ),
            (1, format!("{index} __leaves = __last - __first + 1;")),
        ];
        lines.extend(
            Self::adjacent_pair_fold_lines(&level, "__leaves", combine)
                .into_iter()
                .map(|(depth, line)| (depth + 1, line)),
        );
        lines.extend([
            (1, format!("(({element}*){out})[__d] = {total};")),
            (0, "}".into()),
        ]);
        for (name, ..) in Self::scatter_add_tree_scratch(count, out_count, precision)
            .iter()
            .rev()
        {
            lines.extend([
                (
                    0,
                    format!("chelis_tensor_end_write({prefix}_{name}_guard);"),
                ),
                (0, format!("chelis_tensor_release({prefix}_{name}_tensor);")),
            ]);
        }
        lines
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
        let fold = Self::adjacent_pair_fold_lines(
            &format!("__sum_level_{id}"),
            &format!("__sum_n_{id}"),
            |left, right| {
                if matches!(precision, Prim::F16 | Prim::Bf16) {
                    let load = Self::reduced_to_f32_fn(precision);
                    let store = Self::f32_to_reduced_fn(precision);
                    format!("{store}({load}({left}) + {load}({right}))")
                } else if precision.is_integer() {
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
                }
            },
        );
        self.emit_relative_lines(fold);
        // [04-NUM-2]: the stored total is finalized once. A carried leaf (a
        // one-element group) has had no addition, so at f16 and bf16 it
        // finalizes by the same widening and canonicalizing narrowing a sum
        // takes.
        let total = format!("__sum_n_{id} ? __sum_level_{id}[0] : {zero}");
        let total = if matches!(precision, Prim::F16 | Prim::Bf16) {
            format!(
                "{}({}({total}))",
                Self::f32_to_reduced_fn(precision),
                Self::reduced_to_f32_fn(precision)
            )
        } else {
            finalize_elem(
                self.nan_finalization,
                total,
                &TensorType {
                    dims: vec![],
                    precision,
                },
            )
        };
        self.line(&format!("(({et}*)t{id}_data)[outer] = {total};"));
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
        self.open_element_loop(
            id,
            "outer",
            &format!("t{id}_size"),
            Some("#pragma omp parallel for"),
        );
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
        // An integer sum under an activation adds zeros where its row is
        // inactive, so it cannot overflow there (spec/10 section 3.2).
        let load = self.gated(load, 0);
        self.line(&format!(
            "__sum_level_{id}[__reduce_i] = ({acc_et})({load});"
        ));
        self.indent -= 1;
        self.line("}");
        self.emit_sum_fold(id, ty.precision);
        self.close_element_loop();
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
        self.emit_reduce_extreme(id, axis, inputs, ty, dag, true)
    }

    /// [05-OP-12..13] stored-width max/min selection. The selected result is
    /// copied from the source slot rather than reconstructed from an
    /// accumulator, preserving the first NaN payload/sign and the first
    /// stored representation among numerically equal values.
    fn emit_reduce_extreme(
        &mut self,
        id: usize,
        axis: usize,
        inputs: &[NodeId],
        ty: &TensorType,
        dag: VerifiedDagView<'_>,
        take_max: bool,
    ) -> Result<(), Unsupported> {
        let a = inputs[0].0;
        let input_ty = &dag
            .get(inputs[0])
            .expect("extrema input exists")
            .output_type;
        debug_assert_eq!(
            input_ty.precision, ty.precision,
            "verified extrema reductions preserve operand dtype"
        );
        let prim = input_ty.precision;
        let operation = if take_max { "max_reduce" } else { "min_reduce" };
        let runtime_operation = if take_max {
            "CHELIS_REDUCE_MAX"
        } else {
            "CHELIS_REDUCE_MIN"
        };
        let cmp = if take_max { ">" } else { "<" };
        let storage_et = Self::elem_type(input_ty);
        let arithmetic_et = if matches!(prim, Prim::F16 | Prim::Bf16) {
            "float"
        } else {
            storage_et
        };
        let load = |index: &str| {
            let stored = format!("((const {storage_et}*)t{a}_data)[{index}]");
            if matches!(prim, Prim::F16 | Prim::Bf16) {
                format!("{}({stored})", Self::reduced_to_f32_fn(prim))
            } else {
                stored
            }
        };
        self.emit_reduction_plan(id, Some(a), input_ty, &[axis], ty, runtime_operation, false);
        let empty = self.gated_check(&format!("t{id}_leaf_count == 0"));
        self.line(&format!("if ({empty}) {{"));
        self.indent += 1;
        let trap = NumericTrap::Domain {
            op: operation,
            prim,
        }
        .to_string();
        self.line(&format!("chelis_numeric_trap({trap:?});"));
        self.indent -= 1;
        self.line("}");
        self.emit_slot_wrapper(id, ty);
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line("int64_t best_src = -1;");
        self.line(&format!("{arithmetic_et} best_value = ({arithmetic_et})0;"));
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < t{id}_leaf_count; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src_idx = chelis_reduction_index(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)__reduce_i));"));
        self.line(&format!("{arithmetic_et} candidate = {};", load("src_idx")));
        let replace = if prim.is_float() {
            format!(
                "best_src < 0 || (!isnan(best_value) && (isnan(candidate) || candidate {cmp} best_value))"
            )
        } else {
            format!("best_src < 0 || candidate {cmp} best_value")
        };
        self.line(&format!("if ({replace}) {{"));
        self.indent += 1;
        self.line("best_src = src_idx;");
        self.line("best_value = candidate;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        // Under an activation an empty axis may reach here unchecked: it
        // selects nothing and reduces to zero, reading no element.
        let selected = format!("((const {storage_et}*)t{a}_data)[best_src]");
        let selected = if self.gate.is_some() {
            format!("(best_src < 0 ? ({storage_et})0 : {selected})")
        } else {
            selected
        };
        self.line(&format!("(({storage_et}*)t{id}_data)[outer] = {selected};"));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_reduction_plan_release(t{id}_reduction);"));
        Ok(())
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
            let reduced = finalize_elem(
                self.nan_finalization,
                format!("{fn_name}((const float*)t{a}_data, t{a}_size)"),
                ty,
            );
            self.line(&format!("((float*)t{id}_data)[0] = {reduced};"));
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
        let acc = finalize_elem(self.nan_finalization, "acc".into(), ty);
        self.line(&format!("((float*)t{id}_data)[outer] = {acc};"));
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
                            UnsupportedKind::Construct("a window parameter outside i64".into()),
                            format!("the C DAG emitter (node {id})"),
                            Stage::Codegen("c"),
                            chelis_types::deliberate_rejection!(
                                "[05-RWIN-1]",
                                "window extents and strides require positive i64 values"
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
        let prim = input_node.output_type.precision;
        if ty.precision != prim {
            return Err(Unsupported::new(
                UnsupportedKind::Op("reduce_window_*".to_string()),
                format!(
                    "input `{}` and output `{}` dtypes in the C DAG emitter (node {id})",
                    prim.name(),
                    ty.precision.name()
                ),
                Stage::Codegen("c"),
                chelis_types::deliberate_rejection!(
                    "[05-OP-39]",
                    "window reductions preserve the operand dtype"
                ),
            ));
        }
        if matches!(reducer, ReduceWindowKind::Mean) && !prim.is_float() {
            return Err(Unsupported::new(
                UnsupportedKind::Op("reduce_window_mean".to_string()),
                format!("`{}` tensors in the C DAG emitter (node {id})", prim.name()),
                Stage::Codegen("c"),
                chelis_types::deliberate_rejection!(
                    "[05-OP-39]",
                    "reduce_window_mean admits only active float tensor dtypes"
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
        let storage_et = Self::elem_type(&input_node.output_type);
        let arithmetic_prim = if matches!(prim, Prim::F16 | Prim::Bf16) {
            Prim::F32
        } else {
            prim
        };
        let arithmetic_ty = TensorType {
            dims: vec![],
            precision: arithmetic_prim,
        };
        let arithmetic_et = Self::elem_type(&arithmetic_ty);
        let arithmetic_dtype = Self::dtype_macro(&arithmetic_ty);
        let load = |index: &str| {
            let stored = format!("((const {storage_et}*)t{a}_data)[{index}]");
            if matches!(prim, Prim::F16 | Prim::Bf16) {
                format!("{}({stored})", Self::reduced_to_f32_fn(prim))
            } else {
                stored
            }
        };
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        if matches!(reducer, ReduceWindowKind::Max | ReduceWindowKind::Min) {
            let cmp = if matches!(reducer, ReduceWindowKind::Max) {
                ">"
            } else {
                "<"
            };
            self.line("int64_t best_src = -1;");
            self.line(&format!("{arithmetic_et} best_value = ({arithmetic_et})0;"));
            self.line(&format!(
                "for (int64_t leaf = 0; leaf < t{id}_window_count; leaf++) {{"
            ));
            self.indent += 1;
            self.line(&format!("int64_t src_idx = chelis_window_index(t{id}_window, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)leaf));"));
            self.line(&format!("{arithmetic_et} candidate = {};", load("src_idx")));
            let replace = if prim.is_float() {
                format!(
                    "best_src < 0 || (!isnan(best_value) && (isnan(candidate) || candidate {cmp} best_value))"
                )
            } else {
                format!("best_src < 0 || candidate {cmp} best_value")
            };
            self.line(&format!("if ({replace}) {{"));
            self.indent += 1;
            self.line("best_src = src_idx;");
            self.line("best_value = candidate;");
            self.indent -= 1;
            self.line("}");
            self.indent -= 1;
            self.line("}");
            self.line(&format!(
                "(({storage_et}*)t{id}_data)[outer] = ((const {storage_et}*)t{a}_data)[best_src];"
            ));
        } else {
            self.line(&format!("int64_t level_n = t{id}_window_count;"));
            self.line(&format!(
                "chelis_tensor *level_tensor = chelis_alloc(1, &level_n, {arithmetic_dtype});"
            ));
            self.line(
                "chelis_tensor_write *level_guard = chelis_tensor_begin_write(level_tensor);",
            );
            self.line(&format!(
                "{arithmetic_et} *level = ({arithmetic_et}*)chelis_tensor_write_view(level_guard).data;"
            ));
            self.line(&format!(
                "for (int64_t leaf = 0; leaf < t{id}_window_count; leaf++) {{"
            ));
            self.indent += 1;
            self.line(&format!("int64_t src_idx = chelis_window_index(t{id}_window, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)leaf));"));
            self.line(&format!(
                "level[leaf] = ({arithmetic_et})({});",
                load("src_idx")
            ));
            self.indent -= 1;
            self.line("}");
            let fold = Self::adjacent_pair_fold_lines("level", "level_n", |left, right| {
                if prim.is_integer() {
                    let bits = Self::integer_width(prim);
                    let trap = NumericTrap::Overflow {
                        op: "reduce_window_sum",
                        prim,
                    }
                    .to_string();
                    format!(
                        "({arithmetic_et})chelis_int_checked_add((int64_t){left}, (int64_t){right}, {bits}, {trap:?})"
                    )
                } else {
                    format!("{left} + {right}")
                }
            });
            self.emit_relative_lines(fold);
            self.line(&format!("{arithmetic_et} result = level[0];"));
            if matches!(reducer, ReduceWindowKind::Mean) {
                self.line(&format!(
                    "result = result / ({arithmetic_et})t{id}_window_count;"
                ));
            }
            if matches!(prim, Prim::F16 | Prim::Bf16) {
                self.line(&format!(
                    "((uint16_t*)t{id}_data)[outer] = {}(result);",
                    Self::f32_to_reduced_fn(prim)
                ));
            } else {
                let result = finalize_elem(self.nan_finalization, "result".into(), ty);
                self.line(&format!(
                    "(({storage_et}*)t{id}_data)[outer] = ({storage_et}){result};"
                ));
            }
            self.line("chelis_tensor_end_write(level_guard);");
            self.line("chelis_tensor_release(level_tensor);");
        }
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
    /// `Max`/`Min` route the first-NaN cotangent or split `g / k` across
    /// equal non-NaN extrema. Geometry follows `spec/05-risc-primitives.md`
    /// §2.3.1.
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
        let prim = x_node.output_type.precision;
        if !prim.is_float() || g_node.output_type.precision != prim || ty.precision != prim {
            return Err(Unsupported::new(
                UnsupportedKind::Op("reduce_window_grad".to_string()),
                format!(
                    "x `{}`, g `{}`, and output `{}` dtypes in the C DAG emitter (node {id})",
                    prim.name(),
                    g_node.output_type.precision.name(),
                    ty.precision.name()
                ),
                Stage::Codegen("c"),
                chelis_types::deliberate_rejection!(
                    "[05-OP-39]",
                    "window adjoints require one matching active float dtype"
                ),
            ));
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
        let storage_et = Self::elem_type(ty);
        let arithmetic_prim = if matches!(prim, Prim::F16 | Prim::Bf16) {
            Prim::F32
        } else {
            prim
        };
        let arithmetic_ty = TensorType {
            dims: vec![],
            precision: arithmetic_prim,
        };
        let arithmetic_et = Self::elem_type(&arithmetic_ty);
        let arithmetic_dtype = Self::dtype_macro(&arithmetic_ty);
        let load_x = |index: &str| {
            let stored = format!("((const {storage_et}*)t{x}_data)[{index}]");
            if matches!(prim, Prim::F16 | Prim::Bf16) {
                format!("{}({stored})", Self::reduced_to_f32_fn(prim))
            } else {
                stored
            }
        };
        let load_g = |index: &str| {
            let stored = format!("((const {storage_et}*)t{g}_data)[{index}]");
            if matches!(prim, Prim::F16 | Prim::Bf16) {
                format!("{}({stored})", Self::reduced_to_f32_fn(prim))
            } else {
                stored
            }
        };
        // [05-RWIN-1]: gather one destination's contributions in increasing
        // row-major output order, then combine them with the canonical
        // adjacent-pair tree. This is deterministic even for overlapping
        // windows and needs no racing scatter.
        self.line(&format!(
            "for (int64_t dst_idx = 0; dst_idx < t{id}_size; dst_idx++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t contribution_capacity = t{g}_size;"));
        self.line(&format!(
            "chelis_tensor *contribution_tensor = chelis_alloc(1, &contribution_capacity, {arithmetic_dtype});"
        ));
        self.line("chelis_tensor_write *contribution_guard = chelis_tensor_begin_write(contribution_tensor);");
        self.line(&format!(
            "{arithmetic_et} *contributions = ({arithmetic_et}*)chelis_tensor_write_view(contribution_guard).data;"
        ));
        self.line("int64_t contribution_n = 0;");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{g}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line("int in_window = 0;");
        self.line(&format!(
            "for (int64_t leaf = 0; leaf < t{id}_window_count; leaf++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src_idx = chelis_window_index(t{id}_window, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)leaf));"));
        self.line("if (src_idx == dst_idx) in_window = 1;");
        self.indent -= 1;
        self.line("}");
        self.line("if (!in_window) continue;");
        self.line(&format!(
            "{arithmetic_et} contribution = {};",
            load_g("outer")
        ));
        match reducer {
            ReduceWindowKind::Sum => {}
            ReduceWindowKind::Mean => self.line(&format!(
                "contribution = contribution / ({arithmetic_et})t{id}_window_count;"
            )),
            ReduceWindowKind::Max | ReduceWindowKind::Min => {
                let cmp = if matches!(reducer, ReduceWindowKind::Max) {
                    ">"
                } else {
                    "<"
                };
                self.line("int64_t best_src = -1;");
                self.line(&format!("{arithmetic_et} best_value = ({arithmetic_et})0;"));
                self.line(&format!(
                    "for (int64_t leaf = 0; leaf < t{id}_window_count; leaf++) {{"
                ));
                self.indent += 1;
                self.line(&format!("int64_t src_idx = chelis_window_index(t{id}_window, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)leaf));"));
                self.line(&format!(
                    "{arithmetic_et} candidate = {};",
                    load_x("src_idx")
                ));
                self.line(&format!(
                    "if (best_src < 0 || (!isnan(best_value) && (isnan(candidate) || candidate {cmp} best_value))) {{"
                ));
                self.indent += 1;
                self.line("best_src = src_idx;");
                self.line("best_value = candidate;");
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
                self.line("if (isnan(best_value)) {");
                self.indent += 1;
                self.line("if (dst_idx != best_src) continue;");
                self.indent -= 1;
                self.line("} else {");
                self.indent += 1;
                self.line(&format!(
                    "if (!({} == best_value)) continue;",
                    load_x("dst_idx")
                ));
                self.line("int64_t tie_count = 0;");
                self.line(&format!(
                    "for (int64_t leaf = 0; leaf < t{id}_window_count; leaf++) {{"
                ));
                self.indent += 1;
                self.line(&format!("int64_t src_idx = chelis_window_index(t{id}_window, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)leaf));"));
                self.line(&format!(
                    "if ({} == best_value) tie_count++;",
                    load_x("src_idx")
                ));
                self.indent -= 1;
                self.line("}");
                self.line(&format!(
                    "contribution = contribution / ({arithmetic_et})tie_count;"
                ));
                self.indent -= 1;
                self.line("}");
            }
        }
        self.line("contributions[contribution_n++] = contribution;");
        self.indent -= 1;
        self.line("}");
        let fold =
            Self::adjacent_pair_fold_lines("contributions", "contribution_n", |left, right| {
                format!("{left} + {right}")
            });
        self.emit_relative_lines(fold);
        self.line(&format!(
            "{arithmetic_et} result = contribution_n ? contributions[0] : ({arithmetic_et})0;"
        ));
        if matches!(prim, Prim::F16 | Prim::Bf16) {
            self.line(&format!(
                "((uint16_t*)t{id}_data)[dst_idx] = {}(result);",
                Self::f32_to_reduced_fn(prim)
            ));
        } else {
            let result = finalize_elem(self.nan_finalization, "result".into(), ty);
            self.line(&format!(
                "(({storage_et}*)t{id}_data)[dst_idx] = ({storage_et}){result};"
            ));
        }
        self.line("chelis_tensor_end_write(contribution_guard);");
        self.line("chelis_tensor_release(contribution_tensor);");
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_window_plan_release(t{id}_window);"));
        Ok(())
    }

    // ---- Argmax / Argmin ----
    //
    // Emits an exact i64 index-tracking reduction.
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
        let input_ty = &dag
            .get(inputs[0])
            .expect("argument reduction input exists")
            .output_type;
        debug_assert_eq!(
            ty.precision,
            Prim::Int64,
            "verified argmax/argmin output dtype is i64"
        );
        let prim = input_ty.precision;
        let operation = if is_argmax {
            "argmax_reduce"
        } else {
            "argmin_reduce"
        };
        let cmp = if is_argmax { ">" } else { "<" };
        let storage_et = Self::elem_type(input_ty);
        let arithmetic_et = if matches!(prim, Prim::F16 | Prim::Bf16) {
            "float"
        } else {
            storage_et
        };
        let load = |index: &str| {
            let stored = format!("((const {storage_et}*)t{a}_data)[{index}]");
            if matches!(prim, Prim::F16 | Prim::Bf16) {
                format!("{}({stored})", Self::reduced_to_f32_fn(prim))
            } else {
                stored
            }
        };
        self.emit_reduction_plan(
            id,
            Some(a),
            input_ty,
            &[axis],
            ty,
            if is_argmax {
                "CHELIS_REDUCE_ARGMAX"
            } else {
                "CHELIS_REDUCE_ARGMIN"
            },
            false,
        );
        let empty = self.gated_check(&format!("t{id}_leaf_count == 0"));
        self.line(&format!("if ({empty}) {{"));
        self.indent += 1;
        let trap = NumericTrap::Domain {
            op: operation,
            prim: Prim::Int64,
        }
        .to_string();
        self.line(&format!("chelis_numeric_trap({trap:?});"));
        self.indent -= 1;
        self.line("}");
        self.emit_slot_wrapper(id, ty);
        let dst_et = Self::elem_type(ty);
        self.line("#pragma omp parallel for");
        self.line(&format!(
            "for (int64_t outer = 0; outer < t{id}_size; outer++) {{"
        ));
        self.indent += 1;
        self.line(&format!("{arithmetic_et} best_value = ({arithmetic_et})0;"));
        self.line("int64_t best_idx = -1;");
        self.line(&format!(
            "for (int64_t __reduce_i = 0; __reduce_i < t{id}_leaf_count; __reduce_i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src_idx = chelis_reduction_index(t{id}_reduction, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)outer), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)__reduce_i));"));
        self.line(&format!("{arithmetic_et} candidate = {};", load("src_idx")));
        let replace = if prim.is_float() {
            format!(
                "best_idx < 0 || (!isnan(best_value) && (isnan(candidate) || candidate {cmp} best_value))"
            )
        } else {
            format!("best_idx < 0 || candidate {cmp} best_value")
        };
        self.line(&format!("if ({replace}) {{"));
        self.indent += 1;
        self.line("best_value = candidate;");
        self.line("best_idx = __reduce_i;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        // Under an activation an empty axis may reach here unchecked: it
        // selects no index and reduces to zero.
        let selected = if self.gate.is_some() {
            "(best_idx < 0 ? 0 : best_idx)"
        } else {
            "best_idx"
        };
        self.line(&format!(
            "(({dst_et}*)t{id}_data)[outer] = ({dst_et}){selected};"
        ));
        self.indent -= 1;
        self.line("}");
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
        // zero, `float acc = 0.0f` initializer, `chelis_cr_expf`-family
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
        if reduce_kind == "max" {
            // The checked plan reports zero leaves for an empty result even
            // when the selected axis is nonempty ([05-OP-12]). The Domain
            // condition belongs to the axis itself, not that plan count.
            let selected_extent = Self::emit_dim_info(&fused_input_type.dims[axis]);
            let empty = self.gated_check(&format!("({selected_extent}) == 0"));
            let trap = NumericTrap::Domain {
                op: "max_reduce",
                prim: out_ty.precision,
            }
            .to_string();
            self.line(&format!(
                "if ({empty}) {{ chelis_numeric_trap({trap:?}); }}"
            ));
        }
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
                    format!("({a}) - ({b})")
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
                    format!("chelis_cr_expf({a})")
                }
                FusedStepOp::Log => {
                    let a = resolve(&step.input_indices[0]);
                    format!("chelis_cr_logf({a})")
                }
                FusedStepOp::Sin => {
                    let a = resolve(&step.input_indices[0]);
                    format!("chelis_cr_sinf({a})")
                }
                FusedStepOp::Sqrt => {
                    let a = resolve(&step.input_indices[0]);
                    format!("sqrtf({a})")
                }
                FusedStepOp::Cos => {
                    let a = resolve(&step.input_indices[0]);
                    format!("chelis_cr_cosf({a})")
                }
                FusedStepOp::Tan => {
                    let a = resolve(&step.input_indices[0]);
                    format!("chelis_cr_tanf({a})")
                }
                FusedStepOp::Atan => {
                    let a = resolve(&step.input_indices[0]);
                    format!("chelis_cr_atanf({a})")
                }
                FusedStepOp::Tanh => {
                    let a = resolve(&step.input_indices[0]);
                    format!("chelis_cr_tanhf({a})")
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
            // [04-NUM-2]: arithmetic finalizes NaN to the canonical quiet
            // NaN; extrema select and keep their operand's bits.
            let expr = crate::fp_env::finalize_float(
                &expr,
                false,
                crate::fp_env::fused_step_nan_finalization(&step.op),
            );
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
        _dag: VerifiedDagView<'_>,
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
            .map(|(axis, dim)| (axis, Self::bound_c_expr(dim, inputs, a, axis)))
            .collect();
        // Read from its target's scalars and its operand's metadata: a
        // claim-sized `reshape` declares them before its branch.
        self.before_inactive_zeros_branch(id, |emitter| {
            emitter.emit_runtime_dim_sites(id, &extents);
        });
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
        self.emit_affine_bounds(
            &format!("t{id}_axes"),
            &axes.iter().map(usize::to_string).collect::<Vec<_>>(),
        );
        self.line(&format!("chelis_movement_plan *t{id}_movement = chelis_tensor_permute_plan(t{a}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {}), t{id}_axes);", axes.len()));
        self.line(&format!("chelis_movement_check_target(t{id}_movement, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {}), {});", Self::ndim(ty), Self::tagged_shape_literal(ty)));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "for (int64_t i = 0; i < chelis_movement_count(t{id}_movement); i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src = chelis_movement_index(t{id}_movement, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i));"));
        self.line(&format!(
            "(({elem_type}*)t{id}_data)[i] = ((const {elem_type}*)t{a}_data)[src];"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_movement_plan_release(t{id}_movement);"));
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
        let extent = Self::bound_c_expr(size, inputs, a, axis);
        if self.runtime_dim_sites.contains_key(&(id, axis))
            || self.local_dim_guard_sites.contains_key(&id)
            || self
                .inherited_result_sites
                .iter()
                .any(|site| site.producer() == NodeId(id))
        {
            // The inserted and forwarded axes belong to one result producer.
            // Supply them together so the shared consumer follows declaration
            // order, rather than checking forwarded axes in an earlier hook.
            let mut extents = self.input_axis_result_extents(id, inputs);
            if !extents.iter().any(|(existing, _)| *existing == axis) {
                extents.push((axis, extent.clone()));
            }
            // Its operands' metadata and scalars, read before any plan: a
            // claim-sized `expand` declares them before its branch.
            self.before_inactive_zeros_branch(id, |emitter| {
                emitter.emit_runtime_dim_sites(id, &extents)
            });
        }
        let operation = match dag.expansion_kind(NodeId(id)) {
            chelis_ir::axis_sources::ExpansionKind::Expand => "CHELIS_MOVEMENT_EXPAND",
            chelis_ir::axis_sources::ExpansionKind::Insert => "CHELIS_MOVEMENT_INSERT",
        };
        let elem_type = Self::elem_type(ty);
        self.line(&format!("chelis_movement_plan *t{id}_movement = chelis_tensor_expand_plan(t{a}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {axis}), chelis_scalar_from_bits(CHELIS_DTYPE_I64, ({extent})), {operation});"));
        self.line(&format!("chelis_movement_check_target(t{id}_movement, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {}), {});", Self::ndim(ty), Self::tagged_shape_literal(ty)));
        self.emit_slot_wrapper(id, ty);
        self.line(&format!(
            "for (int64_t i = 0; i < chelis_movement_count(t{id}_movement); i++) {{"
        ));
        self.indent += 1;
        self.line(&format!("int64_t src = chelis_movement_index(t{id}_movement, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i));"));
        self.line(&format!(
            "(({elem_type}*)t{id}_data)[i] = ((const {elem_type}*)t{a}_data)[src];"
        ));
        self.indent -= 1;
        self.line("}");
        self.line(&format!("chelis_movement_plan_release(t{id}_movement);"));
    }

    // ---- Pad ----
    /// chelis#616: the C integer expression for a movement [`RtDim`] at run
    /// time. `Lit` is a literal; `ToEnd` reads the input tensor's runtime axis
    /// checked runtime extent; `Node(i)` reads the rank-0 integer bound
    /// scalar `t{inputs[i]}->data[0]` with its declared element type, cast to
    /// `int` for use as a C index.
    fn bound_c_expr(bound: &RtDim, inputs: &[NodeId], a: usize, axis: usize) -> String {
        match bound {
            RtDim::Lit(n) => n.to_string(),
            RtDim::ToEnd => format!("chelis_tensor_shape(t{a}, {axis})"),
            RtDim::Node(i) => {
                let n = inputs[*i].0;
                format!("((int64_t*)t{n}_data)[0]")
            }
            RtDim::InputAxis {
                tensor,
                axis: RtAxis::Lit(source_axis),
            } => {
                let source = inputs[*tensor].0;
                format!("chelis_tensor_shape(t{source}, {source_axis})")
            }
            // A symbolic dim (reshape targets only; verify rejects it in
            // movement bounds) is a declared C variable, exactly as
            // `emit_dim_info` renders a runtime-bound named dimension.
            RtDim::Sym(name) => extent_read(name),
        }
    }

    /// Declare every name whose extent this operation's axis produces.
    /// Guard scheduling is separate from C variable declaration.
    ///
    /// The first name reads the extent expression; each later name reads the
    /// first. Two spellings of one extent is the ordinary case rather than a
    /// corner: chelis#665's kept axis carries the lowerer's fresh
    /// `_anon_dim_2_1` for the extent the `stride` before it already
    /// declares, and `spec/design/runtime_extents.md` C4 forwards an
    /// unchanged axis's exact input axis, which is this declaration.
    fn emit_runtime_dim_site(&mut self, id: usize, axis: usize, extent_expr: &str) {
        let Some(names) = self.runtime_dim_sites.get(&(id, axis)).cloned() else {
            return;
        };
        for (name, extent_expr) in
            Self::runtime_dim_declarations(&names, extent_expr, &mut self.declared_dim_names)
        {
            self.line(&format!("int64_t {name} = {extent_expr};"));
        }
    }

    /// The marked name and right-hand side of each declaration of the
    /// `names` not yet in `declared`, which records them. Every name reads
    /// `extent_expr`, and after the first that variable holds the name
    /// declared before it, so two spellings of one extent land as
    /// `int64_t b = a;` rather than as a second read of the same expression.
    fn runtime_dim_declarations(
        names: &[String],
        extent_expr: &str,
        declared: &mut chelis_unord::UnordSet<String>,
    ) -> Vec<(String, String)> {
        let mut extent_expr = extent_expr.to_string();
        let mut declarations = Vec::new();
        for name in names {
            if declared.contains(name) {
                continue;
            }
            declared.insert(name.clone());
            let read = extent_read(name);
            declarations.push((
                extent_declaration(name),
                std::mem::replace(&mut extent_expr, read),
            ));
        }
        declarations
    }

    /// Declare supported runtime extents, then consume this operation's
    /// claims in declaration order. Supplying all axes together preserves
    /// that order even when the output permutes the signature's dimensions.
    fn emit_runtime_dim_sites(&mut self, id: usize, extents: &[(usize, String)]) {
        // A claim-sized node declares these before its activation's branch
        // ([`Self::open_inactive_zeros`]); where no row is active an axis
        // takes the extent its unchecked claim states, as its zero value
        // does in the evaluator ([`Self::inactive_extent`]).
        let gated;
        let extents = match self
            .inactive_zeros
            .filter(|zeros| zeros.node == id)
            .and_then(|_| self.claim_sized_activity(id))
        {
            Some(active) => {
                gated = extents
                    .iter()
                    .map(|(axis, extent)| {
                        let inactive = self.inactive_extent(id, *axis, extent);
                        let extent = if inactive == *extent {
                            extent.clone()
                        } else {
                            format!("(({active}) ? ({extent}) : {inactive})")
                        };
                        (*axis, extent)
                    })
                    .collect::<Vec<_>>();
                gated.as_slice()
            }
            None => extents,
        };
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
        self.emit_local_dim_guards_matching(id, extents, false);
        self.emit_inherited_result_guards(id, extents, false);
    }

    fn is_same_shape_observation(
        observation: &chelis_ir::axis_sources::LocalGuardObservation,
    ) -> bool {
        matches!(
            observation,
            chelis_ir::axis_sources::LocalGuardObservation::SameShapeAgreement(_)
                | chelis_ir::axis_sources::LocalGuardObservation::MalformedSameShapeAgreement(_)
        )
    }

    fn emit_local_dim_guards_matching(
        &mut self,
        id: usize,
        extents: &[(usize, String)],
        same_shape: bool,
    ) {
        // The guard site and the claim it compares against are the
        // derivation's, and the rendering is [04-NUM-9]'s: the complete
        // user-facing line is `numeric trap: domain in <op> at i64` with no
        // prefix and no suffix, `<op>` naming the operation that introduces
        // the extent, and `<prim>` always `i64` because the guard finalizes
        // an extent under [05-DIM-1]. Section 4.7's required context - the
        // disagreeing names, the axis and each observed value - is its own
        // `fprintf`, so the trap line stays exactly one line.
        //
        // That context names the OPERATION rather than the node id. Section
        // 4.7 asks for "the names of the disagreeing sources", binding "the
        // information conveyed and not the bytes rendered", and a node id is
        // not a source name: `spec/06` section 5.2-5.4's dead-code and
        // common-subexpression passes renumber nodes, so the same defect
        // printed a different number depending on what else the program
        // contained. Every other extent diagnostic on both lanes already
        // spells it `<source> axis <axis> = <value>`; this was the last pair
        // that did not.
        // The comparison operand is the class's CANONICAL VALUE, supplied by
        // the derivation: the binder name where a lane declares one, the
        // literal the checker resolved the claim to otherwise. The emitter
        // consults no declaration table, so a claim resolved to a literal over
        // a RUNTIME read still gets the comparison section 4.7 owes between
        // the claimed extent and the value observed - the same comparison the
        // entry path emits for a `Literal` claim (chelis#1377). Keying it on
        // whether a C variable happened to be allocated narrowed a required
        // check to an implementation convenience.
        let sites = self
            .local_dim_guard_sites
            .get(&id)
            .into_iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        // Consume claims, not axes: multiple claims on one axis can be
        // interleaved with claims on another axis in declaration order.
        for (axis, site) in sites {
            if Self::is_same_shape_observation(&site.observed) != same_shape {
                continue;
            }
            let guard_key = (id, axis, site.clone());
            if self.emitted_local_dim_guards.contains(&guard_key) {
                continue;
            }
            // Only the extent forms supported by this movement consumer are
            // supplied here. Other local source kinds retain their existing
            // ownership in runtime_extents.md B2b-0b.
            let Some((_, extent_expr)) = extents.iter().find(|(a, _)| *a == axis) else {
                continue;
            };
            self.emitted_local_dim_guards.push(guard_key);
            // Read the exact captured scalar, the canonical input binding,
            // or the checker-resolved literal, according to the guard plan.
            let operand = Self::canonical_extent_operand(&site.canonical, axis);
            let mismatch = format!("({extent_expr}) != {operand}");
            // The claim's activation is its carrier's owner activation
            // (spec/10 section 3.2): rank 0 at an arm, one Bool per row under
            // `vmap`. The extent is every row's, so the guard runs when any
            // row is active, as the evaluator's `local_guard_is_active`.
            let predicate = if let Some(activation) = site.activation.node() {
                let act = activation.0;
                self.finish_tensor_write_for_checked_read(act);
                let bool_et = Self::prim_elem_type(Prim::Bool);
                self.line("{");
                self.indent += 1;
                self.line("int __local_guard_active = 0;");
                self.line(&format!(
                    "for (int64_t __r = 0; __r < chelis_tensor_numel(t{act}); ++__r) __local_guard_active |= (((const {bool_et}*)t{act}_data)[__r] != 0);"
                ));
                format!("__local_guard_active && ({mismatch})")
            } else {
                mismatch.clone()
            };
            let (name, op) = (site.claim, site.op);
            let name_fmt = chelis_ir::span_sanitize::sanitize_for_format_string(&name);
            self.line(&format!("if ({predicate}) {{"));
            self.indent += 1;
            self.line(&format!(
                "fprintf(stderr, \"extent `{name_fmt}`: claimed = %lld, {op} axis {axis} = %lld\\n\", (long long)({operand}), (long long)({extent_expr}));"
            ));
            self.line(&format!(
                "chelis_numeric_trap(\"numeric trap: domain in {op} at i64\");"
            ));
            self.indent -= 1;
            self.line("}");
            if site.activation.node().is_some() {
                self.indent -= 1;
                self.line("}");
            }
        }
    }

    /// Emit producer-owned result claims after the same-shape relation has
    /// agreed and before the producer allocates or reads an element.
    fn emit_same_shape_result_guards(&mut self, node: &DagNode) {
        use chelis_ir::axis_sources::LocalGuardObservation;
        let mut extents = Vec::new();
        let add_extent =
            |axis: usize, observation: &LocalGuardObservation, extents: &mut Vec<_>| {
                let LocalGuardObservation::SameShapeAgreement(agreement) = observation else {
                    return;
                };
                let member = agreement
                    .members()
                    .first()
                    .expect("verified nonempty same-shape agreement");
                if !extents
                    .iter()
                    .any(|(existing, _): &(usize, String)| *existing == axis)
                {
                    extents.push((axis, format!("chelis_tensor_shape(t{}, {axis})", member.0)));
                }
            };
        if let Some(sites) = self.local_dim_guard_sites.get(&node.id.0) {
            for (axis, claim) in sites {
                add_extent(*axis, &claim.observed, &mut extents);
            }
        }
        for site in &self.inherited_result_sites {
            if site.producer() != node.id {
                continue;
            }
            let RtAxis::Lit(axis) = site.producer_axis();
            add_extent(axis as usize, site.observation(), &mut extents);
        }
        if extents.is_empty() {
            return;
        }
        for (axis, extent) in &extents {
            self.emit_runtime_dim_site(node.id.0, *axis, extent);
        }
        self.emit_local_dim_guards_matching(node.id.0, &extents, true);
        self.emit_inherited_result_guards(node.id.0, &extents, true);
    }

    /// Input-axis observations, shared by ordinary and movement producers.
    fn input_axis_result_extents(&self, id: usize, inputs: &[NodeId]) -> Vec<(usize, String)> {
        use chelis_ir::axis_sources::LocalGuardObservation;
        let mut carriers = Vec::new();
        if let Some(sites) = self.local_dim_guard_sites.get(&id) {
            for (axis, claim) in sites {
                if let LocalGuardObservation::Carrier(carrier @ RtDim::InputAxis { .. }) =
                    &claim.observed
                {
                    carriers.push((*axis, carrier.clone()));
                }
            }
        }
        for site in &self.inherited_result_sites {
            if site.producer() == NodeId(id)
                && let LocalGuardObservation::Carrier(carrier @ RtDim::InputAxis { .. }) =
                    site.observation()
            {
                let RtAxis::Lit(axis) = site.producer_axis();
                if !carriers
                    .iter()
                    .any(|(existing, _)| *existing == axis as usize)
                {
                    carriers.push((axis as usize, carrier.clone()));
                }
            }
        }
        if carriers.is_empty() {
            return Vec::new();
        }
        let operand = inputs
            .first()
            .expect("input-axis observation has an operand")
            .0;
        carriers
            .into_iter()
            .map(|(axis, carrier)| (axis, Self::bound_c_expr(&carrier, inputs, operand, axis)))
            .collect()
    }

    /// Shape-preserving producers know their result extent from input metadata.
    /// Check it before allocating or evaluating a potentially trapping element.
    fn emit_input_axis_result_guards(&mut self, node: &DagNode, _dag: VerifiedDagView<'_>) {
        // Movement owners consume all their result axes together at their
        // dedicated preallocation hook, including forwarded input axes.
        if matches!(node.op, RiscOp::Expand { .. } | RiscOp::Reshape { .. }) {
            return;
        }
        let extents = self.input_axis_result_extents(node.id.0, &node.inputs);
        self.emit_runtime_dim_sites(node.id.0, &extents);
    }

    /// Consume invocation frames in declaration/axis order at one producer.
    fn emit_inherited_result_guards(
        &mut self,
        id: usize,
        extents: &[(usize, String)],
        same_shape: bool,
    ) {
        let sites = self
            .inherited_result_sites
            .iter()
            .filter(|site| {
                site.producer() == NodeId(id)
                    && Self::is_same_shape_observation(site.observation()) == same_shape
            })
            .cloned()
            .collect::<Vec<_>>();
        if sites.is_empty() {
            return;
        }
        let observations = sites
            .iter()
            .filter_map(|site| {
                let chelis_ir::dag::RtAxis::Lit(producer_axis) = site.producer_axis();
                let chelis_ir::dag::RtAxis::Lit(output_axis) = site.output_axis();
                extents
                    .iter()
                    .find(|(axis, _)| *axis == producer_axis as usize)
                    .map(|(_, value)| format!("{{ {output_axis}, {producer_axis}, ({value}) }}"))
            })
            .collect::<Vec<_>>();
        if observations.is_empty() {
            return;
        }
        let rank = self
            .inherited_result_rank
            .expect("inherited result sites require an existing result root");
        let op = sites[0].operation();
        self.line(&format!("__chelis_check_host_result_extent_claims(__chelis_caller_result_claims, {}, (const int64_t[][3]){{ {} }}, {}, \"{op}\", \"numeric trap: domain in {op} at i64\");",
            rank, observations.join(", "), observations.len()));
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
    fn emit_affine_plan(&mut self, id: usize, a: usize, ty: &TensorType, op: &str, bounds: &str) {
        let rank = ty.dims.len();
        let operation = match op {
            "pad" => "CHELIS_MOVEMENT_PAD",
            "shrink" => "CHELIS_MOVEMENT_SHRINK",
            "stride" => "CHELIS_MOVEMENT_STRIDE",
            _ => unreachable!("affine emitter operation"),
        };
        let plan = format!(
            "chelis_tensor_affine_plan(t{a}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {bounds}, {operation})"
        );
        // Under an activation that holds in no row the operation reads no
        // bound (spec/10 section 3.2): it builds no plan, each axis it
        // declares takes the extent a claim states for it or else its
        // operand's ([`Self::inactive_extent`]), and
        // [`Self::emit_movement_copy`] zero-fills its result.
        let active = self.gate.as_ref().map(|gate| gate.any.clone());
        match &active {
            Some(any) => self.line(&format!(
                "chelis_movement_plan *t{id}_movement = {any} ? {plan} : NULL;"
            )),
            None => self.line(&format!("chelis_movement_plan *t{id}_movement = {plan};")),
        }
        let extents = (0..rank)
            .map(|axis| {
                let extent = format!("chelis_movement_extent(t{id}_movement, CHELIS_MOVEMENT_RESULT, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {axis}))");
                let extent = match &active {
                    Some(_) => format!(
                        "(t{id}_movement ? {extent} : {})",
                        self.inactive_extent(id, axis, &format!("chelis_tensor_shape(t{a}, {axis})"))
                    ),
                    None => extent,
                };
                (axis, extent)
            })
            .collect::<Vec<_>>();
        self.emit_runtime_dim_sites(id, &extents);
        let check = format!(
            "chelis_movement_check_target(t{id}_movement, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {});",
            Self::tagged_shape_literal(ty)
        );
        match &active {
            Some(_) => self.line(&format!("if (t{id}_movement) {check}")),
            None => self.line(&check),
        }
    }

    /// The extent axis `axis` of node `id` declares where its activation
    /// holds in no row: the extent a claim states for it
    /// ([`Self::claimed_extents`], the unchecked claim's canonical value), as
    /// the evaluator's `inactive_unchecked_value` takes it, and otherwise
    /// `carried`, what its carrier reads. A binder is that extent only once
    /// declared; before that this axis is what declares it.
    fn inactive_extent(&self, id: usize, axis: usize, carried: &str) -> String {
        let claimed = self.claimed_extents.get(&id).and_then(|claims| {
            claims
                .iter()
                .find(|(claimed_axis, _)| *claimed_axis == axis)
                .and_then(|(_, canonical)| match canonical {
                    chelis_ir::axis_sources::CanonicalExtent::Binder(name)
                        if !self.declared_dim_names.contains(name) =>
                    {
                        None
                    }
                    canonical => Some(Self::canonical_extent_operand(canonical, axis)),
                })
        });
        claimed.unwrap_or_else(|| carried.to_string())
    }

    /// A claim's canonical extent as the C operand that reads it for `axis`:
    /// the declared binder, the declaring witness's scalar, or the resolved
    /// size.
    fn canonical_extent_operand(
        canonical: &chelis_ir::axis_sources::CanonicalExtent,
        axis: usize,
    ) -> String {
        match canonical {
            chelis_ir::axis_sources::CanonicalExtent::Binder(name) => extent_read(name),
            chelis_ir::axis_sources::CanonicalExtent::Witness(witness) => {
                Self::bound_c_expr(&RtDim::Node(0), &[*witness], witness.0, axis)
            }
            chelis_ir::axis_sources::CanonicalExtent::Resolved(extent) => extent.to_string(),
        }
    }

    /// Emit an affine movement operation's copy (`copy`, which reads
    /// `t{id}_movement`) and the plan's release. Under an activation the
    /// plan exists only where some row is active
    /// ([`Self::emit_affine_plan`]); elsewhere the result is zero-filled, a
    /// value of its declared type no check produced (spec/10 section 3.2).
    fn emit_movement_copy(&mut self, id: usize, copy: impl FnOnce(&mut Self)) {
        let gated = self.gate.is_some();
        if gated {
            self.line(&format!("if (t{id}_movement) {{"));
            self.indent += 1;
        }
        copy(self);
        self.line(&format!("chelis_movement_plan_release(t{id}_movement);"));
        if gated {
            self.indent -= 1;
            self.line("} else {");
            self.indent += 1;
            self.emit_zero_fill(id);
            self.indent -= 1;
            self.line("}");
        }
    }

    fn emit_pad(
        &mut self,
        id: usize,
        padding: &[(RtDim, RtDim)],
        fill: ScalarValue,
        inputs: &[NodeId],
        ty: &TensorType,
        _dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        let before = padding
            .iter()
            .enumerate()
            .map(|(axis, (n, _))| Self::bound_c_expr(n, inputs, a, axis))
            .collect::<Vec<_>>();
        let after = padding
            .iter()
            .enumerate()
            .map(|(axis, (_, n))| Self::bound_c_expr(n, inputs, a, axis))
            .collect::<Vec<_>>();
        self.emit_affine_bounds(&format!("t{id}_before"), &before);
        self.emit_affine_bounds(&format!("t{id}_after"), &after);
        self.emit_affine_plan(id, a, ty, "pad", &format!("t{id}_before, t{id}_after"));
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
        let fill = if ty.precision == Prim::Int64 {
            let value = fill.as_i64_exact().expect("verified i64 pad fill");
            format!(
                "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t){}));",
                Self::i64_c_literal(value)
            )
        } else {
            let literal = match ty.precision {
                Prim::F32 => format!("UINT32_C(0x{bits:08x})"),
                Prim::F64 => format!("UINT64_C(0x{bits:016x})"),
                _ => format!("UINT64_C({bits})"),
            };
            format!(
                "chelis_fill_scalar(t{id}_write_guard, chelis_scalar_from_bits({dtype}, {literal}));"
            )
        };
        self.emit_movement_copy(id, |emitter| {
            emitter.line(&fill);
            emitter.line(&format!(
                "for (int64_t i = 0; i < chelis_movement_count(t{id}_movement); i++) {{"
            ));
            emitter.indent += 1;
            emitter.line(&format!("int64_t dst = chelis_movement_index(t{id}_movement, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i));"));
            emitter.line(&format!(
                "(({et}*)t{id}_data)[dst] = ((const {et}*)t{a}_data)[i];"
            ));
            emitter.indent -= 1;
            emitter.line("}");
        });
    }

    fn emit_shrink(
        &mut self,
        id: usize,
        bounds: &[(RtDim, RtDim)],
        inputs: &[NodeId],
        ty: &TensorType,
        _dag: VerifiedDagView<'_>,
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
            .map(|(axis, (n, _))| Self::bound_c_expr(n, inputs, a, axis))
            .collect::<Vec<_>>();
        let end = bounds
            .iter()
            .enumerate()
            .map(|(axis, (_, n))| Self::bound_c_expr(n, inputs, a, axis))
            .collect::<Vec<_>>();
        self.emit_affine_bounds(&format!("t{id}_start"), &start);
        self.emit_affine_bounds(&format!("t{id}_end"), &end);
        self.emit_affine_plan(id, a, ty, "shrink", &format!("t{id}_start, t{id}_end"));
        // Preserve the existing runtime-bound empty-range rejection shared
        // with Eval. The metadata API also serves statically empty tensors;
        // this operation-level admission rule is separate from shape safety.
        //
        // It stays AFTER the plan, and therefore after any extent guard the
        // plan's site emits, because `spec/05-risc-primitives.md` section
        // 2.4.1 does NOT make an empty span a runtime-bound error: its closed
        // list is a negative bound, a shrink range overshoot, a non-positive
        // stride step and the two reshape errors. `spec/04-type-system.md`
        // section 4.7.2 makes only a NEGATIVE size an error. So an extent-0
        // result under a declared `tensor[2, f32]` is a CLAIM mismatch and the
        // guard reporting it is the conforming diagnostic; this rejection is
        // an operation-level admission rule the numbered spec does not require,
        // and the evaluator's matching rejection is what diverges from it
        // (chelis#1795). Round 1 of chelis#1397 read the order the other way
        // round and this comment records why that reading was wrong, so the
        // next reader does not re-derive it.
        for (axis, (start, end)) in bounds.iter().enumerate() {
            if start.node_input().is_some() || end.node_input().is_some() {
                let empty = self.gated_check(&format!(
                    "t{id}_start[{axis}].bits == t{id}_end[{axis}].bits"
                ));
                self.line(&format!("if ({empty}) {{ chelis_numeric_trap(\"numeric trap: domain in shrink at i64\"); }}"));
            }
        }
        self.emit_slot_wrapper(id, ty);
        self.emit_movement_copy(id, |emitter| {
            emitter.line(&format!(
                "for (int64_t i = 0; i < chelis_movement_count(t{id}_movement); i++) {{"
            ));
            emitter.indent += 1;
            emitter.line(&format!("int64_t src = chelis_movement_index(t{id}_movement, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i));"));
            emitter.line(&format!(
                "(({et}*)t{id}_data)[i] = ((const {et}*)t{a}_data)[src];"
            ));
            emitter.indent -= 1;
            emitter.line("}");
        });
    }

    fn emit_stride(
        &mut self,
        id: usize,
        strides: &[RtDim],
        inputs: &[NodeId],
        ty: &TensorType,
        _dag: VerifiedDagView<'_>,
    ) {
        let a = inputs[0].0;
        let et = Self::elem_type(ty);
        let steps = strides
            .iter()
            .enumerate()
            .map(|(axis, n)| Self::bound_c_expr(n, inputs, a, axis))
            .collect::<Vec<_>>();
        self.emit_affine_bounds(&format!("t{id}_steps"), &steps);
        self.emit_affine_plan(id, a, ty, "stride", &format!("t{id}_steps, NULL"));
        self.emit_slot_wrapper(id, ty);
        self.emit_movement_copy(id, |emitter| {
            emitter.line(&format!(
                "for (int64_t i = 0; i < chelis_movement_count(t{id}_movement); i++) {{"
            ));
            emitter.indent += 1;
            emitter.line(&format!("int64_t src = chelis_movement_index(t{id}_movement, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i));"));
            emitter.line(&format!(
                "(({et}*)t{id}_data)[i] = ((const {et}*)t{a}_data)[src];"
            ));
            emitter.indent -= 1;
            emitter.line("}");
        });
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
    // wrong on every cross-precision arm (f32<->f64, f32<->i32,
    // i32<->i64, ...). See
    // `docs/archive/investigations/cbackend_cast_memcpy_diagnosis.md`. The host
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
        let gated_source = self.gated(format!("(({src_et}*)t{a}_data)[idx]"), 0);
        let trap_conditions = checked_plan.map(|plan| {
            let probe = gated_source.clone();
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
        let pragma = if trapping {
            self.line(&format!("int64_t {first_trap_index} = INT64_MAX;"));
            Some(format!(
                "#pragma omp parallel for reduction(min:{first_trap_index})"
            ))
        } else if checked_plan.is_some() {
            Some("#pragma omp parallel for".to_string())
        } else {
            None
        };
        self.open_element_loop(id, "i", &format!("t{id}_size"), pragma.as_deref());
        self.line(&format!("int64_t idx = i * t{id}_input{a}_step;"));
        let src_elem = gated_source;
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
        self.close_element_loop();
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

// Marks the emitter puts around an extent name it renders as a C
// identifier (`EXTENT_READ`) and around the name a declaration introduces
// (`EXTENT_DECLARATION`), each closed by `EXTENT_END`, and which
// `settle_extent_marks` removes before the function is returned. They are C0
// control characters, which no text from a parsed program carries: span IDs
// may not contain them (spec/03 §1.1.1) and the Deep parser rejects one that
// does, the comment and format-string sanitizers escape them, string
// literals are written as octal escapes, and source identifiers are ASCII
// words. A graph built without the parser can still carry one in a name the
// emitter writes unescaped; settling refuses it unless it encloses one of the
// function's own extent names.
const EXTENT_READ: char = '\u{1}';
const EXTENT_DECLARATION: char = '\u{2}';
const EXTENT_END: char = '\u{3}';
const EXTENT_MARKS: [char; 3] = [EXTENT_READ, EXTENT_DECLARATION, EXTENT_END];

/// `name` as the emitter renders it where C reads the extent.
fn extent_read(name: &str) -> String {
    format!("{EXTENT_READ}{name}{EXTENT_END}")
}

/// `name` as the emitter writes it in its `int64_t <name> = ...;`
/// declaration.
fn extent_declaration(name: &str) -> String {
    format!("{EXTENT_DECLARATION}{name}{EXTENT_END}")
}

/// Remove the extent marks from `lines` and return the first name read, in
/// line order, before its declaration (chelis#2883). Within a line a
/// declaration's right-hand side is a read and comes first, as C evaluates
/// it. Only the emitter writes the marks, so an identifier that merely has
/// a name's spelling, such as a helper's parameter or a member access, is
/// never taken for a read.
///
/// Every mark must open, enclose a name `is_extent` accepts, and close before
/// the next mark. Anything else is a mark character the emitter did not
/// write, and the error says where it is.
fn settle_extent_marks(
    lines: &mut [String],
    is_extent: impl Fn(&str) -> bool,
) -> Result<Option<String>, String> {
    let mut declared = BTreeSet::new();
    let mut read_first = None;
    for (index, line) in lines.iter_mut().enumerate() {
        if !line.contains(EXTENT_MARKS) {
            continue;
        }
        let malformed = |what: String| format!("line {}: {what}", index + 1);
        let mut plain = String::with_capacity(line.len());
        let mut declaring = Vec::new();
        let mut rest = line.as_str();
        while let Some(start) = rest.find(EXTENT_MARKS) {
            plain.push_str(&rest[..start]);
            // Each mark is one byte, so the slices below stay on boundaries.
            let mark = char::from(rest.as_bytes()[start]);
            if mark == EXTENT_END {
                return Err(malformed("an end mark closes no extent mark".into()));
            }
            let after = &rest[start + 1..];
            let end = after
                .find(EXTENT_MARKS)
                .filter(|end| after.as_bytes()[*end] == EXTENT_END as u8)
                .ok_or_else(|| malformed("an extent mark is not closed".into()))?;
            let name = &after[..end];
            if !is_extent(name) {
                return Err(malformed(format!(
                    "the marked name `{}` is no extent of this function",
                    name.escape_debug()
                )));
            }
            plain.push_str(name);
            if mark == EXTENT_DECLARATION {
                declaring.push(name.to_string());
            } else if !declared.contains(name) && read_first.is_none() {
                read_first = Some(name.to_string());
            }
            rest = &after[end + 1..];
        }
        plain.push_str(rest);
        declared.extend(declaring);
        *line = plain;
    }
    Ok(read_first)
}

#[cfg(test)]
mod tests {
    use super::*;

    use chelis_ir::dag::{ComparisonKind, Dag, DimInfo, RiscOp, RtDim, TensorType};
    use chelis_types::types::Prim;

    /// Every name is an extent, for the order tests below.
    fn any_name(_: &str) -> bool {
        true
    }

    // chelis#2883: the order check reads the emitter's own marks, so a read
    // before the declaration is found and the marks are removed, while an
    // unmarked identifier with a name's spelling is never a read.
    #[test]
    fn extent_marks_order_reads_after_declarations_and_leave_plain_c() {
        let mut lines = vec![
            format!(
                "    if ((chelis_tensor_shape(t5, 0)) != {}) {{",
                extent_read("d5")
            ),
            format!(
                "    int64_t {} = ((int64_t*)t12_data)[0];",
                extent_declaration("d5")
            ),
        ];
        assert_eq!(
            settle_extent_marks(&mut lines, any_name),
            Ok(Some("d5".to_string()))
        );
        assert_eq!(
            lines,
            [
                "    if ((chelis_tensor_shape(t5, 0)) != d5) {",
                "    int64_t d5 = ((int64_t*)t12_data)[0];",
            ]
        );

        // Declared, then read, including by a later declaration's right-hand
        // side. The unmarked `index` and `value` are a helper's parameters,
        // `dtype` is a local and `.dtype` a member access, each sharing a
        // name's spelling.
        let mut lines = vec![
            "static uint64_t mix(uint64_t value, int64_t index) { return value + index; }"
                .to_string(),
            "    chelis_dtype dtype = chelis_tensor_read_view(inputs[0]).dtype;".to_string(),
            format!(
                "    int64_t {} = chelis_tensor_shape(inputs[0], 0);",
                extent_declaration("index")
            ),
            format!(
                "    int64_t {} = {};",
                extent_declaration("value"),
                extent_read("index")
            ),
            format!(
                "    chelis_tensor *t1 = chelis_alloc(1, (int64_t[]){{ {} }}, CHELIS_DTYPE_I64);",
                extent_read("value")
            ),
        ];
        assert_eq!(settle_extent_marks(&mut lines, any_name), Ok(None));
        assert!(lines.iter().all(|line| !line.contains(EXTENT_MARKS)));
        assert_eq!(lines[3], "    int64_t value = index;");

        // A declaration whose right-hand side reads a name declared only on
        // a later line reads it first.
        let mut lines = vec![
            format!(
                "    int64_t {} = {};",
                extent_declaration("n"),
                extent_read("d5")
            ),
            format!("    int64_t {} = 3;", extent_declaration("d5")),
        ];
        assert_eq!(
            settle_extent_marks(&mut lines, any_name),
            Ok(Some("d5".to_string()))
        );
    }

    // A mark character the emitter did not write is an error naming its
    // line, never a panic and never a silent rewrite: an unclosed mark, an
    // end mark that closes nothing, a mark opened inside another, and a
    // well-formed pair around a name that is no extent of the function.
    #[test]
    fn extent_marks_the_emitter_did_not_write_are_errors() {
        let is_d5 = |name: &str| name == "d5";
        let declared = format!("    int64_t {} = 3;", extent_declaration("d5"));
        for (stray, expected) in [
            (
                format!("// span: n_{EXTENT_READ}001"),
                "line 2: an extent mark is not closed",
            ),
            (
                format!("// span: zz{EXTENT_END}"),
                "line 2: an end mark closes no extent mark",
            ),
            (
                format!("// span: {EXTENT_READ}a{EXTENT_DECLARATION}d5{EXTENT_END}"),
                "line 2: an extent mark is not closed",
            ),
            (
                format!("// span: {EXTENT_READ}n{EXTENT_END}"),
                "line 2: the marked name `n` is no extent of this function",
            ),
        ] {
            let mut lines = vec![declared.clone(), stray.clone()];
            assert_eq!(
                settle_extent_marks(&mut lines, is_d5),
                Err(expected.to_string()),
                "{stray:?}"
            );
        }
        // The same marks the emitter writes, around its own name, settle.
        let mut lines = vec![declared, format!("    return {};", extent_read("d5"))];
        assert_eq!(settle_extent_marks(&mut lines, is_d5), Ok(None));
        assert_eq!(lines[1], "    return d5;");
    }

    // Every place the emitter renders an extent name for C to read marks it,
    // so the order check sees that read. Each assertion fails when its site
    // writes the bare name instead.
    #[test]
    fn every_extent_read_site_marks_the_name() {
        // A runtime-bound named dimension, as allocations and loops read it.
        assert_eq!(
            CEmitter::emit_dim_info(&DimInfo::Named("n".into(), None)),
            extent_read("n")
        );
        assert_eq!(
            CEmitter::emit_dim_info(&DimInfo::Named("n".into(), Some(3))),
            "3"
        );
        // A symbolic reshape bound.
        assert_eq!(
            CEmitter::bound_c_expr(&RtDim::Sym("n".into()), &[], 0, 0),
            extent_read("n")
        );
        // A claim's binder, as a local guard's operand and as the extent an
        // inactive claim-sized axis declares.
        assert_eq!(
            CEmitter::canonical_extent_operand(
                &chelis_ir::axis_sources::CanonicalExtent::Binder("n".into()),
                0
            ),
            extent_read("n")
        );
        assert_eq!(
            CEmitter::canonical_extent_operand(
                &chelis_ir::axis_sources::CanonicalExtent::Resolved(3),
                0
            ),
            "3"
        );
        // A second spelling of one extent reads the first, and a name
        // already declared is not declared again.
        let mut declared = chelis_unord::UnordSet::new();
        declared.insert("c".to_string());
        assert_eq!(
            CEmitter::runtime_dim_declarations(
                &["a".into(), "c".into(), "b".into()],
                "chelis_tensor_shape(t4, 0)",
                &mut declared
            ),
            [
                (
                    extent_declaration("a"),
                    "chelis_tensor_shape(t4, 0)".to_string()
                ),
                (extent_declaration("b"), extent_read("a")),
            ]
        );
    }

    fn emit_test_dag(dag: &Dag, name: &str) -> Result<String, Unsupported> {
        let verified = crate::testing::verified_dag(dag, crate::CodegenOptions::default())
            .expect("C emitter unit-test DAG must verify ownership");
        CEmitter::emit_dag(verified, name)
    }

    /// `x: [name]` doubled, with `span` on the addition.
    fn doubled_named(name: &str, span: &str) -> Dag {
        let ty = TensorType {
            dims: vec![DimInfo::Named(name.to_string(), None)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![x, x], ty, Some(span.to_string()));
        dag
    }

    // A graph built without the Deep parser can carry a mark character in a
    // span ID. The comment sanitizer escapes it, so it is no mark and the
    // span comment keeps its escaped form. A dimension name carrying a
    // malformed mark is refused rather than read as a mark or written into
    // the C. A well-formed mark around one of the function's own extent
    // names is not detected; chelis#2908 tracks validating such names where
    // the graph is built.
    #[test]
    fn a_mark_character_from_a_constructed_graph_is_never_a_mark() {
        for span in [
            format!("n_{EXTENT_READ}001"),
            format!("{EXTENT_READ}zz{EXTENT_END}"),
            format!("{EXTENT_READ}n{EXTENT_END}"),
            format!("{EXTENT_DECLARATION}n{EXTENT_END}"),
        ] {
            let c = emit_test_dag(&doubled_named("n", &span), "f")
                .unwrap_or_else(|error| panic!("{span:?}: {error}"));
            let escaped = chelis_ir::span_sanitize::sanitize_for_comment(&span);
            assert!(
                c.contains(&format!("// span: {escaped}\n")),
                "{span:?}:\n{c}"
            );
            assert!(!c.contains(EXTENT_MARKS), "{span:?}:\n{c}");
        }

        let error = emit_test_dag(&doubled_named(&format!("n{EXTENT_END}"), "s"), "f")
            .expect_err("a mark character in a dimension name");
        assert!(
            error
                .to_string()
                .contains("extent mark the emitter did not write"),
            "{error}"
        );
    }

    /// `add(neg(x), filled)`, where `neg` restamps `x`'s `m` axis as `n` and
    /// `filled` is a zero `expand`ed to a run-time count, the only site that
    /// produces `n`. With `declaration_first`, `filled` precedes `neg`.
    fn restamp_and_its_extent_site(declaration_first: bool) -> Dag {
        let named = |name: &str| TensorType {
            dims: vec![DimInfo::Named(name.to_string(), None)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            named("m"),
            None,
        );
        let restamp = |dag: &mut Dag| dag.add_node(decl, RiscOp::Neg, vec![x], named("n"), None);
        let filled = |dag: &mut Dag| {
            let count = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::Int64, 3.0),
                vec![],
                TensorType {
                    dims: vec![],
                    precision: Prim::Int64,
                },
                None,
            );
            let zero = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::F32, 0.0),
                vec![],
                scalar_f32(),
                None,
            );
            dag.add_node(
                decl,
                RiscOp::Expand {
                    axis: 0,
                    size: RtDim::Node(1),
                },
                vec![zero, count],
                named("n"),
                None,
            )
        };
        let (negated, zeros) = if declaration_first {
            let zeros = filled(&mut dag);
            (restamp(&mut dag), zeros)
        } else {
            let negated = restamp(&mut dag);
            (negated, filled(&mut dag))
        };
        let sum = dag.add_node(decl, RiscOp::Add, vec![negated, zeros], named("n"), None);
        dag.set_roots(vec![sum]);
        dag
    }

    // chelis#2883: no parsed program reaches the order refusal once a
    // helper's binders take the caller's extent, so a constructed graph
    // drives it. `neg`'s restamp guard compares against `n`, which only the
    // `expand` declares; placed before that site the guard reads `n` first and
    // the emitter refuses the function with the typed #1277 receipt rather
    // than writing C that does not compile. The same graph with the `expand`
    // first emits, declaring `n` before every read.
    #[test]
    fn an_extent_read_before_its_declaration_is_a_typed_refusal() {
        let error = emit_test_dag(&restamp_and_its_extent_site(false), "f")
            .expect_err("`neg` reads `n` before the `expand` declares it");
        let identity = error.identity();
        assert_eq!(
            identity.what,
            UnsupportedKind::Construct("extent `n` is rendered before it is declared".into()),
            "{error}"
        );
        assert_eq!(identity.context, "emitted function `f`");
        assert_eq!(identity.stage, Stage::Codegen("c"));
        assert_eq!(
            identity.disposition,
            chelis_types::unsupported::RejectionAuthorityKind::Unimplemented
        );
        assert_eq!(
            identity.tracking_issue.map(|issue| issue.number()),
            Some(1277)
        );

        let c = emit_test_dag(&restamp_and_its_extent_site(true), "f")
            .unwrap_or_else(|error| panic!("declaration first: {error}"));
        let declaration = c
            .find("int64_t n = ((int64_t*)t1_data)[0];")
            .unwrap_or_else(|| panic!("`n` is declared from the count:\n{c}"));
        let guard = c
            .find("if ((chelis_tensor_shape(t0, 0)) != n)")
            .unwrap_or_else(|| panic!("`neg`'s restamp guard reads `n`:\n{c}"));
        assert!(declaration < guard, "{c}");
        assert!(!c.contains(EXTENT_MARKS), "{c}");
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
        let direct_decl = direct.declare("test");
        let x = direct.add_node(
            direct_decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let out = direct.add_node(direct_decl, RiscOp::Abs, vec![x], ty.clone(), None);
        direct.set_roots(vec![out]);
        let c =
            emit_test_dag(&direct, "integer_abs").expect("direct integer abs has a typed C kernel");
        assert!(c.contains("chelis_int_abs_guard"));
        assert!(c.contains("numeric trap: overflow in abs at i64"));
        assert!(!c.contains("fabsf(__in_a_"));

        let mut fused = Dag::new();
        let fused_decl = fused.declare("test");
        let x = fused.add_node(
            fused_decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let out = fused.add_node(
            fused_decl,
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
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::Int64, 7.0),
            vec![],
            ty,
            None,
        );
        let fill = chelis_types::scalar_from_i64("pad", Prim::Int64, exact).unwrap();
        let padded = dag.add_node(
            decl,
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
            "the exact i64 fill must never pass through its rounded f64 image:\n{c}"
        );
    }

    /// chelis#1859's class at the integer-constant fill (chelis#2413 B5):
    /// `key_from_seed(i64::MIN)` makes an `i64::MIN` `Const` reach this
    /// fill, which must use the pad fill's exact-width signed spelling.
    ///
    /// Evidentiary status: REGRESSION TEST. At `b005bb19b` the fill spelled
    /// `(uint64_t)(int64_t)-9223372036854775808`, whose positive magnitude
    /// is no signed C literal.
    #[test]
    fn issue_1859_int64_constant_fill_emits_portable_signed_extremes() {
        for (value, spelling) in [
            (i64::MIN, "INT64_MIN"),
            (i64::MIN + 1, "-INT64_C(9223372036854775807)"),
            (-1, "-INT64_C(1)"),
            (0, "INT64_C(0)"),
            (i64::MAX, "INT64_C(9223372036854775807)"),
        ] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let constant = dag.add_node(
                decl,
                RiscOp::Const {
                    value: chelis_types::scalar_from_i64("const", Prim::Int64, value).unwrap(),
                },
                vec![],
                tensor_ty(&[], Prim::Int64),
                None,
            );
            dag.set_roots(vec![constant]);
            let c = emit_test_dag(&dag, "int64_constant").unwrap();
            assert!(
                c.contains(&format!(
                    "chelis_fill_scalar(t0_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t){spelling}));"
                )),
                "{c}"
            );
            assert!(!c.contains("9223372036854775808"), "{c}");
        }
    }

    #[test]
    fn issue_878_pad_emits_portable_signed_int64_extremes() {
        for (value, spelling) in [
            (i64::MIN, "INT64_MIN"),
            (-9_007_199_254_740_993, "-INT64_C(9007199254740993)"),
        ] {
            let ty = tensor_ty(&[1], Prim::Int64);
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let input = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::Int64, 7.0),
                vec![],
                ty.clone(),
                None,
            );
            let fill = chelis_types::scalar_from_i64("pad", Prim::Int64, value).unwrap();
            let padded = dag.add_node(
                decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_tensor_elementwise_index_step_for_shape"));
        assert!(c.contains("i * t2_input0_step"));
        assert!(c.contains("+"));
    }

    #[test]
    fn neg_emits_unary_minus() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Neg, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        // The slow (non-contiguous) path emits a typed pointer cast then negates.
        assert!(c.contains("((float*)t0_data)[idx]"));
    }

    #[test]
    fn exp_emits_expf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Exp, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_cr_expf("));
    }

    #[test]
    fn log_emits_logf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Log, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_cr_logf("));
    }

    #[test]
    fn sin_emits_sinf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Sin, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_cr_sinf("));
    }

    #[test]
    fn sqrt_emits_sqrtf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 4.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Sqrt, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("sqrtf("));
    }

    #[test]
    fn cmplt_emits_canonical_bool8() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Compare(ComparisonKind::CmpLt),
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("#pragma omp parallel for"));
    }

    #[test]
    fn sum_emits_reduction_loop() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let copy = dag.add_node(decl, RiscOp::Copy, vec![x], vec_f32(4), None);
        let disposable = dag.add_node(decl, RiscOp::Copy, vec![x], vec_f32(4), None);
        dag.add_node(decl, RiscOp::Drop, vec![disposable], vec_f32(4), None);
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
        let decl = dag.declare("test");
        let first_ty = tensor_ty(&[2, 2], Prim::F32);
        let first = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            first_ty.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Neg, vec![first], first_ty, None);
        let output = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let borrowed = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(decl, RiscOp::Drop, vec![borrowed], vec_f32(4), None);
        let output = dag.add_node(decl, RiscOp::Copy, vec![borrowed], vec_f32(4), None);
        dag.add_root(output);

        let c = emit_test_dag(&dag, "test_borrowed_drop").unwrap();

        assert!(!c.contains("chelis_tensor_release(t0);"), "{c}");
    }

    #[test]
    fn max_reduce_emits_nan_propagating_max() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::MaxReduce { axis: 0 },
            vec![a],
            scalar_f32(),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        // [05-OP-12] selects the first NaN, otherwise the first strictly
        // greatest stored value. Source-index selection preserves the exact
        // stored representation, including NaN payloads and signed zero.
        assert!(
            c.contains("isnan(candidate) || candidate > best_value"),
            "max_reduce must select the first NaN or strict greater value:\n{c}"
        );
        assert!(
            c.contains("((float*)t1_data)[outer] = ((const float*)t0_data)[best_src];"),
            "max_reduce must copy the selected source representation:\n{c}"
        );
        assert!(!c.contains("fmaxf("), "{c}");
        assert!(
            c.contains("numeric trap: domain in max_reduce at f32"),
            "{c}"
        );
    }

    #[test]
    fn reduce_window_max_min_select_first_nan_and_stored_representation() {
        for (reducer, comparison, forbidden) in [
            (ReduceWindowKind::Max, "candidate > best_value", "fmaxf("),
            (ReduceWindowKind::Min, "candidate < best_value", "fminf("),
        ] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let a = dag.add_node(
                decl,
                RiscOp::synth_const(vec_f32(4).precision, 1.0),
                vec![],
                vec_f32(4),
                None,
            );
            dag.add_node(
                decl,
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
                c.contains(&format!("isnan(candidate) || {comparison}")),
                "reduce_window {reducer:?} must select the first NaN or strict extremum:\n{c}"
            );
            assert!(
                c.contains("((float*)t1_data)[outer] = ((const float*)t0_data)[best_src];"),
                "reduce_window {reducer:?} must copy the selected source representation:\n{c}"
            );
            assert!(!c.contains(forbidden), "{c}");
        }
    }

    #[test]
    fn mul_emits_star_op() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 4.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Mul, vec![a, b], scalar_f32(), None);
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
            let decl = dag.declare("test");
            let a = dag.add_node(
                decl,
                RiscOp::synth_const(scalar_f32().precision, 1.0),
                vec![],
                scalar_f32(),
                None,
            );
            let b = dag.add_node(
                decl,
                RiscOp::synth_const(scalar_f32().precision, 2.0),
                vec![],
                scalar_f32(),
                None,
            );
            dag.add_node(decl, op, vec![a, b], scalar_f32(), None);
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(6).precision, 1.0),
            vec![],
            vec_f32(6),
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Permute { axes: vec![1, 0] },
            vec![a],
            mat_f32(3, 2),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_tensor_permute_plan(t0,"));
        assert!(c.contains("chelis_movement_index(t1_movement,"));
        assert!(c.contains("t1_axes[2] = { chelis_scalar_from_bits(CHELIS_DTYPE_I64, (1)), chelis_scalar_from_bits(CHELIS_DTYPE_I64, (0)) }"));
        assert!(!c.contains("t1->strides[0] ="));
    }

    #[test]
    fn expand_materializes_broadcast_coordinates() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(1).precision, 1.0),
            vec![],
            vec_f32(1),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            vec_f32(4),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_tensor_expand_plan(t0,"));
        assert!(c.contains("CHELIS_MOVEMENT_EXPAND"));
        assert!(c.contains("chelis_movement_index(t1_movement,"));
        assert!(!c.contains("t1->strides[0] ="));
    }

    #[test]
    fn store_is_alias() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Store { name: "out".into() },
            vec![a],
            scalar_f32(),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("memcpy(t1_data, t0_data, (size_t)t1_byte_capacity); /* store: out */"));
    }

    #[test]
    fn bitwise_shift_loops_preserve_first_negative_count() {
        for kind in [
            chelis_types::BitwiseKind::ShiftLeft,
            chelis_types::BitwiseKind::ShiftRight,
        ] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let ty = tensor_ty(&[2], Prim::Int64);
            let inputs = ["values", "counts"].map(|name| {
                dag.add_node(
                    decl,
                    RiscOp::Load { name: name.into() },
                    vec![],
                    ty.clone(),
                    None,
                )
            });
            dag.add_node(decl, RiscOp::Bitwise(kind), inputs.to_vec(), ty, None);
            let c = emit_test_dag(&dag, "test_fn").expect("emit shifts");
            assert!(c.contains(&format!("chelis_int_{}", kind.name())));
            // Distinct negative counts render distinct errors. Both contiguous
            // and strided paths must visit elements in logical order; macOS
            // without OpenMP cannot expose this race through execution alone.
            assert!(!c.contains("#pragma omp"), "{kind:?}: {c}");
        }
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
            let decl = dag.declare("test");
            let a = dag.add_node(
                decl,
                RiscOp::synth_const(scalar_f32().precision, 1.0),
                vec![],
                scalar_f32(),
                None,
            );
            let dst_ty = TensorType {
                dims: vec![],
                precision,
            };
            dag.add_node(decl, op, vec![a], dst_ty, None);
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
                let decl = dag.declare("test");
                let source_ty = tensor_ty(&[2], source);
                let input = dag.add_node(
                    decl,
                    RiscOp::Load {
                        name: "input".into(),
                    },
                    vec![],
                    source_ty,
                    None,
                );
                let output = dag.add_node(
                    decl,
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
        // See `docs/archive/investigations/cbackend_cast_memcpy_diagnosis.md`.
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
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
            decl,
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
            c.contains(
                "((double*)t1_data)[i] = __chelis_nan_f64((double)(((float*)t0_data)[idx]));"
            ),
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(6),
            None,
        );
        let s = dag.add_node(
            decl,
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2)],
            },
            vec![x],
            vec_f32(3),
            None,
        );
        dag.add_node(decl, RiscOp::Realize, vec![s], vec_f32(3), None);

        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_tensor *t2 = chelis_alloc("));
        assert!(c.contains("i * t2_input1_step"));
        assert!(!c.contains("chelis_alloc_view"));
    }

    #[test]
    fn load_emits_input_reference() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let x0 = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let x1 = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![x0, x1], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("if (n_in != 1)"));
        assert!(c.contains("chelis_tensor *t0 = inputs[0];"));
        assert!(c.contains("chelis_tensor *t1 = inputs[0];"));
    }

    #[test]
    fn input_labels_follow_first_load_occurrence() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let b0 = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let b1 = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(3).precision, 1.0),
            vec![],
            vec_f32(3),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::zero_pad(Prim::F32, vec![(RtDim::Lit(1), RtDim::Lit(1))]),
            vec![a],
            vec_f32(5),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains(
            "chelis_fill_scalar(t1_write_guard, chelis_scalar_from_bits(CHELIS_DTYPE_F32,"
        ));
        assert!(c.contains("chelis_movement_index(t1_movement,"));
    }

    #[test]
    fn shrink_emits_offset_copy() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(5).precision, 1.0),
            vec![],
            vec_f32(5),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::Lit(4))],
            },
            vec![a],
            vec_f32(3),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_movement_index(t1_movement,"));
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(5).precision, 1.0),
            vec![],
            vec_f32(5),
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2)],
            },
            vec![a],
            vec_f32(2),
            None,
        );
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("chelis_movement_index(t1_movement,"));
        assert!(!c.contains("t1->strides[0] ="));
    }

    #[test]
    fn add_then_mul_chains() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        let d = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Mul, vec![c, d], scalar_f32(), None);
        let code = emit_test_dag(&dag, "test_fn").unwrap();
        // t2 is add result, t4 is mul result
        assert!(code.contains("t2_data"));
        assert!(code.contains("t4_data"));
    }

    #[test]
    fn vector_add_uses_correct_shape() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32(4), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("(int64_t[]){ 4 }"));
    }

    #[test]
    fn sum_then_neg_chains() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(3).precision, 2.0),
            vec![],
            vec_f32(3),
            None,
        );
        let s = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![a],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Neg, vec![s], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("__sum_level_"));
        assert!(!c.contains("chelis_sum_f32("));
        // The slow (non-contiguous) path emits a typed pointer cast then negates.
        assert!(c.contains("((float*)t1_data)[idx]"));
    }

    #[test]
    fn load_is_borrowed_not_freed() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Compare(ComparisonKind::CmpLt),
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
    fn path_sensitive_uniform_reads_uint8_bool() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let rank0 = |precision| TensorType {
            dims: vec![],
            precision,
        };
        let template = dag.add_node(
            decl,
            RiscOp::Load {
                name: "template".into(),
            },
            vec![],
            vec_f32(2),
            None,
        );
        let activation = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::Bool, 1.0),
            vec![],
            rank0(Prim::Bool),
            None,
        );
        let seed = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::Int64, 11.0),
            vec![],
            rank0(Prim::Int64),
            None,
        );
        let low = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 0.0),
            vec![],
            rank0(Prim::F32),
            None,
        );
        let high = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            rank0(Prim::F32),
            None,
        );
        let key = dag.add_node(
            decl,
            RiscOp::KeyFromSeed,
            vec![seed],
            rank0(Prim::Key),
            None,
        );
        let draw = dag.add_node(
            chelis_ir::dag::Owner::new(decl, Some(activation)),
            RiscOp::UniformLike,
            vec![template, low, high, key],
            vec_f32(2),
            None,
        );
        dag.add_root(draw);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        // chelis#1308 stores Bool tensors as one uint8 per element; the
        // draw gate must read the predicate at that width. A `(float*)`
        // read of the one-byte allocation is out of bounds and
        // platform-divergent (the Linux-only RNG parity break on PR #1302).
        let gate = format!("(((const uint8_t*)t{}_data)[0] != 0)", activation.0);
        assert!(c.contains(&gate), "{c}");
        assert!(!c.contains(&format!("((float*)t{}_data)[0] != 0.0f", activation.0)));
        assert!(!c.contains(&format!("((bool*)t{}_data)", activation.0)));
        // An inactive draw neither validates its bounds nor samples
        // (chelis#2410): the bound check sits inside the activation's gate,
        // and each element is zeroed before the sampler when inactive.
        let draw_gate = format!("if ({gate}) {{");
        assert!(
            guarded_block(&c, &draw_gate).contains("numeric trap: domain in uniform_like at f32"),
            "{c}"
        );
        assert!(c.contains(&format!("if (!({gate})) {{")), "{c}");
    }

    /// The body of the one block `header` opens, up to its matching brace.
    fn guarded_block<'a>(c: &'a str, header: &str) -> &'a str {
        assert_eq!(
            c.matches(header).count(),
            1,
            "{header} opens one block:\n{c}"
        );
        let start = c.find(header).unwrap() + header.len();
        let mut depth = 1usize;
        for (offset, byte) in c[start..].bytes().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &c[start..start + offset];
                    }
                }
                _ => {}
            }
        }
        panic!("{header} is never closed:\n{c}");
    }

    #[test]
    fn int64_const_does_not_panic() {
        // Regression: dtype_macro used to panic for i64 tensors.
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
            "generated C must carry the exact i64 tag and bits through chelis_fill_scalar"
        );
    }

    #[test]
    fn f64_const_uses_exact_tagged_fill_and_dtype_macro() {
        // v0.2.3: f64 tensors are a first-class precision. The const path must
        // emit CHELIS_DTYPE_F64 and chelis_fill_f64 (not chelis_fill_f32, which would
        // silently downcast).
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F64,
        };
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(ty.precision, 1.0),
            vec![],
            ty.clone(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(ty.precision, 2.0),
            vec![],
            ty.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], ty, None);
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
        // Emitting chelis_cr_expf() would silently truncate to float and lose precision.
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F64,
        };
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(ty.precision, 0.0),
            vec![],
            ty.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Exp, vec![a], ty, None);
        let verified = crate::testing::verified_dag(&dag, crate::CodegenOptions::default())
            .expect("f64 exp test DAG must verify ownership");
        let c =
            CEmitter::emit_dag_with_options(verified, "test_fn", crate::CodegenOptions::default())
                .unwrap();
        assert!(
            c.contains("chelis_cr_exp("),
            "f64 exp must emit chelis_cr_exp(, not chelis_cr_expf(:\n{c}"
        );
        assert!(
            !c.contains("chelis_cr_expf("),
            "f64 exp must not emit the f32 chelis_cr_expf( kernel:\n{c}"
        );
    }

    #[test]
    fn int64_add_does_not_panic() {
        // Regression: binary elementwise over i64 tensors used to panic.
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int64,
        };
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(ty.precision, 1.0),
            vec![],
            ty.clone(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(ty.precision, 2.0),
            vec![],
            ty.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], ty, None);
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            decl,
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
            decl,
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
            decl,
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            decl,
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
            decl,
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
            decl,
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32(4), None);
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
        let decl = dag.declare("test");
        // Expand materializes the repeated value into a contiguous [4] tensor.
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(1).precision, 1.0),
            vec![],
            vec_f32(1),
            None,
        );
        let a_exp = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a_exp, b], vec_f32(4), None);
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 5.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), None);
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            decl,
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
        dag.add_node(
            decl,
            RiscOp::FusedElem { ops },
            vec![a, b],
            vec_f32(4),
            None,
        );
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let scale = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let ops = vec![FusedStep {
            op: FusedStepOp::Mul,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        }];
        dag.add_node(
            decl,
            RiscOp::FusedElem { ops },
            vec![x, scale],
            vec_f32(4),
            None,
        );

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
        let decl = dag.declare("test");
        let values = dag.add_node(
            decl,
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );
        let indices = dag.add_node(
            decl,
            RiscOp::synth_const(tensor_ty(&[3], Prim::Int32).precision, 0.0),
            vec![],
            tensor_ty(&[3], Prim::Int32),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Gather {
                axis: 0,
                batch_rank: 0,
            },
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
        let decl = dag.declare("test");
        let values = dag.add_node(
            decl,
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[50000, 1024], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            decl,
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            tensor_ty(&[128], Prim::Int32),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Gather {
                axis: 0,
                batch_rank: 0,
            },
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
        let decl = dag.declare("test");
        let values = dag.add_node(
            decl,
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(3).precision, 0.0),
            vec![],
            vec_f32(3),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Gather {
                axis: 0,
                batch_rank: 0,
            },
            vec![values, indices],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );

        let error = match chelis_ir::ownership::lower_dag_ownership(dag) {
            Err(error) => error,
            Ok(_) => panic!("float sparse indices must not become verified backend input"),
        };
        assert!(error.to_string().contains("requires i32/i64 indices"));
    }

    /// chelis#3047: an f16 or bf16 scatter-add widens the stored element and
    /// the update to f32, adds there, and narrows once; adding the `uint16_t`
    /// encodings turns 2 + 2 into inf. Without the widening the emission-time
    /// check in `assert_nan_finalized` rejects the node.
    #[test]
    fn reduced_float_scatter_add_adds_at_f32_and_narrows_once() {
        for (precision, load, store) in [
            (Prim::F16, "chelis_f16_to_f32", "chelis_f32_to_f16"),
            (Prim::Bf16, "chelis_bf16_to_f32", "chelis_f32_to_bf16"),
        ] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let target = dag.add_node(
                decl,
                RiscOp::Load {
                    name: "target".into(),
                },
                vec![],
                tensor_ty(&[4], precision),
                None,
            );
            let indices = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::Int64, 0.0),
                vec![],
                tensor_ty(&[3], Prim::Int64),
                None,
            );
            let updates = dag.add_node(
                decl,
                RiscOp::Load {
                    name: "updates".into(),
                },
                vec![],
                tensor_ty(&[3], precision),
                None,
            );
            dag.add_node(
                decl,
                RiscOp::ScatterAdd {
                    axis: 0,
                    batch_rank: 0,
                },
                vec![target, indices, updates],
                tensor_ty(&[4], precision),
                None,
            );

            let c = emit_test_dag(&dag, "test_fn").unwrap();

            assert!(
                c.contains(&format!(
                    "t3_scatter_level[__pair] = (__pair_right < __leaves) ? {store}({load}(t3_scatter_level[__pair_left]) + {load}(t3_scatter_level[__pair_right])) : t3_scatter_level[__pair_left];"
                )),
                "{}: {c}",
                precision.name()
            );
            assert!(
                c.contains(&format!(
                    "((uint16_t*)t3_data)[__d] = {store}({load}(t3_scatter_level[0]));"
                )),
                "{}: {c}",
                precision.name()
            );
            assert!(
                !c.contains("t3_scatter_level[__pair_left] + t3_scatter_level"),
                "{}",
                precision.name()
            );
        }
    }

    #[test]
    fn sparse_scatter_add_uses_typed_indices_payload_and_copy_size() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let target = dag.add_node(
            decl,
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );
        let indices = dag.add_node(
            decl,
            RiscOp::synth_const(tensor_ty(&[3], Prim::Int64).precision, 0.0),
            vec![],
            tensor_ty(&[3], Prim::Int64),
            None,
        );
        let updates = dag.add_node(
            decl,
            RiscOp::synth_const(tensor_ty(&[3, 2], Prim::F64).precision, 1.0),
            vec![],
            tensor_ty(&[3, 2], Prim::F64),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::ScatterAdd {
                axis: 0,
                batch_rank: 0,
            },
            vec![target, indices, updates],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );

        let c = emit_test_dag(&dag, "test_fn").unwrap();

        assert!(c.contains("((const int64_t*)t1_data)[t3_index_slot]"));
        assert!(c.contains("memcpy(t3_data, t0_data, (size_t)t3_byte_capacity);"));
        // [05-OP-33]: the base leaf and the destination's updates in update
        // order, folded by the canonical balanced tree and finalized once;
        // no update is added straight into the destination.
        for fragment in [
            "t3_scatter_level[0] = ((double*)t3_data)[__d];",
            "t3_scatter_level[__k - __first + 1] = ((const double*)t2_data)[t3_scatter_order[__k]];",
            "t3_scatter_level[__pair] = (__pair_right < __leaves) ? t3_scatter_level[__pair_left] + t3_scatter_level[__pair_right] : t3_scatter_level[__pair_left];",
            "((double*)t3_data)[__d] = __chelis_nan_f64(t3_scatter_level[0]);",
        ] {
            assert!(c.contains(fragment), "missing {fragment:?}: {c}");
        }
        assert!(!c.contains("((double*)t3_data)[t3_base_index]"), "{c}");
    }

    #[test]
    fn sparse_scatter_add_rejects_float_indices_at_verified_boundary() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let target = dag.add_node(
            decl,
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(3).precision, 0.0),
            vec![],
            vec_f32(3),
            None,
        );
        let updates = dag.add_node(
            decl,
            RiscOp::synth_const(tensor_ty(&[3, 2], Prim::F32).precision, 1.0),
            vec![],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::ScatterAdd {
                axis: 0,
                batch_rank: 0,
            },
            vec![target, indices, updates],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );

        let error = match chelis_ir::ownership::lower_dag_ownership(dag) {
            Err(error) => error,
            Ok(_) => panic!("float sparse indices must not become verified backend input"),
        };
        assert!(error.to_string().contains("requires i32/i64 indices"));
    }

    #[test]
    fn target_fused_in_place_consumes_the_verified_reuse_proof() {
        // The shared planner turns the DAG's liveness hint into the sealed
        // provenance/capacity proof consumed by C emission.
        use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let owned = dag.add_node(decl, RiscOp::Copy, vec![x], vec_f32(4), None);
        let scale = dag.add_node(
            decl,
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
            decl,
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let scale = dag.add_node(
            decl,
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
            decl,
            RiscOp::FusedElem { ops },
            vec![x, scale],
            vec_f32(4),
            None,
        );
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_ty(&[2, 2], Prim::F32),
            None,
        );
        let flat = dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(4)],
            },
            vec![x],
            vec_f32(4),
            None,
        );
        let scale = dag.add_node(
            decl,
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
            decl,
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let scale = dag.add_node(
            decl,
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
            decl,
            RiscOp::FusedElem { ops },
            vec![x, scale],
            vec_f32(4),
            None,
        );
        dag.set_reusable_input(fused, x);
        let other = dag.add_node(decl, RiscOp::Neg, vec![x], vec_f32(4), None);
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Cos, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(
            c.contains("chelis_cr_cosf("),
            "expected chelis_cr_cosf( in:\n{c}"
        );
    }

    #[test]
    fn tan_emits_tanf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Tan, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(
            c.contains("chelis_cr_tanf("),
            "expected chelis_cr_tanf( in:\n{c}"
        );
    }

    #[test]
    fn atan_emits_atanf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Atan, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(
            c.contains("chelis_cr_atanf("),
            "expected chelis_cr_atanf( in:\n{c}"
        );
    }

    #[test]
    fn abs_emits_fabsf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, -2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Abs, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("fabsf("), "expected fabsf( in:\n{c}");
    }

    #[test]
    fn floor_emits_floorf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.7),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Floor, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("floorf("), "expected floorf( in:\n{c}");
    }

    #[test]
    fn ceil_emits_ceilf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.3),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Ceil, vec![a], scalar_f32(), None);
        let c = emit_test_dag(&dag, "test_fn").unwrap();
        assert!(c.contains("ceilf("), "expected ceilf( in:\n{c}");
    }

    #[test]
    fn round_emits_rintf() {
        // `round` lowers to `rintf` (round-to-nearest-ties-to-even under
        // the default rounding mode), NOT `roundf` (ties-away-from-zero).
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.5),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Round, vec![a], scalar_f32(), None);
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, std::f64::consts::PI / 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let out = dag.add_node(decl, RiscOp::Cos, vec![x], scalar_f32(), None);
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, -2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let out = dag.add_node(decl, RiscOp::Abs, vec![x], scalar_f32(), None);
        dag.add_root(out);
        let results = chelis_ir::eval::eval_scalar(&dag, &chelis_unord::UnordMap::new());
        let val = results[&out] as f32;
        assert!(
            (val - 2.0_f32).abs() < 1e-4,
            "abs(-2.0) should be 2.0, got {val}"
        );
    }

    /// chelis#1788. The per-scope rename's two naming rules, checked on the
    /// rewritten graph rather than on emitted text, so a change to the emitter
    /// cannot make this pass for the wrong reason.
    ///
    /// Three roots. `seq` is declared by two of them, so the later scope must
    /// be renamed; a third root independently declares `seq__s1`, so the name
    /// the rename would mint is already taken and it must step past it. The
    /// occupant keeps its own spelling, and the untouched binder `batch` and
    /// the first scope's `seq` are unchanged.
    #[test]
    fn scope_rename_mints_a_fresh_identity_past_one_the_graph_declares() {
        use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
        let ty = |dims: Vec<DimInfo>| TensorType {
            dims,
            precision: chelis_types::types::Prim::F32,
        };
        let named = |name: &str| DimInfo::Named(name.into(), None);

        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![named("seq")]),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            ty(vec![named("batch"), named("seq")]),
            None,
        );
        let w = dag.add_node(
            decl,
            RiscOp::Load { name: "w".into() },
            vec![],
            ty(vec![named("seq__s1")]),
            None,
        );
        let from_x = dag.add_node(decl, RiscOp::Neg, vec![x], ty(vec![named("seq")]), None);
        let from_y = dag.add_node(
            decl,
            RiscOp::Neg,
            vec![y],
            ty(vec![named("batch"), named("seq")]),
            None,
        );
        let from_w = dag.add_node(decl, RiscOp::Neg, vec![w], ty(vec![named("seq__s1")]), None);
        dag.add_root(from_x);
        dag.add_root(from_y);
        dag.add_root(from_w);

        let renamed = CEmitter::rename_scoped_dims(dag);
        let dims = |id: NodeId| {
            renamed
                .get(id)
                .expect("node survives the rename")
                .output_type
                .dims
                .clone()
        };
        assert_eq!(
            dims(x),
            vec![named("seq")],
            "the first scope keeps the name"
        );
        assert_eq!(
            dims(y),
            vec![named("batch"), named("seq__s2")],
            "the later scope steps past the taken `seq__s1`"
        );
        assert_eq!(
            dims(w),
            vec![named("seq__s1")],
            "and the occupant is untouched, since it declares the name only once"
        );
        assert_eq!(dims(from_y), vec![named("batch"), named("seq__s2")]);
    }

    /// The negative twin. A binder each of whose declarations sits in ONE scope
    /// is renamed by nothing, so a single-root graph and an unshared name come
    /// out byte-identical. Without this, a pass that renamed everything would
    /// satisfy the row above.
    #[test]
    fn scope_rename_leaves_a_binder_no_second_scope_declares_alone() {
        use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
        let ty = |dims: Vec<DimInfo>| TensorType {
            dims,
            precision: chelis_types::types::Prim::F32,
        };
        let named = |name: &str| DimInfo::Named(name.into(), None);

        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![named("seq")]),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            ty(vec![named("batch")]),
            None,
        );
        let from_x = dag.add_node(decl, RiscOp::Neg, vec![x], ty(vec![named("seq")]), None);
        let from_y = dag.add_node(decl, RiscOp::Neg, vec![y], ty(vec![named("batch")]), None);
        dag.add_root(from_x);
        dag.add_root(from_y);

        let renamed = CEmitter::rename_scoped_dims(dag);
        assert_eq!(
            renamed
                .get(x)
                .expect("node survives")
                .output_type
                .dims
                .clone(),
            vec![named("seq")]
        );
        assert_eq!(
            renamed
                .get(y)
                .expect("node survives")
                .output_type
                .dims
                .clone(),
            vec![named("batch")]
        );
    }
}
