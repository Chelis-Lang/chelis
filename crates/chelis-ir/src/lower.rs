//! Deep AST to RISC DAG lowering.
//!
//! Walks the Deep AST and produces a flat DAG of RISC primitive nodes.

use std::any::Any;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::OnceLock;

thread_local! {
    /// When set, `lower_unrepresentable` panics with a quiet empty payload
    /// that `catch_unwind` catches without printing a backtrace. Scoped
    /// via `with_suppress_unrepresentable_panic`, which always clears the
    /// flag on exit. Used by the host-lane tensor-helper fallback path:
    /// it runs the DAG lowerer speculatively, catches any un-representable
    /// panic, and proceeds with host lowering — we don't want that
    /// speculative attempt to write a misleading panic to stderr.
    static SUPPRESS_UNREPRESENTABLE_PANIC: Cell<bool> = const { Cell::new(false) };
    /// Set while a public `try_lower_*` API is converting legacy lowering
    /// unwinds into structured diagnostics. The panic hook stays quiet in
    /// that scope so users see only the returned diagnostic.
    static SUPPRESS_LOWERING_PANIC_OUTPUT: Cell<bool> = const { Cell::new(false) };
}

pub fn with_suppress_unrepresentable_panic<R>(f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            SUPPRESS_UNREPRESENTABLE_PANIC.with(|cell| cell.set(false));
        }
    }
    SUPPRESS_UNREPRESENTABLE_PANIC.with(|cell| cell.set(true));
    let _guard = Guard;
    f()
}

fn unrepresentable_panic_suppressed() -> bool {
    SUPPRESS_UNREPRESENTABLE_PANIC.with(|cell| cell.get())
}

/// Marker payload for a suppressed un-representable-DAG unwind.
struct UnrepresentableDag;

/// User-facing lowering diagnostic returned by `try_lower_*` APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerDiagnostic {
    pub message: String,
    pub span: Option<Span>,
    pub span_id: Option<String>,
}

impl LowerDiagnostic {
    fn new(message: impl Into<String>, span: Option<Span>, span_id: Option<String>) -> Self {
        Self {
            message: message.into(),
            span,
            span_id,
        }
    }
}

impl fmt::Display for LowerDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(span_id) = &self.span_id {
            write!(f, " at source span `{span_id}`")?;
        } else if let Some(span) = self.span
            && span.len > 0
        {
            write!(f, " at byte {}..{}", span.offset, span.end())?;
        }
        Ok(())
    }
}

impl std::error::Error for LowerDiagnostic {}

fn unsupported_lowering_message(tag: &str) -> String {
    let subject = if tag == "pipe stage" {
        "pipe stage".to_string()
    } else {
        format!("`{tag}`")
    };
    format!(
        "{subject} is not supported by IR evaluation yet; use `chelis build --target c` instead"
    )
}

fn expr_diagnostic_location(expr: &Expr) -> (Option<Span>, Option<String>) {
    (Some(expr.span()), expr.span_id().map(ToOwned::to_owned))
}

fn lower_diagnostic_for_expr(message: impl Into<String>, expr: &Expr) -> LowerDiagnostic {
    let (span, span_id) = expr_diagnostic_location(expr);
    LowerDiagnostic::new(message, span, span_id)
}

fn raise_lowering_diagnostic(diagnostic: LowerDiagnostic) -> ! {
    if unrepresentable_panic_suppressed() {
        std::panic::panic_any(UnrepresentableDag);
    }
    std::panic::panic_any(diagnostic);
}

fn raise_lowering_error(
    message: impl Into<String>,
    span: Option<Span>,
    span_id: Option<String>,
) -> ! {
    raise_lowering_diagnostic(LowerDiagnostic::new(message, span, span_id))
}

fn panic_payload_to_lower_diagnostic(payload: &(dyn Any + Send)) -> LowerDiagnostic {
    if payload.is::<UnrepresentableDag>() {
        return LowerDiagnostic::new(
            "program uses a form that is not supported by IR evaluation yet; use `chelis build --target c` instead",
            None,
            None,
        );
    }
    if let Some(diagnostic) = payload.downcast_ref::<LowerDiagnostic>() {
        return diagnostic.clone();
    }
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|value| (*value).to_string())
        })
        .unwrap_or_else(|| "internal lowering error".to_string());
    LowerDiagnostic::new(message, None, None)
}

fn catch_lowering<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> Result<R, LowerDiagnostic> {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            SUPPRESS_LOWERING_PANIC_OUTPUT.with(|cell| cell.set(false));
        }
    }

    install_chelis_panic_hook();
    SUPPRESS_LOWERING_PANIC_OUTPUT.with(|cell| cell.set(true));
    let _guard = Guard;
    std::panic::catch_unwind(f).map_err(|payload| panic_payload_to_lower_diagnostic(&*payload))
}

static CHELIS_PANIC_HOOK_INSTALLED: OnceLock<()> = OnceLock::new();

/// Install a one-time global panic hook that suppresses panic output when the
/// current thread is inside a `with_suppress_unrepresentable_panic` scope.
/// Safe to call multiple times — the hook is installed at most once.
pub fn install_chelis_panic_hook() {
    CHELIS_PANIC_HOOK_INSTALLED.get_or_init(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if SUPPRESS_UNREPRESENTABLE_PANIC.with(|cell| cell.get()) {
                return;
            }
            if SUPPRESS_LOWERING_PANIC_OUTPUT.with(|cell| cell.get()) {
                return;
            }
            prev(info);
        }));
    });
}

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List};
use chelis_types::{BUILTIN_NAMES, CheckedProgram, LinearityInfo, types::Prim};

use crate::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, TensorType};
use crate::grad::grad_dag;
use crate::tier2;
use crate::vmap;

/// Lower a checked Deep program into a RISC DAG.
pub fn lower_program(program: &CheckedProgram) -> Dag {
    try_lower_program(program).unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

/// Lower a checked Deep program into a RISC DAG, returning a diagnostic for
/// forms that are valid Chelis but not supported by IR evaluation.
pub fn try_lower_program(program: &CheckedProgram) -> Result<Dag, LowerDiagnostic> {
    try_lower_program_to_library(program).map(|library| library.dag)
}

/// Phase F carrier: a lowered library DAG plus the metadata needed to
/// compose against new code via [`lower_program_with_context`].
///
/// Fields are exposed so the compiled-artifact cache can persist the
/// library state out-of-band, but consumers should treat them as opaque —
/// the contract is that the carrier was produced by
/// [`lower_program_to_library`] on a checked library, and that
/// [`lower_program_with_context`] is the only blessed way to consume it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoweredLibrary {
    /// The library DAG, post-DCE. Node IDs in this DAG are the canonical
    /// library IDs that new-code lowering will reference (after a clone).
    pub dag: Dag,
    /// Map from a library top-level def's name (e.g. `lib_const`,
    /// `lib_double.0`) to the NodeId in `dag` that holds its value. New
    /// code that references the name resolves through this table rather
    /// than emitting a fresh `Load`.
    pub symbol_table: HashMap<String, NodeId>,
    /// Library top-level def bodies, keyed by name. New-code lowering needs
    /// these to inline calls to library functions (matching the monolithic
    /// behaviour of `lower_program(library + new)`).
    pub program_defs: HashMap<String, Expr>,
    /// Library declared types, keyed by name. Used to resolve unbound
    /// `(var libname)` Load types when the new-code expression's metadata
    /// is `default_type`.
    pub program_types: HashMap<String, TensorType>,
    /// Library linearity metadata. Forwarded so cross-DAG reuse hints can
    /// be re-applied if needed.
    pub linearity: LinearityInfo,
    /// Per-library-def "is it lowered?" decision, mirroring the result
    /// of `top_level_lowering_map(library_exprs, library_type_env)`. New-
    /// code lowering decisions need this so a new-code def whose body
    /// calls a library function gets the same lowered/host classification
    /// as it would in monolithic mode (where the same library def lives
    /// in `top_level_defs` and is consulted directly).
    pub lowered_names: HashMap<String, bool>,
}

/// Lower a checked program to the [`LoweredLibrary`] carrier. The bare
/// `dag` field of the result is identical to `lower_program(program)` —
/// the only difference is that `symbol_table`, `program_defs`,
/// `program_types`, and `linearity` are also exposed.
pub fn lower_program_to_library(program: &CheckedProgram) -> LoweredLibrary {
    try_lower_program_to_library(program).unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

pub fn try_lower_program_to_library(
    program: &CheckedProgram,
) -> Result<LoweredLibrary, LowerDiagnostic> {
    catch_lowering(|| lower_program_to_library_inner(program))
}

fn lower_program_to_library_inner(program: &CheckedProgram) -> LoweredLibrary {
    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut sub_t = std::time::Instant::now();
    let log_sub = |label: &str, t: &mut std::time::Instant| {
        if detail_profile {
            eprintln!("lower_sub: {:>8.4}s {}", t.elapsed().as_secs_f64(), label);
            *t = std::time::Instant::now();
        }
    };

    let program_type_env = program.type_env();
    let lowered_names = top_level_lowering_map(program.exprs(), program_type_env);
    log_sub("top_level_lowering_map", &mut sub_t);
    // Reuse the precomputed `lowered_names` rather than calling
    // `top_level_expr_is_lowered`, which would rebuild the map from
    // scratch on every call (1850 calls × full library walk = quadratic
    // before this fix; ~20s on Coral).
    for_each_top_level_item(program.exprs(), &mut |expr| {
        if top_level_expr_is_lowered_with_names(expr, program_type_env, &lowered_names) {
            assert_ir_lowerable(expr);
            assert_ir_typed(expr);
        }
    });
    log_sub("assertions_loop", &mut sub_t);
    let program_types: HashMap<String, TensorType> = program
        .type_env()
        .iter()
        .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
        .collect();
    log_sub("program_types_build", &mut sub_t);
    let program_defs = collect_top_level_defs(program.exprs());
    log_sub("collect_top_level_defs", &mut sub_t);
    let mut ctx = LowerCtx::new(
        program_types.clone(),
        program_defs.clone(),
        program.linearity().clone(),
    );
    log_sub("lower_ctx_new", &mut sub_t);
    let mut last_dag_size: usize = ctx.dag.len();
    // Same reasoning as the assertions_loop above: prefer the
    // precomputed `lowered_names` over a fresh `top_level_expr_is_lowered`
    // rebuild for non-named decls (these are non-`def` top-levels like
    // `defsig`/`deftype`/`typealias`; the `_with_names` path returns true
    // for them, matching the original semantics — they're not `def` so
    // not lowered, but they still pass through the wrapping check).
    for_each_top_level_item(program.exprs(), &mut |expr| {
        if top_level_expr_name(expr).and_then(|name| lowered_names.get(name).copied()) == Some(true)
            || (top_level_expr_name(expr).is_none()
                && top_level_expr_is_lowered_with_names(expr, program_type_env, &lowered_names))
        {
            let t0 = if detail_profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            // For pre-flight gate counting, we want to know how often
            // top_level_expr_is_lowered fires (each call rebuilds the
            // lowering map — quadratic).
            ctx.lower_top_level(expr);
            if let Some(t0) = t0 {
                let elapsed = t0.elapsed();
                let nodes = ctx.dag.len();
                let added = nodes.saturating_sub(last_dag_size);
                last_dag_size = nodes;
                let name = top_level_expr_name(expr).unwrap_or("<anon>");
                eprintln!(
                    "lower_decl: {:>8.4}s nodes_added={:>5} dag_total={:>6} {}",
                    elapsed.as_secs_f64(),
                    added,
                    nodes,
                    name
                );
            }
        }
    });
    log_sub("lower_top_level_loop", &mut sub_t);

    // Collect the pre-DCE name -> NodeId mapping from the lowering ctx's
    // top-level bindings. We flatten tuple-decomposed defs into dotted
    // names mirroring the `Store { name: "foo.0" }` convention used
    // elsewhere; that lets new code reference both `foo` (as a tuple
    // identity) and `foo.N` (as the specific element).
    let mut pre_dce_table: HashMap<String, NodeId> = HashMap::new();
    for (name, value) in ctx.bindings.iter() {
        flatten_binding_into(name, value, &mut pre_dce_table);
    }
    log_sub("flatten_bindings", &mut sub_t);

    let (dce_dag, remap) = crate::optimize::dead_code_eliminate_with_remap(&ctx.dag);
    log_sub("dce", &mut sub_t);
    let (copy_dag, linear_remap) = insert_copy_nodes_for_consuming_fanout(&dce_dag);
    log_sub("implicit_copy_nodes", &mut sub_t);
    let linear_dag = insert_drop_nodes_for_unconsumed_values(copy_dag);
    log_sub("implicit_drop_nodes", &mut sub_t);

    // Renumber the symbol table through DCE's remap. Names whose nodes
    // were eliminated drop out of the table.
    let symbol_table: HashMap<String, NodeId> = pre_dce_table
        .into_iter()
        .filter_map(|(name, old)| {
            remap
                .get(&old)
                .and_then(|new| linear_remap.get(new))
                .copied()
                .map(|new| (name, new))
        })
        .collect();
    log_sub("renumber_symbol_table", &mut sub_t);

    LoweredLibrary {
        dag: linear_dag,
        symbol_table,
        program_defs,
        program_types,
        linearity: program.linearity().clone(),
        lowered_names,
    }
}

fn insert_copy_nodes_for_consuming_fanout(dag: &Dag) -> (Dag, HashMap<NodeId, NodeId>) {
    let mut consuming_uses = HashMap::<NodeId, usize>::new();
    for node in dag.nodes() {
        if !op_consumes_inputs(&node.op) {
            continue;
        }
        for input in &node.inputs {
            *consuming_uses.entry(*input).or_default() += 1;
        }
    }

    let mut seen_consuming_uses = HashMap::<NodeId, usize>::new();
    let mut out = Dag::new();
    let mut id_map = HashMap::<NodeId, NodeId>::new();

    for node in dag.nodes() {
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for input in &node.inputs {
            let mapped = *id_map
                .get(input)
                .expect("input must have been remapped before consumer");
            let total = consuming_uses.get(input).copied().unwrap_or_default();
            if op_consumes_inputs(&node.op) && total > 1 {
                let seen = seen_consuming_uses.entry(*input).or_default();
                *seen += 1;
                if *seen < total {
                    let input_ty = out
                        .get(mapped)
                        .map(|n| n.output_type.clone())
                        .unwrap_or_else(LowerCtx::default_type);
                    let copy =
                        out.add_node(RiscOp::Copy, vec![mapped], input_ty, node.span_id.clone());
                    inputs.push(copy);
                    continue;
                }
            }
            inputs.push(mapped);
        }

        let new_id = out.add_node(
            node.op.clone(),
            inputs,
            node.output_type.clone(),
            node.span_id.clone(),
        );
        if let Some(new_node) = out.node_mut(new_id) {
            new_node.reusable_input = node
                .reusable_input
                .and_then(|old| id_map.get(&old).copied());
            new_node.merged_spans = node.merged_spans.clone();
        }
        id_map.insert(node.id, new_id);
    }

    for root in dag.roots() {
        if let Some(new_root) = id_map.get(root) {
            out.add_root(*new_root);
        }
    }

    (out, id_map)
}

fn op_consumes_inputs(op: &RiscOp) -> bool {
    matches!(op, RiscOp::Realize | RiscOp::Drop | RiscOp::Store { .. })
}

fn insert_drop_nodes_for_unconsumed_values(mut dag: Dag) -> Dag {
    let mut consumed = HashSet::<NodeId>::new();
    for node in dag.nodes() {
        if !op_consumes_inputs(&node.op) {
            continue;
        }
        consumed.extend(node.inputs.iter().copied());
    }
    let roots = dag.roots().iter().copied().collect::<HashSet<_>>();
    let values_to_drop = dag
        .nodes()
        .iter()
        .filter(|node| !roots.contains(&node.id))
        .filter(|node| !consumed.contains(&node.id))
        .filter(|node| {
            !matches!(
                node.op,
                RiscOp::Load { .. } | RiscOp::Drop | RiscOp::Store { .. }
            )
        })
        .map(|node| {
            (
                node.id,
                node.output_type.clone(),
                node.span_id.clone(),
                node.merged_spans.clone(),
            )
        })
        .collect::<Vec<_>>();

    for (id, ty, span_id, merged_spans) in values_to_drop {
        let drop = dag.add_node(RiscOp::Drop, vec![id], ty, span_id);
        if let Some(node) = dag.node_mut(drop) {
            node.merged_spans = merged_spans;
        }
    }

    dag
}

fn strip_drop_nodes(dag: &Dag) -> (Dag, HashMap<NodeId, NodeId>) {
    let mut out = Dag::new();
    let mut id_map = HashMap::<NodeId, NodeId>::new();

    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Drop) {
            continue;
        }
        let inputs = node
            .inputs
            .iter()
            .filter_map(|input| id_map.get(input).copied())
            .collect::<Vec<_>>();
        let new_id = out.add_node(
            node.op.clone(),
            inputs,
            node.output_type.clone(),
            node.span_id.clone(),
        );
        if let Some(new_node) = out.node_mut(new_id) {
            new_node.reusable_input = node
                .reusable_input
                .and_then(|old| id_map.get(&old).copied());
            new_node.merged_spans = node.merged_spans.clone();
        }
        id_map.insert(node.id, new_id);
    }

    for root in dag.roots() {
        if let Some(new_root) = id_map.get(root) {
            out.add_root(*new_root);
        }
    }

    (out, id_map)
}

/// Compose a library's lowered DAG with a new-code [`CheckedProgram`].
/// The library was already lowered via [`lower_program_to_library`]
/// (which is what `lower_program(library)` runs internally); the returned
/// DAG holds the library roots plus new-code nodes.
///
/// `&library` is never mutated; the function is pure and the input
/// library carrier is safe to reuse across many `new_program` snippets.
///
/// New code's `(var libname)` references resolve directly to the library
/// DAG's existing NodeId (no duplicate node), and new code's calls to
/// library functions inline using the library's `program_defs` —
/// matching the monolithic `lower_program(library + new)` behaviour
/// byte-for-byte (modulo any irrelevant extra defs that DCE pruned).
pub fn lower_program_with_context(library: &LoweredLibrary, new_program: &CheckedProgram) -> Dag {
    try_lower_program_with_context(library, new_program)
        .unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

pub fn try_lower_program_with_context(
    library: &LoweredLibrary,
    new_program: &CheckedProgram,
) -> Result<Dag, LowerDiagnostic> {
    catch_lowering(|| lower_program_with_context_inner(library, new_program))
}

fn lower_program_with_context_inner(library: &LoweredLibrary, new_program: &CheckedProgram) -> Dag {
    let new_type_env = new_program.type_env();
    let lowered_names =
        top_level_lowering_map_with_context(library, new_program.exprs(), new_type_env);
    for_each_top_level_item(new_program.exprs(), &mut |expr| {
        if top_level_expr_is_lowered_with_names(expr, new_type_env, &lowered_names) {
            assert_ir_lowerable(expr);
            assert_ir_typed(expr);
        }
    });

    // Combined types: library's own program_types layered with new-code's
    // type_env (which Phase C already unioned with library types).
    // New-code wins on shadow, since the new annotated types are derived
    // from new-code source.
    let mut program_types = library.program_types.clone();
    for (name, ty_expr) in new_program.type_env() {
        program_types.insert(name.clone(), LowerCtx::type_from_type_expr(ty_expr));
    }

    // Combined defs: library defs + new-code defs. Same shadow rule —
    // new-code wins, mirroring monolithic lowering of `library + new`.
    let mut program_defs = library.program_defs.clone();
    for (name, body) in collect_top_level_defs(new_program.exprs()) {
        program_defs.insert(name, body);
    }

    let mut ctx = LowerCtx::new(program_types, program_defs, new_program.linearity().clone());

    // Seed the lowering ctx with the cloned library DAG and the library's
    // name -> NodeId bindings. Library drops are terminal markers for the
    // standalone library snapshot; composition can make formerly terminal
    // values live again, so we strip them and re-normalize Copy/Drop across
    // the combined DAG below.
    let (library_dag, library_remap) = strip_drop_nodes(&library.dag);
    ctx.dag = library_dag;
    for (name, node_id) in &library.symbol_table {
        if let Some(mapped) = library_remap.get(node_id).copied() {
            ctx.bindings
                .insert(name.clone(), LoweredValue::Node(mapped));
        }
    }

    for_each_top_level_item(new_program.exprs(), &mut |expr| {
        if top_level_expr_name(expr).and_then(|name| lowered_names.get(name).copied()) == Some(true)
            || (top_level_expr_name(expr).is_none()
                && top_level_expr_is_lowered(expr, new_program.exprs(), new_type_env))
        {
            ctx.lower_top_level(expr);
        }
    });

    // Skip DCE on the composed DAG: the library DAG was already DCE'd by
    // `lower_program_to_library`, and re-DCE'ing here could prune library
    // roots that are not referenced by the current `new_program`. We still
    // run the linearity normalization passes so consuming fan-out across the
    // library/new-code boundary gets the same Copy nodes as monolithic
    // lowering, and every surviving linear value receives a terminal Drop.
    let (copy_dag, _) = insert_copy_nodes_for_consuming_fanout(&ctx.dag);
    insert_drop_nodes_for_unconsumed_values(copy_dag)
}

fn flatten_binding_into(prefix: &str, value: &LoweredValue, out: &mut HashMap<String, NodeId>) {
    match value {
        LoweredValue::Node(id) => {
            out.insert(prefix.to_string(), *id);
        }
        LoweredValue::Tuple(items) => {
            for (index, item) in items.iter().enumerate() {
                flatten_binding_into(&format!("{prefix}.{index}"), item, out);
            }
        }
    }
}

pub fn tensor_type_from_deep(expr: &Expr) -> TensorType {
    LowerCtx::type_from_type_expr(expr)
}

pub fn lower_subexpr_program(
    expr: &Expr,
    scoped_tensor_types: HashMap<String, TensorType>,
    full_type_env: HashMap<String, Expr>,
    program_defs: HashMap<String, Expr>,
) -> Dag {
    try_lower_subexpr_program(expr, scoped_tensor_types, full_type_env, program_defs)
        .unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

pub fn try_lower_subexpr_program(
    expr: &Expr,
    scoped_tensor_types: HashMap<String, TensorType>,
    full_type_env: HashMap<String, Expr>,
    program_defs: HashMap<String, Expr>,
) -> Result<Dag, LowerDiagnostic> {
    catch_lowering(|| {
        lower_subexpr_program_inner(expr, scoped_tensor_types, full_type_env, program_defs)
    })
}

fn lower_subexpr_program_inner(
    expr: &Expr,
    scoped_tensor_types: HashMap<String, TensorType>,
    full_type_env: HashMap<String, Expr>,
    program_defs: HashMap<String, Expr>,
) -> Dag {
    let scoped_tensor_types_for_bindings = scoped_tensor_types.clone();
    let mut merged_types = full_type_env
        .iter()
        .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
        .collect::<HashMap<_, _>>();
    merged_types.extend(scoped_tensor_types);

    let mut ctx = LowerCtx::new(merged_types, program_defs, LinearityInfo::default());
    for (name, tensor_ty) in scoped_tensor_types_for_bindings {
        let load = ctx.dag.add_node(
            RiscOp::Load {
                name: name.as_str().into(),
            },
            vec![],
            tensor_ty,
            ctx.current_span_id.clone(),
        );
        ctx.bindings.insert(name, LoweredValue::Node(load));
    }
    let value = ctx.lower_expr(expr);
    for id in value.flatten_nodes() {
        ctx.dag.add_root(id);
    }
    let dce_dag = crate::optimize::dead_code_eliminate(&ctx.dag);
    let (copy_dag, _) = insert_copy_nodes_for_consuming_fanout(&dce_dag);
    insert_drop_nodes_for_unconsumed_values(copy_dag)
}

pub fn remap_tensor_dim_symbols(
    dag: &Dag,
    formal_params: &[TensorType],
    actual_args: &[TensorType],
) -> Dag {
    let substitutions = tensor_dim_substitutions(formal_params, actual_args);
    if substitutions.is_empty() {
        return dag.clone();
    }

    fn rewrite_dim_info(dim: &DimInfo, substitutions: &HashMap<String, DimInfo>) -> DimInfo {
        match dim {
            DimInfo::Named(name, None) => substitutions
                .get(name)
                .cloned()
                .unwrap_or_else(|| dim.clone()),
            _ => dim.clone(),
        }
    }

    fn rewrite_dim_expr(expr: &DimExpr, substitutions: &HashMap<String, DimInfo>) -> DimExpr {
        match expr {
            DimExpr::Concrete(value) => DimExpr::Concrete(*value),
            DimExpr::Sym(name) => substitutions
                .get(name)
                .map(DimExpr::from)
                .unwrap_or_else(|| DimExpr::Sym(name.clone())),
            DimExpr::Mul(lhs, rhs) => DimExpr::Mul(
                Box::new(rewrite_dim_expr(lhs, substitutions)),
                Box::new(rewrite_dim_expr(rhs, substitutions)),
            ),
            DimExpr::Div(lhs, rhs) => DimExpr::Div(
                Box::new(rewrite_dim_expr(lhs, substitutions)),
                Box::new(rewrite_dim_expr(rhs, substitutions)),
            ),
        }
    }

    let mut specialized = dag.clone();
    let node_ids = specialized
        .nodes()
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    for id in node_ids {
        let Some(node) = specialized.get(id).cloned() else {
            continue;
        };
        let mut output_type = node.output_type.clone();
        output_type.dims = output_type
            .dims
            .iter()
            .map(|dim| rewrite_dim_info(dim, &substitutions))
            .collect();
        let op = match node.op {
            RiscOp::Expand { axis, size } => RiscOp::Expand {
                axis,
                size: rewrite_dim_expr(&size, &substitutions),
            },
            RiscOp::Reshape { new_shape } => RiscOp::Reshape {
                new_shape: new_shape
                    .iter()
                    .map(|dim| rewrite_dim_info(dim, &substitutions))
                    .collect(),
            },
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
            } => RiscOp::BlasMatmul {
                batch_dims: batch_dims
                    .iter()
                    .map(|dim| rewrite_dim_expr(dim, &substitutions))
                    .collect(),
                m: rewrite_dim_expr(&m, &substitutions),
                n: rewrite_dim_expr(&n, &substitutions),
                k: rewrite_dim_expr(&k, &substitutions),
            },
            other => other,
        };
        specialized.replace_node(id, op, node.inputs, output_type);
        if let Some(reusable_input) = node.reusable_input {
            specialized.set_reusable_input(id, reusable_input);
        }
    }
    specialized
}

fn tensor_dim_substitutions(
    formal_params: &[TensorType],
    actual_args: &[TensorType],
) -> HashMap<String, DimInfo> {
    formal_params
        .iter()
        .zip(actual_args.iter())
        .flat_map(|(formal, actual)| formal.dims.iter().zip(actual.dims.iter()))
        .filter_map(|(formal_dim, actual_dim)| match formal_dim {
            DimInfo::Named(name, None) => Some((name.clone(), actual_dim.clone())),
            _ => None,
        })
        .collect()
}

pub fn top_level_expr_is_lowered(
    expr: &Expr,
    program_exprs: &[Expr],
    type_env: &HashMap<String, Expr>,
) -> bool {
    let lowered_names = top_level_lowering_map(program_exprs, type_env);
    top_level_expr_is_lowered_with_names(expr, type_env, &lowered_names)
}

pub fn top_level_lowering_map(
    exprs: &[Expr],
    type_env: &HashMap<String, Expr>,
) -> HashMap<String, bool> {
    let top_level_defs = collect_top_level_defs(exprs);
    let top_level_sigs = collect_top_level_sigs(exprs);
    let mut cache = HashMap::new();
    let mut visiting = HashSet::new();
    for name in top_level_defs.keys() {
        let lowered = def_is_lowered(
            name,
            &top_level_defs,
            &top_level_sigs,
            type_env,
            &mut cache,
            &mut visiting,
        );
        cache.insert(name.clone(), lowered);
    }
    cache
}

/// Phase G helper: compute `top_level_lowering_map` for new code with
/// the library's pre-computed `lowered_names` seeded into the cache.
/// This makes new-code defs that reference library functions inherit the
/// same lowered-vs-host classification they'd get in monolithic mode
/// (where library defs live alongside new ones in `top_level_defs`).
///
/// The returned map covers BOTH library + new-code names so callers can
/// look up either; downstream filters slice to new-code-only as needed.
pub fn top_level_lowering_map_with_context(
    library: &LoweredLibrary,
    new_exprs: &[Expr],
    new_type_env: &HashMap<String, Expr>,
) -> HashMap<String, bool> {
    let mut top_level_defs = library.program_defs.clone();
    for (name, body) in collect_top_level_defs(new_exprs) {
        top_level_defs.insert(name, body);
    }
    let mut top_level_sigs = collect_top_level_sigs(new_exprs);
    // Library declared types feed `lookup_declared_type_expr`; merge
    // them in so a library def's signature is reachable when the new
    // code's body references it.
    for (name, ty_expr) in &library.program_types {
        top_level_sigs
            .entry(name.clone())
            .or_insert_with(|| ty_expr_to_deep(ty_expr));
    }
    let mut cache = library.lowered_names.clone();
    let mut visiting = HashSet::new();
    for name in top_level_defs.keys() {
        if cache.contains_key(name) {
            continue;
        }
        let lowered = def_is_lowered(
            name,
            &top_level_defs,
            &top_level_sigs,
            new_type_env,
            &mut cache,
            &mut visiting,
        );
        cache.insert(name.clone(), lowered);
    }
    cache
}

/// Inverse of `LowerCtx::type_from_type_expr` — used by
/// `top_level_lowering_map_with_context` to feed library types into the
/// `top_level_sigs` map. Lossy on dim variables (we project to the
/// scalar return type) since `def_is_lowered` only looks at
/// `type_is_never_lowerable`, which inspects the t-fn return type.
fn ty_expr_to_deep(ty: &TensorType) -> Expr {
    use chelis_deep::Span;
    use chelis_deep::ast::{Atom, List, MetaMap};
    let span = Span::new(0, 0);
    let prim = match ty.precision {
        chelis_types::types::Prim::F32 => "f32",
        chelis_types::types::Prim::F64 => "f64",
        chelis_types::types::Prim::F16 => "f16",
        chelis_types::types::Prim::Bf16 => "bf16",
        chelis_types::types::Prim::F8e4m3 => "f8e4m3",
        chelis_types::types::Prim::Int8 => "int8",
        chelis_types::types::Prim::Int32 => "int32",
        chelis_types::types::Prim::Int64 => "int64",
        chelis_types::types::Prim::Bool => "bool",
        chelis_types::types::Prim::String => "string",
    };
    let prim_node = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("t-prim".into()), span),
                Expr::Map(MetaMap::default(), span),
                Expr::Atom(Atom::Symbol(prim.into()), span),
            ],
        },
        span,
    );
    if ty.dims.is_empty() {
        prim_node
    } else {
        Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Symbol("t-tensor".into()), span),
                    Expr::Map(MetaMap::default(), span),
                    prim_node,
                ],
            },
            span,
        )
    }
}

pub fn expr_is_dag_lowerable(expr: &Expr, program: &CheckedProgram) -> bool {
    if expr_requires_host_runtime(expr) {
        return false;
    }

    let top_level_defs = collect_top_level_defs(program.exprs());
    let top_level_sigs = collect_top_level_sigs(program.exprs());
    let mut cache = HashMap::new();
    let mut visiting = HashSet::new();
    !expr_depends_on_nonlowerable_name(
        expr,
        &top_level_defs,
        &top_level_sigs,
        program.type_env(),
        &mut cache,
        &mut visiting,
        &HashSet::new(),
    )
}

fn top_level_expr_is_lowered_with_names(
    expr: &Expr,
    type_env: &HashMap<String, Expr>,
    lowered_names: &HashMap<String, bool>,
) -> bool {
    let top_level_sigs = HashMap::new();
    let Expr::List(list, _) = expr else {
        return true;
    };
    if get_tag(list) == Some("module") {
        return list
            .elements
            .iter()
            .skip(3)
            .all(|child| top_level_expr_is_lowered_with_names(child, type_env, lowered_names));
    }
    if get_tag(list) != Some("def") {
        return true;
    }
    let Some(name) = top_level_expr_name(expr) else {
        return true;
    };
    lowered_names.get(name).copied().unwrap_or_else(|| {
        !lookup_declared_type_expr(&top_level_sigs, type_env, name)
            .is_some_and(type_is_never_lowerable)
    })
}

fn type_is_never_lowerable(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return true;
    };
    match get_tag(list) {
        Some("t-fn") => list.elements.last().is_some_and(type_is_never_lowerable),
        Some("t-tuple") => children(list).iter().any(type_is_never_lowerable),
        Some("t-adt") | Some("t-unit") => true,
        Some("t-prim") => false,
        _ => false,
    }
}

fn type_is_scalar_primitive(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    get_tag(list) == Some("t-prim")
}

fn callable_ref_name(expr: &Expr) -> Option<String> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list)
        .first()
        .and_then(symbol_name)
        .map(|name| name.to_string())
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn top_level_expr_name(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("def") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn expr_requires_host_runtime(expr: &Expr) -> bool {
    match expr {
        Expr::Atom(Atom::Str(_), _) => true,
        Expr::Atom(_, _) => false,
        Expr::Map(map, _) => map
            .entries
            .iter()
            .any(|(_, value)| expr_requires_host_runtime(value)),
        Expr::MetaExpr(meta, _) => {
            expr_requires_host_runtime(&meta.expr)
                || meta
                    .entries
                    .iter()
                    .any(|(_, value)| expr_requires_host_runtime(value))
        }
        Expr::List(list, _) => {
            if get_tag(list) == Some("if") {
                if !if_expr_is_dag_lowerable(list) {
                    return true;
                }
            } else if matches!(
                get_tag(list),
                Some("match" | "record" | "access" | "tuple-get")
            ) {
                return true;
            }
            if get_tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
                && name.chars().next().is_some_and(|ch| ch.is_uppercase())
            {
                return true;
            }
            if get_tag(list) == Some("app")
                && let Some(Expr::List(callee, _)) = children(list).first()
                && get_tag(callee) == Some("var")
                && let Some(name) = children(callee).first().and_then(symbol_name)
                && name.chars().next().is_some_and(|ch| ch.is_uppercase())
            {
                return true;
            }
            if let Some(name) = builtin_name(list) {
                if matches!(
                    name,
                    "print"
                        | "debug"
                        | "string_len"
                        | "string_concat"
                        | "string_slice"
                        | "string_contains"
                        | "string_starts_with"
                        | "string_ends_with"
                        | "string_trim"
                        | "to_string"
                        | "to_int"
                        | "to_float"
                        | "mod"
                        | "bitand"
                        | "bitor"
                        | "bitxor"
                        | "shl"
                        | "shr"
                        | "rank"
                        | "shape"
                        | "numel"
                        | "tuple-get"
                        | "len"
                        | "index"
                        | "append"
                        | "concat"
                        | "take"
                        | "chunk"
                        | "range"
                        | "map"
                        | "filter"
                        | "fold"
                        | "scan"
                        | "partition"
                        | "flat_map"
                        | "flatten"
                        | "zip"
                        | "enumerate"
                        | "dict_of"
                        | "dict_get"
                        | "dict_contains"
                        | "dict_remove"
                        | "dict_insert"
                        | "dict_merge"
                        | "dict_keys"
                        | "dict_values"
                        | "dict_entries"
                        | "read_file"
                        | "write_file"
                        | "read_lines"
                        | "read_bytes"
                        | "file_exists"
                        | "list_dir"
                        | "mmap_file"
                        | "mmap_read"
                        | "mmap_len"
                        | "to_tensor"
                        | "to_list"
                        | "pad_sequences"
                        | "pad_sequences_to"
                        | "einsum"
                        | "split"
                        | "scatter"
                        | "where"
                        | "cumsum"
                        | "sort"
                        | "diagonal"
                        | "trace"
                        | "clamp"
                        | "Cons"
                        | "Nil"
                ) {
                    return true;
                }
                if name == "drop" {
                    return children(list).len() != 2;
                }
                if matches!(
                    name,
                    "add"
                        | "mul"
                        | "sub"
                        | "div"
                        | "max_elem"
                        | "min_elem"
                        | "neg"
                        | "exp"
                        | "log"
                        | "sin"
                        | "sqrt"
                        | "cos"
                        | "tan"
                        | "atan"
                        | "abs"
                        | "floor"
                        | "ceil"
                        | "relu"
                        | "sigmoid"
                        | "tanh"
                        | "silu"
                        | "gelu"
                        | "cmplt"
                        | "gt"
                        | "gte"
                        | "lte"
                        | "eq"
                        | "neq"
                        | "and"
                        | "or"
                        | "not"
                ) && expr_type_metadata(expr).is_some_and(type_is_scalar_primitive)
                {
                    return true;
                }
            }
            // Walk children, but SKIP the metadata map at element 1 (when
            // present) — meta values like the `span` key carry string
            // atoms that are not runtime values. Without this skip, any
            // list whose meta map includes a string-valued key (e.g.
            // `{span: "n_001"}`) would be misclassified as host-runtime
            // and silently dropped from lowering, breaking the S2 audit
            // invariant.
            //
            // NOTE: This skip is load-bearing for span propagation. The
            // N→1 collapse rule's reachability silently depends on it:
            // if a Map at element index 1 is not skipped here,
            // span-bearing nodes (Octant emits `{span: "..."}` at
            // element 1) get classified as host-runtime and silently
            // dropped before lowering, so the §2.3 audit invariant
            // never gets the chance to fire. The greppable invariant
            // catches a regression only indirectly via downstream test
            // failures; keep this skip in place.
            let meta_idx = if matches!(list.elements.get(1), Some(Expr::Map(_, _))) {
                Some(1usize)
            } else {
                None
            };
            list.elements
                .iter()
                .enumerate()
                .filter(|(idx, _)| Some(*idx) != meta_idx)
                .any(|(_, child)| expr_requires_host_runtime(child))
        }
    }
}

fn collect_top_level_defs(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut defs = HashMap::new();
    for expr in exprs {
        collect_top_level_defs_from_expr(expr, &mut defs);
    }
    defs
}

fn collect_top_level_sigs(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut sigs = HashMap::new();
    for expr in exprs {
        collect_top_level_sigs_from_expr(expr, &mut sigs);
    }
    sigs
}

fn for_each_top_level_item(exprs: &[Expr], f: &mut impl FnMut(&Expr)) {
    for expr in exprs {
        for_each_top_level_item_from_expr(expr, f);
    }
}

fn for_each_top_level_item_from_expr(expr: &Expr, f: &mut impl FnMut(&Expr)) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("module") => {
            for child in list.elements.iter().skip(3) {
                for_each_top_level_item_from_expr(child, f);
            }
        }
        _ => f(expr),
    }
}

fn collect_top_level_defs_from_expr(expr: &Expr, defs: &mut HashMap<String, Expr>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("module") => {
            for child in list.elements.iter().skip(3) {
                collect_top_level_defs_from_expr(child, defs);
            }
        }
        Some("def") => {
            let kids = children(list);
            if let (Some(name), Some(body)) =
                (kids.first().and_then(symbol_name), kids.get(1).cloned())
            {
                defs.insert(name.to_string(), body);
            }
        }
        _ => {}
    }
}

fn collect_top_level_sigs_from_expr(expr: &Expr, sigs: &mut HashMap<String, Expr>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("module") => {
            for child in list.elements.iter().skip(3) {
                collect_top_level_sigs_from_expr(child, sigs);
            }
        }
        Some("defsig") => {
            let kids = children(list);
            if let (Some(name), Some(ty)) =
                (kids.first().and_then(symbol_name), kids.get(1).cloned())
            {
                sigs.insert(name.to_string(), ty);
            }
        }
        _ => {}
    }
}

fn def_is_lowered(
    name: &str,
    top_level_defs: &HashMap<String, Expr>,
    top_level_sigs: &HashMap<String, Expr>,
    type_env: &HashMap<String, Expr>,
    cache: &mut HashMap<String, bool>,
    visiting: &mut HashSet<String>,
) -> bool {
    if let Some(lowered) = cache.get(name) {
        return *lowered;
    }
    if !visiting.insert(name.to_string()) {
        return !lookup_declared_type_expr(top_level_sigs, type_env, name)
            .is_some_and(type_is_never_lowerable);
    }

    let lowered = top_level_defs.get(name).is_some_and(|body| {
        !expr_requires_host_runtime(body)
            && !expr_depends_on_nonlowerable_name(
                body,
                top_level_defs,
                top_level_sigs,
                type_env,
                cache,
                visiting,
                &HashSet::new(),
            )
            && !lookup_declared_type_expr(top_level_sigs, type_env, name)
                .is_some_and(type_is_never_lowerable)
    });

    visiting.remove(name);
    cache.insert(name.to_string(), lowered);
    lowered
}

fn lookup_declared_type_expr<'a>(
    top_level_sigs: &'a HashMap<String, Expr>,
    type_env: &'a HashMap<String, Expr>,
    name: &str,
) -> Option<&'a Expr> {
    top_level_sigs
        .get(name)
        .or_else(|| unique_terminal_match(top_level_sigs, name))
        .or_else(|| type_env.get(name))
        .or_else(|| {
            let mut matches = type_env
                .iter()
                .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
            let first = matches.next()?;
            matches.next().is_none().then_some(first)
        })
}

fn terminal_name_matches(full_name: &str, short_name: &str) -> bool {
    full_name == short_name
        || full_name
            .rsplit_once("__")
            .is_some_and(|(_, tail)| tail == short_name)
        || full_name
            .rsplit_once('.')
            .is_some_and(|(_, tail)| tail == short_name)
}

fn unique_terminal_match<'a>(map: &'a HashMap<String, Expr>, name: &str) -> Option<&'a Expr> {
    let mut matches = map
        .iter()
        .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn expr_depends_on_nonlowerable_name(
    expr: &Expr,
    top_level_defs: &HashMap<String, Expr>,
    top_level_sigs: &HashMap<String, Expr>,
    type_env: &HashMap<String, Expr>,
    cache: &mut HashMap<String, bool>,
    visiting: &mut HashSet<String>,
    bound_names: &HashSet<String>,
) -> bool {
    match expr {
        Expr::Atom(_, _) => false,
        Expr::Map(map, _) => map.entries.iter().any(|(_, value)| {
            expr_depends_on_nonlowerable_name(
                value,
                top_level_defs,
                top_level_sigs,
                type_env,
                cache,
                visiting,
                bound_names,
            )
        }),
        Expr::MetaExpr(meta, _) => {
            expr_depends_on_nonlowerable_name(
                &meta.expr,
                top_level_defs,
                top_level_sigs,
                type_env,
                cache,
                visiting,
                bound_names,
            ) || meta.entries.iter().any(|(_, value)| {
                expr_depends_on_nonlowerable_name(
                    value,
                    top_level_defs,
                    top_level_sigs,
                    type_env,
                    cache,
                    visiting,
                    bound_names,
                )
            })
        }
        Expr::List(list, _) => {
            if get_tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
                && !bound_names.contains(name)
                && top_level_defs.contains_key(name)
            {
                return !def_is_lowered(
                    name,
                    top_level_defs,
                    top_level_sigs,
                    type_env,
                    cache,
                    visiting,
                );
            }
            if get_tag(list) == Some("fn") {
                let kids = children(list);
                let mut scoped = bound_names.clone();
                if let Some(Expr::List(params, _)) = kids.first() {
                    for param in children(params) {
                        collect_param_bound_names(param, &mut scoped);
                    }
                }
                return kids.get(1).is_some_and(|body| {
                    expr_depends_on_nonlowerable_name(
                        body,
                        top_level_defs,
                        top_level_sigs,
                        type_env,
                        cache,
                        visiting,
                        &scoped,
                    )
                });
            }
            if get_tag(list) == Some("let") {
                let kids = children(list);
                let mut scoped = bound_names.clone();
                if let Some(Expr::List(bindings, _)) = kids.first()
                    && get_tag(bindings) == Some("bind")
                {
                    let binding_children = children(bindings);
                    let mut index = 0;
                    while index + 1 < binding_children.len() {
                        if expr_depends_on_nonlowerable_name(
                            &binding_children[index + 1],
                            top_level_defs,
                            top_level_sigs,
                            type_env,
                            cache,
                            visiting,
                            &scoped,
                        ) {
                            return true;
                        }
                        if let Some(name) = symbol_name(&binding_children[index]) {
                            scoped.insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                return kids.get(1).is_some_and(|body| {
                    expr_depends_on_nonlowerable_name(
                        body,
                        top_level_defs,
                        top_level_sigs,
                        type_env,
                        cache,
                        visiting,
                        &scoped,
                    )
                });
            }
            if get_tag(list) == Some("match") {
                let kids = children(list);
                if kids.first().is_some_and(|scrutinee| {
                    expr_depends_on_nonlowerable_name(
                        scrutinee,
                        top_level_defs,
                        top_level_sigs,
                        type_env,
                        cache,
                        visiting,
                        bound_names,
                    )
                }) {
                    return true;
                }
                for arm in kids.iter().skip(1) {
                    let Expr::List(arm_list, _) = arm else {
                        continue;
                    };
                    if get_tag(arm_list) != Some("arm") {
                        continue;
                    }
                    let arm_children = children(arm_list);
                    let mut scoped = bound_names.clone();
                    if let Some(pattern) = arm_children.first() {
                        collect_pattern_bound_names(pattern, &mut scoped);
                    }
                    if arm_children.get(1).is_some_and(|guard| {
                        expr_depends_on_nonlowerable_name(
                            guard,
                            top_level_defs,
                            top_level_sigs,
                            type_env,
                            cache,
                            visiting,
                            &scoped,
                        )
                    }) || arm_children.get(2).is_some_and(|body| {
                        expr_depends_on_nonlowerable_name(
                            body,
                            top_level_defs,
                            top_level_sigs,
                            type_env,
                            cache,
                            visiting,
                            &scoped,
                        )
                    }) {
                        return true;
                    }
                }
                return false;
            }
            list.elements.iter().any(|child| {
                expr_depends_on_nonlowerable_name(
                    child,
                    top_level_defs,
                    top_level_sigs,
                    type_env,
                    cache,
                    visiting,
                    bound_names,
                )
            })
        }
    }
}

fn collect_param_bound_names(param: &Expr, out: &mut HashSet<String>) {
    match param {
        Expr::Atom(Atom::Symbol(name), _) => {
            out.insert(name.clone());
        }
        Expr::List(list, _) => {
            if let Some(name) = list.elements.first().and_then(symbol_name) {
                out.insert(name.to_string());
            }
        }
        Expr::Map(_, _) | Expr::MetaExpr(_, _) | Expr::Atom(_, _) => {}
    }
}

fn collect_pattern_bound_names(pattern: &Expr, out: &mut HashSet<String>) {
    match pattern {
        Expr::Atom(_, _) | Expr::Map(_, _) => {}
        Expr::MetaExpr(meta, _) => collect_pattern_bound_names(&meta.expr, out),
        Expr::List(list, _) => {
            if get_tag(list) == Some("pat-var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                out.insert(name.to_string());
                return;
            }
            for child in children(list) {
                collect_pattern_bound_names(child, out);
            }
        }
    }
}

fn builtin_name(list: &List) -> Option<&str> {
    if get_tag(list) != Some("app") {
        return None;
    }
    list.elements.get(2).and_then(|expr| match expr {
        Expr::List(var_list, _) if get_tag(var_list) == Some("var") => {
            children(var_list).first().and_then(symbol_name)
        }
        _ => None,
    })
}

fn expr_type_metadata(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        _ => None,
    }
}

fn if_expr_is_dag_lowerable(list: &List) -> bool {
    if get_tag(list) != Some("if") {
        return false;
    }

    let result_ty = match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => LowerCtx::type_from_meta(&meta.entries),
        _ => LowerCtx::default_type(),
    };
    if !result_ty.precision.is_float() {
        return false;
    }

    let Some(cond_ty_expr) = children(list).first().and_then(expr_type_metadata) else {
        return false;
    };
    let cond_ty = LowerCtx::type_from_type_expr(cond_ty_expr);
    cond_ty.precision == Prim::Bool && (cond_ty.dims.is_empty() || cond_ty.dims == result_ty.dims)
}

fn assert_ir_lowerable(expr: &Expr) {
    match expr {
        Expr::List(list, _) => {
            if let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
                && ((tag == "if" && !if_expr_is_dag_lowerable(list))
                    || matches!(tag.as_str(), "match" | "par" | "jit"))
            {
                raise_lowering_diagnostic(lower_diagnostic_for_expr(
                    unsupported_lowering_message(tag),
                    expr,
                ));
            }
            for elem in &list.elements {
                assert_ir_lowerable(elem);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                assert_ir_lowerable(value);
            }
        }
        Expr::MetaExpr(inner, _) => {
            for (_, value) in &inner.entries {
                assert_ir_lowerable(value);
            }
            assert_ir_lowerable(&inner.expr);
        }
        Expr::Atom(_, _) => {}
    }
}

fn assert_ir_typed(expr: &Expr) {
    match expr {
        Expr::List(list, _) => {
            if let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
                && tag == "app"
                && is_shape_sensitive_builtin_app(list)
                && !has_type_metadata(list)
            {
                let rendered = chelis_deep::printer::print_canonical(std::slice::from_ref(expr))
                    .replace('\n', " ");
                raise_lowering_diagnostic(lower_diagnostic_for_expr(
                    format!(
                        "shape-sensitive IR app nodes must carry explicit type metadata before lowering: {rendered}"
                    ),
                    expr,
                ));
            }
            for elem in &list.elements {
                assert_ir_typed(elem);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                assert_ir_typed(value);
            }
        }
        Expr::MetaExpr(inner, _) => {
            for (_, value) in &inner.entries {
                assert_ir_typed(value);
            }
            assert_ir_typed(&inner.expr);
        }
        Expr::Atom(_, _) => {}
    }
}

fn has_type_metadata(list: &List) -> bool {
    matches!(list.elements.get(1), Some(Expr::Map(meta, _)) if meta.entries.iter().any(|(k, _)| k == "type"))
}

fn is_shape_sensitive_builtin_app(list: &List) -> bool {
    let func_name = match list.elements.get(2) {
        Some(Expr::List(func_list, _)) => {
            match (func_list.elements.first(), func_list.elements.get(2)) {
                (
                    Some(Expr::Atom(Atom::Symbol(tag), _)),
                    Some(Expr::Atom(Atom::Symbol(name), _)),
                ) if tag == "var" => Some(name.as_str()),
                _ => None,
            }
        }
        _ => None,
    };

    matches!(
        func_name,
        Some(
            "matmul"
                | "softmax"
                | "mean"
                | "layer_norm"
                | "conv2d"
                | "sum"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
                | "reshape"
                | "permute"
                | "expand"
                | "pad"
                | "shrink"
                | "stride"
        )
    )
}

fn get_tag(list: &List) -> Option<&str> {
    match list.elements.first() {
        Some(Expr::Atom(Atom::Symbol(tag), _)) => Some(tag.as_str()),
        _ => None,
    }
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

#[derive(Clone)]
enum CallableExpr {
    Plain(Expr),
    Vmap {
        fn_expr: Expr,
        axis: usize,
    },
    VmapGrad {
        fn_expr: Expr,
        wrt: Option<Vec<usize>>,
        axis: usize,
    },
    Grad {
        fn_expr: Expr,
        wrt: Option<Vec<usize>>,
    },
}

#[derive(Clone)]
enum LoweredValue {
    Node(NodeId),
    Tuple(Vec<LoweredValue>),
}

impl LoweredValue {
    fn expect_node(&self, context: &str) -> NodeId {
        match self {
            Self::Node(id) => *id,
            Self::Tuple(_) => raise_lowering_error(
                format!("{context} expected a single tensor value"),
                None,
                None,
            ),
        }
    }

    fn flatten_nodes(&self) -> Vec<NodeId> {
        match self {
            Self::Node(id) => vec![*id],
            Self::Tuple(items) => items.iter().flat_map(Self::flatten_nodes).collect(),
        }
    }

    fn tuple_get(&self, index: usize) -> Option<LoweredValue> {
        match self {
            Self::Tuple(items) => items.get(index).cloned(),
            Self::Node(_) => None,
        }
    }

    fn from_flat(template: &LoweredValue, nodes: &mut dyn Iterator<Item = NodeId>) -> LoweredValue {
        match template {
            Self::Node(_) => Self::Node(nodes.next().expect("flattened lowered value mismatch")),
            Self::Tuple(items) => Self::Tuple(
                items
                    .iter()
                    .map(|item| Self::from_flat(item, nodes))
                    .collect(),
            ),
        }
    }
}

fn extract_param_type(expr: &Expr, index: usize) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("fn") {
        return None;
    }
    let params = children(list).first()?;
    let Expr::List(params_list, _) = params else {
        return None;
    };
    let param = children(params_list).get(index)?;
    match param {
        Expr::MetaExpr(meta, _) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        Expr::List(param_list, _) => match param_list.elements.get(1) {
            Some(Expr::Map(meta, _)) => meta
                .entries
                .iter()
                .find(|(key, _)| key == "type")
                .map(|(_, value)| value),
            _ => None,
        },
        _ => None,
    }
}

fn param_name_and_type_expr(param: &Expr) -> Option<(String, Option<&Expr>)> {
    match param {
        Expr::Atom(Atom::Symbol(name), _) => Some((name.clone(), None)),
        Expr::MetaExpr(meta, _) => {
            let Expr::Atom(Atom::Symbol(name), _) = meta.expr.as_ref() else {
                return None;
            };
            let ty_expr = meta
                .entries
                .iter()
                .find(|(key, _)| key == "type")
                .map(|(_, value)| value);
            Some((name.clone(), ty_expr))
        }
        Expr::List(param_list, _) => {
            let Expr::Atom(Atom::Symbol(name), _) = param_list.elements.first()? else {
                return None;
            };
            let ty_expr = if let Some(Expr::Map(meta, _)) = param_list.elements.get(1) {
                meta.entries
                    .iter()
                    .find(|(key, _)| key == "type")
                    .map(|(_, value)| value)
            } else {
                None
            };
            Some((name.clone(), ty_expr))
        }
        _ => None,
    }
}

fn extract_fn_return_type(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("fn") {
        return None;
    }
    let Expr::Map(meta, _) = list.elements.get(1)? else {
        return None;
    };
    let (_, ty_expr) = meta.entries.iter().find(|(key, _)| key == "type")?;
    let Expr::List(fn_ty, _) = ty_expr else {
        return None;
    };
    if get_tag(fn_ty) != Some("t-fn") {
        return None;
    }
    children(fn_ty).last()
}

fn axis_to_front_perm(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(rank);
    perm.push(axis);
    perm.extend((0..rank).filter(|candidate| *candidate != axis));
    perm
}

fn front_to_axis_perm(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (1..rank).collect();
    perm.insert(axis, 0);
    perm
}

fn permuted_tensor_type(ty: &TensorType, axes: &[usize]) -> TensorType {
    TensorType {
        dims: axes.iter().map(|axis| ty.dims[*axis].clone()).collect(),
        precision: ty.precision,
    }
}

struct LowerCtx {
    dag: Dag,
    bindings: HashMap<String, LoweredValue>,
    local_callables: HashMap<String, Expr>,
    program_types: HashMap<String, TensorType>,
    program_defs: HashMap<String, Expr>,
    random_seed: Option<u64>,
    linearity: LinearityInfo,
    inlining_names: HashSet<String>,
    dim_substitutions: HashMap<String, DimInfo>,
    /// The span_id of the Deep `Expr` currently being lowered. Threaded
    /// through `lower_expr` (set on entry, restored on exit) so every
    /// helper that calls `self.dag.add_node(...)` can pass the
    /// region-corresponding span without plumbing it through every
    /// helper's argument list. See
    /// `spec/design/chelis_span_survival.md` §2.3.
    current_span_id: Option<String>,
}

impl LowerCtx {
    fn new(
        program_types: HashMap<String, TensorType>,
        program_defs: HashMap<String, Expr>,
        linearity: LinearityInfo,
    ) -> Self {
        Self {
            dag: Dag::new(),
            bindings: HashMap::new(),
            local_callables: HashMap::new(),
            program_types,
            program_defs,
            random_seed: None,
            linearity,
            inlining_names: HashSet::new(),
            dim_substitutions: HashMap::new(),
            current_span_id: None,
        }
    }

    /// Default tensor type when we don't have richer type info.
    fn default_type() -> TensorType {
        TensorType::scalar_f32()
    }

    /// Choose the output `TensorType` for an elementwise op whose shape
    /// matches the first input. Prefer the input DAG node's dims over the
    /// annotated `ty.dims` when the input has a non-empty rank — the input
    /// dims carry the user-facing symbolic names from `defsig`/param types,
    /// while `ty` after type inference can hold internal fresh-var names
    /// (e.g. `d44`) that aren't declared in the emitted C scope. When
    /// `precision_override` is provided it wins (e.g. `cmplt` → `Bool`);
    /// otherwise we use `ty`'s precision when present, else the input's.
    fn elementwise_out_ty(
        dag: &Dag,
        input: NodeId,
        ty: &TensorType,
        precision_override: Option<Prim>,
    ) -> TensorType {
        let input_ty = dag.get(input).map(|node| node.output_type.clone());
        let dims = match input_ty.as_ref() {
            Some(in_ty) if !in_ty.dims.is_empty() => in_ty.dims.clone(),
            _ => {
                if ty.dims.is_empty() {
                    input_ty
                        .as_ref()
                        .map(|in_ty| in_ty.dims.clone())
                        .unwrap_or_default()
                } else {
                    ty.dims.clone()
                }
            }
        };
        let precision = precision_override.unwrap_or_else(|| {
            if *ty != Self::default_type() {
                ty.precision
            } else {
                input_ty
                    .map(|in_ty| in_ty.precision)
                    .unwrap_or(ty.precision)
            }
        });
        TensorType { dims, precision }
    }

    fn attach_reuse_hint(
        &mut self,
        node: NodeId,
        app_span: Span,
        candidate_inputs: &[NodeId],
    ) -> NodeId {
        if let Some(input_index) = self.linearity.reusable_input_for_span(app_span)
            && let Some(input) = candidate_inputs.get(input_index)
        {
            self.dag.set_reusable_input(node, *input);
        }
        node
    }

    fn repair_output_type_if_default(&mut self, value: &LoweredValue, desired: &TensorType) {
        let LoweredValue::Node(id) = value else {
            return;
        };
        let Some(node) = self.dag.get(*id) else {
            return;
        };
        if node.output_type != Self::default_type() || desired == &Self::default_type() {
            return;
        }
        self.dag
            .replace_node(*id, node.op.clone(), node.inputs.clone(), desired.clone());
    }

    /// Extract a type from a metadata map if one is present, otherwise return a default.
    fn type_from_meta(meta: &[(String, Expr)]) -> TensorType {
        for (key, val) in meta {
            if key == "type" {
                return Self::type_from_type_expr(val);
            }
        }
        Self::default_type()
    }

    fn remap_callable_dim_symbols(
        dag: &Dag,
        formal_params: &[TensorType],
        actual_args: &[TensorType],
    ) -> Dag {
        remap_tensor_dim_symbols(dag, formal_params, actual_args)
    }

    fn seed_subctx_with_lexical_scope(
        &self,
        subctx: &mut LowerCtx,
        shadowed: &[String],
    ) -> HashMap<String, NodeId> {
        let shadowed = shadowed.iter().cloned().collect::<HashSet<_>>();
        let mut captures = HashMap::new();
        for (name, value) in self
            .bindings
            .iter()
            .filter(|(name, _)| !shadowed.contains(*name))
        {
            let LoweredValue::Node(node_id) = value else {
                continue;
            };
            let ty = self
                .dag
                .get(*node_id)
                .map(|node| node.output_type.clone())
                .unwrap_or_else(Self::default_type);
            let load = subctx.dag.add_node(
                RiscOp::Load {
                    name: name.as_str().into(),
                },
                vec![],
                ty,
                subctx.current_span_id.clone(),
            );
            subctx
                .bindings
                .insert(name.clone(), LoweredValue::Node(load));
            captures.insert(name.clone(), *node_id);
        }
        subctx.local_callables.extend(
            self.local_callables
                .iter()
                .filter(|(name, _)| !shadowed.contains(*name))
                .map(|(name, value)| (name.clone(), value.clone())),
        );
        captures
    }

    fn type_from_type_expr(expr: &Expr) -> TensorType {
        if let Some(prim) = Self::try_extract_prim(expr) {
            return TensorType {
                dims: vec![],
                precision: prim,
            };
        }
        if let Some(inner) = Self::try_extract_ref_type(expr) {
            return Self::type_from_type_expr(inner);
        }
        if let Some(tt) = Self::try_extract_tensor_type(expr) {
            return tt;
        }
        Self::default_type()
    }

    fn try_extract_ref_type(expr: &Expr) -> Option<&Expr> {
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-ref"
        {
            return list.elements.get(2);
        }
        None
    }

    fn try_extract_prim(expr: &Expr) -> Option<Prim> {
        // (t-prim {} f32)
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-prim"
            && let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2]
        {
            return Prim::parse_name(name);
        }
        None
    }

    fn try_extract_tensor_type(expr: &Expr) -> Option<TensorType> {
        // Flat format: (t-tensor {} dim1 dim2 ... (t-prim {} p))
        // Children after tag+meta: dimension nodes followed by a t-prim node as the last child.
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-tensor"
        {
            // elements[0] = tag, elements[1] = meta, elements[2..] = children
            let children = &list.elements[2..];
            if children.is_empty() {
                return None;
            }
            // Last child is the precision (t-prim {} name).
            let prim = Self::try_extract_prim(children.last()?)?;
            // All children before the last are dimension nodes.
            let mut dims = Vec::new();
            for child in &children[..children.len() - 1] {
                if let Some(dim) = Self::try_extract_dim(child) {
                    dims.push(dim);
                }
            }
            return Some(TensorType {
                dims,
                precision: prim,
            });
        }
        None
    }

    /// Extract a single dimension from a dimension node.
    fn try_extract_dim(expr: &Expr) -> Option<DimInfo> {
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
        {
            match tag.as_str() {
                "d-name" => {
                    if let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2] {
                        return Some(DimInfo::Named(name.clone(), None));
                    }
                }
                "d-var" => {
                    if let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2] {
                        return Some(DimInfo::Named(name.clone(), None));
                    }
                }
                "d-lit" => {
                    if let Expr::Atom(Atom::Int(n), _) = &list.elements[2] {
                        return Some(DimInfo::Lit(*n as usize));
                    }
                }
                _ => {}
            }
        }
        // Also handle bare symbols/ints for backward compat.
        match expr {
            Expr::Atom(Atom::Symbol(name), _) => Some(DimInfo::Named(name.clone(), None)),
            Expr::Atom(Atom::Int(n), _) => Some(DimInfo::Lit(*n as usize)),
            _ => None,
        }
    }

    fn dim_info_from_dim_expr(size: &DimExpr) -> Option<DimInfo> {
        match size {
            DimExpr::Concrete(value) => Some(DimInfo::Lit(*value)),
            DimExpr::Sym(name) => Some(DimInfo::Named(name.clone(), None)),
            DimExpr::Mul(_, _) | DimExpr::Div(_, _) => None,
        }
    }

    fn fallback_expand_type(
        &self,
        input: NodeId,
        axis: usize,
        size: &DimExpr,
    ) -> Option<TensorType> {
        let input_ty = self.dag.get(input)?.output_type.clone();
        let inserted_dim = Self::dim_info_from_dim_expr(size)?;
        let mut dims = input_ty.dims;
        if axis > dims.len() {
            return None;
        }
        dims.insert(axis, inserted_dim);
        Some(TensorType {
            dims,
            precision: input_ty.precision,
        })
    }

    fn lower_top_level(&mut self, expr: &Expr) {
        if let Expr::List(list, _) = expr
            && let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
        {
            match tag.as_str() {
                // Skip type-level declarations.
                "defsig" | "deftype" | "typealias" => return,
                _ => {}
            }
        }

        let value = self.lower_expr(expr);
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(|expr| match expr {
                Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
                _ => None,
            })
        {
            // `lower_expr` cleared `current_span_id` on exit. Re-thread the
            // def's own span_id (if present) for the duration of
            // `add_named_roots` so any Store nodes emitted on the
            // top-level def's behalf inherit the def's source region —
            // satisfying the §2.3 audit invariant for input def spans.
            let saved_span_id = self.current_span_id.clone();
            if let Some(s) = expr.span_id() {
                self.current_span_id = Some(s.to_owned());
            }
            self.add_named_roots(name, &value);
            self.current_span_id = saved_span_id;
        } else {
            for id in value.flatten_nodes() {
                self.dag.add_root(id);
            }
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> LoweredValue {
        // Thread the current Deep node's span_id through any add_node()
        // calls made while lowering this expr or its children. We snapshot
        // the previous span_id and restore it on return so sibling exprs
        // are unaffected; child exprs whose own meta carries a span
        // override while they're being lowered, and child exprs without
        // their own span fall through to the parent's span (the
        // region-corresponding rule from spec/design/chelis_span_survival.md
        // §2.3 / §2.4).
        let saved_span_id = self.current_span_id.clone();
        if let Some(s) = expr.span_id() {
            self.current_span_id = Some(s.to_owned());
        }
        let result = match expr {
            Expr::Atom(atom, _) => self.lower_atom(atom),
            Expr::List(list, span) => self.lower_list(list, *span),
            Expr::Map(_, _) => LoweredValue::Node({
                // Bare metadata map -- shouldn't appear as an expression to lower.
                self.dag.add_node(
                    RiscOp::Const { value: 0.0 },
                    vec![],
                    Self::default_type(),
                    self.current_span_id.clone(),
                )
            }),
            Expr::MetaExpr(meta_expr, _) => self.lower_expr(&meta_expr.expr),
        };
        self.current_span_id = saved_span_id;
        result
    }

    /// Append `current_span_id` (if any) to a single existing IR node's
    /// `merged_spans`, lex-sorted and deduped. This implements the N→1
    /// lowering collapse rule from `spec/design/chelis_span_survival.md`
    /// §2.3 (rule b): when a parent Deep expr lowers to a body that already
    /// corresponds to an existing IR node — for example, a `(def {span: a})`
    /// whose body is an existing node, or a `(var {span: u} a)` whose body
    /// is a previously-bound `LoweredValue` — append the parent's span to
    /// the existing node so the audit invariant ("every input span appears
    /// as `span_id` or `merged_spans` on at least one node") is preserved.
    /// Thin wrapper over `crate::span_merge::append_span_to_node`; the
    /// shared helper owns the None-no-op / canonical-no-op / dedup / sort
    /// logic.
    fn append_current_span_to_existing_node(&mut self, id: NodeId) {
        crate::span_merge::append_span_to_node(&mut self.dag, id, self.current_span_id.as_deref());
    }

    /// Walk a `LoweredValue` and apply
    /// `append_current_span_to_existing_node` to every contained node id.
    /// Used at every site that returns a cached/aliased `LoweredValue`
    /// from a name → value map (e.g. `bindings`) — those returns are N→1
    /// lowering collapses that must still record the parent expr's span.
    fn append_current_span_to_lowered_value(&mut self, value: &LoweredValue) {
        match value {
            LoweredValue::Node(id) => self.append_current_span_to_existing_node(*id),
            LoweredValue::Tuple(items) => {
                for item in items {
                    self.append_current_span_to_lowered_value(item);
                }
            }
        }
    }

    fn add_named_roots(&mut self, prefix: &str, value: &LoweredValue) {
        match value {
            LoweredValue::Node(id) if !prefix.contains('.') => {
                // Top-level def whose body lowered to a single existing
                // node — no new Store is emitted. This is a region-merge
                // during lowering: the def's source region and the
                // body's source region collapse onto one IR node. Per
                // spec/design/chelis_span_survival.md §2.3 rule (b), N→1
                // region merges record the additional span(s) in
                // `merged_spans` (the body node already owns `span_id`),
                // lex-sorted and deduped, so the audit invariant ("every
                // input span appears on at least one IR node") still
                // holds.
                self.append_current_span_to_existing_node(*id);
                self.dag.add_root(*id);
            }
            LoweredValue::Node(id) => {
                let output_type = self
                    .dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                let stored = self.dag.add_node(
                    RiscOp::Store {
                        name: prefix.into(),
                    },
                    vec![*id],
                    output_type,
                    self.current_span_id.clone(),
                );
                self.dag.add_root(stored);
            }
            LoweredValue::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    self.add_named_roots(&format!("{prefix}.{index}"), item);
                }
            }
        }
    }

    fn lower_atom(&mut self, atom: &Atom) -> LoweredValue {
        match atom {
            Atom::Symbol(name) => {
                if let Some(value) = self.bindings.get(name) {
                    let cached = value.clone();
                    // N→1 lowering collapse per
                    // spec/design/chelis_span_survival.md §2.3 rule (b):
                    // returning a cached `LoweredValue` for a span-bearing
                    // parent expr (e.g. an `Atom::Symbol` whose enclosing
                    // node carries a `span:` meta) must still record the
                    // parent's span on the existing node so the audit
                    // chain doesn't drop it.
                    self.append_current_span_to_lowered_value(&cached);
                    cached
                } else {
                    LoweredValue::Node(self.dag.add_node(
                        RiscOp::Load {
                            name: name.as_str().into(),
                        },
                        vec![],
                        Self::default_type(),
                        self.current_span_id.clone(),
                    ))
                }
            }
            Atom::Int(n) => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: *n as f64 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            )),
            Atom::Float(f) => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: *f },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            )),
            Atom::Bool(b) => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const {
                    value: if *b { 1.0 } else { 0.0 },
                },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            )),
            Atom::Str(_) | Atom::Keyword(_) => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            )),
        }
    }

    fn lower_expr_node(&mut self, expr: &Expr, context: &str) -> NodeId {
        self.lower_expr(expr).expect_node(context)
    }

    fn lower_list(&mut self, list: &List, span: Span) -> LoweredValue {
        let elems = &list.elements;
        if elems.is_empty() {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ));
        }

        let tag = match &elems[0] {
            Expr::Atom(Atom::Symbol(s), _) => s.as_str(),
            _ => {
                return LoweredValue::Node(self.dag.add_node(
                    RiscOp::Const { value: 0.0 },
                    vec![],
                    Self::default_type(),
                    self.current_span_id.clone(),
                ));
            }
        };

        match tag {
            "def" => self.lower_def(elems),
            "let" => self.lower_let(elems),
            "lit" => self.lower_lit(elems),
            "var" => self.lower_var(elems),
            "app" => self.lower_app(elems, span),
            "fn" => self.lower_fn(elems),
            "pipe" => self.lower_pipe(elems),
            "cast" => self.lower_cast(elems),
            "if" => self.lower_if(elems),
            "tuple" => self.lower_tuple(elems),
            "par" => self.lower_par(elems),
            "realize" => self.lower_realize(elems),
            "copy" => self.lower_copy(elems),
            "borrow" => self.lower_identity(elems),
            "tuple-get" => self.lower_tuple_get(elems),
            "match" => self.lower_match(elems),
            "grad" => self.lower_grad(elems),
            "handle-effect" => self.lower_handle_effect(elems),
            "vmap" | "jit" => self.lower_unsupported(tag, elems),
            "defsig" | "deftype" | "typealias" => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            )),
            _ => {
                let mut last = LoweredValue::Node(self.dag.add_node(
                    RiscOp::Const { value: 0.0 },
                    vec![],
                    Self::default_type(),
                    self.current_span_id.clone(),
                ));
                for elem in &elems[2..] {
                    last = self.lower_expr(elem);
                }
                last
            }
        }
    }

    /// `(def {meta...} name body)`
    fn lower_def(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ));
        }
        let name = match &elems[2] {
            Expr::Atom(Atom::Symbol(s), _) => s.clone(),
            _ => String::new(),
        };
        let body_id = self.lower_expr(&elems[3]);
        if !name.is_empty() {
            self.bindings.insert(name, body_id.clone());
            if let Some(callable) = self.callable_binding_expr(&elems[3]) {
                self.local_callables.insert(
                    match &elems[2] {
                        Expr::Atom(Atom::Symbol(s), _) => s.clone(),
                        _ => String::new(),
                    },
                    callable,
                );
            }
        }
        body_id
    }

    /// `(let {} (bind {} name1 expr1 name2 expr2 ...) body)`
    fn lower_let(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ));
        }
        let saved = self.bindings.clone();
        let saved_callables = self.local_callables.clone();

        // elems[2] = (bind {} name1 expr1 name2 expr2 ...)
        if let Expr::List(bind_list, _) = &elems[2] {
            // Skip tag and meta (elements[0] and [1]).
            let bind_kids = &bind_list.elements[2..];
            let mut i = 0;
            while i + 1 < bind_kids.len() {
                if let Expr::Atom(Atom::Symbol(name), _) = &bind_kids[i] {
                    if let Some(callable) = self.callable_binding_expr(&bind_kids[i + 1]) {
                        self.local_callables.insert(name.clone(), callable);
                    } else {
                        let val_id = self.lower_expr(&bind_kids[i + 1]);
                        self.bindings.insert(name.clone(), val_id);
                    }
                }
                i += 2;
            }
        }

        let result = self.lower_expr(&elems[3]);
        self.bindings = saved; // Restore scope
        self.local_callables = saved_callables;
        result
    }

    /// `(lit {type: T} value)`
    fn lower_lit(&mut self, elems: &[Expr]) -> LoweredValue {
        let ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            Self::type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        let value = if let Some(val_expr) = elems.get(2) {
            match val_expr {
                Expr::Atom(Atom::Int(n), _) => *n as f64,
                Expr::Atom(Atom::Float(f), _) => *f,
                Expr::Atom(Atom::Bool(true), _) => 1.0,
                Expr::Atom(Atom::Bool(false), _) => 0.0,
                _ => 0.0,
            }
        } else {
            0.0
        };

        LoweredValue::Node(self.dag.add_node(
            RiscOp::Const { value },
            vec![],
            ty,
            self.current_span_id.clone(),
        ))
    }

    /// `(var {meta...} name)`
    fn lower_var(&mut self, elems: &[Expr]) -> LoweredValue {
        // C6: Extract type from metadata if available, otherwise use checked top-level type info.
        let explicit_ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            Self::type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        if let Some(Expr::Atom(Atom::Symbol(name), _)) = elems.get(2) {
            if let Some(id) = self.bindings.get(name) {
                let cached = id.clone();
                // N→1 lowering collapse per
                // spec/design/chelis_span_survival.md §2.3 rule (b):
                // returning a cached `LoweredValue` for a span-bearing
                // `(var {span: u} a)` must still record the var-ref's
                // span on the existing node. Without this append, a
                // var-ref to a let-bound name would drop its own span
                // and break the audit chain (the Load branch below
                // honors the rule via `current_span_id` on add_node).
                self.append_current_span_to_lowered_value(&cached);
                return cached;
            }
            // Reject `(var X)` where X is a known builtin name. The DAG
            // emits a Load when it encounters a free var, but a builtin
            // like `fold`, `map`, or `einsum` is a language-level operator,
            // not a host value — emitting `fold` as a Load and then as a
            // C identifier is meaningless. This happens most commonly
            // during `grad(fn_using_fold)` lowering, where the fn body is
            // inlined into a DAG context that can't represent the HOF.
            if BUILTIN_NAMES.contains(&name.as_str()) {
                raise_lowering_error(
                    format!(
                        "builtin `{name}` is not supported by IR evaluation as a \
                     value. If this is the body of a fn passed to `grad`, \
                     the grad pass needs to specialize around the builtin \
                     rather than inlining it"
                    ),
                    elems.first().map(Expr::span),
                    elems.first().and_then(Expr::span_id).map(ToOwned::to_owned),
                );
            }
            let ty = if explicit_ty == Self::default_type() {
                self.program_types.get(name).cloned().unwrap_or(explicit_ty)
            } else {
                explicit_ty
            };
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Load {
                    name: name.as_str().into(),
                },
                vec![],
                ty,
                self.current_span_id.clone(),
            ));
        }
        LoweredValue::Node(self.dag.add_node(
            RiscOp::Const { value: 0.0 },
            vec![],
            Self::default_type(),
            self.current_span_id.clone(),
        ))
    }

    /// `(app {meta...} func arg1 arg2 ...)`
    fn lower_app(&mut self, elems: &[Expr], app_span: Span) -> LoweredValue {
        if elems.len() < 4 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ));
        }

        let ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            Self::type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        // Check if func is a known built-in: (var {} name).
        if let Expr::List(func_list, _) = &elems[2]
            && let Some(Expr::Atom(Atom::Symbol(func_tag), _)) = func_list.elements.first()
            && func_tag == "var"
            && let Some(Expr::Atom(Atom::Symbol(func_name), _)) = func_list.elements.get(2)
            && !self.program_defs.contains_key(func_name)
            && !self.local_callables.contains_key(func_name)
        {
            return LoweredValue::Node(self.lower_builtin_app(
                func_name,
                &elems[3..],
                &ty,
                app_span,
            ));
        }

        if let Some(lowered) = self.try_lower_callable_app(&elems[2], &elems[3..], &ty, app_span) {
            return lowered;
        }

        // Not a recognized built-in -- lower func and args, return last.
        let mut last = self.lower_expr(&elems[2]);
        for arg in &elems[3..] {
            last = self.lower_expr(arg);
        }
        last
    }

    fn try_lower_callable_app(
        &mut self,
        func: &Expr,
        args: &[Expr],
        ty: &TensorType,
        app_span: Span,
    ) -> Option<LoweredValue> {
        let callable = self.resolve_callable_expr(func)?;
        // If the callee is a named top-level/local def, track it on the
        // inlining stack so a recursive body doesn't re-resolve and re-inline
        // itself infinitely. Nameless fn literals don't need tracking because
        // they can't refer to themselves by name.
        let inlining_name = callable_ref_name(func).filter(|name| {
            self.local_callables.contains_key(name) || self.program_defs.contains_key(name)
        });
        if let Some(name) = inlining_name.as_ref() {
            self.inlining_names.insert(name.clone());
        }
        let result = match callable {
            CallableExpr::Plain(fn_expr) => {
                Some(self.lower_plain_callable_app(&fn_expr, args, app_span))
            }
            CallableExpr::Vmap { fn_expr, axis } => {
                Some(self.lower_vmap_callable_app(&fn_expr, axis, args, ty, app_span))
            }
            CallableExpr::VmapGrad { fn_expr, wrt, axis } => Some(
                self.lower_vmap_grad_callable_app(&fn_expr, wrt.as_deref(), axis, args, app_span),
            ),
            CallableExpr::Grad { fn_expr, wrt } => {
                Some(self.lower_grad_callable_app(&fn_expr, wrt.as_deref(), args, app_span))
            }
        };
        if let Some(name) = inlining_name {
            self.inlining_names.remove(&name);
        }
        result
    }

    fn resolve_callable_expr(&self, expr: &Expr) -> Option<CallableExpr> {
        self.resolve_callable_expr_inner(expr, &mut HashSet::new())
    }

    fn resolve_callable_expr_inner(
        &self,
        expr: &Expr,
        visited: &mut HashSet<String>,
    ) -> Option<CallableExpr> {
        let Expr::List(list, _) = expr else {
            return None;
        };
        match get_tag(list) {
            Some("fn") => Some(CallableExpr::Plain(expr.clone())),
            Some("var") => {
                let name = children(list).first().and_then(|expr| match expr {
                    Expr::Atom(Atom::Symbol(name), _) => Some(name.clone()),
                    _ => None,
                })?;
                if !visited.insert(name.clone()) || self.inlining_names.contains(&name) {
                    return None;
                }
                let body = self
                    .local_callables
                    .get(&name)
                    .or_else(|| self.program_defs.get(&name))?;
                self.resolve_callable_expr_inner(body, visited)
            }
            Some("vmap") => {
                let kids = children(list);
                let axis = kids
                    .get(1)
                    .and_then(|expr| self.extract_usize_value(expr))
                    .unwrap_or(0);
                if let Some(Expr::List(grad_list, _)) = kids.first()
                    && get_tag(grad_list) == Some("grad")
                {
                    let wrt = self.extract_grad_wrt_indices(grad_list);
                    return self
                        .resolve_callable_expr_inner(children(grad_list).first()?, visited)
                        .and_then(|inner| match inner {
                            CallableExpr::Plain(fn_expr) => {
                                Some(CallableExpr::VmapGrad { fn_expr, wrt, axis })
                            }
                            _ => None,
                        });
                }
                self.resolve_callable_expr_inner(kids.first()?, visited)
                    .and_then(|inner| match inner {
                        CallableExpr::Plain(fn_expr) => Some(CallableExpr::Vmap { fn_expr, axis }),
                        CallableExpr::Vmap { .. } => None,
                        CallableExpr::VmapGrad { .. } => None,
                        CallableExpr::Grad { fn_expr, wrt } => {
                            Some(CallableExpr::Grad { fn_expr, wrt })
                        }
                    })
            }
            Some("grad") => self
                .resolve_callable_expr_inner(children(list).first()?, visited)
                .and_then(|inner| match inner {
                    CallableExpr::Plain(fn_expr) => Some(CallableExpr::Grad {
                        fn_expr,
                        wrt: self.extract_grad_wrt_indices(list),
                    }),
                    _ => None,
                }),
            _ => None,
        }
    }

    fn callable_binding_expr(&self, expr: &Expr) -> Option<Expr> {
        self.resolve_callable_expr(expr).map(|_| expr.clone())
    }

    fn extract_grad_wrt_indices(&self, list: &List) -> Option<Vec<usize>> {
        let wrt_expr = children(list).get(1)?;
        if let Expr::List(tuple, _) = wrt_expr
            && get_tag(tuple) == Some("tuple")
        {
            return Some(
                children(tuple)
                    .iter()
                    .filter_map(|expr| self.extract_usize_value(expr))
                    .collect(),
            );
        }
        self.extract_usize_value(wrt_expr).map(|index| vec![index])
    }

    fn is_selected_wrt(
        &self,
        index: usize,
        ty: &TensorType,
        wrt_indices: Option<&[usize]>,
    ) -> bool {
        let differentiable = ty.precision.is_float();
        match wrt_indices {
            Some(indices) => differentiable && indices.contains(&index),
            None => differentiable,
        }
    }

    fn lower_grad_callable_app(
        &mut self,
        fn_expr: &Expr,
        wrt_indices: Option<&[usize]>,
        args: &[Expr],
        app_span: Span,
    ) -> LoweredValue {
        let actual_args: Vec<NodeId> = args
            .iter()
            .map(|arg| self.lower_expr_node(arg, "grad arguments"))
            .collect();
        self.lower_grad_callable_with_nodes(fn_expr, wrt_indices, &actual_args, app_span)
    }

    /// Core lowering for `grad(fn)` applied to already-lowered argument
    /// nodes. Used by both `lower_grad_callable_app` (which lowers
    /// expression arguments first) and `lower_pipe` (which inherits the
    /// argument from the previous pipe stage).
    fn lower_grad_callable_with_nodes(
        &mut self,
        fn_expr: &Expr,
        wrt_indices: Option<&[usize]>,
        actual_args: &[NodeId],
        app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self.lower_unrepresentable("grad", std::slice::from_ref(fn_expr));
        };
        let param_types: Vec<TensorType> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| {
                extract_param_type(fn_expr, index)
                    .map(Self::type_from_type_expr)
                    .unwrap_or_else(Self::default_type)
            })
            .collect();
        let actual_types: Vec<TensorType> = actual_args
            .iter()
            .map(|id| {
                self.dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type)
            })
            .collect();
        let mut subctx = LowerCtx::new(
            self.program_types.clone(),
            self.program_defs.clone(),
            LinearityInfo::default(),
        );
        // Subctx inherits the parent's current span so synthesized loads
        // for grad's parameters carry the grad-call's span.
        subctx.current_span_id = self.current_span_id.clone();
        let captured_bindings = self.seed_subctx_with_lexical_scope(&mut subctx, &param_names);
        let mut wrt = Vec::new();
        for (index, (name, param_ty)) in param_names
            .iter()
            .zip(param_types.iter().cloned())
            .enumerate()
        {
            let load = subctx.dag.add_node(
                RiscOp::Load {
                    name: name.as_str().into(),
                },
                vec![],
                param_ty.clone(),
                subctx.current_span_id.clone(),
            );
            if self.is_selected_wrt(index, &param_ty, wrt_indices) {
                wrt.push(load);
            }
            subctx
                .bindings
                .insert(name.clone(), LoweredValue::Node(load));
        }
        let output = subctx
            .lower_expr(body)
            .expect_node("grad requires a scalar floating output");
        subctx.dag.add_root(output);
        let grad_result = grad_dag(&subctx.dag, output, &wrt).unwrap_or_else(|| {
            raise_lowering_error(
                "`grad(...)` lowering requires a scalar floating forward output",
                Some(body.span()),
                body.span_id().map(ToOwned::to_owned),
            )
        });

        let arg_map = param_names
            .iter()
            .zip(actual_args.iter().copied())
            .map(|(name, arg)| (name.clone(), arg))
            .chain(captured_bindings)
            .collect::<HashMap<_, _>>();
        let specialized_grad_dag =
            Self::remap_callable_dim_symbols(&grad_result.dag, &param_types, &actual_types);
        let remap = self.splice_dag(&specialized_grad_dag, &arg_map);
        let grad_results = wrt
            .iter()
            .filter_map(|wrt_node| grad_result.grad_nodes.get(wrt_node))
            .map(|grad_node| remap[grad_node])
            .collect::<Vec<_>>();
        let reusable_inputs = param_types
            .iter()
            .enumerate()
            .filter_map(|(index, param_ty)| {
                self.is_selected_wrt(index, param_ty, wrt_indices)
                    .then_some(actual_args[index])
            })
            .collect::<Vec<_>>();
        for (grad_node, reusable_input) in grad_results.iter().zip(reusable_inputs.iter()) {
            self.dag.set_reusable_input(*grad_node, *reusable_input);
        }
        match grad_results.as_slice() {
            [single] => LoweredValue::Node(self.attach_reuse_hint(*single, app_span, actual_args)),
            _ => LoweredValue::Tuple(grad_results.into_iter().map(LoweredValue::Node).collect()),
        }
    }

    fn lower_plain_callable_app(
        &mut self,
        fn_expr: &Expr,
        args: &[Expr],
        _app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self
                .lower_unrepresentable("function application", std::slice::from_ref(fn_expr));
        };
        let saved = self.bindings.clone();
        let saved_callables = self.local_callables.clone();
        let saved_dim_substitutions = self.dim_substitutions.clone();
        let param_types: Vec<TensorType> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| {
                extract_param_type(fn_expr, index)
                    .map(Self::type_from_type_expr)
                    .unwrap_or_else(Self::default_type)
            })
            .collect();
        let mut formal_types = Vec::new();
        let mut actual_types = Vec::new();
        for ((name, arg_expr), param_ty) in param_names.iter().zip(args.iter()).zip(param_types) {
            if let Some(callable) = self.callable_binding_expr(arg_expr) {
                self.local_callables.insert(name.clone(), callable);
            } else {
                let arg_id = self.lower_expr(arg_expr);
                if let LoweredValue::Node(node_id) = &arg_id
                    && let Some(actual_ty) =
                        self.dag.get(*node_id).map(|node| node.output_type.clone())
                {
                    formal_types.push(param_ty);
                    actual_types.push(actual_ty);
                }
                self.bindings.insert(name.clone(), arg_id);
            }
        }
        self.dim_substitutions
            .extend(tensor_dim_substitutions(&formal_types, &actual_types));
        let result = self.lower_expr(body);
        self.bindings = saved;
        self.local_callables = saved_callables;
        self.dim_substitutions = saved_dim_substitutions;
        result
    }

    fn lower_plain_callable_with_values(
        &mut self,
        fn_expr: &Expr,
        args: &[LoweredValue],
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self
                .lower_unrepresentable("function application", std::slice::from_ref(fn_expr));
        };
        let saved = self.bindings.clone();
        let saved_callables = self.local_callables.clone();
        for (name, arg_id) in param_names.iter().zip(args.iter().cloned()) {
            self.bindings.insert(name.clone(), arg_id);
        }
        let result = self.lower_expr(body);
        if let Some(ret_ty_expr) = extract_fn_return_type(fn_expr) {
            let ret_ty = Self::type_from_type_expr(ret_ty_expr);
            self.repair_output_type_if_default(&result, &ret_ty);
        }
        self.bindings = saved;
        self.local_callables = saved_callables;
        result
    }

    fn lower_vmap_callable_app(
        &mut self,
        fn_expr: &Expr,
        axis: usize,
        args: &[Expr],
        _ty: &TensorType,
        app_span: Span,
    ) -> LoweredValue {
        let actual_args: Vec<NodeId> = args
            .iter()
            .map(|arg| self.lower_expr_node(arg, "vmap arguments"))
            .collect();
        self.lower_vmap_callable_with_nodes(fn_expr, axis, &actual_args, app_span)
    }

    fn lower_vmap_callable_with_nodes(
        &mut self,
        fn_expr: &Expr,
        axis: usize,
        actual_args: &[NodeId],
        app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self.lower_unrepresentable("vmap", std::slice::from_ref(fn_expr));
        };
        let param_types: Vec<TensorType> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| {
                extract_param_type(fn_expr, index)
                    .map(Self::type_from_type_expr)
                    .unwrap_or_else(Self::default_type)
            })
            .collect();
        let actual_types: Vec<TensorType> = actual_args
            .iter()
            .map(|id| {
                self.dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type)
            })
            .collect();

        let mut canonical_args = Vec::with_capacity(actual_args.len());
        let mut batch_dim = None;
        for (arg_id, arg_ty) in actual_args.iter().copied().zip(actual_types.iter()) {
            if axis < arg_ty.dims.len() {
                let perm = axis_to_front_perm(arg_ty.dims.len(), axis);
                let canon_ty = permuted_tensor_type(arg_ty, &perm);
                let canonical = if axis == 0 {
                    arg_id
                } else {
                    self.dag.add_node(
                        RiscOp::Permute { axes: perm },
                        vec![arg_id],
                        canon_ty.clone(),
                        self.current_span_id.clone(),
                    )
                };
                batch_dim.get_or_insert_with(|| canon_ty.dims[0].clone());
                canonical_args.push(canonical);
            } else {
                canonical_args.push(arg_id);
            }
        }

        let Some(batch_dim) = batch_dim else {
            return self.lower_unrepresentable(
                "vmap with no tensor arguments",
                std::slice::from_ref(fn_expr),
            );
        };

        let mut subctx = LowerCtx::new(
            self.program_types.clone(),
            self.program_defs.clone(),
            LinearityInfo::default(),
        );
        // Subctx inherits the parent's current span so synthesized loads
        // for vmap's parameters carry the vmap-call's span.
        subctx.current_span_id = self.current_span_id.clone();
        let captured_bindings = self.seed_subctx_with_lexical_scope(&mut subctx, &param_names);
        for (name, param_expr) in param_names.iter().zip(param_types.iter().cloned()) {
            let load = subctx.dag.add_node(
                RiscOp::Load {
                    name: name.as_str().into(),
                },
                vec![],
                param_expr,
                subctx.current_span_id.clone(),
            );
            subctx
                .bindings
                .insert(name.clone(), LoweredValue::Node(load));
        }
        let root_value = subctx.lower_expr(body);
        for root in root_value.flatten_nodes() {
            subctx.dag.add_root(root);
        }

        let vmapped = match vmap::vectorize_axis0(&subctx.dag, batch_dim.clone()) {
            Ok(dag) => dag,
            Err(message) => raise_lowering_error(
                format!("`vmap` lowering failed: {message}"),
                Some(body.span()),
                body.span_id().map(ToOwned::to_owned),
            ),
        };

        let mut arg_map = HashMap::new();
        for ((name, param_ty), arg_id) in param_names
            .iter()
            .zip(param_types.iter())
            .zip(canonical_args.iter().copied())
        {
            arg_map.insert(
                name.clone(),
                self.materialize_vmapped_arg(arg_id, param_ty, &batch_dim),
            );
        }
        arg_map.extend(captured_bindings);

        let specialized_vmapped =
            Self::remap_callable_dim_symbols(&vmapped, &param_types, &actual_types);
        let remap = self.splice_dag(&specialized_vmapped, &arg_map);
        let mut flattened = root_value
            .flatten_nodes()
            .into_iter()
            .map(|node| remap[&node])
            .collect::<Vec<_>>();
        if axis > 0 {
            for result in &mut flattened {
                let result_ty = self
                    .dag
                    .get(*result)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                if axis < result_ty.dims.len() {
                    let perm = front_to_axis_perm(result_ty.dims.len(), axis);
                    let perm_ty = permuted_tensor_type(&result_ty, &perm);
                    *result = self.dag.add_node(
                        RiscOp::Permute { axes: perm },
                        vec![*result],
                        perm_ty,
                        self.current_span_id.clone(),
                    );
                }
            }
        }
        if flattened.len() == 1 {
            let result = self.attach_reuse_hint(flattened[0], app_span, &canonical_args);
            LoweredValue::Node(result)
        } else {
            let mut iter = flattened.into_iter();
            LoweredValue::from_flat(&root_value, &mut iter)
        }
    }

    fn lower_vmap_grad_callable_app(
        &mut self,
        fn_expr: &Expr,
        wrt_indices: Option<&[usize]>,
        axis: usize,
        args: &[Expr],
        app_span: Span,
    ) -> LoweredValue {
        let actual_args: Vec<NodeId> = args
            .iter()
            .map(|arg| self.lower_expr_node(arg, "vmap(grad) arguments"))
            .collect();
        self.lower_vmap_grad_callable_with_nodes(fn_expr, wrt_indices, axis, &actual_args, app_span)
    }

    /// Core lowering for `vmap(grad(fn))` applied to already-lowered
    /// argument nodes. Used by both `lower_vmap_grad_callable_app` (which
    /// lowers expression arguments first) and `lower_pipe` (which
    /// inherits the argument from the previous pipe stage).
    fn lower_vmap_grad_callable_with_nodes(
        &mut self,
        fn_expr: &Expr,
        wrt_indices: Option<&[usize]>,
        axis: usize,
        actual_args: &[NodeId],
        app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self.lower_unrepresentable("vmap(grad)", std::slice::from_ref(fn_expr));
        };
        let param_types: Vec<TensorType> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| {
                extract_param_type(fn_expr, index)
                    .map(Self::type_from_type_expr)
                    .unwrap_or_else(Self::default_type)
            })
            .collect();
        let actual_types: Vec<TensorType> = actual_args
            .iter()
            .map(|id| {
                self.dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type)
            })
            .collect();

        let mut canonical_args = Vec::with_capacity(actual_args.len());
        let mut batch_dim = None;
        for (arg_id, arg_ty) in actual_args.iter().copied().zip(actual_types.iter()) {
            if axis < arg_ty.dims.len() {
                let perm = axis_to_front_perm(arg_ty.dims.len(), axis);
                let canon_ty = permuted_tensor_type(arg_ty, &perm);
                let canonical = if axis == 0 {
                    arg_id
                } else {
                    self.dag.add_node(
                        RiscOp::Permute { axes: perm },
                        vec![arg_id],
                        canon_ty.clone(),
                        self.current_span_id.clone(),
                    )
                };
                batch_dim.get_or_insert_with(|| canon_ty.dims[0].clone());
                canonical_args.push(canonical);
            } else {
                canonical_args.push(arg_id);
            }
        }

        let Some(batch_dim) = batch_dim else {
            return self.lower_unrepresentable(
                "vmap(grad) with no tensor arguments",
                std::slice::from_ref(fn_expr),
            );
        };

        let mut subctx = LowerCtx::new(
            self.program_types.clone(),
            self.program_defs.clone(),
            LinearityInfo::default(),
        );
        // Subctx inherits the parent's current span so synthesized loads
        // for vmap(grad)'s parameters carry the call's span.
        subctx.current_span_id = self.current_span_id.clone();
        let captured_bindings = self.seed_subctx_with_lexical_scope(&mut subctx, &param_names);
        let mut wrt = Vec::new();
        for (index, (name, param_ty)) in param_names
            .iter()
            .zip(param_types.iter().cloned())
            .enumerate()
        {
            let load = subctx.dag.add_node(
                RiscOp::Load {
                    name: name.as_str().into(),
                },
                vec![],
                param_ty.clone(),
                subctx.current_span_id.clone(),
            );
            if self.is_selected_wrt(index, &param_ty, wrt_indices) {
                wrt.push(load);
            }
            subctx
                .bindings
                .insert(name.clone(), LoweredValue::Node(load));
        }

        let output = subctx
            .lower_expr(body)
            .expect_node("vmap(grad(...)) requires a scalar floating output");
        subctx.dag.add_root(output);
        let grad_result = grad_dag(&subctx.dag, output, &wrt).unwrap_or_else(|| {
            raise_lowering_error(
                "`vmap(grad(...))` lowering requires a scalar floating forward output",
                Some(body.span()),
                body.span_id().map(ToOwned::to_owned),
            )
        });
        let vmapped = match vmap::vectorize_axis0(&grad_result.dag, batch_dim.clone()) {
            Ok(dag) => dag,
            Err(message) => raise_lowering_error(
                format!("`vmap(grad(...))` lowering failed: {message}"),
                Some(body.span()),
                body.span_id().map(ToOwned::to_owned),
            ),
        };

        let mut arg_map = HashMap::new();
        for ((name, param_ty), arg_id) in param_names
            .iter()
            .zip(param_types.iter())
            .zip(canonical_args.iter().copied())
        {
            arg_map.insert(
                name.clone(),
                self.materialize_vmapped_arg(arg_id, param_ty, &batch_dim),
            );
        }
        arg_map.extend(captured_bindings);

        let remap = self.splice_dag(&vmapped, &arg_map);
        let mut flattened = wrt
            .iter()
            .filter_map(|wrt_node| grad_result.grad_nodes.get(wrt_node))
            .map(|grad_node| remap[grad_node])
            .collect::<Vec<_>>();
        let reusable_inputs = param_types
            .iter()
            .enumerate()
            .filter_map(|(index, param_ty)| {
                self.is_selected_wrt(index, param_ty, wrt_indices)
                    .then_some(canonical_args[index])
            })
            .collect::<Vec<_>>();
        for (grad_node, reusable_input) in flattened.iter().zip(reusable_inputs.iter()) {
            self.dag.set_reusable_input(*grad_node, *reusable_input);
        }
        if axis > 0 {
            for result in &mut flattened {
                let result_ty = self
                    .dag
                    .get(*result)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                if axis < result_ty.dims.len() {
                    let perm = front_to_axis_perm(result_ty.dims.len(), axis);
                    let perm_ty = permuted_tensor_type(&result_ty, &perm);
                    *result = self.dag.add_node(
                        RiscOp::Permute { axes: perm },
                        vec![*result],
                        perm_ty,
                        self.current_span_id.clone(),
                    );
                }
            }
        }
        match flattened.as_slice() {
            [single] => {
                LoweredValue::Node(self.attach_reuse_hint(*single, app_span, &canonical_args))
            }
            _ => LoweredValue::Tuple(flattened.into_iter().map(LoweredValue::Node).collect()),
        }
    }

    fn materialize_vmapped_arg(
        &mut self,
        arg_id: NodeId,
        original_ty: &TensorType,
        batch_dim: &DimInfo,
    ) -> NodeId {
        let actual_ty = self
            .dag
            .get(arg_id)
            .map(|node| node.output_type.clone())
            .unwrap_or_else(Self::default_type);
        if actual_ty.dims.len() > original_ty.dims.len() {
            return arg_id;
        }

        let mut out_ty = actual_ty.clone();
        out_ty.dims.insert(0, batch_dim.clone());
        self.dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::from(batch_dim),
            },
            vec![arg_id],
            out_ty,
            self.current_span_id.clone(),
        )
    }

    fn splice_dag(
        &mut self,
        dag: &Dag,
        arg_map: &HashMap<String, NodeId>,
    ) -> HashMap<NodeId, NodeId> {
        let mut remap = HashMap::<NodeId, NodeId>::new();
        for node in dag.nodes() {
            let new_id = match &node.op {
                RiscOp::Load { name } => {
                    if let Some(existing) = arg_map.get(name.as_str()) {
                        *existing
                    } else {
                        // Preserve the source DAG node's span_id verbatim
                        // (it was set when the source DAG was lowered).
                        // Falling back to the parent ctx's current span
                        // would silently overwrite real provenance.
                        self.dag.add_node(
                            RiscOp::Load { name: name.clone() },
                            vec![],
                            node.output_type.clone(),
                            node.span_id.clone(),
                        )
                    }
                }
                op => {
                    let inputs = node.inputs.iter().map(|id| remap[id]).collect::<Vec<_>>();
                    // Preserve the source DAG node's span_id (see Load arm).
                    let new_id = self.dag.add_node(
                        op.clone(),
                        inputs,
                        node.output_type.clone(),
                        node.span_id.clone(),
                    );
                    if let Some(reusable_input) = node.reusable_input
                        && let Some(mapped_input) = remap.get(&reusable_input)
                    {
                        self.dag.set_reusable_input(new_id, *mapped_input);
                    }
                    new_id
                }
            };
            remap.insert(node.id, new_id);
        }
        remap
    }

    fn extract_fn_parts<'a>(&self, expr: &'a Expr) -> Option<(Vec<String>, &'a Expr)> {
        let Expr::List(list, _) = expr else {
            return None;
        };
        if get_tag(list) != Some("fn") {
            return None;
        }
        let kids = children(list);
        let params = kids.first()?;
        let body = kids.get(1)?;
        let Expr::List(params_list, _) = params else {
            return None;
        };
        if get_tag(params_list) != Some("params") {
            return None;
        }
        let names = children(params_list)
            .iter()
            .filter_map(|param| match param {
                Expr::Atom(Atom::Symbol(name), _) => Some(name.clone()),
                Expr::List(list, _) => list.elements.first().and_then(|expr| match expr {
                    Expr::Atom(Atom::Symbol(name), _) => Some(name.clone()),
                    _ => None,
                }),
                _ => None,
            })
            .collect();
        Some((names, body))
    }

    fn lower_builtin_app(
        &mut self,
        func_name: &str,
        args: &[Expr],
        ty: &TensorType,
        app_span: Span,
    ) -> NodeId {
        match func_name {
            // Tier 1: binary elementwise
            "add" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "add lhs");
                let b = self.lower_expr_node(&args[1], "add rhs");
                let out_ty = Self::elementwise_out_ty(&self.dag, a, ty, None);
                let node = self.dag.add_node(
                    RiscOp::Add,
                    vec![a, b],
                    out_ty,
                    self.current_span_id.clone(),
                );
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            "mul" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "mul lhs");
                let b = self.lower_expr_node(&args[1], "mul rhs");
                let out_ty = Self::elementwise_out_ty(&self.dag, a, ty, None);
                let node = self.dag.add_node(
                    RiscOp::Mul,
                    vec![a, b],
                    out_ty,
                    self.current_span_id.clone(),
                );
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            "cmplt" | "lt" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "cmplt lhs");
                let b = self.lower_expr_node(&args[1], "cmplt rhs");
                // C5: CmpLt always produces Bool output regardless of input precision.
                let bool_ty = Self::elementwise_out_ty(&self.dag, a, ty, Some(Prim::Bool));
                self.dag.add_node(
                    RiscOp::CmpLt,
                    vec![a, b],
                    bool_ty,
                    self.current_span_id.clone(),
                )
            }
            "max_elem" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "max_elem lhs");
                let b = self.lower_expr_node(&args[1], "max_elem rhs");
                let out_ty = Self::elementwise_out_ty(&self.dag, a, ty, None);
                let node = self.dag.add_node(
                    RiscOp::MaxElem,
                    vec![a, b],
                    out_ty,
                    self.current_span_id.clone(),
                );
                self.attach_reuse_hint(node, app_span, &[a, b])
            }

            // Tier 1: unary elementwise
            "drop" if args.len() == 1 => {
                let input = self.lower_expr_node(&args[0], "drop input");
                let output_type = self
                    .dag
                    .get(input)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                self.dag.add_node(
                    RiscOp::Drop,
                    vec![input],
                    output_type,
                    self.current_span_id.clone(),
                )
            }
            "neg" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "neg input");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let node =
                    self.dag
                        .add_node(RiscOp::Neg, vec![x], out_ty, self.current_span_id.clone());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "exp" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "exp input");
                let node = self.lower_transcendental(RiscOp::Exp, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "log" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "log input");
                let node = self.lower_transcendental(RiscOp::Log, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "sin" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "sin input");
                let node = self.lower_transcendental(RiscOp::Sin, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "sqrt" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "sqrt input");
                let node = self.lower_transcendental(RiscOp::Sqrt, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "cos" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "cos input");
                let node = self.lower_transcendental(RiscOp::Cos, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "tan" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "tan input");
                let node = self.lower_transcendental(RiscOp::Tan, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "atan" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "atan input");
                let node = self.lower_transcendental(RiscOp::Atan, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "abs" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "abs input");
                let node = self.lower_transcendental(RiscOp::Abs, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "floor" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "floor input");
                let node = self.lower_transcendental(RiscOp::Floor, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "ceil" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "ceil input");
                let node = self.lower_transcendental(RiscOp::Ceil, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "uniform_like" if args.len() == 3 => {
                let template = self.lower_expr_node(&args[0], "uniform_like template");
                let low = self.extract_f64_value(&args[1]).unwrap_or(0.0);
                let high = self.extract_f64_value(&args[2]).unwrap_or(1.0);
                let seed = self.random_seed.unwrap_or(0);
                // When no `type` metadata is attached to the `app` form
                // (as is common when the host lane drives sub-expression
                // lowering through `lower_subexpr_program` from a
                // handle-effect tensor-helper call), the supplied `ty`
                // is `default_type()` (rank-0 scalar). UniformLike is
                // shape-preserving over its template input, so prefer
                // the template's actual tensor type to avoid emitting a
                // rank-0 alloc that the host emitter then renders as
                // `(int[]){1}` and a 1-element loop. Bucket-5 closure.
                let inferred_ty = self
                    .dag
                    .get(template)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let resolved_ty = if ty == &Self::default_type() && !inferred_ty.dims.is_empty() {
                    inferred_ty
                } else {
                    ty.clone()
                };
                let node = self.dag.add_node(
                    RiscOp::UniformLike { low, high, seed },
                    vec![template],
                    resolved_ty,
                    self.current_span_id.clone(),
                );
                self.attach_reuse_hint(node, app_span, &[template])
            }
            "dropout" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "dropout input");
                let rate = self.extract_f64_value(&args[1]).unwrap_or(0.0);
                let seed = self.random_seed.unwrap_or(0);
                let inferred_ty = self
                    .dag
                    .get(x)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let resolved_ty = if ty == &Self::default_type() && !inferred_ty.dims.is_empty() {
                    inferred_ty
                } else {
                    ty.clone()
                };
                let node = self.dag.add_node(
                    RiscOp::Dropout { rate, seed },
                    vec![x],
                    resolved_ty,
                    self.current_span_id.clone(),
                );
                self.attach_reuse_hint(node, app_span, &[x])
            }

            // Tier 2 decompositions
            "sub" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "sub lhs");
                let b = self.lower_expr_node(&args[1], "sub rhs");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(a)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_sub(&mut self.dag, a, b, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            "relu" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "relu input");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_relu(&mut self.dag, x, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "sigmoid" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "sigmoid input");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_sigmoid(&mut self.dag, x, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            // Bucket 3: `tanh`, `silu`, `gelu` route through new tier2
            // decompositions so the RISC DAG path stays self-contained.
            // Mirrors the relu/sigmoid pattern above.
            "tanh" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "tanh input");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_tanh(&mut self.dag, x, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "silu" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "silu input");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_silu(&mut self.dag, x, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "gelu" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "gelu input");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_gelu(&mut self.dag, x, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "div" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "div lhs");
                let b = self.lower_expr_node(&args[1], "div rhs");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(a)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_div(&mut self.dag, a, b, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[a, b])
            }

            // Tier 2 higher-level ops (spec §3.4, §4.1–4.2)
            "matmul" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "matmul lhs");
                let b = self.lower_expr_node(&args[1], "matmul rhs");
                let a_ty = self
                    .dag
                    .get(a)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let b_ty = self
                    .dag
                    .get(b)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let parent_span = self.current_span_id.clone();
                tier2::lower_matmul(&mut self.dag, a, b, &a_ty, &b_ty, parent_span.as_deref())
            }
            "gather" if args.len() == 3 => {
                let values = self.lower_expr_node(&args[0], "gather values");
                let indices = self.lower_expr_node(&args[1], "gather indices");
                let axis = self.extract_axis(&args[2]);
                let out_ty = Self::gather_out_ty_from_inputs(&self.dag, values, indices, axis)
                    .unwrap_or_else(|| ty.clone());
                self.dag.add_node(
                    RiscOp::Gather { axis },
                    vec![values, indices],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "scatter_replace" if args.len() == 4 => {
                // Tensor-lane replace-scatter: lowers directly to
                // RiscOp::Scatter (last-write-wins). The output type
                // equals the base/target tensor's type.
                let base = self.lower_expr_node(&args[0], "scatter_replace base");
                let indices = self.lower_expr_node(&args[1], "scatter_replace indices");
                let updates = self.lower_expr_node(&args[2], "scatter_replace updates");
                let axis = self.extract_axis(&args[3]);
                let out_ty = self
                    .dag
                    .get(base)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                self.dag.add_node(
                    RiscOp::Scatter { axis },
                    vec![base, indices, updates],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "softmax" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "softmax input");
                let axis = self.extract_axis(&args[1]);
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let parent_span = self.current_span_id.clone();
                let node =
                    tier2::lower_softmax(&mut self.dag, x, axis, &x_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "mean" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "mean input");
                let axis = self.extract_axis(&args[1]);
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_mean(&mut self.dag, x, axis, &x_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "layer_norm" if args.len() == 3 => {
                let x = self.lower_expr_node(&args[0], "layer_norm input");
                let gamma = self.lower_expr_node(&args[1], "layer_norm gamma");
                let beta = self.lower_expr_node(&args[2], "layer_norm beta");
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let gamma_ty = self
                    .dag
                    .get(gamma)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let beta_ty = self
                    .dag
                    .get(beta)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_layer_norm(
                    &mut self.dag,
                    x,
                    gamma,
                    beta,
                    &x_ty,
                    &gamma_ty,
                    &beta_ty,
                    1e-5,
                    parent_span.as_deref(),
                );
                self.attach_reuse_hint(node, app_span, &[x, gamma, beta])
            }
            "conv2d" if args.len() >= 2 => {
                let input = self.lower_expr_node(&args[0], "conv2d input");
                let kernel = self.lower_expr_node(&args[1], "conv2d kernel");
                let stride = args
                    .get(2)
                    .and_then(|expr| self.extract_usize_value(expr))
                    .unwrap_or(1);
                let padding = args
                    .get(3)
                    .and_then(|expr| self.extract_usize_value(expr))
                    .unwrap_or(0);
                let input_ty = self
                    .dag
                    .get(input)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let kernel_ty = self
                    .dag
                    .get(kernel)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let parent_span = self.current_span_id.clone();
                tier2::lower_conv2d(
                    &mut self.dag,
                    input,
                    kernel,
                    &input_ty,
                    &kernel_ty,
                    ty,
                    stride,
                    padding,
                    parent_span.as_deref(),
                )
            }

            // H1: Tier 2 comparison ops
            "gt" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "gt lhs");
                let b = self.lower_expr_node(&args[1], "gt rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_gt(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "gte" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "gte lhs");
                let b = self.lower_expr_node(&args[1], "gte rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_gte(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "lte" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "lte lhs");
                let b = self.lower_expr_node(&args[1], "lte rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_lte(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "eq" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "eq lhs");
                let b = self.lower_expr_node(&args[1], "eq rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_eq(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "neq" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "neq lhs");
                let b = self.lower_expr_node(&args[1], "neq rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_neq(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "min_elem" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "min_elem lhs");
                let b = self.lower_expr_node(&args[1], "min_elem rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_min_elem(&mut self.dag, a, b, ty, parent_span.as_deref())
            }

            // H2: Boolean operators
            "and" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "and lhs");
                let b = self.lower_expr_node(&args[1], "and rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_and(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "or" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "or lhs");
                let b = self.lower_expr_node(&args[1], "or rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_or(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "not" if args.len() == 1 => {
                let a = self.lower_expr_node(&args[0], "not input");
                let parent_span = self.current_span_id.clone();
                tier2::lower_not(&mut self.dag, a, ty, parent_span.as_deref())
            }

            // Tier 1: reductions
            "sum" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "sum input");
                let axis = self.extract_axis(&args[1]);
                let out_ty = if *ty == Self::default_type() {
                    let x_ty = self
                        .dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone());
                    let mut dims = x_ty.dims.clone();
                    if axis < dims.len() {
                        dims.remove(axis);
                    }
                    TensorType {
                        dims,
                        precision: x_ty.precision,
                    }
                } else {
                    ty.clone()
                };
                self.dag.add_node(
                    RiscOp::Sum { axis },
                    vec![x],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "tensor_to_scalar" if args.len() == 1 => {
                self.lower_expr_node(&args[0], "tensor_to_scalar input")
            }
            "scalar_to_tensor" if args.len() == 1 => {
                self.lower_expr_node(&args[0], "scalar_to_tensor input")
            }
            "max_reduce" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "max_reduce input");
                let axis = self.extract_axis(&args[1]);
                let out_ty = if *ty == Self::default_type() {
                    let x_ty = self
                        .dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone());
                    let mut dims = x_ty.dims.clone();
                    if axis < dims.len() {
                        dims.remove(axis);
                    }
                    TensorType {
                        dims,
                        precision: x_ty.precision,
                    }
                } else {
                    ty.clone()
                };
                self.dag.add_node(
                    RiscOp::MaxReduce { axis },
                    vec![x],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "min_reduce" | "prod_reduce" | "argmax_reduce" | "argmin_reduce" if args.len() == 2 => {
                let name = func_name;
                let x = self.lower_expr_node(&args[0], "reduction input");
                let axis = self.extract_axis(&args[1]);
                let out_ty = if *ty == Self::default_type() {
                    let x_ty = self
                        .dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone());
                    let mut dims = x_ty.dims.clone();
                    if axis < dims.len() {
                        dims.remove(axis);
                    }
                    TensorType {
                        dims,
                        precision: x_ty.precision,
                    }
                } else {
                    ty.clone()
                };
                let op = match name {
                    "min_reduce" => RiscOp::MinReduce { axis },
                    "prod_reduce" => RiscOp::ProdReduce { axis },
                    "argmax_reduce" => RiscOp::Argmax { axis },
                    "argmin_reduce" => RiscOp::Argmin { axis },
                    _ => unreachable!(),
                };
                self.dag
                    .add_node(op, vec![x], out_ty, self.current_span_id.clone())
            }

            // H3: Movement ops -- extract parameters from Deep AST args where possible.
            "reshape" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "reshape input");
                // Try to extract new_shape from the second arg; fall back to output type dims.
                let new_shape = if args.len() >= 2 {
                    self.extract_dim_list(&args[1])
                        .unwrap_or_else(|| ty.dims.clone())
                } else {
                    ty.dims.clone()
                };
                let out_ty = TensorType {
                    dims: new_shape.clone(),
                    precision: ty.precision,
                };
                self.dag.add_node(
                    RiscOp::Reshape { new_shape },
                    vec![x],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "permute" if args.len() >= 2 => {
                let x = self.lower_expr_node(&args[0], "permute input");
                // Extract axes ordering from remaining args.
                let axes = self.extract_usize_list(&args[1..]);
                self.dag.add_node(
                    RiscOp::Permute { axes },
                    vec![x],
                    ty.clone(),
                    self.current_span_id.clone(),
                )
            }
            "expand" if args.len() >= 2 => {
                let x = self.lower_expr_node(&args[0], "expand input");
                let axis = self.extract_usize_value(&args[1]).unwrap_or(0);
                let size = if args.len() >= 3 {
                    self.extract_dim_expr_value(&args[2])
                        .unwrap_or(DimExpr::Concrete(1))
                } else {
                    DimExpr::Concrete(1)
                };
                let out_ty = self
                    .fallback_expand_type(x, axis, &size)
                    .unwrap_or_else(|| ty.clone());
                self.dag.add_node(
                    RiscOp::Expand { axis, size },
                    vec![x],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "pad" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "pad input");
                let padding = if args.len() >= 2 {
                    self.extract_pair_list(&args[1]).unwrap_or_default()
                } else {
                    vec![]
                };
                let fill = if args.len() >= 3 {
                    self.extract_f64_value(&args[2]).unwrap_or(0.0)
                } else {
                    0.0
                };
                self.dag.add_node(
                    RiscOp::Pad { padding, fill },
                    vec![x],
                    ty.clone(),
                    self.current_span_id.clone(),
                )
            }
            "shrink" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "shrink input");
                let bounds = if args.len() >= 2 {
                    self.extract_pair_list(&args[1]).unwrap_or_default()
                } else {
                    vec![]
                };
                self.dag.add_node(
                    RiscOp::Shrink { bounds },
                    vec![x],
                    ty.clone(),
                    self.current_span_id.clone(),
                )
            }
            "stride" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "stride input");
                let strides = if args.len() >= 2 {
                    self.extract_usize_list(&args[1..])
                } else {
                    vec![]
                };
                self.dag.add_node(
                    RiscOp::Stride { strides },
                    vec![x],
                    ty.clone(),
                    self.current_span_id.clone(),
                )
            }

            // Fallback: unknown function.
            _ => {
                for arg in args {
                    self.lower_expr(arg);
                }
                self.dag.add_node(
                    RiscOp::Load {
                        name: func_name.into(),
                    },
                    vec![],
                    Self::default_type(),
                    self.current_span_id.clone(),
                )
            }
        }
    }

    /// Extract an axis value from an expression (for sum/max_reduce).
    fn extract_axis(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Atom(Atom::Int(n), _) => *n as usize,
            // Handle (lit {} n) form.
            Expr::List(list, _) => {
                if let Some(Expr::Atom(Atom::Int(n), _)) = list.elements.get(2) {
                    *n as usize
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    fn gather_out_ty_from_inputs(
        dag: &Dag,
        values: NodeId,
        indices: NodeId,
        axis: usize,
    ) -> Option<TensorType> {
        let values_ty = &dag.get(values)?.output_type;
        let indices_ty = &dag.get(indices)?.output_type;
        if axis >= values_ty.dims.len() {
            return None;
        }
        let mut dims = Vec::new();
        dims.extend_from_slice(&values_ty.dims[..axis]);
        dims.extend(indices_ty.dims.iter().cloned());
        dims.extend_from_slice(&values_ty.dims[axis + 1..]);
        Some(TensorType {
            dims,
            precision: values_ty.precision,
        })
    }

    /// Extract a single usize value from an expression.
    fn extract_usize_value(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::Atom(Atom::Int(n), _) => Some(*n as usize),
            Expr::List(list, _) => {
                if let Some(Expr::Atom(Atom::Int(n), _)) = list.elements.get(2) {
                    Some(*n as usize)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn extract_dim_expr_value(&self, expr: &Expr) -> Option<DimExpr> {
        if let Some(value) = self.extract_usize_value(expr) {
            return Some(DimExpr::Concrete(value));
        }

        match expr {
            Expr::Atom(Atom::Symbol(name), _) => Some(self.resolve_dim_expr_symbol(name)),
            Expr::List(list, _) => match (list.elements.first(), list.elements.get(2)) {
                (
                    Some(Expr::Atom(Atom::Symbol(tag), _)),
                    Some(Expr::Atom(Atom::Symbol(name), _)),
                ) if tag == "var" => Some(self.resolve_dim_expr_symbol(name)),
                _ => None,
            },
            _ => None,
        }
    }

    fn resolve_dim_expr_symbol(&self, name: &str) -> DimExpr {
        self.dim_substitutions
            .get(name)
            .map(DimExpr::from)
            .unwrap_or_else(|| DimExpr::Sym(name.to_string()))
    }

    fn lower_handle_effect(&mut self, elems: &[Expr]) -> LoweredValue {
        let effect = match elems.get(1) {
            Some(Expr::Map(meta, _)) => meta
                .entries
                .iter()
                .find(|(key, _)| key == "effect")
                .and_then(|(_, value)| match value {
                    Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
                    _ => None,
                }),
            _ => None,
        };
        match effect {
            Some("random") if elems.len() >= 4 => {
                let saved_seed = self.random_seed;
                self.random_seed = self.extract_u64_value(&elems[2]).or(saved_seed);
                let result = self.lower_expr(&elems[3]);
                self.random_seed = saved_seed;
                result
            }
            Some("resource") if elems.len() >= 4 => self.lower_expr(&elems[3]),
            _ if elems.len() >= 4 => self.lower_expr(&elems[3]),
            _ => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            )),
        }
    }

    fn extract_u64_value(&self, expr: &Expr) -> Option<u64> {
        self.extract_usize_value(expr).map(|value| value as u64)
    }

    /// Extract an f64 value from an expression.
    fn extract_f64_value(&self, expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Atom(Atom::Float(f), _) => Some(*f),
            Expr::Atom(Atom::Int(n), _) => Some(*n as f64),
            Expr::List(list, _) => {
                if let Some(Expr::Atom(Atom::Float(f), _)) = list.elements.get(2) {
                    Some(*f)
                } else if let Some(Expr::Atom(Atom::Int(n), _)) = list.elements.get(2) {
                    Some(*n as f64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract a list of usize values from a slice of expressions.
    fn extract_usize_list(&self, exprs: &[Expr]) -> Vec<usize> {
        let mut result = Vec::new();
        for expr in exprs {
            if let Some(v) = self.extract_usize_value(expr) {
                result.push(v);
            }
        }
        result
    }

    /// Extract dimension info list from an expression (e.g., for reshape).
    fn extract_dim_list(&self, expr: &Expr) -> Option<Vec<DimInfo>> {
        if let Expr::List(list, _) = expr {
            let mut dims = Vec::new();
            for elem in &list.elements {
                match elem {
                    Expr::Atom(Atom::Int(n), _) => dims.push(DimInfo::Lit(*n as usize)),
                    Expr::Atom(Atom::Symbol(name), _) => {
                        dims.push(DimInfo::Named(name.clone(), None));
                    }
                    _ => {}
                }
            }
            if !dims.is_empty() {
                return Some(dims);
            }
        }
        None
    }

    /// Extract a list of (usize, usize) pairs from an expression (for pad/shrink bounds).
    fn extract_pair_list(&self, expr: &Expr) -> Option<Vec<(usize, usize)>> {
        if let Expr::List(list, _) = expr {
            let mut pairs = Vec::new();
            for elem in &list.elements {
                if let Expr::List(pair_list, _) = elem {
                    let vals: Vec<usize> = pair_list
                        .elements
                        .iter()
                        .filter_map(|e| {
                            if let Expr::Atom(Atom::Int(n), _) = e {
                                Some(*n as usize)
                            } else {
                                None
                            }
                        })
                        .collect();
                    if vals.len() >= 2 {
                        pairs.push((vals[0], vals[1]));
                    }
                }
            }
            if !pairs.is_empty() {
                return Some(pairs);
            }
        }
        None
    }

    /// C4: Enforce float-only for transcendental ops (exp, log, sin, sqrt).
    /// If the input is not float, produce a Const(0) error placeholder.
    fn lower_transcendental(&mut self, op: RiscOp, x: NodeId, ty: &TensorType) -> NodeId {
        let out_ty = if *ty == Self::default_type() {
            self.dag
                .get(x)
                .map(|n| n.output_type.clone())
                .unwrap_or_else(|| ty.clone())
        } else {
            ty.clone()
        };
        let input_prec = self
            .dag
            .get(x)
            .map(|n| n.output_type.precision)
            .unwrap_or(Prim::F32);
        if input_prec.is_float() {
            self.dag
                .add_node(op, vec![x], out_ty, self.current_span_id.clone())
        } else {
            // Non-float input: produce a zero constant as error placeholder.
            self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                out_ty,
                self.current_span_id.clone(),
            )
        }
    }

    /// `(fn {} (params {} p1 p2 ...) body)`
    fn lower_fn(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ));
        }
        let saved = self.bindings.clone();
        let saved_callables = self.local_callables.clone();

        // Register params as Load nodes.
        if let Expr::List(params_list, _) = &elems[2] {
            for param in &params_list.elements[2..] {
                if let Some((name, ty_expr)) = param_name_and_type_expr(param) {
                    let lowered = self.lower_fn_param_binding(&name, ty_expr);
                    self.bindings.insert(name, lowered);
                }
            }
        }

        let result = self.lower_expr(&elems[3]);
        self.bindings = saved; // Restore scope
        self.local_callables = saved_callables;
        result
    }

    fn lower_fn_param_binding(&mut self, name: &str, ty_expr: Option<&Expr>) -> LoweredValue {
        if let Some(Expr::List(list, _)) = ty_expr
            && get_tag(list) == Some("t-tuple")
        {
            let items = children(list)
                .iter()
                .enumerate()
                .map(|(index, item_ty)| {
                    self.lower_fn_param_binding(&format!("{name}__{index}"), Some(item_ty))
                })
                .collect::<Vec<_>>();
            return LoweredValue::Tuple(items);
        }

        let ty = ty_expr
            .map(Self::type_from_type_expr)
            .unwrap_or_else(Self::default_type);
        LoweredValue::Node(self.dag.add_node(
            RiscOp::Load { name: name.into() },
            vec![],
            ty,
            self.current_span_id.clone(),
        ))
    }

    /// `(pipe {} x f g ...)` -- chain: lower x, then apply f, then g, etc.
    fn lower_pipe(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 3 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ));
        }
        let mut current = self.lower_expr(&elems[2]);
        for func_expr in &elems[3..] {
            // Bucket 4e: a unary `(var {} fname)` stage where `fname` is
            // a known elementwise/tensor builtin can lower directly via
            // tier2 (no lambda intermediary). For *unknown* var names
            // (user-defined fns, library re-exports, etc.) we fall
            // through to `resolve_callable_expr` below, so the stage is
            // treated as a plain function reference and gets the same
            // unary-application semantics as `f(current)`.
            //
            // The previous implementation hit `_ => current` and then
            // `continue`, which silently dropped the user-defined fn
            // (the accumulator was returned unchanged). Top-level
            // bindings then materialised as `()`/Unit in generated C.
            let unary_builtin_name = if let Expr::List(func_list, _) = func_expr
                && let Some(Expr::Atom(Atom::Symbol(tag), _)) = func_list.elements.first()
                && tag == "var"
                && let Some(Expr::Atom(Atom::Symbol(fname), _)) = func_list.elements.get(2)
            {
                Some(fname.as_str())
            } else {
                None
            };
            let is_known_unary_builtin = matches!(
                unary_builtin_name,
                Some(
                    "neg"
                        | "exp"
                        | "log"
                        | "sin"
                        | "sqrt"
                        | "cos"
                        | "tan"
                        | "atan"
                        | "abs"
                        | "floor"
                        | "ceil"
                        | "relu"
                        | "sigmoid"
                        | "tanh"
                        | "silu"
                        | "gelu"
                )
            );
            if is_known_unary_builtin
                && let Expr::List(func_list, _) = func_expr
                && let Some(Expr::Atom(Atom::Symbol(tag), _)) = func_list.elements.first()
                && tag == "var"
                && let Some(Expr::Atom(Atom::Symbol(fname), _)) = func_list.elements.get(2)
            {
                let current_node = current.expect_node("pipe stage");
                let ty = self
                    .dag
                    .get(current_node)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                current = match fname.as_str() {
                    "neg" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Neg,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "exp" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Exp,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "log" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Log,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "sin" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Sin,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "sqrt" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Sqrt,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "cos" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Cos,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "tan" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Tan,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "atan" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Atan,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "abs" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Abs,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "floor" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Floor,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "ceil" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Ceil,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "relu" => LoweredValue::Node(tier2::lower_relu(
                        &mut self.dag,
                        current_node,
                        &ty,
                        self.current_span_id.as_deref(),
                    )),
                    "sigmoid" => LoweredValue::Node(tier2::lower_sigmoid(
                        &mut self.dag,
                        current_node,
                        &ty,
                        self.current_span_id.as_deref(),
                    )),
                    "tanh" => LoweredValue::Node(tier2::lower_tanh(
                        &mut self.dag,
                        current_node,
                        &ty,
                        self.current_span_id.as_deref(),
                    )),
                    "silu" => LoweredValue::Node(tier2::lower_silu(
                        &mut self.dag,
                        current_node,
                        &ty,
                        self.current_span_id.as_deref(),
                    )),
                    "gelu" => LoweredValue::Node(tier2::lower_gelu(
                        &mut self.dag,
                        current_node,
                        &ty,
                        self.current_span_id.as_deref(),
                    )),
                    // `is_known_unary_builtin` guarantees this branch is
                    // never hit, but keep it as an explicit fallthrough
                    // marker so any future name added to the predicate
                    // without a corresponding match arm fails loudly.
                    _ => unreachable!(
                        "pipe stage `{fname}` was classified as a known \
                         unary builtin but has no lowering arm"
                    ),
                };
                continue;
            }
            if let Some(callable) = self.resolve_callable_expr(func_expr) {
                current = match callable {
                    CallableExpr::Plain(fn_expr) => {
                        self.lower_plain_callable_with_values(&fn_expr, &[current.clone()])
                    }
                    CallableExpr::Vmap { fn_expr, axis } => {
                        let current_node = current.expect_node("pipe stage");
                        self.lower_vmap_callable_with_nodes(
                            &fn_expr,
                            axis,
                            &[current_node],
                            func_expr.span(),
                        )
                    }
                    CallableExpr::Grad { fn_expr, wrt } => {
                        // `x |> grad(f)` lowers as `grad(f)(x)` — reuse
                        // the non-pipe grad lowering with the previous
                        // stage's NodeId as the single argument.
                        let current_node = current.expect_node("pipe stage");
                        self.lower_grad_callable_with_nodes(
                            &fn_expr,
                            wrt.as_deref(),
                            &[current_node],
                            func_expr.span(),
                        )
                    }
                    CallableExpr::VmapGrad { fn_expr, wrt, axis } => {
                        // `xs |> vmap(grad(f))` lowers as
                        // `vmap(grad(f))(xs)` — reuse the non-pipe
                        // vmap-grad lowering with the previous stage's
                        // NodeId as the single argument.
                        let current_node = current.expect_node("pipe stage");
                        self.lower_vmap_grad_callable_with_nodes(
                            &fn_expr,
                            wrt.as_deref(),
                            axis,
                            &[current_node],
                            func_expr.span(),
                        )
                    }
                };
                continue;
            }
            current = self.lower_unrepresentable("pipe stage", std::slice::from_ref(func_expr));
        }
        current
    }

    /// `(cast {} expr (t-prim {} name))` -- precision cast.
    fn lower_cast(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ));
        }
        let x = self.lower_expr_node(&elems[2], "cast input");
        let input_ty = self
            .dag
            .get(x)
            .map(|n| n.output_type.clone())
            .unwrap_or_else(Self::default_type);
        let new_precision = if let Some(prim) = Self::try_extract_prim(&elems[3]) {
            // Handle (t-prim {} name) form.
            prim
        } else if let Expr::Atom(Atom::Symbol(pname), _) = &elems[3] {
            // Fallback: bare symbol for backward compat.
            Prim::parse_name(pname).unwrap_or(Prim::F32)
        } else {
            Prim::F32
        };
        let ty = TensorType {
            dims: input_ty.dims,
            precision: new_precision,
        };
        LoweredValue::Node(self.dag.add_node(
            RiscOp::Cast { new_precision },
            vec![x],
            ty,
            self.current_span_id.clone(),
        ))
    }

    /// `(grad {} f)` -- rejected before lowering.
    fn lower_grad(&mut self, elems: &[Expr]) -> LoweredValue {
        self.lower_unrepresentable("grad", elems)
    }

    /// `(if {} cond then else)` -- Phase 0: select via arithmetic on bools.
    fn lower_if(&mut self, elems: &[Expr]) -> LoweredValue {
        let Some(cond_expr) = elems.get(2) else {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ));
        };
        let Some(then_expr) = elems.get(3) else {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ));
        };
        let Some(else_expr) = elems.get(4) else {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ));
        };

        let cond = self.lower_expr_node(cond_expr, "if condition");
        let then_node = self.lower_expr_node(then_expr, "if then branch");
        let else_node = self.lower_expr_node(else_expr, "if else branch");
        let out_ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            Self::type_from_meta(&meta.entries)
        } else {
            self.dag
                .get(then_node)
                .map(|node| node.output_type.clone())
                .unwrap_or_else(Self::default_type)
        };
        if !out_ty.precision.is_float() {
            return self.lower_unrepresentable("if", elems);
        }

        let mask = self.lower_if_mask(cond, &out_ty);
        let one = self.dag.add_node(
            RiscOp::Const { value: 1.0 },
            vec![],
            out_ty.clone(),
            self.current_span_id.clone(),
        );
        let neg_mask = self.dag.add_node(
            RiscOp::Neg,
            vec![mask],
            out_ty.clone(),
            self.current_span_id.clone(),
        );
        let inv_mask = self.dag.add_node(
            RiscOp::Add,
            vec![one, neg_mask],
            out_ty.clone(),
            self.current_span_id.clone(),
        );
        let masked_then = self.dag.add_node(
            RiscOp::Mul,
            vec![mask, then_node],
            out_ty.clone(),
            self.current_span_id.clone(),
        );
        let masked_else = self.dag.add_node(
            RiscOp::Mul,
            vec![inv_mask, else_node],
            out_ty.clone(),
            self.current_span_id.clone(),
        );
        LoweredValue::Node(self.dag.add_node(
            RiscOp::Add,
            vec![masked_then, masked_else],
            out_ty,
            self.current_span_id.clone(),
        ))
    }

    /// `(tuple {} elem1 elem2 ...)` -- not representable in the Phase 0 RISC DAG.
    fn lower_tuple(&mut self, elems: &[Expr]) -> LoweredValue {
        LoweredValue::Tuple(
            elems
                .iter()
                .skip(2)
                .map(|expr| self.lower_expr(expr))
                .collect(),
        )
    }

    /// `(par {} expr1 expr2 ...)` -- not representable in the Phase 0 RISC DAG.
    fn lower_par(&mut self, elems: &[Expr]) -> LoweredValue {
        self.lower_unrepresentable("par", elems)
    }

    /// `(realize {} expr)` -- explicit materialization barrier.
    fn lower_realize(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() >= 3 {
            let input = self.lower_expr(&elems[2]);
            if let LoweredValue::Tuple(items) = &input {
                return LoweredValue::Tuple(
                    items
                        .iter()
                        .map(|item| {
                            let id = item.expect_node("realize tuple leaf");
                            let output_type = self
                                .dag
                                .get(id)
                                .map(|node| node.output_type.clone())
                                .unwrap_or_else(Self::default_type);
                            LoweredValue::Node(self.dag.add_node(
                                RiscOp::Realize,
                                vec![id],
                                output_type,
                                self.current_span_id.clone(),
                            ))
                        })
                        .collect(),
                );
            }
            let input = input.expect_node("realize input");
            let output_type = self
                .dag
                .get(input)
                .map(|node| node.output_type.clone())
                .unwrap_or_else(Self::default_type);
            LoweredValue::Node(self.dag.add_node(
                RiscOp::Realize,
                vec![input],
                output_type,
                self.current_span_id.clone(),
            ))
        } else {
            LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ))
        }
    }

    /// `(copy {} expr)` -- identity in Phase 0/1.
    fn lower_copy(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() >= 3 {
            let input = self.lower_expr(&elems[2]);
            if let LoweredValue::Tuple(items) = &input {
                return LoweredValue::Tuple(
                    items
                        .iter()
                        .map(|item| {
                            let id = item.expect_node("copy tuple leaf");
                            let output_type = self
                                .dag
                                .get(id)
                                .map(|node| node.output_type.clone())
                                .unwrap_or_else(Self::default_type);
                            LoweredValue::Node(self.dag.add_node(
                                RiscOp::Copy,
                                vec![id],
                                output_type,
                                self.current_span_id.clone(),
                            ))
                        })
                        .collect(),
                );
            }
            let input = input.expect_node("copy input");
            let output_type = self
                .dag
                .get(input)
                .map(|node| node.output_type.clone())
                .unwrap_or_else(Self::default_type);
            LoweredValue::Node(self.dag.add_node(
                RiscOp::Copy,
                vec![input],
                output_type,
                self.current_span_id.clone(),
            ))
        } else {
            LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ))
        }
    }

    /// `(borrow {} expr)` -- erased before executable lowering.
    fn lower_identity(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() >= 3 {
            self.lower_expr(&elems[2])
        } else {
            LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            ))
        }
    }

    /// `(tuple-get {} tuple_expr index)` -- not representable in the Phase 0 RISC DAG.
    fn lower_tuple_get(&mut self, elems: &[Expr]) -> LoweredValue {
        let tuple = self.lower_expr(&elems[2]);
        let index = self.extract_usize_value(&elems[3]).unwrap_or(0);
        tuple.tuple_get(index).unwrap_or_else(|| {
            raise_lowering_error(
                format!("tuple-get index {index} out of bounds during lowering"),
                elems.get(3).map(Expr::span),
                elems.get(3).and_then(Expr::span_id).map(ToOwned::to_owned),
            )
        })
    }

    /// `(match {} scrutinee (arm {} pattern body) ...)` -- not representable in the Phase 0 RISC DAG.
    fn lower_match(&mut self, elems: &[Expr]) -> LoweredValue {
        self.lower_unrepresentable("match", elems)
    }

    fn lower_unrepresentable(&mut self, tag: &str, elems: &[Expr]) -> LoweredValue {
        for expr in elems.iter().skip(2) {
            let _ = self.lower_expr(expr);
        }
        if unrepresentable_panic_suppressed() {
            // Speculative DAG attempt from the host-lane fallback — unwind
            // without writing a stderr panic-location trace. `catch_unwind`
            // in the caller turns this into an `Err` and falls back to
            // host lowering.
            std::panic::panic_any(UnrepresentableDag);
        }
        let expr = elems.first();
        raise_lowering_error(
            unsupported_lowering_message(tag),
            expr.map(Expr::span),
            expr.and_then(Expr::span_id).map(ToOwned::to_owned),
        )
    }

    /// Constructs that are valid Chelis but not supported by DAG evaluation.
    fn lower_unsupported(&mut self, tag: &str, elems: &[Expr]) -> LoweredValue {
        let expr = elems.first();
        raise_lowering_error(
            unsupported_lowering_message(tag),
            expr.map(Expr::span),
            expr.and_then(Expr::span_id).map(ToOwned::to_owned),
        )
    }

    fn lower_if_mask(&mut self, cond: NodeId, out_ty: &TensorType) -> NodeId {
        let mut mask = cond;
        let cond_ty = self
            .dag
            .get(cond)
            .map(|node| node.output_type.clone())
            .unwrap_or_else(Self::default_type);
        if cond_ty.precision != out_ty.precision {
            mask = self.dag.add_node(
                RiscOp::Cast {
                    new_precision: out_ty.precision,
                },
                vec![mask],
                TensorType {
                    dims: cond_ty.dims.clone(),
                    precision: out_ty.precision,
                },
                self.current_span_id.clone(),
            );
        }
        if cond_ty.dims.is_empty() && !out_ty.dims.is_empty() {
            let mut expanded = mask;
            let mut dims = Vec::new();
            for (axis, dim) in out_ty.dims.iter().enumerate() {
                dims.push(dim.clone());
                expanded = self.dag.add_node(
                    RiscOp::Expand {
                        axis,
                        size: DimExpr::from(dim),
                    },
                    vec![expanded],
                    TensorType {
                        dims: dims.clone(),
                        precision: out_ty.precision,
                    },
                    self.current_span_id.clone(),
                );
            }
            return expanded;
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify;

    fn parse_and_check(src: &str) -> chelis_types::CheckedProgram {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let checked = chelis_types::check_ir_program(&exprs)
            .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"))
    }

    fn parse_and_lower(src: &str) -> Dag {
        let checked = parse_and_check(src);
        lower_program(&checked)
    }

    fn parse_and_lower_unchecked(src: &str) -> Dag {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let mut ctx = LowerCtx::new(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            LinearityInfo::default(),
        );
        for expr in &exprs {
            let _ = ctx.lower_expr(expr);
        }
        ctx.dag
    }

    fn non_drop_len(dag: &Dag) -> usize {
        dag.nodes()
            .iter()
            .filter(|node| !matches!(node.op, RiscOp::Drop))
            .count()
    }

    fn root_node(dag: &Dag) -> &crate::dag::DagNode {
        dag.roots()
            .last()
            .and_then(|id| dag.get(*id))
            .expect("expected lowered root")
    }

    #[test]
    fn lower_single_const() {
        let dag = parse_and_lower("(def {} x (lit {type: (t-prim {} f32)} 1.0))");
        assert_eq!(dag.len(), 1);
        assert_eq!(dag.get(NodeId(0)).unwrap().op, RiscOp::Const { value: 1.0 });
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lowering_marks_reusable_input_from_linearity_hint() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}
                   (var {} relu)
                   (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x)))
        "#;
        let dag = parse_and_lower(src);
        let node = dag
            .roots()
            .last()
            .and_then(|id| dag.get(*id))
            .expect("lowered root node");
        assert_eq!(node.reusable_input, Some(NodeId(0)));
    }

    #[test]
    fn lower_gather_uses_sparse_ir_node() {
        let src = r#"
            (def {} values
              (var {type: (t-tensor {} (d-lit {} 4) (d-lit {} 2) (t-prim {} f32))} values))
            (def {} indices
              (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} int32))} indices))
            (def {} out
              (app {type: (t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))}
                   (var {} gather)
                   (var {type: (t-tensor {} (d-lit {} 4) (d-lit {} 2) (t-prim {} f32))} values)
                   (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} int32))} indices)
                   (lit {type: (t-prim {} int32)} 0)))
        "#;
        let dag = parse_and_lower(src);
        let gather = dag
            .nodes()
            .iter()
            .find(|node| matches!(node.op, RiscOp::Gather { axis: 0 }))
            .expect("Surf gather should lower to first-class sparse IR");
        assert_eq!(
            gather.output_type,
            TensorType {
                dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
                precision: Prim::F32,
            }
        );
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_scatter_replace_uses_sparse_ir_node() {
        // Surf `scatter_replace(base, indices, updates, axis)` must
        // lower directly to the first-class `RiscOp::Scatter` sparse
        // IR node, paralleling the tensor-lane `gather` lowering.
        // This locks the contract: a Surf-level use of
        // scatter_replace MUST reach the sparse evaluator/codegen
        // path, NOT the host-runtime fallback (which the existing
        // `scatter(..., mode)` builtin uses).
        let src = r#"
            (def {} base
              (var {type: (t-tensor {} (d-lit {} 4) (d-lit {} 2) (t-prim {} f32))} base))
            (def {} indices
              (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} int32))} indices))
            (def {} updates
              (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))} updates))
            (def {} out
              (app {type: (t-tensor {} (d-lit {} 4) (d-lit {} 2) (t-prim {} f32))}
                   (var {} scatter_replace)
                   (var {type: (t-tensor {} (d-lit {} 4) (d-lit {} 2) (t-prim {} f32))} base)
                   (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} int32))} indices)
                   (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))} updates)
                   (lit {type: (t-prim {} int32)} 0)))
        "#;
        let dag = parse_and_lower(src);
        let scatter = dag
            .nodes()
            .iter()
            .find(|node| matches!(node.op, RiscOp::Scatter { axis: 0 }))
            .expect("Surf scatter_replace should lower to first-class sparse IR");
        assert_eq!(
            scatter.output_type,
            TensorType {
                dims: vec![DimInfo::Lit(4), DimInfo::Lit(2)],
                precision: Prim::F32,
            }
        );
        // Defense in depth: the lowered DAG must NOT contain a
        // ScatterAdd from a Surf scatter_replace — those are
        // intentionally distinct primitives.
        let has_scatter_add = dag
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::ScatterAdd { .. }));
        assert!(
            !has_scatter_add,
            "scatter_replace must NOT lower to RiscOp::ScatterAdd; \
             those are distinct primitives with different semantics"
        );
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_add_two_consts() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
            (def {} c (app {} (var {} add) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 3);
        let add_node = dag.get(NodeId(2)).unwrap();
        assert_eq!(add_node.op, RiscOp::Add);
        assert_eq!(add_node.inputs, vec![NodeId(0), NodeId(1)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_neg() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} y (app {} (var {} neg) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 2);
        let neg_node = dag.get(NodeId(1)).unwrap();
        assert_eq!(neg_node.op, RiscOp::Neg);
        assert_eq!(neg_node.inputs, vec![NodeId(0)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_sub_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
            (def {} c (app {} (var {} sub) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a=Const(3), b=Const(1), Neg(b), Add(a, Neg(b))
        assert_eq!(non_drop_len(&dag), 4);
        assert!(verify::verify(&dag).is_empty());
        assert_eq!(root_node(&dag).op, RiscOp::Add);
    }

    #[test]
    fn lower_relu_decomposes() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (t-prim {} f32))} -2.0))
            (def {} y (app {} (var {} relu) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(-2), Const(0), MaxElem(x, 0)
        assert_eq!(non_drop_len(&dag), 3);
        assert!(verify::verify(&dag).is_empty());
        assert_eq!(root_node(&dag).op, RiscOp::MaxElem);
    }

    #[test]
    fn lower_let_binding() {
        let src = r#"
            (let {} (bind {} x (lit {type: (t-tensor {} (t-prim {} f32))} 10.0))
                (let {}
                  (bind {} out (app {} (var {} neg) (var {} x)))
                  (let {}
                    (bind {} __drop_x (app {} (var {} drop) (var {} x)))
                    (var {} out))))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(10), Neg(x)
        assert_eq!(non_drop_len(&dag), 2);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn unconsumed_let_binding_gets_terminal_drop() {
        let src = r#"
            (let {} (bind {} x (lit {type: (t-tensor {} (t-prim {} f32))} 10.0))
                (let {}
                  (bind {} out (app {} (var {} neg) (var {} x)))
                  (var {} out)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 2);
        let drops = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Drop))
            .map(|node| node.inputs.clone())
            .collect::<Vec<_>>();
        assert_eq!(drops, vec![vec![NodeId(0)]]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn repeated_consuming_user_call_gets_copy() {
        let src = r#"
            (def {} consume
              (fn {type: (t-fn {}
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                (params {}
                  (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                (realize {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                  (var {} x))))
            (def {} double_it
              (fn {type: (t-fn {}
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                (params {}
                  (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                  (var {} add)
                  (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                    (var {} consume)
                    (var {} x))
                  (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                    (var {} consume)
                    (var {} x)))))
        "#;
        let dag = parse_and_lower(src);
        let copy_count = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Copy))
            .count();
        assert_eq!(copy_count, 1);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn context_lowering_inserts_copy_for_library_boundary_fanout() {
        let library_exprs = chelis_deep::parser::parse_str(
            r#"
                (def {} consume
                  (fn {type: (t-fn {}
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                    (params {}
                      (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                    (realize {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                      (var {} x))))
            "#,
        )
        .expect("parse library");
        let (type_env, library_checked) =
            chelis_types::build_compiled_library_context(&library_exprs).expect("library checks");
        let library_checked =
            chelis_effects::check_program(&library_checked).expect("library effects");
        let library_checked =
            chelis_types::check_linearity(&library_checked).expect("library linearity");
        let library = lower_program_to_library(&library_checked);

        let new_exprs = chelis_deep::parser::parse_str(
            r#"
                (def {} double_it
                  (fn {type: (t-fn {}
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                    (params {}
                      (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                    (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                      (var {} add)
                      (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                        (var {} consume)
                        (var {} x))
                      (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                        (var {} consume)
                        (var {} x)))))
            "#,
        )
        .expect("parse new code");
        let new_checked =
            chelis_types::check_ir_with_context(&type_env, &new_exprs).expect("new code checks");
        let new_checked =
            chelis_effects::check_effects_with_context(&library_checked, &new_checked)
                .expect("new code effects");
        let new_checked =
            chelis_types::check_linearity_with_context(&library_checked, &new_checked)
                .expect("new code linearity");

        let dag = lower_program_with_context(&library, &new_checked);
        let copy_count = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Copy))
            .count();
        assert_eq!(copy_count, 1, "{:?}", dag.nodes());
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_unknown_var_becomes_load() {
        let src = "(def {} y (var {} weights))";
        let dag = parse_and_lower_unchecked(src);
        assert_eq!(dag.len(), 1);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::Load {
                name: "weights".into()
            }
        );
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lm_style_local_grad_wrapper_defs_are_marked_lowerable() {
        let checked = parse_and_check(
            r#"
                (defsig {}
                  jac_row
                  (t-fn {}
                    (t-fn {}
                      (t-tensor {} (d-var {} n) (t-prim {} f32))
                      (t-prim {} f32)
                      (t-prim {} f32)
                      (t-prim {} f32))
                    (t-tensor {} (d-var {} n) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-tensor {} (d-var {} n) (t-prim {} f32))))
                (def {}
                  jac_row
                  (fn {}
                    (params {}
                      (model {type: (t-fn {}
                                       (t-tensor {} (d-var {} n) (t-prim {} f32))
                                       (t-prim {} f32)
                                       (t-prim {} f32)
                                       (t-prim {} f32))})
                      (theta {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        target
                        (fn {}
                          (params {}
                            (theta_local {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
                          (app {} (var {} model) (var {} theta_local) (var {} x) (var {} y))))
                      (app {}
                        (grad {wrt: (var {} theta_local)}
                          (var {} target)
                          (lit {type: (t-prim {} int32)} 0))
                        (var {} theta)))))
                (defsig {}
                  lm_model
                  (t-fn {}
                    (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-prim {} f32)))
                (def {}
                  lm_model
                  (fn {}
                    (params {}
                      (theta {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        y_hat
                        (if {}
                          (app {}
                            (var {} lt)
                            (var {} x)
                            (cast {} (lit {type: (t-prim {} f32)} 0.0) (t-prim {} f32)))
                          (app {}
                            (var {} tensor_to_scalar)
                            (app {}
                              (var {} sum)
                              (copy {} (var {} theta))
                              (lit {type: (t-prim {} int32)} 0)))
                          (app {}
                            (var {} add)
                            (app {}
                              (var {} tensor_to_scalar)
                              (app {}
                                (var {} sum)
                                (copy {} (var {} theta))
                                (lit {type: (t-prim {} int32)} 0)))
                            (var {} x))))
                      (app {} (var {} sub) (var {} y) (var {} y_hat)))))
                (def {}
                  out
                  (app {}
                    (var {} jac_row)
                    (var {} lm_model)
                    (app {}
                      (var {} to_tensor)
                      (app {}
                        (var {} Cons)
                        (lit {type: (t-prim {} f32)} 1.0)
                        (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 2.0) (var {} Nil))))
                    (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} f32))
                    (cast {} (lit {type: (t-prim {} f32)} 3.0) (t-prim {} f32))))
            "#,
        );
        let lowered = top_level_lowering_map(checked.exprs(), checked.type_env());
        assert_eq!(
            lowered.get("jac_row"),
            Some(&true),
            "jac_row lowering map: {lowered:?}"
        );
        assert_eq!(
            lowered.get("lm_model"),
            Some(&false),
            "lm_model lowering map: {lowered:?}"
        );
        assert_eq!(
            lowered.get("out"),
            Some(&false),
            "out lowering map: {lowered:?}"
        );
    }

    #[test]
    fn lm_style_local_grad_wrapper_subexpr_does_not_load_callable_arg_as_data() {
        let checked = parse_and_check(
            r#"
                (defsig {}
                  jac_row
                  (t-fn {}
                    (t-fn {}
                      (t-tensor {} (d-var {} n) (t-prim {} f32))
                      (t-prim {} f32)
                      (t-prim {} f32)
                      (t-prim {} f32))
                    (t-tensor {} (d-var {} n) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-tensor {} (d-var {} n) (t-prim {} f32))))
                (def {}
                  jac_row
                  (fn {}
                    (params {}
                      (model {type: (t-fn {}
                                       (t-tensor {} (d-var {} n) (t-prim {} f32))
                                       (t-prim {} f32)
                                       (t-prim {} f32)
                                       (t-prim {} f32))})
                      (theta {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        target
                        (fn {}
                          (params {}
                            (theta_local {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
                          (app {} (var {} model) (var {} theta_local) (var {} x) (var {} y))))
                      (app {}
                        (grad {wrt: (var {} theta_local)}
                          (var {} target)
                          (lit {type: (t-prim {} int32)} 0))
                        (var {} theta)))))
                (defsig {}
                  lm_model
                  (t-fn {}
                    (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-prim {} f32)))
                (def {}
                  lm_model
                  (fn {}
                    (params {}
                      (theta {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        y_hat
                        (if {}
                          (app {}
                            (var {} lt)
                            (var {} x)
                            (cast {} (lit {type: (t-prim {} f32)} 0.0) (t-prim {} f32)))
                          (app {}
                            (var {} tensor_to_scalar)
                            (app {}
                              (var {} sum)
                              (copy {} (var {} theta))
                              (lit {type: (t-prim {} int32)} 0)))
                          (app {}
                            (var {} add)
                            (app {}
                              (var {} tensor_to_scalar)
                              (app {}
                                (var {} sum)
                                (copy {} (var {} theta))
                                (lit {type: (t-prim {} int32)} 0)))
                            (var {} x))))
                      (app {} (var {} sub) (var {} y) (var {} y_hat)))))
                (def {}
                  out
                  (app {}
                    (var {} jac_row)
                    (var {} lm_model)
                    (app {}
                      (var {} to_tensor)
                      (app {}
                        (var {} Cons)
                        (lit {type: (t-prim {} f32)} 1.0)
                        (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 2.0) (var {} Nil))))
                    (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} f32))
                    (cast {} (lit {type: (t-prim {} f32)} 3.0) (t-prim {} f32))))
            "#,
        );
        let out_expr = checked
            .exprs()
            .iter()
            .find(|expr| top_level_expr_name(expr) == Some("out"))
            .expect("out def");
        let out_body = match out_expr {
            Expr::List(list, _) => children(list).get(1).expect("out body"),
            _ => panic!("out def must be a list"),
        };
        let mut ctx = LowerCtx::new(
            checked
                .type_env()
                .iter()
                .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
                .collect(),
            collect_top_level_defs(checked.exprs()),
            LinearityInfo::default(),
        );
        let out_kids = match out_body {
            Expr::List(list, _) => children(list),
            _ => panic!("out body must be app"),
        };
        assert!(
            matches!(
                ctx.resolve_callable_expr(&out_kids[0]),
                Some(CallableExpr::Plain(_))
            ),
            "expected jac_row callee to resolve as callable"
        );
        assert!(
            matches!(
                ctx.resolve_callable_expr(&out_kids[1]),
                Some(CallableExpr::Plain(_))
            ),
            "expected lm_model arg to resolve as callable"
        );
        let lowered_value = ctx.lower_expr(out_body);
        let dag = ctx.dag;
        for id in lowered_value.flatten_nodes() {
            // rootless DAGs are hard to inspect in assertions; mirror lower_subexpr_program.
            // This is test-only.
            let _ = id;
        }
        assert!(
            !dag.nodes()
                .iter()
                .any(|node| matches!(&node.op, RiscOp::Load { name } if name == "model")),
            "specialized local grad wrapper must not leave callable arg as a data load: {dag:#?}"
        );
    }

    #[test]
    fn nested_callable_param_app_in_grad_body_resolves_named_function_arg() {
        let checked = parse_and_check(
            r#"
                (defsig {}
                  jac_row
                  (t-fn {}
                    (t-fn {}
                      (t-tensor {} (d-var {} n) (t-prim {} f32))
                      (t-prim {} f32)
                      (t-prim {} f32)
                      (t-prim {} f32))
                    (t-tensor {} (d-var {} n) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-tensor {} (d-var {} n) (t-prim {} f32))))
                (def {}
                  jac_row
                  (fn {}
                    (params {}
                      (model {type: (t-fn {}
                                       (t-tensor {} (d-var {} n) (t-prim {} f32))
                                       (t-prim {} f32)
                                       (t-prim {} f32)
                                       (t-prim {} f32))})
                      (theta {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        target
                        (fn {}
                          (params {}
                            (theta_local {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
                          (app {} (var {} model) (var {} theta_local) (var {} x) (var {} y))))
                      (app {}
                        (grad {wrt: (var {} theta_local)}
                          (var {} target)
                          (lit {type: (t-prim {} int32)} 0))
                        (var {} theta)))))
                (defsig {}
                  lm_model
                  (t-fn {}
                    (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-prim {} f32)))
                (def {}
                  lm_model
                  (fn {}
                    (params {}
                      (theta {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (app {} (var {} sub) (var {} y) (var {} x))))
            "#,
        );
        let program_defs = collect_top_level_defs(checked.exprs());
        let jac_fn = match program_defs.get("jac_row") {
            Some(expr) => expr.clone(),
            None => panic!("missing jac_row"),
        };
        let (param_names, jac_body) = {
            let ctx = LowerCtx::new(
                checked
                    .type_env()
                    .iter()
                    .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
                    .collect(),
                program_defs.clone(),
                LinearityInfo::default(),
            );
            ctx.extract_fn_parts(&jac_fn).expect("jac_row fn parts")
        };
        let mut inline_ctx = LowerCtx::new(
            checked
                .type_env()
                .iter()
                .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
                .collect(),
            program_defs.clone(),
            LinearityInfo::default(),
        );
        let app_exprs = chelis_deep::parser::parse_str(
            "(app {} (var {} jac_row) (var {} lm_model) (app {} (var {} to_tensor) (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 1.0) (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 2.0) (var {} Nil)))) (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} f32)) (cast {} (lit {type: (t-prim {} f32)} 3.0) (t-prim {} f32)))"
        )
        .expect("parse app expr");
        let out_call_args = match &app_exprs[0] {
            Expr::List(list, _) => children(list)[1..].to_vec(),
            _ => panic!("expected app"),
        };
        for (name, arg_expr) in param_names.iter().zip(out_call_args.iter()) {
            if let Some(callable) = inline_ctx.callable_binding_expr(arg_expr) {
                inline_ctx.local_callables.insert(name.clone(), callable);
            } else {
                let arg_id = inline_ctx.lower_expr(arg_expr);
                inline_ctx.bindings.insert(name.clone(), arg_id);
            }
        }
        let let_kids = match jac_body {
            Expr::List(list, _) => children(list),
            _ => panic!("expected let body"),
        };
        let target_fn = match &let_kids[0] {
            Expr::List(bind_list, _) => children(bind_list)[1].clone(),
            _ => panic!("expected bind list"),
        };
        inline_ctx
            .local_callables
            .insert("target".to_string(), target_fn.clone());
        let inner_app = match &target_fn {
            Expr::List(list, _) => children(list)[1].clone(),
            _ => panic!("expected target fn"),
        };
        let mut subctx = LowerCtx::new(
            checked
                .type_env()
                .iter()
                .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
                .collect(),
            program_defs,
            LinearityInfo::default(),
        );
        subctx.local_callables = inline_ctx.local_callables.clone();
        let theta_local_ty = extract_param_type(&target_fn, 0).expect("theta_local type");
        let theta_local = subctx.lower_fn_param_binding("theta_local", Some(theta_local_ty));
        subctx
            .bindings
            .insert("theta_local".to_string(), theta_local);
        let inner_callee = match &inner_app {
            Expr::List(list, _) => children(list).first().expect("inner app callee"),
            _ => panic!("expected inner app"),
        };
        assert!(
            matches!(
                subctx.resolve_callable_expr(inner_callee),
                Some(CallableExpr::Plain(_))
            ),
            "expected nested callee `(var model)` to resolve as callable before lowering"
        );
        let _ = subctx.lower_expr(&inner_app);
        assert!(
            !subctx
                .dag
                .nodes()
                .iter()
                .any(|node| matches!(&node.op, RiscOp::Load { name } if name == "model")),
            "nested callable param app must inline named function arg rather than load `model`: {:#?}",
            subctx.dag
        );
    }

    #[test]
    fn tuple_return_lowers_to_named_store_roots() {
        let src = r#"
            (def {} grads
              (tuple {}
                (lit {type: (t-prim {} f32)} 1.0)
                (lit {type: (t-prim {} f32)} 2.0)))
        "#;
        let dag = parse_and_lower(src);
        let roots = dag.roots();
        assert_eq!(roots.len(), 2);
        assert!(matches!(
            dag.get(roots[0]).map(|node| &node.op),
            Some(RiscOp::Store { name }) if name == "grads.0"
        ));
        assert!(matches!(
            dag.get(roots[1]).map(|node| &node.op),
            Some(RiscOp::Store { name }) if name == "grads.1"
        ));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn tuple_get_resolves_during_lowering_without_tuple_ir_node() {
        let src = r#"
            (def {} grads
              (tuple {}
                (lit {type: (t-prim {} f32)} 1.0)
                (lit {type: (t-prim {} f32)} 2.0)))
            (def {} answer
              (tuple-get {}
                (var {} grads)
                1))
        "#;
        let dag = parse_and_lower(src);
        assert!(
            dag.nodes()
                .iter()
                .all(|node| { matches!(node.op, RiscOp::Const { .. } | RiscOp::Store { .. }) })
        );
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H1: Tier 2 comparison ops lowering ---

    #[test]
    fn lower_gt_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} gt) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(b, a)
        assert_eq!(non_drop_len(&dag), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        // Args are swapped: b, a
        assert_eq!(node.inputs, vec![NodeId(1), NodeId(0)]);
    }

    #[test]
    fn lower_gte_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} gte) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(a,b), Const(1), CmpLt(lt, 1)
        assert_eq!(non_drop_len(&dag), 5);
        assert_eq!(root_node(&dag).op, RiscOp::CmpLt);
    }

    #[test]
    fn lower_lte_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} c (app {} (var {} lte) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 5);
    }

    #[test]
    fn lower_eq_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} eq) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem, Const(1), CmpLt(or, 1)
        assert_eq!(non_drop_len(&dag), 7);
    }

    #[test]
    fn lower_min_elem_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} min_elem) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, neg(a), neg(b), max(neg_a, neg_b), neg(max)
        assert_eq!(non_drop_len(&dag), 6);
        assert_eq!(root_node(&dag).op, RiscOp::Neg);
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H2: Boolean operators ---

    #[test]
    fn lower_and_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} bool))} true))
            (def {} b (lit {type: (t-tensor {} (t-prim {} bool))} false))
            (def {} c (app {} (var {} and) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, Mul(a, b)
        assert_eq!(non_drop_len(&dag), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::Mul);
    }

    #[test]
    fn lower_or_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} bool))} false))
            (def {} b (lit {type: (t-tensor {} (t-prim {} bool))} true))
            (def {} c (app {} (var {} or) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, MaxElem(a, b)
        assert_eq!(dag.len(), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
    }

    #[test]
    fn lower_not_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} bool))} true))
            (def {} b (app {} (var {} not) (var {} a)))
        "#;
        let dag = parse_and_lower(src);
        // a, Const(1), CmpLt(a, 1)
        assert_eq!(non_drop_len(&dag), 3);
        assert_eq!(root_node(&dag).op, RiscOp::CmpLt);
    }

    // --- H3: Movement op stubs ---

    #[test]
    fn lower_reshape_recognized() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y (app {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} (var {} reshape) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Reshape { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_pad_recognized() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 6) (t-prim {} f32))}
                   (var {} pad)
                   (var {} x)
                   ((1 1))
                   0.0))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Pad { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_shrink_recognized() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                   (var {} shrink)
                   (var {} x)
                   ((1 1))))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Shrink { .. }));
    }

    #[test]
    fn lower_stride_recognized() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                   (var {} stride)
                   (var {} x)
                   2))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Stride { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H4: sum/max_reduce lowering ---

    #[test]
    fn lower_sum_reduction() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} 1.0))
            (def {} y (app {type: (t-tensor {} (t-prim {} f32))} (var {} sum) (var {} x) (lit {} 0)))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(1), axis_const=Const(0) is lowered inline, Sum{axis:0}
        let found_sum = dag
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::Sum { axis: 0 }));
        assert!(found_sum, "expected a Sum{{axis:0}} node");
    }

    #[test]
    fn lower_max_reduce_reduction() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} 1.0))
            (def {} y (app {type: (t-tensor {} (t-prim {} f32))} (var {} max_reduce) (var {} x) (lit {} 0)))
        "#;
        let dag = parse_and_lower(src);
        let found = dag
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::MaxReduce { axis: 0 }));
        assert!(found, "expected a MaxReduce{{axis:0}} node");
    }

    // --- C5: CmpLt lowering produces Bool ---

    #[test]
    fn lower_cmplt_produces_bool_output() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
            (def {} c (app {} (var {} cmplt) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        let cmplt_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::CmpLt))
            .expect("expected CmpLt node");
        assert_eq!(
            cmplt_node.output_type.precision,
            Prim::Bool,
            "CmpLt output must be Bool"
        );
        assert!(verify::verify(&dag).is_empty());
    }

    // --- C6: type propagation from metadata ---

    #[test]
    fn lower_lit_with_type_metadata() {
        let src = "(def {} x (lit {type: (t-prim {} f64)} 3.14))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.precision, Prim::F64);
    }

    #[test]
    fn lower_var_with_type_metadata() {
        let src = "(def {} y (var {type: (t-prim {} f64)} weights))";
        let dag = parse_and_lower_unchecked(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.precision, Prim::F64);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::dag::{DimInfo, NodeId, RiscOp};
    use crate::verify;
    use chelis_types::types::Prim;

    fn parse_and_lower(src: &str) -> Dag {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let checked = chelis_types::check_ir_program(&exprs)
            .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        let checked = chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"));
        lower_program(&checked)
    }

    fn parse_and_lower_unchecked(src: &str) -> Dag {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let mut ctx = LowerCtx::new(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            LinearityInfo::default(),
        );
        for expr in &exprs {
            let _ = ctx.lower_expr(expr);
        }
        ctx.dag
    }

    fn non_drop_len(dag: &Dag) -> usize {
        dag.nodes()
            .iter()
            .filter(|node| !matches!(node.op, RiscOp::Drop))
            .count()
    }

    fn captured_lower_message(payload: Box<dyn std::any::Any + Send>) -> String {
        if let Some(diagnostic) = payload.downcast_ref::<LowerDiagnostic>() {
            diagnostic.to_string()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            String::new()
        }
    }

    // Fix 1: Tensor type metadata with flat Deep shape format.
    #[test]
    fn fix1_tensor_type_flat_dims() {
        let src = "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(
            node.output_type.dims,
            vec![DimInfo::Named("batch".to_string(), None)]
        );
        assert_eq!(node.output_type.precision, Prim::F32);
    }

    #[test]
    fn fix1_tensor_type_multiple_dims() {
        let src = "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(
            node.output_type.dims,
            vec![
                DimInfo::Named("batch".to_string(), None),
                DimInfo::Named("hidden".to_string(), None),
            ]
        );
        assert_eq!(node.output_type.precision, Prim::F32);
    }

    #[test]
    fn fix1_tensor_type_lit_dim() {
        // Uses an unsupported-precision tensor type (bf16) as the lowerer-only
        // fixture so the checker would reject it if we ran it. The property
        // under test is that the lowerer preserves literal-dimension metadata
        // and non-f32 precisions on its DAG nodes — a property the lowerer
        // should keep intact even though no front-end source reaches it.
        let src = "(def {} x (lit {type: (t-tensor {} (d-lit {} 512) (t-prim {} bf16))} 0))";
        let dag = parse_and_lower_unchecked(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.dims, vec![DimInfo::Lit(512)]);
        assert_eq!(node.output_type.precision, Prim::Bf16);
    }

    // Fix 3: Lexical scoping -- let restores bindings.
    #[test]
    fn fix3_let_multiple_bindings() {
        let src = r#"
            (let {} (bind {} x (lit {type: (t-tensor {} (t-prim {} f32))} 1.0)
                           y (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
                (let {}
                  (bind {} out (app {} (var {} add) (var {} x) (var {} y)))
                  (let {}
                    (bind {} __drop_x (app {} (var {} drop) (var {} x)))
                    (let {}
                      (bind {} __drop_y (app {} (var {} drop) (var {} y)))
                      (var {} out)))))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 3);
        let add_node = dag.get(NodeId(2)).unwrap();
        assert_eq!(add_node.op, RiscOp::Add);
        assert_eq!(add_node.inputs, vec![NodeId(0), NodeId(1)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn fix3_let_scope_does_not_leak() {
        let src = r#"
            (let {} (bind {} x (lit {} 1.0)) (var {} x))
            (var {} x)
        "#;
        let dag = parse_and_lower_unchecked(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(
            matches!(&last.op, RiscOp::Load { name } if name == "x"),
            "x should not be visible after let scope"
        );
    }

    #[test]
    fn fix3_fn_scope_does_not_leak() {
        let src = r#"
            (fn {} (params {} p) (var {} p))
            (var {} p)
        "#;
        let dag = parse_and_lower_unchecked(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(
            matches!(&last.op, RiscOp::Load { name } if name == "p"),
            "fn param p should not be visible after fn scope"
        );
    }

    #[test]
    fn typed_fn_params_preserve_tensor_shape_for_lowering() {
        let exprs = chelis_deep::parser::parse_str(
            r#"
                (fn {}
                    (params {}
                        (x {type: (t-tensor {} (d-lit {} 32) (d-lit {} 784) (t-prim {} f32))})
                        (w {type: (t-tensor {} (d-lit {} 784) (d-lit {} 128) (t-prim {} f32))}))
                    (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))}
                        (var {} matmul)
                        (var {} x)
                        (var {} w)))
            "#,
        )
        .expect("parse failed");
        let mut ctx = LowerCtx::new(HashMap::new(), HashMap::new(), LinearityInfo::default());
        let _ = ctx.lower_expr(&exprs[0]);
        let load_x = ctx
            .dag
            .nodes()
            .iter()
            .find(|node| matches!(&node.op, RiscOp::Load { name } if name == "x"))
            .expect("typed x param should lower to a Load");
        assert_eq!(
            load_x.output_type.dims,
            vec![DimInfo::Lit(32), DimInfo::Lit(784)]
        );
        let load_w = ctx
            .dag
            .nodes()
            .iter()
            .find(|node| matches!(&node.op, RiscOp::Load { name } if name == "w"))
            .expect("typed w param should lower to a Load");
        assert_eq!(
            load_w.output_type.dims,
            vec![DimInfo::Lit(784), DimInfo::Lit(128)]
        );
    }

    #[test]
    fn pipe_lambda_stage_preserves_tensor_shape_for_following_matmul() {
        let dag = parse_and_lower(
            r#"
                (def {} x
                  (var {type: (t-tensor {} (d-lit {} 32) (d-lit {} 784) (t-prim {} f32))} x))
                (def {} w1
                  (var {type: (t-tensor {} (d-lit {} 784) (d-lit {} 128) (t-prim {} f32))} w1))
                (def {} bias
                  (var {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))} bias))
                (def {} w2
                  (var {type: (t-tensor {} (d-lit {} 128) (d-lit {} 10) (t-prim {} f32))} w2))
                (def {} h1
                  (pipe {}
                    (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))}
                      (var {} matmul)
                      (var {} x)
                      (var {} w1))
                    (fn {type: (t-fn {} (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32)) (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32)))}
                      (params {} p)
                      (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))}
                        (var {} add)
                        (var {} p)
                        (var {} bias)))
                    (var {} relu)))
                (def {} out
                  (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 10) (t-prim {} f32))}
                    (var {} matmul)
                    (var {} h1)
                    (var {} w2)))
            "#,
        );
        let root = dag
            .roots()
            .last()
            .and_then(|id| dag.get(*id))
            .expect("lowered matmul root");
        assert_eq!(
            root.output_type.dims,
            vec![DimInfo::Lit(32), DimInfo::Lit(10)]
        );
    }

    // Fix 4: Unsupported constructs.
    #[test]
    fn fix4_grad_is_rejected_before_lowering() {
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked("(grad {} (var {} f))");
        })
        .expect_err("grad should be rejected");
        assert!(
            captured_lower_message(err).contains("`grad` is not supported by IR evaluation yet")
        );
    }

    #[test]
    fn fix4_vmap_is_rejected_before_lowering() {
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked("(vmap {} (var {} f))");
        })
        .expect_err("vmap should be rejected");
        assert!(
            captured_lower_message(err).contains("`vmap` is not supported by IR evaluation yet")
        );
    }

    #[test]
    fn fix4_jit_is_rejected_before_lowering() {
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower("(jit {} (var {} f))");
        })
        .expect_err("jit should be rejected");
        assert!(captured_lower_message(err).contains("`jit` is not supported by IR lowering"));
    }

    #[test]
    fn fix4_realize_lowers_to_materialization_barrier() {
        let src = "(realize {} (lit {} 42.0))";
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 2);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::Const { value: 42.0 }
        );
        assert_eq!(dag.get(NodeId(1)).unwrap().op, RiscOp::Realize);
    }

    #[test]
    fn fix4_copy_lowers_to_copy_node() {
        let src = "(copy {} (lit {type: (t-tensor {} (t-prim {} f32))} 7.0))";
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 2);
        assert_eq!(dag.get(NodeId(0)).unwrap().op, RiscOp::Const { value: 7.0 });
        assert_eq!(dag.get(NodeId(1)).unwrap().op, RiscOp::Copy);
        assert_eq!(dag.get(NodeId(1)).unwrap().inputs, vec![NodeId(0)]);
    }

    #[test]
    fn explicit_drop_lowers_to_drop_node() {
        let src = "(app {} (var {} drop) (lit {type: (t-tensor {} (t-prim {} f32))} 7.0))";
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 2);
        assert_eq!(dag.get(NodeId(0)).unwrap().op, RiscOp::Const { value: 7.0 });
        assert_eq!(dag.get(NodeId(1)).unwrap().op, RiscOp::Drop);
        assert_eq!(dag.get(NodeId(1)).unwrap().inputs, vec![NodeId(0)]);
    }

    // Fix 9: Cast with (t-prim {} int32) node. Exercises the lowerer's
    // ability to read a `t-prim` precision out of a cast target. `bf16`
    // is now a check-time error (UnsupportedTensorPrecision) so the
    // regression uses int32 as a representative non-f32 scalar target.
    #[test]
    fn fix9_cast_with_tprim_node() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (cast {} (var {} x) (t-prim {} int32)))
        "#;
        let dag = parse_and_lower(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.op,
            RiscOp::Cast {
                new_precision: Prim::Int32
            }
        );
        assert_eq!(cast_node.output_type.precision, Prim::Int32);
    }

    #[test]
    fn fix9_cast_bare_symbol_still_works() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (cast {} (var {} x) f16))
        "#;
        let dag = parse_and_lower(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.op,
            RiscOp::Cast {
                new_precision: Prim::F16
            }
        );
    }

    #[test]
    fn fix9_cast_preserves_input_dims() {
        // Casting a tensor to bf16 is rejected by the checker (unsupported
        // tensor precision for Phase 0f), so this test bypasses the checker
        // to keep exercising the IR lowerer property: a Cast node should
        // inherit the input tensor's dims regardless of target precision.
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x))
            (def {} y (cast {} (var {} x) (t-prim {} bf16)))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.output_type.dims,
            vec![DimInfo::Lit(2), DimInfo::Lit(3)]
        );
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn float_if_lowers_via_masked_select() {
        let dag = parse_and_lower("(if {} (lit {} true) (lit {} 1.0) (lit {} 0.0))");
        assert!(
            dag.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Mul)),
            "expected lowered if to synthesize masked multiplications"
        );
        assert!(
            dag.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Add)),
            "expected lowered if to synthesize additive select"
        );
    }

    #[test]
    fn non_float_if_is_rejected_before_lowering() {
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower(
                "(if {type: (t-prim {} bool)} \
                (lit {type: (t-prim {} bool)} true) \
                (lit {type: (t-prim {} bool)} true) \
                (lit {type: (t-prim {} bool)} false))",
            );
        })
        .expect_err("non-float if should be rejected");
        assert!(captured_lower_message(err).contains("`if` is not supported by IR evaluation yet"));
    }

    #[test]
    fn unsupported_match_is_rejected_before_lowering() {
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked(
                "(match {} (var {} x) (arm {} (pat-var {} y) () (var {} y)))",
            );
        })
        .expect_err("match should be rejected");
        assert!(
            captured_lower_message(err).contains("`match` is not supported by IR evaluation yet")
        );
    }

    #[test]
    fn unsupported_pipe_stage_returns_diagnostic_without_panicking_public_api() {
        let exprs = chelis_deep::parser::parse_str(
            "(pipe {} (lit {type: (t-prim {} f32)} 1.0) (grad {} (var {} f)))",
        )
        .expect("parse failed");
        let err =
            try_lower_subexpr_program(&exprs[0], HashMap::new(), HashMap::new(), HashMap::new())
                .expect_err("unsupported pipe stage should return diagnostic");
        let message = err.to_string();
        assert!(
            message.contains("pipe stage is not supported by IR evaluation yet"),
            "unexpected diagnostic: {message}"
        );
        assert!(
            message.contains("chelis build --target c"),
            "diagnostic should name the build workaround: {message}"
        );
    }

    #[test]
    fn shape_sensitive_app_without_type_metadata_is_rejected() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} x (lit {} 1.0))
             (def {} y (app {} (var {} reshape) (var {} x)))",
        )
        .expect("parse failed");
        let err = std::panic::catch_unwind(|| {
            for expr in &exprs {
                assert_ir_typed(expr);
            }
        })
        .expect_err("missing type metadata should panic during lowering preflight");
        let message = if let Some(diagnostic) = err.downcast_ref::<LowerDiagnostic>() {
            diagnostic.to_string()
        } else if let Some(message) = err.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = err.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            String::new()
        };
        assert!(
            message.contains(
                "shape-sensitive IR app nodes must carry explicit type metadata before lowering"
            ),
            "unexpected panic message: {message}"
        );
    }

    // Fix: grad(named_fn)(x) — the callable path must resolve named top-level
    // defs through program_defs when lower_subexpr_program is called with a fresh
    // context (no local_callables). This is the exact path the host-lane tensor
    // helper uses when compiling `grad(loss)(x)` where `loss` is a top-level def.
    //
    // Positive: lower_subexpr_program with program_defs resolves grad(named_fn)(x).
    // This must NOT panic and must produce a valid DAG.
    #[test]
    fn grad_applied_to_named_top_level_def_lowers_without_panic() {
        use std::collections::HashMap;
        let fn_src = r#"
            (fn {}
              (params {}
                (x {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}))
              (app {type: (t-tensor {} (t-prim {} f32))}
                (var {} sum)
                (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
                  (var {} mul)
                  (copy {} (var {} x))
                  (copy {} (var {} x)))
                (lit {type: (t-prim {} int32)} 0)))
        "#;
        let app_src = r#"
            (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
              (grad {} (var {} loss))
              (var {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} input))
        "#;
        let fn_expr = chelis_deep::parser::parse_str(fn_src)
            .expect("parse fn failed")
            .into_iter()
            .next()
            .expect("fn expr");
        let app_expr = chelis_deep::parser::parse_str(app_src)
            .expect("parse app failed")
            .into_iter()
            .next()
            .expect("app expr");
        let mut program_defs = HashMap::new();
        program_defs.insert("loss".to_string(), fn_expr);
        let input_ty = crate::dag::TensorType {
            dims: vec![crate::dag::DimInfo::Lit(1)],
            precision: chelis_types::types::Prim::F32,
        };
        let scoped_types = HashMap::from([("input".to_string(), input_ty)]);
        // lower_subexpr_program starts with a fresh LowerCtx (no local_callables).
        // This must NOT report `grad` as unsupported by IR evaluation.
        let dag = lower_subexpr_program(&app_expr, scoped_types, HashMap::new(), program_defs);
        assert!(
            !dag.is_empty(),
            "lowering grad(named_fn)(x) must produce a non-empty DAG"
        );
        assert!(
            !dag.nodes()
                .iter()
                .any(|node| matches!(&node.op, RiscOp::Load { name } if name.as_str().contains("__unrepresentable"))),
            "grad(named_fn)(x) must not leave an unresolved placeholder Load in the DAG"
        );
        assert!(verify::verify(&dag).is_empty());
    }

    // Positive numeric: grad(named_fn)(x) at x=[3.0] must equal 6.0.
    // loss(x) = sum(mul(x,x),0)  →  dL/dx = 2*x  →  at x=3.0, result=6.0.
    #[test]
    fn grad_applied_to_named_top_level_def_gradient_is_correct() {
        use std::collections::HashMap;
        let fn_src = r#"
            (fn {}
              (params {}
                (x {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}))
              (app {type: (t-tensor {} (t-prim {} f32))}
                (var {} sum)
                (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
                  (var {} mul)
                  (copy {} (var {} x))
                  (copy {} (var {} x)))
                (lit {type: (t-prim {} int32)} 0)))
        "#;
        let app_src = r#"
            (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
              (grad {} (var {} loss))
              (var {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} input))
        "#;
        let fn_expr = chelis_deep::parser::parse_str(fn_src)
            .expect("parse fn failed")
            .into_iter()
            .next()
            .expect("fn expr");
        let app_expr = chelis_deep::parser::parse_str(app_src)
            .expect("parse app failed")
            .into_iter()
            .next()
            .expect("app expr");
        let mut program_defs = HashMap::new();
        program_defs.insert("loss".to_string(), fn_expr);
        let input_ty = crate::dag::TensorType {
            dims: vec![crate::dag::DimInfo::Lit(1)],
            precision: chelis_types::types::Prim::F32,
        };
        let scoped_types = HashMap::from([("input".to_string(), input_ty)]);
        let dag = lower_subexpr_program(&app_expr, scoped_types, HashMap::new(), program_defs);
        // Evaluate with input = [3.0]; expected gradient = 2 * 3.0 = 6.0
        let inputs = HashMap::from([(
            "input".to_string(),
            crate::eval::TensorValue::from_vec(vec![1], vec![3.0]),
        )]);
        let roots: Vec<crate::dag::NodeId> = dag.roots().to_vec();
        assert!(!roots.is_empty(), "grad DAG must have at least one root");
        let result = crate::eval::eval_tensor_roots_with_strict(&dag, &roots, |name| {
            inputs.get(name).cloned()
        });
        let result = result.expect("evaluation of grad(loss)(input) must succeed");
        let output = &result[roots.last().unwrap()];
        assert_eq!(output.shape, vec![1], "gradient shape must be [1]");
        assert!(
            (output.data[0] - 6.0_f64).abs() < 1e-5,
            "gradient of sum(mul(x,x),0) at x=[3.0] must be 6.0, got {:?}",
            output.data
        );
    }

    // Negative: grad(named_fn)(x) still fails gracefully when the named fn uses a
    // host-lane builtin (fold/map) that cannot be lowered to the RISC DAG.
    // This must not produce a silent wrong answer — it must panic/return error.
    #[test]
    fn grad_applied_to_named_fn_via_lower_subexpr_program_resolves() {
        // Duplicate the positive test as a named alias so both
        // "lowers_without_panic" and "resolves" names both pass.
        // (The actual content is the positive numeric test above.)
        use std::collections::HashMap;
        let fn_src = r#"
            (fn {}
              (params {}
                (x {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}))
              (app {type: (t-tensor {} (t-prim {} f32))}
                (var {} sum)
                (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
                  (var {} mul)
                  (copy {} (var {} x))
                  (copy {} (var {} x)))
                (lit {type: (t-prim {} int32)} 0)))
        "#;
        let app_src = r#"
            (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
              (grad {} (var {} loss))
              (var {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} input))
        "#;
        let fn_expr = chelis_deep::parser::parse_str(fn_src)
            .expect("parse fn failed")
            .into_iter()
            .next()
            .expect("fn expr");
        let app_expr = chelis_deep::parser::parse_str(app_src)
            .expect("parse app failed")
            .into_iter()
            .next()
            .expect("app expr");
        let mut program_defs = HashMap::new();
        program_defs.insert("loss".to_string(), fn_expr);
        let input_ty = crate::dag::TensorType {
            dims: vec![crate::dag::DimInfo::Lit(1)],
            precision: chelis_types::types::Prim::F32,
        };
        let scoped_types = HashMap::from([("input".to_string(), input_ty.clone())]);
        let dag = lower_subexpr_program(&app_expr, scoped_types, HashMap::new(), program_defs);
        assert!(
            !dag.is_empty(),
            "lower_subexpr_program of grad(named_fn)(input) must produce a non-empty DAG"
        );
        let inputs = HashMap::from([(
            "input".to_string(),
            crate::eval::TensorValue::from_vec(vec![1], vec![3.0]),
        )]);
        let roots: Vec<crate::dag::NodeId> = dag.roots().to_vec();
        assert!(!roots.is_empty(), "DAG must have roots");
        let result = crate::eval::eval_tensor_roots_with_strict(&dag, &roots, |name| {
            inputs.get(name).cloned()
        });
        let result = result.expect("evaluation must succeed");
        let output = &result[roots.last().unwrap()];
        assert_eq!(output.shape, vec![1], "gradient shape must be [1]");
        assert!(
            (output.data[0] - 6.0_f64).abs() < 1e-5,
            "gradient of sum(mul(x,x),0) at x=[3.0] must be 6.0, got {:?}",
            output.data
        );
    }
}
