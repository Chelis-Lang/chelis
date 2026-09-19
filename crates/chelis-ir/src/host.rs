mod signature_entry;
pub mod staged;
pub use signature_entry::SignatureEntryPlan;

use chelis_deep::{DeepTag, ExprCarrier};
use chelis_unord::{UnordMap, UnordSet};
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use chelis_deep::ast::{Atom, Expr, List, Metadata};
use chelis_deep::decode_effect_kind;
use chelis_types::adt::{AdtDef, AdtRegistry, TypeAliasDef};
use chelis_types::infer::type_to_deep_expr;
use chelis_types::types::{Dim, NominalArg, NominalParamKind, Prim, TensorPrec, Type};
use chelis_types::{BUILTIN_NAMES, CheckedProgram};
use chelis_vocab::EffectKind;

use crate::dag::{DimExpr, DimInfo, RiscOp, TensorType};
use crate::host_type_state::{
    ConcreteHostType, HostInferenceVar, HostPrecisionTerm, HostShapeSlot, HostShapeTerm,
    HostTensorTypeTerm, HostTypeDecodeError, HostTypeTerm, decode_host_type,
};
use crate::lower::top_level_lowering_map;

thread_local! {
    // Tracks top-level callee names currently being inlined by
    // `inline_top_level_host_call`. Prevents infinite specialization for
    // recursive/mutually recursive definitions — the specialized body would
    // re-encounter the same call and inline forever.
    static INLINING_STACK: RefCell<UnordSet<String>> = RefCell::new(UnordSet::new());
    // The per-program memos that used to live here are fields of
    // `HostLoweringSession` (chelis#1835). The push/pop stacks stay: they
    // track where the lowerer currently IS, which is a property of the
    // thread's call stack rather than of the program.
    static TENSOR_HELPER_PREFLIGHT_STACK:
        RefCell<Vec<UnordMap<usize, TensorHelperPreflightFacts>>> = const { RefCell::new(Vec::new()) };
    // chelis#1829: kernel-decision summary probes that missed the memo and ran
    // a full callee lowering. `HostWorkProfile` beside it is `cfg(test)`-only
    // and therefore invisible to downstream crates, so this one is always
    // compiled: it lets a counted receipt in `chelis-compiler-api` bound the
    // interpreter's probe work without a wall clock.
    static HOST_SUMMARY_PROBE_BUILDS: Cell<u64> = const { Cell::new(0) };
    // chelis#1158: bounded memoized monomorphization of recursive generic
    // host calls. Keyed by the callee's canonical checked type application;
    // one specialized definition per key, with in-progress entries visible
    // so a recursive edge lowers to an ordinary call to the (possibly
    // still-lowering) specialized symbol. Lifetime: one `lower_host_program`
    // invocation (cleared at its entry, drained into the emitted program at
    // its exit).
    static MONO_SPECIALIZATIONS: RefCell<MonoSpecializationState> =
        RefCell::new(MonoSpecializationState::default());
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct HostWorkProfile {
    host_expr_visits: usize,
    max_host_expr_depth: usize,
    app_clone_nodes: usize,
    tensor_helper_attempts: usize,
    tensor_helper_successes: usize,
    tensor_helper_fallbacks: usize,
    tensor_helper_input_nodes: usize,
    tensor_helper_fail_guard_rejections: usize,
    tensor_helper_dag_rejections: usize,
    tensor_helper_builtin_load_rejections: usize,
    tensor_helper_preflight_rejections: usize,
    tensor_helper_preflight_nodes: usize,
    tensor_helper_preflight_lookup_hits: usize,
    tensor_helper_preflight_lookup_misses: usize,
    callable_scope_work: usize,
    grad_scan_nodes: usize,
    fail_scan_nodes: usize,
    program_def_collections: usize,
    program_def_clone_nodes: usize,
    type_env_clone_nodes: usize,
    helper_summary_builds: usize,
    /// chelis#2181: nodes visited by `substitute_expr`. One inline of a
    /// callable-parameter callee must not cost 2^(binder count); other
    /// shapes are legitimately superlinear, so this bounds the growth
    /// rate rather than asserting linearity.
    substitution_nodes: usize,
}

#[cfg(test)]
thread_local! {
    static HOST_WORK_PROFILE: RefCell<HostWorkProfile> =
        RefCell::new(HostWorkProfile::default());
    static HOST_EXPR_PROFILE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
fn record_host_work(update: impl FnOnce(&mut HostWorkProfile)) {
    HOST_WORK_PROFILE.with(|profile| update(&mut profile.borrow_mut()));
}

#[cfg(not(test))]
#[inline(always)]
fn record_host_work(_update: impl FnOnce(&mut HostWorkProfile)) {}

struct HostExprProfileGuard;

#[cfg(test)]
fn enter_host_expr_profile() -> HostExprProfileGuard {
    HOST_EXPR_PROFILE_DEPTH.with(|depth| {
        let next = depth.get() + 1;
        depth.set(next);
        record_host_work(|profile| {
            profile.host_expr_visits += 1;
            profile.max_host_expr_depth = profile.max_host_expr_depth.max(next);
        });
    });
    HostExprProfileGuard
}

#[cfg(not(test))]
#[inline(always)]
fn enter_host_expr_profile() -> HostExprProfileGuard {
    HostExprProfileGuard
}

#[cfg(test)]
impl Drop for HostExprProfileGuard {
    fn drop(&mut self) {
        HOST_EXPR_PROFILE_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

#[cfg(not(test))]
impl Drop for HostExprProfileGuard {
    fn drop(&mut self) {}
}

#[cfg(test)]
fn reset_host_work_profile() {
    HOST_WORK_PROFILE.with(|profile| *profile.borrow_mut() = HostWorkProfile::default());
    HOST_EXPR_PROFILE_DEPTH.with(|depth| depth.set(0));
}

#[cfg(test)]
fn take_host_work_profile() -> HostWorkProfile {
    HOST_WORK_PROFILE.with(|profile| std::mem::take(&mut *profile.borrow_mut()))
}

fn deep_metadata_nodes(metadata: &chelis_deep::Metadata) -> usize {
    let mut total = 0;
    metadata.visit_expressions(&mut |value, _| total += deep_expr_nodes(value));
    total
}
fn deep_expr_nodes(expr: &Expr) -> usize {
    1 + match expr {
        Expr::Atom(_, _) => 0,
        Expr::Map(map, _) => deep_metadata_nodes(map),
        Expr::MetaExpr(meta, _) => {
            deep_expr_nodes(&meta.expr) + deep_metadata_nodes(&meta.metadata)
        }
        Expr::List(list, _) => list.elements.iter().map(deep_expr_nodes).sum(),
        Expr::Node(node, _) => {
            deep_metadata_nodes(node.meta())
                + node
                    .children_slice()
                    .iter()
                    .map(deep_expr_nodes)
                    .sum::<usize>()
        }
        Expr::BareList(elements, _) => elements.iter().map(deep_expr_nodes).sum(),
        Expr::UnknownForm(data) => {
            deep_metadata_nodes(&data.meta)
                + data.children.iter().map(deep_expr_nodes).sum::<usize>()
        }
    }
}

#[derive(Default, Clone)]
struct MonoSpecializationState {
    /// Canonical `(def identity, type application)` key -> specialized symbol.
    memo: UnordMap<String, String>,
    /// Minted symbol -> the canonical key it was minted from. A second key
    /// arriving at an existing symbol is a hash collision and fails loudly
    /// (harden-bounded-monomorphization D4).
    symbol_keys: UnordMap<String, String>,
    /// Completed specialized definitions, in completion order.
    functions: Vec<LoweredHostFunction>,
    /// Stack of specializations currently being lowered. A recursive edge
    /// into one of these reuses its symbol instead of expanding again — the
    /// memoization that terminates mutual recursion.
    in_progress: Vec<InProgressMonoSpecialization>,
}

/// RAII guard making speculative lowering side-effect-free on the
/// specialization state (harden-bounded-monomorphization D1): the state is
/// snapshotted at probe entry and restored on drop, so a probe can neither
/// add emitted definitions, reorder them, nor leave a memo entry whose
/// definition was never pushed. Restore (not clear) semantics preserve an
/// outer in-progress frame when a probe runs while a specialization body is
/// itself being lowered.
struct MonoProbeGuard {
    snapshot: MonoSpecializationState,
}

impl MonoProbeGuard {
    fn begin() -> Self {
        MonoProbeGuard {
            snapshot: MONO_SPECIALIZATIONS.with(|state| state.borrow().clone()),
        }
    }
}

impl Drop for MonoProbeGuard {
    fn drop(&mut self) {
        MONO_SPECIALIZATIONS.with(|state| *state.borrow_mut() = std::mem::take(&mut self.snapshot));
    }
}

#[derive(Clone)]
struct InProgressMonoSpecialization {
    def_name: String,
    param_tys: Vec<HostTypeTerm>,
    ret_ty: HostTypeTerm,
}

static NEXT_HOST_INFERENCE_VAR: AtomicU32 = AtomicU32::new(0);

fn fresh_host_inference() -> HostTypeTerm {
    HostTypeTerm::InferenceVariable(HostInferenceVar(
        NEXT_HOST_INFERENCE_VAR.fetch_add(1, Ordering::Relaxed),
    ))
}

fn is_inlining(name: &str) -> bool {
    INLINING_STACK.with(|stack| stack.borrow().contains(name))
}

fn push_inlining(name: &str) -> bool {
    INLINING_STACK.with(|stack| stack.borrow_mut().insert(name.to_string()))
}

fn pop_inlining(name: &str) {
    INLINING_STACK.with(|stack| {
        stack.borrow_mut().remove(name);
    });
}

/// The per-program facts host lowering derives once and reads many times.
///
/// Each field is its own `RefCell` rather than one cell over the struct,
/// because a miss on one memo re-enters another: computing a definition's
/// helper-summary rejection runs a probe lowering, and that lowering asks for
/// the program's definitions, its call graph and other definitions' summaries.
/// One cell would make that safe only by the discipline of never holding a
/// borrow across a computation; seven make it safe structurally.
#[derive(Default)]
struct DefLaneFacts {
    /// Per def: does the checker-recorded authored signature carry a stored
    /// type variable?
    type_polymorphic: RefCell<UnordMap<String, bool>>,
    /// Per def: does a probe lowering of its body register a summary
    /// rejection? This is the memo the kernel decision reads, and the one
    /// whose miss costs a full callee lowering (chelis#1835).
    helper_summary_rejects: RefCell<UnordMap<String, bool>>,
    /// Program-wide: the top-level fn call graph.
    call_graph: RefCell<Option<BTreeMap<String, BTreeSet<String>>>>,
    /// Program-wide: every top-level definition body by name.
    program_defs: RefCell<Option<Arc<BTreeMap<String, Expr>>>>,
    /// Program-wide: the checker's effect row per definition.
    def_effect_rows: RefCell<Option<Arc<BTreeMap<String, chelis_types::types::EffectSet>>>>,
    /// Program-wide: the subexpression lowering context the evaluation
    /// profile and the C execution plan both read.
    subexpr_lowering_context: RefCell<Option<crate::lower::SubexprLoweringContext>>,
    /// Program-wide: which definitions reach a runtime-shaped `to_tensor`.
    dynamic_to_tensor_def_summaries: RefCell<Option<Arc<BTreeMap<String, bool>>>>,
}

/// One host-lowering session: a checked program, plus the facts host lowering
/// derives from it.
///
/// This is the representation chelis#1835 puts where a thread-local memo used
/// to be. That memo was seven caches keyed on
/// `program as *const CheckedProgram as usize` and gated on a thread-local
/// flag an entry point had to arm. #935 introduced it for two caches, #1332
/// grew it to seven, and #1531 added a third entry point, `host_def_kernel`,
/// that never armed it, so the interpreter paid a call-graph expansion per
/// applied definition (chelis#1829).
///
/// What the type buys, in order of how much it matters:
///
/// - There is no key, so no entry can be stale. A freed program's address can
///   be reused by a later one; a borrow cannot.
/// - The borrow checker binds the facts to the program they describe, so they
///   cannot outlive it.
/// - Nested entries are two sessions that share nothing, so a whole-program
///   lowering reached from inside an evaluation can neither read nor clear
///   the evaluation's facts. The arming flag had no way to express that.
/// - An entry point that does not establish a session does not compile, which
///   is the part no convention could supply.
///
/// `Deref` is what keeps the change mechanical: every function that took
/// `&CheckedProgram` takes `&HostLoweringSession` and reads the program
/// through it unchanged.
pub struct HostLoweringSession<'program> {
    program: &'program CheckedProgram,
    facts: DefLaneFacts,
}

impl<'program> HostLoweringSession<'program> {
    /// Begin a session over `program`. Nothing is derived here; every field
    /// fills in on its first miss.
    pub fn new(program: &'program CheckedProgram) -> Self {
        Self {
            program,
            facts: DefLaneFacts::default(),
        }
    }

    /// The program this session derives its facts from.
    pub fn program(&self) -> &'program CheckedProgram {
        self.program
    }

    /// The checked subexpression-lowering context for this program.
    ///
    /// Host-side execution routes use this accessor rather than rebuilding a
    /// context from the ordinary type environment, because the checker-owned
    /// local tensor-ascription obligations are an independent artifact
    /// channel and do not exist in inferred type metadata.
    pub fn checked_subexpr_lowering_context(&self) -> crate::lower::SubexprLoweringContext {
        cached_subexpr_lowering_context(self)
    }
}

impl Deref for HostLoweringSession<'_> {
    type Target = CheckedProgram;

    fn deref(&self) -> &Self::Target {
        self.program
    }
}

/// A caught lowering panic cannot leave a session's facts WRONG, only
/// incomplete. Every field is written once, with a fully computed value, and
/// a `RefCell` guard releases its borrow while unwinding, so a retry after a
/// caught panic recomputes the missing fact rather than reading a half-built
/// one. `try_lower_compiled_program_with_lane_overrides` runs the whole-program
/// lowering inside `catch_lowering_external` while holding the session, and
/// this is the claim that call site needs. Asserting it here rather than with
/// an `AssertUnwindSafe` at the call site keeps the claim attached to the type
/// that has to satisfy it, so a future field cannot smuggle itself past a
/// blanket assertion.
impl std::panic::RefUnwindSafe for HostLoweringSession<'_> {}

/// Kernel-decision summary probes on this thread that missed the memo and ran
/// a full callee lowering. A counted receipt bounds this instead of timing the
/// evaluation, so it cannot flake under load. chelis#1829.
pub fn host_summary_probe_builds() -> u64 {
    HOST_SUMMARY_PROBE_BUILDS.with(Cell::get)
}

/// Reset [`host_summary_probe_builds`] for this thread.
pub fn reset_host_summary_probe_builds() {
    HOST_SUMMARY_PROBE_BUILDS.with(|builds| builds.set(0));
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TensorHelperPreflightFacts {
    reaches_dynamic_to_tensor: bool,
    contains_grad_like: bool,
}

/// The lexical callable bindings visible while precomputing tensor-helper
/// facts. This type deliberately does not implement `Clone`: nested `fn` and
/// `let` expressions must update one scope and then restore it, rather than
/// copying a growing map at every boundary.
#[derive(Default)]
struct CallableScope {
    bindings: UnordMap<String, Option<String>>,
}

struct CallableScopeUndo {
    name: String,
    previous: Option<Option<String>>,
}

impl CallableScope {
    fn contains_key(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    fn get(&self, name: &str) -> Option<&Option<String>> {
        self.bindings.get(name)
    }

    fn bind(&mut self, name: String, target: Option<String>) -> CallableScopeUndo {
        record_host_work(|profile| profile.callable_scope_work += 1);
        let previous = self.bindings.insert(name.clone(), target);
        CallableScopeUndo { name, previous }
    }

    fn restore(&mut self, undos: impl IntoIterator<Item = CallableScopeUndo>) {
        for undo in undos.into_iter() {
            record_host_work(|profile| profile.callable_scope_work += 1);
            if let Some(previous) = undo.previous {
                self.bindings.insert(undo.name, previous);
            } else {
                self.bindings.remove(&undo.name);
            }
        }
    }
}

/// Limits pointer-keyed tensor-helper preflight facts to the lifetime of the
/// owned Deep expression they describe. Host lowering creates rewritten body
/// trees for callable-let substitution; their addresses are stable only while
/// that body is being lowered and must never escape into a program-wide cache.
struct TensorHelperPreflightGuard;

impl TensorHelperPreflightGuard {
    fn begin(expr: &Expr, program: &HostLoweringSession<'_>) -> Self {
        let summaries = cached_dynamic_to_tensor_def_summaries(program);
        let mut facts = UnordMap::new();
        analyze_tensor_helper_preflight(expr, &summaries, &mut facts);
        TENSOR_HELPER_PREFLIGHT_STACK.with(|stack| stack.borrow_mut().push(facts));
        Self
    }

    fn begin_if_uncovered(expr: &Expr, program: &HostLoweringSession<'_>) -> Option<Self> {
        let key = expr as *const Expr as usize;
        let covered = TENSOR_HELPER_PREFLIGHT_STACK.with(|stack| {
            stack
                .borrow()
                .iter()
                .rev()
                .any(|facts| facts.contains_key(&key))
        });
        (!covered).then(|| Self::begin(expr, program))
    }
}

impl Drop for TensorHelperPreflightGuard {
    fn drop(&mut self) {
        TENSOR_HELPER_PREFLIGHT_STACK.with(|stack| {
            stack
                .borrow_mut()
                .pop()
                .expect("tensor-helper preflight guards must be balanced");
        });
    }
}

#[derive(Debug, Clone)]
pub struct CompiledProgram {
    pub dag: Option<crate::Dag>,
    pub host: Option<ConcreteHostProgram>,
}

/// An existing concrete host program together with lowering-owned execution
/// associations for its fixed-control tensor helpers. The program's public
/// representation stays unchanged; only consuming ownership/codegen APIs can
/// retain the private associations.
///
/// ```compile_fail
/// use chelis_ir::host::HostExecutionPlan;
/// fn replace_helper(mut plan: HostExecutionPlan) {
///     plan.program().global_tensor_helpers.clear();
/// }
/// ```
///
/// ```compile_fail
/// use chelis_ir::host::{ConcreteHostProgram, HostExecutionPlan};
/// fn substitute(plan: HostExecutionPlan, forged: ConcreteHostProgram) {
///     let _ = plan.project(forged);
/// }
/// ```
#[derive(Debug)]
pub struct HostExecutionPlan {
    program: ConcreteHostProgram,
    global: Vec<HelperExecutionProduct>,
    functions: Vec<Vec<HelperExecutionProduct>>,
}

pub(crate) type HostExecutionAssociations = (
    Vec<Option<crate::evaluation::ExecutionMetadata>>,
    Vec<Vec<Option<crate::evaluation::ExecutionMetadata>>>,
);
type HostExecutionParts = (ConcreteHostProgram, HostExecutionAssociations);

#[derive(Debug, Clone)]
struct HelperExecutionProduct {
    execution: Option<crate::evaluation::ExecutionMetadata>,
    #[cfg(feature = "lowering-trace")]
    trace: Option<crate::lowering_trace::HelperLoweringTrace>,
}

impl HostExecutionPlan {
    pub fn program(&self) -> &ConcreteHostProgram {
        &self.program
    }

    /// Recover the ordinary public host payload only when no execution
    /// association would be discarded.
    pub fn into_ordinary(self) -> Result<ConcreteHostProgram, String> {
        let retains_private_metadata = self.has_execution_helpers() || {
            #[cfg(feature = "lowering-trace")]
            {
                self.has_helper_traces()
            }
            #[cfg(not(feature = "lowering-trace"))]
            {
                false
            }
        };
        if retains_private_metadata {
            Err("cannot discard retained host execution or trace metadata".into())
        } else {
            Ok(self.program)
        }
    }

    /// Apply a manifest-owned rewrite to global observation metadata. Tensor
    /// helpers and function bodies are not exposed through this capability.
    pub fn try_transform_globals<E>(
        mut self,
        transform: impl FnOnce(&mut Vec<ConcreteHostBinding>, &[ConcreteHostFunction]) -> Result<(), E>,
    ) -> Result<Self, E> {
        transform(&mut self.program.globals, &self.program.functions)?;
        Ok(self)
    }

    /// Select compiler-proved reachable functions by name, moving the exact
    /// original bodies, helper graphs, and metadata slots together. Entry
    /// projection drops globals just like the ordinary compiler projection.
    pub fn project_functions(self, names: &[String]) -> Result<Self, String> {
        let mut wanted = names
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        if wanted.len() != names.len() {
            return Err("host execution projection repeats a function name".into());
        }
        let Self {
            program,
            global: _,
            functions: execution,
        } = self;
        let mut functions = Vec::new();
        let mut selected_execution = Vec::new();
        for (function, metadata) in program.functions.into_iter().zip(execution) {
            if wanted.remove(&function.name) {
                functions.push(function);
                selected_execution.push(metadata);
            }
        }
        if let Some(missing) = wanted.into_iter().next() {
            return Err(format!(
                "projected function `{missing}` has no execution-plan origin"
            ));
        }
        let summary_rejections = functions
            .iter()
            .flat_map(|function| function.summary_rejections.iter().cloned())
            .collect();
        let projected = Self {
            program: ConcreteHostProgram {
                globals: Vec::new(),
                global_tensor_helpers: Vec::new(),
                functions,
                summary_rejections,
            },
            global: Vec::new(),
            functions: selected_execution,
        };
        projected.validate()?;
        Ok(projected)
    }

    /// Dropout requires a source-owned execution association. Ordinary
    /// UniformLike helpers retain their existing C host admission and stream
    /// advancement; their presence must not invalidate a planned sibling.
    pub fn has_unplanned_dropout_helper(&self) -> bool {
        fn dropout(helper: &HostTensorHelper) -> bool {
            helper
                .dag
                .nodes()
                .iter()
                .any(|node| matches!(node.op, crate::dag::RiscOp::Dropout { .. }))
        }
        self.program
            .global_tensor_helpers
            .iter()
            .zip(&self.global)
            .any(|(helper, product)| dropout(helper) && product.execution.is_none())
            || self
                .program
                .functions
                .iter()
                .zip(&self.functions)
                .any(|(function, executions)| {
                    function
                        .tensor_helpers
                        .iter()
                        .zip(executions)
                        .any(|(helper, product)| dropout(helper) && product.execution.is_none())
                })
    }

    /// Whether lowering retained at least one source-owned helper execution
    /// schedule. Callers use this to distinguish the additive C ingress from
    /// an ordinary host program whose metadata slots are all empty.
    pub fn has_execution_helpers(&self) -> bool {
        self.global
            .iter()
            .any(|product| product.execution.is_some())
            || self
                .functions
                .iter()
                .flatten()
                .any(|product| product.execution.is_some())
    }

    #[cfg(feature = "lowering-trace")]
    pub fn has_helper_traces(&self) -> bool {
        self.global.iter().any(|product| product.trace.is_some())
            || self
                .functions
                .iter()
                .flatten()
                .any(|product| product.trace.is_some())
    }

    /// Explicitly discard only opt-in helper observations. Execution
    /// metadata and exact helper graphs remain attached; callers still cannot
    /// recover an ordinary host program while either is retained.
    #[cfg(feature = "lowering-trace")]
    pub fn discard_helper_traces(mut self) -> Self {
        for product in self
            .global
            .iter_mut()
            .chain(self.functions.iter_mut().flatten())
        {
            product.trace = None;
        }
        self
    }

    #[cfg(feature = "lowering-trace")]
    pub fn global_helper_trace(
        &self,
        helper: usize,
    ) -> Result<Option<&crate::lowering_trace::HelperLoweringTrace>, String> {
        self.global
            .get(helper)
            .map(|product| product.trace.as_ref())
            .ok_or_else(|| format!("global helper {helper} has no execution-plan origin"))
    }

    #[cfg(feature = "lowering-trace")]
    pub fn global_helper_full_spine(
        &self,
        helper: usize,
    ) -> Result<Option<crate::lowering_trace::FullSpineObservation>, String> {
        self.global
            .get(helper)
            .map(|product| {
                product
                    .execution
                    .as_ref()
                    .map(|execution| execution.spine.full_observation())
            })
            .ok_or_else(|| format!("global helper {helper} has no execution-plan origin"))
    }

    #[cfg(feature = "lowering-trace")]
    pub fn function_helper_trace(
        &self,
        function: &str,
        helper: usize,
    ) -> Result<Option<&crate::lowering_trace::HelperLoweringTrace>, String> {
        let index = self
            .program
            .functions
            .iter()
            .position(|candidate| candidate.name == function)
            .ok_or_else(|| format!("function `{function}` has no execution-plan origin"))?;
        self.functions[index]
            .get(helper)
            .map(|product| product.trace.as_ref())
            .ok_or_else(|| {
                format!("function `{function}` helper {helper} has no execution-plan origin")
            })
    }

    #[cfg(feature = "lowering-trace")]
    pub fn function_helper_full_spine(
        &self,
        function: &str,
        helper: usize,
    ) -> Result<Option<crate::lowering_trace::FullSpineObservation>, String> {
        let index = self
            .program
            .functions
            .iter()
            .position(|candidate| candidate.name == function)
            .ok_or_else(|| format!("function `{function}` has no execution-plan origin"))?;
        self.functions[index]
            .get(helper)
            .map(|product| {
                product
                    .execution
                    .as_ref()
                    .map(|execution| execution.spine.full_observation())
            })
            .ok_or_else(|| {
                format!("function `{function}` helper {helper} has no execution-plan origin")
            })
    }

    fn validate(&self) -> Result<(), String> {
        if self.global.len() != self.program.global_tensor_helpers.len()
            || self.functions.len() != self.program.functions.len()
        {
            return Err("host execution metadata does not match helper topology".into());
        }
        for (helper, product) in self.program.global_tensor_helpers.iter().zip(&self.global) {
            if let Some(execution) = &product.execution {
                execution.validate_for_dag(&helper.dag)?;
            }
        }
        for (function, executions) in self.program.functions.iter().zip(&self.functions) {
            if executions.len() != function.tensor_helpers.len() {
                return Err(format!(
                    "function `{}` execution metadata does not match helper topology",
                    function.name
                ));
            }
            for (helper, product) in function.tensor_helpers.iter().zip(executions) {
                if let Some(execution) = &product.execution {
                    execution.validate_for_dag(&helper.dag)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn into_parts(self) -> HostExecutionParts {
        let global = self
            .global
            .into_iter()
            .map(|product| product.execution)
            .collect();
        let functions = self
            .functions
            .into_iter()
            .map(|products| {
                products
                    .into_iter()
                    .map(|product| product.execution)
                    .collect()
            })
            .collect();
        (self.program, (global, functions))
    }
}

#[derive(Debug)]
struct TensorHelperSink {
    transferred_result_claim_axes: Vec<crate::dag::RtAxis>,
    helpers: Vec<HostTensorHelper>,
    products: Vec<HelperExecutionProduct>,
    /// Checked declaration whose body is currently being lowered. Local
    /// ascription identities are artifact-local, so synthetic helper regions
    /// must select records by declaration as well as by source span.
    declaration_name: Option<String>,
    collect_execution: bool,
    collect_trace: bool,
}

#[derive(Debug, Clone)]
struct LoweredHostFunction {
    function: HostFunction,
    products: Vec<HelperExecutionProduct>,
}

impl TensorHelperSink {
    fn new(collect_execution: bool, collect_trace: bool) -> Self {
        Self {
            transferred_result_claim_axes: Vec::new(),
            helpers: Vec::new(),
            products: Vec::new(),
            declaration_name: None,
            collect_execution,
            collect_trace,
        }
    }

    fn for_declaration(
        collect_execution: bool,
        collect_trace: bool,
        declaration_name: impl Into<String>,
    ) -> Self {
        let mut sink = Self::new(collect_execution, collect_trace);
        sink.declaration_name = Some(declaration_name.into());
        sink
    }

    fn push_helper(
        &mut self,
        helper: HostTensorHelper,
        execution: Option<crate::evaluation::ExecutionMetadata>,
        #[cfg(feature = "lowering-trace")] trace: Option<
            crate::lowering_trace::HelperLoweringTrace,
        >,
    ) {
        self.helpers.push(helper);
        self.products.push(HelperExecutionProduct {
            execution,
            #[cfg(feature = "lowering-trace")]
            trace,
        });
    }

    fn into_parts(self) -> (Vec<HostTensorHelper>, Vec<HelperExecutionProduct>) {
        (self.helpers, self.products)
    }
}

impl Deref for TensorHelperSink {
    type Target = Vec<HostTensorHelper>;

    fn deref(&self) -> &Self::Target {
        &self.helpers
    }
}

impl DerefMut for TensorHelperSink {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.helpers
    }
}

#[derive(Debug, Clone)]
pub struct HostProgram<T = HostTypeTerm> {
    pub globals: Vec<HostBinding<T>>,
    pub global_tensor_helpers: Vec<HostTensorHelper>,
    pub functions: Vec<HostFunction<T>>,
    /// Structured rejections collected during sparse-helper summary
    /// recognition. Populated by [`try_lower_compiled_program`] when a
    /// near-summary-eligible callsite (helper body contains a sparse
    /// `RiscOp::Gather`, `RiscOp::ScatterAdd`, or `RiscOp::Scatter`)
    /// is rejected by the recognizer.
    ///
    /// Each entry carries the helper identity, the callsite + helper
    /// body spans, the rejection class, and structured class-specific
    /// detail. Consumers (CLI diagnostic reporter, downstream tooling,
    /// red-team tests) MUST pattern-match on the enum variants and
    /// struct fields rather than parsing the `Display` rendering.
    ///
    /// See `crates/chelis-cli/tests/cross_library_semantic_gap_diagnostics.rs`
    /// for the acceptance oracle that locks the structured shape of
    /// each rejection class.
    pub summary_rejections: Vec<SummaryRejection>,
}

impl<T> Default for HostProgram<T> {
    fn default() -> Self {
        Self {
            globals: Vec::new(),
            global_tensor_helpers: Vec::new(),
            functions: Vec::new(),
            summary_rejections: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HostBinding<T = HostTypeTerm> {
    pub name: String,
    pub display_name: Option<String>,
    /// Manifest-selected leaf observations. Empty preserves the legacy
    /// `display_name` behavior; non-empty entries carry structural paths so
    /// compiled observation never rediscovers tuple/ADT topology.
    pub display_roots: Vec<HostDisplayRoot>,
    pub ty: T,
    pub value: HostExpr<T>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostDisplayRoot {
    pub name: String,
    pub path: Vec<chelis_types::manifest::RootPathStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostFunctionOrigin {
    Authored,
    Monomorphized,
}

#[derive(Debug, Clone)]
pub struct HostFunction<T = HostTypeTerm> {
    /// Exact authored literal axes whose obligations lowering installed in
    /// this body's tensor helpers. Empty means the host owns those claims.
    pub helper_result_claim_axes: Vec<crate::dag::RtAxis>,
    pub name: String,
    pub params: Vec<HostParam<T>>,
    pub ret_ty: T,
    pub body: HostExpr<T>,
    pub tensor_helpers: Vec<HostTensorHelper>,
    /// Semantic provenance survives ABI projection. Symbol spelling cannot
    /// supply provenance because valid authored snake_case names can match
    /// the private mangling grammar.
    pub origin: HostFunctionOrigin,
    pub specialization: Option<HostFunctionSpecialization>,
    /// Structured rejections collected from this function's tensor
    /// helpers and from the function-level summary-derivation pass.
    /// See `HostProgram::summary_rejections`.
    pub summary_rejections: Vec<SummaryRejection>,
}

#[derive(Debug, Clone)]
pub struct HostParam<T = HostTypeTerm> {
    pub name: String,
    pub ty: T,
}

pub type ConcreteHostProgram = HostProgram<ConcreteHostType>;
pub type ConcreteHostBinding = HostBinding<ConcreteHostType>;
pub type ConcreteHostFunction = HostFunction<ConcreteHostType>;
pub type ConcreteHostParam = HostParam<ConcreteHostType>;

impl<T> HostFunction<T> {
    pub fn is_monomorphized_specialization(&self) -> bool {
        self.origin == HostFunctionOrigin::Monomorphized
    }
}

#[derive(Debug, Clone)]
pub struct HostTensorHelper {
    pub name: String,
    pub dag: crate::Dag,
    pub inputs: Vec<HostTensorInput>,
    pub output: TensorType,
    pub specialization: Option<HostTensorSpecialization>,
    /// Partial rejection captured at helper-construction time when the
    /// helper body is summary-near-eligible (root op is one of the
    /// three sparse RiscOps, or contains one in a recognizable place)
    /// but the summarizer rejected it.
    ///
    /// Carries everything the helper itself knows: rejection class,
    /// structured detail, and helper-body span. The owning function's
    /// name (the `HelperPath`) and the callsite span are filled in by
    /// [`derive_host_function_specializations`] when the function-level
    /// pass walks each fn's tensor helpers; the per-helper data is
    /// then promoted into a fully-formed `SummaryRejection` on
    /// `HostFunction::summary_rejections` and
    /// `HostProgram::summary_rejections`.
    pub summary_rejection: Option<HelperSummaryRejection>,
}

#[derive(Debug, Clone)]
pub struct HostTensorInput {
    pub name: String,
    pub ty: TensorType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostTensorSpecialization {
    BlasMatmul(HostBlasMatmulSummary),
    SparseGather(HostSparseOpSummary),
    SparseScatterAdd(HostSparseOpSummary),
    SparseScatterReplace(HostSparseOpSummary),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostFunctionSpecialization {
    BlasMatmul(HostBlasMatmulSummary),
    SparseGather(HostSparseOpSummary),
    SparseScatterAdd(HostSparseOpSummary),
    SparseScatterReplace(HostSparseOpSummary),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostBlasMatmulSummary {
    pub lhs_input: usize,
    pub rhs_input: usize,
    pub input_tys: Vec<TensorType>,
    pub output: TensorType,
    pub batch_dims: Vec<DimExpr>,
    pub m: DimExpr,
    pub n: DimExpr,
    pub k: DimExpr,
}

/// Compiler-derived summary for a sparse helper body (`Gather`,
/// `ScatterAdd`, or `Scatter`/replace).
///
/// `input_indices` is the ordered list of helper input positions that
/// supply the sparse op's operands. The order matches the RiscOp's
/// `inputs` order:
///   * Gather: `[values, indices]`
///   * ScatterAdd / Scatter: `[target, indices, updates]`
///
/// `input_tys` is the full ordered tuple of helper-input types (one
/// entry per helper input parameter, not just the sparse operands).
/// `output` is the tensor type the sparse op produces. The remapper
/// uses `input_tys` to verify that the callsite arg types still match
/// the recognized helper-body types after wrapper propagation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostSparseOpSummary {
    pub axis: usize,
    pub input_indices: Vec<usize>,
    pub input_tys: Vec<TensorType>,
    pub output: TensorType,
}

// =========================================================================
// W4-A — M5(c) structured rejection diagnostics
// =========================================================================
//
// `SummaryRejection` is the public diagnostic shape emitted when a
// callsite is summary-eligible (helper body contains a sparse RiscOp)
// but the summarizer rejects it. The shape is intentionally
// **programmatically matchable**: downstream consumers must pattern-
// match on `rejection_class` (an enum) and the strongly-typed
// `detail` enum payload rather than parsing the rendered `Display`
// string. The acceptance oracle for this contract is
// `crates/chelis-cli/tests/cross_library_semantic_gap_diagnostics.rs`.
//
// Variant naming mirrors W3-B's seven enumerated rejection categories
// (see `crates/chelis-ir/tests/host_sparse_summary.rs` and the cli
// counterparts in `crates/chelis-cli/tests/cross_library_sparse_summaries.rs`):
//
//   1. `MultipleRoots`               — helper body has multiple DAG roots
//   2. `MultipleReturnPaths`         — helper body branches via if/then/else
//   3. `NonLoadOperand`              — sparse-op operand is not a direct Load
//   4. `PostProcessingAfterSparseOp` — helper post-processes the sparse result
//   5. `IndicesDTypeMismatch`        — indices precision not i32/i64
//   6. `PayloadDTypeMismatch`        — values/target/updates/output disagree
//   7. `WildcardDim`                 — `Named("*", None)` placeholder in input/output
//
// Three additional variants (`UnrecognizedShape`, `NonContiguousLayout`,
// `RankMismatch`) are reserved for future helper-shape categories that
// the current summarizer does not check today but the plan's pinned
// shape names explicitly.

/// Identifies the rejected helper for diagnostic attribution.
///
/// `module` is the producing module path (today always `None` — the
/// host program is flat — but reserved for the Reef/library-import
/// case where helpers come from imported modules and a fully qualified
/// helper identity matters).
///
/// `def_name` is the Surf `def` name of the function whose body owns
/// the rejected tensor helper. For a top-level expression binding
/// (e.g. `result = gather(...)`) this is the synthetic global name
/// (`__global` etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelperPath {
    pub module: Option<String>,
    pub def_name: String,
}

impl HelperPath {
    /// Convenience constructor for the module-less (flat-program) case.
    pub fn local(def_name: impl Into<String>) -> Self {
        Self {
            module: None,
            def_name: def_name.into(),
        }
    }
}

impl fmt::Display for HelperPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(module) = &self.module {
            write!(f, "{module}.{}", self.def_name)
        } else {
            write!(f, "{}", self.def_name)
        }
    }
}

/// Which payload role a `PayloadDTypeMismatch` is reporting on. The
/// sparse RiscOps have different operand roles:
///   * `Gather`: only `Values` is a payload (the indexed-into tensor);
///     the output type's precision must match.
///   * `ScatterAdd` / `Scatter`: `Target` (the base), `Updates` (the
///     scattered values), and the output type's precision must all
///     match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadRole {
    Values,
    Target,
    Updates,
    Output,
}

impl fmt::Display for PayloadRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PayloadRole::Values => "values",
            PayloadRole::Target => "target",
            PayloadRole::Updates => "updates",
            PayloadRole::Output => "output",
        };
        f.write_str(s)
    }
}

/// Which tensor (input or output) carries the wildcard `Named("*",
/// None)` dim that disqualified the helper. `Input(i)` is the
/// positional helper input index (matches `HostSparseOpSummary::input_indices`
/// indexing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WildcardLocation {
    Output,
    Input(usize),
}

impl fmt::Display for WildcardLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WildcardLocation::Output => f.write_str("output"),
            WildcardLocation::Input(i) => write!(f, "input[{i}]"),
        }
    }
}

/// Which sparse RiscOp the helper's body root names.
///
/// `Unknown` covers the case where the body root isn't a sparse op at
/// all (the helper is post-processing a sparse op, or the root is
/// something else entirely). The summarizer reports the rejection
/// against the deepest sparse op it finds in the helper body so the
/// diagnostic still names a specific op when possible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseOpKind {
    Gather,
    ScatterAdd,
    ScatterReplace,
    Unknown,
}

impl fmt::Display for SparseOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SparseOpKind::Gather => "gather",
            SparseOpKind::ScatterAdd => "scatter_add",
            SparseOpKind::ScatterReplace => "scatter_replace",
            SparseOpKind::Unknown => "<unknown>",
        };
        f.write_str(s)
    }
}

/// Closed enumeration of summary-rejection classes. Each variant is a
/// distinct public contract; consumers MUST match on the variant. New
/// classes are additive: adding a variant is a breaking change for
/// exhaustive matches but never silently re-routes an existing case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SummaryRejectionClass {
    /// Helper body has more than one DAG root (multi-output helper).
    /// Distinct from `MultipleReturnPaths` because the multi-root case
    /// is a DAG-structural property; the multi-return-path case is a
    /// host-side host-expr-shape property (the body lowered to `If`).
    MultipleRoots,
    /// Helper body branches via if/then/else (or another host-side
    /// conditional). Detected by the function-level pass when the
    /// function body is a `HostExprKind::If` rather than a `TensorCall`.
    MultipleReturnPaths,
    /// One of the sparse-op operands is not a direct `RiscOp::Load`
    /// of a helper input (e.g. an intervening elementwise op, a cast,
    /// or a reshape).
    NonLoadOperand,
    /// The helper body has an extra op on top of the sparse op (e.g.
    /// `add(gather(...), zero)`). The sparse op is not the DAG root.
    PostProcessingAfterSparseOp,
    /// The indices operand's precision is neither `i32` nor `i64`.
    IndicesDTypeMismatch,
    /// One of the payload precisions (values, target, updates, output)
    /// disagrees with the others.
    PayloadDTypeMismatch,
    /// Helper input or output carries a `Named("*", None)` wildcard
    /// dim — a type-inference placeholder that does not bind to a
    /// unique callsite axis.
    WildcardDim,
    /// Reserved: helper body shape does not match any recognized
    /// sparse-op pattern. Today the summarizer routes most "shape
    /// doesn't match" cases through `NonLoadOperand` /
    /// `PostProcessingAfterSparseOp`; this variant is in the public
    /// surface for forward-compatibility with future shape extensions.
    UnrecognizedShape,
    /// Reserved: helper body uses non-contiguous tensor layouts that
    /// the summary contract requires to be contiguous. No current
    /// summarizer check fires this; reserved for future use.
    NonContiguousLayout,
    /// Reserved: helper input or output rank disagrees with the rank
    /// the sparse op requires. Today the summarizer rejects rank
    /// mismatches through `NonLoadOperand` (the type-equality check on
    /// the Load op fails); reserved for future explicit rank checks.
    RankMismatch,
    // ---------------------------------------------------------------
    // W6 Task A — BLAS-recognizer rejection classes.
    //
    // The BLAS recognizer (`try_summarize_blas_helper` in
    // `crates/chelis-ir/src/host.rs`) exposes six structural failure
    // points where a helper-DAG that *almost* matched
    // `RiscOp::BlasMatmul` was rejected. Each is a distinct
    // BLAS-prefixed variant: the source recognizer is encoded in the
    // variant name so tooling pattern-matching on the enum sees both
    // "what shape failed" and "which recognizer rejected it" without
    // needing to inspect an out-of-band recognizer-identity tag.
    //
    // The variant names are NOT collapsed with the sparse-named
    // equivalents (`MultipleRoots`, `NonLoadOperand`) even though the
    // detail payload shapes coincide today; future divergence is
    // cheap to absorb when each path owns its own variant.
    // ---------------------------------------------------------------
    /// BLAS helper DAG (post-specialize) has more than one root.
    /// Mirrors `MultipleRoots` for sparse helpers but identifies the
    /// BLAS recognizer as the source.
    BlasMultipleRoots,
    /// BLAS helper's declared output precision is not `f32`. Today
    /// the recognizer requires `f32` output for the BlasMatmul
    /// path; non-`f32` outputs (e.g. an `f64` matmul helper) silently
    /// skipped through `Option::None` before W6 — now they are
    /// diagnosed.
    BlasOutputPrecisionMismatch,
    /// The helper's specialized DAG root is not `RiscOp::BlasMatmul`,
    /// so the recognizer could not extract `batch_dims`, `m`, `n`,
    /// `k`. Includes both the "root op is unrelated" case and the
    /// "root op shape doesn't match BlasMatmul's expected operand
    /// count / output precision" rejection.
    BlasNotMatmulPattern,
    /// One of the matmul operands is not a direct `RiscOp::Load` of a
    /// helper input. Mirrors `NonLoadOperand` for sparse helpers but
    /// identifies the BLAS recognizer as the source.
    BlasNonLoadOperand,
    /// At least one of the helper's input tensors has precision other
    /// than `f32`. Today the recognizer requires every helper input to
    /// be `f32`; mixed-precision inputs (e.g. an `f32 @ i8` quantized
    /// matmul helper) silently skipped before W6.
    BlasInputPrecisionMismatch,
    /// One of `batch_dims`, `m`, `n`, `k` could not be bound to any
    /// helper input dim by name. The recognizer requires every
    /// matmul-derived dim symbol to appear on at least one input
    /// tensor type's dim list; an unbindable dim means the helper
    /// signature does not name its own contraction axes.
    BlasDimensionBindingFailure,
}

impl fmt::Display for SummaryRejectionClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SummaryRejectionClass::MultipleRoots => "multiple-roots",
            SummaryRejectionClass::MultipleReturnPaths => "multiple-return-paths",
            SummaryRejectionClass::NonLoadOperand => "non-load-operand",
            SummaryRejectionClass::PostProcessingAfterSparseOp => "post-processing-after-sparse-op",
            SummaryRejectionClass::IndicesDTypeMismatch => "indices-dtype-mismatch",
            SummaryRejectionClass::PayloadDTypeMismatch => "payload-dtype-mismatch",
            SummaryRejectionClass::WildcardDim => "wildcard-dim",
            SummaryRejectionClass::UnrecognizedShape => "unrecognized-shape",
            SummaryRejectionClass::NonContiguousLayout => "non-contiguous-layout",
            SummaryRejectionClass::RankMismatch => "rank-mismatch",
            SummaryRejectionClass::BlasMultipleRoots => "blas-multiple-roots",
            SummaryRejectionClass::BlasOutputPrecisionMismatch => "blas-output-precision-mismatch",
            SummaryRejectionClass::BlasNotMatmulPattern => "blas-not-matmul-pattern",
            SummaryRejectionClass::BlasNonLoadOperand => "blas-non-load-operand",
            SummaryRejectionClass::BlasInputPrecisionMismatch => "blas-input-precision-mismatch",
            SummaryRejectionClass::BlasDimensionBindingFailure => "blas-dimension-binding-failure",
        };
        f.write_str(s)
    }
}

/// Class-specific structured payload for a `SummaryRejection`. Each
/// variant mirrors a `SummaryRejectionClass` variant, but only the
/// classes that carry additional structured information have a payload;
/// the payload-less classes use the unit `*Empty` variants. This split
/// keeps the public surface programmatically matchable on
/// (`rejection_class`, `detail`) jointly without forcing every consumer
/// to inspect `detail` for an empty payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SummaryRejectionDetail {
    MultipleRoots {
        /// Number of DAG roots observed in the helper body.
        root_count: usize,
    },
    MultipleReturnPaths {
        /// Approximate number of return-arm branches observed in the
        /// function body. Today the host lowerer collapses multiple
        /// branches into a single nested `If`; this field records the
        /// branch count at the outermost level.
        branch_count: usize,
    },
    NonLoadOperand {
        /// Sparse op the helper's body root names.
        op: SparseOpKind,
        /// Positional index of the operand that wasn't a Load (0 =
        /// first operand, 1 = second, etc.). Matches the RiscOp's
        /// `inputs` ordering.
        operand_index: usize,
    },
    PostProcessingAfterSparseOp {
        /// Sparse op that was post-processed (the deepest sparse op in
        /// the helper body).
        op: SparseOpKind,
        /// Snake-case name of the tail op sitting on top of the sparse
        /// op (e.g. `"add"`, `"reshape"`).
        tail_op: String,
    },
    IndicesDTypeMismatch {
        op: SparseOpKind,
        observed: Prim,
    },
    PayloadDTypeMismatch {
        op: SparseOpKind,
        /// Which payload role's dtype did not match the others.
        which: PayloadRole,
        /// Expected precision (the output / target / values precision
        /// the others should have matched).
        expected: Prim,
        /// Observed precision on the mismatching payload.
        observed: Prim,
    },
    WildcardDim {
        location: WildcardLocation,
    },
    /// Empty payload for the reserved classes (UnrecognizedShape,
    /// NonContiguousLayout, RankMismatch). Carries no structured
    /// information today.
    Reserved,
    // ---------------------------------------------------------------
    // W6 Task A — BLAS-recognizer detail payloads.
    //
    // Each variant mirrors a `SummaryRejectionClass::Blas*` variant.
    // The payloads name the observed-precision / failing-dim values
    // so tooling can distinguish e.g. "f64 helper rejected" from
    // "i32 helper rejected" without re-running the recognizer.
    // ---------------------------------------------------------------
    BlasMultipleRoots {
        /// Number of DAG roots observed in the helper's
        /// post-specialize body.
        root_count: usize,
    },
    BlasOutputPrecisionMismatch {
        /// Helper's declared output precision (the one that disagreed
        /// with the recognizer's required `f32`).
        observed: Prim,
    },
    BlasNotMatmulPattern {
        /// Snake-case canonical name of the root op the recognizer
        /// observed in place of `BlasMatmul` (e.g. `"add"`,
        /// `"reshape"`, or `"<other>"` for ops outside the canonical
        /// name whitelist). When the root IS `BlasMatmul` but its
        /// rank/precision doesn't match (e.g. wrong input count or
        /// non-`f32` matmul output), `tail_op` is `"blas_matmul"` and
        /// the caller still sees this variant.
        tail_op: String,
    },
    BlasNonLoadOperand {
        /// Positional index of the operand that was not a direct
        /// `Load` (0 = lhs, 1 = rhs).
        operand_index: usize,
    },
    BlasInputPrecisionMismatch {
        /// Positional index of the helper input whose precision was
        /// not `f32`.
        input_index: usize,
        /// Observed precision on the mismatching helper input.
        observed: Prim,
    },
    BlasDimensionBindingFailure {
        /// Symbolic role of the dim that could not be bound: one of
        /// `"batch"`, `"m"`, `"n"`, or `"k"`. The recognizer reports
        /// the first failing role.
        role: BlasDimRole,
    },
}

/// Which matmul dim symbol failed to bind to any helper input in the
/// `BlasDimensionBindingFailure` rejection. The four roles match the
/// `HostBlasMatmulSummary` field layout (`batch_dims`, `m`, `n`, `k`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlasDimRole {
    Batch,
    M,
    N,
    K,
}

impl fmt::Display for BlasDimRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BlasDimRole::Batch => "batch",
            BlasDimRole::M => "m",
            BlasDimRole::N => "n",
            BlasDimRole::K => "k",
        };
        f.write_str(s)
    }
}

impl fmt::Display for SummaryRejectionDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SummaryRejectionDetail::MultipleRoots { root_count } => {
                write!(f, "{root_count} DAG roots in helper body")
            }
            SummaryRejectionDetail::MultipleReturnPaths { branch_count } => {
                write!(f, "{branch_count} return-path branches")
            }
            SummaryRejectionDetail::NonLoadOperand { op, operand_index } => write!(
                f,
                "{op} operand[{operand_index}] is not a direct load of a helper input"
            ),
            SummaryRejectionDetail::PostProcessingAfterSparseOp { op, tail_op } => {
                write!(
                    f,
                    "{tail_op} on top of {op}; sparse op is not the helper root"
                )
            }
            SummaryRejectionDetail::IndicesDTypeMismatch { op, observed } => {
                write!(f, "{op} indices dtype {observed:?} is not i32 / i64")
            }
            SummaryRejectionDetail::PayloadDTypeMismatch {
                op,
                which,
                expected,
                observed,
            } => write!(
                f,
                "{op} payload `{which}` dtype {observed:?} disagrees with expected {expected:?}"
            ),
            SummaryRejectionDetail::WildcardDim { location } => {
                write!(f, "wildcard dim on helper {location}")
            }
            SummaryRejectionDetail::Reserved => f.write_str("<reserved>"),
            SummaryRejectionDetail::BlasMultipleRoots { root_count } => {
                write!(f, "{root_count} DAG roots in BLAS helper body")
            }
            SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } => {
                write!(f, "BLAS matmul output precision {observed:?} is not f32")
            }
            SummaryRejectionDetail::BlasNotMatmulPattern { tail_op } => write!(
                f,
                "BLAS helper root op `{tail_op}` does not match the BlasMatmul pattern"
            ),
            SummaryRejectionDetail::BlasNonLoadOperand { operand_index } => write!(
                f,
                "BLAS matmul operand[{operand_index}] is not a direct load of a helper input"
            ),
            SummaryRejectionDetail::BlasInputPrecisionMismatch {
                input_index,
                observed,
            } => write!(
                f,
                "BLAS helper input[{input_index}] precision {observed:?} is not f32"
            ),
            SummaryRejectionDetail::BlasDimensionBindingFailure { role } => write!(
                f,
                "BLAS matmul dim `{role}` could not be bound to any helper input"
            ),
        }
    }
}

/// Public diagnostic shape for a rejected summary callsite. This is
/// the central correctness contract of W4-A. Downstream consumers
/// (CLI diagnostic reporter, red-team tests, future tooling) MUST
/// pattern-match on the enum variants and struct fields rather than
/// parsing the `Display` rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryRejection {
    pub rejection_class: SummaryRejectionClass,
    pub helper_path: HelperPath,
    /// Surf-source span ID of the callsite that would have consumed
    /// the summary (`surf:<start>..<end>` per
    /// `spec/design/chelis_span_survival.md` §1, threaded through
    /// host lowering as `HostExpr::span_id`). `None` only when the
    /// caller chain has no surf-side span at all (synthesized code).
    pub callsite_span: Option<String>,
    /// Surf-source span ID of the helper body itself (the helper-DAG
    /// root). `None` for synthetic helpers that never had a Surf
    /// span attached (rare; matches the `HostExpr::span_id`
    /// convention).
    pub helper_body_span: Option<String>,
    pub detail: SummaryRejectionDetail,
}

impl fmt::Display for SummaryRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let callsite = self.callsite_span.as_deref().unwrap_or("<no-span>");
        let body = self.helper_body_span.as_deref().unwrap_or("<no-span>");
        write!(
            f,
            "rejected summary for `{}` ({}): {}; callsite={}, helper-body={}",
            self.helper_path, self.rejection_class, self.detail, callsite, body,
        )
    }
}

/// Partial rejection captured at tensor-helper construction time, when
/// the owning function's name + callsite span are not yet known. The
/// function-level pass ([`derive_host_function_specializations`])
/// promotes each `HelperSummaryRejection` into a fully-formed
/// `SummaryRejection` on `HostFunction::summary_rejections`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelperSummaryRejection {
    pub rejection_class: SummaryRejectionClass,
    pub helper_body_span: Option<String>,
    pub detail: SummaryRejectionDetail,
}

#[derive(Debug, Clone)]
pub struct HostCallback<T = HostTypeTerm> {
    pub kind: HostCallbackKind<T>,
    pub ret_ty: T,
}

#[derive(Debug, Clone)]
pub struct HostMatchArm<T = HostTypeTerm> {
    pub ctor: String,
    pub bindings: Vec<HostPatternBinding<T>>,
    pub expr: HostExpr<T>,
}

#[derive(Debug, Clone)]
pub struct HostPatternBinding<T = HostTypeTerm> {
    pub name: String,
    pub ty: T,
    pub field_index: usize,
}

#[derive(Debug, Clone)]
pub struct HostAdtField<T = HostTypeTerm> {
    pub name: Option<String>,
    pub ty: T,
}

#[derive(Debug, Clone)]
pub enum HostCallbackKind<T = HostTypeTerm> {
    Named {
        function: String,
        params: Vec<HostParam<T>>,
    },
    Inline {
        params: Vec<HostParam<T>>,
        body: Box<HostExpr<T>>,
    },
}

/// Host-lane expression with span-survival metadata.
///
/// `HostExpr` is a struct wrapper around `HostExprKind` carrying the
/// canonical span ID and any spans accumulated through N→1 merges, mirroring
/// the `DagNode` schema documented in `spec/design/chelis_span_survival.md`
/// §2.2 / §2.3 (S6 host-side rule table).
///
/// **Schema is locked.** Both fields stay even though the host-side passes
/// shipped today don't all produce N→1 merges; the schema mirrors `DagNode`
/// for orchestrator-tooling uniformity (one audit consumer reads both
/// schemas) and so future host-side optimizations have the field they need.
/// Reject simplification proposals on the same grounds the spec rejects
/// collapsing `DagNode.merged_spans` back to a single `Option<String>`.
#[derive(Debug, Clone)]
pub struct HostExpr<T = HostTypeTerm> {
    pub kind: HostExprKind<T>,
    /// Canonical span ID, populated by host-lane lowering from the Deep
    /// `Expr`'s `meta["span"]` value. Threaded through later host-side
    /// passes per the rules in `spec/design/chelis_span_survival.md` §2.3
    /// host-side table.
    ///
    /// `None` is the normal case for hand-written Chelis or for nodes
    /// synthesized in places where the source-region rule does not apply.
    pub span_id: Option<String>,
    /// Additional spans accumulated when N→1 host-side merge passes
    /// collapse multiple source nodes into a single result node.
    ///
    /// Backend host emission (S6 step 5) emits one `// span:` line per
    /// `span_id ∪ merged_spans` so the audit invariant holds: every span
    /// ID present on any input Deep node appears on at least one IR or
    /// HostExpr node.
    pub merged_spans: Vec<String>,
}

impl<T> HostExpr<T> {
    /// Construct a HostExpr from a kind with no span metadata. The host-side
    /// lowering layer (§2.3 host-side table, rule "Lowering") populates
    /// `span_id` from the enclosing Deep expr's `meta["span"]` via the
    /// `with_span` constructor; default constructions (e.g. tests) start
    /// span-free.
    pub fn new(kind: HostExprKind<T>) -> Self {
        Self {
            kind,
            span_id: None,
            merged_spans: Vec::new(),
        }
    }

    /// Construct a HostExpr from a kind with an explicit span ID. Empty
    /// `merged_spans`. Used by the host-lane lowering pass (`lower_host_expr`)
    /// to attach the current Deep expr's span to every freshly-produced node.
    pub fn with_span(kind: HostExprKind<T>, span_id: Option<String>) -> Self {
        Self {
            kind,
            span_id,
            merged_spans: Vec::new(),
        }
    }

    /// Append a single span to this node's `merged_spans`, lex-sorted and
    /// deduped, with the same no-op rules as `crate::span_merge::append_span_to_node`:
    ///   * `span` is `None` (passthrough),
    ///   * the node's `span_id` already equals `span`,
    ///   * `merged_spans` already contains `span`.
    ///
    /// Used by the host-side N→1 lowering collapse rule (§2.3 host-side
    /// table, rule "Lowering. Body collapses to existing HostExpr"): when
    /// a parent Deep expr lowers to an already-constructed inner HostExpr
    /// (e.g. `(realize ...)`, `(handle-effect ... body)`, `(lit ...)` whose
    /// child is the canonical node), the parent's `span_id` appends here so
    /// the audit invariant ("every input span appears as `span_id` or in
    /// `merged_spans` on at least one node") still holds.
    pub fn append_merged_span(&mut self, span: Option<&str>) {
        let Some(span) = span else {
            return;
        };
        if self.span_id.as_deref() == Some(span) {
            return;
        }
        if self.merged_spans.iter().any(|s| s == span) {
            return;
        }
        self.merged_spans.push(span.to_owned());
        self.merged_spans.sort();
    }
}

#[derive(Debug, Clone)]
pub enum HostExprKind<T = HostTypeTerm> {
    /// A retained invocation's signature check, after its actual arguments
    /// have been evaluated and before its specialized body runs. All operands
    /// are borrowed observations, including parameters unused by that body.
    SignatureEntry {
        plan: SignatureEntryPlan,
        args: Vec<HostExpr<T>>,
    },
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    List(Vec<HostExpr<T>>, T),
    Tuple(Vec<HostExpr<T>>, T),
    Var(String, T),
    Call {
        function: String,
        args: Vec<HostExpr<T>>,
        arg_tys: Vec<T>,
        ty: T,
    },
    Builtin {
        name: String,
        args: Vec<HostExpr<T>>,
        ty: T,
    },
    AdtConstruct {
        ctor: String,
        fields: Vec<HostExpr<T>>,
        ty: T,
    },
    AdtFieldAccess {
        base: Box<HostExpr<T>>,
        field_index: usize,
        ty: T,
    },
    If {
        cond: Box<HostExpr<T>>,
        then_expr: Box<HostExpr<T>>,
        else_expr: Box<HostExpr<T>>,
        ty: T,
    },
    MatchOption {
        scrutinee: Box<HostExpr<T>>,
        bind_name: String,
        some_expr: Box<HostExpr<T>>,
        none_expr: Box<HostExpr<T>>,
        ty: T,
    },
    MatchAdt {
        scrutinee: Box<HostExpr<T>>,
        arms: Vec<HostMatchArm<T>>,
        default_expr: Option<Box<HostExpr<T>>>,
        ty: T,
    },
    Let {
        bindings: Vec<HostBinding<T>>,
        body: Box<HostExpr<T>>,
        ty: T,
    },
    Map {
        callback: HostCallback<T>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    Filter {
        callback: HostCallback<T>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    Fold {
        callback: HostCallback<T>,
        init: Box<HostExpr<T>>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    Scan {
        callback: HostCallback<T>,
        init: Box<HostExpr<T>>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    Partition {
        callback: HostCallback<T>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    FlatMap {
        callback: HostCallback<T>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    WithSeed {
        seed: Box<HostExpr<T>>,
        body: Box<HostExpr<T>>,
        ty: T,
    },
    TensorCall {
        helper: usize,
        args: Vec<HostExpr<T>>,
        ty: T,
    },
    Unit,
}

pub type ConcreteHostCallback = HostCallback<ConcreteHostType>;
pub type ConcreteHostMatchArm = HostMatchArm<ConcreteHostType>;
pub type ConcreteHostPatternBinding = HostPatternBinding<ConcreteHostType>;
pub type ConcreteHostAdtField = HostAdtField<ConcreteHostType>;
pub type ConcreteHostCallbackKind = HostCallbackKind<ConcreteHostType>;
pub type ConcreteHostExpr = HostExpr<ConcreteHostType>;
pub type ConcreteHostExprKind = HostExprKind<ConcreteHostType>;

fn resolve_host_program(
    program: HostProgram,
) -> Result<ConcreteHostProgram, crate::HostTypeResolutionError> {
    Ok(ConcreteHostProgram {
        globals: program
            .globals
            .into_iter()
            .map(resolve_host_binding)
            .collect::<Result<Vec<_>, _>>()?,
        global_tensor_helpers: program.global_tensor_helpers,
        functions: program
            .functions
            .into_iter()
            .map(resolve_host_function)
            .collect::<Result<Vec<_>, _>>()?,
        summary_rejections: program.summary_rejections,
    })
}

fn resolve_host_binding(
    binding: HostBinding,
) -> Result<ConcreteHostBinding, crate::HostTypeResolutionError> {
    Ok(ConcreteHostBinding {
        name: binding.name,
        display_name: binding.display_name,
        display_roots: binding.display_roots,
        ty: binding.ty.into_concrete()?,
        value: resolve_host_expr(binding.value)?,
    })
}

fn resolve_host_function(
    function: HostFunction,
) -> Result<ConcreteHostFunction, crate::HostTypeResolutionError> {
    Ok(ConcreteHostFunction {
        helper_result_claim_axes: function.helper_result_claim_axes,
        name: function.name,
        params: function
            .params
            .into_iter()
            .map(resolve_host_param)
            .collect::<Result<Vec<_>, _>>()?,
        ret_ty: function.ret_ty.into_concrete()?,
        body: resolve_host_expr(function.body)?,
        tensor_helpers: function.tensor_helpers,
        origin: function.origin,
        specialization: function.specialization,
        summary_rejections: function.summary_rejections,
    })
}

fn resolve_host_param(
    param: HostParam,
) -> Result<ConcreteHostParam, crate::HostTypeResolutionError> {
    Ok(ConcreteHostParam {
        name: param.name,
        ty: param.ty.into_concrete()?,
    })
}

fn resolve_host_callback(
    callback: HostCallback,
) -> Result<ConcreteHostCallback, crate::HostTypeResolutionError> {
    let kind = match callback.kind {
        HostCallbackKind::Named { function, params } => ConcreteHostCallbackKind::Named {
            function,
            params: params
                .into_iter()
                .map(resolve_host_param)
                .collect::<Result<Vec<_>, _>>()?,
        },
        HostCallbackKind::Inline { params, body } => ConcreteHostCallbackKind::Inline {
            params: params
                .into_iter()
                .map(resolve_host_param)
                .collect::<Result<Vec<_>, _>>()?,
            body: Box::new(resolve_host_expr(*body)?),
        },
    };
    Ok(ConcreteHostCallback {
        kind,
        ret_ty: callback.ret_ty.into_concrete()?,
    })
}

fn resolve_host_expr(expr: HostExpr) -> Result<ConcreteHostExpr, crate::HostTypeResolutionError> {
    let kind = match expr.kind {
        HostExprKind::SignatureEntry { plan, args } => ConcreteHostExprKind::SignatureEntry {
            plan,
            args: args
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
        },
        HostExprKind::Int(value) => ConcreteHostExprKind::Int(value),
        HostExprKind::Float(value) => ConcreteHostExprKind::Float(value),
        HostExprKind::Bool(value) => ConcreteHostExprKind::Bool(value),
        HostExprKind::String(value) => ConcreteHostExprKind::String(value),
        HostExprKind::List(items, ty) => ConcreteHostExprKind::List(
            items
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty.into_concrete()?,
        ),
        HostExprKind::Tuple(items, ty) => ConcreteHostExprKind::Tuple(
            items
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty.into_concrete()?,
        ),
        HostExprKind::Var(name, ty) => ConcreteHostExprKind::Var(name, ty.into_concrete()?),
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        } => ConcreteHostExprKind::Call {
            function,
            args: args
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            arg_tys: arg_tys
                .into_iter()
                .map(HostTypeTerm::into_concrete)
                .collect::<Result<Vec<_>, _>>()?,
            ty: ty.into_concrete()?,
        },
        HostExprKind::Builtin { name, args, ty } => ConcreteHostExprKind::Builtin {
            name,
            args: args
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty: ty.into_concrete()?,
        },
        HostExprKind::AdtConstruct { ctor, fields, ty } => ConcreteHostExprKind::AdtConstruct {
            ctor,
            fields: fields
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty: ty.into_concrete()?,
        },
        HostExprKind::AdtFieldAccess {
            base,
            field_index,
            ty,
        } => ConcreteHostExprKind::AdtFieldAccess {
            base: Box::new(resolve_host_expr(*base)?),
            field_index,
            ty: ty.into_concrete()?,
        },
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ty,
        } => ConcreteHostExprKind::If {
            cond: Box::new(resolve_host_expr(*cond)?),
            then_expr: Box::new(resolve_host_expr(*then_expr)?),
            else_expr: Box::new(resolve_host_expr(*else_expr)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        } => ConcreteHostExprKind::MatchOption {
            scrutinee: Box::new(resolve_host_expr(*scrutinee)?),
            bind_name,
            some_expr: Box::new(resolve_host_expr(*some_expr)?),
            none_expr: Box::new(resolve_host_expr(*none_expr)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        } => ConcreteHostExprKind::MatchAdt {
            scrutinee: Box::new(resolve_host_expr(*scrutinee)?),
            arms: arms
                .into_iter()
                .map(|arm| {
                    Ok(ConcreteHostMatchArm {
                        ctor: arm.ctor,
                        bindings: arm
                            .bindings
                            .into_iter()
                            .map(|binding| {
                                Ok(ConcreteHostPatternBinding {
                                    name: binding.name,
                                    ty: binding.ty.into_concrete()?,
                                    field_index: binding.field_index,
                                })
                            })
                            .collect::<Result<Vec<_>, crate::HostTypeResolutionError>>()?,
                        expr: resolve_host_expr(arm.expr)?,
                    })
                })
                .collect::<Result<Vec<_>, crate::HostTypeResolutionError>>()?,
            default_expr: default_expr
                .map(|expr| resolve_host_expr(*expr).map(Box::new))
                .transpose()?,
            ty: ty.into_concrete()?,
        },
        HostExprKind::Let { bindings, body, ty } => ConcreteHostExprKind::Let {
            bindings: bindings
                .into_iter()
                .map(resolve_host_binding)
                .collect::<Result<Vec<_>, _>>()?,
            body: Box::new(resolve_host_expr(*body)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::Map { callback, list, ty } => ConcreteHostExprKind::Map {
            callback: resolve_host_callback(callback)?,
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::Filter { callback, list, ty } => ConcreteHostExprKind::Filter {
            callback: resolve_host_callback(callback)?,
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        } => ConcreteHostExprKind::Fold {
            callback: resolve_host_callback(callback)?,
            init: Box::new(resolve_host_expr(*init)?),
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        } => ConcreteHostExprKind::Scan {
            callback: resolve_host_callback(callback)?,
            init: Box::new(resolve_host_expr(*init)?),
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::Partition { callback, list, ty } => ConcreteHostExprKind::Partition {
            callback: resolve_host_callback(callback)?,
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::FlatMap { callback, list, ty } => ConcreteHostExprKind::FlatMap {
            callback: resolve_host_callback(callback)?,
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::WithSeed { seed, body, ty } => ConcreteHostExprKind::WithSeed {
            seed: Box::new(resolve_host_expr(*seed)?),
            body: Box::new(resolve_host_expr(*body)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::TensorCall { helper, args, ty } => ConcreteHostExprKind::TensorCall {
            helper,
            args: args
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty: ty.into_concrete()?,
        },
        HostExprKind::Unit => ConcreteHostExprKind::Unit,
    };
    Ok(ConcreteHostExpr {
        kind,
        span_id: expr.span_id,
        merged_spans: expr.merged_spans,
    })
}

/// Lower a checked program through the host-type resolution boundary.
///
/// The result stays fallible because valid checked programs may retain
/// unresolved host terms (for example a context-free empty list), and because
/// legacy lowering invariants may still unwind internally. This public edge
/// converts both classes into `LowerDiagnostic`; there is deliberately no
/// infallible wrapper that can turn a user-reachable rejection into a panic.
///
/// This also catches lowering panics (e.g. WS-A5 RT-3a F2: an unresolved
/// tensor precision tvar
/// reaching the `Type::Tensor` -> `HostTypeTerm::Tensor` boundary, which
/// `try_extract_tensor_type` panics on per spec/04-type-system.md
/// \u{00a7}5.8.1) and returns them as `LowerDiagnostic` so the build
/// surfaces can return a clean structured error rather than a thread panic.
pub fn try_lower_compiled_program(
    program: &CheckedProgram,
) -> Result<CompiledProgram, crate::lower::LowerDiagnostic> {
    try_lower_compiled_program_with_lane_overrides(program, None, false, false)
        .map(|(compiled, _)| compiled)
}

/// Lower through the realizability phase boundary carried by a manifested
/// program. Observable definitions use the manifest's target-specific lane;
/// non-root helper definitions retain the legacy classifier until the
/// manifest grows an all-def lane table.
pub fn try_lower_manifested_program(
    program: &chelis_types::manifest::ManifestedProgram,
) -> Result<CompiledProgram, crate::lower::LowerDiagnostic> {
    try_lower_compiled_program_with_lane_overrides(
        program.checked(),
        Some(program.manifest()),
        false,
        false,
    )
    .map(|(compiled, _)| compiled)
}

/// Lower a checked program using the supplied root manifest as the authority
/// for observable-definition lanes. This is the CLI bridge, whose pipeline
/// owns the checked program and manifest as separate values.
pub fn try_lower_compiled_program_with_manifest(
    program: &CheckedProgram,
    manifest: &chelis_types::manifest::RootManifest,
) -> Result<CompiledProgram, crate::lower::LowerDiagnostic> {
    try_lower_compiled_program_with_lane_overrides(program, Some(manifest), false, false)
        .map(|(compiled, _)| compiled)
}

/// Lower a manifested program while retaining source execution authority for
/// fixed-control C tensor helpers. Ordinary host lowering remains unchanged.
pub fn try_lower_manifested_execution_program(
    program: &chelis_types::manifest::ManifestedProgram,
) -> Result<(Option<crate::Dag>, Option<HostExecutionPlan>), crate::lower::LowerDiagnostic> {
    lower_execution_program(program.checked(), Some(program.manifest()), false)
}

#[cfg(feature = "lowering-trace")]
pub fn try_lower_manifested_execution_program_with_trace(
    program: &chelis_types::manifest::ManifestedProgram,
) -> Result<(Option<crate::Dag>, Option<HostExecutionPlan>), crate::lower::LowerDiagnostic> {
    lower_execution_program(program.checked(), Some(program.manifest()), true)
}

/// CLI counterpart of [`try_lower_manifested_execution_program`].
pub fn try_lower_execution_program_with_manifest(
    program: &CheckedProgram,
    manifest: &chelis_types::manifest::RootManifest,
) -> Result<(Option<crate::Dag>, Option<HostExecutionPlan>), crate::lower::LowerDiagnostic> {
    lower_execution_program(program, Some(manifest), false)
}

#[cfg(feature = "lowering-trace")]
pub fn try_lower_execution_program_with_manifest_and_trace(
    program: &CheckedProgram,
    manifest: &chelis_types::manifest::RootManifest,
) -> Result<(Option<crate::Dag>, Option<HostExecutionPlan>), crate::lower::LowerDiagnostic> {
    lower_execution_program(program, Some(manifest), true)
}

fn lower_execution_program(
    program: &CheckedProgram,
    manifest: Option<&chelis_types::manifest::RootManifest>,
    collect_trace: bool,
) -> Result<(Option<crate::Dag>, Option<HostExecutionPlan>), crate::lower::LowerDiagnostic> {
    let (compiled, execution) =
        try_lower_compiled_program_with_lane_overrides(program, manifest, true, collect_trace)?;
    Ok((
        compiled.dag,
        compiled.host.map(|host| HostExecutionPlan {
            program: host,
            global: execution.global,
            functions: execution.functions,
        }),
    ))
}

#[derive(Default)]
struct HostExecutionMetadata {
    global: Vec<HelperExecutionProduct>,
    functions: Vec<Vec<HelperExecutionProduct>>,
}

fn try_lower_compiled_program_with_lane_overrides(
    program: &CheckedProgram,
    manifest: Option<&chelis_types::manifest::RootManifest>,
    collect_execution: bool,
    collect_trace: bool,
) -> Result<(CompiledProgram, HostExecutionMetadata), crate::lower::LowerDiagnostic> {
    // The whole-program lowering owns its session. Every function below reads
    // the program through it, so there is no entry to this lowering that can
    // forget to establish one (chelis#1835).
    let session = HostLoweringSession::new(program);
    let program = &session;
    let mut lowered_names = top_level_lowering_map(program.exprs(), program.type_env());
    if let Some(manifest) = manifest {
        for entry in &manifest.entries {
            lowered_names.insert(
                entry.def_name.clone(),
                entry.lane == chelis_types::types::Lane::Tensor,
            );
        }
    }
    // Issue #197: a *fatal* diagnostic from IR lowering (e.g. the AD
    // pass refused to differentiate a non-differentiable op) must
    // propagate to the user. Falling through to the host path here
    // emits a call to an undefined symbol (the unlowered grad
    // function) which compiles cleanly via the `chelis build` step
    // and then fails opaquely at gcc-link time. Non-fatal
    // diagnostics (the historical un-representable cases) keep the
    // original fallback semantics so host-only programs still build.
    let dag = match crate::lower::try_lower_program(program) {
        Ok(dag) => Some(dag),
        Err(diagnostic) if diagnostic.fatal => return Err(diagnostic),
        Err(_) => None,
    };
    let (host_terms, execution) = crate::lower::catch_lowering_external(|| {
        lower_host_program_with_execution(program, &lowered_names, collect_execution, collect_trace)
    })??;
    let host = resolve_host_program(host_terms).map_err(|error| {
        crate::lower::LowerDiagnostic::new(
            format!(
                "host type did not resolve before the code-generation boundary: {error} \
                 ([05-UNS-1]; chelis#730)"
            ),
            None,
            None,
        )
        .fatal()
    })?;

    Ok((
        CompiledProgram {
            dag: dag.filter(|dag| !dag.roots().is_empty()),
            host: if host.globals.is_empty() && host.functions.is_empty() {
                None
            } else {
                Some(host)
            },
        },
        execution,
    ))
}

pub fn host_program_requires_host_backend(program: &ConcreteHostProgram) -> bool {
    if !program.globals.is_empty() {
        return true;
    }

    let tensor_only_functions = program
        .functions
        .iter()
        .filter(|function| {
            matches!(function.ret_ty, ConcreteHostType::Tensor(_))
                && function
                    .params
                    .iter()
                    .all(|param| matches!(param.ty, ConcreteHostType::Tensor(_)))
        })
        .map(|function| function.name.clone())
        .collect::<UnordSet<_>>();

    program.functions.iter().any(|function| {
        !tensor_only_functions.contains(&function.name)
            || !host_expr_stays_on_tensor_path(&function.body, &tensor_only_functions)
    })
}

fn host_expr_stays_on_tensor_path(
    expr: &ConcreteHostExpr,
    tensor_only_functions: &UnordSet<String>,
) -> bool {
    match &expr.kind {
        ConcreteHostExprKind::Var(_, ConcreteHostType::Tensor(_)) => true,
        ConcreteHostExprKind::TensorCall { .. } => true,
        ConcreteHostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        } => {
            matches!(ty, ConcreteHostType::Tensor(_))
                && tensor_only_functions.contains(function)
                && arg_tys
                    .iter()
                    .all(|ty| matches!(ty, ConcreteHostType::Tensor(_)))
                && args
                    .iter()
                    .all(|arg| host_expr_stays_on_tensor_path(arg, tensor_only_functions))
        }
        ConcreteHostExprKind::Let { bindings, body, ty } => {
            matches!(ty, ConcreteHostType::Tensor(_))
                && bindings.iter().all(|binding| {
                    matches!(binding.ty, ConcreteHostType::Tensor(_))
                        && host_expr_stays_on_tensor_path(&binding.value, tensor_only_functions)
                })
                && host_expr_stays_on_tensor_path(body, tensor_only_functions)
        }
        _ => false,
    }
}

pub fn preferred_tensor_entry_name(program: &ConcreteHostProgram) -> Option<&str> {
    fn tensor_signature(function: &ConcreteHostFunction) -> bool {
        matches!(function.ret_ty, ConcreteHostType::Tensor(_))
            && function
                .params
                .iter()
                .all(|param| matches!(param.ty, ConcreteHostType::Tensor(_)))
    }

    if let Some(function) = program
        .functions
        .iter()
        .find(|function| function.name == "main" && tensor_signature(function))
    {
        return Some(function.name.as_str());
    }

    // A monomorphized specialization is compiler-internal and must never
    // displace the authored entry, however tensor-shaped its signature
    // (harden-bounded-monomorphization D2).
    program
        .functions
        .iter()
        .rev()
        .find(|function| !function.is_monomorphized_specialization() && tensor_signature(function))
        .map(|function| function.name.as_str())
}

type NamedTensorEntryLoweringInputs = (
    Expr,
    Vec<(String, TensorType)>,
    Option<TensorType>,
    Arc<BTreeMap<String, Expr>>,
);

fn named_tensor_entry_lowering_inputs(
    program: &HostLoweringSession<'_>,
    name: &str,
) -> Option<NamedTensorEntryLoweringInputs> {
    let defs = cached_program_defs(program);
    let body = lookup_program_def(&defs, name)?.clone();
    let Expr::List(list, _) = &body else {
        return None;
    };
    if tag(list) != Some(DeepTag::Fn) {
        return None;
    }

    let kids = children(list);
    let params_list = kids.first().and_then(as_list)?;
    if tag(params_list) != Some(DeepTag::Params) {
        return None;
    }

    let declared_param_tys = lookup_declared_type_expr(program, name)
        .and_then(|ty| parse_expanded_fn_type_expr(program, &ty))
        .map(|(params, _)| params)
        .unwrap_or_default();
    let mut scope = Vec::new();
    for (index, param) in children(params_list).iter().enumerate() {
        let pname = param_name(param)?;
        let pty = declared_param_tys
            .get(index)
            .cloned()
            .or_else(|| param_host_type(param).map(|ty| expand_host_type_aliases(program, ty)))
            .filter(|ty| !ty.is_unresolved())?;
        let tensor_ty = tensor_type_from_host_input(&pty)?;
        scope.push((pname, tensor_ty));
    }

    let body_expr = kids.get(1)?.clone();
    let result_claim = lookup_declared_type_expr(program, name)
        .and_then(|ty| parse_expanded_fn_type_expr(program, &ty))
        .and_then(|(_, result)| tensor_type_from_host_input(&result));
    Some((body_expr, scope, result_claim, defs))
}

pub fn lower_named_tensor_entry_dag(program: &CheckedProgram, name: &str) -> Option<crate::Dag> {
    // A whole-program entry owns its session, so nothing outside this crate
    // has to know one exists (chelis#1835).
    let session = HostLoweringSession::new(program);
    let program = &session;
    let (body_expr, scope, result_claim, defs) = named_tensor_entry_lowering_inputs(program, name)?;
    // Issue #197: a fatal lowering diagnostic (AD-rejection) must
    // propagate as a panic so the outer `catch_lowering_external`
    // surfaces it to the user. Silently absorbing it with `.ok()`
    // would let the host fallback emit an undefined-symbol call to
    // the grad function.
    let context = crate::lower::prepare_checked_subexpr_lowering_context(
        program,
        defs.clone(),
        Arc::new(crate::lower::collect_top_level_sigs(program.exprs())),
    );
    match crate::lower::try_lower_subexpr_program_with_ordered_inputs(
        &body_expr,
        scope,
        &context,
        result_claim.as_ref(),
        None,
        0,
        // `scope` is this def's own declared parameter list.
        true,
    ) {
        Ok((dag, _)) => Some(dag),
        Err(diagnostic) if diagnostic.fatal => {
            crate::lower::raise_fatal_lowering_diagnostic(diagnostic)
        }
        Err(_) => None,
    }
}

/// Lower one named tensor entry with source-owned fixed-control execution.
///
/// `Ok(None)` means the entry has no Dropout and the legacy DAG path remains
/// authoritative. A Dropout entry is either returned as one sealed unfused
/// plan or rejected; callers must not retry it through raw DAG lowering.
pub fn lower_named_tensor_entry_execution_plan(
    program: &CheckedProgram,
    name: &str,
) -> Result<Option<crate::evaluation::EvaluationPlan>, crate::lower::LowerDiagnostic> {
    // A whole-program entry owns its session, so nothing outside this crate
    // has to know one exists (chelis#1835).
    let session = HostLoweringSession::new(program);
    let program = &session;
    lower_named_tensor_entry_execution_with(
        program,
        name,
        crate::lower::try_lower_subexpr_c_execution_with_ordered_inputs,
    )
}

#[cfg(feature = "lowering-trace")]
pub fn lower_named_tensor_entry_execution_plan_with_trace(
    program: &CheckedProgram,
    name: &str,
) -> Result<
    Option<(
        crate::evaluation::EvaluationPlan,
        crate::lowering_trace::HelperLoweringTrace,
    )>,
    crate::lower::LowerDiagnostic,
> {
    let session = HostLoweringSession::new(program);
    lower_named_tensor_entry_execution_with(
        &session,
        name,
        crate::lower::try_lower_subexpr_c_execution_with_ordered_inputs_and_trace,
    )
}

// Observation must not duplicate the admission policy, source scope or initial
// Random frame. Only the actual lowerer's optional return product differs.
fn lower_named_tensor_entry_execution_with<T>(
    program: &HostLoweringSession<'_>,
    name: &str,
    lower: impl FnOnce(
        &Expr,
        Vec<(String, TensorType)>,
        &crate::lower::SubexprLoweringContext,
        Option<&TensorType>,
        bool,
        &crate::evaluation::RandomExecutionContext,
    ) -> Result<T, crate::lower::LowerDiagnostic>,
) -> Result<Option<T>, crate::lower::LowerDiagnostic> {
    let Some((body_expr, scope, result_claim, defs)) =
        named_tensor_entry_lowering_inputs(program, name)
    else {
        return Ok(None);
    };
    let context = crate::lower::prepare_checked_subexpr_lowering_context(
        program,
        defs,
        Arc::new(crate::lower::collect_top_level_sigs(program.exprs())),
    );
    let profile = context.c_execution_profile(&body_expr, &scope);
    match profile {
        crate::evaluation::EvaluationProfile::Legacy(
            crate::evaluation::LegacyEvaluationReason::NoDropout,
        ) => return Ok(None),
        crate::evaluation::EvaluationProfile::Legacy(reason) => {
            return Err(crate::lower::LowerDiagnostic::new(
                format!(
                    "tensor entry `{name}` has Dropout execution that is not fixed-control: {reason:?}"
                ),
                None,
                None,
            )
            .fatal());
        }
        crate::evaluation::EvaluationProfile::FixedControl => {}
    }
    let planning = crate::evaluation::RandomExecutionContext::new(RandomLoweringState {
        seed: None,
        counter: 0,
    });
    lower(
        &body_expr,
        scope,
        &context,
        result_claim.as_ref(),
        true,
        &planning,
    )
    .map(Some)
}

/// Does the named top-level def have a pure tensor signature — every
/// parameter a tensor and a tensor result?
///
/// This is the per-*entry* analogue of the tensor-signature predicate
/// [`preferred_tensor_entry_name`] uses. The entry-scoped compiled
/// metadata lane (#817/#818) only claims tensor-signature entries; a
/// scalar/record/ADT-returning def selected by name must stay on the
/// host lane.
pub fn function_has_tensor_signature(program: &ConcreteHostProgram, name: &str) -> bool {
    program.functions.iter().any(|function| {
        function.name == name
            && !function.is_monomorphized_specialization()
            && matches!(function.ret_ty, ConcreteHostType::Tensor(_))
            && function
                .params
                .iter()
                .all(|param| matches!(param.ty, ConcreteHostType::Tensor(_)))
    })
}

/// Does the named entry def's body use a `grad`/`vmap`/`vmap-grad` form?
///
/// Such an entry MUST stay on the host lane even though it is
/// tensor-signature and [`lower_named_tensor_entry_dag`] *can* produce a
/// DAG for it: the host lane owns the multi-root grad-tuple emission
/// (assembling a real `chelis_tuple` from per-`wrt` gradient outputs, see
/// #309), which the single-root entry-scoped kernel path does not.
///
/// This is deliberately narrower than "the body needs the host runtime":
/// a DAG-lowerable host-runtime builtin such as `concat` lowers cleanly
/// through `lower_named_tensor_entry_dag` (that IS the #818 fix), so it is
/// NOT excluded here. Only genuinely host-lane-owned forms are.
pub fn named_entry_uses_grad_like(program: &CheckedProgram, name: &str) -> bool {
    // A whole-program entry owns its session, so nothing outside this crate
    // has to know one exists (chelis#1835).
    let session = HostLoweringSession::new(program);
    let program = &session;
    let defs = cached_program_defs(program);
    match lookup_program_def(&defs, name) {
        Some(body) => expr_contains_grad_like(body),
        None => false,
    }
}

/// Does the checked program bind any top-level VALUE binding — a
/// `(def name body)` whose body is not a `fn`, i.e. a global like
/// `glb = 2.0` or `total = add(...)`?
///
/// This is the SOURCE-LEVEL predicate, deliberately distinct from
/// checking the lowered `HostProgram::globals`: `lower_host_program`
/// drops a top-level value binding from `host.globals` when it is
/// DAG-lowerable, uncaptured by any `fn` def, and the program emits no
/// host `main` (`skip_for_lowered` in `lower_host_program` below), so a
/// lowered-artifact check
/// misses exactly those bindings. The compiled-execution entry lane
/// (#819) keys its `HasGlobals` decline on this predicate because its
/// standalone entry lowering would silently drop such a binding's
/// computation. Top-level type signatures are separate items (not
/// `def`-tagged) and do not count; a `def` with an `fn` body is a
/// function definition, not a value binding.
pub fn program_has_top_level_value_bindings(program: &CheckedProgram) -> bool {
    top_level_items(program.exprs()).iter().any(|expr| {
        let children = match expr.carrier() {
            ExprCarrier::DecodedNode(DeepTag::Def, _, children) => children,
            ExprCarrier::DecodedNode(_, _, _)
            | ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_)
            | ExprCarrier::MalformedLegacyList(_) => return false,
        };
        !matches!(
            children.get(1).map(Expr::carrier),
            Some(ExprCarrier::DecodedNode(DeepTag::Fn, _, _))
        )
    })
}

#[cfg(test)]
fn lower_host_program(
    program: &HostLoweringSession<'_>,
    lowered_names: &BTreeMap<String, bool>,
) -> Result<HostProgram, crate::lower::LowerDiagnostic> {
    lower_host_program_with_execution(program, lowered_names, false, false).map(|(host, _)| host)
}

fn lower_host_program_with_execution(
    program: &HostLoweringSession<'_>,
    lowered_names: &BTreeMap<String, bool>,
    collect_execution: bool,
    collect_trace: bool,
) -> Result<(HostProgram, HostExecutionMetadata), crate::lower::LowerDiagnostic> {
    // chelis#1158: specialization state is per-invocation; a fresh program
    // lowering must rebuild every specialization (and must not inherit a
    // failed run's partial memo).
    MONO_SPECIALIZATIONS.with(|state| *state.borrow_mut() = MonoSpecializationState::default());
    let mut host = HostProgram::default();
    let mut global_tensor_helpers = TensorHelperSink::new(collect_execution, collect_trace);
    let mut function_execution = Vec::new();
    let mut global_scope = UnordMap::new();
    // Count pure-tensor `fn`-body top-level defs in the program. When there
    // is more than one, the legacy DAG-only path would collapse them into a
    // single file-named entry point that drops all but one def's parameters
    // (Nautilus Bug 3c). In that case we emit a host wrapper per def so each
    // gets its own C symbol.
    let lowered_fn_def_count = top_level_items(program.exprs())
        .iter()
        .filter(|expr| {
            let Expr::List(list, _) = expr else {
                return false;
            };
            if tag(list) != Some(DeepTag::Def) {
                return false;
            }
            let kids = children(list);
            let Some(def_name) = kids.first().and_then(symbol_name) else {
                return false;
            };
            let is_fn_body = matches!(kids.get(1), Some(Expr::List(body_list, _)) if tag(body_list) == Some(DeepTag::Fn));
            is_fn_body && lowered_names.get(def_name).copied().unwrap_or(false)
        })
        .count();
    // Issue #378: names referenced by any `fn`-body top-level def. A
    // captured top-level *scalar* (or other value) binding is classified
    // `lowered` by `def_is_lowered` (a `(lit ...)` body materializes into
    // the DAG `main()`), so `skip_for_lowered` would drop it from
    // `host.globals` entirely — but a host-lane function that captures it
    // then emits `__arg = c;` against an undeclared `c`. The #376 hoist
    // already serves *tensor* captures because tensor bindings reach
    // `host.globals`; this set lets a captured value binding stay in
    // `host.globals` too (the C emitter's `captured_global_names` only
    // declares globals a function actually references, so a non-captured
    // binding still costs nothing in the pure-DAG case).
    let mut names_captured_by_fn_defs: UnordSet<String> = UnordSet::new();
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some(DeepTag::Def) {
            continue;
        }
        let kids = children(list);
        let Some(body) = kids.get(1) else {
            continue;
        };
        if matches!(body, Expr::List(body_list, _) if tag(body_list) == Some(DeepTag::Fn)) {
            collect_deep_var_names(body, &mut names_captured_by_fn_defs);
        }
    }
    // Issue #750: does this program emit a host `main()`? A non-`fn`
    // top-level binding reaches `host.globals` today iff `skip_for_lowered`
    // does NOT drop it, i.e. it is non-lowered (needs host runtime, e.g. a
    // `print`-effecting `shown = print(x)`) or it is a #378 captured value
    // binding. `emit_main` runs exactly when `host.globals` is non-empty
    // (`host_emit.rs`), so the presence of any such binding is the precise,
    // rescue-independent predicate for "this program emits a host `main`".
    // The #750 rescue below is gated on this so it only ever RE-ATTACHES a
    // labeled root to a `main` that already exists — it never flips a
    // pure-DAG program (which emits a callable kernel with `outputs[]`, no
    // `main`) onto the host lane.
    let program_emits_host_main = top_level_items(program.exprs()).iter().any(|expr| {
        let Expr::List(list, _) = expr else {
            return false;
        };
        if tag(list) != Some(DeepTag::Def) {
            return false;
        }
        let kids = children(list);
        let Some(binding_name) = kids.first().and_then(symbol_name) else {
            return false;
        };
        let is_fn_body =
            matches!(kids.get(1), Some(Expr::List(body_list, _)) if tag(body_list) == Some(DeepTag::Fn));
        !is_fn_body
            && (!lowered_names.get(binding_name).copied().unwrap_or(false)
                || names_captured_by_fn_defs.contains(binding_name))
    });
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some(DeepTag::Def) {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        let ty_expr = lookup_declared_type_expr(program, name);
        // WS-A8: skip polymorphic-precision sigs at host emission time.
        // A `t-fn` whose tensor types carry `(t-var {} _)` precision
        // slots has no concrete monomorphization on its own; reaching
        // `parse_host_type` for its parameters trips the F2 backend
        // tripwire panic per spec/04-type-system.md §5.8.1. Such a def
        // is reachable from concrete client code via call-site inlining
        // (the inliner threads the call site's concrete precision into
        // the body); the standalone host symbol is intentionally
        // omitted because no caller can use it without supplying the
        // monomorphization binding the inliner provides.
        if ty_expr
            .as_ref()
            .is_some_and(crate::lower::type_expr_has_precision_var)
        {
            continue;
        }
        // Tier-2 rank polymorphism (spec/design/rank_polymorphism.md): skip
        // rank-polymorphic sigs at host emission time, the exact analogue of
        // the precision-var skip above. A `t-fn` whose tensor types carry a
        // sole `(d-rank {} r)` rank slot has no standalone monomorphization;
        // reaching `parse_host_type` for its parameters would trip the
        // rank-poly lowering tripwire (a surviving `Dim::Rank` is a
        // monomorphization bug, not a backend input). Such a def is reachable
        // from concrete client code via call-site inlining (the inliner
        // threads the call site's concrete shape into the body — see
        // `tensor_rank_substitutions` in `lower.rs`); the standalone host
        // symbol is intentionally omitted because no caller can use it
        // without supplying the monomorphization binding the inliner provides.
        if ty_expr
            .as_ref()
            .is_some_and(crate::lower::type_expr_has_rank_var)
            || top_level_fn_is_nested_rank_polymorphic(program, name)
        {
            continue;
        }
        // chelis#935: ordinary type-polymorphic host functions likewise
        // have no standalone concrete ABI. They are specialized by
        // call-site inlining below, where the checked application's result
        // type supplies the concrete ADT arguments. Emitting the generic
        // body here would force a `Box[a]` (or sibling generic term) across
        // the concrete host boundary before any call site can bind `a`.
        if top_level_fn_is_type_polymorphic(program, name) {
            continue;
        }
        // Pure-tensor top-level function defs are normally lowered to the
        // DAG. But when the program also has host-lane bindings (i.e. some
        // def is NOT DAG-lowerable), downstream host-lane callers still
        // need a real C function symbol for the wrapper. In that case,
        // emit a HostFunction wrapper alongside the DAG lowering.
        //
        // A single pure-tensor function def can still use the legacy
        // DAG-only path (the file-named entry point wraps it 1:1 with the
        // correct signature). But when there are multiple pure-tensor
        // function defs in the same program, the DAG path would collapse
        // them into a single file-named entry that silently drops all but
        // one def's parameters and outputs (Nautilus Bug 3c). Emit a host
        // wrapper per def in that case so each gets its own C symbol.
        let is_fn_body = matches!(body, Expr::List(list, _) if tag(list) == Some(DeepTag::Fn));
        let has_any_host_lane_def = lowered_names.values().any(|lowered| !*lowered);
        let has_callable_params = lookup_declared_fn_type(program, name)
            .is_some_and(|(params, _)| params.iter().any(|ty| matches!(ty, HostTypeTerm::Fn(..))));
        // Non-F32/Bool tensor precisions (e.g. i32, i64) aren't
        // representable in the Phase 0f DAG-only codegen path — it still
        // hard-asserts f32/bool. Force a host-lane wrapper for any fn whose
        // signature carries such a tensor so the program stays on the
        // host-lane code path instead of panicking in DAG emit.
        let has_non_dag_tensor =
            lookup_declared_fn_type(program, name).is_some_and(|(params, ret)| {
                fn ty_has_non_dag_tensor(ty: &HostTypeTerm) -> bool {
                    match ty {
                        HostTypeTerm::Tensor(tensor) => !matches!(
                            tensor.precision,
                            chelis_types::types::Prim::F32 | chelis_types::types::Prim::Bool
                        ),
                        HostTypeTerm::Option(inner) | HostTypeTerm::List(inner) => {
                            ty_has_non_dag_tensor(inner)
                        }
                        HostTypeTerm::Tuple(items) => items.iter().any(ty_has_non_dag_tensor),
                        HostTypeTerm::Dict(k, v) => {
                            ty_has_non_dag_tensor(k) || ty_has_non_dag_tensor(v)
                        }
                        HostTypeTerm::Fn(params, ret) => {
                            params.iter().any(ty_has_non_dag_tensor) || ty_has_non_dag_tensor(ret)
                        }
                        _ => false,
                    }
                }
                params.iter().any(ty_has_non_dag_tensor) || ty_has_non_dag_tensor(&ret)
            });
        // `has_callable_params` blocks the wrapper for fn-taking-fn signatures
        // because the host emitter doesn't lower higher-order wrappers
        // cleanly in every shape (grad specialization etc.). But if the
        // signature ALSO contains a non-F32/Bool tensor, the DAG-only
        // fallback panics — so in that combination still force the wrapper.
        //
        // Bucket 4d: a higher-order signature whose non-callable params and
        // return type are *scalars* (e.g. `(model: f32 -> f32, x: f32) -> f32`)
        // can never be DAG-lowered (the DAG-only path is tensor-only) and
        // the host emitter handles `model(x)` cleanly because every value
        // is a plain C scalar. Force the host wrapper for those signatures
        // so they actually get a definition emitted -- the previous logic
        // was silently dropping them, leaving `gcc` to fail with
        // `implicit declaration of function 'apply'` on the caller side.
        //
        // Known limitation: when a caller references a callable-param fn by
        // name with a tensor-typed shape (e.g. `(model: tensor[n, f32] ->
        // f32, ...)`) and the body doesn't lower cleanly through the host
        // wrapper, the def is still dropped from emission. Tracked as a
        // residual HOF emission issue (red-team A20.1).
        let scalar_only_callable_signature = has_callable_params
            && lookup_declared_fn_type(program, name).is_some_and(|(params, ret)| {
                fn ty_is_scalar_or_callable_scalar(ty: &HostTypeTerm) -> bool {
                    match ty {
                        HostTypeTerm::Tensor(_) => false,
                        HostTypeTerm::Fn(params, ret) => {
                            params.iter().all(ty_is_scalar_or_callable_scalar)
                                && ty_is_scalar_or_callable_scalar(ret)
                        }
                        HostTypeTerm::Option(inner) | HostTypeTerm::List(inner) => {
                            ty_is_scalar_or_callable_scalar(inner)
                        }
                        HostTypeTerm::Tuple(items) => {
                            items.iter().all(ty_is_scalar_or_callable_scalar)
                        }
                        HostTypeTerm::Dict(k, v) => {
                            ty_is_scalar_or_callable_scalar(k) && ty_is_scalar_or_callable_scalar(v)
                        }
                        // Primitive (f32/i32/bool/...), Unit -- scalar OK.
                        _ => true,
                    }
                }
                params.iter().all(ty_is_scalar_or_callable_scalar)
                    && ty_is_scalar_or_callable_scalar(&ret)
            });
        // #1872: a source-fixed tensor entry needs its sealed execution
        // helper even when no sibling happens to select the host lane.
        // Classify the original checked body with its declared input scope;
        // the raw-DAG effect gate remains authoritative for ordinary lowering.
        // FixedControl can inherit Random. Only an inferred closed body may
        // introduce this standalone wrapper, whose public ABI has no RNG frame.
        let has_fixed_execution = collect_execution
            && is_fn_body
            && !has_callable_params
            && cached_def_effect_rows(program)
                .get(name)
                .is_some_and(|row| !row.contains(&chelis_types::types::Effect::Random))
            && named_tensor_entry_lowering_inputs(program, name).is_some_and(
                |(body, scope, _, _)| {
                    cached_subexpr_lowering_context(program).c_execution_profile(&body, &scope)
                        == crate::evaluation::EvaluationProfile::FixedControl
                },
            );
        let needs_host_wrapper = is_fn_body
            && (has_non_dag_tensor
                || scalar_only_callable_signature
                || has_fixed_execution
                || (!has_callable_params && (has_any_host_lane_def || lowered_fn_def_count > 1)));
        // Issue #378: a non-`fn` value binding (a `(def name (lit ...))`)
        // that a host-lane function captures must reach `host.globals` so
        // the emitted C declares it; otherwise the function body references
        // an undeclared identifier. Do not skip it even though the DAG lane
        // also claims it (the DAG lane inlines its own copy for tensor
        // roots; the global is emitted only when a function references it).
        let captured_value_binding = !is_fn_body && names_captured_by_fn_defs.contains(name);
        // Issue #750: a DAG-lowered non-`fn` value binding (e.g.
        // `troot = mk()`, a def-call-valued root) that a program's host
        // `main` should print is otherwise dropped by `skip_for_lowered`,
        // because labeled roots are emitted ONLY from `host.globals` and
        // there is no DAG-root -> labeled-print bridge. Rescue it into
        // `host.globals` exactly like the #378 captured-value carve-out so
        // the CLI display-name pass stamps it and `emit_main` renders it,
        // byte-identical to a direct-construction root of the same value.
        // Gated on `program_emits_host_main` so the rescue only re-attaches
        // a root to a `main` the program ALREADY emits — a pure-DAG program
        // keeps emitting a kernel with `outputs[]` and no `main`.
        let display_root_binding = !is_fn_body && program_emits_host_main;
        let skip_for_lowered = lowered_names.get(name).copied().unwrap_or(false)
            && !needs_host_wrapper
            && !captured_value_binding
            && !display_root_binding;
        // A fused `vmap(grad(...))` template with a callable parameter has
        // no standalone C ABI: the tensor entry cannot carry the function
        // argument, while the host lane cannot execute the transform as a
        // first-class value. Its concrete call sites are specialized by the
        // whole-program lowerer, so omit only this unusable generic wrapper.
        let call_site_only_vmap_grad =
            is_fn_body && has_callable_params && expr_contains_vmap_grad(body);
        if skip_for_lowered || call_site_only_vmap_grad {
            continue;
        }
        // The wrapper emitter doesn't know every pattern the DAG-path
        // specializer does (e.g. `grad(local_fn)(theta)`). When the host
        // lowering would degrade to a fallback `Builtin { name: "call" }`,
        // emitting the wrapper produces broken C (`__result = call(...)`).
        // In that case prefer the DAG path if it's available (lowered_names
        // says so); otherwise we have no good lowering and must drop the
        // def — at least the caller will get `implicit declaration` rather
        // than `call(…)` undefined-symbol.
        if let Some(lowered) = lower_host_function(
            name,
            body,
            ty_expr.as_ref(),
            program,
            collect_execution,
            collect_trace,
        )? {
            let mut function = lowered.function;
            // N→1 lowering collapse per `spec/design/chelis_span_survival.md`
            // §2.3 host-side table, rule "Lowering. Top-level def collapses
            // to fn body": when a `(def {span: a} name (fn ... body))` lowers
            // to a HostFunction whose `body` is the lowered fn body, the
            // def's `span_id` appends to the body node's `merged_spans` so
            // the def's source region surfaces in the audit chain. (The fn
            // node's own span, if any, is already on the body via the
            // body-collapse rule applied by `lower_host_expr`.)
            function.body.append_merged_span(expr.span_id());
            global_scope.insert(
                name.to_string(),
                HostTypeTerm::Fn(
                    function
                        .params
                        .iter()
                        .map(|param| param.ty.clone())
                        .collect(),
                    Box::new(function.ret_ty.clone()),
                ),
            );
            host.functions.push(function);
            function_execution.push(lowered.products);
        } else {
            // Inline any local callable bindings in the global binding's
            // body so `let g = grad(f); g(x)` rewrites to `(grad(f))(x)`
            // before host lowering — same rationale as in
            // `lower_host_function`.
            global_tensor_helpers.declaration_name = Some(name.to_string());
            let inlined_body = inline_local_callable_lets(body);
            let mut value = lower_host_expr(
                &inlined_body,
                program,
                &global_scope,
                &mut global_tensor_helpers,
            )?;
            // N→1 lowering collapse per `spec/design/chelis_span_survival.md`
            // §2.3 host-side table, rule "Lowering. Top-level def collapses
            // to body": when a `(def {span: a} name body)` lowers to a
            // HostBinding whose `value` is the body's HostExpr, the def's
            // own `span_id` (sourced from the def's `meta["span"]`) appends
            // to the value node's `merged_spans` so the def's source region
            // doesn't drop out of the audit chain.
            value.append_merged_span(expr.span_id());
            let inferred_ty = host_expr_type(&value);
            // RT-4 F1: respect the surface-level type annotation on a
            // global binding. Without this the type-checker-validated
            // declaration `x: tensor[3, f64] = [1.0, 2.0, 3.0]` was
            // silently lowered with the inferred f32 type because the
            // `to_tensor` builtin's return type defaults to f32 for any
            // float-tagged list. The downstream host emitter uses this
            // type to dispatch the typed runtime call so the storage
            // matches the declared dtype.
            let declared_ty = ty_expr
                .as_ref()
                .map(|ty| decode_host_type_or_raise(ty, &UnordMap::new()))
                .map(|ty| expand_host_type_aliases(program, ty))
                .filter(|ty| !ty.is_unresolved());
            // chelis#1137: movement-helper lowering may prove a different
            // rank from the checker's provisional type (notably `expand`
            // followed by `shrink`).  Retagging the helper call with that
            // stale rank makes the emitted helper ABI disagree with the DAG
            // it invokes.  A helper's actual root rank is the lowering
            // authority; checked context may still sharpen precision and
            // extents when the ranks agree.
            let helper_rank_disagrees = matches!(value.kind, HostExprKind::TensorCall { .. })
                && matches!(
                    (&declared_ty, &inferred_ty),
                    (
                        Some(HostTypeTerm::Tensor(declared)),
                        HostTypeTerm::Tensor(inferred)
                    ) if declared.dims.len() != inferred.dims.len()
                );
            let ty = if helper_rank_disagrees {
                inferred_ty.clone()
            } else {
                declared_ty.clone().unwrap_or_else(|| inferred_ty.clone())
            };
            // When the declared type sharpens the inferred type (e.g.
            // declared f64, inferred f32), retag the outermost value
            // type so downstream host emit sees the right precision
            // for the typed to_tensor runtime call. This is a
            // structural retag; the type checker has already validated
            // that the literal assignment is sound.
            if !helper_rank_disagrees
                && let Some(declared) = declared_ty.as_ref()
                && declared != &inferred_ty
            {
                value = force_host_expr_type(value, declared.clone());
            }
            host.globals.push(HostBinding {
                name: name.to_string(),
                display_name: None,
                display_roots: Vec::new(),
                ty: ty.clone(),
                value: value.clone(),
            });
            global_scope.insert(name.to_string(), ty);
        }
    }
    // chelis#1158: append the monomorphized specializations produced while
    // lowering the program's functions and globals. They join before the
    // refinement fixpoint below so signature refinement and type
    // conformance treat them like any other definition, and the emitted C
    // therefore contains no reference to an omitted generic definition.
    let monomorphized =
        MONO_SPECIALIZATIONS.with(|state| std::mem::take(&mut state.borrow_mut().functions));
    for lowered in monomorphized {
        host.functions.push(lowered.function);
        function_execution.push(lowered.products);
    }
    loop {
        let mut changed = false;
        changed |= refine_host_function_signatures(&mut host.functions);
        changed |= refine_host_globals(&mut host.globals, &host.functions);
        changed |= propagate_named_callback_signatures(&mut host.functions, &host.globals);
        if !changed {
            break;
        }
    }
    let (helpers, global) = global_tensor_helpers.into_parts();
    host.global_tensor_helpers = helpers;
    // The checker has already proved every binding/function/callback
    // context.  Materialize those expected types into the host expression
    // tree before resolving terms: the backend must receive one coherent
    // typed program, not re-run contextual refinement or silently choose
    // between a declared type and a coarse local inference.
    conform_host_program_types(&mut host);
    derive_host_function_specializations(&mut host.functions);
    // Also collect rejections from global tensor helpers (top-level
    // expressions like `result = gather(...)` lower into
    // `host.global_tensor_helpers`, not into a function's
    // tensor_helpers). The owning "function" for diagnostic
    // attribution is the synthetic global name; today we report it
    // under the helper's own name since global tensor helpers don't
    // share a binding name directly.
    collect_program_summary_rejections(&mut host);
    Ok((
        host,
        HostExecutionMetadata {
            global,
            functions: function_execution,
        },
    ))
}

/// Aggregate per-function rejections into `HostProgram::summary_rejections`
/// and also fold in any rejections attached to `global_tensor_helpers`
/// (rejections detected on top-level tensor expression bindings).
fn collect_program_summary_rejections(host: &mut HostProgram) {
    host.summary_rejections.clear();
    for function in &host.functions {
        for rejection in &function.summary_rejections {
            host.summary_rejections.push(rejection.clone());
        }
    }
    // Per-global rejections: the global tensor helpers carry partial
    // rejections from `finish_tensor_helper_call`. The HelperPath is
    // synthesized from the helper's own name (e.g.
    // `__global_tensor_helper_0`) when there is no enclosing
    // function. This is sufficient for diagnostic emission today;
    // future work can thread the binding name through.
    for helper in &host.global_tensor_helpers {
        if let Some(partial) = &helper.summary_rejection {
            host.summary_rejections.push(SummaryRejection {
                rejection_class: partial.rejection_class.clone(),
                helper_path: HelperPath::local(&helper.name),
                callsite_span: None,
                helper_body_span: partial.helper_body_span.clone(),
                detail: partial.detail.clone(),
            });
        }
    }
}

/// Public accessor for the structured summary rejections collected
/// during host lowering. CLI diagnostic reporters and downstream
/// tests pattern-match on the returned slice's enum variants and
/// struct fields. The order is deterministic: per-function rejections
/// appear in function-declaration order, followed by per-global
/// rejections in helper-declaration order.
pub fn host_program_summary_rejections<T>(program: &HostProgram<T>) -> &[SummaryRejection] {
    &program.summary_rejections
}

/// The internal marker names `lower_app_host_expr` gives its generic
/// fallback when it cannot resolve the callee. The leading `#` cannot
/// appear in a Surf or Deep identifier, so no user definition can
/// collide with either marker (chelis#841); the C backend's ABI
/// projection rejects both with the frozen `unsupported:` diagnostic
/// before emission.
///
/// Two markers, routed at the fallback site by what the callee was:
/// an AD transform in callee position (`grad(f)(x)` whose transform the
/// host lane could not lower) takes [`HOST_UNRESOLVED_TRANSFORM_MARKER`]
/// and earns the CLI's grad/vmap workaround text; every other
/// unresolved callee takes [`HOST_UNRESOLVED_CALLABLE_MARKER`] and the
/// function-value diagnostic. Keeping the split in the marker itself
/// means a plain callable bug is never misdescribed as an AD failure
/// just because an unrelated transform exists elsewhere in the program.
pub const HOST_UNRESOLVED_CALLABLE_MARKER: &str = "#chelis-unresolved-callable";
pub const HOST_UNRESOLVED_TRANSFORM_MARKER: &str = "#chelis-unresolved-transform";

/// True for exactly the internal unresolved-callee marker names.
pub fn is_host_unresolved_marker(name: &str) -> bool {
    name == HOST_UNRESOLVED_CALLABLE_MARKER || name == HOST_UNRESOLVED_TRANSFORM_MARKER
}

/// Scan a host program for any function or global whose body still
/// contains an unresolved-callee marker (either kind). These sites
/// never reach emission: ABI projection rejects them with the frozen
/// `unsupported:` diagnostic.
pub fn host_program_unresolved_call_sites<T>(program: &HostProgram<T>) -> Vec<String> {
    // Markers appear as a `Builtin` when nothing about the callee
    // resolved, and as a `Call` function when the callee's TYPE resolved
    // but the callee itself did not.
    host_program_call_name_sites(
        program,
        &is_host_unresolved_marker,
        &is_host_unresolved_marker,
    )
}

/// Scan a host program for defs whose bodies carry the transform marker:
/// an AD transform the host lane recognized in callee position but could
/// not lower. The CLI's pre-codegen UX gate names exactly these defs in
/// its grad/vmap workaround text; plain callable markers fall through to
/// ABI projection's function-value diagnostic instead (chelis#730,
/// chelis#841).
pub fn host_program_unresolved_transform_sites<T>(program: &HostProgram<T>) -> Vec<String> {
    host_program_call_name_sites(
        program,
        &|name| name == HOST_UNRESOLVED_TRANSFORM_MARKER,
        &|function| function == HOST_UNRESOLVED_TRANSFORM_MARKER,
    )
}

fn host_program_call_name_sites<T>(
    program: &HostProgram<T>,
    builtin_matches: &dyn Fn(&str) -> bool,
    function_matches: &dyn Fn(&str) -> bool,
) -> Vec<String> {
    let mut out = Vec::new();
    for function in &program.functions {
        if host_body_has_call_matching(&function.body, builtin_matches, function_matches) {
            out.push(function.name.clone());
        }
    }
    for binding in &program.globals {
        if host_body_has_call_matching(&binding.value, builtin_matches, function_matches) {
            out.push(binding.name.clone());
        }
    }
    out
}

/// Builtins available only under `chelis eval` / `chelis test` (the host
/// evaluator) and deliberately absent from every compiled backend:
/// emitting the C codegen catch-all for them would produce a silent wrong
/// value (the chelis#734 class). Both public build entry points -- the CLI
/// build pipeline and `chelis-compiler-api::compile_for_execution` (the
/// chelis-python path) -- reject them via
/// [`find_eval_only_host_builtin`], sharing this one list so the two
/// gates cannot drift (chelis#891 review finding 13).
pub const EVAL_ONLY_HOST_BUILTINS: &[&str] = &[
    "process_run",
    "round_to",
    // Host-lane CSV I/O (chelis#903): the compiler-owned text-table
    // carrier is evaluator-only. Compiled structured I/O lives in the
    // source-defined Std.Io modules instead.
    "parse_csv",
    "to_csv",
    "csv_f64s",
    "csv_ints",
    "csv_strs",
    "csv_nrows",
    "csv_cols",
    "csv_f64",
    "csv_int",
    "csv_str",
];

/// First eval/test-only builtin applied anywhere in the lowered host
/// program, if any (see [`EVAL_ONLY_HOST_BUILTINS`]).
pub fn find_eval_only_host_builtin<T>(program: &HostProgram<T>) -> Option<&'static str> {
    EVAL_ONLY_HOST_BUILTINS
        .iter()
        .copied()
        .find(|builtin| host_program_uses_builtin(program, builtin))
}

/// Returns `true` if any global binding or function body in `program`
/// applies the named builtin. Used by the build backends to reject
/// eval/test-only builtins (e.g. `process_run`, Hull Phase 0a) with a
/// clean diagnostic rather than the silent `/* unsupported builtin */ 0`
/// fallthrough in C codegen.
pub fn host_program_uses_builtin<T>(program: &HostProgram<T>, builtin: &str) -> bool {
    program
        .globals
        .iter()
        .any(|binding| host_body_uses_builtin(&binding.value, builtin))
        || program
            .functions
            .iter()
            .any(|function| host_body_uses_builtin(&function.body, builtin))
}

/// Find a direct application of one of `builtins` in checked Deep before host
/// expression lowering descends into its arguments. Target frontends use this
/// to preserve the owning builtin rejection when an argument (such as an
/// inline callback) deliberately has no standalone compiled representation.
/// The walk covers the whole checked program, including nested callback and
/// otherwise-unreachable function bodies.
pub fn find_direct_builtin_call(program: &CheckedProgram, builtins: &[&str]) -> Option<String> {
    program
        .exprs()
        .iter()
        .find_map(|expr| find_direct_builtin_call_in_expr(expr, builtins))
}

fn find_direct_builtin_call_in_expr(expr: &Expr, builtins: &[&str]) -> Option<String> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, metadata, children) => {
            if tag == DeepTag::App
                && let Some(ExprCarrier::DecodedNode(DeepTag::Var, _, callee_children)) =
                    children.first().map(Expr::carrier)
                && let Some(name) = callee_children.first().and_then(symbol_name)
                && builtins.contains(&name)
            {
                return Some(name.to_string());
            }
            metadata
                .find_expression(|value| find_direct_builtin_call_in_expr(value, builtins))
                .or_else(|| {
                    children
                        .iter()
                        .find_map(|expr| find_direct_builtin_call_in_expr(expr, builtins))
                })
        }
        ExprCarrier::StructuralList(elements) => elements
            .iter()
            .find_map(|expr| find_direct_builtin_call_in_expr(expr, builtins)),
        ExprCarrier::UndecodableHead(_, metadata, children) => metadata
            .find_expression(|value| find_direct_builtin_call_in_expr(value, builtins))
            .or_else(|| {
                children
                    .iter()
                    .find_map(|expr| find_direct_builtin_call_in_expr(expr, builtins))
            }),
        ExprCarrier::MetadataMap(map) => {
            map.find_expression(|value| find_direct_builtin_call_in_expr(value, builtins))
        }
        ExprCarrier::MetadataExpression(meta) => {
            find_direct_builtin_call_in_expr(&meta.expr, builtins).or_else(|| {
                meta.metadata
                    .find_expression(|value| find_direct_builtin_call_in_expr(value, builtins))
            })
        }
        ExprCarrier::MalformedLegacyList(list) => list
            .elements
            .iter()
            .find_map(|expr| find_direct_builtin_call_in_expr(expr, builtins)),
        ExprCarrier::Atom(_) => None,
    }
}

fn host_callback_uses_builtin<T>(callback: &HostCallback<T>, builtin: &str) -> bool {
    match &callback.kind {
        HostCallbackKind::Inline { body, .. } => host_body_uses_builtin(body, builtin),
        HostCallbackKind::Named { .. } => false,
    }
}

fn host_body_uses_builtin<T>(expr: &HostExpr<T>, builtin: &str) -> bool {
    match &expr.kind {
        HostExprKind::Builtin { name, args, .. } => {
            name == builtin || args.iter().any(|arg| host_body_uses_builtin(arg, builtin))
        }
        HostExprKind::Call { args, .. } | HostExprKind::SignatureEntry { args, .. } => {
            args.iter().any(|arg| host_body_uses_builtin(arg, builtin))
        }
        HostExprKind::TensorCall { args, .. } => {
            args.iter().any(|arg| host_body_uses_builtin(arg, builtin))
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            fields.iter().any(|f| host_body_uses_builtin(f, builtin))
        }
        HostExprKind::Tuple(items, _) | HostExprKind::List(items, _) => {
            items.iter().any(|i| host_body_uses_builtin(i, builtin))
        }
        HostExprKind::AdtFieldAccess { base, .. } => host_body_uses_builtin(base, builtin),
        HostExprKind::Let { bindings, body, .. } => {
            bindings
                .iter()
                .any(|b| host_body_uses_builtin(&b.value, builtin))
                || host_body_uses_builtin(body, builtin)
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            host_body_uses_builtin(cond, builtin)
                || host_body_uses_builtin(then_expr, builtin)
                || host_body_uses_builtin(else_expr, builtin)
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            host_body_uses_builtin(scrutinee, builtin)
                || host_body_uses_builtin(some_expr, builtin)
                || host_body_uses_builtin(none_expr, builtin)
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            host_body_uses_builtin(scrutinee, builtin)
                || arms
                    .iter()
                    .any(|arm| host_body_uses_builtin(&arm.expr, builtin))
                || default_expr
                    .as_ref()
                    .is_some_and(|d| host_body_uses_builtin(d, builtin))
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            host_callback_uses_builtin(callback, builtin) || host_body_uses_builtin(list, builtin)
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            host_callback_uses_builtin(callback, builtin)
                || host_body_uses_builtin(init, builtin)
                || host_body_uses_builtin(list, builtin)
        }
        HostExprKind::WithSeed { seed, body, .. } => {
            host_body_uses_builtin(seed, builtin) || host_body_uses_builtin(body, builtin)
        }
        HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Var(_, _)
        | HostExprKind::Unit => false,
    }
}

fn derive_host_function_specializations(functions: &mut [HostFunction]) {
    let mut summaries = functions
        .iter()
        .filter_map(|function| {
            function
                .specialization
                .clone()
                .map(|summary| (function.name.clone(), summary))
        })
        .collect::<UnordMap<_, _>>();

    loop {
        let mut changed = false;
        for function in functions.iter_mut() {
            if summaries.contains_key(&function.name) {
                continue;
            }
            let Some(summary) = derive_host_function_specialization(function, &summaries) else {
                continue;
            };
            summaries.insert(function.name.clone(), summary.clone());
            function.specialization = Some(summary);
            changed = true;
        }
        if !changed {
            break;
        }
    }

    // After spec derivation, collect structured summary rejections per
    // function. This pass owns:
    //
    //   * promoting partial `HelperSummaryRejection`s on each tensor
    //     helper into fully-formed `SummaryRejection`s attached to
    //     `HostFunction::summary_rejections` (and to
    //     `HostProgram::summary_rejections` via `try_lower_compiled_program`),
    //   * detecting the function-level `MultipleReturnPaths` case
    //     (helper body is `If`, not `TensorCall`) and emitting a
    //     rejection for it. This is the W3-B-enumerated category 2
    //     case; it is not visible to `try_summarize_sparse_helper`
    //     because the function body never reaches the tensor-helper
    //     summarization path.
    for function in functions.iter_mut() {
        function.summary_rejections.clear();
        collect_function_summary_rejections(function);
    }
}

/// Promote each tensor helper's `HelperSummaryRejection` into a
/// fully-formed `SummaryRejection` on this function. Also detects the
/// function-level `MultipleReturnPaths` case (function body lowered
/// to `If` rather than `TensorCall`, branching across multiple
/// sparse-op return paths).
fn collect_function_summary_rejections(function: &mut HostFunction) {
    // Track which helpers are referenced from the body so we can
    // attach the right callsite span.
    let callsite_span_for_helper = body_callsite_span_per_helper(&function.body);

    let helper_path = HelperPath::local(&function.name);

    for (idx, helper) in function.tensor_helpers.iter().enumerate() {
        if let Some(partial) = &helper.summary_rejection {
            let callsite_span = callsite_span_for_helper.get(&idx).cloned().unwrap_or(None);
            function.summary_rejections.push(SummaryRejection {
                rejection_class: partial.rejection_class.clone(),
                helper_path: helper_path.clone(),
                callsite_span,
                helper_body_span: partial.helper_body_span.clone(),
                detail: partial.detail.clone(),
            });
        }
    }

    // Category 2 (MultipleReturnPaths): function body is `If` whose
    // arms both name a sparse op (directly or via a wrapper call).
    // Detect by walking the body shape; if the body is exactly `If`
    // and at least one arm references a sparse-op helper, emit.
    if let Some(rejection) = detect_multiple_return_paths_rejection(function) {
        function.summary_rejections.push(rejection);
    }
}

/// Build a map from `helper_index -> callsite_span` by walking a
/// HostExpr for `TensorCall { helper, .. }` nodes. The first
/// observed callsite span wins; a `None` entry means the helper is
/// referenced but the call site has no span. Helpers never referenced
/// are absent from the map.
fn body_callsite_span_per_helper(expr: &HostExpr) -> UnordMap<usize, Option<String>> {
    fn walk(expr: &HostExpr, out: &mut UnordMap<usize, Option<String>>) {
        match &expr.kind {
            HostExprKind::TensorCall { helper, args, .. } => {
                out.entry(*helper).or_insert_with(|| expr.span_id.clone());
                for arg in args {
                    walk(arg, out);
                }
            }
            HostExprKind::Call { args, .. } => {
                for arg in args {
                    walk(arg, out);
                }
            }
            HostExprKind::Builtin { args, .. } | HostExprKind::SignatureEntry { args, .. } => {
                for arg in args {
                    walk(arg, out);
                }
            }
            HostExprKind::If {
                cond,
                then_expr,
                else_expr,
                ..
            } => {
                walk(cond, out);
                walk(then_expr, out);
                walk(else_expr, out);
            }
            HostExprKind::Let { bindings, body, .. } => {
                for binding in bindings {
                    walk(&binding.value, out);
                }
                walk(body, out);
            }
            HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
                for item in items {
                    walk(item, out);
                }
            }
            HostExprKind::AdtConstruct { fields, .. } => {
                for field in fields {
                    walk(field, out);
                }
            }
            HostExprKind::AdtFieldAccess { base, .. } => {
                walk(base, out);
            }
            HostExprKind::MatchOption {
                scrutinee,
                some_expr,
                none_expr,
                ..
            } => {
                walk(scrutinee, out);
                walk(some_expr, out);
                walk(none_expr, out);
            }
            HostExprKind::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ..
            } => {
                walk(scrutinee, out);
                for arm in arms {
                    walk(&arm.expr, out);
                }
                if let Some(d) = default_expr {
                    walk(d, out);
                }
            }
            HostExprKind::Map { list, .. }
            | HostExprKind::Filter { list, .. }
            | HostExprKind::Partition { list, .. }
            | HostExprKind::FlatMap { list, .. } => walk(list, out),
            HostExprKind::Fold { init, list, .. } | HostExprKind::Scan { init, list, .. } => {
                walk(init, out);
                walk(list, out);
            }
            HostExprKind::WithSeed { seed, body, .. } => {
                walk(seed, out);
                walk(body, out);
            }
            _ => {}
        }
    }
    let mut out = UnordMap::new();
    walk(expr, &mut out);
    out
}

/// Detect the `MultipleReturnPaths` rejection (W3-B category 2):
/// function body lowers to a HostExprKind::If with at least two
/// distinct return arms, and at least one arm names a sparse op
/// (directly via a TensorCall on a sparse-summarized helper, or
/// transitively via a Call to a function with a sparse
/// specialization). The callsite_span is the If's own span; the
/// helper_body_span is the deepest available arm span.
fn detect_multiple_return_paths_rejection(function: &HostFunction) -> Option<SummaryRejection> {
    let HostExprKind::If {
        then_expr,
        else_expr,
        ..
    } = &function.body.kind
    else {
        return None;
    };
    // Count branches: a chain of nested if/else collapses into one
    // detection but `branch_count` records the visible structural arm
    // count (then + else, recursively counting else-as-if).
    fn count_branches(expr: &HostExpr) -> usize {
        match &expr.kind {
            HostExprKind::If {
                then_expr,
                else_expr,
                ..
            } => count_branches(then_expr) + count_branches(else_expr),
            _ => 1,
        }
    }
    let branch_count = count_branches(&function.body);
    // Heuristic: at least one arm contains a TensorCall to a sparse-
    // summarized helper OR a Call to a known sparse function. We err
    // on the side of "yes, this is a sparse-helper rejection" if any
    // arm has a tensor call. The narrower check is a follow-up; for
    // W4-A the structural marker is what matters.
    fn arm_references_sparse_helper(expr: &HostExpr, function: &HostFunction) -> bool {
        match &expr.kind {
            HostExprKind::TensorCall { helper, .. } => function
                .tensor_helpers
                .get(*helper)
                .and_then(|h| h.specialization.as_ref())
                .is_some_and(|s| {
                    matches!(
                        s,
                        HostTensorSpecialization::SparseGather(_)
                            | HostTensorSpecialization::SparseScatterAdd(_)
                            | HostTensorSpecialization::SparseScatterReplace(_)
                    )
                }),
            HostExprKind::If {
                then_expr,
                else_expr,
                ..
            } => {
                arm_references_sparse_helper(then_expr, function)
                    || arm_references_sparse_helper(else_expr, function)
            }
            HostExprKind::Let { body, bindings, .. } => {
                arm_references_sparse_helper(body, function)
                    || bindings
                        .iter()
                        .any(|b| arm_references_sparse_helper(&b.value, function))
            }
            _ => false,
        }
    }
    if !arm_references_sparse_helper(then_expr, function)
        && !arm_references_sparse_helper(else_expr, function)
    {
        return None;
    }
    let callsite_span = function.body.span_id.clone();
    let helper_body_span = then_expr
        .span_id
        .clone()
        .or_else(|| else_expr.span_id.clone());
    Some(SummaryRejection {
        rejection_class: SummaryRejectionClass::MultipleReturnPaths,
        helper_path: HelperPath::local(&function.name),
        callsite_span,
        helper_body_span,
        detail: SummaryRejectionDetail::MultipleReturnPaths { branch_count },
    })
}

fn derive_host_function_specialization(
    function: &HostFunction,
    summaries: &UnordMap<String, HostFunctionSpecialization>,
) -> Option<HostFunctionSpecialization> {
    match &function.body.kind {
        HostExprKind::TensorCall { helper, args, .. } => {
            match function
                .tensor_helpers
                .get(*helper)?
                .specialization
                .as_ref()?
            {
                HostTensorSpecialization::BlasMatmul(summary) => {
                    remap_blas_summary_to_params(summary, args, &function.params)
                        .map(HostFunctionSpecialization::BlasMatmul)
                }
                HostTensorSpecialization::SparseGather(summary) => {
                    remap_sparse_summary_to_params(summary, args, &function.params)
                        .map(HostFunctionSpecialization::SparseGather)
                }
                HostTensorSpecialization::SparseScatterAdd(summary) => {
                    remap_sparse_summary_to_params(summary, args, &function.params)
                        .map(HostFunctionSpecialization::SparseScatterAdd)
                }
                HostTensorSpecialization::SparseScatterReplace(summary) => {
                    remap_sparse_summary_to_params(summary, args, &function.params)
                        .map(HostFunctionSpecialization::SparseScatterReplace)
                }
            }
        }
        HostExprKind::Call {
            function: callee,
            args,
            ..
        } => match summaries.get(callee)? {
            HostFunctionSpecialization::BlasMatmul(summary) => {
                remap_blas_summary_to_params(summary, args, &function.params)
                    .map(HostFunctionSpecialization::BlasMatmul)
            }
            HostFunctionSpecialization::SparseGather(summary) => {
                remap_sparse_summary_to_params(summary, args, &function.params)
                    .map(HostFunctionSpecialization::SparseGather)
            }
            HostFunctionSpecialization::SparseScatterAdd(summary) => {
                remap_sparse_summary_to_params(summary, args, &function.params)
                    .map(HostFunctionSpecialization::SparseScatterAdd)
            }
            HostFunctionSpecialization::SparseScatterReplace(summary) => {
                remap_sparse_summary_to_params(summary, args, &function.params)
                    .map(HostFunctionSpecialization::SparseScatterReplace)
            }
        },
        _ => None,
    }
}

/// Remap a sparse-op summary's positional `input_indices` so they
/// reference the caller's parameters rather than the callee's
/// parameters. The shape of `args` must be a tuple of `Var` references
/// to caller parameters (the pure-pass-through wrapper case); any other
/// shape disqualifies the callsite.
fn remap_sparse_summary_to_params(
    summary: &HostSparseOpSummary,
    args: &[HostExpr],
    params: &[HostParam],
) -> Option<HostSparseOpSummary> {
    let arg_to_param = args
        .iter()
        .map(|arg| {
            let HostExprKind::Var(name, HostTypeTerm::Tensor(_)) = &arg.kind else {
                return None;
            };
            params.iter().position(|param| param.name == *name)
        })
        .collect::<Option<Vec<_>>>()?;
    if arg_to_param.len() != summary.input_tys.len() {
        return None;
    }
    let input_tys = params
        .iter()
        .map(|param| match &param.ty {
            HostTypeTerm::Tensor(ty) => Some(ty.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    // Verify that each summarized callee-input position still maps to a
    // caller parameter whose tensor type matches the callee's
    // requirement. This locks the contract that wrapper propagation
    // cannot widen or coerce the sparse-op operand types.
    let input_indices = summary
        .input_indices
        .iter()
        .map(|callee_idx| {
            let caller_idx = *arg_to_param.get(*callee_idx)?;
            let caller_ty = input_tys.get(caller_idx)?;
            let callee_ty = summary.input_tys.get(*callee_idx)?;
            if caller_ty != callee_ty {
                return None;
            }
            Some(caller_idx)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(HostSparseOpSummary {
        axis: summary.axis,
        input_indices,
        input_tys,
        output: summary.output.clone(),
    })
}

fn remap_blas_summary_to_params(
    summary: &HostBlasMatmulSummary,
    args: &[HostExpr],
    params: &[HostParam],
) -> Option<HostBlasMatmulSummary> {
    let arg_to_param = args
        .iter()
        .map(|arg| {
            let HostExprKind::Var(name, HostTypeTerm::Tensor(_)) = &arg.kind else {
                return None;
            };
            params.iter().position(|param| param.name == *name)
        })
        .collect::<Option<Vec<_>>>()?;
    let lhs_input = *arg_to_param.get(summary.lhs_input)?;
    let rhs_input = *arg_to_param.get(summary.rhs_input)?;
    let input_tys = params
        .iter()
        .map(|param| match &param.ty {
            HostTypeTerm::Tensor(ty) => Some(ty.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    if input_tys.get(lhs_input)? != summary.input_tys.get(summary.lhs_input)?
        || input_tys.get(rhs_input)? != summary.input_tys.get(summary.rhs_input)?
    {
        return None;
    }
    Some(HostBlasMatmulSummary {
        lhs_input,
        rhs_input,
        input_tys,
        output: summary.output.clone(),
        batch_dims: summary.batch_dims.clone(),
        m: summary.m.clone(),
        n: summary.n.clone(),
        k: summary.k.clone(),
    })
}

fn host_type_is_unresolved(ty: &HostTypeTerm) -> bool {
    ty.is_unresolved()
}

fn host_body_has_call_matching<T>(
    expr: &HostExpr<T>,
    builtin_matches: &dyn Fn(&str) -> bool,
    function_matches: &dyn Fn(&str) -> bool,
) -> bool {
    let recurse =
        |e: &HostExpr<T>| host_body_has_call_matching(e, builtin_matches, function_matches);
    match &expr.kind {
        HostExprKind::SignatureEntry { args, .. } => args.iter().any(recurse),
        HostExprKind::Builtin { name, args, .. } => {
            builtin_matches(name) || args.iter().any(recurse)
        }
        HostExprKind::Call { function, args, .. } => {
            function_matches(function) || args.iter().any(recurse)
        }
        HostExprKind::Let { bindings, body, .. } => {
            bindings.iter().any(|b| recurse(&b.value)) || recurse(body)
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => recurse(cond) || recurse(then_expr) || recurse(else_expr),
        HostExprKind::Tuple(items, _) | HostExprKind::List(items, _) => items.iter().any(recurse),
        HostExprKind::Map { list, .. }
        | HostExprKind::Filter { list, .. }
        | HostExprKind::Fold { list, .. }
        | HostExprKind::Scan { list, .. }
        | HostExprKind::Partition { list, .. }
        | HostExprKind::FlatMap { list, .. } => recurse(list),
        HostExprKind::TensorCall { args, .. } => args.iter().any(recurse),
        HostExprKind::AdtFieldAccess { base, .. } => recurse(base),
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => recurse(scrutinee) || recurse(some_expr) || recurse(none_expr),
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            recurse(scrutinee)
                || arms.iter().any(|arm| recurse(&arm.expr))
                || default_expr.as_ref().is_some_and(|d| recurse(d))
        }
        HostExprKind::AdtConstruct { fields, .. } => fields.iter().any(recurse),
        HostExprKind::WithSeed { seed, body, .. } => recurse(seed) || recurse(body),
        _ => false,
    }
}

/// Random-stream state a host evaluator threads into a kernel lowering so the
/// draws inside the kernel advance the enclosing handler's stream, the same
/// contract the transforms use through
/// `try_lower_subexpr_program_with_random_state_progress`. The C lowering
/// passes `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomLoweringState {
    pub seed: Option<u64>,
    pub counter: u64,
}

/// The kernel the C host program emits for a def whose body
/// `lower_host_function` decides to lower through the tensor DAG, in the form
/// an evaluator can run directly.
///
/// chelis#1277 B2h: the eval interpreter applies a host-lane def through this
/// kernel, so eval executes exactly the DAG C emits for the def ([05-MOV-1])
/// and the runtime-extent classes and guards derived from that DAG fire on
/// both lanes (runtime_extents.md C2.5). The decision is made once, here,
/// before any lowering: `Ok(None)` is the host lane, `Ok(Some)` the kernel,
/// and `Err` a kernel decision whose lowering failed.
#[derive(Debug, Clone)]
pub struct HostDefKernel {
    pub dag: crate::Dag,
    /// Ordered host scalar sources and checked tensor helpers, when this
    /// activation cannot execute as one tensor kernel.
    pub staged: Option<staged::HostStagedPlan>,
    /// Kernel inputs in DAG `Load` order: the referenced params by declared
    /// name and any captured top-level tensor names, each with its declared
    /// `TensorType` after dimension-symbol remapping.
    pub inputs: Vec<HostTensorInput>,
    pub output: TensorType,
    /// Declared params in signature order, so positional arguments map onto
    /// `inputs` by name and the unreferenced ones are dropped, as the C
    /// wrapper drops them.
    pub params: Vec<HostParam>,
    /// The next unused Random ordinal when a `RandomLoweringState` was given.
    pub next_random_counter: Option<u64>,
}

/// Evaluator-only transport; the public/serialized legacy kernel is not
/// extended with fields whose loss could change Random execution.
#[derive(Debug, Clone)]
pub struct HostDefEvaluationPlan {
    kernel: HostDefKernel,
    plan: Option<crate::evaluation::EvaluationPlan>,
    profile: crate::evaluation::EvaluationProfile,
    staged_plan: Option<Box<crate::evaluation::StagedEvaluationPlan>>,
}

impl HostDefEvaluationPlan {
    pub fn kernel_for_inspection(&self) -> &HostDefKernel {
        &self.kernel
    }
    pub fn plan(&self) -> Option<&crate::evaluation::EvaluationPlan> {
        self.plan.as_ref()
    }
    pub fn profile(&self) -> crate::evaluation::EvaluationProfile {
        self.profile
    }
    pub fn staged_plan(&self) -> Option<&crate::evaluation::StagedEvaluationPlan> {
        self.staged_plan.as_deref()
    }
}

/// What `lower_host_function` and [`host_def_kernel`] both start from.
struct HostDefSignature {
    name: String,
    params: Vec<HostParam>,
    scope: UnordMap<String, HostTypeTerm>,
    ret_ty: HostTypeTerm,
    body_expr: Expr,
}

/// The kernel-or-host decision for one def body, made before lowering.
enum DefBodyDecision {
    Host,
    Kernel(TensorType),
    /// A body that is a bare tensor variable: the C lane emits a variable
    /// access rather than a kernel, and an evaluator reads the binding.
    TensorVar(String, TensorType),
}

/// Decide whether `name`'s body is a kernel and, if so, lower it.
///
/// This is the decision `lower_host_function` makes for the C host program,
/// exposed so the eval interpreter applies the def through the same DAG.
/// `Ok(None)`: the host lane runs the body. `Ok(Some)`: the kernel. `Err`: the
/// decision was kernel and the lowering failed, fatal or not; the caller
/// decides what a failed kernel lowering means on its lane (the C lane's
/// fall-through to host lowering is chelis#1515 and is not part of this
/// function).
///
/// The decision is answered once per definition per session, and the session
/// is what makes "once" a type-level fact rather than a convention: the memo
/// is its field, so a caller holding only a checked program has nothing to
/// read a memoized decision out of and cannot reach this function at all.
/// That is the chelis#1835 repair. It replaces a thread-local flag that this
/// entry point, added by chelis#1531, never armed.
///
/// ```compile_fail
/// # use chelis_types::CheckedProgram;
/// fn bypass(program: &CheckedProgram) {
///     // No session: `&CheckedProgram` is not `&HostLoweringSession`, and
///     // `Deref` runs the other way.
///     let _ = chelis_ir::host::host_def_kernel(program, "f", None);
/// }
/// ```
///
/// The session form is the one that compiles:
///
/// ```no_run
/// # use chelis_ir::host::HostLoweringSession;
/// # use chelis_types::CheckedProgram;
/// fn decide(program: &CheckedProgram) {
///     let session = HostLoweringSession::new(program);
///     let _ = chelis_ir::host::host_def_kernel(&session, "f", None);
/// }
/// ```
pub fn host_def_kernel(
    program: &HostLoweringSession<'_>,
    name: &str,
    random: Option<RandomLoweringState>,
) -> Result<Option<HostDefKernel>, crate::lower::LowerDiagnostic> {
    host_def_kernel_product(program, name, random, None)
        .map(|product| product.map(|product| product.kernel))
}

/// The evaluator's counterpart to [`host_def_kernel`], carrying the execution
/// plan beside the kernel. It requires a session for the same reason and with
/// the same force: a caller holding only a checked program cannot reach it.
///
/// ```compile_fail
/// # use chelis_ir::evaluation::RandomExecutionContext;
/// # use chelis_ir::host::RandomLoweringState;
/// # use chelis_types::CheckedProgram;
/// fn bypass(program: &CheckedProgram) {
///     let context = RandomExecutionContext::new(RandomLoweringState { seed: None, counter: 0 });
///     let _ = chelis_ir::host::host_def_evaluation_plan(program, "f", &context);
/// }
/// ```
///
/// ```no_run
/// # use chelis_ir::evaluation::RandomExecutionContext;
/// # use chelis_ir::host::RandomLoweringState;
/// # use chelis_ir::host::HostLoweringSession;
/// # use chelis_types::CheckedProgram;
/// fn plan(program: &CheckedProgram) {
///     let session = HostLoweringSession::new(program);
///     let context = RandomExecutionContext::new(RandomLoweringState { seed: None, counter: 0 });
///     let _ = chelis_ir::host::host_def_evaluation_plan(&session, "f", &context);
/// }
/// ```
pub fn host_def_evaluation_plan(
    program: &HostLoweringSession<'_>,
    name: &str,
    context: &crate::evaluation::RandomExecutionContext,
) -> Result<Option<HostDefEvaluationPlan>, crate::lower::LowerDiagnostic> {
    host_def_kernel_product(program, name, Some(context.state()), Some(context))
}

fn host_def_kernel_product(
    program: &HostLoweringSession<'_>,
    name: &str,
    random: Option<RandomLoweringState>,
    execution: Option<&crate::evaluation::RandomExecutionContext>,
) -> Result<Option<HostDefEvaluationPlan>, crate::lower::LowerDiagnostic> {
    // The declaration's own name is the key for every per-def fact (the
    // effect row, the call graph); a caller may hand in a shorter spelling
    // that `find_top_level_def_named` resolves.
    let Some((canonical, body)) = find_top_level_def_named(program.exprs(), name) else {
        return Ok(None);
    };
    let name = canonical;
    let transfer_literal_result_claims =
        top_level_fn_transfers_literal_result_claims(program, name);
    // A generic definition has no standalone kernel. Its checked application
    // supplies the precision/rank bindings, including when its result is bool
    // and would otherwise look concrete enough to classify as a kernel.
    if top_level_fn_is_type_polymorphic(program, name)
        || lookup_declared_type_expr(program, name).is_some_and(|signature| {
            crate::lower::type_expr_has_precision_var(&signature)
                || crate::lower::type_expr_has_rank_var(&signature)
        })
    {
        return Ok(None);
    }
    // `ty_expr` is redundant with the lookup `host_def_signature` performs
    // first (the C caller passes that same lookup's result), so `None` here
    // yields the identical signature.
    let Some(signature) = host_def_signature(name, body, None, program) else {
        return Ok(None);
    };
    let _preflight_guard = TensorHelperPreflightGuard::begin(&signature.body_expr, program);
    // Classify lexical controls before host signature preparation substitutes
    // local callable aliases: that rewrite does not carry closure captures.
    let profile_body = as_list(body)
        .filter(|list| tag(list) == Some(DeepTag::Fn))
        .and_then(|list| children(list).get(1))
        .unwrap_or(&signature.body_expr);
    let profile = match execution {
        Some(_) => cached_subexpr_lowering_context(program).evaluation_profile(
            profile_body,
            &kernel_scope_types(&signature.scope, Some(&signature.params)),
        ),
        None => crate::evaluation::EvaluationProfile::Legacy(
            crate::evaluation::LegacyEvaluationReason::LegacyApi,
        ),
    };
    // Claims are constructed before partitioning, while the evaluator keeps
    // the source-owned Random associations across those same stage cuts.
    let mut staged_plan = None;
    {
        let execution_out = (profile == crate::evaluation::EvaluationProfile::FixedControl)
            .then_some(&mut staged_plan);
        match staged_def_kernel_product(program, &signature, random, execution_out)? {
            staged::StagingAttempt::Ready(kernel) => {
                return Ok(Some(HostDefEvaluationPlan {
                    kernel,
                    plan: None,
                    profile,
                    staged_plan: staged_plan.map(Box::new),
                }));
            }
            staged::StagingAttempt::HostControlBoundary => return Ok(None),
            staged::StagingAttempt::NotApplicable => {}
        }
    }
    let expected = match def_body_decision_impl(
        program,
        &signature,
        profile == crate::evaluation::EvaluationProfile::FixedControl,
    )? {
        DefBodyDecision::Kernel(expected) => expected,
        DefBodyDecision::Host => return Ok(None),
        DefBodyDecision::TensorVar(..) => return Ok(None),
    };
    let plan = if profile == crate::evaluation::EvaluationProfile::FixedControl {
        let context = cached_subexpr_lowering_context(program);
        let scoped = kernel_scope_types(&signature.scope, Some(&signature.params));
        let execution = execution.expect("fixed profile is only selected by the evaluator");
        let plan = if transfer_literal_result_claims {
            crate::lower::try_lower_tensor_helper_evaluation_with_ordered_inputs(
                &signature.body_expr,
                scoped,
                &context,
                Some(&expected),
                true,
                execution,
            )?
        } else {
            crate::lower::try_lower_subexpr_evaluation_with_ordered_inputs(
                &signature.body_expr,
                scoped,
                &context,
                Some(&expected),
                true,
                execution,
            )?
        };
        let rebound =
            remap_tensor_helper_dim_symbols(plan.dag_for_inspection(), &signature.scope, &expected);
        Some(
            plan.rebind_dimensions(rebound).map_err(|message| {
                crate::lower::LowerDiagnostic::new(message, None, None).fatal()
            })?,
        )
    } else {
        None
    };
    let (dag, next_random_counter) = if let Some(plan) = &plan {
        (plan.dag_for_inspection().clone(), None)
    } else {
        lower_kernel_dag(
            &signature.body_expr,
            program,
            &signature.scope,
            Some(&signature.params),
            &expected,
            random,
            transfer_literal_result_claims,
        )?
    };
    if let Some(builtin) = kernel_dag_loads_builtin(&dag, &signature.scope) {
        return Err(crate::lower::LowerDiagnostic::new(
            format!(
                "the kernel body of `{name}` reaches the host-only builtin `{builtin}`, which has \
                 no tensor-DAG lowering"
            ),
            None,
            None,
        ));
    }
    let inputs = tensor_helper_inputs(&dag);
    let output = dag
        .roots()
        .first()
        .and_then(|id| dag.get(*id))
        .map(|node| node.output_type.clone())
        .unwrap_or_else(|| expected.clone());
    Ok(Some(HostDefEvaluationPlan {
        kernel: HostDefKernel {
            dag,
            staged: None,
            inputs,
            output,
            params: signature.params,
            next_random_counter,
        },
        plan,
        profile,
        staged_plan: None,
    }))
}

fn staged_def_kernel(
    program: &HostLoweringSession<'_>,
    signature: &HostDefSignature,
    random: Option<RandomLoweringState>,
) -> Result<staged::StagingAttempt<HostDefKernel>, crate::lower::LowerDiagnostic> {
    staged_def_kernel_product(program, signature, random, None)
}

fn staged_def_kernel_product(
    program: &HostLoweringSession<'_>,
    signature: &HostDefSignature,
    random: Option<RandomLoweringState>,
    execution_out: Option<&mut Option<crate::evaluation::StagedEvaluationPlan>>,
) -> Result<staged::StagingAttempt<HostDefKernel>, crate::lower::LowerDiagnostic> {
    use staged::StagingAttempt;
    let HostTypeTerm::Tensor(expected) = &signature.ret_ty else {
        return Ok(StagingAttempt::NotApplicable);
    };
    // This is only a cheap candidate scan. Admission below uses the checked
    // expression type in its lexical activation, never this spelling scan.
    let definitions = cached_program_defs(program);
    let mut pending = vec![&signature.body_expr];
    let mut visited = BTreeSet::new();
    let mut reshape = false;
    let mut host_source = false;
    while let Some(expression) = pending.pop() {
        let mut names = UnordSet::new();
        collect_deep_var_names(expression, &mut names);
        for name in names.into_sorted() {
            reshape |= name == "reshape";
            host_source |= HOST_ONLY_BUILTINS.contains(&name.as_str());
            if visited.insert(name.clone())
                && let Some(body) = definitions.get(&name)
            {
                pending.push(body);
            }
        }
        fn has_control(expr: &Expr) -> bool {
            match expr {
                Expr::List(list, _) => {
                    matches!(tag(list), Some(DeepTag::If | DeepTag::Match))
                        || children(list).iter().any(has_control)
                }
                Expr::Node(node, _) => {
                    matches!(node.tag(), DeepTag::If | DeepTag::Match)
                        || node.children_slice().iter().any(has_control)
                }
                Expr::MetaExpr(meta, _) => has_control(&meta.expr),
                _ => false,
            }
        }
        host_source |= has_control(expression);
    }
    if !reshape || !host_source {
        return Ok(StagingAttempt::NotApplicable);
    }
    if def_effect_row_forbids_kernel(program, &signature.name)
        || signature
            .params
            .iter()
            .any(|param| matches!(param.ty, HostTypeTerm::Fn(..)))
    {
        return Ok(StagingAttempt::NotApplicable);
    }
    // chelis#1779: a runtime-shaped `to_tensor` lowers to a deliberate rank-0
    // `Load { name: "to_tensor" }` placeholder (`lower::lower_builtin_app`)
    // whose whole purpose is to be refused, so the def routes to the host
    // lane. The partition below cannot carry that marker: the placeholder is
    // neither a declared external input nor a staged producer, so the
    // exactly-one check in `staged.rs` rejects the entire def and the build
    // fails after a clean check. Read the same signal the tensor-helper
    // extractor reads and decline, because the marker already says this is
    // not our lane. Asked about the body rather than the def name: this
    // signature's body can be a rewritten tree (inlined callable lets, hoisted
    // locals), and the body is what the partition would receive.
    if expr_reaches_dynamic_to_tensor(&signature.body_expr, program) {
        return Ok(StagingAttempt::NotApplicable);
    }
    let context = cached_subexpr_lowering_context(program);
    let lowered = crate::lower::try_lower_staged_host_region(
        &signature.body_expr,
        &signature.params,
        program,
        &context,
        expected,
        crate::lower::StagedHostRegionLoweringOptions::for_declaration(
            random,
            top_level_fn_transfers_literal_result_claims(program, &signature.name),
        ),
        execution_out,
    );
    let lowered = match lowered {
        Ok(lowered) => lowered,
        Err(diagnostic) if diagnostic.fatal => return Err(diagnostic),
        Err(_) => return Ok(StagingAttempt::NotApplicable),
    };
    let (dag, plan) = match lowered {
        StagingAttempt::Ready(region) => region,
        StagingAttempt::NotApplicable => return Ok(StagingAttempt::NotApplicable),
        StagingAttempt::HostControlBoundary => return Ok(StagingAttempt::HostControlBoundary),
    };
    Ok(StagingAttempt::Ready(HostDefKernel {
        inputs: tensor_helper_inputs(&dag),
        output: dag
            .get(dag.roots()[0])
            .expect("staged root")
            .output_type
            .clone(),
        dag,
        staged: Some(plan),
        params: signature.params.clone(),
        next_random_counter: None,
    }))
}

/// Whether `name`'s checked effect row carries an effect with no DAG form
/// (`IO`, `Test`, `Resource`). Read off the same effect inference the root
/// manifest uses, cached per program like the other host-lowering facts.
fn def_effect_row_forbids_kernel(program: &HostLoweringSession<'_>, name: &str) -> bool {
    cached_def_effect_rows(program)
        .get(name)
        .is_some_and(|row| {
            row.iter().any(|effect| {
                matches!(
                    effect,
                    chelis_types::types::Effect::Io
                        | chelis_types::types::Effect::Test
                        | chelis_types::types::Effect::Resource(_)
                )
            })
        })
}

fn cached_def_effect_rows(
    program: &HostLoweringSession<'_>,
) -> Arc<BTreeMap<String, chelis_types::types::EffectSet>> {
    if let Some(cached) = program.facts.def_effect_rows.borrow().clone() {
        return cached;
    }
    let rows = Arc::new(chelis_effects::def_effect_rows(program));
    *program.facts.def_effect_rows.borrow_mut() = Some(rows.clone());
    rows
}

/// A form the kernel lowering cannot carry, found before lowering by walking
/// the body and every inlined callee with lexical scoping (chelis#1277 B2h:
/// each class is one the byte-identity corpus found the lowerer rejecting
/// after a kernel decision, named with the lowerer's own reason). An `if` is
/// deliberately not a class: the kernel lowering's `lower_if` accepts more
/// than the syntactic `if_expr_is_dag_lowerable` does, and treating the
/// latter as the rule flipped three corpus kernels to host code on C.
/// `Some(reason)` keeps the def in host code on both lanes; on C that is the
/// lane the non-fatal fall-through (chelis#1515) already chose, so the emitted
/// program is unchanged, which the corpus capture proves rather than assumes.
pub(crate) fn body_form_the_dag_cannot_carry(
    program: &HostLoweringSession<'_>,
    body: &Expr,
    params: &[HostParam],
    evaluation_dropout: bool,
) -> Option<String> {
    let defs = cached_program_defs(program);
    let mut walk = UncarriableWalk {
        defs: &defs,
        active: UnordSet::new(),
        completed: UnordMap::new(),
        cycle_cutoff: false,
        #[cfg(test)]
        def_visits: 0,
        evaluation_dropout,
        scopes: vec![
            params
                .iter()
                .map(|param| {
                    (
                        param.name.clone(),
                        AdmissionBinding {
                            constructor: false,
                            concat: ConcatInputFact::from_type(&param.ty),
                        },
                    )
                })
                .collect(),
        ],
    };
    walk.expr(body)
}

/// Whether `expr` applies a callable that is itself a computed value rather
/// than one of the callee forms the host lowering has a direct representation
/// for. This is checked before staged/kernel lowering as well as by the
/// admission walk: otherwise a tensor helper may beta-reduce the inner call
/// and erase the outer function application before the C ABI can fence it
/// (#1951).
fn contains_computed_callable_application(expr: &Expr) -> bool {
    fn recognized_callee(expr: &Expr) -> bool {
        match expr {
            Expr::MetaExpr(meta, _) => recognized_callee(&meta.expr),
            Expr::List(list, _) => {
                matches!(
                    tag(list),
                    Some(DeepTag::Var | DeepTag::Fn | DeepTag::Grad | DeepTag::Vmap)
                ) || list.unknown_tag_symbol() == Some("vmap-grad")
            }
            Expr::Node(node, _) => {
                matches!(
                    node.tag(),
                    DeepTag::Var | DeepTag::Fn | DeepTag::Grad | DeepTag::Vmap
                )
            }
            Expr::UnknownForm(data) => data.head == "vmap-grad",
            Expr::Atom(_, _) | Expr::Map(_, _) | Expr::BareList(_, _) => false,
        }
    }

    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => false,
        Expr::MetaExpr(meta, _) => contains_computed_callable_application(&meta.expr),
        Expr::BareList(items, _) => items.iter().any(contains_computed_callable_application),
        Expr::UnknownForm(data) => data
            .children
            .iter()
            .any(contains_computed_callable_application),
        Expr::Node(node, span) => {
            contains_computed_callable_application(&Expr::List(node.to_list(*span), *span))
        }
        Expr::List(list, _) => {
            let kids = children(list);
            (tag(list) == Some(DeepTag::App)
                && kids
                    .first()
                    .is_some_and(|callee| !recognized_callee(callee)))
                || kids.iter().any(contains_computed_callable_application)
        }
    }
}

/// Lexical scopes retain constructor scrutinees (chelis#520 D1) and direct
/// concat input facts. Neither class evaluates arbitrary tensor expressions.
struct UncarriableWalk<'a> {
    defs: &'a BTreeMap<String, Expr>,
    active: UnordSet<String>,
    completed: UnordMap<String, Vec<CompletedAdmission>>,
    cycle_cutoff: bool,
    #[cfg(test)]
    def_visits: usize,
    evaluation_dropout: bool,
    scopes: Vec<UnordMap<String, AdmissionBinding>>,
}

struct CompletedAdmission {
    inputs: Vec<AdmissionBinding>,
    reason: Option<String>,
}

#[derive(Clone, Default, PartialEq, Eq)]
struct AdmissionBinding {
    constructor: bool,
    concat: ConcatInputFact,
}

/// Only direct input forwarding, lexical aliases, and literal List spines.
/// An unknown computed tensor is not evidence that its DAG cannot be built.
#[derive(Clone, Default, PartialEq, Eq)]
enum ConcatInputFact {
    #[default]
    Unknown,
    Tensor(TensorType),
    List(Vec<ConcatInputFact>),
}

impl ConcatInputFact {
    fn from_type(ty: &HostTypeTerm) -> Self {
        match ty {
            HostTypeTerm::Tensor(tensor) => Self::Tensor(tensor.clone()),
            _ => Self::Unknown,
        }
    }
}

impl UncarriableWalk<'_> {
    fn bound(&self, name: &str) -> Option<bool> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).map(|binding| binding.constructor))
    }

    fn concat_input_fact(&self, expr: &Expr) -> ConcatInputFact {
        if let Some((DeepTag::Var, _, kids)) = stamped_parts(expr)
            && let Some(name) = kids.first().and_then(symbol_name)
        {
            return self
                .scopes
                .iter()
                .rev()
                .find_map(|scope| scope.get(name).map(|binding| binding.concat.clone()))
                .unwrap_or(ConcatInputFact::Unknown);
        }
        if let Some(elements) = crate::lower::collect_cons_chain(expr) {
            return ConcatInputFact::List(
                elements
                    .iter()
                    .map(|item| self.concat_input_fact(item))
                    .collect(),
            );
        }
        ConcatInputFact::Unknown
    }

    fn concat_requires_host(&self, list: &Expr, axis: &Expr) -> bool {
        let Some(axis) = crate::lower::extract_int_axis(axis) else {
            return false;
        };
        let ConcatInputFact::List(elements) = self.concat_input_fact(list) else {
            return false;
        };
        elements.iter().any(|element| {
            let ConcatInputFact::Tensor(tensor) = element else {
                return false;
            };
            let normalized = if axis < 0 {
                axis + tensor.dims.len() as i64
            } else {
                axis
            };
            usize::try_from(normalized)
                .ok()
                .and_then(|axis| tensor.dims.get(axis))
                .is_some_and(|dim| crate::lower::concrete_dim_len(dim).is_none())
        })
    }

    fn expr(&mut self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Atom(Atom::Str(_), _) => {
                Some("a string literal, which has no numeric IR constant (chelis#856)".to_string())
            }
            Expr::Atom(_, _) | Expr::Map(_, _) => None,
            Expr::MetaExpr(meta, _) => self.expr(&meta.expr),
            Expr::Node(node, span) => self.expr(&Expr::List(node.to_list(*span), *span)),
            Expr::BareList(elems, _) => elems.iter().find_map(|elem| self.expr(elem)),
            Expr::UnknownForm(data) => data.children.iter().find_map(|child| self.expr(child)),
            Expr::List(list, _) => self.list(list),
        }
    }

    fn list(&mut self, list: &List) -> Option<String> {
        let kids = children(list);
        match tag(list) {
            Some(DeepTag::Var) => {
                let name = kids.first().and_then(symbol_name)?;
                if self.bound(name).is_some()
                    || BUILTIN_NAMES.contains(&name)
                    || (self.evaluation_dropout && name == "dropout")
                    || name.chars().next().is_some_and(char::is_uppercase)
                {
                    return None;
                }
                match self.defs.get(name) {
                    // A callee inlines into the kernel, so its body is walked
                    // under its own parameters; a value binding becomes a
                    // `Load` and carries nothing.
                    Some(def_body) => self.def(name, def_body, None),
                    None => Some(format!(
                        "the name `{name}`, which is neither a parameter nor a definition \
                         of this program (a library definition under a compiled context)"
                    )),
                }
            }
            Some(DeepTag::App) => {
                // A C tensor helper can only call a direct named/local
                // callable, an inline lambda, or a transform form that the
                // host lowerer recognizes in callee position.  A computed
                // callee such as `(pick(h))(x)` is a first-class function
                // value: admitting it to the helper path erases the
                // application and can return `x` unchanged (#1951). Keep it
                // on the host path, where the existing unspellable callable
                // marker reaches the C ABI's typed #879 rejection before
                // artifact emission.
                if let Some(callee) = kids.first() {
                    let callee_tag = stamped_parts(callee).map(|(tag, _, _)| tag);
                    let recognized_callee = matches!(
                        callee_tag,
                        Some(DeepTag::Var | DeepTag::Fn | DeepTag::Grad | DeepTag::Vmap)
                    ) || matches!(
                        callee,
                        Expr::UnknownForm(data) if data.head == "vmap-grad"
                    );
                    if !recognized_callee {
                        return Some(
                            "an application whose callee is a computed function value, which \
                             the C host ABI rejects rather than erasing (chelis#1951)"
                                .to_string(),
                        );
                    }
                }
                if let Some(callee) = kids.first().and_then(as_list)
                    && tag(callee) == Some(DeepTag::Var)
                    && let Some(name) = children(callee).first().and_then(symbol_name)
                    && self.bound(name).is_none()
                    && !self.defs.contains_key(name)
                {
                    if HOST_ONLY_BUILTINS.contains(&name) {
                        return Some(format!(
                            "the builtin `{name}`, which has no tensor-DAG lowering"
                        ));
                    }
                    // [05-OP-62]: the static Pad+Add concat lowerer cannot
                    // sum an input's runtime width. Decide Host before a
                    // kernel is selected, preserving the real List operation.
                    if name == "concat"
                        && kids.len() == 3
                        && self.concat_requires_host(&kids[1], &kids[2])
                    {
                        return Some("tensor concat over a directly forwarded runtime input extent (chelis#1906)".to_string());
                    }
                    if name.starts_with("reduce_window_")
                        && kids[1..]
                            .iter()
                            .skip(1)
                            .any(|list_arg| !is_literal_int_list(list_arg))
                    {
                        return Some(format!(
                            "`{name}` with a non-literal window or stride list, which the \
                             compiled lowering does not carry (chelis#1058)"
                        ));
                    }
                    if name == "pad" && kids.get(3).is_some_and(|fill| !is_static_scalar(fill)) {
                        return Some(
                            "`pad` with a fill value that does not resolve statically \
                             (chelis#776)"
                                .to_string(),
                        );
                    }
                    // A bare name handed directly to a builtin that the program
                    // does not define is a dimension name (a named axis of a
                    // reduction or `expand`, spec/04-type-system.md section
                    // 4.5.3), which the lowering resolves against the operand;
                    // the checker has already bound every value name. A callee
                    // that is neither a builtin nor a definition is walked as a
                    // name and reported as unresolvable.
                    if BUILTIN_NAMES.contains(&name)
                        || (self.evaluation_dropout && name == "dropout")
                    {
                        return kids
                            .iter()
                            .skip(1)
                            .filter(|kid| !is_bare_lowercase_var(kid))
                            .find_map(|kid| self.expr(kid));
                    }
                }
                if let Some((DeepTag::Var, _, callee)) = kids.first().and_then(stamped_parts)
                    && let Some(name) = callee.first().and_then(symbol_name)
                    && self.bound(name).is_none()
                    && let Some(body) = self.defs.get(name)
                {
                    // Resolve every actual in the caller before any callee
                    // formal shadows it; two calls can carry different widths.
                    if let Some(found) = kids.iter().skip(1).find_map(|arg| self.expr(arg)) {
                        return Some(found);
                    }
                    let actuals = kids
                        .iter()
                        .skip(1)
                        .map(|arg| self.concat_input_fact(arg))
                        .collect();
                    return self.def(name, body, Some(actuals));
                }
                kids.iter().find_map(|kid| self.expr(kid))
            }
            Some(DeepTag::Fn) => {
                let params = kids
                    .first()
                    .map_or_else(Vec::new, fn_param_names)
                    .into_iter()
                    .map(|name| (name, AdmissionBinding::default()))
                    .collect();
                self.scopes.push(params);
                let found = kids.get(1).and_then(|body| self.expr(body));
                self.scopes.pop();
                found
            }
            Some(DeepTag::Let) => {
                self.scopes.push(UnordMap::new());
                let mut found = None;
                if let Some(bindings) = kids.first().and_then(as_list)
                    && tag(bindings) == Some(DeepTag::Bind)
                {
                    let binding_kids = children(bindings);
                    let mut index = 0;
                    while index + 1 < binding_kids.len() {
                        found = found.or_else(|| self.expr(&binding_kids[index + 1]));
                        let static_ctor = is_static_constructor(&binding_kids[index + 1], self);
                        // Only a whole-value alias inherits a whole-value fact;
                        // this walk does not project tuple/list destructuring.
                        let concat = if symbol_name(&binding_kids[index]).is_some() {
                            self.concat_input_fact(&binding_kids[index + 1])
                        } else {
                            ConcatInputFact::Unknown
                        };
                        let mut bound = UnordSet::new();
                        collect_binder_names(&binding_kids[index], &mut bound);
                        if let Some(scope) = self.scopes.last_mut() {
                            for name in bound.into_sorted() {
                                scope.insert(
                                    name,
                                    AdmissionBinding {
                                        constructor: static_ctor,
                                        concat: concat.clone(),
                                    },
                                );
                            }
                        }
                        index += 2;
                    }
                }
                let found = found.or_else(|| kids.get(1).and_then(|body| self.expr(body)));
                self.scopes.pop();
                found
            }
            Some(DeepTag::Access) => {
                let base = kids.first()?;
                if let Some(found) = self.expr(base) {
                    return Some(found);
                }
                if !is_static_constructor(base, self) {
                    return Some(
                        "a field projection on a runtime record value, which IR lowering \
                         resolves only for a compile-time-known record construction \
                         (chelis#520 D1)"
                            .to_string(),
                    );
                }
                None
            }
            Some(DeepTag::Match) => {
                let scrutinee = kids.first()?;
                if let Some(found) = self.expr(scrutinee) {
                    return Some(found);
                }
                if !is_static_constructor(scrutinee, self) {
                    return Some(
                        "a `match` on a runtime scrutinee, which IR lowering resolves only \
                         for a compile-time-known constructor value (chelis#520 D1)"
                            .to_string(),
                    );
                }
                for arm in kids.iter().skip(1) {
                    let Some(arm_list) = as_list(arm) else {
                        continue;
                    };
                    if tag(arm_list) != Some(DeepTag::Arm) {
                        continue;
                    }
                    let arm_kids = children(arm_list);
                    let mut bound = UnordSet::new();
                    if let Some(pattern) = arm_kids.first() {
                        collect_binder_names(pattern, &mut bound);
                    }
                    self.scopes.push(
                        bound
                            .into_sorted()
                            .into_iter()
                            .map(|n| (n, AdmissionBinding::default()))
                            .collect(),
                    );
                    let found = arm_kids.iter().skip(1).find_map(|kid| self.expr(kid));
                    self.scopes.pop();
                    if found.is_some() {
                        return found;
                    }
                }
                None
            }
            _ => kids.iter().find_map(|kid| self.expr(kid)),
        }
    }

    fn def(
        &mut self,
        name: &str,
        body: &Expr,
        actuals: Option<Vec<ConcatInputFact>>,
    ) -> Option<String> {
        let Some((DeepTag::Fn, _, fn_kids)) = stamped_parts(body) else {
            return None;
        };
        if self.active.contains(name) {
            self.cycle_cutoff = true;
            return None;
        }
        let params: Vec<_> = fn_kids
            .first()
            .map_or_else(Vec::new, fn_param_names)
            .into_iter()
            .enumerate()
            .map(|(index, param)| {
                let concat = match &actuals {
                    Some(actuals) => actuals
                        .get(index)
                        .cloned()
                        .unwrap_or(ConcatInputFact::Unknown),
                    None => fn_kids
                        .first()
                        .and_then(stamped_parts)
                        .and_then(|(_, _, params)| params.get(index))
                        .and_then(param_host_type)
                        .map(|ty| ConcatInputFact::from_type(&ty))
                        .unwrap_or(ConcatInputFact::Unknown),
                };
                (
                    param,
                    AdmissionBinding {
                        constructor: false,
                        concat,
                    },
                )
            })
            .collect();
        // A callee runs under only these resolved bindings. The definition
        // table and evaluation mode are invariant for this walk; free names
        // resolve there, never in the caller's discarded lexical scopes.
        // Compare exact facts (including constructor status), not a wildcard
        // join or a name-only visited bit that conflates distinct actuals.
        let inputs: Vec<_> = params.iter().map(|(_, binding)| binding.clone()).collect();
        if let Some(completed) = self
            .completed
            .get(name)
            .and_then(|contexts| contexts.iter().find(|context| context.inputs == inputs))
        {
            return completed.reason.clone();
        }
        self.active.insert(name.to_string());
        let enclosing_cutoff = std::mem::replace(&mut self.cycle_cutoff, false);
        #[cfg(test)]
        {
            self.def_visits += 1;
        }
        let saved = std::mem::replace(&mut self.scopes, vec![params.into_iter().collect()]);
        let found = fn_kids.get(1).and_then(|fn_body| self.expr(fn_body));
        self.scopes = saved;
        self.active.remove(name);
        // A cycle-truncated traversal depends on the active stack and is not
        // a completed context. Propagate that fact through its ancestors.
        if !self.cycle_cutoff {
            self.completed
                .entry(name.to_string())
                .or_default()
                .push(CompletedAdmission {
                    inputs,
                    reason: found.clone(),
                });
        }
        self.cycle_cutoff |= enclosing_cutoff;
        found
    }
}

/// Builtins the kernel lowering has no arm for: the lowerer's own host-side
/// list (`lower.rs`, `expr_requires_host_runtime_with_ctx`) minus the names
/// its `lower_builtin_app` does handle (`count`, `shape`, `concat`, `fold`,
/// a static `to_tensor`, which the preflight covers). Measured against the
/// arms on `801f92c02`.
const HOST_ONLY_BUILTINS: &[&str] = &[
    "print",
    "debug",
    "char_code",
    "char_from_code",
    "string_len",
    "string_concat",
    "string_slice",
    "string_contains",
    "string_starts_with",
    "string_ends_with",
    "string_trim",
    "to_string",
    "to_int",
    "to_float",
    "bitand",
    "bitor",
    "bitxor",
    "shl",
    "shr",
    "rank",
    "numel",
    "len",
    "index",
    "append",
    "take",
    "chunk",
    "range",
    "map",
    "filter",
    "scan",
    "tensor_scan",
    "partition",
    "flat_map",
    "flatten",
    "zip",
    "enumerate",
    "dict_of",
    "dict_get",
    "dict_contains",
    "dict_remove",
    "dict_insert",
    "dict_merge",
    "dict_keys",
    "dict_values",
    "dict_entries",
    "read_file",
    "write_file",
    "read_lines",
    "read_bytes",
    "file_exists",
    "list_dir",
    "mmap_file",
    "mmap_read",
    "mmap_len",
    "process_run",
    "round_to",
    "parse_csv",
    "to_csv",
    "csv_ints",
    "csv_strs",
    "csv_nrows",
    "csv_cols",
    "csv_int",
    "csv_str",
    "to_list",
    "pad_sequences",
    "pad_sequences_to",
    "einsum",
    "split",
    "scatter",
    "where",
    "cumsum",
    "sort",
    "diagonal",
    "trace",
    "clamp",
];

/// A `(var name)` whose name starts lowercase: the shape a dimension-name
/// argument takes in a builtin call.
fn is_bare_lowercase_var(expr: &Expr) -> bool {
    let Some((DeepTag::Var, _, kids)) = stamped_parts(expr) else {
        return false;
    };
    kids.first()
        .and_then(symbol_name)
        .is_some_and(|name| name.chars().next().is_some_and(char::is_lowercase))
}

fn fn_param_names(params: &Expr) -> Vec<String> {
    let Some(list) = as_list(params) else {
        return Vec::new();
    };
    if tag(list) != Some(DeepTag::Params) {
        return Vec::new();
    }
    children(list).iter().filter_map(param_name).collect()
}

/// Every name a `let` binder or a match pattern introduces (`pat-var` leaves,
/// plus the bare-name and tuple forms a `bind` uses).
fn collect_binder_names(pattern: &Expr, out: &mut UnordSet<String>) {
    match pattern {
        Expr::Atom(Atom::Name(name), _) => {
            out.insert(name.clone());
        }
        Expr::Atom(_, _) | Expr::Map(_, _) => {}
        Expr::MetaExpr(meta, _) => collect_binder_names(&meta.expr, out),
        Expr::List(_, _) | Expr::Node(_, _) => {
            let Some((pattern_tag, _, kids)) = stamped_parts(pattern) else {
                return;
            };
            if matches!(pattern_tag, DeepTag::PatVar | DeepTag::Var)
                && let Some(name) = kids.first().and_then(symbol_name)
            {
                out.insert(name.to_string());
                return;
            }
            for kid in kids {
                collect_binder_names(kid, out);
            }
        }
        Expr::BareList(elems, _) => {
            for elem in elems {
                collect_binder_names(elem, out);
            }
        }
        Expr::UnknownForm(data) => {
            for kid in &data.children {
                collect_binder_names(kid, out);
            }
        }
    }
}

/// A compile-time-known constructor value: an uppercase variable, an
/// application of one, or a binder holding one.
fn is_static_constructor(expr: &Expr, walk: &UncarriableWalk<'_>) -> bool {
    let Some((expr_tag, _, kids)) = stamped_parts(expr) else {
        return false;
    };
    match expr_tag {
        // `(record {} Ctor (kv {} field value) ..)`: the constructor tag and
        // field layout are compile-time facts, which is exactly what
        // `lower_record` lowers to a `LoweredValue::Adt` and what both
        // `lower_access` and `lower_match` then resolve. Only the field
        // VALUES are runtime. Without this arm a record literal bound to a
        // name reads as a runtime value, so a projection or a `match` on it
        // would leave the DAG for the host lane even though the DAG carries
        // it (chelis#1266).
        DeepTag::Record => true,
        DeepTag::Var => kids.first().and_then(symbol_name).is_some_and(|name| {
            name.chars().next().is_some_and(char::is_uppercase) || walk.bound(name) == Some(true)
        }),
        DeepTag::App => kids
            .first()
            .and_then(as_list)
            .filter(|callee| tag(callee) == Some(DeepTag::Var))
            .and_then(|callee| children(callee).first().and_then(symbol_name))
            .is_some_and(|name| name.chars().next().is_some_and(char::is_uppercase)),
        _ => false,
    }
}

/// A window or stride list the compiled lowering accepts: a literal `Cons`
/// chain of integer literals (`lower_builtin_app`'s `reduce_window_*` rule).
fn is_literal_int_list(expr: &Expr) -> bool {
    crate::lower::collect_cons_chain(expr).is_some_and(|elems| {
        elems
            .iter()
            .all(|elem| crate::lower::extract_int_for_dim(elem).is_some())
    })
}

/// A fill value `lower_builtin_app` resolves statically for `pad`: a numeric
/// literal, optionally under `neg` or a float `cast` (chelis#776).
fn is_static_scalar(expr: &Expr) -> bool {
    match expr {
        Expr::Atom(Atom::Float(_), _) | Expr::Atom(Atom::Int(_), _) => true,
        Expr::MetaExpr(meta, _) => is_static_scalar(&meta.expr),
        Expr::List(_, _) | Expr::Node(_, _) => {
            let Some((expr_tag, _, kids)) = stamped_parts(expr) else {
                return false;
            };
            match expr_tag {
                DeepTag::Lit => kids.first().is_some_and(is_static_scalar),
                DeepTag::Cast => kids.first().is_some_and(is_static_scalar),
                DeepTag::App => {
                    kids.first()
                        .and_then(as_list)
                        .filter(|callee| tag(callee) == Some(DeepTag::Var))
                        .and_then(|callee| children(callee).first().and_then(symbol_name))
                        == Some("neg")
                        && kids.get(1).is_some_and(is_static_scalar)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// The predicate list `lower_host_function` has always applied, evaluated
/// before lowering: declared tensor result, no callable parameter, no
/// recursive, callable-parameter or summary-rejecting callee reached, root
/// not kept in the host lane, no dynamic `to_tensor` reach, no forward `fail`.
fn def_body_decision(
    program: &HostLoweringSession<'_>,
    signature: &HostDefSignature,
) -> Result<DefBodyDecision, crate::lower::LowerDiagnostic> {
    let fixed_dropout = cached_subexpr_lowering_context(program).c_execution_profile(
        &signature.body_expr,
        &kernel_scope_types(&signature.scope, Some(&signature.params)),
    ) == crate::evaluation::EvaluationProfile::FixedControl;
    def_body_decision_impl(program, signature, fixed_dropout)
}

fn def_body_decision_impl(
    program: &HostLoweringSession<'_>,
    signature: &HostDefSignature,
    evaluation_dropout: bool,
) -> Result<DefBodyDecision, crate::lower::LowerDiagnostic> {
    let body_expr = &signature.body_expr;
    // Skip the tensor-helper path when any param is callable: the DAG
    // helper has no representation for fn-pointer inputs and would otherwise
    // coerce the callable into `chelis_scalar_tensor_from_f64`, emitting C
    // that gcc rejects. Go straight through host-lane lowering so the fn
    // application becomes a direct `f(x)` call.
    let any_callable_param = signature
        .params
        .iter()
        .any(|param| matches!(param.ty, HostTypeTerm::Fn(_, _)));
    let HostTypeTerm::Tensor(expected) = signature.ret_ty.clone() else {
        return Ok(DefBodyDecision::Host);
    };
    // chelis#1528: a body whose checked effect row carries an effect the DAG
    // cannot represent stays in host code on both lanes. `spec/04-type-system.md`
    // section 7.1 makes `IO` the host-side observable interaction effect,
    // `Test` the test runner's, and `Resource(Device)` a placement region;
    // [05-HOST-2] forbids representing "a device-only kernel may not perform
    // IO" as an inert stub, which is what a kernel that drops `print` is.
    // `Random` (`UniformLike`) and `Accum` (gradient accumulation) are carried
    // by the DAG.
    if def_effect_row_forbids_kernel(program, &signature.name) {
        return Ok(DefBodyDecision::Host);
    }
    // A computed function value has no tensor-helper representation. Keep it
    // in the host lane before any DAG path can erase the outer application;
    // the C ABI then emits its typed unsupported-callable fence (#1951).
    if contains_computed_callable_application(body_expr) {
        return Ok(DefBodyDecision::Host);
    }
    // A form the kernel lowering cannot carry keeps the def in host code on
    // both lanes, decided here rather than discovered by a failed lowering
    // (chelis#1277 B2h; the classes the byte-identity corpus found).
    if body_form_the_dag_cannot_carry(program, body_expr, &signature.params, evaluation_dropout)
        .is_some()
    {
        return Ok(DefBodyDecision::Host);
    }
    // The callee summary probe is asked LAST of the host-lane predicates, and
    // only after the declared result type, the effect row and the body form
    // have each had their chance to answer. It is the only one that lowers a
    // callee, so every cheaper predicate that answers first is a probe not
    // run. Asking it eagerly made a non-tensor definition pay for a result
    // the very next line discarded, which is what every definition in
    // `Std.Io.Json` was doing; `||` short-circuits, so the order IS the
    // saving. The predicates are independent, so the decision is unchanged.
    if any_callable_param
        || expr_needs_host_lane_tensor_lowering(body_expr, program)
        || expr_calls_top_level_fn_with_callable_param(body_expr, program)
        || should_keep_tensor_expr_in_host_lane(body_expr)
        || expr_calls_summary_rejecting_top_level_fn(body_expr, program)?
    {
        return Ok(DefBodyDecision::Host);
    }
    if let Expr::List(list, _) = body_expr
        && tag(list) == Some(DeepTag::Var)
        && let Some(name) = children(list).first().and_then(symbol_name)
    {
        return Ok(DefBodyDecision::TensorVar(name.to_string(), expected));
    }
    if tensor_helper_preflight_rejects(body_expr) {
        record_host_work(|profile| {
            profile.tensor_helper_fallbacks += 1;
            profile.tensor_helper_preflight_rejections += 1;
        });
        return Ok(DefBodyDecision::Host);
    }
    record_host_work(|profile| {
        profile.tensor_helper_attempts += 1;
        profile.tensor_helper_input_nodes += deep_expr_nodes(body_expr);
    });
    // chelis#631: never swallow a fail-reaching forward body into a kernel;
    // the DAG lane lowers `fail` to a mask-selected placeholder while the
    // host lane keeps it as real control flow (see `lower_tensor_helper_dag`).
    let defs = cached_program_defs(program);
    if expr_reaches_forward_fail(body_expr, &defs, &mut UnordSet::new()) {
        record_host_work(|profile| {
            profile.tensor_helper_fail_guard_rejections += 1;
            profile.tensor_helper_fallbacks += 1;
        });
        return Ok(DefBodyDecision::Host);
    }
    Ok(DefBodyDecision::Kernel(expected))
}

/// The C lane's use of the shared decision: a kernel call, or `None` for the
/// host lane. A failed kernel lowering falls through to host lowering here,
/// which is the pre-existing chelis#1515 split; `host_def_kernel` does not
/// inherit it.
fn lower_def_body_kernel(
    program: &HostLoweringSession<'_>,
    signature: &HostDefSignature,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<Option<HostExpr>, crate::lower::LowerDiagnostic> {
    // A returned/dynamically-computed callable has no C value ABI. Do not
    // let a staged or tensor helper erase its outer application; the host
    // lowering route emits the existing unspellable marker, and the C ABI
    // turns that into the typed #879 rejection before artifacts exist.
    if contains_computed_callable_application(&signature.body_expr) {
        return Ok(None);
    }
    let transfer_literal_result_claims =
        top_level_fn_transfers_literal_result_claims(program, &signature.name);
    match staged_def_kernel(program, signature, None)? {
        staged::StagingAttempt::Ready(kernel) => {
            return lower_staged_host_plan(
                kernel.staged.as_ref().expect("staged kernel"),
                program,
                &signature.scope,
                tensor_helpers,
            )
            .map(|body| {
                record_literal_result_transfer(signature, tensor_helpers);
                Some(body)
            });
        }
        staged::StagingAttempt::HostControlBoundary => return Ok(None),
        staged::StagingAttempt::NotApplicable => {}
    }
    let expected = match def_body_decision(program, signature)? {
        DefBodyDecision::Host => return Ok(None),
        DefBodyDecision::TensorVar(name, expected) => {
            return Ok(Some(HostExpr::new(HostExprKind::Var(
                name,
                HostTypeTerm::Tensor(expected),
            ))));
        }
        DefBodyDecision::Kernel(expected) => expected,
    };
    if tensor_helpers.collect_execution {
        let context = cached_subexpr_lowering_context(program);
        let scoped = kernel_scope_types(&signature.scope, Some(&signature.params));
        if context.c_execution_profile(&signature.body_expr, &scoped)
            == crate::evaluation::EvaluationProfile::FixedControl
        {
            let planning = crate::evaluation::RandomExecutionContext::new(RandomLoweringState {
                seed: None,
                counter: 0,
            });
            #[cfg(feature = "lowering-trace")]
            let (plan, trace) = if tensor_helpers.collect_trace && transfer_literal_result_claims {
                let (plan, trace) =
                    crate::lower::try_lower_tensor_helper_c_execution_with_ordered_inputs_and_trace(
                        &signature.body_expr,
                        scoped,
                        &context,
                        Some(&expected),
                        true,
                        &planning,
                    )?;
                (plan, Some(trace))
            } else if tensor_helpers.collect_trace {
                let (plan, trace) =
                    crate::lower::try_lower_subexpr_c_execution_with_ordered_inputs_and_trace(
                        &signature.body_expr,
                        scoped,
                        &context,
                        Some(&expected),
                        true,
                        &planning,
                    )?;
                (plan, Some(trace))
            } else if transfer_literal_result_claims {
                (
                    crate::lower::try_lower_tensor_helper_c_execution_with_ordered_inputs(
                        &signature.body_expr,
                        scoped,
                        &context,
                        Some(&expected),
                        true,
                        &planning,
                    )?,
                    None,
                )
            } else {
                (
                    crate::lower::try_lower_subexpr_c_execution_with_ordered_inputs(
                        &signature.body_expr,
                        scoped,
                        &context,
                        Some(&expected),
                        true,
                        &planning,
                    )?,
                    None,
                )
            };
            #[cfg(not(feature = "lowering-trace"))]
            let plan = if transfer_literal_result_claims {
                crate::lower::try_lower_tensor_helper_c_execution_with_ordered_inputs(
                    &signature.body_expr,
                    scoped,
                    &context,
                    Some(&expected),
                    true,
                    &planning,
                )?
            } else {
                crate::lower::try_lower_subexpr_c_execution_with_ordered_inputs(
                    &signature.body_expr,
                    scoped,
                    &context,
                    Some(&expected),
                    true,
                    &planning,
                )?
            };
            let rebound = remap_tensor_helper_dim_symbols(
                plan.dag_for_inspection(),
                &signature.scope,
                &expected,
            );
            let plan = plan.rebind_dimensions(rebound).map_err(|message| {
                crate::lower::LowerDiagnostic::new(message, None, None).fatal()
            })?;
            #[cfg(feature = "lowering-trace")]
            let mut trace = trace;
            #[cfg(feature = "lowering-trace")]
            if let Some(trace) = &mut trace {
                trace.record_dimension_rebinding(plan.dag_for_inspection());
            }
            let (dag, execution) = plan.into_parts();
            record_literal_result_transfer(signature, tensor_helpers);
            return Ok(Some(finish_tensor_helper_product(
                dag,
                Some(execution),
                #[cfg(feature = "lowering-trace")]
                trace,
                &signature.scope,
                tensor_helpers,
                expected,
            )));
        }
    }
    #[cfg(feature = "lowering-trace")]
    let lowered = if tensor_helpers.collect_trace && transfer_literal_result_claims {
        let context = cached_subexpr_lowering_context(program);
        let scoped = kernel_scope_types(&signature.scope, Some(&signature.params));
        crate::lower::try_lower_tensor_helper_program_with_ordered_inputs_and_trace(
            &signature.body_expr,
            scoped,
            &context,
            Some(&expected),
            None,
            0,
            true,
        )
        .map(|(dag, _, trace)| ((dag, None), Some(trace)))
    } else if tensor_helpers.collect_trace {
        let context = cached_subexpr_lowering_context(program);
        let scoped = kernel_scope_types(&signature.scope, Some(&signature.params));
        crate::lower::try_lower_subexpr_program_with_ordered_inputs_and_trace(
            &signature.body_expr,
            scoped,
            &context,
            Some(&expected),
            None,
            0,
            true,
        )
        .map(|(dag, _, trace)| ((dag, None), Some(trace)))
    } else {
        lower_kernel_dag(
            &signature.body_expr,
            program,
            &signature.scope,
            Some(&signature.params),
            &expected,
            None,
            transfer_literal_result_claims,
        )
        .map(|lowered| (lowered, None))
    };
    #[cfg(not(feature = "lowering-trace"))]
    let lowered = lower_kernel_dag(
        &signature.body_expr,
        program,
        &signature.scope,
        Some(&signature.params),
        &expected,
        None,
        transfer_literal_result_claims,
    );
    let (dag, _trace) = match lowered {
        #[cfg(feature = "lowering-trace")]
        Ok(((dag, _), trace)) => {
            let dag = remap_tensor_helper_dim_symbols(&dag, &signature.scope, &expected);
            let mut trace = trace;
            if let Some(trace) = &mut trace {
                trace.record_dimension_rebinding(&dag);
            }
            (dag, trace)
        }
        #[cfg(not(feature = "lowering-trace"))]
        Ok((dag, _)) => (dag, ()),
        Err(diagnostic) if diagnostic.fatal => {
            crate::lower::raise_fatal_lowering_diagnostic(diagnostic)
        }
        Err(_) => {
            record_host_work(|profile| {
                profile.tensor_helper_dag_rejections += 1;
                profile.tensor_helper_fallbacks += 1;
            });
            return Ok(None);
        }
    };
    if kernel_dag_loads_builtin(&dag, &signature.scope).is_some() {
        record_host_work(|profile| {
            profile.tensor_helper_fallbacks += 1;
            profile.tensor_helper_builtin_load_rejections += 1;
        });
        return Ok(None);
    }
    record_host_work(|profile| profile.tensor_helper_successes += 1);
    record_literal_result_transfer(signature, tensor_helpers);
    Ok(Some(finish_tensor_helper_product(
        dag,
        None,
        #[cfg(feature = "lowering-trace")]
        _trace,
        &signature.scope,
        tensor_helpers,
        expected,
    )))
}

fn record_literal_result_transfer(signature: &HostDefSignature, sink: &mut TensorHelperSink) {
    sink.transferred_result_claim_axes = tensor_type_from_host_input(&signature.ret_ty)
        .into_iter()
        .flat_map(|ty| ty.dims.into_iter().enumerate())
        .filter_map(|(axis, dim)| {
            matches!(dim, crate::dag::DimInfo::Lit(_)).then(|| {
                crate::dag::RtAxis::Lit(i32::try_from(axis).expect("declared rank fits i32"))
            })
        })
        .collect();
}

/// A `Load` named after a builtin means the lowerer treated a host-lane
/// builtin as a free variable; emitting that DAG would reference a symbol
/// that does not exist.
fn kernel_dag_loads_builtin(
    dag: &crate::Dag,
    scope: &UnordMap<String, HostTypeTerm>,
) -> Option<String> {
    dag.nodes().iter().find_map(|node| match &node.op {
        crate::dag::RiscOp::Load { name }
            if BUILTIN_NAMES.contains(&name.as_str())
                && scope
                    .get(name.as_str())
                    .and_then(tensor_type_from_host_input)
                    .is_none() =>
        {
            Some(name.as_str().to_string())
        }
        _ => None,
    })
}

/// The one kernel lowering: the body over its declared tensor scope, then
/// dimension symbols remapped onto the declared signature. Every failure is
/// an `Err`; the callers decide what it means on their lane.
fn lower_kernel_dag(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    declaring_params: Option<&[HostParam]>,
    expected: &TensorType,
    random: Option<RandomLoweringState>,
    transfer_literal_result_claims: bool,
) -> Result<(crate::Dag, Option<u64>), crate::lower::LowerDiagnostic> {
    let context = cached_subexpr_lowering_context(program);
    let scope_types = kernel_scope_types(scope, declaring_params);
    let (dag, next_random_counter) = match random {
        None => {
            let lowered = if transfer_literal_result_claims {
                crate::lower::try_lower_tensor_helper_program_with_ordered_inputs(
                    expr,
                    scope_types,
                    &context,
                    declaring_params.is_some().then_some(expected),
                    None,
                    0,
                    declaring_params.is_some(),
                )?
            } else {
                crate::lower::try_lower_subexpr_program_with_ordered_inputs(
                    expr,
                    scope_types,
                    &context,
                    declaring_params.is_some().then_some(expected),
                    None,
                    0,
                    declaring_params.is_some(),
                )?
            };
            (lowered.0, None)
        }
        Some(state) => {
            let (dag, counter) = if transfer_literal_result_claims {
                crate::lower::try_lower_tensor_helper_program_with_ordered_inputs(
                    expr,
                    scope_types,
                    &context,
                    declaring_params.is_some().then_some(expected),
                    state.seed,
                    state.counter,
                    declaring_params.is_some(),
                )?
            } else {
                crate::lower::try_lower_subexpr_program_with_ordered_inputs(
                    expr,
                    scope_types,
                    &context,
                    declaring_params.is_some().then_some(expected),
                    state.seed,
                    state.counter,
                    declaring_params.is_some(),
                )?
            };
            (dag, Some(counter))
        }
    };
    Ok((
        remap_tensor_helper_dim_symbols(&dag, scope, expected),
        next_random_counter,
    ))
}

fn kernel_scope_types(
    scope: &UnordMap<String, HostTypeTerm>,
    declaring_params: Option<&[HostParam]>,
) -> Vec<(String, TensorType)> {
    match declaring_params {
        Some(params) => params
            .iter()
            .filter_map(|param| {
                tensor_type_from_host_input(&param.ty).map(|ty| (param.name.clone(), ty))
            })
            .collect(),
        None => collect_tensor_scope(scope).into_sorted(),
    }
}

/// The declared parameters, their host types, the declared result type and
/// the inlined body that both `lower_host_function` and [`host_def_kernel`]
/// start from. `None` when the def has no lowerable shape at all.
fn host_def_signature(
    name: &str,
    body: &Expr,
    ty_expr: Option<&Expr>,
    program: &HostLoweringSession<'_>,
) -> Option<HostDefSignature> {
    // Preserve authored dimension identities while expanding only the
    // checker-validated nominal aliases below. Replacing this expression with
    // the canonical checked signature turns `batch` into an anonymous `dN`
    // and can collapse distinct runtime guards onto one checker variable.
    let declared_fn_type_expr =
        lookup_declared_type_expr(program, name).or_else(|| ty_expr.cloned());
    let fn_type_parts = declared_fn_type_expr
        .as_ref()
        .and_then(parse_fn_type_expr_parts);
    // Always parse the canonical declared signature when one exists. Some
    // callers (notably summary classification) intentionally do not carry
    // the enclosing `def`'s type child, but they must not re-lower the same
    // checked function under fresh inference variables: that loses symbolic
    // dimension provenance and can turn a valid tensor helper into a false
    // sourceless runtime-extent materialization rejection.
    let (param_tys, ret_ty) = declared_fn_type_expr
        .as_ref()
        .and_then(|ty| parse_expanded_fn_type_expr(program, ty))
        .or_else(|| {
            expr_fn_type(body).map(|signature| expand_host_fn_type_aliases(program, signature))
        })
        .or_else(|| lookup_declared_fn_type(program, name))
        .unwrap_or((Vec::new(), fresh_host_inference()));

    let mut scope = UnordMap::new();
    let mut params = Vec::new();
    let body_expr = if let Expr::List(list, _) = body {
        if tag(list) == Some(DeepTag::Fn) {
            let kids = children(list);
            let params_list = kids.first().and_then(as_list)?;
            if tag(params_list) != Some(DeepTag::Params) {
                return None;
            }
            for (index, param) in children(params_list).iter().enumerate() {
                let Some(pname) = param_name(param) else {
                    continue;
                };
                let pty = param_tys
                    .get(index)
                    .cloned()
                    .filter(|ty| !ty.is_unresolved())
                    .or_else(|| {
                        param_host_type(param)
                            .map(|ty| expand_host_type_aliases(program, ty))
                            .filter(|ty| !ty.is_unresolved())
                    })
                    .unwrap_or_else(fresh_host_inference);
                scope.insert(pname.clone(), pty.clone());
                params.push(HostParam {
                    name: pname,
                    ty: pty,
                });
            }
            kids.get(1)?.clone()
        } else {
            if param_tys.is_empty() && ret_ty.is_unresolved() {
                return None;
            }
            for (index, param_ty) in param_tys.iter().enumerate() {
                let pname = format!("arg{index}");
                scope.insert(pname.clone(), param_ty.clone());
                params.push(HostParam {
                    name: pname,
                    ty: param_ty.clone(),
                });
            }
            synthesize_callable_application(
                body,
                &params,
                fn_type_parts.as_ref().map(|(params, _)| params.as_slice()),
                fn_type_parts.as_ref().map(|(_, ret)| ret),
            )
        }
    } else {
        return None;
    };
    // Inline any local callable bindings (fn / grad / vmap / vmap-grad)
    // into the body before lowering. The host backend only recognizes
    // grad/vmap forms in direct callee position of an `app`, so an alias
    // like `let g = grad(f); g(x)` must be rewritten to the inline
    // `(grad(f))(x)` form. Without this pass `g` lowers to the
    // unresolved-callable marker builtin
    // (`HOST_UNRESOLVED_CALLABLE_MARKER`), which ABI projection rejects
    // pre-emission.
    Some(HostDefSignature {
        name: name.to_string(),
        params,
        scope,
        ret_ty,
        body_expr: inline_local_callable_lets(&body_expr),
    })
}

fn lower_host_function(
    name: &str,
    body: &Expr,
    ty_expr: Option<&Expr>,
    program: &HostLoweringSession<'_>,
    collect_execution: bool,
    collect_trace: bool,
) -> Result<Option<LoweredHostFunction>, crate::lower::LowerDiagnostic> {
    let Some(signature) = host_def_signature(name, body, ty_expr, program) else {
        return Ok(None);
    };
    let mut tensor_helpers =
        TensorHelperSink::for_declaration(collect_execution, collect_trace, name);
    // The preflight facts are keyed by the body expression's address, so the
    // guard opens on the signature's own copy, which is not moved until the
    // body has been lowered.
    let _preflight_guard = TensorHelperPreflightGuard::begin(&signature.body_expr, program);
    // If the declared return type is a tensor, the body must produce a
    // tensor even when downstream type-metadata annotations are missing
    // from the reef'd deep AST. Force the body through the tensor-helper
    // path in that case so that pure-tensor wrapper defs like
    // `Std.Tensor.Reduce.min` get a real C function symbol rather than a
    // fallthrough `HostExpr::new(HostExprKind::Builtin)` with an "unsupported builtin"
    // placeholder (Phase 3j-pre Batch 5b bug 4).
    //
    // But skip the tensor-helper path when any param is callable: the DAG
    // helper has no representation for fn-pointer inputs and would otherwise
    // coerce the callable into `chelis_scalar_tensor_from_f64`, emitting C
    // that gcc rejects. Go straight through host-lane lowering so the fn
    // application becomes a direct `f(x)` call.
    //
    // chelis#1277 B2h: the kernel-or-host decision and the kernel lowering
    // are shared with the eval interpreter through `host_def_kernel`, so the
    // two lanes cannot answer "is this def a kernel" differently.
    let mut host_body = match lower_def_body_kernel(program, &signature, &mut tensor_helpers)? {
        Some(kernel_call) => kernel_call,
        None => lower_host_body_with_record_locals(&signature, program, &mut tensor_helpers)?,
    };
    // Per `spec/design/chelis_span_survival.md` §2.3 host-side table, the
    // "Tensor-helper extraction" and "Lowering. Fn-body" rules: every
    // input Deep span must surface as `span_id` or in `merged_spans` on at
    // least one HostExpr node. Two paths above bypass the
    // `lower_host_expr` wrapper's region-corresponding stamping:
    //
    // (1) `try_lower_tensor_helper_call(&body_expr, …)` at the success
    //     branch returns a `TensorCall` constructed via `HostExpr::new(…)`
    //     directly — `body_expr.span_id()` (the fn-body inner expr's
    //     span) is dropped.
    // (2) When `body` is a `(fn {span: …} (params …) body_expr)` form,
    //     all three subpaths above lower `body_expr` (kids[1]) but
    //     never see `body` itself — so the `(fn …)` form's own
    //     `meta["span"]` is dropped on every path.
    //
    // Append both spans to `host_body.merged_spans`. The
    // `append_merged_span` helper handles None-noop, dedup, lex-sort,
    // and canonical-equal-noop, so paths that already have the span as
    // canonical (the wrapper-routed lowering) are a no-op.
    host_body.append_merged_span(signature.body_expr.span_id());
    host_body.append_merged_span(body.span_id());
    let HostDefSignature {
        mut params, ret_ty, ..
    } = signature;
    refine_function_params_from_body(&mut params, &host_body);
    let ret_ty = if ret_ty.is_unresolved() {
        host_expr_type(&host_body)
    } else {
        ret_ty
    };
    let helper_result_claim_axes = tensor_helpers.transferred_result_claim_axes.clone();
    let (tensor_helpers, products) = tensor_helpers.into_parts();
    Ok(Some(LoweredHostFunction {
        function: HostFunction {
            helper_result_claim_axes,
            name: name.to_string(),
            params,
            ret_ty,
            body: host_body,
            tensor_helpers,
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        },
        products,
    }))
}

fn callable_type_metadata(ty: Option<&Expr>) -> chelis_deep::Metadata {
    ty.map(|ty| {
        chelis_deep::Metadata::from(chelis_deep::annotations::MetadataValue::Type(
            chelis_deep::annotations::TypeSyntax::try_new(ty.clone())
                .expect("checked callable type"),
        ))
    })
    .unwrap_or_default()
}

fn synthesize_callable_application(
    body: &Expr,
    params: &[HostParam],
    param_type_exprs: Option<&[Expr]>,
    ret_type_expr: Option<&Expr>,
) -> Expr {
    let span = body.span();
    let mut elements = vec![
        Expr::Atom(Atom::Tag(DeepTag::App), span),
        Expr::Map(callable_type_metadata(ret_type_expr), span),
        body.clone(),
    ];
    for (index, param) in params.iter().enumerate() {
        let var_meta = callable_type_metadata(param_type_exprs.and_then(|tys| tys.get(index)));
        elements.push(Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Tag(DeepTag::Var), span),
                    Expr::Map(var_meta, span),
                    Expr::Atom(Atom::Name(param.name.clone()), span),
                ],
            },
            span,
        ));
    }
    Expr::List(List { elements }, span)
}

/// Force-lower an expression through the tensor-helper path using an
/// explicit expected tensor type hint. This is used by
/// `lower_host_function` so that pure-tensor wrapper function bodies get a
/// real C function definition even when downstream type metadata is
/// missing on the reef'd deep AST's `app` nodes.
fn try_lower_tensor_helper_call(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected: TensorType,
) -> Option<HostExpr> {
    try_lower_tensor_helper_call_inner(expr, program, scope, tensor_helpers, expected, None)
}

fn try_lower_tensor_helper_call_with_context(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected: TensorType,
    lowering_context: &crate::lower::SubexprLoweringContext,
) -> Option<HostExpr> {
    try_lower_tensor_helper_call_inner(
        expr,
        program,
        scope,
        tensor_helpers,
        expected,
        Some(lowering_context),
    )
}

fn try_lower_tensor_helper_call_inner(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected: TensorType,
    lowering_context: Option<&crate::lower::SubexprLoweringContext>,
) -> Option<HostExpr> {
    if let Expr::List(list, _) = expr
        && tag(list) == Some(DeepTag::Var)
        && let Some(name) = children(list).first().and_then(symbol_name)
    {
        return Some(HostExpr::new(HostExprKind::Var(
            name.to_string(),
            HostTypeTerm::Tensor(expected),
        )));
    }
    let _preflight_guard = TensorHelperPreflightGuard::begin_if_uncovered(expr, program);
    if tensor_helper_preflight_rejects(expr) {
        record_host_work(|profile| {
            profile.tensor_helper_fallbacks += 1;
            profile.tensor_helper_preflight_rejections += 1;
        });
        return None;
    }
    record_host_work(|profile| {
        profile.tensor_helper_attempts += 1;
        profile.tensor_helper_input_nodes += deep_expr_nodes(expr);
    });
    let Some(product) = lower_tensor_helper_product(
        expr,
        program,
        scope,
        &expected,
        tensor_helpers.collect_execution,
        tensor_helpers.collect_trace,
        lowering_context,
    ) else {
        record_host_work(|profile| profile.tensor_helper_fallbacks += 1);
        return None;
    };
    // Reject unresolved builtin inputs. A typed lexical input named `fold`
    // still denotes that input (spec/04 section 8.6, chelis#2023). Otherwise
    // a `Load("fold")`
    // (or `einsum`, `map`, etc.) means the lowerer fell back to treating a
    // host-lane builtin as a free variable. Emitting this DAG would generate
    // C with a `__tensor_scalar0_0 = fold;` line — `fold` is not a C symbol.
    // Fall back to `lower_host_expr` which handles HOFs directly.
    if kernel_dag_loads_builtin(&product.dag, scope).is_some() {
        record_host_work(|profile| {
            profile.tensor_helper_fallbacks += 1;
            profile.tensor_helper_builtin_load_rejections += 1;
        });
        return None;
    }
    record_host_work(|profile| profile.tensor_helper_successes += 1);
    Some(finish_tensor_helper_product(
        product.dag,
        product.execution,
        #[cfg(feature = "lowering-trace")]
        product.trace,
        scope,
        tensor_helpers,
        expected,
    ))
}

/// A tensor-typed field projection lifted out of a tensor-helper subtree
/// (chelis#1266).
struct HoistedProjection {
    /// The projection's path, base first: `["inputs", "features", "mask"]`.
    /// IDENTITY is the path, never the rendered name. A name built by joining
    /// the segments is not injective, because `_` both separates segments and
    /// occurs inside field names: `inputs.features.mask` and
    /// `inputs.features_mask` render the same string, and deduplicating on
    /// that string gave the second projection the first one's tensor, silently
    /// and only on the C lane.
    path: Vec<String>,
    /// The host local the projection is bound to. The leading index is what
    /// makes it injective; the joined tail is there so the emitted C names the
    /// field a reader is looking for.
    name: String,
    /// The `(access {} base field)` expression, lowered host-side.
    source: Expr,
    ty: TensorType,
}

/// Record `candidate` if its path is new, and return the local's name.
///
/// Both the lookup and the name are keyed on the PATH. The index is the
/// position of the path's first occurrence in this def, so one path always
/// resolves to one local and two paths never collide however their segments
/// spell out.
fn hoisted_local_name(hoists: &mut Vec<HoistedProjection>, candidate: HoistedProjection) -> String {
    if let Some(existing) = hoists.iter().find(|entry| entry.path == candidate.path) {
        return existing.name.clone();
    }
    let index = hoists.len();
    let name = format!("__host_record_field_{index}_{}", candidate.path.join("_"));
    hoists.push(HoistedProjection {
        name: name.clone(),
        ..candidate
    });
    name
}

/// Bind every tensor-typed projection of a runtime record in `expr` to a host
/// local, returning the rewritten subtree (chelis#1266).
///
/// The DAG carries no runtime record, so an `access` anywhere inside a subtree
/// stops that whole subtree from becoming a tensor helper -- and the C host
/// lane's expression vocabulary is deliberately narrower than the checked
/// builtin vocabulary ([04-TOT-2]), so an operation like `expand` left behind
/// there has no emission at all. The same program with the projection bound to
/// a local first compiles and runs today, which is why chelis#1266 reports the
/// prologue-local rewrite as a workaround downstream applies by hand. Binding
/// it here is that rewrite, performed once by the compiler: the field's tensor
/// becomes an ordinary host value, and the helper takes it as an input like
/// any other tensor in scope.
///
/// Only a projection whose base chain bottoms out at an unshadowed
/// record-typed name in `scope` is lifted; a compile-time-known record
/// construction never reaches here, because `body_form_the_dag_cannot_carry`
/// keeps that def on its DAG route. Nested fields (`inp.inner.q`) lift as one
/// local each, keyed by path so a repeated projection binds once.
fn hoist_record_projections(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    bound: &UnordSet<String>,
    hoists: &mut Vec<HoistedProjection>,
) -> Expr {
    match expr {
        Expr::Node(node, span) => {
            if node.tag() == DeepTag::Access
                && let Some(hoist) = record_projection_hoist(expr, program, scope, bound)
            {
                return var_expr_node(&hoisted_local_name(hoists, hoist), *span);
            }
            let children = node
                .children_slice()
                .iter()
                .enumerate()
                .map(|(index, child)| {
                    if is_bound_bare_projection(node.tag(), index, child) {
                        child.clone()
                    } else {
                        hoist_record_projections(child, program, scope, bound, hoists)
                    }
                })
                .collect();
            let mut rewritten = node.clone();
            // A rewrite that the node gate refuses leaves the subtree alone;
            // the caller then takes the unhoisted path it takes today.
            if rewritten.try_replace_children(children).is_err() {
                return expr.clone();
            }
            Expr::Node(rewritten, *span)
        }
        Expr::List(list, span) => {
            if tag(list) == Some(DeepTag::Access)
                && let Some(hoist) = record_projection_hoist(expr, program, scope, bound)
            {
                return var_expr_node(&hoisted_local_name(hoists, hoist), *span);
            }
            let elements = list
                .elements
                .iter()
                .enumerate()
                .map(|(index, child)| {
                    if is_bound_bare_projection(
                        tag(list).unwrap_or(DeepTag::Var),
                        // `Expr::List` carries the tag and metadata first.
                        index.wrapping_sub(2),
                        child,
                    ) {
                        child.clone()
                    } else {
                        hoist_record_projections(child, program, scope, bound, hoists)
                    }
                })
                .collect();
            Expr::List(List { elements }, *span)
        }
        Expr::BareList(items, span) => Expr::BareList(
            items
                .iter()
                .map(|item| hoist_record_projections(item, program, scope, bound, hoists))
                .collect(),
            *span,
        ),
        // Round 2, P3: an unstamped form carries children like any other, and
        // a typed parameter is one, so the rewriter descends it rather than
        // relying on the reader having refused first.
        Expr::UnknownForm(data) => Expr::UnknownForm(Box::new(chelis_deep::ast::UnknownFormData {
            head: data.head.clone(),
            meta: data.meta.clone(),
            children: data
                .children
                .iter()
                .map(|child| hoist_record_projections(child, program, scope, bound, hoists))
                .collect(),
            span: data.span,
        })),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                metadata: meta.metadata.clone(),
                expr: Box::new(hoist_record_projections(
                    &meta.expr, program, scope, bound, hoists,
                )),
            },
            *span,
        ),
        _ => expr.clone(),
    }
}

/// Lower a host-lane def body, binding any tensor-typed runtime-record
/// projection inside it to a host local first (chelis#1266).
///
/// Only the host route takes this: a body the DAG can carry never reaches it,
/// so a compile-time-known record construction keeps its kernel lowering.
fn lower_host_body_with_record_locals(
    signature: &HostDefSignature,
    program: &HostLoweringSession<'_>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let mut bound = UnordSet::new();
    if names_bound_in(&signature.body_expr, &mut bound).is_err() {
        // Fail closed: hoist nothing, and the definition lowers exactly as it
        // did before this mechanism existed.
        return lower_host_expr(
            &signature.body_expr,
            program,
            &signature.scope,
            tensor_helpers,
        );
    }
    let mut hoists: Vec<HoistedProjection> = Vec::new();
    let rewritten = hoist_record_projections(
        &signature.body_expr,
        program,
        &signature.scope,
        &bound,
        &mut hoists,
    );
    if hoists.is_empty() {
        return lower_host_expr(
            &signature.body_expr,
            program,
            &signature.scope,
            tensor_helpers,
        );
    }
    let mut scope = signature.scope.clone();
    for hoist in &hoists {
        scope.insert(hoist.name.clone(), HostTypeTerm::Tensor(hoist.ty.clone()));
    }
    // Re-ask the kernel question on the rewritten body. The projection was the
    // form the DAG could not carry, and it is gone, so a body that is now
    // wholly tensor-valued becomes ONE kernel instead of a host `let` around a
    // body-only helper. That matters for provenance, not for tidiness: a
    // `let`-bound `shape` read split across the boundary reaches the helper as
    // an opaque scalar, and the size then resolves to no tensor axis -- the
    // check-clean / build-red divergence again, for the spelling section
    // 4.7.2's own suggestion text asks for ("bind that read to a `let`").
    // Re-asking through `lower_def_body_kernel` rather than calling the helper
    // directly is what keeps every other reason to stay in host code -- an
    // effect row the DAG cannot represent (chelis#1528), a callable parameter,
    // a host-only builtin -- deciding exactly as it did before.
    //
    // The hoisted locals join the parameter list for that question only. They
    // ARE parameters of the rewritten body: the DAG binds them like any other
    // tensor input, and the emitted function keeps its own declared parameters,
    // which `lower_host_function` reads from the original signature.
    let mut params = signature.params.clone();
    params.extend(hoists.iter().map(|hoist| HostParam {
        name: hoist.name.clone(),
        ty: HostTypeTerm::Tensor(hoist.ty.clone()),
    }));
    let rewritten_signature = HostDefSignature {
        name: signature.name.clone(),
        params,
        scope: scope.clone(),
        ret_ty: signature.ret_ty.clone(),
        body_expr: rewritten.clone(),
    };
    let body = match lower_def_body_kernel(program, &rewritten_signature, tensor_helpers)? {
        Some(kernel_call) => kernel_call,
        None => lower_host_expr(&rewritten, program, &scope, tensor_helpers)?,
    };
    let mut bindings = Vec::with_capacity(hoists.len());
    for hoist in &hoists {
        let value = lower_host_expr(&hoist.source, program, &signature.scope, tensor_helpers)?;
        bindings.push(HostBinding {
            name: hoist.name.clone(),
            display_name: None,
            display_roots: Vec::new(),
            ty: HostTypeTerm::Tensor(hoist.ty.clone()),
            value,
        });
    }
    let ty = host_expr_type(&body);
    Ok(HostExpr::new(HostExprKind::Let {
        bindings,
        body: Box::new(body),
        ty,
    }))
}

/// Every name bound anywhere in `expr`, read from the closed vocabulary's own
/// `Binder` child role rather than from a hand-written list of tag shapes.
///
/// This is the question the hoist actually needs: it must not substitute a
/// local for a projection whose base some inner construct rebinds. Asking
/// "is this name bound ANYWHERE under the body" instead of reconstructing
/// lexical scope is deliberately coarse -- a rebinding in an unrelated branch
/// suppresses a hoist that would have been safe -- and coarse in the only
/// direction that is safe: the projection then stays where it is, which is
/// the behaviour this definition had before the hoist existed.
///
/// Two defects came out of the hand-written version, and both were the same
/// shape: a binder spelling the reader did not know. `Deep` renders a TYPED
/// parameter as `(inp {type: ..})`, an `UnknownForm` whose head is the
/// parameter name, which no tag match recognized, so a lambda shadowing the
/// record base had its inner projection rewritten to the outer record's
/// local -- a wrong answer on C alone, silently and with exit zero. Reading
/// `child_stamp_role` makes the vocabulary the authority: `Params` children,
/// `Bind`'s even children and `PatVar`'s child are binder positions because
/// `spec/03-deep-syntax.md`'s role table says so.
///
/// FAILS CLOSED. A binder position this reader cannot decode returns `Err`,
/// naming the construct, and the caller then hoists nothing at all. A
/// spelling we cannot read is a spelling we cannot prove safe, and refusing
/// the optimization is an honest exit where a silent rewrite is not.
fn names_bound_in(expr: &Expr, out: &mut UnordSet<String>) -> Result<(), String> {
    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => Ok(()),
        Expr::MetaExpr(meta, _) => names_bound_in(&meta.expr, out),
        Expr::BareList(items, _) => items.iter().try_for_each(|item| names_bound_in(item, out)),
        Expr::UnknownForm(data) => data
            .children
            .iter()
            .try_for_each(|child| names_bound_in(child, out)),
        Expr::Node(_, _) | Expr::List(_, _) => {
            let Some((node_tag, _, kids)) = stamped_parts(expr) else {
                // Round 3, P3: an unstamped form still carries children.
                // Today the only one at a binder position is a typed
                // parameter, which holds no binders, so descending changes
                // nothing; not descending would have been the next place this
                // class hid.
                let Expr::List(list, _) = expr else {
                    return Ok(());
                };
                return list
                    .elements
                    .iter()
                    .try_for_each(|child| names_bound_in(child, out));
            };
            for (index, child) in kids.iter().enumerate() {
                if chelis_deep::role::child_stamp_role(node_tag, index, kids.len())
                    == chelis_deep::role::ChildStampRole::Binder
                {
                    collect_binder_position_names(node_tag, index, child, out)?;
                }
                names_bound_in(child, out)?;
            }
            Ok(())
        }
    }
}

/// Decode the name a binder-position child carries, or account for why it
/// carries none.
///
/// Three shapes reach a binder position and only one of them is a name:
///
///   * a STAMPED vocabulary node is a container, not a binder. `fn`'s binder
///     position holds `params`, and `params`'s own children are the names; the
///     whole-tree walk visits them, so there is nothing to decode here.
///   * an UNSTAMPED form whose head is the name is the binder itself. A typed
///     parameter is written `(t {type: (t-tensor ..)})`, which no closed tag
///     matches, so `stamped_parts` reads it as nothing and only its head says
///     what it binds. Two earlier readers missed exactly this, which is why
///     the oracle beside it builds every fixture through the parser.
///   * a metadata map or a non-name atom binds nothing.
///
/// Anything else FAILS CLOSED, naming the construct. The caller then abandons
/// the hoist for that definition, which lowers as it did before the mechanism
/// existed. A spelling we cannot read is one we cannot prove safe.
fn collect_binder_position_names(
    node_tag: DeepTag,
    index: usize,
    child: &Expr,
    sink: &mut UnordSet<String>,
) -> Result<(), String> {
    if stamped_parts(child).is_some() {
        return Ok(());
    }
    if let Some(name) = binder_child_name(child) {
        sink.insert(name);
        return Ok(());
    }
    if matches!(child, Expr::Map(_, _) | Expr::Atom(_, _)) {
        return Ok(());
    }
    Err(format!(
        "a binder position of `{}` at index {index} carries a spelling this \
         walk cannot read a name from",
        node_tag.as_str()
    ))
}

/// The name an UNSTAMPED binder-position child spells.
fn binder_child_name(child: &Expr) -> Option<String> {
    match child {
        Expr::Atom(Atom::Name(name), _) => Some(name.clone()),
        Expr::MetaExpr(meta, _) => binder_child_name(&meta.expr),
        Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .map(str::to_string),
        Expr::BareList(items, _) => items.first().and_then(symbol_name).map(str::to_string),
        _ => None,
    }
}
/// A `(var {} name)` reference, the node a hoisted projection leaves behind.
fn var_expr_node(name: &str, span: chelis_deep::span::Span) -> Expr {
    Expr::Node(
        Box::new(chelis_deep::node::Node::new(
            DeepTag::Var,
            chelis_deep::Metadata::default(),
            vec![Expr::Atom(Atom::Name(name.to_string()), span)],
        )),
        span,
    )
}

/// True when this child is a `let` binding's value AND that value is itself
/// the bare projection: `q = inp.q` already binds the field to a local, so
/// hoisting it would add a second alias and change the emitted code for a
/// spelling that lowers today.
///
/// The test is on the VALUE, not on the slot. Exempting the whole slot left
/// every projection NESTED inside a bind value unhoisted -- including
/// `a_dim = cast(shape(inp.q, cast(0, i32)), i64)`, which is the spelling
/// the section 4.7.2 diagnostic's own suggestion text asks for -- so the
/// checker admitted sizes the C lane then refused, which is the check-clean /
/// build-red divergence the whole provenance walk exists to prevent.
fn is_bound_bare_projection(node_tag: DeepTag, index: usize, child: &Expr) -> bool {
    node_tag == DeepTag::Bind
        && index % 2 == 1
        && matches!(stamped_parts(child), Some((DeepTag::Access, _, _)))
}

/// Decide whether one `(access {} base field)` expression is a tensor-typed
/// projection of a runtime record, and name the local it binds to.
fn record_projection_hoist(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    bound: &UnordSet<String>,
) -> Option<HoistedProjection> {
    let HostTypeTerm::Tensor(ty) = expr_host_type(expr, program, scope) else {
        return None;
    };
    let path = record_projection_path(expr)?;
    let base = path.first()?;
    // Rebound anywhere under this body: leave the projection where it is.
    if bound.contains(base) {
        return None;
    }
    // The base must be a host value this scope carries and NOT itself a
    // tensor: a tensor base is not a record, and a name the scope does not
    // carry is not this subtree's to lift.
    let base_ty = scope.get(base)?;
    if tensor_type_from_host_input(base_ty).is_some() {
        return None;
    }
    Some(HoistedProjection {
        path,
        // `hoisted_local_name` renders this; the path is the identity.
        name: String::new(),
        source: expr.clone(),
        ty,
    })
}

/// The `[base, field, ..]` path of a projection chain whose root is a bare
/// `var`, or `None` for any other target shape.
fn record_projection_path(expr: &Expr) -> Option<Vec<String>> {
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag == DeepTag::Var {
        return kids
            .first()
            .and_then(symbol_name)
            .map(|name| vec![name.to_string()]);
    }
    if tag != DeepTag::Access {
        return None;
    }
    let mut path = record_projection_path(kids.first()?)?;
    path.push(kids.get(1).and_then(symbol_name)?.to_string());
    Some(path)
}

// Observation changes the returned product, never helper admission or failure
// policy. Keep this guard outside both concrete lowering implementations.
fn lower_tensor_helper_with<T>(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    lower: impl FnOnce() -> Result<T, crate::lower::LowerDiagnostic>,
) -> Option<T> {
    // Do not lower a returned/dynamically-computed callable through the DAG:
    // that path beta-reduces the inner application and loses the outer call.
    // The caller falls back to host lowering, which preserves the unsupported
    // callable marker for the C ABI fence (#1951).
    if contains_computed_callable_application(expr) {
        record_host_work(|profile| profile.tensor_helper_dag_rejections += 1);
        return None;
    }
    let defs = cached_program_defs(program);
    // chelis#631: never swallow a fail-reaching FORWARD body into a
    // tensor helper. The DAG lane lowers `fail` to a mask-selected zero
    // placeholder, so a forward helper would compile into a binary that
    // returns zeros where `chelis eval` aborts with the user's message
    // (silent-wrong, the one outcome the soundness bar forbids). Bail to
    // the host lane, whose `if`/`fail` are real control flow
    // (`chelis_fail` in the C emit). Applied defs are consulted because
    // helper lowering INLINES them into the DAG. A `grad`/`vmap` subtree is
    // exempt because transformed-subtree behavior is outside chelis#662.
    // This scope guard does not endorse replacement of a taken internal
    // `fail` with a numeric placeholder; chelis#1464 owns that pre-existing
    // divergence. A sibling forward `fail` still routes the enclosing body
    // through real host control flow.
    if expr_reaches_forward_fail(expr, &defs, &mut UnordSet::new()) {
        record_host_work(|profile| profile.tensor_helper_fail_guard_rejections += 1);
        return None;
    }
    // Issue #197: surface a fatal AD rejection from the tensor-
    // helper sub-lowering instead of swallowing it; the host
    // fallback would otherwise emit an undefined-symbol call to
    // the rejected grad function.
    match lower() {
        Ok(product) => Some(product),
        Err(diagnostic) if diagnostic.fatal => {
            crate::lower::raise_fatal_lowering_diagnostic(diagnostic)
        }
        Err(_) => {
            record_host_work(|profile| profile.tensor_helper_dag_rejections += 1);
            None
        }
    }
}

fn lower_tensor_helper_dag(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    expected: &TensorType,
    context: &crate::lower::SubexprLoweringContext,
) -> Option<crate::Dag> {
    lower_tensor_helper_with(expr, program, || {
        let (dag, _) = crate::lower::try_lower_tensor_helper_program_with_ordered_inputs(
            expr,
            collect_tensor_scope(scope).into_sorted(),
            context,
            None,
            None,
            0,
            false,
        )?;
        Ok(remap_tensor_helper_dim_symbols(&dag, scope, expected))
    })
}

struct LoweredTensorHelper {
    dag: crate::Dag,
    execution: Option<crate::evaluation::ExecutionMetadata>,
    #[cfg(feature = "lowering-trace")]
    trace: Option<crate::lowering_trace::HelperLoweringTrace>,
}

fn lower_tensor_helper_product(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    expected: &TensorType,
    collect_execution: bool,
    collect_trace: bool,
    lowering_context: Option<&crate::lower::SubexprLoweringContext>,
) -> Option<LoweredTensorHelper> {
    #[cfg(not(feature = "lowering-trace"))]
    let _ = collect_trace;
    // The fixed-control execution branch below lowers directly rather than
    // passing through `lower_tensor_helper_with`. Keep the #1951 fence at
    // this common entry so neither route can beta-reduce a returned callable
    // and erase its application.
    if contains_computed_callable_application(expr) {
        record_host_work(|profile| profile.tensor_helper_dag_rejections += 1);
        return None;
    }
    let context = lowering_context
        .cloned()
        .unwrap_or_else(|| cached_subexpr_lowering_context(program));
    if collect_execution {
        let scoped = collect_tensor_scope(scope).into_sorted();
        if context.c_execution_profile(expr, &scoped)
            == crate::evaluation::EvaluationProfile::FixedControl
        {
            let planning = crate::evaluation::RandomExecutionContext::new(RandomLoweringState {
                seed: None,
                counter: 0,
            });
            #[cfg(feature = "lowering-trace")]
            let lowered = if collect_trace {
                crate::lower::try_lower_tensor_helper_c_execution_with_ordered_inputs_and_trace(
                    expr,
                    scoped,
                    &context,
                    Some(expected),
                    false,
                    &planning,
                )
                .map(|(plan, trace)| (plan, Some(trace)))
            } else {
                crate::lower::try_lower_tensor_helper_c_execution_with_ordered_inputs(
                    expr,
                    scoped,
                    &context,
                    Some(expected),
                    false,
                    &planning,
                )
                .map(|plan| (plan, None))
            };
            #[cfg(not(feature = "lowering-trace"))]
            let lowered = crate::lower::try_lower_tensor_helper_c_execution_with_ordered_inputs(
                expr,
                scoped,
                &context,
                Some(expected),
                false,
                &planning,
            )
            .map(|plan| (plan, ()));
            let (plan, _trace) = match lowered {
                Ok(lowered) => lowered,
                Err(diagnostic) if diagnostic.fatal => {
                    crate::lower::raise_fatal_lowering_diagnostic(diagnostic)
                }
                Err(_) => return None,
            };
            let rebound =
                remap_tensor_helper_dim_symbols(plan.dag_for_inspection(), scope, expected);
            let plan = plan.rebind_dimensions(rebound).unwrap_or_else(|message| {
                crate::lower::raise_fatal_lowering_diagnostic(
                    crate::lower::LowerDiagnostic::new(message, None, None).fatal(),
                )
            });
            #[cfg(feature = "lowering-trace")]
            let mut trace = _trace;
            #[cfg(feature = "lowering-trace")]
            if let Some(trace) = &mut trace {
                trace.record_dimension_rebinding(plan.dag_for_inspection());
            }
            let (dag, metadata) = plan.into_parts();
            return Some(LoweredTensorHelper {
                dag,
                execution: Some(metadata),
                #[cfg(feature = "lowering-trace")]
                trace,
            });
        }
    }
    #[cfg(feature = "lowering-trace")]
    if collect_trace {
        return lower_tensor_helper_with(expr, program, || {
            let scoped = collect_tensor_scope(scope).into_sorted();
            let (dag, _, trace) =
                crate::lower::try_lower_tensor_helper_program_with_ordered_inputs_and_trace(
                    expr, scoped, &context, None, None, 0, false,
                )?;
            let dag = remap_tensor_helper_dim_symbols(&dag, scope, expected);
            let mut trace = trace;
            trace.record_dimension_rebinding(&dag);
            Ok(LoweredTensorHelper {
                dag,
                execution: None,
                trace: Some(trace),
            })
        });
    }
    lower_tensor_helper_dag(expr, program, scope, expected, &context).map(|dag| {
        LoweredTensorHelper {
            dag,
            execution: None,
            #[cfg(feature = "lowering-trace")]
            trace: None,
        }
    })
}

fn lower_tensor_helper_dag_with_controls(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    expected: &TensorType,
) -> Option<crate::lower::LoweredSubexprWithControls> {
    let defs = cached_program_defs(program);
    if expr_reaches_forward_fail(expr, &defs, &mut UnordSet::new()) {
        record_host_work(|profile| profile.tensor_helper_fail_guard_rejections += 1);
        return None;
    }
    let context = cached_subexpr_lowering_context(program);
    let mut lowered = match crate::lower::try_lower_subexpr_program_with_context_and_controls(
        expr,
        collect_tensor_scope(scope),
        &context,
    ) {
        Ok(lowered) => lowered,
        Err(diagnostic) if diagnostic.fatal => {
            crate::lower::raise_fatal_lowering_diagnostic(diagnostic)
        }
        Err(_) => {
            record_host_work(|profile| profile.tensor_helper_dag_rejections += 1);
            return None;
        }
    };
    lowered.dag = remap_tensor_helper_dim_symbols(&lowered.dag, scope, expected);
    Some(lowered)
}

fn finish_tensor_helper_call(
    dag: crate::Dag,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected: TensorType,
) -> HostExpr {
    finish_tensor_helper_product(
        dag,
        None,
        #[cfg(feature = "lowering-trace")]
        None,
        scope,
        tensor_helpers,
        expected,
    )
}

fn finish_tensor_helper_product(
    dag: crate::Dag,
    execution: Option<crate::evaluation::ExecutionMetadata>,
    #[cfg(feature = "lowering-trace")] trace: Option<crate::lowering_trace::HelperLoweringTrace>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected: TensorType,
) -> HostExpr {
    let helper_index = tensor_helpers.len();
    let helper_name = format!("__host_tensor_helper_{helper_index}");
    let inputs = tensor_helper_inputs(&dag);
    let output = dag
        .roots()
        .first()
        .and_then(|id| dag.get(*id))
        .map(|node| node.output_type.clone())
        .unwrap_or_else(|| expected.clone());
    // Issue #309: a helper whose body has more than one DAG root (the
    // canonical case is a multi-`wrt` `grad`, which differentiates a
    // scalar w.r.t. several tensor params and so produces one gradient
    // tensor per param) returns a TUPLE of tensors, not a single
    // tensor. The DAG emit wires `roots[i]` to `outputs[i]` with
    // `n_out = roots().len()`, and a downstream `.N` projection reads
    // root `N`. Typing the call as a single `Tensor` here made the
    // projection emit `chelis_tuple_get` over a `chelis_tensor*`
    // receiver (a mistyped crash) and sized the helper output array to
    // one slot for a two-output helper. Mirror the IR/eval-lane
    // `LoweredValue::Tuple` semantics by typing the multi-root call as
    // a `Tuple` of the per-root tensor types, in root order.
    let root_tys: Vec<HostTypeTerm> = dag
        .roots()
        .iter()
        .map(|id| {
            dag.get(*id)
                .map(|node| HostTypeTerm::Tensor(node.output_type.clone()))
                .unwrap_or_else(|| HostTypeTerm::Tensor(expected.clone()))
        })
        .collect();
    let call_ty = match root_tys.as_slice() {
        [only] => only.clone(),
        [] => HostTypeTerm::Tensor(expected.clone()),
        _ => HostTypeTerm::Tuple(root_tys),
    };
    let args = tensor_helper_args(&inputs, scope);
    let (sparse_specialization, sparse_rejection) =
        match try_summarize_sparse_helper(&dag, &inputs, &output) {
            Ok(spec) => (Some(spec), None),
            Err(SparseSummaryAttempt::NotEligible) => (None, None),
            Err(SparseSummaryAttempt::Rejected(rejection)) => (None, Some(rejection)),
        };
    // W6 Task A — drive the BLAS recognizer through the structured
    // entry point so a BLAS-near rejection threads through to
    // `summary_rejection` as a `Blas*` `SummaryRejection` (rather
    // than the prior silent `Option::None` drop).
    let (blas_specialization, blas_rejection) =
        match try_summarize_blas_helper(&dag, &inputs, &output) {
            Ok(spec) => (Some(HostTensorSpecialization::BlasMatmul(spec)), None),
            Err(BlasSummaryAttempt::NotEligible) => (None, None),
            Err(BlasSummaryAttempt::Rejected(rejection)) => (None, Some(rejection)),
        };
    let specialization = blas_specialization.or(sparse_specialization);
    // Reconcile sparse vs BLAS rejections:
    //
    //   * If either recognizer accepted the helper, no rejection
    //     should be reported on this helper (the specialization
    //     takes over).
    //   * If sparse rejected, the helper body had a sparse op —
    //     that's the more specific signal; report the sparse
    //     rejection.
    //   * If sparse said NotEligible (no sparse op anywhere) and
    //     BLAS rejected, report the BLAS rejection.
    //   * If neither recognizer reached the rejection arm
    //     (both NotEligible), report nothing.
    let summary_rejection = if specialization.is_some() {
        None
    } else {
        sparse_rejection.or(blas_rejection)
    };
    tensor_helpers.push_helper(
        HostTensorHelper {
            name: helper_name,
            dag,
            inputs,
            output,
            specialization,
            summary_rejection,
        },
        execution,
        #[cfg(feature = "lowering-trace")]
        trace,
    );
    HostExpr::new(HostExprKind::TensorCall {
        helper: helper_index,
        args,
        ty: call_ty,
    })
}

fn lower_staged_host_plan(
    plan: &staged::HostStagedPlan,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    helpers: &mut TensorHelperSink,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let mut scope = scope.clone();
    let mut reserved = scope
        .to_sorted()
        .into_iter()
        .map(|(name, _)| name.clone())
        .collect::<BTreeSet<_>>();
    for stage in plan.stages() {
        match stage {
            staged::HostStage::Source {
                expression, output, ..
            } => {
                let mut names = UnordSet::new();
                collect_deep_var_names(expression, &mut names);
                reserved.extend(names.into_sorted());
                reserved.insert(output.clone());
            }
            staged::HostStage::Kernel { outputs, .. } => reserved.extend(outputs.iter().cloned()),
        }
    }
    let mut bindings = Vec::new();
    let mut callable_sources = BTreeMap::<String, String>::new();
    let mut callable_spans = Vec::new();
    for stage in plan.stages() {
        match stage {
            staged::HostStage::Source {
                expression,
                captures,
                output,
                ty,
            } => {
                if matches!(ty, HostTypeTerm::Fn(..)) {
                    let reference = stamped_parts(expression)
                        .filter(|(tag, _, _)| *tag == DeepTag::Var)
                        .and_then(|(_, _, kids)| kids.first().and_then(symbol_name))
                        .ok_or_else(|| {
                            host_expr_lowering_error(
                                expression,
                                "a staged callable needs a resolved definition reference",
                            )
                        })?;
                    let function = if let Some(capture) =
                        captures.iter().find(|capture| capture.binding == reference)
                    {
                        callable_sources.get(&capture.value).cloned()
                    } else {
                        find_top_level_def_named(program.exprs(), reference)
                            .map(|(name, _)| name.to_owned())
                    }
                    .ok_or_else(|| {
                        host_expr_lowering_error(
                            expression,
                            "a staged callable has no resolved definition",
                        )
                    })?;
                    if let Some(span) = expression.span_id() {
                        callable_spans.push(span);
                    }
                    callable_sources.insert(output.clone(), function);
                    scope.insert(output.clone(), ty.clone());
                    continue;
                }
                let mut captured_scope = scope.clone();
                let mut captured_bindings = Vec::new();
                let mut callable_aliases = BTreeMap::new();
                for capture in captures {
                    if let Some(function) = callable_sources.get(&capture.value) {
                        callable_aliases.insert(capture.binding.clone(), function.clone());
                        captured_scope.insert(capture.binding.clone(), capture.ty.clone());
                        continue;
                    }
                    let value_ty = scope
                        .get(&capture.value)
                        .expect("available capture")
                        .clone();
                    let mut value =
                        HostExpr::new(HostExprKind::Var(capture.value.clone(), value_ty.clone()));
                    if matches!(capture.ty, HostTypeTerm::Scalar(_))
                        && matches!(value_ty, HostTypeTerm::Tensor(_))
                    {
                        value = HostExpr::new(HostExprKind::Builtin {
                            name: "tensor_to_scalar".into(),
                            args: vec![value],
                            ty: capture.ty.clone(),
                        });
                    }
                    captured_bindings.push(HostBinding {
                        name: capture.binding.clone(),
                        display_name: None,
                        display_roots: Vec::new(),
                        ty: capture.ty.clone(),
                        value,
                    });
                    captured_scope.insert(capture.binding.clone(), capture.ty.clone());
                }
                let ty = ty.clone();
                let mut body = lower_host_expr_with_expected(
                    expression,
                    program,
                    &captured_scope,
                    helpers,
                    Some(&ty),
                )?;
                staged::resolve_callable_aliases(&mut body, &callable_aliases);
                let value = HostExpr::new(HostExprKind::Let {
                    bindings: captured_bindings,
                    body: Box::new(body),
                    ty: ty.clone(),
                });
                bindings.push(HostBinding {
                    name: output.clone(),
                    display_name: None,
                    display_roots: Vec::new(),
                    ty: ty.clone(),
                    value,
                });
                scope.insert(output.clone(), ty);
            }
            staged::HostStage::Kernel { dag, outputs } => {
                let expected = dag
                    .get(dag.roots()[0])
                    .expect("kernel root")
                    .output_type
                    .clone();
                let call = finish_tensor_helper_call(dag.clone(), &scope, helpers, expected);
                let ty = host_expr_type(&call);
                if outputs.len() == 1 {
                    bindings.push(HostBinding {
                        name: outputs[0].clone(),
                        display_name: None,
                        display_roots: Vec::new(),
                        ty: ty.clone(),
                        value: call,
                    });
                    scope.insert(outputs[0].clone(), ty);
                } else {
                    let mut tuple_name = format!("{}__tuple", outputs[0]);
                    while reserved.contains(&tuple_name) {
                        tuple_name.push('_');
                    }
                    reserved.insert(tuple_name.clone());
                    bindings.push(HostBinding {
                        name: tuple_name.clone(),
                        display_name: None,
                        display_roots: Vec::new(),
                        ty: ty.clone(),
                        value: call,
                    });
                    for (index, (output, root)) in outputs.iter().zip(dag.roots()).enumerate() {
                        let output_ty = HostTypeTerm::Tensor(
                            dag.get(*root).expect("kernel output").output_type.clone(),
                        );
                        let value = HostExpr::new(HostExprKind::Builtin {
                            name: "tuple-get".into(),
                            args: vec![
                                HostExpr::new(HostExprKind::Var(tuple_name.clone(), ty.clone())),
                                HostExpr::new(HostExprKind::Int(
                                    i64::try_from(index).expect("tuple index"),
                                )),
                            ],
                            ty: output_ty.clone(),
                        });
                        bindings.push(HostBinding {
                            name: output.clone(),
                            display_name: None,
                            display_roots: Vec::new(),
                            ty: output_ty.clone(),
                            value,
                        });
                        scope.insert(output.clone(), output_ty);
                    }
                }
            }
        }
    }
    let ty = scope.get(plan.output()).expect("planned result").clone();
    let body = HostExpr::new(HostExprKind::Var(plan.output().to_owned(), ty.clone()));
    let mut result = HostExpr::new(HostExprKind::Let {
        bindings,
        body: Box::new(body),
        ty,
    });
    for span in callable_spans {
        result.append_merged_span(Some(span));
    }
    Ok(result)
}

/// Outcome of the structured BLAS-helper recognizer (W6 Task A).
/// Mirrors `SparseSummaryAttempt`: distinguishes "not even a BLAS
/// helper" (silent skip) from "near-eligible but rejected for a
/// specific structural reason" (emit a diagnostic).
///
/// The `NotEligible` arm is treated as a non-error skip by the outer
/// summary-derivation pass; the `Rejected` arm carries a
/// `HelperSummaryRejection` that is promoted to a fully-formed
/// `SummaryRejection` once the owning function's name + callsite span
/// are known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlasSummaryAttempt {
    /// Helper body's post-specialize root is not (anywhere near) a
    /// `BlasMatmul` op — i.e. there is no matmul-shape subgraph for
    /// the BLAS recognizer to fold. Today this fires when the
    /// specialized DAG has zero roots (empty body or fully-DCE'd
    /// body). No diagnostic should be emitted.
    NotEligible,
    /// Helper body had at least one specialized root that the BLAS
    /// recognizer attempted to match, but the structural check
    /// failed. The carried rejection identifies the failure class.
    Rejected(HelperSummaryRejection),
}

/// Pub-test entry point for the structured-rejection-aware BLAS
/// summarizer. Mirrors `try_summarize_sparse_helper_for_test`.
///
/// IR-level tests in `crates/chelis-ir/tests/host_blas_summary_diagnostics.rs`
/// drive synthetic helper DAGs through this entry point to lock the
/// six BLAS rejection variants without going through the full
/// `try_lower_compiled_program` pipeline.
#[doc(hidden)]
pub fn try_summarize_blas_helper_for_test(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Result<HostBlasMatmulSummary, BlasSummaryAttempt> {
    try_summarize_blas_helper(dag, inputs, output)
}

/// Derive a BLAS-matmul summary for a helper whose specialized DAG is
/// a single `RiscOp::BlasMatmul` root whose operands are direct
/// `RiscOp::Load`s referencing helper inputs.
///
/// Rejection cases — each maps to a `SummaryRejectionClass::Blas*`
/// variant (the six W6 Task A variants):
///
///   * `BlasOutputPrecisionMismatch` — helper output precision is not `f32`
///   * `BlasMultipleRoots` — specialized DAG has more than one root
///   * `BlasNotMatmulPattern` — root op is not `BlasMatmul`, or its
///     operand count / output precision doesn't match the BlasMatmul
///     shape
///   * `BlasNonLoadOperand` — a matmul operand is not a direct `Load`
///   * `BlasInputPrecisionMismatch` — a helper input has precision
///     other than `f32`
///   * `BlasDimensionBindingFailure` — a matmul dim (batch/M/N/K)
///     cannot be bound to any helper input
///
/// The pre-eligibility check that produces `NotEligible` (silent skip,
/// not a diagnostic) fires when the specialized DAG has zero roots —
/// the body had nothing for the BLAS recognizer to look at.
fn try_summarize_blas_helper(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Result<HostBlasMatmulSummary, BlasSummaryAttempt> {
    let specialized = crate::specialize::specialize_for_blas(dag);
    if specialized.roots().is_empty() {
        return Err(BlasSummaryAttempt::NotEligible);
    }
    // Helper-body span: prefer the specialized root's span; fall back
    // to the pre-specialize root's span; otherwise None.
    let body_span = specialized
        .roots()
        .first()
        .and_then(|id| specialized.get(*id))
        .and_then(|n| n.span_id.clone())
        .or_else(|| {
            dag.roots()
                .first()
                .and_then(|id| dag.get(*id))
                .and_then(|n| n.span_id.clone())
        });
    // Pre-eligibility: was the helper body matmul-near at all? If
    // the specialized root is neither `BlasMatmul` (the accepted
    // shape) nor a `Sum(Mul(Expand, Expand))` pattern (the
    // matmul-near shape that `specialize_for_blas` keeps as-is when
    // it cannot replace, e.g. non-F32 precision), the recognizer
    // should NOT emit a diagnostic; this is just a non-BLAS helper.
    //
    // `is_matmul_near` returns `true` for both BLAS-shaped and
    // matmul-pattern-shaped specialized roots, so we can distinguish
    // "near-eligible BLAS helper" from "totally unrelated helper".
    if specialized.roots().len() == 1 {
        let only_root = specialized.roots()[0];
        if let Some(root_node) = specialized.get(only_root)
            && !is_matmul_near(&specialized, root_node)
        {
            return Err(BlasSummaryAttempt::NotEligible);
        }
    }
    if specialized.roots().len() != 1 {
        // Even a multi-root helper qualifies as "BLAS-near" only if
        // at least one root is matmul-shape; otherwise it's an
        // unrelated multi-output helper and we silently skip.
        let any_matmul_near = specialized
            .roots()
            .iter()
            .filter_map(|id| specialized.get(*id))
            .any(|n| is_matmul_near(&specialized, n));
        if !any_matmul_near {
            return Err(BlasSummaryAttempt::NotEligible);
        }
        return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::BlasMultipleRoots,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::BlasMultipleRoots {
                root_count: specialized.roots().len(),
            },
        }));
    }
    // From here we are committed: the specialized DAG is BLAS-near
    // and has exactly one root. Output-precision is the next gate.
    // WS-A1/A2/A3 lift: f32 (sgemm), f64 (dgemm), bf16/f16 (hipblasGemmEx
    // with f32 accumulator) are all admitted; integer matmul is rejected
    // upstream at the type checker per spec §5.7.2.
    if !matches!(
        output.precision,
        Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16
    ) {
        return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::BlasOutputPrecisionMismatch,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::BlasOutputPrecisionMismatch {
                observed: output.precision,
            },
        }));
    }
    let root = *specialized
        .roots()
        .first()
        .expect("checked roots().len() == 1 above");
    let root_node = match specialized.get(root) {
        Some(node) => node,
        None => return Err(BlasSummaryAttempt::NotEligible),
    };
    let (batch_dims, m, n, k) = match &root_node.op {
        RiscOp::BlasMatmul {
            batch_dims,
            m,
            n,
            k,
            ..
        } => (batch_dims.clone(), m.clone(), n.clone(), k.clone()),
        other => {
            return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
                rejection_class: SummaryRejectionClass::BlasNotMatmulPattern,
                helper_body_span: body_span,
                detail: SummaryRejectionDetail::BlasNotMatmulPattern {
                    tail_op: risc_op_canonical_name(other).to_string(),
                },
            }));
        }
    };
    let root_precision_admitted = matches!(
        root_node.output_type.precision,
        Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16
    );
    if !root_precision_admitted || root_node.inputs.len() != 2 {
        // Root IS BlasMatmul but its rank/precision doesn't match
        // the recognized shape. Still a BlasNotMatmulPattern
        // rejection; the variant name covers both "wrong op" and
        // "right op, wrong shape".
        return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::BlasNotMatmulPattern,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::BlasNotMatmulPattern {
                tail_op: "blas_matmul".to_string(),
            },
        }));
    }
    let lhs_input = match helper_load_input_index(&specialized, root_node.inputs[0], inputs) {
        Some(idx) => idx,
        None => {
            return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
                rejection_class: SummaryRejectionClass::BlasNonLoadOperand,
                helper_body_span: body_span,
                detail: SummaryRejectionDetail::BlasNonLoadOperand { operand_index: 0 },
            }));
        }
    };
    let rhs_input = match helper_load_input_index(&specialized, root_node.inputs[1], inputs) {
        Some(idx) => idx,
        None => {
            return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
                rejection_class: SummaryRejectionClass::BlasNonLoadOperand,
                helper_body_span: body_span,
                detail: SummaryRejectionDetail::BlasNonLoadOperand { operand_index: 1 },
            }));
        }
    };
    let input_tys = inputs
        .iter()
        .map(|input| input.ty.clone())
        .collect::<Vec<_>>();
    // WS-A1/A2/A3: admit f32 (sgemm), f64 (dgemm), bf16/f16 (hipblasGemmEx
    // with f32 accumulator). Integer matmul is rejected upstream at the
    // type checker per spec §5.7.2 so it never reaches here.
    if let Some((input_index, ty)) = input_tys
        .iter()
        .enumerate()
        .find(|(_, ty)| !matches!(ty.precision, Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16))
    {
        return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::BlasInputPrecisionMismatch,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::BlasInputPrecisionMismatch {
                input_index,
                observed: ty.precision,
            },
        }));
    }
    if !summary_dims_bind_to_inputs(&input_tys, &batch_dims) {
        return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::BlasDimensionBindingFailure,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::BlasDimensionBindingFailure {
                role: BlasDimRole::Batch,
            },
        }));
    }
    for (dim, role) in [
        (&m, BlasDimRole::M),
        (&n, BlasDimRole::N),
        (&k, BlasDimRole::K),
    ] {
        if !summary_dims_bind_to_inputs(&input_tys, std::slice::from_ref(dim)) {
            return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
                rejection_class: SummaryRejectionClass::BlasDimensionBindingFailure,
                helper_body_span: body_span,
                detail: SummaryRejectionDetail::BlasDimensionBindingFailure { role },
            }));
        }
    }
    Ok(HostBlasMatmulSummary {
        lhs_input,
        rhs_input,
        input_tys,
        output: output.clone(),
        batch_dims,
        m,
        n,
        k,
    })
}

/// Pub-test entry point for the sparse summarizer. Wraps the
/// crate-private `summarize_sparse_helper_from_parts` so integration
/// tests in `crates/chelis-ir/tests/` can lock the recognizer
/// directly without driving a full `try_lower_compiled_program` pipeline.
///
/// Surface code today has no path that lowers to `RiscOp::ScatterAdd`
/// (that op is produced exclusively by AD adjoint of `gather`), so
/// the integration test that exercises ScatterAdd-helper recognition
/// constructs a synthetic helper DAG and calls through here.
#[doc(hidden)]
pub fn summarize_sparse_helper_for_test(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Option<HostTensorSpecialization> {
    summarize_sparse_helper_from_parts(dag, inputs, output)
}

/// Pub-test entry point for the structured-rejection-aware sparse
/// summarizer. Mirrors `summarize_sparse_helper_for_test` but exposes
/// the W4-A `SparseSummaryAttempt` result so IR-level tests can lock
/// the structured rejection class + detail for each near-eligible
/// rejection case.
#[doc(hidden)]
pub fn try_summarize_sparse_helper_for_test(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Result<HostTensorSpecialization, SparseSummaryAttempt> {
    try_summarize_sparse_helper(dag, inputs, output)
}

/// Outcome of the structured sparse-helper recognizer. Distinguishes
/// "not even a sparse helper" (silent skip) from "near-eligible but
/// rejected for a specific structural reason" (emit a diagnostic).
///
/// The `NotEligible` arm is treated as a non-error skip by the
/// outer summary-derivation pass; the `Rejected` arm carries a
/// `HelperSummaryRejection` that will be promoted to a fully-formed
/// `SummaryRejection` once the owning function's name + callsite
/// span are known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseSummaryAttempt {
    /// Helper body contains no sparse RiscOp anywhere. Not a
    /// near-summary case; no diagnostic should be emitted. The outer
    /// pass falls through to the BLAS recognizer.
    NotEligible,
    /// Helper body has a sparse RiscOp in a position where the
    /// summarizer attempts recognition, but the structural check
    /// failed. The carried rejection identifies the failure class.
    Rejected(HelperSummaryRejection),
}

/// Back-compat helper around `try_summarize_sparse_helper` that
/// collapses both error arms to `None`. Used by the existing
/// `summarize_sparse_helper_for_test` IR-level test entry point and
/// by any code path that does not consume the structured rejection.
fn summarize_sparse_helper_from_parts(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Option<HostTensorSpecialization> {
    try_summarize_sparse_helper(dag, inputs, output).ok()
}

fn literal_result_claim_owner_input<'a>(
    dag: &'a crate::Dag,
    root: &'a crate::dag::DagNode,
    literal_result_claim_producers: &[bool],
) -> Option<&'a crate::dag::DagNode> {
    if !matches!(root.op, RiscOp::Copy)
        || root.inputs.len() != 1
        || root.shape_deps.is_empty()
        || !root.shape_deps.iter().all(|dependency| {
            matches!(
                dag.get(*dependency).map(|node| &node.op),
                Some(RiscOp::ExtentWitness {
                    site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
                    ..
                })
            ) || literal_result_claim_producers
                .get(dependency.0)
                .copied()
                .unwrap_or(false)
        })
    {
        return None;
    }
    let input = dag.get(root.inputs[0])?;
    (input.output_type == root.output_type).then_some(input)
}

/// Derive a sparse-op summary for a helper whose DAG is a single
/// `RiscOp::Gather`, `RiscOp::ScatterAdd`, or `RiscOp::Scatter` root
/// whose operands are direct `RiscOp::Load`s referencing helper
/// inputs. When the helper body contains a sparse op but the
/// recognizer rejects, the returned `Err(SparseSummaryAttempt::Rejected(...))`
/// carries a structured rejection class + detail per W4-A; consumers
/// programmatically match on the class rather than on rendered
/// strings.
///
/// Rejection cases — each maps to a `SummaryRejectionClass` variant:
///
/// * `MultipleRoots` — helper body has more than one DAG root.
/// * `WildcardDim` — helper input/output carries a `Named("*", None)` wildcard.
/// * `PostProcessingAfterSparseOp` — the semantic root op is not sparse but a
///   sparse op appears in the body. A compiler-internal `Copy` whose only
///   shape dependencies are literal-result declaration tokens is an ownership
///   carrier, not a semantic root.
/// * `NonLoadOperand` — a sparse-op operand is not a direct `RiscOp::Load`, or
///   its `Load` does not match a helper input by name + type (rank / type
///   mismatch surfaces here until a dedicated `RankMismatch` check is added).
/// * `IndicesDTypeMismatch` — indices precision is not i32 / i64.
/// * `PayloadDTypeMismatch` — values / target / updates precision disagrees
///   with output precision.
fn try_summarize_sparse_helper(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Result<HostTensorSpecialization, SparseSummaryAttempt> {
    if dag.is_empty() {
        return Err(SparseSummaryAttempt::NotEligible);
    }
    // Pre-scan: does the DAG mention any sparse op at all? If not, the
    // recognizer is not the right code path; the BLAS recognizer (or
    // no specialization) takes over silently.
    let deepest_sparse_op = dag
        .nodes()
        .iter()
        .find_map(|node| sparse_op_kind(&node.op).map(|kind| (kind, node)));
    if deepest_sparse_op.is_none() {
        return Err(SparseSummaryAttempt::NotEligible);
    }
    let (deepest_op_kind, _deepest_node) = deepest_sparse_op.unwrap();

    // Helper-body span: prefer the deepest sparse node's span; fall
    // back to the helper root's span; otherwise None.
    let body_span = dag
        .roots()
        .first()
        .and_then(|id| dag.get(*id))
        .and_then(|n| n.span_id.clone())
        .or_else(|| {
            dag.nodes()
                .iter()
                .find_map(|n| sparse_op_kind(&n.op).and(n.span_id.clone()))
        });

    if dag.roots().len() != 1 {
        return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::MultipleRoots,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::MultipleRoots {
                root_count: dag.roots().len(),
            },
        }));
    }
    let root_id = *dag
        .roots()
        .first()
        .expect("checked roots().len() == 1 above");
    let returned_root = match dag.get(root_id) {
        Some(node) => node,
        None => return Err(SparseSummaryAttempt::NotEligible),
    };
    if returned_root.output_type != *output {
        // Output-type mismatch is a structural shape issue — treat as
        // NotEligible to avoid emitting a diagnostic for cases that
        // are routed through a different specialization path.
        return Err(SparseSummaryAttempt::NotEligible);
    }
    if let Some(location) = wildcard_location(output, inputs) {
        return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::WildcardDim,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::WildcardDim { location },
        }));
    }
    let input_tys = inputs.iter().map(|i| i.ty.clone()).collect::<Vec<_>>();
    let literal_result_claim_producers = crate::axis_sources::literal_result_claim_producers(dag);
    let mut root = returned_root;
    while let Some(input) =
        literal_result_claim_owner_input(dag, root, &literal_result_claim_producers)
    {
        root = input;
    }

    // The semantic root must itself be a sparse op. The lowerer may install a
    // chain of same-typed Copies above it solely to retain a literal-result
    // producer and its token through nested lowering boundaries; the helper
    // above peels only those exact carriers. Any ordinary Copy or other op
    // remains post-processing.
    let root_sparse_kind = sparse_op_kind(&root.op);
    if root_sparse_kind.is_none() {
        return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::PostProcessingAfterSparseOp,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::PostProcessingAfterSparseOp {
                op: deepest_op_kind,
                tail_op: risc_op_canonical_name(&root.op).to_string(),
            },
        }));
    }
    let root_kind = root_sparse_kind.expect("root_sparse_kind is Some by guard");

    match &root.op {
        RiscOp::Gather { axis } => {
            // Inputs: [values, indices].
            if root.inputs.len() != 2 {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::NonLoadOperand,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::NonLoadOperand {
                        op: root_kind,
                        operand_index: root.inputs.len().min(1),
                    },
                }));
            }
            let values_idx = match helper_load_input_index(dag, root.inputs[0], inputs) {
                Some(i) => i,
                None => {
                    return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                        rejection_class: SummaryRejectionClass::NonLoadOperand,
                        helper_body_span: body_span,
                        detail: SummaryRejectionDetail::NonLoadOperand {
                            op: root_kind,
                            operand_index: 0,
                        },
                    }));
                }
            };
            let indices_idx = match helper_load_input_index(dag, root.inputs[1], inputs) {
                Some(i) => i,
                None => {
                    return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                        rejection_class: SummaryRejectionClass::NonLoadOperand,
                        helper_body_span: body_span,
                        detail: SummaryRejectionDetail::NonLoadOperand {
                            op: root_kind,
                            operand_index: 1,
                        },
                    }));
                }
            };
            let values_ty = match dag.get(root.inputs[0]) {
                Some(n) => n.output_type.clone(),
                None => return Err(SparseSummaryAttempt::NotEligible),
            };
            let indices_ty = match dag.get(root.inputs[1]) {
                Some(n) => n.output_type.clone(),
                None => return Err(SparseSummaryAttempt::NotEligible),
            };
            if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::IndicesDTypeMismatch,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::IndicesDTypeMismatch {
                        op: root_kind,
                        observed: indices_ty.precision,
                    },
                }));
            }
            if values_ty.precision != output.precision {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::PayloadDTypeMismatch,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::PayloadDTypeMismatch {
                        op: root_kind,
                        which: PayloadRole::Values,
                        expected: output.precision,
                        observed: values_ty.precision,
                    },
                }));
            }
            Ok(HostTensorSpecialization::SparseGather(
                HostSparseOpSummary {
                    axis: *axis,
                    input_indices: vec![values_idx, indices_idx],
                    input_tys,
                    output: output.clone(),
                },
            ))
        }
        RiscOp::ScatterAdd { axis } | RiscOp::Scatter { axis } => {
            // Inputs: [target, indices, updates].
            if root.inputs.len() != 3 {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::NonLoadOperand,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::NonLoadOperand {
                        op: root_kind,
                        operand_index: root.inputs.len().min(2),
                    },
                }));
            }
            let target_idx = match helper_load_input_index(dag, root.inputs[0], inputs) {
                Some(i) => i,
                None => {
                    return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                        rejection_class: SummaryRejectionClass::NonLoadOperand,
                        helper_body_span: body_span,
                        detail: SummaryRejectionDetail::NonLoadOperand {
                            op: root_kind,
                            operand_index: 0,
                        },
                    }));
                }
            };
            let indices_idx = match helper_load_input_index(dag, root.inputs[1], inputs) {
                Some(i) => i,
                None => {
                    return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                        rejection_class: SummaryRejectionClass::NonLoadOperand,
                        helper_body_span: body_span,
                        detail: SummaryRejectionDetail::NonLoadOperand {
                            op: root_kind,
                            operand_index: 1,
                        },
                    }));
                }
            };
            let updates_idx = match helper_load_input_index(dag, root.inputs[2], inputs) {
                Some(i) => i,
                None => {
                    return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                        rejection_class: SummaryRejectionClass::NonLoadOperand,
                        helper_body_span: body_span,
                        detail: SummaryRejectionDetail::NonLoadOperand {
                            op: root_kind,
                            operand_index: 2,
                        },
                    }));
                }
            };
            let target_ty = match dag.get(root.inputs[0]) {
                Some(n) => n.output_type.clone(),
                None => return Err(SparseSummaryAttempt::NotEligible),
            };
            let indices_ty = match dag.get(root.inputs[1]) {
                Some(n) => n.output_type.clone(),
                None => return Err(SparseSummaryAttempt::NotEligible),
            };
            let updates_ty = match dag.get(root.inputs[2]) {
                Some(n) => n.output_type.clone(),
                None => return Err(SparseSummaryAttempt::NotEligible),
            };
            if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::IndicesDTypeMismatch,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::IndicesDTypeMismatch {
                        op: root_kind,
                        observed: indices_ty.precision,
                    },
                }));
            }
            if target_ty.precision != output.precision {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::PayloadDTypeMismatch,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::PayloadDTypeMismatch {
                        op: root_kind,
                        which: PayloadRole::Target,
                        expected: output.precision,
                        observed: target_ty.precision,
                    },
                }));
            }
            if updates_ty.precision != output.precision {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::PayloadDTypeMismatch,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::PayloadDTypeMismatch {
                        op: root_kind,
                        which: PayloadRole::Updates,
                        expected: output.precision,
                        observed: updates_ty.precision,
                    },
                }));
            }
            let summary = HostSparseOpSummary {
                axis: *axis,
                input_indices: vec![target_idx, indices_idx, updates_idx],
                input_tys,
                output: output.clone(),
            };
            Ok(match &root.op {
                RiscOp::ScatterAdd { .. } => HostTensorSpecialization::SparseScatterAdd(summary),
                RiscOp::Scatter { .. } => HostTensorSpecialization::SparseScatterReplace(summary),
                _ => unreachable!("matched arm guarantees op kind"),
            })
        }
        _ => unreachable!("root_sparse_kind.is_some() guard rules out non-sparse roots"),
    }
}

/// `true` when `root_node` is the root of a matmul-near subgraph in
/// the specialized DAG. Used by the BLAS recognizer's
/// pre-eligibility check (W6 Task A) to distinguish "this helper's
/// body has a matmul shape that the recognizer attempted to fold"
/// from "this helper is totally unrelated to matmul".
///
/// Returns `true` for two shapes:
///   * `RiscOp::BlasMatmul` — the post-specialize accepted shape
///   * `RiscOp::Sum` whose sole input is `RiscOp::Mul` of two
///     `RiscOp::Expand`s — the matmul-pattern shape that
///     `specialize_for_blas` leaves as-is when it cannot replace
///     (e.g. when `detect_matmul_pattern` rejects on non-F32
///     precision per the W5 P0 fix).
///
/// Returns `false` for everything else (elementwise helpers,
/// pure-sparse helpers, etc.). Those produce `NotEligible` rather
/// than a structured rejection.
fn is_matmul_near(dag: &crate::Dag, root_node: &crate::DagNode) -> bool {
    if matches!(&root_node.op, RiscOp::BlasMatmul { .. }) {
        return true;
    }
    if !matches!(&root_node.op, RiscOp::Sum { .. }) {
        return false;
    }
    if root_node.inputs.len() != 1 {
        return false;
    }
    let Some(mul_node) = dag.get(root_node.inputs[0]) else {
        return false;
    };
    if !matches!(&mul_node.op, RiscOp::Mul) || mul_node.inputs.len() != 2 {
        return false;
    }
    let Some(expand_a) = dag.get(mul_node.inputs[0]) else {
        return false;
    };
    let Some(expand_b) = dag.get(mul_node.inputs[1]) else {
        return false;
    };
    matches!(&expand_a.op, RiscOp::Expand { .. }) && matches!(&expand_b.op, RiscOp::Expand { .. })
}

/// Map a `RiscOp` to a `SparseOpKind`. Returns `None` for non-sparse
/// ops. Used to detect "near-eligible" helpers and to populate
/// rejection-class details with the specific sparse op that was
/// rejected.
fn sparse_op_kind(op: &RiscOp) -> Option<SparseOpKind> {
    match op {
        RiscOp::Gather { .. } => Some(SparseOpKind::Gather),
        RiscOp::ScatterAdd { .. } => Some(SparseOpKind::ScatterAdd),
        RiscOp::Scatter { .. } => Some(SparseOpKind::ScatterReplace),
        _ => None,
    }
}

/// Canonical snake-case name for a `RiscOp` for use in
/// `SummaryRejectionDetail::PostProcessingAfterSparseOp::tail_op`.
/// Mirrors the user-facing Surf builtin name where one exists. The
/// surface is intentionally a small whitelist of "tail ops that
/// commonly appear above a rejected sparse op in user code"; opaque
/// `<other>` is the safe default for everything else.
fn risc_op_canonical_name(op: &RiscOp) -> &'static str {
    match op {
        RiscOp::Add => "add",
        RiscOp::Sub => "sub",
        RiscOp::Mul => "mul",
        RiscOp::MaxElem => "max_elem",
        RiscOp::MinElem => "min_elem",
        RiscOp::Relu => "relu",
        RiscOp::ReluAdjoint => "relu_adjoint",
        RiscOp::Neg => "neg",
        RiscOp::Abs => "abs",
        RiscOp::Reshape { .. } => "reshape",
        RiscOp::Expand { .. } => "expand",
        RiscOp::Cast { .. } => "cast",
        RiscOp::CastTrunc { .. } => "cast_trunc",
        RiscOp::Permute { .. } => "permute",
        RiscOp::Load { .. } => "load",
        RiscOp::Const { .. } => "const",
        RiscOp::BlasMatmul { .. } => "blas_matmul",
        RiscOp::Gather { .. } => "gather",
        RiscOp::ScatterAdd { .. } => "scatter_add",
        RiscOp::Scatter { .. } => "scatter_replace",
        RiscOp::Sum { .. } => "sum",
        RiscOp::Count { .. } => "count",
        RiscOp::Copy => "copy",
        RiscOp::Drop => "drop",
        RiscOp::Realize => "realize",
        // Fallback: opaque rather than panicking, because the
        // canonical-name surface is exhaustive for the ops the
        // recognizer cares about but not for every RiscOp.
        _ => "<other>",
    }
}

/// Returns the location of the first `Named("*", None)` wildcard dim
/// in (output, inputs[0], inputs[1], ...), or `None` if no wildcard
/// is present. Output is checked first so that "wildcard in output"
/// is reported preferentially.
fn wildcard_location(output: &TensorType, inputs: &[HostTensorInput]) -> Option<WildcardLocation> {
    if tensor_type_has_wildcard_dim(output) {
        return Some(WildcardLocation::Output);
    }
    for (idx, input) in inputs.iter().enumerate() {
        if tensor_type_has_wildcard_dim(&input.ty) {
            return Some(WildcardLocation::Input(idx));
        }
    }
    None
}

/// `true` when `ty` carries at least one wildcard dim
/// (`Named("*", None)`). Wildcards survive from type inference when a
/// dim was unconstrained at the use site and were never bound to a
/// concrete or symbolic axis. They make summary-derived contract
/// assertions meaningless because all wildcards in a helper share the
/// same string name and would falsely collapse to one axis.
fn tensor_type_has_wildcard_dim(ty: &TensorType) -> bool {
    ty.dims
        .iter()
        .any(|dim| matches!(dim, DimInfo::Named(name, None) if name == "*"))
}

fn helper_load_input_index(
    dag: &crate::Dag,
    id: crate::dag::NodeId,
    inputs: &[HostTensorInput],
) -> Option<usize> {
    let node = dag.get(id)?;
    let RiscOp::Load { name } = &node.op else {
        return None;
    };
    inputs
        .iter()
        .position(|input| input.name == name.as_str() && input.ty == node.output_type)
}

fn summary_dims_bind_to_inputs(input_tys: &[TensorType], dims: &[DimExpr]) -> bool {
    let available = input_tys
        .iter()
        .flat_map(|ty| ty.dims.iter())
        .filter_map(|dim| match dim {
            DimInfo::Named(name, None) => Some(name.as_str()),
            _ => None,
        })
        .collect::<UnordSet<_>>();
    dims.iter()
        .flat_map(|dim| dim.symbolic_names().into_sorted())
        .all(|name| available.contains(name.as_str()))
}

fn lower_host_expr(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    lower_host_expr_with_expected(expr, program, scope, tensor_helpers, None)
}

/// `lower_host_expr_with_expected` for callers that already hold an
/// `Option` (chelis#1201). Keeps the match-arm call sites readable rather
/// than repeating the `match expected_ty` at each one.
fn lower_host_expr_with_expected_opt(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected_ty: Option<&HostTypeTerm>,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    lower_host_expr_with_expected(expr, program, scope, tensor_helpers, expected_ty)
}

fn lower_host_expr_with_expected(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected_ty: Option<&HostTypeTerm>,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let _preflight_guard = TensorHelperPreflightGuard::begin_if_uncovered(expr, program);
    let _profile_guard = enter_host_expr_profile();
    let mut result = lower_host_expr_kind(expr, program, scope, tensor_helpers, expected_ty)?;
    // Per `spec/design/chelis_span_survival.md` §2.3 host-side table,
    // rule "Lowering": every freshly-produced HostExpr inherits the
    // enclosing Deep `Expr`'s `meta["span"]` as its `span_id`. When the
    // result already carries a different `span_id` (because lowering
    // recursed into a child whose own span was attached first — the N→1
    // body-collapse case), the parent's span appends to `merged_spans`,
    // lex-sorted and deduped, so the audit chain doesn't drop it. The
    // helper handles the no-op cases (None, equal canonical, already
    // present).
    if let Some(span) = expr.span_id() {
        if result.span_id.is_none() {
            result.span_id = Some(span.to_owned());
        } else {
            result.append_merged_span(Some(span));
        }
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct SequentialHostLetBinding<'expr> {
    binding: &'expr Expr,
    initializer: &'expr Expr,
    bind_span: Option<&'expr str>,
    layer: usize,
}

struct SequentialHostLetLayer {
    span_id: Option<String>,
    explicit_ty: HostTypeTerm,
}

struct SequentialHostLetChain<'expr> {
    bindings: Vec<SequentialHostLetBinding<'expr>>,
    layers: Vec<SequentialHostLetLayer>,
    bind_expr: Expr,
    body: &'expr Expr,
}

/// Flatten only the direct tail chain produced for sequential Surf bindings:
/// `(let b0 (let b1 (... body)))`. Branches and other nested expressions are
/// not traversed. This gives a checked local ascription visibility over the
/// same lexical predecessors that source evaluation gives it.
fn sequential_host_let_chain<'expr>(
    expr: &'expr Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
) -> Option<SequentialHostLetChain<'expr>> {
    let mut bindings = Vec::new();
    let mut layers = Vec::new();
    let mut current = expr;
    let mut layer = 0;
    let body = loop {
        let Expr::List(list, _) = current else {
            return None;
        };
        if tag(list) != Some(DeepTag::Let) {
            return None;
        }
        layers.push(SequentialHostLetLayer {
            span_id: current.span_id().map(str::to_owned),
            explicit_ty: expr_host_type(current, program, scope),
        });
        let kids = children(list);
        let bind_expr = kids.first()?;
        let bind_list = as_list(bind_expr)?;
        if tag(bind_list) != Some(DeepTag::Bind) {
            return None;
        }
        let bind_span = bind_expr.span_id();
        let bind_kids = children(bind_list);
        if !bind_kids.len().is_multiple_of(2) {
            return None;
        }
        for pair in bind_kids.as_chunks::<2>().0 {
            bindings.push(SequentialHostLetBinding {
                binding: &pair[0],
                initializer: &pair[1],
                bind_span,
                layer,
            });
        }
        let next = kids.get(1)?;
        if matches!(next, Expr::List(next_list, _) if tag(next_list) == Some(DeepTag::Let)) {
            current = next;
            layer += 1;
        } else {
            break next;
        }
    };

    let span = expr.span();
    let mut bind_elements = vec![
        Expr::Atom(Atom::Tag(DeepTag::Bind), span),
        Expr::Map(Metadata::default(), span),
    ];
    for binding in &bindings {
        bind_elements.push(binding.binding.clone());
        bind_elements.push(binding.initializer.clone());
    }
    Some(SequentialHostLetChain {
        bindings,
        layers,
        bind_expr: Expr::List(
            List {
                elements: bind_elements,
            },
            span,
        ),
        body,
    })
}

fn lower_checked_local_ascription_region(
    name: &str,
    initializer: &Expr,
    local_region: &crate::lower::LocalAscriptionBindingRegion,
    checked_lowering: &crate::lower::SubexprLoweringContext,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let region_context = checked_lowering.for_local_ascription_region(local_region);
    let expected = expr_tensor_type(initializer, program, scope)
        .or_else(|| tensor_type_from_host_input(&expr_host_type(initializer, program, scope)))
        .ok_or_else(|| {
            host_expr_lowering_error(
                initializer,
                format!(
                    "checked local tensor-ascription region rooted at `{name}` has no concrete \
                     tensor type at its initializer"
                ),
            )
        })?;

    try_lower_tensor_helper_call_with_context(
        local_region.expression(),
        program,
        scope,
        tensor_helpers,
        expected,
        &region_context,
    )
    .ok_or_else(|| {
        host_expr_lowering_error(
            initializer,
            format!(
                "checked local tensor-ascription region rooted at `{name}` cannot be represented \
                 by the tensor execution lane"
            ),
        )
    })
}

fn host_expr_lowering_error(
    expr: &Expr,
    detail: impl Into<String>,
) -> crate::lower::LowerDiagnostic {
    let (construct, authority) = match expr {
        Expr::List(list, _) if tag(list) == Some(DeepTag::Fn) => (
            "anonymous function value `fn`".to_string(),
            chelis_types::unimplemented_rejection!(
                879,
                "general C-host function values are not implemented; use a contextual \
                     callback position or run under `chelis eval`"
            ),
        ),
        Expr::List(list, _) => (
            format!(
                "Deep expression `{}`",
                tag(list).map(DeepTag::as_str).unwrap_or("<malformed>")
            ),
            chelis_types::deliberate_rejection!(
                "[04-TOT-3]",
                "a malformed or unhandled Deep form cannot lower to a substitute host value"
            ),
        ),
        Expr::Atom(_, _) => malformed_host_authority("raw Deep atom expression"),
        Expr::Map(_, _) => malformed_host_authority("raw Deep metadata map expression"),
        Expr::MetaExpr(_, _) => malformed_host_authority("wrapped Deep expression"),
        Expr::Node(node, _) => {
            if node.tag() == DeepTag::Fn {
                (
                    "anonymous function value `fn`".to_string(),
                    chelis_types::unimplemented_rejection!(
                        879,
                        "general C-host function values are not implemented; use a contextual \
                         callback position or run under `chelis eval`"
                    ),
                )
            } else {
                malformed_host_authority(&format!("Deep expression `{}`", node.tag().as_str()))
            }
        }
        Expr::BareList(_, _) => malformed_host_authority("bare list expression"),
        Expr::UnknownForm(data) => {
            malformed_host_authority(&format!("unknown form `{}`", data.head))
        }
    };
    let unsupported = chelis_types::unsupported::Unsupported::new(
        chelis_types::unsupported::UnsupportedKind::Construct(construct),
        format!("host expression lowering: {}", detail.into()),
        chelis_types::unsupported::Stage::Lowering,
        authority,
    );
    crate::lower::LowerDiagnostic::new(
        unsupported.to_string(),
        Some(expr.span()),
        expr.span_id().map(str::to_string),
    )
    .fatal()
}

fn malformed_host_authority(
    construct: &str,
) -> (String, chelis_types::unsupported::RejectionAuthority) {
    (
        construct.to_string(),
        chelis_types::deliberate_rejection!(
            "[04-TOT-3]",
            "a malformed or unhandled Deep form cannot lower to a substitute host value"
        ),
    )
}

fn lower_host_expr_kind(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected_ty: Option<&HostTypeTerm>,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let is_app_expr = matches!(expr, Expr::List(list, _) if tag(list) == Some(DeepTag::App));
    if !is_app_expr
        && let Some(tensor_ty) = expr_tensor_type(expr, program, scope)
        && !should_keep_tensor_expr_in_host_lane(expr)
        && let Some(tensor_call) =
            try_lower_tensor_helper_call(expr, program, scope, tensor_helpers, tensor_ty.clone())
    {
        return Ok(tensor_call);
    }

    let lowered = match expr {
        Expr::Atom(Atom::Int(value), _) => HostExpr::new(HostExprKind::Int(*value)),
        Expr::Atom(Atom::Float(value), _) => HostExpr::new(HostExprKind::Float(*value)),
        Expr::Atom(Atom::Bool(value), _) => HostExpr::new(HostExprKind::Bool(*value)),
        Expr::Atom(Atom::Str(value), _) => HostExpr::new(HostExprKind::String(value.clone())),
        Expr::List(list, _) if tag(list) == Some(DeepTag::Tuple) => {
            if children(list).is_empty() {
                return Ok(HostExpr::new(HostExprKind::Unit));
            }
            let items = children(list)
                .iter()
                .map(|child| lower_host_expr(child, program, scope, tensor_helpers))
                .collect::<Result<Vec<_>, _>>()?;
            let ty = expr_host_type(expr, program, scope);
            let ty = if ty.is_unresolved() {
                HostTypeTerm::Tuple(items.iter().map(host_expr_type).collect())
            } else {
                ty
            };
            HostExpr::new(HostExprKind::Tuple(items, ty))
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Record) => {
            lower_record_host_expr(list, program, scope, tensor_helpers, expected_ty)?
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Lit) => {
            let child = children(list).first().ok_or_else(|| {
                host_expr_lowering_error(expr, "a `lit` node has no literal child")
            })?;
            let value = lower_host_expr(child, program, scope, tensor_helpers)?;

            // The lexical carriers are i64/f64, but the checked `lit` owns
            // the value's width ([04-LIT-1]). Finalize before any return,
            // binding or enclosing cast: otherwise ownership sees an i64
            // return from an i32 function (#1732), or widening observes an
            // unfinalized decimal (#1110). A marked integer-source float
            // likewise casts directly from the exact integer, never via f64.
            let lexical_precision = match &value.kind {
                HostExprKind::Int(_) => Some(chelis_types::types::Prim::Int64),
                HostExprKind::Float(_) => Some(chelis_types::types::Prim::F64),
                _ => None,
            };
            if let Some(lexical_precision) = lexical_precision
                && let Some(precision) = expr_scalar_primitive(expr)
                && (precision.is_integer() || precision.is_float())
                && precision != lexical_precision
            {
                HostExpr::new(HostExprKind::Builtin {
                    name: "cast".to_string(),
                    args: vec![value],
                    ty: HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision)),
                })
            } else {
                value
            }
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Var) => {
            let name = children(list)
                .first()
                .and_then(symbol_name)
                .ok_or_else(|| host_expr_lowering_error(expr, "a `var` node has no symbol"))?
                .to_string();
            let ty = expr_host_type(expr, program, scope);
            if name == "Nil" {
                return Ok(HostExpr::new(HostExprKind::List(
                    Vec::new(),
                    match ty {
                        HostTypeTerm::List(_) => ty,
                        _ => HostTypeTerm::List(Box::new(fresh_host_inference())),
                    },
                )));
            }
            // An ordinary lexical binding wins in bare value position. Only
            // a name absent from host scope may lower as a nullary ADT
            // constructor (spec/01 §3.2; chelis#1076).
            if !scope.contains_key(&name) {
                match resolve_adt_constructor_definition_for_type(program, &name, &ty) {
                    AdtConstructorResolution::Unique(definition) if definition.is_nullary() => {
                        let instantiated =
                            definition.instantiate_nullary_term(&ty).map_err(|error| {
                                host_expr_lowering_error(
                                    expr,
                                    format!(
                                        "constructor `{name}` is not concretely instantiated: {error}"
                                    ),
                                )
                            })?;
                        return Ok(HostExpr::new(HostExprKind::AdtConstruct {
                            ctor: name,
                            fields: Vec::new(),
                            ty: instantiated.ty,
                        }));
                    }
                    // Whether this reference is a construction at all depends
                    // on which declaration answers it, so an ambiguous name
                    // cannot fall through to the variable path: that would
                    // emit a bare C identifier nothing declares and trade a
                    // rejection for a silently different lowering (chelis#730's
                    // shape). Fail closed instead.
                    AdtConstructorResolution::Ambiguous(candidates) => {
                        return Err(ambiguous_constructor_error(expr, &name, &candidates));
                    }
                    AdtConstructorResolution::Unique(_) | AdtConstructorResolution::Missing => {}
                }
            }
            HostExpr::new(HostExprKind::Var(name, ty))
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::If) => {
            let kids = children(list);
            if kids.len() != 3 {
                return Err(host_expr_lowering_error(
                    expr,
                    format!("an `if` node requires 3 children, found {}", kids.len()),
                ));
            }
            // chelis#1201: both branches are RESULT positions, so a generic
            // ADT constructed in one resolves its instantiation from the
            // caller's expected type — the same rule the match arms follow.
            // The condition is not a result position and keeps plain lowering.
            let then_expr = lower_host_expr_with_expected_opt(
                &kids[1],
                program,
                scope,
                tensor_helpers,
                expected_ty,
            )?;
            let else_expr = lower_host_expr_with_expected_opt(
                &kids[2],
                program,
                scope,
                tensor_helpers,
                expected_ty,
            )?;
            let explicit_ty = expr_host_type(expr, program, scope);
            let ty = if explicit_ty.is_unresolved() {
                let then_ty = host_expr_type(&then_expr);
                if then_ty.is_unresolved() {
                    host_expr_type(&else_expr)
                } else {
                    then_ty
                }
            } else {
                explicit_ty
            };
            HostExpr::new(HostExprKind::If {
                cond: Box::new(lower_host_expr(&kids[0], program, scope, tensor_helpers)?),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
                ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Match) => {
            lower_match_host_expr(list, program, scope, tensor_helpers, expected_ty)?
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Block) => {
            // chelis#859: sequenced expressions, value is the last child's
            // (spec/03 §2.3). Encoded as a Let whose non-last children bind
            // fresh discarded names, so effectful children (e.g. `print`)
            // still emit as statements in the generated host code, mirroring
            // the eval lane's in-order evaluation.
            let kids = children(list);
            let Some((last, init)) = kids.split_last() else {
                return Err(host_expr_lowering_error(
                    expr,
                    "a `block` node has no children",
                ));
            };
            let mut bindings = Vec::new();
            for (index, child) in init.iter().enumerate() {
                let value = lower_host_expr(child, program, scope, tensor_helpers)?;
                let ty = host_expr_type(&value);
                bindings.push(HostBinding {
                    name: format!("__chelis_block_{index}"),
                    display_name: None,
                    display_roots: Vec::new(),
                    ty,
                    value,
                });
            }
            let body = lower_host_expr(last, program, scope, tensor_helpers)?;
            let body_ty = host_expr_type(&body);
            HostExpr::new(HostExprKind::Let {
                bindings,
                body: Box::new(body),
                ty: body_ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Let) => {
            let kids = children(list);
            let mut scoped = scope.clone();
            let mut bindings = Vec::new();
            let checked_lowering = cached_subexpr_lowering_context(program);
            let flattened = sequential_host_let_chain(expr, program, scope);
            let flattened_regions = if let Some(chain) = flattened.as_ref() {
                checked_lowering.local_ascription_binding_regions(
                    &chain.bind_expr,
                    tensor_helpers.declaration_name.as_deref(),
                )
            } else {
                Vec::new()
            };
            let crosses_nested_let = flattened.as_ref().is_some_and(|chain| {
                flattened_regions.iter().any(|region| {
                    let producer_layer = chain.bindings[region.producer_binding_index() / 2].layer;
                    region
                        .ascription_binding_indices()
                        .iter()
                        .any(|index| chain.bindings[*index / 2].layer != producer_layer)
                })
            });

            let (let_bindings, bind_expr, body_expr, nested_layers) = if crosses_nested_let {
                let chain = flattened.expect("cross-layer region requires a flattened let chain");
                (
                    chain.bindings,
                    Cow::Owned(chain.bind_expr),
                    chain.body,
                    Some(chain.layers),
                )
            } else {
                let bind_first = kids.first().ok_or_else(|| {
                    host_expr_lowering_error(expr, "a `let` node has no bindings")
                })?;
                let bind_list = as_list(bind_first).ok_or_else(|| {
                    host_expr_lowering_error(expr, "a `let` node has malformed bindings")
                })?;
                if tag(bind_list) != Some(DeepTag::Bind) {
                    return Err(host_expr_lowering_error(
                        expr,
                        "a `let` node has malformed bindings",
                    ));
                }
                let bind_span = bind_first.span_id();
                let current = children(bind_list)
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| SequentialHostLetBinding {
                        binding: &pair[0],
                        initializer: &pair[1],
                        bind_span,
                        layer: 0,
                    })
                    .collect();
                (
                    current,
                    Cow::Borrowed(bind_first),
                    kids.get(1).ok_or_else(|| {
                        host_expr_lowering_error(expr, "a `let` node has no body")
                    })?,
                    None,
                )
            };
            let local_regions = checked_lowering
                .local_ascription_binding_regions(
                    bind_expr.as_ref(),
                    tensor_helpers.declaration_name.as_deref(),
                )
                .into_iter()
                .map(|region| (region.producer_binding_index(), region))
                .collect::<BTreeMap<_, _>>();
            let mut nested_bindings = nested_layers
                .as_ref()
                .map(|layers| (0..layers.len()).map(|_| Vec::new()).collect::<Vec<_>>());
            for (ordinal, binding) in let_bindings.iter().enumerate() {
                if let Some(name) = symbol_name(binding.binding) {
                    let initializer = binding.initializer;
                    let index = ordinal * 2;
                    let mut value = if let Some(local_region) = local_regions.get(&index) {
                        lower_checked_local_ascription_region(
                            name,
                            initializer,
                            local_region,
                            &checked_lowering,
                            program,
                            &scoped,
                            tensor_helpers,
                        )?
                    } else {
                        lower_host_expr(initializer, program, &scoped, tensor_helpers)?
                    };
                    // The original `(bind {span: a} ...)` node wraps this
                    // value even when a cross-let local-ascription region
                    // caused the sequential tail chain to be flattened.
                    value.append_merged_span(binding.bind_span);
                    let bind_ty = host_expr_type(&value);
                    let host_binding = HostBinding {
                        name: name.to_string(),
                        display_name: None,
                        display_roots: Vec::new(),
                        ty: bind_ty.clone(),
                        value,
                    };
                    if let Some(nested_bindings) = nested_bindings.as_mut() {
                        nested_bindings[binding.layer].push(host_binding);
                    } else {
                        bindings.push(host_binding);
                    }
                    scoped.insert(name.to_string(), bind_ty);
                }
            }
            let mut body = lower_host_expr(body_expr, program, &scoped, tensor_helpers)?;
            if let (Some(layers), Some(mut nested_bindings)) = (nested_layers, nested_bindings) {
                for (layer, layer_bindings) in
                    layers.into_iter().zip(nested_bindings.drain(..)).rev()
                {
                    let ty = if layer.explicit_ty.is_unresolved() {
                        host_expr_type(&body)
                    } else {
                        layer.explicit_ty
                    };
                    let mut nested = HostExpr::new(HostExprKind::Let {
                        bindings: layer_bindings,
                        body: Box::new(body),
                        ty,
                    });
                    if let Some(span) = layer.span_id {
                        nested.span_id = Some(span);
                    }
                    body = nested;
                }
                body
            } else {
                let explicit_ty = expr_host_type(expr, program, scope);
                HostExpr::new(HostExprKind::Let {
                    bindings,
                    body: Box::new(body.clone()),
                    ty: if explicit_ty.is_unresolved() {
                        host_expr_type(&body)
                    } else {
                        explicit_ty
                    },
                })
            }
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::TupleGet) => {
            lower_tuple_get_host_expr(list, program, scope, tensor_helpers)?
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Access) => {
            lower_access_host_expr(list, program, scope, tensor_helpers)?
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Cast) => {
            // chelis#730 Phase 1 (census row 13's host-lane half,
            // chelis#744): a cast target naming an unrecognized primitive
            // raises the same fatal branded diagnostic as the IR-lane
            // `lower_cast` - the build lane previously typed it Unknown,
            // fell back to the inferred operand type, and shipped a
            // working binary while eval rejected the same file. Raw
            // `(t-prim {} name)` and bare symbols retain name validation;
            // `t-var` targets are legal once their checked identity has
            // been actualized by the active specialization (#1418).
            // The result metadata uses checker IDs, unlike the authored
            // target spelling; expr_host_type applies the matching active
            // substitution. Never select the target from the operand.
            let ty = expr_host_type(expr, program, scope);
            let bogus_target_name =
                match children(list).get(1) {
                    Some(Expr::List(tlist, _))
                        if tag(tlist) == Some(DeepTag::TPrim)
                            && children(tlist).first().and_then(symbol_name).is_some_and(
                                |name| chelis_types::types::Prim::parse_name(name).is_none(),
                            ) =>
                    {
                        children(tlist).first().and_then(symbol_name)
                    }
                    Some(Expr::Atom(Atom::Name(name), _))
                        if chelis_types::types::Prim::parse_name(name).is_none() =>
                    {
                        Some(name.as_str())
                    }
                    _ => None,
                };
            let unresolved_target_name = ty.is_unresolved().then(|| {
                children(list)
                    .get(1)
                    .and_then(|target| {
                        symbol_name(target).or_else(|| {
                            stamped_parts(target)
                                .and_then(|(_, _, kids)| kids.first())
                                .and_then(symbol_name)
                        })
                    })
                    .unwrap_or("an unresolved cast target")
            });
            if let Some(bogus) = bogus_target_name.or(unresolved_target_name) {
                let unsupported = chelis_types::unsupported::Unsupported::new(
                    chelis_types::unsupported::UnsupportedKind::Dtype(bogus.to_string()),
                    "a `cast` target in host lowering",
                    chelis_types::unsupported::Stage::Lowering,
                    chelis_types::deliberate_rejection!(
                        "[04-DTYPE-1]",
                        "the cast target must name an active primitive type \
                         (spec/04-type-system.md section 1.1); a bogus target previously \
                         lowered as the operand type silently in the build lane \
                         (chelis#744, chelis#730 census row 13)"
                    ),
                );
                crate::lower::raise_fatal_lowering_diagnostic(
                    crate::lower::LowerDiagnostic::from_unsupported(
                        unsupported,
                        None,
                        expr.span_id().map(ToOwned::to_owned),
                    )
                    .fatal(),
                );
            }
            let operand = children(list)
                .first()
                .ok_or_else(|| host_expr_lowering_error(expr, "a `cast` node has no operand"))?;
            let mut value = lower_host_expr(operand, program, scope, tensor_helpers)?;
            if binder_float_literal_keeps_f32_source(operand, &ty) {
                value = HostExpr::new(HostExprKind::Builtin {
                    name: "cast".to_string(),
                    args: vec![value],
                    ty: HostTypeTerm::Float32,
                });
            }
            // The rung travels with the callable name so the host lane
            // and the DAG lane land on the same C guard ([05-OP-6]).
            let name = match chelis_deep::cast_mode_of(children(list)) {
                Ok(mode) => mode.keyword().to_string(),
                Err(selector) => {
                    return Err(host_expr_lowering_error(
                        expr,
                        format!("`{selector}` is not a recognized cast mode selector"),
                    ));
                }
            };
            HostExpr::new(HostExprKind::Builtin {
                name,
                args: vec![value],
                ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::App) => {
            lower_app_host_expr(expr, list, program, scope, tensor_helpers, expected_ty)?
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Pipe) => {
            // A pipe must never reach the host lowerer either. The checker's
            // input fold (`chelis_deep::pipe::fold_pipe`) replaced it with the
            // application spec/02-surf-syntax.md §0.1 says it denotes, so the
            // host emitter sees that application like every other consumer.
            // This arm existed to beta-reduce the stage itself, which is the
            // second derivation of one sentence that chelis#1923 and
            // chelis#1791 came from; it fails closed instead.
            let _ = list;
            return Err(host_expr_lowering_error(
                expr,
                "a pipe reached lowering unfolded",
            ));
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::HandleEffect) => {
            // `with seed(...) { body }` and similar effect handlers are
            // pure-result from the host emitter's perspective. Random
            // handlers still need a host-lane seed scope so calls into
            // separately emitted stdlib/helper functions see the active seed.
            let kids = children(list);
            // chelis#730 Phase 1 (census row 20; the host-lane sibling of
            // row 9, discovered during the row 9 conversion): the former
            // unconditional body-passthrough silently dropped the handler
            // for every non-`random` effect kind, including unknown ones.
            // chelis#730 Phase 2 (section C4.4): the kind is parsed once
            // into the closed [`EffectKind`] set and dispatched with an
            // exhaustive `match` (no `_` arm), so a new kind is a compile
            // error here. A decode error raises the same
            // fatal branded diagnostic as the IR-lane arm.
            let effect_kind = decode_effect_kind(list).map_err(|error| {
                let unsupported = chelis_types::unsupported::Unsupported::new(
                    chelis_types::unsupported::UnsupportedKind::EffectKind(error.to_string()),
                    "a `handle-effect` form in host lowering",
                    chelis_types::unsupported::Stage::Lowering,
                    chelis_types::deliberate_rejection!(
                        "[04-EFF-1]",
                        "known effect kinds are `random` and `resource` \
                         (spec/03-deep-syntax.md); an unknown kind previously dropped its \
                         handler silently (chelis#730 census rows 9/20)"
                    ),
                );
                crate::lower::LowerDiagnostic::from_unsupported(
                    unsupported,
                    None,
                    expr.span_id().map(ToOwned::to_owned),
                )
                .fatal()
            })?;
            let body = kids.get(1).or_else(|| kids.first()).ok_or_else(|| {
                host_expr_lowering_error(expr, "a `handle-effect` node has no body")
            })?;
            match effect_kind {
                EffectKind::Random => {
                    let seed_expr = kids.first().ok_or_else(|| {
                        host_expr_lowering_error(expr, "a random handler has no seed")
                    })?;
                    let seed = lower_host_expr(seed_expr, program, scope, tensor_helpers)?;
                    let body = lower_host_expr(body, program, scope, tensor_helpers)?;
                    let ty = host_expr_type(&body);
                    return Ok(HostExpr::new(HostExprKind::WithSeed {
                        seed: Box::new(seed),
                        body: Box::new(body),
                        ty,
                    }));
                }
                EffectKind::Resource => lower_host_expr(body, program, scope, tensor_helpers)?,
            }
        }
        Expr::MetaExpr(meta, _) => {
            lower_host_expr_with_expected(&meta.expr, program, scope, tensor_helpers, expected_ty)?
        }
        Expr::List(list, _)
            if matches!(tag(list), Some(DeepTag::Grad | DeepTag::Vmap))
                || list.unknown_tag_symbol() == Some("vmap-grad") =>
        {
            return Err(host_expr_lowering_error(
                expr,
                "a higher-order transform appeared as a value instead of a supported direct call",
            ));
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Copy) => {
            // `(copy {} x)` exists for linearity bookkeeping. In the host
            // lane, lower as a tagged Builtin whose C emission is a direct
            // tensor-copy or a pass-through for non-tensors. Without this
            // arm the `copy` tag silently fell through to `HostExpr::new(HostExprKind::Unit)`,
            // which caused tensor-if bodies in folds to collapse to
            // `int new_t; new_t = 0;` (Nautilus P2 tensor-if-in-fold).
            let inner = lower_host_expr(
                children(list).first().ok_or_else(|| {
                    host_expr_lowering_error(expr, "a `copy` node has no operand")
                })?,
                program,
                scope,
                tensor_helpers,
            )?;
            let inner_ty = host_expr_type(&inner);
            let explicit = expr_host_type(expr, program, scope);
            let ty = if explicit.is_unresolved() {
                inner_ty
            } else {
                explicit
            };
            HostExpr::new(HostExprKind::Builtin {
                name: "copy".to_string(),
                args: vec![inner],
                ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Realize) => {
            // Passthrough for phase-0 semantics: realize is an identity in
            // host lane.
            lower_host_expr(
                children(list).first().ok_or_else(|| {
                    host_expr_lowering_error(expr, "a `realize` node has no operand")
                })?,
                program,
                scope,
                tensor_helpers,
            )?
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Jit) => {
            // `spec/03-deep-syntax.md` §2.7: jit is a compilation trigger
            // and a semantic no-op at evaluation. Host-lane lowering
            // pass-through to the inner expression, mirroring `lower_jit`
            // in `crates/chelis-ir/src/lower.rs` and the IR DAG behavior.
            //
            // Without this arm jit fell through to `HostExpr::Unit`, so a
            // binding like `result = jit(to_tensor([...]))` emitted
            // `int result = 0; printf("()\n");` — silent data loss
            // (Finding 1 of red-team PR #51).
            lower_host_expr(
                children(list)
                    .first()
                    .ok_or_else(|| host_expr_lowering_error(expr, "a `jit` node has no operand"))?,
                program,
                scope,
                tensor_helpers,
            )?
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Par) => {
            // `spec/03-deep-syntax.md` §2.3: par v1 is sequential
            // composition. Lower the children in order and bind the value
            // of the last child as the par's value, mirroring `lower_par`
            // in `crates/chelis-ir/src/lower.rs`. We do not currently
            // thread intermediate children through a sequence node; if
            // they have side effects (e.g. `print`, `realize`), those
            // primitives have their own host-lane arms and the emitted C
            // will reach them through whatever scope the par appears in.
            // A future change can introduce a HostExpr::Sequence kind if
            // par needs to preserve non-IO side effects across children.
            //
            // Without this arm par fell through to `HostExpr::Unit`, so
            // `result = par {..; to_tensor(..)}` emitted
            // `int result = 0; printf("()\n");` (Finding 2 of red-team
            // PR #51).
            let kids = children(list);
            let mut last: Option<HostExpr> = None;
            for child in kids {
                last = Some(lower_host_expr(child, program, scope, tensor_helpers)?);
            }
            last.ok_or_else(|| host_expr_lowering_error(expr, "an empty `par` has no value"))?
        }
        _ => {
            return Err(host_expr_lowering_error(
                expr,
                "no host-expression lowering rule exists for this checked form",
            ));
        }
    };
    Ok(lowered)
}

fn refine_function_params_from_body(params: &mut [HostParam], body: &HostExpr) {
    let HostExprKind::Call { args, arg_tys, .. } = &body.kind else {
        return;
    };
    for (index, param) in params.iter_mut().enumerate() {
        if !param.ty.is_unresolved() {
            continue;
        }
        let Some(HostExpr {
            kind: HostExprKind::Var(name, _),
            ..
        }) = args.get(index)
        else {
            continue;
        };
        if name != &param.name {
            continue;
        }
        let Some(inferred) = arg_tys.get(index) else {
            continue;
        };
        if !inferred.is_unresolved() {
            param.ty = inferred.clone();
        }
    }
}

fn refine_host_function_signatures(functions: &mut [HostFunction]) -> bool {
    let mut any_changed = false;
    loop {
        let mut changed = false;
        let signatures = functions
            .iter()
            .map(|function| {
                (
                    function.name.clone(),
                    (
                        function
                            .params
                            .iter()
                            .map(|param| param.ty.clone())
                            .collect::<Vec<_>>(),
                        function.ret_ty.clone(),
                    ),
                )
            })
            .collect::<UnordMap<_, _>>();

        for function in functions.iter_mut() {
            let inferred_param_fns = infer_callable_param_types(&function.params, &function.body);
            for param in function.params.iter_mut() {
                if param.ty.is_unresolved()
                    && let Some(inferred) = inferred_param_fns.get(&param.name)
                    && !host_type_is_unresolved(inferred)
                    && param.ty != *inferred
                {
                    param.ty = inferred.clone();
                    changed = true;
                }
            }

            let mut scope = function
                .params
                .iter()
                .map(|param| (param.name.clone(), param.ty.clone()))
                .collect::<UnordMap<_, _>>();
            if refine_host_expr_types(&mut function.body, &mut scope, &signatures) {
                changed = true;
            }
            if function.ret_ty.is_unresolved() {
                let body_ty = host_expr_type(&function.body);
                if !body_ty.is_unresolved() {
                    function.ret_ty = body_ty;
                    changed = true;
                }
            }

            let HostExprKind::Call {
                function: callee,
                args,
                arg_tys,
                ty,
            } = &mut function.body.kind
            else {
                continue;
            };
            let Some((callee_params, callee_ret)) = signatures.get(callee) else {
                continue;
            };

            if function.ret_ty.is_unresolved() && !callee_ret.is_unresolved() {
                function.ret_ty = callee_ret.clone();
                if ty.is_unresolved() {
                    *ty = callee_ret.clone();
                }
                changed = true;
            }

            for (index, param) in function.params.iter_mut().enumerate() {
                if !param.ty.is_unresolved() {
                    continue;
                }
                let Some(HostExpr {
                    kind: HostExprKind::Var(name, arg_ty),
                    ..
                }) = args.get_mut(index)
                else {
                    continue;
                };
                if name != &param.name {
                    continue;
                }
                let Some(inferred) = callee_params.get(index) else {
                    continue;
                };
                if inferred.is_unresolved() {
                    continue;
                }
                param.ty = inferred.clone();
                *arg_ty = inferred.clone();
                if let Some(call_arg_ty) = arg_tys.get_mut(index) {
                    *call_arg_ty = inferred.clone();
                }
                changed = true;
            }
        }

        if !changed {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn refine_host_globals(globals: &mut [HostBinding], functions: &[HostFunction]) -> bool {
    let mut any_changed = false;
    loop {
        let signatures = functions
            .iter()
            .map(|function| {
                (
                    function.name.clone(),
                    (
                        function
                            .params
                            .iter()
                            .map(|param| param.ty.clone())
                            .collect::<Vec<_>>(),
                        function.ret_ty.clone(),
                    ),
                )
            })
            .collect::<UnordMap<_, _>>();

        let mut changed = false;
        let mut scope = UnordMap::new();
        for binding in globals.iter_mut() {
            if refine_host_expr_types(&mut binding.value, &mut scope, &signatures) {
                changed = true;
            }
            let inferred = host_expr_type(&binding.value);
            if binding.ty != inferred && !inferred.is_unresolved() {
                binding.ty = inferred.clone();
                changed = true;
            }
            scope.insert(binding.name.clone(), binding.ty.clone());
        }

        if !changed {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn propagate_named_callback_signatures(
    functions: &mut [HostFunction],
    globals: &[HostBinding],
) -> bool {
    let mut inferred = UnordMap::<String, (Vec<HostTypeTerm>, HostTypeTerm)>::new();
    for function in functions.iter() {
        collect_named_callback_signatures(&function.body, &mut inferred);
    }
    for binding in globals {
        collect_named_callback_signatures(&binding.value, &mut inferred);
    }

    let mut changed = false;
    for function in functions.iter_mut() {
        let Some((param_tys, ret_ty)) = inferred.get(&function.name) else {
            continue;
        };
        for (param, inferred_ty) in function.params.iter_mut().zip(param_tys.iter()) {
            if host_type_is_unresolved(&param.ty) && !host_type_is_unresolved(inferred_ty) {
                param.ty = inferred_ty.clone();
                changed = true;
            }
        }
        if host_type_is_unresolved(&function.ret_ty) && !host_type_is_unresolved(ret_ty) {
            function.ret_ty = ret_ty.clone();
            changed = true;
        }
    }
    changed
}

fn collect_named_callback_signatures(
    expr: &HostExpr,
    out: &mut UnordMap<String, (Vec<HostTypeTerm>, HostTypeTerm)>,
) {
    match &expr.kind {
        HostExprKind::Call { args, .. }
        | HostExprKind::Builtin { args, .. }
        | HostExprKind::SignatureEntry { args, .. } => {
            for arg in args {
                collect_named_callback_signatures(arg, out);
            }
        }
        HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
            for item in items {
                collect_named_callback_signatures(item, out);
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields {
                collect_named_callback_signatures(field, out);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => {
            collect_named_callback_signatures(base, out);
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            collect_named_callback_signatures(cond, out);
            collect_named_callback_signatures(then_expr, out);
            collect_named_callback_signatures(else_expr, out);
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            collect_named_callback_signatures(scrutinee, out);
            collect_named_callback_signatures(some_expr, out);
            collect_named_callback_signatures(none_expr, out);
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            collect_named_callback_signatures(scrutinee, out);
            for arm in arms {
                collect_named_callback_signatures(&arm.expr, out);
            }
            if let Some(default_expr) = default_expr {
                collect_named_callback_signatures(default_expr, out);
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                collect_named_callback_signatures(&binding.value, out);
            }
            collect_named_callback_signatures(body, out);
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            merge_named_callback_signature(callback, out);
            collect_named_callback_signatures_in_callback(callback, out);
            collect_named_callback_signatures(list, out);
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            merge_named_callback_signature(callback, out);
            collect_named_callback_signatures_in_callback(callback, out);
            collect_named_callback_signatures(init, out);
            collect_named_callback_signatures(list, out);
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args {
                collect_named_callback_signatures(arg, out);
            }
        }
        HostExprKind::WithSeed { seed, body, .. } => {
            collect_named_callback_signatures(seed, out);
            collect_named_callback_signatures(body, out);
        }
        HostExprKind::Var(_, _)
        | HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
    }
}

fn collect_named_callback_signatures_in_callback(
    callback: &HostCallback,
    out: &mut UnordMap<String, (Vec<HostTypeTerm>, HostTypeTerm)>,
) {
    if let HostCallbackKind::Inline { body, .. } = &callback.kind {
        collect_named_callback_signatures(body, out);
    }
}

fn merge_named_callback_signature(
    callback: &HostCallback,
    out: &mut UnordMap<String, (Vec<HostTypeTerm>, HostTypeTerm)>,
) {
    let HostCallbackKind::Named { function, params } = &callback.kind else {
        return;
    };
    let entry = out.entry(function.clone()).or_insert_with(|| {
        (
            std::iter::repeat_with(fresh_host_inference)
                .take(params.len())
                .collect(),
            fresh_host_inference(),
        )
    });
    if entry.0.len() < params.len() {
        entry.0.extend(
            std::iter::repeat_with(fresh_host_inference).take(params.len() - entry.0.len()),
        );
    }
    for (index, param) in params.iter().enumerate() {
        if entry.0[index].is_unresolved() && !param.ty.is_unresolved() {
            entry.0[index] = param.ty.clone();
        }
    }
    if entry.1.is_unresolved() && !callback.ret_ty.is_unresolved() {
        entry.1 = callback.ret_ty.clone();
    }
}

fn infer_callable_param_types(
    params: &[HostParam],
    body: &HostExpr,
) -> UnordMap<String, HostTypeTerm> {
    let unknown = params
        .iter()
        .filter(|param| param.ty.is_unresolved())
        .map(|param| param.name.clone())
        .collect::<UnordSet<_>>();
    let mut out = UnordMap::new();
    infer_callable_param_types_in_expr(body, &unknown, &mut out);
    out
}

fn infer_callable_param_types_in_expr(
    expr: &HostExpr,
    unknown: &UnordSet<String>,
    out: &mut UnordMap<String, HostTypeTerm>,
) {
    match &expr.kind {
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ..
        } => {
            if unknown.contains(function) {
                out.entry(function.clone()).or_insert_with(|| {
                    HostTypeTerm::Fn(arg_tys.clone(), Box::new(HostTypeTerm::Tuple(Vec::new())))
                });
            }
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
            for item in items {
                infer_callable_param_types_in_expr(item, unknown, out);
            }
        }
        HostExprKind::Builtin { args, .. } | HostExprKind::SignatureEntry { args, .. } => {
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields {
                infer_callable_param_types_in_expr(field, unknown, out);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => {
            infer_callable_param_types_in_expr(base, unknown, out);
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(cond, unknown, out);
            infer_callable_param_types_in_expr(then_expr, unknown, out);
            infer_callable_param_types_in_expr(else_expr, unknown, out);
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(scrutinee, unknown, out);
            infer_callable_param_types_in_expr(some_expr, unknown, out);
            infer_callable_param_types_in_expr(none_expr, unknown, out);
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(scrutinee, unknown, out);
            for arm in arms {
                infer_callable_param_types_in_expr(&arm.expr, unknown, out);
            }
            if let Some(default_expr) = default_expr {
                infer_callable_param_types_in_expr(default_expr, unknown, out);
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                infer_callable_param_types_in_expr(&binding.value, unknown, out);
            }
            infer_callable_param_types_in_expr(body, unknown, out);
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            infer_callable_param_types_in_callback(callback, unknown, out);
            infer_callable_param_types_in_expr(list, unknown, out);
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            infer_callable_param_types_in_callback(callback, unknown, out);
            infer_callable_param_types_in_expr(init, unknown, out);
            infer_callable_param_types_in_expr(list, unknown, out);
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExprKind::WithSeed { seed, body, .. } => {
            infer_callable_param_types_in_expr(seed, unknown, out);
            infer_callable_param_types_in_expr(body, unknown, out);
        }
        HostExprKind::Var(_, _)
        | HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
    }
}

fn infer_callable_param_types_in_callback(
    callback: &HostCallback,
    unknown: &UnordSet<String>,
    out: &mut UnordMap<String, HostTypeTerm>,
) {
    if let HostCallbackKind::Inline { body, .. } = &callback.kind {
        infer_callable_param_types_in_expr(body, unknown, out);
    }
}

fn refine_host_expr_types(
    expr: &mut HostExpr,
    scope: &mut UnordMap<String, HostTypeTerm>,
    signatures: &UnordMap<String, (Vec<HostTypeTerm>, HostTypeTerm)>,
) -> bool {
    let mut changed = false;
    match &mut expr.kind {
        HostExprKind::Var(name, ty) => {
            if host_type_is_unresolved(ty)
                && let Some(inferred) = scope.get(name)
                && !host_type_is_unresolved(inferred)
            {
                *ty = inferred.clone();
                changed = true;
            }
        }
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
            let inferred_sig = scope.get(function).and_then(|ty| match ty {
                HostTypeTerm::Fn(params, ret) => Some((params.clone(), (**ret).clone())),
                _ => None,
            });
            let declared_sig = signatures.get(function).cloned();
            let sig = inferred_sig.or(declared_sig);
            if let Some((params, ret)) = sig {
                for (index, arg_ty) in arg_tys.iter_mut().enumerate() {
                    if let Some(inferred) = params.get(index)
                        && !inferred.is_unresolved()
                        && *arg_ty != *inferred
                    {
                        *arg_ty = inferred.clone();
                        changed = true;
                    }
                }
                if ty.is_unresolved() && !ret.is_unresolved() {
                    *ty = ret;
                    changed = true;
                }
            }
        }
        HostExprKind::SignatureEntry { args, .. } => {
            for arg in args {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
        }
        HostExprKind::Builtin { name, args, ty } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
            // An empty list remains genuinely underconstrained on its own,
            // but `to_tensor([])` is a language-level empty tensor literal:
            // its established checked/runtime representation is rank-1 f32.
            // Resolve that context here, rather than teaching the generic
            // empty-list state to manufacture an element type.
            if name == "to_tensor"
                && let Some(HostExpr {
                    kind: HostExprKind::List(items, list_ty),
                    ..
                }) = args.first_mut()
                && items.is_empty()
                && list_ty.is_unresolved()
            {
                *list_ty = HostTypeTerm::List(Box::new(HostTypeTerm::Float32));
                changed = true;
            }
            // RT-4 F1: only override `ty` when the current value has
            // unresolved type variables. Previously this clobbered any
            // declared-type retag (e.g. a global binding annotated as
            // `tensor[3, f64]` would be overwritten back to the
            // builtin's default `tensor[list, f32]` inferred return
            // type, defeating the F1 fix's typed runtime dispatch).
            if host_type_is_unresolved(ty)
                && let Some(inferred) = infer_builtin_host_type(name, args)
                && !host_type_is_unresolved(&inferred)
                && *ty != inferred
            {
                *ty = inferred;
                changed = true;
            }
        }
        HostExprKind::List(items, ty) => {
            for item in items.iter_mut() {
                changed |= refine_host_expr_types(item, scope, signatures);
            }
            let item_ty = items
                .iter()
                .map(host_expr_type)
                .find(|item_ty| !host_type_is_unresolved(item_ty))
                .unwrap_or_else(fresh_host_inference);
            let inferred = HostTypeTerm::List(Box::new(item_ty));
            if host_type_is_unresolved(ty) && !host_type_is_unresolved(&inferred) {
                *ty = inferred;
                changed = true;
            }
        }
        HostExprKind::Tuple(items, ty) => {
            for item in items.iter_mut() {
                changed |= refine_host_expr_types(item, scope, signatures);
            }
            let inferred = HostTypeTerm::Tuple(items.iter().map(host_expr_type).collect());
            if host_type_is_unresolved(ty) && !host_type_is_unresolved(&inferred) {
                *ty = inferred;
                changed = true;
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields.iter_mut() {
                changed |= refine_host_expr_types(field, scope, signatures);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => {
            changed |= refine_host_expr_types(base, scope, signatures);
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(cond, scope, signatures);
            changed |= refine_host_expr_types(then_expr, scope, signatures);
            changed |= refine_host_expr_types(else_expr, scope, signatures);
            if ty.is_unresolved() {
                let then_ty = host_expr_type(then_expr);
                let else_ty = host_expr_type(else_expr);
                let inferred = if !then_ty.is_unresolved() {
                    then_ty
                } else {
                    else_ty
                };
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(scrutinee, scope, signatures);
            let mut some_scope = scope.clone();
            let inner_ty = option_inner_type(scrutinee);
            if !inner_ty.is_unresolved() {
                some_scope.insert(bind_name.clone(), inner_ty);
            }
            changed |= refine_host_expr_types(some_expr, &mut some_scope, signatures);
            changed |= refine_host_expr_types(none_expr, scope, signatures);
            if ty.is_unresolved() {
                let some_ty = host_expr_type(some_expr);
                let none_ty = host_expr_type(none_expr);
                let inferred = if !some_ty.is_unresolved() {
                    some_ty
                } else {
                    none_ty
                };
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(scrutinee, scope, signatures);
            for arm in arms.iter_mut() {
                let mut arm_scope = scope.clone();
                for binding in &arm.bindings {
                    arm_scope.insert(binding.name.clone(), binding.ty.clone());
                }
                changed |= refine_host_expr_types(&mut arm.expr, &mut arm_scope, signatures);
            }
            if let Some(default_expr) = default_expr {
                changed |= refine_host_expr_types(default_expr, scope, signatures);
            }
            if ty.is_unresolved() {
                let inferred = arms
                    .iter()
                    .map(|arm| host_expr_type(&arm.expr))
                    .find(|ty| !ty.is_unresolved())
                    .or_else(|| default_expr.as_ref().map(|expr| host_expr_type(expr)));
                if let Some(inferred) = inferred
                    && !inferred.is_unresolved()
                {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Let { bindings, body, ty } => {
            let mut local_scope = scope.clone();
            for binding in bindings.iter_mut() {
                changed |= refine_host_expr_types(&mut binding.value, &mut local_scope, signatures);
                if binding.ty.is_unresolved() {
                    let inferred = host_expr_type(&binding.value);
                    if !inferred.is_unresolved() {
                        binding.ty = inferred.clone();
                        changed = true;
                    }
                }
                local_scope.insert(binding.name.clone(), binding.ty.clone());
            }
            changed |= refine_host_expr_types(body, &mut local_scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(body);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Map { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            let expected_ret = list_item_type(ty);
            changed |= specialize_host_callback_types(callback, &[item_ty], &expected_ret);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(list);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Filter { callback, list, ty }
        | HostExprKind::Partition { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            changed |= specialize_host_callback_types(callback, &[item_ty], &HostTypeTerm::Bool);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(list);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::FlatMap { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            let expected_ret = match ty {
                HostTypeTerm::List(inner) => HostTypeTerm::List(Box::new((**inner).clone())),
                _ => fresh_host_inference(),
            };
            changed |= specialize_host_callback_types(callback, &[item_ty], &expected_ret);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(list);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        } => {
            changed |= refine_host_expr_types(init, scope, signatures);
            changed |= refine_host_expr_types(list, scope, signatures);
            let init_ty = host_expr_type(init);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            changed |=
                specialize_host_callback_types(callback, &[init_ty.clone(), item_ty], &init_ty);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(init);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
        }
        HostExprKind::WithSeed { seed, body, ty } => {
            changed |= refine_host_expr_types(seed, scope, signatures);
            changed |= refine_host_expr_types(body, scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(body);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
    }
    changed
}

fn list_item_type(ty: &HostTypeTerm) -> HostTypeTerm {
    match ty {
        HostTypeTerm::List(inner) => (**inner).clone(),
        _ => fresh_host_inference(),
    }
}

fn callback_params_mut(callback: &mut HostCallback) -> &mut [HostParam] {
    match &mut callback.kind {
        HostCallbackKind::Named { params, .. } | HostCallbackKind::Inline { params, .. } => params,
    }
}

fn specialize_host_callback_types(
    callback: &mut HostCallback,
    param_tys: &[HostTypeTerm],
    ret_ty: &HostTypeTerm,
) -> bool {
    let mut changed = false;
    for (param, inferred) in callback_params_mut(callback)
        .iter_mut()
        .zip(param_tys.iter())
    {
        if host_type_is_unresolved(&param.ty) && !host_type_is_unresolved(inferred) {
            param.ty = inferred.clone();
            changed = true;
        }
    }
    if host_type_is_unresolved(&callback.ret_ty) && !host_type_is_unresolved(ret_ty) {
        callback.ret_ty = ret_ty.clone();
        changed = true;
    }
    changed
}

fn refine_host_callback_types(
    callback: &mut HostCallback,
    scope: &mut UnordMap<String, HostTypeTerm>,
    signatures: &UnordMap<String, (Vec<HostTypeTerm>, HostTypeTerm)>,
) -> bool {
    match &mut callback.kind {
        HostCallbackKind::Inline { params, body } => {
            let mut callback_scope = scope.clone();
            for param in params.iter() {
                callback_scope.insert(param.name.clone(), param.ty.clone());
            }
            let mut changed = refine_host_expr_types(body, &mut callback_scope, signatures);
            let inferred = host_expr_type(body);
            if !host_type_is_unresolved(&inferred) && callback.ret_ty != inferred {
                callback.ret_ty = inferred;
                changed = true;
            }
            changed
        }
        HostCallbackKind::Named { .. } => false,
    }
}

fn should_keep_tensor_expr_in_host_lane(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    if tag(list) == Some(DeepTag::If) {
        // The DAG lowerer already prunes literal conditions. Keep the whole
        // declaring signature on that kernel, instead of extracting a host
        // branch as a signatureless helper and losing its result obligation.
        // A dynamic condition retains host control flow and its activation.
        let kids = children(list);
        let literal_bool = kids.first().and_then(|condition| {
            let condition = match condition {
                Expr::List(lit, _) if tag(lit) == Some(DeepTag::Lit) => children(lit).first()?,
                other => other,
            };
            match condition {
                Expr::Atom(Atom::Bool(value), _) => Some(*value),
                _ => None,
            }
        });
        return match literal_bool {
            Some(taken) => kids
                .get(if taken { 1 } else { 2 })
                .is_none_or(should_keep_tensor_expr_in_host_lane),
            None => true,
        };
    }
    if matches!(
        tag(list),
        Some(DeepTag::TupleGet | DeepTag::Match | DeepTag::HandleEffect)
    ) {
        return true;
    }
    if tag(list) != Some(DeepTag::App) {
        return false;
    }
    let Some(callee) = children(list).first().and_then(as_list) else {
        return false;
    };
    if tag(callee) != Some(DeepTag::Var) {
        return false;
    }
    let name = children(callee).first().and_then(symbol_name);
    matches!(
        name,
        Some(
            "copy"
                | "to_tensor"
                | "scalar_to_tensor"
                | "pad_sequences"
                | "pad_sequences_to"
                | "concat"
                | "split"
                | "scatter"
                | "where"
                | "cumsum"
                | "map"
                | "filter"
                | "fold"
                | "scan"
                | "tensor_scan"
                | "partition"
                | "flat_map"
                | "sort"
                | "diagonal"
                | "trace"
                | "clamp"
                | "einsum"
        )
    )
}

fn lower_match_host_expr(
    list: &List,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    // chelis#1201: the match's RESULT type, when the caller knows it. Arm
    // bodies are result positions, so a generic ADT constructed in an arm
    // resolves its instantiation from this. Without it, a specialized
    // generic body's own constructors reach lowering unresolved even though
    // `lower_mono_specialized_function` already knows the concrete return
    // type. The scrutinee is NOT a result position and keeps plain lowering.
    expected_ty: Option<&HostTypeTerm>,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let kids = children(list);
    let match_expr = Expr::List(list.clone(), list_span(list));
    let scrutinee_node = kids
        .first()
        .ok_or_else(|| host_expr_lowering_error(&match_expr, "a `match` node has no scrutinee"))?;
    let scrutinee = lower_host_expr(scrutinee_node, program, scope, tensor_helpers)?;
    let scrutinee_ty = host_expr_type(&scrutinee);
    if matches!(
        scrutinee_ty,
        HostTypeTerm::Int64
            | HostTypeTerm::Float64
            | HostTypeTerm::Float32
            | HostTypeTerm::Bool
            | HostTypeTerm::String
    ) {
        return lower_literal_match_host_expr(
            list,
            program,
            scope,
            tensor_helpers,
            scrutinee,
            scrutinee_ty,
        );
    }
    let mut bind_name = "value".to_string();
    let mut some_expr = None;
    let mut none_expr = None;
    let mut generic_arms = Vec::new();
    let mut generic_default = None;
    let option_match = matches!(&scrutinee_ty, HostTypeTerm::Option(_));

    for arm in kids.iter().skip(1) {
        let Some(arm_list) = as_list(arm) else {
            continue;
        };
        if tag(arm_list) != Some(DeepTag::Arm) {
            continue;
        }
        let arm_kids = children(arm_list);
        let Some(pattern) = arm_kids.first().and_then(as_list) else {
            continue;
        };
        if tag(pattern) == Some(DeepTag::PatWild) {
            let body = arm_kids.get(2).ok_or_else(|| {
                host_expr_lowering_error(&match_expr, "a wildcard match arm has no body")
            })?;
            generic_default = Some(Box::new(lower_host_expr_with_expected_opt(
                body,
                program,
                scope,
                tensor_helpers,
                expected_ty,
            )?));
            continue;
        }
        if let Some(DeepTag::PatCtor | DeepTag::PatRecord) = tag(pattern) {
            let ctor = children(pattern).first().and_then(symbol_name);
            match ctor {
                Some("Some") if option_match => {
                    if let Some(bound) = children(pattern).get(1).and_then(as_list)
                        && tag(bound) == Some(DeepTag::PatVar)
                        && let Some(name) = children(bound).first().and_then(symbol_name)
                    {
                        bind_name = name.to_string();
                    }
                    let mut scoped = scope.clone();
                    scoped.insert(bind_name.clone(), option_inner_type(&scrutinee));
                    let body = arm_kids.get(2).ok_or_else(|| {
                        host_expr_lowering_error(&match_expr, "a `Some` match arm has no body")
                    })?;
                    some_expr = Some(lower_host_expr_with_expected_opt(
                        body,
                        program,
                        &scoped,
                        tensor_helpers,
                        expected_ty,
                    )?);
                }
                Some("None") if option_match => {
                    let body = arm_kids.get(2).ok_or_else(|| {
                        host_expr_lowering_error(&match_expr, "a `None` match arm has no body")
                    })?;
                    none_expr = Some(lower_host_expr_with_expected_opt(
                        body,
                        program,
                        scope,
                        tensor_helpers,
                        expected_ty,
                    )?);
                }
                Some(ctor_name) => {
                    let ctor_fields = match resolve_adt_constructor_definition_for_type(
                        program,
                        ctor_name,
                        &scrutinee_ty,
                    ) {
                        AdtConstructorResolution::Unique(definition) => {
                            instantiate_adt_constructor(program, &definition, &scrutinee_ty)
                                .map_err(|error| {
                                    host_expr_lowering_error(
                                        &match_expr,
                                        format!(
                                            "match constructor `{ctor_name}` is not concretely instantiated: {error}"
                                        ),
                                    )
                                })?
                                .fields
                        }
                        // The field list decides which index each binder
                        // reads, so answering an ambiguous pattern with
                        // either candidate silently binds the other
                        // package's field (chelis#1271). The `type_env`
                        // fallback below is for names that are not
                        // constructors at all and must not absorb this.
                        AdtConstructorResolution::Ambiguous(candidates) => {
                            return Err(ambiguous_constructor_error(
                                &match_expr,
                                ctor_name,
                                &candidates,
                            ));
                        }
                        AdtConstructorResolution::Missing => program
                            .type_env()
                            .get(ctor_name)
                            .and_then(parse_fn_type_expr)
                            .map(|(args, _)| {
                                args.into_iter()
                                    .map(|ty| HostAdtField { name: None, ty })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default(),
                    };
                    let mut scoped = scope.clone();
                    let mut bindings = Vec::new();
                    for (field_index, subpat, field_ty) in
                        pattern_field_bindings(pattern, &ctor_fields)
                    {
                        let Some(subpat_list) = as_list(subpat) else {
                            continue;
                        };
                        if tag(subpat_list) != Some(DeepTag::PatVar) {
                            continue;
                        }
                        let Some(name) = children(subpat_list).first().and_then(symbol_name) else {
                            continue;
                        };
                        let ty = field_ty
                            .or_else(|| expr_type(subpat))
                            .unwrap_or_else(fresh_host_inference);
                        scoped.insert(name.to_string(), ty.clone());
                        bindings.push(HostPatternBinding {
                            name: name.to_string(),
                            ty,
                            field_index,
                        });
                    }
                    generic_arms.push(HostMatchArm {
                        ctor: ctor_name.to_string(),
                        bindings,
                        expr: lower_host_expr_with_expected_opt(
                            arm_kids.get(2).ok_or_else(|| {
                                host_expr_lowering_error(
                                    &match_expr,
                                    "an ADT match arm has no body",
                                )
                            })?,
                            program,
                            &scoped,
                            tensor_helpers,
                            expected_ty,
                        )?,
                    });
                }
                None => {}
            }
        }
    }

    let ty = {
        let explicit = expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        );
        if explicit.is_unresolved() {
            // For ADT matches with multiple arms, Some/None exprs aren't set —
            // the arm bodies live in `generic_arms` / `generic_default`. Fall
            // back through those first so the match's result type reflects the
            // arms' actual shape. Otherwise Unit propagates and the match
            // target is emitted as `int`, which is the wrong C type for any
            // pointer-valued arm (regression hit by Coral groupby's
            // `next = match agg_spec { ... }` ADT match).
            let arm_ty = generic_arms
                .iter()
                .map(|arm| host_expr_type(&arm.expr))
                .find(|ty| !ty.is_unresolved())
                .or_else(|| {
                    generic_default
                        .as_deref()
                        .map(host_expr_type)
                        .filter(|ty| !ty.is_unresolved())
                });
            if let Some(ty) = arm_ty {
                ty
            } else {
                let some = some_expr.as_ref().ok_or_else(|| {
                    host_expr_lowering_error(&match_expr, "an Option match has no `Some` arm")
                })?;
                let none = none_expr.as_ref().ok_or_else(|| {
                    host_expr_lowering_error(&match_expr, "an Option match has no `None` arm")
                })?;
                let some_ty = host_expr_type(some);
                if some_ty.is_unresolved() {
                    host_expr_type(none)
                } else {
                    some_ty
                }
            }
        } else {
            explicit
        }
    };

    if matches!(scrutinee_ty, HostTypeTerm::Adt(_, _)) {
        return Ok(HostExpr::new(HostExprKind::MatchAdt {
            scrutinee: Box::new(scrutinee),
            arms: generic_arms,
            default_expr: generic_default,
            ty,
        }));
    }

    let some_expr = some_expr.ok_or_else(|| {
        host_expr_lowering_error(&match_expr, "an Option match has no `Some` arm")
    })?;
    let none_expr = none_expr.ok_or_else(|| {
        host_expr_lowering_error(&match_expr, "an Option match has no `None` arm")
    })?;
    Ok(HostExpr::new(HostExprKind::MatchOption {
        scrutinee: Box::new(scrutinee),
        bind_name,
        some_expr: Box::new(some_expr),
        none_expr: Box::new(none_expr),
        ty,
    }))
}

fn lower_literal_match_host_expr(
    list: &List,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    scrutinee: HostExpr,
    scrutinee_ty: HostTypeTerm,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let kids = children(list);
    let mut literal_arms = Vec::new();
    let mut default_expr = None;
    let match_expr = Expr::List(list.clone(), list_span(list));
    for arm in kids.iter().skip(1) {
        let Some(arm_list) = as_list(arm) else {
            continue;
        };
        if tag(arm_list) != Some(DeepTag::Arm) {
            continue;
        }
        let arm_kids = children(arm_list);
        let Some(pattern) = arm_kids.first().and_then(as_list) else {
            continue;
        };
        let body = lower_host_expr(
            arm_kids.get(2).ok_or_else(|| {
                host_expr_lowering_error(&match_expr, "a literal match arm has no body")
            })?,
            program,
            scope,
            tensor_helpers,
        )?;
        match tag(pattern) {
            Some(DeepTag::PatWild) => default_expr = Some(body),
            Some(DeepTag::PatLit) => {
                if let Some(lit) = children(pattern)
                    .first()
                    .and_then(|expr| host_literal_expr(expr, &scrutinee_ty))
                {
                    literal_arms.push((lit, body));
                }
            }
            _ => {}
        }
    }

    let explicit = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    let mut body = default_expr.ok_or_else(|| {
        host_expr_lowering_error(
            &match_expr,
            "a host-lowered literal match requires an explicit default arm",
        )
    })?;
    let result_ty = if explicit.is_unresolved() {
        host_expr_type(&body)
    } else {
        explicit
    };
    for (lit, arm_expr) in literal_arms.into_iter().rev() {
        body = HostExpr::new(HostExprKind::If {
            cond: Box::new(HostExpr::new(HostExprKind::Builtin {
                name: "eq".to_string(),
                args: vec![scrutinee.clone(), lit],
                ty: HostTypeTerm::Bool,
            })),
            then_expr: Box::new(arm_expr),
            else_expr: Box::new(body),
            ty: result_ty.clone(),
        });
    }
    Ok(body)
}

fn host_literal_expr(expr: &Expr, expected_ty: &HostTypeTerm) -> Option<HostExpr> {
    match (expr, expected_ty) {
        (Expr::Atom(Atom::Int(value), _), &HostTypeTerm::Int64) => {
            Some(HostExpr::new(HostExprKind::Int(*value)))
        }
        (Expr::Atom(Atom::Float(value), _), &HostTypeTerm::Float64) => {
            Some(HostExpr::new(HostExprKind::Float(*value)))
        }
        (Expr::Atom(Atom::Bool(value), _), &HostTypeTerm::Bool) => {
            Some(HostExpr::new(HostExprKind::Bool(*value)))
        }
        (Expr::Atom(Atom::Str(value), _), &HostTypeTerm::String) => {
            Some(HostExpr::new(HostExprKind::String(value.clone())))
        }
        (Expr::Atom(Atom::Int(value), _), _) => Some(HostExpr::new(HostExprKind::Int(*value))),
        (Expr::Atom(Atom::Float(value), _), _) => Some(HostExpr::new(HostExprKind::Float(*value))),
        (Expr::Atom(Atom::Bool(value), _), _) => Some(HostExpr::new(HostExprKind::Bool(*value))),
        (Expr::Atom(Atom::Str(value), _), _) => {
            Some(HostExpr::new(HostExprKind::String(value.clone())))
        }
        _ => None,
    }
}

fn lower_record_host_expr(
    list: &List,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected_ty: Option<&HostTypeTerm>,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let kids = children(list);
    let record_expr = Expr::List(list.clone(), list_span(list));
    let ctor = kids
        .first()
        .and_then(symbol_name)
        .ok_or_else(|| {
            host_expr_lowering_error(&record_expr, "a `record` node has no constructor symbol")
        })?
        .to_string();
    let checked_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    let resolution_ty = expected_ty
        .filter(|_| checked_ty.is_unresolved())
        .unwrap_or(&checked_ty);
    // The checker stamps a constructor expression with its nominal owner and
    // concrete instantiation. Resolve through that receipt before lowering
    // fields so same-named declarations cannot be selected by registry order
    // (chelis#1076/#1271).
    let ctor_definition =
        match resolve_adt_constructor_definition_for_type(program, &ctor, resolution_ty) {
            AdtConstructorResolution::Unique(definition) => definition,
            // The field-name check below reports the AUTHORED constructor, so
            // validating against a different declaration produced a
            // self-contradicting "has no field" rejection of a valid program
            // (chelis#1271). Name both candidates instead of choosing.
            AdtConstructorResolution::Ambiguous(candidates) => {
                return Err(ambiguous_constructor_error(
                    &record_expr,
                    &ctor,
                    &candidates,
                ));
            }
            AdtConstructorResolution::Missing => {
                return Err(host_expr_lowering_error(
                    &record_expr,
                    format!("record constructor `{ctor}` has no matching ADT declaration"),
                ));
            }
        };
    let explicit_ty = expected_ty
        .filter(|expected| {
            checked_ty.is_unresolved()
                && matches!(expected, HostTypeTerm::Adt(name, _) if terminal_name_matches(name, &ctor_definition.adt_name))
        })
        .cloned()
        .unwrap_or(checked_ty);
    let ctor_info =
        instantiate_adt_constructor(program, &ctor_definition, &explicit_ty).map_err(|error| {
            host_expr_lowering_error(
                &record_expr,
                format!("record constructor `{ctor}` is not concretely instantiated: {error}"),
            )
        })?;
    let mut supplied = UnordMap::new();
    for field in kids.iter().skip(1) {
        let kv_list = as_list(field).ok_or_else(|| {
            host_expr_lowering_error(&record_expr, "a record field is not a `kv` node")
        })?;
        if tag(kv_list) != Some(DeepTag::Kv) {
            return Err(host_expr_lowering_error(
                &record_expr,
                "a record field is not a `kv` node",
            ));
        }
        let kv_kids = children(kv_list);
        let name = kv_kids
            .first()
            .and_then(symbol_name)
            .ok_or_else(|| host_expr_lowering_error(&record_expr, "a record field has no name"))?;
        let expected_field = ctor_info
            .fields
            .iter()
            .find(|field| field.name.as_deref() == Some(name))
            .ok_or_else(|| {
                host_expr_lowering_error(
                    &record_expr,
                    format!("record constructor `{ctor}` has no field `{name}`"),
                )
            })?;
        // The instantiated constructor field is the checker-owned expected
        // type. Thread it into nested generic calls before lowering rather
        // than forcing it afterward: `Frame[Unit].cols` must specialize
        // `singleton` as `Hamt[Column[Unit]]` before its `Leaf` body lowers.
        let value = lower_host_expr_with_expected(
            kv_kids.get(1).ok_or_else(|| {
                host_expr_lowering_error(
                    &record_expr,
                    format!("record field `{name}` has no value"),
                )
            })?,
            program,
            scope,
            tensor_helpers,
            Some(&expected_field.ty),
        )?;
        if supplied.insert(name.to_string(), value).is_some() {
            return Err(host_expr_lowering_error(
                &record_expr,
                format!("record field `{name}` is supplied more than once"),
            ));
        }
    }
    let fields = ctor_info
        .fields
        .iter()
        .map(|field| {
            let name = field.name.as_ref().ok_or_else(|| {
                host_expr_lowering_error(
                    &record_expr,
                    format!("constructor `{ctor}` contains an unnamed record field"),
                )
            })?;
            let value = supplied.remove(name).ok_or_else(|| {
                host_expr_lowering_error(
                    &record_expr,
                    format!("record constructor `{ctor}` is missing field `{name}`"),
                )
            })?;
            Ok(force_host_expr_type(value, field.ty.clone()))
        })
        .collect::<Result<Vec<_>, crate::lower::LowerDiagnostic>>()?;
    if let Some((extra, _)) = supplied.to_sorted().into_iter().next() {
        return Err(host_expr_lowering_error(
            &record_expr,
            format!("record constructor `{ctor}` has no field `{extra}`"),
        ));
    }
    Ok(HostExpr::new(HostExprKind::AdtConstruct {
        ctor,
        fields,
        ty: ctor_info.ty,
    }))
}

fn lower_access_host_expr(
    list: &List,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let kids = children(list);
    let access_expr = Expr::List(list.clone(), list_span(list));
    let base = lower_host_expr(
        kids.first().ok_or_else(|| {
            host_expr_lowering_error(&access_expr, "an `access` node has no base expression")
        })?,
        program,
        scope,
        tensor_helpers,
    )?;
    let field_name = kids.get(1).and_then(symbol_name).ok_or_else(|| {
        host_expr_lowering_error(&access_expr, "an `access` node has no field symbol")
    })?;
    let (field_index, field_ty) = lookup_access_field(program, &base, field_name)
        .map_err(|error| {
            host_expr_lowering_error(
                &access_expr,
                format!("field `{field_name}` has no concrete ADT instantiation: {error}"),
            )
        })?
        .ok_or_else(|| {
            host_expr_lowering_error(
                &access_expr,
                format!("field `{field_name}` is absent or ambiguous on the resolved ADT type"),
            )
        })?;
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    Ok(HostExpr::new(HostExprKind::AdtFieldAccess {
        base: Box::new(base),
        field_index,
        ty: if !explicit_ty.is_unresolved() {
            explicit_ty
        } else {
            field_ty
        },
    }))
}

fn pattern_field_bindings<'a>(
    pattern: &'a List,
    ctor_fields: &'a [HostAdtField],
) -> Vec<(usize, &'a Expr, Option<HostTypeTerm>)> {
    match tag(pattern) {
        Some(DeepTag::PatRecord) => children(pattern)
            .iter()
            .skip(1)
            .filter_map(|kv_expr| {
                let kv_list = as_list(kv_expr)?;
                if tag(kv_list) != Some(DeepTag::Kv) {
                    return None;
                }
                let kv_kids = children(kv_list);
                let field_name = kv_kids.first().and_then(symbol_name)?;
                let field_index = ctor_fields
                    .iter()
                    .position(|field| field.name.as_deref() == Some(field_name))?;
                Some((
                    field_index,
                    kv_kids.get(1)?,
                    ctor_fields.get(field_index).map(|field| field.ty.clone()),
                ))
            })
            .collect(),
        _ => children(pattern)
            .iter()
            .skip(1)
            .enumerate()
            .map(|(field_index, subpat)| {
                (
                    field_index,
                    subpat,
                    ctor_fields.get(field_index).map(|field| field.ty.clone()),
                )
            })
            .collect(),
    }
}

fn lower_tuple_get_host_expr(
    list: &List,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let kids = children(list);
    let tuple_get_expr = Expr::List(list.clone(), list_span(list));
    if kids.len() != 2 {
        return Err(host_expr_lowering_error(
            &tuple_get_expr,
            format!(
                "a `tuple-get` node requires 2 children, found {}",
                kids.len()
            ),
        ));
    }
    let args = vec![
        lower_host_expr(&kids[0], program, scope, tensor_helpers)?,
        lower_host_expr(&kids[1], program, scope, tensor_helpers)?,
    ];
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    let ty = if explicit_ty.is_unresolved() {
        infer_builtin_host_type("tuple-get", &args).unwrap_or_else(fresh_host_inference)
    } else {
        explicit_ty
    };
    Ok(HostExpr::new(HostExprKind::Builtin {
        name: "tuple-get".to_string(),
        args,
        ty,
    }))
}

// ---------------------------------------------------------------------------
// Host-lane scalar forward-mode AD (chelis#405).
//
// Per `spec/design/phase5_host_scalar_ad.md`, the locked design is
// forward-mode dual numbers. A scalar function `f: f32 -> f32` (or
// multi-scalar-param) that lands in the host lane has no reverse-mode
// transform, so `grad(f, wrt=p)(args)` previously rejected with the
// unresolved-callable marker. This pass implements the dual transform
// entirely at compile time: it walks `f`'s pure-scalar body and produces
// two parallel HostExpr trees — a value tree and a derivative tree — using
// only the existing host scalar builtins (`add`/`mul`/`sub`/`div`/`neg`/
// `exp`/`log`/`sin`/`cos`/`tanh`/`sqrt`/`pow`/`abs`/`cast`). No new runtime
// struct and no new C builtin are required: the dual "struct" is split into
// two `double`-typed expression trees at lowering time, which is the
// forward-mode dual-number scheme the spec prescribes (one directional
// derivative per pass).
//
// Multi-parameter `wrt=(p1, p2, ...)` emits one derivative tree per
// parameter (each with that parameter's seed = 1.0 and the rest = 0.0) and
// combines them into a host tuple — the gradient tuple.
//
// `wrt` over a host container (list/dict/ADT/tuple) is rejected: this pass
// returns `None`, the caller falls through to the unresolved-callable
// marker, and the existing `cmd_build` guard surfaces the clean diagnostic.
// Tensor-lane reverse-mode AD is untouched: a grad whose differentiated fn
// is tensor-typed is handled by `lower_grad_callable_app` on the DAG path
// and never reaches this host-lane pass.

/// A dual value: the primal value expression and its derivative expression,
/// both ordinary scalar (`Float64`) host expressions.
#[derive(Clone)]
struct Dual {
    value: HostExpr,
    deriv: HostExpr,
}

fn dual_float(value: f64, deriv: f64) -> Dual {
    Dual {
        value: HostExpr::new(HostExprKind::Float(value)),
        deriv: HostExpr::new(HostExprKind::Float(deriv)),
    }
}

fn scalar_builtin(name: &str, args: Vec<HostExpr>) -> HostExpr {
    HostExpr::new(HostExprKind::Builtin {
        name: name.to_string(),
        args,
        ty: HostTypeTerm::Float64,
    })
}

fn host_float(value: f64) -> HostExpr {
    HostExpr::new(HostExprKind::Float(value))
}

/// `true` if a host type is a scalar this pass can differentiate. Integer
/// inputs are accepted (their derivative seed is 0 unless they are the
/// active `wrt`, but `wrt` over an integer is still a directional
/// derivative). Everything else — list/dict/ADT/tuple/tensor/option — is a
/// container and is rejected.
fn is_dual_scalar_type(ty: &HostTypeTerm) -> bool {
    matches!(
        ty,
        &HostTypeTerm::Float64 | &HostTypeTerm::Float32 | &HostTypeTerm::Int64
    )
}

/// Resolve a top-level scalar def by name into `(param_names, param_tys, body)`.
/// Returns `None` if the def is not a `(fn (params ...) body)` form or any
/// parameter is non-scalar.
fn resolve_scalar_def<'a>(
    program: &'a HostLoweringSession<'a>,
    name: &str,
) -> Option<(Vec<String>, Vec<HostTypeTerm>, &'a Expr)> {
    let mut found: Option<(&'a List,)> = None;
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some(DeepTag::Def) {
            continue;
        }
        let kids = children(list);
        let Some(def_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if !terminal_name_matches(def_name, name) {
            continue;
        }
        let Some(Expr::List(body_list, _)) = kids.get(1) else {
            continue;
        };
        if tag(body_list) != Some(DeepTag::Fn) {
            continue;
        }
        found = Some((body_list,));
        break;
    }
    let (fn_list,) = found?;
    let fn_kids = children(fn_list);
    let params_list = fn_kids.first().and_then(as_list)?;
    if tag(params_list) != Some(DeepTag::Params) {
        return None;
    }
    let mut param_names = Vec::new();
    let mut param_tys = Vec::new();
    for param in children(params_list) {
        let pname = param_name(param)?;
        let pty = param_host_type(param)
            .or_else(|| {
                lookup_declared_fn_type(program, name).and_then(|(tys, _)| tys.first().cloned())
            })
            .unwrap_or(HostTypeTerm::Float64);
        param_names.push(pname);
        param_tys.push(pty);
    }
    let body = fn_kids.get(1)?;
    Some((param_names, param_tys, body))
}

/// Try to lower `app(grad(f, wrt=...), arg0, ...)` as a host-lane scalar
/// forward-mode derivative. Returns `Some(host_expr)` on success, `None`
/// when this is not a scalar-grad app this pass handles (tensor lane,
/// structured `wrt`, unsupported op, unresolvable callee — all fall through
/// to the existing unresolved-callable-marker rejection path).
fn try_lower_scalar_grad_app(
    list: &List,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<Option<HostExpr>, crate::lower::LowerDiagnostic> {
    let kids = children(list);
    let Some(callee) = kids.first().and_then(as_list) else {
        return Ok(None);
    };
    if tag(callee) != Some(DeepTag::Grad) {
        return Ok(None);
    }
    // The function being differentiated: `(grad {wrt:...} (var f) (lit 0))`.
    let grad_kids = children(callee);
    let Some(fn_var) = grad_kids.first().and_then(as_list) else {
        return Ok(None);
    };
    if tag(fn_var) != Some(DeepTag::Var) {
        return Ok(None);
    }
    let Some(fn_name) = children(fn_var).first().and_then(symbol_name) else {
        return Ok(None);
    };
    let fn_name = fn_name.to_string();

    let Some((param_names, param_tys, body)) = resolve_scalar_def(program, &fn_name) else {
        return Ok(None);
    };

    // Return type must be scalar; reject (fall through) otherwise. We infer
    // it from the def's declared signature when available.
    if let Some((_, ret_ty)) = lookup_declared_fn_type(program, &fn_name)
        && !is_dual_scalar_type(&ret_ty)
    {
        return Ok(None);
    }
    // Every parameter must be a scalar. A structured parameter that is not the
    // `wrt` target is still fine to treat as a constant, but the call args
    // would be containers we cannot evaluate in the dual tree, so reject the
    // whole app (the recursive structured-AD lane below owns those calls).
    if param_tys.iter().any(|ty| !is_dual_scalar_type(ty)) {
        return Ok(None);
    }

    // Resolve the `wrt` parameter names from the grad meta. Absent `wrt`
    // means "all parameters" (single-param defs commonly omit it).
    let Some(wrt_names) = grad_wrt_param_names(callee, &param_names) else {
        return Ok(None);
    };
    if wrt_names.is_empty() {
        return Ok(None);
    }

    // Lower each call argument once into a value HostExpr. Their derivative
    // seed is determined per `wrt` pass below.
    let call_args = &kids[1..];
    if call_args.len() != param_names.len() {
        return Ok(None);
    }
    let arg_values: Vec<HostExpr> = call_args
        .iter()
        .map(|arg| lower_host_expr(arg, program, scope, tensor_helpers))
        .collect::<Result<Vec<_>, _>>()?;

    // One forward pass per `wrt` parameter.
    let mut derivs = Vec::new();
    for wrt_name in &wrt_names {
        let mut env: UnordMap<String, Dual> = UnordMap::new();
        for (idx, pname) in param_names.iter().enumerate() {
            let seed = if pname == wrt_name { 1.0 } else { 0.0 };
            env.insert(
                pname.clone(),
                Dual {
                    value: arg_values[idx].clone(),
                    deriv: host_float(seed),
                },
            );
        }
        let Some(dual) = dual_eval(body, &env, program, 0) else {
            return Ok(None);
        };
        derivs.push(dual.deriv);
    }

    if derivs.len() == 1 {
        Ok(derivs.pop())
    } else {
        let tys = derivs.iter().map(|_| HostTypeTerm::Float64).collect();
        Ok(Some(HostExpr::new(HostExprKind::Tuple(
            derivs,
            HostTypeTerm::Tuple(tys),
        ))))
    }
}

#[derive(Clone)]
enum GradPackPlan {
    Leaf(HostTypeTerm),
    Unit,
    List {
        ty: HostTypeTerm,
        items: Vec<GradPackPlan>,
    },
    Tuple {
        ty: HostTypeTerm,
        items: Vec<GradPackPlan>,
    },
    Adt {
        ty: HostTypeTerm,
        ctor: String,
        items: Vec<GradPackPlan>,
    },
}

impl GradPackPlan {
    fn host_type(&self) -> HostTypeTerm {
        match self {
            Self::Leaf(ty)
            | Self::List { ty, .. }
            | Self::Tuple { ty, .. }
            | Self::Adt { ty, .. } => ty.clone(),
            Self::Unit => HostTypeTerm::Unit,
        }
    }

    fn leaf_count(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Unit => 0,
            Self::List { items, .. } | Self::Tuple { items, .. } | Self::Adt { items, .. } => {
                items.iter().map(Self::leaf_count).sum()
            }
        }
    }

    fn first_tensor_type(&self) -> Option<TensorType> {
        match self {
            Self::Leaf(ty) => tensor_type_from_host_input(ty),
            Self::Unit => None,
            Self::List { items, .. } | Self::Tuple { items, .. } | Self::Adt { items, .. } => {
                items.iter().find_map(Self::first_tensor_type)
            }
        }
    }

    fn needs_host_reconstruction(&self) -> bool {
        match self {
            Self::List { .. }
            | Self::Tuple { .. }
            | Self::Adt { .. }
            | Self::Leaf(HostTypeTerm::Scalar(_)) => true,
            Self::Leaf(_) | Self::Unit => false,
        }
    }
}

fn static_list_spine_items(expr: &Expr) -> Option<Vec<Expr>> {
    let mut items = Vec::new();
    let mut cursor = expr;
    loop {
        let (node_tag, _, kids) = stamped_parts(cursor)?;
        match node_tag {
            DeepTag::Var if kids.first().and_then(symbol_name) == Some("Nil") => {
                return Some(items);
            }
            DeepTag::App => {
                let callee = kids.first().and_then(direct_var_name)?;
                if terminal_name(callee) != "Cons" {
                    return None;
                }
                items.push(kids.get(1)?.clone());
                cursor = kids.get(2)?;
            }
            _ => return None,
        }
    }
}

/// Resolve only the finite recursive shape of a List actual. This uses the
/// ordinary binder-aware call inliner and top-level definition lookup, so
/// result reconstruction is independent of whether the caller wrote a
/// literal, a named value, or one or more pure List-returning wrappers. The
/// executable helper still receives the original argument expression; this
/// walk is shape evidence, not argument evaluation.
fn resolve_list_grad_shape_expr(
    actual: &Expr,
    program: &HostLoweringSession<'_>,
    defs: &BTreeMap<String, Expr>,
) -> Expr {
    let mut resolved = actual.clone();
    for _ in 0..=MAX_DUAL_INLINE_DEPTH {
        if static_list_spine_items(&resolved).is_some() {
            break;
        }
        if let Some(name) = direct_var_name(&resolved)
            && let Some(body) = lookup_program_def(defs, name)
        {
            resolved = body.clone();
            continue;
        }
        if let Some(inlined) = beta_reduce_inline_host_call(&resolved)
            .or_else(|| inline_top_level_host_call(&resolved, program))
        {
            resolved = inlined;
            continue;
        }
        break;
    }
    resolved
}

fn grad_pack_plan(
    ty: &HostTypeTerm,
    actual: &Expr,
    program: &HostLoweringSession<'_>,
) -> Option<GradPackPlan> {
    if let HostTypeTerm::Adt(name, arguments) = ty
        && let Some((_, alias)) = program
            .adt_registry()
            .aliases
            .iter()
            .find(|(candidate, _)| terminal_name_matches(candidate, name))
    {
        let checker_parameter_names = alias
            .param_vars
            .iter()
            .zip(&alias.params)
            .map(|(variable, name)| (format!("t{}", variable.0), name.clone()))
            .collect::<UnordMap<_, _>>();
        let expanded = rename_host_type_variables(
            decode_host_type_or_raise(&type_to_deep_expr(&alias.body), &UnordMap::new()),
            &checker_parameter_names,
        );
        let substitutions = alias
            .params
            .iter()
            .cloned()
            .zip(arguments.iter().cloned())
            .collect();
        return grad_pack_plan(
            &substitute_host_type_term(expanded, &substitutions),
            actual,
            program,
        );
    }
    match ty {
        HostTypeTerm::List(element_ty) => {
            let items = static_list_spine_items(actual)?;
            let items = items
                .iter()
                .map(|item| grad_pack_plan(element_ty, item, program))
                .collect::<Option<Vec<_>>>()?;
            Some(GradPackPlan::List {
                ty: ty.clone(),
                items,
            })
        }
        HostTypeTerm::Tuple(item_tys) => {
            let (DeepTag::Tuple, _, item_exprs) = stamped_parts(actual)? else {
                return None;
            };
            if item_tys.len() != item_exprs.len() {
                return None;
            }
            let items = item_tys
                .iter()
                .zip(item_exprs)
                .map(|(item_ty, item)| grad_pack_plan(item_ty, item, program))
                .collect::<Option<Vec<_>>>()?;
            Some(GradPackPlan::Tuple {
                ty: HostTypeTerm::Tuple(items.iter().map(GradPackPlan::host_type).collect()),
                items,
            })
        }
        HostTypeTerm::Adt(_, _) => {
            let (ctor, supplied): (String, Vec<(Option<String>, &Expr)>) =
                match stamped_parts(actual)? {
                    (DeepTag::Record, _, kids) => {
                        let ctor = kids.first().and_then(symbol_name)?.to_string();
                        let fields = kids
                            .iter()
                            .skip(1)
                            .map(|field| {
                                let (DeepTag::Kv, _, kv) = stamped_parts(field)? else {
                                    return None;
                                };
                                Some((
                                    Some(kv.first().and_then(symbol_name)?.to_string()),
                                    kv.get(1)?,
                                ))
                            })
                            .collect::<Option<Vec<_>>>()?;
                        (ctor, fields)
                    }
                    (DeepTag::App, _, kids) => {
                        let ctor = kids.first().and_then(direct_var_name)?.to_string();
                        (
                            ctor,
                            kids.iter().skip(1).map(|field| (None, field)).collect(),
                        )
                    }
                    (DeepTag::Var, _, kids) => {
                        (kids.first().and_then(symbol_name)?.to_string(), Vec::new())
                    }
                    _ => return None,
                };
            let AdtConstructorResolution::Unique(definition) =
                resolve_adt_constructor_definition_for_type(program, &ctor, ty)
            else {
                return None;
            };
            let instantiated = definition.instantiate(ty).ok()?;
            let field_exprs = if supplied.iter().all(|(name, _)| name.is_some()) {
                instantiated
                    .fields
                    .iter()
                    .map(|field| {
                        supplied
                            .iter()
                            .find(|(name, _)| name.as_ref() == field.name.as_ref())
                            .map(|(_, expr)| *expr)
                    })
                    .collect::<Option<Vec<_>>>()?
            } else {
                supplied.iter().map(|(_, expr)| *expr).collect()
            };
            if instantiated.fields.len() != field_exprs.len() {
                return None;
            }
            let items = instantiated
                .fields
                .iter()
                .zip(field_exprs)
                .map(|(field, expr)| grad_pack_plan(&field.ty, expr, program))
                .collect::<Option<Vec<_>>>()?;
            Some(GradPackPlan::Adt {
                ty: ty.clone(),
                ctor,
                items,
            })
        }
        HostTypeTerm::Tensor(tensor) if tensor.precision.is_float() => {
            Some(GradPackPlan::Leaf(ty.clone()))
        }
        HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision)) if precision.is_float() => {
            Some(GradPackPlan::Leaf(ty.clone()))
        }
        _ => Some(GradPackPlan::Unit),
    }
}

fn tensor_helper_root_expr(
    binding_name: &str,
    binding_ty: &HostTypeTerm,
    root_tys: &[TensorType],
    index: usize,
) -> HostExpr {
    let binding = || {
        HostExpr::new(HostExprKind::Var(
            binding_name.to_string(),
            binding_ty.clone(),
        ))
    };
    if root_tys.len() == 1 {
        return force_host_expr_type(binding(), HostTypeTerm::Tensor(root_tys[index].clone()));
    }
    HostExpr::new(HostExprKind::Builtin {
        name: "tuple-get".to_string(),
        args: vec![binding(), HostExpr::new(HostExprKind::Int(index as i64))],
        ty: HostTypeTerm::Tensor(root_tys[index].clone()),
    })
}

fn pack_grad_roots(plan: &GradPackPlan, roots: &mut impl Iterator<Item = HostExpr>) -> HostExpr {
    match plan {
        GradPackPlan::Leaf(ty) => {
            let root = roots.next().expect("gradient root count was checked");
            // A primitive cotangent crosses the existing tensor-to-scalar
            // boundary. Rank-zero tensor cotangents keep their tensor identity.
            if matches!(ty, HostTypeTerm::Scalar(_)) {
                HostExpr::new(HostExprKind::Builtin {
                    name: "tensor_to_scalar".to_string(),
                    args: vec![root],
                    ty: ty.clone(),
                })
            } else {
                force_host_expr_type(root, ty.clone())
            }
        }
        GradPackPlan::Unit => HostExpr::new(HostExprKind::Unit),
        GradPackPlan::List { ty, items } => HostExpr::new(HostExprKind::List(
            items
                .iter()
                .map(|item| pack_grad_roots(item, roots))
                .collect(),
            ty.clone(),
        )),
        GradPackPlan::Tuple { ty, items } => HostExpr::new(HostExprKind::Tuple(
            items
                .iter()
                .map(|item| pack_grad_roots(item, roots))
                .collect(),
            ty.clone(),
        )),
        GradPackPlan::Adt { ty, ctor, items } => HostExpr::new(HostExprKind::AdtConstruct {
            ctor: ctor.clone(),
            fields: items
                .iter()
                .map(|item| pack_grad_roots(item, roots))
                .collect(),
            ty: ty.clone(),
        }),
    }
}

#[derive(Clone, Copy, Default)]
struct ScalarGradBodyRequirements {
    has_tensor_value: bool,
    reaches_host_collection_transform: bool,
}

impl ScalarGradBodyRequirements {
    fn merge(&mut self, other: Self) {
        self.has_tensor_value |= other.has_tensor_value;
        self.reaches_host_collection_transform |= other.reaches_host_collection_transform;
    }
}

/// Classify the body rather than its public signature when choosing between
/// scalar-host AD and reverse-DAG AD. A scalar-returning function can still
/// contain rank-zero tensor values through [05-OP-50], including in a called
/// scalar helper. Conversely, host collection transforms carry callbacks and
/// must keep the deliberate scalar-host rejection even when their callbacks
/// contain tensor intermediates.
fn scalar_grad_body_requirements(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    visiting: &mut UnordSet<String>,
) -> ScalarGradBodyRequirements {
    let mut requirements = ScalarGradBodyRequirements {
        has_tensor_value: matches!(
            expr_host_type(expr, program, scope),
            HostTypeTerm::Tensor(_)
        ),
        reaches_host_collection_transform: false,
    };

    if let Some((DeepTag::App, _, kids)) = stamped_parts(expr)
        && let Some(name) = kids.first().and_then(direct_var_name)
    {
        let scalar_def = resolve_scalar_def(program, name);
        if scalar_def.is_none() {
            requirements.reaches_host_collection_transform |= matches!(
                terminal_name(name),
                "map" | "filter" | "fold" | "scan" | "partition" | "flat_map"
            );
        }
        if !requirements.reaches_host_collection_transform
            && let Some((param_names, param_tys, body)) = scalar_def
            && visiting.insert(name.to_string())
        {
            let callee_scope = param_names
                .iter()
                .cloned()
                .zip(param_tys.iter().cloned())
                .collect();
            requirements.merge(scalar_grad_body_requirements(
                body,
                program,
                &callee_scope,
                visiting,
            ));
            visiting.remove(name);
        }
    }

    visit_semantic_expr_children(expr, |child| {
        requirements.merge(scalar_grad_body_requirements(
            child, program, scope, visiting,
        ));
    });
    requirements
}

fn pure_scalar_callable_needs_reverse_dag(program: &HostLoweringSession<'_>, name: &str) -> bool {
    if top_level_fn_needs_host_lane_tensor_lowering(program, name) {
        return false;
    }
    let Some((param_names, param_tys, body)) = resolve_scalar_def(program, name) else {
        return false;
    };
    let scope = param_names
        .iter()
        .cloned()
        .zip(param_tys.iter().cloned())
        .collect();
    let mut visiting = UnordSet::new();
    visiting.insert(name.to_string());
    let requirements = scalar_grad_body_requirements(body, program, &scope, &mut visiting);
    requirements.has_tensor_value && !requirements.reaches_host_collection_transform
}

/// Reconstruct primitive and finite recursive cotangents from one reverse DAG.
/// The IR stages each target as typed leaves plus runtime List controls and
/// retains the forward activation as a dependency of its cotangents. This host
/// step projects those roots into the checked public result type, preserving
/// primitive scalars, rank-zero tensors and recursive structures distinctly.
fn try_lower_general_list_grad_app(
    app_expr: &Expr,
    list: &List,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected_ty: Option<&HostTypeTerm>,
) -> Result<Option<HostExpr>, crate::lower::LowerDiagnostic> {
    let kids = children(list);
    let Some(callee) = kids.first().and_then(as_list) else {
        return Ok(None);
    };
    if tag(callee) != Some(DeepTag::Grad) {
        return Ok(None);
    }
    let defs = cached_program_defs(program);
    let grad_kids = children(callee);
    let Some(fn_name) = grad_kids.first().and_then(direct_var_name) else {
        return Ok(None);
    };
    // A scalar signature does not determine the lowering route. Pure scalar
    // arithmetic stays at the scalar-host AD boundary, and an unsupported
    // host collection transform deliberately declines there to the
    // unresolved-transform marker. A body with [05-OP-50] tensor values needs
    // the reverse DAG even when its parameters and result are all scalar.
    //
    // This body-sensitive decision precedes lowering: mixed/tensor callables
    // and DAG-backed scalar bodies still enter the #2078 path, so their
    // forward extent failures and deliberate DAG diagnostics propagate
    // without fallback.
    let pure_scalar_callable =
        lookup_declared_fn_type(program, fn_name).is_some_and(|(parameters, result)| {
            parameters.iter().all(is_dual_scalar_type) && is_dual_scalar_type(&result)
        });
    if pure_scalar_callable && !pure_scalar_callable_needs_reverse_dag(program, fn_name) {
        return Ok(None);
    }
    let Some(Expr::List(fn_list, _)) = lookup_program_def(&defs, fn_name) else {
        return Ok(None);
    };
    if tag(fn_list) != Some(DeepTag::Fn) {
        return Ok(None);
    }
    let fn_kids = children(fn_list);
    let Some(params) = fn_kids.first().and_then(as_list) else {
        return Ok(None);
    };
    let params = children(params);
    if params.len() != kids.len().saturating_sub(1) {
        return Ok(None);
    }
    let param_names = params.iter().map(param_name).collect::<Option<Vec<_>>>();
    let Some(param_names) = param_names else {
        return Ok(None);
    };
    let Some(wrt_names) = grad_wrt_param_names(callee, &param_names) else {
        return Ok(None);
    };
    let has_explicit_wrt = matches!(
        callee.elements.get(1),
        Some(Expr::Map(meta, _)) if meta.wrt().is_some()
    );

    let mut rewritten_elements = list.elements.clone();
    let mut parameter_plans = UnordMap::new();
    for (param_index, param) in params.iter().enumerate() {
        let Some(param_ty) = param_host_type(param) else {
            return Ok(None);
        };
        let actual_index = param_index + 1;
        let Some(actual) = kids.get(actual_index) else {
            return Ok(None);
        };
        let mut rewritten_actual = actual.clone();
        if matches!(param_ty, HostTypeTerm::List(_))
            && let Some(name) = direct_var_name(actual)
            && let Some(body) = lookup_program_def(&defs, name)
        {
            rewritten_actual = body.clone();
            rewritten_elements[actual_index + 2] = rewritten_actual.clone();
        }
        if wrt_names.contains(&param_names[param_index]) {
            let shape_actual =
                resolve_list_grad_shape_expr(&rewritten_actual, program, defs.as_ref());
            let Some(plan) = grad_pack_plan(&param_ty, &shape_actual, program) else {
                return Ok(None);
            };
            rewritten_actual = shape_actual;
            rewritten_elements[actual_index + 2] = rewritten_actual;
            // Spec/06 section 2.2: absent `wrt` selects only parameters
            // containing a differentiable float leaf. Preserve an explicit
            // target's zero-runtime-leaf plan (for example an empty
            // `List[f32]`) because its checked element type still defines a
            // differentiable, shape-preserving empty cotangent.
            if has_explicit_wrt || plan.leaf_count() != 0 {
                parameter_plans.insert(param_names[param_index].clone(), plan);
            }
        }
    }
    // The IR already orders complete cotangent groups by the written target
    // list. Reconstruct in that same order, including repeated selectors;
    // argument rewriting above still visits each primal exactly once.
    let mut selected_plans = Vec::with_capacity(wrt_names.len());
    for name in &wrt_names {
        if let Some(plan) = parameter_plans.get(name) {
            selected_plans.push(plan.clone());
        } else if has_explicit_wrt {
            // Explicit names were validated against the parameters and each
            // selected parameter must have a plan, even with zero leaves.
            return Err(crate::lower::LowerDiagnostic::new(
                format!("gradient target `{name}` has no result reconstruction plan"),
                Some(app_expr.span()),
                None,
            )
            .fatal());
        }
        // An implicit selection intentionally excludes nondifferentiable
        // parameters, whose plans were omitted above.
    }
    if !selected_plans
        .iter()
        .any(GradPackPlan::needs_host_reconstruction)
    {
        return Ok(None);
    }

    let checked_ty = expr_host_type(app_expr, program, scope);
    let inferred_result_ty = if selected_plans.len() == 1 {
        selected_plans[0].host_type()
    } else {
        HostTypeTerm::Tuple(selected_plans.iter().map(GradPackPlan::host_type).collect())
    };
    let result_ty = expected_ty
        .filter(|ty| !ty.is_unresolved())
        .cloned()
        .or_else(|| (!checked_ty.is_unresolved()).then_some(checked_ty))
        .unwrap_or(inferred_result_ty);
    let plan = if selected_plans.len() == 1 {
        selected_plans.pop().expect("one selected plan")
    } else {
        GradPackPlan::Tuple {
            ty: result_ty.clone(),
            items: selected_plans,
        }
    };
    let expected = plan
        .first_tensor_type()
        .unwrap_or_else(TensorType::scalar_f32);
    let rewritten = Expr::List(
        List {
            elements: rewritten_elements,
        },
        app_expr.span(),
    );
    let Some(lowered) =
        lower_tensor_helper_dag_with_controls(&rewritten, program, scope, &expected)
    else {
        return Ok(None);
    };
    if lowered.value_root_count != plan.leaf_count() {
        return Ok(None);
    }
    let root_tys = lowered
        .dag
        .roots()
        .iter()
        .filter_map(|root| lowered.dag.get(*root).map(|node| node.output_type.clone()))
        .collect::<Vec<_>>();
    let expected_control_roots = lowered
        .list_checks
        .iter()
        .map(|check| match check {
            crate::lower::RuntimeListCheckDescriptor::NonNegative { .. } => 1,
            crate::lower::RuntimeListCheckDescriptor::IndexBounds => 2,
        })
        .sum::<usize>();
    if root_tys.len() != lowered.value_root_count + expected_control_roots {
        return Ok(None);
    }
    if root_tys.is_empty() {
        let mut roots = std::iter::empty();
        return Ok(Some(pack_grad_roots(&plan, &mut roots)));
    }

    let helper_number = tensor_helpers.len();
    let call = finish_tensor_helper_call(lowered.dag, scope, tensor_helpers, expected);
    let binding_name = format!("__grad_result_{helper_number}");
    let binding_ty = host_expr_type(&call);
    let mut value_roots = (0..lowered.value_root_count)
        .map(|index| tensor_helper_root_expr(&binding_name, &binding_ty, &root_tys, index));
    let mut body = pack_grad_roots(&plan, &mut value_roots);

    let mut control_offset = lowered.value_root_count + expected_control_roots;
    for check in lowered.list_checks.iter().rev() {
        match check {
            crate::lower::RuntimeListCheckDescriptor::NonNegative {
                operation,
                argument,
            } => {
                control_offset -= 1;
                let value =
                    tensor_helper_root_expr(&binding_name, &binding_ty, &root_tys, control_offset);
                let value = HostExpr::new(HostExprKind::Builtin {
                    name: "tensor_to_scalar".to_string(),
                    args: vec![value],
                    ty: HostTypeTerm::Int64,
                });
                let cond = HostExpr::new(HostExprKind::Builtin {
                    name: "lt".to_string(),
                    args: vec![value, HostExpr::new(HostExprKind::Int(0))],
                    ty: HostTypeTerm::Bool,
                });
                body = HostExpr::new(HostExprKind::If {
                    cond: Box::new(cond),
                    then_expr: Box::new(HostExpr::new(HostExprKind::Builtin {
                        name: "fail".to_string(),
                        args: vec![HostExpr::new(HostExprKind::String(format!(
                            "{operation} requires non-negative {argument}"
                        )))],
                        ty: result_ty.clone(),
                    })),
                    else_expr: Box::new(body),
                    ty: result_ty.clone(),
                });
            }
            crate::lower::RuntimeListCheckDescriptor::IndexBounds => {
                control_offset -= 2;
                let index =
                    tensor_helper_root_expr(&binding_name, &binding_ty, &root_tys, control_offset);
                let len = tensor_helper_root_expr(
                    &binding_name,
                    &binding_ty,
                    &root_tys,
                    control_offset + 1,
                );
                let index = HostExpr::new(HostExprKind::Builtin {
                    name: "tensor_to_scalar".to_string(),
                    args: vec![index],
                    ty: HostTypeTerm::Int64,
                });
                let len = HostExpr::new(HostExprKind::Builtin {
                    name: "tensor_to_scalar".to_string(),
                    args: vec![len],
                    ty: HostTypeTerm::Int64,
                });
                let message = HostExpr::new(HostExprKind::Builtin {
                    name: "string_concat".to_string(),
                    args: vec![
                        HostExpr::new(HostExprKind::String("index ".to_string())),
                        HostExpr::new(HostExprKind::Builtin {
                            name: "string_concat".to_string(),
                            args: vec![
                                HostExpr::new(HostExprKind::Builtin {
                                    name: "to_string".to_string(),
                                    args: vec![index.clone()],
                                    ty: HostTypeTerm::String,
                                }),
                                HostExpr::new(HostExprKind::Builtin {
                                    name: "string_concat".to_string(),
                                    args: vec![
                                        HostExpr::new(HostExprKind::String(
                                            " out of bounds for list of len ".to_string(),
                                        )),
                                        HostExpr::new(HostExprKind::Builtin {
                                            name: "to_string".to_string(),
                                            args: vec![len.clone()],
                                            ty: HostTypeTerm::String,
                                        }),
                                    ],
                                    ty: HostTypeTerm::String,
                                }),
                            ],
                            ty: HostTypeTerm::String,
                        }),
                    ],
                    ty: HostTypeTerm::String,
                });
                let cond = HostExpr::new(HostExprKind::Builtin {
                    name: "gte".to_string(),
                    args: vec![index, len],
                    ty: HostTypeTerm::Bool,
                });
                body = HostExpr::new(HostExprKind::If {
                    cond: Box::new(cond),
                    then_expr: Box::new(HostExpr::new(HostExprKind::Builtin {
                        name: "fail".to_string(),
                        args: vec![message],
                        ty: result_ty.clone(),
                    })),
                    else_expr: Box::new(body),
                    ty: result_ty.clone(),
                });
            }
        }
    }
    debug_assert_eq!(control_offset, lowered.value_root_count);
    Ok(Some(HostExpr::new(HostExprKind::Let {
        bindings: vec![HostBinding {
            name: binding_name,
            display_name: None,
            display_roots: Vec::new(),
            ty: binding_ty,
            value: call,
        }],
        body: Box::new(body),
        ty: result_ty,
    })))
}

/// Read the `wrt` meta off a `grad` list and resolve it to a list of
/// parameter names. `None` is returned when `wrt` references something that
/// is not a parameter name (e.g. a tuple element / field access), which is
/// the container-AD case the spec rejects. Absent `wrt` defaults to all
/// parameters.
fn grad_wrt_param_names(grad_list: &List, param_names: &[String]) -> Option<Vec<String>> {
    let meta = match grad_list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta,
        _ => return Some(param_names.to_vec()),
    };
    let Some(targets) = meta.wrt() else {
        return Some(param_names.to_vec());
    };
    let names: Vec<String> = targets
        .variables()
        .map(|var| var.name().value().clone())
        .collect();
    // Each named target must actually be a parameter of the differentiated
    // function. A name that is not a parameter is a container/field access
    // we don't support.
    if names.iter().all(|n| param_names.contains(n)) {
        Some(names)
    } else {
        None
    }
}

/// Maximum nesting of inlined user-defined scalar calls and `let` blocks the
/// dual transform will follow. A non-recursive scalar def nests shallowly;
/// the cap exists so a (mutually) recursive scalar callee fails closed —
/// falling through to the unresolved-callable-marker rejection, not looping
/// forever or producing an unbounded dual tree.
const MAX_DUAL_INLINE_DEPTH: usize = 64;

/// Forward-mode dual evaluation of a pure-scalar Deep body. Returns `None`
/// for any construct this pass does not support (non-scalar op, unresolved
/// var, control flow) so the caller falls through to the rejection path.
/// `depth` tracks inlined-call / `let` nesting against `MAX_DUAL_INLINE_DEPTH`.
fn dual_eval(
    expr: &Expr,
    env: &UnordMap<String, Dual>,
    program: &HostLoweringSession<'_>,
    depth: usize,
) -> Option<Dual> {
    if depth > MAX_DUAL_INLINE_DEPTH {
        return None;
    }
    match expr {
        Expr::Atom(Atom::Float(v), _) => Some(dual_float(*v, 0.0)),
        Expr::Atom(Atom::Int(v), _) => Some(dual_float(*v as f64, 0.0)),
        Expr::List(list, _) => match tag(list) {
            Some(DeepTag::Lit) => {
                let inner = children(list).first()?;
                dual_eval(inner, env, program, depth)
            }
            Some(DeepTag::Var) => {
                let name = children(list).first().and_then(symbol_name)?;
                let dual = env.get(name)?;
                Some(Dual {
                    value: dual.value.clone(),
                    deriv: dual.deriv.clone(),
                })
            }
            Some(DeepTag::App) => dual_eval_app(list, env, program, depth),
            // `(let (bind n0 v0 n1 v1 ...) body)`: forward-mode through a
            // block body. Each binding's value is dual-evaluated in the
            // environment built so far (sequential scoping — a later binding
            // may reference an earlier one), then added to a cloned
            // environment under which the body is evaluated. The value and
            // derivative trees are substituted at each use site rather than
            // bound to host-let variables; this is correct because the dual
            // trees are pure `Float64` arithmetic with no side effects. The
            // canonical scalar-AD shapes (single-variable derivatives,
            // Black-Scholes Greeks) reuse each intermediate a small number of
            // times, so the substituted trees stay small.
            Some(DeepTag::Let) => dual_eval_let(list, env, program, depth),
            Some(DeepTag::Block) => {
                // chelis#859: dual-eval every child in order; the value is
                // the last child's. Host scalar bodies are pure, so the
                // non-last children contribute nothing to the duals.
                let kids = children(list);
                let (last, init) = kids.split_last()?;
                for child in init {
                    let _ = dual_eval(child, env, program, depth)?;
                }
                dual_eval(last, env, program, depth)
            }
            _ => None,
        },
        Expr::MetaExpr(meta, _) => dual_eval(&meta.expr, env, program, depth),
        _ => None,
    }
}

/// Forward-mode dual evaluation of a `(let (bind ...) body)` block. Returns
/// `None` if the binding structure is unexpected or any bound value / the
/// body contains a construct `dual_eval` does not support.
fn dual_eval_let(
    list: &List,
    env: &UnordMap<String, Dual>,
    program: &HostLoweringSession<'_>,
    depth: usize,
) -> Option<Dual> {
    let kids = children(list);
    let bind_list = kids.first().and_then(as_list)?;
    if tag(bind_list) != Some(DeepTag::Bind) {
        return None;
    }
    let body = kids.get(1)?;
    let bind_kids = children(bind_list);
    // Bindings are alternating `name value` pairs; an odd count is malformed.
    if !bind_kids.len().is_multiple_of(2) {
        return None;
    }
    let mut local_env = env.clone();
    for pair in bind_kids.as_chunks::<2>().0 {
        let name = symbol_name(&pair[0])?;
        let dual = dual_eval(&pair[1], &local_env, program, depth + 1)?;
        local_env.insert(name.to_string(), dual);
    }
    dual_eval(body, &local_env, program, depth + 1)
}

fn dual_eval_app(
    list: &List,
    env: &UnordMap<String, Dual>,
    program: &HostLoweringSession<'_>,
    depth: usize,
) -> Option<Dual> {
    let kids = children(list);
    let callee = kids.first().and_then(as_list)?;
    if tag(callee) != Some(DeepTag::Var) {
        return None;
    }
    let op = children(callee).first().and_then(symbol_name)?;
    let arg_exprs = &kids[1..];
    let mut args: Vec<Dual> = Vec::new();
    for a in arg_exprs {
        args.push(dual_eval(a, env, program, depth)?);
    }

    // Helper closures over scalar builtins.
    let v = |d: &Dual| d.value.clone();
    let dv = |d: &Dual| d.deriv.clone();

    match (op, args.len()) {
        ("add", 2) => Some(Dual {
            value: scalar_builtin("add", vec![v(&args[0]), v(&args[1])]),
            deriv: scalar_builtin("add", vec![dv(&args[0]), dv(&args[1])]),
        }),
        ("sub", 2) => Some(Dual {
            value: scalar_builtin("sub", vec![v(&args[0]), v(&args[1])]),
            deriv: scalar_builtin("sub", vec![dv(&args[0]), dv(&args[1])]),
        }),
        ("mul", 2) => {
            // (uv)' = u'v + uv'
            let lhs = scalar_builtin("mul", vec![dv(&args[0]), v(&args[1])]);
            let rhs = scalar_builtin("mul", vec![v(&args[0]), dv(&args[1])]);
            Some(Dual {
                value: scalar_builtin("mul", vec![v(&args[0]), v(&args[1])]),
                deriv: scalar_builtin("add", vec![lhs, rhs]),
            })
        }
        ("div", 2) => {
            // (u/v)' = (u'v - uv') / v^2
            let num_l = scalar_builtin("mul", vec![dv(&args[0]), v(&args[1])]);
            let num_r = scalar_builtin("mul", vec![v(&args[0]), dv(&args[1])]);
            let num = scalar_builtin("sub", vec![num_l, num_r]);
            let den = scalar_builtin("mul", vec![v(&args[1]), v(&args[1])]);
            Some(Dual {
                value: scalar_builtin("div", vec![v(&args[0]), v(&args[1])]),
                deriv: scalar_builtin("div", vec![num, den]),
            })
        }
        ("neg", 1) => Some(Dual {
            value: scalar_builtin("neg", vec![v(&args[0])]),
            deriv: scalar_builtin("neg", vec![dv(&args[0])]),
        }),
        ("exp", 1) => {
            // (e^u)' = e^u * u'
            let value = scalar_builtin("exp", vec![v(&args[0])]);
            Some(Dual {
                deriv: scalar_builtin("mul", vec![value.clone(), dv(&args[0])]),
                value,
            })
        }
        ("log", 1) => {
            // (ln u)' = u' / u
            Some(Dual {
                value: scalar_builtin("log", vec![v(&args[0])]),
                deriv: scalar_builtin("div", vec![dv(&args[0]), v(&args[0])]),
            })
        }
        ("sin", 1) => {
            // (sin u)' = cos(u) * u'
            let cos = scalar_builtin("cos", vec![v(&args[0])]);
            Some(Dual {
                value: scalar_builtin("sin", vec![v(&args[0])]),
                deriv: scalar_builtin("mul", vec![cos, dv(&args[0])]),
            })
        }
        ("cos", 1) => {
            // (cos u)' = -sin(u) * u'
            let sin = scalar_builtin("sin", vec![v(&args[0])]);
            let neg_sin = scalar_builtin("neg", vec![sin]);
            Some(Dual {
                value: scalar_builtin("cos", vec![v(&args[0])]),
                deriv: scalar_builtin("mul", vec![neg_sin, dv(&args[0])]),
            })
        }
        ("tanh", 1) => {
            // (tanh u)' = (1 - tanh(u)^2) * u'
            let t = scalar_builtin("tanh", vec![v(&args[0])]);
            let t2 = scalar_builtin("mul", vec![t.clone(), t.clone()]);
            let one_minus = scalar_builtin("sub", vec![host_float(1.0), t2]);
            Some(Dual {
                value: t,
                deriv: scalar_builtin("mul", vec![one_minus, dv(&args[0])]),
            })
        }
        ("sqrt", 1) => {
            // (sqrt u)' = u' / (2 sqrt(u))
            let s = scalar_builtin("sqrt", vec![v(&args[0])]);
            let den = scalar_builtin("mul", vec![host_float(2.0), s.clone()]);
            Some(Dual {
                value: s,
                deriv: scalar_builtin("div", vec![dv(&args[0]), den]),
            })
        }
        ("pow", 2) => {
            // Only constant exponents are supported in forward mode here:
            // (u^c)' = c * u^(c-1) * u'. A non-constant exponent (`deriv`
            // not identically zero) needs the general
            // u^v * (v' ln u + v u'/u) form; reject to stay correct.
            let exponent = float_const(&args[1].value)?;
            if !is_zero_float(&args[1].deriv) {
                return None;
            }
            let pow_inner = scalar_builtin("pow", vec![v(&args[0]), host_float(exponent - 1.0)]);
            let coeff = scalar_builtin("mul", vec![host_float(exponent), pow_inner]);
            Some(Dual {
                value: scalar_builtin("pow", vec![v(&args[0]), v(&args[1])]),
                deriv: scalar_builtin("mul", vec![coeff, dv(&args[0])]),
            })
        }
        // `cast` between scalar precisions is value-preserving for the dual
        // tree (host scalars are all `double`); the derivative passes
        // through unchanged.
        // `cast_trunc` deliberately has NO arm here: falling through to
        // the user-call path yields `None`, which is the [05-OP-6]
        // `no_grad` rejection. A passthrough dual would be the silent
        // zero-derivative the atom forbids.
        ("cast", _) if !args.is_empty() => Some(Dual {
            value: v(&args[0]),
            deriv: dv(&args[0]),
        }),
        // A call to a user-defined scalar def (`d1(...)`, `normal_cdf(...)`):
        // inline the callee's body into the dual tree. The callee must be a
        // top-level scalar def with scalar parameters; its body is
        // dual-evaluated in a fresh environment binding each parameter to the
        // corresponding already-computed dual argument (the chain rule is
        // carried by the argument derivatives). `resolve_scalar_def` rejects
        // non-scalar parameters, and `dual_eval` rejects any body construct
        // this pass does not support, so an unsupported callee falls through
        // to `None` (the unresolved-callable-marker rejection path).
        _ => dual_eval_user_call(op, &args, program, depth),
    }
}

/// Inline a call to a user-defined scalar def into the dual tree. Returns
/// `None` when the callee is not a resolvable scalar def, its arity does not
/// match, or its body uses an unsupported construct.
fn dual_eval_user_call(
    op: &str,
    args: &[Dual],
    program: &HostLoweringSession<'_>,
    depth: usize,
) -> Option<Dual> {
    let (param_names, param_tys, body) = resolve_scalar_def(program, op)?;
    if param_names.len() != args.len() {
        return None;
    }
    if param_tys.iter().any(|ty| !is_dual_scalar_type(ty)) {
        return None;
    }
    let mut call_env: UnordMap<String, Dual> = UnordMap::new();
    for (name, arg) in param_names.iter().zip(args.iter()) {
        call_env.insert(name.clone(), arg.clone());
    }
    dual_eval(body, &call_env, program, depth + 1)
}

/// Extract a compile-time float constant from a HostExpr if it is a literal.
fn float_const(expr: &HostExpr) -> Option<f64> {
    match &expr.kind {
        HostExprKind::Float(v) => Some(*v),
        HostExprKind::Int(v) => Some(*v as f64),
        _ => None,
    }
}

fn is_zero_float(expr: &HostExpr) -> bool {
    matches!(&expr.kind, HostExprKind::Float(v) if *v == 0.0)
        || matches!(&expr.kind, HostExprKind::Int(0))
}

fn lower_app_host_expr(
    app_expr: &Expr,
    list: &List,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected_ty: Option<&HostTypeTerm>,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    if let Some(grad_lowered) = try_lower_general_list_grad_app(
        app_expr,
        list,
        program,
        scope,
        tensor_helpers,
        expected_ty,
    )? {
        return Ok(grad_lowered);
    }
    // chelis#405: host-lane scalar forward-mode AD. When the callee is a
    // `grad(...)` form differentiating a scalar `f32 -> f32` (or
    // multi-scalar-param) top-level def, emit the dual-propagated derivative
    // directly. A `None` return falls through to the generic path, which
    // produces the unresolved-callable marker; grad/vmap-applying
    // programs get the `cmd_build` workaround text and everything else
    // is rejected at ABI projection (container `wrt`, tensor-lane grad,
    // unsupported op).
    if let Some(grad_lowered) = try_lower_scalar_grad_app(list, program, scope, tensor_helpers)? {
        return Ok(grad_lowered);
    }
    let kids = children(list);
    // Anonymous functions are represented at a statically-known call site by
    // beta reduction, not by inventing a nullable scalar/function value. This
    // is also the terminal step after a higher-order top-level call is
    // specialized below: substituting an `fn` argument into `f(x)` creates an
    // app whose callee is that inline function.
    if let Some(lowered) =
        lower_inline_host_invocation(app_expr, program, scope, tensor_helpers, expected_ty)?
    {
        return Ok(lowered);
    }
    let name = kids
        .first()
        .and_then(as_list)
        .and_then(|inner| {
            if tag(inner) == Some(DeepTag::Var) {
                children(inner).first().and_then(symbol_name)
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            // Route the fallback by what the callee actually was: an AD
            // transform in callee position earns the transform marker
            // (and the CLI's grad/vmap workaround text), anything else
            // is a plain unresolved callable. Both markers are
            // unspellable, so per-site routing survives without
            // consulting whole-program state (chelis#841 review,
            // finding 1).
            if kids.first().and_then(as_list).is_some_and(|inner| {
                matches!(tag(inner), Some(DeepTag::Grad | DeepTag::Vmap))
                    || inner.unknown_tag_symbol() == Some("vmap-grad")
            }) {
                HOST_UNRESOLVED_TRANSFORM_MARKER
            } else {
                HOST_UNRESOLVED_CALLABLE_MARKER
            }
        })
        .to_string();
    let fn_sig = scope
        .get(&name)
        .and_then(host_fn_signature)
        .or_else(|| lookup_declared_fn_type(program, &name))
        .or_else(|| kids.first().and_then(expr_fn_type));
    // Ordinary lexical lookup precedes builtin callable routes. A
    // function-typed parameter named `round_to` or `map` is a call through
    // that parameter, not a builtin selected by spelling
    // (spec/04-type-system.md §8.6; chelis#1076). Keep the override exact to
    // `BUILTIN_NAMES`: applied uppercase heads retain constructor precedence
    // under spec/01-nomenclature.md §3.2.
    let callee_is_local_callable = scope
        .get(&name)
        .is_some_and(|ty| matches!(ty, HostTypeTerm::Fn(_, _)));
    let callee_shadows_builtin = BUILTIN_NAMES.contains(&name.as_str()) && callee_is_local_callable;
    let active_compiler_name = (!callee_shadows_builtin).then_some(name.as_str());
    let checked_ty = expr_host_type(app_expr, program, scope);
    let explicit_ty = expected_ty
        .filter(|_| checked_ty.is_unresolved())
        .cloned()
        .unwrap_or(checked_ty);
    // Std.Io.Json owns canonical object observation. Keep generic
    // `dict_entries` insertion-ordered and lower only this exact private
    // package identity to the generated-C-local sorter. The name is exact so
    // a user function with the same terminal spelling cannot acquire magic
    // behavior.
    if kids.len() == 2
        && find_top_level_def_named(program.exprs(), &name).is_some_and(|(resolved, _)| {
            matches!(
                resolved,
                "Pkg__chelis__std__Std__Io__Json__canonical_object_entries"
                    | "pkg__chelis__std__Std__Io__Json__canonical_object_entries"
            )
        })
    {
        let entries = lower_host_expr(&kids[1], program, scope, tensor_helpers)?;
        return Ok(HostExpr::new(HostExprKind::Builtin {
            name: "__json_canonical_object_entries".to_string(),
            args: vec![entries],
            ty: explicit_ty,
        }));
    }
    let ctor_definition = match active_compiler_name
        .map(|_| resolve_adt_constructor_definition_for_type(program, &name, &explicit_ty))
        .unwrap_or(AdtConstructorResolution::Missing)
    {
        AdtConstructorResolution::Unique(definition) => Some(definition),
        // The resolved declaration supplies both the constructed ADT name
        // and each argument's expected field type, which lowering then
        // FORCES onto the argument. Answering an ambiguous name with the
        // wrong package's declaration silently rewrote a payload's dtype
        // (chelis#1271), so refuse rather than pick.
        AdtConstructorResolution::Ambiguous(candidates) => {
            return Err(ambiguous_constructor_error(app_expr, &name, &candidates));
        }
        // Not a constructor: this is the ordinary call path.
        AdtConstructorResolution::Missing => None,
    };
    let inferred_ret_ty = fn_sig
        .as_ref()
        .map(|(_, ret_ty)| ret_ty.clone())
        .unwrap_or_else(fresh_host_inference);
    // chelis#935/#936: specialize these bounded generic forms before any
    // helper-summary probe attempts to lower their standalone generic body.
    // The checked application metadata is the authoritative applied result
    // type; source-level reconstruction is neither needed nor allowed.
    let callee_is_nullary_generic_constructor_wrapper =
        top_level_fn_is_nullary_generic_constructor_wrapper(program, &name);
    // Precision and rank polymorphism have dedicated tensor-aware
    // specialization paths below. A nested rank parameter also appears as a
    // stored ADT type variable, so the broad ordinary-generic predicate must
    // not preempt that path (chelis#968).
    let callee_is_polymorphic_precision = lookup_declared_type_expr(program, &name)
        .as_ref()
        .is_some_and(crate::lower::type_expr_has_precision_var);
    let callee_is_polymorphic_rank = lookup_declared_type_expr(program, &name)
        .as_ref()
        .is_some_and(crate::lower::type_expr_has_rank_var)
        || top_level_fn_is_nested_rank_polymorphic(program, &name);
    // chelis#1201: a NON-recursive ordinary-generic call is no longer
    // inlined. Value-level inlining substitutes the argument expression over
    // the parameter name and drops the parameter's declared type, so any
    // generic-ADT term whose instantiation is only recoverable from those
    // annotations reached lowering unresolved. Such a call now falls through
    // to the same bounded monomorphization the recursive path uses, which
    // keys on the checked type application instead of pasting syntax.
    if callee_is_nullary_generic_constructor_wrapper
        && let Some(specialized) = inline_top_level_host_call(app_expr, program)
    {
        let pushed = push_inlining(&name);
        let definitions = adt_constructor_definitions(program);
        let specialized_ty =
            canonicalize_representation_erased_adt_args(explicit_ty.clone(), &definitions);
        let lowered = lower_host_expr_with_expected(
            &specialized,
            program,
            scope,
            tensor_helpers,
            (!specialized_ty.is_unresolved()).then_some(&specialized_ty),
        )
        .map(|body| {
            if specialized_ty.is_unresolved() {
                body
            } else {
                force_host_expr_type(body, specialized_ty.clone())
            }
        });
        if pushed {
            pop_inlining(&name);
        }
        return lowered;
    }
    if active_compiler_name == Some("Cons") && kids.len() == 3 {
        let expr = Expr::List(list.clone(), chelis_deep::Span::new(0, 0));
        if let Some(items) = lower_list_literal_items(&expr, program, scope, tensor_helpers)? {
            let ty = expr_host_type(&expr, program, scope);
            let ty = if ty.is_unresolved() {
                HostTypeTerm::List(Box::new(
                    items
                        .first()
                        .map(host_expr_type)
                        .unwrap_or_else(fresh_host_inference),
                ))
            } else {
                ty
            };
            return Ok(HostExpr::new(HostExprKind::List(items, ty)));
        }
    }
    // `Some` has a dedicated Option ABI only when no checked user ADT
    // constructor owns this application. A local `Wrapper::Some` shares the
    // spelling but lowers through the generic ADT path below; raw-name
    // dispatch would contradict the checker's nominal result type.
    if active_compiler_name == Some("Some") && kids.len() == 2 && ctor_definition.is_none() {
        let arg = lower_host_expr(&kids[1], program, scope, tensor_helpers)?;
        return Ok(HostExpr::new(HostExprKind::Builtin {
            name,
            args: vec![arg.clone()],
            ty: if !explicit_ty.is_unresolved() {
                explicit_ty
            } else {
                HostTypeTerm::Option(Box::new(host_expr_type(&arg)))
            },
        }));
    }
    if active_compiler_name == Some("map") && kids.len() == 3 {
        let callback =
            lower_host_callback(&kids[1], program, scope, tensor_helpers)?.ok_or_else(|| {
                host_expr_lowering_error(app_expr, "`map` requires a lowerable callback")
            })?;
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers)?;
        let ty = expr_host_type(app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            HostTypeTerm::List(Box::new(callback.ret_ty.clone()))
        } else {
            ty
        };
        return Ok(HostExpr::new(HostExprKind::Map {
            callback,
            list: Box::new(list_expr),
            ty,
        }));
    }
    if active_compiler_name == Some("filter") && kids.len() == 3 {
        let callback =
            lower_host_callback(&kids[1], program, scope, tensor_helpers)?.ok_or_else(|| {
                host_expr_lowering_error(app_expr, "`filter` requires a lowerable callback")
            })?;
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers)?;
        let ty = expr_host_type(app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            host_expr_type(&list_expr)
        } else {
            ty
        };
        return Ok(HostExpr::new(HostExprKind::Filter {
            callback,
            list: Box::new(list_expr),
            ty,
        }));
    }
    if active_compiler_name == Some("fold") && kids.len() == 4 {
        let callback =
            lower_host_callback(&kids[1], program, scope, tensor_helpers)?.ok_or_else(|| {
                host_expr_lowering_error(app_expr, "`fold` requires a lowerable callback")
            })?;
        // chelis#939: the checked callback signature owns the accumulator
        // type. Materialize it onto an empty/unresolved initializer rather
        // than reconstructing the element type from source syntax.
        let accumulator_ty = match &callback.kind {
            HostCallbackKind::Named { params, .. } | HostCallbackKind::Inline { params, .. } => {
                params.first().map(|param| param.ty.clone())
            }
        }
        .filter(|ty| !ty.is_unresolved())
        .unwrap_or_else(|| callback.ret_ty.clone());
        let init_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers)?;
        let init_expr = if accumulator_ty.is_unresolved() {
            init_expr
        } else {
            force_host_expr_type(init_expr, accumulator_ty.clone())
        };
        let list_expr = lower_host_expr(&kids[3], program, scope, tensor_helpers)?;
        let ty = expr_host_type(app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            accumulator_ty
        } else {
            ty
        };
        return Ok(HostExpr::new(HostExprKind::Fold {
            callback,
            init: Box::new(init_expr),
            list: Box::new(list_expr),
            ty,
        }));
    }
    if active_compiler_name == Some("scan") && kids.len() == 4 {
        let callback =
            lower_host_callback(&kids[1], program, scope, tensor_helpers)?.ok_or_else(|| {
                host_expr_lowering_error(app_expr, "`scan` requires a lowerable callback")
            })?;
        let init_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers)?;
        let list_expr = lower_host_expr(&kids[3], program, scope, tensor_helpers)?;
        let ty = expr_host_type(app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            HostTypeTerm::List(Box::new(host_expr_type(&init_expr)))
        } else {
            ty
        };
        return Ok(HostExpr::new(HostExprKind::Scan {
            callback,
            init: Box::new(init_expr),
            list: Box::new(list_expr),
            ty,
        }));
    }
    if active_compiler_name == Some("partition") && kids.len() == 3 {
        let callback =
            lower_host_callback(&kids[1], program, scope, tensor_helpers)?.ok_or_else(|| {
                host_expr_lowering_error(app_expr, "`partition` requires a lowerable callback")
            })?;
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers)?;
        let ty = expr_host_type(app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            let list_ty = host_expr_type(&list_expr);
            HostTypeTerm::Tuple(vec![list_ty.clone(), list_ty])
        } else {
            ty
        };
        return Ok(HostExpr::new(HostExprKind::Partition {
            callback,
            list: Box::new(list_expr),
            ty,
        }));
    }
    if active_compiler_name == Some("flat_map") && kids.len() == 3 {
        let callback =
            lower_host_callback(&kids[1], program, scope, tensor_helpers)?.ok_or_else(|| {
                host_expr_lowering_error(app_expr, "`flat_map` requires a lowerable callback")
            })?;
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers)?;
        let ty = expr_host_type(app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            match callback.ret_ty.clone() {
                HostTypeTerm::List(inner) => HostTypeTerm::List(inner),
                _ => fresh_host_inference(),
            }
        } else {
            ty
        };
        return Ok(HostExpr::new(HostExprKind::FlatMap {
            callback,
            list: Box::new(list_expr),
            ty,
        }));
    }
    let helper_tensor_ty = expr_tensor_type(app_expr, program, scope)
        .or_else(|| {
            if let HostTypeTerm::Tensor(tensor_ty) = &inferred_ret_ty {
                Some(tensor_ty.clone())
            } else if let HostTypeTerm::Tensor(tensor_ty) = &explicit_ty {
                Some(tensor_ty.clone())
            } else {
                None
            }
        })
        .or_else(|| {
            // When the callee is a `(grad {} fn_arg)` node (not a plain `var`),
            // `infer_app_expr_host_type` returns None because it only handles `var`
            // callees — so `helper_tensor_ty` is None and the tensor-helper path is
            // skipped entirely.  For `grad(named_fn)(x)` the output shape equals the
            // shape of the first differentiable argument `x`, so infer it from there.
            // This lets `try_lower_tensor_helper_call` succeed even when the outer
            // `app` node carries no explicit type annotation.
            let callee = kids.first().and_then(as_list)?;
            if tag(callee) != Some(DeepTag::Grad) {
                return None;
            }
            kids.get(1)
                .and_then(|first_arg| expr_tensor_type(first_arg, program, scope))
        });
    let (helper_expr, helper_scope, helper_bindings) =
        hoist_host_lane_tensor_bindings(app_expr, program, scope, fn_sig.as_ref(), tensor_helpers)?;
    // Local callable params (e.g. `f` in `def apply(f: fn, x) = f(x)`) are
    // not representable in the tensor-helper DAG — the DAG path would box
    // the fn pointer into `chelis_scalar_tensor_from_f64` and emit C that
    // gcc rejects. Skip both tensor-helper branches and fall through to
    // the generic `HostExpr::new(HostExprKind::Call)` path so the wrapper emits `return f(x);`.
    // A callee carrying a callable (fn-pointer) parameter cannot be
    // summarized through the tensor-helper DAG: the DAG has no
    // representation for a fn-pointer input. Such a callee is lowered by
    // inlining its body at the call site (the `has_callable_params`
    // branch below), where the local-wrapper grad/vmap form keeps its
    // rank-polymorphic shape symbolic. Skip both tensor-helper branches
    // so the inline path wins; otherwise the call monomorphizes against
    // the concrete arg shapes. This restores the lowering altitude the
    // over-broad recursion classification used to force.
    let has_callable_params = fn_sig
        .as_ref()
        .is_some_and(|(params, _)| params.iter().any(|ty| matches!(ty, HostTypeTerm::Fn(..))));
    // A function-valued formal has no tensor-helper representation. Probing
    // its standalone summary before the call-site inline path also erases the
    // concrete callable captured by a nested transform target (chelis#676).
    // Both helper branches below already exclude this case; keep summary
    // preparation behind the same boundary.
    let helper_summary_rejects = if has_callable_params {
        false
    } else {
        top_level_fn_helper_summary_rejects(program, &name)?
    };
    // A call into a staged function must retain that function's shared plan.
    // Re-extracting a tensor-only summary here loses its host scalar producers
    // and the claims attached before the original graph was partitioned.
    let callee_has_stages = if !callee_is_local_callable
        && !callee_is_polymorphic_precision
        && !callee_is_polymorphic_rank
        && let Some((canonical, body)) = find_top_level_def_named(program.exprs(), &name)
        && let Some(signature) = host_def_signature(canonical, body, None, program)
    {
        !matches!(
            staged_def_kernel(program, &signature, None)?,
            staged::StagingAttempt::NotApplicable
        )
    } else {
        false
    };
    if let Some(tensor_ty) = helper_tensor_ty.clone()
        && !callee_is_local_callable
        && !has_callable_params
        && !callee_has_stages
        && !top_level_fn_needs_host_lane_tensor_lowering(program, &name)
        && !helper_summary_rejects
        && !should_keep_tensor_expr_in_host_lane(app_expr)
        && let Some(tensor_call) = try_lower_tensor_helper_call(
            helper_expr.as_ref(),
            program,
            helper_scope.as_ref(),
            tensor_helpers,
            tensor_ty.clone(),
        )
    {
        if helper_bindings.is_empty() {
            return Ok(tensor_call);
        }
        let ty = host_expr_type(&tensor_call);
        return Ok(HostExpr::new(HostExprKind::Let {
            bindings: helper_bindings,
            body: Box::new(tensor_call),
            ty,
        }));
    }
    if let Some(tensor_ty) = helper_tensor_ty
        && !callee_is_local_callable
        && !has_callable_params
        && !callee_has_stages
        && !top_level_fn_needs_host_lane_tensor_lowering(program, &name)
        && !helper_summary_rejects
        && !should_keep_tensor_expr_in_host_lane(app_expr)
        && let Some(specialized) = inline_top_level_host_call(app_expr, program)
    {
        let pushed = push_inlining(&name);
        let lowered =
            try_lower_tensor_helper_call(&specialized, program, scope, tensor_helpers, tensor_ty);
        if pushed {
            pop_inlining(&name);
        }
        if let Some(tensor_call) = lowered {
            return Ok(tensor_call);
        }
    }
    // WS-A8: if the callee is a polymorphic-precision sig, the host
    // emitter elided its standalone definition (per the
    // `type_expr_has_precision_var` skip in `lower_host_program`).
    // Calling such a name in C produces an undefined-symbol link
    // error; the only legal lowering is to inline the body at the
    // call site so the precision is supplied from the call's
    // concrete arg types. Force inlining for this case.
    // Tier-2 rank polymorphism (spec/design/rank_polymorphism.md): the exact
    // analogue of the precision case above. The host emitter elided the
    // rank-poly callee's standalone definition (per the `type_expr_has_rank_var`
    // skip in `lower_host_program`); calling such a name in C is an
    // undefined-symbol link error. The only legal lowering is to inline the
    // body at the call site so the rank var is monomorphized from the call's
    // concrete arg shapes (via the DAG `tensor_rank_substitutions` path).
    // chelis#935: a generic host function has no standalone C symbol.
    // Inline it at a checked call site and materialize the application's
    // concrete result type onto the specialized body. This is the ordinary
    // type-variable analogue of the precision/rank specialization paths
    // below, kept separate so those tensor-specific paths retain their own
    // lowering rules.
    if (callee_is_polymorphic_precision || callee_is_polymorphic_rank)
        && let Some(specialized) = inline_top_level_host_call(app_expr, program)
    {
        let pushed = push_inlining(&name);
        let definitions = adt_constructor_definitions(program);
        let specialized_ty =
            canonicalize_representation_erased_adt_args(explicit_ty.clone(), &definitions);
        // chelis#2152: the inlined body keeps the callee's checked types, which
        // name the callee's own type variables. The DAG lane actualizes a
        // TENSOR precision from the concrete arguments by itself, but a host-
        // lane node inside the body (a scalar `cast(k, p)`, for one) reads its
        // type through the active substitution. Without one it stayed
        // `TypeVariable` at every call site and was rejected as an unactualized
        // cast target, even for a concrete f32 caller. Pin this call site's
        // bindings exactly as a monomorphized specialization does.
        let _subst_guard = ActiveTypeSubstGuard::push(inline_call_type_subst(
            &name,
            &kids[1..],
            &explicit_ty,
            program,
            scope,
        ));
        let lowered = lower_host_expr_with_expected(
            &specialized,
            program,
            scope,
            tensor_helpers,
            (!specialized_ty.is_unresolved()).then_some(&specialized_ty),
        );
        if pushed {
            pop_inlining(&name);
        }
        return lowered;
    }
    if has_callable_params
        && let Some(guarded) = lower_guarded_host_invocation(
            app_expr,
            &name,
            &explicit_ty,
            program,
            scope,
            tensor_helpers,
        )?
    {
        return Ok(guarded);
    }
    if has_callable_params && let Some(specialized) = inline_top_level_host_call(app_expr, program)
    {
        let pushed = push_inlining(&name);
        // PR #1215 review: thread the call's checked result type through the
        // inlined body exactly as the precision/rank inline path above does.
        // Plain lowering dropped it, so a generic ADT constructor in a
        // callable-parameter callee's body — whose instantiation is only
        // recoverable from the call-site result type — stayed unresolved and
        // failed closed (`apply[a](f: (a) -> a, x: a) -> Box[a]`).
        let definitions = adt_constructor_definitions(program);
        let specialized_ty =
            canonicalize_representation_erased_adt_args(explicit_ty.clone(), &definitions);
        let lowered = lower_host_expr_with_expected(
            &specialized,
            program,
            scope,
            tensor_helpers,
            (!specialized_ty.is_unresolved()).then_some(&specialized_ty),
        );
        if pushed {
            pop_inlining(&name);
        }
        return lowered;
    }
    // chelis#1158: a recursive ordinary-generic function has no standalone
    // C symbol and cannot be inlined (the call graph has a cycle). Compile
    // it through bounded memoized monomorphization: one specialized
    // definition per distinct checked type application, recursive edges
    // preserved as calls to the owning specialized symbol. A call whose
    // instantiation never resolves stays on the fail-closed [05-UNS]
    // boundary below.
    // chelis#1216: the same reasoning covers a recursive function generic
    // over an ERASED ADT dimension (chelis#940's `Frame[n] -> Column[n] ->
    // tensor[n, _]` shape). Its standalone definition is elided exactly like
    // the rank-polymorphic case above, and the inline path that would
    // normally specialize it refuses on the recursive edge, so without this
    // the call falls through to a plain call to a symbol that was never
    // emitted, carrying the ADT's own unresolved parameter variable. A true
    // rank variable is deliberately NOT routed here: variable rank is
    // monomorphized through the DAG `tensor_rank_substitutions` path, not by
    // specializing on a checked type application.
    if top_level_fn_is_type_polymorphic(program, &name)
        || top_level_fn_is_nested_rank_polymorphic(program, &name)
    {
        return lower_recursive_generic_call(
            app_expr,
            &name,
            &kids[1..],
            &explicit_ty,
            &inferred_ret_ty,
            program,
            scope,
            tensor_helpers,
        );
    }
    let args = kids[1..]
        .iter()
        .map(|arg| lower_host_expr(arg, program, scope, tensor_helpers))
        .collect::<Result<Vec<_>, _>>()?;
    let construct_ty = if let Some(definition) = &ctor_definition {
        if matches!(explicit_ty, HostTypeTerm::Adt(_, _)) {
            explicit_ty.clone()
        } else {
            HostTypeTerm::Adt(definition.adt_name.clone(), Vec::new())
        }
    } else {
        inferred_ret_ty.clone()
    };
    if let Some(definition) = ctor_definition {
        let instantiated = instantiate_adt_constructor(program, &definition, &construct_ty)
            .map_err(|error| {
                host_expr_lowering_error(
                    app_expr,
                    format!("constructor `{name}` is not concretely instantiated: {error}"),
                )
            })?;
        let expected_fields = instantiated.fields;
        let fields = args
            .into_iter()
            .enumerate()
            .map(|(index, field)| {
                if let Some(expected) = expected_fields.get(index) {
                    force_host_expr_type(field, expected.ty.clone())
                } else {
                    field
                }
            })
            .collect();
        return Ok(HostExpr::new(HostExprKind::AdtConstruct {
            ctor: name,
            fields,
            ty: instantiated.ty,
        }));
    }
    if (callee_shadows_builtin || !BUILTIN_NAMES.contains(&name.as_str()))
        && name != "Some"
        && name != "None"
        && fn_sig.is_some()
    {
        // When the explicit metadata type is Unknown OR contains unresolved
        // unresolved type variables preserve their named term; prefer
        // the inferred return type from the function's declared signature.
        // The metadata can decay to "Tuple([Unknown, Unknown])" when the
        // node-level annotator re-runs inference with a fresh subst that
        // doesn't share the outer pass's tvar bindings.
        let prefer_inferred = host_type_is_unresolved(&explicit_ty);
        return Ok(HostExpr::new(HostExprKind::Call {
            function: name,
            args,
            arg_tys: fn_sig
                .as_ref()
                .map(|(param_tys, _)| param_tys.clone())
                .unwrap_or_default(),
            ty: if prefer_inferred {
                inferred_ret_ty
            } else {
                explicit_ty
            },
        }));
    }
    let ty = if !explicit_ty.is_unresolved() {
        explicit_ty
    } else {
        infer_builtin_host_type(&name, &args).unwrap_or_else(fresh_host_inference)
    };
    Ok(HostExpr::new(HostExprKind::Builtin { name, args, ty }))
}

/// Beta-reduce an anonymous call for structural shape evidence only.
/// Executable lowering uses `lower_inline_host_invocation` to retain eager
/// actual evaluation and signature entry before the substituted body. This
/// binder-aware syntax walk does not itself represent an executed invocation.
fn beta_reduce_inline_host_call(expr: &Expr) -> Option<Expr> {
    let Expr::List(app_list, _) = expr else {
        return None;
    };
    if tag(app_list) != Some(DeepTag::App) {
        return None;
    }
    let kids = children(app_list);
    let callee = kids.first()?;
    let callee = match callee {
        Expr::MetaExpr(meta, _) => &meta.expr,
        direct => direct,
    };
    let fn_list = as_list(callee)?;
    if tag(fn_list) != Some(DeepTag::Fn) {
        return None;
    }
    let fn_kids = children(fn_list);
    let params = fn_kids.first().and_then(as_list)?;
    if tag(params) != Some(DeepTag::Params) {
        return None;
    }
    let args = kids.get(1..)?;
    if children(params).len() != args.len() {
        return None;
    }
    let substitutions = children(params)
        .iter()
        .zip(args)
        .filter_map(|(param, arg)| param_name(param).map(|name| (name, arg.clone())))
        .collect::<UnordMap<_, _>>();
    Some(inline_local_callable_lets(&substitute_expr(
        fn_kids.get(1)?,
        &substitutions,
        &UnordSet::new(),
    )))
}

/// A specialized invocation retains its own authored entry contract even
/// when its caller already checked a different signature on the same values.
struct RetainedHostInvocation<'a> {
    params: &'a [HostParam],
    body: &'a Expr,
    entry: SignatureEntryPlan,
    callable_entries: Vec<Option<SignatureEntryPlan>>,
    name: Option<&'a str>,
}

impl<'a> RetainedHostInvocation<'a> {
    fn new(params: &'a [HostParam], body: &'a Expr) -> Self {
        let entry = SignatureEntryPlan::new(params.iter().filter_map(|param| {
            let HostTypeTerm::Tensor(ty) = &param.ty else {
                return None;
            };
            Some(HostTensorInput {
                name: param.name.clone(),
                ty: ty.clone(),
            })
        }));
        let callable_entries = params
            .iter()
            .map(|param| {
                let HostTypeTerm::Fn(param_tys, _) = &param.ty else {
                    return None;
                };
                let entry = SignatureEntryPlan::new(param_tys.iter().enumerate().filter_map(
                    |(index, ty)| {
                        let HostTypeTerm::Tensor(ty) = ty else {
                            return None;
                        };
                        Some(HostTensorInput {
                            name: format!("arg{index}"),
                            ty: ty.clone(),
                        })
                    },
                ));
                (!entry.guards().is_empty()).then_some(entry)
            })
            .collect();
        Self {
            params,
            body,
            entry,
            callable_entries,
            name: None,
        }
    }

    fn has_entry_obligations(&self) -> bool {
        !self.entry.guards().is_empty() || self.callable_entries.iter().any(Option::is_some)
    }
}

fn typed_host_syntax_node(
    tag: DeepTag,
    metadata: Metadata,
    children: Vec<Expr>,
    span: chelis_deep::Span,
) -> Expr {
    let mut elements = vec![Expr::Atom(Atom::Tag(tag), span), Expr::Map(metadata, span)];
    elements.extend(children);
    Expr::List(List { elements }, span)
}

fn host_precision_syntax(precision: &HostPrecisionTerm, span: chelis_deep::Span) -> Option<Expr> {
    let name = match precision {
        HostPrecisionTerm::Concrete(precision) => precision.name(),
        HostPrecisionTerm::Variable(name) => name,
    };
    let tag = match precision {
        HostPrecisionTerm::Concrete(_) => DeepTag::TPrim,
        HostPrecisionTerm::Variable(_) => DeepTag::TVar,
    };
    Some(typed_host_syntax_node(
        tag,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name(name.to_string()), span)],
        span,
    ))
}

fn host_dim_syntax(dim: &DimInfo, span: chelis_deep::Span) -> Option<Expr> {
    match dim {
        DimInfo::Named(name, _) => Some(typed_host_syntax_node(
            DeepTag::DName,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name(name.clone()), span)],
            span,
        )),
        DimInfo::Lit(value) => Some(typed_host_syntax_node(
            DeepTag::DLit,
            Metadata::default(),
            vec![Expr::Atom(Atom::Int(i64::try_from(*value).ok()?), span)],
            span,
        )),
    }
}

fn host_shape_slot_syntax(slot: &HostShapeSlot, span: chelis_deep::Span) -> Option<Expr> {
    match slot {
        HostShapeSlot::Dim(dim) => host_dim_syntax(dim, span),
        HostShapeSlot::RankVariable(name) => Some(typed_host_syntax_node(
            DeepTag::DRank,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name(name.clone()), span)],
            span,
        )),
    }
}

fn host_adt_syntax(
    name: &str,
    args: impl IntoIterator<Item = Expr>,
    span: chelis_deep::Span,
) -> Expr {
    let mut children = vec![Expr::Atom(Atom::Name(name.to_string()), span)];
    children.extend(args);
    typed_host_syntax_node(DeepTag::TAdt, Metadata::default(), children, span)
}

fn host_type_syntax(ty: &HostTypeTerm, span: chelis_deep::Span) -> Option<Expr> {
    match ty {
        HostTypeTerm::Scalar(precision) => host_precision_syntax(precision, span),
        HostTypeTerm::Fn(params, ret) => {
            let mut children = params
                .iter()
                .map(|param| host_type_syntax(param, span))
                .collect::<Option<Vec<_>>>()?;
            children.push(host_type_syntax(ret, span)?);
            Some(typed_host_syntax_node(
                DeepTag::TFn,
                Metadata::default(),
                children,
                span,
            ))
        }
        HostTypeTerm::Adt(name, args) => Some(host_adt_syntax(
            name,
            args.iter()
                .map(|arg| host_type_syntax(arg, span))
                .collect::<Option<Vec<_>>>()?,
            span,
        )),
        HostTypeTerm::List(inner) => Some(host_adt_syntax(
            "List",
            [host_type_syntax(inner, span)?],
            span,
        )),
        HostTypeTerm::Dict(key, value) => Some(host_adt_syntax(
            "Dict",
            [host_type_syntax(key, span)?, host_type_syntax(value, span)?],
            span,
        )),
        HostTypeTerm::Tuple(items) => Some(typed_host_syntax_node(
            DeepTag::TTuple,
            Metadata::default(),
            items
                .iter()
                .map(|item| host_type_syntax(item, span))
                .collect::<Option<Vec<_>>>()?,
            span,
        )),
        HostTypeTerm::Tensor(tensor) => {
            let mut children = tensor
                .dims
                .iter()
                .map(|dim| host_dim_syntax(dim, span))
                .collect::<Option<Vec<_>>>()?;
            children.push(host_precision_syntax(
                &HostPrecisionTerm::Concrete(tensor.precision),
                span,
            )?);
            Some(typed_host_syntax_node(
                DeepTag::TTensor,
                Metadata::default(),
                children,
                span,
            ))
        }
        HostTypeTerm::PolymorphicTensor(tensor) => {
            let mut children = match &tensor.shape {
                HostShapeTerm::Concrete(dims) => dims
                    .iter()
                    .map(|dim| host_dim_syntax(dim, span))
                    .collect::<Option<Vec<_>>>()?,
                HostShapeTerm::Polymorphic(slots) => slots
                    .iter()
                    .map(|slot| host_shape_slot_syntax(slot, span))
                    .collect::<Option<Vec<_>>>()?,
            };
            children.push(host_precision_syntax(&tensor.precision, span)?);
            Some(typed_host_syntax_node(
                DeepTag::TTensor,
                Metadata::default(),
                children,
                span,
            ))
        }
        HostTypeTerm::Option(inner) => Some(host_adt_syntax(
            "Option",
            [host_type_syntax(inner, span)?],
            span,
        )),
        HostTypeTerm::MappedFile => Some(host_adt_syntax("MappedFile", [], span)),
        HostTypeTerm::Unit => Some(typed_host_syntax_node(
            DeepTag::TUnit,
            Metadata::default(),
            Vec::new(),
            span,
        )),
        HostTypeTerm::TypeVariable(name) => Some(typed_host_syntax_node(
            DeepTag::TVar,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name(name.clone()), span)],
            span,
        )),
        HostTypeTerm::InferenceVariable(_) | HostTypeTerm::Never => None,
    }
}

/// Reify a higher-order formal's entry contract around its supplied callable.
///
/// The actual stays the adapter body's callee, so its own declaration still
/// governs its body. The adapter carries the formal type which specialization
/// would otherwise erase; ordinary inline-invocation lowering then evaluates
/// payload actuals once, executes this formal entry plan, and only then calls
/// the supplied function. A formal with no executable entry obligation needs
/// no adapter.
fn retain_callable_entry_contract(
    actual: &Expr,
    formal: &HostTypeTerm,
    entry: &SignatureEntryPlan,
    formal_index: usize,
    reserved: &mut UnordSet<String>,
    span: chelis_deep::Span,
) -> Expr {
    let HostTypeTerm::Fn(param_tys, _) = formal else {
        return actual.clone();
    };
    let mut param_names = Vec::with_capacity(param_tys.len());
    for param_index in 0..param_tys.len() {
        let mut serial = 0;
        let name = loop {
            let candidate = if serial == 0 {
                format!("arg{param_index}")
            } else {
                format!("__chelis_indirect_arg_{formal_index}_{param_index}_{serial}")
            };
            if reserved.insert(candidate.clone()) {
                break candidate;
            }
            serial += 1;
        };
        param_names.push(name);
    }
    if entry.guards().is_empty() {
        return actual.clone();
    }

    let declarations = param_names
        .iter()
        .zip(param_tys)
        .map(|(name, ty)| {
            let type_syntax = host_type_syntax(ty, span);
            Expr::List(
                List {
                    elements: vec![
                        Expr::Atom(Atom::Name(name.clone()), span),
                        Expr::Map(callable_type_metadata(type_syntax.as_ref()), span),
                    ],
                },
                span,
            )
        })
        .collect();
    let params = typed_host_syntax_node(DeepTag::Params, Metadata::default(), declarations, span);
    let mut call_children = vec![actual.clone()];
    call_children.extend(param_names.iter().zip(param_tys).map(|(name, ty)| {
        let type_syntax = host_type_syntax(ty, span);
        typed_host_syntax_node(
            DeepTag::Var,
            callable_type_metadata(type_syntax.as_ref()),
            vec![Expr::Atom(Atom::Name(name.clone()), span)],
            span,
        )
    }));
    let ret_syntax = match formal {
        HostTypeTerm::Fn(_, ret) => host_type_syntax(ret, span),
        _ => None,
    };
    let body = typed_host_syntax_node(
        DeepTag::App,
        callable_type_metadata(ret_syntax.as_ref()),
        call_children,
        span,
    );
    let fn_syntax = host_type_syntax(formal, span);
    typed_host_syntax_node(
        DeepTag::Fn,
        callable_type_metadata(fn_syntax.as_ref()),
        vec![params, body],
        span,
    )
}

/// Executable beta reduction first retains actual evaluation and the lambda's
/// own signature. The AST-only reducer above is reserved for shape evidence.
fn lower_inline_host_invocation(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
    expected: Option<&HostTypeTerm>,
) -> Result<Option<HostExpr>, crate::lower::LowerDiagnostic> {
    let Some(call) = as_list(expr) else {
        return Ok(None);
    };
    let Some(callee) = children(call).first() else {
        return Ok(None);
    };
    let callee = match callee {
        Expr::MetaExpr(meta, _) => &meta.expr,
        direct => direct,
    };
    let Some(function) = as_list(callee).filter(|list| tag(list) == Some(DeepTag::Fn)) else {
        return Ok(None);
    };
    let children = children(function);
    let Some(declarations) = children
        .first()
        .and_then(as_list)
        .filter(|list| tag(list) == Some(DeepTag::Params))
    else {
        return Err(host_expr_lowering_error(
            expr,
            "inline invocation lost its parameters",
        ));
    };
    let Some(body) = children.get(1) else {
        return Err(host_expr_lowering_error(
            expr,
            "inline invocation lost its body",
        ));
    };
    let inferred = expr_fn_type(callee).map(|(params, _)| params);
    let mut params = Vec::new();
    for (index, declaration) in self::children(declarations).iter().enumerate() {
        let Some(name) = param_name(declaration) else {
            return Err(host_expr_lowering_error(
                expr,
                "inline invocation lost a parameter name",
            ));
        };
        // The authored annotation is the obligation. Actualized expression
        // metadata supplies only parameters whose annotation is absent.
        let ty = param_host_type(declaration)
            .or_else(|| {
                inferred
                    .as_ref()
                    .and_then(|params| params.get(index))
                    .cloned()
            })
            .map(|ty| expand_host_type_aliases(program, ty))
            .unwrap_or_else(fresh_host_inference);
        params.push(HostParam { name, ty });
    }
    lower_retained_host_invocation(
        expr,
        RetainedHostInvocation::new(&params, body),
        expected,
        program,
        scope,
        tensor_helpers,
    )
    .map(Some)
}

/// Keep the signature boundary which named body substitution otherwise erases.
fn lower_guarded_host_invocation(
    expr: &Expr,
    name: &str,
    expected: &HostTypeTerm,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<Option<HostExpr>, crate::lower::LowerDiagnostic> {
    if is_inlining(name) {
        return Ok(None);
    }
    let Some((canonical, body)) = find_top_level_def_named(program.exprs(), name) else {
        return Ok(None);
    };
    let Some(signature) = host_def_signature(canonical, body, None, program) else {
        return Ok(None);
    };
    let mut invocation = RetainedHostInvocation::new(&signature.params, &signature.body_expr);
    if !invocation.has_entry_obligations() {
        return Ok(None);
    }
    invocation.name = Some(name);
    lower_retained_host_invocation(
        expr,
        invocation,
        Some(expected),
        program,
        scope,
        tensor_helpers,
    )
    .map(Some)
}

/// Evaluate payload actuals once in caller order, then execute the complete
/// entry plan before the substituted body. Callable actuals remain syntax,
/// wrapped in their formal checked adapter when that formal owns entry guards.
fn lower_retained_host_invocation(
    expr: &Expr,
    invocation: RetainedHostInvocation<'_>,
    expected: Option<&HostTypeTerm>,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    let Expr::List(call, span) = expr else {
        return Err(host_expr_lowering_error(
            expr,
            "retained invocation is not an application",
        ));
    };
    let call_children = children(call);
    let Some((_, args)) = call_children.split_first() else {
        return Err(host_expr_lowering_error(
            expr,
            "retained invocation lost its actuals",
        ));
    };
    if args.len() != invocation.params.len() {
        return Err(host_expr_lowering_error(
            expr,
            "retained invocation argument count differs from its signature",
        ));
    }
    let mut reserved = scope
        .to_sorted()
        .into_iter()
        .map(|(name, _)| name.clone())
        .collect::<UnordSet<_>>();
    collect_deep_var_names(expr, &mut reserved);
    collect_deep_var_names(invocation.body, &mut reserved);
    names_bound_in(invocation.body, &mut reserved)
        .map_err(|detail| host_expr_lowering_error(expr, detail))?;
    let mut substitutions = UnordMap::new();
    let mut local_scope = scope.clone();
    let mut bindings = Vec::new();
    let mut observations = Vec::new();
    for (index, ((arg, formal), callable_entry)) in args
        .iter()
        .zip(invocation.params)
        .zip(&invocation.callable_entries)
        .enumerate()
    {
        if let Some(callable_entry) = callable_entry {
            substitutions.insert(
                formal.name.clone(),
                retain_callable_entry_contract(
                    arg,
                    &formal.ty,
                    callable_entry,
                    index,
                    &mut reserved,
                    *span,
                ),
            );
            continue;
        }
        if matches!(formal.ty, HostTypeTerm::Fn(..)) {
            substitutions.insert(formal.name.clone(), arg.clone());
            continue;
        }
        let value = lower_host_expr(arg, program, scope, tensor_helpers)?;
        let ty = host_expr_type(&value);
        let mut serial = index;
        let local = loop {
            let candidate = format!("__chelis_entry_arg_{serial}");
            if reserved.insert(candidate.clone()) {
                break candidate;
            }
            serial += 1;
        };
        bindings.push(HostBinding {
            name: local.clone(),
            display_name: None,
            display_roots: Vec::new(),
            ty: ty.clone(),
            value,
        });
        local_scope.insert(local.clone(), ty.clone());
        if matches!(formal.ty, HostTypeTerm::Tensor(_)) {
            observations.push(HostExpr::new(HostExprKind::Var(local.clone(), ty)));
        }
        substitutions.insert(
            formal.name.clone(),
            Expr::List(
                List {
                    elements: vec![
                        Expr::Atom(Atom::Tag(DeepTag::Var), *span),
                        Expr::Map(Metadata::default(), *span),
                        Expr::Atom(Atom::Name(local), *span),
                    ],
                },
                *span,
            ),
        );
    }
    let specialized = inline_local_callable_lets(&substitute_expr(
        invocation.body,
        &substitutions,
        &UnordSet::new(),
    ));
    if !invocation.entry.guards().is_empty() {
        let mut serial = bindings.len();
        let guard_name = loop {
            let candidate = format!("__chelis_entry_check_{serial}");
            if reserved.insert(candidate.clone()) {
                break candidate;
            }
            serial += 1;
        };
        bindings.push(HostBinding {
            name: guard_name,
            display_name: None,
            display_roots: Vec::new(),
            ty: HostTypeTerm::Unit,
            value: HostExpr::new(HostExprKind::SignatureEntry {
                plan: invocation.entry,
                args: observations,
            }),
        });
    }
    let definitions = adt_constructor_definitions(program);
    let expected = expected.map(|expected| {
        canonicalize_representation_erased_adt_args(expected.clone(), &definitions)
    });
    let pushed = invocation.name.is_some_and(push_inlining);
    let lowered = lower_host_expr_with_expected(
        &specialized,
        program,
        &local_scope,
        tensor_helpers,
        expected.as_ref().filter(|ty| !ty.is_unresolved()),
    );
    if pushed {
        pop_inlining(invocation.name.expect("named invocation"));
    }
    let body = lowered?;
    let ty = host_expr_type(&body);
    Ok(HostExpr::new(HostExprKind::Let {
        bindings,
        body: Box::new(body),
        ty,
    }))
}

fn inline_top_level_host_call(expr: &Expr, program: &HostLoweringSession<'_>) -> Option<Expr> {
    let Expr::List(app_list, _span) = expr else {
        return None;
    };
    if tag(app_list) != Some(DeepTag::App) {
        return None;
    }
    let kids = children(app_list);
    let callee_name = kids
        .first()
        .and_then(as_list)
        .filter(|callee| tag(callee) == Some(DeepTag::Var))
        .and_then(|callee| children(callee).first().and_then(symbol_name))?;
    if is_inlining(callee_name) {
        return None;
    }
    let defs = cached_program_defs(program);
    let body = lookup_program_def(&defs, callee_name)?;
    let Expr::List(fn_list, _) = body else {
        return None;
    };
    if tag(fn_list) != Some(DeepTag::Fn) {
        return None;
    }
    let fn_kids = children(fn_list);
    let params_list = fn_kids.first().and_then(as_list)?;
    if tag(params_list) != Some(DeepTag::Params) {
        return None;
    }
    let args = kids.get(1..)?;
    if args.len() != children(params_list).len() {
        return None;
    }
    let substitutions = children(params_list)
        .iter()
        .zip(args.iter())
        .filter_map(|(param, arg)| param_name(param).map(|name| (name, arg.clone())))
        .collect::<UnordMap<_, _>>();
    Some(inline_local_callable_lets(&substitute_expr(
        fn_kids.get(1)?,
        &substitutions,
        &UnordSet::new(),
    )))
}

/// Lower a call to a recursive type-polymorphic top-level function through
/// bounded memoized monomorphization (chelis#1158; spec/04 §3.1.1 supplies
/// the boundedness precondition, which this stage assumes and does not
/// re-litigate).
#[allow(clippy::too_many_arguments)]
fn lower_recursive_generic_call(
    app_expr: &Expr,
    name: &str,
    args: &[Expr],
    explicit_ty: &HostTypeTerm,
    inferred_ret_ty: &HostTypeTerm,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<HostExpr, crate::lower::LowerDiagnostic> {
    // Resolve the callee to its defining declaration first: the interned
    // identity is the definition's own name, never the call site's spelling,
    // so a qualified and a short reference to one def intern one
    // specialization per instantiation (harden-bounded-monomorphization D4).
    let Some((canonical_name, def_body)) = find_top_level_def_named(program.exprs(), name) else {
        return Err(host_expr_lowering_error(
            app_expr,
            format!(
                "generic host call `{name}` has no top-level definition to \
                 specialize (chelis#1226; [05-UNS-1])"
            ),
        ));
    };
    let canonical_name = canonical_name.to_string();
    let def_body = def_body.clone();

    // Derive the checked type application from the call site FIRST. Inside a
    // specialization the enclosing scope carries concrete parameter types,
    // so a recursive edge derives its own key even when it permutes or
    // narrows the caller's instantiation — [04-INF-2] admits any renaming of
    // the caller's own type parameters, and the renaming orbit is finite, so
    // each orbit member gets (and memoizes) its own specialized symbol.
    // Reusing the innermost in-progress symbol without checking the
    // instantiation would wire a permuted edge to a wrong-typed definition.
    let definitions = adt_constructor_definitions(program);
    let mut param_tys = Vec::with_capacity(args.len());
    for arg in args {
        let mut ty = canonicalize_representation_erased_adt_args(
            expr_host_type(arg, program, scope),
            &definitions,
        );
        // chelis#1216: the structural walk does not reconstruct every node's
        // type — a list literal lowers through `Cons` applications and comes
        // back as a fresh inference variable — but the checker stamped the
        // real type onto the node. Fall back to that stamp, and only adopt
        // it once canonicalization has erased the representation-irrelevant
        // ADT arguments: an argument whose sole unresolved part is an erased
        // ADT dimension then yields a concrete key instead of sinking the
        // whole application. A stamp that is still unresolved after erasure
        // is genuinely underconstrained and is left to the residue.
        if ty.is_unresolved()
            && let Some(stamped) = expr_type(arg)
        {
            let stamped = canonicalize_representation_erased_adt_args(stamped, &definitions);
            if !stamped.is_unresolved() {
                ty = stamped;
            }
        }
        param_tys.push(ty);
    }
    let ret_ty = if !explicit_ty.is_unresolved() {
        explicit_ty.clone()
    } else {
        inferred_ret_ty.clone()
    };
    let mut ret_ty = canonicalize_representation_erased_adt_args(ret_ty, &definitions);

    // Per-slot completion for a self-recursive edge with unconstrained
    // arguments (e.g. `loop(Empty)`, or a permuted edge whose third slot is
    // `Empty`): [04-INF-2] types an unconstrained argument at the caller's
    // own instantiation, and for a same-def edge callee slot i corresponds
    // to caller slot i, so each still-unresolved slot fills from the
    // innermost in-progress specialization's matching parameter. The
    // derived (possibly permuted) slots are KEPT — the merged application
    // then goes through the ordinary memo, never a blind whole-vector
    // reuse, so a permuted orbit member still gets its own symbol.
    if param_tys.iter().any(HostTypeTerm::is_unresolved) || ret_ty.is_unresolved() {
        let in_progress = MONO_SPECIALIZATIONS.with(|state| {
            state
                .borrow()
                .in_progress
                .iter()
                .rev()
                .find(|spec| spec.def_name == canonical_name && spec.param_tys.len() == args.len())
                .cloned()
        });
        if let Some(spec) = in_progress {
            for (slot, param_ty) in param_tys.iter_mut().enumerate() {
                if param_ty.is_unresolved() {
                    *param_ty = spec.param_tys[slot].clone();
                }
            }
            if ret_ty.is_unresolved() {
                ret_ty = spec.ret_ty.clone();
            }
        }
    }

    // chelis#1201: a NON-recursive generic call has no in-progress
    // specialization to complete from, so the block above cannot reach it.
    // The information it needs is still available: the callee's declared
    // parameter types. Decode those, solve the call site's type variables
    // against the slots that DID resolve, then re-decode the unresolved
    // slots under that substitution. This is what value-level inlining used
    // to discard by substituting the argument expression over the parameter
    // name and dropping the parameter's declared type.
    if param_tys.iter().any(HostTypeTerm::is_unresolved) || ret_ty.is_unresolved() {
        // A def with no `params` list contributes no declared types to solve
        // against. Spelled out rather than defaulted, so the empty case is a
        // stated outcome and not a swallowed one: those slots simply stay
        // unresolved and reach the residue below
        // (spec/design/loud_unsupported.md C4.3).
        let declared: Vec<Option<Expr>> = match params_list_of(&def_body) {
            Some(params) => children(params)
                .iter()
                .map(param_declared_type_expr)
                .collect::<Vec<_>>(),
            None => Vec::new(),
        };
        let mut subst: UnordMap<String, HostTypeTerm> = UnordMap::new();
        for (slot, declared_expr) in declared.iter().enumerate() {
            if let Some(declared_expr) = declared_expr
                && let Some(actual) = param_tys.get(slot)
                && !actual.is_unresolved()
            {
                let declared_term = decode_host_type_or_raise(declared_expr, &UnordMap::new());
                solve_host_type_vars(&declared_term, actual, &mut subst);
            }
        }
        // The RETURN type is a binding source too, and often the only one:
        // `hamt_from_pairs[a](pairs: List[(string, a)]) -> Hamt[a]` called
        // with an unresolved `pairs` still has a concrete checked result
        // (`Hamt[Column[...]]`), which pins `a` and thereby completes the
        // argument slot. Solving from arguments alone leaves the
        // substitution empty here and the call falls to the [05-UNS-1]
        // residue (chelis#1201, coral#26's `from_pairs` shape).
        if !ret_ty.is_unresolved()
            && let Some(declared_ret) = declared_return_type_expr(program.exprs(), &canonical_name)
        {
            let declared_ret_term = decode_host_type_or_raise(&declared_ret, &UnordMap::new());
            solve_host_type_vars(&declared_ret_term, &ret_ty, &mut subst);
        }
        if !subst.is_empty() {
            for (slot, param_ty) in param_tys.iter_mut().enumerate() {
                if param_ty.is_unresolved()
                    && let Some(Some(declared_expr)) = declared.get(slot)
                {
                    let completed = decode_host_type_or_raise(declared_expr, &subst);
                    if !completed.is_unresolved() {
                        *param_ty = completed;
                    }
                }
            }
            if ret_ty.is_unresolved()
                && let Some(declared_ret) =
                    declared_return_type_expr(program.exprs(), &canonical_name)
            {
                let completed = decode_host_type_or_raise(&declared_ret, &subst);
                if !completed.is_unresolved() {
                    ret_ty = completed;
                }
            }
        }
    }

    let derived_concrete =
        !param_tys.iter().any(HostTypeTerm::is_unresolved) && !ret_ty.is_unresolved();
    if derived_concrete {
        let symbol = ensure_mono_specialization(
            app_expr,
            &canonical_name,
            &def_body,
            &param_tys,
            &ret_ty,
            program,
            tensor_helpers,
        )?;
        let lowered_args = args
            .iter()
            .zip(param_tys.iter())
            .map(|(arg, param_ty)| {
                lower_host_expr_with_expected(arg, program, scope, tensor_helpers, Some(param_ty))
                    .map(|lowered| force_host_expr_type(lowered, param_ty.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(HostExpr::new(HostExprKind::Call {
            function: symbol,
            args: lowered_args,
            arg_tys: param_tys,
            ty: ret_ty,
        }));
    }

    // PR #1215 review: a NON-recursive top-level call whose instantiation
    // never resolves (`pick[a, b](x: a, y: Box[b]) -> a` applied to `Empty`
    // — `b` unconstrained, the parameter untouched) has the
    // pre-specialization lowering available: guarded value-level inlining,
    // with the call's checked result type threaded through like every other
    // inline path. Two gates keep this strictly a top-level-call fallback:
    // inside an in-progress specialization the [04-INF-2] per-slot
    // completion above owns the semantics and a mutually-recursive
    // cross-member edge must stay on the fail-closed residue below; and the
    // inlining stack bounds recursion — a recursive body re-entering here
    // finds its own name, `inline_top_level_host_call` returns `None`, and
    // the edge falls through to the residue.
    let specializing = MONO_SPECIALIZATIONS.with(|state| !state.borrow().in_progress.is_empty());
    if !specializing && let Some(specialized) = inline_top_level_host_call(app_expr, program) {
        let pushed_canonical = push_inlining(&canonical_name);
        let pushed_spelled = name != canonical_name && push_inlining(name);
        let lowered = lower_host_expr_with_expected(
            &specialized,
            program,
            scope,
            tensor_helpers,
            (!ret_ty.is_unresolved()).then_some(&ret_ty),
        );
        if pushed_spelled {
            pop_inlining(name);
        }
        if pushed_canonical {
            pop_inlining(&canonical_name);
        }
        return lowered;
    }

    // Fail-closed residue: no concrete checked instantiation to key a
    // specialized definition on and no inlinable shape — an outer recursive
    // call that never pins the type parameter, or a mutually-recursive
    // cross-member edge whose argument leaves the callee parameter
    // unconstrained (no positional correspondence exists across different
    // defs' parameters in host lowering). Emitting a reference to the
    // omitted generic definition is never legal, so reject loud instead.
    // The wording is recursion-neutral and the citation is the OPEN residue
    // tracker (chelis#1226): the call reaching here need not be recursive,
    // and chelis#1158 closed with its delivery ([05-UNS-5] requires a live
    // authority).
    Err(host_expr_lowering_error(
        app_expr,
        format!(
            "generic host call `{name}` has no concrete checked type \
             application to specialize (chelis#1226; [05-UNS-1])"
        ),
    ))
}

/// Deterministic specialized-symbol name: the def's canonical name plus an
/// FNV-1a hash of the canonical type-application key. No randomized hasher
/// may be used here — the emitted symbol set must be identical across
/// builds.
fn mono_specialization_symbol(name: &str, canonical_key: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in canonical_key.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{name}__mono_{hash:016x}")
}

/// Purpose-built canonical rendering of one host type term for the
/// specialization interning key (harden-bounded-monomorphization D4).
/// Every variant has an explicit, stable spelling — the key must not depend
/// on `derive(Debug)` output shape.
fn write_canonical_host_type_key(ty: &HostTypeTerm, out: &mut String) {
    use std::fmt::Write;
    match ty {
        HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(prim)) => {
            out.push_str("s:");
            out.push_str(prim.name());
        }
        HostTypeTerm::Scalar(HostPrecisionTerm::Variable(name)) => {
            out.push_str("sv:");
            out.push_str(name);
        }
        HostTypeTerm::Fn(params, ret) => {
            out.push_str("fn(");
            for param in params {
                write_canonical_host_type_key(param, out);
                out.push(',');
            }
            out.push_str(")->");
            write_canonical_host_type_key(ret, out);
        }
        HostTypeTerm::Adt(name, args) => {
            out.push_str("adt:");
            out.push_str(name);
            out.push('[');
            for arg in args {
                write_canonical_host_type_key(arg, out);
                out.push(',');
            }
            out.push(']');
        }
        HostTypeTerm::List(inner) => {
            out.push_str("list[");
            write_canonical_host_type_key(inner, out);
            out.push(']');
        }
        HostTypeTerm::Dict(key, value) => {
            out.push_str("dict[");
            write_canonical_host_type_key(key, out);
            out.push(',');
            write_canonical_host_type_key(value, out);
            out.push(']');
        }
        HostTypeTerm::Tuple(items) => {
            out.push_str("tuple(");
            for item in items {
                write_canonical_host_type_key(item, out);
                out.push(',');
            }
            out.push(')');
        }
        HostTypeTerm::Tensor(tensor) => {
            out.push_str("tensor[");
            for dim in &tensor.dims {
                match dim {
                    crate::dag::DimInfo::Lit(extent) => {
                        let _ = write!(out, "{extent}");
                    }
                    crate::dag::DimInfo::Named(name, Some(extent)) => {
                        let _ = write!(out, "{name}={extent}");
                    }
                    crate::dag::DimInfo::Named(name, None) => out.push_str(name),
                }
                out.push(',');
            }
            out.push(';');
            out.push_str(tensor.precision.name());
            out.push(']');
        }
        // Unresolved states cannot reach an interning key (the concreteness
        // gate rejects them first), but the writer stays total with stable
        // spellings so a future caller cannot silently fall back to `Debug`.
        HostTypeTerm::PolymorphicTensor(tensor) => {
            out.push_str("ptensor[");
            match &tensor.shape {
                HostShapeTerm::Concrete(dims) => {
                    out.push_str("concrete(");
                    for dim in dims {
                        match dim {
                            DimInfo::Lit(extent) => {
                                let _ = write!(out, "lit:{extent},");
                            }
                            DimInfo::Named(name, Some(extent)) => {
                                let _ = write!(out, "named:{name}={extent},");
                            }
                            DimInfo::Named(name, None) => {
                                let _ = write!(out, "named:{name},");
                            }
                        }
                    }
                    out.push(')');
                }
                HostShapeTerm::Polymorphic(slots) => {
                    out.push_str("poly(");
                    for slot in slots {
                        match slot {
                            crate::host_type_state::HostShapeSlot::Dim(dim) => match dim {
                                DimInfo::Lit(extent) => {
                                    let _ = write!(out, "dim:lit:{extent},");
                                }
                                DimInfo::Named(name, Some(extent)) => {
                                    let _ = write!(out, "dim:named:{name}={extent},");
                                }
                                DimInfo::Named(name, None) => {
                                    let _ = write!(out, "dim:named:{name},");
                                }
                            },
                            crate::host_type_state::HostShapeSlot::RankVariable(name) => {
                                let _ = write!(out, "rank:{name},");
                            }
                        }
                    }
                    out.push(')');
                }
            }
            out.push(';');
            match &tensor.precision {
                HostPrecisionTerm::Concrete(prim) => {
                    out.push_str("precision:");
                    out.push_str(prim.name());
                }
                HostPrecisionTerm::Variable(name) => {
                    out.push_str("precision-var:");
                    out.push_str(name);
                }
            }
            out.push(']');
        }
        HostTypeTerm::TypeVariable(name) => {
            out.push_str("tv:");
            out.push_str(name);
        }
        HostTypeTerm::InferenceVariable(var) => {
            let _ = write!(out, "iv:{}", var.0);
        }
        HostTypeTerm::Option(inner) => {
            out.push_str("option[");
            write_canonical_host_type_key(inner, out);
            out.push(']');
        }
        HostTypeTerm::MappedFile => out.push_str("mappedfile"),
        HostTypeTerm::Unit => out.push_str("unit"),
        HostTypeTerm::Never => out.push_str("never"),
    }
}

/// The canonical interning key for `(def identity, type application)`.
fn mono_specialization_key(
    canonical_name: &str,
    param_tys: &[HostTypeTerm],
    ret_ty: &HostTypeTerm,
) -> String {
    let mut key = String::new();
    key.push_str(canonical_name);
    key.push('\u{1}');
    for param_ty in param_tys {
        write_canonical_host_type_key(param_ty, &mut key);
        key.push('\u{1}');
    }
    key.push('\u{1}');
    write_canonical_host_type_key(ret_ty, &mut key);
    key
}

/// Register `symbol` as minted from `canonical_key`. `Err` carries the
/// previously registered key when the same symbol arrives for a different
/// key — an FNV-1a collision that would otherwise silently collapse two
/// instantiations into one C definition.
fn register_mono_symbol_key(
    state: &mut MonoSpecializationState,
    symbol: &str,
    canonical_key: &str,
) -> Result<(), String> {
    match state.symbol_keys.get(symbol) {
        Some(existing) if existing != canonical_key => Err(existing.clone()),
        Some(_) => Ok(()),
        None => {
            state
                .symbol_keys
                .insert(symbol.to_string(), canonical_key.to_string());
            Ok(())
        }
    }
}

/// Return the specialized symbol for `(canonical def, param_tys, ret_ty)`,
/// lowering the specialized definition first if this is the key's first
/// request. The memo and in-progress entries are registered before the body
/// is lowered, so recursive edges encountered mid-emission resolve to this
/// symbol.
fn ensure_mono_specialization(
    app_expr: &Expr,
    name: &str,
    body: &Expr,
    param_tys: &[HostTypeTerm],
    ret_ty: &HostTypeTerm,
    program: &HostLoweringSession<'_>,
    tensor_helpers: &TensorHelperSink,
) -> Result<String, crate::lower::LowerDiagnostic> {
    let canonical_key = mono_specialization_key(name, param_tys, ret_ty);
    if let Some(symbol) =
        MONO_SPECIALIZATIONS.with(|state| state.borrow().memo.get(&canonical_key).cloned())
    {
        return Ok(symbol);
    }
    let symbol = mono_specialization_symbol(name, &canonical_key);
    let collision = MONO_SPECIALIZATIONS
        .with(|state| register_mono_symbol_key(&mut state.borrow_mut(), &symbol, &canonical_key));
    if let Err(existing_key) = collision {
        return Err(host_expr_lowering_error(
            app_expr,
            format!(
                "internal: specialized symbol `{symbol}` collides across two distinct \
                 canonical signatures (`{existing_key}` vs `{canonical_key}`); refusing to \
                 emit a definition that serves either (chelis#1226; [05-UNS-1])"
            ),
        ));
    }

    let params = as_list(body)
        .filter(|fn_list| tag(fn_list) == Some(DeepTag::Fn))
        .and_then(|fn_list| {
            let fn_kids = children(fn_list);
            let params_list = fn_kids.first().and_then(as_list)?;
            if tag(params_list) != Some(DeepTag::Params) {
                return None;
            }
            let body_expr = fn_kids.get(1)?.clone();
            Some((children(params_list).to_vec(), body_expr))
        });
    let Some((param_exprs, body_expr)) = params else {
        return Err(host_expr_lowering_error(
            app_expr,
            format!(
                "generic host call `{name}` does not name a function-bodied \
                 definition (chelis#1226; [05-UNS-1])"
            ),
        ));
    };
    if param_exprs.len() != param_tys.len() {
        return Err(host_expr_lowering_error(
            app_expr,
            format!(
                "generic host call `{name}` supplies {} argument(s) for {} \
                 parameter(s) (chelis#1226; [05-UNS-1])",
                param_tys.len(),
                param_exprs.len()
            ),
        ));
    }
    let mut spec_scope = UnordMap::new();
    let mut spec_params = Vec::with_capacity(param_exprs.len());
    for (param, param_ty) in param_exprs.iter().zip(param_tys.iter()) {
        let Some(pname) = param_name(param) else {
            return Err(host_expr_lowering_error(
                app_expr,
                format!(
                    "generic host call `{name}` has an unnameable parameter \
                     (chelis#1226; [05-UNS-1])"
                ),
            ));
        };
        spec_scope.insert(pname.clone(), param_ty.clone());
        spec_params.push(HostParam {
            name: pname,
            ty: param_ty.clone(),
        });
    }

    MONO_SPECIALIZATIONS.with(|state| {
        let mut state = state.borrow_mut();
        state.memo.insert(canonical_key, symbol.clone());
        state.in_progress.push(InProgressMonoSpecialization {
            def_name: name.to_string(),
            param_tys: param_tys.to_vec(),
            ret_ty: ret_ty.clone(),
        });
    });
    let lowered = lower_mono_specialized_function(MonoSpecializedFunctionInput {
        symbol: &symbol,
        declaration_name: name,
        params: spec_params,
        ret_ty,
        body_expr: &body_expr,
        fn_expr: body,
        program,
        spec_scope: &spec_scope,
        collect_execution: tensor_helpers.collect_execution,
        collect_trace: tensor_helpers.collect_trace,
    });
    MONO_SPECIALIZATIONS.with(|state| {
        state.borrow_mut().in_progress.pop();
    });
    let function = lowered?;
    MONO_SPECIALIZATIONS.with(|state| state.borrow_mut().functions.push(function));
    Ok(symbol)
}

struct MonoSpecializedFunctionInput<'a> {
    symbol: &'a str,
    declaration_name: &'a str,
    params: Vec<HostParam>,
    ret_ty: &'a HostTypeTerm,
    body_expr: &'a Expr,
    fn_expr: &'a Expr,
    program: &'a HostLoweringSession<'a>,
    spec_scope: &'a UnordMap<String, HostTypeTerm>,
    collect_execution: bool,
    collect_trace: bool,
}

fn lower_mono_specialized_function(
    input: MonoSpecializedFunctionInput<'_>,
) -> Result<LoweredHostFunction, crate::lower::LowerDiagnostic> {
    let MonoSpecializedFunctionInput {
        symbol,
        declaration_name,
        mut params,
        ret_ty,
        body_expr,
        fn_expr,
        program,
        spec_scope,
        collect_execution,
        collect_trace,
    } = input;
    let body_expr = inline_local_callable_lets(body_expr);
    let mut fn_tensor_helpers =
        TensorHelperSink::for_declaration(collect_execution, collect_trace, declaration_name);
    // chelis#1201: pin this specialization's type variables for the body.
    // The body's checked types are the generic ones the checker recorded, so
    // without this a generic ADT constructed inside the body (coral#26's
    // `Hamt` nodes) reads as unresolved even though the call site pinned it.
    //
    // Solve against the `fn` node's OWN recorded type, NOT the declared
    // `defsig`. The two name their variables in different spaces: a
    // signature says `a`, while every node the checker stamped inside the
    // body says `t376`. A substitution keyed on the declared names installs
    // correctly and then matches nothing, because no body node ever mentions
    // `a`. Only the recorded generic signature shares the body's namespace.
    let mut subst: UnordMap<String, HostTypeTerm> = UnordMap::new();
    if let Some((generic_params, generic_ret)) = expr_fn_type(fn_expr) {
        for (generic, actual) in generic_params.iter().zip(params.iter()) {
            solve_host_type_vars(generic, &actual.ty, &mut subst);
        }
        solve_host_type_vars(&generic_ret, ret_ty, &mut subst);
    }
    let _subst_guard = ActiveTypeSubstGuard::push(subst);
    let mut host_body = lower_host_expr_with_expected(
        &body_expr,
        program,
        spec_scope,
        &mut fn_tensor_helpers,
        Some(ret_ty),
    )?;
    // Span survival (spec/design/chelis_span_survival.md §2.3): the fn
    // form's and body's source regions surface on the specialized body.
    host_body.append_merged_span(body_expr.span_id());
    host_body.append_merged_span(fn_expr.span_id());
    refine_function_params_from_body(&mut params, &host_body);
    let ret_ty = if ret_ty.is_unresolved() {
        host_expr_type(&host_body)
    } else {
        ret_ty.clone()
    };
    let (tensor_helpers, products) = fn_tensor_helpers.into_parts();
    Ok(LoweredHostFunction {
        function: HostFunction {
            helper_result_claim_axes: Vec::new(),
            name: symbol.to_string(),
            params,
            ret_ty,
            body: host_body,
            tensor_helpers,
            origin: HostFunctionOrigin::Monomorphized,
            specialization: None,
            summary_rejections: Vec::new(),
        },
        products,
    })
}

/// Whether `name` has the narrow chelis#935 specialization shape: a
/// zero-argument type-polymorphic function whose body is exactly a nullary
/// generic ADT constructor. Broader generic functions keep their existing
/// lowering path; eagerly inlining those can expand recursive library code
/// exponentially.
fn top_level_fn_is_nullary_generic_constructor_wrapper(
    program: &HostLoweringSession<'_>,
    name: &str,
) -> bool {
    let Some((params, ret)) = lookup_declared_fn_type(program, name) else {
        return false;
    };
    if !params.is_empty() || !ret.is_unresolved() {
        return false;
    }
    let Some(body) = find_top_level_def_expr(program.exprs(), name) else {
        return false;
    };
    let Expr::List(fn_list, _) = body else {
        return false;
    };
    if tag(fn_list) != Some(DeepTag::Fn) {
        return false;
    }
    let fn_kids = children(fn_list);
    if fn_kids
        .first()
        .and_then(as_list)
        .is_none_or(|params| tag(params) != Some(DeepTag::Params) || !children(params).is_empty())
    {
        return false;
    }
    let mut constructor = fn_kids.get(1);
    while let Some(Expr::MetaExpr(meta, _)) = constructor {
        constructor = Some(&meta.expr);
    }
    let Some(Expr::List(var, _)) = constructor else {
        return false;
    };
    if tag(var) != Some(DeepTag::Var) {
        return false;
    }
    let Some(ctor_name) = children(var).first().and_then(symbol_name) else {
        return false;
    };
    // Predicate disposition (chelis#1271): an ambiguous constructor name
    // answers `false`, which only declines this narrow specialization.
    // The wrapper's body still lowers through the bare-variable path,
    // which rejects the same ambiguity, so the program fails closed.
    lookup_adt_constructor_definition(program, ctor_name)
        .is_some_and(|definition| !definition.parameters.is_empty() && definition.is_nullary())
}

/// Every name this expression REFERENCES, as a `(var {} name)` node
/// (chelis#2163).
///
/// A deliberate over-approximation of the free names: a name bound inside the
/// expression is collected too. Capture avoidance only ever renames a binder,
/// which preserves meaning, so an extra rename is harmless while a missed one
/// is a miscompile.
fn collect_referenced_names(expr: &Expr, out: &mut UnordSet<String>) {
    if let Some((DeepTag::Var, _, kids)) = stamped_parts(expr)
        && let Some(name) = kids.first().and_then(symbol_name)
    {
        out.insert(name.to_string());
    }
    match expr {
        Expr::MetaExpr(meta, _) => collect_referenced_names(&meta.expr, out),
        Expr::List(list, _) => {
            for child in &list.elements {
                collect_referenced_names(child, out);
            }
        }
        Expr::BareList(items, _) => {
            for child in items {
                collect_referenced_names(child, out);
            }
        }
        Expr::Node(node, _) => {
            for child in node.children_slice() {
                collect_referenced_names(child, out);
            }
        }
        _ => {}
    }
}

/// EVERY name occurring anywhere in `expr`, bound as well as referenced
/// (chelis#2163).
///
/// This is the avoid-set feeder, and it is deliberately the widest possible
/// collection: a fresh binder name must collide with nothing, and a name is
/// dangerous whether it is referenced or merely bound. Collecting only `var`
/// references misses a binder that is never read, and a fresh name landing on
/// one of those captures it - the defect that shipped in the first revision of
/// this fix. Over-approximating what to AVOID only makes a fresh name more
/// exotic; under-approximating it miscompiles.
fn collect_occurring_names(expr: &Expr, out: &mut UnordSet<String>) {
    if let Expr::Atom(Atom::Name(name), _) = expr {
        out.insert(name.clone());
    }
    match expr {
        Expr::MetaExpr(meta, _) => {
            collect_occurring_names(&meta.expr, out);
            // Annotation expressions carry names too, and `substitute_expr`
            // rewrites them, so they are in scope for collisions.
            let _ = meta.metadata.map_expressions(&mut |value, _| {
                collect_occurring_names(value, out);
                value.clone()
            });
        }
        Expr::List(list, _) => {
            for child in &list.elements {
                collect_occurring_names(child, out);
            }
        }
        Expr::BareList(items, _) => {
            for child in items {
                collect_occurring_names(child, out);
            }
        }
        Expr::Node(node, _) => {
            for child in node.children_slice() {
                collect_occurring_names(child, out);
            }
        }
        _ => {}
    }
}

/// The names referenced by the replacements that can still reach this scope
/// (chelis#2163).
///
/// A replacement whose parameter is shadowed here is never inserted below, so
/// it cannot be captured and its binder needs no rename. The narrowing is not
/// cosmetic: a rename is only meaning-preserving when the fresh name is truly
/// fresh, so every rename that fires is a risk, and one that cannot prevent a
/// capture is pure risk. Removing this narrowing is what made
/// `a_shadowed_parameter_does_not_trigger_a_rename` miscompile.
fn live_replacement_names(
    substitutions: &UnordMap<String, Expr>,
    shadowed: &UnordSet<String>,
) -> UnordSet<String> {
    let mut out = UnordSet::default();
    for (name, replacement) in substitutions.to_sorted() {
        if shadowed.contains(name) {
            continue;
        }
        collect_referenced_names(replacement, &mut out);
    }
    out
}

/// A binder name that collides with nothing in `avoid` (chelis#2163).
///
/// Deterministic: the first free `<name>__inl<k>`, never a global counter, so
/// the emitted source of an unchanged program does not move between builds.
/// `avoid` must be fed by [`collect_occurring_names`], not by the referenced
/// names alone: an existing `x__inl1` binder that nothing reads is still a
/// collision.
fn fresh_binder_name(name: &str, avoid: &UnordSet<String>) -> String {
    (1..)
        .map(|index| format!("{name}__inl{index}"))
        .find(|candidate| !avoid.contains(candidate))
        .expect("an unused binder name exists")
}

/// Rename bound occurrences of `renames` inside `expr` (chelis#2163).
///
/// A rename, not a substitution: it rewrites the NAME inside each `(var {}
/// name)` node and keeps that node's metadata, because the checker's recorded
/// type travels in it and host lowering rejects a `var` without one.
///
/// It mirrors [`substitute_expr`]'s shadowing, so an inner binder that spells
/// a renamed name stops the rename exactly where it would stop a
/// substitution.
fn rename_bound_names(
    expr: &Expr,
    renames: &[(String, String)],
    shadowed: &UnordSet<String>,
) -> Expr {
    if renames.is_empty() {
        return expr.clone();
    }
    let renamed_var = |kids: &[Expr]| -> Option<String> {
        kids.first()
            .and_then(symbol_name)
            .filter(|name| !shadowed.contains(*name))
            .and_then(|name| {
                renames
                    .iter()
                    .find(|(from, _)| from == name)
                    .map(|(_, to)| to.clone())
            })
    };
    match expr {
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                metadata: meta.metadata.clone(),
                expr: Box::new(rename_bound_names(&meta.expr, renames, shadowed)),
            },
            *span,
        ),
        Expr::Node(node, span) if node.tag() == DeepTag::Var => {
            match renamed_var(node.children_slice()) {
                Some(to) => Expr::Node(
                    Box::new(chelis_deep::node::Node::new(
                        DeepTag::Var,
                        node.meta().clone(),
                        vec![Expr::Atom(Atom::Name(to), *span)],
                    )),
                    *span,
                ),
                None => expr.clone(),
            }
        }
        Expr::List(list, span) if tag(list) == Some(DeepTag::Var) => {
            match renamed_var(children(list)) {
                Some(to) => {
                    let mut elements = list.elements.clone();
                    if let Some(slot) = elements.get_mut(2) {
                        *slot = Expr::Atom(Atom::Name(to), *span);
                    }
                    Expr::List(List { elements }, *span)
                }
                None => expr.clone(),
            }
        }
        Expr::List(list, span) if tag(list) == Some(DeepTag::Fn) => {
            let kids = children(list);
            let mut inner = shadowed.clone();
            if let Some(params) = kids.first().and_then(as_list) {
                for param in children(params) {
                    if let Some(name) = param_name(param) {
                        inner.insert(name);
                    }
                }
            }
            let elements = list
                .elements
                .iter()
                .enumerate()
                .map(|(index, child)| {
                    if index == 3 {
                        rename_bound_names(child, renames, &inner)
                    } else {
                        child.clone()
                    }
                })
                .collect();
            Expr::List(List { elements }, *span)
        }
        Expr::List(list, span) if tag(list) == Some(DeepTag::Let) => {
            let kids = children(list);
            let mut inner = shadowed.clone();
            if let Some(bind_list) = kids.first().and_then(as_list)
                && tag(bind_list) == Some(DeepTag::Bind)
            {
                let bind_kids = children(bind_list);
                for index in (0..bind_kids.len()).step_by(2) {
                    if let Some(name) = bind_kids.get(index).and_then(symbol_name) {
                        inner.insert(name.to_string());
                    }
                }
            }
            let elements = list
                .elements
                .iter()
                .enumerate()
                .map(|(index, child)| match index {
                    2 => rename_bound_names(child, renames, shadowed),
                    3 => rename_bound_names(child, renames, &inner),
                    _ => child.clone(),
                })
                .collect();
            Expr::List(List { elements }, *span)
        }
        Expr::List(list, span) => Expr::List(
            List {
                elements: list
                    .elements
                    .iter()
                    .map(|child| rename_bound_names(child, renames, shadowed))
                    .collect(),
            },
            *span,
        ),
        Expr::Node(node, span) => Expr::Node(
            Box::new(chelis_deep::node::Node::new(
                node.tag(),
                node.meta().clone(),
                node.children_slice()
                    .iter()
                    .map(|child| rename_bound_names(child, renames, shadowed))
                    .collect(),
            )),
            *span,
        ),
        Expr::BareList(items, span) => Expr::BareList(
            items
                .iter()
                .map(|child| rename_bound_names(child, renames, shadowed))
                .collect(),
            *span,
        ),
        other => other.clone(),
    }
}

/// Rewrite the name a binder-position child spells, keeping its shape and any
/// metadata (chelis#2163).
fn rename_binder_child(expr: &Expr, to: &str) -> Expr {
    match expr {
        Expr::Atom(Atom::Name(_), span) => Expr::Atom(Atom::Name(to.to_string()), *span),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                metadata: meta.metadata.clone(),
                expr: Box::new(rename_binder_child(&meta.expr, to)),
            },
            *span,
        ),
        Expr::List(list, span) => {
            let mut elements = list.elements.clone();
            if let Some(first) = elements.first_mut()
                && symbol_name(first).is_some()
            {
                *first = Expr::Atom(Atom::Name(to.to_string()), *span);
            }
            Expr::List(List { elements }, *span)
        }
        other => other.clone(),
    }
}

fn substitute_expr(
    expr: &Expr,
    substitutions: &UnordMap<String, Expr>,
    shadowed: &UnordSet<String>,
) -> Expr {
    record_host_work(|profile| profile.substitution_nodes += 1);
    match expr {
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                metadata: meta
                    .metadata
                    .map_expressions(&mut |value, _| {
                        substitute_expr(value, substitutions, shadowed)
                    })
                    .expect("substitution preserves annotation roles"),
                expr: Box::new(substitute_expr(&meta.expr, substitutions, shadowed)),
            },
            *span,
        ),
        Expr::List(list, _span) if tag(list) == Some(DeepTag::Var) => {
            if let Some(name) = children(list).first().and_then(symbol_name)
                && !shadowed.contains(name)
                && let Some(replacement) = substitutions.get(name)
            {
                return replacement.clone();
            }
            expr.clone()
        }
        Expr::List(list, span) if tag(list) == Some(DeepTag::App) => {
            // Applied uppercase heads are constructor syntax (spec/01 §3.2),
            // not value references. Higher-order specialization may replace
            // a bare uppercase parameter elsewhere, but it must not rewrite
            // the callee of `N(x)` into the parameter's argument and thereby
            // turn a checked constructor application into an ordinary C call.
            let elements = list
                .elements
                .iter()
                .enumerate()
                .map(|(index, child)| {
                    if index == 2 && is_constructor_application_head(child) {
                        child.clone()
                    } else {
                        substitute_expr(child, substitutions, shadowed)
                    }
                })
                .collect();
            Expr::List(List { elements }, *span)
        }
        Expr::List(list, span) if tag(list) == Some(DeepTag::Fn) => {
            let kids = children(list);
            let params_list = kids.first().and_then(as_list);
            // A `fn` with no params list contributes no binder names. Spelled
            // out rather than defaulted: an empty list is the stated outcome
            // here, not a fallback (loud_unsupported.md B2.5).
            let param_names: Vec<Option<String>> = match params_list {
                Some(params) => children(params).iter().map(param_name).collect(),
                None => Vec::new(),
            };
            // chelis#2163: the arguments being substituted in carry the
            // CALLER's names. A binder here that spells one of them would
            // capture it, so rename the binder first. Renaming a binder
            // preserves meaning ONLY when the fresh name is genuinely unused,
            // so `live` stays narrow (rename no more than necessary) while
            // `avoid` stays wide (collide with nothing).
            let mut bound = shadowed.clone();
            for name in param_names.iter().flatten() {
                bound.insert(name.clone());
            }
            let live = live_replacement_names(substitutions, &bound);
            let mut avoid = UnordSet::default();
            for (_, replacement) in substitutions.to_sorted() {
                collect_occurring_names(replacement, &mut avoid);
            }
            collect_occurring_names(expr, &mut avoid);
            for name in param_names.iter().flatten() {
                avoid.insert(name.clone());
            }
            let mut renames: Vec<(String, String)> = Vec::new();
            for name in param_names.iter().flatten() {
                if live.contains(name) {
                    let fresh = fresh_binder_name(name, &avoid);
                    avoid.insert(fresh.clone());
                    renames.push((name.clone(), fresh));
                }
            }
            let mut next_shadowed = shadowed.clone();
            for name in param_names.iter().flatten() {
                match renames.iter().find(|(from, _)| from == name) {
                    Some((_, fresh)) => next_shadowed.insert(fresh.clone()),
                    None => next_shadowed.insert(name.clone()),
                };
            }
            let mut elements = Vec::with_capacity(list.elements.len());
            elements.push(list.elements[0].clone());
            elements.push(list.elements[1].clone());
            if let Some(params) = kids.first() {
                elements.push(match (as_list(params), renames.is_empty()) {
                    (Some(params_list), false) => Expr::List(
                        List {
                            elements: params_list
                                .elements
                                .iter()
                                .map(|param| {
                                    match param_name(param).and_then(|name| {
                                        renames.iter().find(|(from, _)| *from == name).cloned()
                                    }) {
                                        Some((_, fresh)) => rename_binder_child(param, &fresh),
                                        None => param.clone(),
                                    }
                                })
                                .collect(),
                        },
                        *span,
                    ),
                    _ => params.clone(),
                });
            }
            if let Some(body) = kids.get(1) {
                let body = rename_bound_names(body, &renames, &UnordSet::new());
                elements.push(substitute_expr(&body, substitutions, &next_shadowed));
            }
            Expr::List(List { elements }, *span)
        }
        Expr::List(list, span) if tag(list) == Some(DeepTag::Let) => {
            let kids = children(list);
            let bind_list = kids
                .first()
                .and_then(as_list)
                .filter(|bind_list| tag(bind_list) == Some(DeepTag::Bind));
            // As above: a `let` with no `bind` list binds nothing here.
            let bound_names: Vec<Option<String>> = match bind_list {
                Some(bind_list) => {
                    let bind_kids = children(bind_list);
                    (0..bind_kids.len())
                        .step_by(2)
                        .map(|index| {
                            bind_kids
                                .get(index)
                                .and_then(symbol_name)
                                .map(str::to_string)
                        })
                        .collect()
                }
                None => Vec::new(),
            };
            // chelis#2163: same capture rule as the `fn` arm above, for a
            // `let` binder, with the same narrow-`live`/wide-`avoid` split.
            let mut bound = shadowed.clone();
            for name in bound_names.iter().flatten() {
                bound.insert(name.clone());
            }
            let live = live_replacement_names(substitutions, &bound);
            let mut avoid = UnordSet::default();
            for (_, replacement) in substitutions.to_sorted() {
                collect_occurring_names(replacement, &mut avoid);
            }
            collect_occurring_names(expr, &mut avoid);
            let mut renames: Vec<(String, String)> = Vec::new();
            for name in bound_names.iter().flatten() {
                if live.contains(name) {
                    let fresh = fresh_binder_name(name, &avoid);
                    avoid.insert(fresh.clone());
                    renames.push((name.clone(), fresh));
                }
            }
            let mut next_shadowed = shadowed.clone();
            for name in bound_names.iter().flatten() {
                match renames.iter().find(|(from, _)| from == name) {
                    Some((_, fresh)) => next_shadowed.insert(fresh.clone()),
                    None => next_shadowed.insert(name.clone()),
                };
            }
            if kids.len() >= 2 {
                let mut rebuilt = list.elements.clone();
                // A binding's value is evaluated before that name is in
                // scope, so a rename applies only to the values that FOLLOW
                // it and to the body. Renaming an earlier value would rewrite
                // a reference to an outer binding of the same name.
                rebuilt[2] = match (as_list(&list.elements[2]), renames.is_empty()) {
                    (Some(bind_list), false) => {
                        // `bind_list.elements` is [tag, metadata, name, value,
                        // name, value, ...]: the names start at index 2.
                        let mut applied: Vec<(String, String)> = Vec::new();
                        let mut rebuilt_binds = bind_list.elements.clone();
                        // A binding's own value is evaluated BEFORE its name is
                        // in scope, so this binding's rename must not reach it:
                        // `x = add(x, m)` refers to an outer `x`, and rewriting
                        // it to `x__inl1 = add(x__inl1, m)` is a self-reference
                        // that fails ownership lowering. The rename is recorded
                        // only after its own value slot has been rewritten, so
                        // it applies to LATER values and to the body.
                        let mut pending: Option<(String, String)> = None;
                        for (index, slot) in rebuilt_binds.iter_mut().enumerate().skip(2) {
                            if index % 2 == 0 {
                                let renamed = symbol_name(slot).and_then(|name| {
                                    renames.iter().find(|(from, _)| from == name).cloned()
                                });
                                if let Some((from, fresh)) = renamed {
                                    *slot = Expr::Atom(Atom::Name(fresh.clone()), *span);
                                    pending = Some((from, fresh));
                                }
                            } else {
                                *slot = rename_bound_names(slot, &applied, &UnordSet::new());
                                if let Some(entry) = pending.take() {
                                    applied.push(entry);
                                }
                            }
                        }
                        substitute_expr(
                            &Expr::List(
                                List {
                                    elements: rebuilt_binds,
                                },
                                *span,
                            ),
                            substitutions,
                            shadowed,
                        )
                    }
                    _ => substitute_expr(&list.elements[2], substitutions, shadowed),
                };
                let body = rename_bound_names(&list.elements[3], &renames, &UnordSet::new());
                rebuilt[3] = substitute_expr(&body, substitutions, &next_shadowed);
                Expr::List(List { elements: rebuilt }, *span)
            } else {
                // A `let` carries exactly two children (`role.rs`'s arity
                // table: `DeepTag::Let => Fixed(2)`), so this arm is
                // unreachable for well-formed Deep and is retained only as
                // the defensive fallback it always was. It matters that the
                // substitution happens HERE rather than before the branch:
                // run eagerly, its result was discarded in the ordinary
                // two-child case, so every nested binder substituted its
                // subtree twice and one inline cost 2^(binder count)
                // (chelis#2181).
                let elements = list
                    .elements
                    .iter()
                    .map(|child| substitute_expr(child, substitutions, shadowed))
                    .collect();
                Expr::List(List { elements }, *span)
            }
        }
        Expr::List(list, span) => Expr::List(
            List {
                elements: list
                    .elements
                    .iter()
                    .map(|child| substitute_expr(child, substitutions, shadowed))
                    .collect(),
            },
            *span,
        ),
        _ => expr.clone(),
    }
}

fn is_constructor_application_head(expr: &Expr) -> bool {
    let expr = match expr {
        Expr::MetaExpr(meta, _) => &meta.expr,
        direct => direct,
    };
    let Some(var) = as_list(expr).filter(|list| tag(list) == Some(DeepTag::Var)) else {
        return false;
    };
    children(var)
        .first()
        .and_then(symbol_name)
        .map(terminal_name)
        .and_then(|name| name.chars().next())
        .is_some_and(|first| first.is_ascii_uppercase())
}

/// Whether a let-bound value is a callable expression that can be β-substituted
/// into every use site in the body. The host backend can only lower these
/// callable forms directly in callee position of an `app` node, so an alias
/// like `let g = grad(f); g(x)` must be rewritten to the inline form
/// `(grad(f))(x)` before host-lane lowering. Recognizes:
///   - `(fn (params …) body)` — anonymous function
///   - `(grad … fn …)` — gradient transform
///   - `(vmap … fn …)` — vmap transform
///   - `(vmap-grad … fn …)` — vmap of grad
///
/// Looks through a wrapping `MetaExpr` so type-annotated bindings still match.
fn is_inlinable_callable_binding_value(expr: &Expr) -> bool {
    match expr {
        Expr::List(inner, _) => {
            matches!(
                tag(inner),
                Some(DeepTag::Fn) | Some(DeepTag::Grad) | Some(DeepTag::Vmap)
            ) || inner.unknown_tag_symbol() == Some("vmap-grad")
        }
        Expr::MetaExpr(meta, _) => is_inlinable_callable_binding_value(&meta.expr),
        _ => false,
    }
}

fn inline_local_callable_lets(expr: &Expr) -> Expr {
    let Expr::List(list, span) = expr else {
        return expr.clone();
    };
    if tag(list) != Some(DeepTag::Let) {
        return Expr::List(
            List {
                elements: list
                    .elements
                    .iter()
                    .map(inline_local_callable_lets)
                    .collect(),
            },
            *span,
        );
    }

    let kids = children(list);
    let Some(bind_list) = kids.first().and_then(as_list) else {
        return expr.clone();
    };
    if tag(bind_list) != Some(DeepTag::Bind) {
        return expr.clone();
    }

    let bind_kids = children(bind_list);
    let mut rebuilt_pairs = Vec::<(String, Expr)>::new();
    let mut body = kids
        .get(1)
        .map(inline_local_callable_lets)
        .unwrap_or_else(|| {
            Expr::List(
                List {
                    elements: Vec::new(),
                },
                *span,
            )
        });

    for index in (0..bind_kids.len()).step_by(2).rev() {
        let Some(name) = bind_kids.get(index).and_then(symbol_name) else {
            continue;
        };
        let Some(value) = bind_kids.get(index + 1) else {
            continue;
        };
        let value = inline_local_callable_lets(value);
        if is_inlinable_callable_binding_value(&value) {
            // β-substitute the callable into every use site in the body.
            // This applies to local fn bindings (`let f = fn (x) => …; f(y)`),
            // and also to higher-order callable forms `grad`, `vmap`, and
            // `vmap-grad` (`let g = grad(f); g(x)`). The grad/vmap forms are
            // not first-class host values — the host backend recognizes them
            // only when they appear directly in callee position of an `app`
            // (the inline form `grad(f)(x)` lowers cleanly). Inlining the
            // alias rewrites the let-bound form to the inline form so the
            // host backend can lower it the same way.
            body = substitute_expr(
                &body,
                &UnordMap::from([(name.to_string(), value)]),
                &UnordSet::new(),
            );
        } else {
            rebuilt_pairs.push((name.to_string(), value));
        }
    }

    if rebuilt_pairs.is_empty() {
        return body;
    }

    rebuilt_pairs.reverse();
    let mut rebuilt_bind = vec![bind_list.elements[0].clone(), bind_list.elements[1].clone()];
    for (name, value) in rebuilt_pairs {
        rebuilt_bind.push(Expr::Atom(Atom::Name(name), *span));
        rebuilt_bind.push(value);
    }
    Expr::List(
        List {
            elements: vec![
                list.elements[0].clone(),
                list.elements[1].clone(),
                Expr::List(
                    List {
                        elements: rebuilt_bind,
                    },
                    *span,
                ),
                body,
            ],
        },
        *span,
    )
}

fn expr_needs_host_lane_tensor_lowering(expr: &Expr, program: &HostLoweringSession<'_>) -> bool {
    let graph = top_level_fn_call_graph(program);
    let recursive = recursive_top_level_fn_names_from_graph(&graph);
    let fn_names = graph.keys().cloned().collect::<BTreeSet<_>>();
    collect_called_top_level_fns(expr, &fn_names)
        .into_iter()
        .any(|name| call_graph_reaches_any(&graph, &name, &recursive))
}

fn top_level_fn_needs_host_lane_tensor_lowering(
    program: &HostLoweringSession<'_>,
    name: &str,
) -> bool {
    let graph = top_level_fn_call_graph(program);
    let recursive = recursive_top_level_fn_names_from_graph(&graph);
    call_graph_reaches_any(&graph, name, &recursive)
}

/// A caller of `name` must fall back to a plain host function call
/// (`name(args)`) rather than inlining `name`'s body and re-summarizing
/// it, when `name`'s own tensor-helper lowering would be a *summary
/// rejection*. Inlining a rejected helper into the caller would either
/// register a false sparse/BLAS summary on the caller or rebuild the
/// caller around the rejected helper's `__tensor_*` shim; the
/// unspecialized helper is the only honest lowering, so the caller
/// emits a direct call to it.
///
/// This is the structured replacement for the previous incidental
/// trigger: the over-broad recursion classification (every function was
/// marked recursive by the reflexive `call_graph_reaches_any`) forced
/// every caller through the host-lane fallback, which happened to route
/// rejected helpers to a host call. With recursion classification
/// corrected, the fallback is driven directly off the
/// `SummaryRejection` machinery instead.
fn top_level_fn_helper_summary_rejects(
    program: &HostLoweringSession<'_>,
    name: &str,
) -> Result<bool, crate::lower::LowerDiagnostic> {
    // chelis#935/#936 bounded generic functions are specialized before
    // helper extraction. Probing their unspecialized body here would discard
    // the checked call-site application and fail before the specialization
    // path gets a chance to run.
    if top_level_fn_is_type_polymorphic(program, name)
        || top_level_fn_is_nested_rank_polymorphic(program, name)
    {
        return Ok(false);
    }
    // Guard against unbounded re-entry: while we are already inlining
    // `name` we must not recursively re-lower it to probe its
    // rejections.
    if is_inlining(name) {
        return Ok(false);
    }
    if let Some(cached) = program
        .facts
        .helper_summary_rejects
        .borrow()
        .get(name)
        .copied()
    {
        return Ok(cached);
    }
    let Some(body) = find_top_level_def_expr(program.exprs(), name) else {
        return Ok(false);
    };
    // Like a type-polymorphic declaration, an unresolved source-rate
    // template has no standalone helper summary. Its actual call is
    // classified and lowered with the caller's static controls (#1764).
    if cached_subexpr_lowering_context(program).evaluation_profile(body, &[])
        == crate::evaluation::EvaluationProfile::Legacy(
            crate::evaluation::LegacyEvaluationReason::RuntimeRate,
        )
    {
        return Ok(false);
    }
    if !matches!(body, Expr::List(list, _) if tag(list) == Some(DeepTag::Fn)) {
        return Ok(false);
    }
    record_host_work(|profile| profile.helper_summary_builds += 1);
    // This is a diagnostic count, never a capacity key. The chelis#893
    // runtime-representation inventory's `CAPACITY_FOLDS` rule keys on the
    // `saturating_*` and `checked_mul` method names wherever they appear in
    // IR-class code, so the ceiling is spelled out here rather than classified
    // as capacity arithmetic it is not; chelis#1851 tracks that name-keyed rule.
    HOST_SUMMARY_PROBE_BUILDS.with(|builds| {
        let counted = builds.get();
        builds.set(if counted == u64::MAX {
            counted
        } else {
            counted + 1
        });
    });
    let pushed = push_inlining(name);
    // This lowering is a probe: its result is inspected and discarded, so
    // it must leave no trace on specialization state
    // (harden-bounded-monomorphization D1).
    let probe_guard = MonoProbeGuard::begin();
    let lowered = crate::lower::catch_lowering_external(|| {
        lower_host_function(name, body, None, program, false, false)
    });
    drop(probe_guard);
    if pushed {
        pop_inlining(name);
    }
    // A fatal tensor-helper rejection can unwind from lower_host_function
    // (#1922). Return that diagnostic only after restoring the probe state
    // and inlining marker, and never cache it as "no summary rejection".
    let lowered = lowered?;
    // An already-returned probe error is not a build failure, including an
    // error returned by a nested summary probe. Preserve that deferral
    // (harden-bounded-monomorphization D1): this lowering was speculative,
    // and a genuine defect in the callee resurfaces — with call-free
    // attribution — when the callee is lowered for real. A failed probe
    // simply reports "no summary rejection".
    let rejects = lowered.ok().flatten().is_some_and(|mut lowered| {
        collect_function_summary_rejections(&mut lowered.function);
        !lowered.function.summary_rejections.is_empty()
    });
    program
        .facts
        .helper_summary_rejects
        .borrow_mut()
        .insert(name.to_string(), rejects);
    Ok(rejects)
}

/// A top-level expression body needs host-lane lowering when it calls a
/// top-level fn whose lowering cannot be cleanly summarized into a
/// tensor helper at the call site. Today that is any callee carrying a
/// callable (fn-pointer) parameter: the tensor-helper DAG has no
/// representation for a fn-pointer input and would coerce it into a
/// scalar-tensor placeholder, so the call must stay in the host lane
/// where the fn application lowers to a direct `f(x)` call.
///
/// This replaces the prior incidental trigger (the over-broad recursion
/// classification) for the local-wrapper-over-callable-param case.
fn expr_calls_top_level_fn_with_callable_param(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
) -> bool {
    let graph = top_level_fn_call_graph(program);
    let fn_names = graph.keys().cloned().collect::<BTreeSet<_>>();
    collect_called_top_level_fns(expr, &fn_names)
        .iter()
        .any(|name| top_level_fn_has_callable_param(program, name))
}

fn top_level_fn_has_callable_param(program: &HostLoweringSession<'_>, name: &str) -> bool {
    let Some((param_tys, _)) = lookup_declared_fn_type(program, name) else {
        return false;
    };
    param_tys
        .iter()
        .any(|ty| matches!(ty, HostTypeTerm::Fn(_, _)))
}

/// A tensor-returning body that calls a summary-rejecting top-level fn
/// must not be summarized through the tensor-helper DAG: doing so
/// inlines the rejected helper's body into the caller (registering a
/// false summary or rebuilding the caller around the rejected shim).
/// Route such bodies through host-lane lowering so the call lowers to a
/// direct host function call to the unspecialized helper.
fn expr_calls_summary_rejecting_top_level_fn(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
) -> Result<bool, crate::lower::LowerDiagnostic> {
    let graph = top_level_fn_call_graph(program);
    let fn_names = graph.keys().cloned().collect::<BTreeSet<_>>();
    // Sorted, not UnordSet order: each probe is state-isolated by
    // `MonoProbeGuard`, but probe ORDER must not vary per process either
    // (harden-bounded-monomorphization D1).
    let mut called: Vec<String> = collect_called_top_level_fns(expr, &fn_names)
        .into_iter()
        .collect();
    called.sort_unstable();
    for name in called {
        if top_level_fn_helper_summary_rejects(program, &name)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn top_level_fn_call_graph(
    program: &HostLoweringSession<'_>,
) -> BTreeMap<String, BTreeSet<String>> {
    if let Some(cached) = program.facts.call_graph.borrow().clone() {
        return cached;
    }
    let defs = cached_program_defs(program);
    let fn_names = defs
        .iter()
        .filter(|(_, body)| matches!(body, Expr::List(list, _) if tag(list) == Some(DeepTag::Fn)))
        .map(|(name, _)| name.clone())
        .collect::<BTreeSet<_>>();

    let graph: BTreeMap<String, BTreeSet<String>> = fn_names
        .iter()
        .map(|name| {
            let callees = defs
                .get(name)
                .map(|body| collect_called_top_level_fns(body, &fn_names))
                .unwrap_or_default();
            (name.clone(), callees)
        })
        .collect();
    *program.facts.call_graph.borrow_mut() = Some(graph.clone());
    graph
}

/// A pure declaration used from another declaration is a private tensor
/// helper. Its authored literal result obligations transfer into that helper's
/// DAG. Effecting definitions and definitions reached only from top-level
/// roots retain the legacy direct lowering contract.
fn top_level_fn_transfers_literal_result_claims(
    program: &HostLoweringSession<'_>,
    name: &str,
) -> bool {
    cached_def_effect_rows(program)
        .get(name)
        .is_some_and(|effects| effects.is_empty())
        && top_level_fn_call_graph(program)
            .iter()
            .any(|(caller, callees)| caller != name && callees.contains(name))
}

fn recursive_top_level_fn_names_from_graph(
    graph: &BTreeMap<String, BTreeSet<String>>,
) -> BTreeSet<String> {
    graph
        .keys()
        .filter(|name| call_graph_reaches_any(graph, name, &BTreeSet::from([(*name).clone()])))
        .cloned()
        .collect()
}

fn collect_called_top_level_fns(expr: &Expr, fn_names: &BTreeSet<String>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match current {
            Expr::MetaExpr(meta, _) => stack.push(&meta.expr),
            Expr::List(list, _) => {
                if tag(list) == Some(DeepTag::App)
                    && let Some(callee) = children(list).first().and_then(as_list)
                    && tag(callee) == Some(DeepTag::Var)
                    && let Some(name) = children(callee).first().and_then(symbol_name)
                    && fn_names.contains(name)
                {
                    out.insert(name.to_string());
                }
                stack.extend(children(list).iter());
            }
            _ => {}
        }
    }
    out
}

fn call_graph_reaches_any(
    graph: &BTreeMap<String, BTreeSet<String>>,
    start: &str,
    targets: &BTreeSet<String>,
) -> bool {
    let mut visited = UnordSet::new();
    let mut stack = graph
        .get(start)
        .into_iter()
        .flat_map(|callees| callees.iter().cloned())
        .collect::<Vec<_>>();
    while let Some(name) = stack.pop() {
        if targets.contains(&name) {
            return true;
        }
        if !visited.insert(name.clone()) {
            continue;
        }
        if let Some(next) = graph.get(&name) {
            stack.extend(next.iter().cloned());
        }
    }
    false
}

type HoistedHostLaneBindings<'expr, 'scope> = (
    Cow<'expr, Expr>,
    Cow<'scope, UnordMap<String, HostTypeTerm>>,
    Vec<HostBinding>,
);

fn hoist_host_lane_tensor_bindings<'expr, 'scope>(
    expr: &'expr Expr,
    program: &HostLoweringSession<'_>,
    scope: &'scope UnordMap<String, HostTypeTerm>,
    fn_sig: Option<&(Vec<HostTypeTerm>, HostTypeTerm)>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<HoistedHostLaneBindings<'expr, 'scope>, crate::lower::LowerDiagnostic> {
    let Expr::List(list, span) = expr else {
        return Ok((Cow::Borrowed(expr), Cow::Borrowed(scope), Vec::new()));
    };
    if tag(list) != Some(DeepTag::App) {
        return Ok((Cow::Borrowed(expr), Cow::Borrowed(scope), Vec::new()));
    }

    let kids = children(list);
    if kids.is_empty() {
        return Ok((Cow::Borrowed(expr), Cow::Borrowed(scope), Vec::new()));
    }
    // A host-only descendant beneath a tensor expression still needs a
    // host boundary: matmul(softmax(where(...)), v) cannot lower as one
    // kernel. Evaluate both matrix operands once, in source order, then
    // pass their typed values to the ordinary rank-generic matrix helper.
    // This also preserves effects in the other operand when only one has
    // a host-only descendant. Other builtins may carry static axis/list
    // arguments, so this two-tensor boundary is specific to matmul.
    let hoist_matmul_operands = kids.len() == 3 && direct_var_name(&kids[0]) == Some("matmul");
    if !hoist_matmul_operands
        && !kids
            .iter()
            .skip(1)
            .any(should_keep_tensor_expr_in_host_lane)
    {
        return Ok((Cow::Borrowed(expr), Cow::Borrowed(scope), Vec::new()));
    }

    let mut new_elements = vec![
        list.elements[0].clone(),
        list.elements[1].clone(),
        kids[0].clone(),
    ];
    let mut scoped = scope.clone();
    let mut bindings = Vec::new();

    for (index, arg) in kids.iter().enumerate().skip(1) {
        if hoist_matmul_operands || should_keep_tensor_expr_in_host_lane(arg) {
            let value = lower_host_expr(arg, program, scope, tensor_helpers)?;
            let preferred_ty = fn_sig
                .and_then(|(param_tys, _)| param_tys.get(index - 1))
                // The generic declaration is not an argument actualization.
                // Keep the checked actual's dtype and extents so the helper's
                // ordinary call inliner can bind the declaration identities.
                .filter(|ty| !ty.is_unresolved())
                .cloned()
                .or_else(|| expr_tensor_type(arg, program, scope).map(HostTypeTerm::Tensor))
                .or_else(|| {
                    let ty = expr_host_type(arg, program, scope);
                    (!ty.is_unresolved()).then_some(ty)
                });
            let ty = preferred_ty.unwrap_or_else(|| host_expr_type(&value));
            let value = force_host_expr_type(value, ty.clone());
            let name = format!("__host_tensor_arg_{index}");
            bindings.push(HostBinding {
                name: name.clone(),
                display_name: None,
                display_roots: Vec::new(),
                ty: ty.clone(),
                value,
            });
            scoped.insert(name.clone(), ty);
            new_elements.push(Expr::List(
                List {
                    elements: vec![
                        Expr::Atom(Atom::Tag(DeepTag::Var), *span),
                        Expr::Map(chelis_deep::Metadata::default(), *span),
                        Expr::Atom(Atom::Name(name), *span),
                    ],
                },
                *span,
            ));
        } else {
            new_elements.push(arg.clone());
        }
    }

    let rewritten = Expr::List(
        List {
            elements: new_elements,
        },
        *span,
    );
    record_host_work(|profile| profile.app_clone_nodes += deep_expr_nodes(&rewritten));
    Ok((Cow::Owned(rewritten), Cow::Owned(scoped), bindings))
}

fn host_fn_signature(ty: &HostTypeTerm) -> Option<(Vec<HostTypeTerm>, HostTypeTerm)> {
    match ty {
        HostTypeTerm::Fn(params, ret) => Some((params.clone(), (**ret).clone())),
        _ => None,
    }
}

fn top_level_fn_is_type_polymorphic(program: &HostLoweringSession<'_>, name: &str) -> bool {
    if let Some(cached) = program.facts.type_polymorphic.borrow().get(name).copied() {
        return cached;
    }
    // Only checker-recorded authored/synthesized signatures establish
    // source-level polymorphism. Generalized local callbacks deliberately
    // carry `authored_signature = false`.
    let polymorphic = checked_authored_function_signature(program, name).is_some_and(|signature| {
        checker_type_has_stored_variable(signature, program.adt_registry(), &mut UnordSet::new())
    });
    program
        .facts
        .type_polymorphic
        .borrow_mut()
        .insert(name.to_string(), polymorphic);
    polymorphic
}

fn top_level_fn_is_nested_rank_polymorphic(program: &HostLoweringSession<'_>, name: &str) -> bool {
    // chelis#940: a dimension may be hidden behind `Frame[n] -> Column[n] ->
    // tensor[n, _]`, so a signature-local `d-var` scan is insufficient.
    // Inspect checker-owned ADT parameter roles specifically: a direct
    // `tensor[n, _]` dimension remains a valid generic C surface, while an
    // ADT argument used only as a nested tensor dimension has no stored ABI
    // field and must be specialized at its call site.
    checked_authored_function_signature(program, name).is_some_and(|signature| {
        checker_type_has_erased_adt_variable(
            signature,
            program.adt_registry(),
            &mut UnordSet::new(),
        )
    })
}

fn checked_authored_function_signature<'a>(
    program: &'a HostLoweringSession<'a>,
    name: &str,
) -> Option<&'a Type> {
    if let Some(inference) = program.signature_inference().functions.get(name)
        && inference.authored_signature
    {
        return Some(
            inference
                .authored_signature_type
                .as_ref()
                .unwrap_or(&inference.checked_signature),
        );
    }
    let mut matches = program
        .signature_inference()
        .functions
        .iter()
        .filter(|(candidate, inference)| {
            inference.authored_signature && terminal_name_matches(candidate, name)
        })
        .map(|(_, inference)| {
            inference
                .authored_signature_type
                .as_ref()
                .unwrap_or(&inference.checked_signature)
        });
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn checker_type_has_stored_variable(
    ty: &Type,
    registry: &AdtRegistry,
    visiting: &mut UnordSet<(String, usize)>,
) -> bool {
    match ty {
        Type::Var(_) => true,
        Type::Tensor(_, TensorPrec::Var(_)) => true,
        Type::Tensor(_, TensorPrec::Concrete(_)) | Type::Prim(_) | Type::Unit => false,
        Type::Fn(params, ret) => {
            params
                .iter()
                .any(|ty| checker_type_has_stored_variable(ty, registry, visiting))
                || checker_type_has_stored_variable(ret, registry, visiting)
        }
        Type::Ref(inner) => checker_type_has_stored_variable(inner, registry, visiting),
        Type::Tuple(items) => items
            .iter()
            .any(|ty| checker_type_has_stored_variable(ty, registry, visiting)),
        Type::Adt(name, args) => {
            if let Some(definition) = registry.lookup(name) {
                args.iter().enumerate().any(|(index, arg)| {
                    checker_adt_parameter_is_stored(definition, index, registry, visiting)
                        && checker_type_has_stored_variable(arg, registry, visiting)
                })
            } else {
                args.iter()
                    .any(|ty| checker_type_has_stored_variable(ty, registry, visiting))
            }
        }
        Type::KindedAdt(name, args) => {
            if let Some(definition) = registry.lookup(name) {
                args.iter()
                    .enumerate()
                    .any(|(index, argument)| match argument {
                        NominalArg::Type(ty) => {
                            checker_adt_parameter_is_stored(definition, index, registry, visiting)
                                && checker_type_has_stored_variable(ty, registry, visiting)
                        }
                        NominalArg::Dimension(_) => false,
                    })
            } else {
                args.iter().any(|argument| {
                    argument
                        .as_type()
                        .is_some_and(|ty| checker_type_has_stored_variable(ty, registry, visiting))
                })
            }
        }
        Type::Error(_) => true,
    }
}

fn checker_type_has_variable(ty: &Type) -> bool {
    match ty {
        Type::Var(_) | Type::Tensor(_, TensorPrec::Var(_)) => true,
        Type::Tensor(dims, TensorPrec::Concrete(_)) => dims.iter().any(|dim| {
            matches!(
                dim,
                chelis_types::types::Dim::Var(_) | chelis_types::types::Dim::Rank(_)
            )
        }),
        Type::Fn(params, ret) => {
            params.iter().any(checker_type_has_variable) || checker_type_has_variable(ret)
        }
        Type::Ref(inner) => checker_type_has_variable(inner),
        Type::Adt(_, args) | Type::Tuple(args) => args.iter().any(checker_type_has_variable),
        Type::KindedAdt(_, args) => args.iter().any(|argument| match argument {
            NominalArg::Type(ty) => checker_type_has_variable(ty),
            NominalArg::Dimension(Dim::Var(_) | Dim::Rank(_)) => true,
            NominalArg::Dimension(_) => false,
        }),
        Type::Prim(_) | Type::Unit => false,
        Type::Error(_) => true,
    }
}

fn checker_type_has_erased_adt_variable(
    ty: &Type,
    registry: &AdtRegistry,
    visiting: &mut UnordSet<(String, usize)>,
) -> bool {
    match ty {
        Type::Fn(params, ret) => {
            params
                .iter()
                .any(|ty| checker_type_has_erased_adt_variable(ty, registry, visiting))
                || checker_type_has_erased_adt_variable(ret, registry, visiting)
        }
        Type::Ref(inner) => checker_type_has_erased_adt_variable(inner, registry, visiting),
        Type::Tuple(items) => items
            .iter()
            .any(|ty| checker_type_has_erased_adt_variable(ty, registry, visiting)),
        Type::Adt(name, args) => {
            let Some(definition) = registry.lookup(name) else {
                return false;
            };
            args.iter().enumerate().any(|(index, arg)| {
                (!checker_adt_parameter_is_stored(definition, index, registry, visiting)
                    && checker_type_has_variable(arg))
                    || checker_type_has_erased_adt_variable(arg, registry, visiting)
            })
        }
        Type::KindedAdt(name, args) => {
            let Some(definition) = registry.lookup(name) else {
                return false;
            };
            args.iter()
                .enumerate()
                .any(|(index, argument)| match argument {
                    NominalArg::Type(ty) => {
                        (!checker_adt_parameter_is_stored(definition, index, registry, visiting)
                            && checker_type_has_variable(ty))
                            || checker_type_has_erased_adt_variable(ty, registry, visiting)
                    }
                    NominalArg::Dimension(Dim::Var(_) | Dim::Rank(_)) => true,
                    NominalArg::Dimension(_) => false,
                })
        }
        Type::Var(_) | Type::Tensor(_, _) | Type::Prim(_) | Type::Unit | Type::Error(_) => false,
    }
}

fn lower_host_callback(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<Option<HostCallback>, crate::lower::LowerDiagnostic> {
    match expr {
        Expr::MetaExpr(meta, _) => lower_host_callback(&meta.expr, program, scope, tensor_helpers),
        Expr::List(list, _) if tag(list) == Some(DeepTag::Fn) => {
            let kids = children(list);
            let Some(params_list) = kids.first().and_then(as_list) else {
                return Ok(None);
            };
            if tag(params_list) != Some(DeepTag::Params) {
                return Ok(None);
            }
            let (param_tys, ret_ty) =
                expr_fn_type(expr).unwrap_or((Vec::new(), fresh_host_inference()));
            let mut callback_scope = scope.clone();
            let mut params = Vec::new();
            for (index, param) in children(params_list).iter().enumerate() {
                let Some(name) = param_name(param) else {
                    return Ok(None);
                };
                let ty = param_tys
                    .get(index)
                    .cloned()
                    .filter(|ty| !ty.is_unresolved())
                    .or_else(|| param_host_type(param))
                    .unwrap_or_else(fresh_host_inference);
                callback_scope.insert(name.clone(), ty.clone());
                params.push(HostParam { name, ty });
            }
            let Some(body_expr) = kids.get(1) else {
                return Ok(None);
            };
            let mut body = lower_host_expr(body_expr, program, &callback_scope, tensor_helpers)?;
            // Actualized callback types choose representation; the authored
            // parameter still supplies the obligation checked at invocation.
            let plan =
                SignatureEntryPlan::new(children(params_list).iter().zip(&params).filter_map(
                    |(declaration, param)| {
                        let declared = param_host_type(declaration)
                            .filter(|ty| matches!(ty, HostTypeTerm::Tensor(_)))
                            .unwrap_or_else(|| param.ty.clone());
                        let HostTypeTerm::Tensor(ty) = declared else {
                            return None;
                        };
                        Some(HostTensorInput {
                            name: param.name.clone(),
                            ty,
                        })
                    },
                ));
            if !plan.guards().is_empty() {
                let args = params
                    .iter()
                    .filter(|param| matches!(param.ty, HostTypeTerm::Tensor(_)))
                    .map(|param| {
                        HostExpr::new(HostExprKind::Var(param.name.clone(), param.ty.clone()))
                    })
                    .collect();
                let mut reserved = callback_scope
                    .to_sorted()
                    .into_iter()
                    .map(|(name, _)| name.clone())
                    .collect::<UnordSet<_>>();
                collect_deep_var_names(body_expr, &mut reserved);
                names_bound_in(body_expr, &mut reserved)
                    .map_err(|detail| host_expr_lowering_error(expr, detail))?;
                let mut serial = 0;
                let name = loop {
                    let candidate = format!("__chelis_callback_entry_{serial}");
                    if reserved.insert(candidate.clone()) {
                        break candidate;
                    }
                    serial += 1;
                };
                let ty = host_expr_type(&body);
                body = HostExpr::new(HostExprKind::Let {
                    bindings: vec![HostBinding {
                        name,
                        display_name: None,
                        display_roots: Vec::new(),
                        ty: HostTypeTerm::Unit,
                        value: HostExpr::new(HostExprKind::SignatureEntry { plan, args }),
                    }],
                    body: Box::new(body),
                    ty,
                });
            }
            let ret_ty = if ret_ty.is_unresolved() {
                host_expr_type(&body)
            } else {
                ret_ty
            };
            Ok(Some(HostCallback {
                kind: HostCallbackKind::Inline {
                    params,
                    body: Box::new(body),
                },
                ret_ty,
            }))
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Var) => {
            let Some(name) = children(list).first().and_then(symbol_name) else {
                return Ok(None);
            };
            let Some((param_tys, ret_ty)) = scope
                .get(name)
                .and_then(host_fn_signature)
                .or_else(|| {
                    lookup_declared_host_type(program, name).and_then(|ty| host_fn_signature(&ty))
                })
                .or_else(|| lookup_declared_fn_type(program, name))
                .or_else(|| {
                    lookup_program_def(&cached_program_defs(program), name).and_then(expr_fn_type)
                })
            else {
                return Ok(None);
            };
            let params = param_tys
                .into_iter()
                .enumerate()
                .map(|(index, ty)| HostParam {
                    name: format!("arg{index}"),
                    ty,
                })
                .collect();
            Ok(Some(HostCallback {
                kind: HostCallbackKind::Named {
                    function: name.to_string(),
                    params,
                },
                ret_ty,
            }))
        }
        _ => Ok(None),
    }
}

fn lower_list_literal_items(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
    tensor_helpers: &mut TensorHelperSink,
) -> Result<Option<Vec<HostExpr>>, crate::lower::LowerDiagnostic> {
    match expr {
        Expr::MetaExpr(meta, _) => {
            lower_list_literal_items(&meta.expr, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::Var) => {
            Ok((children(list).first().and_then(symbol_name) == Some("Nil")).then(Vec::new))
        }
        Expr::List(list, _) if tag(list) == Some(DeepTag::App) => {
            let kids = children(list);
            if kids.len() != 3 {
                return Ok(None);
            }
            let func_name = kids
                .first()
                .and_then(as_list)
                .and_then(|inner| (tag(inner) == Some(DeepTag::Var)).then_some(inner))
                .and_then(|inner| children(inner).first().and_then(symbol_name));
            if func_name != Some("Cons") {
                return Ok(None);
            }
            let head = lower_host_expr(&kids[1], program, scope, tensor_helpers)?;
            let Some(mut tail) =
                lower_list_literal_items(&kids[2], program, scope, tensor_helpers)?
            else {
                return Ok(None);
            };
            tail.insert(0, head);
            Ok(Some(tail))
        }
        _ => Ok(None),
    }
}

fn tensor_helper_args(
    inputs: &[HostTensorInput],
    scope: &UnordMap<String, HostTypeTerm>,
) -> Vec<HostExpr> {
    inputs
        .iter()
        .map(|input| {
            HostExpr::new(HostExprKind::Var(
                input.name.clone(),
                scope
                    .get(&input.name)
                    .cloned()
                    .unwrap_or_else(|| host_type_from_tensor_input(&input.ty)),
            ))
        })
        .collect()
}

fn tensor_helper_inputs(dag: &crate::Dag) -> Vec<HostTensorInput> {
    let mut seen = UnordSet::new();
    dag.nodes()
        .iter()
        .filter_map(|node| match &node.op {
            crate::RiscOp::Load { name } if seen.insert(name.as_str().to_string()) => {
                Some(HostTensorInput {
                    name: name.as_str().to_string(),
                    ty: node.output_type.clone(),
                })
            }
            _ => None,
        })
        .collect()
}

fn remap_tensor_helper_dim_symbols(
    dag: &crate::Dag,
    scope: &UnordMap<String, HostTypeTerm>,
    expected_output: &TensorType,
) -> crate::Dag {
    fn tensor_type_has_synthetic_dims(tensor_ty: &TensorType) -> bool {
        tensor_ty.dims.iter().any(|dim| {
            matches!(dim, crate::dag::DimInfo::Named(name, None) if {
                let mut chars = name.chars();
                matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
            })
        })
    }

    let mut expected_output = expected_output.clone();
    if let Some(mut returned) = dag.roots().first().copied() {
        loop {
            let node = dag.get(returned).expect("helper root belongs to DAG");
            for dependency in &node.shape_deps {
                if let Some(crate::dag::DagNode {
                    op:
                        crate::dag::RiscOp::ExtentWitness {
                            site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
                            axis: crate::dag::RtAxis::Lit(axis),
                            ..
                        },
                    ..
                }) = dag.get(*dependency)
                    && let Some(dim) = expected_output.dims.get_mut(*axis as usize)
                    && matches!(dim, crate::dag::DimInfo::Lit(_))
                {
                    *dim = crate::dag::DimInfo::Named("*".into(), None);
                }
            }
            if !matches!(
                node.op,
                crate::dag::RiscOp::Copy
                    | crate::dag::RiscOp::Cast { .. }
                    | crate::dag::RiscOp::CastTrunc { .. }
            ) {
                break;
            }
            let Some(input) = node.inputs.first() else {
                break;
            };
            returned = *input;
        }
    }
    let expected_output = &expected_output;
    let formal_inputs = tensor_helper_inputs(dag);
    let mut actual_inputs = formal_inputs
        .iter()
        .map(|input| match scope.get(&input.name) {
            Some(HostTypeTerm::Tensor(actual)) => actual.clone(),
            _ => input.ty.clone(),
        })
        .collect::<Vec<_>>();
    let mut formal_params = formal_inputs
        .iter()
        .map(|input| input.ty.clone())
        .collect::<Vec<_>>();
    if let Some(root) = dag.roots().first().and_then(|id| dag.get(*id)) {
        formal_params.push(root.output_type.clone());
        let actual_output = if tensor_type_has_synthetic_dims(expected_output) {
            match root.op {
                crate::dag::RiscOp::Permute { ref axes } => root
                    .inputs
                    .first()
                    .and_then(|id| dag.get(*id))
                    .map(|node| {
                        let mut output = node.output_type.clone();
                        output.dims = axes
                            .iter()
                            .filter_map(|axis| node.output_type.dims.get(*axis).cloned())
                            .collect();
                        output
                    })
                    .unwrap_or_else(|| expected_output.clone()),
                crate::dag::RiscOp::UniformLike { .. } | crate::dag::RiscOp::Dropout { .. } => root
                    .inputs
                    .first()
                    .and_then(|id| dag.get(*id))
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| expected_output.clone()),
                _ => expected_output.clone(),
            }
        } else {
            expected_output.clone()
        };
        actual_inputs.push(actual_output);
    }
    let mut remapped = crate::lower::remap_tensor_dim_symbols(dag, &formal_params, &actual_inputs);
    // chelis#632 (needed by the chelis#631 oracle): anon wildcards are no
    // longer substitution keys in `tensor_dim_substitutions`, so the
    // declared return no longer paints its dims across every
    // wildcard-typed node (distinct runtime extents conflated under one
    // symbol → runtime-dim guard aborts on well-formed programs). The
    // declared return still owns the ROOT's shape: retype the root
    // POSITIONALLY, anon axis by anon axis — but ONLY on axes the root
    // op itself can declare at run time (`dag::op_declarable_axes`). A
    // symbol painted anywhere else has no declaring Load or op and resolves
    // to no extent origin at emission; those axes stay anon and size
    // themselves per node.
    if let (Some(root_id), Some(actual_output)) =
        (dag.roots().first().copied(), actual_inputs.last())
        && let Some(root) = remapped.get(root_id)
        && root.output_type.dims.len() == actual_output.dims.len()
    {
        let is_anon = |dim: &crate::dag::DimInfo| matches!(dim, crate::dag::DimInfo::Named(name, None) if name.is_empty() || name == "*");
        let declarable = crate::dag::op_declarable_axes(&remapped, root);
        let mut output = root.output_type.clone();
        let mut changed = false;
        for (axis, (dim, actual_dim)) in output
            .dims
            .iter_mut()
            .zip(actual_output.dims.iter())
            .enumerate()
        {
            if declarable.contains(&axis) && is_anon(dim) && !is_anon(actual_dim) {
                *dim = actual_dim.clone();
                changed = true;
            }
        }
        if changed {
            let op = root.op.clone();
            let inputs = root.inputs.clone();
            remapped.replace_node(root_id, op, inputs, output);
        }
    }
    actualize_tensor_helper_types(&remapped, scope)
}

fn actualize_tensor_helper_types(
    dag: &crate::Dag,
    scope: &UnordMap<String, HostTypeTerm>,
) -> crate::Dag {
    fn inferred_load_type(
        name: &str,
        scope: &UnordMap<String, HostTypeTerm>,
        fallback: &TensorType,
    ) -> TensorType {
        match scope.get(name) {
            Some(HostTypeTerm::Tensor(actual)) => actual.clone(),
            _ => fallback.clone(),
        }
    }

    fn precision_like(input: &TensorType, precision: chelis_types::types::Prim) -> TensorType {
        TensorType {
            dims: input.dims.clone(),
            precision,
        }
    }

    fn synthetic_dims(tensor_ty: &TensorType) -> bool {
        tensor_ty.dims.iter().any(|dim| {
            matches!(dim, crate::dag::DimInfo::Named(name, None) if {
                let mut chars = name.chars();
                matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
            })
        })
    }

    fn synthetic_dim(dim: &crate::dag::DimInfo) -> bool {
        matches!(dim, crate::dag::DimInfo::Named(name, None) if {
            let mut chars = name.chars();
            matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
        })
    }

    fn merge_dim(lhs: &crate::dag::DimInfo, rhs: &crate::dag::DimInfo) -> crate::dag::DimInfo {
        match (synthetic_dim(lhs), synthetic_dim(rhs)) {
            (true, false) => rhs.clone(),
            _ => lhs.clone(),
        }
    }

    fn merge_binary_tensor_types(
        lhs: &TensorType,
        rhs: &TensorType,
        precision: chelis_types::types::Prim,
    ) -> TensorType {
        if lhs.dims.len() != rhs.dims.len() {
            return precision_like(lhs, precision);
        }
        TensorType {
            dims: lhs
                .dims
                .iter()
                .zip(rhs.dims.iter())
                .map(|(lhs, rhs)| merge_dim(lhs, rhs))
                .collect(),
            precision,
        }
    }

    fn reserve_runtime_dim_name(
        occupied: &mut UnordSet<String>,
        operation: &str,
        node: crate::dag::NodeId,
        axis: usize,
    ) -> String {
        let base = format!("_rt_{operation}_dim_{}_{}", node.0, axis);
        let mut candidate = base.clone();
        let mut suffix = 0usize;
        while !occupied.insert(candidate.clone()) {
            suffix += 1;
            candidate = format!("{base}_{suffix}");
        }
        candidate
    }

    fn known_extent(dim: &crate::dag::DimInfo) -> Option<usize> {
        match dim {
            crate::dag::DimInfo::Lit(value) | crate::dag::DimInfo::Named(_, Some(value)) => {
                Some(*value)
            }
            crate::dag::DimInfo::Named(_, None) => None,
        }
    }

    fn rank_preserving_movement_type(
        op: &crate::dag::RiscOp,
        node_id: crate::dag::NodeId,
        input: &TensorType,
        fallback: &TensorType,
        occupied_dim_names: &mut UnordSet<String>,
    ) -> Option<TensorType> {
        // Pad, Shrink, and Stride all preserve input rank, but each axis can
        // either forward the input extent, compute a static extent, or mint
        // an op-declared runtime extent. Keep that complete shape calculus in
        // one function: a downstream movement consumer must never fall back
        // to checker metadata whose rank predates helper actualization.
        fn unresolved_axis(
            operation: &str,
            node_id: crate::dag::NodeId,
            axis: usize,
            input_rank: usize,
            fallback: &TensorType,
            occupied_dim_names: &mut UnordSet<String>,
        ) -> crate::dag::DimInfo {
            if fallback.dims.len() == input_rank
                && let Some(dim) = fallback.dims.get(axis)
                && !matches!(dim, crate::dag::DimInfo::Named(name, None) if name.is_empty() || name == "*")
            {
                return dim.clone();
            }
            crate::dag::DimInfo::Named(
                reserve_runtime_dim_name(occupied_dim_names, operation, node_id, axis),
                None,
            )
        }

        let dims = match op {
            crate::dag::RiscOp::Pad { padding, .. } => {
                if padding.len() != input.dims.len() {
                    return None;
                }
                padding
                    .iter()
                    .zip(input.dims.iter())
                    .enumerate()
                    .map(
                        |(axis, ((before, after), input_dim))| match (before, after) {
                            (crate::dag::RtDim::Lit(0), crate::dag::RtDim::Lit(0)) => {
                                Some(input_dim.clone())
                            }
                            (crate::dag::RtDim::Lit(before), crate::dag::RtDim::Lit(after)) => {
                                known_extent(input_dim)
                                    .and_then(|extent| extent.checked_add(*before))
                                    .and_then(|extent| extent.checked_add(*after))
                                    .map(crate::dag::DimInfo::Lit)
                                    .or_else(|| {
                                        Some(unresolved_axis(
                                            "pad",
                                            node_id,
                                            axis,
                                            input.dims.len(),
                                            fallback,
                                            occupied_dim_names,
                                        ))
                                    })
                            }
                            // Any RUNTIME bound. chelis#1397's rule, which
                            // the `Shrink` arm below states and this arm
                            // did not apply: minting a fresh symbol over an
                            // axis whose DECLARED extent is a claim discards
                            // the claim rather than discharging it, so the
                            // admission is ASKED rather than assumed.
                            // `op_computed_axis_extent` is the one answer
                            // `local_dim_guard_sites` uses to decide whether
                            // this axis gets a guard site, and
                            // `unresolved_axis` keeps the declared dim from
                            // the fallback so that site can form.
                            //
                            // Until the `pad` admission landed, the coupling
                            // was structural here too: no runtime-bound pad
                            // axis was admitted, so minting could not discard
                            // a claim. Admitting the owner without this made
                            // the claim silent in the exported and
                            // value-binding forms while the inlined root, whose
                            // root type is not actualized through this
                            // function, still trapped. Measured at
                            // `779c46626`: `-> tensor[2, f32] =
                            // pad(x, [[shape(y, 0i32), 1i64]], 0.0f32)` over
                            // three elements printed `shape=[6]` at exit zero
                            // on both lanes as a value binding and trapped as
                            // an inlined root.
                            _ => {
                                if crate::axis_sources::op_computed_axis_extent(op, axis).is_some()
                                {
                                    Some(unresolved_axis(
                                        "pad",
                                        node_id,
                                        axis,
                                        input.dims.len(),
                                        fallback,
                                        occupied_dim_names,
                                    ))
                                } else {
                                    Some(crate::dag::DimInfo::Named(
                                        reserve_runtime_dim_name(
                                            occupied_dim_names,
                                            "pad",
                                            node_id,
                                            axis,
                                        ),
                                        None,
                                    ))
                                }
                            }
                        },
                    )
                    .collect::<Option<Vec<_>>>()?
            }
            crate::dag::RiscOp::Shrink { bounds } => {
                if bounds.len() != input.dims.len() {
                    return None;
                }
                bounds
                    .iter()
                    .zip(input.dims.iter())
                    .enumerate()
                    // No wildcard, deliberately. A `None` from any axis
                    // collapses the whole result and sends the caller back
                    // to `node.output_type`, which is the stale checker
                    // metadata chelis#1137 exists to stop trusting. A future
                    // `RtDim` variant absorbed by a `_` arm would restore
                    // that defect silently, with no conflict marker and no
                    // failing test; with every pair named, it is an `E0004`
                    // non-exhaustive-match build error instead. `Stride`
                    // below is total for the same reason.
                    .map(|(axis, ((start, end), input_dim))| match (start, end) {
                        // Both bounds static: the extent is their difference.
                        // A `start > end` range is invalid and `verify`'s
                        // "has invalid bounds" check rejects it; `checked_sub`
                        // declines deliberately instead of wrapping.
                        (crate::dag::RtDim::Lit(start), crate::dag::RtDim::Lit(end)) => {
                            end.checked_sub(*start).map(crate::dag::DimInfo::Lit)
                        }
                        // The `SHRINK_TO_END` full-axis sentinel: the slice is
                        // the whole axis, so it keeps the input axis identity.
                        (crate::dag::RtDim::Lit(0), crate::dag::RtDim::ToEnd) => {
                            Some(input_dim.clone())
                        }
                        // A sentinel under a nonzero or runtime start is
                        // malformed, and since chelis#1480 every stage that
                        // validates a bound rejects it by name: `verify`'s
                        // Shrink arm, the WireDag decoder's Shrink arm, and
                        // `bind_symbolic_dims`, which errors with
                        // "shrink-to-end sentinel requires a literal zero
                        // start". `grad`'s Shrink adjoint also panics with
                        // "malformed ToEnd sentinel in shrink adjoint" at the
                        // producer. `grad`'s committed
                        // `shrink_adjoint_malformed_sentinel_fails_loud` is a
                        // `#[should_panic]` control over exactly this shape.
                        // Nothing builds a malformed one either. The list
                        // below was established by searching every
                        // `RtDim::ToEnd` construction across `chelis-ir`,
                        // `chelis-compiler-api` and the three backends,
                        // separating production from test by each file's
                        // `cfg(test)` line, rather than by recall; an
                        // enumeration this argument rests on has to be
                        // complete, because the next person to add a
                        // constructor will check it instead of re-deriving it.
                        // `lower::lower_one_bound` yields only `Lit`/`Node`,
                        // so no Surf or Deep `shrink` can spell `ToEnd` at
                        // all. The production constructors are the Pad,
                        // ProdReduce and Stride adjoints in `grad`, each
                        // pairing it with `Lit(0)`, and `vmap`'s prepended
                        // batch axis (`batch_shrink_end` in `vmap.rs`), which
                        // is a sentinel only while the mapped batch stays
                        // symbolic and is paired with `Lit(0)` at its
                        // construction site. `vmap` rewrites every node's
                        // axes, and `shift_input_axis` clones every
                        // non-`InputAxis` variant unchanged, so an incoming
                        // pair keeps its pairing through the shift. Declining
                        // here is consistency with a rule enforced elsewhere,
                        // not a gap being papered over.
                        (crate::dag::RtDim::Lit(_), crate::dag::RtDim::ToEnd)
                        | (crate::dag::RtDim::Node(_), crate::dag::RtDim::ToEnd) => None,
                        // At least one bound is only known at run time, so the
                        // extent is an op-declared runtime dimension.
                        (crate::dag::RtDim::Node(_), crate::dag::RtDim::Node(_))
                        | (crate::dag::RtDim::Node(_), crate::dag::RtDim::Lit(_))
                        | (crate::dag::RtDim::Lit(_), crate::dag::RtDim::Node(_)) => {
                            // chelis#1397: a DECLARED extent on this axis is
                            // a claim, and minting a fresh symbol over it
                            // discards the claim rather than discharging it.
                            // `unresolved_axis` mints only when the node
                            // carries no usable dimension of its own, which
                            // is the same rule the `pad` arm above applies.
                            // That sentence described only pad's `(Lit, Lit)`
                            // case until chelis#1911's round 1; pad's runtime
                            // case minted unconditionally and discarded the
                            // claim.
                            //
                            // Keeping the claim is only safe when something
                            // checks it, so the admission is ASKED rather than
                            // assumed: `op_computed_axis_extent` is the one
                            // answer `local_dim_guard_sites` uses to decide
                            // whether this axis gets a guard site. Until this
                            // call existed the coupling was structural - only
                            // `shrink` reaches this arm, and `shrink` is the
                            // one admitted owner - so removing `shrink` from
                            // admission would have kept the claim with nothing
                            // to enforce it, silently.
                            if crate::axis_sources::op_computed_axis_extent(op, axis).is_some() {
                                Some(unresolved_axis(
                                    "shrink",
                                    node_id,
                                    axis,
                                    input.dims.len(),
                                    fallback,
                                    occupied_dim_names,
                                ))
                            } else {
                                Some(crate::dag::DimInfo::Named(
                                    reserve_runtime_dim_name(
                                        occupied_dim_names,
                                        "shrink",
                                        node_id,
                                        axis,
                                    ),
                                    None,
                                ))
                            }
                        }
                        // `ToEnd` is an `end`-only marker; `Sym` is a
                        // `Reshape` target only; and `InputAxis` belongs only
                        // to `Expand`/`Reshape`. `verify` rejects all three
                        // misplaced carriers in a `Shrink` bound.
                        (crate::dag::RtDim::ToEnd, _)
                        | (crate::dag::RtDim::Sym(_), _)
                        | (crate::dag::RtDim::InputAxis { .. }, _)
                        | (_, crate::dag::RtDim::Sym(_))
                        | (_, crate::dag::RtDim::InputAxis { .. }) => None,
                    })
                    .collect::<Option<Vec<_>>>()?
            }
            crate::dag::RiscOp::Stride { strides } => {
                if strides.len() != input.dims.len() {
                    return None;
                }
                strides
                    .iter()
                    .zip(input.dims.iter())
                    .enumerate()
                    .map(|(axis, (stride, input_dim))| match stride {
                        crate::dag::RtDim::Lit(0) => None,
                        crate::dag::RtDim::Lit(1) => Some(input_dim.clone()),
                        crate::dag::RtDim::Lit(stride) => known_extent(input_dim)
                            .map(|extent| crate::dag::DimInfo::Lit(extent.div_ceil(*stride)))
                            .or_else(|| {
                                Some(unresolved_axis(
                                    "stride",
                                    node_id,
                                    axis,
                                    input.dims.len(),
                                    fallback,
                                    occupied_dim_names,
                                ))
                            }),
                        crate::dag::RtDim::Node(_) => {
                            // A runtime step computes a fresh extent, but a
                            // declared fallback dimension is a claim that the
                            // local `StrideSpan` guard must retain and check.
                            // Mint only when there is no usable declaration,
                            // exactly as the admitted pad and shrink owners do
                            // above.
                            if crate::axis_sources::op_computed_axis_extent(op, axis).is_some() {
                                Some(unresolved_axis(
                                    "stride",
                                    node_id,
                                    axis,
                                    input.dims.len(),
                                    fallback,
                                    occupied_dim_names,
                                ))
                            } else {
                                Some(crate::dag::DimInfo::Named(
                                    reserve_runtime_dim_name(
                                        occupied_dim_names,
                                        "stride",
                                        node_id,
                                        axis,
                                    ),
                                    None,
                                ))
                            }
                        }
                        crate::dag::RtDim::ToEnd
                        | crate::dag::RtDim::Sym(_)
                        | crate::dag::RtDim::InputAxis { .. } => None,
                    })
                    .collect::<Option<Vec<_>>>()?
            }
            _ => return None,
        };
        Some(TensorType {
            dims,
            precision: fallback.precision,
        })
    }

    let mut inferred = UnordMap::<crate::dag::NodeId, TensorType>::new();
    let mut authoritative_shape_nodes = UnordSet::<crate::dag::NodeId>::new();
    // Generated runtime extents share the `DimInfo::Named` carrier with
    // source dimensions. Reserve the canonical complete DAG namespace plus
    // names that host-scope actualization can introduce later, then mint by
    // insertion into that finite set. A prefix convention alone is not an
    // identity boundary (chelis#1137 red-team rounds 2-3).
    let mut occupied_dim_names = crate::dag::dimension_identity_names(dag);
    // `to_sorted` here is an order-insensitive drain, not an order claim:
    // every element lands in a set whose membership is what the minting
    // loop below reads.
    occupied_dim_names.extend(
        scope
            .to_sorted()
            .into_iter()
            .filter_map(|(_, term)| match term {
                HostTypeTerm::Tensor(tensor) => Some(tensor),
                _ => None,
            })
            .flat_map(|tensor| tensor.dims.iter())
            .filter_map(|dim| match dim {
                crate::dag::DimInfo::Named(name, _) => Some(name.clone()),
                crate::dag::DimInfo::Lit(_) => None,
            }),
    );
    let mut uses = UnordMap::<crate::dag::NodeId, Vec<crate::dag::NodeId>>::new();
    for node in dag.nodes() {
        for input in &node.inputs {
            uses.entry(*input).or_default().push(node.id);
        }
    }
    for node in dag.nodes() {
        let actual = match &node.op {
            crate::dag::RiscOp::Load { name } => {
                Some(inferred_load_type(name.as_str(), scope, &node.output_type))
            }
            crate::dag::RiscOp::Add
            | crate::dag::RiscOp::Sub
            | crate::dag::RiscOp::Mul
            | crate::dag::RiscOp::Div
            | crate::dag::RiscOp::FloorDiv
            | crate::dag::RiscOp::TruncDiv
            | crate::dag::RiscOp::MaxElem
            | crate::dag::RiscOp::MinElem => node
                .inputs
                .first()
                .and_then(|lhs| inferred.get(lhs))
                .map(|lhs| {
                    node.inputs
                        .get(1)
                        .and_then(|rhs| inferred.get(rhs))
                        .map(|rhs| merge_binary_tensor_types(lhs, rhs, node.output_type.precision))
                        .unwrap_or_else(|| precision_like(lhs, node.output_type.precision))
                }),
            crate::dag::RiscOp::ExtremaAdjoint { .. } => node
                .inputs
                .first()
                .and_then(|lhs| inferred.get(lhs))
                .map(|lhs| {
                    let forward = node
                        .inputs
                        .get(1)
                        .and_then(|rhs| inferred.get(rhs))
                        .map(|rhs| merge_binary_tensor_types(lhs, rhs, node.output_type.precision))
                        .unwrap_or_else(|| precision_like(lhs, node.output_type.precision));
                    node.inputs
                        .get(2)
                        .and_then(|gradient| inferred.get(gradient))
                        .map(|gradient| {
                            merge_binary_tensor_types(
                                &forward,
                                gradient,
                                node.output_type.precision,
                            )
                        })
                        .unwrap_or(forward)
                }),
            crate::dag::RiscOp::ReluAdjoint => node
                .inputs
                .first()
                .and_then(|input| inferred.get(input))
                .map(|input| {
                    node.inputs
                        .get(1)
                        .and_then(|gradient| inferred.get(gradient))
                        .map(|gradient| {
                            merge_binary_tensor_types(input, gradient, node.output_type.precision)
                        })
                        .unwrap_or_else(|| precision_like(input, node.output_type.precision))
                }),
            crate::dag::RiscOp::Neg
            | crate::dag::RiscOp::Relu
            | crate::dag::RiscOp::Exp
            | crate::dag::RiscOp::Log
            | crate::dag::RiscOp::Sin
            | crate::dag::RiscOp::Sqrt
            | crate::dag::RiscOp::Cos
            | crate::dag::RiscOp::Tan
            | crate::dag::RiscOp::Atan
            | crate::dag::RiscOp::Abs
            | crate::dag::RiscOp::Floor
            | crate::dag::RiscOp::Ceil
            | crate::dag::RiscOp::Round
            | crate::dag::RiscOp::Recip
            | crate::dag::RiscOp::UniformLike { .. }
            | crate::dag::RiscOp::Dropout { .. }
            | crate::dag::RiscOp::Copy
            | crate::dag::RiscOp::Drop
            | crate::dag::RiscOp::Realize
            | crate::dag::RiscOp::Cast { .. }
            | crate::dag::RiscOp::CastTrunc { .. }
            | crate::dag::RiscOp::FusedElem { .. } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| precision_like(input, node.output_type.precision)),
            crate::dag::RiscOp::Sum { axis, .. }
            | crate::dag::RiscOp::MaxReduce { axis }
            | crate::dag::RiscOp::MinReduce { axis }
            | crate::dag::RiscOp::ProdReduce { axis }
            | crate::dag::RiscOp::Argmax { axis }
            | crate::dag::RiscOp::Argmin { axis } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| TensorType {
                    dims: input
                        .dims
                        .iter()
                        .enumerate()
                        .filter_map(|(index, dim)| (index != *axis).then_some(dim.clone()))
                        .collect(),
                    precision: node.output_type.precision,
                }),
            crate::dag::RiscOp::Count { axes } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| TensorType {
                    dims: input
                        .dims
                        .iter()
                        .enumerate()
                        .filter_map(|(index, dim)| (!axes.contains(&index)).then_some(dim.clone()))
                        .collect(),
                    precision: Prim::Int64,
                }),
            crate::dag::RiscOp::Permute { axes } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| TensorType {
                    dims: axes
                        .iter()
                        .filter_map(|axis| input.dims.get(*axis).cloned())
                        .collect(),
                    precision: node.output_type.precision,
                }),
            crate::dag::RiscOp::Expand { axis, size } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .and_then(|input| {
                    // `RiscOp::Expand` carries both movement forms, told apart
                    // by the node's output rank against its operand's, exactly
                    // as `verify.rs`, `eval.rs` and the C emitter tell them
                    // apart. Inserting unconditionally would give a same-rank
                    // node a rank+1 helper type, which is a silently wrong
                    // signature rather than a declined one.
                    let mut dims = input.dims.clone();
                    let inserts = node.output_type.dims.len() == dims.len() + 1;
                    if (inserts && *axis > dims.len()) || (!inserts && *axis >= dims.len()) {
                        return None;
                    }
                    let extent = match size {
                        crate::dag::RtDim::Lit(value) => crate::dag::DimInfo::Lit(*value),
                        crate::dag::RtDim::InputAxis {
                            tensor,
                            axis: crate::dag::RtAxis::Lit(source_axis),
                        } => {
                            let source_id = *node.inputs.get(*tensor)?;
                            let source = inferred.get(&source_id)?;
                            let source_axis = usize::try_from(*source_axis).ok()?;
                            source.dims.get(source_axis)?.clone()
                        }
                        crate::dag::RtDim::Node(_) | crate::dag::RtDim::Sym(_) => {
                            node.output_type.dims.get(*axis)?.clone()
                        }
                        crate::dag::RtDim::ToEnd => return None,
                    };
                    if inserts {
                        dims.insert(*axis, extent);
                    } else {
                        dims[*axis] = extent;
                    }
                    Some(TensorType {
                        dims,
                        precision: node.output_type.precision,
                    })
                }),
            crate::dag::RiscOp::Pad { .. }
            | crate::dag::RiscOp::Shrink { .. }
            | crate::dag::RiscOp::Stride { .. } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .and_then(|input| {
                    let actual = rank_preserving_movement_type(
                        &node.op,
                        node.id,
                        input,
                        &node.output_type,
                        &mut occupied_dim_names,
                    );
                    if actual.is_some() {
                        authoritative_shape_nodes.insert(node.id);
                    }
                    actual
                }),
            // A reusable input is a storage hint, so it may supply a shape
            // only when this pass has no operation-specific rule. Keep this
            // fallback inside the unhandled arm: `None` from any explicit
            // rule is a deliberate decline for malformed or unavailable
            // shape evidence and must preserve `node.output_type`.
            _ => node
                .reusable_input
                .and_then(|id| inferred.get(&id))
                .filter(|input| input.dims.len() == node.output_type.dims.len())
                .map(|input| precision_like(input, node.output_type.precision)),
        };
        if let Some(actual) = actual {
            inferred.insert(node.id, actual);
        }
    }

    loop {
        let mut changed = false;
        for node in dag.nodes() {
            if !matches!(node.op, crate::dag::RiscOp::Expand { .. })
                || !synthetic_dims(&node.output_type)
            {
                continue;
            }
            let Some(actual) = uses.get(&node.id).and_then(|user_ids| {
                user_ids
                    .iter()
                    .filter_map(|user_id| inferred.get(user_id))
                    .find(|user_ty| user_ty.dims.len() == node.output_type.dims.len())
                    .cloned()
            }) else {
                continue;
            };
            let entry = inferred
                .entry(node.id)
                .or_insert_with(|| node.output_type.clone());
            if entry.dims != actual.dims {
                *entry = TensorType {
                    dims: actual.dims,
                    precision: node.output_type.precision,
                };
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut actualized = dag.clone();
    let node_ids = actualized
        .nodes()
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    let mut synthetic_renames = UnordMap::<String, crate::dag::DimInfo>::new();
    for id in node_ids {
        let Some(node) = actualized.get(id).cloned() else {
            continue;
        };
        let Some(actual) = inferred.get(&id) else {
            continue;
        };
        let operation_shape_is_authoritative = authoritative_shape_nodes.contains(&node.id);
        if (!synthetic_dims(&node.output_type) && !operation_shape_is_authoritative)
            || (node.output_type.dims.len() != actual.dims.len()
                && !operation_shape_is_authoritative)
        {
            continue;
        }
        // Record which minted `dN` alias each output axis resolved to,
        // so op-internal references to the same alias can be renamed in
        // lockstep below.
        for (old_dim, new_dim) in node.output_type.dims.iter().zip(actual.dims.iter()) {
            if let crate::dag::DimInfo::Named(name, None) = old_dim
                && synthetic_dim(old_dim)
                && old_dim != new_dim
            {
                match synthetic_renames.entry(name.clone()) {
                    chelis_unord::Entry::Vacant(slot) => {
                        slot.insert(new_dim.clone());
                    }
                    chelis_unord::Entry::Occupied(existing) => {
                        // A single checker dim-var has a single extent in
                        // a well-typed program; a conflicting re-bind
                        // means the helper DAG was already inconsistent.
                        // Fail loudly in debug rather than renaming op
                        // fields with the wrong extent (review #363 N1).
                        debug_assert_eq!(
                            existing.get(),
                            new_dim,
                            "synthetic dim `{name}` resolved to conflicting actuals"
                        );
                    }
                }
            }
        }
        actualized.replace_node(id, node.op, node.inputs, actual.clone());
        if let Some(reusable_input) = node.reusable_input {
            actualized.set_reusable_input(id, reusable_input);
        }
    }
    if synthetic_renames.is_empty() {
        return actualized;
    }
    // chelis#345 (op-internal half): `replace_node` above rewrites
    // OUTPUT types only, leaving op-internal fields (`Expand::size`,
    // `Reshape::new_shape`, `BlasMatmul` dims) holding the stale minted
    // names — the mixed state (`type: [Named("n")]` next to
    // `size: Sym("d47")`) the declaration derivation rejects because
    // nothing declares the alias. Apply the
    // collected renames to every dim reference so the helper DAG stays
    // internally consistent. Non-synthetic (user-facing) names are
    // never in the map and pass through untouched.
    crate::lower::apply_dim_substitutions(&actualized, &synthetic_renames)
}

/// chelis#631/#662: does this Deep expr — or any def it transitively
/// references — contain a forward `fail` application outside a
/// `grad`/`vmap`/`vmap-grad` subtree? Conservative: any `(var fail)`
/// reference counts, and a referenced def is walked once (the `visiting`
/// set both breaks recursion cycles and memoizes). A transformed subtree is
/// opaque here because transformed-subtree behavior is outside chelis#662;
/// chelis#1464 separately owns preservation of an internal taken `fail`.
/// Only forward siblings are classified by this traversal.
fn expr_reaches_forward_fail(
    expr: &Expr,
    defs: &BTreeMap<String, Expr>,
    visiting: &mut UnordSet<String>,
) -> bool {
    record_host_work(|profile| profile.fail_scan_nodes += 1);
    match expr {
        Expr::List(list, _) => {
            if matches!(tag(list), Some(DeepTag::Grad | DeepTag::Vmap))
                || list.unknown_tag_symbol() == Some("vmap-grad")
            {
                record_host_work(|profile| profile.grad_scan_nodes += 1);
                return false;
            }
            if tag(list) == Some(DeepTag::Var)
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                if name == "fail" {
                    return true;
                }
                if let Some(body) = defs.get(name)
                    && visiting.insert(name.to_string())
                {
                    return expr_reaches_forward_fail(body, defs, visiting);
                }
                return false;
            }
            list.elements
                .iter()
                .any(|kid| expr_reaches_forward_fail(kid, defs, visiting))
        }
        Expr::MetaExpr(meta, _) => expr_reaches_forward_fail(&meta.expr, defs, visiting),
        _ => false,
    }
}

/// Does this Deep expr contain a `grad`/`vmap`/`vmap-grad` node anywhere?
/// Entry-point routing uses the whole-expression answer because transformed
/// result packaging is host-owned; the forward-fail gate above intentionally
/// uses the more precise subtree-aware traversal instead.
fn expr_contains_grad_like(expr: &Expr) -> bool {
    record_host_work(|profile| profile.grad_scan_nodes += 1);
    match expr {
        Expr::List(list, _) => {
            matches!(tag(list), Some(DeepTag::Grad | DeepTag::Vmap))
                || list.unknown_tag_symbol() == Some("vmap-grad")
                || list.elements.iter().any(expr_contains_grad_like)
        }
        Expr::MetaExpr(meta, _) => expr_contains_grad_like(&meta.expr),
        _ => false,
    }
}

fn expr_contains_vmap_grad(expr: &Expr) -> bool {
    match expr {
        Expr::List(list, _) => {
            list.unknown_tag_symbol() == Some("vmap-grad")
                || (tag(list) == Some(DeepTag::Vmap)
                    && children(list)
                        .first()
                        .is_some_and(|target| target.tag() == Some(DeepTag::Grad)))
                || list.elements.iter().any(expr_contains_vmap_grad)
        }
        Expr::Node(node, span) => expr_contains_vmap_grad(&Expr::List(node.to_list(*span), *span)),
        Expr::MetaExpr(meta, _) => expr_contains_vmap_grad(&meta.expr),
        Expr::BareList(items, _) => items.iter().any(expr_contains_vmap_grad),
        Expr::UnknownForm(data) => {
            data.head == "vmap-grad" || data.children.iter().any(expr_contains_vmap_grad)
        }
        Expr::Atom(_, _) | Expr::Map(_, _) => false,
    }
}

fn cached_program_defs(program: &HostLoweringSession<'_>) -> Arc<BTreeMap<String, Expr>> {
    if let Some(cached) = program.facts.program_defs.borrow().clone() {
        return cached;
    }
    let defs = Arc::new(collect_program_defs(program.exprs()));
    *program.facts.program_defs.borrow_mut() = Some(defs.clone());
    defs
}

fn cached_subexpr_lowering_context(
    program: &HostLoweringSession<'_>,
) -> crate::lower::SubexprLoweringContext {
    if let Some(cached) = program.facts.subexpr_lowering_context.borrow().clone() {
        return cached;
    }
    record_host_work(|profile| {
        profile.type_env_clone_nodes += program
            .type_env()
            .values()
            .map(deep_expr_nodes)
            .sum::<usize>();
    });
    let context = crate::lower::prepare_checked_subexpr_lowering_context(
        program,
        cached_program_defs(program),
        Arc::new(crate::lower::collect_top_level_sigs(program.exprs())),
    );
    *program.facts.subexpr_lowering_context.borrow_mut() = Some(context.clone());
    context
}

fn cached_dynamic_to_tensor_def_summaries(
    program: &HostLoweringSession<'_>,
) -> Arc<BTreeMap<String, bool>> {
    if let Some(cached) = program
        .facts
        .dynamic_to_tensor_def_summaries
        .borrow()
        .clone()
    {
        return cached;
    }

    let defs = cached_program_defs(program);
    let def_names = defs.keys().cloned().collect::<BTreeSet<_>>();
    let mut summaries = defs
        .keys()
        .map(|name| (name.clone(), false))
        .collect::<BTreeMap<_, _>>();
    let mut reverse_edges = UnordMap::<String, Vec<String>>::new();
    let mut queue = VecDeque::new();

    for (name, body) in defs.iter() {
        let mut referenced_defs = BTreeSet::new();
        let mut callable_scope = CallableScope::default();
        let directly_dynamic = collect_dynamic_to_tensor_def_refs(
            body,
            &def_names,
            &mut callable_scope,
            &mut referenced_defs,
        );
        if directly_dynamic {
            summaries.insert(name.clone(), true);
            queue.push_back(name.clone());
        }
        for referenced in referenced_defs {
            reverse_edges
                .entry(referenced)
                .or_default()
                .push(name.clone());
        }
    }

    // Propagate the property backwards through the call graph. A worklist is
    // both cycle-safe and linear in definitions plus reference edges; a DFS
    // memo can incorrectly seal one member of a cycle before another member's
    // direct runtime-shaped `to_tensor` is discovered.
    while let Some(dynamic_name) = queue.pop_front() {
        for caller in reverse_edges.get(&dynamic_name).into_iter().flatten() {
            let reaches = summaries
                .get_mut(caller)
                .expect("call-graph names originate in program definitions");
            if !*reaches {
                *reaches = true;
                queue.push_back(caller.clone());
            }
        }
    }

    let summaries = Arc::new(summaries);
    *program.facts.dynamic_to_tensor_def_summaries.borrow_mut() = Some(summaries.clone());
    summaries
}

fn collect_dynamic_to_tensor_def_refs(
    expr: &Expr,
    def_names: &BTreeSet<String>,
    callable_scope: &mut CallableScope,
    referenced_defs: &mut BTreeSet<String>,
) -> bool {
    record_host_work(|profile| profile.tensor_helper_preflight_nodes += 1);
    let mut directly_dynamic =
        !callable_scope.contains_key("to_tensor") && expr_is_runtime_shaped_to_tensor(expr);
    if let Some(name) = app_callee_name(expr)
        && let Some(target) = resolve_top_level_callable_from_names(name, callable_scope, def_names)
    {
        referenced_defs.insert(target);
    }

    let Some((expr_tag, _, kids)) = stamped_parts(expr) else {
        visit_semantic_expr_children(expr, |child| {
            directly_dynamic |= collect_dynamic_to_tensor_def_refs(
                child,
                def_names,
                callable_scope,
                referenced_defs,
            );
        });
        return directly_dynamic;
    };
    match expr_tag {
        DeepTag::Fn => {
            if let Some(params) = kids.first() {
                directly_dynamic |= collect_dynamic_to_tensor_def_refs(
                    params,
                    def_names,
                    callable_scope,
                    referenced_defs,
                );
            }
            let mut undos = Vec::new();
            if let Some(params) = kids.first().and_then(as_list) {
                for param in children(params) {
                    if let Some(name) = param_name(param) {
                        undos.push(callable_scope.bind(name, None));
                    }
                }
            }
            if let Some(body) = kids.get(1) {
                directly_dynamic |= collect_dynamic_to_tensor_def_refs(
                    body,
                    def_names,
                    callable_scope,
                    referenced_defs,
                );
            }
            callable_scope.restore(undos.into_iter().rev());
        }
        DeepTag::Let => {
            let mut undos = Vec::new();
            if let Some(bind_list) = kids.first().and_then(as_list)
                && tag(bind_list) == Some(DeepTag::Bind)
            {
                let bind_kids = children(bind_list);
                for index in (0..bind_kids.len()).step_by(2) {
                    let Some(name) = bind_kids.get(index).and_then(symbol_name) else {
                        continue;
                    };
                    let Some(value) = bind_kids.get(index + 1) else {
                        undos.push(callable_scope.bind(name.to_string(), None));
                        continue;
                    };
                    directly_dynamic |= collect_dynamic_to_tensor_def_refs(
                        value,
                        def_names,
                        callable_scope,
                        referenced_defs,
                    );
                    let target = direct_var_name(value).and_then(|target| {
                        resolve_top_level_callable_from_names(target, callable_scope, def_names)
                    });
                    undos.push(callable_scope.bind(name.to_string(), target));
                }
            } else if let Some(bindings) = kids.first() {
                directly_dynamic |= collect_dynamic_to_tensor_def_refs(
                    bindings,
                    def_names,
                    callable_scope,
                    referenced_defs,
                );
            }
            if let Some(body) = kids.get(1) {
                directly_dynamic |= collect_dynamic_to_tensor_def_refs(
                    body,
                    def_names,
                    callable_scope,
                    referenced_defs,
                );
            }
            callable_scope.restore(undos.into_iter().rev());
        }
        _ => visit_semantic_expr_children(expr, |child| {
            directly_dynamic |= collect_dynamic_to_tensor_def_refs(
                child,
                def_names,
                callable_scope,
                referenced_defs,
            );
        }),
    }
    directly_dynamic
}

fn analyze_tensor_helper_preflight(
    expr: &Expr,
    def_summaries: &BTreeMap<String, bool>,
    out: &mut UnordMap<usize, TensorHelperPreflightFacts>,
) -> TensorHelperPreflightFacts {
    analyze_tensor_helper_preflight_scoped(expr, def_summaries, &mut CallableScope::default(), out)
}

fn analyze_tensor_helper_preflight_scoped(
    expr: &Expr,
    def_summaries: &BTreeMap<String, bool>,
    callable_scope: &mut CallableScope,
    out: &mut UnordMap<usize, TensorHelperPreflightFacts>,
) -> TensorHelperPreflightFacts {
    record_host_work(|profile| profile.tensor_helper_preflight_nodes += 1);
    let mut facts = TensorHelperPreflightFacts {
        reaches_dynamic_to_tensor: !callable_scope.contains_key("to_tensor")
            && expr_is_runtime_shaped_to_tensor(expr),
        contains_grad_like: stamped_parts(expr)
            .is_some_and(|(tag, _, _)| matches!(tag, DeepTag::Grad | DeepTag::Vmap))
            || matches!(expr, Expr::List(list, _) if list.unknown_tag_symbol() == Some("vmap-grad")),
    };
    if let Some(name) = app_callee_name(expr)
        && let Some(target) =
            resolve_top_level_callable_from_summaries(name, callable_scope, def_summaries)
    {
        facts.reaches_dynamic_to_tensor |= def_summaries.get(&target).copied().unwrap_or(false);
    }

    let merge_child =
        |facts: &mut TensorHelperPreflightFacts,
         child: &Expr,
         callable_scope: &mut CallableScope,
         out: &mut UnordMap<usize, TensorHelperPreflightFacts>| {
            let child_facts =
                analyze_tensor_helper_preflight_scoped(child, def_summaries, callable_scope, out);
            facts.reaches_dynamic_to_tensor |= child_facts.reaches_dynamic_to_tensor;
            facts.contains_grad_like |= child_facts.contains_grad_like;
        };
    match stamped_parts(expr) {
        Some((DeepTag::Fn, _, kids)) => {
            if let Some(params) = kids.first() {
                merge_child(&mut facts, params, callable_scope, out);
            }
            let mut undos = Vec::new();
            if let Some(params) = kids.first().and_then(as_list) {
                for param in children(params) {
                    if let Some(name) = param_name(param) {
                        undos.push(callable_scope.bind(name, None));
                    }
                }
            }
            if let Some(body) = kids.get(1) {
                merge_child(&mut facts, body, callable_scope, out);
            }
            callable_scope.restore(undos.into_iter().rev());
        }
        Some((DeepTag::Let, _, kids)) => {
            let mut undos = Vec::new();
            if let Some(bind_list) = kids.first().and_then(as_list)
                && tag(bind_list) == Some(DeepTag::Bind)
            {
                let bind_kids = children(bind_list);
                for index in (0..bind_kids.len()).step_by(2) {
                    let Some(name) = bind_kids.get(index).and_then(symbol_name) else {
                        continue;
                    };
                    let Some(value) = bind_kids.get(index + 1) else {
                        undos.push(callable_scope.bind(name.to_string(), None));
                        continue;
                    };
                    merge_child(&mut facts, value, callable_scope, out);
                    let target = direct_var_name(value).and_then(|target| {
                        resolve_top_level_callable_from_summaries(
                            target,
                            callable_scope,
                            def_summaries,
                        )
                    });
                    undos.push(callable_scope.bind(name.to_string(), target));
                }
            } else if let Some(bindings) = kids.first() {
                merge_child(&mut facts, bindings, callable_scope, out);
            }
            if let Some(body) = kids.get(1) {
                merge_child(&mut facts, body, callable_scope, out);
            }
            callable_scope.restore(undos.into_iter().rev());
        }
        _ => visit_semantic_expr_children(expr, |child| {
            merge_child(&mut facts, child, callable_scope, out);
        }),
    }
    out.insert(expr as *const Expr as usize, facts);
    facts
}

fn app_callee_name(expr: &Expr) -> Option<&str> {
    let (DeepTag::App, _, kids) = stamped_parts(expr)? else {
        return None;
    };
    direct_var_name(kids.first()?)
}

fn direct_var_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::MetaExpr(meta, _) => direct_var_name(&meta.expr),
        _ => {
            let (DeepTag::Var, _, kids) = stamped_parts(expr)? else {
                return None;
            };
            kids.first().and_then(symbol_name)
        }
    }
}

fn resolve_top_level_callable_from_names(
    name: &str,
    callable_scope: &CallableScope,
    def_names: &BTreeSet<String>,
) -> Option<String> {
    match callable_scope.get(name) {
        Some(target) => target.clone(),
        None => def_names.contains(name).then(|| name.to_string()),
    }
}

fn resolve_top_level_callable_from_summaries(
    name: &str,
    callable_scope: &CallableScope,
    def_summaries: &BTreeMap<String, bool>,
) -> Option<String> {
    match callable_scope.get(name) {
        Some(target) => target.clone(),
        None => def_summaries.contains_key(name).then(|| name.to_string()),
    }
}

fn visit_semantic_expr_children(expr: &Expr, mut visit: impl FnMut(&Expr)) {
    match expr {
        Expr::Atom(_, _) => {}
        Expr::List(list, _) => {
            if tag(list).is_some() {
                for child in children(list) {
                    visit(child);
                }
            } else {
                for child in &list.elements {
                    visit(child);
                }
            }
        }
        Expr::Node(node, _) => {
            for child in node.children_slice() {
                visit(child);
            }
        }
        Expr::Map(map, _) => {
            map.visit_expressions(&mut |value, _| visit(value));
        }
        Expr::MetaExpr(meta, _) => visit(&meta.expr),
        Expr::BareList(elements, _) => {
            for child in elements {
                visit(child);
            }
        }
        Expr::UnknownForm(data) => {
            for child in &data.children {
                visit(child);
            }
        }
    }
}

fn expr_is_runtime_shaped_to_tensor(expr: &Expr) -> bool {
    let Some((DeepTag::App, _, kids)) = stamped_parts(expr) else {
        return false;
    };
    let Some((DeepTag::Var, _, callee_kids)) = kids.first().and_then(stamped_parts) else {
        return false;
    };
    let is_to_tensor = callee_kids.first().and_then(symbol_name) == Some("to_tensor");
    is_to_tensor && !crate::lower::is_static_to_tensor_literal(expr)
}

/// Does this expression reach a runtime-shaped `to_tensor`, accounting for
/// lexical shadowing of the name and for the program's call graph?
///
/// This is the extractor's `reaches_dynamic_to_tensor` fact asked about one
/// body directly, rather than through the pointer-keyed preflight stack, so a
/// decision taken before that stack exists can read the same signal.
/// chelis#1779.
fn expr_reaches_dynamic_to_tensor(expr: &Expr, program: &HostLoweringSession<'_>) -> bool {
    let summaries = cached_dynamic_to_tensor_def_summaries(program);
    let mut facts = UnordMap::new();
    analyze_tensor_helper_preflight(expr, &summaries, &mut facts).reaches_dynamic_to_tensor
}

fn tensor_helper_preflight_rejects(expr: &Expr) -> bool {
    let key = expr as *const Expr as usize;
    TENSOR_HELPER_PREFLIGHT_STACK.with(|stack| {
        let stack = stack.borrow();
        let facts = stack
            .iter()
            .rev()
            .find_map(|facts| facts.get(&key).copied());
        record_host_work(|profile| {
            if facts.is_some() {
                profile.tensor_helper_preflight_lookup_hits += 1;
            } else {
                profile.tensor_helper_preflight_lookup_misses += 1;
            }
        });
        facts.is_some_and(|facts| facts.reaches_dynamic_to_tensor && !facts.contains_grad_like)
    })
}

fn collect_program_defs(exprs: &[Expr]) -> BTreeMap<String, Expr> {
    record_host_work(|profile| profile.program_def_collections += 1);
    let mut defs = BTreeMap::new();
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) == Some(DeepTag::Def)
            && let (Some(name), Some(body)) = (
                children(list).first().and_then(symbol_name),
                children(list).get(1),
            )
        {
            record_host_work(|profile| {
                profile.program_def_clone_nodes += deep_expr_nodes(body);
            });
            defs.insert(name.to_string(), body.clone());
        }
    }
    defs
}

fn top_level_items(exprs: &[Expr]) -> Vec<&Expr> {
    let mut out = Vec::new();
    for expr in exprs {
        collect_top_level_items(expr, &mut out);
    }
    out
}

fn collect_top_level_items<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let ExprCarrier::DecodedNode(DeepTag::Module, _, children) = expr.carrier() {
        for child in children.iter().skip(1) {
            collect_top_level_items(child, out);
        }
        return;
    }
    out.push(expr);
}

fn collect_tensor_scope(scope: &UnordMap<String, HostTypeTerm>) -> UnordMap<String, TensorType> {
    scope
        .to_sorted()
        .into_iter()
        .filter_map(|(name, ty)| {
            tensor_type_from_host_input(ty).map(|tensor| (name.clone(), tensor))
        })
        .collect()
}

pub(crate) fn tensor_type_from_host_input(ty: &HostTypeTerm) -> Option<TensorType> {
    match ty {
        HostTypeTerm::Tensor(tensor) => Some(tensor.clone()),
        HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(prim))
            if prim.is_admissible_active() && *prim != Prim::String =>
        {
            Some(TensorType {
                dims: vec![],
                precision: *prim,
            })
        }
        _ => None,
    }
}

fn host_type_from_tensor_input(ty: &TensorType) -> HostTypeTerm {
    if ty.dims.is_empty() {
        if ty.precision == Prim::String {
            HostTypeTerm::String
        } else {
            HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(ty.precision))
        }
    } else {
        HostTypeTerm::Tensor(ty.clone())
    }
}

/// A node's host type, with the active specialization's type variables
/// resolved (chelis#1201).
///
/// The raw form returns the GENERIC checked type the checker recorded, so
/// inside a monomorphized body a generic ADT reads as unresolved even though
/// the call site pinned it. Applying the substitution here catches every
/// route a type arrives by — node metadata, `CheckedProgram` lookup, and the
/// structural fallbacks — rather than only the metadata funnel. Outside a
/// specialization the substitution is empty and this is the identity.
pub(crate) fn expr_host_type(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
) -> HostTypeTerm {
    let raw = expr_host_type_raw(expr, program, scope);
    let subst = active_type_subst();
    let specialized = if subst.is_empty() {
        raw
    } else {
        apply_host_type_subst(&raw, &subst)
    };
    expand_host_type_aliases(program, specialized)
}

/// Resolve a transparent alias after reef linking has qualified the use-site
/// spelling but the checker registry still owns the declaration's authored
/// spelling. Exact identity wins; a terminal fallback is valid only when it
/// identifies one alias, matching the other linked-name lookups in this lane.
fn resolve_host_type_alias<'a>(registry: &'a AdtRegistry, name: &str) -> Option<&'a TypeAliasDef> {
    if let Some(alias) = registry.resolve_alias(name) {
        return Some(alias);
    }
    if registry.lookup(name).is_some() {
        return None;
    }
    let mut matches = registry
        .aliases
        .iter()
        .filter_map(|(key, alias)| terminal_name_matches(key, name).then_some(alias));
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

/// Expand checker-validated aliases before a host type drives layout,
/// constructor, field, or match decisions.
///
/// The checker keeps an authored alias name in expression metadata even
/// though aliases are semantically transparent. Host lowering previously
/// happened to work only when the type was a direct ADT. In particular,
/// `type Pair[n] = Column[n]` left an access through `Pair[2]` looking for a
/// nonexistent `Pair` constructor layout. Resolve the alias at the single
/// host-type boundary so every downstream consumer sees the expanded field or
/// tensor type. Dimension-kinded alias arguments remain dimensions during
/// substitution; they are erased only when an enclosing nominal value's host
/// layout deliberately has no slot for them.
fn expand_host_type_aliases(program: &HostLoweringSession<'_>, ty: HostTypeTerm) -> HostTypeTerm {
    fn expand(
        program: &HostLoweringSession<'_>,
        ty: HostTypeTerm,
        visiting: &mut UnordSet<String>,
    ) -> HostTypeTerm {
        match ty {
            HostTypeTerm::Adt(name, args) => {
                let args = args
                    .into_iter()
                    .map(|argument| expand(program, argument, visiting))
                    .collect::<Vec<_>>();
                let Some(alias) = resolve_host_type_alias(program.adt_registry(), &name) else {
                    return HostTypeTerm::Adt(name, args);
                };
                if alias.params.len() != args.len() || !visiting.insert(name.clone()) {
                    return HostTypeTerm::Adt(name, args);
                }
                let checker_parameter_names = alias
                    .param_args
                    .iter()
                    .zip(&alias.params)
                    .filter_map(|(argument, parameter)| match argument {
                        NominalArg::Type(Type::Var(variable)) => {
                            Some((format!("t{}", variable.0), parameter.clone()))
                        }
                        NominalArg::Dimension(Dim::Var(variable)) => {
                            Some((format!("d{}", variable.0), parameter.clone()))
                        }
                        _ => None,
                    })
                    .collect::<UnordMap<_, _>>();
                let body = rename_host_type_variables(
                    decode_host_type_or_raise(
                        &type_to_deep_expr(&program.adt_registry().expand_aliases(&alias.body)),
                        &UnordMap::new(),
                    ),
                    &checker_parameter_names,
                );
                let parameter_kinds = if alias.param_kinds.len() == alias.params.len() {
                    Cow::Borrowed(alias.param_kinds.as_slice())
                } else {
                    Cow::Owned(vec![NominalParamKind::Type; alias.params.len()])
                };
                let mut type_substitutions = UnordMap::new();
                let mut dimension_substitutions = UnordMap::new();
                for ((parameter, kind), argument) in
                    alias.params.iter().zip(parameter_kinds.iter()).zip(args)
                {
                    match (kind, argument) {
                        (NominalParamKind::Type, argument) => {
                            type_substitutions.insert(parameter.clone(), argument);
                        }
                        (NominalParamKind::Dimension, HostTypeTerm::TypeVariable(argument)) => {
                            dimension_substitutions
                                .insert(parameter.clone(), DimInfo::Named(argument, None));
                        }
                        (NominalParamKind::Dimension, _) => {}
                    }
                }
                let expanded = expand(
                    program,
                    substitute_host_dimension_terms(
                        substitute_host_type_term(body, &type_substitutions),
                        &dimension_substitutions,
                    ),
                    visiting,
                );
                visiting.remove(&name);
                expanded
            }
            HostTypeTerm::Fn(params, ret) => HostTypeTerm::Fn(
                params
                    .into_iter()
                    .map(|param| expand(program, param, visiting))
                    .collect(),
                Box::new(expand(program, *ret, visiting)),
            ),
            HostTypeTerm::List(inner) => {
                HostTypeTerm::List(Box::new(expand(program, *inner, visiting)))
            }
            HostTypeTerm::Dict(key, value) => HostTypeTerm::Dict(
                Box::new(expand(program, *key, visiting)),
                Box::new(expand(program, *value, visiting)),
            ),
            HostTypeTerm::Tuple(items) => HostTypeTerm::Tuple(
                items
                    .into_iter()
                    .map(|item| expand(program, item, visiting))
                    .collect(),
            ),
            HostTypeTerm::Option(inner) => {
                HostTypeTerm::Option(Box::new(expand(program, *inner, visiting)))
            }
            other => other,
        }
    }

    expand(program, ty, &mut UnordSet::new())
}

/// Replace every bound `TypeVariable` in a host type (chelis#1201).
fn apply_host_type_subst(
    ty: &HostTypeTerm,
    subst: &UnordMap<String, HostTypeTerm>,
) -> HostTypeTerm {
    match ty {
        HostTypeTerm::TypeVariable(name) => subst.get(name).cloned().unwrap_or_else(|| ty.clone()),
        HostTypeTerm::Adt(name, args) => HostTypeTerm::Adt(
            name.clone(),
            args.iter()
                .map(|a| apply_host_type_subst(a, subst))
                .collect(),
        ),
        HostTypeTerm::List(inner) => {
            HostTypeTerm::List(Box::new(apply_host_type_subst(inner, subst)))
        }
        HostTypeTerm::Option(inner) => {
            HostTypeTerm::Option(Box::new(apply_host_type_subst(inner, subst)))
        }
        HostTypeTerm::Dict(key, value) => HostTypeTerm::Dict(
            Box::new(apply_host_type_subst(key, subst)),
            Box::new(apply_host_type_subst(value, subst)),
        ),
        HostTypeTerm::Tuple(items) => HostTypeTerm::Tuple(
            items
                .iter()
                .map(|i| apply_host_type_subst(i, subst))
                .collect(),
        ),
        HostTypeTerm::Fn(params, ret) => HostTypeTerm::Fn(
            params
                .iter()
                .map(|p| apply_host_type_subst(p, subst))
                .collect(),
            Box::new(apply_host_type_subst(ret, subst)),
        ),
        _ => ty.clone(),
    }
}

fn expr_host_type_raw(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
) -> HostTypeTerm {
    match expr {
        Expr::Atom(Atom::Int(_), _) => HostTypeTerm::Int64,
        Expr::Atom(Atom::Float(_), _) => HostTypeTerm::Float64,
        Expr::Atom(Atom::Bool(_), _) => HostTypeTerm::Bool,
        Expr::Atom(Atom::Str(_), _) => HostTypeTerm::String,
        Expr::List(_, _) | Expr::Node(_, _) if expr.tag() == Some(DeepTag::Var) => {
            stamped_parts(expr)
                .and_then(|(_, _, kids)| kids.first())
                .and_then(symbol_name)
                .and_then(|name| {
                    expr_type(expr)
                        .filter(|ty| !ty.is_unresolved())
                        .or_else(|| {
                            scope
                                .get(name)
                                .cloned()
                                .or_else(|| lookup_declared_host_type(program, name))
                        })
                })
                .unwrap_or_else(fresh_host_inference)
        }
        Expr::List(_, _) | Expr::Node(_, _) if expr.tag() == Some(DeepTag::App) => {
            let explicit = expr_type(expr).unwrap_or_else(fresh_host_inference);
            if app_expr_needs_inferred_type(&explicit) {
                let inferred = infer_app_expr_host_type(expr, program, scope)
                    .unwrap_or_else(fresh_host_inference);
                if should_prefer_inferred_app_type(&explicit, &inferred) {
                    inferred
                } else if !explicit.is_unresolved() {
                    explicit
                } else {
                    inferred
                }
            } else {
                explicit
            }
        }
        _ => expr_type(expr).unwrap_or_else(fresh_host_inference),
    }
}

fn app_expr_needs_inferred_type(explicit: &HostTypeTerm) -> bool {
    explicit.is_unresolved() || host_type_has_synthetic_tensor_dims(explicit)
}

fn infer_app_expr_host_type(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
) -> Option<HostTypeTerm> {
    let (DeepTag::App, _, kids) = stamped_parts(expr)? else {
        return None;
    };
    let callee = kids.first()?;
    let (DeepTag::Var, _, callee_kids) = stamped_parts(callee)? else {
        return None;
    };
    let name = callee_kids.first().and_then(symbol_name)?;
    if !BUILTIN_NAMES.contains(&name) {
        // Predicate disposition (chelis#1271): this returns the type an
        // application would produce, so answering an ambiguous name would
        // stamp the wrong package's ADT name onto the expression. Decline
        // instead; the application itself rejects the same ambiguity.
        if let Some(definition) = lookup_adt_constructor_definition(program, name) {
            let mut substitutions = UnordMap::new();
            for (field, argument) in definition.fields.iter().zip(kids.iter().skip(1)) {
                let applied = expr_host_type(argument, program, scope);
                collect_host_type_variable_substitutions(&field.term, &applied, &mut substitutions);
            }
            let result = HostTypeTerm::Adt(
                definition.adt_name,
                definition
                    .parameters
                    .into_iter()
                    .map(HostTypeTerm::TypeVariable)
                    .collect(),
            );
            return Some(substitute_host_type_term(result, &substitutions));
        }
        let (params, ret) = lookup_declared_fn_type(program, name)?;
        let mut substitutions = UnordMap::new();
        for (pattern, argument) in params.iter().zip(kids.iter().skip(1)) {
            let applied = expr_host_type(argument, program, scope);
            collect_host_type_variable_substitutions(pattern, &applied, &mut substitutions);
        }
        let ret = substitute_host_type_term(ret, &substitutions);
        return Some(ret);
    }
    if name == "einsum" {
        let equation = match kids.get(1) {
            Some(Expr::Atom(Atom::Str(value), _)) => value.as_str(),
            _ => return Some(fresh_host_inference()),
        };
        let tensors = kids[2..]
            .iter()
            .map(|arg| match expr_host_type(arg, program, scope) {
                HostTypeTerm::Tensor(tensor_ty) => Some(tensor_ty),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        return infer_einsum_tensor_type(equation, &tensors).map(HostTypeTerm::Tensor);
    }
    if name == "count"
        && let Some(input) = kids.get(1)
        && let HostTypeTerm::Tensor(tensor_ty) = expr_host_type(input, program, scope)
    {
        let rank = tensor_ty.dims.len();
        let mut axes = Vec::with_capacity(kids.len().saturating_sub(2));
        for axis_expr in &kids[2..] {
            let raw = expr_int_literal(axis_expr)?;
            let axis = if raw < 0 {
                rank.checked_sub(raw.unsigned_abs() as usize)?
            } else {
                usize::try_from(raw).ok().filter(|&axis| axis < rank)?
            };
            if axes.contains(&axis) {
                return None;
            }
            axes.push(axis);
        }
        if axes.is_empty() {
            return None;
        }
        return Some(HostTypeTerm::Tensor(TensorType {
            dims: tensor_ty
                .dims
                .into_iter()
                .enumerate()
                .filter_map(|(axis, dim)| (!axes.contains(&axis)).then_some(dim))
                .collect(),
            precision: Prim::Int64,
        }));
    }
    // chelis#340: the whole named-axis reduction family is type-inferred
    // here (the positional/int-literal axis form), not only `sum`/`mean`.
    // Each drops the reduced axis; `argmax_reduce`/`argmin_reduce` return an
    // i64 index tensor while the value reductions keep the operand
    // precision. Recovering the tensor type lets the host lane route the
    // call through `try_lower_tensor_helper_call` (the tensor-DAG kernel
    // lane the C backend uses) instead of falling through to the
    // host-emit "unsupported builtin" path when the checker's `type`
    // annotation is absent or carries synthetic rank-poly dims.
    if matches!(
        name,
        "sum"
            | "mean"
            | "max_reduce"
            | "min_reduce"
            | "prod_reduce"
            | "argmax_reduce"
            | "argmin_reduce"
    ) && let (Some(input), Some(axis_expr)) = (kids.get(1), kids.get(2))
        && let HostTypeTerm::Tensor(tensor_ty) = expr_host_type(input, program, scope)
        && let Some(axis) = expr_int_literal(axis_expr)
    {
        let mut reduced = reduce_axis_tensor_type(&tensor_ty, axis as usize);
        if matches!(name, "argmax_reduce" | "argmin_reduce") {
            reduced.precision = chelis_types::types::Prim::Int64;
        }
        return Some(HostTypeTerm::Tensor(reduced));
    }
    if (name == "expand" || name == "insert")
        && let (Some(input), Some(axis_expr), Some(size_expr)) =
            (kids.get(1), kids.get(2), kids.get(3))
        && let HostTypeTerm::Tensor(tensor_ty) = expr_host_type(input, program, scope)
        && let (Some(axis), Some(size)) = (expr_int_literal(axis_expr), expr_int_literal(size_expr))
    {
        let mut dims = tensor_ty.dims.clone();
        let axis = axis as usize;
        if axis <= dims.len() {
            dims.insert(axis, crate::dag::DimInfo::Lit(size as usize));
            return Some(HostTypeTerm::Tensor(TensorType {
                dims,
                precision: tensor_ty.precision,
            }));
        }
    }
    // Issue #308: `scalar_to_tensor` result precision must follow the
    // operand's *float* precision. The coarse host lane collapses f32
    // and f64 scalars into a single `HostTypeTerm::Float64`, so the
    // arg-ty-based fallback below cannot distinguish them and defaults
    // to f32 — mis-typing an f64 const-broadcast
    // (`scalar_to_tensor(cast(c, f64))`) as `Tensor(F32)`, which makes
    // the C emitter select `chelis_scalar_tensor_from_f32` (4-byte
    // storage) for an f64 value. Recover the precision from the Deep
    // operand itself (checker annotation or explicit cast target)
    // while it is still visible.
    if name == "scalar_to_tensor"
        && let Some(arg) = kids.get(1)
        && let Some(precision) = expr_scalar_float_precision(arg)
    {
        return Some(HostTypeTerm::Tensor(TensorType {
            dims: vec![],
            precision,
        }));
    }
    // chelis#631: a tensor-list `concat`'s concat axis is sized at run
    // time by `chelis_tensor_concat`; the coarse host type must not carry
    // the ELEMENT's extent on that axis. (The precise type is the
    // checker's `tensor_concat_result_type`, spec §4.5.4; this fallback
    // fires when the checker annotation is absent or synthetic.)
    if name == "concat"
        && let Some(input) = kids.get(1)
        && let HostTypeTerm::List(inner) = expr_host_type(input, program, scope)
        && let HostTypeTerm::Tensor(element) = inner.as_ref()
    {
        let axis = kids
            .get(2)
            .and_then(expr_int_literal)
            .and_then(|raw| normalize_host_axis(element.dims.len(), raw));
        return Some(concat_host_tensor_type(element, axis));
    }
    let arg_tys = kids[1..]
        .iter()
        .map(|arg| expr_host_type(arg, program, scope))
        .collect::<Vec<_>>();
    must_infer_builtin_host_type_from_arg_tys(name, &arg_tys)
}

/// Issue #308: recover the float precision of a scalar Deep expression
/// for `scalar_to_tensor` result typing. Reads, in order:
///
///   1. the checker's `type` meta when it is a float `(t-prim {} p)`;
///   2. an explicit `(cast {} _ (t-prim {} p))` target when `p` is a
///      float precision.
///
/// Returns `None` for integer/bool operands (the coarse
/// `infer_builtin_host_type_from_arg_tys` arms already type those
/// correctly) and when the precision is not recoverable — in that case
/// the caller falls back to the coarse f32 default, which the C emit
/// dispatch and the consuming tensor-helper Load both share, so the
/// write and read sides stay consistent even in the fallback.
fn expr_scalar_float_precision(expr: &Expr) -> Option<chelis_types::types::Prim> {
    expr_scalar_primitive(expr).filter(|prim| prim.is_float())
}

fn expr_scalar_primitive(expr: &Expr) -> Option<chelis_types::types::Prim> {
    if let Expr::MetaExpr(meta, _) = expr {
        return expr_scalar_primitive(&meta.expr);
    }
    let (node_tag, meta, kids) = stamped_parts(expr)?;
    let prim_of_t_prim = |type_expr: &Expr| -> Option<chelis_types::types::Prim> {
        let (DeepTag::TPrim, _, kids) = stamped_parts(type_expr)? else {
            return None;
        };
        kids.first()
            .and_then(symbol_name)
            .and_then(chelis_types::types::Prim::parse_name)
    };
    if let Some(type_expr) = meta.ty().map(|ty| ty.expression())
        && let Some(prim) = prim_of_t_prim(type_expr)
    {
        return Some(prim);
    }
    if node_tag == DeepTag::Cast
        && let Some(target) = kids.get(1)
        && let Some(prim) = prim_of_t_prim(target)
    {
        return Some(prim);
    }
    None
}

/// The §5.3 source width retained when a binder-adopted decimal is
/// specialized across families to an integer cast target.
fn binder_float_literal_keeps_f32_source(operand: &Expr, target: &HostTypeTerm) -> bool {
    let Some(source) = chelis_deep::classify_literal_source(operand) else {
        return false;
    };
    if !matches!(source.numeric_atom(), Some(Atom::Float(_)))
        || !source.admitted_by(chelis_deep::DtypeFamily::Numeric)
    {
        return false;
    }
    // Unique typed annotations give every lane the same binder identity.
    let binder_typed = source
        .metadata()
        .ty()
        .map(|value| value.expression())
        .is_some_and(|ty| chelis_deep::exact_type_variable_name(ty).is_some());
    binder_typed
        && matches!(
            target,
            HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision)) if precision.is_integer()
        )
}

fn should_prefer_inferred_app_type(explicit: &HostTypeTerm, inferred: &HostTypeTerm) -> bool {
    explicit.is_unresolved()
        || matches!(
            (explicit, inferred),
            (HostTypeTerm::Tensor(_), HostTypeTerm::Tensor(_)) if host_type_has_synthetic_tensor_dims(explicit)
                && !host_type_has_synthetic_tensor_dims(inferred)
        )
}

fn host_type_has_synthetic_tensor_dims(ty: &HostTypeTerm) -> bool {
    // Known exposure (review #363 N5, pre-existing): a USER dim literally
    // named `d2` matches this minted-name heuristic and would be treated
    // as synthetic. The checker's dim-var minting owns the `d<digits>`
    // namespace today; if user-facing single-letter+digit dims ever
    // matter, the minting needs a reserved prefix instead.
    fn synthetic_dim_name(name: &str) -> bool {
        let mut chars = name.chars();
        matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
    }

    match ty {
        HostTypeTerm::Tensor(tensor_ty) => tensor_ty.dims.iter().any(
            |dim| matches!(dim, crate::dag::DimInfo::Named(name, None) if synthetic_dim_name(name)),
        ),
        HostTypeTerm::Tuple(items) => items.iter().any(host_type_has_synthetic_tensor_dims),
        HostTypeTerm::List(inner) | HostTypeTerm::Option(inner) => {
            host_type_has_synthetic_tensor_dims(inner)
        }
        _ => false,
    }
}

fn infer_einsum_tensor_type(equation: &str, tensors: &[TensorType]) -> Option<TensorType> {
    let (inputs, output) = equation.split_once("->")?;
    let input_specs = inputs.split(',').map(str::trim).collect::<Vec<_>>();
    if input_specs.len() != tensors.len() {
        return None;
    }

    let mut labels = UnordMap::<char, crate::dag::DimInfo>::new();
    for (spec, tensor) in input_specs.iter().zip(tensors.iter()) {
        let axes = spec
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<Vec<_>>();
        if axes.len() != tensor.dims.len() {
            return None;
        }
        for (axis, dim) in axes.into_iter().zip(tensor.dims.iter().cloned()) {
            labels.entry(axis).or_insert(dim);
        }
    }

    let dims = output
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .map(|axis| labels.get(&axis).cloned())
        .collect::<Option<Vec<_>>>()?;
    let precision = tensors
        .first()
        .map(|tensor| tensor.precision)
        .unwrap_or(chelis_types::types::Prim::F32);
    Some(TensorType { dims, precision })
}

fn expr_int_literal(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Atom(Atom::Int(value), _) => Some(*value),
        Expr::MetaExpr(meta, _) => expr_int_literal(&meta.expr),
        Expr::List(_, _) | Expr::Node(_, _) if expr.tag() == Some(DeepTag::Lit) => {
            stamped_parts(expr)?.2.first().and_then(expr_int_literal)
        }
        // Movement-op axis/size args are routinely written as
        // `cast(0, i32)` / `cast(2, i32)` (the canonical integer-
        // literal form, since bare int literals default to i32 and the
        // axis/size parameters are i32). A `cast` whose operand is an
        // integer literal carries the same compile-time value, so see
        // through it: otherwise `infer_app_expr_host_type`'s `expand`
        // shape handler bails and the result type degrades to a
        // dims-less placeholder, splitting a constant-broadcast `let`
        // binding into an unsupported host-lane builtin (issue #300).
        Expr::List(_, _) | Expr::Node(_, _) if expr.tag() == Some(DeepTag::Cast) => {
            stamped_parts(expr)?.2.first().and_then(expr_int_literal)
        }
        _ => None,
    }
}

/// chelis#631: an integer literal reaching this HostExpr position,
/// seeing through the canonical `cast(N, i32)` spelling (the HostExpr
/// analog of [`expr_int_literal`]'s cast peel).
fn host_expr_int_literal(expr: &HostExpr) -> Option<i64> {
    match &expr.kind {
        HostExprKind::Int(value) => Some(*value),
        HostExprKind::Builtin { name, args, .. } if name == "cast" || name == "cast_trunc" => {
            args.first().and_then(host_expr_int_literal)
        }
        _ => None,
    }
}

/// chelis#631: normalize a possibly-negative literal axis against a rank
/// (`-1` is the last axis); `None` when out of range — the checker owns
/// the user-facing out-of-bounds diagnostic, this lane just degrades to
/// all-wildcard.
fn normalize_host_axis(rank: usize, raw: i64) -> Option<usize> {
    let rank = rank as i64;
    let axis = if raw < 0 { rank + raw } else { raw };
    (0..rank).contains(&axis).then_some(axis as usize)
}

/// chelis#631: the coarse host-lane type of a tensor-list `concat` — the
/// element type with the concat axis wildcarded (every axis when the
/// axis is unknown in the calling lane). The wildcard renames to a
/// per-node anon dim in the C emitter and is sized from the runtime
/// `chelis_tensor_concat` result, so helper signatures stay honest.
fn concat_host_tensor_type(element: &TensorType, axis: Option<usize>) -> HostTypeTerm {
    let anon = || crate::dag::DimInfo::Named("*".to_string(), None);
    let mut dims = element.dims.clone();
    match axis {
        Some(axis) if axis < dims.len() => dims[axis] = anon(),
        _ => dims.fill(anon()),
    }
    HostTypeTerm::Tensor(TensorType {
        dims,
        precision: element.precision,
    })
}

fn reduce_axis_tensor_type(tensor_ty: &TensorType, axis: usize) -> TensorType {
    let mut dims = tensor_ty.dims.clone();
    if axis < dims.len() {
        dims.remove(axis);
    }
    TensorType {
        dims,
        precision: tensor_ty.precision,
    }
}

fn lookup_type_expr<'a>(type_env: &'a BTreeMap<String, Expr>, name: &str) -> Option<&'a Expr> {
    type_env.get(name).or_else(|| {
        let mut matches = type_env
            .iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    })
}

fn lookup_authored_defsig_type_expr(program: &HostLoweringSession<'_>, name: &str) -> Option<Expr> {
    let mut exact = None;
    let mut terminal_matches = Vec::new();
    for expr in top_level_items(program.exprs()) {
        let Some((DeepTag::Defsig, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(candidate), Some(signature)) = (kids.first().and_then(symbol_name), kids.last())
        else {
            continue;
        };
        if candidate == name {
            exact = Some(signature.clone());
            break;
        }
        if terminal_name_matches(candidate, name) {
            terminal_matches.push(signature.clone());
        }
    }
    exact.or_else(|| {
        (terminal_matches.len() == 1)
            .then(|| terminal_matches.into_iter().next())
            .flatten()
    })
}

fn lookup_declared_type_expr(program: &HostLoweringSession<'_>, name: &str) -> Option<Expr> {
    lookup_authored_defsig_type_expr(program, name)
        .or_else(|| checked_authored_function_signature(program, name).map(type_to_deep_expr))
        .or_else(|| lookup_type_expr(program.type_env(), name).cloned())
}

fn lookup_declared_host_type(
    program: &HostLoweringSession<'_>,
    name: &str,
) -> Option<HostTypeTerm> {
    lookup_declared_type_expr(program, name)
        .as_ref()
        .and_then(|ty| decode_expanded_host_type_expr(program, ty))
}

fn lookup_declared_fn_type(
    program: &HostLoweringSession<'_>,
    name: &str,
) -> Option<(Vec<HostTypeTerm>, HostTypeTerm)> {
    lookup_declared_type_expr(program, name)
        .as_ref()
        .and_then(|ty| parse_expanded_fn_type_expr(program, ty))
}

fn lookup_program_def<'a>(defs: &'a BTreeMap<String, Expr>, name: &str) -> Option<&'a Expr> {
    defs.get(name).or_else(|| {
        let mut matches = defs
            .iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    })
}

fn terminal_name_matches(full_name: &str, short_name: &str) -> bool {
    full_name == short_name || terminal_name(full_name) == terminal_name(short_name)
}

fn terminal_name(name: &str) -> &str {
    name.rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(name)
}

fn find_top_level_def_expr<'a>(exprs: &'a [Expr], name: &str) -> Option<&'a Expr> {
    find_top_level_def_named(exprs, name).map(|(_, body)| body)
}

/// Resolve a (possibly short-spelled) reference to its defining top-level
/// declaration, returning the declaration's own name alongside the body.
/// The declaration's name is the canonical identity for specialization
/// interning (harden-bounded-monomorphization D4).
fn find_top_level_def_named<'a>(exprs: &'a [Expr], name: &str) -> Option<(&'a str, &'a Expr)> {
    let mut terminal_match = None;
    let mut terminal_is_ambiguous = false;
    for expr in top_level_items(exprs) {
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some(def_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        if def_name == name {
            return Some((def_name, body));
        }
        if terminal_name_matches(def_name, name) {
            if terminal_match.is_some() {
                terminal_is_ambiguous = true;
            } else {
                terminal_match = Some((def_name, body));
            }
        }
    }
    (!terminal_is_ambiguous).then_some(terminal_match).flatten()
}

fn expr_tensor_type(
    expr: &Expr,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
) -> Option<TensorType> {
    match expr_host_type(expr, program, scope) {
        HostTypeTerm::Tensor(ty) => Some(ty),
        _ => None,
    }
}

fn expr_type(expr: &Expr) -> Option<HostTypeTerm> {
    // Parameter binders are untagged list carriers `(name {type: ...})`,
    // not Deep vocabulary nodes. Preserve their source-authored metadata
    // while reading stamped vocabulary nodes directly; routing all Lists
    // through `stamped_parts` erases symbolic binder names such as `n` and
    // replaces them with checker-minted `dN` aliases at host emission.
    let meta = match expr {
        Expr::List(list, _) => match list.elements.get(1) {
            Some(Expr::Map(meta, _)) => meta,
            _ => return None,
        },
        Expr::BareList(elements, _) => match elements.get(1) {
            Some(Expr::Map(meta, _)) => meta,
            _ => return None,
        },
        Expr::Node(node, _) => node.meta(),
        _ => return None,
    };
    meta.ty()
        .map(|ty| ty.expression())
        .map(|value| decode_host_type_or_raise(value, &active_type_subst()))
}

fn expr_fn_type(expr: &Expr) -> Option<(Vec<HostTypeTerm>, HostTypeTerm)> {
    let meta = match expr {
        Expr::List(list, _) => match list.elements.get(1) {
            Some(Expr::Map(meta, _)) => meta,
            _ => return None,
        },
        Expr::Node(node, _) => node.meta(),
        _ => return None,
    };
    meta.ty()
        .map(|ty| ty.expression())
        .and_then(parse_fn_type_expr)
}

fn parse_fn_type_expr(expr: &Expr) -> Option<(Vec<HostTypeTerm>, HostTypeTerm)> {
    let (args, ret) = parse_fn_type_expr_parts(expr)?;
    Some((
        args.iter()
            .map(|arg| decode_host_type_or_raise(arg, &UnordMap::new()))
            .collect(),
        decode_host_type_or_raise(&ret, &UnordMap::new()),
    ))
}

fn expand_host_fn_type_aliases(
    program: &HostLoweringSession<'_>,
    signature: (Vec<HostTypeTerm>, HostTypeTerm),
) -> (Vec<HostTypeTerm>, HostTypeTerm) {
    let (params, ret) = signature;
    (
        params
            .into_iter()
            .map(|param| expand_host_type_aliases(program, param))
            .collect(),
        expand_host_type_aliases(program, ret),
    )
}

fn authored_nominal_dimension(expr: &Expr) -> Option<DimInfo> {
    match expr {
        Expr::MetaExpr(meta, _) => return authored_nominal_dimension(&meta.expr),
        Expr::Atom(Atom::Name(name), _) => return Some(DimInfo::Named(name.clone(), None)),
        Expr::Atom(Atom::Int(value), _) => {
            return usize::try_from(*value).ok().map(DimInfo::Lit);
        }
        _ => {}
    }
    let (tag, _, children) = stamped_parts(expr)?;
    match tag {
        // Before nominal-parameter kinds are resolved, Surf spells every
        // symbolic application argument as `t-var`. The declaration's kind
        // is the authority that lets this boundary interpret it as a
        // dimension without changing the public Deep vocabulary.
        DeepTag::TVar | DeepTag::DName | DeepTag::DVar => children
            .first()
            .and_then(symbol_name)
            .map(|name| DimInfo::Named(name.to_string(), None)),
        DeepTag::DLit => match children.first() {
            Some(Expr::Atom(Atom::Int(value), _)) => usize::try_from(*value).ok().map(DimInfo::Lit),
            _ => None,
        },
        DeepTag::DRank => None,
        _ => None,
    }
}

fn decode_expanded_host_type_expr(
    program: &HostLoweringSession<'_>,
    expr: &Expr,
) -> Option<HostTypeTerm> {
    if let Expr::MetaExpr(meta, _) = expr {
        return decode_expanded_host_type_expr(program, &meta.expr);
    }
    if let Some((DeepTag::TAdt, _, children)) = stamped_parts(expr)
        && let Some((name, arguments)) = children
            .split_first()
            .and_then(|(name, arguments)| symbol_name(name).map(|name| (name, arguments)))
        && let Some(alias) = resolve_host_type_alias(program.adt_registry(), name)
        && alias.params.len() == arguments.len()
    {
        let parameter_kinds = if alias.param_kinds.len() == alias.params.len() {
            Cow::Borrowed(alias.param_kinds.as_slice())
        } else {
            Cow::Owned(vec![NominalParamKind::Type; alias.params.len()])
        };
        let mut type_substitutions = UnordMap::new();
        let mut dimension_substitutions = UnordMap::new();
        for ((parameter, kind), argument) in alias
            .params
            .iter()
            .zip(parameter_kinds.iter())
            .zip(arguments)
        {
            match kind {
                NominalParamKind::Type => {
                    type_substitutions.insert(
                        parameter.clone(),
                        decode_expanded_host_type_expr(program, argument)?,
                    );
                }
                NominalParamKind::Dimension => {
                    dimension_substitutions
                        .insert(parameter.clone(), authored_nominal_dimension(argument)?);
                }
            }
        }
        let checker_parameter_names = alias
            .param_args
            .iter()
            .zip(&alias.params)
            .filter_map(|(argument, parameter)| match argument {
                NominalArg::Type(Type::Var(variable)) => {
                    Some((format!("t{}", variable.0), parameter.clone()))
                }
                NominalArg::Dimension(Dim::Var(variable)) => {
                    Some((format!("d{}", variable.0), parameter.clone()))
                }
                _ => None,
            })
            .collect::<UnordMap<_, _>>();
        let body = rename_host_type_variables(
            decode_host_type(&type_to_deep_expr(
                &program.adt_registry().expand_aliases(&alias.body),
            ))
            .ok()?,
            &checker_parameter_names,
        );
        return Some(expand_host_type_aliases(
            program,
            substitute_host_dimension_terms(
                substitute_host_type_term(body, &type_substitutions),
                &dimension_substitutions,
            ),
        ));
    }
    decode_host_type(expr)
        .ok()
        .map(|term| expand_host_type_aliases(program, term))
}

fn parse_expanded_fn_type_expr(
    program: &HostLoweringSession<'_>,
    expr: &Expr,
) -> Option<(Vec<HostTypeTerm>, HostTypeTerm)> {
    let (args, ret) = parse_fn_type_expr_parts(expr)?;
    Some((
        args.iter()
            .map(|argument| decode_expanded_host_type_expr(program, argument))
            .collect::<Option<Vec<_>>>()?,
        decode_expanded_host_type_expr(program, &ret)?,
    ))
}

fn parse_fn_type_expr_parts(expr: &Expr) -> Option<(Vec<Expr>, Expr)> {
    let (DeepTag::TFn, _, kids) = stamped_parts(expr)? else {
        return None;
    };
    let (ret, args) = kids.split_last()?;
    Some((args.to_vec(), ret.clone()))
}

fn decode_host_type_with_subst(
    expr: &Expr,
    subst: &UnordMap<String, HostTypeTerm>,
) -> Result<HostTypeTerm, HostTypeDecodeError> {
    decode_host_type(expr).map(|term| substitute_host_type_term(term, subst))
}

fn substitute_host_type_term(
    term: HostTypeTerm,
    subst: &UnordMap<String, HostTypeTerm>,
) -> HostTypeTerm {
    match term {
        HostTypeTerm::Scalar(HostPrecisionTerm::Variable(name)) => match subst.get(&name) {
            Some(HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision))) => {
                HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(*precision))
            }
            _ => HostTypeTerm::Scalar(HostPrecisionTerm::Variable(name)),
        },
        HostTypeTerm::TypeVariable(name) => subst
            .get(&name)
            .cloned()
            .unwrap_or(HostTypeTerm::TypeVariable(name)),
        HostTypeTerm::Fn(params, ret) => HostTypeTerm::Fn(
            params
                .into_iter()
                .map(|param| substitute_host_type_term(param, subst))
                .collect(),
            Box::new(substitute_host_type_term(*ret, subst)),
        ),
        HostTypeTerm::Adt(name, args) => HostTypeTerm::Adt(
            name,
            args.into_iter()
                .map(|arg| substitute_host_type_term(arg, subst))
                .collect(),
        ),
        HostTypeTerm::List(inner) => {
            HostTypeTerm::List(Box::new(substitute_host_type_term(*inner, subst)))
        }
        HostTypeTerm::Dict(key, value) => HostTypeTerm::Dict(
            Box::new(substitute_host_type_term(*key, subst)),
            Box::new(substitute_host_type_term(*value, subst)),
        ),
        HostTypeTerm::Tuple(items) => HostTypeTerm::Tuple(
            items
                .into_iter()
                .map(|item| substitute_host_type_term(item, subst))
                .collect(),
        ),
        HostTypeTerm::Option(inner) => {
            HostTypeTerm::Option(Box::new(substitute_host_type_term(*inner, subst)))
        }
        HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm { precision, shape }) => {
            let precision = match precision {
                HostPrecisionTerm::Variable(name) => match subst.get(&name) {
                    Some(HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision))) => {
                        HostPrecisionTerm::Concrete(*precision)
                    }
                    _ => HostPrecisionTerm::Variable(name),
                },
                concrete => concrete,
            };
            match (&precision, &shape) {
                (HostPrecisionTerm::Concrete(precision), HostShapeTerm::Concrete(dims)) => {
                    HostTypeTerm::Tensor(TensorType {
                        dims: dims.clone(),
                        precision: *precision,
                    })
                }
                _ => HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm { precision, shape }),
            }
        }
        concrete_or_state => concrete_or_state,
    }
}

fn substitute_host_dimension_terms(
    term: HostTypeTerm,
    subst: &UnordMap<String, DimInfo>,
) -> HostTypeTerm {
    let substitute_dim = |dim: DimInfo| match dim {
        DimInfo::Named(name, _) if subst.contains_key(&name) => subst[&name].clone(),
        other => other,
    };
    match term {
        HostTypeTerm::Fn(params, ret) => HostTypeTerm::Fn(
            params
                .into_iter()
                .map(|param| substitute_host_dimension_terms(param, subst))
                .collect(),
            Box::new(substitute_host_dimension_terms(*ret, subst)),
        ),
        HostTypeTerm::Adt(name, args) => HostTypeTerm::Adt(
            name,
            args.into_iter()
                .map(|argument| substitute_host_dimension_terms(argument, subst))
                .collect(),
        ),
        HostTypeTerm::List(inner) => {
            HostTypeTerm::List(Box::new(substitute_host_dimension_terms(*inner, subst)))
        }
        HostTypeTerm::Dict(key, value) => HostTypeTerm::Dict(
            Box::new(substitute_host_dimension_terms(*key, subst)),
            Box::new(substitute_host_dimension_terms(*value, subst)),
        ),
        HostTypeTerm::Tuple(items) => HostTypeTerm::Tuple(
            items
                .into_iter()
                .map(|item| substitute_host_dimension_terms(item, subst))
                .collect(),
        ),
        HostTypeTerm::Option(inner) => {
            HostTypeTerm::Option(Box::new(substitute_host_dimension_terms(*inner, subst)))
        }
        HostTypeTerm::Tensor(TensorType { dims, precision }) => HostTypeTerm::Tensor(TensorType {
            dims: dims.into_iter().map(substitute_dim).collect(),
            precision,
        }),
        HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm { precision, shape }) => {
            let shape = match shape {
                HostShapeTerm::Concrete(dims) => {
                    HostShapeTerm::Concrete(dims.into_iter().map(substitute_dim).collect())
                }
                HostShapeTerm::Polymorphic(slots) => HostShapeTerm::Polymorphic(
                    slots
                        .into_iter()
                        .map(|slot| match slot {
                            crate::host_type_state::HostShapeSlot::Dim(dim) => {
                                crate::host_type_state::HostShapeSlot::Dim(substitute_dim(dim))
                            }
                            rank => rank,
                        })
                        .collect(),
                ),
            };
            HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm { precision, shape })
        }
        other => other,
    }
}

fn collect_host_type_variable_substitutions(
    pattern: &HostTypeTerm,
    applied: &HostTypeTerm,
    subst: &mut UnordMap<String, HostTypeTerm>,
) {
    match (pattern, applied) {
        (
            HostTypeTerm::Scalar(HostPrecisionTerm::Variable(name)),
            HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision)),
        ) => {
            subst
                .entry(name.clone())
                .or_insert(HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(
                    *precision,
                )));
        }
        (
            HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
                precision: HostPrecisionTerm::Variable(name),
                ..
            }),
            HostTypeTerm::Tensor(tensor),
        ) => {
            subst
                .entry(name.clone())
                .or_insert(HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(
                    tensor.precision,
                )));
        }
        (HostTypeTerm::TypeVariable(name), concrete) if !concrete.is_unresolved() => {
            subst
                .entry(name.clone())
                .or_insert_with(|| concrete.clone());
        }
        (HostTypeTerm::Fn(pattern_params, pattern_ret), HostTypeTerm::Fn(params, ret))
            if pattern_params.len() == params.len() =>
        {
            for (pattern, applied) in pattern_params.iter().zip(params) {
                collect_host_type_variable_substitutions(pattern, applied, subst);
            }
            collect_host_type_variable_substitutions(pattern_ret, ret, subst);
        }
        (HostTypeTerm::Adt(pattern_name, pattern_args), HostTypeTerm::Adt(name, args))
            if terminal_name_matches(pattern_name, name) && pattern_args.len() == args.len() =>
        {
            for (pattern, applied) in pattern_args.iter().zip(args) {
                collect_host_type_variable_substitutions(pattern, applied, subst);
            }
        }
        (HostTypeTerm::List(pattern), HostTypeTerm::List(applied))
        | (HostTypeTerm::Option(pattern), HostTypeTerm::Option(applied)) => {
            collect_host_type_variable_substitutions(pattern, applied, subst);
        }
        (HostTypeTerm::Dict(pattern_key, pattern_value), HostTypeTerm::Dict(key, value)) => {
            collect_host_type_variable_substitutions(pattern_key, key, subst);
            collect_host_type_variable_substitutions(pattern_value, value, subst);
        }
        (HostTypeTerm::Tuple(pattern_items), HostTypeTerm::Tuple(items))
            if pattern_items.len() == items.len() =>
        {
            for (pattern, applied) in pattern_items.iter().zip(items) {
                collect_host_type_variable_substitutions(pattern, applied, subst);
            }
        }
        _ => {}
    }
}

fn decode_host_type_or_raise(expr: &Expr, subst: &UnordMap<String, HostTypeTerm>) -> HostTypeTerm {
    decode_host_type_with_subst(expr, subst).unwrap_or_else(|error| {
        crate::lower::raise_fatal_lowering_diagnostic(
            crate::lower::LowerDiagnostic::new(
                format!("invalid checked host type metadata: {error} ([05-UNS-1]; chelis#730)"),
                Some(expr.span()),
                expr.span_id().map(str::to_string),
            )
            .fatal(),
        )
    })
}

fn option_inner_type(expr: &HostExpr) -> HostTypeTerm {
    match host_expr_type(expr) {
        HostTypeTerm::Option(inner) => (*inner).clone(),
        _ => fresh_host_inference(),
    }
}

fn host_expr_type(expr: &HostExpr) -> HostTypeTerm {
    match &expr.kind {
        HostExprKind::Int(_) => HostTypeTerm::Int64,
        HostExprKind::Float(_) => HostTypeTerm::Float64,
        HostExprKind::Bool(_) => HostTypeTerm::Bool,
        HostExprKind::String(_) => HostTypeTerm::String,
        HostExprKind::List(_, ty) => ty.clone(),
        HostExprKind::Tuple(_, ty) => ty.clone(),
        HostExprKind::Var(_, ty)
        | HostExprKind::Call { ty, .. }
        | HostExprKind::Builtin { ty, .. }
        | HostExprKind::AdtConstruct { ty, .. }
        | HostExprKind::AdtFieldAccess { ty, .. }
        | HostExprKind::If { ty, .. }
        | HostExprKind::MatchOption { ty, .. }
        | HostExprKind::MatchAdt { ty, .. }
        | HostExprKind::Let { ty, .. }
        | HostExprKind::Map { ty, .. }
        | HostExprKind::Filter { ty, .. }
        | HostExprKind::Fold { ty, .. }
        | HostExprKind::Scan { ty, .. }
        | HostExprKind::Partition { ty, .. }
        | HostExprKind::FlatMap { ty, .. }
        | HostExprKind::WithSeed { ty, .. }
        | HostExprKind::TensorCall { ty, .. } => ty.clone(),
        HostExprKind::Unit | HostExprKind::SignatureEntry { .. } => HostTypeTerm::Unit,
    }
}

fn force_host_expr_type(expr: HostExpr, ty: HostTypeTerm) -> HostExpr {
    let HostExpr {
        kind,
        span_id,
        merged_spans,
    } = expr;
    let new_kind = match kind {
        HostExprKind::List(items, _) => {
            let items = match &ty {
                HostTypeTerm::List(inner) => items
                    .into_iter()
                    .map(|item| force_host_expr_type(item, (**inner).clone()))
                    .collect(),
                _ => items,
            };
            HostExprKind::List(items, ty)
        }
        HostExprKind::Tuple(items, _) => {
            let items = match &ty {
                HostTypeTerm::Tuple(expected) if expected.len() == items.len() => items
                    .into_iter()
                    .zip(expected.iter())
                    .map(|(item, expected)| force_host_expr_type(item, expected.clone()))
                    .collect(),
                _ => items,
            };
            HostExprKind::Tuple(items, ty)
        }
        HostExprKind::Var(name, _) => HostExprKind::Var(name, ty),
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ..
        } => {
            let args = args
                .into_iter()
                .enumerate()
                .map(|(index, arg)| {
                    if let Some(expected) = arg_tys.get(index) {
                        force_host_expr_type(arg, expected.clone())
                    } else {
                        arg
                    }
                })
                .collect();
            HostExprKind::Call {
                function,
                args,
                arg_tys,
                ty,
            }
        }
        HostExprKind::Builtin { name, args, .. } => {
            let args = conform_builtin_arguments(&name, args, &ty);
            HostExprKind::Builtin { name, args, ty }
        }
        HostExprKind::AdtConstruct { ctor, fields, .. } => {
            HostExprKind::AdtConstruct { ctor, fields, ty }
        }
        HostExprKind::AdtFieldAccess {
            base, field_index, ..
        } => HostExprKind::AdtFieldAccess {
            base,
            field_index,
            ty,
        },
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => HostExprKind::If {
            cond: Box::new(force_host_expr_type(*cond, HostTypeTerm::Bool)),
            then_expr: Box::new(force_host_expr_type(*then_expr, ty.clone())),
            else_expr: Box::new(force_host_expr_type(*else_expr, ty.clone())),
            ty,
        },
        HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ..
        } => HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr: Box::new(force_host_expr_type(*some_expr, ty.clone())),
            none_expr: Box::new(force_host_expr_type(*none_expr, ty.clone())),
            ty,
        },
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => HostExprKind::MatchAdt {
            scrutinee,
            arms: arms
                .into_iter()
                .map(|mut arm| {
                    arm.expr = force_host_expr_type(arm.expr, ty.clone());
                    arm
                })
                .collect(),
            default_expr: default_expr
                .map(|expr| Box::new(force_host_expr_type(*expr, ty.clone()))),
            ty,
        },
        HostExprKind::Let {
            bindings,
            body,
            ty: actual_ty,
        } => {
            let mut substitutions = UnordMap::new();
            collect_host_type_variable_substitutions(&actual_ty, &ty, &mut substitutions);
            collect_host_type_variable_substitutions(
                &host_expr_type(&body),
                &ty,
                &mut substitutions,
            );
            let bindings = bindings
                .into_iter()
                .map(|binding| {
                    let binding_ty = substitute_host_type_term(binding.ty, &substitutions);
                    HostBinding {
                        value: force_host_expr_type(binding.value, binding_ty.clone()),
                        ty: binding_ty,
                        ..binding
                    }
                })
                .collect();
            HostExprKind::Let {
                bindings,
                body: Box::new(force_host_expr_type(*body, ty.clone())),
                ty,
            }
        }
        HostExprKind::Map { callback, list, .. } => HostExprKind::Map { callback, list, ty },
        HostExprKind::Filter { callback, list, .. } => HostExprKind::Filter { callback, list, ty },
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        } => HostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        },
        HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => HostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        },
        HostExprKind::Partition { callback, list, .. } => {
            HostExprKind::Partition { callback, list, ty }
        }
        HostExprKind::FlatMap { callback, list, .. } => {
            HostExprKind::FlatMap { callback, list, ty }
        }
        HostExprKind::WithSeed { seed, body, .. } => HostExprKind::WithSeed {
            seed: Box::new(force_host_expr_type(*seed, HostTypeTerm::Int64)),
            body: Box::new(force_host_expr_type(*body, ty.clone())),
            ty,
        },
        HostExprKind::TensorCall { helper, args, .. } => {
            HostExprKind::TensorCall { helper, args, ty }
        }
        other => other,
    };
    HostExpr {
        kind: new_kind,
        span_id,
        merged_spans,
    }
}

/// Materialize checked context into every host-program root before term
/// resolution. This is the sole expectation/refinement boundary named by
/// loud_unsupported.md C6.3; codegen therefore receives one resolved tree
/// and has no authority to pick a type when annotations disagree.
fn conform_host_program_types(program: &mut HostProgram) {
    for binding in &mut program.globals {
        binding.value = force_host_expr_type(binding.value.clone(), binding.ty.clone());
    }
    for function in &mut program.functions {
        function.body = force_host_expr_type(function.body.clone(), function.ret_ty.clone());
    }
}

fn conform_builtin_arguments(
    name: &str,
    args: Vec<HostExpr>,
    result_ty: &HostTypeTerm,
) -> Vec<HostExpr> {
    let scalar_numeric = matches!(
        name,
        "add"
            | "sub"
            | "mul"
            | "div"
            | "trunc_div"
            | "floor_div"
            | "mod"
            | "neg"
            | "sqrt"
            | "exp"
            | "log"
            | "sin"
            | "cos"
            | "tanh"
            | "pow"
            | "abs"
            | "min"
            | "max"
    ) && matches!(result_ty, HostTypeTerm::Scalar(_));

    args.into_iter()
        .enumerate()
        .map(|(index, arg)| {
            let expected = match (name, result_ty, index) {
                ("Some", HostTypeTerm::Option(inner), 0) => Some((**inner).clone()),
                ("copy" | "debug", _, 0) => Some(result_ty.clone()),
                ("append", HostTypeTerm::List(_), 0) => Some(result_ty.clone()),
                ("append", HostTypeTerm::List(inner), 1) => Some((**inner).clone()),
                ("dict_of", HostTypeTerm::Dict(key, value), 0) => {
                    Some(HostTypeTerm::List(Box::new(HostTypeTerm::Tuple(vec![
                        (**key).clone(),
                        (**value).clone(),
                    ]))))
                }
                ("to_tensor", HostTypeTerm::Tensor(tensor), 0) => {
                    let mut source =
                        HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(tensor.precision));
                    for _ in &tensor.dims {
                        source = HostTypeTerm::List(Box::new(source));
                    }
                    Some(source)
                }
                ("scalar_to_tensor", HostTypeTerm::Tensor(tensor), 0) => Some(
                    HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(tensor.precision)),
                ),
                _ if scalar_numeric => Some(result_ty.clone()),
                _ => None,
            };
            if let Some(expected) = expected {
                force_host_expr_type(arg, expected)
            } else {
                arg
            }
        })
        .collect()
}

fn infer_builtin_host_type(name: &str, args: &[HostExpr]) -> Option<HostTypeTerm> {
    let arg_tys = args.iter().map(host_expr_type).collect::<Vec<_>>();
    match name {
        "einsum" => {
            let equation = match args.first().map(|e| &e.kind) {
                Some(HostExprKind::String(value)) => value.as_str(),
                _ => return Some(fresh_host_inference()),
            };
            let tensors = args[1..]
                .iter()
                .map(|arg| match host_expr_type(arg) {
                    HostTypeTerm::Tensor(tensor_ty) => Some(tensor_ty),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            infer_einsum_tensor_type(equation, &tensors).map(HostTypeTerm::Tensor)
        }
        "tuple-get" => match (arg_tys.first(), args.get(1).and_then(host_expr_int_literal)) {
            (Some(HostTypeTerm::Tuple(items)), Some(index)) => items
                .get(index as usize)
                .cloned()
                .or(Some(fresh_host_inference())),
            _ => Some(fresh_host_inference()),
        },
        // chelis#631: this lane still sees the axis ARGUMENT (unlike the
        // arg-tys-only fallback), so a literal axis wildcards only the
        // concat axis and keeps the element's other extents.
        "concat" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match inner.as_ref() {
                HostTypeTerm::Tensor(element) => {
                    let axis = args
                        .get(1)
                        .and_then(host_expr_int_literal)
                        .and_then(|raw| normalize_host_axis(element.dims.len(), raw));
                    Some(concat_host_tensor_type(element, axis))
                }
                _ => must_infer_builtin_host_type_from_arg_tys(name, &arg_tys),
            },
            _ => must_infer_builtin_host_type_from_arg_tys(name, &arg_tys),
        },
        _ => must_infer_builtin_host_type_from_arg_tys(name, &arg_tys),
    }
}

/// A builtin application reached host-result inference with an argument
/// shape that cannot satisfy that builtin's typed contract.
///
/// This is deliberately not represented by an inference variable: invalid
/// input and valid-but-underconstrained input are different states.  The
/// former is a typed lowering failure; only the latter may allocate a fresh
/// [`HostInferenceVar`].
#[derive(Debug, Clone, PartialEq)]
struct HostBuiltinInferenceError {
    builtin: String,
    argument_types: Vec<HostTypeTerm>,
    requirement: String,
}

impl fmt::Display for HostBuiltinInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid host result inference for builtin `{}` with arguments {:?}: {} \
             (chelis#730; [05-UNS-1])",
            self.builtin, self.argument_types, self.requirement
        )
    }
}

impl std::error::Error for HostBuiltinInferenceError {}

fn invalid_builtin_inference(
    name: &str,
    arg_tys: &[HostTypeTerm],
    requirement: impl Into<String>,
) -> HostBuiltinInferenceError {
    HostBuiltinInferenceError {
        builtin: name.to_string(),
        argument_types: arg_tys.to_vec(),
        requirement: requirement.into(),
    }
}

fn term_matches_or_is_unresolved(
    ty: &HostTypeTerm,
    predicate: impl FnOnce(&HostTypeTerm) -> bool,
) -> bool {
    ty.is_unresolved() || predicate(ty)
}

/// Infer a builtin result without laundering invalid input into an anonymous
/// type hole.  A successful unresolved result means inference genuinely
/// needs context (for example arg-types-only `tuple-get` cannot see its index).
fn infer_builtin_host_type_from_arg_tys(
    name: &str,
    arg_tys: &[HostTypeTerm],
) -> Result<Option<HostTypeTerm>, HostBuiltinInferenceError> {
    let require_first = |description: &str,
                         predicate: fn(&HostTypeTerm) -> bool|
     -> Result<(), HostBuiltinInferenceError> {
        match arg_tys.first() {
            Some(ty) if term_matches_or_is_unresolved(ty, predicate) => Ok(()),
            _ => Err(invalid_builtin_inference(
                name,
                arg_tys,
                format!("first argument must be {description}"),
            )),
        }
    };
    let is_tensor = |ty: &HostTypeTerm| matches!(ty, HostTypeTerm::Tensor(_));
    let is_list = |ty: &HostTypeTerm| matches!(ty, HostTypeTerm::List(_));
    let is_dict = |ty: &HostTypeTerm| matches!(ty, HostTypeTerm::Dict(_, _));

    match name {
        "sum" | "mean" | "max_reduce" | "min_reduce" | "prod_reduce" | "count"
        | "argmax_reduce" | "argmin_reduce" | "reshape" | "expand" | "insert" | "pad"
        | "shrink" | "stride" | "permute" | "split" | "sort" | "tensor_to_scalar" | "to_list" => {
            require_first("a tensor", is_tensor)?;
            if name == "to_list"
                && matches!(
                    arg_tys.first(),
                    Some(HostTypeTerm::Tensor(TensorType {
                        precision: Prim::F8e4m3,
                        ..
                    }))
                )
            {
                return Err(invalid_builtin_inference(
                    name,
                    arg_tys,
                    "f8e4m3 is deferred by spec/04-type-system.md section 1.1.1",
                ));
            }
        }
        "index" | "take" | "chunk" | "flatten" | "enumerate" | "dict_of" | "to_tensor"
        | "pad_sequences" | "pad_sequences_to" => {
            require_first("a list", is_list)?;
        }
        "drop" if arg_tys.len() != 1 => require_first("a list", is_list)?,
        "concat" => {
            require_first(
                "a list of tensors",
                |ty| matches!(ty, HostTypeTerm::List(inner) if matches!(inner.as_ref(), HostTypeTerm::Tensor(_))),
            )?;
            if arg_tys.get(1).is_none() {
                return Err(invalid_builtin_inference(
                    name,
                    arg_tys,
                    "concat requires an axis argument",
                ));
            }
        }
        "zip" | "dict_merge" => {
            require_first(
                if name == "zip" { "a list" } else { "a dict" },
                if name == "zip" { is_list } else { is_dict },
            )?;
            let second_ok = arg_tys.get(1).is_some_and(|ty| {
                term_matches_or_is_unresolved(ty, if name == "zip" { is_list } else { is_dict })
            });
            if !second_ok {
                return Err(invalid_builtin_inference(
                    name,
                    arg_tys,
                    if name == "zip" {
                        "second argument must be a list"
                    } else {
                        "second argument must be a dict"
                    },
                ));
            }
        }
        "dict_get" | "dict_remove" | "dict_insert" | "dict_keys" | "dict_values"
        | "dict_entries" => require_first("a dict", is_dict)?,
        "map" | "filter" | "fold" | "scan" | "partition" | "flat_map" => {
            let list_index = if matches!(name, "fold" | "scan") {
                2
            } else {
                1
            };
            if !arg_tys
                .get(list_index)
                .is_some_and(|ty| term_matches_or_is_unresolved(ty, is_list))
            {
                return Err(invalid_builtin_inference(
                    name,
                    arg_tys,
                    format!("argument {list_index} must be a list"),
                ));
            }
        }
        "scalar_to_tensor" => {
            require_first("a scalar", |ty| matches!(ty, HostTypeTerm::Scalar(_)))?
        }
        _ => {}
    }

    Ok(infer_builtin_host_type_from_arg_tys_unchecked(
        name, arg_tys,
    ))
}

fn must_infer_builtin_host_type_from_arg_tys(
    name: &str,
    arg_tys: &[HostTypeTerm],
) -> Option<HostTypeTerm> {
    infer_builtin_host_type_from_arg_tys(name, arg_tys).unwrap_or_else(|error| panic!("{error}"))
}

fn infer_builtin_host_type_from_arg_tys_unchecked(
    name: &str,
    arg_tys: &[HostTypeTerm],
) -> Option<HostTypeTerm> {
    let tensor_arg = arg_tys.iter().find_map(|ty| match ty {
        HostTypeTerm::Tensor(tensor_ty) => Some(tensor_ty.clone()),
        _ => None,
    });
    match name {
        "add" | "sub" | "mul" | "div" | "floor_div" | "trunc_div" | "neg" | "exp" | "log"
        | "sin" | "sqrt" | "relu" | "sigmoid" | "tanh" | "silu" | "gelu" | "max_elem"
        | "min_elem" | "copy" | "uniform_like" | "dropout" | "softmax" => {
            if let Some(tensor_ty) = tensor_arg {
                Some(HostTypeTerm::Tensor(tensor_ty))
            } else if arg_tys
                .iter()
                .any(|ty| matches!(ty, &HostTypeTerm::Float64))
            {
                Some(HostTypeTerm::Float64)
            } else if arg_tys
                .iter()
                .any(|ty| matches!(ty, &HostTypeTerm::Float32))
            {
                // WS-4: an f32 scalar operand keeps the result in f32; only
                // a genuine f64 operand widens the result. Falling through
                // to `Int64` here would mis-type an all-f32 scalar
                // arithmetic result as an integer.
                Some(HostTypeTerm::Float32)
            } else if arg_tys.iter().any(HostTypeTerm::is_unresolved) {
                // chelis#730 Phase 1 (census row 7, chelis#714/#718): an
                // Unknown-typed operand (an f16/bf16/i8/i16 scalar with
                // no host representation) must not silently type the result
                // as Int64 - propagate the Unknown so the C emitter's
                // baking-point guard rejects loudly instead of emitting
                // int64_t arithmetic over garbage.
                Some(fresh_host_inference())
            } else {
                Some(HostTypeTerm::Int64)
            }
        }
        // chelis#340: the named-axis reduction family classifies as a
        // tensor in the coarse host lane (the real output shape — the
        // reduced axis dropped — is recomputed inside the tensor-helper
        // DAG; this coarse type only needs to keep the result classified as
        // a Tensor so the call routes through the tensor-DAG kernel lane).
        // `max_reduce`/`min_reduce`/`prod_reduce` keep the operand
        // precision; `argmax_reduce`/`argmin_reduce` return an i64 index
        // tensor.
        "sum" | "mean" | "max_reduce" | "min_reduce" | "prod_reduce" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tensor(tensor_ty.clone())),
            _ => Some(fresh_host_inference()),
        },
        "count" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tensor(TensorType {
                dims: tensor_ty.dims.clone(),
                precision: Prim::Int64,
            })),
            _ => Some(fresh_host_inference()),
        },
        "argmax_reduce" | "argmin_reduce" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tensor(TensorType {
                dims: tensor_ty.dims.clone(),
                precision: chelis_types::types::Prim::Int64,
            })),
            _ => Some(fresh_host_inference()),
        },
        "mod" | "bitand" | "bitor" | "bitxor" | "shl" | "shr" | "char_code" | "string_len"
        | "rank" | "shape" | "numel" => Some(HostTypeTerm::Int64),
        "cmplt" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tensor(TensorType {
                dims: tensor_ty.dims.clone(),
                precision: chelis_types::types::Prim::Bool,
            })),
            _ => Some(HostTypeTerm::Bool),
        },
        // Movement ops preserve the element precision and stay tensors.
        // The concrete output shape (e.g. `expand`'s broadcast axis) is
        // recomputed inside the tensor-helper DAG; the host `HostTypeTerm`
        // is coarse (precision + a dims placeholder used only for the
        // typed-runtime dispatch), so propagating the *input* tensor
        // type here is sufficient to keep the result classified as a
        // tensor. Without these arms `expand`/`pad`/`shrink`/`stride`/
        // `permute` fell through to `None`, so a host-lane `let k =
        // expand(scalar_to_tensor(c), 0, n)` binding lost its tensor
        // type and was emitted as `void* k = /* unsupported builtin
        // expand */ 0`, then mistyped as a scalar at the consuming
        // tensor-helper callsite (`(float)(void* k)`, issue #300).
        "reshape" | "expand" | "insert" | "pad" | "shrink" | "stride" | "permute" => match arg_tys
            .first()
        {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tensor(tensor_ty.clone())),
            _ => Some(fresh_host_inference()),
        },
        "lt" | "gt" | "gte" | "lte" | "eq" | "neq" | "and" | "or" | "not" | "string_contains"
        | "string_starts_with" | "string_ends_with" => Some(HostTypeTerm::Bool),
        "char_from_code" | "string_concat" | "string_trim" | "string_slice" | "to_string" => {
            Some(HostTypeTerm::String)
        }
        "to_int" => Some(HostTypeTerm::Option(Box::new(HostTypeTerm::Int64))),
        "to_float" => Some(HostTypeTerm::Option(Box::new(HostTypeTerm::Float64))),
        "len" => Some(HostTypeTerm::Int64),
        "index" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => Some((**inner).clone()),
            _ => Some(fresh_host_inference()),
        },
        "append" => arg_tys.first().cloned(),
        // chelis#631: this arg-tys-only lane cannot see the axis VALUE, so
        // every axis wildcards — returning the element type VERBATIM baked
        // the element's concat-axis extent into tensor-helper signatures
        // (the host-lane copy of the checker's pre-#631 concat bug).
        "concat" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(HostTypeTerm::List(inner)), Some(&HostTypeTerm::Int64))
                if matches!(inner.as_ref(), HostTypeTerm::Tensor(_)) =>
            {
                match inner.as_ref() {
                    HostTypeTerm::Tensor(element) => Some(concat_host_tensor_type(element, None)),
                    _ => Some(fresh_host_inference()),
                }
            }
            (Some(lhs), Some(_)) => Some(lhs.clone()),
            _ => Some(fresh_host_inference()),
        },
        "split" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::List(Box::new(
                HostTypeTerm::Tensor(tensor_ty.clone()),
            ))),
            _ => Some(fresh_host_inference()),
        },
        "scatter" | "scatter_replace" | "scatter_elements" | "where" | "cumsum" | "diagonal"
        | "trace" | "clamp" => arg_tys.first().cloned(),
        "sort" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tuple(vec![
                HostTypeTerm::Tensor(tensor_ty.clone()),
                HostTypeTerm::Tensor(TensorType {
                    dims: tensor_ty.dims.clone(),
                    precision: chelis_types::types::Prim::Int64,
                }),
            ])),
            _ => Some(fresh_host_inference()),
        },
        "tuple-get" => Some(fresh_host_inference()),
        "drop" if arg_tys.len() == 1 => Some(HostTypeTerm::Unit),
        "take" | "drop" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => {
                Some(HostTypeTerm::List(Box::new((**inner).clone())))
            }
            _ => Some(fresh_host_inference()),
        },
        "chunk" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => Some(HostTypeTerm::List(Box::new(
                HostTypeTerm::List(Box::new((**inner).clone())),
            ))),
            _ => Some(fresh_host_inference()),
        },
        "range" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::Int64))),
        "map" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(_), Some(HostTypeTerm::List(inner))) => {
                Some(HostTypeTerm::List(Box::new((**inner).clone())))
            }
            _ => Some(fresh_host_inference()),
        },
        "filter" => arg_tys.get(1).cloned(),
        "fold" => arg_tys.get(1).cloned(),
        "scan" => match arg_tys.get(1) {
            Some(init_ty) => Some(HostTypeTerm::List(Box::new(init_ty.clone()))),
            None => Some(fresh_host_inference()),
        },
        // Issue #257: `tensor_scan(initial: T, fn: (T, i64) -> T, n: i64) -> tensor[n, T]`.
        //
        // We deliberately do NOT synthesize a concrete `HostTypeTerm::Tensor`
        // precision here. `HostTypeTerm` is a *coarse* host-IR class: every
        // integer width collapses to `Int64` and every float width to
        // `Float64` (see `host_type_from_tensor_input`), so by the time
        // the initial value's type reaches this arm its real dtype
        // (`i8`..`i64`, `f16`..`f64`) is already gone. Any concrete
        // precision we picked would be a guess — e.g. an earlier version
        // mapped `Float64 -> F32`, which is wrong for an `f64` initial,
        // and `Int64` for an `i32` initial. The authoritative element
        // type lives in the real type checker (`chelis-types`
        // `infer.rs::infer_app` "tensor_scan" arm) and in the runtime,
        // which reads the precision straight off the initial value.
        //
        // Returning `Unknown` is sound because this host type is *never
        // consumed*: `tensor_scan` is host-only, so any compiled-backend
        // program that reaches it is rejected up front by
        // `compiler.rs::reject_host_only_builtins` (which keys off the
        // `Builtin` node, not its inferred type) before the C/HIP emitter
        // runs, and the `chelis eval`/`chelis test` interpreter never
        // consults host-IR types at all. Emitting `Unknown` keeps this
        // arm from advertising a precision it cannot actually know.
        "tensor_scan" => Some(fresh_host_inference()),
        "partition" => match arg_tys.get(1) {
            Some(list_ty) => Some(HostTypeTerm::Tuple(vec![list_ty.clone(), list_ty.clone()])),
            None => Some(fresh_host_inference()),
        },
        "flat_map" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(_), Some(HostTypeTerm::List(_))) => match arg_tys.first() {
                Some(ty) if ty.is_unresolved() => Some(fresh_host_inference()),
                Some(_) => None,
                None => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "flatten" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match inner.as_ref() {
                HostTypeTerm::List(nested) => {
                    Some(HostTypeTerm::List(Box::new((**nested).clone())))
                }
                _ => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "zip" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(HostTypeTerm::List(lhs)), Some(HostTypeTerm::List(rhs))) => {
                Some(HostTypeTerm::List(Box::new(HostTypeTerm::Tuple(vec![
                    (**lhs).clone(),
                    (**rhs).clone(),
                ]))))
            }
            _ => Some(fresh_host_inference()),
        },
        "enumerate" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => {
                Some(HostTypeTerm::List(Box::new(HostTypeTerm::Tuple(vec![
                    HostTypeTerm::Int64,
                    (**inner).clone(),
                ]))))
            }
            _ => Some(fresh_host_inference()),
        },
        "dict_of" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match &**inner {
                HostTypeTerm::Tuple(parts) if parts.len() == 2 => Some(HostTypeTerm::Dict(
                    Box::new(parts[0].clone()),
                    Box::new(parts[1].clone()),
                )),
                _ => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "dict_get" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(_, value)) => {
                Some(HostTypeTerm::Option(Box::new((**value).clone())))
            }
            _ => Some(fresh_host_inference()),
        },
        "dict_contains" => Some(HostTypeTerm::Bool),
        "dict_remove" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(key, value)) => Some(HostTypeTerm::Dict(
                Box::new((**key).clone()),
                Box::new((**value).clone()),
            )),
            _ => Some(fresh_host_inference()),
        },
        "dict_insert" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(key, value)) => Some(HostTypeTerm::Dict(
                Box::new((**key).clone()),
                Box::new((**value).clone()),
            )),
            _ => Some(fresh_host_inference()),
        },
        "dict_merge" => match (arg_tys.first(), arg_tys.get(1)) {
            (
                Some(HostTypeTerm::Dict(lhs_key, lhs_value)),
                Some(HostTypeTerm::Dict(rhs_key, rhs_value)),
            ) if **lhs_key == **rhs_key && **lhs_value == **rhs_value => Some(HostTypeTerm::Dict(
                Box::new((**lhs_key).clone()),
                Box::new((**lhs_value).clone()),
            )),
            _ => Some(fresh_host_inference()),
        },
        "dict_keys" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(key, _)) => Some(HostTypeTerm::List(Box::new((**key).clone()))),
            _ => Some(fresh_host_inference()),
        },
        "dict_values" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(_, value)) => {
                Some(HostTypeTerm::List(Box::new((**value).clone())))
            }
            _ => Some(fresh_host_inference()),
        },
        "dict_entries" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(key, value)) => {
                Some(HostTypeTerm::List(Box::new(HostTypeTerm::Tuple(vec![
                    (**key).clone(),
                    (**value).clone(),
                ]))))
            }
            _ => Some(fresh_host_inference()),
        },
        "print" => Some(HostTypeTerm::Unit),
        "fail" => Some(HostTypeTerm::Never),
        "debug" => arg_tys.first().cloned(),
        "tensor_to_scalar" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor)) => Some(match tensor.precision {
                chelis_types::types::Prim::Bool => HostTypeTerm::Bool,
                chelis_types::types::Prim::Int8
                | chelis_types::types::Prim::Int32
                | chelis_types::types::Prim::Int64 => HostTypeTerm::Int64,
                _ => HostTypeTerm::Float64,
            }),
            _ => Some(HostTypeTerm::Float64),
        },
        // Issue #308: `HostTypeTerm::Float64` classifies BOTH f32 and f64
        // host scalars, so this coarse arm cannot recover the true
        // operand precision and defaults the float case to f32 (the
        // IR `Const` default for unsuffixed float literals). The
        // Deep-level `infer_app_expr_host_type` `scalar_to_tensor` arm
        // overrides this with the real precision (checker annotation
        // or explicit `cast(_, p)` target) whenever the Deep operand
        // is still visible; this fallback only decides when nothing
        // upstream knew better, and then both the C emit dispatch and
        // the consuming tensor-helper Load share the same f32 answer,
        // so storage width stays consistent.
        "scalar_to_tensor" => match arg_tys.first() {
            Some(&HostTypeTerm::Int64) => Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::Int64,
            })),
            Some(&HostTypeTerm::Bool) => Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::Bool,
            })),
            Some(&HostTypeTerm::Float64) => Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::F32,
            })),
            _ => None,
        },
        "to_tensor" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match &**inner {
                HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision))
                    if *precision != chelis_types::types::Prim::String =>
                {
                    Some(HostTypeTerm::Tensor(TensorType {
                        dims: vec![crate::dag::DimInfo::Named("list".to_string(), None)],
                        precision: *precision,
                    }))
                }
                HostTypeTerm::Scalar(HostPrecisionTerm::Variable(_)) => {
                    Some(fresh_host_inference())
                }
                _ => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "to_list" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor)) if tensor.dims.len() == 1 => {
                let element_ty = match tensor.precision {
                    chelis_types::types::Prim::Bool => HostTypeTerm::Bool,
                    chelis_types::types::Prim::Int8
                    | chelis_types::types::Prim::Int16
                    | chelis_types::types::Prim::Int32
                    | chelis_types::types::Prim::Int64 => HostTypeTerm::Int64,
                    chelis_types::types::Prim::F16
                    | chelis_types::types::Prim::Bf16
                    | chelis_types::types::Prim::F32
                    | chelis_types::types::Prim::F64 => HostTypeTerm::Float64,
                    // The checked wrapper rejects this deferred dtype with
                    // `HostBuiltinInferenceError` before entering this
                    // implementation. Keep this arm non-defaulting if the
                    // match is edited independently in the future.
                    chelis_types::types::Prim::F8e4m3 => return None,
                    _ => fresh_host_inference(),
                };
                Some(HostTypeTerm::List(Box::new(element_ty)))
            }
            Some(HostTypeTerm::Tensor(_)) => Some(fresh_host_inference()),
            _ => Some(fresh_host_inference()),
        },
        "pad_sequences" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match &**inner {
                HostTypeTerm::List(nested) => match **nested {
                    HostTypeTerm::Int64 => Some(HostTypeTerm::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::Int64,
                    })),
                    HostTypeTerm::Float64 => Some(HostTypeTerm::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::F32,
                    })),
                    _ => Some(fresh_host_inference()),
                },
                _ => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "pad_sequences_to" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match &**inner {
                HostTypeTerm::List(nested) => match **nested {
                    HostTypeTerm::Int64 => Some(HostTypeTerm::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::Int64,
                    })),
                    HostTypeTerm::Float64 => Some(HostTypeTerm::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::F32,
                    })),
                    _ => Some(fresh_host_inference()),
                },
                _ => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "read_file" => Some(HostTypeTerm::String),
        "write_file" => Some(HostTypeTerm::Unit),
        "read_lines" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::String))),
        "read_bytes" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::Int64))),
        "file_exists" => Some(HostTypeTerm::Bool),
        "list_dir" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::String))),
        "mmap_file" => Some(HostTypeTerm::MappedFile),
        "mmap_read" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::Int64))),
        "mmap_len" => Some(HostTypeTerm::Int64),
        // Hull Phase 0a: `process_run(cmd, args) -> (exit_code, stdout, stderr)`.
        // Eval/test-only; the C/HIP build backends reject it before codegen
        // (see `host_program_uses_builtin` / `reject_eval_only_builtins_host`).
        "process_run" => Some(HostTypeTerm::Tuple(vec![
            HostTypeTerm::Int64,
            HostTypeTerm::String,
            HostTypeTerm::String,
        ])),
        "parse_csv" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::Dict(
            Box::new(HostTypeTerm::String),
            Box::new(HostTypeTerm::String),
        )))),
        "to_csv" | "csv_str" => Some(HostTypeTerm::String),
        "csv_f64" => Some(HostTypeTerm::Float64),
        "csv_int" | "csv_nrows" => Some(HostTypeTerm::Int64),
        "csv_f64s" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::Float64))),
        "csv_ints" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::Int64))),
        "csv_strs" | "csv_cols" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::String))),
        // `round_to` preserves its operand's float dtype ([05-OP-1]: f64 or
        // f32, decided by the checker); an unresolved operand stays an
        // inference hole rather than advertising a width this table cannot
        // know.
        "round_to" => match arg_tys.first() {
            Some(
                term @ HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(
                    chelis_types::types::Prim::F64 | chelis_types::types::Prim::F32,
                )),
            ) => Some(term.clone()),
            _ => Some(fresh_host_inference()),
        },
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct GenericAdtConstructor {
    adt_name: String,
    ctor_name: String,
    parameters: Vec<String>,
    stored_parameters: Vec<bool>,
    fields: Vec<GenericAdtField>,
}

#[derive(Debug, Clone)]
struct GenericAdtField {
    name: Option<String>,
    term: HostTypeTerm,
}

#[derive(Debug, Clone)]
struct InstantiatedAdtConstructor {
    ty: HostTypeTerm,
    fields: Vec<HostAdtField>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AdtInstantiationError {
    MissingAppliedType {
        adt: String,
    },
    WrongAppliedType {
        expected: String,
        got: String,
    },
    Arity {
        adt: String,
        expected: usize,
        got: usize,
    },
    UnresolvedArgument {
        adt: String,
        index: usize,
    },
    UnresolvedField {
        adt: String,
        field: String,
    },
}

impl fmt::Display for AdtInstantiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingAppliedType { adt } => {
                write!(f, "generic ADT `{adt}` has no applied type arguments")
            }
            Self::WrongAppliedType { expected, got } => {
                write!(f, "constructor for `{expected}` was applied as `{got}`")
            }
            Self::Arity { adt, expected, got } => write!(
                f,
                "generic ADT `{adt}` expects {expected} type arguments, got {got}"
            ),
            Self::UnresolvedArgument { adt, index } => {
                write!(f, "generic ADT `{adt}` type argument {index} is unresolved")
            }
            Self::UnresolvedField { adt, field } => write!(
                f,
                "generic ADT `{adt}` field `{field}` remains unresolved after substitution"
            ),
        }
    }
}

impl GenericAdtConstructor {
    fn is_nullary(&self) -> bool {
        self.fields.is_empty()
    }

    fn instantiate(
        &self,
        instantiated_ty: &HostTypeTerm,
    ) -> Result<InstantiatedAdtConstructor, AdtInstantiationError> {
        let (ty, arguments) = match instantiated_ty {
            HostTypeTerm::Adt(name, arguments) => {
                if !terminal_name_matches(name, &self.adt_name) {
                    return Err(AdtInstantiationError::WrongAppliedType {
                        expected: self.adt_name.clone(),
                        got: name.clone(),
                    });
                }
                (instantiated_ty.clone(), arguments.as_slice())
            }
            _ if self.parameters.is_empty() => (
                HostTypeTerm::Adt(self.adt_name.clone(), Vec::new()),
                &[][..],
            ),
            _ => {
                return Err(AdtInstantiationError::MissingAppliedType {
                    adt: self.adt_name.clone(),
                });
            }
        };
        if arguments.len() != self.parameters.len() {
            return Err(AdtInstantiationError::Arity {
                adt: self.adt_name.clone(),
                expected: self.parameters.len(),
                got: arguments.len(),
            });
        }
        if let Some((index, _)) = arguments
            .iter()
            .enumerate()
            .find(|(_, argument)| argument.is_unresolved())
        {
            return Err(AdtInstantiationError::UnresolvedArgument {
                adt: self.adt_name.clone(),
                index,
            });
        }
        let substitutions = self
            .parameters
            .iter()
            .cloned()
            .zip(arguments.iter().cloned())
            .collect();
        let fields = self
            .fields
            .iter()
            .map(|field| {
                let ty = substitute_host_type_term(field.term.clone(), &substitutions);
                if ty.is_unresolved() {
                    return Err(AdtInstantiationError::UnresolvedField {
                        adt: self.adt_name.clone(),
                        field: field
                            .name
                            .clone()
                            .unwrap_or_else(|| "<positional>".to_string()),
                    });
                }
                Ok(HostAdtField {
                    name: field.name.clone(),
                    ty,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(InstantiatedAdtConstructor { ty, fields })
    }

    /// A nullary constructor has no field layout to instantiate yet, so it
    /// may retain the declaration's named type arguments until the sole
    /// checked-expectation boundary (`conform_host_program_types`) supplies
    /// a concrete enclosing type. Wrong ADT names and arities still reject
    /// immediately; only genuinely absent/unresolved arguments survive.
    fn instantiate_nullary_term(
        &self,
        instantiated_ty: &HostTypeTerm,
    ) -> Result<InstantiatedAdtConstructor, AdtInstantiationError> {
        debug_assert!(self.is_nullary());
        match self.instantiate(instantiated_ty) {
            Ok(instantiated) => Ok(instantiated),
            Err(AdtInstantiationError::MissingAppliedType { .. })
                if instantiated_ty.is_unresolved() =>
            {
                Ok(InstantiatedAdtConstructor {
                    ty: HostTypeTerm::Adt(
                        self.adt_name.clone(),
                        self.parameters
                            .iter()
                            .cloned()
                            .map(HostTypeTerm::TypeVariable)
                            .collect(),
                    ),
                    fields: Vec::new(),
                })
            }
            Err(AdtInstantiationError::UnresolvedArgument { .. }) => {
                Ok(InstantiatedAdtConstructor {
                    ty: instantiated_ty.clone(),
                    fields: Vec::new(),
                })
            }
            Err(error) => Err(error),
        }
    }
}

fn adt_constructor_definitions(program: &HostLoweringSession<'_>) -> Vec<GenericAdtConstructor> {
    let mut definitions = Vec::new();
    for definition in program.adt_registry().defs.values() {
        // These prelude declarations have dedicated host representations and
        // constructor lowering. A composed checker registry includes them,
        // unlike the old source-local scan, so keep them out of the generic
        // ADT constructor table.
        if matches!(
            definition.name.as_str(),
            "Option" | "List" | "Dict" | "MappedFile"
        ) {
            continue;
        }
        let checker_parameter_names = definition
            .param_args
            .iter()
            .zip(&definition.type_params)
            .filter_map(|(argument, name)| match argument {
                NominalArg::Type(Type::Var(variable)) => {
                    Some((format!("t{}", variable.0), name.clone()))
                }
                NominalArg::Dimension(Dim::Var(variable)) => {
                    Some((format!("d{}", variable.0), name.clone()))
                }
                _ => None,
            })
            .collect::<UnordMap<_, _>>();
        let parameters = definition.type_params.clone();
        let stored_parameters = definition
            .type_params
            .iter()
            .enumerate()
            .map(|(index, _)| {
                checker_adt_parameter_is_stored(
                    definition,
                    index,
                    program.adt_registry(),
                    &mut UnordSet::new(),
                )
            })
            .collect::<Vec<_>>();
        for variant in &definition.variants {
            let fields = variant
                .fields
                .iter()
                .map(|(name, ty)| GenericAdtField {
                    name: name.clone(),
                    term: rename_host_type_variables(
                        decode_host_type_or_raise(&type_to_deep_expr(ty), &UnordMap::new()),
                        &checker_parameter_names,
                    ),
                })
                .collect();
            definitions.push(GenericAdtConstructor {
                adt_name: definition.name.clone(),
                ctor_name: variant.name.clone(),
                parameters: parameters.clone(),
                stored_parameters: stored_parameters.clone(),
                fields,
            });
        }
    }
    definitions.sort_by(|left, right| {
        (&left.adt_name, &left.ctor_name).cmp(&(&right.adt_name, &right.ctor_name))
    });
    definitions
}

fn rename_host_type_variables(
    term: HostTypeTerm,
    names: &UnordMap<String, String>,
) -> HostTypeTerm {
    let rename_dim = |dim: DimInfo| match dim {
        DimInfo::Named(name, extent) => {
            DimInfo::Named(names.get(&name).cloned().unwrap_or(name), extent)
        }
        literal => literal,
    };
    match term {
        HostTypeTerm::TypeVariable(name) => {
            HostTypeTerm::TypeVariable(names.get(&name).cloned().unwrap_or(name))
        }
        HostTypeTerm::Scalar(HostPrecisionTerm::Variable(name)) => HostTypeTerm::Scalar(
            HostPrecisionTerm::Variable(names.get(&name).cloned().unwrap_or(name)),
        ),
        HostTypeTerm::Fn(params, ret) => HostTypeTerm::Fn(
            params
                .into_iter()
                .map(|param| rename_host_type_variables(param, names))
                .collect(),
            Box::new(rename_host_type_variables(*ret, names)),
        ),
        HostTypeTerm::Adt(name, args) => HostTypeTerm::Adt(
            name,
            args.into_iter()
                .map(|arg| rename_host_type_variables(arg, names))
                .collect(),
        ),
        HostTypeTerm::List(inner) => {
            HostTypeTerm::List(Box::new(rename_host_type_variables(*inner, names)))
        }
        HostTypeTerm::Dict(key, value) => HostTypeTerm::Dict(
            Box::new(rename_host_type_variables(*key, names)),
            Box::new(rename_host_type_variables(*value, names)),
        ),
        HostTypeTerm::Tuple(items) => HostTypeTerm::Tuple(
            items
                .into_iter()
                .map(|item| rename_host_type_variables(item, names))
                .collect(),
        ),
        HostTypeTerm::Option(inner) => {
            HostTypeTerm::Option(Box::new(rename_host_type_variables(*inner, names)))
        }
        HostTypeTerm::Tensor(TensorType { dims, precision }) => HostTypeTerm::Tensor(TensorType {
            dims: dims.into_iter().map(rename_dim).collect(),
            precision,
        }),
        HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm { precision, shape }) => {
            let precision = match precision {
                HostPrecisionTerm::Variable(name) => {
                    HostPrecisionTerm::Variable(names.get(&name).cloned().unwrap_or(name))
                }
                concrete => concrete,
            };
            let shape = match shape {
                HostShapeTerm::Concrete(dims) => {
                    HostShapeTerm::Concrete(dims.into_iter().map(rename_dim).collect())
                }
                HostShapeTerm::Polymorphic(slots) => HostShapeTerm::Polymorphic(
                    slots
                        .into_iter()
                        .map(|slot| match slot {
                            crate::host_type_state::HostShapeSlot::Dim(dim) => {
                                crate::host_type_state::HostShapeSlot::Dim(rename_dim(dim))
                            }
                            crate::host_type_state::HostShapeSlot::RankVariable(name) => {
                                crate::host_type_state::HostShapeSlot::RankVariable(
                                    names.get(&name).cloned().unwrap_or(name),
                                )
                            }
                        })
                        .collect(),
                ),
            };
            HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm { precision, shape })
        }
        other => other,
    }
}

fn checker_adt_parameter_is_stored(
    definition: &AdtDef,
    parameter_index: usize,
    registry: &AdtRegistry,
    visiting: &mut UnordSet<(String, usize)>,
) -> bool {
    let Some(argument) = definition.param_args.get(parameter_index) else {
        return true;
    };
    let NominalArg::Type(Type::Var(parameter)) = argument else {
        return false;
    };
    let key = (definition.name.clone(), parameter_index);
    if !visiting.insert(key.clone()) {
        return false;
    }
    let stored = definition.variants.iter().any(|variant| {
        variant
            .fields
            .iter()
            .any(|(_, field)| checker_type_stores_variable(field, *parameter, registry, visiting))
    });
    visiting.remove(&key);
    stored
}

fn checker_type_stores_variable(
    ty: &Type,
    target: chelis_types::types::TypeVar,
    registry: &AdtRegistry,
    visiting: &mut UnordSet<(String, usize)>,
) -> bool {
    match ty {
        Type::Var(variable) => *variable == target,
        Type::Tensor(_, TensorPrec::Var(variable)) => *variable == target,
        Type::Tensor(_, TensorPrec::Concrete(_)) | Type::Prim(_) | Type::Unit => false,
        Type::Fn(params, ret) => {
            params
                .iter()
                .any(|ty| checker_type_stores_variable(ty, target, registry, visiting))
                || checker_type_stores_variable(ret, target, registry, visiting)
        }
        Type::Ref(inner) => checker_type_stores_variable(inner, target, registry, visiting),
        Type::Tuple(items) => items
            .iter()
            .any(|ty| checker_type_stores_variable(ty, target, registry, visiting)),
        Type::Adt(name, args) => {
            if let Some(nested) = registry.lookup(name) {
                args.iter().enumerate().any(|(index, arg)| {
                    checker_adt_parameter_is_stored(nested, index, registry, visiting)
                        && checker_type_stores_variable(arg, target, registry, visiting)
                })
            } else {
                // Built-in/container ADTs have value-represented arguments.
                args.iter()
                    .any(|ty| checker_type_stores_variable(ty, target, registry, visiting))
            }
        }
        Type::KindedAdt(name, args) => {
            if let Some(nested) = registry.lookup(name) {
                args.iter()
                    .enumerate()
                    .any(|(index, argument)| match argument {
                        NominalArg::Type(ty) => {
                            checker_adt_parameter_is_stored(nested, index, registry, visiting)
                                && checker_type_stores_variable(ty, target, registry, visiting)
                        }
                        NominalArg::Dimension(_) => false,
                    })
            } else {
                args.iter().any(|argument| {
                    argument.as_type().is_some_and(|ty| {
                        checker_type_stores_variable(ty, target, registry, visiting)
                    })
                })
            }
        }
        // A checked program cannot carry Error, but fail closed if a legacy
        // deserialized artifact violates that invariant.
        Type::Error(_) => true,
    }
}

/// Canonicalize only representation-erased generic ADT arguments before
/// host layout instantiation (chelis#940).
///
/// A dimension parameter can survive the checker as a named ADT argument even
/// after the tensor carrying that dimension has a concrete checked shape. It
/// has no host ABI representation: `Frame[n] -> Hamt[Column[n]] ->
/// tensor[n, _]` stores the tensor handle, not `n`. Replace such arguments
/// with `Unit` as a private layout witness. Ordinary value parameters remain
/// untouched and therefore still fail loudly if unresolved.
fn instantiate_adt_constructor(
    program: &HostLoweringSession<'_>,
    definition: &GenericAdtConstructor,
    instantiated_ty: &HostTypeTerm,
) -> Result<InstantiatedAdtConstructor, AdtInstantiationError> {
    let definitions = adt_constructor_definitions(program);
    let canonical =
        canonicalize_representation_erased_adt_args(instantiated_ty.clone(), &definitions);
    definition.instantiate(&canonical)
}

fn canonicalize_representation_erased_adt_args(
    ty: HostTypeTerm,
    definitions: &[GenericAdtConstructor],
) -> HostTypeTerm {
    match ty {
        HostTypeTerm::Fn(params, ret) => HostTypeTerm::Fn(
            params
                .into_iter()
                .map(|ty| canonicalize_representation_erased_adt_args(ty, definitions))
                .collect(),
            Box::new(canonicalize_representation_erased_adt_args(
                *ret,
                definitions,
            )),
        ),
        HostTypeTerm::Adt(name, args) => {
            let args = args
                .into_iter()
                .enumerate()
                .map(|(index, arg)| {
                    let arg = canonicalize_representation_erased_adt_args(arg, definitions);
                    if arg.is_unresolved() && !adt_parameter_is_stored(&name, index, definitions) {
                        HostTypeTerm::Unit
                    } else {
                        arg
                    }
                })
                .collect();
            HostTypeTerm::Adt(name, args)
        }
        HostTypeTerm::Tuple(items) => HostTypeTerm::Tuple(
            items
                .into_iter()
                .map(|ty| canonicalize_representation_erased_adt_args(ty, definitions))
                .collect(),
        ),
        HostTypeTerm::List(inner) => HostTypeTerm::List(Box::new(
            canonicalize_representation_erased_adt_args(*inner, definitions),
        )),
        HostTypeTerm::Option(inner) => HostTypeTerm::Option(Box::new(
            canonicalize_representation_erased_adt_args(*inner, definitions),
        )),
        HostTypeTerm::Dict(key, value) => HostTypeTerm::Dict(
            Box::new(canonicalize_representation_erased_adt_args(
                *key,
                definitions,
            )),
            Box::new(canonicalize_representation_erased_adt_args(
                *value,
                definitions,
            )),
        ),
        other => other,
    }
}

fn adt_parameter_is_stored(
    adt_name: &str,
    parameter_index: usize,
    definitions: &[GenericAdtConstructor],
) -> bool {
    definitions_owning(definitions, adt_name, adt_name_of)
        .candidates()
        .iter()
        .any(|definition| {
            definition
                .stored_parameters
                .get(parameter_index)
                .copied()
                .unwrap_or(true)
        })
}

/// Which spelling answered when narrowing declarations to a name.
enum NameMatch<'a> {
    /// At least one declaration carries the name exactly.
    Exact(Vec<&'a GenericAdtConstructor>),
    /// No declaration carries the name exactly; these share its terminal
    /// segment. Empty when nothing matched at all.
    Terminal(Vec<&'a GenericAdtConstructor>),
}

impl<'a> NameMatch<'a> {
    fn candidates(&self) -> &[&'a GenericAdtConstructor] {
        match self {
            Self::Exact(candidates) | Self::Terminal(candidates) => candidates,
        }
    }
}

/// Narrow `definitions` to the declarations that own `name` under `key`,
/// preferring an exact spelling over the terminal-segment fallback.
///
/// The reef linker rewrites every cross-package binding to
/// `Pkg__<pkg>__<Module>__<Name>`, and spec/04-type-system.md's "Module
/// identity" rule makes that mangled spelling the declaration's identity.
/// An exact spelling therefore resolves on its own and must never consult
/// the terminal segment: two packages in one dependency graph may declare
/// same-named types, whose terminals then collide by construction, and
/// answering such a reference with the other package's declaration
/// lowered a valid program against the wrong record (chelis#1271). The
/// terminal fallback stays for short, unqualified spellings, which only
/// arise when no declaration carries the exact name.
fn definitions_owning<'a>(
    definitions: &'a [GenericAdtConstructor],
    name: &str,
    key: fn(&GenericAdtConstructor) -> &str,
) -> NameMatch<'a> {
    let exact: Vec<&GenericAdtConstructor> = definitions
        .iter()
        .filter(|definition| key(definition) == name)
        .collect();
    if !exact.is_empty() {
        return NameMatch::Exact(exact);
    }
    NameMatch::Terminal(
        definitions
            .iter()
            .filter(|definition| terminal_name_matches(key(definition), name))
            .collect(),
    )
}

fn constructor_name_of(definition: &GenericAdtConstructor) -> &str {
    &definition.ctor_name
}

fn adt_name_of(definition: &GenericAdtConstructor) -> &str {
    &definition.adt_name
}

/// How a constructor reference resolved against the checked program's ADT
/// registry (chelis#1271).
enum AdtConstructorResolution {
    /// Exactly one declaration owns the reference.
    Unique(GenericAdtConstructor),
    /// No declaration owns the reference. Call sites that also accept
    /// plain functions read this as "not a constructor".
    Missing,
    /// The reference is spelled short and more than one declaration's
    /// terminal segment answers it. Carries every candidate's declared
    /// name so the diagnostic can name both sides instead of silently
    /// picking whichever sorted first.
    Ambiguous(Vec<String>),
}

fn resolve_adt_constructor_definition(
    program: &HostLoweringSession<'_>,
    ctor_name: &str,
) -> AdtConstructorResolution {
    let definitions = adt_constructor_definitions(program);
    let owning = definitions_owning(&definitions, ctor_name, constructor_name_of);
    resolve_adt_constructor_candidates(&owning)
}

/// Resolve a positional construction using the nominal result already
/// established by the checker. Two ADTs may deliberately share an exact
/// constructor name; the checked result type is the structural owner receipt
/// that lowering needs, whereas the sorted registry population is not scope.
fn resolve_adt_constructor_definition_for_type(
    program: &HostLoweringSession<'_>,
    ctor_name: &str,
    checked_ty: &HostTypeTerm,
) -> AdtConstructorResolution {
    let definitions = adt_constructor_definitions(program);
    let owning = definitions_owning(&definitions, ctor_name, constructor_name_of);
    if let HostTypeTerm::Adt(adt_name, _) = checked_ty {
        let exact_owner = owning
            .candidates()
            .iter()
            .copied()
            .filter(|definition| definition.adt_name == *adt_name)
            .collect::<Vec<_>>();
        if let [only] = exact_owner.as_slice() {
            return AdtConstructorResolution::Unique((*only).clone());
        }
        let terminal_owner = owning
            .candidates()
            .iter()
            .copied()
            .filter(|definition| terminal_name_matches(&definition.adt_name, adt_name))
            .collect::<Vec<_>>();
        if let [only] = terminal_owner.as_slice() {
            return AdtConstructorResolution::Unique((*only).clone());
        }
    }
    resolve_adt_constructor_candidates(&owning)
}

fn resolve_adt_constructor_candidates(owning: &NameMatch<'_>) -> AdtConstructorResolution {
    match (owning, owning.candidates()) {
        (_, []) => AdtConstructorResolution::Missing,
        (_, [only]) => AdtConstructorResolution::Unique((*only).clone()),
        // Two declarations carrying the SAME name are not a
        // qualified-versus-unqualified ambiguity: no name rule can tell
        // them apart, so there is nothing here for a name rule to
        // decide. That case is a settled decision with no open owner,
        // and its gate is the checker, which either resolves a same-name
        // constructor (by call shape) or rejects it. A program that
        // survives the checker therefore does not expose which of two
        // identically-named declarations this sorted table returns.
        // Keep the existing choice rather than widening this repair.
        (NameMatch::Exact(_), [first, ..]) => AdtConstructorResolution::Unique((*first).clone()),
        (NameMatch::Terminal(_), many) => AdtConstructorResolution::Ambiguous(
            many.iter()
                .map(|definition| definition.ctor_name.clone())
                .collect(),
        ),
    }
}

/// The fail-closed diagnostic for a constructor reference that more than
/// one declaration answers to.
///
/// Lowering has no disambiguator at this point: the reference is short,
/// the candidates are distinct types, and choosing one would rebuild the
/// chelis#1271 defect with a different first-match rule.
///
/// The rendering BORROWS `host_expr_lowering_error`'s generic Deep-form
/// brand, whose [04-TOT-3] text reads "a malformed or unhandled Deep
/// form" - a contested but well-formed constructor spelling is neither.
/// Citation accuracy in that shared block is chelis#1260's subject and
/// this change leaves the block byte-identical, so the brand is borrowed
/// here rather than corrected in passing.
fn ambiguous_constructor_error(
    expr: &Expr,
    ctor_name: &str,
    candidates: &[String],
) -> crate::lower::LowerDiagnostic {
    let listed = candidates
        .iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ");
    host_expr_lowering_error(
        expr,
        format!(
            "constructor `{ctor_name}` is answered by {count} declarations \
             ({listed}) whose names share a terminal segment; each is a \
             distinct type, so lowering will not choose one. Spell the \
             constructor with its declaring module or package",
            count = candidates.len(),
        ),
    )
}

/// The single declaration that owns `ctor_name`, if there is one.
///
/// Two call sites are PREDICATES over a name that may or may not be a
/// constructor, and neither can raise: they ask whether a callee has a
/// narrow specialization shape and what an application's result type is.
/// Both answer "no" for an ambiguous reference, which is safe because
/// every path that turns a constructor reference into a construction
/// (bare-variable, record, and positional application) rejects ambiguity
/// on its own, so the program still fails closed rather than lowering
/// down a silently different path.
fn lookup_adt_constructor_definition(
    program: &HostLoweringSession<'_>,
    ctor_name: &str,
) -> Option<GenericAdtConstructor> {
    match resolve_adt_constructor_definition(program, ctor_name) {
        AdtConstructorResolution::Unique(definition) => Some(definition),
        AdtConstructorResolution::Missing | AdtConstructorResolution::Ambiguous(_) => None,
    }
}

fn lookup_access_field(
    program: &HostLoweringSession<'_>,
    base: &HostExpr,
    field_name: &str,
) -> Result<Option<(usize, HostTypeTerm)>, AdtInstantiationError> {
    match &base.kind {
        HostExprKind::AdtConstruct { ctor, ty, .. } => {
            // The base is an already-lowered construction, and every path
            // that builds one rejects an ambiguous constructor name, so a
            // `None` here means the name is unknown rather than
            // contested. The caller renders that as "absent or ambiguous
            // on the resolved ADT type" (chelis#1271).
            let AdtConstructorResolution::Unique(definition) =
                resolve_adt_constructor_definition_for_type(program, ctor, ty)
            else {
                return Ok(None);
            };
            let instantiated = instantiate_adt_constructor(program, &definition, ty)?;
            Ok(instantiated
                .fields
                .iter()
                .enumerate()
                .find_map(|(index, field)| {
                    (field.name.as_deref() == Some(field_name)).then_some((index, field.ty.clone()))
                }))
        }
        _ => match host_expr_type(base) {
            HostTypeTerm::Adt(adt_name, args) => {
                lookup_adt_field_on_type(program, &adt_name, &args, field_name)
            }
            _ => Ok(None),
        },
    }
}

fn lookup_adt_field_on_type(
    program: &HostLoweringSession<'_>,
    adt_name: &str,
    args: &[HostTypeTerm],
    field_name: &str,
) -> Result<Option<(usize, HostTypeTerm)>, AdtInstantiationError> {
    let mut found = None;
    let instantiated_ty = HostTypeTerm::Adt(adt_name.to_string(), args.to_vec());
    // Field access resolves through the ADT's own name, and it has the
    // same chelis#1271 collision: two packages' `Wrapped` types share a
    // terminal, so admitting both as candidates made them disagree on the
    // index of a field one of them really has, and a valid access was
    // rejected as "absent or ambiguous". An exact name answers alone.
    let definitions = adt_constructor_definitions(program);
    for definition in definitions_owning(&definitions, adt_name, adt_name_of).candidates() {
        let instantiated = instantiate_adt_constructor(program, definition, &instantiated_ty)?;
        if let Some((index, field)) = instantiated
            .fields
            .iter()
            .enumerate()
            .find(|(_, field)| field.name.as_deref() == Some(field_name))
        {
            let candidate = (index, field.ty.clone());
            if let Some(existing) = &found
                && existing != &candidate
            {
                return Ok(None);
            }
            found = Some(candidate);
        }
    }
    Ok(found)
}

fn tag(list: &List) -> Option<DeepTag> {
    list.tag()
}

fn list_span(list: &List) -> chelis_deep::Span {
    list.elements
        .first()
        .map(Expr::span)
        .unwrap_or_else(|| chelis_deep::Span::new(0, 0))
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

/// Borrow a canonical stamped node without reconstructing a legacy `List`.
// Transitional E5b adapter; `Expr::carrier` owns physical-carrier decoding.
fn stamped_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match expr.carrier() {
        chelis_deep::ExprCarrier::DecodedNode(tag, metadata, children) => {
            Some((tag, metadata, children))
        }
        chelis_deep::ExprCarrier::StructuralList(_)
        | chelis_deep::ExprCarrier::UndecodableHead(_, _, _)
        | chelis_deep::ExprCarrier::Atom(_)
        | chelis_deep::ExprCarrier::MetadataMap(_)
        | chelis_deep::ExprCarrier::MetadataExpression(_)
        | chelis_deep::ExprCarrier::MalformedLegacyList(_) => None,
    }
}

fn as_list(expr: &Expr) -> Option<&List> {
    match expr {
        Expr::List(list, _) => Some(list),
        _ => None,
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

/// Collect every `(var {} <name>)` reference reachable in a Deep `expr`
/// into `out`. Issue #378: used to decide whether a top-level *value*
/// binding (a non-`fn` `def` body) is captured by a host-lane function
/// body and must therefore be materialized into `HostProgram::globals`
/// rather than dropped via `skip_for_lowered`. A bound parameter and a
/// captured global both appear as the same `(var ...)` form here; the
/// caller narrows to top-level binding names, so the over-approximation
/// is harmless (the C emitter's `captured_global_names` is the final
/// gate — an emitted global is only declared if a function actually
/// references it).
pub(crate) fn collect_deep_var_names(expr: &Expr, out: &mut UnordSet<String>) {
    match expr {
        Expr::List(list, _) => {
            if tag(list) == Some(DeepTag::Var)
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                out.insert(name.to_string());
            }
            for child in &list.elements {
                collect_deep_var_names(child, out);
            }
        }
        Expr::MetaExpr(meta, _) => collect_deep_var_names(&meta.expr, out),
        _ => {}
    }
}

fn param_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name.clone()),
        Expr::MetaExpr(meta, _) => param_name(&meta.expr),
        Expr::BareList(elements, _) => elements.first().and_then(symbol_name).map(str::to_string),
        Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .or_else(|| children(list).first().and_then(symbol_name))
            .map(str::to_string),
        _ => None,
    }
}

fn param_host_type(expr: &Expr) -> Option<HostTypeTerm> {
    match expr {
        Expr::MetaExpr(meta, _) => expr_type(expr)
            .filter(|ty| !ty.is_unresolved())
            .or_else(|| param_host_type(&meta.expr)),
        Expr::List(_, _) | Expr::BareList(_, _) => expr_type(expr).filter(|ty| !ty.is_unresolved()),
        _ => None,
    }
}

thread_local! {
    /// Type-variable bindings in force while a monomorphized body is being
    /// lowered (chelis#1201). A stack, because specializing one body can
    /// specialize another nested inside it.
    ///
    /// A specialized body's nodes still carry the GENERIC checked types the
    /// checker recorded (`Hamt[a]`), so every generic-ADT construction inside
    /// it reads as unresolved even though the specialization pinned `a` at
    /// its call site. Threading a substitution parameter would touch ~130
    /// call sites across this file; scoping it here matches the idiom
    /// `MONO_SPECIALIZATIONS` and `INLINING_STACK` already use,
    /// and applies at the single point that decodes a node's checked type.
    static ACTIVE_TYPE_SUBST: RefCell<Vec<UnordMap<String, HostTypeTerm>>> =
        const { RefCell::new(Vec::new()) };
}

/// The substitution in force for the innermost specialization, if any.
///
/// Outside every specialization the answer is an empty substitution, and
/// that is a real answer rather than a missing one: no type variable is
/// bound there, so applying it is the identity. Spelled as an explicit
/// branch instead of a default so the empty map is visibly the stated
/// outcome (spec/design/loud_unsupported.md C4.3).
fn active_type_subst() -> UnordMap<String, HostTypeTerm> {
    ACTIVE_TYPE_SUBST.with(|stack| match stack.borrow().last() {
        Some(subst) => subst.clone(),
        None => UnordMap::new(),
    })
}

/// RAII scope for a specialization's type bindings (chelis#1201).
struct ActiveTypeSubstGuard;

impl ActiveTypeSubstGuard {
    fn push(subst: UnordMap<String, HostTypeTerm>) -> Self {
        ACTIVE_TYPE_SUBST.with(|stack| stack.borrow_mut().push(subst));
        ActiveTypeSubstGuard
    }
}

impl Drop for ActiveTypeSubstGuard {
    fn drop(&mut self) {
        ACTIVE_TYPE_SUBST.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

/// The `params` list of a top-level `fn` body (chelis#1201).
fn params_list_of(fn_expr: &Expr) -> Option<&List> {
    let Expr::List(fn_list, _) = fn_expr else {
        return None;
    };
    if tag(fn_list) != Some(DeepTag::Fn) {
        return None;
    }
    children(fn_list)
        .first()
        .and_then(as_list)
        .filter(|params| tag(params) == Some(DeepTag::Params))
}

/// The declared type on a `params` entry: `(name {type: T})` (chelis#1201).
fn param_declared_type_expr(param: &Expr) -> Option<Expr> {
    let meta = match param {
        Expr::List(list, _) => match list.elements.get(1) {
            Some(Expr::Map(meta, _)) => meta,
            _ => return None,
        },
        Expr::BareList(elements, _) => match elements.get(1) {
            Some(Expr::Map(meta, _)) => meta,
            _ => return None,
        },
        Expr::Node(node, _) => node.meta(),
        _ => return None,
    };
    meta.ty().map(|ty| ty.expression()).cloned()
}

/// The declared return type of a top-level def, read from its sibling
/// `defsig`'s `t-fn` (chelis#1201).
///
/// A `def`'s own `fn` node carries no signature metadata — the declared
/// signature is a separate top-level `(defsig {} name (t-fn {} ...))`
/// item — so this looks the sibling up by the same exact-then-terminal
/// name rule `find_top_level_def_named` uses.
fn declared_return_type_expr(exprs: &[Expr], name: &str) -> Option<Expr> {
    let mut terminal_match: Option<Expr> = None;
    let mut terminal_is_ambiguous = false;
    for expr in top_level_items(exprs) {
        let Some((DeepTag::Defsig, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some(sig_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(ret) = kids
            .last()
            .and_then(as_list)
            .filter(|tfn| tag(tfn) == Some(DeepTag::TFn))
            .and_then(|tfn| children(tfn).last().cloned())
        else {
            continue;
        };
        if sig_name == name {
            return Some(ret);
        }
        if terminal_name_matches(sig_name, name) {
            if terminal_match.is_some() {
                terminal_is_ambiguous = true;
            } else {
                terminal_match = Some(ret);
            }
        }
    }
    (!terminal_is_ambiguous).then_some(terminal_match).flatten()
}

/// The type substitution for a generic callee inlined at one call site
/// (chelis#2152).
///
/// This is the inline counterpart of the substitution
/// `lower_mono_specialized_function` pins for a monomorphized body. It is
/// solved against the callee `fn` node's OWN recorded type, whose variables
/// share the inlined body's namespace, and against the concrete argument and
/// result types of this call.
///
/// It extends the enclosing substitution rather than replacing it. Inlining
/// substitutes the caller's argument expressions into the body, and those
/// still carry the CALLER's type variables, which only the enclosing
/// substitution resolves. Existing bindings win over new ones, as in
/// `solve_host_type_vars`.
fn inline_call_type_subst(
    name: &str,
    args: &[Expr],
    call_ty: &HostTypeTerm,
    program: &HostLoweringSession<'_>,
    scope: &UnordMap<String, HostTypeTerm>,
) -> UnordMap<String, HostTypeTerm> {
    let mut subst = active_type_subst();
    let Some((_, fn_expr)) = find_top_level_def_named(program.exprs(), name) else {
        return subst;
    };
    let Some((generic_params, generic_ret)) = expr_fn_type(fn_expr) else {
        return subst;
    };
    for (generic, arg) in generic_params.iter().zip(args.iter()) {
        let actual = expr_host_type(arg, program, scope);
        solve_host_type_vars(generic, &actual, &mut subst);
    }
    solve_host_type_vars(&generic_ret, call_ty, &mut subst);
    subst
}

/// Structurally match a declared host type against a resolved one, binding
/// each `TypeVariable` on the declared side (chelis#1201).
///
/// Only the shapes a generic signature can name are walked; anything else
/// contributes no binding rather than guessing one.
fn solve_host_type_vars(
    declared: &HostTypeTerm,
    actual: &HostTypeTerm,
    out: &mut UnordMap<String, HostTypeTerm>,
) {
    match (declared, actual) {
        (HostTypeTerm::TypeVariable(name), resolved) if !resolved.is_unresolved() => {
            out.entry(name.clone()).or_insert_with(|| resolved.clone());
        }
        // chelis#2152: a tensor's precision variable is the same checked
        // variable a scalar occurrence spells as `TypeVariable(name)`. Bind it
        // to the scalar of that precision, which is also the shape
        // `substitute_host_type_term` expects for a precision binding.
        (
            HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
                precision: HostPrecisionTerm::Variable(name),
                ..
            }),
            HostTypeTerm::Tensor(TensorType { precision, .. })
            | HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
                precision: HostPrecisionTerm::Concrete(precision),
                ..
            }),
        ) => {
            out.entry(name.clone())
                .or_insert_with(|| HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(*precision)));
        }
        (HostTypeTerm::Adt(dname, dargs), HostTypeTerm::Adt(aname, aargs))
            if terminal_name_matches(dname, aname) && dargs.len() == aargs.len() =>
        {
            for (d, a) in dargs.iter().zip(aargs.iter()) {
                solve_host_type_vars(d, a, out);
            }
        }
        (HostTypeTerm::List(d), HostTypeTerm::List(a))
        | (HostTypeTerm::Option(d), HostTypeTerm::Option(a)) => solve_host_type_vars(d, a, out),
        (HostTypeTerm::Dict(dk, dv), HostTypeTerm::Dict(ak, av)) => {
            solve_host_type_vars(dk, ak, out);
            solve_host_type_vars(dv, av, out);
        }
        (HostTypeTerm::Tuple(ds), HostTypeTerm::Tuple(as_)) if ds.len() == as_.len() => {
            for (d, a) in ds.iter().zip(as_.iter()) {
                solve_host_type_vars(d, a, out);
            }
        }
        (HostTypeTerm::Fn(dp, dr), HostTypeTerm::Fn(ap, ar)) if dp.len() == ap.len() => {
            for (d, a) in dp.iter().zip(ap.iter()) {
                solve_host_type_vars(d, a, out);
            }
            solve_host_type_vars(dr, ar, out);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DimInfo, RiscOp};
    use chelis_types::types::Prim;

    #[test]
    fn kernel_builtin_load_rejection_requires_a_typed_lexical_input() {
        for name in ["mean", "fold", "map"] {
            let mut dag = crate::Dag::new();
            dag.add_node(
                RiscOp::Load { name: name.into() },
                vec![],
                TensorType {
                    dims: vec![],
                    precision: Prim::F32,
                },
                None,
            );
            let mut scope = UnordMap::new();
            assert_eq!(kernel_dag_loads_builtin(&dag, &scope), Some(name.into()));
            scope.insert(
                "unrelated".into(),
                HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::F32)),
            );
            assert_eq!(kernel_dag_loads_builtin(&dag, &scope), Some(name.into()));
            scope.insert(name.into(), HostTypeTerm::String);
            assert_eq!(kernel_dag_loads_builtin(&dag, &scope), Some(name.into()));
            scope.insert(
                name.into(),
                HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::F32)),
            );
            assert_eq!(kernel_dag_loads_builtin(&dag, &scope), None);
        }
    }

    fn parse_one_expr(source: &str) -> Expr {
        deep_expr(source)
    }

    #[test]
    fn direct_builtin_expr_reader_covers_each_recursive_carrier() {
        let span = chelis_deep::Span::new(0, 0);
        let call = parse_one_expr("(app {} (var {} abs) (lit {} 1))");
        let metadata = Metadata::from(chelis_deep::annotations::MetadataValue::PropertySeed(
            chelis_deep::annotations::RuntimeExpression::try_new(call.clone())
                .expect("a builtin call is valid runtime metadata"),
        ));
        let carriers = [
            Expr::BareList(vec![call.clone()], span),
            Expr::UnknownForm(Box::new(chelis_deep::ast::UnknownFormData {
                head: "future-form".to_string(),
                meta: Metadata::default(),
                children: vec![call.clone()],
                span,
            })),
            Expr::Map(metadata, span),
            Expr::MetaExpr(
                chelis_deep::ast::MetaExpr {
                    metadata: Metadata::default(),
                    expr: Box::new(call.clone()),
                },
                span,
            ),
            Expr::List(
                List {
                    elements: vec![Expr::Atom(Atom::Name("malformed".to_string()), span), call],
                },
                span,
            ),
        ];

        for carrier in carriers {
            assert_eq!(
                find_direct_builtin_call_in_expr(&carrier, &["abs"]),
                Some("abs".to_string())
            );
            assert_eq!(find_direct_builtin_call_in_expr(&carrier, &["sqrt"]), None);
        }
    }

    #[test]
    fn literal_result_claim_transfer_requires_a_pure_called_helper() {
        let checked = surf_check(
            "def pure[n](x: tensor[n, f32]) -> tensor[2, f32] = \
                 shrink(x, [[1i64, shape(x, 0i32)]])\n\
             def random[n](x: tensor[n, f32]) -> tensor[2, f32] = \
                 dropout(shrink(x, [[1i64, shape(x, 0i32)]]), 0.5f32)\n\
             def caller[n](x: tensor[n, f32]) = (pure(copy(x)), random(x))\n",
        );
        let session = HostLoweringSession::new(&checked);
        assert!(top_level_fn_transfers_literal_result_claims(
            &session, "pure"
        ));
        assert!(!top_level_fn_transfers_literal_result_claims(
            &session, "random"
        ));
    }

    #[test]
    fn concat_admission_uses_static_named_extent_metadata() {
        let body = parse_one_expr(
            "(app {} (var {} concat) (app {} (var {} Cons) (var {} x) (app {} (var {} Cons) (var {} x) (var {} Nil))) (lit {} 1))",
        );
        let defs = BTreeMap::new();
        for (width, host) in [
            (DimInfo::Lit(2), false),
            (DimInfo::Named("width".to_string(), Some(2)), false),
            (DimInfo::Named("width".to_string(), None), true),
        ] {
            let tensor = TensorType {
                dims: vec![DimInfo::Lit(2), width],
                precision: Prim::F32,
            };
            let mut walk = UncarriableWalk {
                defs: &defs,
                active: UnordSet::new(),
                completed: UnordMap::new(),
                cycle_cutoff: false,
                def_visits: 0,
                evaluation_dropout: false,
                scopes: vec![UnordMap::from([(
                    "x".to_string(),
                    AdmissionBinding {
                        constructor: false,
                        concat: ConcatInputFact::Tensor(tensor.clone()),
                    },
                )])],
            };
            assert_eq!(walk.expr(&body).is_some(), host, "{tensor:?}");
            if !host {
                crate::lower::try_lower_subexpr_program(
                    &body,
                    UnordMap::from([("x".to_string(), tensor)]),
                    UnordMap::new(),
                    UnordMap::new(),
                )
                .expect("the same known extent is representable by static concat");
            }
        }
    }

    #[test]
    fn concat_admission_does_not_project_destructuring_aliases() {
        // Private structural control: the walk has no tuple/list projection
        // proof. A nested binder must not inherit its RHS's whole List fact.
        let body = parse_one_expr(
            "(let {} (bind {} (tuple {} (var {} a) (var {} b)) (var {} xs)) (app {} (var {} concat) (var {} b) (lit {} 1)))",
        );
        let defs = BTreeMap::new();
        let mut walk = UncarriableWalk {
            defs: &defs,
            active: UnordSet::new(),
            completed: UnordMap::new(),
            cycle_cutoff: false,
            def_visits: 0,
            evaluation_dropout: false,
            scopes: vec![UnordMap::from([(
                "xs".to_string(),
                AdmissionBinding {
                    constructor: false,
                    concat: ConcatInputFact::List(vec![ConcatInputFact::Tensor(TensorType {
                        dims: vec![DimInfo::Lit(2), DimInfo::Named("width".to_string(), None)],
                        precision: Prim::F32,
                    })]),
                },
            )])],
        };
        assert!(walk.expr(&body).is_none());
    }

    #[test]
    fn concat_admission_captures_actuals_before_formal_shadowing() {
        // Admission-only: the older tensor lowerer has a separate x/y
        // substitution collision for this source. This test does not claim
        // kernel execution; the fresh-formal integration twin covers that.
        for (x_width, y_width, host) in [("*", "2", true), ("2", "*", false)] {
            let source = format!(
                "def second[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([y, y], 1i32)\ndef run[s](x: tensor[s, {x_width}, f32], y: tensor[s, {y_width}, f32]) -> tensor[s, *, f32] = softmax(second(y, x), -1)\n"
            );
            let program = surf_check(&source);
            let defs = cached_program_defs(&HostLoweringSession::new(&program));
            let (_, _, run) = stamped_parts(&defs["run"]).unwrap();
            let (_, _, params) = stamped_parts(&run[0]).unwrap();
            let params: Vec<_> = params
                .iter()
                .map(|param| HostParam {
                    name: param_name(param).unwrap(),
                    ty: param_host_type(param).unwrap(),
                })
                .collect();
            assert_eq!(
                body_form_the_dag_cannot_carry(
                    &HostLoweringSession::new(&program),
                    &run[1],
                    &params,
                    false
                )
                .is_some(),
                host,
                "{source}"
            );
        }
    }

    fn admission_test_walk(defs: &BTreeMap<String, Expr>) -> UncarriableWalk<'_> {
        UncarriableWalk {
            defs,
            active: UnordSet::new(),
            completed: UnordMap::new(),
            cycle_cutoff: false,
            def_visits: 0,
            evaluation_dropout: false,
            scopes: Vec::new(),
        }
    }

    #[test]
    fn concat_admission_shared_context_visits_each_helper_once() {
        // Sharing must not expand into every call path. Count body visits,
        // not wall time; the trailing rank must still choose Host.
        let depth = 12;
        let mut source = "def level_0(x: tensor[2, f32]) -> tensor[2, f32] = x\n".to_string();
        for level in 1..=depth {
            source.push_str(&format!(
                "def level_{level}(x: tensor[2, f32]) -> tensor[2, f32] = add(level_{}(copy(x)), level_{}(copy(x)))\n",
                level - 1, level - 1
            ));
        }
        source.push_str(&format!(
            "def run(x: tensor[2, f32]) -> tensor[2, f32] = {{\n z = level_{depth}(copy(x))\n _ = rank(x)\n z\n}}\n"
        ));
        let program = surf_check(&source);
        let defs = cached_program_defs(&HostLoweringSession::new(&program));
        let mut walk = admission_test_walk(&defs);
        assert!(
            walk.def("run", &defs["run"], None)
                .unwrap()
                .contains("rank")
        );
        assert_eq!(walk.def_visits, depth + 2);
    }

    #[test]
    fn concat_admission_completed_context_retains_order_and_exact_facts() {
        let program = surf_check(
            "def second[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([y, y], 1i32)\n",
        );
        let defs = cached_program_defs(&HostLoweringSession::new(&program));
        let fixed = ConcatInputFact::Tensor(TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
            precision: Prim::F32,
        });
        let dynamic = ConcatInputFact::Tensor(TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Named("width".to_string(), None)],
            precision: Prim::F32,
        });
        for order in [[false, true, false, true], [true, false, true, false]] {
            let mut walk = admission_test_walk(&defs);
            for host in order {
                let actuals = if host {
                    vec![fixed.clone(), dynamic.clone()]
                } else {
                    vec![dynamic.clone(), fixed.clone()]
                };
                assert_eq!(
                    walk.def("second", &defs["second"], Some(actuals)).is_some(),
                    host
                );
            }
            assert_eq!(
                walk.def_visits, 2,
                "two distinct ordered contexts, not four calls"
            );
        }
    }

    #[test]
    fn concat_admission_completed_callee_does_not_capture_caller_bindings() {
        let program = surf_check(
            "def global(x: tensor[2, f32]) -> i32 = rank(x)\ndef call(x: tensor[2, f32]) -> i32 = global(x)\n",
        );
        let defs = cached_program_defs(&HostLoweringSession::new(&program));
        let mut walk = admission_test_walk(&defs);
        // A free callee name resolves against program definitions, not a
        // same-spelled caller alias, including its constructor classification.
        for constructor in [false, true] {
            walk.scopes = vec![UnordMap::from([(
                "global".to_string(),
                AdmissionBinding {
                    constructor,
                    concat: ConcatInputFact::Unknown,
                },
            )])];
            assert!(
                walk.def("call", &defs["call"], None)
                    .unwrap()
                    .contains("rank")
            );
            assert_eq!(walk.bound("global"), Some(constructor));
        }
        assert_eq!(walk.def_visits, 2, "call and global each visited once");
    }

    #[test]
    fn concat_admission_cycle_cutoff_is_not_a_completed_negative() {
        // Private walk control: b reached under active a cannot conclude that
        // a lacks Host forms. A later fresh b must still see a's trailing rank.
        let defs = BTreeMap::from([
            (
                "a".to_string(),
                parse_one_expr(
                    "(fn {} (params {}) (tuple {} (app {} (var {} b)) (app {} (var {} rank) (lit {} 1))))",
                ),
            ),
            (
                "b".to_string(),
                parse_one_expr("(fn {} (params {}) (app {} (var {} a)))"),
            ),
        ]);
        let mut walk = admission_test_walk(&defs);
        for name in ["a", "b", "a", "b"] {
            assert!(
                walk.def(name, &defs[name], Some(Vec::new()))
                    .unwrap()
                    .contains("rank")
            );
            assert!(walk.active.is_empty(), "active guard must unwind");
        }
    }

    #[test]
    fn linked_alias_resolution_accepts_one_terminal_match() {
        let mut registry = AdtRegistry::new();
        registry.register_alias(
            "PairAlias".to_string(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Type::Tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::F32)]),
        );

        let resolved = resolve_host_type_alias(
            &registry,
            "Pkg__issue__1293__recursive__cotangents__Demo__Main__PairAlias",
        )
        .expect("a qualified linked reference must find its unique transparent alias");
        assert!(matches!(resolved.body, Type::Tuple(_)));
    }

    #[test]
    fn linked_alias_resolution_refuses_an_ambiguous_terminal_match() {
        let mut registry = AdtRegistry::new();
        for name in ["Adep__PairAlias", "Blib__PairAlias"] {
            registry.register_alias(
                name.to_string(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Type::Tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::F32)]),
            );
        }

        assert!(
            resolve_host_type_alias(&registry, "Pkg__PairAlias").is_none(),
            "a non-exact terminal spelling must not silently select one colliding alias"
        );
        assert!(
            resolve_host_type_alias(&registry, "Adep__PairAlias").is_some(),
            "an exact linked alias identity remains authoritative"
        );
    }

    #[test]
    fn linked_alias_resolution_preserves_an_exact_adt_identity() {
        let mut registry = AdtRegistry::new();
        registry.defs.insert(
            "Pkg__collidelib__Demo__Main__Wrapped".to_string(),
            AdtDef {
                name: "Pkg__collidelib__Demo__Main__Wrapped".to_string(),
                type_params: Vec::new(),
                param_kinds: Vec::new(),
                param_vars: Vec::new(),
                param_args: Vec::new(),
                variants: Vec::new(),
                opaque: false,
                defining_module: None,
            },
        );
        registry.register_alias(
            "Pkg__collidedep__Demo__Main__Wrapped".to_string(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Type::Tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::F32)]),
        );

        assert!(
            resolve_host_type_alias(&registry, "Pkg__collidelib__Demo__Main__Wrapped").is_none(),
            "an exact ADT identity must not be reinterpreted as another package's alias"
        );
    }

    #[test]
    fn issue_662_recursive_def_cycle_does_not_hide_a_later_forward_fail() {
        let defs = BTreeMap::from([
            (
                "a".to_string(),
                parse_one_expr("(tuple {} (var {} b) (var {} fail))"),
            ),
            ("b".to_string(), parse_one_expr("(var {} a)")),
        ]);
        assert!(
            expr_reaches_forward_fail(&parse_one_expr("(var {} a)"), &defs, &mut UnordSet::new()),
            "breaking the a -> b -> a cycle must continue with a's fail sibling"
        );
    }

    #[test]
    fn issue_662_recursive_def_cycle_without_fail_terminates_negative() {
        let defs = BTreeMap::from([
            ("a".to_string(), parse_one_expr("(var {} b)")),
            ("b".to_string(), parse_one_expr("(var {} a)")),
        ]);
        assert!(
            !expr_reaches_forward_fail(&parse_one_expr("(var {} a)"), &defs, &mut UnordSet::new()),
            "a fail-free recursive cycle must terminate without inventing reachability"
        );
    }

    // chelis#1087's two `substitute_var` pass-through locks went with the
    // function, which chelis#1923 deleted along with this lane's own pipe
    // beta-reduction: the checker now folds a pipe into the application
    // spec/02-surf-syntax.md §0.1 says it denotes, so no lane substitutes a
    // stage parameter any more. The property they pinned, that a
    // transitional variant is passed through unchanged rather than rebuilt,
    // is a property of the surviving substitution and is pinned there by
    // `chelis_deep::pipe::tests::a_transitional_variant_is_passed_through`.

    // ── harden-bounded-monomorphization D1/D2/D4 unit locks ──

    /// Provenance, not symbol spelling, identifies an internal
    /// specialization. A valid authored snake_case name can exactly match the
    /// mangling grammar.
    #[test]
    fn mono_specialization_provenance_is_explicit() {
        let name = "authored__mono_0123456789abcdef";
        let authored = synthetic_tensor_function(name, HostFunctionOrigin::Authored);
        let generated = synthetic_tensor_function(name, HostFunctionOrigin::Monomorphized);
        assert!(!authored.is_monomorphized_specialization());
        assert!(generated.is_monomorphized_specialization());
        let minted = mono_specialization_symbol(
            "depth",
            &mono_specialization_key("depth", &[HostTypeTerm::Int64], &HostTypeTerm::Int64),
        );
        assert!(minted.starts_with("depth__mono_"));
        assert_eq!(minted.len(), "depth__mono_".len() + 16);
    }

    /// The interning key is a purpose-built canonical rendering with a
    /// golden spelling — never `derive(Debug)` output, whose shape is stable
    /// only by accident.
    #[test]
    fn mono_specialization_key_is_stable() {
        let box_int32 = HostTypeTerm::Adt(
            "Box".to_string(),
            vec![HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(
                Prim::Int32,
            ))],
        );
        let i32 = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32));
        let key = mono_specialization_key("depth", &[box_int32, i32.clone()], &i32);
        assert_eq!(key, "depth\u{1}adt:Box[s:i32,]\u{1}s:i32\u{1}\u{1}s:i32");
        let tensor = HostTypeTerm::Tensor(TensorType {
            dims: vec![DimInfo::Lit(4), DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        });
        let mut rendered = String::new();
        write_canonical_host_type_key(&tensor, &mut rendered);
        assert_eq!(rendered, "tensor[4,n,;f32]");

        let polymorphic = HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
            precision: HostPrecisionTerm::Variable("p".to_string()),
            shape: HostShapeTerm::Polymorphic(vec![
                crate::host_type_state::HostShapeSlot::RankVariable("pre".to_string()),
                crate::host_type_state::HostShapeSlot::Dim(DimInfo::Named("seq".to_string(), None)),
            ]),
        });
        rendered.clear();
        write_canonical_host_type_key(&polymorphic, &mut rendered);
        assert_eq!(
            rendered,
            "ptensor[poly(rank:pre,dim:named:seq,);precision-var:p]"
        );
        rendered.clear();
        write_canonical_host_type_key(
            &HostTypeTerm::InferenceVariable(HostInferenceVar(17)),
            &mut rendered,
        );
        assert_eq!(rendered, "iv:17");
    }

    /// An FNV-1a collision must fail loudly, never collapse two
    /// instantiations into one C definition.
    #[test]
    fn mono_symbol_collision_is_loud() {
        let mut state = MonoSpecializationState::default();
        register_mono_symbol_key(&mut state, "depth__mono_0123456789abcdef", "key-a")
            .expect("first registration succeeds");
        register_mono_symbol_key(&mut state, "depth__mono_0123456789abcdef", "key-a")
            .expect("re-registration of the same key is a no-op");
        let collision =
            register_mono_symbol_key(&mut state, "depth__mono_0123456789abcdef", "key-b");
        assert_eq!(collision, Err("key-a".to_string()));
    }

    /// A speculative lowering error reports no summary rejection and restores
    /// every field in the specialization state. The same error must surface
    /// when the function lowers outside the probe.
    #[test]
    fn failed_mono_probe_restores_state_and_defers_the_real_error() {
        let checked = surf_check(
            r#"
type Box[a] =
  | Empty
  | Full { value: a }
def bad_wrap() -> bool = bad(Empty)
def bad[b](box: Box[b]) -> bool =
  match box with {
    | Empty => true
    | Full { value: item } => bad(Empty)
  }
"#,
        );
        let i32 = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32));
        let seeded = MonoSpecializationState {
            memo: UnordMap::from([("seed-key".to_string(), "seed-symbol".to_string())]),
            symbol_keys: UnordMap::from([("seed-symbol".to_string(), "seed-key".to_string())]),
            functions: vec![LoweredHostFunction {
                function: HostFunction {
                    helper_result_claim_axes: Vec::new(),
                    name: "seeded__mono_0123456789abcdef".to_string(),
                    params: Vec::new(),
                    ret_ty: HostTypeTerm::Unit,
                    body: HostExpr::new(HostExprKind::Unit),
                    tensor_helpers: Vec::new(),
                    origin: HostFunctionOrigin::Monomorphized,
                    specialization: None,
                    summary_rejections: Vec::new(),
                },
                products: Vec::new(),
            }],
            in_progress: vec![InProgressMonoSpecialization {
                def_name: "seeded".to_string(),
                param_tys: vec![i32.clone()],
                ret_ty: i32,
            }],
        };
        MONO_SPECIALIZATIONS.with(|state| *state.borrow_mut() = seeded.clone());

        assert_eq!(
            top_level_fn_helper_summary_rejects(&HostLoweringSession::new(&checked), "bad_wrap"),
            Ok(false),
            "a speculative lowering error is not a summary rejection"
        );
        MONO_SPECIALIZATIONS.with(|state| {
            let restored = state.borrow();
            assert_eq!(restored.memo, seeded.memo);
            assert_eq!(restored.symbol_keys, seeded.symbol_keys);
            assert_eq!(
                format!("{:#?}", restored.functions),
                format!("{:#?}", seeded.functions),
                "the complete emitted-function state must be restored"
            );
            assert_eq!(restored.in_progress.len(), seeded.in_progress.len());
            for (restored, expected) in restored.in_progress.iter().zip(&seeded.in_progress) {
                assert_eq!(restored.def_name, expected.def_name);
                assert_eq!(restored.param_tys, expected.param_tys);
                assert_eq!(restored.ret_ty, expected.ret_ty);
            }
        });

        let body = find_top_level_def_expr(checked.exprs(), "bad_wrap")
            .expect("the checked program contains bad_wrap");
        let error = lower_host_function(
            "bad_wrap",
            body,
            None,
            &HostLoweringSession::new(&checked),
            false,
            false,
        )
        .expect_err("real lowering must report the genuine bad call");
        assert_eq!(
            error.message,
            "unsupported: Deep expression `app` on host expression lowering: generic host \
             call `bad` has no concrete checked type application to specialize \
             (chelis#1226; [05-UNS-1]) (lowering); deliberate [04-TOT-3]: a malformed or \
             unhandled Deep form cannot lower to a substitute host value"
        );
        MONO_SPECIALIZATIONS.with(|state| *state.borrow_mut() = MonoSpecializationState::default());
    }

    /// A nested probe's already-returned error retains the existing deferral.
    /// The public source still rejects in real lowering (API/CLI regression).
    #[test]
    fn issue_1922_nested_returned_probe_error_retains_deferral_and_cleanup() {
        assert_issue_1922_summary_probe("sink_output", false);
    }

    /// A fatal unwind raised directly in this probe returns the original
    /// diagnostic only after cleanup, and never installs a false cache row.
    #[test]
    fn issue_1922_direct_fatal_summary_probe_restores_state_before_returning_error() {
        assert_issue_1922_summary_probe("causal_sdpa_with_sink", true);
    }

    fn assert_issue_1922_summary_probe(name: &str, raises_here: bool) {
        let checked = surf_check(include_str!(
            "../../../tests/support/helper_summary_fatal.ch"
        ));
        let session = HostLoweringSession::new(&checked);
        let _restore = MonoProbeGuard::begin();
        MONO_SPECIALIZATIONS.with(|state| {
            state
                .borrow_mut()
                .memo
                .insert("seed-key".into(), "seed-symbol".into());
        });
        let seeded = MONO_SPECIALIZATIONS.with(|state| state.borrow().clone());
        for _ in 0..2 {
            let caught =
                std::panic::catch_unwind(|| top_level_fn_helper_summary_rejects(&session, name));
            if let Err(payload) = &caught {
                eprintln!(
                    "escaped diagnostic: {:?}",
                    payload.downcast_ref::<crate::lower::LowerDiagnostic>()
                );
            }
            let result = caught.expect("summary probe must not unwind");
            if raises_here {
                let error = result.expect_err("direct fatal diagnostic must not be deferred");
                assert!(error.fatal, "{error:?}");
                assert_eq!(
                    error.message,
                    "`insert` size resolves to `seq`, but no in-scope tensor axis supplies that extent. Use an i64 literal or a shape(tensor, i32-axis) read. Tracked by Chelis-Lang/chelis#469"
                );
                assert_eq!(error.span_id.as_deref(), Some("surf:471..509"));
            } else {
                assert_eq!(
                    result,
                    Ok(false),
                    "preserve already-returned error deferral"
                );
            }
            assert!(!is_inlining("sink_output"));
            assert!(!is_inlining("causal_sdpa_with_sink"));
            assert_eq!(
                session
                    .facts
                    .helper_summary_rejects
                    .borrow()
                    .get(name)
                    .copied(),
                if raises_here { None } else { Some(false) }
            );
            MONO_SPECIALIZATIONS.with(|state| {
                let restored = state.borrow();
                assert_eq!(restored.memo, seeded.memo);
                assert_eq!(restored.symbol_keys, seeded.symbol_keys);
                assert_eq!(
                    format!("{:#?}", restored.functions),
                    format!("{:#?}", seeded.functions)
                );
                assert_eq!(restored.in_progress.len(), seeded.in_progress.len());
            });
        }
    }

    #[test]
    fn named_tensor_entry_exposes_only_source_fixed_execution() {
        let fixed = surf_check(
            r#"
def main(x: tensor[4, f32]) -> tensor[4, f32] = with seed(0i64) {
  dropout(x, 0.5f32)
}
"#,
        );
        let plan = lower_named_tensor_entry_execution_plan(&fixed, "main")
            .expect("fixed entry lowers")
            .expect("fixed entry retains its execution plan");
        plan.verify_ownership()
            .expect("entry plan validates and seals against its DAG");

        let ordinary = surf_check("def main(x: tensor[4, f32]) -> tensor[4, f32] = add(x, x)");
        assert!(
            lower_named_tensor_entry_execution_plan(&ordinary, "main")
                .expect("ordinary entry classification")
                .is_none(),
            "a no-Dropout entry stays on the legacy DAG path"
        );
    }

    #[test]
    fn named_execution_plan_keeps_authored_unused_interface_obligations() {
        let checked = surf_check(
            r#"
def main[n](x: tensor[n, f32], unused: tensor[n, f32]) -> tensor[n, f32] = with seed(0i64) {
  dropout(x, 0.5f32)
}
"#,
        );
        let plan = lower_named_tensor_entry_execution_plan(&checked, "main")
            .expect("fixed entry lowers")
            .expect("fixed entry retains its execution plan");
        let loads = plan
            .dag_for_inspection()
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                crate::dag::RiscOp::Load { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            loads,
            vec!["x", "unused"],
            "an authored repeated-dimension obligation keeps an unread parameter"
        );
        assert!(
            plan.dag_for_inspection()
                .nodes()
                .iter()
                .any(|node| matches!(
                    &node.op,
                    crate::dag::RiscOp::ExtentWitness { parameter, claims, .. }
                        if parameter == "unused" && !claims.is_empty()
                ))
        );
        plan.verify_ownership()
            .expect("authored interface plan remains sealed");
    }

    #[test]
    fn named_tensor_entry_rejects_runtime_dropout_control() {
        let checked = surf_check(
            r#"
def main(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = with seed(0i64) {
  dropout(x, rate)
}
"#,
        );
        let diagnostic = lower_named_tensor_entry_execution_plan(&checked, "main")
            .expect_err("runtime rate has no fixed execution plan");
        assert!(diagnostic.fatal, "unsupported Dropout control stays loud");
        assert!(diagnostic.message.contains("not fixed-control"));
    }

    fn synthetic_tensor_function(name: &str, origin: HostFunctionOrigin) -> ConcreteHostFunction {
        let tensor = || {
            ConcreteHostType::Tensor(TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            })
        };
        HostFunction {
            helper_result_claim_axes: Vec::new(),
            name: name.to_string(),
            params: vec![HostParam {
                name: "x".to_string(),
                ty: tensor(),
            }],
            ret_ty: tensor(),
            body: HostExpr::new(HostExprKind::Unit),
            tensor_helpers: Vec::new(),
            origin,
            specialization: None,
            summary_rejections: Vec::new(),
        }
    }

    /// A tensor-shaped specialization must not displace the authored
    /// preferred tensor entry (the PR #1202-measured ABI flip).
    #[test]
    fn preferred_tensor_entry_ignores_specializations() {
        let mut program = ConcreteHostProgram::default();
        program.functions.push(synthetic_tensor_function(
            "entry",
            HostFunctionOrigin::Authored,
        ));
        program.functions.push(synthetic_tensor_function(
            "tloop__mono_0123456789abcdef",
            HostFunctionOrigin::Monomorphized,
        ));
        assert_eq!(preferred_tensor_entry_name(&program), Some("entry"));
    }

    #[test]
    fn tensor_signature_classification_ignores_specializations() {
        let mut program = ConcreteHostProgram::default();
        program.functions.push(synthetic_tensor_function(
            "entry",
            HostFunctionOrigin::Authored,
        ));
        program.functions.push(synthetic_tensor_function(
            "tloop__mono_0123456789abcdef",
            HostFunctionOrigin::Monomorphized,
        ));
        assert!(function_has_tensor_signature(&program, "entry"));
        assert!(!function_has_tensor_signature(
            &program,
            "tloop__mono_0123456789abcdef"
        ));
    }

    fn parse_and_check(src: &str) -> CheckedProgram {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let checked = chelis_types::check_ir_program(&exprs)
            .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"))
    }

    fn surf_check(src: &str) -> CheckedProgram {
        let decls = chelis_surf::parser::parse_str(src).expect("surf parse failed");
        let deep =
            chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
        chelis_types::check_ir_program(&deep)
            .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors))
    }

    /// chelis#2181: a callee that takes a function-typed parameter, whose
    /// body is a chain of `binders` `let` bindings.
    ///
    /// The C lane has no function-pointer representation, so this callee is
    /// inlined at its call site rather than emitted as a C function. That
    /// makes the inliner's cost, not the emitter's, the quantity under test.
    fn issue_2181_binder_chain_source(binders: usize) -> String {
        let mut lines = vec![
            "module Demo.BinderChain".to_string(),
            "def sq(x: f32) -> f32 = mul(x, x)".to_string(),
            "def body(f: f32 -> f32, x: f32) -> f32 = {".to_string(),
            "  v1 = f(x)".to_string(),
        ];
        for index in 2..=binders {
            lines.push(format!("  v{index} = add(v{prev}, x)", prev = index - 1));
        }
        lines.push(format!("  v{binders}"));
        lines.push("}".to_string());
        lines.push("def main() -> f32 = body(sq, cast(2.0, f32))".to_string());
        lines.join("\n") + "\n"
    }

    /// A chain of mutually recursive defs, each calling the next from two
    /// argument positions. The kernel-decision probe expands that call graph
    /// as a tree, so an unmemoized run costs 2^depth summary builds while a
    /// memoized one costs one per definition.
    fn issue_1829_fanout_source(depth: usize) -> String {
        let mut lines = vec!["module Demo.Fanout".to_string(), String::new()];
        for level in 0..depth {
            lines.push(format!(
                "def f{level}(x: i64) -> i64 = add(f{next}(x), f{next}(x))",
                next = level + 1
            ));
        }
        lines.push(format!("def f{depth}(x: i64) -> i64 = x"));
        lines.push(String::new());
        lines.join("\n")
    }

    /// chelis#1829 class receipt, reshaped for the session. The interpreter
    /// reaches `host_def_kernel` once per applied definition, and the summary
    /// probes behind those calls must be bounded by the definition count
    /// rather than expanding the call graph as a tree.
    ///
    /// What this pins that `issue_1835_kernel_decision_work_is_linear` does
    /// not: the always-compiled probe counter and the `cfg(test)` profile
    /// count the SAME misses. The counted receipt in `chelis-compiler-api`
    /// reads only the former, so a divergence there would let that receipt
    /// pass on the wrong number.
    ///
    /// Evidentiary status: REGRESSION TEST against the base sha, where
    /// `host_def_kernel` took a bare program, nothing armed the memo, and
    /// this fixture cost 398,574 builds for thirteen definitions. It was a
    /// disposition lock on chelis#1843's armed scope; the session makes the
    /// same assertion hold with no scope to arm.
    #[test]
    fn issue_1829_interpreter_session_bounds_kernel_decision_probes() {
        let depth = 12;
        std::thread::Builder::new()
            .name("issue-1829-fanout".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let checked = surf_check(&issue_1829_fanout_source(depth));
                let names = (0..=depth)
                    .map(|level| format!("f{level}"))
                    .collect::<Vec<_>>();
                reset_host_work_profile();
                reset_host_summary_probe_builds();
                {
                    let session = HostLoweringSession::new(&checked);
                    for name in &names {
                        host_def_kernel(&session, name, None)
                            .expect("#1829 fixture reaches a kernel decision");
                    }
                }
                let profile = take_host_work_profile();
                let probes = host_summary_probe_builds();
                eprintln!(
                    "#1829 depth={depth} defs={} helper_summary_builds={} probe_builds={probes}",
                    names.len(),
                    profile.helper_summary_builds
                );
                assert!(
                    profile.helper_summary_builds <= names.len(),
                    "#1829: a session must build at most one summary per definition; \
                     {} definitions produced {} builds",
                    names.len(),
                    profile.helper_summary_builds
                );
                assert_eq!(
                    probes,
                    u64::try_from(profile.helper_summary_builds).expect("probe count fits"),
                    "#1829: the always-compiled probe counter and the test-only profile \
                     must count the same misses, or a downstream receipt reads the wrong number"
                );
            })
            .expect("#1829 probe thread starts")
            .join()
            .expect("#1829 probe thread completes");
    }

    /// chelis#2181: inlining a callable-parameter callee is not exponential in
    /// its binder count.
    ///
    /// The claim is deliberately "not exponential" rather than "linear":
    /// substitution work is superlinear for other shapes (a lambda every
    /// fourth binder measures about 7.4n^2 on this tree), which this test
    /// does not and should not assert against.
    ///
    /// `substitute_expr`'s `let` arm used to substitute every child into a
    /// `elements` vector, then discard it and redo the work as `rebuilt` for
    /// the ordinary two-child `let`. Every nested binder therefore substituted
    /// its own subtree twice, so one inline cost 2^(binders) and a 40-binder
    /// callee such as nautilus `Roots.brent_rec` never finished.
    ///
    /// The assertion is a ratio rather than an absolute count, so it states
    /// the asymptotic promise directly and needs no machine-specific budget:
    /// doubling the binder count may at most triple the substitution work.
    ///
    /// Evidentiary status: REGRESSION TEST, established by restoring the
    /// discarded pass on this tree and keeping this counter. Unfixed, 8
    /// binders cost 6,119 substituted nodes and 16 cost 1,572,839, a ratio of
    /// 257.0x = 2^8, and this assertion fails. Fixed, the same fixture costs
    /// a ratio near 2.
    #[test]
    fn issue_2181_callable_parameter_inlining_is_not_exponential_in_binder_count() {
        fn substitution_nodes(binders: usize) -> usize {
            let checked = surf_check(&issue_2181_binder_chain_source(binders));
            let lowered = top_level_lowering_map(checked.exprs(), checked.type_env());
            reset_host_work_profile();
            lower_host_program(&HostLoweringSession::new(&checked), &lowered)
                .expect("#2181 fixture must lower");
            take_host_work_profile().substitution_nodes
        }
        let small = substitution_nodes(8);
        let doubled = substitution_nodes(16);
        assert!(
            doubled <= small.saturating_mul(3),
            "#2181: doubling the binder count must at most triple substitution \
             work; 8 binders cost {small} nodes and 16 cost {doubled}, a \
             factor of {factor}. An exponential inliner shows about 2^8 here.",
            factor = doubled / small.max(1),
        );
    }

    /// chelis#1829 hazard (1), restated for the session. The interpreter holds
    /// one session across a whole evaluation, and a whole-program lowering
    /// reached from inside it builds a second. Under the thread-local memo
    /// that inner entry could disarm the flag and clear every cache, so the
    /// hazard needed a refcount to describe. Two sessions share nothing, so
    /// the property to hold is now the simpler one: an inner session neither
    /// clears the enclosing session's facts nor reads them.
    ///
    /// Evidentiary status: REGRESSION TEST against the base sha, where both
    /// programs' facts lived in one thread-local map and the outer program's
    /// rows were observable from the inner entry (and, before chelis#1843's
    /// refcount, cleared by its drop). It replaces
    /// `issue_1829_nested_scope_keeps_the_outer_memo_armed_and_populated`,
    /// whose flag and refcount no longer exist.
    #[test]
    fn issue_1829_a_nested_session_neither_clears_nor_reads_the_enclosing_one() {
        let outer = surf_check("module Demo.Outer\n\ndef outer_fn(x: i64) -> i64 = x\n");
        let inner = surf_check("module Demo.Inner\n\ndef inner_fn(x: i64) -> i64 = x\n");
        let outer_session = HostLoweringSession::new(&outer);

        assert!(
            cached_program_defs(&outer_session).contains_key("outer_fn"),
            "#1829: the enclosing session must actually derive facts, or this \
             receipt would pass without measuring anything"
        );

        {
            let inner_session = HostLoweringSession::new(&inner);
            let inner_defs = cached_program_defs(&inner_session);
            assert!(
                inner_defs.contains_key("inner_fn") && !inner_defs.contains_key("outer_fn"),
                "#1829: an inner session's facts describe its own program only"
            );
            assert!(
                !cached_program_defs(&outer_session).contains_key("inner_fn"),
                "#1829: an inner session must not write into the enclosing one"
            );
        }

        // The inner session is gone. The enclosing one is untouched, and the
        // borrow checker, not a refcount, is what made that true.
        assert!(
            cached_program_defs(&outer_session).contains_key("outer_fn"),
            "#1829: an inner session's drop must not clear the enclosing session"
        );
    }

    /// chelis#1835 fixture: a fan-out chain of defs, each calling the next
    /// from two argument positions.
    ///
    /// Two shapes, because the kernel decision costs differently on each. A
    /// TENSOR-returning chain reaches the helper-summary probe and stays on
    /// the kernel path. A SCALAR-returning chain reaches the same probe
    /// first, because `def_body_decision_impl` asks it before the declared
    /// result type can rule the def out, and the probe's own lowering then
    /// takes the host lane, whose inliner duplicates the callee body per call
    /// site. The second shape is therefore the expensive one, and both are
    /// the same defect: a per-definition fact recomputed per ask.
    fn issue_1835_fanout_source(depth: usize, tensor_result: bool) -> String {
        let (module, ty, leaf) = if tensor_result {
            ("Demo.KernelFanout", "tensor[4, f32]", "mul(x, x)")
        } else {
            ("Demo.ScalarFanout", "i64", "mul(x, x)")
        };
        let mut lines = vec![format!("module {module}"), String::new()];
        for level in 0..depth {
            lines.push(format!(
                "def f{level}(x: {ty}) -> {ty} = add(f{next}(x), f{next}(x))",
                next = level + 1
            ));
        }
        lines.push(format!("def f{depth}(x: {ty}) -> {ty} = {leaf}"));
        lines.push(String::new());
        lines.join("\n")
    }

    /// Which defs the C lane lowered through the tensor DAG. A kernel body
    /// becomes a call to an extracted tensor helper attached to the emitted
    /// host function; a host-lane body has none.
    fn issue_1835_c_lane_kernel_defs(checked: &CheckedProgram, names: &[String]) -> Vec<bool> {
        let compiled = try_lower_compiled_program(checked).expect("#1835 fixture lowers on C");
        let host = compiled.host.expect("#1835 fixture emits a host program");
        names
            .iter()
            .map(|name| {
                host.functions
                    .iter()
                    .find(|function| &function.name == name)
                    .is_some_and(|function| !function.tensor_helpers.is_empty())
            })
            .collect()
    }

    /// chelis#1835 class receipt. `host_def_kernel` is the entry every lane
    /// asks for one def's kernel decision, and its cost must be linear in the
    /// definition count with NO opt-in: no flag to arm, no scope to hold, no
    /// argument beyond the session every caller must now build. A second full
    /// pass must add no work at all, and the answers must agree with the lane
    /// that compiles the whole program (chelis#1277 B2h).
    ///
    /// Evidentiary status: REGRESSION TEST for every assertion, watched red on
    /// the base sha `6abca2406`, where the memo is gated on a thread-local
    /// flag that no `host_def_kernel` caller arms. Thirteen definitions cost
    /// 78 summary builds and 728 definition collections on the tensor chain,
    /// and 398,574 builds and 797,174 collections on the scalar chain; the
    /// second pass repeated both figures exactly, which is the memo never
    /// being read. The pull request records the runs.
    #[test]
    fn issue_1835_kernel_decision_work_is_linear() {
        const DEPTH: usize = 12;
        for tensor_result in [true, false] {
            std::thread::Builder::new()
                .name(format!("issue-1835-fanout-{tensor_result}"))
                .stack_size(512 * 1024 * 1024)
                .spawn(move || {
                    let typed = surf_check(&issue_1835_fanout_source(DEPTH, tensor_result));
                    let effected = chelis_effects::check_program(&typed).expect("effect check");
                    let checked =
                        chelis_types::check_linearity(&effected).expect("linearity check");
                    let names = (0..=DEPTH)
                        .map(|level| format!("f{level}"))
                        .collect::<Vec<_>>();

                    // One session, established by the caller because the type
                    // leaves it no choice, and NO other opt-in.
                    let session = HostLoweringSession::new(&checked);
                    reset_host_work_profile();
                    let first_answers = names
                        .iter()
                        .map(|name| {
                            host_def_kernel(&session, name, None)
                                .expect("#1835 fixture reaches a kernel decision")
                                .is_some()
                        })
                        .collect::<Vec<_>>();
                    let first = take_host_work_profile();
                    let second_answers = names
                        .iter()
                        .map(|name| {
                            host_def_kernel(&session, name, None)
                                .expect("#1835 fixture reaches a kernel decision")
                                .is_some()
                        })
                        .collect::<Vec<_>>();
                    let second = take_host_work_profile();
                    eprintln!(
                        "#1835 tensor_result={tensor_result} depth={DEPTH} defs={} \
                         helper_summary_builds={}+{} program_def_collections={}+{}",
                        names.len(),
                        first.helper_summary_builds,
                        second.helper_summary_builds,
                        first.program_def_collections,
                        second.program_def_collections
                    );

                    assert!(
                        first.helper_summary_builds <= names.len(),
                        "#1835: an unopted `host_def_kernel` caller must build at most one \
                         summary per definition; {} definitions produced {} builds \
                         (tensor_result={tensor_result})",
                        names.len(),
                        first.helper_summary_builds
                    );
                    assert_eq!(
                        second.helper_summary_builds, 0,
                        "#1835: a second full pass over the same program must build no \
                         further summaries (tensor_result={tensor_result})"
                    );
                    assert_eq!(
                        first.program_def_collections + second.program_def_collections,
                        1,
                        "#1835: program definitions must be collected once for the session, \
                         not once per decision (tensor_result={tensor_result})"
                    );
                    assert_eq!(
                        first_answers, second_answers,
                        "#1835: a memoized decision must not differ from the one that built \
                         it (tensor_result={tensor_result})"
                    );
                    assert_eq!(
                        first_answers,
                        issue_1835_c_lane_kernel_defs(&checked, &names),
                        "#1835: the shared decision must answer what the C lane compiles \
                         (chelis#1277 B2h; tensor_result={tensor_result})"
                    );
                })
                .expect("#1835 fanout thread starts")
                .join()
                .expect("#1835 fanout thread completes");
        }

        // Both chains above answer uniformly, all-kernel or all-host, so their
        // cross-lane comparison cannot fail for a PER-DEFINITION disagreement:
        // two constant vectors of the same constant are equal whatever the
        // decision did. This program's answers are non-uniform by
        // construction, one definition per documented class, so the
        // comparison has something to disagree about.
        let mixed = surf_check(
            "module Demo.MixedLanes\n\n             def k(x: tensor[4, f32]) -> tensor[4, f32] = mul(x, x)\n             def loud(x: tensor[4, f32]) -> tensor[4, f32] ! {IO} = {\n               _ = print(x)\n  x\n}\n             def s(x: i64) -> i64 = add(x, x)\n             def k2(x: tensor[4, f32]) -> tensor[4, f32] = add(k(x), k(x))\n",
        );
        let mixed = chelis_types::check_linearity(
            &chelis_effects::check_program(&mixed).expect("mixed effect check"),
        )
        .expect("mixed linearity check");
        let mixed_names = ["k", "loud", "s", "k2"].map(String::from).to_vec();
        let mixed_session = HostLoweringSession::new(&mixed);
        let mixed_answers = mixed_names
            .iter()
            .map(|name| {
                host_def_kernel(&mixed_session, name, None)
                    .expect("#1835 mixed fixture reaches a kernel decision")
                    .is_some()
            })
            .collect::<Vec<_>>();
        eprintln!("#1835 mixed answers={mixed_answers:?}");
        assert!(
            mixed_answers.iter().any(|kernel| *kernel)
                && mixed_answers.iter().any(|kernel| !*kernel),
            "#1835: the mixed fixture must answer non-uniformly, or the cross-lane \
             comparison below is two constant vectors again; got {mixed_answers:?}"
        );
        assert_eq!(
            mixed_answers,
            vec![true, false, false, true],
            "#1835: a pure tensor body is a kernel, an IO effect row and a non-tensor \
             result are host code (chelis#1277 B2h's documented classes)"
        );
        assert_eq!(
            mixed_answers,
            issue_1835_c_lane_kernel_defs(&mixed, &mixed_names),
            "#1835: the shared decision must answer what the C lane compiles, per \
             definition and not merely in aggregate (chelis#1277 B2h)"
        );
    }

    fn issue_1205_source(operations: usize, flat: bool) -> String {
        let module = if flat { "Flat" } else { "Nested" };
        let mut lines = vec![
            format!("module FrontEndPerformance.{module}N{operations}"),
            "def bc(c: f32) -> tensor[8, f32] = \
             reshape(insert(to_tensor([c]), 0, 8i64), [8i64])"
                .to_string(),
        ];
        if flat {
            lines.push(
                "def st(s: tensor[8, f32], i: i64) -> tensor[8, f32] = \
                 if gte(i, 5i64) then s else {"
                    .to_string(),
            );
            let mut previous = "s".to_string();
            for index in 0..operations {
                lines.push(format!(
                    "  t{index} = mul(add({previous}, bc(cast(1.0, f32))), \
                     bc(cast(0.5, f32)))"
                ));
                previous = format!("t{index}");
            }
            lines.extend([format!("  st({previous}, add(i, 1i64))"), "}".to_string()]);
        } else {
            let mut body = "s".to_string();
            for _ in 0..operations {
                body = format!("mul(add({body}, bc(cast(1.0, f32))), bc(cast(0.5, f32)))");
            }
            lines.push(format!(
                "def st(s: tensor[8, f32], i: i64) -> tensor[8, f32] = \
                 if gte(i, 5i64) then s else st({body}, add(i, 1i64))"
            ));
        }
        lines.push("r = index(to_list(st(bc(cast(1.0, f32)), 0i64)), 0i64)".to_string());
        lines.join("\n") + "\n"
    }

    fn issue_1205_host_profile(operations: usize, flat: bool) -> HostWorkProfile {
        std::thread::Builder::new()
            .name(format!("issue-1205-host-{operations}"))
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let typed = surf_check(&issue_1205_source(operations, flat));
                let effected = chelis_effects::check_program(&typed).expect("effect check");
                let checked = chelis_types::check_linearity(&effected).expect("linearity check");
                reset_host_work_profile();
                try_lower_compiled_program(&checked).expect("#1205 fixture lowers");
                take_host_work_profile()
            })
            .expect("#1205 host profile thread starts")
            .join()
            .expect("#1205 host profile thread completes")
    }

    fn issue_1205_find_named_app<'expr>(expr: &'expr Expr, expected: &str) -> Option<&'expr Expr> {
        if let Some((DeepTag::App, _, kids)) = stamped_parts(expr)
            && let Some((DeepTag::Var, _, callee_kids)) = kids.first().and_then(stamped_parts)
            && callee_kids.first().and_then(symbol_name) == Some(expected)
        {
            return Some(expr);
        }
        match expr {
            Expr::List(list, _) => list
                .elements
                .iter()
                .find_map(|child| issue_1205_find_named_app(child, expected)),
            Expr::Node(node, _) => node
                .children_slice()
                .iter()
                .find_map(|child| issue_1205_find_named_app(child, expected)),
            Expr::MetaExpr(meta, _) => issue_1205_find_named_app(&meta.expr, expected),
            Expr::BareList(elements, _) => elements
                .iter()
                .find_map(|child| issue_1205_find_named_app(child, expected)),
            Expr::UnknownForm(data) => data
                .children
                .iter()
                .find_map(|child| issue_1205_find_named_app(child, expected)),
            Expr::Map(map, _) => {
                map.find_expression(|value| issue_1205_find_named_app(value, expected))
            }
            Expr::Atom(_, _) => None,
        }
    }

    #[test]
    fn issue_1205_host_lowering_work_is_linear() {
        for flat in [false, true] {
            let mut rows = Vec::new();
            for operations in [20, 40, 80, 160] {
                let profile = issue_1205_host_profile(operations, flat);
                let total_work = profile.host_expr_visits
                    + profile.app_clone_nodes
                    + profile.tensor_helper_input_nodes
                    + profile.tensor_helper_preflight_nodes
                    + profile.callable_scope_work
                    + profile.grad_scan_nodes
                    + profile.fail_scan_nodes
                    + profile.program_def_clone_nodes
                    + profile.type_env_clone_nodes;
                eprintln!(
                    "#1205 host shape={} n={operations} profile={profile:?} total_work={total_work}",
                    if flat { "flat" } else { "nested" }
                );
                assert_eq!(
                    profile.program_def_collections, 1,
                    "#1205 program definitions must be collected once per compiled program"
                );
                assert_eq!(
                    profile.helper_summary_builds, 2,
                    "#1205 helper summaries must be cached per definition"
                );
                assert_eq!(
                    profile.tensor_helper_builtin_load_rejections, 0,
                    "#1205 dynamic to_tensor candidates must be rejected before DAG lowering"
                );
                assert_eq!(
                    profile.tensor_helper_preflight_lookup_misses, 0,
                    "#1205 every tensor-helper candidate must have precomputed facts"
                );
                assert!(
                    profile.tensor_helper_preflight_rejections > 0,
                    "#1205 fixture must exercise the dynamic to_tensor preflight"
                );
                rows.push((operations, total_work, profile));
            }
            for pair in rows.windows(2) {
                let (previous_n, previous, previous_profile) = pair[0];
                let (current_n, current, profile) = pair[1];
                assert_eq!(
                    profile.app_clone_nodes, previous_profile.app_clone_nodes,
                    "#1205 descendant app cloning must not grow with expression nesting"
                );
                assert!(
                    current * 10 <= previous * 22,
                    "#1205 host work must grow linearly: shape={} {previous_n}->{current_n} \
                     previous={previous} current={current} profile={profile:?}",
                    if flat { "flat" } else { "nested" }
                );
            }
        }
    }

    #[test]
    fn issue_1205_callable_scope_work_is_linear() {
        std::thread::Builder::new()
            .name("issue-1205-nested-callable-scopes".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                for nested_fns in [true, false] {
                    let mut rows = Vec::new();
                    for depth in [20, 40, 80, 160] {
                        let body = if nested_fns {
                            let mut body = "bc(c)".to_string();
                            for index in (0..depth).rev() {
                                body = format!("(fn (p{index}: f32) -> {body})(c)");
                            }
                            body
                        } else {
                            let mut body = format!("f{}(c)", depth - 1);
                            for index in (0..depth).rev() {
                                let target = if index == 0 {
                                    "bc".to_string()
                                } else {
                                    format!("f{}", index - 1)
                                };
                                body = format!("{{ f{index} = {target}\n{body}\n}}");
                            }
                            body
                        };
                        let shape = if nested_fns { "Fn" } else { "Let" };
                        let source = format!(
                            "module FrontEndPerformance.Nested{shape}{depth}\n\
                             def bc(c: f32) -> tensor[1, f32] = to_tensor([c])\n\
                             def nested(c: f32) -> tensor[1, f32] = {body}\n\
                             out = index(to_list(nested(cast(1.0, f32))), 0i64)\n"
                        );
                        let typed = surf_check(&source);
                        let defs = cached_program_defs(&HostLoweringSession::new(&typed));
                        let nested_body = lookup_program_def(&defs, "nested")
                            .expect("nested definition is present");

                        reset_host_work_profile();
                        let summaries = cached_dynamic_to_tensor_def_summaries(
                            &HostLoweringSession::new(&typed),
                        );
                        assert_eq!(summaries.get("nested"), Some(&true));
                        let summary_profile = take_host_work_profile();
                        let summary_work = summary_profile.tensor_helper_preflight_nodes
                            + summary_profile.callable_scope_work;

                        reset_host_work_profile();
                        let mut facts = UnordMap::new();
                        let nested_facts =
                            analyze_tensor_helper_preflight(nested_body, &summaries, &mut facts);
                        assert!(nested_facts.reaches_dynamic_to_tensor);
                        let expression_profile = take_host_work_profile();
                        let expression_work = expression_profile.tensor_helper_preflight_nodes
                            + expression_profile.callable_scope_work;

                        eprintln!(
                            "#1205 callable scopes shape={shape} depth={depth} \
                             summary={summary_profile:?} summary_work={summary_work} \
                             expression={expression_profile:?} expression_work={expression_work}"
                        );
                        rows.push((depth, summary_work, expression_work));
                    }
                    for pair in rows.windows(2) {
                        let (previous_depth, previous_summary, previous_expression) = pair[0];
                        let (current_depth, current_summary, current_expression) = pair[1];
                        for (path, previous_work, current_work) in [
                            ("definition summary", previous_summary, current_summary),
                            (
                                "per-expression facts",
                                previous_expression,
                                current_expression,
                            ),
                        ] {
                            assert!(
                                current_work * 10 <= previous_work * 23,
                                "#1205 callable-scope work must grow linearly: \
                                 shape={} path={path} depth {previous_depth}->{current_depth} \
                                 previous={previous_work} current={current_work}",
                                if nested_fns { "Fn" } else { "Let" }
                            );
                        }
                    }
                }
            })
            .expect("#1205 nested callable scope profile thread starts")
            .join()
            .expect("#1205 nested callable scope profile thread completes");
    }

    #[test]
    fn issue_1205_callable_scope_restore_semantics() {
        let mut direct = CallableScope::default();
        let outer = direct.bind("f".to_string(), Some("bc".to_string()));
        let duplicate = direct.bind("f".to_string(), Some("other".to_string()));
        let inner = direct.bind("f".to_string(), None);
        direct.restore([inner, duplicate, outer]);
        assert!(
            !direct.contains_key("f"),
            "restoring repeated shadow bindings must recover the absent outer scope"
        );

        let typed = surf_check(
            "module FrontEndPerformance.ScopeRestore\n\
             def bc(c: f32) -> tensor[1, f32] = to_tensor([c])\n\
             def sibling(c: f32) -> tensor[1, f32] = {\n\
               probe = (fn (bc: f32) -> bc)(c)\n\
               bc(c)\n\
             }\n\
             def nested(c: f32) -> tensor[1, f32] = {\n\
               f = bc\n\
               probe = (fn (f: f32) -> (fn (f: f32) -> f)(f))(c)\n\
               f(c)\n\
             }\n\
             def duplicate(c: f32) -> tensor[1, f32] = {\n\
               f = bc\n\
               f = f\n\
               f(c)\n\
             }\n\
             def sequential(c: f32) -> tensor[1, f32] = {\n\
               f = bc\n\
               g = f\n\
               h = g\n\
               h(c)\n\
             }\n\
             out = index(to_list(sibling(cast(1.0, f32))), 0i64)\n",
        );
        let summaries = cached_dynamic_to_tensor_def_summaries(&HostLoweringSession::new(&typed));
        let defs = cached_program_defs(&HostLoweringSession::new(&typed));
        for (definition, final_callee) in [
            ("sibling", "bc"),
            ("nested", "f"),
            ("duplicate", "f"),
            ("sequential", "h"),
        ] {
            assert_eq!(
                summaries.get(definition),
                Some(&true),
                "lexical shadow restoration must preserve the final dynamic call in {definition}"
            );
            let body = lookup_program_def(&defs, definition).expect("fixture definition exists");
            let final_app = issue_1205_find_named_app(body, final_callee)
                .expect("fixture contains its final named application");
            let mut facts = UnordMap::new();
            let body_facts = analyze_tensor_helper_preflight(body, &summaries, &mut facts);
            assert!(
                body_facts.reaches_dynamic_to_tensor,
                "per-expression analysis must preserve the final call in {definition}"
            );
            assert!(
                facts[&(final_app as *const Expr as usize)].reaches_dynamic_to_tensor,
                "the final {final_callee} application in {definition} must inherit its dynamic summary"
            );
        }
    }

    #[test]
    fn issue_1205_preflight_preserves_static_to_tensor_literals() {
        let typed = surf_check(
            "module FrontEndPerformance.StaticLiteral\n\
             def lit() -> tensor[2, f32] = \
               to_tensor([cast(1.0, f32), cast(2.0, f32)])\n\
             r = index(to_list(lit()), 0i64)\n",
        );
        fn find_to_tensor(expr: &Expr) -> Option<&Expr> {
            if let Some((DeepTag::App, _, kids)) = stamped_parts(expr)
                && let Some((DeepTag::Var, _, callee_kids)) = kids.first().and_then(stamped_parts)
                && callee_kids.first().and_then(symbol_name) == Some("to_tensor")
            {
                return Some(expr);
            }
            match expr {
                Expr::List(list, _) => list.elements.iter().find_map(find_to_tensor),
                Expr::Node(node, _) => node.children_slice().iter().find_map(find_to_tensor),
                Expr::MetaExpr(meta, _) => find_to_tensor(&meta.expr),
                Expr::BareList(elements, _) => elements.iter().find_map(find_to_tensor),
                Expr::UnknownForm(data) => data.children.iter().find_map(find_to_tensor),
                Expr::Map(map, _) => map.find_expression(|value| find_to_tensor(value)),
                Expr::Atom(_, _) => None,
            }
        }
        let literal = typed
            .exprs()
            .iter()
            .find_map(find_to_tensor)
            .expect("fixture contains a to_tensor application");
        assert!(
            crate::lower::is_static_to_tensor_literal(literal),
            "fixture must exercise the lowerer's literal recognizer"
        );
        assert!(
            !expr_is_runtime_shaped_to_tensor(literal),
            "the preflight must share the lowerer's static-literal exemption"
        );
        let effected = chelis_effects::check_program(&typed).expect("effect check");
        let checked = chelis_types::check_linearity(&effected).expect("linearity check");
        reset_host_work_profile();
        try_lower_compiled_program(&checked).expect("static to_tensor fixture lowers");
        let profile = take_host_work_profile();
        assert_eq!(
            profile.tensor_helper_preflight_rejections, 0,
            "the runtime-shaped preflight must not reject a static literal: {profile:?}"
        );
        assert_eq!(profile.tensor_helper_builtin_load_rejections, 0);
    }

    #[test]
    fn issue_1205_preflight_does_not_confuse_shadowed_operands_with_calls() {
        let typed = surf_check(
            "module FrontEndPerformance.ShadowedOperand\n\
             def bc(c: f32) -> tensor[1, f32] = to_tensor([c])\n\
             def twice(bc: tensor[1, f32]) -> tensor[1, f32] = add(bc, bc)\n\
             r = index(to_list(twice(to_tensor([cast(1.0, f32)]))), 0i64)\n",
        );
        let summaries = cached_dynamic_to_tensor_def_summaries(&HostLoweringSession::new(&typed));
        assert_eq!(
            summaries.get("bc"),
            Some(&true),
            "the top-level bc definition must exercise the transitive summary"
        );

        let add = typed
            .exprs()
            .iter()
            .find_map(|expr| issue_1205_find_named_app(expr, "add"))
            .expect("fixture contains the twice body");
        let mut facts = UnordMap::new();
        let add_facts = analyze_tensor_helper_preflight(add, &summaries, &mut facts);
        assert!(
            !add_facts.reaches_dynamic_to_tensor,
            "a tensor operand named bc is not a call to the top-level bc definition"
        );
    }

    #[test]
    fn issue_1205_preflight_tracks_callable_shadowing_and_aliases() {
        let shadowed = surf_check(
            "module FrontEndPerformance.ShadowedCallable\n\
             def bc(c: f32) -> tensor[1, f32] = to_tensor([c])\n\
             def local(x: tensor[1, f32]) -> tensor[1, f32] = {\n\
               bc = fn (y: tensor[1, f32]) -> add(y, y)\n\
               bc(x)\n\
             }\n\
             def wrapper(x: tensor[1, f32]) -> tensor[1, f32] = local(x)\n\
             out = index(to_list(wrapper(to_tensor([cast(1.0, f32)]))), 0i64)\n",
        );
        let shadowed_summaries =
            cached_dynamic_to_tensor_def_summaries(&HostLoweringSession::new(&shadowed));
        assert_eq!(shadowed_summaries.get("bc"), Some(&true));
        for name in ["local", "wrapper", "out"] {
            assert_eq!(
                shadowed_summaries.get(name),
                Some(&false),
                "a local callable named bc must shadow the top-level helper in {name}"
            );
        }
        let shadowed_defs = cached_program_defs(&HostLoweringSession::new(&shadowed));
        let local_body = lookup_program_def(&shadowed_defs, "local").expect("local definition");
        let local_call = issue_1205_find_named_app(local_body, "bc").expect("local bc call");
        let mut shadowed_facts = UnordMap::new();
        analyze_tensor_helper_preflight(local_body, &shadowed_summaries, &mut shadowed_facts);
        assert!(
            !shadowed_facts[&(local_call as *const Expr as usize)].reaches_dynamic_to_tensor,
            "the local bc call must not inherit the top-level bc summary"
        );

        let aliased = surf_check(
            "module FrontEndPerformance.AliasedCallable\n\
             def bc(c: f32) -> tensor[1, f32] = to_tensor([c])\n\
             def aliased(c: f32) -> tensor[1, f32] = {\n\
               f = bc\n\
               f(c)\n\
             }\n\
             def wrapper(c: f32) -> tensor[1, f32] = aliased(c)\n\
             out = index(to_list(wrapper(cast(1.0, f32))), 0i64)\n",
        );
        let aliased_summaries =
            cached_dynamic_to_tensor_def_summaries(&HostLoweringSession::new(&aliased));
        for name in ["bc", "aliased", "wrapper", "out"] {
            assert_eq!(
                aliased_summaries.get(name),
                Some(&true),
                "the alias f = bc must propagate the dynamic helper summary through {name}"
            );
        }
        let aliased_defs = cached_program_defs(&HostLoweringSession::new(&aliased));
        let aliased_body =
            lookup_program_def(&aliased_defs, "aliased").expect("aliased definition");
        let alias_call = issue_1205_find_named_app(aliased_body, "f").expect("local f call");
        let mut aliased_facts = UnordMap::new();
        analyze_tensor_helper_preflight(aliased_body, &aliased_summaries, &mut aliased_facts);
        assert!(
            aliased_facts[&(alias_call as *const Expr as usize)].reaches_dynamic_to_tensor,
            "the local f call must inherit the aliased top-level bc summary"
        );

        let effected = chelis_effects::check_program(&aliased).expect("effect check");
        let checked = chelis_types::check_linearity(&effected).expect("linearity check");
        reset_host_work_profile();
        try_lower_compiled_program(&checked).expect("aliased fixture lowers");
        let profile = take_host_work_profile();
        assert_eq!(
            profile.tensor_helper_builtin_load_rejections, 0,
            "aliased dynamic helpers must be rejected by preflight, not after DAG lowering: {profile:?}"
        );
    }

    /// harden-bounded-monomorphization D4: the interning identity is the
    /// definition's own name, so a short and a qualified spelling of one def
    /// produce one canonical key and one minted symbol per instantiation.
    #[test]
    fn mono_interning_identity_is_the_definitions_own_name() {
        let checked = parse_and_check(
            "(defsig {} Demo.depth (a) (t-fn {} (t-var {} a) (t-prim {} i32)))\n\
             (def {} Demo.depth\n\
               (fn {} (params {} (x {type: (t-var {} a)}))\n\
                 (lit {type: (t-prim {} i32)} 1)))\n",
        );
        let (canonical_short, _) = find_top_level_def_named(checked.exprs(), "depth")
            .expect("short spelling resolves to the def");
        let (canonical_qualified, _) = find_top_level_def_named(checked.exprs(), "Demo.depth")
            .expect("qualified spelling resolves to the def");
        assert_eq!(canonical_short, "Demo.depth");
        assert_eq!(canonical_qualified, "Demo.depth");
        let i32 = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32));
        let key = mono_specialization_key(canonical_short, std::slice::from_ref(&i32), &i32);
        assert_eq!(
            key,
            mono_specialization_key(canonical_qualified, std::slice::from_ref(&i32), &i32)
        );
        let symbol = mono_specialization_symbol(canonical_short, &key);
        assert!(symbol.starts_with("Demo.depth__mono_"));
        assert_eq!(symbol.len(), "Demo.depth__mono_".len() + 16);
    }

    #[test]
    fn adt_constructor_layouts_come_from_checked_registry() {
        let checked = surf_check(
            r#"
type Column[n, a] = | Column(tensor[n, a])
type Hamt[a] = | Leaf { value: a }
type Frame[n, a] = | Frame { cols: Hamt[Column[n, a]] }
def singleton[a](value: a) -> Hamt[a] = Leaf { value }
def from_column[n, a](column: Column[n, a]) -> Frame[n, a] =
  Frame { cols: singleton(column) }
"#,
        );
        let definitions = adt_constructor_definitions(&HostLoweringSession::new(&checked));
        assert!(
            definitions
                .iter()
                .all(|definition| definition.adt_name != "Option"),
            "prelude ADTs with dedicated host layouts must not enter the generic constructor table"
        );
        let column = definitions
            .iter()
            .find(|definition| definition.adt_name == "Column")
            .expect("Column constructor metadata");
        assert_eq!(column.parameters.len(), 2);
        assert_eq!(
            column.stored_parameters,
            [false, true],
            "the dimension parameter is erased while the dtype is stored"
        );
        assert!(matches!(
            column.fields[0].term,
            HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
                precision: HostPrecisionTerm::Variable(_),
                shape: HostShapeTerm::Concrete(_),
            })
        ));
        let frame = definitions
            .iter()
            .find(|definition| definition.adt_name == "Frame")
            .expect("Frame constructor metadata");
        assert_eq!(
            frame.stored_parameters,
            [false, true],
            "nested Frame -> Hamt -> Column storage classification"
        );
        assert!(
            checked
                .signature_inference()
                .functions
                .get("from_column")
                .is_some_and(|signature| {
                    signature.authored_signature && signature.authored_signature_type.is_some()
                }),
            "checker metadata must record the authored generic signature"
        );
        assert!(
            top_level_fn_is_type_polymorphic(&HostLoweringSession::new(&checked), "from_column"),
            "the stored dtype behind Frame -> Hamt -> Column must classify from checker metadata"
        );
    }

    #[test]
    fn generalized_unannotated_callback_is_not_classified_as_authored_polymorphism() {
        let checked = surf_check(
            "def apply(callback, value) = callback(value)\n\
             def increment(value: i32) -> i32 = add(value, 1)\n\
             def use_callback() -> i32 = apply(increment, 1)\n",
        );
        let apply = checked
            .signature_inference()
            .functions
            .get("apply")
            .expect("generalized callback metadata");
        assert!(!apply.authored_signature);
        assert!(apply.authored_signature_type.is_none());
        assert!(
            !top_level_fn_is_type_polymorphic(&HostLoweringSession::new(&checked), "apply"),
            "an inferred/generalized callback is not an authored generic ABI"
        );
    }

    #[test]
    fn authored_signature_classifier_requires_exact_or_unambiguous_name() {
        let checked = parse_and_check(
            r#"
                (defsig {} Left.identity (a)
                  (t-fn {} (t-var {} a) (t-var {} a)))
                (def {} Left.identity
                  (fn {} (params {} (x {type: (t-var {} a)})) (var {} x)))
                (defsig {} Right.identity (b)
                  (t-fn {} (t-var {} b) (t-var {} b)))
                (def {} Right.identity
                  (fn {} (params {} (x {type: (t-var {} b)})) (var {} x)))
            "#,
        );
        assert!(
            top_level_fn_is_type_polymorphic(&HostLoweringSession::new(&checked), "Left.identity"),
            "an exact qualified checker record must classify"
        );
        assert!(
            top_level_fn_is_type_polymorphic(&HostLoweringSession::new(&checked), "Right.identity"),
            "the other exact qualified checker record must classify"
        );
        assert!(
            !top_level_fn_is_type_polymorphic(&HostLoweringSession::new(&checked), "identity"),
            "an ambiguous terminal name must not select either qualified checker record"
        );
    }

    #[test]
    fn empty_to_tensor_uses_checked_default_before_concrete_resolution() {
        let checked = surf_check("result = numel(to_tensor([]))\n");
        let compiled = try_lower_compiled_program(&checked)
            .expect("the checked empty tensor default must resolve before codegen");
        assert!(compiled.host.is_some());
    }

    #[test]
    fn host_program_uses_builtin_detects_process_run() {
        // Hull subprocess exec: a global binding that applies process_run is
        // detected so the build backends can reject it before codegen.
        let checked = surf_check("result = process_run(\"echo\", [\"hi\"])\n");
        let compiled = try_lower_compiled_program(&checked).expect("fixture must lower");
        let host = compiled.host.expect("host program present");
        assert!(
            host_program_uses_builtin(&host, "process_run"),
            "host_program_uses_builtin must detect a process_run global binding"
        );
    }

    #[test]
    fn host_program_uses_builtin_is_false_without_process_run() {
        // Negative parity: a program that uses only file IO must not report
        // process_run usage, so the positive assertion is not vacuous.
        let checked = surf_check("contents = read_file(\"dataset.txt\")\n");
        let compiled = try_lower_compiled_program(&checked).expect("fixture must lower");
        let host = compiled.host.expect("host program present");
        assert!(
            !host_program_uses_builtin(&host, "process_run"),
            "host_program_uses_builtin must be false for a read_file-only program"
        );
    }

    /// chelis#336: restore the guard that a function-typed parameter survives
    /// as `HostTypeTerm::Fn` through `try_lower_compiled_program`. #331 deleted the
    /// only test asserting this (it relied on a reef-package HOF). A
    /// *same-module* HOF like `def apply(f, x) = f(x)` is inlined away, so a
    /// surviving fn-param needs a non-inlined HOF: here `apply_each` is kept
    /// as a real host function because it is called with a fn argument and
    /// uses the `map` combinator. The lowered `apply_each` must keep its
    /// `f: f32 -> f32` parameter typed `HostTypeTerm::Fn`, not collapsed to a
    /// value type.
    #[test]
    fn host_program_preserves_fn_typed_param_through_lowering() {
        let checked = surf_check(
            "def apply_each(f: f32 -> f32, xs: List[f32]) -> List[f32] = map(fn (v: f32) -> f(v), xs)\n\
             def double(x: f32) -> f32 = mul(x, 2.0)\n\
             out = apply_each(double, [1.0, 2.0, 3.0])\n",
        );
        let compiled = try_lower_compiled_program(&checked).expect("fixture must lower");
        let host = compiled.host.expect("host program present");
        let apply_each = host
            .functions
            .iter()
            .find(|f| f.name == "apply_each")
            .expect("apply_each must survive as a host function (not inlined)");
        let f_param = apply_each
            .params
            .iter()
            .find(|p| p.name == "f")
            .expect("apply_each must keep its `f` parameter");
        match &f_param.ty {
            ConcreteHostType::Function(params, ret) => {
                assert_eq!(params.len(), 1, "f takes one scalar arg, got {params:?}");
                assert!(
                    matches!(params[0], ConcreteHostType::Scalar(Prim::F32 | Prim::F64)),
                    "f's arg must be a scalar, got {:?}",
                    params[0]
                );
                assert!(
                    matches!(**ret, ConcreteHostType::Scalar(Prim::F32 | Prim::F64)),
                    "f's return must be a scalar, got {ret:?}"
                );
            }
            other => {
                panic!("fn-typed param `f` must lower to ConcreteHostType::Function, got {other:?}")
            }
        }
    }

    /// chelis#336 negative parity: a program with no higher-order parameter
    /// must produce no `HostTypeTerm::Fn` parameter, so the positive guard above
    /// is not vacuously satisfied by some unrelated fn-typed param.
    #[test]
    fn host_program_has_no_fn_typed_param_without_hof() {
        let checked = surf_check(
            "def double(x: f32) -> f32 = mul(x, 2.0)\n\
             out = double(2.0)\n",
        );
        let compiled = try_lower_compiled_program(&checked).expect("fixture must lower");
        let host = compiled.host.expect("host program present");
        let has_fn_param = host
            .functions
            .iter()
            .flat_map(|f| f.params.iter())
            .any(|p| matches!(p.ty, ConcreteHostType::Function(_, _)));
        assert!(
            !has_fn_param,
            "a non-HOF program must not lower any HostTypeTerm::Fn parameter"
        );
    }

    // ── Issue #308: scalar_to_tensor operand-precision plumbing ──
    //
    // `HostTypeTerm::Float64` collapses f32 and f64, so the Deep-level
    // `infer_app_expr_host_type` must recover the operand precision
    // before the coarse arg-ty fallback erases it to f32.

    fn parse_deep_app(src: &str) -> Expr {
        chelis_deep::parser::parse_str(src)
            .expect("parse failed")
            .into_iter()
            .next()
            .expect("one expr")
    }

    fn infer_scalar_to_tensor_host_type(arg_src: &str) -> Option<HostTypeTerm> {
        let app = parse_deep_app(&format!("(app {{}} (var {{}} scalar_to_tensor) {arg_src})"));
        // Empty checked program: the arm under test must not depend on
        // program-level lookups for the precision recovery.
        let program = surf_check("unrelated = 1\n");
        infer_app_expr_host_type(&app, &HostLoweringSession::new(&program), &UnordMap::new())
    }

    #[test]
    fn scalar_to_tensor_infers_f64_from_cast_target() {
        // The #308 reproducer shape: `scalar_to_tensor(cast(1.1, f64))`
        // with NO checker annotation on the app or the cast — the cast
        // target alone must drive the result precision.
        let inferred = infer_scalar_to_tensor_host_type(
            "(cast {} (lit {type: (t-prim {} f32)} 1.1) (t-prim {} f64))",
        );
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: Prim::F64,
            })),
            "scalar_to_tensor(cast(_, f64)) must infer a rank-0 f64 tensor",
        );
    }

    #[test]
    fn scalar_to_tensor_infers_f64_from_checker_annotation() {
        let inferred = infer_scalar_to_tensor_host_type("(var {type: (t-prim {} f64)} c)");
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: Prim::F64,
            })),
            "scalar_to_tensor of an f64-annotated operand must infer a rank-0 f64 tensor",
        );
    }

    #[test]
    fn scalar_to_tensor_keeps_f32_default_for_f32_operand() {
        // Negative parity (pins the #300/#306 f32 path): an f32 operand
        // — whether via cast target or bare default literal — must keep
        // the rank-0 f32 result so `chelis_scalar_tensor_from_f32`
        // storage and the consuming helper's f32 read stay paired.
        for arg_src in [
            "(cast {} (lit {type: (t-prim {} f32)} 2.5) (t-prim {} f32))",
            "(lit {type: (t-prim {} f32)} 2.5)",
        ] {
            let inferred = infer_scalar_to_tensor_host_type(arg_src);
            assert_eq!(
                inferred,
                Some(HostTypeTerm::Tensor(TensorType {
                    dims: vec![],
                    precision: Prim::F32,
                })),
                "scalar_to_tensor({arg_src}) must keep the f32 default",
            );
        }
    }

    #[test]
    fn scalar_to_tensor_float_recovery_does_not_hijack_integer_operands() {
        // Negative parity: the #308 float-precision recovery must not
        // claim integer operands. At this Deep-expr layer a
        // cast-wrapped int infers `None` (pre-#308 behavior:
        // `expr_host_type` does not see through `cast`, so the coarse
        // fallback gets `Unknown` and abstains); the Int64 tensor
        // typing happens post-lowering via `host_expr_type` on the
        // lowered operand and the `infer_builtin_host_type_from_arg_tys`
        // Int64 arm.
        let inferred = infer_scalar_to_tensor_host_type(
            "(cast {} (lit {type: (t-prim {} i32)} 3) (t-prim {} i32))",
        );
        assert_eq!(
            inferred, None,
            "integer operands must fall through unchanged (no float hijack)",
        );
        // The coarse arm still owns the lowered-lane integer answer.
        assert_eq!(
            infer_builtin_host_type_from_arg_tys("scalar_to_tensor", &[HostTypeTerm::Int64]),
            Ok(Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: Prim::Int64,
            }))),
        );
    }

    // ── chelis#631: host-lane concat result typing ──
    //
    // The host lane must never carry the ELEMENT's extent on the concat
    // axis: `chelis_tensor_concat` sizes that axis at run time, and a
    // baked element extent aborts the binary at the runtime-dim guard.

    fn rank2_element() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(1), DimInfo::Lit(2)],
            precision: Prim::F32,
        }
    }

    fn anon_dim() -> DimInfo {
        DimInfo::Named("*".to_string(), None)
    }

    fn infer_concat_host_type(axis_src: &str) -> Option<HostTypeTerm> {
        let app = parse_deep_app(&format!(
            "(app {{}} (var {{}} concat) (var {{}} rows) {axis_src})"
        ));
        let program = surf_check("unrelated = 1\n");
        let mut scope = UnordMap::new();
        scope.insert(
            "rows".to_string(),
            HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(rank2_element()))),
        );
        infer_app_expr_host_type(&app, &HostLoweringSession::new(&program), &scope)
    }

    #[test]
    fn concat_app_expr_host_type_wildcards_concat_axis_only() {
        // Literal axis 0 (bare and cast-wrapped, the canonical spelling):
        // the concat axis is anon, the trailing element extent survives.
        for axis_src in ["(lit {} 0)", "(cast {} (lit {} 0) (t-prim {} i32))"] {
            let inferred = infer_concat_host_type(axis_src);
            assert_eq!(
                inferred,
                Some(HostTypeTerm::Tensor(TensorType {
                    dims: vec![anon_dim(), DimInfo::Lit(2)],
                    precision: Prim::F32,
                })),
                "concat(rows, {axis_src}) must wildcard only axis 0",
            );
        }
    }

    #[test]
    fn concat_app_expr_host_type_negative_axis_indexes_from_end() {
        let inferred = infer_concat_host_type("(lit {} -1)");
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(1), anon_dim()],
                precision: Prim::F32,
            })),
            "concat(rows, -1) must wildcard the LAST axis and keep axis 0",
        );
    }

    #[test]
    fn concat_app_expr_host_type_unknown_axis_wildcards_all_axes() {
        // Negative parity: a non-literal axis leaves no per-axis claim.
        let inferred = infer_concat_host_type("(var {} ax)");
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![anon_dim(), anon_dim()],
                precision: Prim::F32,
            })),
            "a runtime concat axis must wildcard every axis",
        );
    }

    #[test]
    fn concat_arg_tys_fallback_wildcards_all_axes() {
        // The arg-tys-only lane cannot see the axis value: pre-chelis#631
        // it returned the element type VERBATIM (dims [1, 2]) — the baked
        // extent that aborted guarded forward binaries.
        let inferred = infer_builtin_host_type_from_arg_tys(
            "concat",
            &[
                HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(rank2_element()))),
                HostTypeTerm::Int64,
            ],
        );
        assert_eq!(
            inferred,
            Ok(Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![anon_dim(), anon_dim()],
                precision: Prim::F32,
            }))),
        );
    }

    #[test]
    fn tuple_projection_retains_a_typed_literal_index() {
        let pair = HostExpr::new(HostExprKind::Var(
            "pair".to_string(),
            HostTypeTerm::Tuple(vec![
                HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32)),
                HostTypeTerm::Bool,
            ]),
        ));
        for index in [1, -1, 2] {
            let index_expr = HostExpr::new(HostExprKind::Builtin {
                name: "cast".to_string(),
                args: vec![HostExpr::new(HostExprKind::Int(index))],
                ty: HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32)),
            });
            let result = infer_builtin_host_type("tuple-get", &[pair.clone(), index_expr]);
            if index == 1 {
                assert_eq!(result, Some(HostTypeTerm::Bool));
            } else {
                assert!(result.is_some_and(|ty| ty.is_unresolved()));
            }
        }
    }

    #[test]
    fn concat_host_expr_lane_peels_cast_wrapped_axis() {
        // infer_builtin_host_type sees HostExpr args: a cast-wrapped int
        // axis still selects the single concat axis.
        let list_arg = HostExpr {
            kind: HostExprKind::Var(
                "rows".to_string(),
                HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(rank2_element()))),
            ),
            span_id: None,
            merged_spans: Vec::new(),
        };
        let axis_arg = HostExpr {
            kind: HostExprKind::Builtin {
                name: "cast".to_string(),
                args: vec![HostExpr {
                    kind: HostExprKind::Int(0),
                    span_id: None,
                    merged_spans: Vec::new(),
                }],
                ty: HostTypeTerm::Int64,
            },
            span_id: None,
            merged_spans: Vec::new(),
        };
        let inferred = infer_builtin_host_type("concat", &[list_arg, axis_arg]);
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![anon_dim(), DimInfo::Lit(2)],
                precision: Prim::F32,
            })),
        );
    }

    /// E2 (WS-A0 RT-1 fixup): the `to_list` host classification arm
    /// returns a typed error for f8e4m3 rather than panicking or silently
    /// classifying the deferred dtype as Float64.
    #[test]
    fn to_list_classifier_rejects_f8e4m3_tensor_per_spec_1_1_1() {
        let f8_tensor = HostTypeTerm::Tensor(crate::dag::TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F8e4m3,
        });
        let error = infer_builtin_host_type_from_arg_tys("to_list", &[f8_tensor])
            .expect_err("the deferred dtype must be a typed inference error");
        assert!(error.to_string().contains("f8e4m3 is deferred"));
    }

    #[test]
    fn invalid_builtin_shape_is_not_an_inference_variable() {
        let error = infer_builtin_host_type_from_arg_tys(
            "sum",
            &[HostTypeTerm::List(Box::new(HostTypeTerm::Int64))],
        )
        .expect_err("sum over a list is invalid, not underconstrained");
        assert_eq!(error.builtin, "sum");
        assert!(error.requirement.contains("must be a tensor"));
    }

    #[test]
    fn genuinely_underconstrained_tuple_get_keeps_a_fresh_identity() {
        let first = infer_builtin_host_type_from_arg_tys(
            "tuple-get",
            &[HostTypeTerm::Tuple(vec![HostTypeTerm::Int64])],
        )
        .expect("the arg-types-only lane lacks the index, but is not invalid")
        .expect("tuple-get is a known builtin");
        let second = infer_builtin_host_type_from_arg_tys(
            "tuple-get",
            &[HostTypeTerm::Tuple(vec![HostTypeTerm::Int64])],
        )
        .expect("the arg-types-only lane lacks the index, but is not invalid")
        .expect("tuple-get is a known builtin");
        assert!(matches!(first, HostTypeTerm::InferenceVariable(_)));
        assert!(matches!(second, HostTypeTerm::InferenceVariable(_)));
        assert_ne!(first, second, "fresh holes must retain distinct identity");
    }

    #[test]
    fn named_tensor_entry_inserts_copy_for_consuming_fanout() {
        let checked = parse_and_check(
            r#"
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
            "#,
        );
        let dag = lower_named_tensor_entry_dag(&checked, "double_it").expect("lower entry");
        let copy_count = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Copy))
            .count();
        assert_eq!(copy_count, 1, "{:?}", dag.nodes());
    }

    #[test]
    fn named_entry_dag_accepts_scalar_numeric_params() {
        let checked = parse_and_check(
            r#"
                (def {} add_scalar
                  (fn {type: (t-fn {}
                                (t-prim {} f32)
                                (t-prim {} f32)
                                (t-prim {} f32))}
                    (params {}
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (app {type: (t-prim {} f32)}
                      (var {} add)
                      (var {} x)
                      (var {} y))))
            "#,
        );

        let dag = lower_named_tensor_entry_dag(&checked, "add_scalar").expect("lower scalar entry");
        let root = dag.roots().first().and_then(|id| dag.get(*id)).unwrap();

        assert_eq!(root.op, RiscOp::Add, "{:?}", dag.nodes());
        let scalar_loads = dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.op, RiscOp::Load { .. })
                    && node.output_type.dims.is_empty()
                    && node.output_type.precision == Prim::F32
            })
            .count();
        assert_eq!(scalar_loads, 2, "{:?}", dag.nodes());
    }

    // ── WS-4: declared float width survives entry-param lowering ──
    //
    // `tensor_type_from_host_input` previously mapped every
    // `HostTypeTerm::Float64` (which collapsed both f32 and f64) to
    // `Prim::F32`, silently truncating an f64 scalar entry parameter to
    // 4-byte storage. The declared `f32`/`f64` width now threads through
    // `parse_host_type_with_subst` so each lowers to a `Load` of its own
    // precision.

    #[test]
    fn named_entry_dag_lowers_f64_scalar_params_to_f64_loads() {
        let checked = parse_and_check(
            r#"
                (def {} add_scalar
                  (fn {type: (t-fn {}
                                (t-prim {} f64)
                                (t-prim {} f64)
                                (t-prim {} f64))}
                    (params {}
                      (x {type: (t-prim {} f64)})
                      (y {type: (t-prim {} f64)}))
                    (app {type: (t-prim {} f64)}
                      (var {} add)
                      (var {} x)
                      (var {} y))))
            "#,
        );

        let dag =
            lower_named_tensor_entry_dag(&checked, "add_scalar").expect("lower f64 scalar entry");
        let root = dag.roots().first().and_then(|id| dag.get(*id)).unwrap();
        assert_eq!(root.op, RiscOp::Add, "{:?}", dag.nodes());

        let f64_loads = dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.op, RiscOp::Load { .. })
                    && node.output_type.dims.is_empty()
                    && node.output_type.precision == Prim::F64
            })
            .count();
        assert_eq!(
            f64_loads,
            2,
            "an f64 scalar entry param must lower to an f64 Load, not f32: {:?}",
            dag.nodes()
        );
    }

    #[test]
    fn named_entry_dag_does_not_downgrade_f64_scalar_params_to_f32() {
        // Negative twin: the historical bug. NO scalar Load produced for
        // a declared-f64 entry param may carry `Prim::F32`.
        let checked = parse_and_check(
            r#"
                (def {} add_scalar
                  (fn {type: (t-fn {}
                                (t-prim {} f64)
                                (t-prim {} f64)
                                (t-prim {} f64))}
                    (params {}
                      (x {type: (t-prim {} f64)})
                      (y {type: (t-prim {} f64)}))
                    (app {type: (t-prim {} f64)}
                      (var {} add)
                      (var {} x)
                      (var {} y))))
            "#,
        );

        let dag =
            lower_named_tensor_entry_dag(&checked, "add_scalar").expect("lower f64 scalar entry");
        let f32_scalar_loads = dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.op, RiscOp::Load { .. })
                    && node.output_type.dims.is_empty()
                    && node.output_type.precision == Prim::F32
            })
            .count();
        assert_eq!(
            f32_scalar_loads,
            0,
            "an f64 scalar entry param must NOT be silently downgraded to an f32 Load: {:?}",
            dag.nodes()
        );
    }

    #[test]
    fn parse_host_type_preserves_declared_float_width() {
        // Unit-level pin on the precision-threading site: the `t-prim`
        // syntax for f32 and f64 must map to distinct host scalar
        // variants so the entry-param boundary can recover the width.
        let f32_ty = parse_deep_app("(t-prim {} f32)");
        assert_eq!(
            decode_host_type_with_subst(&f32_ty, &UnordMap::new()),
            Ok(HostTypeTerm::Float32),
        );

        let f64_ty = parse_deep_app("(t-prim {} f64)");
        assert_eq!(
            decode_host_type_with_subst(&f64_ty, &UnordMap::new()),
            Ok(HostTypeTerm::Float64),
        );
    }

    #[test]
    fn host_input_tensor_round_trip_preserves_every_active_scalar_dtype() {
        for prim in [
            Prim::F16,
            Prim::Bf16,
            Prim::F32,
            Prim::F64,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
        ] {
            let host = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(prim));
            let tensor = TensorType {
                dims: vec![],
                precision: prim,
            };
            assert_eq!(
                tensor_type_from_host_input(&host),
                Some(tensor.clone()),
                "{prim:?}"
            );
            assert_eq!(host_type_from_tensor_input(&tensor), host, "{prim:?}");
        }
    }

    #[test]
    fn host_input_tensor_boundary_rejects_non_tensor_scalar_domains() {
        assert_eq!(tensor_type_from_host_input(&HostTypeTerm::String), None);
        assert_eq!(
            tensor_type_from_host_input(&HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(
                Prim::F8e4m3
            ))),
            None
        );
    }

    #[test]
    fn call_graph_recursion_detection_does_not_mark_every_function_recursive() {
        let graph = BTreeMap::from([
            ("plain".to_string(), BTreeSet::from(["leaf".to_string()])),
            ("leaf".to_string(), BTreeSet::new()),
            (
                "self_rec".to_string(),
                BTreeSet::from(["self_rec".to_string()]),
            ),
            ("mut_a".to_string(), BTreeSet::from(["mut_b".to_string()])),
            ("mut_b".to_string(), BTreeSet::from(["mut_a".to_string()])),
        ]);

        let recursive = recursive_top_level_fn_names_from_graph(&graph);

        assert!(
            !recursive.contains("plain"),
            "ordinary top-level callers must not be classified as recursive"
        );
        assert!(
            !recursive.contains("leaf"),
            "leaf functions must not be classified as recursive"
        );
        assert!(
            recursive.contains("self_rec"),
            "direct self-recursion must still be detected"
        );
        assert!(
            recursive.contains("mut_a") && recursive.contains("mut_b"),
            "mutual recursion must still be detected"
        );
    }

    #[test]
    fn tensor_helper_exp_body_keeps_exp_root() {
        let checked = parse_and_check(
            r#"
                (def {} softplus
                  (fn {type: (t-fn {}
                                (t-tensor {} (d-lit {} 4) (t-prim {} f32))
                                (t-tensor {} (d-lit {} 4) (t-prim {} f32)))}
                    (params {}
                      (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
                    (app {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}
                      (var {} exp)
                      (var {} x))))
            "#,
        );
        let defs = collect_program_defs(checked.exprs());
        let body = lookup_program_def(&defs, "softplus").unwrap();
        let fn_list = as_list(body).unwrap();
        let body = children(fn_list).get(1).unwrap();
        let mut scope = UnordMap::new();
        scope.insert(
            "x".to_string(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            }),
        );
        let expected = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let session = HostLoweringSession::new(&checked);
        let context = cached_subexpr_lowering_context(&session);
        let dag = lower_tensor_helper_dag(body, &session, &scope, &expected, &context)
            .expect("helper dag");
        let root = dag.roots().first().and_then(|id| dag.get(*id)).unwrap();
        assert_eq!(root.op, RiscOp::Exp, "{:?}", dag.nodes());
    }

    #[test]
    fn tensor_helper_hoists_host_lane_tensor_args_with_f32_type() {
        let checked = parse_and_check(
            r#"
                (defsig {}
                  jac_row
                  (n)
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
                          (lit {type: (t-prim {} i32)} 0))
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
                              (lit {type: (t-prim {} i32)} 0)))
                          (app {}
                            (var {} add)
                            (app {}
                              (var {} tensor_to_scalar)
                              (app {}
                                (var {} sum)
                                (copy {} (var {} theta))
                                (lit {type: (t-prim {} i32)} 0)))
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
        let host = lower_host_program(&HostLoweringSession::new(&checked), &lowered)
            .expect("host program must lower");
        let out_binding = host
            .globals
            .iter()
            .find(|binding| binding.name == "out")
            .expect("out host binding");
        let (bindings, body) = match &out_binding.value.kind {
            HostExprKind::Let { bindings, body, .. } => (bindings, body.as_ref()),
            other => panic!("expected out to hoist host-lane tensor arg into let, got {other:?}"),
        };
        let theta_binding = bindings
            .iter()
            .find(|binding| binding.name.starts_with("__host_tensor_arg_"))
            .unwrap_or_else(|| {
                panic!(
                    "theta temp binding missing; hoisted bindings were {:?}",
                    bindings
                        .iter()
                        .map(|binding| (&binding.name, &binding.ty))
                        .collect::<Vec<_>>()
                )
            });
        match &theta_binding.ty {
            HostTypeTerm::Tensor(tensor) => assert_eq!(tensor.precision, Prim::F32),
            other => panic!("expected hoisted theta binding to be tensor-typed, got {other:?}"),
        }
        let helper_index = match &body.kind {
            HostExprKind::TensorCall { helper, .. } => *helper,
            other => panic!("expected hoisted body to call tensor helper, got {other:?}"),
        };
        let helper = host
            .global_tensor_helpers
            .get(helper_index)
            .expect("helper index in range");
        assert!(
            helper
                .inputs
                .iter()
                .any(|input| input.name == theta_binding.name),
            "helper inputs should reference hoisted tensor temp: {:?}",
            helper
                .inputs
                .iter()
                .map(|input| (&input.name, &input.ty))
                .collect::<Vec<_>>()
        );
        assert!(
            helper.inputs.iter().all(|input| input.name != "to_tensor"),
            "helper inputs must not contain raw `to_tensor` load: {:?}",
            helper
                .inputs
                .iter()
                .map(|input| (&input.name, &input.ty))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn host_type_parser_unwraps_borrowed_tensor_refs() {
        let expr = chelis_deep::parser::parse_str(
            "(t-ref {} (t-tensor {} (d-name {} n) (t-prim {} f32)))",
        )
        .expect("parse ref type")
        .into_iter()
        .next()
        .expect("one type expr");

        match decode_host_type(&expr).expect("borrowed tensor type decodes") {
            HostTypeTerm::Tensor(tensor) => {
                assert_eq!(
                    tensor.dims,
                    vec![crate::dag::DimInfo::Named("n".into(), None)]
                );
                assert_eq!(tensor.precision, Prim::F32);
            }
            other => panic!("expected borrowed tensor ref to parse as tensor, got {other:?}"),
        }
    }

    #[test]
    fn helper_dimension_rebinding_boundary_has_a_nonidentity_discriminator() {
        use crate::dag::{Dag, DimInfo, RiscOp, TensorType};

        let symbolic = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let concrete = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let mut before = Dag::new();
        let root = before.add_node(RiscOp::Load { name: "x".into() }, vec![], symbolic, None);
        before.add_root(root);
        let mut scope = UnordMap::new();
        scope.insert("x".into(), HostTypeTerm::Tensor(concrete.clone()));

        let after = remap_tensor_helper_dim_symbols(&before, &scope, &concrete);
        assert_ne!(
            bincode::serialize(&before).unwrap(),
            bincode::serialize(&after).unwrap(),
            "the actual host boundary must rewrite the symbolic helper"
        );
        assert_eq!(after.get(after.roots()[0]).unwrap().output_type, concrete);
        assert_eq!(
            before.get(before.roots()[0]).unwrap().output_type.dims,
            vec![DimInfo::Named("n".into(), None)],
            "the pass input remains an immutable historical snapshot"
        );
        #[cfg(feature = "lowering-trace")]
        {
            let collector = crate::lowering_trace::Collector::new_helper();
            collector.normalization(
                &before,
                &before,
                before.clone(),
                &before,
                &UnordMap::new(),
                &UnordMap::new(),
            );
            collector.helper_result(before.roots(), crate::lowering_trace::Value::Node(root));
            let mut trace = collector.finish_helper();
            trace.record_dimension_rebinding(&after);
            let normalization = &trace.lowering.normalization;
            for historical in [
                &normalization.before_dce,
                &normalization.after_dce,
                &normalization.after_copies,
                &normalization.after_drops,
            ] {
                assert_eq!(
                    bincode::serialize(historical).unwrap(),
                    bincode::serialize(&before).unwrap()
                );
            }
            assert_eq!(
                bincode::serialize(trace.after_dimension_rebinding.as_ref().unwrap()).unwrap(),
                bincode::serialize(&after).unwrap(),
            );
        }
    }

    #[test]
    fn tensor_helper_actualization_merges_matmul_synthetic_expand_dims() {
        use crate::dag::{Dag, DimInfo, RiscOp, RtAxis, RtDim, TensorType};

        let batch = DimInfo::Named("batch".into(), None);
        let in_dim = DimInfo::Named("in_dim".into(), None);
        let out_dim = DimInfo::Named("out_dim".into(), None);
        let d417 = DimInfo::Named("d417".into(), None);
        let d420 = DimInfo::Named("d420".into(), None);
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![batch.clone(), in_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let w = dag.add_node(
            RiscOp::Load { name: "w".into() },
            vec![],
            TensorType {
                dims: vec![in_dim.clone(), out_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let expanded_x = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(1),
                },
            },
            vec![x, w],
            TensorType {
                dims: vec![batch.clone(), in_dim.clone(), d420.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let expanded_w = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![w, x],
            TensorType {
                dims: vec![d417, in_dim.clone(), out_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let product = dag.add_node(
            RiscOp::Mul,
            vec![expanded_x, expanded_w],
            TensorType {
                dims: vec![batch.clone(), in_dim.clone(), d420],
                precision: Prim::F32,
            },
            None,
        );
        let root = dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![product],
            TensorType {
                dims: vec![batch.clone(), out_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(root);

        let mut scope = UnordMap::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![batch.clone(), in_dim.clone()],
                precision: Prim::F32,
            }),
        );
        scope.insert(
            "w".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![in_dim.clone(), out_dim.clone()],
                precision: Prim::F32,
            }),
        );

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        for node in actualized.nodes() {
            assert!(
                !node.output_type.dims.iter().any(|dim| {
                    matches!(dim, DimInfo::Named(name, None) if name == "d417" || name == "d420")
                }),
                "node retained synthetic dims: {node:?}"
            );
        }
    }

    /// Actualization may rewrite a checker-minted output dimension, but the
    /// structural InputAxis carrier continues to reference the same tensor
    /// input slot and axis.
    #[test]
    fn tensor_helper_actualization_preserves_structural_expand_size() {
        use crate::dag::{Dag, DimInfo, RiscOp, RtAxis, RtDim, TensorType};

        let n = DimInfo::Named("n".into(), None);
        let d47 = DimInfo::Named("d47".into(), None);
        let mut dag = Dag::new();
        // Load typed with the minted alias; the scope knows the
        // user-facing symbol.
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![d47.clone()],
                precision: Prim::F32,
            },
            None,
        );
        // Scalar upstream gradient, as the Sum adjoint produces.
        let g = dag.add_node(
            RiscOp::Load { name: "g".into() },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        // The Sum adjoint's expand-back reads the original tensor shape.
        let expanded_g = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![g, x],
            TensorType {
                dims: vec![d47.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let root = dag.add_node(
            RiscOp::Mul,
            vec![expanded_g, x],
            TensorType {
                dims: vec![d47],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(root);

        let mut scope = UnordMap::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![n.clone()],
                precision: Prim::F32,
            }),
        );

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        let expand = actualized.get(expanded_g).expect("expand node");
        assert_eq!(
            expand.output_type.dims,
            vec![n],
            "expand output dim must actualize to the user-facing symbol"
        );
        assert_eq!(
            expand.op,
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            "actualization must preserve the structural witness slot"
        );
        let errors = crate::verify::verify(&actualized);
        assert!(
            errors.is_empty(),
            "actualized helper must verify: {errors:?}"
        );
    }

    /// Negative parity: an already user-facing witness remains unchanged.
    #[test]
    fn tensor_helper_actualization_leaves_user_facing_input_axis_alone() {
        use crate::dag::{Dag, DimInfo, RiscOp, RtAxis, RtDim, TensorType};

        let batch = DimInfo::Named("batch".into(), None);
        let mut dag = Dag::new();
        let g = dag.add_node(
            RiscOp::Load { name: "g".into() },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![batch.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let expanded = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![g, x],
            TensorType {
                dims: vec![batch.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let root = dag.add_node(
            RiscOp::Mul,
            vec![expanded, x],
            TensorType {
                dims: vec![batch],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(root);

        let mut scope = UnordMap::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Named("batch".into(), None)],
                precision: Prim::F32,
            }),
        );

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        let expand = actualized.get(expanded).expect("expand node");
        assert_eq!(
            expand.op,
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            "user-facing structural sizes must not be rewritten"
        );
    }

    #[test]
    fn tensor_helper_actualization_repairs_shrink_rank_from_bounds() {
        use crate::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load {
                name: "expanded".into(),
            },
            vec![],
            TensorType {
                // The checked helper arrived with the stale pre-expand rank.
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        let shrink = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![
                    (RtDim::Lit(0), RtDim::Lit(1)),
                    (RtDim::Lit(0), RtDim::Lit(2)),
                ],
            },
            vec![input],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(shrink);

        let mut scope = UnordMap::new();
        scope.insert(
            "expanded".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
                precision: Prim::F32,
            }),
        );

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        assert_eq!(
            actualized.get(shrink).expect("shrink node").output_type,
            TensorType {
                dims: vec![DimInfo::Lit(1), DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            "the shrink result rank and literal extents come from its actual input and bounds"
        );
    }

    #[test]
    fn tensor_helper_actualization_preserves_full_axis_shrink_extent() {
        use crate::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        let batch = DimInfo::Named("batch".into(), None);
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![batch.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let shrink = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::ToEnd)],
            },
            vec![input],
            TensorType {
                dims: vec![DimInfo::Named("*".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(shrink);

        let mut scope = UnordMap::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![batch.clone()],
                precision: Prim::F32,
            }),
        );

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        assert_eq!(
            actualized
                .get(shrink)
                .expect("full-axis shrink")
                .output_type
                .dims,
            vec![batch],
            "the full-axis sentinel preserves the input axis identity"
        );
    }

    /// NEGATIVE PARITY: a `ToEnd` sentinel under a nonzero start is malformed,
    /// not a shape actualization may interpret.
    ///
    /// `grad`'s Shrink adjoint panics on this exact shape and
    /// `shrink_adjoint_malformed_sentinel_fails_loud` is its `#[should_panic]`
    /// control; `bind_symbolic_dims` rejects the non-literal-start form.
    /// This is the third component's half of that rule. No producer builds
    /// one - `lower::lower_one_bound` gives the front end only `Lit`/`Node`,
    /// and every `grad` adjoint emitting the sentinel pairs it with `Lit(0)` -
    /// so the DAG is hand-built, and the assertion is about disposition: the
    /// exhaustive `Shrink` arm declines and leaves the node's declared type
    /// alone. It must not forward the input axis identity the way the
    /// well-formed `(0, ToEnd)` sentinel does, which would silently claim
    /// extent 4 for a slice that starts at 1.
    #[test]
    fn tensor_helper_actualization_declines_a_nonzero_start_shrink_sentinel() {
        use crate::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let declared = TensorType {
            dims: vec![DimInfo::Named("*".into(), None)],
            precision: Prim::F32,
        };
        let shrink = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::ToEnd)],
            },
            vec![input],
            declared.clone(),
            None,
        );
        dag.add_root(shrink);

        let scope = UnordMap::from([(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            }),
        )]);

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        assert_eq!(
            actualized
                .get(shrink)
                .expect("malformed sentinel shrink")
                .output_type,
            declared,
            "a malformed sentinel must not be given an extent by actualization"
        );
    }

    /// NEGATIVE PARITY: [05-MOV-1]'s owner matrix reserves `InputAxis` for
    /// `expand` and `reshape`. Host helper actualization must therefore
    /// decline it in either `shrink` bound and in a `stride`, preserving the
    /// declared type until verification reports the malformed carrier.
    #[test]
    fn tensor_helper_actualization_declines_input_axis_for_shrink_and_stride() {
        use crate::dag::{Dag, DimInfo, RiscOp, RtAxis, RtDim, TensorType};

        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("actual_input".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let declared = TensorType {
            dims: vec![DimInfo::Named("d47".into(), None)],
            precision: Prim::F32,
        };
        let input_axis = || RtDim::InputAxis {
            tensor: 1,
            axis: RtAxis::Lit(0),
        };
        let malformed = [
            dag.add_node(
                RiscOp::Shrink {
                    bounds: vec![(input_axis(), RtDim::Lit(2))],
                },
                vec![input, input],
                declared.clone(),
                None,
            ),
            dag.add_node(
                RiscOp::Shrink {
                    bounds: vec![(RtDim::Lit(0), input_axis())],
                },
                vec![input, input],
                declared.clone(),
                None,
            ),
            dag.add_node(
                RiscOp::Stride {
                    strides: vec![input_axis()],
                },
                vec![input, input],
                declared.clone(),
                None,
            ),
        ];
        for node in malformed {
            // `reusable_input` is a storage-reuse hint, not shape authority.
            // Exercise the generic fallback explicitly: a declined malformed
            // movement carrier must not inherit its reusable buffer's shape.
            dag.set_reusable_input(node, input);
            dag.add_root(node);
        }

        let scope = UnordMap::from([(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Named("actual_input".into(), None)],
                precision: Prim::F32,
            }),
        )]);

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        for node in malformed {
            assert_eq!(
                actualized
                    .get(node)
                    .expect("malformed movement node")
                    .output_type,
                declared,
                "InputAxis outside expand/reshape must not be actualized"
            );
        }
    }

    fn runtime_shrink_bound_nodes(
        dag: &mut crate::dag::Dag,
    ) -> (crate::dag::NodeId, crate::dag::NodeId) {
        use crate::dag::{RiscOp, TensorType};

        let scalar_i64 = TensorType {
            dims: vec![],
            precision: Prim::Int64,
        };
        let start = dag.add_node(
            RiscOp::synth_const(Prim::Int64, 1.0),
            vec![],
            scalar_i64.clone(),
            None,
        );
        let end = dag.add_node(
            RiscOp::synth_const(Prim::Int64, 3.0),
            vec![],
            scalar_i64,
            None,
        );
        (start, end)
    }

    #[test]
    fn runtime_shrink_name_reserves_scope_actualized_base_and_suffixes() {
        use crate::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(8)],
                precision: Prim::F32,
            },
            None,
        );
        let (start, end) = runtime_shrink_bound_nodes(&mut dag);
        let shrink = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Node(1), RtDim::Node(2))],
            },
            vec![input, start, end],
            TensorType {
                dims: vec![DimInfo::Named("*".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(shrink);

        let base = format!("_rt_shrink_dim_{}_0", shrink.0);
        let suffixed = format!("{base}_1");
        let mut scope = UnordMap::from([(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(8)],
                precision: Prim::F32,
            }),
        )]);
        scope.insert(
            "scope_only_reservations".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![
                    DimInfo::Named(base.clone(), None),
                    DimInfo::Named(suffixed, None),
                ],
                precision: Prim::F32,
            }),
        );

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        assert_eq!(
            actualized
                .get(shrink)
                .expect("runtime shrink")
                .output_type
                .dims,
            vec![DimInfo::Named(format!("{base}_2"), None)],
            "names introduced by host-scope actualization share the same identity namespace"
        );
    }

    #[test]
    fn runtime_shrink_name_reserves_op_internal_symbols_in_both_node_orders() {
        use crate::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        for internal_before_shrink in [false, true] {
            let mut dag = Dag::new();
            let input = dag.add_node(
                RiscOp::Load { name: "x".into() },
                vec![],
                TensorType {
                    dims: vec![DimInfo::Lit(8)],
                    precision: Prim::F32,
                },
                None,
            );
            let (start, end) = runtime_shrink_bound_nodes(&mut dag);
            let expected_shrink_id = if internal_before_shrink { 4 } else { 3 };
            let base = format!("_rt_shrink_dim_{expected_shrink_id}_0");
            let internal_shape = vec![RtDim::Sym(base.clone()), RtDim::Sym(format!("{base}_1"))];
            let add_internal_carrier = |dag: &mut Dag| {
                dag.add_node(
                    RiscOp::Reshape {
                        new_shape: internal_shape.clone(),
                    },
                    vec![input],
                    TensorType {
                        dims: vec![DimInfo::Lit(1), DimInfo::Lit(8)],
                        precision: Prim::F32,
                    },
                    None,
                )
            };
            if internal_before_shrink {
                let _ = add_internal_carrier(&mut dag);
            }
            let shrink = dag.add_node(
                RiscOp::Shrink {
                    bounds: vec![(RtDim::Node(1), RtDim::Node(2))],
                },
                vec![input, start, end],
                TensorType {
                    dims: vec![DimInfo::Named("*".into(), None)],
                    precision: Prim::F32,
                },
                None,
            );
            if !internal_before_shrink {
                let _ = add_internal_carrier(&mut dag);
            }
            dag.add_root(shrink);
            assert_eq!(shrink.0, expected_shrink_id);

            let scope = UnordMap::from([(
                "x".into(),
                HostTypeTerm::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(8)],
                    precision: Prim::F32,
                }),
            )]);
            let actualized = actualize_tensor_helper_types(&dag, &scope);
            assert_eq!(
                actualized
                    .get(shrink)
                    .expect("runtime shrink")
                    .output_type
                    .dims,
                vec![DimInfo::Named(format!("{base}_2"), None)],
                "op-internal symbols must be reserved before allocation regardless of node order"
            );
        }
    }

    #[test]
    fn multiple_runtime_shrinks_keep_distinct_names_through_fanout_consumers() {
        use crate::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};

        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(8)],
                precision: Prim::F32,
            },
            None,
        );
        let (start, end) = runtime_shrink_bound_nodes(&mut dag);
        let wildcard = TensorType {
            dims: vec![DimInfo::Named("*".into(), None)],
            precision: Prim::F32,
        };
        let shrink_a = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Node(1), RtDim::Node(2))],
            },
            vec![input, start, end],
            wildcard.clone(),
            None,
        );
        let shrink_b = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Node(1), RtDim::Node(2))],
            },
            vec![input, start, end],
            wildcard.clone(),
            None,
        );
        let synthetic = |name: &str| TensorType {
            dims: vec![DimInfo::Named(name.into(), None)],
            precision: Prim::F32,
        };
        let consumers = [
            dag.add_node(
                RiscOp::Add,
                vec![shrink_a, shrink_a],
                synthetic("d701"),
                None,
            ),
            dag.add_node(
                RiscOp::Mul,
                vec![shrink_a, shrink_a],
                synthetic("d702"),
                None,
            ),
            dag.add_node(
                RiscOp::Add,
                vec![shrink_b, shrink_b],
                synthetic("d703"),
                None,
            ),
            dag.add_node(
                RiscOp::Mul,
                vec![shrink_b, shrink_b],
                synthetic("d704"),
                None,
            ),
        ];
        for consumer in consumers {
            dag.add_root(consumer);
        }
        let scope = UnordMap::from([(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(8)],
                precision: Prim::F32,
            }),
        )]);

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        let dims_a = actualized
            .get(shrink_a)
            .expect("first shrink")
            .output_type
            .dims
            .clone();
        let dims_b = actualized
            .get(shrink_b)
            .expect("second shrink")
            .output_type
            .dims
            .clone();
        assert_ne!(
            dims_a, dims_b,
            "each runtime Shrink owns a distinct extent identity"
        );
        assert_eq!(
            actualized
                .get(consumers[0])
                .expect("a/add")
                .output_type
                .dims,
            dims_a
        );
        assert_eq!(
            actualized
                .get(consumers[1])
                .expect("a/mul")
                .output_type
                .dims,
            dims_a
        );
        assert_eq!(
            actualized
                .get(consumers[2])
                .expect("b/add")
                .output_type
                .dims,
            dims_b
        );
        assert_eq!(
            actualized
                .get(consumers[3])
                .expect("b/mul")
                .output_type
                .dims,
            dims_b
        );
    }

    fn program_with_global_builtin(builtin: &str) -> HostProgram {
        let mut program = HostProgram::default();
        program.globals.push(HostBinding {
            name: "probe".into(),
            display_name: None,
            display_roots: Vec::new(),
            ty: HostTypeTerm::Unit,
            value: HostExpr::new(HostExprKind::Builtin {
                name: builtin.into(),
                args: Vec::new(),
                ty: HostTypeTerm::Unit,
            }),
        });
        program
    }

    /// Only the unspellable markers are unresolved call sites; names a
    /// user could spell (`call`, `__unresolved_grad`) are ordinary
    /// identifiers (chelis#841). The transform marker routes to the
    /// transform scan; the callable marker does not.
    #[test]
    fn unresolved_call_sites_match_only_the_unspellable_markers() {
        let callable = program_with_global_builtin(HOST_UNRESOLVED_CALLABLE_MARKER);
        assert_eq!(
            host_program_unresolved_call_sites(&callable),
            vec!["probe".to_string()]
        );
        assert!(host_program_unresolved_transform_sites(&callable).is_empty());

        let transform = program_with_global_builtin(HOST_UNRESOLVED_TRANSFORM_MARKER);
        assert_eq!(
            host_program_unresolved_call_sites(&transform),
            vec!["probe".to_string()]
        );
        assert_eq!(
            host_program_unresolved_transform_sites(&transform),
            vec!["probe".to_string()]
        );

        for spellable in ["call", "__unresolved_grad"] {
            let program = program_with_global_builtin(spellable);
            assert!(
                host_program_unresolved_call_sites(&program).is_empty(),
                "user-spellable name {spellable:?} must not read as a marker"
            );
        }

        let mut marker_call = HostProgram::default();
        marker_call.globals.push(HostBinding {
            name: "probe".into(),
            display_name: None,
            display_roots: Vec::new(),
            ty: HostTypeTerm::Unit,
            value: HostExpr::new(HostExprKind::Call {
                function: HOST_UNRESOLVED_CALLABLE_MARKER.into(),
                args: Vec::new(),
                arg_tys: Vec::new(),
                ty: HostTypeTerm::Unit,
            }),
        });
        assert_eq!(
            host_program_unresolved_call_sites(&marker_call),
            vec!["probe".to_string()],
            "a typed-but-unresolved callee carries a marker in `Call` position"
        );
    }

    #[test]
    fn nullary_generic_constructor_retains_named_term_until_checked_expectation() {
        let definition = GenericAdtConstructor {
            adt_name: "Box".into(),
            ctor_name: "Empty".into(),
            parameters: vec!["a".into()],
            stored_parameters: vec![false],
            fields: Vec::new(),
        };
        let carried = definition
            .instantiate_nullary_term(&fresh_host_inference())
            .expect("nullary generic term may await its checked context");
        assert_eq!(
            carried.ty,
            HostTypeTerm::Adt("Box".into(), vec![HostTypeTerm::TypeVariable("a".into())])
        );
        let concrete = force_host_expr_type(
            HostExpr::new(HostExprKind::AdtConstruct {
                ctor: "Empty".into(),
                fields: Vec::new(),
                ty: carried.ty,
            }),
            HostTypeTerm::Adt("Box".into(), vec![HostTypeTerm::Float32]),
        );
        assert_eq!(
            host_expr_type(&concrete),
            HostTypeTerm::Adt("Box".into(), vec![HostTypeTerm::Float32])
        );
    }

    #[test]
    fn nullary_generic_constructor_still_rejects_wrong_name_and_arity() {
        let definition = GenericAdtConstructor {
            adt_name: "Box".into(),
            ctor_name: "Empty".into(),
            parameters: vec!["a".into()],
            stored_parameters: vec![false],
            fields: Vec::new(),
        };
        assert!(matches!(
            definition.instantiate_nullary_term(&HostTypeTerm::Adt(
                "Other".into(),
                vec![HostTypeTerm::Float32]
            )),
            Err(AdtInstantiationError::WrongAppliedType { .. })
        ));
        assert!(matches!(
            definition.instantiate_nullary_term(&HostTypeTerm::Adt("Box".into(), Vec::new())),
            Err(AdtInstantiationError::Arity {
                expected: 1,
                got: 0,
                ..
            })
        ));
    }

    /// chelis#1201: a non-recursive generic is no longer inlined — it is
    /// specialized through the same bounded monomorphization the recursive
    /// path uses. The lowering assertion is the contract; the predicate that
    /// used to gate inlining is gone with the inlining.
    #[test]
    fn nonrecursive_generic_match_is_a_bounded_callsite_specialization() {
        let checked = surf_check(
            "type Box[a] =\n\
               | Empty\n\
               | Full { value: a }\n\
             def empty[a]() -> Box[a] = Empty\n\
             def is_empty[a](value: Box[a]) -> bool = match value with {\n\
               | Empty => true\n\
               | Full { value: item } => false\n\
             }\n\
             def concrete() -> Box[f32] = empty()\n\
             def main() -> bool = is_empty(concrete())\n",
        );
        let lowered = try_lower_compiled_program(&checked)
            .expect("the bounded generic match must lower through its concrete caller");
        let host = lowered
            .host
            .as_ref()
            .expect("the program lowers a host lane");
        assert!(
            host.functions
                .iter()
                .any(|function| function.origin == HostFunctionOrigin::Monomorphized),
            "the non-recursive generic must reach lowering as a specialization, not an inline paste"
        );
    }

    #[test]
    fn manifested_host_lane_overrides_legacy_tensor_root_classification() {
        let checked = surf_check(
            "x = insert(scalar_to_tensor(cast(0.1, f64)), 0, 4i64)\n\
             y = mul(x, x)\n",
        );
        let realizability = chelis_effects::realizability::infer_realizability(
            &checked,
            &[
                Prim::F32,
                Prim::Bool,
                Prim::Bf16,
                Prim::F16,
                Prim::Int32,
                Prim::Int64,
            ],
        );
        let manifest =
            chelis_effects::realizability::compute_root_manifest(&checked, &realizability);
        assert!(
            manifest
                .entries
                .iter()
                .any(|entry| entry.name == "y" && entry.lane == chelis_types::types::Lane::Host),
            "the C-like capability set must route the f64 root to Host"
        );

        let legacy = try_lower_compiled_program(&checked).expect("legacy lowering");
        assert!(
            legacy.host.is_none(),
            "the fixture must exercise the legacy classifier's Tensor decision"
        );
        let manifested = try_lower_compiled_program_with_manifest(&checked, &manifest)
            .expect("manifested lowering");
        assert!(
            manifested.host.is_some(),
            "manifested lowering must retain the C-target Host realization"
        );
    }

    /// chelis#1201 / coral#26: a generic ADT constructed in a position the
    /// caller's expected type cannot reach.
    ///
    /// The arm-body and `if`-branch propagation resolves a constructor that
    /// sits in a RESULT position. A block-local binding is not one, so this
    /// resolves only through the specialization's own type substitution —
    /// and that substitution only matches if it is keyed on the same
    /// variables the checker stamped into the body. This is the exact shape
    /// coral's `Hamt` node constructors take.
    #[test]
    fn generic_adt_built_outside_a_result_position_specializes() {
        let checked = surf_check(
            "type Store[a] =\n\
               | Vacant\n\
               | Held { key: string, value: a }\n\
             def unwrap[a](s: Store[a], fallback: a) -> a = match s with {\n\
               | Vacant => fallback\n\
               | Held { key: k, value: v } => v\n\
             }\n\
             def put[a](s: Store[a], key: string, value: a) -> a = {\n\
               fresh = Held { key, value }\n\
               unwrap(fresh, value)\n\
             }\n\
             def read() -> i64 = put(Vacant, \"a\", cast(1, i64))\n",
        );
        try_lower_compiled_program(&checked).expect(
            "a generic ADT built in a block-local binding must resolve from the specialization",
        );
    }

    /// chelis#1201: a bare nullary generic constructor used as a call
    /// ARGUMENT. Nothing at its own site pins the parameter; it resolves
    /// only from the callee's declared parameter type.
    ///
    /// This already passes without the substitution fix above — it is
    /// coverage for the shape #1201's body describes, not a guard on that
    /// fix. The guard is
    /// `generic_adt_built_outside_a_result_position_specializes`.
    #[test]
    fn nullary_generic_constructor_in_argument_position_specializes() {
        let checked = surf_check(
            "type Store[a] =\n\
               | Vacant\n\
               | Held { key: string, value: a }\n\
             def fresh_store[a]() -> Store[a] = Vacant\n\
             def unwrap[a](s: Store[a], fallback: a) -> a = match s with {\n\
               | Vacant => fallback\n\
               | Held { key: k, value: v } => v\n\
             }\n\
             def seed[a](value: a) -> a = unwrap(fresh_store(), value)\n\
             def read() -> i64 = seed(cast(7, i64))\n",
        );
        try_lower_compiled_program(&checked)
            .expect("a nullary generic constructor in argument position must lower");
    }

    /// chelis#1201: a generic container instantiated at a DIMENSION-generic
    /// element type.
    ///
    /// #1201's body asserts that dim-generic tensor code and generic
    /// container ADTs are disjoint populations. coral's `Frame` is both —
    /// `Hamt[Column[n]]` — so this combination is pinned here directly.
    ///
    /// It already passes without the substitution fix above; it exists
    /// because the claim it refutes was load-bearing in #1201's triage, not
    /// as a guard on that fix.
    #[test]
    fn generic_container_over_a_dimension_generic_adt_specializes() {
        let checked = surf_check(
            "type Col[n] =\n\
               | FloatCol(tensor[n, f32])\n\
             type Box[a] =\n\
               | Nothing\n\
               | Just { item: a }\n\
             def box_it[a](item: a) -> Box[a] = Just { item }\n\
             def unbox[a](b: Box[a], fallback: a) -> a = match b with {\n\
               | Nothing => fallback\n\
               | Just { item: i } => i\n\
             }\n\
             def roundtrip[n](c: Col[n]) -> Col[n] = unbox(box_it(c), c)\n\
             def total[n](c: Col[n]) -> tensor[n, f32] = match roundtrip(c) with {\n\
               | FloatCol(t) => t\n\
             }\n\
             def main() -> tensor[2, f32] =\n\
               total(FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)])))\n",
        );
        try_lower_compiled_program(&checked)
            .expect("a generic container over a dim-generic element must lower");
    }

    /// chelis#1216: a RECURSIVE function generic over an erased ADT
    /// dimension.
    ///
    /// The inline path specializes the outer call but refuses the recursive
    /// edge, and that edge then fell through to a plain call to a symbol the
    /// emitter elided, carrying the ADT's own parameter variable to the
    /// code-generation boundary. Such a callee is now monomorphized instead,
    /// so the recursive edge has an in-progress specialization to complete
    /// its slots from. coral's `column_lengths_match` -> `all_eq_len` is
    /// this shape.
    #[test]
    fn recursive_dimension_generic_call_is_monomorphized() {
        let checked = surf_check(
            "type Col[n] =\n\
               | FloatCol(tensor[n, f32])\n\
             def zero_i64() -> i64 = cast(0, i64)\n\
             def one_i64() -> i64 = cast(1, i64)\n\
             def col_len[n](col: Col[n]) -> i64 = match col with {\n\
               | FloatCol(xs) => numel(xs)\n\
             }\n\
             def all_eq_len[n](pairs: List[(string, Col[n])], expected: i64) -> bool =\n\
               if eq(len(pairs), zero_i64()) then true else {\n\
                 entry = index(pairs, zero_i64())\n\
                 if neq(col_len(entry.1), expected) then false else all_eq_len(drop(pairs, one_i64()), expected)\n\
               }\n\
             def main() -> bool = all_eq_len([(\"a\", FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)])))], cast(2, i64))\n",
        );
        let lowered = try_lower_compiled_program(&checked)
            .expect("a recursive dimension-generic call must lower through monomorphization");
        let host = lowered
            .host
            .as_ref()
            .expect("the program lowers a host lane");
        // PR #1218 review: any-Monomorphized was too weak — pin the
        // specialization to `all_eq_len` itself...
        let specialization = host
            .functions
            .iter()
            .find(|function| {
                function.origin == HostFunctionOrigin::Monomorphized
                    && function.name.starts_with("all_eq_len__mono_")
            })
            .expect("the specialization must belong to `all_eq_len`, not merely exist");
        // ...and pin its recursive edge to the OWNING specialized symbol:
        // the reported defect was precisely that edge falling through to a
        // plain call to the elided generic symbol. Direct recursion at one
        // instantiation memoizes to one symbol, so the edge must name the
        // specialization it lives in.
        let body = format!("{:?}", specialization.body);
        assert!(
            body.contains(&format!("function: \"{}\"", specialization.name)),
            "the recursive edge must call the owning specialized symbol \
             `{}`, got body:\n{body}",
            specialization.name
        );
        assert!(
            !body.contains("function: \"all_eq_len\""),
            "no edge may still call the elided generic symbol `all_eq_len`, \
             got body:\n{body}"
        );
    }

    // ── chelis#1271: cross-package constructor-name collision ────────
    //
    // Two packages in one dependency graph may declare types whose
    // constructors share an unqualified name. The reef linker gives each
    // a distinct `Pkg__<pkg>__<Module>__<Name>` identity, so lowering
    // must resolve on the full spelling and must not silently pick one
    // when only a short spelling is available. The end-to-end faces live
    // in `crates/chelis-cli/tests/issue_1271_cross_package_ctor_collision.rs`;
    // these cover the resolution rule itself and the ambiguity arm.
    //
    // No whole program reaches that arm, and the reason is structural
    // rather than a property of today's checker: it needs mangled
    // DECLARATIONS together with a SHORT reference, mangled declarations
    // exist only in reef linker output, and there every reference is
    // mangled too. A hand-authored mangled declaration is a
    // `ReservedLinkerName` declaration error (spec/04-type-system.md,
    // "Reserved linker name format"). The arm is therefore a fail-closed
    // guard, and unit tests are the only place it can be exercised.

    /// Two declarations whose mangled names differ only in their package
    /// segment, with `Adep__` sorting ahead of `Blib__` so a first-match
    /// rule always answers with the wrong one.
    const COLLIDING_DECLARATIONS: &str = concat!(
        "(deftype {} Adep__Wrapped () (variant {} Adep__Wrapped ",
        "(field {} amount (t-prim {} f32)) (field {} extra (t-prim {} f32))))\n",
        "(deftype {} Blib__Wrapped () (variant {} Blib__Wrapped ",
        "(field {} extra (t-prim {} f32)) (field {} amount (t-prim {} f32))))\n"
    );

    /// Parse one Deep expression into the `Expr::List` carrier that
    /// `lower_host_expr_kind` dispatches on, bridging stamped `Node`s the
    /// way the surrounding lowering code already does (chelis#908).
    fn deep_expr(source: &str) -> Expr {
        fn bridge(expr: &Expr) -> Expr {
            match expr {
                Expr::Node(node, span) => Expr::List(
                    List {
                        elements: node.to_list(*span).elements.iter().map(bridge).collect(),
                    },
                    *span,
                ),
                Expr::List(list, span) => Expr::List(
                    List {
                        elements: list.elements.iter().map(bridge).collect(),
                    },
                    *span,
                ),
                other => other.clone(),
            }
        }
        let parsed = chelis_deep::parser::parse_str(source)
            .expect("parse failed")
            .into_iter()
            .next()
            .expect("one expression");
        bridge(&parsed)
    }

    fn lower_against(program: &CheckedProgram, source: &str) -> Result<HostExpr, String> {
        let program = &HostLoweringSession::new(program);
        let mut helpers = TensorHelperSink::new(false, false);
        lower_host_expr(&deep_expr(source), program, &UnordMap::new(), &mut helpers)
            .map_err(|diagnostic| diagnostic.message)
    }

    #[test]
    fn constructor_resolution_prefers_the_exact_spelling() {
        // The repair: the mangled name IS the identity
        // (spec/04-type-system.md, "Module identity"), so the
        // later-sorting package still resolves to its own declaration.
        let checked = parse_and_check(COLLIDING_DECLARATIONS);
        let AdtConstructorResolution::Unique(definition) = resolve_adt_constructor_definition(
            &HostLoweringSession::new(&checked),
            "Blib__Wrapped",
        ) else {
            panic!("an exactly-spelled constructor must resolve uniquely");
        };
        assert_eq!(definition.adt_name, "Blib__Wrapped");
        assert_eq!(
            definition
                .fields
                .iter()
                .map(|field| field.name.as_deref())
                .collect::<Vec<_>>(),
            [Some("extra"), Some("amount")],
            "the exact spelling must carry its own field list and order"
        );
    }

    #[test]
    fn constructor_resolution_still_answers_a_unique_terminal_spelling() {
        // The terminal fallback is what lets a short, unqualified
        // spelling reach its declaration; the repair narrows it, it does
        // not remove it.
        let checked = parse_and_check(concat!(
            "(deftype {} Blib__Wrapped () (variant {} Blib__Wrapped ",
            "(field {} amount (t-prim {} f32))))\n"
        ));
        let AdtConstructorResolution::Unique(definition) =
            resolve_adt_constructor_definition(&HostLoweringSession::new(&checked), "Wrapped")
        else {
            panic!("an uncontested terminal spelling must still resolve");
        };
        assert_eq!(definition.adt_name, "Blib__Wrapped");
    }

    #[test]
    fn constructor_resolution_refuses_a_contested_terminal_spelling() {
        // Negative parity: when the short spelling really is ambiguous,
        // lowering must say so and name both sides rather than restore
        // the defect under a different first-match rule.
        let checked = parse_and_check(COLLIDING_DECLARATIONS);
        let AdtConstructorResolution::Ambiguous(candidates) =
            resolve_adt_constructor_definition(&HostLoweringSession::new(&checked), "Wrapped")
        else {
            panic!("a contested terminal spelling must not resolve");
        };
        assert_eq!(candidates, ["Adep__Wrapped", "Blib__Wrapped"]);
        let message =
            ambiguous_constructor_error(&deep_expr("(var {} Wrapped)"), "Wrapped", &candidates)
                .message;
        assert!(
            message.contains("`Adep__Wrapped`") && message.contains("`Blib__Wrapped`"),
            "the diagnostic must name both candidates; got: {message}"
        );
        assert!(
            message.contains("unsupported:") && message.contains("[04-TOT-3]"),
            "the diagnostic must keep the branded lowering shape; got: {message}"
        );
    }

    #[test]
    fn constructor_resolution_keeps_same_name_declarations_deterministic() {
        // Settled by decision, with no open owner: two declarations
        // carrying the SAME name are not a qualified-versus-unqualified
        // ambiguity, so no name rule can separate them. The gate is the
        // checker, which resolves a same-name constructor by call shape
        // or rejects it, and this repair leaves the case exactly as it
        // was. Locked so a later change to that decision is deliberate.
        let checked = parse_and_check(concat!(
            "(deftype {} AaaBox () (variant {} Boxed (field {} amount (t-prim {} f32))))\n",
            "(deftype {} BbbBox () (variant {} Boxed (field {} amount (t-prim {} f32))))\n"
        ));
        let AdtConstructorResolution::Unique(definition) =
            resolve_adt_constructor_definition(&HostLoweringSession::new(&checked), "Boxed")
        else {
            panic!("same-name declarations keep their existing deterministic choice");
        };
        assert_eq!(definition.adt_name, "AaaBox");
    }

    #[test]
    fn adt_field_lookup_prefers_the_exact_type_name() {
        // Field access resolves through the ADT name and had the same
        // collision: admitting both declarations made them disagree on
        // the index of `amount` and rejected a valid access.
        let checked = parse_and_check(COLLIDING_DECLARATIONS);
        let found = lookup_adt_field_on_type(
            &HostLoweringSession::new(&checked),
            "Blib__Wrapped",
            &[],
            "amount",
        )
        .expect("no instantiation error")
        .expect("`amount` is declared on Blib__Wrapped");
        assert_eq!(found.0, 1, "`amount` is field 1 in the exact declaration");
    }

    #[test]
    fn adt_parameter_storage_prefers_the_exact_type_name() {
        // Same narrowing on the storage classification: the erased
        // parameter of one declaration must not be reported as stored
        // because a terminal-colliding declaration stores its own. The
        // declarations are built directly so the collision is the only
        // variable.
        let definitions = vec![
            GenericAdtConstructor {
                adt_name: "Adep__Column".to_string(),
                ctor_name: "Adep__Col".to_string(),
                parameters: vec!["a".to_string()],
                stored_parameters: vec![true],
                fields: Vec::new(),
            },
            GenericAdtConstructor {
                adt_name: "Blib__Column".to_string(),
                ctor_name: "Blib__Col".to_string(),
                parameters: vec!["n".to_string()],
                stored_parameters: vec![false],
                fields: Vec::new(),
            },
        ];
        assert!(
            adt_parameter_is_stored("Adep__Column", 0, &definitions),
            "the stored parameter of `Adep__Column` is stored"
        );
        assert!(
            !adt_parameter_is_stored("Blib__Column", 0, &definitions),
            "the erased parameter of `Blib__Column` must not inherit the \
             terminal-colliding declaration's storage classification"
        );
    }

    #[test]
    fn bare_constructor_reference_refuses_a_contested_terminal_spelling() {
        // Caller disposition: whether a bare reference is a construction
        // at all depends on which declaration answers it, so ambiguity
        // must reject rather than fall through to the variable path and
        // emit a bare C identifier nothing declares.
        let checked = parse_and_check(concat!(
            "(deftype {} Adep__Flag () (variant {} Adep__On (field {} weight (t-prim {} f32))))\n",
            "(deftype {} Blib__Flag () (variant {} Blib__On))\n"
        ));
        let message = lower_against(&checked, "(var {} On)")
            .expect_err("a contested bare constructor reference must reject");
        assert!(
            message.contains("`Adep__On`") && message.contains("`Blib__On`"),
            "the rejection must name both candidates; got: {message}"
        );
    }

    #[test]
    fn record_construction_refuses_a_contested_terminal_spelling() {
        // Caller disposition: the field-name check reports the AUTHORED
        // constructor, so validating against another declaration is what
        // produced the self-contradicting "has no field" rejection.
        let checked = parse_and_check(COLLIDING_DECLARATIONS);
        let message = lower_against(
            &checked,
            "(record {} Wrapped (kv {} extra (lit {} 1.0)) (kv {} amount (lit {} 2.0)))",
        )
        .expect_err("a contested record constructor must reject");
        assert!(
            message.contains("`Adep__Wrapped`") && message.contains("`Blib__Wrapped`"),
            "the rejection must name both candidates; got: {message}"
        );
        assert!(
            !message.contains("has no field"),
            "the contested case must not borrow the missing-field wording; got: {message}"
        );
    }

    #[test]
    fn positional_construction_refuses_a_contested_terminal_spelling() {
        // Caller disposition: the resolved declaration supplies each
        // argument's expected field type, which lowering then forces onto
        // the argument, so a wrong answer rewrites the payload's dtype.
        let checked = parse_and_check(concat!(
            "(deftype {} Adep__Box () (variant {} Adep__Boxed (t-prim {} f64)))\n",
            "(deftype {} Blib__Box () (variant {} Blib__Boxed (t-prim {} f32)))\n"
        ));
        let message = lower_against(&checked, "(app {} (var {} Boxed) (lit {} 1.0))")
            .expect_err("a contested positional constructor must reject");
        assert!(
            message.contains("`Adep__Boxed`") && message.contains("`Blib__Boxed`"),
            "the rejection must name both candidates; got: {message}"
        );
    }

    #[test]
    fn match_pattern_refuses_a_contested_terminal_spelling() {
        // Caller disposition: the resolved field list decides which index
        // each binder reads, so a wrong answer silently binds the other
        // package's field. The `type_env` fallback for names that are not
        // constructors must not absorb this.
        let checked = parse_and_check(COLLIDING_DECLARATIONS);
        let message = lower_against(
            &checked,
            "(match {} (var {} w) (arm {} (pat-record {} Wrapped \
             (kv {} amount (pat-var {} a))) () (var {} a)))",
        )
        .expect_err("a contested match constructor must reject");
        assert!(
            message.contains("`Adep__Wrapped`") && message.contains("`Blib__Wrapped`"),
            "the rejection must name both candidates; got: {message}"
        );
    }
}

#[cfg(test)]
mod record_hoist_binder_vocabulary_tests {
    use super::*;
    use chelis_deep::role::{ChildStampRole, child_stamp_role};
    use chelis_deep::tag::DeepTag;

    /// Parse Surf and hand back the canonical Deep the lowerer receives.
    ///
    /// The oracle builds every fixture this way and never constructs an `Expr`
    /// by hand. Round 2 is why: the previous version built its typed-parameter
    /// fixture as an `UnknownForm`, a shape the parser does not produce at a
    /// binder position, so its typed leg was green against a spelling that
    /// never occurs while the real one -- an unstamped list whose head is the
    /// name -- fell through undecoded. An oracle over a representation nobody
    /// emits proves nothing about the representation everybody emits.
    fn deep_program(source: &str) -> Vec<Expr> {
        let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
        chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar")
    }

    /// The body of the named `def`, which is the subtree the hoist walks.
    fn def_body(program: &[Expr], name: &str) -> Expr {
        let (_, body) = find_top_level_def_named(program, name).expect("the def");
        let Some((DeepTag::Fn, _, kids)) = stamped_parts(body) else {
            panic!("a def body is a `fn` node");
        };
        kids.get(1).expect("the fn body").clone()
    }

    fn bound_names(source: &str, def: &str) -> Result<Vec<String>, String> {
        let program = deep_program(source);
        let mut out = UnordSet::new();
        names_bound_in(&def_body(&program, def), &mut out)?;
        Ok(out.into_sorted())
    }

    /// One Surf spelling that binds `shadowed` inside `f`'s body, named by the
    /// binder form it exercises. Every entry is real source the parser accepts.
    fn binder_spellings() -> Vec<(&'static str, String, &'static str)> {
        vec![
            (
                "typed fn parameter",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = \
                 (fn (shadowed: tensor[1, f32]) -> mul(shadowed, shadowed))(x)\n"
                    .to_string(),
                "shadowed",
            ),
            (
                "untyped fn parameter",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = \
                 (fn (shadowed) -> mul(shadowed, shadowed))(x)\n"
                    .to_string(),
                "shadowed",
            ),
            (
                "two typed fn parameters",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = \
                 (fn (shadowed: tensor[1, f32], other: tensor[1, f32]) -> mul(shadowed, other))(x, x)\n"
                    .to_string(),
                "shadowed",
            ),
            (
                // A typed parameter whose NAME is also a Deep tag spelling.
                // `stamped_parts` would read `(params {type: ..})` as a
                // vocabulary node and collect nothing, so the desugarer emits
                // `^{:type ..} params` instead and the reader's `MetaExpr` arm
                // is what keeps this whole family correct. Round 3 found that
                // arm unguarded: mutating it left every other row green while
                // shipping round 2's defect for this spelling.
                "typed fn parameter named after a Deep tag",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = \
                 (fn (params: tensor[1, f32]) -> mul(params, params))(x)\n"
                    .to_string(),
                "params",
            ),
            (
                "let binding",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = {\n  shadowed = mul(x, x)\n  shadowed\n}\n"
                    .to_string(),
                "shadowed",
            ),
            (
                "match pattern binder",
                "type Holder =\n  | Holder { v: tensor[1, f32] }\n\n\
                 def f(x: tensor[1, f32]) -> tensor[1, f32] = \
                 match Holder { v: x } with { | Holder { v: shadowed } => mul(shadowed, shadowed) }\n"
                    .to_string(),
                "shadowed",
            ),
            (
                "pipe stage over a let binder",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = {\n  \
                 shadowed = x |> mul(x)\n  shadowed\n}\n"
                    .to_string(),
                "shadowed",
            ),
        ]
    }

    /// The vocabulary oracle for chelis#1266's hoist, built through the parser.
    ///
    /// Every binder spelling Surf can write must yield its name, because the
    /// hoist substitutes a local for a projection and must never do so under a
    /// binder it could not see. A spelling that yields nothing is either a
    /// silent wrong answer or, with the fail-closed exit, a definition whose
    /// hoist is abandoned and whose C lane then refuses.
    ///
    /// EVIDENTIARY STATUS: regression test. Measured RED at `2570da8d1` on the
    /// "typed fn parameter" and "two typed fn parameters" rows, where the
    /// reader refused with "a binder position of `fn` at index 0 carries a
    /// spelling this walk cannot read a name from" and bound nothing.
    #[test]
    fn every_surf_binder_spelling_yields_its_name() {
        for (form, source, binder) in binder_spellings() {
            let read = bound_names(&source, "f");
            let names = read.unwrap_or_else(|error| panic!("{form}: the reader refused: {error}"));
            assert!(
                names.iter().any(|name| name == binder),
                "{form}: bound {names:?}, which does not include `{binder}`"
            );
        }
    }

    /// The role table is the authority for WHICH positions bind, and it must
    /// keep naming the forms above. A tag that gains a binder position joins
    /// this set on its own.
    ///
    /// EVIDENTIARY STATUS: disposition lock on the representation choice.
    #[test]
    fn the_role_table_declares_the_binder_positions_the_reader_uses() {
        let mut tags = Vec::new();
        for tag in DeepTag::ALL {
            for arity in [1usize, 2, 3] {
                for index in 0..arity {
                    if child_stamp_role(tag, index, arity) == ChildStampRole::Binder
                        && !tags.contains(&tag)
                    {
                        tags.push(tag);
                    }
                }
            }
        }
        for required in [DeepTag::Params, DeepTag::Bind, DeepTag::PatVar, DeepTag::Fn] {
            assert!(
                tags.contains(&required),
                "`{}` must declare a binder position: {:?}",
                required.as_str(),
                tags.iter().map(|t| t.as_str()).collect::<Vec<_>>()
            );
        }
    }

    /// The fail-closed half: a binder position carrying a spelling the reader
    /// cannot decode refuses, so the caller abandons the hoist rather than
    /// treating the name as unbound.
    ///
    /// This one is built by hand deliberately, because its whole subject is a
    /// shape the parser does NOT produce. It is the only hand-built fixture
    /// here, and it asserts the refusal path rather than a decode.
    ///
    /// EVIDENTIARY STATUS: disposition lock.
    #[test]
    fn an_undecodable_binder_position_refuses_instead_of_reading_no_name() {
        let span = chelis_deep::span::Span::new(0, 0);
        // An UNSTAMPED form whose head is not a name: no closed tag matches
        // it, so it is not a container the walk descends, and nothing in it
        // spells a binder.
        let unreadable = Expr::Node(
            Box::new(chelis_deep::node::Node::new(
                DeepTag::Params,
                chelis_deep::Metadata::default(),
                vec![Expr::BareList(
                    vec![Expr::Atom(Atom::Int(1.into()), span)],
                    span,
                )],
            )),
            span,
        );
        let mut out = UnordSet::new();
        let read = names_bound_in(&unreadable, &mut out);
        assert!(
            read.is_err(),
            "a binder position holding a node with no name must refuse, got {out:?}"
        );
        assert!(
            read.unwrap_err().contains("params"),
            "the refusal names the construct"
        );
    }
}
