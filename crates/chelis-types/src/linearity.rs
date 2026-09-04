use chelis_deep::DeepTag;
use chelis_unord::{UnordMap, UnordSet};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::rc::Rc;

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use serde::{Deserialize, Serialize};

use crate::CheckedProgram;
use crate::cancel::CancelToken;
use crate::errors::{CheckError, CheckErrorKind};
use crate::infer::SignatureInferenceMetadata;
use crate::pipe_stage::resolve_pipe_stage_callee;
use crate::types::Type;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearityInfo {
    reusable_inputs_by_offset: UnordMap<usize, usize>,
}

impl LinearityInfo {
    pub fn reusable_input_for_span(&self, span: Span) -> Option<usize> {
        self.reusable_inputs_by_offset.get(&span.offset).copied()
    }

    fn mark_reusable_input(&mut self, span: Span, input_index: usize) {
        self.reusable_inputs_by_offset
            .entry(span.offset)
            .or_insert(input_index);
    }

    /// Merge two linearity infos into one. Used by
    /// [`crate::CheckedProgram::compose`] to reconstitute a whole-program
    /// linearity map from a cached library half plus a freshly-checked
    /// new-code half. `self` (the library half) wins on an offset clash;
    /// in practice the two halves carry disjoint span offsets because
    /// they come from separately-parsed source regions.
    pub fn merged_with(&self, other: &LinearityInfo) -> LinearityInfo {
        let mut reusable_inputs_by_offset = self.reusable_inputs_by_offset.clone();
        for (offset, input_index) in other.reusable_inputs_by_offset.to_sorted() {
            reusable_inputs_by_offset
                .entry(*offset)
                .or_insert(*input_index);
        }
        LinearityInfo {
            reusable_inputs_by_offset,
        }
    }
}

#[derive(Debug, Clone)]
enum BindingState {
    Live { borrow_sites: Vec<String> },
    Consumed(ConsumeSite),
}

/// Discrimination axis on a consume site (Linearity-F1).
///
/// Replaces the string-prefix check on `ConsumeSite::description`
/// (formerly at `read_or_error`) with a typed field. Phase 0 spec
/// lock (`docs/design/compiler_cleanup_0_7_8_spec_lock.md` Contract 1)
/// pins this as two variants; tuple-destructure tmp bindings are
/// handled as `Aliasing` (for the `let __chelis_tmp = (var ...)`
/// shape) or `Structural` (for the `(tuple-get ...)` reads) by the
/// same rules as any other consume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConsumeKind {
    /// `let alias = x` and similar var-RHS bindings.  At the IR
    /// level `lower_let` maps `alias` to the same NodeId as `x`,
    /// so the value is structurally shared rather than destroyed.
    /// Later borrow-reads of `x` must succeed.  See
    /// `spec/design/implicit_linearity.md` "Copy Insertion" and
    /// "Borrows do not count as fan-out".
    Aliasing,
    /// Every other consume site: realize / drop / store, app-arg,
    /// pipe-stage, closure capture, match scrutinee, etc.  The
    /// value is gone after this point and later borrow-reads are
    /// `UseAfterConsume`.
    Structural,
}

#[derive(Debug, Clone)]
struct ConsumeSite {
    description: String,
    kind: ConsumeKind,
}

/// What a single binding generation was introduced by.  One record per
/// `LinearScope::declare`, keyed by the generation's `BindingId`.
///
/// All fields describe the *binding*, not the value's current state —
/// `BindingState` owns that. They are separate concerns: a binding can be
/// a destructured component and an alias at the same time (a component
/// desugars to `p = (var __chelis_tmpN)`), and `pop` must drop both
/// together.
#[derive(Debug, Clone, Default)]
struct BindingOrigin {
    /// Alias chain link (Linearity-AliasedConsume-F1).  `Some(source)`
    /// when `check_let` recorded a `let y = (var x)` binding as an
    /// `Aliasing` consume; `consume_var_expr` walks the chain via
    /// `resolve_alias_chain` and forwards a `Structural` consume to the
    /// underlying source binding's record.  Multi-level chains
    /// (`let z = y; let y = x`) are walked iteratively.
    ///
    /// The link is the *generation* the alias was taken against,
    /// resolved when the alias bind is recorded, and it never
    /// re-resolves (chelis#1209): re-binding the source's name neither
    /// re-points this link at the new generation nor lets a later
    /// destructure of that name capture the alias.
    alias: Option<BindingId>,
    /// Destructured-component mark (Linearity-F2, chelis#1200).  `true`
    /// when this binding was introduced by a `destructure: true` bind
    /// emitted by `chelis_surf::desugar` for a `let` whose pattern is not
    /// a bare `Var` — i.e. the name denotes a tuple component, or one of
    /// the `__chelis_tmpN` intermediates that carry components.  The
    /// `consume_var_expr` already-consumed arm reads this to decide
    /// whether a consume-after-consume is a hard Linearity-F2 error or
    /// the ordinary implicit-Copy fallthrough.
    ///
    /// This is a per-binding mark rather than a block-scoped depth
    /// counter on purpose.  A depth counter set by one destructuring
    /// `let` covers that let's *body*, and in a block every later
    /// statement is nested inside that body, so the gate fired for every
    /// variable in the rest of the block — including ordinary bindings
    /// with no relationship to the destructure (chelis#1200).
    ///
    /// This field is the *active F2 gate*, and it is region-relative:
    /// `clear_destructured_marks` drops it on branch entry because a
    /// branch body is a new declaration region.  It must never be used to
    /// answer "which binding carries this value" — see `component`.
    destructured: bool,
    /// Permanent destructured-component identity (chelis#1200 review
    /// finding 1).  Set with `destructured` when the bind is recorded, and
    /// *never* cleared by region entry.
    ///
    /// The two facts are genuinely different.  Whether F2 is armed for a
    /// name depends on where you are (a branch body is a fresh region);
    /// whether the name denotes a tuple component whose value lives in a
    /// `__chelis_tmpN` carrier is a property of the binding itself and is
    /// true everywhere the binding is visible.  Reading the region-relative
    /// mark to answer the identity question silently loses the carrier
    /// inside any branch: the closure-capture path then consumed the
    /// component's own entry, left the carrier `Consumed(Aliasing)`, and a
    /// later `realize` of the component upgraded that to `Structural`
    /// without a diagnostic — a consume inside a branch failed to survive
    /// the join, contradicting `spec/design/implicit_linearity.md`.
    component: bool,
}

/// Unique identity of one binding event (chelis#1209).
///
/// Minted by [`LinearScope::declare`], monotonically within one check
/// invocation, and never reused: every `let` bind, function parameter,
/// closure capture, match binder, and pre-declared top-level def gets
/// its own generation. All checker state lives in the [`BindingRecord`]
/// keyed by this id, so a name is only ever a lookup handle (`visible`
/// resolves a use site to the innermost live generation) and re-binding
/// a name cannot transfer or misroute state that belongs to an older
/// generation. Mirrors the `TypeVar(u32)` / `VarGen` shape in
/// `types.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct BindingId(u32);

/// Everything the checker tracks for one binding generation.
#[derive(Debug, Clone)]
struct BindingRecord {
    /// The source-level name this generation was introduced under.
    /// Diagnostics print the name the user wrote at the *use site*, not
    /// this field; it exists for the `pop` LIFO check and debugging.
    name: String,
    ty: Option<Expr>,
    state: BindingState,
    origin: BindingOrigin,
}

#[derive(Debug, Clone, Default)]
struct LinearScope {
    /// All checker state, keyed by binding generation.
    records: UnordMap<BindingId, BindingRecord>,
    /// Name -> stack of generations, innermost last. Shadowing pushes,
    /// scope exit pops. This map answers "which binding does this use
    /// site mean" and nothing else; every consumption mark, alias link,
    /// and component mark lives on the id-keyed record it resolved to.
    visible: UnordMap<String, Vec<BindingId>>,
    /// Generation counter. Shared (`Rc`) across clones so branch and
    /// closure scopes forked from one root cannot mint colliding ids;
    /// `Default` mints a fresh zero counter, which is what makes each
    /// check invocation pure (`check_linearity_with_context` purity
    /// contract). One root scope is built per check invocation, and
    /// every other scope must be a clone of it: a second
    /// `LinearScope::default()` mid-check would fork the counter.
    next: Rc<Cell<u32>>,
}

impl LinearScope {
    fn mint(&self) -> BindingId {
        let id = BindingId(self.next.get());
        self.next.set(id.0 + 1);
        id
    }

    fn declare<S: Into<String>>(&mut self, name: S, ty: Option<Expr>) -> BindingId {
        let name = name.into();
        let id = self.mint();
        // The origin starts blank: a re-`let` of `name` shadows any
        // prior alias link and component mark because the new generation
        // begins unmarked. Callers that establish either follow with
        // `record_alias` / `mark_destructured`, which flip fields on the
        // record this inserted.
        self.records.insert(
            id,
            BindingRecord {
                name: name.clone(),
                ty,
                state: BindingState::Live {
                    borrow_sites: Vec::new(),
                },
                origin: BindingOrigin::default(),
            },
        );
        self.visible.entry(name).or_default().push(id);
        id
    }

    fn pop(&mut self, id: BindingId) -> Option<(Option<Expr>, BindingState)> {
        let record = self.records.remove(&id)?;
        if let Some(stack) = self.visible.get_mut(&record.name) {
            debug_assert_eq!(
                stack.last(),
                Some(&id),
                "scope exit must unwind LIFO per name"
            );
            if let Some(position) = stack.iter().rposition(|entry| *entry == id) {
                stack.remove(position);
            }
            if stack.is_empty() {
                self.visible.remove(&record.name);
            }
        }
        Some((record.ty, record.state))
    }

    /// The innermost live generation for `name`, if any. This is the
    /// single point where a use site's name becomes an identity; every
    /// state read or write past it is id-keyed.
    fn top_id(&self, name: &str) -> Option<BindingId> {
        self.visible
            .get(name)
            .and_then(|stack| stack.last())
            .copied()
    }

    fn record(&self, id: BindingId) -> Option<&BindingRecord> {
        self.records.get(&id)
    }

    fn record_mut(&mut self, id: BindingId) -> Option<&mut BindingRecord> {
        self.records.get_mut(&id)
    }

    fn state(&self, id: BindingId) -> Option<&BindingState> {
        self.record(id).map(|record| &record.state)
    }

    fn ty(&self, name: &str) -> Option<&Expr> {
        self.top_id(name)
            .and_then(|id| self.record(id))
            .and_then(|record| record.ty.as_ref())
    }

    fn consume_id(&mut self, id: BindingId, site: ConsumeSite) {
        if let Some(record) = self.record_mut(id) {
            record.state = BindingState::Consumed(site);
        }
    }

    fn borrow(&mut self, name: &str, site: String) {
        if let Some(id) = self.top_id(name)
            && let Some(record) = self.record_mut(id)
            && let BindingState::Live { borrow_sites } = &mut record.state
        {
            borrow_sites.push(site);
        }
    }

    /// Every generation on every visible stack, sorted for
    /// deterministic iteration. Shadowed generations are included on
    /// purpose: an alias recorded against an older generation can
    /// consume it inside a branch even while its name is shadowed, so a
    /// join that only saw stack tops would drop that consume
    /// (chelis#1209).
    fn all_visible_ids(&self) -> Vec<BindingId> {
        let mut ids: Vec<BindingId> = self
            .visible
            .to_sorted()
            .into_iter()
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Record that the binding generation `alias` is an aliasing copy
    /// of the generation `source`.  The caller resolves `source` from
    /// its name *before* declaring `alias`, so the link points at the
    /// generation the alias was actually taken against — including for
    /// a self-rebind `x = x`, where the source is the older `x`
    /// (chelis#1209).  Used by the `Aliasing` consume producers in
    /// `check_let` and the top-level `def name = (var x)` arm of
    /// `check_top_level` per Linearity-AliasedConsume-F1.
    fn record_alias(&mut self, alias: BindingId, source: BindingId) {
        if let Some(record) = self.record_mut(alias) {
            record.origin.alias = Some(source);
        }
    }

    /// Record that the binding generation `id` is a destructured
    /// component (Linearity-F2, chelis#1200).  Takes the id `declare`
    /// returned for the bind, whose record starts unmarked.  Called by
    /// `check_let` for every name introduced by a `destructure: true`
    /// bind.
    fn mark_destructured(&mut self, id: BindingId) {
        if let Some(record) = self.record_mut(id) {
            record.origin.destructured = true;
            record.origin.component = true;
        }
    }

    /// Whether the binding generation `id` currently carries the active
    /// F2 gate.  Generations that are gone (popped) answer `false`,
    /// matching the pre-#1200 behavior for anything outside a
    /// destructure.
    fn is_destructured_id(&self, id: BindingId) -> bool {
        self.record(id)
            .is_some_and(|record| record.origin.destructured)
    }

    /// Whether the binding generation `id` was introduced by a
    /// destructure, regardless of declaration region (chelis#1200 review
    /// finding 1).
    ///
    /// This is the identity question, and it is the one every
    /// *carrier-resolution* site must ask.  `is_destructured_id` answers
    /// the different, region-relative question of whether the F2 gate is
    /// armed here, and a branch body clears that.  Asking the armed-here
    /// question when you meant the identity question drops the carrier
    /// inside every branch.  The permanent flag rides the record, so
    /// this answer is stable across shadowing and region entry.
    fn is_component_id(&self, id: BindingId) -> bool {
        self.record(id)
            .is_some_and(|record| record.origin.component)
    }

    /// Drop the destructured-component marks on every currently-visible
    /// binding (Linearity-F2, chelis#1200 / reviewer ruling Q1).
    ///
    /// Called when entering a branch scope (`check_if`, `check_match`).
    /// Per `spec/design/implicit_linearity.md` §"Destructured components",
    /// entering a new declaration region re-declares the values it works
    /// on, and a fresh declaration shadows the mark — that is already what
    /// `check_fn` gets for free, because it `declare`s every capture in
    /// the closure's inner scope. Branch scopes clone the enclosing scope
    /// without re-declaring, so without this a component consumed twice
    /// inside one arm errored while the identical closure body compiled.
    /// The reviewer ruled the closure verdict correct, so branches drop
    /// the marks on entry.
    ///
    /// Only the marks are dropped: `BindingState` and alias links are
    /// untouched, so consume tracking across the branch boundary and
    /// `join_branch_states` are unaffected, and a destructure *inside* the
    /// branch marks its own components normally.
    fn clear_destructured_marks(&mut self) {
        let ids = self
            .records
            .to_sorted()
            .into_iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        for id in ids {
            let record = self
                .records
                .get_mut(&id)
                .expect("collected binding id remains present");
            record.origin.destructured = false;
        }
    }

    /// Walk the alias chain from the generation `id` to the underlying
    /// non-alias source generation.  Returns `None` if `id` carries no
    /// alias link; returns `Some(source)` if it aliases `source`
    /// (possibly through one or more intermediate generations).
    /// Bounded by chain length, which is bounded by source-program
    /// nesting depth.
    ///
    /// Each hop follows the `BindingId` the alias was recorded against
    /// (chelis#1209), so the walk cannot be re-routed by a later
    /// re-binding of any name on the chain.  `let` aliases always point
    /// at strictly older generations, but top-level `def a = (var b)`
    /// aliases link pre-declared defs in program order and so can point
    /// forward; the visited set stays as the cycle guard for a mutual
    /// pair, treating a chain that closes a cycle as no alias at all.
    fn resolve_alias_chain(&self, id: BindingId) -> Option<BindingId> {
        let mut current = id;
        let mut visited: UnordSet<BindingId> = UnordSet::new();
        let mut walked = false;
        loop {
            if !visited.insert(current) {
                return None;
            }
            match self.record(current).and_then(|record| record.origin.alias) {
                Some(source) => {
                    current = source;
                    walked = true;
                }
                None => return if walked { Some(current) } else { None },
            }
        }
    }
}

struct Checker {
    errors: Vec<CheckError>,
    info: LinearityInfo,
    top_level_types: BTreeMap<String, Expr>,
    /// Names of ADTs whose definitions (transitively) carry a tensor
    /// field. Computed once per `check_linearity` call by walking
    /// `deftype` declarations in `annotated_exprs`. Used by
    /// `expr_is_owned_or_borrow_linear` so `&adt_value` is accepted as
    /// a borrow whenever the ADT's definition contains a tensor, not
    /// only when the ADT's type *arguments* contain one. Resolves the
    /// downstream blocker for `School` P1.5 (BatchNorm) and P2.5
    /// (optimizer `_step_tree`) where `&BatchNormParams` /
    /// `&AdamState[tensor[..]]` (with the tensor in a record field,
    /// not the ADT-arg position) was rejected with `InvalidBorrow`.
    ///
    /// Keyed on bare ADT name. Two-name collisions are rejected up
    /// front by `collect_declarations` in `infer.rs` with
    /// `CheckErrorKind::DuplicateDefinition`, so by the time the
    /// linearity checker runs, every name in this set corresponds to
    /// exactly one `deftype`. That guarantee is what makes a bare
    /// `String` key safe here; without it the carrier set would be
    /// order-dependent (last-write-wins via `UnordMap::insert` in
    /// `compute_tensor_carrying_adts`). Once Chelis gains qualified
    /// ADT names, this set should migrate to a `Set<AdtId>` queried
    /// off the shared `AdtRegistry` instead of reparsing `deftype`
    /// exprs here. See the function-level note on
    /// [`compute_tensor_carrying_adts`].
    tensor_carrying_adts: UnordSet<String>,
    /// Snapshot of `signature_inference` from the program under check.
    /// Used by `arg_is_borrowed` to recognize call-site borrow
    /// classification on user-defined functions whose params were
    /// inferred read-only (see `infer.rs:infer_signature_metadata`).
    /// Without this, a call to `def reader(t, k: tensor[..]) = t |> add(k)`
    /// would consume `t` at every callsite, defeating the auto-borrow
    /// inference that the inferencer already computed. Closes the
    /// chelis#229 sibling-sweep gap.
    signature_inference: SignatureInferenceMetadata,
    type_headers: crate::deep_type::TypeResolutionEnv,
}

impl Checker {
    fn push_diagnostic(&mut self, error: CheckError) {
        // Linearity-F2 / F3 W2 cascade close: every linearity
        // violation is an error.  W1 PR 1's destructure-cascade
        // warning channel closed once the corpus survey confirmed
        // zero surfaced warnings (see
        // `docs/investigations/linearity_destructure_cleanup_survey.md`).
        self.errors.push(error);
    }
}

/// Pre-declare top-level def names into `scope`, descending through
/// `(module {} name children...)` wrappers. Mirrors the
/// `top_level_decl_items` pattern in `infer.rs:1056-1074` so module-
/// wrapped defs participate in cross-statement linearity tracking.
fn pre_declare_top_level_defs(
    exprs: &[Expr],
    type_env: &BTreeMap<String, Expr>,
    scope: &mut LinearScope,
) {
    for expr in exprs {
        pre_declare_one(expr, type_env, scope);
    }
}

fn pre_declare_one(expr: &Expr, type_env: &BTreeMap<String, Expr>, scope: &mut LinearScope) {
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::Module => {
            for child in kids.iter().skip(1) {
                pre_declare_one(child, type_env, scope);
            }
        }
        DeepTag::Def => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                scope.declare(name, type_env.get(name).cloned());
            }
        }
        _ => {}
    }
}

pub fn check_linearity(program: &CheckedProgram) -> Result<CheckedProgram, Vec<CheckError>> {
    let tensor_carrying_adts = compute_tensor_carrying_adts(program.annotated_exprs());
    let mut checker = Checker {
        errors: Vec::new(),
        info: LinearityInfo::default(),
        top_level_types: program.type_env().clone(),
        tensor_carrying_adts,
        signature_inference: program.signature_inference().clone(),
        type_headers: program.type_headers().clone(),
    };
    let mut scope = LinearScope::default();

    pre_declare_top_level_defs(program.annotated_exprs(), program.type_env(), &mut scope);

    // chelis#930: cooperative cancellation at top-level-declaration
    // granularity — the same grain as the type checker's own schedule, and
    // linearity is the third-largest front-end phase on a declaration-heavy
    // program. Abandoning the walk proves nothing about the tail, so this is a
    // hard failure rather than a partial `Ok` (covered-or-rejected).
    let cancel = crate::cancel::current_cancel_token();
    for expr in program.annotated_exprs() {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            checker
                .errors
                .push(crate::cancel::cancellation_check_error());
            break;
        }
        checker.check_top_level(expr, &mut scope);
    }

    if checker.errors.is_empty() {
        Ok(program.clone().with_linearity(checker.info))
    } else {
        Err(checker.errors)
    }
}

/// Phase E: check linearity of `new_program` against an outer-scope
/// `library_program` whose linearity was already validated when its
/// context was built.
///
/// Library bindings are treated as always-available references —
/// library tensor parameters are scoped to their owning library `def`,
/// not to new-code defs, and so MUST NOT be added to new-code's
/// "consumed" set when called. To enforce that, this entry point walks
/// ONLY `new_program.annotated_exprs()` for body checks; library
/// bodies are never re-walked. Library def names are pre-declared in
/// the top-level scope (with their unioned types from
/// `new_program.type_env()`) so that `(var libname)` references in
/// new code resolve correctly. Library defs all have function types,
/// which the linearity checker treats as non-linear, so referencing
/// one never triggers consumption of new-code free variables.
///
/// Per Phase E acceptance: for `(library, snippet)`, this function's
/// Result on `snippet` must match `check_linearity(library + snippet)`'s
/// Result on the snippet portion. The function is pure — repeated calls
/// with the same `library_program` see no leaked state from prior new
/// programs.
///
/// `library_program` is consumed only by reference; it is left
/// unchanged.
pub fn check_linearity_with_context(
    library_program: &CheckedProgram,
    new_program: &CheckedProgram,
) -> Result<CheckedProgram, Vec<CheckError>> {
    // Pre-compute library callable signatures: a name set keyed on each
    // library `def`, used to ensure new-code call sites referring to a
    // library def don't accidentally walk the library body. The set is
    // currently informational — the checker never walks call targets, so
    // simply not feeding library exprs to `check_top_level` is what
    // enforces the "don't re-walk library bodies" invariant. The
    // pre-computation here is the documented contract surface.
    let _library_callables: UnordSet<String> = library_program
        .annotated_exprs()
        .iter()
        .filter_map(|expr| {
            tagged_children(expr, DeepTag::Def)?
                .first()
                .and_then(symbol_name)
                .map(str::to_string)
        })
        .collect();

    // Top-level types come from new_program.type_env(), which Phase C
    // already unioned (library + new-code). New-code types win on shadow.
    //
    // The carrier set comes from BOTH library and new code, chained
    // into one `compute_tensor_carrying_adts` call so the fixed-point
    // sees every ADT at once. Library decls can introduce tensor-
    // carrying ADTs that new-code borrows; new-code can introduce
    // additional ones whose fields reference library ADTs (the
    // cross-package transitive case). Two independent calls — one
    // per half, each with its own local set — would miss any new-
    // code ADT whose carrying status depends on a library ADT, even
    // though both halves are eventually unioned.
    //
    // ALWAYS-RECOMPUTE INVARIANT: this helper is recomputed from the
    // chained iterator on every call, with no per-call state held by
    // `Checker`, the library `CheckedProgram`, or any global cache.
    // The fixed-point bound (`O(adt_count)` passes, each `O(adt_count
    // * field_count)`) is small relative to the per-expression
    // linearity walk that follows. Future change risk: if a later
    // refactor caches `tensor_carrying_adts` per `CheckedProgram` and
    // composes the library's cached set with a fresh new-code pass,
    // the fixed-point will not re-resolve new-code ADTs whose
    // carrying status depends on library ADTs and borrow semantics
    // will silently desync. The locking test for this contract lives
    // at `tests/linearity_with_context.rs::
    // check_linearity_with_context_is_pure_across_repeated_calls`.
    // Once `CheckedProgram` exposes a shared `AdtRegistry`, the
    // registry query replaces this helper entirely (and the new
    // call site is responsible for re-establishing the same union-
    // and-recompute discipline).
    let tensor_carrying_adts = compute_tensor_carrying_adts(
        library_program
            .annotated_exprs()
            .iter()
            .chain(new_program.annotated_exprs().iter()),
    );
    // Merge library + new-code signature inference. New-code wins on
    // name clash, matching `CheckedProgram::compose`'s rule. Library
    // function signatures are visible at new-code callsites so library
    // user functions inferred as borrow-arg are recognized as such.
    let mut merged_signature_inference = library_program.signature_inference().clone();
    for (name, sig) in &new_program.signature_inference().functions {
        merged_signature_inference
            .functions
            .insert(name.clone(), sig.clone());
    }
    let mut merged_type_headers = library_program.type_headers().clone();
    merged_type_headers.extend_from(new_program.type_headers());
    let mut checker = Checker {
        errors: Vec::new(),
        info: LinearityInfo::default(),
        top_level_types: new_program.type_env().clone(),
        tensor_carrying_adts,
        signature_inference: merged_signature_inference,
        type_headers: merged_type_headers,
    };

    let mut scope = LinearScope::default();

    // Pre-declare library def names so `(var libname)` references in
    // new code resolve to a Live binding with the library's function
    // type. Function types are non-linear (no tensor content), so a
    // pure reference never triggers consumption of new-code locals.
    // Recurse through module wrappers so library .ch sources written
    // with `module Foo` participate in pre-declaration the same as
    // bare-top-level library sources (Linearity-F3 PR 1).
    pre_declare_top_level_defs(
        library_program.annotated_exprs(),
        new_program.type_env(),
        &mut scope,
    );

    // Pre-declare new-code def names. New-code shadows library on
    // collision (declare last → top of stack wins).
    pre_declare_top_level_defs(
        new_program.annotated_exprs(),
        new_program.type_env(),
        &mut scope,
    );

    // Walk ONLY new-code bodies. Library bodies are never re-walked,
    // so library tensor parameters never enter the new-code scope.
    // chelis#930: cancellable at the same grain as [`check_linearity`].
    let cancel = crate::cancel::current_cancel_token();
    for expr in new_program.annotated_exprs() {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            checker
                .errors
                .push(crate::cancel::cancellation_check_error());
            break;
        }
        checker.check_top_level(expr, &mut scope);
    }

    if checker.errors.is_empty() {
        Ok(new_program.clone().with_linearity(checker.info))
    } else {
        Err(checker.errors)
    }
}

impl Checker {
    fn check_top_level(&mut self, expr: &Expr, scope: &mut LinearScope) {
        // Linearity-F3 PR 1 + PR 2: recurse through `(module {} name
        // children...)` wrappers so module-wrapped top-level defs
        // participate in cross-statement linearity tracking. PR 1
        // routed diagnostics raised inside this recursion to a
        // warning channel for a deprecation window; PR 2 removed that
        // channel and unified the severity with bare-top-level
        // violations, so all `push_diagnostic` calls route to
        // `Checker::errors`.
        if let Some((DeepTag::Module, _, kids)) = stamped_parts(expr) {
            for child in kids.iter().skip(1) {
                self.check_top_level(child, scope);
            }
            return;
        }
        if let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) {
            if let (Some(name), Some(body)) = (kids.first().and_then(symbol_name), kids.get(1))
                && !(is_var_expr(body) && var_name(body) == Some(name))
            {
                if matches!(get_tag_expr(body), Some(DeepTag::Borrow)) {
                    self.invalid_borrow(body, "borrow cannot be returned from a function");
                    return;
                }
                // V2-F4: top-level `def name() = x` where the body is a
                // bare `(var x)` of an owned-linear type is an
                // aliasing binding consume; at the IR level
                // `lower_var` returns the cached `bindings["x"]` node
                // for both `(var x)` and the new top-level `name`, so
                // the value is structurally shared, not destroyed.
                // Mirror the `check_let` path
                // (`linearity.rs:397-407`) by tagging the consume
                // with a `"binding `name` at <site>"` description and
                // `ConsumeKind::Aliasing`. The kind feeds the PR #29
                // `read_or_error` tolerance (typed via `ConsumeKind`
                // since Linearity-F1), letting subsequent borrow
                // reads of the original variable succeed. Without
                // this branch the body would fall through to
                // `check_expr -> consume_var_expr(generic_site)` and
                // record a `Structural` consume, which the tolerance
                // does not match.
                if is_var_expr(body) && self.expr_is_owned_linear(body, scope) {
                    // Resolve both generations by name here: top-level
                    // defs are pre-declared exactly once each, so the
                    // stacks are static during this walk and the lookup
                    // is the record-time resolution chelis#1209 wants.
                    // A `def a = (var b)` alias can point at a def
                    // declared *later* in program order; that is fine —
                    // the id is already minted by pre-declaration.
                    let alias_link = var_name(body)
                        .and_then(|source| Some((scope.top_id(name)?, scope.top_id(source)?)));
                    self.consume_var_expr(
                        body,
                        scope,
                        ConsumeSite {
                            description: format!("binding `{name}` {}", diag_site(body)),
                            kind: ConsumeKind::Aliasing,
                        },
                    );
                    if let Some((alias_id, source_id)) = alias_link {
                        scope.record_alias(alias_id, source_id);
                    }
                } else {
                    self.check_expr(body, scope);
                }
            }
            return;
        }
        self.check_expr(expr, scope);
    }

    fn check_expr(&mut self, expr: &Expr, scope: &mut LinearScope) {
        match expr {
            Expr::Atom(_, _) => {}
            Expr::Map(map, _) => {
                for (_, value) in &map.entries {
                    self.check_expr(value, scope);
                }
            }
            Expr::MetaExpr(meta, _) => self.check_expr(&meta.expr, scope),
            Expr::List(list, _) => match get_tag(list) {
                Some(DeepTag::Var) => self.consume_var_expr(expr, scope, generic_site(expr)),
                Some(DeepTag::Copy) => self.check_copy(list, scope),
                Some(DeepTag::Realize) => self.check_realize(expr, list, scope),
                Some(DeepTag::Borrow) => {
                    self.invalid_borrow(expr, "borrow is only valid as a direct call argument")
                }
                Some(DeepTag::App) => self.check_app(expr, list, scope),
                Some(DeepTag::Pipe) => self.check_pipe(list, scope),
                Some(DeepTag::Let) => self.check_let(list, scope),
                Some(DeepTag::Fn) => self.check_fn(expr, list, scope),
                Some(DeepTag::If) => self.check_if(list, scope),
                Some(DeepTag::Match) => self.check_match(list, scope),
                // `tuple-get(t, i)` is a read of `t`, not a
                // consume.  The implicit-linearity IR pass inserts a
                // Copy where needed.  Pre Linearity-F2 the
                // distinction was invisible because destructure tmp
                // bindings were untyped and the consume was skipped
                // via `expr_is_owned_linear`; now that the type
                // metadata threads onto the tmps, `(var __chelis_tmpN)`
                // would otherwise consume through `generic_site` and
                // forward (via the alias chain) to the underlying
                // tuple source.  Treat the var argument as a borrow.
                Some(DeepTag::TupleGet) => self.check_tuple_get(list, scope),
                _ => {
                    for child in children(list) {
                        self.check_expr(child, scope);
                    }
                }
            },
            // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
            Expr::Node(node, span) => {
                let bridged = Expr::List(node.to_list(*span), *span);
                self.check_expr(&bridged, scope);
            }
            Expr::BareList(elems, _) => {
                for elem in elems {
                    self.check_expr(elem, scope);
                }
            }
            Expr::UnknownForm(data) => {
                for child in &data.children {
                    self.check_expr(child, scope);
                }
            }
        }
    }

    fn check_copy(&mut self, list: &List, scope: &mut LinearScope) {
        if let Some(child) = children(list).first() {
            if let Some(borrowed) = borrow_inner(child) {
                self.check_borrow_arg(child, borrowed, scope);
            } else if is_var_expr(child) && self.expr_is_owned_linear(child, scope) {
                self.read_var_expr(child, scope);
            } else {
                self.check_expr(child, scope);
            }
        }
    }

    /// `tuple-get(t, i)` reads the i-th element of `t` without
    /// consuming `t`.  Treat the var argument as a borrow read so
    /// the surrounding destructure desugar of `let (a, b) = pair`
    /// (which produces two tuple-gets on the same source tmp) does
    /// not double-consume.  Linearity-F2: this method is added here
    /// because the destructure type-metadata threading made the
    /// previously-untyped tmp consumes visible.
    fn check_tuple_get(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if let Some(target) = kids.first() {
            if is_var_expr(target) && self.expr_is_owned_linear(target, scope) {
                self.read_var_expr(target, scope);
            } else {
                self.check_expr(target, scope);
            }
        }
        // Walk remaining children (the index lit) so any nested
        // expressions inside the index are still checked.
        for child in kids.iter().skip(1) {
            self.check_expr(child, scope);
        }
    }

    fn check_realize(&mut self, expr: &Expr, list: &List, scope: &mut LinearScope) {
        if let Some(child) = children(list).first() {
            if is_var_expr(child) && self.expr_is_owned_linear(child, scope) {
                self.consume_var_expr(child, scope, realize_site(expr));
            } else {
                self.check_expr(child, scope);
            }
        }
    }

    fn check_app(&mut self, expr: &Expr, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        let builtin = kids.first().and_then(var_name);
        if let Some(func) = kids.first() {
            self.check_expr(func, scope);
        }
        for (index, arg) in kids.iter().enumerate().skip(1) {
            if let Some(borrowed) = borrow_inner(arg) {
                self.check_borrow_arg(arg, borrowed, scope);
            } else if self.arg_is_borrowed(kids.first(), builtin, index - 1, scope)
                && is_var_expr(arg)
                && self.expr_is_owned_linear(arg, scope)
            {
                self.read_var_expr(arg, scope);
            } else if is_var_expr(arg) && self.expr_is_owned_linear(arg, scope) {
                self.consume_var_expr(arg, scope, app_site(expr, list));
            } else {
                self.check_expr(arg, scope);
            }
        }
        self.maybe_mark_reusable_app_input(expr, kids, scope);
    }

    // Issue #226 diagnosis (regression introduced indirectly by #183
    // `fix(types): substitute ADT type params into record-pattern
    // bindings`). Before #183 the destructured record-field bindings
    // (`PosEmbedParams { table: table }` -> a new local `table`) carried
    // a fresh type variable in the annotated Deep, so
    // `expr_is_owned_linear` returned `false` and `check_pipe`
    // accidentally accepted `table |> shape(0)` followed by a later
    // `gather(table, ...)`. #183 stamps the resolved field type onto
    // pattern bindings (the right fix in isolation); that exposed a
    // pre-existing gap in `check_pipe`. The branch below classifies the
    // piped value with `var_name(stage)`, which only matches the bare-
    // var stage shape `(var f)`. For any non-bare-var stage
    // `chelis_surf::desugar::desugar_pipe_stage` emits a synthesized
    // `(fn (params __chelis_pipe) (app callee ... (var __chelis_pipe)
    // ...))` lambda, so `var_name(stage)` returns `None` and
    // `arg_is_borrowed(stage, None, 0, scope)` falls through to a
    // function-type lookup on the LAMBDA itself, not on the inner
    // `callee`. That means borrow-arg builtins called with explicit
    // arguments (`shape(0)`, `add(y)`, `mul(k)`, `matmul(w)`, ...) get
    // mis-tagged as structural consumes of the piped variable, tripping
    // `UseAfterConsume` on any later read with a malformed "pipe into
    // stage at offset 0 from offset 0" message (both offsets zero
    // because the synthesized lambda has no source span). The upcoming
    // fix introduces `resolve_pipe_stage_callee` and peers through the
    // synthesized lambda to recover the inner callee and the piped
    // value's arg position before consulting `arg_is_borrowed`.
    fn check_pipe(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if kids.is_empty() {
            return;
        }
        let mut current = &kids[0];
        for stage in &kids[1..] {
            // Pipe stages with explicit args (e.g. `x |> shape(0)`)
            // are desugared by `chelis_surf::desugar::desugar_pipe_stage`
            // into a synthesized one-arg lambda
            // `(fn (params __chelis_pipe) (app callee ... (var __chelis_pipe) ...))`
            // where the piped value lands at the `__chelis_pipe`
            // position in the inner app's args. To classify whether the
            // piped value is borrowed (read) or consumed by the stage,
            // look through that lambda and ask the inner callee at its
            // actual arg position. Without this peering, every
            // non-bare-var stage falls through to the "stage callee
            // unknown" branch and the piped value is treated as a
            // structural consume — which mis-fires whenever a borrow-
            // arg builtin (`shape`, `add`, `mul`, ...) is invoked with
            // explicit non-piped args. Closes issue #226.
            let (callee_expr, callee_builtin, piped_arg_index) = resolve_pipe_stage_callee(stage);
            if self.arg_is_borrowed(callee_expr, callee_builtin, piped_arg_index, scope) {
                if is_var_expr(current) && self.expr_is_owned_linear(current, scope) {
                    self.read_var_expr(current, scope);
                } else {
                    self.check_expr(current, scope);
                }
            } else if is_var_expr(current) && self.expr_is_owned_linear(current, scope) {
                self.consume_var_expr(current, scope, pipe_site(current, stage));
            } else {
                self.check_expr(current, scope);
            }
            current = stage;
        }
    }

    fn check_borrow_arg(&mut self, borrow_expr: &Expr, inner: &Expr, scope: &mut LinearScope) {
        if !is_var_expr(inner) {
            self.invalid_borrow(
                borrow_expr,
                "borrowed arguments must be direct variable references",
            );
            return;
        }
        if !self.expr_is_owned_or_borrow_linear(inner, scope) {
            self.invalid_borrow(
                borrow_expr,
                "borrowed arguments must be tensor or tensor-carrying values",
            );
            return;
        }
        self.read_var_expr(inner, scope);
    }

    fn check_let(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if kids.len() < 2 {
            return;
        }
        let mut pushed = Vec::new();
        // Linearity-F2 component tracking (chelis#1200).  When the bind
        // is one of the desugarer-synthesized destructure intermediates
        // (`__chelis_tmpN` or a user-visible destructure component,
        // tagged with `destructure: true`), mark the names it introduces
        // as destructured components so the `consume_var_expr`
        // already-consumed arm fires as an error *for those names* rather
        // than the silent fallthrough used by regular bindings.
        //
        // The mark is per-binding and not a scope: the binding values below
        // are checked in the enclosing scope and are ordinary variables
        // (`_ = eat(v)` consumes `v`, which is not a component), and in a
        // block every later statement is nested in this let's body, so a
        // scope-shaped gate would classify the whole rest of the block as
        // destructured.
        let bind_introduces_destructure = bind_introduces_destructure_tmp(&kids[0]);
        if let Some(bind_kids) = tagged_children(&kids[0], DeepTag::Bind) {
            let mut index = 0;
            while index + 1 < bind_kids.len() {
                let Some(name) = symbol_name(&bind_kids[index]) else {
                    index += 2;
                    continue;
                };
                let value = &bind_kids[index + 1];
                // Resolve the alias source's generation BEFORE the
                // `declare` below (chelis#1209): the link must point at
                // the binding the alias was taken against.  Resolving
                // after the declare would make a self-rebind `x = x`
                // link the new generation to itself instead of to the
                // older `x` it actually aliases.
                let mut alias_source_id: Option<BindingId> = None;
                if is_var_expr(value) && self.expr_is_owned_linear(value, scope) {
                    alias_source_id = var_name(value).and_then(|source| scope.top_id(source));
                    self.consume_var_expr(
                        value,
                        scope,
                        ConsumeSite {
                            description: format!("binding `{name}` {}", diag_site(value)),
                            kind: ConsumeKind::Aliasing,
                        },
                    );
                } else if matches!(get_tag_expr(value), Some(DeepTag::Borrow)) {
                    self.invalid_borrow(value, "borrow cannot be stored in a binding");
                } else {
                    self.check_expr(value, scope);
                }
                let id = scope.declare(name, self.expr_type(value, scope).cloned());
                if let Some(source_id) = alias_source_id {
                    scope.record_alias(id, source_id);
                }
                // The marker governs the bindings introduced by this bind
                // and nothing else.  The binding values were checked above
                // in the enclosing scope, so an ordinary source variable
                // appearing in a destructure's RHS is never classified as a
                // component.
                if bind_introduces_destructure {
                    scope.mark_destructured(id);
                }
                pushed.push(id);
                index += 2;
            }
        }
        self.check_expr(&kids[1], scope);
        for id in pushed.into_iter().rev() {
            self.pop_and_check(scope, id, expr_scope_end(&kids[1]));
        }
    }

    fn check_fn(&mut self, expr: &Expr, list: &List, outer_scope: &mut LinearScope) {
        let kids = children(list);
        if kids.len() < 2 {
            return;
        }

        let params = param_names(&kids[0]);
        let captured = free_vars(&kids[1], &params);
        let mut inner_scope = outer_scope.clone();
        let body = &kids[1];
        // Build a temporary `UnordMap<String, Type>` of currently-known
        // user-fn display signatures so the closure-body consuming-use
        // probe can recognize user-defined borrow-arg callees inside
        // the body. The probe's secondary `type_env` lookup also
        // matches direct annotations in `top_level_types`, but
        // `available_signatures` matches by `Type` (the inferencer's
        // own metadata) so passing the inferred display signatures
        // covers user fns whose params were auto-borrow-inferred.
        let available_signatures: UnordMap<String, Type> = self
            .signature_inference
            .functions
            .iter()
            .map(|(name, sig)| (name.clone(), sig.display_signature.clone()))
            .collect();
        for name in captured {
            let Some(ty) = outer_scope.ty(&name) else {
                continue;
            };
            if !type_expr_contains_tensor(ty, &self.tensor_carrying_adts) {
                continue;
            }
            // chelis#237 closure-capture spurious-consume gap: if the
            // closure body never uses `name` in a structurally-consuming
            // position (only borrow-arg reads, etc.), treat the capture
            // as a borrow of the outer binding rather than a structural
            // consume. Mirrors the auto-borrow inference at
            // `infer.rs:1454` (`infer_signature_metadata`) which marks
            // a function parameter read-only when its body has no
            // consuming use. Without this, a closure like
            // `fn (i) -> add(c, c)` (capture only borrow-read) would
            // consume the outer `c` at closure-creation time, so a
            // later borrow-read of `c` outside the closure trips
            // `UseAfterConsume` with the diagnostic "was already
            // consumed by closure capture at <site>". The probe
            // reuses `infer::param_has_consuming_use` so the consume
            // classification stays aligned with what
            // `param_has_consuming_use_inner` already enforces for
            // top-level function-param inference.
            //
            // Negative parity (a closure body that *does* consume the
            // capture — e.g. `fn (i) -> realize(c)` or returning the
            // capture as the body's value) still goes through the
            // structural-consume branch, so `read_or_error` plus
            // `outer_scope.consume` keep firing.
            let body_consumes = crate::infer::param_has_consuming_use(
                body,
                name.as_str(),
                &available_signatures,
                &self.top_level_types,
                &self.type_headers,
            )
            .unwrap_or_else(|result| {
                // The resolver diagnostic is part of linearity's
                // authoritative error result. Classify conservatively while
                // the walk finishes, but never expose that fallback as Ok.
                self.errors.extend(result.errors);
                true
            });
            if body_consumes {
                self.read_or_error(name.as_str(), expr, outer_scope);
                // chelis#1200: forward the capture consume to a
                // destructured component's carrier.
                //
                // A component is an alias of a `__chelis_tmpN` the user
                // cannot name, and that carrier is the component's only
                // identity — every other consume of the component resolves
                // to it. Marking the component's own entry therefore left
                // the carrier `Live`, and capture-then-reuse of a component
                // was silently accepted while the same reuse without the
                // closure errored.
                //
                // The forwarding is deliberately NOT general. For an
                // ordinary `y = x` alias both names are user-visible
                // bindings, and a capture has consumed the name it captured
                // since before this issue: `x = ...; y = x; f = fn () ->
                // eat(y); g = fn () -> eat(x)` compiles, and downstream
                // code relies on that spelling to hand two closures their
                // own name for one value (normatively pinned as
                // spec/04-type-system.md [04-LIN-2]). Forwarding there
                // would be an unrelated ecosystem-breaking tightening —
                // the exact class of change chelis#1200 exists to undo —
                // and it is not what the component misroute needs. A
                // direct (unaliased) capture-then-reuse of an ordinary
                // binding still errors, unchanged, through
                // `read_or_error`.
                // Identity, not the region-relative F2 gate: a branch body
                // clears `destructured`, so reading it here lost the carrier
                // for every capture inside an `if`/`match` arm (chelis#1200
                // review finding 1).
                let use_id = outer_scope.top_id(&name);
                let capture_target = match use_id.and_then(|id| outer_scope.resolve_alias_chain(id))
                {
                    Some(carrier) if outer_scope.is_component_id(carrier) => Some(carrier),
                    _ => use_id,
                };
                if let Some(target) = capture_target {
                    outer_scope.consume_id(
                        target,
                        ConsumeSite {
                            description: format!("closure capture {}", diag_site(expr)),
                            kind: ConsumeKind::Structural,
                        },
                    );
                }
                inner_scope.declare(name.clone(), outer_scope.ty(&name).cloned());
            } else {
                // Borrow capture: error if the outer is already
                // consumed (read_or_error covers that), then record a
                // borrow site on the outer binding. The inner scope
                // sees `name` as a fresh borrow-read binding too, so
                // the body's borrow-reads inside the closure resolve
                // against the captured borrow rather than re-entering
                // the outer binding state.
                self.read_or_error(name.as_str(), expr, outer_scope);
                outer_scope.borrow(name.as_str(), borrow_site(expr));
                inner_scope.declare(name.clone(), outer_scope.ty(&name).cloned());
            }
        }

        let mut pushed: Vec<(String, BindingId)> = Vec::new();
        if let Some(params) = tagged_children(&kids[0], DeepTag::Params) {
            for param in params {
                if let Some((name, ty)) = param_name_and_type(param) {
                    let id = inner_scope.declare(name, ty.cloned());
                    pushed.push((name.to_string(), id));
                }
            }
        }
        for param in params {
            if !pushed.iter().any(|(name, _)| name == &param) {
                let id = inner_scope.declare(param.clone(), None);
                pushed.push((param, id));
            }
        }
        if matches!(get_tag_expr(&kids[1]), Some(DeepTag::Borrow)) {
            self.invalid_borrow(&kids[1], "borrow cannot be returned from a function");
        } else {
            self.check_expr(&kids[1], &mut inner_scope);
        }
        for (_, id) in pushed.into_iter().rev() {
            self.pop_and_check_param(&mut inner_scope, id, expr_scope_end(&kids[1]));
        }
    }

    fn check_if(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if kids.len() < 3 {
            return;
        }
        self.check_expr(&kids[0], scope);
        let visible_ids = scope.all_visible_ids();
        let mut then_scope = scope.clone();
        let mut else_scope = scope.clone();
        // chelis#1200 Q1: a branch body is a new declaration region, so
        // the destructured-component mark does not cross into it. See
        // `LinearScope::clear_destructured_marks`.
        then_scope.clear_destructured_marks();
        else_scope.clear_destructured_marks();
        self.check_expr(&kids[1], &mut then_scope);
        self.check_expr(&kids[2], &mut else_scope);
        self.join_branch_states(scope, &visible_ids, &[then_scope, else_scope]);
    }

    fn check_match(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if kids.is_empty() {
            return;
        }
        if is_var_expr(&kids[0]) && self.expr_is_owned_linear(&kids[0], scope) {
            self.consume_var_expr(
                &kids[0],
                scope,
                ConsumeSite {
                    description: format!("match scrutinee {}", diag_site(&kids[0])),
                    kind: ConsumeKind::Structural,
                },
            );
        } else {
            self.check_expr(&kids[0], scope);
        }

        let visible_ids = scope.all_visible_ids();
        let mut arm_scopes = Vec::new();
        for arm in kids.iter().skip(1) {
            let Some(arm_kids) = tagged_children(arm, DeepTag::Arm) else {
                continue;
            };
            if arm_kids.len() < 3 {
                continue;
            }
            let mut arm_scope = scope.clone();
            // chelis#1200 Q1: an arm body is a new declaration region, so
            // the destructured-component mark does not cross into it. The
            // arm's own pattern binders were already covered by `declare`
            // below; this covers the outer names the arm merely mentions,
            // which is what made the arm reject where the equivalent
            // closure body compiled. See
            // `LinearScope::clear_destructured_marks`.
            arm_scope.clear_destructured_marks();
            let pattern_bindings = pattern_named_types(&arm_kids[0]);
            let mut pattern_ids = Vec::new();
            for (name, ty) in &pattern_bindings {
                pattern_ids.push(arm_scope.declare(name.clone(), ty.clone()));
            }
            self.check_expr(&arm_kids[1], &mut arm_scope);
            self.check_expr(&arm_kids[2], &mut arm_scope);
            for id in pattern_ids.into_iter().rev() {
                self.pop_and_check(&mut arm_scope, id, expr_scope_end(&arm_kids[2]));
            }
            arm_scopes.push(arm_scope);
        }
        self.join_branch_states(scope, &visible_ids, &arm_scopes);
    }

    fn join_branch_states(
        &mut self,
        scope: &mut LinearScope,
        visible_ids: &[BindingId],
        branches: &[LinearScope],
    ) {
        // The snapshot carries every generation visible at branch entry,
        // shadowed ones included: branch clones share those ids with the
        // parent, so a branch-body consume that resolved through an alias
        // chain to a shadowed generation still merges back onto the same
        // record here (chelis#1209).
        for id in visible_ids {
            // A branch's `Structural` consume is the one that destroys the
            // value, so it is the one that must survive the join. Prefer it
            // over an `Aliasing` record from another branch: an alias bind
            // does not destroy anything, and letting it win would report the
            // wrong site.
            let consumed_site = branches
                .iter()
                .find_map(|branch| match branch.state(*id) {
                    Some(BindingState::Consumed(site))
                        if matches!(site.kind, ConsumeKind::Structural) =>
                    {
                        Some(site.clone())
                    }
                    _ => None,
                })
                .or_else(|| {
                    branches.iter().find_map(|branch| match branch.state(*id) {
                        Some(BindingState::Consumed(site)) => Some(site.clone()),
                        _ => None,
                    })
                });
            let Some(site) = consumed_site else {
                continue;
            };
            // chelis#1200 Q2: propagate onto an outer entry that is already
            // `Consumed(Aliasing)`, not only onto a `Live` one.
            //
            // A destructured component desugars to `p = (var __chelis_tmpN)`,
            // which records an `Aliasing` consume on the temp. The temp is the
            // carrier every later consume of `p` resolves to, so it is never
            // `Live` by the time a branch runs. Gating the join on `Live`
            // therefore dropped the branch's `Structural` consume outright, and
            // "component consumed in one arm, then again after the join" was
            // silently accepted — falsifying the true positive F2 exists to
            // protect. An `Aliasing` record is not a destruction, so a
            // `Structural` consume from a branch legitimately replaces it,
            // exactly as `consume_var_expr`'s Aliasing-then-Structural arm does
            // on the straight-line path.
            // The rule (chelis#1200 review finding 2):
            //
            //   Live                      -> a branch consume always wins.
            //   Consumed(Aliasing), and
            //     the name is a component
            //     carrier                 -> a branch Structural consume wins.
            //   Consumed(Aliasing), and
            //     the name is an ordinary
            //     binding                 -> unchanged.
            //
            // The carve-out is deliberately narrow. The justification above
            // is entirely about carriers: a component's `Aliasing` record is
            // bookkeeping for `p = (var __chelis_tmpN)`, never a destruction,
            // so a branch's real consume must replace it. An ordinary `y = x`
            // alias records the same `Aliasing` shape for a completely
            // different reason, and upgrading it there rejects
            // `y = x; if c then { f = fn () -> realize(x) f() } else t;
            // add(y, y)` — which released 0.18.4 accepts. Tightening ordinary
            // aliases is exactly the class of ecosystem-breaking change
            // chelis#1200 exists to undo, so the upgrade asks for component
            // identity (permanent) rather than the region-relative F2 mark.
            let replaces_outer = match scope.state(*id) {
                Some(BindingState::Live { .. }) => true,
                Some(BindingState::Consumed(outer)) => {
                    matches!(outer.kind, ConsumeKind::Aliasing)
                        && matches!(site.kind, ConsumeKind::Structural)
                        && scope.is_component_id(*id)
                }
                None => false,
            };
            if replaces_outer {
                scope.consume_id(*id, site);
            }
        }
    }

    fn maybe_mark_reusable_app_input(&mut self, expr: &Expr, kids: &[Expr], scope: &LinearScope) {
        if !self.expr_is_owned_linear(expr, scope) {
            return;
        }
        let Some(output_ty) = self.expr_type(expr, scope) else {
            return;
        };
        for (index, arg) in kids.iter().skip(1).enumerate() {
            if borrow_inner(arg).is_some()
                || !is_var_expr(arg)
                || !self.expr_is_owned_linear(arg, scope)
            {
                continue;
            }
            if self
                .expr_type(arg, scope)
                .is_some_and(|arg_ty| type_expr_eq(arg_ty, output_ty))
            {
                self.info.mark_reusable_input(expr.span(), index);
                break;
            }
        }
    }

    fn consume_var_expr(&mut self, expr: &Expr, scope: &mut LinearScope, site: ConsumeSite) {
        let Some(name) = var_name(expr) else {
            return;
        };
        if !self.expr_is_owned_linear(expr, scope) {
            return;
        }
        // Resolve the use-site name to its innermost live generation
        // exactly once; everything past this point is id-keyed
        // (chelis#1209).  An unbound name matches the old `None` arm.
        let Some(use_id) = scope.top_id(name) else {
            return;
        };
        // Linearity-AliasedConsume-F1: a `Structural` consume on an
        // aliased binding forwards to the underlying source generation's
        // record, so a later borrow of the source trips `read_or_error`
        // correctly.  `Aliasing` consumes do not forward; they stay
        // pinned to the alias's own record because the alias bind itself
        // is what introduces the aliasing relationship in the IR.
        let target: BindingId = match site.kind {
            ConsumeKind::Structural => scope.resolve_alias_chain(use_id).unwrap_or(use_id),
            ConsumeKind::Aliasing => use_id,
        };
        match scope.state(target) {
            Some(BindingState::Live { .. }) => scope.consume_id(target, site),
            Some(BindingState::Consumed(consumed_at))
                if matches!(consumed_at.kind, ConsumeKind::Structural)
                    && (consumed_at.description.contains("closure capture")
                        || consumed_at.description.contains("match scrutinee")) =>
            {
                let description = consumed_at.description.clone();
                // chelis#1200: report the name the user wrote, not
                // `target`. `target` is the alias chain's terminal, which
                // for a destructured component is the desugarer's
                // `__chelis_tmpN` — a name that appears nowhere in the
                // user's source and that they cannot act on.
                self.push_diagnostic(CheckError::new(
                    CheckErrorKind::UseAfterConsume,
                    with_macro_provenance(
                        expr,
                        format!(
                            "variable `{name}` was already consumed by {description}; later use {} is invalid",
                            diag_site(expr)
                        ),
                    ),
                    vec!["Structural ownership consumes cannot be auto-copied; move the later use before the consume or copy before the structural consume".to_string()],
                ));
            }
            Some(BindingState::Consumed(consumed_at))
                if matches!(consumed_at.kind, ConsumeKind::Aliasing)
                    && matches!(site.kind, ConsumeKind::Structural) =>
            {
                // The target was previously aliased (e.g. `let y = x`
                // recorded an `Aliasing` consume on `x`); a later
                // `Structural` consume forwarded through the alias
                // chain replaces the aliasing record so subsequent
                // borrows of the target trip `read_or_error`.
                scope.consume_id(target, site);
            }
            Some(BindingState::Consumed(consumed_at))
                if matches!(site.kind, ConsumeKind::Structural)
                    && scope.is_destructured_id(target) =>
            {
                // Linearity-F2: a consume-after-consume on a
                // destructured component is an error.  Implicit Copy
                // insertion does not apply because tuple-get produces
                // a fresh owned value rather than an aliased borrow,
                // so reuse of a destructured tensor name must be made
                // explicit via `copy()`.  For every other binding the
                // implicit-linearity pass inserts a Copy for consuming
                // fan-out, matching the spec's "Copy Insertion"
                // semantics; per the existing baseline we do not flag
                // that shape.
                //
                // chelis#1200: the guard is membership on the consumed
                // *target*, not a block-scoped depth.  `target` is the
                // alias chain's terminal generation, so consuming a
                // component through an alias still lands on the
                // component's marked record, while an ordinary binding
                // that merely appears after a destructuring `let` in the
                // same block does not.
                //
                // Only a `Structural` consume can trip this. An
                // `Aliasing` consume — a second `let y = p` bind of the
                // same component — destroys nothing: `lower_let` maps
                // both names onto the component's node, so `(a, b) = p;
                // y = a; z = a` is a fan-out of borrows, which
                // Copy Insertion covers. Gating it here rejected that
                // shape, which 0.18.3 accepted.
                let description = consumed_at.description.clone();
                // Name the binding the user wrote. When that name is
                // itself the component, say so; when it is an ordinary
                // binding that aliases one, say that instead rather than
                // calling the user's own `let` a destructured binding.
                let origin_phrase = if scope.is_destructured_id(use_id) {
                    " (from a destructured binding)"
                } else {
                    " (an alias of a destructured binding)"
                };
                self.push_diagnostic(CheckError::new(
                    CheckErrorKind::UseAfterConsume,
                    with_macro_provenance(
                        expr,
                        format!(
                            "variable `{name}`{origin_phrase} was already consumed by {description}; later use {} is invalid",
                            diag_site(expr)
                        ),
                    ),
                    vec![format!(
                        "Insert `copy({name})` before the first consuming use if you need to reuse it"
                    )],
                ));
            }
            Some(BindingState::Consumed(_)) => {
                // The implicit-linearity pass will insert a Copy for
                // consuming fan-out.  Borrow-after-consume remains an
                // error through `read_or_error`.
            }
            None => {}
        }
    }

    fn read_var_expr(&mut self, expr: &Expr, scope: &mut LinearScope) {
        let Some(name) = var_name(expr) else {
            return;
        };
        if !self.expr_is_owned_linear(expr, scope) {
            return;
        }
        self.read_or_error(name, expr, scope);
        scope.borrow(name, borrow_site(expr));
    }

    fn read_or_error(&mut self, name: &str, expr: &Expr, scope: &LinearScope) {
        // Linearity-AliasedConsume-F1: when `name` is an alias, the
        // structural consume on it would have forwarded to the
        // underlying source (see `consume_var_expr`).  Check the
        // alias chain's terminal generation first so borrows of either
        // the alias or the source surface the violation symmetrically.
        // An unbound name matches the old missing-entry no-op.
        let Some(use_id) = scope.top_id(name) else {
            return;
        };
        let resolved = scope.resolve_alias_chain(use_id).unwrap_or(use_id);
        let Some(BindingState::Consumed(site)) = scope.state(resolved) else {
            return;
        };
        // Var-RHS let-bindings (`alias = x`) are `ConsumeKind::Aliasing`
        // consumes: at the IR level `lower_let` maps `alias` to the
        // same NodeId as `x` (the `Load { name: "x" }` node), so the
        // value is structurally shared, not destroyed.  Per
        // `spec/design/implicit_linearity.md` §"Copy Insertion" and
        // "Borrows do not count as fan-out", later borrow-reads
        // (`mul`, `add`, `matmul`, ...) of `x` must succeed: the DAG
        // keeps `x` and `alias` pointing to the same source and the
        // Copy-insertion pass at `crates/chelis-ir/src/lower.rs:364`
        // only forks values reached by multiple `Realize | Drop |
        // Store` consumers.  Real consumes (realize / drop / store,
        // app-arg, pipe-stage, closure capture, match scrutinee)
        // remain hard errors here: once a value is truly gone,
        // borrow-reads of it would alias freed storage at runtime.
        //
        // Linearity-F1 (`docs/gap_synthesis.md`) replaced the prior
        // string-prefix check on the description with this typed
        // discrimination via `ConsumeKind`.  The description text
        // stays for diagnostic rendering only.
        if matches!(site.kind, ConsumeKind::Aliasing) {
            return;
        }
        let description = site.description.clone();
        let message = with_macro_provenance(
            expr,
            format!(
                "variable `{name}` was already consumed by {}; later use {} is invalid",
                description,
                diag_site(expr)
            ),
        );
        let suggestion =
            format!("Insert `copy({name})` before the first consuming use if you need to reuse it");
        self.push_diagnostic(CheckError::new(
            CheckErrorKind::UseAfterConsume,
            message,
            vec![suggestion],
        ));
    }

    fn invalid_borrow(&mut self, expr: &Expr, message: &str) {
        let formatted = with_macro_provenance(expr, format!("{message} ({})", diag_site(expr)));
        self.push_diagnostic(CheckError::new(
            CheckErrorKind::InvalidBorrow,
            formatted,
            vec!["Use `&x` only as a direct function-call argument".to_string()],
        ));
    }

    fn pop_and_check(&mut self, scope: &mut LinearScope, id: BindingId, end_offset: usize) {
        self.pop_and_check_inner(scope, id, end_offset, false);
    }

    fn pop_and_check_param(&mut self, scope: &mut LinearScope, id: BindingId, end_offset: usize) {
        self.pop_and_check_inner(scope, id, end_offset, true);
    }

    fn pop_and_check_inner(
        &mut self,
        scope: &mut LinearScope,
        id: BindingId,
        end_offset: usize,
        allow_param_boundary_drop: bool,
    ) {
        let Some((ty, state)) = scope.pop(id) else {
            return;
        };
        if !ty
            .as_ref()
            .is_some_and(|ty| type_expr_is_owned_linear(ty, &self.tensor_carrying_adts))
        {
            return;
        }
        let BindingState::Live { borrow_sites } = state else {
            return;
        };
        if allow_param_boundary_drop {
            return;
        }
        let _ = (borrow_sites, end_offset);
        // Unconsumed locals are handled by inserted Drop nodes in lowered IR.
    }

    fn expr_type<'a>(&'a self, expr: &'a Expr, scope: &'a LinearScope) -> Option<&'a Expr> {
        type_metadata(expr).or_else(|| {
            var_name(expr)
                .and_then(|name| scope.ty(name).or_else(|| self.top_level_types.get(name)))
                .or_else(|| {
                    // Linearity-F2: `(tuple-get (var t) i)` does not
                    // carry `:type` metadata because the `tuple-get`
                    // tag is in `should_attach_type_metadata`'s deny
                    // list.  Recover the element type by inspecting
                    // the underlying tuple var's `:type` and indexing
                    // into its `t-tuple` children.  Used by the
                    // destructure-let desugar path where the
                    // synthesized tmp bind's value is the unannotated
                    // tuple-get.
                    tuple_get_element_type(expr, scope, self)
                })
        })
    }

    fn arg_is_borrowed(
        &self,
        func: Option<&Expr>,
        builtin: Option<&str>,
        arg_index: usize,
        scope: &LinearScope,
    ) -> bool {
        if builtin_arg_is_borrowed(builtin, arg_index) {
            return true;
        }
        if func.is_some_and(callee_is_observational_higher_order) {
            // Implicit-copy fan-out v3 Shape B: `grad(f, ...)(args)` and
            // `vmap(f, ...)(args)` call sites are observational at the
            // linearity level.  Lowering at
            // `chelis-ir::lower::lower_grad_callable_with_nodes` synthesizes
            // a fresh closure body around Load nodes for every arg and
            // splices the caller's NodeId in without destroying it; the
            // forward and backward (or batched) DAGs both read from the
            // same Load.  Classifying every arg position as a borrow lets
            // the linearity checker permit fan-out across multiple grad or
            // vmap calls of the same arg followed by later borrow-reads,
            // matching the semantics already implemented in the IR.
            return true;
        }
        // chelis#229 / chelis#237 direct-call gap: when the callee is
        // a `(var name)` and `name` is a user-defined function whose
        // parameter at `arg_index` was inferred read-only by
        // `infer_signature_metadata`, treat the arg as a borrow even
        // though the annotated `type_env` still has the owned tvar
        // signature. The display-time `&T` rewrite the inferencer
        // produces in `display_signature` is the contract callers
        // see; linearity must honor it or every auto-borrow-inferred
        // user fn spuriously consumes at its call sites.
        if let Some(callee) = func.and_then(var_name)
            && let Some(meta) = self.signature_inference.functions.get(callee)
            && let Some(param) = meta.params.get(arg_index)
            && matches!(param.display_type, Type::Ref(_))
        {
            return true;
        }
        let Some(func_ty) = func.and_then(|expr| self.expr_type(expr, scope)) else {
            return false;
        };
        type_expr_fn_arg(func_ty, arg_index).is_some_and(type_expr_is_ref)
    }

    fn expr_is_owned_linear(&self, expr: &Expr, scope: &LinearScope) -> bool {
        self.expr_type(expr, scope)
            .is_some_and(|ty| type_expr_is_owned_linear(ty, &self.tensor_carrying_adts))
    }

    fn expr_is_owned_or_borrow_linear(&self, expr: &Expr, scope: &LinearScope) -> bool {
        self.expr_type(expr, scope).is_some_and(|ty| {
            type_expr_contains_tensor(ty, &self.tensor_carrying_adts)
                || type_expr_is_unresolved_tvar(ty)
        })
    }
}

fn get_tag(list: &List) -> Option<DeepTag> {
    list.tag()
}

fn with_macro_provenance(expr: &Expr, message: String) -> String {
    let Some(source) = macro_source(expr) else {
        return message;
    };
    format!("{message} (in expansion of {source})")
}

fn macro_source(expr: &Expr) -> Option<String> {
    let (_, meta, _) = stamped_parts(expr)?;
    let source = meta
        .entries
        .iter()
        .find(|(key, _)| key == "source")
        .map(|(_, value)| value)?;
    let rendered = chelis_deep::printer::print_canonical(std::slice::from_ref(source));
    Some(rendered.replace('\n', " ").trim().to_string())
}

fn get_tag_expr(expr: &Expr) -> Option<DeepTag> {
    stamped_parts(expr).map(|(tag, _, _)| tag)
}

fn stamped_parts(expr: &Expr) -> Option<(DeepTag, &MetaMap, &[Expr])> {
    match expr {
        Expr::List(list, _) => {
            let tag = get_tag(list)?;
            let meta = get_meta(list)?;
            Some((tag, meta, children(list)))
        }
        Expr::Node(node, _) => Some((node.tag(), node.meta(), node.children_slice())),
        _ => None,
    }
}

fn tagged_children(expr: &Expr, expected: DeepTag) -> Option<&[Expr]> {
    stamped_parts(expr).and_then(|(tag, _, children)| (tag == expected).then_some(children))
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn is_var_expr(expr: &Expr) -> bool {
    get_tag_expr(expr) == Some(DeepTag::Var)
}

fn var_name(expr: &Expr) -> Option<&str> {
    tagged_children(expr, DeepTag::Var)?
        .first()
        .and_then(symbol_name)
}

fn borrow_inner(expr: &Expr) -> Option<&Expr> {
    tagged_children(expr, DeepTag::Borrow)?.first()
}

/// True when `expr` is a `(grad ...)`, `(vmap ...)`, or nested
/// composition thereof.  Used by `arg_is_borrowed` to mark grad-app and
/// vmap-app call sites as observational at the linearity level.
fn callee_is_observational_higher_order(expr: &Expr) -> bool {
    matches!(get_tag_expr(expr), Some(DeepTag::Grad | DeepTag::Vmap))
}

fn param_names(expr: &Expr) -> Vec<String> {
    let Some(params) = tagged_children(expr, DeepTag::Params) else {
        return Vec::new();
    };
    params
        .iter()
        .filter_map(|param| match param {
            Expr::Atom(Atom::Name(name), _) => Some(name.clone()),
            Expr::List(param_list, _) => param_list
                .elements
                .first()
                .and_then(symbol_name)
                .map(str::to_string),
            Expr::BareList(elements, _) => {
                elements.first().and_then(symbol_name).map(str::to_string)
            }
            // chelis#343: a typed param whose name collides with a Deep tag
            // is desugared to the caret-metadata wrapper `^{:type T} name`
            // (a `MetaExpr`); recover the name from its inner symbol.
            Expr::MetaExpr(meta, _) => symbol_name(meta.expr.as_ref()).map(str::to_string),
            _ => None,
        })
        .collect()
}

fn pattern_names(expr: &Expr) -> Vec<String> {
    let mut names = Vec::new();
    collect_pattern_names(expr, &mut names);
    names
}

fn collect_pattern_names(expr: &Expr, names: &mut Vec<String>) {
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                names.push(name.to_string());
            }
        }
        DeepTag::PatAs => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                names.push(name.to_string());
            }
            if let Some(inner) = kids.get(1) {
                collect_pattern_names(inner, names);
            }
        }
        _ => {
            for child in kids {
                collect_pattern_names(child, names);
            }
        }
    }
}

/// Like [`pattern_names`], but also returns each binding's resolved
/// type expression when the inferencer stamped one onto the pattern
/// node's metadata. Used by `check_match` to populate arm `LinearScope`
/// entries with their concrete types — required for destructured
/// fields whose type comes from the scrutinee's ADT instantiation
/// rather than a `let`-style RHS. (closes #181)
fn pattern_named_types(expr: &Expr) -> Vec<(String, Option<Expr>)> {
    let mut bindings = Vec::new();
    collect_pattern_named_types(expr, &mut bindings);
    bindings
}

fn collect_pattern_named_types(expr: &Expr, bindings: &mut Vec<(String, Option<Expr>)>) {
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                bindings.push((name.to_string(), type_metadata(expr).cloned()));
            }
        }
        DeepTag::PatAs => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                bindings.push((name.to_string(), type_metadata(expr).cloned()));
            }
            if let Some(inner) = kids.get(1) {
                collect_pattern_named_types(inner, bindings);
            }
        }
        _ => {
            for child in kids {
                collect_pattern_named_types(child, bindings);
            }
        }
    }
}

/// Free variables of `expr`, in a DETERMINISTIC (sorted) order.
///
/// chelis#1200: `check_fn` walks this list mutating `outer_scope` as it
/// goes — it consumes or borrows each capture and can raise
/// `UseAfterConsume`. When two captures are on one alias chain
/// (`y = x; fn () -> add(realize(x), realize(y))`), the verdict depends
/// on which is visited first, so a `UnordSet`'s iteration order made the
/// same program compile or fail run to run (measured: 11/12 reject,
/// 1/12 accept). Sorting is the cheap half of the fix; forwarding the
/// capture consume through the alias chain in `check_fn` is the half
/// that makes both orders agree.
///
/// chelis#1209's generation ids did NOT remove this order-sensitivity:
/// `check_fn` deliberately does not forward an ordinary alias's capture
/// to its source ([04-LIN-2] — two user-visible bindings of one value
/// are distinct for capture), so when two captures sit on one alias
/// chain the verdict still depends on which capture consumes first.
/// The sort stays semantically necessary, not merely cosmetic.
fn free_vars(expr: &Expr, params: &[String]) -> Vec<String> {
    let mut bound = vec![params.iter().cloned().collect::<UnordSet<_>>()];
    let mut free = UnordSet::new();
    collect_free_vars(expr, &mut bound, &mut free);
    free.into_sorted()
}

fn collect_free_vars(expr: &Expr, bound: &mut Vec<UnordSet<String>>, free: &mut UnordSet<String>) {
    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => {}
        Expr::MetaExpr(meta, _) => collect_free_vars(&meta.expr, bound, free),
        Expr::List(list, _) => match get_tag(list) {
            Some(DeepTag::Var) => {
                if let Some(name) = children(list).first().and_then(symbol_name)
                    && !bound.iter().rev().any(|scope| scope.contains(name))
                {
                    free.insert(name.to_string());
                }
            }
            Some(DeepTag::Fn) => {
                let kids = children(list);
                if kids.len() >= 2 {
                    bound.push(param_names(&kids[0]).into_iter().collect());
                    collect_free_vars(&kids[1], bound, free);
                    bound.pop();
                }
            }
            Some(DeepTag::Let) => {
                let kids = children(list);
                if kids.len() < 2 {
                    return;
                }
                let mut let_scope = UnordSet::new();
                if let Some(bind_kids) = tagged_children(&kids[0], DeepTag::Bind) {
                    let mut index = 0;
                    while index + 1 < bind_kids.len() {
                        collect_free_vars(&bind_kids[index + 1], bound, free);
                        if let Some(name) = symbol_name(&bind_kids[index]) {
                            let_scope.insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                bound.push(let_scope);
                collect_free_vars(&kids[1], bound, free);
                bound.pop();
            }
            Some(DeepTag::Match) => {
                let kids = children(list);
                if kids.is_empty() {
                    return;
                }
                collect_free_vars(&kids[0], bound, free);
                for arm in kids.iter().skip(1) {
                    let Some(arm_kids) = tagged_children(arm, DeepTag::Arm) else {
                        continue;
                    };
                    if arm_kids.len() < 3 {
                        continue;
                    }
                    bound.push(pattern_names(&arm_kids[0]).into_iter().collect());
                    collect_free_vars(&arm_kids[1], bound, free);
                    collect_free_vars(&arm_kids[2], bound, free);
                    bound.pop();
                }
            }
            _ => {
                for child in children(list) {
                    collect_free_vars(child, bound, free);
                }
            }
        },
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        Expr::Node(node, span) => {
            let bridged = Expr::List(node.to_list(*span), *span);
            collect_free_vars(&bridged, bound, free);
        }
        Expr::BareList(elems, _) => {
            for elem in elems {
                collect_free_vars(elem, bound, free);
            }
        }
        Expr::UnknownForm(data) => {
            for child in &data.children {
                collect_free_vars(child, bound, free);
            }
        }
    }
}

fn builtin_arg_is_borrowed(name: Option<&str>, arg_index: usize) -> bool {
    // Tensor→host conversions read the tensor without taking ownership — the
    // runtime implementations (`chelis_list_from_tensor`, `chelis_tensor_to_f64`,
    // `chelis_tensor_rank`, `chelis_tensor_shape`,
    // `chelis_tensor_numel`, `tensor_to_string`) all read via the pointer and
    // never call `chelis_tensor_release`, so the caller still owns the input
    // afterwards.
    // Keeping these observational avoids forcing callers to sprinkle
    // `copy(x)` before every query or host-lane conversion.
    //
    // chelis#527: the same justification covers the read-only `List`/`Dict`
    // queries `len` (arg 0) and `index` (arg 0).  `chelis_list_len` /
    // `chelis_list_index` both take a `const chelis_list *` and never free it
    // (`index` *retains* the element it returns), so the caller still owns the
    // container afterwards.  Before this, a `List[tensor]` parameter named
    // `params` (or any Deep-tag-colliding identifier — see chelis#343) was
    // consume-tracked, so the idiomatic "read a list's length/element, then
    // reuse the list" optimizer shape no longer type-checked and there was no
    // non-consuming form to express it.  Classifying these as borrows restores
    // read-then-reuse without an extra `copy()`.
    let Some(name) = name else {
        return false;
    };
    matches!(
        name,
        "add"
            | "mul"
            | "max_elem"
            | "sub"
            | "div"
            | "floor_div"
            | "trunc_div"
            | "eq"
            | "neq"
            | "lt"
            | "gt"
            | "lte"
            | "gte"
            | "and"
            | "or"
            | "matmul"
            | "layer_norm"
            | "where"
            | "clamp"
            | "test_assert_close_tensor"
            | "test_assert_eq_tensor"
    ) || matches!(
        (name, arg_index),
        (
            "neg"
                | "recip"
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
                | "round"
                | "uniform_like"
                | "cmplt"
                | "not"
                | "relu"
                | "sigmoid"
                | "softmax"
                | "normalize"
                | "mean"
                | "min_elem"
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
                | "dropout"
                | "print"
                | "debug"
                | "to_string"
                | "rank"
                | "shape"
                | "numel"
                | "to_list"
                | "tensor_to_scalar"
                | "len"
                | "index",
            0
        ) | ("conv2d", 0 | 1)
            | ("einsum", 1 | 2)
            | ("split", 0)
            | ("gather", 0 | 1)
            | ("cumsum", 0)
            | ("sort", 0)
            | ("diagonal", 0)
            | ("trace", 0)
    )
}

fn type_metadata(expr: &Expr) -> Option<&Expr> {
    let (_, meta, _) = stamped_parts(expr)?;
    meta.entries
        .iter()
        .find(|(key, _)| key == "type")
        .map(|(_, value)| value)
}

/// Render the source site of `expr` for linearity diagnostics.
///
/// Desugared Surf nodes carry their source location as `span:
/// "surf:a..b"` METADATA (`chelis_surf::desugar::attach_span_metadata`);
/// their STRUCTURAL span is the desugarer's zero placeholder. Reading
/// the structural span made every linearity diagnostic on Surf-derived
/// code print the constant `offset 0` — unlocalizable in a multi-file
/// unit, which is how chelis#329's package-wide
/// `InvalidBorrow (offset 0)` report came to be misattributed to a
/// grad-helper collision. Prefer the metadata (rendered `at
/// surf:a..b`, matching `infer.rs::validator_span_suffix`); fall back
/// to the structural offset only for fully synthesized nodes that
/// carry no span entry (e.g. desugared pipe-stage lambdas).
fn diag_site(expr: &Expr) -> String {
    match span_metadata_id(expr) {
        Some(id) => format!("at {id}"),
        None => format!("at offset {}", expr.span().offset),
    }
}

/// Extract the `span: "surf:a..b"` metadata entry, when present.
fn span_metadata_id(expr: &Expr) -> Option<&str> {
    let (_, meta, _) = stamped_parts(expr)?;
    meta.entries.iter().find_map(|(key, value)| match value {
        Expr::Atom(Atom::Str(id), _) if key == "span" => Some(id.as_str()),
        _ => None,
    })
}

fn param_name_and_type(param: &Expr) -> Option<(&str, Option<&Expr>)> {
    match param {
        Expr::Atom(Atom::Name(name), _) => Some((name.as_str(), None)),
        Expr::List(param_list, _) => Some((
            param_list.elements.first().and_then(symbol_name)?,
            get_meta(param_list)
                .and_then(|meta| meta.entries.iter().find(|(k, _)| k == "type"))
                .map(|(_, value)| value),
        )),
        Expr::BareList(elements, _) => {
            let name = elements.first().and_then(symbol_name)?;
            let ty = elements.get(1).and_then(|meta| {
                let Expr::Map(meta, _) = meta else {
                    return None;
                };
                meta.entries
                    .iter()
                    .find(|(key, _)| key == "type")
                    .map(|(_, value)| value)
            });
            Some((name, ty))
        }
        // chelis#343: a typed param whose name collides with a Deep tag
        // (`params`, `fn`, `let`, ...) cannot use the `(name {type: T})`
        // list form — `(params {type: T})` is indistinguishable from a
        // `params` tag list — so the desugarer emits the caret-metadata
        // wrapper `^{:type T} name` (a `MetaExpr`) instead
        // (`typed_param_needs_meta_wrapper`). `extract_params` in infer.rs
        // already reads this form; linearity must too, or the param is
        // declared with NO type and a borrow of it (`&params`) trips a
        // spurious "borrowed arguments must be tensor or tensor-carrying
        // values". Mirrors infer.rs `extract_params`'s MetaExpr arm.
        Expr::MetaExpr(meta, _) => {
            let Expr::Atom(Atom::Name(name), _) = meta.expr.as_ref() else {
                return None;
            };
            Some((
                name.as_str(),
                meta.entries
                    .iter()
                    .find(|(k, _)| k == "type")
                    .map(|(_, value)| value),
            ))
        }
        _ => None,
    }
}

fn get_meta(list: &List) -> Option<&MetaMap> {
    match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => Some(meta),
        _ => None,
    }
}

/// Linearity-F2 helper: resolve the element type of a
/// `(tuple-get (var t) i)` expression by looking up the tuple var
/// `t`'s scope-type and indexing into its `t-tuple` children.
/// Returns `None` if any part of the lookup fails.  Used by
/// `Checker::expr_type` when the synthesized destructure-tmp bind
/// value is an unannotated tuple-get.
fn tuple_get_element_type<'a>(
    expr: &'a Expr,
    scope: &'a LinearScope,
    checker: &'a Checker,
) -> Option<&'a Expr> {
    let kids = tagged_children(expr, DeepTag::TupleGet)?;
    let tuple_expr = kids.first()?;
    let tuple_ty = type_metadata(tuple_expr).or_else(|| {
        var_name(tuple_expr)
            .and_then(|name| scope.ty(name).or_else(|| checker.top_level_types.get(name)))
    })?;
    let index_expr = kids.get(1)?;
    let index = match tagged_children(index_expr, DeepTag::Lit) {
        Some(literal_children) => literal_children.first().and_then(|child| match child {
            Expr::Atom(Atom::Int(n), _) => Some(*n as usize),
            _ => None,
        }),
        None => match index_expr {
            Expr::Atom(Atom::Int(n), _) => Some(*n as usize),
            _ => None,
        },
    }?;
    let tys = tagged_children(tuple_ty, DeepTag::TTuple)?;
    tys.get(index)
}

/// Linearity-F2 destructured-component marker.  Returns `true` if
/// `bind_list` is a `(bind {meta} name value ...)` whose meta-map
/// contains the `destructure: true` marker injected by
/// `chelis_surf::desugar` when synthesizing the `__chelis_tmpN`
/// intermediates and component binds for a `let` whose pattern is not
/// a bare `Var`.  Used by `Checker::check_let` to call
/// `LinearScope::mark_destructured` on the generations the bind
/// introduces; the `consume_var_expr` already-consumed arm reads that
/// per-binding mark so use-after-consume on a destructured component
/// surfaces as an error rather than the silent fallthrough used by
/// regular bindings (where implicit Copy insertion covers consuming
/// fan-out).
///
/// chelis#1200: this marker previously drove a block-scoped depth
/// counter, which made the F2 error fire for every variable in the
/// remainder of an enclosing block.  The marker itself was never the
/// defect and is unchanged; only its consumer moved to per-binding
/// marks.
fn bind_introduces_destructure_tmp(bind_expr: &Expr) -> bool {
    let Some((DeepTag::Bind, meta, _)) = stamped_parts(bind_expr) else {
        return false;
    };
    meta.entries.iter().any(|(key, value)| {
        key == "destructure" && matches!(value, Expr::Atom(Atom::Bool(true), _))
    })
}

fn type_expr_contains_tensor(expr: &Expr, tensor_carrying_adts: &UnordSet<String>) -> bool {
    let Some((tag, _, children)) = stamped_parts(expr) else {
        return false;
    };
    match tag {
        DeepTag::TTensor => true,
        DeepTag::TRef => children
            .iter()
            .any(|c| type_expr_contains_tensor(c, tensor_carrying_adts)),
        DeepTag::TTuple => children
            .iter()
            .any(|c| type_expr_contains_tensor(c, tensor_carrying_adts)),
        DeepTag::TAdt => {
            // An ADT is tensor-carrying if EITHER one of its type
            // arguments is (the original behavior — e.g. `Wrapper[a]`
            // where `a` is `tensor[..]`), OR the ADT's own definition
            // has a variant with a tensor-carrying field (the
            // chelis#153 fix for `&BatchNormParams { weight: tensor[..],
            // ... }`). The pre-computed set in `tensor_carrying_adts`
            // already accounts for transitive ADT-field tensor-carry.
            let name_carries = children
                .first()
                .and_then(symbol_name)
                .is_some_and(|n| tensor_carrying_adts.contains(n));
            name_carries
                || children
                    .iter()
                    .skip(1) // skip the name; only check type args
                    .any(|c| type_expr_contains_tensor(c, tensor_carrying_adts))
        }
        DeepTag::TFn => false,
        _ => false,
    }
}

fn type_expr_is_ref(expr: &Expr) -> bool {
    matches!(get_tag_expr(expr), Some(DeepTag::TRef))
}

/// Issue #256: detect a stamped `(t-var ...)` (or `(t-ref (t-var ...))`)
/// whose underlying type variable was left unresolved by the
/// annotation pass. This shape arises when a let-bound name receives
/// its type from a polymorphic-return call (e.g. `relu(prev_out)`)
/// whose dim variables are pinned only after the borrow site by a
/// later unification (typically the receiving function's `&tensor[..]`
/// parameter). The inference-layer borrow arm at
/// `infer.rs::borrow` accepts a `Type::Var` borrow precisely so that
/// later unification can pin it; the linearity classification must
/// not reject the same shape and re-introduce the bug. If the
/// underlying variable is genuinely free (not a tensor in any
/// instantiation), the inference layer's downstream unification --
/// not linearity -- surfaces the type mismatch.
///
/// Negative parity: a borrow whose inner is genuinely not a tensor
/// or carrier (e.g. `&int32` against a non-borrow consumer) is
/// rejected by the inference-layer `borrow` arm before reaching
/// linearity (the `_ => TypeMismatch` arm fires for `Type::Prim`,
/// `Type::Unit`, `Type::Fn`, etc.), so this leniency cannot leak.
fn type_expr_is_unresolved_tvar(expr: &Expr) -> bool {
    match get_tag_expr(expr) {
        Some(DeepTag::TVar) => true,
        Some(DeepTag::TRef) => stamped_parts(expr)
            .and_then(|(_, _, children)| children.first())
            .is_some_and(type_expr_is_unresolved_tvar),
        _ => false,
    }
}

fn type_expr_is_owned_linear(expr: &Expr, tensor_carrying_adts: &UnordSet<String>) -> bool {
    type_expr_contains_tensor(expr, tensor_carrying_adts) && !type_expr_is_ref(expr)
}

/// Walk top-level declarations and return the set of ADT names whose
/// definitions (transitively) carry a tensor field. Used by the
/// linearity checker to recognize `&MyParams` as a valid borrow when
/// `MyParams` is a record with a `tensor[...]` field — previously the
/// `t-adt` arm of `type_expr_contains_tensor` only inspected the ADT's
/// type *arguments*, missing tensor fields declared in the variant.
///
/// # Caller contract
///
/// The caller MUST pass every `deftype` whose name might appear
/// (transitively) in the field type of any other `deftype` in the
/// same call. The fixed point converges only over the ADTs visible
/// in `exprs`: an unrelated-library ADT whose carrier status would
/// flip a new-code ADT into the result is invisible if the library
/// half is not chained in. `check_linearity_with_context` enforces
/// this by chaining `library_program.annotated_exprs()` with
/// `new_program.annotated_exprs()` in one call; any future caller
/// (e.g. an incremental `AdtRegistry`-backed query) must preserve
/// the same union-and-recompute discipline or the carrier set will
/// silently desync from the cross-package borrow rules. The locking
/// regression test is at `tests/linearity_with_context.rs::
/// check_linearity_with_context_is_pure_across_repeated_calls`.
///
/// # Complexity
///
/// Fixed-point iteration handles ADTs whose fields reference other
/// ADTs (the standard "is this type transitively tensor-carrying?"
/// graph walk). Each pass scans every (adt, field) pair; the loop
/// terminates after at most `O(adt_count)` passes (one per
/// transitive layer), giving a worst-case bound of
/// `O(adt_count^2 * fields_per_adt)`. In practice 2-3 passes suffice
/// on the existing corpus, so the cost is small relative to the
/// per-expression linearity walk. The function is recomputed on
/// every `check_linearity` / `check_linearity_with_context` call —
/// notably including REPL-driven re-evaluations (`chelis surf`,
/// `chelis eval`). When that cost becomes load-bearing, the
/// migration trigger is exposure of a shared `AdtRegistry` query on
/// `CheckedProgram`: replace this helper with a registry lookup of
/// variant-field types, keeping the same union-and-recompute
/// invariant on the lookup side.
///
/// Typealiases (`typealias`) are NOT walked here. The inference layer
/// owns alias resolution — see `resolve_type_aliases` in `infer.rs`
/// (exercised by `typealias_zero_param_resolves_in_defsig` and
/// siblings) — and any case where a `typealias` name reaches the
/// linearity checker still wearing a `(t-adt {} Alias ...)` shape is
/// a bug in inference, not in this carrier set. If/when the linearity
/// checker gains direct access to a shared `AdtRegistry`, this helper
/// retires in favor of querying that registry's variant-field types
/// (which already know about aliases too).
fn compute_tensor_carrying_adts<'a, I>(exprs: I) -> UnordSet<String>
where
    I: IntoIterator<Item = &'a Expr>,
{
    // Step 1: collect every (adt_name, field_type_exprs) pair from
    // `(deftype {} Name (params?) (variant {} VariantName [field_or_tyarg]...)...)`
    // declarations, descending through `(module {} name ...)` wrappers.
    //
    // The caller decides what to include: a single program passes its
    // own `annotated_exprs()`; the with-context entry chains library
    // and new-code so cross-package field references (a new-code ADT
    // wrapping a library tensor-carrying ADT) are resolved by the same
    // fixed-point pass instead of two independent ones.
    let mut adt_field_types: UnordMap<String, Vec<Expr>> = UnordMap::new();
    fn collect(expr: &Expr, out: &mut UnordMap<String, Vec<Expr>>) {
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            return;
        };
        match tag {
            DeepTag::Module => {
                // `(module {} name body...)` — `children()` skips tag
                // and meta, leaving `[name, body...]`; skip the name
                // for the same shape the `deftype` branch below uses.
                for child in kids.iter().skip(1) {
                    collect(child, out);
                }
            }
            DeepTag::Deftype => {
                let Some(name) = kids.first().and_then(symbol_name) else {
                    return;
                };
                let mut field_tys: Vec<Expr> = Vec::new();
                for child in kids.iter().skip(1) {
                    let Some(variant_children) = tagged_children(child, DeepTag::Variant) else {
                        continue;
                    };
                    // variant children: name, then either `(field name ty)`
                    // entries (record-style) or bare type exprs (positional).
                    for value in variant_children.iter().skip(1) {
                        if let Some(field_children) = tagged_children(value, DeepTag::Field) {
                            if let Some(ty) = field_children.get(1) {
                                field_tys.push(ty.clone());
                            }
                        } else {
                            field_tys.push(value.clone());
                        }
                    }
                }
                out.insert(name.to_string(), field_tys);
            }
            _ => {}
        }
    }
    for expr in exprs {
        collect(expr, &mut adt_field_types);
    }

    // Step 2: fixed-point iteration. An ADT is tensor-carrying iff any
    // of its field types contains a tensor (looking up other ADTs in
    // the current set). Reuses `type_expr_contains_tensor` against
    // the in-progress carrier set, so the recursive `t-adt` lookup
    // walks the same code path used at check time. Stop when a pass
    // adds no new names; bounded by the ADT count.
    let mut carriers: UnordSet<String> = UnordSet::new();
    loop {
        let mut grew = false;
        for (name, field_tys) in adt_field_types.to_sorted() {
            if carriers.contains(name) {
                continue;
            }
            if field_tys
                .iter()
                .any(|ty| type_expr_contains_tensor(ty, &carriers))
            {
                carriers.insert(name.clone());
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    carriers
}

fn type_expr_fn_arg(expr: &Expr, index: usize) -> Option<&Expr> {
    let kids = tagged_children(expr, DeepTag::TFn)?;
    if index >= kids.len().saturating_sub(1) {
        return None;
    }
    kids.get(index)
}

fn type_expr_eq(lhs: &Expr, rhs: &Expr) -> bool {
    chelis_deep::printer::print_canonical(std::slice::from_ref(lhs))
        == chelis_deep::printer::print_canonical(std::slice::from_ref(rhs))
}

fn app_site(expr: &Expr, list: &List) -> ConsumeSite {
    let name = children(list)
        .first()
        .and_then(var_name)
        .map(|name| format!("call to `{name}`"))
        .unwrap_or_else(|| "call".to_string());
    ConsumeSite {
        description: format!("{name} {}", diag_site(expr)),
        kind: ConsumeKind::Structural,
    }
}

fn generic_site(expr: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!("use {}", diag_site(expr)),
        kind: ConsumeKind::Structural,
    }
}

fn realize_site(expr: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!("realize {}", diag_site(expr)),
        kind: ConsumeKind::Structural,
    }
}

fn pipe_site(current: &Expr, stage: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!(
            "pipe into stage {} from {}",
            diag_site(stage),
            diag_site(current)
        ),
        kind: ConsumeKind::Structural,
    }
}

fn borrow_site(expr: &Expr) -> String {
    format!("borrow {}", diag_site(expr))
}

fn expr_scope_end(expr: &Expr) -> usize {
    expr.span().offset + expr.span().len
}

#[cfg(test)]
mod tests {
    //! Unit tests for the private linearity helpers. Locks the contract
    //! `pattern_named_types` must uphold for `check_match` to populate
    //! arm scopes with the correct binding types after the issue #181
    //! substitution fix.

    use super::*;
    use chelis_deep::Span;
    use chelis_deep::ast::{Atom, Expr, List, MetaMap};

    fn span() -> Span {
        Span::new(0, 0)
    }

    fn sym(name: &str) -> Expr {
        Expr::Atom(Atom::Name(name.to_string()), span())
    }

    fn meta(entries: Vec<(&str, Expr)>) -> Expr {
        Expr::Map(
            MetaMap {
                entries: entries
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            },
            span(),
        )
    }

    /// Build `(tag {meta} children...)`.
    fn node(tag: &str, meta_entries: Vec<(&str, Expr)>, children: Vec<Expr>) -> Expr {
        let head = match DeepTag::parse(tag) {
            Some(tag) => Expr::Atom(Atom::Tag(tag), span()),
            None => sym(tag),
        };
        let mut elements = vec![head, meta(meta_entries)];
        elements.extend(children);
        Expr::List(List { elements }, span())
    }

    /// Build a synthetic `(t-tensor {} (d-lit 4) (t-prim f32))` so the
    /// tests can assert metadata is the exact `Expr` we stamped.
    fn tensor_4_f32() -> Expr {
        node(
            "t-tensor",
            vec![],
            vec![
                node("d-lit", vec![], vec![Expr::Atom(Atom::Int(4), span())]),
                node("t-prim", vec![], vec![sym("f32")]),
            ],
        )
    }

    #[test]
    fn pat_var_with_type_metadata_returns_some_ty() {
        // (pat-var {type: <ty>} x)
        let pat = node("pat-var", vec![("type", tensor_4_f32())], vec![sym("x")]);
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, "x");
        assert_eq!(bindings[0].1, Some(tensor_4_f32()));
    }

    #[test]
    fn pat_var_without_type_metadata_returns_none() {
        // (pat-var {} y)
        let pat = node("pat-var", vec![], vec![sym("y")]);
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings, vec![("y".to_string(), None)]);
    }

    #[test]
    fn pat_tuple_walks_into_children() {
        // (pat-tuple {} (pat-var {type:<ty>} a) (pat-var {} b))
        let pat = node(
            "pat-tuple",
            vec![],
            vec![
                node("pat-var", vec![("type", tensor_4_f32())], vec![sym("a")]),
                node("pat-var", vec![], vec![sym("b")]),
            ],
        );
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0], ("a".to_string(), Some(tensor_4_f32())));
        assert_eq!(bindings[1], ("b".to_string(), None));
    }

    #[test]
    fn pat_record_walks_into_kv_children() {
        // (pat-record {} FooState (kv {} x (pat-var {type:<ty>} x)))
        let pat = node(
            "pat-record",
            vec![],
            vec![
                sym("FooState"),
                node(
                    "kv",
                    vec![],
                    vec![
                        sym("x"),
                        node("pat-var", vec![("type", tensor_4_f32())], vec![sym("x")]),
                    ],
                ),
            ],
        );
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings, vec![("x".to_string(), Some(tensor_4_f32()))]);
    }

    #[test]
    fn pat_ctor_walks_into_positional_subpatterns() {
        // (pat-ctor {} Some (pat-var {type:<ty>} v))
        let pat = node(
            "pat-ctor",
            vec![],
            vec![
                sym("Some"),
                node("pat-var", vec![("type", tensor_4_f32())], vec![sym("v")]),
            ],
        );
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings, vec![("v".to_string(), Some(tensor_4_f32()))]);
    }

    #[test]
    fn pat_as_returns_outer_and_inner_bindings() {
        // (pat-as {type:<ty>} whole (pat-var {} x))
        let pat = node(
            "pat-as",
            vec![("type", tensor_4_f32())],
            vec![sym("whole"), node("pat-var", vec![], vec![sym("x")])],
        );
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0], ("whole".to_string(), Some(tensor_4_f32())));
        assert_eq!(bindings[1], ("x".to_string(), None));
    }

    #[test]
    fn pat_wild_returns_no_bindings() {
        // (pat-wild {})
        let pat = node("pat-wild", vec![], vec![]);
        let bindings = pattern_named_types(&pat);
        assert!(bindings.is_empty());
    }

    #[test]
    fn malformed_callee_type_reaches_linearity_exactly_once() {
        let captured = node("x", vec![("type", tensor_4_f32())], vec![]);
        let inner_body = node(
            "app",
            vec![],
            vec![
                node("var", vec![], vec![sym("poison")]),
                node("var", vec![], vec![sym("x")]),
            ],
        );
        let inner_fn = node(
            "fn",
            vec![],
            vec![node("params", vec![], vec![]), inner_body],
        );
        let outer_fn = node(
            "fn",
            vec![],
            vec![node("params", vec![], vec![captured]), inner_fn],
        );
        let program = CheckedProgram::unchecked_for_linearity_diagnostic_test(
            vec![node("def", vec![], vec![sym("outer"), outer_fn])],
            BTreeMap::from([("poison".to_string(), node("t-fn", vec![], vec![]))]),
        );

        let errors = check_linearity(&program)
            .expect_err("malformed callee metadata must make linearity fail");
        assert_eq!(
            errors.len(),
            1,
            "the resolver diagnostic must join the authoritative linearity result exactly once: {errors:?}"
        );
        assert!(errors[0].message.contains("malformed `t-fn`"));
    }
}
