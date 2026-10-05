use chelis_deep::role::{AritySpec, ChildStampRole, arity_contract, child_stamp_role};
use chelis_deep::{DeepTag, ExprCarrier};
use chelis_unord::{UnordMap, UnordSet};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr};
use serde::{Deserialize, Serialize};

use crate::CheckedProgram;
use crate::builtins::{BUILTIN_NAMES, BuiltinSiblingCaseId, COMPARISON_OPS, builtin_decl};
use crate::cancel::CancelToken;
use crate::errors::{CheckError, CheckErrorKind};
use crate::infer::SignatureInferenceMetadata;
use crate::key_admission::{KeyAdmission, KeyRefusal, TagKeys, builtin_key_operand, tag_keys};
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

/// Whether a binding shares a value or consumes it. This classification
/// is independent of the site description used in diagnostics.
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

/// What an application does with one operand under the key allow-list
/// (`crate::key_admission`), when that operand carries a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyOperand {
    /// Refused, and already reported.
    Refused,
    /// Read for its extent only ([`KeyAdmission::ExtentObservation`]): the
    /// key stays live.
    ExtentRead,
    /// Consumed by the call, as every other admitted operand is.
    Consumed,
}

#[derive(Debug, Clone)]
struct ConsumeSite {
    description: String,
    kind: ConsumeKind,
    /// `true` when the consume ends the owner's lifetime: an explicit `drop`
    /// ([05-OP-67]). No inserted copy can keep the owner usable after it, so
    /// every later use of the owner, through any name bound to it, is
    /// `UseAfterConsume` rather than consuming fan-out repaired by copy
    /// insertion ([04-LIN-11]; chelis#3177).
    terminal: bool,
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
    /// [04-LIN-9]: the tuple positions whose key-carrying component a
    /// `tuple-get` has already moved out of this binding. A key component is
    /// projected at most once, and a binding with a moved component cannot
    /// be used whole again. Empty for every binding that carries no key.
    moved_key_components: BTreeSet<usize>,
    /// The sites of closures whose captures borrow this owner (spec/04
    /// section 8.3: a capture whose body uses are all borrow-reads borrows
    /// the outer binding). The checker does not bound a closure value's
    /// lifetime, since it can be stored, returned, or passed on, so the
    /// borrow stays current for the rest of the owner's scope, and a `drop`
    /// of the owner is refused ([04-LIN-2]; chelis#3178).
    closure_borrows: Vec<String>,
    /// [04-LIN-11]: the components a `drop` of a projection moved out of
    /// this owner, each with its `drop` site. A use of the owner whole, or of
    /// a projection overlapping a dropped component, is `UseAfterConsume`;
    /// a projection disjoint from every dropped component stays usable.
    dropped_components: Vec<(Vec<ProjectionStep>, String)>,
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
                moved_key_components: BTreeSet::new(),
                closure_borrows: Vec::new(),
                dropped_components: Vec::new(),
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
    /// [04-LIN-9] / spec/04 section 8.4.1: names of ADTs whose definitions
    /// (transitively) carry a random key. Computed by the same fixed point
    /// over the same declarations as `tensor_carrying_adts`.
    key_carrying_adts: UnordSet<String>,
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
    /// The owner whose projection chain `check_projection_use` already
    /// checked against its dropped components ([04-LIN-11]), while the walk
    /// is inside that chain. Its root variable's read or consume there is a
    /// use of the projected component, not of the owner whole.
    projection_root: Option<BindingId>,
}

impl Checker {
    fn push_diagnostic(&mut self, error: CheckError) {
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
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Module, _, children) => {
            for child in children.iter().skip(1) {
                pre_declare_one(child, type_env, scope);
            }
        }
        ExprCarrier::DecodedNode(DeepTag::Def, _, children) => {
            if let Some(name) = children.first().and_then(symbol_name) {
                scope.declare(name, type_env.get(name).cloned());
            }
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => {}
    }
}

pub fn check_linearity(program: &CheckedProgram) -> Result<CheckedProgram, Vec<CheckError>> {
    let tensor_carrying_adts = compute_tensor_carrying_adts(program.annotated_exprs());
    let key_carrying_adts = compute_key_carrying_adts(program.annotated_exprs());
    let mut checker = Checker {
        errors: Vec::new(),
        info: LinearityInfo::default(),
        top_level_types: program.type_env().clone(),
        tensor_carrying_adts,
        key_carrying_adts,
        signature_inference: program.signature_inference().clone(),
        type_headers: program.type_headers().clone(),
        projection_root: None,
    };
    let mut scope = LinearScope::default();

    pre_declare_top_level_defs(program.annotated_exprs(), program.type_env(), &mut scope);

    // chelis#930: cooperative cancellation at top-level-declaration
    // granularity — the same grain as the type checker's own schedule, and
    // linearity is the third-largest front-end phase on a declaration-heavy
    // program. Abandoning the walk proves nothing about the tail, so this is a
    // hard failure rather than a partial `Ok` (covered-or-rejected).
    checker.check_program_items(program.annotated_exprs(), &mut scope);

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
    // Same union-and-recompute discipline as the tensor carrier set above.
    let key_carrying_adts = compute_key_carrying_adts(
        library_program
            .annotated_exprs()
            .iter()
            .chain(new_program.annotated_exprs().iter()),
    );
    let mut merged_type_headers = library_program.type_headers().clone();
    merged_type_headers.extend_from(new_program.type_headers());
    let mut checker = Checker {
        errors: Vec::new(),
        info: LinearityInfo::default(),
        top_level_types: new_program.type_env().clone(),
        tensor_carrying_adts,
        key_carrying_adts,
        signature_inference: merged_signature_inference,
        type_headers: merged_type_headers,
        projection_root: None,
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
    checker.check_program_items(new_program.annotated_exprs(), &mut scope);

    if checker.errors.is_empty() {
        Ok(new_program.clone().with_linearity(checker.info))
    } else {
        Err(checker.errors)
    }
}

/// The body of a top-level `def` whose initializer is a lambda, which
/// [04-INF-7] makes a function declaration rather than an eager value.
fn function_declaration_body(children: &[Expr]) -> Option<&Expr> {
    children.first().and_then(symbol_name)?;
    children
        .get(1)
        .filter(|body| matches!(get_tag_expr(body), Some(DeepTag::Fn)))
}

impl Checker {
    /// Walk a program's top-level initializers in order, then its function
    /// declarations, cancellable between items (chelis#930).
    fn check_program_items(&mut self, exprs: &[Expr], scope: &mut LinearScope) {
        let cancel = crate::cancel::current_cancel_token();
        let cancelled = || cancel.as_ref().is_some_and(CancelToken::is_cancelled);
        for expr in exprs {
            if cancelled() {
                self.errors.push(crate::cancel::cancellation_check_error());
                return;
            }
            self.check_top_level(expr, scope);
        }
        for expr in exprs {
            if cancelled() {
                self.errors.push(crate::cancel::cancellation_check_error());
                return;
            }
            self.check_function_declarations(expr, scope);
        }
        for expr in exprs {
            self.observe_top_level_key_roots(expr, scope);
        }
    }

    /// [04-LIN-6] and [04-LIN-9]: every top-level value binding is a root
    /// ([05-OBS-7]), and its observation is, in manifest order, a terminal
    /// consuming use. A key-carrying binding that an initializer already
    /// consumed would be used twice, whether the other use is a key
    /// operation, a draw or a projection of one of its components.
    fn observe_top_level_key_roots(&mut self, expr: &Expr, scope: &mut LinearScope) {
        let children = match expr.carrier() {
            ExprCarrier::DecodedNode(DeepTag::Module, _, children) => {
                for child in children.iter().skip(1) {
                    self.observe_top_level_key_roots(child, scope);
                }
                return;
            }
            ExprCarrier::DecodedNode(DeepTag::Def, _, children) => children,
            ExprCarrier::DecodedNode(_, _, _)
            | ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => return,
        };
        let Some(name) = children.first().and_then(symbol_name) else {
            return;
        };
        if function_declaration_body(children).is_some() {
            return;
        }
        let Some(id) = scope.top_id(name) else {
            return;
        };
        let Some(record) = scope.record(id) else {
            return;
        };
        if !record.ty.as_ref().is_some_and(|ty| self.type_holds_key(ty)) {
            return;
        }
        let earlier = match &record.state {
            BindingState::Consumed(site) => Some(site.description.clone()),
            BindingState::Live { .. } if !record.moved_key_components.is_empty() => {
                Some("a projection of one of its key components".to_string())
            }
            BindingState::Live { .. } => None,
        };
        let site = children.get(1).map_or_else(String::new, diag_site);
        if let Some(earlier) = earlier {
            self.push_diagnostic(CheckError::new(
                CheckErrorKind::KeyReuse,
                format!(
                    "top-level key-carrying binding `{name}` is a root, and observing a root \
                     consumes it ([04-LIN-6]), but it was already consumed by {earlier}; a key \
                     is used at most once ([04-LIN-9]), so its initializer {site} cannot also \
                     feed another binding"
                ),
                vec![
                    "Bind only the keys a root shows: derive every other key inside the \
                     binding that uses it, for example `d = dropout(key_from_seed(1i64), x, \
                     0.5f32)`, or split a key inside a `def`"
                        .to_string(),
                ],
            ));
        }
        scope.consume_id(
            id,
            ConsumeSite {
                description: format!("the root observation of `{name}`"),
                kind: ConsumeKind::Structural,
                terminal: false,
            },
        );
    }

    /// chelis#2549: a function declaration ([04-INF-7]) is not a closure
    /// created in the top-level scope, and a call may run after every
    /// top-level initializer, including from another module. Each body is
    /// therefore checked against its own clone of the full top-level scope
    /// as it stands once every initializer has been walked: a value some
    /// initializer consumes is rejected whatever the def's text position,
    /// and nothing the body does changes top-level ownership state.
    fn check_function_declarations(&mut self, expr: &Expr, scope: &LinearScope) {
        match expr.carrier() {
            ExprCarrier::DecodedNode(DeepTag::Module, _, children) => {
                for child in children.iter().skip(1) {
                    self.check_function_declarations(child, scope);
                }
            }
            ExprCarrier::DecodedNode(DeepTag::Def, _, children) => {
                if let Some(body) = function_declaration_body(children) {
                    let mut declaration_scope = scope.clone();
                    self.check_expr(body, &mut declaration_scope);
                }
            }
            ExprCarrier::DecodedNode(_, _, _)
            | ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => {}
        }
    }

    fn check_top_level(&mut self, expr: &Expr, scope: &mut LinearScope) {
        // Linearity-F3 PR 1 + PR 2: recurse through `(module {} name
        // children...)` wrappers so module-wrapped top-level defs
        // participate in cross-statement linearity tracking. PR 1
        // routed diagnostics raised inside this recursion to a
        // warning channel for a deprecation window; PR 2 removed that
        // channel and unified the severity with bare-top-level
        // violations, so all `push_diagnostic` calls route to
        // `Checker::errors`.
        match expr.carrier() {
            ExprCarrier::DecodedNode(DeepTag::Module, _, children) => {
                for child in children.iter().skip(1) {
                    self.check_top_level(child, scope);
                }
            }
            ExprCarrier::DecodedNode(DeepTag::Def, _, children) => {
                if let (Some(name), Some(body)) =
                    (children.first().and_then(symbol_name), children.get(1))
                    && !(is_var_expr(body) && var_name(body) == Some(name))
                {
                    if matches!(get_tag_expr(body), Some(DeepTag::Borrow)) {
                        self.invalid_borrow(body, "borrow cannot be returned from a function");
                        return;
                    }
                    // chelis#2549: a function declaration's body is checked by
                    // `check_function_declarations` once every initializer
                    // has been walked, never at the def's text position.
                    if function_declaration_body(children).is_some() {
                        return;
                    }
                    // A top-level `def name = x` shares the same lowered
                    // node as `x`. Classify its var body as an aliasing
                    // binding, as `check_let` does, so a later borrow of
                    // `x` remains valid.
                    if is_var_expr(body) && self.expr_holds_key(body, scope) {
                        // [04-LIN-9]: `def a = b` of a key holder moves the
                        // key into `a`; it is not an aliasing share.
                        self.consume_var_expr(
                            body,
                            scope,
                            ConsumeSite {
                                description: format!("binding `{name}` {}", diag_site(body)),
                                kind: ConsumeKind::Structural,
                                terminal: false,
                            },
                        );
                    } else if is_var_expr(body) && self.expr_is_owned_linear(body, scope) {
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
                                terminal: false,
                            },
                        );
                        if let Some((alias_id, source_id)) = alias_link {
                            scope.record_alias(alias_id, source_id);
                        }
                    } else {
                        self.check_expr(body, scope);
                    }
                }
            }
            ExprCarrier::DecodedNode(tag, _, _) if is_runtime_expression_tag(tag) => {
                self.check_expr(expr, scope);
            }
            ExprCarrier::DecodedNode(_, _, _) => self.check_structural_payload(expr, scope),
            ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => self.check_expr(expr, scope),
        }
    }

    fn check_expr(&mut self, expr: &Expr, scope: &mut LinearScope) {
        match expr.carrier() {
            ExprCarrier::DecodedNode(tag, _, children) => {
                if !is_runtime_expression_tag(tag) {
                    self.push_diagnostic(CheckError::new(
                        CheckErrorKind::MalformedForm,
                        format!(
                            "non-runtime `{}` reached linearity runtime position {}; expected a \
                             runtime expression ([04-TOT-3]; chelis#1125)",
                            tag.as_str(),
                            diag_site(expr)
                        ),
                        vec![],
                    ));
                    // A structural owner in a runtime slot has no authority
                    // to classify any physical child as syntax-only.
                    for child in children {
                        self.check_untrusted_runtime_descendants(child, scope);
                    }
                    return;
                }
                if !decoded_shape_is_valid(tag, children.len()) {
                    self.push_diagnostic(CheckError::new(
                        CheckErrorKind::MalformedForm,
                        format!(
                            "malformed `{}` reached linearity {}: child count {} violates {:?} \
                             ([04-TOT-3]; chelis#1125)",
                            tag.as_str(),
                            diag_site(expr),
                            children.len(),
                            arity_contract(tag)
                        ),
                        vec![],
                    ));
                    // Once the parent's shape is malformed, its ordinary
                    // child-role table no longer proves that any physical
                    // child is merely structural. Walk every child as a
                    // possible runtime expression so an unexpected nested
                    // consume cannot disappear behind a syntax/type role.
                    for child in children {
                        self.check_untrusted_runtime_descendants(child, scope);
                    }
                    return;
                }
                if let TagKeys::Refuses(refusal) = tag_keys(tag) {
                    self.refuse_key_children(tag, children, refusal, scope);
                }
                let entered_projection = matches!(tag, DeepTag::TupleGet | DeepTag::Access)
                    && self.projection_root.is_none()
                    && self.check_projection_use(expr, scope);
                match tag {
                    DeepTag::Var => self.consume_var_expr(expr, scope, generic_site(expr)),
                    DeepTag::Copy => self.check_copy(children, scope),
                    DeepTag::Realize => self.check_realize(expr, children, scope),
                    DeepTag::Borrow => {
                        self.invalid_borrow(expr, "borrow is only valid as a direct call argument")
                    }
                    DeepTag::App => self.check_app(expr, children, scope),
                    // chelis#1923: a pipe cannot reach linearity, which reads the
                    // checked program, because every checker entry folds it into
                    // the application it denotes. The arm that used to sit here
                    // peered through the desugarer's synthesized stage lambda to
                    // ask the inner callee whether the piped value is borrowed or
                    // consumed; the folded application puts that callee in callee
                    // position, so `check_app` asks the same question once. Fail
                    // closed rather than falling through to the structural walk,
                    // which would answer silently.
                    DeepTag::Pipe => self.push_diagnostic(CheckError::new(
                        CheckErrorKind::MalformedForm,
                        "a pipe reached linearity unfolded: every checker entry folds a pipe into \
                         the application it denotes (spec/02-surf-syntax.md section 0.1; \
                         chelis#1923)"
                            .to_string(),
                        vec![],
                    )),
                    DeepTag::Let => self.check_let(children, scope),
                    DeepTag::Fn => self.check_fn(expr, children, scope),
                    DeepTag::If => self.check_if(children, scope),
                    DeepTag::Match => self.check_match(children, scope),
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
                    DeepTag::TupleGet => self.check_tuple_get(children, scope),
                    DeepTag::Cast => self.check_cast(children, scope),
                    _ => self.check_children_by_role(tag, children, scope),
                }
                if entered_projection {
                    self.projection_root = None;
                }
            }
            ExprCarrier::StructuralList(elements) => {
                self.reject_non_runtime_carrier(expr, "a structural list");
                for element in elements {
                    self.check_expr(element, scope);
                }
            }
            ExprCarrier::UndecodableHead(head, _, children) => {
                self.push_diagnostic(CheckError::new(
                    CheckErrorKind::MalformedForm,
                    format!(
                        "undecodable Deep head `{head}` reached linearity {}; expected a decoded \
                         vocabulary node ([04-TOT-5]; chelis#1125)",
                        diag_site(expr)
                    ),
                    vec![],
                ));
                for child in children {
                    self.check_expr(child, scope);
                }
            }
            ExprCarrier::Atom(Atom::Name(_)) => {
                self.reject_non_runtime_carrier(expr, "an atom");
            }
            // `Node::validate` admits literal atoms at RuntimeExpr positions:
            // they carry no variable use or ownership event for linearity.
            ExprCarrier::Atom(_) => {}
            ExprCarrier::MetadataMap(map) => {
                self.reject_non_runtime_carrier(expr, "a standalone metadata map");
                map.visit_syntax(&mut |_, value| {
                    self.check_expr(value, scope);
                });
            }
            ExprCarrier::MetadataExpression(meta) => {
                self.reject_non_runtime_carrier(expr, "a metadata expression wrapper");
                self.check_expr(&meta.expr, scope);
            }
        }
    }

    fn reject_non_runtime_carrier(&mut self, expr: &Expr, carrier: &str) {
        self.push_diagnostic(CheckError::new(
            CheckErrorKind::MalformedForm,
            format!(
                "{carrier} reached linearity runtime position {}; expected a decoded Deep \
                 vocabulary node ([04-TOT-5]; chelis#1125)",
                diag_site(expr)
            ),
            vec![],
        ));
    }

    /// chelis#3101, [05-OP-63], [05-OP-6], [05-OP-23], [05-OP-24]: every
    /// cast rung's tensor source is a read-only parameter (spec/05 section
    /// 1.3.1). An explicit borrow there is a read, as at a call argument, and
    /// an owned variable auto-borrows: it is read, not consumed, as a
    /// borrowed builtin argument is. A key holder is never borrowed
    /// ([04-LIN-9]), so it keeps the by-role check. The target and mode slots
    /// are types and selectors.
    fn check_cast(&mut self, children: &[Expr], scope: &mut LinearScope) {
        let Some(source) = children.first() else {
            return;
        };
        if let Some(inner) = borrow_inner(source) {
            self.check_borrow_arg(source, inner, scope);
        } else if is_var_expr(source)
            && !self.expr_holds_key(source, scope)
            && self.expr_is_owned_linear(source, scope)
        {
            self.read_var_expr(source, scope);
        } else {
            self.check_child_by_role(DeepTag::Cast, 0, children.len(), source, scope);
        }
        for (index, child) in children.iter().enumerate().skip(1) {
            self.check_child_by_role(DeepTag::Cast, index, children.len(), child, scope);
        }
    }

    fn check_children_by_role(
        &mut self,
        parent_tag: DeepTag,
        children: &[Expr],
        scope: &mut LinearScope,
    ) {
        for (index, child) in children.iter().enumerate() {
            self.check_child_by_role(parent_tag, index, children.len(), child, scope);
        }
    }

    fn check_child_by_role(
        &mut self,
        parent_tag: DeepTag,
        index: usize,
        child_count: usize,
        child: &Expr,
        scope: &mut LinearScope,
    ) {
        match child_stamp_role(parent_tag, index, child_count) {
            ChildStampRole::RuntimeExpr => self.check_expr(child, scope),
            ChildStampRole::ExplicitInferenceBypass | ChildStampRole::EffectHandler => {
                self.check_structural_payload(child, scope);
            }
            ChildStampRole::Syntax
            | ChildStampRole::Selector
            | ChildStampRole::Binder
            | ChildStampRole::Type => {}
        }
    }

    fn check_structural_payload(&mut self, expr: &Expr, scope: &mut LinearScope) {
        match expr.carrier() {
            ExprCarrier::DecodedNode(tag, _, _) if is_runtime_expression_tag(tag) => {
                self.check_expr(expr, scope);
            }
            ExprCarrier::DecodedNode(_, _, children) => {
                for child in children {
                    self.check_structural_payload(child, scope);
                }
            }
            ExprCarrier::StructuralList(elements) => {
                for element in elements {
                    self.check_structural_payload(element, scope);
                }
            }
            ExprCarrier::UndecodableHead(_, _, children) => {
                for child in children {
                    self.check_structural_payload(child, scope);
                }
            }
            ExprCarrier::Atom(_) => {}
            ExprCarrier::MetadataMap(map) => {
                map.visit_syntax(&mut |_, value| {
                    self.check_structural_payload(value, scope);
                });
            }
            ExprCarrier::MetadataExpression(meta) => {
                self.check_structural_payload(&meta.expr, scope);
            }
        }
    }

    fn check_copy(&mut self, children: &[Expr], scope: &mut LinearScope) {
        if let Some(child) = children.first() {
            let copied = borrow_inner(child).unwrap_or(child);
            if self.operand_holds_key(child, scope) {
                // [04-LIN-9]: a key is never copied, explicitly or by the
                // compiler; a second key comes from deriving, not copying.
                // `check_expr` refused it by the allow-list (`tag_keys`); the
                // copied expression is still walked for its own uses.
                if !is_var_expr(copied) {
                    self.check_expr(copied, scope);
                }
                return;
            }
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
    fn check_tuple_get(&mut self, children: &[Expr], scope: &mut LinearScope) {
        if let Some(target) = children.first() {
            if is_var_expr(target) && self.expr_holds_key(target, scope) {
                self.project_key_holder(target, children.get(1), scope);
            } else if is_var_expr(target) && self.expr_is_owned_linear(target, scope) {
                self.read_var_expr(target, scope);
            } else {
                self.check_expr(target, scope);
            }
        }
        // Preserve the canonical role decision for the selector rather than
        // treating a legal bare integer index as a runtime expression.
        for (index, child) in children.iter().enumerate().skip(1) {
            self.check_child_by_role(DeepTag::TupleGet, index, children.len(), child, scope);
        }
    }

    fn check_realize(&mut self, expr: &Expr, children: &[Expr], scope: &mut LinearScope) {
        if let Some(child) = children.first() {
            if is_var_expr(child) && self.expr_is_owned_linear(child, scope) {
                self.consume_var_expr(child, scope, realize_site(expr));
            } else {
                self.check_expr(child, scope);
            }
        }
    }

    fn check_app(&mut self, expr: &Expr, children: &[Expr], scope: &mut LinearScope) {
        let builtin = children.first().and_then(var_name);
        let builtin_callee = builtin.filter(|name| self.is_builtin_reference(name, scope));
        // Borrow dispositions also cover intrinsic calls such as `dropout`,
        // which intentionally lives outside the ordinary builtin vocabulary.
        // The closed borrow policy below recognizes the intrinsic; lexical
        // bindings must still use their actual function signature.
        let borrowing_callee = builtin.filter(|name| scope.top_id(name).is_none());
        // A builtin callee is judged operand by operand below; a builtin
        // named as a value is judged by the type checker
        // (`infer::expr::forbid_keys_a_builtin_value_does_not_admit`).
        if let Some(func) = children.first()
            && builtin_callee.is_none()
        {
            self.check_expr(func, scope);
        }
        let verdicts = self.admit_key_operands(builtin_callee, expr, &children[1..], scope);
        let observational = children
            .first()
            .is_some_and(callee_is_observational_higher_order);
        if children
            .first()
            .is_some_and(|callee| get_tag_expr(callee) == Some(DeepTag::Vmap))
        {
            self.reject_broadcast_keys(&children[1..], scope);
        }
        for (index, arg) in children.iter().enumerate().skip(1) {
            if let Some(borrowed) = borrow_inner(arg) {
                self.check_borrow_arg(arg, borrowed, scope);
            } else if is_var_expr(arg)
                && self.expr_holds_key(arg, scope)
                && verdicts[index - 1] == KeyOperand::ExtentRead
            {
                // [04-LIN-9]: a key tensor's extent is not key material, so
                // reading it is not a use and leaves the key live.
                self.read_var_expr(arg, scope);
            } else if is_var_expr(arg) && self.expr_holds_key(arg, scope) {
                // [04-LIN-9]: a key holder is always consumed by a call. The
                // arguments of a `grad(f)(..)` or `vmap(f)(..)` call are
                // otherwise observed; a key argument is consumed there too.
                // A callee position that only borrows cannot take one.
                if verdicts[index - 1] != KeyOperand::Refused
                    && !observational
                    && self.arg_is_borrowed(children.first(), borrowing_callee, index - 1, scope)
                {
                    self.reject_key_read(arg, "borrowed by this call");
                } else {
                    self.consume_var_expr(arg, scope, app_site(expr, children, builtin_callee));
                }
            } else if self.arg_is_borrowed(children.first(), borrowing_callee, index - 1, scope)
                && is_var_expr(arg)
                && self.expr_is_owned_linear(arg, scope)
            {
                self.read_var_expr(arg, scope);
            } else if is_var_expr(arg) && self.expr_is_owned_linear(arg, scope) {
                self.consume_var_expr(arg, scope, app_site(expr, children, builtin_callee));
            } else {
                self.check_expr(arg, scope);
            }
        }
        if builtin_callee == Some("drop")
            && let [_, operand] = children
        {
            self.record_component_drop(operand, &app_site(expr, children, builtin_callee), scope);
        }
        self.maybe_mark_reusable_app_input(expr, children, scope);
    }

    fn check_borrow_arg(&mut self, borrow_expr: &Expr, inner: &Expr, scope: &mut LinearScope) {
        // [04-LIN-9]: a key is never borrowed; the application's allow-list
        // check (`admit_key_operands`) refused it.
        if self.operand_holds_key(borrow_expr, scope) {
            return;
        }
        if !is_var_expr(inner) {
            self.invalid_borrow(
                borrow_expr,
                "borrowed arguments must be direct variable references",
            );
            return;
        }
        if !self.borrow_target_is_linear(borrow_expr, inner, scope) {
            self.invalid_borrow(
                borrow_expr,
                "borrowed arguments must be tensor or tensor-carrying values",
            );
            return;
        }
        self.read_var_expr(inner, scope);
    }

    /// chelis#1589: classify a borrow target, falling back to the
    /// `borrow` node's own resolved stamp when the inner carries no
    /// visible type.
    ///
    /// `expr_type` can see nothing at all for the inner of a borrow
    /// whose parameter annotation was a synthesized inference hole:
    /// `DeepTag::Var` is in `should_attach_type_metadata`'s deny list so
    /// a `(var ..)` node never carries `:type`, and the scope entry that
    /// `param_name_and_type` builds is empty because the parameter was
    /// emitted as a bare name. `expr_is_owned_or_borrow_linear` is an
    /// `is_some_and`, so absent information became a rejection: a
    /// fail-closed, not a classification.
    ///
    /// `spec/04-type-system.md` §8.2 decides the rule on the type the
    /// inner "must be — or must ultimately resolve to", and by the time
    /// linearity runs that resolution has happened: the `borrow` node is
    /// metadata-eligible and the annotate pass stamps it with the
    /// resolved `&T`. Read that stamp instead of rejecting.
    ///
    /// This never weakens the deferred-borrow gate. `check_linearity`
    /// runs only on a successful type analysis
    /// (`chelis_pipeline_core::semantic::complete_checks_in_context`),
    /// and `validate_deferred_borrow_vars` rejects an unsound deferral
    /// during inference, so a borrow whose deferred classification
    /// failed never reaches this function. The validator remains the
    /// sole authority on the deferred path; this gate is a redundancy
    /// check on the resolved type.
    fn borrow_target_is_linear(
        &self,
        borrow_expr: &Expr,
        inner: &Expr,
        scope: &LinearScope,
    ) -> bool {
        if self.expr_type(inner, scope).is_some() {
            return self.expr_is_owned_or_borrow_linear(inner, scope);
        }
        // Both predicates already recurse through `DeepTag::TRef`, so the
        // stamped `(t-ref ..)` is passed through without peeling.
        type_metadata(borrow_expr).is_some_and(|ty| {
            type_expr_proves_tensor(ty, &self.tensor_carrying_adts)
                || type_expr_is_unresolved_tvar(ty)
        })
    }

    fn check_let(&mut self, children: &[Expr], scope: &mut LinearScope) {
        if children.len() < 2 {
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
        let bind_introduces_destructure = bind_introduces_destructure_tmp(&children[0]);
        if let Some(bind_kids) = tagged_children(&children[0], DeepTag::Bind) {
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
                if is_var_expr(value) && self.expr_holds_key(value, scope) {
                    // [04-LIN-9]: `y = k` moves a key holder into `y`. It is
                    // not an aliasing share, so no alias link is recorded and
                    // the source is gone.
                    self.consume_var_expr(
                        value,
                        scope,
                        ConsumeSite {
                            description: format!("binding `{name}` {}", diag_site(value)),
                            kind: ConsumeKind::Structural,
                            terminal: false,
                        },
                    );
                } else if is_var_expr(value) && self.expr_is_owned_linear(value, scope) {
                    alias_source_id = var_name(value).and_then(|source| scope.top_id(source));
                    self.consume_var_expr(
                        value,
                        scope,
                        ConsumeSite {
                            description: format!("binding `{name}` {}", diag_site(value)),
                            kind: ConsumeKind::Aliasing,
                            terminal: false,
                        },
                    );
                } else if matches!(get_tag_expr(value), Some(DeepTag::Borrow)) {
                    self.invalid_borrow(value, "borrow cannot be stored in a binding");
                } else {
                    self.check_expr(value, scope);
                }
                let id = scope.declare(name, self.value_type(value, scope));
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
        self.check_expr(&children[1], scope);
        for id in pushed.into_iter().rev() {
            self.pop_and_check(scope, id, expr_scope_end(&children[1]));
        }
    }

    fn check_fn(&mut self, expr: &Expr, children: &[Expr], outer_scope: &mut LinearScope) {
        if children.len() < 2 {
            return;
        }

        let params = param_names(&children[0]);
        let captured = free_vars(&children[1], &params);
        let mut inner_scope = outer_scope.clone();
        let body = &children[1];
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
            if self.type_holds_key(ty) {
                // [04-LIN-9]: function types record no captures, and a
                // closure that used a captured key would use it once per
                // call. Keys are passed as parameters.
                self.push_diagnostic(CheckError::new(
                    CheckErrorKind::KeyReuse,
                    with_macro_provenance(
                        expr,
                        format!(
                            "closure {} captures key-carrying variable `{name}`; a closure may \
                             not capture a key, because every call would use it again \
                             ([04-LIN-9])",
                            diag_site(expr)
                        ),
                    ),
                    vec![format!(
                        "Pass `{name}` to the closure as a parameter instead of capturing it"
                    )],
                ));
                inner_scope.declare(name.clone(), outer_scope.ty(&name).cloned());
                continue;
            }
            if !type_expr_may_contain_tensor(ty, &self.tensor_carrying_adts) {
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
                            terminal: false,
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
                if let Some(use_id) = outer_scope.top_id(&name) {
                    let owner = outer_scope.resolve_alias_chain(use_id).unwrap_or(use_id);
                    if let Some(record) = outer_scope.record_mut(owner) {
                        record
                            .closure_borrows
                            .push(format!("closure {}", diag_site(expr)));
                    }
                }
                inner_scope.declare(name.clone(), outer_scope.ty(&name).cloned());
            }
        }

        let mut pushed: Vec<(String, BindingId)> = Vec::new();
        // chelis#2544: a parameter's type is the checked function type's, not
        // its annotation. An unannotated parameter the checker resolved to a
        // key (`map(fn (y) -> (y, y), keys)`) is a key holder like an
        // annotated one; reading the annotation alone declared it untyped, so
        // [04-LIN-9] never saw its second use. The annotation is the fallback
        // only where the node carries no resolved type.
        let checked_fn_type = type_metadata(expr);
        if let Some(params) = tagged_children(&children[0], DeepTag::Params) {
            for (index, param) in params.iter().enumerate() {
                if let Some((name, ty)) = param_name_and_type(param) {
                    let checked = checked_fn_type.and_then(|fn_ty| type_expr_fn_arg(fn_ty, index));
                    // The checked type, not the annotation: a `sig` declares
                    // the parameter's type away from the parameter, and an
                    // alias (`type K = key; def f(k: &K)`) reaches the
                    // annotation unresolved.
                    if let Some(declared) = checked.or(ty) {
                        self.reject_borrowed_key_parameter(expr, name, declared);
                    }
                    let id = inner_scope.declare(name, checked.or(ty).cloned());
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
        if matches!(get_tag_expr(&children[1]), Some(DeepTag::Borrow)) {
            self.invalid_borrow(&children[1], "borrow cannot be returned from a function");
        } else {
            self.check_expr(&children[1], &mut inner_scope);
        }
        for (_, id) in pushed.into_iter().rev() {
            self.pop_and_check_param(&mut inner_scope, id, expr_scope_end(&children[1]));
        }
    }

    fn check_if(&mut self, children: &[Expr], scope: &mut LinearScope) {
        if children.len() < 3 {
            return;
        }
        self.check_expr(&children[0], scope);
        let visible_ids = scope.all_visible_ids();
        let mut then_scope = scope.clone();
        let mut else_scope = scope.clone();
        // chelis#1200 Q1: a branch body is a new declaration region, so
        // the destructured-component mark does not cross into it. See
        // `LinearScope::clear_destructured_marks`.
        then_scope.clear_destructured_marks();
        else_scope.clear_destructured_marks();
        self.check_expr(&children[1], &mut then_scope);
        self.check_expr(&children[2], &mut else_scope);
        self.join_branch_states(scope, &visible_ids, &[then_scope, else_scope]);
    }

    fn check_match(&mut self, children: &[Expr], scope: &mut LinearScope) {
        if children.is_empty() {
            return;
        }
        if is_var_expr(&children[0]) && self.expr_is_owned_linear(&children[0], scope) {
            self.consume_var_expr(
                &children[0],
                scope,
                ConsumeSite {
                    description: format!("match scrutinee {}", diag_site(&children[0])),
                    kind: ConsumeKind::Structural,
                    terminal: false,
                },
            );
        } else {
            self.check_expr(&children[0], scope);
        }

        let visible_ids = scope.all_visible_ids();
        let mut arm_scopes = Vec::new();
        // [04-LIN-9] on every control-flow path: an arm's guard runs before
        // its body and, when it fails, before every later arm. `fallthrough`
        // is the state on entry to the next arm, so a key a guard consumed
        // is consumed there; `guard_paths` names the scrutinee components a
        // guard consumed through the arm's own pattern bindings.
        let mut fallthrough = scope.clone();
        let mut guard_paths: Vec<(Vec<PatternStep>, ConsumeSite)> = Vec::new();
        for arm in children.iter().skip(1) {
            let mut arm_scope = fallthrough.clone();
            // An arm, including one whose decoded shape is malformed, is a
            // distinct declaration region. Preserve every nested ownership
            // event in that branch before joining it back into the match.
            arm_scope.clear_destructured_marks();
            let Some(arm_kids) = tagged_children(arm, DeepTag::Arm) else {
                self.check_malformed_match_arm(arm, &mut arm_scope);
                arm_scopes.push(arm_scope);
                continue;
            };
            self.reject_key_as_pattern_aliasing(&arm_kids[0]);
            let pattern_bindings = pattern_bindings_with_paths(&arm_kids[0]);
            let mut pattern_ids = Vec::new();
            for binding in &pattern_bindings {
                let id = arm_scope.declare(binding.name.clone(), binding.ty.clone());
                let holds_key = binding
                    .ty
                    .as_ref()
                    .is_some_and(|ty| self.type_holds_key(ty));
                if holds_key
                    && let Some((_, site)) = guard_paths
                        .iter()
                        .find(|(path, _)| pattern_paths_overlap(path, &binding.path))
                {
                    arm_scope.consume_id(id, site.clone());
                }
                pattern_ids.push(id);
            }
            // The empty list in an arm's guard slot is the grammar's
            // explicit "no guard" sentinel, not a runtime expression.
            // Screen that declared omission here so `check_expr` can reject
            // the same carrier when it actually occupies a runtime slot.
            if !is_absent_match_guard(&arm_kids[1]) {
                self.check_expr(&arm_kids[1], &mut arm_scope);
                self.carry_guard_key_consumes(
                    &arm_scope,
                    &mut fallthrough,
                    &visible_ids,
                    &pattern_bindings,
                    &pattern_ids,
                    &mut guard_paths,
                );
            }
            self.check_expr(&arm_kids[2], &mut arm_scope);
            for id in pattern_ids.into_iter().rev() {
                self.pop_and_check(&mut arm_scope, id, expr_scope_end(&arm_kids[2]));
            }
            arm_scopes.push(arm_scope);
        }
        self.join_branch_states(scope, &visible_ids, &arm_scopes);
    }

    /// [04-LIN-9]: a guard that fails falls through to the next arm, so the
    /// keys it consumed stay consumed on entry to every later arm. An outer
    /// key holder the guard consumed is consumed in `fallthrough`; a key the
    /// guard took through one of the arm's pattern bindings is recorded as
    /// the scrutinee component that binding names, so a later arm binding an
    /// overlapping component starts with it consumed. Only key holders carry
    /// over: a tensor a guard consumes keeps the implicit-copy rules.
    fn carry_guard_key_consumes(
        &self,
        arm_scope: &LinearScope,
        fallthrough: &mut LinearScope,
        visible_ids: &[BindingId],
        pattern_bindings: &[PatternBinding],
        pattern_ids: &[BindingId],
        guard_paths: &mut Vec<(Vec<PatternStep>, ConsumeSite)>,
    ) {
        let fell_through = |site: &ConsumeSite| ConsumeSite {
            description: format!(
                "{} in the guard of an earlier arm, which falls through to this one",
                site.description
            ),
            kind: ConsumeKind::Structural,
            terminal: false,
        };
        for id in visible_ids {
            let Some(record) = arm_scope.record(*id) else {
                continue;
            };
            if !record.ty.as_ref().is_some_and(|ty| self.type_holds_key(ty)) {
                continue;
            }
            let moved = record.moved_key_components.clone();
            if let BindingState::Consumed(site) = &record.state
                && matches!(fallthrough.state(*id), Some(BindingState::Live { .. }))
            {
                fallthrough.consume_id(*id, fell_through(site));
            }
            if let Some(outer) = fallthrough.record_mut(*id) {
                outer.moved_key_components.extend(moved);
            }
        }
        for (binding, id) in pattern_bindings.iter().zip(pattern_ids) {
            let Some(record) = arm_scope.record(*id) else {
                continue;
            };
            if !record.ty.as_ref().is_some_and(|ty| self.type_holds_key(ty)) {
                continue;
            }
            if let BindingState::Consumed(site) = &record.state {
                guard_paths.push((binding.path.clone(), fell_through(site)));
            }
            for position in &record.moved_key_components {
                let mut path = binding.path.clone();
                path.push(PatternStep::Tuple(*position));
                guard_paths.push((
                    path,
                    ConsumeSite {
                        description: format!(
                            "a projection of `{}` in the guard of an earlier arm, which falls \
                             through to this one",
                            binding.name
                        ),
                        kind: ConsumeKind::Structural,
                        terminal: false,
                    },
                ));
            }
        }
    }

    /// [04-LIN-9]: a key inside a value is reached only by consuming that
    /// value, so an as-pattern cannot bind a key-carrying whole together
    /// with a sub-pattern that binds a key-carrying component: the whole
    /// and the component would be two live owners of one key. A sub-pattern
    /// that binds no key (`whole @ Some(_)`) stays legal.
    fn reject_key_as_pattern_aliasing(&mut self, pattern: &Expr) {
        let Some((tag, children)) = decoded_parts(pattern) else {
            return;
        };
        if tag == DeepTag::PatAs
            && let (Some(whole), Some(inner)) =
                (children.first().and_then(symbol_name), children.get(1))
        {
            let component = pattern_bindings_with_paths(inner)
                .into_iter()
                .find(|binding| {
                    binding
                        .ty
                        .as_ref()
                        .is_some_and(|ty| self.type_holds_key(ty))
                });
            if let Some(component) = component {
                self.push_diagnostic(CheckError::new(
                    CheckErrorKind::KeyReuse,
                    with_macro_provenance(
                        pattern,
                        format!(
                            "as-pattern `{whole}` {} binds a key-carrying value together with \
                             its key-carrying component `{}`; a key inside a value is reached \
                             only by consuming that value, so the two names would use one key \
                             twice ([04-LIN-9])",
                            diag_site(pattern),
                            component.name
                        ),
                    ),
                    vec![format!(
                        "Bind either `{whole}` or its key-carrying components, not both; derive \
                         a second key with `split_key(k)` where both uses need one"
                    )],
                ));
            }
        }
        for child in children {
            self.reject_key_as_pattern_aliasing(child);
        }
    }

    fn check_malformed_match_arm(&mut self, arm: &Expr, scope: &mut LinearScope) {
        self.push_diagnostic(CheckError::new(
            CheckErrorKind::MalformedForm,
            format!(
                "malformed match arm reached linearity {}; expected a decoded `arm` with exactly \
                 pattern, guard, and body children ([04-TOT-3]; chelis#1125)",
                diag_site(arm)
            ),
            vec![],
        ));

        // The enclosing Match position, not the malformed owner's own tag,
        // determines the safe fallback. Once `arm` authority is absent, no
        // nested structural owner's role table can prove that its physical
        // descendants are non-runtime.
        self.check_untrusted_runtime_descendants(arm, scope);
    }

    fn check_untrusted_runtime_descendants(&mut self, expr: &Expr, scope: &mut LinearScope) {
        match expr.carrier() {
            ExprCarrier::DecodedNode(tag, _, children)
                if is_runtime_expression_tag(tag)
                    && decoded_shape_is_valid(tag, children.len()) =>
            {
                self.check_expr(expr, scope);
            }
            ExprCarrier::DecodedNode(_, _, children)
            | ExprCarrier::StructuralList(children)
            | ExprCarrier::UndecodableHead(_, _, children) => {
                for child in children {
                    self.check_untrusted_runtime_descendants(child, scope);
                }
            }
            ExprCarrier::Atom(_) => {}
            ExprCarrier::MetadataMap(map) => {
                map.visit_syntax(&mut |_, value| {
                    self.check_untrusted_runtime_descendants(value, scope);
                });
            }
            ExprCarrier::MetadataExpression(meta) => {
                self.check_untrusted_runtime_descendants(&meta.expr, scope);
            }
        }
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
            // [04-LIN-9]: a key component a branch moved out of a holder is
            // gone after the join, whichever arm ran.
            let moved_in_branches: BTreeSet<usize> = branches
                .iter()
                .filter_map(|branch| branch.record(*id))
                .flat_map(|record| record.moved_key_components.iter().copied())
                .collect();
            if !moved_in_branches.is_empty()
                && let Some(record) = scope.record_mut(*id)
            {
                record.moved_key_components.extend(moved_in_branches);
            }
            // A closure a branch created may outlive the join, so its borrow
            // does too.
            for branch in branches {
                let Some(branch_record) = branch.record(*id) else {
                    continue;
                };
                if let Some(record) = scope.record_mut(*id) {
                    for site in &branch_record.closure_borrows {
                        if !record.closure_borrows.contains(site) {
                            record.closure_borrows.push(site.clone());
                        }
                    }
                    // [04-LIN-11]: a component a branch dropped is gone
                    // after the join, whichever arm ran.
                    for dropped in &branch_record.dropped_components {
                        if !record.dropped_components.contains(dropped) {
                            record.dropped_components.push(dropped.clone());
                        }
                    }
                }
            }
            // A branch's `Structural` consume is the one that destroys the
            // value, so it is the one that must survive the join. Prefer it
            // over an `Aliasing` record from another branch: an alias bind
            // does not destroy anything, and letting it win would report the
            // wrong site. A terminal consume ends the owner on its path, so
            // it outranks an ordinary one (chelis#3177).
            let consumed_site = branches
                .iter()
                .find_map(|branch| match branch.state(*id) {
                    Some(BindingState::Consumed(site)) if site.terminal => Some(site.clone()),
                    _ => None,
                })
                .or_else(|| {
                    branches.iter().find_map(|branch| match branch.state(*id) {
                        Some(BindingState::Consumed(site))
                            if matches!(site.kind, ConsumeKind::Structural) =>
                        {
                            Some(site.clone())
                        }
                        _ => None,
                    })
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
            //   Consumed, not by a `drop`,
            //     and a branch dropped
            //     the owner               -> the `drop` wins, as in
            //                                `consume_var_expr` (chelis#3177).
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
                    (site.terminal && !outer.terminal)
                        || (matches!(outer.kind, ConsumeKind::Aliasing)
                            && matches!(site.kind, ConsumeKind::Structural)
                            && scope.is_component_id(*id))
                }
                None => false,
            };
            if replaces_outer {
                scope.consume_id(*id, site);
            }
        }
    }

    fn maybe_mark_reusable_app_input(&mut self, expr: &Expr, kids: &[Expr], scope: &LinearScope) {
        // Buffer reuse is a tensor storage fact; a key is never reused in
        // place, so key ownership ([04-LIN-9]) does not qualify a call here.
        if !self
            .expr_type(expr, scope)
            .is_some_and(|ty| type_expr_is_owned_linear(ty, &self.tensor_carrying_adts))
        {
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

    /// [04-LIN-11]: check a projection chain rooted at a variable against the
    /// components a `drop` moved out of that variable's owner, then mark the
    /// owner as the chain's root so the root's own read or consume inside the
    /// chain is not taken for a use of the owner whole. Returns whether the
    /// root was marked.
    fn check_projection_use(&mut self, expr: &Expr, scope: &LinearScope) -> bool {
        let Some((root, path)) = projection_chain(expr) else {
            return false;
        };
        let Some(name) = var_name(root) else {
            return false;
        };
        let Some(use_id) = scope.top_id(name) else {
            return false;
        };
        let owner = scope.resolve_alias_chain(use_id).unwrap_or(use_id);
        if let Some((dropped, site)) = dropped_component_overlapping(scope, &[use_id, owner], &path)
        {
            self.report_use_after_component_drop(name, &dropped, &site, expr);
        }
        self.projection_root = Some(owner);
        true
    }

    /// [04-LIN-11]: refuse a use of `name`'s owner whole after a `drop` moved
    /// one of its components out. Returns whether it reported.
    fn reject_use_after_component_drop(
        &mut self,
        name: &str,
        expr: &Expr,
        scope: &LinearScope,
    ) -> bool {
        let Some(use_id) = scope.top_id(name) else {
            return false;
        };
        let owner = scope.resolve_alias_chain(use_id).unwrap_or(use_id);
        if self.projection_root == Some(owner) {
            return false;
        }
        match dropped_component_overlapping(scope, &[use_id, owner], &[]) {
            Some((dropped, site)) => {
                self.report_use_after_component_drop(name, &dropped, &site, expr);
                true
            }
            None => false,
        }
    }

    fn report_use_after_component_drop(
        &mut self,
        name: &str,
        dropped: &[ProjectionStep],
        site: &str,
        expr: &Expr,
    ) {
        let component = render_projection(dropped);
        self.push_diagnostic(CheckError::new(
            CheckErrorKind::UseAfterConsume,
            with_macro_provenance(
                expr,
                format!(
                    "variable `{name}` had its component `{name}{component}` moved out and \
                     consumed by {site}; later use {} of `{name}` or of that component is invalid \
                     ([04-LIN-11])",
                    diag_site(expr)
                ),
            ),
            vec![format!(
                "Use only the components of `{name}` that were not dropped, or bind \
                 `copy({name}{component})` and drop that instead"
            )],
        ));
    }

    /// [04-LIN-11]: a `drop` of a projection rooted at an owned variable moves
    /// that component out of the variable's owner.
    fn record_component_drop(
        &mut self,
        operand: &Expr,
        site: &ConsumeSite,
        scope: &mut LinearScope,
    ) {
        let Some((root, path)) = projection_chain(operand) else {
            return;
        };
        if path.is_empty()
            || !self.expr_is_owned_linear(root, scope)
            || self.expr_holds_key(root, scope)
        {
            return;
        }
        let Some(use_id) = var_name(root).and_then(|name| scope.top_id(name)) else {
            return;
        };
        let owner = scope.resolve_alias_chain(use_id).unwrap_or(use_id);
        if let Some(record) = scope.record_mut(owner) {
            record
                .dropped_components
                .push((path, site.description.clone()));
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
        if self.reject_use_after_component_drop(name, expr, scope) {
            return;
        }
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
        if self.expr_holds_key(expr, scope) {
            self.consume_key_holder(expr, name, target, scope, site);
            return;
        }
        // A `drop` ends the owner, so every later use is refused, whatever
        // this use's own kind (chelis#3177). The owner is checked through the
        // alias chain as well as on the use's own binding: after `y = x;
        // c = drop(x)`, binding `z = y` is an `Aliasing` consume whose target
        // is `y`'s live record, but it names the dropped owner.
        let owner = scope.resolve_alias_chain(use_id).unwrap_or(use_id);
        let ended_at = [use_id, owner]
            .into_iter()
            .find_map(|id| match scope.state(id) {
                Some(BindingState::Consumed(consumed_at)) if consumed_at.terminal => {
                    Some(consumed_at.description.clone())
                }
                _ => None,
            });
        if let Some(description) = ended_at {
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
                vec![format!(
                    "Move the later use before the `drop`, or bind `copy({name})` before it"
                )],
            ));
            return;
        }
        // A `drop` cannot end an owner that a closure still borrows: the
        // closure would read it after its lifetime ends ([04-LIN-2]).
        if site.terminal
            && let Some(closure) = scope
                .record(owner)
                .and_then(|record| record.closure_borrows.first().cloned())
        {
            self.push_diagnostic(CheckError::new(
                CheckErrorKind::InvalidBorrow,
                with_macro_provenance(
                    expr,
                    format!(
                        "variable `{name}` is dropped {} while the {closure} still borrows it; \
                         the closure would read it after its lifetime ends ([04-LIN-2])",
                        diag_site(expr)
                    ),
                ),
                vec![format!(
                    "Remove the `drop` and let the compiler release `{name}` after its last \
                     use, or bind a `copy({name})` before the closure and capture that instead"
                )],
            ));
            return;
        }
        match scope.state(target) {
            Some(BindingState::Live { .. }) => scope.consume_id(target, site),
            Some(BindingState::Consumed(consumed_at))
                if matches!(consumed_at.kind, ConsumeKind::Structural)
                    && (consumed_at.description.contains("closure capture")
                        || consumed_at.description.contains("match scrutinee")) =>
            {
                let description = consumed_at.description.clone();
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
            // A `drop` still ends the owner after an earlier consume: copy
            // insertion gives the earlier use the copy, so a use after
            // `a = eat(x); c = drop(x)` is refused.
            Some(BindingState::Consumed(_)) if site.terminal => scope.consume_id(target, site),
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
        // A structural consume of an alias is recorded on its source
        // generation. Check that generation so reads through either
        // name observe the same consumed state.
        if self.reject_use_after_component_drop(name, expr, scope) {
            return;
        }
        let Some(use_id) = scope.top_id(name) else {
            return;
        };
        let resolved = scope.resolve_alias_chain(use_id).unwrap_or(use_id);
        let Some(BindingState::Consumed(site)) = scope.state(resolved) else {
            return;
        };
        // Binding an alias shares the value and permits later borrows.
        // A structural consume invalidates later borrows. The site
        // description below is used only to explain the error.
        if matches!(site.kind, ConsumeKind::Aliasing) {
            return;
        }
        let description = site.description.clone();
        // Only an extent read reaches here with a key ([04-LIN-9]); no copy
        // repairs it, so the diagnostic says where the extent read belongs.
        if self.expr_holds_key(expr, scope) {
            self.key_reuse(
                expr,
                format!(
                    "the extent of key-carrying variable `{name}` is read {} after its key was \
                     consumed by {description}; read a key tensor's extent before its one use \
                     ([04-LIN-9])",
                    diag_site(expr)
                ),
            );
            return;
        }
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
        self.expr_type(expr, scope).is_some_and(|ty| {
            type_expr_is_owned_linear(ty, &self.tensor_carrying_adts)
                || (self.type_holds_key(ty) && !type_expr_is_ref(ty))
        })
    }

    /// [04-LIN-9]: whether `ty` has key evidence: a key, a key tensor, or a
    /// tuple, reference or data type that carries one (spec/04 section 8.4.1).
    fn type_holds_key(&self, ty: &Expr) -> bool {
        type_expr_holds_key(ty, &self.key_carrying_adts)
    }

    fn expr_holds_key(&self, expr: &Expr, scope: &LinearScope) -> bool {
        self.expr_type(expr, scope)
            .is_some_and(|ty| self.type_holds_key(ty))
    }

    /// The type a `let` binds. A tuple literal and a block carry no `:type`
    /// stamp (`infer::annotate::should_attach_type_metadata`), so their types
    /// are rebuilt from their parts here; without this a tuple or block that
    /// carries a key would bind a name whose uses [04-LIN-9] cannot see.
    fn value_type(&self, value: &Expr, scope: &LinearScope) -> Option<Expr> {
        if let Some(ty) = self.expr_type(value, scope) {
            return Some(ty.clone());
        }
        if let Some(elements) = tagged_children(value, DeepTag::Tuple) {
            let element_types = elements
                .iter()
                .map(|element| self.value_type(element, scope))
                .collect::<Option<Vec<_>>>()?;
            return Some(Expr::node(
                DeepTag::TTuple,
                chelis_deep::ast::Metadata::default(),
                element_types,
                value.span(),
            ));
        }
        let (binds, body) = match tagged_children(value, DeepTag::Let) {
            Some([binds, body]) => (binds, body),
            _ => return None,
        };
        let mut inner = scope.clone();
        if let Some(bind_kids) = tagged_children(binds, DeepTag::Bind) {
            for pair in bind_kids.chunks(2) {
                if let [name, bound] = pair
                    && let Some(name) = symbol_name(name)
                {
                    let ty = self.value_type(bound, &inner);
                    inner.declare(name, ty);
                }
            }
        }
        self.value_type(body, &inner)
    }

    fn key_reuse(&mut self, expr: &Expr, message: String) {
        self.push_diagnostic(CheckError::new(
            CheckErrorKind::KeyReuse,
            with_macro_provenance(expr, message),
            vec![KEY_REUSE_SUGGESTION.to_string()],
        ));
    }

    /// [04-LIN-9]: a key holder has no read that leaves it live.
    fn reject_key_read(&mut self, expr: &Expr, how: &str) {
        let name = var_name(expr).unwrap_or("<expression>");
        self.key_reuse(
            expr,
            format!(
                "key-carrying variable `{name}` cannot be {how} {}: a key is used at most once and \
                 has no read that leaves it live ([04-LIN-9])",
                diag_site(expr)
            ),
        );
    }

    /// [04-LIN-9]: a consuming use of a key holder. Every consume of a key
    /// holder is structural, and a second one on any path is a reuse: no
    /// compiler-inserted copy applies.
    fn consume_key_holder(
        &mut self,
        expr: &Expr,
        name: &str,
        target: BindingId,
        scope: &mut LinearScope,
        site: ConsumeSite,
    ) {
        let site = ConsumeSite {
            kind: ConsumeKind::Structural,
            ..site
        };
        match scope.state(target) {
            Some(BindingState::Live { .. }) => {
                let moved = scope
                    .record(target)
                    .map(|record| record.moved_key_components.clone())
                    .unwrap_or_default();
                if !moved.is_empty() {
                    let positions = moved
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.key_reuse(
                        expr,
                        format!(
                            "key-carrying variable `{name}` already had its key component at \
                             position {positions} taken by `tuple_get`; using it whole {} would \
                             use that key again ([04-LIN-9])",
                            diag_site(expr)
                        ),
                    );
                }
                scope.consume_id(target, site);
            }
            Some(BindingState::Consumed(consumed_at)) => {
                let description = consumed_at.description.clone();
                self.key_reuse(
                    expr,
                    format!(
                        "key-carrying variable `{name}` was already consumed by {description}; a \
                         key is used at most once on every path, so the later use {} is invalid \
                         ([04-LIN-9])",
                        diag_site(expr)
                    ),
                );
            }
            None => {}
        }
    }

    /// [04-LIN-9]: `tuple-get(p, i)` on a key holder moves component `i` out
    /// of `p` when that component carries a key, and reads it otherwise. A
    /// destructuring `let` is this projection once per component.
    fn project_key_holder(&mut self, target: &Expr, index: Option<&Expr>, scope: &mut LinearScope) {
        let Some(name) = var_name(target) else {
            return;
        };
        let position = index.and_then(literal_index);
        let component_holds_key = match (position, self.expr_type(target, scope)) {
            (Some(position), Some(ty)) => tagged_children(ty, DeepTag::TTuple)
                .and_then(|elements| elements.get(position))
                .map(|element| self.type_holds_key(element)),
            _ => None,
        };
        let (Some(position), Some(component_holds_key)) = (position, component_holds_key) else {
            // No literal position or no readable tuple type: take the whole
            // holder, which is the conservative reading of a projection.
            self.consume_var_expr(target, scope, generic_site(target));
            return;
        };
        if !component_holds_key {
            self.read_var_expr(target, scope);
            return;
        }
        let Some(use_id) = scope.top_id(name) else {
            return;
        };
        match scope.state(use_id) {
            Some(BindingState::Consumed(consumed_at)) => {
                let description = consumed_at.description.clone();
                self.key_reuse(
                    target,
                    format!(
                        "key-carrying variable `{name}` was already consumed by {description}; \
                         taking its key component at position {position} {} would use that key \
                         again ([04-LIN-9])",
                        diag_site(target)
                    ),
                );
            }
            Some(BindingState::Live { .. }) => {
                let already_moved = scope
                    .record(use_id)
                    .is_some_and(|record| record.moved_key_components.contains(&position));
                if already_moved {
                    self.key_reuse(
                        target,
                        format!(
                            "the key component at position {position} of `{name}` was already \
                             taken; taking it again {} would use that key twice ([04-LIN-9])",
                            diag_site(target)
                        ),
                    );
                } else if let Some(record) = scope.record_mut(use_id) {
                    record.moved_key_components.insert(position);
                }
            }
            None => {}
        }
    }

    /// [04-LIN-9]: a signature never borrows a key holder, so `&key` or
    /// `&(key, ..)` is rejected where a parameter declares it.
    fn reject_borrowed_key_parameter(&mut self, expr: &Expr, name: &str, ty: &Expr) {
        if type_expr_is_ref(ty) && self.type_holds_key(ty) {
            self.push_diagnostic(CheckError::new(
                CheckErrorKind::KeyReuse,
                with_macro_provenance(
                    expr,
                    format!(
                        "parameter `{name}` {} borrows a key-carrying type; a key is never \
                         borrowed, so a parameter takes it owned ([04-LIN-9])",
                        diag_site(expr)
                    ),
                ),
                vec![format!("Drop the `&` from `{name}`'s type")],
            ));
        }
    }

    /// Whether an operand's value carries a key ([04-LIN-9]). A borrow is
    /// read through to its target, whose type carries the evidence.
    fn operand_holds_key(&self, operand: &Expr, scope: &LinearScope) -> bool {
        let operand = borrow_inner(operand).unwrap_or(operand);
        self.value_type(operand, scope)
            .is_some_and(|ty| self.type_holds_key(&ty))
    }

    /// Whether `name` resolves to a builtin here: a builtin name that no
    /// binding in scope shadows.
    fn is_builtin_reference(&self, name: &str, scope: &LinearScope) -> bool {
        BUILTIN_NAMES.contains(&name)
            && builtin_decl(name).is_some()
            && scope.top_id(name).is_none()
    }

    /// [04-LIN-9] and spec/04 section 1.1 at a Deep tag the allow-list
    /// refuses (`tag_keys`): every runtime operand whose value carries a key
    /// is refused, naming the tag.
    fn refuse_key_children(
        &mut self,
        tag: DeepTag,
        children: &[Expr],
        refusal: KeyRefusal,
        scope: &LinearScope,
    ) {
        for (index, child) in children.iter().enumerate() {
            if child_stamp_role(tag, index, children.len()) == ChildStampRole::RuntimeExpr
                && self.operand_holds_key(child, scope)
            {
                self.refuse_key_operand(tag.as_str(), index, child, refusal);
            }
        }
    }

    /// [04-LIN-9] and spec/04 section 1.1 at an application: each operand
    /// whose value carries a key must reach an operation the allow-list
    /// admits (`crate::key_admission`). A builtin decides per operand
    /// (`builtin_key_operand`); every other callee (a user function, a
    /// closure, a constructor, or a `grad`, `vmap` or `jit` application)
    /// takes it through a key-typed parameter, whose type the type checker
    /// fixed ([04-LIN-10]). A borrow of a key is refused whatever the callee.
    /// Returns, per operand, whether it was refused, so no later rule reports
    /// the same operand again, or admitted as an extent read, which leaves
    /// the key live.
    fn admit_key_operands(
        &mut self,
        builtin: Option<&str>,
        call: &Expr,
        args: &[Expr],
        scope: &LinearScope,
    ) -> Vec<KeyOperand> {
        let cases = builtin
            .map(|name| self.selected_cases(name, args, scope))
            .unwrap_or_default();
        let mut verdicts = Vec::with_capacity(args.len());
        for (index, arg) in args.iter().enumerate() {
            let verdict = if borrow_inner(arg).is_some() {
                Err(KeyRefusal::Read)
            } else if let Some(name) = builtin {
                builtin_key_operand(name, &cases, index)
            } else {
                Ok(KeyAdmission::KeyParameter)
            };
            let holds_key = match self.value_type(borrow_inner(arg).unwrap_or(arg), scope) {
                Some(ty) => self.type_holds_key(&ty),
                // A parameter routed to the callback and the result is in
                // the result, so the call's own type decides when the
                // operand's is unreadable.
                None => {
                    matches!(verdict, Err(KeyRefusal::CallbackAndResult { .. }))
                        && self.expr_holds_key(call, scope)
                }
            };
            let refuse = holds_key && verdict.is_err();
            if let (true, Err(refusal)) = (refuse, verdict) {
                let operation = match (borrow_inner(arg), builtin) {
                    (Some(_), _) => DeepTag::Borrow.as_str(),
                    (None, Some(name)) => name,
                    (None, None) => "",
                };
                self.refuse_key_operand(operation, index, arg, refusal);
            }
            verdicts.push(match verdict {
                _ if refuse => KeyOperand::Refused,
                Ok(KeyAdmission::ExtentObservation) => KeyOperand::ExtentRead,
                _ => KeyOperand::Consumed,
            });
        }
        verdicts
    }

    /// The one diagnostic for a key-carrying operand an operation refuses
    /// (spec/04 section 1.1, [04-LIN-9]). It names the operation; `operation`
    /// is empty only for a call whose callee has no name.
    fn refuse_key_operand(
        &mut self,
        operation: &str,
        index: usize,
        operand: &Expr,
        refusal: KeyRefusal,
    ) {
        match refusal {
            KeyRefusal::NotNamed => self.push_diagnostic(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                with_macro_provenance(
                    operand,
                    format!(
                        "`{operation}` does not admit a key-carrying operand at argument {index} \
                         {}: a key-carrying value reaches only a key operation, a random draw, \
                         `drop`, a branch's join, a tuple, record or data constructor or \
                         pattern, a builtin that routes each value to one consumer, or a \
                         parameter whose declared type carries a key \
                         (spec/04-type-system.md section 1.1; [04-LIN-9])",
                        diag_site(operand)
                    ),
                ),
                vec![KEY_OPERATION_SUGGESTION.to_string()],
            )),
            KeyRefusal::Read => {
                let how = match operation {
                    "borrow" => "borrowed".to_string(),
                    "copy" => "copied".to_string(),
                    "" => "borrowed by this call".to_string(),
                    name => format!("borrowed by `{name}`"),
                };
                let target = borrow_inner(operand).unwrap_or(operand);
                self.reject_key_read(target, &how);
            }
            KeyRefusal::CallbackAndResult { parameter } => self.key_reuse(
                operand,
                format!(
                    "`{operation}` {} passes each value of its type parameter `{parameter}` to \
                     its callback and also keeps it in its result, so a key-carrying \
                     instantiation would use each key twice ([04-LIN-9])",
                    diag_site(operand)
                ),
            ),
        }
    }

    /// The sibling cases a call to `name` selects: for `concat`, whose two
    /// cases differ, the one its second operand selects ([05-OP-54],
    /// [05-OP-62]); otherwise every case, each of which must admit an
    /// operand for the builtin to admit it.
    fn selected_cases(
        &self,
        name: &str,
        args: &[Expr],
        scope: &LinearScope,
    ) -> Vec<BuiltinSiblingCaseId> {
        let Some(decl) = builtin_decl(name) else {
            return Vec::new();
        };
        let cases: Vec<BuiltinSiblingCaseId> = decl
            .capability
            .sibling_cases
            .iter()
            .map(|case| case.case)
            .collect();
        if name == "concat" && cases.len() > 1 {
            let axis = args
                .get(1)
                .and_then(|arg| self.value_type(arg, scope))
                .is_some_and(|ty| {
                    tagged_children(&ty, DeepTag::TPrim)
                        .and_then(|kids| kids.first())
                        .and_then(symbol_name)
                        == Some("i32")
                });
            return vec![if axis {
                BuiltinSiblingCaseId::ConcatTensors
            } else {
                BuiltinSiblingCaseId::ConcatList
            }];
        }
        cases
    }

    /// [04-LIN-9] and spec/06 section 3.6: `vmap` maps tensor arguments and
    /// broadcasts every other argument to all rows, so a key-carrying
    /// argument that is not a tensor would be used by every row. Keys reach
    /// the rows only as a mapped `tensor[n, key]`.
    fn reject_broadcast_keys(&mut self, args: &[Expr], scope: &LinearScope) {
        for (index, arg) in args.iter().enumerate() {
            let operand = borrow_inner(arg).unwrap_or(arg);
            let Some(ty) = self.value_type(operand, scope) else {
                continue;
            };
            if !self.type_holds_key(&ty) || get_tag_expr(&ty) == Some(DeepTag::TTensor) {
                continue;
            }
            self.push_diagnostic(CheckError::new(
                CheckErrorKind::KeyReuse,
                with_macro_provenance(
                    arg,
                    format!(
                        "`vmap` broadcasts argument {index} {}, which carries a key and is not a \
                         tensor, to every row, so each row would use its keys again ([04-LIN-9]; \
                         spec/06-transformations.md section 3.6)",
                        diag_site(arg)
                    ),
                ),
                vec![
                    "Map keys instead of broadcasting them: give the function a `key` formal and \
                     pass `split_keys(k, n)`, one key per row"
                        .to_string(),
                ],
            ));
        }
    }

    fn expr_is_owned_or_borrow_linear(&self, expr: &Expr, scope: &LinearScope) -> bool {
        self.expr_type(expr, scope).is_some_and(|ty| {
            type_expr_proves_tensor(ty, &self.tensor_carrying_adts)
                || type_expr_is_unresolved_tvar(ty)
        })
    }
}

fn with_macro_provenance(expr: &Expr, message: String) -> String {
    let Some(source) = macro_source(expr) else {
        return message;
    };
    format!("{message} (in expansion of {source})")
}

fn macro_source(expr: &Expr) -> Option<String> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, metadata, _) => {
            Some(chelis_deep::printer::print_macro_source(metadata.source()?))
        }
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

fn decoded_shape_is_valid(tag: DeepTag, child_count: usize) -> bool {
    match arity_contract(tag) {
        AritySpec::Fixed(expected) => child_count == expected,
        AritySpec::AtLeast(minimum) => child_count >= minimum,
        AritySpec::Range(minimum, maximum) => child_count >= minimum && child_count <= maximum,
    }
}

fn is_runtime_expression_tag(tag: DeepTag) -> bool {
    matches!(
        tag,
        DeepTag::Fn
            | DeepTag::App
            | DeepTag::Let
            | DeepTag::Match
            | DeepTag::If
            | DeepTag::Var
            | DeepTag::Lit
            | DeepTag::Record
            | DeepTag::Access
            | DeepTag::Pipe
            | DeepTag::Block
            | DeepTag::Tuple
            | DeepTag::TupleGet
            | DeepTag::RecordUpdate
            | DeepTag::Par
            | DeepTag::HandleEffect
            | DeepTag::Borrow
            | DeepTag::Grad
            | DeepTag::Vmap
            | DeepTag::Jit
            | DeepTag::Realize
            | DeepTag::Cast
            | DeepTag::Copy
            | DeepTag::Quote
            | DeepTag::Unquote
            | DeepTag::Splice
    )
}

fn is_absent_match_guard(expr: &Expr) -> bool {
    match expr.carrier() {
        ExprCarrier::StructuralList([]) => true,
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => false,
    }
}

fn get_tag_expr(expr: &Expr) -> Option<DeepTag> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children)
            if decoded_shape_is_valid(tag, children.len()) =>
        {
            Some(tag)
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

fn tagged_children(expr: &Expr, expected: DeepTag) -> Option<&[Expr]> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children)
            if tag == expected && decoded_shape_is_valid(tag, children.len()) =>
        {
            Some(children)
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
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
        .filter_map(|param| param_name_and_type(param).map(|(name, _)| name.to_string()))
        .collect()
}

fn pattern_names(expr: &Expr) -> Vec<String> {
    let mut names = Vec::new();
    collect_pattern_names(expr, &mut names);
    names
}

fn collect_pattern_names(expr: &Expr, names: &mut Vec<String>) {
    let (tag, children) = match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => (tag, children),
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => return,
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = children.first().and_then(symbol_name) {
                names.push(name.to_string());
            }
        }
        DeepTag::PatAs => {
            if let Some(name) = children.first().and_then(symbol_name) {
                names.push(name.to_string());
            }
            if let Some(inner) = children.get(1) {
                collect_pattern_names(inner, names);
            }
        }
        _ => {
            for child in children {
                collect_pattern_names(child, names);
            }
        }
    }
}

/// Like [`pattern_names`], but also returns each binding's resolved
/// type expression when the inferencer stamped one onto the pattern
/// node's metadata. `check_match` populates arm `LinearScope` entries
/// with these types — required for destructured fields whose type comes
/// from the scrutinee's ADT instantiation rather than a `let`-style RHS
/// (closes #181) — through [`pattern_bindings_with_paths`], of which
/// this is the (name, type) projection.
#[cfg(test)]
fn pattern_named_types(expr: &Expr) -> Vec<(String, Option<Expr>)> {
    pattern_bindings_with_paths(expr)
        .into_iter()
        .map(|binding| (binding.name, binding.ty))
        .collect()
}

/// One step from a match scrutinee into the sub-value a pattern names.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PatternStep {
    /// A tuple position.
    Tuple(usize),
    /// A positional field of the named constructor.
    Positional(String, usize),
    /// A named field of the named constructor.
    Field(String, String),
}

/// A name a pattern binds, its checked type, and the scrutinee component it
/// names.
#[derive(Debug, Clone)]
struct PatternBinding {
    name: String,
    ty: Option<Expr>,
    path: Vec<PatternStep>,
}

fn decoded_parts(expr: &Expr) -> Option<(DeepTag, &[Expr])> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => Some((tag, children)),
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

/// Every binding `pattern` introduces, in source order, with the path from
/// the scrutinee to the component it names.
fn pattern_bindings_with_paths(pattern: &Expr) -> Vec<PatternBinding> {
    let mut bindings = Vec::new();
    collect_pattern_bindings_with_paths(pattern, &mut Vec::new(), &mut bindings);
    bindings
}

fn collect_pattern_bindings_with_paths(
    pattern: &Expr,
    path: &mut Vec<PatternStep>,
    bindings: &mut Vec<PatternBinding>,
) {
    let Some((tag, children)) = decoded_parts(pattern) else {
        return;
    };
    match tag {
        DeepTag::PatVar | DeepTag::PatAs => {
            if let Some(name) = children.first().and_then(symbol_name) {
                bindings.push(PatternBinding {
                    name: name.to_string(),
                    ty: type_metadata(pattern).cloned(),
                    path: path.clone(),
                });
            }
            if tag == DeepTag::PatAs
                && let Some(inner) = children.get(1)
            {
                collect_pattern_bindings_with_paths(inner, path, bindings);
            }
        }
        DeepTag::PatTuple => {
            for (index, child) in children.iter().enumerate() {
                descend_pattern(PatternStep::Tuple(index), child, path, bindings);
            }
        }
        DeepTag::PatCtor => {
            let ctor = children.first().and_then(symbol_name).unwrap_or_default();
            for (index, child) in children.iter().skip(1).enumerate() {
                descend_pattern(
                    PatternStep::Positional(ctor.to_string(), index),
                    child,
                    path,
                    bindings,
                );
            }
        }
        DeepTag::PatRecord => {
            let ctor = children.first().and_then(symbol_name).unwrap_or_default();
            for field in children.iter().skip(1) {
                if let Some([field_name, field_pattern]) = tagged_children(field, DeepTag::Kv)
                    && let Some(field_name) = symbol_name(field_name)
                {
                    descend_pattern(
                        PatternStep::Field(ctor.to_string(), field_name.to_string()),
                        field_pattern,
                        path,
                        bindings,
                    );
                } else {
                    collect_pattern_bindings_with_paths(field, path, bindings);
                }
            }
        }
        // Every other form binds nothing; a malformed node's children are
        // still walked so a binder under it keeps its (conservative) path.
        _ => {
            for child in children {
                collect_pattern_bindings_with_paths(child, path, bindings);
            }
        }
    }
}

fn descend_pattern(
    step: PatternStep,
    child: &Expr,
    path: &mut Vec<PatternStep>,
    bindings: &mut Vec<PatternBinding>,
) {
    path.push(step);
    collect_pattern_bindings_with_paths(child, path, bindings);
    path.pop();
}

/// Whether two components of one scrutinee can share storage: one path is a
/// prefix of the other, step by step. A value has one constructor, so two
/// different constructors, or two different positions or fields of one, are
/// disjoint; steps that cannot be compared (a positional and a named field
/// of one constructor) are treated as overlapping.
fn pattern_paths_overlap(lhs: &[PatternStep], rhs: &[PatternStep]) -> bool {
    for (left, right) in lhs.iter().zip(rhs) {
        let disjoint = match (left, right) {
            (PatternStep::Tuple(a), PatternStep::Tuple(b)) => a != b,
            (PatternStep::Positional(ca, a), PatternStep::Positional(cb, b)) => ca != cb || a != b,
            (PatternStep::Field(ca, a), PatternStep::Field(cb, b)) => ca != cb || a != b,
            (PatternStep::Positional(ca, _), PatternStep::Field(cb, _))
            | (PatternStep::Field(ca, _), PatternStep::Positional(cb, _)) => ca != cb,
            (PatternStep::Tuple(_), _) | (_, PatternStep::Tuple(_)) => false,
        };
        if disjoint {
            return false;
        }
    }
    true
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

/// Runtime variable references not bound inside the expression, in stable order.
/// Type/effect metadata is not executable code. This is the same lexical
/// binding analysis used to identify captures for ownership checking.
pub fn free_runtime_variables(expr: &Expr) -> Vec<String> {
    free_vars(expr, &[])
}

/// `expr` with each free runtime variable in `renames` respelled, under the
/// binding rules [`free_runtime_variables`] reads (chelis#2619). A reference
/// a binder inside `expr` captures keeps its spelling, so the two agree on
/// which occurrences are free by construction.
pub fn rename_free_runtime_variables(expr: &Expr, renames: &BTreeMap<String, String>) -> Expr {
    if renames.is_empty() {
        return expr.clone();
    }
    rename_free_vars(expr, renames, &mut Vec::new())
}

fn rename_free_vars(
    expr: &Expr,
    renames: &BTreeMap<String, String>,
    bound: &mut Vec<UnordSet<String>>,
) -> Expr {
    let children = |children: &[Expr], bound: &mut Vec<UnordSet<String>>| -> Vec<Expr> {
        children
            .iter()
            .map(|child| rename_free_vars(child, renames, bound))
            .collect()
    };
    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => expr.clone(),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                metadata: meta.metadata.clone(),
                expr: Box::new(rename_free_vars(&meta.expr, renames, bound)),
            },
            *span,
        ),
        Expr::BareList(items, span) => Expr::BareList(children(items, bound), *span),
        Expr::UnknownForm(data) => {
            let mut renamed = data.as_ref().clone();
            renamed.children = children(&data.children, bound);
            Expr::UnknownForm(Box::new(renamed))
        }
        Expr::Node(node, span) => {
            let tag = node.tag();
            let kids = node.children_slice();
            let rebuilt = |renamed: Vec<Expr>| Expr::node(tag, node.meta().clone(), renamed, *span);
            if !is_runtime_expression_tag(tag) || !decoded_shape_is_valid(tag, kids.len()) {
                return rebuilt(children(kids, bound));
            }
            match tag {
                DeepTag::Var => {
                    let respelled = kids
                        .first()
                        .and_then(symbol_name)
                        .filter(|name| !bound.iter().rev().any(|scope| scope.contains(*name)))
                        .and_then(|name| renames.get(name));
                    match respelled {
                        Some(to) => {
                            let mut renamed = kids.to_vec();
                            renamed[0] = Expr::Atom(Atom::Name(to.clone()), kids[0].span());
                            rebuilt(renamed)
                        }
                        None => expr.clone(),
                    }
                }
                DeepTag::Fn if kids.len() >= 2 => {
                    bound.push(param_names(&kids[0]).into_iter().collect());
                    let mut renamed = kids.to_vec();
                    renamed[1] = rename_free_vars(&kids[1], renames, bound);
                    bound.pop();
                    rebuilt(renamed)
                }
                DeepTag::Let if kids.len() >= 2 => {
                    let mut let_scope = UnordSet::new();
                    let mut renamed = kids.to_vec();
                    if let Some(bind_kids) = tagged_children(&kids[0], DeepTag::Bind) {
                        let mut binds = bind_kids.to_vec();
                        let mut index = 0;
                        while index + 1 < binds.len() {
                            binds[index + 1] =
                                rename_free_vars(&bind_kids[index + 1], renames, bound);
                            if let Some(name) = symbol_name(&bind_kids[index]) {
                                let_scope.insert(name.to_string());
                            }
                            index += 2;
                        }
                        if let Expr::Node(bind, bind_span) = &kids[0] {
                            renamed[0] =
                                Expr::node(bind.tag(), bind.meta().clone(), binds, *bind_span);
                        }
                    }
                    bound.push(let_scope);
                    renamed[1] = rename_free_vars(&kids[1], renames, bound);
                    bound.pop();
                    rebuilt(renamed)
                }
                DeepTag::Match if !kids.is_empty() => {
                    let mut renamed = Vec::with_capacity(kids.len());
                    renamed.push(rename_free_vars(&kids[0], renames, bound));
                    for arm in kids.iter().skip(1) {
                        let (Some(arm_kids), Expr::Node(arm_node, arm_span)) =
                            (tagged_children(arm, DeepTag::Arm), arm)
                        else {
                            renamed.push(rename_free_vars(arm, renames, bound));
                            continue;
                        };
                        bound.push(pattern_names(&arm_kids[0]).into_iter().collect());
                        let mut arm_renamed = arm_kids.to_vec();
                        arm_renamed[1] = rename_free_vars(&arm_kids[1], renames, bound);
                        arm_renamed[2] = rename_free_vars(&arm_kids[2], renames, bound);
                        bound.pop();
                        renamed.push(Expr::node(
                            arm_node.tag(),
                            arm_node.meta().clone(),
                            arm_renamed,
                            *arm_span,
                        ));
                    }
                    rebuilt(renamed)
                }
                _ => rebuilt(children(kids, bound)),
            }
        }
    }
}

fn collect_free_vars(expr: &Expr, bound: &mut Vec<UnordSet<String>>, free: &mut UnordSet<String>) {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => {
            if !is_runtime_expression_tag(tag) || !decoded_shape_is_valid(tag, children.len()) {
                for child in children {
                    collect_untrusted_free_vars(child, bound, free);
                }
                return;
            }
            match tag {
                DeepTag::Var => {
                    if let Some(name) = children.first().and_then(symbol_name)
                        && !bound.iter().rev().any(|scope| scope.contains(name))
                    {
                        free.insert(name.to_string());
                    }
                }
                DeepTag::Fn => {
                    if children.len() >= 2 {
                        bound.push(param_names(&children[0]).into_iter().collect());
                        collect_free_vars(&children[1], bound, free);
                        bound.pop();
                    }
                }
                DeepTag::Let => {
                    if children.len() < 2 {
                        return;
                    }
                    let mut let_scope = UnordSet::new();
                    if let Some(bind_kids) = tagged_children(&children[0], DeepTag::Bind) {
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
                    collect_free_vars(&children[1], bound, free);
                    bound.pop();
                }
                DeepTag::Match => {
                    if children.is_empty() {
                        return;
                    }
                    collect_free_vars(&children[0], bound, free);
                    for arm in children.iter().skip(1) {
                        let Some(arm_kids) = tagged_children(arm, DeepTag::Arm) else {
                            collect_untrusted_free_vars(arm, bound, free);
                            continue;
                        };
                        bound.push(pattern_names(&arm_kids[0]).into_iter().collect());
                        collect_free_vars(&arm_kids[1], bound, free);
                        collect_free_vars(&arm_kids[2], bound, free);
                        bound.pop();
                    }
                }
                _ => {
                    for child in children {
                        collect_free_vars(child, bound, free);
                    }
                }
            }
        }
        ExprCarrier::StructuralList(elements) => {
            for element in elements {
                collect_free_vars(element, bound, free);
            }
        }
        ExprCarrier::UndecodableHead(_, _, children) => {
            for child in children {
                collect_free_vars(child, bound, free);
            }
        }
        ExprCarrier::Atom(_) | ExprCarrier::MetadataMap(_) => {}
        ExprCarrier::MetadataExpression(meta) => collect_free_vars(&meta.expr, bound, free),
    }
}

fn collect_untrusted_free_vars(
    expr: &Expr,
    bound: &mut Vec<UnordSet<String>>,
    free: &mut UnordSet<String>,
) {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children)
            if is_runtime_expression_tag(tag) && decoded_shape_is_valid(tag, children.len()) =>
        {
            collect_free_vars(expr, bound, free);
        }
        ExprCarrier::DecodedNode(_, _, children)
        | ExprCarrier::StructuralList(children)
        | ExprCarrier::UndecodableHead(_, _, children) => {
            for child in children {
                collect_untrusted_free_vars(child, bound, free);
            }
        }
        ExprCarrier::Atom(_) => {}
        ExprCarrier::MetadataMap(map) => {
            map.visit_syntax(&mut |_, value| {
                collect_untrusted_free_vars(value, bound, free);
            });
        }
        ExprCarrier::MetadataExpression(meta) => {
            collect_untrusted_free_vars(&meta.expr, bound, free);
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
    if COMPARISON_OPS.contains(&name) {
        return arg_index < 2;
    }
    matches!(
        name,
        "add"
            | "mul"
            | "max_elem"
            | "min_elem"
            | "sub"
            | "div"
            | "floor_div"
            | "trunc_div"
            | "mod"
            | "bitand"
            | "bitor"
            | "bitxor"
            | "shl"
            | "shr"
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
                | "not"
                | "relu"
                | "sigmoid"
                | "tanh"
                | "erf"
                | "erfc"
                | "silu"
                | "gelu"
                | "gelu_tanh"
                | "standard_normal_cdf"
                | "softmax"
                | "mean"
                | "sum"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
                | "reshape"
                | "permute"
                | "expand"
                | "insert"
                | "pad"
                | "shrink"
                | "stride"
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
        ) | ("conv", 0 | 1)
        // [05-OP-8] and [05-OP-37] consume their key (operand 0) and borrow
        // the tensor they read (operand 1).
        | ("uniform_like" | "dropout", 1)
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
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, metadata, _) => metadata.ty().map(|v| v.expression()),
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
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
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, metadata, _) => metadata.span_id().map(|v| v.value()),
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

fn param_name_and_type(param: &Expr) -> Option<(&str, Option<&Expr>)> {
    match param.carrier() {
        ExprCarrier::Atom(Atom::Name(name)) => Some((name.as_str(), None)),
        ExprCarrier::StructuralList(elements) => {
            let name = elements.first().and_then(symbol_name)?;
            let ty = elements.get(1).and_then(|meta| {
                let Expr::Map(meta, _) = meta else {
                    return None;
                };
                meta.ty().map(|v| v.expression())
            });
            Some((name, ty))
        }
        ExprCarrier::UndecodableHead(name, metadata, []) => {
            Some((name, metadata.ty().map(|value| value.expression())))
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
        ExprCarrier::MetadataExpression(meta) => {
            let Expr::Atom(Atom::Name(name), _) = meta.expr.as_ref() else {
                return None;
            };
            Some((name.as_str(), meta.metadata.ty().map(|v| v.expression())))
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_) => None,
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
fn bind_introduces_destructure_tmp(bind_expr: &Expr) -> bool {
    match bind_expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Bind, metadata, _) => metadata.destructure().is_some(),
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorEvidence {
    Contains,
    Absent,
    Unreadable,
}

fn type_syntax_is_well_formed(expr: &Expr) -> bool {
    chelis_deep::annotations::TypeSyntax::try_new(expr.clone()).is_ok()
}

fn tensor_evidence(expr: &Expr, tensor_carrying_adts: &UnordSet<String>) -> TensorEvidence {
    if !type_syntax_is_well_formed(expr) {
        return TensorEvidence::Unreadable;
    }
    tensor_evidence_from_well_formed_type(expr, tensor_carrying_adts)
}

fn tensor_evidence_from_well_formed_type(
    expr: &Expr,
    tensor_carrying_adts: &UnordSet<String>,
) -> TensorEvidence {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => match tag {
            DeepTag::TTensor => TensorEvidence::Contains,
            DeepTag::TRef | DeepTag::TTuple => {
                combine_tensor_evidence(children.iter().map(|child| {
                    tensor_evidence_from_well_formed_type(child, tensor_carrying_adts)
                }))
            }
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
                match combine_tensor_evidence(
                    children
                        .iter()
                        .skip(1) // skip the name; only check type args
                        .map(|child| {
                            tensor_evidence_from_well_formed_type(child, tensor_carrying_adts)
                        }),
                ) {
                    TensorEvidence::Unreadable => TensorEvidence::Unreadable,
                    TensorEvidence::Contains => TensorEvidence::Contains,
                    TensorEvidence::Absent if name_carries => TensorEvidence::Contains,
                    TensorEvidence::Absent => TensorEvidence::Absent,
                }
            }
            DeepTag::TFn => TensorEvidence::Absent,
            _ => TensorEvidence::Absent,
        },
        ExprCarrier::MetadataExpression(meta) => {
            tensor_evidence_from_well_formed_type(&meta.expr, tensor_carrying_adts)
        }
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_) => TensorEvidence::Unreadable,
    }
}

fn combine_tensor_evidence(evidence: impl IntoIterator<Item = TensorEvidence>) -> TensorEvidence {
    let mut unreadable = false;
    let mut contains = false;
    for item in evidence {
        match item {
            TensorEvidence::Contains => contains = true,
            TensorEvidence::Unreadable => unreadable = true,
            TensorEvidence::Absent => {}
        }
    }
    if unreadable {
        TensorEvidence::Unreadable
    } else if contains {
        TensorEvidence::Contains
    } else {
        TensorEvidence::Absent
    }
}

/// Ownership tracking must retain a possible tensor when its type evidence is
/// unreadable, or a malformed carrier could erase a required consume.
fn type_expr_may_contain_tensor(expr: &Expr, tensor_carrying_adts: &UnordSet<String>) -> bool {
    !matches!(
        tensor_evidence(expr, tensor_carrying_adts),
        TensorEvidence::Absent
    )
}

/// Borrowing needs affirmative type evidence. An unreadable carrier is not
/// permission to borrow; the caller turns this `false` into `InvalidBorrow`.
fn type_expr_proves_tensor(expr: &Expr, tensor_carrying_adts: &UnordSet<String>) -> bool {
    matches!(
        tensor_evidence(expr, tensor_carrying_adts),
        TensorEvidence::Contains
    )
}

fn type_expr_is_ref(expr: &Expr) -> bool {
    type_syntax_is_well_formed(expr) && matches!(get_tag_expr(expr), Some(DeepTag::TRef))
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
/// or carrier (e.g. `&i32` against a non-borrow consumer) is
/// rejected by the inference-layer `borrow` arm before reaching
/// linearity (the `_ => TypeMismatch` arm fires for `Type::Prim`,
/// `Type::Unit`, `Type::Fn`, etc.), so this leniency cannot leak.
fn type_expr_is_unresolved_tvar(expr: &Expr) -> bool {
    if !type_syntax_is_well_formed(expr) {
        return false;
    }
    type_expr_is_unresolved_tvar_from_well_formed_type(expr)
}

fn type_expr_is_unresolved_tvar_from_well_formed_type(expr: &Expr) -> bool {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::TVar, _, _) => true,
        ExprCarrier::DecodedNode(DeepTag::TRef, _, children) => children
            .first()
            .is_some_and(type_expr_is_unresolved_tvar_from_well_formed_type),
        ExprCarrier::DecodedNode(_, _, _) => false,
        ExprCarrier::MetadataExpression(meta) => {
            type_expr_is_unresolved_tvar_from_well_formed_type(&meta.expr)
        }
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_) => false,
    }
}

fn type_expr_is_owned_linear(expr: &Expr, tensor_carrying_adts: &UnordSet<String>) -> bool {
    match tensor_evidence(expr, tensor_carrying_adts) {
        TensorEvidence::Absent => false,
        // Unreadable evidence remains conservatively owned and cannot
        // authorize the explicit-reference exemption.
        TensorEvidence::Unreadable => true,
        TensorEvidence::Contains => !matches!(get_tag_expr(expr), Some(DeepTag::TRef)),
    }
}

/// [04-LIN-9]: the one suggestion every key-reuse diagnostic carries. It
/// names the derivations, never `copy`, because a key is never copied.
const KEY_REUSE_SUGGESTION: &str = "Keys are single-use: derive a fresh key for each use with \
     `split_key(k)` or `split_keys(k, n)` instead of reusing `k`";

/// spec/04 section 1.1: the suggestion of an operation that refuses a key.
const KEY_OPERATION_SUGGESTION: &str =
    "Keys only feed `split_key`, `split_keys`, `fold_in` and random draws";

/// Spec/04 section 8.4.1 key evidence: `Contains` for a `key`, a tensor
/// whose element dtype is `key`, and any tuple, reference or data type that
/// carries one. Function types carry none, like tensor evidence (section
/// 8.4). A type variable carries none here; its instantiation is checked
/// where it is instantiated.
fn key_evidence(expr: &Expr, key_carrying_adts: &UnordSet<String>) -> TensorEvidence {
    if !type_syntax_is_well_formed(expr) {
        return TensorEvidence::Unreadable;
    }
    key_evidence_from_well_formed_type(expr, key_carrying_adts)
}

fn key_evidence_from_well_formed_type(
    expr: &Expr,
    key_carrying_adts: &UnordSet<String>,
) -> TensorEvidence {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => match tag {
            DeepTag::TPrim => {
                if children.first().and_then(symbol_name) == Some("key") {
                    TensorEvidence::Contains
                } else {
                    TensorEvidence::Absent
                }
            }
            // The element dtype is the tensor type's last child.
            DeepTag::TTensor => children.last().map_or(TensorEvidence::Absent, |precision| {
                key_evidence_from_well_formed_type(precision, key_carrying_adts)
            }),
            DeepTag::TRef | DeepTag::TTuple => combine_tensor_evidence(
                children
                    .iter()
                    .map(|child| key_evidence_from_well_formed_type(child, key_carrying_adts)),
            ),
            DeepTag::TAdt => {
                let name_carries = children
                    .first()
                    .and_then(symbol_name)
                    .is_some_and(|name| key_carrying_adts.contains(name));
                match combine_tensor_evidence(
                    children
                        .iter()
                        .skip(1)
                        .map(|child| key_evidence_from_well_formed_type(child, key_carrying_adts)),
                ) {
                    TensorEvidence::Absent if name_carries => TensorEvidence::Contains,
                    evidence => evidence,
                }
            }
            _ => TensorEvidence::Absent,
        },
        ExprCarrier::MetadataExpression(meta) => {
            key_evidence_from_well_formed_type(&meta.expr, key_carrying_adts)
        }
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_) => TensorEvidence::Unreadable,
    }
}

/// [04-LIN-9]: affirmative key evidence. An unreadable type keeps its
/// conservative tensor ownership but is not classified as a key holder.
fn type_expr_holds_key(expr: &Expr, key_carrying_adts: &UnordSet<String>) -> bool {
    matches!(
        key_evidence(expr, key_carrying_adts),
        TensorEvidence::Contains
    )
}

/// The literal position of a `tuple-get` selector, when it is one.
/// [04-LIN-11]: one step of a projection chain, `.i` or `.f`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ProjectionStep {
    Index(usize),
    Field(String),
}

/// The variable a chain of tuple projections and field accesses is rooted at,
/// and the chain's steps from the root outward. A bare variable is the empty
/// chain; any other root is no chain.
fn projection_chain(expr: &Expr) -> Option<(&Expr, Vec<ProjectionStep>)> {
    if is_var_expr(expr) {
        return Some((expr, Vec::new()));
    }
    let (step, target) = if let Some([target, index]) = tagged_children(expr, DeepTag::TupleGet) {
        (ProjectionStep::Index(literal_index(index)?), target)
    } else if let Some([target, field]) = tagged_children(expr, DeepTag::Access) {
        let name = match field {
            Expr::Atom(Atom::Name(name), _) => name.to_string(),
            other => symbol_name(other)?.to_string(),
        };
        (ProjectionStep::Field(name), target)
    } else {
        return None;
    };
    let (root, mut path) = projection_chain(target)?;
    path.push(step);
    Some((root, path))
}

/// Two projection paths overlap when one is a prefix of the other: the empty
/// path, the owner whole, overlaps every component.
fn projection_paths_overlap(lhs: &[ProjectionStep], rhs: &[ProjectionStep]) -> bool {
    lhs.iter().zip(rhs).all(|(left, right)| left == right)
}

/// The first dropped component, among the records of `ids`, that overlaps
/// `path`, with its `drop` site.
fn dropped_component_overlapping(
    scope: &LinearScope,
    ids: &[BindingId],
    path: &[ProjectionStep],
) -> Option<(Vec<ProjectionStep>, String)> {
    ids.iter()
        .filter_map(|id| scope.record(*id))
        .flat_map(|record| record.dropped_components.iter())
        .find(|(dropped, _)| projection_paths_overlap(dropped, path))
        .cloned()
}

fn render_projection(path: &[ProjectionStep]) -> String {
    path.iter()
        .map(|step| match step {
            ProjectionStep::Index(index) => format!(".{index}"),
            ProjectionStep::Field(field) => format!(".{field}"),
        })
        .collect()
}

fn literal_index(expr: &Expr) -> Option<usize> {
    let literal = tagged_children(expr, DeepTag::Lit)
        .and_then(|children| children.first())
        .unwrap_or(expr);
    match literal {
        Expr::Atom(Atom::Int(n), _) => usize::try_from(*n).ok(),
        _ => None,
    }
}

/// [04-LIN-9] / spec/04 section 8.4.1: the ADTs whose definitions carry a
/// random key, by the same least fixed point as
/// [`compute_tensor_carrying_adts`] over the same declarations.
fn compute_key_carrying_adts<'a, I>(exprs: I) -> UnordSet<String>
where
    I: IntoIterator<Item = &'a Expr>,
{
    let adt_field_types = collect_adt_field_types(exprs);
    let mut carriers: UnordSet<String> = UnordSet::new();
    loop {
        let mut grew = false;
        for (name, field_tys) in adt_field_types.to_sorted() {
            if carriers.contains(name) {
                continue;
            }
            if field_tys
                .iter()
                .any(|ty| type_expr_holds_key(ty, &carriers))
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

/// Walk top-level declarations and return the set of ADT names whose
/// definitions (transitively) carry a tensor field. Used by the
/// linearity checker to recognize `&MyParams` as a valid borrow when
/// `MyParams` is a record with a `tensor[...]` field.
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
    let adt_field_types = collect_adt_field_types(exprs);

    // Step 2: fixed-point iteration. An ADT is tensor-carrying iff any
    // of its field types contains a tensor (looking up other ADTs in
    // the current set). Reuses the strict tensor-evidence query against
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
                .any(|ty| type_expr_proves_tensor(ty, &carriers))
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

/// Every `(adt_name, field_type_exprs)` pair from the `deftype` declarations
/// in `exprs`, descending through `(module {} name ...)` wrappers. Shared by
/// the tensor and key carrier fixed points so both read the same fields.
fn collect_adt_field_types<'a, I>(exprs: I) -> UnordMap<String, Vec<Expr>>
where
    I: IntoIterator<Item = &'a Expr>,
{
    let mut adt_field_types: UnordMap<String, Vec<Expr>> = UnordMap::new();
    fn collect(expr: &Expr, out: &mut UnordMap<String, Vec<Expr>>) {
        let (tag, children) = match expr.carrier() {
            ExprCarrier::DecodedNode(tag, _, children) => (tag, children),
            ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => return,
        };
        match tag {
            DeepTag::Module => {
                // Decoded children are `[name, body...]`; skip the name for
                // the same shape the `deftype` branch below uses.
                for child in children.iter().skip(1) {
                    collect(child, out);
                }
            }
            DeepTag::Deftype => {
                let Some(name) = children.first().and_then(symbol_name) else {
                    return;
                };
                let mut field_tys: Vec<Expr> = Vec::new();
                for child in children.iter().skip(1) {
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
    adt_field_types
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

/// `builtin_callee` is the callee when it names a builtin rather than a
/// lexical binding; only the builtin `drop` ends its argument's lifetime.
fn app_site(expr: &Expr, children: &[Expr], builtin_callee: Option<&str>) -> ConsumeSite {
    let name = children
        .first()
        .and_then(var_name)
        .map(|name| format!("call to `{name}`"))
        .unwrap_or_else(|| "call".to_string());
    ConsumeSite {
        description: format!("{name} {}", diag_site(expr)),
        kind: ConsumeKind::Structural,
        terminal: builtin_callee == Some("drop"),
    }
}

fn generic_site(expr: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!("use {}", diag_site(expr)),
        kind: ConsumeKind::Structural,
        terminal: false,
    }
}

fn realize_site(expr: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!("realize {}", diag_site(expr)),
        kind: ConsumeKind::Structural,
        terminal: false,
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
    use chelis_deep::ast::{Atom, Expr, MetaExpr, Metadata};

    fn span() -> Span {
        Span::new(0, 0)
    }

    fn sym(name: &str) -> Expr {
        Expr::Atom(Atom::Name(name.to_string()), span())
    }

    fn meta(entries: Vec<(&str, Expr)>) -> Metadata {
        let mut metadata = Metadata::default();
        for (key, value) in entries {
            assert_eq!(key, "type", "linearity fixtures only declare types");
            metadata
                .insert(chelis_deep::annotations::MetadataValue::Type(
                    chelis_deep::annotations::TypeSyntax::try_new(value).unwrap(),
                ))
                .unwrap();
        }
        metadata
    }

    /// Build `(tag {meta} children...)`: a stamped node for a vocabulary tag,
    /// an undecodable form for any other head.
    fn node(tag: &str, meta_entries: Vec<(&str, Expr)>, children: Vec<Expr>) -> Expr {
        match DeepTag::parse(tag) {
            Some(tag) => Expr::node(tag, meta(meta_entries), children, span()),
            None => Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
                head: tag.to_string(),
                meta: meta(meta_entries),
                children,
                span: span(),
            })),
        }
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
    fn admitted_literal_atoms_are_ownership_neutral() {
        let int = |value| Expr::Atom(Atom::Int(value), span());
        let bool_lit = |value| {
            node(
                "lit",
                vec![("type", node("t-prim", vec![], vec![sym("bool")]))],
                vec![Expr::Atom(Atom::Bool(value), span())],
            )
        };
        let cases = [
            ("direct definition body", int(42)),
            (
                "if branches",
                node("if", vec![], vec![bool_lit(true), int(1), int(2)]),
            ),
            ("block child", node("block", vec![], vec![int(1)])),
            (
                "literal effect handler",
                node(
                    "handle-effect",
                    vec![],
                    vec![
                        node(
                            "lit",
                            vec![("type", node("t-prim", vec![], vec![sym("i64")]))],
                            vec![int(7)],
                        ),
                        int(1),
                    ],
                ),
            ),
            (
                "tuple-get selector",
                node(
                    "tuple-get",
                    vec![],
                    vec![node("tuple", vec![], vec![int(7)]), int(0)],
                ),
            ),
        ];

        for (label, body) in cases {
            let program = CheckedProgram::unchecked_for_linearity_diagnostic_test(
                vec![node("def", vec![], vec![sym("accepted"), body])],
                BTreeMap::new(),
            );
            check_linearity(&program)
                .unwrap_or_else(|errors| panic!("{label} must remain admitted: {errors:?}"));
        }
    }

    #[test]
    fn effect_handler_payload_retains_nested_ownership_traversal() {
        let use_x = || node("realize", vec![], vec![node("var", vec![], vec![sym("x")])]);
        let handler = Expr::BareList(vec![use_x()], span());
        let program = CheckedProgram::unchecked_for_linearity_diagnostic_test(
            vec![
                node(
                    "def",
                    vec![],
                    vec![sym("x"), node("var", vec![], vec![sym("x")])],
                ),
                node(
                    "def",
                    vec![],
                    vec![
                        sym("result"),
                        node(
                            "handle-effect",
                            vec![],
                            vec![
                                handler,
                                node("copy", vec![], vec![node("var", vec![], vec![sym("x")])]),
                            ],
                        ),
                    ],
                ),
            ],
            BTreeMap::from([("x".to_string(), tensor_4_f32())]),
        );

        let errors = check_linearity(&program)
            .expect_err("the handler and body must both retain their ownership uses");
        assert!(
            errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume)),
            "the second use must observe the handler payload's first consume: {errors:?}"
        );
    }

    #[test]
    fn every_non_runtime_carrier_is_diagnosed_in_runtime_position() {
        let cases = [
            ("metadata map", Expr::Map(Metadata::default(), span())),
            ("empty structural list", Expr::BareList(Vec::new(), span())),
        ];

        for (label, body) in cases {
            let program = CheckedProgram::unchecked_for_linearity_diagnostic_test(
                vec![node("def", vec![], vec![sym("bad"), body])],
                BTreeMap::new(),
            );
            let errors = check_linearity(&program)
                .expect_err("a non-runtime carrier in runtime position must fail closed");
            assert!(
                errors
                    .iter()
                    .any(|error| matches!(error.kind, CheckErrorKind::MalformedForm)),
                "{label} must produce a linearity-owned malformed-form diagnostic: {errors:?}"
            );
        }
    }

    #[test]
    fn rejected_non_runtime_carriers_still_walk_nested_content() {
        let nested_unknown = || {
            Expr::UnknownForm(Box::new(chelis_deep::ast::UnknownFormData {
                head: "future-runtime-form".to_string(),
                meta: Metadata::default(),
                children: Vec::new(),
                span: span(),
            }))
        };
        let cases = [
            (
                "structural list",
                Expr::BareList(vec![nested_unknown()], span()),
            ),
            (
                "metadata expression",
                Expr::MetaExpr(
                    MetaExpr {
                        metadata: Metadata::default(),
                        expr: Box::new(nested_unknown()),
                    },
                    span(),
                ),
            ),
        ];

        for (label, body) in cases {
            let program = CheckedProgram::unchecked_for_linearity_diagnostic_test(
                vec![node("def", vec![], vec![sym("bad"), body])],
                BTreeMap::new(),
            );
            let errors = check_linearity(&program)
                .expect_err("a non-runtime carrier in runtime position must fail closed");
            let malformed_count = errors
                .iter()
                .filter(|error| matches!(error.kind, CheckErrorKind::MalformedForm))
                .count();
            assert!(
                malformed_count >= 2,
                "{label} must diagnose itself and retain traversal of its nested unknown form: \
                 {errors:?}"
            );
        }
    }

    #[test]
    fn singleton_structural_binder_list_is_not_a_runtime_form() {
        let type_parameters = Expr::BareList(vec![sym("a")], span());
        let program = CheckedProgram::unchecked_for_linearity_diagnostic_test(
            vec![node(
                "defsig",
                vec![],
                vec![
                    sym("identity"),
                    type_parameters,
                    node("t-unit", vec![], vec![]),
                ],
            )],
            BTreeMap::new(),
        );

        check_linearity(&program)
            .expect("a singleton binder list in a structural slot is not a runtime carrier");
    }

    #[test]
    fn undecodable_runtime_carrier_cannot_hide_nested_ownership_uses() {
        let unreadable_body = Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
            head: "future-block".into(),
            meta: Metadata::default(),
            children: vec![
                node("realize", vec![], vec![node("var", vec![], vec![sym("x")])]),
                node(
                    "app",
                    vec![],
                    vec![
                        node("var", vec![], vec![sym("add")]),
                        node("var", vec![], vec![sym("x")]),
                        node("var", vec![], vec![sym("x")]),
                    ],
                ),
            ],
            span: span(),
        }));
        let program = CheckedProgram::unchecked_for_linearity_diagnostic_test(
            vec![
                node(
                    "def",
                    vec![],
                    vec![
                        sym("x"),
                        node("lit", vec![], vec![Expr::Atom(Atom::Int(1), span())]),
                    ],
                ),
                node("def", vec![], vec![sym("bad"), unreadable_body]),
            ],
            BTreeMap::from([("x".to_string(), tensor_4_f32())]),
        );

        let errors = check_linearity(&program)
            .expect_err("an undecodable carrier must conservatively traverse nested uses");
        assert!(
            errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::UseAfterConsume)
                    && error.message.contains("variable `x`")
            }),
            "an unreadable carrier must not become an empty subtree: {errors:?}"
        );
    }

    #[test]
    fn undecodable_runtime_carrier_is_diagnosed_even_without_children() {
        let unreadable_body = Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
            head: "future-empty".into(),
            meta: Metadata::default(),
            children: Vec::new(),
            span: span(),
        }));
        let program = CheckedProgram::unchecked_for_linearity_diagnostic_test(
            vec![node("def", vec![], vec![sym("bad"), unreadable_body])],
            BTreeMap::new(),
        );

        let errors = check_linearity(&program)
            .expect_err("an empty undecodable runtime carrier must fail closed");
        assert!(
            errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::MalformedForm)),
            "expected a linearity-owned undecodable-carrier diagnostic: {errors:?}"
        );
    }

    #[test]
    fn malformed_type_carrier_cannot_authorize_borrowing() {
        let tensor_with_runtime_child = node(
            "t-tensor",
            vec![],
            vec![node("var", vec![], vec![sym("not-a-type")])],
        );
        let type_variable_with_runtime_name = node(
            "t-var",
            vec![],
            vec![node("var", vec![], vec![sym("not-a-name")])],
        );
        let metadata_expression_tensor = Expr::MetaExpr(
            MetaExpr {
                metadata: Metadata::default(),
                expr: Box::new(tensor_4_f32()),
            },
            span(),
        );
        let borrowed_x = || node("borrow", vec![], vec![node("var", vec![], vec![sym("x")])]);

        for malformed_type in [
            tensor_with_runtime_child,
            type_variable_with_runtime_name,
            metadata_expression_tensor,
        ] {
            let body = node(
                "app",
                vec![],
                vec![
                    node("var", vec![], vec![sym("add")]),
                    borrowed_x(),
                    borrowed_x(),
                ],
            );
            let program = CheckedProgram::unchecked_for_linearity_diagnostic_test(
                vec![
                    node(
                        "def",
                        vec![],
                        vec![
                            sym("x"),
                            node("lit", vec![], vec![Expr::Atom(Atom::Int(1), span())]),
                        ],
                    ),
                    node("def", vec![], vec![sym("bad"), body]),
                ],
                BTreeMap::from([("x".to_string(), malformed_type)]),
            );

            let errors = check_linearity(&program)
                .expect_err("malformed type evidence must not authorize borrowing");
            assert!(
                errors
                    .iter()
                    .any(|error| matches!(error.kind, CheckErrorKind::InvalidBorrow)),
                "expected malformed type evidence to be rejected at the borrow: {errors:?}"
            );
        }
    }

    #[test]
    fn redteam_malformed_match_arm_cannot_hide_nested_consume() {
        let realize_x = || node("realize", vec![], vec![node("var", vec![], vec![sym("x")])]);
        let malformed_arms = [
            node("params", vec![], vec![realize_x()]),
            node(
                "params",
                vec![],
                vec![node("params", vec![], vec![realize_x()])],
            ),
            node(
                "params",
                vec![],
                vec![node(
                    "realize",
                    vec![],
                    vec![node(
                        "params",
                        vec![],
                        vec![node("params", vec![], vec![realize_x()])],
                    )],
                )],
            ),
        ];

        for malformed_arm in malformed_arms {
            let malformed_match = node(
                "match",
                vec![],
                vec![
                    node(
                        "lit",
                        vec![("type", node("t-prim", vec![], vec![sym("i32")]))],
                        vec![Expr::Atom(Atom::Int(0), span())],
                    ),
                    malformed_arm,
                ],
            );
            assert_eq!(
                free_runtime_variables(&malformed_match),
                vec!["x".to_string()],
                "a malformed arm must not hide a closure capture"
            );
            let body = node(
                "block",
                vec![],
                vec![
                    malformed_match,
                    node(
                        "app",
                        vec![],
                        vec![
                            node("var", vec![], vec![sym("add")]),
                            node("borrow", vec![], vec![node("var", vec![], vec![sym("x")])]),
                            node("borrow", vec![], vec![node("var", vec![], vec![sym("x")])]),
                        ],
                    ),
                ],
            );
            let program = CheckedProgram::unchecked_for_linearity_diagnostic_test(
                vec![
                    node(
                        "def",
                        vec![],
                        vec![
                            sym("x"),
                            node("lit", vec![], vec![Expr::Atom(Atom::Int(1), span())]),
                        ],
                    ),
                    node("def", vec![], vec![sym("bad"), body]),
                ],
                BTreeMap::from([("x".to_string(), tensor_4_f32())]),
            );

            let errors = check_linearity(&program)
                .expect_err("a malformed arm must diagnose and retain nested ownership traversal");
            assert!(
                errors.iter().any(|error| {
                    matches!(error.kind, CheckErrorKind::UseAfterConsume)
                        && error.message.contains("variable `x`")
                }),
                "the malformed arm's nested realize must remain visible: {errors:?}"
            );
        }
    }
}
