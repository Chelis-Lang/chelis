//! Type environment: maps variable names to type schemes.

use chelis_unord::{UnordMap, UnordSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::session::DeclarationDiagnosticOwner;
use crate::types::*;
use crate::unify::{GenericParameter, Subst};

#[cfg(feature = "generalize-sweep-oracle")]
thread_local! {
    static GENERALIZE_SWEEP_ORACLE_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    static GENERALIZE_SWEEP_ENV_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(feature = "generalize-sweep-oracle")]
fn note_generalize_sweep_env_visit() {
    GENERALIZE_SWEEP_ENV_VISITS.with(|visits| visits.set(visits.get() + 1));
}

/// Run `f` through the production level path without invoking the reference
/// sweep. Used by the structural elimination test in the authoritative
/// parity-oracle build.
#[cfg(all(test, feature = "generalize-sweep-oracle"))]
pub(crate) fn without_generalize_sweep_oracle<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            GENERALIZE_SWEEP_ORACLE_ENABLED.with(|enabled| enabled.set(self.0));
        }
    }

    let previous = GENERALIZE_SWEEP_ORACLE_ENABLED.with(|enabled| enabled.replace(false));
    let _restore = Restore(previous);
    f()
}

#[cfg(all(test, feature = "generalize-sweep-oracle"))]
pub(crate) fn reset_generalize_sweep_env_visits() {
    GENERALIZE_SWEEP_ENV_VISITS.with(|visits| visits.set(0));
}

#[cfg(all(test, feature = "generalize-sweep-oracle"))]
pub(crate) fn generalize_sweep_env_visits() -> usize {
    GENERALIZE_SWEEP_ENV_VISITS.with(std::cell::Cell::get)
}

/// Provenance of a let-bound `int`-valued name, tracked so a runtime
/// `expand` size can be checked for materializability (chelis#397/#469).
///
/// A runtime `expand` size has a backend representation only when its
/// extent is recoverable: either it folds to a compile-time constant, or
/// it provably derives from an in-scope tensor's `shape(t, axis)` read.
/// A *truly sourceless* runtime scalar (a bare `i32`/`i64` parameter)
/// has neither, so it must be rejected at check time to keep
/// check↔build↔eval in sync. The discriminator is PROVENANCE, not the
/// surface spelling: `let-bound`, `cast`-wrapped, and arithmetic spellings
/// all reduce to one of these classes. The inline `shape(...)` and
/// in-scope-tensor-dim spellings are recognized syntactically at the
/// expand site; this map only records what a `let` binding carries forward
/// so a later `expand(b, 0, a_dim)` can recover `a_dim`'s class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeProvenance {
    /// The value folds to a compile-time constant (a literal, `cast(N,_)`,
    /// or integer arithmetic over such values). The host runtime and the
    /// evaluator can compute it; it is a materializable extent.
    Static,
    /// The value provably derives from an in-scope tensor's
    /// `shape(t, axis)` read — directly, through `cast`, through integer
    /// arithmetic, or transitively through another shape-provenance
    /// binding. The backend reads the extent from that tensor's shape.
    ShapeSourced,
}

/// The checker identities owned by one declaration binder list.
///
/// A `defsig` binder list is intentionally unkinded: the same source spelling
/// may occur in type, dimension, and rank positions. Each role therefore owns
/// a distinct internal identity, while every ordinary annotation resolver in
/// the declaration reuses this one object. No annotation is allowed to mint a
/// second identity for a listed name.
#[derive(Debug, Clone, Default)]
pub(crate) struct DeclarationBinderIdentities {
    pub(crate) type_vars: UnordMap<String, TypeVar>,
    pub(crate) dim_vars: UnordMap<String, DimVar>,
    pub(crate) rank_vars: UnordMap<String, RankVar>,
}

impl DeclarationBinderIdentities {
    pub(crate) fn contains_name(&self, name: &str) -> bool {
        self.type_vars.contains_key(name)
            || self.dim_vars.contains_key(name)
            || self.rank_vars.contains_key(name)
    }

    pub(crate) fn type_names(&self) -> UnordMap<TypeVar, String> {
        self.type_vars
            .to_sorted()
            .into_iter()
            .map(|(name, var)| (*var, name.clone()))
            .collect()
    }

    pub(crate) fn dim_names(&self) -> UnordMap<DimVar, String> {
        self.dim_vars
            .to_sorted()
            .into_iter()
            .map(|(name, var)| (*var, name.clone()))
            .collect()
    }

    pub(crate) fn rank_names(&self) -> UnordMap<RankVar, String> {
        self.rank_vars
            .to_sorted()
            .into_iter()
            .map(|(name, var)| (*var, name.clone()))
            .collect()
    }

    fn complete(&mut self, binder_names: &UnordSet<String>, var_gen: &mut VarGen) {
        for name in binder_names.to_sorted() {
            self.type_vars
                .entry(name.clone())
                .or_insert_with(|| var_gen.fresh_tvar());
            self.dim_vars
                .entry(name.clone())
                .or_insert_with(|| var_gen.fresh_dvar());
            self.rank_vars
                .entry(name.clone())
                .or_insert_with(|| var_gen.fresh_rvar());
        }
    }
}

/// Lexical declaration-binder scope for resolving source annotations during
/// one check. A top-level `defsig` owns the identities; cloned [`Env`] values
/// carry the scope through nested `fn`/`let`/`match` inference and discard it
/// when that declaration's cloned environment is dropped.
///
/// This is deliberately check-time-only. It must never enter a serialized
/// [`crate::TypeEnv`], because a later stacked check owns a different set of
/// declarations and therefore a different lexical binder scope.
#[derive(Debug, Clone, Default)]
struct TypeResolutionScope {
    identities: Option<DeclarationBinderIdentities>,
    diagnostic_owner: Option<DeclarationDiagnosticOwner>,
}

/// Constructor identity selected by declaration/import scope.
///
/// Constructor position is structural in Chelis, so an ordinary lexical
/// value binding with the same spelling must not replace this entry. Keeping
/// the owner beside the scheme also avoids rediscovering an arbitrary owner
/// from the ADT registry when two in-scope ADTs use the same constructor name.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConstructorBinding {
    owner: String,
    scheme: Scheme,
}

/// Private structure used only to check the body of a rejected declaration.
#[derive(Debug, Clone)]
pub(crate) struct RejectedSignature {
    pub(crate) scheme: Scheme,
    pub(crate) witness: crate::errors::ErrorWitness,
}

/// Type environment (Γ): maps names to polymorphic type schemes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Env {
    // Schemes are immutable once bound. Lexical snapshots copy the name map,
    // while sharing signature bodies until a scope replaces its own binding.
    bindings: UnordMap<String, Arc<Scheme>>,
    /// Active constructor bindings, separate from ordinary value lookup.
    ///
    /// Every exact owner remains available so constructor syntax can select by
    /// positional versus record shape without consulting out-of-scope registry
    /// entries. Declaration/import order still chooses the active owner when
    /// several candidates have the same shape, matching the value environment's
    /// established constructor binding. A later lexical parameter or block
    /// binding may replace `bindings[name]` for bare value position without
    /// changing constructor position.
    #[serde(default)]
    constructor_bindings: UnordMap<String, Vec<ConstructorBinding>>,
    /// Names introduced by the current lexical scope (function parameters,
    /// block bindings, and pattern bindings). Builtin-specific inference may
    /// only dispatch on a name from the closed builtin vocabulary when that
    /// spelling has not been replaced by one of these bindings. This is
    /// check-time provenance, not part of the reusable or serialized type
    /// environment.
    #[serde(skip)]
    lexical_bindings: UnordSet<String>,
    /// Current declaration's type/dimension/rank binders. Installed only on
    /// the cloned environment used to infer that declaration, inherited by
    /// nested lexical clones, and omitted from cached checker state.
    #[serde(skip)]
    type_resolution_scope: TypeResolutionScope,
    /// Rejected declarations still have usable body constraints. These frames
    /// belong to this check only and never become cached/public signatures.
    #[serde(skip)]
    rejected_signatures: UnordMap<String, Arc<RejectedSignature>>,
    /// Declared result trusted only while checking a linker-reserved
    /// [05-OP-35] stdlib wrapper whose runtime-axis shape proof is owned by
    /// #1298. Never serialized or exposed to entry source.
    #[serde(skip)]
    exact_stdlib_expected_result: Option<Type>,
    /// Checker identities introduced while each `defsig` was resolved, keyed
    /// by definition name.
    ///
    /// The stored identities are PRE-generalization. `instantiate_scheme`
    /// supplies the original-to-fresh hop for all three variable kinds, and
    /// body setup completes names absent from the outer signature. Keeping
    /// type/dimension/rank in one object prevents an ordinary annotation from
    /// receiving lexical permission without declaration-owned identity.
    #[serde(skip)]
    declared_binder_identities: UnordMap<String, DeclarationBinderIdentities>,
    /// Members of the recursive group being inferred whose declared header
    /// omits a type, each with the provisional scheme
    /// [`Self::bind_holed_group_member`] bound it at, shared with its binding
    /// so a lexical snapshot does not copy it.
    #[serde(skip)]
    holed_group_members: UnordMap<String, Arc<Scheme>>,
    /// The declared dtype-family bound of each binder variable those members
    /// share, read from the header when it was bound. The group's completion
    /// can identify the variable with a sibling's copy of it before the
    /// member's contract is decided, and the binding then carries the merged
    /// family, not the declared one.
    #[serde(skip)]
    holed_declared_bounds: UnordMap<TypeVar, Option<TypeVarRestriction>>,
    /// The inference level of the recursive group being inferred.
    #[serde(skip)]
    group_level: Option<u32>,
    /// The composed `fresh TypeVar -> source name` map for the definition
    /// currently being inferred.
    ///
    /// The borrow diagnostic that needs it (`validate_deferred_borrow_vars`)
    /// runs at the per-def drain, after body inference, and never sees the
    /// instantiation that minted the fresh variables. Composing at the
    /// instantiation site and parking the result here is what carries a
    /// source name across that gap.
    #[serde(skip)]
    active_declared_type_names: UnordMap<TypeVar, String>,
    /// The dtype-family bound each of those binders was AUTHORED with, read
    /// before the body is inferred (`None` for an unbounded binder).
    ///
    /// [04-INF-6] quantifies a binder over the instantiations its declaration
    /// admits, so a body rule that must hold at every instantiation (a literal
    /// pattern's, chelis#2442) reads this snapshot rather than the variable's
    /// current restriction, which a body constraint may already have narrowed.
    #[serde(skip)]
    active_declared_type_bounds: UnordMap<TypeVar, Option<TypeVarRestriction>>,
    /// chelis#397/#469: provenance of `let`-bound `int`-valued names, so a
    /// runtime `expand` size built from a `let` binding can be checked for
    /// materializability. Cloned at every lexical scope boundary along with
    /// `bindings` (so it has correct lexical scoping for free) and dropped
    /// from serialization (it is a check-time-only analysis artifact).
    #[serde(skip)]
    size_provenance: UnordMap<String, SizeProvenance>,
    /// Exact values for the `Static` subset of `size_provenance`.
    #[serde(skip)]
    static_size_values: UnordMap<String, i64>,
    /// chelis#631: literal element counts of `let`-bound list expressions,
    /// so `concat(rows, axis)` can size its concat axis through the
    /// binding (a list's length is not part of its type). Same
    /// lexical-scoping-by-`Clone` and add-symmetric mark/clear discipline
    /// as `size_provenance`; check-time-only, dropped from serialization.
    #[serde(skip)]
    list_literal_lens: UnordMap<String, usize>,
    /// chelis#1134 / [04-INF-4]: flattened declaration index of every
    /// top-level eager (non-function) value in the unit being checked.
    ///
    /// Visibility of such a value is a function of its source position, and
    /// of nothing else. Binding presence cannot express that rule: the body
    /// inference schedule reorders function declarations, so "is it bound
    /// yet" and "is it declared yet" are different questions. This map
    /// answers the second one for [`Self::top_level_value_visibility`], which
    /// is the only authority on eager-value scope.
    #[serde(skip)]
    top_level_value_ordinals: UnordMap<String, usize>,
    /// The binding an eager top-level value shadows, when its name was
    /// already bound by an import or a stacked library phase. Before that
    /// value's own declaration the outer binding is still the one in scope,
    /// so lookup falls back here rather than reporting the name unbound.
    #[serde(skip)]
    shadowed_prior_bindings: UnordMap<String, Scheme>,
    /// Flattened declaration index currently being inferred. Set by the
    /// driver loop at each scheduled declaration and inherited by every
    /// lexical clone, so nested scopes compare against the declaration that
    /// owns them rather than against the schedule's position.
    #[serde(skip)]
    current_declaration_ordinal: Option<usize>,
    /// Exact names temporarily visible while one cyclic full-reference
    /// component is co-inferred. This contains the component's provisional
    /// members and, for [04-INF-8] cycle precedence, any later eager target
    /// that the scheduler has already inferred for that component. The
    /// inference driver installs this capability only for the rejected cycle
    /// and restores the prior set on both completion and cancellation; it is
    /// never serialized.
    #[serde(skip)]
    active_top_level_component: UnordSet<String>,
}

/// Result of the [04-INF-4] eager-value scope test, including the exact
/// provisional capability installed for an active rejected cycle. See
/// [`Env::top_level_value_visibility`].
pub(crate) enum TopLevelValueVisibility<'a> {
    /// The name is in scope here, or the rule does not govern it.
    Visible,
    /// The name belongs to a top-level eager value declared after the
    /// declaration being inferred, carrying the outer binding it shadows.
    NotYetDeclared { shadowed: Option<&'a Scheme> },
}

/// An instantiated scheme body, paired with the original-to-fresh renaming of
/// each kind of quantifier that a caller can need to read back.
///
/// The type-variable renaming maps to a `Type` because a quantified type
/// variable may instantiate to any type. Dimension and rank quantifiers each
/// instantiate to a fresh variable of the same kind.
#[derive(Clone)]
pub(crate) struct InstantiatedScheme {
    pub(crate) ty: Type,
    pub(crate) tvars: Vec<(TypeVar, Type)>,
    pub(crate) dvars: Vec<(DimVar, DimVar)>,
    pub(crate) rvars: Vec<(RankVar, RankVar)>,
}

impl Env {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a name. Returns None if unbound.
    pub fn lookup(&self, name: &str) -> Option<&Scheme> {
        self.bindings.get(name).map(Arc::as_ref)
    }

    /// Look up the active constructor owner and scheme for an exact name.
    pub(crate) fn lookup_constructor(&self, name: &str) -> Option<(&str, &Scheme)> {
        self.constructor_bindings
            .get(name)
            .and_then(|bindings| bindings.last())
            .map(|binding| (binding.owner.as_str(), &binding.scheme))
    }

    /// Iterate every in-scope exact constructor candidate in declaration order.
    pub(crate) fn lookup_constructors(
        &self,
        name: &str,
    ) -> impl DoubleEndedIterator<Item = (&str, &Scheme)> {
        self.constructor_bindings
            .get(name)
            .into_iter()
            .flatten()
            .map(|binding| (binding.owner.as_str(), &binding.scheme))
    }

    /// Install the binder set owned by the declaration whose body is about to
    /// be inferred. Callers use a cloned `Env`, so this scope cannot leak to a
    /// sibling declaration or back into a reusable library snapshot.
    pub(crate) fn set_type_resolution_scope(
        &mut self,
        identities: Option<&DeclarationBinderIdentities>,
        diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
    ) {
        self.type_resolution_scope.identities = identities.cloned();
        self.type_resolution_scope.diagnostic_owner = diagnostic_owner.cloned();
    }

    /// Declaration-owned identities visible to a nested source annotation.
    /// Absence means closed input: named `t-var`/`d-var`/`d-rank` nodes do not
    /// allocate inference variables.
    pub(crate) fn type_resolution_binders(&self) -> Option<&DeclarationBinderIdentities> {
        self.type_resolution_scope.identities.as_ref()
    }

    /// Diagnostic identity shared by every annotation resolver in the active
    /// declaration body.
    pub(crate) fn type_resolution_diagnostic_owner(&self) -> Option<&DeclarationDiagnosticOwner> {
        self.type_resolution_scope.diagnostic_owner.as_ref()
    }

    pub(crate) fn set_exact_stdlib_expected_result(&mut self, result: Option<Type>) {
        self.exact_stdlib_expected_result = result;
    }

    pub(crate) fn exact_stdlib_expected_result(&self) -> Option<&Type> {
        self.exact_stdlib_expected_result.as_ref()
    }

    /// Record the size provenance of a `let`-bound name (chelis#397/#469).
    pub fn mark_size_provenance(&mut self, name: &str, prov: SizeProvenance) {
        self.size_provenance.insert(name.to_string(), prov);
        self.static_size_values.remove(name);
    }

    /// Record one checked, fully folded integer extent binding.
    pub fn mark_static_size_value(&mut self, name: &str, value: i64) {
        self.size_provenance
            .insert(name.to_string(), SizeProvenance::Static);
        self.static_size_values.insert(name.to_string(), value);
    }

    /// Clear any recorded size provenance for `name` (chelis#397/#469).
    ///
    /// The provenance map is add-symmetric: it must be CLEARED at every
    /// binding site whose RHS is sourceless, and at every value-parameter
    /// bind, so a name that re-binds (or shadows an outer name) to a
    /// sourceless runtime scalar does not inherit a stale `ShapeSourced`/
    /// `Static` entry. Without this, `len = shape(x, 0); len = k;
    /// expand(b, 0, len)` (BLOCKER B — a re-bind) and a sourceless value
    /// parameter `d` that shadows an outer shape-sourced `d` (BLOCKER C — a
    /// shadow inherited through the derived `Clone`) would both be wrongly
    /// accepted at check, materializing a runtime extent that contradicts
    /// the checked type (a check↔eval divergence / check-clean-fails-build).
    pub fn clear_size_provenance(&mut self, name: &str) {
        self.size_provenance.remove(name);
        self.static_size_values.remove(name);
    }

    /// The recorded size provenance of a name, if any (chelis#397/#469).
    pub fn size_provenance(&self, name: &str) -> Option<SizeProvenance> {
        self.size_provenance.get(name).copied()
    }

    /// Exact checked value of a previously folded lexical extent.
    pub fn static_size_value(&self, name: &str) -> Option<i64> {
        self.static_size_values.get(name).copied()
    }

    /// Record the literal element count of a `let`-bound list (chelis#631).
    pub fn mark_list_literal_len(&mut self, name: &str, len: usize) {
        self.list_literal_lens.insert(name.to_string(), len);
    }

    /// Clear any recorded list-literal length for `name` (chelis#631).
    ///
    /// Add-symmetric like [`Self::clear_size_provenance`]: cleared at
    /// every binding site whose RHS is not a list literal and at every
    /// value-parameter bind, so a re-bind or shadow does not inherit a
    /// stale length and mis-size a later `concat`.
    pub fn clear_list_literal_len(&mut self, name: &str) {
        self.list_literal_lens.remove(name);
    }

    /// The recorded list-literal length of a name, if any (chelis#631).
    pub fn list_literal_len(&self, name: &str) -> Option<usize> {
        self.list_literal_lens.get(name).copied()
    }

    /// True when some in-scope tensor binding carries the named dimension
    /// `name` in its shape — i.e. `name` is a §4.7.2 Form-2 symbolic dim
    /// with a real tensor source the backend can read the extent from
    /// (chelis#397/#469). The check-layer analog of the IR layer's
    /// `LowerCtx::symbol_has_tensor_source`.
    pub fn tensor_carries_dim(&self, name: &str) -> bool {
        self.bindings
            .to_sorted()
            .into_iter()
            .any(|(_, scheme)| type_carries_dim_name(&scheme.body, name))
    }

    pub(crate) fn tensor_carries_dim_with_subst(&self, name: &str, subst: &Subst) -> bool {
        self.bindings
            .to_sorted()
            .into_iter()
            .any(|(_, scheme)| type_carries_dim_name(&subst.semantic_type(&scheme.body), name))
    }

    /// Look up an imported or qualified name by its unique terminal segment.
    pub fn lookup_terminal_unique(&self, name: &str) -> Option<&Scheme> {
        let mut matches = self
            .bindings
            .to_sorted()
            .into_iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first.as_ref())
    }

    /// Extend the environment with a new binding.
    pub fn bind(&mut self, name: String, scheme: Scheme) {
        self.bind_shared(name, Arc::new(scheme));
    }

    fn bind_shared(&mut self, name: String, scheme: Arc<Scheme>) {
        self.rejected_signatures.remove(&name);
        self.bindings.insert(name, scheme);
    }

    pub(crate) fn bind_rejected_signature(
        &mut self,
        name: String,
        recovery: crate::deep_type::RejectedSignatureType,
        subst: &mut Subst,
    ) {
        let scheme = self.generalize(&recovery.ty, subst);
        self.bind(
            name.clone(),
            Scheme::mono(crate::errors::propagate(&recovery.witness)),
        );
        self.rejected_signatures.insert(
            name,
            Arc::new(RejectedSignature {
                scheme,
                witness: recovery.witness,
            }),
        );
    }

    pub(crate) fn rejected_signature(&self, name: &str) -> Option<&RejectedSignature> {
        self.rejected_signatures.get(name).map(Arc::as_ref)
    }

    /// Bind a constructor in both structural constructor position and the
    /// ordinary value environment used by bare/nullary references.
    pub(crate) fn bind_constructor(&mut self, name: String, owner: String, scheme: Scheme) {
        self.rejected_signatures.remove(&name);
        self.bindings.insert(name.clone(), Arc::new(scheme.clone()));
        let candidates = self.constructor_bindings.entry(name).or_default();
        candidates.retain(|candidate| candidate.owner != owner);
        candidates.push(ConstructorBinding { owner, scheme });
    }

    /// Extend the environment with a binding introduced by ordinary lexical
    /// scope. Unlike [`Self::bind`], this also records that builtin callable
    /// dispatch must not claim the name while this environment lives.
    pub(crate) fn bind_lexical(&mut self, name: String, scheme: Scheme) {
        self.lexical_bindings.insert(name.clone());
        self.bind(name, scheme);
    }

    /// Whether an ordinary lexical binding owns `name` in this environment.
    pub(crate) fn is_lexically_bound(&self, name: &str) -> bool {
        self.lexical_bindings.contains(name)
    }

    /// Drop the [04-INF-4] scope state of a previous check unit.
    ///
    /// Source position is a property of one unit's declaration list. A
    /// stacked library or context phase reuses the same environment, and its
    /// values are ordinary imported bindings from the next unit's point of
    /// view, so their positions must not survive into it.
    pub(crate) fn reset_top_level_value_scope(&mut self) {
        self.top_level_value_ordinals.clear();
        self.shadowed_prior_bindings.clear();
        self.current_declaration_ordinal = None;
        self.active_top_level_component.clear();
    }

    /// Record the source position of one top-level eager value, and the
    /// binding it shadows if the name was already in scope.
    pub(crate) fn note_top_level_value_ordinal(
        &mut self,
        name: String,
        ordinal: usize,
        shadowed: Option<Scheme>,
    ) {
        if let Some(prior) = shadowed {
            self.shadowed_prior_bindings.insert(name.clone(), prior);
        }
        self.top_level_value_ordinals.insert(name, ordinal);
    }

    /// The recorded source position of a top-level eager value, if any.
    pub(crate) fn top_level_value_ordinal(&self, name: &str) -> Option<usize> {
        self.top_level_value_ordinals.get(name).copied()
    }

    /// Point the environment at the declaration whose body is being inferred.
    pub(crate) fn set_current_declaration_ordinal(&mut self, ordinal: Option<usize>) {
        self.current_declaration_ordinal = ordinal;
    }

    /// Replace the exact cyclic-component capability whose provisional member
    /// bindings and already-inferred [04-INF-8] precedence targets may bypass
    /// ordinary source-position visibility. The returned capability must be
    /// restored before the inference level is left.
    #[must_use = "restore the prior top-level component capability on every exit"]
    pub(crate) fn replace_active_top_level_component(
        &mut self,
        names: UnordSet<String>,
    ) -> UnordSet<String> {
        std::mem::replace(&mut self.active_top_level_component, names)
    }

    /// [04-INF-4]: whether `name` resolves to a top-level eager value that is
    /// already declared at the current declaration, is a provisional member
    /// of the exact cyclic component currently being co-inferred, or is one
    /// of that rejected component's already-inferred [04-INF-8] precedence
    /// targets.
    ///
    /// `Visible` covers every name this rule does not govern: a lexical
    /// binding that shadows the value, an imported or library name, a
    /// function, and a value declared earlier. A value's own declaration sees
    /// itself, which is what makes the explicitly typed external input
    /// (`x: T = x`) legal without any prebinding. A later value is
    /// `NotYetDeclared`, carrying the outer binding it shadows when there is
    /// one so the caller can resolve against the still-current outer scope.
    /// Active-capability membership is checked after lexical shadowing and
    /// before source order: this lets only exact graph-owned cycle bindings
    /// cross that boundary while preserving local-name precedence.
    pub(crate) fn top_level_value_visibility(&self, name: &str) -> TopLevelValueVisibility<'_> {
        if self.lexical_bindings.contains(name) {
            return TopLevelValueVisibility::Visible;
        }
        if self.active_top_level_component.contains(name) {
            return TopLevelValueVisibility::Visible;
        }
        let (Some(declared_at), Some(current)) = (
            self.top_level_value_ordinals.get(name),
            self.current_declaration_ordinal,
        ) else {
            return TopLevelValueVisibility::Visible;
        };
        if *declared_at <= current {
            return TopLevelValueVisibility::Visible;
        }
        TopLevelValueVisibility::NotYetDeclared {
            shadowed: self.shadowed_prior_bindings.get(name),
        }
    }

    /// Remove a temporary inference binding before generalizing an SCC.
    /// Recursive function components are prebound monomorphically while
    /// their bodies are inferred, then all provisional members are removed
    /// together so their resolved types can be generalized as one unit.
    pub(crate) fn remove_binding(&mut self, name: &str) {
        self.bindings.remove(name);
    }

    /// Record the identities introduced while resolving a declaration's
    /// signature. Body setup remaps every occurrence-derived identity through
    /// the scheme instantiation and completes all roles absent from the outer
    /// signature.
    pub(crate) fn record_declared_binder_identities(
        &mut self,
        name: &str,
        identities: DeclarationBinderIdentities,
    ) {
        if !identities.type_vars.is_empty()
            || !identities.dim_vars.is_empty()
            || !identities.rank_vars.is_empty()
        {
            self.declared_binder_identities
                .insert(name.to_string(), identities);
        }
    }

    /// chelis#2584, chelis#2590: bind `name`, a member of the recursive group
    /// being inferred whose declared header omits a type, at its provisional
    /// type. Returns whether the header had an omitted type.
    ///
    /// [04-INF-5] makes an omitted type whatever the body determines, and
    /// types an in-group reference "at the member's provisional monomorphic
    /// type, as [04-INF-2] provides for a recursive call". [04-INF-2] keeps the
    /// two kinds of signature variable apart. An authored binder admits no
    /// substitute, so the header's binders are instantiated once, at fresh
    /// variables minted inside the component's level, which the member's own
    /// body and its own references share; a sibling's reference takes a copy
    /// of them that the component's completion identifies with them
    /// (`infer::group_link::sibling_instance`), so a call that swaps two of
    /// them identifies them, which [04-INF-6] rejects. An inference hole admits
    /// the caller's own type or a fully concrete one, so the provisional
    /// scheme quantifies the holes alone: the member's body and each in-group
    /// reference take their own instance, and the component's completion
    /// decides each reference's instance against the body's
    /// (`infer::group_link`). A header that omits nothing is the member's
    /// scheme already, and keeps it: polymorphic recursion over its binders
    /// stays available to a declaration whose every type is written.
    ///
    /// The declaration's binder identities are re-pointed at the shared
    /// instantiation, which is how the body's annotations and its rigidity
    /// check name the same variables its in-group callers bind.
    pub(crate) fn bind_holed_group_member(
        &mut self,
        name: &str,
        var_gen: &mut VarGen,
        inference_subst: &Subst,
    ) -> bool {
        let Some(header) = self.lookup(name).cloned() else {
            return false;
        };
        let binders = self.declared_binder_identities.get(name).cloned();
        let binder_tvars = binders
            .as_ref()
            .map(|b| {
                b.type_vars
                    .to_sorted()
                    .into_iter()
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let binder_dvars = binders
            .as_ref()
            .map(|b| {
                b.dim_vars
                    .to_sorted()
                    .into_iter()
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let binder_rvars = binders
            .as_ref()
            .map(|b| {
                b.rank_vars
                    .to_sorted()
                    .into_iter()
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let omits_a_type = header.tvars.iter().any(|v| !binder_tvars.contains(v))
            || header.dvars.iter().any(|v| !binder_dvars.contains(v))
            || header.rvars.iter().any(|v| !binder_rvars.contains(v));
        if !omits_a_type {
            return false;
        }
        // The one instantiation mechanism mints the shared variables and
        // carries each binder's dtype bound and dimension label to them. It
        // records no obligation, because the header's constraints are not
        // passed to it; they are renamed onto the provisional scheme instead,
        // which each in-group use then owes.
        let shared = self.instantiate_scheme(
            &Scheme {
                result_origin: None,
                constraints: Vec::new(),
                ..header.clone()
            },
            var_gen,
            inference_subst,
        );
        let mut renaming = Subst::new();
        for (from, to) in &shared.tvars {
            renaming
                .insert_type(*from, to.clone())
                .expect("a fresh provisional renaming is valid");
        }
        for (from, to) in &shared.dvars {
            renaming.insert_dim(*from, Dim::Var(*to));
        }
        for (from, to) in &shared.rvars {
            renaming.insert_rank(*from, vec![Dim::Rank(*to)]);
        }
        let hole_tvars = shared
            .tvars
            .iter()
            .filter(|(from, _)| !binder_tvars.contains(from))
            .filter_map(|(_, to)| match to {
                Type::Var(fresh) => Some(*fresh),
                _ => None,
            })
            .collect::<Vec<_>>();
        let provisional = Scheme {
            result_origin: None,
            tvar_restrictions: hole_tvars
                .iter()
                .filter_map(|var| {
                    inference_subst
                        .tvar_restriction(*var)
                        .map(|restriction| (*var, restriction))
                })
                .collect(),
            tvars: hole_tvars,
            dvars: shared
                .dvars
                .iter()
                .filter(|(from, _)| !binder_dvars.contains(from))
                .map(|(_, to)| *to)
                .collect(),
            rvars: shared
                .rvars
                .iter()
                .filter(|(from, _)| !binder_rvars.contains(from))
                .map(|(_, to)| *to)
                .collect(),
            constraints: header
                .constraints
                .iter()
                .map(|constraint| constraint.map_types(|ty| renaming.apply(ty)))
                .collect(),
            body: shared.ty.clone(),
        };
        let provisional = Arc::new(provisional);
        self.bind_shared(name.to_string(), Arc::clone(&provisional));
        if let Some(original) = binders {
            let mut repointed = DeclarationBinderIdentities::default();
            for (source_name, var) in original.type_vars.to_sorted() {
                if let Some((_, Type::Var(fresh))) =
                    shared.tvars.iter().find(|(from, _)| from == var)
                {
                    repointed.type_vars.insert(source_name.clone(), *fresh);
                    let declared = header
                        .tvar_restrictions
                        .iter()
                        .find(|(restricted, _)| restricted == var)
                        .map(|(_, restriction)| *restriction);
                    self.holed_declared_bounds.insert(*fresh, declared);
                }
            }
            for (source_name, var) in original.dim_vars.to_sorted() {
                if let Some((_, fresh)) = shared.dvars.iter().find(|(from, _)| from == var) {
                    repointed.dim_vars.insert(source_name.clone(), *fresh);
                }
            }
            for (source_name, var) in original.rank_vars.to_sorted() {
                if let Some((_, fresh)) = shared.rvars.iter().find(|(from, _)| from == var) {
                    repointed.rank_vars.insert(source_name.clone(), *fresh);
                }
            }
            self.declared_binder_identities
                .insert(name.to_string(), repointed);
        }
        self.holed_group_members
            .insert(name.to_string(), provisional);
        true
    }

    /// Enter the recursive group whose members are about to be bound at their
    /// provisional types, inferred at `level`.
    pub(crate) fn begin_group_level(&mut self, level: u32) {
        self.group_level = Some(level);
    }

    /// The level of the recursive group being inferred: an in-group
    /// reference's instance of a hole, and a sibling reference's copy of a
    /// member's type, is lowered to it, so it stays monomorphic until the
    /// group completes, as the group's own variables do.
    pub(crate) fn group_level(&self) -> Option<u32> {
        self.group_level
    }

    /// The dtype-family bound the header of a holed member of the group being
    /// inferred declares for its binder variable `var`, or `None` when `var`
    /// is no such binder or declares none.
    pub(crate) fn declared_group_binder_bound(&self, var: TypeVar) -> Option<TypeVarRestriction> {
        self.holed_declared_bounds.get(&var).copied().flatten()
    }

    /// Whether `name` is bound at its provisional type by
    /// [`Self::bind_holed_group_member`] while its group is inferred.
    pub(crate) fn is_holed_group_member(&self, name: &str) -> bool {
        self.holed_group_members.contains_key(name)
    }

    /// Whether a reference to `name` that resolved to `scheme` is an in-group
    /// reference to a member bound by [`Self::bind_holed_group_member`], and
    /// not to a local binding that shadows it.
    pub(crate) fn is_holed_group_reference(&self, name: &str, scheme: &Scheme) -> bool {
        self.holed_group_members
            .get(name)
            .is_some_and(|provisional| {
                provisional.tvars == scheme.tvars
                    && provisional.dvars == scheme.dvars
                    && provisional.rvars == scheme.rvars
                    && provisional.body == scheme.body
            })
    }

    /// The authored binders of every member bound by
    /// [`Self::bind_holed_group_member`], with their source names: every
    /// in-group reference shares them, so they are no member's
    /// inference-introduced type parameters.
    #[allow(clippy::type_complexity)]
    pub(crate) fn holed_group_binders(
        &self,
    ) -> (
        Vec<(String, TypeVar)>,
        Vec<(String, DimVar)>,
        Vec<(String, RankVar)>,
    ) {
        let mut tvars = Vec::new();
        let mut dvars = Vec::new();
        let mut rvars = Vec::new();
        for (name, _) in self.holed_group_members.to_sorted() {
            if let Some(binders) = self.declared_binder_identities.get(name) {
                for (source, var) in binders.type_vars.to_sorted() {
                    tvars.push((source.clone(), *var));
                }
                for (source, var) in binders.dim_vars.to_sorted() {
                    dvars.push((source.clone(), *var));
                }
                for (source, var) in binders.rank_vars.to_sorted() {
                    rvars.push((source.clone(), *var));
                }
            }
        }
        (tvars, dvars, rvars)
    }

    /// The group is complete or aborted: its members' provisional bindings
    /// are replaced, and their binder identities are the shared variables the
    /// completed schemes quantify.
    pub(crate) fn end_holed_group(&mut self) {
        self.holed_group_members = UnordMap::default();
        self.holed_declared_bounds = UnordMap::default();
        self.group_level = None;
    }

    /// The declared dtype-family bound of the authored binder variable `var`:
    /// the header's, for a binder a holed group member shares, and otherwise
    /// the bound it carries now, before its body runs.
    pub(crate) fn declared_binder_bound(
        &self,
        var: TypeVar,
        subst: &Subst,
    ) -> Option<TypeVarRestriction> {
        match self.holed_declared_bounds.get(&var) {
            Some(declared) => *declared,
            None => subst.tvar_restriction(var),
        }
    }

    /// Build the declaration-owned identity object used by every ordinary
    /// annotation resolver and every post-body rigidity check.
    ///
    /// Signature occurrences retain the exact fresh identities minted by
    /// scheme instantiation. Every listed name absent from a role in the
    /// signature receives one fresh declaration-owned identity for that role.
    /// The same spelling may therefore be used independently as a type,
    /// dimension, or rank variable, as the unkinded binder-list contract
    /// requires.
    pub(crate) fn declared_binder_identities_for_body(
        &self,
        name: &str,
        binder_names: &UnordSet<String>,
        instantiation: Option<&InstantiatedScheme>,
        var_gen: &mut VarGen,
    ) -> DeclarationBinderIdentities {
        let mut resolved = DeclarationBinderIdentities::default();
        if self.holed_group_members.contains_key(name)
            && let Some(shared) = self.declared_binder_identities.get(name)
        {
            // The provisional binding quantifies only the omitted types, so
            // its binder identities are the shared variables themselves.
            resolved = shared.clone();
        } else if let (Some(original), Some(instantiation)) =
            (self.declared_binder_identities.get(name), instantiation)
        {
            for (source_name, original_var) in original.type_vars.to_sorted() {
                if let Some(Type::Var(fresh)) = instantiation
                    .tvars
                    .iter()
                    .find_map(|(from, to)| (*from == *original_var).then_some(to))
                {
                    resolved.type_vars.insert(source_name.clone(), *fresh);
                }
            }
            for (source_name, original_var) in original.dim_vars.to_sorted() {
                if let Some((_, fresh)) = instantiation
                    .dvars
                    .iter()
                    .find(|(from, _)| *from == *original_var)
                {
                    resolved.dim_vars.insert(source_name.clone(), *fresh);
                }
            }
            for (source_name, original_var) in original.rank_vars.to_sorted() {
                if let Some((_, fresh)) = instantiation
                    .rvars
                    .iter()
                    .find(|(from, _)| *from == *original_var)
                {
                    resolved.rank_vars.insert(source_name.clone(), *fresh);
                }
            }
        }
        resolved.complete(binder_names, var_gen);
        resolved
    }

    /// Park the composed map for the definition now being inferred, so the
    /// per-def deferred-borrow drain can name what it reports on.
    pub(crate) fn set_active_declared_type_names(&mut self, names: UnordMap<TypeVar, String>) {
        self.active_declared_type_names = names;
    }

    /// The parked map. Empty when the definition declared no type parameters
    /// or none was recorded; callers fall back to the internal id rather than
    /// inventing a name (spec/04 [04-FIT-10]).
    pub(crate) fn active_declared_type_names(&self) -> &UnordMap<TypeVar, String> {
        &self.active_declared_type_names
    }

    /// Park the authored dtype-family bound of each binder in
    /// [`Self::active_declared_type_names`].
    pub(crate) fn set_active_declared_type_bounds(
        &mut self,
        bounds: UnordMap<TypeVar, Option<TypeVarRestriction>>,
    ) {
        self.active_declared_type_bounds = bounds;
    }

    /// The authored binder of the definition now being inferred that `var`
    /// currently denotes, with its source name and authored bound. `None` for
    /// a variable no authored binder resolves to, which is a flexible
    /// inference variable rather than a rigid binder.
    pub(crate) fn authored_type_binder(
        &self,
        var: TypeVar,
        subst: &Subst,
    ) -> Option<(&str, Option<TypeVarRestriction>)> {
        self.active_declared_type_names
            .to_sorted()
            .into_iter()
            .find(|(declared, _)| subst.apply(&Type::Var(**declared)) == Type::Var(var))
            .map(|(declared, name)| {
                let bound = self
                    .active_declared_type_bounds
                    .get(declared)
                    .copied()
                    .flatten();
                (name.as_str(), bound)
            })
    }

    /// Instantiate a scheme into the caller's inference substitution so
    /// quantified semantic restrictions follow the fresh variables.
    pub fn instantiate(
        &self,
        scheme: &Scheme,
        var_gen: &mut VarGen,
        inference_subst: &Subst,
    ) -> Type {
        self.instantiate_scheme(scheme, var_gen, inference_subst).ty
    }

    /// Instantiate a scheme and return the fresh type minted for each
    /// quantified type variable, in quantifier order. The uniform recursive
    /// instantiation check (spec/04 §3.1.1) records this mapping for
    /// in-group references so the group's final substitution can be
    /// compared against the caller's own instantiation.
    pub fn instantiate_with_tvar_mapping(
        &self,
        scheme: &Scheme,
        var_gen: &mut VarGen,
        inference_subst: &Subst,
    ) -> (Type, Vec<(TypeVar, Type)>) {
        let instantiated = self.instantiate_scheme(scheme, var_gen, inference_subst);
        (instantiated.ty, instantiated.tvars)
    }

    /// The one instantiation mechanism (chelis#260 / chelis#1292).
    ///
    /// Every quantifier is renamed here and nowhere else, so the consumers
    /// cannot drift apart: #1292's installation of quantified type-variable
    /// restrictions and the original-to-fresh type/dimension/rank pairings
    /// needed for declaration-owned binder identity.
    ///
    /// Keeping them in one body is deliberate. Both were separately-authored
    /// copies of this loop at one point, and a second copy is exactly how a
    /// scheme gets instantiated with its restrictions dropped: the omission
    /// compiles, and the only symptom is a program that should have been
    /// rejected type-checking.
    ///
    /// Callable directly by sites that need the complete instantiation,
    /// which is why it is crate-visible rather than another projection beside
    /// the ones above.
    pub(crate) fn instantiate_scheme(
        &self,
        scheme: &Scheme,
        var_gen: &mut VarGen,
        inference_subst: &Subst,
    ) -> InstantiatedScheme {
        let origin = scheme.result_origin.as_ref();
        let mut tvars = origin.map_or(&scheme.tvars, |origin| &origin.tvars).clone();
        let mut dvars = origin.map_or(&scheme.dvars, |origin| &origin.dvars).clone();
        let mut rvars = origin.map_or(&scheme.rvars, |origin| &origin.rvars).clone();
        // The raw graph owns shared identity. A recursive group's solved
        // signature may mention additional variables, but must not freshen a
        // variable the origin deliberately kept shared with its producer.
        let carried_origin = origin.map(|origin| {
            Type::Tuple(
                std::iter::once(origin.body.clone())
                    .chain(
                        origin
                            .equations
                            .iter()
                            .flat_map(|equation| equation.types())
                            .cloned(),
                    )
                    .collect(),
            )
        });
        let shared_tvars = carried_origin.as_ref().map(free_tvars).unwrap_or_default();
        let shared_dvars = carried_origin.as_ref().map(free_dvars).unwrap_or_default();
        let shared_rvars = carried_origin.as_ref().map(free_rvars).unwrap_or_default();
        for var in &scheme.tvars {
            if !tvars.contains(var) && !shared_tvars.contains(var) {
                tvars.push(*var);
            }
        }
        for var in &scheme.dvars {
            if !dvars.contains(var) && !shared_dvars.contains(var) {
                dvars.push(*var);
            }
        }
        for var in &scheme.rvars {
            if !rvars.contains(var) && !shared_rvars.contains(var) {
                rvars.push(*var);
            }
        }
        let mut subst = Subst::new();
        let mut tvar_mapping = Vec::with_capacity(tvars.len());
        for &tv in &tvars {
            let fresh = var_gen.fresh_type();
            subst
                .insert_type(tv, fresh.clone())
                .expect("a fresh quantified type-variable renaming is valid");
            if let Type::Var(fresh_var) = fresh {
                // Raw origins and published signatures can use different
                // representatives. Rename both restriction ledgers with the
                // same quantifiers; neither view may erase the other's domain.
                for (_, restriction) in scheme
                    .tvar_restrictions
                    .iter()
                    .chain(
                        origin
                            .into_iter()
                            .flat_map(|origin| &origin.tvar_restrictions),
                    )
                    .filter(|(restricted, _)| *restricted == tv)
                {
                    inference_subst
                        .narrow_tvar_restriction(fresh_var, *restriction)
                        .expect("a checked scheme's restriction ledgers are compatible");
                }
            }
            // [04-LIN-10]: a generic's type parameter stays key-free at every
            // instantiation. The mark lives on the quantified variable, which
            // generalization marked, so a builtin or constructor scheme, never
            // generalized, carries none.
            if let Type::Var(fresh_var) = fresh
                && let Some(origin) = inference_subst.key_free_origin(tv)
            {
                inference_subst.forbid_key_instantiation(fresh_var, origin);
            }
            tvar_mapping.push((tv, fresh));
        }
        // chelis#1654: renaming the quantifiers is exactly what turns a
        // scheme-level obligation into one this USE owes, so it is done here,
        // in the one instantiation mechanism, for the reason stated above:
        // a second copy of this loop that dropped the obligations would
        // compile, and the only symptom would be a program that should have
        // been rejected type-checking.
        //
        // The renaming is applied AFTER the dimension and rank quantifiers are
        // inserted below, so a constraint whose carried type mentions one gets
        // that variable renamed too.
        let mut dvar_mapping = Vec::with_capacity(scheme.dvars.len());
        for &dv in &dvars {
            // Mint the variable directly rather than destructuring
            // `fresh_dim()`: that is `Dim::Var(fresh_dvar())` today, but a
            // pattern match would silently drop the mapping entry (and the
            // name with it) if it ever returned another shape.
            let fresh_dv = var_gen.fresh_dvar();
            dvar_mapping.push((dv, fresh_dv));
            subst.insert_dim(dv, Dim::Var(fresh_dv));
            if let Some(axis) = inference_subst.mapped_axis(dv) {
                // The renaming substitution must recognize the old identity
                // while rewriting the scheme body, and the active inference
                // substitution must recognize the fresh identity afterward.
                subst.mark_mapped_axis(dv, axis);
                inference_subst.mark_mapped_axis(fresh_dv, axis);
            } else {
                inference_subst.copy_dimension_label(dv, fresh_dv);
            }
        }
        let mut rvar_mapping = Vec::with_capacity(scheme.rvars.len());
        for &rv in &rvars {
            // Each rank var instantiates to a fresh sole-`Rank` shape so every
            // call site gets its own rank (Tier-2 rank polymorphism).
            let fresh_rv = var_gen.fresh_rvar();
            rvar_mapping.push((rv, fresh_rv));
            subst.insert_rank(rv, vec![Dim::Rank(fresh_rv)]);
        }
        for constraint in &scheme.constraints {
            let renamed = constraint.map_types(|ty| subst.apply(ty));
            inference_subst.record_collection_contract(renamed);
        }
        if let Some(origin) = origin {
            for equation in &origin.equations {
                inference_subst.record_result_constraint(equation.map_types(|ty| subst.apply(ty)));
            }
            // Origin controls when an equation may supply input evidence; it
            // never replaces the checked signature's required equalities.
            // Instantiate both views together and retain their compatibility
            // as a result constraint, so it cannot admit an unresolved Grad.
            if origin.body != scheme.body {
                inference_subst.record_result_constraint(
                    crate::types::ResultConstraint::Annotation {
                        actual: subst.apply(&origin.body),
                        declared: subst.apply(&scheme.body),
                    },
                );
            }
        }
        InstantiatedScheme {
            ty: subst.apply(origin.map_or(&scheme.body, |origin| &origin.body)),
            tvars: tvar_mapping,
            dvars: dvar_mapping,
            rvars: rvar_mapping,
        }
    }

    /// Collect all free type variables across all bindings in the environment.
    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub fn free_tvars(&self, subst: &Subst) -> UnordSet<TypeVar> {
        let mut result = UnordSet::new();
        for (_, scheme) in self.bindings.to_sorted() {
            #[cfg(feature = "generalize-sweep-oracle")]
            note_generalize_sweep_env_visit();
            let ty = subst.apply_scheme(scheme);
            let body_vars = free_tvars(&ty);
            for v in body_vars {
                if !scheme.tvars.contains(&v) {
                    result.insert(v);
                }
            }
        }
        result
    }

    /// Collect all free dimension variables across all bindings in the environment.
    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub fn free_dvars(&self, subst: &Subst) -> UnordSet<DimVar> {
        let mut result = UnordSet::new();
        for (_, scheme) in self.bindings.to_sorted() {
            #[cfg(feature = "generalize-sweep-oracle")]
            note_generalize_sweep_env_visit();
            let ty = subst.apply_scheme(scheme);
            let body_dvars = free_dvars(&ty);
            for v in body_dvars {
                if !scheme.dvars.contains(&v) {
                    result.insert(v);
                }
            }
        }
        result
    }

    /// Free rank variables in the environment (Tier-2 rank polymorphism).
    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub fn free_rvars(&self, subst: &Subst) -> UnordSet<RankVar> {
        let mut result = UnordSet::new();
        for (_, scheme) in self.bindings.to_sorted() {
            #[cfg(feature = "generalize-sweep-oracle")]
            note_generalize_sweep_env_visit();
            let ty = subst.apply_scheme(scheme);
            for v in free_rvars(&ty) {
                if !scheme.rvars.contains(&v) {
                    result.insert(v);
                }
            }
        }
        result
    }

    /// Generalize a type over variables not free in the environment.
    pub fn generalize(&self, ty: &Type, subst: &Subst) -> Scheme {
        self.generalize_owned(ty, subst, None, &[])
    }

    /// Generalize one deferred declaration using only the contract instances
    /// created while that declaration was inferred.
    ///
    /// Recursive siblings share one inference level, so level membership alone
    /// cannot distinguish two fully monomorphic checked function values. The
    /// driver records each member's exact instance IDs and supplies them here.
    pub(crate) fn generalize_with_collection_contracts(
        &self,
        ty: &Type,
        subst: &Subst,
        owned_contracts: &[crate::unify::CollectionContractId],
    ) -> Scheme {
        self.generalize_owned(ty, subst, Some(owned_contracts), &[])
    }

    /// Generalize the raw type and its deferred equalities as one scoped
    /// graph. An equality component containing a shared variable stays shared;
    /// a fresh instantiation must never copy only part of that component.
    pub(crate) fn generalize_with_result_constraints(
        &self,
        ty: &Type,
        subst: &Subst,
        owned_contracts: Option<&[crate::unify::CollectionContractId]>,
        equations: &[ResultConstraint],
    ) -> Scheme {
        self.generalize_owned(ty, subst, owned_contracts, equations)
    }

    fn generalize_owned(
        &self,
        ty: &Type,
        subst: &Subst,
        owned_contracts: Option<&[crate::unify::CollectionContractId]>,
        equations: &[ResultConstraint],
    ) -> Scheme {
        let (mut level_scheme, ledger_removals) =
            self.generalize_by_levels(ty, subst, owned_contracts);
        #[cfg(feature = "generalize-sweep-oracle")]
        GENERALIZE_SWEEP_ORACLE_ENABLED.with(|enabled| {
            if enabled.get() {
                let (sweep_scheme, _) = self.generalize_by_sweep(ty, subst, owned_contracts);
                assert_eq!(
                    level_scheme.constraints, sweep_scheme.constraints,
                    "level-based collection obligations diverged from the reference environment sweep"
                );
                assert_eq!(
                    level_scheme.tvars, sweep_scheme.tvars,
                    "level-based type quantifiers diverged from the reference environment sweep"
                );
                assert_eq!(
                    level_scheme.tvar_restrictions, sweep_scheme.tvar_restrictions,
                    "level-based type-variable restrictions diverged from the reference environment sweep"
                );
                assert_eq!(
                    level_scheme.dvars, sweep_scheme.dvars,
                    "level-based dimension quantifiers diverged from the reference environment sweep"
                );
                assert_eq!(
                    level_scheme.rvars, sweep_scheme.rvars,
                    "level-based rank quantifiers diverged from the reference environment sweep"
                );
                assert_eq!(
                    level_scheme.body, sweep_scheme.body,
                    "level-based scheme body diverged from the reference environment sweep"
                );
            }
        });
        crate::result_scope::ResultScope::new(equations)
            .retain_closed_quantifiers(&mut level_scheme);
        // Transport contracts this scheme now owns have moved off the
        // inference-local contract ledger. Each later instantiation installs a
        // fresh renamed instance with its own application identity.
        //
        // The exact instance IDs come back from the split. Removing by relation
        // equality could erase a monomorphic recursive sibling's identical
        // contract, which belongs to a different value.
        if !ledger_removals.is_empty() {
            subst.take_collection_contracts(&ledger_removals);
        }
        // [04-LIN-10]: an authored generic stays key-free. A checked key
        // operation's closed relation alone may select a key-carrying value.
        // Raw origin quantifiers also include equation-local intermediates;
        // their caller marks the solved public parameters after publication.
        if equations.is_empty() {
            let closed_key_variables = closed_key_relation_variables(&level_scheme);
            for tv in &level_scheme.tvars {
                if !closed_key_variables.contains(tv) {
                    subst.forbid_key_instantiation(*tv, GenericParameter::default());
                }
            }
        }
        level_scheme
    }

    /// chelis#1654: split the pending collection constraints into the ones
    /// this generalization quantifies and the variables the rest must keep
    /// monomorphic.
    ///
    /// Only relations already owned by a checked function value reach this
    /// split. Their complete variable footprint must be visible in the value's
    /// type; there are no hidden intermediate variables and no body-inferred
    /// relation graph. Consumed application instances are absent from
    /// `pending_collection_contracts` and therefore cannot be republished.
    fn collection_constraints_to_quantify(
        body: &Type,
        body_variables: &[TypeVar],
        body_candidates: &[TypeVar],
        subst: &Subst,
        owned_contracts: Option<&[crate::unify::CollectionContractId]>,
    ) -> CollectionQuantification {
        let mut split = CollectionQuantification::default();
        let pending = subst.pending_collection_contracts();
        if pending.is_empty() {
            return split;
        }
        let visible = body_variables.iter().copied().collect::<UnordSet<_>>();
        let candidates = body_candidates.iter().copied().collect::<UnordSet<_>>();
        let current_level = subst.current_level();
        for (id, level, constraint) in pending {
            let owned = match owned_contracts {
                Some(owned) => owned.contains(&id),
                None => level > current_level,
            };
            if !owned {
                continue;
            }
            let mut footprint = UnordSet::default();
            for carried in constraint.carried_types() {
                footprint.extend(free_tvars(carried));
            }
            let footprint = footprint.into_sorted();
            let owned = if footprint.is_empty() {
                crate::unify::collection_contract_visible_in_type(&constraint, body)
            } else {
                footprint.iter().all(|var| candidates.contains(var))
            };
            if owned {
                split.ledger_removals.push(id);
                if !split.constraints.contains(&constraint) {
                    split.constraints.push(constraint);
                }
            } else if footprint.is_empty() || footprint.iter().all(|var| !visible.contains(var)) {
                // The expression discarded the function-bearing subvalue:
                // `(len, 1).1` and `ignore(len)` must not leave the detached
                // contract behind to reject an unrelated result.
                split.ledger_removals.push(id);
            } else {
                split
                    .pinned
                    .extend(footprint.into_iter().filter(|var| candidates.contains(var)));
            }
        }
        split
    }

    fn generalize_by_levels(
        &self,
        ty: &Type,
        subst: &Subst,
        owned_contracts: Option<&[crate::unify::CollectionContractId]>,
    ) -> (Scheme, Vec<crate::unify::CollectionContractId>) {
        let ty = subst.apply(ty);
        let ty_tvars = free_tvars(&ty);
        let ty_dvars = free_dvars(&ty);
        let ty_rvars = free_rvars(&ty);
        let level = subst.current_level();
        // chelis#1489: a variable still tied to a pending operand gate, as its
        // operand or its result, stays monomorphic until the gate discharges;
        // see `Subst::pending_gate_vars`. Levels cannot see that tie -- it
        // lives in the gate ledger, not in any unification.
        let (pending_t, pending_d, pending_r) = subst.pending_gate_vars();
        // spec/04 §3.1.1: a variable minted for an in-group recursive
        // instantiation stays monomorphic while its group is inferred, so a
        // let-bound alias of a group member cannot smuggle in polymorphic
        // recursion.
        let generalizable = |v: TypeVar| {
            subst.level_of_tvar(v) > level
                && !crate::infer::recursion::tvar_pinned(v)
                && !pending_t.contains(&v)
        };
        let mut tvars = ty_tvars
            .iter()
            .copied()
            .filter(|v| generalizable(*v))
            .collect::<Vec<_>>();
        // Move only already-checked transport contracts whose complete
        // variable footprint belongs to this generalized function value.
        let split = Self::collection_constraints_to_quantify(
            &ty,
            &ty_tvars,
            &tvars,
            subst,
            owned_contracts,
        );
        tvars.retain(|v| !split.pinned.contains(v));
        let constraints = split.constraints;
        let tvar_restrictions = tvars
            .iter()
            .filter_map(|v| {
                subst
                    .tvar_restriction(*v)
                    .map(|restriction| (*v, restriction))
            })
            .collect();
        (
            Scheme {
                result_origin: None,
                tvars,
                tvar_restrictions,
                dvars: ty_dvars
                    .into_iter()
                    .filter(|v| {
                        subst.level_of_dvar(*v) > level
                            && !pending_d.contains(v)
                            && matches!(subst.constraint_dim(&Dim::Var(*v)), Dim::Var(_))
                    })
                    .collect(),
                rvars: ty_rvars
                    .into_iter()
                    .filter(|v| subst.level_of_rvar(*v) > level && !pending_r.contains(v))
                    .collect(),
                constraints,
                body: ty,
            },
            split.ledger_removals,
        )
    }

    /// The pre-#1207 environment-sweep implementation, kept as the reference
    /// the parity assertion in `generalize` checks the level-based path
    /// against. It is compiled only into tests and the temporary parity-oracle
    /// feature.
    ///
    /// It is a reference for *which variables the environment leaves free*, not
    /// for which of those may be quantified. Inference-state exclusions are
    /// mirrored here deliberately rather than inherited: a recursive group's
    /// instantiation variables (`tvar_pinned`), a pending operand gate's
    /// result variables, and an authored dimension binder still owned by the
    /// active declaration level, and a live recursive group's level. These are
    /// properties of inference state that
    /// no environment sweep can observe, so omitting one here would make the
    /// oracle disagree with a correct production path. Keep the exclusions in
    /// step with their production owners.
    #[cfg(feature = "generalize-sweep-oracle")]
    fn generalize_by_sweep(
        &self,
        ty: &Type,
        subst: &Subst,
        owned_contracts: Option<&[crate::unify::CollectionContractId]>,
    ) -> (Scheme, Vec<crate::unify::CollectionContractId>) {
        let ty = subst.apply(ty);
        let env_tvars = self.free_tvars(subst);
        let env_dvars = self.free_dvars(subst);
        let env_rvars = self.free_rvars(subst);
        let active_tvars = self
            .type_resolution_binders()
            .into_iter()
            .flat_map(|binders| binders.type_vars.to_sorted())
            .map(|(_, var)| *var)
            .collect::<UnordSet<_>>();
        let active_dvars = self
            .type_resolution_binders()
            .into_iter()
            .flat_map(|binders| binders.dim_vars.to_sorted())
            .map(|(_, var)| *var)
            .collect::<UnordSet<_>>();
        let active_rvars = self
            .type_resolution_binders()
            .into_iter()
            .flat_map(|binders| binders.rank_vars.to_sorted())
            .map(|(_, var)| *var)
            .collect::<UnordSet<_>>();
        // chelis#1489: the same exclusion as `generalize_by_levels`, so the
        // parity assertion in `generalize` keeps comparing like with like.
        let (pending_t, pending_d, pending_r) = subst.pending_gate_vars();
        let current_level = subst.current_level();
        // A group's provisional hole scheme carries temporary quantifiers
        // for its in-group references. The ordinary environment sweep sees
        // those as bound and can mistake a same-level use for a free let
        // variable. [04-INF-5] keeps the hole monomorphic until the whole
        // group closes; group_level is the exact lifetime of that obligation.
        let group_level = self.group_level();
        let generalizable = |v: TypeVar| {
            !env_tvars.contains(&v)
                && !crate::infer::recursion::tvar_pinned(v)
                && !pending_t.contains(&v)
                && !(active_tvars.contains(&v) && subst.level_of_tvar(v) <= current_level)
                && !group_level.is_some_and(|group| subst.level_of_tvar(v) <= group)
        };
        let ty_tvars = free_tvars(&ty);
        let mut tvars = ty_tvars
            .iter()
            .copied()
            .filter(|v| generalizable(*v))
            .collect::<Vec<_>>();
        // Mirror the checked-contract split in the reference generalizer.
        let split = Self::collection_constraints_to_quantify(
            &ty,
            &ty_tvars,
            &tvars,
            subst,
            owned_contracts,
        );
        tvars.retain(|v| !split.pinned.contains(v));
        let constraints = split.constraints;
        let tvar_restrictions = tvars
            .iter()
            .filter_map(|v| {
                subst
                    .tvar_restriction(*v)
                    .map(|restriction| (*v, restriction))
            })
            .collect();
        (
            Scheme {
                result_origin: None,
                tvars,
                tvar_restrictions,
                dvars: free_dvars(&ty)
                    .into_iter()
                    .filter(|v| {
                        !env_dvars.contains(v)
                            && !pending_d.contains(v)
                            && !(active_dvars.contains(v)
                                && subst.level_of_dvar(*v) <= current_level)
                            && matches!(subst.constraint_dim(&Dim::Var(*v)), Dim::Var(_))
                    })
                    .collect(),
                rvars: free_rvars(&ty)
                    .into_iter()
                    .filter(|v| {
                        !env_rvars.contains(v)
                            && !pending_r.contains(v)
                            && !(active_rvars.contains(v)
                                && subst.level_of_rvar(*v) <= current_level)
                    })
                    .collect(),
                constraints,
                body: ty,
            },
            split.ledger_removals,
        )
    }
}

/// Variables selected solely by a checked key operation's closed relation.
/// A relation with no matching callable in the value, or a variable also used
/// outside that callable, cannot grant any exception to [04-LIN-10].
pub(crate) fn closed_key_relation_variables(scheme: &Scheme) -> UnordSet<TypeVar> {
    fn outside_callable(ty: &Type, callable: &Type, outside: &mut UnordSet<TypeVar>) -> bool {
        if ty == callable {
            return true;
        }
        match ty {
            Type::Fn(args, ret) => {
                let mut found = outside_callable(ret, callable, outside);
                for arg in args {
                    found |= outside_callable(arg, callable, outside);
                }
                found
            }
            Type::Ref(inner) => outside_callable(inner, callable, outside),
            Type::Adt(_, args) | Type::Tuple(args) => {
                let mut found = false;
                for arg in args {
                    found |= outside_callable(arg, callable, outside);
                }
                found
            }
            Type::KindedAdt(_, args) => {
                let mut found = false;
                for arg in args {
                    if let NominalArg::Type(arg) = arg {
                        found |= outside_callable(arg, callable, outside);
                    }
                }
                found
            }
            other => {
                outside.extend(free_tvars(other));
                false
            }
        }
    }

    let mut closed = UnordSet::new();
    for relation in &scheme.constraints {
        if !matches!(
            relation,
            CollectionConstraint::KeyFromSeed { .. }
                | CollectionConstraint::SplitKey { .. }
                | CollectionConstraint::SplitKeys { .. }
                | CollectionConstraint::FoldIn { .. }
        ) {
            continue;
        }
        let callable = crate::unify::collection_contract_callable_type(relation);
        let mut outside = UnordSet::new();
        if !outside_callable(&scheme.body, &callable, &mut outside) {
            continue;
        }
        // The scheme can own several checked collection relations whose
        // variables unify. A variable carried by another relation is not
        // selected solely by this key operation, even when that other
        // callable is absent from the scheme body's visible shape.
        for other in &scheme.constraints {
            if std::ptr::eq(other, relation) {
                continue;
            }
            for carried in other.carried_types() {
                outside.extend(free_tvars(carried));
            }
        }
        for var in free_tvars(&callable) {
            if scheme.tvars.contains(&var) && !outside.contains(&var) {
                closed.insert(var);
            }
        }
    }
    closed
}

/// chelis#1654: what one generalization decided about the pending collection
/// obligations. See [`Env::collection_constraints_to_quantify`].
#[derive(Default)]
struct CollectionQuantification {
    /// Obligations this scheme now owns, in ledger order.
    constraints: Vec<CollectionConstraint>,
    /// Exact ledger entries either moved into the scheme or discarded with a
    /// function-bearing subvalue no longer visible in the generalized type.
    ledger_removals: Vec<crate::unify::CollectionContractId>,
    /// Body variables that must stay monomorphic because an obligation this
    /// scheme does NOT own still carries them.
    pinned: UnordSet<TypeVar>,
}

/// True when `ty` contains a tensor whose shape carries the named
/// dimension `name` (chelis#397/#469). Walks the same compound-type
/// structure as the free-var collectors above.
fn type_carries_dim_name(ty: &Type, name: &str) -> bool {
    match ty {
        Type::Tensor(dims, _) => dims.iter().any(|d| matches!(d, Dim::Name(n) if n == name)),
        Type::Fn(args, ret) => {
            args.iter().any(|a| type_carries_dim_name(a, name)) || type_carries_dim_name(ret, name)
        }
        Type::Ref(inner) => type_carries_dim_name(inner, name),
        Type::Adt(_, args) | Type::Tuple(args) => {
            args.iter().any(|a| type_carries_dim_name(a, name))
        }
        Type::KindedAdt(_, args) => args.iter().any(|argument| match argument {
            NominalArg::Type(ty) => type_carries_dim_name(ty, name),
            NominalArg::Dimension(Dim::Name(found)) => found == name,
            NominalArg::Dimension(_) => false,
        }),
        _ => false,
    }
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

/// Collect all free type variables in a type.
pub fn free_tvars(ty: &Type) -> Vec<TypeVar> {
    let mut vars = Vec::new();
    collect_tvars(ty, &mut vars);
    vars.sort_by_key(|v| v.0);
    vars.dedup_by_key(|v| v.0);
    vars
}

fn collect_tvars(ty: &Type, vars: &mut Vec<TypeVar>) {
    match ty {
        Type::Var(v) => vars.push(*v),
        Type::Fn(args, ret) => {
            for a in args {
                collect_tvars(a, vars);
            }
            collect_tvars(ret, vars);
        }
        Type::Ref(inner) => collect_tvars(inner, vars),
        Type::Adt(_, args) => {
            for a in args {
                collect_tvars(a, vars);
            }
        }
        Type::KindedAdt(_, args) => {
            for argument in args {
                if let NominalArg::Type(ty) = argument {
                    collect_tvars(ty, vars);
                }
            }
        }
        Type::Tuple(ts) => {
            for t in ts {
                collect_tvars(t, vars);
            }
        }
        // WS-A5 (spec/04-type-system.md §5.8): the tensor precision
        // slot may carry a TypeVar (precision polymorphism). The slot
        // must participate in free-var collection so generalization
        // can pin precision-quantified vars in the resulting Scheme;
        // missing this means each call site reuses the SAME var
        // across instantiations and the second call's precision
        // collides with the first call's binding.
        Type::Tensor(_, prec) => {
            if let TensorPrec::Var(v) = prec {
                vars.push(*v);
            }
        }
        Type::Prim(_) | Type::Unit | Type::Error(_) => {}
    }
}

#[cfg(test)]
mod module_scope_tests {
    use super::*;

    #[test]
    fn exact_lookup_does_not_promote_a_unique_terminal_match() {
        let mut env = Env::new();
        env.bind(
            "pkg__demo__Provider__borrowed".to_string(),
            Scheme::mono(Type::Prim(Prim::Int64)),
        );

        assert!(env.lookup("borrowed").is_none());
        assert!(env.lookup_terminal_unique("borrowed").is_some());
    }

    #[test]
    fn same_level_monomorphic_contracts_keep_distinct_declaration_owners() {
        let env = Env::new();
        let mut var_gen = VarGen::default();
        let mut subst = Subst::new();
        let list = Type::Adt(
            "List".to_string(),
            vec![Type::Tensor(
                vec![Dim::Lit(2), Dim::Lit(3)],
                TensorPrec::Concrete(Prim::F32),
            )],
        );
        let result = Type::Tensor(
            vec![Dim::Wildcard, Dim::Wildcard],
            TensorPrec::Concrete(Prim::F32),
        );
        let constraint = CollectionConstraint::Concat {
            lhs: list.clone(),
            rhs: Type::Prim(Prim::Int32),
            result: result.clone(),
        };
        let checked = Scheme {
            result_origin: None,
            tvars: Vec::new(),
            tvar_restrictions: Vec::new(),
            dvars: Vec::new(),
            rvars: Vec::new(),
            constraints: vec![constraint.clone()],
            body: Type::Fn(vec![list, Type::Prim(Prim::Int32)], Box::new(result)),
        };

        let component = subst.enter_level(&var_gen);
        let first_mark = subst.collection_contract_mark();
        let first_ty = env.instantiate(&checked, &mut var_gen, &subst);
        let first_ids = subst.collection_contract_ids_since(first_mark);
        let second_mark = subst.collection_contract_mark();
        let second_ty = env.instantiate(&checked, &mut var_gen, &subst);
        let second_ids = subst.collection_contract_ids_since(second_mark);
        subst.leave_level(component, &var_gen);

        let first = env.generalize_with_collection_contracts(&first_ty, &subst, &first_ids);
        assert_eq!(first.constraints, vec![constraint.clone()]);
        assert_eq!(
            subst.pending_collection_contracts().len(),
            1,
            "generalizing one recursive sibling must not absorb the other's contract"
        );

        let second = env.generalize_with_collection_contracts(&second_ty, &subst, &second_ids);
        assert_eq!(second.constraints, vec![constraint]);
        assert!(
            subst.pending_collection_contracts().is_empty(),
            "each sibling must consume exactly its own contract instance"
        );
    }

    #[test]
    fn authored_binders_stay_monomorphic_inside_their_declaration_then_generalize_at_boundary() {
        let mut env = Env::new();
        let mut var_gen = VarGen::default();
        let mut subst = Subst::new();

        let declaration = subst.enter_level(&var_gen);
        let authored_type = var_gen.fresh_tvar();
        let authored = var_gen.fresh_dvar();
        let authored_rank = var_gen.fresh_rvar();
        let mut identities = DeclarationBinderIdentities::default();
        identities
            .type_vars
            .insert("element".to_string(), authored_type);
        identities.dim_vars.insert("extent".to_string(), authored);
        identities
            .rank_vars
            .insert("shape".to_string(), authored_rank);
        env.set_type_resolution_scope(Some(&identities), None);
        let ty = Type::Tuple(vec![
            Type::Var(authored_type),
            Type::Tensor(
                vec![Dim::Var(authored), Dim::Rank(authored_rank)],
                TensorPrec::Concrete(Prim::F32),
            ),
        ]);

        let nested = subst.enter_level(&var_gen);
        let nested_type = var_gen.fresh_tvar();
        let nested_local = var_gen.fresh_dvar();
        let nested_rank = var_gen.fresh_rvar();
        subst.leave_level(nested, &var_gen);
        let nested_ty = Type::Tuple(vec![
            ty.clone(),
            Type::Var(nested_type),
            Type::Tensor(
                vec![Dim::Var(nested_local), Dim::Rank(nested_rank)],
                TensorPrec::Concrete(Prim::F32),
            ),
        ]);
        let nested_scheme = env.generalize(&nested_ty, &subst);
        assert_eq!(nested_scheme.tvars, vec![nested_type]);
        assert_eq!(
            nested_scheme.dvars,
            vec![nested_local],
            "a nested let may generalize its own dimension but not its declaration's authored binder"
        );
        assert_eq!(nested_scheme.rvars, vec![nested_rank]);

        subst.leave_level(declaration, &var_gen);
        env.set_type_resolution_scope(None, None);
        let declaration_scheme = env.generalize(&ty, &subst);
        assert_eq!(declaration_scheme.tvars, vec![authored_type]);
        assert_eq!(
            declaration_scheme.dvars,
            vec![authored],
            "the authored dimension becomes a quantifier only at its declaration boundary"
        );
        assert_eq!(declaration_scheme.rvars, vec![authored_rank]);
    }

    #[test]
    fn recursive_group_variable_is_monomorphic_until_group_close() {
        // [04-INF-5]: a provisional group hole is monomorphic inside its
        // group and can be quantified after closure if Γ does not own it.
        let mut var_gen = VarGen::default();
        let mut subst = Subst::new();
        let group = subst.enter_level(&var_gen);
        let body_var = var_gen.fresh_tvar();
        let body = Type::Fn(vec![Type::Var(body_var)], Box::new(Type::Var(body_var)));
        let mut captured = Env::new();
        captured.begin_group_level(subst.current_level());
        let in_group = captured.generalize(&body, &subst);
        assert!(in_group.tvars.is_empty());
        captured.end_holed_group();
        subst.leave_level(group, &var_gen);

        let published = captured.generalize(&body, &subst);
        assert_eq!(published.tvars, vec![body_var]);
        // Installing a monomorphic outer binding also lowers its free
        // variable to that binding's level, as ordinary inference does.
        subst.lower_type_to_current(&body);
        captured.bind("outer".to_string(), Scheme::mono(Type::Var(body_var)));
        let captured_scheme = captured.generalize(&body, &subst);
        assert!(captured_scheme.tvars.is_empty());
    }
}

/// Collect all free dimension variables in a type.
pub fn free_dvars(ty: &Type) -> Vec<DimVar> {
    let mut vars = Vec::new();
    collect_dvars(ty, &mut vars);
    vars.sort_by_key(|v| v.0);
    vars.dedup_by_key(|v| v.0);
    vars
}

fn collect_dvars(ty: &Type, vars: &mut Vec<DimVar>) {
    match ty {
        Type::Tensor(dims, _) => {
            for d in dims {
                if let Dim::Var(v) = d {
                    vars.push(*v);
                }
            }
        }
        Type::Fn(args, ret) => {
            for a in args {
                collect_dvars(a, vars);
            }
            collect_dvars(ret, vars);
        }
        Type::Ref(inner) => collect_dvars(inner, vars),
        Type::Adt(_, args) => {
            for a in args {
                collect_dvars(a, vars);
            }
        }
        Type::KindedAdt(_, args) => {
            for argument in args {
                match argument {
                    NominalArg::Type(ty) => collect_dvars(ty, vars),
                    NominalArg::Dimension(Dim::Var(var)) => vars.push(*var),
                    NominalArg::Dimension(_) => {}
                }
            }
        }
        Type::Tuple(ts) => {
            for t in ts {
                collect_dvars(t, vars);
            }
        }
        _ => {}
    }
}

/// Collect every dimension occurring in tensor positions of `ty`,
/// in traversal order, duplicates preserved. Used by the chelis#273
/// return-position rigidity guard to compare a return-only declared
/// dim parameter's resolution against the dims of the declared
/// parameter positions.
pub fn collect_dims(ty: &Type, dims: &mut Vec<Dim>) {
    match ty {
        Type::Tensor(ds, _) => {
            dims.extend(ds.iter().cloned());
        }
        Type::Fn(args, ret) => {
            for a in args {
                collect_dims(a, dims);
            }
            collect_dims(ret, dims);
        }
        Type::Ref(inner) => collect_dims(inner, dims),
        Type::Adt(_, args) => {
            for a in args {
                collect_dims(a, dims);
            }
        }
        Type::KindedAdt(_, args) => {
            for argument in args {
                match argument {
                    NominalArg::Type(ty) => collect_dims(ty, dims),
                    NominalArg::Dimension(dim) => dims.push(dim.clone()),
                }
            }
        }
        Type::Tuple(ts) => {
            for t in ts {
                collect_dims(t, dims);
            }
        }
        _ => {}
    }
}

/// Collect all free rank variables in a type (Tier-2 rank polymorphism).
pub fn free_rvars(ty: &Type) -> Vec<RankVar> {
    let mut vars = Vec::new();
    collect_rvars(ty, &mut vars);
    vars.sort_by_key(|v| v.0);
    vars.dedup_by_key(|v| v.0);
    vars
}

fn collect_rvars(ty: &Type, vars: &mut Vec<RankVar>) {
    match ty {
        Type::Tensor(dims, _) => {
            for d in dims {
                if let Dim::Rank(r) = d {
                    vars.push(*r);
                }
            }
        }
        Type::Fn(args, ret) => {
            for a in args {
                collect_rvars(a, vars);
            }
            collect_rvars(ret, vars);
        }
        Type::Ref(inner) => collect_rvars(inner, vars),
        Type::Adt(_, args) => {
            for a in args {
                collect_rvars(a, vars);
            }
        }
        Type::KindedAdt(_, args) => {
            for argument in args {
                match argument {
                    NominalArg::Type(ty) => collect_rvars(ty, vars),
                    NominalArg::Dimension(Dim::Rank(var)) => vars.push(*var),
                    NominalArg::Dimension(_) => {}
                }
            }
        }
        Type::Tuple(ts) => {
            for t in ts {
                collect_rvars(t, vars);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_key_variables_require_the_exact_owned_callable() {
        let operand = Type::Var(TypeVar(1));
        let result = Type::Var(TypeVar(2));
        let relation = CollectionConstraint::KeyFromSeed {
            operand: operand.clone(),
            result: result.clone(),
        };
        let callable = crate::unify::collection_contract_callable_type(&relation);
        let mut scheme = Scheme {
            tvars: vec![TypeVar(1), TypeVar(2)],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            constraints: vec![relation],
            result_origin: None,
            body: callable.clone(),
        };
        assert_eq!(
            closed_key_relation_variables(&scheme).into_sorted(),
            vec![TypeVar(1), TypeVar(2)]
        );

        // A function that also exposes the operation's result variable as
        // ordinary data has authored a generic key-carrying surface. The
        // checked relation grants no exception to that occurrence.
        scheme.body = Type::Tuple(vec![callable.clone(), result]);
        assert_eq!(
            closed_key_relation_variables(&scheme).into_sorted(),
            vec![TypeVar(1)]
        );
        scheme.constraints.clear();
        assert!(closed_key_relation_variables(&scheme).is_empty());
        scheme.body = callable;
        assert!(closed_key_relation_variables(&scheme).is_empty());
    }

    #[test]
    fn shared_collection_constraint_keeps_key_relation_variable_key_free() {
        let env = Env::new();
        let mut var_gen = VarGen::default();
        let mut subst = Subst::new();
        let level = subst.enter_level(&var_gen);
        let operand = var_gen.fresh_type();
        let result = var_gen.fresh_type();
        let key_relation = CollectionConstraint::SplitKey {
            operand: operand.clone(),
            result: result.clone(),
        };
        let list = Type::Adt("List".to_string(), vec![operand.clone()]);
        let append_relation = CollectionConstraint::Append {
            list: list.clone(),
            value: operand.clone(),
            result: list,
        };
        subst.record_collection_contract(key_relation.clone());
        subst.record_collection_contract(append_relation.clone());
        subst.leave_level(level, &var_gen);

        let scheme = env.generalize(
            &crate::unify::collection_contract_callable_type(&key_relation),
            &subst,
        );
        assert_eq!(scheme.constraints, vec![key_relation, append_relation]);
        let Type::Var(operand_var) = operand else {
            unreachable!()
        };
        let Type::Var(result_var) = result else {
            unreachable!()
        };
        assert_eq!(
            closed_key_relation_variables(&scheme).into_sorted(),
            vec![result_var],
            "a second owned constraint exposes the operand as an ordinary generic"
        );
        assert!(
            subst.key_free_origin(operand_var).is_some(),
            "[04-LIN-10] must forbid key instantiation of the shared operand"
        );
    }

    /// A scheme exercising every quantifier kind, with a restriction on one
    /// type variable and none on the other, so a route that drops
    /// restrictions and a route that installs them indiscriminately are both
    /// distinguishable from the correct one.
    fn restricted_scheme() -> Scheme {
        Scheme {
            result_origin: None,
            constraints: vec![],
            tvars: vec![TypeVar(1), TypeVar(2)],
            tvar_restrictions: vec![(TypeVar(2), TypeVarRestriction::ActiveFloat)],
            dvars: vec![DimVar(3), DimVar(4)],
            rvars: vec![RankVar(5)],
            body: Type::Tuple(vec![
                Type::Var(TypeVar(1)),
                Type::Tensor(
                    vec![
                        Dim::Var(DimVar(3)),
                        Dim::Var(DimVar(4)),
                        Dim::Rank(RankVar(5)),
                    ],
                    TensorPrec::Var(TypeVar(2)),
                ),
            ]),
        }
    }

    /// chelis#260: `instantiate_scheme` replaced a plain `instantiate` call
    /// at the annotated-def site — first for the dim mapping (Site 1) and
    /// then for the type mapping too (Site 2). It must therefore mint the
    /// SAME variables in the SAME order, or the substitution the checker runs
    /// on would change and these diagnostic-only fixes would perturb
    /// inference. Locking the equivalence rather than assuming it.
    ///
    /// The scheme carries a chelis#1292 restriction, so this also pins that
    /// the two jobs compose: naming a dim parameter must not cost the
    /// instantiation its quantified type-variable restrictions.
    #[test]
    fn dvar_mapping_instantiation_matches_plain_instantiation() {
        let scheme = restricted_scheme();
        let env = Env::new();

        let plain_subst = Subst::new();
        let mut plain_gen = VarGen::default();
        let plain = env.instantiate(&scheme, &mut plain_gen, &plain_subst);

        let mapped_subst = Subst::new();
        let mut mapped_gen = VarGen::default();
        let mapped = env.instantiate_scheme(&scheme, &mut mapped_gen, &mapped_subst);

        assert_eq!(
            plain, mapped.ty,
            "the mapping variant must produce an identical instantiated type"
        );
        assert_eq!(
            format!("{plain_gen:?}"),
            format!("{mapped_gen:?}"),
            "both variants must advance the generator identically"
        );
        assert_eq!(
            mapped
                .dvars
                .iter()
                .map(|(from, _)| *from)
                .collect::<Vec<_>>(),
            scheme.dvars,
            "the mapping must cover every quantified dim in quantifier order"
        );
        let fresh: Vec<DimVar> = mapped.dvars.iter().map(|(_, to)| *to).collect();
        assert!(
            fresh.iter().all(|f| !scheme.dvars.contains(f)),
            "every mapped-to variable must be fresh, got {fresh:?}"
        );
    }

    #[test]
    fn mapped_axis_metadata_follows_the_fresh_scheme_dimension() {
        let quantified = DimVar(9);
        let quantified_rank = RankVar(10);
        let scheme = Scheme {
            result_origin: None,
            constraints: vec![],
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![quantified],
            rvars: vec![quantified_rank],
            body: Type::Tensor(
                vec![Dim::Rank(quantified_rank), Dim::Var(quantified)],
                TensorPrec::Concrete(Prim::F32),
            ),
        };
        let env = Env::new();
        let inference_subst = Subst::new();
        inference_subst.mark_mapped_axis(quantified, 1);
        let mut var_gen = VarGen::default();
        let instantiated = env.instantiate_scheme(&scheme, &mut var_gen, &inference_subst);
        let fresh = instantiated.dvars[0].1;

        assert_eq!(inference_subst.mapped_axis(fresh), Some(1));
        let Type::Tensor(dims, _) = instantiated.ty else {
            panic!("mapped scheme body must remain a tensor")
        };
        assert!(
            dims.contains(&Dim::Var(fresh)),
            "the instantiated body must carry the fresh mapped identity: {dims:?}"
        );
        assert!(
            !dims.contains(&Dim::Var(quantified)),
            "the quantified mapped identity must not escape instantiation: {dims:?}"
        );
    }

    /// chelis#1292 + chelis#260: quantified type-variable restrictions follow
    /// the fresh variables through EVERY instantiation route, not just the
    /// one #1292 happened to touch.
    ///
    /// This is the integration the two changes needed. Before they were
    /// reconciled, the dim-mapping route was a separate copy of the
    /// substitution loop that predated restriction installation, so a
    /// signature instantiated through it silently lost its `ActiveFloat`
    /// domain -- and the only symptom would have been a non-float program
    /// type-checking. A dropped restriction cannot fail loudly, so it is
    /// pinned here rather than left to a downstream rejection test.
    #[test]
    fn every_instantiation_route_installs_quantified_restrictions() {
        let scheme = restricted_scheme();
        let env = Env::new();

        // Every route instantiates the same scheme from the same starting
        // generator state, so all three mint the same fresh variables and the
        // observed restriction sets are directly comparable.
        let restrictions_after = |instantiate: &dyn Fn(&Subst, &mut VarGen)| {
            let subst = Subst::new();
            let mut var_gen = VarGen::default();
            instantiate(&subst, &mut var_gen);
            (0..64)
                .map(TypeVar)
                .filter_map(|v| subst.tvar_restriction(v).map(|r| (v, r)))
                .collect::<Vec<_>>()
        };

        let via_instantiate = restrictions_after(&|subst, var_gen| {
            env.instantiate(&scheme, var_gen, subst);
        });
        let via_tvar_mapping = restrictions_after(&|subst, var_gen| {
            env.instantiate_with_tvar_mapping(&scheme, var_gen, subst);
        });
        let via_instantiate_scheme = restrictions_after(&|subst, var_gen| {
            env.instantiate_scheme(&scheme, var_gen, subst);
        });

        assert_eq!(
            via_instantiate.len(),
            1,
            "exactly the restricted quantifier installs a restriction, got {via_instantiate:?}"
        );
        assert_eq!(
            via_instantiate[0].1,
            TypeVarRestriction::ActiveFloat,
            "the installed restriction must be the one the scheme declared"
        );
        assert_eq!(
            via_instantiate, via_tvar_mapping,
            "the tvar-mapping route must install the same restrictions as the plain route"
        );
        assert_eq!(
            via_instantiate, via_instantiate_scheme,
            "the instantiate_scheme route must install the same restrictions as the plain route"
        );
    }

    /// Signature occurrences preserve their scheme-instantiated identity,
    /// while body-only roles receive a declaration-owned completion.
    #[test]
    fn declaration_binder_identities_remap_and_complete_all_roles() {
        let mut env = Env::new();
        env.record_declared_binder_identities(
            "go",
            DeclarationBinderIdentities {
                type_vars: UnordMap::new(),
                dim_vars: UnordMap::from([
                    ("n".to_string(), DimVar(3)),
                    ("m".to_string(), DimVar(4)),
                ]),
                rank_vars: UnordMap::new(),
            },
        );
        let instantiation = InstantiatedScheme {
            ty: Type::Unit,
            tvars: vec![],
            dvars: vec![(DimVar(3), DimVar(90))],
            rvars: vec![],
        };
        let mut var_gen = VarGen::default();
        let resolved = env.declared_binder_identities_for_body(
            "go",
            &UnordSet::from(["m".to_string(), "n".to_string()]),
            Some(&instantiation),
            &mut var_gen,
        );
        assert_eq!(resolved.dim_vars.get("n"), Some(&DimVar(90)));
        assert_ne!(
            resolved.dim_vars.get("m"),
            Some(&DimVar(90)),
            "an unmapped signature identity must not borrow another name's mapping"
        );
        for name in ["m", "n"] {
            assert!(resolved.type_vars.contains_key(name));
            assert!(resolved.dim_vars.contains_key(name));
            assert!(resolved.rank_vars.contains_key(name));
        }
    }

    #[test]
    fn lexical_value_shadowing_does_not_replace_constructor_authority() {
        let mut env = Env::new();
        let constructor = Scheme::mono(Type::Fn(
            vec![Type::Prim(Prim::F64)],
            Box::new(Type::Adt("Box".to_string(), Vec::new())),
        ));
        env.bind_constructor("N".to_string(), "Box".to_string(), constructor.clone());
        env.bind_lexical("N".to_string(), Scheme::mono(Type::Prim(Prim::F64)));

        assert_eq!(
            env.lookup("N").map(|scheme| &scheme.body),
            Some(&Type::Prim(Prim::F64))
        );
        let (owner, active_constructor) = env
            .lookup_constructor("N")
            .expect("constructor binding survives lexical shadowing");
        assert_eq!(owner, "Box");
        assert_eq!(
            bincode::serialize(active_constructor).expect("serialize active constructor"),
            bincode::serialize(&constructor).expect("serialize expected constructor")
        );
    }

    #[test]
    fn active_constructor_identity_is_last_declaration_wins_and_serialized() {
        let mut env = Env::new();
        env.bind_constructor(
            "Some".to_string(),
            "Option".to_string(),
            Scheme::mono(Type::Adt("Option".to_string(), Vec::new())),
        );
        let wrapper_scheme = Scheme::mono(Type::Adt("Wrapper".to_string(), Vec::new()));
        env.bind_constructor(
            "Some".to_string(),
            "Wrapper".to_string(),
            wrapper_scheme.clone(),
        );

        let encoded = bincode::serialize(&env).expect("serialize env");
        let decoded: Env = bincode::deserialize(&encoded).expect("deserialize env");
        let (owner, active_constructor) = decoded
            .lookup_constructor("Some")
            .expect("serialized constructor authority");
        assert_eq!(owner, "Wrapper");
        assert_eq!(
            bincode::serialize(active_constructor).expect("serialize active constructor"),
            bincode::serialize(&wrapper_scheme).expect("serialize expected constructor")
        );
        assert_eq!(
            decoded
                .lookup_constructors("Some")
                .map(|(owner, _)| owner)
                .collect::<Vec<_>>(),
            vec!["Option", "Wrapper"],
            "shape-aware resolution needs every in-scope owner after serialization"
        );
    }

    #[test]
    fn free_variables_protect_quantified_ids_from_global_substitutions() {
        let quantified_type = TypeVar(10);
        let quantified_dim = DimVar(20);
        let quantified_rank = RankVar(30);
        let scheme = Scheme {
            result_origin: None,
            constraints: vec![],
            tvars: vec![quantified_type],
            tvar_restrictions: vec![],
            dvars: vec![quantified_dim],
            rvars: vec![quantified_rank],
            body: Type::Tuple(vec![
                Type::Var(quantified_type),
                Type::Tensor(
                    vec![Dim::Var(quantified_dim), Dim::Rank(quantified_rank)],
                    TensorPrec::Var(quantified_type),
                ),
            ]),
        };
        let mut env = Env::new();
        env.bind("generic".to_string(), scheme);
        let mut subst = Subst::new();
        subst
            .insert_type(quantified_type, Type::Prim(Prim::F32))
            .expect("unrestricted test substitution accepts f32");
        subst.insert_dim(quantified_dim, Dim::Lit(3));
        subst.insert_rank(quantified_rank, vec![Dim::Lit(4)]);

        assert!(env.free_tvars(&subst).is_empty());
        assert!(env.free_dvars(&subst).is_empty());
        assert!(env.free_rvars(&subst).is_empty());
    }

    #[test]
    fn free_variables_stop_alias_chains_at_quantified_ids() {
        let quantified_type = TypeVar(11);
        let quantified_dim = DimVar(21);
        let quantified_rank = RankVar(31);
        let outer_type = TypeVar(12);
        let outer_dim = DimVar(22);
        let outer_rank = RankVar(32);
        let scheme = Scheme {
            result_origin: None,
            constraints: vec![],
            tvars: vec![quantified_type],
            tvar_restrictions: vec![],
            dvars: vec![quantified_dim],
            rvars: vec![quantified_rank],
            body: Type::Tuple(vec![
                Type::Var(outer_type),
                Type::Tensor(
                    vec![Dim::Var(outer_dim), Dim::Rank(outer_rank)],
                    TensorPrec::Concrete(Prim::F32),
                ),
            ]),
        };
        let mut env = Env::new();
        env.bind("generic".to_string(), scheme);
        let mut subst = Subst::new();
        subst
            .insert_type(outer_type, Type::Var(quantified_type))
            .expect("unrestricted test substitution accepts an alias");
        subst
            .insert_type(quantified_type, Type::Prim(Prim::F64))
            .expect("unrestricted test substitution accepts f64");
        subst.insert_dim(outer_dim, Dim::Var(quantified_dim));
        subst.insert_dim(quantified_dim, Dim::Lit(5));
        subst.insert_rank(outer_rank, vec![Dim::Rank(quantified_rank)]);
        subst.insert_rank(quantified_rank, vec![Dim::Lit(6)]);

        assert!(env.free_tvars(&subst).is_empty());
        assert!(env.free_dvars(&subst).is_empty());
        assert!(env.free_rvars(&subst).is_empty());
    }

    #[test]
    fn free_variables_still_follow_unquantified_substitutions() {
        let source_type = TypeVar(40);
        let source_dim = DimVar(50);
        let source_rank = RankVar(60);
        let target_type = TypeVar(41);
        let target_dim = DimVar(51);
        let target_rank = RankVar(61);
        let scheme = Scheme {
            result_origin: None,
            constraints: vec![],
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Tuple(vec![
                Type::Var(source_type),
                Type::Tensor(
                    vec![Dim::Var(source_dim), Dim::Rank(source_rank)],
                    TensorPrec::Concrete(Prim::F32),
                ),
            ]),
        };
        let mut env = Env::new();
        env.bind("monomorphic".to_string(), scheme);
        let mut subst = Subst::new();
        subst
            .insert_type(source_type, Type::Var(target_type))
            .expect("unrestricted test substitution accepts an alias");
        subst.insert_dim(source_dim, Dim::Var(target_dim));
        subst.insert_rank(source_rank, vec![Dim::Rank(target_rank)]);

        assert_eq!(env.free_tvars(&subst), UnordSet::from([target_type]));
        assert_eq!(env.free_dvars(&subst), UnordSet::from([target_dim]));
        assert_eq!(env.free_rvars(&subst), UnordSet::from([target_rank]));
    }
}
