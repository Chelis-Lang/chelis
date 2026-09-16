//! Type environment: maps variable names to type schemes.

use chelis_unord::{UnordMap, UnordSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::unify::Subst;

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

/// Lexical type-variable scope for resolving source annotations during one
/// check. A top-level `defsig` owns the names; cloned [`Env`] values carry the
/// scope through nested `fn`/`let`/`match` inference and discard it when that
/// declaration's cloned environment is dropped.
///
/// This is deliberately check-time-only. It must never enter a serialized
/// [`crate::TypeEnv`], because a later stacked check owns a different set of
/// declarations and therefore a different lexical binder scope.
#[derive(Debug, Clone, Default)]
struct TypeResolutionScope {
    binders: Option<UnordSet<String>>,
    type_vars: UnordMap<String, TypeVar>,
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
    /// chelis#260: the source name bound to each declared dimension
    /// parameter of a signature, keyed by definition name.
    ///
    /// Recorded when a `defsig` is resolved, where the names are still in
    /// scope, and consumed after instantiation so a declared-dim diagnostic
    /// can say `n` and `m` rather than `d44` and `d45`. The `DimVar` keys are
    /// PRE-generalization; `instantiate_scheme` supplies the
    /// original-to-fresh hop that makes them comparable to what a check on an
    /// instantiated signature actually sees. Checker state only, never
    /// serialized.
    #[serde(skip)]
    declared_dim_names: UnordMap<String, UnordMap<DimVar, String>>,
    /// chelis#260 Site 2: the same provenance for TYPE parameters. Kept
    /// separate from `declared_dim_names` because the two are consumed by
    /// different diagnostics and a signature may declare either alone.
    ///
    /// chelis#1486 / [04-INF-6]: the second consumer. Membership is the
    /// checker's record of which variables in a declaration's scheme are
    /// AUTHORED binders rather than inference holes, so a hole ([04-INF-5])
    /// is absent here and is never subject to the rigidity check. Checker
    /// state only, never serialized.
    #[serde(skip)]
    declared_type_names: UnordMap<String, UnordMap<TypeVar, String>>,
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
/// variable may instantiate to any type; a quantified dimension variable
/// always instantiates to another dimension variable, so that renaming is
/// `DimVar` to `DimVar`.
type InstantiatedScheme = (Type, Vec<(TypeVar, Type)>, Vec<(DimVar, DimVar)>);

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
    pub(crate) fn set_type_resolution_binders(
        &mut self,
        binders: Option<&UnordSet<String>>,
        type_names: &UnordMap<TypeVar, String>,
    ) {
        self.type_resolution_scope.binders = binders.cloned();
        self.type_resolution_scope.type_vars = type_names
            .to_sorted()
            .into_iter()
            .map(|(var, name)| (name.clone(), *var))
            .collect();
    }

    /// Binder names visible to a nested source annotation in this lexical
    /// environment. Absence means closed input: named `t-var`/`d-var`/
    /// `d-rank` nodes do not allocate inference variables.
    pub(crate) fn type_resolution_binders(&self) -> Option<&UnordSet<String>> {
        self.type_resolution_scope.binders.as_ref()
    }

    pub(crate) fn type_resolution_variables(&self) -> &UnordMap<String, TypeVar> {
        &self.type_resolution_scope.type_vars
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
        self.rejected_signatures.remove(&name);
        self.bindings.insert(name, Arc::new(scheme));
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

    /// chelis#260: record the source names of a signature's declared dim
    /// parameters, so a later diagnostic on the instantiated signature can
    /// render them.
    pub(crate) fn record_declared_dim_names(
        &mut self,
        name: &str,
        names: UnordMap<DimVar, String>,
    ) {
        if !names.is_empty() {
            self.declared_dim_names.insert(name.to_string(), names);
        }
    }

    /// Resolve a definition's declared dim-parameter names against the fresh
    /// variables a given instantiation minted (chelis#260).
    ///
    /// `dvar_mapping` is the original-to-fresh pairing returned by
    /// [`Self::instantiate_scheme`]. The result is keyed by the
    /// FRESH variables, which is what a post-instantiation check reports on.
    /// An empty map means the names were never recorded; callers fall back to
    /// the internal id rather than inventing a name.
    pub(crate) fn declared_dim_names_for(
        &self,
        name: &str,
        dvar_mapping: &[(DimVar, DimVar)],
    ) -> UnordMap<DimVar, String> {
        let Some(original) = self.declared_dim_names.get(name) else {
            return UnordMap::new();
        };
        dvar_mapping
            .iter()
            .filter_map(|(from, to)| original.get(from).map(|n| (*to, n.clone())))
            .collect()
    }

    /// chelis#260 Site 2: record the source names of a signature's declared
    /// TYPE parameters, the analogue of [`Self::record_declared_dim_names`].
    ///
    /// chelis#1486 / [04-INF-6]: also the record of which variables are
    /// AUTHORED binders, so the post-body rigidity check can render them and
    /// an inference hole ([04-INF-5]) is excluded by construction.
    pub(crate) fn record_declared_type_names(
        &mut self,
        name: &str,
        names: UnordMap<TypeVar, String>,
    ) {
        if !names.is_empty() {
            self.declared_type_names.insert(name.to_string(), names);
        }
    }

    /// Resolve a definition's declared type-parameter names against the fresh
    /// variables a given instantiation minted (chelis#260 Site 2).
    ///
    /// `tvar_mapping` is the original-to-fresh pairing from
    /// [`Self::instantiate_scheme`]. It maps to a `Type` rather than a
    /// `TypeVar`, so a quantifier instantiated to anything but a bare
    /// variable simply has no fresh variable to name and is skipped: a
    /// concrete type renders itself and needs no provenance.
    ///
    /// An empty map means the declaration authored no type binder, so under
    /// [04-INF-6] nothing in it is rigid.
    pub(crate) fn declared_type_names_for(
        &self,
        name: &str,
        tvar_mapping: &[(TypeVar, Type)],
    ) -> UnordMap<TypeVar, String> {
        let Some(original) = self.declared_type_names.get(name) else {
            return UnordMap::new();
        };
        tvar_mapping
            .iter()
            .filter_map(|(from, to)| match to {
                Type::Var(fresh) => original.get(from).map(|n| (*fresh, n.clone())),
                _ => None,
            })
            .collect()
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

    /// Instantiate a scheme into the caller's inference substitution so
    /// quantified semantic restrictions follow the fresh variables.
    pub fn instantiate(
        &self,
        scheme: &Scheme,
        var_gen: &mut VarGen,
        inference_subst: &Subst,
    ) -> Type {
        self.instantiate_scheme(scheme, var_gen, inference_subst).0
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
        let (ty, tvar_mapping, _) = self.instantiate_scheme(scheme, var_gen, inference_subst);
        (ty, tvar_mapping)
    }

    /// The one instantiation mechanism (chelis#260 / chelis#1292).
    ///
    /// Every quantifier is renamed here and nowhere else, so the two jobs the
    /// callers above need cannot drift apart: #1292's installation of
    /// quantified type-variable restrictions onto the fresh variables, and
    /// #260's original-to-fresh dimension pairing that lets a diagnostic
    /// recover the source name of a declared dim parameter.
    ///
    /// Keeping them in one body is deliberate. Both were separately-authored
    /// copies of this loop at one point, and a second copy is exactly how a
    /// scheme gets instantiated with its restrictions dropped: the omission
    /// compiles, and the only symptom is a program that should have been
    /// rejected type-checking.
    ///
    /// Callable directly by a site that needs more than one of the renamings
    /// at once, which is why it is crate-visible rather than a further
    /// projection beside the ones above.
    pub(crate) fn instantiate_scheme(
        &self,
        scheme: &Scheme,
        var_gen: &mut VarGen,
        inference_subst: &Subst,
    ) -> InstantiatedScheme {
        let mut subst = Subst::new();
        let mut tvar_mapping = Vec::with_capacity(scheme.tvars.len());
        for &tv in &scheme.tvars {
            let fresh = var_gen.fresh_type();
            subst
                .insert_type(tv, fresh.clone())
                .expect("a fresh quantified type-variable renaming is valid");
            if let Type::Var(fresh_var) = fresh
                && let Some((_, restriction)) = scheme
                    .tvar_restrictions
                    .iter()
                    .find(|(restricted, _)| *restricted == tv)
            {
                inference_subst
                    .narrow_tvar_restriction(fresh_var, *restriction)
                    .expect("a fresh instantiation variable carries no prior dtype bound");
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
        for &dv in &scheme.dvars {
            // Mint the variable directly rather than destructuring
            // `fresh_dim()`: that is `Dim::Var(fresh_dvar())` today, but a
            // pattern match would silently drop the mapping entry (and the
            // name with it) if it ever returned another shape.
            let fresh_dv = var_gen.fresh_dvar();
            dvar_mapping.push((dv, fresh_dv));
            subst.insert_dim(dv, Dim::Var(fresh_dv));
            inference_subst.copy_dimension_label(dv, fresh_dv);
        }
        for &rv in &scheme.rvars {
            // Each rank var instantiates to a fresh sole-`Rank` shape so every
            // call site gets its own rank (Tier-2 rank polymorphism).
            subst.insert_rank(rv, vec![Dim::Rank(var_gen.fresh_rvar())]);
        }
        for constraint in &scheme.constraints {
            let renamed = constraint.map_types(|ty| subst.apply(ty));
            inference_subst.record_collection_contract(renamed);
        }
        (subst.apply(&scheme.body), tvar_mapping, dvar_mapping)
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
        self.generalize_owned(ty, subst, None)
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
        self.generalize_owned(ty, subst, Some(owned_contracts))
    }

    fn generalize_owned(
        &self,
        ty: &Type,
        subst: &Subst,
        owned_contracts: Option<&[crate::unify::CollectionContractId]>,
    ) -> Scheme {
        let (level_scheme, ledger_removals) = self.generalize_by_levels(ty, subst, owned_contracts);
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
        // chelis#1489: a variable still tied to a pending operand gate stays
        // monomorphic until the gate discharges; see
        // `Subst::pending_gate_result_vars`. Levels cannot see that tie -- it
        // lives in the gate ledger, not in any unification.
        let (pending_t, pending_d, pending_r) = subst.pending_gate_result_vars();
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
    /// for which of those may be quantified. Two exclusions are therefore
    /// mirrored here deliberately rather than inherited: a recursive group's
    /// instantiation variables (`tvar_pinned`) and, since chelis#1489, a
    /// pending operand gate's result variables. Both are properties of the
    /// inference state that no environment sweep can observe, so omitting
    /// either here would make the oracle disagree with a correct production
    /// path. Keep the two sets of exclusions in step.
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
        // chelis#1489: the same exclusion as `generalize_by_levels`, so the
        // parity assertion in `generalize` keeps comparing like with like.
        let (pending_t, pending_d, pending_r) = subst.pending_gate_result_vars();
        let generalizable = |v: TypeVar| {
            !env_tvars.contains(&v)
                && !crate::infer::recursion::tvar_pinned(v)
                && !pending_t.contains(&v)
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
                tvars,
                tvar_restrictions,
                dvars: free_dvars(&ty)
                    .into_iter()
                    .filter(|v| {
                        !env_dvars.contains(v)
                            && !pending_d.contains(v)
                            && matches!(subst.constraint_dim(&Dim::Var(*v)), Dim::Var(_))
                    })
                    .collect(),
                rvars: free_rvars(&ty)
                    .into_iter()
                    .filter(|v| !env_rvars.contains(v) && !pending_r.contains(v))
                    .collect(),
                constraints,
                body: ty,
            },
            split.ledger_removals,
        )
    }
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

    /// A scheme exercising every quantifier kind, with a restriction on one
    /// type variable and none on the other, so a route that drops
    /// restrictions and a route that installs them indiscriminately are both
    /// distinguishable from the correct one.
    fn restricted_scheme() -> Scheme {
        Scheme {
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
        let (mapped, _tvar_mapping, mapping) =
            env.instantiate_scheme(&scheme, &mut mapped_gen, &mapped_subst);

        assert_eq!(
            plain, mapped,
            "the mapping variant must produce an identical instantiated type"
        );
        assert_eq!(
            format!("{plain_gen:?}"),
            format!("{mapped_gen:?}"),
            "both variants must advance the generator identically"
        );
        assert_eq!(
            mapping.iter().map(|(from, _)| *from).collect::<Vec<_>>(),
            scheme.dvars,
            "the mapping must cover every quantified dim in quantifier order"
        );
        let fresh: Vec<DimVar> = mapping.iter().map(|(_, to)| *to).collect();
        assert!(
            fresh.iter().all(|f| !scheme.dvars.contains(f)),
            "every mapped-to variable must be fresh, got {fresh:?}"
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

    /// chelis#260: a name is only rendered when the mapping vouches for it.
    /// An unrecorded definition, or one whose recorded variables do not
    /// appear in this instantiation, must yield nothing rather than a name
    /// borrowed from another signature.
    #[test]
    fn declared_dim_names_resolve_only_through_the_mapping() {
        let mut env = Env::new();
        env.record_declared_dim_names(
            "go",
            UnordMap::from([(DimVar(3), "n".to_string()), (DimVar(4), "m".to_string())]),
        );

        let resolved = env.declared_dim_names_for("go", &[(DimVar(3), DimVar(90))]);
        assert_eq!(resolved.get(&DimVar(90)).map(String::as_str), Some("n"));
        assert_eq!(resolved.len(), 1, "only mapped variables are named");

        assert!(
            env.declared_dim_names_for("absent", &[(DimVar(3), DimVar(90))])
                .is_empty(),
            "an unrecorded definition names nothing"
        );
        assert!(
            env.declared_dim_names_for("go", &[]).is_empty(),
            "an empty instantiation mapping names nothing"
        );
        assert!(
            env.declared_dim_names_for("go", &[(DimVar(77), DimVar(91))])
                .is_empty(),
            "a variable this signature never declared names nothing"
        );
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
