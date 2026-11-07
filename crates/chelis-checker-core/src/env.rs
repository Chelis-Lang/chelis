//! Type environment: maps variable names to type schemes.

use crate::prelude::*;
use alloc::sync::Arc;
use chelis_unord::{PersistentMap, PersistentSet, UnordMap, UnordSet};

use crate::errors::ErrorWitness;
use crate::session::DeclarationDiagnosticOwner;
use chelis_type_core::*;

#[cfg(test)]
use chelis_numeric_core::prim::Prim;

#[cfg(feature = "generalize-sweep-oracle")]
fn note_generalize_sweep_env_visit(operation: &crate::resources::Operation<'_>) {
    operation.sweep_visits.set(operation.sweep_visits.get() + 1);
}

#[cfg(all(test, feature = "generalize-sweep-oracle"))]
pub(crate) fn without_generalize_sweep_oracle<R>(
    f: impl FnOnce() -> R,
    operation: &crate::resources::Operation<'_>,
) -> R {
    struct Restore<'a>(&'a core::cell::Cell<bool>, bool);
    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            self.0.set(self.1);
        }
    }
    let previous = operation.sweep_enabled.replace(false);
    let _restore = Restore(&operation.sweep_enabled, previous);
    f()
}

#[cfg(all(test, feature = "generalize-sweep-oracle"))]
pub(crate) fn reset_generalize_sweep_env_visits(operation: &crate::resources::Operation<'_>) {
    operation.sweep_visits.set(0);
}
#[cfg(all(test, feature = "generalize-sweep-oracle"))]
pub(crate) fn generalize_sweep_env_visits(operation: &crate::resources::Operation<'_>) -> usize {
    operation.sweep_visits.get()
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
    /// Listed names the signature uses only as a type or rank binder. Body
    /// completion still gives each one a dimension identity, but the name is
    /// not a dimension of the definition (spec/04-type-system.md section
    /// 4.7.2 reads a size name as a dimension, never as a precision or rank).
    pub(crate) non_dimension_binders: UnordSet<String>,
}

impl DeclarationBinderIdentities {
    /// Whether `name` is a dimension binder of the declaration: a listed name
    /// the signature uses as a dimension, or uses in no role at all.
    pub(crate) fn binds_dimension(&self, name: &str) -> bool {
        self.dim_vars.contains_key(name) && !self.non_dimension_binders.contains(name)
    }

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
            if !self.dim_vars.contains_key(name)
                && (self.type_vars.contains_key(name) || self.rank_vars.contains_key(name))
            {
                self.non_dimension_binders.insert(name.clone());
            }
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
#[derive(Debug, Clone)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
struct ConstructorBinding {
    owner: String,
    scheme: Scheme<ErrorWitness>,
}

/// Private structure used only to check the body of a rejected declaration.
#[derive(Debug, Clone)]
pub(crate) struct RejectedSignature {
    pub(crate) scheme: Scheme<ErrorWitness>,
    pub(crate) witness: crate::errors::ErrorWitness,
}

/// What the checker established about the value one binding was bound to,
/// beyond its type. It lives on the binding entry, so every new binding of
/// the name replaces it, and no binder can leave a stale fact visible under
/// a name it shadows (chelis#469).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BindingFacts {
    /// The exact value of a binding whose right-hand side folds to a checked
    /// integer constant, so an `expand`/`insert` size naming it types a
    /// literal extent.
    pub(crate) static_size: Option<i64>,
    /// chelis#631: the element count of a binding to a list literal, so
    /// `concat(rows, axis)` can size its concat axis through the binding (a
    /// list's length is not part of its type).
    pub(crate) list_literal_len: Option<usize>,
}

/// One value binding: its scheme and the check-time facts about its value.
/// Serialized as the scheme alone; the facts are an analysis artifact of
/// one check and never enter a cached environment.
#[derive(Debug, Clone)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(any(test, feature = "serde"), serde(transparent))]
struct ValueBinding {
    scheme: Arc<Scheme<ErrorWitness>>,
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    facts: BindingFacts,
}

/// Type environment (Γ): maps names to polymorphic type schemes.
#[derive(Debug, Clone, Default)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Env {
    // Schemes are immutable once bound. Lexical snapshots copy the name map,
    // while sharing signature bodies until a scope replaces its own binding.
    bindings: PersistentMap<String, ValueBinding>,
    /// Active constructor bindings, separate from ordinary value lookup.
    ///
    /// Every exact owner remains available so constructor syntax can select by
    /// positional versus record shape without consulting out-of-scope registry
    /// entries. Declaration/import order still chooses the active owner when
    /// several candidates have the same shape, matching the value environment's
    /// established constructor binding. A later lexical parameter or block
    /// binding may replace `bindings[name]` for bare value position without
    /// changing constructor position.
    #[cfg_attr(any(test, feature = "serde"), serde(default))]
    constructor_bindings: PersistentMap<String, Vec<ConstructorBinding>>,
    /// Names introduced by the current lexical scope (function parameters,
    /// block bindings, and pattern bindings). Builtin-specific inference may
    /// only dispatch on a name from the closed builtin vocabulary when that
    /// spelling has not been replaced by one of these bindings. This is
    /// check-time provenance, not part of the reusable or serialized type
    /// environment.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    lexical_bindings: PersistentSet<String>,
    /// Current declaration's type/dimension/rank binders. Installed only on
    /// the cloned environment used to infer that declaration, inherited by
    /// nested lexical clones, and omitted from cached checker state.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    type_resolution_scope: TypeResolutionScope,
    /// Rejected declarations still have usable body constraints. These frames
    /// belong to this check only and never become cached/public signatures.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    rejected_signatures: PersistentMap<String, Arc<RejectedSignature>>,
    /// Declared result trusted only while checking a linker-reserved
    /// [05-OP-35] stdlib wrapper whose runtime-axis shape proof is owned by
    /// #1298. Never serialized or exposed to entry source.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    exact_stdlib_expected_result: Option<Type<ErrorWitness>>,
    /// Checker identities introduced while each `defsig` was resolved, keyed
    /// by definition name.
    ///
    /// The stored identities are PRE-generalization. `instantiate_scheme`
    /// supplies the original-to-fresh hop for all three variable kinds, and
    /// body setup completes names absent from the outer signature. Keeping
    /// type/dimension/rank in one object prevents an ordinary annotation from
    /// receiving lexical permission without declaration-owned identity.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    declared_binder_identities: PersistentMap<String, DeclarationBinderIdentities>,
    /// Members of the recursive group being inferred whose declared header
    /// omits a type, each with the provisional scheme
    /// [`Self::bind_holed_group_member`] bound it at, shared with its binding
    /// so a lexical snapshot does not copy it.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    holed_group_members: UnordMap<String, Arc<Scheme<ErrorWitness>>>,
    /// The declared dtype-family bound of each binder variable those members
    /// share, read from the header when it was bound. The group's completion
    /// can identify the variable with a sibling's copy of it before the
    /// member's contract is decided, and the binding then carries the merged
    /// family, not the declared one.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    holed_declared_bounds: UnordMap<TypeVar, Option<TypeVarRestriction>>,
    /// The inference level of the recursive group being inferred.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    group_level: Option<u32>,
    /// The composed `fresh TypeVar -> source name` map for the definition
    /// currently being inferred.
    ///
    /// The borrow diagnostic that needs it (`validate_deferred_borrow_vars`)
    /// runs at the per-def drain, after body inference, and never sees the
    /// instantiation that minted the fresh variables. Composing at the
    /// instantiation site and parking the result here is what carries a
    /// source name across that gap.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    active_declared_type_names: UnordMap<TypeVar, String>,
    /// The dtype-family bound each of those binders was AUTHORED with, read
    /// before the body is inferred (`None` for an unbounded binder).
    ///
    /// [04-INF-6] quantifies a binder over the instantiations its declaration
    /// admits, so a body rule that must hold at every instantiation (a literal
    /// pattern's, chelis#2442) reads this snapshot rather than the variable's
    /// current restriction, which a body constraint may already have narrowed.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    active_declared_type_bounds: UnordMap<TypeVar, Option<TypeVarRestriction>>,
    /// chelis#1134 / [04-INF-4]: flattened declaration index of every
    /// top-level eager (non-function) value in the unit being checked.
    ///
    /// Visibility of such a value is a function of its source position, and
    /// of nothing else. Binding presence cannot express that rule: the body
    /// inference schedule reorders function declarations, so "is it bound
    /// yet" and "is it declared yet" are different questions. This map
    /// answers the second one for [`Self::top_level_value_visibility`], which
    /// is the only authority on eager-value scope.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    top_level_value_ordinals: PersistentMap<String, usize>,
    /// The binding an eager top-level value shadows, when its name was
    /// already bound by an import or a stacked library phase. Before that
    /// value's own declaration the outer binding is still the one in scope,
    /// so lookup falls back here rather than reporting the name unbound.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    shadowed_prior_bindings: PersistentMap<String, Scheme<ErrorWitness>>,
    /// Flattened declaration index currently being inferred. Set by the
    /// driver loop at each scheduled declaration and inherited by every
    /// lexical clone, so nested scopes compare against the declaration that
    /// owns them rather than against the schedule's position.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    current_declaration_ordinal: Option<usize>,
    /// Exact names temporarily visible while one cyclic full-reference
    /// component is co-inferred. This contains the component's provisional
    /// members and, for [04-INF-8] cycle precedence, any later eager target
    /// that the scheduler has already inferred for that component. The
    /// inference driver installs this capability only for the rejected cycle
    /// and restores the prior set on both completion and cancellation; it is
    /// never serialized.
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
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
    NotYetDeclared {
        shadowed: Option<&'a Scheme<ErrorWitness>>,
    },
}

impl Env {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every bound value name, in sorted order.
    pub fn sorted_names(&self) -> Vec<&str> {
        self.bindings
            .to_sorted()
            .into_iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Look up a name. Returns None if unbound.
    pub fn lookup(&self, name: &str) -> Option<&Scheme<ErrorWitness>> {
        self.bindings
            .get(name)
            .map(|binding| binding.scheme.as_ref())
    }

    /// Look up the active constructor owner and scheme for an exact name.
    pub(crate) fn lookup_constructor(&self, name: &str) -> Option<(&str, &Scheme<ErrorWitness>)> {
        self.constructor_bindings
            .get(name)
            .and_then(|bindings| bindings.last())
            .map(|binding| (binding.owner.as_str(), &binding.scheme))
    }

    /// Iterate every in-scope exact constructor candidate in declaration order.
    pub(crate) fn lookup_constructors(
        &self,
        name: &str,
    ) -> impl DoubleEndedIterator<Item = (&str, &Scheme<ErrorWitness>)> {
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

    pub(crate) fn set_exact_stdlib_expected_result(&mut self, result: Option<Type<ErrorWitness>>) {
        self.exact_stdlib_expected_result = result;
    }

    pub(crate) fn exact_stdlib_expected_result(&self) -> Option<&Type<ErrorWitness>> {
        self.exact_stdlib_expected_result.as_ref()
    }

    /// The facts recorded on the binding `name` resolves to here. A name
    /// that is unbound, or that resolves to a top-level value declared after
    /// the current declaration ([04-INF-4]), carries none.
    fn binding_facts(&self, name: &str) -> BindingFacts {
        match self.top_level_value_visibility(name) {
            TopLevelValueVisibility::Visible => self
                .bindings
                .get(name)
                .map(|binding| binding.facts)
                .unwrap_or_default(),
            TopLevelValueVisibility::NotYetDeclared { .. } => BindingFacts::default(),
        }
    }

    /// Exact checked value of the folded extent binding `name` resolves to.
    pub fn static_size_value(&self, name: &str) -> Option<i64> {
        self.binding_facts(name).static_size
    }

    /// The list-literal length of the binding `name` resolves to, if any
    /// (chelis#631).
    pub fn list_literal_len(&self, name: &str) -> Option<usize> {
        self.binding_facts(name).list_literal_len
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
            .any(|(_, binding)| type_carries_dim_name(&binding.scheme.body, name))
    }

    pub(crate) fn tensor_carries_dim_with_subst(
        &self,
        name: &str,
        subst: &Subst<ErrorWitness>,
    ) -> bool {
        self.bindings.to_sorted().into_iter().any(|(_, binding)| {
            type_carries_dim_name(&subst.semantic_type(&binding.scheme.body), name)
        })
    }

    /// Look up an imported or qualified name by its unique terminal segment.
    pub fn lookup_terminal_unique(&self, name: &str) -> Option<&Scheme<ErrorWitness>> {
        let mut matches = self
            .bindings
            .to_sorted()
            .into_iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first.scheme.as_ref())
    }

    /// Extend the environment with a new binding.
    pub fn bind(&mut self, name: String, scheme: Scheme<ErrorWitness>) {
        self.bind_shared(name, Arc::new(scheme));
    }

    /// Extend the environment with a binding whose value carries `facts`.
    pub(crate) fn bind_with_facts(
        &mut self,
        name: String,
        scheme: Scheme<ErrorWitness>,
        facts: BindingFacts,
    ) {
        self.insert_binding(name, Arc::new(scheme), facts);
    }

    fn bind_shared(&mut self, name: String, scheme: Arc<Scheme<ErrorWitness>>) {
        self.insert_binding(name, scheme, BindingFacts::default());
    }

    /// The one way a value binding enters the environment. The new entry
    /// replaces the name's previous entry together with its facts.
    fn insert_binding(
        &mut self,
        name: String,
        scheme: Arc<Scheme<ErrorWitness>>,
        facts: BindingFacts,
    ) {
        self.rejected_signatures.remove(&name);
        self.bindings.insert(name, ValueBinding { scheme, facts });
    }

    pub(crate) fn bind_rejected_signature(
        &mut self,
        name: String,
        recovery: crate::deep_type::RejectedSignatureType,
        subst: &mut Subst<ErrorWitness>,
        operation: &crate::resources::Operation<'_>,
    ) {
        let scheme = self.generalize_in_operation(&recovery.ty, subst, operation);
        self.bind(
            name.clone(),
            Scheme::<ErrorWitness>::mono(crate::errors::propagate(&recovery.witness)),
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
    pub(crate) fn bind_constructor(
        &mut self,
        name: String,
        owner: String,
        scheme: Scheme<ErrorWitness>,
    ) {
        self.insert_binding(
            name.clone(),
            Arc::new(scheme.clone()),
            BindingFacts::default(),
        );
        let candidates = self.constructor_bindings.get_or_insert_with(name, Vec::new);
        candidates.retain(|candidate| candidate.owner != owner);
        candidates.push(ConstructorBinding { owner, scheme });
    }

    /// Extend the environment with a binding introduced by ordinary lexical
    /// scope. Unlike [`Self::bind`], this also records that builtin callable
    /// dispatch must not claim the name while this environment lives.
    pub(crate) fn bind_lexical(&mut self, name: String, scheme: Scheme<ErrorWitness>) {
        self.bind_lexical_with_facts(name, scheme, BindingFacts::default());
    }

    /// [`Self::bind_lexical`] for a binding whose value carries `facts`.
    pub(crate) fn bind_lexical_with_facts(
        &mut self,
        name: String,
        scheme: Scheme<ErrorWitness>,
        facts: BindingFacts,
    ) {
        self.lexical_bindings.insert(name.clone());
        self.bind_with_facts(name, scheme, facts);
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
        shadowed: Option<Scheme<ErrorWitness>>,
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
        core::mem::replace(&mut self.active_top_level_component, names)
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
        inference_subst: &Subst<ErrorWitness>,
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
        let shared = instantiate_scheme(
            &Scheme {
                result_origin: None,
                constraints: Vec::new(),
                ..header.clone()
            },
            var_gen,
            inference_subst,
        );
        let mut renaming = Subst::<ErrorWitness>::new();
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
                Type::<ErrorWitness>::Var(fresh) => Some(*fresh),
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
                if let Some((_, Type::<ErrorWitness>::Var(fresh))) =
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
    pub(crate) fn is_holed_group_reference(
        &self,
        name: &str,
        scheme: &Scheme<ErrorWitness>,
    ) -> bool {
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
        subst: &Subst<ErrorWitness>,
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
        instantiation: Option<&InstantiatedScheme<ErrorWitness>>,
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
                if let Some(Type::<ErrorWitness>::Var(fresh)) = instantiation
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
        subst: &Subst<ErrorWitness>,
    ) -> Option<(&str, Option<TypeVarRestriction>)> {
        self.active_declared_type_names
            .to_sorted()
            .into_iter()
            .find(|(declared, _)| {
                subst.apply(&Type::<ErrorWitness>::Var(**declared))
                    == Type::<ErrorWitness>::Var(var)
            })
            .map(|(declared, name)| {
                let bound = self
                    .active_declared_type_bounds
                    .get(declared)
                    .copied()
                    .flatten();
                (name.as_str(), bound)
            })
    }

    /// Collect all free type variables across all bindings in the environment.
    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub fn free_tvars<S: chelis_stack_services::StackServices>(
        &self,
        subst: &Subst<ErrorWitness>,
        resources: &crate::Resources<'_, S>,
    ) -> UnordSet<TypeVar> {
        resources.run_borrowed(|operation| self.free_tvars_in_operation(subst, operation))
    }

    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub(crate) fn free_tvars_in_operation(
        &self,
        subst: &Subst<ErrorWitness>,
        _operation: &crate::resources::Operation<'_>,
    ) -> UnordSet<TypeVar> {
        chelis_type_core::free_scheme_tvars(
            self.bindings.to_sorted().into_iter().map(|(_, binding)| {
                #[cfg(feature = "generalize-sweep-oracle")]
                note_generalize_sweep_env_visit(_operation);
                binding.scheme.as_ref()
            }),
            subst,
        )
    }

    /// Collect all free dimension variables across all bindings in the environment.
    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub fn free_dvars<S: chelis_stack_services::StackServices>(
        &self,
        subst: &Subst<ErrorWitness>,
        resources: &crate::Resources<'_, S>,
    ) -> UnordSet<DimVar> {
        resources.run_borrowed(|operation| self.free_dvars_in_operation(subst, operation))
    }

    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub(crate) fn free_dvars_in_operation(
        &self,
        subst: &Subst<ErrorWitness>,
        _operation: &crate::resources::Operation<'_>,
    ) -> UnordSet<DimVar> {
        chelis_type_core::free_scheme_dvars(
            self.bindings.to_sorted().into_iter().map(|(_, binding)| {
                #[cfg(feature = "generalize-sweep-oracle")]
                note_generalize_sweep_env_visit(_operation);
                binding.scheme.as_ref()
            }),
            subst,
        )
    }

    /// Free rank variables in the environment (Tier-2 rank polymorphism).
    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub fn free_rvars<S: chelis_stack_services::StackServices>(
        &self,
        subst: &Subst<ErrorWitness>,
        resources: &crate::Resources<'_, S>,
    ) -> UnordSet<RankVar> {
        resources.run_borrowed(|operation| self.free_rvars_in_operation(subst, operation))
    }

    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub(crate) fn free_rvars_in_operation(
        &self,
        subst: &Subst<ErrorWitness>,
        _operation: &crate::resources::Operation<'_>,
    ) -> UnordSet<RankVar> {
        chelis_type_core::free_scheme_rvars(
            self.bindings.to_sorted().into_iter().map(|(_, binding)| {
                #[cfg(feature = "generalize-sweep-oracle")]
                note_generalize_sweep_env_visit(_operation);
                binding.scheme.as_ref()
            }),
            subst,
        )
    }

    /// Generalize a type over variables not free in the environment.
    pub fn generalize<S: chelis_stack_services::StackServices>(
        &self,
        ty: &Type<ErrorWitness>,
        subst: &Subst<ErrorWitness>,
        resources: &crate::Resources<'_, S>,
    ) -> Scheme<ErrorWitness> {
        resources.run_borrowed(|operation| self.generalize_in_operation(ty, subst, operation))
    }

    pub(crate) fn generalize_in_operation(
        &self,
        ty: &Type<ErrorWitness>,
        subst: &Subst<ErrorWitness>,
        operation: &crate::resources::Operation<'_>,
    ) -> Scheme<ErrorWitness> {
        self.generalize_owned(ty, subst, None, &[], operation)
    }

    /// Generalize one deferred declaration using only the contract instances
    /// created while that declaration was inferred.
    ///
    /// Recursive siblings share one inference level, so level membership alone
    /// cannot distinguish two fully monomorphic checked function values. The
    /// driver records each member's exact instance IDs and supplies them here.
    pub(crate) fn generalize_with_collection_contracts(
        &self,
        ty: &Type<ErrorWitness>,
        subst: &Subst<ErrorWitness>,
        owned_contracts: &[chelis_type_core::CollectionContractId],
        operation: &crate::resources::Operation<'_>,
    ) -> Scheme<ErrorWitness> {
        self.generalize_owned(ty, subst, Some(owned_contracts), &[], operation)
    }

    /// Generalize the raw type and its deferred equalities as one scoped
    /// graph. An equality component containing a shared variable stays shared;
    /// a fresh instantiation must never copy only part of that component.
    pub(crate) fn generalize_with_result_constraints(
        &self,
        ty: &Type<ErrorWitness>,
        subst: &Subst<ErrorWitness>,
        owned_contracts: Option<&[chelis_type_core::CollectionContractId]>,
        equations: &[ResultConstraint<ErrorWitness>],
        operation: &crate::resources::Operation<'_>,
    ) -> Scheme<ErrorWitness> {
        self.generalize_owned(ty, subst, owned_contracts, equations, operation)
    }

    fn generalize_owned(
        &self,
        ty: &Type<ErrorWitness>,
        subst: &Subst<ErrorWitness>,
        owned_contracts: Option<&[chelis_type_core::CollectionContractId]>,
        equations: &[ResultConstraint<ErrorWitness>],
        operation: &crate::resources::Operation<'_>,
    ) -> Scheme<ErrorWitness> {
        #[cfg(feature = "generalize-sweep-oracle")]
        let reference = (operation.sweep_enabled.get()).then(|| {
            let (mut scheme, _) = self.generalize_by_sweep(ty, subst, owned_contracts, operation);
            ResultScope::new(equations).retain_closed_quantifiers(&mut scheme);
            scheme
        });
        let scheme = crate::infer::recursion::with_pinned_tvars(
            |excluded| {
                chelis_type_core::generalize(ty, subst, excluded, owned_contracts, equations)
            },
            operation,
        );
        #[cfg(feature = "generalize-sweep-oracle")]
        if let Some(reference) = reference {
            assert_eq!(
                scheme.constraints, reference.constraints,
                "level-based collection obligations diverged from the reference environment sweep"
            );
            assert_eq!(
                scheme.tvars, reference.tvars,
                "level-based type quantifiers diverged from the reference environment sweep"
            );
            assert_eq!(
                scheme.tvar_restrictions, reference.tvar_restrictions,
                "level-based type-variable restrictions diverged from the reference environment sweep"
            );
            assert_eq!(
                scheme.dvars, reference.dvars,
                "level-based dimension quantifiers diverged from the reference environment sweep"
            );
            assert_eq!(
                scheme.rvars, reference.rvars,
                "level-based rank quantifiers diverged from the reference environment sweep"
            );
            assert_eq!(
                scheme.body, reference.body,
                "level-based scheme body diverged from the reference environment sweep"
            );
        }
        scheme
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
    ///
    /// Both results are compared after the same deferred-equality scope
    /// closure. A variable that only a carried equality mentions can be older
    /// than the binding without being in Γ: instantiating a let-bound value
    /// in an enclosing scrutinee mints its join intermediates there. Levels
    /// keep such a variable; a sweep cannot see its age. Its component also
    /// reaches the Γ variable the scrutinee binds, so the closure keeps it
    /// shared under both.
    #[cfg(feature = "generalize-sweep-oracle")]
    fn generalize_by_sweep(
        &self,
        ty: &Type<ErrorWitness>,
        subst: &Subst<ErrorWitness>,
        owned_contracts: Option<&[chelis_type_core::CollectionContractId]>,
        operation: &crate::resources::Operation<'_>,
    ) -> (
        Scheme<ErrorWitness>,
        Vec<chelis_type_core::CollectionContractId>,
    ) {
        let ty = subst.apply(ty);
        let env_tvars = self.free_tvars_in_operation(subst, operation);
        let env_dvars = self.free_dvars_in_operation(subst, operation);
        let env_rvars = self.free_rvars_in_operation(subst, operation);
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
                && !crate::infer::recursion::tvar_pinned(v, operation)
                && !pending_t.contains(&v)
                && !(active_tvars.contains(&v) && subst.level_of_tvar(v) <= current_level)
                && !group_level.is_some_and(|group| subst.level_of_tvar(v) <= group)
        };
        let ty_tvars = free_tvars(&ty);
        let tvars = ty_tvars
            .iter()
            .copied()
            .filter(|v| generalizable(*v))
            .collect::<Vec<_>>();
        let dvars = free_dvars(&ty)
            .into_iter()
            .filter(|v| {
                !env_dvars.contains(v)
                    && !pending_d.contains(v)
                    && !(active_dvars.contains(v) && subst.level_of_dvar(*v) <= current_level)
                    && matches!(subst.constraint_dim(&Dim::Var(*v)), Dim::Var(_))
            })
            .collect();
        let rvars = free_rvars(&ty)
            .into_iter()
            .filter(|v| {
                !env_rvars.contains(v)
                    && !pending_r.contains(v)
                    && !(active_rvars.contains(v) && subst.level_of_rvar(*v) <= current_level)
            })
            .collect();
        chelis_type_core::generalize_candidates(
            ty,
            subst,
            &ty_tvars,
            tvars,
            dvars,
            rvars,
            owned_contracts,
        )
    }
}

/// True when `ty` contains a tensor whose shape carries the named
/// dimension `name` (chelis#397/#469). Walks the same compound-type
/// structure as the free-var collectors above.
fn type_carries_dim_name(ty: &Type<ErrorWitness>, name: &str) -> bool {
    match ty {
        Type::<ErrorWitness>::Tensor(dims, _) => {
            dims.iter().any(|d| matches!(d, Dim::Name(n) if n == name))
        }
        Type::<ErrorWitness>::Fn(args, ret) => {
            args.iter().any(|a| type_carries_dim_name(a, name)) || type_carries_dim_name(ret, name)
        }
        Type::<ErrorWitness>::Ref(inner) => type_carries_dim_name(inner, name),
        Type::<ErrorWitness>::Adt(_, args) | Type::<ErrorWitness>::Tuple(args) => {
            args.iter().any(|a| type_carries_dim_name(a, name))
        }
        Type::<ErrorWitness>::KindedAdt(_, args) => args.iter().any(|argument| match argument {
            NominalArg::<ErrorWitness>::Type(ty) => type_carries_dim_name(ty, name),
            NominalArg::<ErrorWitness>::Dimension(Dim::Name(found)) => found == name,
            NominalArg::<ErrorWitness>::Dimension(_) => false,
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

#[cfg(test)]
mod module_scope_tests {
    use super::*;

    #[test]
    fn exact_lookup_does_not_promote_a_unique_terminal_match() {
        let mut env = Env::new();
        env.bind(
            "pkg__demo__Provider__borrowed".to_string(),
            Scheme::<ErrorWitness>::mono(Type::<ErrorWitness>::Prim(Prim::Int64)),
        );

        assert!(env.lookup("borrowed").is_none());
        assert!(env.lookup_terminal_unique("borrowed").is_some());
    }

    #[test]
    fn same_level_monomorphic_contracts_keep_distinct_declaration_owners() {
        crate::test_support::run(|operation| {
            let env = Env::new();
            let mut var_gen = VarGen::default();
            let mut subst = Subst::<ErrorWitness>::new();
            let list = Type::<ErrorWitness>::adt(
                "List".to_string(),
                vec![Type::<ErrorWitness>::Tensor(
                    vec![Dim::Lit(2), Dim::Lit(3)],
                    TensorPrec::Concrete(Prim::F32),
                )],
            );
            let result = Type::<ErrorWitness>::Tensor(
                vec![Dim::Wildcard, Dim::Wildcard],
                TensorPrec::Concrete(Prim::F32),
            );
            let constraint = CollectionConstraint::<ErrorWitness>::Concat {
                lhs: list.clone(),
                rhs: Type::<ErrorWitness>::Prim(Prim::Int32),
                result: result.clone(),
            };
            let checked = Scheme {
                result_origin: None,
                tvars: Vec::new(),
                tvar_restrictions: Vec::new(),
                dvars: Vec::new(),
                rvars: Vec::new(),
                constraints: vec![constraint.clone()],
                body: Type::<ErrorWitness>::function(
                    vec![list, Type::<ErrorWitness>::Prim(Prim::Int32)],
                    result,
                ),
            };

            let component = subst.enter_level(&var_gen);
            let first_mark = subst.collection_contract_mark();
            let first_ty = instantiate(&checked, &mut var_gen, &subst);
            let first_ids = subst.collection_contract_ids_since(first_mark);
            let second_mark = subst.collection_contract_mark();
            let second_ty = instantiate(&checked, &mut var_gen, &subst);
            let second_ids = subst.collection_contract_ids_since(second_mark);
            subst.leave_level(component, &var_gen);

            let first =
                env.generalize_with_collection_contracts(&first_ty, &subst, &first_ids, operation);
            assert_eq!(first.constraints, vec![constraint.clone()]);
            assert_eq!(
                subst.pending_collection_contracts().len(),
                1,
                "generalizing one recursive sibling must not absorb the other's contract"
            );

            let second = env.generalize_with_collection_contracts(
                &second_ty,
                &subst,
                &second_ids,
                operation,
            );
            assert_eq!(second.constraints, vec![constraint]);
            assert!(
                subst.pending_collection_contracts().is_empty(),
                "each sibling must consume exactly its own contract instance"
            );
        })
    }

    #[test]
    fn authored_binders_stay_monomorphic_inside_their_declaration_then_generalize_at_boundary() {
        crate::test_support::run(|operation| {
            let mut env = Env::new();
            let mut var_gen = VarGen::default();
            let mut subst = Subst::<ErrorWitness>::new();

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
            let ty = Type::<ErrorWitness>::tuple(vec![
                Type::<ErrorWitness>::Var(authored_type),
                Type::<ErrorWitness>::Tensor(
                    vec![Dim::Var(authored), Dim::Rank(authored_rank)],
                    TensorPrec::Concrete(Prim::F32),
                ),
            ]);

            let nested = subst.enter_level(&var_gen);
            let nested_type = var_gen.fresh_tvar();
            let nested_local = var_gen.fresh_dvar();
            let nested_rank = var_gen.fresh_rvar();
            subst.leave_level(nested, &var_gen);
            let nested_ty = Type::<ErrorWitness>::tuple(vec![
                ty.clone(),
                Type::<ErrorWitness>::Var(nested_type),
                Type::<ErrorWitness>::Tensor(
                    vec![Dim::Var(nested_local), Dim::Rank(nested_rank)],
                    TensorPrec::Concrete(Prim::F32),
                ),
            ]);
            let nested_scheme = env.generalize_in_operation(&nested_ty, &subst, operation);
            assert_eq!(nested_scheme.tvars, vec![nested_type]);
            assert_eq!(
                nested_scheme.dvars,
                vec![nested_local],
                "a nested let may generalize its own dimension but not its declaration's authored binder"
            );
            assert_eq!(nested_scheme.rvars, vec![nested_rank]);

            subst.leave_level(declaration, &var_gen);
            env.set_type_resolution_scope(None, None);
            let declaration_scheme = env.generalize_in_operation(&ty, &subst, operation);
            assert_eq!(declaration_scheme.tvars, vec![authored_type]);
            assert_eq!(
                declaration_scheme.dvars,
                vec![authored],
                "the authored dimension becomes a quantifier only at its declaration boundary"
            );
            assert_eq!(declaration_scheme.rvars, vec![authored_rank]);
        })
    }

    #[test]
    fn recursive_group_variable_is_monomorphic_until_group_close() {
        crate::test_support::run(|operation| {
            // [04-INF-5]: a provisional group hole is monomorphic inside its
            // group and can be quantified after closure if Γ does not own it.
            let mut var_gen = VarGen::default();
            let mut subst = Subst::<ErrorWitness>::new();
            let group = subst.enter_level(&var_gen);
            let body_var = var_gen.fresh_tvar();
            let body = Type::<ErrorWitness>::function(
                vec![Type::<ErrorWitness>::Var(body_var)],
                Type::<ErrorWitness>::Var(body_var),
            );
            let mut captured = Env::new();
            captured.begin_group_level(subst.current_level());
            let in_group = captured.generalize_in_operation(&body, &subst, operation);
            assert!(in_group.tvars.is_empty());
            captured.end_holed_group();
            subst.leave_level(group, &var_gen);

            let published = captured.generalize_in_operation(&body, &subst, operation);
            assert_eq!(published.tvars, vec![body_var]);
            // Installing a monomorphic outer binding also lowers its free
            // variable to that binding's level, as ordinary inference does.
            subst.lower_type_to_current(&body);
            captured.bind(
                "outer".to_string(),
                Scheme::<ErrorWitness>::mono(Type::<ErrorWitness>::Var(body_var)),
            );
            let captured_scheme = captured.generalize_in_operation(&body, &subst, operation);
            assert!(captured_scheme.tvars.is_empty());
        })
    }

    #[test]
    fn deferred_equalities_reaching_the_environment_stay_shared_at_a_let_boundary() {
        crate::test_support::run(|operation| {
            // `day = if c then Some(pair) else if d then Some(pair) else None`
            // followed by `match day with | Some(found) => { (a, b) = found ... }`:
            // instantiating `day` in the scrutinee mints join intermediates in the
            // declaration scope, beside the pattern variable `found`. The
            // destructure's `let` then carries those joins with its own body.
            let pair = || {
                Type::<ErrorWitness>::tuple(vec![
                    Type::<ErrorWitness>::Prim(Prim::Int64),
                    Type::<ErrorWitness>::Prim(Prim::Int64),
                ])
            };
            let option =
                |ty: Type<ErrorWitness>| Type::<ErrorWitness>::adt("Option".to_string(), vec![ty]);
            let carried = |body: Type<ErrorWitness>,
                           equations: &[ResultConstraint<ErrorWitness>]| {
                Type::<ErrorWitness>::Tuple(
                    core::iter::once(body)
                        .chain(
                            equations
                                .iter()
                                .flat_map(ResultConstraint::<ErrorWitness>::types)
                                .cloned(),
                        )
                        .collect(),
                )
            };
            let published =
                |env: &Env,
                 subst: &Subst<ErrorWitness>,
                 body: Type<ErrorWitness>,
                 equations: &[ResultConstraint<ErrorWitness>]| {
                    let mut tvars = env
                        .generalize_with_result_constraints(
                            &carried(body, equations),
                            subst,
                            None,
                            equations,
                            operation,
                        )
                        .tvars;
                    tvars.sort_unstable();
                    tvars
                };

            let mut var_gen = VarGen::default();
            let mut subst = Subst::<ErrorWitness>::new();
            let declaration = subst.enter_level(&var_gen);
            let inner_join = var_gen.fresh_tvar();
            let outer_join = var_gen.fresh_tvar();
            let found = var_gen.fresh_tvar();
            let scrutinee_joins = vec![
                ResultConstraint::<ErrorWitness>::Join {
                    inputs: vec![
                        option(pair()),
                        option(Type::<ErrorWitness>::Var(inner_join)),
                    ],
                    result: option(Type::<ErrorWitness>::Var(outer_join)),
                },
                ResultConstraint::<ErrorWitness>::Join {
                    inputs: vec![
                        option(pair()),
                        option(Type::<ErrorWitness>::Var(outer_join)),
                    ],
                    result: option(Type::<ErrorWitness>::Var(found)),
                },
            ];
            let mut env = Env::new();
            subst.lower_type_to_current(&Type::<ErrorWitness>::Var(found));
            env.bind(
                "found".to_string(),
                Scheme::<ErrorWitness>::mono(Type::<ErrorWitness>::Var(found)),
            );

            let binding = subst.enter_level(&var_gen);
            let shared_hole = var_gen.fresh_tvar();
            let shared_result = var_gen.fresh_tvar();
            let own_hole = var_gen.fresh_tvar();
            let own_result = var_gen.fresh_tvar();
            subst.leave_level(binding, &var_gen);

            // `tmp = found`: the scrutinee's joins reach `found`, which Γ owns, so
            // none of their intermediates may be freshened for this binding.
            assert!(
                published(
                    &env,
                    &subst,
                    Type::<ErrorWitness>::Var(found),
                    &scrutinee_joins
                )
                .is_empty()
            );

            // `again = if e then Some(found) else None`: variables minted for this
            // binding share one component with `found`, so they stay shared too.
            let mut through_found = scrutinee_joins.clone();
            through_found.push(ResultConstraint::<ErrorWitness>::Join {
                inputs: vec![
                    option(Type::<ErrorWitness>::Var(found)),
                    option(Type::<ErrorWitness>::Var(shared_hole)),
                ],
                result: option(Type::<ErrorWitness>::Var(shared_result)),
            });
            assert!(
                published(
                    &env,
                    &subst,
                    option(Type::<ErrorWitness>::Var(shared_result)),
                    &through_found
                )
                .is_empty()
            );

            // `fresh = if e then Some(pair) else None`: a component wholly owned by
            // the binding is quantified whole.
            let own = [ResultConstraint::<ErrorWitness>::Join {
                inputs: vec![option(pair()), option(Type::<ErrorWitness>::Var(own_hole))],
                result: option(Type::<ErrorWitness>::Var(own_result)),
            }];
            assert_eq!(
                published(
                    &env,
                    &subst,
                    option(Type::<ErrorWitness>::Var(own_result)),
                    &own
                ),
                vec![own_hole, own_result]
            );
            subst.leave_level(declaration, &var_gen);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "generalize-sweep-oracle")]
    #[test]
    fn production_measurement_scope_restores_nested_and_panicking_oracle_state() {
        crate::test_support::run(|operation| {
            assert!(operation.sweep_enabled.get());
            without_generalize_sweep_oracle(
                || {
                    assert!(!operation.sweep_enabled.get());
                    without_generalize_sweep_oracle(
                        || {
                            assert!(!operation.sweep_enabled.get());
                        },
                        operation,
                    );
                    assert!(!operation.sweep_enabled.get());
                    std::thread::spawn(|| {
                        crate::test_support::run(|operation| {
                            assert!(operation.sweep_enabled.get());
                        })
                    })
                    .join()
                    .unwrap();
                },
                operation,
            );
            assert!(operation.sweep_enabled.get());
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                without_generalize_sweep_oracle(|| panic!("measurement aborted"), operation);
            }));
            assert!(panicked.is_err());
            assert!(operation.sweep_enabled.get());
        })
    }

    #[test]
    fn closed_key_variables_require_the_exact_owned_callable() {
        let operand = Type::<ErrorWitness>::Var(TypeVar(1));
        let result = Type::<ErrorWitness>::Var(TypeVar(2));
        let relation = CollectionConstraint::<ErrorWitness>::KeyFromSeed {
            operand: operand.clone(),
            result: result.clone(),
        };
        let callable = chelis_type_core::collection_contract_callable_type(&relation);
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
        scheme.body = Type::<ErrorWitness>::tuple(vec![callable.clone(), result]);
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
        crate::test_support::run(|operation| {
            let env = Env::new();
            let mut var_gen = VarGen::default();
            let mut subst = Subst::<ErrorWitness>::new();
            let level = subst.enter_level(&var_gen);
            let operand = var_gen.fresh_type();
            let result = var_gen.fresh_type();
            let key_relation = CollectionConstraint::<ErrorWitness>::SplitKey {
                operand: operand.clone(),
                result: result.clone(),
            };
            let list = Type::<ErrorWitness>::adt("List".to_string(), vec![operand.clone()]);
            let append_relation = CollectionConstraint::<ErrorWitness>::Append {
                list: list.clone(),
                value: operand.clone(),
                result: list,
            };
            subst.record_collection_contract(key_relation.clone());
            subst.record_collection_contract(append_relation.clone());
            subst.leave_level(level, &var_gen);

            let scheme = env.generalize_in_operation(
                &chelis_type_core::collection_contract_callable_type(&key_relation),
                &subst,
                operation,
            );
            assert_eq!(scheme.constraints, vec![key_relation, append_relation]);
            let Type::<ErrorWitness>::Var(operand_var) = operand else {
                unreachable!()
            };
            let Type::<ErrorWitness>::Var(result_var) = result else {
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
        })
    }

    /// A scheme exercising every quantifier kind, with a restriction on one
    /// type variable and none on the other, so a route that drops
    /// restrictions and a route that installs them indiscriminately are both
    /// distinguishable from the correct one.
    fn restricted_scheme() -> Scheme<ErrorWitness> {
        Scheme {
            result_origin: None,
            constraints: vec![],
            tvars: vec![TypeVar(1), TypeVar(2)],
            tvar_restrictions: vec![(TypeVar(2), TypeVarRestriction::ActiveFloat)],
            dvars: vec![DimVar(3), DimVar(4)],
            rvars: vec![RankVar(5)],
            body: Type::<ErrorWitness>::tuple(vec![
                Type::<ErrorWitness>::Var(TypeVar(1)),
                Type::<ErrorWitness>::Tensor(
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

        let plain_subst = Subst::<ErrorWitness>::new();
        let mut plain_gen = VarGen::default();
        let plain = instantiate(&scheme, &mut plain_gen, &plain_subst);

        let mapped_subst = Subst::<ErrorWitness>::new();
        let mut mapped_gen = VarGen::default();
        let mapped = instantiate_scheme(&scheme, &mut mapped_gen, &mapped_subst);

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
            body: Type::<ErrorWitness>::Tensor(
                vec![Dim::Rank(quantified_rank), Dim::Var(quantified)],
                TensorPrec::Concrete(Prim::F32),
            ),
        };

        let inference_subst = Subst::<ErrorWitness>::new();
        inference_subst.mark_mapped_axis(quantified, 1);
        let mut var_gen = VarGen::default();
        let instantiated = instantiate_scheme(&scheme, &mut var_gen, &inference_subst);
        let fresh = instantiated.dvars[0].1;

        assert_eq!(inference_subst.mapped_axis(fresh), Some(1));
        let Type::<ErrorWitness>::Tensor(dims, _) = instantiated.ty else {
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

        // Every route instantiates the same scheme from the same starting
        // generator state, so all three mint the same fresh variables and the
        // observed restriction sets are directly comparable.
        let restrictions_after = |instantiate: &dyn Fn(&Subst<ErrorWitness>, &mut VarGen)| {
            let subst = Subst::<ErrorWitness>::new();
            let mut var_gen = VarGen::default();
            instantiate(&subst, &mut var_gen);
            (0..64)
                .map(TypeVar)
                .filter_map(|v| subst.tvar_restriction(v).map(|r| (v, r)))
                .collect::<Vec<_>>()
        };

        let via_instantiate = restrictions_after(&|subst, var_gen| {
            instantiate(&scheme, var_gen, subst);
        });
        let via_tvar_mapping = restrictions_after(&|subst, var_gen| {
            instantiate_with_tvar_mapping(&scheme, var_gen, subst);
        });
        let via_instantiate_scheme = restrictions_after(&|subst, var_gen| {
            instantiate_scheme(&scheme, var_gen, subst);
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
                non_dimension_binders: UnordSet::new(),
            },
        );
        let instantiation = InstantiatedScheme {
            ty: Type::<ErrorWitness>::Unit,
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
        let constructor = Scheme::<ErrorWitness>::mono(Type::<ErrorWitness>::function(
            vec![Type::<ErrorWitness>::Prim(Prim::F64)],
            Type::<ErrorWitness>::adt("Box".to_string(), Vec::new()),
        ));
        env.bind_constructor("N".to_string(), "Box".to_string(), constructor.clone());
        env.bind_lexical(
            "N".to_string(),
            Scheme::<ErrorWitness>::mono(Type::<ErrorWitness>::Prim(Prim::F64)),
        );

        assert_eq!(
            env.lookup("N").map(|scheme| &scheme.body),
            Some(&Type::<ErrorWitness>::Prim(Prim::F64))
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
            Scheme::<ErrorWitness>::mono(Type::<ErrorWitness>::adt(
                "Option".to_string(),
                Vec::new(),
            )),
        );
        let wrapper_scheme = Scheme::<ErrorWitness>::mono(Type::<ErrorWitness>::adt(
            "Wrapper".to_string(),
            Vec::new(),
        ));
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
        crate::test_support::run(|operation| {
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
                body: Type::<ErrorWitness>::tuple(vec![
                    Type::<ErrorWitness>::Var(quantified_type),
                    Type::<ErrorWitness>::Tensor(
                        vec![Dim::Var(quantified_dim), Dim::Rank(quantified_rank)],
                        TensorPrec::Var(quantified_type),
                    ),
                ]),
            };
            let mut env = Env::new();
            env.bind("generic".to_string(), scheme);
            let mut subst = Subst::<ErrorWitness>::new();
            subst
                .insert_type(quantified_type, Type::<ErrorWitness>::Prim(Prim::F32))
                .expect("unrestricted test substitution accepts f32");
            subst.insert_dim(quantified_dim, Dim::Lit(3));
            subst.insert_rank(quantified_rank, vec![Dim::Lit(4)]);

            assert!(env.free_tvars_in_operation(&subst, operation).is_empty());
            assert!(env.free_dvars_in_operation(&subst, operation).is_empty());
            assert!(env.free_rvars_in_operation(&subst, operation).is_empty());
        })
    }

    #[test]
    fn free_variables_stop_alias_chains_at_quantified_ids() {
        crate::test_support::run(|operation| {
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
                body: Type::<ErrorWitness>::tuple(vec![
                    Type::<ErrorWitness>::Var(outer_type),
                    Type::<ErrorWitness>::Tensor(
                        vec![Dim::Var(outer_dim), Dim::Rank(outer_rank)],
                        TensorPrec::Concrete(Prim::F32),
                    ),
                ]),
            };
            let mut env = Env::new();
            env.bind("generic".to_string(), scheme);
            let mut subst = Subst::<ErrorWitness>::new();
            subst
                .insert_type(outer_type, Type::<ErrorWitness>::Var(quantified_type))
                .expect("unrestricted test substitution accepts an alias");
            subst
                .insert_type(quantified_type, Type::<ErrorWitness>::Prim(Prim::F64))
                .expect("unrestricted test substitution accepts f64");
            subst.insert_dim(outer_dim, Dim::Var(quantified_dim));
            subst.insert_dim(quantified_dim, Dim::Lit(5));
            subst.insert_rank(outer_rank, vec![Dim::Rank(quantified_rank)]);
            subst.insert_rank(quantified_rank, vec![Dim::Lit(6)]);

            assert!(env.free_tvars_in_operation(&subst, operation).is_empty());
            assert!(env.free_dvars_in_operation(&subst, operation).is_empty());
            assert!(env.free_rvars_in_operation(&subst, operation).is_empty());
        })
    }

    #[test]
    fn free_variables_still_follow_unquantified_substitutions() {
        crate::test_support::run(|operation| {
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
                body: Type::<ErrorWitness>::tuple(vec![
                    Type::<ErrorWitness>::Var(source_type),
                    Type::<ErrorWitness>::Tensor(
                        vec![Dim::Var(source_dim), Dim::Rank(source_rank)],
                        TensorPrec::Concrete(Prim::F32),
                    ),
                ]),
            };
            let mut env = Env::new();
            env.bind("monomorphic".to_string(), scheme);
            let mut subst = Subst::<ErrorWitness>::new();
            subst
                .insert_type(source_type, Type::<ErrorWitness>::Var(target_type))
                .expect("unrestricted test substitution accepts an alias");
            subst.insert_dim(source_dim, Dim::Var(target_dim));
            subst.insert_rank(source_rank, vec![Dim::Rank(target_rank)]);

            assert_eq!(
                env.free_tvars_in_operation(&subst, operation),
                UnordSet::from([target_type])
            );
            assert_eq!(
                env.free_dvars_in_operation(&subst, operation),
                UnordSet::from([target_dim])
            );
            assert_eq!(
                env.free_rvars_in_operation(&subst, operation),
                UnordSet::from([target_rank])
            );
        })
    }
}

/// Report whether this build contains the reference generalization sweep.
pub const fn generalize_sweep_oracle_available() -> bool {
    cfg!(feature = "generalize-sweep-oracle")
}
