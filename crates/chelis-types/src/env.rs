//! Type environment: maps variable names to type schemes.

use std::collections::{HashMap, HashSet};

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
/// A *truly sourceless* runtime scalar (a bare `int32`/`int64` parameter)
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
    binders: Option<HashSet<String>>,
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

/// Type environment (Γ): maps names to polymorphic type schemes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Env {
    bindings: HashMap<String, Scheme>,
    /// Active constructor bindings, separate from ordinary value lookup.
    ///
    /// Declaration/import order chooses one active owner for an exact name,
    /// matching the value environment's established constructor binding. A
    /// later lexical parameter or block binding may replace `bindings[name]`
    /// for bare value position without changing constructor position.
    #[serde(default)]
    constructor_bindings: HashMap<String, ConstructorBinding>,
    /// Names introduced by the current lexical scope (function parameters,
    /// block bindings, and pattern bindings). Builtin-specific inference may
    /// only dispatch on a name from the closed builtin vocabulary when that
    /// spelling has not been replaced by one of these bindings. This is
    /// check-time provenance, not part of the reusable or serialized type
    /// environment.
    #[serde(skip)]
    lexical_bindings: HashSet<String>,
    /// Current declaration's type/dimension/rank binders. Installed only on
    /// the cloned environment used to infer that declaration, inherited by
    /// nested lexical clones, and omitted from cached checker state.
    #[serde(skip)]
    type_resolution_scope: TypeResolutionScope,
    /// Declared result trusted only while checking a linker-reserved
    /// [05-OP-35] stdlib wrapper whose runtime-axis shape proof is owned by
    /// #1298. Never serialized or exposed to entry source.
    #[serde(skip)]
    exact_stdlib_expected_result: Option<Type>,
    /// chelis#397/#469: provenance of `let`-bound `int`-valued names, so a
    /// runtime `expand` size built from a `let` binding can be checked for
    /// materializability. Cloned at every lexical scope boundary along with
    /// `bindings` (so it has correct lexical scoping for free) and dropped
    /// from serialization (it is a check-time-only analysis artifact).
    #[serde(skip)]
    size_provenance: HashMap<String, SizeProvenance>,
    /// chelis#631: literal element counts of `let`-bound list expressions,
    /// so `concat(rows, axis)` can size its concat axis through the
    /// binding (a list's length is not part of its type). Same
    /// lexical-scoping-by-`Clone` and add-symmetric mark/clear discipline
    /// as `size_provenance`; check-time-only, dropped from serialization.
    #[serde(skip)]
    list_literal_lens: HashMap<String, usize>,
}

impl Env {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a name. Returns None if unbound.
    pub fn lookup(&self, name: &str) -> Option<&Scheme> {
        self.bindings.get(name)
    }

    /// Look up the active constructor owner and scheme for an exact name.
    pub(crate) fn lookup_constructor(&self, name: &str) -> Option<(&str, &Scheme)> {
        self.constructor_bindings
            .get(name)
            .map(|binding| (binding.owner.as_str(), &binding.scheme))
    }

    /// Install the binder set owned by the declaration whose body is about to
    /// be inferred. Callers use a cloned `Env`, so this scope cannot leak to a
    /// sibling declaration or back into a reusable library snapshot.
    pub(crate) fn set_type_resolution_binders(&mut self, binders: Option<&HashSet<String>>) {
        self.type_resolution_scope.binders = binders.cloned();
    }

    /// Binder names visible to a nested source annotation in this lexical
    /// environment. Absence means closed input: named `t-var`/`d-var`/
    /// `d-rank` nodes do not allocate inference variables.
    pub(crate) fn type_resolution_binders(&self) -> Option<&HashSet<String>> {
        self.type_resolution_scope.binders.as_ref()
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
    }

    /// The recorded size provenance of a name, if any (chelis#397/#469).
    pub fn size_provenance(&self, name: &str) -> Option<SizeProvenance> {
        self.size_provenance.get(name).copied()
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
            .values()
            .any(|scheme| type_carries_dim_name(&scheme.body, name))
    }

    /// Look up an imported or qualified name by its unique terminal segment.
    pub fn lookup_terminal_unique(&self, name: &str) -> Option<&Scheme> {
        let mut matches = self
            .bindings
            .iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }

    /// Extend the environment with a new binding.
    pub fn bind(&mut self, name: String, scheme: Scheme) {
        self.bindings.insert(name, scheme);
    }

    /// Bind a constructor in both structural constructor position and the
    /// ordinary value environment used by bare/nullary references.
    pub(crate) fn bind_constructor(&mut self, name: String, owner: String, scheme: Scheme) {
        self.bindings.insert(name.clone(), scheme.clone());
        self.constructor_bindings
            .insert(name, ConstructorBinding { owner, scheme });
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

    /// Remove a temporary inference binding before generalizing an SCC.
    /// Recursive function components are prebound monomorphically while
    /// their bodies are inferred, then all provisional members are removed
    /// together so their resolved types can be generalized as one unit.
    pub(crate) fn remove_binding(&mut self, name: &str) {
        self.bindings.remove(name);
    }

    /// Instantiate a scheme into the caller's inference substitution so
    /// quantified semantic restrictions follow the fresh variables.
    pub fn instantiate(
        &self,
        scheme: &Scheme,
        var_gen: &mut VarGen,
        inference_subst: &Subst,
    ) -> Type {
        self.instantiate_with_tvar_mapping(scheme, var_gen, inference_subst)
            .0
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
        let mut subst = Subst::new();
        let mut mapping = Vec::with_capacity(scheme.tvars.len());
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
                inference_subst.install_tvar_restriction(fresh_var, *restriction);
            }
            mapping.push((tv, fresh));
        }
        for &dv in &scheme.dvars {
            subst.insert_dim(dv, var_gen.fresh_dim());
        }
        for &rv in &scheme.rvars {
            // Each rank var instantiates to a fresh sole-`Rank` shape so every
            // call site gets its own rank (Tier-2 rank polymorphism).
            subst.insert_rank(rv, vec![Dim::Rank(var_gen.fresh_rvar())]);
        }
        (subst.apply(&scheme.body), mapping)
    }

    /// Collect all free type variables across all bindings in the environment.
    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub fn free_tvars(&self, subst: &Subst) -> HashSet<TypeVar> {
        let mut result = HashSet::new();
        for scheme in self.bindings.values() {
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
    pub fn free_dvars(&self, subst: &Subst) -> HashSet<DimVar> {
        let mut result = HashSet::new();
        for scheme in self.bindings.values() {
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
    pub fn free_rvars(&self, subst: &Subst) -> HashSet<RankVar> {
        let mut result = HashSet::new();
        for scheme in self.bindings.values() {
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
        let level_scheme = self.generalize_by_levels(ty, subst);
        #[cfg(feature = "generalize-sweep-oracle")]
        GENERALIZE_SWEEP_ORACLE_ENABLED.with(|enabled| {
            if enabled.get() {
                let sweep_scheme = self.generalize_by_sweep(ty, subst);
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
        level_scheme
    }

    fn generalize_by_levels(&self, ty: &Type, subst: &Subst) -> Scheme {
        let ty = subst.apply(ty);
        let ty_tvars = free_tvars(&ty);
        let ty_dvars = free_dvars(&ty);
        let ty_rvars = free_rvars(&ty);
        let level = subst.current_level();
        let tvars = ty_tvars
            .into_iter()
            .filter(|v| {
                subst.level_of_tvar(*v) > level
                        && !subst.has_deferred_shape_constraint(*v)
                        // spec/04 §3.1.1: a variable minted for an in-group
                        // recursive instantiation stays monomorphic while its
                        // group is inferred, so a let-bound alias of a group
                        // member cannot smuggle in polymorphic recursion.
                        && !crate::infer::recursion::tvar_pinned(*v)
            })
            .collect::<Vec<_>>();
        let tvar_restrictions = tvars
            .iter()
            .filter_map(|v| {
                subst
                    .tvar_restriction(*v)
                    .map(|restriction| (*v, restriction))
            })
            .collect();
        Scheme {
            tvars,
            tvar_restrictions,
            dvars: ty_dvars
                .into_iter()
                .filter(|v| subst.level_of_dvar(*v) > level)
                .collect(),
            rvars: ty_rvars
                .into_iter()
                .filter(|v| subst.level_of_rvar(*v) > level)
                .collect(),
            body: ty,
        }
    }

    /// Exact pre-#1207 environment-sweep implementation. It is compiled only
    /// into tests and the temporary parity-oracle feature.
    #[cfg(feature = "generalize-sweep-oracle")]
    fn generalize_by_sweep(&self, ty: &Type, subst: &Subst) -> Scheme {
        let ty = subst.apply(ty);
        let env_tvars = self.free_tvars(subst);
        let env_dvars = self.free_dvars(subst);
        let env_rvars = self.free_rvars(subst);
        let tvars = free_tvars(&ty)
            .into_iter()
            .filter(|v| {
                !env_tvars.contains(v)
                    && !subst.has_deferred_shape_constraint(*v)
                    && !crate::infer::recursion::tvar_pinned(*v)
            })
            .collect::<Vec<_>>();
        let tvar_restrictions = tvars
            .iter()
            .filter_map(|v| {
                subst
                    .tvar_restriction(*v)
                    .map(|restriction| (*v, restriction))
            })
            .collect();
        Scheme {
            tvars,
            tvar_restrictions,
            dvars: free_dvars(&ty)
                .into_iter()
                .filter(|v| !env_dvars.contains(v))
                .collect(),
            rvars: free_rvars(&ty)
                .into_iter()
                .filter(|v| !env_rvars.contains(v))
                .collect(),
            body: ty,
        }
    }
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
    }

    #[test]
    fn free_variables_protect_quantified_ids_from_global_substitutions() {
        let quantified_type = TypeVar(10);
        let quantified_dim = DimVar(20);
        let quantified_rank = RankVar(30);
        let scheme = Scheme {
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

        assert_eq!(env.free_tvars(&subst), HashSet::from([target_type]));
        assert_eq!(env.free_dvars(&subst), HashSet::from([target_dim]));
        assert_eq!(env.free_rvars(&subst), HashSet::from([target_rank]));
    }
}
