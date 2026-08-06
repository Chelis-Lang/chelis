//! Type environment: maps variable names to type schemes.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::unify::Subst;

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

/// Type environment (Γ): maps names to polymorphic type schemes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Env {
    bindings: HashMap<String, Scheme>,
    /// Current declaration's type/dimension/rank binders. Installed only on
    /// the cloned environment used to infer that declaration, inherited by
    /// nested lexical clones, and omitted from cached checker state.
    #[serde(skip)]
    type_resolution_scope: TypeResolutionScope,
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

    /// Remove a temporary inference binding before generalizing an SCC.
    /// Recursive function components are prebound monomorphically while
    /// their bodies are inferred, then all provisional members are removed
    /// together so their resolved types can be generalized as one unit.
    pub(crate) fn remove_binding(&mut self, name: &str) {
        self.bindings.remove(name);
    }

    /// Instantiate a polymorphic scheme with fresh variables.
    pub fn instantiate(&self, scheme: &Scheme, var_gen: &mut VarGen) -> Type {
        let mut subst = Subst::new();
        for &tv in &scheme.tvars {
            subst.insert_type(tv, var_gen.fresh_type());
        }
        for &dv in &scheme.dvars {
            subst.insert_dim(dv, var_gen.fresh_dim());
        }
        for &rv in &scheme.rvars {
            // Each rank var instantiates to a fresh sole-`Rank` shape so every
            // call site gets its own rank (Tier-2 rank polymorphism).
            subst.insert_rank(rv, vec![Dim::Rank(var_gen.fresh_rvar())]);
        }
        subst.apply(&scheme.body)
    }

    /// Collect all free type variables across all bindings in the environment.
    pub fn free_tvars(&self, subst: &Subst) -> HashSet<TypeVar> {
        let mut result = HashSet::new();
        for scheme in self.bindings.values() {
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
    pub fn free_dvars(&self, subst: &Subst) -> HashSet<DimVar> {
        let mut result = HashSet::new();
        for scheme in self.bindings.values() {
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
    pub fn free_rvars(&self, subst: &Subst) -> HashSet<RankVar> {
        let mut result = HashSet::new();
        for scheme in self.bindings.values() {
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
        let ty = subst.apply(ty);
        let env_tvars = self.free_tvars(subst);
        let env_dvars = self.free_dvars(subst);
        let env_rvars = self.free_rvars(subst);
        let ty_tvars = free_tvars(&ty);
        let ty_dvars = free_dvars(&ty);
        let ty_rvars = free_rvars(&ty);
        Scheme {
            tvars: ty_tvars
                .into_iter()
                .filter(|v| !env_tvars.contains(v) && !subst.has_deferred_shape_constraint(*v))
                .collect(),
            dvars: ty_dvars
                .into_iter()
                .filter(|v| !env_dvars.contains(v))
                .collect(),
            rvars: ty_rvars
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
    fn free_variables_protect_quantified_ids_from_global_substitutions() {
        let quantified_type = TypeVar(10);
        let quantified_dim = DimVar(20);
        let quantified_rank = RankVar(30);
        let scheme = Scheme {
            tvars: vec![quantified_type],
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
        subst.insert_type(quantified_type, Type::Prim(Prim::F32));
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
        subst.insert_type(outer_type, Type::Var(quantified_type));
        subst.insert_type(quantified_type, Type::Prim(Prim::F64));
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
        subst.insert_type(source_type, Type::Var(target_type));
        subst.insert_dim(source_dim, Dim::Var(target_dim));
        subst.insert_rank(source_rank, vec![Dim::Rank(target_rank)]);

        assert_eq!(env.free_tvars(&subst), HashSet::from([target_type]));
        assert_eq!(env.free_dvars(&subst), HashSet::from([target_dim]));
        assert_eq!(env.free_rvars(&subst), HashSet::from([target_rank]));
    }
}
