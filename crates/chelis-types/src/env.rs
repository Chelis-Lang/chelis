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

/// Type environment (Γ): maps names to polymorphic type schemes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Env {
    bindings: HashMap<String, Scheme>,
    /// chelis#397/#469: provenance of `let`-bound `int`-valued names, so a
    /// runtime `expand` size built from a `let` binding can be checked for
    /// materializability. Cloned at every lexical scope boundary along with
    /// `bindings` (so it has correct lexical scoping for free) and dropped
    /// from serialization (it is a check-time-only analysis artifact).
    #[serde(skip)]
    size_provenance: HashMap<String, SizeProvenance>,
}

impl Env {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a name. Returns None if unbound.
    pub fn lookup(&self, name: &str) -> Option<&Scheme> {
        self.bindings.get(name)
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
            let ty = subst.apply(&scheme.body);
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
            let ty = subst.apply(&scheme.body);
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
            let ty = subst.apply(&scheme.body);
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
                .filter(|v| !env_tvars.contains(v))
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
        Type::Prim(_) | Type::Unit | Type::Error => {}
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
