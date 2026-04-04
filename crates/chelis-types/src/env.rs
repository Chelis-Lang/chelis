//! Type environment: maps variable names to type schemes.

use std::collections::HashMap;

use crate::types::*;
use crate::unify::Subst;

/// Type environment (Γ): maps names to polymorphic type schemes.
#[derive(Debug, Clone, Default)]
pub struct Env {
    bindings: HashMap<String, Scheme>,
}

impl Env {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a name. Returns None if unbound.
    pub fn lookup(&self, name: &str) -> Option<&Scheme> {
        self.bindings.get(name)
    }

    /// Extend the environment with a new binding.
    pub fn bind(&mut self, name: String, scheme: Scheme) {
        self.bindings.insert(name, scheme);
    }

    /// Instantiate a polymorphic scheme with fresh variables.
    pub fn instantiate(&self, scheme: &Scheme, var_gen: &mut VarGen) -> Type {
        let mut subst = Subst::new();
        for &tv in &scheme.tvars {
            subst.types.insert(tv, var_gen.fresh_type());
        }
        for &dv in &scheme.dvars {
            subst.dims.insert(dv, var_gen.fresh_dim());
        }
        subst.apply(&scheme.body)
    }

    /// Generalize a type over variables not free in the environment.
    /// For now, a simplified version that generalizes ALL type/dim vars.
    pub fn generalize(&self, ty: &Type, subst: &Subst) -> Scheme {
        let ty = subst.apply(ty);
        let tvars = free_tvars(&ty);
        let dvars = free_dvars(&ty);
        Scheme {
            tvars,
            dvars,
            body: ty,
        }
    }
}

/// Collect all free type variables in a type.
fn free_tvars(ty: &Type) -> Vec<TypeVar> {
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
        Type::Tensor(_, _) | Type::Prim(_) | Type::Unit | Type::Error => {}
    }
}

/// Collect all free dimension variables in a type.
fn free_dvars(ty: &Type) -> Vec<DimVar> {
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
