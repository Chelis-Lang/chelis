//! Type environment: maps variable names to type schemes.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::unify::Subst;

/// Type environment (Γ): maps names to polymorphic type schemes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

    /// Generalize a type over variables not free in the environment.
    pub fn generalize(&self, ty: &Type, subst: &Subst) -> Scheme {
        let ty = subst.apply(ty);
        let env_tvars = self.free_tvars(subst);
        let env_dvars = self.free_dvars(subst);
        let ty_tvars = free_tvars(&ty);
        let ty_dvars = free_dvars(&ty);
        Scheme {
            tvars: ty_tvars
                .into_iter()
                .filter(|v| !env_tvars.contains(v))
                .collect(),
            dvars: ty_dvars
                .into_iter()
                .filter(|v| !env_dvars.contains(v))
                .collect(),
            body: ty,
        }
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
        Type::Tensor(_, _) | Type::Prim(_) | Type::Unit | Type::Error => {}
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
