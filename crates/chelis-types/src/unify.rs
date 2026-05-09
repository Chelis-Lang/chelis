//! Unification algorithm for the Chelis type checker.
//!
//! Solves type equations by binding type variables and dimension variables.
//!
//! ## Path compression
//!
//! `Subst` stores type / dim variable bindings as union-find chains. When a
//! caller resolves a variable via [`Subst::apply`] or [`Subst::apply_dim`],
//! the resolver iteratively follows the chain to its terminal value AND
//! writes the resolved value back into the substitution so future lookups
//! land in O(1). The HashMaps live behind a `RefCell` so this in-place
//! compression remains available to callers that hold an immutable
//! borrow of `Subst` (annotation passes, `Env::generalize`, occurs checks).
//!
//! Without path compression, deeply-nested expressions like Cons chains
//! over wildcard tensor literals (e.g.,
//! `pad_sequences_to([[10×4 floats]], ...)`) accumulate dim-variable
//! chains 1000+ links long; resolving every dim through a 1000-link
//! chain on every type walk drives ~2 billion `apply_dim` calls and
//! pushes phase-0e annotation from ~2s into 10+ minutes for chelis-std's
//! `nn/embedding.ch`. See the gdb backtrace recorded in this commit's
//! body for the canonical reproducer.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::types::*;

/// Type error produced during unification.
#[derive(Debug, Clone)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum TypeErrorKind {
    TypeMismatch,
    PrecisionMismatch,
    DimensionMismatch,
    ArityMismatch,
    OccursCheck,
    NotAFunction,
}

/// Substitution: maps type variables to types and dim variables to dims.
///
/// The HashMaps are wrapped in `Mutex` so [`Subst::apply`] and
/// [`Subst::apply_dim`] can perform in-place path compression while
/// keeping the public method receiver `&self`. The Mutex is uncontended
/// in normal use (each `compile_new_source_in_context` call clones the
/// substitution into its own thread-local state), so the locking cost
/// is one atomic compare-exchange per call. Use [`Subst::insert_type`] /
/// [`Subst::insert_dim`] to record new bindings during unification.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Subst {
    types: Mutex<HashMap<TypeVar, Type>>,
    dims: Mutex<HashMap<DimVar, Dim>>,
}

impl Clone for Subst {
    fn clone(&self) -> Self {
        Subst {
            types: Mutex::new(self.types.lock().expect("subst.types poisoned").clone()),
            dims: Mutex::new(self.dims.lock().expect("subst.dims poisoned").clone()),
        }
    }
}

impl Subst {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of the type-variable bindings (cloned out of the lock).
    /// Useful for serialization, tests, and read-only inspection.
    pub fn types_snapshot(&self) -> HashMap<TypeVar, Type> {
        self.types.lock().expect("subst.types poisoned").clone()
    }

    /// Snapshot of the dim-variable bindings.
    pub fn dims_snapshot(&self) -> HashMap<DimVar, Dim> {
        self.dims.lock().expect("subst.dims poisoned").clone()
    }

    /// Number of type-variable bindings currently in the substitution.
    pub fn types_len(&self) -> usize {
        self.types.lock().expect("subst.types poisoned").len()
    }

    /// Number of dim-variable bindings currently in the substitution.
    pub fn dims_len(&self) -> usize {
        self.dims.lock().expect("subst.dims poisoned").len()
    }

    /// Record a new type-variable binding.
    pub fn insert_type(&mut self, v: TypeVar, ty: Type) {
        self.types
            .lock()
            .expect("subst.types poisoned")
            .insert(v, ty);
    }

    /// Record a new dim-variable binding.
    pub fn insert_dim(&mut self, v: DimVar, dim: Dim) {
        self.dims
            .lock()
            .expect("subst.dims poisoned")
            .insert(v, dim);
    }

    /// Resolve a type variable to its terminal binding (a non-Var, or
    /// an unbound Var). Iteratively walks the chain and rewrites every
    /// link directly to the terminal so future lookups are O(1).
    fn resolve_tvar(&self, start: TypeVar) -> Type {
        let mut map = self.types.lock().expect("subst.types poisoned");
        // Phase 1: walk to the terminal, recording the chain.
        let mut chain: Vec<TypeVar> = Vec::new();
        let mut current = start;
        let terminal = loop {
            match map.get(&current) {
                None => break Type::Var(current),
                Some(Type::Var(v)) if *v == current => break Type::Var(current),
                Some(Type::Var(v)) => {
                    chain.push(current);
                    current = *v;
                }
                Some(other) => {
                    chain.push(current);
                    break other.clone();
                }
            }
        };
        // Phase 2: compress — point every link in the chain directly at
        // the terminal value.
        if chain.len() > 1 {
            for v in chain {
                map.insert(v, terminal.clone());
            }
        }
        terminal
    }

    /// Resolve a dim variable to its terminal binding. Same iterative-
    /// walk-then-compress shape as [`Subst::resolve_tvar`].
    fn resolve_dvar(&self, start: DimVar) -> Dim {
        let mut map = self.dims.lock().expect("subst.dims poisoned");
        let mut chain: Vec<DimVar> = Vec::new();
        let mut current = start;
        let terminal = loop {
            match map.get(&current) {
                None => break Dim::Var(current),
                Some(Dim::Var(v)) if *v == current => break Dim::Var(current),
                Some(Dim::Var(v)) => {
                    chain.push(current);
                    current = *v;
                }
                Some(other) => {
                    chain.push(current);
                    break other.clone();
                }
            }
        };
        if chain.len() > 1 {
            for v in chain {
                map.insert(v, terminal.clone());
            }
        }
        terminal
    }

    /// Apply this substitution to a type, resolving all bound variables.
    /// Path-compresses any chains of length ≥ 2 it encounters so future
    /// lookups land in O(1).
    pub fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(v) => self.resolve_tvar(*v),
            Type::Fn(args, ret) => {
                let args = args.iter().map(|a| self.apply(a)).collect();
                let ret = Box::new(self.apply(ret));
                Type::Fn(args, ret)
            }
            Type::Ref(inner) => Type::Ref(Box::new(self.apply(inner))),
            Type::Tensor(dims, prec) => {
                let dims = dims.iter().map(|d| self.apply_dim(d)).collect();
                Type::Tensor(dims, *prec)
            }
            Type::Adt(name, args) => {
                let args = args.iter().map(|a| self.apply(a)).collect();
                Type::Adt(name.clone(), args)
            }
            Type::Tuple(ts) => {
                let ts = ts.iter().map(|t| self.apply(t)).collect();
                Type::Tuple(ts)
            }
            Type::Prim(_) | Type::Unit | Type::Error => ty.clone(),
        }
    }

    /// Apply this substitution to a dimension. Path-compresses chains.
    pub fn apply_dim(&self, dim: &Dim) -> Dim {
        match dim {
            Dim::Var(v) => self.resolve_dvar(*v),
            _ => dim.clone(),
        }
    }

    /// Compose: apply `other` to all bindings in self, then merge.
    pub fn compose(&mut self, other: &Subst) {
        // Note: `other.apply` / `other.apply_dim` lock `other`'s maps;
        // we must not be holding a lock on `other` simultaneously
        // (which we never do — `self` and `other` are distinct).
        {
            let mut self_types = self.types.lock().expect("subst.types poisoned");
            for val in self_types.values_mut() {
                *val = other.apply(val);
            }
        }
        {
            let mut self_dims = self.dims.lock().expect("subst.dims poisoned");
            for val in self_dims.values_mut() {
                *val = other.apply_dim(val);
            }
        }
        let other_types = other.types_snapshot();
        let other_dims = other.dims_snapshot();
        {
            let mut self_types = self.types.lock().expect("subst.types poisoned");
            for (k, v) in other_types {
                self_types.entry(k).or_insert(v);
            }
        }
        {
            let mut self_dims = self.dims.lock().expect("subst.dims poisoned");
            for (k, v) in other_dims {
                self_dims.entry(k).or_insert(v);
            }
        }
    }
}

/// Unify two types, producing a substitution or a type error.
pub fn unify(t1: &Type, t2: &Type, subst: &mut Subst) -> Result<(), TypeError> {
    let t1 = subst.apply(t1);
    let t2 = subst.apply(t2);

    match (&t1, &t2) {
        // Same type — trivially unified
        (Type::Prim(p1), Type::Prim(p2)) if p1 == p2 => Ok(()),
        (Type::Prim(p1), Type::Prim(p2)) => Err(TypeError {
            kind: TypeErrorKind::PrecisionMismatch,
            message: format!(
                "precision mismatch: expected {}, got {}",
                p1.name(),
                p2.name()
            ),
        }),

        (Type::Unit, Type::Unit) => Ok(()),

        (Type::Ref(inner1), Type::Ref(inner2)) => unify(inner1, inner2, subst),

        // Type variable binding
        (Type::Var(v), _) => bind_tvar(*v, &t2, subst),
        (_, Type::Var(v)) => bind_tvar(*v, &t1, subst),

        // Function types
        (Type::Fn(args1, ret1), Type::Fn(args2, ret2)) => {
            if args1.len() != args2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!(
                        "function arity mismatch: expected {} args, got {}",
                        args1.len(),
                        args2.len()
                    ),
                });
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                unify(a1, a2, subst)?;
            }
            unify(ret1, ret2, subst)
        }

        // Tensor types
        (Type::Tensor(dims1, p1), Type::Tensor(dims2, p2)) => {
            if p1 != p2 {
                return Err(TypeError {
                    kind: TypeErrorKind::PrecisionMismatch,
                    message: format!("tensor precision mismatch: {} vs {}", p1.name(), p2.name()),
                });
            }
            if dims1.len() != dims2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::DimensionMismatch,
                    message: format!(
                        "tensor rank mismatch: {} dims vs {} dims",
                        dims1.len(),
                        dims2.len()
                    ),
                });
            }
            for (d1, d2) in dims1.iter().zip(dims2.iter()) {
                unify_dim(d1, d2, subst)?;
            }
            Ok(())
        }

        // ADT types
        (Type::Adt(n1, args1), Type::Adt(n2, args2)) => {
            if n1 != n2 {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch,
                    message: format!("type mismatch: {n1} vs {n2}"),
                });
            }
            if args1.len() != args2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!(
                        "ADT type argument count mismatch for {n1}: {} vs {}",
                        args1.len(),
                        args2.len()
                    ),
                });
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                unify(a1, a2, subst)?;
            }
            Ok(())
        }

        // Tuple types
        (Type::Tuple(ts1), Type::Tuple(ts2)) => {
            if ts1.len() != ts2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!("tuple length mismatch: {} vs {}", ts1.len(), ts2.len()),
                });
            }
            for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                unify(t1, t2, subst)?;
            }
            Ok(())
        }

        // Error propagation — unifying with Error always succeeds (partial inference)
        (Type::Error, _) | (_, Type::Error) => Ok(()),

        // Everything else is a mismatch
        _ => Err(TypeError {
            kind: TypeErrorKind::TypeMismatch,
            message: format!("type mismatch: {t1} vs {t2}"),
        }),
    }
}

/// Unify two dimensions.
pub fn unify_dim(d1: &Dim, d2: &Dim, subst: &mut Subst) -> Result<(), TypeError> {
    let d1 = subst.apply_dim(d1);
    let d2 = subst.apply_dim(d2);

    match (&d1, &d2) {
        (Dim::Name(n1), Dim::Name(n2)) if n1 == n2 => Ok(()),
        (Dim::Lit(l1), Dim::Lit(l2)) if l1 == l2 => Ok(()),
        (Dim::Wildcard, _) | (_, Dim::Wildcard) => Ok(()),
        (Dim::Var(v), _) => bind_dvar(*v, &d2, subst),
        (_, Dim::Var(v)) => bind_dvar(*v, &d1, subst),
        _ => Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: format!("dimension mismatch: {d1:?} vs {d2:?}"),
        }),
    }
}

fn bind_tvar(v: TypeVar, ty: &Type, subst: &mut Subst) -> Result<(), TypeError> {
    if let Type::Var(v2) = ty
        && *v2 == v
    {
        return Ok(()); // same variable
    }
    if occurs_in(v, ty, subst) {
        return Err(TypeError {
            kind: TypeErrorKind::OccursCheck,
            message: format!("infinite type: ?{} occurs in {ty}", v.0),
        });
    }
    subst.insert_type(v, ty.clone());
    Ok(())
}

fn bind_dvar(v: DimVar, dim: &Dim, subst: &mut Subst) -> Result<(), TypeError> {
    if let Dim::Var(v2) = dim
        && *v2 == v
    {
        return Ok(());
    }
    if occurs_in_dim(v, dim, subst) {
        return Err(TypeError {
            kind: TypeErrorKind::OccursCheck,
            message: format!("infinite dimension: d{} occurs in {dim:?}", v.0),
        });
    }
    subst.insert_dim(v, dim.clone());
    Ok(())
}

fn occurs_in(v: TypeVar, ty: &Type, subst: &Subst) -> bool {
    let ty = subst.apply(ty);
    match &ty {
        Type::Var(v2) => *v2 == v,
        Type::Fn(args, ret) => {
            args.iter().any(|a| occurs_in(v, a, subst)) || occurs_in(v, ret, subst)
        }
        Type::Ref(inner) => occurs_in(v, inner, subst),
        Type::Tensor(_, _) => false, // tensors don't contain type vars in dims
        Type::Adt(_, args) => args.iter().any(|a| occurs_in(v, a, subst)),
        Type::Tuple(ts) => ts.iter().any(|t| occurs_in(v, t, subst)),
        Type::Prim(_) | Type::Unit | Type::Error => false,
    }
}

fn occurs_in_dim(v: DimVar, dim: &Dim, subst: &Subst) -> bool {
    let dim = subst.apply_dim(dim);
    matches!(dim, Dim::Var(v2) if v2 == v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var_gen() -> VarGen {
        VarGen::default()
    }

    #[test]
    fn unify_same_prim() {
        let mut s = Subst::new();
        assert!(unify(&Type::Prim(Prim::F32), &Type::Prim(Prim::F32), &mut s).is_ok());
    }

    #[test]
    fn unify_different_prim_fails() {
        let mut s = Subst::new();
        let err = unify(&Type::Prim(Prim::F32), &Type::Prim(Prim::Bf16), &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
    }

    #[test]
    fn unify_tvar_binds() {
        let mut g = var_gen();
        let v = g.fresh_type();
        let mut s = Subst::new();
        assert!(unify(&v, &Type::Prim(Prim::F32), &mut s).is_ok());
        assert_eq!(s.apply(&v), Type::Prim(Prim::F32));
    }

    #[test]
    fn unify_fn_types() {
        let mut s = Subst::new();
        let f1 = Type::Fn(vec![Type::Prim(Prim::F32)], Box::new(Type::Prim(Prim::F32)));
        let f2 = Type::Fn(vec![Type::Prim(Prim::F32)], Box::new(Type::Prim(Prim::F32)));
        assert!(unify(&f1, &f2, &mut s).is_ok());
    }

    #[test]
    fn unify_fn_arity_mismatch() {
        let mut s = Subst::new();
        let f1 = Type::Fn(vec![Type::Prim(Prim::F32)], Box::new(Type::Prim(Prim::F32)));
        let f2 = Type::Fn(
            vec![Type::Prim(Prim::F32), Type::Prim(Prim::F32)],
            Box::new(Type::Prim(Prim::F32)),
        );
        let err = unify(&f1, &f2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::ArityMismatch));
    }

    #[test]
    fn unify_tensor_same_dims() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(
            vec![Dim::Name("batch".into()), Dim::Name("hidden".into())],
            Prim::F32,
        );
        let t2 = Type::Tensor(
            vec![Dim::Name("batch".into()), Dim::Name("hidden".into())],
            Prim::F32,
        );
        assert!(unify(&t1, &t2, &mut s).is_ok());
    }

    #[test]
    fn unify_tensor_dim_mismatch() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], Prim::F32);
        let t2 = Type::Tensor(vec![Dim::Name("seq".into())], Prim::F32);
        let err = unify(&t1, &t2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_tensor_precision_mismatch() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], Prim::F32);
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], Prim::Bf16);
        let err = unify(&t1, &t2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
    }

    #[test]
    fn unify_dim_var_binds() {
        let mut g = var_gen();
        let dv = g.fresh_dvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Var(dv)], Prim::F32);
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], Prim::F32);
        assert!(unify(&t1, &t2, &mut s).is_ok());
        assert_eq!(s.apply_dim(&Dim::Var(dv)), Dim::Name("batch".into()));
    }

    #[test]
    fn unify_dim_transitive_binding_resolves() {
        let mut g = var_gen();
        let d0 = g.fresh_dvar();
        let d1 = g.fresh_dvar();
        let mut s = Subst::new();
        assert!(unify_dim(&Dim::Var(d0), &Dim::Var(d1), &mut s).is_ok());
        assert!(unify_dim(&Dim::Var(d1), &Dim::Name("batch".into()), &mut s).is_ok());
        assert_eq!(s.apply_dim(&Dim::Var(d0)), Dim::Name("batch".into()));
    }

    #[test]
    fn unify_wildcard_matches_anything() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Wildcard], Prim::F32);
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], Prim::F32);
        assert!(unify(&t1, &t2, &mut s).is_ok());
    }

    #[test]
    fn unify_tensor_rank_mismatch() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("a".into())], Prim::F32);
        let t2 = Type::Tensor(
            vec![Dim::Name("a".into()), Dim::Name("b".into())],
            Prim::F32,
        );
        let err = unify(&t1, &t2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_adt_same() {
        let mut s = Subst::new();
        let a1 = Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]);
        let a2 = Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]);
        assert!(unify(&a1, &a2, &mut s).is_ok());
    }

    #[test]
    fn unify_adt_name_mismatch() {
        let mut s = Subst::new();
        let a1 = Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]);
        let a2 = Type::Adt("Result".into(), vec![Type::Prim(Prim::F32)]);
        assert!(unify(&a1, &a2, &mut s).is_err());
    }

    #[test]
    fn occurs_check_prevents_infinite_type() {
        let mut g = var_gen();
        let v = g.fresh_tvar();
        let mut s = Subst::new();
        // Try to unify ?0 with List(?0) — should fail with occurs check
        let list_v = Type::Adt("List".into(), vec![Type::Var(v)]);
        let err = unify(&Type::Var(v), &list_v, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::OccursCheck));
    }

    #[test]
    fn error_type_unifies_with_anything() {
        let mut s = Subst::new();
        assert!(unify(&Type::Error, &Type::Prim(Prim::F32), &mut s).is_ok());
        assert!(unify(&Type::Prim(Prim::F32), &Type::Error, &mut s).is_ok());
    }

    #[test]
    fn unify_tuple_types() {
        let mut s = Subst::new();
        let t1 = Type::Tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::Int32)]);
        let t2 = Type::Tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::Int32)]);
        assert!(unify(&t1, &t2, &mut s).is_ok());
    }

    #[test]
    fn transitive_binding() {
        let mut g = var_gen();
        let a = g.fresh_tvar();
        let b = g.fresh_tvar();
        let mut s = Subst::new();
        // a = b, b = f32 → a = f32
        unify(&Type::Var(a), &Type::Var(b), &mut s).unwrap();
        unify(&Type::Var(b), &Type::Prim(Prim::F32), &mut s).unwrap();
        assert_eq!(s.apply(&Type::Var(a)), Type::Prim(Prim::F32));
    }
}
