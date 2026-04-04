//! Unification algorithm for the Chelis type checker.
//!
//! Solves type equations by binding type variables and dimension variables.

use std::collections::HashMap;

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
#[derive(Debug, Clone, Default)]
pub struct Subst {
    pub types: HashMap<TypeVar, Type>,
    pub dims: HashMap<DimVar, Dim>,
}

impl Subst {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply this substitution to a type, resolving all bound variables.
    pub fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(v) => match self.types.get(v) {
                Some(bound) => self.apply(bound),
                None => ty.clone(),
            },
            Type::Fn(args, ret) => {
                let args = args.iter().map(|a| self.apply(a)).collect();
                let ret = Box::new(self.apply(ret));
                Type::Fn(args, ret)
            }
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

    /// Apply this substitution to a dimension.
    pub fn apply_dim(&self, dim: &Dim) -> Dim {
        match dim {
            Dim::Var(v) => match self.dims.get(v) {
                Some(bound) => self.apply_dim(bound),
                None => dim.clone(),
            },
            _ => dim.clone(),
        }
    }

    /// Compose: apply `other` to all bindings in self, then merge.
    pub fn compose(&mut self, other: &Subst) {
        for val in self.types.values_mut() {
            *val = other.apply(val);
        }
        for val in self.dims.values_mut() {
            *val = other.apply_dim(val);
        }
        for (k, v) in &other.types {
            self.types.entry(*k).or_insert_with(|| v.clone());
        }
        for (k, v) in &other.dims {
            self.dims.entry(*k).or_insert_with(|| v.clone());
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
    subst.types.insert(v, ty.clone());
    Ok(())
}

fn bind_dvar(v: DimVar, dim: &Dim, subst: &mut Subst) -> Result<(), TypeError> {
    if let Dim::Var(v2) = dim
        && *v2 == v
    {
        return Ok(());
    }
    subst.dims.insert(v, dim.clone());
    Ok(())
}

fn occurs_in(v: TypeVar, ty: &Type, subst: &Subst) -> bool {
    let ty = subst.apply(ty);
    match &ty {
        Type::Var(v2) => *v2 == v,
        Type::Fn(args, ret) => {
            args.iter().any(|a| occurs_in(v, a, subst)) || occurs_in(v, ret, subst)
        }
        Type::Tensor(_, _) => false, // tensors don't contain type vars in dims
        Type::Adt(_, args) => args.iter().any(|a| occurs_in(v, a, subst)),
        Type::Tuple(ts) => ts.iter().any(|t| occurs_in(v, t, subst)),
        Type::Prim(_) | Type::Unit | Type::Error => false,
    }
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
