//! Built-in function signatures for RISC primitives and derived operations.
//!
//! Signature templates:
//! - tensor_binop: ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D,p]
//! - tensor_unop:  ∀D,p. tensor[D,p] → tensor[D,p]
//! - cmplt:        ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D, bool]
//! - logical_binop: ∀D. (tensor[D,bool], tensor[D,bool]) → tensor[D,bool]
//! - logical_unop:  ∀D. tensor[D,bool] → tensor[D,bool]

use crate::env::Env;
use crate::types::*;

/// Create the built-in type environment with all RISC Tier 1 + Tier 2 signatures.
pub fn builtin_env() -> (Env, VarGen) {
    let mut env = Env::new();
    let mut vg = VarGen::default();

    // --- Signature builders ---

    // ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D,p]
    // D is represented as a single DimVar. At unification time, tensor dims are
    // matched positionally (equal rank, element-wise dim unification).
    // The DimVar here acts as a placeholder — when the scheme is instantiated,
    // a fresh DimVar is created, and unification with a concrete tensor will
    // bind dims element-wise.
    fn tensor_binop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let dv = vg.fresh_dvar();
        let pv = vg.fresh_tvar(); // precision as type var (will unify to Prim)
        // We can't put a TypeVar inside Tensor's Prim slot directly.
        // Instead, use a full TypeVar for the whole tensor type.
        // The constraint is: both args and return are the SAME tensor type.
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![tv],
            dvars: vec![dv],
            body: Type::Fn(vec![Type::Var(tv), Type::Var(tv)], Box::new(Type::Var(tv))),
        };
        let _ = pv; // precision enforcement happens during unification
        env.bind(name.to_string(), scheme);
    }

    // ∀D,p. tensor[D,p] → tensor[D,p]
    fn tensor_unop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let _dv = vg.fresh_dvar();
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![tv],
            dvars: vec![],
            body: Type::Fn(vec![Type::Var(tv)], Box::new(Type::Var(tv))),
        };
        env.bind(name.to_string(), scheme);
    }

    // ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D, bool]
    // The key difference: return precision is always Bool.
    fn cmplt_sig(name: &str, env: &mut Env, vg: &mut VarGen) {
        // Input: two tensors with same dims and same precision
        // Output: tensor with same dims but Bool precision
        // We represent this by using a DimVar for dims and constraining
        // the return type explicitly.
        let dv = vg.fresh_dvar();
        let input_tv = vg.fresh_tvar(); // will unify to tensor[D, p]
        // Can't easily express "same dims, different precision" with just TypeVars.
        // For Phase 0: use (T, T) → T but the inference engine will special-case
        // cmplt to swap the return precision to Bool after unification.
        //
        // Better approach: just use TypeVar polymorphism. The args must be the same
        // type (tensor[D,p]), and the return is also the same type. This is wrong
        // for cmplt (should return bool), but fixing it properly requires either:
        // 1. A special case in the inference engine for cmplt
        // 2. A richer signature language that can express precision transformation
        //
        // For now: register cmplt as a SPECIAL FORM that the inference engine handles.
        // The builtin entry just marks it as a 2-arg function.
        let _ = (dv, input_tv);
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![tv],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Var(tv), Type::Var(tv)],
                Box::new(Type::Var(tv)), // inference engine overrides for cmplt
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    // ∀D. (tensor[D,bool], tensor[D,bool]) → tensor[D,bool]
    fn logical_binop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let _dv = vg.fresh_dvar();
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![tv],
            dvars: vec![],
            body: Type::Fn(vec![Type::Var(tv), Type::Var(tv)], Box::new(Type::Var(tv))),
        };
        env.bind(name.to_string(), scheme);
    }

    fn logical_unop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let _dv = vg.fresh_dvar();
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![tv],
            dvars: vec![],
            body: Type::Fn(vec![Type::Var(tv)], Box::new(Type::Var(tv))),
        };
        env.bind(name.to_string(), scheme);
    }

    // --- Register all built-ins ---

    // Tier 1: RISC Primitives
    tensor_binop("add", &mut env, &mut vg);
    tensor_binop("mul", &mut env, &mut vg);
    tensor_binop("max_elem", &mut env, &mut vg);

    tensor_unop("neg", &mut env, &mut vg);
    tensor_unop("exp", &mut env, &mut vg);
    tensor_unop("log", &mut env, &mut vg);
    tensor_unop("sin", &mut env, &mut vg);
    tensor_unop("sqrt", &mut env, &mut vg);

    cmplt_sig("cmplt", &mut env, &mut vg);

    // Tier 2: Derived built-ins
    tensor_binop("sub", &mut env, &mut vg);
    tensor_binop("div", &mut env, &mut vg);
    cmplt_sig("eq", &mut env, &mut vg);
    cmplt_sig("neq", &mut env, &mut vg);
    cmplt_sig("lte", &mut env, &mut vg);
    cmplt_sig("gte", &mut env, &mut vg);

    logical_binop("and", &mut env, &mut vg);
    logical_binop("or", &mut env, &mut vg);
    logical_unop("not", &mut env, &mut vg);

    tensor_unop("relu", &mut env, &mut vg);
    tensor_unop("sigmoid", &mut env, &mut vg);
    tensor_unop("softmax", &mut env, &mut vg);
    tensor_unop("normalize", &mut env, &mut vg);

    tensor_binop("matmul", &mut env, &mut vg);

    (env, vg)
}

/// Names that the inference engine should special-case for return type.
/// cmplt, eq, neq, lte, gte return tensor[D, bool] instead of tensor[D, p].
#[allow(dead_code)] // Used by inference engine (Step 6)
pub const COMPARISON_OPS: &[&str] = &["cmplt", "eq", "neq", "lte", "gte"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_env_has_add() {
        let (env, _) = builtin_env();
        assert!(env.lookup("add").is_some());
    }

    #[test]
    fn builtin_env_has_matmul() {
        let (env, _) = builtin_env();
        assert!(env.lookup("matmul").is_some());
    }

    #[test]
    fn builtin_env_has_cmplt() {
        let (env, _) = builtin_env();
        assert!(env.lookup("cmplt").is_some());
    }

    #[test]
    fn builtin_env_has_logical_ops() {
        let (env, _) = builtin_env();
        assert!(env.lookup("and").is_some());
        assert!(env.lookup("or").is_some());
        assert!(env.lookup("not").is_some());
    }

    #[test]
    fn builtin_env_missing_name_returns_none() {
        let (env, _) = builtin_env();
        assert!(env.lookup("nonexistent").is_none());
    }

    #[test]
    fn instantiate_add_produces_fn_type() {
        let (env, mut vg) = builtin_env();
        let scheme = env.lookup("add").unwrap();
        let ty = env.instantiate(scheme, &mut vg);
        match ty {
            Type::Fn(args, _ret) => assert_eq!(args.len(), 2),
            other => panic!("expected Fn, got {other:?}"),
        }
    }

    #[test]
    fn add_args_unify_with_same_tensor() {
        use crate::unify::{Subst, unify};

        let (env, mut vg) = builtin_env();
        let add_scheme = env.lookup("add").unwrap();
        let add_ty = env.instantiate(add_scheme, &mut vg);

        // add should accept two tensors of the same type
        let tensor_f32 = Type::Tensor(vec![Dim::Name("batch".into())], Prim::F32);
        let expected_fn = Type::Fn(
            vec![tensor_f32.clone(), tensor_f32.clone()],
            Box::new(tensor_f32),
        );

        let mut subst = Subst::new();
        assert!(unify(&add_ty, &expected_fn, &mut subst).is_ok());
    }

    #[test]
    fn add_rejects_mixed_precision() {
        use crate::unify::{Subst, unify};

        let (env, mut vg) = builtin_env();
        let add_scheme = env.lookup("add").unwrap();
        let add_ty = env.instantiate(add_scheme, &mut vg);

        // add(tensor[batch,f32], tensor[batch,bf16]) should fail
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], Prim::F32);
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], Prim::Bf16);
        let bad_fn = Type::Fn(vec![t1, t2], Box::new(Type::Var(vg.fresh_tvar())));

        let mut subst = Subst::new();
        assert!(unify(&add_ty, &bad_fn, &mut subst).is_err());
    }

    #[test]
    fn add_rejects_tensor_plus_int() {
        use crate::unify::{Subst, unify};

        let (env, mut vg) = builtin_env();
        let add_scheme = env.lookup("add").unwrap();
        let add_ty = env.instantiate(add_scheme, &mut vg);

        // add(tensor[batch,f32], int32) should fail
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], Prim::F32);
        let t2 = Type::Prim(Prim::Int32);
        let bad_fn = Type::Fn(vec![t1, t2], Box::new(Type::Var(vg.fresh_tvar())));

        let mut subst = Subst::new();
        assert!(unify(&add_ty, &bad_fn, &mut subst).is_err());
    }
}
