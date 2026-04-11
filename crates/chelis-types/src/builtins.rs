//! Built-in function signatures for RISC primitives and derived operations.
//!
//! Signature templates:
//! - tensor_binop: ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D,p]
//! - tensor_unop:  ∀D,p. tensor[D,p] → tensor[D,p]
//! - cmplt:        ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D, bool]
//! - logical_binop: ∀D. (tensor[D,bool], tensor[D,bool]) → tensor[D,bool]
//! - logical_unop:  ∀D. tensor[D,bool] → tensor[D,bool]

use crate::adt::{AdtDef, AdtRegistry, VariantInfo};
use crate::env::Env;
use crate::types::*;

pub const BUILTIN_NAMES: &[&str] = &[
    "add",
    "mul",
    "max_elem",
    "neg",
    "exp",
    "log",
    "sin",
    "sqrt",
    "cmplt",
    "sub",
    "div",
    "mod",
    "eq",
    "neq",
    "gt",
    "lte",
    "gte",
    "bitand",
    "bitor",
    "bitxor",
    "shl",
    "shr",
    "and",
    "or",
    "not",
    "relu",
    "sigmoid",
    "softmax",
    "normalize",
    "mean",
    "matmul",
    "min_elem",
    "layer_norm",
    "conv2d",
    "sum",
    "max_reduce",
    "reshape",
    "permute",
    "expand",
    "pad",
    "shrink",
    "stride",
    "print",
    "debug",
    "string_len",
    "string_concat",
    "string_slice",
    "string_contains",
    "string_starts_with",
    "string_ends_with",
    "string_trim",
    "to_string",
    "to_int",
    "to_float",
    "rank",
    "shape",
    "numel",
    "tensor_to_scalar",
    "scalar_to_tensor",
    "len",
    "index",
    "append",
    "concat",
    "take",
    "drop",
    "chunk",
    "range",
    "map",
    "filter",
    "fold",
    "scan",
    "partition",
    "flat_map",
    "flatten",
    "zip",
    "enumerate",
    "dict_of",
    "dict_get",
    "dict_contains",
    "dict_insert",
    "dict_merge",
    "dict_keys",
    "dict_values",
    "dict_entries",
    "to_tensor",
    "to_list",
    "pad_sequences",
];

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

    fn tensor_triop_return_first(name: &str, env: &mut Env, vg: &mut VarGen) {
        let t1 = vg.fresh_tvar();
        let t2 = vg.fresh_tvar();
        let t3 = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![t1, t2, t3],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Var(t1), Type::Var(t2), Type::Var(t3)],
                Box::new(Type::Var(t1)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_binop_to_out(name: &str, env: &mut Env, vg: &mut VarGen) {
        let t1 = vg.fresh_tvar();
        let t2 = vg.fresh_tvar();
        let out = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![t1, t2, out],
            dvars: vec![],
            body: Type::Fn(vec![Type::Var(t1), Type::Var(t2)], Box::new(Type::Var(out))),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_reduce_to_out(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let out = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![input, out],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Var(input), Type::Prim(Prim::Int32)],
                Box::new(Type::Var(out)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_expand_to_out(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let out = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![input, out],
            dvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Var(input),
                    Type::Prim(Prim::Int32),
                    Type::Prim(Prim::Int32),
                ],
                Box::new(Type::Var(out)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_with_rate(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![input],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Var(input), Type::Prim(Prim::F32)],
                Box::new(Type::Var(input)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_conv2d(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let kernel = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![input, kernel, output],
            dvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Var(input),
                    Type::Var(kernel),
                    Type::Prim(Prim::Int32),
                    Type::Prim(Prim::Int32),
                ],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_reduce(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![input],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Var(input), Type::Prim(Prim::Int32)],
                Box::new(Type::Var(input)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_unop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![input, output],
            dvars: vec![],
            body: Type::Fn(vec![Type::Var(input)], Box::new(Type::Var(output))),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_binop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let lhs = vg.fresh_tvar();
        let rhs = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![lhs, rhs, output],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Var(lhs), Type::Var(rhs)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_triop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![a, b, c, output],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Var(a), Type::Var(b), Type::Var(c)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_unop_same(name: &str, env: &mut Env, vg: &mut VarGen) {
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
    generic_binop("mod", &mut env, &mut vg);
    cmplt_sig("eq", &mut env, &mut vg);
    cmplt_sig("neq", &mut env, &mut vg);
    cmplt_sig("gt", &mut env, &mut vg);
    cmplt_sig("lte", &mut env, &mut vg);
    cmplt_sig("gte", &mut env, &mut vg);
    generic_binop("bitand", &mut env, &mut vg);
    generic_binop("bitor", &mut env, &mut vg);
    generic_binop("bitxor", &mut env, &mut vg);
    generic_binop("shl", &mut env, &mut vg);
    generic_binop("shr", &mut env, &mut vg);

    logical_binop("and", &mut env, &mut vg);
    logical_binop("or", &mut env, &mut vg);
    logical_unop("not", &mut env, &mut vg);

    tensor_unop("relu", &mut env, &mut vg);
    tensor_unop("sigmoid", &mut env, &mut vg);
    tensor_reduce("softmax", &mut env, &mut vg);
    tensor_unop("normalize", &mut env, &mut vg);
    tensor_reduce_to_out("mean", &mut env, &mut vg);

    tensor_binop_to_out("matmul", &mut env, &mut vg);
    tensor_binop("min_elem", &mut env, &mut vg);
    tensor_triop_return_first("layer_norm", &mut env, &mut vg);
    tensor_conv2d("conv2d", &mut env, &mut vg);
    tensor_reduce_to_out("sum", &mut env, &mut vg);
    tensor_reduce_to_out("max_reduce", &mut env, &mut vg);
    tensor_unop("reshape", &mut env, &mut vg);
    tensor_unop("permute", &mut env, &mut vg);
    tensor_expand_to_out("expand", &mut env, &mut vg);
    tensor_unop("pad", &mut env, &mut vg);
    tensor_unop("shrink", &mut env, &mut vg);
    tensor_unop("stride", &mut env, &mut vg);
    tensor_with_rate("dropout", &mut env, &mut vg);
    generic_unop("print", &mut env, &mut vg);
    generic_unop_same("debug", &mut env, &mut vg);
    generic_unop("string_len", &mut env, &mut vg);
    generic_binop("string_concat", &mut env, &mut vg);
    generic_unop("string_trim", &mut env, &mut vg);
    generic_triop("string_slice", &mut env, &mut vg);
    env.bind(
        "string_contains".to_string(),
        Scheme {
            tvars: vec![],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::String), Type::Prim(Prim::String)],
                Box::new(Type::Prim(Prim::Bool)),
            ),
        },
    );
    env.bind(
        "string_starts_with".to_string(),
        Scheme {
            tvars: vec![],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::String), Type::Prim(Prim::String)],
                Box::new(Type::Prim(Prim::Bool)),
            ),
        },
    );
    env.bind(
        "string_ends_with".to_string(),
        Scheme {
            tvars: vec![],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::String), Type::Prim(Prim::String)],
                Box::new(Type::Prim(Prim::Bool)),
            ),
        },
    );
    generic_unop("to_string", &mut env, &mut vg);
    generic_unop("to_int", &mut env, &mut vg);
    generic_unop("to_float", &mut env, &mut vg);
    generic_unop("rank", &mut env, &mut vg);
    generic_binop("shape", &mut env, &mut vg);
    generic_unop("numel", &mut env, &mut vg);
    generic_unop("tensor_to_scalar", &mut env, &mut vg);
    generic_unop("scalar_to_tensor", &mut env, &mut vg);
    generic_unop("len", &mut env, &mut vg);
    generic_binop("index", &mut env, &mut vg);
    generic_binop("append", &mut env, &mut vg);
    generic_binop("concat", &mut env, &mut vg);
    generic_binop("take", &mut env, &mut vg);
    generic_binop("drop", &mut env, &mut vg);
    generic_binop("chunk", &mut env, &mut vg);
    generic_binop("range", &mut env, &mut vg);
    generic_binop("map", &mut env, &mut vg);
    generic_binop("filter", &mut env, &mut vg);
    let fold_acc = vg.fresh_tvar();
    let fold_item = vg.fresh_tvar();
    let fold_ret = vg.fresh_tvar();
    env.bind(
        "fold".to_string(),
        Scheme {
            tvars: vec![fold_acc, fold_item, fold_ret],
            dvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Var(fold_acc),
                    Type::Var(fold_item),
                    Type::Var(fold_ret),
                ],
                Box::new(Type::Var(fold_ret)),
            ),
        },
    );
    let scan_acc = vg.fresh_tvar();
    let scan_item = vg.fresh_tvar();
    let scan_ret = vg.fresh_tvar();
    env.bind(
        "scan".to_string(),
        Scheme {
            tvars: vec![scan_acc, scan_item, scan_ret],
            dvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Var(scan_acc),
                    Type::Var(scan_item),
                    Type::Var(scan_ret),
                ],
                Box::new(Type::Var(scan_ret)),
            ),
        },
    );
    generic_binop("partition", &mut env, &mut vg);
    generic_binop("flat_map", &mut env, &mut vg);
    generic_unop("flatten", &mut env, &mut vg);
    generic_binop("zip", &mut env, &mut vg);
    generic_unop("enumerate", &mut env, &mut vg);
    generic_unop("dict_of", &mut env, &mut vg);
    generic_binop("dict_get", &mut env, &mut vg);
    generic_binop("dict_contains", &mut env, &mut vg);
    generic_binop("dict_remove", &mut env, &mut vg);
    generic_triop("dict_insert", &mut env, &mut vg);
    generic_binop("dict_merge", &mut env, &mut vg);
    generic_unop("dict_keys", &mut env, &mut vg);
    generic_unop("dict_values", &mut env, &mut vg);
    generic_unop("dict_entries", &mut env, &mut vg);
    generic_unop("to_tensor", &mut env, &mut vg);
    generic_unop("to_list", &mut env, &mut vg);
    generic_binop("pad_sequences", &mut env, &mut vg);

    (env, vg)
}

pub fn register_prelude_adts(env: &mut Env, vg: &mut VarGen, adt_reg: &mut AdtRegistry) {
    let option_tvar = vg.fresh_tvar();
    let option_type = Type::Adt("Option".to_string(), vec![Type::Var(option_tvar)]);

    env.bind(
        "Some".to_string(),
        Scheme {
            tvars: vec![option_tvar],
            dvars: vec![],
            body: Type::Fn(vec![Type::Var(option_tvar)], Box::new(option_type.clone())),
        },
    );
    env.bind(
        "None".to_string(),
        Scheme {
            tvars: vec![option_tvar],
            dvars: vec![],
            body: option_type.clone(),
        },
    );

    adt_reg
        .defs
        .entry("Option".to_string())
        .or_insert_with(|| AdtDef {
            name: "Option".to_string(),
            type_params: vec!["a".to_string()],
            variants: vec![
                VariantInfo {
                    name: "Some".to_string(),
                    fields: vec![(None, Type::Var(option_tvar))],
                },
                VariantInfo {
                    name: "None".to_string(),
                    fields: Vec::new(),
                },
            ],
        });

    let list_tvar = vg.fresh_tvar();
    let list_type = Type::Adt("List".to_string(), vec![Type::Var(list_tvar)]);

    env.bind(
        "Cons".to_string(),
        Scheme {
            tvars: vec![list_tvar],
            dvars: vec![],
            body: Type::Fn(
                vec![Type::Var(list_tvar), list_type.clone()],
                Box::new(list_type.clone()),
            ),
        },
    );
    env.bind(
        "Nil".to_string(),
        Scheme {
            tvars: vec![list_tvar],
            dvars: vec![],
            body: list_type.clone(),
        },
    );

    adt_reg
        .defs
        .entry("List".to_string())
        .or_insert_with(|| AdtDef {
            name: "List".to_string(),
            type_params: vec!["a".to_string()],
            variants: vec![
                VariantInfo {
                    name: "Cons".to_string(),
                    fields: vec![(None, Type::Var(list_tvar)), (None, list_type.clone())],
                },
                VariantInfo {
                    name: "Nil".to_string(),
                    fields: Vec::new(),
                },
            ],
        });
}

/// Names that the inference engine should special-case for return type.
/// cmplt, eq, neq, lte, gte return tensor[D, bool] instead of tensor[D, p].
#[allow(dead_code)] // Used by inference engine (Step 6)
pub const COMPARISON_OPS: &[&str] = &["cmplt", "eq", "neq", "gt", "lte", "gte"];

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
    fn builtin_env_has_layer_norm_and_conv2d() {
        let (env, _) = builtin_env();
        assert!(env.lookup("layer_norm").is_some());
        assert!(env.lookup("conv2d").is_some());
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
    fn builtin_env_has_phase3c_string_helpers() {
        let (env, _) = builtin_env();
        assert!(env.lookup("string_slice").is_some());
        assert!(env.lookup("string_contains").is_some());
        assert!(env.lookup("string_starts_with").is_some());
        assert!(env.lookup("string_ends_with").is_some());
        assert!(env.lookup("string_trim").is_some());
        assert!(env.lookup("to_int").is_some());
        assert!(env.lookup("to_float").is_some());
    }

    #[test]
    fn builtin_env_has_phase3c_integer_helpers() {
        let (env, _) = builtin_env();
        assert!(env.lookup("mod").is_some());
        assert!(env.lookup("bitand").is_some());
        assert!(env.lookup("bitor").is_some());
        assert!(env.lookup("bitxor").is_some());
        assert!(env.lookup("shl").is_some());
        assert!(env.lookup("shr").is_some());
    }

    #[test]
    fn register_prelude_adts_adds_option_constructors() {
        let (mut env, mut vg) = builtin_env();
        let mut adt_reg = AdtRegistry::new();
        register_prelude_adts(&mut env, &mut vg, &mut adt_reg);

        assert!(env.lookup("Some").is_some());
        assert!(env.lookup("None").is_some());
        assert!(adt_reg.lookup("Option").is_some());
        assert_eq!(
            adt_reg.variant_names("Option").expect("option variants"),
            vec!["Some".to_string(), "None".to_string()]
        );
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
