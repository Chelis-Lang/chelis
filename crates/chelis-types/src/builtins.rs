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
    "recip",
    "exp",
    "log",
    "sin",
    "sqrt",
    "cos",
    "tan",
    "atan",
    "abs",
    "floor",
    "ceil",
    "uniform_like",
    "cmplt",
    "sub",
    "div",
    "mod",
    "eq",
    "neq",
    "lt",
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
    "tanh",
    "silu",
    "gelu",
    "softmax",
    "normalize",
    "mean",
    "matmul",
    "min_elem",
    "layer_norm",
    "conv2d",
    "sum",
    "max_reduce",
    "min_reduce",
    "prod_reduce",
    "argmax_reduce",
    "argmin_reduce",
    // §2.3.1 strided windowed reduction (Valid padding). One Surf
    // builtin per reducer; the IR collapses them to a single
    // `RiscOp::ReduceWindow` with a `ReduceWindowKind` discriminator.
    "reduce_window_max",
    "reduce_window_min",
    "reduce_window_sum",
    "reduce_window_mean",
    "reshape",
    "permute",
    "expand",
    "pad",
    "shrink",
    "stride",
    "print",
    "fail",
    "debug",
    "test_assert",
    "test_assert_eq_f32",
    "test_assert_eq_int",
    "test_assert_eq_bool",
    "test_assert_eq_string",
    "test_assert_close_tensor",
    "test_assert_eq_tensor_int64",
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
    "tensor_scan",
    "partition",
    "flat_map",
    "flatten",
    "zip",
    "enumerate",
    "dict_of",
    "dict_get",
    "dict_contains",
    "dict_remove",
    "dict_insert",
    "dict_merge",
    "dict_keys",
    "dict_values",
    "dict_entries",
    "to_tensor",
    "to_list",
    "pad_sequences",
    "pad_sequences_to",
    "read_file",
    "write_file",
    "read_lines",
    "read_bytes",
    "file_exists",
    "list_dir",
    "mmap_file",
    "mmap_read",
    "mmap_len",
    "process_run",
    "einsum",
    "split",
    "gather",
    "scatter",
    "scatter_replace",
    "where",
    "cumsum",
    "sort",
    "diagonal",
    "trace",
    "clamp",
];

/// Shape semantics of a builtin for the Tier-2 rank-polymorphism
/// Body-Discipline check (`spec/design/rank_polymorphism.md` §Soundness
/// Boundary). Keyed on SHAPE SEMANTICS, **not** the HM scheme: `relu`,
/// `reshape`, and `permute` all share `&tv -> tv`, but only `relu` is
/// shape-identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeClass {
    /// Output shape provably equals an input shape with no axis reordering —
    /// pure elementwise ops (the precision may change, e.g. comparisons).
    /// Always admitted inside a rank-polymorphic (`..r`) body.
    Identity,
    /// Shape-changing but *name-tracked*: the op addresses axes by name and the
    /// procedural inference arm computes a symbolic output that carries the
    /// surviving named axes through (named-axis reductions, Tier-3 §4.5.3).
    /// Admitted inside a rank-poly body — the procedural arm is the real gate:
    /// it rejects a non-existent/ambiguous axis or a positional index at
    /// symbolic rank, so no transposition can slip past.
    NameTracked,
    /// Rewrites/reorders the shape positionally, is shape-parameterized, or is a
    /// non-tensor/host op whose output shape is *not* name-trackable at symbolic
    /// rank. Forbidden inside a rank-poly body: against an opaque spread there
    /// are no named axes left to catch a transposition/reshape (§4.2).
    Rewriting,
}

/// Classify a builtin's shape semantics for the Body-Discipline check.
///
/// `Identity` and `NameTracked` are explicit allowlists; everything else falls
/// through to `Rewriting`. That default is the safe direction — a builtin that
/// is not *provably* shape-identity or name-tracked is rejected inside a
/// rank-poly body, so a missed classification can only over-reject, never open
/// a §4.2 hole. The `shape_class_identity_set_is_pinned` test pins the sets so
/// any change is deliberate.
pub fn shape_class(name: &str) -> ShapeClass {
    match name {
        // Pure elementwise — output shape == input shape (precision may change
        // for comparisons/logical). No axis argument, no reordering.
        "add" | "mul" | "sub" | "div" | "mod" | "max_elem" | "min_elem" | "neg" | "recip"
        | "exp" | "log" | "sin" | "sqrt" | "cos" | "tan" | "atan" | "abs" | "floor" | "ceil"
        | "relu" | "sigmoid" | "tanh" | "silu" | "gelu" | "not" | "clamp" | "uniform_like"
        | "where" | "eq" | "neq" | "lt" | "gt" | "lte" | "gte" | "cmplt" | "bitand" | "bitor"
        | "bitxor" | "shl" | "shr" | "and" | "or" => ShapeClass::Identity,
        // Named-axis reductions: address the reduced axis by name and drop
        // exactly it, carrying the surviving named axes through (Tier-3 §4.5.3).
        // Restricted to `sum`/`mean`: these lower through the tensor-DAG backend
        // and build+run end-to-end. `max_reduce`/`min_reduce`/`prod_reduce`/
        // `argmax_reduce`/`argmin_reduce` route through the host lane in a
        // rank-poly inline and don't yet compile (chelis#340), so they stay
        // Rewriting — rejected in a `..r` body — to keep check↔backend in sync
        // (a check-clean program must build). They remain usable at concrete rank.
        //
        // Named-axis expand (chelis#339, the R+1 inverse): `expand` addresses
        // its insertion point by name (trailing end, or before a named anchor)
        // and the procedural arm (`check_expand_signature`) computes the
        // symbolic output row, rejecting positional axes at symbolic rank —
        // the same gate structure as the reductions.
        "sum" | "mean" | "expand" => ShapeClass::NameTracked,
        // Positional reshapes/permutes, matmul/conv, axis-indexed ops,
        // gather/scatter, and every non-tensor/host builtin.
        _ => ShapeClass::Rewriting,
    }
}

/// Create the built-in type environment with all RISC Tier 1 + Tier 2 signatures.
pub fn builtin_env() -> (Env, VarGen) {
    let mut env = Env::new();
    let mut vg = VarGen::default();

    // --- Signature builders ---
    fn borrowed(ty: Type) -> Type {
        Type::Ref(Box::new(ty))
    }

    // ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D,p]
    // D is represented as a single DimVar. At unification time, tensor dims are
    // matched positionally (equal rank, element-wise dim unification).
    // The DimVar here acts as a placeholder — when the scheme is instantiated,
    // a fresh DimVar is created, and unification with a concrete tensor will
    // bind dims element-wise.
    fn tensor_binop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let dv = vg.fresh_dvar();
        // Whole-tensor type variable: both args and return are the
        // SAME tensor type. This continues to work post-WS-A5 because
        // unifying two tensor types unifies both dim lists and the
        // precision slot. Precision-slot polymorphism (TensorPrec::Var)
        // is reserved for sig-quantified type variables in user-written
        // Surf signatures — the builtin scheme keeps the simpler
        // "whole tensor as one type variable" shape.
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![tv],
            dvars: vec![dv],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(tv)), borrowed(Type::Var(tv))],
                Box::new(Type::Var(tv)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    // ∀D,p. tensor[D,p] → tensor[D,p]
    fn tensor_unop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let _dv = vg.fresh_dvar();
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![tv],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(vec![borrowed(Type::Var(tv))], Box::new(Type::Var(tv))),
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
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(tv)), borrowed(Type::Var(tv))],
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
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(tv)), borrowed(Type::Var(tv))],
                Box::new(Type::Var(tv)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn logical_unop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let _dv = vg.fresh_dvar();
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![tv],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(vec![borrowed(Type::Var(tv))], Box::new(Type::Var(tv))),
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
            rvars: vec![],
            body: Type::Fn(
                vec![
                    borrowed(Type::Var(t1)),
                    borrowed(Type::Var(t2)),
                    borrowed(Type::Var(t3)),
                ],
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
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(t1)), borrowed(Type::Var(t2))],
                Box::new(Type::Var(out)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_reduce_to_out(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let out = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![input, out],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(input)), Type::Prim(Prim::Int32)],
                Box::new(Type::Var(out)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    /// Fallback HM scheme for the `reduce_window_*` family. The dedicated
    /// `infer_reduce_window_app` arm in `infer.rs` overrides the result
    /// type with the spec §2.3.1 shape contract; this scheme exists so
    /// the function name is in scope at lookup time and the canonical
    /// arg-arity / list-of-int32 constraints are visible during unification.
    fn tensor_reduce_window(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let out = vg.fresh_tvar();
        let int_list = Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int32)]);
        let scheme = Scheme {
            tvars: vec![input, out],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(input)), int_list.clone(), int_list],
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
            rvars: vec![],
            body: Type::Fn(
                vec![
                    borrowed(Type::Var(input)),
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
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(input)), Type::Prim(Prim::F32)],
                Box::new(Type::Var(input)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_with_bounds(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![input],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    borrowed(Type::Var(input)),
                    Type::Prim(Prim::F32),
                    Type::Prim(Prim::F32),
                ],
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
            rvars: vec![],
            body: Type::Fn(
                vec![
                    borrowed(Type::Var(input)),
                    borrowed(Type::Var(kernel)),
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
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(input)), Type::Prim(Prim::Int32)],
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
            rvars: vec![],
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
            rvars: vec![],
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
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Var(a), Type::Var(b), Type::Var(c)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_triop_first_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![a, b, c, output],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(a)), Type::Var(b), Type::Var(c)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_triop_first_two_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![a, b, c, output],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(a)), borrowed(Type::Var(b)), Type::Var(c)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_triop_all_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![a, b, c, output],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    borrowed(Type::Var(a)),
                    borrowed(Type::Var(b)),
                    borrowed(Type::Var(c)),
                ],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_triop_second_third_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![a, b, c, output],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Var(a), borrowed(Type::Var(b)), borrowed(Type::Var(c))],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_pentaop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let d = vg.fresh_tvar();
        let e = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![a, b, c, d, e, output],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Var(a),
                    Type::Var(b),
                    Type::Var(c),
                    Type::Var(d),
                    Type::Var(e),
                ],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_quadop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let d = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![a, b, c, d, output],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Var(a), Type::Var(b), Type::Var(c), Type::Var(d)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_unop_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![input, output],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(input))],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_unop_borrow_same(name: &str, env: &mut Env, vg: &mut VarGen) {
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![tv],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(vec![borrowed(Type::Var(tv))], Box::new(Type::Var(tv))),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_binop_first_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let lhs = vg.fresh_tvar();
        let rhs = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            tvars: vec![lhs, rhs, output],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(lhs)), Type::Var(rhs)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    // --- Register all built-ins ---

    // Tier 1: RISC Primitives
    tensor_binop("add", &mut env, &mut vg);
    tensor_binop("mul", &mut env, &mut vg);
    // `div` and `recip` were promoted from a Tier 2
    // `exp(neg(log(_)))` decomposition to native Tier 1 primitives
    // (IEEE-754 semantics, correct on the full real line).
    tensor_binop("div", &mut env, &mut vg);
    tensor_binop("max_elem", &mut env, &mut vg);

    tensor_unop("neg", &mut env, &mut vg);
    tensor_unop("recip", &mut env, &mut vg);
    tensor_unop("exp", &mut env, &mut vg);
    tensor_unop("log", &mut env, &mut vg);
    tensor_unop("sin", &mut env, &mut vg);
    tensor_unop("sqrt", &mut env, &mut vg);
    tensor_unop("cos", &mut env, &mut vg);
    tensor_unop("tan", &mut env, &mut vg);
    tensor_unop("atan", &mut env, &mut vg);
    tensor_unop("abs", &mut env, &mut vg);
    tensor_unop("floor", &mut env, &mut vg);
    tensor_unop("ceil", &mut env, &mut vg);
    tensor_with_bounds("uniform_like", &mut env, &mut vg);

    cmplt_sig("cmplt", &mut env, &mut vg);

    // Tier 2: Derived built-ins
    tensor_binop("sub", &mut env, &mut vg);
    generic_binop("mod", &mut env, &mut vg);
    cmplt_sig("eq", &mut env, &mut vg);
    cmplt_sig("neq", &mut env, &mut vg);
    cmplt_sig("lt", &mut env, &mut vg);
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
    // Bucket 3 activation parity: `tanh`, `silu`, `gelu` are pointwise
    // tensor unops with the same `∀D,p. tensor[D,p] → tensor[D,p]`
    // signature shape as `relu`/`sigmoid`. Their host-runtime and
    // C-backend lowerings live in `chelis-compiler-api/src/runtime/host_ops.rs`
    // and `chelis-backend-c/src/host_emit.rs` respectively.
    tensor_unop("tanh", &mut env, &mut vg);
    tensor_unop("silu", &mut env, &mut vg);
    tensor_unop("gelu", &mut env, &mut vg);
    tensor_reduce("softmax", &mut env, &mut vg);
    tensor_unop("normalize", &mut env, &mut vg);
    tensor_reduce_to_out("mean", &mut env, &mut vg);

    tensor_binop_to_out("matmul", &mut env, &mut vg);
    tensor_binop("min_elem", &mut env, &mut vg);
    tensor_triop_return_first("layer_norm", &mut env, &mut vg);
    tensor_conv2d("conv2d", &mut env, &mut vg);
    tensor_reduce_to_out("sum", &mut env, &mut vg);
    tensor_reduce_to_out("max_reduce", &mut env, &mut vg);
    tensor_reduce_to_out("min_reduce", &mut env, &mut vg);
    tensor_reduce_to_out("prod_reduce", &mut env, &mut vg);
    tensor_reduce_to_out("argmax_reduce", &mut env, &mut vg);
    tensor_reduce_to_out("argmin_reduce", &mut env, &mut vg);
    // §2.3.1 reduce_window family. Fallback HM scheme is
    // `&tensor[D, p] -> List[int32] -> List[int32] -> tensor[D', p]`;
    // the actual shape contract (output rank = input rank, trailing
    // axis extents derived from the window/stride formula) is enforced
    // by the dedicated `infer_reduce_window_app` arm in `infer.rs`,
    // which also rejects non-positive window/stride literals.
    tensor_reduce_window("reduce_window_max", &mut env, &mut vg);
    tensor_reduce_window("reduce_window_min", &mut env, &mut vg);
    tensor_reduce_window("reduce_window_sum", &mut env, &mut vg);
    tensor_reduce_window("reduce_window_mean", &mut env, &mut vg);
    // Movement primitives whose RISC lowering reads window parameters from
    // `args[1..]`. The `tensor_unop` scheme below only declares the arity-1
    // fallback; `reshape` and `permute` already have dedicated `infer_*_app`
    // paths in `infer.rs` that accept the parameterized arity. `pad`,
    // `shrink`, and `stride` do not — see issue Chelis-Lang/chelis#187 and
    // the matching dedicated `infer_shrink_app` / `infer_stride_app` paths.
    tensor_unop("reshape", &mut env, &mut vg);
    tensor_unop("permute", &mut env, &mut vg);
    tensor_expand_to_out("expand", &mut env, &mut vg);
    tensor_unop("pad", &mut env, &mut vg);
    tensor_unop("shrink", &mut env, &mut vg);
    tensor_unop("stride", &mut env, &mut vg);
    tensor_with_rate("dropout", &mut env, &mut vg);
    generic_unop_borrow("print", &mut env, &mut vg);
    generic_unop("fail", &mut env, &mut vg);
    generic_unop_borrow_same("debug", &mut env, &mut vg);

    // Test builtins. Effects (`! { Test }`) are not carried on the scheme itself;
    // they are assigned by chelis-effects when the name is encountered in an `app`
    // node, mirroring how `print` / `read_file` acquire IO.
    env.bind(
        "test_assert".to_string(),
        Scheme {
            tvars: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::Bool), Type::Prim(Prim::String)],
                Box::new(Type::Unit),
            ),
        },
    );
    env.bind(
        "test_assert_eq_f32".to_string(),
        Scheme {
            tvars: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Prim(Prim::F32),
                    Type::Prim(Prim::F32),
                    Type::Prim(Prim::String),
                ],
                Box::new(Type::Unit),
            ),
        },
    );
    env.bind(
        "test_assert_eq_int".to_string(),
        Scheme {
            tvars: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Prim(Prim::Int64),
                    Type::Prim(Prim::Int64),
                    Type::Prim(Prim::String),
                ],
                Box::new(Type::Unit),
            ),
        },
    );
    env.bind(
        "test_assert_eq_bool".to_string(),
        Scheme {
            tvars: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Prim(Prim::Bool),
                    Type::Prim(Prim::Bool),
                    Type::Prim(Prim::String),
                ],
                Box::new(Type::Unit),
            ),
        },
    );
    env.bind(
        "test_assert_eq_string".to_string(),
        Scheme {
            tvars: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Prim(Prim::String),
                    Type::Prim(Prim::String),
                    Type::Prim(Prim::String),
                ],
                Box::new(Type::Unit),
            ),
        },
    );
    {
        // test_assert_close_tensor: (tensor a, tensor a, f32, string) -> unit
        let tensor_tv = vg.fresh_tvar();
        env.bind(
            "test_assert_close_tensor".to_string(),
            Scheme {
                tvars: vec![tensor_tv],
                dvars: vec![],
                rvars: vec![],
                body: Type::Fn(
                    vec![
                        borrowed(Type::Var(tensor_tv)),
                        borrowed(Type::Var(tensor_tv)),
                        Type::Prim(Prim::F32),
                        Type::Prim(Prim::String),
                    ],
                    Box::new(Type::Unit),
                ),
            },
        );
    }
    {
        // test_assert_eq_tensor_int64: (tensor a, tensor a, string) -> unit
        // Bit-exact comparison; the Std.Test wrapper restricts a to int64.
        let tensor_tv = vg.fresh_tvar();
        env.bind(
            "test_assert_eq_tensor_int64".to_string(),
            Scheme {
                tvars: vec![tensor_tv],
                dvars: vec![],
                rvars: vec![],
                body: Type::Fn(
                    vec![
                        borrowed(Type::Var(tensor_tv)),
                        borrowed(Type::Var(tensor_tv)),
                        Type::Prim(Prim::String),
                    ],
                    Box::new(Type::Unit),
                ),
            },
        );
    }
    generic_unop("string_len", &mut env, &mut vg);
    generic_binop("string_concat", &mut env, &mut vg);
    generic_unop("string_trim", &mut env, &mut vg);
    generic_triop("string_slice", &mut env, &mut vg);
    env.bind(
        "string_contains".to_string(),
        Scheme {
            tvars: vec![],
            dvars: vec![],
            rvars: vec![],
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
            rvars: vec![],
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
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::String), Type::Prim(Prim::String)],
                Box::new(Type::Prim(Prim::Bool)),
            ),
        },
    );
    generic_unop_borrow("to_string", &mut env, &mut vg);
    generic_unop("to_int", &mut env, &mut vg);
    generic_unop("to_float", &mut env, &mut vg);
    generic_unop_borrow("rank", &mut env, &mut vg);
    generic_binop_first_borrow("shape", &mut env, &mut vg);
    generic_unop_borrow("numel", &mut env, &mut vg);
    generic_unop_borrow("tensor_to_scalar", &mut env, &mut vg);
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
            rvars: vec![],
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
            rvars: vec![],
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
    // `tensor_scan(initial: T, fn: (T, int64) -> T, n: int64) -> tensor[n, T]`.
    // The actual constraint shape (scalar `T`, callback signature, int64 `n`,
    // tensor return) is enforced by the special-case arm in
    // `crates/chelis-types/src/infer.rs` so error reporting can pinpoint each
    // role independently. This loose generic scheme is the type-env entry
    // point; it lets the inference engine see three argument slots and a
    // return slot it will overwrite. Same shape as `fold`/`scan` above.
    let tensor_scan_a = vg.fresh_tvar();
    let tensor_scan_b = vg.fresh_tvar();
    let tensor_scan_c = vg.fresh_tvar();
    let tensor_scan_ret = vg.fresh_tvar();
    env.bind(
        "tensor_scan".to_string(),
        Scheme {
            tvars: vec![tensor_scan_a, tensor_scan_b, tensor_scan_c, tensor_scan_ret],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Var(tensor_scan_a),
                    Type::Var(tensor_scan_b),
                    Type::Var(tensor_scan_c),
                ],
                Box::new(Type::Var(tensor_scan_ret)),
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
    generic_unop_borrow("to_list", &mut env, &mut vg);
    generic_binop("pad_sequences", &mut env, &mut vg);
    generic_triop("pad_sequences_to", &mut env, &mut vg);
    generic_unop("read_file", &mut env, &mut vg);
    generic_binop("write_file", &mut env, &mut vg);
    generic_unop("read_lines", &mut env, &mut vg);
    generic_unop("read_bytes", &mut env, &mut vg);
    generic_unop("file_exists", &mut env, &mut vg);
    generic_unop("list_dir", &mut env, &mut vg);
    generic_unop("mmap_file", &mut env, &mut vg);
    // `process_run(cmd, args)` is an eval/test-only subprocess exec builtin
    // (Hull Phase 0a). The 2-arg `generic_binop` scheme declares the arity;
    // the concrete return tuple `(Int64, String, String)` is pinned in
    // `infer.rs` and the IO effect is assigned in `chelis-effects`, mirroring
    // how `read_file` acquires IO. Rejected by the C/HIP build backends.
    generic_binop("process_run", &mut env, &mut vg);
    generic_triop("mmap_read", &mut env, &mut vg);
    generic_unop("mmap_len", &mut env, &mut vg);
    generic_triop_second_third_borrow("einsum", &mut env, &mut vg);
    generic_triop_first_borrow("split", &mut env, &mut vg);
    generic_triop_first_two_borrow("gather", &mut env, &mut vg);
    generic_pentaop("scatter", &mut env, &mut vg);
    // scatter_replace is the tensor-lane sparse builtin that lowers
    // to RiscOp::Scatter (last-write-wins). Distinct from the
    // host-lane `scatter(base, indices, updates, axis, mode)` which
    // remains the pentaop form with a string mode argument.
    generic_quadop("scatter_replace", &mut env, &mut vg);
    generic_triop_all_borrow("where", &mut env, &mut vg);
    generic_binop_first_borrow("cumsum", &mut env, &mut vg);
    generic_binop_first_borrow("sort", &mut env, &mut vg);
    generic_triop_first_borrow("diagonal", &mut env, &mut vg);
    generic_triop_first_borrow("trace", &mut env, &mut vg);
    generic_triop_all_borrow("clamp", &mut env, &mut vg);

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
            rvars: vec![],
            body: Type::Fn(vec![Type::Var(option_tvar)], Box::new(option_type.clone())),
        },
    );
    env.bind(
        "None".to_string(),
        Scheme {
            tvars: vec![option_tvar],
            dvars: vec![],
            rvars: vec![],
            body: option_type.clone(),
        },
    );

    adt_reg
        .defs
        .entry("Option".to_string())
        .or_insert_with(|| AdtDef {
            name: "Option".to_string(),
            type_params: vec!["a".to_string()],
            param_vars: vec![option_tvar],
            opaque: false,
            defining_module: None,
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
            rvars: vec![],
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
            rvars: vec![],
            body: list_type.clone(),
        },
    );

    adt_reg
        .defs
        .entry("List".to_string())
        .or_insert_with(|| AdtDef {
            name: "List".to_string(),
            type_params: vec!["a".to_string()],
            param_vars: vec![list_tvar],
            opaque: false,
            defining_module: None,
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

    adt_reg
        .defs
        .entry("MappedFile".to_string())
        .or_insert_with(|| AdtDef {
            name: "MappedFile".to_string(),
            type_params: Vec::new(),
            param_vars: Vec::new(),
            opaque: false,
            defining_module: None,
            variants: Vec::new(),
        });
}

/// Names that the inference engine should special-case for return type.
/// cmplt, eq, neq, lt, lte, gte return tensor[D, bool] instead of tensor[D, p].
#[allow(dead_code)] // Used by inference engine (Step 6)
pub const COMPARISON_OPS: &[&str] = &["cmplt", "eq", "neq", "lt", "gt", "lte", "gte"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_env_has_add() {
        let (env, _) = builtin_env();
        assert!(env.lookup("add").is_some());
    }

    // chelis#258 Tier-2 rank polymorphism: shape-class classification lock.

    /// Pin the exact shape-identity allowlist. A change here is the one place
    /// where a builtin becomes admissible inside a rank-polymorphic `..r`
    /// body, so it must be deliberate: misclassifying a shape-rewriting op as
    /// Identity is a §4.2 soundness hole. Every other builtin must be
    /// `Rewriting` (the safe default).
    #[test]
    fn shape_class_identity_set_is_pinned() {
        let identity: &[&str] = &[
            "add",
            "mul",
            "sub",
            "div",
            "mod",
            "max_elem",
            "min_elem",
            "neg",
            "recip",
            "exp",
            "log",
            "sin",
            "sqrt",
            "cos",
            "tan",
            "atan",
            "abs",
            "floor",
            "ceil",
            "relu",
            "sigmoid",
            "tanh",
            "silu",
            "gelu",
            "not",
            "clamp",
            "uniform_like",
            "where",
            "eq",
            "neq",
            "lt",
            "gt",
            "lte",
            "gte",
            "cmplt",
            "bitand",
            "bitor",
            "bitxor",
            "shl",
            "shr",
            "and",
            "or",
        ];
        // Named-axis reductions and named-axis expand are NameTracked
        // (admitted in a `..r` body — the procedural arm is the gate);
        // everything else outside `identity` is Rewriting.
        let name_tracked: &[&str] = &["sum", "mean", "expand"];
        for name in BUILTIN_NAMES {
            let expected = if identity.contains(name) {
                ShapeClass::Identity
            } else if name_tracked.contains(name) {
                ShapeClass::NameTracked
            } else {
                ShapeClass::Rewriting
            };
            assert_eq!(
                shape_class(name),
                expected,
                "builtin `{name}` shape-class drifted from the pinned set"
            );
        }
        // Spot-check the positional shape-rewriters stay Rewriting (the §4.2
        // traps): a positional index is meaningless at symbolic rank.
        for op in ["permute", "reshape", "matmul", "gather", "conv2d"] {
            assert_eq!(
                shape_class(op),
                ShapeClass::Rewriting,
                "`{op}` must be Rewriting"
            );
        }
        // And the named-axis ops are admitted as NameTracked. `expand`
        // moved from Rewriting in chelis#339: its procedural arm now
        // rejects positional axes at symbolic rank, so admitting it in a
        // `..r` body cannot hide a transposition.
        assert_eq!(shape_class("sum"), ShapeClass::NameTracked);
        assert_eq!(shape_class("mean"), ShapeClass::NameTracked);
        assert_eq!(shape_class("expand"), ShapeClass::NameTracked);
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
    fn builtin_env_has_process_run_and_it_is_a_known_name() {
        // Hull subprocess exec: process_run is a registered 2-arg builtin and
        // appears in the closed BUILTIN_NAMES vocabulary (so the lint naming
        // gate and host-lane resolver recognize it).
        let (env, _) = builtin_env();
        let scheme = env.lookup("process_run").expect("process_run registered");
        match &scheme.body {
            Type::Fn(params, _) => assert_eq!(
                params.len(),
                2,
                "process_run takes (cmd, args), got arity {}",
                params.len()
            ),
            other => panic!("process_run should be a function type, got {other:?}"),
        }
        assert!(
            BUILTIN_NAMES.contains(&"process_run"),
            "process_run must be in the closed BUILTIN_NAMES vocabulary"
        );
    }

    #[test]
    fn builtin_env_does_not_register_unknown_name() {
        // Negative parity: a name we never register stays absent, so the
        // process_run presence assertion above is not vacuously true.
        let (env, _) = builtin_env();
        assert!(env.lookup("process_run_definitely_unregistered").is_none());
        assert!(!BUILTIN_NAMES.contains(&"process_run_definitely_unregistered"));
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
        let tensor_f32 = Type::Tensor(
            vec![Dim::Name("batch".into())],
            TensorPrec::Concrete(Prim::F32),
        );
        let expected_fn = Type::Fn(
            vec![
                Type::Ref(Box::new(tensor_f32.clone())),
                Type::Ref(Box::new(tensor_f32.clone())),
            ],
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
        let t1 = Type::Tensor(
            vec![Dim::Name("batch".into())],
            TensorPrec::Concrete(Prim::F32),
        );
        let t2 = Type::Tensor(
            vec![Dim::Name("batch".into())],
            TensorPrec::Concrete(Prim::Bf16),
        );
        let bad_fn = Type::Fn(
            vec![Type::Ref(Box::new(t1)), Type::Ref(Box::new(t2))],
            Box::new(Type::Var(vg.fresh_tvar())),
        );

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
        let t1 = Type::Tensor(
            vec![Dim::Name("batch".into())],
            TensorPrec::Concrete(Prim::F32),
        );
        let t2 = Type::Prim(Prim::Int32);
        let bad_fn = Type::Fn(
            vec![Type::Ref(Box::new(t1)), Type::Ref(Box::new(t2))],
            Box::new(Type::Var(vg.fresh_tvar())),
        );

        let mut subst = Subst::new();
        assert!(unify(&add_ty, &bad_fn, &mut subst).is_err());
    }
}
