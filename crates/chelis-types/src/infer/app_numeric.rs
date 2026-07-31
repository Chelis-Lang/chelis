//! Numeric operation rules and precision diagnostics.

use super::*;

pub(super) const TENSOR_OPS: &[&str] = &[
    "add",
    "mul",
    "sub",
    "div",
    "floor_div",
    "trunc_div",
    "neg",
    "recip",
    "exp",
    "log",
    "sin",
    "sqrt",
    "relu",
    "sigmoid",
    "tanh",
    "silu",
    "gelu",
    "matmul",
    "layer_norm",
    "max_elem",
    "min_elem",
    "normalize",
    "cmplt",
    "eq",
    "neq",
    "lt",
    "gt",
    "lte",
    "gte",
    "and",
    "or",
    "not",
];

pub(super) const LOGICAL_OPS: &[&str] = &["and", "or", "not"];
pub(super) const INT_BINOPS: &[&str] = &["mod", "bitand", "bitor", "bitxor"];
pub(super) const INT_SHIFT_OPS: &[&str] = &["shl", "shr"];

/// Check numeric, comparison, softmax, and reduction argument restrictions.
///
/// `None` permits later operation-family checks. `Some` carries the original
/// early rejection type and its diagnostic.
pub(super) fn validate_numeric_and_reduction_arguments(
    list: &deep::List,
    kids: &[deep::Expr],
    func_name: &Option<String>,
    arg_tys: &[Type],
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type> {
    macro_rules! reject {
        ($($arg:tt)*) => {
            return Some(report($($arg)*))
        };
    }

    // Post-check: shared builtins can operate on either tensors or host scalars.
    if let Some(fname) = func_name
        && TENSOR_OPS.contains(&fname.as_str())
    {
        for arg_ty in arg_tys {
            let resolved = type_for_readonly_check(arg_ty, subst);
            let ok = match fname.as_str() {
                "matmul" | "layer_norm" | "normalize" => {
                    matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                }
                "add" | "mul" | "sub" | "max_elem" | "min_elem" | "neg" => {
                    matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                        || matches!(resolved, Type::Prim(prec) if prec.is_numeric())
                }
                "floor_div" => {
                    // chelis#178: `floor_div` accepts both integer
                    // and float operands (round toward -inf for
                    // ints, `floor(a/b)` for floats), so any
                    // numeric precision is admissible.
                    matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                        || matches!(resolved, Type::Prim(prec) if prec.is_numeric())
                }
                "div" => {
                    // chelis#178: `div` is float-only. Integer
                    // operands are a type error pointing at
                    // `floor_div` / `trunc_div` (see the rejection
                    // diagnostic below). An unresolved
                    // `TensorPrec::Var(_)` is accepted so a
                    // polymorphic body type-checks; the cross-row
                    // pass `validate_polymorphic_op_constraints`
                    // catches integer instantiations at the call
                    // site.
                    matches!(
                        resolved,
                        Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error(_)
                    ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(p)) if p.is_float())
                        || matches!(resolved, Type::Prim(prec) if prec.is_float())
                }
                "trunc_div" => {
                    // chelis#178: `trunc_div` is integer-only (the
                    // C/Rust truncating quotient). Float operands
                    // are a type error. Unresolved precision vars
                    // are accepted for polymorphic bodies.
                    matches!(
                        resolved,
                        Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error(_)
                    ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(p)) if p.is_integer())
                        || matches!(resolved, Type::Prim(prec) if prec.is_integer())
                }
                "exp" | "log" | "sin" | "sqrt" | "relu" | "sigmoid" | "tanh" | "silu" | "gelu"
                | "recip" => {
                    // WS-A8 / RT-3 F3: spec/04-type-system.md §5.4
                    // restricts transcendental ops to float
                    // precisions (f32, f64, bf16, f16). The
                    // tensor form was previously admitted with
                    // any precision, slipping integer instantiations
                    // past the type checker; the scalar form
                    // already enforced this. Tensors carrying
                    // an unresolved `TensorPrec::Var(_)` are
                    // accepted here so a polymorphic body
                    // type-checks; the cross-row enforcement
                    // pass `validate_polymorphic_op_constraints`
                    // catches integer instantiations at the
                    // call site.
                    matches!(
                        resolved,
                        Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error(_)
                    ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(p)) if p.is_float())
                        || matches!(resolved, Type::Prim(prec) if prec.is_float())
                }
                "cmplt" | "lt" | "gt" | "lte" | "gte" => {
                    matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                        || matches!(resolved, Type::Prim(prec) if prec.is_numeric())
                }
                "eq" | "neq" => {
                    matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                        || matches!(resolved, Type::Prim(_))
                }
                "and" | "or" | "not" => {
                    matches!(
                        resolved,
                        Type::Tensor(_, TensorPrec::Concrete(Prim::Bool))
                            | Type::Var(_)
                            | Type::Error(_)
                    ) || matches!(resolved, Type::Prim(Prim::Bool))
                }
                _ => matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_)),
            };
            if !ok {
                // WS-A8: surface a §5.4 citation when a
                // transcendental rejects an integer tensor.
                let is_transcendental = matches!(
                    fname.as_str(),
                    "exp"
                        | "log"
                        | "sin"
                        | "sqrt"
                        | "relu"
                        | "sigmoid"
                        | "tanh"
                        | "silu"
                        | "gelu"
                        | "recip"
                );
                let resolved_int_prec = match &resolved {
                    Type::Tensor(_, TensorPrec::Concrete(p)) if p.is_integer() => Some(p.name()),
                    Type::Prim(p) if p.is_integer() => Some(p.name()),
                    _ => None,
                };
                let resolved_float_prec = match &resolved {
                    Type::Tensor(_, TensorPrec::Concrete(p)) if p.is_float() => Some(p.name()),
                    Type::Prim(p) if p.is_float() => Some(p.name()),
                    _ => None,
                };
                let (kind, message, hints) = if is_transcendental
                    && let Type::Tensor(_, TensorPrec::Concrete(p)) = &resolved
                    && !p.is_float()
                {
                    (
                        CheckErrorKind::PrecisionMismatch,
                        format!(
                            "{fname} on operand precision `{}` is not admitted per \
                             spec/04-type-system.md \u{00a7}5.4: transcendental \
                             operations are restricted to f32, f64, bf16, f16 (not \
                             integer)",
                            p.name()
                        ),
                        vec![format!(
                            "spec/04-type-system.md \u{00a7}5.4: cast to a float \
                             precision before applying `{fname}`."
                        )],
                    )
                } else if fname == "div"
                    && let Some(pname) = resolved_int_prec
                {
                    // chelis#178: integer `div` is rejected; point
                    // the user at the integer-division ops.
                    (
                        CheckErrorKind::PrecisionMismatch,
                        format!(
                            "div on integer operand precision `{pname}` is not admitted \
                             per spec/05-risc-primitives.md \u{00a7}2.1: `div` is \
                             float-only (IEEE-754). Use `floor_div` (round toward -inf) \
                             or `trunc_div` (round toward zero) for integers."
                        ),
                        vec![
                            "spec/05-risc-primitives.md \u{00a7}2.1: integer division \
                             uses `floor_div` or `trunc_div`; `div` requires float \
                             operands."
                                .to_string(),
                        ],
                    )
                } else if fname == "trunc_div"
                    && let Some(pname) = resolved_float_prec
                {
                    // chelis#178: `trunc_div` is integer-only.
                    (
                        CheckErrorKind::PrecisionMismatch,
                        format!(
                            "trunc_div on float operand precision `{pname}` is not \
                             admitted per spec/05-risc-primitives.md \u{00a7}2.1: \
                             `trunc_div` is integer-only. Use `div` for IEEE-754 float \
                             division, or `floor_div` for a floored float quotient."
                        ),
                        vec![
                            "spec/05-risc-primitives.md \u{00a7}2.1: `trunc_div` requires \
                             integer operands."
                                .to_string(),
                        ],
                    )
                } else {
                    (
                        CheckErrorKind::TypeMismatch,
                        format!(
                            "{fname} does not accept argument type {resolved} in this \
                             context"
                        ),
                        vec![],
                    )
                };
                reject!(
                    errors,
                    CheckError::new(
                        kind,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            message,
                        ),
                        hints,
                    ),
                );
            }
        }
    }

    if let Some(fname) = func_name
        && matches!(
            fname.as_str(),
            "softmax"
                | "mean"
                | "sum"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
        )
    {
        if let Some(first_arg) = arg_tys.first() {
            let resolved = type_for_readonly_check(first_arg, subst);
            match &resolved {
                Type::Tensor(_, _) | Type::Var(_) | Type::Error(_) => {}
                _ => {
                    reject!(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("{} expects tensor input, got {}", fname, resolved),
                            ),
                            vec![],
                        ),
                    );
                }
            }
            // WS-A8 / RT-3 F3: softmax is a transcendental row
            // op per spec/04-type-system.md \u{00a7}5.4 and must
            // reject integer operand precisions. Polymorphic
            // (`Var`) precisions are accepted here so a poly
            // body type-checks; the cross-row enforcement pass
            // catches integer call-site instantiations.
            if fname == "softmax"
                && let Type::Tensor(_, TensorPrec::Concrete(p)) = &resolved
                && !p.is_float()
            {
                reject!(
                    errors,
                    CheckError::new(
                        CheckErrorKind::PrecisionMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            format!(
                                "softmax on operand precision `{}` is not admitted per \
                             spec/04-type-system.md \u{00a7}5.4: transcendental \
                             operations are restricted to f32, f64, bf16, f16 \
                             (not integer)",
                                p.name()
                            ),
                        ),
                        vec![format!(
                            "spec/04-type-system.md \u{00a7}5.4: cast to a float \
                         precision before applying softmax."
                        )],
                    ),
                );
            }
        }

        if let Some(axis_arg) = arg_tys.get(1) {
            let resolved = subst.apply(axis_arg);
            match &resolved {
                Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error(_) => {}
                _ => {
                    reject!(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("{} expects int32 axis, got {}", fname, resolved),
                            ),
                            vec![],
                        ),
                    );
                }
            }
        }

        // softmax does not go through `check_reduction_signature`
        // (it is shape-preserving, not shape-reducing), so its
        // axis range is validated here. Negative axes index from
        // the end via `normalize_static_axis`, consistent with
        // the reductions and gather/scatter.
        // Issue #216: cast-aware so `softmax(x, cast(N, int32))`
        // surfaces the bounds-check diagnostic at infer.
        if fname == "softmax"
            && let Some(first_arg) = arg_tys.first()
            && let Type::Tensor(dims, _) = type_for_readonly_check(first_arg, subst)
            && let Some(raw) = kids.get(2).and_then(extract_int_for_dim)
            && normalize_static_axis(dims.len(), raw).is_none()
        {
            reject!(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "softmax axis {raw} is out of bounds for rank {} tensor",
                            dims.len()
                        ),
                    ),
                    vec![],
                ),
            );
        }
    }

    None
}
