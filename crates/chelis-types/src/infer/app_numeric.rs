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
    "tan",
    "atan",
    "sqrt",
    "floor",
    "ceil",
    "round",
    "relu",
    "sigmoid",
    "tanh",
    "silu",
    "gelu",
    "matmul",
    "layer_norm",
    "max_elem",
    "min_elem",
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

/// Arithmetic operations for which `bool` has no authored numeric meaning.
/// Logical `and`/`or`/`not` remain the bool operations.
pub(super) const BOOL_REJECTED_ARITH_OPS: &[&str] = &["add", "sub", "mul", "neg", "floor_div"];

/// chelis#1805: the dtype family an operation's operand admission is stated
/// over.
///
/// The Float, Int and Numeric policies admit the active dtypes of their
/// family. A sufficient authored bound decides admission without choosing a
/// concrete dtype; inferred operands accumulate the same checked restriction.
/// Operations outside this family policy retain their own admission rules.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum OperandFamily {
    /// `f32`, `f64`, `bf16`, `f16`.
    Float,
    /// The active signed integer dtypes.
    Integer,
    Numeric,
}

impl OperandFamily {
    /// The [04-DTYPE-2] bound that admits this family and nothing wider.
    pub(super) fn required_bound(self) -> TypeVarRestriction {
        match self {
            OperandFamily::Float => TypeVarRestriction::ActiveFloat,
            OperandFamily::Integer => TypeVarRestriction::ActiveInt,
            OperandFamily::Numeric => TypeVarRestriction::ActiveNumeric,
        }
    }

    /// Whether a declared bound admits only dtypes this family admits.
    ///
    /// `Numeric` admits integers, so it does NOT satisfy a float-only policy:
    /// a bound that admits one inadmissible dtype cannot make the operation
    /// well typed at every instantiation the binder allows.
    pub(super) fn satisfied_by(self, bound: TypeVarRestriction) -> bool {
        bound.intersect(self.required_bound()) == Some(bound)
    }

    /// The binder spelling the repair hint offers.
    fn binder_spelling(self) -> &'static str {
        match self {
            OperandFamily::Float => "`[p: Float]`",
            OperandFamily::Integer => "`[p: Int]`",
            OperandFamily::Numeric => "`[p: Numeric]`",
        }
    }

    /// What the operand must carry, for the second half of the hint.
    fn operand_gloss(self) -> &'static str {
        match self {
            OperandFamily::Float => "an active float dtype",
            OperandFamily::Integer => "an active signed integer dtype",
            OperandFamily::Numeric => "an active numeric dtype",
        }
    }
}

/// The builtin scheme's family contract, translated for direct diagnostics.
/// Signature-bounded dropout and assert-close retain their own precise schemes.
pub(super) fn operand_family_policy(fname: &str) -> Option<OperandFamily> {
    crate::builtins::operand_dtype_family(fname).map(|family| match family {
        TypeVarRestriction::ActiveFloat => OperandFamily::Float,
        TypeVarRestriction::ActiveInt => OperandFamily::Integer,
        TypeVarRestriction::ActiveNumeric => OperandFamily::Numeric,
        _ => unreachable!("builtin precision contract is a dtype family"),
    })
}

/// chelis#1805: how a dtype-policy diagnostic names the operand's precision.
///
/// One value per diagnostic, so each message below keeps one format string: the
/// concrete spelling reproduces the bytes the message carried before a precision
/// variable could reach it, and the bounded spelling names the binder and its
/// declared family rather than a dtype the source never wrote.
///
/// An absent authored bound is diagnosed by the declaration-contract check,
/// which compares the accumulated requirement with the original signature.
/// This direct-operation diagnostic names concrete or explicitly bounded
/// operands; transported restrictions use the unifier's family diagnostic.
pub(super) enum PrecisionSubject {
    /// The dtype the operand carries.
    Concrete(String),
    /// A binder whose declared bound admits a dtype this operation does not.
    Bounded(String, TypeVarRestriction),
}

impl PrecisionSubject {
    /// The subject, where a concrete dtype spelling would appear.
    pub(super) fn render(&self) -> String {
        match self {
            PrecisionSubject::Concrete(name) => format!("`{name}`"),
            PrecisionSubject::Bounded(name, bound) => format!(
                "`{name}` (a precision variable bounded by {}, {})",
                bound.bound_description(),
                bound.membership_gloss()
            ),
        }
    }

    /// A dtype-kind qualifier only a concrete subject earns.
    ///
    /// A precision variable is inadmissible because of how it was DECLARED, so
    /// calling it an integer or a float would name a dtype the source never
    /// wrote. The concrete arms keep the word they always carried.
    fn qualifier(&self, word: &str) -> String {
        match self {
            PrecisionSubject::Concrete(_) => format!("{word} "),
            _ => String::new(),
        }
    }

    /// The [04-DTYPE-2] repair, for a variable subject only. A concrete dtype's
    /// repair is the operation's own cast hint, which follows this one.
    pub(super) fn binder_hint(&self, required: OperandFamily) -> Option<String> {
        match self {
            PrecisionSubject::Concrete(_) => None,
            PrecisionSubject::Bounded(_, bound) => Some(format!(
                "spec/04-type-system.md §5.9 [04-DTYPE-2]: the binder's `{}` bound admits \
                 dtypes this operation does not. Declare it {}, or give the operand {}.",
                bound.bound_spelling(),
                required.binder_spelling(),
                required.operand_gloss()
            )),
        }
    }
}

/// chelis#1805: the family-policy rejection for `fname` over `subject`.
///
/// ONE implementation, reached from both paths: the concrete arms of
/// [`operand_dtype_rejection`] hand it the operand's dtype, and the
/// precision-variable path hands it the binder. A message therefore cannot
/// drift between the diagnostic a settled `i32` gets and the one its `[p]`
/// binder gets, which is the property the two paths exist to share.
///
/// Every callee [`operand_family_policy`] names has an arm here. A deferred
/// entry that becomes concrete uses the same rejection when replayed.
pub(super) fn family_policy_rejection(
    fname: &str,
    subject: &PrecisionSubject,
    required: OperandFamily,
) -> (CheckErrorKind, String, Vec<String>) {
    let rendered = subject.render();
    let (kind, message, mut hints) = match fname {
        _ if (required != OperandFamily::Float && fname != "trunc_div") || fname == "matmul" => (
            CheckErrorKind::PrecisionMismatch,
            format!(
                "{fname} requires {} but operand precision {rendered} admits other types (spec/04-type-system.md §3.1, [04-DTYPE-2])",
                required.operand_gloss()
            ),
            vec![],
        ),
        "mean" => (
            CheckErrorKind::PrecisionMismatch,
            format!(
                "mean on operand precision {rendered} is not admitted per \
                 the chelis#724 capability decision: mean is float-only \
                 (f32, f64, bf16, f16). An integer mean has no authored \
                 rounding, and a fractional result inside an integer tensor \
                 violates spec/04-type-system.md section 9 [04-NUM-1]"
            ),
            vec![
                "chelis#724: cast to a float precision first, e.g. \
                 `mean(cast(x, f32), 0)`."
                    .to_string(),
            ],
        ),
        "softmax" => (
            CheckErrorKind::PrecisionMismatch,
            format!(
                "softmax on operand precision {rendered} is not admitted per \
                 spec/04-type-system.md §5.4: transcendental operations are \
                 restricted to f32, f64, bf16, f16 (not integer)"
            ),
            vec![
                "spec/04-type-system.md §5.4: cast to a float precision before \
                 applying softmax."
                    .to_string(),
            ],
        ),
        "div" => (
            CheckErrorKind::PrecisionMismatch,
            format!(
                "div on {}operand precision {rendered} is not admitted per \
                 spec/05-risc-primitives.md §2.1: `div` is float-only (IEEE-754). \
                 Use `floor_div` (round toward -inf) or `trunc_div` (round toward \
                 zero) for integers.",
                subject.qualifier("integer")
            ),
            vec![
                "spec/05-risc-primitives.md §2.1: integer division uses `floor_div` \
                 or `trunc_div`; `div` requires float operands."
                    .to_string(),
            ],
        ),
        "trunc_div" => (
            CheckErrorKind::PrecisionMismatch,
            format!(
                "trunc_div on {}operand precision {rendered} is not admitted \
                 per spec/05-risc-primitives.md §2.1: `trunc_div` is integer-only. \
                 Use `div` for IEEE-754 float division, or `floor_div` for a floored \
                 float quotient.",
                subject.qualifier("float")
            ),
            vec![
                "spec/05-risc-primitives.md §2.1: `trunc_div` requires integer \
                 operands."
                    .to_string(),
            ],
        ),
        // The twelve transcendentals share one text, as they always have.
        _ => (
            CheckErrorKind::PrecisionMismatch,
            format!(
                "{fname} on operand precision {rendered} is not admitted per \
                 spec/04-type-system.md §5.4: transcendental operations are \
                 restricted to f32, f64, bf16, f16 (not integer)"
            ),
            vec![format!(
                "spec/04-type-system.md §5.4: cast to a float precision before \
                 applying `{fname}`."
            )],
        ),
    };
    if let Some(binder) = subject.binder_hint(required) {
        hints.insert(0, binder);
    }
    (kind, message, hints)
}

/// chelis#1805: what a family-policy operation does with a tensor operand whose
/// precision is still a variable.
pub(super) enum PrecisionVerdict {
    /// The operation states no family policy, or the variable's declared bound
    /// admits only dtypes the policy admits.
    Admit,
    /// The bound admits a dtype this operation does not. Decided here with no
    /// ledger entry: [04-DTYPE-2] puts the bound in the binder list, so the
    /// declaration already carries everything the decision needs and no call
    /// site can change it.
    Reject(PrecisionSubject, OperandFamily),
}

/// Check a family policy against an authored precision variable.
///
/// `name` is the subject spelling: the declared binder per [04-FIT-9], or the
/// inference identity per [04-FIT-10].
pub(super) fn precision_variable_verdict(
    fname: &str,
    var: TypeVar,
    name: &str,
    subst: &Subst,
) -> PrecisionVerdict {
    let Some(required) = operand_family_policy(fname) else {
        return PrecisionVerdict::Admit;
    };
    match subst.tvar_restriction(var) {
        Some(bound) if required.satisfied_by(bound) => PrecisionVerdict::Admit,
        Some(bound) => {
            PrecisionVerdict::Reject(PrecisionSubject::Bounded(name.to_string(), bound), required)
        }
        None => {
            // Carry the operation requirement through ordinary inference.
            // The declaration's rigidity check compares this inferred domain
            // with its authored bound; a caller cannot silently narrow it.
            subst
                .narrow_tvar_restriction(var, required.required_bound())
                .expect("an unrestricted precision admits its first family constraint");
            PrecisionVerdict::Admit
        }
    }
}

/// Check or accumulate a family policy on a tensor operand
/// whose precision is a variable.
///
/// `Some` means the call was REJECTED, the shape the validators' own `reject!`
/// produces. All three validators route their precision arm through here, so
/// one operand shape cannot be handled three ways.
pub(super) fn decide_precision_variable_operand(
    node: &DeepNode,
    fname: &str,
    var: TypeVar,
    env: &Env,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type> {
    let authored = env
        .active_declared_type_names()
        .to_sorted()
        .into_iter()
        .find(|(declared, _)| subst.apply(&Type::Var(**declared)) == Type::Var(var))
        .map(|(_, name)| name.clone());
    let Some(name) = authored else {
        if let Some(required) = operand_family_policy(fname)
            && let Err(error) = subst.narrow_tvar_restriction(var, required.required_bound())
        {
            return Some(report(errors, error.into()));
        }
        return None;
    };
    match precision_variable_verdict(fname, var, &name, subst) {
        PrecisionVerdict::Admit => None,
        PrecisionVerdict::Reject(subject, required) => {
            let (kind, message, hints) = family_policy_rejection(fname, &subject, required);
            reject(
                errors,
                CheckError::new(kind, with_node_provenance(node, message), hints),
            )
        }
    }
}

/// The post-desugar operand-dtype policy chokepoint from chelis#860.
///
/// Direct applications, reduction data arguments, and bare pipe stages all
/// consult this function. Unresolved operands carry the builtin scheme's
/// checked restriction through ordinary unification; no callee-body walk is used.
pub(super) fn operand_dtype_rejection(
    fname: &str,
    resolved: &Type,
) -> Option<(CheckErrorKind, String, Vec<String>)> {
    let bool_operand = matches!(
        resolved,
        Type::Tensor(_, TensorPrec::Concrete(Prim::Bool)) | Type::Prim(Prim::Bool)
    );
    if bool_operand && BOOL_REJECTED_ARITH_OPS.contains(&fname) {
        return Some((
            CheckErrorKind::PrecisionMismatch,
            format!(
                "{fname} on bool operands is not admitted per the chelis#726 \
                 capability decision and spec/04-type-system.md section 9 \
                 [04-NUM-4]: bool is exactly {{0, 1}} and not a numeric dtype, \
                 so arithmetic on it has no authored meaning"
            ),
            vec![
                "chelis#726: use first-class `count(x, axes...)` to count true values, \
                 and `and`/`or`/`not` for bool logic."
                    .to_string(),
            ],
        ));
    }

    if fname == "mean" {
        let non_float_elem = match resolved {
            Type::Tensor(_, TensorPrec::Concrete(prim)) if !prim.is_float() => Some(prim.name()),
            Type::Prim(prim) if !prim.is_float() => Some(prim.name()),
            _ => None,
        };
        if let Some(prim_name) = non_float_elem {
            // chelis#1805: the concrete path renders through the same function
            // the precision-variable path does, so the two cannot drift.
            return Some(family_policy_rejection(
                fname,
                &PrecisionSubject::Concrete(prim_name.to_string()),
                OperandFamily::Float,
            ));
        }
        return None;
    }

    if fname == "softmax" {
        if let Type::Tensor(_, TensorPrec::Concrete(prim)) = resolved
            && !prim.is_float()
        {
            return Some(family_policy_rejection(
                fname,
                &PrecisionSubject::Concrete(prim.name().to_string()),
                OperandFamily::Float,
            ));
        }
        return None;
    }

    if !TENSOR_OPS.contains(&fname) {
        return None;
    }

    let ok = match fname {
        "matmul" => {
            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
        }
        "layer_norm" => {
            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                || matches!(resolved, Type::Prim(prim) if prim.is_float())
        }
        "add" | "mul" | "sub" | "max_elem" | "min_elem" | "neg" | "floor_div" | "floor"
        | "ceil" | "round" => {
            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                || matches!(resolved, Type::Prim(prim) if prim.is_numeric())
        }
        "div" => {
            matches!(
                resolved,
                Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error(_)
            ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_float())
                || matches!(resolved, Type::Prim(prim) if prim.is_float())
        }
        "trunc_div" => {
            matches!(
                resolved,
                Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error(_)
            ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_integer())
                || matches!(resolved, Type::Prim(prim) if prim.is_integer())
        }
        "exp" | "log" | "sin" | "tan" | "atan" | "sqrt" | "relu" | "sigmoid" | "tanh" | "silu"
        | "gelu" | "recip" => {
            matches!(
                resolved,
                Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error(_)
            ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_float())
                || matches!(resolved, Type::Prim(prim) if prim.is_float())
        }
        "cmplt" | "lt" | "gt" | "lte" | "gte" => {
            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_))
                || matches!(resolved, Type::Prim(prim) if prim.is_numeric())
        }
        // [05-OP-36]: a structured operand's reachable fields are decided by
        // `equality_domain`, which runs before this policy.
        "eq" | "neq" => matches!(
            resolved,
            Type::Tensor(_, _)
                | Type::Var(_)
                | Type::Error(_)
                | Type::Prim(_)
                | Type::Unit
                | Type::Tuple(_)
                | Type::Adt(..)
                | Type::KindedAdt(..)
        ),
        "and" | "or" | "not" => {
            matches!(
                resolved,
                Type::Tensor(_, TensorPrec::Concrete(Prim::Bool)) | Type::Var(_) | Type::Error(_)
            ) || matches!(resolved, Type::Prim(Prim::Bool))
        }
        _ => matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error(_)),
    };
    if ok {
        return None;
    }

    let is_transcendental = matches!(
        fname,
        "exp"
            | "log"
            | "sin"
            | "tan"
            | "atan"
            | "sqrt"
            | "relu"
            | "sigmoid"
            | "tanh"
            | "silu"
            | "gelu"
            | "recip"
    );
    let resolved_int_prim = match resolved {
        Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_integer() => Some(prim.name()),
        Type::Prim(prim) if prim.is_integer() => Some(prim.name()),
        _ => None,
    };
    let resolved_float_prim = match resolved {
        Type::Tensor(_, TensorPrec::Concrete(prim)) if prim.is_float() => Some(prim.name()),
        Type::Prim(prim) if prim.is_float() => Some(prim.name()),
        _ => None,
    };

    if is_transcendental
        && let Type::Tensor(_, TensorPrec::Concrete(prim)) = resolved
        && !prim.is_float()
    {
        Some(family_policy_rejection(
            fname,
            &PrecisionSubject::Concrete(prim.name().to_string()),
            OperandFamily::Float,
        ))
    } else if fname == "div"
        && let Some(prim_name) = resolved_int_prim
    {
        Some(family_policy_rejection(
            fname,
            &PrecisionSubject::Concrete(prim_name.to_string()),
            OperandFamily::Float,
        ))
    } else if fname == "trunc_div"
        && let Some(prim_name) = resolved_float_prim
    {
        Some(family_policy_rejection(
            fname,
            &PrecisionSubject::Concrete(prim_name.to_string()),
            OperandFamily::Integer,
        ))
    } else {
        Some((
            CheckErrorKind::TypeMismatch,
            format!("{fname} does not accept argument type {resolved} in this context"),
            vec![],
        ))
    }
}

/// Check numeric, comparison, softmax, and reduction argument restrictions.
///
/// `None` permits later operation-family checks. `Some` carries the original
/// early rejection type and its diagnostic.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_numeric_and_reduction_arguments(
    node: &DeepNode,
    kids: &[deep::Expr],
    func_name: &Option<String>,
    arg_tys: &[Type],
    env: &Env,
    subst: &Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    route_observed: &mut bool,
    suspension: Option<&DtypeAdmissibilitySite<'_>>,
    result_ty: &Type,
    product: &mut InferenceProduct,
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
        *route_observed = true;
        for arg_ty in arg_tys {
            let resolved = type_for_readonly_check(arg_ty, subst);
            match &resolved {
                // chelis#1512: not admissible YET. `operand_dtype_rejection`
                // admits every unresolved operand, so this loop decided
                // nothing about it and nothing revisited the decision once it
                // settled. Suspending the call runs this same loop against the
                // bound type, where the policy it enforces is the one a direct
                // call gets.
                Type::Var(_) => {
                    if let Some(site) = suspension {
                        site.register(arg_tys, result_ty, subst, product);
                    }
                }
                // chelis#1805: a tensor at an unresolved PRECISION. The outer
                // constructor is known, so the readiness predicate answers
                // ready and the arm below admits it for good; a family policy
                // still has a verdict, from the variable's bound or from the
                // declaration boundary.
                Type::Tensor(_, TensorPrec::Var(var)) => {
                    if let Some(rejected) =
                        decide_precision_variable_operand(node, fname, *var, env, subst, errors)
                    {
                        return Some(rejected);
                    }
                }
                _ => {
                    if matches!(fname.as_str(), "eq" | "neq") {
                        match equality_domain(&resolved, subst, adt_reg) {
                            EqualityDomain::Admitted => {}
                            // chelis#2587: a field type that is still a
                            // variable is not admissible YET. The call resumes
                            // once it binds, and the declaration boundary
                            // decides one that never does.
                            EqualityDomain::Awaits => {
                                if let Some(site) = suspension {
                                    site.register(arg_tys, result_ty, subst, product);
                                }
                                continue;
                            }
                            EqualityDomain::Rejected(incomparable) => {
                                let (kind, message, hints) =
                                    equality_domain_rejection(fname, &resolved, &incomparable);
                                reject!(
                                    errors,
                                    CheckError::new(
                                        kind,
                                        with_node_provenance(node, message),
                                        hints,
                                    ),
                                );
                            }
                        }
                    }
                    if let Some((kind, message, hints)) = operand_dtype_rejection(fname, &resolved)
                    {
                        reject!(
                            errors,
                            CheckError::new(kind, with_node_provenance(node, message,), hints,),
                        );
                    }
                }
            }
        }
    }

    if let Some(fname) = func_name
        && matches!(
            fname.as_str(),
            "softmax"
                | "mean"
                | "sum"
                | "count"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
        )
    {
        *route_observed = true;
        if let Some(first_arg) = arg_tys.first() {
            let resolved = type_for_readonly_check(first_arg, subst);
            match &resolved {
                // chelis#1512: the operand is not a tensor YET, and the dtype
                // rule below admits every unresolved type. Suspend so both
                // checks decide against the bound one.
                Type::Var(_) => {
                    if let Some(site) = suspension {
                        site.register(arg_tys, result_ty, subst, product);
                    }
                }
                // chelis#1805, as in the loop above.
                Type::Tensor(_, TensorPrec::Var(var)) => {
                    if let Some(rejected) =
                        decide_precision_variable_operand(node, fname, *var, env, subst, errors)
                    {
                        return Some(rejected);
                    }
                }
                Type::Tensor(_, _) | Type::Error(_) => {}
                _ => {
                    reject!(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!("{} expects tensor input, got {}", fname, resolved),
                            ),
                            vec![],
                        ),
                    );
                }
            }
            if let Some((kind, message, hints)) = operand_dtype_rejection(fname, &resolved) {
                reject!(
                    errors,
                    CheckError::new(kind, with_node_provenance(node, message,), hints,),
                );
            }
        }

        if let Some(axis_arg) = arg_tys.get(1) {
            let resolved = subst.apply(axis_arg);
            match &resolved {
                Type::Prim(Prim::Int32) => {}
                // chelis#1512: the axis is not an `i32` YET.
                Type::Var(_) => {
                    if let Some(site) = suspension {
                        site.register(arg_tys, result_ty, subst, product);
                    }
                }
                Type::Error(_) => {}
                _ => {
                    reject!(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!("{} expects i32 axis, got {}", fname, resolved),
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
        // Issue #216: cast-aware so `softmax(x, cast(N, i32))`
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
                    with_node_provenance(
                        node,
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

/// Type an integer binary operation or shift.
///
/// Unlike the operand-admissibility checks, these arms decide the call's
/// result type rather than only rejecting a bad one: `Some(ty)` short-circuits
/// the rest of application checking with `ty`, and `None` means `func_name` is
/// not one of these operations.
#[allow(clippy::too_many_arguments)]
pub(super) fn integer_binop_result_type(
    node: &DeepNode,
    func_name: Option<&str>,
    arg_tys: &[Type],
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
    suspension: Option<&DtypeAdmissibilitySite<'_>>,
    result_ty: &Type,
    product: &mut InferenceProduct,
) -> Option<Type> {
    if let Some(fname) = func_name
        && INT_BINOPS.contains(&fname)
    {
        let lhs = arg_tys
            .first()
            .map(|ty| subst.apply(ty))
            .unwrap_or_else(|| vg.fresh_type());
        let rhs = arg_tys
            .get(1)
            .map(|ty| subst.apply(ty))
            .unwrap_or_else(|| vg.fresh_type());
        match (&lhs, &rhs) {
            (Type::Prim(lhs_prec), Type::Prim(rhs_prec))
                if lhs_prec.is_integer() && rhs_prec.is_integer() && lhs_prec == rhs_prec =>
            {
                return Some(Type::Prim(*lhs_prec));
            }
            // chelis#1512: the operands are not known to MATCH yet. Each of
            // these three arms published the left operand without ever
            // comparing the two, so `mod(t, 3i32)` with `t` binding to `i64`
            // was accepted while the same call on a resolved `i64` is
            // rejected. Suspending re-runs this rule against both bound types,
            // and the arm above is the one that then decides.
            (Type::Var(_), Type::Prim(rhs_prec)) if rhs_prec.is_integer() => {
                if let Some(site) = suspension {
                    site.register(arg_tys, result_ty, subst, product);
                }
                return Some(lhs);
            }
            (Type::Prim(lhs_prec), Type::Var(_)) if lhs_prec.is_integer() => {
                if let Some(site) = suspension {
                    site.register(arg_tys, result_ty, subst, product);
                }
                return Some(lhs);
            }
            (Type::Var(_), Type::Var(_)) => {
                if let Some(site) = suspension {
                    site.register(arg_tys, result_ty, subst, product);
                }
                return Some(lhs);
            }
            // chelis#731 cascade suppression, split out of the arm above: an
            // error witness is not a deferred decision, and it never binds.
            (Type::Error(_), _) | (_, Type::Error(_)) => {
                return Some(lhs);
            }
            _ => {
                return reject(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        with_node_provenance(
                            node,
                            format!(
                                "{} requires matching integer arguments, got {} and {}",
                                fname, lhs, rhs
                            ),
                        ),
                        vec![],
                    ),
                );
            }
        }
    }

    if let Some(fname) = func_name
        && INT_SHIFT_OPS.contains(&fname)
    {
        let lhs = arg_tys
            .first()
            .map(|ty| subst.apply(ty))
            .unwrap_or_else(|| vg.fresh_type());
        let rhs = arg_tys
            .get(1)
            .map(|ty| subst.apply(ty))
            .unwrap_or_else(|| vg.fresh_type());
        // chelis#1512: admissibility used to be two `matches!` disjunctions
        // folded into one boolean, which admitted an unresolved operand with no
        // arm to suspend from. The arms below are that boolean, per operand
        // pair: each side is admissible when it is an integer primitive, an
        // unresolved variable, or an error witness, and the call is admitted
        // when both sides are.
        match (&lhs, &rhs) {
            (Type::Prim(lhs_prec), Type::Prim(rhs_prec))
                if lhs_prec.is_integer() && rhs_prec.is_integer() && lhs_prec == rhs_prec =>
            {
                return Some(lhs);
            }
            // At least one operand is still a variable and the other is
            // admissible. Suspend so this rule decides against the bound type:
            // `shl(t, 1i32)` with `t` binding to `f32` is rejected here now,
            // as the same call on a resolved `f32` always was.
            (Type::Var(_), Type::Var(_))
            | (Type::Var(_), Type::Error(_))
            | (Type::Error(_), Type::Var(_)) => {
                if let Some(site) = suspension {
                    site.register(arg_tys, result_ty, subst, product);
                }
                return Some(lhs);
            }
            (Type::Var(_), Type::Prim(prec)) | (Type::Prim(prec), Type::Var(_))
                if prec.is_integer() =>
            {
                if let Some(site) = suspension {
                    site.register(arg_tys, result_ty, subst, product);
                }
                return Some(lhs);
            }
            // chelis#731 cascade suppression: an error witness beside an
            // admissible operand, which the boolean above also admitted.
            (Type::Error(_), Type::Error(_)) => {
                return Some(lhs);
            }
            (Type::Error(_), Type::Prim(prec)) | (Type::Prim(prec), Type::Error(_))
                if prec.is_integer() =>
            {
                return Some(lhs);
            }
            _ => {
                return reject(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        with_node_provenance(
                            node,
                            format!(
                                "{} requires matching integer lhs and shift amount, got {} and {}",
                                fname, lhs, rhs
                            ),
                        ),
                        vec![],
                    ),
                );
            }
        }
    }

    None
}
