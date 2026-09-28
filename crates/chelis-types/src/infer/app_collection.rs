//! Collection operation and constructor rules.
//!
//! These helpers preserve list callback diagnostics and concat shape rules.

use super::*;

pub(super) fn collection_helper_type_error(
    node: &DeepNode,
    helper: &str,
    contract: &str,
    te: TypeError,
) -> CheckError {
    let kind = check_error_kind_from_type_error_kind(&te.kind);
    let suggestions = match kind {
        CheckErrorKind::PrecisionMismatch => vec!["Insert explicit cast".to_string()],
        _ => vec![],
    };
    CheckError::new(
        kind,
        with_node_provenance(node, format!("{helper} {contract}; {}", te.message)),
        suggestions,
    )
}

/// The single decision boundary for operations that combine stored values.
/// The immutable substitution is intentional: equality is checked privately,
/// then emitted as an origin equation, never committed as operand evidence.
/// Direct calls and transported list contracts both enter this boundary.
impl builtins::AggregateRule {
    pub(super) fn decide(
        self,
        operands: &[Type],
        result: &Type,
        subst: &Subst,
    ) -> Result<Option<ResultConstraint>, Box<CheckError>> {
        use builtins::AggregateRule;
        let arity = match self {
            AggregateRule::Append
            | AggregateRule::Concat
            | AggregateRule::DictMerge
            | AggregateRule::Fold
            | AggregateRule::Scan => 2,
            AggregateRule::DictInsert => 3,
        };
        let name = match self {
            AggregateRule::Append => "append",
            AggregateRule::Concat => "concat",
            AggregateRule::DictInsert => "dict_insert",
            AggregateRule::DictMerge => "dict_merge",
            AggregateRule::Fold => "fold",
            AggregateRule::Scan => "scan",
        };
        if operands.len() != arity {
            return Err(Box::new(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("{name} expects {arity} arguments, got {}", operands.len()),
                vec![],
            )));
        }
        let operands = operands
            .iter()
            .map(|ty| subst.apply(ty))
            .collect::<Vec<_>>();
        if operands.iter().any(|ty| matches!(ty, Type::Error(_))) {
            return Ok(None);
        }
        let inputs = match (self, operands.as_slice()) {
            (AggregateRule::Fold | AggregateRule::Scan, [initial, callback_result]) => {
                vec![initial.clone(), callback_result.clone()]
            }
            (AggregateRule::Append, [list @ Type::Adt(name, args), value])
                if name == "List" && args.len() == 1 =>
            {
                vec![
                    list.clone(),
                    Type::Adt("List".to_string(), vec![value.clone()]),
                ]
            }
            (AggregateRule::Concat, [left @ Type::Adt(a, aa), right @ Type::Adt(b, ba)])
                if a == "List" && b == "List" && aa.len() == 1 && ba.len() == 1 =>
            {
                vec![left.clone(), right.clone()]
            }
            (AggregateRule::DictInsert, [dict @ Type::Adt(name, args), key, value])
                if name == "Dict" && args.len() == 2 =>
            {
                vec![
                    dict.clone(),
                    Type::Adt("Dict".to_string(), vec![key.clone(), value.clone()]),
                ]
            }
            (AggregateRule::DictMerge, [left @ Type::Adt(a, aa), right @ Type::Adt(b, ba)])
                if a == "Dict" && b == "Dict" && aa.len() == 2 && ba.len() == 2 =>
            {
                vec![left.clone(), right.clone()]
            }
            // Unknown constructors still owe the operation's admission rule;
            // a value hole inside a known constructor owes only the equation.
            (_, values) if values.iter().any(|ty| matches!(ty, Type::Var(_))) => return Ok(None),
            _ => {
                return Err(Box::new(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "{}; got {}",
                        match self {
                            AggregateRule::Append =>
                                "append expects List input and a compatible value".to_string(),
                            _ => format!("{name} expects compatible collection operands"),
                        },
                        operands
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    vec![],
                )));
            }
        };
        let mut compatibility = subst.clone();
        for input in &inputs[1..] {
            unify(&inputs[0], input, &mut compatibility)
                .map_err(|error| {
                    let message = match self {
                    AggregateRule::Fold | AggregateRule::Scan => format!("{name} expects a callback whose accumulator/result type matches the initial accumulator; {}", error.message),
                    AggregateRule::Append => format!("append expects a value of the list's element type; {}", error.message),
                    AggregateRule::Concat => format!("concat expects matching List inputs; {}", error.message),
                    AggregateRule::DictInsert | AggregateRule::DictMerge => format!("{name} requires matching stored value types; {}", error.message),
                    };
                    let mut diagnostic: CheckError = error.into();
                    diagnostic.message = message;
                    Box::new(diagnostic)
                })?;
        }
        Ok(Some(ResultConstraint::Join {
            inputs,
            result: result.clone(),
        }))
    }
}

/// Stored-value equations retain their result origin. Scalar/tensor operation
/// signatures contain no callable origin to infer backwards, and may settle
/// directly. A computed result also retains its precise type separately from
/// the declared signature's compatibility check.
pub(crate) enum CollectionDecision {
    ResultOrigin(ResultConstraint),
    RequiredEquality {
        actual: Type,
        expected: Type,
    },
    /// A semantic rule computed this call's result. Its precise type remains
    /// available even when the callable's declaration contains wildcards.
    ProducedResult {
        declared: Type,
        produced: Type,
    },
}

impl CollectionDecision {
    pub(crate) fn publish(self, subst: &mut Subst) -> Result<(), TypeError> {
        match self {
            Self::ResultOrigin(equation) => {
                subst.record_result_constraint(equation);
                Ok(())
            }
            Self::RequiredEquality { actual, expected } => unify(&actual, &expected, subst),
            Self::ProducedResult { declared, produced } => unify(&declared, &produced, subst),
        }
    }

    pub(crate) fn produced_result(&self) -> Option<Type> {
        match self {
            Self::ProducedResult { produced, .. } => Some(produced.clone()),
            Self::ResultOrigin(_) | Self::RequiredEquality { .. } => None,
        }
    }
}

/// Decide a transported checked collection contract against settled operands.
///
/// Scheme instantiation installs a fresh relation instance on the
/// inference-local contract ledger. An application binds the exact instances
/// its callee produced to that call's arguments and evidence, then decides them
/// here after ordinary call unification has settled every available operand.
/// An unresolved consumed instance remains owned by that declaration boundary;
/// a merely returned or aggregated function value remains transportable.
///
/// The eager arms in `app_post.rs` and transported tensor-concat calls feed
/// the same call-site evidence to [`tensor_concat_result_type`]. The checked
/// function value carries the generic operation rule; its application
/// contributes the axis expression and any statically visible list elements.
///
/// `Ok(Some(decision))` distinguishes origin equations from ordinary key
/// signatures. Only the origin ledger propagates equality between stored
/// elements or from a published aggregate result. `Ok(None)` means an operand is
/// still undecided -- a variable, or an error witness whose diagnostic is
/// already owned upstream -- and the caller suspends or suppresses. `Err` is
/// the rule's own rejection text.
///
pub(crate) fn decide_collection_constraint(
    constraint: &CollectionConstraint,
    tensor_concat: Option<&TensorConcatCallEvidence>,
    subst: &Subst,
) -> Result<Option<CollectionDecision>, String> {
    let result = constraint.result().clone();
    let joined = |inputs| {
        Some(CollectionDecision::ResultOrigin(ResultConstraint::Join {
            inputs,
            result: result.clone(),
        }))
    };
    let applied = constraint.map_types(|ty| subst.apply(ty));
    if applied
        .operands()
        .iter()
        .any(|ty| matches!(ty, Type::Error(_)))
    {
        return Ok(None);
    }
    match &applied {
        CollectionConstraint::KeyFromSeed {
            operand: Type::Var(_),
            ..
        }
        | CollectionConstraint::SplitKey {
            operand: Type::Var(_),
            ..
        }
        | CollectionConstraint::SplitKeys {
            operand: Type::Var(_),
            ..
        }
        | CollectionConstraint::FoldIn {
            operand: Type::Var(_),
            ..
        } => Ok(None),
        CollectionConstraint::KeyFromSeed { operand, .. } => {
            key_operation_surface(operand, Prim::Int64).map(|ty| {
                Some(CollectionDecision::RequiredEquality {
                    actual: result.clone(),
                    expected: ty,
                })
            })
        }
        CollectionConstraint::SplitKey { operand, .. } => {
            let half = key_operation_surface(operand, Prim::Key)?;
            Ok(Some(CollectionDecision::RequiredEquality {
                actual: result.clone(),
                expected: Type::Tuple(vec![half.clone(), half]),
            }))
        }
        CollectionConstraint::SplitKeys { operand, count, .. } => {
            let key = key_operation_surface(operand, Prim::Key)?;
            let mut compatibility = subst.clone();
            unify(count, &Type::Prim(Prim::Int64), &mut compatibility).map_err(|e| e.message)?;
            let mut dims = match key {
                Type::Tensor(dims, _) => dims,
                _ => vec![],
            };
            dims.push(Dim::Wildcard);
            Ok(Some(CollectionDecision::RequiredEquality {
                actual: Type::Tuple(vec![count.clone(), result.clone()]),
                expected: Type::Tuple(vec![
                    Type::Prim(Prim::Int64),
                    Type::Tensor(dims, TensorPrec::Concrete(Prim::Key)),
                ]),
            }))
        }
        CollectionConstraint::FoldIn { operand, index, .. } => {
            let key = key_operation_surface(operand, Prim::Key)?;
            let expected = match &key {
                Type::Tensor(dims, _) => {
                    Type::Tensor(dims.clone(), TensorPrec::Concrete(Prim::Int64))
                }
                _ => Type::Prim(Prim::Int64),
            };
            let mut compatibility = subst.clone();
            unify(index, &expected, &mut compatibility).map_err(|e| format!("fold_in requires exactly equal shapes and scalar/tensor surfaces ([05-OP-72]): {}", e.message))?;
            // Keep the input shape equality and the result relation in one
            // equation. A private validation must not discard bindings that
            // later applications or local result publication still require.
            Ok(Some(CollectionDecision::RequiredEquality {
                actual: Type::Tuple(vec![index.clone(), result.clone()]),
                expected: Type::Tuple(vec![expected, key]),
            }))
        }
        CollectionConstraint::Len { operand, .. } => match operand {
            Type::Var(_) => Ok(None),
            Type::Adt(name, _) if name == "List" || name == "Dict" => {
                Ok(joined(vec![Type::Prim(Prim::Int64)]))
            }
            other => Err(format!("len expects List or Dict input, got {other}")),
        },
        CollectionConstraint::Index { list, index, .. } => {
            match index {
                Type::Var(_) => return Ok(None),
                Type::Prim(Prim::Int64) => {}
                other => return Err(format!("index expects i64 index, got {other}")),
            }
            match list {
                Type::Var(_) => Ok(None),
                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                    Ok(joined(vec![args[0].clone()]))
                }
                other => Err(format!("index expects List input, got {other}")),
            }
        }
        CollectionConstraint::Append { list, value, .. } => builtins::AggregateRule::Append
            .decide(&[list.clone(), value.clone()], &result, subst)
            .map(|decision| decision.map(CollectionDecision::ResultOrigin))
            .map_err(|error| error.message),
        CollectionConstraint::Concat { lhs, rhs, .. } => match (lhs, rhs) {
            (Type::Var(_), _) | (_, Type::Var(_)) => Ok(None),
            (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                if lhs_name == "List"
                    && rhs_name == "List"
                    && lhs_args.len() == 1
                    && rhs_args.len() == 1 =>
            {
                builtins::AggregateRule::Concat
                    .decide(&[lhs.clone(), rhs.clone()], &result, subst)
                    .map(|decision| decision.map(CollectionDecision::ResultOrigin))
                    .map_err(|error| error.message)
            }
            (Type::Adt(lhs_name, lhs_args), Type::Prim(Prim::Int32))
                if lhs_name == "List" && lhs_args.len() == 1 =>
            {
                let (raw_axis, list_info) = tensor_concat
                    .map(|evidence| (evidence.raw_axis, evidence.list_info.clone()))
                    .unwrap_or((None, ConcatListInfo::BindingLen(None)));
                tensor_concat_result_type(&lhs_args[0], raw_axis, list_info, subst).map(
                    |produced| {
                        Some(CollectionDecision::ProducedResult {
                            declared: result.clone(),
                            produced,
                        })
                    },
                )
            }
            (lhs, rhs) => Err(format!(
                "concat expects matching List inputs, got {lhs} and {rhs}"
            )),
        },
    }
}

/// Direct calls and transported contracts publish the same decisions. Stored
/// value equalities flow through the origin ledger; a computed scalar/tensor
/// result retains its facts after checking the callable's declaration.
pub(super) fn publish_collection_equation(
    constraint: &CollectionConstraint,
    node: &DeepNode,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type> {
    let produced_result;
    match decide_collection_constraint(constraint, None, subst) {
        Ok(Some(decision)) => {
            produced_result = decision.produced_result();
            if let Err(error) = decision.publish(subst) {
                errors.push(error.into());
            }
        }
        Ok(None) => return None,
        Err(message) => {
            return Some(report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_node_provenance(node, message),
                    vec![],
                ),
            ));
        }
    }
    Some(produced_result.unwrap_or_else(|| constraint.result().clone()))
}

fn key_operation_surface(operand: &Type, input: Prim) -> Result<Type, String> {
    match operand {
        Type::Prim(p) if *p == input => Ok(Type::Prim(Prim::Key)),
        Type::Tensor(dims, TensorPrec::Concrete(p)) if *p == input => {
            Ok(Type::Tensor(dims.clone(), TensorPrec::Concrete(Prim::Key)))
        }
        other => Err(format!(
            "key operation expects {} or a tensor of {}, got {other}",
            input.name(),
            input.name()
        )),
    }
}

/// Static source evidence used by the tensor overload of a consumed checked
/// `concat` value. This is application state, not part of the serialized
/// function contract: aliases and imports carry the generic rule, and each
/// call supplies its own axis and visible element shapes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TensorConcatCallEvidence {
    pub(super) raw_axis: Option<i64>,
    pub(super) list_info: ConcatListInfo,
}

/// How the concat arm can see the list's elements (chelis#594).
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ConcatListInfo {
    /// A literal `Cons` chain at the call site: each element's full dim
    /// vector, in list order (read from the current inference epoch's
    /// already-recorded child types).
    /// Ragged literal extents SUM.
    Direct(Vec<Vec<Dim>>),
    /// A variable (or anything else): only a binding-carried literal
    /// LENGTH survives ([`Env::list_literal_len`]); the per-axis extents
    /// come from the §4.5.2 joined element type, so the sum is
    /// `joined extent x length` and requires uniform extents.
    BindingLen(Option<usize>),
}

/// Read the tensor-concat evidence from one already-inferred application.
///
/// Direct and transported calls share this extraction so neither route can
/// retain the axis while dropping literal element extents, or vice versa.
pub(super) fn tensor_concat_call_evidence(
    kids: &[deep::Expr],
    env: &Env,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
    product: &InferenceProduct,
) -> TensorConcatCallEvidence {
    let raw_axis = kids.get(2).and_then(extract_int_for_dim);
    let list_info = match kids.get(1).and_then(collect_cons_chain_for_shape) {
        Some(elements) => ConcatListInfo::Direct(
            elements
                .iter()
                .map(
                    |elem| match product.current_owner_type(elem, subst, errors) {
                        Some(Type::Tensor(dims, _)) => dims,
                        _ => Vec::new(),
                    },
                )
                .collect(),
        ),
        None => ConcatListInfo::BindingLen(static_list_len(kids.get(1), env)),
    };
    TensorConcatCallEvidence {
        raw_axis,
        list_info,
    }
}

/// Result type of a tensor `concat(list, axis)` (spec/04-type-system.md
/// §4.5.4, chelis#631/#594), computed from the joined element type
/// (§4.5.2), the concat-axis value, and the statically-visible elements:
///
/// - literal axis + DIRECT literal list whose every element carries a
///   literal extent on the concat axis → `Lit(sum of extents)` — ragged
///   lists included (chelis#594). Any non-literal element extent (a
///   name, a variable, a wildcard) makes the sum unknown → `Wildcard`.
///   Every other axis is the joined element type's axis unchanged.
/// - literal axis + BINDING-carried length `n >= 1` + joined element
///   extent `Lit(k)` → `Lit(k * n)` (uniform extents only: the join has
///   already widened ragged literals to `*`, and the head-biased
///   `(concrete, wildcard)` join boundary is inherited on this path —
///   the runtime dim guards keep any violation loud, never mis-sized).
/// - literal axis, extents or count unknown → `Wildcard` on the CONCAT
///   axis. (Pre-chelis#631 the LAST axis was wildcarded unconditionally
///   and the concat axis kept the element's dim — a wrong concrete
///   extent the host-program C lane baked into its tensor-helper
///   signatures, aborting guarded forward binaries at run time.)
/// - literal axis out of bounds after negative-axis normalization → Err.
/// - non-literal (runtime) axis → every axis `Wildcard` at the element
///   rank: the host runtime concatenates along a computed axis, so rank
///   is known (§4.5.1 rank uniformity) but no per-axis extent survives.
pub(super) fn tensor_concat_result_type(
    element_ty: &Type,
    raw_axis: Option<i64>,
    list_info: ConcatListInfo,
    subst: &Subst,
) -> Result<Type, String> {
    let Type::Tensor(dims, precision) = element_ty else {
        return Err(format!(
            "concat expects List[tensor[...]] for tensor concatenation, got {element_ty}"
        ));
    };
    if dims.is_empty() {
        return Err("concat expects tensor inputs with at least one axis".to_string());
    }
    let mut out_dims = dims.clone();
    let Some(raw) = raw_axis else {
        out_dims.fill(Dim::Wildcard);
        return Ok(Type::Tensor(out_dims, precision.clone()));
    };
    let rank = out_dims.len();
    let Some(axis) = normalize_static_axis(rank, raw) else {
        return Err(format!("concat axis {raw} out of bounds for rank {rank}"));
    };
    out_dims[axis] = match list_info {
        ConcatListInfo::Direct(elements) if !elements.is_empty() => elements
            .iter()
            .try_fold(0i64, |total, dims| {
                match dims
                    .get(axis)
                    .and_then(|d| subst.observe_dim(d).literal_extent())
                {
                    Some(k) if dims.len() == rank => total.checked_add(k),
                    _ => None,
                }
            })
            .map(Dim::Lit)
            .unwrap_or(Dim::Wildcard),
        ConcatListInfo::BindingLen(Some(n)) if n >= 1 => {
            match subst.observe_dim(&out_dims[axis]).literal_extent() {
                Some(k) => match k.checked_mul(n as i64) {
                    Some(total) => Dim::Lit(total),
                    None => Dim::Wildcard,
                },
                _ => Dim::Wildcard,
            }
        }
        _ => Dim::Wildcard,
    };
    Ok(Type::Tensor(out_dims, precision.clone()))
}

/// Statically-known element count of a list expression (chelis#631): a
/// literal `Cons`/`Nil` chain counts directly; a variable carries a
/// length only when it was bound to a list literal in an enclosing scope
/// ([`Env::list_literal_len`]). `None` for anything else — a function
/// result, a parameter, a `split` output.
pub(super) fn static_list_len(expr: Option<&deep::Expr>, env: &Env) -> Option<usize> {
    let expr = expr?;
    if let Some(elements) = collect_cons_chain_for_shape(expr) {
        return Some(elements.len());
    }
    // chelis#1107: carrier-preserving read; a `List`-only destructure lost the
    // recorded literal length of a stamped `(var {} xs)` on the typed ingress.
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag == DeepTag::Var {
        let name = kids.first().and_then(|e| symbol_name(e))?;
        return env.list_literal_len(name);
    }
    None
}

/// Record (or clear) the statically-known list-literal length of a
/// binding so a later `concat(name, axis)` can count elements
/// (chelis#631). Add-symmetric like the size-provenance marking beside
/// it: a re-bind to a non-literal RHS must clear any stale entry. A
/// `(var other)` RHS propagates an existing entry transitively.
pub(super) fn note_list_literal_binding(env: &mut Env, name: &str, rhs: &deep::Expr) {
    match static_list_len(Some(rhs), env) {
        Some(len) => env.mark_list_literal_len(name, len),
        None => env.clear_list_literal_len(name),
    }
}

/// Resolve an ADT constructor application and enforce its call shape.
pub(super) fn prepare_constructor_application(
    func_name: &Option<String>,
    env: &Env,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> Result<Option<String>, Type> {
    macro_rules! reject {
        ($($arg:tt)*) => {
            return Err(report($($arg)*))
        };
    }

    let constructor = func_name
        .as_ref()
        .and_then(|fname| constructor_for_shape(fname, CallShape::Positional, env, adt_reg));
    let ctor_lookup_name = constructor.map(|_| func_name.as_ref().unwrap().clone());

    // The call site uses positional `(app)` syntax here (named-field
    // record construction lowers through a different builder, not
    // through `infer_app`). When two ADTs in the dep graph define
    // same-named constructors with different shapes (chelis#148: e.g.
    // Coral.Frame.Column.IntCol is positional, School.Data.Dataset.IntCol
    // is record), prefer the positional variant for this call site so
    // the call dispatches to the matching ADT instead of erroring on
    // the colliding record variant. Only emit the "must use named
    // fields" error when EVERY same-named variant in scope is record-
    // shaped, which is the original single-package case the error was
    // written for.
    // RFC D-CHECK: positional application of an out-of-module opaque
    // constructor is rejected (one violation per call site; the
    // callee `var`'s constructor-reference check is suppressed below
    // so the application does not double-report). Inference continues
    // so the call still yields its true type.
    if let Some((adt_name, _, _)) = constructor {
        let adt_name = adt_name.to_string();
        crate::opacity::check_opaque_use(
            crate::opacity::OpaqueAction::CtorApplication,
            &adt_name,
            adt_reg,
            errors,
        );
    }

    // chelis#317: do not emit the record-shape diagnostic for an applied
    // constructor whose name is out of scope (a type-only import that calls
    // `Alpha(...)`). The shape check resolves through the same fuzzy
    // terminal fallback that mis-binds out-of-scope names, so firing it here
    // would mask the real defect with a confusing "must use named fields"
    // message. Let the head's `infer_var` report `unknown constructor`
    // instead.
    let ctor_call_out_of_scope = func_name
        .as_deref()
        .is_some_and(|fname| constructor_out_of_scope(fname, env));
    if !ctor_call_out_of_scope
        && let Some(ref fname) = ctor_lookup_name
        && let Some((_adt_name, _, variant)) = constructor
        && !variant.fields.is_empty()
        && variant
            .fields
            .iter()
            .all(|(field_name, _)| field_name.is_some())
    {
        reject!(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "{fname} is a record constructor and must use named fields: {fname} {{ ... }}"
                ),
                vec![],
            ),
        );
    }

    Ok(ctor_lookup_name)
}

#[cfg(test)]
mod aggregate_origin_tests {
    use super::*;

    #[test]
    fn every_registered_aggregate_returns_equality_without_operand_bindings() {
        use builtins::{AggregateRule, ValueEquality};
        let mut rules = Vec::new();
        for declaration in builtins::BUILTINS {
            for case in declaration.capability.sibling_cases {
                if let ValueEquality::Aggregate(rule) = builtins::case_value_equality(case.case)
                    && !rules.contains(&rule)
                {
                    rules.push(rule);
                }
            }
        }
        for rule in rules {
            let mut vg = VarGen::default();
            let input = vg.fresh_type();
            let result = vg.fresh_type();
            let unknown = Type::Fn(vec![input.clone()], Box::new(vg.fresh_type()));
            let known = Type::Fn(vec![Type::Prim(Prim::F32)], Box::new(Type::Prim(Prim::F32)));
            let list = |value| Type::Adt("List".to_string(), vec![value]);
            let dict = |value| Type::Adt("Dict".to_string(), vec![Type::Prim(Prim::String), value]);
            let operands = match rule {
                AggregateRule::Append => vec![list(known), unknown],
                AggregateRule::Concat => vec![list(unknown), list(known)],
                AggregateRule::DictInsert => vec![dict(known), Type::Prim(Prim::String), unknown],
                AggregateRule::DictMerge => vec![dict(unknown), dict(known)],
                AggregateRule::Fold => vec![unknown, known],
                AggregateRule::Scan => vec![list(unknown), list(known)],
            };
            let subst = Subst::new();
            let equation = rule.decide(&operands, &result, &subst).unwrap().unwrap();
            assert_eq!(subst.apply(&input), input, "{rule:?} bound an input");
            assert_eq!(subst.apply(&result), result, "{rule:?} bound its result");
            let ResultConstraint::Join { inputs, result } = equation else {
                panic!("{rule:?} did not preserve its result equality");
            };
            // The equation remains required: a conflicting later application
            // must fail, even though the operation could not bind the input.
            let mut applied = subst.clone();
            unify(&input, &Type::Prim(Prim::Bool), &mut applied).unwrap();
            assert!(
                inputs
                    .iter()
                    .any(|ty| unify(&result, ty, &mut applied).is_err()),
                "{rule:?} erased a required equality"
            );
        }
    }
}
