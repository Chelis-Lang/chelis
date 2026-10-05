//! Shared source-to-Beacon scalar upper-bound route. No solver fallback.
use super::*;
use crate::discharge::{DischargeEngine, Goal, IntervalBox, IrHandle};
use crate::{BeaconOracleMode, BeaconShim, WireDagByteStore};
use chelis_compiler_api::compiler;
use chelis_surf::ast::{LiteralSuffix, TensorPrecision, UnaryOp};
use chelis_types::{ScalarValue, dtype_semantics::scalar_from_f64, types::Prim};
use sha2::{Digest, Sha256};
use std::borrow::Cow;

fn literal(expr: &Expr) -> Result<f64, String> {
    let value = match expr {
        Expr::Lit(Literal::TypedFloat(value, LiteralSuffix::F64), _) => *value,
        Expr::Unary(UnaryOp::Neg, value, _) => -literal(value)?,
        _ => return Err("Beacon bounds require explicit f64 literals".into()),
    };
    if !value.is_finite() {
        return Err("Beacon bound must be finite".into());
    }
    Ok(value)
}

fn scalar_box(property: &Property) -> Result<IntervalBox, String> {
    let mut bounds = BTreeMap::new();
    for param in &property.params {
        if !matches!(&param.ty, Some(TypeExpr::Tensor(dims, precision, _)) if dims.is_empty() && precision == "f64")
            || bounds.insert(param.name.clone(), (None, None)).is_some()
        {
            return Err(
                "Beacon bounds require distinct named rank-zero tensor[f64] parameters".into(),
            );
        }
    }
    let mut pending: Vec<_> = property.preconditions.iter().collect();
    while let Some(condition) = pending.pop() {
        if let Expr::Binary(BinOp::And, left, right, _) = condition {
            pending.extend([left.as_ref(), right.as_ref()]);
            continue;
        }
        let Expr::Binary(op @ (BinOp::Le | BinOp::Ge), left, right, _) = condition else {
            return Err("Beacon input bounds must be closed scalar inequalities".into());
        };
        let (name, value, lower) = match (scalar_input_name(left), scalar_input_name(right)) {
            (Some(name), None) => (name, literal(right)?, *op == BinOp::Ge),
            (None, Some(name)) => (name, literal(left)?, *op == BinOp::Le),
            _ => return Err("Beacon input bounds must compare a parameter with a literal".into()),
        };
        let entry = bounds.get_mut(name).ok_or("bound names a non-parameter")?;
        let slot = if lower { &mut entry.0 } else { &mut entry.1 };
        if slot.replace(value).is_some() {
            return Err(format!("duplicate bound for `{name}`"));
        }
    }
    let dims = bounds
        .into_iter()
        .map(|(name, (lo, hi))| {
            Ok((
                name,
                lo.ok_or("missing lower bound")?,
                hi.ok_or("missing upper bound")?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(IntervalBox { dims })
}

fn scalar_input_name(expr: &Expr) -> Option<&str> {
    match tensor_operand(expr)? {
        Cow::Borrowed(Expr::Var(name, _)) => Some(name),
        _ => None,
    }
}

fn names_tensor_to_scalar(expr: &Expr) -> bool {
    matches!(expr, Expr::Var(name, _) if name == "tensor_to_scalar")
}

/// The tensor operand of a final `tensor_to_scalar` call, in either spelling
/// the formatter and linter accept: the call `tensor_to_scalar(e)`, or a pipe
/// whose last stage is the bare name `tensor_to_scalar`. Both spellings
/// denote the same application, so matching only the call form would let a
/// change of spelling disconnect a property from Beacon.
fn tensor_operand(expr: &Expr) -> Option<Cow<'_, Expr>> {
    match expr {
        Expr::Apply(function, operands, _)
            if names_tensor_to_scalar(function) && operands.len() == 1 =>
        {
            Some(Cow::Borrowed(&operands[0]))
        }
        Expr::Pipe(seed, stages, span) => {
            let (last, earlier) = stages.split_last()?;
            if !names_tensor_to_scalar(last) {
                return None;
            }
            Some(if earlier.is_empty() {
                Cow::Borrowed(seed.as_ref())
            } else {
                Cow::Owned(Expr::Pipe(seed.clone(), earlier.to_vec(), *span))
            })
        }
        _ => None,
    }
}

fn upper_expression(property: &Property) -> Result<(Cow<'_, Expr>, ScalarValue), String> {
    let Expr::Binary(op @ (BinOp::Le | BinOp::Ge), left, right, _) = &property.body else {
        return Err("Beacon property body must be a non-strict scalar upper bound".into());
    };
    let (expression, threshold) = if *op == BinOp::Le {
        (left.as_ref(), right.as_ref())
    } else {
        (right.as_ref(), left.as_ref())
    };
    let threshold = scalar_from_f64("Beacon threshold transport", Prim::F64, literal(threshold)?)
        .map_err(|error| format!("invalid threshold: {error}"))?;
    Ok((
        tensor_operand(expression)
            .ok_or("Beacon output must use tensor_to_scalar on its scalar graph output")?,
        threshold,
    ))
}

pub(super) fn prove(
    decls: &[Decl],
    property: &Property,
    options: &PropertyRunOptions,
) -> PropertyOutcome {
    let seed = options.effective_seed(property.seed);
    let fail = |reason: String| {
        PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Unsupported,
            PropertyTier::Beacon,
            0,
            seed,
            None,
            Some(reason),
            false,
            Vec::new(),
        )
    };
    let prepared = (|| -> Result<_, String> {
        if !property.contracts.is_empty() {
            return Err("Beacon scalar lane does not consume contract assumptions".into());
        }
        if decls.iter().any(|decl| matches!(decl, Decl::Import { .. })) {
            return Err("Beacon scalar lane requires self-contained source without imports".into());
        }
        if property.params.iter().any(|param| param.name == "tensor_to_scalar") ||
            decls.iter().any(|decl| matches!(decl, Decl::FunDef { name, .. } | Decl::LetDef { name, .. } | Decl::MacroDef { name, .. } | Decl::Sig { name, .. } if name == "tensor_to_scalar")) {
            return Err("Beacon scalar bridge must not be shadowed".into());
        }
        let mut inputs = scalar_box(property)?;
        let (expression, upper) = upper_expression(property)?;
        Goal::scalar_upper_bound(inputs.clone(), upper).map_err(|e| e.to_string())?;
        let body = chelis_surf::format::format_expression(&expression);
        // The goal graph is lowered from the declarations the property was
        // checked against plus generated declarations, never from a printed
        // and re-parsed copy of them (chelis#3172). The generated names avoid
        // every name the declarations or the expression mention; their debug
        // rendering is a superset of those names.
        let mentioned = format!("{decls:?}{expression:?}");
        let fresh = |base: String| {
            let mut name = base;
            while mentioned.contains(&name) {
                name.push('_');
            }
            name
        };
        // Generated declarations carry the property's first parameter span,
        // so a diagnostic about them points into the property.
        let span = property
            .params
            .first()
            .map_or_else(|| chelis_deep::Span::new(0, 0), |param| param.span);
        let tensor_f64 = || TypeExpr::Tensor(Vec::new(), TensorPrecision::new("f64", span), span);
        let entry = fresh("beacon_goal_output".to_string());
        let function = fresh(format!("{entry}_function"));
        let mut program = decls.to_vec();
        program.push(Decl::FunDef {
            name: function.clone(),
            type_binders: Vec::new(),
            params: property
                .params
                .iter()
                .map(|param| Param {
                    name: param.name.clone(),
                    ty: Some(tensor_f64()),
                    span,
                })
                .collect(),
            ret_ty: Some(tensor_f64()),
            effects: None,
            body: expression.clone().into_owned(),
            span,
        });
        let mut arguments = Vec::new();
        let mut input_bindings = BTreeMap::new();
        for (index, param) in property.params.iter().enumerate() {
            let name = fresh(format!("beacon_input_{index}"));
            program.push(Decl::LetDef {
                name: name.clone(),
                ty: None,
                value: Expr::Annotate(Box::new(Expr::Var(name.clone(), span)), tensor_f64(), span),
                span,
            });
            let dimension = inputs
                .dims
                .iter_mut()
                .find(|(original, _, _)| original == &param.name)
                .ok_or("missing input binding")?;
            dimension.0.clone_from(&name);
            input_bindings.insert(name.clone(), param.name.clone());
            arguments.push(Expr::Var(name, span));
        }
        program.push(Decl::LetDef {
            name: entry.clone(),
            ty: None,
            value: Expr::Apply(Box::new(Expr::Var(function, span)), arguments, span),
            span,
        });
        let goal = Goal::scalar_upper_bound(inputs, upper).map_err(|e| e.to_string())?;
        let lowered = compiler::lower_decls(&program, Some(&entry))
            .map_err(|error| format!("Beacon graph lowering failed: {error:?}"))?;
        lowered
            .dag
            .validate_wire_contract()
            .map_err(|e| e.to_string())?;
        let mut root = *lowered
            .named_roots
            .get(&entry)
            .ok_or("lowered entry has no named root")?;
        // A direct identity result is anchored by the compiler's terminal Store.
        // Its value is the single operand; only peel our own generated output
        // anchor, never an internal store or an unrelated named output.
        if let Some(node) = lowered.dag.nodes.get(root as usize)
            && matches!(&node.op,chelis_compiler_api::schema::WireRiscOp::Store { name } if name == &entry)
            && node.inputs.len() == 1
            && node.output_type.dims.is_empty()
            && node.output_type.precision == "f64"
            && lowered
                .dag
                .nodes
                .get(node.inputs[0] as usize)
                .is_some_and(|input| {
                    input.output_type.dims.is_empty() && input.output_type.precision == "f64"
                })
        {
            root = node.inputs[0];
        }
        let (dag, root) = crate::graph_extract::scalar_root_closure(&lowered.dag, root)?;
        let bytes = serde_json::to_vec(&dag).map_err(|e| e.to_string())?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let store = WireDagByteStore::new();
        store.insert(hash.clone(), bytes);
        let binary =
            std::env::var_os(crate::BEACON_BIN_ENV).ok_or("CHELIS_BEACON_BIN is not configured")?;
        let shim = BeaconShim::new(std::path::PathBuf::from(binary), store)
            .with_oracle_mode(BeaconOracleMode::ReluLinear);
        let budget = options
            .beacon_deadline
            .map(|deadline| {
                options.beacon_budget.min(
                    deadline
                        .saturating_duration_since(std::time::Instant::now())
                        .saturating_sub(std::time::Duration::from_secs(5)),
                )
            })
            .unwrap_or(options.beacon_budget);
        let timeout = u64::try_from(budget.as_millis())
            .ok()
            .and_then(|value| value.checked_add(5000))
            .ok_or("Beacon budget exceeds supported duration")?;
        let result = shim.discharge(&goal.with_ir(IrHandle::from_wire_dag(hash, root)), timeout);
        Ok((
            result,
            format!("({body}) - ({:?}f64) <= 0.0f64", upper.as_f64_lossy()),
            input_bindings,
        ))
    })();
    let (discharge, folded_goal, input_bindings) = match prepared {
        Ok(value) => value,
        Err(reason) => return fail(reason),
    };
    use crate::tier_b::TierBResult;
    let (status, witness, reason) = match discharge.result() {
        TierBResult::Proved => (PropertyStatus::Passed, None, None),
        TierBResult::Disproved(witness) => (PropertyStatus::Failed, Some(witness.clone()), None),
        TierBResult::Unknown | TierBResult::Timeout => (
            PropertyStatus::Unsupported,
            None,
            Some(
                discharge.evidence()["semantic_reason"]
                    .as_str()
                    .or_else(|| discharge.evidence()["beacon_evidence"]["reason"].as_str())
                    .unwrap_or("Beacon returned unknown")
                    .into(),
            ),
        ),
        TierBResult::Error(reason) => (PropertyStatus::Error, None, Some(reason.clone())),
    };
    let mut outcome = PropertyOutcome::with_base_discharge(
        property.name.clone(),
        status,
        PropertyTier::Beacon,
        0,
        seed,
        witness,
        reason,
        false,
        Vec::new(),
        Some((discharge.soundness(), discharge.qualifier_set().clone())),
    );
    let mut evidence = discharge.evidence().clone();
    evidence["folded_goal"] = serde_json::json!(folded_goal);
    evidence["input_bindings"] = serde_json::json!(input_bindings);
    outcome.engine_evidence = Some(evidence);
    outcome
}
