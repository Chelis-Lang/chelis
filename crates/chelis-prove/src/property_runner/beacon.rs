//! Shared source-to-Beacon scalar upper-bound route. No solver fallback.
use super::*;
use crate::discharge::{DischargeEngine, Goal, IntervalBox, IrHandle};
use crate::{BeaconOracleMode, BeaconShim, WireDagByteStore};
use chelis_compiler_api::compiler;
use chelis_surf::ast::{LiteralSuffix, TensorPrecision, UnaryOp};
use chelis_types::{ScalarValue, dtype_semantics::scalar_from_f64, types::Prim};
use sha2::{Digest, Sha256};
use std::borrow::Cow;

fn bind_pattern<'a>(pattern: &'a chelis_surf::ast::Pattern, locals: &mut BTreeSet<&'a str>) {
    use chelis_surf::ast::Pattern;
    match pattern {
        Pattern::Var(name, _) => {
            locals.insert(name);
        }
        Pattern::As(name, inner, _) => {
            locals.insert(name);
            bind_pattern(inner, locals);
        }
        Pattern::Constructor(_, patterns, _) | Pattern::Tuple(patterns, _) => {
            for pattern in patterns {
                bind_pattern(pattern, locals);
            }
        }
        Pattern::Record(_, fields, _) => {
            for (_, pattern) in fields {
                bind_pattern(pattern, locals);
            }
        }
        Pattern::Wildcard(_) | Pattern::Lit(_, _) => {}
    }
}

fn bind_let_pattern<'a>(pattern: &'a chelis_surf::ast::LetPattern, locals: &mut BTreeSet<&'a str>) {
    use chelis_surf::ast::LetPattern;
    match pattern {
        LetPattern::Var(name, _) => {
            locals.insert(name);
        }
        LetPattern::Tuple(patterns, _) => {
            for pattern in patterns {
                bind_let_pattern(pattern, locals);
            }
        }
        LetPattern::Wildcard(_) => {}
    }
}

/// Enumerate value references from every Surf expression form, respecting
/// lexical binders. The linker has already resolved package references to
/// their declaration identities; no private-name suffix matching occurs.
fn value_refs(
    expr: &Expr,
    locals: &BTreeSet<&str>,
    names: &BTreeSet<String>,
    out: &mut BTreeSet<String>,
) {
    match expr {
        Expr::Var(name, _) => {
            if !locals.contains(name.as_str()) && names.contains(name) {
                out.insert(name.clone());
            }
        }
        Expr::Apply(callee, args, _) => {
            value_refs(callee, locals, names, out);
            for arg in args {
                value_refs(arg, locals, names, out);
            }
        }
        Expr::Accumulate(call, _, _) => value_refs(call, locals, names, out),
        Expr::List(items, _) | Expr::Tuple(items, _) | Expr::Par(items, _) | Expr::Do(items, _) => {
            for item in items {
                value_refs(item, locals, names, out);
            }
        }
        Expr::Record(_, fields, _) => {
            for (_, value) in fields {
                value_refs(value, locals, names, out);
            }
        }
        Expr::RecordUpdate(base, fields, _) => {
            value_refs(base, locals, names, out);
            for (_, value) in fields {
                value_refs(value, locals, names, out);
            }
        }
        Expr::Access(inner, _, _)
        | Expr::TupleGet(inner, _, _)
        | Expr::Unary(_, inner, _)
        | Expr::Cast(inner, _, _, _)
        | Expr::Grad(inner, _, _)
        | Expr::Vmap(inner, _, _)
        | Expr::Jit(inner, _)
        | Expr::Realize(inner, _)
        | Expr::Copy(inner, _)
        | Expr::Borrow(inner, _)
        | Expr::Quote(inner, _)
        | Expr::Unquote(inner, _)
        | Expr::Splice(inner, _)
        | Expr::Annotate(inner, _, _) => value_refs(inner, locals, names, out),
        Expr::Binary(_, left, right, _) | Expr::WithDevice(left, right, _) => {
            value_refs(left, locals, names, out);
            value_refs(right, locals, names, out);
        }
        Expr::Pipe(seed, stages, _) => {
            value_refs(seed, locals, names, out);
            for stage in stages {
                value_refs(&stage.expression, locals, names, out);
            }
        }
        Expr::If(condition, yes, no, _) => {
            value_refs(condition, locals, names, out);
            value_refs(yes, locals, names, out);
            value_refs(no, locals, names, out);
        }
        Expr::Match(value, arms, _) => {
            value_refs(value, locals, names, out);
            for arm in arms {
                let mut arm_locals = locals.clone();
                bind_pattern(&arm.pattern, &mut arm_locals);
                if let Some(guard) = &arm.guard {
                    value_refs(guard, &arm_locals, names, out);
                }
                value_refs(&arm.body, &arm_locals, names, out);
            }
        }
        Expr::Lambda(params, body, _) => {
            let mut body_locals = locals.clone();
            body_locals.extend(params.iter().map(|param| param.name.as_str()));
            value_refs(body, &body_locals, names, out);
        }
        Expr::Block(bindings, body, _) => {
            let mut block_locals = locals.clone();
            for binding in bindings {
                value_refs(&binding.value, &block_locals, names, out);
                bind_let_pattern(&binding.pattern, &mut block_locals);
            }
            value_refs(body, &block_locals, names, out);
        }
        Expr::Lit(_, _) | Expr::Constructor(_, _) => {}
    }
}

/// The checked declarations and exact property expression that the compiler
/// used to produce a scalar graph. The graph lowerer inlines calls, so its
/// declaration table alone cannot identify the imported source function.
/// This provenance is report data; the Beacon certificate still addresses
/// the content-hashed graph and selected root.
fn source_binding(
    decls: &[Decl],
    property: &Property,
    expression: &Expr,
) -> Result<serde_json::Value, String> {
    let mut bindings = BTreeMap::new();
    for decl in decls {
        let name = match decl {
            Decl::FunDef { name, .. } | Decl::LetDef { name, .. } => name,
            _ => continue,
        };
        if bindings.insert(name.as_str(), decl).is_some() {
            return Err(format!(
                "Beacon source has more than one value declaration named `{name}`"
            ));
        }
    }
    let names = bindings.keys().map(|name| (*name).to_string()).collect();
    let property_params = property
        .params
        .iter()
        .map(|param| param.name.as_str())
        .collect();
    let mut pending = BTreeSet::new();
    value_refs(expression, &property_params, &names, &mut pending);
    let mut selected = BTreeSet::new();
    while let Some(name) = pending.pop_first() {
        if !selected.insert(name.clone()) {
            continue;
        }
        let decl = bindings
            .get(name.as_str())
            .ok_or("Beacon source reference has no checked declaration")?;
        let (body, params) = match decl {
            Decl::FunDef { body, params, .. } => (
                body,
                params.iter().map(|param| param.name.as_str()).collect(),
            ),
            Decl::LetDef { value, .. } => (value, BTreeSet::new()),
            _ => unreachable!("binding map contains values only"),
        };
        value_refs(body, &params, &names, &mut pending);
    }
    let mut source_declarations = Vec::new();
    for decl in decls {
        let name = match decl {
            Decl::FunDef { name, .. } | Decl::LetDef { name, .. } if selected.contains(name) => {
                name
            }
            _ => continue,
        };
        let bytes = serde_json::to_vec(decl).map_err(|error| error.to_string())?;
        source_declarations.push(
            serde_json::json!({"name":name,"sha256":format!("{:x}", Sha256::digest(&bytes))}),
        );
    }
    let expression_bytes = serde_json::to_vec(expression).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "expression_sha256":format!("{:x}", Sha256::digest(&expression_bytes)),
        "declarations":source_declarations,
    }))
}

fn literal(expr: &Expr, dtype: Prim) -> Result<f64, String> {
    let value = match expr {
        Expr::Lit(Literal::TypedFloat(value, suffix), _)
            if matches!(
                (dtype, suffix),
                (Prim::F32, LiteralSuffix::F32) | (Prim::F64, LiteralSuffix::F64)
            ) =>
        {
            *value
        }
        Expr::Unary(UnaryOp::Neg, value, _) => -literal(value, dtype)?,
        _ => {
            return Err(format!(
                "Beacon bounds require explicit {} literals",
                dtype.name()
            ));
        }
    };
    if !value.is_finite() {
        return Err("Beacon bound must be finite".into());
    }
    let stored =
        scalar_from_f64("Beacon source bound", dtype, value).map_err(|error| error.to_string())?;
    if !stored.as_f64_lossy().is_finite() {
        return Err("Beacon bound must be finite at its source dtype".into());
    }
    Ok(stored.as_f64_lossy())
}

fn scalar_dtype(property: &Property, scalar: bool) -> Result<Prim, String> {
    if !scalar {
        return Ok(Prim::F64);
    }
    let Some(first) = property.params.first() else {
        return Err("Beacon scalar goal requires a named input".into());
    };
    match &first.ty {
        Some(TypeExpr::Named(name, _)) if name == "f32" => Ok(Prim::F32),
        Some(TypeExpr::Named(name, _)) if name == "f64" => Ok(Prim::F64),
        _ => Err("Beacon scalar goal requires f32 or f64 inputs".into()),
    }
}

fn scalar_box(property: &Property, scalar: bool, dtype: Prim) -> Result<IntervalBox, String> {
    let mut bounds = BTreeMap::new();
    for param in &property.params {
        let admitted = if scalar {
            matches!(&param.ty, Some(TypeExpr::Named(name, _)) if name == dtype.name())
        } else {
            matches!(&param.ty, Some(TypeExpr::Tensor(dims, precision, _)) if dims.is_empty() && precision == "f64")
        };
        if !admitted || bounds.insert(param.name.clone(), (None, None)).is_some() {
            return Err("Beacon bounds require distinct same-dtype f32/f64 scalar or rank-zero tensor[f64] parameters".into());
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
        let (name, value, lower) = match (
            scalar_input_name(left, scalar),
            scalar_input_name(right, scalar),
        ) {
            (Some(name), None) => (name, literal(right, dtype)?, *op == BinOp::Ge),
            (None, Some(name)) => (name, literal(left, dtype)?, *op == BinOp::Le),
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
                scalar_from_f64(
                    "Beacon lower bound",
                    Prim::F64,
                    lo.ok_or("missing lower bound")?,
                )
                .map_err(|error| error.to_string())?,
                scalar_from_f64(
                    "Beacon upper bound",
                    Prim::F64,
                    hi.ok_or("missing upper bound")?,
                )
                .map_err(|error| error.to_string())?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(IntervalBox { dims })
}

fn scalar_input_name(expr: &Expr, scalar: bool) -> Option<&str> {
    if scalar {
        return match expr {
            Expr::Var(name, _) => Some(name),
            _ => None,
        };
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BoundSide {
    Lower,
    Upper,
}

impl BoundSide {
    fn name(self) -> &'static str {
        match self {
            Self::Lower => "lower",
            Self::Upper => "upper",
        }
    }
}

fn bound_expressions(
    property: &Property,
    scalar: bool,
    dtype: Prim,
) -> Result<(Expr, Vec<(BoundSide, ScalarValue)>), String> {
    let mut pending = vec![&property.body];
    let mut expression: Option<Expr> = None;
    let mut text: Option<String> = None;
    let mut bounds = std::collections::BTreeMap::new();
    while let Some(body) = pending.pop() {
        if let Expr::Binary(BinOp::And, left, right, _) = body {
            pending.extend([left.as_ref(), right.as_ref()]);
            continue;
        }
        let Expr::Binary(op @ (BinOp::Le | BinOp::Ge), left, right, _) = body else {
            return Err(
                "Beacon property body must contain one or two non-strict scalar bounds".into(),
            );
        };
        let (side, raw_expression, threshold) = match (literal(left, dtype), literal(right, dtype))
        {
            (Err(_), Ok(threshold)) => (
                if *op == BinOp::Le {
                    BoundSide::Upper
                } else {
                    BoundSide::Lower
                },
                left.as_ref(),
                threshold,
            ),
            (Ok(threshold), Err(_)) => (
                if *op == BinOp::Le {
                    BoundSide::Lower
                } else {
                    BoundSide::Upper
                },
                right.as_ref(),
                threshold,
            ),
            _ => {
                return Err(format!(
                    "Beacon output bound must compare one expression with an explicit {} literal",
                    dtype.name()
                ));
            }
        };
        let candidate = if scalar {
            raw_expression.clone()
        } else {
            tensor_operand(raw_expression)
                .ok_or("Beacon output must use tensor_to_scalar on its scalar graph output")?
                .into_owned()
        };
        let candidate_text = chelis_surf::format::format_expression(&candidate);
        if text
            .as_ref()
            .is_some_and(|previous| previous != &candidate_text)
        {
            return Err("two-sided Beacon bounds must name the same output expression".into());
        }
        text = Some(candidate_text);
        expression = Some(candidate);
        let bound = scalar_from_f64("Beacon threshold transport", Prim::F64, threshold)
            .map_err(|error| format!("invalid threshold: {error}"))?;
        if bounds.insert(side, bound).is_some() {
            return Err(format!("duplicate Beacon {} output bound", side.name()));
        }
    }
    if bounds.is_empty() {
        return Err("Beacon property body has no scalar bound".into());
    }
    Ok((
        expression.expect("nonempty bounds have an expression"),
        bounds.into_iter().collect(),
    ))
}

/// A scalar proof graph bypasses the ordinary runtime lowerer, so it must
/// perform the compiler's whole-program check before extracting any graph.
/// Check the linked declarations themselves: printing and reparsing them
/// would lose the declaration identities the proof is about.
pub(super) fn check_scalar_source(decls: &[Decl]) -> Result<(), String> {
    let deep = chelis_surf::desugar::desugar_program(decls)
        .map_err(|error| format!("Beacon scalar source desugar failed: {error}"))?;
    let _linked = chelis_types::install_linked_program_guard();
    chelis_types::check_typed_program(&deep).map_err(|infer| {
        let diagnostics = infer
            .errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        format!(
            "Beacon scalar source failed type checking: {}",
            if diagnostics.is_empty() {
                "compiler returned no diagnostics"
            } else {
                &diagnostics
            }
        )
    })?;
    Ok(())
}

pub(super) fn is_scalar_property(property: &Property) -> bool {
    property
        .params
        .iter()
        .all(|param| matches!(&param.ty, Some(TypeExpr::Named(_, _))))
}

pub(super) fn prove(
    decls: &[Decl],
    property: &Property,
    options: &PropertyRunOptions,
    scalar_check: &Result<(), String>,
) -> PropertyOutcome {
    let seed = options.effective_seed(property.seed);
    let scalar = is_scalar_property(property);
    if scalar && let Err(reason) = scalar_check {
        return PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Error,
            PropertyTier::Beacon,
            0,
            seed,
            None,
            Some(reason.clone()),
            false,
            Vec::new(),
        );
    }
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
        if !scalar && (property.params.iter().any(|param| param.name == "tensor_to_scalar") ||
            decls.iter().any(|decl| matches!(decl, Decl::FunDef { name, .. } | Decl::LetDef { name, .. } | Decl::MacroDef { name, .. } | Decl::Sig { name, .. } if name == "tensor_to_scalar"))) {
            return Err("Beacon scalar bridge must not be shadowed".into());
        }
        let dtype = scalar_dtype(property, scalar)?;
        let mut inputs = scalar_box(property, scalar, dtype)?;
        let (expression, bounds) = bound_expressions(property, scalar, dtype)?;
        let source_binding = source_binding(decls, property, &expression)?;
        let body = chelis_surf::format::format_expression(&expression);
        let (dag, root, input_bindings) = if scalar {
            let (dag, root) = super::beacon_scalar::lower(decls, property, &expression)?;
            let input_bindings = property
                .params
                .iter()
                .map(|param| (param.name.clone(), param.name.clone()))
                .collect();
            (dag, root, input_bindings)
        } else {
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
            let tensor_f64 =
                || TypeExpr::Tensor(Vec::new(), TensorPrecision::new("f64", span), span);
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
                body: expression.clone(),
                span,
            });
            let mut arguments = Vec::new();
            let mut input_bindings = BTreeMap::new();
            for (index, param) in property.params.iter().enumerate() {
                let name = fresh(format!("beacon_input_{index}"));
                program.push(Decl::LetDef {
                    name: name.clone(),
                    ty: None,
                    value: Expr::Annotate(
                        Box::new(Expr::Var(name.clone(), span)),
                        tensor_f64(),
                        span,
                    ),
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
            (dag, root, input_bindings)
        };
        if dag.roots.first() != Some(&root) {
            return Err("Beacon proof graph lost its original output root".into());
        }
        let store = WireDagByteStore::new();
        let binary =
            std::env::var_os(crate::BEACON_BIN_ENV).ok_or("CHELIS_BEACON_BIN is not configured")?;
        let shim = BeaconShim::new(std::path::PathBuf::from(binary), store.clone())
            .with_oracle_mode(BeaconOracleMode::ReluLinear);
        let mut results = Vec::new();
        for (side, bound) in bounds {
            let mut side_dag = dag.clone();
            let (selected_root, upper, folded_goal) = match side {
                BoundSide::Upper => (
                    root,
                    bound,
                    format!("({body}) - ({:?}f64) <= 0.0f64", bound.as_f64_lossy()),
                ),
                BoundSide::Lower => {
                    let original = side_dag
                        .nodes
                        .get(root as usize)
                        .ok_or("Beacon original output root is outside the graph")?
                        .clone();
                    let negated_root = side_dag.nodes.len() as u64;
                    side_dag
                        .nodes
                        .push(chelis_compiler_api::schema::WireDagNode {
                            id: negated_root,
                            op: chelis_compiler_api::schema::WireRiscOp::Neg,
                            inputs: vec![root],
                            output_type: original.output_type,
                            shape_deps: Vec::new(),
                            span_id: original.span_id,
                            merged_spans: original.merged_spans,
                            declaration: original.declaration,
                            activation: original.activation,
                        });
                    side_dag.roots.push(negated_root);
                    let upper = scalar_from_f64(
                        "Beacon negated lower bound",
                        Prim::F64,
                        -bound.as_f64_lossy(),
                    )
                    .map_err(|error| error.to_string())?;
                    (
                        negated_root,
                        upper,
                        format!("({:?}f64) - ({body}) <= 0.0f64", bound.as_f64_lossy()),
                    )
                }
            };
            side_dag
                .validate_wire_contract()
                .map_err(|error| error.to_string())?;
            let goal = Goal::scalar_upper_bound(inputs.clone(), upper)
                .map_err(|error| error.to_string())?;
            let bytes = serde_json::to_vec(&side_dag).map_err(|error| error.to_string())?;
            let hash = format!("{:x}", Sha256::digest(&bytes));
            store.insert(hash.clone(), bytes);
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
            let discharge = shim.discharge(
                &goal.with_ir(IrHandle::from_wire_dag(hash, selected_root)),
                timeout,
            );
            results.push((side, bound, discharge, folded_goal));
        }
        Ok((results, input_bindings, source_binding, dtype))
    })();
    let (results, input_bindings, source_binding, dtype) = match prepared {
        Ok(value) => value,
        Err(reason) => return fail(reason),
    };
    use crate::tier_b::TierBResult;
    let mut status = PropertyStatus::Passed;
    let mut witness = None;
    let mut reason = None;
    let mut soundness = crate::discharge::Soundness::Exact;
    let mut qualifiers = crate::discharge::QualifierSet::new();
    for (side, _, discharge, _) in &results {
        match discharge.result() {
            TierBResult::Proved => {
                soundness = soundness.min(discharge.soundness());
                qualifiers = qualifiers.union(discharge.qualifier_set());
            }
            TierBResult::Disproved(model) => {
                status = PropertyStatus::Failed;
                witness = Some(model.clone());
                reason = Some(format!("{} Beacon bound was refuted", side.name()));
            }
            TierBResult::Error(error) if status != PropertyStatus::Failed => {
                status = PropertyStatus::Error;
                reason = Some(format!("{} Beacon bound: {error}", side.name()));
            }
            TierBResult::Unknown | TierBResult::Timeout if status == PropertyStatus::Passed => {
                status = PropertyStatus::Unsupported;
                let detail = discharge.evidence()["semantic_reason"]
                    .as_str()
                    .or_else(|| discharge.evidence()["beacon_evidence"]["reason"].as_str())
                    .unwrap_or("Beacon returned unknown");
                reason = Some(format!("{} Beacon bound: {detail}", side.name()));
            }
            _ => {}
        }
    }
    let base_discharge = (status == PropertyStatus::Passed).then_some((soundness, qualifiers));
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
        base_discharge,
    );
    let mut evidence = if results.len() == 1 {
        let (side, bound, discharge, folded_goal) = &results[0];
        let mut evidence = discharge.evidence().clone();
        evidence["folded_goal"] = serde_json::json!(folded_goal);
        evidence["bound_side"] = serde_json::json!(side.name());
        evidence["source_bound"] = serde_json::json!(bound);
        evidence
    } else {
        let mut sides = serde_json::Map::new();
        for (side, bound, discharge, folded_goal) in &results {
            let result = match discharge.result() {
                TierBResult::Proved => "proved",
                TierBResult::Disproved(_) => "disproved",
                TierBResult::Unknown => "unknown",
                TierBResult::Timeout => "timeout",
                TierBResult::Error(_) => "error",
            };
            sides.insert(
                side.name().into(),
                serde_json::json!({
                    "source_bound": bound,
                    "folded_goal": folded_goal,
                    "engine_evidence": discharge.evidence(),
                    "result": result,
                }),
            );
        }
        serde_json::json!({"bounds": sides})
    };
    evidence["input_bindings"] = serde_json::json!(input_bindings);
    evidence["source_binding"] = source_binding;
    evidence["source_dtype"] = serde_json::json!(dtype.name());
    evidence["proof_graph_dtype"] = serde_json::json!("f64");
    outcome.engine_evidence = Some(evidence);
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_surf::ast::{LetBinding, LetPattern};

    #[test]
    fn source_binding_follows_nested_calls_without_claiming_a_shadowed_definition() {
        let span = chelis_deep::Span::new(0, 0);
        let name = |name: &str| Expr::Var(name.into(), span);
        let expr = Expr::Block(
            vec![LetBinding {
                pattern: LetPattern::Var("neuron".into(), span),
                ty: None,
                value: name("local_source"),
            }],
            Box::new(Expr::Apply(
                Box::new(name("neuron")),
                vec![Expr::If(
                    Box::new(Expr::Lit(Literal::Bool(true), span)),
                    Box::new(Expr::Apply(Box::new(name("model")), vec![name("x")], span)),
                    Box::new(name("x")),
                    span,
                )],
                span,
            )),
            span,
        );
        let names = ["neuron", "local_source", "model"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let locals = BTreeSet::from(["x"]);
        let mut refs = BTreeSet::new();
        value_refs(&expr, &locals, &names, &mut refs);
        assert_eq!(
            refs,
            BTreeSet::from(["local_source".into(), "model".into()])
        );
    }
}
