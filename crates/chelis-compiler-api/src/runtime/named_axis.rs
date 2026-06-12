use std::collections::{HashMap, HashSet};

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_ir::dag::{Dag, DimInfo, NodeId, TensorType};
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_ir::lower::{try_lower_subexpr_program, type_expr_has_rank_var};
use chelis_types::types::Prim;

use super::transforms::*;
use super::*;

impl<'a> EvalContext<'a> {
    /// chelis#338: does evaluating a call to `resolved_name` require
    /// routing through IR lowering because a *named-axis* reduction is
    /// involved? True when the def's body contains a reduction whose
    /// axis argument is a bare `(var name)`, or when it references a
    /// rank-polymorphic def that does. References through defs with
    /// concrete signatures do NOT propagate the flag: the interpreter
    /// descends and routes at the deeper call boundary instead, which
    /// keeps host constructs (print, lists, ...) in the host lane,
    /// mirroring how `chelis build` structures the host/tensor split.
    pub(super) fn def_requires_named_axis_routing(&mut self, resolved_name: &str) -> bool {
        if let Some(&cached) = self.named_axis_route_cache.get(resolved_name) {
            return cached;
        }
        if !self
            .named_axis_route_visiting
            .insert(resolved_name.to_string())
        {
            return false;
        }
        let result = self.def_requires_named_axis_routing_uncached(resolved_name);
        self.named_axis_route_visiting.remove(resolved_name);
        self.named_axis_route_cache
            .insert(resolved_name.to_string(), result);
        result
    }

    fn def_requires_named_axis_routing_uncached(&mut self, resolved_name: &str) -> bool {
        let Some(body) = self.top_level_defs.get(resolved_name).cloned() else {
            return false;
        };
        let mut hit = false;
        let mut vars: Vec<String> = Vec::new();
        scan_expr_for_named_axis_reduction(&body, &mut hit, &mut vars);
        if hit {
            return true;
        }
        for name in vars {
            let Some((referenced, _)) = self.lookup_top_level_def(&name) else {
                continue;
            };
            if referenced == resolved_name {
                continue;
            }
            let rank_poly_sig = self
                .type_env
                .get(&referenced)
                .map(type_expr_has_rank_var)
                .unwrap_or(false);
            if rank_poly_sig && self.def_requires_named_axis_routing(&referenced) {
                return true;
            }
        }
        false
    }

    /// Best-effort static Deep type expression for an expression in the
    /// current frame: the checker's own `{type: ...}` annotation on the
    /// node, else (for a var) the frame binding's declared type, else
    /// the top-level type-env entry. A local binding with an unknown
    /// type masks any same-named top-level entry (explicit `None`
    /// marker in `binding_types`), so a shadow is never mistyped.
    pub(super) fn static_type_expr_of(&self, expr: &Expr) -> Option<Expr> {
        match expr {
            Expr::MetaExpr(meta, _) => {
                if let Some((_, ty)) = meta.entries.iter().find(|(k, _)| k == "type") {
                    return Some(ty.clone());
                }
                self.static_type_expr_of(&meta.expr)
            }
            Expr::List(list, _) => {
                if let Some(meta) = get_meta(list)
                    && let Some((_, ty)) = meta.entries.iter().find(|(k, _)| k == "type")
                {
                    return Some(ty.clone());
                }
                if tag(list) == Some("var")
                    && let Some(name) = children(list).first().and_then(symbol_name)
                {
                    if let Some(declared) = self.binding_types.get(name) {
                        return declared.clone();
                    }
                    if let Some(ty) = self.type_env.get(name) {
                        return Some(ty.clone());
                    }
                    if let Some((resolved, _)) = self.lookup_top_level_def(name)
                        && let Some(ty) = self.type_env.get(&resolved)
                    {
                        return Some(ty.clone());
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// chelis#338 site A: evaluate a reduction whose axis argument is a
    /// named axis by routing `(app <reduce> <operand> <axes...>)`
    /// through IR lowering + the forward DAG evaluator. The operand is
    /// evaluated by the interpreter first (so nested def calls, pipes,
    /// and lets keep host semantics), then staged as a typed
    /// placeholder whose `TensorType` comes from the operand
    /// expression's static type. Lowering resolves the axis name
    /// against those named dims exactly as the C backend does
    /// (`resolve_reduce_axis`).
    pub(super) fn eval_named_axis_reduction_app(
        &mut self,
        reduce_name: &str,
        kids: &[Expr],
    ) -> Result<RuntimeValue, String> {
        let operand_expr = kids
            .get(1)
            .ok_or_else(|| format!("{reduce_name} missing operand"))?;
        let axis_list = kids[2..]
            .iter()
            .filter_map(var_name)
            .collect::<Vec<_>>()
            .join(", ");
        let Some(operand_ty_expr) = self.static_type_expr_of(operand_expr) else {
            return Err(format!(
                "named-axis reduction `{reduce_name}(.., {axis_list})` cannot be evaluated \
                 here: the operand has no statically known tensor type in the host runtime, \
                 so the named axis cannot be resolved to an index (chelis#338). Bind the \
                 operand to a parameter or binding with a declared tensor type."
            ));
        };
        let operand = match self.eval_expr(operand_expr)? {
            RuntimeValue::Tensor(tensor) => tensor,
            other => {
                return Err(format!(
                    "{reduce_name} expects a tensor operand, got {other:?}"
                ));
            }
        };
        let operand_type =
            declared_tensor_type_for_value(&operand_ty_expr, &operand).map_err(|err| {
                format!(
                    "named-axis reduction `{reduce_name}(.., {axis_list})` cannot be \
                     evaluated here: {err} (chelis#338)"
                )
            })?;
        let span = Span::new(0, 0);
        let placeholder = "__chelis_named_axis_operand";
        let mut app_elements = vec![
            Expr::Atom(Atom::Symbol("app".to_string()), span),
            Expr::Map(MetaMap::default(), span),
            Expr::List(
                List {
                    elements: vec![
                        Expr::Atom(Atom::Symbol("var".to_string()), span),
                        Expr::Map(MetaMap::default(), span),
                        Expr::Atom(Atom::Symbol(reduce_name.to_string()), span),
                    ],
                },
                span,
            ),
            make_var_with_type(placeholder, &operand_type, span),
        ];
        app_elements.extend(kids[2..].iter().cloned());
        let app_expr = Expr::List(
            List {
                elements: app_elements,
            },
            span,
        );
        let scoped = HashMap::from([(placeholder.to_string(), operand_type)]);
        let staged = HashMap::from([(placeholder.to_string(), operand.value.clone())]);
        self.route_named_axis_expr(&app_expr, scoped, staged, reduce_name)
            .map_err(NamedAxisRouteError::into_message)
    }

    /// chelis#338 site B: route a call to a def that requires named-axis
    /// routing through IR lowering at this call boundary, staging the
    /// already-evaluated arguments as typed placeholders. Placeholder
    /// types come from the callee's declared formal param types (the
    /// boundary `chelis build` uses when the host lane calls a
    /// signature-typed compiled function); a rank-polymorphic or
    /// missing formal falls back to the argument expression's static
    /// type. Returns `Ok(None)` when no routing strategy applies (an
    /// unmarshalable argument, an untypeable tensor arg, or a body that
    /// does not lower): the caller then interprets the call normally
    /// and the body's reduction is handled at site A with frame-typed
    /// bindings, so terminal failures stay loud rather than silent.
    pub(super) fn try_named_axis_def_call(
        &mut self,
        resolved_name: &str,
        def_expr: &Expr,
        kids: &[Expr],
        args: &[RuntimeValue],
    ) -> Result<Option<RuntimeValue>, String> {
        let span = Span::new(0, 0);
        let mut scoped: HashMap<String, TensorType> = HashMap::with_capacity(args.len());
        let mut staged: HashMap<String, IrTensorValue> = HashMap::with_capacity(args.len());
        let mut app_elements: Vec<Expr> = Vec::with_capacity(3 + args.len());
        app_elements.push(Expr::Atom(Atom::Symbol("app".to_string()), span));
        app_elements.push(Expr::Map(MetaMap::default(), span));
        app_elements.push(Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Symbol("var".to_string()), span),
                    Expr::Map(MetaMap::default(), span),
                    Expr::Atom(Atom::Symbol(resolved_name.to_string()), span),
                ],
            },
            span,
        ));
        for (index, value) in args.iter().enumerate() {
            let placeholder = format!("__chelis_named_reduce_arg_{index}");
            let (tensor_value, tensor_ty) = match value {
                RuntimeValue::Tensor(tensor) => {
                    let from_formal = param_type_expr_at(def_expr, index)
                        .and_then(|formal| declared_tensor_type_for_value(formal, tensor).ok());
                    let resolved_ty = match from_formal {
                        Some(ty) => ty,
                        None => {
                            let from_arg = kids
                                .get(1 + index)
                                .and_then(|arg_expr| self.static_type_expr_of(arg_expr))
                                .and_then(|ty_expr| {
                                    declared_tensor_type_for_value(&ty_expr, tensor).ok()
                                });
                            match from_arg {
                                Some(ty) => ty,
                                None => return Ok(None),
                            }
                        }
                    };
                    (tensor.value.clone(), resolved_ty)
                }
                RuntimeValue::Scalar(_) | RuntimeValue::Bool(_) => {
                    runtime_value_to_dag_input(value, Some(def_expr), index)?
                }
                _ => return Ok(None),
            };
            app_elements.push(make_var_with_type(&placeholder, &tensor_ty, span));
            scoped.insert(placeholder.clone(), tensor_ty);
            staged.insert(placeholder, tensor_value);
        }
        let app_expr = Expr::List(
            List {
                elements: app_elements,
            },
            span,
        );
        match self.route_named_axis_expr(&app_expr, scoped, staged, resolved_name) {
            Ok(value) => Ok(Some(
                self.unwrap_declared_scalar_return(resolved_name, value)?,
            )),
            // Not lowerable (host-shaped body): deterministic ladder,
            // fall back to interpretation; site A handles the body's
            // reduction or fails loudly.
            Err(NamedAxisRouteError::NotLowerable(_)) => Ok(None),
            Err(NamedAxisRouteError::Fatal(message)) => Err(message),
        }
    }

    /// Shared chelis#338 routing core: lower `routed_expr` with the
    /// staged placeholder types + the merged type-env + the full def
    /// table (the same universe `apply_transform` uses), forward-eval
    /// the DAG, and pack the roots back into a `RuntimeValue`.
    fn route_named_axis_expr(
        &mut self,
        routed_expr: &Expr,
        scoped_types: HashMap<String, TensorType>,
        staged_inputs: HashMap<String, IrTensorValue>,
        context_label: &str,
    ) -> Result<RuntimeValue, NamedAxisRouteError> {
        self.pre_resolve_top_level_value_refs(routed_expr)
            .map_err(NamedAxisRouteError::Fatal)?;
        let program_defs = self.top_level_defs.clone();
        if let Some(name) = find_reachable_host_only_builtin_call(routed_expr, &program_defs) {
            return Err(NamedAxisRouteError::NotLowerable(format!(
                "host runtime: named-axis routing of `{context_label}` reaches host-runtime-only \
                 builtin `{name}`, which has no RISC DAG lowering \
                 (spec/05-risc-primitives.md SS3.6)"
            )));
        }
        let dag = try_lower_subexpr_program(
            routed_expr,
            scoped_types,
            self.type_env.clone(),
            program_defs,
        )
        .map_err(|diagnostic| {
            NamedAxisRouteError::NotLowerable(format!(
                "host runtime could not lower the named-axis `{context_label}` call for \
                 evaluation (chelis#338): {diagnostic}"
            ))
        })?;
        let roots: Vec<NodeId> = dag.roots().to_vec();
        if roots.is_empty() {
            return Err(NamedAxisRouteError::Fatal(format!(
                "host runtime: named-axis `{context_label}` lowering produced no roots"
            )));
        }
        let tensor_bindings = self.tensor_bindings;
        let host_bindings = &self.bindings;
        let values = chelis_ir::eval::eval_tensor_roots_with_strict(&dag, &roots, |name| {
            staged_inputs
                .get(name)
                .cloned()
                .or_else(|| tensor_bindings.get(name).map(|t| t.value.clone()))
                .or_else(|| match host_bindings.get(name) {
                    Some(RuntimeValue::Tensor(t)) => Some(t.value.clone()),
                    _ => None,
                })
        })
        .map_err(|err| {
            NamedAxisRouteError::Fatal(format!(
                "host runtime named-axis `{context_label}` evaluation failed: {err}"
            ))
        })?;
        pack_dag_roots(&dag, &roots, &values, context_label).map_err(NamedAxisRouteError::Fatal)
    }

    /// A routed def declared to return a scalar (`-> f32` etc.) comes
    /// back from the DAG as a rank-0 tensor; convert it so downstream
    /// host arithmetic sees the declared scalar type.
    fn unwrap_declared_scalar_return(
        &self,
        resolved_name: &str,
        value: RuntimeValue,
    ) -> Result<RuntimeValue, String> {
        let RuntimeValue::Tensor(ref tensor) = value else {
            return Ok(value);
        };
        if !tensor.value.shape.is_empty() {
            return Ok(value);
        }
        let Some(Expr::List(sig_list, _)) = self.type_env.get(resolved_name) else {
            return Ok(value);
        };
        if tag(sig_list) != Some("t-fn") {
            return Ok(value);
        }
        let Some(ret) = children(sig_list).last() else {
            return Ok(value);
        };
        let is_prim_return =
            matches!(ret, Expr::List(ret_list, _) if tag(ret_list) == Some("t-prim"));
        if !is_prim_return {
            return Ok(value);
        }
        let Some(prim) = extract_prim_from_type_expr(ret) else {
            return Ok(value);
        };
        if !(prim.is_float() || prim.is_integer()) {
            return Ok(value);
        }
        let scalar_value = tensor
            .value
            .data
            .first()
            .copied()
            .ok_or_else(|| "rank-0 tensor with no data in named-axis result".to_string())?;
        let bits = ScalarBits::from_f64_as(prim, scalar_value)?;
        RuntimeValue::scalar(prim, bits)
    }

    /// Force any top-level *value* bindings referenced (transitively
    /// through def bodies) by a routed expression to be evaluated
    /// before DAG evaluation, so the strict load callback can serve
    /// them from `bindings` (the lowerer emits `Load(name)` for such
    /// free names).
    fn pre_resolve_top_level_value_refs(&mut self, root: &Expr) -> Result<(), String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut vars: Vec<String> = Vec::new();
        collect_var_names(root, &mut vars);
        while let Some(name) = vars.pop() {
            if !visited.insert(name.clone()) {
                continue;
            }
            let Some((resolved, body)) = self.lookup_top_level_def(&name) else {
                continue;
            };
            if !visited.insert(resolved.clone()) && resolved != name {
                continue;
            }
            if matches!(&body, Expr::List(body_list, _) if tag(body_list) == Some("fn")) {
                collect_var_names(&body, &mut vars);
            } else if !self.resolving_top_levels.iter().any(|n| n == &resolved) {
                // Skip a binding currently being resolved (this walk is
                // syntactic and may reach the in-flight root through a
                // dead branch); forcing it would raise a spurious
                // "cyclic top-level" error. If the lowered DAG genuinely
                // needs the value, the strict load callback reports it.
                let _ = self.resolve_top_level(&resolved)?;
            }
        }
        Ok(())
    }
}

/// Pack forward-evaluated DAG roots into a `RuntimeValue` (single root
/// becomes a Tensor, several become a Tuple), with precision pulled
/// from each root node's output type. Shared by the grad/vmap
/// transform lane and the chelis#338 named-axis routing lane.
pub(super) fn pack_dag_roots(
    dag: &Dag,
    roots: &[NodeId],
    values: &HashMap<NodeId, IrTensorValue>,
    context_label: &str,
) -> Result<RuntimeValue, String> {
    let mut packed: Vec<RuntimeValue> = Vec::with_capacity(roots.len());
    for root in roots {
        let tensor = values.get(root).cloned().ok_or_else(|| {
            format!(
                "host runtime: missing root {} in {context_label} eval output",
                root.0
            )
        })?;
        let precision = dag
            .get(*root)
            .map(|node| node.output_type.precision)
            .unwrap_or(Prim::F32);
        packed.push(RuntimeValue::Tensor(RuntimeTensorValue {
            value: tensor,
            precision,
        }));
    }
    if packed.len() == 1 {
        Ok(packed.pop().expect("checked length"))
    } else {
        Ok(RuntimeValue::Tuple(packed))
    }
}

/// Reduction builtins whose axis argument may be a *named axis*: a bare
/// `(var name)` naming a dimension of the operand rather than an
/// integer index (Tier-3 rank polymorphism, spec/04-type-system.md
/// §4.5.3). The checker admits only int-literal or named axes for these
/// (`check_reduction_signature`), so at eval time a bare-var axis is
/// always a named axis, never a runtime value reference.
pub(super) const REDUCTION_BUILTIN_NAMES: &[&str] = &[
    "sum",
    "mean",
    "max_reduce",
    "min_reduce",
    "prod_reduce",
    "argmax_reduce",
    "argmin_reduce",
];

/// Error split for the chelis#338 named-axis routing core, so callers
/// can distinguish "this expression does not lower to the tensor DAG"
/// (site B falls back to ordinary interpretation, where site A handles
/// the inner reduction) from genuine post-lowering failures (always
/// terminal). This keeps the strategy ladder deterministic without a
/// silent catch-all fallback.
enum NamedAxisRouteError {
    /// The expression has no tensor-DAG lowering (host-shaped body,
    /// host-only builtin, ...). Recoverable by interpretation.
    NotLowerable(String),
    /// Lowering succeeded but DAG evaluation or packing failed.
    Fatal(String),
}

impl NamedAxisRouteError {
    fn into_message(self) -> String {
        match self {
            NamedAxisRouteError::NotLowerable(message) | NamedAxisRouteError::Fatal(message) => {
                message
            }
        }
    }
}

/// Is this list an `(app <reduce> <operand> <axes...>)` whose callee is
/// a reduction builtin with at least one bare-var (named) axis?
fn app_reduces_named_axis(list: &List) -> bool {
    if tag(list) != Some("app") {
        return false;
    }
    let kids = children(list);
    let Some(callee) = kids.first().and_then(var_name) else {
        return false;
    };
    if !REDUCTION_BUILTIN_NAMES.contains(&callee) {
        return false;
    }
    kids.len() >= 3 && kids[2..].iter().any(|axis| var_name(axis).is_some())
}

/// Walk a Deep expr looking for a named-axis reduction app, collecting
/// every `(var name)` reference on the way so the caller can follow
/// them into def bodies (mirrors [`scan_expr_for_host_only`]).
fn scan_expr_for_named_axis_reduction(expr: &Expr, hit: &mut bool, vars: &mut Vec<String>) {
    if *hit {
        return;
    }
    match expr {
        Expr::MetaExpr(meta, _) => scan_expr_for_named_axis_reduction(&meta.expr, hit, vars),
        Expr::List(list, _) => {
            if app_reduces_named_axis(list) {
                *hit = true;
                return;
            }
            if tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                vars.push(name.to_string());
            }
            for child in &list.elements {
                scan_expr_for_named_axis_reduction(child, hit, vars);
            }
        }
        _ => {}
    }
}

/// Collect every `(var name)` reference in a Deep expr (no early exit).
fn collect_var_names(expr: &Expr, vars: &mut Vec<String>) {
    match expr {
        Expr::MetaExpr(meta, _) => collect_var_names(&meta.expr, vars),
        Expr::List(list, _) => {
            if tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                vars.push(name.to_string());
            }
            for child in &list.elements {
                collect_var_names(child, vars);
            }
        }
        _ => {}
    }
}

/// Strip `t-ref` wrappers (and MetaExpr shells) off a Deep type expr.
fn strip_type_wrappers(ty_expr: &Expr) -> &Expr {
    match ty_expr {
        Expr::List(list, _) if tag(list) == Some("t-ref") => children(list)
            .first()
            .map(strip_type_wrappers)
            .unwrap_or(ty_expr),
        Expr::MetaExpr(meta, _) => strip_type_wrappers(&meta.expr),
        _ => ty_expr,
    }
}

/// Resolve a declared/static Deep type expression against a runtime
/// tensor's shape, producing the concrete `TensorType` (named dims with
/// sizes filled in) used to stage the tensor as a typed DAG input for
/// chelis#338 named-axis routing. Rank-polymorphic types (`d-rank`
/// spreads) are rejected: a spread cannot be split against a bare
/// runtime shape (the split is ambiguous; only the checker's
/// name-anchored unification may do it).
pub(super) fn declared_tensor_type_for_value(
    ty_expr: &Expr,
    tensor: &RuntimeTensorValue,
) -> Result<TensorType, String> {
    declared_tensor_type_for_shape(ty_expr, &tensor.value.shape, tensor.precision)
}

/// Shape-slice core of [`declared_tensor_type_for_value`], shared with
/// the chelis#351 vmap-lane placeholder synthesis (which types the
/// UNBATCHED view of a batched actual against the callee's formal, so
/// it has a bare shape rather than a whole `RuntimeTensorValue`).
pub(super) fn declared_tensor_type_for_shape(
    ty_expr: &Expr,
    shape: &[usize],
    fallback_precision: Prim,
) -> Result<TensorType, String> {
    let stripped = strip_type_wrappers(ty_expr);
    let Expr::List(list, _) = stripped else {
        return Err("the static type is not a tensor type".to_string());
    };
    match tag(list) {
        Some("t-tensor") => {}
        Some("t-prim") if shape.is_empty() => {
            return Ok(TensorType {
                dims: vec![],
                precision: extract_prim_from_type_expr(stripped).unwrap_or(fallback_precision),
            });
        }
        other => {
            return Err(format!(
                "the static type `{}` is not a tensor type",
                other.unwrap_or("?")
            ));
        }
    }
    let kids = children(list);
    let Some((prim_expr, dim_exprs)) = kids.split_last() else {
        return Err("malformed t-tensor type (no children)".to_string());
    };
    let precision = extract_prim_from_type_expr(prim_expr).unwrap_or(fallback_precision);
    if dim_exprs
        .iter()
        .any(|d| matches!(d, Expr::List(dim_list, _) if tag(dim_list) == Some("d-rank")))
    {
        return Err(
            "the operand's declared type is rank-polymorphic (contains a `..spread`), \
             which cannot be split against a runtime shape"
                .to_string(),
        );
    }
    if dim_exprs.len() != shape.len() {
        return Err(format!(
            "declared rank {} does not match runtime rank {}",
            dim_exprs.len(),
            shape.len()
        ));
    }
    let mut dims = Vec::with_capacity(shape.len());
    for (dim_expr, &size) in dim_exprs.iter().zip(shape.iter()) {
        let Expr::List(dim_list, _) = dim_expr else {
            return Err("malformed tensor dimension in static type".to_string());
        };
        match tag(dim_list) {
            Some("d-name") => {
                let name = children(dim_list)
                    .first()
                    .and_then(symbol_name)
                    .ok_or_else(|| "malformed d-name dimension".to_string())?;
                dims.push(DimInfo::Named(name.to_string(), Some(size)));
            }
            Some("d-lit") => {
                let lit = children(dim_list)
                    .first()
                    .and_then(int_value)
                    .ok_or_else(|| "malformed d-lit dimension".to_string())?;
                if lit < 0 || lit as usize != size {
                    return Err(format!(
                        "declared dimension {lit} does not match runtime size {size}"
                    ));
                }
                dims.push(DimInfo::Lit(size));
            }
            // A dim VARIABLE (surf desugars single-lowercase-letter dims
            // like `tensor[a, seq, f32]` to `d-var`): at this staged
            // boundary the runtime shape monomorphizes it, exactly as a
            // build call site binds it. It carries no anchor name, so a
            // concrete Lit is the faithful staging (chelis#346 red-team F5).
            Some("d-var") => dims.push(DimInfo::Lit(size)),
            other => {
                return Err(format!(
                    "unsupported dimension form `{}` in static type",
                    other.unwrap_or("?")
                ));
            }
        }
    }
    Ok(TensorType { dims, precision })
}
