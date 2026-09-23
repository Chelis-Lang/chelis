use chelis_deep::DeepTag;
use chelis_unord::UnordMap;

use chelis_deep::Span;
use chelis_deep::ast::{Expr, Metadata};
use chelis_ir::dag::{Dag, DimInfo, NodeId, TensorType};
use chelis_ir::eval::{TensorInputDemand, TensorValue as IrTensorValue};
use chelis_ir::lower::type_expr_has_rank_var;
use chelis_types::types::Prim;

use super::transforms::*;
use super::*;

#[cfg(test)]
thread_local! {
    /// Named-axis routings on this thread (chelis#2207). A receipt that the
    /// routing lane is reached at all: a fold-count test over a fixture that
    /// stopped routing would otherwise pass vacuously, because the ordinary
    /// host path computes the same answer.
    static NAMED_AXIS_ROUTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Named-axis routings on this thread since the last reset.
#[cfg(test)]
pub(crate) fn named_axis_routes() -> u64 {
    NAMED_AXIS_ROUTES.with(std::cell::Cell::get)
}

/// Reset [`named_axis_routes`] for this thread.
#[cfg(test)]
pub(crate) fn reset_named_axis_routes() {
    NAMED_AXIS_ROUTES.with(|routes| routes.set(0));
}

impl<'a> EvalContext<'a> {
    /// chelis#338: does evaluating a call to `resolved_name` require
    /// routing through IR lowering because a *named-axis* op is
    /// involved? True when the def's body contains a reduction whose
    /// axis argument is a bare `(var name)` — or a named-axis expand
    /// (chelis#339), whose axis slot names the inserted axis — or when
    /// it references a rank-polymorphic def that does. References through defs with
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
        let Some(body) = self.program.defs().get(resolved_name).cloned() else {
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
                .program
                .type_env()
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
                if let Some(ty) = meta.metadata.ty().map(|ty| ty.expression()) {
                    return Some(ty.clone());
                }
                self.static_type_expr_of(&meta.expr)
            }
            Expr::Node(node, _) => {
                if let Some(ty) = node.meta().ty().map(|ty| ty.expression()) {
                    return Some(ty.clone());
                }
                if node.tag() == DeepTag::Var
                    && let Some(name) = node.children_slice().first().and_then(symbol_name)
                {
                    if let Some(declared) = self.binding_types.get(name) {
                        return declared.clone();
                    }
                    if let Some(ty) = self.program.type_env().get(name) {
                        return Some(ty.clone());
                    }
                    if let Some((resolved, _)) = self.lookup_top_level_def(name)
                        && let Some(ty) = self.program.type_env().get(&resolved)
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
    /// named axis — or a chelis#339 named-axis expand — by routing
    /// `(app <op> <operand> <axes...>)`
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
                "named-axis `{reduce_name}(.., {axis_list})` cannot be evaluated \
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
                    "named-axis `{reduce_name}(.., {axis_list})` cannot be \
                     evaluated here: {err} (chelis#338)"
                )
            })?;
        let span = Span::new(0, 0);
        let placeholder = "__chelis_named_axis_operand";
        // The axis arguments keep their `app` argument positions, so no
        // position-bound metadata moves.
        let mut app_children = vec![
            var_expr(reduce_name, span),
            make_var_with_type(placeholder, &operand_type, span),
        ];
        app_children.extend(kids[2..].iter().cloned());
        let app_expr = Expr::node(DeepTag::App, Metadata::default(), app_children, span);
        let scoped = UnordMap::from([(placeholder.to_string(), operand_type)]);
        let staged = UnordMap::from([(placeholder.to_string(), operand.value.clone())]);
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
        retain_source_controls: bool,
    ) -> Result<Option<RuntimeValue>, String> {
        let span = Span::new(0, 0);
        let mut scoped: UnordMap<String, TensorType> = UnordMap::new();
        let mut staged: UnordMap<String, IrTensorValue> = UnordMap::new();
        let mut app_children: Vec<Expr> = Vec::with_capacity(1 + args.len());
        app_children.push(var_expr(resolved_name, span));
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
                    runtime_value_to_dag_input_lossy(value, Some(def_expr), index)?
                }
                _ => return Ok(None),
            };
            // A fixed source call proves its control expressions before any
            // arguments are evaluated. Keep only scalar expressions needed
            // for that proof; data operands still cross this boundary once.
            let argument = make_var_with_type(&placeholder, &tensor_ty, span);
            let argument = if retain_source_controls && matches!(value, RuntimeValue::Scalar(_)) {
                let mut trial = kids.to_vec();
                trial[index + 1] = argument.clone();
                let trial = Expr::node(DeepTag::App, Metadata::default(), trial, span);
                if self.program_evaluation_profile(&trial)
                    == chelis_ir::evaluation::EvaluationProfile::FixedControl
                {
                    argument
                } else {
                    kids[index + 1].clone()
                }
            } else {
                argument
            };
            app_children.push(argument);
            scoped.insert(placeholder.clone(), tensor_ty);
            staged.insert(placeholder, tensor_value);
        }
        let app_expr = Expr::node(DeepTag::App, Metadata::default(), app_children, span);
        // Source actuals and their checked types were prepared in the caller
        // above. Free loads in the named body belong to declaration scope.
        let saved = std::mem::take(&mut self.bindings);
        let saved_types = std::mem::take(&mut self.binding_types);
        let saved_precisions = std::mem::take(&mut self.precision_bindings);
        let routed = self.route_named_axis_expr(&app_expr, scoped, staged, resolved_name);
        self.bindings = saved;
        self.binding_types = saved_types;
        self.precision_bindings = saved_precisions;
        match routed {
            Ok(value) => Ok(Some(
                self.unwrap_declared_scalar_return(resolved_name, value)?,
            )),
            // chelis#1277 B2h: a lowering failure after the routing decision
            // is the evaluation's error, not a silent fall-through to the
            // interpreter; the decision itself stays by classification (the
            // `eval_app` gate and the `Ok(None)` returns above for arguments
            // this boundary cannot type). This retires the eval-lane sibling
            // of the C lane's chelis#1515 fall-through.
            Err(error) => Err(error.into_message()),
        }
    }

    /// Shared chelis#338 routing core: lower `routed_expr` with the
    /// staged placeholder types + the merged type-env + the full def
    /// table (the same universe `apply_transform` uses), forward-eval
    /// the DAG, and pack the roots back into a `RuntimeValue`.
    fn route_named_axis_expr(
        &mut self,
        routed_expr: &Expr,
        scoped_types: UnordMap<String, TensorType>,
        staged_inputs: UnordMap<String, IrTensorValue>,
        context_label: &str,
    ) -> Result<RuntimeValue, NamedAxisRouteError> {
        #[cfg(test)]
        NAMED_AXIS_ROUTES.with(|routes| routes.set(routes.get() + 1));
        if let Some(name) = find_reachable_host_only_builtin_call(routed_expr, self.program.defs())
        {
            return Err(NamedAxisRouteError::NotLowerable(format!(
                "host runtime: named-axis routing of `{context_label}` reaches host-runtime-only \
                 builtin `{name}`, which has no RISC DAG lowering \
                 (spec/05-risc-primitives.md SS3.6)"
            )));
        }
        let profile = self.program_evaluation_profile(routed_expr);
        // The lowering universe of a routed reduction is the program's own
        // type environment and definition table, both fixed for this
        // evaluation context. Both branches below used to hand those two
        // tables to a free `try_lower_*` entry, which sorted them, deep-cloned
        // them and folded the pipes in every definition -- all of `chelis-std`
        // included -- once per routed reduction (chelis#2207). The scope
        // prepares that context once; cloning it here is four `Arc` bumps and
        // releases the borrow on `self` that the input provider below needs.
        let lowering_context = self.program.routing_lowering_context();
        let mut execution_plan = None;
        let lowered = if profile == chelis_ir::evaluation::EvaluationProfile::FixedControl {
            let context = chelis_ir::evaluation::RandomExecutionContext::new(
                chelis_ir::host::RandomLoweringState {
                    seed: self.random_seed,
                    counter: self.random_counter,
                },
            );
            chelis_ir::lower::try_lower_subexpr_evaluation_plan_with_context(
                routed_expr,
                scoped_types,
                &lowering_context,
                &context,
            )
            .map(|plan| {
                let dag = plan.dag_for_inspection().clone();
                execution_plan = Some(plan);
                dag
            })
        } else {
            chelis_ir::lower::try_lower_subexpr_program_with_context(
                routed_expr,
                scoped_types,
                &lowering_context,
            )
        };
        let dag = lowered.map_err(|diagnostic| {
            let message = format!(
                "host runtime could not lower the named-axis `{context_label}` call for \
                 evaluation (chelis#338): {diagnostic}"
            );
            if profile == chelis_ir::evaluation::EvaluationProfile::FixedControl {
                NamedAxisRouteError::Fatal(message)
            } else {
                NamedAxisRouteError::NotLowerable(message)
            }
        })?;
        let roots: Vec<NodeId> = dag.roots().to_vec();
        if roots.is_empty() {
            return Err(NamedAxisRouteError::Fatal(format!(
                "host runtime: named-axis `{context_label}` lowering produced no roots"
            )));
        }
        let preparation_context = chelis_ir::evaluation::RandomExecutionContext::new(
            chelis_ir::host::RandomLoweringState {
                seed: self.random_seed,
                counter: self.random_counter,
            },
        );
        let mut provider_failed = false;
        let prepare = |name: &str, demand| {
            self.prepare_named_axis_input(name, demand, &staged_inputs)
                .inspect_err(|_| provider_failed = true)
        };
        let prepared = if let Some(plan) = &execution_plan {
            chelis_ir::eval::prepare_tensor_plan_inputs_with_demand(
                plan,
                &preparation_context,
                prepare,
            )
        } else {
            chelis_ir::eval::prepare_tensor_roots_inputs_with_demand(&dag, &roots, prepare)
        }
        .map_err(|err| {
            if provider_failed || execution_plan.is_some() {
                NamedAxisRouteError::Fatal(err)
            } else {
                NamedAxisRouteError::Fatal(format!(
                    "host runtime named-axis `{context_label}` evaluation failed: {err}"
                ))
            }
        })?;
        let load = |name: &str| prepared.get(name).cloned();
        let result = if let Some(plan) = &execution_plan {
            // Preparation may enter a fallible initializer. Execute the original
            // plan using the resulting host state, not its pre-preparation copy.
            let mut context = chelis_ir::evaluation::RandomExecutionContext::new(
                chelis_ir::host::RandomLoweringState {
                    seed: self.random_seed,
                    counter: self.random_counter,
                },
            );
            let result = chelis_ir::eval::eval_tensor_plan_with_strict(plan, &mut context, load);
            self.random_counter = context.state().counter;
            result
        } else {
            chelis_ir::eval::eval_tensor_roots_with_strict(&dag, &roots, load)
        };
        let values = result.map_err(|err| {
            if execution_plan.is_some() {
                return NamedAxisRouteError::Fatal(err);
            }
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
        let Some((DeepTag::TFn, sig_kids)) = self
            .program
            .type_env()
            .get(resolved_name)
            .and_then(tagged_expr_children)
        else {
            return Ok(value);
        };
        let Some(ret) = sig_kids.last() else {
            return Ok(value);
        };
        if ret.tag() != Some(DeepTag::TPrim) {
            return Ok(value);
        }
        let Some(prim) = extract_prim_from_type_expr(ret) else {
            return Ok(value);
        };
        if !(prim.is_float() || prim.is_integer()) {
            return Ok(value);
        }
        if tensor.value.is_empty() {
            return Err("rank-0 tensor with no data in named-axis result".to_string());
        }
        // The DAG evaluator finalized this element at the root's declared
        // dtype; re-finalizing the exact element at the annotated prim is
        // the ingress form (identity when the dtypes already agree).
        let element = tensor.value.storage().scalar_at(0);
        let value = match element.as_i64_exact() {
            Some(v) => chelis_types::scalar_from_i64("named_axis", prim, v)
                .map_err(|trap| trap.to_string())?,
            None => chelis_types::scalar_from_f64("named_axis", prim, element.as_f64_lossy())
                .map_err(|trap| trap.to_string())?,
        };
        Ok(RuntimeValue::from_scalar_value(value))
    }

    /// Serve available tensors first. A surplus shape witness is a query, not
    /// permission to enter a declaration initializer or observe a callable.
    pub(super) fn prepare_named_axis_input(
        &mut self,
        name: &str,
        demand: TensorInputDemand,
        staged_inputs: &UnordMap<String, IrTensorValue>,
    ) -> Result<Option<IrTensorValue>, String> {
        if let Some(value) = staged_inputs.get(name) {
            return Ok(Some(value.clone()));
        }
        if let Some(value) = self.tensor_bindings.get(name) {
            return Ok(Some(value.value.clone()));
        }
        if let Some(RuntimeValue::Tensor(value)) = self.bindings.get(name) {
            return Ok(Some(value.value.clone()));
        }
        let declaration = self.lookup_top_level_def(name);
        if let Some((resolved, _)) = &declaration
            && let Some(RuntimeValue::Tensor(value)) = self.declaration_values.get(resolved)
        {
            return Ok(Some(value.value.clone()));
        }
        if demand == TensorInputDemand::AvailableShape {
            return Ok(None);
        }
        let Some((resolved, body)) = declaration else {
            return Ok(None);
        };
        if self
            .program
            .type_env()
            .get(&resolved)
            .is_some_and(|ty| ty.tag() == Some(DeepTag::TFn))
            || body.tag() == Some(DeepTag::Fn)
            || self.resolving_top_levels.iter().any(|n| n == &resolved)
        {
            // Retain the strict missing-input boundary for an in-flight root;
            // input preparation does not add callable observation or recursion.
            return Ok(None);
        }
        match self.resolve_top_level(&resolved)? {
            RuntimeValue::Tensor(value) => Ok(Some(value.value)),
            _ => Ok(None),
        }
    }
}

/// Pack forward-evaluated DAG roots into a `RuntimeValue` (single root
/// becomes a Tensor, several become a Tuple), with precision pulled
/// from each root node's output type. Shared by the grad/vmap
/// transform lane and the chelis#338 named-axis routing lane.
pub(super) fn pack_dag_roots(
    dag: &Dag,
    roots: &[NodeId],
    values: &UnordMap<NodeId, IrTensorValue>,
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
        // chelis#730 Phase 1 (census row 14, section C1.4 raise-or-prove):
        // a root id missing from the DAG it was just packed from is an
        // internal desync with no user-facing driver; erroring beats the
        // former silent F32 precision default.
        let precision = dag
            .get(*root)
            .map(|node| node.output_type.precision)
            .ok_or_else(|| {
                format!(
                    "host runtime: root {} missing from the {context_label} DAG while \
                     packing result precision (internal desync; was a silent F32 \
                     default - spec/design/loud_unsupported.md section C1.4)",
                    root.0
                )
            })?;
        debug_assert_eq!(
            tensor.prim(),
            precision,
            "the DAG evaluator finalizes at the root's declared dtype"
        );
        packed.push(RuntimeValue::Tensor(RuntimeTensorValue::new(tensor)));
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
    "count",
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

/// Is this node an `(app <reduce> <operand> <axes...>)` whose callee is
/// a reduction builtin with at least one bare-var (named) axis?
fn app_reduces_named_axis(node: &chelis_deep::node::Node) -> bool {
    if node.tag() != DeepTag::App {
        return false;
    }
    let kids = node.children_slice();
    let Some(callee) = kids.first().and_then(var_name) else {
        return false;
    };
    if !REDUCTION_BUILTIN_NAMES.contains(&callee) {
        return false;
    }
    kids.len() >= 3 && kids[2..].iter().any(|axis| var_name(axis).is_some())
}

/// chelis#339 twin: is this an `(app expand <operand> <name> <size>
/// <anchor?>)` named-axis expand — the axis slot is a bare `(var name)`
/// naming the inserted axis? (The positional form with an integer axis,
/// possibly carrying a symbolic *size*, is NOT a named-axis app and
/// keeps the host path.)
fn app_expands_named_axis(node: &chelis_deep::node::Node) -> bool {
    if node.tag() != DeepTag::App {
        return false;
    }
    let kids = node.children_slice();
    let Some(callee) = kids.first().and_then(var_name) else {
        return false;
    };
    (callee == "expand" || callee == "insert")
        && kids.len() >= 4
        && kids.get(2).and_then(var_name).is_some()
}

/// Walk a Deep expr looking for a named-axis reduction or expand app,
/// collecting every `(var name)` reference on the way so the caller can
/// follow them into def bodies (mirrors [`scan_expr_for_host_only`]).
fn scan_expr_for_named_axis_reduction(expr: &Expr, hit: &mut bool, vars: &mut Vec<String>) {
    if *hit {
        return;
    }
    match expr {
        Expr::MetaExpr(meta, _) => scan_expr_for_named_axis_reduction(&meta.expr, hit, vars),
        Expr::Node(node, _) => {
            if app_reduces_named_axis(node) || app_expands_named_axis(node) {
                *hit = true;
                return;
            }
            if node.tag() == DeepTag::Var
                && let Some(name) = node.children_slice().first().and_then(symbol_name)
            {
                vars.push(name.to_string());
            }
            for child in node.children_slice() {
                scan_expr_for_named_axis_reduction(child, hit, vars);
            }
        }
        // A structural list (for example a typed parameter) is walked as
        // the untagged list it replaced was.
        Expr::BareList(elements, _) => {
            for element in elements {
                scan_expr_for_named_axis_reduction(element, hit, vars);
            }
        }
        Expr::Atom(..) | Expr::Map(..) | Expr::UnknownForm(..) => {}
    }
}

/// Strip `t-ref` wrappers (and MetaExpr shells) off a Deep type expr.
pub(super) fn strip_type_wrappers(ty_expr: &Expr) -> &Expr {
    match ty_expr {
        Expr::Node(node, _) if node.tag() == DeepTag::TRef => node
            .children_slice()
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
    declared_tensor_type_for_shape(ty_expr, &tensor.value.shape, tensor.precision, false)
}

/// Shape-slice core of [`declared_tensor_type_for_value`], shared with
/// the chelis#351 vmap-lane placeholder synthesis (which types the
/// UNBATCHED view of a batched actual against the callee's formal, so
/// it has a bare shape rather than a whole `RuntimeTensorValue`).
///
/// `name_dim_vars` selects the staging for `d-var` dims (`tensor[a, ..]`):
/// `false` stages them as concrete `Lit`s (the chelis#346 F5 decision —
/// correct wherever the same-rank formal/actual remap concretizes the
/// body's names); `true` stages them as `Named(name, Some(size))`, for
/// the vmap lane where the rank shift skips that remap and the body's
/// d-var names can only bind through the placeholder Load (chelis#351).
pub(super) fn declared_tensor_type_for_shape(
    ty_expr: &Expr,
    shape: &[usize],
    fallback_precision: Prim,
    name_dim_vars: bool,
) -> Result<TensorType, String> {
    let stripped = strip_type_wrappers(ty_expr);
    let Expr::Node(node, _) = stripped else {
        return Err("the static type is not a tensor type".to_string());
    };
    match Some(node.tag()) {
        Some(DeepTag::TTensor) => {}
        Some(DeepTag::TPrim) if shape.is_empty() => {
            return Ok(TensorType {
                dims: vec![],
                precision: extract_prim_from_type_expr(stripped).unwrap_or(fallback_precision),
            });
        }
        other => {
            return Err(format!(
                "the static type `{}` is not a tensor type",
                other.map(DeepTag::as_str).unwrap_or("?")
            ));
        }
    }
    let kids = node.children_slice();
    let Some((prim_expr, dim_exprs)) = kids.split_last() else {
        return Err("malformed t-tensor type (no children)".to_string());
    };
    let precision = extract_prim_from_type_expr(prim_expr).unwrap_or(fallback_precision);
    if dim_exprs.iter().any(|d| d.tag() == Some(DeepTag::DRank)) {
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
        let Some((dim_tag, dim_kids)) = tagged_expr_children(dim_expr) else {
            return Err("malformed tensor dimension in static type".to_string());
        };
        match Some(dim_tag) {
            Some(DeepTag::DName) => {
                let name = dim_kids
                    .first()
                    .and_then(symbol_name)
                    .ok_or_else(|| "malformed d-name dimension".to_string())?;
                dims.push(DimInfo::Named(name.to_string(), Some(size)));
            }
            Some(DeepTag::DLit) => {
                let lit = dim_kids
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
            // build call site binds it. In the same-rank lanes it carries
            // no anchor name, so a concrete Lit is the faithful staging
            // (chelis#346 red-team F5). In the vmap lane the rank shift
            // skips the formal/actual remap that would concretize the
            // body's d-var names, so the name must instead bind through
            // the placeholder Load — exactly like a d-name (chelis#351).
            Some(DeepTag::DVar) if name_dim_vars => {
                let name = dim_kids
                    .first()
                    .and_then(symbol_name)
                    .ok_or_else(|| "malformed d-var dimension".to_string())?;
                dims.push(DimInfo::Named(name.to_string(), Some(size)));
            }
            Some(DeepTag::DVar) => dims.push(DimInfo::Lit(size)),
            other => {
                return Err(format!(
                    "unsupported dimension form `{}` in static type",
                    other.map(DeepTag::as_str).unwrap_or("?")
                ));
            }
        }
    }
    Ok(TensorType { dims, precision })
}
