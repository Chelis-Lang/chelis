use std::collections::{HashMap, HashSet};

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_ir::dag::{DimInfo, TensorType};
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_ir::lower::try_lower_subexpr_program;
use chelis_types::types::Prim;

use super::named_axis::*;
use super::*;

impl<'a> EvalContext<'a> {
    /// Bucket 1 entry point: evaluate `(grad f)(args...)` /
    /// `(vmap f)(args...)` in the host runtime. Synthesizes a Deep
    /// `(app {} <transform-expr> (var __chelis_xform_arg_k))` form and
    /// routes it through `chelis_ir::lower::lower_subexpr_program` +
    /// `chelis_ir::eval::eval_tensor_*`. The IR pipeline already
    /// implements grad and vmap (it's what the C backend uses); we just
    /// reuse it instead of writing a parallel reverse-mode evaluator
    /// inside the host-runtime tree.
    pub(super) fn apply_transform(
        &mut self,
        kind: TransformKind,
        transform_expr: &Expr,
        captured_env: HashMap<String, RuntimeValue>,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, String> {
        // Allocate placeholder names for the call's actual arguments. We
        // synthesize `(var {type: ...} __chelis_xform_arg_K)` inside the
        // app form and feed the corresponding tensor values via the
        // load callback when forward-evaluating the lowered DAG.
        let mut placeholder_names: Vec<String> = Vec::with_capacity(args.len());
        let mut placeholder_types: Vec<TensorType> = Vec::with_capacity(args.len());
        let mut placeholder_tensors: HashMap<String, IrTensorValue> =
            HashMap::with_capacity(args.len());

        // Best-effort fn-expr lookup so we can read the inner
        // function's parameter type metadata. The transform_expr is the
        // captured `(grad ... fn-expr ...)` or `(vmap ... fn-expr
        // axis-lit)` form; the fn-expr is the first child.
        let fn_expr = match transform_expr {
            Expr::List(list, _) => children(list).first(),
            _ => None,
        };

        // chelis#351: in the vmap lane, marshalling the batched actuals
        // with bare Lit dims loses the callee's declared dim names. The
        // inlined body keeps its formal named dims (e.g. a Tier-3
        // rank-poly named reduce's surviving `hidden`), and vmap's rank
        // shift (batched actual = formal rank + 1) defeats the same-rank
        // formal/actual remap at the transform boundary — so the name
        // stays unbound, no Load declares it, and
        // `dag::symbolic_occurrences` ICEs. Type the placeholder from
        // the callee's declared formals instead (the chelis#338/#346
        // pattern for plain def calls): the vmap axis stays `Lit`, the
        // mapped axes carry the formal's names with runtime sizes, and
        // the symbolic-dim machinery binds the body's names against the
        // placeholder Load. Best-effort: any unresolved shape falls back
        // to the Lit-dim marshalling below.
        let vmap_formals = match kind {
            TransformKind::Vmap => {
                resolve_transform_fn_for_formals(transform_expr, &self.top_level_defs)
            }
            TransformKind::Grad => None,
        };

        for (index, value) in args.iter().enumerate() {
            let placeholder = format!("__chelis_xform_arg_{index}");
            let (tensor_value, mut tensor_type) =
                runtime_value_to_dag_input(value, fn_expr, index)?;
            if let (Some((callee_fn, Some(axis))), RuntimeValue::Tensor(tensor)) =
                (&vmap_formals, value)
                && let Some(formal) = param_type_expr_at(callee_fn, index)
                && let Ok(refined) = vmap_lane_placeholder_type(formal, tensor, *axis)
            {
                tensor_type = refined;
            }
            placeholder_tensors.insert(placeholder.clone(), tensor_value);
            placeholder_names.push(placeholder);
            placeholder_types.push(tensor_type);
        }

        // Synthesize `(app {} <transform-expr> (var __chelis_xform_arg_0) ...)`.
        let span = Span::new(0, 0);
        let mut app_elements: Vec<Expr> = Vec::with_capacity(2 + placeholder_names.len());
        app_elements.push(Expr::Atom(Atom::Symbol("app".to_string()), span));
        app_elements.push(Expr::Map(MetaMap::default(), span));
        app_elements.push(transform_expr.clone());
        for (placeholder, ty) in placeholder_names.iter().zip(placeholder_types.iter()) {
            app_elements.push(make_var_with_type(placeholder, ty, span));
        }
        let app_expr = Expr::List(
            List {
                elements: app_elements,
            },
            span,
        );

        let scoped_types: HashMap<String, TensorType> = placeholder_names
            .iter()
            .cloned()
            .zip(placeholder_types.iter().cloned())
            .collect();

        // Build a fresh `program_defs` that includes both top-level
        // defs from the host runtime AND any captured local closures
        // from `captured_env` (so `target = fn (...) -> ...; grad(target)(x)`
        // resolves `target` when the inner DAG lowering reaches it).
        let mut program_defs = self.top_level_defs.clone();
        for (name, value) in captured_env.iter() {
            if let RuntimeValue::Closure { params, body, .. } = value {
                program_defs
                    .entry(name.clone())
                    .or_insert_with(|| synth_fn_expr(params, body));
            }
        }

        // Fail-closed for host-runtime-only builtins reached through
        // grad/vmap. The IR lowerer doesn't recognize `tensor_scan`
        // (spec/05-risc-primitives.md §3.6 marks it host-only with no
        // adjoint), so passing it through `try_lower_subexpr_program`
        // results in confusing downstream errors like an out-of-range
        // axis on a rank-0 operand. Catch it here and emit a tensor_scan-
        // tagged error instead. The search starts at `app_expr` (for the
        // inline `grad(fn (x) -> tensor_scan(...))` case) and follows
        // every `(var ...)` reference transitively into `program_defs`
        // (for the captured-closure case `target = fn ... tensor_scan
        // ...; grad(target)(x)`). It is *reachability*-scoped: an
        // unrelated top-level def that calls `tensor_scan` but is not
        // reached from the transform target does NOT trigger a rejection,
        // so a genuinely differentiable program is not falsely blocked.
        let host_only_hit = find_reachable_host_only_builtin_call(&app_expr, &program_defs);
        if let Some(name) = host_only_hit {
            // Keep the verb honest per transform: `grad` differentiates,
            // `vmap` vectorizes. Both fail for the same root cause (no
            // RISC DAG lowering), but only `grad` additionally needs an
            // adjoint, so only its message mentions the missing adjoint.
            let (kind_label, verb, reason) = match kind {
                TransformKind::Grad => (
                    "grad",
                    "differentiate through",
                    "it has no RISC DAG lowering and no AD adjoint",
                ),
                TransformKind::Vmap => ("vmap", "vectorize over", "it has no RISC DAG lowering"),
            };
            return Err(format!(
                "host runtime: `{kind_label}(...)` cannot {verb} host-runtime-only \
                 builtin `{name}`; {reason} (see \
                 spec/05-risc-primitives.md §3.6 Host-Runtime Builders). Build the per-index \
                 accumulator with tensor-lane primitives (e.g. `range`/`map`/`expand`) before \
                 applying `{kind_label}`."
            ));
        }

        let lower_result =
            try_lower_subexpr_program(&app_expr, scoped_types, self.type_env.clone(), program_defs);
        let dag = match lower_result {
            Ok(dag) => dag,
            Err(diagnostic) => {
                let kind_label = match kind {
                    TransformKind::Grad => "grad",
                    TransformKind::Vmap => "vmap",
                };
                return Err(format!(
                    "host runtime could not lower `{kind_label}(...)` for evaluation: \
                     {diagnostic}"
                ));
            }
        };

        // Forward-evaluate the lowered DAG, satisfying `RiscOp::Load`
        // by looking up placeholder names in our staged inputs (or
        // tensor_bindings as a fallback for any external tensor refs
        // captured by the inner fn body).
        let tensor_bindings = self.tensor_bindings;
        let roots: Vec<chelis_ir::dag::NodeId> = dag.roots().to_vec();
        if roots.is_empty() {
            let kind_label = match kind {
                TransformKind::Grad => "grad",
                TransformKind::Vmap => "vmap",
            };
            return Err(format!(
                "host runtime: `{kind_label}(...)` lowering produced no roots"
            ));
        }
        let values = chelis_ir::eval::eval_tensor_roots_with_strict(&dag, &roots, |name| {
            placeholder_tensors
                .get(name)
                .cloned()
                .or_else(|| tensor_bindings.get(name).map(|t| t.value.clone()))
        })
        .map_err(|err| {
            let kind_label = match kind {
                TransformKind::Grad => "grad",
                TransformKind::Vmap => "vmap",
            };
            format!("host runtime `{kind_label}` evaluation failed: {err}")
        })?;

        // Pack roots back into a RuntimeValue.
        let kind_label = match kind {
            TransformKind::Grad => "grad",
            TransformKind::Vmap => "vmap",
        };
        pack_dag_roots(&dag, &roots, &values, kind_label)
    }
}

/// Bucket 1 helper: convert a host-runtime argument into a
/// `(TensorValue, TensorType)` pair the IR DAG can consume. Scalar args
/// (Int/Float/Bool) are wrapped as rank-0 tensors with the precision
/// pulled from the inner fn's parameter type metadata when available, or
/// from the runtime value as a fallback.
pub(super) fn runtime_value_to_dag_input(
    value: &RuntimeValue,
    fn_expr: Option<&Expr>,
    index: usize,
) -> Result<(IrTensorValue, TensorType), String> {
    match value {
        RuntimeValue::Tensor(tensor) => {
            let dims = tensor
                .value
                .shape
                .iter()
                .map(|&size| DimInfo::Lit(size))
                .collect::<Vec<_>>();
            let ty = TensorType {
                dims,
                precision: tensor.precision,
            };
            Ok((tensor.value.clone(), ty))
        }
        RuntimeValue::Scalar(payload) if payload.dtype().is_float() => {
            let precision = fn_expr
                .and_then(|e| param_precision_at(e, index))
                .unwrap_or(payload.dtype());
            Ok((
                IrTensorValue::scalar(payload.bits().as_f64()),
                TensorType {
                    dims: vec![],
                    precision,
                },
            ))
        }
        RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => {
            let precision = fn_expr
                .and_then(|e| param_precision_at(e, index))
                .unwrap_or(payload.dtype());
            Ok((
                IrTensorValue::scalar(payload.bits().as_i64() as f64),
                TensorType {
                    dims: vec![],
                    precision,
                },
            ))
        }
        RuntimeValue::Bool(value) => Ok((
            IrTensorValue::scalar(if *value { 1.0 } else { 0.0 }),
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
        )),
        other => Err(format!(
            "grad/vmap argument {index} must be a tensor or scalar, got {other:?}"
        )),
    }
}

/// chelis#351: resolve a transform target's underlying `(fn ...)`
/// expression — the source of its declared formal param types — plus
/// the vmap batching axis when the wrapper chain contains exactly one
/// `vmap`. Follows `(var name)` references through `defs` and descends
/// through `grad` wrappers (grad does not change argument shapes), so
/// `vmap(inner)`, `vmap(grad(total))`, and alias chains all resolve.
/// Returns `None` for anything else — closures injected from
/// `captured_env` carry no declared param types, nested `vmap` adds a
/// second batch axis this synthesis does not model, and an unreadable
/// axis literal must not be guessed at. The caller then falls back to
/// Lit-dim placeholder marshalling (the pre-#351 behavior).
fn resolve_transform_fn_for_formals<'a>(
    expr: &'a Expr,
    defs: &'a HashMap<String, Expr>,
) -> Option<(&'a Expr, Option<usize>)> {
    let mut current = expr;
    let mut vmap_axis: Option<usize> = None;
    let mut visited: HashSet<&str> = HashSet::new();
    loop {
        let Expr::List(list, _) = current else {
            return None;
        };
        match tag(list) {
            Some("fn") => return Some((current, vmap_axis)),
            Some("var") => {
                let name = children(list).first().and_then(symbol_name)?;
                if !visited.insert(name) {
                    return None;
                }
                current = defs.get(name)?;
            }
            Some("grad") => {
                current = children(list).first()?;
            }
            Some("vmap") => {
                if vmap_axis.is_some() {
                    return None;
                }
                let kids = children(list);
                let axis = match kids.get(1) {
                    Some(axis_expr) => static_usize_value(axis_expr)?,
                    None => 0,
                };
                vmap_axis = Some(axis);
                current = kids.first()?;
            }
            _ => return None,
        }
    }
}

/// Static non-negative int literal: a bare int atom, `(lit {} n)`, or a
/// `cast(n, int32)` wrapper (mirrors the lowerer's
/// `extract_usize_value` shapes for the vmap axis argument).
fn static_usize_value(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Atom(Atom::Int(n), _) => usize::try_from(*n).ok(),
        Expr::List(list, _) => match tag(list)? {
            "lit" => match children(list).first()? {
                Expr::Atom(Atom::Int(n), _) => usize::try_from(*n).ok(),
                _ => None,
            },
            "cast" => static_usize_value(children(list).first()?),
            _ => None,
        },
        _ => None,
    }
}

/// chelis#351: type a vmap-lane tensor placeholder from the callee's
/// declared formal. The batched actual carries one extra axis at
/// `axis` (the vmap axis); that axis stays `Lit` (it is the mapped
/// axis, not one of the callee's dims), and the remaining axes are
/// typed against the formal exactly as the chelis#338 def-call
/// boundary types its placeholders — except that `d-var` dims are
/// staged as `Named(name, Some(size))` rather than `Lit`: the vmap
/// rank shift skips the same-rank remap that concretizes the body's
/// d-var names in the plain-call/grad lanes, so the names can only
/// bind through the placeholder Load. Errs (caller falls back to Lit
/// dims) when the axis is out of range or the formal does not type the
/// unbatched view — e.g. a broadcast argument the lowering passes
/// through unbatched, or a rank-poly (`..spread`) formal.
fn vmap_lane_placeholder_type(
    formal: &Expr,
    tensor: &RuntimeTensorValue,
    axis: usize,
) -> Result<TensorType, String> {
    let shape = &tensor.value.shape;
    if axis >= shape.len() {
        return Err(format!(
            "vmap axis {axis} is out of range for a rank-{} actual",
            shape.len()
        ));
    }
    let mut unbatched = shape.clone();
    let batch = unbatched.remove(axis);
    let mut ty = declared_tensor_type_for_shape(formal, &unbatched, tensor.precision, true)?;
    ty.dims.insert(axis, DimInfo::Lit(batch));
    Ok(ty)
}

/// The declared `{type: ...}` metadata on a single `(params ...)` child
/// (a `(x {type: T})` list or a MetaExpr-wrapped symbol).
pub(super) fn param_decl_type_expr(param: &Expr) -> Option<&Expr> {
    match param {
        Expr::List(param_list, _) => match param_list.elements.get(1) {
            Some(Expr::Map(meta, _)) => meta
                .entries
                .iter()
                .find(|(key, _)| key == "type")
                .map(|(_, value)| value),
            _ => None,
        },
        Expr::MetaExpr(meta, _) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        _ => None,
    }
}

/// `(fn ...)` param[index]'s declared type expression, if annotated.
pub(super) fn param_type_expr_at(fn_expr: &Expr, index: usize) -> Option<&Expr> {
    let Expr::List(list, _) = fn_expr else {
        return None;
    };
    if tag(list) != Some("fn") {
        return None;
    }
    let params = children(list).first()?;
    let Expr::List(params_list, _) = params else {
        return None;
    };
    param_decl_type_expr(children(params_list).get(index)?)
}

/// Best-effort lookup of `(fn ...)` param[index]'s primitive precision
/// from its `type` metadata.
fn param_precision_at(fn_expr: &Expr, index: usize) -> Option<Prim> {
    extract_prim_from_type_expr(param_type_expr_at(fn_expr, index)?)
}

pub(super) fn extract_prim_from_type_expr(expr: &Expr) -> Option<Prim> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    match tag(list) {
        Some("t-prim") => children(list)
            .first()
            .and_then(symbol_name)
            .and_then(prim_from_name),
        Some("t-tensor") => children(list).last().and_then(extract_prim_from_type_expr),
        _ => None,
    }
}

pub(super) fn prim_from_name(name: &str) -> Option<Prim> {
    Some(match name {
        "f32" => Prim::F32,
        "f64" => Prim::F64,
        "f16" => Prim::F16,
        "bf16" => Prim::Bf16,
        // E2 (WS-A0 RT-1 fixup): per spec/04-type-system.md §1.1.1
        // f8e4m3 is deferred and the type checker rejects every cast
        // and tensor-element use upstream. If the host runtime ever
        // resolves an `f8e4m3` token here, the upstream rejection has
        // a hole — panic loudly rather than carrying the deferred
        // dtype into runtime classification.
        "f8e4m3" => panic!(
            "f8e4m3 is deferred per spec/04-type-system.md §1.1.1 and \
             should have been rejected upstream"
        ),
        "int8" => Prim::Int8,
        "int16" => Prim::Int16,
        "int32" => Prim::Int32,
        "int64" => Prim::Int64,
        "bool" => Prim::Bool,
        "string" => Prim::String,
        _ => return None,
    })
}

/// Build a `(var {type: <encoded ty>} name)` Deep expression from a
/// `TensorType`. Used when synthesizing the placeholder argument refs
/// inside the host runtime's grad/vmap wrapper app.
pub(super) fn make_var_with_type(name: &str, ty: &TensorType, span: Span) -> Expr {
    let prim_name = match ty.precision {
        Prim::F32 => "f32",
        Prim::F64 => "f64",
        Prim::F16 => "f16",
        Prim::Bf16 => "bf16",
        // E2 (WS-A0 RT-1 fixup, sibling sweep): per
        // spec/04-type-system.md §1.1.1 f8e4m3 is deferred and the
        // type checker rejects it upstream. If a TensorType reaches
        // this Deep re-encoder with f8e4m3 precision, the upstream
        // rejection has a hole — panic rather than emit a
        // `(t-prim {} f8e4m3)` node into a synthesized Deep var.
        Prim::F8e4m3 => panic!(
            "f8e4m3 is deferred per spec/04-type-system.md §1.1.1 and \
             should have been rejected upstream"
        ),
        Prim::Int8 => "int8",
        Prim::Int16 => "int16",
        Prim::Int32 => "int32",
        Prim::Int64 => "int64",
        Prim::Bool => "bool",
        Prim::String => "string",
    };
    let prim_node = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("t-prim".to_string()), span),
                Expr::Map(MetaMap::default(), span),
                Expr::Atom(Atom::Symbol(prim_name.to_string()), span),
            ],
        },
        span,
    );
    let ty_expr = if ty.dims.is_empty() {
        prim_node
    } else {
        let mut tensor_elems = vec![
            Expr::Atom(Atom::Symbol("t-tensor".to_string()), span),
            Expr::Map(MetaMap::default(), span),
        ];
        for dim in &ty.dims {
            tensor_elems.push(dim_to_expr(dim, span));
        }
        tensor_elems.push(prim_node);
        Expr::List(
            List {
                elements: tensor_elems,
            },
            span,
        )
    };
    let mut meta = MetaMap::default();
    meta.entries.push(("type".to_string(), ty_expr));
    Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("var".to_string()), span),
                Expr::Map(meta, span),
                Expr::Atom(Atom::Symbol(name.to_string()), span),
            ],
        },
        span,
    )
}

fn dim_to_expr(dim: &DimInfo, span: Span) -> Expr {
    match dim {
        DimInfo::Lit(value) => Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Symbol("d-lit".to_string()), span),
                    Expr::Map(MetaMap::default(), span),
                    Expr::Atom(Atom::Int(*value as i64), span),
                ],
            },
            span,
        ),
        DimInfo::Named(name, _) => Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Symbol("d-name".to_string()), span),
                    Expr::Map(MetaMap::default(), span),
                    Expr::Atom(Atom::Symbol(name.clone()), span),
                ],
            },
            span,
        ),
    }
}

/// Synthesize a `(fn {} (params {} <p>...) <body>)` Deep expression
/// from a host-runtime closure's params + body. Used when injecting
/// captured local closures into the IR `program_defs` table.
fn synth_fn_expr(params: &[String], body: &Expr) -> Expr {
    let span = body.span();
    let param_exprs = params
        .iter()
        .map(|name| Expr::Atom(Atom::Symbol(name.clone()), span))
        .collect::<Vec<_>>();
    let mut params_elements = vec![
        Expr::Atom(Atom::Symbol("params".to_string()), span),
        Expr::Map(MetaMap::default(), span),
    ];
    params_elements.extend(param_exprs);
    let params_list = Expr::List(
        List {
            elements: params_elements,
        },
        span,
    );
    Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("fn".to_string()), span),
                Expr::Map(MetaMap::default(), span),
                params_list,
                body.clone(),
            ],
        },
        span,
    )
}

pub(super) fn var_name(expr: &Expr) -> Option<&str> {
    let list = as_list(expr)?;
    if tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

pub(super) fn runtime_param_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        Expr::MetaExpr(meta, _) => runtime_param_name(&meta.expr),
        Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .or_else(|| children(list).first().and_then(symbol_name)),
        _ => None,
    }
}

pub(super) fn as_list(expr: &Expr) -> Option<&List> {
    match expr {
        Expr::List(list, _) => Some(list),
        _ => None,
    }
}

/// Host-only builtins that have no RISC DAG lowering. A `grad(...)`
/// or `vmap(...)` over a function that calls one of these must fail
/// with a clear, tagged error rather than be passed through to
/// `try_lower_subexpr_program` and produce a confusing downstream
/// error like an out-of-range axis on a phantom rank-0 operand. See
/// spec/05-risc-primitives.md §3.6.
const HOST_ONLY_BUILTIN_NAMES: &[&str] = &["tensor_scan"];

/// Walk a Deep `Expr` collecting (a) the first directly-applied
/// host-only builtin (`(app {} (var <name>) ...)`) and (b) the names
/// of every `(var <name>)` it references, so a reachability walk can
/// follow those names into def bodies. The `vars` set lets the caller
/// resolve the captured-closure case (`target = fn ... tensor_scan ...;
/// grad(target)(x)`) without flagging *unrelated* top-level defs that
/// happen to call `tensor_scan` but are not reachable from the
/// transform target (which would be a false-positive rejection of a
/// perfectly differentiable program — see issue #257 review round 2).
fn scan_expr_for_host_only(expr: &Expr, hit: &mut Option<String>, vars: &mut Vec<String>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    if tag(list) == Some("app")
        && let Some(Expr::List(callee, _)) = children(list).first()
        && tag(callee) == Some("var")
        && let Some(name) = children(callee).first().and_then(symbol_name)
        && HOST_ONLY_BUILTIN_NAMES.contains(&name)
    {
        if hit.is_none() {
            *hit = Some(name.to_string());
        }
        return;
    }
    if tag(list) == Some("var")
        && let Some(name) = children(list).first().and_then(symbol_name)
    {
        vars.push(name.to_string());
    }
    for child in &list.elements {
        scan_expr_for_host_only(child, hit, vars);
    }
}

/// Reachability-scoped search for a host-only builtin call. Starts at
/// `root` (the synthesized `(app {} <transform> <args>...)`), then
/// follows every `(var <name>)` reference transitively into the bodies
/// of `defs` so the transform target's own def — and any helper it
/// calls — is searched, but unrelated top-level defs are not. Returns
/// the name of the first host-only builtin reached, or `None`.
///
/// Known, accepted limitation (issue #257 review item 6): detection
/// matches only a *direct application by name*, `(app (var tensor_scan)
/// ...)`, and only follows references that resolve to a top-level `defs`
/// entry. Two exotic aliasing forms therefore slip past — a `let`-bound
/// alias (`let f = tensor_scan in f(acc, cb, n)`, where `f` is a local
/// binding rather than a `defs` key and the call site `(app (var f)
/// ...)` does not name a host-only builtin), and `tensor_scan` passed as
/// an un-applied value into a higher-order helper whose own body applies
/// it. Both fail *soft*: the transform then reaches
/// `try_lower_subexpr_program`, which rejects the un-lowerable builtin
/// anyway, so the user still gets an error — just the older, less
/// specific one rather than the §3.6-tagged message. The failure mode
/// is message quality in an aliasing corner, never a wrong gradient or
/// a silently-lowered host-only op, so it is left as-is.
pub(super) fn find_reachable_host_only_builtin_call(
    root: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<String> {
    let mut hit: Option<String> = None;
    let mut visited: HashSet<String> = HashSet::new();
    let mut worklist: Vec<&Expr> = vec![root];
    while let Some(expr) = worklist.pop() {
        let mut vars: Vec<String> = Vec::new();
        scan_expr_for_host_only(expr, &mut hit, &mut vars);
        if hit.is_some() {
            return hit;
        }
        for name in vars {
            if visited.insert(name.clone())
                && let Some(def_body) = defs.get(&name)
            {
                worklist.push(def_body);
            }
        }
    }
    hit
}
