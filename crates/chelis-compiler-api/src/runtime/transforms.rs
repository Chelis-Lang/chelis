use chelis_deep::DeepTag;
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
        // Per-argument Deep expression to splice into the synthesized app:
        // a typed placeholder var for tensor/scalar args, or a synthesized
        // ADT construction over per-field placeholders (chelis#520 D2).
        let mut arg_exprs: Vec<Expr> = Vec::with_capacity(args.len());
        // chelis#520 D2: per-differentiated-target repack plan. After
        // evaluation the gradient roots come back as one FLAT tuple (the
        // pytree of every wrt-selected target's fields, concatenated in
        // parameter order); each plan slot says how many of those flat
        // roots the target owns and what structure to fold them back into
        // (a bare tensor, or a constructor-shaped `Adt`). This is the
        // eval-lane twin of the IR lowering's `GradResultPlan` list. Only
        // wrt-selected differentiable targets get a slot; a non-selected
        // or non-differentiable argument still marshals its placeholders
        // (the body may read it) but owns no gradient root.
        enum ArgRepack {
            Tensor,
            Adt {
                ctor: String,
                field_names: Option<Vec<String>>,
                field_count: usize,
            },
        }
        let mut arg_repacks: Vec<ArgRepack> = Vec::with_capacity(args.len());
        // wrt indices for this grad call, if narrowed (`grad(f, wrt=(i))`).
        // `None` means differentiate every differentiable argument, exactly
        // as the checker's `grad_result_type` and the IR lowering's
        // `is_selected_wrt` do.
        let grad_wrt = match kind {
            TransformKind::Grad => grad_wrt_indices_from_transform(transform_expr),
            TransformKind::Vmap => None,
        };

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

        let span = Span::new(0, 0);
        for (index, value) in args.iter().enumerate() {
            // chelis#520 D2: an ADT-valued grad argument marshals as one
            // placeholder per field plus a synthesized construction expr,
            // so the IR lowering sees the static constructor shape and
            // differentiates field-wise.
            if let (
                TransformKind::Grad,
                RuntimeValue::Adt {
                    ctor,
                    fields,
                    field_names,
                },
            ) = (&kind, value)
            {
                // Type-level gate: the checker types `grad` over an ADT
                // as non-differentiable (unit payload) when ANY variant
                // of the type carries a non-float field, or when the
                // type is a pure enum with no fields. The constructed
                // value's own fields may look float-clean (e.g. the
                // clean variant of a mixed sum type), but producing a
                // gradient struct here would contradict the static type.
                // Reject with the reason recorded at context build.
                if let Some(reason) = self.adt_grad_rejections.get(ctor) {
                    return Err(format!(
                        "host runtime: `grad(...)` argument {index}: {reason} \
                         (chelis#520 D2)"
                    ));
                }
                // Field names aligned with `fields` order. `fields` is in
                // DECLARED order (`eval_record` reorders by the deftype
                // table), so prefer the authoritative `adt_fields` entry;
                // the value's own `field_names` is the fallback for Adt
                // values built outside `eval_record`.
                let aligned_names: Option<Vec<String>> = self
                    .adt_fields
                    .get(ctor)
                    .cloned()
                    .filter(|names| names.len() == fields.len())
                    .or_else(|| {
                        field_names
                            .clone()
                            .filter(|names| names.len() == fields.len())
                    });
                let mut field_placeholders: Vec<(String, TensorType)> =
                    Vec::with_capacity(fields.len());
                for (fidx, field_value) in fields.iter().enumerate() {
                    let field_label = aligned_names
                        .as_ref()
                        .and_then(|names| names.get(fidx).cloned())
                        .unwrap_or_else(|| fidx.to_string());
                    let convertible = matches!(
                        field_value,
                        RuntimeValue::Tensor(_) | RuntimeValue::Scalar(_)
                    );
                    if !convertible {
                        return Err(format!(
                            "host runtime: `grad(...)` argument {index}: field \
                             `{field_label}` of constructor `{ctor}` is not a tensor or \
                             scalar; only ADT values whose fields are all float tensors \
                             can be differentiated (chelis#520 D2)"
                        ));
                    }
                    let (tensor_value, tensor_type) =
                        runtime_value_to_dag_input(field_value, None, index)?;
                    let placeholder = format!("__chelis_xform_arg_{index}_field_{fidx}");
                    placeholder_tensors.insert(placeholder.clone(), tensor_value);
                    placeholder_names.push(placeholder.clone());
                    placeholder_types.push(tensor_type.clone());
                    field_placeholders.push((placeholder, tensor_type));
                }
                arg_exprs.push(make_adt_construction_expr(
                    ctor,
                    aligned_names.as_deref(),
                    &field_placeholders,
                    span,
                ));
                // A clean (all-float-field) ADT argument is a differentiated
                // target whenever `wrt` selects it (or `wrt` is the default,
                // all-args form). The rejection above guarantees the type is
                // differentiable, so a wrt-selected slot always owns exactly
                // `fields.len()` gradient roots (the IR lowering zero-fills a
                // field with no adjoint, so every field is represented).
                let selected = grad_wrt
                    .as_ref()
                    .is_none_or(|indices| indices.contains(&index));
                if selected {
                    arg_repacks.push(ArgRepack::Adt {
                        ctor: ctor.clone(),
                        field_names: aligned_names,
                        field_count: fields.len(),
                    });
                }
                continue;
            }
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
            arg_exprs.push(make_var_with_type(&placeholder, &tensor_type, span));
            placeholder_tensors.insert(placeholder.clone(), tensor_value);
            placeholder_names.push(placeholder);
            placeholder_types.push(tensor_type.clone());
            // chelis#520 D2: a wrt-selected float tensor/scalar argument owns
            // one gradient root, packed back as a bare tensor. A non-float or
            // non-selected argument owns none (matching the IR lowering's
            // `is_selected_wrt`), so it gets no repack slot even though its
            // placeholder is still marshalled (the body may read it).
            if matches!(kind, TransformKind::Grad) {
                let differentiable = tensor_type.precision.is_float();
                let selected = differentiable
                    && grad_wrt
                        .as_ref()
                        .is_none_or(|indices| indices.contains(&index));
                if selected {
                    arg_repacks.push(ArgRepack::Tensor);
                }
            }
        }

        // Synthesize `(app {} <transform-expr> <arg-expr_0> ...)`.
        let mut app_elements: Vec<Expr> = Vec::with_capacity(3 + arg_exprs.len());
        app_elements.push(Expr::Atom(Atom::Tag(DeepTag::App), span));
        app_elements.push(Expr::Map(MetaMap::default(), span));
        app_elements.push(transform_expr.clone());
        app_elements.extend(arg_exprs);
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
        // chelis#377: a transform target may capture a top-level tensor
        // binding (`w = to_tensor([...]); def f(x) = sum(mul(x, w), 0);
        // grad(f)(x)`). The captured `w` lowers to a `Load { name: "w" }`
        // in the inner DAG. It is delivered in `captured_env` (NOT
        // `tensor_bindings`, which only carries the host program's own
        // top-level tensors), so the load callback must also serve captured
        // tensor values or the grad/vmap lane fails with `missing required
        // input \`w\``. The C backend already compiles+runs this (it hoists
        // the capture into a file-scope global); this closes the eval-side
        // parity gap pinned by `issue_352_grad_over_capturing_def_eval_gap`.
        // Only `Tensor` captures are served; non-tensor captures (closures,
        // scalars routed elsewhere) are not load inputs here.
        let mut captured_tensors: HashMap<String, IrTensorValue> = captured_env
            .iter()
            .filter_map(|(name, value)| match value {
                RuntimeValue::Tensor(tensor) => Some((name.clone(), tensor.value.clone())),
                _ => None,
            })
            .collect();

        // chelis#377 (vmap-inside-a-def): when the transform is applied
        // inside another def's body (`def fv(xs) = xs |> vmap(dot_w)`), the
        // `captured_env` is that body's local scope (`xs`), so a TOP-LEVEL
        // tensor binding the inner fn captures (`dot_w` referencing top-level
        // `w`) is in neither `captured_env` nor `tensor_bindings`. Walk the
        // lowered DAG's still-unsatisfied `Load` names and resolve each as a
        // top-level binding, so the captured `w` Load is served. This reuses
        // the same `resolve_top_level` path a plain reference would take.
        for node in dag.nodes() {
            let chelis_ir::dag::RiscOp::Load { name } = &node.op else {
                continue;
            };
            let name = name.as_str();
            if placeholder_tensors.contains_key(name)
                || tensor_bindings.contains_key(name)
                || captured_tensors.contains_key(name)
            {
                continue;
            }
            if let Ok(RuntimeValue::Tensor(tensor)) = self.resolve_top_level(name) {
                captured_tensors.insert(name.to_string(), tensor.value.clone());
            }
        }
        // chelis#377: a served capture's value must match the rank its `Load`
        // node was typed with. The vmap lane prepends the batch axis to a
        // captured binding's `Load` (typing top-level `w` as `[batch, ..]`)
        // while the served value keeps its declared rank (`[..]`). Chelis has
        // no implicit broadcasting, so that rank mismatch is unsatisfiable:
        // before this guard it reached an elementwise op and PANICKED the
        // evaluator's shape assertion (`binary_map` left:[batch,..] right:[..]).
        // Reject it here with a clean diagnostic instead. Correct
        // vmap-with-captures must BROADCAST the capture across the batch axis,
        // not batch it — the remaining tracked residual (chelis#377). The grad
        // lane is unaffected: a captured binding's `Load` keeps its declared
        // rank there, so the ranks match and this never fires.
        for node in dag.nodes() {
            let chelis_ir::dag::RiscOp::Load { name } = &node.op else {
                continue;
            };
            if let Some(value) = captured_tensors.get(name.as_str())
                && value.shape.len() != node.output_type.dims.len()
            {
                let kind_label = match kind {
                    TransformKind::Grad => "grad",
                    TransformKind::Vmap => "vmap",
                };
                return Err(format!(
                    "host runtime: `{kind_label}(...)` over a def capturing top-level \
                     binding `{name}` is unsupported: the transform types the capture as \
                     rank {} (batched) but the binding is rank {}. vmap-with-captures must \
                     broadcast the capture across the batch axis, not batch it (tracked \
                     residual, chelis#377).",
                    node.output_type.dims.len(),
                    value.shape.len(),
                ));
            }
        }
        let values = chelis_ir::eval::eval_tensor_roots_with_strict(&dag, &roots, |name| {
            placeholder_tensors
                .get(name)
                .cloned()
                .or_else(|| tensor_bindings.get(name).map(|t| t.value.clone()))
                .or_else(|| captured_tensors.get(name).cloned())
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
        let packed = pack_dag_roots(&dag, &roots, &values, kind_label)?;
        // chelis#520 D2: when at least one differentiated target is an ADT,
        // re-collapse the FLAT gradient roots into the per-argument pytree
        // structure (the gradient of a `Box`-shaped argument is a
        // `Box`-shaped value; a multi-target result is a tuple whose ADT
        // slot is a field-wise gradient struct and whose tensor slot is the
        // bare gradient). A pure-tensor grad needs no re-collapse: `packed`
        // is already the flat tuple the pre-#520 contract specifies, and
        // the eval-root display (chelis#614) walks it component-wise.
        let has_adt_slot = arg_repacks
            .iter()
            .any(|slot| matches!(slot, ArgRepack::Adt { .. }));
        if matches!(kind, TransformKind::Grad) && has_adt_slot {
            let flat: Vec<RuntimeValue> = match packed {
                RuntimeValue::Tuple(items) => items,
                single => vec![single],
            };
            // Every ADT slot always owns exactly `field_count` roots and
            // every tensor slot owns one, because the IR lowering
            // (`GradResultPlan`) zero-fills BOTH adjoint-free ADT fields and,
            // in a multi-target result, adjoint-free tensor slots. So for a
            // well-formed program `flat.len()` always equals `expected` and
            // this guard does not fire. It is retained as a defensive
            // internal-consistency tripwire: if a future lowering change ever
            // re-drops a slot, refuse to guess the per-slot boundaries and
            // fail loudly rather than pack a mislabeled gradient.
            let expected: usize = arg_repacks
                .iter()
                .map(|slot| match slot {
                    ArgRepack::Tensor => 1,
                    ArgRepack::Adt { field_count, .. } => *field_count,
                })
                .sum();
            if flat.len() != expected {
                return Err(format!(
                    "host runtime: `grad(...)` produced {} gradient roots for a \
                     structure expecting {expected}; refusing to pack a misaligned \
                     gradient (internal invariant: the IR lowering should have \
                     zero-filled every adjoint-free slot) (chelis#520 D2)",
                    flat.len()
                ));
            }
            let mut flat_iter = flat.into_iter();
            let mut slots: Vec<RuntimeValue> = Vec::with_capacity(arg_repacks.len());
            for slot in &arg_repacks {
                match slot {
                    ArgRepack::Tensor => {
                        slots.push(flat_iter.next().expect("count checked above"));
                    }
                    ArgRepack::Adt {
                        ctor,
                        field_names,
                        field_count,
                    } => {
                        let mut fields = Vec::with_capacity(*field_count);
                        for _ in 0..*field_count {
                            fields.push(flat_iter.next().expect("count checked above"));
                        }
                        slots.push(RuntimeValue::Adt {
                            ctor: ctor.clone(),
                            fields,
                            field_names: field_names.clone(),
                        });
                    }
                }
            }
            return Ok(match slots.len() {
                // A single differentiated target (one ADT, possibly with
                // other non-selected args present) returns the bare
                // gradient value, not a one-element tuple.
                1 => slots.into_iter().next().expect("non-empty"),
                _ => RuntimeValue::Tuple(slots),
            });
        }
        Ok(packed)
    }
}

/// chelis#520 D2: the `wrt` parameter indices of a direct `(grad {} fn
/// wrt?)` transform form, or `None` for the default all-arguments grad.
/// `wrt` is the optional second child: a `(tuple {} i ...)` of indices or
/// a single index literal (`cast`-wrapped ints are peeled by
/// `static_usize_value`). A non-`grad` head or an unreadable index yields
/// `None`, so the caller falls back to the differentiate-all default.
fn grad_wrt_indices_from_transform(transform_expr: &Expr) -> Option<Vec<usize>> {
    let list = as_list(transform_expr)?;
    if tag(list) != Some(DeepTag::Grad) {
        return None;
    }
    let wrt_expr = children(list).get(1)?;
    if let Expr::List(tuple, _) = wrt_expr
        && tag(tuple) == Some(DeepTag::Tuple)
    {
        return Some(
            children(tuple)
                .iter()
                .filter_map(static_usize_value)
                .collect(),
        );
    }
    static_usize_value(wrt_expr).map(|index| vec![index])
}

/// chelis#520 D2: synthesize the Deep construction expression that
/// rebuilds an ADT argument from its per-field typed placeholders --
/// `(record {} Ctor (kv {} field (var {type: ...} ph)) ...)` for record
/// constructors, `(app {} (var Ctor) (var {type: ...} ph) ...)` for
/// positional ones. The IR lowering resolves either form to a static
/// `LoweredValue::Adt`, which is what lets the differentiated body's
/// `match` destructuring resolve at lowering time.
fn make_adt_construction_expr(
    ctor: &str,
    field_names: Option<&[String]>,
    field_placeholders: &[(String, TensorType)],
    span: Span,
) -> Expr {
    match field_names {
        Some(names) => {
            let mut elements = vec![
                Expr::Atom(Atom::Tag(DeepTag::Record), span),
                Expr::Map(MetaMap::default(), span),
                Expr::Atom(Atom::Name(ctor.to_string()), span),
            ];
            for (name, (placeholder, ty)) in names.iter().zip(field_placeholders.iter()) {
                elements.push(Expr::List(
                    List {
                        elements: vec![
                            Expr::Atom(Atom::Tag(DeepTag::Kv), span),
                            Expr::Map(MetaMap::default(), span),
                            Expr::Atom(Atom::Name(name.clone()), span),
                            make_var_with_type(placeholder, ty, span),
                        ],
                    },
                    span,
                ));
            }
            Expr::List(List { elements }, span)
        }
        None => {
            let mut elements = vec![
                Expr::Atom(Atom::Tag(DeepTag::App), span),
                Expr::Map(MetaMap::default(), span),
                Expr::List(
                    List {
                        elements: vec![
                            Expr::Atom(Atom::Tag(DeepTag::Var), span),
                            Expr::Map(MetaMap::default(), span),
                            Expr::Atom(Atom::Name(ctor.to_string()), span),
                        ],
                    },
                    span,
                ),
            ];
            for (placeholder, ty) in field_placeholders {
                elements.push(make_var_with_type(placeholder, ty, span));
            }
            Expr::List(List { elements }, span)
        }
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
            Some(DeepTag::Fn) => return Some((current, vmap_axis)),
            Some(DeepTag::Var) => {
                let name = children(list).first().and_then(symbol_name)?;
                if !visited.insert(name) {
                    return None;
                }
                current = defs.get(name)?;
            }
            Some(DeepTag::Grad) => {
                current = children(list).first()?;
            }
            Some(DeepTag::Vmap) => {
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
            DeepTag::Lit => match children(list).first()? {
                Expr::Atom(Atom::Int(n), _) => usize::try_from(*n).ok(),
                _ => None,
            },
            DeepTag::Cast => static_usize_value(children(list).first()?),
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
    if tag(list) != Some(DeepTag::Fn) {
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
    let (node_tag, kids) = match expr {
        Expr::List(list, _) => (tag(list)?, children(list)),
        Expr::Node(node, _) => (node.tag(), node.children_slice()),
        _ => return None,
    };
    match node_tag {
        DeepTag::TPrim => kids.first().and_then(symbol_name).and_then(prim_from_name),
        DeepTag::TTensor => kids.last().and_then(extract_prim_from_type_expr),
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
                Expr::Atom(Atom::Tag(DeepTag::TPrim), span),
                Expr::Map(MetaMap::default(), span),
                Expr::Atom(Atom::Name(prim_name.to_string()), span),
            ],
        },
        span,
    );
    let ty_expr = if ty.dims.is_empty() {
        prim_node
    } else {
        let mut tensor_elems = vec![
            Expr::Atom(Atom::Tag(DeepTag::TTensor), span),
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
                Expr::Atom(Atom::Tag(DeepTag::Var), span),
                Expr::Map(meta, span),
                Expr::Atom(Atom::Name(name.to_string()), span),
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
                    Expr::Atom(Atom::Tag(DeepTag::DLit), span),
                    Expr::Map(MetaMap::default(), span),
                    Expr::Atom(Atom::Int(*value as i64), span),
                ],
            },
            span,
        ),
        DimInfo::Named(name, _) => Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Tag(DeepTag::DName), span),
                    Expr::Map(MetaMap::default(), span),
                    Expr::Atom(Atom::Name(name.clone()), span),
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
        .map(|name| Expr::Atom(Atom::Name(name.clone()), span))
        .collect::<Vec<_>>();
    let mut params_elements = vec![
        Expr::Atom(Atom::Tag(DeepTag::Params), span),
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
                Expr::Atom(Atom::Tag(DeepTag::Fn), span),
                Expr::Map(MetaMap::default(), span),
                params_list,
                body.clone(),
            ],
        },
        span,
    )
}

pub(super) fn var_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::List(list, _) if tag(list) == Some(DeepTag::Var) => {
            children(list).first().and_then(symbol_name)
        }
        Expr::Node(node, _) if node.tag() == DeepTag::Var => {
            node.children_slice().first().and_then(symbol_name)
        }
        _ => None,
    }
}

pub(super) fn runtime_param_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
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
    if tag(list) == Some(DeepTag::App)
        && let Some(Expr::List(callee, _)) = children(list).first()
        && tag(callee) == Some(DeepTag::Var)
        && let Some(name) = children(callee).first().and_then(symbol_name)
        && HOST_ONLY_BUILTIN_NAMES.contains(&name)
    {
        if hit.is_none() {
            *hit = Some(name.to_string());
        }
        return;
    }
    if tag(list) == Some(DeepTag::Var)
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
