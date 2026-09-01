use chelis_deep::DeepTag;
use chelis_unord::{UnordMap, UnordSet};

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_ir::dag::{DimInfo, TensorType};
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_ir::lower::try_lower_subexpr_program_with_random_state_progress;
use chelis_types::types::{NominalArg, Prim, TensorPrec, Type, TypeVar};

use super::host_ops::terminal_name_matches;
use super::named_axis::*;
use super::*;

#[derive(Clone)]
enum GradListShape {
    Leaf,
    Unit,
    List(Vec<GradListShape>),
    Tuple(Vec<GradListShape>),
    Adt {
        ctor: String,
        field_names: Option<Vec<String>>,
        fields: Vec<GradListShape>,
    },
}

impl GradListShape {
    fn leaf_count(&self) -> usize {
        match self {
            Self::Leaf => 1,
            Self::Unit => 0,
            Self::List(items) | Self::Tuple(items) => items.iter().map(Self::leaf_count).sum(),
            Self::Adt { fields, .. } => fields.iter().map(Self::leaf_count).sum(),
        }
    }

    fn repack(&self, leaves: &mut impl Iterator<Item = RuntimeValue>) -> RuntimeValue {
        match self {
            Self::Leaf => leaves.next().expect("gradient leaf count checked above"),
            Self::Unit => RuntimeValue::Unit,
            Self::List(items) => {
                RuntimeValue::List(items.iter().map(|item| item.repack(leaves)).collect())
            }
            Self::Tuple(items) => {
                RuntimeValue::Tuple(items.iter().map(|item| item.repack(leaves)).collect())
            }
            Self::Adt {
                ctor,
                field_names,
                fields,
            } => RuntimeValue::Adt {
                ctor: ctor.clone(),
                fields: fields.iter().map(|field| field.repack(leaves)).collect(),
                field_names: field_names.clone(),
            },
        }
    }
}

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
        captured_env: UnordMap<String, RuntimeValue>,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, String> {
        // Allocate placeholder names for the call's actual arguments. We
        // synthesize `(var {type: ...} __chelis_xform_arg_K)` inside the
        // app form and feed the corresponding tensor values via the
        // load callback when forward-evaluating the lowered DAG.
        let mut placeholder_names: Vec<String> = Vec::with_capacity(args.len());
        let mut placeholder_types: Vec<TensorType> = Vec::with_capacity(args.len());
        let mut placeholder_tensors: UnordMap<String, IrTensorValue> = UnordMap::new();
        // Per-argument Deep expression to splice into the synthesized app:
        // a typed placeholder var for tensor/scalar args, or a recursive
        // List/tuple/ADT construction over leaf placeholders.
        let mut arg_exprs: Vec<Expr> = Vec::with_capacity(args.len());
        // Per-differentiated-target repack plan. After
        // evaluation the gradient roots come back as one FLAT tuple (the
        // pytree of every wrt-selected target's fields, concatenated in
        // parameter order); each plan slot says how many of those flat
        // roots the target owns and what structure to fold them back into
        // (a bare tensor, or a recursive List/tuple/ADT value). This is the
        // eval-lane twin of the IR lowering's `GradResultPlan` list. Only
        // wrt-selected differentiable targets get a slot; a non-selected
        // or non-differentiable argument still marshals its placeholders
        // (the body may read it) but owns no gradient root.
        enum ArgRepack {
            Tensor,
            Structured { shape: GradListShape },
        }
        let mut arg_repacks: Vec<ArgRepack> = Vec::with_capacity(args.len());
        // wrt indices for this grad call, if narrowed (`grad(f, wrt=i)`).
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
        let grad_formals = match kind {
            TransformKind::Grad => {
                resolve_transform_fn_for_formals(transform_expr, &self.top_level_defs)
                    .map(|(function, _)| function)
            }
            TransformKind::Vmap => None,
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
            // A handled grad must allocate Random ordinals along the branch
            // actually selected by a concrete discrete argument. Keeping a
            // bool behind a synthetic Load makes `lower_if` lower both arms,
            // so an untaken Random arm advances the stream. The evaluator
            // already has the exact runtime value at this boundary: embed it
            // as a typed literal so the lowering context can prune the
            // untaken arm before it allocates Random nodes or ordinals.
            if matches!(kind, TransformKind::Grad)
                && let RuntimeValue::Bool(value) = value
            {
                arg_exprs.push(make_bool_literal_with_type(*value, span));
                continue;
            }
            // A grad body may use an integer scalar as a discrete selector
            // (for example list_index/take_list/drop_list). A synthetic Load
            // preserves its dtype but erases its exact runtime value before
            // the staged List spine is selected. Embed that non-differentiable
            // argument as an exact typed literal instead; float/tensor
            // arguments still use Loads so the AD roots remain connected.
            if matches!(kind, TransformKind::Grad)
                && let RuntimeValue::Scalar(payload) = value
                && payload.dtype().is_integer()
            {
                let precision = fn_expr
                    .and_then(|expr| param_precision_at(expr, index))
                    .unwrap_or(payload.dtype());
                if precision.is_integer() {
                    arg_exprs.push(make_integer_literal_with_type(
                        payload.as_i64(),
                        precision,
                        span,
                    ));
                    continue;
                }
            }
            if matches!(kind, TransformKind::Grad)
                && matches!(
                    value,
                    RuntimeValue::List(_) | RuntimeValue::Tuple(_) | RuntimeValue::Adt { .. }
                )
            {
                let mut leaf_index = 0;
                let (expr, shape, differentiable) = stage_grad_list_value(
                    value,
                    index,
                    &mut leaf_index,
                    &mut placeholder_names,
                    &mut placeholder_types,
                    &mut placeholder_tensors,
                    span,
                )?;
                arg_exprs.push(expr);
                // A finite executed constructor can have no float leaves
                // even though another variant of its checked nominal type
                // does. Default and explicit `wrt` selection are type-based,
                // not guessed from that one runtime value.
                let statically_differentiable = grad_formals
                    .and_then(|function| param_type_expr_at(function, index))
                    .is_some_and(|ty| {
                        grad_type_expr_has_float(ty, &self.adt_registry, &mut Vec::new())
                    });
                let selected = (differentiable || statically_differentiable)
                    && grad_wrt
                        .as_ref()
                        .is_none_or(|indices| indices.contains(&index));
                if selected {
                    arg_repacks.push(ArgRepack::Structured { shape });
                }
                continue;
            }
            let placeholder = format!("__chelis_xform_arg_{index}");
            let (tensor_value, mut tensor_type) =
                runtime_value_to_dag_input_lossy(value, fn_expr, index)?;
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

        let scoped_types: UnordMap<String, TensorType> = placeholder_names
            .iter()
            .cloned()
            .zip(placeholder_types.iter().cloned())
            .collect();

        // Build a fresh `program_defs` that includes both top-level
        // defs from the host runtime AND any captured local closures
        // from `captured_env` (so `target = fn (...) -> ...; grad(target)(x)`
        // resolves `target` when the inner DAG lowering reaches it).
        let mut program_defs = self.top_level_defs.clone();
        for (name, value) in captured_env.to_sorted() {
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

        let lower_result = try_lower_subexpr_program_with_random_state_progress(
            &app_expr,
            scoped_types,
            self.type_env.clone(),
            program_defs,
            self.random_seed,
            self.random_counter,
        );
        let (dag, next_random_counter) = match lower_result {
            Ok(result) => result,
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
        let starting_random_counter = self.random_counter;
        let path_sensitive_random = dag.nodes().iter().any(|node| {
            matches!(node.op, chelis_ir::dag::RiscOp::UniformLike { .. }) && node.inputs.len() == 2
        });
        if !path_sensitive_random {
            // The ordinary baked-seed lane computes progression statically.
            self.random_counter = next_random_counter;
        }

        // Forward-evaluate the lowered DAG, satisfying `RiscOp::Load`
        // by looking up placeholder names in our staged inputs (or
        // tensor_bindings as a fallback for any external tensor refs
        // captured by the inner fn body).
        let tensor_bindings = self.tensor_bindings;
        let roots: Vec<chelis_ir::dag::NodeId> = dag.roots().to_vec();
        if roots.is_empty() {
            if matches!(kind, TransformKind::Grad)
                && !arg_repacks.is_empty()
                && arg_repacks.iter().all(
                    |slot| matches!(slot, ArgRepack::Structured { shape } if shape.leaf_count() == 0),
                )
            {
                let mut no_leaves = std::iter::empty();
                let mut empty_slots = arg_repacks.iter().map(|slot| match slot {
                    ArgRepack::Structured { shape } => shape.repack(&mut no_leaves),
                    ArgRepack::Tensor => {
                        unreachable!("guarded by empty List repack check")
                    }
                });
                return Ok(if arg_repacks.len() == 1 {
                    empty_slots.next().expect("one empty List slot")
                } else {
                    RuntimeValue::Tuple(empty_slots.collect())
                });
            }
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
        let mut captured_tensors: UnordMap<String, IrTensorValue> = captured_env
            .to_sorted()
            .into_iter()
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
        let (values, executed_random_counter) =
            chelis_ir::eval::eval_tensor_roots_with_strict_random_progress(
                &dag,
                &roots,
                starting_random_counter,
                |name| {
                    placeholder_tensors
                        .get(name)
                        .cloned()
                        .or_else(|| tensor_bindings.get(name).map(|t| t.value.clone()))
                        .or_else(|| captured_tensors.get(name).cloned())
                },
            )
            .map_err(|err| {
                let kind_label = match kind {
                    TransformKind::Grad => "grad",
                    TransformKind::Vmap => "vmap",
                };
                format!("host runtime `{kind_label}` evaluation failed: {err}")
            })?;
        if path_sensitive_random {
            self.random_counter = executed_random_counter;
        }

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
        let has_structured_slot = arg_repacks
            .iter()
            .any(|slot| matches!(slot, ArgRepack::Structured { .. }));
        if matches!(kind, TransformKind::Grad) && has_structured_slot {
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
                    ArgRepack::Structured { shape } => shape.leaf_count(),
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
                    ArgRepack::Structured { shape } => slots.push(shape.repack(&mut flat_iter)),
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

fn lookup_registered_type<'a, T>(
    entries: &'a std::collections::BTreeMap<String, T>,
    name: &str,
) -> Option<(&'a str, &'a T)> {
    entries
        .get_key_value(name)
        .map(|(key, value)| (key.as_str(), value))
        .or_else(|| {
            entries
                .iter()
                .find(|(candidate, _)| terminal_name_matches(candidate, name))
                .map(|(key, value)| (key.as_str(), value))
        })
}

fn registered_type_has_float(
    name: &str,
    argument_flags: Vec<bool>,
    registry: &chelis_types::adt::AdtRegistry,
    visiting: &mut Vec<(String, Vec<bool>)>,
) -> bool {
    if terminal_name_matches(name, "List") {
        return argument_flags.first().copied().unwrap_or(false);
    }
    if let Some((alias_name, alias)) = lookup_registered_type(&registry.aliases, name) {
        let key = (format!("alias:{alias_name}"), argument_flags.clone());
        if visiting.contains(&key) {
            return false;
        }
        visiting.push(key.clone());
        let substitutions = alias
            .param_args
            .iter()
            .zip(argument_flags)
            .filter_map(|(argument, flag)| match argument {
                NominalArg::Type(Type::Var(var)) => Some((*var, flag)),
                _ => None,
            })
            .collect::<UnordMap<_, _>>();
        let result = stored_type_has_float(&alias.body, registry, &substitutions, visiting);
        debug_assert_eq!(visiting.pop().as_ref(), Some(&key));
        return result;
    }
    let Some((definition_name, definition)) = lookup_registered_type(&registry.defs, name) else {
        return false;
    };
    let key = (format!("adt:{definition_name}"), argument_flags.clone());
    if visiting.contains(&key) {
        return false;
    }
    visiting.push(key.clone());
    let substitutions = definition
        .param_args
        .iter()
        .zip(argument_flags)
        .filter_map(|(argument, flag)| match argument {
            NominalArg::Type(Type::Var(var)) => Some((*var, flag)),
            _ => None,
        })
        .collect::<UnordMap<_, _>>();
    let result = definition.variants.iter().any(|variant| {
        variant
            .fields
            .iter()
            .any(|(_, field)| stored_type_has_float(field, registry, &substitutions, visiting))
    });
    debug_assert_eq!(visiting.pop().as_ref(), Some(&key));
    result
}

fn stored_type_has_float(
    ty: &Type,
    registry: &chelis_types::adt::AdtRegistry,
    substitutions: &UnordMap<TypeVar, bool>,
    visiting: &mut Vec<(String, Vec<bool>)>,
) -> bool {
    match ty {
        Type::Prim(prim) => prim.is_float(),
        Type::Tensor(_, TensorPrec::Concrete(prim)) => prim.is_float(),
        Type::Tensor(_, TensorPrec::Var(var)) | Type::Var(var) => {
            substitutions.get(var).copied().unwrap_or(false)
        }
        Type::Ref(inner) => stored_type_has_float(inner, registry, substitutions, visiting),
        Type::Tuple(items) => items
            .iter()
            .any(|item| stored_type_has_float(item, registry, substitutions, visiting)),
        Type::Adt(name, arguments) => registered_type_has_float(
            name,
            arguments
                .iter()
                .map(|argument| stored_type_has_float(argument, registry, substitutions, visiting))
                .collect(),
            registry,
            visiting,
        ),
        Type::KindedAdt(name, arguments) => registered_type_has_float(
            name,
            arguments
                .iter()
                .map(|argument| match argument {
                    NominalArg::Type(ty) => {
                        stored_type_has_float(ty, registry, substitutions, visiting)
                    }
                    NominalArg::Dimension(_) => false,
                })
                .collect(),
            registry,
            visiting,
        ),
        Type::Fn(_, _) | Type::Unit | Type::Error(_) => false,
    }
}

fn grad_type_expr_has_float(
    expr: &Expr,
    registry: &chelis_types::adt::AdtRegistry,
    visiting: &mut Vec<(String, Vec<bool>)>,
) -> bool {
    let (node_tag, kids) = match expr {
        Expr::List(list, _) => (tag(list), children(list)),
        Expr::Node(node, _) => (Some(node.tag()), node.children_slice()),
        _ => (None, &[][..]),
    };
    match node_tag {
        Some(DeepTag::TPrim) => kids
            .first()
            .and_then(symbol_name)
            .and_then(prim_from_name)
            .is_some_and(|prim| prim.is_float()),
        Some(DeepTag::TTensor) => kids
            .last()
            .is_some_and(|precision| grad_type_expr_has_float(precision, registry, visiting)),
        Some(DeepTag::TRef) => kids
            .first()
            .is_some_and(|inner| grad_type_expr_has_float(inner, registry, visiting)),
        Some(DeepTag::TTuple) => kids
            .iter()
            .any(|item| grad_type_expr_has_float(item, registry, visiting)),
        Some(DeepTag::TAdt) => {
            let Some(name) = kids.first().and_then(symbol_name) else {
                return false;
            };
            registered_type_has_float(
                name,
                kids.iter()
                    .skip(1)
                    .map(|argument| grad_type_expr_has_float(argument, registry, visiting))
                    .collect(),
                registry,
                visiting,
            )
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn stage_grad_list_value(
    value: &RuntimeValue,
    argument_index: usize,
    leaf_index: &mut usize,
    placeholder_names: &mut Vec<String>,
    placeholder_types: &mut Vec<TensorType>,
    placeholder_tensors: &mut UnordMap<String, IrTensorValue>,
    span: Span,
) -> Result<(Expr, GradListShape, bool), String> {
    match value {
        RuntimeValue::List(items) => {
            let mut item_exprs = Vec::with_capacity(items.len());
            let mut item_shapes = Vec::with_capacity(items.len());
            // A finite empty List has no runtime leaf from which to infer
            // differentiability, but its checked element type already made
            // the grad target legal. Preserve an empty repack slot so the
            // cotangent is the exact empty List rather than "no roots".
            let mut differentiable = items.is_empty();
            for item in items {
                let (expr, shape, item_differentiable) = stage_grad_list_value(
                    item,
                    argument_index,
                    leaf_index,
                    placeholder_names,
                    placeholder_types,
                    placeholder_tensors,
                    span,
                )?;
                item_exprs.push(expr);
                item_shapes.push(shape);
                differentiable |= item_differentiable;
            }
            Ok((
                make_list_construction_expr(item_exprs, span),
                GradListShape::List(item_shapes),
                differentiable,
            ))
        }
        RuntimeValue::Tuple(items) => {
            let mut item_exprs = Vec::with_capacity(items.len());
            let mut item_shapes = Vec::with_capacity(items.len());
            let mut differentiable = false;
            for item in items {
                let (expr, shape, item_differentiable) = stage_grad_list_value(
                    item,
                    argument_index,
                    leaf_index,
                    placeholder_names,
                    placeholder_types,
                    placeholder_tensors,
                    span,
                )?;
                item_exprs.push(expr);
                item_shapes.push(shape);
                differentiable |= item_differentiable;
            }
            Ok((
                make_tuple_expr(item_exprs, span),
                GradListShape::Tuple(item_shapes),
                differentiable,
            ))
        }
        RuntimeValue::Adt {
            ctor,
            fields,
            field_names,
        } => {
            let mut field_exprs = Vec::with_capacity(fields.len());
            let mut field_shapes = Vec::with_capacity(fields.len());
            let mut differentiable = false;
            for field in fields {
                let (expr, shape, field_differentiable) = stage_grad_list_value(
                    field,
                    argument_index,
                    leaf_index,
                    placeholder_names,
                    placeholder_types,
                    placeholder_tensors,
                    span,
                )?;
                field_exprs.push(expr);
                field_shapes.push(shape);
                differentiable |= field_differentiable;
            }
            Ok((
                make_adt_construction_exprs(ctor, field_names.as_deref(), field_exprs, span),
                GradListShape::Adt {
                    ctor: ctor.clone(),
                    field_names: field_names.clone(),
                    fields: field_shapes,
                },
                differentiable,
            ))
        }
        RuntimeValue::Tensor(_) | RuntimeValue::Scalar(_) | RuntimeValue::Bool(_) => {
            let current_leaf = *leaf_index;
            *leaf_index += 1;
            let (tensor_value, tensor_type) =
                runtime_value_to_dag_input_lossy(value, None, current_leaf)?;
            let placeholder =
                format!("__chelis_xform_arg_{argument_index}_list_leaf_{current_leaf}");
            placeholder_tensors.insert(placeholder.clone(), tensor_value);
            placeholder_names.push(placeholder.clone());
            placeholder_types.push(tensor_type.clone());
            let differentiable = tensor_type.precision.is_float();
            let shape = if differentiable {
                GradListShape::Leaf
            } else {
                GradListShape::Unit
            };
            Ok((
                make_var_with_type(&placeholder, &tensor_type, span),
                shape,
                differentiable,
            ))
        }
        RuntimeValue::Unit => Ok((make_unit_expr(span), GradListShape::Unit, false)),
        _ => Err(format!(
            "host runtime: `grad(...)` structured argument {argument_index}: recursive \
             element is not a supported scalar, tensor, List, tuple, ADT, or unit value"
        )),
    }
}

fn make_tuple_expr(elements: Vec<Expr>, span: Span) -> Expr {
    Expr::List(
        List {
            elements: std::iter::once(Expr::Atom(Atom::Tag(DeepTag::Tuple), span))
                .chain(std::iter::once(Expr::Map(MetaMap::default(), span)))
                .chain(elements)
                .collect(),
        },
        span,
    )
}

fn make_unit_expr(span: Span) -> Expr {
    Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Tuple), span),
                Expr::Map(MetaMap::default(), span),
            ],
        },
        span,
    )
}

fn make_adt_construction_exprs(
    ctor: &str,
    field_names: Option<&[String]>,
    field_exprs: Vec<Expr>,
    span: Span,
) -> Expr {
    match field_names {
        Some(names) if names.len() == field_exprs.len() => {
            let mut elements = vec![
                Expr::Atom(Atom::Tag(DeepTag::Record), span),
                Expr::Map(MetaMap::default(), span),
                Expr::Atom(Atom::Name(ctor.to_string()), span),
            ];
            elements.extend(names.iter().zip(field_exprs).map(|(name, expr)| {
                Expr::List(
                    List {
                        elements: vec![
                            Expr::Atom(Atom::Tag(DeepTag::Kv), span),
                            Expr::Map(MetaMap::default(), span),
                            Expr::Atom(Atom::Name(name.clone()), span),
                            expr,
                        ],
                    },
                    span,
                )
            }));
            Expr::List(List { elements }, span)
        }
        _ => {
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
            elements.extend(field_exprs);
            Expr::List(List { elements }, span)
        }
    }
}

/// Build the exact closed `Cons`/`Nil` value used to stage a runtime List
/// through the tensor DAG transform. Elements may themselves be staged Lists;
/// every scalar or tensor leaf owns one typed placeholder, so reverse mode can
/// reconstruct the complete primal runtime shape and order.
fn make_list_construction_expr(elements: Vec<Expr>, span: Span) -> Expr {
    let mut tail = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Var), span),
                Expr::Map(MetaMap::default(), span),
                Expr::Atom(Atom::Name("Nil".to_string()), span),
            ],
        },
        span,
    );
    for element in elements.into_iter().rev() {
        tail = Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Tag(DeepTag::App), span),
                    Expr::Map(MetaMap::default(), span),
                    Expr::List(
                        List {
                            elements: vec![
                                Expr::Atom(Atom::Tag(DeepTag::Var), span),
                                Expr::Map(MetaMap::default(), span),
                                Expr::Atom(Atom::Name("Cons".to_string()), span),
                            ],
                        },
                        span,
                    ),
                    element,
                    tail,
                ],
            },
            span,
        );
    }
    tail
}

/// Bucket 1 helper: lossily convert a host-runtime argument into the legacy
/// f64-backed `(TensorValue, TensorType)` pair the IR DAG can consume. Scalar
/// args (Int/Float/Bool) are wrapped as rank-0 tensors with the precision
/// pulled from the inner fn's parameter type metadata when available, or
/// from the runtime value as a fallback. Exact transform ingress remains
/// chelis#688 / #729 Phase 2 work.
pub(super) fn runtime_value_to_dag_input_lossy(
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
                IrTensorValue::scalar(payload.as_f64_lossy()),
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
                IrTensorValue::scalar(payload.as_i64() as f64),
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
    defs: &'a UnordMap<String, Expr>,
) -> Option<(&'a Expr, Option<usize>)> {
    let mut current = expr;
    let mut vmap_axis: Option<usize> = None;
    let mut visited: UnordSet<&str> = UnordSet::new();
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

/// Build a `(lit {type: (t-prim {} <integer-dtype>)} value)` expression.
/// Grad uses this for exact runtime-fed discrete arguments whose value must be
/// visible while lowering a staged List spine.
fn make_integer_literal_with_type(value: i64, precision: Prim, span: Span) -> Expr {
    let prim_name = match precision {
        Prim::Int8 => "int8",
        Prim::Int16 => "int16",
        Prim::Int32 => "int32",
        Prim::Int64 => "int64",
        _ => panic!("integer transform argument unexpectedly declared with non-integer dtype"),
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
    let mut meta = MetaMap::default();
    meta.entries.push(("type".to_string(), prim_node));
    Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Lit), span),
                Expr::Map(meta, span),
                Expr::Atom(Atom::Int(value), span),
            ],
        },
        span,
    )
}

/// Build a `(lit {type: (t-prim {} bool)} value)` expression for a concrete
/// non-differentiable transform argument.
fn make_bool_literal_with_type(value: bool, span: Span) -> Expr {
    let prim_node = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::TPrim), span),
                Expr::Map(MetaMap::default(), span),
                Expr::Atom(Atom::Name("bool".to_string()), span),
            ],
        },
        span,
    );
    let mut meta = MetaMap::default();
    meta.entries.push(("type".to_string(), prim_node));
    Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Lit), span),
                Expr::Map(meta, span),
                Expr::Atom(Atom::Bool(value), span),
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
    defs: &UnordMap<String, Expr>,
) -> Option<String> {
    let mut hit: Option<String> = None;
    let mut visited: UnordSet<String> = UnordSet::new();
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
