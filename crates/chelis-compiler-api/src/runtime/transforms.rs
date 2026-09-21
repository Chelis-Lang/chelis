use chelis_deep::DeepTag;
use chelis_unord::{UnordMap, UnordSet};

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, ExprCarrier, List, Metadata};
use chelis_ir::dag::{DimInfo, TensorType};
use chelis_ir::eval::{TensorInputDemand, TensorValue as IrTensorValue};
use chelis_ir::evaluation::{EvaluationProfile, RandomExecutionContext};
use chelis_ir::host::RandomLoweringState;
use chelis_ir::lower::SubexprLoweringContext;
use chelis_types::types::{NominalArg, Prim, TensorPrec, Type, TypeVar};

use super::host_ops::terminal_name_matches;
use super::named_axis::*;
use super::*;

#[derive(Clone)]
enum ArgRepack {
    Tensor,
    Scalar(Prim),
    Structured { shape: GradListShape },
}

#[derive(Clone)]
enum GradListShape {
    Leaf,
    ScalarLeaf(Prim),
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
            Self::Leaf | Self::ScalarLeaf(_) => 1,
            Self::Unit => 0,
            Self::List(items) | Self::Tuple(items) => items.iter().map(Self::leaf_count).sum(),
            Self::Adt { fields, .. } => fields.iter().map(Self::leaf_count).sum(),
        }
    }

    fn repack(
        &self,
        leaves: &mut impl Iterator<Item = RuntimeValue>,
    ) -> Result<RuntimeValue, String> {
        Ok(match self {
            Self::Leaf => leaves.next().expect("gradient leaf count checked above"),
            Self::ScalarLeaf(prim) => repack_scalar_gradient(
                leaves.next().expect("gradient leaf count checked above"),
                *prim,
            )?,
            Self::Unit => RuntimeValue::Unit,
            Self::List(items) => RuntimeValue::List(
                items
                    .iter()
                    .map(|item| item.repack(leaves))
                    .collect::<Result<_, _>>()?,
            ),
            Self::Tuple(items) => RuntimeValue::Tuple(
                items
                    .iter()
                    .map(|item| item.repack(leaves))
                    .collect::<Result<_, _>>()?,
            ),
            Self::Adt {
                ctor,
                field_names,
                fields,
            } => RuntimeValue::Adt {
                ctor: ctor.clone(),
                fields: fields
                    .iter()
                    .map(|field| field.repack(leaves))
                    .collect::<Result<_, _>>()?,
                field_names: field_names.clone(),
            },
        })
    }
}

/// spec/06 §2.1 gives a float scalar the same scalar cotangent type. The
/// DAG uses rank-zero tensors for both kinds, so retain the typed primal's
/// carrier distinction before marshalling and restore it after evaluation.
/// Read the finalized tagged element directly: this is not a numeric cast.
fn repack_scalar_gradient(value: RuntimeValue, prim: Prim) -> Result<RuntimeValue, String> {
    let RuntimeValue::Tensor(tensor) = value else {
        return Err("host runtime: scalar gradient root is not a DAG tensor".into());
    };
    if !tensor.value.shape.is_empty() || tensor.value.len() != 1 || tensor.value.prim() != prim {
        return Err(format!(
            "host runtime: scalar gradient expected one rank-zero {} result, got rank {} with {} elements at {}",
            prim.name(),
            tensor.value.shape.len(),
            tensor.value.len(),
            tensor.value.prim().name()
        ));
    }
    Ok(RuntimeValue::from_scalar_value(
        tensor.value.storage().scalar_at(0),
    ))
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
        captured_env: Frame,
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
        // selected order); each plan slot says how many of those flat
        // roots the target owns and what structure to fold them back into
        // (a scalar, a tensor, or a recursive List/tuple/ADT value). This is the
        // eval-lane twin of the IR lowering's `GradResultPlan` list. Only
        // wrt-selected differentiable targets get a slot; a non-selected
        // or non-differentiable argument still marshals its placeholders
        // (the body may read it) but owns no gradient root.
        let mut arg_repacks: Vec<(usize, ArgRepack)> = Vec::with_capacity(args.len());
        // wrt indices for this grad call, if narrowed (`grad(f, wrt=i)`).
        // `None` means differentiate every differentiable argument, exactly
        // as the checker's `grad_result_type` and the IR lowering's
        // `is_selected_wrt` do.
        let grad_wrt = match kind {
            TransformKind::Grad => grad_wrt_indices_from_transform(transform_expr)?,
            TransformKind::Vmap => None,
        };

        // Read the exact admitted transform carrier so we can inspect the inner
        // function's parameter type metadata. The transform_expr is the
        // captured `(grad ... fn-expr ...)` or `(vmap ... fn-expr
        // axis-lit)` form; the fn-expr is the first child.
        let expected_tag = match kind {
            TransformKind::Grad => DeepTag::Grad,
            TransformKind::Vmap => DeepTag::Vmap,
        };
        let transform_children = match transform_expr.carrier() {
            ExprCarrier::DecodedNode(tag, _, children) if tag == expected_tag => children,
            ExprCarrier::DecodedNode(tag, _, _) => {
                return Err(format!(
                    "host runtime: expected `{}`, found `{}` transform",
                    expected_tag.as_str(),
                    tag.as_str()
                ));
            }
            ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_)
            | ExprCarrier::MalformedLegacyList(_) => {
                return Err(format!(
                    "host runtime: `{}` transform is not a decoded runtime node",
                    expected_tag.as_str()
                ));
            }
        };
        let fn_expr = transform_children.first();
        // #1956: only the already-resolved direct declaration owns free
        // values here. The fresh lowerer has no local callable for this
        // operand; its exact program_defs entry is the original Fn, and
        // captured-closure injection below cannot replace that entry.
        // A present snapshot binding, alias or inline Fn stays on the old
        // lexical path. Do not use current caller bindings or the formals
        // helper to infer identity, and do not rewrite the target.
        let declaration_captures = matches!(kind, TransformKind::Grad)
            && fn_expr.and_then(var_name).is_some_and(|name| {
                !captured_env.contains_key(name)
                    && self.program.defs().get(name).is_some_and(|body| {
                        tagged_expr_children(body).is_some_and(|(tag, _)| tag == DeepTag::Fn)
                    })
            });
        let grad_formals = match kind {
            TransformKind::Grad => {
                resolve_transform_fn_for_formals(transform_expr, self.program.defs())
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
        // stays unbound and no input declares it, so the C lane has no
        // extent source for it. Type the placeholder from
        // the callee's declared formals instead (the chelis#338/#346
        // pattern for plain def calls): the vmap axis stays `Lit`, the
        // mapped axes carry the formal's names with runtime sizes, and
        // the symbolic-dim machinery binds the body's names against the
        // placeholder Load. Best-effort: any unresolved shape falls back
        // to the Lit-dim marshalling below.
        let vmap_formals = match kind {
            TransformKind::Vmap => {
                resolve_transform_fn_for_formals(transform_expr, self.program.defs())
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
            // (for example list_index/take_list/skip_list). A synthetic Load
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
                    arg_repacks.push((index, ArgRepack::Structured { shape }));
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
            // one gradient root, packed back with its original carrier. A
            // non-float or non-selected argument owns none (matching the IR lowering's
            // `is_selected_wrt`), so it gets no repack slot even though its
            // placeholder is still marshalled (the body may read it).
            if matches!(kind, TransformKind::Grad) {
                let differentiable = tensor_type.precision.is_float();
                let selected = differentiable
                    && grad_wrt
                        .as_ref()
                        .is_none_or(|indices| indices.contains(&index));
                if selected {
                    arg_repacks.push((
                        index,
                        match value {
                            RuntimeValue::Scalar(payload) => ArgRepack::Scalar(payload.dtype()),
                            _ => ArgRepack::Tensor,
                        },
                    ));
                }
            }
        }

        // IR emits complete cotangent groups in written `wrt` order.
        // Restore carriers in that same order; argument staging above must
        // stay in primal order, including non-selected argument values.
        let arg_repacks: Vec<ArgRepack> = match grad_wrt.as_ref() {
            Some(indices) => indices
                .iter()
                .filter_map(|index| {
                    arg_repacks
                        .iter()
                        .find(|(parameter, _)| parameter == index)
                        .map(|(_, plan)| plan.clone())
                })
                .collect(),
            None => arg_repacks.into_iter().map(|(_, plan)| plan).collect(),
        };

        // Synthesize `(app {} <transform-expr> <arg-expr_0> ...)`.
        let mut app_elements: Vec<Expr> = Vec::with_capacity(3 + arg_exprs.len());
        app_elements.push(Expr::Atom(Atom::Tag(DeepTag::App), span));
        app_elements.push(Expr::Map(Metadata::default(), span));
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

        // Build a fresh `program_defs` that includes both top-level defs from
        // the host runtime AND any captured local closures. The captured
        // binding wins when its name shadows a top-level def, matching the
        // evaluator's lexical environment and [04-LIN-1]. The closure owns the
        // exact checked function expression: reconstructing one from only
        // parameter names and the body erases parameter types and the checked
        // function signature. A nested function-valued capture then reaches
        // lowering as rank zero and corrupts the backward DAG (chelis#676).
        let mut program_defs = self.program.defs().clone();
        for (name, value) in captured_env.to_sorted() {
            if let RuntimeValue::Closure {
                checked_function, ..
            } = value
            {
                program_defs.insert(name.clone(), checked_function.as_ref().clone());
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

        let profile = self.execution_profile(&app_expr, &program_defs);
        // #1821/#1920: inference renames result dimensions (n -> d43),
        // while invocation witnesses retain the authored parameter binders.
        // Give both routes the declared signature alongside checked types,
        // so the result claim still refers to its activation's witness.
        let lowering = if let Some(session) = &self.session {
            SubexprLoweringContext::from_checked_program(
                session.program(),
                program_defs,
                self.declared_signatures.clone(),
            )
        } else {
            SubexprLoweringContext::new(
                self.program.type_env().clone(),
                program_defs,
                self.declared_signatures.clone(),
            )
        };
        let mut execution_plan = None;
        let lower_result = if profile == EvaluationProfile::FixedControl {
            let context = RandomExecutionContext::new(RandomLoweringState {
                seed: self.random_seed,
                counter: self.random_counter,
            });
            lowering
                .lower_evaluation_plan(&app_expr, scoped_types, &context)
                .map(|plan| {
                    let dag = plan.dag_for_inspection().clone();
                    execution_plan = Some(plan);
                    (dag, self.random_counter)
                })
        } else {
            lowering
                .lower_with_random_state(
                    &app_expr,
                    scoped_types,
                    RandomLoweringState {
                        seed: self.random_seed,
                        counter: self.random_counter,
                    },
                )
                .map(|(dag, random)| (dag, random.counter))
        };
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
        let baked_random_progress = execution_plan.is_none() && !path_sensitive_random;

        // Forward-evaluate the lowered DAG, satisfying `RiscOp::Load`
        // by looking up placeholder names in our staged inputs (or
        // tensor_bindings as a fallback for any external tensor refs
        // captured by the inner fn body).
        let tensor_bindings = self.tensor_bindings;
        let roots: Vec<chelis_ir::dag::NodeId> = dag.roots().to_vec();
        let mut empty_packed = None;
        if roots.is_empty() {
            // Preserve the historical empty-root early-return behavior. In
            // particular, [] must not turn an empty legacy grad into ALL-node
            // input preparation. A source-owned plan still executes below.
            if baked_random_progress {
                self.random_counter = next_random_counter;
            }
            if matches!(kind, TransformKind::Grad)
                && !arg_repacks.is_empty()
                && arg_repacks.iter().all(
                    |slot| matches!(slot, ArgRepack::Structured { shape } if shape.leaf_count() == 0),
                )
            {
                let mut no_leaves = std::iter::empty();
                let mut empty_slots = arg_repacks.iter().map(|slot| match slot {
                    ArgRepack::Structured { shape } => shape.repack(&mut no_leaves),
                    ArgRepack::Tensor | ArgRepack::Scalar(_) => {
                        unreachable!("guarded by empty List repack check")
                    }
                });
                let packed = if arg_repacks.len() == 1 {
                    empty_slots.next().expect("one empty List slot")?
                } else {
                    RuntimeValue::Tuple(empty_slots.collect::<Result<_, _>>()?)
                };
                if execution_plan.is_none() { return Ok(packed); }
                empty_packed = Some(packed);
            }
            if empty_packed.is_none() {
                let kind_label = match kind {
                    TransformKind::Grad => "grad",
                    TransformKind::Vmap => "vmap",
                };
                return Err(format!(
                    "host runtime: `{kind_label}(...)` lowering produced no roots"
                ));
            }
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
        // A direct declaration's served captures come from the canonical
        // provider below, never the caller's same-spelled lexical tensors.
        // Keep the served values in this map for the existing rank guard.
        let mut captured_tensors: UnordMap<String, IrTensorValue> = if declaration_captures {
            UnordMap::new()
        } else {
            captured_env
                .to_sorted()
                .into_iter()
                .filter_map(|(name, value)| match value {
                    RuntimeValue::Tensor(tensor) => Some((name.clone(), tensor.value.clone())),
                    _ => None,
                })
                .collect()
        };

        // chelis#377 (vmap-inside-a-def): when the transform is applied
        // inside another def's body (`def fv(xs) = xs |> vmap(dot_w)`), the
        // `captured_env` is that body's local scope (`xs`), so a TOP-LEVEL
        // tensor binding the inner fn captures (`dot_w` referencing top-level
        // `w`) is in neither `captured_env` nor `tensor_bindings`. Resolve only
        // inputs requested by the same selection authority as execution.
        // A provider error is an entered initializer's error, not an evaluator
        // missing-input diagnostic; preserve it without the legacy prefix.
        let preparation_context = RandomExecutionContext::new(RandomLoweringState {
            seed: self.random_seed,
            counter: self.random_counter,
        });
        let mut provider_failed = false;
        let prepare_input = |name: &str, demand: TensorInputDemand| {
            // eval_compiled supplies manifested Tensor-lane root values in
            // tensor_bindings, not raw caller parameters. Actual arguments
            // were evaluated separately and own the placeholders first.
            if let Some(value) = placeholder_tensors
                .get(name)
                .cloned()
                .or_else(|| tensor_bindings.get(name).map(|t| t.value.clone()))
                .or_else(|| captured_tensors.get(name).cloned())
            {
                return Ok(Some(value));
            }
            // Unknown or ambiguous names may be optional shape declarers.
            // Do not confuse this absence with an error *inside* a known
            // initializer, even if that error also names an unknown binding.
            let Some((resolved, _)) = self.lookup_top_level_def(name) else {
                return Ok(None);
            };
            if declaration_captures
                && demand == TensorInputDemand::AvailableShape
                && !self.declaration_values.contains_key(&resolved)
            {
                // Optional shape queries may reuse an initialized canonical
                // value, but cannot enter a previously caller-masked initializer.
                return Ok(None);
            }
            match self.resolve_top_level(name) {
                Ok(RuntimeValue::Tensor(tensor)) => {
                    captured_tensors.insert(name.to_string(), tensor.value.clone());
                    Ok(Some(tensor.value))
                }
                Ok(_) => Ok(None),
                Err(error) => {
                    provider_failed = true;
                    Err(error)
                }
            }
        };
        let prepared_inputs = if let Some(plan) = &execution_plan {
            chelis_ir::eval::prepare_tensor_plan_inputs_with_demand(
                plan,
                &preparation_context,
                prepare_input,
            )
        } else {
            chelis_ir::eval::prepare_tensor_roots_inputs_with_demand(&dag, &roots, prepare_input)
        }
        .map_err(|error| {
            if provider_failed || execution_plan.is_some() {
                error
            } else {
                let kind_label = match kind {
                    TransformKind::Grad => "grad",
                    TransformKind::Vmap => "vmap",
                };
                format!("host runtime `{kind_label}` evaluation failed: {error}")
            }
        })?;
        // A served capture must match its authored-rank `Load`. Capture-aware
        // vmap preserves that raw load and gives its mapped identity an
        // explicit rank-inserting movement, so the batch axis never widens the
        // load contract itself. Keep this guard as a defensive invariant check
        // before an inconsistent DAG can reach elementwise evaluation.
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
                    "host runtime: `{kind_label}(...)` capture rank invariant failed for \
                     top-level binding `{name}`: the authored `Load` expects rank {} but the \
                     binding has rank {}.",
                    node.output_type.dims.len(),
                    value.shape.len(),
                ));
            }
        }
        // Preparation and capture validation can fail before the callee is
        // entered. Publish the baked lane's existing static progression only
        // once those fallible steps have succeeded; its sampler is unchanged.
        if baked_random_progress {
            self.random_counter = next_random_counter;
        }
        let load = |name: &str| prepared_inputs.get(name).cloned();
        let result = if let Some(plan) = &execution_plan {
            let mut context = RandomExecutionContext::new(RandomLoweringState {
                seed: self.random_seed,
                counter: self.random_counter,
            });
            let result = chelis_ir::eval::eval_tensor_plan_with_strict(plan, &mut context, load);
            self.random_counter = context.state().counter;
            result.map(|values| (values, self.random_counter))
        } else {
            chelis_ir::eval::eval_tensor_roots_with_strict_random_progress(
                &dag,
                &roots,
                starting_random_counter,
                load,
            )
        };
        let (values, executed_random_counter) = result.map_err(|err| {
            if execution_plan.is_some() {
                return err;
            }
            let kind_label = match kind {
                TransformKind::Grad => "grad",
                TransformKind::Vmap => "vmap",
            };
            format!("host runtime `{kind_label}` evaluation failed: {err}")
        })?;
        if execution_plan.is_none() && path_sensitive_random {
            self.random_counter = executed_random_counter;
        }
        if let Some(packed) = empty_packed {
            return Ok(packed);
        }

        // Pack roots back into a RuntimeValue.
        let kind_label = match kind {
            TransformKind::Grad => "grad",
            TransformKind::Vmap => "vmap",
        };
        let packed = pack_dag_roots(&dag, &roots, &values, kind_label)?;
        // Restore every selected target's cotangent carrier and recursive
        // shape, including scalar leaves. DAG rank alone cannot distinguish
        // a scalar from a genuine rank-zero tensor (spec/06 §2.1).
        if matches!(kind, TransformKind::Grad) && !arg_repacks.is_empty() {
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
                    ArgRepack::Tensor | ArgRepack::Scalar(_) => 1,
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
                    ArgRepack::Scalar(prim) => slots.push(repack_scalar_gradient(
                        flat_iter.next().expect("count checked above"),
                        *prim,
                    )?),
                    ArgRepack::Structured { shape } => slots.push(shape.repack(&mut flat_iter)?),
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
/// `static_usize_value`). Only an absent second child means the default
/// all-arguments selection; an unreadable carrier or index is a runtime
/// boundary error rather than the same silent default.
fn grad_wrt_indices_from_transform(transform_expr: &Expr) -> Result<Option<Vec<usize>>, String> {
    let children = match transform_expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Grad, _, children) => children,
        ExprCarrier::DecodedNode(tag, _, _) => {
            return Err(format!(
                "host runtime: expected `grad`, found `{}` transform",
                tag.as_str()
            ));
        }
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => {
            return Err("host runtime: `grad` transform is not a decoded runtime node".to_string());
        }
    };
    let Some(wrt_expr) = children.get(1) else {
        return Ok(None);
    };
    if let ExprCarrier::DecodedNode(DeepTag::Tuple, _, indices) = wrt_expr.carrier() {
        return indices
            .iter()
            .map(|index| {
                static_usize_value(index).ok_or_else(|| {
                    "host runtime: `grad` wrt tuple contains a non-static parameter index"
                        .to_string()
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some);
    }
    static_usize_value(wrt_expr)
        .map(|index| Some(vec![index]))
        .ok_or_else(|| {
            "host runtime: `grad` wrt selector is not a static parameter index".to_string()
        })
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
    let (node_tag, kids) = match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => (Some(tag), children),
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => (None, &[][..]),
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
                match value {
                    RuntimeValue::Scalar(payload) => GradListShape::ScalarLeaf(payload.dtype()),
                    _ => GradListShape::Leaf,
                }
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
                .chain(std::iter::once(Expr::Map(Metadata::default(), span)))
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
                Expr::Map(Metadata::default(), span),
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
                Expr::Map(Metadata::default(), span),
                Expr::Atom(Atom::Name(ctor.to_string()), span),
            ];
            elements.extend(names.iter().zip(field_exprs).map(|(name, expr)| {
                Expr::List(
                    List {
                        elements: vec![
                            Expr::Atom(Atom::Tag(DeepTag::Kv), span),
                            Expr::Map(Metadata::default(), span),
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
                Expr::Map(Metadata::default(), span),
                Expr::List(
                    List {
                        elements: vec![
                            Expr::Atom(Atom::Tag(DeepTag::Var), span),
                            Expr::Map(Metadata::default(), span),
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
                Expr::Map(Metadata::default(), span),
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
                    Expr::Map(Metadata::default(), span),
                    Expr::List(
                        List {
                            elements: vec![
                                Expr::Atom(Atom::Tag(DeepTag::Var), span),
                                Expr::Map(Metadata::default(), span),
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
        let (tag, children) = match current.carrier() {
            ExprCarrier::DecodedNode(tag, _, children) => (tag, children),
            ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_)
            | ExprCarrier::MalformedLegacyList(_) => return None,
        };
        match tag {
            DeepTag::Fn => return Some((current, vmap_axis)),
            DeepTag::Var => {
                let name = children.first().and_then(symbol_name)?;
                if !visited.insert(name) {
                    return None;
                }
                current = defs.get(name)?;
            }
            DeepTag::Grad => {
                current = children.first()?;
            }
            DeepTag::Vmap => {
                if vmap_axis.is_some() {
                    return None;
                }
                let axis = match children.get(1) {
                    Some(axis_expr) => static_usize_value(axis_expr)?,
                    None => 0,
                };
                vmap_axis = Some(axis);
                current = children.first()?;
            }
            _ => return None,
        }
    }
}

/// Static non-negative int literal: a bare int atom, `(lit {} n)`, or a
/// `cast(n, i32)` wrapper (mirrors the lowerer's
/// `extract_usize_value` shapes for the vmap axis argument).
fn static_usize_value(expr: &Expr) -> Option<usize> {
    match expr.carrier() {
        ExprCarrier::Atom(Atom::Int(n)) => usize::try_from(*n).ok(),
        ExprCarrier::DecodedNode(tag, _, children) => match tag {
            DeepTag::Lit => match children.first()? {
                Expr::Atom(Atom::Int(n), _) => usize::try_from(*n).ok(),
                _ => None,
            },
            DeepTag::Cast => static_usize_value(children.first()?),
            _ => None,
        },
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => None,
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
    runtime_param_parts(param)?
        .1
        .and_then(|metadata| metadata.ty().map(|ty| ty.expression()))
}

/// `(fn ...)` param[index]'s declared type expression, if annotated.
pub(super) fn param_type_expr_at(fn_expr: &Expr, index: usize) -> Option<&Expr> {
    let params = match fn_expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Fn, _, children) => children.first()?,
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => return None,
    };
    let params = match params.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Params, _, children) => children,
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => return None,
    };
    param_decl_type_expr(params.get(index)?)
}

/// Best-effort lookup of `(fn ...)` param[index]'s primitive precision
/// from its `type` metadata.
fn param_precision_at(fn_expr: &Expr, index: usize) -> Option<Prim> {
    extract_prim_from_type_expr(param_type_expr_at(fn_expr, index)?)
}

pub(super) fn extract_prim_from_type_expr(expr: &Expr) -> Option<Prim> {
    let (node_tag, kids) = match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => (tag, children),
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => return None,
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
        "i8" => Prim::Int8,
        "i16" => Prim::Int16,
        "i32" => Prim::Int32,
        "i64" => Prim::Int64,
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
        Prim::Int8 => "i8",
        Prim::Int16 => "i16",
        Prim::Int32 => "i32",
        Prim::Int64 => "i64",
        Prim::Bool => "bool",
        Prim::String => "string",
    };
    let prim_node = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::TPrim), span),
                Expr::Map(Metadata::default(), span),
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
            Expr::Map(Metadata::default(), span),
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
    let mut meta = Metadata::default();
    meta.replace(chelis_deep::annotations::MetadataValue::Type(
        chelis_deep::annotations::TypeSyntax::try_new(ty_expr).expect("runtime type annotation"),
    ));
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
        Prim::Int8 => "i8",
        Prim::Int16 => "i16",
        Prim::Int32 => "i32",
        Prim::Int64 => "i64",
        _ => panic!("integer transform argument unexpectedly declared with non-integer dtype"),
    };
    let prim_node = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::TPrim), span),
                Expr::Map(Metadata::default(), span),
                Expr::Atom(Atom::Name(prim_name.to_string()), span),
            ],
        },
        span,
    );
    let mut meta = Metadata::default();
    meta.replace(chelis_deep::annotations::MetadataValue::Type(
        chelis_deep::annotations::TypeSyntax::try_new(prim_node).expect("runtime type annotation"),
    ));
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
                Expr::Map(Metadata::default(), span),
                Expr::Atom(Atom::Name("bool".to_string()), span),
            ],
        },
        span,
    );
    let mut meta = Metadata::default();
    meta.replace(chelis_deep::annotations::MetadataValue::Type(
        chelis_deep::annotations::TypeSyntax::try_new(prim_node).expect("runtime type annotation"),
    ));
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
                    Expr::Map(Metadata::default(), span),
                    Expr::Atom(Atom::Int(*value as i64), span),
                ],
            },
            span,
        ),
        DimInfo::Named(name, _) => Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Tag(DeepTag::DName), span),
                    Expr::Map(Metadata::default(), span),
                    Expr::Atom(Atom::Name(name.clone()), span),
                ],
            },
            span,
        ),
    }
}

pub(super) fn var_name(expr: &Expr) -> Option<&str> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Var, _, children) => {
            children.first().and_then(symbol_name)
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => None,
    }
}

#[cfg(test)]
pub(super) fn runtime_param_name(expr: &Expr) -> Option<&str> {
    runtime_param_parts(expr).map(|(name, _)| name)
}

pub(super) fn runtime_param_parts(expr: &Expr) -> Option<(&str, Option<&Metadata>)> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some((name.as_str(), None)),
        Expr::BareList(elements, _) | Expr::List(List { elements }, _) => {
            let [Expr::Atom(Atom::Name(name), _), Expr::Map(metadata, _)] = elements.as_slice()
            else {
                return None;
            };
            Some((name.as_str(), Some(metadata)))
        }
        Expr::MetaExpr(metadata_expr, _) => {
            let Expr::Atom(Atom::Name(name), _) = metadata_expr.expr.as_ref() else {
                return None;
            };
            Some((name.as_str(), Some(&metadata_expr.metadata)))
        }
        Expr::Node(..) | Expr::UnknownForm(..) | Expr::Atom(..) | Expr::Map(..) => None,
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
    let (tag, children) = match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => (Some(tag), children),
        ExprCarrier::StructuralList(children) => (None, children),
        ExprCarrier::UndecodableHead(_, _, children) => (None, children),
        ExprCarrier::MetadataExpression(meta) => {
            scan_expr_for_host_only(&meta.expr, hit, vars);
            return;
        }
        ExprCarrier::MalformedLegacyList(list) => (None, list.elements.as_slice()),
        ExprCarrier::Atom(_) | ExprCarrier::MetadataMap(_) => return,
    };
    if tag == Some(DeepTag::App)
        && let Some(callee) = children.first()
        && let Some(name) = var_name(callee)
        && HOST_ONLY_BUILTIN_NAMES.contains(&name)
    {
        if hit.is_none() {
            *hit = Some(name.to_string());
        }
        return;
    }
    if tag == Some(DeepTag::Var)
        && let Some(name) = children.first().and_then(symbol_name)
    {
        vars.push(name.to_string());
    }
    for child in children {
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

#[cfg(test)]
mod scalar_gradient_tests {
    use super::*;

    #[test]
    fn scalar_repacking_preserves_declared_dtype_and_signed_zero() {
        for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
            let tensor = RuntimeTensorValue::from_wide("test", prim, vec![], vec![-0.0]).unwrap();
            let result = repack_scalar_gradient(RuntimeValue::Tensor(tensor), prim).unwrap();
            let RuntimeValue::Scalar(payload) = result else {
                panic!("scalar required")
            };
            assert_eq!(payload.dtype(), prim);
            assert!(payload.as_f64_lossy().is_sign_negative());
        }
    }

    #[test]
    fn scalar_repacking_rejects_wrong_rank_or_dtype_without_coercion() {
        for (prim, shape, values) in [
            (Prim::F32, vec![1], vec![1.0]),
            (Prim::F64, vec![], vec![1.0]),
            (Prim::F32, vec![0], vec![]),
        ] {
            let tensor = RuntimeTensorValue::from_wide("test", prim, shape, values).unwrap();
            assert!(repack_scalar_gradient(RuntimeValue::Tensor(tensor), Prim::F32).is_err());
        }
    }
}

#[cfg(test)]
mod issue_1125_carrier_reader_tests {
    use super::*;
    use chelis_deep::annotations::{MetadataValue, TypeSyntax};
    use chelis_deep::ast::{MetaExpr, UnknownFormData};

    fn span() -> Span {
        Span::new(2, 9)
    }

    fn node(tag: DeepTag, children: Vec<Expr>) -> Expr {
        Expr::node(tag, Metadata::default(), children, span())
    }

    fn legacy(tag: DeepTag, children: Vec<Expr>) -> Expr {
        let mut elements = vec![
            Expr::Atom(Atom::Tag(tag), span()),
            Expr::Map(Metadata::default(), span()),
        ];
        elements.extend(children);
        Expr::List(List { elements }, span())
    }

    fn name(value: &str) -> Expr {
        Expr::Atom(Atom::Name(value.to_string()), span())
    }

    fn typed_param(name_value: &str, ty: Expr) -> Expr {
        let mut metadata = Metadata::default();
        metadata.replace(MetadataValue::Type(
            TypeSyntax::try_new(ty).expect("valid parameter type"),
        ));
        Expr::MetaExpr(
            MetaExpr {
                metadata,
                expr: Box::new(name(name_value)),
            },
            span(),
        )
    }

    #[test]
    fn grad_wrt_function_resolution_and_parameter_types_have_carrier_parity() {
        let successor_type = node(DeepTag::TPrim, vec![name("f32")]);
        let legacy_type = legacy(DeepTag::TPrim, vec![name("f32")]);
        let successor_fn = node(
            DeepTag::Fn,
            vec![
                node(
                    DeepTag::Params,
                    vec![
                        typed_param("x", successor_type.clone()),
                        typed_param("y", successor_type),
                    ],
                ),
                node(DeepTag::Var, vec![name("body")]),
            ],
        );
        let legacy_fn = legacy(
            DeepTag::Fn,
            vec![
                legacy(
                    DeepTag::Params,
                    vec![
                        typed_param("x", legacy_type.clone()),
                        typed_param("y", legacy_type),
                    ],
                ),
                legacy(DeepTag::Var, vec![name("body")]),
            ],
        );
        let successor_transform = node(
            DeepTag::Grad,
            vec![
                node(DeepTag::Var, vec![name("pair")]),
                node(DeepTag::Lit, vec![Expr::Atom(Atom::Int(1), span())]),
            ],
        );
        let legacy_transform = legacy(
            DeepTag::Grad,
            vec![
                legacy(DeepTag::Var, vec![name("pair")]),
                legacy(DeepTag::Lit, vec![Expr::Atom(Atom::Int(1), span())]),
            ],
        );
        let successor_defs = UnordMap::from_iter([("pair".to_string(), successor_fn.clone())]);
        let legacy_defs = UnordMap::from_iter([("pair".to_string(), legacy_fn.clone())]);

        assert_eq!(
            grad_wrt_indices_from_transform(&successor_transform).unwrap(),
            Some(vec![1])
        );
        assert_eq!(
            grad_wrt_indices_from_transform(&successor_transform).unwrap(),
            grad_wrt_indices_from_transform(&legacy_transform).unwrap()
        );
        assert!(resolve_transform_fn_for_formals(&successor_transform, &successor_defs).is_some());
        assert!(resolve_transform_fn_for_formals(&legacy_transform, &legacy_defs).is_some());
        assert_eq!(
            param_type_expr_at(&successor_fn, 1).and_then(extract_prim_from_type_expr),
            Some(Prim::F32)
        );
        assert_eq!(
            param_type_expr_at(&successor_fn, 1).and_then(extract_prim_from_type_expr),
            param_type_expr_at(&legacy_fn, 1).and_then(extract_prim_from_type_expr)
        );

        let successor_host_only = node(
            DeepTag::App,
            vec![node(DeepTag::Var, vec![name("tensor_scan")])],
        );
        let legacy_host_only = legacy(
            DeepTag::App,
            vec![legacy(DeepTag::Var, vec![name("tensor_scan")])],
        );
        assert_eq!(
            find_reachable_host_only_builtin_call(&successor_host_only, &UnordMap::new()),
            Some("tensor_scan".to_string())
        );
        assert_eq!(
            find_reachable_host_only_builtin_call(&successor_host_only, &UnordMap::new()),
            find_reachable_host_only_builtin_call(&legacy_host_only, &UnordMap::new())
        );
    }

    #[test]
    fn transform_readers_explicitly_decline_unrelated_carriers() {
        let malformed = Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Tag(DeepTag::Grad), span()),
                    name("not-metadata"),
                ],
            },
            span(),
        );
        let unrelated = [
            Expr::BareList(vec![name("pair")], span()),
            Expr::UnknownForm(Box::new(UnknownFormData {
                head: "future-transform".to_string(),
                meta: Metadata::default(),
                children: vec![name("pair")],
                span: span(),
            })),
            Expr::Atom(Atom::Int(1), span()),
            Expr::Map(Metadata::default(), span()),
            Expr::MetaExpr(
                MetaExpr {
                    metadata: Metadata::default(),
                    expr: Box::new(name("pair")),
                },
                span(),
            ),
            malformed,
        ];
        let defs = UnordMap::new();
        for carrier in unrelated {
            assert!(grad_wrt_indices_from_transform(&carrier).is_err());
            assert!(resolve_transform_fn_for_formals(&carrier, &defs).is_none());
            assert_eq!(param_type_expr_at(&carrier, 0), None);
        }
    }
}
