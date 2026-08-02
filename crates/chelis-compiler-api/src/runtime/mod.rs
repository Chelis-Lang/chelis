use chelis_deep::DeepTag;
use std::collections::{HashMap, HashSet};

use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_ir::lower::top_level_lowering_map;
use chelis_types::{CheckedProgram, types::Prim};

use crate::schema::{DictEntryValue, ExecutionValue, TensorElements, TensorValue};

mod csv;
mod eval;
mod host_ops;
mod invariant;
mod json;
mod named_axis;
#[cfg(test)]
mod tests;
mod transforms;

pub(crate) use host_ops::collect_adt_ctor_fields;
// The [05-OBS-1] single renderer: compiler.rs uses it to pre-render each
// evaluated root's display text while the dtype tags still exist (the wire
// schema's `ExecutionValue` does not carry them), so the CLI's labeled-root
// exit shares the transcript exit's renderer byte-for-byte (chelis#732 P1).
pub(crate) use host_ops::render_value;
// Decode-boundary invariant revalidation surface (RFC D-DECODE). The
// `crate::decode` chokepoint imports these as `crate::runtime::<name>`.
pub(crate) use invariant::{
    DecodeField, DecodeFieldType, InvariantEntry, collect_ctor_field_types,
    collect_type_invariants, collect_zero_arg_constants, revalidate_adt_value,
};
use transforms::extract_prim_from_type_expr;

#[derive(Debug, Clone)]
pub struct RuntimeTensorValue {
    pub(crate) value: IrTensorValue,
    pub(crate) precision: Prim,
}

impl RuntimeTensorValue {
    /// Wrap finalized storage; the precision tag IS the storage dtype
    /// (chelis#729 Phase 1: the tag can no longer disagree with the
    /// buffer, which was the probe-2 F32-pin defect class).
    pub(crate) fn new(value: IrTensorValue) -> Self {
        Self {
            precision: value.prim(),
            value,
        }
    }

    /// Compute-op constructor: finalize a wide f64 buffer at `prim`
    /// (routes through the dtype-semantics module; integer widths trap
    /// on overflow, floats round once at width).
    pub(crate) fn from_wide(
        op: &'static str,
        prim: Prim,
        shape: Vec<usize>,
        wide: Vec<f64>,
    ) -> Result<Self, String> {
        IrTensorValue::finalize_from_wide(op, prim, shape, wide).map(Self::new)
    }

    /// Exact-integer compute-op constructor for paths that computed in
    /// i64.
    pub(crate) fn from_wide_int(
        op: &'static str,
        prim: Prim,
        shape: Vec<usize>,
        wide: Vec<i64>,
    ) -> Result<Self, String> {
        IrTensorValue::finalize_from_wide_int(op, prim, shape, wide).map(Self::new)
    }
}

/// Kind of transform captured by [`RuntimeValue::Transform`].
///
/// Bucket 1 closure: the host runtime needs to honor `grad`, `vmap`, and
/// `realize` so `chelis test`/`chelis eval` agree with the C backend on
/// programs that pass `chelis check`. `realize` is identity in the host
/// lane; `Grad` and `Vmap` capture the inner `(grad/vmap ...)` Deep form
/// and resolve at application time by routing through
/// [`chelis_ir::lower::lower_subexpr_program`] + the forward DAG
/// evaluator — the same machinery the C backend uses.
#[derive(Debug, Clone)]
pub enum TransformKind {
    /// `(grad {wrt: ...} fn-expr [index-expr])` — reverse-mode autodiff.
    Grad,
    /// `(vmap {} fn-expr axis-lit)` — vectorize the leading axis (or
    /// the explicit axis from the trailing literal).
    Vmap,
}

/// Sealed payload for [`RuntimeValue::Scalar`] (WS-A0 RT-1 fixup C1;
/// chelis#729 Phase 1 rebased it onto the dtype-semantics module).
///
/// The payload wraps `chelis_types::dtype_semantics::ScalarValue`, whose
/// storage variant IS the dtype, so a dtype/bits mismatch is
/// unrepresentable and every construction runs through the module's
/// `finalize_scalar` / `scalar_from_*` chokepoints (the section C3
/// privacy contract). The former in-crate `ScalarBits` enum and its
/// wrapping `from_f64_as` / `from_i64_as` raw constructors are deleted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarPayload {
    value: chelis_types::ScalarValue,
}

impl ScalarPayload {
    /// Wrap a module-finalized scalar.
    pub(crate) fn from_value(value: chelis_types::ScalarValue) -> Self {
        Self { value }
    }

    /// Read the source-level dtype (the storage variant's own dtype).
    pub(crate) fn dtype(&self) -> Prim {
        self.value.prim()
    }

    /// The sealed scalar itself, for finalize-driven flows and the
    /// observation channel (`element_ref`).
    pub(crate) fn value(&self) -> chelis_types::ScalarValue {
        self.value
    }

    /// View any numeric scalar as f64. Exact for every float width and
    /// for integers up to 2^53; int64 may lose precision past 2^53 (the
    /// named-lossy read of the dtype-semantics contract).
    pub(crate) fn as_f64_lossy(&self) -> f64 {
        self.value.as_f64_lossy()
    }

    /// View an integer scalar as exact i64. Every caller first establishes
    /// the integer-family guard; a float reaching this helper is an internal
    /// invariant violation, never permission to truncate.
    pub(crate) fn as_i64(&self) -> i64 {
        self.value
            .as_i64_exact()
            .expect("ScalarPayload::as_i64 requires an integer-family dtype")
    }
}

#[derive(Debug, Clone)]
pub enum RuntimeValue {
    Tensor(RuntimeTensorValue),
    /// First-class numeric scalar tagged with its source-level dtype.
    /// Construction must go through [`RuntimeValue::scalar`] (or one of
    /// the convenience constructors) so the `dtype`-vs-`bits` invariant
    /// holds. The variant is now a tuple over the sealed
    /// [`ScalarPayload`] (C1) so struct-literal initialization can no
    /// longer bypass the invariant. See `spec/04-type-system.md` §1.1.
    Scalar(ScalarPayload),
    Bool(bool),
    String(String),
    List(Vec<RuntimeValue>),
    Dict(Vec<(RuntimeValue, RuntimeValue)>),
    Tuple(Vec<RuntimeValue>),
    Adt {
        ctor: String,
        /// Field values in DECLARED order (the deftype's field order),
        /// not source or alphabetical order.
        fields: Vec<RuntimeValue>,
        /// When present, aligned index-for-index with `fields`, so it
        /// also follows declared order. `eval_record` enforces this
        /// (chelis#520 fixed a misalignment where kv source order was
        /// stored against declared-order `fields`).
        field_names: Option<Vec<String>>,
    },
    MappedFile(Vec<u8>),
    Closure {
        params: Vec<String>,
        /// Declared Deep type expression per param, when the `(fn ...)`
        /// carried checker-annotated `{type: ...}` param metadata.
        /// Consulted by the chelis#338 named-axis routing to recover a
        /// frame binding's static tensor type (named dims) at eval time.
        param_types: Vec<Option<Expr>>,
        body: Expr,
        env: HashMap<String, RuntimeValue>,
    },
    /// A captured `grad(f)` / `vmap(f)` waiting to be applied to args. The
    /// `transform_expr` holds the original `(grad ...)` or `(vmap ...)`
    /// Deep form so we can re-emit it as the callee in a synthesized
    /// `(app ...)` expression at apply time. `captured_env` snapshots the
    /// host-runtime bindings active when the transform was constructed so
    /// references to local closures (e.g. `target = fn (...) -> ...; grad(target)`)
    /// still resolve once the synthesized DAG is lowered.
    Transform {
        kind: TransformKind,
        transform_expr: Expr,
        captured_env: HashMap<String, RuntimeValue>,
    },
    Unit,
}

/// Debug-render a runtime value for an error message, truncating huge
/// payloads (tensors, long lists) so a shape diagnostic stays readable
/// instead of dumping the whole value (chelis#903 review). Values whose
/// debug form fits the cap render byte-identically to `{value:?}`, so
/// small-value diagnostics read as before. Used by the chelis#890/#903
/// JSON/CSV runtime modules only -- the crate-wide `expect_*_arg`
/// diagnostics keep their existing rendering.
pub(crate) fn truncated_debug(value: &RuntimeValue) -> String {
    truncate_rendered(format!("{value:?}"))
}

/// String-level half of [`truncated_debug`], for call sites that render
/// something other than a single value (e.g. an ADT's field list).
pub(crate) fn truncate_rendered(full: String) -> String {
    const MAX_LEN: usize = 160;
    if full.len() <= MAX_LEN {
        return full;
    }
    let mut cut = MAX_LEN;
    while !full.is_char_boundary(cut) {
        cut -= 1;
    }
    format!(
        "{}... ({} more bytes elided)",
        &full[..cut],
        full.len() - cut
    )
}

impl RuntimeValue {
    /// Wrap a module-finalized scalar (the WS-A0 invariant holds by
    /// construction: the storage variant IS the dtype).
    pub(crate) fn from_scalar_value(value: chelis_types::ScalarValue) -> Self {
        RuntimeValue::Scalar(ScalarPayload::from_value(value))
    }

    /// Default-narrowed integer literal per spec §5.3: bare integer
    /// values default to `int32` unless the surrounding context says
    /// otherwise. The checker range-guards literals (spec §5.6), so the
    /// ingress constructor cannot trap here.
    pub(crate) fn int_lit(value: i64) -> Self {
        Self::from_scalar_value(
            chelis_types::scalar_from_i64("literal", Prim::Int32, value)
                .expect("checker range-guards int literals at their dtype"),
        )
    }

    /// Default-narrowed float literal per spec §5.3: bare float values
    /// default to `f32` (finalize applies the f32 rounding).
    pub(crate) fn float_lit(value: f64) -> Self {
        Self::from_scalar_value(
            chelis_types::scalar_from_f64("literal", Prim::F32, value)
                .expect("float finalize is total"),
        )
    }

    /// Computed integer value preserving full i64 precision (e.g. `len`,
    /// shape sizes, parsed `to_int` results). Carries dtype `int64`.
    pub(crate) fn int64(value: i64) -> Self {
        Self::from_scalar_value(
            chelis_types::scalar_from_i64("int64", Prim::Int64, value)
                .expect("int64 ingress from i64 is total"),
        )
    }

    /// Computed float value preserving full f64 precision (e.g. `to_float`
    /// parse results, scalar reductions over f64 tensors). Carries dtype
    /// `f64`.
    pub(crate) fn float64(value: f64) -> Self {
        Self::from_scalar_value(
            chelis_types::scalar_from_f64("float64", Prim::F64, value)
                .expect("f64 finalize is total"),
        )
    }

    /// Construct a scalar at the dtype of an existing scalar from an
    /// exact i64 wide value. Finalize applies the section C1 rule, so an
    /// out-of-width integer TRAPS (chelis#680's decided contract)
    /// instead of wrapping through the deleted `from_i64_as` cast.
    pub(crate) fn scalar_like_int(template_dtype: Prim, value: i64) -> Result<Self, String> {
        chelis_types::scalar_from_i64("arithmetic", template_dtype, value)
            .map(Self::from_scalar_value)
            .map_err(|trap| trap.to_string())
    }

    pub(crate) fn scalar_like_float(template_dtype: Prim, value: f64) -> Result<Self, String> {
        chelis_types::scalar_from_f64("arithmetic", template_dtype, value)
            .map(Self::from_scalar_value)
            .map_err(|trap| trap.to_string())
    }

    /// True for any [`RuntimeValue::Scalar`] whose dtype is integer-typed
    /// per `Prim::is_integer`. Used by dispatch sites that previously
    /// matched `RuntimeValue::Int(_)`.
    #[allow(
        dead_code,
        reason = "downstream wave-2 will route through these classification helpers"
    )]
    pub(crate) fn is_int_scalar(&self) -> bool {
        matches!(self, RuntimeValue::Scalar(payload) if payload.dtype().is_integer())
    }

    /// True for any [`RuntimeValue::Scalar`] whose dtype is float-typed
    /// per `Prim::is_float`. Used by dispatch sites that previously
    /// matched `RuntimeValue::Float(_)`.
    #[allow(
        dead_code,
        reason = "downstream wave-2 will route through these classification helpers"
    )]
    pub(crate) fn is_float_scalar(&self) -> bool {
        matches!(self, RuntimeValue::Scalar(payload) if payload.dtype().is_float())
    }

    /// View this value as i64 if it is an integer-typed scalar.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => Some(payload.as_i64()),
            _ => None,
        }
    }

    /// View this value as f64 if it is a float-typed scalar. Mirrors
    /// `as_i64` for the float row.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            RuntimeValue::Scalar(payload) if payload.dtype().is_float() => {
                Some(payload.as_f64_lossy())
            }
            _ => None,
        }
    }

    /// View this value as a bool if it is a [`RuntimeValue::Bool`].
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            RuntimeValue::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Borrow this value as an ADT `(ctor, fields)` pair if it is one.
    /// Read accessor for the decode chokepoint's consumers; field names are
    /// available via the public `RuntimeValue::Adt { field_names, .. }`
    /// variant binding when needed.
    pub fn as_adt(&self) -> Option<(&str, &[RuntimeValue])> {
        match self {
            RuntimeValue::Adt { ctor, fields, .. } => Some((ctor.as_str(), fields.as_slice())),
            _ => None,
        }
    }

    /// Re-encode this value to the machine-facing [`ExecutionValue`] wire
    /// shape. The inverse direction of the decode chokepoint: a value the
    /// chokepoint produced from an `ExecutionValue::Adt` re-encodes to a
    /// value-identical `ExecutionValue::Adt`, so the conformance suite can
    /// assert the decode round-trips its input bit-for-bit. Errors only for
    /// values with no wire shape (mapped files), which decode never yields.
    pub fn to_execution_value(&self) -> Result<ExecutionValue, String> {
        runtime_value_to_schema(self)
    }
}

#[derive(Debug, Default)]
pub(crate) struct RuntimeOutcome {
    pub(crate) host_bindings: HashMap<String, RuntimeValue>,
    /// Applied values of host-lane *zero-argument fn* top-level roots
    /// (the desugared shape of the arrow-form `def name -> T = body`,
    /// which the Surf desugarer wraps as `(def name (fn () body))`).
    /// Such a def is a nullary thunk of type `() -> T`: it is a display
    /// root whose value is the result of APPLYING it, but it is NOT a
    /// value binding, so it never lands in `host_bindings` (the
    /// call-resolution frame — binding the applied tensor there under
    /// the bare name would shadow the callable and break any `name()`
    /// call). Keyed by bare root name; consulted by `eval_compiled` as a
    /// fallback after `lookup_runtime_value_for_root`.
    pub(crate) host_root_values: HashMap<String, RuntimeValue>,
    pub(crate) transcript: Vec<String>,
}

#[cfg(test)]
pub(crate) fn evaluate_host_program(
    program: &CheckedProgram,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
) -> Result<RuntimeOutcome, String> {
    evaluate_host_program_filtered(program, tensor_bindings, None)
}

/// Evaluate top-level non-fn bindings. When `selected_roots` is `Some`, only
/// bindings whose names appear in the filter are *eagerly* evaluated. Other
/// top-level bindings stay registered in `top_level_defs` so the body of a
/// selected binding can lazily resolve references to them via
/// `resolve_top_level`. This is what lets `chelis test` share a single
/// compile across every test in a file: compile once with N synthesized
/// `__chelis_test_k = test_k()` bindings, then run N eval passes each
/// selecting one root — without each pass paying for the other N-1 tests
/// running as module init.
pub(crate) fn evaluate_host_program_filtered(
    program: &CheckedProgram,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
    selected_roots: Option<&[String]>,
) -> Result<RuntimeOutcome, String> {
    evaluate_host_program_with_library(program, &[], None, tensor_bindings, selected_roots)
}

/// Phase G' — host-runtime entry that seeds the `top_level_defs` table
/// with library defs in addition to the new-code program. This is the
/// host-side parity counterpart to `lower_program_with_context`: when
/// new code calls a library function (e.g. `Std.Time.is_leap_year`),
/// `eval_app` looks up that name through `lookup_top_level_def`, and
/// the function body must be reachable. Pre-Phase-G' the runtime only
/// saw `program.exprs()`, so library names errored as `unknown runtime
/// name`.
///
/// Library defs are registered FIRST, then new-code defs, so on a name
/// collision the new-code def shadows the library def — mirroring the
/// type-env stacking semantics in `check_ir_with_context`.
///
/// `library_lowered_names` is the optional library-side
/// lowered-vs-host classification, threaded through so a library def
/// that the lowering pass identifies as "lives in the tensor DAG, not
/// in the host runtime" stays out of the host runtime's eager-eval
/// list. The new code's lowering map (computed locally below) merges
/// on top.
pub(crate) fn evaluate_host_program_with_library(
    program: &CheckedProgram,
    library_exprs: &[Expr],
    library_lowered_names: Option<&HashMap<String, bool>>,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
    selected_roots: Option<&[String]>,
) -> Result<RuntimeOutcome, String> {
    evaluate_host_program_with_library_and_types(
        program,
        library_exprs,
        &HashMap::new(),
        library_lowered_names,
        tensor_bindings,
        selected_roots,
    )
}

/// Variant of [`evaluate_host_program_with_library`] that also takes the
/// library's Deep type-env. Bucket 1 (`grad`/`vmap`/`realize` in the host
/// runtime) needs the merged type-env so the IR
/// `lower_subexpr_program` call resolves library-name free vars in the
/// inner fn body the same way the C backend does.
pub(crate) fn evaluate_host_program_with_library_and_types(
    program: &CheckedProgram,
    library_exprs: &[Expr],
    library_type_env: &HashMap<String, Expr>,
    library_lowered_names: Option<&HashMap<String, bool>>,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
    selected_roots: Option<&[String]>,
) -> Result<RuntimeOutcome, String> {
    // Lowered classification. A new-code value binding that references a
    // library function (e.g. `imported_val = lib_add(20, 22)`) must
    // inherit that function's host-lane-vs-tensor-lane classification —
    // otherwise the bare `top_level_lowering_map(program.exprs())` walk,
    // which has no view of the library's defs, sees `lib_add` as an
    // unknown name and mis-classifies the binding as lowered. That bug
    // (Chelis-Lang/chelis#423) drops the binding from BOTH lanes: the
    // host-order filter below skips it (it looks lowered) while the
    // tensor-root computation in `compile_new_source_in_context`
    // correctly excludes it (its dependency is host-lane), so it
    // disappears from eval output entirely. Classify over the COMBINED
    // library + new-code exprs (with the composed type-env) so both
    // sides agree, then keep the library's own precomputed entries on
    // top for names the combined walk doesn't cover.
    let mut combined_exprs: Vec<Expr> = library_exprs.to_vec();
    combined_exprs.extend(program.exprs().iter().cloned());
    let mut combined_type_env: HashMap<String, Expr> = library_type_env.clone();
    for (name, ty_expr) in program.type_env() {
        combined_type_env.insert(name.clone(), ty_expr.clone());
    }
    let new_lowered_names = top_level_lowering_map(&combined_exprs, &combined_type_env);
    let mut lowered_names: HashMap<String, bool> = HashMap::new();
    if let Some(lib) = library_lowered_names {
        lowered_names.extend(lib.iter().map(|(k, v)| (k.clone(), *v)));
    }
    lowered_names.extend(new_lowered_names);

    // ADT field map covers both library and new-code constructors so a
    // record-pattern match on a library ADT in new code resolves field
    // names correctly.
    let mut adt_fields = collect_adt_ctor_fields(library_exprs);
    adt_fields.extend(collect_adt_ctor_fields(program.exprs()));
    let adt_grad_rejections =
        host_ops::collect_adt_grad_rejections(&[library_exprs, program.exprs()]);

    let mut top_level_defs = HashMap::new();
    let mut top_level_order = Vec::new();

    // Register library defs FIRST. New-code defs will overwrite on
    // name collision below — matching the Phase C type-env shadow rule
    // (new code wins).
    register_top_level_defs(
        library_exprs,
        &lowered_names,
        selected_roots,
        &mut top_level_defs,
        &mut top_level_order,
        /* register_runtime_order = */ false,
    );
    // Register new-code defs. New-code is the only source of eager
    // module-init bindings in `top_level_order` — library was already
    // checked + lowered at context-build time and any side effects
    // would have happened then; re-running them on every per-test
    // worker is exactly the regression we're fixing.
    register_top_level_defs(
        program.exprs(),
        &lowered_names,
        selected_roots,
        &mut top_level_defs,
        &mut top_level_order,
        /* register_runtime_order = */ true,
    );

    // Compose the runtime's type-env from library + new-code program type
    // envs. New code wins on shadow, mirroring `compose_type_env` semantics.
    // We need this for grad/vmap/realize routing through
    // `lower_subexpr_program`: the IR lowerer's `lower_subexpr_program`
    // resolves free names against `full_type_env`.
    let mut type_env: HashMap<String, Expr> = library_type_env.clone();
    for (name, ty_expr) in program.type_env() {
        type_env.insert(name.clone(), ty_expr.clone());
    }

    let mut ctx = EvalContext {
        bindings: HashMap::new(),
        binding_types: HashMap::new(),
        named_axis_route_cache: HashMap::new(),
        named_axis_route_visiting: HashSet::new(),
        top_level_defs,
        type_env,
        adt_fields,
        adt_grad_rejections,
        tensor_bindings,
        transcript: Vec::new(),
        resolving_top_levels: Vec::new(),
        random_seed: None,
        random_counter: 0,
        cancel: chelis_types::current_cancel_token(),
    };

    for name in top_level_order {
        let _ = ctx.resolve_top_level(&name)?;
    }

    // Surface host-lane *zero-argument fn* roots. The arrow-form
    // `def priced -> T = body` desugars to `(def priced (fn () body))`,
    // a nullary thunk of type `() -> T`. `register_top_level_defs`'
    // `is_fn` guard skips it from the eager value-binding order (it looks
    // like a function), yet `root_names_from_checked_exprs` lists it as a
    // display root (unwrapping the declared `t-fn` return type). Without
    // this pass the root has no host binding and `eval_compiled` drops it
    // (`lookup_runtime_value_for_root` → None → filter_map), so an
    // in-context library call like `bs_call_f64_vector(...)` returned
    // `{"roots":[]}` instead of its value (chelis blocker2).
    //
    // We evaluate the value here by APPLYING the thunk (zero args) rather
    // than binding it in `ctx.bindings`, so the call-resolution frame is
    // untouched and a `name()` call elsewhere still resolves the callable.
    // Only host-lane roots are applied — a tensor-lane (lowered) nullary
    // def surfaces through the DAG/`tensor_bindings` path instead. Apply
    // failures are swallowed (the root stays unsurfaced, exactly as
    // before) so this can only add values, never regress.
    let mut host_root_values = HashMap::new();
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some(DeepTag::Def) {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        // Zero-argument fn wrapper only: `(fn (params-empty) inner)`.
        let Expr::List(fn_list, _) = body else {
            continue;
        };
        let is_zero_arg_fn = tag(fn_list) == Some(DeepTag::Fn)
            && matches!(
                children(fn_list).first(),
                Some(Expr::List(params, _))
                    if tag(params) == Some(DeepTag::Params)
                        && children(params).is_empty()
            );
        if !is_zero_arg_fn {
            continue;
        }
        // Effect-free guard (chelis blocker2 red-team). Only surface a
        // root whose body carries NO effect row. The effects checker
        // (`chelis_effects::check_effects_with_context`, run in
        // `compile_new_source_in_context` and threaded through library
        // context) stamps a non-empty `(effects ...)` node under the
        // `"effects"` meta key on this `fn` wrapper when — and only when —
        // the body has a latent effect. Applying such a thunk here would
        // RUN that effect at display time (1× where the host runtime
        // otherwise runs it 0×), and 2× when the root is also consumed by
        // a `name()` call. A root that carries any effect row therefore
        // stays unsurfaced: status quo, no regression, no display-time
        // effect. Purity is decided by the checker's annotation, not by
        // re-inferring here, so a library-inherited effect is honored too.
        if carries_effect_row(fn_list) {
            continue;
        }
        // Host-lane only; tensor-lane roots come through the DAG.
        if lowered_names.get(name).copied().unwrap_or(false) {
            continue;
        }
        // Honor the same selected-roots filter the eager order uses.
        let selected = match selected_roots {
            None => true,
            Some(filter) => filter.iter().any(|s| {
                s == name
                    || s.strip_prefix(name)
                        .is_some_and(|rest| rest.starts_with('.'))
            }),
        };
        if !selected {
            continue;
        }
        // If a host binding for this name already exists, the
        // `lookup_runtime_value_for_root` path already surfaces it and
        // this fallback would be dead. Skip so we do NOT re-apply the
        // thunk a second time: another root may have called `name()`,
        // binding its closure here, and a second application would re-run
        // any effects in the body (e.g. `print`/`debug`) and waste the
        // whole computation, whose value is then discarded anyway
        // (chelis blocker2 red-team: double-execution of body effects).
        if ctx.bindings.contains_key(name) {
            continue;
        }
        // Apply the thunk: build the closure, call it with no args.
        let applied = ctx
            .eval_expr(body)
            .and_then(|closure| ctx.apply_resolved_callable(closure, Vec::new()));
        if let Ok(value) = applied {
            host_root_values.insert(name.to_string(), value);
        }
    }

    Ok(RuntimeOutcome {
        host_bindings: ctx.bindings,
        host_root_values,
        transcript: ctx.transcript,
    })
}

fn register_top_level_defs(
    exprs: &[Expr],
    lowered_names: &HashMap<String, bool>,
    selected_roots: Option<&[String]>,
    top_level_defs: &mut HashMap<String, Expr>,
    top_level_order: &mut Vec<String>,
    register_runtime_order: bool,
) {
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some(DeepTag::Def) {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        top_level_defs.insert(name.to_string(), body.clone());
        if !register_runtime_order {
            continue;
        }
        let is_fn = matches!(body, Expr::List(body_list, _) if tag(body_list) == Some(DeepTag::Fn));
        if !is_fn && !lowered_names.get(name).copied().unwrap_or(false) {
            // chelis#614: a tuple- or ADT-valued binding `out = ...` owns
            // FLATTENED root names (`out.0`, `out.1`, ...) in the caller's
            // `selected_roots` filter, but the def itself is named `out`.
            // Match the base def name against those flattened entries by
            // prefix so the host runtime eager-evaluates `out` and binds
            // the whole tuple/ADT; the display path then walks into it per
            // component. Without this, a multi-target grad binding is never
            // evaluated and its roots come back empty (the misleading
            // "input contains only def declarations" breadcrumb).
            let selected = match selected_roots {
                None => true,
                Some(filter) => filter.iter().any(|s| {
                    s == name
                        || s.strip_prefix(name)
                            .is_some_and(|rest| rest.starts_with('.'))
                }),
            };
            if selected {
                top_level_order.push(name.to_string());
            }
        }
    }
}

/// Compute a lowered-vs-host classification map for a slice of
/// library exprs, using its own type-env. Phase G' threads this from
/// the `CompiledContext`'s `library_checked` into the host runtime so
/// the new-code lowering map merges with library state instead of
/// re-deriving the wrong answer for library names that shadow
/// builtins.
pub(crate) fn library_lowered_names(
    library_exprs: &[Expr],
    library_type_env: &HashMap<String, Expr>,
) -> HashMap<String, bool> {
    top_level_lowering_map(library_exprs, library_type_env)
}

fn top_level_items(exprs: &[Expr]) -> Vec<&Expr> {
    let mut out = Vec::new();
    for expr in exprs {
        collect_top_level_items(expr, &mut out);
    }
    out
}

fn collect_top_level_items<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    match expr {
        Expr::List(list, _) if tag(list) == Some(DeepTag::Module) => {
            for child in list.elements.iter().skip(3) {
                collect_top_level_items(child, out);
            }
        }
        Expr::Node(node, _) if node.tag() == DeepTag::Module => {
            for child in node.children_slice().iter().skip(1) {
                collect_top_level_items(child, out);
            }
        }
        _ => out.push(expr),
    }
}

pub(crate) fn runtime_value_to_schema(value: &RuntimeValue) -> Result<ExecutionValue, String> {
    Ok(match value {
        RuntimeValue::Tensor(tensor) => {
            // Exact per-dtype egress (execution wire v2, chelis#729
            // section C3): the storage view carries every element at its
            // own width, so int64 crosses the wire exactly and every
            // dtype keeps its tag.
            let data = match tensor.value.storage().view() {
                chelis_types::StorageView::F64(v) => TensorElements::F64(v.to_vec()),
                chelis_types::StorageView::F32(v) => TensorElements::F32(v.to_vec()),
                chelis_types::StorageView::F16(v) => {
                    TensorElements::F16(v.iter().map(|&x| f64::from(x)).collect())
                }
                chelis_types::StorageView::Bf16(v) => {
                    TensorElements::Bf16(v.iter().map(|&x| f64::from(x)).collect())
                }
                chelis_types::StorageView::I64(v) => TensorElements::Int64(v.to_vec()),
                chelis_types::StorageView::I32(v) => TensorElements::Int32(v.to_vec()),
                chelis_types::StorageView::I16(v) => TensorElements::Int16(v.to_vec()),
                chelis_types::StorageView::I8(v) => TensorElements::Int8(v.to_vec()),
                chelis_types::StorageView::Bool(v) => {
                    TensorElements::Bool(v.iter().map(|&x| x != 0).collect())
                }
            };
            ExecutionValue::Tensor {
                value: TensorValue {
                    shape: tensor.value.shape.clone(),
                    data,
                },
            }
        }
        RuntimeValue::Scalar(payload) => match payload.value().element_ref() {
            chelis_types::ElementRef::I8(value) => ExecutionValue::Int8 { value },
            chelis_types::ElementRef::I16(value) => ExecutionValue::Int16 { value },
            chelis_types::ElementRef::I32(value) => ExecutionValue::Int32 { value },
            chelis_types::ElementRef::I64(value) => ExecutionValue::Int64 { value },
            chelis_types::ElementRef::F16(value) => ExecutionValue::Float16 {
                value: f64::from(value),
            },
            chelis_types::ElementRef::Bf16(value) => ExecutionValue::Bfloat16 {
                value: f64::from(value),
            },
            chelis_types::ElementRef::F32(value) => ExecutionValue::Float32 { value },
            chelis_types::ElementRef::F64(value) => ExecutionValue::Float64 { value },
            chelis_types::ElementRef::Bool(_) => {
                return Err(
                    "bool ScalarValue unexpectedly reached the numeric RuntimeValue::Scalar wire path"
                        .to_string(),
                );
            }
        },
        RuntimeValue::Bool(value) => ExecutionValue::Bool { value: *value },
        RuntimeValue::String(value) => ExecutionValue::String {
            value: value.clone(),
        },
        RuntimeValue::List(items) => ExecutionValue::List {
            value: items
                .iter()
                .map(runtime_value_to_schema)
                .collect::<Result<Vec<_>, _>>()?,
        },
        RuntimeValue::Dict(entries) => ExecutionValue::Dict {
            entries: entries
                .iter()
                .map(|(key, value)| {
                    Ok(DictEntryValue {
                        key: runtime_value_to_schema(key)?,
                        value: runtime_value_to_schema(value)?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        },
        RuntimeValue::Tuple(items) => ExecutionValue::Tuple {
            value: items
                .iter()
                .map(runtime_value_to_schema)
                .collect::<Result<Vec<_>, _>>()?,
        },
        RuntimeValue::Adt { ctor, fields, .. } => ExecutionValue::Adt {
            // De-mangle the reef-linked `Pkg__..__Ctor` form to the bare,
            // user-facing constructor name. This is the eval `--json` ABI
            // surface; decode (`decode_adt_value`) already keys on bare
            // constructor names, so emitting bare here makes the encode/decode
            // round-trip consistent and stops internal mangling leaking to
            // consumers (chelis#399).
            ctor: chelis_types::demangle_ident(ctor),
            fields: fields
                .iter()
                .map(runtime_value_to_schema)
                .collect::<Result<Vec<_>, _>>()?,
        },
        RuntimeValue::MappedFile(_) => {
            return Err(
                "MappedFile values are not serializable on machine-facing APIs".to_string(),
            );
        }
        RuntimeValue::Closure { .. } => ExecutionValue::String {
            value: "<closure>".to_string(),
        },
        RuntimeValue::Transform { kind, .. } => ExecutionValue::String {
            value: match kind {
                TransformKind::Grad => "<grad>".to_string(),
                TransformKind::Vmap => "<vmap>".to_string(),
            },
        },
        RuntimeValue::Unit => ExecutionValue::Unit,
    })
}

pub(crate) fn lookup_runtime_value_for_root(
    name: &str,
    host_bindings: &HashMap<String, RuntimeValue>,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
) -> Option<RuntimeValue> {
    if let Some(value) = tensor_bindings.get(name) {
        return Some(RuntimeValue::Tensor(value.clone()));
    }

    let mut parts = name.split('.');
    let head = parts.next()?;
    let mut value = host_bindings.get(head)?.clone();
    for part in parts {
        value = match value {
            // Tuple components are keyed by positional index.
            RuntimeValue::Tuple(items) => {
                let index = part.parse::<usize>().ok()?;
                items.into_iter().nth(index)?
            }
            // chelis#614/#520 D2: an ADT-valued component (e.g. the
            // params slot of a multi-target `grad` result, or a field-wise
            // gradient struct) is keyed by field NAME for a record
            // constructor and by positional index for a positional one.
            // Descend into it the same way `flatten_binding_into` /
            // `add_named_roots` build the dotted key, so a nested ADT root
            // reconstructs its value instead of being silently dropped.
            RuntimeValue::Adt {
                fields,
                field_names,
                ..
            } => {
                let index = field_names
                    .as_ref()
                    .and_then(|names| names.iter().position(|n| n == part))
                    .or_else(|| part.parse::<usize>().ok())?;
                fields.into_iter().nth(index)?
            }
            _ => return None,
        };
    }
    Some(value)
}

struct EvalContext<'a> {
    bindings: HashMap<String, RuntimeValue>,
    /// Declared/static Deep type expression for names in `bindings`,
    /// maintained in lockstep with `bindings` (saved/swapped/restored at
    /// every frame boundary). Every locally-bound name gets a key here:
    /// `Some(ty)` when a declared or checker-annotated type is known,
    /// `None` otherwise. The explicit `None` marker matters: it masks a
    /// same-named top-level `type_env` entry so a local shadow is never
    /// typed with the outer binding's type (chelis#338 named-axis routing).
    binding_types: HashMap<String, Option<Expr>>,
    /// Memoized per-def result of [`Self::def_requires_named_axis_routing`].
    named_axis_route_cache: HashMap<String, bool>,
    /// Cycle guard for the recursive routing detection walk.
    named_axis_route_visiting: HashSet<String>,
    top_level_defs: HashMap<String, Expr>,
    /// Combined library + new-code Deep type-env. Threaded into
    /// [`chelis_ir::lower::lower_subexpr_program`] when the host runtime
    /// hits a `grad` / `vmap` form so the lowerer can resolve free names
    /// the same way the C backend does. Empty when no library context is
    /// present (e.g. unit tests that don't need transform support).
    type_env: HashMap<String, Expr>,
    adt_fields: HashMap<String, Vec<String>>,
    /// chelis#520 D2: constructor -> rejection reason for ADT types
    /// outside the field-wise gradient slice (mixed fields in any
    /// variant, or no fields at all). Consulted by the grad argument
    /// marshalling so the eval lane rejects exactly the arguments the
    /// checker types as non-differentiable, instead of fabricating a
    /// gradient value the static type does not admit.
    adt_grad_rejections: HashMap<String, String>,
    tensor_bindings: &'a HashMap<String, RuntimeTensorValue>,
    transcript: Vec<String>,
    resolving_top_levels: Vec<String>,
    random_seed: Option<u64>,
    random_counter: u64,
    /// Cooperative cancellation flag (chelis#914), captured ONCE from the
    /// thread-local install point at construction so the per-node-visit
    /// check in [`Self::eval_expr`] is a relaxed atomic load rather than a
    /// TLS lookup. `None` — the default when no caller installed a token —
    /// makes the check a single `Option` discriminant test.
    cancel: Option<chelis_types::CancelToken>,
}

fn tag(list: &List) -> Option<DeepTag> {
    list.tag()
}

fn get_meta(list: &List) -> Option<&MetaMap> {
    match list.elements.get(1) {
        Some(Expr::Map(map, _)) => Some(map),
        _ => None,
    }
}

/// True when the checked-program effect annotation on this node carries a
/// NON-EMPTY effect row. `chelis_effects`' `update_effect_metadata`
/// stamps an `(effects ...)` node under the `"effects"` meta key on a
/// `fn` node exactly when its inferred latent effect row is non-empty, so
/// a populated `effects` node is a sufficient, checker-authoritative
/// signal that the body is effectful. The host-root surfacing pass uses
/// this to keep effectful roots unsurfaced — an effectful zero-arg root
/// must not run its effect at display time.
fn carries_effect_row(list: &List) -> bool {
    let Some(meta) = get_meta(list) else {
        return false;
    };
    meta.entries.iter().any(|(key, value)| {
        key == "effects"
            && matches!(
                value,
                Expr::List(effects, _)
                    if tag(effects) == Some(DeepTag::Effects)
                        && !children(effects).is_empty()
            )
    })
}

/// Extract the primitive dtype written into a `(lit {type: ...})` meta
/// by the type checker, if any. Returns `None` for non-primitive type
/// metadata (e.g. tensor literal types) or missing metadata; eval_lit
/// then falls back to the spec §5.3 literal default.
fn lit_meta_prim(meta: &MetaMap) -> Option<Prim> {
    let (_, ty_expr) = meta.entries.iter().find(|(k, _)| k == "type")?;
    extract_prim_from_type_expr(ty_expr)
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn int_value(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Atom(Atom::Int(value), _) => Some(*value),
        _ => None,
    }
}

/// Read a *literal* seed at full i64 width, peeling `(lit {meta} …)`
/// wrappers down to the raw `Atom::Int`. Mirrors the compiled C host lane,
/// which reads the raw atom and ignores the int32 default meta (`host.rs`
/// `lower_host_expr`: the `lit` peel forwards to the `Atom::Int(i64)` arm).
///
/// chelis#771: routing a literal seed through `eval_lit` narrows it to the
/// spec/04-type-system.md §5.3 int32 default (int32-truncate then
/// sign-extend), so any seed `>= 2^31` becomes an unrelated `u64` in the
/// evaluator while the compiled lane keeps the full value — the two lanes
/// then sample completely different streams from the "same" seed. The seed
/// is designed int64 (spec/design/checker_totality.md §C1.5 item 5, the
/// int64-suffixed literal contract; #731 Phase 1's FORM gate is unshipped).
///
/// Returns `None` for non-literal (computed) seed expressions; those keep
/// the existing dtype-narrowing `eval_expr` path unchanged.
fn literal_seed_i64(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Atom(Atom::Int(value), _) => Some(*value),
        Expr::List(list, _) if tag(list) == Some(DeepTag::Lit) => {
            children(list).first().and_then(literal_seed_i64)
        }
        _ => None,
    }
}
