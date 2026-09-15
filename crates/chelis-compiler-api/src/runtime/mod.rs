use chelis_deep::DeepTag;
use chelis_unord::{UnordMap, UnordSet};
use std::collections::BTreeMap;

use chelis_deep::ast::{Atom, Expr, List, Metadata};
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_ir::lower::top_level_lowering_map;
use chelis_types::{
    CheckedProgram,
    manifest::{RootEntry, RootPathStep},
    types::Prim,
};

use crate::schema::{DictEntryValue, ExecutionValue, TensorValue};

mod csv;
mod eval;
mod host_ops;
mod invariant;
mod named_axis;
mod numeric_text;
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
        /// Declared result type from the owning `defsig`, when available.
        /// Generic cast actualization matches this against the checker-owned
        /// call expression result, including context-fixed empty containers.
        return_type: Option<Expr>,
        /// The checked function type retains renamed precision binders used
        /// by body metadata, alongside the signature's declared names.
        checked_signature: Option<Expr>,
        /// Authored higher-order formal signatures retained at each
        /// specialization boundary. These execute before this closure's own
        /// entry so a broader supplied callable cannot erase a narrower
        /// invocation contract.
        invocation_contracts: Box<Vec<Expr>>,
        body: Expr,
        env: UnordMap<String, RuntimeValue>,
        /// Lexically captured concrete precision variables. A call derives a
        /// fresh specialization from checked argument/result types and lets
        /// the callee's own binders shadow same-spelled outer binders.
        precision_env: UnordMap<String, Prim>,
        /// chelis#1277 B2h: the top-level def this closure is the body of,
        /// when it is one. Stamped where a def body resolves to its closure
        /// (`stamp_def_closure`), never on a local `fn` literal, so applying
        /// the closure can consult the kernel decision the C lane makes for
        /// that def (`chelis_ir::host::host_def_kernel`).
        def_name: Option<String>,
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
        captured_env: UnordMap<String, RuntimeValue>,
        /// Authored higher-order formal signatures retained at each
        /// specialization boundary, shared with ordinary closures so every
        /// supported runtime callable executes the same invocation protocol.
        invocation_contracts: Box<Vec<Expr>>,
    },
    Unit,
}

impl RuntimeValue {
    fn invocation_contracts(&self) -> Option<&[Expr]> {
        match self {
            Self::Closure {
                invocation_contracts,
                ..
            }
            | Self::Transform {
                invocation_contracts,
                ..
            } => Some(invocation_contracts),
            _ => None,
        }
    }

    fn invocation_contracts_mut(&mut self) -> Option<&mut Vec<Expr>> {
        match self {
            Self::Closure {
                invocation_contracts,
                ..
            }
            | Self::Transform {
                invocation_contracts,
                ..
            } => Some(invocation_contracts),
            _ => None,
        }
    }

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
    pub(crate) host_bindings: UnordMap<String, RuntimeValue>,
    /// Applied values of host-lane *zero-argument fn* top-level roots
    /// (the desugared shape of the arrow-form `def name() -> T = body`,
    /// which the Surf desugarer wraps as `(def name (fn () body))`).
    /// Such a def is a nullary thunk of type `() -> T`: it is a display
    /// root whose value is the result of APPLYING it, but it is NOT a
    /// value binding, so it never lands in `host_bindings` (the
    /// call-resolution frame — binding the applied tensor there under
    /// the bare name would shadow the callable and break any `name()`
    /// call). Keyed by bare root name; consulted by `eval_compiled` as a
    /// fallback after `lookup_runtime_value_for_root`.
    pub(crate) host_root_values: UnordMap<String, RuntimeValue>,
    /// Evaluation failures for those same applied nullary roots. Retained by
    /// root name so the manifest consumer can report them through [05-UNS-1]
    /// instead of either swallowing the cause or returning an unbranded host
    /// evaluator error.
    pub(crate) host_root_errors: UnordMap<String, String>,
    pub(crate) transcript: Vec<String>,
}

#[cfg(test)]
pub(crate) fn evaluate_host_program(
    program: &CheckedProgram,
    tensor_bindings: &UnordMap<String, RuntimeTensorValue>,
) -> Result<RuntimeOutcome, String> {
    evaluate_host_program_filtered(program, tensor_bindings, None, None, None)
        .map_err(|failure| failure.message)
}

pub(crate) struct HostEvaluationInputs<'a> {
    pub(crate) roots: &'a UnordMap<String, RuntimeTensorValue>,
    pub(crate) bindings: Option<&'a UnordMap<String, IrTensorValue>>,
}

/// A failed execution still owes the effects it performed before unwinding.
#[derive(Debug)]
pub(crate) struct RuntimeFailure {
    pub(crate) message: String,
    pub(crate) transcript: Vec<String>,
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
    tensor_bindings: &UnordMap<String, RuntimeTensorValue>,
    selected_roots: Option<&[String]>,
    manifested_lowered_names: Option<&BTreeMap<String, bool>>,
    bound_evaluation_inputs: Option<&UnordMap<String, IrTensorValue>>,
) -> Result<RuntimeOutcome, RuntimeFailure> {
    evaluate_host_program_with_library_and_types(
        program,
        None,
        None,
        HostEvaluationInputs {
            roots: tensor_bindings,
            bindings: bound_evaluation_inputs,
        },
        selected_roots,
        manifested_lowered_names,
    )
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
/// on top. The function also takes the library's Deep type-env. Bucket 1
/// (`grad`/`vmap`/`realize` in the host
/// runtime) needs the merged type-env so the IR
/// `lower_subexpr_program` call resolves library-name free vars in the
/// inner fn body the same way the C backend does.
pub(crate) fn evaluate_host_program_with_library_and_types(
    program: &CheckedProgram,
    library: Option<&CheckedProgram>,
    library_lowered_names: Option<&BTreeMap<String, bool>>,
    inputs: HostEvaluationInputs<'_>,
    selected_roots: Option<&[String]>,
    manifested_lowered_names: Option<&BTreeMap<String, bool>>,
) -> Result<RuntimeOutcome, RuntimeFailure> {
    let HostEvaluationInputs {
        roots: tensor_bindings,
        bindings: bound_evaluation_inputs,
    } = inputs;
    let empty_types = BTreeMap::new();
    let library_exprs = library.map(CheckedProgram::exprs).unwrap_or(&[]);
    let library_type_env = library
        .map(CheckedProgram::type_env)
        .unwrap_or(&empty_types);
    // Looking up an imported function in the new-code-only program silently
    // interpreted it without the kernel's declared shape obligations.
    let kernel_program = library
        .map(|library| {
            CheckedProgram::compose(library, program)
                .ok_or_else(|| "runtime kernel program lost its checked library proof".to_owned())
        })
        .transpose()
        .map_err(|message| RuntimeFailure {
            message,
            transcript: Vec::new(),
        })?;
    // chelis#1829: the kernel-decision probe behind `def_kernel` expands the
    // call graph as a tree, so it must be derived once per definition for the
    // whole evaluation. Before #1693 this program held new code only, so an
    // imported name was not found and never probed; it now composes the
    // library in, so every imported definition takes that path. The session
    // `ctx` owns below is what holds those facts, and the borrow checker, not
    // a declaration order, is what keeps it inside `kernel_program`'s life.
    let eval_program = kernel_program.as_ref().unwrap_or(program);

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
    let mut combined_type_env: BTreeMap<String, Expr> = library_type_env.clone();
    for (name, ty_expr) in program.type_env() {
        combined_type_env.insert(name.clone(), ty_expr.clone());
    }
    let mut new_lowered_names = top_level_lowering_map(&combined_exprs, &combined_type_env);
    if let Some(manifested) = manifested_lowered_names {
        new_lowered_names.extend(
            manifested
                .iter()
                .map(|(name, lowered)| (name.clone(), *lowered)),
        );
    }
    let mut lowered_names: BTreeMap<String, bool> = BTreeMap::new();
    if let Some(lib) = library_lowered_names {
        lowered_names.extend(lib.iter().map(|(k, v)| (k.clone(), *v)));
    }
    lowered_names.extend(new_lowered_names);

    // ADT field map covers both library and new-code constructors so a
    // record-pattern match on a library ADT in new code resolves field
    // names correctly.
    let mut adt_fields = collect_adt_ctor_fields(library_exprs);
    adt_fields.merge(collect_adt_ctor_fields(program.exprs()));

    let mut top_level_defs = UnordMap::new();
    let mut top_level_order = Vec::new();
    let mut declared_signatures = UnordMap::new();

    register_declared_signatures(library_exprs, &mut declared_signatures);
    register_declared_signatures(program.exprs(), &mut declared_signatures);

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
    // module-init bindings in `top_level_order`. Building a library context
    // checks and lowers declarations; it does not execute their effects.
    // Library values initialize on demand in each evaluation context, and
    // successful values are reused only within that context.
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
    let mut type_env = library_type_env
        .iter()
        .map(|(name, ty_expr)| (name.clone(), ty_expr.clone()))
        .collect::<UnordMap<String, Expr>>();
    for (name, ty_expr) in program.type_env() {
        type_env.insert(name.clone(), ty_expr.clone());
    }

    let mut ctx = EvalContext {
        bindings: UnordMap::new(),
        binding_types: UnordMap::new(),
        precision_bindings: UnordMap::new(),
        declaration_values: UnordMap::new(),
        named_axis_route_cache: UnordMap::new(),
        named_axis_route_visiting: UnordSet::new(),
        top_level_defs,
        sorted_defs_snapshot: None,
        declared_signatures,
        adt_registry: program.adt_registry().clone(),
        type_env,
        adt_fields,
        tensor_bindings,
        session: Some(chelis_ir::host::HostLoweringSession::new(eval_program)),
        def_kernels: UnordMap::new(),
        transcript: Vec::new(),
        transcript_capture: crate::transcript_capture::current_transcript_capture(),
        resolving_top_levels: Vec::new(),
        random_seed: None,
        random_counter: 0,
        execution_exclusion: None,
        cancel: chelis_types::current_cancel_token(),
    };

    for name in top_level_order {
        if let Err(message) = ctx.resolve_top_level(&name) {
            return Err(RuntimeFailure {
                message,
                transcript: ctx.transcript,
            });
        }
    }

    // Surface host-lane selected callable roots. The arrow-form
    // `def priced() -> T = body` desugars to `(def priced (fn () body))`,
    // a nullary thunk of type `() -> T`. `register_top_level_defs`'
    // `is_fn` guard skips it from the eager value-binding order (it looks
    // like a function), yet `root_names_from_checked_exprs` lists it as a
    // display root (unwrapping the declared `t-fn` return type). Without
    // this pass the root has no host binding and `eval_compiled` drops it
    // (`lookup_runtime_value_for_root` → None → filter_map), so an
    // in-context library call like `bs_call_f64_vector(...)` returned
    // `{"roots":[]}` instead of its value (chelis blocker2).
    //
    // We evaluate the value here by APPLYING the callable rather than binding
    // it in `ctx.bindings`, so the call-resolution frame is untouched and an
    // explicit `name()` call elsewhere still resolves the callable. A
    // parameterized definition reaches this pass only after the manifest
    // consumer has selected it and verified all live tensor inputs; unused
    // authored parameters receive Unit placeholders solely to satisfy the
    // closure's declared arity. Only Host-lane roots are applied — a
    // Tensor-lane callable surfaces through the DAG/`tensor_bindings` path.
    // Failures stay attached to the root and become [05-UNS-1]; no partial
    // root is fabricated.
    let mut host_root_values = UnordMap::new();
    let mut host_root_errors = UnordMap::new();
    for expr in top_level_items(program.exprs()) {
        let Some((DeepTag::Def, kids)) = tagged_expr_children(expr) else {
            continue;
        };
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        let Some((DeepTag::Fn, fn_children)) = tagged_expr_children(body) else {
            continue;
        };
        // Distinguish an automatic nullary observation from an explicitly
        // selected parameterized call.
        let is_zero_arg_fn = fn_children.first().is_some_and(|params| {
            tagged_expr_children(params)
                .is_some_and(|(tag, children)| tag == DeepTag::Params && children.is_empty())
        });
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
        if is_zero_arg_fn && carries_effect_row(body) {
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
        // Apply the selected callable. A missing authored parameter can only
        // be dead here: live tensor parameters were part of the manifest's
        // required-input set, and the compiler would not have selected this
        // call without their bindings. Unit therefore supplies arity without
        // inventing a numeric value; an incorrect reachability decision still
        // fails loudly when the body tries to use it. If another root already
        // resolved this declaration, its cached value is the callable
        // closure, not its result. Reuse that closure but still apply it: an
        // owed [05-OBS-7] root can never be represented by `<closure>`, and
        // the effect-row guard above proves the automatic application pure.
        let callable = ctx.resolve_top_level(name);
        let applied = callable.and_then(|closure| {
            // Admission and required-input filtering already selected this
            // call. Deliver its supplied tensor actuals independently of the
            // body's execution profile, retaining evaluated-root precedence.
            let args = match &closure {
                RuntimeValue::Closure { params, .. } => params
                    .iter()
                    .map(|param| {
                        tensor_bindings
                            .get(param)
                            .cloned()
                            .map(RuntimeValue::Tensor)
                            .or_else(|| {
                                bound_evaluation_inputs?
                                    .get(param)
                                    .cloned()
                                    .map(RuntimeTensorValue::new)
                                    .map(RuntimeValue::Tensor)
                            })
                            .unwrap_or(RuntimeValue::Unit)
                    })
                    .collect(),
                _ => Vec::new(),
            };
            ctx.apply_resolved_callable(closure, args)
        });
        match applied {
            Ok(value) => {
                host_root_values.insert(name.to_string(), value);
            }
            Err(error) => {
                host_root_errors.insert(name.to_string(), error);
            }
        }
    }

    Ok(RuntimeOutcome {
        // Lexical frames are not output roots. Project successful declaration
        // values separately from applied callable-root observations above.
        host_bindings: ctx.declaration_values,
        host_root_values,
        host_root_errors,
        transcript: ctx.transcript,
    })
}

fn register_declared_signatures(exprs: &[Expr], signatures: &mut UnordMap<String, Expr>) {
    for expr in top_level_items(exprs) {
        let Some((DeepTag::Defsig, kids)) = tagged_expr_children(expr) else {
            continue;
        };
        let (Some(name), Some(signature)) = (kids.first().and_then(symbol_name), kids.get(1))
        else {
            continue;
        };
        signatures.insert(name.to_string(), signature.clone());
    }
}

/// chelis#1277 B2h: a def whose body is a `(fn ...)` literal resolves to a
/// closure that IS the def; record its name so an application can take the
/// kernel the C lane emits for that def. A local `fn` literal inside a body
/// never passes through here and keeps `def_name: None`.
fn stamp_def_closure(value: RuntimeValue, name: &str, body: &Expr) -> RuntimeValue {
    let body_is_fn = tagged_expr_children(body).is_some_and(|(tag, _)| tag == DeepTag::Fn);
    match value {
        RuntimeValue::Closure {
            params,
            param_types,
            return_type,
            checked_signature,
            invocation_contracts,
            body: closure_body,
            env,
            precision_env,
            def_name: None,
        } if body_is_fn => RuntimeValue::Closure {
            params,
            param_types,
            return_type,
            checked_signature,
            invocation_contracts,
            body: closure_body,
            env,
            precision_env,
            def_name: Some(name.to_string()),
        },
        other => other,
    }
}

fn register_top_level_defs(
    exprs: &[Expr],
    lowered_names: &BTreeMap<String, bool>,
    selected_roots: Option<&[String]>,
    top_level_defs: &mut UnordMap<String, Expr>,
    top_level_order: &mut Vec<String>,
    register_runtime_order: bool,
) {
    for expr in top_level_items(exprs) {
        let Some((DeepTag::Def, kids)) = tagged_expr_children(expr) else {
            continue;
        };
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
        let is_fn = tagged_expr_children(body).is_some_and(|(tag, _)| tag == DeepTag::Fn);
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
pub(crate) fn library_lowered_names(library: &CheckedProgram) -> BTreeMap<String, bool> {
    top_level_lowering_map(library.annotated_exprs(), library.type_env())
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
            let value = TensorValue {
                shape: tensor
                    .value
                    .shape
                    .iter()
                    .map(|extent| {
                        i64::try_from(*extent)
                            .map_err(|_| "runtime extent exceeds wire int64".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                data: tensor.value.storage().clone(),
            };
            value.validate()?;
            ExecutionValue::Tensor { value }
        }
        RuntimeValue::Scalar(payload) => ExecutionValue::Scalar {
            value: payload.value().try_into()?,
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

pub(crate) fn lookup_runtime_value_for_manifest_root(
    entry: &RootEntry,
    host_bindings: &UnordMap<String, RuntimeValue>,
    tensor_bindings: &UnordMap<String, RuntimeTensorValue>,
) -> Option<RuntimeValue> {
    if let Some(value) = tensor_bindings.get(entry.name.as_str()) {
        return Some(RuntimeValue::Tensor(value.clone()));
    }

    let mut value = host_bindings
        .get(entry.def_name.as_str())
        .or_else(|| host_bindings.get(entry.name.as_str()))?
        .clone();
    for step in &entry.path {
        value = descend_manifest_path(value, *step)?;
    }
    Some(value)
}

fn descend_manifest_path(value: RuntimeValue, step: RootPathStep) -> Option<RuntimeValue> {
    match (step, value) {
        (RootPathStep::Tuple(index), RuntimeValue::Tuple(items)) => items.into_iter().nth(index),
        (RootPathStep::Adt(index), RuntimeValue::Adt { fields, .. }) => {
            fields.into_iter().nth(index)
        }
        // Lists are the runtime representation of the prelude's recursive
        // `Cons(head, tail)` / `Nil` ADT. Manifest construction sees the
        // checked constructor tree, so preserve its structural indexing:
        // field 0 is the head and field 1 is the remaining list.
        (RootPathStep::Adt(0), RuntimeValue::List(items)) => items.into_iter().next(),
        (RootPathStep::Adt(1), RuntimeValue::List(items)) => {
            (!items.is_empty()).then(|| RuntimeValue::List(items.into_iter().skip(1).collect()))
        }
        _ => None,
    }
}

struct EvalContext<'a> {
    /// Only lexical values; successful declarations never enter this map.
    bindings: UnordMap<String, RuntimeValue>,
    /// Declared/static Deep type expression for names in `bindings`,
    /// maintained in lockstep with `bindings` (saved/swapped/restored at
    /// every frame boundary). Every locally-bound name gets a key here:
    /// `Some(ty)` when a declared or checker-annotated type is known,
    /// `None` otherwise. The explicit `None` marker matters: it masks a
    /// same-named top-level `type_env` entry so a local shadow is never
    /// typed with the outer binding's type (chelis#338 named-axis routing).
    binding_types: UnordMap<String, Option<Expr>>,
    /// Call-frame actualizations for precision variables used by generic
    /// casts. Values come only from checker-owned call-site argument/result
    /// types matched against the callee's declared signature.
    precision_bindings: UnordMap<String, Prim>,
    /// Successful initializations keyed by canonical declaration identity.
    /// Lives for this context (one ordinary request, or one invariant
    /// predicate), independently of lexical frame restoration. Never stores
    /// failed initializations or the results of applying cached callables.
    declaration_values: UnordMap<String, RuntimeValue>,
    /// Memoized per-def result of [`Self::def_requires_named_axis_routing`].
    named_axis_route_cache: UnordMap<String, bool>,
    /// Cycle guard for the recursive routing detection walk.
    named_axis_route_visiting: UnordSet<String>,
    top_level_defs: UnordMap<String, Expr>,
    /// Program-scoped, lazily built sorted snapshot of `top_level_defs`
    /// (chelis#2059). `admit_execution_profile` runs on every closure
    /// application; before this it re-cloned every top-level definition twice
    /// per call to classify the body's execution profile, making a `chelis
    /// test` run over a package with many defs O(defs x applications). The set
    /// of defs is fixed for the context's lifetime (`register_top_level_defs`
    /// runs once at construction), so the sort-and-clone is lifted here and
    /// reused. `None` until the first classification asks for it.
    sorted_defs_snapshot: Option<std::rc::Rc<std::collections::BTreeMap<String, Expr>>>,
    /// Authored `defsig` function types, including source binder spellings.
    /// The inferred `type_env` intentionally freshens those binders, so the
    /// evaluator keeps this separate map for generic cast targets in bodies.
    declared_signatures: UnordMap<String, Expr>,
    /// Checker-owned nominal definitions used when the executed constructor
    /// alone cannot reveal whether the parameter type has a float leaf.
    adt_registry: chelis_types::adt::AdtRegistry,
    /// Combined library + new-code Deep type-env. Threaded into
    /// [`chelis_ir::lower::lower_subexpr_program`] when the host runtime
    /// hits a `grad` / `vmap` form so the lowerer can resolve free names
    /// the same way the C backend does. Empty when no library context is
    /// present (e.g. unit tests that don't need transform support).
    type_env: UnordMap<String, Expr>,
    adt_fields: UnordMap<String, Vec<String>>,
    tensor_bindings: &'a UnordMap<String, RuntimeTensorValue>,
    /// The host-lowering session over the checked program under evaluation.
    /// The kernel decision for a def application is read through it by
    /// `chelis_ir::host::host_def_kernel` (chelis#1277 B2h), so eval and C
    /// answer "is this def a kernel" from one function. One session serves the
    /// whole evaluation, so each definition's decision is derived once
    /// (chelis#1835); before that the memo behind it was gated on a
    /// thread-local flag this entry never armed, and every applied definition
    /// re-expanded the call graph (chelis#1829). `None` only for the
    /// invariant-predicate evaluator, which has no program and therefore no
    /// kernels: it interprets every application.
    session: Option<chelis_ir::host::HostLoweringSession<'a>>,
    /// Per-def kernel decision: `None` is the host lane, `Some` a kernel whose
    /// DAG draws no Random and is reused across applications. A Random-drawing
    /// kernel is re-lowered per application and never cached (see
    /// `EvalContext::def_kernel`).
    def_kernels: UnordMap<String, Option<std::sync::Arc<DefEvaluationKernel>>>,
    transcript: Vec<String>,
    transcript_capture: Option<crate::TranscriptCapture>,
    resolving_top_levels: Vec<String>,
    random_seed: Option<u64>,
    random_counter: u64,
    /// An explicitly excluded caller keeps all nested dispatch legacy. This
    /// is an admission decision, not recovery from a plan error.
    execution_exclusion: Option<chelis_ir::evaluation::LegacyEvaluationReason>,
    /// Cooperative cancellation flag (chelis#914), captured ONCE from the
    /// thread-local install point at construction so the per-node-visit
    /// check in [`Self::eval_expr`] is a relaxed atomic load rather than a
    /// TLS lookup. `None` — the default when no caller installed a token —
    /// makes the check a single `Option` discriminant test.
    cancel: Option<chelis_types::CancelToken>,
}

enum DefEvaluationKernel {
    Legacy(chelis_ir::host::HostDefKernel),
    Planned(chelis_ir::host::HostDefEvaluationPlan),
}

impl DefEvaluationKernel {
    fn kernel_for_inspection(&self) -> &chelis_ir::host::HostDefKernel {
        match self {
            Self::Legacy(kernel) => kernel,
            Self::Planned(plan) => plan.kernel_for_inspection(),
        }
    }
    fn plan(&self) -> Option<&chelis_ir::evaluation::EvaluationPlan> {
        match self {
            Self::Legacy(_) => None,
            Self::Planned(plan) => plan.plan(),
        }
    }
    fn staged_plan(&self) -> Option<&chelis_ir::evaluation::StagedEvaluationPlan> {
        match self {
            Self::Legacy(_) => None,
            Self::Planned(plan) => plan.staged_plan(),
        }
    }
}

fn tag(list: &List) -> Option<DeepTag> {
    list.tag()
}

fn get_meta(list: &List) -> Option<&Metadata> {
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
fn carries_effect_row(expr: &Expr) -> bool {
    let meta = match expr {
        Expr::List(list, _) => get_meta(list),
        Expr::Node(node, _) => Some(node.meta()),
        _ => None,
    };
    let Some(meta) = meta else {
        return false;
    };
    meta.effects().is_some_and(|row| !row.values().is_empty())
}

fn tagged_expr_children(expr: &Expr) -> Option<(DeepTag, &[Expr])> {
    match expr {
        Expr::List(list, _) => tag(list).map(|tag| (tag, children(list))),
        Expr::Node(node, _) => Some((node.tag(), node.children_slice())),
        _ => None,
    }
}

/// Extract the primitive dtype written into a `(lit {type: ...})` meta
/// by the type checker, if any. Returns `None` for non-primitive type
/// metadata (e.g. tensor literal types) or missing metadata; eval_lit
/// then falls back to the spec §5.3 literal default.
fn lit_meta_prim(meta: &Metadata) -> Option<Prim> {
    let ty_expr = meta.ty()?.expression();
    extract_prim_from_type_expr(ty_expr)
}

/// Binder name from a `(lit {type: (t-var {} p)} ...)` stamp (#1544).
fn lit_meta_type_var_name(meta: &Metadata) -> Option<&str> {
    let ty_expr = meta.ty()?.expression();
    chelis_deep::exact_type_variable_name(ty_expr)
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
