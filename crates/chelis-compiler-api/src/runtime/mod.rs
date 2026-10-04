use chelis_deep::DeepTag;
use chelis_unord::{UnordMap, UnordSet};
use std::collections::BTreeMap;
use std::sync::Arc;

use chelis_deep::ast::{Atom, Expr, Metadata};
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
mod frame;
pub use frame::Frame;
use frame::ResultProducer;
mod host_ops;
mod invariant;
mod named_axis;
mod numeric_text;
mod program_scope;
mod shared_values;
#[cfg(test)]
mod std_clock_tests;
mod system;
mod system_adapter;
#[cfg(test)]
mod system_tests;
use program_scope::ProgramScope;
pub use shared_values::{Entries, Values};
#[cfg(test)]
mod tests;
mod transforms;

pub(crate) use host_ops::{collect_adt_ctor_fields, collect_constructor_source_names};
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

    /// Data-movement constructor: store scalars that already carry `prim`
    /// by inserting their bits, so a moved NaN keeps its payload, sign and
    /// signaling bit ([04-NUM-11]). A value produced by arithmetic or
    /// conversion goes through [`Self::from_wide`] instead.
    pub(crate) fn from_scalars(
        prim: Prim,
        shape: Vec<usize>,
        values: &[chelis_types::ScalarValue],
    ) -> Self {
        Self::new(IrTensorValue::from_storage(
            shape,
            chelis_types::tensor_from_scalars(prim, values),
        ))
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
/// It never holds a key: [`RuntimeValue::from_scalar_value`], its one
/// construction site, turns a key element into [`RuntimeValue::Key`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarPayload {
    value: chelis_types::ScalarValue,
}

impl ScalarPayload {
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
    /// for integers up to 2^53; i64 may lose precision past 2^53 (the
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
    /// A scalar random key ([05-OP-69]..[05-OP-72]). It has no numeric
    /// value, so it is its own variant rather than a [`Self::Scalar`]: no
    /// numeric path can read it. A `tensor[n, key]` is a [`Self::Tensor`]
    /// whose storage is a key buffer.
    Key(chelis_types::RandomKey),
    Bool(bool),
    String(String),
    /// Container elements are shared, so a clone is one count increment and
    /// a release drains nested values iteratively (chelis#2567).
    List(Values),
    Dict(Entries),
    Tuple(Values),
    Adt {
        /// The constructor's identity in the checked program: the linker
        /// name in a package build, which pattern matching and field lookup
        /// compare against. Two types may share a source spelling, so the
        /// identity cannot be the spelling.
        ctor: String,
        /// The constructor's declared source spelling, the name every exit
        /// renders ([05-OBS-7]). It is the spelling the compiled lane stores,
        /// derived by the same rule
        /// ([`chelis_types::linked_constructor_source_name`] against the
        /// constructor's declared type); in a program without linker names it
        /// equals `ctor`.
        source_name: String,
        /// Field values in DECLARED order (the deftype's field order),
        /// not source or alphabetical order.
        fields: Values,
        /// When present, aligned index-for-index with `fields`, so it
        /// also follows declared order. `eval_record` enforces this
        /// (chelis#520 fixed a misalignment where kv source order was
        /// stored against declared-order `fields`).
        field_names: Option<Vec<String>>,
    },
    MappedFile(Vec<u8>),
    Closure {
        /// Exact checked `(fn ...)` expression that produced this closure.
        /// Transform lowering reuses this carrier directly: rebuilding a
        /// function from `params` and `body` drops checker-owned parameter
        /// and callable metadata (chelis#676).
        checked_function: Box<Expr>,
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
        env: Frame,
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
        captured_env: Frame,
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

    /// Wrap a module-finalized element (the WS-A0 invariant holds by
    /// construction: the storage variant IS the dtype). A key element, such
    /// as a rank-0 key tensor's one element, becomes a [`RuntimeValue::Key`]:
    /// a key is never a numeric [`RuntimeValue::Scalar`], so every consumer
    /// that reads a key and every renderer sees the key variant.
    pub(crate) fn from_scalar_value(value: chelis_types::ScalarValue) -> Self {
        // The payload's only construction site: a key never becomes one.
        match value.as_key() {
            Some(key) => RuntimeValue::Key(key),
            None => RuntimeValue::Scalar(ScalarPayload { value }),
        }
    }

    /// Default-narrowed integer literal per spec §5.3: bare integer
    /// values default to `i32` unless the surrounding context says
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
    /// shape sizes, parsed `to_int` results). Carries dtype `i64`.
    pub(crate) fn int64(value: i64) -> Self {
        Self::from_scalar_value(
            chelis_types::scalar_from_i64("i64", Prim::Int64, value)
                .expect("i64 ingress from i64 is total"),
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
    pub(crate) host_root_errors: UnordMap<String, RuntimeFailure>,
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
    pub(crate) kind: RuntimeFailureKind,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum RuntimeFailureKind {
    #[default]
    Ordinary,
    NumericTrap,
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
/// new code calls a library function (e.g. `Std.Datetime.is_leap_year`),
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
    evaluate_host_program_with_library_and_types_and_system(
        program,
        library,
        library_lowered_names,
        inputs,
        selected_roots,
        manifested_lowered_names,
        system::EvalSystemBoundary::permissive(),
    )
}

/// A program prepared once for many host evaluations (chelis#3144): the
/// host-lowering program with its shared facts, and the lowering
/// classification of the combined library and new-code definitions, which
/// every evaluation of the program reads and none changes.
pub(crate) struct PreparedHostEvaluation {
    host: chelis_ir::host::PreparedHostProgram,
    lowering_map: std::sync::OnceLock<BTreeMap<String, bool>>,
    scope_maps: std::sync::OnceLock<ScopeMaps>,
}

/// Every top-level definition body in scope and the combined type
/// environment, which an evaluation reads and never changes.
type ScopeMaps = (
    Arc<UnordMap<String, Expr>>,
    Arc<UnordMap<String, Expr>>,
    Arc<program_scope::TerminalIndexCell>,
);

/// Library definitions first, so new code wins on a shared name, as the
/// type environment's shadow rule has it.
fn scope_maps(
    library_exprs: &[Expr],
    library_type_env: &BTreeMap<String, Expr>,
    program: &CheckedProgram,
) -> ScopeMaps {
    let mut defs = UnordMap::new();
    collect_top_level_def_bodies(library_exprs, &mut defs);
    collect_top_level_def_bodies(program.exprs(), &mut defs);
    let mut type_env = library_type_env
        .iter()
        .map(|(name, ty_expr)| (name.clone(), ty_expr.clone()))
        .collect::<UnordMap<String, Expr>>();
    for (name, ty_expr) in program.type_env() {
        type_env.insert(name.clone(), ty_expr.clone());
    }
    (Arc::new(defs), Arc::new(type_env), Arc::default())
}

impl PreparedHostEvaluation {
    pub(crate) fn new(host: chelis_ir::host::PreparedHostProgram) -> Self {
        Self {
            host,
            lowering_map: std::sync::OnceLock::new(),
            scope_maps: std::sync::OnceLock::new(),
        }
    }
}

/// Evaluate against a program prepared once for many evaluations
/// (chelis#3144). `prepared` holds `library` composed with `program`, or
/// `program` alone without a library; its sessions reuse the program-wide
/// host-lowering facts earlier evaluations derived.
pub(crate) fn evaluate_prepared_host_program(
    prepared: &PreparedHostEvaluation,
    program: &CheckedProgram,
    library: Option<&CheckedProgram>,
    library_lowered_names: Option<&BTreeMap<String, bool>>,
    inputs: HostEvaluationInputs<'_>,
    selected_roots: Option<&[String]>,
    manifested_lowered_names: Option<&BTreeMap<String, bool>>,
) -> Result<RuntimeOutcome, RuntimeFailure> {
    evaluate_host_program_core(
        Some(prepared),
        program,
        library,
        library_lowered_names,
        inputs,
        selected_roots,
        manifested_lowered_names,
        system::EvalSystemBoundary::permissive(),
    )
}

/// Keep the checked-program, transcript and failure-kind path identical for
/// default and injected evaluators; only the system port differs.
pub(crate) fn evaluate_host_program_with_library_and_types_and_system(
    program: &CheckedProgram,
    library: Option<&CheckedProgram>,
    library_lowered_names: Option<&BTreeMap<String, bool>>,
    inputs: HostEvaluationInputs<'_>,
    selected_roots: Option<&[String]>,
    manifested_lowered_names: Option<&BTreeMap<String, bool>>,
    system_boundary: system::EvalSystemBoundary,
) -> Result<RuntimeOutcome, RuntimeFailure> {
    evaluate_host_program_core(
        None,
        program,
        library,
        library_lowered_names,
        inputs,
        selected_roots,
        manifested_lowered_names,
        system_boundary,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_host_program_core(
    prepared: Option<&PreparedHostEvaluation>,
    program: &CheckedProgram,
    library: Option<&CheckedProgram>,
    library_lowered_names: Option<&BTreeMap<String, bool>>,
    inputs: HostEvaluationInputs<'_>,
    selected_roots: Option<&[String]>,
    manifested_lowered_names: Option<&BTreeMap<String, bool>>,
    system_boundary: system::EvalSystemBoundary,
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
        .filter(|_| prepared.is_none())
        .map(|library| {
            CheckedProgram::compose(library, program)
                .ok_or_else(|| "runtime kernel program lost its checked library proof".to_owned())
        })
        .transpose()
        .map_err(|message| RuntimeFailure {
            message,
            transcript: Vec::new(),
            kind: RuntimeFailureKind::Ordinary,
        })?;
    // chelis#1829: the kernel-decision probe behind `def_kernel` expands the
    // call graph as a tree, so it must be derived once per definition for the
    // whole evaluation. Before #1693 this program held new code only, so an
    // imported name was not found and never probed; it now composes the
    // library in, so every imported definition takes that path. The session
    // `ctx` owns below is what holds those facts, and the borrow checker, not
    // a declaration order, is what keeps it inside `kernel_program`'s life.
    let eval_program = prepared
        .map(|prepared| prepared.host.program())
        .or(kernel_program.as_ref())
        .unwrap_or(program);

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
    let combined_lowering_map = || {
        let mut combined_exprs: Vec<Expr> = library_exprs.to_vec();
        combined_exprs.extend(program.exprs().iter().cloned());
        let mut combined_type_env: BTreeMap<String, Expr> = library_type_env.clone();
        for (name, ty_expr) in program.type_env() {
            combined_type_env.insert(name.clone(), ty_expr.clone());
        }
        top_level_lowering_map(&combined_exprs, &combined_type_env)
    };
    // The classification depends on the program alone, so a prepared
    // program derives it once for all of its evaluations.
    let mut new_lowered_names = match prepared {
        Some(prepared) => prepared
            .lowering_map
            .get_or_init(combined_lowering_map)
            .clone(),
        None => combined_lowering_map(),
    };
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
    let mut constructor_names = collect_constructor_source_names(library_exprs);
    constructor_names.merge(collect_constructor_source_names(program.exprs()));

    let mut top_level_order = Vec::new();
    let mut declared_signatures = UnordMap::new();

    register_declared_signatures(library_exprs, &mut declared_signatures);
    register_declared_signatures(program.exprs(), &mut declared_signatures);

    // Every library and new-code definition is in scope, new code winning
    // on a shared name, under the composed type-env (the Phase C shadow
    // rule). `grad`, `vmap` and realize routing read that type-env through
    // `lower_subexpr_program`, which resolves free names as the C backend
    // does. A prepared program derives both once.
    let (top_level_defs, type_env, terminal_index) = match prepared {
        Some(prepared) => prepared
            .scope_maps
            .get_or_init(|| scope_maps(library_exprs, library_type_env, program))
            .clone(),
        None => scope_maps(library_exprs, library_type_env, program),
    };
    // New code is the only source of eager module-init bindings in
    // `top_level_order`. Building a library context checks and lowers
    // declarations; it does not execute their effects. Library values
    // initialize on demand in each evaluation context, and successful values
    // are reused only within that context.
    collect_runtime_order(
        program.exprs(),
        &lowered_names,
        selected_roots,
        &mut top_level_order,
    );

    let mut ctx = EvalContext {
        bindings: Frame::new(),
        result_producer: None,
        binding_types: UnordMap::new(),
        precision_bindings: UnordMap::new(),
        declaration_values: UnordMap::new(),
        named_axis_route_cache: UnordMap::new(),
        named_axis_route_visiting: UnordSet::new(),
        program: ProgramScope::shared(top_level_defs, type_env, terminal_index),
        declared_signatures,
        adt_registry: program.adt_registry().clone(),
        adt_fields,
        constructor_names,
        tensor_bindings,
        session: Some(match prepared {
            Some(prepared) => prepared.host.session(),
            None => chelis_ir::host::HostLoweringSession::new(eval_program),
        }),
        active_declaration_names: Vec::new(),
        def_kernels: UnordMap::new(),
        transcript: Vec::new(),
        transcript_capture: crate::transcript_capture::current_transcript_capture(),
        resolving_top_levels: Vec::new(),
        cancel: chelis_types::current_cancel_token(),
        system: system_boundary,
        failure_kind: RuntimeFailureKind::Ordinary,
        activation_extents: Default::default(),
    };

    for name in top_level_order {
        if let Err(message) = ctx.resolve_top_level(&name) {
            return Err(RuntimeFailure {
                message,
                transcript: ctx.transcript,
                kind: ctx.failure_kind,
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
        ctx.failure_kind = RuntimeFailureKind::Ordinary;
        let callable = ctx.resolve_top_level(name);
        let applied = callable.and_then(|closure| {
            // Admission and required-input filtering already selected this
            // call. Deliver its supplied tensor actuals independently of the
            // body's execution profile, retaining evaluated-root precedence.
            let args: Vec<Option<RuntimeValue>> = match &closure {
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
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let present = args.iter().map(Option::is_some).collect::<Vec<_>>();
            let args = args
                .into_iter()
                .map(|arg| arg.unwrap_or(RuntimeValue::Unit))
                .collect();
            ctx.apply_selected_root_callable(closure, args, &present)
        });
        match applied {
            Ok(value) => {
                host_root_values.insert(name.to_string(), value);
            }
            Err(error) => {
                host_root_errors.insert(
                    name.to_string(),
                    RuntimeFailure {
                        message: error,
                        transcript: Vec::new(),
                        kind: ctx.failure_kind,
                    },
                );
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
        let (Some(name), Some(signature)) = (kids.first().and_then(symbol_name), kids.last())
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
            checked_function,
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
            checked_function,
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

#[cfg(test)]
fn register_top_level_defs(
    exprs: &[Expr],
    lowered_names: &BTreeMap<String, bool>,
    selected_roots: Option<&[String]>,
    top_level_defs: &mut UnordMap<String, Expr>,
    top_level_order: &mut Vec<String>,
    register_runtime_order: bool,
) {
    collect_top_level_def_bodies(exprs, top_level_defs);
    if register_runtime_order {
        collect_runtime_order(exprs, lowered_names, selected_roots, top_level_order);
    }
}

/// Every top-level definition's body by name; a later definition replaces
/// an earlier one of the same name.
fn collect_top_level_def_bodies(exprs: &[Expr], top_level_defs: &mut UnordMap<String, Expr>) {
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
    }
}

/// The selected host-lane value definitions of `exprs`, in source order,
/// which evaluation initializes eagerly.
fn collect_runtime_order(
    exprs: &[Expr],
    lowered_names: &BTreeMap<String, bool>,
    selected_roots: Option<&[String]>,
    top_level_order: &mut Vec<String>,
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
        Expr::Node(node, _) if node.tag() == DeepTag::Module => {
            for child in node.children_slice().iter().skip(1) {
                collect_top_level_items(child, out);
            }
        }
        _ => out.push(expr),
    }
}

/// One pending step of [`runtime_value_to_schema`]'s walk.
enum SchemaStep<'a> {
    Visit(&'a RuntimeValue),
    /// Every child of this container is on the output stack, in order.
    Assemble(&'a RuntimeValue),
}

/// The machine-facing value of `value`, converted from a worklist so a value
/// nested far deeper than the native stack converts with bounded native
/// depth (chelis#2567). The first unconvertible leaf in depth-first,
/// left-to-right order is the error, as in a recursive conversion.
pub(crate) fn runtime_value_to_schema(value: &RuntimeValue) -> Result<ExecutionValue, String> {
    let mut steps = vec![SchemaStep::Visit(value)];
    let mut converted: Vec<ExecutionValue> = Vec::new();
    while let Some(step) = steps.pop() {
        match step {
            SchemaStep::Visit(value) => match value {
                RuntimeValue::List(items)
                | RuntimeValue::Tuple(items)
                | RuntimeValue::Adt { fields: items, .. } => {
                    steps.push(SchemaStep::Assemble(value));
                    steps.extend(items.iter().rev().map(SchemaStep::Visit));
                }
                RuntimeValue::Dict(entries) => {
                    steps.push(SchemaStep::Assemble(value));
                    for (key, entry) in entries.iter().rev() {
                        steps.push(SchemaStep::Visit(entry));
                        steps.push(SchemaStep::Visit(key));
                    }
                }
                leaf => converted.push(leaf_to_schema(leaf)?),
            },
            SchemaStep::Assemble(value) => {
                let assembled = match value {
                    RuntimeValue::List(items) => ExecutionValue::List {
                        value: converted.split_off(converted.len() - items.len()),
                    },
                    RuntimeValue::Tuple(items) => ExecutionValue::Tuple {
                        value: converted.split_off(converted.len() - items.len()),
                    },
                    // The eval `--json` ABI surface carries the stored source
                    // spelling, never the linker name (chelis#399, chelis#2889);
                    // decode (`decode_adt_value`) keys on source names, so the
                    // encode/decode round trip stays consistent.
                    RuntimeValue::Adt {
                        source_name,
                        fields,
                        ..
                    } => ExecutionValue::Adt {
                        ctor: source_name.clone(),
                        fields: converted.split_off(converted.len() - fields.len()),
                    },
                    RuntimeValue::Dict(entries) => {
                        let mut flat = converted
                            .split_off(converted.len() - 2 * entries.len())
                            .into_iter();
                        let mut pairs = Vec::with_capacity(entries.len());
                        while let (Some(key), Some(value)) = (flat.next(), flat.next()) {
                            pairs.push(DictEntryValue { key, value });
                        }
                        ExecutionValue::Dict { entries: pairs }
                    }
                    _ => unreachable!("only containers are assembled"),
                };
                converted.push(assembled);
            }
        }
    }
    Ok(converted
        .pop()
        .expect("the schema worklist converts its root"))
}

/// The wire value of a runtime value that holds no nested runtime value.
fn leaf_to_schema(value: &RuntimeValue) -> Result<ExecutionValue, String> {
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
        // spec/10 section 3.2: a scalar key is `{"type":"key","bits":h}`.
        RuntimeValue::Key(key) => ExecutionValue::Key {
            bits: chelis_types::KeyBits::new(*key),
        },
        RuntimeValue::Bool(value) => ExecutionValue::Bool { value: *value },
        RuntimeValue::String(value) => ExecutionValue::String {
            value: value.clone(),
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
        RuntimeValue::List(_)
        | RuntimeValue::Tuple(_)
        | RuntimeValue::Dict(_)
        | RuntimeValue::Adt { .. } => unreachable!("containers are converted by the worklist"),
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
    /// Only lexical values; successful declarations never enter this frame.
    bindings: Frame,
    /// Canonical producer of the tensor returned by the expression currently
    /// completing. Expression entry clears it, and only a producer or
    /// transparent value route may set it.
    result_producer: Option<ResultProducer>,
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
    /// The program's top-level definitions, its Deep type environment, and
    /// the facts derived from both. Both tables are registered once during
    /// construction and are fixed for the context's lifetime. The derived
    /// facts would be stale if either table moved, so `program_scope` owns
    /// them behind accessors and this module cannot reach the fields; that
    /// module's header records why the boundary is a separate file.
    program: ProgramScope,
    /// Authored `defsig` function types, including source binder spellings.
    /// The inferred `type_env` intentionally freshens those binders, so the
    /// evaluator keeps this separate map for generic cast targets in bodies.
    declared_signatures: UnordMap<String, Expr>,
    /// Checker-owned nominal definitions used when the executed constructor
    /// alone cannot reveal whether the parameter type has a float leaf.
    adt_registry: chelis_types::adt::AdtRegistry,
    adt_fields: UnordMap<String, Vec<String>>,
    /// Each linker-named constructor's declared source spelling, keyed by
    /// its linker name (`collect_constructor_source_names`); an ADT value
    /// built here stores it as the name it renders.
    constructor_names: UnordMap<String, String>,
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
    /// Active checked declaration frames. Synthetic local-ascription regions
    /// select artifact-local identities through this stack so two composed
    /// source units with equal names and byte offsets cannot cross-own a
    /// runtime obligation.
    active_declaration_names: Vec<String>,
    /// Per-def kernel decision: `None` is the host lane, `Some` a kernel
    /// reused across applications (see `EvalContext::def_kernel`).
    def_kernels: UnordMap<String, Option<std::sync::Arc<chelis_ir::host::HostDefKernel>>>,
    transcript: Vec<String>,
    transcript_capture: Option<crate::TranscriptCapture>,
    resolving_top_levels: Vec<String>,
    /// Cooperative cancellation flag (chelis#914), captured ONCE from the
    /// thread-local install point at construction so the per-node-visit
    /// check in [`Self::eval_expr`] is a relaxed atomic load rather than a
    /// TLS lookup. `None` — the default when no caller installed a token —
    /// makes the check a single `Option` discriminant test.
    cancel: Option<chelis_types::CancelToken>,
    /// Mandatory policy-checked system port for one evaluation lifetime.
    system: system::EvalSystemBoundary,
    /// Origin of the error currently unwinding through the string-based host
    /// evaluator. Only a trusted numeric producer may set `NumericTrap`.
    failure_kind: RuntimeFailureKind,
    /// The dimension-binder extents of each executing activation.
    activation_extents: eval::ActivationExtents,
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
