use chelis_deep::DeepTag;
use std::collections::{HashMap, HashSet};

use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_ir::lower::top_level_lowering_map;
use chelis_types::{CheckedProgram, types::Prim};

use crate::schema::{DictEntryValue, ExecutionValue, TensorValue};

mod eval;
mod host_ops;
mod invariant;
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

/// Per-dtype scalar storage used by [`RuntimeValue::Scalar`].
///
/// The variant carries the value at exactly the precision the program
/// has assigned. Construction goes through
/// [`RuntimeValue::scalar`]/[`RuntimeValue::int`]/[`RuntimeValue::float`]
/// (or one of the typed `int_*`/`float_*` constructors) so the
/// `dtype`-vs-`bits` invariant cannot be silently violated. See
/// `spec/04-type-system.md` §1.1 for the active dtype set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarBits {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F16(half::f16),
    Bf16(half::bf16),
    F32(f32),
    F64(f64),
}

impl ScalarBits {
    /// The Prim that **must** match the surrounding `Scalar { dtype, bits }`
    /// payload's `dtype` field. The constructor [`RuntimeValue::scalar`]
    /// enforces that invariant.
    pub(crate) fn dtype(&self) -> Prim {
        match self {
            ScalarBits::I8(_) => Prim::Int8,
            ScalarBits::I16(_) => Prim::Int16,
            ScalarBits::I32(_) => Prim::Int32,
            ScalarBits::I64(_) => Prim::Int64,
            ScalarBits::F16(_) => Prim::F16,
            ScalarBits::Bf16(_) => Prim::Bf16,
            ScalarBits::F32(_) => Prim::F32,
            ScalarBits::F64(_) => Prim::F64,
        }
    }

    /// View any integer scalar as i64. Float scalars truncate toward zero
    /// (same convention as the existing `as i64` cast paths the host lane
    /// already used pre-refactor).
    pub(crate) fn as_i64(&self) -> i64 {
        match self {
            ScalarBits::I8(v) => *v as i64,
            ScalarBits::I16(v) => *v as i64,
            ScalarBits::I32(v) => *v as i64,
            ScalarBits::I64(v) => *v,
            ScalarBits::F16(v) => f32::from(*v) as i64,
            ScalarBits::Bf16(v) => f32::from(*v) as i64,
            ScalarBits::F32(v) => *v as i64,
            ScalarBits::F64(v) => *v as i64,
        }
    }

    /// View any numeric scalar as f64. Integer scalars widen losslessly
    /// up to i32; i64 may lose precision past 2^53 (matches IEEE-754
    /// double semantics, which is what the pre-refactor host lane did).
    pub(crate) fn as_f64(&self) -> f64 {
        match self {
            ScalarBits::I8(v) => *v as f64,
            ScalarBits::I16(v) => *v as f64,
            ScalarBits::I32(v) => *v as f64,
            ScalarBits::I64(v) => *v as f64,
            ScalarBits::F16(v) => f32::from(*v) as f64,
            ScalarBits::Bf16(v) => f32::from(*v) as f64,
            ScalarBits::F32(v) => *v as f64,
            ScalarBits::F64(v) => *v,
        }
    }

    /// Re-pack an `f64` as the same dtype as `self`. Used by binary ops
    /// that compute in `f64` and need to stash the result back at the
    /// operand's dtype.
    pub(crate) fn from_f64_as(dtype: Prim, value: f64) -> Result<Self, String> {
        Ok(match dtype {
            Prim::Int8 => ScalarBits::I8(value as i8),
            Prim::Int16 => ScalarBits::I16(value as i16),
            Prim::Int32 => ScalarBits::I32(value as i32),
            Prim::Int64 => ScalarBits::I64(value as i64),
            Prim::F16 => ScalarBits::F16(half::f16::from_f32(value as f32)),
            Prim::Bf16 => ScalarBits::Bf16(half::bf16::from_f32(value as f32)),
            Prim::F32 => ScalarBits::F32(value as f32),
            Prim::F64 => ScalarBits::F64(value),
            other => {
                return Err(format!(
                    "cannot pack scalar bits at non-numeric dtype `{}`",
                    other.name()
                ));
            }
        })
    }

    /// Re-pack an `i64` as the same dtype as `self`.
    pub(crate) fn from_i64_as(dtype: Prim, value: i64) -> Result<Self, String> {
        Ok(match dtype {
            Prim::Int8 => ScalarBits::I8(value as i8),
            Prim::Int16 => ScalarBits::I16(value as i16),
            Prim::Int32 => ScalarBits::I32(value as i32),
            Prim::Int64 => ScalarBits::I64(value),
            Prim::F16 => ScalarBits::F16(half::f16::from_f32(value as f32)),
            Prim::Bf16 => ScalarBits::Bf16(half::bf16::from_f32(value as f32)),
            Prim::F32 => ScalarBits::F32(value as f32),
            Prim::F64 => ScalarBits::F64(value as f64),
            other => {
                return Err(format!(
                    "cannot pack scalar bits at non-numeric dtype `{}`",
                    other.name()
                ));
            }
        })
    }
}

/// Sealed payload for [`RuntimeValue::Scalar`] (WS-A0 RT-1 fixup C1).
///
/// The dtype/bits pairing is enforced inside [`ScalarPayload::new`] —
/// the inner fields are private so no caller (in or out of this crate)
/// can construct a payload via struct-literal syntax that bypasses the
/// invariant. This is the structural fix for the silent-init pattern
/// the RT-1 review found: with a `pub(crate) Scalar { dtype, bits }`
/// variant, any in-crate caller could write
/// `RuntimeValue::Scalar { dtype: F16, bits: ScalarBits::F32(_) }`
/// directly and skip the `RuntimeValue::scalar()` invariant check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarPayload {
    dtype: Prim,
    bits: ScalarBits,
}

/// Error returned by [`ScalarPayload::new`] when the requested dtype
/// disagrees with the bits-variant's intrinsic dtype. Replaces the
/// stringly-typed error returned by the old `RuntimeValue::scalar`
/// constructor so the C1 invariant has a typed failure path.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarMismatchError {
    pub dtype: Prim,
    pub bits_dtype: Prim,
}

impl std::fmt::Display for ScalarMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ScalarPayload dtype/bits mismatch: dtype={} but bits carry dtype {} \
             (spec/04-type-system.md §1.1)",
            self.dtype.name(),
            self.bits_dtype.name(),
        )
    }
}

impl std::error::Error for ScalarMismatchError {}

impl ScalarPayload {
    /// Construct a payload with the dtype/bits invariant enforced.
    /// Returns [`ScalarMismatchError`] for mismatched pairs (e.g.
    /// `dtype = F16, bits = F32(_)`).
    pub(crate) fn new(dtype: Prim, bits: ScalarBits) -> Result<Self, ScalarMismatchError> {
        if dtype != bits.dtype() {
            return Err(ScalarMismatchError {
                dtype,
                bits_dtype: bits.dtype(),
            });
        }
        Ok(Self { dtype, bits })
    }

    /// Read the source-level dtype.
    pub(crate) fn dtype(&self) -> Prim {
        self.dtype
    }

    /// Read the dtype-tagged storage.
    pub(crate) fn bits(&self) -> ScalarBits {
        self.bits
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

impl RuntimeValue {
    /// Construct a `Scalar` payload, asserting that `dtype` matches the
    /// `bits` variant. Returns an error for mismatched pairs (e.g.
    /// `dtype = F16, bits = F32(_)`) so the WS-A0 invariant from
    /// `spec/04-type-system.md` §1.1 is enforced at every construction
    /// site, not silently elided. Used by every typed-literal lowering
    /// path; ad-hoc internal sites that already have the dtype + bits
    /// in sync should prefer one of the typed `int_*` / `float_*`
    /// constructors below.
    #[allow(
        dead_code,
        reason = "WS-A0 invariant constructor; used by acceptance tests"
    )]
    pub(crate) fn scalar(dtype: Prim, bits: ScalarBits) -> Result<Self, String> {
        ScalarPayload::new(dtype, bits)
            .map(RuntimeValue::Scalar)
            .map_err(|e| e.to_string())
    }

    /// Default-narrowed integer literal per spec §5.3: bare integer
    /// values default to `int32` unless the surrounding context says
    /// otherwise.
    pub(crate) fn int_lit(value: i64) -> Self {
        // C1 (WS-A0 RT-1 fixup): route every internal construction
        // through `ScalarPayload::new` so the dtype/bits invariant is
        // enforced uniformly. The pair is hardcoded matching, so
        // `expect` here is a structural assertion, not a runtime check.
        RuntimeValue::Scalar(
            ScalarPayload::new(Prim::Int32, ScalarBits::I32(value as i32))
                .expect("int_lit pair is invariant-correct by construction"),
        )
    }

    /// Default-narrowed float literal per spec §5.3: bare float values
    /// default to `f32`.
    pub(crate) fn float_lit(value: f64) -> Self {
        RuntimeValue::Scalar(
            ScalarPayload::new(Prim::F32, ScalarBits::F32(value as f32))
                .expect("float_lit pair is invariant-correct by construction"),
        )
    }

    /// Computed integer value preserving full i64 precision (e.g. `len`,
    /// shape sizes, parsed `to_int` results). Carries dtype `int64`.
    pub(crate) fn int64(value: i64) -> Self {
        RuntimeValue::Scalar(
            ScalarPayload::new(Prim::Int64, ScalarBits::I64(value))
                .expect("int64 pair is invariant-correct by construction"),
        )
    }

    /// Computed float value preserving full f64 precision (e.g. `to_float`
    /// parse results, scalar reductions over f64 tensors). Carries dtype
    /// `f64`.
    pub(crate) fn float64(value: f64) -> Self {
        RuntimeValue::Scalar(
            ScalarPayload::new(Prim::F64, ScalarBits::F64(value))
                .expect("float64 pair is invariant-correct by construction"),
        )
    }

    /// Construct a scalar at the dtype of an existing scalar (used by
    /// arithmetic ops to keep result-precision = operand-precision).
    pub(crate) fn scalar_like_int(template_dtype: Prim, value: i64) -> Result<Self, String> {
        let bits = ScalarBits::from_i64_as(template_dtype, value)?;
        ScalarPayload::new(template_dtype, bits)
            .map(RuntimeValue::Scalar)
            .map_err(|e| e.to_string())
    }

    pub(crate) fn scalar_like_float(template_dtype: Prim, value: f64) -> Result<Self, String> {
        let bits = ScalarBits::from_f64_as(template_dtype, value)?;
        ScalarPayload::new(template_dtype, bits)
            .map(RuntimeValue::Scalar)
            .map_err(|e| e.to_string())
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
            RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => {
                Some(payload.bits().as_i64())
            }
            _ => None,
        }
    }

    /// View this value as f64 if it is a float-typed scalar. Mirrors
    /// `as_i64` for the float row.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            RuntimeValue::Scalar(payload) if payload.dtype().is_float() => {
                Some(payload.bits().as_f64())
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
    let Expr::List(list, _) = expr else {
        return;
    };
    if tag(list) == Some(DeepTag::Module) {
        for child in list.elements.iter().skip(3) {
            collect_top_level_items(child, out);
        }
        return;
    }
    out.push(expr);
}

pub(crate) fn runtime_value_to_schema(value: &RuntimeValue) -> Result<ExecutionValue, String> {
    Ok(match value {
        RuntimeValue::Tensor(tensor) => ExecutionValue::Tensor {
            value: TensorValue {
                shape: tensor.value.shape.clone(),
                data: tensor.value.data.clone(),
            },
        },
        RuntimeValue::Scalar(payload) => {
            let dtype = payload.dtype();
            let bits = payload.bits();
            if dtype.is_integer() {
                ExecutionValue::Int64 {
                    value: bits.as_i64(),
                }
            } else if dtype.is_float() {
                ExecutionValue::Float64 {
                    value: bits.as_f64(),
                }
            } else {
                return Err(format!(
                    "non-numeric scalar dtype `{}` cannot be encoded into ExecutionValue",
                    dtype.name()
                ));
            }
        }
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
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
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
