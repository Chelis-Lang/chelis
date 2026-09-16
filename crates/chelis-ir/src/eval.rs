//! Tensor-aware evaluator for the Phase 0 RISC DAG.
//!
//! chelis#729 Phase 1: values are stored in the sealed per-dtype
//! [`TensorStorage`] (`chelis_types::dtype_semantics`), and every compute
//! op constructs its result through `finalize_tensor` at the node's output
//! precision (the section C1 finalize-per-op contract). Movement ops that
//! provably preserve elements move storage through the `reuse_*` family
//! instead of re-finalizing. The raw f64 `binary_map`/`unary_map` paths
//! this file used to run on are deleted, not deprecated.
//!
//! chelis#729 Phase 2: elementwise arithmetic dispatches once per buffer
//! into the closed typed kernels. Integer operands never cross binary64,
//! f32/f64 compute at their declared widths, and f16/bf16 compute through
//! binary32 with one storage finalization. Reductions pass only ordered
//! index groups into the same closed typed-kernel boundary; this module
//! does not own numeric accumulation or comparison.

use chelis_unord::{UnordMap, UnordSet};
use std::borrow::Cow;

use crate::dag::{
    Dag, DagNode, DimExpr, DimInfo, ExtremaKind, ExtremaOperand, FusedInput, FusedStepOp, NodeId,
    ReduceWindowKind, RiscOp, RtAxis, RtDim, SHRINK_TO_END, TensorType, bind_symbolic_dims,
};
use chelis_types::dtype_semantics::{
    ArgReduceOp, CheckedCastPlan, CompareOp, ExtremaOperand as KernelExtremaOperand, FloatBinOp,
    FloatExtremaOp, FloatUnOp, IndexedTrapCandidate, IntBinOp, IntUnOp, RawTensor,
    ReduceWindowGradOp, TensorReduceOp, TensorStorage, arg_reduce_tensor_groups, compare_tensors,
    count_tensor_groups, finalize_tensor, float_extrema_adjoint, float_relu, float_relu_adjoint,
    float_tensor_binop, float_tensor_unop, int_tensor_binop, int_tensor_unop,
    integer_is_exactly_representable, reduce_tensor_groups, reduce_window_grad_tensor_groups,
    tensor_from_scalars, uniform_sample,
};
use chelis_types::types::Prim;

/// Public evaluator entry points that do not inherit an enclosing handler
/// still evaluate a path-sensitive DAG from the beginning of its stream.
const INITIAL_RANDOM_STREAM_ORDINAL: u64 = 0;

#[derive(Debug, Clone)]
pub struct TensorValue {
    storage: TensorStorage,
    pub shape: Vec<usize>,
    raw_f64_ingress: bool,
}

impl TensorValue {
    /// Rank-0 raw-f64 host ingress. The consuming Load freezes it at its
    /// declared dtype; this constructor is not an already-tagged value.
    pub fn scalar(value: f64) -> Self {
        Self {
            storage: finalize_tensor("scalar", Prim::F64, RawTensor::Float(vec![value]))
                .expect("f64 finalize is total"),
            shape: vec![],
            raw_f64_ingress: true,
        }
    }

    /// Raw-f64 host ingress constructor (tests and legacy data loaders). The
    /// consuming Load freezes the buffer once at its declared dtype. Typed
    /// wire and runtime values use [`Self::from_storage`] instead, so their
    /// tags can never be contextually substituted.
    pub fn from_vec(shape: Vec<usize>, data: Vec<f64>) -> Self {
        assert_eq!(numel(&shape), data.len());
        Self {
            storage: finalize_tensor("from_vec", Prim::F64, RawTensor::Float(data))
                .expect("f64 finalize is total"),
            shape,
            raw_f64_ingress: true,
        }
    }

    /// Wrap finalized storage with its shape.
    pub fn from_storage(shape: Vec<usize>, storage: TensorStorage) -> Self {
        assert_eq!(numel(&shape), storage.len());
        Self {
            storage,
            shape,
            raw_f64_ingress: false,
        }
    }

    /// The element dtype, from the storage variant itself.
    pub fn prim(&self) -> Prim {
        self.storage.prim()
    }

    pub fn storage(&self) -> &TensorStorage {
        &self.storage
    }

    /// Element count (the shape's numel).
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Widen every element to f64; exact except for i64 magnitudes
    /// above 2^53 (the named-lossy read of section C3).
    pub fn to_f64_lossy_vec(&self) -> Vec<f64> {
        self.storage.to_f64_lossy_vec()
    }

    /// One element widened to f64 (same loss profile).
    pub fn element_f64_lossy(&self, index: usize) -> f64 {
        self.storage.element_f64_lossy(index)
    }

    /// First element widened to f64, or 0.0 for an empty buffer (the
    /// scalar-read convention `eval_scalar` always had).
    pub fn first_f64_lossy_or_zero(&self) -> f64 {
        if self.storage.is_empty() {
            0.0
        } else {
            self.storage.element_f64_lossy(0)
        }
    }

    /// Public compute-op constructor for the host runtime lane: finalize a
    /// wide f64 buffer at `prim` (the same path this evaluator's own ops
    /// take, including the chelis#680 integer saturate adapter).
    pub fn finalize_from_wide(
        op: &'static str,
        prim: Prim,
        shape: Vec<usize>,
        wide: Vec<f64>,
    ) -> Result<Self, String> {
        finalize_wide(op, prim, shape, wide)
    }

    /// Public exact-integer constructor for paths that computed in i64.
    pub fn finalize_from_wide_int(
        op: &'static str,
        prim: Prim,
        shape: Vec<usize>,
        wide: Vec<i64>,
    ) -> Result<Self, String> {
        finalize_wide_int(op, prim, shape, wide)
    }
}

/// Public tensor `cast` for the host runtime lane; same per-direction
/// ladder as this evaluator's `RiscOp::Cast` arm (see [`cast_value`]).
pub fn cast_tensor(input: &TensorValue, src: Prim, dst: Prim) -> Result<TensorValue, String> {
    cast_value(input, src, dst)
}

/// Value equality for tests and fixtures: shapes equal and element values
/// equal, compared exactly per family. Cross-family (an int-family stored
/// result against a float-storage fixture) compares by value only where
/// that fixture's actual dtype carries the integer exactly. This is a
/// significand test, not a magnitude cutoff: powers of two such as
/// `i64::MIN` are exactly representable in f64, while `i64::MAX` is not.
/// NaN keeps `!=` semantics, as the old `Vec<f64>` derive had.
impl PartialEq for TensorValue {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }
        let self_prim = self.prim();
        let other_prim = other.prim();
        match (self.storage.to_raw(), other.storage.to_raw()) {
            (RawTensor::Int(a), RawTensor::Int(b)) => a == b,
            (RawTensor::Float(a), RawTensor::Float(b)) => a == b,
            (RawTensor::Int(ints), RawTensor::Float(floats)) => {
                ints.len() == floats.len()
                    && ints.iter().zip(&floats).all(|(&i, &f)| {
                        integer_is_exactly_representable(i, other_prim) && (i as f64) == f
                    })
            }
            (RawTensor::Float(floats), RawTensor::Int(ints)) => {
                ints.len() == floats.len()
                    && ints.iter().zip(&floats).all(|(&i, &f)| {
                        integer_is_exactly_representable(i, self_prim) && (i as f64) == f
                    })
            }
        }
    }
}

fn numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        // Scalar: no dimensions means a single element.
        1
    } else if shape.contains(&0) {
        // Non-scalar with a zero extent: honor it. A tensor[0, f32]
        // legitimately holds zero elements; inflating to 1 drops data integrity
        // and panics the from_vec length assertion.
        //
        // Short-circuit rather than fold, because the answer does not depend on
        // the other extents and folding them can overflow a product that is
        // defined to be zero: `[2^32, 2^32, 0]` reaches `2^64` before it ever
        // reaches the zero.
        0
    } else {
        shape.iter().product()
    }
}

fn concrete_shape(ty: &TensorType) -> Result<Vec<usize>, String> {
    ty.dims
        .iter()
        .map(|dim| match dim {
            DimInfo::Lit(n) => Ok(*n),
            DimInfo::Named(_, Some(n)) => Ok(*n),
            DimInfo::Named(name, None) => {
                Err(format!("cannot evaluate symbolic dimension `{name}`"))
            }
        })
        .collect()
}

/// chelis#616: like [`concrete_shape`], but an unbound named dim may resolve
/// through `runtime_dims` — the mid-evaluation bindings of op-declared
/// runtime extents (node-valued movement / reshape output dims).
fn concrete_shape_with(
    ty: &TensorType,
    runtime_dims: &UnordMap<String, usize>,
) -> Result<Vec<usize>, String> {
    ty.dims
        .iter()
        .map(|dim| match dim {
            DimInfo::Lit(n) => Ok(*n),
            DimInfo::Named(_, Some(n)) => Ok(*n),
            DimInfo::Named(name, None) => runtime_dims
                .get(name)
                .copied()
                .ok_or_else(|| format!("cannot evaluate symbolic dimension `{name}`")),
        })
        .collect()
}

/// Zero-filled tensor at the type's declared dtype (missing non-strict
/// Load inputs). Zero is a member of every active dtype, so this cannot
/// trap.
fn default_value(ty: &TensorType) -> TensorValue {
    let shape = concrete_shape(ty).unwrap_or_default();
    let n = numel(&shape);
    let storage = if ty.precision.is_float() {
        finalize_tensor("load", ty.precision, RawTensor::Float(vec![0.0; n]))
    } else {
        finalize_tensor("load", ty.precision, RawTensor::Int(vec![0; n]))
    }
    .expect("zero is a member of every active dtype");
    TensorValue::from_storage(shape, storage)
}

/// Freeze a raw-f64 host input once at the Load's declared dtype, or accept
/// an already-tagged value only when its storage dtype matches the Load
/// declaration. A tagged mismatch is not an implicit cast: [04-NUM-11]
/// requires the carrier to preserve its declared dtype exactly.
fn ingress_to_declared(
    name: &str,
    declared: Prim,
    value: &TensorValue,
) -> Result<TensorValue, String> {
    if value.raw_f64_ingress {
        let storage = finalize_tensor("load", declared, value.storage().to_raw())
            .map_err(|trap| format!("input `{name}`: {trap}"))?;
        return Ok(TensorValue::from_storage(value.shape.clone(), storage));
    }
    if value.prim() == declared {
        return Ok(value.clone());
    }
    Err(format!(
        "input `{name}` carries dtype {} but the Load declares {}; provide a value with the declared dtype (casts are explicit in Chelis)",
        value.prim().name(),
        declared.name()
    ))
}

/// Legacy wide-buffer adapter for non-elementwise paths that still produce
/// integral f64 images before finalization. Elementwise arithmetic and static
/// condition folding never enter this adapter: their closed kernels preserve
/// the exact integer storage width.
fn wide_i64_saturating(x: f64) -> i64 {
    x as i64
}

/// Finalize a computed wide buffer at the node's output dtype. THE
/// construction path for every compute op in this evaluator (section C1:
/// kernels compute wide, finalize once per op).
fn finalize_wide(
    op: &'static str,
    prim: Prim,
    shape: Vec<usize>,
    wide: Vec<f64>,
) -> Result<TensorValue, String> {
    let raw = if prim.is_float() {
        RawTensor::Float(wide)
    } else if wide.iter().all(|x| x.is_finite() && x.fract() == 0.0) {
        RawTensor::Int(wide.into_iter().map(wide_i64_saturating).collect())
    } else {
        // A fractional/non-finite element at an integer or bool dtype
        // must Domain-trap through finalize, never truncate.
        RawTensor::Float(wide)
    };
    let storage = finalize_tensor(op, prim, raw).map_err(|trap| trap.to_string())?;
    Ok(TensorValue::from_storage(shape, storage))
}

/// Exact integer finalize for paths that already computed in i64
/// (reductions, casts, one_hot index writes).
fn finalize_wide_int(
    op: &'static str,
    prim: Prim,
    shape: Vec<usize>,
    wide: Vec<i64>,
) -> Result<TensorValue, String> {
    let storage =
        finalize_tensor(op, prim, RawTensor::Int(wide)).map_err(|trap| trap.to_string())?;
    Ok(TensorValue::from_storage(shape, storage))
}

/// Tensor `cast`: consume the shared source/target plan once, apply it to
/// sealed elements, and reduce every trap candidate by row-major flat index.
/// There is no whole-buffer domain pre-pass and no second finalization pass.
fn cast_value(input: &TensorValue, src: Prim, dst: Prim) -> Result<TensorValue, String> {
    if input.prim() != src {
        return Err(format!(
            "checked cast source contract mismatch: declared {}, stored {}",
            src.name(),
            input.prim().name()
        ));
    }
    let plan = CheckedCastPlan::new(src, dst).map_err(|error| error.to_string())?;
    if plan.kind() == chelis_types::CheckedCastKind::Identity {
        return Ok(input.clone());
    }

    let mut cast_values = Vec::with_capacity(input.storage().len());
    let mut selected_trap: Option<IndexedTrapCandidate> = None;
    for flat_index in 0..input.storage().len() {
        match plan.cast_scalar("cast", input.storage().scalar_at(flat_index)) {
            Ok(value) => cast_values.push(value),
            Err(trap) => {
                let candidate = IndexedTrapCandidate { flat_index, trap };
                selected_trap = Some(match selected_trap {
                    Some(selected) => selected.earlier(candidate),
                    None => candidate,
                });
            }
        }
    }
    if let Some(candidate) = selected_trap {
        return Err(candidate.trap.to_string());
    }
    let storage = tensor_from_scalars(dst, &cast_values);
    Ok(TensorValue::from_storage(input.shape.clone(), storage))
}

/// Public entry point for the [05-OP-6] tensor rung, mirroring
/// [`cast_tensor`] so the host runtime and the DAG evaluator share one
/// kernel.
pub fn cast_trunc_tensor(input: &TensorValue, dst: Prim) -> Result<TensorValue, String> {
    cast_trunc_value(input, dst)
}

/// Tensor `cast_trunc` ([05-OP-6]): truncate every float element toward
/// zero, then finalize at the integer target. The source is float and the
/// target an integer width by the checker's contract, so the sealed
/// kernel is reached with exactly the shape it accepts.
fn cast_trunc_value(input: &TensorValue, dst: Prim) -> Result<TensorValue, String> {
    let storage = chelis_types::cast_trunc_tensor("cast_trunc", input.storage().to_raw(), dst)
        .map_err(|trap| trap.to_string())?;
    Ok(TensorValue::from_storage(input.shape.clone(), storage))
}

fn dropout(input: &TensorValue, rate: f64, seed: u64, prim: Prim) -> Result<TensorValue, String> {
    let keep_scale = if rate >= 1.0 {
        0.0
    } else {
        1.0 / (1.0 - rate.max(0.0))
    };
    let data = input
        .to_f64_lossy_vec()
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let sample = dropout_sample(seed, index as u64);
            if sample < rate {
                0.0
            } else {
                value * keep_scale
            }
        })
        .collect();
    finalize_wide("dropout", prim, input.shape.clone(), data)
}

fn dropout_sample(seed: u64, index: u64) -> f64 {
    let mut x = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 11) as f64) / ((1u64 << 53) as f64)
}

fn uniform_like(
    shape: &[usize],
    low: f64,
    high: f64,
    seed: u64,
    prim: Prim,
) -> Result<TensorValue, String> {
    let low_f = low as f32;
    let high_f = high as f32;
    let values = (0..numel(shape))
        .map(|index| uniform_sample(prim, low_f, high_f, seed, index as u64))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| err.to_string())?;
    Ok(TensorValue::from_storage(
        shape.to_vec(),
        tensor_from_scalars(prim, &values),
    ))
}

/// Closed elementwise binary vocabulary for the IR evaluator. The enum is
/// translated to the per-family public kernel only after the finalized
/// operand storage has selected the family; no raw closure or string can
/// become an arithmetic entry point.
#[derive(Debug, Clone, Copy)]
enum ElementwiseBinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    TruncDiv,
    Mod,
    Max,
    Min,
}

impl ElementwiseBinOp {
    const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::FloorDiv => "floor_div",
            Self::TruncDiv => "trunc_div",
            Self::Mod => "mod",
            Self::Max => "max_elem",
            Self::Min => "min_elem",
        }
    }

    const fn int_op(self) -> Option<IntBinOp> {
        match self {
            Self::Add => Some(IntBinOp::Add),
            Self::Sub => Some(IntBinOp::Sub),
            Self::Mul => Some(IntBinOp::Mul),
            Self::FloorDiv => Some(IntBinOp::FloorDiv),
            Self::TruncDiv => Some(IntBinOp::TruncDiv),
            Self::Mod => Some(IntBinOp::Rem),
            Self::Max => Some(IntBinOp::Max),
            Self::Min => Some(IntBinOp::Min),
            Self::Div => None,
        }
    }

    const fn float_op(self) -> Option<FloatBinOp> {
        match self {
            Self::Add => Some(FloatBinOp::Add),
            Self::Sub => Some(FloatBinOp::Sub),
            Self::Mul => Some(FloatBinOp::Mul),
            Self::Div => Some(FloatBinOp::Div),
            Self::FloorDiv => Some(FloatBinOp::FloorDiv),
            Self::Max => Some(FloatBinOp::Max),
            Self::Min => Some(FloatBinOp::Min),
            Self::TruncDiv | Self::Mod => None,
        }
    }
}

/// chelis#664 on the eval lane: an elementwise op indexes every operand
/// through the output's shape, so operands that disagree at run time are a
/// typed error, never an assertion. The routing of host-lane def applications
/// through this evaluator (chelis#1277 B2h) made a runtime disagreement user
/// input; the phrase is the one the host interpreter reports.
fn shape_disagreement(lhs: &TensorValue, rhs: &TensorValue) -> String {
    let render = |shape: &[usize]| {
        let extents = shape.iter().map(usize::to_string).collect::<Vec<_>>();
        format!("[{}]", extents.join(", "))
    };
    format!(
        "tensor shapes must match for elementwise op, got {} vs {}",
        render(&lhs.shape),
        render(&rhs.shape)
    )
}

/// Validate a producer's complete positive-rank agreement relation before
/// reading the extent used by its declared-result claim.
fn same_shape_agreement_extent(
    agreement: &crate::axis_sources::SameShapeAgreement,
    axis: usize,
    values: &UnordMap<NodeId, TensorValue>,
) -> Result<usize, String> {
    let first_id = *agreement
        .members()
        .first()
        .ok_or("same-shape result has no positive-rank agreement member")?;
    let first = values.get(&first_id).ok_or_else(|| {
        format!(
            "same-shape agreement member {} is not available",
            first_id.0
        )
    })?;
    if first.shape.is_empty() {
        return Err(format!(
            "same-shape agreement member {} realized rank zero",
            first_id.0
        ));
    }
    for member in &agreement.members()[1..] {
        let value = values
            .get(member)
            .ok_or_else(|| format!("same-shape agreement member {} is not available", member.0))?;
        if value.shape != first.shape {
            return Err(shape_disagreement(first, value));
        }
    }
    first.shape.get(axis).copied().ok_or_else(|| {
        format!(
            "same-shape result axis {axis} is outside agreed rank {}",
            first.shape.len()
        )
    })
}

/// Broadcast a rank-0 operand over the other's shape, and report any other
/// disagreement as a typed error.
///
/// spec/05 §2.4.1 permits rank-zero operands as the backend's broadcast
/// idiom. Compiler-generated arithmetic, including Tier 2 lowering, uses it.
/// Source-level arithmetic and comparisons still require matching surfaces;
/// #1621 enforces that rule for bounded scalar dtype binders as well. The
/// comparison evaluator separately requires matching shapes under [05-OP-36].
///
/// The shape check below is a third job and belongs to neither: it is what
/// makes `[4]` against `[3]` report the interpreter's own phrase, which
/// `an_elementwise_operand_shape_disagreement_is_a_typed_error_not_a_panic`
/// locks.
fn broadcast_rank0_operands<'a>(
    lhs: &'a TensorValue,
    rhs: &'a TensorValue,
) -> Result<(Cow<'a, TensorValue>, Cow<'a, TensorValue>), String> {
    if lhs.shape == rhs.shape {
        return Ok((Cow::Borrowed(lhs), Cow::Borrowed(rhs)));
    }
    if lhs.shape.is_empty() {
        return Ok((Cow::Owned(splat_rank0(lhs, &rhs.shape)), Cow::Borrowed(rhs)));
    }
    if rhs.shape.is_empty() {
        return Ok((Cow::Borrowed(lhs), Cow::Owned(splat_rank0(rhs, &lhs.shape))));
    }
    Err(shape_disagreement(lhs, rhs))
}

/// The rank-0 element repeated over `shape`, at the operand's own dtype.
/// `reuse_gather` is the element-preserving gather the movement ops use
/// (section C3); a rank-0 broadcast is `expand` of one element, so it
/// qualifies: every output element is `value[0]`, never re-finalized.
fn splat_rank0(value: &TensorValue, shape: &[usize]) -> TensorValue {
    TensorValue {
        storage: value.storage.reuse_gather(&vec![0; numel(shape)]),
        shape: shape.to_vec(),
        raw_f64_ingress: value.raw_f64_ingress,
    }
}

/// Elementwise comparison operands must agree on shape exactly. chelis#1506
/// removed the rank-0 broadcast from this family: `[05-OP-36]` makes a scalar
/// beside a tensor a type error, so the lane preservation B2h added for it has
/// no admitted program left to preserve.
fn require_matching_comparison_shapes(lhs: &TensorValue, rhs: &TensorValue) -> Result<(), String> {
    if lhs.shape == rhs.shape {
        return Ok(());
    }
    Err(shape_disagreement(lhs, rhs))
}

fn binary_elementwise(
    op: ElementwiseBinOp,
    lhs: &TensorValue,
    rhs: &TensorValue,
) -> Result<TensorValue, String> {
    let (lhs, rhs) = broadcast_rank0_operands(lhs, rhs)?;
    let (lhs, rhs) = (&*lhs, &*rhs);
    let storage = if lhs.prim() == Prim::Bool && rhs.prim() == Prim::Bool {
        let lhs_values = lhs
            .storage()
            .to_i64_exact_vec()
            .expect("sealed bool storage has an exact integer view");
        let rhs_values = rhs
            .storage()
            .to_i64_exact_vec()
            .expect("sealed bool storage has an exact integer view");
        let values = match op {
            // Tier-2 lowers `and` and `or` to these two RISC operations.
            // They remain logical operations over sealed bool storage; no
            // numeric-family kernel or raw closure is involved.
            ElementwiseBinOp::Mul => lhs_values
                .into_iter()
                .zip(rhs_values)
                .map(|(lhs, rhs)| i64::from(lhs != 0 && rhs != 0))
                .collect(),
            ElementwiseBinOp::Max => lhs_values
                .into_iter()
                .zip(rhs_values)
                .map(|(lhs, rhs)| i64::from(lhs != 0 || rhs != 0))
                .collect(),
            _ => {
                return Err(format!(
                    "{}: bool storage cannot enter a numeric IR kernel",
                    op.name()
                ));
            }
        };
        return finalize_wide_int(op.name(), Prim::Bool, lhs.shape.clone(), values);
    } else if lhs.prim().is_integer() {
        let kernel = op.int_op().ok_or_else(|| {
            format!(
                "{}: integer operands cannot enter a float-only IR kernel",
                op.name()
            )
        })?;
        int_tensor_binop(kernel, lhs.storage(), rhs.storage())
    } else if lhs.prim().is_float() {
        let kernel = op.float_op().ok_or_else(|| {
            format!(
                "{}: float operands cannot enter an integer-only IR kernel",
                op.name()
            )
        })?;
        float_tensor_binop(kernel, lhs.storage(), rhs.storage())
    } else {
        return Err(format!(
            "{}: dtype `{}` has no numeric IR kernel",
            op.name(),
            lhs.prim().name()
        ));
    }
    .map_err(|error| error.to_string())?;
    Ok(TensorValue::from_storage(lhs.shape.clone(), storage))
}

/// Closed unary twin of [`ElementwiseBinOp`].
#[derive(Debug, Clone, Copy)]
enum ElementwiseUnOp {
    Neg,
    Recip,
    Exp,
    Log,
    Sin,
    Sqrt,
    Cos,
    Tan,
    Atan,
    Abs,
    Floor,
    Ceil,
    Round,
}

impl ElementwiseUnOp {
    const fn name(self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Recip => "recip",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Sqrt => "sqrt",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Atan => "atan",
            Self::Abs => "abs",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
        }
    }

    const fn int_op(self) -> Option<IntUnOp> {
        match self {
            Self::Neg => Some(IntUnOp::Neg),
            Self::Abs => Some(IntUnOp::Abs),
            Self::Floor => Some(IntUnOp::Floor),
            Self::Ceil => Some(IntUnOp::Ceil),
            Self::Round => Some(IntUnOp::Round),
            Self::Recip
            | Self::Exp
            | Self::Log
            | Self::Sin
            | Self::Sqrt
            | Self::Cos
            | Self::Tan
            | Self::Atan => None,
        }
    }

    const fn float_op(self) -> FloatUnOp {
        match self {
            Self::Neg => FloatUnOp::Neg,
            Self::Recip => FloatUnOp::Recip,
            Self::Exp => FloatUnOp::Exp,
            Self::Log => FloatUnOp::Log,
            Self::Sin => FloatUnOp::Sin,
            Self::Sqrt => FloatUnOp::Sqrt,
            Self::Cos => FloatUnOp::Cos,
            Self::Tan => FloatUnOp::Tan,
            Self::Atan => FloatUnOp::Atan,
            Self::Abs => FloatUnOp::Abs,
            Self::Floor => FloatUnOp::Floor,
            Self::Ceil => FloatUnOp::Ceil,
            Self::Round => FloatUnOp::Round,
        }
    }
}

fn unary_elementwise(op: ElementwiseUnOp, input: &TensorValue) -> Result<TensorValue, String> {
    let storage = if input.prim().is_integer() {
        let kernel = op.int_op().ok_or_else(|| {
            format!(
                "{}: integer operands cannot enter a float-only IR kernel",
                op.name()
            )
        })?;
        int_tensor_unop(kernel, input.storage())
    } else if input.prim().is_float() {
        float_tensor_unop(op.float_op(), input.storage())
    } else {
        return Err(format!(
            "{}: dtype `{}` has no numeric IR kernel",
            op.name(),
            input.prim().name()
        ));
    }
    .map_err(|error| error.to_string())?;
    Ok(TensorValue::from_storage(input.shape.clone(), storage))
}

fn compare_elementwise(
    op: CompareOp,
    lhs: &TensorValue,
    rhs: &TensorValue,
) -> Result<TensorValue, String> {
    require_matching_comparison_shapes(lhs, rhs)?;
    let storage =
        compare_tensors(op, lhs.storage(), rhs.storage()).map_err(|error| error.to_string())?;
    Ok(TensorValue::from_storage(lhs.shape.clone(), storage))
}

fn matmul(lhs: &TensorValue, rhs: &TensorValue, prim: Prim) -> Result<TensorValue, String> {
    assert_eq!(lhs.shape.len(), 2);
    assert_eq!(rhs.shape.len(), 2);
    let m = lhs.shape[0];
    let k = lhs.shape[1];
    assert_eq!(rhs.shape[0], k);
    let n = rhs.shape[1];
    let a = lhs.to_f64_lossy_vec();
    let b = rhs.to_f64_lossy_vec();
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            data[i * n + j] = acc;
        }
    }
    finalize_wide("matmul", prim, vec![m, n], data)
}

fn batched_matmul(lhs: &TensorValue, rhs: &TensorValue, prim: Prim) -> Result<TensorValue, String> {
    assert!(lhs.shape.len() >= 2);
    assert!(rhs.shape.len() >= 2);
    assert_eq!(lhs.shape.len(), rhs.shape.len());
    let rank = lhs.shape.len();
    let batch = &lhs.shape[..rank - 2];
    assert_eq!(batch, &rhs.shape[..rank - 2]);
    let m = lhs.shape[rank - 2];
    let k = lhs.shape[rank - 1];
    assert_eq!(rhs.shape[rank - 2], k);
    let n = rhs.shape[rank - 1];
    let batch_count = batch.iter().product::<usize>();
    let mut out_shape = batch.to_vec();
    out_shape.extend([m, n]);
    let a = lhs.to_f64_lossy_vec();
    let b = rhs.to_f64_lossy_vec();
    let mut data = vec![0.0; batch_count * m * n];
    for batch_idx in 0..batch_count {
        let lhs_base = batch_idx * m * k;
        let rhs_base = batch_idx * k * n;
        let out_base = batch_idx * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for kk in 0..k {
                    acc += a[lhs_base + i * k + kk] * b[rhs_base + kk * n + j];
                }
                data[out_base + i * n + j] = acc;
            }
        }
    }
    finalize_wide("matmul", prim, out_shape, data)
}

/// Read one gather/scatter index tensor element: exact for the integer
/// family, the pre-refactor f64 truncation for float storage (index
/// tensors are i64-typed after ingress, so the float arm is legacy
/// tolerance for f64-storage fixtures driving untyped index inputs).
fn index_at(indices: &TensorValue, linear: usize) -> isize {
    match indices.storage().to_raw() {
        RawTensor::Int(v) => v[linear] as isize,
        RawTensor::Float(v) => v[linear] as isize,
    }
}

fn gather(values: &TensorValue, indices: &TensorValue, axis: usize) -> TensorValue {
    assert!(axis < values.shape.len());
    let index_rank = indices.shape.len();
    let mut out_shape = Vec::with_capacity(values.shape.len() - 1 + index_rank);
    out_shape.extend_from_slice(&values.shape[..axis]);
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&values.shape[axis + 1..]);
    let mut picks = Vec::with_capacity(numel(&out_shape));
    for out_linear in 0..numel(&out_shape) {
        let out_index = linear_to_index(out_linear, &out_shape);
        let mut idx_index = Vec::with_capacity(index_rank);
        for pos in 0..index_rank {
            idx_index.push(out_index[axis + pos]);
        }
        let gathered = index_at(indices, index_to_linear(&idx_index, &indices.shape));
        assert!(
            gathered >= 0 && (gathered as usize) < values.shape[axis],
            "gather index {gathered} out of bounds for axis {axis}"
        );
        let mut value_index = Vec::with_capacity(values.shape.len());
        value_index.extend_from_slice(&out_index[..axis]);
        value_index.push(gathered as usize);
        value_index.extend_from_slice(&out_index[axis + index_rank..]);
        picks.push(index_to_linear(&value_index, &values.shape));
    }
    // reuse_* contract: gather is element-preserving (section C3).
    TensorValue::from_storage(out_shape.clone(), values.storage().reuse_gather(&picks))
}

fn scatter_add(
    target: &TensorValue,
    indices: &TensorValue,
    updates: &TensorValue,
    axis: usize,
    prim: Prim,
) -> Result<TensorValue, String> {
    assert!(axis < target.shape.len());
    let index_rank = indices.shape.len();
    let mut expected_updates = Vec::with_capacity(target.shape.len() - 1 + index_rank);
    expected_updates.extend_from_slice(&target.shape[..axis]);
    expected_updates.extend_from_slice(&indices.shape);
    expected_updates.extend_from_slice(&target.shape[axis + 1..]);
    assert_eq!(updates.shape, expected_updates);

    let mut out = target.to_f64_lossy_vec();
    let upd = updates.to_f64_lossy_vec();
    for (update_linear, update) in upd.iter().enumerate() {
        let update_index = linear_to_index(update_linear, &updates.shape);
        let mut idx_index = Vec::with_capacity(index_rank);
        for pos in 0..index_rank {
            idx_index.push(update_index[axis + pos]);
        }
        let gathered = index_at(indices, index_to_linear(&idx_index, &indices.shape));
        assert!(
            gathered >= 0 && (gathered as usize) < target.shape[axis],
            "scatter_add index {gathered} out of bounds for axis {axis}"
        );
        let mut target_index = Vec::with_capacity(target.shape.len());
        target_index.extend_from_slice(&update_index[..axis]);
        target_index.push(gathered as usize);
        target_index.extend_from_slice(&update_index[axis + index_rank..]);
        let target_linear = index_to_linear(&target_index, &target.shape);
        out[target_linear] += update;
    }
    finalize_wide("scatter_add", prim, target.shape.clone(), out)
}

/// Replace-scatter (last-write-wins) over duplicate target indices.
///
/// Per `spec/05-risc-primitives.md` §3.5, the deterministic-order
/// rule is **updates-tensor row-major (C order) flat iteration**:
/// updates are written into the target in ascending flat-index order
/// over `updates.shape`. When two updates target the same cell, the
/// write with the larger flat index in `updates` is the final value.
/// This is intentionally distinct from `scatter_add` (whose
/// duplicate-index semantics are commutative accumulation) and is
/// the reason replace-scatter has no well-defined AD adjoint.
fn scatter_replace(
    target: &TensorValue,
    indices: &TensorValue,
    updates: &TensorValue,
    axis: usize,
) -> TensorValue {
    assert!(axis < target.shape.len());
    let index_rank = indices.shape.len();
    let mut expected_updates = Vec::with_capacity(target.shape.len() - 1 + index_rank);
    expected_updates.extend_from_slice(&target.shape[..axis]);
    expected_updates.extend_from_slice(&indices.shape);
    expected_updates.extend_from_slice(&target.shape[axis + 1..]);
    assert_eq!(updates.shape, expected_updates);

    let mut writes = Vec::with_capacity(updates.len());
    for update_linear in 0..updates.len() {
        let update_index = linear_to_index(update_linear, &updates.shape);
        let mut idx_index = Vec::with_capacity(index_rank);
        for pos in 0..index_rank {
            idx_index.push(update_index[axis + pos]);
        }
        let gathered = index_at(indices, index_to_linear(&idx_index, &indices.shape));
        assert!(
            gathered >= 0 && (gathered as usize) < target.shape[axis],
            "scatter_replace index {gathered} out of bounds for axis {axis}"
        );
        let mut target_index = Vec::with_capacity(target.shape.len());
        target_index.extend_from_slice(&update_index[..axis]);
        target_index.push(gathered as usize);
        target_index.extend_from_slice(&update_index[axis + index_rank..]);
        // Last-write-wins: deterministic-order overwrite.
        writes.push((index_to_linear(&target_index, &target.shape), update_linear));
    }
    // reuse_* contract: replace-scatter moves existing elements only
    // (section C3, element-preserving).
    TensorValue::from_storage(
        target.shape.clone(),
        target.storage().reuse_overwrite(updates.storage(), writes),
    )
}

/// Element-wise replace-scatter with ONNX `ScatterElements` semantics
/// (`spec/05-risc-primitives.md` §3.5.1).
///
/// Distinct from `scatter_replace` (hyperplane semantics): here
/// `data`/`indices`/`updates` share a rank, `indices.shape ==
/// updates.shape`, and `output.shape == data.shape`. For each
/// coordinate `c` over `indices`, the value `updates[c]` is written to
/// `output` at the coordinate `c` with its `axis` component replaced by
/// `indices[c]`. Duplicate writes resolve last-write-wins in
/// updates-tensor row-major (C order) flat-iteration order, matching
/// `scatter_replace`'s determinism rule; AD is fail-closed for the same
/// reason.
fn scatter_elements(
    data: &TensorValue,
    indices: &TensorValue,
    updates: &TensorValue,
    axis: usize,
) -> TensorValue {
    assert!(axis < data.shape.len());
    assert_eq!(
        indices.shape, updates.shape,
        "scatter_elements requires indices.shape == updates.shape"
    );
    assert_eq!(
        indices.shape.len(),
        data.shape.len(),
        "scatter_elements requires data, indices, and updates to share a rank"
    );

    let mut writes = Vec::with_capacity(updates.len());
    for update_linear in 0..updates.len() {
        let coord = linear_to_index(update_linear, &updates.shape);
        let gathered = index_at(indices, update_linear);
        assert!(
            gathered >= 0 && (gathered as usize) < data.shape[axis],
            "scatter_elements index {gathered} out of bounds for axis {axis}"
        );
        let mut target_index = coord.clone();
        target_index[axis] = gathered as usize;
        // Last-write-wins: deterministic-order overwrite.
        writes.push((index_to_linear(&target_index, &data.shape), update_linear));
    }
    // reuse_* contract: element-preserving overwrite (section C3).
    TensorValue::from_storage(
        data.shape.clone(),
        data.storage().reuse_overwrite(updates.storage(), writes),
    )
}

/// Strided windowed reduction over the trailing `window_shape.len()` axes.
///
/// Per `spec/05-risc-primitives.md` §2.3.1 (Valid padding):
/// - Leading `rank - n` axes pass through unchanged.
/// - Each windowed output axis has extent
///   `floor((input_dim - window_shape[i]) / strides[i]) + 1`.
/// - `Mean` is windowed `Sum` divided by the window volume.
///
/// Panics if the input rank is smaller than `window_shape.len()`, if any
/// window/stride entry is zero, or if any windowed dim has `window > input_dim`.
/// The IR verifier and type checker reject those statically before the
/// evaluator runs.
fn reduce_window(
    input: &TensorValue,
    reducer: ReduceWindowKind,
    window_shape: &[usize],
    strides: &[usize],
    prim: Prim,
) -> Result<TensorValue, String> {
    assert_eq!(
        window_shape.len(),
        strides.len(),
        "window_shape and strides must have equal length"
    );
    let rank = input.shape.len();
    let n = window_shape.len();
    assert!(
        rank >= n,
        "reduce_window: input rank {rank} smaller than window arity {n}"
    );
    let leading = rank - n;

    let mut out_shape = input.shape[..leading].to_vec();
    for i in 0..n {
        let in_dim = input.shape[leading + i];
        let w = window_shape[i];
        let s = strides[i];
        assert!(w >= 1, "reduce_window: window axis {i} must be >= 1");
        assert!(s >= 1, "reduce_window: stride axis {i} must be >= 1");
        assert!(
            in_dim >= w,
            "reduce_window: axis {i} input dim {in_dim} < window {w}"
        );
        out_shape.push((in_dim - w) / s + 1);
    }

    let out_len = numel(&out_shape);
    let mut groups = Vec::with_capacity(out_len);
    for out_flat in 0..out_len {
        let out_idx = linear_to_index(out_flat, &out_shape);
        let mut group = Vec::with_capacity(window_shape.iter().product());
        for_each_window_pos(window_shape, n, |window_pos| {
            let mut src_idx = vec![0usize; rank];
            src_idx[..leading].copy_from_slice(&out_idx[..leading]);
            for i in 0..n {
                src_idx[leading + i] = out_idx[leading + i] * strides[i] + window_pos[i];
            }
            group.push(index_to_linear(&src_idx, &input.shape));
        });
        groups.push(group);
    }
    let op = match reducer {
        ReduceWindowKind::Max => TensorReduceOp::ReduceWindowMax,
        ReduceWindowKind::Min => TensorReduceOp::ReduceWindowMin,
        ReduceWindowKind::Sum => TensorReduceOp::ReduceWindowSum,
        ReduceWindowKind::Mean => TensorReduceOp::ReduceWindowMean,
    };
    let storage =
        reduce_tensor_groups(op, input.storage(), &groups).map_err(|err| err.to_string())?;
    if storage.prim() != prim {
        return Err(format!(
            "{} produced {} storage for {} IR output",
            op.name(),
            storage.prim().name(),
            prim.name()
        ));
    }
    Ok(TensorValue::from_storage(out_shape, storage))
}

/// Reverse-mode adjoint of [`reduce_window`] (see `RiscOp::ReduceWindowGrad`).
///
/// `x` is the original windowed input (shape `S_in`); `g` is the upstream
/// cotangent (shape `S_out`, the forward output). Returns `din` with shape
/// `S_in`, accumulating contributions over every (overlapping) window:
/// - `Sum`:  each window-source position receives the owning window's `g`.
/// - `Mean`: as `Sum`, scaled by `1 / window_volume`.
/// - `Max` / `Min`: the window's `g` is routed to every position equal to
///   that window's max / min (ties distribute, matching the `max_reduce` /
///   `min_reduce` subgradient convention).
///
/// Panics on the same structural violations as [`reduce_window`]; the IR
/// verifier and type checker reject those before the evaluator runs.
fn reduce_window_grad(
    x: &TensorValue,
    g: &TensorValue,
    reducer: ReduceWindowKind,
    window_shape: &[usize],
    strides: &[usize],
    prim: Prim,
) -> Result<TensorValue, String> {
    assert_eq!(
        window_shape.len(),
        strides.len(),
        "window_shape and strides must have equal length"
    );
    let rank = x.shape.len();
    let n = window_shape.len();
    assert!(
        rank >= n,
        "reduce_window_grad: input rank {rank} smaller than window arity {n}"
    );
    let leading = rank - n;
    // Output dim per windowed axis: floor((in - w) / s) + 1 (Valid padding),
    // matching the forward. The cotangent `g` is indexed by this shape.
    let mut out_shape = x.shape[..leading].to_vec();
    for i in 0..n {
        let in_dim = x.shape[leading + i];
        let w = window_shape[i];
        let s = strides[i];
        assert!(w >= 1, "reduce_window_grad: window axis {i} must be >= 1");
        assert!(s >= 1, "reduce_window_grad: stride axis {i} must be >= 1");
        assert!(
            in_dim >= w,
            "reduce_window_grad: axis {i} input dim {in_dim} < window {w}"
        );
        out_shape.push((in_dim - w) / s + 1);
    }
    assert_eq!(
        g.shape, out_shape,
        "reduce_window_grad: cotangent shape {:?} != forward output shape {out_shape:?}",
        g.shape
    );

    let mut groups = Vec::with_capacity(g.len());
    for out_flat in 0..g.len() {
        let out_idx = linear_to_index(out_flat, &out_shape);
        let mut group = Vec::with_capacity(window_shape.iter().product());
        for_each_window_pos(window_shape, n, |window_pos| {
            let mut src_idx = vec![0usize; rank];
            src_idx[..leading].copy_from_slice(&out_idx[..leading]);
            for i in 0..n {
                src_idx[leading + i] = out_idx[leading + i] * strides[i] + window_pos[i];
            }
            group.push(index_to_linear(&src_idx, &x.shape));
        });
        groups.push(group);
    }
    let op = match reducer {
        ReduceWindowKind::Sum => ReduceWindowGradOp::Sum,
        ReduceWindowKind::Mean => ReduceWindowGradOp::Mean,
        ReduceWindowKind::Max => ReduceWindowGradOp::Max,
        ReduceWindowKind::Min => ReduceWindowGradOp::Min,
    };
    let storage = reduce_window_grad_tensor_groups(op, x.storage(), g.storage(), &groups)
        .map_err(|err| err.to_string())?;
    if storage.prim() != prim {
        return Err(format!(
            "{} produced {} storage for {} IR output",
            op.name(),
            storage.prim().name(),
            prim.name()
        ));
    }
    Ok(TensorValue::from_storage(x.shape.clone(), storage))
}

/// Invoke `f` once per multi-index inside an `n`-dimensional window of
/// extent `window_shape` (row-major / mixed-radix order). For `n == 0`
/// (no windowed axes) `f` is invoked once with an empty index.
fn for_each_window_pos(window_shape: &[usize], n: usize, mut f: impl FnMut(&[usize])) {
    let mut window_pos = vec![0usize; n];
    loop {
        f(&window_pos);
        if n == 0 {
            break;
        }
        let mut carry = n;
        for i in (0..n).rev() {
            window_pos[i] += 1;
            if window_pos[i] < window_shape[i] {
                carry = i;
                break;
            }
            window_pos[i] = 0;
        }
        if carry == n {
            break;
        }
    }
}

/// Build the exact, ordered source-index group for every output element of
/// an axis reduction. Shape planning remains local; arithmetic does not.
fn axis_reduction_groups(input_shape: &[usize], axis: usize) -> (Vec<usize>, Vec<Vec<usize>>) {
    assert!(axis < input_shape.len());
    let mut out_shape = input_shape.to_vec();
    let axis_len = out_shape.remove(axis);
    let out_len = numel(&out_shape);
    let mut groups = Vec::with_capacity(out_len);
    for out_flat in 0..out_len {
        let out_index = linear_to_index(out_flat, &out_shape);
        let mut group = Vec::with_capacity(axis_len);
        for axis_index in 0..axis_len {
            let mut input_index = Vec::with_capacity(input_shape.len());
            let mut output_axis = 0;
            for input_axis in 0..input_shape.len() {
                if input_axis == axis {
                    input_index.push(axis_index);
                } else {
                    input_index.push(out_index[output_axis]);
                    output_axis += 1;
                }
            }
            group.push(index_to_linear(&input_index, input_shape));
        }
        groups.push(group);
    }
    (out_shape, groups)
}

/// Axis reduction through the closed Phase 2 typed kernel.
fn reduce(input: &TensorValue, axis: usize, op: TensorReduceOp) -> Result<TensorValue, String> {
    let (out_shape, groups) = axis_reduction_groups(&input.shape, axis);
    let storage =
        reduce_tensor_groups(op, input.storage(), &groups).map_err(|err| err.to_string())?;
    Ok(TensorValue::from_storage(out_shape, storage))
}

/// Reduce along `axis`, tracking the index of the element that wins under
/// `better(current_best, candidate)`. Used for Argmax / Argmin.
///
/// Ties are broken by the smallest index (first-seen wins), matching numpy's
/// default argmax/argmin semantics. The output holds exact i64 indices
/// (the `RiscOp::Argmax` spec invariant; per-dtype storage ended the
/// f64-image detour of chelis#233).
fn reduce_argcmp(input: &TensorValue, axis: usize, op: ArgReduceOp) -> Result<TensorValue, String> {
    let (out_shape, groups) = axis_reduction_groups(&input.shape, axis);
    let storage =
        arg_reduce_tensor_groups(op, input.storage(), &groups).map_err(|err| err.to_string())?;
    Ok(TensorValue::from_storage(out_shape, storage))
}

/// [05-OP-29] multi-axis bool count. Source elements are partitioned into
/// result groups by removing the selected coordinates. Each group is filled
/// by scanning the input in its original row-major order. Arithmetic and the
/// canonical adjacent-pair checked-i64 tree live only in the typed kernel.
pub fn count_tensor(input: &TensorValue, axes: &[usize]) -> Result<TensorValue, String> {
    if axes.is_empty()
        || axes.iter().any(|&axis| axis >= input.shape.len())
        || axes.windows(2).any(|pair| pair[0] <= pair[1])
    {
        return Err(format!(
            "count axes must be non-empty, unique, in range, and strictly descending; got {axes:?}"
        ));
    }

    let selected: UnordSet<usize> = axes.iter().copied().collect();
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &extent)| (!selected.contains(&axis)).then_some(extent))
        .collect();
    let mut groups = vec![Vec::<usize>::new(); numel(&out_shape)];
    for flat in 0..input.len() {
        let input_coord = linear_to_index(flat, &input.shape);
        let output_coord: Vec<usize> = input_coord
            .iter()
            .enumerate()
            .filter_map(|(axis, &coord)| (!selected.contains(&axis)).then_some(coord))
            .collect();
        let group = index_to_linear(&output_coord, &out_shape);
        groups[group].push(flat);
    }
    let storage =
        count_tensor_groups(input.storage(), &groups).map_err(|error| error.to_string())?;
    Ok(TensorValue::from_storage(out_shape, storage))
}

fn reshape(input: &TensorValue, shape: Vec<usize>) -> TensorValue {
    assert_eq!(input.len(), numel(&shape));
    // reuse_* contract: reshape is element-preserving (section C3); the
    // buffer moves unchanged under a new shape.
    TensorValue::from_storage(shape, input.storage().clone())
}

fn permute(input: &TensorValue, axes: &[usize]) -> TensorValue {
    assert_eq!(axes.len(), input.shape.len());
    let out_shape: Vec<usize> = axes.iter().map(|&axis| input.shape[axis]).collect();
    let out_len = numel(&out_shape);
    let mut picks = Vec::with_capacity(out_len);
    for flat_idx in 0..out_len {
        let out_index = linear_to_index(flat_idx, &out_shape);
        let mut in_index = vec![0; input.shape.len()];
        for (out_axis, &in_axis) in axes.iter().enumerate() {
            in_index[in_axis] = out_index[out_axis];
        }
        picks.push(index_to_linear(&in_index, &input.shape));
    }
    // reuse_* contract: permute is element-preserving (section C3).
    TensorValue::from_storage(out_shape, input.storage().reuse_gather(&picks))
}

fn expand(input: &TensorValue, axis: usize, _size: usize, out_shape: Vec<usize>) -> TensorValue {
    assert!(axis <= input.shape.len());
    let out_len = numel(&out_shape);
    let mut picks = Vec::with_capacity(out_len);
    for flat_idx in 0..out_len {
        let out_index = linear_to_index(flat_idx, &out_shape);
        let in_index = if out_shape.len() == input.shape.len() {
            let mut idx = out_index;
            idx[axis] = 0;
            idx
        } else {
            let mut idx = out_index;
            idx.remove(axis);
            idx
        };
        picks.push(index_to_linear(&in_index, &input.shape));
    }
    // reuse_* contract: expand is element-preserving (section C3).
    TensorValue::from_storage(out_shape, input.storage().reuse_gather(&picks))
}

fn one_hot(indices: &TensorValue, vocab: usize, prim: Prim) -> Result<TensorValue, String> {
    let mut out_shape = indices.shape.clone();
    out_shape.push(vocab);
    let mut out = vec![0i64; numel(&out_shape)];
    for index_linear in 0..indices.len() {
        let class = index_at(indices, index_linear);
        assert!(
            class >= 0 && (class as usize) < vocab,
            "one_hot index {class} out of bounds for vocab {vocab}"
        );
        let mut out_index = linear_to_index(index_linear, &indices.shape);
        out_index.push(class as usize);
        out[index_to_linear(&out_index, &out_shape)] = 1;
    }
    finalize_wide_int("one_hot", prim, out_shape, out)
}

/// chelis#616: resolve a movement [`RtDim`] to a concrete extent at eval time.
/// `Lit` is itself; `ToEnd` is the axis's input extent; `Node(i)` reads the
/// rank-0 integer value at `node.inputs[i]`; and `InputAxis` reads an explicit
/// tensor input's shape metadata.
fn resolve_eval_bound(
    bound: &RtDim,
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
    input_extent: usize,
) -> Result<usize, String> {
    match bound {
        RtDim::Lit(n) => Ok(*n),
        RtDim::ToEnd => Ok(input_extent),
        RtDim::Node(i) => {
            let src = values.get(&node.inputs[*i]).ok_or_else(|| {
                format!(
                    "movement bound at node {}: missing value for bound-source input slot {i}",
                    node.id.0
                )
            })?;
            if src.is_empty() {
                return Err(format!(
                    "movement bound at node {}: bound-source input slot {i} is empty",
                    node.id.0
                ));
            }
            let raw = src.storage().scalar_at(0).as_i64_exact().ok_or_else(|| {
                format!(
                    "movement bound at node {}: bound-source input slot {i} must be i64",
                    node.id.0
                )
            })?;
            if raw < 0 {
                return Err(format!(
                    "movement bound at node {}: bound-source (slot {i}) must be a non-negative \
                     integer, got {raw}",
                    node.id.0
                ));
            }
            usize::try_from(raw).map_err(|_| {
                format!(
                    "movement bound at node {}: bound-source (slot {i}) exceeds host extent capacity: {raw}",
                    node.id.0
                )
            })
        }
        RtDim::InputAxis {
            tensor,
            axis: crate::dag::RtAxis::Lit(axis),
        } => {
            let source_id = node.inputs.get(*tensor).ok_or_else(|| {
                format!(
                    "movement bound at node {}: missing InputAxis tensor slot {tensor}",
                    node.id.0
                )
            })?;
            let source = values.get(source_id).ok_or_else(|| {
                format!(
                    "movement bound at node {}: missing value for InputAxis tensor slot {tensor}",
                    node.id.0
                )
            })?;
            let axis = usize::try_from(*axis).map_err(|_| {
                format!(
                    "movement bound at node {}: InputAxis axis {axis} was not normalized",
                    node.id.0
                )
            })?;
            source.shape.get(axis).copied().ok_or_else(|| {
                format!(
                    "movement bound at node {}: InputAxis axis {axis} out of bounds for rank {} tensor in slot {tensor}",
                    node.id.0,
                    source.shape.len()
                )
            })
        }
        // `Sym` is legal only in a `Reshape` target and is rewritten to
        // `Lit` by `bind_symbolic_dims` before evaluation; a movement bound
        // never carries it (verify rejects it there).
        RtDim::Sym(name) => Err(format!(
            "movement bound at node {}: unbound symbolic dim `{name}` reached the evaluator",
            node.id.0
        )),
    }
}

/// chelis#616: resolve a `(start, end)` bound-pair list against the input shape
/// (used for `Pad` / `Shrink`).
fn resolve_eval_pairs(
    bounds: &[(RtDim, RtDim)],
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
    input_shape: &[usize],
) -> Result<Vec<(usize, usize)>, String> {
    bounds
        .iter()
        .enumerate()
        .map(|(axis, (s, e))| {
            let extent = input_shape.get(axis).copied().unwrap_or(0);
            Ok((
                resolve_eval_bound(s, node, values, extent)?,
                resolve_eval_bound(e, node, values, extent)?,
            ))
        })
        .collect()
}

/// The stride operation's runtime domain failure, in the compiled lane's
/// exact words. The signed step is checked against this domain before any
/// unsigned conversion, extent division, allocation, or element access.
const STRIDE_DOMAIN_TRAP: &str = "Domain: stride step must be positive\n\
                                  numeric trap: domain in stride at i64";

fn resolve_eval_stride_step(
    step: &RtDim,
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
    input_extent: usize,
) -> Result<std::num::NonZeroUsize, String> {
    let resolved = match step {
        RtDim::Lit(value) => *value,
        RtDim::Node(i) => {
            let src = values.get(&node.inputs[*i]).ok_or_else(|| {
                format!(
                    "stride step at node {}: missing value for step-source input slot {i}",
                    node.id.0
                )
            })?;
            if src.is_empty() {
                return Err(format!(
                    "stride step at node {}: step-source input slot {i} is empty",
                    node.id.0
                ));
            }
            let raw = src.storage().scalar_at(0).as_i64_exact().ok_or_else(|| {
                format!(
                    "stride step at node {}: step-source input slot {i} must be i64",
                    node.id.0
                )
            })?;
            if raw <= 0 {
                return Err(STRIDE_DOMAIN_TRAP.to_string());
            }
            usize::try_from(raw).map_err(|_| {
                format!(
                    "stride step at node {}: step-source input slot {i} exceeds host extent capacity: {raw}",
                    node.id.0
                )
            })?
        }
        RtDim::ToEnd | RtDim::Sym(_) | RtDim::InputAxis { .. } => {
            resolve_eval_bound(step, node, values, input_extent)?
        }
    };
    std::num::NonZeroUsize::new(resolved).ok_or_else(|| STRIDE_DOMAIN_TRAP.to_string())
}

/// chelis#616: resolve a `Stride` step list against the input shape.
fn resolve_eval_strides(
    strides: &[RtDim],
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
    input_shape: &[usize],
) -> Result<Vec<std::num::NonZeroUsize>, String> {
    strides
        .iter()
        .enumerate()
        .map(|(axis, b)| {
            let extent = input_shape.get(axis).copied().unwrap_or(0);
            resolve_eval_stride_step(b, node, values, extent)
        })
        .collect()
}

fn pad(
    input: &TensorValue,
    padding: &[(usize, usize)],
    fill: chelis_types::ScalarValue,
) -> Result<TensorValue, String> {
    assert_eq!(padding.len(), input.shape.len());
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(padding.iter())
        .map(|(dim, (before, after))| dim + before + after)
        .collect();
    let mut map: Vec<Option<usize>> = vec![None; numel(&out_shape)];
    for flat_idx in 0..input.len() {
        let in_index = linear_to_index(flat_idx, &input.shape);
        let out_index: Vec<usize> = in_index
            .iter()
            .zip(padding.iter())
            .map(|(idx, (before, _))| idx + before)
            .collect();
        map[index_to_linear(&out_index, &out_shape)] = Some(flat_idx);
    }
    if fill.prim() != input.prim() {
        return Err(format!(
            "pad fill dtype {} does not match input dtype {}",
            fill.prim().name(),
            input.prim().name()
        ));
    }
    // reuse_* contract: pad moves existing elements and places a
    // finalized fill (section C3, element-preserving).
    Ok(TensorValue::from_storage(
        out_shape,
        input.storage().reuse_fill_gather(&fill, &map),
    ))
}

/// The out-of-domain `shrink` bound rejection, in the compiled lane's exact
/// words.
///
/// `spec/05-risc-primitives.md` section 2.4.1 makes "a shrink range overshoot"
/// one of the runtime-bound errors that "are validated in every execution mode
/// with matching language errors", so this lane does not get to invent its own
/// text for it. The compiled lane reaches the same rejection through
/// `chelis_tensor_affine_plan`: `ShapeMetadata::shrunk`
/// (`crates/chelis-abi/src/metadata.rs`) answers a negative start, an inverted
/// pair, or an end past the operand extent with ONE `Domain` message, and
/// `affine_result` (`crates/chelis-runtime/src/lib.rs`) prints that message and
/// then [04-NUM-9]'s trap line before exiting. These are those two lines,
/// verbatim and in that order.
///
/// The three conditions share one message because the compiled lane gives them
/// one message. Splitting them here to name the axis would read better in
/// isolation and would be a lane divergence in exactly the sentence section
/// 2.4.1 writes, so the compiled lane's wording wins.
///
/// The eval lane's reporter prefixes `error: ` to the first line of every
/// diagnostic it raises, as it does to the extent guard's [04-NUM-9] line; the
/// message body below is what the two lanes hold in common. Where this lane
/// raises the diagnostic DIRECTLY that prefix is the whole of the difference.
/// A host transform adds its own wrapper above it: `grad` renders
/// ``host runtime `grad` evaluation failed: `` before this text
/// (`chelis-compiler-api/src/runtime/transforms.rs`). That wrapper is
/// pre-existing and uniform over every error it carries, so chelis#1797
/// neither introduces nor closes it; the text is merely newly reachable there,
/// because the same program used to panic. The exit status agrees either way.
const SHRINK_DOMAIN_TRAP: &str = "Domain: shrink bounds outside input extent\n\
                                  numeric trap: domain in shrink at i64";

fn shrink(input: &TensorValue, bounds: &[(usize, usize)]) -> Result<TensorValue, String> {
    assert_eq!(bounds.len(), input.shape.len());
    // chelis#368: defensive backstop for the `SHRINK_TO_END` full-axis
    // sentinel (the Pad adjoint of the differentiable `concat` over an
    // unpadded axis). `bind_symbolic_dims` normally resolves the sentinel to
    // the axis's bound extent before eval (and the `needs_symbolic_binding`
    // gate now routes every sentinel-bearing DAG through it), so this branch
    // should not fire in the normal flow. If a sentinel ever does survive,
    // resolve it here to the axis's runtime extent — `SHRINK_TO_END` means
    // "to the end of this axis", which is exactly `input.shape[axis]` — rather
    // than letting `end - start = usize::MAX` overflow `numel`.
    // chelis#523: out-of-domain bounds. `verify`'s C10 shrink check runs BEFORE
    // `bind_symbolic_dims` and skips symbolic axes, and
    // `verify_bound_movement_node` can only check a compile-time `(Lit, Lit)`
    // pair, so a shrink whose `end` exceeds the (now concrete) input extent, or
    // whose `start > end`, reaches here and would either underflow
    // `end - start` (usize) or read out-of-bounds in the copy loop below (a
    // silent wrong gradient on higher rank, where the flat index can alias a
    // valid slot). Do NOT clamp to a valid range: that would hide the producing
    // pass's bug.
    //
    // chelis#1797: this rejection is a typed `Err`, not an `assert!`. A runtime
    // (`RtDim::Node`) end is a perfectly ordinary user-reachable value - the
    // issue's reproducer computes `shape(x, 0) + 3` - so the answer to it is
    // the language error section 2.4.1 names, which the lane boundary renders
    // as a diagnostic. A panic was neither that error nor any other, and it
    // exited 101 where the compiled lane exits 1.
    let mut out_shape: Vec<usize> = Vec::with_capacity(input.shape.len());
    for (extent, (start, end)) in input.shape.iter().zip(bounds.iter()) {
        let end = if *end == SHRINK_TO_END { *extent } else { *end };
        if *start > end || end > *extent {
            return Err(SHRINK_DOMAIN_TRAP.to_string());
        }
        out_shape.push(end - start);
    }
    let out_len = numel(&out_shape);
    let mut picks = Vec::with_capacity(out_len);
    for flat_idx in 0..out_len {
        let out_index = linear_to_index(flat_idx, &out_shape);
        let in_index: Vec<usize> = out_index
            .iter()
            .zip(bounds.iter())
            .map(|(idx, (start, _))| idx + start)
            .collect();
        picks.push(index_to_linear(&in_index, &input.shape));
    }
    // reuse_* contract: shrink is element-preserving (section C3).
    Ok(TensorValue::from_storage(
        out_shape,
        input.storage().reuse_gather(&picks),
    ))
}

fn stride(input: &TensorValue, strides: &[std::num::NonZeroUsize]) -> TensorValue {
    assert_eq!(strides.len(), input.shape.len());
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(strides.iter())
        .map(|(dim, step)| (*dim).div_ceil(step.get()))
        .collect();
    let out_len = numel(&out_shape);
    let mut picks = Vec::with_capacity(out_len);
    for flat_idx in 0..out_len {
        let out_index = linear_to_index(flat_idx, &out_shape);
        let in_index: Vec<usize> = out_index
            .iter()
            .zip(strides.iter())
            .map(|(idx, step)| idx * step.get())
            .collect();
        picks.push(index_to_linear(&in_index, &input.shape));
    }
    // reuse_* contract: stride is element-preserving (section C3).
    TensorValue::from_storage(out_shape, input.storage().reuse_gather(&picks))
}

fn linear_to_index(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut index = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        index[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    index
}

fn index_to_linear(index: &[usize], shape: &[usize]) -> usize {
    let mut linear = 0usize;
    for (axis, &value) in index.iter().enumerate() {
        linear *= shape[axis];
        linear += value;
    }
    linear
}

fn validate_shape_against_type(
    input_name: &str,
    value: &TensorValue,
    ty: &TensorType,
) -> Result<(), String> {
    if value.shape.len() != ty.dims.len() {
        return Err(format!(
            "input `{input_name}` rank mismatch: expected {}, got {}",
            ty.dims.len(),
            value.shape.len()
        ));
    }
    for (axis, (actual, dim)) in value.shape.iter().zip(&ty.dims).enumerate() {
        match dim {
            DimInfo::Lit(expected) => {
                if actual != expected {
                    return Err(format!(
                        "input `{input_name}` axis {axis} mismatch: expected {expected}, got {actual}"
                    ));
                }
            }
            DimInfo::Named(name, Some(expected)) => {
                if actual != expected {
                    return Err(format!(
                        "input `{input_name}` axis {axis} named dim `{name}` mismatch: expected {expected}, got {actual}"
                    ));
                }
            }
            DimInfo::Named(_, None) => {}
        }
    }
    Ok(())
}

/// chelis#523: post-`bind_symbolic_dims` movement-op bound re-check.
///
/// `verify` (invoked before binding) skips C10 shrink/stride bound checks on
/// symbolic axes because the extent is unknown. After binding makes those
/// extents concrete, re-validate every LIVE `Shrink`/`Stride` node's bounds
/// against its (now concrete) input extent so an out-of-bounds shrink/stride
/// — which a future pass rewriting a Stride/Shrink's input vs output symbols
/// independently could introduce — fails loud with a clean `Err` rather than a
/// silent out-of-bounds read in the evaluator. Scoped to shrink/stride bounds
/// on live nodes only: the full structural `verify` would reject the dead
/// symbolic-dim-source Loads that root-scoped eval keeps alive.
fn verify_bound_movement_bounds(dag: &Dag, live: Option<&[bool]>) -> Result<(), String> {
    for node in dag.nodes() {
        if live.is_none_or(|mask| mask.get(node.id.0).copied().unwrap_or(false)) {
            verify_bound_movement_node(dag, node)?;
        }
    }
    Ok(())
}

fn verify_bound_movement_node(dag: &Dag, node: &DagNode) -> Result<(), String> {
    fn known_size(dim: &DimInfo) -> Option<usize> {
        match dim {
            DimInfo::Lit(n) => Some(*n),
            DimInfo::Named(_, Some(n)) => Some(*n),
            DimInfo::Named(_, None) => None,
        }
    }
    let input_dims = match node.inputs.first().and_then(|id| dag.get(*id)) {
        Some(input) => &input.output_type.dims,
        None => return Ok(()),
    };
    match &node.op {
        RiscOp::Shrink { bounds } => {
            for (axis, (start, end)) in bounds.iter().enumerate() {
                // chelis#616: a `ToEnd` sentinel is resolved to the axis
                // extent by `bind_symbolic_dims`, and a runtime `Node` bound
                // is validated by the evaluator (it needs the input values).
                // Only compile-time `(Lit, Lit)` bounds are statically
                // checkable here.
                //
                // chelis#1797: "validated by the evaluator" now means what it
                // says. The runtime case is checked in `shrink` itself, where
                // the resolved bounds and the operand's realized shape both
                // exist, and it returns `SHRINK_DOMAIN_TRAP` - section 2.4.1's
                // overshoot error in the compiled lane's words - rather than
                // asserting. The gap this comment used to describe was that
                // the sentence was true of the check's LOCATION and false of
                // its outcome.
                let (Some(start), Some(end)) = (start.as_lit(), end.as_lit()) else {
                    continue;
                };
                if start > end {
                    return Err(format!(
                        "post-bind shrink at node {} axis {axis}: start {start} > end {end} \
                             (chelis#523)",
                        node.id.0
                    ));
                }
                if let Some(in_size) = input_dims.get(axis).and_then(known_size)
                    && end > in_size
                {
                    return Err(format!(
                        "post-bind shrink at node {} axis {axis}: end {end} > input extent \
                             {in_size} (chelis#523)",
                        node.id.0
                    ));
                }
            }
        }
        RiscOp::Stride { strides } => {
            for step in strides {
                if step.as_lit() == Some(0) {
                    return Err(STRIDE_DOMAIN_TRAP.to_string());
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// The pre-eval bindings for every runtime extent a caller supplies.
///
/// chelis#1566: this reads the scoped derivation
/// ([`crate::axis_sources::derive_dim_witnesses`]), which is the same
/// derivation the C and HIP prologues declare and guard from, so the three
/// lanes cannot disagree about which axes are one extent. The legacy
/// name-keyed grouping it replaces put two independent signatures that merely
/// SPELL a binder `seq` into one group and rejected a correct merged kernel;
/// `split_by_scope` separates them by the results they reach.
///
/// **One structural limit, stated rather than papered over.** The map
/// [`crate::dag::bind_symbolic_dims`] consumes is keyed by NAME, so two
/// scopes of one name that resolve to DIFFERENT extents cannot both be
/// represented. The honest rule is therefore: bind the name once when the
/// scopes agree, and bind nothing for it when they disagree. Nothing is lost
/// silently - an unbound multi-scope name leaves every axis carrying it to be
/// computed from actual values, and a live node that reads the name BY VALUE
/// (a `Reshape` target's `RtDim::Sym`) still refuses in `bind_symbolic_dims`.
/// The structural fix is a per-scope rename and belongs to the claim
/// transport, not here.
///
/// The second return value is exactly the names this function deliberately
/// leaves for run-time binding: names whose supplied scopes disagree, plus a
/// name whose selected origin is a local scalar input to its declaring
/// operation. The former cannot be re-derived because disagreement belongs
/// to the supplied values; the latter is explicit in `dim_extent_origins` and
/// does not exist until that operation executes. Neither case excuses an
/// omitted external binding.
fn infer_symbolic_bindings_from_inputs(
    dag: &Dag,
    inputs: &UnordMap<String, TensorValue>,
    selection: &SymbolicInputSelection,
) -> Result<(UnordMap<String, usize>, UnordSet<String>), String> {
    let mut bindings = UnordMap::new();
    let mut load_types = UnordMap::<String, TensorType>::new();

    for node in dag.nodes() {
        if let RiscOp::Load { name } = &node.op {
            load_types
                .entry(name.as_str().to_string())
                .or_insert_with(|| node.output_type.clone());
        }
    }

    for (name, ty) in load_types.to_sorted() {
        if let Some(value) = inputs.get(name) {
            validate_shape_against_type(name, value, ty)?;
        }
    }

    let read_axis = |label: &str, axis: usize, claim: &str| -> Result<usize, String> {
        let value = inputs.get(label).ok_or_else(|| {
            format!("missing required input `{label}` for symbolic dimension `{claim}`")
        })?;
        value.shape.get(axis).copied().ok_or_else(|| {
            format!("input `{label}` is missing axis {axis} for symbolic dimension `{claim}`")
        })
    };

    let mut per_scope: Vec<(String, usize)> = Vec::new();
    for choice in &selection.witnesses {
        let SymbolicWitnesses {
            name,
            canonical: (canonical_label, canonical_axis),
            remaining,
        } = choice.as_ref().map_err(Clone::clone)?;
        let value = read_axis(canonical_label, *canonical_axis, name)?;
        for (label, axis) in remaining {
            let other = read_axis(label, *axis, name)?;
            if other != value {
                return Err(format!(
                    "symbolic dimension `{name}` mismatch: canonical {canonical_label}[{canonical_axis}] = {value}, but {label}[{axis}] = {other}",
                ));
            }
        }
        per_scope.push((name.clone(), value));
    }
    let mut folded: Vec<(String, Option<usize>)> = Vec::new();
    for (name, value) in per_scope {
        match folded.iter_mut().find(|(existing, _)| *existing == name) {
            Some((_, resolved)) => {
                if *resolved != Some(value) {
                    *resolved = None;
                }
            }
            None => folded.push((name, Some(value))),
        }
    }
    let mut deliberately_unbound = UnordSet::new();
    for (name, value) in folded {
        match value {
            Some(value) => {
                bindings.insert(name, value);
            }
            None => {
                deliberately_unbound.insert(name);
            }
        }
    }

    // Class-answered names were excluded by the same selection consumed above.
    // In particular fallback cannot overrule deliberately differing scopes.
    for (name, origin) in &selection.fallback {
        match *origin {
            crate::axis_sources::ExtentOrigin::ExternalAxis { load, axis } => {
                let Some(RiscOp::Load { name: label }) = dag.get(load).map(|node| &node.op) else {
                    continue;
                };
                // Absence is not an error here: the name may belong to a dead
                // scope the caller supplied nothing for, and a live one that
                // genuinely needs it fails loudly in `bind_symbolic_dims`.
                if let Some(value) = inputs.get(label.as_str())
                    && let Some(extent) = value.shape.get(axis)
                {
                    bindings.insert(name.clone(), *extent);
                }
            }
            crate::axis_sources::ExtentOrigin::Literal(extent) => {
                if let Ok(extent) = usize::try_from(extent) {
                    bindings.insert(name.clone(), extent);
                }
            }
            crate::axis_sources::ExtentOrigin::OpComputed { .. } => {}
            crate::axis_sources::ExtentOrigin::ScalarInput { .. } => {
                // A node-valued reshape/expand target has no entry value to
                // bind. `bind_symbolic_dims` may leave its output type
                // symbolic; `op_declared_axes_by_node` binds the exact
                // observed extent immediately after the owner executes.
                deliberately_unbound.insert(name.clone());
            }
        }
    }

    Ok((bindings, deliberately_unbound))
}

#[derive(Default)]
struct SymbolicInputSelection {
    witnesses: Vec<Result<SymbolicWitnesses, String>>,
    fallback: Vec<(String, crate::axis_sources::ExtentOrigin)>,
    required_shape_inputs: UnordSet<String>,
}

struct SymbolicWitnesses {
    name: String,
    canonical: (String, usize),
    remaining: Vec<(String, usize)>,
}

/// Extract the inference's structural choice once. Errors are pending actions:
/// ingress validation and provider effects still precede inference diagnostics.
fn select_symbolic_inputs(
    dag: &Dag,
    required_symbols: &UnordSet<String>,
    live: Option<&[bool]>,
) -> SymbolicInputSelection {
    let mut selected = SymbolicInputSelection::default();
    if required_symbols.is_empty() {
        return selected;
    }
    let live_loads = dag
        .nodes()
        .iter()
        .filter(|node| live.is_none_or(|mask| mask[node.id.0]))
        .filter_map(|node| match &node.op {
            RiscOp::Load { name } => Some(name.as_str()),
            _ => None,
        })
        .collect::<UnordSet<_>>();
    let op_declared = crate::dag::op_declared_dim_names(dag);
    let mut claimed = UnordSet::new();
    for class in crate::axis_sources::derive_dim_witnesses(dag) {
        let crate::axis_sources::DimClaim::Name(name) = class.claim else {
            continue;
        };
        if !required_symbols.contains(&name) {
            continue;
        }
        let mut witnesses = class
            .members
            .iter()
            .filter_map(|member| {
                let (load, axis) = crate::axis_sources::member_load_axis(dag, member)?;
                let RiscOp::Load { name: label } = &dag.get(load)?.op else {
                    return None;
                };
                Some((label.as_str(), axis))
            })
            .collect::<Vec<_>>();
        let has_live_local_source = class.members.iter().any(|member| {
            matches!(
                member.source,
                crate::axis_sources::AxisSource::OpComputed { .. }
                    | crate::axis_sources::AxisSource::ScalarInput { .. }
            ) && live.is_none_or(|mask| mask[member.node.0])
        });
        let live_witnesses = witnesses
            .iter()
            .copied()
            .filter(|(label, _)| live_loads.contains(label))
            .collect::<Vec<_>>();
        if !live_witnesses.is_empty() {
            witnesses = live_witnesses;
        } else if has_live_local_source && op_declared.contains(&name) {
            claimed.insert(name);
            continue;
        } else {
            let mut dead_loads = witnesses
                .iter()
                .map(|(label, _)| *label)
                .collect::<Vec<_>>();
            dead_loads.sort_unstable();
            dead_loads.dedup();
            if dead_loads.len() > 1 {
                selected.witnesses.push(Err(format!(
                    "ambiguous dead-load sources {dead_loads:?} for live symbolic dimension `{name}`"
                )));
                // Later witnesses/fallback cannot authorize initialization:
                // inference will stop at this action, after earlier reads.
                return selected;
            }
        }
        let Some((canonical, remaining)) = witnesses.split_first() else {
            continue;
        };
        claimed.insert(name.clone());
        for (label, _) in &witnesses {
            if !live_loads.contains(label) {
                selected.required_shape_inputs.insert((*label).to_owned());
            }
        }
        selected.witnesses.push(Ok(SymbolicWitnesses {
            name,
            canonical: (canonical.0.to_owned(), canonical.1),
            remaining: remaining
                .iter()
                .map(|(label, axis)| ((*label).to_owned(), *axis))
                .collect(),
        }));
    }
    selected.fallback = crate::axis_sources::dim_extent_origins(dag)
        .into_iter()
        .filter(|(name, _)| required_symbols.contains(name) && !claimed.contains(name))
        .collect();
    // Observe the ACTUAL binder traversal only when an uncovered external
    // source could demand an otherwise unselected input.
    let mut prebinding = None;
    for (name, origin) in &selected.fallback {
        let crate::axis_sources::ExtentOrigin::ExternalAxis { load, .. } = origin else {
            continue;
        };
        let Some(RiscOp::Load { name: label }) = dag.get(*load).map(|node| &node.op) else {
            continue;
        };
        if !live_loads.contains(label.as_str())
            && prebinding
                .get_or_insert_with(|| crate::dag::required_prebinding_symbols(dag))
                .contains(name)
        {
            selected
                .required_shape_inputs
                .insert(label.as_str().to_owned());
        }
    }
    selected
}

fn collect_dim_expr_symbols(expr: &DimExpr, symbols: &mut UnordSet<String>) {
    match expr {
        DimExpr::Concrete(_) => {}
        DimExpr::Sym(name) => {
            symbols.insert(name.clone());
        }
        DimExpr::Mul(lhs, rhs) | DimExpr::Div(lhs, rhs) => {
            collect_dim_expr_symbols(lhs, symbols);
            collect_dim_expr_symbols(rhs, symbols);
        }
    }
}

/// Symbolic dimensions that can affect the selected roots. Root-scoped eval
/// deliberately ignores generic declarations from unrelated dependency
/// modules (chelis#991), while `live_mask_for_roots` has already retained any
/// shape dependencies a live node genuinely needs (chelis#351/#616).
fn required_symbolic_dims(dag: &Dag, live: Option<&[bool]>) -> UnordSet<String> {
    let mut symbols = UnordSet::new();
    for node in dag.nodes() {
        if live.is_some_and(|mask| !mask[node.id.0]) {
            continue;
        }
        for dim in &node.output_type.dims {
            if let DimInfo::Named(name, None) = dim
                && !name.is_empty()
                && name != "*"
            {
                symbols.insert(name.clone());
            }
        }
        match &node.op {
            RiscOp::Reshape { new_shape } => {
                for dim in new_shape {
                    if let RtDim::Sym(name) = dim {
                        symbols.insert(name.clone());
                    }
                }
            }
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
                ..
            } => {
                for dim in batch_dims {
                    collect_dim_expr_symbols(dim, &mut symbols);
                }
                collect_dim_expr_symbols(m, &mut symbols);
                collect_dim_expr_symbols(n, &mut symbols);
                collect_dim_expr_symbols(k, &mut symbols);
            }
            _ => {}
        }
    }
    symbols
}

fn resolve_load_inputs<F>(
    dag: &Dag,
    live: Option<&[bool]>,
    strict_loads: bool,
    symbolic_dim_load_inputs: &UnordSet<&str>,
    required_shape_inputs: &UnordSet<String>,
    mut load_input: F,
) -> Result<UnordMap<String, TensorValue>, String>
where
    F: FnMut(&str, TensorInputDemand) -> Result<Option<TensorValue>, String>,
{
    let mut inputs = UnordMap::new();
    for node in dag.nodes() {
        let RiscOp::Load { name } = &node.op else {
            continue;
        };
        let is_live = live.is_none_or(|mask| mask[node.id.0]);
        // A dead Load is still resolved when its type may declare a
        // symbolic dim a live node needs (chelis#351) — but its absence
        // is never a strict-load error; only inference may complain
        // about it, with the dim-targeted message.
        if !is_live && !symbolic_dim_load_inputs.contains(name.as_str()) {
            continue;
        }
        if inputs.contains_key(name.as_str()) {
            continue;
        }
        let demand = if is_live {
            TensorInputDemand::Selected
        } else if required_shape_inputs.contains(name.as_str()) {
            TensorInputDemand::RequiredShape
        } else {
            TensorInputDemand::AvailableShape
        };
        match load_input(name.as_str(), demand)? {
            Some(value) => {
                inputs.insert(name.as_str().to_string(), value);
            }
            None if strict_loads && is_live => {
                return Err(format!("missing required input `{name}`"));
            }
            None => {}
        }
    }
    Ok(inputs)
}

fn live_mask_for_roots(dag: &Dag, roots: &[NodeId]) -> Vec<bool> {
    let mut live = vec![false; dag.len()];
    let mut stack: Vec<NodeId> = roots.to_vec();
    while let Some(id) = stack.pop() {
        if live[id.0] {
            continue;
        }
        live[id.0] = true;
        if let Some(node) = dag.get(id) {
            stack.extend(node.inputs.iter().copied());
            // chelis#616: a runtime-dim declarer kept via `shape_deps` must
            // actually EVALUATE so the mid-evaluation binding sees its
            // extent (the consumer reads the dim, not the value).
            stack.extend(node.shape_deps.iter().copied());
            stack.extend(node.result_claim_deps.iter().copied());
        }
    }
    live
}

struct PreparedTensorInputs {
    inputs: UnordMap<String, TensorValue>,
    required_symbols: UnordSet<String>,
    needs_symbolic_binding: bool,
    symbolic_selection: SymbolicInputSelection,
}

fn prepare_tensor_inputs<F>(
    dag: &Dag,
    live: Option<&[bool]>,
    strict_loads: bool,
    mut load_input: F,
) -> Result<PreparedTensorInputs, String>
where
    F: FnMut(&str, TensorInputDemand) -> Result<Option<TensorValue>, String>,
{
    // chelis#1277 C4.1: eval is a production path, so the source check runs
    // here too, before `bind_symbolic_dims` resolves any extent.
    if let Err(unsupported) =
        crate::axis_sources::check_axis_sources(dag, chelis_types::unsupported::Stage::Runtime)
    {
        return Err(unsupported.to_string());
    }
    let required_symbols = required_symbolic_dims(dag, live);
    let needs_symbolic_binding = !required_symbols.is_empty()
        // chelis#368: a `Shrink` carrying the `SHRINK_TO_END` full-axis
        // sentinel (the Pad adjoint of the differentiable `concat` over an
        // unpadded axis) must route through `bind_symbolic_dims` so the
        // sentinel is resolved from the node's now-bound output type. The
        // sentinel lives in the OP BOUNDS, not in a type, so the dim-scan
        // clauses above miss it whenever the axis monomorphized to a concrete
        // extent at eval time (a symbolic non-concat axis bound to a literal
        // by the concrete call argument). Without this trigger the orphaned
        // `usize::MAX` bound reaches the `shrink` evaluator and overflows
        // `numel` (`attempt to multiply with overflow`) instead of computing
        // the gradient. `bind_symbolic_dims` with empty bindings is an
        // identity for concrete dims and resolves the sentinel from the bound
        // output type.
        || dag.nodes().iter().any(|node| {
            live.is_none_or(|mask| mask[node.id.0])
                && matches!(&node.op, RiscOp::Shrink { bounds }
                if bounds.iter().any(|(_, end)| matches!(end, RtDim::ToEnd)))
        });
    // chelis#351: symbolic-dim inference reads shapes from the Loads
    // the derivation nominates as each dim's declaring
    // inputs — and such a Load can be DEAD under the roots' live mask
    // while the dim itself is live (e.g. `vmap(grad(f))` where the
    // gradient is constant in `x`: the backward DAG never consumes the
    // `x` Load, but its `Expand { size: Sym(..) }` still needs the dim
    // bound from `x`'s shape). Resolve named-dim-typed Loads even when
    // masked off (a superset of the nominated occurrence labels, which
    // always point at a Load carrying the symbol in its dims),
    // tolerating absence: if a needed one is genuinely unavailable,
    // `infer_symbolic_bindings_from_inputs` reports the targeted
    // "missing required input ... for symbolic dimension" error
    // instead of the strict-load one.
    let symbolic_dim_load_inputs: UnordSet<&str> = if needs_symbolic_binding && live.is_some() {
        dag.nodes()
            .iter()
            .filter_map(|node| {
                match &node.op {
                RiscOp::Load { name }
                    if node.output_type.dims.iter().any(|dim| {
                        matches!(dim, DimInfo::Named(name, _) if required_symbols.contains(name))
                    }) =>
                {
                    Some(name.as_str())
                }
                _ => None,
            }
            })
            .collect()
    } else {
        UnordSet::new()
    };
    let symbolic_selection = select_symbolic_inputs(dag, &required_symbols, live);
    let resolved_inputs = resolve_load_inputs(
        dag,
        live,
        strict_loads,
        &symbolic_dim_load_inputs,
        &symbolic_selection.required_shape_inputs,
        &mut load_input,
    )?;
    Ok(PreparedTensorInputs {
        inputs: resolved_inputs,
        required_symbols,
        needs_symbolic_binding,
        symbolic_selection,
    })
}

fn eval_tensor_internal<F>(
    dag: &Dag,
    live: Option<&[bool]>,
    strict_loads: bool,
    random_counter: u64,
    execution: Option<&mut crate::evaluation::ExecutionFrame<'_>>,
    load_input: F,
) -> Result<(UnordMap<NodeId, TensorValue>, u64), String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    eval_tensor_internal_with_result_claims(
        dag,
        live,
        strict_loads,
        random_counter,
        execution,
        &[],
        load_input,
    )
}

fn eval_tensor_internal_with_result_claims<F>(
    dag: &Dag,
    live: Option<&[bool]>,
    strict_loads: bool,
    random_counter: u64,
    mut execution: Option<&mut crate::evaluation::ExecutionFrame<'_>>,
    result_claims: &[crate::TensorType],
    mut load_input: F,
) -> Result<(UnordMap<NodeId, TensorValue>, u64), String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    let PreparedTensorInputs {
        inputs: resolved_inputs,
        required_symbols,
        needs_symbolic_binding,
        symbolic_selection,
    } = prepare_tensor_inputs(dag, live, strict_loads, |name, _| Ok(load_input(name)))?;
    let mut path_random_counter = random_counter;
    // Guard claims come from the unbound DAG; observed extents come from
    // actual caller inputs. Both host lanes consume the same individual
    // schedule before symbolic inference or dependent operations. Missing
    // inputs belonging only to an unrelated root remain optional (#991).
    let entry_guards = crate::axis_sources::entry_extent_guards(dag);
    // chelis#1374: a named witness claim whose pair the entry schedule above
    // already compares is checked there, once (spec/04 §4.7). The claim stays
    // in the graph because it is also what retains a declared-but-unread
    // parameter's interface witness.
    let entry_covered = crate::axis_sources::entry_covered_witness_claims(dag);
    // Rank and literal ABI checks remain the complement of the claim schedule.
    for node in dag.nodes() {
        let RiscOp::Load { name } = &node.op else {
            continue;
        };
        let Some(value) = resolved_inputs.get(name.as_str()) else {
            continue;
        };
        let dims = &node.output_type.dims;
        let fully_ranked = !dims.is_empty()
            && dims
                .iter()
                .all(|dim| matches!(dim, DimInfo::Lit(_) | DimInfo::Named(_, _)));
        if fully_ranked && value.shape.len() != dims.len() {
            return Err(format!(
                "input `{name}` expected rank {}, got {}",
                dims.len(),
                value.shape.len()
            ));
        }
        for (axis, dim) in dims.iter().enumerate() {
            let DimInfo::Lit(declared) = dim else {
                continue;
            };
            if entry_guards.iter().any(|guard| {
                matches!(guard,
                crate::axis_sources::EntryExtentGuard::Literal { required, observed }
                    if *required == *declared && *observed == (node.id, axis))
            }) {
                continue;
            }
            let Some(observed) = value.shape.get(axis).copied() else {
                continue;
            };
            if observed != *declared {
                return Err(format!(
                    "extent `{declared}`: claimed = {declared}, {name} axis {axis} = {observed}\n\
                     numeric trap: domain in load at i64"
                ));
            }
        }
    }

    for guard in entry_guards {
        use crate::axis_sources::EntryExtentGuard;
        let read = |(load, axis): (NodeId, usize)| {
            let RiscOp::Load { name } = &dag.get(load)?.op else {
                return None;
            };
            let extent = *resolved_inputs.get(name.as_str())?.shape.get(axis)?;
            Some((name.as_str(), axis, extent))
        };
        let context = match guard {
            EntryExtentGuard::Named {
                claim,
                canonical,
                observed,
            } => {
                let (Some((left_label, left_axis, left)), Some((right_label, right_axis, right))) =
                    (read(canonical), read(observed))
                else {
                    continue;
                };
                if left == right {
                    continue;
                }
                format!(
                    "extent `{claim}`: {left_label} axis {left_axis} = {left}, {right_label} axis {right_axis} = {right}"
                )
            }
            EntryExtentGuard::Literal { required, observed } => {
                let Some((label, axis, actual)) = read(observed) else {
                    continue;
                };
                if actual == required {
                    continue;
                }
                format!("extent `{required}`: claimed = {required}, {label} axis {axis} = {actual}")
            }
        };
        return Err(format!("{context}\nnumeric trap: domain in load at i64"));
    }

    let mut prebound_dims: UnordMap<String, usize> = UnordMap::new();
    let bound_dag = if needs_symbolic_binding {
        let (mut bindings, deliberately_unbound) =
            infer_symbolic_bindings_from_inputs(dag, &resolved_inputs, &symbolic_selection)?;
        // `bind_symbolic_dims` rebuilds the complete DAG to preserve node ids.
        // Dead named dimensions therefore need a harmless placeholder even
        // though their nodes cannot execute under this root mask. Live names
        // are never defaulted: they were inferred above or failed loudly.
        if live.is_some() {
            for name in crate::axis_sources::rendered_dim_names(dag) {
                if !required_symbols.contains(&name) {
                    bindings.entry(name).or_insert(1);
                }
            }
        }
        prebound_dims = bindings.clone();
        let bound = bind_symbolic_dims(dag, &bindings, &deliberately_unbound)?;
        // chelis#523: `verify` (grad.rs, before eval) runs BEFORE binding, so
        // its C10 shrink/stride bound checks are SKIPPED on symbolic axes
        // (extent unknown). Once `bind_symbolic_dims` makes those extents
        // concrete, re-check the movement-op bounds so an out-of-bounds
        // shrink/stride that only becomes visible post-bind fails loud with a
        // clean `Err` here instead of reaching the evaluator as a silent OOB
        // read / wrong gradient. This is the defensive-failsafe half of #523
        // (the `eval::shrink` clamp is the other half); it is not an active
        // wrong-result today, but closes the gap should a future pass break the
        // symbol-identity invariant. The check is scoped to shrink/stride
        // bounds on LIVE nodes only — the full `verify` would flag the dead
        // symbolic-dim-source Loads that root-scoped eval deliberately keeps
        // alive ("node N is dangling"), over-rejecting legitimate programs.
        if execution.is_none() {
            verify_bound_movement_bounds(&bound, live)?;
        }
        bound
    } else {
        dag.clone()
    };

    let mut values: UnordMap<NodeId, TensorValue> = UnordMap::new();

    // chelis#616: op-declared runtime dims (node-valued movement / reshape
    // output extents) have no pre-eval binding; each binds to its actual
    // extent when its declaring node evaluates, seeded with the Load-bound
    // dims so a declared-vs-computed disagreement errs loudly (the eval
    // mirror of the C backend's runtime equality-abort guard).
    let op_declared_axes = crate::dag::op_declared_axes_by_node(&bound_dag);
    // chelis#1277 C1.3: the eval lane's LOCAL guards, from the same
    // `local_dim_guard_sites` the C emitter reads (C2.5). Derived from `dag`
    // and not from `bound_dag` for the same reason the entry guards are:
    // binding rewrites a resolved `Named(n, None)` to `Named(n, Some(4))`, and
    // the derivation reads a member's own dim to decide whether the checker
    // already proved its extent, so on the bound graph every local member
    // looks statically proved and every site disappears. Node ids survive
    // `bind_symbolic_dims`, which rebuilds the graph to preserve them, so a
    // site derived here addresses the node the loop below evaluates.
    let mut local_guard_sites: UnordMap<
        NodeId,
        Vec<(usize, crate::axis_sources::LocalGuardClaim)>,
    > = UnordMap::new();
    for ((node, axis), claim) in crate::axis_sources::local_dim_guard_sites(dag) {
        local_guard_sites
            .entry(NodeId(node))
            .or_default()
            .push((axis, claim));
    }
    if !result_claims.is_empty() && dag.roots().len() == 1 {
        let root = dag.roots()[0];
        let rank = dag.get(root).expect("result root").output_type.dims.len();
        let result_sites = crate::axis_sources::result_extent_sites(dag, root);
        for claim in result_claims
            .iter()
            .filter(|claim| claim.dims.len() == rank)
        {
            for site in &result_sites {
                let RtAxis::Lit(output_axis) = site.output_axis();
                let RtAxis::Lit(producer_axis) = site.producer_axis();
                let DimInfo::Lit(required) = &claim.dims[output_axis as usize] else {
                    continue;
                };
                local_guard_sites.entry(site.producer()).or_default().push((
                    producer_axis as usize,
                    crate::axis_sources::LocalGuardClaim {
                        claim: required.to_string(),
                        canonical: crate::axis_sources::CanonicalExtent::Resolved(*required),
                        op: site.operation(),
                        observed: site.observation().clone(),
                    },
                ));
            }
        }
    }
    let mut runtime_dims = prebound_dims;

    // chelis#914: cooperative cancellation. `eval_compiled` runs two lanes —
    // this tensor-DAG lane and the host runtime's `EvalContext::eval_expr` —
    // so guarding only the host lane would leave a tensor-heavy program
    // uninterruptible. Captured once here; the per-node cost is one relaxed
    // load behind an `Option` test.
    let cancel = chelis_types::current_cancel_token();

    let order = match execution.as_ref() {
        Some(frame) => frame.steps().to_vec(),
        None => bound_dag
            .nodes()
            .iter()
            .map(|node| crate::execution_spine::Step::Node(node.id))
            .collect(),
    };
    for step in order {
        let id = match step {
            crate::execution_spine::Step::Node(id) => id,
            crate::execution_spine::Step::Control { control, .. } => {
                execution
                    .as_deref_mut()
                    .expect("only source plans have controls")
                    .control(control)?;
                continue;
            }
        };
        let node = bound_dag
            .get(id)
            .ok_or("evaluation schedule references a missing node")?;
        if let Some(cancel) = &cancel
            && cancel.is_cancelled()
        {
            return Err(chelis_types::EVAL_CANCELLED_MSG.to_string());
        }
        if let Some(mask) = live
            && !mask[node.id.0]
        {
            continue;
        }
        if execution.is_some() {
            verify_bound_movement_node(&bound_dag, node)?;
        }

        // `spec/05-risc-primitives.md` section 2.4.1 makes every stride step
        // one operation-level precondition. Resolve the COMPLETE vector once
        // before any per-axis `StrideSpan` claim can run: otherwise an early
        // axis's claim mismatch can mask a zero or negative later step. The C
        // movement plan validates this same vector before emitting any local
        // extent site. Reuse the resolved vector for both the sites and the
        // operation so Eval has one signed-validation boundary as well.
        let resolved_stride_steps = if let RiscOp::Stride { strides } = &node.op {
            let input = values.get(&node.inputs[0]).ok_or_else(|| {
                format!(
                    "stride at node {}: missing value for tensor operand",
                    node.id.0
                )
            })?;
            Some(resolve_eval_strides(strides, node, &values, &input.shape)?)
        } else {
            None
        };

        // `spec/04-type-system.md` section 4.7: a guard comparing a locally
        // computed value "takes the source position of the operation that
        // introduces the guarded extent", so it runs BEFORE that operation,
        // where the C lane emits it. Placing it after the value existed would
        // let the operation's own failure - a `reshape` numel mismatch that
        // the disagreeing extent caused - be reported instead of the claim
        // that is actually wrong, which is the C lane's behaviour before the
        // guard site existed.
        // One consumer for every local site the derivation yields, whatever
        // kind of claim it is. Which extent to read, and therefore when it can
        // be read, is the site's own `LocalGuardObservation`; this loop takes
        // the ones readable BEFORE the node runs and the loop at the foot of
        // the body takes the ones readable only after.
        //
        // `spec/04-type-system.md` section 4.7: a guard comparing a locally
        // computed value "takes the source position of the operation that
        // introduces the guarded extent", so it runs BEFORE that operation,
        // where the C lane emits it. Placing it after the value existed would
        // let the operation's own failure - a `reshape` numel mismatch that the
        // disagreeing extent caused - be reported instead of the claim that is
        // actually wrong, which is the C lane's behaviour before the guard site
        // existed.
        if let Some(sites) = local_guard_sites.get(&node.id) {
            for (axis, claim) in sites {
                // Exhaustive on purpose: a future observation kind has to say
                // here whether it is readable before the node runs, rather
                // than falling through a wildcard and disappearing from this
                // lane while the C lane keeps emitting it.
                let observed = match &claim.observed {
                    crate::axis_sources::LocalGuardObservation::Carrier(carrier) => {
                        resolve_eval_bound(carrier, node, &values, 0)?
                    }
                    crate::axis_sources::LocalGuardObservation::ComputedExtent(computed) => {
                        match computed_axis_extent_value(
                            computed,
                            node,
                            &values,
                            resolved_stride_steps.as_deref(),
                        )? {
                            Some(extent) => extent,
                            // A span that selects nothing, or that runs past
                            // the operand's own extent, computes no extent to
                            // compare, so the guard yields rather than
                            // comparing a fabricated number.
                            //
                            // chelis#1797 added the second of those. An
                            // overshooting span has an arithmetic width, and
                            // comparing a claim against it reported a claim
                            // mismatch for a program whose claim was not the
                            // defect: `-> tensor[6, f32]` over a span of 6 that
                            // reads past the end AGREED with the claim and the
                            // guard passed. `shrink` now returns section
                            // 2.4.1's overshoot error instead, in the compiled
                            // lane's words, so declining here is what lets the
                            // operation report it.
                            //
                            // An earlier version of this comment justified the
                            // decline for the EMPTY case by saying the C
                            // runtime's movement plan rejects such a span
                            // before the site is reached. That was checkable
                            // and false: `ShapeMetadata::shrunk` does not
                            // reject `start == end` (it rejects a negative
                            // start, `end < start`, and an overshoot), so an
                            // empty span builds a plan of extent 0 and C's
                            // guard runs and reports the claim.
                            //
                            // C is the conforming lane there.
                            // `spec/05-risc-primitives.md` section 2.4.1's
                            // closed list of runtime-bound errors does not
                            // include an empty span, and section 4.7.2 makes
                            // only a NEGATIVE size an error, so an extent-0
                            // result under a declared `tensor[2, f32]` is a
                            // claim mismatch. This lane instead rejects the
                            // span itself under an operation-level admission
                            // rule the numbered spec does not require; the
                            // divergence is pre-existing, is tracked by
                            // chelis#1795, and is pinned rather than repaired
                            // here.
                            None => continue,
                        }
                    }
                    crate::axis_sources::LocalGuardObservation::SameShapeAgreement(agreement) => {
                        same_shape_agreement_extent(agreement, *axis, &values)?
                    }
                    crate::axis_sources::LocalGuardObservation::MalformedSameShapeAgreement(
                        reason,
                    ) => return Err(reason.clone()),
                    // Read only after the node has produced its value; taken
                    // by the loop at the foot of the body.
                    crate::axis_sources::LocalGuardObservation::RealizedExtent => continue,
                };
                local_guard_verdict(*axis, claim, observed, &mut runtime_dims, &values)?;
            }
        }

        let out_prim = node.output_type.precision;
        let value = match &node.op {
            RiscOp::Const { value } => {
                // chelis#616: a Const whose symbolic dims resolve neither
                // statically nor through the runtime bindings may carry a
                // shape-dep naming its shape source (a `lower_if` branch
                // placeholder / mask constant shaped like a sibling); size it
                // from the dep's actual value.
                let shape = concrete_shape_with(&node.output_type, &runtime_dims)
                    .ok()
                    .or_else(|| {
                        node.shape_deps
                            .iter()
                            .find_map(|dep| values.get(dep).map(|v| v.shape.clone()))
                    })
                    .unwrap_or_default();
                let n = numel(&shape);
                // The SEALED payload materializes exactly (chelis#856):
                // integer/bool payloads splat through the exact i64
                // lane (no f64 laundering above 2^53), float payloads
                // through their exact f64 images. A payload whose value
                // does not survive the node's dtype traps loudly.
                match value.as_i64_exact() {
                    Some(i) => finalize_wide_int("const", out_prim, shape, vec![i; n])?,
                    None => finalize_wide("const", out_prim, shape, vec![value.as_f64_lossy(); n])?,
                }
            }
            RiscOp::ConstTensor { data } => {
                let shape =
                    concrete_shape_with(&node.output_type, &runtime_dims).unwrap_or_default();
                // Sealed per-dtype storage: exact integer lane for
                // integer/bool payloads, exact f64 images otherwise
                // (chelis#856).
                match data.to_i64_exact_vec() {
                    Some(ints) => finalize_wide_int("const", out_prim, shape, ints)?,
                    None => finalize_wide("const", out_prim, shape, data.to_f64_lossy_vec())?,
                }
            }
            RiscOp::Shape { axis } => {
                // Runtime extent of the input tensor along `axis`, as a
                // rank-0 integer scalar. The DAG is already bound
                // (symbolic dims resolved to concrete extents) before
                // eval, so the input value's `.shape` is concrete here.
                let input = &values[&node.inputs[0]];
                let extent = *input.shape.get(*axis).ok_or_else(|| {
                    format!(
                        "shape read axis {axis} out of bounds for rank {} input",
                        input.shape.len()
                    )
                })?;
                finalize_wide_int("shape", out_prim, vec![], vec![extent as i64])?
            }
            RiscOp::ExtentWitness {
                site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
                requirements,
                ..
            } => finalize_wide_int(
                "shape",
                out_prim,
                vec![],
                vec![
                    requirements[0]
                        .as_i64_exact()
                        .expect("verified literal result requirement"),
                ],
            )?,
            RiscOp::ExtentWitness {
                site: crate::dag::ExtentWitnessSite::LocalAscriptionClaim { .. },
                requirements,
                ..
            } if !requirements.is_empty() => finalize_wide_int(
                "shape",
                out_prim,
                vec![],
                vec![
                    requirements[0]
                        .as_i64_exact()
                        .expect("verified local ascription requirement"),
                ],
            )?,
            RiscOp::ExtentWitness {
                site,
                parameter,
                axis,
                requirements,
                claims,
            } => {
                let operation = match site {
                    crate::dag::ExtentWitnessSite::Caller => "load",
                    crate::dag::ExtentWitnessSite::LocalExpand => "expand",
                    crate::dag::ExtentWitnessSite::ResultClaim { .. } => "shape",
                    crate::dag::ExtentWitnessSite::LocalAscriptionClaim { .. } => "shape",
                    crate::dag::ExtentWitnessSite::LiteralResultClaim => {
                        unreachable!("literal role handled above")
                    }
                };
                let parameter = match site {
                    crate::dag::ExtentWitnessSite::Caller => parameter.clone(),
                    crate::dag::ExtentWitnessSite::ResultClaim { .. } => parameter.clone(),
                    crate::dag::ExtentWitnessSite::LocalAscriptionClaim { .. } => parameter.clone(),
                    crate::dag::ExtentWitnessSite::LiteralResultClaim => {
                        unreachable!("literal role handled above")
                    }
                    crate::dag::ExtentWitnessSite::LocalExpand => {
                        format!("node {}", node.inputs[0].0)
                    }
                };
                let crate::dag::RtAxis::Lit(axis) = axis;
                let input = &values[&node.inputs[0]];
                let observed = *input
                    .shape
                    .get(*axis as usize)
                    .ok_or_else(|| format!("extent witness axis {axis} out of bounds"))?;
                for required in requirements.iter().chain(
                    crate::axis_sources::literal_result_witness_requirements(dag, node.id).iter(),
                ) {
                    let required = required
                        .as_i64_exact()
                        .ok_or_else(|| "extent witness requires i64".to_string())?;
                    if i64::try_from(observed).ok() != Some(required) {
                        return Err(format!(
                            "extent `{required}`: claimed = {required}, {parameter} axis {axis} = {observed}\nnumeric trap: domain in {operation} at i64"
                        ));
                    }
                }
                // chelis#1374/#1376: §4.7.2's named half. Each requirement
                // edge is another witness of the same activation, so both
                // records read a fact the graph states. The declaring side
                // leads, as `entry_extent_guards` renders the `Load`-witnessed
                // form of the same contract.
                for (index, (claim, edge)) in
                    claims.iter().zip(node.inputs.iter().skip(1)).enumerate()
                {
                    if entry_covered.contains(&(node.id, index)) {
                        continue;
                    }
                    let (required_parameter, required_axis) =
                        match &dag.get(*edge).ok_or("missing extent claim edge")?.op {
                            RiscOp::ExtentWitness {
                                parameter,
                                axis: crate::dag::RtAxis::Lit(axis),
                                ..
                            } => (parameter.clone(), *axis),
                            _ => return Err("extent claim requires a witness edge".into()),
                        };
                    let required = values[edge]
                        .storage()
                        .scalar_at(0)
                        .as_i64_exact()
                        .ok_or("extent claim requires i64")?;
                    if i64::try_from(observed).ok() == Some(required) {
                        continue;
                    }
                    let here = format!("{parameter} axis {axis} = {observed}");
                    let there = format!("{required_parameter} axis {required_axis} = {required}");
                    let (first, second) = if claim.requirement_declares {
                        (there, here)
                    } else {
                        (here, there)
                    };
                    return Err(format!(
                        "extent `{}`: {first}, {second}\nnumeric trap: domain in {operation} at i64",
                        claim.claim
                    ));
                }
                finalize_wide_int(
                    "shape",
                    out_prim,
                    vec![],
                    vec![i64::try_from(observed).map_err(|_| "extent exceeds i64")?],
                )?
            }
            RiscOp::CheckedReshapeExtent {
                claims,
                axis: crate::dag::RtAxis::Lit(axis),
            } => {
                let actual = values[&node.inputs[0]]
                    .storage()
                    .scalar_at(0)
                    .as_i64_exact()
                    .ok_or("checked reshape actual must be i64")?;
                for (claim, input) in claims.iter().zip(&node.inputs[1..]) {
                    let required = values[input]
                        .storage()
                        .scalar_at(0)
                        .as_i64_exact()
                        .ok_or("checked reshape requirement must be i64")?;
                    if actual != required {
                        return Err(format!(
                            "extent `{claim}`: claimed = {required}, reshape axis {axis} = {actual}\nnumeric trap: domain in reshape at i64"
                        ));
                    }
                }
                values[&node.inputs[0]].clone()
            }
            RiscOp::CheckedUnitAxis { .. } => values[&node.inputs[0]].clone(),
            RiscOp::Load { name } => match resolved_inputs.get(name.as_str()) {
                Some(value) => ingress_to_declared(name.as_str(), out_prim, value)?,
                None if strict_loads => return Err(format!("missing required input `{name}`")),
                None => default_value(&node.output_type),
            },
            RiscOp::Store { .. } | RiscOp::Copy | RiscOp::Drop | RiscOp::Realize => {
                values[&node.inputs[0]].clone()
            }
            RiscOp::Add => binary_elementwise(
                ElementwiseBinOp::Add,
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
            )?,
            RiscOp::Sub => binary_elementwise(
                ElementwiseBinOp::Sub,
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
            )?,
            RiscOp::Mul => binary_elementwise(
                ElementwiseBinOp::Mul,
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
            )?,
            RiscOp::Div => binary_elementwise(
                ElementwiseBinOp::Div,
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
            )?,
            // chelis#178: floor division rounds the quotient toward -inf.
            // For integer-valued operands `(a / b).floor()` yields the
            // integer floored quotient, and for float operands it is
            // `floor(a / b)` directly.
            //
            // chelis#550: integer operands trap on a zero divisor with the
            // shared diagnostic, mirroring the host evaluator (`host_ops`,
            // i64 + trap) and the C backend (`chelis_int_div_guard`) so this
            // reference lane fails closed instead of emitting `floor(x/0)`.
            // Float operands keep IEEE semantics (`floor(+inf) == +inf`,
            // never traps), per spec/05-risc-primitives.md §2.1. The integer
            // case is gated on the output precision, which equals the operand
            // precision for `floor_div` (the §2.1 precision rule).
            RiscOp::FloorDiv => {
                let lhs = &values[&node.inputs[0]];
                let rhs = &values[&node.inputs[1]];
                binary_elementwise(ElementwiseBinOp::FloorDiv, lhs, rhs)?
            }
            // chelis#178: truncating (round-toward-zero) integer division.
            // `(a / b).trunc()` matches C/Rust integer `/` for the
            // integer-valued operands this op is restricted to.
            //
            // chelis#550: `trunc_div` is integer-only (the checker rejects
            // float operands), so a zero divisor always traps with the shared
            // diagnostic — matching `host_ops::eval_trunc_div` and the C
            // backend guard.
            RiscOp::Mod => {
                let lhs = &values[&node.inputs[0]];
                let rhs = &values[&node.inputs[1]];
                binary_elementwise(ElementwiseBinOp::Mod, lhs, rhs)?
            }
            RiscOp::TruncDiv => {
                let lhs = &values[&node.inputs[0]];
                let rhs = &values[&node.inputs[1]];
                binary_elementwise(ElementwiseBinOp::TruncDiv, lhs, rhs)?
            }
            RiscOp::Neg => unary_elementwise(ElementwiseUnOp::Neg, &values[&node.inputs[0]])?,
            RiscOp::Recip => unary_elementwise(ElementwiseUnOp::Recip, &values[&node.inputs[0]])?,
            RiscOp::Exp => unary_elementwise(ElementwiseUnOp::Exp, &values[&node.inputs[0]])?,
            RiscOp::Log => unary_elementwise(ElementwiseUnOp::Log, &values[&node.inputs[0]])?,
            RiscOp::Sin => unary_elementwise(ElementwiseUnOp::Sin, &values[&node.inputs[0]])?,
            RiscOp::Sqrt => unary_elementwise(ElementwiseUnOp::Sqrt, &values[&node.inputs[0]])?,
            RiscOp::Cos => unary_elementwise(ElementwiseUnOp::Cos, &values[&node.inputs[0]])?,
            RiscOp::Tan => unary_elementwise(ElementwiseUnOp::Tan, &values[&node.inputs[0]])?,
            RiscOp::Atan => unary_elementwise(ElementwiseUnOp::Atan, &values[&node.inputs[0]])?,
            RiscOp::Abs => unary_elementwise(ElementwiseUnOp::Abs, &values[&node.inputs[0]])?,
            RiscOp::Floor => unary_elementwise(ElementwiseUnOp::Floor, &values[&node.inputs[0]])?,
            RiscOp::Ceil => unary_elementwise(ElementwiseUnOp::Ceil, &values[&node.inputs[0]])?,
            // Round-half-to-even (banker's rounding), matching the C
            // backend's `rintf` under the default rounding mode. NOT
            // `f64::round`, which rounds half away from zero.
            RiscOp::Round => unary_elementwise(ElementwiseUnOp::Round, &values[&node.inputs[0]])?,
            RiscOp::UniformLike { low, high, seed } => {
                let key = if let Some(frame) = execution.as_deref_mut() {
                    let active = match node.inputs.get(1) {
                        Some(activation) => match values[activation].storage().to_raw() {
                            RawTensor::Int(values) if values.len() == 1 => values[0] != 0,
                            _ => {
                                return Err(
                                    "uniform_like path activation is not a scalar Bool".into()
                                );
                            }
                        },
                        None => true,
                    };
                    frame.uniform_key(node.id, *seed, active)?
                } else if let Some(activation) = node.inputs.get(1) {
                    let active = match values[activation].storage().to_raw() {
                        RawTensor::Int(values) if values.len() == 1 => values[0] != 0,
                        _ => {
                            return Err(format!(
                                "uniform_like path activation at node {} is not a scalar Bool",
                                node.id.0
                            ));
                        }
                    };
                    if active {
                        Some((*seed, path_random_counter))
                    } else {
                        None
                    }
                } else {
                    None
                };
                // One legacy value-kernel seed fold for both planned keys
                // and activation-gated legacy execution. No canonical
                // UniformLike numeric claim is made by plan admission.
                let effective_seed = key.map_or(*seed, |(raw_seed, path_random_counter)| {
                    raw_seed ^ path_random_counter.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                });
                if execution.is_none() && key.is_some() {
                    path_random_counter = path_random_counter.saturating_add(1);
                }
                uniform_like(
                    &values[&node.inputs[0]].shape.clone(),
                    *low,
                    *high,
                    effective_seed,
                    out_prim,
                )?
            }
            RiscOp::Dropout { rate, seed } => match execution.as_deref_mut() {
                Some(frame) => frame.dropout(node.id, &values[&node.inputs[0]], *rate, *seed)?,
                None => dropout(&values[&node.inputs[0]], *rate, *seed, out_prim)?,
            },
            RiscOp::MaxElem => binary_elementwise(
                ElementwiseBinOp::Max,
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
            )?,
            RiscOp::MinElem => binary_elementwise(
                ElementwiseBinOp::Min,
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
            )?,
            RiscOp::ExtremaAdjoint { kind, operand } => {
                let lhs = &values[&node.inputs[0]];
                let rhs = &values[&node.inputs[1]];
                let cotangent = &values[&node.inputs[2]];
                let kind = match kind {
                    ExtremaKind::Max => FloatExtremaOp::Max,
                    ExtremaKind::Min => FloatExtremaOp::Min,
                };
                let operand = match operand {
                    ExtremaOperand::Left => KernelExtremaOperand::Left,
                    ExtremaOperand::Right => KernelExtremaOperand::Right,
                };
                let storage = float_extrema_adjoint(
                    kind,
                    operand,
                    lhs.storage(),
                    rhs.storage(),
                    cotangent.storage(),
                )
                .map_err(|error| error.to_string())?;
                TensorValue::from_storage(lhs.shape.clone(), storage)
            }
            RiscOp::Relu => {
                let input = &values[&node.inputs[0]];
                let storage = float_relu(input.storage()).map_err(|error| error.to_string())?;
                TensorValue::from_storage(input.shape.clone(), storage)
            }
            RiscOp::ReluAdjoint => {
                let input = &values[&node.inputs[0]];
                let cotangent = &values[&node.inputs[1]];
                let storage = float_relu_adjoint(input.storage(), cotangent.storage())
                    .map_err(|error| error.to_string())?;
                TensorValue::from_storage(input.shape.clone(), storage)
            }
            RiscOp::CmpLt => compare_elementwise(
                CompareOp::Lt,
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
            )?,
            RiscOp::Sum { axis, accumulator } => reduce(
                &values[&node.inputs[0]],
                *axis,
                TensorReduceOp::Sum {
                    accumulator: *accumulator,
                    result: out_prim,
                },
            )?,
            RiscOp::Count { axes } => count_tensor(&values[&node.inputs[0]], axes)?,
            RiscOp::MaxReduce { axis } => {
                reduce(&values[&node.inputs[0]], *axis, TensorReduceOp::MaxReduce)?
            }
            RiscOp::MinReduce { axis } => {
                reduce(&values[&node.inputs[0]], *axis, TensorReduceOp::MinReduce)?
            }
            RiscOp::ProdReduce { axis } => {
                reduce(&values[&node.inputs[0]], *axis, TensorReduceOp::ProdReduce)?
            }
            RiscOp::ReduceWindow {
                reducer,
                window_shape,
                strides,
            } => reduce_window(
                &values[&node.inputs[0]],
                *reducer,
                window_shape,
                strides,
                out_prim,
            )?,
            RiscOp::ReduceWindowGrad {
                reducer,
                window_shape,
                strides,
            } => reduce_window_grad(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                *reducer,
                window_shape,
                strides,
                out_prim,
            )?,
            RiscOp::Argmax { axis } => {
                reduce_argcmp(&values[&node.inputs[0]], *axis, ArgReduceOp::Argmax)?
            }
            RiscOp::Argmin { axis } => {
                reduce_argcmp(&values[&node.inputs[0]], *axis, ArgReduceOp::Argmin)?
            }
            RiscOp::Reshape { new_shape } => {
                let shape: Vec<usize> = new_shape
                    .iter()
                    .map(|dim| match dim {
                        RtDim::Lit(n) => Ok(*n),
                        // chelis#616: a runtime target extent reads its rank-0
                        // integer scalar exactly like a movement bound.
                        RtDim::Node(_) => resolve_eval_bound(dim, node, &values, 0),
                        RtDim::InputAxis { .. } => resolve_eval_bound(dim, node, &values, 0),
                        // chelis#616: an op-declared symbol resolves from the
                        // mid-evaluation bindings.
                        RtDim::Sym(name) => runtime_dims.get(name).copied().ok_or_else(|| {
                            format!("cannot reshape to symbolic dimension `{name}`")
                        }),
                        RtDim::ToEnd => {
                            Err("reshape target dim cannot be a shrink-to-end sentinel".to_string())
                        }
                    })
                    .collect::<Result<_, _>>()?;
                // chelis#616: with runtime target extents the numel invariant
                // is only checkable here — report a clean error (mirrored by
                // the C backend's runtime numel abort), never a panic.
                let input = &values[&node.inputs[0]];
                let expected: usize = shape.iter().product();
                // The phrase is the interpreter's and the C runtime's
                // (`host_emit.rs`), so every lane reports the mismatch alike.
                if expected != input.len() {
                    return Err(format!(
                        "reshape expects {} elements but tensor has {}",
                        expected,
                        input.len()
                    ));
                }
                reshape(input, shape)
            }
            RiscOp::Permute { axes } => permute(&values[&node.inputs[0]], axes),
            RiscOp::Expand { axis, size } => {
                let input = &values[&node.inputs[0]];
                let size_value = resolve_eval_bound(size, node, &values, 0)?;
                let mut out_shape = input.shape.clone();
                if node.output_type.dims.len() == input.shape.len() + 1 {
                    if *axis > out_shape.len() {
                        return Err(format!(
                            "expand at node {}: axis {} out of bounds for rank {} tensor",
                            node.id.0,
                            axis,
                            input.shape.len()
                        ));
                    }
                    out_shape.insert(*axis, size_value);
                } else if node.output_type.dims.len() == input.shape.len() {
                    let target = out_shape.get_mut(*axis).ok_or_else(|| {
                        format!(
                            "expand at node {}: axis {} out of bounds for rank {} tensor",
                            node.id.0,
                            axis,
                            input.shape.len()
                        )
                    })?;
                    *target = size_value;
                } else {
                    return Err(format!(
                        "expand at node {}: output rank {} must equal input rank {} or {}",
                        node.id.0,
                        node.output_type.dims.len(),
                        input.shape.len(),
                        input.shape.len() + 1
                    ));
                }
                expand(input, *axis, size_value, out_shape)
            }
            RiscOp::OneHot { vocab } => one_hot(&values[&node.inputs[0]], *vocab, out_prim)?,
            RiscOp::Pad { padding, fill } => {
                let input = &values[&node.inputs[0]];
                let resolved = resolve_eval_pairs(padding, node, &values, &input.shape)?;
                pad(input, &resolved, *fill)?
            }
            RiscOp::Shrink { bounds } => {
                let input = &values[&node.inputs[0]];
                let resolved = resolve_eval_pairs(bounds, node, &values, &input.shape)?;
                // chelis#616 on the eval lane: a runtime bound that selects
                // nothing is rejected as the interpreter rejects it and as
                // the C runtime aborts it, never returned as an empty tensor.
                for (axis, (start, end)) in resolved.iter().enumerate() {
                    if start >= end {
                        return Err(format!(
                            "shrink axis {axis} bound [{start}, {end}] is empty or inverted \
                             (start >= end)"
                        ));
                    }
                }
                shrink(input, &resolved)?
            }
            RiscOp::Stride { .. } => {
                let input = &values[&node.inputs[0]];
                let resolved = resolved_stride_steps.as_deref().ok_or_else(|| {
                    format!(
                        "stride at node {}: prevalidated step vector is missing",
                        node.id.0
                    )
                })?;
                stride(input, resolved)
            }
            RiscOp::FusedElem { ops } => {
                // Collect external input TensorValues from the node's DAG inputs.
                let externals: Vec<&TensorValue> =
                    node.inputs.iter().map(|id| &values[id]).collect();

                // chelis#729 Phase 2: every step dispatches from its actual
                // finalized operand storage. That preserves per-op rounding,
                // exact integer width, and division traps even when a trailing
                // comparison changes the fused node output dtype to bool.

                // Walk the fused steps sequentially, building up intermediate results.
                let mut intermediates: Vec<TensorValue> = Vec::with_capacity(ops.len());

                for step in ops {
                    let resolve = |fi: &FusedInput| -> &TensorValue {
                        match fi {
                            FusedInput::External(idx) => externals[*idx],
                            FusedInput::PreviousStep(idx) => &intermediates[*idx],
                        }
                    };

                    let result = match step.op {
                        // Binary ops
                        FusedStepOp::Add => binary_elementwise(
                            ElementwiseBinOp::Add,
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                        )?,
                        FusedStepOp::Sub => binary_elementwise(
                            ElementwiseBinOp::Sub,
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                        )?,
                        FusedStepOp::Mul => binary_elementwise(
                            ElementwiseBinOp::Mul,
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                        )?,
                        FusedStepOp::Div => binary_elementwise(
                            ElementwiseBinOp::Div,
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                        )?,
                        // chelis#178/#550: floor/truncating division shares
                        // the standalone typed-kernel path, including exact
                        // per-dtype integer zero-divisor and overflow traps.
                        FusedStepOp::FloorDiv => {
                            let lhs = resolve(&step.input_indices[0]);
                            let rhs = resolve(&step.input_indices[1]);
                            binary_elementwise(ElementwiseBinOp::FloorDiv, lhs, rhs)?
                        }
                        FusedStepOp::TruncDiv => {
                            let lhs = resolve(&step.input_indices[0]);
                            let rhs = resolve(&step.input_indices[1]);
                            binary_elementwise(ElementwiseBinOp::TruncDiv, lhs, rhs)?
                        }
                        FusedStepOp::MaxElem => binary_elementwise(
                            ElementwiseBinOp::Max,
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                        )?,
                        FusedStepOp::MinElem => binary_elementwise(
                            ElementwiseBinOp::Min,
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                        )?,
                        FusedStepOp::CmpLt => compare_elementwise(
                            CompareOp::Lt,
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                        )?,
                        // Unary ops
                        FusedStepOp::Neg => unary_elementwise(
                            ElementwiseUnOp::Neg,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Recip => unary_elementwise(
                            ElementwiseUnOp::Recip,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Exp => unary_elementwise(
                            ElementwiseUnOp::Exp,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Log => unary_elementwise(
                            ElementwiseUnOp::Log,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Sin => unary_elementwise(
                            ElementwiseUnOp::Sin,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Sqrt => unary_elementwise(
                            ElementwiseUnOp::Sqrt,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Cos => unary_elementwise(
                            ElementwiseUnOp::Cos,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Tan => unary_elementwise(
                            ElementwiseUnOp::Tan,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Atan => unary_elementwise(
                            ElementwiseUnOp::Atan,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Abs => unary_elementwise(
                            ElementwiseUnOp::Abs,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Floor => unary_elementwise(
                            ElementwiseUnOp::Floor,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Ceil => unary_elementwise(
                            ElementwiseUnOp::Ceil,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Round => unary_elementwise(
                            ElementwiseUnOp::Round,
                            resolve(&step.input_indices[0]),
                        )?,
                    };
                    intermediates.push(result);
                }

                // Every typed step already finalizes at its own dtype. The
                // final conversion is therefore only a defensive checker-
                // invariant guard; a comparison tail already carries bool.
                let last = intermediates
                    .pop()
                    .expect("FusedElem must have at least one step");
                if last.prim() == out_prim {
                    last
                } else {
                    let storage = finalize_tensor("fused", out_prim, last.storage().to_raw())
                        .map_err(|trap| trap.to_string())?;
                    TensorValue::from_storage(last.shape.clone(), storage)
                }
            }
            RiscOp::CastTrunc { new_precision } => {
                let input = &values[&node.inputs[0]];
                cast_trunc_value(input, *new_precision)?
            }
            RiscOp::Cast { new_precision } => {
                // #380: a `cast` must apply the dtype conversion, not pass the
                // stored value through unchanged. The per-direction rules live
                // in `cast_wide_element` (chelis#729 Phase 1: float targets
                // genuinely finalize at width, so f16/bf16 casts round; the
                // out-of-range integer rules keep their pre-refactor behavior
                // until chelis#759 authors them). The source precision comes
                // from the input node's output type; the target is the cast's
                // `new_precision`.
                let input = &values[&node.inputs[0]];
                let src_prec = bound_dag
                    .get(node.inputs[0])
                    .map(|n| n.output_type.precision)
                    .unwrap_or(*new_precision);
                cast_value(input, src_prec, *new_precision)?
            }
            RiscOp::BlasMatmul { .. } => {
                let lhs = &values[&node.inputs[0]];
                let rhs = &values[&node.inputs[1]];
                if lhs.shape.len() == 2 && rhs.shape.len() == 2 {
                    matmul(lhs, rhs, out_prim)?
                } else {
                    batched_matmul(lhs, rhs, out_prim)?
                }
            }
            RiscOp::Gather { axis } => {
                gather(&values[&node.inputs[0]], &values[&node.inputs[1]], *axis)
            }
            RiscOp::ScatterAdd { axis } => scatter_add(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                &values[&node.inputs[2]],
                *axis,
                out_prim,
            )?,
            RiscOp::Scatter { axis } => scatter_replace(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                &values[&node.inputs[2]],
                *axis,
            ),
            RiscOp::ScatterElements { axis } => scatter_elements(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                &values[&node.inputs[2]],
                *axis,
            ),
        };
        // chelis#616: bind this node's op-declared runtime dims from the
        // value's actual extents. A disagreement with an existing binding
        // (Load-bound or an earlier declarer for the same symbol) is a real
        // shape error and must err loudly, mirroring the C runtime guard.
        if let Some(axes) = op_declared_axes.get(&node.id) {
            for (symbol, axis) in axes {
                let extent = *value.shape.get(*axis).ok_or_else(|| {
                    format!(
                        "runtime dim `{symbol}`: node {} produced rank {} but axis {axis} \
                         was expected",
                        node.id.0,
                        value.shape.len()
                    )
                })?;
                match runtime_dims.get(symbol) {
                    Some(previous) if *previous != extent => {
                        return Err(format!(
                            "runtime dim `{symbol}` mismatch: node {} axis {axis} computed \
                             {extent}, but an earlier declaration bound {previous}",
                            node.id.0
                        ));
                    }
                    _ => {
                        runtime_dims.insert(symbol.clone(), extent);
                    }
                }
            }
        }
        // The same consumer, for the sites whose extent only exists once this
        // node has produced it. Section 4.7 places these "after its producers
        // and before the first allocation or element access whose shape
        // depends on the guarded extent": this node IS the producer, and the
        // allocation that depends on the extent belongs to its consumer, which
        // has not run. A unit-extent claim is the case, and the site is keyed
        // on the operand rather than on the `expand` that makes the claim,
        // because that is where the C emitter renders the extent.
        if let Some(sites) = local_guard_sites.get(&node.id) {
            for (axis, claim) in sites {
                if !matches!(
                    claim.observed,
                    crate::axis_sources::LocalGuardObservation::RealizedExtent
                ) {
                    continue;
                }
                let Some(&observed) = value.shape.get(*axis) else {
                    continue;
                };
                local_guard_verdict(*axis, claim, observed, &mut runtime_dims, &values)?;
            }
        }
        values.insert(node.id, value);
    }

    Ok((values, path_random_counter))
}

/// One local extent guard, compared and reported.
///
/// Both consumers below call this and nothing else formats a local guard on
/// this lane, so the two kinds of site cannot drift into two diagnostics. The
/// text is [04-NUM-9]'s complete line with no prefix and no suffix, and
/// section 4.7's context on its own preceding line, in the C lane's wording:
/// the two lanes report one guard.
///
/// A binder this lane has not bound yet is BOUND from the first site that
/// observes it, and every later site is compared against that value. That is
/// C2.4's rule for a class whose canonical value no literal resolves: the
/// first member is canonical and the rest guard against it.
///
/// This mirrors the C lane case for case rather than describing it. Once
/// control reaches `emit_runtime_dim_site`'s GUARD branch it consults no
/// binding table and emits its comparison against the binder unconditionally,
/// so a lane that SKIPPED an unbound binder would leave C comparing against an
/// identifier and this lane comparing against nothing. `runtime_dim_sites`
/// (`chelis-backend-c/src/emit.rs`) makes the FIRST op-declared occurrence of a
/// symbol no `Load` declares a DECLARE site, emitting `int64_t m = <extent>;`
/// before its guard; the `runtime_dims` insert below is that declaration on
/// this lane, at the same site and from the same observed value.
///
/// The residual this replaces was a derivation site for a symbol nothing
/// declares, recorded rather than closed because no reviewer could construct a
/// reaching class. Admitting op-computed members makes one buildable - two axes
/// of one `shrink` under a result name no parameter declares - so the asymmetry
/// between the two consumers is removed here instead of being re-recorded.
fn local_guard_verdict(
    axis: usize,
    claim: &crate::axis_sources::LocalGuardClaim,
    observed: usize,
    runtime_dims: &mut UnordMap<String, usize>,
    values: &UnordMap<NodeId, TensorValue>,
) -> Result<(), String> {
    let claimed = match &claim.canonical {
        crate::axis_sources::CanonicalExtent::Resolved(value) => *value,
        crate::axis_sources::CanonicalExtent::Witness(witness) => values
            .get(witness)
            .filter(|value| value.shape.is_empty() && value.len() == 1)
            .and_then(|value| value.storage().scalar_at(0).as_i64_exact())
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("missing or invalid declaring extent witness {}", witness.0))?,
        crate::axis_sources::CanonicalExtent::Binder(name) => match runtime_dims.get(name) {
            Some(value) => *value,
            None => {
                runtime_dims.insert(name.clone(), observed);
                observed
            }
        },
    };
    if observed != claimed {
        return Err(format!(
            "extent `{}`: claimed = {claimed}, {} axis {axis} = {observed}\n\
             numeric trap: domain in {} at i64",
            claim.claim, claim.op, claim.op,
        ));
    }
    Ok(())
}

/// Evaluate a source-owned plan. The context records the executed prefix on
/// both success and failure; legacy Dag-only entrypoints remain unchanged.
///
/// Inputs may first be resolved with [`prepare_tensor_plan_inputs`]. Execute
/// the same plan using a lookup into that map to avoid repeating provider effects.
pub fn eval_tensor_plan_with_strict<F>(
    plan: &crate::evaluation::EvaluationPlan,
    context: &mut crate::evaluation::RandomExecutionContext,
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    let dag = plan.dag_for_inspection();
    let starting_counter = context.state().counter;
    let mut frame = plan.frame(context)?;
    // Plan validation requires the complete selected graph, including dead
    // executed nodes. Value liveness must not prune this execution slice.
    let live = vec![true; dag.len()];
    eval_tensor_internal(
        dag,
        Some(&live),
        true,
        starting_counter,
        Some(&mut frame),
        load_input,
    )
    .map(|(values, _)| values)
}

/// Execute the same source plan with invocation-local literal result claims.
/// The extra obligations neither modify the DAG nor replace its own guards.
pub fn eval_tensor_plan_with_result_claims<F>(
    plan: &crate::evaluation::EvaluationPlan,
    context: &mut crate::evaluation::RandomExecutionContext,
    result_claims: &[crate::TensorType],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    let dag = plan.dag_for_inspection();
    let starting_counter = context.state().counter;
    let mut frame = plan.frame(context)?;
    let live = vec![true; dag.len()];
    eval_tensor_internal_with_result_claims(
        dag,
        Some(&live),
        true,
        starting_counter,
        Some(&mut frame),
        result_claims,
        load_input,
    )
    .map(|(values, _)| values)
}

/// Why a selected-input preparation callback is being queried.
/// This is an input obligation, not declaration identity or a cache certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorInputDemand {
    /// An executed Load, including shape dependencies and plan-retained Loads.
    /// Strict preparation refuses absence immediately.
    Selected,
    /// An exact external shape source needed outside the execution mask.
    /// Absence keeps the existing deferred inference/binding diagnostic.
    RequiredShape,
    /// A surplus symbolic declarer: return a value only if already available.
    /// This query does not authorize evaluating an unresolved initializer.
    AvailableShape,
}

/// Resolve the inputs selected by strict root evaluation, without evaluating
/// graph nodes. Empty roots select the whole DAG, as in
/// [`eval_tensor_roots_with_strict`]. Shape dependencies and optional symbolic
/// dimension declarers follow the same selection rules as evaluation.
///
/// Providers run in DAG order. Successful names are requested once; absent
/// optional names may be requested repeatedly. Provider errors propagate
/// unchanged. Axis-source and Drop-root validation precede provider calls;
/// rank, extent, dtype and numeric validation remain evaluation's responsibility.
/// Execute the same DAG/roots with a lookup into the returned map to avoid
/// repeating provider effects. The map is not a certificate of successful execution.
pub fn prepare_tensor_roots_inputs<F>(
    dag: &Dag,
    roots: &[NodeId],
    mut load_input: F,
) -> Result<UnordMap<String, TensorValue>, String>
where
    F: FnMut(&str) -> Result<Option<TensorValue>, String>,
{
    prepare_tensor_roots_inputs_with_demand(dag, roots, |name, _| load_input(name))
}

/// Role-aware form of [`prepare_tensor_roots_inputs`], with identical selection,
/// ordering, deduplication and validation stages. RequiredShape absence remains
/// deferred; AvailableShape is an available-value query, not initializer demand.
pub fn prepare_tensor_roots_inputs_with_demand<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<UnordMap<String, TensorValue>, String>
where
    F: FnMut(&str, TensorInputDemand) -> Result<Option<TensorValue>, String>,
{
    reject_drop_roots(dag, roots)?;
    let live = (!roots.is_empty()).then(|| live_mask_for_roots(dag, roots));
    prepare_tensor_inputs(dag, live.as_deref(), true, load_input).map(|prepared| prepared.inputs)
}

/// Resolve every selected plan input, including value-dead executed loads,
/// without starting an execution frame or advancing `context`.
///
/// Plan integrity and inherited seed are validated before provider effects.
/// Provider ordering, errors and deferred value validation follow
/// [`prepare_tensor_roots_inputs`]. Execute this same plan with a map lookup;
/// the caller may first incorporate provider effects into its execution context.
pub fn prepare_tensor_plan_inputs<F>(
    plan: &crate::evaluation::EvaluationPlan,
    context: &crate::evaluation::RandomExecutionContext,
    mut load_input: F,
) -> Result<UnordMap<String, TensorValue>, String>
where
    F: FnMut(&str) -> Result<Option<TensorValue>, String>,
{
    prepare_tensor_plan_inputs_with_demand(plan, context, |name, _| load_input(name))
}

/// Role-aware form of [`prepare_tensor_plan_inputs`]. Every retained plan Load
/// is Selected, even if value-dead. Plan/seed validation precedes the provider;
/// preparation does not start a frame or advance the supplied context.
pub fn prepare_tensor_plan_inputs_with_demand<F>(
    plan: &crate::evaluation::EvaluationPlan,
    context: &crate::evaluation::RandomExecutionContext,
    load_input: F,
) -> Result<UnordMap<String, TensorValue>, String>
where
    F: FnMut(&str, TensorInputDemand) -> Result<Option<TensorValue>, String>,
{
    plan.validate_for_context(context)?;
    let dag = plan.dag_for_inspection();
    let live = vec![true; dag.len()];
    prepare_tensor_inputs(dag, Some(&live), true, load_input).map(|prepared| prepared.inputs)
}

pub(crate) fn eval_tensor_segment_with_strict<F>(
    dag: &Dag,
    frame: &mut crate::evaluation::ExecutionFrame<'_>,
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    eval_tensor_internal(dag, None, true, 0, Some(frame), load_input).map(|(values, _)| values)
}

/// The extent an op-computed axis is about to produce, read from the site
/// node's own bounds before that node runs.
///
/// `None` means the operation computes no extent here: a `shrink` span whose
/// start is not below its end selects nothing, so there is nothing to compare
/// and the guard yields. THIS lane then reports the span itself. The C lane
/// does not: `spec/05-risc-primitives.md` section 2.4.1's closed list of
/// runtime-bound errors does not include an empty span, so an extent-0 result
/// under a declared literal is a claim mismatch there and C reports the claim.
/// That divergence is chelis#1795's, not this function's; the inline comment
/// at the call site carries the full argument.
///
/// A span whose END exceeds the operand's extent is declined for the same
/// reason, and chelis#1797 is why it now can be. Such a span is out of domain,
/// and section 2.4.1 makes "a shrink range overshoot" an error every execution
/// mode reports with matching language, so the guard must not compare a claim
/// against the span's arithmetic width and report a claim mismatch: the claim
/// is not what is wrong. Declining hands the report to the operation, and
/// `shrink` now answers an out-of-domain bound with `SHRINK_DOMAIN_TRAP`,
/// which is the compiled lane's own two lines. Before that repair the
/// operation's answer was an `assert!`, so declining would have traded a guard
/// naming the wrong reason for a panic, and this slice bounded its cross-lane
/// claim to IN-DOMAIN spans instead.
///
/// This is also the order the compiled lane takes, rather than a choice made
/// twice: C builds the movement plan before the guard site runs, so
/// `ShapeMetadata::shrunk` rejects the bounds and the guard never executes.
fn computed_axis_extent_value(
    computed: &crate::axis_sources::ComputedAxisExtent,
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
    resolved_stride_steps: Option<&[std::num::NonZeroUsize]>,
) -> Result<Option<usize>, String> {
    match computed {
        crate::axis_sources::ComputedAxisExtent::ShrinkSpan {
            start,
            end,
            operand_axis,
        } => {
            // `RtDim::ToEnd` resolves to the OPERAND's realized extent, which
            // is readable here because the operand is one of this node's
            // producers. `resolve_eval_pairs` resolves the same two bounds
            // against the same shape when the operation itself runs.
            //
            // An absent operand or axis is a malformed graph that `verify`
            // rejects, and the guard declines rather than substituting a
            // number for it: a fabricated extent would compare a claim
            // against a value nothing produced, and the operation's own
            // failure is the one that names the defect.
            let Some(extent) = node
                .inputs
                .first()
                .and_then(|id| values.get(id))
                .and_then(|operand| operand.shape.get(*operand_axis).copied())
            else {
                return Ok(None);
            };
            let start = resolve_eval_bound(start, node, values, extent)?;
            let end = resolve_eval_bound(end, node, values, extent)?;
            if end > extent {
                return Ok(None);
            }
            Ok(end.checked_sub(start).filter(|span| *span > 0))
        }
        crate::axis_sources::ComputedAxisExtent::PadSpan {
            before,
            after,
            operand_axis,
        } => {
            // The operand is this node's producer, so its realized extent is
            // readable here, before the pad allocates. An absent operand or
            // axis is a malformed graph the verifier rejects, and the guard
            // declines rather than substituting a number: a fabricated extent
            // would compare a claim against a value nothing produced.
            let Some(extent) = node
                .inputs
                .first()
                .and_then(|id| values.get(id))
                .and_then(|operand| operand.shape.get(*operand_axis).copied())
            else {
                return Ok(None);
            };
            // `resolve_eval_bound`'s fourth argument resolves `RtDim::ToEnd`,
            // which a padding bound never is; passing the operand's extent
            // keeps one resolver for both owners rather than a second that
            // differs only in what it refuses.
            let before = resolve_eval_bound(before, node, values, extent)?;
            let after = resolve_eval_bound(after, node, values, extent)?;
            // A sum past the host's extent capacity computes no extent. The
            // pad's own allocation owns that failure, so the guard yields
            // instead of comparing a wrapped number.
            Ok(extent
                .checked_add(before)
                .and_then(|widened| widened.checked_add(after)))
        }
        crate::axis_sources::ComputedAxisExtent::StrideSpan { step, operand_axis } => {
            // Preserve the operand-axis identity and the signed step carrier
            // through the shared observation. The complete stride vector was
            // resolved at the node boundary before ANY local claim, so a
            // later invalid axis cannot be masked by this axis's mismatch.
            let Some(carrier) = (match &node.op {
                RiscOp::Stride { strides } => strides.get(*operand_axis),
                _ => None,
            }) else {
                return Err(format!(
                    "stride extent at node {}: carrier for axis {} is missing",
                    node.id.0, operand_axis
                ));
            };
            if carrier != step {
                return Err(format!(
                    "stride extent at node {}: carrier for axis {} disagrees with its site",
                    node.id.0, operand_axis
                ));
            }
            let Some(extent) = node
                .inputs
                .first()
                .and_then(|id| values.get(id))
                .and_then(|operand| operand.shape.get(*operand_axis).copied())
            else {
                return Ok(None);
            };
            let step = resolved_stride_steps
                .and_then(|steps| steps.get(*operand_axis))
                .ok_or_else(|| {
                    format!(
                        "stride extent at node {}: prevalidated step for axis {} is missing",
                        node.id.0, operand_axis
                    )
                })?;
            Ok(Some(extent.div_ceil(step.get())))
        }
    }
}

pub fn eval_tensor_with<F>(
    dag: &Dag,
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    eval_tensor_internal(
        dag,
        None,
        false,
        INITIAL_RANDOM_STREAM_ORDINAL,
        None,
        load_input,
    )
    .map(|(values, _)| values)
}

pub fn eval_tensor_with_strict<F>(
    dag: &Dag,
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    eval_tensor_internal(
        dag,
        None,
        true,
        INITIAL_RANDOM_STREAM_ORDINAL,
        None,
        load_input,
    )
    .map(|(values, _)| values)
}

pub fn eval_tensor_roots_with<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    if roots.is_empty() {
        return eval_tensor_internal(
            dag,
            None,
            false,
            INITIAL_RANDOM_STREAM_ORDINAL,
            None,
            load_input,
        )
        .map(|(values, _)| values);
    }
    reject_drop_roots(dag, roots)?;
    let live = live_mask_for_roots(dag, roots);
    eval_tensor_internal(
        dag,
        Some(&live),
        false,
        INITIAL_RANDOM_STREAM_ORDINAL,
        None,
        load_input,
    )
    .map(|(values, _)| values)
}

pub fn eval_tensor_roots_with_strict<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    if roots.is_empty() {
        return eval_tensor_internal(
            dag,
            None,
            true,
            INITIAL_RANDOM_STREAM_ORDINAL,
            None,
            load_input,
        )
        .map(|(values, _)| values);
    }
    reject_drop_roots(dag, roots)?;
    let live = live_mask_for_roots(dag, roots);
    eval_tensor_internal(
        dag,
        Some(&live),
        true,
        INITIAL_RANDOM_STREAM_ORDINAL,
        None,
        load_input,
    )
    .map(|(values, _)| values)
}

/// Evaluate roots while threading the executed Random path's next ordinal.
/// Only `UniformLike` nodes carrying a scalar Bool activation participate;
/// ordinary baked-seed DAGs retain their historical behavior.
pub fn eval_tensor_roots_with_strict_random_progress<F>(
    dag: &Dag,
    roots: &[NodeId],
    random_counter: u64,
    load_input: F,
) -> Result<(UnordMap<NodeId, TensorValue>, u64), String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    if roots.is_empty() {
        return eval_tensor_internal(dag, None, true, random_counter, None, load_input);
    }
    reject_drop_roots(dag, roots)?;
    let live = live_mask_for_roots(dag, roots);
    eval_tensor_internal(dag, Some(&live), true, random_counter, None, load_input)
}

/// Evaluate roots with the caller's literal result obligations at their
/// producing operations, retaining the ordinary executed Random prefix.
pub fn eval_tensor_roots_with_result_claims<F>(
    dag: &Dag,
    roots: &[NodeId],
    random_counter: u64,
    result_claims: &[crate::TensorType],
    load_input: F,
) -> Result<(UnordMap<NodeId, TensorValue>, u64), String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    reject_drop_roots(dag, roots)?;
    let live = live_mask_for_roots(dag, roots);
    eval_tensor_internal_with_result_claims(
        dag,
        Some(&live),
        true,
        random_counter,
        None,
        result_claims,
        load_input,
    )
}

fn reject_drop_roots(dag: &Dag, roots: &[NodeId]) -> Result<(), String> {
    for root in roots {
        if let Some(node) = dag.get(*root)
            && matches!(node.op, RiscOp::Drop)
        {
            return Err(format!(
                "drop at node {} is terminal and cannot be evaluated as a root",
                root.0
            ));
        }
    }
    Ok(())
}

pub fn eval_tensor(
    dag: &Dag,
    inputs: &UnordMap<String, TensorValue>,
) -> Result<UnordMap<NodeId, TensorValue>, String> {
    eval_tensor_with(dag, |name| inputs.get(name).cloned())
}

/// Evaluate a DAG on scalar inputs. Each node produces a single f64.
pub fn eval_scalar(dag: &Dag, inputs: &UnordMap<String, f64>) -> UnordMap<NodeId, f64> {
    let tensor_inputs: UnordMap<String, TensorValue> = inputs
        .to_sorted()
        .into_iter()
        .map(|(name, value)| (name.clone(), TensorValue::scalar(*value)))
        .collect();
    eval_tensor(dag, &tensor_inputs)
        .expect("scalar evaluation should not fail")
        .into_sorted()
        .into_iter()
        .map(|(id, value)| (id, value.first_f64_lossy_or_zero()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::RiscOp;
    use crate::lower::{LoweredLibrary, lower_program_to_library};
    use chelis_deep::parser::parse_str;
    use chelis_types::types::Prim;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn vec3_f32() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        }
    }

    fn tensor_ty(dims: &[usize], precision: Prim) -> TensorType {
        TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision,
        }
    }

    #[test]
    fn path_sensitive_uniform_like_advances_only_active_nodes() {
        let mut dag = Dag::new();
        let ty = tensor_ty(&[2], Prim::F32);
        let template = dag.add_node(
            RiscOp::Load {
                name: "template".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let inactive = dag.add_node(
            RiscOp::synth_const(Prim::Bool, 0.0),
            vec![],
            tensor_ty(&[], Prim::Bool),
            None,
        );
        let active = dag.add_node(
            RiscOp::synth_const(Prim::Bool, 1.0),
            vec![],
            tensor_ty(&[], Prim::Bool),
            None,
        );
        let skipped = dag.add_node(
            RiscOp::UniformLike {
                low: 0.0,
                high: 1.0,
                seed: 17,
            },
            vec![template, inactive],
            ty.clone(),
            None,
        );
        let executed = dag.add_node(
            RiscOp::UniformLike {
                low: 0.0,
                high: 1.0,
                seed: 17,
            },
            vec![template, active],
            ty,
            None,
        );
        dag.add_root(skipped);
        dag.add_root(executed);

        let roots = dag.roots().to_vec();
        let (values, next) =
            eval_tensor_roots_with_strict_random_progress(&dag, &roots, 7, |name| {
                (name == "template").then(|| TensorValue::from_vec(vec![2], vec![0.0; 2]))
            })
            .expect("path-sensitive Random DAG evaluates");
        assert_eq!(next, 8, "only the active draw consumes an ordinal");
        let expected_seed = 17 ^ 7_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let expected = uniform_like(&[2], 0.0, 1.0, expected_seed, Prim::F32).unwrap();
        assert_eq!(values[&executed], expected);
        let skipped_expected = uniform_like(&[2], 0.0, 1.0, 17, Prim::F32).unwrap();
        assert_eq!(values[&skipped], skipped_expected);

        let (values, next) =
            eval_tensor_roots_with_strict_random_progress(&dag, &roots, u64::MAX, |name| {
                (name == "template").then(|| TensorValue::from_vec(vec![2], vec![0.0; 2]))
            })
            .expect("legacy Random progress retains its saturation boundary");
        assert_eq!(next, u64::MAX);
        let expected_seed = 17 ^ u64::MAX.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        assert_eq!(
            values[&executed],
            uniform_like(&[2], 0.0, 1.0, expected_seed, Prim::F32).unwrap()
        );
        assert_eq!(values[&skipped], skipped_expected);
    }

    #[test]
    fn load_ingress_rejects_an_already_tagged_dtype_substitution() {
        let f16 = finalize_tensor("test", Prim::F16, RawTensor::Float(vec![1.5])).unwrap();
        let input = TensorValue::from_storage(vec![1], f16);
        let err = ingress_to_declared("x", Prim::F32, &input)
            .expect_err("a tagged f16 input must not be contextually cast to f32");
        assert!(err.contains("f16"), "source dtype must be named: {err}");
        assert!(err.contains("f32"), "declared dtype must be named: {err}");

        let i8 = finalize_tensor("test", Prim::Int8, RawTensor::Int(vec![7])).unwrap();
        let input = TensorValue::from_storage(vec![1], i8);
        let err = ingress_to_declared("x", Prim::Int64, &input)
            .expect_err("a tagged i8 input must not be contextually cast to i64");
        assert!(err.contains("i8"), "source dtype must be named: {err}");
        assert!(err.contains("i64"), "declared dtype must be named: {err}");
    }

    #[test]
    fn load_ingress_preserves_an_exact_dtype_match() {
        let f32 = finalize_tensor("test", Prim::F32, RawTensor::Float(vec![1.5])).unwrap();
        let input = TensorValue::from_storage(vec![1], f32);
        assert_eq!(ingress_to_declared("x", Prim::F32, &input).unwrap(), input);
    }

    #[test]
    fn load_ingress_freezes_raw_host_values_at_the_declared_dtype() {
        let raw = TensorValue::from_vec(vec![1], vec![0.1]);
        let frozen = ingress_to_declared("x", Prim::F32, &raw)
            .expect("raw host ingress is finalized once at the declared dtype");
        assert_eq!(frozen.prim(), Prim::F32);
        assert_eq!(frozen.element_f64_lossy(0), 0.1_f32 as f64);

        let err = ingress_to_declared(
            "count",
            Prim::Int8,
            &TensorValue::from_vec(vec![1], vec![300.0]),
        )
        .expect_err("raw ingress still domain-checks at the declared dtype");
        assert!(err.contains("numeric trap: overflow"), "{err}");
    }

    /// chelis#368: the `shrink` evaluator must not overflow when a
    /// `SHRINK_TO_END` (`usize::MAX`) full-axis sentinel survives into eval.
    /// `bind_symbolic_dims` normally resolves the sentinel first (and the
    /// `needs_symbolic_binding` gate now routes every sentinel-bearing DAG
    /// through it), but this is the defensive backstop: the sentinel means
    /// "to the end of this axis", so it resolves to the runtime extent
    /// (`input.shape[axis]`) instead of computing `usize::MAX - start` and
    /// overflowing `numel` ("attempt to multiply with overflow").
    #[test]
    fn shrink_clamps_shrink_to_end_sentinel_without_overflow() {
        // 2x3 input [[1,2,3],[4,5,6]]; axis 0 sliced [1,2] (concrete row 1),
        // axis 1 a full-axis identity via the sentinel.
        let input = TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = shrink(&input, &[(1, 2), (0, SHRINK_TO_END)]).expect("an in-domain slice");
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.to_f64_lossy_vec(), vec![4.0, 5.0, 6.0]);
    }

    /// chelis#1797: a genuine (non-sentinel) `end` beyond the input extent is
    /// answered with `spec/05-risc-primitives.md` section 2.4.1's overshoot
    /// error, in the compiled lane's words, rather than read out of bounds or
    /// returned as a silently wrong slice. Do NOT clamp to a valid range (that
    /// hides the producing pass's bug), and do not panic (that is neither this
    /// error nor any other).
    ///
    /// EVIDENTIARY STATUS: regression test. On `6abca2406` this input panicked
    /// with `eval::shrink: axis 0 bound end 5 exceeds input extent 3`, which is
    /// what `#[should_panic(expected = "end 5 exceeds input extent 3")]` pinned.
    #[test]
    fn shrink_rejects_an_end_beyond_extent_as_a_domain_trap() {
        let input = TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]);
        let err = shrink(&input, &[(0, 5)]).expect_err("an overshoot produces no value");
        assert_eq!(err, SHRINK_DOMAIN_TRAP);
    }

    /// chelis#1797: an inverted bound (`start > end`) takes the same typed
    /// rejection rather than underflowing `end - start` (usize), because
    /// `ShapeMetadata::shrunk` gives it the same `Domain` message on the
    /// compiled lane.
    ///
    /// This kernel branch is a backstop from the DAG evaluator's side: the
    /// `RiscOp::Shrink` arm rejects `start >= end` first, with chelis#616's
    /// admission-rule wording, so a program cannot reach this branch through
    /// that path. That earlier rejection is a separate lane divergence tracked
    /// by chelis#1795 and is NOT repaired here.
    ///
    /// EVIDENTIARY STATUS: regression test. On `6abca2406` this input panicked
    /// with `eval::shrink: axis 0 bound start 3 exceeds end 1`.
    #[test]
    fn shrink_rejects_inverted_bounds_as_a_domain_trap() {
        let input = TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]);
        let err = shrink(&input, &[(3, 1)]).expect_err("an inverted bound produces no value");
        assert_eq!(err, SHRINK_DOMAIN_TRAP);
    }

    /// Negative parity for both rows above: an in-domain slice of the same
    /// operand still produces its elements, so the rejection is keyed on the
    /// bounds and not on the operation.
    ///
    /// EVIDENTIARY STATUS: regression test for the `Result` return, disposition
    /// lock for the slice itself, which `6abca2406` already produced.
    #[test]
    fn shrink_still_slices_an_in_domain_range() {
        let input = TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]);
        let out = shrink(&input, &[(1, 3)]).expect("an in-domain slice");
        assert_eq!(out.shape, vec![2]);
        assert_eq!(out.to_f64_lossy_vec(), vec![2.0, 3.0]);
    }

    /// chelis#523: the post-bind re-verify. `verify` runs BEFORE
    /// `bind_symbolic_dims` and skips the C10 shrink-bound check on symbolic
    /// axes, so an out-of-bounds shrink whose overrun only becomes visible once
    /// the symbolic input extent is bound must be caught after binding — a
    /// clean `Err`, not a panic or silent OOB read. Here the input is symbolic
    /// `tensor[n]`; the shrink's `end = 10` exceeds the extent once `n` binds to
    /// 4 from the runtime input.
    #[test]
    fn post_bind_reverify_rejects_out_of_bounds_symbolic_shrink() {
        let mut dag = Dag::new();
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], sym_ty, None);
        // A shrink whose end (10) exceeds the eventual concrete extent (4).
        let shr = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::Lit(10))],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(10)],
                precision: Prim::F32,
            },
            None,
        );
        let err = eval_tensor_roots_with_strict(&dag, &[shr], |name| match name {
            "x" => Some(TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0])),
            _ => None,
        })
        .unwrap_err();
        assert!(
            err.contains("chelis#523") && err.contains("end 10 > input extent 4"),
            "expected the post-bind OOB-shrink re-verify error, got: {err}"
        );
    }

    /// Negative parity: an IN-bounds symbolic shrink must still eval cleanly
    /// (the re-verify must not over-reject legitimate programs). `tensor[n]`
    /// with `n` bound to 4, shrink `[1, 3)` -> the middle two elements.
    #[test]
    fn post_bind_reverify_allows_in_bounds_symbolic_shrink() {
        let mut dag = Dag::new();
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], sym_ty, None);
        let shr = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::Lit(3))],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        let vals = eval_tensor_roots_with_strict(&dag, &[shr], |name| match name {
            "x" => Some(TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0])),
            _ => None,
        })
        .unwrap();
        assert_eq!(vals[&shr], TensorValue::from_vec(vec![2], vec![2.0, 3.0]));
    }

    // ---- reduce_window reverse-mode adjoint (RiscOp::ReduceWindowGrad) ----

    /// Scalar loss `sum(reduce_window(x))` used as the finite-difference
    /// oracle: its cotangent w.r.t. every forward output is exactly 1, so
    /// the analytic input gradient is `reduce_window_grad(x, ones)`.
    fn rw_loss(x: &TensorValue, r: ReduceWindowKind, w: &[usize], s: &[usize]) -> f64 {
        reduce_window(x, r, w, s, Prim::F64)
            .unwrap()
            .to_f64_lossy_vec()
            .iter()
            .sum()
    }

    /// Central finite-difference gradient of `rw_loss` w.r.t. each element.
    fn rw_fd_grad(x: &TensorValue, r: ReduceWindowKind, w: &[usize], s: &[usize]) -> Vec<f64> {
        let eps = 1e-4;
        (0..x.len())
            .map(|i| {
                let mut plus = x.to_f64_lossy_vec();
                let mut minus = x.to_f64_lossy_vec();
                plus[i] += eps;
                minus[i] -= eps;
                let xp = TensorValue::from_vec(x.shape.clone(), plus);
                let xm = TensorValue::from_vec(x.shape.clone(), minus);
                (rw_loss(&xp, r, w, s) - rw_loss(&xm, r, w, s)) / (2.0 * eps)
            })
            .collect()
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "{ctx}: element {i} differs: {x} vs {y} (tol {tol})"
            );
        }
    }

    #[test]
    fn reduce_window_grad_matches_finite_difference_all_reducers() {
        // Distinct values (no ties) so Max/Min gradients are unique and
        // both the analytic adjoint and the central difference agree.
        let x = TensorValue::from_vec(
            vec![4, 4],
            vec![
                3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.0, 3.5, 8.0, 9.5, 7.0, 0.5, 2.5, 6.5,
            ],
        );
        // Overlapping (stride < window) and a non-unit stride exercise the
        // overlap-add / select-and-scatter accumulation across windows.
        for (w, s) in [
            (vec![2, 2], vec![1, 1]),
            (vec![2, 2], vec![2, 2]),
            (vec![3, 3], vec![1, 1]),
            (vec![2, 3], vec![2, 1]),
        ] {
            for r in [
                ReduceWindowKind::Sum,
                ReduceWindowKind::Mean,
                ReduceWindowKind::Max,
                ReduceWindowKind::Min,
            ] {
                let out = reduce_window(&x, r, &w, &s, Prim::F64).unwrap();
                let ones = TensorValue::from_vec(out.shape.clone(), vec![1.0; out.len()]);
                let analytic = reduce_window_grad(&x, &ones, r, &w, &s, Prim::F64).unwrap();
                assert_eq!(analytic.shape, x.shape);
                let fd = rw_fd_grad(&x, r, &w, &s);
                assert_close(
                    &analytic.to_f64_lossy_vec(),
                    &fd,
                    1e-3,
                    &format!("reducer={r:?} window={w:?} stride={s:?}"),
                );
            }
        }
    }

    #[test]
    fn reduce_window_grad_sum_is_window_cover_count() {
        // With g = ones, Sum's adjoint at position i is exactly the number
        // of windows covering i. For a [4] input, window=2, stride=1 the
        // windows are [0,1],[1,2],[2,3]; interior positions are covered
        // twice, the two ends once.
        let x = TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]);
        let out = reduce_window(&x, ReduceWindowKind::Sum, &[2], &[1], Prim::F64).unwrap();
        let ones = TensorValue::from_vec(out.shape.clone(), vec![1.0; out.len()]);
        let din =
            reduce_window_grad(&x, &ones, ReduceWindowKind::Sum, &[2], &[1], Prim::F64).unwrap();
        assert_eq!(din.to_f64_lossy_vec(), vec![1.0, 2.0, 2.0, 1.0]);

        // Mean is Sum scaled by 1 / window_volume (= 2 here).
        let din_mean =
            reduce_window_grad(&x, &ones, ReduceWindowKind::Mean, &[2], &[1], Prim::F64).unwrap();
        assert_eq!(din_mean.to_f64_lossy_vec(), vec![0.5, 1.0, 1.0, 0.5]);
    }

    #[test]
    fn phase2_reduce_window_grad_accumulates_overlaps_at_declared_width() {
        let f32_tensor = |shape, values| {
            TensorValue::from_storage(
                shape,
                finalize_tensor("test", Prim::F32, RawTensor::Float(values)).unwrap(),
            )
        };
        let x = f32_tensor(vec![5], vec![0.0; 5]);
        let witness = f32_tensor(vec![3], vec![16_777_216.0, 1.0, -16_777_216.0]);
        let actual =
            reduce_window_grad(&x, &witness, ReduceWindowKind::Sum, &[3], &[1], Prim::F32).unwrap();
        assert_eq!(
            actual.to_f64_lossy_vec(),
            vec![
                16_777_216.0,
                16_777_216.0,
                0.0,
                -16_777_215.0,
                -16_777_216.0
            ],
            "the center receives all three cotangents and must round after each f32 add"
        );

        let control = f32_tensor(vec![3], vec![4.0, 1.0, -4.0]);
        let actual =
            reduce_window_grad(&x, &control, ReduceWindowKind::Sum, &[3], &[1], Prim::F32).unwrap();
        assert_eq!(actual.to_f64_lossy_vec(), vec![4.0, 5.0, 1.0, -3.0, -4.0]);
    }

    #[test]
    fn reduce_window_grad_max_routes_to_argmax_and_accumulates_overlap() {
        // Strictly increasing input over a [4] window=2 stride=1: windows
        // [0,1]->max@1, [1,2]->max@2, [2,3]->max@3. With distinct upstream
        // gradients, position 0 gets nothing, 1 gets g[0], 2 gets g[1],
        // 3 gets g[2].
        let x = TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let g = TensorValue::from_vec(vec![3], vec![5.0, 7.0, 11.0]);
        let din = reduce_window_grad(&x, &g, ReduceWindowKind::Max, &[2], &[1], Prim::F64).unwrap();
        assert_eq!(din.to_f64_lossy_vec(), vec![0.0, 5.0, 7.0, 11.0]);

        // Min over the same increasing input routes to the window minimum:
        // windows pick positions 0,1,2; position 3 gets nothing.
        let din_min =
            reduce_window_grad(&x, &g, ReduceWindowKind::Min, &[2], &[1], Prim::F64).unwrap();
        assert_eq!(din_min.to_f64_lossy_vec(), vec![5.0, 7.0, 11.0, 0.0]);
    }

    #[test]
    fn reduce_window_grad_max_distributes_to_ties() {
        // A flat window: every position equals the max, so each tied
        // position receives the full upstream gradient (the max_reduce
        // mask convention), not a 1/k share.
        let x = TensorValue::from_vec(vec![3], vec![2.0, 2.0, 2.0]);
        let g = TensorValue::from_vec(vec![2], vec![4.0, 4.0]);
        let din = reduce_window_grad(&x, &g, ReduceWindowKind::Max, &[2], &[1], Prim::F64).unwrap();
        // windows [0,1] and [1,2]: pos0 += 4 (win0), pos1 += 4+4, pos2 += 4.
        assert_eq!(din.to_f64_lossy_vec(), vec![4.0, 8.0, 4.0]);
    }

    #[test]
    fn eval_add() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let vals = eval_scalar(&dag, &UnordMap::new());
        assert!((vals[&c] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eval_vector_add() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec3_f32(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec3_f32(), None);
        let c = dag.add_node(RiscOp::Add, vec![a, b], vec3_f32(), None);
        let mut inputs = UnordMap::new();
        inputs.insert(
            "a".into(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        inputs.insert(
            "b".into(),
            TensorValue::from_vec(vec![3], vec![4.0, 5.0, 6.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        assert_eq!(
            vals[&c],
            TensorValue::from_vec(vec![3], vec![5.0, 7.0, 9.0])
        );
    }

    /// [04-NUM-8]/dtype-semantics C5: the IR reference evaluator must call
    /// the exact-width integer kernel, never project i64 operands through
    /// binary64 before arithmetic.
    #[test]
    fn int64_elementwise_add_is_exact_above_binary64_mantissa() {
        let dag = int_div_dag(RiscOp::Add, Prim::Int64);
        let mut inputs = UnordMap::new();
        inputs.insert(
            "a".into(),
            TensorValue::finalize_from_wide_int(
                "test",
                Prim::Int64,
                vec![2],
                vec![(1_i64 << 53) + 1, (1_i64 << 53) + 2],
            )
            .unwrap(),
        );
        inputs.insert(
            "b".into(),
            TensorValue::finalize_from_wide_int("test", Prim::Int64, vec![2], vec![1, -1]).unwrap(),
        );
        let values = eval_tensor(&dag, &inputs).expect("exact i64 add");
        assert_eq!(
            values[&dag.roots()[0]].storage().to_i64_exact_vec(),
            Some(vec![(1_i64 << 53) + 2, (1_i64 << 53) + 1])
        );
    }

    /// Negative twin for the exact integer kernel: overflow traps at the
    /// declared width and names both the operation and dtype.
    #[test]
    fn int8_elementwise_add_overflow_uses_branded_trap() {
        let dag = int_div_dag(RiscOp::Add, Prim::Int8);
        let mut inputs = UnordMap::new();
        inputs.insert(
            "a".into(),
            TensorValue::finalize_from_wide_int("test", Prim::Int8, vec![2], vec![127, 1]).unwrap(),
        );
        inputs.insert(
            "b".into(),
            TensorValue::finalize_from_wide_int("test", Prim::Int8, vec![2], vec![1, 1]).unwrap(),
        );
        assert_eq!(
            eval_tensor(&dag, &inputs).expect_err("i8 overflow must trap"),
            "numeric trap: overflow in add at i8"
        );
    }

    /// Comparison reads the finalized operand storage, not the result dtype
    /// and not a lossy float projection.
    #[test]
    fn int64_comparison_is_exact_above_binary64_mantissa() {
        let mut dag = Dag::new();
        let ty = tensor_ty(&[2], Prim::Int64);
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty, None);
        let out = dag.add_node(RiscOp::CmpLt, vec![a, b], tensor_ty(&[2], Prim::Bool), None);
        dag.add_root(out);

        let mut inputs = UnordMap::new();
        inputs.insert(
            "a".into(),
            TensorValue::finalize_from_wide_int(
                "test",
                Prim::Int64,
                vec![2],
                vec![1_i64 << 53, (1_i64 << 53) + 2],
            )
            .unwrap(),
        );
        inputs.insert(
            "b".into(),
            TensorValue::finalize_from_wide_int(
                "test",
                Prim::Int64,
                vec![2],
                vec![(1_i64 << 53) + 1, (1_i64 << 53) + 1],
            )
            .unwrap(),
        );
        let values = eval_tensor(&dag, &inputs).expect("exact i64 comparison");
        assert_eq!(values[&out].storage().to_i64_exact_vec(), Some(vec![1, 0]));
    }

    /// Fused elementwise execution owes the same exact-width contract as
    /// standalone nodes; fusion cannot reopen the f64 closure seam.
    #[test]
    fn fused_int64_arithmetic_is_exact_above_binary64_mantissa() {
        let mut dag = Dag::new();
        let ty = tensor_ty(&[1], Prim::Int64);
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
        let one = dag.add_node(
            RiscOp::Load { name: "one".into() },
            vec![],
            ty.clone(),
            None,
        );
        let add = dag.add_node(RiscOp::Add, vec![a, one], ty.clone(), None);
        let mul = dag.add_node(RiscOp::Mul, vec![add, one], ty, None);
        dag.add_root(mul);
        let fused = crate::fuse::fuse(&dag);
        assert!(
            fused
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::FusedElem { .. }))
        );

        let mut inputs = UnordMap::new();
        inputs.insert(
            "a".into(),
            TensorValue::finalize_from_wide_int(
                "test",
                Prim::Int64,
                vec![1],
                vec![(1_i64 << 53) + 1],
            )
            .unwrap(),
        );
        inputs.insert(
            "one".into(),
            TensorValue::finalize_from_wide_int("test", Prim::Int64, vec![1], vec![1]).unwrap(),
        );
        let values = eval_tensor(&fused, &inputs).expect("exact fused i64 arithmetic");
        let root = fused.roots()[0];
        assert_eq!(
            values[&root].storage().to_i64_exact_vec(),
            Some(vec![(1_i64 << 53) + 2])
        );
    }

    #[test]
    fn eval_sparse_gather_axis1_preserves_outer_and_inner_layout() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[2, 4, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            tensor_ty(&[3], Prim::Int64),
            None,
        );
        let out = dag.add_node(
            RiscOp::Gather { axis: 1 },
            vec![values, indices],
            tensor_ty(&[2, 3, 2], Prim::F32),
            None,
        );
        dag.add_root(out);

        let inputs = UnordMap::from([
            (
                "values".to_string(),
                TensorValue::from_vec(vec![2, 4, 2], (0..16).map(|x| x as f64).collect()),
            ),
            (
                "indices".to_string(),
                TensorValue::from_vec(vec![3], vec![2.0, 0.0, 3.0]),
            ),
        ]);
        let vals = eval_tensor(&dag, &inputs).unwrap();
        assert_eq!(
            vals[&out],
            TensorValue::from_vec(
                vec![2, 3, 2],
                vec![
                    4.0, 5.0, 0.0, 1.0, 6.0, 7.0, 12.0, 13.0, 8.0, 9.0, 14.0, 15.0
                ],
            )
        );
    }

    #[test]
    fn eval_sparse_scatter_add_axis1_accumulates_duplicate_indices() {
        let mut dag = Dag::new();
        let target = dag.add_node(
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[2, 3, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            tensor_ty(&[4], Prim::Int32),
            None,
        );
        let updates = dag.add_node(
            RiscOp::Load {
                name: "updates".into(),
            },
            vec![],
            tensor_ty(&[2, 4, 2], Prim::F32),
            None,
        );
        let out = dag.add_node(
            RiscOp::ScatterAdd { axis: 1 },
            vec![target, indices, updates],
            tensor_ty(&[2, 3, 2], Prim::F32),
            None,
        );
        dag.add_root(out);

        let inputs = UnordMap::from([
            (
                "target".to_string(),
                TensorValue::from_vec(vec![2, 3, 2], vec![0.0; 12]),
            ),
            (
                "indices".to_string(),
                TensorValue::from_vec(vec![4], vec![1.0, 0.0, 1.0, 2.0]),
            ),
            (
                "updates".to_string(),
                TensorValue::from_vec(
                    vec![2, 4, 2],
                    vec![
                        1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0,
                        70.0, 8.0, 80.0,
                    ],
                ),
            ),
        ]);
        let vals = eval_tensor(&dag, &inputs).unwrap();
        assert_eq!(
            vals[&out],
            TensorValue::from_vec(
                vec![2, 3, 2],
                vec![
                    2.0, 20.0, 4.0, 40.0, 4.0, 40.0, 6.0, 60.0, 12.0, 120.0, 8.0, 80.0,
                ],
            )
        );
    }

    #[test]
    fn eval_same_rank_expand_broadcast() {
        let mut dag = Dag::new();
        let in_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::F32,
        };
        let out_ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
        let y = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::RtDim::Lit(4),
            },
            vec![x],
            out_ty,
            None,
        );
        let mut inputs = UnordMap::new();
        inputs.insert("x".into(), TensorValue::from_vec(vec![1], vec![2.5]));
        let vals = eval_tensor(&dag, &inputs).unwrap();
        assert_eq!(
            vals[&y],
            TensorValue::from_vec(vec![4], vec![2.5, 2.5, 2.5, 2.5])
        );
    }

    #[test]
    fn eval_root_scoped_does_not_require_unrelated_inputs() {
        let mut dag = Dag::new();
        let ty = vec3_f32();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], ty.clone(), None);
        let sum = dag.add_node(RiscOp::Add, vec![x, x], ty.clone(), None);
        let dead = dag.add_node(RiscOp::Add, vec![y, y], ty.clone(), None);
        dag.add_root(sum);
        dag.add_root(dead);

        let vals = eval_tensor_roots_with(&dag, &[sum], |name| match name {
            "x" => Some(TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])),
            _ => None,
        })
        .unwrap();

        assert_eq!(
            vals[&sum],
            TensorValue::from_vec(vec![3], vec![2.0, 4.0, 6.0])
        );
        assert!(
            !vals.contains_key(&dead),
            "dead branch should not be evaluated in root-scoped mode"
        );
    }

    #[test]
    fn eval_strict_missing_input_is_error() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec3_f32(), None);
        let err = eval_tensor_with_strict(&dag, |_| None).unwrap_err();
        assert!(err.contains("missing required input `x`"));
        assert_eq!(x, NodeId(0));
    }

    #[test]
    fn eval_root_scoped_strict_only_requires_live_inputs() {
        let mut dag = Dag::new();
        let ty = vec3_f32();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], ty.clone(), None);
        let live = dag.add_node(RiscOp::Add, vec![x, x], ty.clone(), None);
        let _dead = dag.add_node(RiscOp::Add, vec![y, y], ty, None);

        let vals = eval_tensor_roots_with_strict(&dag, &[live], |name| match name {
            "x" => Some(TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])),
            _ => None,
        })
        .unwrap();
        assert_eq!(
            vals[&live],
            TensorValue::from_vec(vec![3], vec![2.0, 4.0, 6.0])
        );
    }

    /// chelis#991: a dependency package may contribute generic declarations
    /// outside the selected eval root. Their symbolic inputs are dead and
    /// must not become invented top-level requirements for the live result.
    #[test]
    fn eval_root_scoped_ignores_unrelated_dead_symbolic_input() {
        let mut dag = Dag::new();
        let dead_ty = TensorType {
            dims: vec![DimInfo::Named("k".to_string(), None)],
            precision: Prim::F32,
        };
        let _dead = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], dead_ty, None);
        let live = dag.add_node(
            RiscOp::synth_const(Prim::F32, 7.0),
            vec![],
            scalar_f32(),
            None,
        );

        let values = eval_tensor_roots_with_strict(&dag, &[live], |_| None)
            .expect("dead generic dependency input must not escape into root-scoped eval");
        assert_eq!(values[&live], TensorValue::scalar(7.0));
    }

    #[test]
    fn eval_root_scoped_ignores_dead_canonical_source_with_live_same_named_dim() {
        let mut dag = Dag::new();
        let symbolic = TensorType {
            dims: vec![DimInfo::Named("k".to_string(), None)],
            precision: Prim::F32,
        };
        let _dead = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            symbolic.clone(),
            None,
        );
        let live = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], symbolic, None);

        let values = eval_tensor_roots_with_strict(&dag, &[live], |name| {
            (name == "x").then(|| TensorValue::from_vec(vec![2], vec![3.0, 4.0]))
        })
        .expect("dead same-named symbolic source must not override the live input");
        assert_eq!(values[&live].to_f64_lossy_vec(), vec![3.0, 4.0]);
    }

    /// A shape-only `InputAxis` edge is an ordinary live dependency even
    /// when the tensor's elements are otherwise unused.
    #[test]
    fn eval_root_scoped_strict_resolves_shape_only_input_axis() {
        let mut dag = Dag::new();
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_ty.clone(),
            None,
        );
        let one = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let ones = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::RtDim::InputAxis {
                    tensor: 1,
                    axis: crate::dag::RtAxis::Lit(0),
                },
            },
            vec![one, x],
            sym_ty,
            None,
        );

        let vals = eval_tensor_roots_with_strict(&dag, &[ones], |name| match name {
            "x" => Some(TensorValue::from_vec(vec![3], vec![5.0, 6.0, 7.0])),
            _ => None,
        })
        .unwrap();
        assert_eq!(
            vals[&ones],
            TensorValue::from_vec(vec![3], vec![1.0, 1.0, 1.0]),
            "InputAxis must read 3 from the explicit `x` shape-only input"
        );
    }

    /// Negative parity: an unavailable structural witness is a missing live
    /// input, never a guessed extent.
    #[test]
    fn eval_root_scoped_strict_missing_input_axis_witness_is_input_error() {
        let mut dag = Dag::new();
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_ty.clone(),
            None,
        );
        let one = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let ones = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::RtDim::InputAxis {
                    tensor: 1,
                    axis: crate::dag::RtAxis::Lit(0),
                },
            },
            vec![one, x],
            sym_ty,
            None,
        );

        let err = eval_tensor_roots_with_strict(&dag, &[ones], |_| None).unwrap_err();
        assert!(
            err.contains("missing required input `x`"),
            "expected the structural input error, got: {err}"
        );
    }

    #[test]
    fn eval_root_scoped_input_axis_selects_one_exact_shape_source() {
        let mut dag = Dag::new();
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("k".to_string(), None)],
            precision: Prim::F32,
        };
        let _unrelated = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            sym_ty.clone(),
            None,
        );
        let required = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_ty.clone(),
            None,
        );
        let one = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let ones = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::RtDim::InputAxis {
                    tensor: 1,
                    axis: crate::dag::RtAxis::Lit(0),
                },
            },
            vec![one, required],
            sym_ty,
            None,
        );

        let values = eval_tensor_roots_with_strict(&dag, &[ones], |name| {
            (name == "x").then(|| TensorValue::from_vec(vec![2], vec![3.0, 4.0]))
        })
        .expect("the exact x witness must resolve without consulting same-named a");
        assert_eq!(values[&ones].shape, vec![2]);

        let err = eval_tensor_roots_with_strict(&dag, &[ones], |name| {
            (name == "a").then(|| TensorValue::from_vec(vec![2], vec![3.0, 4.0]))
        })
        .unwrap_err();
        assert!(err.contains("missing required input `x`"), "{err}");
    }

    #[test]
    fn eval_root_scoped_accepts_repeated_symbol_axes_from_one_dead_source() {
        let mut dag = Dag::new();
        let square_ty = TensorType {
            dims: vec![
                DimInfo::Named("n".to_string(), None),
                DimInfo::Named("n".to_string(), None),
            ],
            precision: Prim::F32,
        };
        let shape_source = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], square_ty, None);
        let one = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let vector_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let ones = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::RtDim::InputAxis {
                    tensor: 1,
                    axis: crate::dag::RtAxis::Lit(1),
                },
            },
            vec![one, shape_source],
            vector_ty,
            None,
        );

        let values = eval_tensor_roots_with_strict(&dag, &[ones], |name| {
            (name == "x").then(|| TensorValue::from_vec(vec![2, 2], vec![0.0; 4]))
        })
        .expect("one explicit source may expose the same extent on multiple axes");
        assert_eq!(
            values[&ones],
            TensorValue::from_vec(vec![2], vec![1.0, 1.0])
        );
    }

    fn lower(src: &str) -> Dag {
        lower_library(src).dag().clone()
    }

    fn lower_library(src: &str) -> LoweredLibrary {
        let exprs = parse_str(src).expect("parse failed");
        let checked = chelis_types::check_ir_program(&exprs)
            .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        let checked = chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"));
        lower_program_to_library(&checked)
    }

    #[test]
    fn lowered_relu_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
            (def {} y (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} (var {} relu) (var {} x)))
        "#;
        let dag = lower(src);
        let mut inputs = UnordMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![3], vec![-2.0, 0.5, 4.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(dag.roots().last().expect("DAG root")).unwrap();
        assert_eq!(*last, TensorValue::from_vec(vec![3], vec![0.0, 0.5, 4.0]));
    }

    /// #380: the DAG evaluator must apply the dtype conversion for `cast`,
    /// not pass the f64-backed value through unchanged. The checked default
    /// rejects fractional float->int values rather than choosing a rounding
    /// policy implicitly.
    #[test]
    fn lowered_cast_fractional_float_to_int_traps_domain() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
            (def {} y
              (cast {type: (t-tensor {} (d-lit {} 3) (t-prim {} i32))}
                    (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)
                    (t-prim {} i32)))
        "#;
        let dag = lower(src);
        let mut inputs = UnordMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![3], vec![2.7, -2.7, 3.0]),
        );
        let err = eval_tensor(&dag, &inputs).unwrap_err();
        assert_eq!(
            err, "numeric trap: domain in cast at i32",
            "fractional cast to i32 must trap in the DAG evaluator (#380)",
        );
    }

    /// #380 negative / direction guard: int->float preserves the value
    /// exactly (no spurious truncation on the int->float direction), and
    /// float->f32 narrows. Pins that the sealed cast ladder preserves the
    /// int->float direction rather than applying the checked int-target rule.
    #[test]
    fn lowered_cast_int_to_float_preserves_value() {
        // i32 input carries integral f64 storage; cast to f64 must keep it.
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} i32))} x))
            (def {} y
              (cast {type: (t-tensor {} (d-lit {} 2) (t-prim {} f64))}
                    (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} i32))} x)
                    (t-prim {} f64)))
        "#;
        let dag = lower(src);
        let mut inputs = UnordMap::new();
        inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![7.0, -3.0]));
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(dag.roots().last().expect("DAG root")).unwrap();
        assert_eq!(
            *last,
            TensorValue::from_vec(vec![2], vec![7.0, -3.0]),
            "int->float cast must preserve the value (#380)",
        );
    }

    #[test]
    fn lowered_matmul_has_correct_numeric_result() {
        let src = r#"
            (def {} a (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} a))
            (def {} b (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))} b))
            (def {} c
              (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
                   (var {} matmul) (var {} a) (var {} b)))
        "#;
        let dag = lower(src);
        let mut inputs = UnordMap::new();
        inputs.insert(
            "a".into(),
            TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        inputs.insert(
            "b".into(),
            TensorValue::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(dag.roots().last().expect("DAG root")).unwrap();
        assert_eq!(
            *last,
            TensorValue::from_vec(vec![2, 2], vec![58.0, 64.0, 139.0, 154.0])
        );
    }

    #[test]
    fn lowered_softmax_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
                   (var {} softmax) (var {} x) (lit {} 0)))
        "#;
        let dag = lower(src);
        let mut inputs = UnordMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(dag.roots().last().expect("DAG root")).unwrap();
        let expected = [0.09003057, 0.24472847, 0.66524096];
        for (actual, target) in last.to_f64_lossy_vec().iter().zip(expected.iter()) {
            assert!((actual - target).abs() < 1e-5);
        }
    }

    #[test]
    fn lowered_layer_norm_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))} x))
            (def {} gamma (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} gamma))
            (def {} beta (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} beta))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
                   (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta) (lit {type: (t-prim {} f32)} 0.00001)))
        "#;
        let dag = lower(src);
        let mut inputs = UnordMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 5.0]),
        );
        inputs.insert(
            "gamma".into(),
            TensorValue::from_vec(vec![2], vec![1.0, 1.5]),
        );
        inputs.insert(
            "beta".into(),
            TensorValue::from_vec(vec![2], vec![0.5, -0.5]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(dag.roots().last().expect("DAG root")).unwrap();
        let expected = [-0.49998, 0.99997, -0.499995, 0.99998875];
        for (actual, target) in last.to_f64_lossy_vec().iter().zip(expected.iter()) {
            assert!((actual - target).abs() < 2e-4, "{actual} vs {target}");
        }
    }

    #[test]
    fn lowered_conv_1x1_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))} x))
            (def {} k (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 1) (d-lit {} 1) (t-prim {} f32))} k))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
                   (var {} conv) (var {} x) (var {} k) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (var {} Nil))) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 0) (lit {type: (t-prim {} i64)} 0)) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 0) (lit {type: (t-prim {} i64)} 0)) (var {} Nil)))))
        "#;
        let library = lower_library(src);
        let dag = library.dag();
        let mut inputs = UnordMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        );
        inputs.insert(
            "k".into(),
            TensorValue::from_vec(vec![1, 1, 1, 1], vec![2.0]),
        );
        let vals = eval_tensor(dag, &inputs).unwrap();
        let root = library
            .symbol_table()
            .get("y")
            .expect("named convolution result is lowered");
        let last = vals.get(root).unwrap();
        assert_eq!(
            *last,
            TensorValue::from_vec(vec![1, 1, 2, 2], vec![2.0, 4.0, 6.0, 8.0])
        );
    }

    #[test]
    fn lowered_conv_2x2_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} x))
            (def {} k (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))} k))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
                   (var {} conv) (var {} x) (var {} k) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (var {} Nil))) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 0) (lit {type: (t-prim {} i64)} 0)) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 0) (lit {type: (t-prim {} i64)} 0)) (var {} Nil)))))
        "#;
        let library = lower_library(src);
        let dag = library.dag();
        let mut inputs = UnordMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(
                vec![1, 1, 3, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ),
        );
        inputs.insert(
            "k".into(),
            TensorValue::from_vec(vec![1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]),
        );
        let vals = eval_tensor(dag, &inputs).unwrap();
        let root = library
            .symbol_table()
            .get("y")
            .expect("named convolution result is lowered");
        let last = vals.get(root).unwrap();
        assert_eq!(
            *last,
            TensorValue::from_vec(vec![1, 1, 2, 2], vec![12.0, 16.0, 24.0, 28.0])
        );
    }

    #[test]
    fn lowered_dropout_is_deterministic_for_same_seed() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 32) (t-prim {} f32))} 1.0))
            (def {} y
              (handle-effect {effect: random}
                (lit {type: (t-prim {} i64)} 42)
                (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))))
        "#;
        let dag = lower(src);
        let vals_a = eval_tensor(&dag, &UnordMap::new()).unwrap();
        let vals_b = eval_tensor(&dag, &UnordMap::new()).unwrap();
        let out_a = vals_a.get(dag.roots().last().expect("DAG root")).unwrap();
        let out_b = vals_b.get(dag.roots().last().expect("DAG root")).unwrap();
        assert_eq!(out_a, out_b);
        assert!(out_a.to_f64_lossy_vec().contains(&0.0));
        assert!(out_a.to_f64_lossy_vec().iter().any(|value| *value > 0.0));
    }

    #[test]
    fn lowered_dropout_changes_with_different_seed() {
        let src_a = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 32) (t-prim {} f32))} 1.0))
            (def {} y
              (handle-effect {effect: random}
                (lit {type: (t-prim {} i64)} 42)
                (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))))
        "#;
        let src_b = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 32) (t-prim {} f32))} 1.0))
            (def {} y
              (handle-effect {effect: random}
                (lit {type: (t-prim {} i64)} 43)
                (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))))
        "#;
        let dag_a = lower(src_a);
        let dag_b = lower(src_b);
        let out_a = eval_tensor(&dag_a, &UnordMap::new())
            .unwrap()
            .remove(&NodeId(dag_a.len() - 1))
            .unwrap();
        let out_b = eval_tensor(&dag_b, &UnordMap::new())
            .unwrap()
            .remove(&NodeId(dag_b.len() - 1))
            .unwrap();
        assert_ne!(out_a, out_b);
    }

    #[test]
    fn uniform_like_f32_affine_mirrors_c_f32_sampler() {
        // chelis#770: the affine is a single correctly-rounded FMA
        // (`span_f.mul_add(unit_f, low_f)`), conforming to the compiled C
        // sampler `chelis_uniform_sample_f32` (which the default toolchain
        // contracts to the same FMA). seed=42, shape=[8], [2,5).
        let out = uniform_like(&[8], 2.0, 5.0, 42, Prim::F32).unwrap();
        // elem[4]: where the OLD f64 affine diverged from the C f32 sampler by
        // 1 ULP (the #735 sweep: eval 0x404215a9 vs C 0x404215aa).
        assert_eq!(
            out.to_f64_lossy_vec()[4].to_bits(),
            (f32::from_bits(0x404215aa) as f64).to_bits(),
            "elem[4] must be the C f32 sampler value (0x404215aa)",
        );
        let old_f64_affine = 2.0 + (5.0 - 2.0) * dropout_sample(42, 4);
        assert_eq!((old_f64_affine as f32).to_bits(), 0x404215a9);
        assert_ne!(
            (out.to_f64_lossy_vec()[4] as f32).to_bits(),
            (old_f64_affine as f32).to_bits()
        );
        // elem[6]/[7]: where a single-rounding FMA and a plain two-rounding
        // `low_f + span_f * unit_f` disagree by 1 ULP. Pin the FMA values and
        // show the two-rounding form does NOT reproduce elem[6] — this is the
        // exact bit the compiled C lane flips between `-ffp-contract=fast`
        // (FMA, 0x408f5273) and `-ffp-contract=off` (two roundings, 0x408f5274).
        assert_eq!(
            out.to_f64_lossy_vec()[6].to_bits(),
            (f32::from_bits(0x408f5273) as f64).to_bits(),
            "elem[6] must be the single-rounding FMA value (0x408f5273)",
        );
        assert_eq!(
            out.to_f64_lossy_vec()[7].to_bits(),
            (f32::from_bits(0x403ec1e7) as f64).to_bits(),
            "elem[7] must be the single-rounding FMA value (0x403ec1e7)",
        );
        let unit6 = dropout_sample(42, 6) as f32;
        let two_rounding_6 = 2.0f32 + (5.0f32 - 2.0f32) * unit6;
        assert_eq!(two_rounding_6.to_bits(), 0x408f5274);
        assert_ne!(
            (out.to_f64_lossy_vec()[6] as f32).to_bits(),
            two_rounding_6.to_bits()
        );
    }

    #[test]
    fn uniform_like_f32_affine_negative_range_is_f32() {
        // chelis#770: negative range at unit level (the C cross-lane path
        // can't be driven with a bare negative literal — a separate lowering
        // gap). seed=42, index=3, low=-3.0, high=-1.0 → 0xc010167a.
        let out = uniform_like(&[8], -3.0, -1.0, 42, Prim::F32).unwrap();
        assert_eq!(
            out.to_f64_lossy_vec()[3].to_bits(),
            (f32::from_bits(0xc010167a) as f64).to_bits(),
        );
    }

    #[test]
    fn uniform_like_f64_uses_the_f64_affine() {
        let out = uniform_like(&[8], 2.0, 5.0, 42, Prim::F64).unwrap();
        let expected = (5.0f64 - 2.0).mul_add(dropout_sample(42, 4), 2.0);
        assert_eq!(out.to_f64_lossy_vec()[4].to_bits(), expected.to_bits());
        assert_ne!(
            out.to_f64_lossy_vec()[4].to_bits(),
            (f32::from_bits(0x404215aa) as f64).to_bits(),
        );
    }

    #[test]
    fn cross_family_equality_handles_signed_min_and_actual_float_width() {
        let int_min = TensorValue::from_storage(
            vec![],
            finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![i64::MIN])).unwrap(),
        );
        let float_min = TensorValue::from_storage(
            vec![],
            finalize_tensor("test", Prim::F64, RawTensor::Float(vec![i64::MIN as f64])).unwrap(),
        );
        assert_eq!(int_min, float_min, "i64::MIN is an exact power of two");

        let int_max = TensorValue::from_storage(
            vec![],
            finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![i64::MAX])).unwrap(),
        );
        let rounded_float_max = TensorValue::from_storage(
            vec![],
            finalize_tensor("test", Prim::F64, RawTensor::Float(vec![i64::MAX as f64])).unwrap(),
        );
        assert_ne!(
            int_max, rounded_float_max,
            "a lossy float fixture must not mask i64::MAX"
        );

        let exact_f32 = TensorValue::from_storage(
            vec![],
            finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![1 << 24])).unwrap(),
        );
        let float_f32 = TensorValue::from_storage(
            vec![],
            finalize_tensor("test", Prim::F32, RawTensor::Float(vec![(1 << 24) as f64])).unwrap(),
        );
        assert_eq!(exact_f32, float_f32);

        let inexact_f32 = TensorValue::from_storage(
            vec![],
            finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![(1 << 24) + 1])).unwrap(),
        );
        assert_ne!(inexact_f32, float_f32);
    }

    #[test]
    fn issue_878_pad_fill_preserves_exact_int64_above_f64_boundary() {
        let exact = 9_007_199_254_740_993i64;
        let input = TensorValue::from_storage(
            vec![1],
            finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![7])).unwrap(),
        );
        let fill = chelis_types::scalar_from_i64("pad", Prim::Int64, exact).unwrap();
        let out = pad(&input, &[(1, 1)], fill).unwrap();
        assert_eq!(
            out.storage().to_i64_exact_vec(),
            Some(vec![exact, 7, exact])
        );
    }

    // ---- Phase 3j-pre: new reduction ops ----

    fn mat_f32(rows: usize, cols: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
            precision: Prim::F32,
        }
    }

    fn row_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    /// 2x3 tensor:
    ///   [ 1.0,  4.0, -2.0]
    ///   [ 3.0, -1.0,  5.0]
    fn build_2x3_with(op: RiscOp) -> (Dag, NodeId) {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            mat_f32(2, 3),
            None,
        );
        let out_ty = match &op {
            RiscOp::MinReduce { axis }
            | RiscOp::ProdReduce { axis }
            | RiscOp::Argmax { axis }
            | RiscOp::Argmin { axis } => {
                if *axis == 0 {
                    row_f32(3)
                } else {
                    row_f32(2)
                }
            }
            _ => panic!("unexpected op"),
        };
        let y = dag.add_node(op, vec![x], out_ty, None);
        (dag, y)
    }

    fn inputs_2x3() -> UnordMap<String, TensorValue> {
        let mut inputs = UnordMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![2, 3], vec![1.0, 4.0, -2.0, 3.0, -1.0, 5.0]),
        );
        inputs
    }

    #[test]
    fn adv_min_reduce_axis0() {
        let (dag, y) = build_2x3_with(RiscOp::MinReduce { axis: 0 });
        let vals = eval_tensor(&dag, &inputs_2x3()).unwrap();
        // column-wise min: min(1,3)=1, min(4,-1)=-1, min(-2,5)=-2
        assert_eq!(
            vals[&y],
            TensorValue::from_vec(vec![3], vec![1.0, -1.0, -2.0])
        );
    }

    #[test]
    fn adv_min_reduce_axis1() {
        let (dag, y) = build_2x3_with(RiscOp::MinReduce { axis: 1 });
        let vals = eval_tensor(&dag, &inputs_2x3()).unwrap();
        assert_eq!(vals[&y], TensorValue::from_vec(vec![2], vec![-2.0, -1.0]));
    }

    #[test]
    fn adv_prod_reduce_axis0() {
        let (dag, y) = build_2x3_with(RiscOp::ProdReduce { axis: 0 });
        let vals = eval_tensor(&dag, &inputs_2x3()).unwrap();
        // column-wise product: 1*3=3, 4*-1=-4, -2*5=-10
        assert_eq!(
            vals[&y],
            TensorValue::from_vec(vec![3], vec![3.0, -4.0, -10.0])
        );
    }

    #[test]
    fn adv_prod_reduce_axis1() {
        let (dag, y) = build_2x3_with(RiscOp::ProdReduce { axis: 1 });
        let vals = eval_tensor(&dag, &inputs_2x3()).unwrap();
        // row-wise product: 1*4*-2=-8, 3*-1*5=-15
        assert_eq!(vals[&y], TensorValue::from_vec(vec![2], vec![-8.0, -15.0]));
    }

    #[test]
    fn adv_argmax_axis0() {
        let (dag, y) = build_2x3_with(RiscOp::Argmax { axis: 0 });
        let vals = eval_tensor(&dag, &inputs_2x3()).unwrap();
        // column-wise argmax: max(1,3)@1, max(4,-1)@0, max(-2,5)@1
        assert_eq!(
            vals[&y],
            TensorValue::from_vec(vec![3], vec![1.0, 0.0, 1.0])
        );
    }

    #[test]
    fn adv_argmax_axis1() {
        let (dag, y) = build_2x3_with(RiscOp::Argmax { axis: 1 });
        let vals = eval_tensor(&dag, &inputs_2x3()).unwrap();
        // row 0: max at col 1 (4.0); row 1: max at col 2 (5.0)
        assert_eq!(vals[&y], TensorValue::from_vec(vec![2], vec![1.0, 2.0]));
    }

    #[test]
    fn adv_argmin_axis0() {
        let (dag, y) = build_2x3_with(RiscOp::Argmin { axis: 0 });
        let vals = eval_tensor(&dag, &inputs_2x3()).unwrap();
        // col-wise argmin: min(1,3)@0, min(4,-1)@1, min(-2,5)@0
        assert_eq!(
            vals[&y],
            TensorValue::from_vec(vec![3], vec![0.0, 1.0, 0.0])
        );
    }

    #[test]
    fn adv_argmin_axis1() {
        let (dag, y) = build_2x3_with(RiscOp::Argmin { axis: 1 });
        let vals = eval_tensor(&dag, &inputs_2x3()).unwrap();
        // row 0 min at col 2 (-2.0); row 1 min at col 1 (-1.0)
        assert_eq!(vals[&y], TensorValue::from_vec(vec![2], vec![2.0, 1.0]));
    }

    /// Pins the IR-level argmax/argmin storage: the precision-erased
    /// TensorValue stores integer indices as integer-valued f64. The
    /// host-runtime adapter widens the surrounding precision tag to
    /// `Prim::Int64` per chelis#233. See `RiscOp::Argmax` doc comment.
    #[test]
    fn adv_argmax_output_stores_integer_valued_floats() {
        let (dag, y) = build_2x3_with(RiscOp::Argmax { axis: 1 });
        let vals = eval_tensor(&dag, &inputs_2x3()).unwrap();
        let v = &vals[&y];
        for x in &v.to_f64_lossy_vec() {
            assert!(
                x.fract() == 0.0,
                "argmax output {x} should be integer-valued"
            );
        }
    }

    /// Negative test: reduction verify rejects axes that are out of range
    /// for the new reduction variants. Mirrors the existing c3_sum check.
    #[test]
    fn adv_new_reductions_reject_out_of_range_axis() {
        use crate::verify::verify;
        for op in [
            RiscOp::MinReduce { axis: 7 },
            RiscOp::ProdReduce { axis: 7 },
            RiscOp::Argmax { axis: 7 },
            RiscOp::Argmin { axis: 7 },
        ] {
            let mut dag = Dag::new();
            let x = dag.add_node(
                RiscOp::synth_const(row_f32(3).precision, 1.0),
                vec![],
                row_f32(3),
                None,
            );
            dag.add_node(op, vec![x], scalar_f32(), None);
            let errs = verify(&dag);
            assert!(
                errs.iter().any(|e| e.contains("axis 7")),
                "expected axis-out-of-range error, got: {errs:?}"
            );
        }
    }

    // ---- chelis#550: integer floor_div / trunc_div zero-divisor trap ----
    //
    // The chelis-ir evaluator is a backend-agreement numeric reference (all
    // values are f64). #178 added `floor_div` / `trunc_div` but computed
    // `(a/b).floor()` / `.trunc()` with no zero-divisor trap, so an integer
    // divide by zero silently produced `floor(x/0) == ±inf` rather than
    // halting. Spec §2.1 scopes the `integer division or remainder by zero`
    // trap to BOTH the evaluator and the C backend; these pin that this lane
    // now fails closed on integer operands, while float `floor_div` keeps the
    // IEEE no-trap semantics §2.1 also mandates.

    fn int_div_dag(op: RiscOp, precision: Prim) -> Dag {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            tensor_ty(&[2], precision),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            tensor_ty(&[2], precision),
            None,
        );
        let out = dag.add_node(op, vec![a, b], tensor_ty(&[2], precision), None);
        dag.add_root(out);
        dag
    }

    fn divisor_inputs(b: Vec<f64>) -> UnordMap<String, TensorValue> {
        let mut inputs = UnordMap::new();
        inputs.insert("a".into(), TensorValue::from_vec(vec![2], vec![10.0, 7.0]));
        inputs.insert("b".into(), TensorValue::from_vec(vec![2], b));
        inputs
    }

    #[test]
    fn checked_remainder_preserves_signed_values_and_traps() {
        for prim in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
            let minimum = match prim {
                Prim::Int8 => i8::MIN as i64,
                Prim::Int16 => i16::MIN as i64,
                Prim::Int32 => i32::MIN as i64,
                Prim::Int64 => i64::MIN,
                _ => unreachable!(),
            };
            let dag = int_div_dag(RiscOp::Mod, prim);
            let mut inputs = UnordMap::new();
            inputs.insert("a".into(), TensorValue::from_vec(vec![2], vec![-7.0, 7.0]));
            inputs.insert("b".into(), TensorValue::from_vec(vec![2], vec![2.0, -2.0]));
            let values = eval_tensor(&dag, &inputs).unwrap();
            assert_eq!(values[&dag.roots()[0]].to_f64_lossy_vec(), vec![-1.0, 1.0]);
            let exact = |values: &[i64]| {
                TensorValue::from_storage(
                    vec![2],
                    tensor_from_scalars(
                        prim,
                        &values
                            .iter()
                            .map(|&n| chelis_types::scalar_from_i64("mod", prim, n).unwrap())
                            .collect::<Vec<_>>(),
                    ),
                )
            };
            inputs.insert("a".into(), exact(&[minimum, minimum + 1]));
            inputs.insert("b".into(), exact(&[-1, 2]));
            let values = eval_tensor(&dag, &inputs).unwrap();
            assert_eq!(values[&dag.roots()[0]].to_f64_lossy_vec(), vec![0.0, -1.0]);
            let error = eval_tensor(&dag, &divisor_inputs(vec![0.0, 2.0])).unwrap_err();
            assert_eq!(
                error,
                format!("numeric trap: division by zero in mod at {}", prim.name())
            );
        }
    }

    #[test]
    fn floor_div_integer_traps_on_zero_divisor() {
        let dag = int_div_dag(RiscOp::FloorDiv, Prim::Int32);
        let err = eval_tensor(&dag, &divisor_inputs(vec![2.0, 0.0]))
            .expect_err("integer floor_div by zero must trap");
        assert_eq!(
            err, "numeric trap: division by zero in floor_div at i32",
            "trap diagnostic must match the shared message exactly"
        );
    }

    #[test]
    fn trunc_div_integer_traps_on_zero_divisor() {
        let dag = int_div_dag(RiscOp::TruncDiv, Prim::Int64);
        let err = eval_tensor(&dag, &divisor_inputs(vec![0.0, 2.0]))
            .expect_err("integer trunc_div by zero must trap");
        assert_eq!(err, "numeric trap: division by zero in trunc_div at i64");
    }

    #[test]
    fn floor_div_integer_nonzero_divisor_is_floored_quotient() {
        // floor_div rounds toward -inf: floor_div(10, 4) == 2,
        // floor_div(7, -2) == -4 (7 / -2 == -3.5 -> floor -4).
        let dag = int_div_dag(RiscOp::FloorDiv, Prim::Int32);
        let mut inputs = UnordMap::new();
        inputs.insert("a".into(), TensorValue::from_vec(vec![2], vec![10.0, 7.0]));
        inputs.insert("b".into(), TensorValue::from_vec(vec![2], vec![4.0, -2.0]));
        let vals = eval_tensor(&dag, &inputs).expect("non-zero divisor must not trap");
        let root = dag.roots()[0];
        assert_eq!(vals[&root].to_f64_lossy_vec(), vec![2.0, -4.0]);
    }

    #[test]
    fn trunc_div_integer_nonzero_divisor_is_truncated_quotient() {
        // trunc_div rounds toward zero: trunc_div(10, 4) == 2,
        // trunc_div(7, -2) == -3 (7 / -2 == -3.5 -> trunc -3).
        let dag = int_div_dag(RiscOp::TruncDiv, Prim::Int32);
        let mut inputs = UnordMap::new();
        inputs.insert("a".into(), TensorValue::from_vec(vec![2], vec![10.0, 7.0]));
        inputs.insert("b".into(), TensorValue::from_vec(vec![2], vec![4.0, -2.0]));
        let vals = eval_tensor(&dag, &inputs).expect("non-zero divisor must not trap");
        let root = dag.roots()[0];
        assert_eq!(vals[&root].to_f64_lossy_vec(), vec![2.0, -3.0]);
    }

    #[test]
    fn floor_div_float_does_not_trap_on_zero_divisor() {
        // Spec §2.1: float `floor_div` follows IEEE division and is NOT
        // guarded; `floor(x / 0.0)` yields ±inf (or NaN for 0/0), never a
        // trap. This pins that the integer trap does not bleed into floats.
        let dag = int_div_dag(RiscOp::FloorDiv, Prim::F32);
        let vals = eval_tensor(&dag, &divisor_inputs(vec![0.0, 2.0]))
            .expect("float floor_div by zero must NOT trap");
        let root = dag.roots()[0];
        assert!(
            vals[&root].to_f64_lossy_vec()[0].is_infinite()
                && vals[&root].to_f64_lossy_vec()[0] > 0.0,
            "float floor_div(10.0, 0.0) must be +inf, got {}",
            vals[&root].to_f64_lossy_vec()[0]
        );
        assert_eq!(
            vals[&root].to_f64_lossy_vec()[1],
            3.0,
            "floor(7.0 / 2.0) == 3.0"
        );
    }

    #[test]
    fn fused_integer_floor_div_traps_on_zero_divisor() {
        // floor_div feeding an add forms a fusible 2-node integer chain.
        // After `fuse`, the divide-by-zero must still trap through the
        // FusedElem path (the trap is gated on the fused output precision).
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            tensor_ty(&[2], Prim::Int32),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            tensor_ty(&[2], Prim::Int32),
            None,
        );
        let d = dag.add_node(
            RiscOp::FloorDiv,
            vec![a, b],
            tensor_ty(&[2], Prim::Int32),
            None,
        );
        let e = dag.add_node(RiscOp::Add, vec![d, a], tensor_ty(&[2], Prim::Int32), None);
        dag.add_root(e);
        let fused = crate::fuse::fuse(&dag);
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, RiscOp::FusedElem { .. })),
            "the floor_div -> add chain must fuse for this test to exercise the fused path"
        );
        let err = eval_tensor(&fused, &divisor_inputs(vec![2.0, 0.0]))
            .expect_err("fused integer floor_div by zero must trap");
        assert_eq!(err, "numeric trap: division by zero in floor_div at i32");
    }

    fn exact_tensor(prim: Prim, values: RawTensor) -> TensorValue {
        let len = match &values {
            RawTensor::Int(values) => values.len(),
            RawTensor::Float(values) => values.len(),
        };
        let storage = finalize_tensor("phase2-reduction-test", prim, values)
            .expect("test values are in range");
        TensorValue::from_storage(vec![len], storage)
    }

    #[test]
    fn checked_cast_selects_the_lowest_flat_index_across_trap_kinds() {
        let overflow_first = exact_tensor(Prim::F32, RawTensor::Float(vec![300.0, f64::NAN]));
        assert_eq!(
            cast_tensor(&overflow_first, Prim::F32, Prim::Int8).unwrap_err(),
            "numeric trap: overflow in cast at i8"
        );

        let domain_first = exact_tensor(Prim::F32, RawTensor::Float(vec![f64::NAN, 300.0]));
        assert_eq!(
            cast_tensor(&domain_first, Prim::F32, Prim::Int8).unwrap_err(),
            "numeric trap: domain in cast at i8"
        );
    }

    #[test]
    fn phase2_window_reduction_uses_declared_width_and_exact_storage() {
        let f32_input = exact_tensor(
            Prim::F32,
            RawTensor::Float(vec![16_777_216.0, 1.0, -16_777_216.0]),
        );
        let f32_out = reduce_window(&f32_input, ReduceWindowKind::Sum, &[3], &[1], Prim::F32)
            .expect("in-range f32 reduction");
        assert_eq!(f32_out.to_f64_lossy_vec(), vec![0.0]);

        let i64_input = exact_tensor(Prim::Int64, RawTensor::Int(vec![9_007_199_254_740_992, 1]));
        let i64_out = reduce_window(&i64_input, ReduceWindowKind::Sum, &[2], &[1], Prim::Int64)
            .expect("exact i64 reduction");
        assert_eq!(
            i64_out.storage().to_i64_exact_vec(),
            Some(vec![9_007_199_254_740_993])
        );
    }

    #[test]
    fn phase2_window_reduction_traps_intermediate_overflow_at_operand_width() {
        let input = exact_tensor(Prim::Int8, RawTensor::Int(vec![100, 100, -100]));
        let err = reduce_window(&input, ReduceWindowKind::Sum, &[3], &[1], Prim::Int8)
            .expect_err("100i8 + 100i8 must trap before the later -100");
        assert_eq!(err, "numeric trap: overflow in reduce_window_sum at i8");

        let control = exact_tensor(Prim::Int8, RawTensor::Int(vec![40, 40, -40]));
        let output = reduce_window(&control, ReduceWindowKind::Sum, &[3], &[1], Prim::Int8)
            .expect("in-range i8 control");
        assert_eq!(output.storage().to_i64_exact_vec(), Some(vec![40]));
    }

    #[test]
    fn phase2_axis_sum_traps_in_the_stride4_combine() {
        let input = exact_tensor(
            Prim::Int32,
            RawTensor::Int(vec![i64::from(i32::MAX), 1, -1]),
        );
        let err = reduce(
            &input,
            0,
            TensorReduceOp::Sum {
                accumulator: Prim::Int32,
                result: Prim::Int32,
            },
        )
        .expect_err("lane0 + lane1 overflows the i32 accumulator");
        assert_eq!(err, "numeric trap: overflow in sum at i32");

        let control = exact_tensor(
            Prim::Int32,
            RawTensor::Int(vec![i64::from(i32::MAX) - 1, 1, -1]),
        );
        let output = reduce(
            &control,
            0,
            TensorReduceOp::Sum {
                accumulator: Prim::Int32,
                result: Prim::Int32,
            },
        )
        .expect("below-overflow control");
        assert_eq!(
            output.storage().to_i64_exact_vec(),
            Some(vec![i64::from(i32::MAX) - 1])
        );
    }

    #[test]
    fn phase2_arg_reductions_compare_int64_without_binary64() {
        let max_input = exact_tensor(
            Prim::Int64,
            RawTensor::Int(vec![9_007_199_254_740_992, 9_007_199_254_740_993]),
        );
        let max_out = reduce_argcmp(&max_input, 0, ArgReduceOp::Argmax).expect("argmax");
        assert_eq!(max_out.storage().to_i64_exact_vec(), Some(vec![1]));

        let min_input = exact_tensor(
            Prim::Int64,
            RawTensor::Int(vec![9_007_199_254_740_993, 9_007_199_254_740_992]),
        );
        let min_out = reduce_argcmp(&min_input, 0, ArgReduceOp::Argmin).expect("argmin");
        assert_eq!(min_out.storage().to_i64_exact_vec(), Some(vec![1]));
    }
}
