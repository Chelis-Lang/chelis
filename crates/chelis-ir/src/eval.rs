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

use chelis_abi::metadata::{MetadataError, ShapeMetadata};
use chelis_unord::{UnordMap, UnordSet};
use std::borrow::Cow;
use std::collections::BTreeSet;

use crate::dag::{
    ComparisonKind, Dag, DagNode, DimExpr, DimInfo, ExtremaKind, ExtremaOperand, FusedInput,
    FusedStepOp, LogicalKind, NamedCastMode, NodeId, ReduceWindowKind, RiscOp, RtAxis, RtDim,
    RuntimeCheck, SHRINK_TO_END, TensorType, bind_symbolic_dims,
};
use chelis_types::dtype_semantics::{
    ArgReduceOp, CheckedCastPlan, CompareOp, ExtremaOperand as KernelExtremaOperand, FloatBinOp,
    FloatExtremaOp, FloatUnOp, IndexedTrapCandidate, IntBinOp, IntUnOp, NumericTrap, RawTensor,
    ReduceWindowGradOp, TensorReduceOp, TensorStorage, arg_reduce_tensor_groups, compare_tensors,
    count_tensor_groups, finalize_tensor, float_extrema_adjoint, float_relu, float_relu_adjoint,
    float_tensor_binop, float_tensor_unop, int_tensor_binop, int_tensor_unop,
    integer_is_exactly_representable, reduce_tensor_groups, reduce_window_grad_tensor_groups,
    scatter_add_tensor_groups, tensor_from_scalars,
};
use chelis_types::dtype_semantics::{
    PreparedDropout, fold_in_storage, key_from_seed_storage, split_key_storage, split_keys_storage,
    uniform_like_bound_adjoint_rows,
};
use chelis_types::types::Prim;
use chelis_types::{PreparedUniformLike, RandomKey, uniform_like_bound_adjoint};

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

/// The C runtime's report when a tensor's storage cannot be allocated
/// (`allocate_tensor` in `crates/chelis-runtime/src/lib.rs`), verbatim.
const TENSOR_ALLOCATION_FAILED: &str = "Domain: chelis_alloc tensor allocation failed";

/// [05-OP-33]'s checked admission of a result before allocation: the one
/// gate every evaluator path that sizes a new tensor from its extents passes
/// through, before it allocates anything proportional to them (chelis#2491).
///
/// The metadata is the C runtime's own authority, `chelis_abi`'s
/// [`ShapeMetadata`], so the checks and their order are the compiled lane's:
/// the element count, the byte size at the result's representation, and the
/// stride products each fit i64, and the byte size fits the allocation
/// domain. A failure is `op`'s [04-NUM-9] trap under the metadata report,
/// the two lines the C runtime's operation plans (`expand`, `pad`, the
/// reductions) print for it. Where the compiled lane instead allocates
/// through bare `chelis_alloc` (`split_keys`, `const`), it prints the same
/// report under that runtime symbol and no trap line; [04-NUM-9] requires the
/// trap to name the operation, so this lane does not copy that rendering.
///
/// An admitted result can still be one this lane cannot hold: the evaluator
/// keeps per-element scratch wider than the representation, at most one
/// index group (`Vec<usize>`) per result element in a reduction. A result
/// whose scratch would not fit Rust's allocation domain is refused with the
/// C runtime's allocation failure, which is what C reports at that size: its
/// request is then at least 2^58 bytes, more than any current 64-bit virtual
/// address space holds.
///
/// Returns the admitted element count.
fn admit_result(op: &'static str, shape: &[usize], prim: Prim) -> Result<usize, String> {
    let dtype = prim.runtime_dtype().map_err(|error| error.to_string())?;
    let metadata = i64_extents(shape)
        .and_then(|extents| ShapeMetadata::contiguous(&extents, dtype))
        .and_then(|metadata| metadata.bytes().allocation().map(|_| metadata))
        .map_err(|error| admission_trap(op, &error))?;
    metadata
        .elements()
        .scratch_len::<Vec<usize>>()
        .map_err(|_| TENSOR_ALLOCATION_FAILED.to_string())
}

/// Host extents as the metadata's i64 extents.
fn i64_extents(extents: &[usize]) -> Result<Vec<i64>, MetadataError> {
    extents
        .iter()
        .map(|&extent| {
            i64::try_from(extent).map_err(|_| MetadataError::Overflow("extent exceeds i64"))
        })
        .collect()
}

/// A metadata failure in `op`, as the C runtime's `affine_result` prints it:
/// the metadata report, then [04-NUM-9]'s trap line at i64.
fn admission_trap(op: &'static str, error: &MetadataError) -> String {
    let trap = match error {
        MetadataError::Domain(_) => NumericTrap::Domain {
            op,
            prim: Prim::Int64,
        },
        MetadataError::Overflow(_) => NumericTrap::Overflow {
            op,
            prim: Prim::Int64,
        },
    };
    format!("{error}\n{trap}")
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
/// Load inputs). Zero is a member of every active dtype, so no value traps;
/// only the declared extents can fail admission.
fn default_value(ty: &TensorType) -> Result<TensorValue, String> {
    let shape = concrete_shape(ty).unwrap_or_default();
    let n = admit_result("load", &shape, ty.precision)?;
    let storage = if ty.precision.is_float() {
        finalize_tensor("load", ty.precision, RawTensor::Float(vec![0.0; n]))
    } else {
        finalize_tensor("load", ty.precision, RawTensor::Int(vec![0; n]))
    }
    .expect("zero is a member of every active dtype");
    Ok(TensorValue::from_storage(shape, storage))
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
    if value.raw_f64_ingress && declared == Prim::Key {
        return Err(format!(
            "input `{name}` declares key, which has no raw numeric ingress; provide key storage"
        ));
    }
    if value.raw_f64_ingress {
        let storage = finalize_tensor("load", declared, value.storage().to_raw())
            .map_err(|trap| format!("input `{name}`\n{trap}"))?;
        return Ok(TensorValue::from_storage(value.shape.clone(), storage));
    }
    if value.prim() == declared {
        return Ok(value.clone());
    }
    Err(format!(
        "input `{name}` carries dtype {} but the Load declares {}; provide a value with the declared dtype (casts are explicit in Chelis)\n{}",
        value.prim().name(),
        declared.name(),
        NumericTrap::Domain {
            op: "load",
            prim: declared
        }
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

/// Public entry point for the named cast rungs of the chelis#759 ladder,
/// mirroring [`cast_tensor`] so the host runtime and the DAG evaluator
/// share one kernel per rung.
pub fn named_cast_tensor(
    mode: NamedCastMode,
    input: &TensorValue,
    dst: Prim,
) -> Result<TensorValue, String> {
    named_cast_value(mode, input, dst)
}

/// Tensor named cast: the rung's sealed kernel over every element, in
/// order. The checker admits only the rung's source and target pairs, so
/// the kernel is reached with exactly the shapes it accepts.
fn named_cast_value(
    mode: NamedCastMode,
    input: &TensorValue,
    dst: Prim,
) -> Result<TensorValue, String> {
    let storage = chelis_types::named_cast_tensor(mode, input.storage().to_raw(), dst)
        .map_err(|trap| trap.to_string())?;
    Ok(TensorValue::from_storage(input.shape.clone(), storage))
}

/// Read a rank-0 value's one scalar.
fn rank0_scalar(value: &TensorValue, what: &str) -> Result<chelis_types::ScalarValue, String> {
    if !value.shape.is_empty() || value.len() != 1 {
        return Err(format!("{what} is not a rank-0 scalar"));
    }
    Ok(value.storage().scalar_at(0))
}

fn rank0_bool(value: &TensorValue, what: &str) -> Result<bool, String> {
    match rank0_scalar(value, what)?.as_i64_exact() {
        Some(bits) if value.prim() == Prim::Bool => Ok(bits != 0),
        _ => Err(format!("{what} is not a rank-0 Bool")),
    }
}

/// Where a node checks (`spec/10-serialization.md` §3.2): a node whose
/// activation is false is still computed, since a `Where` may read its value,
/// but checks nothing.
enum Activity {
    /// The node has no activation, or its activation holds in every row.
    All,
    /// The activation holds in no row: the node checks nothing.
    Inactive,
    /// A per-row activation (a `vmap`ped `if`): row `r` of the activation's
    /// shape checks exactly when `rows[r]`.
    Rows { shape: Vec<usize>, rows: Vec<bool> },
}

/// The run-time [`Activity`] of `node`, read from its owner's activation.
fn node_activity(
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
) -> Result<Activity, String> {
    let Some(activation) = node.owner.activation else {
        return Ok(Activity::All);
    };
    let value = values.get(&activation).ok_or_else(|| {
        format!(
            "node {}'s activation {} is not available",
            node.id.0, activation.0
        )
    })?;
    if value.prim() != Prim::Bool {
        return Err(format!(
            "node {}'s activation {} is not a Bool",
            node.id.0, activation.0
        ));
    }
    let rows = value
        .storage()
        .to_i64_exact_vec()
        .expect("sealed bool storage has an exact integer view")
        .into_iter()
        .map(|bit| bit != 0)
        .collect::<Vec<_>>();
    Ok(if rows.iter().all(|row| *row) {
        Activity::All
    } else if !rows.iter().any(|row| *row) {
        Activity::Inactive
    } else {
        Activity::Rows {
            shape: value.shape.clone(),
            rows,
        }
    })
}

/// `value` with the elements of the rows its node checks nothing in
/// replaced by `neutral`; `value`'s leading axes are the activation's `shape`.
fn neutral_rows(value: &TensorValue, rows: &[bool], neutral: i64) -> Result<TensorValue, String> {
    let len = value.len();
    let per_row = if rows.is_empty() { 0 } else { len / rows.len() };
    let mask = (0..len)
        .map(|index| i64::from(rows[index / per_row.max(1)]))
        .collect::<Vec<_>>();
    let mask = finalize_wide_int("activation", Prim::Bool, value.shape.clone(), mask)?;
    let fill = finalize_wide(
        "activation",
        value.prim(),
        value.shape.clone(),
        vec![neutral as f64; len],
    )?;
    where_elementwise(&mask, value, &fill)
}

/// Replace the operands of a gated `node` ([`crate::dag::TrapSeeds::is_activation_gated`],
/// `gated`) whose activation is false, in the rows where it is false, with
/// values its checks accept ([`DagNode::inactive_operand`]), so the node
/// computes a value and reports nothing.
/// Returns the replaced values, which the caller restores once the node has
/// run: its operands' other consumers read them unchanged.
fn neutralize_inactive_operands(
    node: &DagNode,
    gated: bool,
    values: &mut UnordMap<NodeId, TensorValue>,
) -> Result<Vec<(NodeId, TensorValue)>, String> {
    if !gated || node.inactive_operand(0).is_none() {
        return Ok(Vec::new());
    }
    let activity = node_activity(node, values)?;
    if matches!(activity, Activity::All) {
        return Ok(Vec::new());
    }
    // One operand may fill several slots (`x / x`); the larger neutral
    // (one) is accepted in every slot. A slot with no inactive value (a
    // guarded abort's fallback) is read unchanged.
    let mut neutrals = Vec::<(NodeId, i64)>::new();
    for (slot, input) in node.inputs.iter().enumerate() {
        let Some(neutral) = node.inactive_operand(slot) else {
            continue;
        };
        match neutrals.iter_mut().find(|(id, _)| id == input) {
            Some((_, existing)) => *existing = (*existing).max(neutral),
            None => neutrals.push((*input, neutral)),
        }
    }
    // Under a per-row activation, a rank-0 operand is the same scalar in
    // every row; it takes each row's shape so each row decides on its own.
    let carrier = match &activity {
        Activity::Rows { shape, .. } => neutrals
            .iter()
            .filter_map(|(id, _)| values.get(id))
            .find(|value| !value.shape.is_empty() && value.shape.starts_with(shape))
            .map(|value| value.shape.clone()),
        _ => None,
    };
    let mut replaced = Vec::new();
    for (id, neutral) in neutrals {
        let original = values
            .get(&id)
            .ok_or_else(|| format!("node {} reads missing operand {}", node.id.0, id.0))?;
        let neutralized = match &activity {
            Activity::All => continue,
            Activity::Inactive => finalize_wide(
                "activation",
                original.prim(),
                original.shape.clone(),
                vec![neutral as f64; original.len()],
            )?,
            Activity::Rows { shape, rows } => {
                if !original.shape.is_empty() && original.shape.starts_with(shape) {
                    neutral_rows(original, rows, neutral)?
                } else if let Some(carrier) = carrier.as_ref().filter(|_| original.shape.is_empty())
                {
                    neutral_rows(&splat_rank0(original, carrier), rows, neutral)?
                } else {
                    // An operand no row indexes: some row checks it.
                    continue;
                }
            }
        };
        let original = values.insert(id, neutralized).expect("operand present");
        replaced.push((id, original));
    }
    Ok(replaced)
}

/// The keys a key-operand random primitive draws with
/// (`spec/10-serialization.md` §3.2).
enum KeyOperand<'a> {
    /// A rank-0 activation is false: the primitive validates and draws
    /// nothing and produces positive zeros.
    Inactive,
    /// A rank-0 key.
    Scalar(RandomKey),
    /// A key batch of `shape`, any positive rank: row `b` of the data, the
    /// elements whose leading indices are the key index `b` in row-major
    /// order, draws with `keys[b]`, and only when `active` is absent or its
    /// element for row `b` is true.
    Rows {
        keys: &'a [RandomKey],
        shape: &'a [usize],
        active: Option<&'a TensorValue>,
    },
}

impl KeyOperand<'_> {
    fn row_active(&self, row: usize) -> bool {
        match self {
            Self::Rows {
                keys,
                active: Some(active),
                ..
            } => {
                let element = leading_row(row, keys.len(), active.len());
                active.storage().scalar_at(element).as_bool_exact() == Some(true)
            }
            _ => true,
        }
    }
}

/// The element that row `row` of `rows` reads from an operand of `len`
/// elements shaped like the keys' leading axes: the row's index over those
/// axes. A nonempty key batch has a nonempty leading part, so `len > 0`
/// whenever a row exists.
fn leading_row(row: usize, rows: usize, len: usize) -> usize {
    row / (rows / len)
}

/// A key-operand random primitive's keys, under the node's own activation
/// ([`crate::dag::Owner::activation`]). A key that has no value under an
/// active primitive is a malformed graph.
fn draw_keys<'a>(
    node: &DagNode,
    key_slot: usize,
    values: &'a UnordMap<NodeId, TensorValue>,
) -> Result<KeyOperand<'a>, String> {
    let activation = match node.owner.activation {
        Some(activation) => Some(
            values
                .get(&activation)
                .ok_or("random primitive activation is not available")?,
        ),
        None => None,
    };
    if let Some(activation) = activation
        && activation.shape.is_empty()
        && !rank0_bool(activation, "random primitive activation")?
    {
        return Ok(KeyOperand::Inactive);
    }
    let key = node
        .inputs
        .get(key_slot)
        .and_then(|key| values.get(key))
        .ok_or_else(|| {
            format!(
                "random primitive at node {} is active but its key's draw was not",
                node.id.0
            )
        })?;
    let keys = key
        .storage()
        .keys()
        .ok_or_else(|| format!("random primitive at node {} has a non-key key", node.id.0))?;
    if key.shape.is_empty() {
        return Ok(KeyOperand::Scalar(keys[0]));
    }
    let active = activation.filter(|activation| !activation.shape.is_empty());
    if active.is_some_and(|active| !key.shape.starts_with(&active.shape)) {
        return Err(format!(
            "random primitive at node {}: a batched activation must be shaped like a leading part of its keys' shape {:?}",
            node.id.0, key.shape
        ));
    }
    Ok(KeyOperand::Rows {
        keys,
        shape: &key.shape,
        active,
    })
}

/// Row `row` of a batched draw's control: the one scalar of a rank-0
/// control, or the element a control shaped like the keys' leading axes
/// holds for the row.
fn row_control(
    value: &TensorValue,
    row: usize,
    shape: &[usize],
    what: &str,
) -> Result<chelis_types::ScalarValue, String> {
    if value.shape.is_empty() {
        return rank0_scalar(value, what);
    }
    if !shape.starts_with(&value.shape) {
        return Err(format!(
            "{what} must be rank 0 or shaped like a leading part of its keys' shape {shape:?}"
        ));
    }
    Ok(value
        .storage()
        .scalar_at(leading_row(row, numel(shape), value.len())))
}

/// A key-operand random primitive's key batch against its operands, checked
/// before it reads one, in [`RiscOp::draw_batch_layout`]'s order and with
/// the C lane's report (spec/10 §3.2, rule V5): a key batch's shape is its
/// data's leading axes, and each per-row control, then the node's own
/// activation, is a leading part of that shape. The verifier relates the declared dims; this
/// relates the values, so no row index rests on an extent nothing has
/// checked. A key an inactive draw key withheld is rank 0, so it batches
/// nothing. This lane builds each result from its data's shape, so it reads
/// no declared result extent.
fn check_draw_extents(
    dag: &Dag,
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
) -> Result<(), String> {
    let layout = node
        .op
        .draw_batch_layout()
        .ok_or("check_draw_extents reads only a key-operand random primitive")?;
    let op = layout.op;
    let value = |slot: usize| node.inputs.get(slot).and_then(|input| values.get(input));
    let Some(key) = value(layout.key).filter(|key| !key.shape.is_empty()) else {
        return Ok(());
    };
    let key_dims = &dag
        .get(node.inputs[layout.key])
        .ok_or("random primitive key is not in its graph")?
        .output_type
        .dims;
    let data = value(layout.data_input).ok_or("random primitive data is not available")?;
    let operands = std::iter::once((layout.data_input, data, key.shape.len())).chain(
        layout
            .per_row
            .iter()
            .filter_map(|slot| value(*slot).map(|operand| (*slot, operand, operand.shape.len()))),
    );
    for (slot, operand, axes) in operands {
        if axes > key.shape.len() || operand.shape.len() < axes {
            return Err(format!(
                "{op} input {slot} has rank {}, which its rank-{} key batch does not index",
                operand.shape.len(),
                key.shape.len()
            ));
        }
        check_operand_extents(op, key_dims, key, &format!("input {slot}"), operand, axes)?;
    }
    if let Some(activation) = node
        .owner
        .activation
        .and_then(|activation| values.get(&activation))
    {
        let axes = activation.shape.len();
        if axes > key.shape.len() {
            return Err(format!(
                "{op} activation has rank {axes}, which its rank-{} key batch does not index",
                key.shape.len()
            ));
        }
        check_operand_extents(op, key_dims, key, "activation", activation, axes)?;
    }
    Ok(())
}

/// `operand`, which the report calls `what` (its input slot, `input 1`, or
/// the node's `activation`), against `reference`, whose declared axes are
/// `reference_dims`, on its first `axes` extents: the first that disagrees
/// reports the reference's claim and traps `Domain` in `op` at i64, the C
/// lane's operand extent guard's report.
fn check_operand_extents(
    op: &'static str,
    reference_dims: &[DimInfo],
    reference: &TensorValue,
    what: &str,
    operand: &TensorValue,
    axes: usize,
) -> Result<(), String> {
    for axis in 0..axes {
        let (claimed, observed) = (reference.shape[axis], operand.shape[axis]);
        if claimed != observed {
            let claim = match reference_dims.get(axis) {
                Some(DimInfo::Lit(value)) => value.to_string(),
                Some(DimInfo::Named(name, _)) => name.clone(),
                None => claimed.to_string(),
            };
            return Err(format!(
                "extent `{claim}`: claimed = {claimed}, {op} {what} axis {axis} = {observed}\n\
                 {}",
                NumericTrap::Domain {
                    op,
                    prim: Prim::Int64
                }
            ));
        }
    }
    Ok(())
}

/// The element count of each row of `data` that a batch of keys of `shape`
/// splits it into: the data's leading axes are the keys' shape.
fn batched_row_len(data: &TensorValue, shape: &[usize], node: &DagNode) -> Result<usize, String> {
    if !data.shape.starts_with(shape) {
        return Err(format!(
            "random primitive at node {}: its keys of shape {shape:?} must match its data's leading axes",
            node.id.0
        ));
    }
    // An empty row's trailing extents need not have a representable
    // product; `numel` answers zero without folding them.
    Ok(numel(&data.shape[shape.len()..]))
}

fn row_of(storage: &TensorStorage, row: usize, row_len: usize) -> TensorStorage {
    let indices = (row * row_len..(row + 1) * row_len).collect::<Vec<_>>();
    storage.reuse_gather(&indices)
}

/// Positive zeros at `prim`, the value of an inactive draw or row.
fn zero_storage(prim: Prim, len: usize) -> Result<TensorStorage, String> {
    finalize_tensor("random", prim, RawTensor::Float(vec![0.0; len]))
        .map_err(|trap| trap.to_string())
}

/// Concatenate row results into one buffer in row order.
fn concat_rows(prim: Prim, rows: &[TensorStorage]) -> TensorStorage {
    let scalars = rows
        .iter()
        .flat_map(|row| (0..row.len()).map(|index| row.scalar_at(index)))
        .collect::<Vec<_>>();
    tensor_from_scalars(prim, &scalars)
}

/// A draw over `data`'s elements at `prim`, as the stack of its rows' draws
/// (`spec/10-serialization.md` §3.2): a rank-0 key is one row of every
/// element, and a key batch splits the data by its leading axes. `draw`
/// receives each active row's index, key, key shape and element count, and
/// is the only place a row's controls are read and validated. So a batch
/// validates exactly its active rows, and one with no rows validates
/// nothing, as `vmap` over no calls does. An inactive row, and every row
/// under a false rank-0 activation, is positive zeros.
fn stack_draw_rows(
    node: &DagNode,
    data: &TensorValue,
    prim: Prim,
    keys: KeyOperand<'_>,
    mut draw: impl FnMut(usize, RandomKey, &[usize], usize) -> Result<TensorStorage, String>,
) -> Result<TensorStorage, String> {
    match keys {
        KeyOperand::Inactive => zero_storage(prim, data.len()),
        KeyOperand::Scalar(key) => draw(0, key, &[], data.len()),
        KeyOperand::Rows {
            keys: rows,
            shape,
            active,
        } => {
            let row_len = batched_row_len(data, shape, node)?;
            let keys = KeyOperand::Rows {
                keys: rows,
                shape,
                active,
            };
            let mut out = Vec::with_capacity(rows.len());
            for (row, key) in rows.iter().enumerate() {
                out.push(if keys.row_active(row) {
                    draw(row, *key, shape, row_len)?
                } else {
                    zero_storage(prim, row_len)?
                });
            }
            Ok(concat_rows(prim, &out))
        }
    }
}

/// `[05-OP-37]`'s dropout (or its replay, on the cotangent) under `keys`.
fn eval_dropout(
    node: &DagNode,
    data: &TensorValue,
    rate: &TensorValue,
    keys: KeyOperand<'_>,
) -> Result<TensorValue, String> {
    let storage = stack_draw_rows(node, data, data.prim(), keys, |row, key, shape, row_len| {
        let gathered;
        let input = if row_len == data.len() {
            data.storage()
        } else {
            gathered = row_of(data.storage(), row, row_len);
            &gathered
        };
        let rate = row_control(rate, row, shape, "dropout rate")?;
        PreparedDropout::new(input, rate)
            .and_then(|prepared| prepared.apply(key))
            .map_err(|error| error.to_string())
    })?;
    Ok(TensorValue::from_storage(data.shape.clone(), storage))
}

/// `[05-OP-8]`'s sampler for a template of `shape` under `keys`.
fn eval_uniform_like(
    node: &DagNode,
    template: &TensorValue,
    low: &TensorValue,
    high: &TensorValue,
    prim: Prim,
    keys: KeyOperand<'_>,
) -> Result<TensorValue, String> {
    let storage = stack_draw_rows(node, template, prim, keys, |row, key, shape, row_len| {
        let low = row_control(low, row, shape, "uniform_like low bound")?;
        let high = row_control(high, row, shape, "uniform_like high bound")?;
        PreparedUniformLike::new(prim, row_len, low, high)
            .and_then(|prepared| prepared.apply(key))
            .map_err(|error| error.to_string())
    })?;
    Ok(TensorValue::from_storage(template.shape.clone(), storage))
}

/// `[05-OP-8]`'s bound adjoint of the cotangent `g` under `keys`. A batched
/// draw's adjoint has its bound's shape, the keys' leading `c` axes: each
/// element is one balanced tree over the contributions, in row-major order,
/// of the rows that share that bound element. Rank 0 folds every row, and
/// the keys' own shape folds each row alone.
fn eval_uniform_bound_adjoint(
    node: &DagNode,
    g: &TensorValue,
    bound: crate::dag::UniformBound,
    prim: Prim,
    keys: KeyOperand<'_>,
) -> Result<TensorValue, String> {
    let bound = match bound {
        crate::dag::UniformBound::Low => chelis_types::UniformBound::Low,
        crate::dag::UniformBound::High => chelis_types::UniformBound::High,
    };
    match keys {
        KeyOperand::Inactive => {
            zero_tensor("uniform_like", &concrete_shape(&node.output_type)?, prim)
        }
        KeyOperand::Scalar(key) => {
            let value = uniform_like_bound_adjoint(g.storage(), key, bound)
                .map_err(|error| error.to_string())?;
            Ok(TensorValue::from_storage(
                Vec::new(),
                tensor_from_scalars(prim, &[value]),
            ))
        }
        KeyOperand::Rows {
            keys: rows,
            shape,
            active,
        } => {
            let row_len = batched_row_len(g, shape, node)?;
            let keys = KeyOperand::Rows {
                keys: rows,
                shape,
                active,
            };
            let masked = (0..rows.len())
                .map(|row| {
                    if keys.row_active(row) {
                        Ok(row_of(g.storage(), row, row_len))
                    } else {
                        zero_storage(prim, row_len)
                    }
                })
                .collect::<Result<Vec<_>, String>>()?;
            let out_shape = shape.get(..node.output_type.dims.len()).ok_or_else(|| {
                format!(
                    "uniform bound adjoint at node {}: its result is not a leading part of its keys' shape {shape:?}",
                    node.id.0
                )
            })?;
            // Row `b` joins the group of the result element it would read as
            // a control of the result's shape; each group is contiguous.
            let groups_len = admit_result("uniform_like", out_shape, prim)?;
            let mut groups = vec![Vec::new(); groups_len];
            for row in 0..rows.len() {
                groups[leading_row(row, rows.len(), groups_len)].push(row);
            }
            let values = groups
                .iter()
                .map(|group| {
                    let cotangent = group
                        .iter()
                        .map(|row| masked[*row].clone())
                        .collect::<Vec<_>>();
                    let keys = group.iter().map(|row| rows[*row]).collect::<Vec<_>>();
                    uniform_like_bound_adjoint_rows(&concat_rows(prim, &cotangent), &keys, bound)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(TensorValue::from_storage(
                out_shape.to_vec(),
                tensor_from_scalars(prim, &values),
            ))
        }
    }
}

fn key_value(
    value: &TensorValue,
    storage: Result<TensorStorage, chelis_types::NumericKernelError>,
) -> Result<TensorValue, String> {
    let storage = storage.map_err(|error| error.to_string())?;
    Ok(TensorValue::from_storage(value.shape.clone(), storage))
}

fn zero_tensor(op: &'static str, shape: &[usize], prim: Prim) -> Result<TensorValue, String> {
    let len = admit_result(op, shape, prim)?;
    let storage = finalize_tensor("random", prim, RawTensor::Float(vec![0.0; len]))
        .map_err(|trap| trap.to_string())?;
    Ok(TensorValue::from_storage(shape.to_vec(), storage))
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
            Self::Mod => Some(FloatBinOp::Rem),
            Self::TruncDiv => None,
        }
    }
}

/// chelis#664 on the eval lane: an elementwise op indexes every operand
/// through the output's shape, so operands that disagree at run time are a
/// `Domain` trap in `op` (spec/04-type-system.md section 4.7), never an
/// assertion. The routing of host-lane def applications through this
/// evaluator (chelis#1277 B2h) made a runtime disagreement user input; the
/// rendering is the one every lane shares.
/// The failure of `node`'s operand agreement between `lhs` and `rhs`: a
/// `Domain` trap in the node's operation, or in `matmul` when the node is
/// matmul's decomposed product ([`crate::tier2::is_matmul_product`]), with
/// the operands as written.
fn agreement_failure(dag: &Dag, node: &DagNode, lhs: &TensorValue, rhs: &TensorValue) -> String {
    if crate::tier2::is_matmul_product(node, |id| dag.get(id)) {
        let exact = |shape: &[usize]| {
            shape
                .iter()
                .map(|&extent| i64::try_from(extent).unwrap_or(i64::MAX))
                .collect::<Vec<_>>()
        };
        return chelis_abi::failure::matmul_product_disagreement(
            &exact(&lhs.shape),
            &exact(&rhs.shape),
        );
    }
    shape_disagreement(crate::grad::risc_op_name(&node.op), lhs, rhs)
}

/// A scatter's updates disagree at run time with the gathered shape they must
/// take: a section 4.7 `Domain` trap in the scatter, as compiled C renders it.
fn sparse_updates_disagreement(op: &str, expected: &[usize], updates: &TensorValue) -> String {
    let exact = |shape: &[usize]| {
        shape
            .iter()
            .map(|&extent| i64::try_from(extent).unwrap_or(i64::MAX))
            .collect::<Vec<_>>()
    };
    chelis_abi::failure::operand_shape_disagreement(op, &exact(expected), &exact(&updates.shape))
}

fn shape_disagreement(op: &str, lhs: &TensorValue, rhs: &TensorValue) -> String {
    let exact = |shape: &[usize]| {
        shape
            .iter()
            .map(|&extent| i64::try_from(extent).unwrap_or(i64::MAX))
            .collect::<Vec<_>>()
    };
    chelis_abi::failure::operand_shape_disagreement(op, &exact(&lhs.shape), &exact(&rhs.shape))
}

/// Validate a producer's complete positive-rank agreement relation before
/// reading the extent used by its declared-result claim.
fn same_shape_agreement_extent(
    dag: &Dag,
    node: &DagNode,
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
            return Err(agreement_failure(dag, node, first, value));
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
    op: &str,
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
    Err(shape_disagreement(op, lhs, rhs))
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
fn require_matching_comparison_shapes(
    op: &str,
    lhs: &TensorValue,
    rhs: &TensorValue,
) -> Result<(), String> {
    if lhs.shape == rhs.shape {
        return Ok(());
    }
    Err(shape_disagreement(op, lhs, rhs))
}

fn binary_elementwise(
    op: ElementwiseBinOp,
    lhs: &TensorValue,
    rhs: &TensorValue,
) -> Result<TensorValue, String> {
    let (lhs, rhs) = broadcast_rank0_operands(op.name(), lhs, rhs)?;
    let (lhs, rhs) = (&*lhs, &*rhs);
    let storage = if lhs.prim() == Prim::Bool || rhs.prim() == Prim::Bool {
        return Err(format!(
            "{}: bool storage cannot enter a numeric IR kernel",
            op.name()
        ));
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
    Tanh,
    Erf,
    Erfc,
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
            Self::Tanh => "tanh",
            Self::Erf => "erf",
            Self::Erfc => "erfc",
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
            | Self::Atan
            | Self::Tanh
            | Self::Erf
            | Self::Erfc => None,
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
            Self::Tanh => FloatUnOp::Tanh,
            Self::Erf => FloatUnOp::Erf,
            Self::Erfc => FloatUnOp::Erfc,
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
    kind: ComparisonKind,
    lhs: &TensorValue,
    rhs: &TensorValue,
) -> Result<TensorValue, String> {
    require_matching_comparison_shapes(kind.surf_name(), lhs, rhs)?;
    let op = comparison_kernel(kind);
    let storage =
        compare_tensors(op, lhs.storage(), rhs.storage()).map_err(|error| error.to_string())?;
    Ok(TensorValue::from_storage(lhs.shape.clone(), storage))
}

fn comparison_kernel(kind: ComparisonKind) -> CompareOp {
    match kind {
        ComparisonKind::CmpLt | ComparisonKind::Lt => CompareOp::Lt,
        ComparisonKind::Eq => CompareOp::Eq,
        ComparisonKind::Neq => CompareOp::Ne,
        ComparisonKind::Gt => CompareOp::Gt,
        ComparisonKind::Gte => CompareOp::Gte,
        ComparisonKind::Lte => CompareOp::Lte,
    }
}

fn logical_elementwise(
    kind: LogicalKind,
    lhs: &TensorValue,
    rhs: Option<&TensorValue>,
) -> Result<TensorValue, String> {
    if lhs.prim() != Prim::Bool {
        return Err(format!(
            "{}: logical operands must have bool storage",
            kind.surf_name()
        ));
    }
    let lhs_values = lhs
        .storage()
        .to_i64_exact_vec()
        .expect("sealed bool storage has an exact integer view");
    let values = match kind {
        LogicalKind::Not => lhs_values
            .into_iter()
            .map(|value| i64::from(value == 0))
            .collect(),
        LogicalKind::And | LogicalKind::Or => {
            let rhs = rhs.ok_or_else(|| format!("{} requires two operands", kind.surf_name()))?;
            if rhs.prim() != Prim::Bool {
                return Err(format!(
                    "{}: logical operands must have bool storage",
                    kind.surf_name()
                ));
            }
            require_matching_comparison_shapes(kind.surf_name(), lhs, rhs)?;
            let rhs_values = rhs
                .storage()
                .to_i64_exact_vec()
                .expect("sealed bool storage has an exact integer view");
            lhs_values
                .into_iter()
                .zip(rhs_values)
                .map(|(lhs, rhs)| match kind {
                    LogicalKind::And => i64::from(lhs != 0 && rhs != 0),
                    LogicalKind::Or => i64::from(lhs != 0 || rhs != 0),
                    LogicalKind::Not => unreachable!(),
                })
                .collect()
        }
    };
    finalize_wide_int(kind.surf_name(), Prim::Bool, lhs.shape.clone(), values)
}

fn where_elementwise(
    condition: &TensorValue,
    then_value: &TensorValue,
    else_value: &TensorValue,
) -> Result<TensorValue, String> {
    if condition.prim() != Prim::Bool {
        return Err("where: condition must have bool storage".into());
    }
    if then_value.prim() != else_value.prim() {
        return Err("where: branch dtypes must match exactly".into());
    }
    let condition_values = condition
        .storage()
        .to_i64_exact_vec()
        .expect("sealed bool storage has an exact integer view");
    // [05-OP-53]: the condition's shape equals the shape of every branch it
    // selects. A branch selected nowhere is neither read nor shape-checked,
    // so a condition selecting one branch everywhere yields that branch, and
    // an empty condition yields an empty result of its own shape.
    let then_selected = condition_values.iter().any(|selected| *selected != 0);
    let else_selected = condition_values.contains(&0);
    // spec/04 section 4.7: a disagreement is a `Domain` trap in `where`,
    // comparing the condition with a branch as every lane does.
    let shape_error = |branch: &TensorValue| shape_disagreement("where", condition, branch);
    match (then_selected, else_selected) {
        (true, false) if condition.shape != then_value.shape => {
            return Err(shape_error(then_value));
        }
        (true, false) => return Ok(then_value.clone()),
        (false, true) if condition.shape != else_value.shape => {
            return Err(shape_error(else_value));
        }
        (false, true) => return Ok(else_value.clone()),
        (false, false) => {
            return Ok(TensorValue::from_storage(
                condition.shape.clone(),
                tensor_from_scalars(then_value.prim(), &[]),
            ));
        }
        (true, true) => {}
    }
    if condition.shape != then_value.shape {
        return Err(shape_error(then_value));
    }
    if condition.shape != else_value.shape {
        return Err(shape_error(else_value));
    }
    let writes = condition_values
        .into_iter()
        .enumerate()
        .filter_map(|(index, selected)| (selected != 0).then_some((index, index)));
    Ok(TensorValue::from_storage(
        then_value.shape.clone(),
        else_value
            .storage()
            .reuse_overwrite(then_value.storage(), writes),
    ))
}

fn matmul(lhs: &TensorValue, rhs: &TensorValue, prim: Prim) -> Result<TensorValue, String> {
    assert_eq!(lhs.shape.len(), 2);
    assert_eq!(rhs.shape.len(), 2);
    let m = lhs.shape[0];
    let k = lhs.shape[1];
    assert_eq!(rhs.shape[0], k);
    let n = rhs.shape[1];
    let len = admit_result("matmul", &[m, n], prim)?;
    let a = lhs.to_f64_lossy_vec();
    let b = rhs.to_f64_lossy_vec();
    let mut data = vec![0.0; len];
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
    let mut out_shape = batch.to_vec();
    out_shape.extend([m, n]);
    let len = admit_result("matmul", &out_shape, prim)?;
    // An empty result has no element to compute, and its batch extents'
    // product need not be representable: `[2^62, 2^62, 0, 0]` is admitted.
    let batch_count = if len == 0 { 0 } else { len / (m * n) };
    let a = lhs.to_f64_lossy_vec();
    let b = rhs.to_f64_lossy_vec();
    let mut data = vec![0.0; len];
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

fn sparse_domain_shape(
    base: &[usize],
    indices: &[usize],
    axis: usize,
    batch_rank: usize,
) -> Vec<usize> {
    assert!(batch_rank <= axis && batch_rank <= indices.len());
    assert_eq!(&base[..batch_rank], &indices[..batch_rank]);
    base[..axis]
        .iter()
        .chain(&indices[batch_rank..])
        .chain(&base[axis + 1..])
        .copied()
        .collect()
}

fn sparse_index_coordinate(
    domain_index: &[usize],
    axis: usize,
    batch_rank: usize,
    index_rank: usize,
) -> Vec<usize> {
    domain_index[..batch_rank]
        .iter()
        .chain(&domain_index[axis..axis + index_rank - batch_rank])
        .copied()
        .collect()
}

fn gather(
    values: &TensorValue,
    indices: &TensorValue,
    axis: usize,
    batch_rank: usize,
) -> Result<TensorValue, String> {
    assert!(axis < values.shape.len());
    let index_rank = indices.shape.len();
    let index_suffix_rank = index_rank - batch_rank;
    let out_shape = sparse_domain_shape(&values.shape, &indices.shape, axis, batch_rank);
    let out_len = admit_result("gather", &out_shape, values.prim())?;
    let mut picks = Vec::with_capacity(out_len);
    for out_linear in 0..out_len {
        let out_index = linear_to_index(out_linear, &out_shape);
        let idx_index = sparse_index_coordinate(&out_index, axis, batch_rank, index_rank);
        let gathered = index_at(indices, index_to_linear(&idx_index, &indices.shape));
        // [05-SPARSE-1]: an index outside the axis is a `Domain` trap in the
        // primitive, rendered as every lane renders it; no lane panics on it
        // ([04-NUM-10]).
        if gathered < 0 || gathered as usize >= values.shape[axis] {
            return Err(chelis_abi::failure::sparse_index_out_of_bounds(
                "gather",
                gathered as i64,
                axis,
                i64::try_from(values.shape[axis]).unwrap_or(i64::MAX),
            ));
        }
        let mut value_index = Vec::with_capacity(values.shape.len());
        value_index.extend_from_slice(&out_index[..axis]);
        value_index.push(gathered as usize);
        value_index.extend_from_slice(&out_index[axis + index_suffix_rank..]);
        picks.push(index_to_linear(&value_index, &values.shape));
    }
    // reuse_* contract: gather is element-preserving (section C3).
    Ok(TensorValue::from_storage(
        out_shape.clone(),
        values.storage().reuse_gather(&picks),
    ))
}

fn scatter_add(
    target: &TensorValue,
    indices: &TensorValue,
    updates: &TensorValue,
    axis: usize,
    batch_rank: usize,
    prim: Prim,
) -> Result<TensorValue, String> {
    assert!(axis < target.shape.len());
    let index_rank = indices.shape.len();
    let index_suffix_rank = index_rank - batch_rank;
    let expected_updates = sparse_domain_shape(&target.shape, &indices.shape, axis, batch_rank);
    if updates.shape != expected_updates {
        return Err(sparse_updates_disagreement(
            "scatter",
            &expected_updates,
            updates,
        ));
    }

    // [05-OP-33]: each destination's leaves are its target value, then the
    // targeting updates in row-major update order.
    let mut leaves = vec![Vec::new(); target.len()];
    for update_linear in 0..updates.len() {
        let update_index = linear_to_index(update_linear, &updates.shape);
        let idx_index = sparse_index_coordinate(&update_index, axis, batch_rank, index_rank);
        let gathered = index_at(indices, index_to_linear(&idx_index, &indices.shape));
        // [05-SPARSE-1]: an index outside the axis is a `Domain` trap in the
        // primitive, rendered as every lane renders it; no lane panics on it
        // ([04-NUM-10]).
        if gathered < 0 || gathered as usize >= target.shape[axis] {
            return Err(chelis_abi::failure::sparse_index_out_of_bounds(
                "scatter",
                gathered as i64,
                axis,
                i64::try_from(target.shape[axis]).unwrap_or(i64::MAX),
            ));
        }
        let mut target_index = Vec::with_capacity(target.shape.len());
        target_index.extend_from_slice(&update_index[..axis]);
        target_index.push(gathered as usize);
        target_index.extend_from_slice(&update_index[axis + index_suffix_rank..]);
        let target_linear = index_to_linear(&target_index, &target.shape);
        leaves[target_linear].push(update_linear);
    }
    debug_assert_eq!(target.prim(), prim);
    let storage = scatter_add_tensor_groups(target.storage(), updates.storage(), &leaves)
        .map_err(|err| err.to_string())?;
    Ok(TensorValue::from_storage(target.shape.clone(), storage))
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
    batch_rank: usize,
) -> Result<TensorValue, String> {
    assert!(axis < target.shape.len());
    let index_rank = indices.shape.len();
    let index_suffix_rank = index_rank - batch_rank;
    let expected_updates = sparse_domain_shape(&target.shape, &indices.shape, axis, batch_rank);
    if updates.shape != expected_updates {
        return Err(sparse_updates_disagreement(
            "scatter_replace",
            &expected_updates,
            updates,
        ));
    }

    let mut writes = Vec::with_capacity(updates.len());
    for update_linear in 0..updates.len() {
        let update_index = linear_to_index(update_linear, &updates.shape);
        let idx_index = sparse_index_coordinate(&update_index, axis, batch_rank, index_rank);
        let gathered = index_at(indices, index_to_linear(&idx_index, &indices.shape));
        // [05-SPARSE-1]: an index outside the axis is a `Domain` trap in the
        // primitive, rendered as every lane renders it; no lane panics on it
        // ([04-NUM-10]).
        if gathered < 0 || gathered as usize >= target.shape[axis] {
            return Err(chelis_abi::failure::sparse_index_out_of_bounds(
                "scatter_replace",
                gathered as i64,
                axis,
                i64::try_from(target.shape[axis]).unwrap_or(i64::MAX),
            ));
        }
        let mut target_index = Vec::with_capacity(target.shape.len());
        target_index.extend_from_slice(&update_index[..axis]);
        target_index.push(gathered as usize);
        target_index.extend_from_slice(&update_index[axis + index_suffix_rank..]);
        // Last-write-wins: deterministic-order overwrite.
        writes.push((index_to_linear(&target_index, &target.shape), update_linear));
    }
    // reuse_* contract: replace-scatter moves existing elements only
    // (section C3, element-preserving).
    Ok(TensorValue::from_storage(
        target.shape.clone(),
        target.storage().reuse_overwrite(updates.storage(), writes),
    ))
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
) -> Result<TensorValue, String> {
    assert!(axis < data.shape.len());
    if indices.shape != updates.shape {
        return Err(sparse_updates_disagreement(
            "scatter_elements",
            &indices.shape,
            updates,
        ));
    }
    assert_eq!(
        indices.shape.len(),
        data.shape.len(),
        "scatter_elements requires data, indices, and updates to share a rank"
    );

    let mut writes = Vec::with_capacity(updates.len());
    for update_linear in 0..updates.len() {
        let coord = linear_to_index(update_linear, &updates.shape);
        let gathered = index_at(indices, update_linear);
        // [05-SPARSE-1]: an index outside the axis is a `Domain` trap in the
        // primitive, rendered as every lane renders it; no lane panics on it
        // ([04-NUM-10]).
        if gathered < 0 || gathered as usize >= data.shape[axis] {
            return Err(chelis_abi::failure::sparse_index_out_of_bounds(
                "scatter_elements",
                gathered as i64,
                axis,
                i64::try_from(data.shape[axis]).unwrap_or(i64::MAX),
            ));
        }
        let mut target_index = coord.clone();
        target_index[axis] = gathered as usize;
        // Last-write-wins: deterministic-order overwrite.
        writes.push((index_to_linear(&target_index, &data.shape), update_linear));
    }
    // reuse_* contract: element-preserving overwrite (section C3).
    Ok(TensorValue::from_storage(
        data.shape.clone(),
        data.storage().reuse_overwrite(updates.storage(), writes),
    ))
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
/// an axis reduction. Shape planning remains local; arithmetic does not. The
/// result is admitted as `op`'s at `prim` first: removing an axis of an empty
/// operand can leave extents whose product is not representable.
fn axis_reduction_groups(
    op: &'static str,
    prim: Prim,
    input_shape: &[usize],
    axis: usize,
) -> Result<(Vec<usize>, Vec<Vec<usize>>), String> {
    assert!(axis < input_shape.len());
    let mut out_shape = input_shape.to_vec();
    let axis_len = out_shape.remove(axis);
    let out_len = admit_result(op, &out_shape, prim)?;
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
    Ok((out_shape, groups))
}

/// Axis reduction through the closed Phase 2 typed kernel.
/// `name` is the primitive's [04-NUM-9] name and `result` its result dtype.
fn reduce(
    name: &'static str,
    input: &TensorValue,
    axis: usize,
    op: TensorReduceOp,
    result: Prim,
) -> Result<TensorValue, String> {
    let (out_shape, groups) = axis_reduction_groups(name, result, &input.shape, axis)?;
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
    let name = match op {
        ArgReduceOp::Argmax => "argmax_reduce",
        ArgReduceOp::Argmin => "argmin_reduce",
    };
    let (out_shape, groups) = axis_reduction_groups(name, Prim::Int64, &input.shape, axis)?;
    let storage =
        arg_reduce_tensor_groups(op, input.storage(), &groups).map_err(|err| err.to_string())?;
    Ok(TensorValue::from_storage(out_shape, storage))
}

fn is_runtime_mean_div(dag: &Dag, node: &DagNode) -> bool {
    let [sum_id, divisor_id] = node.inputs.as_slice() else {
        return false;
    };
    let storage_sum = |id| {
        let node = dag.get(id)?;
        match node.op {
            RiscOp::Cast { new_precision }
                if node.inputs.len() == 1 && new_precision == node.output_type.precision =>
            {
                dag.get(node.inputs[0])
            }
            _ => Some(node),
        }
    };
    let (Some(sum), Some(divisor)) = (storage_sum(*sum_id), storage_sum(*divisor_id)) else {
        return false;
    };
    let (
        RiscOp::Sum { axis: sum_axis, .. },
        RiscOp::Sum {
            axis: count_axis, ..
        },
    ) = (&sum.op, &divisor.op)
    else {
        return false;
    };
    if sum_axis != count_axis {
        return false;
    }
    let Some(ones) = divisor.inputs.first().and_then(|input| dag.get(*input)) else {
        return false;
    };
    matches!(
        ones.op,
        RiscOp::Const { value } if value.as_f64_lossy() == 1.0
    )
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
    let mut groups = vec![Vec::<usize>::new(); admit_result("count", &out_shape, Prim::Int64)?];
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

/// A permutation keeps the element count but not the stride products: an
/// empty `[2^62, 2^62, 0]` admits and its `[0, 2^62, 2^62]` does not, so the
/// result is admitted as the C runtime's permutation plan admits it.
fn permute(input: &TensorValue, axes: &[usize]) -> Result<TensorValue, String> {
    assert_eq!(axes.len(), input.shape.len());
    let out_shape: Vec<usize> = axes.iter().map(|&axis| input.shape[axis]).collect();
    let out_len = admit_result("permute", &out_shape, input.prim())?;
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
    Ok(TensorValue::from_storage(
        out_shape,
        input.storage().reuse_gather(&picks),
    ))
}

/// `op` is the expansion's primitive, `expand` or `insert`, whose result is
/// admitted before its index map is built.
fn expand(
    op: &'static str,
    input: &TensorValue,
    axis: usize,
    out_shape: Vec<usize>,
) -> Result<TensorValue, String> {
    assert!(axis <= input.shape.len());
    let out_len = admit_result(op, &out_shape, input.prim())?;
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
    Ok(TensorValue::from_storage(
        out_shape,
        input.storage().reuse_gather(&picks),
    ))
}

fn one_hot(indices: &TensorValue, vocab: usize, prim: Prim) -> Result<TensorValue, String> {
    let mut out_shape = indices.shape.clone();
    out_shape.push(vocab);
    let mut out = vec![0i64; admit_result("one_hot", &out_shape, prim)?];
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
    axis: usize,
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
                return Err(
                    negative_bound_failure(node, values, axis, raw).unwrap_or_else(|| {
                        format!(
                            "movement bound at node {}: bound-source (slot {i}) must be a \
                         non-negative integer, got {raw}",
                            node.id.0
                        )
                    }),
                );
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

/// The non-negativity guard of spec/04-type-system.md section 4.7 for a
/// negative runtime bound at `axis` of `node`, rendered as every lane renders
/// it: a target extent of `expand`, `insert` or `reshape`, or a bound of
/// `pad` or `shrink`. `None` for an operation whose bounds carry their own
/// guard before they are resolved here.
fn negative_bound_failure(
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
    axis: usize,
    raw: i64,
) -> Option<String> {
    match &node.op {
        RiscOp::Expand { .. } => {
            // One more output axis than the operand has is an `insert`, as
            // the expansion below decides.
            let input_rank = values.get(node.inputs.first()?)?.shape.len();
            let op = if node.output_type.dims.len() == input_rank + 1 {
                crate::axis_sources::ExpansionKind::Insert
            } else {
                crate::axis_sources::ExpansionKind::Expand
            };
            Some(chelis_abi::failure::negative_target_extent(
                op.primitive_name(),
                axis,
                raw,
            ))
        }
        RiscOp::Reshape { .. } => Some(chelis_abi::failure::negative_target_extent(
            "reshape", axis, raw,
        )),
        RiscOp::Pad { .. } => Some(chelis_abi::failure::negative_movement_bound(
            "pad", axis, raw,
        )),
        RiscOp::Shrink { .. } => Some(chelis_abi::failure::negative_movement_bound(
            "shrink", axis, raw,
        )),
        _ => None,
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
                resolve_eval_bound(s, node, values, extent, axis)?,
                resolve_eval_bound(e, node, values, extent, axis)?,
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
            resolve_eval_bound(step, node, values, input_extent, 0)?
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
    // The padded extents are the C runtime's `ShapeMetadata::padded`, so an
    // extent past i64 is its `Overflow`, before anything is allocated.
    let (before, after): (Vec<usize>, Vec<usize>) = padding.iter().copied().unzip();
    let dtype = input
        .prim()
        .runtime_dtype()
        .map_err(|error| error.to_string())?;
    let padded = i64_extents(&input.shape)
        .and_then(|extents| ShapeMetadata::contiguous(&extents, dtype))
        .and_then(|metadata| metadata.padded(&i64_extents(&before)?, &i64_extents(&after)?))
        .map_err(|error| admission_trap("pad", &error))?;
    let out_shape: Vec<usize> = padded
        .shape()
        .iter()
        .map(|&extent| usize::try_from(extent).map_err(|_| "padded extent exceeds usize"))
        .collect::<Result<_, _>>()?;
    let mut map: Vec<Option<usize>> = vec![None; admit_result("pad", &out_shape, input.prim())?];
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
/// The post-bind movement-bound failure of each live node. Each is raised
/// when its node's turn comes, so an earlier node's trap, a draw's rate
/// validation among them, is reported first as the sequential reading orders
/// it.
fn bound_movement_failures(dag: &Dag, live: Option<&[bool]>) -> UnordMap<NodeId, String> {
    dag.nodes()
        .iter()
        .filter(|node| live.is_none_or(|mask| mask.get(node.id.0).copied().unwrap_or(false)))
        .filter_map(|node| {
            verify_bound_movement_node(dag, node)
                .err()
                .map(|error| (node.id, error))
        })
        .collect()
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
) -> Result<(UnordMap<String, TensorValue>, Vec<NodeId>), String>
where
    F: FnMut(&str, TensorInputDemand) -> Result<Option<TensorValue>, String>,
{
    let mut inputs = UnordMap::new();
    let mut resolved_loads = Vec::new();
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
                resolved_loads.push(node.id);
            }
            None if strict_loads && is_live => {
                return Err(format!("missing required input `{name}`"));
            }
            None => {}
        }
    }
    Ok((inputs, resolved_loads))
}

/// The nodes `roots` need, with every observable root
/// ([`crate::dag::TrapSeeds::is_observable_root`]) of a declaration the selection enters.
fn live_mask_for_roots(dag: &Dag, roots: &[NodeId]) -> Vec<bool> {
    let unselected = dag.outside_selection(roots);
    live_mask_from(dag, roots.to_vec(), &unselected)
}

/// The names of the `Load`s an evaluation of `roots` reads: every `Load` in
/// its live set ([`live_mask_for_roots`]), which is the roots' value graphs
/// plus every observable root of a declaration the selection enters
/// (spec/06 §5.2). This is exactly the set `resolve_load_inputs` demands, so
/// an input router that reads it can never filter away a binding the
/// evaluator then reports missing: a discarded trapping node's parameter is
/// required although no root's value reads it.
pub fn required_load_names(dag: &Dag, roots: &[NodeId]) -> BTreeSet<String> {
    let live = live_mask_for_roots(dag, roots);
    dag.nodes()
        .iter()
        .filter(|node| live[node.id.0])
        .filter_map(|node| match &node.op {
            RiscOp::Load { name } => Some(name.as_str().to_owned()),
            _ => None,
        })
        .collect()
}

/// Whether an evaluation of `roots` runs a node none of their values reads:
/// an observable root ([`crate::dag::TrapSeeds::is_observable_root`]) of a
/// declaration the selection enters that sits outside every root's value
/// graph, such as the trap of a discarded `let` (spec/06 §5.2, spec/03
/// §4.4), or a node only such a root reads. It compares the evaluator's own
/// live set ([`live_mask_for_roots`]) with the roots' value graphs.
///
/// A lane that implements a graph by its roots' values alone, returning a
/// parameter unchanged or calling one runtime routine for a recognized root
/// operation, agrees with the evaluator exactly when this is false; where it
/// is true that lane would drop the discarded work, and with it the trap.
pub fn runs_discarded_work(dag: &Dag, roots: &[NodeId]) -> bool {
    let live = live_mask_for_roots(dag, roots);
    let values = dependency_closure(dag, roots.to_vec());
    live.iter()
        .zip(&values)
        .any(|(live, value)| *live && !*value)
}

fn live_mask_from(dag: &Dag, mut stack: Vec<NodeId>, unselected: &[bool]) -> Vec<bool> {
    // chelis#2368: effect nodes are live because they are effects, not
    // because a value reaches them; chelis#2440 and chelis#2413: so is a
    // potentially trapping node, numeric or random. One predicate names the
    // class ([`crate::dag::TrapSeeds::is_observable_root`]).
    //
    // chelis#2476 scopes it to the selection ([`Dag::outside_selection`]).
    // An abort, or a trap, in a declaration this evaluation does not enter is
    // not its concern, so firing it would abort on behalf of code the caller
    // excluded, and would demand that declaration's parameters as inputs. The
    // seed still reaches every observable root of the declarations the
    // selection enters, including the discarded ones, which is what
    // [05-OP-68] is about.
    let seeds = dag.trap_seeds();
    stack.extend(
        dag.nodes()
            .iter()
            .filter(|node| seeds.is_observable_root(node) && !unselected[node.id.0])
            .map(|node| node.id),
    );
    dependency_closure(dag, stack)
}

/// The nodes `stack` reaches through everything a node reads to run.
fn dependency_closure(dag: &Dag, mut stack: Vec<NodeId>) -> Vec<bool> {
    let mut live = vec![false; dag.len()];
    while let Some(id) = stack.pop() {
        if live[id.0] {
            continue;
        }
        live[id.0] = true;
        if let Some(node) = dag.get(id) {
            stack.extend(node.inputs.iter().copied());
            // chelis#616: a runtime-dim declarer kept via `shape_deps`
            // must actually EVALUATE so the mid-evaluation binding sees
            // its extent (the consumer reads the dim, not the value).
            stack.extend(node.shape_deps.iter().copied());
            stack.extend(node.result_claim_deps.iter().copied());
            // A node reads its activation to decide whether it checks.
            stack.extend(node.owner.activation);
        }
    }
    live
}

/// The node selection an evaluation runs under, and which of its values the
/// caller needs back.
///
/// chelis#828 work class 1. The evaluator used to hold every executed node's
/// tensor until it returned, so peak host memory tracked the number of
/// EXECUTED NODES rather than the number of tensors simultaneously reachable.
/// A deep feed-forward model therefore paid for every activation it had ever
/// computed, which is why an operator-complete model is OOM-killed under
/// `chelis eval` (`Chelis-Lang/hydronnx#54`).
///
/// The distinction this enum draws is the one the caller already makes. An
/// entrypoint that NAMES ITS ROOTS has said which values are results, so it
/// asks for [`EvaluationScope::Roots`] and the evaluation loop drops every
/// other value once the last step that reads it has run. An entrypoint that
/// names no roots cannot know what to keep, so it keeps everything: that is
/// the all-values API the tracker preserves, and `eval_tensor`,
/// `eval_tensor_with`, `eval_tensor_with_strict` and the segment entrypoint
/// all use it.
///
/// The live mask travels with the selection rather than beside it, so a run
/// cannot mask off one set of nodes while retaining the roots of another.
#[derive(Clone, Copy)]
enum EvaluationScope<'a> {
    /// Execute every node in the graph and return every value.
    WholeDag,
    /// Execute only the masked nodes and return every executed value: the
    /// all-values baseline the reclamation tests measure against.
    #[cfg(test)]
    MaskedAllValues(&'a [bool]),
    /// Execute only the masked nodes and return exactly `roots`. Every other
    /// value is freed after the last step that reads it.
    Roots {
        live: &'a [bool],
        roots: &'a [NodeId],
    },
}

impl<'a> EvaluationScope<'a> {
    fn live(&self) -> Option<&'a [bool]> {
        match self {
            Self::WholeDag => None,
            #[cfg(test)]
            Self::MaskedAllValues(live) => Some(live),
            Self::Roots { live, .. } => Some(live),
        }
    }

    /// The values this run must keep, or `None` when it keeps all of them.
    fn retained_roots(&self) -> Option<&'a [NodeId]> {
        match self {
            Self::WholeDag => None,
            #[cfg(test)]
            Self::MaskedAllValues(_) => None,
            Self::Roots { roots, .. } => Some(roots),
        }
    }
}

/// One evaluation's outputs, and what holding them cost.
struct EvaluatedValues {
    values: UnordMap<NodeId, TensorValue>,
    /// The most entries `values` ever held at once, sampled after each node's
    /// own value is inserted and before that step's reclamation runs, so it
    /// records the true transient rather than the post-reclamation residue.
    ///
    /// This is chelis#828's receipt: under [`EvaluationScope::Roots`] it must
    /// track the live working set instead of the executed node count. It is
    /// read by this module's reclamation tests; nothing in the shipped lanes
    /// consumes it, and exporting it would be public surface the tracker did
    /// not ask for.
    #[cfg_attr(not(test), allow(dead_code))]
    peak_live_values: usize,
    /// The most tensor ELEMENTS `values` ever held at once, by the same
    /// sampling rule. [`TensorStorage`] owns its buffer outright rather than
    /// sharing a handle, so removing an entry releases that entry's elements
    /// and this count is a faithful proxy for the evaluator's own footprint.
    #[cfg_attr(not(test), allow(dead_code))]
    peak_live_elements: usize,
}

/// For each step of `order`, the values that become unreachable once that step
/// has run.
///
/// A value stays reachable while some LATER step still reads it, and this
/// evaluator reads a value through more edges than [`DagNode::inputs`]:
///
/// - a `Const` sizes itself from a `shape_deps` value's realized shape;
/// - a producer's declared-result obligation names its witness through
///   `result_claim_deps`;
/// - a local extent guard reads its `activation`, its declaring
///   [`crate::axis_sources::CanonicalExtent::Witness`], and every member of a
///   [`crate::axis_sources::SameShapeAgreement`], all by node id rather than
///   through the site node's operand list.
///
/// Every one of those is a read, so every one of them extends a lifetime here.
/// Freeing a value one of them still reads would be a wrong answer or a
/// spurious "not available" error, which is a far worse defect than the memory
/// growth this reclamation exists to fix. A movement bound (`RtDim::Node`,
/// `RtDim::InputAxis`) and every `ComputedAxisExtent` observation address the
/// site node's own `inputs` slots, so `inputs` already covers them.
///
/// Steps the live mask skips never run and therefore never read; the scan
/// skips them for the same reason the evaluation loop does, so a masked-off
/// consumer does not pin a value its live consumers have finished with.
///
/// `retain` is never freed. A node that is both a selected root and an
/// intermediate is therefore kept, which is what makes overlapping root cones
/// safe.
fn value_free_schedule(
    dag: &Dag,
    order: &[NodeId],
    live: Option<&[bool]>,
    local_guard_sites: &UnordMap<NodeId, Vec<(usize, crate::axis_sources::LocalGuardClaim)>>,
    retain: &[NodeId],
) -> Vec<Vec<NodeId>> {
    let mut last_use: Vec<Option<usize>> = vec![None; dag.len()];
    let mut produced_at: Vec<Option<usize>> = vec![None; dag.len()];
    for (index, id) in order.iter().enumerate() {
        let Some(node) = dag.get(*id) else {
            continue;
        };
        if live.is_some_and(|mask| !mask[node.id.0]) {
            continue;
        }
        {
            let mut read = |source: NodeId| {
                if let Some(slot) = last_use.get_mut(source.0) {
                    *slot = Some(index);
                }
            };
            for input in &node.inputs {
                read(*input);
            }
            for dep in &node.shape_deps {
                read(*dep);
            }
            for dep in &node.result_claim_deps {
                read(*dep);
            }
            if let Some(activation) = node.owner.activation {
                read(activation);
            }
            if let Some(sites) = local_guard_sites.get(&node.id) {
                for (_, claim) in sites {
                    if let Some(activation) = claim.activation.node() {
                        read(activation);
                    }
                    if let crate::axis_sources::CanonicalExtent::Witness(witness) = &claim.canonical
                    {
                        read(*witness);
                    }
                    if let crate::axis_sources::LocalGuardObservation::SameShapeAgreement(
                        agreement,
                    ) = &claim.observed
                    {
                        for member in agreement.members() {
                            read(*member);
                        }
                    }
                }
            }
        }
        if let Some(slot) = produced_at.get_mut(node.id.0) {
            *slot = Some(index);
        }
    }

    let mut retained = vec![false; dag.len()];
    for root in retain {
        if let Some(slot) = retained.get_mut(root.0) {
            *slot = true;
        }
    }

    let mut schedule = vec![Vec::new(); order.len()];
    for (raw, produced) in produced_at.iter().enumerate() {
        let Some(produced) = *produced else {
            continue;
        };
        if retained[raw] {
            continue;
        }
        // A topologically ordered schedule always reads a value after it is
        // produced; taking the later of the two costs nothing and keeps a
        // malformed schedule from scheduling a free before the insert it
        // would undo.
        let free_at = last_use[raw].unwrap_or(produced).max(produced);
        schedule[free_at].push(NodeId(raw));
    }
    schedule
}

struct PreparedTensorInputs {
    inputs: UnordMap<String, TensorValue>,
    resolved_loads: Vec<NodeId>,
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
    let (resolved_inputs, resolved_loads) = resolve_load_inputs(
        dag,
        live,
        strict_loads,
        &symbolic_dim_load_inputs,
        &symbolic_selection.required_shape_inputs,
        &mut load_input,
    )?;
    Ok(PreparedTensorInputs {
        inputs: resolved_inputs,
        resolved_loads,
        required_symbols,
        needs_symbolic_binding,
        symbolic_selection,
    })
}

fn eval_tensor_internal<F>(
    dag: &Dag,
    scope: EvaluationScope<'_>,
    strict_loads: bool,
    load_input: F,
) -> Result<EvaluatedValues, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    eval_tensor_internal_with_result_claims(dag, scope, strict_loads, &[], load_input)
}

/// One caller's declared-result obligation on a graph's single root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InheritedResultClaim {
    pub rank: usize,
    pub axes: Vec<InheritedResultAxis>,
}

/// One claimed result axis: its required extent and, for a named claim, the
/// binder and declaring parameter axis its context names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InheritedResultAxis {
    pub axis: usize,
    pub required: usize,
    pub source: Option<crate::axis_sources::ClaimSource>,
}

fn eval_tensor_internal_with_result_claims<F>(
    dag: &Dag,
    scope: EvaluationScope<'_>,
    strict_loads: bool,
    result_claims: &[InheritedResultClaim],
    mut load_input: F,
) -> Result<EvaluatedValues, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    // Reject an incomplete claim/producer relationship before any input
    // preparation or entry guard can observe runtime state.
    let declared_local_guard_sites = crate::axis_sources::local_dim_guard_sites(dag)?;
    let live = scope.live();
    let PreparedTensorInputs {
        inputs: mut resolved_inputs,
        resolved_loads,
        required_symbols,
        needs_symbolic_binding,
        symbolic_selection,
    } = prepare_tensor_inputs(dag, live, strict_loads, |name, _| Ok(load_input(name)))?;
    // Guard claims come from the unbound DAG; observed extents come from
    // actual caller inputs. Both host lanes consume the same individual
    // schedule before symbolic inference or dependent operations. Missing
    // inputs belonging only to an unrelated root remain optional (#991).
    // chelis#1374: a named witness claim whose pair the entry schedule above
    // already compares is checked there, once (spec/04 §4.7). The claim stays
    // in the graph because it is also what retains a declared-but-unread
    // parameter's interface witness.
    let entry_covered = crate::axis_sources::entry_covered_witness_claims(dag);
    // One IR plan supplies the same slot/axis order to Eval and direct C.
    // Freeze tagged/raw values at entry, before symbolic binding or body
    // execution; a Load later reads only its admitted value.
    for step in crate::axis_sources::entry_validation_plan_for_resolved_loads(dag, &resolved_loads)
    {
        use crate::axis_sources::{EntryExtentGuard, EntryValidationStep};
        match step {
            EntryValidationStep::DType { load } => {
                let node = dag.get(load).expect("entry input");
                let RiscOp::Load { name } = &node.op else {
                    unreachable!()
                };
                if let Some(value) = resolved_inputs.get(name.as_str()) {
                    let admitted =
                        ingress_to_declared(name.as_str(), node.output_type.precision, value)?;
                    resolved_inputs.insert(name.as_str().to_string(), admitted);
                }
            }
            EntryValidationStep::Rank { load } => {
                let node = dag.get(load).expect("entry input");
                let RiscOp::Load { name } = &node.op else {
                    unreachable!()
                };
                if let Some(value) = resolved_inputs.get(name.as_str())
                    && value.shape.len() != node.output_type.dims.len()
                {
                    return Err(format!(
                        "input `{name}` expected rank {}, got {}\nnumeric trap: domain in load at i64",
                        node.output_type.dims.len(),
                        value.shape.len()
                    ));
                }
            }
            EntryValidationStep::LiteralAxis {
                load,
                axis,
                required,
            } => {
                let node = dag.get(load).expect("entry input");
                let RiscOp::Load { name } = &node.op else {
                    unreachable!()
                };
                if let Some(observed) = resolved_inputs
                    .get(name.as_str())
                    .and_then(|value| value.shape.get(axis))
                    .copied()
                    && observed != required
                {
                    return Err(format!(
                        "extent `{required}`: claimed = {required}, {name} axis {axis} = {observed}\nnumeric trap: domain in load at i64"
                    ));
                }
            }
            EntryValidationStep::Extent(guard) => {
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
                        let (
                            Some((left_label, left_axis, left)),
                            Some((right_label, right_axis, right)),
                        ) = (read(canonical), read(observed))
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
                        format!(
                            "extent `{required}`: claimed = {required}, {label} axis {axis} = {actual}"
                        )
                    }
                };
                return Err(format!("{context}\nnumeric trap: domain in load at i64"));
            }
        }
    }

    let mut prebound_dims: UnordMap<String, usize> = UnordMap::new();
    let mut movement_failures: UnordMap<NodeId, String> = UnordMap::new();
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
        // Each failure is raised at its node, before the node executes.
        movement_failures = bound_movement_failures(&bound, live);
        bound
    } else {
        dag.clone()
    };

    let mut values: UnordMap<NodeId, TensorValue> = UnordMap::new();
    // chelis#828's receipt, sampled once per executed node. `live_elements`
    // is maintained incrementally so the sample costs two comparisons rather
    // than a walk of the map.
    let mut live_elements: usize = 0;
    let mut peak_live_values: usize = 0;
    let mut peak_live_elements: usize = 0;

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
    // The extent each claim states for the node it sizes
    // ([`crate::axis_sources::GuardActivation::sized_axis`]): the extent that
    // node declares at that axis where, inactive, it produces zeros.
    let mut claimed_extents: UnordMap<NodeId, Vec<(usize, crate::axis_sources::CanonicalExtent)>> =
        UnordMap::new();
    for (_, claim) in &declared_local_guard_sites {
        if let Some(axis) = claim.activation.sized_axis() {
            claimed_extents
                .entry(claim.activation.claimed())
                .or_default()
                .push((axis, claim.canonical.clone()));
        }
    }
    let mut local_guard_sites: UnordMap<
        NodeId,
        Vec<(usize, crate::axis_sources::LocalGuardClaim)>,
    > = UnordMap::new();
    for ((node, axis), claim) in declared_local_guard_sites {
        local_guard_sites
            .entry(NodeId(node))
            .or_default()
            .push((axis, claim));
    }
    if !result_claims.is_empty() && dag.roots().len() == 1 {
        let root = dag.roots()[0];
        let rank = dag.get(root).expect("result root").output_type.dims.len();
        let result_sites = crate::axis_sources::result_extent_sites(dag, root);
        for claim in result_claims.iter().filter(|claim| claim.rank == rank) {
            for site in &result_sites {
                let RtAxis::Lit(output_axis) = site.output_axis();
                let RtAxis::Lit(producer_axis) = site.producer_axis();
                let Some(axis) = claim
                    .axes
                    .iter()
                    .find(|axis| axis.axis == output_axis as usize)
                else {
                    continue;
                };
                local_guard_sites.entry(site.producer()).or_default().push((
                    producer_axis as usize,
                    crate::axis_sources::LocalGuardClaim {
                        claim: axis.source.as_ref().map_or_else(
                            || axis.required.to_string(),
                            |source| source.claim.clone(),
                        ),
                        canonical: crate::axis_sources::CanonicalExtent::Resolved(axis.required),
                        op: site.operation(),
                        observed: site.observation().clone(),
                        activation: crate::axis_sources::GuardActivation::sizing(
                            dag,
                            site.producer(),
                            producer_axis as usize,
                        )?,
                        source: axis.source.clone(),
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

    let order = bound_dag
        .nodes()
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    // chelis#828 work class 1. Built from the BOUND graph, whose node ids
    // `bind_symbolic_dims` preserves, and from the same live mask and guard
    // sites the loop below consults, so a step's reads in the scan are exactly
    // the reads it performs.
    let free_schedule = scope
        .retained_roots()
        .map(|roots| value_free_schedule(&bound_dag, &order, live, &local_guard_sites, roots));

    // One seed query for the walk: the literal result claims a witness
    // checks are a whole-graph derivation, read from the unbound graph as
    // the C lane reads them.
    let seeds = dag.trap_seeds();
    for (index, id) in order.into_iter().enumerate() {
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
        // spec/10 §3.2: a node whose activation holds in no row checks
        // nothing. An operand-value check reads neutral operands below; an
        // extent or bound check is skipped here, since one tensor's rows
        // share their extents and it runs when any row is active. Whether
        // the node is gated is read from the unbound graph, as the C lane
        // reads it, since binding renames no node.
        let gated = dag
            .get(node.id)
            .is_some_and(|source| seeds.is_activation_gated(source));
        let claim_sized = gated
            && dag
                .get(node.id)
                .is_some_and(|source| seeds.is_claim_sized(source));
        let inactive = gated && matches!(node_activity(node, &values)?, Activity::Inactive);
        if let Some(failure) = movement_failures.remove(&node.id)
            && !inactive
        {
            return Err(failure);
        }
        // `spec/05-risc-primitives.md` section 2.4.1 makes every stride step
        // one operation-level precondition. Resolve the COMPLETE vector once
        // before any per-axis `StrideSpan` claim can run: otherwise an early
        // axis's claim mismatch can mask a zero or negative later step. The C
        // movement plan validates this same vector before emitting any local
        // extent site. Reuse the resolved vector for both the sites and the
        // operation so Eval has one signed-validation boundary as well.
        let resolved_stride_steps = if inactive {
            None
        } else if let RiscOp::Stride { strides } = &node.op {
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
                if !local_guard_is_active(claim, &values)? {
                    continue;
                }
                // Exhaustive on purpose: a future observation kind has to say
                // here whether it is readable before the node runs, rather
                // than falling through a wildcard and disappearing from this
                // lane while the C lane keeps emitting it.
                let observed = match &claim.observed {
                    crate::axis_sources::LocalGuardObservation::Carrier(carrier) => {
                        resolve_eval_bound(carrier, node, &values, 0, *axis)?
                    }
                    crate::axis_sources::LocalGuardObservation::ComputedExtent(computed) => {
                        match computed_axis_extent_value(
                            computed,
                            node,
                            &values,
                            resolved_stride_steps.as_deref(),
                        )? {
                            Some(extent) => extent,
                            // An inverted span, or one that runs past the
                            // operand's own extent, computes no extent to
                            // compare, so the guard yields rather than
                            // comparing a fabricated number, and `shrink`
                            // reports section 2.4.1's error in the compiled
                            // lane's words (chelis#1797). An empty span is the
                            // real extent 0, which the guard compares like any
                            // other (chelis#1795).
                            None => continue,
                        }
                    }
                    crate::axis_sources::LocalGuardObservation::SameShapeAgreement(agreement) => {
                        same_shape_agreement_extent(dag, node, agreement, *axis, &values)?
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

        // spec/10 §3.2: a node under a false activation computes a value
        // from operands its checks accept, and checks nothing.
        let inactive_operands = neutralize_inactive_operands(node, gated, &mut values)?;
        let mut inactive_value = if inactive {
            inactive_unchecked_value(
                node,
                claim_sized,
                claimed_extents.get(&node.id).map_or(&[], Vec::as_slice),
                dag,
                &values,
                &runtime_dims,
            )?
        } else {
            None
        };
        if inactive_value.is_none() {
            inactive_value = inactive_disagreeing_elementwise(node, &values)?;
        }
        let out_prim = node.output_type.precision;
        let value = match &node.op {
            // An extent or bound check under a false activation: the value
            // of its declared type, checked by nothing.
            _ if inactive_value.is_some() => inactive_value
                .take()
                .expect("an inactive node's unchecked value"),
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
                let n = admit_result("const", &shape, out_prim)?;
                // The SEALED payload materializes exactly (chelis#856):
                // integer/bool payloads splat through the exact i64
                // lane (no f64 laundering above 2^53), float payloads
                // through their exact f64 images. A payload whose value
                // does not survive the node's dtype traps loudly.
                // A payload already at the node's dtype is moved, bits
                // and all, so a NaN constant keeps its encoding.
                match value.as_i64_exact() {
                    _ if value.prim() == out_prim => TensorValue::from_storage(
                        shape,
                        chelis_types::tensor_from_scalars(out_prim, &vec![*value; n]),
                    ),
                    Some(i) => finalize_wide_int("const", out_prim, shape, vec![i; n])?,
                    None => finalize_wide("const", out_prim, shape, vec![value.as_f64_lossy(); n])?,
                }
            }
            RiscOp::ConstTensor { data } => {
                let shape =
                    concrete_shape_with(&node.output_type, &runtime_dims).unwrap_or_default();
                admit_result("const", &shape, out_prim)?;
                // Sealed per-dtype storage: exact integer lane for
                // integer/bool payloads, exact f64 images otherwise
                // (chelis#856).
                match data.to_i64_exact_vec() {
                    _ if data.prim() == out_prim
                        && data.len() == shape.iter().product::<usize>() =>
                    {
                        TensorValue::from_storage(shape, data.clone())
                    }
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
                for required in requirements
                    .iter()
                    .chain(seeds.literal_result_witness_requirements(node.id))
                {
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
                // The entry plan freezes the selected declaration once. A
                // raw, unverified DAG can still contain a second live Load
                // with this name but a conflicting type; do not silently
                // hand it the first Load's admitted storage.
                Some(value) => {
                    ingress_to_declared(name.as_str(), node.output_type.precision, value)?
                }
                None if strict_loads => return Err(format!("missing required input `{name}`")),
                None => default_value(&node.output_type)?,
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
            RiscOp::Mul => {
                let (lhs, rhs) = (&values[&node.inputs[0]], &values[&node.inputs[1]]);
                // A disagreement inside matmul's decomposed product is
                // matmul's own `Domain` trap (spec/04 section 4.7).
                if lhs.shape != rhs.shape && crate::tier2::is_matmul_product(node, |id| dag.get(id))
                {
                    return Err(agreement_failure(dag, node, lhs, rhs));
                }
                binary_elementwise(ElementwiseBinOp::Mul, lhs, rhs)?
            }
            RiscOp::Div => {
                if is_runtime_mean_div(dag, node)
                    && values[&node.inputs[1]]
                        .storage()
                        .to_f64_lossy_vec()
                        .contains(&0.0)
                {
                    return Err(NumericTrap::Domain {
                        op: "mean",
                        prim: out_prim,
                    }
                    .to_string());
                }
                binary_elementwise(
                    ElementwiseBinOp::Div,
                    &values[&node.inputs[0]],
                    &values[&node.inputs[1]],
                )?
            }
            // chelis#178: floor division rounds the quotient toward -inf.
            // For integer-valued operands `(a / b).floor()` yields the
            // integer floored quotient, and for float operands it is
            // `floor(a / b)` directly.
            //
            // chelis#550: integer operands trap on a zero divisor with the
            // shared diagnostic, mirroring the host evaluator (`host_ops`,
            // i64 + trap) and the C backend (`chelis_int_checked_floor_div`) so this
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
            RiscOp::Bitwise(kind) => {
                let lhs = &values[&node.inputs[0]];
                let rhs = &values[&node.inputs[1]];
                require_matching_comparison_shapes(kind.name(), lhs, rhs)?;
                let storage = chelis_types::bitwise_tensor(*kind, lhs.storage(), rhs.storage())
                    .map_err(|error| error.to_string())?;
                TensorValue::from_storage(lhs.shape.clone(), storage)
            }
            RiscOp::Mod => {
                let lhs = &values[&node.inputs[0]];
                let rhs = &values[&node.inputs[1]];
                binary_elementwise(ElementwiseBinOp::Mod, lhs, rhs)?
            }
            // chelis#178: truncating (round-toward-zero) integer division.
            // `(a / b).trunc()` matches C/Rust integer `/` for the
            // integer-valued operands this op is restricted to.
            //
            // chelis#550: `trunc_div` is integer-only (the checker rejects
            // float operands), so a zero divisor always traps with the shared
            // diagnostic — matching `host_ops::eval_trunc_div` and the C
            // backend guard.
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
            RiscOp::Tanh => unary_elementwise(ElementwiseUnOp::Tanh, &values[&node.inputs[0]])?,
            RiscOp::Erf => unary_elementwise(ElementwiseUnOp::Erf, &values[&node.inputs[0]])?,
            RiscOp::Erfc => unary_elementwise(ElementwiseUnOp::Erfc, &values[&node.inputs[0]])?,
            RiscOp::Abs => unary_elementwise(ElementwiseUnOp::Abs, &values[&node.inputs[0]])?,
            RiscOp::Floor => unary_elementwise(ElementwiseUnOp::Floor, &values[&node.inputs[0]])?,
            RiscOp::Ceil => unary_elementwise(ElementwiseUnOp::Ceil, &values[&node.inputs[0]])?,
            // Round-half-to-even (banker's rounding), matching the C
            // backend's `rintf` under the default rounding mode. NOT
            // `f64::round`, which rounds half away from zero.
            RiscOp::Round => unary_elementwise(ElementwiseUnOp::Round, &values[&node.inputs[0]])?,
            RiscOp::Dropout | RiscOp::DropoutReplay => {
                check_draw_extents(&bound_dag, node, &values)?;
                eval_dropout(
                    node,
                    &values[&node.inputs[0]],
                    &values[&node.inputs[1]],
                    draw_keys(node, 2, &values)?,
                )?
            }
            RiscOp::ListMapCapture { .. } => {
                let count = values[&node.inputs[1]].shape[0];
                let source = &values[&node.inputs[0]];
                TensorValue::from_storage(
                    vec![count],
                    source.storage().reuse_gather(&vec![0; count]),
                )
            }
            RiscOp::OrderedAdjointSum { groups } => {
                let mut leaves = vec![
                    finalize_tensor(
                        "adjoint",
                        node.output_type.precision,
                        RawTensor::Float(vec![0.0]),
                    )
                    .map_err(|e| e.to_string())?,
                ];
                let mut offset = 0;
                for &width in groups {
                    let inputs = &node.inputs[offset..offset + width];
                    let count = values[&inputs[0]].len();
                    if inputs.iter().any(|input| values[input].len() != count) {
                        return Err("ordered List cotangent columns have different lengths".into());
                    }
                    for row in 0..count {
                        for input in inputs {
                            leaves.push(values[input].storage().reuse_gather(&[row]));
                        }
                    }
                    offset += width;
                }
                while leaves.len() > 1 {
                    leaves = leaves
                        .chunks(2)
                        .map(|pair| {
                            if pair.len() == 1 {
                                Ok(pair[0].clone())
                            } else {
                                float_tensor_binop(FloatBinOp::Add, &pair[0], &pair[1])
                                    .map_err(|e| e.to_string())
                            }
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                }
                TensorValue::from_storage(vec![], leaves.pop().expect("positive-zero base"))
            }
            RiscOp::Iota => {
                let endpoint = |slot| {
                    values[&node.inputs[slot]]
                        .storage()
                        .scalar_at(0)
                        .as_i64_exact()
                        .ok_or_else(|| "iota requires exact i64 scalar endpoints".to_string())
                };
                let start = endpoint(0)?;
                let end = endpoint(1)?;
                let count = if end <= start {
                    0
                } else {
                    end.checked_sub(start).ok_or_else(|| {
                        NumericTrap::Overflow {
                            op: "range",
                            prim: Prim::Int64,
                        }
                        .to_string()
                    })?
                };
                let count = usize::try_from(count).map_err(|_| {
                    NumericTrap::Overflow {
                        op: "range",
                        prim: Prim::Int64,
                    }
                    .to_string()
                })?;
                admit_result("range", &[count], Prim::Int64)?;
                let mut elements = Vec::new();
                elements
                    .try_reserve_exact(count)
                    .map_err(|_| "range allocation failed".to_string())?;
                for index in 0..count {
                    // count and the last element were bounded by the exact endpoints.
                    elements.push(start + i64::try_from(index).expect("range index fits i64"));
                }
                TensorValue::from_storage(
                    vec![count],
                    finalize_tensor("range", Prim::Int64, RawTensor::Int(elements))
                        .map_err(|e| e.to_string())?,
                )
            }
            RiscOp::UniformLike => {
                check_draw_extents(&bound_dag, node, &values)?;
                eval_uniform_like(
                    node,
                    &values[&node.inputs[0]],
                    &values[&node.inputs[1]],
                    &values[&node.inputs[2]],
                    out_prim,
                    draw_keys(node, 3, &values)?,
                )?
            }
            RiscOp::UniformBoundAdjoint { bound } => {
                check_draw_extents(&bound_dag, node, &values)?;
                eval_uniform_bound_adjoint(
                    node,
                    &values[&node.inputs[1]],
                    *bound,
                    out_prim,
                    draw_keys(node, 2, &values)?,
                )?
            }
            RiscOp::KeyFromSeed => {
                let seeds = &values[&node.inputs[0]];
                key_value(seeds, key_from_seed_storage(seeds.storage()))?
            }
            RiscOp::Split { branch } => {
                let keys = &values[&node.inputs[0]];
                key_value(keys, split_key_storage(keys.storage(), branch.half()))?
            }
            RiscOp::FoldIn => {
                let keys = &values[&node.inputs[0]];
                let ns = &values[&node.inputs[1]];
                if keys.shape.len() != ns.shape.len() {
                    return Err(format!(
                        "fold_in at node {}: key shape {:?} and count shape {:?} must be equal",
                        node.id.0, keys.shape, ns.shape
                    ));
                }
                // Each count extent against the key's, before either is
                // read, with the C lane's report.
                let key_dims = &bound_dag
                    .get(node.inputs[0])
                    .ok_or("fold_in key is not in its graph")?
                    .output_type
                    .dims;
                check_operand_extents("fold_in", key_dims, keys, "input 1", ns, keys.shape.len())?;
                key_value(keys, fold_in_storage(keys.storage(), ns.storage()))?
            }
            RiscOp::KeySelect => {
                // The else keys' extents against the then keys', before
                // either is read, with the C lane's report.
                let then_keys = &values[&node.inputs[0]];
                let key_dims = &bound_dag
                    .get(node.inputs[0])
                    .ok_or("join key is not in its graph")?
                    .output_type
                    .dims;
                let else_keys = &values[&node.inputs[1]];
                if then_keys.shape.len() != else_keys.shape.len() {
                    return Err(format!(
                        "join at node {}: then keys {:?} and else keys {:?} must share one shape",
                        node.id.0, then_keys.shape, else_keys.shape
                    ));
                }
                check_operand_extents(
                    "if",
                    key_dims,
                    then_keys,
                    "input 1",
                    else_keys,
                    then_keys.shape.len(),
                )?;
                eval_key_select(node, &values)?
            }
            RiscOp::SplitN { count } => {
                let keys = &values[&node.inputs[0]];
                let count = if key_operation_is_live(node, &values)? {
                    // [05-OP-71]: a negative runtime count traps before
                    // allocation, with the C lane's `Domain` trap.
                    if let RtDim::Node(slot) = count
                        && let Some(value) =
                            node.inputs.get(*slot).and_then(|input| values.get(input))
                        && rank0_scalar(value, "split_keys count")?
                            .as_i64_exact()
                            .is_some_and(|count| count < 0)
                    {
                        return Err(NumericTrap::Domain {
                            op: "split_keys",
                            prim: Prim::Int64,
                        }
                        .to_string());
                    }
                    resolve_eval_bound(count, node, &values, 0, 0)?
                } else {
                    inactive_split_count(&node.output_type, &runtime_dims)?
                };
                let mut shape = keys.shape.clone();
                shape.push(count);
                // The key's extents and the count are the result's; every
                // extent its type declares is a claim about them, checked
                // before any key exists, as the C lane checks it.
                check_declared_extents("split_keys", &node.output_type, &shape, &runtime_dims)?;
                // [05-OP-33]: the result's count and bytes fit their domains
                // before any key is derived into it (chelis#2491).
                admit_result("split_keys", &shape, Prim::Key)?;
                let storage =
                    split_keys_storage(keys.storage(), count).map_err(|error| error.to_string())?;
                TensorValue::from_storage(shape, storage)
            }
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
            RiscOp::Softmax { axis } => {
                let input = &values[&node.inputs[0]];
                let mut graph = Dag::new();
                let owner = graph.declare("softmax");
                let ty = TensorType {
                    dims: input.shape.iter().copied().map(DimInfo::Lit).collect(),
                    precision: node.output_type.precision,
                };
                let x = graph.add_node(
                    owner,
                    RiscOp::Load { name: "x".into() },
                    vec![],
                    ty.clone(),
                    None,
                );
                let result =
                    crate::tier2::decompose_softmax(owner.into(), &mut graph, x, *axis, &ty, None);
                graph.add_root(result);
                let mut load = |_: &str| Some(input.clone());
                let load: &mut dyn FnMut(&str) -> Option<TensorValue> = &mut load;
                eval_tensor_roots_with_strict(&graph, &[result], load)?
                    .remove(&result)
                    .expect("softmax root evaluated")
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
            RiscOp::Compare(kind) => {
                compare_elementwise(*kind, &values[&node.inputs[0]], &values[&node.inputs[1]])?
            }
            RiscOp::Logical(kind) => logical_elementwise(
                *kind,
                &values[&node.inputs[0]],
                node.inputs.get(1).map(|input| &values[input]),
            )?,
            RiscOp::Where => where_elementwise(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                &values[&node.inputs[2]],
            )?,
            // chelis#1464 / [05-OP-68]: the guard fires BEFORE the fallback
            // is carried, so a taken abort never produces a value. Returning
            // the message as an evaluation error is what makes the evaluator
            // agree with the compiled lane's `chelis_fail`.
            RiscOp::GuardedFail {
                message,
                trap_on_true,
            } => {
                let condition = &values[&node.inputs[0]];
                if condition.prim() != Prim::Bool {
                    return Err("guarded_fail: condition must have bool storage".into());
                }
                let condition_values = condition
                    .storage()
                    .to_i64_exact_vec()
                    .expect("sealed bool storage has an exact integer view");
                // A batched condition aborts when ANY mapped element fires
                // ([05-OP-68]); an unbatched condition has exactly one.
                if condition_values
                    .into_iter()
                    .any(|selected| (selected != 0) == *trap_on_true)
                {
                    return Err(message.clone());
                }
                values[&node.inputs[1]].clone()
            }
            RiscOp::Sum { axis, accumulator } => reduce(
                "sum",
                &values[&node.inputs[0]],
                *axis,
                TensorReduceOp::Sum {
                    accumulator: *accumulator,
                    result: out_prim,
                },
                out_prim,
            )?,
            RiscOp::Count { axes } => count_tensor(&values[&node.inputs[0]], axes)?,
            RiscOp::MaxReduce { axis } => reduce(
                "max_reduce",
                &values[&node.inputs[0]],
                *axis,
                TensorReduceOp::MaxReduce,
                out_prim,
            )?,
            RiscOp::MinReduce { axis } => reduce(
                "min_reduce",
                &values[&node.inputs[0]],
                *axis,
                TensorReduceOp::MinReduce,
                out_prim,
            )?,
            RiscOp::ProdReduce { axis } => reduce(
                "prod_reduce",
                &values[&node.inputs[0]],
                *axis,
                TensorReduceOp::ProdReduce,
                out_prim,
            )?,
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
                    .enumerate()
                    .map(|(axis, dim)| match dim {
                        RtDim::Lit(n) => Ok(*n),
                        // chelis#616: a runtime target extent reads its rank-0
                        // integer scalar exactly like a movement bound. A
                        // negative one fails spec/04 section 4.7's
                        // non-negativity guard: a `Domain` trap in `reshape`.
                        RtDim::Node(slot) => {
                            let negative = node
                                .inputs
                                .get(*slot)
                                .and_then(|input| values.get(input))
                                .filter(|value| !value.is_empty())
                                .and_then(|value| value.storage().scalar_at(0).as_i64_exact())
                                .filter(|raw| *raw < 0);
                            if let Some(raw) = negative {
                                return Err(chelis_abi::failure::negative_target_extent(
                                    "reshape", axis, raw,
                                ));
                            }
                            resolve_eval_bound(dim, node, &values, 0, axis)
                        }
                        RtDim::InputAxis { .. } => resolve_eval_bound(dim, node, &values, 0, axis),
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
                // the C backend's runtime numel abort), never a panic. The
                // target's count is the admitted one, so an unrepresentable
                // product traps rather than overflowing (chelis#2491).
                let input = &values[&node.inputs[0]];
                let expected = admit_result("reshape", &shape, input.prim())?;
                // The rendering is every lane's, so the lanes report the
                // `Domain` trap alike (spec/04-type-system.md section 4.7).
                if expected != input.len() {
                    return Err(chelis_abi::failure::reshape_element_count_disagreement(
                        u64::try_from(expected).unwrap_or(u64::MAX),
                        u64::try_from(input.len()).unwrap_or(u64::MAX),
                    ));
                }
                reshape(input, shape)
            }
            RiscOp::Permute { axes } => permute(&values[&node.inputs[0]], axes)?,
            RiscOp::Expand { axis, size } => {
                let input = &values[&node.inputs[0]];
                let size_value = resolve_eval_bound(size, node, &values, 0, *axis)?;
                let mut out_shape = input.shape.clone();
                let kind = if node.output_type.dims.len() == input.shape.len() + 1 {
                    if *axis > out_shape.len() {
                        return Err(format!(
                            "expand at node {}: axis {} out of bounds for rank {} tensor",
                            node.id.0,
                            axis,
                            input.shape.len()
                        ));
                    }
                    out_shape.insert(*axis, size_value);
                    crate::axis_sources::ExpansionKind::Insert
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
                    crate::axis_sources::ExpansionKind::Expand
                } else {
                    return Err(format!(
                        "expand at node {}: output rank {} must equal input rank {} or {}",
                        node.id.0,
                        node.output_type.dims.len(),
                        input.shape.len(),
                        input.shape.len() + 1
                    ));
                };
                expand(kind.primitive_name(), input, *axis, out_shape)?
            }
            RiscOp::OneHot { vocab } => one_hot(&values[&node.inputs[0]], *vocab, out_prim)?,
            RiscOp::Pad { padding, fill } => {
                let input = &values[&node.inputs[0]];
                let resolved = resolve_eval_pairs(padding, node, &values, &input.shape)?;
                pad(input, &resolved, *fill)?
            }
            RiscOp::Shrink { bounds } => {
                let input = &values[&node.inputs[0]];
                // spec/05 section 2.4.1 closes the runtime-bound errors: equal
                // endpoints select an empty axis (chelis#1795), and only an
                // inverted or overshooting range traps, inside `shrink`.
                let resolved = resolve_eval_pairs(bounds, node, &values, &input.shape)?;
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
                        FusedStepOp::Tanh => unary_elementwise(
                            ElementwiseUnOp::Tanh,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Erf => unary_elementwise(
                            ElementwiseUnOp::Erf,
                            resolve(&step.input_indices[0]),
                        )?,
                        FusedStepOp::Erfc => unary_elementwise(
                            ElementwiseUnOp::Erfc,
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
            RiscOp::NamedCast {
                mode,
                new_precision,
            } => {
                let input = &values[&node.inputs[0]];
                named_cast_value(*mode, input, *new_precision)?
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
            RiscOp::Gather { axis, batch_rank } => gather(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                *axis,
                *batch_rank,
            )?,
            RiscOp::ScatterAdd { axis, batch_rank } => scatter_add(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                &values[&node.inputs[2]],
                *axis,
                *batch_rank,
                out_prim,
            )?,
            RiscOp::Scatter { axis, batch_rank } => scatter_replace(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                &values[&node.inputs[2]],
                *axis,
                *batch_rank,
            )?,
            RiscOp::ScatterElements { axis } => scatter_elements(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                &values[&node.inputs[2]],
                *axis,
            )?,
        };
        for (id, original) in inactive_operands {
            values.insert(id, original);
        }
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
                if !local_guard_is_active(claim, &values)? {
                    continue;
                }
                let Some(&observed) = value.shape.get(*axis) else {
                    continue;
                };
                local_guard_verdict(*axis, claim, observed, &mut runtime_dims, &values)?;
            }
        }
        let produced_elements = value.len();
        if let Some(replaced) = values.insert(node.id, value) {
            live_elements -= replaced.len();
        }
        live_elements += produced_elements;
        peak_live_values = peak_live_values.max(values.len());
        peak_live_elements = peak_live_elements.max(live_elements);
        // Reclaim AFTER the sample above: the sample is the transient this
        // step actually held, and reclamation is what keeps the next one
        // bounded. Under an all-values scope there is no schedule and
        // nothing is removed, so that lane's map is byte-for-byte what it
        // always was.
        if let Some(schedule) = &free_schedule {
            for dead in &schedule[index] {
                if let Some(freed) = values.remove(dead) {
                    live_elements -= freed.len();
                }
            }
        }
    }

    Ok(EvaluatedValues {
        values,
        peak_live_values,
        peak_live_elements,
    })
}

/// spec/10 §3.2 for an elementwise node whose activation holds in no row:
/// its operands' extent agreement is a check, so it checks nothing. An
/// untaken arm's claim-sized node is zeros at the claimed extent (decisions
/// section 25), while a node the arm sizes from an unclaimed origin keeps
/// that origin's extent (a `grad` body's accumulator sized from its
/// parameter), so an untaken arm's operands can disagree. The node then
/// yields zeros at its first operand's shape, read only by the arm's own
/// nodes and by the join, which does not read an unselected branch
/// ([05-OP-53]). A `where` is one such node: an untaken arm's inner join
/// checks its condition against no branch, and yields zeros at the
/// condition's shape. `None` where the operands agree or some row is active.
fn inactive_disagreeing_elementwise(
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
) -> Result<Option<TensorValue>, String> {
    if !matches!(
        node.op,
        RiscOp::Add
            | RiscOp::Sub
            | RiscOp::Mul
            | RiscOp::Div
            | RiscOp::FloorDiv
            | RiscOp::TruncDiv
            | RiscOp::Mod
            | RiscOp::Bitwise(_)
            | RiscOp::MaxElem
            | RiscOp::MinElem
            | RiscOp::Compare(_)
            | RiscOp::Logical(_)
            | RiscOp::FusedElem { .. }
            | RiscOp::Where
    ) {
        return Ok(None);
    }
    let shapes = node
        .inputs
        .iter()
        .filter_map(|input| values.get(input))
        .map(|value| &value.shape)
        .filter(|shape| !shape.is_empty())
        .collect::<Vec<_>>();
    let Some(first) = shapes.first() else {
        return Ok(None);
    };
    if shapes.iter().all(|shape| shape == first)
        || !matches!(node_activity(node, values)?, Activity::Inactive)
    {
        return Ok(None);
    }
    let prim = node.output_type.precision;
    let len = admit_result("activation", first, prim)?;
    Ok(Some(TensorValue::from_storage(
        first.to_vec(),
        zero_storage(prim, len)?,
    )))
}

/// The value a node whose activation holds in no row produces without
/// checking anything (spec/10 §3.2), for the classes that check an extent or
/// a bound rather than an operand's values ([`RuntimeCheck`]), and for a
/// node whose declared extent rests on a claim checked under its activation
/// (`claim_sized`, [`crate::dag::TrapSeeds::is_claim_sized`]); `None` for a
/// node that computes from its operands as usual.
///
/// A movement node reads no bound and produces zeros of its declared type,
/// so an empty or out-of-range bound in an untaken arm allocates and traps
/// on nothing. A claim-sized node produces zeros of its declared type the
/// same way, so an operand whose extent disagrees with the claim is never
/// read. In both, an axis nothing has bound yet takes the extent a claim
/// states for it (`claimed`, the unchecked claim's canonical value), and
/// otherwise the extent its carrier reads (for a movement, its operand's
/// axis). A reduction over an empty axis reduces to zeros. An extent claim
/// compares nothing and produces the value it would have checked.
fn inactive_unchecked_value(
    node: &DagNode,
    claim_sized: bool,
    claimed: &[(usize, crate::axis_sources::CanonicalExtent)],
    source: &Dag,
    values: &UnordMap<NodeId, TensorValue>,
    runtime_dims: &UnordMap<String, usize>,
) -> Result<Option<TensorValue>, String> {
    let operand = || {
        node.inputs
            .first()
            .and_then(|input| values.get(input))
            .ok_or_else(|| format!("node {} reads a missing operand", node.id.0))
    };
    // All-zero bits, the C lane's zero-fill: positive zeros, and at `key`
    // the key whose bits are zero (`key_from_seed(0)`).
    let zeros = |shape: &[usize]| -> Result<TensorValue, String> {
        let prim = node.output_type.precision;
        let len = admit_result("activation", shape, prim)?;
        let storage = if prim == Prim::Key {
            let seed = chelis_types::scalar_from_i64("activation", Prim::Int64, 0)
                .map_err(|trap| trap.to_string())?;
            let zero = RandomKey::from_seed(seed).map_err(|error| error.to_string())?;
            TensorStorage::from_keys(vec![zero; len])
        } else {
            zero_storage(prim, len)?
        };
        Ok(TensorValue::from_storage(shape.to_vec(), storage))
    };
    // The extent a claim states for `axis`: its resolved size, its declaring
    // witness's scalar, or its binder as bound.
    let claimed_extent = |axis: usize| {
        claimed
            .iter()
            .find(|(claimed_axis, _)| *claimed_axis == axis)
            .and_then(|(_, canonical)| match canonical {
                crate::axis_sources::CanonicalExtent::Resolved(extent) => Some(*extent),
                crate::axis_sources::CanonicalExtent::Witness(witness) => {
                    usize::try_from(values.get(witness)?.storage().scalar_at(0).as_i64_exact()?)
                        .ok()
                }
                crate::axis_sources::CanonicalExtent::Binder(name) => {
                    runtime_dims.get(name).copied()
                }
            })
    };
    // The declared type's extents: a literal or a resolved name as stated, a
    // runtime name as bound, and a name nothing has bound yet as the extent a
    // claim states for its axis, or else the one `unbound` gives it (the
    // axis, when neither gives one).
    let declared_shape = |unbound: &dyn Fn(usize) -> Option<usize>| {
        node.output_type
            .dims
            .iter()
            .enumerate()
            .map(|(axis, dim)| match dim {
                DimInfo::Lit(extent) | DimInfo::Named(_, Some(extent)) => Ok(*extent),
                DimInfo::Named(name, None) => runtime_dims
                    .get(name)
                    .copied()
                    .or_else(|| claimed_extent(axis))
                    .or_else(|| unbound(axis))
                    .ok_or(axis),
            })
            .collect::<Result<Vec<_>, usize>>()
    };
    match node.runtime_check() {
        RuntimeCheck::MovementBounds => {
            let operand = operand()?;
            // A movement keeps its operand's rank, so the operand has every
            // axis the node declares.
            let shape =
                declared_shape(&|axis| operand.shape.get(axis).copied()).map_err(|axis| {
                    format!(
                        "node {} declares axis {axis}, which its operand of rank {} does not have",
                        node.id.0,
                        operand.shape.len()
                    )
                })?;
            zeros(&shape).map(Some)
        }
        _ if claim_sized => {
            // An unbound name no claim states takes what the node's own
            // carrier for its axis reads, an operand's axis or a scalar
            // operand, as the C lane declares it before the node's branch.
            let sources = crate::axis_sources::output_axis_sources(source, node.id);
            let operand = |input: &usize| node.inputs.get(*input).and_then(|id| values.get(id));
            let carried = |axis: usize| match sources.get(axis)? {
                crate::axis_sources::AxisSource::InputAxis {
                    input,
                    axis: RtAxis::Lit(read),
                } => operand(input)?
                    .shape
                    .get(usize::try_from(*read).ok()?)
                    .copied(),
                crate::axis_sources::AxisSource::ScalarInput { input } => {
                    usize::try_from(operand(input)?.storage().scalar_at(0).as_i64_exact()?).ok()
                }
                _ => None,
            };
            let shape = declared_shape(&carried).map_err(|axis| {
                format!(
                    "node {} declares axis {axis}, which no bound name or operand sizes",
                    node.id.0
                )
            })?;
            zeros(&shape).map(Some)
        }
        RuntimeCheck::EmptyAxis => {
            let axis = match &node.op {
                RiscOp::Softmax { axis }
                | RiscOp::MaxReduce { axis }
                | RiscOp::MinReduce { axis }
                | RiscOp::Argmax { axis }
                | RiscOp::Argmin { axis } => *axis,
                _ => return Ok(None),
            };
            let operand = operand()?;
            if operand.shape.get(axis) != Some(&0) {
                return Ok(None);
            }
            let mut shape = operand.shape.clone();
            if !matches!(node.op, RiscOp::Softmax { .. }) {
                shape.remove(axis);
            }
            zeros(&shape).map(Some)
        }
        RuntimeCheck::ExtentClaims => match &node.op {
            RiscOp::ExtentWitness {
                axis: RtAxis::Lit(axis),
                ..
            } => {
                let extent = operand()?
                    .shape
                    .get(*axis as usize)
                    .copied()
                    .ok_or_else(|| format!("extent witness axis {axis} out of bounds"))?;
                finalize_wide_int(
                    "shape",
                    node.output_type.precision,
                    vec![],
                    vec![i64::try_from(extent).map_err(|_| "extent exceeds i64")?],
                )
                .map(Some)
            }
            RiscOp::CheckedReshapeExtent { .. } => Ok(Some(operand()?.clone())),
            _ => Ok(None),
        },
        RuntimeCheck::Nothing
        | RuntimeCheck::OperandValues
        | RuntimeCheck::MeanDivisor
        | RuntimeCheck::Random
        | RuntimeCheck::Abort
        | RuntimeCheck::SparseIndex
        | RuntimeCheck::Ungated => Ok(None),
    }
}

fn local_guard_is_active(
    claim: &crate::axis_sources::LocalGuardClaim,
    values: &UnordMap<NodeId, TensorValue>,
) -> Result<bool, String> {
    let Some(activation) = claim.activation.node() else {
        return Ok(true);
    };
    let value = values.get(&activation).ok_or_else(|| {
        format!(
            "local extent guard activation at node {} is not available",
            activation.0
        )
    })?;
    // An extent is shared by every row of its tensor, so under a per-row
    // activation the guard runs when any row is active.
    match value.storage().to_raw() {
        RawTensor::Int(values) if value.prim() == Prim::Bool => {
            Ok(values.iter().any(|value| *value != 0))
        }
        _ => Err(format!(
            "local extent guard activation at node {} is not a Bool",
            activation.0
        )),
    }
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
        let canonical = match &claim.source {
            None => format!("claimed = {claimed}"),
            Some(source) => format!("{} axis {} = {claimed}", source.parameter, source.axis),
        };
        return Err(format!(
            "extent `{}`: {canonical}, {} axis {axis} = {observed}\n\
             numeric trap: domain in {} at i64",
            claim.claim, claim.op, claim.op,
        ));
    }
    Ok(())
}

/// The extents `shape` an operation computed against the extents its
/// declared type claims: a literal, a bound name, or a name an earlier
/// operation declared. A name nothing has bound yet is declared by this
/// result (`op_declared_axes`), and an anonymous one claims nothing. The
/// report is [`local_guard_verdict`]'s, which the C lane mirrors.
/// A branch's join ([`RiscOp::KeySelect`]): element `i` is the then key's
/// where the then activation holds for its row, and the else key's
/// elsewhere. An activation is rank 0, or shaped like a leading part of the
/// keys' shape, and row `r` of `n` keys reads its element `r / (n / len)`, as
/// a draw's activation does. The C lane selects the same way.
fn eval_key_select(
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
) -> Result<TensorValue, String> {
    let value = |slot: usize| {
        node.inputs
            .get(slot)
            .and_then(|input| values.get(input))
            .ok_or_else(|| format!("join at node {} has no input {slot}", node.id.0))
    };
    let (then_value, else_value, active) = (value(0)?, value(1)?, value(2)?);
    let keys = |value: &TensorValue| {
        value
            .storage()
            .keys()
            .map(<[chelis_types::RandomKey]>::to_vec)
            .ok_or_else(|| format!("join at node {} reads a non-key key", node.id.0))
    };
    if then_value.shape != else_value.shape {
        return Err(format!(
            "join at node {}: then keys {:?} and else keys {:?} must share one shape",
            node.id.0, then_value.shape, else_value.shape
        ));
    }
    if active.prim() != Prim::Bool || !then_value.shape.starts_with(&active.shape) {
        return Err(format!(
            "join at node {}: its activation must be a Bool shaped like a leading part of its keys' shape {:?}",
            node.id.0, then_value.shape
        ));
    }
    let (then_keys, mut selected) = (keys(then_value)?, keys(else_value)?);
    let (rows, len) = (selected.len(), active.len());
    for (row, key) in selected.iter_mut().enumerate() {
        let taken = active.storage().scalar_at(leading_row(row, rows, len));
        if taken.as_bool_exact() == Some(true) {
            *key = then_keys[row];
        }
    }
    Ok(TensorValue::from_storage(
        then_value.shape.clone(),
        TensorStorage::from_keys(selected),
    ))
}

/// Whether a key operation's own activation (spec/10 §3.2,
/// [`crate::dag::Owner::activation`]) holds in some row: absent, or a Bool
/// with at least one true element. A key operation whose activation holds in
/// no row reads no count.
fn key_operation_is_live(
    node: &DagNode,
    values: &UnordMap<NodeId, TensorValue>,
) -> Result<bool, String> {
    let Some(input) = node.owner.activation else {
        return Ok(true);
    };
    let active = values
        .get(&input)
        .ok_or("key operation activation is not available")?;
    if active.prim() != Prim::Bool {
        return Err("key operation activation is not a Bool".into());
    }
    let storage = active.storage();
    Ok((0..storage.len()).any(|index| storage.scalar_at(index).as_bool_exact() == Some(true)))
}

/// The count axis of a `SplitN` whose activation holds in no row
/// ([`RiscOp::SplitN`]): the extent its type declares where a literal or an
/// earlier binding fixes it, and zero where the split itself would declare
/// it. The C lane reads the same declaration.
fn inactive_split_count(
    declared: &TensorType,
    runtime_dims: &UnordMap<String, usize>,
) -> Result<usize, String> {
    match declared.dims.last() {
        Some(DimInfo::Lit(value) | DimInfo::Named(_, Some(value))) => Ok(*value),
        Some(DimInfo::Named(name, None)) => Ok(match runtime_dims.get(name) {
            Some(value) => *value,
            // Nothing has bound the name, so the split declares its own
            // count axis, and an unselected split's axis is empty.
            None => 0,
        }),
        None => Err("split_keys declares no count axis".into()),
    }
}

fn check_declared_extents(
    op: &str,
    declared: &TensorType,
    shape: &[usize],
    runtime_dims: &UnordMap<String, usize>,
) -> Result<(), String> {
    if declared.dims.len() != shape.len() {
        return Err(format!(
            "{op} computed rank {}, but its type declares rank {}",
            shape.len(),
            declared.dims.len()
        ));
    }
    for (axis, (dim, &observed)) in declared.dims.iter().zip(shape).enumerate() {
        let (claim, claimed) = match dim {
            DimInfo::Lit(value) => (value.to_string(), *value),
            DimInfo::Named(name, Some(value)) => (name.clone(), *value),
            DimInfo::Named(name, None) => match runtime_dims.get(name) {
                Some(value) => (name.clone(), *value),
                None => continue,
            },
        };
        if observed != claimed {
            return Err(format!(
                "extent `{claim}`: claimed = {claimed}, {op} axis {axis} = {observed}\n\
                 numeric trap: domain in {op} at i64"
            ));
        }
    }
    Ok(())
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

/// The extent an op-computed axis is about to produce, read from the site
/// node's own bounds before that node runs.
///
/// `None` means the operation computes no extent here: a `shrink` span whose
/// start is above its end is inverted, so there is nothing to compare and the
/// guard yields to the operation's own domain trap. An empty span computes
/// the real extent 0 (`spec/05-risc-primitives.md` section 2.4.1,
/// chelis#1795), which a declared literal claim then compares on both lanes.
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
        crate::axis_sources::ComputedAxisExtent::RangeSpan => {
            let start = values[&node.inputs[0]]
                .storage()
                .scalar_at(0)
                .as_i64_exact()
                .ok_or("range requires i64")?;
            let end = values[&node.inputs[1]]
                .storage()
                .scalar_at(0)
                .as_i64_exact()
                .ok_or("range requires i64")?;
            let count = if end <= start {
                0
            } else {
                end.checked_sub(start).ok_or_else(|| {
                    NumericTrap::Overflow {
                        op: "range",
                        prim: Prim::Int64,
                    }
                    .to_string()
                })?
            };
            Ok(usize::try_from(count).ok())
        }
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
            let start = resolve_eval_bound(start, node, values, extent, *operand_axis)?;
            let end = resolve_eval_bound(end, node, values, extent, *operand_axis)?;
            if end > extent {
                return Ok(None);
            }
            Ok(end.checked_sub(start))
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
            let before = resolve_eval_bound(before, node, values, extent, *operand_axis)?;
            let after = resolve_eval_bound(after, node, values, extent, *operand_axis)?;
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
    eval_tensor_internal(dag, EvaluationScope::WholeDag, false, load_input)
        .map(|evaluated| evaluated.values)
}

pub fn eval_tensor_with_strict<F>(
    dag: &Dag,
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    eval_tensor_internal(dag, EvaluationScope::WholeDag, true, load_input)
        .map(|evaluated| evaluated.values)
}

/// Evaluate `roots` and return their values.
///
/// The result holds exactly the named roots. Every other value is freed once
/// the last step that reads it has run, so peak memory tracks the live working
/// set rather than the executed node count (chelis#828). Empty roots select
/// the whole DAG and keep the all-values contract, as do [`eval_tensor`],
/// [`eval_tensor_with`] and [`eval_tensor_with_strict`].
pub fn eval_tensor_roots_with<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    if roots.is_empty() {
        // Empty roots select the whole DAG, which is the all-values contract.
        return eval_tensor_internal(dag, EvaluationScope::WholeDag, false, load_input)
            .map(|evaluated| evaluated.values);
    }
    reject_drop_roots(dag, roots)?;
    let live = live_mask_for_roots(dag, roots);
    eval_tensor_internal(
        dag,
        EvaluationScope::Roots { live: &live, roots },
        false,
        load_input,
    )
    .map(|evaluated| evaluated.values)
}

/// Evaluate `roots` and return their values.
///
/// The result holds exactly the named roots. Every other value is freed once
/// the last step that reads it has run, so peak memory tracks the live working
/// set rather than the executed node count (chelis#828). Empty roots select
/// the whole DAG and keep the all-values contract, as do [`eval_tensor`],
/// [`eval_tensor_with`] and [`eval_tensor_with_strict`].
pub fn eval_tensor_roots_with_strict<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    if roots.is_empty() {
        // Empty roots select the whole DAG, which is the all-values contract.
        return eval_tensor_internal(dag, EvaluationScope::WholeDag, true, load_input)
            .map(|evaluated| evaluated.values);
    }
    reject_drop_roots(dag, roots)?;
    let live = live_mask_for_roots(dag, roots);
    eval_tensor_internal(
        dag,
        EvaluationScope::Roots { live: &live, roots },
        true,
        load_input,
    )
    .map(|evaluated| evaluated.values)
}

/// Evaluate exactly `roots` with strict loads.
///
/// The result holds exactly the named roots. Unlike
/// [`eval_tensor_roots_with_strict`], empty roots select no value node: only
/// effect nodes and random nodes that can trap by themselves execute.
pub fn eval_tensor_roots_exact<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    eval_tensor_roots_exact_with_result_claims(dag, roots, &[], load_input)
}

/// [`eval_tensor_roots_exact`] with invocation-local literal result claims
/// checked against the realized roots.
pub fn eval_tensor_roots_exact_with_result_claims<F>(
    dag: &Dag,
    roots: &[NodeId],
    result_claims: &[InheritedResultClaim],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    reject_drop_roots(dag, roots)?;
    let live = live_mask_for_roots(dag, roots);
    eval_tensor_internal_with_result_claims(
        dag,
        EvaluationScope::Roots { live: &live, roots },
        true,
        result_claims,
        load_input,
    )
    .map(|evaluated| evaluated.values)
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

    /// [05-OP-33] (chelis#2972): add-mode scatter combines each destination's
    /// base value and targeting updates by the canonical balanced tree at
    /// the operand width. i64 stays exact above 2^53; at f32
    /// `(2^24 + 1) + 1` rounds back to 2^24 at each node; i32 overflow traps
    /// under the operation's name, as compiled C does.
    ///
    /// Evidentiary status: REGRESSION TEST. At 08939bc0e the i64 row returns
    /// 2^53, the f32 row 2^24 + 2, and the i32 row names `scatter_add`.
    #[test]
    fn scatter_add_combines_at_the_operand_width() {
        let index = finalize_wide_int("test", Prim::Int64, vec![2], vec![0, 0]).unwrap();
        let above = (1_i64 << 53) + 1;
        let target = finalize_wide_int("test", Prim::Int64, vec![3], vec![above, 0, 0]).unwrap();
        let updates = finalize_wide_int("test", Prim::Int64, vec![2], vec![1, 1]).unwrap();
        let out = scatter_add(&target, &index, &updates, 0, 0, Prim::Int64).unwrap();
        assert_eq!(
            out.storage().to_i64_exact_vec(),
            Some(vec![above + 2, 0, 0])
        );

        let target = finalize_wide("test", Prim::F32, vec![2], vec![16_777_216.0, 0.0]).unwrap();
        let updates = finalize_wide("test", Prim::F32, vec![2], vec![1.0, 1.0]).unwrap();
        let out = scatter_add(&target, &index, &updates, 0, 0, Prim::F32).unwrap();
        assert_eq!(out.to_f64_lossy_vec(), vec![16_777_216.0, 0.0]);

        let index = finalize_wide_int("test", Prim::Int64, vec![1], vec![0]).unwrap();
        let target =
            finalize_wide_int("test", Prim::Int32, vec![1], vec![i64::from(i32::MAX)]).unwrap();
        let updates = finalize_wide_int("test", Prim::Int32, vec![1], vec![1]).unwrap();
        assert_eq!(
            scatter_add(&target, &index, &updates, 0, 0, Prim::Int32).unwrap_err(),
            "numeric trap: overflow in scatter at i32"
        );
    }

    /// [05-OP-53]: the condition's shape equals the shape of every branch it
    /// selects, and a branch it selects nowhere is neither read nor
    /// shape-checked. A uniform condition yields the selected branch when
    /// that branch is shaped like the condition, whatever the other branch's
    /// shape, and is refused when it is not; an empty one yields an empty
    /// result of its shape, and a mixed one still requires every shape to
    /// agree (decisions section 25).
    ///
    /// Evidentiary status: REGRESSION TEST for the accepted uniform and
    /// empty rows (each fails the shape check at 096daea8c) and for the
    /// refused uniform rows (each returns its selected branch at 1a026f823);
    /// DISPOSITION LOCK for the mixed rows.
    #[test]
    fn where_checks_only_the_branches_its_condition_selects() {
        let condition = |flags: &[i64]| {
            finalize_wide_int("where", Prim::Bool, vec![flags.len()], flags.to_vec()).unwrap()
        };
        let three = TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]);
        let two = TensorValue::from_vec(vec![2], vec![7.0, 8.0]);
        let selected = where_elementwise(&condition(&[1, 1]), &two, &three).unwrap();
        assert_eq!(selected.shape, vec![2]);
        assert_eq!(selected.to_f64_lossy_vec(), vec![7.0, 8.0]);
        let selected = where_elementwise(&condition(&[0, 0, 0]), &two, &three).unwrap();
        assert_eq!(selected.shape, vec![3]);
        assert_eq!(selected.to_f64_lossy_vec(), vec![1.0, 2.0, 3.0]);
        // spec/04 section 4.7: the condition disagrees with the one branch
        // it selects, a `Domain` trap in `where`.
        for (flags, context) in [
            (&[1, 1, 1][..], "lhs [3] has 3, rhs [2] has 2"),
            (&[0, 0][..], "lhs [2] has 2, rhs [3] has 3"),
        ] {
            assert_eq!(
                where_elementwise(&condition(flags), &two, &three).unwrap_err(),
                format!(
                    "where operands disagree at axis 0: {context}\n\
                     numeric trap: domain in where at i64"
                ),
                "{flags:?}"
            );
        }
        let empty = where_elementwise(&condition(&[]), &two, &three).unwrap();
        assert_eq!(empty.shape, vec![0]);
        assert_eq!(empty.prim(), Prim::F64);
        assert_eq!(
            where_elementwise(&condition(&[1, 0, 1]), &three, &two).unwrap_err(),
            "where operands disagree at axis 0: lhs [3] has 3, rhs [2] has 2\n\
             numeric trap: domain in where at i64"
        );
        let mixed = where_elementwise(
            &condition(&[1, 0, 1]),
            &three,
            &TensorValue::from_vec(vec![3], vec![4.0, 5.0, 6.0]),
        )
        .unwrap();
        assert_eq!(mixed.to_f64_lossy_vec(), vec![1.0, 5.0, 3.0]);
    }

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
    /// The `RiscOp::Shrink` arm reaches this branch for every runtime bound,
    /// so a program's inverted span reports this trap on eval as on C
    /// (chelis#1795).
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
        let decl = dag.declare("test");
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_ty,
            None,
        );
        // A shrink whose end (10) exceeds the eventual concrete extent (4).
        let shr = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_ty,
            None,
        );
        let shr = dag.add_node(
            decl,
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
    fn reduce_window_grad_max_splits_ties() {
        // A flat window: every position equals the max, so each tied
        // position receives g/k for its owning window.
        let x = TensorValue::from_vec(vec![3], vec![2.0, 2.0, 2.0]);
        let g = TensorValue::from_vec(vec![2], vec![4.0, 4.0]);
        let din = reduce_window_grad(&x, &g, ReduceWindowKind::Max, &[2], &[1], Prim::F64).unwrap();
        // windows [0,1] and [1,2]: pos0 += 2, pos1 += 2+2, pos2 += 2.
        assert_eq!(din.to_f64_lossy_vec(), vec![2.0, 4.0, 2.0]);
    }

    #[test]
    fn eval_add() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        let vals = eval_scalar(&dag, &UnordMap::new());
        assert!((vals[&c] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eval_vector_add() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            vec3_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            vec3_f32(),
            None,
        );
        let c = dag.add_node(decl, RiscOp::Add, vec![a, b], vec3_f32(), None);
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
        let decl = dag.declare("test");
        let ty = tensor_ty(&[2], Prim::Int64);
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            ty.clone(),
            None,
        );
        let b = dag.add_node(decl, RiscOp::Load { name: "b".into() }, vec![], ty, None);
        let out = dag.add_node(
            decl,
            RiscOp::Compare(ComparisonKind::CmpLt),
            vec![a, b],
            tensor_ty(&[2], Prim::Bool),
            None,
        );
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
        let decl = dag.declare("test");
        let ty = tensor_ty(&[1], Prim::Int64);
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            ty.clone(),
            None,
        );
        let one = dag.add_node(
            decl,
            RiscOp::Load { name: "one".into() },
            vec![],
            ty.clone(),
            None,
        );
        let add = dag.add_node(decl, RiscOp::Add, vec![a, one], ty.clone(), None);
        let mul = dag.add_node(decl, RiscOp::Mul, vec![add, one], ty, None);
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
        let decl = dag.declare("test");
        let values = dag.add_node(
            decl,
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[2, 4, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            decl,
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            tensor_ty(&[3], Prim::Int64),
            None,
        );
        let out = dag.add_node(
            decl,
            RiscOp::Gather {
                axis: 1,
                batch_rank: 0,
            },
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
        let decl = dag.declare("test");
        let target = dag.add_node(
            decl,
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[2, 3, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            decl,
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            tensor_ty(&[4], Prim::Int32),
            None,
        );
        let updates = dag.add_node(
            decl,
            RiscOp::Load {
                name: "updates".into(),
            },
            vec![],
            tensor_ty(&[2, 4, 2], Prim::F32),
            None,
        );
        let out = dag.add_node(
            decl,
            RiscOp::ScatterAdd {
                axis: 1,
                batch_rank: 0,
            },
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
        let decl = dag.declare("test");
        let in_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::F32,
        };
        let out_ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let x = dag.add_node(decl, RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
        let y = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let ty = vec3_f32();
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            ty.clone(),
            None,
        );
        let sum = dag.add_node(decl, RiscOp::Add, vec![x, x], ty.clone(), None);
        let dead = dag.add_node(decl, RiscOp::Add, vec![y, y], ty.clone(), None);
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3_f32(),
            None,
        );
        let err = eval_tensor_with_strict(&dag, |_| None).unwrap_err();
        assert!(err.contains("missing required input `x`"));
        assert_eq!(x, NodeId(0));
    }

    #[test]
    fn eval_root_scoped_strict_only_requires_live_inputs() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = vec3_f32();
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            ty.clone(),
            None,
        );
        let live = dag.add_node(decl, RiscOp::Add, vec![x, x], ty.clone(), None);
        let _dead = dag.add_node(decl, RiscOp::Add, vec![y, y], ty, None);

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

    /// Scoping is by declaration: `g`'s nodes are outside a selection of
    /// `main`, even one `main` reaches, and `main`'s are not. It may only
    /// ever drop work no selected root needs, so a node of `g` that the
    /// selected root reaches is still live.
    #[test]
    fn a_node_the_selection_reaches_is_live_whatever_its_declaration() {
        let mut dag = Dag::new();
        let g_decl = dag.declare("g");
        let main_decl = dag.declare("main");
        let ty = vec3_f32();
        let shared = dag.add_node(
            g_decl,
            RiscOp::Load {
                name: "shared".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let only_g = dag.add_node(
            g_decl,
            RiscOp::Load { name: "z".into() },
            vec![],
            ty.clone(),
            None,
        );
        let g = dag.add_node(g_decl, RiscOp::Add, vec![shared, only_g], ty.clone(), None);
        let main = dag.add_node(main_decl, RiscOp::Neg, vec![shared], ty, None);
        dag.add_root(g);
        dag.add_root(main);

        let owned = dag.outside_selection(&[main]);
        assert!(
            owned[g.0],
            "`g`'s own body is owned by the root nobody selected"
        );
        assert!(owned[only_g.0], "and so is the input only `g` reads");
        assert!(owned[shared.0], "as is every node of `g`'s declaration");
        assert!(!owned[main.0]);
        assert!(
            live_mask_for_roots(&dag, &[main])[shared.0],
            "but a node the selected root reaches is never dropped"
        );

        assert!(
            dag.outside_selection(&[main, g]).iter().all(|owned| !owned),
            "selecting every root leaves nothing unselected"
        );
        assert!(
            dag.outside_selection(&[]).iter().all(|owned| !owned),
            "and selecting nothing is a no-op, not an exclusion of everything"
        );
    }

    /// Liveness follows every backward edge from the selection, into any
    /// declaration: an abort of the unselected `g` that the selected root
    /// reaches only through a `shape_deps` or `result_claim_deps` edge
    /// still executes, although its seed is scoped out.
    #[test]
    fn ownership_follows_every_edge_liveness_propagates_over() {
        for edge in ["shape_dep", "result_claim_dep"] {
            let mut dag = Dag::new();
            let g_decl = dag.declare("g");
            let main_decl = dag.declare("main");
            let ty = vec3_f32();
            let bool_ty = TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision: Prim::Bool,
            };
            let z = dag.add_node(
                g_decl,
                RiscOp::Load { name: "z".into() },
                vec![],
                ty.clone(),
                None,
            );
            let cond = dag.add_node(
                g_decl,
                RiscOp::Compare(crate::dag::ComparisonKind::Gt),
                vec![z, z],
                bool_ty,
                None,
            );
            let fallback = dag.add_node(
                g_decl,
                RiscOp::synth_const(ty.precision, 0.0),
                vec![],
                ty.clone(),
                None,
            );
            let abort = dag.add_node(
                g_decl,
                RiscOp::GuardedFail {
                    message: "reached only by a dependency edge".to_string(),
                    trap_on_true: true,
                },
                vec![cond, fallback],
                ty.clone(),
                None,
            );
            // The unselected root consumes the abort through an ORDINARY
            // input edge, so it is an ancestor of the abort either way. Only
            // the dependency edge below can save the seed.
            let only_unselected = dag.add_node(
                g_decl,
                RiscOp::Load {
                    name: "only_g".into(),
                },
                vec![],
                ty.clone(),
                None,
            );
            let unselected = dag.add_node(
                g_decl,
                RiscOp::Mul,
                vec![abort, only_unselected],
                ty.clone(),
                None,
            );
            let x = dag.add_node(
                main_decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty.clone(),
                None,
            );
            let main = dag.add_node(main_decl, RiscOp::Add, vec![x, x], ty, None);
            // The selected root reaches the abort ONLY through this edge.
            match edge {
                "shape_dep" => dag.add_shape_dep(main, abort),
                _ => dag.add_result_claim_dep(main, abort),
            }
            dag.add_root(unselected);
            dag.add_root(main);

            let owned = dag.outside_selection(&[main]);
            assert!(owned[abort.0], "{edge}: the abort is `g`'s");
            assert!(
                live_mask_for_roots(&dag, &[main])[abort.0],
                "{edge}: and it must still execute"
            );
            assert!(
                owned[only_unselected.0],
                "{edge}: while the input only the unselected root reads is scoped out"
            );
            let _ = (unselected, z);
        }
    }

    /// chelis#2440's trap seed meets chelis#2476's scoping. The integer
    /// arithmetic in a declaration nobody selected can overflow, so the seed
    /// would mark it live and `resolve_load_inputs` would then demand that
    /// declaration's parameter — the chelis#991 shape, arriving through the
    /// trap seed instead of the abort one.
    ///
    /// The float twin is the control: it cannot trap, so it is never seeded
    /// and the scoping is not what keeps it dead.
    #[test]
    fn a_trapping_node_owned_by_an_unselected_root_is_not_this_evaluation_s_concern() {
        fn program(precision: Prim) -> (Dag, NodeId, NodeId, NodeId) {
            let ty = TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision,
            };
            let mut dag = Dag::new();
            let g_decl = dag.declare("g");
            let main_decl = dag.declare("main");
            let z = dag.add_node(
                g_decl,
                RiscOp::Load { name: "z".into() },
                vec![],
                ty.clone(),
                None,
            );
            let g = dag.add_node(g_decl, RiscOp::Mul, vec![z, z], ty.clone(), None);
            let x = dag.add_node(
                main_decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty.clone(),
                None,
            );
            let main = dag.add_node(main_decl, RiscOp::Add, vec![x, x], ty, None);
            dag.add_root(g);
            dag.add_root(main);
            (dag, z, g, main)
        }

        let (dag, z, g, main) = program(Prim::Int32);
        assert!(
            dag.trap_seeds()
                .is_observable_root(dag.get(g).expect("node")),
            "precondition: integer arithmetic must be a trapping node, or this \
             test passes for the wrong reason"
        );
        let live = live_mask_for_roots(&dag, &[main]);
        assert!(
            !live[g.0],
            "a trapping node owned by the unselected root is not seeded"
        );
        assert!(
            !live[z.0],
            "so its parameter never becomes a required input"
        );
        assert!(
            live_mask_for_roots(&dag, &[main, g])[g.0],
            "selecting its owner runs it, so the trap still occurs"
        );

        let (float_dag, float_z, float_g, float_main) = program(Prim::F32);
        assert!(
            !float_dag
                .trap_seeds()
                .is_observable_root(float_dag.get(float_g).expect("node")),
            "control: float arithmetic cannot trap, so it is never seeded at all"
        );
        let float_live = live_mask_for_roots(&float_dag, &[float_main]);
        assert!(!float_live[float_g.0] && !float_live[float_z.0]);
    }

    /// With no declared roots the whole graph is the program, so the scoping
    /// is a no-op and a discarded trapping node is still seeded — it must
    /// execute, and its input is genuinely required.
    #[test]
    fn a_rootless_dag_still_seeds_its_discarded_trapping_node() {
        let mut dag = Dag::new();
        let decl = dag.declare("main");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::Int32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            ty.clone(),
            None,
        );
        let live_add = dag.add_node(decl, RiscOp::Add, vec![x, x], ty.clone(), None);
        let discarded = dag.add_node(decl, RiscOp::Add, vec![y, y], ty, None);

        let live = live_mask_for_roots(&dag, &[live_add]);
        assert!(
            live[discarded.0] && live[y.0],
            "nothing was deselected, so the trap seed still reaches it"
        );
    }

    /// chelis#2413: a graph returning its parameter `v` beside a discarded
    /// integer `add(v, v)` runs discarded work, so no lane may implement it
    /// by its root's value alone. The float twin cannot trap and runs none,
    /// and neither does a trapping node the root's value reads.
    #[test]
    fn a_discarded_trapping_node_is_work_beyond_the_root_values() {
        let returns_parameter_beside = |precision: Prim, reads_the_add: bool| {
            let mut dag = Dag::new();
            let decl = dag.declare("g");
            let ty = TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision,
            };
            let v = dag.add_node(
                decl,
                RiscOp::Load { name: "v".into() },
                vec![],
                ty.clone(),
                None,
            );
            let sum = dag.add_node(decl, RiscOp::Add, vec![v, v], ty.clone(), None);
            let root = if reads_the_add {
                sum
            } else {
                dag.add_node(decl, RiscOp::Copy, vec![v], ty, None)
            };
            dag.add_root(root);
            dag
        };
        let integer = returns_parameter_beside(Prim::Int32, false);
        assert!(runs_discarded_work(&integer, integer.roots()));
        let float = returns_parameter_beside(Prim::F32, false);
        assert!(
            !runs_discarded_work(&float, float.roots()),
            "control: a float add cannot trap, so nothing beyond the root runs"
        );
        let read = returns_parameter_beside(Prim::Int32, true);
        assert!(
            !runs_discarded_work(&read, read.roots()),
            "control: a trapping node the root reads is the root's own work"
        );
    }

    /// chelis#2413 put a trapping draw beside the `[05-OP-68]` abort in the
    /// same seed, and chelis#2476's rebase scoped both together: they are one
    /// class under `spec/06` §5.2, so leaving the newer one unscoped would
    /// re-arm #2476 along the second seed — a draw inside an uncalled `def`
    /// would demand that def's parameters, and then trap on behalf of code
    /// the caller excluded.
    ///
    /// The graph holds one declaration per root it models. The
    /// precondition assert keeps the test honest: a draw that cannot trap is
    /// never seeded, and the assertions below would then pass vacuously.
    #[test]
    fn a_trapping_draw_owned_by_an_unselected_root_is_not_this_evaluation_s_concern() {
        let mut dag = Dag::new();
        let g_decl = dag.declare("g");
        let main_decl = dag.declare("main");
        let ty = vec3_f32();
        let key_ty = TensorType {
            dims: vec![],
            precision: Prim::Key,
        };
        // `g(z, rate, k)`: an uncalled declaration holding a trapping draw.
        let z = dag.add_node(
            g_decl,
            RiscOp::Load { name: "z".into() },
            vec![],
            ty.clone(),
            None,
        );
        let rate = dag.add_node(
            g_decl,
            RiscOp::Load {
                name: "rate".into(),
            },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        // The draw validates its own runtime rate, so it can trap by itself.
        let k = dag.add_node(
            g_decl,
            RiscOp::Load { name: "k".into() },
            vec![],
            key_ty,
            None,
        );
        let draw = dag.add_node(g_decl, RiscOp::Dropout, vec![z, rate, k], ty.clone(), None);
        let x = dag.add_node(
            main_decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let main = dag.add_node(main_decl, RiscOp::Add, vec![x, x], ty, None);
        dag.add_root(draw);
        dag.add_root(main);

        assert!(
            dag.random_node_may_trap(dag.get(draw).expect("draw node")),
            "precondition: this draw must be one that can trap by itself, or \
             the assertions below pass for the wrong reason"
        );

        let selected = live_mask_for_roots(&dag, &[main]);
        assert!(
            !selected[draw.0],
            "a trapping draw owned by the unselected root `g` must not be seeded"
        );
        for (name, id) in [("z", z), ("rate", rate), ("k", k)] {
            assert!(
                !selected[id.0],
                "and `{name}` must stay dead, so it never becomes a required input"
            );
        }

        assert!(
            live_mask_for_roots(&dag, &[main, draw])[draw.0],
            "selecting its owner runs it, so the [05-OP-37] trap still fires"
        );
    }

    /// chelis#2476. A lowered program makes every top-level `def` a DAG
    /// root, so an uncalled `def g(z)` puts `z` in the DAG as a `Load`
    /// indistinguishable from the selected root's own input. Value
    /// reachability ignores it; a seed must too, or `g`'s parameter becomes
    /// a required input for a program that never calls `g`.
    ///
    /// The abort is the seed available to test this today. Both directions
    /// are asserted: excluded when `g` is not selected, live when it is.
    #[test]
    fn an_abort_owned_by_an_unselected_root_is_not_this_evaluation_s_concern() {
        fn program() -> (Dag, NodeId, NodeId) {
            let mut dag = Dag::new();
            let g_decl = dag.declare("g");
            let main_decl = dag.declare("main");
            let ty = vec3_f32();
            // `g(z)`: an uncalled declaration whose body aborts.
            let z = dag.add_node(
                g_decl,
                RiscOp::Load { name: "z".into() },
                vec![],
                ty.clone(),
                None,
            );
            let cond = dag.add_node(
                g_decl,
                RiscOp::Compare(crate::dag::ComparisonKind::Gt),
                vec![z, z],
                TensorType {
                    dims: vec![DimInfo::Lit(3)],
                    precision: Prim::Bool,
                },
                None,
            );
            let fallback = dag.add_node(
                g_decl,
                RiscOp::synth_const(ty.precision, 0.0),
                vec![],
                ty.clone(),
                None,
            );
            let g = dag.add_node(
                g_decl,
                RiscOp::GuardedFail {
                    message: "uncalled".to_string(),
                    trap_on_true: true,
                },
                vec![cond, fallback],
                ty.clone(),
                None,
            );
            // `main(x)`: the root actually being evaluated.
            let x = dag.add_node(
                main_decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty.clone(),
                None,
            );
            let main = dag.add_node(main_decl, RiscOp::Add, vec![x, x], ty, None);
            dag.add_root(g);
            dag.add_root(main);
            (dag, g, main)
        }

        let (dag, g, main) = program();
        let selected = live_mask_for_roots(&dag, &[main]);
        assert!(
            !selected[g.0],
            "an abort owned by the unselected root `g` must not be seeded"
        );
        assert!(
            selected.iter().enumerate().all(|(id, live)| !live
                || !matches!(&dag.nodes()[id].op, RiscOp::Load { name } if name.as_str() == "z")),
            "and `z` must stay dead, so it is never a required input"
        );

        let both = live_mask_for_roots(&dag, &[main, g]);
        assert!(
            both[g.0],
            "selecting `g` runs it, so its abort is seeded as [05-OP-68] requires"
        );
    }

    /// A DISCARDED abort is the case that matters for chelis#2368: nothing
    /// consumes it and it is no root, so reachability never reaches it, and
    /// its declaration decides. Discarded in the selected `main`, it is
    /// seeded; the unselected `g`'s own body is not.
    #[test]
    fn an_abort_owned_by_no_root_is_seeded_for_every_selection() {
        let mut dag = Dag::new();
        let g_decl = dag.declare("g");
        let main_decl = dag.declare("main");
        let ty = vec3_f32();
        let x = dag.add_node(
            main_decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let cond = dag.add_node(
            main_decl,
            RiscOp::Compare(crate::dag::ComparisonKind::Gt),
            vec![x, x],
            TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision: Prim::Bool,
            },
            None,
        );
        let fallback = dag.add_node(
            main_decl,
            RiscOp::synth_const(ty.precision, 0.0),
            vec![],
            ty.clone(),
            None,
        );
        // Discarded: nothing consumes it and it is not a root.
        let abort = dag.add_node(
            main_decl,
            RiscOp::GuardedFail {
                message: "discarded".to_string(),
                trap_on_true: true,
            },
            vec![cond, fallback],
            ty.clone(),
            None,
        );
        let other = dag.add_node(
            g_decl,
            RiscOp::Load { name: "z".into() },
            vec![],
            ty.clone(),
            None,
        );
        let unselected = dag.add_node(g_decl, RiscOp::Mul, vec![other, other], ty.clone(), None);
        let main = dag.add_node(main_decl, RiscOp::Add, vec![x, x], ty, None);
        dag.add_root(unselected);
        dag.add_root(main);

        let live = live_mask_for_roots(&dag, &[main]);
        assert!(
            live[abort.0],
            "a discarded abort of the selected declaration is seeded"
        );
        assert!(
            !live[unselected.0],
            "while the unselected root's own body is still excluded"
        );
    }

    /// chelis#991: a dependency package may contribute generic declarations
    /// outside the selected eval root. Their symbolic inputs are dead and
    /// must not become invented top-level requirements for the live result.
    #[test]
    fn eval_root_scoped_ignores_unrelated_dead_symbolic_input() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let dead_ty = TensorType {
            dims: vec![DimInfo::Named("k".to_string(), None)],
            precision: Prim::F32,
        };
        let _dead = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            dead_ty,
            None,
        );
        let live = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let symbolic = TensorType {
            dims: vec![DimInfo::Named("k".to_string(), None)],
            precision: Prim::F32,
        };
        let _dead = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            symbolic.clone(),
            None,
        );
        let live = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            symbolic,
            None,
        );

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
        let decl = dag.declare("test");
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_ty.clone(),
            None,
        );
        let one = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let ones = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_ty.clone(),
            None,
        );
        let one = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let ones = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("k".to_string(), None)],
            precision: Prim::F32,
        };
        let _unrelated = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            sym_ty.clone(),
            None,
        );
        let required = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_ty.clone(),
            None,
        );
        let one = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let ones = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let square_ty = TensorType {
            dims: vec![
                DimInfo::Named("n".to_string(), None),
                DimInfo::Named("n".to_string(), None),
            ],
            precision: Prim::F32,
        };
        let shape_source = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            square_ty,
            None,
        );
        let one = dag.add_node(
            decl,
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
            decl,
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

    // [05-RNG-1] transcribed from the spec text, never the kernel.
    fn spec_uniform_splitmix64(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The `[05-OP-8]` kernel over a shape, bounds at `prim` and one draw key.
    fn uniform_like(
        shape: &[usize],
        low: f64,
        high: f64,
        key: RandomKey,
        prim: Prim,
    ) -> Result<TensorValue, String> {
        let bound = |value| chelis_types::scalar_from_f64("uniform_like", prim, value);
        let prepared = PreparedUniformLike::new(
            prim,
            numel(shape),
            bound(low).map_err(|trap| trap.to_string())?,
            bound(high).map_err(|trap| trap.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let storage = prepared.apply(key).map_err(|error| error.to_string())?;
        Ok(TensorValue::from_storage(shape.to_vec(), storage))
    }

    /// `[05-OP-69]`'s key of the seed 42.
    fn spec_uniform_key42() -> RandomKey {
        RandomKey::from_seed(chelis_types::scalar_from_i64("test", Prim::Int64, 42).unwrap())
            .unwrap()
    }

    /// [05-RNG-2]'s `word` and [05-RNG-1]'s unit, transcribed from the spec text.
    fn spec_uniform_unit(key_bits: u64, index: u64) -> f64 {
        let word =
            spec_uniform_splitmix64(key_bits ^ spec_uniform_splitmix64(index).rotate_left(41));
        (word >> 11) as f64 / (1_u64 << 53) as f64
    }

    #[test]
    fn uniform_like_f32_affine_mirrors_c_f32_sampler() {
        // chelis#770: the affine is a single correctly-rounded FMA
        // (`span_f.mul_add(unit_f, low_f)`), conforming to the compiled C
        // sampler `chelis_uniform_sample_f32`. Key `key_from_seed(42)`, shape
        // [12], [2,5); the bits are exact-rational evaluations of [05-RNG-2]'s
        // word (`key_ref.py`) and [05-OP-8] (`keys-b-h9-probes/port_ref.py`).
        let key = spec_uniform_key42();
        let out = uniform_like(&[12], 2.0, 5.0, key, Prim::F32).unwrap();
        let bits = |index: usize| (out.to_f64_lossy_vec()[index] as f32).to_bits();
        // elem[11]: where an f64 affine rounded to f32 lands 1 ULP away.
        assert_eq!(bits(11), 0x406d_6dc6);
        let old_f64_affine = 2.0 + (5.0 - 2.0) * spec_uniform_unit(key.bits(), 11);
        assert_eq!((old_f64_affine as f32).to_bits(), 0x406d_6dc5);
        // elem[10]: where a plain two-rounding `low_f + span_f * unit_f`
        // disagrees by 1 ULP, the bit the compiled C lane would flip between
        // `-ffp-contract=fast` and `-ffp-contract=off` without `fmaf`.
        assert_eq!(bits(10), 0x4034_fb45);
        let two_rounding_10 =
            2.0f32 + (5.0f32 - 2.0f32) * (spec_uniform_unit(key.bits(), 10) as f32);
        assert_eq!(two_rounding_10.to_bits(), 0x4034_fb44);
        assert_eq!(bits(7), 0x408f_92d9);
    }

    #[test]
    fn uniform_like_f32_affine_negative_range_is_f32() {
        // chelis#770: negative range, key `key_from_seed(42)`, index 3, [-3, -1).
        let out = uniform_like(&[8], -3.0, -1.0, spec_uniform_key42(), Prim::F32).unwrap();
        assert_eq!(
            out.to_f64_lossy_vec()[3].to_bits(),
            (f32::from_bits(0xc027_e1a4) as f64).to_bits(),
        );
    }

    #[test]
    fn uniform_like_f64_uses_the_f64_affine() {
        let key = spec_uniform_key42();
        let out = uniform_like(&[8], 2.0, 5.0, key, Prim::F64).unwrap();
        let expected = (5.0f64 - 2.0).mul_add(spec_uniform_unit(key.bits(), 4), 2.0);
        assert_eq!(expected.to_bits(), 0x4005_8b85_e511_043a);
        assert_eq!(out.to_f64_lossy_vec()[4].to_bits(), expected.to_bits());
        // 0x402c5c2f is element 4 of the f32 draw under the same key.
        assert_ne!(
            out.to_f64_lossy_vec()[4].to_bits(),
            (f32::from_bits(0x402c_5c2f) as f64).to_bits(),
            "f64 samples must not be widened f32 values"
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
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
        let y = dag.add_node(decl, op, vec![x], out_ty, None);
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
            let decl = dag.declare("test");
            let x = dag.add_node(
                decl,
                RiscOp::synth_const(row_f32(3).precision, 1.0),
                vec![],
                row_f32(3),
                None,
            );
            dag.add_node(decl, op, vec![x], scalar_f32(), None);
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
    // halting. [05-OP-64] requires the `DivZero` trap in BOTH the
    // evaluator and the C backend; these pin that this lane
    // now fails closed on integer operands, while float `floor_div` keeps the
    // IEEE no-trap semantics §2.1 also mandates.

    fn int_div_dag(op: RiscOp, precision: Prim) -> Dag {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            tensor_ty(&[2], precision),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            tensor_ty(&[2], precision),
            None,
        );
        let out = dag.add_node(decl, op, vec![a, b], tensor_ty(&[2], precision), None);
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            tensor_ty(&[2], Prim::Int32),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            tensor_ty(&[2], Prim::Int32),
            None,
        );
        let d = dag.add_node(
            decl,
            RiscOp::FloorDiv,
            vec![a, b],
            tensor_ty(&[2], Prim::Int32),
            None,
        );
        let e = dag.add_node(
            decl,
            RiscOp::Add,
            vec![d, a],
            tensor_ty(&[2], Prim::Int32),
            None,
        );
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
            "sum",
            &input,
            0,
            TensorReduceOp::Sum {
                accumulator: Prim::Int32,
                result: Prim::Int32,
            },
            Prim::Int32,
        )
        .expect_err("lane0 + lane1 overflows the i32 accumulator");
        assert_eq!(err, "numeric trap: overflow in sum at i32");

        let control = exact_tensor(
            Prim::Int32,
            RawTensor::Int(vec![i64::from(i32::MAX) - 1, 1, -1]),
        );
        let output = reduce(
            "sum",
            &control,
            0,
            TensorReduceOp::Sum {
                accumulator: Prim::Int32,
                result: Prim::Int32,
            },
            Prim::Int32,
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

/// chelis#828 work class 1: values reclaimed after their last use.
///
/// The receipt is a counted quantity, not a wall clock: `peak_live_values` is
/// the most entries the evaluator's value map ever held at once, sampled after
/// each node's own value lands and before that step's reclamation runs. Under
/// the all-values contract that number is the executed node count; under a
/// root-scoped run it must be the live working set.
///
/// The negative controls matter more than the receipt, and the two reader
/// families fail differently. An operand read indexes `values[&node.inputs[i]]`
/// directly, so freeing one early panics. The `Const` shape-dependency
/// fallback does not: it ends in a default-empty unwrap, so a freed dependency
/// silently becomes shape `[]` and the Const materializes one element. That
/// quiet case is the one worth an evaluator-level test, and
/// `a_const_sized_from_a_freed_shape_dependency_is_silently_the_wrong_shape`
/// is it.
///
/// The remaining edges are covered at the schedule level only. `shape_deps`
/// now has both; `result_claim_deps` has no evaluator test because this lane
/// never reads it, and it is in the reader set as a deliberate conservatism;
/// the three guard edges raise rather than answer wrongly, and building a real
/// guard site needs the axis-source derivation rather than a hand-built DAG.
#[cfg(test)]
mod value_reclamation {
    use super::*;
    use crate::dag::RiscOp;
    use chelis_types::types::Prim;

    fn vec3() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        }
    }

    fn x_input() -> TensorValue {
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])
    }

    fn load_x() -> impl FnMut(&str) -> Option<TensorValue> {
        |name: &str| (name == "x").then(x_input)
    }

    /// `Load x` followed by `links` chained `Neg` nodes. Every intermediate
    /// has exactly one consumer and is dead the moment that consumer runs, so
    /// the live working set never exceeds two values however long the chain.
    fn neg_chain(links: usize) -> (Dag, NodeId) {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let mut last = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3(),
            None,
        );
        for _ in 0..links {
            last = dag.add_node(decl, RiscOp::Neg, vec![last], vec3(), None);
        }
        dag.add_root(last);
        (dag, last)
    }

    fn run(dag: &Dag, scope: EvaluationScope<'_>) -> EvaluatedValues {
        eval_tensor_internal(dag, scope, true, load_x()).expect("the fixture evaluates")
    }

    fn elements(values: &UnordMap<NodeId, TensorValue>, id: NodeId) -> Vec<f64> {
        values
            .get(&id)
            .unwrap_or_else(|| panic!("node {} is absent from the result", id.0))
            .to_f64_lossy_vec()
    }

    /// The receipt.
    ///
    /// This test was run against this same tree with the reclamation
    /// neutralized (`free_schedule` forced to `None`), so it is known to fail
    /// without it. The root-scoped peak then equalled the executed node count:
    /// `peak_live_values` was 65 for the 64-link chain and 129 for the
    /// 128-link chain, against the 2 asserted below. (The 128 figure needed
    /// its own run, because the loop aborts at 64 otherwise.) The all-values
    /// assertions passed in both configurations, which is the point of
    /// keeping them here: they pin the contract the tracker asked to
    /// preserve, and they are what measures 195 and 387 elements.
    #[test]
    fn root_scoped_peak_tracks_the_working_set_not_the_executed_node_count() {
        for links in [64usize, 128] {
            let (dag, root) = neg_chain(links);
            let live = live_mask_for_roots(&dag, &[root]);

            let retaining = run(&dag, EvaluationScope::MaskedAllValues(&live));
            let reclaiming = run(
                &dag,
                EvaluationScope::Roots {
                    live: &live,
                    roots: &[root],
                },
            );

            assert_eq!(
                retaining.peak_live_values,
                links + 1,
                "the all-values contract holds every executed node's value"
            );
            assert_eq!(
                retaining.peak_live_elements,
                3 * (links + 1),
                "and every one of their elements"
            );

            assert_eq!(
                reclaiming.peak_live_values, 2,
                "a {links}-link chain is never more than two values wide"
            );
            assert_eq!(reclaiming.peak_live_elements, 6);
            assert_eq!(
                reclaiming.values.len(),
                1,
                "only the selected root survives the run"
            );

            // Identical, not merely close: the same chain of exact f32
            // negations either way.
            let expected = if links % 2 == 0 {
                vec![1.0, 2.0, 3.0]
            } else {
                vec![-1.0, -2.0, -3.0]
            };
            assert_eq!(elements(&reclaiming.values, root), expected);
            assert_eq!(
                elements(&reclaiming.values, root),
                elements(&retaining.values, root)
            );
        }
    }

    /// `x` feeds the head of a chain AND its tail, four steps later. Freeing
    /// it when the first consumer runs would panic in `Add`'s operand index;
    /// freeing it at the right step still leaves the answer exact.
    fn diamond() -> (Dag, [NodeId; 5]) {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3(),
            None,
        );
        let a = dag.add_node(decl, RiscOp::Neg, vec![x], vec3(), None);
        let b = dag.add_node(decl, RiscOp::Neg, vec![a], vec3(), None);
        let c = dag.add_node(decl, RiscOp::Neg, vec![b], vec3(), None);
        let d = dag.add_node(decl, RiscOp::Add, vec![c, x], vec3(), None);
        dag.add_root(d);
        (dag, [x, a, b, c, d])
    }

    #[test]
    fn a_value_with_a_later_second_consumer_outlives_its_first() {
        let (dag, [x, _a, _b, _c, d]) = diamond();
        let live = live_mask_for_roots(&dag, &[d]);

        let reclaiming = run(
            &dag,
            EvaluationScope::Roots {
                live: &live,
                roots: &[d],
            },
        );
        let retaining = run(&dag, EvaluationScope::MaskedAllValues(&live));

        // -x + x, elementwise and exact.
        assert_eq!(elements(&reclaiming.values, d), vec![0.0, 0.0, 0.0]);
        assert_eq!(
            elements(&reclaiming.values, d),
            elements(&retaining.values, d)
        );
        assert_eq!(reclaiming.values.len(), 1);
        assert!(
            reclaiming.values.get(&x).is_none(),
            "x is not a selected root and does not survive the run"
        );
        // Three values are live while the chain passes the retained `x`:
        // `x` itself, the chain's previous link, and the new one.
        assert_eq!(reclaiming.peak_live_values, 3);
        assert_eq!(retaining.peak_live_values, 5);
    }

    #[test]
    fn a_selected_root_that_is_also_an_intermediate_is_kept() {
        let (dag, [_x, _a, b, _c, d]) = diamond();
        let roots = [b, d];
        let live = live_mask_for_roots(&dag, &roots);

        let reclaiming = run(
            &dag,
            EvaluationScope::Roots {
                live: &live,
                roots: &roots,
            },
        );
        let retaining = run(&dag, EvaluationScope::MaskedAllValues(&live));

        // `b` is `-(-x)`, and it still feeds `c`.
        assert_eq!(elements(&reclaiming.values, b), vec![1.0, 2.0, 3.0]);
        assert_eq!(elements(&reclaiming.values, d), vec![0.0, 0.0, 0.0]);
        assert_eq!(
            elements(&reclaiming.values, b),
            elements(&retaining.values, b)
        );
        assert_eq!(
            elements(&reclaiming.values, d),
            elements(&retaining.values, d)
        );
        assert_eq!(
            reclaiming.values.len(),
            2,
            "both selected roots survive and nothing else does"
        );
        assert_eq!(reclaiming.peak_live_values, 4);
    }

    #[test]
    fn overlapping_root_cones_share_one_intermediate() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3(),
            None,
        );
        let shared = dag.add_node(decl, RiscOp::Neg, vec![x], vec3(), None);
        let left = dag.add_node(decl, RiscOp::Neg, vec![shared], vec3(), None);
        let right = dag.add_node(decl, RiscOp::Add, vec![shared, shared], vec3(), None);
        dag.add_root(left);
        dag.add_root(right);
        let roots = [left, right];
        let live = live_mask_for_roots(&dag, &roots);

        let reclaiming = run(
            &dag,
            EvaluationScope::Roots {
                live: &live,
                roots: &roots,
            },
        );
        let retaining = run(&dag, EvaluationScope::MaskedAllValues(&live));

        assert_eq!(elements(&reclaiming.values, left), vec![1.0, 2.0, 3.0]);
        assert_eq!(elements(&reclaiming.values, right), vec![-2.0, -4.0, -6.0]);
        assert_eq!(
            elements(&reclaiming.values, right),
            elements(&retaining.values, right)
        );
        assert_eq!(reclaiming.values.len(), 2);
        assert!(
            reclaiming.values.get(&shared).is_none(),
            "the shared intermediate is freed once the second cone has read it"
        );
    }

    /// The preserved contract. Every entrypoint that names no roots returns
    /// one entry per executed node, exactly as before chelis#828.
    #[test]
    fn the_all_values_entrypoints_still_return_every_executed_node() {
        let (dag, root) = neg_chain(8);
        let expected = dag.len();

        let with_strict = eval_tensor_with_strict(&dag, load_x()).expect("whole-DAG eval");
        assert_eq!(with_strict.len(), expected);

        let lenient = eval_tensor_with(&dag, load_x()).expect("whole-DAG eval");
        assert_eq!(lenient.len(), expected);

        let mut inputs = UnordMap::new();
        inputs.insert("x".to_string(), x_input());
        assert_eq!(eval_tensor(&dag, &inputs).expect("eval").len(), expected);

        // Empty roots select the whole DAG, so they keep the same contract.
        let empty_roots = eval_tensor_roots_with_strict(&dag, &[], load_x()).expect("eval");
        assert_eq!(empty_roots.len(), expected);

        // And a named root returns only that root, with the same answer.
        let scoped = eval_tensor_roots_with_strict(&dag, &[root], load_x()).expect("eval");
        assert_eq!(scoped.len(), 1);
        assert_eq!(elements(&scoped, root), elements(&with_strict, root));
    }

    /// The result-claims entry point has no empty-roots fallback, unlike its
    /// three siblings. Its doc comment now says so; this is the run behind
    /// that sentence, so the difference is locked rather than asserted from
    /// reading the branch that is missing.
    #[test]
    fn empty_roots_select_nothing_for_the_result_claims_entry_point() {
        let (dag, root) = neg_chain(4);

        let empty = eval_tensor_roots_exact_with_result_claims(&dag, &[], &[], load_x())
            .expect("empty roots are not an error");
        assert!(
            empty.is_empty(),
            "every node is masked off, so nothing executes"
        );

        // Its siblings take the whole DAG instead, which is the rule this
        // entry point does NOT share.
        let whole = eval_tensor_roots_with_strict(&dag, &[], load_x()).expect("whole DAG");
        assert_eq!(whole.len(), dag.len());

        // And with a root named, it returns that root alone.
        let scoped = eval_tensor_roots_exact_with_result_claims(&dag, &[root], &[], load_x())
            .expect("named root");
        assert_eq!(scoped.len(), 1);
        assert_eq!(elements(&scoped, root), vec![1.0, 2.0, 3.0]);
    }

    /// The one reader edge whose loss is SILENT, exercised through the
    /// evaluator rather than through the schedule.
    ///
    /// `Const` sizes itself from `concrete_shape_with`, and when that fails it
    /// falls back to a `shape_deps` value's realized shape. That fallback ends
    /// in a default-empty unwrap, so a freed shape dependency does not raise:
    /// the shape silently becomes `[]` and the Const materializes one element
    /// instead of the operand's width. Nothing downstream complains, because a
    /// rank-zero operand is a legal broadcast.
    ///
    /// `x` is reachable ONLY through the shape dependency here. No node takes
    /// it as an operand, so `inputs` alone gives it no reader and the schedule
    /// frees it at its own production step.
    ///
    /// Measured on this tree with the extra reader edges removed from
    /// `value_free_schedule` (the `shape_deps`, `result_claim_deps` and guard
    /// blocks skipped, which is the reviewer's M2 mutation): this test failed
    /// with `left: ([], [-2.0])` against `right: ([3], [-2.0, -2.0, -2.0])`.
    /// With the edges in place it passes. It is the evaluator-level binding
    /// the five schedule tests below do not provide.
    ///
    /// `Named("*", None)` is what keeps the fallback reachable:
    /// `required_symbolic_dims` skips that name, so no symbolic binding runs,
    /// `runtime_dims` stays empty, and `concrete_shape_with` fails as the
    /// fallback's own precondition requires.
    #[test]
    fn a_const_sized_from_a_freed_shape_dependency_is_silently_the_wrong_shape() {
        let starred = TensorType {
            dims: vec![DimInfo::Named("*".to_string(), None)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3(),
            None,
        );
        let sized = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 2.0),
            vec![],
            starred.clone(),
            None,
        );
        dag.add_shape_dep(sized, x);
        let out = dag.add_node(decl, RiscOp::Neg, vec![sized], starred, None);
        dag.add_root(out);

        let live = live_mask_for_roots(&dag, &[out]);
        let reclaiming = run(
            &dag,
            EvaluationScope::Roots {
                live: &live,
                roots: &[out],
            },
        );

        let value = reclaiming
            .values
            .get(&out)
            .expect("the selected root survives");
        // Shape and elements in ONE assertion: asserting the shape first would
        // pre-empt the element vector, and then this comment could only report
        // half of what a losing run prints.
        assert_eq!(
            (value.shape.clone(), value.to_f64_lossy_vec()),
            (vec![3], vec![-2.0, -2.0, -2.0]),
            "the Const takes its width from the shape dependency's realized shape"
        );

        // The same answer through the all-values lane, which frees nothing.
        let retaining = run(&dag, EvaluationScope::MaskedAllValues(&live));
        assert_eq!(elements(&retaining.values, out), value.to_f64_lossy_vec());
    }

    // ------------------------------------------------------------------
    // The reader edges that are not `DagNode::inputs`. Each test pairs the
    // extra edge against the same graph without it, so it shows both that
    // the edge extends the lifetime and that the lifetime would otherwise
    // end early. Without the pair, the assertion could pass for a schedule
    // that never frees anything.
    // ------------------------------------------------------------------

    fn node_order(dag: &Dag) -> Vec<NodeId> {
        dag.nodes().iter().map(|node| node.id).collect()
    }

    /// The step index at which `id` is scheduled to be freed, or `None` when
    /// the schedule keeps it to the end.
    fn freed_at(schedule: &[Vec<NodeId>], id: NodeId) -> Option<usize> {
        schedule.iter().position(|step| step.contains(&id))
    }

    fn schedule_for(dag: &Dag, retain: &[NodeId]) -> Vec<Vec<NodeId>> {
        value_free_schedule(dag, &node_order(dag), None, &UnordMap::new(), retain)
    }

    /// A chain plus one spectator node, so a graph without the extra edge
    /// frees the spectator's source at step 1 and a graph with it does not.
    fn spectator_graph() -> (Dag, NodeId, NodeId, NodeId) {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3(),
            None,
        );
        let a = dag.add_node(decl, RiscOp::Neg, vec![x], vec3(), None);
        let late = dag.add_node(decl, RiscOp::Neg, vec![a], vec3(), None);
        dag.add_root(late);
        (dag, x, a, late)
    }

    #[test]
    fn without_an_extra_edge_the_source_is_freed_at_its_only_consumer() {
        let (dag, x, _a, late) = spectator_graph();
        let schedule = schedule_for(&dag, &[late]);
        assert_eq!(
            freed_at(&schedule, x),
            Some(1),
            "x's only reader is the node at index 1"
        );
    }

    #[test]
    fn a_shape_dependency_extends_the_lifetime_it_reads() {
        let (mut dag, x, _a, late) = spectator_graph();
        dag.add_shape_dep(late, x);
        let schedule = schedule_for(&dag, &[late]);
        assert_eq!(
            freed_at(&schedule, x),
            Some(2),
            "a Const sizes itself from a shape dependency's realized shape, \
             so the dependency is a read"
        );
    }

    #[test]
    fn a_result_claim_dependency_extends_the_lifetime_it_reads() {
        let (mut dag, x, _a, late) = spectator_graph();
        dag.add_result_claim_dep(late, x);
        let schedule = schedule_for(&dag, &[late]);
        assert_eq!(
            freed_at(&schedule, x),
            Some(2),
            "the producer discharges its obligation against the witness's value"
        );
    }

    fn guard_claim(
        canonical: crate::axis_sources::CanonicalExtent,
        observed: crate::axis_sources::LocalGuardObservation,
        activation: crate::axis_sources::GuardActivation,
    ) -> crate::axis_sources::LocalGuardClaim {
        crate::axis_sources::LocalGuardClaim {
            claim: "n".to_string(),
            canonical,
            op: "expand",
            observed,
            activation,
            source: None,
        }
    }

    fn schedule_with_guard(
        dag: &Dag,
        site: NodeId,
        claim: crate::axis_sources::LocalGuardClaim,
        retain: &[NodeId],
    ) -> Vec<Vec<NodeId>> {
        let mut sites = UnordMap::new();
        sites.insert(site, vec![(0usize, claim)]);
        value_free_schedule(dag, &node_order(dag), None, &sites, retain)
    }

    #[test]
    fn a_guard_activation_extends_the_lifetime_it_reads() {
        let (mut dag, x, a, late) = spectator_graph();
        // The guard at `late` is checked under the activation of the node
        // its claim is about, `a`'s: `x`, which `a` already reads earlier.
        dag.node_mut(a).expect("a").owner.activation = Some(x);
        let activation = crate::axis_sources::GuardActivation::reading(&dag, a).unwrap();
        let schedule = schedule_with_guard(
            &dag,
            late,
            guard_claim(
                crate::axis_sources::CanonicalExtent::Resolved(3),
                crate::axis_sources::LocalGuardObservation::RealizedExtent,
                activation,
            ),
            &[late],
        );
        assert_eq!(
            freed_at(&schedule, x),
            Some(2),
            "`local_guard_is_active` reads the activation node's value at the site"
        );
    }

    #[test]
    fn a_guard_witness_extends_the_lifetime_it_reads() {
        let (dag, x, _a, late) = spectator_graph();
        let schedule = schedule_with_guard(
            &dag,
            late,
            guard_claim(
                crate::axis_sources::CanonicalExtent::Witness(x),
                crate::axis_sources::LocalGuardObservation::RealizedExtent,
                crate::axis_sources::GuardActivation::sizing(&dag, late, 0).unwrap(),
            ),
            &[late],
        );
        assert_eq!(
            freed_at(&schedule, x),
            Some(2),
            "`local_guard_verdict` reads the declaring witness's scalar"
        );
    }

    /// A graph whose penultimate node is a same-shape producer over `x`, so
    /// an agreement derived from it names `x` while the final node does not
    /// read `x` through any operand slot.
    fn agreement_graph() -> (Dag, NodeId, NodeId, NodeId) {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3(),
            None,
        );
        let a = dag.add_node(decl, RiscOp::Neg, vec![x], vec3(), None);
        let mix = dag.add_node(decl, RiscOp::Add, vec![x, a], vec3(), None);
        let late = dag.add_node(decl, RiscOp::Neg, vec![mix], vec3(), None);
        dag.add_root(late);
        (dag, x, mix, late)
    }

    #[test]
    fn without_an_agreement_the_last_operand_read_ends_the_lifetime() {
        let (dag, x, mix, late) = agreement_graph();
        let schedule = schedule_for(&dag, &[late]);
        assert_eq!(
            freed_at(&schedule, x),
            Some(mix.0),
            "x's last operand reader is the same-shape producer"
        );
    }

    #[test]
    fn a_same_shape_agreement_member_extends_the_lifetime_it_reads() {
        let (dag, x, mix, late) = agreement_graph();
        let agreement = crate::axis_sources::same_shape_result_agreement(&dag, mix)
            .expect("a same-shape producer")
            .expect("positive rank");
        assert_eq!(agreement.members(), &[x, NodeId(1)]);

        // The site is the FINAL node, which reads neither member through an
        // operand slot. Only the agreement keeps them alive that far.
        let schedule = schedule_with_guard(
            &dag,
            late,
            guard_claim(
                crate::axis_sources::CanonicalExtent::Resolved(3),
                crate::axis_sources::LocalGuardObservation::SameShapeAgreement(agreement),
                crate::axis_sources::GuardActivation::sizing(&dag, late, 0).unwrap(),
            ),
            &[late],
        );
        assert_eq!(
            freed_at(&schedule, x),
            Some(late.0),
            "`same_shape_agreement_extent` compares every member's realized shape"
        );
    }

    #[test]
    fn a_masked_off_consumer_does_not_pin_a_value() {
        // `dead` reads `x` but never executes under the root mask, so it must
        // not hold `x` past the live consumer that finishes with it.
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3(),
            None,
        );
        let live_use = dag.add_node(decl, RiscOp::Neg, vec![x], vec3(), None);
        let dead = dag.add_node(decl, RiscOp::Neg, vec![x], vec3(), None);
        dag.add_root(live_use);
        dag.add_root(dead);

        let mask = live_mask_for_roots(&dag, &[live_use]);
        let schedule = value_free_schedule(
            &dag,
            &node_order(&dag),
            Some(&mask),
            &UnordMap::new(),
            &[live_use],
        );
        assert_eq!(freed_at(&schedule, x), Some(1));
        assert_eq!(
            freed_at(&schedule, dead),
            None,
            "a node the mask skips produces nothing to free"
        );

        // And with both selected, the dead consumer runs and pins `x`.
        let both = live_mask_for_roots(&dag, &[live_use, dead]);
        let schedule = value_free_schedule(
            &dag,
            &node_order(&dag),
            Some(&both),
            &UnordMap::new(),
            &[live_use, dead],
        );
        assert_eq!(freed_at(&schedule, x), Some(2));
    }
}
