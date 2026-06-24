//! Tensor-aware evaluator for the Phase 0 RISC DAG.

use std::collections::{HashMap, HashSet};

use crate::dag::{
    Dag, DimInfo, FusedInput, FusedStepOp, NodeId, ReduceWindowKind, RiscOp, TensorType,
    bind_symbolic_dims, symbolic_bindings,
};
use chelis_types::types::Prim;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorValue {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl TensorValue {
    pub fn scalar(value: f64) -> Self {
        Self {
            data: vec![value],
            shape: vec![],
        }
    }

    pub fn from_vec(shape: Vec<usize>, data: Vec<f64>) -> Self {
        assert_eq!(numel(&shape), data.len());
        Self { data, shape }
    }
}

fn numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        // Scalar: no dimensions means a single element.
        1
    } else {
        // Non-scalar: honor every dimension, including zero. A tensor[0, f32]
        // legitimately holds zero elements; inflating to 1 drops data integrity
        // and panics the from_vec length assertion.
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

fn default_value(ty: &TensorType) -> TensorValue {
    let shape = concrete_shape(ty).unwrap_or_default();
    TensorValue {
        data: vec![0.0; numel(&shape)],
        shape,
    }
}

/// Element-wise `cast` conversion for the DAG evaluator (#380), returning the
/// converted value in this evaluator's `f64` storage. Mirrors
/// `chelis-compiler-api`'s `runtime::host_ops::convert_scalar_data` and the C
/// backend's element-wise C cast so all three lanes agree:
/// float->int truncates toward zero, float->f32 drops precision through f32,
/// int->float preserves the value, and bool encodes as 0.0 / 1.0 (decoded via
/// `!= 0.0`). `chelis-ir` sits below `chelis-compiler-api`, so the logic is
/// duplicated here rather than shared (no upward dependency).
fn convert_cast_data(x: f64, src: Prim, dst: Prim) -> f64 {
    if src == dst {
        return x;
    }
    // Project the source storage into a normalized representation: ints route
    // through i64, bools through 0/1, floats stay floats.
    let as_int: Option<i64> = match src {
        Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64 => Some(x as i64),
        Prim::Bool => Some(if x != 0.0 { 1 } else { 0 }),
        _ => None,
    };
    // Emit in the target precision's storage convention.
    match dst {
        Prim::F64 => match as_int {
            Some(i) => i as f64,
            None => x,
        },
        Prim::F32 => match as_int {
            Some(i) => (i as f32) as f64,
            None => (x as f32) as f64,
        },
        Prim::Int8 => match as_int {
            Some(i) => (i as i8) as f64,
            None => (x as i8) as f64,
        },
        Prim::Int16 => match as_int {
            Some(i) => (i as i16) as f64,
            None => (x as i16) as f64,
        },
        Prim::Int32 => match as_int {
            Some(i) => (i as i32) as f64,
            None => (x as i32) as f64,
        },
        Prim::Int64 => match as_int {
            Some(i) => i as f64,
            None => (x as i64) as f64,
        },
        Prim::Bool => {
            let nonzero = match as_int {
                Some(i) => i != 0,
                None => x != 0.0,
            };
            if nonzero { 1.0 } else { 0.0 }
        }
        // Reduced floats (bf16/f16) route through f32 storage in this f64
        // evaluator; string is not a tensor element type. Fall back to the
        // f32 narrowing for reduced floats and pass other targets through.
        Prim::Bf16 | Prim::F16 => match as_int {
            Some(i) => (i as f32) as f64,
            None => (x as f32) as f64,
        },
        _ => x,
    }
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

fn infer_symbolic_bindings_from_inputs(
    dag: &Dag,
    inputs: &HashMap<String, TensorValue>,
) -> Result<HashMap<String, usize>, String> {
    let mut bindings = HashMap::new();
    let mut load_types = HashMap::<String, TensorType>::new();

    for node in dag.nodes() {
        if let RiscOp::Load { name } = &node.op {
            load_types
                .entry(name.as_str().to_string())
                .or_insert_with(|| node.output_type.clone());
        }
    }

    for (name, ty) in &load_types {
        if let Some(value) = inputs.get(name) {
            validate_shape_against_type(name, value, ty)?;
        }
    }

    for binding in symbolic_bindings(dag) {
        let canonical_value = inputs.get(&binding.canonical.input_label).ok_or_else(|| {
            format!(
                "missing required input `{}` for symbolic dimension `{}`",
                binding.canonical.input_label, binding.name
            )
        })?;
        let value = *canonical_value
            .shape
            .get(binding.canonical.axis)
            .ok_or_else(|| {
                format!(
                    "input `{}` is missing axis {} for symbolic dimension `{}`",
                    binding.canonical.input_label, binding.canonical.axis, binding.name
                )
            })?;
        bindings.insert(binding.name.clone(), value);

        for occurrence in &binding.others {
            let other_value = inputs.get(&occurrence.input_label).ok_or_else(|| {
                format!(
                    "missing required input `{}` for symbolic dimension `{}`",
                    occurrence.input_label, binding.name
                )
            })?;
            let other = *other_value.shape.get(occurrence.axis).ok_or_else(|| {
                format!(
                    "input `{}` is missing axis {} for symbolic dimension `{}`",
                    occurrence.input_label, occurrence.axis, binding.name
                )
            })?;
            if other != value {
                return Err(format!(
                    "symbolic dimension `{}` mismatch: canonical {}[{}] = {}, but {}[{}] = {}",
                    binding.name,
                    binding.canonical.input_label,
                    binding.canonical.axis,
                    value,
                    occurrence.input_label,
                    occurrence.axis,
                    other
                ));
            }
        }
    }

    Ok(bindings)
}

fn resolve_load_inputs<F>(
    dag: &Dag,
    live: Option<&[bool]>,
    strict_loads: bool,
    symbolic_dim_load_inputs: &HashSet<&str>,
    mut load_input: F,
) -> Result<HashMap<String, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    let mut inputs = HashMap::new();
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
        match load_input(name.as_str()) {
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

fn unary_map(input: &TensorValue, f: impl Fn(f64) -> f64) -> TensorValue {
    TensorValue {
        data: input.data.iter().copied().map(f).collect(),
        shape: input.shape.clone(),
    }
}

fn dropout(input: &TensorValue, rate: f64, seed: u64) -> TensorValue {
    let keep_scale = if rate >= 1.0 {
        0.0
    } else {
        1.0 / (1.0 - rate.max(0.0))
    };
    let data = input
        .data
        .iter()
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
    TensorValue {
        data,
        shape: input.shape.clone(),
    }
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

fn uniform_like(shape: &[usize], low: f64, high: f64, seed: u64) -> TensorValue {
    let span = high - low;
    let data = (0..numel(shape))
        .map(|index| low + span * dropout_sample(seed, index as u64))
        .collect();
    TensorValue {
        data,
        shape: shape.to_vec(),
    }
}

fn binary_map(lhs: &TensorValue, rhs: &TensorValue, f: impl Fn(f64, f64) -> f64) -> TensorValue {
    assert_eq!(lhs.shape, rhs.shape);
    TensorValue {
        data: lhs
            .data
            .iter()
            .copied()
            .zip(rhs.data.iter().copied())
            .map(|(a, b)| f(a, b))
            .collect(),
        shape: lhs.shape.clone(),
    }
}

fn matmul(lhs: &TensorValue, rhs: &TensorValue) -> TensorValue {
    assert_eq!(lhs.shape.len(), 2);
    assert_eq!(rhs.shape.len(), 2);
    let m = lhs.shape[0];
    let k = lhs.shape[1];
    assert_eq!(rhs.shape[0], k);
    let n = rhs.shape[1];
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += lhs.data[i * k + kk] * rhs.data[kk * n + j];
            }
            data[i * n + j] = acc;
        }
    }
    TensorValue {
        data,
        shape: vec![m, n],
    }
}

fn batched_matmul(lhs: &TensorValue, rhs: &TensorValue) -> TensorValue {
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
    let mut data = vec![0.0; batch_count * m * n];
    for batch_idx in 0..batch_count {
        let lhs_base = batch_idx * m * k;
        let rhs_base = batch_idx * k * n;
        let out_base = batch_idx * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for kk in 0..k {
                    acc += lhs.data[lhs_base + i * k + kk] * rhs.data[rhs_base + kk * n + j];
                }
                data[out_base + i * n + j] = acc;
            }
        }
    }
    TensorValue {
        data,
        shape: out_shape,
    }
}

fn gather(values: &TensorValue, indices: &TensorValue, axis: usize) -> TensorValue {
    assert!(axis < values.shape.len());
    let index_rank = indices.shape.len();
    let mut out_shape = Vec::with_capacity(values.shape.len() - 1 + index_rank);
    out_shape.extend_from_slice(&values.shape[..axis]);
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&values.shape[axis + 1..]);
    let mut data = vec![0.0; numel(&out_shape)];
    for (out_linear, out_slot) in data.iter_mut().enumerate() {
        let out_index = linear_to_index(out_linear, &out_shape);
        let mut idx_index = Vec::with_capacity(index_rank);
        for pos in 0..index_rank {
            idx_index.push(out_index[axis + pos]);
        }
        let gathered = indices.data[index_to_linear(&idx_index, &indices.shape)] as isize;
        assert!(
            gathered >= 0 && (gathered as usize) < values.shape[axis],
            "gather index {gathered} out of bounds for axis {axis}"
        );
        let mut value_index = Vec::with_capacity(values.shape.len());
        value_index.extend_from_slice(&out_index[..axis]);
        value_index.push(gathered as usize);
        value_index.extend_from_slice(&out_index[axis + index_rank..]);
        *out_slot = values.data[index_to_linear(&value_index, &values.shape)];
    }
    TensorValue {
        data,
        shape: out_shape,
    }
}

fn scatter_add(
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

    let mut out = target.clone();
    for update_linear in 0..updates.data.len() {
        let update_index = linear_to_index(update_linear, &updates.shape);
        let mut idx_index = Vec::with_capacity(index_rank);
        for pos in 0..index_rank {
            idx_index.push(update_index[axis + pos]);
        }
        let gathered = indices.data[index_to_linear(&idx_index, &indices.shape)] as isize;
        assert!(
            gathered >= 0 && (gathered as usize) < target.shape[axis],
            "scatter_add index {gathered} out of bounds for axis {axis}"
        );
        let mut target_index = Vec::with_capacity(target.shape.len());
        target_index.extend_from_slice(&update_index[..axis]);
        target_index.push(gathered as usize);
        target_index.extend_from_slice(&update_index[axis + index_rank..]);
        let target_linear = index_to_linear(&target_index, &target.shape);
        out.data[target_linear] += updates.data[update_linear];
    }
    out
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

    let mut out = target.clone();
    for update_linear in 0..updates.data.len() {
        let update_index = linear_to_index(update_linear, &updates.shape);
        let mut idx_index = Vec::with_capacity(index_rank);
        for pos in 0..index_rank {
            idx_index.push(update_index[axis + pos]);
        }
        let gathered = indices.data[index_to_linear(&idx_index, &indices.shape)] as isize;
        assert!(
            gathered >= 0 && (gathered as usize) < target.shape[axis],
            "scatter_replace index {gathered} out of bounds for axis {axis}"
        );
        let mut target_index = Vec::with_capacity(target.shape.len());
        target_index.extend_from_slice(&update_index[..axis]);
        target_index.push(gathered as usize);
        target_index.extend_from_slice(&update_index[axis + index_rank..]);
        let target_linear = index_to_linear(&target_index, &target.shape);
        // Last-write-wins: deterministic-order overwrite.
        out.data[target_linear] = updates.data[update_linear];
    }
    out
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

    let mut out = data.clone();
    for update_linear in 0..updates.data.len() {
        let coord = linear_to_index(update_linear, &updates.shape);
        let gathered = indices.data[update_linear] as isize;
        assert!(
            gathered >= 0 && (gathered as usize) < data.shape[axis],
            "scatter_elements index {gathered} out of bounds for axis {axis}"
        );
        let mut target_index = coord.clone();
        target_index[axis] = gathered as usize;
        let target_linear = index_to_linear(&target_index, &data.shape);
        // Last-write-wins: deterministic-order overwrite.
        out.data[target_linear] = updates.data[update_linear];
    }
    out
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
) -> TensorValue {
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

    let window_volume: usize = window_shape.iter().product();
    let out_len = numel(&out_shape);
    let mut out = vec![0.0_f64; out_len];

    let init_acc = |r: ReduceWindowKind| -> f64 {
        match r {
            ReduceWindowKind::Max => f64::NEG_INFINITY,
            ReduceWindowKind::Min => f64::INFINITY,
            ReduceWindowKind::Sum | ReduceWindowKind::Mean => 0.0,
        }
    };
    let combine = |r: ReduceWindowKind, acc: f64, x: f64| -> f64 {
        match r {
            ReduceWindowKind::Max => acc.max(x),
            ReduceWindowKind::Min => acc.min(x),
            ReduceWindowKind::Sum | ReduceWindowKind::Mean => acc + x,
        }
    };

    for (out_flat, slot) in out.iter_mut().enumerate() {
        let out_idx = linear_to_index(out_flat, &out_shape);

        // Walk the window: iterate over all positions inside the
        // window_shape multi-index. The source index per dimension is
        // `out_idx[axis] * stride + window_pos` for windowed axes,
        // matching the leading-axis passthrough rule above.
        let mut acc = init_acc(reducer);
        let mut window_pos = vec![0usize; n];
        loop {
            let mut src_idx = vec![0usize; rank];
            src_idx[..leading].copy_from_slice(&out_idx[..leading]);
            for i in 0..n {
                src_idx[leading + i] = out_idx[leading + i] * strides[i] + window_pos[i];
            }
            let src_flat = index_to_linear(&src_idx, &input.shape);
            acc = combine(reducer, acc, input.data[src_flat]);

            // Increment window_pos (mixed-radix carry).
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
        if matches!(reducer, ReduceWindowKind::Mean) {
            acc /= window_volume as f64;
        }
        *slot = acc;
    }

    TensorValue {
        data: out,
        shape: out_shape,
    }
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
) -> TensorValue {
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
    let window_volume: f64 = window_shape.iter().product::<usize>() as f64;

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

    let mut din = vec![0.0_f64; x.data.len()];

    // Walk each output position `o` (one upstream gradient value `g[o]`),
    // then walk that window's source positions. `Max`/`Min` need the
    // window extreme first; `Sum`/`Mean` scatter unconditionally.
    for (out_flat, &g_val) in g.data.iter().enumerate() {
        let out_idx = linear_to_index(out_flat, &out_shape);

        let src_flat_at = |window_pos: &[usize]| -> usize {
            let mut src_idx = vec![0usize; rank];
            src_idx[..leading].copy_from_slice(&out_idx[..leading]);
            for i in 0..n {
                src_idx[leading + i] = out_idx[leading + i] * strides[i] + window_pos[i];
            }
            index_to_linear(&src_idx, &x.shape)
        };

        // First pass (Max/Min only): find the window extreme.
        let extreme = match reducer {
            ReduceWindowKind::Max | ReduceWindowKind::Min => {
                let mut acc = match reducer {
                    ReduceWindowKind::Max => f64::NEG_INFINITY,
                    _ => f64::INFINITY,
                };
                for_each_window_pos(window_shape, n, |window_pos| {
                    let v = x.data[src_flat_at(window_pos)];
                    acc = match reducer {
                        ReduceWindowKind::Max => acc.max(v),
                        _ => acc.min(v),
                    };
                });
                Some(acc)
            }
            ReduceWindowKind::Sum | ReduceWindowKind::Mean => None,
        };

        // Second pass: scatter the contribution into `din`.
        for_each_window_pos(window_shape, n, |window_pos| {
            let src = src_flat_at(window_pos);
            match reducer {
                ReduceWindowKind::Sum => din[src] += g_val,
                ReduceWindowKind::Mean => din[src] += g_val / window_volume,
                ReduceWindowKind::Max | ReduceWindowKind::Min => {
                    // Distribute to every position equal to the window
                    // extreme (ties get the full gradient, mirroring the
                    // `max_reduce` `eq`-mask adjoint).
                    if x.data[src] == extreme.expect("extreme computed for Max/Min") {
                        din[src] += g_val;
                    }
                }
            }
        });
    }

    TensorValue {
        data: din,
        shape: x.shape.clone(),
    }
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

fn reduce(input: &TensorValue, axis: usize, init: f64, f: impl Fn(f64, f64) -> f64) -> TensorValue {
    assert!(axis < input.shape.len());
    let mut out_shape = input.shape.clone();
    out_shape.remove(axis);
    let out_len = numel(&out_shape);
    let mut out = vec![init; out_len];
    for (flat_idx, &value) in input.data.iter().enumerate() {
        let mut idx = linear_to_index(flat_idx, &input.shape);
        idx.remove(axis);
        let out_idx = index_to_linear(&idx, &out_shape);
        out[out_idx] = f(out[out_idx], value);
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

/// Reduce along `axis`, tracking the index of the element that wins under
/// `better(current_best, candidate)`. Used for Argmax / Argmin.
///
/// Ties are broken by the smallest index (first-seen wins), matching numpy's
/// default argmax/argmin semantics. The output stores integer indices as f64
/// in the same precision-erased TensorValue layout other reductions use; the
/// host-runtime adapter (`tensor_reduce_host`) is responsible for attaching
/// the `Prim::Int64` precision tag on the produced `RuntimeTensorValue` per
/// chelis#233. See `RiscOp::Argmax` for the spec-level invariants.
fn reduce_argcmp(
    input: &TensorValue,
    axis: usize,
    init: f64,
    better: impl Fn(f64, f64) -> bool,
) -> TensorValue {
    assert!(axis < input.shape.len());
    let mut out_shape = input.shape.clone();
    out_shape.remove(axis);
    let out_len = numel(&out_shape);
    let mut best_val = vec![init; out_len];
    let mut best_idx = vec![-1i64; out_len];
    for (flat_idx, &value) in input.data.iter().enumerate() {
        let full = linear_to_index(flat_idx, &input.shape);
        let axis_pos = full[axis] as i64;
        let mut reduced = full.clone();
        reduced.remove(axis);
        let out_idx = index_to_linear(&reduced, &out_shape);
        if best_idx[out_idx] < 0 || better(best_val[out_idx], value) {
            best_val[out_idx] = value;
            best_idx[out_idx] = axis_pos;
        }
    }
    TensorValue {
        data: best_idx.into_iter().map(|i| i as f64).collect(),
        shape: out_shape,
    }
}

fn reshape(input: &TensorValue, shape: Vec<usize>) -> TensorValue {
    assert_eq!(input.data.len(), numel(&shape));
    TensorValue {
        data: input.data.clone(),
        shape,
    }
}

fn permute(input: &TensorValue, axes: &[usize]) -> TensorValue {
    assert_eq!(axes.len(), input.shape.len());
    let out_shape: Vec<usize> = axes.iter().map(|&axis| input.shape[axis]).collect();
    let out_len = numel(&out_shape);
    let mut out = vec![0.0; out_len];
    for (flat_idx, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_index(flat_idx, &out_shape);
        let mut in_index = vec![0; input.shape.len()];
        for (out_axis, &in_axis) in axes.iter().enumerate() {
            in_index[in_axis] = out_index[out_axis];
        }
        *slot = input.data[index_to_linear(&in_index, &input.shape)];
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

fn expand(input: &TensorValue, axis: usize, _size: usize, out_shape: Vec<usize>) -> TensorValue {
    assert!(axis <= input.shape.len());
    let out_len = numel(&out_shape);
    let mut out = vec![0.0; out_len];
    for (flat_idx, slot) in out.iter_mut().enumerate() {
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
        *slot = input.data[index_to_linear(&in_index, &input.shape)];
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

fn one_hot(indices: &TensorValue, vocab: usize) -> TensorValue {
    let mut out_shape = indices.shape.clone();
    out_shape.push(vocab);
    let mut out = vec![0.0; numel(&out_shape)];
    for (index_linear, &raw_index) in indices.data.iter().enumerate() {
        let class = raw_index as isize;
        assert!(
            class >= 0 && (class as usize) < vocab,
            "one_hot index {class} out of bounds for vocab {vocab}"
        );
        let mut out_index = linear_to_index(index_linear, &indices.shape);
        out_index.push(class as usize);
        out[index_to_linear(&out_index, &out_shape)] = 1.0;
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

fn pad(input: &TensorValue, padding: &[(usize, usize)], fill: f64) -> TensorValue {
    assert_eq!(padding.len(), input.shape.len());
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(padding.iter())
        .map(|(dim, (before, after))| dim + before + after)
        .collect();
    let mut out = vec![fill; numel(&out_shape)];
    for (flat_idx, &value) in input.data.iter().enumerate() {
        let in_index = linear_to_index(flat_idx, &input.shape);
        let out_index: Vec<usize> = in_index
            .iter()
            .zip(padding.iter())
            .map(|(idx, (before, _))| idx + before)
            .collect();
        let out_flat = index_to_linear(&out_index, &out_shape);
        out[out_flat] = value;
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

fn shrink(input: &TensorValue, bounds: &[(usize, usize)]) -> TensorValue {
    assert_eq!(bounds.len(), input.shape.len());
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(bounds.iter())
        .map(|(_, (start, end))| end - start)
        .collect();
    let out_len = numel(&out_shape);
    let mut out = vec![0.0; out_len];
    for (flat_idx, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_index(flat_idx, &out_shape);
        let in_index: Vec<usize> = out_index
            .iter()
            .zip(bounds.iter())
            .map(|(idx, (start, _))| idx + start)
            .collect();
        *slot = input.data[index_to_linear(&in_index, &input.shape)];
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

fn stride(input: &TensorValue, strides: &[usize]) -> TensorValue {
    assert_eq!(strides.len(), input.shape.len());
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(strides.iter())
        .map(|(dim, step)| {
            if *step == 0 {
                *dim
            } else {
                (*dim).div_ceil(*step)
            }
        })
        .collect();
    let out_len = numel(&out_shape);
    let mut out = vec![0.0; out_len];
    for (flat_idx, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_index(flat_idx, &out_shape);
        let in_index: Vec<usize> = out_index
            .iter()
            .zip(strides.iter())
            .map(|(idx, step)| idx * step.max(&1))
            .collect();
        *slot = input.data[index_to_linear(&in_index, &input.shape)];
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
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
        }
    }
    live
}

fn eval_tensor_internal<F>(
    dag: &Dag,
    live: Option<&[bool]>,
    strict_loads: bool,
    mut load_input: F,
) -> Result<HashMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    let needs_symbolic_binding = dag
        .nodes()
        .iter()
        .any(|node| matches!(&node.op, RiscOp::Expand { size, .. } if !size.is_concrete()))
        || dag.nodes().iter().any(|node| {
            node.output_type
                .dims
                .iter()
                .any(|dim| matches!(dim, DimInfo::Named(_, None)))
        })
        || dag.nodes().iter().any(|node| match &node.op {
            RiscOp::Reshape { new_shape } => new_shape
                .iter()
                .any(|dim| matches!(dim, DimInfo::Named(_, None))),
            _ => false,
        });
    // chelis#351: symbolic-dim inference reads shapes from the Loads
    // that `symbolic_occurrences` nominates as each dim's declaring
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
    let symbolic_dim_load_inputs: HashSet<&str> = if needs_symbolic_binding && live.is_some() {
        dag.nodes()
            .iter()
            .filter_map(|node| match &node.op {
                RiscOp::Load { name }
                    if node
                        .output_type
                        .dims
                        .iter()
                        .any(|dim| matches!(dim, DimInfo::Named(_, _))) =>
                {
                    Some(name.as_str())
                }
                _ => None,
            })
            .collect()
    } else {
        HashSet::new()
    };
    let resolved_inputs = resolve_load_inputs(
        dag,
        live,
        strict_loads,
        &symbolic_dim_load_inputs,
        &mut load_input,
    )?;
    let bound_dag = if needs_symbolic_binding {
        let bindings = infer_symbolic_bindings_from_inputs(dag, &resolved_inputs)?;
        bind_symbolic_dims(dag, &bindings)?
    } else {
        dag.clone()
    };

    let mut values: HashMap<NodeId, TensorValue> = HashMap::new();

    for node in bound_dag.nodes() {
        if let Some(mask) = live
            && !mask[node.id.0]
        {
            continue;
        }

        let value = match &node.op {
            RiscOp::Const { value } => {
                let shape = concrete_shape(&node.output_type).unwrap_or_default();
                TensorValue {
                    data: vec![*value; numel(&shape)],
                    shape,
                }
            }
            RiscOp::Load { name } => match resolved_inputs.get(name.as_str()) {
                Some(value) => value.clone(),
                None if strict_loads => return Err(format!("missing required input `{name}`")),
                None => default_value(&node.output_type),
            },
            RiscOp::Store { .. } | RiscOp::Copy | RiscOp::Drop | RiscOp::Realize => {
                values[&node.inputs[0]].clone()
            }
            RiscOp::Add => binary_map(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                |a, b| a + b,
            ),
            RiscOp::Mul => binary_map(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                |a, b| a * b,
            ),
            RiscOp::Div => binary_map(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                |a, b| a / b,
            ),
            RiscOp::Neg => unary_map(&values[&node.inputs[0]], |x| -x),
            RiscOp::Recip => unary_map(&values[&node.inputs[0]], |x| 1.0 / x),
            RiscOp::Exp => unary_map(&values[&node.inputs[0]], f64::exp),
            RiscOp::Log => unary_map(&values[&node.inputs[0]], f64::ln),
            RiscOp::Sin => unary_map(&values[&node.inputs[0]], f64::sin),
            RiscOp::Sqrt => unary_map(&values[&node.inputs[0]], f64::sqrt),
            RiscOp::Cos => unary_map(&values[&node.inputs[0]], f64::cos),
            RiscOp::Tan => unary_map(&values[&node.inputs[0]], f64::tan),
            RiscOp::Atan => unary_map(&values[&node.inputs[0]], f64::atan),
            RiscOp::Abs => unary_map(&values[&node.inputs[0]], f64::abs),
            RiscOp::Floor => unary_map(&values[&node.inputs[0]], f64::floor),
            RiscOp::Ceil => unary_map(&values[&node.inputs[0]], f64::ceil),
            // Round-half-to-even (banker's rounding), matching the C
            // backend's `rintf` under the default rounding mode. NOT
            // `f64::round`, which rounds half away from zero.
            RiscOp::Round => unary_map(&values[&node.inputs[0]], f64::round_ties_even),
            RiscOp::UniformLike { low, high, seed } => {
                uniform_like(&values[&node.inputs[0]].shape, *low, *high, *seed)
            }
            RiscOp::Dropout { rate, seed } => dropout(&values[&node.inputs[0]], *rate, *seed),
            RiscOp::MaxElem => {
                binary_map(&values[&node.inputs[0]], &values[&node.inputs[1]], f64::max)
            }
            RiscOp::CmpLt => binary_map(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                |a, b| {
                    if a < b { 1.0 } else { 0.0 }
                },
            ),
            RiscOp::Sum { axis, .. } => {
                reduce(&values[&node.inputs[0]], *axis, 0.0, |acc, x| acc + x)
            }
            RiscOp::MaxReduce { axis } => {
                reduce(&values[&node.inputs[0]], *axis, f64::NEG_INFINITY, f64::max)
            }
            RiscOp::MinReduce { axis } => {
                reduce(&values[&node.inputs[0]], *axis, f64::INFINITY, f64::min)
            }
            RiscOp::ProdReduce { axis } => {
                reduce(&values[&node.inputs[0]], *axis, 1.0, |acc, x| acc * x)
            }
            RiscOp::ReduceWindow {
                reducer,
                window_shape,
                strides,
            } => reduce_window(&values[&node.inputs[0]], *reducer, window_shape, strides),
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
            ),
            RiscOp::Argmax { axis } => reduce_argcmp(
                &values[&node.inputs[0]],
                *axis,
                f64::NEG_INFINITY,
                |a, b| b > a,
            ),
            RiscOp::Argmin { axis } => {
                reduce_argcmp(&values[&node.inputs[0]], *axis, f64::INFINITY, |a, b| b < a)
            }
            RiscOp::Reshape { new_shape } => {
                let shape: Vec<usize> = new_shape
                    .iter()
                    .map(|dim| match dim {
                        DimInfo::Lit(n) => Ok(*n),
                        DimInfo::Named(_, Some(n)) => Ok(*n),
                        DimInfo::Named(name, None) => {
                            Err(format!("cannot reshape to symbolic dimension `{name}`"))
                        }
                    })
                    .collect::<Result<_, _>>()?;
                reshape(&values[&node.inputs[0]], shape)
            }
            RiscOp::Permute { axes } => permute(&values[&node.inputs[0]], axes),
            RiscOp::Expand { axis, size } => expand(
                &values[&node.inputs[0]],
                *axis,
                size.as_concrete()
                    .expect("symbolic expands must be rebound before evaluation"),
                concrete_shape(&node.output_type)?,
            ),
            RiscOp::OneHot { vocab } => one_hot(&values[&node.inputs[0]], *vocab),
            RiscOp::Pad { padding, fill } => pad(&values[&node.inputs[0]], padding, *fill),
            RiscOp::Shrink { bounds } => shrink(&values[&node.inputs[0]], bounds),
            RiscOp::Stride { strides } => stride(&values[&node.inputs[0]], strides),
            RiscOp::FusedElem { ops } => {
                // Collect external input TensorValues from the node's DAG inputs.
                let externals: Vec<&TensorValue> =
                    node.inputs.iter().map(|id| &values[id]).collect();

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
                        FusedStepOp::Add => binary_map(
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                            |a, b| a + b,
                        ),
                        FusedStepOp::Mul => binary_map(
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                            |a, b| a * b,
                        ),
                        FusedStepOp::Div => binary_map(
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                            |a, b| a / b,
                        ),
                        FusedStepOp::MaxElem => binary_map(
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                            f64::max,
                        ),
                        FusedStepOp::CmpLt => binary_map(
                            resolve(&step.input_indices[0]),
                            resolve(&step.input_indices[1]),
                            |a, b| if a < b { 1.0 } else { 0.0 },
                        ),
                        // Unary ops
                        FusedStepOp::Neg => unary_map(resolve(&step.input_indices[0]), |x| -x),
                        FusedStepOp::Recip => {
                            unary_map(resolve(&step.input_indices[0]), |x| 1.0 / x)
                        }
                        FusedStepOp::Exp => unary_map(resolve(&step.input_indices[0]), f64::exp),
                        FusedStepOp::Log => unary_map(resolve(&step.input_indices[0]), f64::ln),
                        FusedStepOp::Sin => unary_map(resolve(&step.input_indices[0]), f64::sin),
                        FusedStepOp::Sqrt => unary_map(resolve(&step.input_indices[0]), f64::sqrt),
                        FusedStepOp::Cos => unary_map(resolve(&step.input_indices[0]), f64::cos),
                        FusedStepOp::Tan => unary_map(resolve(&step.input_indices[0]), f64::tan),
                        FusedStepOp::Atan => unary_map(resolve(&step.input_indices[0]), f64::atan),
                        FusedStepOp::Abs => unary_map(resolve(&step.input_indices[0]), f64::abs),
                        FusedStepOp::Floor => {
                            unary_map(resolve(&step.input_indices[0]), f64::floor)
                        }
                        FusedStepOp::Ceil => unary_map(resolve(&step.input_indices[0]), f64::ceil),
                        FusedStepOp::Round => {
                            unary_map(resolve(&step.input_indices[0]), f64::round_ties_even)
                        }
                    };
                    intermediates.push(result);
                }

                // The last step's output is the node's result.
                intermediates
                    .pop()
                    .expect("FusedElem must have at least one step")
            }
            RiscOp::Cast { new_precision } => {
                // #380: a `cast` must apply the dtype conversion, not pass the
                // f64-backed value through unchanged. Float->int truncates
                // toward zero, float->f32 drops precision through f32, int->
                // float preserves the value, and bool encodes as 0.0/1.0 — the
                // same conversion `chelis-compiler-api`'s `convert_scalar_data`
                // (and the C backend's element-wise C cast) apply. The source
                // precision comes from the input node's output type; the target
                // is the cast's `new_precision`. Pre-fix the DAG-evaluator lane
                // (grad / e2e-train) read `cast(3.7, int32)` as `3.7`.
                let input = &values[&node.inputs[0]];
                let src_prec = bound_dag
                    .get(node.inputs[0])
                    .map(|n| n.output_type.precision)
                    .unwrap_or(*new_precision);
                let converted = input
                    .data
                    .iter()
                    .map(|&x| convert_cast_data(x, src_prec, *new_precision))
                    .collect();
                TensorValue {
                    data: converted,
                    shape: input.shape.clone(),
                }
            }
            RiscOp::BlasMatmul { .. } => {
                let lhs = &values[&node.inputs[0]];
                let rhs = &values[&node.inputs[1]];
                if lhs.shape.len() == 2 && rhs.shape.len() == 2 {
                    matmul(lhs, rhs)
                } else {
                    batched_matmul(lhs, rhs)
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
            ),
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
        values.insert(node.id, value);
    }

    Ok(values)
}

pub fn eval_tensor_with<F>(dag: &Dag, load_input: F) -> Result<HashMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    eval_tensor_internal(dag, None, false, load_input)
}

pub fn eval_tensor_with_strict<F>(
    dag: &Dag,
    load_input: F,
) -> Result<HashMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    eval_tensor_internal(dag, None, true, load_input)
}

pub fn eval_tensor_roots_with<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<HashMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    if roots.is_empty() {
        return eval_tensor_internal(dag, None, false, load_input);
    }
    reject_drop_roots(dag, roots)?;
    let live = live_mask_for_roots(dag, roots);
    eval_tensor_internal(dag, Some(&live), false, load_input)
}

pub fn eval_tensor_roots_with_strict<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<HashMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    if roots.is_empty() {
        return eval_tensor_internal(dag, None, true, load_input);
    }
    reject_drop_roots(dag, roots)?;
    let live = live_mask_for_roots(dag, roots);
    eval_tensor_internal(dag, Some(&live), true, load_input)
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
    inputs: &HashMap<String, TensorValue>,
) -> Result<HashMap<NodeId, TensorValue>, String> {
    eval_tensor_with(dag, |name| inputs.get(name).cloned())
}

/// Evaluate a DAG on scalar inputs. Each node produces a single f64.
pub fn eval_scalar(dag: &Dag, inputs: &HashMap<String, f64>) -> HashMap<NodeId, f64> {
    let tensor_inputs: HashMap<String, TensorValue> = inputs
        .iter()
        .map(|(name, value)| (name.clone(), TensorValue::scalar(*value)))
        .collect();
    eval_tensor(dag, &tensor_inputs)
        .expect("scalar evaluation should not fail")
        .into_iter()
        .map(|(id, value)| (id, *value.data.first().unwrap_or(&0.0)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::RiscOp;
    use crate::lower::lower_program;
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

    // ---- reduce_window reverse-mode adjoint (RiscOp::ReduceWindowGrad) ----

    /// Scalar loss `sum(reduce_window(x))` used as the finite-difference
    /// oracle: its cotangent w.r.t. every forward output is exactly 1, so
    /// the analytic input gradient is `reduce_window_grad(x, ones)`.
    fn rw_loss(x: &TensorValue, r: ReduceWindowKind, w: &[usize], s: &[usize]) -> f64 {
        reduce_window(x, r, w, s).data.iter().sum()
    }

    /// Central finite-difference gradient of `rw_loss` w.r.t. each element.
    fn rw_fd_grad(x: &TensorValue, r: ReduceWindowKind, w: &[usize], s: &[usize]) -> Vec<f64> {
        let eps = 1e-4;
        (0..x.data.len())
            .map(|i| {
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp.data[i] += eps;
                xm.data[i] -= eps;
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
        let x = TensorValue {
            data: vec![
                3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.0, 3.5, 8.0, 9.5, 7.0, 0.5, 2.5, 6.5,
            ],
            shape: vec![4, 4],
        };
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
                let out = reduce_window(&x, r, &w, &s);
                let ones = TensorValue {
                    data: vec![1.0; out.data.len()],
                    shape: out.shape.clone(),
                };
                let analytic = reduce_window_grad(&x, &ones, r, &w, &s);
                assert_eq!(analytic.shape, x.shape);
                let fd = rw_fd_grad(&x, r, &w, &s);
                assert_close(
                    &analytic.data,
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
        let x = TensorValue {
            data: vec![10.0, 20.0, 30.0, 40.0],
            shape: vec![4],
        };
        let out = reduce_window(&x, ReduceWindowKind::Sum, &[2], &[1]);
        let ones = TensorValue {
            data: vec![1.0; out.data.len()],
            shape: out.shape.clone(),
        };
        let din = reduce_window_grad(&x, &ones, ReduceWindowKind::Sum, &[2], &[1]);
        assert_eq!(din.data, vec![1.0, 2.0, 2.0, 1.0]);

        // Mean is Sum scaled by 1 / window_volume (= 2 here).
        let din_mean = reduce_window_grad(&x, &ones, ReduceWindowKind::Mean, &[2], &[1]);
        assert_eq!(din_mean.data, vec![0.5, 1.0, 1.0, 0.5]);
    }

    #[test]
    fn reduce_window_grad_max_routes_to_argmax_and_accumulates_overlap() {
        // Strictly increasing input over a [4] window=2 stride=1: windows
        // [0,1]->max@1, [1,2]->max@2, [2,3]->max@3. With distinct upstream
        // gradients, position 0 gets nothing, 1 gets g[0], 2 gets g[1],
        // 3 gets g[2].
        let x = TensorValue {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![4],
        };
        let g = TensorValue {
            data: vec![5.0, 7.0, 11.0],
            shape: vec![3],
        };
        let din = reduce_window_grad(&x, &g, ReduceWindowKind::Max, &[2], &[1]);
        assert_eq!(din.data, vec![0.0, 5.0, 7.0, 11.0]);

        // Min over the same increasing input routes to the window minimum:
        // windows pick positions 0,1,2; position 3 gets nothing.
        let din_min = reduce_window_grad(&x, &g, ReduceWindowKind::Min, &[2], &[1]);
        assert_eq!(din_min.data, vec![5.0, 7.0, 11.0, 0.0]);
    }

    #[test]
    fn reduce_window_grad_max_distributes_to_ties() {
        // A flat window: every position equals the max, so each tied
        // position receives the full upstream gradient (the max_reduce
        // mask convention), not a 1/k share.
        let x = TensorValue {
            data: vec![2.0, 2.0, 2.0],
            shape: vec![3],
        };
        let g = TensorValue {
            data: vec![4.0, 4.0],
            shape: vec![2],
        };
        let din = reduce_window_grad(&x, &g, ReduceWindowKind::Max, &[2], &[1]);
        // windows [0,1] and [1,2]: pos0 += 4 (win0), pos1 += 4+4, pos2 += 4.
        assert_eq!(din.data, vec![4.0, 8.0, 4.0]);
    }

    #[test]
    fn eval_add() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let vals = eval_scalar(&dag, &HashMap::new());
        assert!((vals[&c] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eval_vector_add() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec3_f32(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec3_f32(), None);
        let c = dag.add_node(RiscOp::Add, vec![a, b], vec3_f32(), None);
        let mut inputs = HashMap::new();
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

        let inputs = HashMap::from([
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

        let inputs = HashMap::from([
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
                size: crate::dag::DimExpr::Concrete(4),
            },
            vec![x],
            out_ty,
            None,
        );
        let mut inputs = HashMap::new();
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

    /// chelis#351: a Load can be DEAD under the roots' live mask while a
    /// live node still needs a symbolic dim that only that Load
    /// declares (`vmap(grad(f))` where the gradient is constant in the
    /// input: the backward DAG never consumes `x`, but its
    /// `Expand { size: Sym(n) }` must bind `n` from `x`'s shape).
    /// The occurrence input is resolved despite the mask, and the dim
    /// binds from its shape.
    #[test]
    fn eval_root_scoped_strict_resolves_dead_load_for_symbolic_dim() {
        let mut dag = Dag::new();
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let _x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_ty.clone(),
            None,
        );
        let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let ones = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::DimExpr::Sym("n".to_string()),
            },
            vec![one],
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
            "`n` must bind to 3 from the dead `x` Load's runtime shape"
        );
    }

    /// Negative parity for the dead-load resolution above: when the
    /// declaring occurrence input is genuinely unavailable, the failure
    /// is the dim-targeted inference error — never a silent default
    /// shape and never the bare strict-load error (the Load is dead, so
    /// strict mode has no claim on it).
    #[test]
    fn eval_root_scoped_strict_missing_dead_symbolic_load_is_dim_error() {
        let mut dag = Dag::new();
        let sym_ty = TensorType {
            dims: vec![DimInfo::Named("n".to_string(), None)],
            precision: Prim::F32,
        };
        let _x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_ty.clone(),
            None,
        );
        let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let ones = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::DimExpr::Sym("n".to_string()),
            },
            vec![one],
            sym_ty,
            None,
        );

        let err = eval_tensor_roots_with_strict(&dag, &[ones], |_| None).unwrap_err();
        assert!(
            err.contains("missing required input `x` for symbolic dimension `n`"),
            "expected the dim-targeted inference error, got: {err}"
        );
    }

    fn lower(src: &str) -> Dag {
        let exprs = parse_str(src).expect("parse failed");
        let checked = chelis_types::check_ir_program(&exprs)
            .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        let checked = chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"));
        lower_program(&checked)
    }

    #[test]
    fn lowered_relu_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
            (def {} y (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} (var {} relu) (var {} x)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![3], vec![-2.0, 0.5, 4.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(dag.roots().last().expect("DAG root")).unwrap();
        assert_eq!(*last, TensorValue::from_vec(vec![3], vec![0.0, 0.5, 4.0]));
    }

    /// #380: the DAG evaluator must apply the dtype conversion for `cast`,
    /// not pass the f64-backed value through unchanged. Float->int truncates
    /// toward zero. Pre-fix `cast([2.7, -2.7, 3.0], int32)` evaluated to
    /// `[2.7, -2.7, 3.0]`; it must be `[2.0, -2.0, 3.0]`.
    #[test]
    fn lowered_cast_float_to_int_truncates_toward_zero() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
            (def {} y
              (cast {type: (t-tensor {} (d-lit {} 3) (t-prim {} int32))}
                    (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)
                    (t-prim {} int32)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![3], vec![2.7, -2.7, 3.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(dag.roots().last().expect("DAG root")).unwrap();
        assert_eq!(
            *last,
            TensorValue::from_vec(vec![3], vec![2.0, -2.0, 3.0]),
            "cast to int32 must truncate toward zero in the DAG evaluator (#380)",
        );
    }

    /// #380 negative / direction guard: int->float preserves the value
    /// exactly (no spurious truncation on the int->float direction), and
    /// float->f32 narrows. Pins that `convert_cast_data` only truncates on
    /// the float->int direction, not blindly.
    #[test]
    fn lowered_cast_int_to_float_preserves_value() {
        // int32 input carries integral f64 storage; cast to f64 must keep it.
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} int32))} x))
            (def {} y
              (cast {type: (t-tensor {} (d-lit {} 2) (t-prim {} f64))}
                    (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} int32))} x)
                    (t-prim {} f64)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
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
        let mut inputs = HashMap::new();
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
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(dag.roots().last().expect("DAG root")).unwrap();
        let expected = [0.09003057, 0.24472847, 0.66524096];
        for (actual, target) in last.data.iter().zip(expected.iter()) {
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
                   (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
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
        for (actual, target) in last.data.iter().zip(expected.iter()) {
            assert!((actual - target).abs() < 2e-4, "{actual} vs {target}");
        }
    }

    #[test]
    fn lowered_conv2d_1x1_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))} x))
            (def {} k (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 1) (d-lit {} 1) (t-prim {} f32))} k))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
                   (var {} conv2d) (var {} x) (var {} k) (lit {} 1) (lit {} 0)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        );
        inputs.insert(
            "k".into(),
            TensorValue::from_vec(vec![1, 1, 1, 1], vec![2.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(dag.roots().last().expect("DAG root")).unwrap();
        assert_eq!(
            *last,
            TensorValue::from_vec(vec![1, 1, 2, 2], vec![2.0, 4.0, 6.0, 8.0])
        );
    }

    #[test]
    fn lowered_conv2d_2x2_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} x))
            (def {} k (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))} k))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
                   (var {} conv2d) (var {} x) (var {} k) (lit {} 1) (lit {} 0)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
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
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(dag.roots().last().expect("DAG root")).unwrap();
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
                (lit {type: (t-prim {} int32)} 42)
                (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))))
        "#;
        let dag = lower(src);
        let vals_a = eval_tensor(&dag, &HashMap::new()).unwrap();
        let vals_b = eval_tensor(&dag, &HashMap::new()).unwrap();
        let out_a = vals_a.get(dag.roots().last().expect("DAG root")).unwrap();
        let out_b = vals_b.get(dag.roots().last().expect("DAG root")).unwrap();
        assert_eq!(out_a, out_b);
        assert!(out_a.data.contains(&0.0));
        assert!(out_a.data.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn lowered_dropout_changes_with_different_seed() {
        let src_a = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 32) (t-prim {} f32))} 1.0))
            (def {} y
              (handle-effect {effect: random}
                (lit {type: (t-prim {} int32)} 42)
                (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))))
        "#;
        let src_b = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 32) (t-prim {} f32))} 1.0))
            (def {} y
              (handle-effect {effect: random}
                (lit {type: (t-prim {} int32)} 43)
                (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))))
        "#;
        let dag_a = lower(src_a);
        let dag_b = lower(src_b);
        let out_a = eval_tensor(&dag_a, &HashMap::new())
            .unwrap()
            .remove(&NodeId(dag_a.len() - 1))
            .unwrap();
        let out_b = eval_tensor(&dag_b, &HashMap::new())
            .unwrap()
            .remove(&NodeId(dag_b.len() - 1))
            .unwrap();
        assert_ne!(out_a, out_b);
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

    fn inputs_2x3() -> HashMap<String, TensorValue> {
        let mut inputs = HashMap::new();
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
        for x in &v.data {
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
            let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], row_f32(3), None);
            dag.add_node(op, vec![x], scalar_f32(), None);
            let errs = verify(&dag);
            assert!(
                errs.iter().any(|e| e.contains("axis 7")),
                "expected axis-out-of-range error, got: {errs:?}"
            );
        }
    }
}
