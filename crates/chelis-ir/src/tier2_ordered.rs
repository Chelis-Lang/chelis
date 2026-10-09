//! Tier 2 lowerings for `diagonal`, `trace`, `cumsum`, and `einsum`, whose
//! atoms pin an exact selection, scan, or contraction order ([05-OP-53],
//! [05-OP-33], [05-OP-51]).
//!
//! Each lowering composes existing Tier 1 movement and arithmetic nodes so
//! that the operation's adjoint follows from the IR graph; no backend
//! supplies an adjoint (spec/05 §5). Every graph here is built only from
//! statically known extents; a runtime extent is an [`OrderedGap`] the caller
//! rejects (chelis#3378).
//!
//! Float forms only for `trace`, `cumsum`, and `einsum`: their integer forms are
//! forward-only and check overflow at every arithmetic step in the atom's
//! order, which an elementwise IR graph does not reproduce trap for trap
//! (chelis#3377).

use chelis_types::types::Prim;

use crate::dag::{Dag, DimInfo, NodeId, Owner, RiscOp, RtDim, TensorType};
use crate::tier2::add_synth;

/// Why an ordered lowering produced no graph.
pub enum OrderedGap {
    /// An integer form, forward-only and kept on its checked host kernel.
    IntegerForm(String),
    /// An operand axis whose extent is known only at run time.
    RuntimeExtent(String),
    /// Arguments the checker should have rejected.
    Malformed(String),
}

impl From<String> for OrderedGap {
    fn from(reason: String) -> Self {
        Self::Malformed(reason)
    }
}

fn static_dims(ty: &TensorType, op: &str) -> Result<Vec<usize>, OrderedGap> {
    ty.dims
        .iter()
        .map(|dim| match dim {
            DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => Ok(*n),
            DimInfo::Named(name, None) => Err(OrderedGap::RuntimeExtent(format!(
                "`{op}` operand axis `{name}` is a runtime extent"
            ))),
        })
        .collect()
}

fn lit_ty(dims: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().map(|&n| DimInfo::Lit(n)).collect(),
        precision,
    }
}

struct Builder<'a> {
    owner: Owner,
    dag: &'a mut Dag,
    span: Option<&'a str>,
}

impl Builder<'_> {
    fn node(&mut self, op: RiscOp, inputs: Vec<NodeId>, dims: &[usize], precision: Prim) -> NodeId {
        add_synth(
            self.owner,
            self.dag,
            op,
            inputs,
            lit_ty(dims, precision),
            self.span,
        )
    }

    fn cast(&mut self, x: NodeId, dims: &[usize], from: Prim, to: Prim) -> NodeId {
        if from == to {
            return x;
        }
        self.node(RiscOp::Cast { new_precision: to }, vec![x], dims, to)
    }

    /// `out[i] = x[axes[i]]`; the identity permutation emits nothing.
    fn permute(
        &mut self,
        x: NodeId,
        dims: &[usize],
        axes: Vec<usize>,
        precision: Prim,
    ) -> (NodeId, Vec<usize>) {
        let out: Vec<usize> = axes.iter().map(|&axis| dims[axis]).collect();
        if axes.iter().enumerate().all(|(i, &axis)| i == axis) {
            return (x, out);
        }
        (
            self.node(RiscOp::Permute { axes }, vec![x], &out, precision),
            out,
        )
    }

    fn reshape(
        &mut self,
        x: NodeId,
        dims: &[usize],
        target: Vec<usize>,
        precision: Prim,
    ) -> NodeId {
        if dims == target.as_slice() {
            return x;
        }
        let new_shape = target.iter().map(|&n| RtDim::Lit(n)).collect();
        self.node(RiscOp::Reshape { new_shape }, vec![x], &target, precision)
    }

    fn shrink_axis(
        &mut self,
        x: NodeId,
        dims: &[usize],
        axis: usize,
        start: usize,
        end: usize,
        precision: Prim,
    ) -> NodeId {
        if start == 0 && end == dims[axis] {
            return x;
        }
        let bounds = dims
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                if i == axis {
                    (RtDim::Lit(start), RtDim::Lit(end))
                } else {
                    (RtDim::Lit(0), RtDim::Lit(n))
                }
            })
            .collect();
        let mut out = dims.to_vec();
        out[axis] = end - start;
        self.node(RiscOp::Shrink { bounds }, vec![x], &out, precision)
    }

    fn pad_axis(
        &mut self,
        x: NodeId,
        dims: &[usize],
        axis: usize,
        before: usize,
        after: usize,
        precision: Prim,
    ) -> NodeId {
        let padding = (0..dims.len())
            .map(|i| {
                if i == axis {
                    (RtDim::Lit(before), RtDim::Lit(after))
                } else {
                    (RtDim::Lit(0), RtDim::Lit(0))
                }
            })
            .collect();
        let mut out = dims.to_vec();
        out[axis] += before + after;
        self.node(
            RiscOp::zero_pad(precision, padding),
            vec![x],
            &out,
            precision,
        )
    }
}

/// [05-OP-53] `diagonal(x, axis1, axis2)` for distinct normalized axes, as
/// pure movement: move both axes last, trim them to `k = min(m, n)`,
/// flatten the `k x k` block, and stride by `k + 1` to read the cells with
/// equal coordinates. The diagonal axis takes `axis1`'s place with `axis2`
/// removed. Every node preserves stored bits, so every dtype is admitted;
/// the float adjoint (stride, reshape, shrink, and permute adjoints)
/// scatters `g` into an otherwise zero tensor.
pub fn lower_diagonal(
    owner: Owner,
    dag: &mut Dag,
    x: NodeId,
    x_ty: &TensorType,
    axis1: usize,
    axis2: usize,
    span: Option<&str>,
) -> Result<NodeId, OrderedGap> {
    let dims = static_dims(x_ty, "diagonal")?;
    let rank = dims.len();
    if axis1 == axis2 || axis1 >= rank || axis2 >= rank {
        return Err(format!(
            "`diagonal` axes ({axis1}, {axis2}) are not two distinct axes of a rank-{rank} operand"
        )
        .into());
    }
    let precision = x_ty.precision;
    let mut b = Builder { owner, dag, span };
    let mut axes: Vec<usize> = (0..rank).filter(|&a| a != axis1 && a != axis2).collect();
    axes.extend([axis1, axis2]);
    let (moved, moved_dims) = b.permute(x, &dims, axes, precision);
    let k = dims[axis1].min(dims[axis2]);
    let rest = rank - 2;
    let trimmed = b.shrink_axis(moved, &moved_dims, rest, 0, k, precision);
    let mut trimmed_dims = moved_dims.clone();
    trimmed_dims[rest] = k;
    let trimmed = b.shrink_axis(trimmed, &trimmed_dims, rest + 1, 0, k, precision);
    trimmed_dims[rest + 1] = k;
    let mut flat_dims = trimmed_dims[..rest].to_vec();
    flat_dims.push(k * k);
    let flat = b.reshape(trimmed, &trimmed_dims, flat_dims.clone(), precision);
    let mut diag_dims = flat_dims.clone();
    diag_dims[rest] = k;
    let diag = if k > 1 {
        let mut strides = vec![RtDim::Lit(1); rest];
        strides.push(RtDim::Lit(k + 1));
        b.node(
            RiscOp::Stride { strides },
            vec![flat],
            &diag_dims,
            precision,
        )
    } else {
        flat
    };
    // Move the trailing diagonal axis to axis1's position in the source
    // order with axis2 removed.
    let position = if axis1 < axis2 { axis1 } else { axis1 - 1 };
    let mut back: Vec<usize> = (0..rest).collect();
    back.insert(position, rest);
    Ok(b.permute(diag, &diag_dims, back, precision).0)
}

/// [05-OP-33] float `trace(x, axis1, axis2)`: [`lower_diagonal`] followed by
/// [05-OP-30]'s canonical balanced tree at the default accumulator over the
/// diagonal axis, finalized to sum's result dtype. The adjoint expands the
/// cotangent along the diagonal and scatters it onto the selected cells.
pub fn lower_trace(
    owner: Owner,
    dag: &mut Dag,
    x: NodeId,
    x_ty: &TensorType,
    axis1: usize,
    axis2: usize,
    span: Option<&str>,
) -> Result<NodeId, OrderedGap> {
    let precision = x_ty.precision;
    if !precision.is_float() {
        return Err(OrderedGap::IntegerForm(format!(
            "`trace` over `{}` is forward-only",
            precision.name()
        )));
    }
    let diag = lower_diagonal(owner, dag, x, x_ty, axis1, axis2, span)?;
    let mut diag_dims = static_dims(x_ty, "trace")?;
    let k = diag_dims[axis1].min(diag_dims[axis2]);
    diag_dims[axis1] = k;
    diag_dims.remove(axis2);
    let position = if axis1 < axis2 { axis1 } else { axis1 - 1 };
    let accumulator = precision.default_reduce_sum_accumulator()?;
    let result = precision.sum_result_precision(accumulator);
    let mut out_dims = diag_dims.clone();
    out_dims.remove(position);
    let mut b = Builder { owner, dag, span };
    let sum = b.node(
        RiscOp::Sum {
            axis: position,
            accumulator,
        },
        vec![diag],
        &out_dims,
        accumulator,
    );
    Ok(b.cast(sum, &out_dims, accumulator, result))
}

/// [05-OP-33] float `cumsum(x, axis)`: in increasing axis order an exact-zero
/// accumulator adds each input at the accumulator dtype, and every prefix is
/// finalized to the result dtype. The graph is that sequential chain over
/// unit slabs of the axis, `r_0 = 0 + x_0` and `r_k = r_(k-1) + x_k`.
///
/// Slabs are cut and the prefixes rejoined by balanced halving, so the
/// adjoint's work is `O(numel * log n)` rather than one zero-padded
/// contribution per slab. Rejoining adds disjoint zero-padded halves; a
/// prefix is never `-0` (it starts from `+0`), so each join is exact. The
/// reverse derivative of the chain is the inclusive decreasing-axis scan of
/// the cotangent at the same accumulator dtype, finalized at the operand
/// dtype by the leading cast's adjoint.
pub fn lower_cumsum(
    owner: Owner,
    dag: &mut Dag,
    x: NodeId,
    x_ty: &TensorType,
    axis: usize,
    span: Option<&str>,
) -> Result<NodeId, OrderedGap> {
    let precision = x_ty.precision;
    if !precision.is_float() {
        return Err(OrderedGap::IntegerForm(format!(
            "`cumsum` over `{}` is forward-only",
            precision.name()
        )));
    }
    let dims = static_dims(x_ty, "cumsum")?;
    if axis >= dims.len() {
        return Err(format!(
            "`cumsum` axis {axis} is out of range for rank {}",
            dims.len()
        )
        .into());
    }
    let accumulator = precision.default_reduce_sum_accumulator()?;
    let result = precision.sum_result_precision(accumulator);
    let mut b = Builder { owner, dag, span };
    let n = dims[axis];
    if n == 0 {
        return Ok(b.cast(x, &dims, precision, result));
    }
    let wide = b.cast(x, &dims, precision, accumulator);
    let mut slab_dims = dims.clone();
    slab_dims[axis] = 1;
    let mut slabs = Vec::with_capacity(n);
    split_slabs(&mut b, wide, &dims, axis, accumulator, &mut slabs);
    let mut running = b.node(
        RiscOp::synth_const(accumulator, 0.0),
        vec![],
        &slab_dims,
        accumulator,
    );
    let mut prefixes = Vec::with_capacity(n);
    for slab in slabs {
        // Addition commutes exactly. The slab is the left operand so an axis
        // query reads its extent through the O(log n) slab cut instead of
        // walking the whole running chain, which made extent resolution over
        // every prefix quadratic in n.
        running = b.node(RiscOp::Add, vec![slab, running], &slab_dims, accumulator);
        prefixes.push(running);
    }
    let joined = join_slabs(&mut b, &prefixes, &slab_dims, axis, accumulator);
    Ok(b.cast(joined, &dims, accumulator, result))
}

fn split_slabs(
    b: &mut Builder<'_>,
    x: NodeId,
    dims: &[usize],
    axis: usize,
    precision: Prim,
    out: &mut Vec<NodeId>,
) {
    let n = dims[axis];
    if n == 1 {
        out.push(x);
        return;
    }
    let mid = n / 2;
    let left = b.shrink_axis(x, dims, axis, 0, mid, precision);
    let right = b.shrink_axis(x, dims, axis, mid, n, precision);
    let mut left_dims = dims.to_vec();
    left_dims[axis] = mid;
    let mut right_dims = dims.to_vec();
    right_dims[axis] = n - mid;
    split_slabs(b, left, &left_dims, axis, precision, out);
    split_slabs(b, right, &right_dims, axis, precision, out);
}

fn join_slabs(
    b: &mut Builder<'_>,
    slabs: &[NodeId],
    slab_dims: &[usize],
    axis: usize,
    precision: Prim,
) -> NodeId {
    if slabs.len() == 1 {
        return slabs[0];
    }
    let mid = slabs.len() / 2;
    let left = join_slabs(b, &slabs[..mid], slab_dims, axis, precision);
    let right = join_slabs(b, &slabs[mid..], slab_dims, axis, precision);
    let mut left_dims = slab_dims.to_vec();
    left_dims[axis] = mid;
    let mut right_dims = slab_dims.to_vec();
    right_dims[axis] = slabs.len() - mid;
    let left = b.pad_axis(left, &left_dims, axis, 0, slabs.len() - mid, precision);
    let right = b.pad_axis(right, &right_dims, axis, mid, 0, precision);
    let mut dims = slab_dims.to_vec();
    dims[axis] = slabs.len();
    b.node(RiscOp::Add, vec![left, right], &dims, precision)
}

/// [05-OP-33] / [05-OP-51] float `einsum(equation, a, b)` at `accumulator`.
///
/// Repeated labels inside one operand first take that operand's diagonal.
/// The contraction then runs over the canonical label order `F = output ++
/// reductions`, where reductions are the non-output labels in
/// first-occurrence order scanning left then right. Each operand converts
/// exactly to the accumulator, is permuted to `F` order, and is broadcast
/// along its missing labels as ONE flattened axis (in `F` order) before it
/// is reshaped out. Each product finalizes at the accumulator, and the
/// reduction labels are flattened row-major into one axis that the
/// canonical balanced tree sums, matching the atom's assignment order. No
/// reduction label means no sum.
///
/// The adjoint therefore contracts the cotangent with the other operand,
/// and each operand cell's contributions arrive on the one flattened
/// broadcast axis in forward output-then-reduction order, combined by the
/// same balanced tree at the accumulator dtype before finalizing at the
/// operand dtype.
pub fn lower_einsum(
    owner: Owner,
    dag: &mut Dag,
    equation: &str,
    operands: [(NodeId, &TensorType); 2],
    accumulator: Option<Prim>,
    span: Option<&str>,
) -> Result<NodeId, OrderedGap> {
    let precision = operands[0].1.precision;
    if operands[1].1.precision != precision {
        return Err("`einsum` operands must share one dtype".to_string().into());
    }
    if !precision.is_float() {
        return Err(OrderedGap::IntegerForm(format!(
            "`einsum` over `{}` is forward-only",
            precision.name()
        )));
    }
    let accumulator = match accumulator {
        Some(accumulator) => accumulator,
        None => precision.default_reduce_sum_accumulator()?,
    };
    let result = precision.sum_result_precision(accumulator);
    let (inputs, output) = equation
        .split_once("->")
        .ok_or_else(|| format!("`einsum` equation `{equation}` has no `->`"))?;
    let (lhs, rhs) = inputs
        .split_once(',')
        .ok_or_else(|| format!("`einsum` equation `{equation}` does not name two operands"))?;
    let labels = [lhs.as_bytes().to_vec(), rhs.as_bytes().to_vec()];
    let output = output.as_bytes().to_vec();

    let mut extent = [None::<usize>; 256];
    let mut operand_dims = Vec::with_capacity(2);
    for ((_, ty), labels) in operands.iter().zip(&labels) {
        let dims = static_dims(ty, "einsum")?;
        if dims.len() != labels.len() {
            return Err(
                format!("`einsum` equation `{equation}` does not match operand ranks").into(),
            );
        }
        for (&label, &n) in labels.iter().zip(&dims) {
            match extent[usize::from(label)] {
                Some(seen) if seen != n => {
                    return Err(format!(
                        "`einsum` label `{}` has inconsistent extents",
                        char::from(label)
                    )
                    .into());
                }
                _ => extent[usize::from(label)] = Some(n),
            }
        }
        operand_dims.push(dims);
    }
    let mut full = output.clone();
    for &label in labels[0].iter().chain(&labels[1]) {
        if !full.contains(&label) {
            full.push(label);
        }
    }
    for &label in &output {
        if extent[usize::from(label)].is_none() {
            return Err(format!(
                "`einsum` output label `{}` occurs in no operand",
                char::from(label)
            )
            .into());
        }
    }
    let size = |label: u8| extent[usize::from(label)].expect("every label has an extent");
    let full_dims: Vec<usize> = full.iter().map(|&label| size(label)).collect();

    let mut b = Builder { owner, dag, span };
    let mut aligned = Vec::with_capacity(2);
    for (((node, ty), labels), dims) in operands.iter().zip(labels).zip(operand_dims) {
        let (mut node, mut labels, mut dims) = (*node, labels, dims);
        // Repeated labels denote a diagonal: keep the first occurrence.
        while let Some((first, second)) = repeated_label(&labels) {
            let diag_ty = lit_ty(&dims, ty.precision);
            node = lower_diagonal(owner, b.dag, node, &diag_ty, first, second, span)?;
            labels.remove(second);
            dims.remove(second);
        }
        node = b.cast(node, &dims, precision, accumulator);
        // Present labels in F order, then the missing ones as one trailing
        // broadcast axis, then split and interleave into F order.
        let mut present: Vec<usize> = (0..labels.len()).collect();
        present.sort_by_key(|&i| position(&full, labels[i]));
        let (moved, moved_dims) = b.permute(node, &dims, present.clone(), accumulator);
        let present_labels: Vec<u8> = present.iter().map(|&i| labels[i]).collect();
        let missing: Vec<u8> = full
            .iter()
            .copied()
            .filter(|l| !present_labels.contains(l))
            .collect();
        if missing.is_empty() {
            aligned.push(moved);
            continue;
        }
        let broadcast: usize = missing.iter().map(|&l| size(l)).product();
        let mut expanded_dims = moved_dims.clone();
        expanded_dims.push(broadcast);
        let expanded = b.node(
            RiscOp::Expand {
                axis: moved_dims.len(),
                size: RtDim::Lit(broadcast),
            },
            vec![moved],
            &expanded_dims,
            accumulator,
        );
        let mut split_dims = moved_dims.clone();
        split_dims.extend(missing.iter().map(|&l| size(l)));
        let split = b.reshape(expanded, &expanded_dims, split_dims.clone(), accumulator);
        let split_labels: Vec<u8> = present_labels.iter().chain(&missing).copied().collect();
        let order = full.iter().map(|&l| position(&split_labels, l)).collect();
        aligned.push(b.permute(split, &split_dims, order, accumulator).0);
    }
    let product = b.node(RiscOp::Mul, aligned, &full_dims, accumulator);
    let out_dims: Vec<usize> = output.iter().map(|&l| size(l)).collect();
    let reduced = if full.len() == output.len() {
        product
    } else {
        let reduction: usize = full_dims[output.len()..].iter().product();
        let mut flat_dims = out_dims.clone();
        flat_dims.push(reduction);
        let flat = b.reshape(product, &full_dims, flat_dims, accumulator);
        b.node(
            RiscOp::Sum {
                axis: output.len(),
                accumulator,
            },
            vec![flat],
            &out_dims,
            accumulator,
        )
    };
    Ok(b.cast(reduced, &out_dims, accumulator, result))
}

fn repeated_label(labels: &[u8]) -> Option<(usize, usize)> {
    (0..labels.len()).find_map(|i| {
        labels[i + 1..]
            .iter()
            .position(|&l| l == labels[i])
            .map(|offset| (i, i + 1 + offset))
    })
}

fn position(labels: &[u8], label: u8) -> usize {
    labels
        .iter()
        .position(|&l| l == label)
        .expect("label occurs in the canonical order")
}
