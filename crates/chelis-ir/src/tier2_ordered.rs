//! Tier 2 lowerings for `diagonal`, `trace`, `cumsum`, and `einsum`, whose
//! atoms pin an exact selection, scan, or contraction order ([05-OP-53],
//! [05-OP-33], [05-OP-51]).
//!
//! Movement and contraction lowerings compose Tier 1 nodes. The directional
//! scan retains its ordered identity so its adjoint reverses traversal without
//! introducing an additive base. Scans accept runtime extents, and so does an
//! einsum's matmul-shaped contraction form on the labels only one operand
//! carries; the other ordered lowerings require statically known extents
//! (chelis#3378).
//!
//! Integer scans are forward-only and check every accumulator addition.
//! Integer trace and einsum retain their checked host paths (chelis#3377).

use chelis_types::types::Prim;

use crate::dag::{Dag, DimInfo, NodeId, Owner, RiscOp, RtAxis, RtDim, TensorType};
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

/// [05-OP-33] ordered scan. Keeping the identity through AD and vmap
/// preserves the inclusive reverse rule without padded cotangent additions.
pub fn lower_cumsum(
    owner: Owner,
    dag: &mut Dag,
    x: NodeId,
    x_ty: &TensorType,
    axis: usize,
    span: Option<&str>,
) -> Result<NodeId, OrderedGap> {
    let p = x_ty.precision;
    if !(p.is_float() || p.is_integer()) {
        return Err(format!(
            "cumsum requires signed-integer or float input, got {}",
            p.name()
        )
        .into());
    }
    if axis >= x_ty.dims.len() {
        return Err(format!(
            "cumsum axis {axis} is out of range for rank {}",
            x_ty.dims.len()
        )
        .into());
    }
    let result = p.default_reduce_sum_result_precision()?;
    Ok(add_synth(
        owner,
        dag,
        RiscOp::Cumsum {
            axis,
            direction: crate::dag::ScanDirection::Increasing,
        },
        vec![x],
        TensorType {
            dims: x_ty.dims.clone(),
            precision: result,
        },
        span,
    ))
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

/// A float `einsum` whose [05-OP-33] contraction graph is spec/05 §4.1's
/// product sum over one axis: one reduction label `k` that both operands
/// carry, an output ending in a row label `m` of one operand and a column
/// label `n` of the other, and no diagonal. Every label both operands carry
/// has one static extent, so the contraction owes no runtime label check.
///
/// The products are then `[lead.., m, k, n]`, and the reduction visits `k`
/// in increasing order, the atom's one assignment order for a single
/// reduction label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EinsumContraction {
    /// Whether the product's left operand is the einsum's right operand.
    /// Both orders give the same bits: float multiplication commutes, and
    /// every product finalizes its NaN canonically ([04-NUM-2]).
    swapped: bool,
    /// The output labels before `m` and `n`, in output order.
    lead: Vec<u8>,
    m: u8,
    k: u8,
    n: u8,
    /// The left and right product operands' labels.
    labels: [Vec<u8>; 2],
}

/// The contraction form of `einsum(equation, lhs, rhs)`, when it has one.
pub fn einsum_contraction(
    equation: &str,
    lhs: &TensorType,
    rhs: &TensorType,
) -> Option<EinsumContraction> {
    if lhs.precision != rhs.precision || !lhs.precision.is_float() {
        return None;
    }
    let (inputs, output) = equation.split_once("->")?;
    let (a, b) = inputs.split_once(',')?;
    let [a, b, output] = [a, b, output].map(|labels| labels.as_bytes().to_vec());
    let distinct = |labels: &[u8]| {
        labels.iter().all(u8::is_ascii_lowercase)
            && (0..labels.len()).all(|i| !labels[i + 1..].contains(&labels[i]))
    };
    if !distinct(&a)
        || !distinct(&b)
        || !distinct(&output)
        || a.len() != lhs.dims.len()
        || b.len() != rhs.dims.len()
        || output.iter().any(|l| !a.contains(l) && !b.contains(l))
    {
        return None;
    }
    for (i, label) in a.iter().enumerate() {
        if let Some(j) = b.iter().position(|l| l == label) {
            match (static_extent(&lhs.dims[i]), static_extent(&rhs.dims[j])) {
                (Some(x), Some(y)) if x == y => {}
                _ => return None,
            }
        }
    }
    let mut reductions = a.iter().chain(&b).filter(|l| !output.contains(l));
    let k = *reductions.next()?;
    if reductions.any(|&l| l != k) || !a.contains(&k) || !b.contains(&k) {
        return None;
    }
    let [.., m, n] = output.as_slice() else {
        return None;
    };
    let (m, n) = (*m, *n);
    let only =
        |side: &[u8], other: &[u8], label: u8| side.contains(&label) && !other.contains(&label);
    let swapped = if only(&a, &b, m) && only(&b, &a, n) {
        false
    } else if only(&b, &a, m) && only(&a, &b, n) {
        true
    } else {
        return None;
    };
    Some(EinsumContraction {
        swapped,
        lead: output[..output.len() - 2].to_vec(),
        m,
        k,
        n,
        labels: if swapped { [b, a] } else { [a, b] },
    })
}

fn static_extent(dim: &DimInfo) -> Option<usize> {
    match dim {
        DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => Some(*n),
        DimInfo::Named(_, None) => None,
    }
}

/// [05-OP-33] `einsum` in its [`EinsumContraction`] form, built as the
/// graph spec/05 §4.1 gives `matmul`: each operand converts exactly to the
/// accumulator, is permuted to `[lead.., m, k]` or `[lead.., k, n]` and
/// expanded along the lead labels it lacks, and the two meet as
/// `Sum(Mul(Expand(lhs), Expand(rhs)))` over `k`, which a lane runs as its
/// product-sum kernel. Each product finalizes at the accumulator and the `k`
/// products enter the canonical balanced tree in increasing order, the bits
/// [05-OP-33] fixes; the result finalizes once at `sum_result(p, accumulator)`.
///
/// A label only one operand carries may have a runtime extent; an expansion
/// along it reads the extent from that operand's shape.
pub fn lower_einsum_contraction(
    owner: Owner,
    dag: &mut Dag,
    plan: &EinsumContraction,
    operands: [(NodeId, &TensorType); 2],
    accumulator: Option<Prim>,
    span: Option<&str>,
) -> Result<NodeId, String> {
    let operands = if plan.swapped {
        [operands[1], operands[0]]
    } else {
        operands
    };
    let precision = operands[0].1.precision;
    let accumulator = match accumulator {
        Some(accumulator) => accumulator,
        None => precision.default_reduce_sum_accumulator()?,
    };
    let result = precision.sum_result_precision(accumulator);
    // The first operand carrying `label`, its axis, and that axis's extent.
    let source = |label: u8| {
        (0..2)
            .find_map(|side| {
                let axis = plan.labels[side].iter().position(|&l| l == label)?;
                Some((side, axis, operands[side].1.dims[axis].clone()))
            })
            .expect("every contraction label occurs in an operand")
    };
    let at = |dims: &[DimInfo]| TensorType {
        dims: dims.to_vec(),
        precision: accumulator,
    };
    let expand = |dag: &mut Dag, node: NodeId, dims: &mut Vec<DimInfo>, axis: usize, label: u8| {
        let (side, source_axis, dim) = source(label);
        let (size, inputs) = match static_extent(&dim) {
            Some(n) => (RtDim::Lit(n), vec![node]),
            None => (
                RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(
                        i32::try_from(source_axis).expect("tensor rank fits the i32 axis carrier"),
                    ),
                },
                vec![node, operands[side].0],
            ),
        };
        dims.insert(axis, dim);
        add_synth(
            owner,
            dag,
            RiscOp::Expand { axis, size },
            inputs,
            at(dims),
            span,
        )
    };
    let mut views = Vec::with_capacity(2);
    for (side, &(mut node, ty)) in operands.iter().enumerate() {
        let labels = &plan.labels[side];
        if precision != accumulator {
            let cast = RiscOp::Cast {
                new_precision: accumulator,
            };
            node = add_synth(owner, dag, cast, vec![node], at(&ty.dims), span);
        }
        let tail = if side == 0 {
            [plan.m, plan.k]
        } else {
            [plan.k, plan.n]
        };
        let axes: Vec<usize> = plan
            .lead
            .iter()
            .filter(|l| labels.contains(l))
            .chain(&tail)
            .map(|&l| labels.iter().position(|&own| own == l).expect("own label"))
            .collect();
        let mut dims: Vec<DimInfo> = axes.iter().map(|&axis| ty.dims[axis].clone()).collect();
        if axes.iter().enumerate().any(|(i, &axis)| i != axis) {
            node = add_synth(
                owner,
                dag,
                RiscOp::Permute { axes },
                vec![node],
                at(&dims),
                span,
            );
        }
        for (axis, &label) in plan.lead.iter().enumerate() {
            if !labels.contains(&label) {
                node = expand(dag, node, &mut dims, axis, label);
            }
        }
        let (axis, label) = if side == 0 {
            (plan.lead.len() + 2, plan.n)
        } else {
            (plan.lead.len(), plan.m)
        };
        views.push(expand(dag, node, &mut dims, axis, label));
    }
    let mut dims: Vec<DimInfo> = plan
        .lead
        .iter()
        .chain(&[plan.m, plan.k, plan.n])
        .map(|&l| source(l).2)
        .collect();
    let product = add_synth(owner, dag, RiscOp::Mul, views, at(&dims), span);
    let axis = plan.lead.len() + 1;
    dims.remove(axis);
    let sum = RiscOp::Sum { axis, accumulator };
    let sum = add_synth(owner, dag, sum, vec![product], at(&dims), span);
    if result == accumulator {
        return Ok(sum);
    }
    let cast = RiscOp::Cast {
        new_precision: result,
    };
    let ty = TensorType {
        dims,
        precision: result,
    };
    Ok(add_synth(owner, dag, cast, vec![sum], ty, span))
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

#[cfg(test)]
mod einsum_contraction_tests {
    use super::*;
    use crate::fuse::reduction_inlined_products;
    use crate::verify;

    fn ty(dims: &[DimInfo], precision: Prim) -> TensorType {
        TensorType {
            dims: dims.to_vec(),
            precision,
        }
    }

    fn lit(dims: &[usize]) -> Vec<DimInfo> {
        dims.iter().map(|&n| DimInfo::Lit(n)).collect()
    }

    fn runtime(name: &str) -> DimInfo {
        DimInfo::Named(name.to_string(), None)
    }

    fn plan(equation: &str, a: &[DimInfo], b: &[DimInfo], p: Prim) -> Option<EinsumContraction> {
        einsum_contraction(equation, &ty(a, p), &ty(b, p))
    }

    /// [05-OP-33]'s one-reduction-label contractions whose output ends in a
    /// row of one operand and a column of the other.
    #[test]
    fn matmul_shaped_float_contractions_have_the_contraction_form() {
        let f32 = Prim::F32;
        for (equation, a, b) in [
            ("mk,kn->mn", lit(&[3, 37]), lit(&[37, 5])),
            ("amk,kn->amn", lit(&[2, 3, 37]), lit(&[37, 5])),
            ("mk,nk->mn", lit(&[3, 37]), lit(&[5, 37])),
            ("km,kn->mn", lit(&[37, 3]), lit(&[37, 5])),
            ("bmk,bkn->bmn", lit(&[2, 3, 17]), lit(&[2, 17, 4])),
            ("mk,bkn->bmn", lit(&[3, 9]), lit(&[2, 9, 4])),
            ("ij,jk->ki", lit(&[3, 7]), lit(&[7, 5])),
            (
                "mk,kn->mn",
                vec![runtime("m"), DimInfo::Lit(8)],
                lit(&[8, 5]),
            ),
            (
                "amk,kn->amn",
                vec![
                    runtime("a"),
                    runtime("m"),
                    DimInfo::Named("k".into(), Some(8)),
                ],
                vec![DimInfo::Lit(8), runtime("n")],
            ),
        ] {
            assert!(plan(equation, &a, &b, f32).is_some(), "`{equation}`");
        }
        for p in [Prim::F64, Prim::F16, Prim::Bf16] {
            assert!(
                plan("mk,kn->mn", &lit(&[3, 7]), &lit(&[7, 5]), p).is_some(),
                "{p:?}"
            );
        }
        let swapped = plan("nk,mk->mn", &lit(&[5, 7]), &lit(&[3, 7]), f32).unwrap();
        assert!(swapped.swapped);
        assert!(
            !plan("mk,kn->mn", &lit(&[3, 7]), &lit(&[7, 5]), f32)
                .unwrap()
                .swapped
        );
    }

    /// Everything else keeps the runtime kernel: integers, diagonals, no or
    /// several reduction labels, an output that does not end in a row and a
    /// column, and any label both operands carry without one static extent.
    #[test]
    fn every_other_einsum_has_no_contraction_form() {
        let f32 = Prim::F32;
        for p in [Prim::Int32, Prim::Int64, Prim::Int8] {
            assert!(
                plan("mk,kn->mn", &lit(&[3, 7]), &lit(&[7, 5]), p).is_none(),
                "{p:?}"
            );
        }
        assert!(
            einsum_contraction(
                "mk,kn->mn",
                &ty(&lit(&[3, 7]), f32),
                &ty(&lit(&[7, 5]), Prim::F64)
            )
            .is_none()
        );
        for (equation, a, b) in [
            ("ii,i->i", lit(&[4, 4]), lit(&[4])),
            ("mkk,kn->mn", lit(&[3, 4, 4]), lit(&[4, 5])),
            ("i,j->ij", lit(&[3]), lit(&[5])),
            ("ij,ij->", lit(&[3, 7]), lit(&[3, 7])),
            ("ijk,jk->i", lit(&[3, 4, 5]), lit(&[4, 5])),
            ("mjk,kn->mn", lit(&[3, 2, 7]), lit(&[7, 5])),
            ("ij,j->i", lit(&[5, 7]), lit(&[7])),
            ("bmk,bkn->mbn", lit(&[2, 3, 7]), lit(&[2, 7, 4])),
            ("mk,kn->m", lit(&[3, 7]), lit(&[7, 5])),
            (
                "mk,kn->mn",
                vec![DimInfo::Lit(3), runtime("k")],
                vec![runtime("k"), DimInfo::Lit(5)],
            ),
            (
                "bmk,bkn->bmn",
                vec![runtime("b"), DimInfo::Lit(3), DimInfo::Lit(7)],
                vec![runtime("b"), DimInfo::Lit(7), DimInfo::Lit(4)],
            ),
            ("mk,kn->mn", lit(&[3, 7]), lit(&[6, 5])),
            ("mk,kn->mn", lit(&[3, 7, 1]), lit(&[7, 5])),
            ("mk,kn", lit(&[3, 7]), lit(&[7, 5])),
            ("mK,Kn->mn", lit(&[3, 7]), lit(&[7, 5])),
            ("mk,kn->mnn", lit(&[3, 7]), lit(&[7, 5])),
            ("mk,kn->mx", lit(&[3, 7]), lit(&[7, 5])),
        ] {
            assert!(plan(equation, &a, &b, f32).is_none(), "`{equation}`");
        }
    }

    /// The graph is spec/05 §4.1's product sum, which the lanes' fused
    /// product-sum kernels recognize in its blocked row-and-column layout.
    #[test]
    fn the_contraction_lowers_to_the_blocked_product_sum() {
        for (equation, a, b, p) in [
            ("amk,kn->amn", lit(&[2, 3, 37]), lit(&[37, 5]), Prim::F32),
            (
                "mk,nk->mn",
                vec![runtime("m"), DimInfo::Lit(9)],
                vec![runtime("n"), DimInfo::Lit(9)],
                Prim::F64,
            ),
            ("nk,mk->mn", lit(&[5, 7]), lit(&[3, 7]), Prim::F16),
            ("mk,bkn->bmn", lit(&[3, 9]), lit(&[2, 9, 4]), Prim::Bf16),
        ] {
            let plan = plan(equation, &a, &b, p).expect("contraction form");
            let mut dag = Dag::new();
            let owner = Owner::from(dag.declare("test"));
            let a_ty = ty(&a, p);
            let b_ty = ty(&b, p);
            let x = dag.add_node(
                owner,
                RiscOp::Load { name: "x".into() },
                vec![],
                a_ty.clone(),
                None,
            );
            let y = dag.add_node(
                owner,
                RiscOp::Load { name: "y".into() },
                vec![],
                b_ty.clone(),
                None,
            );
            let root = lower_einsum_contraction(
                owner,
                &mut dag,
                &plan,
                [(x, &a_ty), (y, &b_ty)],
                None,
                None,
            )
            .expect("lowered");
            dag.set_roots(vec![root]);
            assert!(
                verify::verify(&dag).is_empty(),
                "`{equation}`: {:?}",
                verify::verify(&dag)
            );
            let inlined = reduction_inlined_products(&dag);
            assert_eq!(inlined.len(), 1, "`{equation}` product sum");
            let product = dag.get(inlined[0].product).unwrap();
            let rank = product.output_type.dims.len();
            let accumulator = p.default_reduce_sum_accumulator().unwrap();
            assert_eq!(product.output_type.precision, accumulator);
            assert!(
                matches!(dag.get(inlined[0].sum).unwrap().op, RiscOp::Sum { axis, .. } if axis == rank - 2)
            );
            assert!(
                matches!(dag.get(inlined[0].views[0]).unwrap().op, RiscOp::Expand { axis, .. } if axis == rank - 1)
            );
            assert!(
                matches!(dag.get(inlined[0].views[1]).unwrap().op, RiscOp::Expand { axis, .. } if axis == rank - 3)
            );
            let root = dag.get(root).unwrap();
            assert_eq!(root.output_type.precision, p, "`{equation}` result dtype");
        }
    }
}
