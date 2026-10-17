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

use chelis_abi::metadata::ShapeMetadata;
use chelis_types::types::Prim;

use crate::dag::{Dag, DimExpr, DimInfo, NodeId, Owner, RiscOp, RtAxis, RtDim, TensorType};
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
/// product sum over one axis, with nothing materialized beyond an operand's
/// or the result's own size.
///
/// Its operand dtype is its resolved accumulator, so each product multiplies
/// the stored operands in place; a reduced float or a widened accumulator
/// would store an accumulator-width copy of each operand. Every shape the
/// graph plans has strides, an element count and a byte size that fit i64.
/// It has one reduction label `k` that both operands carry, no diagonal, and
/// an output `[batch.., rows.., m, n]`: `m` and `rows` are labels only the row
/// operand carries, `n` a label only the column operand carries, and `batch`
/// labels both carry. Every label both operands carry has one static extent,
/// so the contraction owes no runtime label check. The row operand's `rows`
/// and `m` flatten into one row axis, so the column operand is never copied
/// along a label it lacks; that flattened extent is static, or is one runtime
/// extent beside static extents of one.
///
/// The products are then `[batch.., rows * m, k, n]`, and the reduction
/// visits `k` in increasing order, the atom's one assignment order for a
/// single reduction label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EinsumContraction {
    /// Whether the product's left (row) operand is the einsum's right
    /// operand. Both orders give the same bits: float multiplication
    /// commutes, and every product finalizes its NaN canonically
    /// ([04-NUM-2]).
    swapped: bool,
    /// Output labels both operands carry, in output order.
    batch: Vec<u8>,
    /// Output labels before `m` only the row operand carries, in output
    /// order.
    rows: Vec<u8>,
    m: u8,
    k: u8,
    n: u8,
    /// The row and column operands' labels.
    labels: [Vec<u8>; 2],
}

/// The contraction form of `einsum(equation, lhs, rhs)`, when it has one.
pub fn einsum_contraction(
    equation: &str,
    lhs: &TensorType,
    rhs: &TensorType,
    accumulator: Option<Prim>,
) -> Option<EinsumContraction> {
    let precision = lhs.precision;
    if rhs.precision != precision || !precision.is_float() {
        return None;
    }
    // Only an operand already at the accumulator multiplies in place: any
    // other dtype would store an accumulator-width copy of each operand.
    let accumulator = match accumulator {
        Some(accumulator) => accumulator,
        None => precision.default_reduce_sum_accumulator().ok()?,
    };
    if accumulator != precision {
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
    let [lead @ .., m, n] = output.as_slice() else {
        return None;
    };
    let (m, n) = (*m, *n);
    let only =
        |side: &[u8], other: &[u8], label: u8| side.contains(&label) && !other.contains(&label);
    let (swapped, row, column, row_ty) = if only(&a, &b, m) && only(&b, &a, n) {
        (false, &a, &b, lhs)
    } else if only(&b, &a, m) && only(&a, &b, n) {
        (true, &b, &a, rhs)
    } else {
        return None;
    };
    // Lead labels are batch labels, then row-only labels; a column-only lead
    // label would copy the row operand along it.
    let batch_len = lead
        .iter()
        .take_while(|l| row.contains(l) && column.contains(l))
        .count();
    let (batch, rows) = lead.split_at(batch_len);
    if rows.iter().any(|l| column.contains(l) || !row.contains(l)) {
        return None;
    }
    let flat = flattened_rows(
        rows.iter()
            .chain([&m])
            .map(|l| &row_ty.dims[row.iter().position(|own| own == l).expect("row label")]),
    )?;
    let plan = EinsumContraction {
        swapped,
        batch: batch.to_vec(),
        rows: rows.to_vec(),
        m,
        k,
        n,
        labels: if swapped { [b, a] } else { [a, b] },
    };
    // Every shape the graph plans, the expansion included, has strides, an
    // element count and a byte size that fit i64.
    let (row_ty, column_ty) = if swapped { (rhs, lhs) } else { (lhs, rhs) };
    let shapes = plan.planned_shapes(row_ty, column_ty, flat);
    shapes
        .iter()
        .all(|dims| representable(dims, precision))
        .then_some(plan)
}

impl EinsumContraction {
    /// The flattened row operand, the product and expansions, and the output.
    fn planned_shapes(
        &self,
        row: &TensorType,
        column: &TensorType,
        flat: DimInfo,
    ) -> [Vec<DimInfo>; 3] {
        let dim = |ty: &TensorType, side: usize, label: u8| {
            ty.dims[self.labels[side]
                .iter()
                .position(|&l| l == label)
                .expect("own label")]
            .clone()
        };
        let batch: Vec<DimInfo> = self.batch.iter().map(|&l| dim(row, 0, l)).collect();
        let mut rows = batch.clone();
        rows.extend([flat, dim(row, 0, self.k)]);
        let mut product = rows.clone();
        product.push(dim(column, 1, self.n));
        let mut output = batch;
        output.extend(self.rows.iter().chain(&[self.m]).map(|&l| dim(row, 0, l)));
        output.push(dim(column, 1, self.n));
        [rows, product, output]
    }
}

/// Whether the static extents of `dims` pass the runtime's own
/// contiguous-metadata check at `precision`: their strides, element count and
/// byte size fit i64, an empty shape included. The runtime plan that reads a
/// runtime extent checks that extent's own factor.
fn representable(dims: &[DimInfo], precision: Prim) -> bool {
    let Ok(dtype) = precision.runtime_dtype() else {
        return false;
    };
    let extents: Option<Vec<i64>> = dims
        .iter()
        .filter_map(static_extent)
        .map(|n| i64::try_from(n).ok())
        .collect();
    extents.is_some_and(|extents| ShapeMetadata::contiguous(&extents, dtype).is_ok())
}

/// One row axis for `dims`: their static product, or their one runtime
/// extent when every other extent is one.
fn flattened_rows<'a>(dims: impl Iterator<Item = &'a DimInfo>) -> Option<DimInfo> {
    let mut product = DimExpr::Concrete(1);
    let mut runtime = None;
    for dim in dims {
        match static_extent(dim) {
            Some(_) => product = DimExpr::Mul(Box::new(product), Box::new(DimExpr::from(dim))),
            None if runtime.is_none() => runtime = Some(dim.clone()),
            None => return None,
        }
    }
    // An unrepresentable static product has no row axis.
    let product = product.as_concrete()?;
    match runtime {
        None => Some(DimInfo::Lit(product)),
        Some(dim) if product == 1 => Some(dim),
        Some(_) => None,
    }
}

fn static_extent(dim: &DimInfo) -> Option<usize> {
    match dim {
        DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => Some(*n),
        DimInfo::Named(_, None) => None,
    }
}

/// [05-OP-33] `einsum` in its [`EinsumContraction`] form, built as the
/// graph spec/05 §4.1 gives `matmul`. The operands, already at the
/// accumulator, are permuted to `[batch.., rows.., m, k]` or
/// `[batch.., k, n]`; the row operand's `rows.., m` reshape into one row axis.
/// The two meet as `Sum(Mul(Expand(row), Expand(column)))` over `k`, which a
/// lane runs as its product-sum kernel without storing either expansion, and
/// the sum reshapes back to the output. Each product finalizes at the
/// accumulator and the `k` products enter the canonical balanced tree in
/// increasing order, the bits [05-OP-33] fixes. No stored node holds more
/// bytes than an operand or the result.
///
/// A label only one operand carries may have a runtime extent; the column
/// expansion reads the row extent from the row operand's shape.
pub fn lower_einsum_contraction(
    owner: Owner,
    dag: &mut Dag,
    plan: &EinsumContraction,
    operands: [(NodeId, &TensorType); 2],
    accumulator: Option<Prim>,
    span: Option<&str>,
) -> Result<NodeId, String> {
    let [(row, row_ty), (column, column_ty)] = if plan.swapped {
        [operands[1], operands[0]]
    } else {
        operands
    };
    let precision = row_ty.precision;
    let accumulator = match accumulator {
        Some(accumulator) => accumulator,
        None => precision.default_reduce_sum_accumulator()?,
    };
    if accumulator != precision {
        return Err("the contraction form multiplies at the operand dtype".to_string());
    }
    let at = |dims: &[DimInfo]| TensorType {
        dims: dims.to_vec(),
        precision: accumulator,
    };
    let dim = |side: usize, label: u8| {
        let ty = if side == 0 { row_ty } else { column_ty };
        let axis = plan.labels[side].iter().position(|&l| l == label);
        ty.dims[axis.expect("own label")].clone()
    };
    // Permute one operand to `order`.
    let arrange = |dag: &mut Dag, side: usize, node: NodeId, ty: &TensorType, order: &[u8]| {
        let mut node = node;
        let labels = &plan.labels[side];
        let axes: Vec<usize> = order
            .iter()
            .map(|&l| labels.iter().position(|&own| own == l).expect("own label"))
            .collect();
        let dims: Vec<DimInfo> = axes.iter().map(|&axis| ty.dims[axis].clone()).collect();
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
        (node, dims)
    };
    let batch = plan.batch.len();
    let row_order: Vec<u8> = plan
        .batch
        .iter()
        .chain(&plan.rows)
        .chain(&[plan.m, plan.k])
        .copied()
        .collect();
    let (mut row, row_dims) = arrange(dag, 0, row, row_ty, &row_order);
    let flat = flattened_rows(row_dims[batch..row_dims.len() - 1].iter())
        .ok_or("`einsum` rows do not flatten to one axis")?;
    let mut flat_dims = row_dims[..batch].to_vec();
    flat_dims.extend([flat.clone(), dim(0, plan.k)]);
    if !plan.rows.is_empty() {
        let new_shape = flat_dims.iter().map(RtDim::from_dim_info).collect();
        row = add_synth(
            owner,
            dag,
            RiscOp::Reshape { new_shape },
            vec![row],
            at(&flat_dims),
            span,
        );
    }
    let column_order: Vec<u8> = plan
        .batch
        .iter()
        .chain(&[plan.k, plan.n])
        .copied()
        .collect();
    let (column, column_dims) = arrange(dag, 1, column, column_ty, &column_order);
    let mut product_dims = flat_dims.clone();
    product_dims.push(dim(1, plan.n));
    let n_size = match static_extent(&dim(1, plan.n)) {
        Some(n) => (RtDim::Lit(n), vec![row]),
        None => (
            RtDim::InputAxis {
                tensor: 1,
                axis: rt_axis(column_dims.len() - 1),
            },
            vec![row, column],
        ),
    };
    let m_size = match static_extent(&flat) {
        Some(n) => (RtDim::Lit(n), vec![column]),
        None => (
            RtDim::InputAxis {
                tensor: 1,
                axis: rt_axis(batch),
            },
            vec![column, row],
        ),
    };
    let expand = |dag: &mut Dag, axis: usize, (size, inputs): (RtDim, Vec<NodeId>)| {
        add_synth(
            owner,
            dag,
            RiscOp::Expand { axis, size },
            inputs,
            at(&product_dims),
            span,
        )
    };
    let views = vec![expand(dag, batch + 2, n_size), expand(dag, batch, m_size)];
    let product = add_synth(owner, dag, RiscOp::Mul, views, at(&product_dims), span);
    let mut dims = product_dims.clone();
    dims.remove(batch + 1);
    let sum = RiscOp::Sum {
        axis: batch + 1,
        accumulator,
    };
    let mut node = add_synth(owner, dag, sum, vec![product], at(&dims), span);
    if !plan.rows.is_empty() {
        let mut out_dims = row_dims[..row_dims.len() - 1].to_vec();
        out_dims.push(dim(1, plan.n));
        let new_shape = out_dims.iter().map(RtDim::from_dim_info).collect();
        node = add_synth(
            owner,
            dag,
            RiscOp::Reshape { new_shape },
            vec![node],
            at(&out_dims),
            span,
        );
    }
    Ok(node)
}

fn rt_axis(axis: usize) -> RtAxis {
    RtAxis::Lit(i32::try_from(axis).expect("tensor rank fits the i32 axis carrier"))
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
        einsum_contraction(equation, &ty(a, p), &ty(b, p), None)
    }

    /// [05-OP-33]'s one-reduction-label contractions whose output is
    /// `[batch.., rows.., m, n]`, with the row operand's labels flattening
    /// to one axis, at an operand dtype that is its own accumulator.
    #[test]
    fn matmul_shaped_float_contractions_have_the_contraction_form() {
        let f32 = Prim::F32;
        for (equation, a, b) in [
            ("mk,kn->mn", lit(&[3, 37]), lit(&[37, 5])),
            ("amk,kn->amn", lit(&[2, 3, 37]), lit(&[37, 5])),
            ("mk,nk->mn", lit(&[3, 37]), lit(&[5, 37])),
            ("km,kn->mn", lit(&[37, 3]), lit(&[37, 5])),
            ("bmk,bkn->bmn", lit(&[2, 3, 17]), lit(&[2, 17, 4])),
            ("bamk,bkn->bamn", lit(&[2, 5, 3, 17]), lit(&[2, 17, 4])),
            ("mak,kn->amn", lit(&[3, 2, 7]), lit(&[7, 4])),
            ("ij,jk->ki", lit(&[3, 7]), lit(&[7, 5])),
            (
                "mk,kn->mn",
                vec![runtime("m"), DimInfo::Lit(8)],
                lit(&[8, 5]),
            ),
            (
                "amk,kn->amn",
                vec![
                    DimInfo::Lit(1),
                    runtime("m"),
                    DimInfo::Named("k".into(), Some(8)),
                ],
                vec![DimInfo::Lit(8), runtime("n")],
            ),
        ] {
            assert!(plan(equation, &a, &b, f32).is_some(), "`{equation}`");
        }
        assert!(plan("mk,kn->mn", &lit(&[3, 7]), &lit(&[7, 5]), Prim::F64).is_some());
        let explicit = einsum_contraction(
            "mk,kn->mn",
            &ty(&lit(&[3, 7]), f32),
            &ty(&lit(&[7, 5]), f32),
            Some(f32),
        );
        assert!(
            explicit.is_some(),
            "an explicit accumulator equal to the operand dtype"
        );
        let swapped = plan("nk,mk->mn", &lit(&[5, 7]), &lit(&[3, 7]), f32).unwrap();
        assert!(swapped.swapped);
        assert!(
            !plan("mk,kn->mn", &lit(&[3, 7]), &lit(&[7, 5]), f32)
                .unwrap()
                .swapped
        );
    }

    /// Everything else keeps the runtime kernel: integers, an operand dtype
    /// that is not its accumulator, diagonals, no or several reduction labels,
    /// an output that does not end in a row and a column, a shared label
    /// without one static extent, a lead label the contraction could only
    /// reach by copying an operand, and a planned shape i64 cannot hold.
    #[test]
    fn every_other_einsum_has_no_contraction_form() {
        let f32 = Prim::F32;
        for p in [Prim::Int32, Prim::Int64, Prim::Int8, Prim::F16, Prim::Bf16] {
            assert!(
                plan("mk,kn->mn", &lit(&[3, 7]), &lit(&[7, 5]), p).is_none(),
                "{p:?}"
            );
        }
        let widened = einsum_contraction(
            "mk,kn->mn",
            &ty(&lit(&[3, 7]), f32),
            &ty(&lit(&[7, 5]), f32),
            Some(Prim::F64),
        );
        assert!(widened.is_none(), "an f64 accumulator over f32 operands");
        assert!(
            einsum_contraction(
                "mk,kn->mn",
                &ty(&lit(&[3, 7]), f32),
                &ty(&lit(&[7, 5]), Prim::F64),
                None
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
            // A lead label only the column operand carries.
            ("mk,bkn->bmn", lit(&[3, 9]), lit(&[2, 9, 4])),
            // A row-only lead label before a batch label.
            ("abmk,bkn->abmn", lit(&[2, 3, 4, 5]), lit(&[3, 5, 6])),
            // Rows that flatten to no single extent.
            (
                "amk,kn->amn",
                vec![runtime("a"), runtime("m"), DimInfo::Lit(8)],
                lit(&[8, 5]),
            ),
            (
                "amk,kn->amn",
                vec![DimInfo::Lit(2), runtime("m"), DimInfo::Lit(8)],
                lit(&[8, 5]),
            ),
            (
                "amk,kn->amn",
                lit(&[usize::MAX / 2 + 1, 2, 8]),
                lit(&[8, 5]),
            ),
            // An empty batch whose expansion strides exceed i64.
            (
                "bmk,bkn->bmn",
                lit(&[0, 4_194_304, 4_194_304]),
                lit(&[0, 4_194_304, 4_194_304]),
            ),
        ] {
            assert!(plan(equation, &a, &b, f32).is_none(), "`{equation}`");
        }
    }

    fn lowered(
        equation: &str,
        a: &[DimInfo],
        b: &[DimInfo],
        p: Prim,
        plan: &EinsumContraction,
    ) -> (Dag, NodeId) {
        let mut dag = Dag::new();
        let owner = Owner::from(dag.declare("test"));
        let a_ty = ty(a, p);
        let b_ty = ty(b, p);
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
        let root =
            lower_einsum_contraction(owner, &mut dag, plan, [(x, &a_ty), (y, &b_ty)], None, None)
                .expect("lowered");
        dag.set_roots(vec![root]);
        assert!(
            verify::verify(&dag).is_empty(),
            "`{equation}`: {:?}",
            verify::verify(&dag)
        );
        (dag, root)
    }

    /// The graph is spec/05 §4.1's product sum, which the lanes' fused
    /// product-sum kernels recognize in its blocked row-and-column layout.
    #[test]
    fn the_contraction_lowers_to_the_blocked_product_sum() {
        for (equation, a, b, p, out) in [
            (
                "amk,kn->amn",
                lit(&[2, 3, 37]),
                lit(&[37, 5]),
                Prim::F32,
                lit(&[2, 3, 5]),
            ),
            (
                "mk,nk->mn",
                vec![runtime("m"), DimInfo::Lit(9)],
                vec![runtime("n"), DimInfo::Lit(9)],
                Prim::F64,
                vec![runtime("m"), runtime("n")],
            ),
            (
                "nk,mk->mn",
                lit(&[5, 7]),
                lit(&[3, 7]),
                Prim::F32,
                lit(&[3, 5]),
            ),
            (
                "bamk,bkn->bamn",
                lit(&[2, 5, 3, 9]),
                lit(&[2, 9, 4]),
                Prim::F64,
                lit(&[2, 5, 3, 4]),
            ),
            (
                "amk,kn->amn",
                vec![DimInfo::Lit(1), runtime("s"), DimInfo::Lit(8)],
                vec![DimInfo::Lit(8), runtime("e")],
                Prim::F32,
                vec![DimInfo::Lit(1), runtime("s"), runtime("e")],
            ),
        ] {
            let contraction = plan(equation, &a, &b, p).expect("contraction form");
            let (dag, root) = lowered(equation, &a, &b, p, &contraction);
            let inlined = reduction_inlined_products(&dag);
            assert_eq!(inlined.len(), 1, "`{equation}` product sum");
            let product = dag.get(inlined[0].product).unwrap();
            let rank = product.output_type.dims.len();
            assert_eq!(product.output_type.precision, p);
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
            assert_eq!(
                root.output_type,
                TensorType {
                    dims: out,
                    precision: p
                },
                "`{equation}` result type"
            );
        }
    }

    /// Every two-operand equation over up to four labels, each operand of
    /// rank one to three.
    fn equations() -> Vec<String> {
        let letters = *b"abcd";
        let mut sequences = vec![Vec::new()];
        for _ in 0..3 {
            let mut longer = Vec::new();
            for sequence in &sequences {
                for &letter in &letters {
                    if !sequence.contains(&letter) {
                        let mut next: Vec<u8> = sequence.clone();
                        next.push(letter);
                        longer.push(next);
                    }
                }
            }
            sequences.extend(
                longer
                    .iter()
                    .filter(|s| !sequences.contains(s))
                    .cloned()
                    .collect::<Vec<_>>(),
            );
        }
        sequences.sort();
        sequences.dedup();
        let operands: Vec<&Vec<u8>> = sequences.iter().filter(|s| !s.is_empty()).collect();
        let mut out = Vec::new();
        for a in &operands {
            for b in &operands {
                let union: Vec<u8> = letters
                    .iter()
                    .copied()
                    .filter(|l| a.contains(l) || b.contains(l))
                    .collect();
                for output in sequences
                    .iter()
                    .filter(|s| s.iter().all(|l| union.contains(l)))
                {
                    out.push(format!(
                        "{},{}->{}",
                        String::from_utf8_lossy(a),
                        String::from_utf8_lossy(b),
                        String::from_utf8_lossy(output)
                    ));
                }
            }
        }
        out
    }

    const LARGE: usize = 1 << 21;

    /// The class oracle over the classifier's own acceptances: for every
    /// generated equation, operand dtype and accumulator, and per-label
    /// extents drawn from zero, one, three, a runtime extent and a large one,
    /// an accepted contraction lowers to one product sum whose planned shapes
    /// fit i64 and which stores no node larger in bytes than an operand or
    /// the result. A runtime extent is evaluated at 0 and at 7.
    #[test]
    fn every_accepted_contraction_stores_no_more_than_its_operands_or_result() {
        let precisions = [
            (Prim::F16, None),
            (Prim::Bf16, None),
            (Prim::F32, None),
            (Prim::F64, None),
            (Prim::F32, Some(Prim::F64)),
        ];
        let extents = [
            DimInfo::Lit(0),
            DimInfo::Lit(1),
            DimInfo::Lit(3),
            runtime("r"),
            DimInfo::Lit(LARGE),
        ];
        let mut state = 0x3777_u64;
        let mut accepted = 0usize;
        for equation in equations() {
            let (inputs, _) = equation.split_once("->").unwrap();
            let (la, lb) = inputs.split_once(',').unwrap();
            for _ in 0..4 {
                let mut extent = std::collections::BTreeMap::new();
                for label in la.bytes().chain(lb.bytes()) {
                    extent.entry(label).or_insert_with(|| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1);
                        let mut dim = extents[(state >> 33) as usize % extents.len()].clone();
                        if let DimInfo::Named(_, None) = dim {
                            dim = runtime(&format!("r{}", char::from(label)));
                        }
                        dim
                    });
                }
                let a: Vec<DimInfo> = la.bytes().map(|l| extent[&l].clone()).collect();
                let b: Vec<DimInfo> = lb.bytes().map(|l| extent[&l].clone()).collect();
                for &(p, accumulator) in &precisions {
                    let Some(contraction) =
                        einsum_contraction(&equation, &ty(&a, p), &ty(&b, p), accumulator)
                    else {
                        continue;
                    };
                    accepted += 1;
                    let (dag, root) = lowered(&equation, &a, &b, p, &contraction);
                    let inlined = reduction_inlined_products(&dag);
                    assert_eq!(inlined.len(), 1, "`{equation}` {a:?} {b:?} {p:?}");
                    let skipped = inlined[0].skipped();
                    let product = &dag.get(inlined[0].product).unwrap().output_type;
                    for binding in [0u128, 7] {
                        let count = |ty: &TensorType| -> u128 {
                            ty.dims
                                .iter()
                                .map(|dim| static_extent(dim).map_or(binding, |n| n as u128))
                                .product()
                        };
                        let bytes = |ty: &TensorType| -> u128 {
                            let width = ty.precision.runtime_dtype().unwrap().byte_width() as u128;
                            count(ty) * width
                        };
                        let bound = bytes(&ty(&a, p))
                            .max(bytes(&ty(&b, p)))
                            .max(bytes(&dag.get(root).unwrap().output_type));
                        for node in dag.nodes() {
                            if skipped.contains(&node.id) {
                                continue;
                            }
                            assert!(
                                bytes(&node.output_type) <= bound,
                                "`{equation}` {a:?} {b:?} {p:?}/{accumulator:?} stores {:?} of {} bytes, over {bound}",
                                node.op,
                                bytes(&node.output_type)
                            );
                        }
                        // Every stride and the byte size of the planned
                        // expansion fit i64.
                        let mut suffix = 1u128;
                        for dim in product.dims.iter().rev() {
                            suffix *= static_extent(dim).map_or(binding.max(1), |n| n as u128);
                            assert!(
                                suffix <= i64::MAX as u128,
                                "`{equation}` {a:?} {b:?} expansion stride"
                            );
                        }
                        assert!(
                            bytes(product) <= i64::MAX as u128,
                            "`{equation}` {a:?} {b:?} expansion bytes"
                        );
                    }
                }
            }
        }
        assert!(
            accepted > 1000,
            "the oracle reached {accepted} accepted cases"
        );
    }
}
