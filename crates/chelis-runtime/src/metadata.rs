//! Runtime-specific plans over the shared checked descriptor metadata owner.

#[allow(unused_imports)]
pub(crate) use chelis_abi::metadata::{
    AllocationBytes, ByteCount, ElementCount, MetadataError, ShapeMetadata,
};
use chelis_vocab::RuntimeDType;

pub(crate) struct AxisDecomposition {
    outer: ElementCount,
    extent: ElementCount,
    inner: ElementCount,
}

/// A contraction's iteration domain need not be a materialized tensor: a
/// zero product has no reachable index and owes no unused storage stride.
pub(crate) struct IterationSpace {
    shape: Box<[i64]>,
    elements: ElementCount,
}

/// One checked row-major axis projection; no tensor data or per-index scratch.
struct AxisProjection {
    divisor: i64,
    modulus: i64,
    offset: i64,
    step: i64,
    stride: i64,
}

#[derive(Clone, Copy)]
pub(crate) enum MovementOp {
    Pad,
    Shrink,
    Stride,
}

pub(crate) struct MovementMetadata {
    input: ShapeMetadata,
    result: ShapeMetadata,
    projection: Box<[AxisProjection]>,
    source_domain: bool,
}
impl MovementMetadata {
    fn build(
        input: &ShapeMetadata,
        result: ShapeMetadata,
        source_domain: bool,
        axes: impl IntoIterator<Item = (usize, usize, i64, i64)>,
    ) -> Result<Self, MetadataError> {
        result.bytes().allocation()?;
        let (domain, target) = if source_domain {
            (input, &result)
        } else {
            (&result, input)
        };
        let count = ElementCount::scratch_entries(domain.shape().len(), 0)?;
        let mut projection = Vec::with_capacity(count.scratch_len::<AxisProjection>()?);
        for (domain_axis, target_axis, offset, step) in axes {
            if projection.len() >= domain.shape().len()
                || domain_axis >= domain.shape().len()
                || target_axis >= target.shape().len()
            {
                return Err(MetadataError::Domain(
                    "movement projection rank mismatch".into(),
                ));
            }
            // The checked capacity cannot grow through this construction path.
            projection.push(AxisProjection {
                divisor: domain.strides()[domain_axis],
                modulus: domain.shape()[domain_axis],
                offset,
                step,
                stride: target.strides()[target_axis],
            });
        }
        Ok(Self {
            input: input.clone(),
            result,
            projection: projection.into(),
            source_domain,
        })
    }
    pub(crate) fn permuted(input: &ShapeMetadata, axes: &[i64]) -> Result<Self, MetadataError> {
        if axes.len() != input.shape().len() {
            return Err(MetadataError::Domain("permutation rank mismatch".into()));
        }
        let count = ElementCount::scratch_entries(axes.len(), 0)?;
        let mut shape = Vec::with_capacity(count.scratch_len::<i64>()?);
        let mut normalized = Vec::with_capacity(count.scratch_len::<usize>()?);
        for &axis in axes {
            let axis = input.normalize_axis(axis)?;
            normalized.push(axis);
            shape.push(input.shape()[axis]);
        }
        let result = ShapeMetadata::contiguous(&shape, input.dtype())?;
        input.require_permutation(&result, axes)?;
        Self::build(
            input,
            result,
            false,
            normalized
                .into_iter()
                .enumerate()
                .map(|(out, source)| (out, source, 0, 1)),
        )
    }
    pub(crate) fn expanded(
        input: &ShapeMetadata,
        axis: i64,
        size: i64,
        insert: bool,
    ) -> Result<Self, MetadataError> {
        let rank = input
            .shape()
            .len()
            .checked_add(usize::from(insert))
            .ok_or(MetadataError::Overflow("expanded rank exceeds usize"))?;
        let rank_i32 = ShapeMetadata::checked_rank(rank)?;
        let axis = if axis < 0 {
            axis + i64::from(rank_i32)
        } else {
            axis
        };
        if axis < 0 || axis >= i64::from(rank_i32) || size < 0 {
            return Err(MetadataError::Domain(
                "expansion axis or extent outside domain".into(),
            ));
        }
        let axis = usize::try_from(axis)
            .map_err(|_| MetadataError::Overflow("expansion axis exceeds usize"))?;
        let count = ElementCount::scratch_entries(rank, 0)?;
        let mut shape = Vec::with_capacity(count.scratch_len::<i64>()?);
        shape.extend_from_slice(input.shape());
        if insert {
            shape.insert(axis, size);
        } else {
            shape[axis] = size;
        }
        let result = ShapeMetadata::contiguous(&shape, input.dtype())?;
        input.require_expansion(&result, axis as i32)?;
        let axes = (0..input.shape().len()).filter_map(|source| {
            if !insert && source == axis {
                None
            } else {
                Some((source + usize::from(insert && source >= axis), source, 0, 1))
            }
        });
        Self::build(input, result, false, axes)
    }
    pub(crate) fn affine(
        input: &ShapeMetadata,
        first: &[i64],
        second: &[i64],
        op: MovementOp,
    ) -> Result<Self, MetadataError> {
        let result = match op {
            MovementOp::Pad => input.padded(first, second)?,
            MovementOp::Shrink => input.shrunk(first, second)?,
            MovementOp::Stride => input.strided(first)?,
        };
        let axes = (0..input.shape().len()).map(|axis| match op {
            MovementOp::Pad | MovementOp::Shrink => (axis, axis, first[axis], 1),
            MovementOp::Stride => (axis, axis, 0, first[axis]),
        });
        Self::build(input, result, matches!(op, MovementOp::Pad), axes)
    }
    pub(crate) fn input(&self) -> &ShapeMetadata {
        &self.input
    }
    pub(crate) fn result(&self) -> &ShapeMetadata {
        &self.result
    }
    pub(crate) fn count(&self) -> ElementCount {
        if self.source_domain {
            self.input.elements()
        } else {
            self.result.elements()
        }
    }
    pub(crate) fn index(&self, linear: i64) -> Result<i64, MetadataError> {
        let (domain, target) = if self.source_domain {
            (&self.input, &self.result)
        } else {
            (&self.result, &self.input)
        };
        domain.require_index(linear)?;
        let mut flat = 0_i64;
        for axis in &self.projection {
            let coordinate = linear
                .checked_div(axis.divisor)
                .and_then(|n| n.checked_rem(axis.modulus))
                .and_then(|n| n.checked_mul(axis.step))
                .and_then(|n| n.checked_add(axis.offset))
                .and_then(|n| n.checked_mul(axis.stride))
                .and_then(|n| flat.checked_add(n))
                .ok_or(MetadataError::Overflow("movement projection exceeds int64"))?;
            flat = coordinate;
        }
        target.require_index(flat)?;
        Ok(flat)
    }
}

/// Valid-padding window geometry independent of source storage and arithmetic.
pub(crate) struct WindowMetadata {
    input: ShapeMetadata,
    result: ShapeMetadata,
    window: Box<[i64]>,
    steps: Box<[i64]>,
    leading: usize,
    count: ElementCount,
}

impl WindowMetadata {
    pub(crate) fn new(
        input: &ShapeMetadata,
        window: &[i64],
        steps: &[i64],
    ) -> Result<Self, MetadataError> {
        if window.is_empty() || window.len() != steps.len() || window.len() > input.shape().len() {
            return Err(MetadataError::Domain(
                "window and stride arity outside input rank".into(),
            ));
        }
        ElementCount::scratch_entries(window.len(), 0)?.scratch_len::<i64>()?;
        let leading = input.shape().len() - window.len();
        let mut shape = input.shape().to_vec();
        for (i, (&w, &step)) in window.iter().zip(steps).enumerate() {
            let axis = leading + i;
            if w <= 0 || step <= 0 || w > input.shape()[axis] {
                return Err(MetadataError::Domain(
                    "window or stride outside valid-padding domain".into(),
                ));
            }
            shape[axis] = input.shape()[axis]
                .checked_sub(w)
                .and_then(|n| n.checked_div(step))
                .and_then(|n| n.checked_add(1))
                .ok_or(MetadataError::Overflow("window extent exceeds int64"))?;
        }
        let result = ShapeMetadata::contiguous(&shape, input.dtype())?;
        result.bytes().allocation()?;
        let count = ElementCount::from_extents(if result.elements().get() == 0 {
            &[0]
        } else {
            window
        })?;
        Ok(Self {
            input: input.clone(),
            result,
            window: window.into(),
            steps: steps.into(),
            leading,
            count,
        })
    }
    pub(crate) fn input(&self) -> &ShapeMetadata {
        &self.input
    }
    pub(crate) fn result(&self) -> &ShapeMetadata {
        &self.result
    }
    pub(crate) fn count(&self) -> ElementCount {
        self.count
    }
    pub(crate) fn index(&self, group: i64, leaf: i64) -> Result<i64, MetadataError> {
        self.result.require_index(group)?;
        if leaf < 0 || leaf >= self.count.get() {
            return Err(MetadataError::Domain(
                "window leaf outside iteration domain".into(),
            ));
        }
        let mut group_remaining = group;
        let mut leaf_remaining = leaf;
        let mut source_index = 0_i64;
        for axis in (0..self.input.shape().len()).rev() {
            let coordinate = group_remaining % self.result.shape()[axis];
            group_remaining /= self.result.shape()[axis];
            let coordinate = if axis < self.leading {
                coordinate
            } else {
                let window_axis = axis - self.leading;
                let offset = leaf_remaining % self.window[window_axis];
                leaf_remaining /= self.window[window_axis];
                coordinate
                    .checked_mul(self.steps[window_axis])
                    .and_then(|n| n.checked_add(offset))
                    .ok_or(MetadataError::Overflow("window coordinate exceeds int64"))?
            };
            if coordinate >= self.input.shape()[axis] {
                return Err(MetadataError::Domain(
                    "window coordinate outside source".into(),
                ));
            }
            source_index = coordinate
                .checked_mul(self.input.strides()[axis])
                .and_then(|n| source_index.checked_add(n))
                .ok_or(MetadataError::Overflow("window index exceeds int64"))?;
        }
        self.input.require_index(source_index)?;
        Ok(source_index)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum MatmulPart {
    Left,
    Right,
    Result,
}
#[derive(Clone, Copy)]
pub(crate) enum MatmulDimension {
    Rows,
    Columns,
    Reduction,
}

/// One checked matrix domain after explicit batch alignment. No payload is held.
pub(crate) struct MatmulMetadata {
    result: ShapeMetadata,
    dimensions: [i64; 3],
    matrices: [ElementCount; 3],
    totals: [ElementCount; 3],
    batches: ElementCount,
}

impl MatmulMetadata {
    pub(crate) fn new(
        left: &ShapeMetadata,
        right: &ShapeMetadata,
        dtype: RuntimeDType,
    ) -> Result<Self, MetadataError> {
        let rank = left.shape().len();
        if rank < 2
            || right.shape().len() != rank
            || left.dtype() != right.dtype()
            || left.shape()[..rank - 2] != right.shape()[..rank - 2]
            || left.shape()[rank - 1] != right.shape()[rank - 2]
        {
            return Err(MetadataError::Domain(
                "matmul operand shape or dtype mismatch".into(),
            ));
        }
        let m = left.shape()[rank - 2];
        let n = right.shape()[rank - 1];
        let k = left.shape()[rank - 1];
        let mut shape = left.shape().to_vec();
        shape[rank - 1] = n;
        let result = ShapeMetadata::contiguous(&shape, dtype)?;
        result.bytes().allocation()?;
        let empty = result.elements().get() == 0;
        let batches = ElementCount::from_extents(if empty { &[0] } else { &shape[..rank - 2] })?;
        let matrices = if empty {
            [ElementCount::from_extents(&[0])?; 3]
        } else {
            [
                ElementCount::from_extents(&[m, k])?,
                ElementCount::from_extents(&[k, n])?,
                ElementCount::from_extents(&[m, n])?,
            ]
        };
        let totals = [left.elements(), right.elements(), result.elements()];
        Ok(Self {
            result,
            dimensions: [m, n, k],
            matrices,
            totals,
            batches,
        })
    }
    pub(crate) fn result(&self) -> &ShapeMetadata {
        &self.result
    }
    pub(crate) fn dimension(&self, dimension: MatmulDimension) -> i64 {
        self.dimensions[match dimension {
            MatmulDimension::Rows => 0,
            MatmulDimension::Columns => 1,
            MatmulDimension::Reduction => 2,
        }]
    }
    pub(crate) fn batches(&self) -> ElementCount {
        self.batches
    }
    fn part_index(part: MatmulPart) -> usize {
        match part {
            MatmulPart::Left => 0,
            MatmulPart::Right => 1,
            MatmulPart::Result => 2,
        }
    }
    pub(crate) fn matrix_count(&self, part: MatmulPart) -> ElementCount {
        self.matrices[Self::part_index(part)]
    }
    pub(crate) fn index(
        &self,
        part: MatmulPart,
        batch: i64,
        element: i64,
    ) -> Result<i64, MetadataError> {
        let matrix = self.matrix_count(part).get();
        if batch < 0 || batch >= self.batches.get() || element < 0 || element >= matrix {
            return Err(MetadataError::Domain(
                "matmul index outside batch or matrix".into(),
            ));
        }
        let index = batch
            .checked_mul(matrix)
            .and_then(|n| n.checked_add(element))
            .ok_or(MetadataError::Overflow("matmul index exceeds int64"))?;
        if index >= self.totals[Self::part_index(part)].get() {
            return Err(MetadataError::Domain("matmul index outside operand".into()));
        }
        Ok(index)
    }
    pub(crate) fn check_vendor(&self, limit: i64) -> Result<(), MetadataError> {
        if limit <= 0 {
            return Err(MetadataError::Domain(
                "matmul vendor dimension limit must be positive".into(),
            ));
        }
        if self.batches.get() != 0
            && self.dimension(MatmulDimension::Reduction) != 0
            && self.dimensions.iter().any(|&extent| extent > limit)
        {
            return Err(MetadataError::Overflow(
                "matmul dimension exceeds vendor argument domain",
            ));
        }
        Ok(())
    }
}

/// The checked row-major domain shared by gather and scatter consumers.
pub(crate) struct SparseMetadata {
    base: ShapeMetadata,
    indices: ShapeMetadata,
    domain: ShapeMetadata,
    axis: usize,
    elementwise: bool,
    inner: ElementCount,
    source_axis: AxisDecomposition,
}

impl SparseMetadata {
    pub(crate) fn new(
        base: &ShapeMetadata,
        indices: &ShapeMetadata,
        axis: i64,
        elementwise: bool,
    ) -> Result<Self, MetadataError> {
        let axis = base.normalize_axis(axis)?;
        let shape = if elementwise {
            if base.shape().len() != indices.shape().len()
                || base
                    .shape()
                    .iter()
                    .zip(indices.shape().iter())
                    .enumerate()
                    .any(|(d, (base, index))| d != axis && index > base)
            {
                return Err(MetadataError::Domain(
                    "element-wise scatter shape mismatch".into(),
                ));
            }
            indices.shape().to_vec()
        } else {
            let rank = base
                .shape()
                .len()
                .checked_sub(1)
                .and_then(|rank| rank.checked_add(indices.shape().len()))
                .ok_or(MetadataError::Overflow("sparse domain rank overflow"))?;
            ShapeMetadata::checked_rank(rank)?;
            ElementCount::scratch_entries(rank, 0)?.scratch_len::<i64>()?;
            base.shape()[..axis]
                .iter()
                .chain(indices.shape().iter())
                .chain(base.shape()[axis + 1..].iter())
                .copied()
                .collect()
        };
        let domain = ShapeMetadata::contiguous(&shape, base.dtype())?;
        domain.bytes().allocation()?;
        let inner = if domain.elements().get() == 0 || elementwise {
            ElementCount::from_extents(&[0])?
        } else {
            ElementCount::from_extents(&base.shape()[axis + 1..])?
        };
        Ok(Self {
            base: base.clone(),
            indices: indices.clone(),
            domain,
            axis,
            elementwise,
            inner,
            source_axis: axis_decomposition(base, axis)?,
        })
    }

    pub(crate) fn base(&self) -> &ShapeMetadata {
        &self.base
    }
    pub(crate) fn domain(&self) -> &ShapeMetadata {
        &self.domain
    }
    pub(crate) fn index_slot(&self, linear: i64) -> Result<i64, MetadataError> {
        self.domain.require_index(linear)?;
        Ok(if self.elementwise {
            linear
        } else {
            linear / self.inner.get() % self.indices.elements().get()
        })
    }
    pub(crate) fn data_index(&self, linear: i64, selected: i64) -> Result<i64, MetadataError> {
        self.domain.require_index(linear)?;
        if selected < 0 || selected >= self.base.shape()[self.axis] {
            return Err(MetadataError::Domain(
                "sparse index outside base axis".into(),
            ));
        }
        if self.elementwise {
            let mut remaining = linear;
            let mut offset = 0_i64;
            for (axis, &extent) in self.indices.shape().iter().enumerate().rev() {
                let coordinate = if axis == self.axis {
                    selected
                } else {
                    remaining % extent
                };
                remaining /= extent;
                offset = coordinate
                    .checked_mul(self.base.strides()[axis])
                    .and_then(|n| offset.checked_add(n))
                    .ok_or(MetadataError::Overflow("sparse offset exceeds int64"))?;
            }
            self.base.require_index(offset)?;
            Ok(offset)
        } else {
            let outer = linear / self.inner.get() / self.indices.elements().get();
            let inner = linear % self.inner.get();
            let offset = self.source_axis.linear_index(
                usize::try_from(outer)
                    .map_err(|_| MetadataError::Overflow("sparse outer exceeds target"))?,
                usize::try_from(selected)
                    .map_err(|_| MetadataError::Overflow("sparse index exceeds target"))?,
                usize::try_from(inner)
                    .map_err(|_| MetadataError::Overflow("sparse inner exceeds target"))?,
            )?;
            i64::try_from(offset)
                .map_err(|_| MetadataError::Overflow("sparse offset exceeds int64"))
        }
    }
}

/// Checked reduction grouping, independent of input payload storage and lifetime.
pub(crate) struct ReductionMetadata {
    input: IterationSpace,
    selected: Box<[bool]>,
    result: ShapeMetadata,
    leaves: ElementCount,
}

impl ReductionMetadata {
    pub(crate) fn new(
        shape: &[i64],
        axes: &[i64],
        dtype: RuntimeDType,
    ) -> Result<Self, MetadataError> {
        let input = IterationSpace::new(shape)?;
        let rank = ElementCount::scratch_entries(shape.len(), 0)?;
        rank.scratch_len::<bool>()?;
        rank.scratch_len::<i64>()?;
        if axes.is_empty() {
            return Err(MetadataError::Domain(
                "reduction axes must be nonempty and strictly descending".into(),
            ));
        }
        let mut selected = vec![false; shape.len()];
        let mut previous = shape.len();
        for &axis in axes {
            let axis = if axis < 0 {
                axis + shape.len() as i64
            } else {
                axis
            };
            let axis = usize::try_from(axis)
                .map_err(|_| MetadataError::Domain("reduction axis outside rank".into()))?;
            if axis >= previous {
                return Err(MetadataError::Domain(
                    "normalized reduction axes must be strictly descending".into(),
                ));
            }
            previous = axis;
            let slot = selected
                .get_mut(axis)
                .ok_or(MetadataError::Domain("reduction axis outside rank".into()))?;
            *slot = true;
        }
        let output: Vec<_> = shape
            .iter()
            .zip(&selected)
            .filter_map(|(&n, &selected)| (!selected).then_some(n))
            .collect();
        let result = ShapeMetadata::contiguous(&output, dtype)?;
        result.bytes().allocation()?;
        // No output group can observe the selected domain when the result is empty.
        let leaves = if result.elements().get() == 0 {
            ElementCount::from_extents(&[0])?
        } else {
            let extents: Vec<_> = shape
                .iter()
                .zip(&selected)
                .filter_map(|(&n, &selected)| selected.then_some(n))
                .collect();
            ElementCount::from_extents(&extents)?
        };
        Ok(Self {
            input,
            selected: selected.into(),
            result,
            leaves,
        })
    }
    pub(crate) fn result(&self) -> &ShapeMetadata {
        &self.result
    }
    pub(crate) fn leaves(&self) -> ElementCount {
        self.leaves
    }
    pub(crate) fn extent(&self, axis: i64) -> Result<i64, MetadataError> {
        Ok(self.result.shape()[self.result.normalize_axis(axis)?])
    }
    pub(crate) fn index(&self, mut outer: i64, mut leaf: i64) -> Result<i64, MetadataError> {
        if outer < 0
            || outer >= self.result.elements().get()
            || leaf < 0
            || leaf >= self.leaves.get()
        {
            return Err(MetadataError::Domain(
                "reduction index outside group or leaf domain".into(),
            ));
        }
        let mut index = 0_i64;
        let mut stride = 1_i64;
        for (&extent, &selected) in self.input.shape.iter().zip(self.selected.iter()).rev() {
            let remaining = if selected { &mut leaf } else { &mut outer };
            let coordinate = *remaining % extent;
            *remaining /= extent;
            index = coordinate
                .checked_mul(stride)
                .and_then(|n| index.checked_add(n))
                .ok_or(MetadataError::Overflow("reduction offset exceeds int64"))?;
            stride = stride
                .checked_mul(extent)
                .ok_or(MetadataError::Overflow("reduction stride exceeds int64"))?;
        }
        Ok(index)
    }
}

impl IterationSpace {
    pub(crate) fn new(shape: &[i64]) -> Result<Self, MetadataError> {
        ShapeMetadata::checked_rank(shape.len())?;
        let elements = ElementCount::from_extents(shape)?;
        Ok(Self {
            shape: shape.into(),
            elements,
        })
    }
    pub(crate) fn elements(&self) -> ElementCount {
        self.elements
    }
    pub(crate) fn elementwise_index_step(
        &self,
        input: &ShapeMetadata,
    ) -> Result<i64, MetadataError> {
        if input.rank() == 0 {
            Ok(0)
        } else if input.shape() == self.shape.as_ref() {
            Ok(1)
        } else {
            Err(MetadataError::Domain(
                "elementwise input shape does not match iteration domain".into(),
            ))
        }
    }
    pub(crate) fn unravel(&self, linear: i64, out: &mut [i64]) -> Result<(), MetadataError> {
        unravel(&self.shape, self.elements, linear, out)
    }
}

pub(crate) fn axis_decomposition(
    metadata: &ShapeMetadata,
    axis: usize,
) -> Result<AxisDecomposition, MetadataError> {
    let &extent = metadata
        .shape()
        .get(axis)
        .ok_or(MetadataError::Domain("axis outside tensor rank".into()))?;
    if metadata.elements().get() == 0 {
        return Ok(AxisDecomposition {
            outer: ElementCount::from_extents(&[0])?,
            extent: ElementCount::from_extents(&[extent])?,
            inner: ElementCount::from_extents(&[0])?,
        });
    }
    Ok(AxisDecomposition {
        outer: ElementCount::from_extents(&metadata.shape()[..axis])?,
        extent: ElementCount::from_extents(&[extent])?,
        inner: ElementCount::from_extents(&metadata.shape()[axis + 1..])?,
    })
}

impl AxisDecomposition {
    pub(crate) fn outer(&self) -> ElementCount {
        self.outer
    }
    pub(crate) fn extent(&self) -> ElementCount {
        self.extent
    }
    pub(crate) fn inner(&self) -> ElementCount {
        self.inner
    }
    pub(crate) fn linear_index(
        &self,
        outer: usize,
        axis: usize,
        inner: usize,
    ) -> Result<usize, MetadataError> {
        let outer_count = self.outer.as_usize()?;
        let extent = self.extent.as_usize()?;
        let inner_count = self.inner.as_usize()?;
        if outer >= outer_count || axis >= extent || inner >= inner_count {
            return Err(MetadataError::Domain("axis iteration outside shape".into()));
        }
        outer
            .checked_mul(extent)
            .and_then(|n| n.checked_add(axis))
            .and_then(|n| n.checked_mul(inner_count))
            .and_then(|n| n.checked_add(inner))
            .ok_or(MetadataError::Overflow("axis index offset exceeds usize"))
    }
    pub(crate) fn reduced_index(&self, outer: usize, inner: usize) -> Result<usize, MetadataError> {
        if outer >= self.outer.as_usize()? || inner >= self.inner.as_usize()? {
            return Err(MetadataError::Domain(
                "reduced iteration outside shape".into(),
            ));
        }
        outer
            .checked_mul(self.inner.as_usize()?)
            .and_then(|n| n.checked_add(inner))
            .ok_or(MetadataError::Overflow(
                "reduced index offset exceeds usize",
            ))
    }
}

fn unravel(
    shape: &[i64],
    elements: ElementCount,
    linear: i64,
    out: &mut [i64],
) -> Result<(), MetadataError> {
    if out.len() != shape.len() {
        return Err(MetadataError::Domain(
            "index rank does not match tensor rank".into(),
        ));
    }
    unravel_into(shape, elements, linear, |axis, index| out[axis] = index)
}

fn unravel_into(
    shape: &[i64],
    elements: ElementCount,
    mut linear: i64,
    mut write: impl FnMut(usize, i64),
) -> Result<(), MetadataError> {
    if linear < 0 || linear >= elements.get() {
        return Err(MetadataError::Domain(
            "tensor index outside element count".into(),
        ));
    }
    // The checked nonempty range proves every divisor positive.
    for (axis, &extent) in shape.iter().enumerate().rev() {
        let coordinate = linear % extent;
        write(axis, coordinate);
        linear /= extent;
    }
    Ok(())
}
