//! Shared checked descriptor metadata authority: [04-SHAPE-1], [05-OP-31/33/44].

use chelis_vocab::RuntimeDType;

/// The `Domain` detail every lane prints when an `expand`/`insert` axis or
/// extent is outside its domain, a runtime negative size among them
/// (spec/04-type-system.md section 4.7.2). The C runtime reaches it through
/// [`MovementMetadata::expanded`]; the host interpreter, which builds no
/// metadata plan, names it directly so both lanes render one line.
pub const EXPANSION_DOMAIN: &str = "expansion axis or extent outside domain";

#[derive(Debug, Eq, PartialEq)]
pub enum MetadataError {
    Domain(std::borrow::Cow<'static, str>),
    Overflow(&'static str),
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain(message) => write!(f, "Domain: {message}"),
            Self::Overflow(message) => write!(f, "Overflow: {message}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ElementCount(i64);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteCount(i64);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationBytes(usize);

// Both layouts own exactly one domain. This is the only descriptor-domain
// constructor; contiguous runtime indexing never receives arbitrary strides.
#[derive(Clone, Debug)]
struct CheckedDomain {
    shape: Box<[i64]>,
    rank: i32,
    elements: ElementCount,
    bytes: ByteCount,
    dtype: RuntimeDType,
}

impl CheckedDomain {
    fn new(shape: &[i64], dtype: RuntimeDType) -> Result<Self, MetadataError> {
        let rank = ShapeMetadata::checked_rank(shape.len())?;
        let elements = ElementCount::from_extents(shape)?;
        let bytes = elements.bytes(dtype)?;
        Ok(Self {
            shape: shape.into(),
            rank,
            elements,
            bytes,
            dtype,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ShapeMetadata {
    domain: CheckedDomain,
    strides: Box<[i64]>,
}

/// A zero-offset forward view. It cannot enter contiguous runtime indexing.
/// Logical bytes describe the tensor's values; required_span describes storage.
#[derive(Clone, Debug)]
pub struct StridedMetadata {
    domain: CheckedDomain,
    strides: Box<[i64]>,
    required_span: ByteCount,
}

impl StridedMetadata {
    pub fn new(
        shape: &[i64],
        strides: &[i64],
        dtype: RuntimeDType,
        byte_capacity: ByteCount,
    ) -> Result<Self, MetadataError> {
        if shape.len() != strides.len() {
            return Err(MetadataError::Domain(
                "stride rank does not match shape".into(),
            ));
        }
        // Every stride is validated even when no element will be reachable.
        if strides.iter().any(|&stride| stride < 0) {
            return Err(MetadataError::Domain(
                "negative stride in forward view".into(),
            ));
        }
        let domain = CheckedDomain::new(shape, dtype)?;
        let span = if domain.elements.get() == 0 {
            ElementCount::from_extents(&[0])?
        } else {
            let mut largest = 0_i64;
            for (&extent, &stride) in shape.iter().zip(strides) {
                let part = (extent - 1)
                    .checked_mul(stride)
                    .ok_or(MetadataError::Overflow("view offset product exceeds i64"))?;
                largest = largest
                    .checked_add(part)
                    .ok_or(MetadataError::Overflow("view offset sum exceeds i64"))?;
            }
            let span = largest
                .checked_add(1)
                .ok_or(MetadataError::Overflow("view element span exceeds i64"))?;
            ElementCount::from_extents(&[span])?
        };
        let required_span = span.bytes(dtype)?;
        required_span.allocation()?;
        let metadata = Self {
            domain,
            strides: strides.into(),
            required_span,
        };
        metadata.require_capacity(byte_capacity)?;
        Ok(metadata)
    }
    pub fn rank(&self) -> i32 {
        self.domain.rank
    }
    pub fn shape(&self) -> &[i64] {
        &self.domain.shape
    }
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }
    pub fn elements(&self) -> ElementCount {
        self.domain.elements
    }
    pub fn bytes(&self) -> ByteCount {
        self.domain.bytes
    }
    pub fn dtype(&self) -> RuntimeDType {
        self.domain.dtype
    }
    pub fn required_span(&self) -> ByteCount {
        self.required_span
    }
    /// Map a logical row-major index through this immutable supplied-stride plan.
    pub fn byte_offset(&self, linear: i64) -> Result<AllocationBytes, MetadataError> {
        let mut offset = Some(0_i64);
        unravel_into(
            &self.domain.shape,
            self.domain.elements,
            linear,
            |axis, coordinate| {
                offset = offset.and_then(|offset| {
                    coordinate
                        .checked_mul(self.strides[axis])
                        .and_then(|part| offset.checked_add(part))
                });
            },
        )?;
        // Immutable checked extents and strides prove this location is within
        // required_span. Width and target projection use the existing authority.
        let offset = offset.ok_or(MetadataError::Overflow("view index offset exceeds i64"))?;
        ElementCount(offset).bytes(self.domain.dtype)?.allocation()
    }
    pub fn require_capacity(&self, capacity: ByteCount) -> Result<(), MetadataError> {
        capacity.allocation()?;
        require_byte_capacity(self.required_span, capacity)
    }
}

fn require_byte_capacity(required: ByteCount, capacity: ByteCount) -> Result<(), MetadataError> {
    if required.get() > capacity.get() {
        Err(MetadataError::Domain(
            format!(
                "byte capacity {} is smaller than required {}",
                capacity.get(),
                required.get()
            )
            .into(),
        ))
    } else {
        Ok(())
    }
}

pub struct AxisDecomposition {
    outer: ElementCount,
    extent: ElementCount,
    inner: ElementCount,
}

/// A contraction's iteration domain need not be a materialized tensor: a
/// zero product has no reachable index and owes no unused storage stride.
pub struct IterationSpace {
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
pub enum MovementOp {
    Pad,
    Shrink,
    Stride,
}

pub struct MovementMetadata {
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
    pub fn permuted(input: &ShapeMetadata, axes: &[i64]) -> Result<Self, MetadataError> {
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
    pub fn expanded(
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
            return Err(MetadataError::Domain(EXPANSION_DOMAIN.into()));
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
    pub fn affine(
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
    pub fn input(&self) -> &ShapeMetadata {
        &self.input
    }
    pub fn result(&self) -> &ShapeMetadata {
        &self.result
    }
    pub fn count(&self) -> ElementCount {
        if self.source_domain {
            self.input.elements()
        } else {
            self.result.elements()
        }
    }
    pub fn index(&self, linear: i64) -> Result<i64, MetadataError> {
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
                .ok_or(MetadataError::Overflow("movement projection exceeds i64"))?;
            flat = coordinate;
        }
        target.require_index(flat)?;
        Ok(flat)
    }
}

/// Valid-padding window geometry independent of source storage and arithmetic.
pub struct WindowMetadata {
    input: ShapeMetadata,
    result: ShapeMetadata,
    window: Box<[i64]>,
    steps: Box<[i64]>,
    leading: usize,
    count: ElementCount,
}

impl WindowMetadata {
    pub fn new(
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
                .ok_or(MetadataError::Overflow("window extent exceeds i64"))?;
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
    pub fn input(&self) -> &ShapeMetadata {
        &self.input
    }
    pub fn result(&self) -> &ShapeMetadata {
        &self.result
    }
    pub fn count(&self) -> ElementCount {
        self.count
    }
    pub fn index(&self, group: i64, leaf: i64) -> Result<i64, MetadataError> {
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
                    .ok_or(MetadataError::Overflow("window coordinate exceeds i64"))?
            };
            if coordinate >= self.input.shape()[axis] {
                return Err(MetadataError::Domain(
                    "window coordinate outside source".into(),
                ));
            }
            source_index = coordinate
                .checked_mul(self.input.strides()[axis])
                .and_then(|n| source_index.checked_add(n))
                .ok_or(MetadataError::Overflow("window index exceeds i64"))?;
        }
        self.input.require_index(source_index)?;
        Ok(source_index)
    }
}

#[derive(Clone, Copy)]
pub enum MatmulPart {
    Left,
    Right,
    Result,
}
#[derive(Clone, Copy)]
pub enum MatmulDimension {
    Rows,
    Columns,
    Reduction,
}

/// One checked matrix domain after explicit batch alignment. No payload is held.
pub struct MatmulMetadata {
    result: ShapeMetadata,
    dimensions: [i64; 3],
    matrices: [ElementCount; 3],
    totals: [ElementCount; 3],
    batches: ElementCount,
}

impl MatmulMetadata {
    pub fn new(
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
    pub fn result(&self) -> &ShapeMetadata {
        &self.result
    }
    pub fn dimension(&self, dimension: MatmulDimension) -> i64 {
        self.dimensions[match dimension {
            MatmulDimension::Rows => 0,
            MatmulDimension::Columns => 1,
            MatmulDimension::Reduction => 2,
        }]
    }
    pub fn batches(&self) -> ElementCount {
        self.batches
    }
    fn part_index(part: MatmulPart) -> usize {
        match part {
            MatmulPart::Left => 0,
            MatmulPart::Right => 1,
            MatmulPart::Result => 2,
        }
    }
    pub fn matrix_count(&self, part: MatmulPart) -> ElementCount {
        self.matrices[Self::part_index(part)]
    }
    pub fn index(&self, part: MatmulPart, batch: i64, element: i64) -> Result<i64, MetadataError> {
        let matrix = self.matrix_count(part).get();
        if batch < 0 || batch >= self.batches.get() || element < 0 || element >= matrix {
            return Err(MetadataError::Domain(
                "matmul index outside batch or matrix".into(),
            ));
        }
        let index = batch
            .checked_mul(matrix)
            .and_then(|n| n.checked_add(element))
            .ok_or(MetadataError::Overflow("matmul index exceeds i64"))?;
        if index >= self.totals[Self::part_index(part)].get() {
            return Err(MetadataError::Domain("matmul index outside operand".into()));
        }
        Ok(index)
    }
    pub fn check_vendor(&self, limit: i64) -> Result<(), MetadataError> {
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
pub struct SparseMetadata {
    base: ShapeMetadata,
    indices: ShapeMetadata,
    domain: ShapeMetadata,
    axis: usize,
    elementwise: bool,
    inner: ElementCount,
    source_axis: AxisDecomposition,
    index_per_batch: ElementCount,
    base_outer_per_batch: ElementCount,
}

impl SparseMetadata {
    pub fn new(
        base: &ShapeMetadata,
        indices: &ShapeMetadata,
        axis: i64,
        elementwise: bool,
    ) -> Result<Self, MetadataError> {
        Self::new_paired(base, indices, axis, elementwise, 0)
    }

    pub fn new_paired(
        base: &ShapeMetadata,
        indices: &ShapeMetadata,
        axis: i64,
        elementwise: bool,
        batch_rank: usize,
    ) -> Result<Self, MetadataError> {
        let axis = base.normalize_axis(axis)?;
        if batch_rank > axis
            || batch_rank > indices.shape().len()
            || base.shape()[..batch_rank] != indices.shape()[..batch_rank]
        {
            return Err(MetadataError::Domain(
                "invalid paired sparse batch prefix".into(),
            ));
        }
        let index_per_batch = ElementCount::from_extents(&indices.shape()[batch_rank..])?;
        let base_outer_per_batch = ElementCount::from_extents(&base.shape()[batch_rank..axis])?;
        let shape = if elementwise {
            if batch_rank != 0 {
                return Err(MetadataError::Domain(
                    "element-wise sparse plan has no paired batch prefix".into(),
                ));
            }
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
                .checked_sub(1 + batch_rank)
                .and_then(|rank| rank.checked_add(indices.shape().len()))
                .ok_or(MetadataError::Overflow("sparse domain rank overflow"))?;
            ShapeMetadata::checked_rank(rank)?;
            ElementCount::scratch_entries(rank, 0)?.scratch_len::<i64>()?;
            base.shape()[..axis]
                .iter()
                .chain(indices.shape()[batch_rank..].iter())
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
            source_axis: base.axis_decomposition(axis)?,
            index_per_batch,
            base_outer_per_batch,
        })
    }

    pub fn base(&self) -> &ShapeMetadata {
        &self.base
    }
    pub fn domain(&self) -> &ShapeMetadata {
        &self.domain
    }
    pub fn index_slot(&self, linear: i64) -> Result<i64, MetadataError> {
        self.domain.require_index(linear)?;
        Ok(if self.elementwise {
            linear
        } else {
            (linear / self.inner.get() / self.index_per_batch.get())
                / self.base_outer_per_batch.get()
                * self.index_per_batch.get()
                + linear / self.inner.get() % self.index_per_batch.get()
        })
    }
    pub fn data_index(&self, linear: i64, selected: i64) -> Result<i64, MetadataError> {
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
                    .checked_mul(self.base.strides[axis])
                    .and_then(|n| offset.checked_add(n))
                    .ok_or(MetadataError::Overflow("sparse offset exceeds i64"))?;
            }
            self.base.require_index(offset)?;
            Ok(offset)
        } else {
            let outer = linear / self.inner.get() / self.index_per_batch.get();
            let inner = linear % self.inner.get();
            let offset = self.source_axis.linear_index(
                usize::try_from(outer)
                    .map_err(|_| MetadataError::Overflow("sparse outer exceeds target"))?,
                usize::try_from(selected)
                    .map_err(|_| MetadataError::Overflow("sparse index exceeds target"))?,
                usize::try_from(inner)
                    .map_err(|_| MetadataError::Overflow("sparse inner exceeds target"))?,
            )?;
            i64::try_from(offset).map_err(|_| MetadataError::Overflow("sparse offset exceeds i64"))
        }
    }
}

/// Checked reduction grouping, independent of input payload storage and lifetime.
pub struct ReductionMetadata {
    input: IterationSpace,
    selected: Box<[bool]>,
    result: ShapeMetadata,
    leaves: ElementCount,
}

impl ReductionMetadata {
    pub fn new(shape: &[i64], axes: &[i64], dtype: RuntimeDType) -> Result<Self, MetadataError> {
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
    pub fn result(&self) -> &ShapeMetadata {
        &self.result
    }
    pub fn leaves(&self) -> ElementCount {
        self.leaves
    }
    pub fn extent(&self, axis: i64) -> Result<i64, MetadataError> {
        Ok(self.result.shape()[self.result.normalize_axis(axis)?])
    }
    pub fn index(&self, mut outer: i64, mut leaf: i64) -> Result<i64, MetadataError> {
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
                .ok_or(MetadataError::Overflow("reduction offset exceeds i64"))?;
            stride = stride
                .checked_mul(extent)
                .ok_or(MetadataError::Overflow("reduction stride exceeds i64"))?;
        }
        Ok(index)
    }
}

impl IterationSpace {
    pub fn new(shape: &[i64]) -> Result<Self, MetadataError> {
        ShapeMetadata::checked_rank(shape.len())?;
        let elements = ElementCount::from_extents(shape)?;
        Ok(Self {
            shape: shape.into(),
            elements,
        })
    }
    pub fn elements(&self) -> ElementCount {
        self.elements
    }
    pub fn elementwise_index_step(&self, input: &ShapeMetadata) -> Result<i64, MetadataError> {
        input.index_step_for_checked_shape(&self.shape)
    }
    pub fn unravel(&self, linear: i64, out: &mut [i64]) -> Result<(), MetadataError> {
        unravel(&self.shape, self.elements, linear, out)
    }
}

impl ElementCount {
    pub fn scratch_entries(length: usize, extra: usize) -> Result<Self, MetadataError> {
        let length = length
            .checked_add(extra)
            .ok_or(MetadataError::Overflow("scratch entry count exceeds usize"))?;
        i64::try_from(length)
            .map(Self)
            .map_err(|_| MetadataError::Overflow("scratch entry count exceeds i64"))
    }

    pub fn from_extents(extents: &[i64]) -> Result<Self, MetadataError> {
        if let Some((axis, extent)) = extents.iter().enumerate().find(|(_, extent)| **extent < 0) {
            return Err(MetadataError::Domain(
                format!("has negative extent {extent} at axis {axis}").into(),
            ));
        }
        // Validate the whole domain before observing zero. A count is not a
        // running prefix: valid empty shapes never overflow the total count.
        if extents.contains(&0) {
            return Ok(Self(0));
        }
        extents.iter().try_fold(Self(1), |product, &extent| {
            product
                .0
                .checked_mul(extent)
                .map(Self)
                .ok_or(MetadataError::Overflow("extent product exceeds i64"))
        })
    }
    pub fn get(self) -> i64 {
        self.0
    }
    pub fn as_usize(self) -> Result<usize, MetadataError> {
        usize::try_from(self.0).map_err(|_| MetadataError::Overflow("element count exceeds usize"))
    }
    /// Physical Rust scratch entries can include an accumulator and an index;
    /// their allocation layout is distinct from a Chelis tensor representation.
    pub fn scratch_len<T>(self) -> Result<usize, MetadataError> {
        self.layout_bytes(size_of::<T>())?.allocation()?;
        self.as_usize()
    }
    pub fn bytes(self, dtype: RuntimeDType) -> Result<ByteCount, MetadataError> {
        self.layout_bytes(dtype.contract().repr().byte_width())
    }
    fn layout_bytes(self, width: usize) -> Result<ByteCount, MetadataError> {
        let width = i64::try_from(width)
            .map_err(|_| MetadataError::Overflow("representation width exceeds i64"))?;
        self.0
            .checked_mul(width)
            .map(ByteCount)
            .ok_or(MetadataError::Overflow("byte size exceeds i64"))
    }
}

impl ByteCount {
    pub fn from_declared(value: i64) -> Result<Self, MetadataError> {
        if value < 0 {
            Err(MetadataError::Domain(
                format!("has negative byte capacity {value}").into(),
            ))
        } else {
            Ok(Self(value))
        }
    }
    pub fn get(self) -> i64 {
        self.0
    }
    pub fn project_limit(self, limit: u64) -> Result<u64, MetadataError> {
        let value = u64::try_from(self.0)
            .map_err(|_| MetadataError::Domain("negative byte count".into()))?;
        if value > limit {
            Err(MetadataError::Overflow(
                "byte size exceeds target allocation domain",
            ))
        } else {
            Ok(value)
        }
    }
    pub fn allocation(self) -> Result<AllocationBytes, MetadataError> {
        // Rust pointer arithmetic and Vec allocations additionally require a
        // single object to fit isize, even when size_t admits a larger value.
        let value = self.project_limit(isize::MAX as u64)?;
        usize::try_from(value)
            .map(AllocationBytes)
            .map_err(|_| MetadataError::Overflow("byte size exceeds usize"))
    }
}

impl AllocationBytes {
    pub fn get(self) -> usize {
        self.0
    }
}

impl ShapeMetadata {
    pub fn checked_rank(rank: usize) -> Result<i32, MetadataError> {
        i32::try_from(rank).map_err(|_| MetadataError::Overflow("rank exceeds i32"))
    }
    pub fn contiguous(shape: &[i64], dtype: RuntimeDType) -> Result<Self, MetadataError> {
        let domain = CheckedDomain::new(shape, dtype)?;
        // Suffix strides have their own domain even when the count is zero.
        let mut strides = vec![0; shape.len()];
        let mut stride = 1_i64;
        for axis in (0..shape.len()).rev() {
            strides[axis] = stride;
            stride = stride
                .checked_mul(shape[axis])
                .ok_or(MetadataError::Overflow("stride product exceeds i64"))?;
        }
        Ok(Self {
            domain,
            strides: strides.into_boxed_slice(),
        })
    }
    pub fn rank(&self) -> i32 {
        self.domain.rank
    }
    pub fn shape(&self) -> &[i64] {
        &self.domain.shape
    }
    pub fn extent_at(&self, axis: i64) -> Result<i64, MetadataError> {
        Ok(self.domain.shape[self.normalize_axis(axis)?])
    }
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }
    pub fn elements(&self) -> ElementCount {
        self.domain.elements
    }
    pub fn bytes(&self) -> ByteCount {
        self.domain.bytes
    }
    pub fn dtype(&self) -> RuntimeDType {
        self.domain.dtype
    }
    pub fn elementwise_index_step(&self, domain: &Self) -> Result<i64, MetadataError> {
        self.index_step_for_checked_shape(&domain.domain.shape)
    }
    // Only checked owners in this module may provide a domain. In particular,
    // scalar and empty shortcuts must never bypass construction of that owner.
    fn index_step_for_checked_shape(&self, domain: &[i64]) -> Result<i64, MetadataError> {
        if self.domain.rank == 0 {
            Ok(0)
        } else if self.domain.shape.as_ref() == domain {
            Ok(1)
        } else {
            Err(MetadataError::Domain(
                "elementwise input shape does not match iteration domain".into(),
            ))
        }
    }
    pub fn flat_index(&self, indices: &[i64]) -> Result<usize, MetadataError> {
        if indices.len() != self.domain.shape.len() {
            return Err(MetadataError::Domain(
                "index rank does not match tensor rank".into(),
            ));
        }
        self.flat_index_by(|axis| indices[axis])
    }
    /// Decode foreign coordinates without allocating a second coordinate array.
    /// The callback supplies values, never strides or an unchecked offset.
    pub fn flat_index_by(
        &self,
        mut coordinate: impl FnMut(usize) -> i64,
    ) -> Result<usize, MetadataError> {
        self.try_flat_index_by(|axis| Ok(coordinate(axis)))
    }
    fn try_flat_index_by(
        &self,
        mut coordinate: impl FnMut(usize) -> Result<i64, MetadataError>,
    ) -> Result<usize, MetadataError> {
        let mut flat = 0_i64;
        for (axis, (&extent, &stride)) in self.domain.shape.iter().zip(self.strides()).enumerate() {
            let index = coordinate(axis)?;
            if index < 0 || index >= extent {
                return Err(MetadataError::Domain("tensor index outside shape".into()));
            }
            flat = index
                .checked_mul(stride)
                .and_then(|part| flat.checked_add(part))
                .ok_or(MetadataError::Overflow("tensor index offset exceeds i64"))?;
        }
        self.require_index(flat)?;
        usize::try_from(flat).map_err(|_| MetadataError::Overflow("tensor index exceeds usize"))
    }
    pub fn affine_index_by(
        &self,
        mut terms: impl FnMut(usize) -> (i64, i64, i64),
    ) -> Result<usize, MetadataError> {
        self.try_flat_index_by(|axis| {
            let (coordinate, offset, step) = terms(axis);
            if coordinate < 0 || offset < 0 || step <= 0 {
                return Err(MetadataError::Domain(
                    "invalid affine coordinate, offset, or step".into(),
                ));
            }
            coordinate
                .checked_mul(step)
                .and_then(|n| n.checked_add(offset))
                .ok_or(MetadataError::Overflow("affine coordinate exceeds i64"))
        })
    }
    fn movement_shape(
        &self,
        mut extent: impl FnMut(usize, i64) -> Result<i64, MetadataError>,
    ) -> Result<Self, MetadataError> {
        let count = ElementCount::scratch_entries(self.domain.shape.len(), 0)?;
        let mut shape = Vec::with_capacity(count.scratch_len::<i64>()?);
        for (axis, &input) in self.domain.shape.iter().enumerate() {
            shape.push(extent(axis, input)?);
        }
        let result = Self::contiguous(&shape, self.domain.dtype)?;
        result.domain.bytes.allocation()?;
        Ok(result)
    }
    pub fn padded(&self, before: &[i64], after: &[i64]) -> Result<Self, MetadataError> {
        self.require_bound_ranks(before, after)?;
        self.movement_shape(|axis, input| {
            if before[axis] < 0 || after[axis] < 0 {
                return Err(MetadataError::Domain("negative padding bound".into()));
            }
            input
                .checked_add(before[axis])
                .and_then(|n| n.checked_add(after[axis]))
                .ok_or(MetadataError::Overflow("padded extent exceeds i64"))
        })
    }
    pub fn shrunk(&self, start: &[i64], end: &[i64]) -> Result<Self, MetadataError> {
        self.require_bound_ranks(start, end)?;
        self.movement_shape(|axis, input| {
            if start[axis] < 0 || end[axis] < start[axis] || end[axis] > input {
                return Err(MetadataError::Domain(
                    "shrink bounds outside input extent".into(),
                ));
            }
            // The ordered nonnegative bounds prove this difference representable.
            Ok(end[axis] - start[axis])
        })
    }
    pub fn strided(&self, steps: &[i64]) -> Result<Self, MetadataError> {
        self.require_bound_ranks(steps, steps)?;
        self.movement_shape(|axis, input| {
            let step = steps[axis];
            if step <= 0 {
                return Err(MetadataError::Domain("stride step must be positive".into()));
            }
            // Unlike (input + step - 1) / step, this is valid at i64::MAX.
            Ok(input / step + i64::from(input % step != 0))
        })
    }
    fn require_bound_ranks(&self, first: &[i64], second: &[i64]) -> Result<(), MetadataError> {
        if first.len() != self.domain.shape.len() || second.len() != self.domain.shape.len() {
            return Err(MetadataError::Domain("movement bound rank mismatch".into()));
        }
        Ok(())
    }
    pub fn unravel(&self, linear: i64, out: &mut [i64]) -> Result<(), MetadataError> {
        unravel(&self.domain.shape, self.domain.elements, linear, out)
    }
    pub fn unravel_into(
        &self,
        linear: i64,
        write: impl FnMut(usize, i64),
    ) -> Result<(), MetadataError> {
        unravel_into(&self.domain.shape, self.domain.elements, linear, write)
    }
    pub fn require_permutation(&self, target: &Self, axes: &[i64]) -> Result<(), MetadataError> {
        if self.domain.rank != target.domain.rank
            || axes.len() != self.domain.shape.len()
            || self.domain.dtype != target.domain.dtype
        {
            return Err(MetadataError::Domain(
                "permutation rank or representation mismatch".into(),
            ));
        }
        for (out_axis, &axis) in axes.iter().enumerate() {
            let axis = self.normalize_axis(axis)?;
            for &previous in &axes[..out_axis] {
                if self.normalize_axis(previous)? == axis {
                    return Err(MetadataError::Domain(
                        "permutation axes are not a bijection".into(),
                    ));
                }
            }
            if target.domain.shape[out_axis] != self.domain.shape[axis] {
                return Err(MetadataError::Domain(
                    "permutation target extent mismatch".into(),
                ));
            }
        }
        Ok(())
    }
    pub fn require_expansion(&self, target: &Self, axis: i32) -> Result<(), MetadataError> {
        let inserted = i64::from(target.domain.rank) == i64::from(self.domain.rank) + 1;
        if (!inserted && target.domain.rank != self.domain.rank)
            || target.domain.dtype != self.domain.dtype
        {
            return Err(MetadataError::Domain(
                "expansion rank or representation mismatch".into(),
            ));
        }
        let axis = target.normalize_axis(i64::from(axis))?;
        if !inserted && self.domain.shape[axis] != 1 {
            return Err(MetadataError::Domain(
                "expansion replacement axis is not unit".into(),
            ));
        }
        for (out_axis, &extent) in target.domain.shape.iter().enumerate() {
            if out_axis == axis {
                continue;
            }
            let input_axis = if inserted && out_axis > axis {
                out_axis - 1
            } else {
                out_axis
            };
            if extent != self.domain.shape[input_axis] {
                return Err(MetadataError::Domain(
                    "expansion bystander extent mismatch".into(),
                ));
            }
        }
        Ok(())
    }
    fn normalize_axis(&self, axis: i64) -> Result<usize, MetadataError> {
        let axis = if axis < 0 {
            axis + i64::from(self.domain.rank)
        } else {
            axis
        };
        if axis < 0 || axis >= i64::from(self.domain.rank) {
            return Err(MetadataError::Domain("axis outside tensor rank".into()));
        }
        usize::try_from(axis).map_err(|_| MetadataError::Overflow("axis exceeds usize"))
    }
    pub fn byte_offset(&self, linear: i64) -> Result<AllocationBytes, MetadataError> {
        self.require_index(linear)?;
        ElementCount(linear).bytes(self.domain.dtype)?.allocation()
    }
    pub fn require_capacity(&self, capacity: ByteCount) -> Result<(), MetadataError> {
        require_byte_capacity(self.domain.bytes, capacity)
    }
    pub fn axis_decomposition(&self, axis: usize) -> Result<AxisDecomposition, MetadataError> {
        let &extent = self
            .domain
            .shape
            .get(axis)
            .ok_or(MetadataError::Domain("axis outside tensor rank".into()))?;
        if self.domain.elements.0 == 0 {
            return Ok(AxisDecomposition {
                outer: ElementCount(0),
                extent: ElementCount(extent),
                inner: ElementCount(0),
            });
        }
        Ok(AxisDecomposition {
            outer: ElementCount::from_extents(&self.domain.shape[..axis])?,
            extent: ElementCount(extent),
            inner: ElementCount::from_extents(&self.domain.shape[axis + 1..])?,
        })
    }

    fn require_index(&self, linear: i64) -> Result<(), MetadataError> {
        if linear < 0 || linear >= self.domain.elements.0 {
            Err(MetadataError::Domain(
                "tensor index outside element count".into(),
            ))
        } else {
            Ok(())
        }
    }
}

impl AxisDecomposition {
    pub fn outer(&self) -> ElementCount {
        self.outer
    }
    pub fn extent(&self) -> ElementCount {
        self.extent
    }
    pub fn inner(&self) -> ElementCount {
        self.inner
    }
    pub fn linear_index(
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
    pub fn reduced_index(&self, outer: usize, inner: usize) -> Result<usize, MetadataError> {
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
    if linear < 0 || linear >= elements.0 {
        return Err(MetadataError::Domain(
            "tensor index outside element count".into(),
        ));
    }
    // The checked nonempty range proves every divisor positive.
    for (axis, &extent) in shape.iter().enumerate().rev() {
        write(axis, linear % extent);
        linear /= extent;
    }
    Ok(())
}
