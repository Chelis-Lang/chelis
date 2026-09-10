//! Shared checked descriptor metadata authority: [04-SHAPE-1], [05-OP-31/33/44].

use chelis_vocab::RuntimeDType;

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

#[derive(Clone, Debug)]
pub struct ShapeMetadata {
    shape: Box<[i64]>,
    strides: Box<[i64]>,
    rank: i32,
    elements: ElementCount,
    bytes: ByteCount,
    dtype: RuntimeDType,
}

/// A zero-offset forward view with a checked supplied-stride storage span.
///
/// This cannot enter contiguous host indexing: logical bytes describe the
/// tensor values, while `required_span` describes reachable storage.
#[derive(Clone, Debug)]
pub struct StridedMetadata {
    shape: Box<[i64]>,
    strides: Box<[i64]>,
    rank: i32,
    elements: ElementCount,
    bytes: ByteCount,
    dtype: RuntimeDType,
    required_span: ByteCount,
}

impl ElementCount {
    pub fn scratch_entries(length: usize, extra: usize) -> Result<Self, MetadataError> {
        let length = length
            .checked_add(extra)
            .ok_or(MetadataError::Overflow("scratch entry count exceeds usize"))?;
        i64::try_from(length)
            .map(Self)
            .map_err(|_| MetadataError::Overflow("scratch entry count exceeds int64"))
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
                .ok_or(MetadataError::Overflow("extent product exceeds int64"))
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
            .map_err(|_| MetadataError::Overflow("representation width exceeds int64"))?;
        self.0
            .checked_mul(width)
            .map(ByteCount)
            .ok_or(MetadataError::Overflow("byte size exceeds int64"))
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
        if strides.iter().any(|&stride| stride < 0) {
            return Err(MetadataError::Domain(
                "negative stride in forward view".into(),
            ));
        }
        let rank = ShapeMetadata::checked_rank(shape.len())?;
        let elements = ElementCount::from_extents(shape)?;
        let bytes = elements.bytes(dtype)?;
        let span = if elements.get() == 0 {
            ElementCount::from_extents(&[0])?
        } else {
            let mut largest = 0_i64;
            for (&extent, &stride) in shape.iter().zip(strides) {
                let part = (extent - 1)
                    .checked_mul(stride)
                    .ok_or(MetadataError::Overflow("view offset product exceeds int64"))?;
                largest = largest
                    .checked_add(part)
                    .ok_or(MetadataError::Overflow("view offset sum exceeds int64"))?;
            }
            ElementCount::from_extents(&[largest
                .checked_add(1)
                .ok_or(MetadataError::Overflow("view element span exceeds int64"))?])?
        };
        let required_span = span.bytes(dtype)?;
        required_span.allocation()?;
        let metadata = Self {
            shape: shape.into(),
            strides: strides.into(),
            rank,
            elements,
            bytes,
            dtype,
            required_span,
        };
        metadata.require_capacity(byte_capacity)?;
        Ok(metadata)
    }

    pub fn rank(&self) -> i32 {
        self.rank
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    pub fn elements(&self) -> ElementCount {
        self.elements
    }

    pub fn bytes(&self) -> ByteCount {
        self.bytes
    }

    pub fn dtype(&self) -> RuntimeDType {
        self.dtype
    }

    pub fn required_span(&self) -> ByteCount {
        self.required_span
    }

    pub fn byte_offset(&self, linear: i64) -> Result<AllocationBytes, MetadataError> {
        let mut offset = Some(0_i64);
        unravel_into(&self.shape, self.elements, linear, |axis, coordinate| {
            offset = offset.and_then(|offset| {
                coordinate
                    .checked_mul(self.strides[axis])
                    .and_then(|part| offset.checked_add(part))
            });
        })?;
        let offset = offset.ok_or(MetadataError::Overflow("view index offset exceeds int64"))?;
        ElementCount(offset).bytes(self.dtype)?.allocation()
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

impl ShapeMetadata {
    pub fn checked_rank(rank: usize) -> Result<i32, MetadataError> {
        i32::try_from(rank).map_err(|_| MetadataError::Overflow("rank exceeds int32"))
    }

    pub fn contiguous(shape: &[i64], dtype: RuntimeDType) -> Result<Self, MetadataError> {
        let rank = Self::checked_rank(shape.len())?;
        let elements = ElementCount::from_extents(shape)?;
        let bytes = elements.bytes(dtype)?;
        // Suffix strides have their own domain even when the count is zero.
        let mut strides = vec![0; shape.len()];
        let mut stride = 1_i64;
        for axis in (0..shape.len()).rev() {
            strides[axis] = stride;
            stride = stride
                .checked_mul(shape[axis])
                .ok_or(MetadataError::Overflow("stride product exceeds int64"))?;
        }
        Ok(Self {
            shape: shape.into(),
            strides: strides.into_boxed_slice(),
            rank,
            elements,
            bytes,
            dtype,
        })
    }

    pub fn rank(&self) -> i32 {
        self.rank
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn extent_at(&self, axis: i64) -> Result<i64, MetadataError> {
        Ok(self.shape[self.normalize_axis(axis)?])
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    pub fn elements(&self) -> ElementCount {
        self.elements
    }

    pub fn bytes(&self) -> ByteCount {
        self.bytes
    }

    pub fn dtype(&self) -> RuntimeDType {
        self.dtype
    }

    pub fn elementwise_index_step(&self, domain: &Self) -> Result<i64, MetadataError> {
        self.index_step_for_checked_shape(&domain.shape)
    }

    fn index_step_for_checked_shape(&self, domain: &[i64]) -> Result<i64, MetadataError> {
        if self.rank == 0 {
            Ok(0)
        } else if self.shape.as_ref() == domain {
            Ok(1)
        } else {
            Err(MetadataError::Domain(
                "elementwise input shape does not match iteration domain".into(),
            ))
        }
    }

    pub fn flat_index(&self, indices: &[i64]) -> Result<usize, MetadataError> {
        if indices.len() != self.shape.len() {
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
        for (axis, (&extent, &stride)) in self.shape.iter().zip(self.strides()).enumerate() {
            let index = coordinate(axis)?;
            if index < 0 || index >= extent {
                return Err(MetadataError::Domain("tensor index outside shape".into()));
            }
            flat = index
                .checked_mul(stride)
                .and_then(|part| flat.checked_add(part))
                .ok_or(MetadataError::Overflow("tensor index offset exceeds int64"))?;
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
                .ok_or(MetadataError::Overflow("affine coordinate exceeds int64"))
        })
    }

    fn movement_shape(
        &self,
        mut extent: impl FnMut(usize, i64) -> Result<i64, MetadataError>,
    ) -> Result<Self, MetadataError> {
        let count = ElementCount::scratch_entries(self.shape.len(), 0)?;
        let mut shape = Vec::with_capacity(count.scratch_len::<i64>()?);
        for (axis, &input) in self.shape.iter().enumerate() {
            shape.push(extent(axis, input)?);
        }
        let result = Self::contiguous(&shape, self.dtype)?;
        result.bytes.allocation()?;
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
                .ok_or(MetadataError::Overflow("padded extent exceeds int64"))
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
            // Unlike (input + step - 1) / step, this is valid at int64::MAX.
            Ok(input / step + i64::from(input % step != 0))
        })
    }

    fn require_bound_ranks(&self, first: &[i64], second: &[i64]) -> Result<(), MetadataError> {
        if first.len() != self.shape.len() || second.len() != self.shape.len() {
            return Err(MetadataError::Domain("movement bound rank mismatch".into()));
        }
        Ok(())
    }

    pub fn unravel(&self, linear: i64, out: &mut [i64]) -> Result<(), MetadataError> {
        unravel(&self.shape, self.elements, linear, out)
    }

    pub fn unravel_into(
        &self,
        linear: i64,
        write: impl FnMut(usize, i64),
    ) -> Result<(), MetadataError> {
        unravel_into(&self.shape, self.elements, linear, write)
    }

    pub fn require_permutation(&self, target: &Self, axes: &[i64]) -> Result<(), MetadataError> {
        if self.rank != target.rank || axes.len() != self.shape.len() || self.dtype != target.dtype
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
            if target.shape[out_axis] != self.shape[axis] {
                return Err(MetadataError::Domain(
                    "permutation target extent mismatch".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn require_expansion(&self, target: &Self, axis: i32) -> Result<(), MetadataError> {
        let inserted = i64::from(target.rank) == i64::from(self.rank) + 1;
        if (!inserted && target.rank != self.rank) || target.dtype != self.dtype {
            return Err(MetadataError::Domain(
                "expansion rank or representation mismatch".into(),
            ));
        }
        let axis = target.normalize_axis(i64::from(axis))?;
        if !inserted && self.shape[axis] != 1 {
            return Err(MetadataError::Domain(
                "expansion replacement axis is not unit".into(),
            ));
        }
        for (out_axis, &extent) in target.shape.iter().enumerate() {
            if out_axis == axis {
                continue;
            }
            let input_axis = if inserted && out_axis > axis {
                out_axis - 1
            } else {
                out_axis
            };
            if extent != self.shape[input_axis] {
                return Err(MetadataError::Domain(
                    "expansion bystander extent mismatch".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn normalize_axis(&self, axis: i64) -> Result<usize, MetadataError> {
        let axis = if axis < 0 {
            axis + i64::from(self.rank)
        } else {
            axis
        };
        if axis < 0 || axis >= i64::from(self.rank) {
            return Err(MetadataError::Domain("axis outside tensor rank".into()));
        }
        usize::try_from(axis).map_err(|_| MetadataError::Overflow("axis exceeds usize"))
    }

    pub fn byte_offset(&self, linear: i64) -> Result<AllocationBytes, MetadataError> {
        self.require_index(linear)?;
        ElementCount(linear).bytes(self.dtype)?.allocation()
    }

    pub fn require_capacity(&self, capacity: ByteCount) -> Result<(), MetadataError> {
        require_byte_capacity(self.bytes, capacity)
    }

    pub fn require_index(&self, linear: i64) -> Result<(), MetadataError> {
        if linear < 0 || linear >= self.elements.0 {
            Err(MetadataError::Domain(
                "tensor index outside element count".into(),
            ))
        } else {
            Ok(())
        }
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
