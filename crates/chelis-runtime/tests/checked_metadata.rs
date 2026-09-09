//! [04-SHAPE-1], [05-OP-31], [05-OP-33]: metadata is checked before access.
//! Include the private owner so these tests exercise its actual implementation
//! without making metadata a new public runtime API.
#[path = "../src/metadata.rs"]
mod metadata;

use chelis_vocab::RuntimeDType;
use metadata::{ByteCount, ElementCount, IterationSpace, MetadataError, ShapeMetadata};

#[test]
fn checked_iteration_steps_preserve_exact_large_domains_without_storage() {
    let scalar = ShapeMetadata::contiguous(&[], RuntimeDType::F64).unwrap();
    for extent in [i32::MAX as i64 + 1, 9_007_199_254_740_993, i64::MAX] {
        let input = ShapeMetadata::contiguous(&[extent], RuntimeDType::I8).unwrap();
        let domain = IterationSpace::new(&[extent]).unwrap();
        assert_eq!(domain.elementwise_index_step(&input).unwrap(), 1);
        assert_eq!(input.elementwise_index_step(&input).unwrap(), 1);
        assert_eq!(scalar.elementwise_index_step(&input).unwrap(), 0);
        assert!(matches!(
            input.elementwise_index_step(&scalar),
            Err(MetadataError::Domain(_))
        ));
        assert_eq!(domain.elementwise_index_step(&scalar).unwrap(), 0);
        assert_eq!(domain.elements().get(), extent);
        let different = IterationSpace::new(&[1, extent]).unwrap();
        assert!(matches!(
            different.elementwise_index_step(&input),
            Err(MetadataError::Domain(_))
        ));
    }
    let empty = IterationSpace::new(&[0, i64::MAX, i64::MAX]).unwrap();
    assert_eq!(empty.elements().get(), 0);
    assert_eq!(empty.elementwise_index_step(&scalar).unwrap(), 0);
    assert!(matches!(
        IterationSpace::new(&[0, -1]),
        Err(MetadataError::Domain(_))
    ));
    assert!(matches!(
        IterationSpace::new(&[3_074_457_345_618_258_603, 3]),
        Err(MetadataError::Overflow(_))
    ));
}

#[test]
fn rank_zero_and_each_dtype_have_exact_count_bytes_and_strides() {
    for dtype in RuntimeDType::ALL {
        let scalar = ShapeMetadata::contiguous(&[], dtype).unwrap();
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.elements().get(), 1);
        assert_eq!(scalar.bytes().get(), dtype.byte_width() as i64);
        assert!(scalar.shape().is_empty());
        assert!(scalar.strides().is_empty());
        assert_eq!(scalar.dtype(), dtype);
        assert_eq!(scalar.flat_index(&[]).unwrap(), 0);
        assert!(scalar.flat_index(&[0]).is_err());
        scalar.unravel(0, &mut []).unwrap();
        assert!(scalar.unravel(1, &mut []).is_err());
        let matrix = ShapeMetadata::contiguous(&[2, 3, 4], dtype).unwrap();
        assert_eq!(matrix.rank(), 3);
        assert_eq!(matrix.elements().get(), 24);
        assert_eq!(matrix.strides(), &[12, 4, 1]);
        assert_eq!(matrix.bytes().get(), 24 * dtype.byte_width() as i64);
        let width = dtype.byte_width() as i64;
        let largest = i64::MAX / width;
        assert_eq!(
            ElementCount::from_extents(&[largest])
                .unwrap()
                .bytes(dtype)
                .unwrap()
                .get(),
            largest * width
        );
        if width > 1 {
            assert!(matches!(
                ElementCount::from_extents(&[largest + 1])
                    .unwrap()
                    .bytes(dtype),
                Err(MetadataError::Overflow(_))
            ));
        } else {
            assert!(ElementCount::from_extents(&[i64::MAX, 2]).is_err());
        }
    }
}

#[test]
fn zeros_do_not_hide_negative_extents_or_canonical_stride_overflow() {
    for shape in [&[0, -1][..], &[-1, 0][..]] {
        assert!(matches!(
            ShapeMetadata::contiguous(shape, RuntimeDType::I8),
            Err(MetadataError::Domain(_))
        ));
    }
    let valid = ShapeMetadata::contiguous(&[i64::MAX, i64::MAX, 0], RuntimeDType::F64).unwrap();
    assert_eq!(valid.elements().get(), 0);
    assert_eq!(valid.bytes().get(), 0);
    assert_eq!(valid.strides(), &[0, 0, 1]);
    assert!(matches!(
        ShapeMetadata::contiguous(&[0, i64::MAX, i64::MAX], RuntimeDType::I8),
        Err(MetadataError::Overflow(_))
    ));
}

#[test]
fn exact_int64_counts_reject_product_and_representation_byte_overflow() {
    let count = ElementCount::from_extents(&[4, 1_073_741_826]).unwrap();
    assert_eq!(count.get(), 4_294_967_304);
    assert_eq!(count.bytes(RuntimeDType::I8).unwrap().get(), 4_294_967_304);
    assert_eq!(ElementCount::from_extents(&[i64::MAX, 0]).unwrap().get(), 0);
    assert!(matches!(
        ElementCount::from_extents(&[i64::MAX, 2]),
        Err(MetadataError::Overflow(_))
    ));
    let max = ElementCount::from_extents(&[i64::MAX]).unwrap();
    assert_eq!(max.bytes(RuntimeDType::I8).unwrap().get(), i64::MAX);
    assert!(matches!(
        max.bytes(RuntimeDType::F64),
        Err(MetadataError::Overflow(_))
    ));
    assert!(matches!(
        ByteCount::from_declared(-1),
        Err(MetadataError::Domain(_))
    ));
}

#[test]
fn target_projection_is_checked_without_requesting_a_large_allocation() {
    let below = ByteCount::from_declared(i64::from(u32::MAX)).unwrap();
    let above = ByteCount::from_declared(i64::from(u32::MAX) + 1).unwrap();
    assert_eq!(
        below.project_limit(u64::from(u32::MAX)).unwrap(),
        u64::from(u32::MAX)
    );
    assert!(matches!(
        above.project_limit(u64::from(u32::MAX)),
        Err(MetadataError::Overflow(_))
    ));
    assert_eq!(
        ByteCount::from_declared(0)
            .unwrap()
            .allocation()
            .unwrap()
            .get(),
        0
    );
    assert_eq!(
        ByteCount::from_declared(24)
            .unwrap()
            .allocation()
            .unwrap()
            .get(),
        24
    );
}

#[test]
fn checked_indexing_and_byte_ranges_reject_out_of_bounds_before_access() {
    let metadata = ShapeMetadata::contiguous(&[2, 3], RuntimeDType::I32).unwrap();
    for (indices, expected) in [([0, 0], 0), ([1, 2], 5)] {
        assert_eq!(metadata.flat_index(&indices).unwrap(), expected);
        let mut roundtrip = [0, 0];
        metadata.unravel(expected as i64, &mut roundtrip).unwrap();
        assert_eq!(roundtrip, indices);
    }
    for indices in [&[2, 0][..], &[0, 3][..], &[-1, 0][..], &[0][..]] {
        assert!(matches!(
            metadata.flat_index(indices),
            Err(MetadataError::Domain(_))
        ));
    }
    for linear in [-1, 6] {
        assert!(matches!(
            metadata.unravel(linear, &mut [0, 0]),
            Err(MetadataError::Domain(_))
        ));
    }
    assert_eq!(metadata.byte_offset(5).unwrap().get(), 20);
    assert!(matches!(
        metadata.byte_offset(6),
        Err(MetadataError::Domain(_))
    ));
    assert!(metadata
        .require_capacity(ByteCount::from_declared(24).unwrap())
        .is_ok());
    assert!(metadata
        .require_capacity(ByteCount::from_declared(23).unwrap())
        .is_err());
    let empty = ShapeMetadata::contiguous(&[0, 3], RuntimeDType::I32).unwrap();
    assert!(empty
        .require_capacity(ByteCount::from_declared(0).unwrap())
        .is_ok());
    assert!(empty.flat_index(&[0, 0]).is_err());
    assert!(empty.byte_offset(0).is_err());
}

#[test]
fn axis_decomposition_checks_axis_and_preserves_empty_short_circuit() {
    let metadata = ShapeMetadata::contiguous(&[2, 3, 4], RuntimeDType::I8).unwrap();
    let axis = metadata.axis_decomposition(1).unwrap();
    assert_eq!(
        (axis.outer().get(), axis.extent().get(), axis.inner().get()),
        (2, 3, 4)
    );
    assert_eq!(axis.linear_index(1, 2, 3).unwrap(), 23);
    assert!(axis.linear_index(2, 0, 0).is_err());
    assert!(axis.linear_index(0, 3, 0).is_err());
    assert!(axis.linear_index(0, 0, 4).is_err());
    assert_eq!(axis.reduced_index(1, 3).unwrap(), 7);
    assert!(axis.reduced_index(2, 0).is_err());
    assert!(axis.reduced_index(0, 4).is_err());
    assert!(matches!(
        metadata.axis_decomposition(3),
        Err(MetadataError::Domain(_))
    ));
    let empty = ShapeMetadata::contiguous(&[i64::MAX, i64::MAX, 0], RuntimeDType::I8).unwrap();
    let axis = empty.axis_decomposition(2).unwrap();
    assert_eq!(axis.outer().get(), 0);
    assert_eq!(axis.inner().get(), 0);
}

#[test]
fn reduction_iteration_is_not_a_tensor_stride_contract() {
    // These are loop extents, not storage strides; no iteration can reach a
    // byte, so an empty contraction must not compute irrelevant suffixes.
    let space = IterationSpace::new(&[0, i64::MAX, i64::MAX]).unwrap();
    assert_eq!(space.elements().get(), 0);
    assert!(space.unravel(0, &mut [0, 0, 0]).is_err());
    let space = IterationSpace::new(&[2, 3]).unwrap();
    let mut index = [0, 0];
    space.unravel(5, &mut index).unwrap();
    assert_eq!(index, [1, 2]);
    assert!(space.unravel(6, &mut index).is_err());
    assert!(IterationSpace::new(&[-1, 0]).is_err());
    assert!(IterationSpace::new(&[i64::MAX, 2]).is_err());
}

#[test]
fn scratch_lengths_and_rank_projections_are_checked_before_allocation() {
    assert_eq!(ElementCount::scratch_entries(3, 1).unwrap().get(), 4);
    assert_eq!(ElementCount::scratch_entries(0, 0).unwrap().get(), 0);
    assert!(matches!(
        ElementCount::scratch_entries(usize::MAX, 1),
        Err(MetadataError::Overflow(_))
    ));
    if usize::BITS >= 64 {
        assert!(matches!(
            ElementCount::scratch_entries(i64::MAX as usize, 1),
            Err(MetadataError::Overflow(_))
        ));
    }
    let count = ElementCount::from_extents(&[3]).unwrap();
    assert_eq!(count.scratch_len::<(f32, i64)>().unwrap(), 3);
    assert!(matches!(
        ElementCount::from_extents(&[i64::MAX])
            .unwrap()
            .scratch_len::<(f32, i64)>(),
        Err(MetadataError::Overflow(_))
    ));
    assert_eq!(ShapeMetadata::checked_rank(3).unwrap(), 3);
    assert!(matches!(
        ShapeMetadata::checked_rank(i32::MAX as usize + 1),
        Err(MetadataError::Overflow(_))
    ));
}

#[test]
fn checked_movement_relations_reject_invalid_bijections_and_bystanders() {
    let shape = |dims: &[i64]| ShapeMetadata::contiguous(dims, RuntimeDType::I8).unwrap();
    let input = shape(&[2, 3]);
    assert!(input
        .require_permutation(&shape(&[3, 2]), &[-1, -2])
        .is_ok());
    assert!(input.require_permutation(&shape(&[2, 2]), &[0, 0]).is_err());
    assert!(input.require_permutation(&shape(&[2, 3]), &[1, 0]).is_err());
    assert!(input.require_permutation(&shape(&[3, 2]), &[1]).is_err());
    assert!(input.require_permutation(&shape(&[3, 2]), &[1, 2]).is_err());
    let unit = shape(&[2, 1]);
    assert!(unit.require_expansion(&shape(&[2, 3]), -1).is_ok());
    assert!(unit.require_expansion(&shape(&[4, 2, 1]), 0).is_ok());
    assert!(input.require_expansion(&shape(&[2, 4]), 1).is_err());
    assert!(unit.require_expansion(&shape(&[3, 4]), 1).is_err());
    assert!(unit.require_expansion(&shape(&[4, 3, 1]), 0).is_err());
    assert!(unit.require_expansion(&shape(&[2]), 0).is_err());
    let other = ShapeMetadata::contiguous(&[2, 3], RuntimeDType::I16).unwrap();
    assert!(input.require_permutation(&other, &[0, 1]).is_err());
    assert!(unit.require_expansion(&other, 1).is_err());
}

#[test]
fn checked_movement_coordinates_preserve_exact_large_indices_without_storage() {
    let extent = 9_007_199_254_740_995;
    let metadata = ShapeMetadata::contiguous(&[2, extent], RuntimeDType::I8).unwrap();
    let mut coordinates = [0, 0];
    let linear = extent * 2 - 1;
    metadata
        .unravel_into(linear, |axis, value| coordinates[axis] = value)
        .unwrap();
    assert_eq!(coordinates, [1, extent - 1]);
    assert_eq!(
        metadata.flat_index_by(|axis| coordinates[axis]).unwrap(),
        linear as usize
    );
    assert!(metadata
        .unravel_into(extent * 2, |_, _| panic!("invalid index must not write"))
        .is_err());
    assert!(metadata.flat_index_by(|_| -1).is_err());
}
