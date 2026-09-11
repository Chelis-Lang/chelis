//! The existing checked authority must move without weakening these contracts.
//! C2.2, [05-OP-31/33/44]; no tensor storage allocation is needed by this suite.

use chelis_abi::metadata::{ByteCount, ElementCount, MetadataError, ShapeMetadata};
use chelis_vocab::RuntimeDType;

fn domain<T>(result: Result<T, MetadataError>) {
    assert!(matches!(result, Err(MetadataError::Domain(_))));
}

fn overflow<T>(result: Result<T, MetadataError>) {
    assert!(matches!(result, Err(MetadataError::Overflow(_))));
}

#[test]
fn rank_zero_has_one_element_and_rank_overflow_is_rejected() {
    let scalar = ShapeMetadata::contiguous(&[], RuntimeDType::F64).unwrap();
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.elements().get(), 1);
    assert_eq!(scalar.bytes().get(), 8);
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.strides(), &[]);
    overflow(ShapeMetadata::checked_rank(i32::MAX as usize + 1));
}

#[test]
fn rank_one_eight_and_nine_have_no_fixed_array_limit() {
    for rank in [1, 8, 9] {
        let shape = vec![1_i64; rank];
        let metadata = ShapeMetadata::contiguous(&shape, RuntimeDType::F32).unwrap();
        assert_eq!(metadata.rank(), rank as i32);
        assert_eq!(metadata.shape(), shape);
        assert_eq!(metadata.strides(), vec![1; rank]);
        assert_eq!(metadata.elements().get(), 1);
        let mut invalid = shape;
        invalid[rank - 1] = -1;
        domain(ShapeMetadata::contiguous(&invalid, RuntimeDType::F32));
    }
}

#[test]
fn int64_extent_count_and_capacity_survive_above_int32() {
    let count = i64::from(i32::MAX) + 1;
    let metadata = ShapeMetadata::contiguous(&[count], RuntimeDType::F64).unwrap();
    assert_eq!(metadata.shape(), &[count]);
    assert_eq!(metadata.elements().get(), count);
    assert_eq!(metadata.bytes().get(), count * 8);
    metadata
        .require_capacity(ByteCount::from_declared(count * 8).unwrap())
        .unwrap();
    domain(metadata.require_capacity(ByteCount::from_declared(count * 8 - 1).unwrap()));
}

#[test]
fn zero_counts_do_not_hide_negative_extents() {
    for shape in [[0, i64::MAX, i64::MAX], [i64::MAX, i64::MAX, 0]] {
        assert_eq!(ElementCount::from_extents(&shape).unwrap().get(), 0);
    }
    for shape in [[0, -1], [-1, 0]] {
        domain(ElementCount::from_extents(&shape));
    }
}

#[test]
fn empty_shape_still_requires_representable_canonical_strides() {
    let empty = ShapeMetadata::contiguous(&[i64::MAX, i64::MAX, 0], RuntimeDType::F64).unwrap();
    assert_eq!(empty.elements().get(), 0);
    assert_eq!(empty.bytes().get(), 0);
    assert_eq!(empty.strides(), &[0, 0, 1]);
    overflow(ShapeMetadata::contiguous(
        &[0, i64::MAX, i64::MAX],
        RuntimeDType::F64,
    ));
}

#[test]
fn extent_product_overflow_is_distinct_from_byte_overflow() {
    assert_eq!(
        ElementCount::from_extents(&[i64::MAX]).unwrap().get(),
        i64::MAX
    );
    overflow(ElementCount::from_extents(&[i64::MAX, 2]));
    let count = ElementCount::from_extents(&[i64::MAX]).unwrap();
    assert_eq!(count.bytes(RuntimeDType::I8).unwrap().get(), i64::MAX);
    overflow(count.bytes(RuntimeDType::F64));
}

#[test]
fn target_projection_checks_declared_capacity_without_allocating_payload() {
    let capacity = ByteCount::from_declared(i64::from(u32::MAX)).unwrap();
    assert_eq!(
        capacity.project_limit(u64::from(u32::MAX)).unwrap(),
        u64::from(u32::MAX)
    );
    overflow(capacity.project_limit(u64::from(u32::MAX) - 1));
    domain(ByteCount::from_declared(-1));
    assert_eq!(
        ByteCount::from_declared(0)
            .unwrap()
            .allocation()
            .unwrap()
            .get(),
        0
    );
}

#[test]
fn exact_dtype_is_retained_when_storage_width_matches() {
    let float = ShapeMetadata::contiguous(&[2, 3], RuntimeDType::F32).unwrap();
    let integer = ShapeMetadata::contiguous(&[2, 3], RuntimeDType::I32).unwrap();
    assert_eq!(float.bytes(), integer.bytes());
    assert_eq!(float.dtype(), RuntimeDType::F32);
    assert_eq!(integer.dtype(), RuntimeDType::I32);
    assert_ne!(float.dtype().repr(), integer.dtype().repr());
}

#[test]
fn byte_offsets_reject_out_of_range_coordinates() {
    let metadata = ShapeMetadata::contiguous(&[2, 3], RuntimeDType::F64).unwrap();
    assert_eq!(metadata.flat_index(&[1, 2]).unwrap(), 5);
    assert_eq!(metadata.byte_offset(5).unwrap().get(), 40);
    domain(metadata.flat_index(&[2, 0]));
    domain(metadata.flat_index(&[-1, 0]));
    domain(metadata.flat_index(&[0]));
    domain(metadata.byte_offset(6));
}
