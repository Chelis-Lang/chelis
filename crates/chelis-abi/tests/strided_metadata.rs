//! C2.2: supplied-stride metadata has its own checked span, while contiguous
//! host indexing continues to use `ShapeMetadata`.

use chelis_abi::metadata::{ByteCount, MetadataError, ShapeMetadata, StridedMetadata};
use chelis_vocab::RuntimeDType;

fn capacity(value: i64) -> ByteCount {
    ByteCount::from_declared(value).unwrap()
}

fn domain<T>(result: Result<T, MetadataError>) {
    assert!(matches!(result, Err(MetadataError::Domain(_))));
}

fn overflow<T>(result: Result<T, MetadataError>) {
    assert!(matches!(result, Err(MetadataError::Overflow(_))));
}

#[test]
fn permutation_and_gapped_views_check_reachable_span_not_logical_bytes() {
    for dtype in RuntimeDType::ALL {
        let width = dtype.byte_width() as i64;
        let permuted = StridedMetadata::new(&[3, 2], &[1, 3], dtype, capacity(6 * width)).unwrap();
        assert_eq!(permuted.shape(), &[3, 2]);
        assert_eq!(permuted.strides(), &[1, 3]);
        assert_eq!(permuted.elements().get(), 6);
        assert_eq!(permuted.bytes().get(), 6 * width);
        assert_eq!(permuted.required_span().get(), 6 * width);
        let gapped = StridedMetadata::new(&[2, 2], &[4, 2], dtype, capacity(7 * width)).unwrap();
        assert_eq!(gapped.bytes().get(), 4 * width);
        assert_eq!(gapped.required_span().get(), 7 * width);
        domain(StridedMetadata::new(
            &[2, 2],
            &[4, 2],
            dtype,
            capacity(7 * width - 1),
        ));
        domain(gapped.require_capacity(capacity(7 * width - 1)));
    }
}

#[test]
fn broadcast_counts_are_logical_and_zero_stride_needs_only_reachable_storage() {
    let view = StridedMetadata::new(&[1_000, 3], &[0, 1], RuntimeDType::F64, capacity(24)).unwrap();
    assert_eq!(view.elements().get(), 3_000);
    assert_eq!(view.bytes().get(), 24_000);
    assert_eq!(view.required_span().get(), 24);
    view.require_capacity(capacity(24)).unwrap();
    domain(view.require_capacity(capacity(23)));
    let wide = StridedMetadata::new(&[i64::MAX], &[0], RuntimeDType::I8, capacity(1)).unwrap();
    assert_eq!(wide.elements().get(), i64::MAX);
    assert_eq!(wide.required_span().get(), 1);
    overflow(StridedMetadata::new(
        &[i64::MAX],
        &[0],
        RuntimeDType::F64,
        capacity(8),
    ));
}

#[test]
fn empty_views_validate_fields_without_unused_suffix_or_span_arithmetic() {
    for shape in [[0, i64::MAX, i64::MAX], [i64::MAX, i64::MAX, 0]] {
        let view =
            StridedMetadata::new(&shape, &[i64::MAX; 3], RuntimeDType::F64, capacity(0)).unwrap();
        assert_eq!(view.elements().get(), 0);
        assert_eq!(view.bytes().get(), 0);
        assert_eq!(view.required_span().get(), 0);
    }
    domain(StridedMetadata::new(
        &[0, -1],
        &[0, 0],
        RuntimeDType::F32,
        capacity(0),
    ));
    domain(StridedMetadata::new(
        &[0, 1],
        &[0, -1],
        RuntimeDType::F32,
        capacity(0),
    ));
    domain(StridedMetadata::new(
        &[0, 1],
        &[0],
        RuntimeDType::F32,
        capacity(0),
    ));
}

#[test]
fn rank_zero_is_one_element_and_dynamic_rank_has_no_fixed_array_bound() {
    for dtype in RuntimeDType::ALL {
        let width = dtype.byte_width() as i64;
        let scalar = StridedMetadata::new(&[], &[], dtype, capacity(width)).unwrap();
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.elements().get(), 1);
        assert_eq!(scalar.required_span().get(), width);
        domain(StridedMetadata::new(&[], &[], dtype, capacity(width - 1)));
        for rank in [1, 8, 9, 33] {
            let view = StridedMetadata::new(&vec![1; rank], &vec![0; rank], dtype, capacity(width))
                .unwrap();
            assert_eq!(view.rank(), rank as i32);
            domain(StridedMetadata::new(
                &vec![1; rank],
                &vec![0; rank + 1],
                dtype,
                capacity(width),
            ));
        }
    }
}

#[test]
fn offset_span_and_representation_overflow_are_checked_independently() {
    overflow(StridedMetadata::new(
        &[3],
        &[i64::MAX],
        RuntimeDType::I8,
        capacity(i64::MAX),
    ));
    overflow(StridedMetadata::new(
        &[2],
        &[i64::MAX],
        RuntimeDType::I8,
        capacity(i64::MAX),
    ));
    overflow(StridedMetadata::new(
        &[2],
        &[i64::MAX / 2],
        RuntimeDType::F64,
        capacity(i64::MAX),
    ));
    domain(StridedMetadata::new(
        &[2],
        &[-1],
        RuntimeDType::I8,
        capacity(2),
    ));
    let singleton = StridedMetadata::new(&[1], &[i64::MAX], RuntimeDType::I8, capacity(1)).unwrap();
    assert_eq!(singleton.required_span().get(), 1);
}

#[test]
fn host_coordinate_and_iteration_contracts_remain_contiguous() {
    let host = ShapeMetadata::contiguous(&[2, 3], RuntimeDType::F64).unwrap();
    assert_eq!(host.strides(), &[3, 1]);
    assert_eq!(host.flat_index(&[1, 2]).unwrap(), 5);
    assert_eq!(host.byte_offset(5).unwrap().get(), 40);
    domain(host.flat_index(&[2, 0]));
    domain(host.byte_offset(6));
    let view = StridedMetadata::new(&[2, 3], &[0, 1], RuntimeDType::F64, capacity(24)).unwrap();
    assert_eq!(view.strides(), &[0, 1]);
    assert_eq!(host.strides(), &[3, 1]);
}
