//! Logical index projection uses the checked metadata authority.

use chelis_abi::metadata::{ByteCount, MetadataError, ShapeMetadata, StridedMetadata};
use chelis_vocab::RuntimeDType;

#[test]
fn offsets_materialize_permuted_gapped_and_broadcast_logical_order() {
    for dtype in RuntimeDType::ALL {
        let width = dtype.byte_width() as i64;
        for (shape, strides, span, expected) in [
            ([3, 2], [1, 3], 6, vec![0, 3, 1, 4, 2, 5]),
            ([2, 2], [4, 2], 7, vec![0, 2, 4, 6]),
            ([2, 3], [0, 1], 3, vec![0, 1, 2, 0, 1, 2]),
        ] {
            let metadata = StridedMetadata::new(
                &shape,
                &strides,
                dtype,
                ByteCount::from_declared(span * width).unwrap(),
            )
            .unwrap();
            let actual: Vec<_> = (0..metadata.elements().get())
                .map(|index| metadata.byte_offset(index).unwrap().get())
                .collect();
            assert_eq!(
                actual,
                expected
                    .iter()
                    .map(|index| (index * width) as usize)
                    .collect::<Vec<_>>()
            );
            assert!(matches!(
                metadata.byte_offset(-1),
                Err(MetadataError::Domain(_))
            ));
            assert!(matches!(
                metadata.byte_offset(metadata.elements().get()),
                Err(MetadataError::Domain(_))
            ));
        }
    }
}

#[test]
fn scalar_empty_wide_and_contiguous_offsets_keep_exact_domains() {
    let byte_capacity = |bytes| ByteCount::from_declared(bytes).unwrap();
    let scalar = StridedMetadata::new(&[], &[], RuntimeDType::F64, byte_capacity(8)).unwrap();
    assert_eq!(scalar.byte_offset(0).unwrap().get(), 0);
    assert!(matches!(
        scalar.byte_offset(1),
        Err(MetadataError::Domain(_))
    ));
    let empty = StridedMetadata::new(
        &[0, i64::MAX],
        &[i64::MAX; 2],
        RuntimeDType::F64,
        byte_capacity(0),
    )
    .unwrap();
    assert!(matches!(
        empty.byte_offset(0),
        Err(MetadataError::Domain(_))
    ));
    let broadcast =
        StridedMetadata::new(&[i64::MAX], &[0], RuntimeDType::I8, byte_capacity(1)).unwrap();
    assert_eq!(broadcast.byte_offset(i64::MAX - 1).unwrap().get(), 0);
    assert!(matches!(
        broadcast.byte_offset(i64::MAX),
        Err(MetadataError::Domain(_))
    ));
    let contiguous = ShapeMetadata::contiguous(&[2, 3], RuntimeDType::F32).unwrap();
    let strided =
        StridedMetadata::new(&[2, 3], &[3, 1], RuntimeDType::F32, byte_capacity(24)).unwrap();
    for index in 0..6 {
        assert_eq!(
            contiguous.byte_offset(index).unwrap(),
            strided.byte_offset(index).unwrap()
        );
    }
    assert!(matches!(
        contiguous.byte_offset(6),
        Err(MetadataError::Domain(_))
    ));
}
