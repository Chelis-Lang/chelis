//! Transferred unchanged contracts from Phase1 main49ac08e:
//! [04-SHAPE-1], [05-OP-31/33], metadata validation precedes storage/access.
//! These plans own checked geometry, never tensor payload storage.
use chelis_abi::metadata::{
    ByteCount, MatmulDimension, MatmulMetadata, MatmulPart, MetadataError, MovementMetadata,
    MovementOp, ShapeMetadata, WindowMetadata,
};
use chelis_vocab::RuntimeDType;

#[test]
fn movement_plans_project_checked_domains_without_coordinate_scratch() {
    let shape = |s: &[i64], dtype| ShapeMetadata::contiguous(s, dtype).unwrap();
    for dtype in RuntimeDType::ALL {
        let source = shape(&[2, 3], dtype);
        let permute = MovementMetadata::permuted(&source, &[-1, 0]).unwrap();
        assert_eq!(permute.result().shape(), &[3, 2]);
        for i in 0..6 {
            assert_eq!(permute.index(i).unwrap(), (i % 2) * 3 + i / 2);
        }
        let insert = MovementMetadata::expanded(&source, 1, 4, true).unwrap();
        let expand = MovementMetadata::expanded(&shape(&[2, 1, 3], dtype), 1, 4, false).unwrap();
        for plan in [&insert, &expand] {
            assert_eq!(plan.result().shape(), &[2, 4, 3]);
            assert_eq!(plan.count().get(), 24);
            for i in 0..24 {
                assert_eq!(plan.index(i).unwrap(), i / 12 * 3 + i % 3);
            }
        }
        let pad = MovementMetadata::affine(&source, &[1, 2], &[0, 1], MovementOp::Pad).unwrap();
        assert_eq!(pad.result().shape(), &[3, 6]);
        assert_eq!(pad.count().get(), 6);
        for i in 0..6 {
            assert_eq!(pad.index(i).unwrap(), (i / 3 + 1) * 6 + i % 3 + 2);
        }
        let larger = shape(&[3, 5], dtype);
        let shrink =
            MovementMetadata::affine(&larger, &[1, 2], &[3, 5], MovementOp::Shrink).unwrap();
        let stride = MovementMetadata::affine(&larger, &[2, 2], &[], MovementOp::Stride).unwrap();
        for plan in [&shrink, &stride] {
            assert_eq!(plan.result().shape(), &[2, 3]);
        }
        for i in 0..6 {
            assert_eq!(shrink.index(i).unwrap(), (i / 3 + 1) * 5 + i % 3 + 2);
            assert_eq!(stride.index(i).unwrap(), i / 3 * 2 * 5 + i % 3 * 2);
        }
        for plan in [permute, insert, expand, pad, shrink, stride] {
            assert_eq!(plan.input().dtype(), dtype);
            assert_eq!(plan.result().dtype(), dtype);
            for i in [-1, plan.count().get()] {
                assert!(matches!(plan.index(i), Err(MetadataError::Domain(_))));
            }
        }
    }
}

#[test]
fn movement_plans_reject_bad_geometry_and_preserve_rank_zero_empty_and_int64() {
    let shape = |s: &[i64]| ShapeMetadata::contiguous(s, RuntimeDType::I8).unwrap();
    let input = shape(&[2, 3]);
    for axes in [vec![0], vec![0, 0], vec![0, 2], vec![-3, 1]] {
        assert!(MovementMetadata::permuted(&input, &axes).is_err());
    }
    for (axis, size, insert) in [(0, 3, false), (2, 3, false), (3, 3, true), (0, -1, true)] {
        assert!(MovementMetadata::expanded(&input, axis, size, insert).is_err());
    }
    for (first, second, op) in [
        (vec![-1, 0], vec![0, 0], MovementOp::Pad),
        (vec![0], vec![0], MovementOp::Pad),
        (vec![i64::MAX, 0], vec![0, 0], MovementOp::Pad),
        (vec![0, 0], vec![3, 3], MovementOp::Shrink),
        (vec![0, 0], vec![1, -1], MovementOp::Shrink),
        (vec![0, 1], vec![], MovementOp::Stride),
        (vec![1], vec![], MovementOp::Stride),
    ] {
        assert!(MovementMetadata::affine(&input, &first, &second, op).is_err());
    }
    let scalar = shape(&[]);
    assert_eq!(
        MovementMetadata::permuted(&scalar, &[])
            .unwrap()
            .index(0)
            .unwrap(),
        0
    );
    for op in [MovementOp::Pad, MovementOp::Shrink, MovementOp::Stride] {
        assert_eq!(
            MovementMetadata::affine(&scalar, &[], &[], op)
                .unwrap()
                .index(0)
                .unwrap(),
            0
        );
    }
    let inserted = MovementMetadata::expanded(&scalar, 0, 2, true).unwrap();
    assert_eq!(inserted.index(1).unwrap(), 0);
    let high_rank = shape(&[1; 9]);
    assert_eq!(
        MovementMetadata::permuted(&high_rank, &[8, 7, 6, 5, 4, 3, 2, 1, 0])
            .unwrap()
            .index(0)
            .unwrap(),
        0
    );
    let empty = MovementMetadata::affine(&shape(&[0]), &[1], &[1], MovementOp::Pad).unwrap();
    assert_eq!(empty.result().shape(), &[2]);
    assert_eq!(empty.count().get(), 0);
    assert!(empty.index(0).is_err());
    let huge = shape(&[i64::MAX]);
    let stride = MovementMetadata::affine(&huge, &[2], &[], MovementOp::Stride).unwrap();
    assert_eq!(
        stride.index(stride.count().get() - 1).unwrap(),
        i64::MAX - 1
    );
    assert!(MovementMetadata::expanded(&shape(&[2]), 0, i64::MAX, true).is_err());
}

#[test]
fn window_metadata_binds_valid_padding_and_row_major_source_indices() {
    let shape = |s: &[i64], dtype| ShapeMetadata::contiguous(s, dtype).unwrap();
    for dtype in RuntimeDType::ALL {
        let input = shape(&[2, 5, 6], dtype);
        let plan = WindowMetadata::new(&input, &[2, 3], &[2, 2]).unwrap();
        assert_eq!(plan.input().shape(), &[2, 5, 6]);
        assert_eq!(plan.result().shape(), &[2, 2, 2]);
        assert_eq!(plan.result().dtype(), dtype);
        assert_eq!(plan.count().get(), 6);
        for group in 0..8 {
            for leaf in 0..6 {
                let expected = (group / 4) * 30
                    + ((group % 4) / 2 * 2 + leaf / 3) * 6
                    + (group % 2 * 2 + leaf % 3);
                assert_eq!(plan.index(group, leaf).unwrap(), expected);
            }
        }
        for (group, leaf) in [(-1, 0), (8, 0), (0, -1), (0, 6)] {
            assert!(matches!(
                plan.index(group, leaf),
                Err(MetadataError::Domain(_))
            ));
        }
        drop(input);
        assert_eq!(plan.index(7, 5).unwrap(), 52);
    }
    let f = RuntimeDType::F32;
    for (input, windows, steps) in [
        (&[4][..], &[][..], &[][..]),
        (&[4], &[2], &[]),
        (&[4], &[1, 1], &[1, 1]),
        (&[4], &[0], &[1]),
        (&[4], &[-1], &[1]),
        (&[4], &[2], &[0]),
        (&[4], &[2], &[-1]),
        (&[4], &[5], &[2]),
        (&[0], &[1], &[1]),
    ] {
        assert!(matches!(
            WindowMetadata::new(&shape(input, f), windows, steps),
            Err(MetadataError::Domain(_))
        ));
    }
    let empty = WindowMetadata::new(&shape(&[0, i64::MAX], f), &[i64::MAX], &[1]).unwrap();
    assert_eq!(empty.result().shape(), &[0, 1]);
    assert_eq!(empty.count().get(), 0);
    assert!(empty.index(0, 0).is_err());
    let huge = WindowMetadata::new(&shape(&[i64::MAX], RuntimeDType::I8), &[2], &[2]).unwrap();
    assert_eq!(huge.result().shape(), &[i64::MAX / 2]);
    assert_eq!(huge.index(i64::MAX / 2 - 1, 1).unwrap(), i64::MAX - 2);
    let whole =
        WindowMetadata::new(&shape(&[i64::MAX], RuntimeDType::I8), &[i64::MAX], &[1]).unwrap();
    assert_eq!(whole.count().get(), i64::MAX);
    assert_eq!(whole.index(0, i64::MAX - 1).unwrap(), i64::MAX - 1);
}

#[test]
fn matmul_metadata_binds_matrix_spans_and_vendor_projection_without_storage() {
    let shape = |s: &[i64], dtype| ShapeMetadata::contiguous(s, dtype).unwrap();
    for dtype in RuntimeDType::ALL {
        let a = shape(&[2, 3, 4], dtype);
        let b = shape(&[2, 4, 5], dtype);
        let p = MatmulMetadata::new(&a, &b, dtype).unwrap();
        assert_eq!(p.result().shape(), &[2, 3, 5]);
        assert_eq!(p.result().dtype(), dtype);
        assert_eq!(p.dimension(MatmulDimension::Rows), 3);
        assert_eq!(p.dimension(MatmulDimension::Columns), 5);
        assert_eq!(p.dimension(MatmulDimension::Reduction), 4);
        assert_eq!(p.batches().get(), 2);
        for (part, n) in [
            (MatmulPart::Left, 12),
            (MatmulPart::Right, 20),
            (MatmulPart::Result, 15),
        ] {
            assert_eq!(p.matrix_count(part).get(), n);
            for batch in 0..2 {
                for element in 0..n {
                    assert_eq!(p.index(part, batch, element).unwrap(), batch * n + element);
                }
            }
            assert!(p.index(part, -1, 0).is_err());
            assert!(p.index(part, 2, 0).is_err());
            assert!(p.index(part, 0, -1).is_err());
            assert!(p.index(part, 0, n).is_err());
            p.matrix_count(part)
                .bytes(RuntimeDType::F64)
                .unwrap()
                .allocation()
                .unwrap();
        }
        p.check_vendor(i64::from(i32::MAX)).unwrap();
        assert!(matches!(p.check_vendor(4), Err(MetadataError::Overflow(_))));
        assert!(matches!(p.check_vendor(0), Err(MetadataError::Domain(_))));
        drop(a);
        drop(b);
        assert_eq!(p.index(MatmulPart::Right, 1, 19).unwrap(), 39);
    }
    let f = RuntimeDType::F32;
    for (a, b) in [
        (&[3][..], &[3, 4][..]),
        (&[2, 3, 4], &[4, 5]),
        (&[2, 3, 4], &[3, 4, 5]),
        (&[2, 3, 4], &[2, 6, 5]),
    ] {
        assert!(matches!(
            MatmulMetadata::new(&shape(a, f), &shape(b, f), f),
            Err(MetadataError::Domain(_))
        ));
    }
    assert!(
        MatmulMetadata::new(&shape(&[3, 4], f), &shape(&[4, 5], RuntimeDType::F64), f).is_err()
    );
    let empty =
        MatmulMetadata::new(&shape(&[0, i64::MAX, 0], f), &shape(&[0, 0, 1], f), f).unwrap();
    assert_eq!(empty.result().shape(), &[0, i64::MAX, 1]);
    assert_eq!(empty.batches().get(), 0);
    assert_eq!(empty.matrix_count(MatmulPart::Result).get(), 0);
    empty.check_vendor(i64::from(i32::MAX)).unwrap();
    assert!(empty.index(MatmulPart::Result, 0, 0).is_err());
    let zero_k = MatmulMetadata::new(&shape(&[3, 0], f), &shape(&[0, 5], f), f).unwrap();
    assert_eq!(zero_k.batches().get(), 1);
    assert_eq!(zero_k.matrix_count(MatmulPart::Result).get(), 15);
    assert_eq!(zero_k.matrix_count(MatmulPart::Left).get(), 0);
    zero_k.check_vendor(1).unwrap();
    let huge = i64::from(i32::MAX) + 1;
    let large = MatmulMetadata::new(
        &shape(&[huge, 1], RuntimeDType::I8),
        &shape(&[1, 1], RuntimeDType::I8),
        RuntimeDType::I8,
    )
    .unwrap();
    assert_eq!(
        large.index(MatmulPart::Result, 0, huge - 1).unwrap(),
        huge - 1
    );
    assert!(large.check_vendor(i64::from(i32::MAX)).is_err());
    large.check_vendor(i64::MAX).unwrap();
    // Operand storage may fit while f32 conversion scratch does not. Exercise
    // that boundary virtually without asking the allocator for huge buffers.
    for (extent, fits) in [(i64::MAX / 4, true), (i64::MAX / 4 + 1, false)] {
        let p = MatmulMetadata::new(
            &shape(&[extent, 1], RuntimeDType::F16),
            &shape(&[1, 1], RuntimeDType::F16),
            RuntimeDType::F16,
        )
        .unwrap();
        let bytes = p
            .matrix_count(MatmulPart::Left)
            .bytes(RuntimeDType::F32)
            .and_then(ByteCount::allocation);
        assert_eq!(bytes.is_ok(), fits);
    }
    let x = shape(&[1_i64 << 32, 1], RuntimeDType::I8);
    let y = shape(&[1, 1_i64 << 32], RuntimeDType::I8);
    assert!(matches!(
        MatmulMetadata::new(&x, &y, RuntimeDType::I8),
        Err(MetadataError::Overflow(_))
    ));
}
