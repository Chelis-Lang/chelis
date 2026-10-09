//! [04-SHAPE-1], [05-OP-31], [05-OP-33]: metadata is checked before access.
//! Include the private owner so these tests exercise its actual implementation
//! without making metadata a new public runtime API.
#[path = "../src/metadata.rs"]
mod metadata;

use chelis_vocab::RuntimeDType;
use metadata::{
    AffineProjection, ByteCount, ElementCount, IterationSpace, MatmulDimension, MatmulMetadata,
    MatmulPart, MetadataError, MovementMetadata, MovementOp, ProjectionPart, ReductionMetadata,
    ShapeMetadata, SparseMetadata, WindowMetadata,
};

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
    // A repeated unit axis stays inside its target; only the bijection rejects it.
    assert!(MovementMetadata::permuted(&shape(&[1, 1]), &[0, 0]).is_err());
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

#[test]
fn sparse_metadata_binds_indices_to_exact_hyperplane_and_elementwise_domains() {
    let shape = |s: &[i64], dtype| ShapeMetadata::contiguous(s, dtype).unwrap();
    for dtype in RuntimeDType::ALL {
        let base = shape(&[2, 3, 2], dtype);
        let indices = shape(&[2, 2], RuntimeDType::I8);
        let plan = SparseMetadata::new(&base, &indices, -2, false).unwrap();
        assert_eq!(plan.base().shape(), &[2, 3, 2]);
        assert_eq!(plan.base().extent_at(-1).unwrap(), 2);
        assert_eq!(plan.domain().shape(), &[2, 2, 2, 2]);
        assert_eq!(plan.domain().dtype(), dtype);
        let selected = [2, 0, 1, 2];
        let expected = [4, 5, 0, 1, 2, 3, 4, 5, 10, 11, 6, 7, 8, 9, 10, 11];
        for (linear, expected) in expected.into_iter().enumerate() {
            let slot = plan.index_slot(linear as i64).unwrap();
            assert_eq!(slot, (linear as i64 / 2) % 4);
            assert_eq!(
                plan.data_index(linear as i64, selected[slot as usize])
                    .unwrap(),
                expected
            );
        }
        for bad in [-1, 16] {
            assert!(plan.index_slot(bad).is_err());
            assert!(plan.data_index(bad, 0).is_err());
        }
        for bad in [-1, 3] {
            assert!(plan.data_index(0, bad).is_err());
        }
        assert!(SparseMetadata::new(&base, &indices, 3, false).is_err());
        assert!(SparseMetadata::new(&base, &indices, -4, false).is_err());
        let base = shape(&[2, 3], dtype);
        let plan = SparseMetadata::new(&base, &indices, 1, true).unwrap();
        assert_eq!(plan.domain().shape(), &[2, 2]);
        for (linear, (selected, expected)) in
            [(2, 2), (0, 0), (1, 4), (2, 5)].into_iter().enumerate()
        {
            assert_eq!(plan.index_slot(linear as i64).unwrap(), linear as i64);
            assert_eq!(plan.data_index(linear as i64, selected).unwrap(), expected);
        }
        assert!(SparseMetadata::new(&base, &shape(&[3, 2], RuntimeDType::I8), 1, true).is_err());
        assert!(SparseMetadata::new(&base, &shape(&[2], RuntimeDType::I8), 1, true).is_err());
    }
    let base = shape(&[0, 3, 2], RuntimeDType::I64);
    let indices = shape(&[4], RuntimeDType::I32);
    let empty = SparseMetadata::new(&base, &indices, 1, false).unwrap();
    assert_eq!(empty.domain().elements().get(), 0);
    assert!(empty.index_slot(0).is_err());
    assert!(empty.data_index(0, 0).is_err());
    let base = shape(&[2, 0, 3], RuntimeDType::I8);
    let no_valid_index = SparseMetadata::new(&base, &indices, 1, false).unwrap();
    assert_eq!(no_valid_index.domain().elements().get(), 24);
    assert!(no_valid_index.data_index(0, 0).is_err());
    let large = shape(&[i64::MAX, 0], RuntimeDType::I64);
    assert!(matches!(
        SparseMetadata::new(&large, &indices, 1, false),
        Err(MetadataError::Overflow(_))
    ));
    let base = shape(&[1, i64::MAX], RuntimeDType::I8);
    let scalar_index = shape(&[], RuntimeDType::I64);
    let large = SparseMetadata::new(&base, &scalar_index, 0, false).unwrap();
    assert_eq!(large.domain().elements().get(), i64::MAX);
    assert_eq!(large.index_slot(i64::MAX - 1).unwrap(), 0);
    assert_eq!(large.data_index(i64::MAX - 1, 0).unwrap(), i64::MAX - 1);
}

#[test]
fn reduction_metadata_binds_grouping_to_checked_input_and_result_domains() {
    for dtype in RuntimeDType::ALL {
        let plan = ReductionMetadata::new(&[2, 3], &[1], dtype).unwrap();
        assert_eq!(plan.result().dtype(), dtype);
        assert_eq!(plan.result().bytes().get(), 2 * dtype.byte_width() as i64);
        assert_eq!(
            plan.leaves().bytes(dtype).unwrap().get(),
            3 * dtype.byte_width() as i64
        );
        let large = ReductionMetadata::new(&[i64::MAX, 0], &[1], dtype);
        assert_eq!(large.is_err(), dtype.byte_width() > 1);
    }
    let plan = ReductionMetadata::new(&[2, 3, 2], &[-1, -3], RuntimeDType::I64).unwrap();
    assert_eq!(plan.result().shape(), &[3]);
    assert_eq!(plan.extent(-1).unwrap(), 3);
    assert!(plan.extent(-2).is_err());
    assert_eq!(plan.leaves().get(), 4);
    for (outer, indices) in [[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]]
        .iter()
        .enumerate()
    {
        for (leaf, expected) in indices.iter().enumerate() {
            assert_eq!(plan.index(outer as i64, leaf as i64).unwrap(), *expected);
        }
    }
    for (outer, leaf) in [(-1, 0), (3, 0), (0, -1), (0, 4)] {
        assert!(matches!(
            plan.index(outer, leaf),
            Err(MetadataError::Domain(_))
        ));
    }
    for axes in [
        vec![],
        vec![1, 1],
        vec![0, 1],
        vec![-4],
        vec![3],
        vec![2, -1],
    ] {
        assert!(matches!(
            ReductionMetadata::new(&[2, 3, 2], &axes, RuntimeDType::I64),
            Err(MetadataError::Domain(_))
        ));
    }
    assert!(matches!(
        ReductionMetadata::new(&[i64::MAX, 0], &[1], RuntimeDType::I64),
        Err(MetadataError::Overflow(_))
    ));
    assert!(matches!(
        ReductionMetadata::new(&[0, i64::MAX, i64::MAX, 0], &[3], RuntimeDType::I8),
        Err(MetadataError::Overflow(_))
    ));
    let empty =
        ReductionMetadata::new(&[0, i64::MAX, i64::MAX], &[2, 1], RuntimeDType::I64).unwrap();
    assert_eq!(empty.leaves().get(), 0);
    assert!(empty.index(0, 0).is_err());
    let huge = ReductionMetadata::new(&[i64::MAX], &[0], RuntimeDType::I64).unwrap();
    assert!(huge.leaves().bytes(RuntimeDType::I64).is_err());
    assert_eq!(huge.index(0, i64::MAX - 1).unwrap(), i64::MAX - 1);
}

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

#[test]
fn affine_metadata_checks_exact_extents_and_offsets_without_storage() {
    let extent = 9_007_199_254_740_995;
    let large = ShapeMetadata::contiguous(&[extent], RuntimeDType::I8).unwrap();
    assert_eq!(
        large.strided(&[2]).unwrap().shape(),
        &[4_503_599_627_370_498]
    );
    assert_eq!(
        large
            .affine_index_by(|_| (4_503_599_627_370_496, 1, 2))
            .unwrap(),
        9_007_199_254_740_993_usize
    );
    assert_eq!(large.padded(&[1], &[2]).unwrap().shape(), &[extent + 3]);
    assert_eq!(
        large.shrunk(&[extent], &[extent]).unwrap().elements().get(),
        0
    );
    assert!(matches!(
        large.padded(&[i64::MAX], &[0]),
        Err(MetadataError::Overflow(_))
    ));
    assert!(large.padded(&[-1], &[0]).is_err());
    assert!(large.shrunk(&[0], &[extent + 1]).is_err());
    assert!(large.strided(&[0]).is_err());
    assert!(large.strided(&[]).is_err());
    assert!(matches!(
        large.affine_index_by(|_| (i64::MAX, 1, 2)),
        Err(MetadataError::Overflow(_))
    ));
    assert!(large.affine_index_by(|_| (extent, 0, 1)).is_err());
    let empty = ShapeMetadata::contiguous(&[0, i64::MAX], RuntimeDType::I8).unwrap();
    assert!(empty.padded(&[0, 0], &[0, 1]).is_err());
    assert!(empty.strided(&[1, -1]).is_err());
    assert!(empty.affine_index_by(|_| (0, 0, 1)).is_err());
    let wide = ShapeMetadata::contiguous(&[1], RuntimeDType::F64).unwrap();
    assert!(wide.padded(&[0], &[i64::MAX / 8]).is_err());
}

// [05-OP-33] projection terms. Generated C indexes through a plan's
// projection with plain arithmetic, so every reachable position's term sum
// must equal the checked index, computed here from each operation's own
// coordinate rule rather than from the projection.

fn row_major(shape: &[i64]) -> Vec<Vec<i64>> {
    let count: i64 = shape.iter().product();
    (0..count)
        .map(|mut linear| {
            let mut coordinate = vec![0; shape.len()];
            for axis in (0..shape.len()).rev() {
                coordinate[axis] = linear % shape[axis];
                linear /= shape[axis];
            }
            coordinate
        })
        .collect()
}

fn flatten(shape: &[i64], coordinate: &[i64]) -> i64 {
    coordinate
        .iter()
        .zip(shape)
        .fold(0, |flat, (&c, &extent)| flat * extent + c)
}

/// The generated C expression for one projection: the outermost term needs
/// no modulus and the innermost has divisor one. Checked arithmetic makes an
/// overflow that C would silently commit fail the test.
fn c_projection(projection: &AffineProjection, position: i64) -> i64 {
    let terms = projection.terms();
    terms
        .iter()
        .enumerate()
        .fold(projection.base(), |sum, (k, term)| {
            let coordinate = match (k == 0, k + 1 == terms.len()) {
                (true, true) => position,
                (true, false) => position / term.divisor,
                (false, true) => position % term.modulus,
                (false, false) => position / term.divisor % term.modulus,
            };
            coordinate
                .checked_mul(term.scale)
                .and_then(|n| sum.checked_add(n))
                .expect("projection term overflows")
        })
}

/// The generated C movement loop: nested loops over each term's modulus
/// accumulate one scaled step per axis.
fn c_odometer(projection: &AffineProjection) -> Vec<i64> {
    let terms = projection.terms();
    let shape: Vec<i64> = terms.iter().map(|term| term.modulus).collect();
    row_major(&shape)
        .into_iter()
        .map(|coordinate| {
            coordinate
                .iter()
                .zip(terms)
                .fold(projection.base(), |offset, (&c, term)| {
                    offset + c * term.scale
                })
        })
        .collect()
}

fn require_domain_terms(projection: &AffineProjection, domain: &[i64]) {
    let terms = projection.terms();
    assert_eq!(terms.len(), domain.len());
    for (term, &extent) in terms.iter().zip(domain) {
        assert_eq!(term.modulus, extent);
    }
    if domain.contains(&0) {
        assert_eq!(projection.base(), 0);
        assert!(terms
            .iter()
            .all(|term| term.divisor == 1 && term.scale == 0));
    } else {
        for (k, term) in terms.iter().enumerate() {
            assert_eq!(term.divisor, domain[k + 1..].iter().product::<i64>());
        }
    }
}

fn shapes(rank: usize) -> Vec<Vec<i64>> {
    row_major(&vec![4; rank])
}

#[test]
fn movement_projection_terms_reproduce_every_checked_index() {
    let f = RuntimeDType::F32;
    let mut checked = 0;
    for rank in 0..=3 {
        for input in shapes(rank) {
            let source = ShapeMetadata::contiguous(&input, f).unwrap();
            // Padding counts its source; every other movement counts its result.
            type Oracle = Box<dyn Fn(&[i64]) -> i64>;
            let mut plans: Vec<(MovementMetadata, bool, Oracle)> = Vec::new();
            let rotated: Vec<i64> = (0..rank as i64).map(|a| (a + 1) % rank as i64).collect();
            let axes = rotated.clone();
            let shape = input.clone();
            plans.push((
                MovementMetadata::permuted(&source, &rotated).unwrap(),
                false,
                Box::new(move |c: &[i64]| {
                    let mut s = vec![0; c.len()];
                    for (out, &axis) in axes.iter().enumerate() {
                        s[axis as usize] = c[out];
                    }
                    flatten(&shape, &s)
                }),
            ));
            for axis in 0..=rank {
                let shape = input.clone();
                plans.push((
                    MovementMetadata::expanded(&source, axis as i64, 3, true).unwrap(),
                    false,
                    Box::new(move |c: &[i64]| {
                        let mut s = c.to_vec();
                        s.remove(axis);
                        flatten(&shape, &s)
                    }),
                ));
                if axis < rank && input[axis] == 1 {
                    let shape = input.clone();
                    plans.push((
                        MovementMetadata::expanded(&source, axis as i64, 3, false).unwrap(),
                        false,
                        Box::new(move |c: &[i64]| {
                            let mut s = c.to_vec();
                            s[axis] = 0;
                            flatten(&shape, &s)
                        }),
                    ));
                }
            }
            let before: Vec<i64> = (0..rank as i64).map(|a| a % 2).collect();
            let after: Vec<i64> = (0..rank as i64).map(|a| (a + 1) % 3).collect();
            let pad = MovementMetadata::affine(&source, &before, &after, MovementOp::Pad).unwrap();
            let padded = pad.result().shape().to_vec();
            let low = before.clone();
            plans.push((
                pad,
                true,
                Box::new(move |c: &[i64]| {
                    let s: Vec<i64> = c.iter().zip(&low).map(|(c, b)| c + b).collect();
                    flatten(&padded, &s)
                }),
            ));
            let start: Vec<i64> = input.iter().map(|&n| i64::from(n >= 2)).collect();
            let end: Vec<i64> = input
                .iter()
                .map(|&n| if n >= 3 { n - 1 } else { n })
                .collect();
            let shape = input.clone();
            let first = start.clone();
            plans.push((
                MovementMetadata::affine(&source, &start, &end, MovementOp::Shrink).unwrap(),
                false,
                Box::new(move |c: &[i64]| {
                    let s: Vec<i64> = c.iter().zip(&first).map(|(c, a)| c + a).collect();
                    flatten(&shape, &s)
                }),
            ));
            let steps: Vec<i64> = (0..rank as i64).map(|a| a % 3 + 1).collect();
            let shape = input.clone();
            let step = steps.clone();
            plans.push((
                MovementMetadata::affine(&source, &steps, &[], MovementOp::Stride).unwrap(),
                false,
                Box::new(move |c: &[i64]| {
                    let s: Vec<i64> = c.iter().zip(&step).map(|(c, k)| c * k).collect();
                    flatten(&shape, &s)
                }),
            ));
            for (plan, source_domain, oracle) in plans {
                let domain = if source_domain {
                    plan.input().shape().to_vec()
                } else {
                    plan.result().shape().to_vec()
                };
                let projection = plan.projection();
                require_domain_terms(projection, &domain);
                let odometer = c_odometer(projection);
                assert_eq!(odometer.len() as i64, plan.count().get());
                for (position, coordinate) in row_major(&domain).iter().enumerate() {
                    let position = position as i64;
                    let expected = oracle(coordinate);
                    assert_eq!(plan.index(position).unwrap(), expected, "{input:?}");
                    assert_eq!(c_projection(projection, position), expected, "{input:?}");
                    assert_eq!(odometer[position as usize], expected, "{input:?}");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 3_000, "{checked}");
}

#[test]
fn reduction_and_window_projection_terms_reproduce_every_checked_index() {
    let f = RuntimeDType::F32;
    let mut checked = 0;
    for rank in 1..=3 {
        for input in shapes(rank) {
            // Every strictly descending nonempty axis set.
            for mask in 1..(1_u32 << rank) {
                let axes: Vec<i64> = (0..rank as i64)
                    .rev()
                    .filter(|&a| mask & (1 << a) != 0)
                    .collect();
                let plan = ReductionMetadata::new(&input, &axes, f).unwrap();
                let selected = |axis: usize| mask & (1 << axis) != 0;
                let group_shape: Vec<i64> = (0..rank)
                    .filter(|&a| !selected(a))
                    .map(|a| input[a])
                    .collect();
                let leaf_shape: Vec<i64> = (0..rank)
                    .filter(|&a| selected(a))
                    .map(|a| input[a])
                    .collect();
                let reachable = !input.contains(&0);
                for (part, domain) in [
                    (ProjectionPart::Group, &group_shape),
                    (ProjectionPart::Leaf, &leaf_shape),
                ] {
                    let projection = plan.projection(part);
                    if reachable {
                        require_domain_terms(projection, domain);
                    } else {
                        assert_eq!(projection.base(), 0);
                        assert!(projection
                            .terms()
                            .iter()
                            .all(|t| t.divisor == 1 && t.scale == 0));
                    }
                }
                if !reachable {
                    assert_eq!(plan.leaves().get() * plan.result().elements().get(), 0);
                    continue;
                }
                for (outer, group) in row_major(&group_shape).iter().enumerate() {
                    for (leaf, within) in row_major(&leaf_shape).iter().enumerate() {
                        let (mut g, mut l) = (group.iter(), within.iter());
                        let source: Vec<i64> = (0..rank)
                            .map(|a| *if selected(a) { l.next() } else { g.next() }.unwrap())
                            .collect();
                        let expected = flatten(&input, &source);
                        let (outer, leaf) = (outer as i64, leaf as i64);
                        assert_eq!(plan.index(outer, leaf).unwrap(), expected);
                        let c = c_projection(plan.projection(ProjectionPart::Group), outer)
                            + c_projection(plan.projection(ProjectionPart::Leaf), leaf);
                        assert_eq!(c, expected, "{input:?} {axes:?}");
                        checked += 1;
                    }
                }
            }
            // Every window and step of extent one to two on the trailing axes.
            for count in 1..=rank {
                let leading = rank - count;
                for window in row_major(&vec![2; count]) {
                    let window: Vec<i64> = window.iter().map(|w| w + 1).collect();
                    let steps: Vec<i64> = window.iter().rev().cloned().collect();
                    let source = ShapeMetadata::contiguous(&input, f).unwrap();
                    let Ok(plan) = WindowMetadata::new(&source, &window, &steps) else {
                        assert!(input[leading..].iter().zip(&window).any(|(n, w)| w > n));
                        continue;
                    };
                    let result = plan.result().shape().to_vec();
                    for part in [ProjectionPart::Group, ProjectionPart::Leaf] {
                        let domain = if part == ProjectionPart::Group {
                            &result
                        } else {
                            &window
                        };
                        if plan.result().elements().get() != 0 {
                            require_domain_terms(plan.projection(part), domain);
                        }
                    }
                    for (outer, group) in row_major(&result).iter().enumerate() {
                        for (leaf, offset) in row_major(&window).iter().enumerate() {
                            let source: Vec<i64> = (0..rank)
                                .map(|a| {
                                    if a < leading {
                                        group[a]
                                    } else {
                                        group[a] * steps[a - leading] + offset[a - leading]
                                    }
                                })
                                .collect();
                            let expected = flatten(&input, &source);
                            let (outer, leaf) = (outer as i64, leaf as i64);
                            assert_eq!(plan.index(outer, leaf).unwrap(), expected);
                            let c = c_projection(plan.projection(ProjectionPart::Group), outer)
                                + c_projection(plan.projection(ProjectionPart::Leaf), leaf);
                            assert_eq!(c, expected, "{input:?} {window:?}");
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 3_000, "{checked}");
}

#[test]
fn projections_of_extreme_valid_plans_stay_representable() {
    let i8 = RuntimeDType::I8;
    let huge = ShapeMetadata::contiguous(&[i64::MAX], i8).unwrap();
    let stride = MovementMetadata::affine(&huge, &[2], &[], MovementOp::Stride).unwrap();
    let last = stride.count().get() - 1;
    assert_eq!(c_projection(stride.projection(), last), i64::MAX - 1);
    // A step larger than its axis leaves one coordinate; its product with the
    // axis stride is unobservable and must not be evaluated.
    let wide = ShapeMetadata::contiguous(&[3, i64::MAX / 4], i8).unwrap();
    let sparse = MovementMetadata::affine(&wide, &[i64::MAX, 1], &[], MovementOp::Stride).unwrap();
    assert_eq!(sparse.result().shape(), &[1, i64::MAX / 4]);
    assert_eq!(sparse.projection().terms()[0].scale, 0);
    let whole = WindowMetadata::new(&huge, &[i64::MAX], &[1]).unwrap();
    assert_eq!(
        c_projection(whole.projection(ProjectionPart::Leaf), i64::MAX - 1),
        i64::MAX - 1
    );
    let reduction = ReductionMetadata::new(&[i64::MAX], &[0], RuntimeDType::I64).unwrap();
    assert_eq!(
        c_projection(reduction.projection(ProjectionPart::Leaf), i64::MAX - 1),
        i64::MAX - 1
    );
    // Empty domains never form a stride product.
    let empty =
        ReductionMetadata::new(&[0, i64::MAX, i64::MAX], &[2, 1], RuntimeDType::I64).unwrap();
    for part in [ProjectionPart::Group, ProjectionPart::Leaf] {
        assert!(empty
            .projection(part)
            .terms()
            .iter()
            .all(|t| t.divisor == 1 && t.scale == 0));
    }
    let empty_window = WindowMetadata::new(
        &ShapeMetadata::contiguous(&[0, i64::MAX], i8).unwrap(),
        &[i64::MAX],
        &[1],
    )
    .unwrap();
    assert_eq!(
        empty_window.projection(ProjectionPart::Leaf).terms()[0].modulus,
        i64::MAX
    );
}
