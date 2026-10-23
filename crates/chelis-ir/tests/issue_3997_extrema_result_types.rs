//! [05-OP-40]: both direct extrema preserve the complete operand type.
use chelis_ir::dag::{Dag, DimInfo, ExtremaKind, RiscOp, TensorType};
use chelis_ir::{tier2, verify};
use chelis_numeric_core::prim::Prim;

fn ty(dims: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}

fn extrema_dag(op: RiscOp, left: TensorType, right: TensorType, out: TensorType) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("extrema");
    let a = dag.add_node(
        decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        left,
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        right,
        None,
    );
    let result = dag.add_node(decl, op, vec![a, b], out, None);
    dag.add_root(result);
    dag
}

#[test]
fn extrema_verifier_rejects_output_rank_dtype_and_extent_invention() {
    for op in [RiscOp::MinElem, RiscOp::MaxElem] {
        let input = ty(&[2, 3], Prim::Int32);
        for output in [
            ty(&[], Prim::Int32),
            ty(&[6], Prim::Int32),
            ty(&[2, 3, 1], Prim::Int32),
            ty(&[2, 4], Prim::Int32),
            ty(&[2, 3], Prim::F32),
        ] {
            let dag = extrema_dag(op.clone(), input.clone(), input.clone(), output.clone());
            let errors = verify::verify(&dag);
            assert!(
                errors
                    .iter()
                    .any(|error| error.contains("output must match")),
                "{op:?}: {input:?} -> {output:?} escaped: {errors:?}"
            );
        }
        let dag = extrema_dag(
            op.clone(),
            ty(&[], Prim::F32),
            ty(&[], Prim::F32),
            ty(&[1], Prim::F32),
        );
        assert!(
            verify::verify(&dag)
                .iter()
                .any(|error| error.contains("output must match")),
            "{op:?} invented a scalar's axis"
        );
    }
}

#[test]
fn extrema_verifier_rejects_bool_and_input_disagreement() {
    for op in [RiscOp::MinElem, RiscOp::MaxElem] {
        let boolean = ty(&[2], Prim::Bool);
        assert!(
            verify::verify(&extrema_dag(
                op.clone(),
                boolean.clone(),
                boolean.clone(),
                boolean
            ))
            .iter()
            .any(|error| error.contains("signed-integer or float dtype")),
            "{op:?} admitted bool"
        );
        let left = ty(&[2, 3], Prim::F32);
        for right in [
            ty(&[2, 3], Prim::F64),
            ty(&[3, 2], Prim::F32),
            ty(&[6], Prim::F32),
            ty(&[], Prim::F32),
        ] {
            assert!(
                !verify::verify(&extrema_dag(
                    op.clone(),
                    left.clone(),
                    right.clone(),
                    left.clone()
                ))
                .is_empty(),
                "{op:?}: {left:?} and {right:?} accepted"
            );
        }
    }
}

#[test]
fn extrema_verifier_accepts_physical_and_logical_axis_labels() {
    for op in [RiscOp::MinElem, RiscOp::MaxElem] {
        let left = TensorType {
            dims: vec![DimInfo::Named("rows".into(), None), DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let right = TensorType {
            dims: vec![
                DimInfo::Named("other".into(), None),
                DimInfo::Named("cols".into(), Some(3)),
            ],
            precision: Prim::F32,
        };
        assert!(verify::verify(&extrema_dag(op, left, right, ty(&[2, 3], Prim::F32))).is_empty());
    }
}

#[test]
fn paired_constructor_preserves_complete_operand_type_order_and_span() {
    for precision in [
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::F16,
        Prim::Bf16,
        Prim::F32,
        Prim::F64,
    ] {
        for dims in [
            vec![],
            vec![DimInfo::Lit(2)],
            vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            vec![DimInfo::Lit(2), DimInfo::Lit(0)],
            vec![
                DimInfo::Named("rows".into(), None),
                DimInfo::Named("cols".into(), Some(3)),
            ],
        ] {
            for kind in [ExtremaKind::Min, ExtremaKind::Max] {
                for span in [Some("selection.expr"), None] {
                    let mut dag = Dag::new();
                    let decl = dag.declare("extrema");
                    let input = TensorType {
                        dims: dims.clone(),
                        precision,
                    };
                    let a = dag.add_node(
                        decl,
                        RiscOp::Load {
                            name: "left".into(),
                        },
                        vec![],
                        input.clone(),
                        None,
                    );
                    let b = dag.add_node(
                        decl,
                        RiscOp::Load {
                            name: "right".into(),
                        },
                        vec![],
                        input.clone(),
                        None,
                    );
                    let result = tier2::lower_extrema(decl.into(), &mut dag, kind, a, b, span);
                    dag.add_root(result);
                    let node = dag.get(result).unwrap();
                    assert_eq!(node.output_type, input);
                    assert_eq!(
                        node.op,
                        match kind {
                            ExtremaKind::Min => RiscOp::MinElem,
                            ExtremaKind::Max => RiscOp::MaxElem,
                        }
                    );
                    assert_eq!(node.inputs, vec![a, b]);
                    assert_eq!(
                        node.span_id.as_deref(),
                        Some(span.unwrap_or(tier2::TIER2_SYNTH_MARKER))
                    );
                    assert_eq!(
                        dag.len(),
                        3,
                        "one direct selection, no surrogate arithmetic or casts"
                    );
                    assert!(verify::verify(&dag).is_empty());
                }
            }
        }
    }
}

#[test]
fn unknown_left_extent_cannot_hide_a_right_result_contradiction() {
    for op in [RiscOp::MinElem, RiscOp::MaxElem] {
        let left = TensorType {
            dims: vec![DimInfo::Named("*".into(), None)],
            precision: Prim::F32,
        };
        let errors = verify::verify(&extrema_dag(
            op.clone(),
            left,
            ty(&[3], Prim::F32),
            ty(&[2], Prim::F32),
        ));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("output must match")),
            "{op:?} missed pairwise contradiction: {errors:?}"
        );
    }
}
