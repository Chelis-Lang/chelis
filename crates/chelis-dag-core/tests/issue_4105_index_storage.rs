//! chelis#4105: the verifier owns two storage invariants that backends rely
//! on when they size and copy element buffers.
//!
//! - [05-OP-15]: `argmax_reduce`/`argmin_reduce` return i64 indices with the
//!   reduced axis removed, whatever the operand dtype.
//! - A shape-only movement (`reshape`, `permute`, `expand`, `shrink`,
//!   `stride`; `pad` already had the rule) preserves the element dtype of its operand.
//!
//! A graph that breaks either one must be rejected before emission, where a
//! mislabeled index result would be stored at the operand width and then
//! copied by byte count into a wider buffer.

use chelis_dag_core::dag::{Dag, DeclId, DimInfo, RiscOp, RtDim, TensorType};
use chelis_dag_core::verify;
use chelis_numeric_core::prim::Prim;

fn tensor(dims: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}

fn symbolic(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Named("sequence".into(), None),
        ],
        precision,
    }
}

fn single_op(input: TensorType, op: RiscOp, output: TensorType) -> Dag {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = dag.add_node(
        DeclId(0),
        RiscOp::Load { name: "x".into() },
        vec![],
        input,
        Some("x".into()),
    );
    let out = dag.add_node(DeclId(0), op, vec![x], output, None);
    dag.add_root(out);
    dag
}

fn errors_mentioning(dag: &Dag, needle: &str) -> Vec<String> {
    let errors = verify::verify(dag);
    assert_eq!(errors, verify::verify(dag), "diagnostics are deterministic");
    errors
        .into_iter()
        .filter(|error| error.contains(needle))
        .collect()
}

#[test]
fn index_reductions_admit_only_i64_results_for_every_operand_dtype() {
    for operand in [Prim::Int32, Prim::Int64, Prim::F32, Prim::F64, Prim::Int8] {
        for op in [RiscOp::Argmax { axis: 1 }, RiscOp::Argmin { axis: 1 }] {
            for (input, output_dims) in [
                (tensor(&[2, 3], operand), tensor(&[2], Prim::Int64)),
                (
                    symbolic(operand),
                    TensorType {
                        dims: vec![DimInfo::Named("batch".into(), None)],
                        precision: Prim::Int64,
                    },
                ),
            ] {
                let accepted = single_op(input.clone(), op.clone(), output_dims.clone());
                assert_eq!(
                    verify::verify(&accepted),
                    Vec::<String>::new(),
                    "{op:?} over {operand:?} with an i64 result must verify"
                );
                for wrong in [Prim::Int32, Prim::F32, Prim::Int8, Prim::F64] {
                    let mut output = output_dims.clone();
                    output.precision = wrong;
                    let rejected = single_op(input.clone(), op.clone(), output);
                    let errors = errors_mentioning(&rejected, "index reduction");
                    assert_eq!(
                        errors,
                        vec![format!(
                            "index reduction at node 1 requires i64 output, got {wrong:?}"
                        )],
                        "{op:?} over {operand:?} labelled {wrong:?} must be rejected"
                    );
                }
            }
        }
    }
}

fn movement_cases(precision: Prim) -> Vec<(TensorType, RiscOp, TensorType)> {
    vec![
        (
            tensor(&[2, 3], precision),
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(6)],
            },
            tensor(&[6], precision),
        ),
        (
            tensor(&[2, 3], precision),
            RiscOp::Permute { axes: vec![1, 0] },
            tensor(&[3, 2], precision),
        ),
        (
            tensor(&[1, 3], precision),
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Lit(2),
            },
            tensor(&[2, 3], precision),
        ),
        (
            tensor(&[4], precision),
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::Lit(3))],
            },
            tensor(&[2], precision),
        ),
        (
            tensor(&[4], precision),
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2)],
            },
            tensor(&[2], precision),
        ),
    ]
}

#[test]
fn shape_only_movements_preserve_the_operand_element_dtype() {
    for (operand, wrong) in [
        (Prim::Int32, Prim::Int64),
        (Prim::Int64, Prim::Int32),
        (Prim::F32, Prim::F64),
        (Prim::F64, Prim::Int64),
    ] {
        for (input, op, output) in movement_cases(operand) {
            let accepted = single_op(input.clone(), op.clone(), output.clone());
            assert_eq!(
                verify::verify(&accepted),
                Vec::<String>::new(),
                "{op:?} preserving {operand:?} must verify"
            );
            let mut relabelled = output;
            relabelled.precision = wrong;
            let rejected = single_op(input, op.clone(), relabelled);
            let errors = errors_mentioning(&rejected, "precision");
            assert!(
                errors.iter().any(|error| error
                    == &format!(
                        "movement at node 1: output precision {wrong:?} != input precision {operand:?}"
                    )),
                "{op:?} relabelling {operand:?} as {wrong:?} must be rejected, got {errors:?}"
            );
        }
    }
}

/// The issue's shape: an i32 index reduction followed by a reshape into an
/// i64 result. Either mislabel alone is caught, so the byte-count copy from
/// an i32 buffer into an i64 buffer cannot reach a backend.
#[test]
fn the_reported_i32_argmax_then_i64_reshape_chain_is_rejected() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = dag.add_node(
        DeclId(0),
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor(&[2, 3], Prim::Int32),
        Some("x".into()),
    );
    let index = dag.add_node(
        DeclId(0),
        RiscOp::Argmax { axis: 1 },
        vec![x],
        tensor(&[2], Prim::Int32),
        None,
    );
    let out = dag.add_node(
        DeclId(0),
        RiscOp::Reshape {
            new_shape: vec![RtDim::Lit(2)],
        },
        vec![index],
        tensor(&[2], Prim::Int64),
        None,
    );
    dag.add_root(out);
    let errors = verify::verify(&dag);
    assert!(
        errors.contains(&"index reduction at node 1 requires i64 output, got Int32".to_string()),
        "{errors:?}"
    );
    assert!(
        errors.contains(
            &"movement at node 2: output precision Int64 != input precision Int32".to_string()
        ),
        "{errors:?}"
    );
}
