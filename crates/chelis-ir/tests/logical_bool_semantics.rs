//! Authoritative IR oracle for chelis#1284, chelis#630, and chelis#666.

use chelis_ir::dag::{
    ComparisonKind, Dag, DimInfo, ExtentClaim, ExtentWitnessSite, LogicalKind, RiscOp, RtAxis,
    RtDim, TensorType,
};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::fuse::fuse;
use chelis_ir::grad::{AdError, AdRejectionReason, grad_dag_checked};
use chelis_ir::{tier2, verify};
use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

fn ty(dims: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}

fn value(precision: Prim, shape: Vec<usize>, values: Vec<f64>) -> TensorValue {
    TensorValue::from_storage(
        shape,
        finalize_tensor("logical oracle", precision, RawTensor::Float(values)).unwrap(),
    )
}

fn bool_value(shape: Vec<usize>, values: Vec<i64>) -> TensorValue {
    TensorValue::finalize_from_wide_int("logical oracle", Prim::Bool, shape, values).unwrap()
}

/// Evaluate `dag` with each named input a rank-1 tensor of ones of its
/// length. chelis#2642: a same-shape operation whose operand extents the
/// graph does not prove equal verifies, and checks their agreement when it
/// runs (runtime_extents.md C2.3), so disagreeing lengths fail here.
fn evaluate_lengths(dag: &Dag, precision: Prim, inputs: &[(&str, usize)]) -> Result<(), String> {
    let inputs = inputs
        .iter()
        .map(|(name, length)| {
            let ones = if precision == Prim::Bool {
                bool_value(vec![*length], vec![1; *length])
            } else {
                value(precision, vec![*length], vec![1.0; *length])
            };
            ((*name).to_string(), ones)
        })
        .collect::<UnordMap<_, _>>();
    eval_tensor(dag, &inputs).map(|_| ())
}

/// `dag` verifies, runs with agreeing lengths, and fails with disagreeing
/// ones (chelis#2642).
fn verifies_and_checks_agreement_when_run(
    dag: &Dag,
    precision: Prim,
    agreeing: &[(&str, usize)],
    disagreeing: &[(&str, usize)],
    why: &str,
) {
    assert_eq!(verify::verify(dag), Vec::<String>::new(), "{why}");
    evaluate_lengths(dag, precision, agreeing)
        .unwrap_or_else(|error| panic!("{why}: agreeing lengths: {error}"));
    if evaluate_lengths(dag, precision, disagreeing).is_ok() {
        panic!("{why}: disagreeing lengths must fail when the graph runs");
    }
}

fn scalar_bits(value: &TensorValue, index: usize) -> serde_json::Value {
    serde_json::to_value(value.storage().scalar_at(index)).unwrap()
}

#[test]
fn tier2_preserves_every_comparison_and_logical_identity() {
    type ComparisonLowerer = fn(
        chelis_ir::dag::Owner,
        &mut Dag,
        chelis_ir::dag::NodeId,
        chelis_ir::dag::NodeId,
        &TensorType,
        Option<&str>,
    ) -> chelis_ir::dag::NodeId;
    let comparisons: [(ComparisonKind, ComparisonLowerer); 7] = [
        (ComparisonKind::CmpLt, tier2::lower_cmplt),
        (ComparisonKind::Lt, tier2::lower_lt),
        (ComparisonKind::Eq, tier2::lower_eq),
        (ComparisonKind::Neq, tier2::lower_neq),
        (ComparisonKind::Gt, tier2::lower_gt),
        (ComparisonKind::Gte, tier2::lower_gte),
        (ComparisonKind::Lte, tier2::lower_lte),
    ];
    for (kind, lower) in comparisons {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let operand_ty = ty(&[2], Prim::F32);
        let left = dag.add_node(
            decl,
            RiscOp::Load {
                name: "left".into(),
            },
            vec![],
            operand_ty.clone(),
            None,
        );
        let right = dag.add_node(
            decl,
            RiscOp::Load {
                name: "right".into(),
            },
            vec![],
            operand_ty.clone(),
            None,
        );
        let result = lower(
            decl.into(),
            &mut dag,
            left,
            right,
            &operand_ty,
            Some("comparison"),
        );
        assert_eq!(
            dag.get(result).unwrap().op,
            RiscOp::Compare(kind),
            "{kind:?} lost its identity"
        );
        assert_eq!(dag.get(result).unwrap().inputs, vec![left, right]);
        assert_eq!(dag.len(), 3, "{kind:?} introduced a decomposition");
        assert!(verify::verify(&dag).is_empty());
    }

    for (kind, arity) in [
        (LogicalKind::And, 2usize),
        (LogicalKind::Or, 2),
        (LogicalKind::Not, 1),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let bool_ty = ty(&[2], Prim::Bool);
        let left = dag.add_node(
            decl,
            RiscOp::Load {
                name: "left".into(),
            },
            vec![],
            bool_ty.clone(),
            None,
        );
        let right = (kind != LogicalKind::Not).then(|| {
            dag.add_node(
                decl,
                RiscOp::Load {
                    name: "right".into(),
                },
                vec![],
                bool_ty.clone(),
                None,
            )
        });
        let result = match kind {
            LogicalKind::And => {
                tier2::lower_and(decl.into(), &mut dag, left, right.unwrap(), &bool_ty, None)
            }
            LogicalKind::Or => {
                tier2::lower_or(decl.into(), &mut dag, left, right.unwrap(), &bool_ty, None)
            }
            LogicalKind::Not => tier2::lower_not(decl.into(), &mut dag, left, &bool_ty, None),
        };
        assert_eq!(dag.get(result).unwrap().op, RiscOp::Logical(kind));
        assert_eq!(dag.get(result).unwrap().inputs.len(), arity);
        assert!(
            dag.nodes()
                .iter()
                .all(|node| !matches!(node.op, RiscOp::Mul | RiscOp::MaxElem))
        );
        assert!(verify::verify(&dag).is_empty());
    }
}

#[test]
fn symbolic_comparison_inherits_proven_operand_shape_without_authored_claims() {
    let symbolic = |name: &str, precision| TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision,
    };
    let inferred_result = symbolic("d44", Prim::Bool);

    let mut valid = Dag::new();
    let valid_decl = valid.declare("test");
    let left = valid.add_node(
        valid_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        symbolic("n", Prim::F32),
        None,
    );
    let right = valid.add_node(
        valid_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        symbolic("n", Prim::F32),
        None,
    );
    let comparison = tier2::lower_lt(
        valid_decl.into(),
        &mut valid,
        left,
        right,
        &inferred_result,
        None,
    );
    assert_eq!(
        valid.get(comparison).unwrap().output_type,
        symbolic("n", Prim::Bool),
        "the direct comparison must project the checker-proven operand shape"
    );
    assert!(
        valid.get(comparison).unwrap().shape_deps.is_empty(),
        "same-shaped symbolic operands do not require an authored shape claim"
    );
    assert!(verify::verify(&valid).is_empty());

    let mut witnessed = Dag::new();
    let witnessed_decl = witnessed.declare("test");
    let left = witnessed.add_node(
        witnessed_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        symbolic("d44", Prim::F32),
        None,
    );
    let right = witnessed.add_node(
        witnessed_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        symbolic("n", Prim::F32),
        None,
    );
    let witness_ty = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let declaration = witnessed.add_node(
        witnessed_decl,
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::Caller,
            parameter: "left".into(),
            axis: RtAxis::Lit(0),
            requirements: vec![],
            claims: vec![],
        },
        vec![left],
        witness_ty.clone(),
        None,
    );
    let equality = witnessed.add_node(
        witnessed_decl,
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::Caller,
            parameter: "right".into(),
            axis: RtAxis::Lit(0),
            requirements: vec![],
            claims: vec![ExtentClaim {
                claim: "n".into(),
                requirement_declares: true,
            }],
        },
        vec![right, declaration],
        witness_ty,
        None,
    );
    let comparison = tier2::lower_lt(
        witnessed_decl.into(),
        &mut witnessed,
        left,
        right,
        &inferred_result,
        None,
    );
    witnessed.add_shape_dep(comparison, equality);
    witnessed.add_root(comparison);
    assert!(
        verify::verify(&witnessed).is_empty(),
        "an earlier exact extent witness transports the checker's symbolic equality: {:?}",
        verify::verify(&witnessed)
    );

    let mut unproved = Dag::new();
    let unproved_decl = unproved.declare("test");
    let left = unproved.add_node(
        unproved_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        symbolic("n", Prim::F32),
        None,
    );
    let right = unproved.add_node(
        unproved_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        symbolic("m", Prim::F32),
        None,
    );
    tier2::lower_lt(
        unproved_decl.into(),
        &mut unproved,
        left,
        right,
        &inferred_result,
        None,
    );
    verifies_and_checks_agreement_when_run(
        &unproved,
        Prim::F32,
        &[("left", 2), ("right", 2)],
        &[("left", 2), ("right", 3)],
        "distinct unresolved symbols are neither proved equal nor contradictory",
    );
}

#[test]
fn evaluator_implements_ieee_comparisons_for_every_active_numeric_dtype() {
    let kinds = [
        ComparisonKind::CmpLt,
        ComparisonKind::Lt,
        ComparisonKind::Eq,
        ComparisonKind::Neq,
        ComparisonKind::Gt,
        ComparisonKind::Gte,
        ComparisonKind::Lte,
    ];
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
        for kind in kinds {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let operand_ty = ty(&[4], precision);
            let bool_ty = ty(&[4], Prim::Bool);
            let left = dag.add_node(
                decl,
                RiscOp::Load {
                    name: "left".into(),
                },
                vec![],
                operand_ty.clone(),
                None,
            );
            let right = dag.add_node(
                decl,
                RiscOp::Load {
                    name: "right".into(),
                },
                vec![],
                operand_ty,
                None,
            );
            let result = dag.add_node(
                decl,
                RiscOp::Compare(kind),
                vec![left, right],
                bool_ty,
                None,
            );
            dag.add_root(result);

            let (lhs, rhs, expected) = if precision.is_float() {
                let expected = match kind {
                    ComparisonKind::Neq => vec![1, 1, 0, 0],
                    ComparisonKind::Eq => vec![0, 0, 1, 1],
                    ComparisonKind::CmpLt | ComparisonKind::Lt => vec![0, 0, 0, 0],
                    ComparisonKind::Gt => vec![0, 0, 0, 0],
                    ComparisonKind::Gte | ComparisonKind::Lte => vec![0, 0, 1, 1],
                };
                (
                    value(precision, vec![4], vec![f64::NAN, 1.0, -0.0, 4.0]),
                    value(precision, vec![4], vec![0.0, f64::NAN, 0.0, 4.0]),
                    expected,
                )
            } else {
                let expected = match kind {
                    ComparisonKind::CmpLt | ComparisonKind::Lt => vec![1, 0, 0, 0],
                    ComparisonKind::Eq => vec![0, 1, 0, 1],
                    ComparisonKind::Neq => vec![1, 0, 1, 0],
                    ComparisonKind::Gt => vec![0, 0, 1, 0],
                    ComparisonKind::Gte => vec![0, 1, 1, 1],
                    ComparisonKind::Lte => vec![1, 1, 0, 1],
                };
                (
                    TensorValue::finalize_from_wide_int(
                        "logical oracle",
                        precision,
                        vec![4],
                        vec![-2, 0, 3, 4],
                    )
                    .unwrap(),
                    TensorValue::finalize_from_wide_int(
                        "logical oracle",
                        precision,
                        vec![4],
                        vec![-1, 0, 2, 4],
                    )
                    .unwrap(),
                    expected,
                )
            };
            let values = eval_tensor(
                &dag,
                &UnordMap::from([("left".into(), lhs), ("right".into(), rhs)]),
            )
            .unwrap();
            assert_eq!(
                values[&result].storage().to_i64_exact_vec().unwrap(),
                expected,
                "{kind:?} at {precision:?}"
            );
        }
    }
}

#[test]
fn evaluator_logical_truth_tables_are_bool_only_and_non_short_circuiting() {
    for (kind, expected) in [
        (LogicalKind::And, vec![0, 0, 0, 1]),
        (LogicalKind::Or, vec![0, 1, 1, 1]),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let bool_ty = ty(&[4], Prim::Bool);
        let left = dag.add_node(
            decl,
            RiscOp::Load {
                name: "left".into(),
            },
            vec![],
            bool_ty.clone(),
            None,
        );
        let right = dag.add_node(
            decl,
            RiscOp::Load {
                name: "right".into(),
            },
            vec![],
            bool_ty.clone(),
            None,
        );
        let output = dag.add_node(
            decl,
            RiscOp::Logical(kind),
            vec![left, right],
            bool_ty,
            None,
        );
        let values = eval_tensor(
            &dag,
            &UnordMap::from([
                ("left".into(), bool_value(vec![4], vec![0, 0, 1, 1])),
                ("right".into(), bool_value(vec![4], vec![0, 1, 0, 1])),
            ]),
        )
        .unwrap();
        assert_eq!(
            values[&output].storage().to_i64_exact_vec().unwrap(),
            expected
        );
    }

    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let bool_ty = ty(&[2], Prim::Bool);
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        bool_ty.clone(),
        None,
    );
    let output = dag.add_node(
        decl,
        RiscOp::Logical(LogicalKind::Not),
        vec![input],
        bool_ty,
        None,
    );
    let values = eval_tensor(
        &dag,
        &UnordMap::from([("x".into(), bool_value(vec![2], vec![0, 1]))]),
    )
    .unwrap();
    assert_eq!(
        values[&output].storage().to_i64_exact_vec().unwrap(),
        vec![1, 0]
    );
}

#[test]
fn where_selects_stored_bits_and_evaluates_both_branches_first() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let bool_ty = ty(&[2], Prim::Bool);
    let data_ty = ty(&[2], Prim::F64);
    let condition = dag.add_node(
        decl,
        RiscOp::Load {
            name: "condition".into(),
        },
        vec![],
        bool_ty,
        None,
    );
    let then_value = dag.add_node(
        decl,
        RiscOp::Load {
            name: "then".into(),
        },
        vec![],
        data_ty.clone(),
        None,
    );
    let else_value = dag.add_node(
        decl,
        RiscOp::Load {
            name: "else".into(),
        },
        vec![],
        data_ty.clone(),
        None,
    );
    let output = dag.add_node(
        decl,
        RiscOp::Where,
        vec![condition, then_value, else_value],
        data_ty,
        None,
    );
    let then_input = value(
        Prim::F64,
        vec![2],
        vec![f64::from_bits(0xfff8_1234_5678_9abc), -0.0],
    );
    let else_input = value(
        Prim::F64,
        vec![2],
        vec![f64::from_bits(0x7ff8_abcd_1234_5678), 0.0],
    );
    let values = eval_tensor(
        &dag,
        &UnordMap::from([
            ("condition".into(), bool_value(vec![2], vec![1, 0])),
            ("then".into(), then_input.clone()),
            ("else".into(), else_input.clone()),
        ]),
    )
    .unwrap();
    assert_eq!(
        scalar_bits(&values[&output], 0),
        scalar_bits(&then_input, 0)
    );
    assert_eq!(
        scalar_bits(&values[&output], 1),
        scalar_bits(&else_input, 1)
    );

    let mut eager = Dag::new();
    let eager_decl = eager.declare("test");
    let scalar_bool = ty(&[], Prim::Bool);
    let scalar_i32 = ty(&[], Prim::Int32);
    let cond = eager.add_node(
        eager_decl,
        RiscOp::synth_const(Prim::Bool, 1.0),
        vec![],
        scalar_bool,
        None,
    );
    let selected = eager.add_node(
        eager_decl,
        RiscOp::synth_const(Prim::Int32, 7.0),
        vec![],
        scalar_i32.clone(),
        None,
    );
    let one = eager.add_node(
        eager_decl,
        RiscOp::synth_const(Prim::Int32, 1.0),
        vec![],
        scalar_i32.clone(),
        None,
    );
    let zero = eager.add_node(
        eager_decl,
        RiscOp::synth_const(Prim::Int32, 0.0),
        vec![],
        scalar_i32.clone(),
        None,
    );
    let trapping_unselected = eager.add_node(
        eager_decl,
        RiscOp::TruncDiv,
        vec![one, zero],
        scalar_i32.clone(),
        None,
    );
    eager.add_node(
        eager_decl,
        RiscOp::Where,
        vec![cond, selected, trapping_unselected],
        scalar_i32,
        None,
    );
    let error = eval_tensor(&eager, &UnordMap::new()).unwrap_err();
    assert!(
        error.contains("division by zero"),
        "unselected branch did not run: {error}"
    );
}

#[test]
fn verifier_rejects_invalid_domains_shapes_outputs_and_arities() {
    let cases = [
        (
            RiscOp::Compare(ComparisonKind::Lt),
            vec![ty(&[2], Prim::Bool), ty(&[2], Prim::Bool)],
            ty(&[2], Prim::Bool),
            "ordered comparison",
        ),
        (
            RiscOp::Compare(ComparisonKind::Eq),
            vec![ty(&[2], Prim::F32), ty(&[3], Prim::F32)],
            ty(&[2], Prim::Bool),
            "shape",
        ),
        (
            RiscOp::Compare(ComparisonKind::Eq),
            vec![ty(&[2], Prim::F32), ty(&[2], Prim::F64)],
            ty(&[2], Prim::Bool),
            "precision",
        ),
        (
            RiscOp::Compare(ComparisonKind::Eq),
            vec![ty(&[2], Prim::String), ty(&[2], Prim::String)],
            ty(&[2], Prim::Bool),
            "numeric or bool",
        ),
        (
            RiscOp::Compare(ComparisonKind::Eq),
            vec![ty(&[2], Prim::F8e4m3), ty(&[2], Prim::F8e4m3)],
            ty(&[2], Prim::Bool),
            "active numeric",
        ),
        (
            RiscOp::Compare(ComparisonKind::Eq),
            vec![ty(&[2], Prim::F32), ty(&[2], Prim::F32)],
            ty(&[2], Prim::F32),
            "Bool",
        ),
        (
            RiscOp::Logical(LogicalKind::And),
            vec![ty(&[2], Prim::F32), ty(&[2], Prim::F32)],
            ty(&[2], Prim::Bool),
            "bool",
        ),
        (
            RiscOp::Where,
            vec![
                ty(&[2], Prim::F32),
                ty(&[2], Prim::F32),
                ty(&[2], Prim::F32),
            ],
            ty(&[2], Prim::F32),
            "condition",
        ),
        (
            RiscOp::Where,
            vec![
                ty(&[2], Prim::Bool),
                ty(&[2], Prim::F32),
                ty(&[2], Prim::F64),
            ],
            ty(&[2], Prim::F32),
            "branches",
        ),
        (
            RiscOp::Where,
            vec![
                ty(&[2], Prim::Bool),
                ty(&[2], Prim::F8e4m3),
                ty(&[2], Prim::F8e4m3),
            ],
            ty(&[2], Prim::F8e4m3),
            "active data element",
        ),
        (
            RiscOp::Where,
            vec![
                ty(&[2], Prim::Bool),
                ty(&[2], Prim::String),
                ty(&[2], Prim::String),
            ],
            ty(&[2], Prim::String),
            "active data element",
        ),
    ];
    for (op, input_types, output_type, needle) in cases {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let inputs = input_types
            .into_iter()
            .enumerate()
            .map(|(index, input_type)| {
                dag.add_node(
                    decl,
                    RiscOp::Load {
                        name: format!("input_{index}").into(),
                    },
                    vec![],
                    input_type,
                    None,
                )
            })
            .collect();
        dag.add_node(decl, op.clone(), inputs, output_type, None);
        let errors = verify::verify(&dag);
        assert!(
            errors.iter().any(|error| error
                .to_ascii_lowercase()
                .contains(&needle.to_ascii_lowercase())),
            "{op:?} should reject with {needle:?}, got {errors:#?}"
        );
    }

    for (op, inputs) in [
        (RiscOp::Compare(ComparisonKind::Eq), 1usize),
        (RiscOp::Logical(LogicalKind::And), 1),
        (RiscOp::Logical(LogicalKind::Not), 2),
        (RiscOp::Where, 2),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let bool_ty = ty(&[], Prim::Bool);
        let ids = (0..inputs)
            .map(|index| {
                dag.add_node(
                    decl,
                    RiscOp::Load {
                        name: format!("input_{index}").into(),
                    },
                    vec![],
                    bool_ty.clone(),
                    None,
                )
            })
            .collect();
        dag.add_node(decl, op.clone(), ids, bool_ty.clone(), None);
        let errors = verify::verify(&dag);
        assert!(
            errors.iter().any(|error| error.contains("inputs")),
            "{op:?} wrong arity was accepted: {errors:#?}"
        );
    }
}

#[test]
fn verifier_compares_resolved_extents_semantically_without_merging_unresolved_symbols() {
    let lit = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let named_left = TensorType {
        dims: vec![DimInfo::Named("left".into(), Some(2))],
        precision: Prim::F32,
    };
    let named_right = TensorType {
        dims: vec![DimInfo::Named("right".into(), Some(2))],
        precision: Prim::F32,
    };
    let bool_named = TensorType {
        dims: vec![DimInfo::Named("predicate".into(), Some(2))],
        precision: Prim::Bool,
    };

    let mut compare = Dag::new();
    let compare_decl = compare.declare("test");
    let left = compare.add_node(
        compare_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        lit.clone(),
        None,
    );
    let right = compare.add_node(
        compare_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        named_left.clone(),
        None,
    );
    let compared = compare.add_node(
        compare_decl,
        RiscOp::Compare(ComparisonKind::Eq),
        vec![left, right],
        bool_named.clone(),
        None,
    );
    compare.add_root(compared);
    assert_eq!(
        verify::verify(&compare),
        Vec::<String>::new(),
        "resolved named and literal extents are the same semantic shape"
    );

    let mut logical = Dag::new();
    let logical_decl = logical.declare("test");
    let left = logical.add_node(
        logical_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Bool,
        },
        None,
    );
    let right = logical.add_node(
        logical_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("right".into(), Some(2))],
            precision: Prim::Bool,
        },
        None,
    );
    let combined = logical.add_node(
        logical_decl,
        RiscOp::Logical(LogicalKind::And),
        vec![left, right],
        TensorType {
            dims: vec![DimInfo::Named("output".into(), Some(2))],
            precision: Prim::Bool,
        },
        None,
    );
    logical.add_root(combined);
    assert_eq!(
        verify::verify(&logical),
        Vec::<String>::new(),
        "logical operations must accept equivalent resolved input and output shapes"
    );

    let mut where_dag = Dag::new();
    let where_dag_decl = where_dag.declare("test");
    let condition = where_dag.add_node(
        where_dag_decl,
        RiscOp::Load {
            name: "condition".into(),
        },
        vec![],
        bool_named,
        None,
    );
    let then_value = where_dag.add_node(
        where_dag_decl,
        RiscOp::Load {
            name: "then".into(),
        },
        vec![],
        named_left,
        None,
    );
    let else_value = where_dag.add_node(
        where_dag_decl,
        RiscOp::Load {
            name: "else".into(),
        },
        vec![],
        lit,
        None,
    );
    let selected = where_dag.add_node(
        where_dag_decl,
        RiscOp::Where,
        vec![condition, then_value, else_value],
        named_right,
        None,
    );
    where_dag.add_root(selected);
    assert_eq!(
        verify::verify(&where_dag),
        Vec::<String>::new(),
        "where must accept equivalent resolved branch, condition, and output shapes"
    );

    let mut runtime_actualized = Dag::new();
    let runtime_actualized_decl = runtime_actualized.declare("test");
    let source = runtime_actualized.add_node(
        runtime_actualized_decl,
        RiscOp::Load {
            name: "source".into(),
        },
        vec![],
        ty(&[2], Prim::F32),
        None,
    );
    let extent = runtime_actualized.add_node(
        runtime_actualized_decl,
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::Caller,
            parameter: "source".into(),
            axis: RtAxis::Lit(0),
            requirements: vec![],
            claims: vec![],
        },
        vec![source],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    let reshaped = runtime_actualized.add_node(
        runtime_actualized_decl,
        RiscOp::Reshape {
            new_shape: vec![chelis_ir::dag::RtDim::Node(1)],
        },
        vec![source, extent],
        TensorType {
            dims: vec![DimInfo::Named("_rt_dim_2_0".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    let compared = runtime_actualized.add_node(
        runtime_actualized_decl,
        RiscOp::Compare(ComparisonKind::Eq),
        vec![source, reshaped],
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Bool,
        },
        None,
    );
    runtime_actualized.add_root(compared);
    assert_eq!(
        verify::verify(&runtime_actualized),
        Vec::<String>::new(),
        "an exact extent witness actualizes the same semantic extent under a synthesized name"
    );

    let mut wildcard_where = Dag::new();
    let wildcard_where_decl = wildcard_where.declare("test");
    let condition = wildcard_where.add_node(
        wildcard_where_decl,
        RiscOp::Load {
            name: "condition".into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("runtime".into(), None)],
            precision: Prim::Bool,
        },
        None,
    );
    let values = wildcard_where.add_node(
        wildcard_where_decl,
        RiscOp::Load {
            name: "values".into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("runtime".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    let zero = wildcard_where.add_node(
        wildcard_where_decl,
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        TensorType {
            dims: vec![DimInfo::Named(String::new(), None)],
            precision: Prim::F32,
        },
        None,
    );
    let selected = wildcard_where.add_node(
        wildcard_where_decl,
        RiscOp::Where,
        vec![condition, zero, values],
        TensorType {
            dims: vec![DimInfo::Named(String::new(), None)],
            precision: Prim::F32,
        },
        None,
    );
    wildcard_where.add_root(selected);
    assert_eq!(
        wildcard_where.get(zero).unwrap().shape_deps,
        vec![values],
        "where must preserve the exact producer that actualizes an anonymous branch shape"
    );
    assert_eq!(
        verify::verify(&wildcard_where),
        Vec::<String>::new(),
        "an anonymous uniform branch takes its shape from the matching where branch"
    );

    let unresolved = |name: &str| TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision: Prim::F32,
    };
    let mut distinct_symbols = Dag::new();
    let distinct_symbols_decl = distinct_symbols.declare("test");
    let left = distinct_symbols.add_node(
        distinct_symbols_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        unresolved("batch"),
        None,
    );
    let right = distinct_symbols.add_node(
        distinct_symbols_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        unresolved("sequence"),
        None,
    );
    let compared = distinct_symbols.add_node(
        distinct_symbols_decl,
        RiscOp::Compare(ComparisonKind::Eq),
        vec![left, right],
        TensorType {
            dims: vec![DimInfo::Named("batch".into(), None)],
            precision: Prim::Bool,
        },
        None,
    );
    distinct_symbols.add_root(compared);
    verifies_and_checks_agreement_when_run(
        &distinct_symbols,
        Prim::F32,
        &[("left", 2), ("right", 2)],
        &[("left", 2), ("right", 3)],
        "distinct unresolved symbols are checked when the comparison runs",
    );

    let mut distinct_logical_symbols = Dag::new();
    let distinct_logical_symbols_decl = distinct_logical_symbols.declare("test");
    let left = distinct_logical_symbols.add_node(
        distinct_logical_symbols_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("batch".into(), None)],
            precision: Prim::Bool,
        },
        None,
    );
    let right = distinct_logical_symbols.add_node(
        distinct_logical_symbols_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("sequence".into(), None)],
            precision: Prim::Bool,
        },
        None,
    );
    distinct_logical_symbols.add_node(
        distinct_logical_symbols_decl,
        RiscOp::Logical(LogicalKind::And),
        vec![left, right],
        TensorType {
            dims: vec![DimInfo::Named("batch".into(), None)],
            precision: Prim::Bool,
        },
        None,
    );
    verifies_and_checks_agreement_when_run(
        &distinct_logical_symbols,
        Prim::Bool,
        &[("left", 2), ("right", 2)],
        &[("left", 2), ("right", 3)],
        "distinct unresolved symbols are checked when the logical operation runs",
    );

    // Literal extents that differ contradict, and the verifier still
    // refuses each operation over them.
    let contradiction = |op: RiscOp, precisions: &[Prim], output: Prim| {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let inputs = precisions
            .iter()
            .enumerate()
            .map(|(index, precision)| {
                dag.add_node(
                    decl,
                    RiscOp::Load {
                        name: format!("input{index}").into(),
                    },
                    vec![],
                    ty(&[2 + index.min(1)], *precision),
                    None,
                )
            })
            .collect::<Vec<_>>();
        dag.add_node(decl, op, inputs, ty(&[2], output), None);
        verify::verify(&dag)
    };
    let errors = contradiction(
        RiscOp::Compare(ComparisonKind::Eq),
        &[Prim::F32, Prim::F32],
        Prim::Bool,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("contradicting operand shapes")),
        "{errors:?}"
    );
    let errors = contradiction(
        RiscOp::Logical(LogicalKind::And),
        &[Prim::Bool, Prim::Bool],
        Prim::Bool,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("output shape must match its operands")),
        "{errors:?}"
    );
    let errors = contradiction(
        RiscOp::Where,
        &[Prim::Bool, Prim::F32, Prim::F32],
        Prim::F32,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("shapes that contradict")),
        "{errors:?}"
    );
}

#[test]
fn verifier_defers_unproved_operand_shapes_and_rejects_producerless_anonymous_outputs() {
    let named = |name: &str, precision| TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision,
    };
    let anonymous = |precision| TensorType {
        dims: vec![DimInfo::Named(String::new(), None)],
        precision,
    };

    let mut compare = Dag::new();
    let compare_decl = compare.declare("test");
    let decoy = compare.add_node(
        compare_decl,
        RiscOp::Load {
            name: "decoy".into(),
        },
        vec![],
        named("runtime", Prim::F32),
        None,
    );
    let left = compare.add_node(
        compare_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        named("runtime", Prim::F32),
        None,
    );
    let right = compare.add_node(
        compare_decl,
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        anonymous(Prim::F32),
        None,
    );
    compare.add_shape_dep(right, decoy);
    compare.add_node(
        compare_decl,
        RiscOp::Compare(ComparisonKind::Eq),
        vec![left, right],
        named("runtime", Prim::Bool),
        None,
    );
    verifies_and_checks_agreement_when_run(
        &compare,
        Prim::F32,
        &[("decoy", 2), ("left", 2)],
        &[("decoy", 3), ("left", 2)],
        "a comparison operand shaped by an unrelated dependency is checked when it runs",
    );

    let mut logical = Dag::new();
    let logical_decl = logical.declare("test");
    let decoy = logical.add_node(
        logical_decl,
        RiscOp::Load {
            name: "decoy".into(),
        },
        vec![],
        named("runtime", Prim::Bool),
        None,
    );
    let left = logical.add_node(
        logical_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        named("runtime", Prim::Bool),
        None,
    );
    let right = logical.add_node(
        logical_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        anonymous(Prim::Bool),
        None,
    );
    logical.add_shape_dep(right, decoy);
    logical.add_node(
        logical_decl,
        RiscOp::Logical(LogicalKind::And),
        vec![left, right],
        named("runtime", Prim::Bool),
        None,
    );
    verifies_and_checks_agreement_when_run(
        &logical,
        Prim::Bool,
        &[("decoy", 2), ("left", 2), ("right", 2)],
        &[("decoy", 2), ("left", 2), ("right", 3)],
        "a logical operand shaped by an unrelated dependency is checked when it runs",
    );

    let mut where_dag = Dag::new();
    let where_dag_decl = where_dag.declare("test");
    let decoy = where_dag.add_node(
        where_dag_decl,
        RiscOp::Load {
            name: "decoy".into(),
        },
        vec![],
        named("runtime", Prim::F32),
        None,
    );
    let condition = where_dag.add_node(
        where_dag_decl,
        RiscOp::Load {
            name: "condition".into(),
        },
        vec![],
        named("runtime", Prim::Bool),
        None,
    );
    let then_value = where_dag.add_node(
        where_dag_decl,
        RiscOp::Load {
            name: "then".into(),
        },
        vec![],
        named("runtime", Prim::F32),
        None,
    );
    let else_value = where_dag.add_node(
        where_dag_decl,
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        anonymous(Prim::F32),
        None,
    );
    where_dag.add_shape_dep(else_value, decoy);
    where_dag.add_node(
        where_dag_decl,
        RiscOp::Where,
        vec![condition, then_value, else_value],
        named("runtime", Prim::F32),
        None,
    );
    assert_eq!(
        verify::verify(&where_dag),
        Vec::<String>::new(),
        "a where branch shaped by an unrelated dependency is checked when it runs"
    );

    let mut anonymous_compare_output = Dag::new();
    let anonymous_compare_output_decl = anonymous_compare_output.declare("test");
    let left = anonymous_compare_output.add_node(
        anonymous_compare_output_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        named("runtime", Prim::F32),
        None,
    );
    let right = anonymous_compare_output.add_node(
        anonymous_compare_output_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        named("runtime", Prim::F32),
        None,
    );
    anonymous_compare_output.add_node(
        anonymous_compare_output_decl,
        RiscOp::Compare(ComparisonKind::Eq),
        vec![left, right],
        anonymous(Prim::Bool),
        None,
    );
    assert!(
        verify::verify(&anonymous_compare_output)
            .iter()
            .any(|error| error.contains("output shape")),
        "comparison output needs explicit authority for an anonymous dimension"
    );

    let mut anonymous_logical_output = Dag::new();
    let anonymous_logical_output_decl = anonymous_logical_output.declare("test");
    let left = anonymous_logical_output.add_node(
        anonymous_logical_output_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        named("runtime", Prim::Bool),
        None,
    );
    let right = anonymous_logical_output.add_node(
        anonymous_logical_output_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        named("runtime", Prim::Bool),
        None,
    );
    anonymous_logical_output.add_node(
        anonymous_logical_output_decl,
        RiscOp::Logical(LogicalKind::And),
        vec![left, right],
        anonymous(Prim::Bool),
        None,
    );
    assert!(
        verify::verify(&anonymous_logical_output)
            .iter()
            .any(|error| error.contains("output shape")),
        "logical output needs explicit authority for an anonymous dimension"
    );

    let mut unresolved_compare_authority = Dag::new();
    let unresolved_compare_authority_decl = unresolved_compare_authority.declare("test");
    let left = unresolved_compare_authority.add_node(
        unresolved_compare_authority_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        anonymous(Prim::F32),
        None,
    );
    let right = unresolved_compare_authority.add_node(
        unresolved_compare_authority_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        anonymous(Prim::F32),
        None,
    );
    let output = unresolved_compare_authority.add_node(
        unresolved_compare_authority_decl,
        RiscOp::Compare(ComparisonKind::Eq),
        vec![left, right],
        anonymous(Prim::Bool),
        None,
    );
    unresolved_compare_authority.add_shape_dep(output, left);
    verifies_and_checks_agreement_when_run(
        &unresolved_compare_authority,
        Prim::F32,
        &[("left", 2), ("right", 2)],
        &[("left", 2), ("right", 3)],
        "an anonymous comparison output takes its operand's checked extent",
    );

    let mut unresolved_logical_authority = Dag::new();
    let unresolved_logical_authority_decl = unresolved_logical_authority.declare("test");
    let left = unresolved_logical_authority.add_node(
        unresolved_logical_authority_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        anonymous(Prim::Bool),
        None,
    );
    let right = unresolved_logical_authority.add_node(
        unresolved_logical_authority_decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        anonymous(Prim::Bool),
        None,
    );
    let output = unresolved_logical_authority.add_node(
        unresolved_logical_authority_decl,
        RiscOp::Logical(LogicalKind::And),
        vec![left, right],
        anonymous(Prim::Bool),
        None,
    );
    unresolved_logical_authority.add_shape_dep(output, left);
    verifies_and_checks_agreement_when_run(
        &unresolved_logical_authority,
        Prim::Bool,
        &[("left", 2), ("right", 2)],
        &[("left", 2), ("right", 3)],
        "an anonymous logical output takes its operand's checked extent",
    );
}

#[test]
fn verifier_rejects_shape_dependencies_that_override_operation_provenance() {
    let anonymous = |precision| TensorType {
        dims: vec![DimInfo::Named(String::new(), None)],
        precision,
    };

    let mut add_dag = Dag::new();
    let add_dag_decl = add_dag.declare("test");
    let left = add_dag.add_node(
        add_dag_decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        anonymous(Prim::F32),
        None,
    );
    let unrelated = add_dag.add_node(
        add_dag_decl,
        RiscOp::Load {
            name: "unrelated".into(),
        },
        vec![],
        anonymous(Prim::F32),
        None,
    );
    let sum = add_dag.add_node(
        add_dag_decl,
        RiscOp::Add,
        vec![left, unrelated],
        anonymous(Prim::F32),
        None,
    );
    let comparison = add_dag.add_node(
        add_dag_decl,
        RiscOp::Compare(ComparisonKind::Eq),
        vec![sum, left],
        anonymous(Prim::Bool),
        None,
    );
    add_dag.add_shape_dep(comparison, left);
    verifies_and_checks_agreement_when_run(
        &add_dag,
        Prim::F32,
        &[("left", 2), ("unrelated", 2)],
        &[("left", 2), ("unrelated", 3)],
        "the addition checks its operands before the comparison reads its result",
    );

    let mut pad_dag = Dag::new();
    let pad_dag_decl = pad_dag.declare("test");
    let input = pad_dag.add_node(
        pad_dag_decl,
        RiscOp::Load {
            name: "input".into(),
        },
        vec![],
        anonymous(Prim::F32),
        None,
    );
    let padded = pad_dag.add_node(
        pad_dag_decl,
        RiscOp::Pad {
            padding: vec![(RtDim::Lit(1), RtDim::Lit(0))],
            fill: chelis_types::scalar_from_f64("pad provenance", Prim::F32, 0.0).unwrap(),
        },
        vec![input],
        anonymous(Prim::F32),
        None,
    );
    pad_dag.add_shape_dep(padded, input);
    let comparison = pad_dag.add_node(
        pad_dag_decl,
        RiscOp::Compare(ComparisonKind::Eq),
        vec![padded, input],
        anonymous(Prim::Bool),
        None,
    );
    pad_dag.add_shape_dep(comparison, input);
    // The pad's extent is one more than its input's, whatever the shape
    // dependency says, so the comparison's agreement check refuses every run.
    assert_eq!(verify::verify(&pad_dag), Vec::<String>::new());
    let error = evaluate_lengths(&pad_dag, Prim::F32, &[("input", 2)])
        .expect_err("a nonzero Pad shape dependency must not override its computed axis");
    assert!(
        error.contains("eq operands disagree at axis 0: lhs [3] has 3, rhs [2] has 2"),
        "{error}"
    );
}

#[test]
fn comparisons_have_zero_cotangents_logicals_reject_and_where_routes_g() {
    for kind in [
        ComparisonKind::CmpLt,
        ComparisonKind::Lt,
        ComparisonKind::Eq,
        ComparisonKind::Neq,
        ComparisonKind::Gt,
        ComparisonKind::Gte,
        ComparisonKind::Lte,
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let scalar_f64 = ty(&[], Prim::F64);
        let scalar_bool = ty(&[], Prim::Bool);
        let left = dag.add_node(
            decl,
            RiscOp::Load {
                name: "left".into(),
            },
            vec![],
            scalar_f64.clone(),
            None,
        );
        let right = dag.add_node(
            decl,
            RiscOp::Load {
                name: "right".into(),
            },
            vec![],
            scalar_f64.clone(),
            None,
        );
        let predicate = dag.add_node(
            decl,
            RiscOp::Compare(kind),
            vec![left, right],
            scalar_bool.clone(),
            None,
        );
        let then_value = dag.add_node(
            decl,
            RiscOp::Load {
                name: "then".into(),
            },
            vec![],
            scalar_f64.clone(),
            None,
        );
        let else_value = dag.add_node(
            decl,
            RiscOp::Load {
                name: "else".into(),
            },
            vec![],
            scalar_f64.clone(),
            None,
        );
        let output = dag.add_node(
            decl,
            RiscOp::Where,
            vec![predicate, then_value, else_value],
            scalar_f64,
            None,
        );
        let differentiated =
            grad_dag_checked(&dag, output, &[left, right, then_value, else_value]).unwrap();
        let values = eval_tensor(
            &differentiated.dag,
            &UnordMap::from([
                ("left".into(), value(Prim::F64, vec![], vec![f64::NAN])),
                ("right".into(), value(Prim::F64, vec![], vec![0.0])),
                ("then".into(), value(Prim::F64, vec![], vec![11.0])),
                ("else".into(), value(Prim::F64, vec![], vec![13.0])),
            ]),
        )
        .unwrap();
        assert_eq!(
            values[&differentiated.grad_nodes[&left]].to_f64_lossy_vec(),
            vec![0.0]
        );
        assert_eq!(
            values[&differentiated.grad_nodes[&right]].to_f64_lossy_vec(),
            vec![0.0]
        );
        let (then_grad, else_grad) = if kind == ComparisonKind::Neq {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };
        assert_eq!(
            values[&differentiated.grad_nodes[&then_value]].to_f64_lossy_vec(),
            vec![then_grad]
        );
        assert_eq!(
            values[&differentiated.grad_nodes[&else_value]].to_f64_lossy_vec(),
            vec![else_grad]
        );
    }

    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let scalar_bool = ty(&[], Prim::Bool);
    let scalar_f64 = ty(&[], Prim::F64);
    let left = dag.add_node(
        decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        scalar_bool.clone(),
        None,
    );
    let right = dag.add_node(
        decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        scalar_bool.clone(),
        None,
    );
    let logical = dag.add_node(
        decl,
        RiscOp::Logical(LogicalKind::And),
        vec![left, right],
        scalar_bool,
        None,
    );
    let output = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F64,
        },
        vec![logical],
        scalar_f64,
        None,
    );
    let error = match grad_dag_checked(&dag, output, &[left]) {
        Ok(_) => panic!("logical operation unexpectedly differentiated"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        AdError::NotSupported {
            op: "and",
            reason: AdRejectionReason::LogicalOperation,
        }
    );
}

#[test]
fn logical_random_activation_is_control_only_during_grad() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let scalar_bool = ty(&[], Prim::Bool);
    let scalar_f64 = ty(&[], Prim::F64);
    let left = dag.add_node(
        decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        scalar_bool.clone(),
        None,
    );
    let right = dag.add_node(
        decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        scalar_bool.clone(),
        None,
    );
    let activation = dag.add_node(
        decl,
        RiscOp::Logical(LogicalKind::And),
        vec![left, right],
        scalar_bool,
        None,
    );
    let template = dag.add_node(
        decl,
        RiscOp::Load {
            name: "template".into(),
        },
        vec![],
        scalar_f64.clone(),
        None,
    );
    let scalar = |precision| TensorType {
        dims: vec![],
        precision,
    };
    let low = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, -1.0),
        vec![],
        scalar(Prim::F32),
        None,
    );
    let high = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        scalar(Prim::F32),
        None,
    );
    let seed = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::Int64, 17.0),
        vec![],
        scalar(Prim::Int64),
        None,
    );
    let key = dag.add_node(
        decl,
        RiscOp::KeyFromSeed,
        vec![seed],
        scalar(Prim::Key),
        None,
    );
    let output = dag.add_node(
        chelis_ir::dag::Owner::new(decl, Some(activation)),
        RiscOp::UniformLike,
        vec![template, low, high, key],
        scalar_f64,
        None,
    );

    let differentiated = grad_dag_checked(&dag, output, &[template])
        .unwrap_or_else(|error| panic!("random path activation leaked into float AD: {error}"));
    let values = eval_tensor(
        &differentiated.dag,
        &UnordMap::from([
            ("left".into(), bool_value(vec![], vec![1])),
            ("right".into(), bool_value(vec![], vec![1])),
            ("template".into(), value(Prim::F64, vec![], vec![5.0])),
        ]),
    )
    .unwrap();
    assert_eq!(
        values[&differentiated.grad_nodes[&template]].to_f64_lossy_vec(),
        vec![0.0]
    );
}

#[test]
fn comparison_logical_and_where_are_fusion_barriers_and_vmap_shape_preserving() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let data_ty = ty(&[2], Prim::F32);
    let bool_ty = ty(&[2], Prim::Bool);
    let left = dag.add_node(
        decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        data_ty.clone(),
        None,
    );
    let right = dag.add_node(
        decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        data_ty.clone(),
        None,
    );
    let cmp = dag.add_node(
        decl,
        RiscOp::Compare(ComparisonKind::Gte),
        vec![left, right],
        bool_ty.clone(),
        None,
    );
    let logical = dag.add_node(
        decl,
        RiscOp::Logical(LogicalKind::Not),
        vec![cmp],
        bool_ty,
        None,
    );
    let selected = dag.add_node(
        decl,
        RiscOp::Where,
        vec![logical, left, right],
        data_ty,
        None,
    );
    dag.add_root(selected);
    let fused = fuse(&dag);
    assert!(
        fused
            .nodes()
            .iter()
            .any(|node| { matches!(node.op, RiscOp::Compare(ComparisonKind::Gte)) })
    );
    assert!(
        fused
            .nodes()
            .iter()
            .any(|node| { matches!(node.op, RiscOp::Logical(LogicalKind::Not)) })
    );
    assert!(
        fused
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Where))
    );
    assert!(
        fused
            .nodes()
            .iter()
            .all(|node| !matches!(node.op, RiscOp::FusedElem { .. }))
    );

    let vmapped =
        chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(3)).expect("vmap typed predicates");
    for node in vmapped.nodes().iter().filter(|node| {
        matches!(
            node.op,
            RiscOp::Compare(_) | RiscOp::Logical(_) | RiscOp::Where
        )
    }) {
        assert_eq!(node.output_type.dims[0], DimInfo::Lit(3));
    }
    assert!(verify::verify(&vmapped).is_empty());
}
