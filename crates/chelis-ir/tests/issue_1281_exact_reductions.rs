//! Executable acceptance leg for chelis#1281 exact evaluator and AD semantics.

use chelis_ir::dag::{Dag, DimInfo, ReduceWindowKind, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::{AdError, AdRejectionReason, grad_dag_checked};
use chelis_ir::lower::try_lower_program;
use chelis_ir::tier2;
use chelis_types::types::Prim;
use chelis_types::{RawTensor, StorageView, check_ir_program, finalize_tensor};
use chelis_unord::UnordMap;

fn tensor_type(dims: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}

fn scalar_type(precision: Prim) -> TensorType {
    tensor_type(&[], precision)
}

fn exact_tensor(shape: Vec<usize>, precision: Prim, values: RawTensor) -> TensorValue {
    TensorValue::from_storage(
        shape,
        finalize_tensor("issue_1281_input", precision, values).unwrap(),
    )
}

fn evaluate_single_input(
    dag: &Dag,
    name: &str,
    input: TensorValue,
) -> Result<chelis_unord::UnordMap<chelis_ir::NodeId, TensorValue>, String> {
    let mut inputs = UnordMap::new();
    inputs.insert(name.to_string(), input);
    eval_tensor(dag, &inputs)
}

fn lower_surf(source: &str) -> Dag {
    let decls = chelis_surf::parser::parse_str(source).expect("Surf fixture parses");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture desugars"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("Surf fixture expands")
    .into_exprs();
    let checked = check_ir_program(&exprs)
        .unwrap_or_else(|report| panic!("Surf fixture checks: {:#?}", report.errors));
    let checked = chelis_effects::check_program(&checked).expect("Surf fixture effects check");
    let checked = chelis_types::check_linearity(&checked).expect("Surf fixture linearity check");
    try_lower_program(&checked).expect("Surf fixture lowers")
}

#[test]
fn extrema_and_window_extrema_preserve_first_nan_and_first_equal_bits() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_type(&[3], Prim::F32),
        None,
    );
    let max = dag.add_node(
        RiscOp::MaxReduce { axis: 0 },
        vec![x],
        scalar_type(Prim::F32),
        None,
    );
    let window = dag.add_node(
        RiscOp::ReduceWindow {
            reducer: ReduceWindowKind::Min,
            window_shape: vec![3],
            strides: vec![1],
        },
        vec![x],
        tensor_type(&[1], Prim::F32),
        None,
    );
    let first_nan = f32::from_bits(0xffc1_2345);
    let later_nan = f32::from_bits(0x7fc5_4321);
    let values = evaluate_single_input(
        &dag,
        "x",
        exact_tensor(
            vec![3],
            Prim::F32,
            RawTensor::Float(vec![f64::from(first_nan), 1.0, f64::from(later_nan)]),
        ),
    )
    .unwrap();
    for node in [max, window] {
        let StorageView::F32(storage) = values[&node].storage().view() else {
            panic!("f32 reduction must preserve f32 storage");
        };
        assert_eq!(storage[0].to_bits(), first_nan.to_bits());
    }

    let zeros = evaluate_single_input(
        &dag,
        "x",
        exact_tensor(vec![3], Prim::F32, RawTensor::Float(vec![-0.0, 0.0, 0.0])),
    )
    .unwrap();
    let StorageView::F32(storage) = zeros[&max].storage().view() else {
        panic!("f32 reduction must preserve f32 storage");
    };
    assert_eq!(storage[0].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn arg_reductions_use_exact_i64_comparison_and_lowest_nan_index() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_type(&[4], Prim::Int64),
        None,
    );
    let argmax = dag.add_node(
        RiscOp::Argmax { axis: 0 },
        vec![x],
        scalar_type(Prim::Int64),
        None,
    );
    let exact = evaluate_single_input(
        &dag,
        "x",
        exact_tensor(
            vec![4],
            Prim::Int64,
            RawTensor::Int(vec![
                9_007_199_254_740_992,
                9_007_199_254_740_993,
                9_007_199_254_740_993,
                0,
            ]),
        ),
    )
    .unwrap();
    assert_eq!(exact[&argmax].storage().to_i64_exact_vec(), Some(vec![1]));

    let mut nan_dag = Dag::new();
    let n = nan_dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_type(&[4], Prim::F32),
        None,
    );
    let max = nan_dag.add_node(
        RiscOp::Argmax { axis: 0 },
        vec![n],
        scalar_type(Prim::Int64),
        None,
    );
    let min = nan_dag.add_node(
        RiscOp::Argmin { axis: 0 },
        vec![n],
        scalar_type(Prim::Int64),
        None,
    );
    let nan_values = evaluate_single_input(
        &nan_dag,
        "x",
        exact_tensor(
            vec![4],
            Prim::F32,
            RawTensor::Float(vec![5.0, f64::NAN, 9.0, f64::NAN]),
        ),
    )
    .unwrap();
    assert_eq!(nan_values[&max].storage().to_i64_exact_vec(), Some(vec![1]));
    assert_eq!(nan_values[&min].storage().to_i64_exact_vec(), Some(vec![1]));
}

#[test]
fn runtime_empty_mean_extrema_and_arg_reductions_trap_domain() {
    for (op, precision, expected_name) in [
        (RiscOp::MaxReduce { axis: 0 }, Prim::F32, "max_reduce"),
        (RiscOp::MinReduce { axis: 0 }, Prim::F32, "min_reduce"),
        (RiscOp::Argmax { axis: 0 }, Prim::Int64, "argmax_reduce"),
        (RiscOp::Argmin { axis: 0 }, Prim::Int64, "argmin_reduce"),
    ] {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("n".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(op, vec![x], scalar_type(precision), None);
        let error = evaluate_single_input(
            &dag,
            "x",
            exact_tensor(vec![0], Prim::F32, RawTensor::Float(vec![])),
        )
        .expect_err("empty reduction must trap");
        assert_eq!(
            error,
            format!(
                "numeric trap: domain in {expected_name} at {}",
                precision.name()
            )
        );
    }

    for precision in [Prim::F16, Prim::Bf16, Prim::F32] {
        let mut mean_dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision,
        };
        let x = mean_dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let mean = tier2::lower_mean(&mut mean_dag, x, 0, &ty, None);
        let error = evaluate_single_input(
            &mean_dag,
            "x",
            exact_tensor(vec![0], precision, RawTensor::Float(vec![])),
        )
        .expect_err("empty runtime mean must trap through storage casts");
        assert_eq!(
            error,
            format!("numeric trap: domain in mean at {}", precision.name())
        );

        let values = evaluate_single_input(
            &mean_dag,
            "x",
            exact_tensor(vec![2], precision, RawTensor::Float(vec![2.0, 4.0])),
        )
        .expect("non-empty runtime mean must remain admitted");
        assert_eq!(values[&mean].to_f64_lossy_vec(), vec![3.0]);
    }
}

#[test]
fn variadic_named_axis_mean_reduces_highest_original_axis_first() {
    for axes in [("head", "seq"), ("seq", "head")] {
        let dag = lower_surf(&format!(
            "def main(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = \
             mean(x, {}, {})",
            axes.0, axes.1
        ));
        let sum_axes = dag
            .nodes()
            .iter()
            .filter_map(|node| match node.op {
                RiscOp::Sum { axis, .. } => Some(axis),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sum_axes,
            vec![2, 2, 1, 1],
            "variadic mean axes {axes:?} must execute in descending original position"
        );
    }
}

#[test]
fn mean_uses_canonical_sum_then_divide_at_declared_f32_width() {
    let mut dag = Dag::new();
    let ty = tensor_type(&[4], Prim::F32);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let mean = tier2::lower_mean(&mut dag, x, 0, &ty, None);
    let values = evaluate_single_input(
        &dag,
        "x",
        exact_tensor(
            vec![4],
            Prim::F32,
            RawTensor::Float(vec![-1.0e-7, 3.0, 16_777_216.0, -33_554_432.0]),
        ),
    )
    .unwrap();
    let StorageView::F32(storage) = values[&mean].storage().view() else {
        panic!("f32 mean must preserve f32 storage");
    };
    assert_eq!(storage[0].to_bits(), 0xca7f_fffd);
}

fn reduction_grad(op: RiscOp, input: TensorValue) -> TensorValue {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_type(&[input.len()], Prim::F32),
        None,
    );
    let out = dag.add_node(op, vec![x], scalar_type(Prim::F32), None);
    let grad = grad_dag_checked(&dag, out, &[x]).unwrap();
    let values = evaluate_single_input(&grad.dag, "x", input).unwrap();
    values[&grad.grad_nodes[&x]].clone()
}

#[test]
fn extrema_ad_splits_non_nan_ties_and_routes_first_nan() {
    for op in [RiscOp::MaxReduce { axis: 0 }, RiscOp::MinReduce { axis: 0 }] {
        let third = f64::from(1.0f32 / 3.0);
        let ties = reduction_grad(
            op.clone(),
            exact_tensor(vec![3], Prim::F32, RawTensor::Float(vec![2.0, 2.0, 2.0])),
        );
        assert_eq!(ties.to_f64_lossy_vec(), vec![third, third, third]);

        let nan = reduction_grad(
            op,
            exact_tensor(
                vec![3],
                Prim::F32,
                RawTensor::Float(vec![f64::NAN, f64::NAN, 1.0]),
            ),
        );
        assert_eq!(nan.to_f64_lossy_vec(), vec![1.0, 0.0, 0.0]);
    }
}

#[test]
fn extrema_ad_infinity_ties_split_g_over_equal_positive_and_negative_infinity() {
    let max = reduction_grad(
        RiscOp::MaxReduce { axis: 0 },
        exact_tensor(
            vec![3],
            Prim::F32,
            RawTensor::Float(vec![f64::INFINITY, f64::INFINITY, 1.0]),
        ),
    );
    assert_eq!(max.to_f64_lossy_vec(), vec![0.5, 0.5, 0.0]);

    let min = reduction_grad(
        RiscOp::MinReduce { axis: 0 },
        exact_tensor(
            vec![3],
            Prim::F32,
            RawTensor::Float(vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 1.0]),
        ),
    );
    assert_eq!(min.to_f64_lossy_vec(), vec![0.5, 0.5, 0.0]);
}

#[test]
fn runtime_extent_extrema_ad_routes_the_first_nan_without_a_static_axis_size() {
    let mut dag = Dag::new();
    let runtime_ty = TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], runtime_ty, None);
    let out = dag.add_node(
        RiscOp::MaxReduce { axis: 0 },
        vec![x],
        scalar_type(Prim::F32),
        None,
    );
    let grad = grad_dag_checked(&dag, out, &[x])
        .expect("runtime extent must not silently lose the extrema adjoint");
    let values = evaluate_single_input(
        &grad.dag,
        "x",
        exact_tensor(
            vec![3],
            Prim::F32,
            RawTensor::Float(vec![f64::NAN, f64::NAN, 2.0]),
        ),
    )
    .unwrap();
    assert_eq!(
        values[&grad.grad_nodes[&x]].to_f64_lossy_vec(),
        vec![1.0, 0.0, 0.0]
    );
}

#[test]
fn window_extrema_ad_splits_ties_routes_nan_and_accumulates_overlap() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_type(&[3], Prim::F32),
        None,
    );
    let window = dag.add_node(
        RiscOp::ReduceWindow {
            reducer: ReduceWindowKind::Max,
            window_shape: vec![2],
            strides: vec![1],
        },
        vec![x],
        tensor_type(&[2], Prim::F32),
        None,
    );
    let loss = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![window],
        scalar_type(Prim::F32),
        None,
    );
    let grad = grad_dag_checked(&dag, loss, &[x]).unwrap();

    let ties = evaluate_single_input(
        &grad.dag,
        "x",
        exact_tensor(vec![3], Prim::F32, RawTensor::Float(vec![f64::INFINITY; 3])),
    )
    .unwrap();
    assert_eq!(
        ties[&grad.grad_nodes[&x]].to_f64_lossy_vec(),
        vec![0.5, 1.0, 0.5]
    );

    let nan = evaluate_single_input(
        &grad.dag,
        "x",
        exact_tensor(
            vec![3],
            Prim::F32,
            RawTensor::Float(vec![f64::NAN, f64::NAN, 2.0]),
        ),
    )
    .unwrap();
    assert_eq!(
        nan[&grad.grad_nodes[&x]].to_f64_lossy_vec(),
        vec![1.0, 1.0, 0.0]
    );
}

#[test]
fn integer_extrema_and_argument_outputs_are_structurally_non_differentiable() {
    for (op, expected_op, reason) in [
        (
            RiscOp::MaxReduce { axis: 0 },
            "max_reduce",
            AdRejectionReason::IntegerArithmeticOutput,
        ),
        (
            RiscOp::MinReduce { axis: 0 },
            "min_reduce",
            AdRejectionReason::IntegerArithmeticOutput,
        ),
    ] {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_type(&[2], Prim::Int64),
            None,
        );
        let reduced = dag.add_node(op, vec![x], scalar_type(Prim::Int64), None);
        let cast = dag.add_node(
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![reduced],
            scalar_type(Prim::F32),
            None,
        );
        assert_eq!(
            grad_dag_checked(&dag, cast, &[x]).err(),
            Some(AdError::NotSupported {
                op: expected_op,
                reason: reason.clone(),
            })
        );
    }
}
