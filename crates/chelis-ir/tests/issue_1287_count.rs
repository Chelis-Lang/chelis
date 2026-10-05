//! Dedicated Count IR semantics for chelis#1287 / [05-OP-29].

use chelis_unord::UnordMap;

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::fuse::fuse;
use chelis_ir::grad::{AdError, AdRejectionReason, grad_dag_checked};
use chelis_ir::lower::try_lower_program;
use chelis_ir::verify;
use chelis_ir::vmap::vectorize_axis0;
use chelis_types::check_typed_program;
use chelis_types::types::Prim;

fn ty(dims: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}

fn count_dag(input_shape: &[usize], axes: Vec<usize>, output_shape: &[usize]) -> Dag {
    let mut dag = Dag::default();
    let decl = dag.declare("test");
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(input_shape, Prim::Bool),
        None,
    );
    let output = dag.add_node(
        decl,
        RiscOp::Count { axes },
        vec![input],
        ty(output_shape, Prim::Int64),
        None,
    );
    dag.add_root(output);
    dag
}

#[test]
fn evaluator_counts_multi_axis_groups_in_original_row_major_order() {
    let dag = count_dag(&[2, 3, 2], vec![2, 0], &[3]);
    assert!(verify::verify(&dag).is_empty());
    let input = TensorValue::finalize_from_wide_int(
        "test",
        Prim::Bool,
        vec![2, 3, 2],
        vec![1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    )
    .expect("bool fixture");
    let values = eval_tensor(&dag, &UnordMap::from([("x".into(), input)])).expect("count eval");
    let output = &values[&chelis_ir::dag::NodeId(1)];
    assert_eq!(output.shape, vec![3]);
    assert_eq!(output.to_f64_lossy_vec(), vec![3.0, 3.0, 2.0]);
    assert_eq!(output.prim(), Prim::Int64);
}

#[test]
fn evaluator_returns_zero_for_a_selected_zero_extent() {
    let dag = count_dag(&[2, 0, 3], vec![1], &[2, 3]);
    assert!(verify::verify(&dag).is_empty());
    let input = TensorValue::finalize_from_wide_int("test", Prim::Bool, vec![2, 0, 3], vec![])
        .expect("empty bool fixture");
    let values = eval_tensor(&dag, &UnordMap::from([("x".into(), input)])).expect("count eval");
    assert_eq!(
        values[&chelis_ir::dag::NodeId(1)].to_f64_lossy_vec(),
        vec![0.0; 6]
    );
}

#[test]
fn evaluator_count_only_plans_groups_and_calls_the_typed_kernel() {
    let source = include_str!("../src/eval.rs");
    let count_body = source
        .split_once("pub fn count_tensor")
        .expect("Count evaluator exists")
        .1
        .split_once("fn reshape")
        .expect("Count evaluator ends before reshape")
        .0;
    assert!(
        count_body.contains("count_tensor_groups(input.storage(), &groups)"),
        "Count must call the closed typed kernel after planning ordered groups"
    );
    assert!(
        !count_body.contains("try_fold")
            && !count_body.contains(".fold(")
            && !count_body.contains("wrapping_add")
            && !count_body.contains("checked_add"),
        "IR Count must not own arithmetic; the typed kernel owns the canonical tree"
    );
}

#[test]
fn verifier_rejects_noncanonical_axes_wrong_dtype_and_wrong_shape() {
    for (axes, input_prim, output_dims, output_prim, needle) in [
        (vec![], Prim::Bool, vec![2, 3], Prim::Int64, "non-empty"),
        (vec![0, 1], Prim::Bool, vec![], Prim::Int64, "descending"),
        (vec![1, 1], Prim::Bool, vec![2], Prim::Int64, "descending"),
        (vec![2], Prim::Bool, vec![2], Prim::Int64, "out of range"),
        (vec![1], Prim::F32, vec![2], Prim::Int64, "bool"),
        (vec![1], Prim::Bool, vec![2], Prim::F32, "i64"),
        (vec![1], Prim::Bool, vec![3], Prim::Int64, "shape"),
    ] {
        let mut dag = Dag::default();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(&[2, 3], input_prim),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Count { axes },
            vec![input],
            ty(&output_dims, output_prim),
            None,
        );
        let found = verify::verify(&dag);
        assert!(
            found.iter().any(|error| error.contains(needle)),
            "expected {needle:?}, got {found:#?}"
        );
    }
}

fn lower_surf(source: &str) -> Result<Dag, String> {
    let decls = chelis_surf::parser::parse_str(source).map_err(|e| format!("parse: {e:?}"))?;
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&exprs).map_err(|r| format!("check: {:#?}", r.errors))?;
    let checked = chelis_effects::check_program(&checked).map_err(|e| format!("effects: {e:?}"))?;
    let checked =
        chelis_types::check_linearity(&checked).map_err(|e| format!("linearity: {e:?}"))?;
    try_lower_program(&checked).map_err(|diag| format!("lower: {diag:?}"))
}

#[test]
fn lowering_emits_one_count_with_original_axes_normalized_descending() {
    let dag = lower_surf(
        r#"
def main() -> tensor[3, i64] = {
  x: tensor[2, 3, 4, bool] = [
    [[true, false, true, false], [true, false, true, false], [true, false, true, false]],
    [[true, false, true, false], [true, false, true, false], [true, false, true, false]]
  ]
  count(&x, 0, -1)
}
"#,
    )
    .expect("count lowers");
    let counts: Vec<_> = dag
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            RiscOp::Count { axes } => Some(axes.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        counts,
        vec![vec![2, 0]],
        "lowered ops: {:#?}",
        dag.nodes().iter().map(|node| &node.op).collect::<Vec<_>>()
    );
    assert!(
        dag.nodes().iter().all(|node| !matches!(
            node.op,
            RiscOp::Sum { .. } | RiscOp::Cast { .. } | RiscOp::NamedCast { .. }
        )),
        "Count must not regress to cast-plus-sum or a cast compatibility alias"
    );
    assert!(verify::verify(&dag).is_empty());
}

#[test]
fn vmap_shifts_every_count_axis_and_preserves_the_batch_axis() {
    let dag = count_dag(&[2, 3, 4], vec![2, 0], &[3]);
    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(5)).expect("count vmap");
    let count = vmapped
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Count { .. }))
        .expect("Count survives vmap");
    assert_eq!(count.op, RiscOp::Count { axes: vec![3, 1] });
    assert_eq!(
        count.output_type.dims,
        vec![DimInfo::Lit(5), DimInfo::Lit(3)]
    );
    assert!(verify::verify(&vmapped).is_empty());
}

#[test]
fn count_is_a_fusion_barrier() {
    let dag = count_dag(&[2, 3], vec![1], &[2]);
    let fused = fuse(&dag);
    assert!(
        fused
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Count { ref axes } if axes == &[1]))
    );
}

#[test]
fn grad_rejects_a_live_count_with_a_structured_reason() {
    let mut dag = count_dag(&[2, 3], vec![1, 0], &[]);
    let decl = dag.nodes()[0].owner.decl;
    let count = dag.roots()[0];
    let output = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F64,
        },
        vec![count],
        ty(&[], Prim::F64),
        None,
    );
    dag.add_root(output);
    let err = match grad_dag_checked(&dag, output, &[chelis_ir::dag::NodeId(0)]) {
        Ok(_) => panic!("count has no adjoint"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("integer-reduction output"),
        "Count's rendered rejection must describe a reduction, not an index: {err}"
    );
    assert_eq!(
        err,
        AdError::NotSupported {
            op: "count",
            reason: AdRejectionReason::IntegerReductionOutput,
        }
    );
}
