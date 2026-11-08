#![cfg(test)]

use super::*;
use chelis_dag_core::dag::{Dag, TensorType};
use chelis_numeric_core::prim::Prim;
use core::sync::atomic::{AtomicU64, Ordering};

const C_POLICY: StoragePolicy = StoragePolicy {
    mirror_inputs: false,
    shared_views: false,
    skipped_keys: false,
    alias_stores: false,
    recurrence_tape: true,
};
const HIP_POLICY: StoragePolicy = StoragePolicy {
    mirror_inputs: true,
    shared_views: true,
    skipped_keys: true,
    alias_stores: true,
    recurrence_tape: false,
};

fn analysis_view(dag: &Dag) -> DagAnalysisView<'_> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    super::super::dag::DagOwnershipPlan::lower(dag)
        .unwrap()
        .verify(dag)
        .unwrap();
    let scope =
        crate::capacity_key::CapacityScope::from_identity(NEXT.fetch_add(1, Ordering::Relaxed));
    DagAnalysisView::new(dag, scope)
}

fn calculate_plan(
    dag: DagAnalysisView<'_>,
    policy: StoragePolicy,
) -> Result<BuiltStoragePlan<ReuseDecision>, OwnershipError> {
    let views = (!policy.mirror_inputs).then(|| dag.contraction_views());
    let mut skipped = dag.reduction_inlined_fused_elems();
    if let Some(views) = &views {
        skipped.extend(
            dag.nodes()
                .iter()
                .filter(|node| views.is_deferred(node.id))
                .map(|node| node.id),
        );
        skipped.extend(
            dag.reduction_inlined_products()
                .iter()
                .flat_map(chelis_dag_core::fuse::InlinedProduct::skipped),
        );
    }
    build_storage_plan(dag, policy, &skipped, views.as_ref(), |decision| decision)
}

fn vector(elements: usize, precision: Prim) -> TensorType {
    TensorType {
        dims: vec![chelis_dag_core::dag::DimInfo::Lit(elements)],
        precision,
    }
}

fn scalar(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision,
    }
}

fn matrix(rows: usize, columns: usize, precision: Prim) -> TensorType {
    TensorType {
        dims: vec![
            chelis_dag_core::dag::DimInfo::Lit(rows),
            chelis_dag_core::dag::DimInfo::Lit(columns),
        ],
        precision,
    }
}

fn dropped_then_root(first: TensorType, second: TensorType) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let first_node = dag.add_node(
        decl,
        RiscOp::synth_const(first.precision, 1.0),
        vec![],
        first.clone(),
        None,
    );
    dag.add_node(decl, RiscOp::Drop, vec![first_node], first, None);
    let second_node = dag.add_node(
        decl,
        RiscOp::synth_const(second.precision, 2.0),
        vec![],
        second,
        None,
    );
    dag.add_root(second_node);
    dag
}

#[test]
fn deferred_payloads_keep_original_bound_and_claim_reads_live() {
    for dependency in ["shape", "none"] {
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let scalar = TensorType {
            dims: vec![],
            precision: Prim::Int64,
        };
        let source = dag.add_node(
            owner,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        let claim = dag.add_node(
            owner,
            RiscOp::synth_const(Prim::Int64, 3.0),
            vec![],
            scalar.clone(),
            None,
        );
        let zero = dag.add_node(
            owner,
            RiscOp::synth_const(Prim::Int64, 0.0),
            vec![],
            scalar.clone(),
            None,
        );
        let bound = dag.add_node(owner, RiscOp::Add, vec![claim, zero], scalar.clone(), None);
        let intervening = dag.add_node(
            owner,
            RiscOp::synth_const(Prim::Int64, 0.0),
            vec![],
            scalar,
            None,
        );
        let expanded = dag.add_node(
            owner,
            RiscOp::Expand {
                axis: 0,
                size: chelis_dag_core::dag::RtDim::Node(1),
            },
            vec![source, bound],
            vector(3, Prim::F32),
            None,
        );
        match dependency {
            "shape" => dag.add_shape_dep(expanded, claim),
            "none" => (),
            _ => unreachable!(),
        }
        let sum = dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            vec![expanded],
            TensorType::scalar_f32(),
            None,
        );
        dag.set_roots(vec![sum, intervening]);
        let analysis = analysis_view(&dag);
        let views = analysis.contraction_views();
        assert!(views.is_deferred(expanded));
        let placements = classify_nodes(analysis, C_POLICY, &[expanded].into_iter().collect());
        let owners = compute_owner_map(&placements);
        let mut requirements = owner_requirements(analysis, &placements, &owners).unwrap();
        extend_lifetimes(
            analysis,
            C_POLICY,
            &placements,
            &owners,
            &mut requirements,
            Some(&views),
        );
        assert_eq!(requirements[&bound].last_use_index, expanded.0);
        assert_eq!(
            requirements[&claim].last_use_index,
            if dependency == "none" {
                bound.0
            } else {
                expanded.0
            }
        );
        let (placements, slots, _, _) = calculate_plan(analysis, C_POLICY).unwrap();
        assert!(matches!(
            placements.get(expanded.0),
            Some(StoragePlacement::Skipped)
        ));
        let slot_for_node = |node: NodeId| placements[node.0].slot();
        let last = |node| slots[slot_for_node(node).unwrap().index()].last_use_index();
        assert_eq!(
            last(bound),
            expanded.0,
            "runtime bound is read at the original node"
        );
        if dependency != "none" {
            assert!(
                last(claim) >= expanded.0,
                "{dependency} witness cannot be repurposed before its check"
            );
            assert_ne!(slot_for_node(claim), slot_for_node(intervening));
        }
    }
}

#[test]
fn deferred_product_keeps_result_claim_witness_live() {
    use chelis_dag_core::dag::{ExtentWitnessSite, RtAxis};
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let source = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        vector(3, Prim::F32),
        None,
    );
    let claim = dag.add_node(
        owner,
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::LiteralResultClaim,
            parameter: String::new(),
            axis: RtAxis::Lit(0),
            requirements: vec![
                chelis_numeric_core::dtype_semantics::scalar_from_i64("test", Prim::Int64, 3)
                    .unwrap(),
            ],
            claims: vec![],
        },
        vec![],
        scalar(Prim::Int64),
        None,
    );
    let intervening = dag.add_node(
        owner,
        RiscOp::synth_const(Prim::Int64, 0.0),
        vec![],
        scalar(Prim::Int64),
        None,
    );
    let product = dag.add_node(
        owner,
        RiscOp::Mul,
        vec![source, source],
        vector(3, Prim::F32),
        None,
    );
    dag.add_result_claim_dep(product, claim);
    let sum = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![product],
        TensorType::scalar_f32(),
        None,
    );
    dag.set_roots(vec![sum, intervening]);
    let analysis = analysis_view(&dag);
    assert!(analysis.contraction_views().is_deferred(product));
    let (placements, slots, _, _) = calculate_plan(analysis, C_POLICY).unwrap();
    let slot_for_node = |node: NodeId| placements[node.0].slot();
    assert_ne!(slot_for_node(claim), slot_for_node(intervening));
    assert!(slots[slot_for_node(claim).unwrap().index()].last_use_index() >= product.0);
}

#[test]
fn destructive_drop_retires_an_exact_capacity_slot_in_both_lanes() {
    let dag = dropped_then_root(matrix(2, 2, Prim::F32), vector(4, Prim::F32));
    let analysis = analysis_view(&dag);
    let (_, c_slots, _, _) = calculate_plan(analysis, C_POLICY).expect("same-byte C capacity plan");
    let (_, hip_slots, _, _) =
        calculate_plan(analysis, HIP_POLICY).expect("same-byte HIP capacity plan");

    assert_eq!(c_slots.len(), 2);
    assert_eq!(hip_slots.len(), 2);
}

#[test]
fn equal_width_different_representations_never_share_a_slot() {
    let dag = dropped_then_root(vector(4, Prim::F32), vector(4, Prim::Int32));
    let (_, slots, _, _) =
        calculate_plan(analysis_view(&dag), C_POLICY).expect("mixed-representation plan");
    assert_eq!(slots.len(), 2);
}
