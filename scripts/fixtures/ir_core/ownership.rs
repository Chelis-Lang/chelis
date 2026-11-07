use std::collections::BTreeMap;

use chelis_dag_core::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir_core::capacity_key::{CapacityKeyBuild, CapacityScope, capacity_key_for_node};
use chelis_ir_core::host_type_state::ConcreteHostType;
use chelis_ir_core::ownership::classify::{HeapKind, Placement, ValueClass};
use chelis_ir_core::ownership::dag::{DagAnalysisView, DagOwnershipPlan};
use chelis_ir_core::ownership::ir::{
    Block, BlockId, HostSiteAction, HostSiteId, HostSiteKind, HostSiteMap, HostSiteRecord,
    Op, OpId, Operand, Operation, OperationRole, OwnerId, OwnerInfo, OwnerOrigin,
    OwnershipProgram, ScheduleState, Terminator, Unit, UnitId, UnitKind,
};
use chelis_ir_core::ownership::last_use;
use chelis_ir_core::ownership::locations::SitesByUnit;
use chelis_ir_core::ownership::storage::{StoragePlacement, StoragePolicy, build_storage_plan};
use chelis_ir_core::ownership::{LiveByteBound, OwnershipError, verify};
use chelis_numeric_core::prim::Prim;

fn key(view: DagAnalysisView<'_>, node: NodeId) -> chelis_ir_core::capacity_key::CapacityKey {
    match capacity_key_for_node(view, node).expect("capacity calculation") {
        CapacityKeyBuild::Exact(key) => key,
        CapacityKeyBuild::NotProven(reason) => panic!("expected an exact key: {reason}"),
    }
}

fn tensor(dims: Vec<DimInfo>) -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    let decl = dag.declare("consumer");
    let ty = TensorType { dims, precision: Prim::F32 };
    let input = dag.add_node(decl, RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let output = dag.add_node(decl, RiscOp::Copy, vec![input], ty, None);
    dag.add_root(output);
    (dag, input, output)
}

fn main() {
    let mut program = OwnershipProgram {
        units: vec![Unit {
            id: UnitId(0),
            name: "roots".into(),
            kind: UnitKind::Roots,
            schedule: ScheduleState::Phase2ScopeExit,
            callable_body: None,
            entry: BlockId(0),
            blocks: vec![Block {
                id: BlockId(0),
                params: vec![],
                ops: vec![
                    Operation {
                        id: OpId(0),
                        role: OperationRole::Semantic,
                        kind: Op::Define { dest: OwnerId(0), label: "fresh".into() },
                    },
                    Operation {
                        id: OpId(1),
                        role: OperationRole::Semantic,
                        kind: Op::RootConsume { root: "result".into(), owner: Operand::move_(OwnerId(0)) },
                    },
                ],
                terminator: Terminator::Exit,
            }],
            owners: BTreeMap::from([(OwnerId(0), OwnerInfo {
                ty: ConcreteHostType::Scalar(Prim::String),
                class: ValueClass::Heap(HeapKind::String),
                placement: Placement::Value,
                origin: OwnerOrigin::Owned,
                names: vec!["value".into()],
                span_id: None,
            })]),
            captured_globals: BTreeMap::new(),
        }],
    };
    let mut sites = HostSiteMap::from_records(vec![
        HostSiteRecord {
            id: HostSiteId::from_index(0), unit: 0, kind: HostSiteKind::Expression,
            actions: vec![HostSiteAction::Operation { unit: 0, block: BlockId(0), operation: OpId(0) }],
        },
        HostSiteRecord {
            id: HostSiteId::from_index(1), unit: 0, kind: HostSiteKind::ManifestRoot,
            actions: vec![HostSiteAction::Operation { unit: 0, block: BlockId(0), operation: OpId(1) }],
        },
    ]);
    let verification = last_use::schedule(&mut program, &mut sites).expect("schedule").into_verification();
    assert_eq!(program.units[0].schedule, ScheduleState::CanonicalLastUse);
    assert_eq!(verification.live_set_bound().max_live_heap_owners(), 1);
    assert_eq!(verification.live_set_bound().max_live_bytes, LiveByteBound::Unknown);
    assert!(matches!(verification.locations().operation(&program, 0, BlockId(0), OpId(0)).map(|op| &op.kind), Some(Op::Define { .. })));
    assert_eq!(SitesByUnit::new(&sites).of(0), &[0, 1]);
    let mut invalid = program.clone();
    invalid.units[0].schedule = ScheduleState::Phase2ScopeExit;
    invalid.units[0].blocks[0].ops.pop();
    assert!(matches!(verify::verify(&invalid), Err(OwnershipError::MissingTerminal { owner: 0, .. })));

    let (dag, input, output) = tensor(vec![DimInfo::Lit(2), DimInfo::Lit(3)]);
    let plan = DagOwnershipPlan::lower(&dag).expect("ordinary DAG ownership");
    plan.verify(&dag).expect("ordinary DAG verification");
    let view = DagAnalysisView::new(&dag, CapacityScope::from_identity(10));
    let (flat, _, flat_output) = tensor(vec![DimInfo::Lit(6)]);
    let flat_view = DagAnalysisView::new(&flat, CapacityScope::from_identity(11));
    assert!(key(view, output).prove_equal(&key(flat_view, flat_output)).is_ok());
    let policy = StoragePolicy {
        mirror_inputs: false, shared_views: false, skipped_keys: false,
        alias_stores: false, recurrence_tape: true,
    };
    let (placements, slots, reusable, bound) = build_storage_plan(
        view, policy, &chelis_unord::UnordSet::new(), None, |decision| decision,
    ).expect("ordinary storage calculation");
    assert!(matches!(placements[input.0], StoragePlacement::BorrowedEntry));
    assert!(matches!(placements[output.0], StoragePlacement::OwnedSlot { .. }));
    assert_eq!(slots.len(), 1);
    assert!(reusable.is_empty());
    assert_eq!(bound, LiveByteBound::Exact(24));

    let (dynamic, _, dynamic_output) = tensor(vec![DimInfo::Named("N".into(), None)]);
    let local = DagAnalysisView::new(&dynamic, CapacityScope::from_identity(20));
    let foreign = DagAnalysisView::new(&dynamic, CapacityScope::from_identity(21));
    assert!(key(local, dynamic_output).prove_equal(&key(foreign, dynamic_output)).is_err());
    println!("IR_OWNERSHIP_CONSUMER {{\"scheduled_units\":1,\"site_indices\":2,\"literal_capacity_equal\":true,\"storage_slots\":1,\"live_bytes\":24,\"invalid_ownership_refused\":true,\"foreign_scope_refused\":true}}");
}
