use std::collections::BTreeMap;

use chelis_deep::ast::{Atom, Expr};
use chelis_deep::span::Span;
use chelis_types::manifest::{RootEntry, RootManifest};
use chelis_types::types::Lane;
use chelis_types::types::Prim;

use super::OwnershipError;
use super::classify::{HeapKind, NonHeapKind, Placement, ValueClass, classify};
use super::ir::{
    ApplyKind, Block, BlockId, BlockParam, CallableBody, Edge, EdgeId, EdgeTerminal,
    HostSiteAction, HostSiteId, HostSiteKind, HostSiteMap, HostSiteRecord, Op, OpId, Operand,
    Operation, OperationRole, OperationSchema, OwnerId, OwnerInfo, OwnerOrigin,
    OwnershipProgram as RawProgram, OwnershipUse, ParamMode, ScheduleState, Terminal, Terminator,
    Unit, UnitId, UnitKind,
};
use super::{DagDirective, DagOwnershipPlan};
use crate::host::{
    ConcreteHostFunction, ConcreteHostProgram, HostBinding, HostDisplayRoot, HostExpr,
    HostExprKind, HostFunctionOrigin,
};
use crate::host_type_state::ConcreteHostType;
use crate::{Dag, DimInfo, RiscOp, TensorType};

fn ty(prim: Prim) -> ConcreteHostType {
    ConcreteHostType::Scalar(prim)
}

fn info(prim: Prim, origin: OwnerOrigin) -> OwnerInfo {
    let ty = ty(prim);
    OwnerInfo {
        class: classify(&ty, Placement::Value).unwrap(),
        ty,
        placement: Placement::Value,
        origin,
        names: Vec::new(),
    }
}

fn parameter_info(prim: Prim, origin: OwnerOrigin) -> OwnerInfo {
    let ty = ty(prim);
    OwnerInfo {
        class: classify(&ty, Placement::Parameter).unwrap(),
        ty,
        placement: Placement::Parameter,
        origin,
        names: Vec::new(),
    }
}

fn block(id: u32, params: Vec<BlockParam>, ops: Vec<Op>, mut terminator: Terminator) -> Block {
    for (edge_index, edge) in terminator.edges_mut().enumerate() {
        if edge.id == EdgeId::UNASSIGNED {
            edge.id = EdgeId((id << 8) | u32::try_from(edge_index).unwrap());
        }
    }
    Block {
        id: BlockId(id),
        params,
        ops: ops
            .into_iter()
            .enumerate()
            .map(|(index, kind)| Operation {
                id: OpId((id << 16) | u32::try_from(index).unwrap()),
                role: OperationRole::Semantic,
                kind,
            })
            .collect(),
        terminator,
    }
}

fn roots(blocks: Vec<Block>, owners: BTreeMap<OwnerId, OwnerInfo>) -> RawProgram {
    RawProgram {
        units: vec![Unit {
            id: UnitId(0),
            name: "roots".into(),
            kind: UnitKind::Roots,
            schedule: ScheduleState::Phase2ScopeExit,
            callable_body: None,
            entry: BlockId(0),
            blocks,
            owners,
        }],
    }
}

fn edge(target: u32, args: Vec<Operand>) -> Edge {
    Edge {
        id: EdgeId::UNASSIGNED,
        target: BlockId(target),
        args,
        terminals: Vec::new(),
    }
}

fn edge_terminal(id: u32, kind: Terminal) -> EdgeTerminal {
    EdgeTerminal { id: OpId(id), kind }
}

fn verify_raw(program: RawProgram) -> Result<RawProgram, OwnershipError> {
    super::verify::verify(&program)?;
    Ok(program)
}

fn define(id: u32) -> Op {
    Op::Define {
        dest: OwnerId(id),
        label: "fresh".into(),
    }
}

fn root(id: u32) -> Op {
    Op::RootConsume {
        root: "result".into(),
        owner: Operand {
            owner: OwnerId(id),
            use_: OwnershipUse::Move,
        },
    }
}

fn provisional(id: u32, owner: u32, heap: bool) -> Operation {
    Operation {
        id: OpId(id),
        role: OperationRole::ProvisionalScopeExit,
        kind: if heap {
            Op::Drop {
                owner: Operand::move_(OwnerId(owner)),
            }
        } else {
            Op::Discard {
                owner: OwnerId(owner),
            }
        },
    }
}

fn borrow_op(owner: u32) -> Op {
    Op::Apply {
        dest: None,
        label: "observe".into(),
        kind: ApplyKind::Intrinsic,
        schema: OperationSchema::new(vec![OwnershipUse::Borrow], None),
        args: vec![Operand::borrow(OwnerId(owner))],
    }
}

#[test]
fn one_owner_one_terminal_verifies_and_renders_stably() {
    let program = roots(
        vec![block(0, vec![], vec![define(0), root(0)], Terminator::Exit)],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );
    assert_eq!(
        super::render::render(&verify_raw(program).unwrap()),
        "unit roots Roots\n  b0 ():\n    %0 = fresh\n    root result move %0\n    exit\n"
    );
}

#[test]
fn unterminated_heap_owner_is_rejected() {
    let program = roots(
        vec![block(0, vec![], vec![define(0)], Terminator::Exit)],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );
    assert!(matches!(
        verify_raw(program),
        Err(OwnershipError::MissingTerminal { owner: 0, .. })
    ));
}

#[test]
fn clone_only_mints_an_owner_through_copy() {
    let program = roots(
        vec![block(
            0,
            vec![],
            vec![
                define(0),
                Op::Copy {
                    dest: OwnerId(1),
                    source: Operand {
                        owner: OwnerId(0),
                        use_: OwnershipUse::Clone,
                    },
                },
                root(0),
                root(1),
            ],
            Terminator::Exit,
        )],
        BTreeMap::from([
            (OwnerId(0), info(Prim::String, OwnerOrigin::Owned)),
            (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        ]),
    );
    verify_raw(program).unwrap();
}

#[test]
fn clone_disposition_outside_copy_is_rejected() {
    let program = roots(
        vec![block(
            0,
            vec![],
            vec![
                define(0),
                Op::Apply {
                    dest: None,
                    label: "bad".into(),
                    kind: ApplyKind::Intrinsic,
                    schema: OperationSchema::new(vec![OwnershipUse::Borrow], None),
                    args: vec![Operand {
                        owner: OwnerId(0),
                        use_: OwnershipUse::Clone,
                    }],
                },
                root(0),
            ],
            Terminator::Exit,
        )],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );
    assert!(matches!(
        verify_raw(program),
        Err(OwnershipError::WrongUse { owner: 0, .. })
    ));
}

#[test]
fn typed_apply_schema_accepts_its_exact_mode_and_rejects_a_forged_one() {
    let make = |use_| {
        roots(
            vec![block(
                0,
                vec![],
                vec![
                    define(0),
                    Op::Apply {
                        dest: None,
                        label: "diagnostic spelling is not semantics".into(),
                        kind: ApplyKind::Intrinsic,
                        schema: OperationSchema::new(vec![OwnershipUse::Move], None),
                        args: vec![Operand {
                            owner: OwnerId(0),
                            use_,
                        }],
                    },
                ],
                Terminator::Exit,
            )],
            BTreeMap::from([(OwnerId(0), info(Prim::Int64, OwnerOrigin::Owned))]),
        )
    };
    verify_raw(make(OwnershipUse::Move)).unwrap();
    assert!(matches!(
        verify_raw(make(OwnershipUse::Borrow)),
        Err(OwnershipError::WrongUse { owner: 0, .. })
    ));
}

#[test]
fn every_owned_identity_needs_a_terminal_even_when_nonheap() {
    let bad = roots(
        vec![block(0, vec![], vec![define(0)], Terminator::Exit)],
        BTreeMap::from([(OwnerId(0), info(Prim::Int64, OwnerOrigin::Owned))]),
    );
    assert!(matches!(
        verify_raw(bad),
        Err(OwnershipError::MissingTerminal { owner: 0, .. })
    ));

    let good = roots(
        vec![block(
            0,
            vec![],
            vec![define(0), Op::Discard { owner: OwnerId(0) }],
            Terminator::Exit,
        )],
        BTreeMap::from([(OwnerId(0), info(Prim::Int64, OwnerOrigin::Owned))]),
    );
    verify_raw(good).unwrap();

    let heap_discard = roots(
        vec![block(
            0,
            vec![],
            vec![define(0), Op::Discard { owner: OwnerId(0) }],
            Terminator::Exit,
        )],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );
    assert!(matches!(
        verify_raw(heap_discard),
        Err(OwnershipError::HeapDiscard { owner: 0, .. })
    ));
}

#[test]
fn stable_operation_ids_survive_reordering_and_duplicates_are_rejected() {
    let mut program = roots(
        vec![block(
            0,
            vec![],
            vec![
                define(0),
                define(1),
                Op::Discard { owner: OwnerId(1) },
                Op::Discard { owner: OwnerId(0) },
            ],
            Terminator::Exit,
        )],
        BTreeMap::from([
            (OwnerId(0), info(Prim::Int64, OwnerOrigin::Owned)),
            (OwnerId(1), info(Prim::Int64, OwnerOrigin::Owned)),
        ]),
    );
    let record = |index, action| HostSiteRecord {
        id: HostSiteId::from_index(index),
        unit: 0,
        kind: HostSiteKind::Expression,
        actions: vec![action],
    };
    let sites = HostSiteMap {
        records: vec![
            record(
                0,
                HostSiteAction::Operation {
                    unit: 0,
                    block: BlockId(0),
                    operation: OpId(0),
                },
            ),
            record(
                1,
                HostSiteAction::Operation {
                    unit: 0,
                    block: BlockId(0),
                    operation: OpId(1),
                },
            ),
            record(
                2,
                HostSiteAction::Operation {
                    unit: 0,
                    block: BlockId(0),
                    operation: OpId(2),
                },
            ),
            record(
                3,
                HostSiteAction::Operation {
                    unit: 0,
                    block: BlockId(0),
                    operation: OpId(3),
                },
            ),
            HostSiteRecord {
                id: HostSiteId::from_index(4),
                unit: 0,
                kind: HostSiteKind::FunctionReturn,
                actions: vec![HostSiteAction::Terminator {
                    unit: 0,
                    block: BlockId(0),
                }],
            },
        ],
    };
    super::verify::verify_host_actions(&program, &RootManifest { entries: vec![] }, &sites)
        .unwrap();

    let mut duplicate_unit = program.clone();
    let mut duplicate = duplicate_unit.units[0].clone();
    duplicate.name = "same structural unit identity".into();
    duplicate_unit.units.push(duplicate);
    assert!(matches!(
        super::verify::verify(&duplicate_unit),
        Err(OwnershipError::DuplicateIdentity {
            kind: "unit",
            id: 0,
            ..
        })
    ));

    program.units[0].blocks[0].ops.swap(0, 1);
    let verification = super::verify::verify(&program).unwrap();
    super::verify::verify_host_actions(&program, &RootManifest { entries: vec![] }, &sites)
        .unwrap();
    let action =
        super::verified_host_action(&program, &verification, &sites.records[0].actions[0]).unwrap();
    assert!(matches!(
        action,
        super::VerifiedHostAction::Operation(super::VerifiedHostOperation::Define {
            dest,
            ..
        }) if dest.id().key == 0
    ));

    program.units[0].blocks[0].ops[1].id = OpId(1);
    assert!(matches!(
        super::verify::verify(&program),
        Err(OwnershipError::DuplicateIdentity {
            kind: "operation",
            id: 1,
            ..
        })
    ));
}

#[test]
fn edge_terminals_are_closed_path_local_consumes() {
    let make = |else_terminals| {
        roots(
            vec![
                block(
                    0,
                    vec![BlockParam {
                        owner: OwnerId(0),
                        mode: ParamMode::EntryBorrow,
                    }],
                    vec![define(1), define(2)],
                    Terminator::Branch {
                        condition: Operand::borrow(OwnerId(0)),
                        then_edge: Edge {
                            id: EdgeId::UNASSIGNED,
                            target: BlockId(1),
                            args: Vec::new(),
                            terminals: vec![
                                edge_terminal(90, Terminal::Drop(OwnerId(1))),
                                edge_terminal(91, Terminal::Discard(OwnerId(2))),
                            ],
                        },
                        else_edge: Edge {
                            id: EdgeId::UNASSIGNED,
                            target: BlockId(1),
                            args: Vec::new(),
                            terminals: else_terminals,
                        },
                    },
                ),
                block(1, vec![], vec![], Terminator::Exit),
            ],
            BTreeMap::from([
                (OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow)),
                (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
                (OwnerId(2), info(Prim::Int64, OwnerOrigin::Owned)),
            ]),
        )
    };

    verify_raw(make(vec![
        edge_terminal(92, Terminal::Drop(OwnerId(1))),
        edge_terminal(93, Terminal::Discard(OwnerId(2))),
    ]))
    .unwrap();
    assert!(matches!(
        verify_raw(make(Vec::new())),
        Err(OwnershipError::JoinMismatch { block: 1, .. })
    ));
}

#[test]
fn edge_arguments_are_transferred_before_path_local_terminals() {
    let program = roots(
        vec![
            block(
                0,
                vec![],
                vec![define(0)],
                Terminator::Jump(Edge {
                    id: EdgeId::UNASSIGNED,
                    target: BlockId(1),
                    args: vec![Operand::borrow(OwnerId(0))],
                    terminals: vec![edge_terminal(90, Terminal::Drop(OwnerId(0)))],
                }),
            ),
            block(
                1,
                vec![BlockParam {
                    owner: OwnerId(1),
                    mode: ParamMode::Borrowed,
                }],
                vec![],
                Terminator::Exit,
            ),
        ],
        BTreeMap::from([
            (OwnerId(0), info(Prim::String, OwnerOrigin::Owned)),
            (
                OwnerId(1),
                info(Prim::String, OwnerOrigin::BorrowedFrom(OwnerId(0))),
            ),
        ]),
    );
    assert!(matches!(
        verify_raw(program),
        Err(OwnershipError::LiveBorrowAtConsume {
            owner: 0,
            borrow: 1,
            block: 0,
            ..
        })
    ));
}

#[test]
fn edge_terminal_kind_is_derived_from_the_owner_class() {
    let make = |prim, terminal| {
        roots(
            vec![
                block(
                    0,
                    vec![],
                    vec![define(0)],
                    Terminator::Jump(Edge {
                        id: EdgeId::UNASSIGNED,
                        target: BlockId(1),
                        args: vec![],
                        terminals: vec![edge_terminal(90, terminal)],
                    }),
                ),
                block(1, vec![], vec![], Terminator::Exit),
            ],
            BTreeMap::from([(OwnerId(0), info(prim, OwnerOrigin::Owned))]),
        )
    };

    assert!(matches!(
        verify_raw(make(Prim::Int64, Terminal::Drop(OwnerId(0)))),
        Err(OwnershipError::NonHeapDrop { owner: 0, .. })
    ));
    assert!(matches!(
        verify_raw(make(Prim::String, Terminal::Discard(OwnerId(0)))),
        Err(OwnershipError::HeapDiscard { owner: 0, .. })
    ));
}

fn direct_call_program(transform_result: bool, callee: UnitId) -> RawProgram {
    let direct_call = Op::Apply {
        dest: Some(OwnerId(0)),
        label: "diagnostic spelling cannot select a callee".into(),
        kind: ApplyKind::DirectCall { callee },
        schema: OperationSchema::new(
            Vec::new(),
            Some(ValueClass::NonHeap(NonHeapKind::Scalar(Prim::Int64))),
        ),
        args: Vec::new(),
    };
    let caller_blocks = if transform_result {
        vec![block(
            0,
            vec![],
            vec![
                direct_call,
                Op::Apply {
                    dest: Some(OwnerId(1)),
                    label: "ordinary value operation".into(),
                    kind: ApplyKind::Intrinsic,
                    schema: OperationSchema::new(
                        vec![OwnershipUse::Move],
                        Some(ValueClass::NonHeap(NonHeapKind::Scalar(Prim::Int64))),
                    ),
                    args: vec![Operand::move_(OwnerId(0))],
                },
            ],
            Terminator::Return {
                result: Operand::move_(OwnerId(1)),
            },
        )]
    } else {
        vec![
            block(
                0,
                vec![],
                vec![direct_call],
                Terminator::Jump(edge(1, vec![Operand::move_(OwnerId(0))])),
            ),
            block(
                1,
                vec![BlockParam {
                    owner: OwnerId(1),
                    mode: ParamMode::Owned,
                }],
                vec![],
                Terminator::Return {
                    result: Operand::move_(OwnerId(1)),
                },
            ),
        ]
    };
    let caller_owners = BTreeMap::from([
        (OwnerId(0), info(Prim::Int64, OwnerOrigin::Owned)),
        (OwnerId(1), info(Prim::Int64, OwnerOrigin::Owned)),
    ]);
    RawProgram {
        units: vec![
            Unit {
                id: UnitId(0),
                name: "roots".into(),
                kind: UnitKind::Roots,
                schedule: ScheduleState::Phase2ScopeExit,
                callable_body: None,
                entry: BlockId(0),
                blocks: vec![block(0, vec![], vec![], Terminator::Exit)],
                owners: BTreeMap::new(),
            },
            Unit {
                id: UnitId(1),
                name: "caller".into(),
                kind: UnitKind::Function,
                schedule: ScheduleState::Phase2ScopeExit,
                callable_body: Some(CallableBody::new(BlockId(0))),
                entry: BlockId(0),
                blocks: caller_blocks,
                owners: caller_owners,
            },
            Unit {
                id: UnitId(2),
                name: "callee".into(),
                kind: UnitKind::Function,
                schedule: ScheduleState::Phase2ScopeExit,
                callable_body: Some(CallableBody::new(BlockId(0))),
                entry: BlockId(0),
                blocks: vec![block(
                    0,
                    vec![],
                    vec![define(0)],
                    Terminator::Return {
                        result: Operand::move_(OwnerId(0)),
                    },
                )],
                owners: BTreeMap::from([(OwnerId(0), info(Prim::Int64, OwnerOrigin::Owned))]),
            },
        ],
    }
}

#[test]
fn resolved_direct_call_identity_drives_tail_chain_classification() {
    let direct = direct_call_program(false, UnitId(2));
    let verification = super::verify::verify(&direct).unwrap();
    assert!(verification.is_tail_call(UnitId(1), OpId(0)));

    let transformed = direct_call_program(true, UnitId(2));
    let verification = super::verify::verify(&transformed).unwrap();
    assert!(!verification.is_tail_call(UnitId(1), OpId(0)));

    let mut label_spoof = direct_call_program(false, UnitId(2));
    let Op::Apply { label, kind, .. } = &mut label_spoof.units[1].blocks[0].ops[0].kind else {
        unreachable!()
    };
    *label = "call:callee".into();
    *kind = ApplyKind::Intrinsic;
    let verification = super::verify::verify(&label_spoof).unwrap();
    assert!(!verification.is_tail_call(UnitId(1), OpId(0)));

    let missing = direct_call_program(false, UnitId(99));
    assert!(matches!(
        super::verify::verify(&missing),
        Err(OwnershipError::MissingUnit { unit: 99, .. })
    ));

    let roots_target = direct_call_program(false, UnitId(0));
    assert!(matches!(
        super::verify::verify(&roots_target),
        Err(OwnershipError::NonFunctionCallee { unit: 0, .. })
    ));

    let mut wrong_schema = direct_call_program(false, UnitId(2));
    let Op::Apply { schema, .. } = &mut wrong_schema.units[1].blocks[0].ops[0].kind else {
        unreachable!()
    };
    schema.result = None;
    assert!(matches!(
        super::verify::verify(&wrong_schema),
        Err(OwnershipError::DirectCallSchema { unit: 2, .. })
    ));
}

#[test]
fn direct_call_schema_is_derived_from_the_actual_callable_body() {
    let mut forged = direct_call_program(false, UnitId(2));
    forged.units[1]
        .owners
        .insert(OwnerId(0), info(Prim::Bool, OwnerOrigin::Owned));
    forged.units[1]
        .owners
        .insert(OwnerId(1), info(Prim::Bool, OwnerOrigin::Owned));
    let Op::Apply { schema, .. } = &mut forged.units[1].blocks[0].ops[0].kind else {
        unreachable!()
    };
    schema.result = Some(ValueClass::NonHeap(NonHeapKind::Scalar(Prim::Bool)));
    assert!(matches!(
        super::verify::verify(&forged),
        Err(OwnershipError::DirectCallSchema { unit: 2, .. })
    ));

    let mut wrong_return_mode = direct_call_program(false, UnitId(2));
    let Terminator::Return { result } = &mut wrong_return_mode.units[2].blocks[0].terminator else {
        unreachable!()
    };
    result.use_ = OwnershipUse::Borrow;
    assert!(matches!(
        super::verify::verify(&wrong_return_mode),
        Err(OwnershipError::FunctionReturnMode {
            unit,
            block: 0,
            ..
        }) if unit == "callee"
    ));

    let mut inconsistent_returns = direct_call_program(false, UnitId(2));
    inconsistent_returns.units[2].blocks = vec![
        block(
            0,
            vec![],
            vec![define(0)],
            Terminator::Branch {
                condition: Operand::borrow(OwnerId(0)),
                then_edge: Edge {
                    id: EdgeId::UNASSIGNED,
                    target: BlockId(1),
                    args: vec![],
                    terminals: vec![edge_terminal(90, Terminal::Discard(OwnerId(0)))],
                },
                else_edge: Edge {
                    id: EdgeId::UNASSIGNED,
                    target: BlockId(2),
                    args: vec![],
                    terminals: vec![edge_terminal(91, Terminal::Discard(OwnerId(0)))],
                },
            },
        ),
        block(
            1,
            vec![],
            vec![define(1)],
            Terminator::Return {
                result: Operand::move_(OwnerId(1)),
            },
        ),
        block(
            2,
            vec![],
            vec![define(2)],
            Terminator::Return {
                result: Operand::move_(OwnerId(2)),
            },
        ),
    ];
    inconsistent_returns.units[2].owners = BTreeMap::from([
        (OwnerId(0), info(Prim::Bool, OwnerOrigin::Owned)),
        (OwnerId(1), info(Prim::Int64, OwnerOrigin::Owned)),
        (OwnerId(2), info(Prim::Bool, OwnerOrigin::Owned)),
    ]);
    assert!(matches!(
        super::verify::verify(&inconsistent_returns),
        Err(OwnershipError::FunctionReturnClass {
            unit,
            block: 2,
            ..
        }) if unit == "callee"
    ));
}

#[test]
fn direct_call_argument_class_matches_the_callable_body_parameter() {
    let mut program = direct_call_program(false, UnitId(2));
    program.units[1]
        .owners
        .insert(OwnerId(2), info(Prim::Int64, OwnerOrigin::Owned));
    program.units[1].blocks[0].ops.insert(
        0,
        Operation {
            id: OpId(20),
            role: OperationRole::Semantic,
            kind: define(2),
        },
    );
    let Op::Apply { schema, args, .. } = &mut program.units[1].blocks[0].ops[1].kind else {
        unreachable!()
    };
    schema.operands = vec![OwnershipUse::Borrow];
    args.push(Operand::borrow(OwnerId(2)));
    program.units[1].blocks[0].ops.push(Operation {
        id: OpId(21),
        role: OperationRole::Semantic,
        kind: Op::Discard { owner: OwnerId(2) },
    });
    program.units[2].blocks[0].params.push(BlockParam {
        owner: OwnerId(1),
        mode: ParamMode::Borrowed,
    });
    program.units[2].owners.insert(
        OwnerId(1),
        parameter_info(Prim::Int64, OwnerOrigin::ExternalBorrow),
    );
    super::verify::verify(&program).unwrap();

    for prim in [Prim::Int32, Prim::F32, Prim::F64] {
        let mut exact = program.clone();
        exact.units[1]
            .owners
            .insert(OwnerId(2), info(prim, OwnerOrigin::Owned));
        exact.units[2].owners.insert(
            OwnerId(1),
            parameter_info(prim, OwnerOrigin::ExternalBorrow),
        );
        super::verify::verify(&exact).unwrap();
    }

    let mut wrong_mode = program.clone();
    let Op::Apply { schema, args, .. } = &mut wrong_mode.units[1].blocks[0].ops[1].kind else {
        unreachable!()
    };
    schema.operands[0] = OwnershipUse::Move;
    args[0].use_ = OwnershipUse::Move;
    assert!(matches!(
        super::verify::verify(&wrong_mode),
        Err(OwnershipError::DirectCallSchema { unit: 2, .. })
    ));

    let mut wrong_bool_family = program.clone();
    wrong_bool_family.units[1]
        .owners
        .insert(OwnerId(2), info(Prim::Bool, OwnerOrigin::Owned));
    assert!(matches!(
        super::verify::verify(&wrong_bool_family),
        Err(OwnershipError::DirectCallArgumentClass {
            unit: 2,
            argument: 0,
            ..
        })
    ));

    let mut wrong_integer_width = program.clone();
    wrong_integer_width.units[1]
        .owners
        .insert(OwnerId(2), info(Prim::Int32, OwnerOrigin::Owned));
    assert!(matches!(
        super::verify::verify(&wrong_integer_width),
        Err(OwnershipError::DirectCallArgumentClass {
            unit: 2,
            argument: 0,
            ..
        })
    ));

    let mut wrong_float_width = program.clone();
    wrong_float_width.units[1]
        .owners
        .insert(OwnerId(2), info(Prim::F32, OwnerOrigin::Owned));
    wrong_float_width.units[2].owners.insert(
        OwnerId(1),
        parameter_info(Prim::F64, OwnerOrigin::ExternalBorrow),
    );
    assert!(matches!(
        super::verify::verify(&wrong_float_width),
        Err(OwnershipError::DirectCallArgumentClass {
            unit: 2,
            argument: 0,
            ..
        })
    ));

    let mut wrong_unit = program.clone();
    wrong_unit.units[1].owners.insert(
        OwnerId(2),
        OwnerInfo {
            ty: ConcreteHostType::Unit,
            class: ValueClass::NonHeap(NonHeapKind::Unit),
            placement: Placement::Value,
            origin: OwnerOrigin::Owned,
            names: Vec::new(),
        },
    );
    assert!(matches!(
        super::verify::verify(&wrong_unit),
        Err(OwnershipError::DirectCallArgumentClass {
            unit: 2,
            argument: 0,
            ..
        })
    ));

    let mut wrong_heap = program;
    wrong_heap.units[1]
        .owners
        .insert(OwnerId(2), info(Prim::String, OwnerOrigin::Owned));
    assert!(matches!(
        super::verify::verify(&wrong_heap),
        Err(OwnershipError::DirectCallArgumentClass {
            unit: 2,
            argument: 0,
            ..
        })
    ));
}

#[test]
fn callable_environment_borrows_are_not_direct_call_arguments() {
    let mut program = direct_call_program(false, UnitId(2));
    program.units[2].blocks[0].params.push(BlockParam {
        owner: OwnerId(1),
        mode: ParamMode::Borrowed,
    });
    program.units[2]
        .owners
        .insert(OwnerId(1), info(Prim::String, OwnerOrigin::ExternalBorrow));
    super::verify::verify(&program).unwrap();

    program.units[2].blocks[0].params[0].mode = ParamMode::Owned;
    assert!(matches!(
        super::verify::verify(&program),
        Err(OwnershipError::LoweringInvariant { unit, .. }) if unit == "callee"
    ));

    let mut wrong_origin = direct_call_program(false, UnitId(2));
    wrong_origin.units[2].blocks[0].params.push(BlockParam {
        owner: OwnerId(1),
        mode: ParamMode::Borrowed,
    });
    wrong_origin.units[2]
        .owners
        .insert(OwnerId(1), info(Prim::String, OwnerOrigin::Owned));
    assert!(matches!(
        super::verify::verify(&wrong_origin),
        Err(OwnershipError::LoweringInvariant { unit, .. }) if unit == "callee"
    ));

    let mut capture_before_formal = direct_call_program(false, UnitId(2));
    capture_before_formal.units[2].blocks[0].params = vec![
        BlockParam {
            owner: OwnerId(1),
            mode: ParamMode::Borrowed,
        },
        BlockParam {
            owner: OwnerId(2),
            mode: ParamMode::Owned,
        },
    ];
    capture_before_formal.units[2].owners.extend([
        (OwnerId(1), info(Prim::String, OwnerOrigin::ExternalBorrow)),
        (OwnerId(2), parameter_info(Prim::Int64, OwnerOrigin::Owned)),
    ]);
    assert!(matches!(
        super::verify::verify(&capture_before_formal),
        Err(OwnershipError::LoweringInvariant { unit, .. }) if unit == "callee"
    ));
}

#[test]
fn verifier_derives_a_sealed_live_heap_owner_bound() {
    let make = |count: u32| {
        let mut ops = (0..count).map(define).collect::<Vec<_>>();
        ops.extend((0..count).rev().map(|owner| Op::Drop {
            owner: Operand::move_(OwnerId(owner)),
        }));
        roots(
            vec![block(0, vec![], ops, Terminator::Exit)],
            (0..count)
                .map(|owner| (OwnerId(owner), info(Prim::String, OwnerOrigin::Owned)))
                .collect(),
        )
    };
    for expected in 0..=2 {
        let verification = super::verify::verify(&make(expected)).unwrap();
        assert_eq!(
            verification.live_set_bound().max_live_heap_owners(),
            expected as usize
        );
    }
}

#[test]
fn entry_borrow_can_be_copied_but_not_consumed() {
    let params = vec![BlockParam {
        owner: OwnerId(0),
        mode: ParamMode::EntryBorrow,
    }];
    let owners = BTreeMap::from([
        (OwnerId(0), info(Prim::String, OwnerOrigin::ExternalBorrow)),
        (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
    ]);
    let good = roots(
        vec![block(
            0,
            params.clone(),
            vec![
                Op::Copy {
                    dest: OwnerId(1),
                    source: Operand {
                        owner: OwnerId(0),
                        use_: OwnershipUse::Clone,
                    },
                },
                root(1),
            ],
            Terminator::Exit,
        )],
        owners.clone(),
    );
    verify_raw(good).unwrap();

    let bad = roots(
        vec![block(0, params, vec![root(0)], Terminator::Exit)],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::ExternalBorrow))]),
    );
    assert!(matches!(
        verify_raw(bad),
        Err(OwnershipError::BorrowConsumed { owner: 0, .. })
    ));
}

#[test]
fn owned_edges_join_through_one_fresh_block_parameter() {
    let join = Edge {
        id: EdgeId::UNASSIGNED,
        target: BlockId(1),
        args: vec![Operand {
            owner: OwnerId(1),
            use_: OwnershipUse::Move,
        }],
        terminals: Vec::new(),
    };
    let owners = BTreeMap::from([
        (OwnerId(0), info(Prim::Bool, OwnerOrigin::Owned)),
        (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        (OwnerId(2), info(Prim::String, OwnerOrigin::Owned)),
    ]);
    let blocks = vec![
        block(
            0,
            vec![],
            vec![define(0), define(1)],
            Terminator::Branch {
                condition: Operand {
                    owner: OwnerId(0),
                    use_: OwnershipUse::Borrow,
                },
                then_edge: join.clone(),
                else_edge: join,
            },
        ),
        block(
            1,
            vec![BlockParam {
                owner: OwnerId(2),
                mode: ParamMode::Owned,
            }],
            vec![Op::Discard { owner: OwnerId(0) }, root(2)],
            Terminator::Exit,
        ),
    ];
    verify_raw(roots(blocks, owners)).unwrap();
}

#[test]
fn owned_edge_rejects_a_borrow_disposition() {
    let edge = |use_| Edge {
        id: EdgeId::UNASSIGNED,
        target: BlockId(1),
        args: vec![Operand {
            owner: OwnerId(1),
            use_,
        }],
        terminals: Vec::new(),
    };
    let owners = BTreeMap::from([
        (OwnerId(0), info(Prim::Bool, OwnerOrigin::Owned)),
        (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        (OwnerId(2), info(Prim::String, OwnerOrigin::Owned)),
    ]);
    let blocks = vec![
        block(
            0,
            vec![],
            vec![define(0), define(1)],
            Terminator::Branch {
                condition: Operand {
                    owner: OwnerId(0),
                    use_: OwnershipUse::Borrow,
                },
                then_edge: edge(OwnershipUse::Move),
                else_edge: edge(OwnershipUse::Borrow),
            },
        ),
        block(
            1,
            vec![BlockParam {
                owner: OwnerId(2),
                mode: ParamMode::Owned,
            }],
            vec![root(2)],
            Terminator::Exit,
        ),
    ];
    assert!(matches!(
        verify_raw(roots(blocks, owners)),
        Err(OwnershipError::WrongUse { owner: 1, .. })
    ));
}

#[test]
fn moved_owner_cannot_be_used_again() {
    let program = roots(
        vec![block(
            0,
            vec![],
            vec![define(0), root(0), root(0)],
            Terminator::Exit,
        )],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );
    assert!(matches!(
        verify_raw(program),
        Err(OwnershipError::OwnerNotLive { owner: 0, .. })
    ));
}

#[test]
fn classification_is_total_for_opaque_first_class_function_values() {
    assert_eq!(
        classify(&ty(Prim::String), Placement::Value),
        Ok(ValueClass::Heap(HeapKind::String))
    );
    assert_eq!(
        ValueClass::NonHeap(NonHeapKind::Unit),
        classify(&ConcreteHostType::Unit, Placement::Value).unwrap()
    );
    let function = ConcreteHostType::Function(vec![], Box::new(ty(Prim::String)));
    assert_eq!(
        classify(&function, Placement::Value),
        Ok(ValueClass::NonHeap(NonHeapKind::FirstClassFunction))
    );
    assert_eq!(
        classify(
            &ConcreteHostType::Option(Box::new(function)),
            Placement::Value,
        ),
        Ok(ValueClass::Heap(HeapKind::Option))
    );
    assert_ne!(HeapKind::TensorStorage, HeapKind::Tensor);
}

#[test]
fn unreachable_block_is_rejected() {
    let program = roots(
        vec![
            block(0, vec![], vec![], Terminator::Exit),
            block(1, vec![], vec![], Terminator::Exit),
        ],
        BTreeMap::new(),
    );
    assert!(matches!(
        verify_raw(program),
        Err(OwnershipError::UnreachableBlock { block: 1, .. })
    ));
}

#[test]
fn host_payload_site_and_directive_universes_are_bijective() {
    let host = ConcreteHostProgram {
        globals: vec![HostBinding {
            name: "result".into(),
            display_name: None,
            display_roots: vec![HostDisplayRoot {
                name: "result".into(),
                path: Vec::new(),
            }],
            ty: ConcreteHostType::String,
            value: HostExpr::new(HostExprKind::String("value".into())),
        }],
        ..ConcreteHostProgram::default()
    };
    let manifest = RootManifest {
        entries: vec![RootEntry {
            name: "result".into(),
            path: Vec::new(),
            def_name: "result".into(),
            ty: Expr::Atom(Atom::Name("string".into()), Span::new(0, 0)),
            lane: Lane::Host,
            required_inputs: Default::default(),
            reasons: Vec::new(),
        }],
    };
    let program = roots(
        vec![block(0, vec![], vec![define(0), root(0)], Terminator::Exit)],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );
    let record = |index, kind, actions| HostSiteRecord {
        id: HostSiteId::from_index(index),
        unit: 0,
        kind,
        actions,
    };
    let valid = HostSiteMap {
        records: vec![
            record(0, HostSiteKind::Binding, vec![]),
            record(
                1,
                HostSiteKind::Expression,
                vec![HostSiteAction::Operation {
                    unit: 0,
                    block: BlockId(0),
                    operation: OpId(0),
                }],
            ),
            record(
                2,
                HostSiteKind::ManifestRoot,
                vec![
                    HostSiteAction::Root {
                        unit: 0,
                        manifest_index: Some(0),
                        owner: OwnerId(0),
                    },
                    HostSiteAction::Operation {
                        unit: 0,
                        block: BlockId(0),
                        operation: OpId(1),
                    },
                ],
            ),
            record(
                3,
                HostSiteKind::FunctionReturn,
                vec![HostSiteAction::Terminator {
                    unit: 0,
                    block: BlockId(0),
                }],
            ),
        ],
    };
    super::verify::verify_host_sites(&host, &manifest, &program, &valid).unwrap();

    let mut missing = valid.clone();
    missing.records[2]
        .actions
        .retain(|action| !matches!(action, HostSiteAction::Operation { .. }));
    assert!(matches!(
        super::verify::verify_host_sites(&host, &manifest, &program, &missing),
        Err(OwnershipError::HostSiteMap { .. })
    ));

    let mut missing_root = valid.clone();
    missing_root.records[2]
        .actions
        .retain(|action| !matches!(action, HostSiteAction::Root { .. }));
    assert!(matches!(
        super::verify::verify_host_sites(&host, &manifest, &program, &missing_root),
        Err(OwnershipError::HostSiteMap { .. })
    ));

    let mut duplicate_root = valid.clone();
    duplicate_root.records[2]
        .actions
        .push(HostSiteAction::Root {
            unit: 0,
            manifest_index: Some(0),
            owner: OwnerId(0),
        });
    assert!(matches!(
        super::verify::verify_host_sites(&host, &manifest, &program, &duplicate_root),
        Err(OwnershipError::HostSiteMap { .. })
    ));

    let mut extra = valid.clone();
    extra.records[3].actions.push(HostSiteAction::Operation {
        unit: 0,
        block: BlockId(0),
        operation: OpId(1),
    });
    assert!(matches!(
        super::verify::verify_host_sites(&host, &manifest, &program, &extra),
        Err(OwnershipError::HostSiteMap { .. })
    ));

    let mut mismatched = valid;
    mismatched.records.swap(0, 1);
    assert!(matches!(
        super::verify::verify_host_sites(&host, &manifest, &program, &mismatched),
        Err(OwnershipError::HostSiteMap { .. })
    ));
}

#[test]
fn payload_census_rejects_a_missing_match_option_binding_and_wrong_kind() {
    let int_ty = ConcreteHostType::Int64;
    let host = ConcreteHostProgram {
        globals: vec![HostBinding {
            name: "out".into(),
            display_name: None,
            display_roots: Vec::new(),
            ty: int_ty.clone(),
            value: HostExpr::new(HostExprKind::MatchOption {
                scrutinee: Box::new(HostExpr::new(HostExprKind::Var(
                    "maybe".into(),
                    ConcreteHostType::Option(Box::new(int_ty.clone())),
                ))),
                bind_name: "item".into(),
                some_expr: Box::new(HostExpr::new(HostExprKind::Var(
                    "item".into(),
                    int_ty.clone(),
                ))),
                none_expr: Box::new(HostExpr::new(HostExprKind::Int(0))),
                ty: int_ty,
            }),
        }],
        ..ConcreteHostProgram::default()
    };
    let manifest = RootManifest { entries: vec![] };
    let kinds = [
        HostSiteKind::Binding,
        HostSiteKind::Expression,
        HostSiteKind::Expression,
        HostSiteKind::MatchArm,
        HostSiteKind::MatchArm,
        HostSiteKind::Binding,
        HostSiteKind::Expression,
        HostSiteKind::Expression,
        HostSiteKind::FunctionReturn,
    ];
    let records = kinds
        .into_iter()
        .enumerate()
        .map(|(index, kind)| HostSiteRecord {
            id: HostSiteId::from_index(index),
            unit: 0,
            kind,
            actions: Vec::new(),
        })
        .collect::<Vec<_>>();
    let valid = HostSiteMap { records };
    super::verify::verify_host_payload_sites(&host, &manifest, &valid).unwrap();

    let mut missing_binding = valid.clone();
    missing_binding.records.remove(5);
    for (index, record) in missing_binding.records.iter_mut().enumerate() {
        record.id = HostSiteId::from_index(index);
    }
    assert!(matches!(
        super::verify::verify_host_payload_sites(&host, &manifest, &missing_binding),
        Err(OwnershipError::HostSiteMap { .. })
    ));

    let mut wrong_kind = valid;
    wrong_kind.records[5].kind = HostSiteKind::Argument;
    assert!(matches!(
        super::verify::verify_host_payload_sites(&host, &manifest, &wrong_kind),
        Err(OwnershipError::HostSiteMap { .. })
    ));
}

#[test]
fn host_payload_sites_and_actions_are_bound_to_their_structural_unit() {
    let host = ConcreteHostProgram {
        functions: vec![ConcreteHostFunction {
            name: "identity".into(),
            params: Vec::new(),
            ret_ty: ConcreteHostType::Unit,
            body: HostExpr::new(HostExprKind::Unit),
            tensor_helpers: Vec::new(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        ..ConcreteHostProgram::default()
    };
    let manifest = RootManifest { entries: vec![] };
    let record = |index, unit, kind, actions| HostSiteRecord {
        id: HostSiteId::from_index(index),
        unit,
        kind,
        actions,
    };
    let structural = HostSiteMap {
        records: vec![
            record(0, 0, HostSiteKind::FunctionReturn, vec![]),
            record(1, 1, HostSiteKind::FunctionEntry, vec![]),
            record(2, 1, HostSiteKind::FunctionReturn, vec![]),
            record(3, 1, HostSiteKind::Expression, vec![]),
        ],
    };
    super::verify::verify_host_payload_sites(&host, &manifest, &structural).unwrap();

    let mut wrong_payload_unit = structural;
    wrong_payload_unit.records[3].unit = 0;
    assert!(matches!(
        super::verify::verify_host_payload_sites(&host, &manifest, &wrong_payload_unit),
        Err(OwnershipError::HostSiteMap { .. })
    ));

    let program = RawProgram {
        units: vec![
            Unit {
                id: UnitId(0),
                name: "roots".into(),
                kind: UnitKind::Roots,
                schedule: ScheduleState::Phase2ScopeExit,
                callable_body: None,
                entry: BlockId(0),
                blocks: vec![block(0, vec![], vec![], Terminator::Exit)],
                owners: BTreeMap::new(),
            },
            Unit {
                id: UnitId(1),
                name: "identity".into(),
                kind: UnitKind::Function,
                schedule: ScheduleState::Phase2ScopeExit,
                callable_body: Some(CallableBody::new(BlockId(0))),
                entry: BlockId(0),
                blocks: vec![block(0, vec![], vec![], Terminator::Exit)],
                owners: BTreeMap::new(),
            },
        ],
    };
    let actions = HostSiteMap {
        records: vec![
            record(
                0,
                0,
                HostSiteKind::Expression,
                vec![HostSiteAction::Terminator {
                    unit: 0,
                    block: BlockId(0),
                }],
            ),
            record(
                1,
                1,
                HostSiteKind::Expression,
                vec![HostSiteAction::Terminator {
                    unit: 1,
                    block: BlockId(0),
                }],
            ),
        ],
    };
    super::verify::verify_host_actions(&program, &manifest, &actions).unwrap();

    let mut wrong_action_unit = actions;
    wrong_action_unit.records[1].unit = 0;
    assert!(matches!(
        super::verify::verify_host_actions(&program, &manifest, &wrong_action_unit),
        Err(OwnershipError::HostSiteMap { .. })
    ));
}

#[test]
fn control_and_root_actions_are_complete_unique_and_kind_checked() {
    let branch = roots(
        vec![
            block(
                0,
                vec![],
                vec![],
                Terminator::Branch {
                    condition: Operand::borrow(OwnerId(0)),
                    then_edge: Edge {
                        id: EdgeId::UNASSIGNED,
                        target: BlockId(1),
                        args: vec![],
                        terminals: Vec::new(),
                    },
                    else_edge: Edge {
                        id: EdgeId::UNASSIGNED,
                        target: BlockId(2),
                        args: vec![],
                        terminals: Vec::new(),
                    },
                },
            ),
            block(1, vec![], vec![], Terminator::Exit),
            block(2, vec![], vec![], Terminator::Exit),
        ],
        BTreeMap::from([(OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow))]),
    );
    let record = |index, kind, actions| HostSiteRecord {
        id: HostSiteId::from_index(index),
        unit: 0,
        kind,
        actions,
    };
    let controls = HostSiteMap {
        records: vec![
            record(
                0,
                HostSiteKind::Expression,
                vec![HostSiteAction::Terminator {
                    unit: 0,
                    block: BlockId(0),
                }],
            ),
            record(
                1,
                HostSiteKind::BranchEdge,
                vec![HostSiteAction::ControlEdge {
                    unit: 0,
                    edge: EdgeId(0),
                    source: BlockId(0),
                    target: BlockId(1),
                }],
            ),
            record(
                2,
                HostSiteKind::BranchEdge,
                vec![HostSiteAction::ControlEdge {
                    unit: 0,
                    edge: EdgeId(1),
                    source: BlockId(0),
                    target: BlockId(2),
                }],
            ),
            record(
                3,
                HostSiteKind::Expression,
                vec![HostSiteAction::Terminator {
                    unit: 0,
                    block: BlockId(1),
                }],
            ),
            record(
                4,
                HostSiteKind::Expression,
                vec![HostSiteAction::Terminator {
                    unit: 0,
                    block: BlockId(2),
                }],
            ),
        ],
    };
    super::verify::verify_host_actions(&branch, &RootManifest { entries: vec![] }, &controls)
        .unwrap();

    let mut missing = controls.clone();
    missing.records[1].actions.clear();
    assert!(
        super::verify::verify_host_actions(&branch, &RootManifest { entries: vec![] }, &missing,)
            .is_err()
    );

    let mut duplicate = controls.clone();
    duplicate.records[2].actions = duplicate.records[1].actions.clone();
    assert!(
        super::verify::verify_host_actions(&branch, &RootManifest { entries: vec![] }, &duplicate,)
            .is_err()
    );

    let mut wrong_kind = controls;
    wrong_kind.records[1].kind = HostSiteKind::Expression;
    assert!(super::verify::verify_host_actions(
        &branch,
        &RootManifest { entries: vec![] },
        &wrong_kind,
    )
    .is_err());
}

#[test]
fn dag_verification_checks_mutated_directives_and_terminal_completeness() {
    let ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let load = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let copied = dag.add_node(RiscOp::Copy, vec![load], ty.clone(), None);
    let dropped = dag.add_node(RiscOp::Drop, vec![copied], ty.clone(), None);
    let mut plan = DagOwnershipPlan::lower(&dag).unwrap();
    plan.verify(&dag).unwrap();
    plan.directives[2] = DagDirective::OwnedDrop {
        node: dropped,
        source: load,
    };
    assert!(matches!(
        plan.verify(&dag),
        Err(OwnershipError::DagDirectiveMap { .. })
    ));

    let mut borrowed_drop = Dag::new();
    let borrowed =
        borrowed_drop.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let dropped = borrowed_drop.add_node(RiscOp::Drop, vec![borrowed], ty.clone(), None);
    let mut plan = DagOwnershipPlan::lower(&borrowed_drop).unwrap();
    assert!(matches!(
        plan.directives[1],
        DagDirective::BorrowedDrop { .. }
    ));
    plan.directives[1] = DagDirective::OwnedDrop {
        node: dropped,
        source: borrowed,
    };
    assert!(matches!(
        plan.verify(&borrowed_drop),
        Err(OwnershipError::DagDirectiveMap { .. })
    ));

    let mut unterminated = Dag::new();
    let load = unterminated.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    unterminated.add_node(RiscOp::Neg, vec![load], ty, None);
    let mut plan = DagOwnershipPlan::lower(&unterminated).unwrap();
    plan.directives.pop();
    assert!(matches!(
        plan.verify(&unterminated),
        Err(OwnershipError::DagDirectiveMap { .. })
    ));

    let mut borrowed_realize = Dag::new();
    let borrowed = borrowed_realize.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let realized = borrowed_realize.add_node(
        RiscOp::Realize,
        vec![borrowed],
        TensorType::scalar_f32(),
        None,
    );
    borrowed_realize.add_root(realized);
    let mut plan = DagOwnershipPlan::lower(&borrowed_realize).unwrap();
    assert!(matches!(
        plan.directives[1],
        DagDirective::CloneProduce { .. }
    ));
    plan.directives[1] = DagDirective::MoveProduce {
        node: realized,
        source: borrowed,
    };
    assert!(matches!(
        plan.verify(&borrowed_realize),
        Err(OwnershipError::DagDirectiveMap { .. })
    ));

    let mut borrowed_store = Dag::new();
    let borrowed = borrowed_store.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let stored = borrowed_store.add_node(
        RiscOp::Store { name: "out".into() },
        vec![borrowed],
        TensorType::scalar_f32(),
        None,
    );
    borrowed_store.add_root(stored);
    let mut plan = DagOwnershipPlan::lower(&borrowed_store).unwrap();
    assert!(matches!(
        plan.directives[1],
        DagDirective::CloneStore { .. }
    ));
    plan.directives[1] = DagDirective::MoveStore {
        node: stored,
        source: borrowed,
    };
    assert!(matches!(
        plan.verify(&borrowed_store),
        Err(OwnershipError::DagDirectiveMap { .. })
    ));
}

#[test]
fn store_terminal_can_be_an_exported_root_but_drop_cannot() {
    let ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let mut stored = Dag::new();
    let load = stored.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let copied = stored.add_node(RiscOp::Copy, vec![load], ty.clone(), None);
    let output = stored.add_node(
        RiscOp::Store { name: "out".into() },
        vec![copied],
        ty.clone(),
        None,
    );
    stored.add_root(output);
    let plan = DagOwnershipPlan::lower(&stored).unwrap();
    plan.verify(&stored).unwrap();
    assert!(plan.render().contains("store-root n2"));

    let mut dropped = Dag::new();
    let value = dropped.add_node(
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        ty.clone(),
        None,
    );
    let terminal = dropped.add_node(RiscOp::Drop, vec![value], ty, None);
    dropped.add_root(terminal);
    let error = DagOwnershipPlan::lower(&dropped).unwrap_err();
    assert!(matches!(error, OwnershipError::LoweringInvariant { .. }));
    assert!(
        error
            .to_string()
            .contains("terminal and cannot be a DAG root")
    );
}

fn schedule(program: &mut RawProgram) -> Result<super::last_use::ScheduleFacts, OwnershipError> {
    super::last_use::schedule(
        program,
        &mut HostSiteMap {
            records: Vec::new(),
        },
    )
}

fn scheduled_block_terminal(block: &Block, owner: u32) -> Option<usize> {
    block.ops.iter().position(|operation| {
        operation.role == OperationRole::ScheduledScopeExit
            && match &operation.kind {
                Op::Drop { owner: dropped } => dropped.owner == OwnerId(owner),
                Op::Discard { owner: discarded } => *discarded == OwnerId(owner),
                _ => false,
            }
    })
}

fn typed_info(ty: ConcreteHostType, origin: OwnerOrigin) -> OwnerInfo {
    OwnerInfo {
        class: classify(&ty, Placement::Value).unwrap(),
        ty,
        placement: Placement::Value,
        origin,
        names: Vec::new(),
    }
}

#[test]
fn scheduler_places_straight_line_and_unused_owners_at_their_death_frontiers() {
    let mut entry = block(
        0,
        vec![],
        vec![define(0), borrow_op(0), define(1)],
        Terminator::Exit,
    );
    entry
        .ops
        .extend([provisional(90, 1, false), provisional(91, 0, true)]);
    let mut program = roots(
        vec![entry],
        BTreeMap::from([
            (OwnerId(0), info(Prim::String, OwnerOrigin::Owned)),
            (OwnerId(1), info(Prim::Int64, OwnerOrigin::Owned)),
        ]),
    );

    schedule(&mut program).unwrap();
    super::last_use::verify_canonical(&program).unwrap();
    let ops = &program.units[0].blocks[0].ops;
    assert_eq!(
        scheduled_block_terminal(&program.units[0].blocks[0], 0),
        Some(2)
    );
    assert_eq!(
        scheduled_block_terminal(&program.units[0].blocks[0], 1),
        Some(4)
    );
    assert_eq!(ops[2].id, OpId(91));
    assert_eq!(ops[4].id, OpId(90));

    let delayed = program.units[0].blocks[0].ops.remove(2);
    program.units[0].blocks[0].ops.push(delayed);
    assert!(matches!(
        super::last_use::verify_canonical(&program),
        Err(OwnershipError::NonCanonicalTerminal { owner: 0, .. })
    ));
}

#[test]
fn scheduler_splits_one_arm_death_and_rejects_omission_or_hoisting() {
    let branch = Block {
        id: BlockId(0),
        params: vec![BlockParam {
            owner: OwnerId(0),
            mode: ParamMode::EntryBorrow,
        }],
        ops: vec![Operation {
            id: OpId(0),
            role: OperationRole::Semantic,
            kind: define(1),
        }],
        terminator: Terminator::Branch {
            condition: Operand::borrow(OwnerId(0)),
            then_edge: Edge::new(EdgeId(10), BlockId(1), vec![]),
            else_edge: Edge::new(EdgeId(11), BlockId(2), vec![]),
        },
    };
    let then_block = block(
        1,
        vec![],
        vec![borrow_op(1)],
        Terminator::Jump(Edge::new(EdgeId(12), BlockId(3), vec![])),
    );
    let else_block = block(
        2,
        vec![],
        vec![],
        Terminator::Jump(Edge::new(EdgeId(13), BlockId(3), vec![])),
    );
    let mut join = block(3, vec![], vec![], Terminator::Exit);
    join.ops.push(provisional(90, 1, true));
    let base = roots(
        vec![branch, then_block, else_block, join],
        BTreeMap::from([
            (OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow)),
            (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        ]),
    );

    let mut duplicate_edge = base.clone();
    duplicate_edge.units[0].blocks[0]
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap()
        .id = EdgeId(10);
    assert!(matches!(
        super::verify::verify(&duplicate_edge),
        Err(OwnershipError::DuplicateIdentity {
            kind: "edge",
            id: 10,
            ..
        })
    ));

    let mut scheduled = base.clone();
    schedule(&mut scheduled).unwrap();
    super::last_use::verify_canonical(&scheduled).unwrap();
    assert_eq!(
        scheduled_block_terminal(&scheduled.units[0].blocks[1], 1),
        Some(1)
    );
    assert_eq!(
        scheduled.units[0].blocks[0]
            .terminator
            .edges()
            .find(|edge| edge.id == EdgeId(11))
            .unwrap()
            .terminals
            .iter()
            .map(|terminal| terminal.kind.owner())
            .collect::<Vec<_>>(),
        vec![OwnerId(1)]
    );

    let mut omitted = scheduled.clone();
    omitted.units[0].blocks[0]
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap()
        .terminals
        .clear();
    assert!(matches!(
        super::last_use::verify_canonical(&omitted),
        Err(OwnershipError::NonCanonicalTerminal { owner: 1, .. })
    ));

    let mut duplicated = scheduled.clone();
    let terminal = duplicated.units[0].blocks[0]
        .terminator
        .edges()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap()
        .terminals[0];
    duplicated.units[0].blocks[0]
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap()
        .terminals
        .push(EdgeTerminal {
            id: OpId(99),
            ..terminal
        });
    assert!(matches!(
        super::last_use::verify_canonical(&duplicated),
        Err(OwnershipError::NonCanonicalTerminal { owner: 1, .. })
    ));

    let mut wrong_edge = scheduled.clone();
    let terminal = wrong_edge.units[0].blocks[0]
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap()
        .terminals
        .pop()
        .unwrap();
    wrong_edge.units[0].blocks[0]
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(10))
        .unwrap()
        .terminals
        .push(terminal);
    assert!(matches!(
        super::last_use::verify_canonical(&wrong_edge),
        Err(OwnershipError::NonCanonicalTerminal { owner: 1, .. })
    ));

    let mut both_arms = base;
    both_arms.units[0].blocks[2].ops.push(Operation {
        id: OpId(2 << 16),
        role: OperationRole::Semantic,
        kind: borrow_op(1),
    });
    schedule(&mut both_arms).unwrap();
    let late_arm_terminal = both_arms.units[0].blocks[2].ops.pop().unwrap();
    assert_eq!(late_arm_terminal.role, OperationRole::ScheduledScopeExit);
    let edge = both_arms.units[0].blocks[0]
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap();
    edge.terminals.push(EdgeTerminal {
        id: late_arm_terminal.id,
        kind: Terminal::Drop(OwnerId(1)),
    });
    assert!(matches!(
        super::last_use::verify_canonical(&both_arms),
        Err(OwnershipError::NonCanonicalTerminal { owner: 1, .. })
    ));
}

#[test]
fn scheduler_preserves_owned_join_moves_and_rejects_drop_plus_move() {
    let mut program = roots(
        vec![
            block(
                0,
                vec![],
                vec![define(0)],
                Terminator::Jump(Edge::new(
                    EdgeId(10),
                    BlockId(1),
                    vec![Operand::move_(OwnerId(0))],
                )),
            ),
            block(
                1,
                vec![BlockParam {
                    owner: OwnerId(1),
                    mode: ParamMode::Owned,
                }],
                vec![root(1)],
                Terminator::Exit,
            ),
        ],
        BTreeMap::from([
            (OwnerId(0), info(Prim::String, OwnerOrigin::Owned)),
            (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        ]),
    );
    schedule(&mut program).unwrap();
    super::last_use::verify_canonical(&program).unwrap();

    program.units[0].blocks[0]
        .terminator
        .edges_mut()
        .next()
        .unwrap()
        .terminals
        .push(EdgeTerminal {
            id: OpId(91),
            kind: Terminal::Drop(OwnerId(0)),
        });
    assert!(matches!(
        super::last_use::verify_canonical(&program),
        Err(OwnershipError::CarriedOwnerDropped { owner: 0, .. })
    ));
}

#[test]
fn match_scrutinee_survives_projection_and_default_paths_cannot_leak() {
    let option_ty = ConcreteHostType::Option(Box::new(ConcreteHostType::String));
    let payload_class = classify(&ConcreteHostType::String, Placement::Value).unwrap();
    let match_block = Block {
        id: BlockId(0),
        params: vec![],
        ops: vec![Operation {
            id: OpId(0),
            role: OperationRole::Semantic,
            kind: define(0),
        }],
        terminator: Terminator::Match {
            scrutinee: Operand::borrow(OwnerId(0)),
            arms: vec![
                Edge::new(EdgeId(10), BlockId(1), vec![]),
                Edge::new(EdgeId(11), BlockId(2), vec![]),
            ],
        },
    };
    let some = block(
        1,
        vec![],
        vec![
            Op::Apply {
                dest: Some(OwnerId(1)),
                label: "option_payload".into(),
                kind: ApplyKind::Intrinsic,
                schema: OperationSchema::new(vec![OwnershipUse::Borrow], Some(payload_class)),
                args: vec![Operand::borrow(OwnerId(0))],
            },
            Op::Drop {
                owner: Operand::move_(OwnerId(1)),
            },
        ],
        Terminator::Jump(Edge::new(EdgeId(12), BlockId(3), vec![])),
    );
    let none = block(
        2,
        vec![],
        vec![],
        Terminator::Jump(Edge::new(EdgeId(13), BlockId(3), vec![])),
    );
    let mut join = block(3, vec![], vec![], Terminator::Exit);
    join.ops.push(provisional(90, 0, true));
    let mut program = roots(
        vec![match_block, some, none, join],
        BTreeMap::from([
            (OwnerId(0), typed_info(option_ty, OwnerOrigin::Owned)),
            (
                OwnerId(1),
                typed_info(ConcreteHostType::String, OwnerOrigin::Owned),
            ),
        ]),
    );

    schedule(&mut program).unwrap();
    super::last_use::verify_canonical(&program).unwrap();
    let projection = program.units[0].blocks[1]
        .ops
        .iter()
        .position(|operation| matches!(&operation.kind, Op::Apply { label, .. } if label == "option_payload"))
        .unwrap();
    assert_eq!(
        scheduled_block_terminal(&program.units[0].blocks[1], 0),
        Some(projection + 1)
    );
    let default_edge = program.units[0].blocks[0]
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap();
    assert_eq!(default_edge.terminals.len(), 1);
    default_edge.terminals.clear();
    assert!(matches!(
        super::last_use::verify_canonical(&program),
        Err(OwnershipError::NonCanonicalTerminal { owner: 0, .. })
    ));
}

#[test]
fn match_adt_default_receives_its_own_terminal() {
    let match_block = Block {
        id: BlockId(0),
        params: vec![],
        ops: vec![Operation {
            id: OpId(0),
            role: OperationRole::Semantic,
            kind: define(0),
        }],
        terminator: Terminator::Match {
            scrutinee: Operand::borrow(OwnerId(0)),
            arms: vec![
                Edge::new(EdgeId(10), BlockId(1), vec![]),
                Edge::new(EdgeId(11), BlockId(2), vec![]),
                Edge::new(EdgeId(12), BlockId(3), vec![]),
            ],
        },
    };
    let payload = Op::Apply {
        dest: None,
        label: "adt_payload:Some:0".into(),
        kind: ApplyKind::Intrinsic,
        schema: OperationSchema::new(vec![OwnershipUse::Borrow], None),
        args: vec![Operand::borrow(OwnerId(0))],
    };
    let mut join = block(4, vec![], vec![], Terminator::Exit);
    join.ops.push(provisional(90, 0, true));
    let mut program = roots(
        vec![
            match_block,
            block(
                1,
                vec![],
                vec![payload],
                Terminator::Jump(Edge::new(EdgeId(13), BlockId(4), vec![])),
            ),
            block(
                2,
                vec![],
                vec![],
                Terminator::Jump(Edge::new(EdgeId(14), BlockId(4), vec![])),
            ),
            block(
                3,
                vec![],
                vec![],
                Terminator::Jump(Edge::new(EdgeId(15), BlockId(4), vec![])),
            ),
            join,
        ],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );

    schedule(&mut program).unwrap();
    assert_eq!(
        scheduled_block_terminal(&program.units[0].blocks[1], 0),
        Some(1)
    );
    let default = program.units[0].blocks[0]
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(12))
        .unwrap();
    assert_eq!(default.terminals.len(), 1);
    default.terminals.clear();
    assert!(matches!(
        super::last_use::verify_canonical(&program),
        Err(OwnershipError::NonCanonicalTerminal { owner: 0, .. })
    ));
}

#[test]
fn diamond_uses_the_post_dominating_join_instead_of_lexical_block_order() {
    let branch = Block {
        id: BlockId(0),
        params: vec![BlockParam {
            owner: OwnerId(0),
            mode: ParamMode::EntryBorrow,
        }],
        ops: vec![Operation {
            id: OpId(0),
            role: OperationRole::Semantic,
            kind: define(1),
        }],
        terminator: Terminator::Branch {
            condition: Operand::borrow(OwnerId(0)),
            then_edge: Edge::new(EdgeId(10), BlockId(1), vec![]),
            else_edge: Edge::new(EdgeId(11), BlockId(2), vec![]),
        },
    };
    let left = block(
        1,
        vec![],
        vec![],
        Terminator::Jump(Edge::new(EdgeId(12), BlockId(3), vec![])),
    );
    let right = block(
        2,
        vec![],
        vec![],
        Terminator::Jump(Edge::new(EdgeId(13), BlockId(3), vec![])),
    );
    let mut join = block(3, vec![], vec![borrow_op(1)], Terminator::Exit);
    join.ops.push(provisional(90, 1, true));
    let mut program = roots(
        vec![branch, left, right, join],
        BTreeMap::from([
            (OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow)),
            (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        ]),
    );

    let facts = schedule(&mut program).unwrap();
    assert_eq!(
        facts.successor_count(
            UnitId(0),
            super::last_use::SchedulePoint::BeforeTerminator(BlockId(0)),
        ),
        Some(2)
    );
    assert_eq!(
        facts.predecessor_count(
            UnitId(0),
            super::last_use::SchedulePoint::BlockEntry(BlockId(3)),
        ),
        Some(2)
    );
    assert!(facts.postdominates(
        UnitId(0),
        super::last_use::SchedulePoint::AfterOperation {
            block: BlockId(3),
            operation: OpId(3 << 16),
        },
        super::last_use::SchedulePoint::BeforeTerminator(BlockId(0)),
    ));
    assert_eq!(
        scheduled_block_terminal(&program.units[0].blocks[3], 1),
        Some(1)
    );
    super::last_use::verify_canonical(&program).unwrap();
}

#[test]
fn excluded_loop_and_tail_shapes_fail_closed_and_stale_provisionals_are_rejected() {
    let mut loop_program = roots(
        vec![block(
            0,
            vec![],
            vec![],
            Terminator::Loop {
                list: Operand::borrow(OwnerId(0)),
                body_edge: Edge::new(EdgeId(10), BlockId(0), vec![]),
                exit_edge: Edge::new(EdgeId(11), BlockId(0), vec![]),
            },
        )],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::ExternalBorrow))]),
    );
    assert!(matches!(
        schedule(&mut loop_program),
        Err(OwnershipError::LastUseSchedulingDeferred {
            feature: "loop",
            ..
        })
    ));

    let mut tail_program = direct_call_program(false, UnitId(2));
    assert!(matches!(
        schedule(&mut tail_program),
        Err(OwnershipError::LastUseSchedulingDeferred {
            feature: "direct call",
            ..
        })
    ));

    let mut stale = roots(
        vec![block(0, vec![], vec![define(0)], Terminator::Exit)],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );
    stale.units[0].blocks[0].ops.push(provisional(90, 0, true));
    assert!(matches!(
        super::last_use::verify_canonical(&stale),
        Err(OwnershipError::StaleProvisionalTerminal { owner: 0, .. })
    ));
}

#[test]
fn scheduler_derives_terminal_kind_and_reverse_definition_order() {
    let observe_both = Op::Apply {
        dest: None,
        label: "observe-both".into(),
        kind: ApplyKind::Intrinsic,
        schema: OperationSchema::new(vec![OwnershipUse::Borrow, OwnershipUse::Borrow], None),
        args: vec![Operand::borrow(OwnerId(0)), Operand::borrow(OwnerId(1))],
    };
    let mut entry = block(
        0,
        vec![],
        vec![define(0), define(1), observe_both],
        Terminator::Exit,
    );
    entry
        .ops
        .extend([provisional(90, 0, true), provisional(91, 1, true)]);
    let mut program = roots(
        vec![entry],
        BTreeMap::from([
            (OwnerId(0), info(Prim::String, OwnerOrigin::Owned)),
            (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        ]),
    );
    schedule(&mut program).unwrap();
    assert_eq!(
        program.units[0].blocks[0].ops[3..]
            .iter()
            .map(|operation| terminal_from_test_op(&operation.kind))
            .collect::<Vec<_>>(),
        vec![Some(OwnerId(1)), Some(OwnerId(0))]
    );
    super::last_use::verify_canonical(&program).unwrap();

    program.units[0].blocks[0].ops.swap(3, 4);
    assert!(matches!(
        super::last_use::verify_canonical(&program),
        Err(OwnershipError::NonCanonicalTerminal { .. })
    ));

    let mut wrong_kind = roots(
        vec![block(0, vec![], vec![define(0)], Terminator::Exit)],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );
    wrong_kind.units[0].blocks[0]
        .ops
        .push(provisional(90, 0, false));
    assert!(matches!(
        schedule(&mut wrong_kind),
        Err(OwnershipError::HeapDiscard { owner: 0, .. })
    ));
}

fn terminal_from_test_op(op: &Op) -> Option<OwnerId> {
    match op {
        Op::Drop { owner } => Some(owner.owner),
        Op::Discard { owner } => Some(*owner),
        _ => None,
    }
}

#[test]
fn scheduler_rebuilds_host_sites_from_stable_operation_and_edge_identities() {
    let branch = Block {
        id: BlockId(0),
        params: vec![BlockParam {
            owner: OwnerId(0),
            mode: ParamMode::EntryBorrow,
        }],
        ops: vec![Operation {
            id: OpId(0),
            role: OperationRole::Semantic,
            kind: define(1),
        }],
        terminator: Terminator::Branch {
            condition: Operand::borrow(OwnerId(0)),
            then_edge: Edge::new(EdgeId(10), BlockId(1), vec![]),
            else_edge: Edge::new(EdgeId(11), BlockId(2), vec![]),
        },
    };
    let then_block = block(
        1,
        vec![],
        vec![borrow_op(1)],
        Terminator::Jump(Edge::new(EdgeId(12), BlockId(3), vec![])),
    );
    let else_block = block(
        2,
        vec![],
        vec![],
        Terminator::Jump(Edge::new(EdgeId(13), BlockId(3), vec![])),
    );
    let mut join = block(3, vec![], vec![], Terminator::Exit);
    join.ops.push(provisional(90, 1, true));
    let mut program = roots(
        vec![branch, then_block, else_block, join],
        BTreeMap::from([
            (OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow)),
            (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        ]),
    );
    let record = |index, kind, actions| HostSiteRecord {
        id: HostSiteId::from_index(index),
        unit: 0,
        kind,
        actions,
    };
    let mut sites = HostSiteMap {
        records: vec![
            record(
                0,
                HostSiteKind::Expression,
                vec![HostSiteAction::Operation {
                    unit: 0,
                    block: BlockId(3),
                    operation: OpId(90),
                }],
            ),
            record(
                1,
                HostSiteKind::Expression,
                vec![HostSiteAction::Operation {
                    unit: 0,
                    block: BlockId(1),
                    operation: OpId(1 << 16),
                }],
            ),
            record(
                2,
                HostSiteKind::BranchEdge,
                vec![HostSiteAction::ControlEdge {
                    unit: 0,
                    edge: EdgeId(11),
                    source: BlockId(0),
                    target: BlockId(2),
                }],
            ),
        ],
    };

    super::last_use::schedule(&mut program, &mut sites).unwrap();
    assert!(sites.records[0].actions.is_empty());
    assert_eq!(sites.records[1].actions.len(), 2);
    assert!(matches!(
        sites.records[1].actions[1],
        HostSiteAction::Operation {
            block: BlockId(1),
            operation: OpId(90),
            ..
        }
    ));
    let edge_terminal = program.units[0].blocks[0]
        .terminator
        .edges()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap()
        .terminals[0]
        .id;
    assert!(matches!(
        sites.records[2].actions.as_slice(),
        [
            HostSiteAction::ControlEdge {
                edge: EdgeId(11),
                ..
            },
            HostSiteAction::Operation {
                operation,
                ..
            }
        ] if *operation == edge_terminal
    ));
}

#[test]
fn unused_owned_block_parameter_dies_after_incoming_transfer_at_its_exact_site() {
    let mut target = block(
        1,
        vec![BlockParam {
            owner: OwnerId(1),
            mode: ParamMode::Owned,
        }],
        vec![],
        Terminator::Exit,
    );
    target.ops.push(provisional(90, 1, true));
    let mut program = roots(
        vec![
            block(
                0,
                vec![],
                vec![define(0)],
                Terminator::Jump(Edge::new(
                    EdgeId(10),
                    BlockId(1),
                    vec![Operand::move_(OwnerId(0))],
                )),
            ),
            target,
        ],
        BTreeMap::from([
            (OwnerId(0), info(Prim::String, OwnerOrigin::Owned)),
            (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        ]),
    );
    let record = |index, actions| HostSiteRecord {
        id: HostSiteId::from_index(index),
        unit: 0,
        kind: HostSiteKind::Expression,
        actions,
    };
    let mut sites = HostSiteMap {
        records: vec![
            record(
                0,
                vec![
                    HostSiteAction::Operation {
                        unit: 0,
                        block: BlockId(0),
                        operation: OpId(0),
                    },
                    HostSiteAction::Terminator {
                        unit: 0,
                        block: BlockId(0),
                    },
                ],
            ),
            record(
                1,
                vec![
                    HostSiteAction::Terminator {
                        unit: 0,
                        block: BlockId(1),
                    },
                    HostSiteAction::Operation {
                        unit: 0,
                        block: BlockId(1),
                        operation: OpId(90),
                    },
                ],
            ),
        ],
    };

    super::last_use::schedule(&mut program, &mut sites).unwrap();
    assert!(program.units[0].blocks[1].ops.is_empty());
    let edge = program.units[0].blocks[0]
        .terminator
        .edges()
        .next()
        .unwrap();
    assert_eq!(
        edge.terminals,
        vec![edge_terminal(90, Terminal::Drop(OwnerId(1)))]
    );
    assert!(matches!(
        sites.records[0].actions.as_slice(),
        [
            HostSiteAction::Operation {
                operation: OpId(0),
                ..
            },
            HostSiteAction::Terminator {
                block: BlockId(0),
                ..
            },
            HostSiteAction::Operation {
                operation: OpId(90),
                ..
            }
        ]
    ));
    super::verify::verify_host_actions(
        &program,
        &RootManifest {
            entries: Vec::new(),
        },
        &sites,
    )
    .unwrap();

    let mut before_transfer = sites;
    before_transfer.records[0].actions.swap(1, 2);
    assert!(matches!(
        super::verify::verify_host_actions(
            &program,
            &RootManifest {
                entries: Vec::new()
            },
            &before_transfer,
        ),
        Err(OwnershipError::HostSiteMap { .. })
    ));
}

#[test]
fn untyped_back_edge_shape_is_deferred_instead_of_self_certifying() {
    let mut program = roots(
        vec![block(
            0,
            vec![],
            vec![],
            Terminator::Jump(Edge::new(EdgeId(10), BlockId(0), vec![])),
        )],
        BTreeMap::new(),
    );
    assert!(matches!(
        schedule(&mut program),
        Err(OwnershipError::LastUseSchedulingDeferred {
            feature: "cyclic control flow",
            ..
        })
    ));
}
