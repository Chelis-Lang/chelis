use std::collections::BTreeMap;

use chelis_deep::ast::{Atom, Expr};
use chelis_deep::span::Span;
use chelis_types::manifest::{RootEntry, RootManifest};
use chelis_types::types::Lane;
use chelis_types::types::Prim;

use super::OwnershipError;
use super::classify::{HeapKind, NonHeapKind, Placement, ValueClass, classify};
use super::ir::{
    ApplyKind, Block, BlockId, BlockParam, Edge, HostSiteAction, HostSiteId, HostSiteKind,
    HostSiteMap, HostSiteRecord, Op, OpId, Operand, Operation, OperationSchema, OwnerId, OwnerInfo,
    OwnerOrigin, OwnershipProgram as RawProgram, OwnershipUse, ParamMode, Terminal, Terminator,
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

fn block(id: u32, params: Vec<BlockParam>, ops: Vec<Op>, terminator: Terminator) -> Block {
    Block {
        id: BlockId(id),
        params,
        ops: ops
            .into_iter()
            .enumerate()
            .map(|(index, kind)| Operation {
                id: OpId((id << 16) | u32::try_from(index).unwrap()),
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
            function_schema: None,
            entry: BlockId(0),
            blocks,
            owners,
        }],
    }
}

fn edge(target: u32, args: Vec<Operand>) -> Edge {
    Edge {
        target: BlockId(target),
        args,
        terminals: Vec::new(),
    }
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
                            target: BlockId(1),
                            args: Vec::new(),
                            terminals: vec![
                                Terminal::Drop(OwnerId(1)),
                                Terminal::Discard(OwnerId(2)),
                            ],
                        },
                        else_edge: Edge {
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
        Terminal::Drop(OwnerId(1)),
        Terminal::Discard(OwnerId(2)),
    ]))
    .unwrap();
    assert!(matches!(
        verify_raw(make(Vec::new())),
        Err(OwnershipError::JoinMismatch { block: 1, .. })
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
                function_schema: None,
                entry: BlockId(0),
                blocks: vec![block(0, vec![], vec![], Terminator::Exit)],
                owners: BTreeMap::new(),
            },
            Unit {
                id: UnitId(1),
                name: "caller".into(),
                kind: UnitKind::Function,
                function_schema: Some(OperationSchema::new(
                    Vec::new(),
                    Some(ValueClass::NonHeap(NonHeapKind::Scalar(Prim::Int64))),
                )),
                entry: BlockId(0),
                blocks: caller_blocks,
                owners: caller_owners,
            },
            Unit {
                id: UnitId(2),
                name: "callee".into(),
                kind: UnitKind::Function,
                function_schema: Some(OperationSchema::new(
                    Vec::new(),
                    Some(ValueClass::NonHeap(NonHeapKind::Scalar(Prim::Int64))),
                )),
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
fn verifier_derives_a_sealed_live_heap_owner_bound() {
    let program = roots(
        vec![block(
            0,
            vec![],
            vec![
                define(0),
                define(1),
                Op::Drop {
                    owner: Operand::move_(OwnerId(1)),
                },
                Op::Drop {
                    owner: Operand::move_(OwnerId(0)),
                },
            ],
            Terminator::Exit,
        )],
        BTreeMap::from([
            (OwnerId(0), info(Prim::String, OwnerOrigin::Owned)),
            (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        ]),
    );
    let verification = super::verify::verify(&program).unwrap();
    assert_eq!(verification.live_set_bound().max_live_heap_owners(), 2);
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
                function_schema: None,
                entry: BlockId(0),
                blocks: vec![block(0, vec![], vec![], Terminator::Exit)],
                owners: BTreeMap::new(),
            },
            Unit {
                id: UnitId(1),
                name: "identity".into(),
                kind: UnitKind::Function,
                function_schema: Some(OperationSchema::new(
                    Vec::new(),
                    Some(ValueClass::NonHeap(NonHeapKind::Unit)),
                )),
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
                        target: BlockId(1),
                        args: vec![],
                        terminals: Vec::new(),
                    },
                    else_edge: Edge {
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
                    source: BlockId(0),
                    target: BlockId(1),
                }],
            ),
            record(
                2,
                HostSiteKind::BranchEdge,
                vec![HostSiteAction::ControlEdge {
                    unit: 0,
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
