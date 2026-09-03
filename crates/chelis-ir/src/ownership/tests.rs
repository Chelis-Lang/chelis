use std::collections::BTreeMap;

use chelis_types::types::Prim;

use super::classify::{HeapKind, NonHeapKind, Placement, ValueClass, classify};
use super::ir::{
    Block, BlockId, BlockParam, Edge, Op, Operand, OwnerId, OwnerInfo, OwnerOrigin,
    OwnershipProgram as RawProgram, OwnershipUse, ParamMode, Terminator, Unit, UnitKind,
};
use super::{OwnershipError, OwnershipProgram, verify_ownership};
use crate::host_type_state::ConcreteHostType;

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
        ops,
        terminator,
    }
}

fn roots(blocks: Vec<Block>, owners: BTreeMap<OwnerId, OwnerInfo>) -> OwnershipProgram {
    OwnershipProgram(RawProgram {
        units: vec![Unit {
            name: "roots".into(),
            kind: UnitKind::Roots,
            entry: BlockId(0),
            blocks,
            owners,
        }],
    })
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
        verify_ownership(program).unwrap().render(),
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
        verify_ownership(program),
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
    verify_ownership(program).unwrap();
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
        verify_ownership(program),
        Err(OwnershipError::WrongUse { owner: 0, .. })
    ));
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
    verify_ownership(good).unwrap();

    let bad = roots(
        vec![block(0, params, vec![root(0)], Terminator::Exit)],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::ExternalBorrow))]),
    );
    assert!(matches!(
        verify_ownership(bad),
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
            vec![root(2)],
            Terminator::Exit,
        ),
    ];
    verify_ownership(roots(blocks, owners)).unwrap();
}

#[test]
fn owned_edge_rejects_a_borrow_disposition() {
    let edge = |use_| Edge {
        target: BlockId(1),
        args: vec![Operand {
            owner: OwnerId(1),
            use_,
        }],
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
        verify_ownership(roots(blocks, owners)),
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
        verify_ownership(program),
        Err(OwnershipError::OwnerNotLive { owner: 0, .. })
    ));
}

#[test]
fn classification_is_total_and_rejects_function_containers() {
    assert_eq!(
        classify(&ty(Prim::String), Placement::Value),
        Ok(ValueClass::Heap(HeapKind::String))
    );
    assert_eq!(
        ValueClass::NonHeap(NonHeapKind::Unit),
        classify(&ConcreteHostType::Unit, Placement::Value).unwrap()
    );
    let function = ConcreteHostType::Function(vec![], Box::new(ty(Prim::String)));
    assert!(
        classify(
            &ConcreteHostType::Option(Box::new(function)),
            Placement::Value
        )
        .is_err()
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
        verify_ownership(program),
        Err(OwnershipError::UnreachableBlock { block: 1, .. })
    ));
}
