use std::collections::BTreeMap;

use chelis_deep::ast::{Atom, Expr};
use chelis_deep::span::Span;
use chelis_numeric_core::prim::Prim;
use chelis_types::manifest::{RootEntry, RootManifest};
use chelis_types::types::Lane;

use super::OwnershipError;
use super::classify::{HeapKind, NonHeapKind, Placement, ValueClass, classify};
use super::ir::{
    ApplyKind, Block, BlockId, BlockParam, CallableBody, Edge, EdgeId, EdgeTerminal,
    HostSiteAction, HostSiteId, HostSiteKind, HostSiteMap, HostSiteRecord, Op, OpId, Operand,
    Operation, OperationRole, OperationSchema, OwnerId, OwnerInfo, OwnerOrigin,
    OwnershipProgram as RawProgram, OwnershipUse, ParamMode, ScheduleState, Terminal, Terminator,
    Unit, UnitId, UnitKind,
};
use super::{DagDirective, DagOwnershipPlan, LiveByteBound};
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
        span_id: None,
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
        span_id: None,
    }
}

fn value_info(ty: ConcreteHostType, origin: OwnerOrigin) -> OwnerInfo {
    OwnerInfo {
        class: classify(&ty, Placement::Value).unwrap(),
        ty,
        placement: Placement::Value,
        origin,
        names: Vec::new(),
        span_id: None,
    }
}

fn fixed_tensor_info(elements: usize) -> OwnerInfo {
    value_info(
        ConcreteHostType::Tensor(TensorType {
            dims: vec![DimInfo::Lit(elements)],
            precision: Prim::F32,
        }),
        OwnerOrigin::Owned,
    )
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
            captured_globals: BTreeMap::new(),
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
    let sites = HostSiteMap::from_records(vec![
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
    ]);
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
        super::verified_host_action(&program, &verification, &sites.raw_records()[0].actions[0])
            .unwrap();
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

/// chelis#2122: a join mismatch has to say WHICH owner the two paths disagree
/// about, and what it is called in source. Two owner-id sets alone are not
/// actionable.
#[test]
fn join_mismatch_names_the_owners_the_paths_disagree_about() {
    let make = |named: bool| {
        let mut second = info(Prim::Int64, OwnerOrigin::Owned);
        if named {
            second.names = vec!["carry".into()];
        }
        let mut first = info(Prim::String, OwnerOrigin::Owned);
        if named {
            first.names = vec!["text".into()];
        }
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
                            terminals: vec![edge_terminal(92, Terminal::Drop(OwnerId(1)))],
                        },
                    },
                ),
                block(1, vec![], vec![], Terminator::Exit),
            ],
            BTreeMap::from([
                (OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow)),
                (OwnerId(1), first),
                (OwnerId(2), second),
            ]),
        )
    };

    // The whole message, not fragments of it: fragment assertions cannot see
    // the two directions being swapped, the counts being swapped, or stray
    // whitespace in the format literal.
    assert_eq!(
        verify_raw(make(true)).unwrap_err().to_string(),
        "block b1 in `roots` is reached with inconsistent live owners: \
         live only on this path: %2[carry]; live only on the path already \
         verified: none (2 live here, 1 earlier)"
            .replace("         ", " ")
    );

    // Negative parity: with no source names, the message still identifies the
    // owner by id and prints no empty bracket.
    let anonymous = verify_raw(make(false)).unwrap_err().to_string();
    assert!(
        anonymous.contains("%2") && !anonymous.contains("[]"),
        "an unnamed owner is identified by id alone: {anonymous}"
    );
}

/// chelis#2122: when lowering recorded a span for the owner, the join mismatch
/// names the source region beside the binding name. Without one it prints the
/// owner exactly as before, with no placeholder.
#[test]
fn join_mismatch_reports_the_span_when_lowering_recorded_one() {
    let make = |span: Option<&str>| {
        let mut carried = info(Prim::Int64, OwnerOrigin::Owned);
        carried.names = vec!["carry".into()];
        carried.span_id = span.map(str::to_string);
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
                            terminals: vec![edge_terminal(92, Terminal::Drop(OwnerId(1)))],
                        },
                    },
                ),
                block(1, vec![], vec![], Terminator::Exit),
            ],
            BTreeMap::from([
                (OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow)),
                (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
                (OwnerId(2), carried),
            ]),
        )
    };

    assert_eq!(
        verify_raw(make(Some("surf:120..135")))
            .unwrap_err()
            .to_string(),
        "block b1 in `roots` is reached with inconsistent live owners: \
         live only on this path: %2[carry]@surf:120..135; live only on the path \
         already verified: none (2 live here, 1 earlier)"
            .replace("         ", " ")
    );

    assert_eq!(
        verify_raw(make(None)).unwrap_err().to_string(),
        "block b1 in `roots` is reached with inconsistent live owners: \
         live only on this path: %2[carry]; live only on the path already \
         verified: none (2 live here, 1 earlier)"
            .replace("         ", " ")
    );
}

/// chelis#2122: a span id is an opaque producer string. One that is empty, or
/// that carries the separators this single-line message is built from, must not
/// reach the diagnostic: it would end the label in a dangling `@`, or forge a
/// clause or a second owner. Such an id renders as no span at all.
#[test]
fn join_mismatch_drops_a_span_id_that_could_restructure_the_message() {
    let render_with = |span: &str| {
        let mut carried = info(Prim::Int64, OwnerOrigin::Owned);
        carried.names = vec!["carry".into()];
        carried.span_id = Some(span.to_string());
        verify_raw(roots(
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
                            terminals: vec![edge_terminal(92, Terminal::Drop(OwnerId(1)))],
                        },
                    },
                ),
                block(1, vec![], vec![], Terminator::Exit),
            ],
            BTreeMap::from([
                (OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow)),
                (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
                (OwnerId(2), carried),
            ]),
        ))
        .unwrap_err()
        .to_string()
    };

    let clean = render_with("surf:120..135");
    assert!(clean.contains("%2[carry]@surf:120..135"), "{clean}");

    for hostile in [
        "",
        "   ",
        "surf:1..2; live only on the path already verified: %99[forged]",
        "surf:1..2, %98[forged]",
        "surf:1..2\nINJECTED",
    ] {
        let rendered = render_with(hostile);
        assert!(
            rendered.contains("%2[carry];"),
            "an id that cannot be rendered safely leaves the owner unadorned, got: {rendered}"
        );
        assert!(
            !rendered.contains("forged") && !rendered.contains("INJECTED"),
            "no part of the id reaches the message: {rendered}"
        );
        assert_eq!(
            rendered.lines().count(),
            1,
            "the diagnostic stays one line: {rendered}"
        );
    }
}

/// The two sides are not interchangeable: an owner live only on the path being
/// verified now must not be reported as live only on the path verified before,
/// and the counts follow the same order (chelis#2122 red team, F2).
#[test]
fn join_mismatch_reports_each_direction_on_its_own_side() {
    let mut alpha = info(Prim::String, OwnerOrigin::Owned);
    alpha.names = vec!["alpha".into()];
    let mut beta = info(Prim::String, OwnerOrigin::Owned);
    beta.names = vec!["beta".into(), "beta_alias".into()];
    let gamma = info(Prim::String, OwnerOrigin::Owned);
    let program = roots(
        vec![
            block(
                0,
                vec![BlockParam {
                    owner: OwnerId(0),
                    mode: ParamMode::EntryBorrow,
                }],
                vec![define(1), define(2), define(3)],
                Terminator::Branch {
                    condition: Operand::borrow(OwnerId(0)),
                    then_edge: Edge {
                        id: EdgeId::UNASSIGNED,
                        target: BlockId(1),
                        args: Vec::new(),
                        terminals: vec![edge_terminal(90, Terminal::Drop(OwnerId(1)))],
                    },
                    else_edge: Edge {
                        id: EdgeId::UNASSIGNED,
                        target: BlockId(1),
                        args: Vec::new(),
                        terminals: vec![
                            edge_terminal(91, Terminal::Drop(OwnerId(2))),
                            edge_terminal(92, Terminal::Drop(OwnerId(3))),
                        ],
                    },
                },
            ),
            block(1, vec![], vec![], Terminator::Exit),
        ],
        BTreeMap::from([
            (OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow)),
            (OwnerId(1), alpha),
            (OwnerId(2), beta),
            (OwnerId(3), gamma),
        ]),
    );

    assert_eq!(
        verify_raw(program).unwrap_err().to_string(),
        "block b1 in `roots` is reached with inconsistent live owners: \
         live only on this path: %1[alpha]; live only on the path already \
         verified: %2[beta,beta_alias], %3 (2 live here, 3 earlier)"
            .replace("         ", " ")
    );
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
                captured_globals: BTreeMap::new(),
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
                captured_globals: BTreeMap::new(),
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
                captured_globals: BTreeMap::new(),
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
            span_id: None,
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
    program.units[2]
        .captured_globals
        .insert(OwnerId(1), "g".into());
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
    wrong_origin.units[2]
        .captured_globals
        .insert(OwnerId(1), "g".into());
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
    capture_before_formal.units[2]
        .captured_globals
        .insert(OwnerId(1), "g".into());
    assert!(matches!(
        super::verify::verify(&capture_before_formal),
        Err(OwnershipError::LoweringInvariant { unit, detail })
            if unit == "callee" && detail.contains("follows captured environment parameters")
    ));
}

/// chelis#3839: every callable environment parameter carries the identity of
/// the global it captured, and only environment parameters do. The C
/// projection reads that record instead of searching the owner's alias names.
#[test]
fn every_environment_parameter_records_exactly_its_captured_global() {
    let captured = || {
        let mut program = direct_call_program(false, UnitId(2));
        program.units[2].blocks[0].params.push(BlockParam {
            owner: OwnerId(1),
            mode: ParamMode::Borrowed,
        });
        program.units[2]
            .owners
            .insert(OwnerId(1), info(Prim::String, OwnerOrigin::ExternalBorrow));
        program
    };
    let disagrees = |program: &RawProgram| {
        matches!(
            super::verify::verify(program),
            Err(OwnershipError::LoweringInvariant { unit, detail })
                if unit == "callee" && detail.contains("disagree with the captured-global records")
        )
    };

    let mut recorded = captured();
    recorded.units[2]
        .captured_globals
        .insert(OwnerId(1), "@chelis_global_67".into());
    super::verify::verify(&recorded).unwrap();

    // An environment parameter without a record.
    assert!(disagrees(&captured()));

    // A record for an owner that is not an environment parameter.
    let mut stray = recorded.clone();
    stray.units[2]
        .captured_globals
        .insert(OwnerId(0), "g".into());
    assert!(disagrees(&stray));

    // The roots unit captures nothing.
    let mut roots_record = recorded.clone();
    roots_record.units[0]
        .captured_globals
        .insert(OwnerId(0), "g".into());
    assert!(matches!(
        super::verify::verify(&roots_record),
        Err(OwnershipError::LoweringInvariant { unit, detail })
            if unit == "roots" && detail.contains("records a captured global")
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
fn verified_live_byte_bound_uses_exact_tensor_payload_costs() {
    let program = roots(
        vec![block(
            0,
            vec![],
            vec![
                define(0),
                Op::Drop {
                    owner: Operand::move_(OwnerId(0)),
                },
            ],
            Terminator::Exit,
        )],
        BTreeMap::from([(OwnerId(0), fixed_tensor_info(2))]),
    );

    let verified = super::verify::verify(&program).unwrap();
    assert_eq!(
        verified.live_set_bound().max_live_bytes,
        LiveByteBound::Exact(8),
        "two f32 elements carry eight logical TensorStorage bytes; the descriptor is zero-byte ledger metadata"
    );
}

#[test]
fn verified_live_byte_bound_marks_dynamic_heap_costs_unknown() {
    let program = roots(
        vec![block(
            0,
            vec![],
            vec![
                define(0),
                Op::Drop {
                    owner: Operand::move_(OwnerId(0)),
                },
            ],
            Terminator::Exit,
        )],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );

    let verified = super::verify::verify(&program).unwrap();
    assert_eq!(
        verified.live_set_bound().max_live_bytes,
        LiveByteBound::Unknown,
        "a string's runtime len+1 allocation cannot be derived from its owner type"
    );
}

#[test]
fn verified_live_byte_bound_marks_runtime_tensor_dimensions_unknown() {
    let dynamic = value_info(
        ConcreteHostType::Tensor(TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        }),
        OwnerOrigin::Owned,
    );
    let program = roots(
        vec![block(
            0,
            vec![],
            vec![
                define(0),
                Op::Drop {
                    owner: Operand::move_(OwnerId(0)),
                },
            ],
            Terminator::Exit,
        )],
        BTreeMap::from([(OwnerId(0), dynamic)]),
    );

    let verified = super::verify::verify(&program).unwrap();
    assert_eq!(
        verified.live_set_bound().max_live_bytes,
        LiveByteBound::Unknown
    );
}

/// A tensor whose byte size exceeds u64 has no finite representable bound.
/// The summary says so; it is never a build failure, because the bound is an
/// analysis result and the allocation's own checked admission owns the trap
/// (chelis#3418).
#[test]
fn verified_live_byte_bound_overflow_is_unbounded() {
    let too_large = value_info(
        ConcreteHostType::Tensor(TensorType {
            dims: vec![DimInfo::Lit(usize::MAX)],
            precision: Prim::F32,
        }),
        OwnerOrigin::Owned,
    );
    let program = roots(
        vec![block(
            0,
            vec![],
            vec![
                define(0),
                Op::Drop {
                    owner: Operand::move_(OwnerId(0)),
                },
            ],
            Terminator::Exit,
        )],
        BTreeMap::from([(OwnerId(0), too_large)]),
    );

    let verified = super::verify::verify(&program).unwrap();
    assert_eq!(
        verified.live_set_bound().max_live_bytes,
        LiveByteBound::Unbounded
    );
}

/// Two owned heap tensors whose byte costs each fit u64 and whose sum does
/// not: every allocation is representable, so the program is valid and the
/// summary is unbounded rather than an error (chelis#3418).
///
/// The summation inside `live_byte_cost` is what overflows here, rather than
/// the per-operation transient accounting, because a `DirectCall` destination
/// deliberately skips that transient (a callee's peak is composed separately).
/// The post-operation live-set sum is therefore the first addition to see both
/// owners.
#[test]
fn verified_live_byte_summation_overflow_is_unbounded() {
    // 2^61 f32 elements is 2^63 bytes: each owner converts, the pair does not.
    let half = 1usize << 61;
    let body = block(
        0,
        Vec::new(),
        vec![
            define(0),
            Op::Apply {
                dest: Some(OwnerId(1)),
                label: "second live tensor".into(),
                kind: ApplyKind::DirectCall { callee: UnitId(1) },
                schema: OperationSchema::new(Vec::new(), Some(ValueClass::Heap(HeapKind::Tensor))),
                args: Vec::new(),
            },
            Op::Drop {
                owner: Operand::move_(OwnerId(0)),
            },
        ],
        Terminator::Return {
            result: Operand::move_(OwnerId(1)),
        },
    );
    let program = RawProgram {
        units: vec![
            Unit {
                id: UnitId(0),
                name: "roots".into(),
                kind: UnitKind::Roots,
                schedule: ScheduleState::Phase2ScopeExit,
                callable_body: None,
                entry: BlockId(0),
                blocks: vec![block(0, Vec::new(), Vec::new(), Terminator::Exit)],
                owners: BTreeMap::new(),
                captured_globals: BTreeMap::new(),
            },
            Unit {
                id: UnitId(1),
                name: "summed".into(),
                kind: UnitKind::Function,
                schedule: ScheduleState::Phase2ScopeExit,
                callable_body: Some(CallableBody::new(BlockId(0))),
                entry: BlockId(0),
                blocks: vec![body],
                owners: BTreeMap::from([
                    (OwnerId(0), fixed_tensor_info(half)),
                    (OwnerId(1), fixed_tensor_info(half)),
                ]),
                captured_globals: BTreeMap::new(),
            },
        ],
    };

    let verified = super::verify::verify(&program).unwrap();
    assert_eq!(
        verified.live_set_bound().max_live_bytes,
        LiveByteBound::Unbounded
    );
}

#[test]
fn verified_live_byte_bound_rejects_a_deferred_tensor_dtype() {
    let deferred = value_info(
        ConcreteHostType::Tensor(TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::F8e4m3,
        }),
        OwnerOrigin::Owned,
    );
    let program = roots(
        vec![block(
            0,
            vec![],
            vec![
                define(0),
                Op::Drop {
                    owner: Operand::move_(OwnerId(0)),
                },
            ],
            Terminator::Exit,
        )],
        BTreeMap::from([(OwnerId(0), deferred)]),
    );

    assert!(matches!(
        super::verify::verify(&program),
        Err(OwnershipError::LiveByteBoundDType {
            owner: 0,
            dtype: "f8e4m3",
            ..
        })
    ));
}

#[test]
fn verified_live_byte_bound_safely_overapproximates_aliasing_clones() {
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
            (OwnerId(0), fixed_tensor_info(2)),
            (OwnerId(1), fixed_tensor_info(2)),
        ]),
    );

    let verified = super::verify::verify(&program).unwrap();
    assert_eq!(
        verified.live_set_bound().max_live_bytes,
        LiveByteBound::Exact(16),
        "two logical owners alias one eight-byte storage allocation, so owner-summing is a safe upper bound rather than ledger equality"
    );
}

#[test]
fn verified_live_byte_bound_composes_caller_carry_plus_callee_peak() {
    let mut program = direct_call_program(false, UnitId(2));
    program.units[1]
        .owners
        .insert(OwnerId(2), fixed_tensor_info(2));
    program.units[1].blocks[0].ops.insert(
        0,
        Operation {
            id: OpId(20),
            role: OperationRole::Semantic,
            kind: define(2),
        },
    );
    program.units[1].blocks[0].ops.push(Operation {
        id: OpId(21),
        role: OperationRole::Semantic,
        kind: Op::Drop {
            owner: Operand::move_(OwnerId(2)),
        },
    });
    program.units[2]
        .owners
        .insert(OwnerId(1), fixed_tensor_info(4));
    program.units[2].blocks[0].ops.insert(
        0,
        Operation {
            id: OpId(20),
            role: OperationRole::Semantic,
            kind: define(1),
        },
    );
    program.units[2].blocks[0].ops.push(Operation {
        id: OpId(21),
        role: OperationRole::Semantic,
        kind: Op::Drop {
            owner: Operand::move_(OwnerId(1)),
        },
    });

    let verified = super::verify::verify(&program).unwrap();
    assert_eq!(
        verified.live_set_bound().max_live_bytes,
        LiveByteBound::Exact(24),
        "the caller's live eight-byte tensor must be added to the callee's sixteen-byte local peak"
    );
}

fn recursive_byte_bound_program(drop_before_call: bool) -> RawProgram {
    let tensor = Operation {
        id: OpId(10),
        role: OperationRole::Semantic,
        kind: define(0),
    };
    let terminal = Operation {
        id: OpId(11),
        role: OperationRole::Semantic,
        kind: Op::Drop {
            owner: Operand::move_(OwnerId(0)),
        },
    };
    let call = Operation {
        id: OpId(12),
        role: OperationRole::Semantic,
        kind: Op::Apply {
            dest: Some(OwnerId(1)),
            label: "diagnostic recursion label".into(),
            kind: ApplyKind::DirectCall { callee: UnitId(1) },
            schema: OperationSchema::new(
                Vec::new(),
                Some(ValueClass::NonHeap(NonHeapKind::Scalar(Prim::Int64))),
            ),
            args: Vec::new(),
        },
    };
    let ops = if drop_before_call {
        vec![tensor, terminal, call]
    } else {
        vec![tensor, call, terminal]
    };
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
                captured_globals: BTreeMap::new(),
            },
            Unit {
                id: UnitId(1),
                name: "recursive".into(),
                kind: UnitKind::Function,
                schedule: ScheduleState::Phase2ScopeExit,
                callable_body: Some(CallableBody::new(BlockId(0))),
                entry: BlockId(0),
                blocks: vec![Block {
                    id: BlockId(0),
                    params: Vec::new(),
                    ops,
                    terminator: Terminator::Return {
                        result: Operand::move_(OwnerId(1)),
                    },
                }],
                owners: BTreeMap::from([
                    (OwnerId(0), fixed_tensor_info(4)),
                    (OwnerId(1), info(Prim::Int64, OwnerOrigin::Owned)),
                ]),
                captured_globals: BTreeMap::new(),
            },
        ],
    }
}

#[test]
fn zero_carry_recursive_scc_has_a_finite_verified_byte_bound() {
    let verified = super::verify::verify(&recursive_byte_bound_program(true)).unwrap();
    assert_eq!(
        verified.live_set_bound().max_live_bytes,
        LiveByteBound::Exact(16)
    );
}

#[test]
fn positive_carry_recursive_scc_is_verified_unbounded() {
    let verified = super::verify::verify(&recursive_byte_bound_program(false)).unwrap();
    assert_eq!(
        verified.live_set_bound().max_live_bytes,
        LiveByteBound::Unbounded
    );
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
    let valid = HostSiteMap::from_records(vec![
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
    ]);
    super::verify::verify_host_sites(&host, &manifest, &program, &valid).unwrap();

    let mut missing = valid.clone();
    missing.raw_records_mut()[2]
        .actions
        .retain(|action| !matches!(action, HostSiteAction::Operation { .. }));
    assert!(matches!(
        super::verify::verify_host_sites(&host, &manifest, &program, &missing),
        Err(OwnershipError::HostSiteMap { .. })
    ));

    let mut missing_root = valid.clone();
    missing_root.raw_records_mut()[2]
        .actions
        .retain(|action| !matches!(action, HostSiteAction::Root { .. }));
    assert!(matches!(
        super::verify::verify_host_sites(&host, &manifest, &program, &missing_root),
        Err(OwnershipError::HostSiteMap { .. })
    ));

    let mut duplicate_root = valid.clone();
    duplicate_root.raw_records_mut()[2]
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
    extra.raw_records_mut()[3]
        .actions
        .push(HostSiteAction::Operation {
            unit: 0,
            block: BlockId(0),
            operation: OpId(1),
        });
    assert!(matches!(
        super::verify::verify_host_sites(&host, &manifest, &program, &extra),
        Err(OwnershipError::HostSiteMap { .. })
    ));

    let mut mismatched = valid;
    mismatched.raw_records_mut().swap(0, 1);
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
    let valid = HostSiteMap::from_records(records);
    super::verify::verify_host_payload_sites(&host, &manifest, &valid).unwrap();

    let mut missing_binding = valid.clone();
    missing_binding.raw_records_mut().remove(5);
    for (index, record) in missing_binding.raw_records_mut().iter_mut().enumerate() {
        record.id = HostSiteId::from_index(index);
    }
    assert!(matches!(
        super::verify::verify_host_payload_sites(&host, &manifest, &missing_binding),
        Err(OwnershipError::HostSiteMap { .. })
    ));

    let mut wrong_kind = valid;
    wrong_kind.raw_records_mut()[5].kind = HostSiteKind::Argument;
    assert!(matches!(
        super::verify::verify_host_payload_sites(&host, &manifest, &wrong_kind),
        Err(OwnershipError::HostSiteMap { .. })
    ));
}

#[test]
fn host_payload_sites_and_actions_are_bound_to_their_structural_unit() {
    let host = ConcreteHostProgram {
        functions: vec![ConcreteHostFunction {
            helper_result_claim_axes: Vec::new(),
            result_claim: None,
            entry_claims: Vec::new(),
            entry_contract: Default::default(),
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
    let structural = HostSiteMap::from_records(vec![
        record(0, 0, HostSiteKind::FunctionReturn, vec![]),
        record(1, 1, HostSiteKind::FunctionEntry, vec![]),
        record(2, 1, HostSiteKind::FunctionReturn, vec![]),
        record(3, 1, HostSiteKind::Expression, vec![]),
    ]);
    super::verify::verify_host_payload_sites(&host, &manifest, &structural).unwrap();

    let mut wrong_payload_unit = structural;
    wrong_payload_unit.raw_records_mut()[3].unit = 0;
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
                captured_globals: BTreeMap::new(),
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
                captured_globals: BTreeMap::new(),
            },
        ],
    };
    let actions = HostSiteMap::from_records(vec![
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
    ]);
    super::verify::verify_host_actions(&program, &manifest, &actions).unwrap();

    let mut wrong_action_unit = actions;
    wrong_action_unit.raw_records_mut()[1].unit = 0;
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
    let controls = HostSiteMap::from_records(vec![
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
    ]);
    super::verify::verify_host_actions(&branch, &RootManifest { entries: vec![] }, &controls)
        .unwrap();

    let mut missing = controls.clone();
    missing.raw_records_mut()[1].actions.clear();
    assert!(
        super::verify::verify_host_actions(&branch, &RootManifest { entries: vec![] }, &missing,)
            .is_err()
    );

    let mut duplicate = controls.clone();
    duplicate.raw_records_mut()[2].actions = duplicate.raw_records_mut()[1].actions.clone();
    assert!(
        super::verify::verify_host_actions(&branch, &RootManifest { entries: vec![] }, &duplicate,)
            .is_err()
    );

    let mut wrong_kind = controls;
    wrong_kind.raw_records_mut()[1].kind = HostSiteKind::Expression;
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
    let decl = dag.declare("test");
    let load = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let copied = dag.add_node(decl, RiscOp::Copy, vec![load], ty.clone(), None);
    let dropped = dag.add_node(decl, RiscOp::Drop, vec![copied], ty.clone(), None);
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
    let borrowed_drop_decl = borrowed_drop.declare("test");
    let borrowed = borrowed_drop.add_node(
        borrowed_drop_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let dropped = borrowed_drop.add_node(
        borrowed_drop_decl,
        RiscOp::Drop,
        vec![borrowed],
        ty.clone(),
        None,
    );
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
    let unterminated_decl = unterminated.declare("test");
    let load = unterminated.add_node(
        unterminated_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    unterminated.add_node(unterminated_decl, RiscOp::Neg, vec![load], ty, None);
    let mut plan = DagOwnershipPlan::lower(&unterminated).unwrap();
    plan.directives.pop();
    assert!(matches!(
        plan.verify(&unterminated),
        Err(OwnershipError::DagDirectiveMap { .. })
    ));

    let mut borrowed_realize = Dag::new();
    let borrowed_realize_decl = borrowed_realize.declare("test");
    let borrowed = borrowed_realize.add_node(
        borrowed_realize_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let realized = borrowed_realize.add_node(
        borrowed_realize_decl,
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
    let borrowed_store_decl = borrowed_store.declare("test");
    let borrowed = borrowed_store.add_node(
        borrowed_store_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let stored = borrowed_store.add_node(
        borrowed_store_decl,
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
    let stored_decl = stored.declare("test");
    let load = stored.add_node(
        stored_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let copied = stored.add_node(stored_decl, RiscOp::Copy, vec![load], ty.clone(), None);
    let output = stored.add_node(
        stored_decl,
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
    let dropped_decl = dropped.declare("test");
    let value = dropped.add_node(
        dropped_decl,
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        ty.clone(),
        None,
    );
    let terminal = dropped.add_node(dropped_decl, RiscOp::Drop, vec![value], ty, None);
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
    super::last_use::schedule(program, &mut HostSiteMap::from_records(Vec::new()))
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
        span_id: None,
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
fn branch_frontier_combines_a_fixed_consume_with_an_inserted_drop() {
    let entry = Block {
        id: BlockId(0),
        params: vec![BlockParam {
            owner: OwnerId(1),
            mode: ParamMode::EntryBorrow,
        }],
        ops: vec![Operation {
            id: OpId(0),
            role: OperationRole::Semantic,
            kind: define(0),
        }],
        terminator: Terminator::Branch {
            condition: Operand::borrow(OwnerId(1)),
            then_edge: Edge::new(EdgeId(10), BlockId(1), vec![]),
            else_edge: Edge::new(EdgeId(11), BlockId(2), vec![]),
        },
    };
    let fixed = block(
        1,
        vec![],
        vec![Op::Drop {
            owner: Operand::move_(OwnerId(0)),
        }],
        Terminator::Exit,
    );
    let mut needs_inserted = block(2, vec![], vec![], Terminator::Exit);
    needs_inserted.ops.push(provisional(90, 0, true));
    let mut program = roots(
        vec![entry, fixed, needs_inserted],
        BTreeMap::from([
            (OwnerId(0), info(Prim::String, OwnerOrigin::Owned)),
            (OwnerId(1), info(Prim::Bool, OwnerOrigin::ExternalBorrow)),
        ]),
    );

    schedule(&mut program).unwrap();
    assert!(matches!(
        program.units[0].blocks[1].ops.as_slice(),
        [Operation {
            role: OperationRole::Semantic,
            kind: Op::Drop { owner },
            ..
        }] if owner.owner == OwnerId(0)
    ));
    let inserted = program.units[0].blocks[0]
        .terminator
        .edges()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap();
    assert_eq!(
        inserted.terminals,
        vec![edge_terminal(90, Terminal::Drop(OwnerId(0)))]
    );
    super::last_use::verify_canonical(&program).unwrap();

    let mut omitted = program;
    omitted.units[0].blocks[0]
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap()
        .terminals
        .clear();
    assert!(matches!(
        super::last_use::verify_canonical(&omitted),
        Err(OwnershipError::NonCanonicalTerminal { owner: 0, .. })
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
    super::last_use::require_postdominates_for_test(
        &program.units[0],
        super::last_use::SchedulePoint::AfterOperation {
            block: BlockId(3),
            operation: OpId(3 << 16),
        },
        super::last_use::SchedulePoint::BeforeTerminator(BlockId(0)),
    )
    .unwrap();
    assert!(matches!(
        super::last_use::require_postdominates_for_test(
            &program.units[0],
            super::last_use::SchedulePoint::Edge(EdgeId(12)),
            super::last_use::SchedulePoint::BeforeTerminator(BlockId(0)),
        ),
        Err(OwnershipError::LoweringInvariant { detail, .. })
            if detail.contains("does not post-dominate")
    ));
    assert_eq!(
        scheduled_block_terminal(&program.units[0].blocks[3], 1),
        Some(1)
    );
    super::last_use::verify_canonical(&program).unwrap();
}

#[test]
fn postdominance_queries_scale_over_long_straight_line_cfgs() {
    const OP_COUNT: u32 = 4_096;
    let program = roots(
        vec![block(
            0,
            vec![],
            (0..OP_COUNT).map(|_| borrow_op(0)).collect(),
            Terminator::Exit,
        )],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::ExternalBorrow))]),
    );
    let unit = &program.units[0];
    let entry = super::last_use::SchedulePoint::BlockEntry(BlockId(0));
    let first = super::last_use::SchedulePoint::AfterOperation {
        block: BlockId(0),
        operation: OpId(0),
    };
    let last = super::last_use::SchedulePoint::AfterOperation {
        block: BlockId(0),
        operation: OpId(OP_COUNT - 1),
    };

    super::last_use::require_postdominates_for_test(unit, last, entry).unwrap();
    assert!(matches!(
        super::last_use::require_postdominates_for_test(unit, first, last),
        Err(OwnershipError::LoweringInvariant { detail, .. })
            if detail.contains("does not post-dominate")
    ));
}

#[test]
fn stale_provisionals_are_rejected() {
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
fn scheduler_drops_iteration_local_owner_before_back_edge_and_rejects_delay() {
    let mut body = block(
        2,
        vec![],
        vec![define(1), borrow_op(1)],
        Terminator::Jump(Edge::new(EdgeId(13), BlockId(1), vec![])),
    );
    body.ops.push(provisional(90, 1, true));
    let mut program = roots(
        vec![
            block(
                0,
                vec![BlockParam {
                    owner: OwnerId(0),
                    mode: ParamMode::EntryBorrow,
                }],
                vec![],
                Terminator::Jump(Edge::new(EdgeId(10), BlockId(1), vec![])),
            ),
            block(
                1,
                vec![],
                vec![],
                Terminator::Loop {
                    list: Operand::borrow(OwnerId(0)),
                    body_edge: Edge::new(EdgeId(11), BlockId(2), vec![]),
                    exit_edge: Edge::new(EdgeId(12), BlockId(3), vec![]),
                },
            ),
            body,
            block(3, vec![], vec![], Terminator::Exit),
        ],
        BTreeMap::from([
            (OwnerId(0), info(Prim::String, OwnerOrigin::ExternalBorrow)),
            (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        ]),
    );

    schedule(&mut program).unwrap();
    let body = &program.units[0].blocks[2];
    assert_eq!(scheduled_block_terminal(body, 1), Some(2));
    super::last_use::verify_canonical(&program).unwrap();

    let delayed = program.units[0].blocks[2].ops.pop().unwrap();
    assert_eq!(delayed.role, OperationRole::ScheduledScopeExit);
    program.units[0].blocks[2]
        .terminator
        .edges_mut()
        .next()
        .unwrap()
        .terminals
        .push(edge_terminal(90, Terminal::Drop(OwnerId(1))));
    assert!(matches!(
        super::last_use::verify_canonical(&program),
        Err(OwnershipError::NonCanonicalTerminal { owner: 1, .. })
    ));
}

#[test]
fn scheduler_keeps_owned_loop_list_through_body_and_drops_only_on_exit_edge() {
    let mut exit = block(3, vec![], vec![], Terminator::Exit);
    exit.ops.push(provisional(90, 0, true));
    let mut program = roots(
        vec![
            block(
                0,
                vec![],
                vec![define(0)],
                Terminator::Jump(Edge::new(EdgeId(10), BlockId(1), vec![])),
            ),
            block(
                1,
                vec![],
                vec![],
                Terminator::Loop {
                    list: Operand::borrow(OwnerId(0)),
                    body_edge: Edge::new(EdgeId(11), BlockId(2), vec![]),
                    exit_edge: Edge::new(EdgeId(12), BlockId(3), vec![]),
                },
            ),
            block(
                2,
                vec![],
                vec![borrow_op(0)],
                Terminator::Jump(Edge::new(EdgeId(13), BlockId(1), vec![])),
            ),
            exit,
        ],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );

    schedule(&mut program).unwrap();
    let header = &program.units[0].blocks[1];
    let body_edge = header
        .terminator
        .edges()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap();
    let exit_edge = header
        .terminator
        .edges()
        .find(|edge| edge.id == EdgeId(12))
        .unwrap();
    assert!(body_edge.terminals.is_empty());
    assert_eq!(
        exit_edge.terminals,
        vec![edge_terminal(90, Terminal::Drop(OwnerId(0)))]
    );
    super::last_use::verify_canonical(&program).unwrap();

    let mut wrong = program;
    let header = &mut wrong.units[0].blocks[1];
    let terminal = header
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(12))
        .unwrap()
        .terminals
        .pop()
        .unwrap();
    header
        .terminator
        .edges_mut()
        .find(|edge| edge.id == EdgeId(11))
        .unwrap()
        .terminals
        .push(terminal);
    assert!(matches!(
        super::last_use::verify_canonical(&wrong),
        Err(OwnershipError::NonCanonicalTerminal { owner: 0, .. })
    ));
}

#[test]
fn scheduler_terminates_nonargument_frame_owner_before_typed_tail_call() {
    let mut program = direct_call_program(false, UnitId(2));
    program.units[1]
        .owners
        .insert(OwnerId(2), info(Prim::String, OwnerOrigin::Owned));
    program.units[1].blocks[0].ops.insert(
        0,
        Operation {
            id: OpId(40),
            role: OperationRole::Semantic,
            kind: define(2),
        },
    );
    program.units[1].blocks[0]
        .ops
        .push(provisional(90, 2, true));

    schedule(&mut program).unwrap();
    let caller = &program.units[1].blocks[0];
    let drop = scheduled_block_terminal(caller, 2).unwrap();
    let call = caller
        .ops
        .iter()
        .position(|operation| {
            matches!(
                operation.kind,
                Op::Apply {
                    kind: ApplyKind::DirectCall { .. },
                    ..
                }
            )
        })
        .unwrap();
    assert!(drop < call, "tail-frame drop must precede direct call");
    super::last_use::verify_canonical(&program).unwrap();

    let mut delayed = program;
    let terminal = delayed.units[1].blocks[0].ops.remove(drop);
    let call = delayed.units[1].blocks[0]
        .ops
        .iter()
        .position(|operation| {
            matches!(
                operation.kind,
                Op::Apply {
                    kind: ApplyKind::DirectCall { .. },
                    ..
                }
            )
        })
        .unwrap();
    delayed.units[1].blocks[0].ops.insert(call + 1, terminal);
    assert!(matches!(
        super::last_use::verify_canonical(&delayed),
        Err(OwnershipError::NonCanonicalTerminal { owner: 2, .. })
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
    let mut sites = HostSiteMap::from_records(vec![
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
    ]);

    super::last_use::schedule(&mut program, &mut sites).unwrap();
    assert!(sites.raw_records_mut()[0].actions.is_empty());
    assert_eq!(sites.raw_records_mut()[1].actions.len(), 2);
    assert!(matches!(
        sites.raw_records_mut()[1].actions[1],
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
        sites.raw_records_mut()[2].actions.as_slice(),
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
fn scheduler_rebuilds_a_tail_terminal_before_the_direct_call_action() {
    let mut program = direct_call_program(false, UnitId(2));
    program.units[1]
        .owners
        .insert(OwnerId(2), info(Prim::String, OwnerOrigin::Owned));
    program.units[1].blocks[0].ops.insert(
        0,
        Operation {
            id: OpId(20),
            role: OperationRole::Semantic,
            kind: define(2),
        },
    );
    program.units[1].blocks[1]
        .ops
        .push(provisional(90, 2, true));
    let mut sites = HostSiteMap::from_records(vec![
        HostSiteRecord {
            id: HostSiteId::from_index(0),
            unit: 1,
            kind: HostSiteKind::Expression,
            actions: vec![
                HostSiteAction::Operation {
                    unit: 1,
                    block: BlockId(0),
                    operation: OpId(20),
                },
                HostSiteAction::Operation {
                    unit: 1,
                    block: BlockId(0),
                    operation: OpId(0),
                },
            ],
        },
        HostSiteRecord {
            id: HostSiteId::from_index(1),
            unit: 1,
            kind: HostSiteKind::FunctionReturn,
            actions: vec![HostSiteAction::Operation {
                unit: 1,
                block: BlockId(1),
                operation: OpId(90),
            }],
        },
    ]);

    super::last_use::schedule(&mut program, &mut sites).unwrap();
    assert!(matches!(
        sites.raw_records_mut()[0].actions.as_slice(),
        [
            HostSiteAction::Operation {
                operation: OpId(20),
                ..
            },
            HostSiteAction::Operation {
                operation: OpId(90),
                ..
            },
            HostSiteAction::Operation {
                operation: OpId(0),
                ..
            }
        ]
    ));
    assert!(sites.raw_records_mut()[1].actions.is_empty());
}

#[test]
fn host_site_rebuild_qualifies_equal_operation_ids_by_unit() {
    let mut root_block = block(0, vec![], vec![define(0)], Terminator::Exit);
    root_block.ops.push(provisional(90, 0, true));
    let mut function_block = block(
        0,
        vec![],
        vec![define(0)],
        Terminator::Return {
            result: Operand::move_(OwnerId(0)),
        },
    );
    function_block.ops[0].id = OpId(90);
    let mut program = roots(
        vec![root_block],
        BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
    );
    program.units.push(Unit {
        id: UnitId(1),
        name: "function".into(),
        kind: UnitKind::Function,
        schedule: ScheduleState::Phase2ScopeExit,
        callable_body: Some(CallableBody::new(BlockId(0))),
        entry: BlockId(0),
        blocks: vec![function_block],
        owners: BTreeMap::from([(OwnerId(0), info(Prim::String, OwnerOrigin::Owned))]),
        captured_globals: BTreeMap::new(),
    });
    let record = |index, unit, operation| HostSiteRecord {
        id: HostSiteId::from_index(index),
        unit,
        kind: HostSiteKind::Expression,
        actions: vec![HostSiteAction::Operation {
            unit,
            block: BlockId(0),
            operation,
        }],
    };
    let mut sites = HostSiteMap::from_records(vec![
        record(0, 0, OpId(90)),
        record(1, 0, OpId(0)),
        record(2, 1, OpId(90)),
    ]);

    super::last_use::schedule(&mut program, &mut sites).unwrap();
    assert!(
        sites.raw_records_mut()[0].actions.is_empty(),
        "the roots unit's provisional o90 must be removed"
    );
    assert!(matches!(
        sites.raw_records_mut()[1].actions.as_slice(),
        [
            HostSiteAction::Operation {
                unit: 0,
                operation: OpId(0),
                ..
            },
            HostSiteAction::Operation {
                unit: 0,
                operation: OpId(90),
                ..
            }
        ]
    ));
    assert!(matches!(
        sites.raw_records_mut()[2].actions.as_slice(),
        [HostSiteAction::Operation {
            unit: 1,
            operation: OpId(90),
            ..
        }]
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
    let mut sites = HostSiteMap::from_records(vec![
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
    ]);

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
        sites.raw_records_mut()[0].actions.as_slice(),
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
    before_transfer.raw_records_mut()[0].actions.swap(1, 2);
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
fn nonterminating_back_edge_fails_closed_without_an_exit_path() {
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
        Err(OwnershipError::LastUseSchedulingUnsupported { unit, feature })
            if unit == "roots" && feature.contains("nonterminating")
    ));
}

#[test]
fn cyclic_jump_with_a_reachable_exit_uses_the_fixed_point() {
    let mut body = block(
        2,
        vec![],
        vec![define(1), borrow_op(1)],
        Terminator::Jump(Edge::new(EdgeId(13), BlockId(1), vec![])),
    );
    body.ops.push(provisional(90, 1, true));
    let mut program = roots(
        vec![
            block(
                0,
                vec![BlockParam {
                    owner: OwnerId(0),
                    mode: ParamMode::EntryBorrow,
                }],
                vec![],
                Terminator::Jump(Edge::new(EdgeId(10), BlockId(1), vec![])),
            ),
            block(
                1,
                vec![],
                vec![],
                Terminator::Branch {
                    condition: Operand::borrow(OwnerId(0)),
                    then_edge: Edge::new(EdgeId(11), BlockId(2), vec![]),
                    else_edge: Edge::new(EdgeId(12), BlockId(3), vec![]),
                },
            ),
            body,
            block(3, vec![], vec![], Terminator::Exit),
        ],
        BTreeMap::from([
            (OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow)),
            (OwnerId(1), info(Prim::String, OwnerOrigin::Owned)),
        ]),
    );

    schedule(&mut program).unwrap();
    assert_eq!(
        scheduled_block_terminal(&program.units[0].blocks[2], 1),
        Some(2),
        "the iteration-local owner must die before the ordinary Jump back-edge"
    );
    super::last_use::verify_canonical(&program).unwrap();
}

/// chelis#2205: the container-consumer table's key is the row's heap kind on
/// both the result and the named operand, not a bare result-equals-operand
/// comparison.
///
/// Evidentiary status: DISPOSITION LOCK for the representation the #2225
/// review arrived at. Each negative row names the one comparison that rejects
/// it, and deleting that comparison from `container_consumer_operand` makes
/// exactly that row match. Without the kind on the row, `builtin:concat`
/// against a dictionary matches and routes a dictionary to
/// `chelis_list_concat_owned`; the kind is what makes that shape not a row by
/// construction rather than an emitter guard firing after the scheduler has
/// already retired the operand's terminal.
#[test]
fn a_consumer_row_matches_only_its_own_heap_kind() {
    use super::ir::container_consumer_operand;
    let list = ValueClass::Heap(HeapKind::List);
    let dict = ValueClass::Heap(HeapKind::Dict);
    let tensor = ValueClass::Heap(HeapKind::Tensor);
    let string = ValueClass::Heap(HeapKind::String);

    for (label, class) in [
        ("builtin:append", list),
        ("builtin:concat", list),
        ("builtin:skip", list),
        ("builtin:take", list),
        ("builtin:dict_insert", dict),
        ("builtin:dict_merge", dict),
        ("builtin:dict_remove", dict),
        ("builtin:string_concat", string),
    ] {
        assert_eq!(
            container_consumer_operand(label, Some(class), |_| Some(class)),
            Some(0),
            "{label} consumes its operand at its own kind"
        );
    }

    // Rejected by the result comparison: the row's kind is List.
    assert_eq!(
        container_consumer_operand("builtin:concat", Some(dict), |_| Some(dict)),
        None,
        "a dictionary application never matches the list-kinded `concat` row"
    );
    // The same test on chelis#2334's row. `skip` is the one row whose
    // result is a sub-range of its operand rather than a grown copy, so
    // it is worth pinning that membership still turns on the kind and
    // not on the shape.
    assert_eq!(
        container_consumer_operand("builtin:skip", Some(dict), |_| Some(dict)),
        None,
        "a dictionary application never matches the list-kinded `skip` row"
    );
    // chelis#3724's `take`, the prefix twin of `skip`, keyed the same way.
    assert_eq!(
        container_consumer_operand("builtin:take", Some(dict), |_| Some(dict)),
        None,
        "a dictionary application never matches the list-kinded `take` row"
    );
    // Rejected by the result comparison: the row's kind is Dict.
    assert_eq!(
        container_consumer_operand("builtin:dict_merge", Some(list), |_| Some(list)),
        None,
        "a list application never matches the dictionary-kinded `dict_merge` row"
    );
    // Rejected by the operand comparison: the result carries the row's kind
    // but the named operand does not.
    assert_eq!(
        container_consumer_operand("builtin:dict_insert", Some(dict), |_| Some(list)),
        None,
        "the named operand must carry the row's kind too"
    );
    // Rejected by the result comparison, and the shape tensor `concat`
    // actually has: a `List[tensor]` of parts producing a tensor.
    assert_eq!(
        container_consumer_operand("builtin:concat", Some(tensor), |_| Some(list)),
        None,
        "tensor `concat` shares the label and is still not a row"
    );
    // Rejected by the result comparison: the row's kind is String. A list
    // `concat` and a string `concat` are different labels, but a reader who
    // dropped the kind would have `builtin:string_concat` accept a list and
    // route it to `chelis_string_concat_owned`.
    assert_eq!(
        container_consumer_operand("builtin:string_concat", Some(list), |_| Some(list)),
        None,
        "a list application never matches the string-kinded `string_concat` row"
    );
    // Rejected by the result comparison: the row's kind is List, and a string
    // is the operand class `string_concat` shares with nothing else here.
    assert_eq!(
        container_consumer_operand("builtin:append", Some(string), |_| Some(string)),
        None,
        "a string application never matches the list-kinded `append` row"
    );
    // Rejected by the label lookup: kind agreement alone is not membership.
    // `chunk`, `map`, `flatten`, `zip` and `enumerate` all take a list and
    // return a list.
    for label in [
        "builtin:chunk",
        "builtin:map",
        "builtin:flatten",
        "builtin:zip",
        "builtin:enumerate",
        "builtin:dict_keys",
    ] {
        assert_eq!(
            container_consumer_operand(label, Some(list), |_| Some(list)),
            None,
            "{label} is not a row, however well its classes agree"
        );
    }
    // The two string builtins that produce a substring of their operand are
    // deliberately not rows: they return a sub-range rather than a grown
    // copy, so in-place is a memmove of the remainder rather than an
    // amortised append, and neither is an accumulation shape. This is the
    // same disposition `drop` and `take` carry for lists under chelis#943.
    for label in ["builtin:string_slice", "builtin:string_trim"] {
        assert_eq!(
            container_consumer_operand(label, Some(string), |_| Some(string)),
            None,
            "{label} produces a substring and is not a consumer row"
        );
    }
    // Rejected by the result comparison: a non-heap result cannot carry a
    // heap kind.
    assert_eq!(
        container_consumer_operand(
            "builtin:append",
            Some(ValueClass::NonHeap(NonHeapKind::Unit)),
            |_| { Some(list) }
        ),
        None,
        "a unit-returning application is not a container consumer"
    );
}

/// A consuming application of one string-classed owner per operand, with a
/// string result, for the copy-at-last-use rewrite tests.
fn consume_op(dest: u32, operands: &[u32]) -> Op {
    Op::Apply {
        dest: Some(OwnerId(dest)),
        label: "consume".into(),
        kind: ApplyKind::Intrinsic,
        schema: OperationSchema::new(
            vec![OwnershipUse::Move; operands.len()],
            Some(ValueClass::Heap(HeapKind::String)),
        ),
        args: operands
            .iter()
            .map(|owner| Operand::move_(OwnerId(*owner)))
            .collect(),
    }
}

fn copy_op(dest: u32, source: u32) -> Op {
    Op::Copy {
        dest: OwnerId(dest),
        source: Operand::clone_(OwnerId(source)),
    }
}

fn string_owners(count: u32) -> BTreeMap<OwnerId, OwnerInfo> {
    (0..count)
        .map(|owner| (OwnerId(owner), info(Prim::String, OwnerOrigin::Owned)))
        .collect()
}

fn ops_of(program: &RawProgram) -> Vec<Op> {
    program.units[0].blocks[0]
        .ops
        .iter()
        .map(|operation| operation.kind.clone())
        .collect()
}

/// chelis#2206: a copy whose source the scheduler retires directly after it
/// was that source's last read, so the rewrite hands the source itself to the
/// copy's consumer. The copy, the release and the copy's identity disappear.
#[test]
fn a_copy_at_its_sources_last_use_becomes_a_move() {
    let mut entry = block(
        0,
        vec![],
        vec![define(0), copy_op(1, 0), consume_op(2, &[1]), root(2)],
        Terminator::Exit,
    );
    entry.ops.push(provisional(90, 0, true));
    let mut program = roots(vec![entry], string_owners(3));
    schedule(&mut program).unwrap();
    assert_eq!(
        ops_of(&program),
        vec![define(0), consume_op(2, &[0]), root(2)]
    );
    assert!(!program.units[0].owners.contains_key(&OwnerId(1)));
}

/// chelis#3716: one application naming the owner in two consuming slots gets
/// one copy per slot from lowering; only the copy directly before the owner's
/// release is its last read, so exactly that slot moves the owner and the
/// other keeps its inserted copy.
#[test]
fn two_copies_for_one_application_keep_one_and_move_the_last() {
    let mut entry = block(
        0,
        vec![],
        vec![
            define(0),
            copy_op(1, 0),
            copy_op(2, 0),
            consume_op(3, &[1, 2]),
            root(3),
        ],
        Terminator::Exit,
    );
    entry.ops.push(provisional(90, 0, true));
    let mut program = roots(vec![entry], string_owners(4));
    schedule(&mut program).unwrap();
    assert_eq!(
        ops_of(&program),
        vec![define(0), copy_op(1, 0), consume_op(3, &[1, 0]), root(3)]
    );
}

/// Negative twin: a read of the source after the copy keeps the copy, and the
/// source is released after that read.
#[test]
fn a_copy_before_a_later_read_of_its_source_stays_a_copy() {
    let mut entry = block(
        0,
        vec![],
        vec![
            define(0),
            copy_op(1, 0),
            borrow_op(0),
            consume_op(2, &[1]),
            root(2),
        ],
        Terminator::Exit,
    );
    entry.ops.push(provisional(90, 0, true));
    let mut program = roots(vec![entry], string_owners(3));
    schedule(&mut program).unwrap();
    let ops = ops_of(&program);
    assert_eq!(ops[1], copy_op(1, 0));
    assert_eq!(
        ops[3],
        Op::Drop {
            owner: Operand::move_(OwnerId(0)),
        }
    );
    assert_eq!(ops[4], consume_op(2, &[1]));
}

/// The rewrite's safety rests on the verifier, which must still reject an
/// application that moves one owner through two slots (chelis#3716's
/// hand-forged control): the second slot reads an owner the first consumed.
#[test]
fn one_owner_moved_through_two_slots_of_one_application_is_rejected() {
    let program = roots(
        vec![block(
            0,
            vec![],
            vec![define(0), consume_op(1, &[0, 0]), root(1)],
            Terminator::Exit,
        )],
        string_owners(2),
    );
    assert!(matches!(
        verify_raw(program),
        Err(OwnershipError::OwnerNotLive { owner: 0, .. })
    ));
}

/// The copy rewrite takes its operand census once per unit, however many
/// copies it upgrades: a per-candidate census is a scan of the unit per copy,
/// quadratic in the unit (round-1 finding on chelis#3751). Eight independent
/// copy-at-last-use chains in one unit all upgrade under one census.
#[test]
fn the_copy_rewrite_takes_one_operand_census_per_unit() {
    const CHAINS: u32 = 8;
    let mut ops = Vec::new();
    for chain in 0..CHAINS {
        let base = chain * 3;
        ops.extend([
            define(base),
            copy_op(base + 1, base),
            consume_op(base + 2, &[base + 1]),
            root(base + 2),
        ]);
    }
    let mut entry = block(0, vec![], ops, Terminator::Exit);
    entry
        .ops
        .extend((0..CHAINS).map(|chain| provisional(900 + chain, chain * 3, true)));
    let mut program = roots(vec![entry], string_owners(CHAINS * 3));
    let before = super::last_use::OPERAND_CENSUSES.with(std::cell::Cell::get);
    schedule(&mut program).unwrap();
    let censuses = super::last_use::OPERAND_CENSUSES.with(std::cell::Cell::get) - before;
    let copies = ops_of(&program)
        .iter()
        .filter(|op| matches!(op, Op::Copy { .. }))
        .count();
    assert_eq!(
        copies, 0,
        "every chain's copy is its source's last read and upgrades"
    );
    assert_eq!(
        censuses, 1,
        "{CHAINS} upgrades took {censuses} operand censuses of one unit"
    );
}

/// The largest growth factor between consecutive sizes of a doubling sweep.
/// Every count must be positive: a sweep that never reaches the counted work
/// proves nothing.
fn max_doubling_ratio(visits: &[u64]) -> f64 {
    assert!(
        visits.iter().all(|&count| count > 0),
        "the sweep must exercise the counted work: {visits:?}"
    );
    visits
        .windows(2)
        .map(|pair| pair[1] as f64 / pair[0] as f64)
        .fold(0.0, f64::max)
}

/// chelis#3723: a chain of `count` self-recursive functions, each calling the
/// previous one, so the call graph has `count` singleton cyclic components.
fn recursive_call_chain_program(count: u32) -> RawProgram {
    let call = |dest: u32, callee: u32| Op::Apply {
        dest: Some(OwnerId(dest)),
        label: "chain call".into(),
        kind: ApplyKind::DirectCall {
            callee: UnitId(callee),
        },
        schema: OperationSchema::new(
            Vec::new(),
            Some(ValueClass::NonHeap(NonHeapKind::Scalar(Prim::Int64))),
        ),
        args: Vec::new(),
    };
    let mut units = vec![Unit {
        id: UnitId(0),
        name: "roots".into(),
        kind: UnitKind::Roots,
        schedule: ScheduleState::Phase2ScopeExit,
        callable_body: None,
        entry: BlockId(0),
        blocks: vec![block(0, vec![], vec![], Terminator::Exit)],
        owners: BTreeMap::new(),
        captured_globals: BTreeMap::new(),
    }];
    for index in 1..=count {
        let ops = if index == 1 {
            vec![call(0, index)]
        } else {
            vec![
                call(0, index),
                Op::Discard { owner: OwnerId(0) },
                call(1, index - 1),
            ]
        };
        let result = if index == 1 { 0 } else { 1 };
        units.push(Unit {
            id: UnitId(index),
            name: format!("h{index}"),
            kind: UnitKind::Function,
            schedule: ScheduleState::Phase2ScopeExit,
            callable_body: Some(CallableBody::new(BlockId(0))),
            entry: BlockId(0),
            blocks: vec![block(
                0,
                vec![],
                ops,
                Terminator::Return {
                    result: Operand::move_(OwnerId(result)),
                },
            )],
            owners: (0..=result)
                .map(|owner| (OwnerId(owner), info(Prim::Int64, OwnerOrigin::Owned)))
                .collect(),
            captured_globals: BTreeMap::new(),
        });
    }
    RawProgram { units }
}

#[test]
fn live_byte_composition_finds_call_graph_components_in_linear_work() {
    // Call-graph nodes visited while composing the verified live-byte bound
    // (chelis#3723). Pairwise reachability makes a call chain cubic, about 8x
    // per doubling.
    let visits = [50, 100, 200, 400]
        .iter()
        .map(|&count| {
            let before = super::verify::call_graph_visits();
            let verified = super::verify::verify(&recursive_call_chain_program(count)).unwrap();
            assert_eq!(
                verified.live_set_bound().max_live_bytes,
                LiveByteBound::Exact(0)
            );
            super::verify::call_graph_visits() - before
        })
        .collect::<Vec<_>>();
    assert!(
        max_doubling_ratio(&visits) <= 2.2,
        "call-graph visits per doubling must stay linear: {visits:?}"
    );
}

/// chelis#2330: a direct call's carry is the live-byte bound without the
/// owners it moves. The running totals subtract the moved owners instead of
/// copying the set; the test build cross-checks every read against the
/// copy-and-rescan, across unknown, exact, unbounded and overflowing costs
/// (the order-dependent case included), a duplicate move, and moving
/// everything.
/// The callee does not exist, so each program is rejected at the call, after
/// its carry is costed.
#[test]
fn direct_call_carry_excludes_moved_owners_like_a_rescan() {
    let half = 1usize << 61;
    let dynamic = value_info(
        ConcreteHostType::Tensor(TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        }),
        OwnerOrigin::Owned,
    );
    let layouts = [
        [
            dynamic.clone(),
            fixed_tensor_info(half),
            fixed_tensor_info(half),
        ],
        [
            fixed_tensor_info(half),
            dynamic.clone(),
            fixed_tensor_info(half),
        ],
        [
            fixed_tensor_info(half),
            fixed_tensor_info(half),
            dynamic.clone(),
        ],
        [
            fixed_tensor_info(half),
            fixed_tensor_info(half),
            fixed_tensor_info(3),
        ],
        [dynamic.clone(), dynamic.clone(), fixed_tensor_info(half)],
        // A tensor whose byte size exceeds u64 costs `Unbounded`.
        [
            fixed_tensor_info(usize::MAX),
            fixed_tensor_info(3),
            dynamic.clone(),
        ],
        [
            fixed_tensor_info(3),
            fixed_tensor_info(usize::MAX),
            fixed_tensor_info(usize::MAX),
        ],
    ];
    let move_sets: [&[u32]; 9] = [
        &[],
        &[0],
        &[1],
        &[2],
        &[0, 1],
        &[1, 2],
        &[0, 2],
        &[1, 1],
        &[0, 1, 2],
    ];
    for layout in &layouts {
        for moved in move_sets {
            let mut ops = (0..3).map(define).collect::<Vec<_>>();
            ops.push(Op::Apply {
                dest: Some(OwnerId(3)),
                label: "carry call".into(),
                kind: ApplyKind::DirectCall { callee: UnitId(99) },
                schema: OperationSchema::new(
                    Vec::new(),
                    Some(ValueClass::NonHeap(NonHeapKind::Scalar(Prim::Int64))),
                ),
                args: moved
                    .iter()
                    .map(|owner| Operand::move_(OwnerId(*owner)))
                    .collect(),
            });
            let mut owners = layout
                .iter()
                .cloned()
                .enumerate()
                .map(|(owner, info)| (OwnerId(u32::try_from(owner).unwrap()), info))
                .collect::<BTreeMap<_, _>>();
            owners.insert(OwnerId(3), info(Prim::Int64, OwnerOrigin::Owned));
            let program = roots(vec![block(0, vec![], ops, Terminator::Exit)], owners);
            assert!(
                matches!(
                    super::verify::verify(&program),
                    Err(OwnershipError::MissingUnit { unit: 99, .. })
                ),
                "moving {moved:?} must reach the call"
            );
        }
    }
}

/// chelis#3723: deterministic pseudo-random call graphs, cyclic and acyclic.
/// The test build checks the linear components against pairwise
/// reachability on every graph of at most 64 units, and the composed bound
/// is the same whichever way the components are found.
#[test]
fn call_graph_components_agree_with_pairwise_reachability() {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = |bound: u32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        u32::try_from(state % u64::from(bound)).unwrap()
    };
    for _ in 0..64 {
        let count = 2 + next(30);
        let mut units = vec![Unit {
            id: UnitId(0),
            name: "roots".into(),
            kind: UnitKind::Roots,
            schedule: ScheduleState::Phase2ScopeExit,
            callable_body: None,
            entry: BlockId(0),
            blocks: vec![block(0, vec![], vec![], Terminator::Exit)],
            owners: BTreeMap::new(),
            captured_globals: BTreeMap::new(),
        }];
        for index in 1..=count {
            let calls = next(4);
            // A tensor live across every call gives each call a positive
            // carry, so a cycle is unbounded and the bound sees the partition.
            let mut ops = vec![define(0)];
            let mut owners = BTreeMap::from([(OwnerId(0), fixed_tensor_info(1 + index as usize))]);
            for call in 0..calls {
                let dest = 1 + call;
                ops.push(Op::Apply {
                    dest: Some(OwnerId(dest)),
                    label: "random call".into(),
                    kind: ApplyKind::DirectCall {
                        callee: UnitId(1 + next(count)),
                    },
                    schema: OperationSchema::new(
                        Vec::new(),
                        Some(ValueClass::NonHeap(NonHeapKind::Scalar(Prim::Int64))),
                    ),
                    args: Vec::new(),
                });
                ops.push(Op::Discard {
                    owner: OwnerId(dest),
                });
                owners.insert(OwnerId(dest), info(Prim::Int64, OwnerOrigin::Owned));
            }
            let result = 1 + calls;
            ops.push(Op::Drop {
                owner: Operand::move_(OwnerId(0)),
            });
            ops.push(define(result));
            owners.insert(OwnerId(result), info(Prim::Int64, OwnerOrigin::Owned));
            units.push(Unit {
                id: UnitId(index),
                name: format!("f{index}"),
                kind: UnitKind::Function,
                schedule: ScheduleState::Phase2ScopeExit,
                callable_body: Some(CallableBody::new(BlockId(0))),
                entry: BlockId(0),
                blocks: vec![block(
                    0,
                    vec![],
                    ops,
                    Terminator::Return {
                        result: Operand::move_(OwnerId(result)),
                    },
                )],
                owners,
                captured_globals: BTreeMap::new(),
            });
        }
        super::verify::verify(&RawProgram { units }).unwrap();
    }
}

/// chelis#2330: `n` fixed tensors defined in one block, carried across a jump
/// and dropped in reverse order in the next. Every operation reads the
/// live-heap count and live-byte bound, every drop moves an owner, and the
/// jump copies and checks the whole live set once per block.
fn wide_live_set_program(count: u32) -> RawProgram {
    let defines = (0..count).map(define).collect::<Vec<_>>();
    let drops = (0..count)
        .rev()
        .map(|owner| Op::Drop {
            owner: Operand::move_(OwnerId(owner)),
        })
        .collect::<Vec<_>>();
    roots(
        vec![
            block(0, vec![], defines, Terminator::Jump(edge(1, vec![]))),
            block(1, vec![], drops, Terminator::Exit),
        ],
        (0..count)
            .map(|owner| (OwnerId(owner), fixed_tensor_info(4)))
            .collect(),
    )
}

/// The live-set entries one `verify` visits for each size in a doubling sweep.
fn live_entry_visit_sweep(sizes: &[u32]) -> Vec<u64> {
    sizes
        .iter()
        .map(|&count| {
            let before = super::verify::live_entry_visits();
            let verified = super::verify::verify(&wide_live_set_program(count)).unwrap();
            assert_eq!(
                verified.live_set_bound().max_live_heap_owners(),
                count as usize
            );
            assert_eq!(
                verified.live_set_bound().max_live_bytes,
                LiveByteBound::Exact(16 * u64::from(count))
            );
            super::verify::live_entry_visits() - before
        })
        .collect()
}

#[test]
fn verify_reads_live_set_aggregates_without_rescanning_the_live_set() {
    // The quadratic quantity itself (chelis#2330): live-set entries visited
    // across the whole verification. A per-operation scan of a live set that
    // grows with the body makes it grow about 4x per doubling.
    let visits = live_entry_visit_sweep(&[80, 160, 320, 640]);
    assert!(
        max_doubling_ratio(&visits) <= 2.2,
        "live-set entries visited per doubling must stay linear: {visits:?}"
    );
}

#[test]
fn live_set_visit_guard_catches_a_reintroduced_per_operation_scan() {
    // Negative control: the test build cross-checks every running aggregate
    // against a full scan of the live set. Counting those scans is exactly a
    // per-operation rescan, and the guard above must see it go quadratic.
    let visits = super::verify::with_reference_scans_counted(|| {
        live_entry_visit_sweep(&[80, 160, 320, 640])
    });
    assert!(
        max_doubling_ratio(&visits) >= 3.5,
        "a per-operation live-set scan must read as quadratic: {visits:?}"
    );
}

/// chelis#2330: the running live-byte totals reproduce the owner-order sum
/// even where that sum depends on order. With an unknown cost on the lowest
/// owner, the sum is unknown before the two exact costs overflow u64; the
/// test build cross-checks every read against that ordered full scan.
#[test]
fn running_live_byte_totals_keep_the_owner_order_sum() {
    let half = 1usize << 61;
    let dynamic = value_info(
        ConcreteHostType::Tensor(TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        }),
        OwnerOrigin::Owned,
    );
    let program = |owners: [OwnerInfo; 3]| {
        let mut ops = (0..3).map(define).collect::<Vec<_>>();
        ops.extend((0..3).rev().map(|owner| Op::Drop {
            owner: Operand::move_(OwnerId(owner)),
        }));
        roots(
            vec![block(0, vec![], ops, Terminator::Exit)],
            owners
                .into_iter()
                .enumerate()
                .map(|(owner, info)| (OwnerId(u32::try_from(owner).unwrap()), info))
                .collect(),
        )
    };
    let unknown_first = program([
        dynamic.clone(),
        fixed_tensor_info(half),
        fixed_tensor_info(half),
    ]);
    assert_eq!(
        super::verify::verify(&unknown_first)
            .unwrap()
            .live_set_bound()
            .max_live_bytes,
        LiveByteBound::Unknown
    );
    let unknown_last = program([fixed_tensor_info(half), fixed_tensor_info(half), dynamic]);
    assert_eq!(
        super::verify::verify(&unknown_last)
            .unwrap()
            .live_set_bound()
            .max_live_bytes,
        LiveByteBound::Unbounded
    );
}

fn lower_source_for_ownership(source: &str) -> super::HostOwnershipProgram {
    let declarations = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("surf parse");
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .expect("desugar");
    let checked = chelis_types::check_typed_program(&deep)
        .unwrap_or_else(|errors| panic!("type check: {:?}", errors.errors));
    let checked = chelis_effects::check_program(&checked)
        .unwrap_or_else(|error| panic!("effects: {error:?}"));
    let checked = chelis_types::check_linearity(&checked)
        .unwrap_or_else(|error| panic!("linearity: {error:?}"));
    let realizability =
        chelis_effects::realizability::infer_realizability(&checked, &[Prim::F32, Prim::Bool]);
    let manifest = chelis_effects::realizability::compute_root_manifest(&checked, &realizability);
    let compiled = crate::host::try_lower_compiled_program_with_manifest(&checked, &manifest)
        .unwrap_or_else(|diagnostic| panic!("host lowering: {diagnostic:?}"));
    let host = compiled.host.expect("test source must have a host lane");
    let manifested = chelis_types::manifest::ManifestedProgram::new(
        checked,
        manifest,
        chelis_types::types::Target::C,
    );
    super::lower_host_ownership(&manifested, host).unwrap()
}

#[test]
fn host_build_verifies_each_distinct_ownership_program_once() {
    // chelis#3730: the conservative pre-schedule graph and the scheduled
    // program are verified once each. The boundary reuses the scheduled
    // program's verification rather than running it again on the same
    // sealed program.
    let source = "def bump(x: i64) -> i64 = add(x, 1i64)\nr = bump(3i64)\n";
    let before = super::verify::verify_calls();
    let lowered = lower_source_for_ownership(source);
    let verified = super::verify_ownership(lowered).unwrap();
    assert_eq!(super::verify::verify_calls() - before, 2);
    assert!(verified.render().contains("unit bump Function"));
}

/// chelis#3755: a body whose owners and operations both grow with `count`.
/// Block 0 borrows a branch condition, defines `count` owners that are each
/// read once right after their definition, and defines `count` more that are
/// carried to a join: the then arm reads every carried owner, and both arms
/// move them into the join's `count` unused owned parameters. Every owner asks
/// the scheduler and the canonical re-check for its definition, its uses and
/// future need, its placement anchors and the edges entering its block.
fn last_use_scaling_program(count: u32) -> RawProgram {
    let line = |index: u32| 1 + index;
    let carried = |index: u32| 1 + count + index;
    let joined = |index: u32| 1 + 2 * count + index;
    let moves = || {
        (0..count)
            .map(|index| Operand::move_(OwnerId(carried(index))))
            .collect::<Vec<_>>()
    };
    let mut ops = Vec::new();
    for index in 0..count {
        ops.push(define(line(index)));
        ops.push(borrow_op(line(index)));
    }
    ops.extend((0..count).map(|index| define(carried(index))));
    let mut entry = block(
        0,
        vec![BlockParam {
            owner: OwnerId(0),
            mode: ParamMode::EntryBorrow,
        }],
        ops,
        Terminator::Branch {
            condition: Operand::borrow(OwnerId(0)),
            then_edge: edge(1, vec![]),
            else_edge: edge(2, moves()),
        },
    );
    entry
        .ops
        .extend((0..count).map(|index| provisional(0x4000_0000 + index, line(index), true)));
    let then_block = block(
        1,
        vec![],
        (0..count).map(|index| borrow_op(carried(index))).collect(),
        Terminator::Jump(edge(2, moves())),
    );
    let mut join = block(
        2,
        (0..count)
            .map(|index| BlockParam {
                owner: OwnerId(joined(index)),
                mode: ParamMode::Owned,
            })
            .collect(),
        vec![],
        Terminator::Exit,
    );
    join.ops
        .extend((0..count).map(|index| provisional(0x5000_0000 + index, joined(index), true)));
    let mut owners = BTreeMap::from([(OwnerId(0), info(Prim::Bool, OwnerOrigin::ExternalBorrow))]);
    owners.extend(
        (0..count)
            .flat_map(|index| [line(index), carried(index), joined(index)])
            .map(|owner| (OwnerId(owner), info(Prim::String, OwnerOrigin::Owned))),
    );
    roots(vec![entry, then_block, join], owners)
}

/// chelis#3755: a chain of `count` blocks, each defining an owner that the
/// next block reads, so each owner's frontier is found in another block and
/// every lookup by block identity ranges over a unit that grows with it.
fn last_use_block_chain_program(count: u32) -> RawProgram {
    let blocks = (0..=count)
        .map(|index| {
            let mut ops = Vec::new();
            if index > 0 {
                ops.push(borrow_op(index - 1));
            }
            if index < count {
                ops.push(define(index));
            }
            let terminator = if index < count {
                Terminator::Jump(edge(index + 1, vec![]))
            } else {
                Terminator::Exit
            };
            let mut block = block(index, vec![], ops, terminator);
            if index > 0 {
                block
                    .ops
                    .push(provisional(0x4000_0000 + index, index - 1, true));
            }
            block
        })
        .collect();
    roots(
        blocks,
        (0..count)
            .map(|owner| (OwnerId(owner), info(Prim::String, OwnerOrigin::Owned)))
            .collect(),
    )
}

/// chelis#3755: `count` owners defined up front and all read by one final
/// operation, so every owner's single frontier sits at the end of the body
/// and post-dominates a definition that much earlier.
fn last_use_late_read_program(count: u32) -> RawProgram {
    let mut ops = (0..count).map(define).collect::<Vec<_>>();
    ops.push(Op::Apply {
        dest: None,
        label: "observe all".into(),
        kind: ApplyKind::Intrinsic,
        schema: OperationSchema::new(vec![OwnershipUse::Borrow; count as usize], None),
        args: (0..count)
            .map(|owner| Operand::borrow(OwnerId(owner)))
            .collect(),
    });
    let mut entry = block(0, vec![], ops, Terminator::Exit);
    entry
        .ops
        .extend((0..count).map(|owner| provisional(0x4000_0000 + owner, owner, true)));
    roots(
        vec![entry],
        (0..count)
            .map(|owner| (OwnerId(owner), info(Prim::String, OwnerOrigin::Owned)))
            .collect(),
    )
}

/// The scheduled scope exits of a unit: in blocks, and on edges.
fn scheduled_terminal_counts(unit: &Unit) -> (usize, usize) {
    let in_block = unit
        .blocks
        .iter()
        .flat_map(|block| &block.ops)
        .filter(|operation| operation.role == OperationRole::ScheduledScopeExit)
        .count();
    let on_edges = unit
        .blocks
        .iter()
        .flat_map(|block| block.terminator.edges())
        .map(|edge| edge.terminals.len())
        .sum();
    (in_block, on_edges)
}

/// chelis#3755: a recursive function whose body chains `count` let
/// bindings, lowered from source so the scheduler also re-anchors every
/// terminal in a real host-site map.
fn last_use_lowered_chain_source(count: u32) -> String {
    let mut source =
        String::from("def st(s: i64, i: i64) -> i64 = if gte(i, 5i64) then s else {\n");
    let mut previous = "s".to_string();
    for index in 0..count {
        source.push_str(&format!("  t{index} = add({previous}, 1i64)\n"));
        previous = format!("t{index}");
    }
    source.push_str(&format!(
        "  st({previous}, add(i, 1i64))\n}}\nr = st(1i64, 0i64)\n"
    ));
    source
}

/// Run `run` on a thread with a stack deep enough for the front end on a
/// long let chain in a debug build. The lookup counters are per thread, so the
/// whole measurement runs there.
fn on_large_stack<T: Send>(run: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn_scoped(scope, run)
            .expect("spawn the measurement thread")
            .join()
            .expect("the measurement thread completes")
    })
}

/// The lookup visits the scheduler and the canonical re-check of its result
/// make for each size in a doubling sweep, per shape: the wide body, the
/// block chain, the late read and the lowered let chain.
fn owner_lookup_visit_sweep(sizes: &[u32]) -> [Vec<u64>; 4] {
    let mut visits = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for &count in sizes {
        let mut wide = last_use_scaling_program(count);
        let before = super::last_use::lookup_visits();
        schedule(&mut wide).unwrap();
        visits[0].push(super::last_use::lookup_visits() - before);
        // Each line owner dies after its read; each joined parameter dies on
        // both edges into the join.
        assert_eq!(
            scheduled_terminal_counts(&wide.units[0]),
            (count as usize, 2 * count as usize)
        );

        let mut chain = last_use_block_chain_program(count);
        let before = super::last_use::lookup_visits();
        schedule(&mut chain).unwrap();
        visits[1].push(super::last_use::lookup_visits() - before);
        // Each chain owner dies after its read in the next block.
        assert_eq!(
            scheduled_terminal_counts(&chain.units[0]),
            (count as usize, 0)
        );

        let mut late = last_use_late_read_program(count);
        let before = super::last_use::lookup_visits();
        schedule(&mut late).unwrap();
        visits[2].push(super::last_use::lookup_visits() - before);
        // Every owner dies right after the final read.
        assert_eq!(
            scheduled_terminal_counts(&late.units[0]),
            (count as usize, 0)
        );

        let source = last_use_lowered_chain_source(count);
        let before = super::last_use::lookup_visits();
        let lowered = lower_source_for_ownership(&source);
        visits[3].push(super::last_use::lookup_visits() - before);
        let super::OwnershipProof::Host { program, .. } = &lowered.proof else {
            panic!("source lowering must produce a host ownership proof");
        };
        let owners = program
            .units
            .iter()
            .map(|unit| unit.owners.len())
            .sum::<usize>();
        assert!(
            owners >= count as usize,
            "the lowered chain must bind an owner per let: {owners}"
        );
    }
    visits
}

#[test]
fn last_use_lookups_stay_linear_in_the_body() {
    // The quadratic quantity itself (chelis#3755): operations, parameters,
    // edges, site actions and index entries the per-owner and per-terminal
    // lookups of the scheduler and its canonical re-check examine. A scan of
    // the unit, a block or the site map per owner or per terminal makes it
    // grow about 4x per doubling.
    for visits in on_large_stack(|| owner_lookup_visit_sweep(&[32, 64, 128, 256])) {
        assert!(
            max_doubling_ratio(&visits) <= 2.2,
            "lookup visits per doubling must stay linear: {visits:?}"
        );
    }
}

#[test]
fn owner_lookup_guard_catches_a_reintroduced_per_owner_scan() {
    // Negative control: the test build checks every indexed lookup against
    // the scan it replaced. Counting those scans is exactly a per-owner or
    // per-terminal scan, and the guard above must see it go quadratic.
    let sweeps = on_large_stack(|| {
        super::last_use::with_reference_scans_counted(|| {
            owner_lookup_visit_sweep(&[32, 64, 128, 256])
        })
    });
    for visits in sweeps {
        assert!(
            max_doubling_ratio(&visits) >= 3.5,
            "a per-owner scan must read as quadratic: {visits:?}"
        );
    }
}
