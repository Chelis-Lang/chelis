use std::collections::BTreeMap;

use chelis_numeric_core::prim::Prim;
use chelis_types::manifest::RootManifest;

use super::OwnershipError;
use super::ir::{
    ApplyKind, Block, BlockId, BlockParam, Edge, EdgeId, HostSiteAction, HostSiteId, HostSiteKind,
    HostSiteMap, HostSiteRecord, Op, OpId, Operand, Operation, OperationRole, OperationSchema,
    OwnerId, OwnerInfo, OwnerOrigin, OwnershipProgram as RawProgram, OwnershipUse, ParamMode,
    ScheduleState, Terminator, Unit, UnitId, UnitKind,
};
use chelis_ir_core::host_type_state::ConcreteHostType;
use chelis_ir_core::ownership::classify::{Placement, classify};

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

fn define(id: u32) -> Op {
    Op::Define {
        dest: OwnerId(id),
        label: "fresh".into(),
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
    super::verify::verify_host_actions(
        &program,
        &super::NativeManifest(&RootManifest { entries: vec![] }),
        &sites,
    )
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
    let verification = super::verify::verify(&program)
        .unwrap()
        .map_live_set_bound(|data| super::VerifiedLiveSetBound { data });
    super::verify::verify_host_actions(
        &program,
        &super::NativeManifest(&RootManifest { entries: vec![] }),
        &sites,
    )
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

fn schedule(program: &mut RawProgram) -> Result<super::last_use::ScheduleFacts, OwnershipError> {
    super::last_use::schedule_observed(
        program,
        &mut HostSiteMap::from_records(Vec::new()),
        Some(&super::SiteObserver),
    )
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
    // chelis#3730: the conservative pre-schedule graph and the scheduled program are verified once each.
    // The boundary reuses the scheduled program's verification rather than running it again on the same sealed program.
    let source = "def bump(x: i64) -> i64 = add(x, 1i64)\nr = bump(3i64)\n";
    let before = super::OWNERSHIP_VERIFICATIONS.with(std::cell::Cell::get);
    let lowered = lower_source_for_ownership(source);
    let verified = super::verify_ownership(lowered).unwrap();
    assert_eq!(
        super::OWNERSHIP_VERIFICATIONS.with(std::cell::Cell::get) - before,
        2
    );
    assert!(verified.render().contains("unit bump Function"));
}

/// chelis#3755: a body whose owners and operations both grow with `count`.
/// Block 0 borrows a branch condition, defines `count` owners that are each
/// read once right after their definition, and defines `count` more that are
/// carried to a join: the then arm reads every carried owner, and both arms
/// move them into the join's `count` unused owned parameters.
/// Every owner asks
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
/// long let chain in a debug build.
/// The lookup counters are per thread, so the
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
        let before = super::OWNERSHIP_LOOKUP_ENTRIES.with(std::cell::Cell::get);
        schedule(&mut wide).unwrap();
        visits[0].push(super::OWNERSHIP_LOOKUP_ENTRIES.with(std::cell::Cell::get) - before);
        // Each line owner dies after its read.
        // each joined parameter dies on both edges into the join.
        assert_eq!(
            scheduled_terminal_counts(&wide.units[0]),
            (count as usize, 2 * count as usize)
        );

        let mut chain = last_use_block_chain_program(count);
        let before = super::OWNERSHIP_LOOKUP_ENTRIES.with(std::cell::Cell::get);
        schedule(&mut chain).unwrap();
        visits[1].push(super::OWNERSHIP_LOOKUP_ENTRIES.with(std::cell::Cell::get) - before);
        // Each chain owner dies after its read in the next block.
        assert_eq!(
            scheduled_terminal_counts(&chain.units[0]),
            (count as usize, 0)
        );

        let mut late = last_use_late_read_program(count);
        let before = super::OWNERSHIP_LOOKUP_ENTRIES.with(std::cell::Cell::get);
        schedule(&mut late).unwrap();
        visits[2].push(super::OWNERSHIP_LOOKUP_ENTRIES.with(std::cell::Cell::get) - before);
        // Every owner dies right after the final read.
        assert_eq!(
            scheduled_terminal_counts(&late.units[0]),
            (count as usize, 0)
        );

        let source = last_use_lowered_chain_source(count);
        let before = super::OWNERSHIP_LOOKUP_ENTRIES.with(std::cell::Cell::get);
        let lowered = lower_source_for_ownership(&source);
        visits[3].push(super::OWNERSHIP_LOOKUP_ENTRIES.with(std::cell::Cell::get) - before);
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
    // The quadratic quantity itself (chelis#3755): operations, parameters, edges.
    // Site actions and index entries the per-owner and per-terminal lookups of the scheduler and its canonical re-check examine.
    // A scan of the unit, a block or the site map per owner or per terminal makes it grow about 4x per doubling.
    for visits in on_large_stack(|| owner_lookup_visit_sweep(&[32, 64, 128, 256])) {
        assert!(
            max_doubling_ratio(&visits) <= 2.2,
            "lookup visits per doubling must stay linear: {visits:?}"
        );
    }
}

#[test]
fn owner_lookup_guard_catches_a_reintroduced_per_owner_scan() {
    // Negative control: the test build checks every indexed lookup against the scan it replaced.
    // Counting those scans is exactly a per-owner or per-terminal scan, and the guard above must see it go quadratic.
    let sweeps = on_large_stack(|| {
        super::with_reference_scans_counted(|| owner_lookup_visit_sweep(&[32, 64, 128, 256]))
    });
    for visits in sweeps {
        assert!(
            max_doubling_ratio(&visits) >= 3.5,
            "a per-owner scan must read as quadratic: {visits:?}"
        );
    }
}
