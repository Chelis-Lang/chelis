//! Where each block, operation, edge-local terminal and edge of a unit sits.
//!
//! Host-site verification and the verified emission views resolve every
//! site action to the block, operation or edge it names. Finding each one by
//! scanning the unit made both quadratic in a body whose operations share one
//! block, as a `let` chain's do (chelis#3825). [`ProgramLocations`] answers
//! them from tables built in one pass over the program.
//!
//! The tables hold positions, not references, so the verification that owns
//! them can travel beside the program it describes. Where an identity repeats,
//! the first occurrence in block, operation and edge order wins, which is the
//! answer the scans gave; verification rejects repeated identities on its own.
//! The last-use scheduler's `UnitIndex` answers different questions (owner
//! mentions over a unit's non-scheduled operations) and borrows the unit it
//! describes, so it is not reused here.

use std::cell::Cell;
use std::collections::BTreeMap;

use super::ir::{
    Block, BlockId, Edge, EdgeId, EdgeTerminal, OpId, Operation, OwnershipProgram, Terminator, Unit,
};

thread_local! {
    /// Entries the site lookups visited while a test counts them, so it can
    /// bound their work by the program's size (chelis#3825). `None` when
    /// nothing counts.
    static SITE_LOOKUP_ENTRIES: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Run `work`, counting the entries the host-site lookups visit on this
/// thread meanwhile: each table entry built, and each lookup answered.
#[doc(hidden)]
pub fn counting_site_lookup_entries<R>(work: impl FnOnce() -> R) -> (R, u64) {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SITE_LOOKUP_ENTRIES.with(|count| count.set(self.0));
        }
    }
    let restore = Restore(SITE_LOOKUP_ENTRIES.with(|count| count.replace(Some(0))));
    let result = work();
    let counted = SITE_LOOKUP_ENTRIES
        .with(Cell::get)
        .expect("the count stays armed while the work runs");
    drop(restore);
    (result, counted)
}

/// Add `entries` to the site-lookup count while a test counts.
pub(super) fn note_site_lookup_entries(entries: usize) {
    SITE_LOOKUP_ENTRIES.with(|count| {
        if let Some(counted) = count.get() {
            count.set(Some(counted + entries as u64));
        }
    });
}

/// The position tables of every unit of one ownership program.
#[derive(Debug, Default)]
pub(super) struct ProgramLocations {
    units: Vec<UnitLocations>,
}

#[derive(Debug, Default)]
struct UnitLocations {
    /// Block identity to its position in the unit.
    blocks: BTreeMap<BlockId, usize>,
    /// An operation identity in a block to its position among the block's
    /// operations.
    operations: BTreeMap<(BlockId, OpId), usize>,
    /// An edge-local terminal identity in a block to the position of its edge
    /// in [`super::ir::Terminator::edges`] and its own position on that edge.
    terminals: BTreeMap<(BlockId, OpId), (usize, usize)>,
    /// A control edge, by its source block, identity and target, to its
    /// position in [`super::ir::Terminator::edges`].
    edges: BTreeMap<(BlockId, EdgeId, BlockId), usize>,
}

impl ProgramLocations {
    pub(super) fn new(program: &OwnershipProgram) -> Self {
        Self {
            units: program.units.iter().map(UnitLocations::new).collect(),
        }
    }

    /// The first block of `unit` whose identity is `block`.
    pub(super) fn block<'p>(
        &self,
        program: &'p OwnershipProgram,
        unit: usize,
        block: BlockId,
    ) -> Option<&'p Block> {
        note_site_lookup_entries(1);
        let position = *self.units.get(unit)?.blocks.get(&block)?;
        let found = program.units.get(unit)?.blocks.get(position)?;
        (found.id == block).then_some(found)
    }

    /// The first operation of that block whose identity is `operation`.
    pub(super) fn operation<'p>(
        &self,
        program: &'p OwnershipProgram,
        unit: usize,
        block: BlockId,
        operation: OpId,
    ) -> Option<&'p Operation> {
        let found_block = self.block(program, unit, block)?;
        note_site_lookup_entries(1);
        let position = *self.units.get(unit)?.operations.get(&(block, operation))?;
        let found = found_block.ops.get(position)?;
        (found.id == operation).then_some(found)
    }

    /// The first edge-local terminal of that block, in edge order, whose
    /// identity is `operation`, with the edge that carries it.
    pub(super) fn terminal<'p>(
        &self,
        program: &'p OwnershipProgram,
        unit: usize,
        block: BlockId,
        operation: OpId,
    ) -> Option<(&'p Edge, &'p EdgeTerminal)> {
        let found_block = self.block(program, unit, block)?;
        note_site_lookup_entries(1);
        let (edge, terminal) = *self.units.get(unit)?.terminals.get(&(block, operation))?;
        let edge = edge_at(&found_block.terminator, edge)?;
        let found = edge.terminals.get(terminal)?;
        (found.id == operation).then_some((edge, found))
    }

    /// The first edge of `source`'s terminator with identity `edge` and
    /// target `target`.
    pub(super) fn edge<'p>(
        &self,
        program: &'p OwnershipProgram,
        unit: usize,
        source: BlockId,
        edge: EdgeId,
        target: BlockId,
    ) -> Option<&'p Edge> {
        let found_block = self.block(program, unit, source)?;
        note_site_lookup_entries(1);
        let position = *self.units.get(unit)?.edges.get(&(source, edge, target))?;
        let found = edge_at(&found_block.terminator, position)?;
        (found.id == edge && found.target == target).then_some(found)
    }
}

impl UnitLocations {
    fn new(unit: &Unit) -> Self {
        let mut locations = Self::default();
        for (block_position, block) in unit.blocks.iter().enumerate() {
            note_site_lookup_entries(1);
            if locations.blocks.contains_key(&block.id) {
                // A later block with a repeated identity is never the one a
                // lookup by identity finds.
                continue;
            }
            locations.blocks.insert(block.id, block_position);
            for (position, operation) in block.ops.iter().enumerate() {
                note_site_lookup_entries(1);
                locations
                    .operations
                    .entry((block.id, operation.id))
                    .or_insert(position);
            }
            for (edge_position, edge) in block.terminator.edges().enumerate() {
                note_site_lookup_entries(1);
                locations
                    .edges
                    .entry((block.id, edge.id, edge.target))
                    .or_insert(edge_position);
                for (position, terminal) in edge.terminals.iter().enumerate() {
                    note_site_lookup_entries(1);
                    locations
                        .terminals
                        .entry((block.id, terminal.id))
                        .or_insert((edge_position, position));
                }
            }
        }
        locations
    }
}

/// The edge at `position` in [`Terminator::edges`] order, without walking the
/// edges before it. A disagreement with that order fails the identity check
/// every lookup makes, never a silent answer.
fn edge_at(terminator: &Terminator, position: usize) -> Option<&Edge> {
    match (terminator, position) {
        (Terminator::Jump(edge), 0) => Some(edge),
        (
            Terminator::Branch { then_edge, .. }
            | Terminator::Loop {
                body_edge: then_edge,
                ..
            },
            0,
        ) => Some(then_edge),
        (
            Terminator::Branch { else_edge, .. }
            | Terminator::Loop {
                exit_edge: else_edge,
                ..
            },
            1,
        ) => Some(else_edge),
        (Terminator::Match { arms, .. }, position) => arms.get(position),
        _ => None,
    }
}
