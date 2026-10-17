//! One unit's structural lookup tables for the last-use scheduler and its
//! canonical re-check.
//!
//! Both passes ask the same questions once per owner: where the owner is
//! defined, which operations, terminators and edges name it, which edges enter
//! a block, and which operation a placement anchors after. Answering each by
//! scanning the unit or the block made both passes quadratic in a body whose
//! owners grow with it (chelis#3755). [`UnitIndex`] answers them from tables
//! built in one pass over the unit, so an owner's lookups cost what its own
//! mentions and the blocks it reaches cost. The tables describe the unit as it
//! stands and carry no scheduling decision: the scheduler and the verifier each
//! build their own from the unit they read, and the verifier still consumes no
//! scheduler fact.
//!
//! Positions count the block's non-scheduled operations, as
//! [`super::non_scheduled_ops`] yields them; neither pass builds an index over
//! a unit that still holds a provisional scope exit.

use std::collections::{BTreeMap, BTreeSet};

use super::super::error::OwnershipError;
use super::super::ir::{
    Block, BlockId, EdgeId, Op, OpId, Operation, OwnerId, OwnershipUse, Terminator, Unit,
};
use super::{SchedulePoint, destination, lookup_visited, non_scheduled_ops};

pub(super) struct UnitIndex<'a> {
    unit: &'a Unit,
    positions: BTreeMap<BlockId, usize>,
    /// Each block's non-scheduled operations in order, by block position.
    operations: Vec<Vec<&'a Operation>>,
    /// The edges entering each block, in block then edge order.
    incoming: BTreeMap<BlockId, Vec<IncomingEdge>>,
    owners: BTreeMap<OwnerId, OwnerMentions>,
}

/// One edge entering a block.
#[derive(Debug, Clone, Copy)]
pub(super) struct IncomingEdge {
    pub(super) source: BlockId,
    /// The source block's position in the unit.
    pub(super) source_position: usize,
    pub(super) edge: EdgeId,
}

#[derive(Debug, Default)]
struct OwnerMentions {
    /// The first definition in block order, a block parameter before an
    /// operation of the same block: the block and how many non-scheduled
    /// operations precede the owner's life there.
    definition: Option<(BlockId, usize)>,
    /// The blocks that name the owner, by block position.
    blocks: BTreeMap<usize, BlockMentions>,
}

/// How one block names one owner.
#[derive(Debug, Default)]
pub(super) struct BlockMentions {
    /// A parameter of the block is the owner.
    pub(super) param: bool,
    /// A non-scheduled operation of the block defines the owner.
    pub(super) redefines: bool,
    /// The non-scheduled operations naming the owner as an operand or as their
    /// destination, ascending by position, each once.
    operations: Vec<OperationMention>,
    /// The terminator's own operand reads the owner without ending it.
    pub(super) terminator_nonterminal: bool,
    /// The terminator's own operand moves the owner out (a `Return`).
    pub(super) terminator_fixed: bool,
    /// The edges whose arguments name the owner, ascending by their index in
    /// [`Terminator::edges`].
    edges: Vec<EdgeMention>,
}

#[derive(Debug, Clone, Copy)]
struct OperationMention {
    position: usize,
    id: OpId,
    defines: bool,
    nonterminal: bool,
    fixed: bool,
}

#[derive(Debug, Clone, Copy)]
struct EdgeMention {
    index: usize,
    id: EdgeId,
    nonterminal: bool,
    fixed: bool,
}

static NO_MENTIONS: BlockMentions = BlockMentions {
    param: false,
    redefines: false,
    operations: Vec::new(),
    terminator_nonterminal: false,
    terminator_fixed: false,
    edges: Vec::new(),
};

impl<'a> UnitIndex<'a> {
    pub(super) fn new(unit: &'a Unit) -> Result<Self, OwnershipError> {
        let mut positions = BTreeMap::new();
        let mut operations = Vec::with_capacity(unit.blocks.len());
        let mut incoming = BTreeMap::<BlockId, Vec<IncomingEdge>>::new();
        let mut owners = BTreeMap::<OwnerId, OwnerMentions>::new();
        for (block_position, block) in unit.blocks.iter().enumerate() {
            // A repeated block identity has no single answer to "the block
            // with this id"; ordinary verification rejects it the same way.
            if positions.insert(block.id, block_position).is_some() {
                return Err(OwnershipError::DuplicateIdentity {
                    unit: unit.name.clone(),
                    kind: "block",
                    id: block.id.0,
                });
            }
            lookup_visited(block.params.len());
            for param in &block.params {
                let owner = owners.entry(param.owner).or_default();
                owner.definition.get_or_insert((block.id, 0));
                owner.blocks.entry(block_position).or_default().param = true;
            }
            let ops = non_scheduled_ops(block).collect::<Vec<_>>();
            lookup_visited(ops.len());
            for (position, operation) in ops.iter().enumerate() {
                let mut mention = |owner: OwnerId, apply: &dyn Fn(&mut OperationMention)| {
                    let mentions = owners.entry(owner).or_default();
                    let block_mentions = mentions.blocks.entry(block_position).or_default();
                    if block_mentions
                        .operations
                        .last()
                        .is_none_or(|last| last.position != position)
                    {
                        block_mentions.operations.push(OperationMention {
                            position,
                            id: operation.id,
                            defines: false,
                            nonterminal: false,
                            fixed: false,
                        });
                    }
                    let last = block_mentions
                        .operations
                        .last_mut()
                        .expect("the mention was just pushed");
                    apply(last);
                };
                for_each_operand(&operation.kind, |owner, ends| {
                    lookup_visited(1);
                    mention(owner, &|entry| {
                        if ends {
                            entry.fixed = true;
                        } else {
                            entry.nonterminal = true;
                        }
                    });
                });
                if let Some(owner) = destination(&operation.kind) {
                    mention(owner, &|entry| entry.defines = true);
                    let mentions = owners
                        .get_mut(&owner)
                        .expect("the mention created the owner");
                    mentions.definition.get_or_insert((block.id, position + 1));
                    mentions
                        .blocks
                        .get_mut(&block_position)
                        .expect("the mention created the block entry")
                        .redefines = true;
                }
            }
            if let Some((owner, nonterminal, fixed)) = terminator_operand(&block.terminator) {
                lookup_visited(1);
                let block_mentions = owners
                    .entry(owner)
                    .or_default()
                    .blocks
                    .entry(block_position)
                    .or_default();
                block_mentions.terminator_nonterminal |= nonterminal;
                block_mentions.terminator_fixed |= fixed;
            }
            for (index, edge) in block.terminator.edges().enumerate() {
                incoming.entry(edge.target).or_default().push(IncomingEdge {
                    source: block.id,
                    source_position: block_position,
                    edge: edge.id,
                });
                lookup_visited(1 + edge.args.len());
                for arg in &edge.args {
                    let block_mentions = owners
                        .entry(arg.owner)
                        .or_default()
                        .blocks
                        .entry(block_position)
                        .or_default();
                    if block_mentions
                        .edges
                        .last()
                        .is_none_or(|last| last.index != index)
                    {
                        block_mentions.edges.push(EdgeMention {
                            index,
                            id: edge.id,
                            nonterminal: false,
                            fixed: false,
                        });
                    }
                    let last = block_mentions
                        .edges
                        .last_mut()
                        .expect("the mention was just pushed");
                    if arg.use_ == OwnershipUse::Move {
                        last.fixed = true;
                    } else {
                        last.nonterminal = true;
                    }
                }
            }
            operations.push(ops);
        }
        Ok(Self {
            unit,
            positions,
            operations,
            incoming,
            owners,
        })
    }

    /// The unit the reference scans re-read.
    #[cfg(test)]
    pub(super) fn unit(&self) -> &'a Unit {
        self.unit
    }

    /// The block with `id` and its position in the unit.
    pub(super) fn block(&self, id: BlockId) -> Result<(usize, &'a Block), OwnershipError> {
        lookup_visited(1);
        self.positions
            .get(&id)
            .map(|&position| (position, &self.unit.blocks[position]))
            .ok_or_else(|| OwnershipError::MissingBlock {
                unit: self.unit.name.clone(),
                block: id.0,
            })
    }

    /// Where `owner` starts living: its block, and how many non-scheduled
    /// operations of that block precede it (0 for a block parameter).
    pub(super) fn definition(&self, owner: OwnerId) -> Result<(BlockId, usize), OwnershipError> {
        lookup_visited(1);
        let definition = self
            .owners
            .get(&owner)
            .and_then(|mentions| mentions.definition)
            .ok_or_else(|| OwnershipError::IncompleteOwner {
                unit: self.unit.name.clone(),
                owner: owner.0,
                missing: "definition",
            });
        #[cfg(test)]
        assert_eq!(
            definition,
            super::reference::definition_start(self.unit, owner),
            "indexed definition diverged from the unit"
        );
        definition
    }

    /// The non-scheduled operation a placement `after` operations into
    /// `block` sits behind; `after` is at least one.
    pub(super) fn anchor(
        &self,
        block: BlockId,
        after: usize,
    ) -> Result<Option<OpId>, OwnershipError> {
        let (position, _) = self.block(block)?;
        let anchor = after
            .checked_sub(1)
            .and_then(|index| self.operations[position].get(index))
            .map(|operation| operation.id);
        #[cfg(test)]
        assert_eq!(
            anchor,
            super::reference::anchor(self.unit, block, after),
            "indexed placement anchor diverged from the block"
        );
        Ok(anchor)
    }

    /// How the block at `position` names `owner`.
    pub(super) fn mentions(&self, owner: OwnerId, position: usize) -> &BlockMentions {
        lookup_visited(1);
        let mentions = self
            .owners
            .get(&owner)
            .and_then(|mentions| mentions.blocks.get(&position))
            .unwrap_or(&NO_MENTIONS);
        #[cfg(test)]
        mentions.check_against(&self.unit.blocks[position], owner);
        mentions
    }

    /// The blocks naming `owner`, ascending by position.
    pub(super) fn mentioning_blocks(
        &self,
        owner: OwnerId,
    ) -> impl Iterator<Item = (BlockId, &BlockMentions)> + '_ {
        self.owners
            .get(&owner)
            .into_iter()
            .flat_map(|mentions| &mentions.blocks)
            .map(|(&position, mentions)| {
                lookup_visited(1);
                (self.unit.blocks[position].id, mentions)
            })
    }

    /// The edges entering `block`, in block then edge order.
    pub(super) fn incoming(&self, block: BlockId) -> &[IncomingEdge] {
        let edges = self.incoming.get(&block).map_or(&[][..], Vec::as_slice);
        lookup_visited(1 + edges.len());
        edges
    }

    /// The points where an operation, a `Return` or an edge argument of the
    /// unit moves `owner` out.
    pub(super) fn fixed_consume_points(&self, owner: OwnerId) -> BTreeSet<SchedulePoint> {
        let mut points = BTreeSet::new();
        for (block, mentions) in self.mentioning_blocks(owner) {
            lookup_visited(mentions.operations.len() + mentions.edges.len());
            points.extend(
                mentions
                    .operations
                    .iter()
                    .filter(|operation| operation.fixed)
                    .map(|operation| SchedulePoint::AfterOperation {
                        block,
                        operation: operation.id,
                    }),
            );
            if mentions.terminator_fixed {
                points.insert(SchedulePoint::BeforeTerminator(block));
            }
            points.extend(
                mentions
                    .edges
                    .iter()
                    .filter(|edge| edge.fixed)
                    .map(|edge| SchedulePoint::Edge(edge.id)),
            );
        }
        #[cfg(test)]
        assert_eq!(
            points,
            super::reference::fixed_consume_points(self.unit, owner),
            "indexed fixed consumes diverged from the unit"
        );
        points
    }
}

impl BlockMentions {
    /// An operation at or after position `start` moves the owner out.
    pub(super) fn fixed_consume_from(&self, start: usize) -> bool {
        let tail = self.operations_from(start);
        lookup_visited(tail.len());
        tail.iter().any(|operation| operation.fixed)
    }

    /// One past the last operation at or after position `start` that reads
    /// the owner without ending it.
    pub(super) fn last_nonterminal_use_from(&self, start: usize) -> Option<usize> {
        let tail = self.operations_from(start);
        lookup_visited(tail.len());
        tail.iter()
            .rev()
            .find(|operation| operation.nonterminal)
            .map(|operation| operation.position + 1)
    }

    /// An operation before the block's first definition of the owner uses it,
    /// with or without ending it. Without a definition, any use counts.
    pub(super) fn used_before_redefinition(&self) -> bool {
        lookup_visited(self.operations.len());
        self.operations
            .iter()
            .take_while(|operation| !operation.defines)
            .any(|operation| operation.nonterminal || operation.fixed)
    }

    /// An edge argument of the block names the owner, read or moved.
    pub(super) fn edge_use(&self) -> bool {
        !self.edges.is_empty()
    }

    /// The edge at `index` in [`Terminator::edges`] moves the owner out.
    pub(super) fn edge_fixed_consume(&self, index: usize) -> bool {
        lookup_visited(1);
        self.edges
            .binary_search_by_key(&index, |edge| edge.index)
            .is_ok_and(|found| self.edges[found].fixed)
    }

    fn operations_from(&self, start: usize) -> &[OperationMention] {
        let first = self
            .operations
            .partition_point(|operation| operation.position < start);
        &self.operations[first..]
    }

    #[cfg(test)]
    pub(super) fn check_against(&self, block: &Block, owner: OwnerId) {
        use super::reference::{
            edge_fixed_consume, edge_nonterminal_use, operation_fixed_consume,
            operation_nonterminal_use, terminator_fixed_consume, terminator_nonterminal_use,
        };
        let ops = non_scheduled_ops(block).collect::<Vec<_>>();
        let named = ops
            .iter()
            .enumerate()
            .filter(|(_, operation)| {
                operation_nonterminal_use(&operation.kind, owner)
                    || operation_fixed_consume(&operation.kind, owner)
                    || destination(&operation.kind) == Some(owner)
            })
            .map(|(position, operation)| {
                (
                    position,
                    destination(&operation.kind) == Some(owner),
                    operation_nonterminal_use(&operation.kind, owner),
                    operation_fixed_consume(&operation.kind, owner),
                )
            })
            .collect::<Vec<_>>();
        let indexed = self
            .operations
            .iter()
            .map(|operation| {
                (
                    operation.position,
                    operation.defines,
                    operation.nonterminal,
                    operation.fixed,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(indexed, named, "indexed operation mentions diverged");
        assert_eq!(
            self.redefines,
            named.iter().any(|(_, defines, _, _)| *defines),
            "indexed redefinition diverged"
        );
        assert_eq!(
            self.param,
            block.params.iter().any(|param| param.owner == owner),
            "indexed parameter mention diverged"
        );
        assert_eq!(
            (self.terminator_nonterminal, self.terminator_fixed),
            (
                terminator_nonterminal_use(&block.terminator, owner),
                terminator_fixed_consume(&block.terminator, owner)
            ),
            "indexed terminator mention diverged"
        );
        let edges = block
            .terminator
            .edges()
            .enumerate()
            .filter(|(_, edge)| {
                edge_nonterminal_use(edge, owner) || edge_fixed_consume(edge, owner)
            })
            .map(|(index, edge)| {
                (
                    index,
                    edge.id,
                    edge_nonterminal_use(edge, owner),
                    edge_fixed_consume(edge, owner),
                )
            })
            .collect::<Vec<_>>();
        let indexed_edges = self
            .edges
            .iter()
            .map(|edge| (edge.index, edge.id, edge.nonterminal, edge.fixed))
            .collect::<Vec<_>>();
        assert_eq!(indexed_edges, edges, "indexed edge mentions diverged");
    }
}

/// Call `note` with every owner `op` names as an operand and whether that
/// operand ends it: an operand that ends the owner is a fixed consume, any
/// other is a non-terminal use. A `Drop`, `Discard` or `RootConsume` ends its
/// owner whatever the operand's disposition; any other operand ends it only
/// when it moves it.
fn for_each_operand(op: &Op, mut note: impl FnMut(OwnerId, bool)) {
    match op {
        Op::Apply { args, .. } => {
            for arg in args {
                note(arg.owner, arg.use_ == OwnershipUse::Move);
            }
        }
        Op::Copy { source, .. } | Op::Project { source } => {
            note(source.owner, source.use_ == OwnershipUse::Move);
        }
        Op::LoopItem { list, .. } => note(list.owner, list.use_ == OwnershipUse::Move),
        Op::Drop { owner: operand } | Op::RootConsume { owner: operand, .. } => {
            note(operand.owner, true);
        }
        Op::Discard { owner } => note(*owner, true),
        Op::Define { .. } => {}
    }
}

/// The owner a terminator's own operand names, with whether it reads it
/// without ending it and whether it moves it out. A branch condition, match
/// scrutinee or loop list is read whatever its disposition; a `Return` result
/// ends the owner only when it moves it. Edge arguments are separate.
fn terminator_operand(terminator: &Terminator) -> Option<(OwnerId, bool, bool)> {
    match terminator {
        Terminator::Branch { condition, .. } => Some((condition.owner, true, false)),
        Terminator::Match { scrutinee, .. } => Some((scrutinee.owner, true, false)),
        Terminator::Loop { list, .. } => Some((list.owner, true, false)),
        Terminator::Return { result } => {
            Some((result.owner, false, result.use_ == OwnershipUse::Move))
        }
        Terminator::Jump(_) | Terminator::Exit => None,
    }
}
