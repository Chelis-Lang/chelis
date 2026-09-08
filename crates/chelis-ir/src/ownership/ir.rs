use std::collections::BTreeMap;

use crate::host_type_state::ConcreteHostType;

use super::classify::{Placement, ValueClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct OwnerId(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BlockId(pub(crate) u32);

/// Stable identity for one ownership unit. Unlike the diagnostic unit name,
/// this identity is assigned structurally and is the only admissible direct-
/// call target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct UnitId(pub(crate) u32);

/// Stable identity for one operation within an ownership unit. Scheduling may
/// reorder operations without invalidating host-site associations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct OpId(pub(crate) u32);

/// Stable structural identity for one control-flow edge. It is assigned by
/// lowering before host-site actions are recorded and survives scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct EdgeId(pub(crate) u32);

impl EdgeId {
    pub(crate) const UNASSIGNED: Self = Self(u32::MAX);
}

/// Opaque identity for one structurally selected host-emission site. The
/// numeric key is private: downstream code can compare identities but cannot
/// manufacture a carrier from an integer or derive ownership from spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HostSiteId {
    key: HostSiteKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct HostSiteKey(u32);

impl HostSiteId {
    pub(crate) fn from_index(index: usize) -> Self {
        Self {
            key: HostSiteKey(index.try_into().expect("host site census exceeds u32")),
        }
    }

    pub(crate) fn index(self) -> usize {
        self.key.0 as usize
    }
}

/// The closed ownership-use algebra. No unknown or backend-specific state can
/// cross verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OwnershipUse {
    Borrow,
    Move,
    Clone,
}

impl OwnershipUse {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Borrow => "borrow",
            Self::Move => "move",
            Self::Clone => "clone",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Operand {
    pub(crate) owner: OwnerId,
    pub(crate) use_: OwnershipUse,
}

impl Operand {
    pub(crate) fn borrow(owner: OwnerId) -> Self {
        Self {
            owner,
            use_: OwnershipUse::Borrow,
        }
    }

    pub(crate) fn move_(owner: OwnerId) -> Self {
        Self {
            owner,
            use_: OwnershipUse::Move,
        }
    }

    pub(crate) fn clone_(owner: OwnerId) -> Self {
        Self {
            owner,
            use_: OwnershipUse::Clone,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParamMode {
    Owned,
    Borrowed,
    EntryBorrow,
}

impl ParamMode {
    pub(crate) fn use_(self) -> OwnershipUse {
        match self {
            Self::Owned => OwnershipUse::Move,
            Self::Borrowed | Self::EntryBorrow => OwnershipUse::Borrow,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OwnerOrigin {
    Owned,
    BorrowedFrom(OwnerId),
    ExternalBorrow,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OwnerInfo {
    pub(crate) ty: ConcreteHostType,
    pub(crate) class: ValueClass,
    pub(crate) placement: Placement,
    pub(crate) origin: OwnerOrigin,
    pub(crate) names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BlockParam {
    pub(crate) owner: OwnerId,
    pub(crate) mode: ParamMode,
}

/// The verifier's semantic input for an application. `label` remains a
/// diagnostic rendering only; it cannot choose operand ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OperationSchema {
    pub(crate) operands: Vec<OwnershipUse>,
    pub(crate) result: Option<ValueClass>,
}

/// Closed semantic class for an application. A free-form diagnostic label
/// cannot create a direct call or select its callee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApplyKind {
    Intrinsic,
    IndirectCall,
    DirectCall { callee: UnitId },
}

impl OperationSchema {
    pub(crate) fn new(operands: Vec<OwnershipUse>, result: Option<ValueClass>) -> Self {
        Self { operands, result }
    }
}

/// Backend-neutral operations. `Apply` covers calls, constructors, and reads;
/// lowering assigns each operand disposition before this representation exists.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Op {
    Define {
        dest: OwnerId,
        label: String,
    },
    Apply {
        dest: Option<OwnerId>,
        label: String,
        kind: ApplyKind,
        schema: OperationSchema,
        args: Vec<Operand>,
    },
    Copy {
        dest: OwnerId,
        source: Operand,
    },
    /// Project an existing logical owner into the current host payload slot.
    /// This changes no ownership state: it gives emission a typed, spelling-
    /// independent binding for a value escaping a nested expression scope.
    Project {
        source: Operand,
    },
    /// Materialize the current element of a verified host list loop.
    LoopItem {
        dest: OwnerId,
        list: Operand,
    },
    Drop {
        owner: Operand,
    },
    /// A typed terminal for an owned nonheap identity. This is deliberately
    /// distinct from Apply: diagnostic spelling cannot turn an arbitrary
    /// operation into a discard.
    Discard {
        owner: OwnerId,
    },
    RootConsume {
        root: String,
        owner: Operand,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Operation {
    pub(crate) id: OpId,
    pub(crate) role: OperationRole,
    pub(crate) kind: Op,
}

/// Typed scheduling authority. Diagnostic labels and operation spelling never
/// decide whether a terminal is movable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OperationRole {
    Semantic,
    ProvisionalScopeExit,
    ScheduledScopeExit,
}

/// The only operations admissible on a selected control-flow edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Terminal {
    Drop(OwnerId),
    Discard(OwnerId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EdgeTerminal {
    pub(crate) id: OpId,
    pub(crate) kind: Terminal,
}

impl Terminal {
    pub(crate) fn owner(self) -> OwnerId {
        match self {
            Self::Drop(owner) | Self::Discard(owner) => owner,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Edge {
    pub(crate) id: EdgeId,
    pub(crate) target: BlockId,
    pub(crate) args: Vec<Operand>,
    pub(crate) terminals: Vec<EdgeTerminal>,
}

#[cfg(test)]
impl Edge {
    pub(crate) fn new(id: EdgeId, target: BlockId, args: Vec<Operand>) -> Self {
        Self {
            id,
            target,
            args,
            terminals: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Terminator {
    Return {
        result: Operand,
    },
    Jump(Edge),
    Branch {
        condition: Operand,
        then_edge: Edge,
        else_edge: Edge,
    },
    /// Borrow a tagged scrutinee and select exactly one successor. Payload
    /// projection happens inside the selected block; the result itself still
    /// joins through an owned block parameter.
    Match {
        scrutinee: Operand,
        arms: Vec<Edge>,
    },
    /// Borrow the source list and move the carried owner to either the body
    /// or exit edge. A body back-edge supplies the next carried owner.
    Loop {
        list: Operand,
        body_edge: Edge,
        exit_edge: Edge,
    },
    Exit,
}

impl Terminator {
    pub(crate) fn edges(&self) -> Box<dyn Iterator<Item = &Edge> + '_> {
        match self {
            Self::Return { .. } | Self::Exit => Box::new(std::iter::empty()),
            Self::Jump(edge) => Box::new(std::iter::once(edge)),
            Self::Branch {
                then_edge,
                else_edge,
                ..
            }
            | Self::Loop {
                body_edge: then_edge,
                exit_edge: else_edge,
                ..
            } => Box::new(std::iter::once(then_edge).chain(std::iter::once(else_edge))),
            Self::Match { arms, .. } => Box::new(arms.iter()),
        }
    }

    pub(crate) fn edges_mut(&mut self) -> Box<dyn Iterator<Item = &mut Edge> + '_> {
        match self {
            Self::Return { .. } | Self::Exit => Box::new(std::iter::empty()),
            Self::Jump(edge) => Box::new(std::iter::once(edge)),
            Self::Branch {
                then_edge,
                else_edge,
                ..
            }
            | Self::Loop {
                body_edge: then_edge,
                exit_edge: else_edge,
                ..
            } => Box::new(std::iter::once(then_edge).chain(std::iter::once(else_edge))),
            Self::Match { arms, .. } => Box::new(arms.iter_mut()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Block {
    pub(crate) id: BlockId,
    pub(crate) params: Vec<BlockParam>,
    pub(crate) ops: Vec<Operation>,
    pub(crate) terminator: Terminator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnitKind {
    Roots,
    Function,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduleState {
    Phase2ScopeExit,
    CanonicalLastUse,
}

/// Structural identity of the callable body boundary. Authored functions may
/// have a distinct `Unit::entry` adapter whose `EntryBorrow` parameters are
/// copied before owned internal formals; direct calls target this body entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CallableBody {
    entry: BlockId,
}

impl CallableBody {
    pub(crate) fn new(entry: BlockId) -> Self {
        Self { entry }
    }

    pub(crate) fn entry(self) -> BlockId {
        self.entry
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Unit {
    pub(crate) id: UnitId,
    pub(crate) name: String,
    pub(crate) kind: UnitKind,
    pub(crate) schedule: ScheduleState,
    pub(crate) callable_body: Option<CallableBody>,
    pub(crate) entry: BlockId,
    pub(crate) blocks: Vec<Block>,
    pub(crate) owners: BTreeMap<OwnerId, OwnerInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OwnershipProgram {
    pub(crate) units: Vec<Unit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostSiteKind {
    Binding,
    Expression,
    Argument,
    BranchEdge,
    MatchArm,
    LoopEdge,
    FunctionEntry,
    FunctionReturn,
    ManifestRoot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostSiteAction {
    Operation {
        unit: usize,
        block: BlockId,
        operation: OpId,
    },
    Terminator {
        unit: usize,
        block: BlockId,
    },
    ControlEdge {
        unit: usize,
        edge: EdgeId,
        source: BlockId,
        target: BlockId,
    },
    Root {
        unit: usize,
        manifest_index: Option<usize>,
        owner: OwnerId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostSiteRecord {
    pub(crate) id: HostSiteId,
    pub(crate) unit: usize,
    pub(crate) kind: HostSiteKind,
    pub(crate) actions: Vec<HostSiteAction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostSiteMap {
    pub(crate) records: Vec<HostSiteRecord>,
}

#[derive(Debug, Default)]
pub(crate) struct HostSiteBuilder {
    records: Vec<HostSiteRecord>,
}

impl HostSiteBuilder {
    pub(crate) fn add(&mut self, unit: usize, kind: HostSiteKind) -> HostSiteId {
        let id = HostSiteId::from_index(self.records.len());
        self.records.push(HostSiteRecord {
            id,
            unit,
            kind,
            actions: Vec::new(),
        });
        id
    }

    pub(crate) fn record(&mut self, id: HostSiteId, action: HostSiteAction) {
        self.records[id.index()].actions.push(action);
    }

    pub(crate) fn finish(self) -> HostSiteMap {
        HostSiteMap {
            records: self.records,
        }
    }
}
