use crate::prelude::*;
use alloc::collections::BTreeMap;

use crate::host_type_state::ConcreteHostType;

use super::classify::{HeapKind, Placement, ValueClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnerId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

/// Stable identity for one ownership unit.
/// Unlike the diagnostic unit name, this identity is assigned structurally and is the only admissible direct- call target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnitId(pub u32);

/// Stable identity for one operation within an ownership unit.
/// Scheduling may reorder operations without invalidating host-site associations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId(pub u32);

/// Stable structural identity for one control-flow edge.
/// It is assigned by lowering before host-site actions are recorded and survives scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeId(pub u32);

impl EdgeId {
    pub const UNASSIGNED: Self = Self(u32::MAX);
}

/// Ordinary identity for one structural host site.
/// This identity grants no native emission authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HostSiteId {
    key: HostSiteKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct HostSiteKey(u32);

impl HostSiteId {
    pub fn from_index(index: usize) -> Self {
        Self {
            key: HostSiteKey(index.try_into().expect("host site census exceeds u32")),
        }
    }

    pub fn index(self) -> usize {
        self.key.0 as usize
    }
}

/// The closed ownership-use algebra.
/// No unknown or backend-specific state can cross verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipUse {
    Borrow,
    Move,
    Clone,
}

impl OwnershipUse {
    pub fn name(self) -> &'static str {
        match self {
            Self::Borrow => "borrow",
            Self::Move => "move",
            Self::Clone => "clone",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    pub owner: OwnerId,
    pub use_: OwnershipUse,
}

impl Operand {
    pub fn borrow(owner: OwnerId) -> Self {
        Self {
            owner,
            use_: OwnershipUse::Borrow,
        }
    }

    pub fn move_(owner: OwnerId) -> Self {
        Self {
            owner,
            use_: OwnershipUse::Move,
        }
    }

    pub fn clone_(owner: OwnerId) -> Self {
        Self {
            owner,
            use_: OwnershipUse::Clone,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamMode {
    Owned,
    Borrowed,
    EntryBorrow,
}

impl ParamMode {
    pub fn use_(self) -> OwnershipUse {
        match self {
            Self::Owned => OwnershipUse::Move,
            Self::Borrowed | Self::EntryBorrow => OwnershipUse::Borrow,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnerOrigin {
    Owned,
    BorrowedFrom(OwnerId),
    ExternalBorrow,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OwnerInfo {
    pub ty: ConcreteHostType,
    pub class: ValueClass,
    pub placement: Placement,
    pub origin: OwnerOrigin,
    pub names: Vec<String>,
    /// Span id of the expression this owner was minted for, taken verbatim from `HostExpr::span_id` (see `spec/design/chelis_span_survival.md`).
    /// Surf lowering issues `surf:<start>..<end>`, but the value is an opaque producer-issued string.
    /// A Deep input may carry any id, including an empty one.
    /// `None` for owners minted outside an expression, such as unit parameters, and for programs whose Deep nodes carry no span (chelis#2122).
    pub span_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockParam {
    pub owner: OwnerId,
    pub mode: ParamMode,
}

/// The verifier's semantic input for an application.
/// `label` remains a diagnostic rendering only.
/// it cannot choose operand ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationSchema {
    pub operands: Vec<OwnershipUse>,
    pub result: Option<ValueClass>,
}

/// The builtins that build a container from a container operand of the same heap kind, which they may consume (chelis#2205).
/// This table is the one authority the last-use scheduler consults.
/// And its key is the callable's shape rather than its label alone.
/// A row matches only when the application's result carries the row's heap kind and the named operand carries it too.
/// Tensor `concat` shares the label `builtin:concat` but takes a `List[tensor]` of parts and produces a tensor.
///
/// So it is not a row here.
/// Every other builtin operand stays borrowed, and `Some`'s payload keeps its [05-OP-44] constructor clone.
/// The consuming runtime entry point mutates in place only when the strong-owner count is one.
/// So a retained alias (a tuple, an option, an ADT, or a callee that stored the container) still sees a clone.
/// The kind belongs to the row rather than being read off the application.
///
/// Result-class-equals-operand-class alone is satisfied by callables that are nothing like `append`: `chunk`, `map`, `flatten`.
/// `zip` and `enumerate` each take a list and return a list.
/// Naming the kind keeps membership a reviewed decision.
/// And it keeps a future `builtin:concat`0 out of the `builtin:concat`1 row by construction rather.
/// Than by an emitter guard firing after the scheduler has already retired the operand's terminal.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContainerConsumer {
    /// The application label the ownership lowering assigns.
    pub label: &'static str,
    /// The heap kind this row governs.
    /// Both the application's result and the named operand must carry it.
    pub kind: HeapKind,
    /// The operand position that may move.
    /// the callable's other operands are borrowed.
    pub operand: usize,
}

/// Every container-producing builtin that may consume a same-kind operand.
pub const CONTAINER_CONSUMERS: &[ContainerConsumer] = &[
    ContainerConsumer {
        label: "builtin:append",
        kind: HeapKind::List,
        operand: 0,
    },
    ContainerConsumer {
        label: "builtin:concat",
        kind: HeapKind::List,
        operand: 0,
    },
    // chelis#2334.
    // `skip` returns a suffix of its operand rather than a grown copy.
    // So the in-place story is the offset rather than a push.
    // The consuming entry point releases the leading elements and advances a private head.
    // The row's shape test still applies, so only a List-classed operand producing a List is upgraded.
    //
    ContainerConsumer {
        label: "builtin:skip",
        kind: HeapKind::List,
        operand: 0,
    },
    // chelis#3724.
    // `take` returns a prefix of its operand, so the in-place story is a truncation.
    // The consuming entry point releases the elements past the count and shortens the list, with no offset.
    ContainerConsumer {
        label: "builtin:take",
        kind: HeapKind::List,
        operand: 0,
    },
    ContainerConsumer {
        label: "builtin:dict_insert",
        kind: HeapKind::Dict,
        operand: 0,
    },
    ContainerConsumer {
        label: "builtin:dict_merge",
        kind: HeapKind::Dict,
        operand: 0,
    },
    ContainerConsumer {
        label: "builtin:dict_remove",
        kind: HeapKind::Dict,
        operand: 0,
    },
    ContainerConsumer {
        label: "builtin:string_concat",
        kind: HeapKind::String,
        operand: 0,
    },
];

/// The operand a container-producing builtin may consume.
/// Or `None` when the application is not a row of [`CONTAINER_CONSUMERS`]: the label is not listed.
/// The result does not carry the row's heap kind, or the named operand does not carry it.
/// `operand_class` answers the class of the operand at a position.
pub fn container_consumer_operand(
    label: &str,
    result: Option<ValueClass>,
    operand_class: impl Fn(usize) -> Option<ValueClass>,
) -> Option<usize> {
    let row = CONTAINER_CONSUMERS.iter().find(|row| row.label == label)?;
    let class = ValueClass::Heap(row.kind);
    if result != Some(class) {
        return None;
    }
    (operand_class(row.operand) == Some(class)).then_some(row.operand)
}

/// Closed semantic class for an application.
/// A free-form diagnostic label cannot create a direct call or select its callee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyKind {
    Intrinsic,
    IndirectCall,
    /// A source-bound provider admitted by the checker and host lowerer.
    NativeProviderCall,
    /// The indirect callee owner was produced by this closed key operation.
    CheckedBuiltinCall(crate::host_type_state::CheckedBuiltinCallable),
    DirectCall {
        callee: UnitId,
    },
}

impl OperationSchema {
    pub fn new(operands: Vec<OwnershipUse>, result: Option<ValueClass>) -> Self {
        Self { operands, result }
    }
}

/// Backend-neutral operations.
/// `Apply` covers calls, constructors, and reads.
/// lowering assigns each operand disposition before this representation exists.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
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
    /// This changes no ownership state: it gives emission a typed, spelling- independent binding for a value escaping a nested expression scope.
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
    /// A typed terminal for an owned nonheap identity.
    /// This is deliberately distinct from Apply: diagnostic spelling cannot turn an arbitrary operation into a discard.
    Discard {
        owner: OwnerId,
    },
    RootConsume {
        root: String,
        owner: Operand,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    pub id: OpId,
    pub role: OperationRole,
    pub kind: Op,
}

/// Typed scheduling authority.
/// Diagnostic labels and operation spelling never decide whether a terminal is movable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationRole {
    Semantic,
    ProvisionalScopeExit,
    ScheduledScopeExit,
}

/// The only operations admissible on a selected control-flow edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminal {
    Drop(OwnerId),
    Discard(OwnerId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeTerminal {
    pub id: OpId,
    pub kind: Terminal,
}

impl Terminal {
    pub fn owner(self) -> OwnerId {
        match self {
            Self::Drop(owner) | Self::Discard(owner) => owner,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub id: EdgeId,
    pub target: BlockId,
    pub args: Vec<Operand>,
    pub terminals: Vec<EdgeTerminal>,
}

#[cfg(test)]
impl Edge {
    pub fn new(id: EdgeId, target: BlockId, args: Vec<Operand>) -> Self {
        Self {
            id,
            target,
            args,
            terminals: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    Return {
        result: Operand,
    },
    Jump(Edge),
    Branch {
        condition: Operand,
        then_edge: Edge,
        else_edge: Edge,
    },
    /// Borrow a tagged scrutinee and select exactly one successor.
    /// Payload projection happens inside the selected block.
    /// the result itself still joins through an owned block parameter.
    Match {
        scrutinee: Operand,
        arms: Vec<Edge>,
    },
    /// Borrow the source list and move the carried owner to either the body or exit edge.
    /// A body back-edge supplies the next carried owner.
    Loop {
        list: Operand,
        body_edge: Edge,
        exit_edge: Edge,
    },
    Exit,
}

impl Terminator {
    pub fn edges(&self) -> Box<dyn Iterator<Item = &Edge> + '_> {
        match self {
            Self::Return { .. } | Self::Exit => Box::new(core::iter::empty()),
            Self::Jump(edge) => Box::new(core::iter::once(edge)),
            Self::Branch {
                then_edge,
                else_edge,
                ..
            }
            | Self::Loop {
                body_edge: then_edge,
                exit_edge: else_edge,
                ..
            } => Box::new(core::iter::once(then_edge).chain(core::iter::once(else_edge))),
            Self::Match { arms, .. } => Box::new(arms.iter()),
        }
    }

    pub fn edges_mut(&mut self) -> Box<dyn Iterator<Item = &mut Edge> + '_> {
        match self {
            Self::Return { .. } | Self::Exit => Box::new(core::iter::empty()),
            Self::Jump(edge) => Box::new(core::iter::once(edge)),
            Self::Branch {
                then_edge,
                else_edge,
                ..
            }
            | Self::Loop {
                body_edge: then_edge,
                exit_edge: else_edge,
                ..
            } => Box::new(core::iter::once(then_edge).chain(core::iter::once(else_edge))),
            Self::Match { arms, .. } => Box::new(arms.iter_mut()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub id: BlockId,
    pub params: Vec<BlockParam>,
    pub ops: Vec<Operation>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitKind {
    Roots,
    Function,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleState {
    Phase2ScopeExit,
    CanonicalLastUse,
}

/// Structural identity of the callable body boundary.
/// Authored functions may have a distinct `Unit::entry` adapter whose `EntryBorrow` parameters are copied before owned internal formals.
/// direct calls target this body entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallableBody {
    entry: BlockId,
}

impl CallableBody {
    pub fn new(entry: BlockId) -> Self {
        Self { entry }
    }

    pub fn entry(self) -> BlockId {
        self.entry
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    pub id: UnitId,
    pub name: String,
    pub kind: UnitKind,
    pub schedule: ScheduleState,
    pub callable_body: Option<CallableBody>,
    pub entry: BlockId,
    pub blocks: Vec<Block>,
    pub owners: BTreeMap<OwnerId, OwnerInfo>,
    /// The top-level binding each captured environment parameter borrows, spelled exactly as the host body reads it.
    /// A top-level label, or the bare source name.
    /// Lowering records it when it mints the capture.
    /// Later local aliases add their own spellings to the owner's `names`.
    /// And a local may be spelled like a top-level value, so the names cannot identify the capture (chelis#3839).
    ///
    pub captured_globals: BTreeMap<OwnerId, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OwnershipProgram {
    pub units: Vec<Unit>,
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
pub enum HostSiteAction {
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
pub struct HostSiteRecord {
    pub id: HostSiteId,
    pub unit: usize,
    pub kind: HostSiteKind,
    pub actions: Vec<HostSiteAction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostSiteMap {
    /// Private, so every read of a record goes through an accessor that counts the records it yields.
    /// A scan of the whole map per query is then visible to the site-lookup oracle however it is spelled (chelis#3847).
    records: Vec<HostSiteRecord>,
}

impl HostSiteMap {
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> impl ExactSizeIterator<Item = &HostSiteRecord> + Clone {
        self.records_observed(None)
    }
    pub fn records_observed<'a>(
        &'a self,
        observer: Option<&'a dyn crate::resources::Observer>,
    ) -> impl ExactSizeIterator<Item = &'a HostSiteRecord> + Clone {
        self.records
            .iter()
            .inspect(move |_| super::locations::note_site_lookup_entries(observer, 1))
    }
    pub fn record(&self, index: usize) -> Option<&HostSiteRecord> {
        self.record_observed(index, None)
    }
    pub fn record_observed(
        &self,
        index: usize,
        observer: Option<&dyn crate::resources::Observer>,
    ) -> Option<&HostSiteRecord> {
        super::locations::note_site_lookup_entries(observer, 1);
        self.records.get(index)
    }
    pub fn records_mut(&mut self) -> impl ExactSizeIterator<Item = &mut HostSiteRecord> {
        self.records_mut_observed(None)
    }
    pub fn records_mut_observed<'a>(
        &'a mut self,
        observer: Option<&'a dyn crate::resources::Observer>,
    ) -> impl ExactSizeIterator<Item = &'a mut HostSiteRecord> {
        self.records
            .iter_mut()
            .inspect(move |_| super::locations::note_site_lookup_entries(observer, 1))
    }
    pub fn record_mut(&mut self, index: usize) -> Option<&mut HostSiteRecord> {
        self.record_mut_observed(index, None)
    }
    pub fn record_mut_observed(
        &mut self,
        index: usize,
        observer: Option<&dyn crate::resources::Observer>,
    ) -> Option<&mut HostSiteRecord> {
        super::locations::note_site_lookup_entries(observer, 1);
        self.records.get_mut(index)
    }

    pub fn from_records(records: Vec<HostSiteRecord>) -> Self {
        Self { records }
    }

    /// The records for a test to build or corrupt a map directly, uncounted.
    pub fn raw_records(&self) -> &Vec<HostSiteRecord> {
        &self.records
    }

    #[cfg(test)]
    pub fn raw_records_mut(&mut self) -> &mut Vec<HostSiteRecord> {
        &mut self.records
    }
}

#[derive(Debug, Default)]
pub struct HostSiteBuilder {
    records: Vec<HostSiteRecord>,
}

impl HostSiteBuilder {
    pub fn add(&mut self, unit: usize, kind: HostSiteKind) -> HostSiteId {
        let id = HostSiteId::from_index(self.records.len());
        self.records.push(HostSiteRecord {
            id,
            unit,
            kind,
            actions: Vec::new(),
        });
        id
    }

    pub fn record(&mut self, id: HostSiteId, action: HostSiteAction) {
        self.records[id.index()].actions.push(action);
    }

    pub fn finish(self) -> HostSiteMap {
        HostSiteMap {
            records: self.records,
        }
    }
}
