use std::collections::BTreeMap;

use crate::host_type_state::ConcreteHostType;

use super::classify::{Placement, ValueClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct OwnerId(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BlockId(pub(crate) u32);

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
        args: Vec<Operand>,
    },
    Copy {
        dest: OwnerId,
        source: Operand,
    },
    Drop {
        owner: Operand,
    },
    RootConsume {
        root: String,
        owner: Operand,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Edge {
    pub(crate) target: BlockId,
    pub(crate) args: Vec<Operand>,
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Block {
    pub(crate) id: BlockId,
    pub(crate) params: Vec<BlockParam>,
    pub(crate) ops: Vec<Op>,
    pub(crate) terminator: Terminator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnitKind {
    Roots,
    Function,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Unit {
    pub(crate) name: String,
    pub(crate) kind: UnitKind,
    pub(crate) entry: BlockId,
    pub(crate) blocks: Vec<Block>,
    pub(crate) owners: BTreeMap<OwnerId, OwnerInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OwnershipProgram {
    pub(crate) units: Vec<Unit>,
}
