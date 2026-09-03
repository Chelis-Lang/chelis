use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OwnershipError {
    #[error("ownership program has {actual} roots units, expected exactly one")]
    RootUnitCount { actual: usize },
    #[error("duplicate {kind} {id} in `{unit}`")]
    DuplicateIdentity {
        unit: String,
        kind: &'static str,
        id: u32,
    },
    #[error("missing block b{block} in `{unit}`")]
    MissingBlock { unit: String, block: u32 },
    #[error("unreachable block b{block} in `{unit}`")]
    UnreachableBlock { unit: String, block: u32 },
    #[error("owner %{owner} in `{unit}` has no {missing}")]
    IncompleteOwner {
        unit: String,
        owner: u32,
        missing: &'static str,
    },
    #[error("owner %{owner} in `{unit}` b{block} is not live")]
    OwnerNotLive {
        unit: String,
        owner: u32,
        block: u32,
    },
    #[error("owner %{owner} in `{unit}` b{block} has use `{actual}`, expected `{expected}`")]
    WrongUse {
        unit: String,
        owner: u32,
        block: u32,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("borrowed owner %{owner} is consumed in `{unit}` b{block}")]
    BorrowConsumed {
        unit: String,
        owner: u32,
        block: u32,
    },
    #[error("owner %{owner} is consumed while borrow %{borrow} is live in `{unit}` b{block}")]
    LiveBorrowAtConsume {
        unit: String,
        owner: u32,
        borrow: u32,
        block: u32,
    },
    #[error("owner %{owner} in `{unit}` has invalid class: {detail}")]
    BadClass {
        unit: String,
        owner: u32,
        detail: String,
    },
    #[error("edge b{block}->b{target} in `{unit}` has {actual} arguments, expected {expected}")]
    EdgeArity {
        unit: String,
        block: u32,
        target: u32,
        expected: usize,
        actual: usize,
    },
    #[error("block b{block} in `{unit}` has inconsistent live owners: {expected:?} vs {actual:?}")]
    JoinMismatch {
        unit: String,
        block: u32,
        expected: BTreeSet<u32>,
        actual: BTreeSet<u32>,
    },
    #[error("owner %{owner} reaches `{unit}` b{block} exit without a terminal use")]
    MissingTerminal {
        unit: String,
        owner: u32,
        block: u32,
    },
    #[error("nonheap owner %{owner} is dropped in `{unit}` b{block}")]
    NonHeapDrop {
        unit: String,
        owner: u32,
        block: u32,
    },
    #[error("root sink `{root}` appears outside the roots unit `{unit}`")]
    RootSinkOutsideRoots { unit: String, root: String },
    #[error("branch condition %{owner} is not bool in `{unit}` b{block}")]
    NonBoolBranch {
        unit: String,
        owner: u32,
        block: u32,
    },
    #[error("{kind} unit `{unit}` has invalid terminal `{terminal}`")]
    WrongTerminal {
        unit: String,
        kind: &'static str,
        terminal: &'static str,
    },
}
