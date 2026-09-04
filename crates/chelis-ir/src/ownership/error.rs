use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OwnershipError {
    #[error("ownership lowering has no checked signature for `{function}`")]
    MissingSignature { function: String },
    #[error("call to `{callee}` in `{unit}` supplies {supplied} arguments, expected {declared}")]
    CallArityMismatch {
        unit: String,
        callee: String,
        supplied: usize,
        declared: usize,
    },
    #[error(
        "call argument {argument} to `{callee}` in `{unit}` has checked type `{actual}`, expected `{expected}`"
    )]
    CallArgumentType {
        unit: String,
        callee: String,
        argument: usize,
        expected: String,
        actual: String,
    },
    #[error("ownership lowering in `{unit}` names unknown callee `{callee}`")]
    UnknownCallee { unit: String, callee: String },
    #[error("ownership lowering in `{unit}` references unbound name `{name}`")]
    UnboundName { unit: String, name: String },
    #[error("manifest root `{root}` names missing host binding `{def_name}`")]
    ManifestRootWithoutBinding { root: String, def_name: String },
    #[error("match-option scrutinee in `{unit}` has type `{ty}`")]
    MatchScrutineeNotOption { unit: String, ty: String },
    #[error("host loop source in `{unit}` has type `{ty}`, expected a list")]
    LoopListNotList { unit: String, ty: String },
    #[error("first-class function value `{name}` is not representable in `{unit}` (chelis#879)")]
    FirstClassFunctionValue { unit: String, name: String },
    #[error("container type `{ty}` in `{unit}` holds a function value (chelis#879)")]
    FunctionContainer { unit: String, ty: String },
    #[error("ownership lowering invariant failed in `{unit}`: {detail}")]
    LoweringInvariant { unit: String, detail: String },
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
    #[error("direct call in `{caller}` names missing ownership unit u{unit}")]
    MissingUnit { caller: String, unit: u32 },
    #[error("direct call in `{caller}` targets non-function ownership unit u{unit}")]
    NonFunctionCallee { caller: String, unit: u32 },
    #[error("direct call in `{caller}` does not match ownership signature of u{unit}")]
    DirectCallSchema { caller: String, unit: u32 },
    #[error(
        "direct call argument {argument} in `{caller}` has class `{actual}`, callable u{unit} requires `{expected}`"
    )]
    DirectCallArgumentClass {
        caller: String,
        unit: u32,
        argument: usize,
        expected: String,
        actual: String,
    },
    #[error("function `{unit}` b{block} returns with `{actual}`, expected `move`")]
    FunctionReturnMode {
        unit: String,
        block: u32,
        actual: &'static str,
    },
    #[error(
        "function `{unit}` b{block} returns class `{actual}`, another reachable return has `{expected}`"
    )]
    FunctionReturnClass {
        unit: String,
        block: u32,
        expected: String,
        actual: String,
    },
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
    #[error(
        "operation `{label}` in `{unit}` b{block} has {actual} operands, schema requires {expected}"
    )]
    OperationArity {
        unit: String,
        block: u32,
        label: String,
        expected: usize,
        actual: usize,
    },
    #[error(
        "operation `{label}` result in `{unit}` b{block} has class `{actual}`, schema requires `{expected}`"
    )]
    OperationResultClass {
        unit: String,
        block: u32,
        label: String,
        expected: String,
        actual: String,
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
    #[error("heap owner %{owner} is discarded in `{unit}` b{block}")]
    HeapDiscard {
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
    #[error("host site map is not total: {detail}")]
    HostSiteMap { detail: String },
    #[error("DAG node n{node} references missing or non-dominating input n{input}")]
    DagInput { node: usize, input: usize },
    #[error("DAG node n{node} operation `{operation}` requires {expected} inputs, found {actual}")]
    DagArity {
        node: usize,
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("DAG owner n{owner} has more than one terminal ownership directive")]
    DagDuplicateTerminal { owner: usize },
    #[error("borrowed DAG owner n{owner} is consumed by n{consumer}")]
    DagBorrowConsumed { owner: usize, consumer: usize },
    #[error("DAG owner n{owner} is used by n{consumer} after its terminal ownership directive")]
    DagUseAfterTerminal { owner: usize, consumer: usize },
    #[error("owned DAG producer n{owner} has no terminal ownership directive")]
    DagMissingTerminal { owner: usize },
    #[error("DAG ownership directive map is not total: {detail}")]
    DagDirectiveMap { detail: String },
    #[error("DAG root n{root} does not name an owned producer")]
    DagInvalidRoot { root: usize },
}
