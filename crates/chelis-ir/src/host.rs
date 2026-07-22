use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use chelis_deep::ast::{Atom, Expr, List};
use chelis_deep::decode_effect_kind;
use chelis_types::types::Prim;
use chelis_types::{BUILTIN_NAMES, CheckedProgram};
use chelis_vocab::EffectKind;

use crate::dag::{DimExpr, DimInfo, RiscOp, TensorType};
use crate::host_type_state::{
    ConcreteHostType, HostInferenceVar, HostPrecisionTerm, HostTypeDecodeError, HostTypeTerm,
    decode_host_type,
};
use crate::lower::top_level_lowering_map;

thread_local! {
    // Tracks top-level callee names currently being inlined by
    // `inline_top_level_host_call`. Prevents infinite specialization for
    // recursive/mutually recursive definitions — the specialized body would
    // re-encounter the same call and inline forever.
    static INLINING_STACK: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

static NEXT_HOST_INFERENCE_VAR: AtomicU32 = AtomicU32::new(0);

fn fresh_host_inference() -> HostTypeTerm {
    HostTypeTerm::InferenceVariable(HostInferenceVar(
        NEXT_HOST_INFERENCE_VAR.fetch_add(1, Ordering::Relaxed),
    ))
}

fn is_inlining(name: &str) -> bool {
    INLINING_STACK.with(|stack| stack.borrow().contains(name))
}

fn push_inlining(name: &str) -> bool {
    INLINING_STACK.with(|stack| stack.borrow_mut().insert(name.to_string()))
}

fn pop_inlining(name: &str) {
    INLINING_STACK.with(|stack| {
        stack.borrow_mut().remove(name);
    });
}

#[derive(Debug, Clone)]
pub struct CompiledProgram {
    pub dag: Option<crate::Dag>,
    pub host: Option<ConcreteHostProgram>,
}

#[derive(Debug, Clone)]
pub struct HostProgram<T = HostTypeTerm> {
    pub globals: Vec<HostBinding<T>>,
    pub global_tensor_helpers: Vec<HostTensorHelper>,
    pub functions: Vec<HostFunction<T>>,
    /// Structured rejections collected during sparse-helper summary
    /// recognition. Populated by [`lower_compiled_program`] when a
    /// near-summary-eligible callsite (helper body contains a sparse
    /// `RiscOp::Gather`, `RiscOp::ScatterAdd`, or `RiscOp::Scatter`)
    /// is rejected by the recognizer.
    ///
    /// Each entry carries the helper identity, the callsite + helper
    /// body spans, the rejection class, and structured class-specific
    /// detail. Consumers (CLI diagnostic reporter, downstream tooling,
    /// red-team tests) MUST pattern-match on the enum variants and
    /// struct fields rather than parsing the `Display` rendering.
    ///
    /// See `crates/chelis-cli/tests/cross_library_semantic_gap_diagnostics.rs`
    /// for the acceptance oracle that locks the structured shape of
    /// each rejection class.
    pub summary_rejections: Vec<SummaryRejection>,
}

impl<T> Default for HostProgram<T> {
    fn default() -> Self {
        Self {
            globals: Vec::new(),
            global_tensor_helpers: Vec::new(),
            functions: Vec::new(),
            summary_rejections: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HostBinding<T = HostTypeTerm> {
    pub name: String,
    pub display_name: Option<String>,
    pub ty: T,
    pub value: HostExpr<T>,
}

#[derive(Debug, Clone)]
pub struct HostFunction<T = HostTypeTerm> {
    pub name: String,
    pub params: Vec<HostParam<T>>,
    pub ret_ty: T,
    pub body: HostExpr<T>,
    pub tensor_helpers: Vec<HostTensorHelper>,
    pub specialization: Option<HostFunctionSpecialization>,
    /// Structured rejections collected from this function's tensor
    /// helpers and from the function-level summary-derivation pass.
    /// See `HostProgram::summary_rejections`.
    pub summary_rejections: Vec<SummaryRejection>,
}

#[derive(Debug, Clone)]
pub struct HostParam<T = HostTypeTerm> {
    pub name: String,
    pub ty: T,
}

pub type ConcreteHostProgram = HostProgram<ConcreteHostType>;
pub type ConcreteHostBinding = HostBinding<ConcreteHostType>;
pub type ConcreteHostFunction = HostFunction<ConcreteHostType>;
pub type ConcreteHostParam = HostParam<ConcreteHostType>;

#[derive(Debug, Clone)]
pub struct HostTensorHelper {
    pub name: String,
    pub dag: crate::Dag,
    pub inputs: Vec<HostTensorInput>,
    pub output: TensorType,
    pub specialization: Option<HostTensorSpecialization>,
    /// Partial rejection captured at helper-construction time when the
    /// helper body is summary-near-eligible (root op is one of the
    /// three sparse RiscOps, or contains one in a recognizable place)
    /// but the summarizer rejected it.
    ///
    /// Carries everything the helper itself knows: rejection class,
    /// structured detail, and helper-body span. The owning function's
    /// name (the `HelperPath`) and the callsite span are filled in by
    /// [`derive_host_function_specializations`] when the function-level
    /// pass walks each fn's tensor helpers; the per-helper data is
    /// then promoted into a fully-formed `SummaryRejection` on
    /// `HostFunction::summary_rejections` and
    /// `HostProgram::summary_rejections`.
    pub summary_rejection: Option<HelperSummaryRejection>,
}

#[derive(Debug, Clone)]
pub struct HostTensorInput {
    pub name: String,
    pub ty: TensorType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostTensorSpecialization {
    BlasMatmul(HostBlasMatmulSummary),
    SparseGather(HostSparseOpSummary),
    SparseScatterAdd(HostSparseOpSummary),
    SparseScatterReplace(HostSparseOpSummary),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostFunctionSpecialization {
    BlasMatmul(HostBlasMatmulSummary),
    SparseGather(HostSparseOpSummary),
    SparseScatterAdd(HostSparseOpSummary),
    SparseScatterReplace(HostSparseOpSummary),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostBlasMatmulSummary {
    pub lhs_input: usize,
    pub rhs_input: usize,
    pub input_tys: Vec<TensorType>,
    pub output: TensorType,
    pub batch_dims: Vec<DimExpr>,
    pub m: DimExpr,
    pub n: DimExpr,
    pub k: DimExpr,
}

/// Compiler-derived summary for a sparse helper body (`Gather`,
/// `ScatterAdd`, or `Scatter`/replace).
///
/// `input_indices` is the ordered list of helper input positions that
/// supply the sparse op's operands. The order matches the RiscOp's
/// `inputs` order:
///   * Gather: `[values, indices]`
///   * ScatterAdd / Scatter: `[target, indices, updates]`
///
/// `input_tys` is the full ordered tuple of helper-input types (one
/// entry per helper input parameter, not just the sparse operands).
/// `output` is the tensor type the sparse op produces. The remapper
/// uses `input_tys` to verify that the callsite arg types still match
/// the recognized helper-body types after wrapper propagation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostSparseOpSummary {
    pub axis: usize,
    pub input_indices: Vec<usize>,
    pub input_tys: Vec<TensorType>,
    pub output: TensorType,
}

// =========================================================================
// W4-A — M5(c) structured rejection diagnostics
// =========================================================================
//
// `SummaryRejection` is the public diagnostic shape emitted when a
// callsite is summary-eligible (helper body contains a sparse RiscOp)
// but the summarizer rejects it. The shape is intentionally
// **programmatically matchable**: downstream consumers must pattern-
// match on `rejection_class` (an enum) and the strongly-typed
// `detail` enum payload rather than parsing the rendered `Display`
// string. The acceptance oracle for this contract is
// `crates/chelis-cli/tests/cross_library_semantic_gap_diagnostics.rs`.
//
// Variant naming mirrors W3-B's seven enumerated rejection categories
// (see `crates/chelis-ir/tests/host_sparse_summary.rs` and the cli
// counterparts in `crates/chelis-cli/tests/cross_library_sparse_summaries.rs`):
//
//   1. `MultipleRoots`               — helper body has multiple DAG roots
//   2. `MultipleReturnPaths`         — helper body branches via if/then/else
//   3. `NonLoadOperand`              — sparse-op operand is not a direct Load
//   4. `PostProcessingAfterSparseOp` — helper post-processes the sparse result
//   5. `IndicesDTypeMismatch`        — indices precision not int32/int64
//   6. `PayloadDTypeMismatch`        — values/target/updates/output disagree
//   7. `WildcardDim`                 — `Named("*", None)` placeholder in input/output
//
// Three additional variants (`UnrecognizedShape`, `NonContiguousLayout`,
// `RankMismatch`) are reserved for future helper-shape categories that
// the current summarizer does not check today but the plan's pinned
// shape names explicitly.

/// Identifies the rejected helper for diagnostic attribution.
///
/// `module` is the producing module path (today always `None` — the
/// host program is flat — but reserved for the Reef/library-import
/// case where helpers come from imported modules and a fully qualified
/// helper identity matters).
///
/// `def_name` is the Surf `def` name of the function whose body owns
/// the rejected tensor helper. For a top-level expression binding
/// (e.g. `result = gather(...)`) this is the synthetic global name
/// (`__global` etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelperPath {
    pub module: Option<String>,
    pub def_name: String,
}

impl HelperPath {
    /// Convenience constructor for the module-less (flat-program) case.
    pub fn local(def_name: impl Into<String>) -> Self {
        Self {
            module: None,
            def_name: def_name.into(),
        }
    }
}

impl fmt::Display for HelperPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(module) = &self.module {
            write!(f, "{module}.{}", self.def_name)
        } else {
            write!(f, "{}", self.def_name)
        }
    }
}

/// Which payload role a `PayloadDTypeMismatch` is reporting on. The
/// sparse RiscOps have different operand roles:
///   * `Gather`: only `Values` is a payload (the indexed-into tensor);
///     the output type's precision must match.
///   * `ScatterAdd` / `Scatter`: `Target` (the base), `Updates` (the
///     scattered values), and the output type's precision must all
///     match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadRole {
    Values,
    Target,
    Updates,
    Output,
}

impl fmt::Display for PayloadRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PayloadRole::Values => "values",
            PayloadRole::Target => "target",
            PayloadRole::Updates => "updates",
            PayloadRole::Output => "output",
        };
        f.write_str(s)
    }
}

/// Which tensor (input or output) carries the wildcard `Named("*",
/// None)` dim that disqualified the helper. `Input(i)` is the
/// positional helper input index (matches `HostSparseOpSummary::input_indices`
/// indexing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WildcardLocation {
    Output,
    Input(usize),
}

impl fmt::Display for WildcardLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WildcardLocation::Output => f.write_str("output"),
            WildcardLocation::Input(i) => write!(f, "input[{i}]"),
        }
    }
}

/// Which sparse RiscOp the helper's body root names.
///
/// `Unknown` covers the case where the body root isn't a sparse op at
/// all (the helper is post-processing a sparse op, or the root is
/// something else entirely). The summarizer reports the rejection
/// against the deepest sparse op it finds in the helper body so the
/// diagnostic still names a specific op when possible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseOpKind {
    Gather,
    ScatterAdd,
    ScatterReplace,
    Unknown,
}

impl fmt::Display for SparseOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SparseOpKind::Gather => "gather",
            SparseOpKind::ScatterAdd => "scatter_add",
            SparseOpKind::ScatterReplace => "scatter_replace",
            SparseOpKind::Unknown => "<unknown>",
        };
        f.write_str(s)
    }
}

/// Closed enumeration of summary-rejection classes. Each variant is a
/// distinct public contract; consumers MUST match on the variant. New
/// classes are additive: adding a variant is a breaking change for
/// exhaustive matches but never silently re-routes an existing case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SummaryRejectionClass {
    /// Helper body has more than one DAG root (multi-output helper).
    /// Distinct from `MultipleReturnPaths` because the multi-root case
    /// is a DAG-structural property; the multi-return-path case is a
    /// host-side host-expr-shape property (the body lowered to `If`).
    MultipleRoots,
    /// Helper body branches via if/then/else (or another host-side
    /// conditional). Detected by the function-level pass when the
    /// function body is a `HostExprKind::If` rather than a `TensorCall`.
    MultipleReturnPaths,
    /// One of the sparse-op operands is not a direct `RiscOp::Load`
    /// of a helper input (e.g. an intervening elementwise op, a cast,
    /// or a reshape).
    NonLoadOperand,
    /// The helper body has an extra op on top of the sparse op (e.g.
    /// `add(gather(...), zero)`). The sparse op is not the DAG root.
    PostProcessingAfterSparseOp,
    /// The indices operand's precision is neither `int32` nor `int64`.
    IndicesDTypeMismatch,
    /// One of the payload precisions (values, target, updates, output)
    /// disagrees with the others.
    PayloadDTypeMismatch,
    /// Helper input or output carries a `Named("*", None)` wildcard
    /// dim — a type-inference placeholder that does not bind to a
    /// unique callsite axis.
    WildcardDim,
    /// Reserved: helper body shape does not match any recognized
    /// sparse-op pattern. Today the summarizer routes most "shape
    /// doesn't match" cases through `NonLoadOperand` /
    /// `PostProcessingAfterSparseOp`; this variant is in the public
    /// surface for forward-compatibility with future shape extensions.
    UnrecognizedShape,
    /// Reserved: helper body uses non-contiguous tensor layouts that
    /// the summary contract requires to be contiguous. No current
    /// summarizer check fires this; reserved for future use.
    NonContiguousLayout,
    /// Reserved: helper input or output rank disagrees with the rank
    /// the sparse op requires. Today the summarizer rejects rank
    /// mismatches through `NonLoadOperand` (the type-equality check on
    /// the Load op fails); reserved for future explicit rank checks.
    RankMismatch,
    // ---------------------------------------------------------------
    // W6 Task A — BLAS-recognizer rejection classes.
    //
    // The BLAS recognizer (`try_summarize_blas_helper` in
    // `crates/chelis-ir/src/host.rs`) exposes six structural failure
    // points where a helper-DAG that *almost* matched
    // `RiscOp::BlasMatmul` was rejected. Each is a distinct
    // BLAS-prefixed variant: the source recognizer is encoded in the
    // variant name so tooling pattern-matching on the enum sees both
    // "what shape failed" and "which recognizer rejected it" without
    // needing to inspect an out-of-band recognizer-identity tag.
    //
    // The variant names are NOT collapsed with the sparse-named
    // equivalents (`MultipleRoots`, `NonLoadOperand`) even though the
    // detail payload shapes coincide today; future divergence is
    // cheap to absorb when each path owns its own variant.
    // ---------------------------------------------------------------
    /// BLAS helper DAG (post-specialize) has more than one root.
    /// Mirrors `MultipleRoots` for sparse helpers but identifies the
    /// BLAS recognizer as the source.
    BlasMultipleRoots,
    /// BLAS helper's declared output precision is not `f32`. Today
    /// the recognizer requires `f32` output for the BlasMatmul
    /// path; non-`f32` outputs (e.g. an `f64` matmul helper) silently
    /// skipped through `Option::None` before W6 — now they are
    /// diagnosed.
    BlasOutputPrecisionMismatch,
    /// The helper's specialized DAG root is not `RiscOp::BlasMatmul`,
    /// so the recognizer could not extract `batch_dims`, `m`, `n`,
    /// `k`. Includes both the "root op is unrelated" case and the
    /// "root op shape doesn't match BlasMatmul's expected operand
    /// count / output precision" rejection.
    BlasNotMatmulPattern,
    /// One of the matmul operands is not a direct `RiscOp::Load` of a
    /// helper input. Mirrors `NonLoadOperand` for sparse helpers but
    /// identifies the BLAS recognizer as the source.
    BlasNonLoadOperand,
    /// At least one of the helper's input tensors has precision other
    /// than `f32`. Today the recognizer requires every helper input to
    /// be `f32`; mixed-precision inputs (e.g. an `f32 @ int8` quantized
    /// matmul helper) silently skipped before W6.
    BlasInputPrecisionMismatch,
    /// One of `batch_dims`, `m`, `n`, `k` could not be bound to any
    /// helper input dim by name. The recognizer requires every
    /// matmul-derived dim symbol to appear on at least one input
    /// tensor type's dim list; an unbindable dim means the helper
    /// signature does not name its own contraction axes.
    BlasDimensionBindingFailure,
}

impl fmt::Display for SummaryRejectionClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SummaryRejectionClass::MultipleRoots => "multiple-roots",
            SummaryRejectionClass::MultipleReturnPaths => "multiple-return-paths",
            SummaryRejectionClass::NonLoadOperand => "non-load-operand",
            SummaryRejectionClass::PostProcessingAfterSparseOp => "post-processing-after-sparse-op",
            SummaryRejectionClass::IndicesDTypeMismatch => "indices-dtype-mismatch",
            SummaryRejectionClass::PayloadDTypeMismatch => "payload-dtype-mismatch",
            SummaryRejectionClass::WildcardDim => "wildcard-dim",
            SummaryRejectionClass::UnrecognizedShape => "unrecognized-shape",
            SummaryRejectionClass::NonContiguousLayout => "non-contiguous-layout",
            SummaryRejectionClass::RankMismatch => "rank-mismatch",
            SummaryRejectionClass::BlasMultipleRoots => "blas-multiple-roots",
            SummaryRejectionClass::BlasOutputPrecisionMismatch => "blas-output-precision-mismatch",
            SummaryRejectionClass::BlasNotMatmulPattern => "blas-not-matmul-pattern",
            SummaryRejectionClass::BlasNonLoadOperand => "blas-non-load-operand",
            SummaryRejectionClass::BlasInputPrecisionMismatch => "blas-input-precision-mismatch",
            SummaryRejectionClass::BlasDimensionBindingFailure => "blas-dimension-binding-failure",
        };
        f.write_str(s)
    }
}

/// Class-specific structured payload for a `SummaryRejection`. Each
/// variant mirrors a `SummaryRejectionClass` variant, but only the
/// classes that carry additional structured information have a payload;
/// the payload-less classes use the unit `*Empty` variants. This split
/// keeps the public surface programmatically matchable on
/// (`rejection_class`, `detail`) jointly without forcing every consumer
/// to inspect `detail` for an empty payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SummaryRejectionDetail {
    MultipleRoots {
        /// Number of DAG roots observed in the helper body.
        root_count: usize,
    },
    MultipleReturnPaths {
        /// Approximate number of return-arm branches observed in the
        /// function body. Today the host lowerer collapses multiple
        /// branches into a single nested `If`; this field records the
        /// branch count at the outermost level.
        branch_count: usize,
    },
    NonLoadOperand {
        /// Sparse op the helper's body root names.
        op: SparseOpKind,
        /// Positional index of the operand that wasn't a Load (0 =
        /// first operand, 1 = second, etc.). Matches the RiscOp's
        /// `inputs` ordering.
        operand_index: usize,
    },
    PostProcessingAfterSparseOp {
        /// Sparse op that was post-processed (the deepest sparse op in
        /// the helper body).
        op: SparseOpKind,
        /// Snake-case name of the tail op sitting on top of the sparse
        /// op (e.g. `"add"`, `"reshape"`).
        tail_op: String,
    },
    IndicesDTypeMismatch {
        op: SparseOpKind,
        observed: Prim,
    },
    PayloadDTypeMismatch {
        op: SparseOpKind,
        /// Which payload role's dtype did not match the others.
        which: PayloadRole,
        /// Expected precision (the output / target / values precision
        /// the others should have matched).
        expected: Prim,
        /// Observed precision on the mismatching payload.
        observed: Prim,
    },
    WildcardDim {
        location: WildcardLocation,
    },
    /// Empty payload for the reserved classes (UnrecognizedShape,
    /// NonContiguousLayout, RankMismatch). Carries no structured
    /// information today.
    Reserved,
    // ---------------------------------------------------------------
    // W6 Task A — BLAS-recognizer detail payloads.
    //
    // Each variant mirrors a `SummaryRejectionClass::Blas*` variant.
    // The payloads name the observed-precision / failing-dim values
    // so tooling can distinguish e.g. "f64 helper rejected" from
    // "int32 helper rejected" without re-running the recognizer.
    // ---------------------------------------------------------------
    BlasMultipleRoots {
        /// Number of DAG roots observed in the helper's
        /// post-specialize body.
        root_count: usize,
    },
    BlasOutputPrecisionMismatch {
        /// Helper's declared output precision (the one that disagreed
        /// with the recognizer's required `f32`).
        observed: Prim,
    },
    BlasNotMatmulPattern {
        /// Snake-case canonical name of the root op the recognizer
        /// observed in place of `BlasMatmul` (e.g. `"add"`,
        /// `"reshape"`, or `"<other>"` for ops outside the canonical
        /// name whitelist). When the root IS `BlasMatmul` but its
        /// rank/precision doesn't match (e.g. wrong input count or
        /// non-`f32` matmul output), `tail_op` is `"blas_matmul"` and
        /// the caller still sees this variant.
        tail_op: String,
    },
    BlasNonLoadOperand {
        /// Positional index of the operand that was not a direct
        /// `Load` (0 = lhs, 1 = rhs).
        operand_index: usize,
    },
    BlasInputPrecisionMismatch {
        /// Positional index of the helper input whose precision was
        /// not `f32`.
        input_index: usize,
        /// Observed precision on the mismatching helper input.
        observed: Prim,
    },
    BlasDimensionBindingFailure {
        /// Symbolic role of the dim that could not be bound: one of
        /// `"batch"`, `"m"`, `"n"`, or `"k"`. The recognizer reports
        /// the first failing role.
        role: BlasDimRole,
    },
}

/// Which matmul dim symbol failed to bind to any helper input in the
/// `BlasDimensionBindingFailure` rejection. The four roles match the
/// `HostBlasMatmulSummary` field layout (`batch_dims`, `m`, `n`, `k`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlasDimRole {
    Batch,
    M,
    N,
    K,
}

impl fmt::Display for BlasDimRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BlasDimRole::Batch => "batch",
            BlasDimRole::M => "m",
            BlasDimRole::N => "n",
            BlasDimRole::K => "k",
        };
        f.write_str(s)
    }
}

impl fmt::Display for SummaryRejectionDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SummaryRejectionDetail::MultipleRoots { root_count } => {
                write!(f, "{root_count} DAG roots in helper body")
            }
            SummaryRejectionDetail::MultipleReturnPaths { branch_count } => {
                write!(f, "{branch_count} return-path branches")
            }
            SummaryRejectionDetail::NonLoadOperand { op, operand_index } => write!(
                f,
                "{op} operand[{operand_index}] is not a direct load of a helper input"
            ),
            SummaryRejectionDetail::PostProcessingAfterSparseOp { op, tail_op } => {
                write!(
                    f,
                    "{tail_op} on top of {op}; sparse op is not the helper root"
                )
            }
            SummaryRejectionDetail::IndicesDTypeMismatch { op, observed } => {
                write!(f, "{op} indices dtype {observed:?} is not int32 / int64")
            }
            SummaryRejectionDetail::PayloadDTypeMismatch {
                op,
                which,
                expected,
                observed,
            } => write!(
                f,
                "{op} payload `{which}` dtype {observed:?} disagrees with expected {expected:?}"
            ),
            SummaryRejectionDetail::WildcardDim { location } => {
                write!(f, "wildcard dim on helper {location}")
            }
            SummaryRejectionDetail::Reserved => f.write_str("<reserved>"),
            SummaryRejectionDetail::BlasMultipleRoots { root_count } => {
                write!(f, "{root_count} DAG roots in BLAS helper body")
            }
            SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } => {
                write!(f, "BLAS matmul output precision {observed:?} is not f32")
            }
            SummaryRejectionDetail::BlasNotMatmulPattern { tail_op } => write!(
                f,
                "BLAS helper root op `{tail_op}` does not match the BlasMatmul pattern"
            ),
            SummaryRejectionDetail::BlasNonLoadOperand { operand_index } => write!(
                f,
                "BLAS matmul operand[{operand_index}] is not a direct load of a helper input"
            ),
            SummaryRejectionDetail::BlasInputPrecisionMismatch {
                input_index,
                observed,
            } => write!(
                f,
                "BLAS helper input[{input_index}] precision {observed:?} is not f32"
            ),
            SummaryRejectionDetail::BlasDimensionBindingFailure { role } => write!(
                f,
                "BLAS matmul dim `{role}` could not be bound to any helper input"
            ),
        }
    }
}

/// Public diagnostic shape for a rejected summary callsite. This is
/// the central correctness contract of W4-A. Downstream consumers
/// (CLI diagnostic reporter, red-team tests, future tooling) MUST
/// pattern-match on the enum variants and struct fields rather than
/// parsing the `Display` rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryRejection {
    pub rejection_class: SummaryRejectionClass,
    pub helper_path: HelperPath,
    /// Surf-source span ID of the callsite that would have consumed
    /// the summary (`surf:<start>..<end>` per
    /// `spec/design/chelis_span_survival.md` §1, threaded through
    /// host lowering as `HostExpr::span_id`). `None` only when the
    /// caller chain has no surf-side span at all (synthesized code).
    pub callsite_span: Option<String>,
    /// Surf-source span ID of the helper body itself (the helper-DAG
    /// root). `None` for synthetic helpers that never had a Surf
    /// span attached (rare; matches the `HostExpr::span_id`
    /// convention).
    pub helper_body_span: Option<String>,
    pub detail: SummaryRejectionDetail,
}

impl fmt::Display for SummaryRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let callsite = self.callsite_span.as_deref().unwrap_or("<no-span>");
        let body = self.helper_body_span.as_deref().unwrap_or("<no-span>");
        write!(
            f,
            "rejected summary for `{}` ({}): {}; callsite={}, helper-body={}",
            self.helper_path, self.rejection_class, self.detail, callsite, body,
        )
    }
}

/// Partial rejection captured at tensor-helper construction time, when
/// the owning function's name + callsite span are not yet known. The
/// function-level pass ([`derive_host_function_specializations`])
/// promotes each `HelperSummaryRejection` into a fully-formed
/// `SummaryRejection` on `HostFunction::summary_rejections`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelperSummaryRejection {
    pub rejection_class: SummaryRejectionClass,
    pub helper_body_span: Option<String>,
    pub detail: SummaryRejectionDetail,
}

#[derive(Debug, Clone)]
pub struct HostCallback<T = HostTypeTerm> {
    pub kind: HostCallbackKind<T>,
    pub ret_ty: T,
}

#[derive(Debug, Clone)]
pub struct HostMatchArm<T = HostTypeTerm> {
    pub ctor: String,
    pub bindings: Vec<HostPatternBinding<T>>,
    pub expr: HostExpr<T>,
}

#[derive(Debug, Clone)]
pub struct HostPatternBinding<T = HostTypeTerm> {
    pub name: String,
    pub ty: T,
    pub field_index: usize,
}

#[derive(Debug, Clone)]
pub struct HostAdtField<T = HostTypeTerm> {
    pub name: Option<String>,
    pub ty: T,
}

#[derive(Debug, Clone)]
pub enum HostCallbackKind<T = HostTypeTerm> {
    Named {
        function: String,
        params: Vec<HostParam<T>>,
    },
    Inline {
        params: Vec<HostParam<T>>,
        body: Box<HostExpr<T>>,
    },
}

/// Host-lane expression with span-survival metadata.
///
/// `HostExpr` is a struct wrapper around `HostExprKind` carrying the
/// canonical span ID and any spans accumulated through N→1 merges, mirroring
/// the `DagNode` schema documented in `spec/design/chelis_span_survival.md`
/// §2.2 / §2.3 (S6 host-side rule table).
///
/// **Schema is locked.** Both fields stay even though the host-side passes
/// shipped today don't all produce N→1 merges; the schema mirrors `DagNode`
/// for orchestrator-tooling uniformity (one audit consumer reads both
/// schemas) and so future host-side optimizations have the field they need.
/// Reject simplification proposals on the same grounds the spec rejects
/// collapsing `DagNode.merged_spans` back to a single `Option<String>`.
#[derive(Debug, Clone)]
pub struct HostExpr<T = HostTypeTerm> {
    pub kind: HostExprKind<T>,
    /// Canonical span ID, populated by host-lane lowering from the Deep
    /// `Expr`'s `meta["span"]` value. Threaded through later host-side
    /// passes per the rules in `spec/design/chelis_span_survival.md` §2.3
    /// host-side table.
    ///
    /// `None` is the normal case for hand-written Chelis or for nodes
    /// synthesized in places where the source-region rule does not apply.
    pub span_id: Option<String>,
    /// Additional spans accumulated when N→1 host-side merge passes
    /// collapse multiple source nodes into a single result node.
    ///
    /// Backend host emission (S6 step 5) emits one `// span:` line per
    /// `span_id ∪ merged_spans` so the audit invariant holds: every span
    /// ID present on any input Deep node appears on at least one IR or
    /// HostExpr node.
    pub merged_spans: Vec<String>,
}

impl<T> HostExpr<T> {
    /// Construct a HostExpr from a kind with no span metadata. The host-side
    /// lowering layer (§2.3 host-side table, rule "Lowering") populates
    /// `span_id` from the enclosing Deep expr's `meta["span"]` via the
    /// `with_span` constructor; default constructions (e.g. tests) start
    /// span-free.
    pub fn new(kind: HostExprKind<T>) -> Self {
        Self {
            kind,
            span_id: None,
            merged_spans: Vec::new(),
        }
    }

    /// Construct a HostExpr from a kind with an explicit span ID. Empty
    /// `merged_spans`. Used by the host-lane lowering pass (`lower_host_expr`)
    /// to attach the current Deep expr's span to every freshly-produced node.
    pub fn with_span(kind: HostExprKind<T>, span_id: Option<String>) -> Self {
        Self {
            kind,
            span_id,
            merged_spans: Vec::new(),
        }
    }

    /// Append a single span to this node's `merged_spans`, lex-sorted and
    /// deduped, with the same no-op rules as `crate::span_merge::append_span_to_node`:
    ///   * `span` is `None` (passthrough),
    ///   * the node's `span_id` already equals `span`,
    ///   * `merged_spans` already contains `span`.
    ///
    /// Used by the host-side N→1 lowering collapse rule (§2.3 host-side
    /// table, rule "Lowering. Body collapses to existing HostExpr"): when
    /// a parent Deep expr lowers to an already-constructed inner HostExpr
    /// (e.g. `(realize ...)`, `(handle-effect ... body)`, `(lit ...)` whose
    /// child is the canonical node), the parent's `span_id` appends here so
    /// the audit invariant ("every input span appears as `span_id` or in
    /// `merged_spans` on at least one node") still holds.
    pub fn append_merged_span(&mut self, span: Option<&str>) {
        let Some(span) = span else {
            return;
        };
        if self.span_id.as_deref() == Some(span) {
            return;
        }
        if self.merged_spans.iter().any(|s| s == span) {
            return;
        }
        self.merged_spans.push(span.to_owned());
        self.merged_spans.sort();
    }
}

#[derive(Debug, Clone)]
pub enum HostExprKind<T = HostTypeTerm> {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    List(Vec<HostExpr<T>>, T),
    Tuple(Vec<HostExpr<T>>, T),
    Var(String, T),
    Call {
        function: String,
        args: Vec<HostExpr<T>>,
        arg_tys: Vec<T>,
        ty: T,
    },
    Builtin {
        name: String,
        args: Vec<HostExpr<T>>,
        ty: T,
    },
    AdtConstruct {
        ctor: String,
        fields: Vec<HostExpr<T>>,
        ty: T,
    },
    AdtFieldAccess {
        base: Box<HostExpr<T>>,
        field_index: usize,
        ty: T,
    },
    If {
        cond: Box<HostExpr<T>>,
        then_expr: Box<HostExpr<T>>,
        else_expr: Box<HostExpr<T>>,
        ty: T,
    },
    MatchOption {
        scrutinee: Box<HostExpr<T>>,
        bind_name: String,
        some_expr: Box<HostExpr<T>>,
        none_expr: Box<HostExpr<T>>,
        ty: T,
    },
    MatchAdt {
        scrutinee: Box<HostExpr<T>>,
        arms: Vec<HostMatchArm<T>>,
        default_expr: Option<Box<HostExpr<T>>>,
        ty: T,
    },
    Let {
        bindings: Vec<HostBinding<T>>,
        body: Box<HostExpr<T>>,
        ty: T,
    },
    Map {
        callback: HostCallback<T>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    Filter {
        callback: HostCallback<T>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    Fold {
        callback: HostCallback<T>,
        init: Box<HostExpr<T>>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    Scan {
        callback: HostCallback<T>,
        init: Box<HostExpr<T>>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    Partition {
        callback: HostCallback<T>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    FlatMap {
        callback: HostCallback<T>,
        list: Box<HostExpr<T>>,
        ty: T,
    },
    WithSeed {
        seed: Box<HostExpr<T>>,
        body: Box<HostExpr<T>>,
        ty: T,
    },
    TensorCall {
        helper: usize,
        args: Vec<HostExpr<T>>,
        ty: T,
    },
    Unit,
}

pub type ConcreteHostCallback = HostCallback<ConcreteHostType>;
pub type ConcreteHostMatchArm = HostMatchArm<ConcreteHostType>;
pub type ConcreteHostPatternBinding = HostPatternBinding<ConcreteHostType>;
pub type ConcreteHostAdtField = HostAdtField<ConcreteHostType>;
pub type ConcreteHostCallbackKind = HostCallbackKind<ConcreteHostType>;
pub type ConcreteHostExpr = HostExpr<ConcreteHostType>;
pub type ConcreteHostExprKind = HostExprKind<ConcreteHostType>;

fn resolve_host_program(
    program: HostProgram,
) -> Result<ConcreteHostProgram, crate::HostTypeResolutionError> {
    Ok(ConcreteHostProgram {
        globals: program
            .globals
            .into_iter()
            .map(resolve_host_binding)
            .collect::<Result<Vec<_>, _>>()?,
        global_tensor_helpers: program.global_tensor_helpers,
        functions: program
            .functions
            .into_iter()
            .map(resolve_host_function)
            .collect::<Result<Vec<_>, _>>()?,
        summary_rejections: program.summary_rejections,
    })
}

fn resolve_host_binding(
    binding: HostBinding,
) -> Result<ConcreteHostBinding, crate::HostTypeResolutionError> {
    Ok(ConcreteHostBinding {
        name: binding.name,
        display_name: binding.display_name,
        ty: binding.ty.into_concrete()?,
        value: resolve_host_expr(binding.value)?,
    })
}

fn resolve_host_function(
    function: HostFunction,
) -> Result<ConcreteHostFunction, crate::HostTypeResolutionError> {
    Ok(ConcreteHostFunction {
        name: function.name,
        params: function
            .params
            .into_iter()
            .map(resolve_host_param)
            .collect::<Result<Vec<_>, _>>()?,
        ret_ty: function.ret_ty.into_concrete()?,
        body: resolve_host_expr(function.body)?,
        tensor_helpers: function.tensor_helpers,
        specialization: function.specialization,
        summary_rejections: function.summary_rejections,
    })
}

fn resolve_host_param(
    param: HostParam,
) -> Result<ConcreteHostParam, crate::HostTypeResolutionError> {
    Ok(ConcreteHostParam {
        name: param.name,
        ty: param.ty.into_concrete()?,
    })
}

fn resolve_host_callback(
    callback: HostCallback,
) -> Result<ConcreteHostCallback, crate::HostTypeResolutionError> {
    let kind = match callback.kind {
        HostCallbackKind::Named { function, params } => ConcreteHostCallbackKind::Named {
            function,
            params: params
                .into_iter()
                .map(resolve_host_param)
                .collect::<Result<Vec<_>, _>>()?,
        },
        HostCallbackKind::Inline { params, body } => ConcreteHostCallbackKind::Inline {
            params: params
                .into_iter()
                .map(resolve_host_param)
                .collect::<Result<Vec<_>, _>>()?,
            body: Box::new(resolve_host_expr(*body)?),
        },
    };
    Ok(ConcreteHostCallback {
        kind,
        ret_ty: callback.ret_ty.into_concrete()?,
    })
}

fn resolve_host_expr(expr: HostExpr) -> Result<ConcreteHostExpr, crate::HostTypeResolutionError> {
    let kind = match expr.kind {
        HostExprKind::Int(value) => ConcreteHostExprKind::Int(value),
        HostExprKind::Float(value) => ConcreteHostExprKind::Float(value),
        HostExprKind::Bool(value) => ConcreteHostExprKind::Bool(value),
        HostExprKind::String(value) => ConcreteHostExprKind::String(value),
        HostExprKind::List(items, ty) => ConcreteHostExprKind::List(
            items
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty.into_concrete()?,
        ),
        HostExprKind::Tuple(items, ty) => ConcreteHostExprKind::Tuple(
            items
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty.into_concrete()?,
        ),
        HostExprKind::Var(name, ty) => ConcreteHostExprKind::Var(name, ty.into_concrete()?),
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        } => ConcreteHostExprKind::Call {
            function,
            args: args
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            arg_tys: arg_tys
                .into_iter()
                .map(HostTypeTerm::into_concrete)
                .collect::<Result<Vec<_>, _>>()?,
            ty: ty.into_concrete()?,
        },
        HostExprKind::Builtin { name, args, ty } => ConcreteHostExprKind::Builtin {
            name,
            args: args
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty: ty.into_concrete()?,
        },
        HostExprKind::AdtConstruct { ctor, fields, ty } => ConcreteHostExprKind::AdtConstruct {
            ctor,
            fields: fields
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty: ty.into_concrete()?,
        },
        HostExprKind::AdtFieldAccess {
            base,
            field_index,
            ty,
        } => ConcreteHostExprKind::AdtFieldAccess {
            base: Box::new(resolve_host_expr(*base)?),
            field_index,
            ty: ty.into_concrete()?,
        },
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ty,
        } => ConcreteHostExprKind::If {
            cond: Box::new(resolve_host_expr(*cond)?),
            then_expr: Box::new(resolve_host_expr(*then_expr)?),
            else_expr: Box::new(resolve_host_expr(*else_expr)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        } => ConcreteHostExprKind::MatchOption {
            scrutinee: Box::new(resolve_host_expr(*scrutinee)?),
            bind_name,
            some_expr: Box::new(resolve_host_expr(*some_expr)?),
            none_expr: Box::new(resolve_host_expr(*none_expr)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        } => ConcreteHostExprKind::MatchAdt {
            scrutinee: Box::new(resolve_host_expr(*scrutinee)?),
            arms: arms
                .into_iter()
                .map(|arm| {
                    Ok(ConcreteHostMatchArm {
                        ctor: arm.ctor,
                        bindings: arm
                            .bindings
                            .into_iter()
                            .map(|binding| {
                                Ok(ConcreteHostPatternBinding {
                                    name: binding.name,
                                    ty: binding.ty.into_concrete()?,
                                    field_index: binding.field_index,
                                })
                            })
                            .collect::<Result<Vec<_>, crate::HostTypeResolutionError>>()?,
                        expr: resolve_host_expr(arm.expr)?,
                    })
                })
                .collect::<Result<Vec<_>, crate::HostTypeResolutionError>>()?,
            default_expr: default_expr
                .map(|expr| resolve_host_expr(*expr).map(Box::new))
                .transpose()?,
            ty: ty.into_concrete()?,
        },
        HostExprKind::Let { bindings, body, ty } => ConcreteHostExprKind::Let {
            bindings: bindings
                .into_iter()
                .map(resolve_host_binding)
                .collect::<Result<Vec<_>, _>>()?,
            body: Box::new(resolve_host_expr(*body)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::Map { callback, list, ty } => ConcreteHostExprKind::Map {
            callback: resolve_host_callback(callback)?,
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::Filter { callback, list, ty } => ConcreteHostExprKind::Filter {
            callback: resolve_host_callback(callback)?,
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        } => ConcreteHostExprKind::Fold {
            callback: resolve_host_callback(callback)?,
            init: Box::new(resolve_host_expr(*init)?),
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        } => ConcreteHostExprKind::Scan {
            callback: resolve_host_callback(callback)?,
            init: Box::new(resolve_host_expr(*init)?),
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::Partition { callback, list, ty } => ConcreteHostExprKind::Partition {
            callback: resolve_host_callback(callback)?,
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::FlatMap { callback, list, ty } => ConcreteHostExprKind::FlatMap {
            callback: resolve_host_callback(callback)?,
            list: Box::new(resolve_host_expr(*list)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::WithSeed { seed, body, ty } => ConcreteHostExprKind::WithSeed {
            seed: Box::new(resolve_host_expr(*seed)?),
            body: Box::new(resolve_host_expr(*body)?),
            ty: ty.into_concrete()?,
        },
        HostExprKind::TensorCall { helper, args, ty } => ConcreteHostExprKind::TensorCall {
            helper,
            args: args
                .into_iter()
                .map(resolve_host_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty: ty.into_concrete()?,
        },
        HostExprKind::Unit => ConcreteHostExprKind::Unit,
    };
    Ok(ConcreteHostExpr {
        kind,
        span_id: expr.span_id,
        merged_spans: expr.merged_spans,
    })
}

pub fn lower_compiled_program(program: &CheckedProgram) -> CompiledProgram {
    try_lower_compiled_program(program).unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

/// Fallible variant of [`lower_compiled_program`] that catches lowering
/// panics (e.g. WS-A5 RT-3a F2: an unresolved tensor precision tvar
/// reaching the `Type::Tensor` -> `HostTypeTerm::Tensor` boundary, which
/// `try_extract_tensor_type` panics on per spec/04-type-system.md
/// \u{00a7}5.8.1) and returns them as `LowerDiagnostic` so the build
/// CLI can surface a clean user-facing error rather than a thread
/// panic.
pub fn try_lower_compiled_program(
    program: &CheckedProgram,
) -> Result<CompiledProgram, crate::lower::LowerDiagnostic> {
    let lowered_names = top_level_lowering_map(program.exprs(), program.type_env());
    // Issue #197: a *fatal* diagnostic from IR lowering (e.g. the AD
    // pass refused to differentiate a non-differentiable op) must
    // propagate to the user. Falling through to the host path here
    // emits a call to an undefined symbol (the unlowered grad
    // function) which compiles cleanly via the `chelis build` step
    // and then fails opaquely at gcc-link time. Non-fatal
    // diagnostics (the historical un-representable cases) keep the
    // original fallback semantics so host-only programs still build.
    let dag = match crate::lower::try_lower_program(program) {
        Ok(dag) => Some(dag),
        Err(diagnostic) if diagnostic.fatal => return Err(diagnostic),
        Err(_) => None,
    };
    let host_terms =
        crate::lower::catch_lowering_external(|| lower_host_program(program, &lowered_names))?;
    let host = resolve_host_program(host_terms).map_err(|error| {
        crate::lower::LowerDiagnostic::new(
            format!(
                "host type did not resolve before the code-generation boundary: {error} \
                 ([05-UNS-1]; chelis#730)"
            ),
            None,
            None,
        )
        .fatal()
    })?;

    Ok(CompiledProgram {
        dag: dag.filter(|dag| !dag.roots().is_empty()),
        host: if host.globals.is_empty() && host.functions.is_empty() {
            None
        } else {
            Some(host)
        },
    })
}

pub fn host_program_requires_host_backend(program: &ConcreteHostProgram) -> bool {
    if !program.globals.is_empty() {
        return true;
    }

    let tensor_only_functions = program
        .functions
        .iter()
        .filter(|function| {
            matches!(function.ret_ty, ConcreteHostType::Tensor(_))
                && function
                    .params
                    .iter()
                    .all(|param| matches!(param.ty, ConcreteHostType::Tensor(_)))
        })
        .map(|function| function.name.clone())
        .collect::<HashSet<_>>();

    program.functions.iter().any(|function| {
        !tensor_only_functions.contains(&function.name)
            || !host_expr_stays_on_tensor_path(&function.body, &tensor_only_functions)
    })
}

fn host_expr_stays_on_tensor_path(
    expr: &ConcreteHostExpr,
    tensor_only_functions: &HashSet<String>,
) -> bool {
    match &expr.kind {
        ConcreteHostExprKind::Var(_, ConcreteHostType::Tensor(_)) => true,
        ConcreteHostExprKind::TensorCall { .. } => true,
        ConcreteHostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        } => {
            matches!(ty, ConcreteHostType::Tensor(_))
                && tensor_only_functions.contains(function)
                && arg_tys
                    .iter()
                    .all(|ty| matches!(ty, ConcreteHostType::Tensor(_)))
                && args
                    .iter()
                    .all(|arg| host_expr_stays_on_tensor_path(arg, tensor_only_functions))
        }
        ConcreteHostExprKind::Let { bindings, body, ty } => {
            matches!(ty, ConcreteHostType::Tensor(_))
                && bindings.iter().all(|binding| {
                    matches!(binding.ty, ConcreteHostType::Tensor(_))
                        && host_expr_stays_on_tensor_path(&binding.value, tensor_only_functions)
                })
                && host_expr_stays_on_tensor_path(body, tensor_only_functions)
        }
        _ => false,
    }
}

pub fn preferred_tensor_entry_name(program: &ConcreteHostProgram) -> Option<&str> {
    fn tensor_signature(function: &ConcreteHostFunction) -> bool {
        matches!(function.ret_ty, ConcreteHostType::Tensor(_))
            && function
                .params
                .iter()
                .all(|param| matches!(param.ty, ConcreteHostType::Tensor(_)))
    }

    if let Some(function) = program
        .functions
        .iter()
        .find(|function| function.name == "main" && tensor_signature(function))
    {
        return Some(function.name.as_str());
    }

    program
        .functions
        .iter()
        .rev()
        .find(|function| tensor_signature(function))
        .map(|function| function.name.as_str())
}

pub fn lower_named_tensor_entry_dag(program: &CheckedProgram, name: &str) -> Option<crate::Dag> {
    let defs = collect_program_defs(program.exprs());
    let body = lookup_program_def(&defs, name)?.clone();
    let Expr::List(list, _) = &body else {
        return None;
    };
    if tag(list) != Some("fn") {
        return None;
    }

    let kids = children(list);
    let params_list = kids.first().and_then(as_list)?;
    if tag(params_list) != Some("params") {
        return None;
    }

    let declared_param_tys = lookup_declared_type_expr(program, name)
        .and_then(parse_fn_type_expr)
        .map(|(params, _)| params)
        .unwrap_or_default();
    let mut scope = HashMap::new();
    for (index, param) in children(params_list).iter().enumerate() {
        let pname = param_name(param)?;
        let pty = param_host_type(param)
            .or_else(|| declared_param_tys.get(index).cloned())
            .filter(|ty| !ty.is_unresolved())?;
        let tensor_ty = tensor_type_from_host_input(&pty)?;
        scope.insert(pname, tensor_ty);
    }

    let body_expr = kids.get(1)?;
    // Issue #197: a fatal lowering diagnostic (AD-rejection) must
    // propagate as a panic so the outer `catch_lowering_external`
    // surfaces it to the user. Silently absorbing it with `.ok()`
    // would let the host fallback emit an undefined-symbol call to
    // the grad function.
    match crate::lower::try_lower_subexpr_program(
        body_expr,
        scope,
        program.type_env().clone(),
        defs,
    ) {
        Ok(dag) => Some(dag),
        Err(diagnostic) if diagnostic.fatal => {
            crate::lower::raise_fatal_lowering_diagnostic(diagnostic)
        }
        Err(_) => None,
    }
}

fn lower_host_program(
    program: &CheckedProgram,
    lowered_names: &HashMap<String, bool>,
) -> HostProgram {
    let mut host = HostProgram::default();
    let mut global_scope = HashMap::new();
    // Count pure-tensor `fn`-body top-level defs in the program. When there
    // is more than one, the legacy DAG-only path would collapse them into a
    // single file-named entry point that drops all but one def's parameters
    // (Nautilus Bug 3c). In that case we emit a host wrapper per def so each
    // gets its own C symbol.
    let lowered_fn_def_count = top_level_items(program.exprs())
        .iter()
        .filter(|expr| {
            let Expr::List(list, _) = expr else {
                return false;
            };
            if tag(list) != Some("def") {
                return false;
            }
            let kids = children(list);
            let Some(def_name) = kids.first().and_then(symbol_name) else {
                return false;
            };
            let is_fn_body = matches!(kids.get(1), Some(Expr::List(body_list, _)) if tag(body_list) == Some("fn"));
            is_fn_body && lowered_names.get(def_name).copied().unwrap_or(false)
        })
        .count();
    // Issue #378: names referenced by any `fn`-body top-level def. A
    // captured top-level *scalar* (or other value) binding is classified
    // `lowered` by `def_is_lowered` (a `(lit ...)` body materializes into
    // the DAG `main()`), so `skip_for_lowered` would drop it from
    // `host.globals` entirely — but a host-lane function that captures it
    // then emits `__arg = c;` against an undeclared `c`. The #376 hoist
    // already serves *tensor* captures because tensor bindings reach
    // `host.globals`; this set lets a captured value binding stay in
    // `host.globals` too (the C emitter's `captured_global_names` only
    // declares globals a function actually references, so a non-captured
    // binding still costs nothing in the pure-DAG case).
    let mut names_captured_by_fn_defs: HashSet<String> = HashSet::new();
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("def") {
            continue;
        }
        let kids = children(list);
        let Some(body) = kids.get(1) else {
            continue;
        };
        if matches!(body, Expr::List(body_list, _) if tag(body_list) == Some("fn")) {
            collect_deep_var_names(body, &mut names_captured_by_fn_defs);
        }
    }
    // Issue #750: does this program emit a host `main()`? A non-`fn`
    // top-level binding reaches `host.globals` today iff `skip_for_lowered`
    // does NOT drop it, i.e. it is non-lowered (needs host runtime, e.g. a
    // `print`-effecting `shown = print(x)`) or it is a #378 captured value
    // binding. `emit_main` runs exactly when `host.globals` is non-empty
    // (`host_emit.rs`), so the presence of any such binding is the precise,
    // rescue-independent predicate for "this program emits a host `main`".
    // The #750 rescue below is gated on this so it only ever RE-ATTACHES a
    // labeled root to a `main` that already exists — it never flips a
    // pure-DAG program (which emits a callable kernel with `outputs[]`, no
    // `main`) onto the host lane.
    let program_emits_host_main = top_level_items(program.exprs()).iter().any(|expr| {
        let Expr::List(list, _) = expr else {
            return false;
        };
        if tag(list) != Some("def") {
            return false;
        }
        let kids = children(list);
        let Some(binding_name) = kids.first().and_then(symbol_name) else {
            return false;
        };
        let is_fn_body =
            matches!(kids.get(1), Some(Expr::List(body_list, _)) if tag(body_list) == Some("fn"));
        !is_fn_body
            && (!lowered_names.get(binding_name).copied().unwrap_or(false)
                || names_captured_by_fn_defs.contains(binding_name))
    });
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("def") {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        let ty_expr = lookup_declared_type_expr(program, name);
        // WS-A8: skip polymorphic-precision sigs at host emission time.
        // A `t-fn` whose tensor types carry `(t-var {} _)` precision
        // slots has no concrete monomorphization on its own; reaching
        // `parse_host_type` for its parameters trips the F2 backend
        // tripwire panic per spec/04-type-system.md §5.8.1. Such a def
        // is reachable from concrete client code via call-site inlining
        // (the inliner threads the call site's concrete precision into
        // the body); the standalone host symbol is intentionally
        // omitted because no caller can use it without supplying the
        // monomorphization binding the inliner provides.
        if ty_expr.is_some_and(crate::lower::type_expr_has_precision_var) {
            continue;
        }
        // Tier-2 rank polymorphism (spec/design/rank_polymorphism.md): skip
        // rank-polymorphic sigs at host emission time, the exact analogue of
        // the precision-var skip above. A `t-fn` whose tensor types carry a
        // sole `(d-rank {} r)` rank slot has no standalone monomorphization;
        // reaching `parse_host_type` for its parameters would trip the
        // rank-poly lowering tripwire (a surviving `Dim::Rank` is a
        // monomorphization bug, not a backend input). Such a def is reachable
        // from concrete client code via call-site inlining (the inliner
        // threads the call site's concrete shape into the body — see
        // `tensor_rank_substitutions` in `lower.rs`); the standalone host
        // symbol is intentionally omitted because no caller can use it
        // without supplying the monomorphization binding the inliner provides.
        if ty_expr.is_some_and(crate::lower::type_expr_has_rank_var) {
            continue;
        }
        // Pure-tensor top-level function defs are normally lowered to the
        // DAG. But when the program also has host-lane bindings (i.e. some
        // def is NOT DAG-lowerable), downstream host-lane callers still
        // need a real C function symbol for the wrapper. In that case,
        // emit a HostFunction wrapper alongside the DAG lowering.
        //
        // A single pure-tensor function def can still use the legacy
        // DAG-only path (the file-named entry point wraps it 1:1 with the
        // correct signature). But when there are multiple pure-tensor
        // function defs in the same program, the DAG path would collapse
        // them into a single file-named entry that silently drops all but
        // one def's parameters and outputs (Nautilus Bug 3c). Emit a host
        // wrapper per def in that case so each gets its own C symbol.
        let is_fn_body = matches!(body, Expr::List(list, _) if tag(list) == Some("fn"));
        let has_any_host_lane_def = lowered_names.values().any(|lowered| !*lowered);
        let has_callable_params = lookup_declared_fn_type(program, name)
            .is_some_and(|(params, _)| params.iter().any(|ty| matches!(ty, HostTypeTerm::Fn(..))));
        // Non-F32/Bool tensor precisions (e.g. int32, int64) aren't
        // representable in the Phase 0f DAG-only codegen path — it still
        // hard-asserts f32/bool. Force a host-lane wrapper for any fn whose
        // signature carries such a tensor so the program stays on the
        // host-lane code path instead of panicking in DAG emit.
        let has_non_dag_tensor =
            lookup_declared_fn_type(program, name).is_some_and(|(params, ret)| {
                fn ty_has_non_dag_tensor(ty: &HostTypeTerm) -> bool {
                    match ty {
                        HostTypeTerm::Tensor(tensor) => !matches!(
                            tensor.precision,
                            chelis_types::types::Prim::F32 | chelis_types::types::Prim::Bool
                        ),
                        HostTypeTerm::Option(inner) | HostTypeTerm::List(inner) => {
                            ty_has_non_dag_tensor(inner)
                        }
                        HostTypeTerm::Tuple(items) => items.iter().any(ty_has_non_dag_tensor),
                        HostTypeTerm::Dict(k, v) => {
                            ty_has_non_dag_tensor(k) || ty_has_non_dag_tensor(v)
                        }
                        HostTypeTerm::Fn(params, ret) => {
                            params.iter().any(ty_has_non_dag_tensor) || ty_has_non_dag_tensor(ret)
                        }
                        _ => false,
                    }
                }
                params.iter().any(ty_has_non_dag_tensor) || ty_has_non_dag_tensor(&ret)
            });
        // `has_callable_params` blocks the wrapper for fn-taking-fn signatures
        // because the host emitter doesn't lower higher-order wrappers
        // cleanly in every shape (grad specialization etc.). But if the
        // signature ALSO contains a non-F32/Bool tensor, the DAG-only
        // fallback panics — so in that combination still force the wrapper.
        //
        // Bucket 4d: a higher-order signature whose non-callable params and
        // return type are *scalars* (e.g. `(model: f32 -> f32, x: f32) -> f32`)
        // can never be DAG-lowered (the DAG-only path is tensor-only) and
        // the host emitter handles `model(x)` cleanly because every value
        // is a plain C scalar. Force the host wrapper for those signatures
        // so they actually get a definition emitted -- the previous logic
        // was silently dropping them, leaving `gcc` to fail with
        // `implicit declaration of function 'apply'` on the caller side.
        //
        // Known limitation: when a caller references a callable-param fn by
        // name with a tensor-typed shape (e.g. `(model: tensor[n, f32] ->
        // f32, ...)`) and the body doesn't lower cleanly through the host
        // wrapper, the def is still dropped from emission. Tracked as a
        // residual HOF emission issue (red-team A20.1).
        let scalar_only_callable_signature = has_callable_params
            && lookup_declared_fn_type(program, name).is_some_and(|(params, ret)| {
                fn ty_is_scalar_or_callable_scalar(ty: &HostTypeTerm) -> bool {
                    match ty {
                        HostTypeTerm::Tensor(_) => false,
                        HostTypeTerm::Fn(params, ret) => {
                            params.iter().all(ty_is_scalar_or_callable_scalar)
                                && ty_is_scalar_or_callable_scalar(ret)
                        }
                        HostTypeTerm::Option(inner) | HostTypeTerm::List(inner) => {
                            ty_is_scalar_or_callable_scalar(inner)
                        }
                        HostTypeTerm::Tuple(items) => {
                            items.iter().all(ty_is_scalar_or_callable_scalar)
                        }
                        HostTypeTerm::Dict(k, v) => {
                            ty_is_scalar_or_callable_scalar(k) && ty_is_scalar_or_callable_scalar(v)
                        }
                        // Primitive (f32/int32/bool/...), Unit -- scalar OK.
                        _ => true,
                    }
                }
                params.iter().all(ty_is_scalar_or_callable_scalar)
                    && ty_is_scalar_or_callable_scalar(&ret)
            });
        let needs_host_wrapper = is_fn_body
            && (has_non_dag_tensor
                || scalar_only_callable_signature
                || (!has_callable_params && (has_any_host_lane_def || lowered_fn_def_count > 1)));
        // Issue #378: a non-`fn` value binding (a `(def name (lit ...))`)
        // that a host-lane function captures must reach `host.globals` so
        // the emitted C declares it; otherwise the function body references
        // an undeclared identifier. Do not skip it even though the DAG lane
        // also claims it (the DAG lane inlines its own copy for tensor
        // roots; the global is emitted only when a function references it).
        let captured_value_binding = !is_fn_body && names_captured_by_fn_defs.contains(name);
        // Issue #750: a DAG-lowered non-`fn` value binding (e.g.
        // `troot = mk()`, a def-call-valued root) that a program's host
        // `main` should print is otherwise dropped by `skip_for_lowered`,
        // because labeled roots are emitted ONLY from `host.globals` and
        // there is no DAG-root -> labeled-print bridge. Rescue it into
        // `host.globals` exactly like the #378 captured-value carve-out so
        // the CLI display-name pass stamps it and `emit_main` renders it,
        // byte-identical to a direct-construction root of the same value.
        // Gated on `program_emits_host_main` so the rescue only re-attaches
        // a root to a `main` the program ALREADY emits — a pure-DAG program
        // keeps emitting a kernel with `outputs[]` and no `main`.
        let display_root_binding = !is_fn_body && program_emits_host_main;
        let skip_for_lowered = lowered_names.get(name).copied().unwrap_or(false)
            && !needs_host_wrapper
            && !captured_value_binding
            && !display_root_binding;
        if skip_for_lowered {
            continue;
        }
        // The wrapper emitter doesn't know every pattern the DAG-path
        // specializer does (e.g. `grad(local_fn)(theta)`). When the host
        // lowering would degrade to a fallback `Builtin { name: "call" }`,
        // emitting the wrapper produces broken C (`__result = call(...)`).
        // In that case prefer the DAG path if it's available (lowered_names
        // says so); otherwise we have no good lowering and must drop the
        // def — at least the caller will get `implicit declaration` rather
        // than `call(…)` undefined-symbol.
        if let Some(mut function) = lower_host_function(name, body, ty_expr, program) {
            // N→1 lowering collapse per `spec/design/chelis_span_survival.md`
            // §2.3 host-side table, rule "Lowering. Top-level def collapses
            // to fn body": when a `(def {span: a} name (fn ... body))` lowers
            // to a HostFunction whose `body` is the lowered fn body, the
            // def's `span_id` appends to the body node's `merged_spans` so
            // the def's source region surfaces in the audit chain. (The fn
            // node's own span, if any, is already on the body via the
            // body-collapse rule applied by `lower_host_expr`.)
            function.body.append_merged_span(expr.span_id());
            global_scope.insert(
                name.to_string(),
                HostTypeTerm::Fn(
                    function
                        .params
                        .iter()
                        .map(|param| param.ty.clone())
                        .collect(),
                    Box::new(function.ret_ty.clone()),
                ),
            );
            host.functions.push(function);
        } else {
            // Inline any local callable bindings in the global binding's
            // body so `let g = grad(f); g(x)` rewrites to `(grad(f))(x)`
            // before host lowering — same rationale as in
            // `lower_host_function`.
            let inlined_body = inline_local_callable_lets(body);
            let mut value = lower_host_expr(
                &inlined_body,
                program,
                &global_scope,
                &mut host.global_tensor_helpers,
            );
            // N→1 lowering collapse per `spec/design/chelis_span_survival.md`
            // §2.3 host-side table, rule "Lowering. Top-level def collapses
            // to body": when a `(def {span: a} name body)` lowers to a
            // HostBinding whose `value` is the body's HostExpr, the def's
            // own `span_id` (sourced from the def's `meta["span"]`) appends
            // to the value node's `merged_spans` so the def's source region
            // doesn't drop out of the audit chain.
            value.append_merged_span(expr.span_id());
            let inferred_ty = host_expr_type(&value);
            // RT-4 F1: respect the surface-level type annotation on a
            // global binding. Without this the type-checker-validated
            // declaration `x: tensor[3, f64] = [1.0, 2.0, 3.0]` was
            // silently lowered with the inferred f32 type because the
            // `to_tensor` builtin's return type defaults to f32 for any
            // float-tagged list. The downstream host emitter uses this
            // type to dispatch the typed runtime call so the storage
            // matches the declared dtype.
            let declared_ty = ty_expr
                .map(|ty| decode_host_type_or_raise(ty, &HashMap::new()))
                .filter(|ty| !ty.is_unresolved());
            let ty = declared_ty.clone().unwrap_or_else(|| inferred_ty.clone());
            // When the declared type sharpens the inferred type (e.g.
            // declared f64, inferred f32), retag the outermost value
            // type so downstream host emit sees the right precision
            // for the typed to_tensor runtime call. This is a
            // structural retag; the type checker has already validated
            // that the literal assignment is sound.
            if let Some(declared) = declared_ty.as_ref()
                && declared != &inferred_ty
            {
                value = force_host_expr_type(value, declared.clone());
            }
            host.globals.push(HostBinding {
                name: name.to_string(),
                display_name: None,
                ty: ty.clone(),
                value: value.clone(),
            });
            global_scope.insert(name.to_string(), ty);
        }
    }
    loop {
        let mut changed = false;
        changed |= refine_host_function_signatures(&mut host.functions);
        changed |= refine_host_globals(&mut host.globals, &host.functions);
        changed |= propagate_named_callback_signatures(&mut host.functions, &host.globals);
        if !changed {
            break;
        }
    }
    // The checker has already proved every binding/function/callback
    // context.  Materialize those expected types into the host expression
    // tree before resolving terms: the backend must receive one coherent
    // typed program, not re-run contextual refinement or silently choose
    // between a declared type and a coarse local inference.
    conform_host_program_types(&mut host);
    derive_host_function_specializations(&mut host.functions);
    // Also collect rejections from global tensor helpers (top-level
    // expressions like `result = gather(...)` lower into
    // `host.global_tensor_helpers`, not into a function's
    // tensor_helpers). The owning "function" for diagnostic
    // attribution is the synthetic global name; today we report it
    // under the helper's own name since global tensor helpers don't
    // share a binding name directly.
    collect_program_summary_rejections(&mut host);
    host
}

/// Aggregate per-function rejections into `HostProgram::summary_rejections`
/// and also fold in any rejections attached to `global_tensor_helpers`
/// (rejections detected on top-level tensor expression bindings).
fn collect_program_summary_rejections(host: &mut HostProgram) {
    host.summary_rejections.clear();
    for function in &host.functions {
        for rejection in &function.summary_rejections {
            host.summary_rejections.push(rejection.clone());
        }
    }
    // Per-global rejections: the global tensor helpers carry partial
    // rejections from `finish_tensor_helper_call`. The HelperPath is
    // synthesized from the helper's own name (e.g.
    // `__global_tensor_helper_0`) when there is no enclosing
    // function. This is sufficient for diagnostic emission today;
    // future work can thread the binding name through.
    for helper in &host.global_tensor_helpers {
        if let Some(partial) = &helper.summary_rejection {
            host.summary_rejections.push(SummaryRejection {
                rejection_class: partial.rejection_class.clone(),
                helper_path: HelperPath::local(&helper.name),
                callsite_span: None,
                helper_body_span: partial.helper_body_span.clone(),
                detail: partial.detail.clone(),
            });
        }
    }
}

/// Public accessor for the structured summary rejections collected
/// during host lowering. CLI diagnostic reporters and downstream
/// tests pattern-match on the returned slice's enum variants and
/// struct fields. The order is deterministic: per-function rejections
/// appear in function-declaration order, followed by per-global
/// rejections in helper-declaration order.
pub fn host_program_summary_rejections<T>(program: &HostProgram<T>) -> &[SummaryRejection] {
    &program.summary_rejections
}

/// Walk a HostExpr and report whether any node is the generic-fallback
/// `Builtin { name: "call", ... }` that `lower_app_host_expr` emits when
/// it doesn't recognize the callee. A wrapper containing this node would
/// emit broken C (`__result = call(...);`) downstream — preferring the
/// DAG path's inline specialization is safer than emitting that wrapper.
/// Scan a host program for any function whose body still contains the
/// `Builtin { name: "call" }` fallback — the lowerer emits this when
/// it can't recognize the callee (typically `grad(f)(theta)` where the
/// callee is itself an application). Emitting such a wrapper produces
/// `__result = call(...)` C code that doesn't link. The CLI uses this
/// to surface a clean error instead of shipping broken C.
pub fn host_program_unresolved_call_sites<T>(program: &HostProgram<T>) -> Vec<String> {
    let mut out = Vec::new();
    for function in &program.functions {
        if host_body_has_fallback_call(&function.body) {
            out.push(function.name.clone());
        }
    }
    for binding in &program.globals {
        if host_body_has_fallback_call(&binding.value) {
            out.push(binding.name.clone());
        }
    }
    out
}

/// Returns `true` if any global binding or function body in `program`
/// applies the named builtin. Used by the build backends to reject
/// eval/test-only builtins (e.g. `process_run`, Hull Phase 0a) with a
/// clean diagnostic rather than the silent `/* unsupported builtin */ 0`
/// fallthrough in C codegen.
pub fn host_program_uses_builtin<T>(program: &HostProgram<T>, builtin: &str) -> bool {
    program
        .globals
        .iter()
        .any(|binding| host_body_uses_builtin(&binding.value, builtin))
        || program
            .functions
            .iter()
            .any(|function| host_body_uses_builtin(&function.body, builtin))
}

fn host_callback_uses_builtin<T>(callback: &HostCallback<T>, builtin: &str) -> bool {
    match &callback.kind {
        HostCallbackKind::Inline { body, .. } => host_body_uses_builtin(body, builtin),
        HostCallbackKind::Named { .. } => false,
    }
}

fn host_body_uses_builtin<T>(expr: &HostExpr<T>, builtin: &str) -> bool {
    match &expr.kind {
        HostExprKind::Builtin { name, args, .. } => {
            name == builtin || args.iter().any(|arg| host_body_uses_builtin(arg, builtin))
        }
        HostExprKind::Call { args, .. } => {
            args.iter().any(|arg| host_body_uses_builtin(arg, builtin))
        }
        HostExprKind::TensorCall { args, .. } => {
            args.iter().any(|arg| host_body_uses_builtin(arg, builtin))
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            fields.iter().any(|f| host_body_uses_builtin(f, builtin))
        }
        HostExprKind::Tuple(items, _) | HostExprKind::List(items, _) => {
            items.iter().any(|i| host_body_uses_builtin(i, builtin))
        }
        HostExprKind::AdtFieldAccess { base, .. } => host_body_uses_builtin(base, builtin),
        HostExprKind::Let { bindings, body, .. } => {
            bindings
                .iter()
                .any(|b| host_body_uses_builtin(&b.value, builtin))
                || host_body_uses_builtin(body, builtin)
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            host_body_uses_builtin(cond, builtin)
                || host_body_uses_builtin(then_expr, builtin)
                || host_body_uses_builtin(else_expr, builtin)
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            host_body_uses_builtin(scrutinee, builtin)
                || host_body_uses_builtin(some_expr, builtin)
                || host_body_uses_builtin(none_expr, builtin)
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            host_body_uses_builtin(scrutinee, builtin)
                || arms
                    .iter()
                    .any(|arm| host_body_uses_builtin(&arm.expr, builtin))
                || default_expr
                    .as_ref()
                    .is_some_and(|d| host_body_uses_builtin(d, builtin))
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            host_callback_uses_builtin(callback, builtin) || host_body_uses_builtin(list, builtin)
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            host_callback_uses_builtin(callback, builtin)
                || host_body_uses_builtin(init, builtin)
                || host_body_uses_builtin(list, builtin)
        }
        HostExprKind::WithSeed { seed, body, .. } => {
            host_body_uses_builtin(seed, builtin) || host_body_uses_builtin(body, builtin)
        }
        HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Var(_, _)
        | HostExprKind::Unit => false,
    }
}

fn derive_host_function_specializations(functions: &mut [HostFunction]) {
    let mut summaries = functions
        .iter()
        .filter_map(|function| {
            function
                .specialization
                .clone()
                .map(|summary| (function.name.clone(), summary))
        })
        .collect::<HashMap<_, _>>();

    loop {
        let mut changed = false;
        for function in functions.iter_mut() {
            if summaries.contains_key(&function.name) {
                continue;
            }
            let Some(summary) = derive_host_function_specialization(function, &summaries) else {
                continue;
            };
            summaries.insert(function.name.clone(), summary.clone());
            function.specialization = Some(summary);
            changed = true;
        }
        if !changed {
            break;
        }
    }

    // After spec derivation, collect structured summary rejections per
    // function. This pass owns:
    //
    //   * promoting partial `HelperSummaryRejection`s on each tensor
    //     helper into fully-formed `SummaryRejection`s attached to
    //     `HostFunction::summary_rejections` (and to
    //     `HostProgram::summary_rejections` via `lower_compiled_program`),
    //   * detecting the function-level `MultipleReturnPaths` case
    //     (helper body is `If`, not `TensorCall`) and emitting a
    //     rejection for it. This is the W3-B-enumerated category 2
    //     case; it is not visible to `try_summarize_sparse_helper`
    //     because the function body never reaches the tensor-helper
    //     summarization path.
    for function in functions.iter_mut() {
        function.summary_rejections.clear();
        collect_function_summary_rejections(function);
    }
}

/// Promote each tensor helper's `HelperSummaryRejection` into a
/// fully-formed `SummaryRejection` on this function. Also detects the
/// function-level `MultipleReturnPaths` case (function body lowered
/// to `If` rather than `TensorCall`, branching across multiple
/// sparse-op return paths).
fn collect_function_summary_rejections(function: &mut HostFunction) {
    // Track which helpers are referenced from the body so we can
    // attach the right callsite span.
    let callsite_span_for_helper = body_callsite_span_per_helper(&function.body);

    let helper_path = HelperPath::local(&function.name);

    for (idx, helper) in function.tensor_helpers.iter().enumerate() {
        if let Some(partial) = &helper.summary_rejection {
            let callsite_span = callsite_span_for_helper.get(&idx).cloned().unwrap_or(None);
            function.summary_rejections.push(SummaryRejection {
                rejection_class: partial.rejection_class.clone(),
                helper_path: helper_path.clone(),
                callsite_span,
                helper_body_span: partial.helper_body_span.clone(),
                detail: partial.detail.clone(),
            });
        }
    }

    // Category 2 (MultipleReturnPaths): function body is `If` whose
    // arms both name a sparse op (directly or via a wrapper call).
    // Detect by walking the body shape; if the body is exactly `If`
    // and at least one arm references a sparse-op helper, emit.
    if let Some(rejection) = detect_multiple_return_paths_rejection(function) {
        function.summary_rejections.push(rejection);
    }
}

/// Build a map from `helper_index -> callsite_span` by walking a
/// HostExpr for `TensorCall { helper, .. }` nodes. The first
/// observed callsite span wins; a `None` entry means the helper is
/// referenced but the call site has no span. Helpers never referenced
/// are absent from the map.
fn body_callsite_span_per_helper(expr: &HostExpr) -> HashMap<usize, Option<String>> {
    fn walk(expr: &HostExpr, out: &mut HashMap<usize, Option<String>>) {
        match &expr.kind {
            HostExprKind::TensorCall { helper, args, .. } => {
                out.entry(*helper).or_insert_with(|| expr.span_id.clone());
                for arg in args {
                    walk(arg, out);
                }
            }
            HostExprKind::Call { args, .. } => {
                for arg in args {
                    walk(arg, out);
                }
            }
            HostExprKind::Builtin { args, .. } => {
                for arg in args {
                    walk(arg, out);
                }
            }
            HostExprKind::If {
                cond,
                then_expr,
                else_expr,
                ..
            } => {
                walk(cond, out);
                walk(then_expr, out);
                walk(else_expr, out);
            }
            HostExprKind::Let { bindings, body, .. } => {
                for binding in bindings {
                    walk(&binding.value, out);
                }
                walk(body, out);
            }
            HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
                for item in items {
                    walk(item, out);
                }
            }
            HostExprKind::AdtConstruct { fields, .. } => {
                for field in fields {
                    walk(field, out);
                }
            }
            HostExprKind::AdtFieldAccess { base, .. } => {
                walk(base, out);
            }
            HostExprKind::MatchOption {
                scrutinee,
                some_expr,
                none_expr,
                ..
            } => {
                walk(scrutinee, out);
                walk(some_expr, out);
                walk(none_expr, out);
            }
            HostExprKind::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ..
            } => {
                walk(scrutinee, out);
                for arm in arms {
                    walk(&arm.expr, out);
                }
                if let Some(d) = default_expr {
                    walk(d, out);
                }
            }
            HostExprKind::Map { list, .. }
            | HostExprKind::Filter { list, .. }
            | HostExprKind::Partition { list, .. }
            | HostExprKind::FlatMap { list, .. } => walk(list, out),
            HostExprKind::Fold { init, list, .. } | HostExprKind::Scan { init, list, .. } => {
                walk(init, out);
                walk(list, out);
            }
            HostExprKind::WithSeed { seed, body, .. } => {
                walk(seed, out);
                walk(body, out);
            }
            _ => {}
        }
    }
    let mut out = HashMap::new();
    walk(expr, &mut out);
    out
}

/// Detect the `MultipleReturnPaths` rejection (W3-B category 2):
/// function body lowers to a HostExprKind::If with at least two
/// distinct return arms, and at least one arm names a sparse op
/// (directly via a TensorCall on a sparse-summarized helper, or
/// transitively via a Call to a function with a sparse
/// specialization). The callsite_span is the If's own span; the
/// helper_body_span is the deepest available arm span.
fn detect_multiple_return_paths_rejection(function: &HostFunction) -> Option<SummaryRejection> {
    let HostExprKind::If {
        then_expr,
        else_expr,
        ..
    } = &function.body.kind
    else {
        return None;
    };
    // Count branches: a chain of nested if/else collapses into one
    // detection but `branch_count` records the visible structural arm
    // count (then + else, recursively counting else-as-if).
    fn count_branches(expr: &HostExpr) -> usize {
        match &expr.kind {
            HostExprKind::If {
                then_expr,
                else_expr,
                ..
            } => count_branches(then_expr) + count_branches(else_expr),
            _ => 1,
        }
    }
    let branch_count = count_branches(&function.body);
    // Heuristic: at least one arm contains a TensorCall to a sparse-
    // summarized helper OR a Call to a known sparse function. We err
    // on the side of "yes, this is a sparse-helper rejection" if any
    // arm has a tensor call. The narrower check is a follow-up; for
    // W4-A the structural marker is what matters.
    fn arm_references_sparse_helper(expr: &HostExpr, function: &HostFunction) -> bool {
        match &expr.kind {
            HostExprKind::TensorCall { helper, .. } => function
                .tensor_helpers
                .get(*helper)
                .and_then(|h| h.specialization.as_ref())
                .is_some_and(|s| {
                    matches!(
                        s,
                        HostTensorSpecialization::SparseGather(_)
                            | HostTensorSpecialization::SparseScatterAdd(_)
                            | HostTensorSpecialization::SparseScatterReplace(_)
                    )
                }),
            HostExprKind::If {
                then_expr,
                else_expr,
                ..
            } => {
                arm_references_sparse_helper(then_expr, function)
                    || arm_references_sparse_helper(else_expr, function)
            }
            HostExprKind::Let { body, bindings, .. } => {
                arm_references_sparse_helper(body, function)
                    || bindings
                        .iter()
                        .any(|b| arm_references_sparse_helper(&b.value, function))
            }
            _ => false,
        }
    }
    if !arm_references_sparse_helper(then_expr, function)
        && !arm_references_sparse_helper(else_expr, function)
    {
        return None;
    }
    let callsite_span = function.body.span_id.clone();
    let helper_body_span = then_expr
        .span_id
        .clone()
        .or_else(|| else_expr.span_id.clone());
    Some(SummaryRejection {
        rejection_class: SummaryRejectionClass::MultipleReturnPaths,
        helper_path: HelperPath::local(&function.name),
        callsite_span,
        helper_body_span,
        detail: SummaryRejectionDetail::MultipleReturnPaths { branch_count },
    })
}

fn derive_host_function_specialization(
    function: &HostFunction,
    summaries: &HashMap<String, HostFunctionSpecialization>,
) -> Option<HostFunctionSpecialization> {
    match &function.body.kind {
        HostExprKind::TensorCall { helper, args, .. } => {
            match function
                .tensor_helpers
                .get(*helper)?
                .specialization
                .as_ref()?
            {
                HostTensorSpecialization::BlasMatmul(summary) => {
                    remap_blas_summary_to_params(summary, args, &function.params)
                        .map(HostFunctionSpecialization::BlasMatmul)
                }
                HostTensorSpecialization::SparseGather(summary) => {
                    remap_sparse_summary_to_params(summary, args, &function.params)
                        .map(HostFunctionSpecialization::SparseGather)
                }
                HostTensorSpecialization::SparseScatterAdd(summary) => {
                    remap_sparse_summary_to_params(summary, args, &function.params)
                        .map(HostFunctionSpecialization::SparseScatterAdd)
                }
                HostTensorSpecialization::SparseScatterReplace(summary) => {
                    remap_sparse_summary_to_params(summary, args, &function.params)
                        .map(HostFunctionSpecialization::SparseScatterReplace)
                }
            }
        }
        HostExprKind::Call {
            function: callee,
            args,
            ..
        } => match summaries.get(callee)? {
            HostFunctionSpecialization::BlasMatmul(summary) => {
                remap_blas_summary_to_params(summary, args, &function.params)
                    .map(HostFunctionSpecialization::BlasMatmul)
            }
            HostFunctionSpecialization::SparseGather(summary) => {
                remap_sparse_summary_to_params(summary, args, &function.params)
                    .map(HostFunctionSpecialization::SparseGather)
            }
            HostFunctionSpecialization::SparseScatterAdd(summary) => {
                remap_sparse_summary_to_params(summary, args, &function.params)
                    .map(HostFunctionSpecialization::SparseScatterAdd)
            }
            HostFunctionSpecialization::SparseScatterReplace(summary) => {
                remap_sparse_summary_to_params(summary, args, &function.params)
                    .map(HostFunctionSpecialization::SparseScatterReplace)
            }
        },
        _ => None,
    }
}

/// Remap a sparse-op summary's positional `input_indices` so they
/// reference the caller's parameters rather than the callee's
/// parameters. The shape of `args` must be a tuple of `Var` references
/// to caller parameters (the pure-pass-through wrapper case); any other
/// shape disqualifies the callsite.
fn remap_sparse_summary_to_params(
    summary: &HostSparseOpSummary,
    args: &[HostExpr],
    params: &[HostParam],
) -> Option<HostSparseOpSummary> {
    let arg_to_param = args
        .iter()
        .map(|arg| {
            let HostExprKind::Var(name, HostTypeTerm::Tensor(_)) = &arg.kind else {
                return None;
            };
            params.iter().position(|param| param.name == *name)
        })
        .collect::<Option<Vec<_>>>()?;
    if arg_to_param.len() != summary.input_tys.len() {
        return None;
    }
    let input_tys = params
        .iter()
        .map(|param| match &param.ty {
            HostTypeTerm::Tensor(ty) => Some(ty.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    // Verify that each summarized callee-input position still maps to a
    // caller parameter whose tensor type matches the callee's
    // requirement. This locks the contract that wrapper propagation
    // cannot widen or coerce the sparse-op operand types.
    let input_indices = summary
        .input_indices
        .iter()
        .map(|callee_idx| {
            let caller_idx = *arg_to_param.get(*callee_idx)?;
            let caller_ty = input_tys.get(caller_idx)?;
            let callee_ty = summary.input_tys.get(*callee_idx)?;
            if caller_ty != callee_ty {
                return None;
            }
            Some(caller_idx)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(HostSparseOpSummary {
        axis: summary.axis,
        input_indices,
        input_tys,
        output: summary.output.clone(),
    })
}

fn remap_blas_summary_to_params(
    summary: &HostBlasMatmulSummary,
    args: &[HostExpr],
    params: &[HostParam],
) -> Option<HostBlasMatmulSummary> {
    let arg_to_param = args
        .iter()
        .map(|arg| {
            let HostExprKind::Var(name, HostTypeTerm::Tensor(_)) = &arg.kind else {
                return None;
            };
            params.iter().position(|param| param.name == *name)
        })
        .collect::<Option<Vec<_>>>()?;
    let lhs_input = *arg_to_param.get(summary.lhs_input)?;
    let rhs_input = *arg_to_param.get(summary.rhs_input)?;
    let input_tys = params
        .iter()
        .map(|param| match &param.ty {
            HostTypeTerm::Tensor(ty) => Some(ty.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    if input_tys.get(lhs_input)? != summary.input_tys.get(summary.lhs_input)?
        || input_tys.get(rhs_input)? != summary.input_tys.get(summary.rhs_input)?
    {
        return None;
    }
    Some(HostBlasMatmulSummary {
        lhs_input,
        rhs_input,
        input_tys,
        output: summary.output.clone(),
        batch_dims: summary.batch_dims.clone(),
        m: summary.m.clone(),
        n: summary.n.clone(),
        k: summary.k.clone(),
    })
}

fn host_type_is_unresolved(ty: &HostTypeTerm) -> bool {
    ty.is_unresolved()
}

fn host_body_has_fallback_call<T>(expr: &HostExpr<T>) -> bool {
    match &expr.kind {
        HostExprKind::Builtin { name, args, .. } => {
            name == "call"
                || name.starts_with("__unresolved_")
                || args.iter().any(host_body_has_fallback_call)
        }
        HostExprKind::Call { function, args, .. } => {
            function == "call" || args.iter().any(host_body_has_fallback_call)
        }
        HostExprKind::Let { bindings, body, .. } => {
            bindings
                .iter()
                .any(|b| host_body_has_fallback_call(&b.value))
                || host_body_has_fallback_call(body)
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            host_body_has_fallback_call(cond)
                || host_body_has_fallback_call(then_expr)
                || host_body_has_fallback_call(else_expr)
        }
        HostExprKind::Tuple(items, _) | HostExprKind::List(items, _) => {
            items.iter().any(host_body_has_fallback_call)
        }
        HostExprKind::Map { list, .. }
        | HostExprKind::Filter { list, .. }
        | HostExprKind::Fold { list, .. }
        | HostExprKind::Scan { list, .. }
        | HostExprKind::Partition { list, .. }
        | HostExprKind::FlatMap { list, .. } => host_body_has_fallback_call(list),
        HostExprKind::TensorCall { args, .. } => args.iter().any(host_body_has_fallback_call),
        HostExprKind::AdtFieldAccess { base, .. } => host_body_has_fallback_call(base),
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            host_body_has_fallback_call(scrutinee)
                || host_body_has_fallback_call(some_expr)
                || host_body_has_fallback_call(none_expr)
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            host_body_has_fallback_call(scrutinee)
                || arms
                    .iter()
                    .any(|arm| host_body_has_fallback_call(&arm.expr))
                || default_expr
                    .as_ref()
                    .is_some_and(|d| host_body_has_fallback_call(d))
        }
        HostExprKind::AdtConstruct { fields, .. } => fields.iter().any(host_body_has_fallback_call),
        HostExprKind::WithSeed { seed, body, .. } => {
            host_body_has_fallback_call(seed) || host_body_has_fallback_call(body)
        }
        _ => false,
    }
}

fn lower_host_function(
    name: &str,
    body: &Expr,
    ty_expr: Option<&Expr>,
    program: &CheckedProgram,
) -> Option<HostFunction> {
    let declared_fn_type_expr = ty_expr
        .cloned()
        .or_else(|| lookup_declared_type_expr(program, name).cloned());
    let fn_type_parts = declared_fn_type_expr
        .as_ref()
        .and_then(parse_fn_type_expr_parts);
    let (param_tys, ret_ty) = ty_expr
        .and_then(parse_fn_type_expr)
        .or_else(|| expr_fn_type(body))
        .or_else(|| lookup_declared_fn_type(program, name))
        .unwrap_or((Vec::new(), fresh_host_inference()));

    let mut scope = HashMap::new();
    let mut params = Vec::new();
    let mut tensor_helpers = Vec::new();
    let body_expr = if let Expr::List(list, _) = body {
        if tag(list) == Some("fn") {
            let kids = children(list);
            let params_list = kids.first().and_then(as_list)?;
            if tag(params_list) != Some("params") {
                return None;
            }
            for (index, param) in children(params_list).iter().enumerate() {
                let Some(pname) = param_name(param) else {
                    continue;
                };
                let pty = param_host_type(param)
                    .filter(|ty| !ty.is_unresolved())
                    .or_else(|| {
                        param_tys
                            .get(index)
                            .cloned()
                            .filter(|ty| !ty.is_unresolved())
                    })
                    .unwrap_or_else(fresh_host_inference);
                scope.insert(pname.clone(), pty.clone());
                params.push(HostParam {
                    name: pname,
                    ty: pty,
                });
            }
            kids.get(1)?.clone()
        } else {
            if param_tys.is_empty() && ret_ty.is_unresolved() {
                return None;
            }
            for (index, param_ty) in param_tys.iter().enumerate() {
                let pname = format!("arg{index}");
                scope.insert(pname.clone(), param_ty.clone());
                params.push(HostParam {
                    name: pname,
                    ty: param_ty.clone(),
                });
            }
            synthesize_callable_application(
                body,
                &params,
                fn_type_parts.as_ref().map(|(params, _)| params.as_slice()),
                fn_type_parts.as_ref().map(|(_, ret)| ret),
            )
        }
    } else {
        return None;
    };
    // Inline any local callable bindings (fn / grad / vmap / vmap-grad)
    // into the body before lowering. The host backend only recognizes
    // grad/vmap forms in direct callee position of an `app`, so an alias
    // like `let g = grad(f); g(x)` must be rewritten to the inline
    // `(grad(f))(x)` form. Without this pass `g` lowers to an
    // `__unresolved_grad` builtin and the call falls through to a
    // generic `call(g, …)` host builtin, which `host_program_unresolved_call_sites`
    // rejects pre-codegen.
    let body_expr = inline_local_callable_lets(&body_expr);
    // If the declared return type is a tensor, the body must produce a
    // tensor even when downstream type-metadata annotations are missing
    // from the reef'd deep AST. Force the body through the tensor-helper
    // path in that case so that pure-tensor wrapper defs like
    // `Std.Tensor.Reduce.min` get a real C function symbol rather than a
    // fallthrough `HostExpr::new(HostExprKind::Builtin)` with an "unsupported builtin"
    // placeholder (Phase 3j-pre Batch 5b bug 4).
    //
    // But skip the tensor-helper path when any param is callable: the DAG
    // helper has no representation for fn-pointer inputs and would otherwise
    // coerce the callable into `chelis_scalar_tensor_from_f64`, emitting C
    // that gcc rejects. Go straight through host-lane lowering so the fn
    // application becomes a direct `f(x)` call.
    let any_callable_param = params
        .iter()
        .any(|param| matches!(param.ty, HostTypeTerm::Fn(_, _)));
    let mut host_body = if let HostTypeTerm::Tensor(expected) = ret_ty.clone()
        && !any_callable_param
        && !expr_needs_host_lane_tensor_lowering(&body_expr, program)
        && !expr_calls_top_level_fn_with_callable_param(&body_expr, program)
        && !expr_calls_summary_rejecting_top_level_fn(&body_expr, program)
        && !should_keep_tensor_expr_in_host_lane(&body_expr)
    {
        try_lower_tensor_helper_call(&body_expr, program, &scope, &mut tensor_helpers, expected)
            .unwrap_or_else(|| lower_host_expr(&body_expr, program, &scope, &mut tensor_helpers))
    } else {
        lower_host_expr(&body_expr, program, &scope, &mut tensor_helpers)
    };
    // Per `spec/design/chelis_span_survival.md` §2.3 host-side table, the
    // "Tensor-helper extraction" and "Lowering. Fn-body" rules: every
    // input Deep span must surface as `span_id` or in `merged_spans` on at
    // least one HostExpr node. Two paths above bypass the
    // `lower_host_expr` wrapper's region-corresponding stamping:
    //
    // (1) `try_lower_tensor_helper_call(&body_expr, …)` at the success
    //     branch returns a `TensorCall` constructed via `HostExpr::new(…)`
    //     directly — `body_expr.span_id()` (the fn-body inner expr's
    //     span) is dropped.
    // (2) When `body` is a `(fn {span: …} (params …) body_expr)` form,
    //     all three subpaths above lower `body_expr` (kids[1]) but
    //     never see `body` itself — so the `(fn …)` form's own
    //     `meta["span"]` is dropped on every path.
    //
    // Append both spans to `host_body.merged_spans`. The
    // `append_merged_span` helper handles None-noop, dedup, lex-sort,
    // and canonical-equal-noop, so paths that already have the span as
    // canonical (the wrapper-routed lowering) are a no-op.
    host_body.append_merged_span(body_expr.span_id());
    host_body.append_merged_span(body.span_id());
    refine_function_params_from_body(&mut params, &host_body);
    let ret_ty = if ret_ty.is_unresolved() {
        host_expr_type(&host_body)
    } else {
        ret_ty
    };
    Some(HostFunction {
        name: name.to_string(),
        params,
        ret_ty,
        body: host_body,
        tensor_helpers,
        specialization: None,
        summary_rejections: Vec::new(),
    })
}

fn synthesize_callable_application(
    body: &Expr,
    params: &[HostParam],
    param_type_exprs: Option<&[Expr]>,
    ret_type_expr: Option<&Expr>,
) -> Expr {
    let span = body.span();
    let mut elements = vec![
        Expr::Atom(Atom::Symbol("app".to_string()), span),
        Expr::Map(
            chelis_deep::ast::MetaMap {
                entries: ret_type_expr
                    .cloned()
                    .map(|ret| vec![("type".to_string(), ret)])
                    .unwrap_or_default(),
            },
            span,
        ),
        body.clone(),
    ];
    for (index, param) in params.iter().enumerate() {
        let var_meta = chelis_deep::ast::MetaMap {
            entries: param_type_exprs
                .and_then(|tys| tys.get(index))
                .cloned()
                .map(|ty| vec![("type".to_string(), ty)])
                .unwrap_or_default(),
        };
        elements.push(Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Symbol("var".to_string()), span),
                    Expr::Map(var_meta, span),
                    Expr::Atom(Atom::Symbol(param.name.clone()), span),
                ],
            },
            span,
        ));
    }
    Expr::List(List { elements }, span)
}

/// Force-lower an expression through the tensor-helper path using an
/// explicit expected tensor type hint. This is used by
/// `lower_host_function` so that pure-tensor wrapper function bodies get a
/// real C function definition even when downstream type metadata is
/// missing on the reef'd deep AST's `app` nodes.
fn try_lower_tensor_helper_call(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
    expected: TensorType,
) -> Option<HostExpr> {
    if let Expr::List(list, _) = expr
        && tag(list) == Some("var")
        && let Some(name) = children(list).first().and_then(symbol_name)
    {
        return Some(HostExpr::new(HostExprKind::Var(
            name.to_string(),
            HostTypeTerm::Tensor(expected),
        )));
    }
    let dag = lower_tensor_helper_dag(expr, program, scope, &expected)?;
    // Reject DAGs whose inputs reference known builtin names: a `Load("fold")`
    // (or `einsum`, `map`, etc.) means the lowerer fell back to treating a
    // host-lane builtin as a free variable. Emitting this DAG would generate
    // C with a `__tensor_scalar0_0 = fold;` line — `fold` is not a C symbol.
    // Fall back to `lower_host_expr` which handles HOFs directly.
    for node in dag.nodes() {
        if let crate::dag::RiscOp::Load { name } = &node.op
            && BUILTIN_NAMES.contains(&name.as_str())
        {
            return None;
        }
    }
    Some(finish_tensor_helper_call(
        dag,
        scope,
        tensor_helpers,
        expected,
    ))
}

fn lower_tensor_helper_dag(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    expected: &TensorType,
) -> Option<crate::Dag> {
    let defs = collect_program_defs(program.exprs());
    // chelis#631: never swallow a fail-reaching FORWARD body into a
    // tensor helper. The DAG lane lowers `fail` to a mask-selected zero
    // placeholder, so a forward helper would compile into a binary that
    // returns zeros where `chelis eval` aborts with the user's message
    // (silent-wrong, the one outcome the soundness bar forbids). Bail to
    // the host lane, whose `if`/`fail` are real control flow
    // (`chelis_fail` in the C emit). Applied defs are consulted because
    // helper lowering INLINES them into the DAG. A `grad`/`vmap`-carrying
    // expr is exempt: it can ONLY lower through the DAG lane, where the
    // guard-as-mask-arithmetic form (zero placeholder included) is the
    // documented chelis#616 differentiation semantics. KNOWN RESIDUAL
    // (chelis#662, pre-existing): the exemption is whole-expression, so a
    // forward `fail` sitting BESIDE a grad call in one body keeps mask
    // semantics and its C binary silently zeros where eval aborts; the
    // precise fix is scoping the exemption to the differentiated
    // sub-expression.
    if !expr_contains_grad_like(expr) && expr_reaches_fail(expr, &defs, &mut HashSet::new()) {
        return None;
    }
    // Issue #197: surface a fatal AD rejection from the tensor-
    // helper sub-lowering instead of swallowing it; the host
    // fallback would otherwise emit an undefined-symbol call to
    // the rejected grad function.
    let dag = match crate::lower::try_lower_subexpr_program(
        expr,
        collect_tensor_scope(scope),
        program.type_env().clone(),
        defs,
    ) {
        Ok(dag) => dag,
        Err(diagnostic) if diagnostic.fatal => {
            crate::lower::raise_fatal_lowering_diagnostic(diagnostic)
        }
        Err(_) => return None,
    };
    Some(remap_tensor_helper_dim_symbols(&dag, scope, expected))
}

fn finish_tensor_helper_call(
    dag: crate::Dag,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
    expected: TensorType,
) -> HostExpr {
    let helper_index = tensor_helpers.len();
    let helper_name = format!("__host_tensor_helper_{helper_index}");
    let inputs = tensor_helper_inputs(&dag);
    let output = dag
        .roots()
        .first()
        .and_then(|id| dag.get(*id))
        .map(|node| node.output_type.clone())
        .unwrap_or_else(|| expected.clone());
    // Issue #309: a helper whose body has more than one DAG root (the
    // canonical case is a multi-`wrt` `grad`, which differentiates a
    // scalar w.r.t. several tensor params and so produces one gradient
    // tensor per param) returns a TUPLE of tensors, not a single
    // tensor. The DAG emit wires `roots[i]` to `outputs[i]` with
    // `n_out = roots().len()`, and a downstream `.N` projection reads
    // root `N`. Typing the call as a single `Tensor` here made the
    // projection emit `chelis_tuple_get` over a `chelis_tensor*`
    // receiver (a mistyped crash) and sized the helper output array to
    // one slot for a two-output helper. Mirror the IR/eval-lane
    // `LoweredValue::Tuple` semantics by typing the multi-root call as
    // a `Tuple` of the per-root tensor types, in root order.
    let root_tys: Vec<HostTypeTerm> = dag
        .roots()
        .iter()
        .map(|id| {
            dag.get(*id)
                .map(|node| HostTypeTerm::Tensor(node.output_type.clone()))
                .unwrap_or_else(|| HostTypeTerm::Tensor(expected.clone()))
        })
        .collect();
    let call_ty = if root_tys.len() > 1 {
        HostTypeTerm::Tuple(root_tys)
    } else {
        HostTypeTerm::Tensor(expected.clone())
    };
    let args = tensor_helper_args(&inputs, scope);
    let (sparse_specialization, sparse_rejection) =
        match try_summarize_sparse_helper(&dag, &inputs, &output) {
            Ok(spec) => (Some(spec), None),
            Err(SparseSummaryAttempt::NotEligible) => (None, None),
            Err(SparseSummaryAttempt::Rejected(rejection)) => (None, Some(rejection)),
        };
    // W6 Task A — drive the BLAS recognizer through the structured
    // entry point so a BLAS-near rejection threads through to
    // `summary_rejection` as a `Blas*` `SummaryRejection` (rather
    // than the prior silent `Option::None` drop).
    let (blas_specialization, blas_rejection) =
        match try_summarize_blas_helper(&dag, &inputs, &output) {
            Ok(spec) => (Some(HostTensorSpecialization::BlasMatmul(spec)), None),
            Err(BlasSummaryAttempt::NotEligible) => (None, None),
            Err(BlasSummaryAttempt::Rejected(rejection)) => (None, Some(rejection)),
        };
    let specialization = blas_specialization.or(sparse_specialization);
    // Reconcile sparse vs BLAS rejections:
    //
    //   * If either recognizer accepted the helper, no rejection
    //     should be reported on this helper (the specialization
    //     takes over).
    //   * If sparse rejected, the helper body had a sparse op —
    //     that's the more specific signal; report the sparse
    //     rejection.
    //   * If sparse said NotEligible (no sparse op anywhere) and
    //     BLAS rejected, report the BLAS rejection.
    //   * If neither recognizer reached the rejection arm
    //     (both NotEligible), report nothing.
    let summary_rejection = if specialization.is_some() {
        None
    } else {
        sparse_rejection.or(blas_rejection)
    };
    tensor_helpers.push(HostTensorHelper {
        name: helper_name,
        dag,
        inputs,
        output,
        specialization,
        summary_rejection,
    });
    HostExpr::new(HostExprKind::TensorCall {
        helper: helper_index,
        args,
        ty: call_ty,
    })
}

/// Outcome of the structured BLAS-helper recognizer (W6 Task A).
/// Mirrors `SparseSummaryAttempt`: distinguishes "not even a BLAS
/// helper" (silent skip) from "near-eligible but rejected for a
/// specific structural reason" (emit a diagnostic).
///
/// The `NotEligible` arm is treated as a non-error skip by the outer
/// summary-derivation pass; the `Rejected` arm carries a
/// `HelperSummaryRejection` that is promoted to a fully-formed
/// `SummaryRejection` once the owning function's name + callsite span
/// are known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlasSummaryAttempt {
    /// Helper body's post-specialize root is not (anywhere near) a
    /// `BlasMatmul` op — i.e. there is no matmul-shape subgraph for
    /// the BLAS recognizer to fold. Today this fires when the
    /// specialized DAG has zero roots (empty body or fully-DCE'd
    /// body). No diagnostic should be emitted.
    NotEligible,
    /// Helper body had at least one specialized root that the BLAS
    /// recognizer attempted to match, but the structural check
    /// failed. The carried rejection identifies the failure class.
    Rejected(HelperSummaryRejection),
}

/// Pub-test entry point for the structured-rejection-aware BLAS
/// summarizer. Mirrors `try_summarize_sparse_helper_for_test`.
///
/// IR-level tests in `crates/chelis-ir/tests/host_blas_summary_diagnostics.rs`
/// drive synthetic helper DAGs through this entry point to lock the
/// six BLAS rejection variants without going through the full
/// `lower_compiled_program` pipeline.
#[doc(hidden)]
pub fn try_summarize_blas_helper_for_test(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Result<HostBlasMatmulSummary, BlasSummaryAttempt> {
    try_summarize_blas_helper(dag, inputs, output)
}

/// Derive a BLAS-matmul summary for a helper whose specialized DAG is
/// a single `RiscOp::BlasMatmul` root whose operands are direct
/// `RiscOp::Load`s referencing helper inputs.
///
/// Rejection cases — each maps to a `SummaryRejectionClass::Blas*`
/// variant (the six W6 Task A variants):
///
///   * `BlasOutputPrecisionMismatch` — helper output precision is not `f32`
///   * `BlasMultipleRoots` — specialized DAG has more than one root
///   * `BlasNotMatmulPattern` — root op is not `BlasMatmul`, or its
///     operand count / output precision doesn't match the BlasMatmul
///     shape
///   * `BlasNonLoadOperand` — a matmul operand is not a direct `Load`
///   * `BlasInputPrecisionMismatch` — a helper input has precision
///     other than `f32`
///   * `BlasDimensionBindingFailure` — a matmul dim (batch/M/N/K)
///     cannot be bound to any helper input
///
/// The pre-eligibility check that produces `NotEligible` (silent skip,
/// not a diagnostic) fires when the specialized DAG has zero roots —
/// the body had nothing for the BLAS recognizer to look at.
fn try_summarize_blas_helper(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Result<HostBlasMatmulSummary, BlasSummaryAttempt> {
    let specialized = crate::specialize::specialize_for_blas(dag);
    if specialized.roots().is_empty() {
        return Err(BlasSummaryAttempt::NotEligible);
    }
    // Helper-body span: prefer the specialized root's span; fall back
    // to the pre-specialize root's span; otherwise None.
    let body_span = specialized
        .roots()
        .first()
        .and_then(|id| specialized.get(*id))
        .and_then(|n| n.span_id.clone())
        .or_else(|| {
            dag.roots()
                .first()
                .and_then(|id| dag.get(*id))
                .and_then(|n| n.span_id.clone())
        });
    // Pre-eligibility: was the helper body matmul-near at all? If
    // the specialized root is neither `BlasMatmul` (the accepted
    // shape) nor a `Sum(Mul(Expand, Expand))` pattern (the
    // matmul-near shape that `specialize_for_blas` keeps as-is when
    // it cannot replace, e.g. non-F32 precision), the recognizer
    // should NOT emit a diagnostic; this is just a non-BLAS helper.
    //
    // `is_matmul_near` returns `true` for both BLAS-shaped and
    // matmul-pattern-shaped specialized roots, so we can distinguish
    // "near-eligible BLAS helper" from "totally unrelated helper".
    if specialized.roots().len() == 1 {
        let only_root = specialized.roots()[0];
        if let Some(root_node) = specialized.get(only_root)
            && !is_matmul_near(&specialized, root_node)
        {
            return Err(BlasSummaryAttempt::NotEligible);
        }
    }
    if specialized.roots().len() != 1 {
        // Even a multi-root helper qualifies as "BLAS-near" only if
        // at least one root is matmul-shape; otherwise it's an
        // unrelated multi-output helper and we silently skip.
        let any_matmul_near = specialized
            .roots()
            .iter()
            .filter_map(|id| specialized.get(*id))
            .any(|n| is_matmul_near(&specialized, n));
        if !any_matmul_near {
            return Err(BlasSummaryAttempt::NotEligible);
        }
        return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::BlasMultipleRoots,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::BlasMultipleRoots {
                root_count: specialized.roots().len(),
            },
        }));
    }
    // From here we are committed: the specialized DAG is BLAS-near
    // and has exactly one root. Output-precision is the next gate.
    // WS-A1/A2/A3 lift: f32 (sgemm), f64 (dgemm), bf16/f16 (hipblasGemmEx
    // with f32 accumulator) are all admitted; integer matmul is rejected
    // upstream at the type checker per spec §5.7.2.
    if !matches!(
        output.precision,
        Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16
    ) {
        return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::BlasOutputPrecisionMismatch,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::BlasOutputPrecisionMismatch {
                observed: output.precision,
            },
        }));
    }
    let root = *specialized
        .roots()
        .first()
        .expect("checked roots().len() == 1 above");
    let root_node = match specialized.get(root) {
        Some(node) => node,
        None => return Err(BlasSummaryAttempt::NotEligible),
    };
    let (batch_dims, m, n, k) = match &root_node.op {
        RiscOp::BlasMatmul {
            batch_dims,
            m,
            n,
            k,
            ..
        } => (batch_dims.clone(), m.clone(), n.clone(), k.clone()),
        other => {
            return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
                rejection_class: SummaryRejectionClass::BlasNotMatmulPattern,
                helper_body_span: body_span,
                detail: SummaryRejectionDetail::BlasNotMatmulPattern {
                    tail_op: risc_op_canonical_name(other).to_string(),
                },
            }));
        }
    };
    let root_precision_admitted = matches!(
        root_node.output_type.precision,
        Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16
    );
    if !root_precision_admitted || root_node.inputs.len() != 2 {
        // Root IS BlasMatmul but its rank/precision doesn't match
        // the recognized shape. Still a BlasNotMatmulPattern
        // rejection; the variant name covers both "wrong op" and
        // "right op, wrong shape".
        return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::BlasNotMatmulPattern,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::BlasNotMatmulPattern {
                tail_op: "blas_matmul".to_string(),
            },
        }));
    }
    let lhs_input = match helper_load_input_index(&specialized, root_node.inputs[0], inputs) {
        Some(idx) => idx,
        None => {
            return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
                rejection_class: SummaryRejectionClass::BlasNonLoadOperand,
                helper_body_span: body_span,
                detail: SummaryRejectionDetail::BlasNonLoadOperand { operand_index: 0 },
            }));
        }
    };
    let rhs_input = match helper_load_input_index(&specialized, root_node.inputs[1], inputs) {
        Some(idx) => idx,
        None => {
            return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
                rejection_class: SummaryRejectionClass::BlasNonLoadOperand,
                helper_body_span: body_span,
                detail: SummaryRejectionDetail::BlasNonLoadOperand { operand_index: 1 },
            }));
        }
    };
    let input_tys = inputs
        .iter()
        .map(|input| input.ty.clone())
        .collect::<Vec<_>>();
    // WS-A1/A2/A3: admit f32 (sgemm), f64 (dgemm), bf16/f16 (hipblasGemmEx
    // with f32 accumulator). Integer matmul is rejected upstream at the
    // type checker per spec §5.7.2 so it never reaches here.
    if let Some((input_index, ty)) = input_tys
        .iter()
        .enumerate()
        .find(|(_, ty)| !matches!(ty.precision, Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16))
    {
        return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::BlasInputPrecisionMismatch,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::BlasInputPrecisionMismatch {
                input_index,
                observed: ty.precision,
            },
        }));
    }
    if !summary_dims_bind_to_inputs(&input_tys, &batch_dims) {
        return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::BlasDimensionBindingFailure,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::BlasDimensionBindingFailure {
                role: BlasDimRole::Batch,
            },
        }));
    }
    for (dim, role) in [
        (&m, BlasDimRole::M),
        (&n, BlasDimRole::N),
        (&k, BlasDimRole::K),
    ] {
        if !summary_dims_bind_to_inputs(&input_tys, std::slice::from_ref(dim)) {
            return Err(BlasSummaryAttempt::Rejected(HelperSummaryRejection {
                rejection_class: SummaryRejectionClass::BlasDimensionBindingFailure,
                helper_body_span: body_span,
                detail: SummaryRejectionDetail::BlasDimensionBindingFailure { role },
            }));
        }
    }
    Ok(HostBlasMatmulSummary {
        lhs_input,
        rhs_input,
        input_tys,
        output: output.clone(),
        batch_dims,
        m,
        n,
        k,
    })
}

/// Pub-test entry point for the sparse summarizer. Wraps the
/// crate-private `summarize_sparse_helper_from_parts` so integration
/// tests in `crates/chelis-ir/tests/` can lock the recognizer
/// directly without driving a full `lower_compiled_program` pipeline.
///
/// Surface code today has no path that lowers to `RiscOp::ScatterAdd`
/// (that op is produced exclusively by AD adjoint of `gather`), so
/// the integration test that exercises ScatterAdd-helper recognition
/// constructs a synthetic helper DAG and calls through here.
#[doc(hidden)]
pub fn summarize_sparse_helper_for_test(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Option<HostTensorSpecialization> {
    summarize_sparse_helper_from_parts(dag, inputs, output)
}

/// Pub-test entry point for the structured-rejection-aware sparse
/// summarizer. Mirrors `summarize_sparse_helper_for_test` but exposes
/// the W4-A `SparseSummaryAttempt` result so IR-level tests can lock
/// the structured rejection class + detail for each near-eligible
/// rejection case.
#[doc(hidden)]
pub fn try_summarize_sparse_helper_for_test(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Result<HostTensorSpecialization, SparseSummaryAttempt> {
    try_summarize_sparse_helper(dag, inputs, output)
}

/// Outcome of the structured sparse-helper recognizer. Distinguishes
/// "not even a sparse helper" (silent skip) from "near-eligible but
/// rejected for a specific structural reason" (emit a diagnostic).
///
/// The `NotEligible` arm is treated as a non-error skip by the
/// outer summary-derivation pass; the `Rejected` arm carries a
/// `HelperSummaryRejection` that will be promoted to a fully-formed
/// `SummaryRejection` once the owning function's name + callsite
/// span are known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseSummaryAttempt {
    /// Helper body contains no sparse RiscOp anywhere. Not a
    /// near-summary case; no diagnostic should be emitted. The outer
    /// pass falls through to the BLAS recognizer.
    NotEligible,
    /// Helper body has a sparse RiscOp in a position where the
    /// summarizer attempts recognition, but the structural check
    /// failed. The carried rejection identifies the failure class.
    Rejected(HelperSummaryRejection),
}

/// Back-compat helper around `try_summarize_sparse_helper` that
/// collapses both error arms to `None`. Used by the existing
/// `summarize_sparse_helper_for_test` IR-level test entry point and
/// by any code path that does not consume the structured rejection.
fn summarize_sparse_helper_from_parts(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Option<HostTensorSpecialization> {
    try_summarize_sparse_helper(dag, inputs, output).ok()
}

/// Derive a sparse-op summary for a helper whose DAG is a single
/// `RiscOp::Gather`, `RiscOp::ScatterAdd`, or `RiscOp::Scatter` root
/// whose operands are direct `RiscOp::Load`s referencing helper
/// inputs. When the helper body contains a sparse op but the
/// recognizer rejects, the returned `Err(SparseSummaryAttempt::Rejected(...))`
/// carries a structured rejection class + detail per W4-A; consumers
/// programmatically match on the class rather than on rendered
/// strings.
///
/// Rejection cases — each maps to a `SummaryRejectionClass` variant:
///
/// * `MultipleRoots` — helper body has more than one DAG root.
/// * `WildcardDim` — helper input/output carries a `Named("*", None)` wildcard.
/// * `PostProcessingAfterSparseOp` — root op is not sparse but a sparse op
///   appears in the body.
/// * `NonLoadOperand` — a sparse-op operand is not a direct `RiscOp::Load`, or
///   its `Load` does not match a helper input by name + type (rank / type
///   mismatch surfaces here until a dedicated `RankMismatch` check is added).
/// * `IndicesDTypeMismatch` — indices precision is not int32 / int64.
/// * `PayloadDTypeMismatch` — values / target / updates precision disagrees
///   with output precision.
fn try_summarize_sparse_helper(
    dag: &crate::Dag,
    inputs: &[HostTensorInput],
    output: &TensorType,
) -> Result<HostTensorSpecialization, SparseSummaryAttempt> {
    if dag.is_empty() {
        return Err(SparseSummaryAttempt::NotEligible);
    }
    // Pre-scan: does the DAG mention any sparse op at all? If not, the
    // recognizer is not the right code path; the BLAS recognizer (or
    // no specialization) takes over silently.
    let deepest_sparse_op = dag
        .nodes()
        .iter()
        .find_map(|node| sparse_op_kind(&node.op).map(|kind| (kind, node)));
    if deepest_sparse_op.is_none() {
        return Err(SparseSummaryAttempt::NotEligible);
    }
    let (deepest_op_kind, _deepest_node) = deepest_sparse_op.unwrap();

    // Helper-body span: prefer the deepest sparse node's span; fall
    // back to the helper root's span; otherwise None.
    let body_span = dag
        .roots()
        .first()
        .and_then(|id| dag.get(*id))
        .and_then(|n| n.span_id.clone())
        .or_else(|| {
            dag.nodes()
                .iter()
                .find_map(|n| sparse_op_kind(&n.op).and(n.span_id.clone()))
        });

    if dag.roots().len() != 1 {
        return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::MultipleRoots,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::MultipleRoots {
                root_count: dag.roots().len(),
            },
        }));
    }
    let root_id = *dag
        .roots()
        .first()
        .expect("checked roots().len() == 1 above");
    let root = match dag.get(root_id) {
        Some(node) => node,
        None => return Err(SparseSummaryAttempt::NotEligible),
    };
    if root.output_type != *output {
        // Output-type mismatch is a structural shape issue — treat as
        // NotEligible to avoid emitting a diagnostic for cases that
        // are routed through a different specialization path.
        return Err(SparseSummaryAttempt::NotEligible);
    }
    if let Some(location) = wildcard_location(output, inputs) {
        return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::WildcardDim,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::WildcardDim { location },
        }));
    }
    let input_tys = inputs.iter().map(|i| i.ty.clone()).collect::<Vec<_>>();

    // Root must itself be a sparse op. If a sparse op exists deeper in
    // the body but the root is something else, the recognizer rejects
    // with `PostProcessingAfterSparseOp` (and names the tail op).
    let root_sparse_kind = sparse_op_kind(&root.op);
    if root_sparse_kind.is_none() {
        return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
            rejection_class: SummaryRejectionClass::PostProcessingAfterSparseOp,
            helper_body_span: body_span,
            detail: SummaryRejectionDetail::PostProcessingAfterSparseOp {
                op: deepest_op_kind,
                tail_op: risc_op_canonical_name(&root.op).to_string(),
            },
        }));
    }
    let root_kind = root_sparse_kind.expect("root_sparse_kind is Some by guard");

    match &root.op {
        RiscOp::Gather { axis } => {
            // Inputs: [values, indices].
            if root.inputs.len() != 2 {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::NonLoadOperand,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::NonLoadOperand {
                        op: root_kind,
                        operand_index: root.inputs.len().min(1),
                    },
                }));
            }
            let values_idx = match helper_load_input_index(dag, root.inputs[0], inputs) {
                Some(i) => i,
                None => {
                    return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                        rejection_class: SummaryRejectionClass::NonLoadOperand,
                        helper_body_span: body_span,
                        detail: SummaryRejectionDetail::NonLoadOperand {
                            op: root_kind,
                            operand_index: 0,
                        },
                    }));
                }
            };
            let indices_idx = match helper_load_input_index(dag, root.inputs[1], inputs) {
                Some(i) => i,
                None => {
                    return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                        rejection_class: SummaryRejectionClass::NonLoadOperand,
                        helper_body_span: body_span,
                        detail: SummaryRejectionDetail::NonLoadOperand {
                            op: root_kind,
                            operand_index: 1,
                        },
                    }));
                }
            };
            let values_ty = match dag.get(root.inputs[0]) {
                Some(n) => n.output_type.clone(),
                None => return Err(SparseSummaryAttempt::NotEligible),
            };
            let indices_ty = match dag.get(root.inputs[1]) {
                Some(n) => n.output_type.clone(),
                None => return Err(SparseSummaryAttempt::NotEligible),
            };
            if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::IndicesDTypeMismatch,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::IndicesDTypeMismatch {
                        op: root_kind,
                        observed: indices_ty.precision,
                    },
                }));
            }
            if values_ty.precision != output.precision {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::PayloadDTypeMismatch,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::PayloadDTypeMismatch {
                        op: root_kind,
                        which: PayloadRole::Values,
                        expected: output.precision,
                        observed: values_ty.precision,
                    },
                }));
            }
            Ok(HostTensorSpecialization::SparseGather(
                HostSparseOpSummary {
                    axis: *axis,
                    input_indices: vec![values_idx, indices_idx],
                    input_tys,
                    output: output.clone(),
                },
            ))
        }
        RiscOp::ScatterAdd { axis } | RiscOp::Scatter { axis } => {
            // Inputs: [target, indices, updates].
            if root.inputs.len() != 3 {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::NonLoadOperand,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::NonLoadOperand {
                        op: root_kind,
                        operand_index: root.inputs.len().min(2),
                    },
                }));
            }
            let target_idx = match helper_load_input_index(dag, root.inputs[0], inputs) {
                Some(i) => i,
                None => {
                    return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                        rejection_class: SummaryRejectionClass::NonLoadOperand,
                        helper_body_span: body_span,
                        detail: SummaryRejectionDetail::NonLoadOperand {
                            op: root_kind,
                            operand_index: 0,
                        },
                    }));
                }
            };
            let indices_idx = match helper_load_input_index(dag, root.inputs[1], inputs) {
                Some(i) => i,
                None => {
                    return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                        rejection_class: SummaryRejectionClass::NonLoadOperand,
                        helper_body_span: body_span,
                        detail: SummaryRejectionDetail::NonLoadOperand {
                            op: root_kind,
                            operand_index: 1,
                        },
                    }));
                }
            };
            let updates_idx = match helper_load_input_index(dag, root.inputs[2], inputs) {
                Some(i) => i,
                None => {
                    return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                        rejection_class: SummaryRejectionClass::NonLoadOperand,
                        helper_body_span: body_span,
                        detail: SummaryRejectionDetail::NonLoadOperand {
                            op: root_kind,
                            operand_index: 2,
                        },
                    }));
                }
            };
            let target_ty = match dag.get(root.inputs[0]) {
                Some(n) => n.output_type.clone(),
                None => return Err(SparseSummaryAttempt::NotEligible),
            };
            let indices_ty = match dag.get(root.inputs[1]) {
                Some(n) => n.output_type.clone(),
                None => return Err(SparseSummaryAttempt::NotEligible),
            };
            let updates_ty = match dag.get(root.inputs[2]) {
                Some(n) => n.output_type.clone(),
                None => return Err(SparseSummaryAttempt::NotEligible),
            };
            if !matches!(indices_ty.precision, Prim::Int32 | Prim::Int64) {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::IndicesDTypeMismatch,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::IndicesDTypeMismatch {
                        op: root_kind,
                        observed: indices_ty.precision,
                    },
                }));
            }
            if target_ty.precision != output.precision {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::PayloadDTypeMismatch,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::PayloadDTypeMismatch {
                        op: root_kind,
                        which: PayloadRole::Target,
                        expected: output.precision,
                        observed: target_ty.precision,
                    },
                }));
            }
            if updates_ty.precision != output.precision {
                return Err(SparseSummaryAttempt::Rejected(HelperSummaryRejection {
                    rejection_class: SummaryRejectionClass::PayloadDTypeMismatch,
                    helper_body_span: body_span,
                    detail: SummaryRejectionDetail::PayloadDTypeMismatch {
                        op: root_kind,
                        which: PayloadRole::Updates,
                        expected: output.precision,
                        observed: updates_ty.precision,
                    },
                }));
            }
            let summary = HostSparseOpSummary {
                axis: *axis,
                input_indices: vec![target_idx, indices_idx, updates_idx],
                input_tys,
                output: output.clone(),
            };
            Ok(match &root.op {
                RiscOp::ScatterAdd { .. } => HostTensorSpecialization::SparseScatterAdd(summary),
                RiscOp::Scatter { .. } => HostTensorSpecialization::SparseScatterReplace(summary),
                _ => unreachable!("matched arm guarantees op kind"),
            })
        }
        _ => unreachable!("root_sparse_kind.is_some() guard rules out non-sparse roots"),
    }
}

/// `true` when `root_node` is the root of a matmul-near subgraph in
/// the specialized DAG. Used by the BLAS recognizer's
/// pre-eligibility check (W6 Task A) to distinguish "this helper's
/// body has a matmul shape that the recognizer attempted to fold"
/// from "this helper is totally unrelated to matmul".
///
/// Returns `true` for two shapes:
///   * `RiscOp::BlasMatmul` — the post-specialize accepted shape
///   * `RiscOp::Sum` whose sole input is `RiscOp::Mul` of two
///     `RiscOp::Expand`s — the matmul-pattern shape that
///     `specialize_for_blas` leaves as-is when it cannot replace
///     (e.g. when `detect_matmul_pattern` rejects on non-F32
///     precision per the W5 P0 fix).
///
/// Returns `false` for everything else (elementwise helpers,
/// pure-sparse helpers, etc.). Those produce `NotEligible` rather
/// than a structured rejection.
fn is_matmul_near(dag: &crate::Dag, root_node: &crate::DagNode) -> bool {
    if matches!(&root_node.op, RiscOp::BlasMatmul { .. }) {
        return true;
    }
    if !matches!(&root_node.op, RiscOp::Sum { .. }) {
        return false;
    }
    if root_node.inputs.len() != 1 {
        return false;
    }
    let Some(mul_node) = dag.get(root_node.inputs[0]) else {
        return false;
    };
    if !matches!(&mul_node.op, RiscOp::Mul) || mul_node.inputs.len() != 2 {
        return false;
    }
    let Some(expand_a) = dag.get(mul_node.inputs[0]) else {
        return false;
    };
    let Some(expand_b) = dag.get(mul_node.inputs[1]) else {
        return false;
    };
    matches!(&expand_a.op, RiscOp::Expand { .. }) && matches!(&expand_b.op, RiscOp::Expand { .. })
}

/// Map a `RiscOp` to a `SparseOpKind`. Returns `None` for non-sparse
/// ops. Used to detect "near-eligible" helpers and to populate
/// rejection-class details with the specific sparse op that was
/// rejected.
fn sparse_op_kind(op: &RiscOp) -> Option<SparseOpKind> {
    match op {
        RiscOp::Gather { .. } => Some(SparseOpKind::Gather),
        RiscOp::ScatterAdd { .. } => Some(SparseOpKind::ScatterAdd),
        RiscOp::Scatter { .. } => Some(SparseOpKind::ScatterReplace),
        _ => None,
    }
}

/// Canonical snake-case name for a `RiscOp` for use in
/// `SummaryRejectionDetail::PostProcessingAfterSparseOp::tail_op`.
/// Mirrors the user-facing Surf builtin name where one exists. The
/// surface is intentionally a small whitelist of "tail ops that
/// commonly appear above a rejected sparse op in user code"; opaque
/// `<other>` is the safe default for everything else.
fn risc_op_canonical_name(op: &RiscOp) -> &'static str {
    match op {
        RiscOp::Add => "add",
        RiscOp::Mul => "mul",
        RiscOp::Neg => "neg",
        RiscOp::Abs => "abs",
        RiscOp::Reshape { .. } => "reshape",
        RiscOp::Expand { .. } => "expand",
        RiscOp::Cast { .. } => "cast",
        RiscOp::Permute { .. } => "permute",
        RiscOp::Load { .. } => "load",
        RiscOp::Const { .. } => "const",
        RiscOp::BlasMatmul { .. } => "blas_matmul",
        RiscOp::Gather { .. } => "gather",
        RiscOp::ScatterAdd { .. } => "scatter_add",
        RiscOp::Scatter { .. } => "scatter_replace",
        RiscOp::Sum { .. } => "sum",
        RiscOp::Copy => "copy",
        RiscOp::Drop => "drop",
        RiscOp::Realize => "realize",
        // Fallback: opaque rather than panicking, because the
        // canonical-name surface is exhaustive for the ops the
        // recognizer cares about but not for every RiscOp.
        _ => "<other>",
    }
}

/// Returns the location of the first `Named("*", None)` wildcard dim
/// in (output, inputs[0], inputs[1], ...), or `None` if no wildcard
/// is present. Output is checked first so that "wildcard in output"
/// is reported preferentially.
fn wildcard_location(output: &TensorType, inputs: &[HostTensorInput]) -> Option<WildcardLocation> {
    if tensor_type_has_wildcard_dim(output) {
        return Some(WildcardLocation::Output);
    }
    for (idx, input) in inputs.iter().enumerate() {
        if tensor_type_has_wildcard_dim(&input.ty) {
            return Some(WildcardLocation::Input(idx));
        }
    }
    None
}

/// `true` when `ty` carries at least one wildcard dim
/// (`Named("*", None)`). Wildcards survive from type inference when a
/// dim was unconstrained at the use site and were never bound to a
/// concrete or symbolic axis. They make summary-derived contract
/// assertions meaningless because all wildcards in a helper share the
/// same string name and would falsely collapse to one axis.
fn tensor_type_has_wildcard_dim(ty: &TensorType) -> bool {
    ty.dims
        .iter()
        .any(|dim| matches!(dim, DimInfo::Named(name, None) if name == "*"))
}

fn helper_load_input_index(
    dag: &crate::Dag,
    id: crate::dag::NodeId,
    inputs: &[HostTensorInput],
) -> Option<usize> {
    let node = dag.get(id)?;
    let RiscOp::Load { name } = &node.op else {
        return None;
    };
    inputs
        .iter()
        .position(|input| input.name == name.as_str() && input.ty == node.output_type)
}

fn summary_dims_bind_to_inputs(input_tys: &[TensorType], dims: &[DimExpr]) -> bool {
    let available = input_tys
        .iter()
        .flat_map(|ty| ty.dims.iter())
        .filter_map(|dim| match dim {
            DimInfo::Named(name, None) => Some(name.as_str()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    dims.iter()
        .flat_map(|dim| dim.symbolic_names())
        .all(|name| available.contains(name.as_str()))
}

fn lower_host_expr(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let mut result = lower_host_expr_kind(expr, program, scope, tensor_helpers);
    // Per `spec/design/chelis_span_survival.md` §2.3 host-side table,
    // rule "Lowering": every freshly-produced HostExpr inherits the
    // enclosing Deep `Expr`'s `meta["span"]` as its `span_id`. When the
    // result already carries a different `span_id` (because lowering
    // recursed into a child whose own span was attached first — the N→1
    // body-collapse case), the parent's span appends to `merged_spans`,
    // lex-sorted and deduped, so the audit chain doesn't drop it. The
    // helper handles the no-op cases (None, equal canonical, already
    // present).
    if let Some(span) = expr.span_id() {
        if result.span_id.is_none() {
            result.span_id = Some(span.to_owned());
        } else {
            result.append_merged_span(Some(span));
        }
    }
    result
}

/// Bucket 4e helper: rewrite a pipe stage `f` applied to an accumulator
/// `x` into a Deep expression that downstream host lowering can handle
/// without falling back to `Builtin { name: "call" }`. Three cases:
///
/// - `(var f) x` becomes `(app {} (var f) x)`.
/// - `(fn (params p) body) x` beta-reduces to `body[p := x]`.
/// - Other / nested apps become `(app {} stage x)` and let
///   `lower_app_host_expr` work them out.
///
/// The beta-reduction case matters because `xs |> mul(b)` desugars to
/// `(fn (params __chelis_pipe) (app mul __chelis_pipe b))`. Wrapping it
/// in an outer `(app (fn ...) x)` would produce a fallback `Builtin {
/// name: "call" }` because `lower_app_host_expr` reads the function
/// name from the first child as a `(var ...)` -- a lambda head doesn't
/// match. Beta-reduction skips the outer app entirely and the existing
/// `(app mul x b)` form lowers cleanly.
fn beta_reduce_pipe_stage(stage: &Expr, acc: Expr) -> Expr {
    use chelis_deep::Span;
    use chelis_deep::ast::{Atom, List, MetaMap};
    let span = Span::new(0, 0);
    if let Expr::List(stage_list, _) = stage
        && tag(stage_list) == Some("fn")
    {
        let kids = children(stage_list);
        if let Some(params_expr) = kids.first()
            && let Some(params_list) = as_list(params_expr)
            && tag(params_list) == Some("params")
        {
            let param_names: Vec<String> = children(params_list)
                .iter()
                .filter_map(param_name)
                .collect();
            // Single-param lambdas are the only shape the desugarer
            // produces for pipe stages (`__chelis_pipe`). Multi-param
            // lambdas in pipe position would be a user error and we
            // bail to the wrap-in-app path; the resulting fallback
            // call diagnostic surfaces a clean error from the CLI.
            if param_names.len() == 1
                && let Some(body) = kids.get(1)
            {
                return substitute_var(body, &param_names[0], &acc);
            }
        }
    }
    let elements = vec![
        Expr::Atom(Atom::Symbol("app".to_string()), span),
        Expr::Map(MetaMap::default(), span),
        stage.clone(),
        acc,
    ];
    Expr::List(List { elements }, span)
}

/// Substitute every `(var {} name)` reference in `expr` with
/// `replacement`. Only walks nodes that the host pipe rewrite produces
/// from desugaring (vars, apps, lits, fn-bodies); other Deep tags pass
/// through unchanged on the assumption they don't bind or shadow the
/// pipe parameter (which the surf desugarer guarantees by using a
/// fresh `__chelis_pipe` name).
fn substitute_var(expr: &Expr, name: &str, replacement: &Expr) -> Expr {
    match expr {
        Expr::List(list, span) => {
            if tag(list) == Some("var")
                && children(list).first().and_then(symbol_name) == Some(name)
            {
                return replacement.clone();
            }
            // `(fn (params x) body)` shadows `name` only if `x == name`.
            // The desugarer's pipe-param name (`__chelis_pipe`) is
            // unique per stage so shadowing inside a stage's body is
            // not expected, but defend against it for correctness.
            if tag(list) == Some("fn")
                && let Some(params_expr) = list.elements.get(2)
                && let Some(params_list) = as_list(params_expr)
                && tag(params_list) == Some("params")
                && children(params_list)
                    .iter()
                    .filter_map(param_name)
                    .any(|p| p == name)
            {
                return expr.clone();
            }
            let mut elements = Vec::with_capacity(list.elements.len());
            for el in &list.elements {
                elements.push(substitute_var(el, name, replacement));
            }
            Expr::List(chelis_deep::ast::List { elements }, *span)
        }
        Expr::MetaExpr(meta, span) => {
            let inner = substitute_var(&meta.expr, name, replacement);
            let entries = meta
                .entries
                .iter()
                .map(|(key, value)| (key.clone(), substitute_var(value, name, replacement)))
                .collect();
            Expr::MetaExpr(
                chelis_deep::ast::MetaExpr {
                    expr: Box::new(inner),
                    entries,
                },
                *span,
            )
        }
        Expr::Atom(_, _) | Expr::Map(_, _) => expr.clone(),
    }
}

fn lower_host_expr_kind(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let is_app_expr = matches!(expr, Expr::List(list, _) if tag(list) == Some("app"));
    if !is_app_expr
        && let Some(tensor_ty) = expr_tensor_type(expr, program, scope)
        && !should_keep_tensor_expr_in_host_lane(expr)
        && let Some(tensor_call) =
            try_lower_tensor_helper_call(expr, program, scope, tensor_helpers, tensor_ty.clone())
    {
        return tensor_call;
    }

    match expr {
        Expr::Atom(Atom::Int(value), _) => HostExpr::new(HostExprKind::Int(*value)),
        Expr::Atom(Atom::Float(value), _) => HostExpr::new(HostExprKind::Float(*value)),
        Expr::Atom(Atom::Bool(value), _) => HostExpr::new(HostExprKind::Bool(*value)),
        Expr::Atom(Atom::Str(value), _) => HostExpr::new(HostExprKind::String(value.clone())),
        Expr::List(list, _) if tag(list) == Some("tuple") => {
            let items = children(list)
                .iter()
                .map(|child| lower_host_expr(child, program, scope, tensor_helpers))
                .collect::<Vec<_>>();
            let ty = expr_host_type(expr, program, scope);
            let ty = if ty.is_unresolved() {
                HostTypeTerm::Tuple(items.iter().map(host_expr_type).collect())
            } else {
                ty
            };
            HostExpr::new(HostExprKind::Tuple(items, ty))
        }
        Expr::List(list, _) if tag(list) == Some("record") => {
            lower_record_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("lit") => lower_host_expr(
            children(list).first().unwrap_or(expr),
            program,
            scope,
            tensor_helpers,
        ),
        Expr::List(list, _) if tag(list) == Some("var") => {
            let name = children(list)
                .first()
                .and_then(symbol_name)
                .unwrap_or("_")
                .to_string();
            let ty = expr_type(expr)
                .filter(|ty| !ty.is_unresolved())
                .or_else(|| scope.get(&name).cloned())
                .or_else(|| lookup_declared_host_type(program, &name))
                .unwrap_or_else(fresh_host_inference);
            if name == "Nil" {
                return HostExpr::new(HostExprKind::List(
                    Vec::new(),
                    match ty {
                        HostTypeTerm::List(_) => ty,
                        _ => HostTypeTerm::List(Box::new(fresh_host_inference())),
                    },
                ));
            }
            if let Some((adt_name, fields)) = lookup_adt_ctor(program, &name)
                && fields.is_empty()
            {
                return HostExpr::new(HostExprKind::AdtConstruct {
                    ctor: name,
                    fields: Vec::new(),
                    ty: HostTypeTerm::Adt(adt_name, Vec::new()),
                });
            }
            HostExpr::new(HostExprKind::Var(name, ty))
        }
        Expr::List(list, _) if tag(list) == Some("if") => {
            let kids = children(list);
            let then_expr = lower_host_expr(&kids[1], program, scope, tensor_helpers);
            let else_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
            let explicit_ty = expr_host_type(expr, program, scope);
            let ty = if explicit_ty.is_unresolved() {
                let then_ty = host_expr_type(&then_expr);
                if then_ty.is_unresolved() {
                    host_expr_type(&else_expr)
                } else {
                    then_ty
                }
            } else {
                explicit_ty
            };
            HostExpr::new(HostExprKind::If {
                cond: Box::new(lower_host_expr(&kids[0], program, scope, tensor_helpers)),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
                ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some("match") => {
            lower_match_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("let") => {
            let kids = children(list);
            let mut scoped = scope.clone();
            let mut bindings = Vec::new();
            if let Some(bind_first) = kids.first()
                && let Some(bind_list) = as_list(bind_first)
                && tag(bind_list) == Some("bind")
            {
                // The `(bind {span: a} ...)` node carries its own span;
                // when its child value is lowered to an existing HostExpr
                // (the value HostExpr), the bind's span appends to the
                // value's `merged_spans` per §2.3 host-side rule (b)
                // (N→1 lowering collapse — bind wraps value).
                let bind_span = bind_first.span_id().map(|s| s.to_owned());
                let bind_children = children(bind_list);
                let mut index = 0;
                while index + 1 < bind_children.len() {
                    if let Some(name) = symbol_name(&bind_children[index]) {
                        let mut value = lower_host_expr(
                            &bind_children[index + 1],
                            program,
                            &scoped,
                            tensor_helpers,
                        );
                        value.append_merged_span(bind_span.as_deref());
                        let bind_ty = host_expr_type(&value);
                        bindings.push(HostBinding {
                            name: name.to_string(),
                            display_name: None,
                            ty: bind_ty.clone(),
                            value,
                        });
                        scoped.insert(name.to_string(), bind_ty);
                    }
                    index += 2;
                }
            }
            let body = kids
                .get(1)
                .map(|child| lower_host_expr(child, program, &scoped, tensor_helpers))
                .unwrap_or(HostExpr::new(HostExprKind::Unit));
            let explicit_ty = expr_host_type(expr, program, scope);
            HostExpr::new(HostExprKind::Let {
                bindings,
                body: Box::new(body.clone()),
                ty: if explicit_ty.is_unresolved() {
                    host_expr_type(&body)
                } else {
                    explicit_ty
                },
            })
        }
        Expr::List(list, _) if tag(list) == Some("tuple-get") => {
            lower_tuple_get_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("access") => {
            lower_access_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("cast") => {
            // chelis#730 Phase 1 (census row 13's host-lane half,
            // chelis#744): a cast target naming an unrecognized primitive
            // raises the same fatal branded diagnostic as the IR-lane
            // `lower_cast` - the build lane previously typed it Unknown,
            // fell back to the inferred operand type, and shipped a
            // working binary while eval rejected the same file. Only the
            // `(t-prim {} name)` and bare-symbol spellings are validated;
            // `t-var` targets (precision-polymorphic casts) stay legal.
            let bogus_target_name =
                match children(list).get(1) {
                    Some(Expr::List(tlist, _))
                        if tag(tlist) == Some("t-prim")
                            && children(tlist).first().and_then(symbol_name).is_some_and(
                                |name| chelis_types::types::Prim::parse_name(name).is_none(),
                            ) =>
                    {
                        children(tlist).first().and_then(symbol_name)
                    }
                    Some(Expr::Atom(Atom::Symbol(name), _))
                        if chelis_types::types::Prim::parse_name(name).is_none() =>
                    {
                        Some(name.as_str())
                    }
                    _ => None,
                };
            if let Some(bogus) = bogus_target_name {
                let unsupported = chelis_types::unsupported::Unsupported::new(
                    chelis_types::unsupported::UnsupportedKind::Dtype(bogus.to_string()),
                    "a `cast` target in host lowering",
                    chelis_types::unsupported::Stage::Lowering,
                    "the cast target must name an active primitive type \
                     (spec/04-type-system.md section 1.1); a bogus target previously \
                     lowered as the operand type silently in the build lane \
                     (chelis#744, chelis#730 census row 13)",
                );
                crate::lower::raise_fatal_lowering_diagnostic(crate::lower::LowerDiagnostic {
                    message: unsupported.to_string(),
                    span: None,
                    span_id: expr.span_id().map(ToOwned::to_owned),
                    fatal: true,
                });
            }
            let value = lower_host_expr(
                children(list).first().unwrap_or(expr),
                program,
                scope,
                tensor_helpers,
            );
            let inferred_ty = host_expr_type(&value);
            let ty = expr_host_type(expr, program, scope);
            HostExpr::new(HostExprKind::Builtin {
                name: "cast".to_string(),
                args: vec![value],
                ty: if ty.is_unresolved() { inferred_ty } else { ty },
            })
        }
        Expr::List(list, _) if tag(list) == Some("app") => {
            lower_app_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("pipe") => {
            // Bucket 4e: a pipe expression that survives to host
            // lowering (top-level value bindings, or pipes whose seed
            // can't be type-resolved) is rewritten into the equivalent
            // nested-app form so downstream lowering sees the same
            // shape used for explicit nested calls. Without this arm
            // the whole form fell through to
            // `HostExpr::new(HostExprKind::Unit)`, so a top-level
            // pipe binding to a user-defined fn materialised as `()`
            // in generated C even though `chelis check` accepted the
            // tensor-typed shape. The IR-side `lower_pipe` fix at
            // `chelis_ir::lower::lower_pipe` already handles the DAG
            // path; this is the host-lane sibling.
            let kids = children(list);
            let Some((seed, stages)) = kids.split_first() else {
                return HostExpr::new(HostExprKind::Unit);
            };
            let mut current = (*seed).clone();
            for stage in stages {
                current = beta_reduce_pipe_stage(stage, current);
            }
            lower_host_expr(&current, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("handle-effect") => {
            // `with seed(...) { body }` and similar effect handlers are
            // pure-result from the host emitter's perspective. Random
            // handlers still need a host-lane seed scope so calls into
            // separately emitted stdlib/helper functions see the active seed.
            let kids = children(list);
            // chelis#730 Phase 1 (census row 20; the host-lane sibling of
            // row 9, discovered during the row 9 conversion): the former
            // unconditional body-passthrough silently dropped the handler
            // for every non-`random` effect kind, including unknown ones.
            // chelis#730 Phase 2 (section C4.4): the kind is parsed once
            // into the closed [`EffectKind`] set and dispatched with an
            // exhaustive `match` (no `_` arm), so a new kind is a compile
            // error here. A decode error raises the same
            // fatal branded diagnostic as the IR-lane arm.
            let effect_kind = decode_effect_kind(list).unwrap_or_else(|error| {
                let unsupported = chelis_types::unsupported::Unsupported::new(
                    chelis_types::unsupported::UnsupportedKind::EffectKind(error.to_string()),
                    "a `handle-effect` form in host lowering",
                    chelis_types::unsupported::Stage::Lowering,
                    "known effect kinds are `random` and `resource` \
                     (spec/03-deep-syntax.md); an unknown kind previously dropped its \
                     handler silently (chelis#730 census rows 9/20)",
                );
                crate::lower::raise_fatal_lowering_diagnostic(crate::lower::LowerDiagnostic {
                    message: unsupported.to_string(),
                    span: None,
                    span_id: expr.span_id().map(ToOwned::to_owned),
                    fatal: true,
                });
            });
            let body = kids.get(1).or_else(|| kids.first());
            if let Some(body) = body {
                match effect_kind {
                    EffectKind::Random => {
                        if let Some(seed_expr) = kids.first() {
                            let seed = lower_host_expr(seed_expr, program, scope, tensor_helpers);
                            let body = lower_host_expr(body, program, scope, tensor_helpers);
                            let ty = host_expr_type(&body);
                            return HostExpr::new(HostExprKind::WithSeed {
                                seed: Box::new(seed),
                                body: Box::new(body),
                                ty,
                            });
                        }
                        lower_host_expr(body, program, scope, tensor_helpers)
                    }
                    EffectKind::Resource => lower_host_expr(body, program, scope, tensor_helpers),
                }
            } else {
                HostExpr::new(HostExprKind::Unit)
            }
        }
        Expr::MetaExpr(meta, _) => lower_host_expr(&meta.expr, program, scope, tensor_helpers),
        Expr::List(list, _) if matches!(tag(list), Some("grad" | "vmap" | "vmap-grad")) => {
            // Higher-order differentiation/vmap expressions aren't representable
            // as host-lane values. When one appears in host position (e.g.
            // `g = grad(f)` bound to a local), lowering silently fell through
            // to `HostExpr::new(HostExprKind::Unit)`, which later produced `int g = 0` and a
            // no-op `/* unsupported builtin g */` in the emitted C — a silent
            // wrong-answer. Produce a recognizable marker Builtin instead so
            // `host_program_unresolved_call_sites` can surface it as a clean
            // pre-codegen error.
            let tag_name = tag(list).unwrap_or("grad").to_string();
            HostExpr::new(HostExprKind::Builtin {
                name: format!("__unresolved_{tag_name}"),
                args: Vec::new(),
                ty: expr_host_type(expr, program, scope),
            })
        }
        Expr::List(list, _) if tag(list) == Some("copy") => {
            // `(copy {} x)` exists for linearity bookkeeping. In the host
            // lane, lower as a tagged Builtin whose C emission is a direct
            // tensor-copy or a pass-through for non-tensors. Without this
            // arm the `copy` tag silently fell through to `HostExpr::new(HostExprKind::Unit)`,
            // which caused tensor-if bodies in folds to collapse to
            // `int new_t; new_t = 0;` (Nautilus P2 tensor-if-in-fold).
            let inner = children(list)
                .first()
                .map(|child| lower_host_expr(child, program, scope, tensor_helpers))
                .unwrap_or(HostExpr::new(HostExprKind::Unit));
            let inner_ty = host_expr_type(&inner);
            let explicit = expr_host_type(expr, program, scope);
            let ty = if explicit.is_unresolved() {
                inner_ty
            } else {
                explicit
            };
            HostExpr::new(HostExprKind::Builtin {
                name: "copy".to_string(),
                args: vec![inner],
                ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some("realize") => {
            // Passthrough for phase-0 semantics: realize is an identity in
            // host lane.
            children(list)
                .first()
                .map(|child| lower_host_expr(child, program, scope, tensor_helpers))
                .unwrap_or(HostExpr::new(HostExprKind::Unit))
        }
        Expr::List(list, _) if tag(list) == Some("jit") => {
            // `spec/03-deep-syntax.md` §2.7: jit is a compilation trigger
            // and a semantic no-op at evaluation. Host-lane lowering
            // pass-through to the inner expression, mirroring `lower_jit`
            // in `crates/chelis-ir/src/lower.rs` and the IR DAG behavior.
            //
            // Without this arm jit fell through to `HostExpr::Unit`, so a
            // binding like `result = jit(to_tensor([...]))` emitted
            // `int result = 0; printf("()\n");` — silent data loss
            // (Finding 1 of red-team PR #51).
            children(list)
                .first()
                .map(|child| lower_host_expr(child, program, scope, tensor_helpers))
                .unwrap_or(HostExpr::new(HostExprKind::Unit))
        }
        Expr::List(list, _) if tag(list) == Some("par") => {
            // `spec/03-deep-syntax.md` §2.3: par v1 is sequential
            // composition. Lower the children in order and bind the value
            // of the last child as the par's value, mirroring `lower_par`
            // in `crates/chelis-ir/src/lower.rs`. We do not currently
            // thread intermediate children through a sequence node; if
            // they have side effects (e.g. `print`, `realize`), those
            // primitives have their own host-lane arms and the emitted C
            // will reach them through whatever scope the par appears in.
            // A future change can introduce a HostExpr::Sequence kind if
            // par needs to preserve non-IO side effects across children.
            //
            // Without this arm par fell through to `HostExpr::Unit`, so
            // `result = par {..; to_tensor(..)}` emitted
            // `int result = 0; printf("()\n");` (Finding 2 of red-team
            // PR #51).
            let kids = children(list);
            let mut last: Option<HostExpr> = None;
            for child in kids {
                last = Some(lower_host_expr(child, program, scope, tensor_helpers));
            }
            last.unwrap_or(HostExpr::new(HostExprKind::Unit))
        }
        _ => HostExpr::new(HostExprKind::Unit),
    }
}

fn refine_function_params_from_body(params: &mut [HostParam], body: &HostExpr) {
    let HostExprKind::Call { args, arg_tys, .. } = &body.kind else {
        return;
    };
    for (index, param) in params.iter_mut().enumerate() {
        if !param.ty.is_unresolved() {
            continue;
        }
        let Some(HostExpr {
            kind: HostExprKind::Var(name, _),
            ..
        }) = args.get(index)
        else {
            continue;
        };
        if name != &param.name {
            continue;
        }
        let Some(inferred) = arg_tys.get(index) else {
            continue;
        };
        if !inferred.is_unresolved() {
            param.ty = inferred.clone();
        }
    }
}

fn refine_host_function_signatures(functions: &mut [HostFunction]) -> bool {
    let mut any_changed = false;
    loop {
        let mut changed = false;
        let signatures = functions
            .iter()
            .map(|function| {
                (
                    function.name.clone(),
                    (
                        function
                            .params
                            .iter()
                            .map(|param| param.ty.clone())
                            .collect::<Vec<_>>(),
                        function.ret_ty.clone(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();

        for function in functions.iter_mut() {
            let inferred_param_fns = infer_callable_param_types(&function.params, &function.body);
            for param in function.params.iter_mut() {
                if param.ty.is_unresolved()
                    && let Some(inferred) = inferred_param_fns.get(&param.name)
                    && !host_type_is_unresolved(inferred)
                    && param.ty != *inferred
                {
                    param.ty = inferred.clone();
                    changed = true;
                }
            }

            let mut scope = function
                .params
                .iter()
                .map(|param| (param.name.clone(), param.ty.clone()))
                .collect::<HashMap<_, _>>();
            if refine_host_expr_types(&mut function.body, &mut scope, &signatures) {
                changed = true;
            }
            if function.ret_ty.is_unresolved() {
                let body_ty = host_expr_type(&function.body);
                if !body_ty.is_unresolved() {
                    function.ret_ty = body_ty;
                    changed = true;
                }
            }

            let HostExprKind::Call {
                function: callee,
                args,
                arg_tys,
                ty,
            } = &mut function.body.kind
            else {
                continue;
            };
            let Some((callee_params, callee_ret)) = signatures.get(callee) else {
                continue;
            };

            if function.ret_ty.is_unresolved() && !callee_ret.is_unresolved() {
                function.ret_ty = callee_ret.clone();
                if ty.is_unresolved() {
                    *ty = callee_ret.clone();
                }
                changed = true;
            }

            for (index, param) in function.params.iter_mut().enumerate() {
                if !param.ty.is_unresolved() {
                    continue;
                }
                let Some(HostExpr {
                    kind: HostExprKind::Var(name, arg_ty),
                    ..
                }) = args.get_mut(index)
                else {
                    continue;
                };
                if name != &param.name {
                    continue;
                }
                let Some(inferred) = callee_params.get(index) else {
                    continue;
                };
                if inferred.is_unresolved() {
                    continue;
                }
                param.ty = inferred.clone();
                *arg_ty = inferred.clone();
                if let Some(call_arg_ty) = arg_tys.get_mut(index) {
                    *call_arg_ty = inferred.clone();
                }
                changed = true;
            }
        }

        if !changed {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn refine_host_globals(globals: &mut [HostBinding], functions: &[HostFunction]) -> bool {
    let mut any_changed = false;
    loop {
        let signatures = functions
            .iter()
            .map(|function| {
                (
                    function.name.clone(),
                    (
                        function
                            .params
                            .iter()
                            .map(|param| param.ty.clone())
                            .collect::<Vec<_>>(),
                        function.ret_ty.clone(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();

        let mut changed = false;
        let mut scope = HashMap::new();
        for binding in globals.iter_mut() {
            if refine_host_expr_types(&mut binding.value, &mut scope, &signatures) {
                changed = true;
            }
            let inferred = host_expr_type(&binding.value);
            if binding.ty != inferred && !inferred.is_unresolved() {
                binding.ty = inferred.clone();
                changed = true;
            }
            scope.insert(binding.name.clone(), binding.ty.clone());
        }

        if !changed {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn propagate_named_callback_signatures(
    functions: &mut [HostFunction],
    globals: &[HostBinding],
) -> bool {
    let mut inferred = HashMap::<String, (Vec<HostTypeTerm>, HostTypeTerm)>::new();
    for function in functions.iter() {
        collect_named_callback_signatures(&function.body, &mut inferred);
    }
    for binding in globals {
        collect_named_callback_signatures(&binding.value, &mut inferred);
    }

    let mut changed = false;
    for function in functions.iter_mut() {
        let Some((param_tys, ret_ty)) = inferred.get(&function.name) else {
            continue;
        };
        for (param, inferred_ty) in function.params.iter_mut().zip(param_tys.iter()) {
            if host_type_is_unresolved(&param.ty) && !host_type_is_unresolved(inferred_ty) {
                param.ty = inferred_ty.clone();
                changed = true;
            }
        }
        if host_type_is_unresolved(&function.ret_ty) && !host_type_is_unresolved(ret_ty) {
            function.ret_ty = ret_ty.clone();
            changed = true;
        }
    }
    changed
}

fn collect_named_callback_signatures(
    expr: &HostExpr,
    out: &mut HashMap<String, (Vec<HostTypeTerm>, HostTypeTerm)>,
) {
    match &expr.kind {
        HostExprKind::Call { args, .. } | HostExprKind::Builtin { args, .. } => {
            for arg in args {
                collect_named_callback_signatures(arg, out);
            }
        }
        HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
            for item in items {
                collect_named_callback_signatures(item, out);
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields {
                collect_named_callback_signatures(field, out);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => {
            collect_named_callback_signatures(base, out);
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            collect_named_callback_signatures(cond, out);
            collect_named_callback_signatures(then_expr, out);
            collect_named_callback_signatures(else_expr, out);
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            collect_named_callback_signatures(scrutinee, out);
            collect_named_callback_signatures(some_expr, out);
            collect_named_callback_signatures(none_expr, out);
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            collect_named_callback_signatures(scrutinee, out);
            for arm in arms {
                collect_named_callback_signatures(&arm.expr, out);
            }
            if let Some(default_expr) = default_expr {
                collect_named_callback_signatures(default_expr, out);
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                collect_named_callback_signatures(&binding.value, out);
            }
            collect_named_callback_signatures(body, out);
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            merge_named_callback_signature(callback, out);
            collect_named_callback_signatures_in_callback(callback, out);
            collect_named_callback_signatures(list, out);
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            merge_named_callback_signature(callback, out);
            collect_named_callback_signatures_in_callback(callback, out);
            collect_named_callback_signatures(init, out);
            collect_named_callback_signatures(list, out);
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args {
                collect_named_callback_signatures(arg, out);
            }
        }
        HostExprKind::WithSeed { seed, body, .. } => {
            collect_named_callback_signatures(seed, out);
            collect_named_callback_signatures(body, out);
        }
        HostExprKind::Var(_, _)
        | HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
    }
}

fn collect_named_callback_signatures_in_callback(
    callback: &HostCallback,
    out: &mut HashMap<String, (Vec<HostTypeTerm>, HostTypeTerm)>,
) {
    if let HostCallbackKind::Inline { body, .. } = &callback.kind {
        collect_named_callback_signatures(body, out);
    }
}

fn merge_named_callback_signature(
    callback: &HostCallback,
    out: &mut HashMap<String, (Vec<HostTypeTerm>, HostTypeTerm)>,
) {
    let HostCallbackKind::Named { function, params } = &callback.kind else {
        return;
    };
    let entry = out.entry(function.clone()).or_insert_with(|| {
        (
            std::iter::repeat_with(fresh_host_inference)
                .take(params.len())
                .collect(),
            fresh_host_inference(),
        )
    });
    if entry.0.len() < params.len() {
        entry.0.extend(
            std::iter::repeat_with(fresh_host_inference).take(params.len() - entry.0.len()),
        );
    }
    for (index, param) in params.iter().enumerate() {
        if entry.0[index].is_unresolved() && !param.ty.is_unresolved() {
            entry.0[index] = param.ty.clone();
        }
    }
    if entry.1.is_unresolved() && !callback.ret_ty.is_unresolved() {
        entry.1 = callback.ret_ty.clone();
    }
}

fn infer_callable_param_types(
    params: &[HostParam],
    body: &HostExpr,
) -> HashMap<String, HostTypeTerm> {
    let unknown = params
        .iter()
        .filter(|param| param.ty.is_unresolved())
        .map(|param| param.name.clone())
        .collect::<HashSet<_>>();
    let mut out = HashMap::new();
    infer_callable_param_types_in_expr(body, &unknown, &mut out);
    out
}

fn infer_callable_param_types_in_expr(
    expr: &HostExpr,
    unknown: &HashSet<String>,
    out: &mut HashMap<String, HostTypeTerm>,
) {
    match &expr.kind {
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ..
        } => {
            if unknown.contains(function) {
                out.entry(function.clone()).or_insert_with(|| {
                    HostTypeTerm::Fn(arg_tys.clone(), Box::new(HostTypeTerm::Tuple(Vec::new())))
                });
            }
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
            for item in items {
                infer_callable_param_types_in_expr(item, unknown, out);
            }
        }
        HostExprKind::Builtin { args, .. } => {
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields {
                infer_callable_param_types_in_expr(field, unknown, out);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => {
            infer_callable_param_types_in_expr(base, unknown, out);
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(cond, unknown, out);
            infer_callable_param_types_in_expr(then_expr, unknown, out);
            infer_callable_param_types_in_expr(else_expr, unknown, out);
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(scrutinee, unknown, out);
            infer_callable_param_types_in_expr(some_expr, unknown, out);
            infer_callable_param_types_in_expr(none_expr, unknown, out);
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(scrutinee, unknown, out);
            for arm in arms {
                infer_callable_param_types_in_expr(&arm.expr, unknown, out);
            }
            if let Some(default_expr) = default_expr {
                infer_callable_param_types_in_expr(default_expr, unknown, out);
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                infer_callable_param_types_in_expr(&binding.value, unknown, out);
            }
            infer_callable_param_types_in_expr(body, unknown, out);
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            infer_callable_param_types_in_callback(callback, unknown, out);
            infer_callable_param_types_in_expr(list, unknown, out);
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            infer_callable_param_types_in_callback(callback, unknown, out);
            infer_callable_param_types_in_expr(init, unknown, out);
            infer_callable_param_types_in_expr(list, unknown, out);
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExprKind::WithSeed { seed, body, .. } => {
            infer_callable_param_types_in_expr(seed, unknown, out);
            infer_callable_param_types_in_expr(body, unknown, out);
        }
        HostExprKind::Var(_, _)
        | HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
    }
}

fn infer_callable_param_types_in_callback(
    callback: &HostCallback,
    unknown: &HashSet<String>,
    out: &mut HashMap<String, HostTypeTerm>,
) {
    if let HostCallbackKind::Inline { body, .. } = &callback.kind {
        infer_callable_param_types_in_expr(body, unknown, out);
    }
}

fn refine_host_expr_types(
    expr: &mut HostExpr,
    scope: &mut HashMap<String, HostTypeTerm>,
    signatures: &HashMap<String, (Vec<HostTypeTerm>, HostTypeTerm)>,
) -> bool {
    let mut changed = false;
    match &mut expr.kind {
        HostExprKind::Var(name, ty) => {
            if host_type_is_unresolved(ty)
                && let Some(inferred) = scope.get(name)
                && !host_type_is_unresolved(inferred)
            {
                *ty = inferred.clone();
                changed = true;
            }
        }
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
            let inferred_sig = scope.get(function).and_then(|ty| match ty {
                HostTypeTerm::Fn(params, ret) => Some((params.clone(), (**ret).clone())),
                _ => None,
            });
            let declared_sig = signatures.get(function).cloned();
            let sig = inferred_sig.or(declared_sig);
            if let Some((params, ret)) = sig {
                for (index, arg_ty) in arg_tys.iter_mut().enumerate() {
                    if let Some(inferred) = params.get(index)
                        && !inferred.is_unresolved()
                        && *arg_ty != *inferred
                    {
                        *arg_ty = inferred.clone();
                        changed = true;
                    }
                }
                if ty.is_unresolved() && !ret.is_unresolved() {
                    *ty = ret;
                    changed = true;
                }
            }
        }
        HostExprKind::Builtin { name, args, ty } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
            // An empty list remains genuinely underconstrained on its own,
            // but `to_tensor([])` is a language-level empty tensor literal:
            // its established checked/runtime representation is rank-1 f32.
            // Resolve that context here, rather than teaching the generic
            // empty-list state to manufacture an element type.
            if name == "to_tensor"
                && let Some(HostExpr {
                    kind: HostExprKind::List(items, list_ty),
                    ..
                }) = args.first_mut()
                && items.is_empty()
                && list_ty.is_unresolved()
            {
                *list_ty = HostTypeTerm::List(Box::new(HostTypeTerm::Float32));
                changed = true;
            }
            // RT-4 F1: only override `ty` when the current value has
            // unresolved type variables. Previously this clobbered any
            // declared-type retag (e.g. a global binding annotated as
            // `tensor[3, f64]` would be overwritten back to the
            // builtin's default `tensor[list, f32]` inferred return
            // type, defeating the F1 fix's typed runtime dispatch).
            if host_type_is_unresolved(ty)
                && let Some(inferred) = infer_builtin_host_type(name, args)
                && !host_type_is_unresolved(&inferred)
                && *ty != inferred
            {
                *ty = inferred;
                changed = true;
            }
        }
        HostExprKind::List(items, ty) => {
            for item in items.iter_mut() {
                changed |= refine_host_expr_types(item, scope, signatures);
            }
            let item_ty = items
                .iter()
                .map(host_expr_type)
                .find(|item_ty| !host_type_is_unresolved(item_ty))
                .unwrap_or_else(fresh_host_inference);
            let inferred = HostTypeTerm::List(Box::new(item_ty));
            if host_type_is_unresolved(ty) && !host_type_is_unresolved(&inferred) {
                *ty = inferred;
                changed = true;
            }
        }
        HostExprKind::Tuple(items, ty) => {
            for item in items.iter_mut() {
                changed |= refine_host_expr_types(item, scope, signatures);
            }
            let inferred = HostTypeTerm::Tuple(items.iter().map(host_expr_type).collect());
            if host_type_is_unresolved(ty) && !host_type_is_unresolved(&inferred) {
                *ty = inferred;
                changed = true;
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields.iter_mut() {
                changed |= refine_host_expr_types(field, scope, signatures);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => {
            changed |= refine_host_expr_types(base, scope, signatures);
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(cond, scope, signatures);
            changed |= refine_host_expr_types(then_expr, scope, signatures);
            changed |= refine_host_expr_types(else_expr, scope, signatures);
            if ty.is_unresolved() {
                let then_ty = host_expr_type(then_expr);
                let else_ty = host_expr_type(else_expr);
                let inferred = if !then_ty.is_unresolved() {
                    then_ty
                } else {
                    else_ty
                };
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(scrutinee, scope, signatures);
            let mut some_scope = scope.clone();
            let inner_ty = option_inner_type(scrutinee);
            if !inner_ty.is_unresolved() {
                some_scope.insert(bind_name.clone(), inner_ty);
            }
            changed |= refine_host_expr_types(some_expr, &mut some_scope, signatures);
            changed |= refine_host_expr_types(none_expr, scope, signatures);
            if ty.is_unresolved() {
                let some_ty = host_expr_type(some_expr);
                let none_ty = host_expr_type(none_expr);
                let inferred = if !some_ty.is_unresolved() {
                    some_ty
                } else {
                    none_ty
                };
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(scrutinee, scope, signatures);
            for arm in arms.iter_mut() {
                let mut arm_scope = scope.clone();
                for binding in &arm.bindings {
                    arm_scope.insert(binding.name.clone(), binding.ty.clone());
                }
                changed |= refine_host_expr_types(&mut arm.expr, &mut arm_scope, signatures);
            }
            if let Some(default_expr) = default_expr {
                changed |= refine_host_expr_types(default_expr, scope, signatures);
            }
            if ty.is_unresolved() {
                let inferred = arms
                    .iter()
                    .map(|arm| host_expr_type(&arm.expr))
                    .find(|ty| !ty.is_unresolved())
                    .or_else(|| default_expr.as_ref().map(|expr| host_expr_type(expr)));
                if let Some(inferred) = inferred
                    && !inferred.is_unresolved()
                {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Let { bindings, body, ty } => {
            let mut local_scope = scope.clone();
            for binding in bindings.iter_mut() {
                changed |= refine_host_expr_types(&mut binding.value, &mut local_scope, signatures);
                if binding.ty.is_unresolved() {
                    let inferred = host_expr_type(&binding.value);
                    if !inferred.is_unresolved() {
                        binding.ty = inferred.clone();
                        changed = true;
                    }
                }
                local_scope.insert(binding.name.clone(), binding.ty.clone());
            }
            changed |= refine_host_expr_types(body, &mut local_scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(body);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Map { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            let expected_ret = list_item_type(ty);
            changed |= specialize_host_callback_types(callback, &[item_ty], &expected_ret);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(list);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Filter { callback, list, ty }
        | HostExprKind::Partition { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            changed |= specialize_host_callback_types(callback, &[item_ty], &HostTypeTerm::Bool);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(list);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::FlatMap { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            let expected_ret = match ty {
                HostTypeTerm::List(inner) => HostTypeTerm::List(Box::new((**inner).clone())),
                _ => fresh_host_inference(),
            };
            changed |= specialize_host_callback_types(callback, &[item_ty], &expected_ret);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(list);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        } => {
            changed |= refine_host_expr_types(init, scope, signatures);
            changed |= refine_host_expr_types(list, scope, signatures);
            let init_ty = host_expr_type(init);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            changed |=
                specialize_host_callback_types(callback, &[init_ty.clone(), item_ty], &init_ty);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(init);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
        }
        HostExprKind::WithSeed { seed, body, ty } => {
            changed |= refine_host_expr_types(seed, scope, signatures);
            changed |= refine_host_expr_types(body, scope, signatures);
            if ty.is_unresolved() {
                let inferred = host_expr_type(body);
                if !inferred.is_unresolved() {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
    }
    changed
}

fn list_item_type(ty: &HostTypeTerm) -> HostTypeTerm {
    match ty {
        HostTypeTerm::List(inner) => (**inner).clone(),
        _ => fresh_host_inference(),
    }
}

fn callback_params_mut(callback: &mut HostCallback) -> &mut [HostParam] {
    match &mut callback.kind {
        HostCallbackKind::Named { params, .. } | HostCallbackKind::Inline { params, .. } => params,
    }
}

fn specialize_host_callback_types(
    callback: &mut HostCallback,
    param_tys: &[HostTypeTerm],
    ret_ty: &HostTypeTerm,
) -> bool {
    let mut changed = false;
    for (param, inferred) in callback_params_mut(callback)
        .iter_mut()
        .zip(param_tys.iter())
    {
        if host_type_is_unresolved(&param.ty) && !host_type_is_unresolved(inferred) {
            param.ty = inferred.clone();
            changed = true;
        }
    }
    if host_type_is_unresolved(&callback.ret_ty) && !host_type_is_unresolved(ret_ty) {
        callback.ret_ty = ret_ty.clone();
        changed = true;
    }
    changed
}

fn refine_host_callback_types(
    callback: &mut HostCallback,
    scope: &mut HashMap<String, HostTypeTerm>,
    signatures: &HashMap<String, (Vec<HostTypeTerm>, HostTypeTerm)>,
) -> bool {
    match &mut callback.kind {
        HostCallbackKind::Inline { params, body } => {
            let mut callback_scope = scope.clone();
            for param in params.iter() {
                callback_scope.insert(param.name.clone(), param.ty.clone());
            }
            let mut changed = refine_host_expr_types(body, &mut callback_scope, signatures);
            let inferred = host_expr_type(body);
            if !host_type_is_unresolved(&inferred) && callback.ret_ty != inferred {
                callback.ret_ty = inferred;
                changed = true;
            }
            changed
        }
        HostCallbackKind::Named { .. } => false,
    }
}

fn should_keep_tensor_expr_in_host_lane(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    if matches!(tag(list), Some("tuple-get" | "if" | "match")) {
        return true;
    }
    if tag(list) != Some("app") {
        return false;
    }
    let Some(callee) = children(list).first().and_then(as_list) else {
        return false;
    };
    if tag(callee) != Some("var") {
        return false;
    }
    let name = children(callee).first().and_then(symbol_name);
    matches!(
        name,
        Some(
            "copy"
                | "reshape"
                | "to_tensor"
                | "scalar_to_tensor"
                | "pad_sequences"
                | "pad_sequences_to"
                | "concat"
                | "split"
                | "scatter"
                | "where"
                | "cumsum"
                | "map"
                | "filter"
                | "fold"
                | "scan"
                | "tensor_scan"
                | "partition"
                | "flat_map"
                | "sort"
                | "diagonal"
                | "trace"
                | "clamp"
                | "einsum"
        )
    )
}

fn lower_match_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let scrutinee = lower_host_expr(&kids[0], program, scope, tensor_helpers);
    let scrutinee_ty = host_expr_type(&scrutinee);
    if matches!(
        scrutinee_ty,
        HostTypeTerm::Int64
            | HostTypeTerm::Float64
            | HostTypeTerm::Float32
            | HostTypeTerm::Bool
            | HostTypeTerm::String
    ) {
        return lower_literal_match_host_expr(
            list,
            program,
            scope,
            tensor_helpers,
            scrutinee,
            scrutinee_ty,
        );
    }
    let mut bind_name = "value".to_string();
    let mut some_expr = HostExpr::new(HostExprKind::Unit);
    let mut none_expr = HostExpr::new(HostExprKind::Unit);
    let mut generic_arms = Vec::new();
    let mut generic_default = None;

    for arm in kids.iter().skip(1) {
        let Some(arm_list) = as_list(arm) else {
            continue;
        };
        if tag(arm_list) != Some("arm") {
            continue;
        }
        let arm_kids = children(arm_list);
        let Some(pattern) = arm_kids.first().and_then(as_list) else {
            continue;
        };
        if tag(pattern) == Some("pat-wild") {
            generic_default = Some(Box::new(lower_host_expr(
                &arm_kids[2],
                program,
                scope,
                tensor_helpers,
            )));
            continue;
        }
        if let Some("pat-ctor" | "pat-record") = tag(pattern) {
            let ctor = children(pattern).first().and_then(symbol_name);
            match ctor {
                Some("Some") => {
                    if let Some(bound) = children(pattern).get(1).and_then(as_list)
                        && tag(bound) == Some("pat-var")
                        && let Some(name) = children(bound).first().and_then(symbol_name)
                    {
                        bind_name = name.to_string();
                    }
                    let mut scoped = scope.clone();
                    scoped.insert(bind_name.clone(), option_inner_type(&scrutinee));
                    some_expr = lower_host_expr(&arm_kids[2], program, &scoped, tensor_helpers);
                }
                Some("None") => {
                    none_expr = lower_host_expr(&arm_kids[2], program, scope, tensor_helpers);
                }
                Some(ctor_name) => {
                    let ctor_fields =
                        lookup_adt_ctor_details_for_type(program, ctor_name, Some(&scrutinee_ty))
                            .map(|(_, fields)| fields)
                            .or_else(|| {
                                program
                                    .type_env()
                                    .get(ctor_name)
                                    .and_then(parse_fn_type_expr)
                                    .map(|(args, _)| {
                                        args.into_iter()
                                            .map(|ty| HostAdtField { name: None, ty })
                                            .collect::<Vec<_>>()
                                    })
                            })
                            .unwrap_or_default();
                    let mut scoped = scope.clone();
                    let mut bindings = Vec::new();
                    for (field_index, subpat, field_ty) in
                        pattern_field_bindings(pattern, &ctor_fields)
                    {
                        let Some(subpat_list) = as_list(subpat) else {
                            continue;
                        };
                        if tag(subpat_list) != Some("pat-var") {
                            continue;
                        }
                        let Some(name) = children(subpat_list).first().and_then(symbol_name) else {
                            continue;
                        };
                        let ty = field_ty
                            .or_else(|| expr_type(subpat))
                            .unwrap_or_else(fresh_host_inference);
                        scoped.insert(name.to_string(), ty.clone());
                        bindings.push(HostPatternBinding {
                            name: name.to_string(),
                            ty,
                            field_index,
                        });
                    }
                    generic_arms.push(HostMatchArm {
                        ctor: ctor_name.to_string(),
                        bindings,
                        expr: lower_host_expr(&arm_kids[2], program, &scoped, tensor_helpers),
                    });
                }
                None => {}
            }
        }
    }

    let ty = {
        let explicit = expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        );
        if explicit.is_unresolved() {
            // For ADT matches with multiple arms, Some/None exprs aren't set —
            // the arm bodies live in `generic_arms` / `generic_default`. Fall
            // back through those first so the match's result type reflects the
            // arms' actual shape. Otherwise Unit propagates and the match
            // target is emitted as `int`, which is the wrong C type for any
            // pointer-valued arm (regression hit by Coral groupby's
            // `next = match agg_spec { ... }` ADT match).
            let arm_ty = generic_arms
                .iter()
                .map(|arm| host_expr_type(&arm.expr))
                .find(|ty| !ty.is_unresolved())
                .or_else(|| {
                    generic_default
                        .as_deref()
                        .map(host_expr_type)
                        .filter(|ty| !ty.is_unresolved())
                });
            if let Some(ty) = arm_ty {
                ty
            } else {
                let some_ty = host_expr_type(&some_expr);
                if some_ty.is_unresolved() {
                    host_expr_type(&none_expr)
                } else {
                    some_ty
                }
            }
        } else {
            explicit
        }
    };

    if matches!(scrutinee_ty, HostTypeTerm::Adt(_, _)) {
        return HostExpr::new(HostExprKind::MatchAdt {
            scrutinee: Box::new(scrutinee),
            arms: generic_arms,
            default_expr: generic_default,
            ty,
        });
    }

    HostExpr::new(HostExprKind::MatchOption {
        scrutinee: Box::new(scrutinee),
        bind_name,
        some_expr: Box::new(some_expr),
        none_expr: Box::new(none_expr),
        ty,
    })
}

fn lower_literal_match_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
    scrutinee: HostExpr,
    scrutinee_ty: HostTypeTerm,
) -> HostExpr {
    let kids = children(list);
    let mut literal_arms = Vec::new();
    let mut default_expr = HostExpr::new(HostExprKind::Unit);
    for arm in kids.iter().skip(1) {
        let Some(arm_list) = as_list(arm) else {
            continue;
        };
        if tag(arm_list) != Some("arm") {
            continue;
        }
        let arm_kids = children(arm_list);
        let Some(pattern) = arm_kids.first().and_then(as_list) else {
            continue;
        };
        let body = lower_host_expr(&arm_kids[2], program, scope, tensor_helpers);
        match tag(pattern) {
            Some("pat-wild") => default_expr = body,
            Some("pat-lit") => {
                if let Some(lit) = children(pattern)
                    .first()
                    .and_then(|expr| host_literal_expr(expr, &scrutinee_ty))
                {
                    literal_arms.push((lit, body));
                }
            }
            _ => {}
        }
    }

    let explicit = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    let mut body = default_expr;
    let result_ty = if explicit.is_unresolved() {
        host_expr_type(&body)
    } else {
        explicit
    };
    for (lit, arm_expr) in literal_arms.into_iter().rev() {
        body = HostExpr::new(HostExprKind::If {
            cond: Box::new(HostExpr::new(HostExprKind::Builtin {
                name: "eq".to_string(),
                args: vec![scrutinee.clone(), lit],
                ty: HostTypeTerm::Bool,
            })),
            then_expr: Box::new(arm_expr),
            else_expr: Box::new(body),
            ty: result_ty.clone(),
        });
    }
    body
}

fn host_literal_expr(expr: &Expr, expected_ty: &HostTypeTerm) -> Option<HostExpr> {
    match (expr, expected_ty) {
        (Expr::Atom(Atom::Int(value), _), &HostTypeTerm::Int64) => {
            Some(HostExpr::new(HostExprKind::Int(*value)))
        }
        (Expr::Atom(Atom::Float(value), _), &HostTypeTerm::Float64) => {
            Some(HostExpr::new(HostExprKind::Float(*value)))
        }
        (Expr::Atom(Atom::Bool(value), _), &HostTypeTerm::Bool) => {
            Some(HostExpr::new(HostExprKind::Bool(*value)))
        }
        (Expr::Atom(Atom::Str(value), _), &HostTypeTerm::String) => {
            Some(HostExpr::new(HostExprKind::String(value.clone())))
        }
        (Expr::Atom(Atom::Int(value), _), _) => Some(HostExpr::new(HostExprKind::Int(*value))),
        (Expr::Atom(Atom::Float(value), _), _) => Some(HostExpr::new(HostExprKind::Float(*value))),
        (Expr::Atom(Atom::Bool(value), _), _) => Some(HostExpr::new(HostExprKind::Bool(*value))),
        (Expr::Atom(Atom::Str(value), _), _) => {
            Some(HostExpr::new(HostExprKind::String(value.clone())))
        }
        _ => None,
    }
}

fn lower_record_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let ctor = kids
        .first()
        .and_then(symbol_name)
        .unwrap_or("_")
        .to_string();
    let ctor_info = lookup_adt_ctor_details(program, &ctor);
    let mut supplied = HashMap::new();
    for field in kids.iter().skip(1) {
        let Some(kv_list) = as_list(field) else {
            continue;
        };
        if tag(kv_list) != Some("kv") {
            continue;
        }
        let kv_kids = children(kv_list);
        let Some(name) = kv_kids.first().and_then(symbol_name) else {
            continue;
        };
        let value = kv_kids
            .get(1)
            .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers))
            .unwrap_or(HostExpr::new(HostExprKind::Unit));
        supplied.insert(name.to_string(), value);
    }
    let fields = ctor_info
        .as_ref()
        .map(|(_, declared)| {
            declared
                .iter()
                .map(|field| {
                    let value = field
                        .name
                        .as_ref()
                        .and_then(|name| supplied.remove(name))
                        .unwrap_or(HostExpr::new(HostExprKind::Unit));
                    force_host_expr_type(value, field.ty.clone())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    HostExpr::new(HostExprKind::AdtConstruct {
        ctor,
        fields,
        ty: if !explicit_ty.is_unresolved() {
            explicit_ty
        } else {
            ctor_info
                .map(|(adt_name, _)| HostTypeTerm::Adt(adt_name, Vec::new()))
                .unwrap_or_else(fresh_host_inference)
        },
    })
}

fn lower_access_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let base = kids
        .first()
        .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers))
        .unwrap_or(HostExpr::new(HostExprKind::Unit));
    let field_name = kids.get(1).and_then(symbol_name).unwrap_or("");
    let (field_index, field_ty) =
        lookup_access_field(program, &base, field_name).unwrap_or((0, fresh_host_inference()));
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    HostExpr::new(HostExprKind::AdtFieldAccess {
        base: Box::new(base),
        field_index,
        ty: if !explicit_ty.is_unresolved() {
            explicit_ty
        } else {
            field_ty
        },
    })
}

fn pattern_field_bindings<'a>(
    pattern: &'a List,
    ctor_fields: &'a [HostAdtField],
) -> Vec<(usize, &'a Expr, Option<HostTypeTerm>)> {
    match tag(pattern) {
        Some("pat-record") => children(pattern)
            .iter()
            .skip(1)
            .filter_map(|kv_expr| {
                let kv_list = as_list(kv_expr)?;
                if tag(kv_list) != Some("kv") {
                    return None;
                }
                let kv_kids = children(kv_list);
                let field_name = kv_kids.first().and_then(symbol_name)?;
                let field_index = ctor_fields
                    .iter()
                    .position(|field| field.name.as_deref() == Some(field_name))?;
                Some((
                    field_index,
                    kv_kids.get(1)?,
                    ctor_fields.get(field_index).map(|field| field.ty.clone()),
                ))
            })
            .collect(),
        _ => children(pattern)
            .iter()
            .skip(1)
            .enumerate()
            .map(|(field_index, subpat)| {
                (
                    field_index,
                    subpat,
                    ctor_fields.get(field_index).map(|field| field.ty.clone()),
                )
            })
            .collect(),
    }
}

fn lower_tuple_get_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let tuple_expr = kids
        .first()
        .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers));
    let index_expr = kids
        .get(1)
        .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers));
    let args = tuple_expr.into_iter().chain(index_expr).collect::<Vec<_>>();
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    let ty = if explicit_ty.is_unresolved() {
        infer_builtin_host_type("tuple-get", &args).unwrap_or_else(fresh_host_inference)
    } else {
        explicit_ty
    };
    HostExpr::new(HostExprKind::Builtin {
        name: "tuple-get".to_string(),
        args,
        ty,
    })
}

// ---------------------------------------------------------------------------
// Host-lane scalar forward-mode AD (chelis#405).
//
// Per `spec/design/phase5_host_scalar_ad.md`, the locked design is
// forward-mode dual numbers. A scalar function `f: f32 -> f32` (or
// multi-scalar-param) that lands in the host lane has no reverse-mode
// transform, so `grad(f, wrt=(p))(args)` previously rejected with the
// `__unresolved_grad` marker. This pass implements the dual transform
// entirely at compile time: it walks `f`'s pure-scalar body and produces
// two parallel HostExpr trees — a value tree and a derivative tree — using
// only the existing host scalar builtins (`add`/`mul`/`sub`/`div`/`neg`/
// `exp`/`log`/`sin`/`cos`/`tanh`/`sqrt`/`pow`/`abs`/`cast`). No new runtime
// struct and no new C builtin are required: the dual "struct" is split into
// two `double`-typed expression trees at lowering time, which is the
// forward-mode dual-number scheme the spec prescribes (one directional
// derivative per pass).
//
// Multi-parameter `wrt=(p1, p2, ...)` emits one derivative tree per
// parameter (each with that parameter's seed = 1.0 and the rest = 0.0) and
// combines them into a host tuple — the gradient tuple.
//
// `wrt` over a host container (list/dict/ADT/tuple) is rejected: this pass
// returns `None`, the caller falls through to the `__unresolved_grad`
// marker, and the existing `cmd_build` guard surfaces the clean diagnostic.
// Tensor-lane reverse-mode AD is untouched: a grad whose differentiated fn
// is tensor-typed is handled by `lower_grad_callable_app` on the DAG path
// and never reaches this host-lane pass.

/// A dual value: the primal value expression and its derivative expression,
/// both ordinary scalar (`Float64`) host expressions.
#[derive(Clone)]
struct Dual {
    value: HostExpr,
    deriv: HostExpr,
}

fn dual_float(value: f64, deriv: f64) -> Dual {
    Dual {
        value: HostExpr::new(HostExprKind::Float(value)),
        deriv: HostExpr::new(HostExprKind::Float(deriv)),
    }
}

fn scalar_builtin(name: &str, args: Vec<HostExpr>) -> HostExpr {
    HostExpr::new(HostExprKind::Builtin {
        name: name.to_string(),
        args,
        ty: HostTypeTerm::Float64,
    })
}

fn host_float(value: f64) -> HostExpr {
    HostExpr::new(HostExprKind::Float(value))
}

/// `true` if a host type is a scalar this pass can differentiate. Integer
/// inputs are accepted (their derivative seed is 0 unless they are the
/// active `wrt`, but `wrt` over an integer is still a directional
/// derivative). Everything else — list/dict/ADT/tuple/tensor/option — is a
/// container and is rejected.
fn is_dual_scalar_type(ty: &HostTypeTerm) -> bool {
    matches!(
        ty,
        &HostTypeTerm::Float64 | &HostTypeTerm::Float32 | &HostTypeTerm::Int64
    )
}

/// Resolve a top-level scalar def by name into `(param_names, param_tys, body)`.
/// Returns `None` if the def is not a `(fn (params ...) body)` form or any
/// parameter is non-scalar.
fn resolve_scalar_def<'a>(
    program: &'a CheckedProgram,
    name: &str,
) -> Option<(Vec<String>, Vec<HostTypeTerm>, &'a Expr)> {
    let mut found: Option<(&'a List,)> = None;
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("def") {
            continue;
        }
        let kids = children(list);
        let Some(def_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if !terminal_name_matches(def_name, name) {
            continue;
        }
        let Some(Expr::List(body_list, _)) = kids.get(1) else {
            continue;
        };
        if tag(body_list) != Some("fn") {
            continue;
        }
        found = Some((body_list,));
        break;
    }
    let (fn_list,) = found?;
    let fn_kids = children(fn_list);
    let params_list = fn_kids.first().and_then(as_list)?;
    if tag(params_list) != Some("params") {
        return None;
    }
    let mut param_names = Vec::new();
    let mut param_tys = Vec::new();
    for param in children(params_list) {
        let pname = param_name(param)?;
        let pty = param_host_type(param)
            .or_else(|| {
                lookup_declared_fn_type(program, name).and_then(|(tys, _)| tys.first().cloned())
            })
            .unwrap_or(HostTypeTerm::Float64);
        param_names.push(pname);
        param_tys.push(pty);
    }
    let body = fn_kids.get(1)?;
    Some((param_names, param_tys, body))
}

/// Try to lower `app(grad(f, wrt=...), arg0, ...)` as a host-lane scalar
/// forward-mode derivative. Returns `Some(host_expr)` on success, `None`
/// when this is not a scalar-grad app this pass handles (tensor lane,
/// container `wrt`, unsupported op, unresolvable callee — all fall through
/// to the existing `__unresolved_grad` rejection path).
fn try_lower_scalar_grad_app(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> Option<HostExpr> {
    let kids = children(list);
    let callee = kids.first().and_then(as_list)?;
    if tag(callee) != Some("grad") {
        return None;
    }
    // The function being differentiated: `(grad {wrt:...} (var f) (lit 0))`.
    let grad_kids = children(callee);
    let fn_var = grad_kids.first().and_then(as_list)?;
    if tag(fn_var) != Some("var") {
        return None;
    }
    let fn_name = children(fn_var).first().and_then(symbol_name)?.to_string();

    let (param_names, param_tys, body) = resolve_scalar_def(program, &fn_name)?;

    // Return type must be scalar; reject (fall through) otherwise. We infer
    // it from the def's declared signature when available.
    if let Some((_, ret_ty)) = lookup_declared_fn_type(program, &fn_name)
        && !is_dual_scalar_type(&ret_ty)
    {
        return None;
    }
    // Every parameter must be a scalar. A container parameter that is not the
    // `wrt` target is still fine to treat as a constant, but the call args
    // would be containers we cannot evaluate in the dual tree, so reject the
    // whole app (the canonical container-AD escalation in the spec).
    if param_tys.iter().any(|ty| !is_dual_scalar_type(ty)) {
        return None;
    }

    // Resolve the `wrt` parameter names from the grad meta. Absent `wrt`
    // means "all parameters" (single-param defs commonly omit it).
    let wrt_names = grad_wrt_param_names(callee, &param_names)?;
    if wrt_names.is_empty() {
        return None;
    }

    // Lower each call argument once into a value HostExpr. Their derivative
    // seed is determined per `wrt` pass below.
    let call_args = &kids[1..];
    if call_args.len() != param_names.len() {
        return None;
    }
    let arg_values: Vec<HostExpr> = call_args
        .iter()
        .map(|arg| lower_host_expr(arg, program, scope, tensor_helpers))
        .collect();

    // One forward pass per `wrt` parameter.
    let mut derivs = Vec::new();
    for wrt_name in &wrt_names {
        let mut env: HashMap<String, Dual> = HashMap::new();
        for (idx, pname) in param_names.iter().enumerate() {
            let seed = if pname == wrt_name { 1.0 } else { 0.0 };
            env.insert(
                pname.clone(),
                Dual {
                    value: arg_values[idx].clone(),
                    deriv: host_float(seed),
                },
            );
        }
        let dual = dual_eval(body, &env, program, 0)?;
        derivs.push(dual.deriv);
    }

    if derivs.len() == 1 {
        Some(derivs.pop().unwrap())
    } else {
        let tys = derivs.iter().map(|_| HostTypeTerm::Float64).collect();
        Some(HostExpr::new(HostExprKind::Tuple(
            derivs,
            HostTypeTerm::Tuple(tys),
        )))
    }
}

/// Read the `wrt` meta off a `grad` list and resolve it to a list of
/// parameter names. `None` is returned when `wrt` references something that
/// is not a parameter name (e.g. a tuple element / field access), which is
/// the container-AD case the spec rejects. Absent `wrt` defaults to all
/// parameters.
fn grad_wrt_param_names(grad_list: &List, param_names: &[String]) -> Option<Vec<String>> {
    let meta = match grad_list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta,
        _ => return Some(param_names.to_vec()),
    };
    let Some((_, wrt_expr)) = meta.entries.iter().find(|(key, _)| key == "wrt") else {
        return Some(param_names.to_vec());
    };
    let mut names = Vec::new();
    collect_wrt_names(wrt_expr, &mut names)?;
    // Each named target must actually be a parameter of the differentiated
    // function. A name that is not a parameter is a container/field access
    // we don't support.
    if names.iter().all(|n| param_names.contains(n)) {
        Some(names)
    } else {
        None
    }
}

fn collect_wrt_names(expr: &Expr, out: &mut Vec<String>) -> Option<()> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => {
            out.push(name.clone());
            Some(())
        }
        Expr::List(list, _) => match tag(list) {
            Some("var") => {
                let name = children(list).first().and_then(symbol_name)?;
                out.push(name.to_string());
                Some(())
            }
            Some("tuple") => {
                for child in children(list) {
                    collect_wrt_names(child, out)?;
                }
                Some(())
            }
            // Field access / index / anything else => container, reject.
            _ => None,
        },
        Expr::MetaExpr(meta, _) => collect_wrt_names(&meta.expr, out),
        _ => None,
    }
}

/// Maximum nesting of inlined user-defined scalar calls and `let` blocks the
/// dual transform will follow. A non-recursive scalar def nests shallowly;
/// the cap exists so a (mutually) recursive scalar callee fails closed —
/// falling through to the `__unresolved_grad` rejection — instead of looping
/// forever or producing an unbounded dual tree.
const MAX_DUAL_INLINE_DEPTH: usize = 64;

/// Forward-mode dual evaluation of a pure-scalar Deep body. Returns `None`
/// for any construct this pass does not support (non-scalar op, unresolved
/// var, control flow) so the caller falls through to the rejection path.
/// `depth` tracks inlined-call / `let` nesting against `MAX_DUAL_INLINE_DEPTH`.
fn dual_eval(
    expr: &Expr,
    env: &HashMap<String, Dual>,
    program: &CheckedProgram,
    depth: usize,
) -> Option<Dual> {
    if depth > MAX_DUAL_INLINE_DEPTH {
        return None;
    }
    match expr {
        Expr::Atom(Atom::Float(v), _) => Some(dual_float(*v, 0.0)),
        Expr::Atom(Atom::Int(v), _) => Some(dual_float(*v as f64, 0.0)),
        Expr::List(list, _) => match tag(list) {
            Some("lit") => {
                let inner = children(list).first()?;
                dual_eval(inner, env, program, depth)
            }
            Some("var") => {
                let name = children(list).first().and_then(symbol_name)?;
                let dual = env.get(name)?;
                Some(Dual {
                    value: dual.value.clone(),
                    deriv: dual.deriv.clone(),
                })
            }
            Some("app") => dual_eval_app(list, env, program, depth),
            // `(let (bind n0 v0 n1 v1 ...) body)`: forward-mode through a
            // block body. Each binding's value is dual-evaluated in the
            // environment built so far (sequential scoping — a later binding
            // may reference an earlier one), then added to a cloned
            // environment under which the body is evaluated. The value and
            // derivative trees are substituted at each use site rather than
            // bound to host-let variables; this is correct because the dual
            // trees are pure `Float64` arithmetic with no side effects. The
            // canonical scalar-AD shapes (single-variable derivatives,
            // Black-Scholes Greeks) reuse each intermediate a small number of
            // times, so the substituted trees stay small.
            Some("let") => dual_eval_let(list, env, program, depth),
            _ => None,
        },
        Expr::MetaExpr(meta, _) => dual_eval(&meta.expr, env, program, depth),
        _ => None,
    }
}

/// Forward-mode dual evaluation of a `(let (bind ...) body)` block. Returns
/// `None` if the binding structure is unexpected or any bound value / the
/// body contains a construct `dual_eval` does not support.
fn dual_eval_let(
    list: &List,
    env: &HashMap<String, Dual>,
    program: &CheckedProgram,
    depth: usize,
) -> Option<Dual> {
    let kids = children(list);
    let bind_list = kids.first().and_then(as_list)?;
    if tag(bind_list) != Some("bind") {
        return None;
    }
    let body = kids.get(1)?;
    let bind_kids = children(bind_list);
    // Bindings are alternating `name value` pairs; an odd count is malformed.
    if !bind_kids.len().is_multiple_of(2) {
        return None;
    }
    let mut local_env = env.clone();
    for pair in bind_kids.chunks_exact(2) {
        let name = symbol_name(&pair[0])?;
        let dual = dual_eval(&pair[1], &local_env, program, depth + 1)?;
        local_env.insert(name.to_string(), dual);
    }
    dual_eval(body, &local_env, program, depth + 1)
}

fn dual_eval_app(
    list: &List,
    env: &HashMap<String, Dual>,
    program: &CheckedProgram,
    depth: usize,
) -> Option<Dual> {
    let kids = children(list);
    let callee = kids.first().and_then(as_list)?;
    if tag(callee) != Some("var") {
        return None;
    }
    let op = children(callee).first().and_then(symbol_name)?;
    let arg_exprs = &kids[1..];
    let mut args: Vec<Dual> = Vec::new();
    for a in arg_exprs {
        args.push(dual_eval(a, env, program, depth)?);
    }

    // Helper closures over scalar builtins.
    let v = |d: &Dual| d.value.clone();
    let dv = |d: &Dual| d.deriv.clone();

    match (op, args.len()) {
        ("add", 2) => Some(Dual {
            value: scalar_builtin("add", vec![v(&args[0]), v(&args[1])]),
            deriv: scalar_builtin("add", vec![dv(&args[0]), dv(&args[1])]),
        }),
        ("sub", 2) => Some(Dual {
            value: scalar_builtin("sub", vec![v(&args[0]), v(&args[1])]),
            deriv: scalar_builtin("sub", vec![dv(&args[0]), dv(&args[1])]),
        }),
        ("mul", 2) => {
            // (uv)' = u'v + uv'
            let lhs = scalar_builtin("mul", vec![dv(&args[0]), v(&args[1])]);
            let rhs = scalar_builtin("mul", vec![v(&args[0]), dv(&args[1])]);
            Some(Dual {
                value: scalar_builtin("mul", vec![v(&args[0]), v(&args[1])]),
                deriv: scalar_builtin("add", vec![lhs, rhs]),
            })
        }
        ("div", 2) => {
            // (u/v)' = (u'v - uv') / v^2
            let num_l = scalar_builtin("mul", vec![dv(&args[0]), v(&args[1])]);
            let num_r = scalar_builtin("mul", vec![v(&args[0]), dv(&args[1])]);
            let num = scalar_builtin("sub", vec![num_l, num_r]);
            let den = scalar_builtin("mul", vec![v(&args[1]), v(&args[1])]);
            Some(Dual {
                value: scalar_builtin("div", vec![v(&args[0]), v(&args[1])]),
                deriv: scalar_builtin("div", vec![num, den]),
            })
        }
        ("neg", 1) => Some(Dual {
            value: scalar_builtin("neg", vec![v(&args[0])]),
            deriv: scalar_builtin("neg", vec![dv(&args[0])]),
        }),
        ("exp", 1) => {
            // (e^u)' = e^u * u'
            let value = scalar_builtin("exp", vec![v(&args[0])]);
            Some(Dual {
                deriv: scalar_builtin("mul", vec![value.clone(), dv(&args[0])]),
                value,
            })
        }
        ("log", 1) => {
            // (ln u)' = u' / u
            Some(Dual {
                value: scalar_builtin("log", vec![v(&args[0])]),
                deriv: scalar_builtin("div", vec![dv(&args[0]), v(&args[0])]),
            })
        }
        ("sin", 1) => {
            // (sin u)' = cos(u) * u'
            let cos = scalar_builtin("cos", vec![v(&args[0])]);
            Some(Dual {
                value: scalar_builtin("sin", vec![v(&args[0])]),
                deriv: scalar_builtin("mul", vec![cos, dv(&args[0])]),
            })
        }
        ("cos", 1) => {
            // (cos u)' = -sin(u) * u'
            let sin = scalar_builtin("sin", vec![v(&args[0])]);
            let neg_sin = scalar_builtin("neg", vec![sin]);
            Some(Dual {
                value: scalar_builtin("cos", vec![v(&args[0])]),
                deriv: scalar_builtin("mul", vec![neg_sin, dv(&args[0])]),
            })
        }
        ("tanh", 1) => {
            // (tanh u)' = (1 - tanh(u)^2) * u'
            let t = scalar_builtin("tanh", vec![v(&args[0])]);
            let t2 = scalar_builtin("mul", vec![t.clone(), t.clone()]);
            let one_minus = scalar_builtin("sub", vec![host_float(1.0), t2]);
            Some(Dual {
                value: t,
                deriv: scalar_builtin("mul", vec![one_minus, dv(&args[0])]),
            })
        }
        ("sqrt", 1) => {
            // (sqrt u)' = u' / (2 sqrt(u))
            let s = scalar_builtin("sqrt", vec![v(&args[0])]);
            let den = scalar_builtin("mul", vec![host_float(2.0), s.clone()]);
            Some(Dual {
                value: s,
                deriv: scalar_builtin("div", vec![dv(&args[0]), den]),
            })
        }
        ("pow", 2) => {
            // Only constant exponents are supported in forward mode here:
            // (u^c)' = c * u^(c-1) * u'. A non-constant exponent (`deriv`
            // not identically zero) needs the general
            // u^v * (v' ln u + v u'/u) form; reject to stay correct.
            let exponent = float_const(&args[1].value)?;
            if !is_zero_float(&args[1].deriv) {
                return None;
            }
            let pow_inner = scalar_builtin("pow", vec![v(&args[0]), host_float(exponent - 1.0)]);
            let coeff = scalar_builtin("mul", vec![host_float(exponent), pow_inner]);
            Some(Dual {
                value: scalar_builtin("pow", vec![v(&args[0]), v(&args[1])]),
                deriv: scalar_builtin("mul", vec![coeff, dv(&args[0])]),
            })
        }
        // `cast` between scalar precisions is value-preserving for the dual
        // tree (host scalars are all `double`); the derivative passes
        // through unchanged.
        ("cast", _) if !args.is_empty() => Some(Dual {
            value: v(&args[0]),
            deriv: dv(&args[0]),
        }),
        // A call to a user-defined scalar def (`d1(...)`, `normal_cdf(...)`):
        // inline the callee's body into the dual tree. The callee must be a
        // top-level scalar def with scalar parameters; its body is
        // dual-evaluated in a fresh environment binding each parameter to the
        // corresponding already-computed dual argument (the chain rule is
        // carried by the argument derivatives). `resolve_scalar_def` rejects
        // non-scalar parameters, and `dual_eval` rejects any body construct
        // this pass does not support, so an unsupported callee falls through
        // to `None` (the `__unresolved_grad` rejection path).
        _ => dual_eval_user_call(op, &args, program, depth),
    }
}

/// Inline a call to a user-defined scalar def into the dual tree. Returns
/// `None` when the callee is not a resolvable scalar def, its arity does not
/// match, or its body uses an unsupported construct.
fn dual_eval_user_call(
    op: &str,
    args: &[Dual],
    program: &CheckedProgram,
    depth: usize,
) -> Option<Dual> {
    let (param_names, param_tys, body) = resolve_scalar_def(program, op)?;
    if param_names.len() != args.len() {
        return None;
    }
    if param_tys.iter().any(|ty| !is_dual_scalar_type(ty)) {
        return None;
    }
    let mut call_env: HashMap<String, Dual> = HashMap::new();
    for (name, arg) in param_names.iter().zip(args.iter()) {
        call_env.insert(name.clone(), arg.clone());
    }
    dual_eval(body, &call_env, program, depth + 1)
}

/// Extract a compile-time float constant from a HostExpr if it is a literal.
fn float_const(expr: &HostExpr) -> Option<f64> {
    match &expr.kind {
        HostExprKind::Float(v) => Some(*v),
        HostExprKind::Int(v) => Some(*v as f64),
        _ => None,
    }
}

fn is_zero_float(expr: &HostExpr) -> bool {
    matches!(&expr.kind, HostExprKind::Float(v) if *v == 0.0)
        || matches!(&expr.kind, HostExprKind::Int(0))
}

fn lower_app_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    // chelis#405: host-lane scalar forward-mode AD. When the callee is a
    // `grad(...)` form differentiating a scalar `f32 -> f32` (or
    // multi-scalar-param) top-level def, emit the dual-propagated derivative
    // directly. A `None` return falls through to the generic path, which
    // produces the `__unresolved_grad` marker for the `cmd_build` guard to
    // reject (container `wrt`, tensor-lane grad, unsupported op).
    if let Some(grad_lowered) = try_lower_scalar_grad_app(list, program, scope, tensor_helpers) {
        return grad_lowered;
    }
    let app_expr = Expr::List(list.clone(), chelis_deep::Span::new(0, 0));
    let kids = children(list);
    let name = kids
        .first()
        .and_then(as_list)
        .and_then(|inner| {
            if tag(inner) == Some("var") {
                children(inner).first().and_then(symbol_name)
            } else {
                None
            }
        })
        .unwrap_or("call")
        .to_string();
    let fn_sig = scope
        .get(&name)
        .and_then(host_fn_signature)
        .or_else(|| lookup_declared_fn_type(program, &name))
        .or_else(|| kids.first().and_then(expr_fn_type));
    let explicit_ty = expr_host_type(&app_expr, program, scope);
    let ctor_info = lookup_adt_ctor(program, &name);
    let inferred_ret_ty = fn_sig
        .as_ref()
        .map(|(_, ret_ty)| ret_ty.clone())
        .unwrap_or_else(fresh_host_inference);
    if name == "Cons" && kids.len() == 3 {
        let expr = Expr::List(list.clone(), chelis_deep::Span::new(0, 0));
        if let Some(items) = lower_list_literal_items(&expr, program, scope, tensor_helpers) {
            let ty = expr_host_type(&expr, program, scope);
            let ty = if ty.is_unresolved() {
                HostTypeTerm::List(Box::new(
                    items
                        .first()
                        .map(host_expr_type)
                        .unwrap_or_else(fresh_host_inference),
                ))
            } else {
                ty
            };
            return HostExpr::new(HostExprKind::List(items, ty));
        }
    }
    if name == "Some" && kids.len() == 2 {
        let arg = lower_host_expr(&kids[1], program, scope, tensor_helpers);
        return HostExpr::new(HostExprKind::Builtin {
            name,
            args: vec![arg.clone()],
            ty: if !explicit_ty.is_unresolved() {
                explicit_ty
            } else {
                HostTypeTerm::Option(Box::new(host_expr_type(&arg)))
            },
        });
    }
    if name == "map"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            HostTypeTerm::List(Box::new(callback.ret_ty.clone()))
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::Map {
            callback,
            list: Box::new(list_expr),
            ty,
        });
    }
    if name == "filter"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            host_expr_type(&list_expr)
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::Filter {
            callback,
            list: Box::new(list_expr),
            ty,
        });
    }
    if name == "fold"
        && kids.len() == 4
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let init_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let list_expr = lower_host_expr(&kids[3], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            host_expr_type(&init_expr)
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::Fold {
            callback,
            init: Box::new(init_expr),
            list: Box::new(list_expr),
            ty,
        });
    }
    if name == "scan"
        && kids.len() == 4
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let init_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let list_expr = lower_host_expr(&kids[3], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            HostTypeTerm::List(Box::new(host_expr_type(&init_expr)))
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::Scan {
            callback,
            init: Box::new(init_expr),
            list: Box::new(list_expr),
            ty,
        });
    }
    if name == "partition"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            let list_ty = host_expr_type(&list_expr);
            HostTypeTerm::Tuple(vec![list_ty.clone(), list_ty])
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::Partition {
            callback,
            list: Box::new(list_expr),
            ty,
        });
    }
    if name == "flat_map"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty.is_unresolved() {
            match callback.ret_ty.clone() {
                HostTypeTerm::List(inner) => HostTypeTerm::List(inner),
                _ => fresh_host_inference(),
            }
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::FlatMap {
            callback,
            list: Box::new(list_expr),
            ty,
        });
    }
    let helper_tensor_ty = expr_tensor_type(&app_expr, program, scope)
        .or_else(|| {
            if let HostTypeTerm::Tensor(tensor_ty) = &inferred_ret_ty {
                Some(tensor_ty.clone())
            } else if let HostTypeTerm::Tensor(tensor_ty) = &explicit_ty {
                Some(tensor_ty.clone())
            } else {
                None
            }
        })
        .or_else(|| {
            // When the callee is a `(grad {} fn_arg)` node (not a plain `var`),
            // `infer_app_expr_host_type` returns None because it only handles `var`
            // callees — so `helper_tensor_ty` is None and the tensor-helper path is
            // skipped entirely.  For `grad(named_fn)(x)` the output shape equals the
            // shape of the first differentiable argument `x`, so infer it from there.
            // This lets `try_lower_tensor_helper_call` succeed even when the outer
            // `app` node carries no explicit type annotation.
            let callee = kids.first().and_then(as_list)?;
            if tag(callee) != Some("grad") {
                return None;
            }
            kids.get(1)
                .and_then(|first_arg| expr_tensor_type(first_arg, program, scope))
        });
    let (helper_expr, helper_scope, helper_bindings) =
        hoist_host_lane_tensor_bindings(&app_expr, program, scope, fn_sig.as_ref(), tensor_helpers);
    // Local callable params (e.g. `f` in `def apply(f: fn, x) = f(x)`) are
    // not representable in the tensor-helper DAG — the DAG path would box
    // the fn pointer into `chelis_scalar_tensor_from_f64` and emit C that
    // gcc rejects. Skip both tensor-helper branches and fall through to
    // the generic `HostExpr::new(HostExprKind::Call)` path so the wrapper emits `return f(x);`.
    let callee_is_local_callable = scope
        .get(&name)
        .is_some_and(|ty| matches!(ty, HostTypeTerm::Fn(_, _)));
    // A callee carrying a callable (fn-pointer) parameter cannot be
    // summarized through the tensor-helper DAG: the DAG has no
    // representation for a fn-pointer input. Such a callee is lowered by
    // inlining its body at the call site (the `has_callable_params`
    // branch below), where the local-wrapper grad/vmap form keeps its
    // rank-polymorphic shape symbolic. Skip both tensor-helper branches
    // so the inline path wins; otherwise the call monomorphizes against
    // the concrete arg shapes. This restores the lowering altitude the
    // over-broad recursion classification used to force.
    let has_callable_params = fn_sig
        .as_ref()
        .is_some_and(|(params, _)| params.iter().any(|ty| matches!(ty, HostTypeTerm::Fn(..))));
    if let Some(tensor_ty) = helper_tensor_ty.clone()
        && !callee_is_local_callable
        && !has_callable_params
        && !top_level_fn_needs_host_lane_tensor_lowering(program, &name)
        && !top_level_fn_helper_summary_rejects(program, &name)
        && !should_keep_tensor_expr_in_host_lane(&app_expr)
        && let Some(tensor_call) = try_lower_tensor_helper_call(
            &helper_expr,
            program,
            &helper_scope,
            tensor_helpers,
            tensor_ty.clone(),
        )
    {
        if helper_bindings.is_empty() {
            return tensor_call;
        }
        let ty = host_expr_type(&tensor_call);
        return HostExpr::new(HostExprKind::Let {
            bindings: helper_bindings,
            body: Box::new(tensor_call),
            ty,
        });
    }
    if let Some(tensor_ty) = helper_tensor_ty
        && !callee_is_local_callable
        && !has_callable_params
        && !top_level_fn_needs_host_lane_tensor_lowering(program, &name)
        && !top_level_fn_helper_summary_rejects(program, &name)
        && !should_keep_tensor_expr_in_host_lane(&app_expr)
        && let Some(specialized) = inline_top_level_host_call(&app_expr, program)
    {
        let pushed = push_inlining(&name);
        let lowered =
            try_lower_tensor_helper_call(&specialized, program, scope, tensor_helpers, tensor_ty);
        if pushed {
            pop_inlining(&name);
        }
        if let Some(tensor_call) = lowered {
            return tensor_call;
        }
    }
    let tensor_result = matches!(explicit_ty, HostTypeTerm::Tensor(_))
        || matches!(inferred_ret_ty, HostTypeTerm::Tensor(_));
    // WS-A8: if the callee is a polymorphic-precision sig, the host
    // emitter elided its standalone definition (per the
    // `type_expr_has_precision_var` skip in `lower_host_program`).
    // Calling such a name in C produces an undefined-symbol link
    // error; the only legal lowering is to inline the body at the
    // call site so the precision is supplied from the call's
    // concrete arg types. Force inlining for this case.
    let callee_is_polymorphic_precision = lookup_declared_type_expr(program, &name)
        .is_some_and(crate::lower::type_expr_has_precision_var);
    // Tier-2 rank polymorphism (spec/design/rank_polymorphism.md): the exact
    // analogue of the precision case above. The host emitter elided the
    // rank-poly callee's standalone definition (per the `type_expr_has_rank_var`
    // skip in `lower_host_program`); calling such a name in C is an
    // undefined-symbol link error. The only legal lowering is to inline the
    // body at the call site so the rank var is monomorphized from the call's
    // concrete arg shapes (via the DAG `tensor_rank_substitutions` path).
    let callee_is_polymorphic_rank =
        lookup_declared_type_expr(program, &name).is_some_and(crate::lower::type_expr_has_rank_var);
    if (callee_is_polymorphic_precision || callee_is_polymorphic_rank)
        && let Some(specialized) = inline_top_level_host_call(&app_expr, program)
    {
        let pushed = push_inlining(&name);
        let lowered = lower_host_expr(&specialized, program, scope, tensor_helpers);
        if pushed {
            pop_inlining(&name);
        }
        return lowered;
    }
    if has_callable_params
        && tensor_result
        && let Some(specialized) = inline_top_level_host_call(&app_expr, program)
    {
        let pushed = push_inlining(&name);
        let lowered = lower_host_expr(&specialized, program, scope, tensor_helpers);
        if pushed {
            pop_inlining(&name);
        }
        return lowered;
    }
    let args = kids[1..]
        .iter()
        .map(|arg| lower_host_expr(arg, program, scope, tensor_helpers))
        .collect::<Vec<_>>();
    let construct_ty = if let Some((adt_name, _)) = &ctor_info {
        if matches!(explicit_ty, HostTypeTerm::Adt(_, _)) {
            explicit_ty.clone()
        } else {
            HostTypeTerm::Adt(adt_name.clone(), Vec::new())
        }
    } else {
        inferred_ret_ty.clone()
    };
    if ctor_info.is_some() && !matches!(name.as_str(), "Some" | "None") {
        let expected_fields = lookup_adt_ctor_details_for_type(program, &name, Some(&construct_ty))
            .map(|(_, fields)| fields)
            .unwrap_or_default();
        let fields = args
            .into_iter()
            .enumerate()
            .map(|(index, field)| {
                if let Some(expected) = expected_fields.get(index) {
                    force_host_expr_type(field, expected.ty.clone())
                } else {
                    field
                }
            })
            .collect();
        return HostExpr::new(HostExprKind::AdtConstruct {
            ctor: name,
            fields,
            ty: construct_ty,
        });
    }
    if !BUILTIN_NAMES.contains(&name.as_str())
        && name != "Some"
        && name != "None"
        && fn_sig.is_some()
    {
        // When the explicit metadata type is Unknown OR contains unresolved
        // unresolved type variables preserve their named term; prefer
        // the inferred return type from the function's declared signature.
        // The metadata can decay to "Tuple([Unknown, Unknown])" when the
        // node-level annotator re-runs inference with a fresh subst that
        // doesn't share the outer pass's tvar bindings.
        let prefer_inferred = host_type_is_unresolved(&explicit_ty);
        return HostExpr::new(HostExprKind::Call {
            function: name,
            args,
            arg_tys: fn_sig
                .as_ref()
                .map(|(param_tys, _)| param_tys.clone())
                .unwrap_or_default(),
            ty: if prefer_inferred {
                inferred_ret_ty
            } else {
                explicit_ty
            },
        });
    }
    let ty = if !explicit_ty.is_unresolved() {
        explicit_ty
    } else {
        infer_builtin_host_type(&name, &args).unwrap_or_else(fresh_host_inference)
    };
    HostExpr::new(HostExprKind::Builtin { name, args, ty })
}

fn inline_top_level_host_call(expr: &Expr, program: &CheckedProgram) -> Option<Expr> {
    let Expr::List(app_list, _span) = expr else {
        return None;
    };
    if tag(app_list) != Some("app") {
        return None;
    }
    let kids = children(app_list);
    let callee_name = kids
        .first()
        .and_then(as_list)
        .filter(|callee| tag(callee) == Some("var"))
        .and_then(|callee| children(callee).first().and_then(symbol_name))?;
    if is_inlining(callee_name) {
        return None;
    }
    let defs = collect_program_defs(program.exprs());
    let body = lookup_program_def(&defs, callee_name)?;
    let Expr::List(fn_list, _) = body else {
        return None;
    };
    if tag(fn_list) != Some("fn") {
        return None;
    }
    let fn_kids = children(fn_list);
    let params_list = fn_kids.first().and_then(as_list)?;
    if tag(params_list) != Some("params") {
        return None;
    }
    let args = kids.get(1..)?;
    if args.len() != children(params_list).len() {
        return None;
    }
    let substitutions = children(params_list)
        .iter()
        .zip(args.iter())
        .filter_map(|(param, arg)| param_name(param).map(|name| (name, arg.clone())))
        .collect::<HashMap<_, _>>();
    Some(inline_local_callable_lets(&substitute_expr(
        fn_kids.get(1)?,
        &substitutions,
        &HashSet::new(),
    )))
}

fn substitute_expr(
    expr: &Expr,
    substitutions: &HashMap<String, Expr>,
    shadowed: &HashSet<String>,
) -> Expr {
    match expr {
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                entries: meta.entries.clone(),
                expr: Box::new(substitute_expr(&meta.expr, substitutions, shadowed)),
            },
            *span,
        ),
        Expr::List(list, _span) if tag(list) == Some("var") => {
            if let Some(name) = children(list).first().and_then(symbol_name)
                && !shadowed.contains(name)
                && let Some(replacement) = substitutions.get(name)
            {
                return replacement.clone();
            }
            expr.clone()
        }
        Expr::List(list, span) if tag(list) == Some("fn") => {
            let kids = children(list);
            let mut next_shadowed = shadowed.clone();
            if let Some(params) = kids.first().and_then(as_list) {
                for param in children(params) {
                    if let Some(name) = param_name(param) {
                        next_shadowed.insert(name);
                    }
                }
            }
            let mut elements = Vec::with_capacity(list.elements.len());
            elements.push(list.elements[0].clone());
            elements.push(list.elements[1].clone());
            if let Some(params) = kids.first() {
                elements.push(params.clone());
            }
            if let Some(body) = kids.get(1) {
                elements.push(substitute_expr(body, substitutions, &next_shadowed));
            }
            Expr::List(List { elements }, *span)
        }
        Expr::List(list, span) if tag(list) == Some("let") => {
            let kids = children(list);
            let mut next_shadowed = shadowed.clone();
            if let Some(bind_list) = kids.first().and_then(as_list)
                && tag(bind_list) == Some("bind")
            {
                let bind_kids = children(bind_list);
                for index in (0..bind_kids.len()).step_by(2) {
                    if let Some(name) = bind_kids.get(index).and_then(symbol_name) {
                        next_shadowed.insert(name.to_string());
                    }
                }
            }
            let elements = list
                .elements
                .iter()
                .map(|child| substitute_expr(child, substitutions, shadowed))
                .collect();
            if kids.len() >= 2 {
                let mut rebuilt = list.elements.clone();
                rebuilt[2] = substitute_expr(&list.elements[2], substitutions, shadowed);
                rebuilt[3] = substitute_expr(&list.elements[3], substitutions, &next_shadowed);
                Expr::List(List { elements: rebuilt }, *span)
            } else {
                Expr::List(List { elements }, *span)
            }
        }
        Expr::List(list, span) => Expr::List(
            List {
                elements: list
                    .elements
                    .iter()
                    .map(|child| substitute_expr(child, substitutions, shadowed))
                    .collect(),
            },
            *span,
        ),
        _ => expr.clone(),
    }
}

/// Whether a let-bound value is a callable expression that can be β-substituted
/// into every use site in the body. The host backend can only lower these
/// callable forms directly in callee position of an `app` node, so an alias
/// like `let g = grad(f); g(x)` must be rewritten to the inline form
/// `(grad(f))(x)` before host-lane lowering. Recognizes:
///   - `(fn (params …) body)` — anonymous function
///   - `(grad … fn …)` — gradient transform
///   - `(vmap … fn …)` — vmap transform
///   - `(vmap-grad … fn …)` — vmap of grad
///
/// Looks through a wrapping `MetaExpr` so type-annotated bindings still match.
fn is_inlinable_callable_binding_value(expr: &Expr) -> bool {
    match expr {
        Expr::List(inner, _) => matches!(
            tag(inner),
            Some("fn") | Some("grad") | Some("vmap") | Some("vmap-grad")
        ),
        Expr::MetaExpr(meta, _) => is_inlinable_callable_binding_value(&meta.expr),
        _ => false,
    }
}

fn inline_local_callable_lets(expr: &Expr) -> Expr {
    let Expr::List(list, span) = expr else {
        return expr.clone();
    };
    if tag(list) != Some("let") {
        return Expr::List(
            List {
                elements: list
                    .elements
                    .iter()
                    .map(inline_local_callable_lets)
                    .collect(),
            },
            *span,
        );
    }

    let kids = children(list);
    let Some(bind_list) = kids.first().and_then(as_list) else {
        return expr.clone();
    };
    if tag(bind_list) != Some("bind") {
        return expr.clone();
    }

    let bind_kids = children(bind_list);
    let mut rebuilt_pairs = Vec::<(String, Expr)>::new();
    let mut body = kids
        .get(1)
        .map(inline_local_callable_lets)
        .unwrap_or_else(|| {
            Expr::List(
                List {
                    elements: Vec::new(),
                },
                *span,
            )
        });

    for index in (0..bind_kids.len()).step_by(2).rev() {
        let Some(name) = bind_kids.get(index).and_then(symbol_name) else {
            continue;
        };
        let Some(value) = bind_kids.get(index + 1) else {
            continue;
        };
        let value = inline_local_callable_lets(value);
        if is_inlinable_callable_binding_value(&value) {
            // β-substitute the callable into every use site in the body.
            // This applies to local fn bindings (`let f = fn (x) => …; f(y)`),
            // and also to higher-order callable forms `grad`, `vmap`, and
            // `vmap-grad` (`let g = grad(f); g(x)`). The grad/vmap forms are
            // not first-class host values — the host backend recognizes them
            // only when they appear directly in callee position of an `app`
            // (the inline form `grad(f)(x)` lowers cleanly). Inlining the
            // alias rewrites the let-bound form to the inline form so the
            // host backend can lower it the same way.
            body = substitute_expr(
                &body,
                &HashMap::from([(name.to_string(), value)]),
                &HashSet::new(),
            );
        } else {
            rebuilt_pairs.push((name.to_string(), value));
        }
    }

    if rebuilt_pairs.is_empty() {
        return body;
    }

    rebuilt_pairs.reverse();
    let mut rebuilt_bind = vec![bind_list.elements[0].clone(), bind_list.elements[1].clone()];
    for (name, value) in rebuilt_pairs {
        rebuilt_bind.push(Expr::Atom(Atom::Symbol(name), *span));
        rebuilt_bind.push(value);
    }
    Expr::List(
        List {
            elements: vec![
                list.elements[0].clone(),
                list.elements[1].clone(),
                Expr::List(
                    List {
                        elements: rebuilt_bind,
                    },
                    *span,
                ),
                body,
            ],
        },
        *span,
    )
}

fn expr_needs_host_lane_tensor_lowering(expr: &Expr, program: &CheckedProgram) -> bool {
    let graph = top_level_fn_call_graph(program);
    let recursive = recursive_top_level_fn_names_from_graph(&graph);
    let fn_names = graph.keys().cloned().collect::<HashSet<_>>();
    collect_called_top_level_fns(expr, &fn_names)
        .into_iter()
        .any(|name| call_graph_reaches_any(&graph, &name, &recursive))
}

fn top_level_fn_needs_host_lane_tensor_lowering(program: &CheckedProgram, name: &str) -> bool {
    let graph = top_level_fn_call_graph(program);
    let recursive = recursive_top_level_fn_names_from_graph(&graph);
    call_graph_reaches_any(&graph, name, &recursive)
}

/// A caller of `name` must fall back to a plain host function call
/// (`name(args)`) rather than inlining `name`'s body and re-summarizing
/// it, when `name`'s own tensor-helper lowering would be a *summary
/// rejection*. Inlining a rejected helper into the caller would either
/// register a false sparse/BLAS summary on the caller or rebuild the
/// caller around the rejected helper's `__tensor_*` shim; the
/// unspecialized helper is the only honest lowering, so the caller
/// emits a direct call to it.
///
/// This is the structured replacement for the previous incidental
/// trigger: the over-broad recursion classification (every function was
/// marked recursive by the reflexive `call_graph_reaches_any`) forced
/// every caller through the host-lane fallback, which happened to route
/// rejected helpers to a host call. With recursion classification
/// corrected, the fallback is driven directly off the
/// `SummaryRejection` machinery instead.
fn top_level_fn_helper_summary_rejects(program: &CheckedProgram, name: &str) -> bool {
    // Guard against unbounded re-entry: while we are already inlining
    // `name` we must not recursively re-lower it to probe its
    // rejections.
    if is_inlining(name) {
        return false;
    }
    let defs = collect_program_defs(program.exprs());
    let Some(body) = lookup_program_def(&defs, name) else {
        return false;
    };
    if !matches!(body, Expr::List(list, _) if tag(list) == Some("fn")) {
        return false;
    }
    let pushed = push_inlining(name);
    let rejects = lower_host_function(name, body, None, program)
        .map(|mut function| {
            collect_function_summary_rejections(&mut function);
            !function.summary_rejections.is_empty()
        })
        .unwrap_or(false);
    if pushed {
        pop_inlining(name);
    }
    rejects
}

/// A top-level expression body needs host-lane lowering when it calls a
/// top-level fn whose lowering cannot be cleanly summarized into a
/// tensor helper at the call site. Today that is any callee carrying a
/// callable (fn-pointer) parameter: the tensor-helper DAG has no
/// representation for a fn-pointer input and would coerce it into a
/// scalar-tensor placeholder, so the call must stay in the host lane
/// where the fn application lowers to a direct `f(x)` call.
///
/// This replaces the prior incidental trigger (the over-broad recursion
/// classification) for the local-wrapper-over-callable-param case.
fn expr_calls_top_level_fn_with_callable_param(expr: &Expr, program: &CheckedProgram) -> bool {
    let graph = top_level_fn_call_graph(program);
    let fn_names = graph.keys().cloned().collect::<HashSet<_>>();
    collect_called_top_level_fns(expr, &fn_names)
        .iter()
        .any(|name| top_level_fn_has_callable_param(program, name))
}

fn top_level_fn_has_callable_param(program: &CheckedProgram, name: &str) -> bool {
    let Some((param_tys, _)) = lookup_declared_fn_type(program, name) else {
        return false;
    };
    param_tys
        .iter()
        .any(|ty| matches!(ty, HostTypeTerm::Fn(_, _)))
}

/// A tensor-returning body that calls a summary-rejecting top-level fn
/// must not be summarized through the tensor-helper DAG: doing so
/// inlines the rejected helper's body into the caller (registering a
/// false summary or rebuilding the caller around the rejected shim).
/// Route such bodies through host-lane lowering so the call lowers to a
/// direct host function call to the unspecialized helper.
fn expr_calls_summary_rejecting_top_level_fn(expr: &Expr, program: &CheckedProgram) -> bool {
    let graph = top_level_fn_call_graph(program);
    let fn_names = graph.keys().cloned().collect::<HashSet<_>>();
    collect_called_top_level_fns(expr, &fn_names)
        .iter()
        .any(|name| top_level_fn_helper_summary_rejects(program, name))
}

fn top_level_fn_call_graph(program: &CheckedProgram) -> HashMap<String, HashSet<String>> {
    let defs = collect_program_defs(program.exprs());
    let fn_names = defs
        .iter()
        .filter(|(_, body)| matches!(body, Expr::List(list, _) if tag(list) == Some("fn")))
        .map(|(name, _)| name.clone())
        .collect::<HashSet<_>>();

    fn_names
        .iter()
        .map(|name| {
            let callees = defs
                .get(name)
                .map(|body| collect_called_top_level_fns(body, &fn_names))
                .unwrap_or_default();
            (name.clone(), callees)
        })
        .collect()
}

fn recursive_top_level_fn_names_from_graph(
    graph: &HashMap<String, HashSet<String>>,
) -> HashSet<String> {
    graph
        .keys()
        .filter(|name| call_graph_reaches_any(graph, name, &HashSet::from([(*name).clone()])))
        .cloned()
        .collect()
}

fn collect_called_top_level_fns(expr: &Expr, fn_names: &HashSet<String>) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match current {
            Expr::MetaExpr(meta, _) => stack.push(&meta.expr),
            Expr::List(list, _) => {
                if tag(list) == Some("app")
                    && let Some(callee) = children(list).first().and_then(as_list)
                    && tag(callee) == Some("var")
                    && let Some(name) = children(callee).first().and_then(symbol_name)
                    && fn_names.contains(name)
                {
                    out.insert(name.to_string());
                }
                stack.extend(children(list).iter());
            }
            _ => {}
        }
    }
    out
}

fn call_graph_reaches_any(
    graph: &HashMap<String, HashSet<String>>,
    start: &str,
    targets: &HashSet<String>,
) -> bool {
    let mut visited = HashSet::new();
    let mut stack = graph
        .get(start)
        .into_iter()
        .flat_map(|callees| callees.iter().cloned())
        .collect::<Vec<_>>();
    while let Some(name) = stack.pop() {
        if targets.contains(&name) {
            return true;
        }
        if !visited.insert(name.clone()) {
            continue;
        }
        if let Some(next) = graph.get(&name) {
            stack.extend(next.iter().cloned());
        }
    }
    false
}

fn hoist_host_lane_tensor_bindings(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    fn_sig: Option<&(Vec<HostTypeTerm>, HostTypeTerm)>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> (Expr, HashMap<String, HostTypeTerm>, Vec<HostBinding>) {
    let Expr::List(list, span) = expr else {
        return (expr.clone(), scope.clone(), Vec::new());
    };
    if tag(list) != Some("app") {
        return (expr.clone(), scope.clone(), Vec::new());
    }

    let kids = children(list);
    if kids.is_empty() {
        return (expr.clone(), scope.clone(), Vec::new());
    }

    let mut new_elements = vec![
        list.elements[0].clone(),
        list.elements[1].clone(),
        kids[0].clone(),
    ];
    let mut scoped = scope.clone();
    let mut bindings = Vec::new();

    for (index, arg) in kids.iter().enumerate().skip(1) {
        if should_keep_tensor_expr_in_host_lane(arg) {
            let value = lower_host_expr(arg, program, scope, tensor_helpers);
            let preferred_ty = fn_sig
                .and_then(|(param_tys, _)| param_tys.get(index - 1))
                .cloned()
                .or_else(|| expr_tensor_type(arg, program, scope).map(HostTypeTerm::Tensor))
                .or_else(|| {
                    let ty = expr_host_type(arg, program, scope);
                    (!ty.is_unresolved()).then_some(ty)
                });
            let ty = preferred_ty.unwrap_or_else(|| host_expr_type(&value));
            let value = force_host_expr_type(value, ty.clone());
            let name = format!("__host_tensor_arg_{index}");
            bindings.push(HostBinding {
                name: name.clone(),
                display_name: None,
                ty: ty.clone(),
                value,
            });
            scoped.insert(name.clone(), ty);
            new_elements.push(Expr::List(
                List {
                    elements: vec![
                        Expr::Atom(Atom::Symbol("var".to_string()), *span),
                        Expr::Map(
                            chelis_deep::ast::MetaMap {
                                entries: Vec::new(),
                            },
                            *span,
                        ),
                        Expr::Atom(Atom::Symbol(name), *span),
                    ],
                },
                *span,
            ));
        } else {
            new_elements.push(arg.clone());
        }
    }

    (
        Expr::List(
            List {
                elements: new_elements,
            },
            *span,
        ),
        scoped,
        bindings,
    )
}

fn host_fn_signature(ty: &HostTypeTerm) -> Option<(Vec<HostTypeTerm>, HostTypeTerm)> {
    match ty {
        HostTypeTerm::Fn(params, ret) => Some((params.clone(), (**ret).clone())),
        _ => None,
    }
}

fn lower_host_callback(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> Option<HostCallback> {
    match expr {
        Expr::MetaExpr(meta, _) => lower_host_callback(&meta.expr, program, scope, tensor_helpers),
        Expr::List(list, _) if tag(list) == Some("fn") => {
            let kids = children(list);
            let params_list = kids.first().and_then(as_list)?;
            if tag(params_list) != Some("params") {
                return None;
            }
            let (param_tys, ret_ty) =
                expr_fn_type(expr).unwrap_or((Vec::new(), fresh_host_inference()));
            let mut callback_scope = scope.clone();
            let mut params = Vec::new();
            for (index, param) in children(params_list).iter().enumerate() {
                let name = param_name(param)?;
                let ty = param_tys
                    .get(index)
                    .cloned()
                    .filter(|ty| !ty.is_unresolved())
                    .or_else(|| param_host_type(param))
                    .unwrap_or_else(fresh_host_inference);
                callback_scope.insert(name.clone(), ty.clone());
                params.push(HostParam { name, ty });
            }
            let body = lower_host_expr(kids.get(1)?, program, &callback_scope, tensor_helpers);
            let ret_ty = if ret_ty.is_unresolved() {
                host_expr_type(&body)
            } else {
                ret_ty
            };
            Some(HostCallback {
                kind: HostCallbackKind::Inline {
                    params,
                    body: Box::new(body),
                },
                ret_ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some("var") => {
            let name = children(list).first().and_then(symbol_name)?;
            let (param_tys, ret_ty) = scope
                .get(name)
                .and_then(host_fn_signature)
                .or_else(|| {
                    lookup_declared_host_type(program, name).and_then(|ty| host_fn_signature(&ty))
                })
                .or_else(|| lookup_declared_fn_type(program, name))
                .or_else(|| {
                    lookup_program_def(&collect_program_defs(program.exprs()), name)
                        .and_then(expr_fn_type)
                })?;
            let params = param_tys
                .into_iter()
                .enumerate()
                .map(|(index, ty)| HostParam {
                    name: format!("arg{index}"),
                    ty,
                })
                .collect();
            Some(HostCallback {
                kind: HostCallbackKind::Named {
                    function: name.to_string(),
                    params,
                },
                ret_ty,
            })
        }
        _ => None,
    }
}

fn lower_list_literal_items(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> Option<Vec<HostExpr>> {
    match expr {
        Expr::MetaExpr(meta, _) => {
            lower_list_literal_items(&meta.expr, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("var") => {
            (children(list).first().and_then(symbol_name) == Some("Nil")).then(Vec::new)
        }
        Expr::List(list, _) if tag(list) == Some("app") => {
            let kids = children(list);
            if kids.len() != 3 {
                return None;
            }
            let func_name = kids
                .first()
                .and_then(as_list)
                .and_then(|inner| (tag(inner) == Some("var")).then_some(inner))
                .and_then(|inner| children(inner).first().and_then(symbol_name));
            if func_name != Some("Cons") {
                return None;
            }
            let head = lower_host_expr(&kids[1], program, scope, tensor_helpers);
            let mut tail = lower_list_literal_items(&kids[2], program, scope, tensor_helpers)?;
            tail.insert(0, head);
            Some(tail)
        }
        _ => None,
    }
}

fn tensor_helper_args(
    inputs: &[HostTensorInput],
    scope: &HashMap<String, HostTypeTerm>,
) -> Vec<HostExpr> {
    inputs
        .iter()
        .map(|input| {
            HostExpr::new(HostExprKind::Var(
                input.name.clone(),
                scope
                    .get(&input.name)
                    .cloned()
                    .unwrap_or_else(|| host_type_from_tensor_input(&input.ty)),
            ))
        })
        .collect()
}

fn tensor_helper_inputs(dag: &crate::Dag) -> Vec<HostTensorInput> {
    let mut seen = HashSet::new();
    dag.nodes()
        .iter()
        .filter_map(|node| match &node.op {
            crate::RiscOp::Load { name } if seen.insert(name.as_str().to_string()) => {
                Some(HostTensorInput {
                    name: name.as_str().to_string(),
                    ty: node.output_type.clone(),
                })
            }
            _ => None,
        })
        .collect()
}

fn remap_tensor_helper_dim_symbols(
    dag: &crate::Dag,
    scope: &HashMap<String, HostTypeTerm>,
    expected_output: &TensorType,
) -> crate::Dag {
    fn tensor_type_has_synthetic_dims(tensor_ty: &TensorType) -> bool {
        tensor_ty.dims.iter().any(|dim| {
            matches!(dim, crate::dag::DimInfo::Named(name, None) if {
                let mut chars = name.chars();
                matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
            })
        })
    }

    let formal_inputs = tensor_helper_inputs(dag);
    let mut actual_inputs = formal_inputs
        .iter()
        .map(|input| match scope.get(&input.name) {
            Some(HostTypeTerm::Tensor(actual)) => actual.clone(),
            _ => input.ty.clone(),
        })
        .collect::<Vec<_>>();
    let mut formal_params = formal_inputs
        .iter()
        .map(|input| input.ty.clone())
        .collect::<Vec<_>>();
    if let Some(root) = dag.roots().first().and_then(|id| dag.get(*id)) {
        formal_params.push(root.output_type.clone());
        let actual_output = if tensor_type_has_synthetic_dims(expected_output) {
            match root.op {
                crate::dag::RiscOp::Permute { ref axes } => root
                    .inputs
                    .first()
                    .and_then(|id| dag.get(*id))
                    .map(|node| {
                        let mut output = node.output_type.clone();
                        output.dims = axes
                            .iter()
                            .filter_map(|axis| node.output_type.dims.get(*axis).cloned())
                            .collect();
                        output
                    })
                    .unwrap_or_else(|| expected_output.clone()),
                crate::dag::RiscOp::UniformLike { .. } | crate::dag::RiscOp::Dropout { .. } => root
                    .inputs
                    .first()
                    .and_then(|id| dag.get(*id))
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| expected_output.clone()),
                _ => expected_output.clone(),
            }
        } else {
            expected_output.clone()
        };
        actual_inputs.push(actual_output);
    }
    let mut remapped = crate::lower::remap_tensor_dim_symbols(dag, &formal_params, &actual_inputs);
    // chelis#632 (needed by the chelis#631 oracle): anon wildcards are no
    // longer substitution keys in `tensor_dim_substitutions`, so the
    // declared return no longer paints its dims across every
    // wildcard-typed node (distinct runtime extents conflated under one
    // symbol → runtime-dim guard aborts on well-formed programs). The
    // declared return still owns the ROOT's shape: retype the root
    // POSITIONALLY, anon axis by anon axis — but ONLY on axes the root
    // op itself can declare at run time (`dag::op_declarable_axes`). A
    // symbol painted anywhere else has no declaring Load or op and trips
    // the `symbolic_occurrences` ICE; those axes stay anon and size
    // themselves per node.
    if let (Some(root_id), Some(actual_output)) =
        (dag.roots().first().copied(), actual_inputs.last())
        && let Some(root) = remapped.get(root_id)
        && root.output_type.dims.len() == actual_output.dims.len()
    {
        let is_anon = |dim: &crate::dag::DimInfo| matches!(dim, crate::dag::DimInfo::Named(name, None) if name.is_empty() || name == "*");
        let declarable = crate::dag::op_declarable_axes(&remapped, root);
        let mut output = root.output_type.clone();
        let mut changed = false;
        for (axis, (dim, actual_dim)) in output
            .dims
            .iter_mut()
            .zip(actual_output.dims.iter())
            .enumerate()
        {
            if declarable.contains(&axis) && is_anon(dim) && !is_anon(actual_dim) {
                *dim = actual_dim.clone();
                changed = true;
            }
        }
        if changed {
            let op = root.op.clone();
            let inputs = root.inputs.clone();
            remapped.replace_node(root_id, op, inputs, output);
        }
    }
    actualize_tensor_helper_types(&remapped, scope)
}

fn actualize_tensor_helper_types(
    dag: &crate::Dag,
    scope: &HashMap<String, HostTypeTerm>,
) -> crate::Dag {
    fn inferred_load_type(
        name: &str,
        scope: &HashMap<String, HostTypeTerm>,
        fallback: &TensorType,
    ) -> TensorType {
        match scope.get(name) {
            Some(HostTypeTerm::Tensor(actual)) => actual.clone(),
            _ => fallback.clone(),
        }
    }

    fn precision_like(input: &TensorType, precision: chelis_types::types::Prim) -> TensorType {
        TensorType {
            dims: input.dims.clone(),
            precision,
        }
    }

    fn synthetic_dims(tensor_ty: &TensorType) -> bool {
        tensor_ty.dims.iter().any(|dim| {
            matches!(dim, crate::dag::DimInfo::Named(name, None) if {
                let mut chars = name.chars();
                matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
            })
        })
    }

    fn synthetic_dim(dim: &crate::dag::DimInfo) -> bool {
        matches!(dim, crate::dag::DimInfo::Named(name, None) if {
            let mut chars = name.chars();
            matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
        })
    }

    fn merge_dim(lhs: &crate::dag::DimInfo, rhs: &crate::dag::DimInfo) -> crate::dag::DimInfo {
        match (synthetic_dim(lhs), synthetic_dim(rhs)) {
            (true, false) => rhs.clone(),
            _ => lhs.clone(),
        }
    }

    fn merge_binary_tensor_types(
        lhs: &TensorType,
        rhs: &TensorType,
        precision: chelis_types::types::Prim,
    ) -> TensorType {
        if lhs.dims.len() != rhs.dims.len() {
            return precision_like(lhs, precision);
        }
        TensorType {
            dims: lhs
                .dims
                .iter()
                .zip(rhs.dims.iter())
                .map(|(lhs, rhs)| merge_dim(lhs, rhs))
                .collect(),
            precision,
        }
    }

    let mut inferred = HashMap::<crate::dag::NodeId, TensorType>::new();
    let mut uses = HashMap::<crate::dag::NodeId, Vec<crate::dag::NodeId>>::new();
    for node in dag.nodes() {
        for input in &node.inputs {
            uses.entry(*input).or_default().push(node.id);
        }
    }
    for node in dag.nodes() {
        let actual = match &node.op {
            crate::dag::RiscOp::Load { name } => {
                Some(inferred_load_type(name.as_str(), scope, &node.output_type))
            }
            crate::dag::RiscOp::Add
            | crate::dag::RiscOp::Mul
            | crate::dag::RiscOp::CmpLt
            | crate::dag::RiscOp::MaxElem => node
                .inputs
                .first()
                .and_then(|lhs| inferred.get(lhs))
                .map(|lhs| {
                    node.inputs
                        .get(1)
                        .and_then(|rhs| inferred.get(rhs))
                        .map(|rhs| merge_binary_tensor_types(lhs, rhs, node.output_type.precision))
                        .unwrap_or_else(|| precision_like(lhs, node.output_type.precision))
                }),
            crate::dag::RiscOp::Neg
            | crate::dag::RiscOp::Exp
            | crate::dag::RiscOp::Log
            | crate::dag::RiscOp::Sin
            | crate::dag::RiscOp::Sqrt
            | crate::dag::RiscOp::Cos
            | crate::dag::RiscOp::Tan
            | crate::dag::RiscOp::Atan
            | crate::dag::RiscOp::Abs
            | crate::dag::RiscOp::Floor
            | crate::dag::RiscOp::Ceil
            | crate::dag::RiscOp::Round
            | crate::dag::RiscOp::UniformLike { .. }
            | crate::dag::RiscOp::Dropout { .. }
            | crate::dag::RiscOp::Copy
            | crate::dag::RiscOp::Drop
            | crate::dag::RiscOp::Realize
            | crate::dag::RiscOp::Cast { .. } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| precision_like(input, node.output_type.precision)),
            crate::dag::RiscOp::Sum { axis, .. }
            | crate::dag::RiscOp::MaxReduce { axis }
            | crate::dag::RiscOp::MinReduce { axis }
            | crate::dag::RiscOp::ProdReduce { axis }
            | crate::dag::RiscOp::Argmax { axis }
            | crate::dag::RiscOp::Argmin { axis } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| TensorType {
                    dims: input
                        .dims
                        .iter()
                        .enumerate()
                        .filter_map(|(index, dim)| (index != *axis).then_some(dim.clone()))
                        .collect(),
                    precision: node.output_type.precision,
                }),
            crate::dag::RiscOp::Permute { axes } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| TensorType {
                    dims: axes
                        .iter()
                        .filter_map(|axis| input.dims.get(*axis).cloned())
                        .collect(),
                    precision: node.output_type.precision,
                }),
            crate::dag::RiscOp::Expand { axis, size } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .and_then(|input| {
                    let mut dims = input.dims.clone();
                    if *axis > dims.len() {
                        return None;
                    }
                    let inserted = match size {
                        crate::dag::DimExpr::Concrete(value) => crate::dag::DimInfo::Lit(*value),
                        crate::dag::DimExpr::Sym(name) => {
                            crate::dag::DimInfo::Named(name.clone(), None)
                        }
                        crate::dag::DimExpr::Mul(_, _) | crate::dag::DimExpr::Div(_, _) => {
                            return None;
                        }
                    };
                    dims.insert(*axis, inserted);
                    Some(TensorType {
                        dims,
                        precision: node.output_type.precision,
                    })
                }),
            _ => None,
        }
        .or_else(|| {
            node.reusable_input
                .and_then(|id| inferred.get(&id))
                .filter(|input| input.dims.len() == node.output_type.dims.len())
                .map(|input| precision_like(input, node.output_type.precision))
        });
        if let Some(actual) = actual {
            inferred.insert(node.id, actual);
        }
    }

    loop {
        let mut changed = false;
        for node in dag.nodes() {
            if !matches!(node.op, crate::dag::RiscOp::Expand { .. })
                || !synthetic_dims(&node.output_type)
            {
                continue;
            }
            let Some(actual) = uses.get(&node.id).and_then(|user_ids| {
                user_ids
                    .iter()
                    .filter_map(|user_id| inferred.get(user_id))
                    .find(|user_ty| user_ty.dims.len() == node.output_type.dims.len())
                    .cloned()
            }) else {
                continue;
            };
            let entry = inferred
                .entry(node.id)
                .or_insert_with(|| node.output_type.clone());
            if entry.dims != actual.dims {
                *entry = TensorType {
                    dims: actual.dims,
                    precision: node.output_type.precision,
                };
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut actualized = dag.clone();
    let node_ids = actualized
        .nodes()
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    let mut synthetic_renames = HashMap::<String, crate::dag::DimInfo>::new();
    for id in node_ids {
        let Some(node) = actualized.get(id).cloned() else {
            continue;
        };
        let Some(actual) = inferred.get(&id) else {
            continue;
        };
        if !synthetic_dims(&node.output_type) || node.output_type.dims.len() != actual.dims.len() {
            continue;
        }
        // Record which minted `dN` alias each output axis resolved to,
        // so op-internal references to the same alias can be renamed in
        // lockstep below.
        for (old_dim, new_dim) in node.output_type.dims.iter().zip(actual.dims.iter()) {
            if let crate::dag::DimInfo::Named(name, None) = old_dim
                && synthetic_dim(old_dim)
                && old_dim != new_dim
            {
                match synthetic_renames.entry(name.clone()) {
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(new_dim.clone());
                    }
                    std::collections::hash_map::Entry::Occupied(existing) => {
                        // A single checker dim-var has a single extent in
                        // a well-typed program; a conflicting re-bind
                        // means the helper DAG was already inconsistent.
                        // Fail loudly in debug rather than renaming op
                        // fields with the wrong extent (review #363 N1).
                        debug_assert_eq!(
                            existing.get(),
                            new_dim,
                            "synthetic dim `{name}` resolved to conflicting actuals"
                        );
                    }
                }
            }
        }
        actualized.replace_node(id, node.op, node.inputs, actual.clone());
        if let Some(reusable_input) = node.reusable_input {
            actualized.set_reusable_input(id, reusable_input);
        }
    }
    if synthetic_renames.is_empty() {
        return actualized;
    }
    // chelis#345 (op-internal half): `replace_node` above rewrites
    // OUTPUT types only, leaving op-internal fields (`Expand::size`,
    // `Reshape::new_shape`, `BlasMatmul` dims) holding the stale minted
    // names — the mixed state (`type: [Named("n")]` next to
    // `size: Sym("d47")`) that `dag::symbolic_occurrences`' Bucket 4d
    // sweep rejects because no Load declares the alias. Apply the
    // collected renames to every dim reference so the helper DAG stays
    // internally consistent. Non-synthetic (user-facing) names are
    // never in the map and pass through untouched.
    crate::lower::apply_dim_substitutions(&actualized, &synthetic_renames)
}

/// chelis#631: does this Deep expr contain a `grad`/`vmap`/`vmap-grad`
/// node? Such exprs lower through the DAG lane only (the host lane
/// cannot resolve them), so the fail-reachability gate must not divert
/// them.
fn expr_contains_grad_like(expr: &Expr) -> bool {
    match expr {
        Expr::List(list, _) => {
            matches!(tag(list), Some("grad" | "vmap" | "vmap-grad"))
                || list.elements.iter().any(expr_contains_grad_like)
        }
        Expr::MetaExpr(meta, _) => expr_contains_grad_like(&meta.expr),
        _ => false,
    }
}

/// chelis#631: does this Deep expr — or any def it (transitively)
/// references — contain a `fail` application? Conservative: any `(var
/// fail)` reference counts, and a referenced def is walked once (the
/// `visiting` set both breaks recursion cycles and memoizes). Used by
/// [`lower_tensor_helper_dag`] to keep fail-reaching bodies out of
/// tensor-helper DAGs, where `fail` is a zero placeholder rather than an
/// abort.
fn expr_reaches_fail(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
    visiting: &mut HashSet<String>,
) -> bool {
    match expr {
        Expr::List(list, _) => {
            if tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                if name == "fail" {
                    return true;
                }
                if let Some(body) = defs.get(name)
                    && visiting.insert(name.to_string())
                {
                    return expr_reaches_fail(body, defs, visiting);
                }
                return false;
            }
            list.elements
                .iter()
                .any(|kid| expr_reaches_fail(kid, defs, visiting))
        }
        Expr::MetaExpr(meta, _) => expr_reaches_fail(&meta.expr, defs, visiting),
        _ => false,
    }
}

fn collect_program_defs(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut defs = HashMap::new();
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) == Some("def")
            && let (Some(name), Some(body)) = (
                children(list).first().and_then(symbol_name),
                children(list).get(1),
            )
        {
            defs.insert(name.to_string(), body.clone());
        }
    }
    defs
}

fn top_level_items(exprs: &[Expr]) -> Vec<&Expr> {
    let mut out = Vec::new();
    for expr in exprs {
        collect_top_level_items(expr, &mut out);
    }
    out
}

fn collect_top_level_items<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    if tag(list) == Some("module") {
        for child in list.elements.iter().skip(3) {
            collect_top_level_items(child, out);
        }
        return;
    }
    out.push(expr);
}

fn collect_tensor_scope(scope: &HashMap<String, HostTypeTerm>) -> HashMap<String, TensorType> {
    scope
        .iter()
        .filter_map(|(name, ty)| {
            tensor_type_from_host_input(ty).map(|tensor| (name.clone(), tensor))
        })
        .collect()
}

fn tensor_type_from_host_input(ty: &HostTypeTerm) -> Option<TensorType> {
    match ty {
        HostTypeTerm::Tensor(tensor) => Some(tensor.clone()),
        // WS-4: map each declared float width to its own precision. The
        // previous unconditional `Float64 -> F32` silently truncated f64
        // scalar entry parameters to 4-byte storage; the declared width
        // now survives because `parse_host_type_with_subst` preserves the
        // `f32`/`f64` distinction.
        &HostTypeTerm::Float32 => Some(TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::F32,
        }),
        &HostTypeTerm::Float64 => Some(TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::F64,
        }),
        &HostTypeTerm::Int64 => Some(TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Int64,
        }),
        &HostTypeTerm::Bool => Some(TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Bool,
        }),
        _ => None,
    }
}

fn host_type_from_tensor_input(ty: &TensorType) -> HostTypeTerm {
    if ty.dims.is_empty() {
        match ty.precision {
            chelis_types::types::Prim::Bool => HostTypeTerm::Bool,
            chelis_types::types::Prim::Int8
            | chelis_types::types::Prim::Int16
            | chelis_types::types::Prim::Int32
            | chelis_types::types::Prim::Int64 => HostTypeTerm::Int64,
            // WS-4: restore the float width on the reverse boundary so a
            // rank-0 f32 tensor round-trips to `Float32` (declared width)
            // rather than collapsing every float to `Float64`. The
            // narrower IEEE floats (`f16`/`bf16`) have no dedicated host
            // scalar variant, so they keep the coarse `Float32` class.
            chelis_types::types::Prim::F32
            | chelis_types::types::Prim::F16
            | chelis_types::types::Prim::Bf16
            | chelis_types::types::Prim::F8e4m3 => HostTypeTerm::Float32,
            chelis_types::types::Prim::F64 => HostTypeTerm::Float64,
            chelis_types::types::Prim::String => HostTypeTerm::String,
        }
    } else {
        HostTypeTerm::Tensor(ty.clone())
    }
}

fn expr_host_type(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
) -> HostTypeTerm {
    match expr {
        Expr::Atom(Atom::Int(_), _) => HostTypeTerm::Int64,
        Expr::Atom(Atom::Float(_), _) => HostTypeTerm::Float64,
        Expr::Atom(Atom::Bool(_), _) => HostTypeTerm::Bool,
        Expr::Atom(Atom::Str(_), _) => HostTypeTerm::String,
        Expr::List(list, _) if tag(list) == Some("var") => children(list)
            .first()
            .and_then(symbol_name)
            .and_then(|name| {
                expr_type(expr)
                    .filter(|ty| !ty.is_unresolved())
                    .or_else(|| {
                        scope
                            .get(name)
                            .cloned()
                            .or_else(|| lookup_declared_host_type(program, name))
                    })
            })
            .unwrap_or_else(fresh_host_inference),
        Expr::List(list, _) if tag(list) == Some("app") => {
            let explicit = expr_type(expr).unwrap_or_else(fresh_host_inference);
            if app_expr_needs_inferred_type(&explicit) {
                let inferred = infer_app_expr_host_type(list, program, scope)
                    .unwrap_or_else(fresh_host_inference);
                if should_prefer_inferred_app_type(&explicit, &inferred) {
                    inferred
                } else if !explicit.is_unresolved() {
                    explicit
                } else {
                    inferred
                }
            } else {
                explicit
            }
        }
        _ => expr_type(expr).unwrap_or_else(fresh_host_inference),
    }
}

fn app_expr_needs_inferred_type(explicit: &HostTypeTerm) -> bool {
    explicit.is_unresolved() || host_type_has_synthetic_tensor_dims(explicit)
}

fn infer_app_expr_host_type(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
) -> Option<HostTypeTerm> {
    let kids = children(list);
    let callee = kids.first().and_then(as_list)?;
    if tag(callee) != Some("var") {
        return None;
    }
    let name = children(callee).first().and_then(symbol_name)?;
    if !BUILTIN_NAMES.contains(&name) {
        return lookup_declared_fn_type(program, name).map(|(_, ret)| ret);
    }
    if name == "einsum" {
        let equation = match kids.get(1) {
            Some(Expr::Atom(Atom::Str(value), _)) => value.as_str(),
            _ => return Some(fresh_host_inference()),
        };
        let tensors = kids[2..]
            .iter()
            .map(|arg| match expr_host_type(arg, program, scope) {
                HostTypeTerm::Tensor(tensor_ty) => Some(tensor_ty),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        return infer_einsum_tensor_type(equation, &tensors).map(HostTypeTerm::Tensor);
    }
    // chelis#340: the whole named-axis reduction family is type-inferred
    // here (the positional/int-literal axis form), not only `sum`/`mean`.
    // Each drops the reduced axis; `argmax_reduce`/`argmin_reduce` return an
    // int64 index tensor while the value reductions keep the operand
    // precision. Recovering the tensor type lets the host lane route the
    // call through `try_lower_tensor_helper_call` (the tensor-DAG kernel
    // lane the C backend uses) instead of falling through to the
    // host-emit "unsupported builtin" path when the checker's `type`
    // annotation is absent or carries synthetic rank-poly dims.
    if matches!(
        name,
        "sum"
            | "mean"
            | "max_reduce"
            | "min_reduce"
            | "prod_reduce"
            | "argmax_reduce"
            | "argmin_reduce"
    ) && let (Some(input), Some(axis_expr)) = (kids.get(1), kids.get(2))
        && let HostTypeTerm::Tensor(tensor_ty) = expr_host_type(input, program, scope)
        && let Some(axis) = expr_int_literal(axis_expr)
    {
        let mut reduced = reduce_axis_tensor_type(&tensor_ty, axis as usize);
        if matches!(name, "argmax_reduce" | "argmin_reduce") {
            reduced.precision = chelis_types::types::Prim::Int64;
        }
        return Some(HostTypeTerm::Tensor(reduced));
    }
    if name == "expand"
        && let (Some(input), Some(axis_expr), Some(size_expr)) =
            (kids.get(1), kids.get(2), kids.get(3))
        && let HostTypeTerm::Tensor(tensor_ty) = expr_host_type(input, program, scope)
        && let (Some(axis), Some(size)) = (expr_int_literal(axis_expr), expr_int_literal(size_expr))
    {
        let mut dims = tensor_ty.dims.clone();
        let axis = axis as usize;
        if axis <= dims.len() {
            dims.insert(axis, crate::dag::DimInfo::Lit(size as usize));
            return Some(HostTypeTerm::Tensor(TensorType {
                dims,
                precision: tensor_ty.precision,
            }));
        }
    }
    // Issue #308: `scalar_to_tensor` result precision must follow the
    // operand's *float* precision. The coarse host lane collapses f32
    // and f64 scalars into a single `HostTypeTerm::Float64`, so the
    // arg-ty-based fallback below cannot distinguish them and defaults
    // to f32 — mis-typing an f64 const-broadcast
    // (`scalar_to_tensor(cast(c, f64))`) as `Tensor(F32)`, which makes
    // the C emitter select `chelis_scalar_tensor_from_f32` (4-byte
    // storage) for an f64 value. Recover the precision from the Deep
    // operand itself (checker annotation or explicit cast target)
    // while it is still visible.
    if name == "scalar_to_tensor"
        && let Some(arg) = kids.get(1)
        && let Some(precision) = expr_scalar_float_precision(arg)
    {
        return Some(HostTypeTerm::Tensor(TensorType {
            dims: vec![],
            precision,
        }));
    }
    // chelis#631: a tensor-list `concat`'s concat axis is sized at run
    // time by `chelis_tensor_concat`; the coarse host type must not carry
    // the ELEMENT's extent on that axis. (The precise type is the
    // checker's `tensor_concat_result_type`, spec §4.5.4; this fallback
    // fires when the checker annotation is absent or synthetic.)
    if name == "concat"
        && let Some(input) = kids.get(1)
        && let HostTypeTerm::List(inner) = expr_host_type(input, program, scope)
        && let HostTypeTerm::Tensor(element) = inner.as_ref()
    {
        let axis = kids
            .get(2)
            .and_then(expr_int_literal)
            .and_then(|raw| normalize_host_axis(element.dims.len(), raw));
        return Some(concat_host_tensor_type(element, axis));
    }
    let arg_tys = kids[1..]
        .iter()
        .map(|arg| expr_host_type(arg, program, scope))
        .collect::<Vec<_>>();
    must_infer_builtin_host_type_from_arg_tys(name, &arg_tys)
}

/// Issue #308: recover the float precision of a scalar Deep expression
/// for `scalar_to_tensor` result typing. Reads, in order:
///
///   1. the checker's `type` meta when it is a float `(t-prim {} p)`;
///   2. an explicit `(cast {} _ (t-prim {} p))` target when `p` is a
///      float precision.
///
/// Returns `None` for integer/bool operands (the coarse
/// `infer_builtin_host_type_from_arg_tys` arms already type those
/// correctly) and when the precision is not recoverable — in that case
/// the caller falls back to the coarse f32 default, which the C emit
/// dispatch and the consuming tensor-helper Load both share, so the
/// write and read sides stay consistent even in the fallback.
fn expr_scalar_float_precision(expr: &Expr) -> Option<chelis_types::types::Prim> {
    if let Expr::MetaExpr(meta, _) = expr {
        return expr_scalar_float_precision(&meta.expr);
    }
    let Expr::List(list, _) = expr else {
        return None;
    };
    let prim_of_t_prim = |type_expr: &Expr| -> Option<chelis_types::types::Prim> {
        let Expr::List(inner, _) = type_expr else {
            return None;
        };
        if tag(inner) != Some("t-prim") {
            return None;
        }
        children(inner)
            .first()
            .and_then(symbol_name)
            .and_then(chelis_types::types::Prim::parse_name)
    };
    if let Some(Expr::Map(meta, _)) = list.elements.get(1)
        && let Some((_, type_expr)) = meta.entries.iter().find(|(key, _)| key == "type")
        && let Some(prim) = prim_of_t_prim(type_expr)
    {
        return prim.is_float().then_some(prim);
    }
    if tag(list) == Some("cast")
        && let Some(target) = children(list).get(1)
        && let Some(prim) = prim_of_t_prim(target)
    {
        return prim.is_float().then_some(prim);
    }
    None
}

fn should_prefer_inferred_app_type(explicit: &HostTypeTerm, inferred: &HostTypeTerm) -> bool {
    explicit.is_unresolved()
        || matches!(
            (explicit, inferred),
            (HostTypeTerm::Tensor(_), HostTypeTerm::Tensor(_)) if host_type_has_synthetic_tensor_dims(explicit)
                && !host_type_has_synthetic_tensor_dims(inferred)
        )
}

fn host_type_has_synthetic_tensor_dims(ty: &HostTypeTerm) -> bool {
    // Known exposure (review #363 N5, pre-existing): a USER dim literally
    // named `d2` matches this minted-name heuristic and would be treated
    // as synthetic. The checker's dim-var minting owns the `d<digits>`
    // namespace today; if user-facing single-letter+digit dims ever
    // matter, the minting needs a reserved prefix instead.
    fn synthetic_dim_name(name: &str) -> bool {
        let mut chars = name.chars();
        matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
    }

    match ty {
        HostTypeTerm::Tensor(tensor_ty) => tensor_ty.dims.iter().any(
            |dim| matches!(dim, crate::dag::DimInfo::Named(name, None) if synthetic_dim_name(name)),
        ),
        HostTypeTerm::Tuple(items) => items.iter().any(host_type_has_synthetic_tensor_dims),
        HostTypeTerm::List(inner) | HostTypeTerm::Option(inner) => {
            host_type_has_synthetic_tensor_dims(inner)
        }
        _ => false,
    }
}

fn infer_einsum_tensor_type(equation: &str, tensors: &[TensorType]) -> Option<TensorType> {
    let (inputs, output) = equation.split_once("->")?;
    let input_specs = inputs.split(',').map(str::trim).collect::<Vec<_>>();
    if input_specs.len() != tensors.len() {
        return None;
    }

    let mut labels = HashMap::<char, crate::dag::DimInfo>::new();
    for (spec, tensor) in input_specs.iter().zip(tensors.iter()) {
        let axes = spec
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<Vec<_>>();
        if axes.len() != tensor.dims.len() {
            return None;
        }
        for (axis, dim) in axes.into_iter().zip(tensor.dims.iter().cloned()) {
            labels.entry(axis).or_insert(dim);
        }
    }

    let dims = output
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .map(|axis| labels.get(&axis).cloned())
        .collect::<Option<Vec<_>>>()?;
    let precision = tensors
        .first()
        .map(|tensor| tensor.precision)
        .unwrap_or(chelis_types::types::Prim::F32);
    Some(TensorType { dims, precision })
}

fn expr_int_literal(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Atom(Atom::Int(value), _) => Some(*value),
        Expr::MetaExpr(meta, _) => expr_int_literal(&meta.expr),
        Expr::List(list, _) if tag(list) == Some("lit") => {
            children(list).first().and_then(expr_int_literal)
        }
        // Movement-op axis/size args are routinely written as
        // `cast(0, int32)` / `cast(2, int32)` (the canonical integer-
        // literal form, since bare int literals default to int32 and the
        // axis/size parameters are int32). A `cast` whose operand is an
        // integer literal carries the same compile-time value, so see
        // through it: otherwise `infer_app_expr_host_type`'s `expand`
        // shape handler bails and the result type degrades to a
        // dims-less placeholder, splitting a constant-broadcast `let`
        // binding into an unsupported host-lane builtin (issue #300).
        Expr::List(list, _) if tag(list) == Some("cast") => {
            children(list).first().and_then(expr_int_literal)
        }
        _ => None,
    }
}

/// chelis#631: an integer literal reaching this HostExpr position,
/// seeing through the canonical `cast(N, int32)` spelling (the HostExpr
/// analog of [`expr_int_literal`]'s cast peel).
fn host_expr_int_literal(expr: &HostExpr) -> Option<i64> {
    match &expr.kind {
        HostExprKind::Int(value) => Some(*value),
        HostExprKind::Builtin { name, args, .. } if name == "cast" => {
            args.first().and_then(host_expr_int_literal)
        }
        _ => None,
    }
}

/// chelis#631: normalize a possibly-negative literal axis against a rank
/// (`-1` is the last axis); `None` when out of range — the checker owns
/// the user-facing out-of-bounds diagnostic, this lane just degrades to
/// all-wildcard.
fn normalize_host_axis(rank: usize, raw: i64) -> Option<usize> {
    let rank = rank as i64;
    let axis = if raw < 0 { rank + raw } else { raw };
    (0..rank).contains(&axis).then_some(axis as usize)
}

/// chelis#631: the coarse host-lane type of a tensor-list `concat` — the
/// element type with the concat axis wildcarded (every axis when the
/// axis is unknown in the calling lane). The wildcard renames to a
/// per-node anon dim in the C emitter and is sized from the runtime
/// `chelis_tensor_concat` result, so helper signatures stay honest.
fn concat_host_tensor_type(element: &TensorType, axis: Option<usize>) -> HostTypeTerm {
    let anon = || crate::dag::DimInfo::Named("*".to_string(), None);
    let mut dims = element.dims.clone();
    match axis {
        Some(axis) if axis < dims.len() => dims[axis] = anon(),
        _ => dims.fill(anon()),
    }
    HostTypeTerm::Tensor(TensorType {
        dims,
        precision: element.precision,
    })
}

fn reduce_axis_tensor_type(tensor_ty: &TensorType, axis: usize) -> TensorType {
    let mut dims = tensor_ty.dims.clone();
    if axis < dims.len() {
        dims.remove(axis);
    }
    TensorType {
        dims,
        precision: tensor_ty.precision,
    }
}

fn lookup_type_expr<'a>(type_env: &'a HashMap<String, Expr>, name: &str) -> Option<&'a Expr> {
    type_env.get(name).or_else(|| {
        let mut matches = type_env
            .iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    })
}

fn lookup_declared_type_expr<'a>(program: &'a CheckedProgram, name: &str) -> Option<&'a Expr> {
    find_top_level_sig_expr(program.exprs(), name)
        .or_else(|| lookup_type_expr(program.type_env(), name))
}

fn lookup_declared_host_type(program: &CheckedProgram, name: &str) -> Option<HostTypeTerm> {
    lookup_declared_type_expr(program, name)
        .map(|ty| decode_host_type_or_raise(ty, &HashMap::new()))
}

fn lookup_declared_fn_type(
    program: &CheckedProgram,
    name: &str,
) -> Option<(Vec<HostTypeTerm>, HostTypeTerm)> {
    lookup_declared_type_expr(program, name).and_then(parse_fn_type_expr)
}

fn lookup_program_def<'a>(defs: &'a HashMap<String, Expr>, name: &str) -> Option<&'a Expr> {
    defs.get(name).or_else(|| {
        let mut matches = defs
            .iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    })
}

fn terminal_name_matches(full_name: &str, short_name: &str) -> bool {
    full_name == short_name || terminal_name(full_name) == terminal_name(short_name)
}

fn terminal_name(name: &str) -> &str {
    name.rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(name)
}

fn find_top_level_sig_expr<'a>(exprs: &'a [Expr], name: &str) -> Option<&'a Expr> {
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("defsig") {
            continue;
        }
        let kids = children(list);
        let Some(sig_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if terminal_name_matches(sig_name, name) {
            return kids.get(1);
        }
    }
    None
}

fn expr_tensor_type(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostTypeTerm>,
) -> Option<TensorType> {
    match expr_host_type(expr, program, scope) {
        HostTypeTerm::Tensor(ty) => Some(ty),
        _ => None,
    }
}

fn expr_type(expr: &Expr) -> Option<HostTypeTerm> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let meta = match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta,
        _ => return None,
    };
    meta.entries
        .iter()
        .find(|(key, _)| key == "type")
        .map(|(_, value)| decode_host_type_or_raise(value, &HashMap::new()))
}

fn expr_fn_type(expr: &Expr) -> Option<(Vec<HostTypeTerm>, HostTypeTerm)> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let meta = match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta,
        _ => return None,
    };
    meta.entries
        .iter()
        .find(|(key, _)| key == "type")
        .and_then(|(_, value)| parse_fn_type_expr(value))
}

fn parse_fn_type_expr(expr: &Expr) -> Option<(Vec<HostTypeTerm>, HostTypeTerm)> {
    let (args, ret) = parse_fn_type_expr_parts(expr)?;
    Some((
        args.iter()
            .map(|arg| decode_host_type_or_raise(arg, &HashMap::new()))
            .collect(),
        decode_host_type_or_raise(&ret, &HashMap::new()),
    ))
}

fn parse_fn_type_expr_parts(expr: &Expr) -> Option<(Vec<Expr>, Expr)> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if tag(list) != Some("t-fn") {
        return None;
    }
    let kids = children(list);
    let (ret, args) = kids.split_last()?;
    Some((args.to_vec(), ret.clone()))
}

fn decode_host_type_with_subst(
    expr: &Expr,
    subst: &HashMap<String, HostTypeTerm>,
) -> Result<HostTypeTerm, HostTypeDecodeError> {
    decode_host_type(expr).map(|term| substitute_host_type_term(term, subst))
}

fn substitute_host_type_term(
    term: HostTypeTerm,
    subst: &HashMap<String, HostTypeTerm>,
) -> HostTypeTerm {
    match term {
        HostTypeTerm::TypeVariable(name) => subst
            .get(&name)
            .cloned()
            .unwrap_or(HostTypeTerm::TypeVariable(name)),
        HostTypeTerm::Fn(params, ret) => HostTypeTerm::Fn(
            params
                .into_iter()
                .map(|param| substitute_host_type_term(param, subst))
                .collect(),
            Box::new(substitute_host_type_term(*ret, subst)),
        ),
        HostTypeTerm::Adt(name, args) => HostTypeTerm::Adt(
            name,
            args.into_iter()
                .map(|arg| substitute_host_type_term(arg, subst))
                .collect(),
        ),
        HostTypeTerm::List(inner) => {
            HostTypeTerm::List(Box::new(substitute_host_type_term(*inner, subst)))
        }
        HostTypeTerm::Dict(key, value) => HostTypeTerm::Dict(
            Box::new(substitute_host_type_term(*key, subst)),
            Box::new(substitute_host_type_term(*value, subst)),
        ),
        HostTypeTerm::Tuple(items) => HostTypeTerm::Tuple(
            items
                .into_iter()
                .map(|item| substitute_host_type_term(item, subst))
                .collect(),
        ),
        HostTypeTerm::Option(inner) => {
            HostTypeTerm::Option(Box::new(substitute_host_type_term(*inner, subst)))
        }
        concrete_or_state => concrete_or_state,
    }
}

fn decode_host_type_or_raise(expr: &Expr, subst: &HashMap<String, HostTypeTerm>) -> HostTypeTerm {
    decode_host_type_with_subst(expr, subst).unwrap_or_else(|error| {
        crate::lower::raise_fatal_lowering_diagnostic(
            crate::lower::LowerDiagnostic::new(
                format!("invalid checked host type metadata: {error} ([05-UNS-1]; chelis#730)"),
                Some(expr.span()),
                expr.span_id().map(str::to_string),
            )
            .fatal(),
        )
    })
}

fn option_inner_type(expr: &HostExpr) -> HostTypeTerm {
    match host_expr_type(expr) {
        HostTypeTerm::Option(inner) => (*inner).clone(),
        _ => fresh_host_inference(),
    }
}

fn host_expr_type(expr: &HostExpr) -> HostTypeTerm {
    match &expr.kind {
        HostExprKind::Int(_) => HostTypeTerm::Int64,
        HostExprKind::Float(_) => HostTypeTerm::Float64,
        HostExprKind::Bool(_) => HostTypeTerm::Bool,
        HostExprKind::String(_) => HostTypeTerm::String,
        HostExprKind::List(_, ty) => ty.clone(),
        HostExprKind::Tuple(_, ty) => ty.clone(),
        HostExprKind::Var(_, ty)
        | HostExprKind::Call { ty, .. }
        | HostExprKind::Builtin { ty, .. }
        | HostExprKind::AdtConstruct { ty, .. }
        | HostExprKind::AdtFieldAccess { ty, .. }
        | HostExprKind::If { ty, .. }
        | HostExprKind::MatchOption { ty, .. }
        | HostExprKind::MatchAdt { ty, .. }
        | HostExprKind::Let { ty, .. }
        | HostExprKind::Map { ty, .. }
        | HostExprKind::Filter { ty, .. }
        | HostExprKind::Fold { ty, .. }
        | HostExprKind::Scan { ty, .. }
        | HostExprKind::Partition { ty, .. }
        | HostExprKind::FlatMap { ty, .. }
        | HostExprKind::WithSeed { ty, .. }
        | HostExprKind::TensorCall { ty, .. } => ty.clone(),
        HostExprKind::Unit => HostTypeTerm::Unit,
    }
}

fn force_host_expr_type(expr: HostExpr, ty: HostTypeTerm) -> HostExpr {
    let HostExpr {
        kind,
        span_id,
        merged_spans,
    } = expr;
    let new_kind = match kind {
        HostExprKind::List(items, _) => {
            let items = match &ty {
                HostTypeTerm::List(inner) => items
                    .into_iter()
                    .map(|item| force_host_expr_type(item, (**inner).clone()))
                    .collect(),
                _ => items,
            };
            HostExprKind::List(items, ty)
        }
        HostExprKind::Tuple(items, _) => {
            let items = match &ty {
                HostTypeTerm::Tuple(expected) if expected.len() == items.len() => items
                    .into_iter()
                    .zip(expected.iter())
                    .map(|(item, expected)| force_host_expr_type(item, expected.clone()))
                    .collect(),
                _ => items,
            };
            HostExprKind::Tuple(items, ty)
        }
        HostExprKind::Var(name, _) => HostExprKind::Var(name, ty),
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ..
        } => {
            let args = args
                .into_iter()
                .enumerate()
                .map(|(index, arg)| {
                    if let Some(expected) = arg_tys.get(index) {
                        force_host_expr_type(arg, expected.clone())
                    } else {
                        arg
                    }
                })
                .collect();
            HostExprKind::Call {
                function,
                args,
                arg_tys,
                ty,
            }
        }
        HostExprKind::Builtin { name, args, .. } => {
            let args = conform_builtin_arguments(&name, args, &ty);
            HostExprKind::Builtin { name, args, ty }
        }
        HostExprKind::AdtConstruct { ctor, fields, .. } => {
            HostExprKind::AdtConstruct { ctor, fields, ty }
        }
        HostExprKind::AdtFieldAccess {
            base, field_index, ..
        } => HostExprKind::AdtFieldAccess {
            base,
            field_index,
            ty,
        },
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => HostExprKind::If {
            cond: Box::new(force_host_expr_type(*cond, HostTypeTerm::Bool)),
            then_expr: Box::new(force_host_expr_type(*then_expr, ty.clone())),
            else_expr: Box::new(force_host_expr_type(*else_expr, ty.clone())),
            ty,
        },
        HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ..
        } => HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr: Box::new(force_host_expr_type(*some_expr, ty.clone())),
            none_expr: Box::new(force_host_expr_type(*none_expr, ty.clone())),
            ty,
        },
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => HostExprKind::MatchAdt {
            scrutinee,
            arms: arms
                .into_iter()
                .map(|mut arm| {
                    arm.expr = force_host_expr_type(arm.expr, ty.clone());
                    arm
                })
                .collect(),
            default_expr: default_expr
                .map(|expr| Box::new(force_host_expr_type(*expr, ty.clone()))),
            ty,
        },
        HostExprKind::Let {
            mut bindings, body, ..
        } => {
            for binding in &mut bindings {
                let placeholder = HostExpr::new(HostExprKind::Unit);
                let value = std::mem::replace(&mut binding.value, placeholder);
                binding.value = force_host_expr_type(value, binding.ty.clone());
            }
            HostExprKind::Let {
                bindings,
                body: Box::new(force_host_expr_type(*body, ty.clone())),
                ty,
            }
        }
        HostExprKind::Map { callback, list, .. } => HostExprKind::Map { callback, list, ty },
        HostExprKind::Filter { callback, list, .. } => HostExprKind::Filter { callback, list, ty },
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        } => HostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        },
        HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => HostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        },
        HostExprKind::Partition { callback, list, .. } => {
            HostExprKind::Partition { callback, list, ty }
        }
        HostExprKind::FlatMap { callback, list, .. } => {
            HostExprKind::FlatMap { callback, list, ty }
        }
        HostExprKind::WithSeed { seed, body, .. } => HostExprKind::WithSeed {
            seed: Box::new(force_host_expr_type(*seed, HostTypeTerm::Int64)),
            body: Box::new(force_host_expr_type(*body, ty.clone())),
            ty,
        },
        HostExprKind::TensorCall { helper, args, .. } => {
            HostExprKind::TensorCall { helper, args, ty }
        }
        other => other,
    };
    HostExpr {
        kind: new_kind,
        span_id,
        merged_spans,
    }
}

/// Materialize checked context into every host-program root before term
/// resolution. This is the sole expectation/refinement boundary named by
/// loud_unsupported.md C6.3; codegen therefore receives one resolved tree
/// and has no authority to pick a type when annotations disagree.
fn conform_host_program_types(program: &mut HostProgram) {
    for binding in &mut program.globals {
        let placeholder = HostExpr::new(HostExprKind::Unit);
        let value = std::mem::replace(&mut binding.value, placeholder);
        binding.value = force_host_expr_type(value, binding.ty.clone());
    }
    for function in &mut program.functions {
        let placeholder = HostExpr::new(HostExprKind::Unit);
        let body = std::mem::replace(&mut function.body, placeholder);
        function.body = force_host_expr_type(body, function.ret_ty.clone());
    }
}

fn conform_builtin_arguments(
    name: &str,
    args: Vec<HostExpr>,
    result_ty: &HostTypeTerm,
) -> Vec<HostExpr> {
    let scalar_numeric = matches!(
        name,
        "add"
            | "sub"
            | "mul"
            | "div"
            | "trunc_div"
            | "floor_div"
            | "mod"
            | "neg"
            | "sqrt"
            | "exp"
            | "log"
            | "sin"
            | "cos"
            | "tanh"
            | "pow"
            | "abs"
            | "min"
            | "max"
    ) && matches!(result_ty, HostTypeTerm::Scalar(_));

    args.into_iter()
        .enumerate()
        .map(|(index, arg)| {
            let expected = match (name, result_ty, index) {
                ("Some", HostTypeTerm::Option(inner), 0) => Some((**inner).clone()),
                ("copy" | "debug", _, 0) => Some(result_ty.clone()),
                ("append", HostTypeTerm::List(_), 0) => Some(result_ty.clone()),
                ("append", HostTypeTerm::List(inner), 1) => Some((**inner).clone()),
                ("dict_of", HostTypeTerm::Dict(key, value), 0) => {
                    Some(HostTypeTerm::List(Box::new(HostTypeTerm::Tuple(vec![
                        (**key).clone(),
                        (**value).clone(),
                    ]))))
                }
                ("to_tensor", HostTypeTerm::Tensor(tensor), 0) => {
                    Some(HostTypeTerm::List(Box::new(HostTypeTerm::Scalar(
                        HostPrecisionTerm::Concrete(tensor.precision),
                    ))))
                }
                ("scalar_to_tensor", HostTypeTerm::Tensor(tensor), 0) => Some(
                    HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(tensor.precision)),
                ),
                _ if scalar_numeric => Some(result_ty.clone()),
                _ => None,
            };
            if let Some(expected) = expected {
                force_host_expr_type(arg, expected)
            } else {
                arg
            }
        })
        .collect()
}

fn infer_builtin_host_type(name: &str, args: &[HostExpr]) -> Option<HostTypeTerm> {
    let arg_tys = args.iter().map(host_expr_type).collect::<Vec<_>>();
    match name {
        "einsum" => {
            let equation = match args.first().map(|e| &e.kind) {
                Some(HostExprKind::String(value)) => value.as_str(),
                _ => return Some(fresh_host_inference()),
            };
            let tensors = args[1..]
                .iter()
                .map(|arg| match host_expr_type(arg) {
                    HostTypeTerm::Tensor(tensor_ty) => Some(tensor_ty),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            infer_einsum_tensor_type(equation, &tensors).map(HostTypeTerm::Tensor)
        }
        "tuple-get" => match (arg_tys.first(), args.get(1).map(|e| &e.kind)) {
            (Some(HostTypeTerm::Tuple(items)), Some(HostExprKind::Int(index))) => items
                .get(*index as usize)
                .cloned()
                .or(Some(fresh_host_inference())),
            _ => Some(fresh_host_inference()),
        },
        // chelis#631: this lane still sees the axis ARGUMENT (unlike the
        // arg-tys-only fallback), so a literal axis wildcards only the
        // concat axis and keeps the element's other extents.
        "concat" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match inner.as_ref() {
                HostTypeTerm::Tensor(element) => {
                    let axis = args
                        .get(1)
                        .and_then(host_expr_int_literal)
                        .and_then(|raw| normalize_host_axis(element.dims.len(), raw));
                    Some(concat_host_tensor_type(element, axis))
                }
                _ => must_infer_builtin_host_type_from_arg_tys(name, &arg_tys),
            },
            _ => must_infer_builtin_host_type_from_arg_tys(name, &arg_tys),
        },
        _ => must_infer_builtin_host_type_from_arg_tys(name, &arg_tys),
    }
}

/// A builtin application reached host-result inference with an argument
/// shape that cannot satisfy that builtin's typed contract.
///
/// This is deliberately not represented by an inference variable: invalid
/// input and valid-but-underconstrained input are different states.  The
/// former is a typed lowering failure; only the latter may allocate a fresh
/// [`HostInferenceVar`].
#[derive(Debug, Clone, PartialEq)]
struct HostBuiltinInferenceError {
    builtin: String,
    argument_types: Vec<HostTypeTerm>,
    requirement: String,
}

impl fmt::Display for HostBuiltinInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid host result inference for builtin `{}` with arguments {:?}: {} \
             (chelis#730; [05-UNS-1])",
            self.builtin, self.argument_types, self.requirement
        )
    }
}

impl std::error::Error for HostBuiltinInferenceError {}

fn invalid_builtin_inference(
    name: &str,
    arg_tys: &[HostTypeTerm],
    requirement: impl Into<String>,
) -> HostBuiltinInferenceError {
    HostBuiltinInferenceError {
        builtin: name.to_string(),
        argument_types: arg_tys.to_vec(),
        requirement: requirement.into(),
    }
}

fn term_matches_or_is_unresolved(
    ty: &HostTypeTerm,
    predicate: impl FnOnce(&HostTypeTerm) -> bool,
) -> bool {
    ty.is_unresolved() || predicate(ty)
}

/// Infer a builtin result without laundering invalid input into an anonymous
/// type hole.  A successful unresolved result means inference genuinely
/// needs context (for example arg-types-only `tuple-get` cannot see its index).
fn infer_builtin_host_type_from_arg_tys(
    name: &str,
    arg_tys: &[HostTypeTerm],
) -> Result<Option<HostTypeTerm>, HostBuiltinInferenceError> {
    let require_first = |description: &str,
                         predicate: fn(&HostTypeTerm) -> bool|
     -> Result<(), HostBuiltinInferenceError> {
        match arg_tys.first() {
            Some(ty) if term_matches_or_is_unresolved(ty, predicate) => Ok(()),
            _ => Err(invalid_builtin_inference(
                name,
                arg_tys,
                format!("first argument must be {description}"),
            )),
        }
    };
    let is_tensor = |ty: &HostTypeTerm| matches!(ty, HostTypeTerm::Tensor(_));
    let is_list = |ty: &HostTypeTerm| matches!(ty, HostTypeTerm::List(_));
    let is_dict = |ty: &HostTypeTerm| matches!(ty, HostTypeTerm::Dict(_, _));

    match name {
        "sum" | "mean" | "max_reduce" | "min_reduce" | "prod_reduce" | "argmax_reduce"
        | "argmin_reduce" | "reshape" | "expand" | "pad" | "shrink" | "stride" | "permute"
        | "split" | "sort" | "tensor_to_scalar" | "to_list" => {
            require_first("a tensor", is_tensor)?;
            if name == "to_list"
                && matches!(
                    arg_tys.first(),
                    Some(HostTypeTerm::Tensor(TensorType {
                        precision: Prim::F8e4m3,
                        ..
                    }))
                )
            {
                return Err(invalid_builtin_inference(
                    name,
                    arg_tys,
                    "f8e4m3 is deferred by spec/04-type-system.md section 1.1.1",
                ));
            }
        }
        "index" | "take" | "chunk" | "flatten" | "enumerate" | "dict_of" | "to_tensor"
        | "pad_sequences" | "pad_sequences_to" => {
            require_first("a list", is_list)?;
        }
        "drop" if arg_tys.len() != 1 => require_first("a list", is_list)?,
        "concat" => {
            require_first(
                "a list of tensors",
                |ty| matches!(ty, HostTypeTerm::List(inner) if matches!(inner.as_ref(), HostTypeTerm::Tensor(_))),
            )?;
            if arg_tys.get(1).is_none() {
                return Err(invalid_builtin_inference(
                    name,
                    arg_tys,
                    "concat requires an axis argument",
                ));
            }
        }
        "zip" | "dict_merge" => {
            require_first(
                if name == "zip" { "a list" } else { "a dict" },
                if name == "zip" { is_list } else { is_dict },
            )?;
            let second_ok = arg_tys.get(1).is_some_and(|ty| {
                term_matches_or_is_unresolved(ty, if name == "zip" { is_list } else { is_dict })
            });
            if !second_ok {
                return Err(invalid_builtin_inference(
                    name,
                    arg_tys,
                    if name == "zip" {
                        "second argument must be a list"
                    } else {
                        "second argument must be a dict"
                    },
                ));
            }
        }
        "dict_get" | "dict_remove" | "dict_insert" | "dict_keys" | "dict_values"
        | "dict_entries" => require_first("a dict", is_dict)?,
        "map" | "filter" | "fold" | "scan" | "partition" | "flat_map" => {
            let list_index = if matches!(name, "fold" | "scan") {
                2
            } else {
                1
            };
            if !arg_tys
                .get(list_index)
                .is_some_and(|ty| term_matches_or_is_unresolved(ty, is_list))
            {
                return Err(invalid_builtin_inference(
                    name,
                    arg_tys,
                    format!("argument {list_index} must be a list"),
                ));
            }
        }
        "scalar_to_tensor" => {
            require_first("a scalar", |ty| matches!(ty, HostTypeTerm::Scalar(_)))?
        }
        _ => {}
    }

    Ok(infer_builtin_host_type_from_arg_tys_unchecked(
        name, arg_tys,
    ))
}

fn must_infer_builtin_host_type_from_arg_tys(
    name: &str,
    arg_tys: &[HostTypeTerm],
) -> Option<HostTypeTerm> {
    infer_builtin_host_type_from_arg_tys(name, arg_tys).unwrap_or_else(|error| panic!("{error}"))
}

fn infer_builtin_host_type_from_arg_tys_unchecked(
    name: &str,
    arg_tys: &[HostTypeTerm],
) -> Option<HostTypeTerm> {
    let tensor_arg = arg_tys.iter().find_map(|ty| match ty {
        HostTypeTerm::Tensor(tensor_ty) => Some(tensor_ty.clone()),
        _ => None,
    });
    match name {
        "add" | "sub" | "mul" | "div" | "floor_div" | "trunc_div" | "neg" | "exp" | "log"
        | "sin" | "sqrt" | "relu" | "sigmoid" | "tanh" | "silu" | "gelu" | "max_elem"
        | "min_elem" | "copy" | "uniform_like" | "dropout" | "softmax" => {
            if let Some(tensor_ty) = tensor_arg {
                Some(HostTypeTerm::Tensor(tensor_ty))
            } else if arg_tys
                .iter()
                .any(|ty| matches!(ty, &HostTypeTerm::Float64))
            {
                Some(HostTypeTerm::Float64)
            } else if arg_tys
                .iter()
                .any(|ty| matches!(ty, &HostTypeTerm::Float32))
            {
                // WS-4: an f32 scalar operand keeps the result in f32; only
                // a genuine f64 operand widens the result. Falling through
                // to `Int64` here would mis-type an all-f32 scalar
                // arithmetic result as an integer.
                Some(HostTypeTerm::Float32)
            } else if arg_tys.iter().any(HostTypeTerm::is_unresolved) {
                // chelis#730 Phase 1 (census row 7, chelis#714/#718): an
                // Unknown-typed operand (an f16/bf16/int8/int16 scalar with
                // no host representation) must not silently type the result
                // as Int64 - propagate the Unknown so the C emitter's
                // baking-point guard rejects loudly instead of emitting
                // int64_t arithmetic over garbage.
                Some(fresh_host_inference())
            } else {
                Some(HostTypeTerm::Int64)
            }
        }
        // chelis#340: the named-axis reduction family classifies as a
        // tensor in the coarse host lane (the real output shape — the
        // reduced axis dropped — is recomputed inside the tensor-helper
        // DAG; this coarse type only needs to keep the result classified as
        // a Tensor so the call routes through the tensor-DAG kernel lane).
        // `max_reduce`/`min_reduce`/`prod_reduce` keep the operand
        // precision; `argmax_reduce`/`argmin_reduce` return an int64 index
        // tensor.
        "sum" | "mean" | "max_reduce" | "min_reduce" | "prod_reduce" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tensor(tensor_ty.clone())),
            _ => Some(fresh_host_inference()),
        },
        "argmax_reduce" | "argmin_reduce" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tensor(TensorType {
                dims: tensor_ty.dims.clone(),
                precision: chelis_types::types::Prim::Int64,
            })),
            _ => Some(fresh_host_inference()),
        },
        "mod" | "bitand" | "bitor" | "bitxor" | "shl" | "shr" | "string_len" | "rank" | "shape"
        | "numel" => Some(HostTypeTerm::Int64),
        "cmplt" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tensor(TensorType {
                dims: tensor_ty.dims.clone(),
                precision: chelis_types::types::Prim::Bool,
            })),
            _ => Some(HostTypeTerm::Bool),
        },
        // Movement ops preserve the element precision and stay tensors.
        // The concrete output shape (e.g. `expand`'s broadcast axis) is
        // recomputed inside the tensor-helper DAG; the host `HostTypeTerm`
        // is coarse (precision + a dims placeholder used only for the
        // typed-runtime dispatch), so propagating the *input* tensor
        // type here is sufficient to keep the result classified as a
        // tensor. Without these arms `expand`/`pad`/`shrink`/`stride`/
        // `permute` fell through to `None`, so a host-lane `let k =
        // expand(scalar_to_tensor(c), 0, n)` binding lost its tensor
        // type and was emitted as `void* k = /* unsupported builtin
        // expand */ 0`, then mistyped as a scalar at the consuming
        // tensor-helper callsite (`(float)(void* k)`, issue #300).
        "reshape" | "expand" | "pad" | "shrink" | "stride" | "permute" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tensor(tensor_ty.clone())),
            _ => Some(fresh_host_inference()),
        },
        "lt" | "gt" | "gte" | "lte" | "eq" | "neq" | "and" | "or" | "not" | "string_contains"
        | "string_starts_with" | "string_ends_with" => Some(HostTypeTerm::Bool),
        "string_concat" | "string_trim" | "string_slice" | "to_string" => {
            Some(HostTypeTerm::String)
        }
        "to_int" => Some(HostTypeTerm::Option(Box::new(HostTypeTerm::Int64))),
        "to_float" => Some(HostTypeTerm::Option(Box::new(HostTypeTerm::Float64))),
        "len" => Some(HostTypeTerm::Int64),
        "index" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => Some((**inner).clone()),
            _ => Some(fresh_host_inference()),
        },
        "append" => arg_tys.first().cloned(),
        // chelis#631: this arg-tys-only lane cannot see the axis VALUE, so
        // every axis wildcards — returning the element type VERBATIM baked
        // the element's concat-axis extent into tensor-helper signatures
        // (the host-lane copy of the checker's pre-#631 concat bug).
        "concat" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(HostTypeTerm::List(inner)), Some(&HostTypeTerm::Int64))
                if matches!(inner.as_ref(), HostTypeTerm::Tensor(_)) =>
            {
                match inner.as_ref() {
                    HostTypeTerm::Tensor(element) => Some(concat_host_tensor_type(element, None)),
                    _ => Some(fresh_host_inference()),
                }
            }
            (Some(lhs), Some(_)) => Some(lhs.clone()),
            _ => Some(fresh_host_inference()),
        },
        "split" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::List(Box::new(
                HostTypeTerm::Tensor(tensor_ty.clone()),
            ))),
            _ => Some(fresh_host_inference()),
        },
        "scatter" | "scatter_replace" | "scatter_elements" | "where" | "cumsum" | "diagonal"
        | "trace" | "clamp" => arg_tys.first().cloned(),
        "sort" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor_ty)) => Some(HostTypeTerm::Tuple(vec![
                HostTypeTerm::Tensor(tensor_ty.clone()),
                HostTypeTerm::Tensor(TensorType {
                    dims: tensor_ty.dims.clone(),
                    precision: chelis_types::types::Prim::Int64,
                }),
            ])),
            _ => Some(fresh_host_inference()),
        },
        "tuple-get" => Some(fresh_host_inference()),
        "drop" if arg_tys.len() == 1 => Some(HostTypeTerm::Unit),
        "take" | "drop" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => {
                Some(HostTypeTerm::List(Box::new((**inner).clone())))
            }
            _ => Some(fresh_host_inference()),
        },
        "chunk" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => Some(HostTypeTerm::List(Box::new(
                HostTypeTerm::List(Box::new((**inner).clone())),
            ))),
            _ => Some(fresh_host_inference()),
        },
        "range" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::Int64))),
        "map" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(_), Some(HostTypeTerm::List(inner))) => {
                Some(HostTypeTerm::List(Box::new((**inner).clone())))
            }
            _ => Some(fresh_host_inference()),
        },
        "filter" => arg_tys.get(1).cloned(),
        "fold" => arg_tys.get(1).cloned(),
        "scan" => match arg_tys.get(1) {
            Some(init_ty) => Some(HostTypeTerm::List(Box::new(init_ty.clone()))),
            None => Some(fresh_host_inference()),
        },
        // Issue #257: `tensor_scan(initial: T, fn: (T, int64) -> T, n: int64) -> tensor[n, T]`.
        //
        // We deliberately do NOT synthesize a concrete `HostTypeTerm::Tensor`
        // precision here. `HostTypeTerm` is a *coarse* host-IR class: every
        // integer width collapses to `Int64` and every float width to
        // `Float64` (see `host_type_from_tensor_input`), so by the time
        // the initial value's type reaches this arm its real dtype
        // (`int8`..`int64`, `f16`..`f64`) is already gone. Any concrete
        // precision we picked would be a guess — e.g. an earlier version
        // mapped `Float64 -> F32`, which is wrong for an `f64` initial,
        // and `Int64` for an `int32` initial. The authoritative element
        // type lives in the real type checker (`chelis-types`
        // `infer.rs::infer_app` "tensor_scan" arm) and in the runtime,
        // which reads the precision straight off the initial value.
        //
        // Returning `Unknown` is sound because this host type is *never
        // consumed*: `tensor_scan` is host-only, so any compiled-backend
        // program that reaches it is rejected up front by
        // `compiler.rs::reject_host_only_builtins` (which keys off the
        // `Builtin` node, not its inferred type) before the C/HIP emitter
        // runs, and the `chelis eval`/`chelis test` interpreter never
        // consults host-IR types at all. Emitting `Unknown` keeps this
        // arm from advertising a precision it cannot actually know.
        "tensor_scan" => Some(fresh_host_inference()),
        "partition" => match arg_tys.get(1) {
            Some(list_ty) => Some(HostTypeTerm::Tuple(vec![list_ty.clone(), list_ty.clone()])),
            None => Some(fresh_host_inference()),
        },
        "flat_map" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(_), Some(HostTypeTerm::List(_))) => match arg_tys.first() {
                Some(ty) if ty.is_unresolved() => Some(fresh_host_inference()),
                Some(_) => None,
                None => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "flatten" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match inner.as_ref() {
                HostTypeTerm::List(nested) => {
                    Some(HostTypeTerm::List(Box::new((**nested).clone())))
                }
                _ => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "zip" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(HostTypeTerm::List(lhs)), Some(HostTypeTerm::List(rhs))) => {
                Some(HostTypeTerm::List(Box::new(HostTypeTerm::Tuple(vec![
                    (**lhs).clone(),
                    (**rhs).clone(),
                ]))))
            }
            _ => Some(fresh_host_inference()),
        },
        "enumerate" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => {
                Some(HostTypeTerm::List(Box::new(HostTypeTerm::Tuple(vec![
                    HostTypeTerm::Int64,
                    (**inner).clone(),
                ]))))
            }
            _ => Some(fresh_host_inference()),
        },
        "dict_of" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match &**inner {
                HostTypeTerm::Tuple(parts) if parts.len() == 2 => Some(HostTypeTerm::Dict(
                    Box::new(parts[0].clone()),
                    Box::new(parts[1].clone()),
                )),
                _ => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "dict_get" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(_, value)) => {
                Some(HostTypeTerm::Option(Box::new((**value).clone())))
            }
            _ => Some(fresh_host_inference()),
        },
        "dict_contains" => Some(HostTypeTerm::Bool),
        "dict_remove" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(key, value)) => Some(HostTypeTerm::Dict(
                Box::new((**key).clone()),
                Box::new((**value).clone()),
            )),
            _ => Some(fresh_host_inference()),
        },
        "dict_insert" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(key, value)) => Some(HostTypeTerm::Dict(
                Box::new((**key).clone()),
                Box::new((**value).clone()),
            )),
            _ => Some(fresh_host_inference()),
        },
        "dict_merge" => match (arg_tys.first(), arg_tys.get(1)) {
            (
                Some(HostTypeTerm::Dict(lhs_key, lhs_value)),
                Some(HostTypeTerm::Dict(rhs_key, rhs_value)),
            ) if **lhs_key == **rhs_key && **lhs_value == **rhs_value => Some(HostTypeTerm::Dict(
                Box::new((**lhs_key).clone()),
                Box::new((**lhs_value).clone()),
            )),
            _ => Some(fresh_host_inference()),
        },
        "dict_keys" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(key, _)) => Some(HostTypeTerm::List(Box::new((**key).clone()))),
            _ => Some(fresh_host_inference()),
        },
        "dict_values" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(_, value)) => {
                Some(HostTypeTerm::List(Box::new((**value).clone())))
            }
            _ => Some(fresh_host_inference()),
        },
        "dict_entries" => match arg_tys.first() {
            Some(HostTypeTerm::Dict(key, value)) => {
                Some(HostTypeTerm::List(Box::new(HostTypeTerm::Tuple(vec![
                    (**key).clone(),
                    (**value).clone(),
                ]))))
            }
            _ => Some(fresh_host_inference()),
        },
        "print" => Some(HostTypeTerm::Unit),
        "fail" => Some(HostTypeTerm::Never),
        "debug" => arg_tys.first().cloned(),
        "tensor_to_scalar" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor)) => Some(match tensor.precision {
                chelis_types::types::Prim::Bool => HostTypeTerm::Bool,
                chelis_types::types::Prim::Int8
                | chelis_types::types::Prim::Int32
                | chelis_types::types::Prim::Int64 => HostTypeTerm::Int64,
                _ => HostTypeTerm::Float64,
            }),
            _ => Some(HostTypeTerm::Float64),
        },
        // Issue #308: `HostTypeTerm::Float64` classifies BOTH f32 and f64
        // host scalars, so this coarse arm cannot recover the true
        // operand precision and defaults the float case to f32 (the
        // IR `Const` default for unsuffixed float literals). The
        // Deep-level `infer_app_expr_host_type` `scalar_to_tensor` arm
        // overrides this with the real precision (checker annotation
        // or explicit `cast(_, p)` target) whenever the Deep operand
        // is still visible; this fallback only decides when nothing
        // upstream knew better, and then both the C emit dispatch and
        // the consuming tensor-helper Load share the same f32 answer,
        // so storage width stays consistent.
        "scalar_to_tensor" => match arg_tys.first() {
            Some(&HostTypeTerm::Int64) => Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::Int64,
            })),
            Some(&HostTypeTerm::Bool) => Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::Bool,
            })),
            Some(&HostTypeTerm::Float64) => Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::F32,
            })),
            _ => None,
        },
        "to_tensor" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match &**inner {
                HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision))
                    if *precision != chelis_types::types::Prim::String =>
                {
                    Some(HostTypeTerm::Tensor(TensorType {
                        dims: vec![crate::dag::DimInfo::Named("list".to_string(), None)],
                        precision: *precision,
                    }))
                }
                HostTypeTerm::Scalar(HostPrecisionTerm::Variable(_)) => {
                    Some(fresh_host_inference())
                }
                _ => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "to_list" => match arg_tys.first() {
            Some(HostTypeTerm::Tensor(tensor)) if tensor.dims.len() == 1 => {
                let element_ty = match tensor.precision {
                    chelis_types::types::Prim::Bool => HostTypeTerm::Bool,
                    chelis_types::types::Prim::Int8
                    | chelis_types::types::Prim::Int16
                    | chelis_types::types::Prim::Int32
                    | chelis_types::types::Prim::Int64 => HostTypeTerm::Int64,
                    chelis_types::types::Prim::F16
                    | chelis_types::types::Prim::Bf16
                    | chelis_types::types::Prim::F32
                    | chelis_types::types::Prim::F64 => HostTypeTerm::Float64,
                    // The checked wrapper rejects this deferred dtype with
                    // `HostBuiltinInferenceError` before entering this
                    // implementation. Keep this arm non-defaulting if the
                    // match is edited independently in the future.
                    chelis_types::types::Prim::F8e4m3 => return None,
                    _ => fresh_host_inference(),
                };
                Some(HostTypeTerm::List(Box::new(element_ty)))
            }
            Some(HostTypeTerm::Tensor(_)) => Some(fresh_host_inference()),
            _ => Some(fresh_host_inference()),
        },
        "pad_sequences" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match &**inner {
                HostTypeTerm::List(nested) => match &**nested {
                    &HostTypeTerm::Int64 => Some(HostTypeTerm::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::Int64,
                    })),
                    &HostTypeTerm::Float64 => Some(HostTypeTerm::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::F32,
                    })),
                    _ => Some(fresh_host_inference()),
                },
                _ => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "pad_sequences_to" => match arg_tys.first() {
            Some(HostTypeTerm::List(inner)) => match &**inner {
                HostTypeTerm::List(nested) => match &**nested {
                    &HostTypeTerm::Int64 => Some(HostTypeTerm::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::Int64,
                    })),
                    &HostTypeTerm::Float64 => Some(HostTypeTerm::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::F32,
                    })),
                    _ => Some(fresh_host_inference()),
                },
                _ => Some(fresh_host_inference()),
            },
            _ => Some(fresh_host_inference()),
        },
        "read_file" => Some(HostTypeTerm::String),
        "write_file" => Some(HostTypeTerm::Unit),
        "read_lines" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::String))),
        "read_bytes" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::Int64))),
        "file_exists" => Some(HostTypeTerm::Bool),
        "list_dir" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::String))),
        "mmap_file" => Some(HostTypeTerm::MappedFile),
        "mmap_read" => Some(HostTypeTerm::List(Box::new(HostTypeTerm::Int64))),
        "mmap_len" => Some(HostTypeTerm::Int64),
        // Hull Phase 0a: `process_run(cmd, args) -> (exit_code, stdout, stderr)`.
        // Eval/test-only; the C/HIP build backends reject it before codegen
        // (see `host_program_uses_builtin` / `reject_eval_only_builtins_host`).
        "process_run" => Some(HostTypeTerm::Tuple(vec![
            HostTypeTerm::Int64,
            HostTypeTerm::String,
            HostTypeTerm::String,
        ])),
        _ => None,
    }
}

fn lookup_adt_ctor(
    program: &CheckedProgram,
    ctor_name: &str,
) -> Option<(String, Vec<HostTypeTerm>)> {
    lookup_adt_ctor_details(program, ctor_name).map(|(adt_name, fields)| {
        (
            adt_name,
            fields.into_iter().map(|field| field.ty).collect::<Vec<_>>(),
        )
    })
}

fn lookup_adt_ctor_details(
    program: &CheckedProgram,
    ctor_name: &str,
) -> Option<(String, Vec<HostAdtField>)> {
    lookup_adt_ctor_details_for_type(program, ctor_name, None)
}

fn lookup_adt_ctor_details_for_type(
    program: &CheckedProgram,
    ctor_name: &str,
    instantiated_ty: Option<&HostTypeTerm>,
) -> Option<(String, Vec<HostAdtField>)> {
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("deftype") {
            continue;
        }
        let kids = children(list);
        let Some(adt_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let subst = adt_type_substitution(children(list).get(1), adt_name, instantiated_ty);
        for variant in kids.iter().skip(2) {
            let Some(variant_list) = as_list(variant) else {
                continue;
            };
            if tag(variant_list) != Some("variant") {
                continue;
            }
            let variant_kids = children(variant_list);
            let Some(name) = variant_kids.first().and_then(symbol_name) else {
                continue;
            };
            if !terminal_name_matches(name, ctor_name) {
                continue;
            }
            let mut fields = Vec::new();
            for field in variant_kids.iter().skip(1) {
                if let Some(field_list) = as_list(field)
                    && tag(field_list) == Some("field")
                {
                    let field_kids = children(field_list);
                    if let Some(ty_expr) = field_kids.get(1) {
                        fields.push(HostAdtField {
                            name: field_kids.first().and_then(symbol_name).map(str::to_string),
                            ty: decode_host_type_or_raise(ty_expr, &subst),
                        });
                    }
                } else {
                    fields.push(HostAdtField {
                        name: None,
                        ty: decode_host_type_or_raise(field, &subst),
                    });
                }
            }
            return Some((adt_name.to_string(), fields));
        }
    }
    None
}

fn lookup_access_field(
    program: &CheckedProgram,
    base: &HostExpr,
    field_name: &str,
) -> Option<(usize, HostTypeTerm)> {
    match &base.kind {
        HostExprKind::AdtConstruct { ctor, .. } => {
            lookup_adt_ctor_details(program, ctor).and_then(|(_, fields)| {
                fields.iter().enumerate().find_map(|(index, field)| {
                    (field.name.as_deref() == Some(field_name)).then_some((index, field.ty.clone()))
                })
            })
        }
        _ => match host_expr_type(base) {
            HostTypeTerm::Adt(adt_name, args) => {
                lookup_adt_field_on_type(program, &adt_name, &args, field_name)
            }
            _ => None,
        },
    }
}

fn lookup_adt_field_on_type(
    program: &CheckedProgram,
    adt_name: &str,
    args: &[HostTypeTerm],
    field_name: &str,
) -> Option<(usize, HostTypeTerm)> {
    let mut found = None;
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("deftype") {
            continue;
        }
        let kids = children(list);
        if kids.first().and_then(symbol_name) != Some(adt_name) {
            continue;
        }
        let subst = adt_type_substitution(
            children(list).get(1),
            adt_name,
            Some(&HostTypeTerm::Adt(adt_name.to_string(), args.to_vec())),
        );
        for variant in kids.iter().skip(2) {
            let Some(variant_list) = as_list(variant) else {
                continue;
            };
            if tag(variant_list) != Some("variant") {
                continue;
            }
            for (index, field) in children(variant_list).iter().skip(1).enumerate() {
                let Some(field_list) = as_list(field) else {
                    continue;
                };
                if tag(field_list) != Some("field") {
                    continue;
                }
                if children(field_list).first().and_then(symbol_name) == Some(field_name) {
                    let ty = children(field_list)
                        .get(1)
                        .map(|expr| decode_host_type_or_raise(expr, &subst))
                        .unwrap_or_else(fresh_host_inference);
                    if let Some(existing) = &found
                        && existing != &(index, ty.clone())
                    {
                        return None;
                    }
                    found = Some((index, ty));
                }
            }
        }
    }
    found
}

fn adt_type_substitution(
    params_expr: Option<&Expr>,
    adt_name: &str,
    instantiated_ty: Option<&HostTypeTerm>,
) -> HashMap<String, HostTypeTerm> {
    let params = params_expr
        .and_then(as_list)
        .map(|list| {
            list.elements
                .iter()
                .filter_map(symbol_name)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let actuals = match instantiated_ty {
        Some(HostTypeTerm::Adt(name, args)) if terminal_name_matches(name, adt_name) => {
            args.clone()
        }
        _ => Vec::new(),
    };
    params.into_iter().zip(actuals).collect()
}

fn tag(list: &List) -> Option<&str> {
    list.elements.first().and_then(|expr| match expr {
        Expr::Atom(Atom::Symbol(tag), _) => Some(tag.as_str()),
        _ => None,
    })
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn as_list(expr: &Expr) -> Option<&List> {
    match expr {
        Expr::List(list, _) => Some(list),
        _ => None,
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

/// Collect every `(var {} <name>)` reference reachable in a Deep `expr`
/// into `out`. Issue #378: used to decide whether a top-level *value*
/// binding (a non-`fn` `def` body) is captured by a host-lane function
/// body and must therefore be materialized into `HostProgram::globals`
/// rather than dropped via `skip_for_lowered`. A bound parameter and a
/// captured global both appear as the same `(var ...)` form here; the
/// caller narrows to top-level binding names, so the over-approximation
/// is harmless (the C emitter's `captured_global_names` is the final
/// gate — an emitted global is only declared if a function actually
/// references it).
fn collect_deep_var_names(expr: &Expr, out: &mut HashSet<String>) {
    match expr {
        Expr::List(list, _) => {
            if tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                out.insert(name.to_string());
            }
            for child in &list.elements {
                collect_deep_var_names(child, out);
            }
        }
        Expr::MetaExpr(meta, _) => collect_deep_var_names(&meta.expr, out),
        _ => {}
    }
}

fn param_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.clone()),
        Expr::MetaExpr(meta, _) => param_name(&meta.expr),
        Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .or_else(|| children(list).first().and_then(symbol_name))
            .map(str::to_string),
        _ => None,
    }
}

fn param_host_type(expr: &Expr) -> Option<HostTypeTerm> {
    match expr {
        Expr::MetaExpr(meta, _) => expr_type(expr)
            .filter(|ty| !ty.is_unresolved())
            .or_else(|| param_host_type(&meta.expr)),
        Expr::List(_, _) => expr_type(expr).filter(|ty| !ty.is_unresolved()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DimInfo, RiscOp};
    use chelis_types::types::Prim;

    fn parse_and_check(src: &str) -> CheckedProgram {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let checked = chelis_types::check_ir_program(&exprs)
            .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"))
    }

    fn surf_check(src: &str) -> CheckedProgram {
        let decls = chelis_surf::parser::parse_str(src).expect("surf parse failed");
        let deep = chelis_surf::desugar::desugar_program(&decls);
        chelis_types::check_ir_program(&deep)
            .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors))
    }

    #[test]
    fn empty_to_tensor_uses_checked_default_before_concrete_resolution() {
        let checked = surf_check("result = numel(to_tensor([]))\n");
        let compiled = try_lower_compiled_program(&checked)
            .expect("the checked empty tensor default must resolve before codegen");
        assert!(compiled.host.is_some());
    }

    #[test]
    fn host_program_uses_builtin_detects_process_run() {
        // Hull subprocess exec: a global binding that applies process_run is
        // detected so the build backends can reject it before codegen.
        let checked = surf_check("result = process_run(\"echo\", [\"hi\"])\n");
        let compiled = lower_compiled_program(&checked);
        let host = compiled.host.expect("host program present");
        assert!(
            host_program_uses_builtin(&host, "process_run"),
            "host_program_uses_builtin must detect a process_run global binding"
        );
    }

    #[test]
    fn host_program_uses_builtin_is_false_without_process_run() {
        // Negative parity: a program that uses only file IO must not report
        // process_run usage, so the positive assertion is not vacuous.
        let checked = surf_check("contents = read_file(\"dataset.txt\")\n");
        let compiled = lower_compiled_program(&checked);
        let host = compiled.host.expect("host program present");
        assert!(
            !host_program_uses_builtin(&host, "process_run"),
            "host_program_uses_builtin must be false for a read_file-only program"
        );
    }

    /// chelis#336: restore the guard that a function-typed parameter survives
    /// as `HostTypeTerm::Fn` through `lower_compiled_program`. #331 deleted the
    /// only test asserting this (it relied on a reef-package HOF). A
    /// *same-module* HOF like `def apply(f, x) = f(x)` is inlined away, so a
    /// surviving fn-param needs a non-inlined HOF: here `apply_each` is kept
    /// as a real host function because it is called with a fn argument and
    /// uses the `map` combinator. The lowered `apply_each` must keep its
    /// `f: f32 -> f32` parameter typed `HostTypeTerm::Fn`, not collapsed to a
    /// value type.
    #[test]
    fn host_program_preserves_fn_typed_param_through_lowering() {
        let checked = surf_check(
            "def apply_each(f: f32 -> f32, xs: List[f32]) -> List[f32] = map(fn (v: f32) -> f(v), xs)\n\
             def double(x: f32) -> f32 = mul(x, 2.0)\n\
             out = apply_each(double, [1.0, 2.0, 3.0])\n",
        );
        let compiled = lower_compiled_program(&checked);
        let host = compiled.host.expect("host program present");
        let apply_each = host
            .functions
            .iter()
            .find(|f| f.name == "apply_each")
            .expect("apply_each must survive as a host function (not inlined)");
        let f_param = apply_each
            .params
            .iter()
            .find(|p| p.name == "f")
            .expect("apply_each must keep its `f` parameter");
        match &f_param.ty {
            ConcreteHostType::Function(params, ret) => {
                assert_eq!(params.len(), 1, "f takes one scalar arg, got {params:?}");
                assert!(
                    matches!(params[0], ConcreteHostType::Scalar(Prim::F32 | Prim::F64)),
                    "f's arg must be a scalar, got {:?}",
                    params[0]
                );
                assert!(
                    matches!(**ret, ConcreteHostType::Scalar(Prim::F32 | Prim::F64)),
                    "f's return must be a scalar, got {ret:?}"
                );
            }
            other => {
                panic!("fn-typed param `f` must lower to ConcreteHostType::Function, got {other:?}")
            }
        }
    }

    /// chelis#336 negative parity: a program with no higher-order parameter
    /// must produce no `HostTypeTerm::Fn` parameter, so the positive guard above
    /// is not vacuously satisfied by some unrelated fn-typed param.
    #[test]
    fn host_program_has_no_fn_typed_param_without_hof() {
        let checked = surf_check(
            "def double(x: f32) -> f32 = mul(x, 2.0)\n\
             out = double(2.0)\n",
        );
        let compiled = lower_compiled_program(&checked);
        let host = compiled.host.expect("host program present");
        let has_fn_param = host
            .functions
            .iter()
            .flat_map(|f| f.params.iter())
            .any(|p| matches!(p.ty, ConcreteHostType::Function(_, _)));
        assert!(
            !has_fn_param,
            "a non-HOF program must not lower any HostTypeTerm::Fn parameter"
        );
    }

    // ── Issue #308: scalar_to_tensor operand-precision plumbing ──
    //
    // `HostTypeTerm::Float64` collapses f32 and f64, so the Deep-level
    // `infer_app_expr_host_type` must recover the operand precision
    // before the coarse arg-ty fallback erases it to f32.

    fn parse_deep_app(src: &str) -> Expr {
        chelis_deep::parser::parse_str(src)
            .expect("parse failed")
            .into_iter()
            .next()
            .expect("one expr")
    }

    fn infer_scalar_to_tensor_host_type(arg_src: &str) -> Option<HostTypeTerm> {
        let app = parse_deep_app(&format!("(app {{}} (var {{}} scalar_to_tensor) {arg_src})"));
        let Expr::List(list, _) = &app else {
            panic!("app expr must be a list");
        };
        // Empty checked program: the arm under test must not depend on
        // program-level lookups for the precision recovery.
        let program = surf_check("unrelated = 1\n");
        infer_app_expr_host_type(list, &program, &HashMap::new())
    }

    #[test]
    fn scalar_to_tensor_infers_f64_from_cast_target() {
        // The #308 reproducer shape: `scalar_to_tensor(cast(1.1, f64))`
        // with NO checker annotation on the app or the cast — the cast
        // target alone must drive the result precision.
        let inferred = infer_scalar_to_tensor_host_type(
            "(cast {} (lit {type: (t-prim {} f32)} 1.1) (t-prim {} f64))",
        );
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: Prim::F64,
            })),
            "scalar_to_tensor(cast(_, f64)) must infer a rank-0 f64 tensor",
        );
    }

    #[test]
    fn scalar_to_tensor_infers_f64_from_checker_annotation() {
        let inferred = infer_scalar_to_tensor_host_type("(var {type: (t-prim {} f64)} c)");
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: Prim::F64,
            })),
            "scalar_to_tensor of an f64-annotated operand must infer a rank-0 f64 tensor",
        );
    }

    #[test]
    fn scalar_to_tensor_keeps_f32_default_for_f32_operand() {
        // Negative parity (pins the #300/#306 f32 path): an f32 operand
        // — whether via cast target or bare default literal — must keep
        // the rank-0 f32 result so `chelis_scalar_tensor_from_f32`
        // storage and the consuming helper's f32 read stay paired.
        for arg_src in [
            "(cast {} (lit {type: (t-prim {} f32)} 2.5) (t-prim {} f32))",
            "(lit {type: (t-prim {} f32)} 2.5)",
        ] {
            let inferred = infer_scalar_to_tensor_host_type(arg_src);
            assert_eq!(
                inferred,
                Some(HostTypeTerm::Tensor(TensorType {
                    dims: vec![],
                    precision: Prim::F32,
                })),
                "scalar_to_tensor({arg_src}) must keep the f32 default",
            );
        }
    }

    #[test]
    fn scalar_to_tensor_float_recovery_does_not_hijack_integer_operands() {
        // Negative parity: the #308 float-precision recovery must not
        // claim integer operands. At this Deep-expr layer a
        // cast-wrapped int infers `None` (pre-#308 behavior:
        // `expr_host_type` does not see through `cast`, so the coarse
        // fallback gets `Unknown` and abstains); the Int64 tensor
        // typing happens post-lowering via `host_expr_type` on the
        // lowered operand and the `infer_builtin_host_type_from_arg_tys`
        // Int64 arm.
        let inferred = infer_scalar_to_tensor_host_type(
            "(cast {} (lit {type: (t-prim {} int32)} 3) (t-prim {} int32))",
        );
        assert_eq!(
            inferred, None,
            "integer operands must fall through unchanged (no float hijack)",
        );
        // The coarse arm still owns the lowered-lane integer answer.
        assert_eq!(
            infer_builtin_host_type_from_arg_tys("scalar_to_tensor", &[HostTypeTerm::Int64]),
            Ok(Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![],
                precision: Prim::Int64,
            }))),
        );
    }

    // ── chelis#631: host-lane concat result typing ──
    //
    // The host lane must never carry the ELEMENT's extent on the concat
    // axis: `chelis_tensor_concat` sizes that axis at run time, and a
    // baked element extent aborts the binary at the runtime-dim guard.

    fn rank2_element() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(1), DimInfo::Lit(2)],
            precision: Prim::F32,
        }
    }

    fn anon_dim() -> DimInfo {
        DimInfo::Named("*".to_string(), None)
    }

    fn infer_concat_host_type(axis_src: &str) -> Option<HostTypeTerm> {
        let app = parse_deep_app(&format!(
            "(app {{}} (var {{}} concat) (var {{}} rows) {axis_src})"
        ));
        let Expr::List(list, _) = &app else {
            panic!("app expr must be a list");
        };
        let program = surf_check("unrelated = 1\n");
        let mut scope = HashMap::new();
        scope.insert(
            "rows".to_string(),
            HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(rank2_element()))),
        );
        infer_app_expr_host_type(list, &program, &scope)
    }

    #[test]
    fn concat_app_expr_host_type_wildcards_concat_axis_only() {
        // Literal axis 0 (bare and cast-wrapped, the canonical spelling):
        // the concat axis is anon, the trailing element extent survives.
        for axis_src in ["(lit {} 0)", "(cast {} (lit {} 0) (t-prim {} int32))"] {
            let inferred = infer_concat_host_type(axis_src);
            assert_eq!(
                inferred,
                Some(HostTypeTerm::Tensor(TensorType {
                    dims: vec![anon_dim(), DimInfo::Lit(2)],
                    precision: Prim::F32,
                })),
                "concat(rows, {axis_src}) must wildcard only axis 0",
            );
        }
    }

    #[test]
    fn concat_app_expr_host_type_negative_axis_indexes_from_end() {
        let inferred = infer_concat_host_type("(lit {} -1)");
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(1), anon_dim()],
                precision: Prim::F32,
            })),
            "concat(rows, -1) must wildcard the LAST axis and keep axis 0",
        );
    }

    #[test]
    fn concat_app_expr_host_type_unknown_axis_wildcards_all_axes() {
        // Negative parity: a non-literal axis leaves no per-axis claim.
        let inferred = infer_concat_host_type("(var {} ax)");
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![anon_dim(), anon_dim()],
                precision: Prim::F32,
            })),
            "a runtime concat axis must wildcard every axis",
        );
    }

    #[test]
    fn concat_arg_tys_fallback_wildcards_all_axes() {
        // The arg-tys-only lane cannot see the axis value: pre-chelis#631
        // it returned the element type VERBATIM (dims [1, 2]) — the baked
        // extent that aborted guarded forward binaries.
        let inferred = infer_builtin_host_type_from_arg_tys(
            "concat",
            &[
                HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(rank2_element()))),
                HostTypeTerm::Int64,
            ],
        );
        assert_eq!(
            inferred,
            Ok(Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![anon_dim(), anon_dim()],
                precision: Prim::F32,
            }))),
        );
    }

    #[test]
    fn concat_host_expr_lane_peels_cast_wrapped_axis() {
        // infer_builtin_host_type sees HostExpr args: a cast-wrapped int
        // axis still selects the single concat axis.
        let list_arg = HostExpr {
            kind: HostExprKind::Var(
                "rows".to_string(),
                HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(rank2_element()))),
            ),
            span_id: None,
            merged_spans: Vec::new(),
        };
        let axis_arg = HostExpr {
            kind: HostExprKind::Builtin {
                name: "cast".to_string(),
                args: vec![HostExpr {
                    kind: HostExprKind::Int(0),
                    span_id: None,
                    merged_spans: Vec::new(),
                }],
                ty: HostTypeTerm::Int64,
            },
            span_id: None,
            merged_spans: Vec::new(),
        };
        let inferred = infer_builtin_host_type("concat", &[list_arg, axis_arg]);
        assert_eq!(
            inferred,
            Some(HostTypeTerm::Tensor(TensorType {
                dims: vec![anon_dim(), DimInfo::Lit(2)],
                precision: Prim::F32,
            })),
        );
    }

    /// E2 (WS-A0 RT-1 fixup): the `to_list` host classification arm
    /// returns a typed error for f8e4m3 rather than panicking or silently
    /// classifying the deferred dtype as Float64.
    #[test]
    fn to_list_classifier_rejects_f8e4m3_tensor_per_spec_1_1_1() {
        let f8_tensor = HostTypeTerm::Tensor(crate::dag::TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F8e4m3,
        });
        let error = infer_builtin_host_type_from_arg_tys("to_list", &[f8_tensor])
            .expect_err("the deferred dtype must be a typed inference error");
        assert!(error.to_string().contains("f8e4m3 is deferred"));
    }

    #[test]
    fn invalid_builtin_shape_is_not_an_inference_variable() {
        let error = infer_builtin_host_type_from_arg_tys(
            "sum",
            &[HostTypeTerm::List(Box::new(HostTypeTerm::Int64))],
        )
        .expect_err("sum over a list is invalid, not underconstrained");
        assert_eq!(error.builtin, "sum");
        assert!(error.requirement.contains("must be a tensor"));
    }

    #[test]
    fn genuinely_underconstrained_tuple_get_keeps_a_fresh_identity() {
        let first = infer_builtin_host_type_from_arg_tys(
            "tuple-get",
            &[HostTypeTerm::Tuple(vec![HostTypeTerm::Int64])],
        )
        .expect("the arg-types-only lane lacks the index, but is not invalid")
        .expect("tuple-get is a known builtin");
        let second = infer_builtin_host_type_from_arg_tys(
            "tuple-get",
            &[HostTypeTerm::Tuple(vec![HostTypeTerm::Int64])],
        )
        .expect("the arg-types-only lane lacks the index, but is not invalid")
        .expect("tuple-get is a known builtin");
        assert!(matches!(first, HostTypeTerm::InferenceVariable(_)));
        assert!(matches!(second, HostTypeTerm::InferenceVariable(_)));
        assert_ne!(first, second, "fresh holes must retain distinct identity");
    }

    #[test]
    fn named_tensor_entry_inserts_copy_for_consuming_fanout() {
        let checked = parse_and_check(
            r#"
                (def {} consume
                  (fn {type: (t-fn {}
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                    (params {}
                      (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                    (realize {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                      (var {} x))))
                (def {} double_it
                  (fn {type: (t-fn {}
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                    (params {}
                      (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                    (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                      (var {} add)
                      (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                        (var {} consume)
                        (var {} x))
                      (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                        (var {} consume)
                        (var {} x)))))
            "#,
        );
        let dag = lower_named_tensor_entry_dag(&checked, "double_it").expect("lower entry");
        let copy_count = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Copy))
            .count();
        assert_eq!(copy_count, 1, "{:?}", dag.nodes());
    }

    #[test]
    fn named_entry_dag_accepts_scalar_numeric_params() {
        let checked = parse_and_check(
            r#"
                (def {} add_scalar
                  (fn {type: (t-fn {}
                                (t-prim {} f32)
                                (t-prim {} f32)
                                (t-prim {} f32))}
                    (params {}
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (app {type: (t-prim {} f32)}
                      (var {} add)
                      (var {} x)
                      (var {} y))))
            "#,
        );

        let dag = lower_named_tensor_entry_dag(&checked, "add_scalar").expect("lower scalar entry");
        let root = dag.roots().first().and_then(|id| dag.get(*id)).unwrap();

        assert_eq!(root.op, RiscOp::Add, "{:?}", dag.nodes());
        let scalar_loads = dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.op, RiscOp::Load { .. })
                    && node.output_type.dims.is_empty()
                    && node.output_type.precision == Prim::F32
            })
            .count();
        assert_eq!(scalar_loads, 2, "{:?}", dag.nodes());
    }

    // ── WS-4: declared float width survives entry-param lowering ──
    //
    // `tensor_type_from_host_input` previously mapped every
    // `HostTypeTerm::Float64` (which collapsed both f32 and f64) to
    // `Prim::F32`, silently truncating an f64 scalar entry parameter to
    // 4-byte storage. The declared `f32`/`f64` width now threads through
    // `parse_host_type_with_subst` so each lowers to a `Load` of its own
    // precision.

    #[test]
    fn named_entry_dag_lowers_f64_scalar_params_to_f64_loads() {
        let checked = parse_and_check(
            r#"
                (def {} add_scalar
                  (fn {type: (t-fn {}
                                (t-prim {} f64)
                                (t-prim {} f64)
                                (t-prim {} f64))}
                    (params {}
                      (x {type: (t-prim {} f64)})
                      (y {type: (t-prim {} f64)}))
                    (app {type: (t-prim {} f64)}
                      (var {} add)
                      (var {} x)
                      (var {} y))))
            "#,
        );

        let dag =
            lower_named_tensor_entry_dag(&checked, "add_scalar").expect("lower f64 scalar entry");
        let root = dag.roots().first().and_then(|id| dag.get(*id)).unwrap();
        assert_eq!(root.op, RiscOp::Add, "{:?}", dag.nodes());

        let f64_loads = dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.op, RiscOp::Load { .. })
                    && node.output_type.dims.is_empty()
                    && node.output_type.precision == Prim::F64
            })
            .count();
        assert_eq!(
            f64_loads,
            2,
            "an f64 scalar entry param must lower to an f64 Load, not f32: {:?}",
            dag.nodes()
        );
    }

    #[test]
    fn named_entry_dag_does_not_downgrade_f64_scalar_params_to_f32() {
        // Negative twin: the historical bug. NO scalar Load produced for
        // a declared-f64 entry param may carry `Prim::F32`.
        let checked = parse_and_check(
            r#"
                (def {} add_scalar
                  (fn {type: (t-fn {}
                                (t-prim {} f64)
                                (t-prim {} f64)
                                (t-prim {} f64))}
                    (params {}
                      (x {type: (t-prim {} f64)})
                      (y {type: (t-prim {} f64)}))
                    (app {type: (t-prim {} f64)}
                      (var {} add)
                      (var {} x)
                      (var {} y))))
            "#,
        );

        let dag =
            lower_named_tensor_entry_dag(&checked, "add_scalar").expect("lower f64 scalar entry");
        let f32_scalar_loads = dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.op, RiscOp::Load { .. })
                    && node.output_type.dims.is_empty()
                    && node.output_type.precision == Prim::F32
            })
            .count();
        assert_eq!(
            f32_scalar_loads,
            0,
            "an f64 scalar entry param must NOT be silently downgraded to an f32 Load: {:?}",
            dag.nodes()
        );
    }

    #[test]
    fn parse_host_type_preserves_declared_float_width() {
        // Unit-level pin on the precision-threading site: the `t-prim`
        // syntax for f32 and f64 must map to distinct host scalar
        // variants so the entry-param boundary can recover the width.
        let f32_ty = parse_deep_app("(t-prim {} f32)");
        assert_eq!(
            decode_host_type_with_subst(&f32_ty, &HashMap::new()),
            Ok(HostTypeTerm::Float32),
        );

        let f64_ty = parse_deep_app("(t-prim {} f64)");
        assert_eq!(
            decode_host_type_with_subst(&f64_ty, &HashMap::new()),
            Ok(HostTypeTerm::Float64),
        );
    }

    #[test]
    fn host_input_tensor_round_trip_preserves_float_width() {
        // The two boundary helpers must agree: a declared f32/f64 host
        // scalar lowers to a rank-0 tensor of the matching precision, and
        // the reverse map restores the same host width.
        assert_eq!(
            tensor_type_from_host_input(&HostTypeTerm::Float32),
            Some(TensorType {
                dims: vec![],
                precision: Prim::F32,
            }),
        );
        assert_eq!(
            tensor_type_from_host_input(&HostTypeTerm::Float64),
            Some(TensorType {
                dims: vec![],
                precision: Prim::F64,
            }),
        );
        assert_eq!(
            host_type_from_tensor_input(&TensorType {
                dims: vec![],
                precision: Prim::F32,
            }),
            HostTypeTerm::Float32,
        );
        assert_eq!(
            host_type_from_tensor_input(&TensorType {
                dims: vec![],
                precision: Prim::F64,
            }),
            HostTypeTerm::Float64,
        );
    }

    #[test]
    fn call_graph_recursion_detection_does_not_mark_every_function_recursive() {
        let graph = HashMap::from([
            ("plain".to_string(), HashSet::from(["leaf".to_string()])),
            ("leaf".to_string(), HashSet::new()),
            (
                "self_rec".to_string(),
                HashSet::from(["self_rec".to_string()]),
            ),
            ("mut_a".to_string(), HashSet::from(["mut_b".to_string()])),
            ("mut_b".to_string(), HashSet::from(["mut_a".to_string()])),
        ]);

        let recursive = recursive_top_level_fn_names_from_graph(&graph);

        assert!(
            !recursive.contains("plain"),
            "ordinary top-level callers must not be classified as recursive"
        );
        assert!(
            !recursive.contains("leaf"),
            "leaf functions must not be classified as recursive"
        );
        assert!(
            recursive.contains("self_rec"),
            "direct self-recursion must still be detected"
        );
        assert!(
            recursive.contains("mut_a") && recursive.contains("mut_b"),
            "mutual recursion must still be detected"
        );
    }

    #[test]
    fn tensor_helper_exp_body_keeps_exp_root() {
        let checked = parse_and_check(
            r#"
                (def {} softplus
                  (fn {type: (t-fn {}
                                (t-tensor {} (d-lit {} 4) (t-prim {} f32))
                                (t-tensor {} (d-lit {} 4) (t-prim {} f32)))}
                    (params {}
                      (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
                    (app {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}
                      (var {} exp)
                      (var {} x))))
            "#,
        );
        let defs = collect_program_defs(checked.exprs());
        let body = lookup_program_def(&defs, "softplus").unwrap();
        let fn_list = as_list(body).unwrap();
        let body = children(fn_list).get(1).unwrap();
        let mut scope = HashMap::new();
        scope.insert(
            "x".to_string(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            }),
        );
        let expected = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let dag = lower_tensor_helper_dag(body, &checked, &scope, &expected).expect("helper dag");
        let root = dag.roots().first().and_then(|id| dag.get(*id)).unwrap();
        assert_eq!(root.op, RiscOp::Exp, "{:?}", dag.nodes());
    }

    #[test]
    fn tensor_helper_hoists_host_lane_tensor_args_with_f32_type() {
        let checked = parse_and_check(
            r#"
                (defsig {}
                  jac_row
                  (t-fn {}
                    (t-fn {}
                      (t-tensor {} (d-var {} n) (t-prim {} f32))
                      (t-prim {} f32)
                      (t-prim {} f32)
                      (t-prim {} f32))
                    (t-tensor {} (d-var {} n) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-tensor {} (d-var {} n) (t-prim {} f32))))
                (def {}
                  jac_row
                  (fn {}
                    (params {}
                      (model {type: (t-fn {}
                                       (t-tensor {} (d-var {} n) (t-prim {} f32))
                                       (t-prim {} f32)
                                       (t-prim {} f32)
                                       (t-prim {} f32))})
                      (theta {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        target
                        (fn {}
                          (params {}
                            (theta_local {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
                          (app {} (var {} model) (var {} theta_local) (var {} x) (var {} y))))
                      (app {}
                        (grad {wrt: (var {} theta_local)}
                          (var {} target)
                          (lit {type: (t-prim {} int32)} 0))
                        (var {} theta)))))
                (defsig {}
                  lm_model
                  (t-fn {}
                    (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-prim {} f32)))
                (def {}
                  lm_model
                  (fn {}
                    (params {}
                      (theta {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        y_hat
                        (if {}
                          (app {}
                            (var {} lt)
                            (var {} x)
                            (cast {} (lit {type: (t-prim {} f32)} 0.0) (t-prim {} f32)))
                          (app {}
                            (var {} tensor_to_scalar)
                            (app {}
                              (var {} sum)
                              (copy {} (var {} theta))
                              (lit {type: (t-prim {} int32)} 0)))
                          (app {}
                            (var {} add)
                            (app {}
                              (var {} tensor_to_scalar)
                              (app {}
                                (var {} sum)
                                (copy {} (var {} theta))
                                (lit {type: (t-prim {} int32)} 0)))
                            (var {} x))))
                      (app {} (var {} sub) (var {} y) (var {} y_hat)))))
                (def {}
                  out
                  (app {}
                    (var {} jac_row)
                    (var {} lm_model)
                    (app {}
                      (var {} to_tensor)
                      (app {}
                        (var {} Cons)
                        (lit {type: (t-prim {} f32)} 1.0)
                        (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 2.0) (var {} Nil))))
                    (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} f32))
                    (cast {} (lit {type: (t-prim {} f32)} 3.0) (t-prim {} f32))))
            "#,
        );
        let lowered = top_level_lowering_map(checked.exprs(), checked.type_env());
        let host = lower_host_program(&checked, &lowered);
        let out_binding = host
            .globals
            .iter()
            .find(|binding| binding.name == "out")
            .expect("out host binding");
        let (bindings, body) = match &out_binding.value.kind {
            HostExprKind::Let { bindings, body, .. } => (bindings, body.as_ref()),
            other => panic!("expected out to hoist host-lane tensor arg into let, got {other:?}"),
        };
        let theta_binding = bindings
            .iter()
            .find(|binding| binding.name.starts_with("__host_tensor_arg_"))
            .unwrap_or_else(|| {
                panic!(
                    "theta temp binding missing; hoisted bindings were {:?}",
                    bindings
                        .iter()
                        .map(|binding| (&binding.name, &binding.ty))
                        .collect::<Vec<_>>()
                )
            });
        match &theta_binding.ty {
            HostTypeTerm::Tensor(tensor) => assert_eq!(tensor.precision, Prim::F32),
            other => panic!("expected hoisted theta binding to be tensor-typed, got {other:?}"),
        }
        let helper_index = match &body.kind {
            HostExprKind::TensorCall { helper, .. } => *helper,
            other => panic!("expected hoisted body to call tensor helper, got {other:?}"),
        };
        let helper = host
            .global_tensor_helpers
            .get(helper_index)
            .expect("helper index in range");
        assert!(
            helper
                .inputs
                .iter()
                .any(|input| input.name == theta_binding.name),
            "helper inputs should reference hoisted tensor temp: {:?}",
            helper
                .inputs
                .iter()
                .map(|input| (&input.name, &input.ty))
                .collect::<Vec<_>>()
        );
        assert!(
            helper.inputs.iter().all(|input| input.name != "to_tensor"),
            "helper inputs must not contain raw `to_tensor` load: {:?}",
            helper
                .inputs
                .iter()
                .map(|input| (&input.name, &input.ty))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn host_type_parser_unwraps_borrowed_tensor_refs() {
        let expr = chelis_deep::parser::parse_str(
            "(t-ref {} (t-tensor {} (d-name {} n) (t-prim {} f32)))",
        )
        .expect("parse ref type")
        .into_iter()
        .next()
        .expect("one type expr");

        match decode_host_type(&expr).expect("borrowed tensor type decodes") {
            HostTypeTerm::Tensor(tensor) => {
                assert_eq!(
                    tensor.dims,
                    vec![crate::dag::DimInfo::Named("n".into(), None)]
                );
                assert_eq!(tensor.precision, Prim::F32);
            }
            other => panic!("expected borrowed tensor ref to parse as tensor, got {other:?}"),
        }
    }

    #[test]
    fn tensor_helper_actualization_merges_matmul_synthetic_expand_dims() {
        use crate::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};

        let batch = DimInfo::Named("batch".into(), None);
        let in_dim = DimInfo::Named("in_dim".into(), None);
        let out_dim = DimInfo::Named("out_dim".into(), None);
        let d417 = DimInfo::Named("d417".into(), None);
        let d420 = DimInfo::Named("d420".into(), None);
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![batch.clone(), in_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let w = dag.add_node(
            RiscOp::Load { name: "w".into() },
            vec![],
            TensorType {
                dims: vec![in_dim.clone(), out_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let expanded_x = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: DimExpr::Sym("d420".into()),
            },
            vec![x],
            TensorType {
                dims: vec![batch.clone(), in_dim.clone(), d420.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let expanded_w = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Sym("d417".into()),
            },
            vec![w],
            TensorType {
                dims: vec![d417, in_dim.clone(), out_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let product = dag.add_node(
            RiscOp::Mul,
            vec![expanded_x, expanded_w],
            TensorType {
                dims: vec![batch.clone(), in_dim.clone(), d420],
                precision: Prim::F32,
            },
            None,
        );
        let root = dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![product],
            TensorType {
                dims: vec![batch.clone(), out_dim.clone()],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(root);

        let mut scope = HashMap::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![batch.clone(), in_dim.clone()],
                precision: Prim::F32,
            }),
        );
        scope.insert(
            "w".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![in_dim.clone(), out_dim.clone()],
                precision: Prim::F32,
            }),
        );

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        for node in actualized.nodes() {
            assert!(
                !node.output_type.dims.iter().any(|dim| {
                    matches!(dim, DimInfo::Named(name, None) if name == "d417" || name == "d420")
                }),
                "node retained synthetic dims: {node:?}"
            );
        }
    }

    /// chelis#345 (op-internal half): actualization rewrote OUTPUT
    /// types via `replace_node` but left `node.op` untouched, so an
    /// `Expand { size: Sym("dN") }` kept the stale checker-minted name
    /// after its output dim had been rewritten to the user-facing
    /// symbol. The Bucket 4d sweep in `dag.rs::symbolic_occurrences`
    /// panics on exactly that mixed state (no Load declares `dN`), and
    /// before the fix the `grad(residual, wrt=(theta))` host-wrapper
    /// canary in `chelis-cli/tests/cli.rs`
    /// (`build_c_tensor_grad_lm_style_mixed_scalar_tensor_args_builds`)
    /// tripped it. Pin that op-internal fields are renamed in lockstep
    /// with output dims — and, negative parity, that user-facing
    /// (non-`dN`) sizes are left alone.
    #[test]
    fn tensor_helper_actualization_rewrites_op_internal_expand_sizes() {
        use crate::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};

        let n = DimInfo::Named("n".into(), None);
        let d47 = DimInfo::Named("d47".into(), None);
        let mut dag = Dag::new();
        // Load typed with the minted alias; the scope knows the
        // user-facing symbol.
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![d47.clone()],
                precision: Prim::F32,
            },
            None,
        );
        // Scalar upstream gradient, as the Sum adjoint produces.
        let g = dag.add_node(
            RiscOp::Load { name: "g".into() },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        // The Sum adjoint's expand-back: size and output dim both carry
        // the minted alias.
        let expanded_g = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Sym("d47".into()),
            },
            vec![g],
            TensorType {
                dims: vec![d47.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let root = dag.add_node(
            RiscOp::Mul,
            vec![expanded_g, x],
            TensorType {
                dims: vec![d47],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(root);

        let mut scope = HashMap::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![n.clone()],
                precision: Prim::F32,
            }),
        );

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        let expand = actualized.get(expanded_g).expect("expand node");
        assert_eq!(
            expand.output_type.dims,
            vec![n],
            "expand output dim must actualize to the user-facing symbol"
        );
        assert_eq!(
            expand.op,
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Sym("n".into()),
            },
            "op-internal Expand size must be renamed in lockstep with \
             the output dim (chelis#345); a stale `d47` is an undeclared \
             identifier downstream"
        );
        // The whole helper must satisfy the Bucket 4d guard: no
        // symbolic reference (output OR op-internal) without a
        // declaring Load.
        let _ = crate::dag::symbolic_occurrences(&actualized);
    }

    /// Negative parity for the rename: user-facing (non-minted) Expand
    /// sizes must survive actualization untouched.
    #[test]
    fn tensor_helper_actualization_leaves_user_facing_expand_sizes_alone() {
        use crate::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};

        let batch = DimInfo::Named("batch".into(), None);
        let mut dag = Dag::new();
        let g = dag.add_node(
            RiscOp::Load { name: "g".into() },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        let expanded = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Sym("batch".into()),
            },
            vec![g],
            TensorType {
                dims: vec![batch.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![batch.clone()],
                precision: Prim::F32,
            },
            None,
        );
        let root = dag.add_node(
            RiscOp::Mul,
            vec![expanded, x],
            TensorType {
                dims: vec![batch],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(root);

        let mut scope = HashMap::new();
        scope.insert(
            "x".into(),
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Named("batch".into(), None)],
                precision: Prim::F32,
            }),
        );

        let actualized = actualize_tensor_helper_types(&dag, &scope);
        let expand = actualized.get(expanded).expect("expand node");
        assert_eq!(
            expand.op,
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Sym("batch".into()),
            },
            "user-facing symbolic sizes are not synthetic and must not \
             be rewritten"
        );
    }
}
