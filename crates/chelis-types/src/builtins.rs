//! Built-in function signatures for RISC primitives and derived operations.
//!
//! Signature templates:
//! - tensor_binop: ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D,p]
//! - tensor_unop:  ∀D,p. tensor[D,p] → tensor[D,p]
//! - cmplt:        ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D, bool]
//! - logical_binop: ∀D. (tensor[D,bool], tensor[D,bool]) → tensor[D,bool]
//! - logical_unop:  ∀D. tensor[D,bool] → tensor[D,bool]

use crate::adt::{AdtDef, AdtRegistry, VariantInfo};
use crate::env::Env;
use crate::types::*;

pub const BUILTIN_NAMES: &[&str] = &[
    "add",
    "mul",
    "max_elem",
    "neg",
    "recip",
    "exp",
    "log",
    "sin",
    "sqrt",
    "cos",
    "tan",
    "atan",
    "abs",
    "floor",
    "ceil",
    "round",
    "uniform_like",
    "cmplt",
    "sub",
    "div",
    "floor_div",
    "trunc_div",
    "mod",
    "eq",
    "neq",
    "lt",
    "gt",
    "lte",
    "gte",
    "bitand",
    "bitor",
    "bitxor",
    "shl",
    "shr",
    "and",
    "or",
    "not",
    "relu",
    "sigmoid",
    "tanh",
    "silu",
    "gelu",
    "softmax",
    "mean",
    "matmul",
    "min_elem",
    "layer_norm",
    "conv",
    "sum",
    "count",
    "max_reduce",
    "min_reduce",
    "prod_reduce",
    "argmax_reduce",
    "argmin_reduce",
    // §2.3.1 strided windowed reduction (Valid padding). One Surf
    // builtin per reducer; the IR collapses them to a single
    // `RiscOp::ReduceWindow` with a `ReduceWindowKind` discriminator.
    "reduce_window_max",
    "reduce_window_min",
    "reduce_window_sum",
    "reduce_window_mean",
    "reshape",
    "permute",
    "expand",
    "insert",
    "pad",
    "shrink",
    "stride",
    "print",
    "fail",
    "debug",
    "test_assert",
    "test_assert_eq",
    "test_assert_close_tensor",
    "test_assert_eq_tensor",
    "char_code",
    "char_from_code",
    "string_len",
    "string_concat",
    "string_slice",
    "string_contains",
    "string_starts_with",
    "string_ends_with",
    "string_trim",
    "to_string",
    "to_int",
    "to_float",
    "rank",
    "shape",
    "numel",
    "tensor_to_scalar",
    "scalar_to_tensor",
    "len",
    "index",
    "append",
    "concat",
    "take",
    "skip",
    "chunk",
    "range",
    "map",
    "filter",
    "fold",
    "scan",
    "tensor_scan",
    "partition",
    "flat_map",
    "flatten",
    "zip",
    "enumerate",
    "dict_of",
    "dict_get",
    "dict_contains",
    "dict_remove",
    "dict_insert",
    "dict_merge",
    "dict_keys",
    "dict_values",
    "dict_entries",
    "to_tensor",
    "to_list",
    "pad_sequences",
    "pad_sequences_to",
    "read_file",
    "write_file",
    "read_lines",
    "read_bytes",
    "file_exists",
    "list_dir",
    "mmap_file",
    "mmap_read",
    "mmap_len",
    "process_run",
    "round_to",
    // Host-lane CSV I/O (chelis#903): RFC-4180-ish parse/serialize plus
    // column accessors over List[Dict[string,string]].
    // Eval-only (`chelis_ir::host::EVAL_ONLY_HOST_BUILTINS`).
    "parse_csv",
    "to_csv",
    "csv_f64s",
    "csv_ints",
    "csv_strs",
    "csv_nrows",
    "csv_cols",
    "csv_f64",
    "csv_int",
    "csv_str",
    "einsum",
    "split",
    "gather",
    "scatter",
    "scatter_replace",
    "scatter_elements",
    "where",
    "cumsum",
    "sort",
    "diagonal",
    "trace",
    "clamp",
    // Linearity, not a container operation: the explicit one-argument
    // consume of [05-OP-67], paired with the `copy` keyword. The list
    // slice that once shared this name is `skip` ([05-OP-54]).
    "drop",
];

/// Zero-based argument slots, after the callee, whose values name tensor
/// axes and therefore carry the [05-DIM-3] `i32` contract.
///
/// This is semantic registration, not a name heuristic. A new axis-taking
/// builtin must select a layout here; inference consults this table before
/// the operation-specific shape rule. `concat` is registered but screened
/// in its overloaded tensor-list arm because list concatenation uses the
/// same surface name with no axis argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisArgumentLayout {
    NoAxes,
    Fixed(&'static [usize]),
    VariadicFrom(usize),
}

/// Target-independent capability domain selected by a builtin declaration.
///
/// This is prerequisite discovery metadata for chelis#1294, not a Phase 4C
/// capability cell. It contains no support status, backend disposition, or
/// fallback. Every [`BuiltinDecl`] states its exact domains so the atom-closure
/// oracle can derive its source universe without a parallel allowlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BuiltinSemanticDomain {
    Numeric,
    Container,
    Boundary,
}

/// Closed, builtin-owned sibling cases needed by the pre-4C atom-closure
/// oracle. Numeric-only declarations have no sibling cases. A case is paired
/// with its exact [`BuiltinSemanticDomain`] by [`BuiltinSiblingCaseDecl`].
///
/// These variants identify checked application shapes; they are not backend
/// implementation receipts. Phase 4C consumes them when it authors the
/// sibling cells, but chelis#1294 owns making the discovery universe explicit
/// first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BuiltinSiblingCaseId {
    PrintRecursive,
    FailString,
    DebugRecursive,
    TestAssertBool,
    TestAssertEq,
    TestAssertEqTensor,
    TestAssertCloseTensor,
    ReadFile,
    WriteFile,
    ReadLines,
    ReadBytes,
    FileExists,
    ListDir,
    MmapFile,
    MmapRead,
    MmapLen,
    ProcessRun,
    ParseCsv,
    ToCsv,
    CsvF64s,
    CsvInts,
    CsvStrs,
    CsvNrows,
    CsvCols,
    CsvF64,
    CsvInt,
    CsvStr,
    CharCode,
    CharFromCode,
    StringLen,
    StringConcat,
    StringSlice,
    StringContains,
    StringStartsWith,
    StringEndsWith,
    StringTrim,
    ToStringUnit,
    ToStringScalar,
    ToStringTensor,
    ToStringList,
    ToStringTuple,
    ToStringDict,
    ToStringOption,
    ToStringAdt,
    ToStringFunction,
    ToInt,
    ToFloat,
    LenList,
    LenDict,
    IndexList,
    AppendList,
    ConcatList,
    ConcatTensors,
    TakeList,
    SkipList,
    DropValue,
    ChunkList,
    RangeList,
    MapList,
    FilterList,
    FoldList,
    ScanList,
    TensorScan,
    PartitionList,
    FlatMapList,
    FlattenList,
    ZipList,
    EnumerateList,
    DictOf,
    DictGet,
    DictContains,
    DictRemove,
    DictInsert,
    DictMerge,
    DictKeys,
    DictValues,
    DictEntries,
    ToTensorList,
    ToListTensor,
    PadSequences,
    PadSequencesTo,
    SplitTensor,
    EqRecursive,
    NeqRecursive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BuiltinSiblingCaseDecl {
    pub domain: BuiltinSemanticDomain,
    pub case: BuiltinSiblingCaseId,
}

/// Complete pre-4C semantic discovery declaration for one builtin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinCapabilityDecl {
    pub domains: &'static [BuiltinSemanticDomain],
    pub sibling_cases: &'static [BuiltinSiblingCaseDecl],
}

const NUMERIC_DOMAIN: &[BuiltinSemanticDomain] = &[BuiltinSemanticDomain::Numeric];
const CONTAINER_DOMAIN: &[BuiltinSemanticDomain] = &[BuiltinSemanticDomain::Container];
const BOUNDARY_DOMAIN: &[BuiltinSemanticDomain] = &[BuiltinSemanticDomain::Boundary];
const NUMERIC_CONTAINER_DOMAINS: &[BuiltinSemanticDomain] = &[
    BuiltinSemanticDomain::Numeric,
    BuiltinSemanticDomain::Container,
];

impl BuiltinCapabilityDecl {
    pub const NUMERIC_ONLY: Self = Self {
        domains: NUMERIC_DOMAIN,
        sibling_cases: &[],
    };
}

const NUMERIC_CAPABILITY: BuiltinCapabilityDecl = BuiltinCapabilityDecl::NUMERIC_ONLY;

macro_rules! sibling_capability {
    ($domains:ident, $domain:ident, $case:ident) => {
        BuiltinCapabilityDecl {
            domains: $domains,
            sibling_cases: &[BuiltinSiblingCaseDecl {
                domain: BuiltinSemanticDomain::$domain,
                case: BuiltinSiblingCaseId::$case,
            }],
        }
    };
}

const EQ_RECURSIVE_CASES: &[BuiltinSiblingCaseDecl] = &[BuiltinSiblingCaseDecl {
    domain: BuiltinSemanticDomain::Container,
    case: BuiltinSiblingCaseId::EqRecursive,
}];
const EQ_CAPABILITY: BuiltinCapabilityDecl = BuiltinCapabilityDecl {
    domains: NUMERIC_CONTAINER_DOMAINS,
    sibling_cases: EQ_RECURSIVE_CASES,
};

const NEQ_RECURSIVE_CASES: &[BuiltinSiblingCaseDecl] = &[BuiltinSiblingCaseDecl {
    domain: BuiltinSemanticDomain::Container,
    case: BuiltinSiblingCaseId::NeqRecursive,
}];
const NEQ_CAPABILITY: BuiltinCapabilityDecl = BuiltinCapabilityDecl {
    domains: NUMERIC_CONTAINER_DOMAINS,
    sibling_cases: NEQ_RECURSIVE_CASES,
};

const TO_STRING_CASES: &[BuiltinSiblingCaseDecl] = &[
    BuiltinSiblingCaseDecl {
        domain: BuiltinSemanticDomain::Boundary,
        case: BuiltinSiblingCaseId::ToStringFunction,
    },
    BuiltinSiblingCaseDecl {
        domain: BuiltinSemanticDomain::Boundary,
        case: BuiltinSiblingCaseId::ToStringUnit,
    },
    BuiltinSiblingCaseDecl {
        domain: BuiltinSemanticDomain::Boundary,
        case: BuiltinSiblingCaseId::ToStringScalar,
    },
    BuiltinSiblingCaseDecl {
        domain: BuiltinSemanticDomain::Boundary,
        case: BuiltinSiblingCaseId::ToStringTensor,
    },
    BuiltinSiblingCaseDecl {
        domain: BuiltinSemanticDomain::Boundary,
        case: BuiltinSiblingCaseId::ToStringList,
    },
    BuiltinSiblingCaseDecl {
        domain: BuiltinSemanticDomain::Boundary,
        case: BuiltinSiblingCaseId::ToStringTuple,
    },
    BuiltinSiblingCaseDecl {
        domain: BuiltinSemanticDomain::Boundary,
        case: BuiltinSiblingCaseId::ToStringDict,
    },
    BuiltinSiblingCaseDecl {
        domain: BuiltinSemanticDomain::Boundary,
        case: BuiltinSiblingCaseId::ToStringOption,
    },
    BuiltinSiblingCaseDecl {
        domain: BuiltinSemanticDomain::Boundary,
        case: BuiltinSiblingCaseId::ToStringAdt,
    },
];
const TO_STRING_CAPABILITY: BuiltinCapabilityDecl = BuiltinCapabilityDecl {
    domains: BOUNDARY_DOMAIN,
    sibling_cases: TO_STRING_CASES,
};

/// Lane realizability declaration for a builtin. Part of `BuiltinDecl`.
/// Determines whether the tensor-DAG path or host path realizes this op.
///
/// This is NOT `#[derive(Default)]` — omitting realizability on a new
/// `BuiltinDecl` entry must be a compile error, not a silent default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Realizability {
    /// Both lanes can realize this op (when types permit).
    Universal,
    /// Only the host lane can realize this op, regardless of types.
    HostOnly,
    /// Tensor-lane when the resolved type is tensor; host-only when scalar.
    /// The precision-capability check (def-level, issue #912 Task 6) is
    /// separate — it is NOT encoded here.
    TensorAtTensorType,
}

/// Whether a builtin is accepted solely by its polymorphic HM signature or
/// must reach checker-owned semantic inference. Required on every
/// [`BuiltinDecl`]; there is intentionally no default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceDisposition {
    /// The signature fully determines the builtin's type behavior. The reason
    /// is part of the reviewed declaration rather than an implicit fallback.
    GenericAccepted { reason: &'static str },
    /// The builtin must reach the named semantic checker family.
    Checked(BuiltinInferenceRule),
}

/// Closed checker families used by builtin declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinInferenceRule {
    /// A shape-computed result that may owe a deferred obligation when an
    /// input is not yet bound.
    ShapeComputed,
    /// A dedicated type/arity/static-value rule in the application dispatcher.
    Specialized,
}

const SHAPE_COMPUTED_INFERENCE_BUILTINS: &[&str] = &[
    "matmul",
    "sum",
    "count",
    "max_reduce",
    "min_reduce",
    "prod_reduce",
    "argmax_reduce",
    "argmin_reduce",
    "mean",
    "expand",
    "insert",
    "layer_norm",
    "conv",
    "scatter_elements",
];

const SPECIALIZED_INFERENCE_BUILTINS: &[&str] = &[
    "add",
    "mul",
    "max_elem",
    "neg",
    "recip",
    "exp",
    "log",
    "sin",
    "tan",
    "atan",
    "sqrt",
    "floor",
    "ceil",
    "round",
    "uniform_like",
    "cmplt",
    "sub",
    "div",
    "floor_div",
    "trunc_div",
    "mod",
    "eq",
    "neq",
    "lt",
    "gt",
    "lte",
    "gte",
    "bitand",
    "bitor",
    "bitxor",
    "shl",
    "shr",
    "and",
    "or",
    "not",
    "relu",
    "sigmoid",
    "tanh",
    "silu",
    "gelu",
    "softmax",
    "min_elem",
    "reduce_window_max",
    "reduce_window_min",
    "reduce_window_sum",
    "reduce_window_mean",
    "reshape",
    "permute",
    "pad",
    "shrink",
    "stride",
    "print",
    "fail",
    "debug",
    "test_assert_close_tensor",
    "char_code",
    "char_from_code",
    "string_len",
    "string_concat",
    "string_slice",
    "string_contains",
    "string_starts_with",
    "string_ends_with",
    "string_trim",
    "to_string",
    "to_int",
    "to_float",
    "rank",
    "shape",
    "numel",
    "tensor_to_scalar",
    "scalar_to_tensor",
    "len",
    "index",
    "append",
    "concat",
    "take",
    "skip",
    "chunk",
    "range",
    "map",
    "filter",
    "fold",
    "scan",
    "tensor_scan",
    "partition",
    "flat_map",
    "flatten",
    "zip",
    "enumerate",
    "dict_of",
    "dict_get",
    "dict_contains",
    "dict_remove",
    "dict_insert",
    "dict_merge",
    "dict_keys",
    "dict_values",
    "dict_entries",
    "to_tensor",
    "to_list",
    "pad_sequences",
    "pad_sequences_to",
    "read_file",
    "write_file",
    "read_lines",
    "read_bytes",
    "file_exists",
    "list_dir",
    "mmap_file",
    "mmap_read",
    "mmap_len",
    "process_run",
    "round_to",
    "parse_csv",
    "to_csv",
    "csv_f64s",
    "csv_ints",
    "csv_strs",
    "csv_nrows",
    "csv_cols",
    "csv_f64",
    "csv_int",
    "csv_str",
    "einsum",
    "split",
    "gather",
    "scatter",
    "scatter_replace",
    "where",
    "cumsum",
    "sort",
    "diagonal",
    "trace",
    "clamp",
    // Linearity, not a container operation: the explicit one-argument
    // consume of [05-OP-67], paired with the `copy` keyword. The list
    // slice that once shared this name is `skip` ([05-OP-54]).
    "drop",
];

/// True only when the declared rule is wired to the corresponding closed
/// application-dispatch family. This is consulted at runtime and by the
/// registry tripwire; a checked declaration with no route fails loudly.
pub(crate) fn has_registered_inference_route(name: &str, rule: BuiltinInferenceRule) -> bool {
    match rule {
        BuiltinInferenceRule::ShapeComputed => SHAPE_COMPUTED_INFERENCE_BUILTINS.contains(&name),
        BuiltinInferenceRule::Specialized => SPECIALIZED_INFERENCE_BUILTINS.contains(&name),
    }
}

/// A builtin's complete declaration: name, semantic discovery capability,
/// inference disposition, realizability, shape class, and axis-argument layout.
/// All fields are required — adding a builtin without any field is a
/// compile error (missing struct field). No `Default` implementation.
///
/// ```compile_fail
/// // Omitting `realizability` must fail to compile.
/// use chelis_types::{AxisArgumentLayout, BuiltinCapabilityDecl, BuiltinDecl, BuiltinInferenceRule, InferenceDisposition, ShapeClass};
/// let _ = BuiltinDecl {
///     name: "x",
///     capability: BuiltinCapabilityDecl::NUMERIC_ONLY,
///     inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
///     shape_class: ShapeClass::Rewriting,
///     axis_arguments: AxisArgumentLayout::NoAxes,
/// };
/// ```
///
/// ```compile_fail
/// // Omitting `shape_class` must fail to compile.
/// use chelis_types::{AxisArgumentLayout, BuiltinCapabilityDecl, BuiltinDecl, BuiltinInferenceRule, InferenceDisposition, Realizability};
/// let _ = BuiltinDecl {
///     name: "x",
///     capability: BuiltinCapabilityDecl::NUMERIC_ONLY,
///     inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
///     realizability: Realizability::Universal,
///     axis_arguments: AxisArgumentLayout::NoAxes,
/// };
/// ```
///
/// ```compile_fail
/// // Omitting `inference` must fail to compile.
/// use chelis_types::{AxisArgumentLayout, BuiltinCapabilityDecl, BuiltinDecl, Realizability, ShapeClass};
/// let _ = BuiltinDecl {
///     name: "x",
///     capability: BuiltinCapabilityDecl::NUMERIC_ONLY,
///     realizability: Realizability::Universal,
///     shape_class: ShapeClass::Rewriting,
///     axis_arguments: AxisArgumentLayout::NoAxes,
/// };
/// ```
///
/// ```compile_fail
/// // Omitting `axis_arguments` must fail to compile.
/// use chelis_types::{BuiltinCapabilityDecl, BuiltinDecl, BuiltinInferenceRule, InferenceDisposition, Realizability, ShapeClass};
/// let _ = BuiltinDecl {
///     name: "x",
///     capability: BuiltinCapabilityDecl::NUMERIC_ONLY,
///     inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
///     realizability: Realizability::Universal,
///     shape_class: ShapeClass::Rewriting,
/// };
/// ```
///
/// ```compile_fail
/// // Omitting `capability` must fail to compile.
/// use chelis_types::{AxisArgumentLayout, BuiltinDecl, BuiltinInferenceRule, InferenceDisposition, Realizability, ShapeClass};
/// let _ = BuiltinDecl {
///     name: "x",
///     inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
///     realizability: Realizability::Universal,
///     shape_class: ShapeClass::Rewriting,
///     axis_arguments: AxisArgumentLayout::NoAxes,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinDecl {
    pub name: &'static str,
    pub capability: BuiltinCapabilityDecl,
    pub inference: InferenceDisposition,
    pub realizability: Realizability,
    pub shape_class: ShapeClass,
    pub axis_arguments: AxisArgumentLayout,
}

/// The consolidated builtin table. Single source of truth for builtin
/// metadata. Each entry declares name, semantic domains/cases, inference
/// disposition, realizability, shape class, and axis layout.
/// Omitting any field is a compile error.
pub const BUILTINS: &[BuiltinDecl] = &[
    // ─── Tensor elementwise (Universal, Identity) ────────────────────
    BuiltinDecl {
        name: "add",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "mul",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "sub",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "div",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "floor_div",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "trunc_div",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "max_elem",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "min_elem",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "neg",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "recip",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "exp",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "log",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "sin",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "sqrt",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "cos",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::GenericAccepted {
            reason: "the polymorphic signature fully determines this builtin type",
        },
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "tan",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "atan",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "abs",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::GenericAccepted {
            reason: "the polymorphic signature fully determines this builtin type",
        },
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "floor",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "ceil",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "round",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "relu",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "sigmoid",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "tanh",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "silu",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "gelu",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "uniform_like",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "cmplt",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "eq",
        capability: EQ_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "neq",
        capability: NEQ_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "lt",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "gt",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "lte",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "gte",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "mod",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "bitand",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "bitor",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "bitxor",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "shl",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "shr",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::TensorAtTensorType,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "and",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "or",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "not",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "where",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "clamp",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Identity,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    // ─── Reductions (Universal, NameTracked) ─────────────────────────
    BuiltinDecl {
        name: "softmax",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[1]),
    },
    BuiltinDecl {
        name: "mean",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::NameTracked,
        axis_arguments: AxisArgumentLayout::VariadicFrom(1),
    },
    BuiltinDecl {
        name: "sum",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::NameTracked,
        axis_arguments: AxisArgumentLayout::VariadicFrom(1),
    },
    BuiltinDecl {
        name: "count",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::NameTracked,
        axis_arguments: AxisArgumentLayout::VariadicFrom(1),
    },
    BuiltinDecl {
        name: "max_reduce",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::NameTracked,
        axis_arguments: AxisArgumentLayout::VariadicFrom(1),
    },
    BuiltinDecl {
        name: "min_reduce",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::NameTracked,
        axis_arguments: AxisArgumentLayout::VariadicFrom(1),
    },
    BuiltinDecl {
        name: "prod_reduce",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::NameTracked,
        axis_arguments: AxisArgumentLayout::VariadicFrom(1),
    },
    BuiltinDecl {
        name: "argmax_reduce",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::NameTracked,
        axis_arguments: AxisArgumentLayout::VariadicFrom(1),
    },
    BuiltinDecl {
        name: "argmin_reduce",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::NameTracked,
        axis_arguments: AxisArgumentLayout::VariadicFrom(1),
    },
    // ─── Windowed reductions ─────────────────────────────────────────
    BuiltinDecl {
        name: "reduce_window_max",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "reduce_window_min",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "reduce_window_sum",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "reduce_window_mean",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    // ─── Shape ops (Universal, Rewriting) ────────────────────────────
    BuiltinDecl {
        name: "matmul",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "layer_norm",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "conv",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "reshape",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "permute",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::VariadicFrom(1),
    },
    BuiltinDecl {
        name: "expand",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::NameTracked,
        axis_arguments: AxisArgumentLayout::Fixed(&[1, 3]),
    },
    BuiltinDecl {
        name: "insert",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::NameTracked,
        axis_arguments: AxisArgumentLayout::Fixed(&[1, 3]),
    },
    BuiltinDecl {
        name: "pad",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "shrink",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "stride",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "gather",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[2]),
    },
    BuiltinDecl {
        name: "scatter",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[3]),
    },
    BuiltinDecl {
        name: "scatter_replace",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[3]),
    },
    BuiltinDecl {
        name: "scatter_elements",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::ShapeComputed),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[3]),
    },
    BuiltinDecl {
        name: "einsum",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "split",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, SplitTensor),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[1]),
    },
    BuiltinDecl {
        name: "cumsum",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[1]),
    },
    BuiltinDecl {
        name: "sort",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[1]),
    },
    BuiltinDecl {
        name: "diagonal",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[1, 2]),
    },
    BuiltinDecl {
        name: "trace",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[1, 2]),
    },
    // ─── IO / effects (HostOnly) ─────────────────────────────────────
    BuiltinDecl {
        name: "print",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, PrintRecursive),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "fail",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, FailString),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "debug",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, DebugRecursive),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "read_file",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ReadFile),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "write_file",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, WriteFile),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "read_lines",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ReadLines),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "read_bytes",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ReadBytes),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "file_exists",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, FileExists),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "list_dir",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ListDir),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "mmap_file",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, MmapFile),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "mmap_read",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, MmapRead),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "mmap_len",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, MmapLen),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "process_run",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ProcessRun),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "round_to",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    // ─── Host-lane CSV I/O (chelis#903, HostOnly, eval-only) ─────────
    BuiltinDecl {
        name: "parse_csv",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ParseCsv),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "to_csv",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ToCsv),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "csv_f64s",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, CsvF64s),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "csv_ints",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, CsvInts),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "csv_strs",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, CsvStrs),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "csv_nrows",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, CsvNrows),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "csv_cols",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, CsvCols),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "csv_f64",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, CsvF64),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "csv_int",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, CsvInt),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "csv_str",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, CsvStr),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    // ─── String ops (HostOnly) ───────────────────────────────────────
    BuiltinDecl {
        name: "char_code",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, CharCode),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "char_from_code",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, CharFromCode),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "string_len",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, StringLen),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "string_concat",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, StringConcat),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "string_slice",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, StringSlice),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "string_contains",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, StringContains),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "string_starts_with",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, StringStartsWith),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "string_ends_with",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, StringEndsWith),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "string_trim",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, StringTrim),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "to_string",
        capability: TO_STRING_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "to_int",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ToInt),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "to_float",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ToFloat),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    // ─── Tensor introspection (HostOnly at scalar type) ──────────────
    BuiltinDecl {
        name: "rank",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "shape",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[1]),
    },
    BuiltinDecl {
        name: "numel",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "tensor_to_scalar",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "scalar_to_tensor",
        capability: NUMERIC_CAPABILITY,
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::Universal,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    // ─── List ops (HostOnly) ─────────────────────────────────────────
    BuiltinDecl {
        name: "len",
        capability: BuiltinCapabilityDecl {
            domains: CONTAINER_DOMAIN,
            sibling_cases: &[
                BuiltinSiblingCaseDecl {
                    domain: BuiltinSemanticDomain::Container,
                    case: BuiltinSiblingCaseId::LenList,
                },
                BuiltinSiblingCaseDecl {
                    domain: BuiltinSemanticDomain::Container,
                    case: BuiltinSiblingCaseId::LenDict,
                },
            ],
        },
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "index",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, IndexList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "append",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, AppendList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "concat",
        capability: BuiltinCapabilityDecl {
            domains: CONTAINER_DOMAIN,
            sibling_cases: &[
                BuiltinSiblingCaseDecl {
                    domain: BuiltinSemanticDomain::Container,
                    case: BuiltinSiblingCaseId::ConcatList,
                },
                BuiltinSiblingCaseDecl {
                    domain: BuiltinSemanticDomain::Container,
                    case: BuiltinSiblingCaseId::ConcatTensors,
                },
            ],
        },
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::Fixed(&[1]),
    },
    BuiltinDecl {
        name: "take",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, TakeList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "skip",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, SkipList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    // `drop` is the one-argument linearity consume of [05-OP-67]. It kept
    // `CONTAINER_DOMAIN` when [05-OP-54]'s list slice moved to `skip`: the
    // closed domain vocabulary is Numeric/Container/Boundary and none of the
    // three names linearity, so a fourth domain would change the identity
    // strings in the normative registry, the discovery validator's
    // Container/Boundary pairing loop, and the identity regex in
    // `scripts/builtin_atom_registry.py`. That is a capability-system change,
    // not part of separating the two operations.
    BuiltinDecl {
        name: "drop",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, DropValue),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "chunk",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, ChunkList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "range",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, RangeList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "map",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, MapList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "filter",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, FilterList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "fold",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, FoldList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "scan",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, ScanList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "tensor_scan",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, TensorScan),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "partition",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, PartitionList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "flat_map",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, FlatMapList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "flatten",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, FlattenList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "zip",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, ZipList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "enumerate",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, EnumerateList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    // ─── Dict ops (HostOnly) ─────────────────────────────────────────
    BuiltinDecl {
        name: "dict_of",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, DictOf),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "dict_get",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, DictGet),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "dict_contains",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, DictContains),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "dict_remove",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, DictRemove),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "dict_insert",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, DictInsert),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "dict_merge",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, DictMerge),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "dict_keys",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, DictKeys),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "dict_values",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, DictValues),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "dict_entries",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, DictEntries),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    // ─── Tensor conversion (HostOnly) ────────────────────────────────
    BuiltinDecl {
        name: "to_tensor",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ToTensorList),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "to_list",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, ToListTensor),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "pad_sequences",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, PadSequences),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "pad_sequences_to",
        capability: sibling_capability!(CONTAINER_DOMAIN, Container, PadSequencesTo),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    // ─── Test builtins (HostOnly) ────────────────────────────────────
    BuiltinDecl {
        name: "test_assert",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, TestAssertBool),
        inference: InferenceDisposition::GenericAccepted {
            reason: "the polymorphic signature fully determines this builtin type",
        },
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "test_assert_eq",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, TestAssertEq),
        inference: InferenceDisposition::GenericAccepted {
            reason: "the polymorphic signature fully determines this builtin type",
        },
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "test_assert_close_tensor",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, TestAssertCloseTensor),
        inference: InferenceDisposition::Checked(BuiltinInferenceRule::Specialized),
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
    BuiltinDecl {
        name: "test_assert_eq_tensor",
        capability: sibling_capability!(BOUNDARY_DOMAIN, Boundary, TestAssertEqTensor),
        inference: InferenceDisposition::GenericAccepted {
            reason: "the polymorphic signature fully determines this builtin type",
        },
        realizability: Realizability::HostOnly,
        shape_class: ShapeClass::Rewriting,
        axis_arguments: AxisArgumentLayout::NoAxes,
    },
];

/// Look up a builtin's declaration by name. Returns `None` for non-builtins.
pub fn builtin_decl(name: &str) -> Option<&'static BuiltinDecl> {
    BUILTINS.iter().find(|b| b.name == name)
}

pub fn axis_argument_layout(name: &str) -> Option<AxisArgumentLayout> {
    match builtin_decl(name)?.axis_arguments {
        AxisArgumentLayout::NoAxes => None,
        layout => Some(layout),
    }
}

/// Look up a builtin's realizability by name.
pub fn realizability(name: &str) -> Option<Realizability> {
    builtin_decl(name).map(|b| b.realizability)
}

/// Shape semantics of a builtin for the Tier-2 rank-polymorphism
/// Body-Discipline check (`spec/design/rank_polymorphism.md` §Soundness
/// Boundary). Keyed on SHAPE SEMANTICS, **not** the HM scheme: `relu`,
/// `reshape`, and `permute` all share `&tv -> tv`, but only `relu` is
/// shape-identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeClass {
    /// Output shape provably equals an input shape with no axis reordering —
    /// pure elementwise ops (the precision may change, e.g. comparisons).
    /// Always admitted inside a rank-polymorphic (`..r`) body.
    Identity,
    /// Shape-changing but *name-tracked*: the op addresses axes by name and the
    /// procedural inference arm computes a symbolic output that carries the
    /// surviving named axes through (named-axis reductions, Tier-3 §4.5.3).
    /// Admitted inside a rank-poly body — the procedural arm is the real gate:
    /// it rejects a non-existent/ambiguous axis or a positional index at
    /// symbolic rank, so no transposition can slip past.
    NameTracked,
    /// Rewrites/reorders the shape positionally, is shape-parameterized, or is a
    /// non-tensor/host op whose output shape is *not* name-trackable at symbolic
    /// rank. Forbidden inside a rank-poly body: against an opaque spread there
    /// are no named axes left to catch a transposition/reshape (§4.2).
    Rewriting,
}

/// Classify a builtin's shape semantics for the Body-Discipline check.
///
/// `Identity` and `NameTracked` are explicit allowlists; everything else falls
/// through to `Rewriting`. That default is the safe direction — a builtin that
/// is not *provably* shape-identity or name-tracked is rejected inside a
/// rank-poly body, so a missed classification can only over-reject, never open
/// a §4.2 hole. The `shape_class_identity_set_is_pinned` test pins the sets so
/// any change is deliberate.
pub fn shape_class(name: &str) -> ShapeClass {
    match name {
        // Pure elementwise — output shape == input shape (precision may change
        // for comparisons/logical). No axis argument, no reordering.
        "add" | "mul" | "sub" | "div" | "floor_div" | "trunc_div" | "mod" | "max_elem"
        | "min_elem" | "neg" | "recip" | "exp" | "log" | "sin" | "sqrt" | "cos" | "tan"
        | "atan" | "abs" | "floor" | "ceil" | "round" | "relu" | "sigmoid" | "tanh" | "silu"
        | "gelu" | "not" | "clamp" | "uniform_like" | "where" | "eq" | "neq" | "lt" | "gt"
        | "lte" | "gte" | "cmplt" | "bitand" | "bitor" | "bitxor" | "shl" | "shr" | "and"
        | "or" => ShapeClass::Identity,
        // Named-axis reductions: address the reduced axis by name and drop
        // exactly it, carrying the surviving named axes through (Tier-3 §4.5.3).
        // The whole reduction family is name-tracked: each lowers through the
        // tensor-DAG backend (`resolve_reduce_axis` resolves the named axis to
        // a positional index against the operand's named dims) and builds+runs
        // end-to-end. chelis#340 closed the host-lane gap that had restricted
        // this to `sum`/`mean`: a rank-poly named-reduce def whose return type
        // is a tensor routes through `try_lower_tensor_helper_call`, and the
        // host-type inference (`infer_app_expr_host_type` /
        // `infer_builtin_host_type_from_arg_tys`) now types
        // `max_reduce`/`min_reduce`/`prod_reduce`/`argmax_reduce`/
        // `argmin_reduce` over a named axis so a *host-lane* occurrence keeps
        // its tensor type instead of falling through to the
        // "unsupported builtin" host emit. `argmax_reduce`/`argmin_reduce`
        // return an i64 index tensor (no-grad). All remain usable at
        // concrete rank.
        //
        // Named-axis expand (chelis#339, the R+1 inverse): `expand` addresses
        // its insertion point by name (trailing end, or before a named anchor)
        // and the procedural arm (`check_expand_signature`) computes the
        // symbolic output row, rejecting positional axes at symbolic rank —
        // the same gate structure as the reductions.
        "sum" | "count" | "mean" | "max_reduce" | "min_reduce" | "prod_reduce"
        | "argmax_reduce" | "argmin_reduce" | "expand" | "insert" => ShapeClass::NameTracked,
        // Positional reshapes/permutes, matmul/conv, axis-indexed ops,
        // gather/scatter, and every non-tensor/host builtin.
        _ => ShapeClass::Rewriting,
    }
}

/// Operand-family contract shared by builtin schemes and direct-call diagnostics.
/// Value constraints attach this contract to whole scalar/tensor operands; authored
/// dtype bounds still constrain primitive precision parameters only.
pub(crate) fn operand_dtype_family(name: &str) -> Option<TypeVarRestriction> {
    use TypeVarRestriction::{ActiveFloat, ActiveInt, ActiveNumeric};
    match name {
        "mean" | "softmax" | "div" | "matmul" | "layer_norm" | "exp" | "log" | "sin" | "cos"
        | "tan" | "atan" | "sqrt" | "relu" | "sigmoid" | "tanh" | "silu" | "gelu" | "recip"
        | "reduce_window_mean" => Some(ActiveFloat),
        // [05-OP-64], [05-OP-47] and truncating division.
        "trunc_div" | "mod" | "bitand" | "bitor" | "bitxor" | "shl" | "shr" => Some(ActiveInt),
        // [05-OP-46], [05-OP-40], [05-OP-36], [05-OP-30]/[05-OP-12..16],
        // and the arithmetic rows of spec/04 section 5.4.
        "add" | "mul" | "sub" | "neg" | "floor_div" | "abs" | "floor" | "ceil" | "round"
        | "max_elem" | "min_elem" | "cmplt" | "lt" | "gt" | "gte" | "lte" | "sum"
        | "max_reduce" | "min_reduce" | "prod_reduce" | "argmax_reduce" | "argmin_reduce"
        | "reduce_window_sum" | "reduce_window_max" | "reduce_window_min" => Some(ActiveNumeric),
        _ => None,
    }
}

fn operand_value_restrictions(
    name: &str,
    operands: &[TypeVar],
) -> Vec<(TypeVar, TypeVarRestriction)> {
    operand_dtype_family(name)
        .map(|family| {
            operands
                .iter()
                .map(|var| (*var, family.for_value()))
                .collect()
        })
        .unwrap_or_default()
}

/// Create the built-in type environment with all RISC Tier 1 + Tier 2 signatures.
pub fn builtin_env() -> (Env, VarGen) {
    let mut env = Env::new();
    let mut vg = VarGen::default();

    // --- Signature builders ---
    fn borrowed(ty: Type) -> Type {
        Type::Ref(Box::new(ty))
    }

    // ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D,p]
    // D is represented as a single DimVar. At unification time, tensor dims are
    // matched positionally (equal rank, element-wise dim unification).
    // The DimVar here acts as a placeholder — when the scheme is instantiated,
    // a fresh DimVar is created, and unification with a concrete tensor will
    // bind dims element-wise.
    fn tensor_binop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let dv = vg.fresh_dvar();
        // Whole-tensor type variable: both args and return are the
        // SAME tensor type. This continues to work post-WS-A5 because
        // unifying two tensor types unifies both dim lists and the
        // precision slot. Precision-slot polymorphism (TensorPrec::Var)
        // is reserved for sig-quantified type variables in user-written
        // Surf signatures — the builtin scheme keeps the simpler
        // "whole tensor as one type variable" shape.
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![tv],
            tvar_restrictions: operand_value_restrictions(name, &[tv]),
            dvars: vec![dv],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(tv)), borrowed(Type::Var(tv))],
                Box::new(Type::Var(tv)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    // ∀D,p. tensor[D,p] → tensor[D,p]
    fn tensor_unop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let _dv = vg.fresh_dvar();
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![tv],
            tvar_restrictions: operand_value_restrictions(name, &[tv]),
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(vec![borrowed(Type::Var(tv))], Box::new(Type::Var(tv))),
        };
        env.bind(name.to_string(), scheme);
    }

    // ∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D, bool]
    // The key difference: return precision is always Bool.
    fn cmplt_sig(name: &str, env: &mut Env, vg: &mut VarGen) {
        // Input: two tensors with same dims and same precision
        // Output: tensor with same dims but Bool precision
        // We represent this by using a DimVar for dims and constraining
        // the return type explicitly.
        let dv = vg.fresh_dvar();
        let input_tv = vg.fresh_tvar(); // will unify to tensor[D, p]
        // Can't easily express "same dims, different precision" with just TypeVars.
        // For Phase 0: use (T, T) → T but the inference engine will special-case
        // cmplt to swap the return precision to Bool after unification.
        //
        // Better approach: just use TypeVar polymorphism. The args must be the same
        // type (tensor[D,p]), and the return is also the same type. This is wrong
        // for cmplt (should return bool), but fixing it properly requires either:
        // 1. A special case in the inference engine for cmplt
        // 2. A richer signature language that can express precision transformation
        //
        // For now: register cmplt as a SPECIAL FORM that the inference engine handles.
        // The builtin entry just marks it as a 2-arg function.
        let _ = (dv, input_tv);
        let tv = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![tv, output],
            tvar_restrictions: operand_value_restrictions(name, &[tv]),
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(tv)), borrowed(Type::Var(tv))],
                // The application checker replaces this with bool on the
                // operand's surface. Keeping the placeholder independent
                // avoids binding an unresolved operand variable to bool
                // before that procedural result rule runs.
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    // ∀D. (tensor[D,bool], tensor[D,bool]) → tensor[D,bool]
    fn logical_binop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let _dv = vg.fresh_dvar();
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![tv],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(tv)), borrowed(Type::Var(tv))],
                Box::new(Type::Var(tv)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn logical_unop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let _dv = vg.fresh_dvar();
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![tv],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(vec![borrowed(Type::Var(tv))], Box::new(Type::Var(tv))),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_layer_norm(name: &str, env: &mut Env, vg: &mut VarGen) {
        let t1 = vg.fresh_tvar();
        let t2 = vg.fresh_tvar();
        let t3 = vg.fresh_tvar();
        let epsilon = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![t1, t2, t3, epsilon],
            tvar_restrictions: operand_value_restrictions(name, &[t1, t2, t3, epsilon]),
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    borrowed(Type::Var(t1)),
                    borrowed(Type::Var(t2)),
                    borrowed(Type::Var(t3)),
                    borrowed(Type::Var(epsilon)),
                ],
                Box::new(Type::Var(t1)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_binop_to_out(name: &str, env: &mut Env, vg: &mut VarGen) {
        let t1 = vg.fresh_tvar();
        let t2 = vg.fresh_tvar();
        let out = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![t1, t2, out],
            tvar_restrictions: operand_value_restrictions(name, &[t1, t2]),
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(t1)), borrowed(Type::Var(t2))],
                Box::new(Type::Var(out)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_reduce_to_out(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let out = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![input, out],
            tvar_restrictions: operand_value_restrictions(name, &[input]),
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(input)), Type::Prim(Prim::Int32)],
                Box::new(Type::Var(out)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    /// Fallback HM scheme for the `reduce_window_*` family. The dedicated
    /// `infer_reduce_window_app` arm in `infer.rs` overrides the result
    /// type with the spec §2.3.1 shape contract; this scheme exists so
    /// the function name is in scope at lookup time and the canonical
    /// arg-arity / list-of-i64 constraints are visible during unification.
    fn tensor_reduce_window(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let out = vg.fresh_tvar();
        let int_list = Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![input, out],
            tvar_restrictions: operand_value_restrictions(name, &[input]),
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(input)), int_list.clone(), int_list],
                Box::new(Type::Var(out)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_expand_to_out(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let out = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![input, out],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    borrowed(Type::Var(input)),
                    // [05-DIM-1]: axis-domain axis (i32), extent-domain
                    // size (i64).
                    Type::Prim(Prim::Int32),
                    Type::Prim(Prim::Int64),
                ],
                Box::new(Type::Var(out)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_with_rate(name: &str, env: &mut Env, vg: &mut VarGen) {
        let precision = vg.fresh_tvar();
        let rank = vg.fresh_rvar();
        let input = Type::Tensor(vec![Dim::Rank(rank)], TensorPrec::Var(precision));
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![precision],
            tvar_restrictions: vec![(precision, TypeVarRestriction::ActiveFloat)],
            dvars: vec![],
            rvars: vec![rank],
            body: Type::Fn(
                vec![borrowed(input.clone()), Type::Var(precision)],
                Box::new(input),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_with_bounds(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![input],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    borrowed(Type::Var(input)),
                    Type::Prim(Prim::F32),
                    Type::Prim(Prim::F32),
                ],
                Box::new(Type::Var(input)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_conv(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let kernel = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![input, kernel, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    borrowed(Type::Var(input)),
                    borrowed(Type::Var(kernel)),
                    Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]),
                    Type::Adt(
                        "List".to_string(),
                        vec![Type::Tuple(vec![
                            Type::Prim(Prim::Int64),
                            Type::Prim(Prim::Int64),
                        ])],
                    ),
                ],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn tensor_reduce(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![input],
            tvar_restrictions: operand_value_restrictions(name, &[input]),
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(input)), Type::Prim(Prim::Int32)],
                Box::new(Type::Var(input)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_unop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: match name {
                "len" => vec![CollectionConstraint::Len {
                    operand: Type::Var(input),
                    result: Type::Var(output),
                }],
                _ => vec![],
            },
            tvars: vec![input, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(vec![Type::Var(input)], Box::new(Type::Var(output))),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_binop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let lhs = vg.fresh_tvar();
        let rhs = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: match name {
                "index" => vec![CollectionConstraint::Index {
                    list: Type::Var(lhs),
                    index: Type::Var(rhs),
                    result: Type::Var(output),
                }],
                "append" => vec![CollectionConstraint::Append {
                    list: Type::Var(lhs),
                    value: Type::Var(rhs),
                    result: Type::Var(output),
                }],
                "concat" => vec![CollectionConstraint::Concat {
                    lhs: Type::Var(lhs),
                    rhs: Type::Var(rhs),
                    result: Type::Var(output),
                }],
                _ => vec![],
            },
            tvars: vec![lhs, rhs, output],
            tvar_restrictions: operand_value_restrictions(name, &[lhs, rhs]),
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Var(lhs), Type::Var(rhs)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_triop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![a, b, c, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Var(a), Type::Var(b), Type::Var(c)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_triop_first_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![a, b, c, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(a)), Type::Var(b), Type::Var(c)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_triop_first_two_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![a, b, c, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(a)), borrowed(Type::Var(b)), Type::Var(c)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_triop_all_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![a, b, c, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    borrowed(Type::Var(a)),
                    borrowed(Type::Var(b)),
                    borrowed(Type::Var(c)),
                ],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_triop_second_third_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![a, b, c, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Var(a), borrowed(Type::Var(b)), borrowed(Type::Var(c))],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_pentaop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let d = vg.fresh_tvar();
        let e = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![a, b, c, d, e, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Var(a),
                    Type::Var(b),
                    Type::Var(c),
                    Type::Var(d),
                    Type::Var(e),
                ],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_quadop(name: &str, env: &mut Env, vg: &mut VarGen) {
        let a = vg.fresh_tvar();
        let b = vg.fresh_tvar();
        let c = vg.fresh_tvar();
        let d = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![a, b, c, d, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Var(a), Type::Var(b), Type::Var(c), Type::Var(d)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_unop_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let input = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![input, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(input))],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_unop_borrow_same(name: &str, env: &mut Env, vg: &mut VarGen) {
        let tv = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![tv],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(vec![borrowed(Type::Var(tv))], Box::new(Type::Var(tv))),
        };
        env.bind(name.to_string(), scheme);
    }

    fn generic_binop_first_borrow(name: &str, env: &mut Env, vg: &mut VarGen) {
        let lhs = vg.fresh_tvar();
        let rhs = vg.fresh_tvar();
        let output = vg.fresh_tvar();
        let scheme = Scheme {
            constraints: vec![],
            tvars: vec![lhs, rhs, output],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![borrowed(Type::Var(lhs)), Type::Var(rhs)],
                Box::new(Type::Var(output)),
            ),
        };
        env.bind(name.to_string(), scheme);
    }

    // --- Register all built-ins ---

    // Tier 1: RISC Primitives
    tensor_binop("add", &mut env, &mut vg);
    tensor_binop("sub", &mut env, &mut vg);
    tensor_binop("mul", &mut env, &mut vg);
    // `div` and `recip` were promoted from a Tier 2
    // `exp(neg(log(_)))` decomposition to native Tier 1 primitives
    // (IEEE-754 semantics, correct on the full real line). Since
    // chelis#178 `div` is float-only; integer division uses the
    // dedicated `floor_div` / `trunc_div` primitives below. The
    // signature templates are fully precision-polymorphic; the
    // precision restrictions (div float-only, trunc_div int-only,
    // floor_div both) are enforced in `infer.rs`.
    tensor_binop("div", &mut env, &mut vg);
    tensor_binop("floor_div", &mut env, &mut vg);
    tensor_binop("trunc_div", &mut env, &mut vg);
    tensor_binop("max_elem", &mut env, &mut vg);
    tensor_binop("min_elem", &mut env, &mut vg);

    tensor_unop("neg", &mut env, &mut vg);
    tensor_unop("recip", &mut env, &mut vg);
    tensor_unop("exp", &mut env, &mut vg);
    tensor_unop("log", &mut env, &mut vg);
    tensor_unop("sin", &mut env, &mut vg);
    tensor_unop("sqrt", &mut env, &mut vg);
    tensor_unop("cos", &mut env, &mut vg);
    tensor_unop("tan", &mut env, &mut vg);
    tensor_unop("atan", &mut env, &mut vg);
    tensor_unop("abs", &mut env, &mut vg);
    tensor_unop("floor", &mut env, &mut vg);
    tensor_unop("ceil", &mut env, &mut vg);
    tensor_unop("round", &mut env, &mut vg);
    tensor_with_bounds("uniform_like", &mut env, &mut vg);

    cmplt_sig("cmplt", &mut env, &mut vg);

    // Tier 2: Derived built-ins
    generic_binop("mod", &mut env, &mut vg);
    cmplt_sig("eq", &mut env, &mut vg);
    cmplt_sig("neq", &mut env, &mut vg);
    cmplt_sig("lt", &mut env, &mut vg);
    cmplt_sig("gt", &mut env, &mut vg);
    cmplt_sig("lte", &mut env, &mut vg);
    cmplt_sig("gte", &mut env, &mut vg);
    generic_binop("bitand", &mut env, &mut vg);
    generic_binop("bitor", &mut env, &mut vg);
    generic_binop("bitxor", &mut env, &mut vg);
    generic_binop("shl", &mut env, &mut vg);
    generic_binop("shr", &mut env, &mut vg);

    logical_binop("and", &mut env, &mut vg);
    logical_binop("or", &mut env, &mut vg);
    logical_unop("not", &mut env, &mut vg);

    tensor_unop("relu", &mut env, &mut vg);
    tensor_unop("sigmoid", &mut env, &mut vg);
    // Bucket 3 activation parity: `tanh`, `silu`, `gelu` are pointwise
    // tensor unops with the same `∀D,p. tensor[D,p] → tensor[D,p]`
    // signature shape as `relu`/`sigmoid`. Their host-runtime and
    // C-backend lowerings live in `chelis-compiler-api/src/runtime/host_ops.rs`
    // and `chelis-backend-c/src/host_emit.rs` respectively.
    tensor_unop("tanh", &mut env, &mut vg);
    tensor_unop("silu", &mut env, &mut vg);
    tensor_unop("gelu", &mut env, &mut vg);
    tensor_reduce("softmax", &mut env, &mut vg);
    tensor_reduce_to_out("mean", &mut env, &mut vg);

    tensor_binop_to_out("matmul", &mut env, &mut vg);
    tensor_layer_norm("layer_norm", &mut env, &mut vg);
    tensor_conv("conv", &mut env, &mut vg);
    tensor_reduce_to_out("sum", &mut env, &mut vg);
    tensor_reduce_to_out("count", &mut env, &mut vg);
    tensor_reduce_to_out("max_reduce", &mut env, &mut vg);
    tensor_reduce_to_out("min_reduce", &mut env, &mut vg);
    tensor_reduce_to_out("prod_reduce", &mut env, &mut vg);
    tensor_reduce_to_out("argmax_reduce", &mut env, &mut vg);
    tensor_reduce_to_out("argmin_reduce", &mut env, &mut vg);
    // §2.3.1 reduce_window family. Fallback HM scheme is
    // `&tensor[D, p] -> List[i64] -> List[i64] -> tensor[D', p]`;
    // the actual shape contract (output rank = input rank, trailing
    // axis extents derived from the window/stride formula) is enforced
    // by the dedicated `infer_reduce_window_app` arm in `infer.rs`,
    // which also rejects non-positive window/stride literals.
    tensor_reduce_window("reduce_window_max", &mut env, &mut vg);
    tensor_reduce_window("reduce_window_min", &mut env, &mut vg);
    tensor_reduce_window("reduce_window_sum", &mut env, &mut vg);
    tensor_reduce_window("reduce_window_mean", &mut env, &mut vg);
    // Movement primitives whose RISC lowering reads window parameters from
    // `args[1..]`. The `tensor_unop` scheme below only declares the arity-1
    // fallback; `reshape` and `permute` already have dedicated `infer_*_app`
    // paths in `infer.rs` that accept the parameterized arity. `pad`,
    // `shrink`, and `stride` do not — see issue Chelis-Lang/chelis#187 and
    // the matching dedicated `infer_shrink_app` / `infer_stride_app` paths.
    tensor_unop("reshape", &mut env, &mut vg);
    tensor_unop("permute", &mut env, &mut vg);
    tensor_expand_to_out("expand", &mut env, &mut vg);
    tensor_expand_to_out("insert", &mut env, &mut vg);
    tensor_unop("pad", &mut env, &mut vg);
    tensor_unop("shrink", &mut env, &mut vg);
    tensor_unop("stride", &mut env, &mut vg);
    tensor_with_rate("dropout", &mut env, &mut vg);
    generic_unop_borrow("print", &mut env, &mut vg);
    generic_unop("fail", &mut env, &mut vg);
    generic_unop_borrow_same("debug", &mut env, &mut vg);

    // Test builtins. Effects (`! { Test }`) are not carried on the scheme itself;
    // they are assigned by chelis-effects when the name is encountered in an `app`
    // node, mirroring how `print` / `read_file` acquire IO.
    env.bind(
        "test_assert".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::Bool), Type::Prim(Prim::String)],
                Box::new(Type::Unit),
            ),
        },
    );
    {
        let value = vg.fresh_tvar();
        env.bind(
            "test_assert_eq".to_string(),
            Scheme {
                constraints: vec![],
                tvars: vec![value],
                tvar_restrictions: vec![],
                dvars: vec![],
                rvars: vec![],
                body: Type::Fn(
                    vec![Type::Var(value), Type::Var(value), Type::Prim(Prim::String)],
                    Box::new(Type::Unit),
                ),
            },
        );
    }
    {
        // test_assert_close_tensor:
        //   (&tensor[..r,p_float], &tensor[..r,p_float], p_float, string) -> unit
        // One quantified type variable occupies both tensor precision slots
        // and the scalar tolerance position. This makes same-dtype equality a
        // structural unification constraint. The quantified variable's
        // ActiveFloat restriction is part of the function value, so aliases,
        // polymorphic wrappers, and higher-order calls retain admissibility.
        // The shared rank variable preserves arbitrary rank and exact shape
        // equality between the tensors.
        let precision = vg.fresh_tvar();
        let rank = vg.fresh_rvar();
        let tensor = Type::Tensor(vec![Dim::Rank(rank)], TensorPrec::Var(precision));
        env.bind(
            "test_assert_close_tensor".to_string(),
            Scheme {
                constraints: vec![],
                tvars: vec![precision],
                tvar_restrictions: vec![(precision, TypeVarRestriction::ActiveFloat)],
                dvars: vec![],
                rvars: vec![rank],
                body: Type::Fn(
                    vec![
                        borrowed(tensor.clone()),
                        borrowed(tensor),
                        Type::Var(precision),
                        Type::Prim(Prim::String),
                    ],
                    Box::new(Type::Unit),
                ),
            },
        );
    }
    {
        // test_assert_eq_tensor: (tensor a, tensor a, string) -> unit
        let tensor_tv = vg.fresh_tvar();
        env.bind(
            "test_assert_eq_tensor".to_string(),
            Scheme {
                constraints: vec![],
                tvars: vec![tensor_tv],
                tvar_restrictions: vec![],
                dvars: vec![],
                rvars: vec![],
                body: Type::Fn(
                    vec![
                        borrowed(Type::Var(tensor_tv)),
                        borrowed(Type::Var(tensor_tv)),
                        Type::Prim(Prim::String),
                    ],
                    Box::new(Type::Unit),
                ),
            },
        );
    }
    env.bind(
        "char_code".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::String)],
                Box::new(Type::Prim(Prim::Int64)),
            ),
        },
    );
    env.bind(
        "char_from_code".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::Int64)],
                Box::new(Type::Prim(Prim::String)),
            ),
        },
    );
    generic_unop("string_len", &mut env, &mut vg);
    generic_binop("string_concat", &mut env, &mut vg);
    generic_unop("string_trim", &mut env, &mut vg);
    generic_triop("string_slice", &mut env, &mut vg);
    env.bind(
        "string_contains".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::String), Type::Prim(Prim::String)],
                Box::new(Type::Prim(Prim::Bool)),
            ),
        },
    );
    env.bind(
        "string_starts_with".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::String), Type::Prim(Prim::String)],
                Box::new(Type::Prim(Prim::Bool)),
            ),
        },
    );
    env.bind(
        "string_ends_with".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Prim(Prim::String), Type::Prim(Prim::String)],
                Box::new(Type::Prim(Prim::Bool)),
            ),
        },
    );
    generic_unop_borrow("to_string", &mut env, &mut vg);
    generic_unop("to_int", &mut env, &mut vg);
    generic_unop("to_float", &mut env, &mut vg);
    generic_unop_borrow("rank", &mut env, &mut vg);
    generic_binop_first_borrow("shape", &mut env, &mut vg);
    generic_unop_borrow("numel", &mut env, &mut vg);
    generic_unop_borrow("tensor_to_scalar", &mut env, &mut vg);
    generic_unop("scalar_to_tensor", &mut env, &mut vg);
    generic_unop("len", &mut env, &mut vg);
    generic_binop("index", &mut env, &mut vg);
    generic_binop("append", &mut env, &mut vg);
    generic_binop("concat", &mut env, &mut vg);
    generic_binop("take", &mut env, &mut vg);
    generic_binop("skip", &mut env, &mut vg);
    // [05-OP-67]: `drop` consumes exactly one value of any type and
    // returns unit. Binding the exact unit result here, rather than a
    // free result variable, keeps a bare `drop` reference as precise as
    // the application route in `infer::app`.
    {
        let consumed = vg.fresh_tvar();
        env.bind(
            "drop".to_string(),
            Scheme {
                constraints: vec![],
                tvars: vec![consumed],
                tvar_restrictions: vec![],
                dvars: vec![],
                rvars: vec![],
                body: Type::Fn(vec![Type::Var(consumed)], Box::new(Type::Unit)),
            },
        );
    }
    generic_binop("chunk", &mut env, &mut vg);
    generic_binop("range", &mut env, &mut vg);
    generic_binop("map", &mut env, &mut vg);
    generic_binop("filter", &mut env, &mut vg);
    let fold_acc = vg.fresh_tvar();
    let fold_item = vg.fresh_tvar();
    let fold_ret = vg.fresh_tvar();
    env.bind(
        "fold".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![fold_acc, fold_item, fold_ret],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Var(fold_acc),
                    Type::Var(fold_item),
                    Type::Var(fold_ret),
                ],
                Box::new(Type::Var(fold_ret)),
            ),
        },
    );
    let scan_acc = vg.fresh_tvar();
    let scan_item = vg.fresh_tvar();
    let scan_ret = vg.fresh_tvar();
    env.bind(
        "scan".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![scan_acc, scan_item, scan_ret],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Var(scan_acc),
                    Type::Var(scan_item),
                    Type::Var(scan_ret),
                ],
                Box::new(Type::Var(scan_ret)),
            ),
        },
    );
    // `tensor_scan(initial: T, fn: (T, i64) -> T, n: i64) -> tensor[n, T]`.
    // The actual constraint shape (scalar `T`, callback signature, i64 `n`,
    // tensor return) is enforced by the special-case arm in
    // `crates/chelis-types/src/infer.rs` so error reporting can pinpoint each
    // role independently. This loose generic scheme is the type-env entry
    // point; it lets the inference engine see three argument slots and a
    // return slot it will overwrite. Same shape as `fold`/`scan` above.
    let tensor_scan_a = vg.fresh_tvar();
    let tensor_scan_b = vg.fresh_tvar();
    let tensor_scan_c = vg.fresh_tvar();
    let tensor_scan_ret = vg.fresh_tvar();
    env.bind(
        "tensor_scan".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![tensor_scan_a, tensor_scan_b, tensor_scan_c, tensor_scan_ret],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![
                    Type::Var(tensor_scan_a),
                    Type::Var(tensor_scan_b),
                    Type::Var(tensor_scan_c),
                ],
                Box::new(Type::Var(tensor_scan_ret)),
            ),
        },
    );
    generic_binop("partition", &mut env, &mut vg);
    generic_binop("flat_map", &mut env, &mut vg);
    generic_unop("flatten", &mut env, &mut vg);
    generic_binop("zip", &mut env, &mut vg);
    generic_unop("enumerate", &mut env, &mut vg);
    generic_unop("dict_of", &mut env, &mut vg);
    generic_binop("dict_get", &mut env, &mut vg);
    generic_binop("dict_contains", &mut env, &mut vg);
    generic_binop("dict_remove", &mut env, &mut vg);
    generic_triop("dict_insert", &mut env, &mut vg);
    generic_binop("dict_merge", &mut env, &mut vg);
    generic_unop("dict_keys", &mut env, &mut vg);
    generic_unop("dict_values", &mut env, &mut vg);
    generic_unop("dict_entries", &mut env, &mut vg);
    generic_unop("to_tensor", &mut env, &mut vg);
    generic_unop_borrow("to_list", &mut env, &mut vg);
    generic_binop("pad_sequences", &mut env, &mut vg);
    generic_triop("pad_sequences_to", &mut env, &mut vg);
    // chelis#2524: [05-OP-60] and [05-OP-38] make every argument and result
    // type of the file and process builtins exact, so each carries its exact
    // signature and ordinary unification rejects any other argument. A generic
    // scheme here admitted an `f32` path. The IO effect is still assigned in
    // `chelis-effects` by name. `process_run` is eval/test-only (Hull Phase 0a)
    // and rejected by the C/HIP build backends.
    let string = || Type::Prim(Prim::String);
    let list_of = |element: Type| Type::Adt("List".to_string(), vec![element]);
    let mapped_file = || Type::Adt("MappedFile".to_string(), Vec::new());
    for (name, params, result) in [
        ("read_file", vec![string()], string()),
        ("write_file", vec![string(), string()], Type::Unit),
        ("read_lines", vec![string()], list_of(string())),
        (
            "read_bytes",
            vec![string()],
            list_of(Type::Prim(Prim::Int64)),
        ),
        ("file_exists", vec![string()], Type::Prim(Prim::Bool)),
        ("list_dir", vec![string()], list_of(string())),
        ("mmap_file", vec![string()], mapped_file()),
        (
            "process_run",
            vec![string(), list_of(string())],
            Type::Tuple(vec![Type::Prim(Prim::Int64), string(), string()]),
        ),
    ] {
        env.bind(
            name.to_string(),
            Scheme::mono(Type::Fn(params, Box::new(result))),
        );
    }
    generic_binop("round_to", &mut env, &mut vg);
    // Host-lane CSV I/O (chelis#903); the concrete contracts live in
    // `check_csv_builtin_signature` (infer/app_hostio.rs).
    generic_unop("parse_csv", &mut env, &mut vg);
    generic_unop("to_csv", &mut env, &mut vg);
    generic_binop("csv_f64s", &mut env, &mut vg);
    generic_binop("csv_ints", &mut env, &mut vg);
    generic_binop("csv_strs", &mut env, &mut vg);
    generic_unop("csv_nrows", &mut env, &mut vg);
    generic_unop("csv_cols", &mut env, &mut vg);
    generic_triop("csv_f64", &mut env, &mut vg);
    generic_triop("csv_int", &mut env, &mut vg);
    generic_triop("csv_str", &mut env, &mut vg);
    // chelis#2524: the mapped-file reads take the handle and exact `i64`
    // offsets and counts ([05-OP-60]). The handle keeps the owned parameter
    // mode the generic scheme gave it, so ownership lowering is unchanged.
    env.bind(
        "mmap_read".to_string(),
        Scheme::mono(Type::Fn(
            vec![
                Type::Adt("MappedFile".to_string(), Vec::new()),
                Type::Prim(Prim::Int64),
                Type::Prim(Prim::Int64),
            ],
            Box::new(Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)])),
        )),
    );
    env.bind(
        "mmap_len".to_string(),
        Scheme::mono(Type::Fn(
            vec![Type::Adt("MappedFile".to_string(), Vec::new())],
            Box::new(Type::Prim(Prim::Int64)),
        )),
    );
    generic_triop_second_third_borrow("einsum", &mut env, &mut vg);
    generic_triop_first_borrow("split", &mut env, &mut vg);
    generic_triop_first_two_borrow("gather", &mut env, &mut vg);
    generic_pentaop("scatter", &mut env, &mut vg);
    // scatter_replace is the tensor-lane sparse builtin that lowers
    // to RiscOp::Scatter (last-write-wins). Distinct from the
    // host-lane `scatter(base, indices, updates, axis, mode)` which
    // remains the pentaop form with a string mode argument.
    generic_quadop("scatter_replace", &mut env, &mut vg);
    // scatter_elements is the tensor-lane sparse builtin that lowers to
    // RiscOp::ScatterElements (ONNX ScatterElements, spec §3.5.1):
    // (data, indices, updates, axis). Same arity as scatter_replace but
    // with the element-wise shape contract enforced at IR verify time.
    generic_quadop("scatter_elements", &mut env, &mut vg);
    generic_triop_all_borrow("where", &mut env, &mut vg);
    generic_binop_first_borrow("cumsum", &mut env, &mut vg);
    generic_binop_first_borrow("sort", &mut env, &mut vg);
    generic_triop_first_borrow("diagonal", &mut env, &mut vg);
    generic_triop_first_borrow("trace", &mut env, &mut vg);
    generic_triop_all_borrow("clamp", &mut env, &mut vg);

    (env, vg)
}

pub fn register_prelude_adts(env: &mut Env, vg: &mut VarGen, adt_reg: &mut AdtRegistry) {
    let option_tvar = vg.fresh_tvar();
    let option_type = Type::Adt("Option".to_string(), vec![Type::Var(option_tvar)]);

    env.bind_constructor(
        "Some".to_string(),
        "Option".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![option_tvar],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(vec![Type::Var(option_tvar)], Box::new(option_type.clone())),
        },
    );
    env.bind_constructor(
        "None".to_string(),
        "Option".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![option_tvar],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: option_type.clone(),
        },
    );

    adt_reg
        .defs
        .entry("Option".to_string())
        .or_insert_with(|| AdtDef {
            name: "Option".to_string(),
            type_params: vec!["a".to_string()],
            param_kinds: vec![NominalParamKind::Type],
            param_vars: vec![option_tvar],
            param_args: vec![NominalArg::Type(Type::Var(option_tvar))],
            opaque: false,
            defining_module: None,
            variants: vec![
                VariantInfo {
                    name: "Some".to_string(),
                    fields: vec![(None, Type::Var(option_tvar))],
                },
                VariantInfo {
                    name: "None".to_string(),
                    fields: Vec::new(),
                },
            ],
        });

    let list_tvar = vg.fresh_tvar();
    let list_type = Type::Adt("List".to_string(), vec![Type::Var(list_tvar)]);

    env.bind_constructor(
        "Cons".to_string(),
        "List".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![list_tvar],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: Type::Fn(
                vec![Type::Var(list_tvar), list_type.clone()],
                Box::new(list_type.clone()),
            ),
        },
    );
    env.bind_constructor(
        "Nil".to_string(),
        "List".to_string(),
        Scheme {
            constraints: vec![],
            tvars: vec![list_tvar],
            tvar_restrictions: vec![],
            dvars: vec![],
            rvars: vec![],
            body: list_type.clone(),
        },
    );

    adt_reg
        .defs
        .entry("List".to_string())
        .or_insert_with(|| AdtDef {
            name: "List".to_string(),
            type_params: vec!["a".to_string()],
            param_kinds: vec![NominalParamKind::Type],
            param_vars: vec![list_tvar],
            param_args: vec![NominalArg::Type(Type::Var(list_tvar))],
            opaque: false,
            defining_module: None,
            variants: vec![
                VariantInfo {
                    name: "Cons".to_string(),
                    fields: vec![(None, Type::Var(list_tvar)), (None, list_type.clone())],
                },
                VariantInfo {
                    name: "Nil".to_string(),
                    fields: Vec::new(),
                },
            ],
        });

    adt_reg
        .defs
        .entry("MappedFile".to_string())
        .or_insert_with(|| AdtDef {
            name: "MappedFile".to_string(),
            type_params: Vec::new(),
            param_kinds: Vec::new(),
            param_vars: Vec::new(),
            param_args: Vec::new(),
            opaque: false,
            defining_module: None,
            variants: Vec::new(),
        });
}

/// Capacity-census enumeration source for Rust-registered prelude value
/// ADTs (spec/design/dtype_semantics.md §C6, the `prelude-adt-numeric`
/// leg): every prelude ADT whose registration lives in this file, exactly
/// as registered. The census tripwire
/// (crates/chelis-cli/tests/capacity_census_tripwire.rs) renders these
/// defs through the same canonical shape identity as the
/// `packages/chelis-std` `.ch` ADT rows, so a numeric field added to a
/// prelude ADT is a censused row -- never an unenumerated numeric channel.
///
/// Deliberately reconstructed from the single registration path
/// (`register_prelude_adts`) rather than a hand-maintained list: an ADT
/// registered there but absent here is impossible.
pub fn prelude_adt_defs() -> Vec<AdtDef> {
    let (mut env, mut vg) = builtin_env();
    let mut adt_reg = AdtRegistry::new();
    register_prelude_adts(&mut env, &mut vg, &mut adt_reg);
    let mut defs: Vec<AdtDef> = adt_reg.defs.into_values().collect();
    defs.sort_by(|a, b| a.name.cmp(&b.name));
    defs
}

/// Names that the inference engine should special-case for return type.
/// cmplt, eq, neq, lt, lte, gte return tensor[D, bool] instead of tensor[D, p].
#[allow(dead_code)] // Used by inference engine (Step 6)
pub const COMPARISON_OPS: &[&str] = &["cmplt", "eq", "neq", "lt", "gt", "lte", "gte"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_env_has_add() {
        let (env, _) = builtin_env();
        assert!(env.lookup("add").is_some());
    }

    // chelis#258 Tier-2 rank polymorphism: shape-class classification lock.

    /// Pin the exact shape-identity allowlist. A change here is the one place
    /// where a builtin becomes admissible inside a rank-polymorphic `..r`
    /// body, so it must be deliberate: misclassifying a shape-rewriting op as
    /// Identity is a §4.2 soundness hole. Every other builtin must be
    /// `Rewriting` (the safe default).
    #[test]
    fn shape_class_identity_set_is_pinned() {
        let identity: &[&str] = &[
            "add",
            "mul",
            "sub",
            "div",
            "floor_div",
            "trunc_div",
            "mod",
            "max_elem",
            "min_elem",
            "neg",
            "recip",
            "exp",
            "log",
            "sin",
            "sqrt",
            "cos",
            "tan",
            "atan",
            "abs",
            "floor",
            "ceil",
            "round",
            "relu",
            "sigmoid",
            "tanh",
            "silu",
            "gelu",
            "not",
            "clamp",
            "uniform_like",
            "where",
            "eq",
            "neq",
            "lt",
            "gt",
            "lte",
            "gte",
            "cmplt",
            "bitand",
            "bitor",
            "bitxor",
            "shl",
            "shr",
            "and",
            "or",
        ];
        // Named-axis reductions and named-axis expand are NameTracked
        // (admitted in a `..r` body — the procedural arm is the gate);
        // everything else outside `identity` is Rewriting. chelis#340 added
        // the full reduction family (`max_reduce`/`min_reduce`/
        // `prod_reduce`/`argmax_reduce`/`argmin_reduce`) here: they route
        // through the tensor-DAG kernel lane like `sum`/`mean` and build+run
        // end-to-end in a rank-poly body.
        let name_tracked: &[&str] = &[
            "sum",
            "count",
            "mean",
            "max_reduce",
            "min_reduce",
            "prod_reduce",
            "argmax_reduce",
            "argmin_reduce",
            "expand",
            "insert",
        ];
        for name in BUILTIN_NAMES {
            let expected = if identity.contains(name) {
                ShapeClass::Identity
            } else if name_tracked.contains(name) {
                ShapeClass::NameTracked
            } else {
                ShapeClass::Rewriting
            };
            assert_eq!(
                shape_class(name),
                expected,
                "builtin `{name}` shape-class drifted from the pinned set"
            );
        }
        // Spot-check the positional shape-rewriters stay Rewriting (the §4.2
        // traps): a positional index is meaningless at symbolic rank.
        for op in ["permute", "reshape", "matmul", "gather", "conv"] {
            assert_eq!(
                shape_class(op),
                ShapeClass::Rewriting,
                "`{op}` must be Rewriting"
            );
        }
        // And the named-axis ops are admitted as NameTracked. `expand`
        // moved from Rewriting in chelis#339: its procedural arm now
        // rejects positional axes at symbolic rank, so admitting it in a
        // `..r` body cannot hide a transposition.
        assert_eq!(shape_class("sum"), ShapeClass::NameTracked);
        assert_eq!(shape_class("count"), ShapeClass::NameTracked);
        assert_eq!(shape_class("mean"), ShapeClass::NameTracked);
        assert_eq!(shape_class("expand"), ShapeClass::NameTracked);
        assert_eq!(shape_class("insert"), ShapeClass::NameTracked);
        // chelis#340: the rest of the reduction family is name-tracked too.
        assert_eq!(shape_class("max_reduce"), ShapeClass::NameTracked);
        assert_eq!(shape_class("min_reduce"), ShapeClass::NameTracked);
        assert_eq!(shape_class("prod_reduce"), ShapeClass::NameTracked);
        assert_eq!(shape_class("argmax_reduce"), ShapeClass::NameTracked);
        assert_eq!(shape_class("argmin_reduce"), ShapeClass::NameTracked);
    }

    /// Lock the §4 "Complete closed vocabulary" block of
    /// `docs/CHELIS_SURFACE.md` to `BUILTIN_NAMES`. That section bills
    /// itself as the audit surface ("diff `BUILTIN_NAMES` against this
    /// block") and promises to list every name verbatim, so any drift in
    /// either direction is a documentation bug: a builtin missing from the
    /// doc (an op that silently fell out of the inventory) or a doc entry
    /// that is not in the array (`const`/`load`/`dropout` live outside it by
    /// design — see the §4 preamble — and must not appear in the block).
    /// `include_str!` makes the doc a compile-time dependency of this test,
    /// so a moved or deleted file fails loudly instead of skipping.
    #[test]
    fn doc_surface_section_4_mirrors_builtin_names() {
        use std::collections::BTreeSet;

        const DOC: &str = include_str!("../../../docs/CHELIS_SURFACE.md");

        // Isolate the single fenced code block under the "## 4." heading.
        let after_heading = DOC
            .split_once("## 4. Complete closed vocabulary")
            .expect("docs/CHELIS_SURFACE.md must contain the '## 4.' section")
            .1;
        let fence_open = after_heading
            .find("```")
            .expect("§4 must contain a fenced code block");
        // Skip the rest of the opening fence line (any info string).
        let body = &after_heading[fence_open + 3..];
        let body = &body[body.find('\n').map(|i| i + 1).unwrap_or(0)..];
        let block = body
            .split_once("```")
            .expect("§4 fenced code block must be closed")
            .0;

        // The block is bare builtin names plus three fixed category labels;
        // strip the labels and every remaining token is a builtin name.
        let cleaned = block
            .replace("Tier-1 DAG:", " ")
            .replace("Tier-2 DAG:", " ")
            .replace("Host lane:", " ");
        let doc_names: BTreeSet<&str> = cleaned.split_whitespace().collect();

        // Fail loudly if the extractor silently produced garbage rather than
        // comparing a malformed set against the array.
        assert!(
            doc_names.contains("add")
                && doc_names.contains("matmul")
                && doc_names.contains("read_file")
                && doc_names.len() > 100,
            "§4 vocabulary parse looks wrong ({} tokens); the block format \
             under '## 4.' likely changed and this extractor needs updating",
            doc_names.len()
        );

        let array_names: BTreeSet<&str> = BUILTIN_NAMES.iter().copied().collect();
        let missing_from_doc: Vec<&str> = array_names.difference(&doc_names).copied().collect();
        let extra_in_doc: Vec<&str> = doc_names.difference(&array_names).copied().collect();

        assert!(
            missing_from_doc.is_empty() && extra_in_doc.is_empty(),
            "docs/CHELIS_SURFACE.md §4 has drifted from BUILTIN_NAMES \
             (crates/chelis-types/src/builtins.rs).\n  in BUILTIN_NAMES but \
             missing from §4: {missing_from_doc:?}\n  in §4 but not a \
             builtin: {extra_in_doc:?}"
        );
    }

    #[test]
    fn builtin_env_has_matmul() {
        let (env, _) = builtin_env();
        assert!(env.lookup("matmul").is_some());
    }

    #[test]
    fn builtin_env_has_layer_norm_and_conv() {
        let (env, _) = builtin_env();
        assert!(env.lookup("layer_norm").is_some());
        assert!(env.lookup("conv").is_some());
    }

    #[test]
    fn builtin_env_has_cmplt() {
        let (env, _) = builtin_env();
        assert!(env.lookup("cmplt").is_some());
    }

    #[test]
    fn builtin_env_has_logical_ops() {
        let (env, _) = builtin_env();
        assert!(env.lookup("and").is_some());
        assert!(env.lookup("or").is_some());
        assert!(env.lookup("not").is_some());
    }

    #[test]
    fn builtin_env_has_phase3c_string_helpers() {
        let (env, _) = builtin_env();
        assert!(env.lookup("string_slice").is_some());
        assert!(env.lookup("string_contains").is_some());
        assert!(env.lookup("string_starts_with").is_some());
        assert!(env.lookup("string_ends_with").is_some());
        assert!(env.lookup("string_trim").is_some());
        assert!(env.lookup("to_int").is_some());
        assert!(env.lookup("to_float").is_some());
    }

    #[test]
    fn builtin_env_has_phase3c_integer_helpers() {
        let (env, _) = builtin_env();
        assert!(env.lookup("mod").is_some());
        assert!(env.lookup("bitand").is_some());
        assert!(env.lookup("bitor").is_some());
        assert!(env.lookup("bitxor").is_some());
        assert!(env.lookup("shl").is_some());
        assert!(env.lookup("shr").is_some());
    }

    #[test]
    fn builtin_env_has_process_run_and_it_is_a_known_name() {
        // Hull subprocess exec: process_run is a registered 2-arg builtin and
        // appears in the closed BUILTIN_NAMES vocabulary (so the lint naming
        // gate and host-lane resolver recognize it).
        let (env, _) = builtin_env();
        let scheme = env.lookup("process_run").expect("process_run registered");
        match &scheme.body {
            Type::Fn(params, _) => assert_eq!(
                params.len(),
                2,
                "process_run takes (cmd, args), got arity {}",
                params.len()
            ),
            other => panic!("process_run should be a function type, got {other:?}"),
        }
        assert!(
            BUILTIN_NAMES.contains(&"process_run"),
            "process_run must be in the closed BUILTIN_NAMES vocabulary"
        );
    }

    #[test]
    fn builtin_env_does_not_register_unknown_name() {
        // Negative parity: a name we never register stays absent, so the
        // process_run presence assertion above is not vacuously true.
        let (env, _) = builtin_env();
        assert!(env.lookup("process_run_definitely_unregistered").is_none());
        assert!(!BUILTIN_NAMES.contains(&"process_run_definitely_unregistered"));
    }

    #[test]
    fn register_prelude_adts_adds_option_constructors() {
        let (mut env, mut vg) = builtin_env();
        let mut adt_reg = AdtRegistry::new();
        register_prelude_adts(&mut env, &mut vg, &mut adt_reg);

        assert!(env.lookup("Some").is_some());
        assert!(env.lookup("None").is_some());
        assert!(adt_reg.lookup("Option").is_some());
        assert_eq!(
            adt_reg.variant_names("Option").expect("option variants"),
            vec!["Some".to_string(), "None".to_string()]
        );
    }

    #[test]
    fn builtin_env_has_csv_io_builtins() {
        // chelis#903 host-lane CSV I/O: every builtin is registered with
        // the declared arity; infer/app_hostio.rs pins the canonical
        // List[Dict[string,string]] table carrier.
        let (env, _) = builtin_env();
        for (name, arity) in [
            ("parse_csv", 1),
            ("to_csv", 1),
            ("csv_f64s", 2),
            ("csv_ints", 2),
            ("csv_strs", 2),
            ("csv_nrows", 1),
            ("csv_cols", 1),
            ("csv_f64", 3),
            ("csv_int", 3),
            ("csv_str", 3),
        ] {
            let scheme = env
                .lookup(name)
                .unwrap_or_else(|| panic!("`{name}` must be registered"));
            match &scheme.body {
                Type::Fn(params, _) => assert_eq!(
                    params.len(),
                    arity,
                    "`{name}` should take {arity} args, got {}",
                    params.len()
                ),
                other => panic!("`{name}` should be a function type, got {other:?}"),
            }
            assert!(
                BUILTIN_NAMES.contains(&name),
                "`{name}` must be in the closed BUILTIN_NAMES vocabulary"
            );
        }
    }

    #[test]
    fn builtin_env_missing_name_returns_none() {
        let (env, _) = builtin_env();
        assert!(env.lookup("nonexistent").is_none());
    }

    #[test]
    fn instantiate_add_produces_fn_type() {
        let (env, mut vg) = builtin_env();
        let scheme = env.lookup("add").unwrap();
        let subst = crate::unify::Subst::new();
        let ty = env.instantiate(scheme, &mut vg, &subst);
        match ty {
            Type::Fn(args, _ret) => assert_eq!(args.len(), 2),
            other => panic!("expected Fn, got {other:?}"),
        }
    }

    #[test]
    fn add_args_unify_with_same_tensor() {
        use crate::unify::{Subst, unify};

        let (env, mut vg) = builtin_env();
        let add_scheme = env.lookup("add").unwrap();
        let mut subst = Subst::new();
        let add_ty = env.instantiate(add_scheme, &mut vg, &subst);

        // add should accept two tensors of the same type
        let tensor_f32 = Type::Tensor(
            vec![Dim::Name("batch".into())],
            TensorPrec::Concrete(Prim::F32),
        );
        let expected_fn = Type::Fn(
            vec![
                Type::Ref(Box::new(tensor_f32.clone())),
                Type::Ref(Box::new(tensor_f32.clone())),
            ],
            Box::new(tensor_f32),
        );

        assert!(unify(&add_ty, &expected_fn, &mut subst).is_ok());
    }

    #[test]
    fn add_rejects_mixed_precision() {
        use crate::unify::{Subst, unify};

        let (env, mut vg) = builtin_env();
        let add_scheme = env.lookup("add").unwrap();
        let mut subst = Subst::new();
        let add_ty = env.instantiate(add_scheme, &mut vg, &subst);

        // add(tensor[batch,f32], tensor[batch,bf16]) should fail
        let t1 = Type::Tensor(
            vec![Dim::Name("batch".into())],
            TensorPrec::Concrete(Prim::F32),
        );
        let t2 = Type::Tensor(
            vec![Dim::Name("batch".into())],
            TensorPrec::Concrete(Prim::Bf16),
        );
        let bad_fn = Type::Fn(
            vec![Type::Ref(Box::new(t1)), Type::Ref(Box::new(t2))],
            Box::new(Type::Var(vg.fresh_tvar())),
        );

        assert!(unify(&add_ty, &bad_fn, &mut subst).is_err());
    }

    #[test]
    fn add_rejects_tensor_plus_int() {
        use crate::unify::{Subst, unify};

        let (env, mut vg) = builtin_env();
        let add_scheme = env.lookup("add").unwrap();
        let mut subst = Subst::new();
        let add_ty = env.instantiate(add_scheme, &mut vg, &subst);

        // add(tensor[batch,f32], i32) should fail
        let t1 = Type::Tensor(
            vec![Dim::Name("batch".into())],
            TensorPrec::Concrete(Prim::F32),
        );
        let t2 = Type::Prim(Prim::Int32);
        let bad_fn = Type::Fn(
            vec![Type::Ref(Box::new(t1)), Type::Ref(Box::new(t2))],
            Box::new(Type::Var(vg.fresh_tvar())),
        );

        assert!(unify(&add_ty, &bad_fn, &mut subst).is_err());
    }

    /// Issue #912: BUILTINS table must contain every entry from BUILTIN_NAMES
    /// and vice versa. This ensures the consolidated table doesn't drift.
    #[test]
    fn builtins_table_matches_builtin_names() {
        use std::collections::BTreeSet;
        let table_names: BTreeSet<&str> = super::BUILTINS.iter().map(|b| b.name).collect();
        let array_names: BTreeSet<&str> = super::BUILTIN_NAMES.iter().copied().collect();

        let in_table_not_array: Vec<&str> = table_names.difference(&array_names).copied().collect();
        let in_array_not_table: Vec<&str> = array_names.difference(&table_names).copied().collect();

        assert!(
            in_table_not_array.is_empty() && in_array_not_table.is_empty(),
            "BUILTINS table and BUILTIN_NAMES must contain the same names.\n\
             In BUILTINS but not BUILTIN_NAMES: {in_table_not_array:?}\n\
             In BUILTIN_NAMES but not BUILTINS: {in_array_not_table:?}"
        );
    }

    #[test]
    fn every_builtin_has_an_exact_nondefault_semantic_discovery_declaration() {
        use std::collections::BTreeSet;

        for builtin in BUILTINS {
            let domains: BTreeSet<_> = builtin.capability.domains.iter().copied().collect();
            assert!(
                !domains.is_empty(),
                "BuiltinDecl `{}` must select at least one semantic domain",
                builtin.name
            );
            assert_eq!(
                domains.len(),
                builtin.capability.domains.len(),
                "BuiltinDecl `{}` has a duplicate semantic domain",
                builtin.name
            );

            let cases: BTreeSet<_> = builtin.capability.sibling_cases.iter().copied().collect();
            assert_eq!(
                cases.len(),
                builtin.capability.sibling_cases.len(),
                "BuiltinDecl `{}` has a duplicate sibling case",
                builtin.name
            );

            for case in builtin.capability.sibling_cases {
                assert_ne!(
                    case.domain,
                    BuiltinSemanticDomain::Numeric,
                    "BuiltinDecl `{}` put a sibling case in Table A's Numeric domain",
                    builtin.name
                );
                assert!(
                    domains.contains(&case.domain),
                    "BuiltinDecl `{}` sibling case {:?} names undeclared domain {:?}",
                    builtin.name,
                    case.case,
                    case.domain
                );
            }

            for sibling_domain in [
                BuiltinSemanticDomain::Container,
                BuiltinSemanticDomain::Boundary,
            ] {
                assert_eq!(
                    domains.contains(&sibling_domain),
                    builtin
                        .capability
                        .sibling_cases
                        .iter()
                        .any(|case| case.domain == sibling_domain),
                    "BuiltinDecl `{}` must enumerate at least one exact case for each selected sibling domain and none for an unselected domain",
                    builtin.name
                );
            }
        }
    }

    /// Issue #912: every BUILTINS entry's shape_class must match the existing
    /// shape_class() function (consistency during migration).
    #[test]
    fn builtins_table_shape_class_consistent() {
        for decl in super::BUILTINS {
            assert_eq!(
                super::shape_class(decl.name),
                decl.shape_class,
                "shape_class mismatch for builtin `{}`",
                decl.name
            );
        }
    }

    #[test]
    fn every_checked_builtin_has_an_exact_inference_route() {
        for decl in super::BUILTINS {
            match decl.inference {
                super::InferenceDisposition::Checked(rule) => assert!(
                    super::has_registered_inference_route(decl.name, rule),
                    "builtin `{}` declares {rule:?} but no exact route owns it",
                    decl.name
                ),
                super::InferenceDisposition::GenericAccepted { reason } => assert!(
                    !reason.trim().is_empty(),
                    "generic acceptance for `{}` requires a reviewable reason",
                    decl.name
                ),
            }
        }
    }

    #[test]
    fn every_inference_route_is_owned_by_the_exact_checked_declaration() {
        for (rule, names) in [
            (
                super::BuiltinInferenceRule::ShapeComputed,
                super::SHAPE_COMPUTED_INFERENCE_BUILTINS,
            ),
            (
                super::BuiltinInferenceRule::Specialized,
                super::SPECIALIZED_INFERENCE_BUILTINS,
            ),
        ] {
            for name in names {
                let decl = super::builtin_decl(name)
                    .unwrap_or_else(|| panic!("inference route names unknown builtin `{name}`"));
                assert_eq!(
                    decl.inference,
                    super::InferenceDisposition::Checked(rule),
                    "inference route for `{name}` is not owned by its exact declaration"
                );
            }
        }
    }
}
