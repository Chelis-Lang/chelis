//! RISC DAG data structure.
//!
//! The DAG is a flat vector of nodes, each referencing earlier nodes by [`NodeId`].
//! Nodes are always in topological order (an invariant maintained by append-only construction).

use chelis_unord::{UnordMap, UnordSet};
use std::fmt;

use chelis_types::types::Prim;
use serde::{Deserialize, Serialize};

use crate::load_store_name::LoadStoreName;

/// Index into the DAG node array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub usize);

/// Tensor type carried on each DAG node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorType {
    pub dims: Vec<DimInfo>,
    pub precision: Prim,
}

impl TensorType {
    /// A default scalar F32 type (no dimensions).
    pub fn scalar_f32() -> Self {
        Self {
            dims: vec![],
            precision: Prim::F32,
        }
    }
}

/// Sentinel `Shrink` bound `end` meaning "to the end of this axis" — a
/// full-axis identity slice whose extent is only known at runtime.
///
/// chelis#368: the reverse-mode adjoint of a `Pad` whose unpadded axes are
/// runtime-derived symbolic dims (the differentiable `concat` lowering)
/// cannot bake a literal `end` for those axes. It emits `(0, SHRINK_TO_END)`
/// there, and since chelis#513 the `Stride` adjoint's trim bound and the
/// `ProdReduce` adjoint's per-element slice bounds emit the same sentinel on
/// symbolic bystander axes. [`bind_symbolic_dims`] resolves the sentinel to
/// the axis's runtime extent (read from the node's bound output type) before
/// evaluation, so the `Shrink` evaluator never sees it. The C backend does
/// not run `bind_symbolic_dims` (the symbolic dim stays a runtime C
/// variable); its `emit_shrink` loop is driven by the output shape, so a
/// well-formed full-axis sentinel (`start == 0` on a symbolic output axis)
/// is already correct structurally and only a MALFORMED sentinel fails loud
/// (chelis#551). The Metal lane requires concrete movement shapes and
/// rejects a symbolic-axis shrink with a clean error before the sentinel
/// matters; the HIP shrink kernel, like C, reads only the per-axis `start`.
pub const SHRINK_TO_END: usize = usize::MAX;

/// A literal axis carried by a folded tensor-shape read.
///
/// Slice A of chelis#1277 admits only the normalized literal form. A
/// node-valued axis is owned by chelis#1298 and will extend this closed carrier
/// together with its own wire-schema version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RtAxis {
    /// A normalized axis in `0..rank(tensor)`.
    Lit(i32),
}

/// A single runtime-capable tensor extent.
///
/// chelis#616: movement bounds were compile-time `usize` only, so a runtime
/// (`shape()`-derived) `shrink`/`stride` bound was dropped to empty at lowering
/// and the windowed axis degraded to a `Named("*")` wildcard. A `RtDim` can now
/// be a compile-time literal, the full-axis sentinel, a *runtime* value read
/// from a rank-0 integer node, or a folded read from a tensor input's shape.
///
/// `Node(i)` is an **absolute** index into the owning node's `inputs`, where
/// `inputs[0]` is always the tensor operand and `inputs[1..]` are rank-0 integer
/// bound scalars (a [`RiscOp::Shape`] read or an integer-arithmetic chain over
/// one). Invariant (checked by `verify`): for `RtDim::Node(i)`,
/// `1 <= i < inputs.len()` and `inputs[i]` is a rank-0 integer node. Keeping the
/// index absolute means `Node(i)` reads exactly like `inputs[i]` at every
/// dispatch site with no offset arithmetic.
///
/// `ToEnd` promotes the [`SHRINK_TO_END`] sentinel to an explicit variant; it is
/// only legal as a `Shrink` `end` (verify rejects it elsewhere). The resolved
/// runtime extent it stands for is still [`SHRINK_TO_END`] after
/// `bind_symbolic_dims` / in the backend loop bookkeeping.
///
/// `Sym(name)` is a symbolic dimension declared elsewhere (a `Load`'s named
/// axis, e.g. a bystander `batch`). It is only legal in a `Reshape` target
/// (verify rejects it in movement-bound positions): a movement bound is a
/// scalar *value*, lowered to `Lit` or `Node`, while a reshape target may
/// restate an axis the checker already named.
///
/// `InputAxis { tensor, axis }` is an **absolute** input-slot reference to a
/// tensor whose shape supplies this extent. It is legal only in `Expand` and
/// `Reshape`; Slice A produces only `RtAxis::Lit` axes. The tensor is a real
/// shape-only value edge, so ordinary DCE and graph rebuilding preserve it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RtDim {
    /// A compile-time-constant bound.
    Lit(usize),
    /// The runtime end of a (symbolic) axis; `Shrink` `end` only.
    ToEnd,
    /// A runtime bound read from `inputs[i]` (a rank-0 integer node).
    Node(usize),
    /// A symbolic dimension declared elsewhere; `Reshape` targets only.
    Sym(String),
    /// A folded read of `inputs[tensor]` shape metadata.
    InputAxis { tensor: usize, axis: RtAxis },
}

impl RtDim {
    /// The compile-time value, if this bound is a literal.
    pub fn as_lit(&self) -> Option<usize> {
        match self {
            RtDim::Lit(n) => Some(*n),
            _ => None,
        }
    }

    /// Whether this bound is only known at runtime (`Node`, `InputAxis`, or
    /// `ToEnd`).
    /// `Sym` is not "runtime" in this sense: its extent is bound from an
    /// input shape before evaluation, not computed by a node.
    pub fn is_runtime(&self) -> bool {
        matches!(
            self,
            RtDim::Node(_) | RtDim::InputAxis { .. } | RtDim::ToEnd
        )
    }

    /// The `inputs` slot index if this bound is node-valued.
    pub fn node_input(&self) -> Option<usize> {
        match self {
            RtDim::Node(i) => Some(*i),
            _ => None,
        }
    }

    /// The behavior-preserving translation of a static dimension descriptor:
    /// a known extent (literal or bound named dim) becomes `Lit`, an unbound
    /// named dim becomes `Sym`. Used wherever a `Reshape` target is built
    /// from an existing `TensorType` (grad adjoints, tier2 lowering, vmap).
    pub fn from_dim_info(dim: &DimInfo) -> RtDim {
        match dim {
            DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => RtDim::Lit(*n),
            DimInfo::Named(name, None) => RtDim::Sym(name.clone()),
        }
    }
}

/// Dimension descriptor for a tensor axis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DimInfo {
    /// A named dimension with an optional known size.
    Named(String, Option<usize>),
    /// A fixed literal size.
    Lit(usize),
}

/// Runtime-capable dimension expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DimExpr {
    Concrete(usize),
    Sym(String),
    Mul(Box<DimExpr>, Box<DimExpr>),
    Div(Box<DimExpr>, Box<DimExpr>),
}

impl DimExpr {
    pub fn evaluate(&self, bindings: &UnordMap<String, usize>) -> Result<usize, String> {
        match self {
            Self::Concrete(value) => Ok(*value),
            Self::Sym(name) => bindings
                .get(name)
                .copied()
                .ok_or_else(|| format!("missing symbolic dimension binding `{name}`")),
            Self::Mul(lhs, rhs) => lhs
                .evaluate(bindings)?
                .checked_mul(rhs.evaluate(bindings)?)
                .ok_or_else(|| "symbolic dimension product overflowed usize".to_string()),
            Self::Div(lhs, rhs) => {
                let lhs = lhs.evaluate(bindings)?;
                let rhs = rhs.evaluate(bindings)?;
                if rhs == 0 {
                    return Err("symbolic dimension expression divided by zero".to_string());
                }
                if lhs % rhs != 0 {
                    return Err(format!(
                        "symbolic dimension expression `{self}` is not evenly divisible"
                    ));
                }
                Ok(lhs / rhs)
            }
        }
    }

    pub fn is_concrete(&self) -> bool {
        self.as_concrete().is_some()
    }

    pub fn as_concrete(&self) -> Option<usize> {
        match self {
            Self::Concrete(value) => Some(*value),
            Self::Sym(_) => None,
            Self::Mul(lhs, rhs) => lhs.as_concrete()?.checked_mul(rhs.as_concrete()?),
            Self::Div(lhs, rhs) => {
                let lhs = lhs.as_concrete()?;
                let rhs = rhs.as_concrete()?;
                if rhs == 0 || lhs % rhs != 0 {
                    None
                } else {
                    Some(lhs / rhs)
                }
            }
        }
    }

    pub fn symbolic_names(&self) -> UnordSet<String> {
        let mut names = UnordSet::new();
        self.collect_symbolic_names(&mut names);
        names
    }

    fn collect_symbolic_names(&self, names: &mut UnordSet<String>) {
        match self {
            Self::Concrete(_) => {}
            Self::Sym(name) => {
                names.insert(name.clone());
            }
            Self::Mul(lhs, rhs) | Self::Div(lhs, rhs) => {
                lhs.collect_symbolic_names(names);
                rhs.collect_symbolic_names(names);
            }
        }
    }

    pub fn bind(&self, bindings: &UnordMap<String, usize>) -> Result<Self, String> {
        match self {
            Self::Concrete(value) => Ok(Self::Concrete(*value)),
            Self::Sym(name) => {
                Ok(Self::Concrete(bindings.get(name).copied().ok_or_else(
                    || format!("missing symbolic dimension binding `{name}`"),
                )?))
            }
            Self::Mul(lhs, rhs) => Ok(Self::Mul(
                Box::new(lhs.bind(bindings)?),
                Box::new(rhs.bind(bindings)?),
            )),
            Self::Div(lhs, rhs) => Ok(Self::Div(
                Box::new(lhs.bind(bindings)?),
                Box::new(rhs.bind(bindings)?),
            )),
        }
    }

    /// chelis#616: like [`Self::bind`], but a symbol in `exempt` (an
    /// op-declared runtime dim, whose value does not exist until its owning
    /// op evaluates) or an ANONYMOUS wildcard (resolved by the evaluator
    /// from a shape-dep value) stays symbolic instead of raising the loud
    /// missing-binding error.
    pub fn bind_except(
        &self,
        bindings: &UnordMap<String, usize>,
        exempt: &UnordSet<String>,
    ) -> Result<Self, String> {
        match self {
            Self::Sym(name)
                if (exempt.contains(name) || name.is_empty() || name == "*")
                    && !bindings.contains_key(name) =>
            {
                Ok(Self::Sym(name.clone()))
            }
            Self::Mul(lhs, rhs) => Ok(Self::Mul(
                Box::new(lhs.bind_except(bindings, exempt)?),
                Box::new(rhs.bind_except(bindings, exempt)?),
            )),
            Self::Div(lhs, rhs) => Ok(Self::Div(
                Box::new(lhs.bind_except(bindings, exempt)?),
                Box::new(rhs.bind_except(bindings, exempt)?),
            )),
            other => other.bind(bindings),
        }
    }
}

impl From<&DimInfo> for DimExpr {
    fn from(value: &DimInfo) -> Self {
        match value {
            DimInfo::Lit(size) => Self::Concrete(*size),
            DimInfo::Named(_, Some(size)) => Self::Concrete(*size),
            DimInfo::Named(name, None) => Self::Sym(name.clone()),
        }
    }
}

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Concrete(value) => write!(f, "{value}"),
            Self::Sym(name) => f.write_str(name),
            Self::Mul(lhs, rhs) => write!(f, "({lhs} * {rhs})"),
            Self::Div(lhs, rhs) => write!(f, "({lhs} / {rhs})"),
        }
    }
}

/// Where a symbolic dim's runtime value comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolicDimSource {
    /// Declared from an input tensor's shape: the C prologue emits
    /// `int name = inputs[slot]->shape[axis];` and the eval lane binds the
    /// value from the corresponding input before evaluation.
    Load { input_label: String, axis: usize },
    /// chelis#616/#1277: declared at run time by the owning op itself — a
    /// movement output axis whose extent is computed from a rank-0 bound
    /// scalar or an explicit `InputAxis` metadata read. The C declaration is
    /// emitted inline at the op (the source may be a computed tensor that
    /// does not exist at prologue time); the eval lane resolves the extent
    /// from actual values during evaluation and never pre-binds the symbol. When
    /// the same symbol also has a `Load` source (or an earlier `OpDeclared`
    /// declarer), this site is an equality-guard site: the C emitter aborts
    /// at run time if the op's extent disagrees with the declared value.
    OpDeclared { node: NodeId, axis: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicDimOccurrence {
    pub name: String,
    pub source: SymbolicDimSource,
}

impl SymbolicDimOccurrence {
    fn load(name: &str, input_label: &str, axis: usize) -> Self {
        SymbolicDimOccurrence {
            name: name.to_string(),
            source: SymbolicDimSource::Load {
                input_label: input_label.to_string(),
                axis,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicDimBinding {
    pub name: String,
    pub canonical: SymbolicDimOccurrence,
    pub others: Vec<SymbolicDimOccurrence>,
}

/// One step in a fused elementwise chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedStep {
    pub op: FusedStepOp,
    /// Indices into the fused node's inputs: either an external input index
    /// or a previous step's output index (offset by external input count).
    pub input_indices: Vec<FusedInput>,
}

/// The operation performed by a single fusion step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FusedStepOp {
    Add,
    Sub,
    Mul,
    Div,
    /// Floor division step (chelis#178); see [`RiscOp::FloorDiv`].
    FloorDiv,
    /// Truncating (round-toward-zero) division step, integer-only
    /// (chelis#178); see [`RiscOp::TruncDiv`].
    TruncDiv,
    MaxElem,
    MinElem,
    CmpLt,
    Neg,
    Recip,
    Exp,
    Log,
    Sin,
    Sqrt,
    Cos,
    Tan,
    Atan,
    Abs,
    Floor,
    Ceil,
    Round,
}

/// Reducer selector for [`RiscOp::ReduceWindow`].
///
/// See `spec/05-risc-primitives.md` §2.3.1 for the full surface
/// contract. The shipped Surf builtins map to the four variants:
/// `reduce_window_max` → [`ReduceWindowKind::Max`],
/// `reduce_window_min` → [`ReduceWindowKind::Min`],
/// `reduce_window_sum` → [`ReduceWindowKind::Sum`],
/// `reduce_window_mean` → [`ReduceWindowKind::Mean`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceWindowKind {
    Max,
    Min,
    Sum,
    Mean,
}

/// Direct extrema identity used by the reverse-mode selection adjoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtremaKind {
    Max,
    Min,
}

/// Forward operand whose complete extrema cotangent is materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtremaOperand {
    Left,
    Right,
}

impl ReduceWindowKind {
    /// Canonical Surf builtin name. Used by [`crate::grad::risc_op_name`]
    /// and by the AD rejection error so error messages reference the
    /// user-visible builtin rather than an internal variant.
    pub fn surf_name(self) -> &'static str {
        match self {
            ReduceWindowKind::Max => "reduce_window_max",
            ReduceWindowKind::Min => "reduce_window_min",
            ReduceWindowKind::Sum => "reduce_window_sum",
            ReduceWindowKind::Mean => "reduce_window_mean",
        }
    }
}

/// Input reference within a fused chain.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FusedInput {
    /// Index into the FusedElem node's `inputs` vec (external inputs from the DAG).
    External(usize),
    /// Output of a previous step in the chain (index into the `ops` vec).
    PreviousStep(usize),
}

/// The semantic diagnostic owner of an extent observation. This is separate
/// from the parameter's display name and from the witness's node identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtentWitnessSite {
    Caller,
    LocalExpand,
}

/// One dimension-binder equality a witness owes against ANOTHER witness.
///
/// `spec/04-type-system.md` §4.7.2: a declared result dimension claiming a
/// named extent that is not statically proven equal to the produced size is
/// checked at execution and traps `Domain`. Both quantities are input tensor
/// axes, so §4.7 places the check at function entry, "in declared signature
/// order, before any other operation of the function runs". The obligation
/// therefore lives ON a witness rather than in a separate node scheduled
/// after the body: it becomes due at the later of its two witnesses, which is
/// the node carrying it, and its requirement edge names the earlier one.
///
/// `requirement_declares` records which of the two witnesses DECLARES
/// `claim`. The declaring witness is not always the earlier one - a result
/// `-> tensor[rows, cols]` whose set axis reads `shape(x, 0)` has `cols`
/// declared by the LATER parameter - and the rendering leads with the
/// declaring side, as `axis_sources::entry_extent_guards` does for the
/// `Load`-witnessed form of the same contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtentClaim {
    /// The dimension binder both witnesses must observe equally.
    pub claim: String,
    /// True when the requirement edge is the binder's declaring witness and
    /// this node observed the produced extent; false when this node declares
    /// the binder and the requirement edge observed the produced extent.
    pub requirement_declares: bool,
}

/// A RISC primitive operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiscOp {
    // --- Binary elementwise ---
    Add,
    /// Direct element-wise subtraction. Integer execution checks the exact
    /// mathematical difference at the stored width; this identity must not
    /// be rewritten as `Add(Neg(rhs))`.
    Sub,
    Mul,
    /// Element-wise IEEE-754 division `a / b`. Primitive because the
    /// algebraic rewrite `mul(a, exp(neg(log(b))))` is NaN for
    /// `b ≤ 0` (`log(b)` is undefined there). Backends emit native
    /// `/`; the Rust evaluator uses native `f32`/`f64` division.
    /// **Float operands only since chelis#178** — integer division
    /// uses [`RiscOp::FloorDiv`] / [`RiscOp::TruncDiv`].
    Div,
    /// Element-wise floor division: `floor(a / b)`, rounding the
    /// quotient toward −∞ (chelis#178). Integer operands round toward
    /// −∞ (native `/` plus a remainder-sign correction); float
    /// operands compute `floor(a / b)`. Matches Python `//` / torch /
    /// JAX / numpy `floor_divide`. Non-differentiable (piecewise
    /// constant); `grad` rejects it. See `spec/05-risc-primitives.md`
    /// §2.1.
    FloorDiv,
    /// Element-wise truncating division (round toward zero) — the
    /// C/Rust integer `/` quotient (chelis#178). **Integer operands
    /// only.** This is the exact quotient semantics chelis-std's
    /// `Std.Decimal` arithmetic relies on. Non-differentiable;
    /// `grad` rejects it. See `spec/05-risc-primitives.md` §2.1.
    TruncDiv,
    /// Exact signed remainder, with DivZero traps at the stored width
    /// and dividend-sign semantics under [05-OP-64].
    Mod,
    CmpLt,
    MaxElem,
    /// Direct element-wise minimum selection. This identity preserves the
    /// selected operand bits and must not be rewritten through negation.
    MinElem,

    /// AD-only exact selected-operand cotangent for [`RiscOp::MaxElem`] and
    /// [`RiscOp::MinElem`]. Inputs are `(left, right, cotangent)`; output is
    /// the complete cotangent where `operand` was selected and exact zero
    /// elsewhere under [05-OP-40].
    ExtremaAdjoint {
        kind: ExtremaKind,
        operand: ExtremaOperand,
    },

    /// [05-OP-43] ReLU identity. This node must survive construction and AD
    /// intact so its zero-at-zero adjoint is not replaced by MaxElem's
    /// first-operand tie rule.
    Relu,
    /// AD-only [05-OP-43] selector. Inputs are `(x, cotangent)`; output is the
    /// complete cotangent exactly where `0 < x`, and exact positive zero
    /// otherwise (including both zeros and NaN).
    ReluAdjoint,

    // --- Unary elementwise ---
    Neg,
    Exp,
    Log,
    Sin,
    Sqrt,
    Cos,
    Tan,
    Atan,
    Abs,
    Floor,
    Ceil,
    /// Element-wise round to nearest, ties to even (IEEE-754
    /// roundTiesToEven / banker's rounding). Backends emit `rintf` (or
    /// the f64 / mixed-precision analog) which honors the default
    /// rounding mode; the evaluator uses Rust's `f64::round_ties_even`.
    /// Non-differentiable (piecewise constant); `grad` rejects it with
    /// `AdRejectionReason::PiecewiseConstant`. See
    /// `spec/05-risc-primitives.md` §2.1.
    Round,
    /// Element-wise IEEE-754 reciprocal `1.0 / x`. Primitive so that
    /// `lower_sigmoid` (and any other reciprocal-shaped lowering)
    /// emits a single op rather than the `exp(neg(log(x)))` chain
    /// that would NaN on non-positive inputs. Backends emit
    /// `1.0f / x` (or the f64 / mixed-precision analog).
    Recip,
    UniformLike {
        low: f64,
        high: f64,
        seed: u64,
    },
    Dropout {
        rate: f64,
        seed: u64,
    },

    // --- Reduction ---
    /// `reduce_sum` over `axis`, with the accumulator precision pinned
    /// per `spec/04-type-system.md` §5.7 / §5.7.1.
    ///
    /// The IR node always carries a populated `accumulator: Prim`; the
    /// "no accumulator parameter" Surf/Deep ergonomic shorthand is
    /// resolved to a concrete dtype before lowering. Callers should
    /// prefer the constructor helpers
    /// [`RiscOp::sum_with_accumulator`] / [`RiscOp::sum_default`] over
    /// inline struct construction so the spec-default rule stays in one
    /// place.
    Sum {
        axis: usize,
        /// Precision used for the running sum. Per §5.7.1 this is also
        /// the result precision of the reduction; the verifier checks
        /// `output_type.precision == accumulator`.
        accumulator: Prim,
    },
    /// Count true elements across one or more axes. `axes` stores the
    /// normalized positions in the original input rank exactly once and in
    /// strictly descending order. The result precision is always int64.
    Count {
        axes: Vec<usize>,
    },
    MaxReduce {
        axis: usize,
    },
    MinReduce {
        axis: usize,
    },
    ProdReduce {
        axis: usize,
    },
    /// Strided windowed reduction over the trailing `window_shape.len()`
    /// axes. Per `spec/05-risc-primitives.md` §2.3.1, the leading
    /// `rank - window_shape.len()` axes pass through unchanged, and each
    /// windowed output axis has extent
    /// `floor((input_dim - window_shape[i]) / strides[i]) + 1` under
    /// `Valid` padding (the only padding mode currently shipped).
    ///
    /// The `reducer` field selects which scalar reduction is applied
    /// inside each window; `Max`, `Min`, `Sum`, and `Mean` are the four
    /// shipped variants. `Mean` is implemented as windowed `Sum` divided
    /// by the window volume, inlined into the same loop nest.
    ///
    /// AD policy: the reverse-mode adjoint lowers to a single
    /// [`RiscOp::ReduceWindowGrad`] node carrying the same
    /// `{reducer, window_shape, strides}` triple (see
    /// `chelis_ir::grad`). Adjoints follow `spec/05-risc-primitives.md`
    /// §2.3.1: `Sum`/`Mean` scatter (overlap-add) the upstream gradient
    /// back over each window, and `Max`/`Min` route it to the window
    /// extreme (distributing to all tied positions, matching the
    /// `max_reduce` / `min_reduce` subgradient convention).
    ReduceWindow {
        reducer: ReduceWindowKind,
        window_shape: Vec<usize>,
        strides: Vec<usize>,
    },
    /// Reverse-mode adjoint (vector-Jacobian product) of
    /// [`RiscOp::ReduceWindow`]. Inputs are `[x, g]` where `x` is the
    /// original windowed input (shape `S_in`) and `g` is the upstream
    /// cotangent (shape `S_out`, the forward output shape). The output
    /// is the input cotangent `din` with shape `S_in` (equal to `x`).
    ///
    /// Semantics per `spec/05-risc-primitives.md` §2.3.1, accumulating
    /// over the (overlapping) windows that cover each input position:
    /// - `Sum`:  `din[i] += g[o]` for every `(o, w)` with `o*stride+w = i`.
    /// - `Mean`: as `Sum` with each contribution scaled by `1 /
    ///   window_volume`.
    /// - `Max` / `Min`: for each window `o`, route `g[o]` to every
    ///   position equal to that window's max / min (ties distribute, so
    ///   the rule is the windowed generalization of the `max_reduce` /
    ///   `min_reduce` mask adjoint). `x`'s values are read for `Max` /
    ///   `Min`; for `Sum` / `Mean` only `x`'s shape is used.
    ///
    /// Like the forward op, the IR evaluator, host runtime, and C backend
    /// implement this directly; HIP codegen is deferred (`todo!`).
    ReduceWindowGrad {
        reducer: ReduceWindowKind,
        window_shape: Vec<usize>,
        strides: Vec<usize>,
    },
    /// Index of maximum element along `axis`.
    ///
    /// argmax / argmin logically return integer indices, and per chelis#230
    /// the type-system result is canonically `tensor[..., int64]`
    /// regardless of input precision. Per chelis#233 the host-runtime
    /// adapter (`chelis_compiler_api::runtime::tensor_reduce_host`) tags
    /// the produced `RuntimeTensorValue` with `Prim::Int64` storage, so
    /// `eq` / `to_list` / `tensor_to_scalar` see a matching precision
    /// label.
    ///
    /// The IR-level evaluator (`chelis_ir::eval::reduce_argcmp`) keeps
    /// the precision-erased f64 representation other reductions use; the
    /// storage widening lives in the host-runtime adapter that attaches
    /// the precision tag.
    ///
    /// The C/HIP backend lanes do not yet carry Int64 tensors natively,
    /// so the backend-runtime continues to store integer-valued floats.
    /// The Std wrapper layer is responsible for any casts the backend
    /// needs (see `packages/chelis-std/src/tensor/reduce.ch`).
    Argmax {
        axis: usize,
    },
    Argmin {
        axis: usize,
    },

    // --- Movement ---
    /// chelis#616: a reshape target dim is a [`RtDim`] so a runtime
    /// (`shape()`-derived) extent can reference a rank-0 integer scalar in
    /// `inputs[1..]` (`RtDim::Node`), exactly like a movement bound. `ToEnd`
    /// is illegal here (verify rejects it). The output `TensorType` stays
    /// `Vec<DimInfo>`.
    Reshape {
        new_shape: Vec<RtDim>,
    },
    Permute {
        axes: Vec<usize>,
    },
    Expand {
        axis: usize,
        size: RtDim,
    },
    /// Internal dense one-hot marker used by IR specialization.
    ///
    /// Input is an integer index tensor. Output shape is
    /// `indices.dims + [vocab]`, with f32 zeros and ones. This op must be
    /// consumed or lowered before backend emission.
    OneHot {
        vocab: usize,
    },
    Pad {
        padding: Vec<(RtDim, RtDim)>,
        fill: chelis_types::ScalarValue,
    },
    Shrink {
        bounds: Vec<(RtDim, RtDim)>,
    },
    Stride {
        strides: Vec<RtDim>,
    },

    // --- Shape query ---
    /// Runtime extent of the input tensor along `axis`, produced as a
    /// rank-0 integer scalar (the precision is carried on the node's
    /// `output_type`; the Surf `shape(tensor, axis)` builtin types it as
    /// `int32`, while the hydronnx ONNX translator constructs it as
    /// `int64` per chelis#558).
    ///
    /// This is the DAG-level realization of a `shape(tensor, axis)` read
    /// used as a *value*. It is distinct from a shape read consumed as an
    /// `expand` / `reshape` extent argument, which the lowering folds into
    /// a `DimExpr` on the movement node rather than into a value node.
    /// Before chelis#513 a scalar shape read had no DAG node and fell
    /// through to a bogus `Load { name: "shape" }` placeholder.
    ///
    /// The single input is the tensor whose extent is read; only the
    /// input's shape metadata is observed, never its element values, so
    /// the reverse-mode adjoint contributes a zero cotangent to the input
    /// (differentiable in the trivial constant sense per chelis#558). The
    /// C backend emits `t{input}->shape[axis]`; the HIP lane rejects it
    /// loudly before codegen (`reject_unsupported_hip_ops`) and the Metal
    /// lane rejects it via its emit-time unsupported-op arm (a clean
    /// `Err`, not a panic), because a runtime-symbolic movement/shape read
    /// is out of their admitted scope.
    Shape {
        axis: usize,
    },
    /// A call's shape-only witness. Requirements are tagged int64 literals,
    /// distinct from the actual input axis read by this scalar operation.
    /// The node is created at call entry, before the callee body, and
    /// its enclosing invocation retains required checks through `shape_deps`.
    ///
    /// `claims` carries the NAMED obligations of the same contract, one per
    /// input edge in `inputs[1..]`, each of which is itself an
    /// `ExtentWitness`. A literal requirement compares this axis against a
    /// constant; a named one compares it against another witness's axis. Both
    /// are checked where this node is scheduled, which is call entry, so
    /// `spec/04-type-system.md` §4.7's entry placement holds for both without
    /// a second carrier. See [`ExtentClaim`].
    ExtentWitness {
        site: ExtentWitnessSite,
        parameter: String,
        axis: RtAxis,
        requirements: Vec<chelis_types::ScalarValue>,
        claims: Vec<ExtentClaim>,
    },
    /// Checks an independently computed reshape target (input 0) against
    /// its declaring witnesses or literal requirements (inputs 1..). Every
    /// input and the result is scalar int64. The nonempty `claims` labels
    /// correspond one-to-one to requirement edges, checked in order. Labels
    /// are diagnostic only; edges identify each activation's requirement.
    CheckedReshapeExtent {
        claims: Vec<String>,
        axis: RtAxis,
    },
    /// Refines one tensor axis to one after its exact ExtentWitness has
    /// checked that obligation. Inputs are the original tensor and witness.
    /// Verification ties the witness to that tensor and axis; metadata alone
    /// cannot authorize the refinement. All other dimensions are unchanged.
    CheckedUnitAxis {
        axis: RtAxis,
    },

    // --- Memory ---
    /// A scalar constant carried as a SEALED finalized value (the
    /// chelis#729 fifth storage layer, chelis#856): the payload is
    /// `chelis_types::ScalarValue`, whose construction runs through the
    /// dtype_semantics module, so an un-finalized or dtype-collapsed
    /// numeric constant is unrepresentable in the IR. Integer literals
    /// stay exact at width (no f64 laundering above 2^53). Synthesize
    /// compiler-internal constants via [`RiscOp::synth_const`].
    Const {
        value: chelis_types::ScalarValue,
    },
    /// A multi-element constant tensor stored as SEALED per-dtype
    /// storage (`chelis_types::TensorStorage`, flat row-major).
    /// Replaces the Const+Pad+Add tree for literal `to_tensor` calls
    /// with non-uniform data. Same fifth-layer contract as [`Self::Const`].
    ConstTensor {
        data: chelis_types::TensorStorage,
    },
    Load {
        name: LoadStoreName,
    },
    Store {
        name: LoadStoreName,
    },
    /// Explicit linearity copy marker. Source-level `copy()` and compiler-inserted
    /// fan-out copies both lower to this operation.
    Copy,
    /// Explicit live-range close marker. Backends emit no computation for this op.
    Drop,
    Realize,

    // --- Cast ---
    Cast {
        new_precision: Prim,
    },
    /// The [05-OP-6] named truncating float-to-integer cast
    /// (`cast_trunc`). A separate op rather than a mode flag on `Cast`
    /// so every backend, evaluator, and adjoint site is forced by
    /// exhaustive matching to state its disposition instead of
    /// inheriting the checked default's.
    CastTrunc {
        new_precision: Prim,
    },

    // --- Fusion ---
    /// A sequence of elementwise ops fused into a single kernel.
    FusedElem {
        ops: Vec<FusedStep>,
    },

    // --- Backend specialization ---
    /// Matmul recognized from the Tier-2 `Sum(Mul(Expand(A), Expand(B)))`
    /// lowering after AD has run.
    ///
    /// Carries a pinned `accumulator: Prim` per
    /// `spec/04-type-system.md` §5.7 / §5.7.1. Result precision matches
    /// the operand precision (the wider accumulator is consumed inside
    /// the op and downcast on output, per §5.7.1). Integer matmul
    /// (operand precision in {int8, int16, int32, int64}) is NOT
    /// admitted; use [`RiscOp::matmul_with_accumulator`] /
    /// [`RiscOp::matmul_default`] for the rejection path.
    BlasMatmul {
        batch_dims: Vec<DimExpr>,
        m: DimExpr,
        n: DimExpr,
        k: DimExpr,
        /// Precision used for the inner-product accumulator. Must be
        /// at least as wide as the operand precision and at least as
        /// wide as the documented spec §5.7.1 default.
        accumulator: Prim,
    },

    /// Sparse gather recognized from the Section 3.5 one-hot lowering after
    /// AD has run. Inputs are `values, indices`.
    Gather {
        axis: usize,
    },

    /// Sparse scatter-add used by gather's adjoint. Inputs are
    /// `target, indices, updates`; duplicate indices accumulate.
    ScatterAdd {
        axis: usize,
    },

    /// Sparse replace-scatter (last-write-wins). Inputs are
    /// `target, indices, updates`. Duplicate indices do not accumulate;
    /// the last write in deterministic-order wins. Per
    /// `spec/05-risc-primitives.md` §3.5 the deterministic order is
    /// updates-tensor row-major (C order) flat iteration: the write at
    /// `target[..., indices[i], ...] = updates[i, ...]` occurs in
    /// ascending flat index over `updates`, so the maximum flat-index
    /// write to any target cell is the final value.
    ///
    /// AD policy: `no_grad`. Reverse-mode AD over `Scatter` is
    /// structurally rejected via `AdError::NotSupported { op:
    /// "scatter_replace", reason:
    /// AdRejectionReason::NonDeterministicAtDuplicateIndices }`. Wrap
    /// in a stop-gradient or restructure the program to use
    /// `ScatterAdd` (whose adjoint is well-defined as `Gather`).
    Scatter {
        axis: usize,
    },

    /// Element-wise replace-scatter with ONNX `ScatterElements`
    /// semantics (`spec/05-risc-primitives.md` §3.5.1). Inputs are
    /// `data, indices, updates` where `data`, `indices`, and `updates`
    /// share a rank, `indices.shape == updates.shape`, and
    /// `output.shape == data.shape`. For each coordinate `c` over
    /// `indices`, `updates[c]` is written to `output` at `c` with its
    /// `axis` component replaced by `indices[c]`. This differs from
    /// `Scatter` (hyperplane semantics: `updates` matches
    /// `data.dims[..axis] ++ indices.dims ++ data.dims[axis+1..]`).
    ///
    /// Duplicate-index semantics and determinism match `Scatter`:
    /// last-write-wins in updates-tensor row-major flat order.
    ///
    /// AD policy: `no_grad`, identical to `Scatter` — rejected via
    /// `AdError::NotSupported { op: "scatter_elements", reason:
    /// AdRejectionReason::NonDeterministicAtDuplicateIndices }`.
    ScatterElements {
        axis: usize,
    },
}

/// Canonical semantic identities discovered from the RISC IR before Phase 4C
/// table population. These identities contain no target support status. They
/// exist so chelis#1294 can prove operation-atom closure from an exhaustive
/// typed source instead of a Python allowlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiscAtomIdentity {
    ReluAdjoint,
    Relu,
    ExtremaAdjoint,
    Count,
    MinElem,
    Sub,
    Add,
    Mul,
    Div,
    FloorDiv,
    TruncDiv,
    Mod,
    CmpLt,
    MaxElem,
    Neg,
    Exp,
    Log,
    Sin,
    Sqrt,
    Cos,
    Tan,
    Atan,
    Abs,
    Floor,
    Ceil,
    Round,
    Recip,
    UniformLike,
    Dropout,
    Sum,
    MaxReduce,
    MinReduce,
    ProdReduce,
    ReduceWindowMax,
    ReduceWindowMin,
    ReduceWindowSum,
    ReduceWindowMean,
    ReduceWindowGrad,
    ArgmaxReduce,
    ArgminReduce,
    Reshape,
    Permute,
    Expand,
    Pad,
    Shrink,
    Stride,
    Shape,
    Cast,
    CastTrunc,
    Matmul,
    Gather,
    Scatter,
    ScatterReplace,
    ScatterElements,
}

impl RiscAtomIdentity {
    pub const ALL: &[Self] = &[
        Self::ReluAdjoint,
        Self::Relu,
        Self::ExtremaAdjoint,
        Self::Count,
        Self::MinElem,
        Self::Sub,
        Self::Add,
        Self::Mul,
        Self::Div,
        Self::FloorDiv,
        Self::TruncDiv,
        Self::Mod,
        Self::CmpLt,
        Self::MaxElem,
        Self::Neg,
        Self::Exp,
        Self::Log,
        Self::Sin,
        Self::Sqrt,
        Self::Cos,
        Self::Tan,
        Self::Atan,
        Self::Abs,
        Self::Floor,
        Self::Ceil,
        Self::Round,
        Self::Recip,
        Self::UniformLike,
        Self::Dropout,
        Self::Sum,
        Self::MaxReduce,
        Self::MinReduce,
        Self::ProdReduce,
        Self::ReduceWindowMax,
        Self::ReduceWindowMin,
        Self::ReduceWindowSum,
        Self::ReduceWindowMean,
        Self::ReduceWindowGrad,
        Self::ArgmaxReduce,
        Self::ArgminReduce,
        Self::Reshape,
        Self::Permute,
        Self::Expand,
        Self::Pad,
        Self::Shrink,
        Self::Stride,
        Self::Shape,
        Self::Cast,
        Self::CastTrunc,
        Self::Matmul,
        Self::Gather,
        Self::Scatter,
        Self::ScatterReplace,
        Self::ScatterElements,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReluAdjoint => "ReluAdjoint",
            Self::Relu => "relu",
            Self::ExtremaAdjoint => "ExtremaAdjoint",
            Self::Count => "count",
            Self::MinElem => "min_elem",
            Self::Sub => "sub",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::FloorDiv => "floor_div",
            Self::TruncDiv => "trunc_div",
            Self::Mod => "mod",
            Self::CmpLt => "cmplt",
            Self::MaxElem => "max_elem",
            Self::Neg => "neg",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Sqrt => "sqrt",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Atan => "atan",
            Self::Abs => "abs",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Recip => "recip",
            Self::UniformLike => "uniform_like",
            Self::Dropout => "dropout",
            Self::Sum => "sum",
            Self::MaxReduce => "max_reduce",
            Self::MinReduce => "min_reduce",
            Self::ProdReduce => "prod_reduce",
            Self::ReduceWindowMax => "reduce_window_max",
            Self::ReduceWindowMin => "reduce_window_min",
            Self::ReduceWindowSum => "reduce_window_sum",
            Self::ReduceWindowMean => "reduce_window_mean",
            Self::ReduceWindowGrad => "ReduceWindowGrad",
            Self::ArgmaxReduce => "argmax_reduce",
            Self::ArgminReduce => "argmin_reduce",
            Self::Reshape => "reshape",
            Self::Permute => "permute",
            Self::Expand => "expand",
            Self::Pad => "pad",
            Self::Shrink => "shrink",
            Self::Stride => "stride",
            Self::Shape => "shape",
            Self::Cast => "cast",
            Self::CastTrunc => "cast_trunc",
            Self::Matmul => "matmul",
            Self::Gather => "gather",
            Self::Scatter => "scatter",
            Self::ScatterReplace => "scatter_replace",
            Self::ScatterElements => "scatter_elements",
        }
    }
}

/// Exact pre-4C semantic disposition of a RISC variant. `Structural` is
/// reserved for compiler/lifetime representation nodes that are not Table-A
/// operation identities; it is explicit in the exhaustive match and cannot be
/// inherited by a future variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiscAtomDisposition {
    Semantic(RiscAtomIdentity),
    Structural,
}

impl RiscOp {
    pub const fn atom_disposition(&self) -> RiscAtomDisposition {
        use RiscAtomDisposition::{Semantic, Structural};
        use RiscAtomIdentity as Id;

        match self {
            Self::ReluAdjoint { .. } => Semantic(Id::ReluAdjoint),
            Self::Relu => Semantic(Id::Relu),
            Self::ExtremaAdjoint { .. } => Semantic(Id::ExtremaAdjoint),
            Self::Count { .. } => Semantic(Id::Count),
            Self::MinElem => Semantic(Id::MinElem),
            Self::Sub => Semantic(Id::Sub),
            Self::Add => Semantic(Id::Add),
            Self::Mul => Semantic(Id::Mul),
            Self::Div => Semantic(Id::Div),
            Self::FloorDiv => Semantic(Id::FloorDiv),
            Self::TruncDiv => Semantic(Id::TruncDiv),
            Self::Mod => Semantic(Id::Mod),
            Self::CmpLt => Semantic(Id::CmpLt),
            Self::MaxElem => Semantic(Id::MaxElem),
            Self::Neg => Semantic(Id::Neg),
            Self::Exp => Semantic(Id::Exp),
            Self::Log => Semantic(Id::Log),
            Self::Sin => Semantic(Id::Sin),
            Self::Sqrt => Semantic(Id::Sqrt),
            Self::Cos => Semantic(Id::Cos),
            Self::Tan => Semantic(Id::Tan),
            Self::Atan => Semantic(Id::Atan),
            Self::Abs => Semantic(Id::Abs),
            Self::Floor => Semantic(Id::Floor),
            Self::Ceil => Semantic(Id::Ceil),
            Self::Round => Semantic(Id::Round),
            Self::Recip => Semantic(Id::Recip),
            Self::UniformLike { .. } => Semantic(Id::UniformLike),
            Self::Dropout { .. } => Semantic(Id::Dropout),
            Self::Sum { .. } => Semantic(Id::Sum),
            Self::MaxReduce { .. } => Semantic(Id::MaxReduce),
            Self::MinReduce { .. } => Semantic(Id::MinReduce),
            Self::ProdReduce { .. } => Semantic(Id::ProdReduce),
            Self::ReduceWindow { reducer, .. } => Semantic(match reducer {
                ReduceWindowKind::Max => Id::ReduceWindowMax,
                ReduceWindowKind::Min => Id::ReduceWindowMin,
                ReduceWindowKind::Sum => Id::ReduceWindowSum,
                ReduceWindowKind::Mean => Id::ReduceWindowMean,
            }),
            Self::ReduceWindowGrad { .. } => Semantic(Id::ReduceWindowGrad),
            Self::Argmax { .. } => Semantic(Id::ArgmaxReduce),
            Self::Argmin { .. } => Semantic(Id::ArgminReduce),
            Self::Reshape { .. } => Semantic(Id::Reshape),
            Self::Permute { .. } => Semantic(Id::Permute),
            Self::Expand { .. } => Semantic(Id::Expand),
            Self::Pad { .. } => Semantic(Id::Pad),
            Self::Shrink { .. } => Semantic(Id::Shrink),
            Self::Stride { .. } => Semantic(Id::Stride),
            Self::Shape { .. } => Semantic(Id::Shape),
            Self::Cast { .. } => Semantic(Id::Cast),
            Self::CastTrunc { .. } => Semantic(Id::CastTrunc),
            Self::BlasMatmul { .. } => Semantic(Id::Matmul),
            Self::Gather { .. } => Semantic(Id::Gather),
            Self::ScatterAdd { .. } => Semantic(Id::Scatter),
            Self::Scatter { .. } => Semantic(Id::ScatterReplace),
            Self::ScatterElements { .. } => Semantic(Id::ScatterElements),
            // Extent witnesses and checks carry the compiler's operation
            // preconditions under [04-NUM-9], not callable Table-A operations.
            // Its tagged requirements and shape-only dependency are checked
            // by the IR verifier and the runtime-extent oracle.
            Self::ExtentWitness { .. }
            | Self::CheckedReshapeExtent { .. }
            | Self::CheckedUnitAxis { .. }
            | Self::OneHot { .. }
            | Self::Const { .. }
            | Self::ConstTensor { .. }
            | Self::Load { .. }
            | Self::Store { .. }
            | Self::Copy
            | Self::Drop
            | Self::Realize
            | Self::FusedElem { .. } => Structural,
        }
    }

    /// Construct a pad from an already-finalized fill value. The dtype tag
    /// travels with the value, so a fill/output mismatch is verifier-visible.
    pub fn pad(padding: Vec<(RtDim, RtDim)>, fill: chelis_types::ScalarValue) -> Self {
        Self::Pad { padding, fill }
    }

    /// Compiler-synthesized zero pad at the output dtype.
    pub fn zero_pad(prim: Prim, padding: Vec<(RtDim, RtDim)>) -> Self {
        let fill = chelis_types::scalar_from_i64("pad", prim, 0).unwrap_or_else(|trap| {
            panic!(
                "internal: zero pad fill does not finalize at {}: {trap}",
                prim.name()
            )
        });
        Self::Pad { padding, fill }
    }

    /// Compiler-SYNTHESIZED scalar constant at `prim` (structural
    /// zeros/ones, adjoint seeds, mask fills; the chelis#729 fifth
    /// storage layer). The wide value finalizes through the sealed
    /// module; synthesis sites only produce integral in-range values,
    /// so a trap here is an internal invariant violation and panics
    /// with the offending value. USER-derived values (literals, fold
    /// products) must not use this: literals finalize in `lower_lit`
    /// with a lowering diagnostic, and folds decline on trap per the
    /// section C2 fold rule.
    pub fn synth_const(prim: Prim, wide: f64) -> RiscOp {
        let value = chelis_types::scalar_from_f64("const", prim, wide).unwrap_or_else(|trap| {
            panic!(
                "internal: synthesized constant {wide} does not finalize at \
                 {}: {trap} (compiler-synthesized constants must be \
                 integral and in range by construction)",
                prim.name()
            )
        });
        RiscOp::Const { value }
    }

    /// Bulk sibling of [`RiscOp::synth_const`] for synthesized constant
    /// tensors; same contract, wide f64 element images only.
    pub fn synth_const_tensor(prim: Prim, wide: Vec<f64>) -> RiscOp {
        let data =
            chelis_types::finalize_tensor("const", prim, chelis_types::RawTensor::Float(wide))
                .unwrap_or_else(|trap| {
                    panic!(
                        "internal: synthesized constant tensor does not finalize at \
                     {}: {trap} (compiler-synthesized constants must be \
                     integral and in range by construction)",
                        prim.name()
                    )
                });
        RiscOp::ConstTensor { data }
    }

    /// Resolve the spec §5.7.1 default reduce-sum accumulator for the
    /// given operand precision. Thin re-export of
    /// [`Prim::default_reduce_sum_accumulator`] so call sites inside
    /// `chelis-ir` and downstream backends can stay `RiscOp::*`
    /// -namespaced. The canonical table lives on `Prim` so the type
    /// checker (`chelis-types`) can resolve the same rule without a
    /// backward dependency on `chelis-ir`. See the chelis-types
    /// method for the per-row mapping.
    pub fn default_reduce_sum_accumulator(operand: Prim) -> Result<Prim, String> {
        operand.default_reduce_sum_accumulator()
    }

    /// Resolve the spec §5.7.1 default matmul accumulator for the given
    /// operand precision. Per the active dtype set:
    ///
    /// - bf16 / f16 → f32
    /// - f32        → f32
    /// - f64        → f64
    ///
    /// Integer operands are NOT admitted per §5.7.2; this returns
    /// `Err`. Use [`RiscOp::sum_with_accumulator`] over an explicit
    /// `expand` + `mul` lowering for integer inner products.
    pub fn default_matmul_accumulator(operand: Prim) -> Result<Prim, String> {
        Ok(match operand {
            Prim::Bf16 | Prim::F16 => Prim::F32,
            Prim::F32 => Prim::F32,
            Prim::F64 => Prim::F64,
            Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64 => {
                return Err(format!(
                    "matmul is not admitted for integer operand dtype `{}` per \
                     spec/04-type-system.md §5.7.2; use reduce_sum over an \
                     explicit expand+mul lowering for integer inner products",
                    operand.name()
                ));
            }
            Prim::F8e4m3 => {
                return Err("matmul: operand dtype `f8e4m3` is deferred per \
                     spec/04-type-system.md §1.1.1 and is not part of the \
                     active numeric primitive set"
                    .to_string());
            }
            Prim::Bool | Prim::String => {
                return Err(format!(
                    "matmul is not defined for operand dtype `{}`",
                    operand.name()
                ));
            }
        })
    }

    /// Resolve the spec §5.7.1 user-facing result precision of
    /// `reduce_sum` for the given operand precision. Thin re-export
    /// of [`Prim::default_reduce_sum_result_precision`] so call sites
    /// inside `chelis-ir` and downstream backends can stay
    /// `RiscOp::*`-namespaced. See the chelis-types method for the
    /// per-row table and the bf16/f16 downcast rationale.
    pub fn default_reduce_sum_result_precision(operand: Prim) -> Result<Prim, String> {
        operand.default_reduce_sum_result_precision()
    }

    /// Construct a `Sum` op with an explicit `accumulator` precision.
    /// Verifies the spec §5.7.1 narrowness rule: `accumulator` must be
    /// at least as wide as the documented default for the operand
    /// precision.
    pub fn sum_with_accumulator(
        axis: usize,
        operand_prim: Prim,
        accumulator: Prim,
    ) -> Result<Self, String> {
        let default = Self::default_reduce_sum_accumulator(operand_prim)?;
        if !accumulator_at_least_as_wide(operand_prim, accumulator, default) {
            return Err(format!(
                "reduce_sum: explicit accumulator `{}` is narrower than the \
                 spec/04-type-system.md §5.7.1 default `{}` for operand precision \
                 `{}`; accumulator must be at least the default width (omit the \
                 parameter to accept the default)",
                accumulator.name(),
                default.name(),
                operand_prim.name(),
            ));
        }
        Ok(RiscOp::Sum { axis, accumulator })
    }

    /// Construct a `Sum` op with the spec-default accumulator for the
    /// operand precision. This is the canonical "no explicit
    /// accumulator parameter" lowering path per §5.7.1.
    pub fn sum_default(axis: usize, operand_prim: Prim) -> Result<Self, String> {
        let accumulator = Self::default_reduce_sum_accumulator(operand_prim)?;
        Ok(RiscOp::Sum { axis, accumulator })
    }

    /// Construct a `BlasMatmul` op with an explicit `accumulator`
    /// precision. Verifies the spec §5.7.1 narrowness rule and
    /// rejects integer operands per §5.7.2.
    pub fn matmul_with_accumulator(
        batch_dims: Vec<DimExpr>,
        m: DimExpr,
        n: DimExpr,
        k: DimExpr,
        operand_prim: Prim,
        accumulator: Prim,
    ) -> Result<Self, String> {
        let default = Self::default_matmul_accumulator(operand_prim)?;
        if !accumulator_at_least_as_wide(operand_prim, accumulator, default) {
            return Err(format!(
                "matmul: explicit accumulator `{}` is narrower than the \
                 spec/04-type-system.md §5.7.1 default `{}` for operand precision \
                 `{}`; accumulator must be at least the default width",
                accumulator.name(),
                default.name(),
                operand_prim.name(),
            ));
        }
        Ok(RiscOp::BlasMatmul {
            batch_dims,
            m,
            n,
            k,
            accumulator,
        })
    }

    /// Construct a `BlasMatmul` with the spec-default accumulator for
    /// the operand precision. Rejects integer operands per §5.7.2.
    pub fn matmul_default(
        batch_dims: Vec<DimExpr>,
        m: DimExpr,
        n: DimExpr,
        k: DimExpr,
        operand_prim: Prim,
    ) -> Result<Self, String> {
        let accumulator = Self::default_matmul_accumulator(operand_prim)?;
        Ok(RiscOp::BlasMatmul {
            batch_dims,
            m,
            n,
            k,
            accumulator,
        })
    }

    /// Whether this op is part of the pinned verifier / Beacon target
    /// subset (master plan WI-2, `spec/design/verification_stack_master_plan.md`
    /// §4.1; transformer corpus in `spec/design/beacon_plan.md` §3.1).
    ///
    /// Beacon discharges a goal by pushing a sound numeric envelope
    /// *forward* through the RISC DAG using abstract domains in the
    /// CROWN lineage (interval, zonotope, linear relaxation). An op is
    /// "verifier-targetable" when Beacon has — or can in-house — a sound
    /// abstract transformer for it that maps an input envelope to an
    /// output envelope. Shape-movement and deterministic structural ops
    /// qualify because the envelope passes through them unchanged (up to
    /// reindexing); numeric ops qualify when a sound relaxation exists
    /// or is in-house buildable per the corpus.
    ///
    /// This is a deliberately exhaustive, wildcard-free match: adding a
    /// new `RiscOp` is a hard compile error here, forcing an explicit
    /// in-or-out classification rather than letting a new op silently
    /// inherit a default. The companion exhaustiveness test
    /// (`every_risc_op_is_classified_for_verifier_subset`) guards the
    /// same invariant at runtime over a constructed sample of every
    /// variant.
    ///
    /// The classification is conservative: an op is excluded unless it
    /// has a clear sound transformer story. Excluding an op is not a
    /// claim that it can never be targeted; it records that, today, the
    /// pinned target surface does not include it, so a producer pinning
    /// to this surface must not assume Beacon bounds it.
    pub fn is_verifier_targetable(&self) -> bool {
        match self {
            // --- Elementwise arithmetic and comparison ---
            // `Add`, `Mul`, `Div` (with a denominator-excludes-zero
            // precondition), and `CmpLt` (the branch predicate that
            // drives branch-and-bound on piecewise definitions such as
            // the `erf64` sign/small-x folds) all have sound interval /
            // linear-relaxation transformers (beacon_plan.md §3.1, §3.3).
            RiscOp::Add
            | RiscOp::Sub
            | RiscOp::Mul
            | RiscOp::Div
            | RiscOp::CmpLt
            | RiscOp::MaxElem
            | RiscOp::MinElem => true,

            // --- Unary elementwise math ---
            // `Exp`, `Log`, `Sqrt` are direct ports of the auto_LiRPA
            // relaxation corpus; `Recip` needs the reciprocal transformer
            // with the denominator-excludes-zero precondition; the
            // remaining transcendental and rounding ops are in-house
            // transformers, not blockers (beacon_plan.md §3.1).
            RiscOp::Neg
            | RiscOp::Exp
            | RiscOp::Log
            | RiscOp::Sin
            | RiscOp::Sqrt
            | RiscOp::Cos
            | RiscOp::Tan
            | RiscOp::Atan
            | RiscOp::Abs
            | RiscOp::Floor
            | RiscOp::Ceil
            | RiscOp::Round
            | RiscOp::Recip => true,

            // --- Reductions ---
            // Sum / max / min / prod reductions and windowed reductions
            // are folds of targetable elementwise transformers; the
            // forward envelope propagates through them.
            RiscOp::Sum { .. }
            | RiscOp::MaxReduce { .. }
            | RiscOp::MinReduce { .. }
            | RiscOp::ProdReduce { .. }
            | RiscOp::ReduceWindow { .. } => true,

            // --- Movement / structural ---
            // Pure reindexing: the numeric envelope passes through
            // unchanged, only the shape map changes.
            RiscOp::Reshape { .. }
            | RiscOp::Permute { .. }
            | RiscOp::Expand { .. }
            | RiscOp::Pad { .. }
            | RiscOp::Shrink { .. }
            | RiscOp::Stride { .. } => true,

            // --- Memory / linear-algebra value nodes ---
            // `Const` is an exact (degenerate) envelope; `Load` is the
            // input box itself; `BlasMatmul` is a sum-of-products fold of
            // targetable transformers (the pricer and NN graphs both hit
            // it).
            RiscOp::Const { .. }
            | RiscOp::ConstTensor { .. }
            | RiscOp::Load { .. }
            | RiscOp::BlasMatmul { .. } => true,

            // --- Cast ---
            // Real-valued first: identity on the real envelope. The
            // roundoff-aware cast transformer is a documented later layer
            // (beacon_plan.md §6); until it lands a Cast discharge stays
            // in the real-valued qualifier, but the op is on the target
            // surface today.
            RiscOp::Cast { .. } => true,

            // --- NOT verifier-targetable (today) ---
            // Stochastic ops have no deterministic value to bound.
            RiscOp::UniformLike { .. } | RiscOp::Dropout { .. } => false,

            // Argmax/argmin return discrete indices, not a numeric
            // envelope over the reals; outside the forward-bound story.
            RiscOp::Argmax { .. } | RiscOp::Argmin { .. } | RiscOp::Count { .. } => false,

            // Floor/truncating integer division (chelis#178) are
            // piecewise-constant, non-differentiable rounding ops; like
            // `Floor`/`Ceil`/`Round` they have a step-function envelope,
            // but the integer-quotient semantics are not part of the
            // pinned real-valued forward-bound surface today.
            RiscOp::FloorDiv | RiscOp::TruncDiv | RiscOp::Mod => false,

            // [05-OP-6] `cast_trunc` is the same shape as the integer
            // quotients above: piecewise constant with an integer output,
            // so it has no real-valued envelope to bound. The CHECKED
            // `cast` stays targetable because its float-to-float leg is
            // real-valued and its integer leg only admits exact values.
            RiscOp::CastTrunc { .. } => false,

            // `OneHot` produces a discrete 0/1 indicator from an integer
            // index; it is an internal lowering marker (dag.rs) consumed
            // before backend emission and is not part of the numeric
            // forward-bound surface.
            RiscOp::OneHot { .. } => false,

            // `Shape` reads a runtime axis extent as a discrete integer
            // scalar derived from tensor metadata, not a bound over the
            // input's real-valued data (its output is constant w.r.t. the
            // element values). Like the arg-reductions it is outside the
            // real-valued forward-bound story (chelis#513 / chelis#558).
            RiscOp::Shape { .. }
            | RiscOp::ExtentWitness { .. }
            | RiscOp::CheckedReshapeExtent { .. }
            | RiscOp::CheckedUnitAxis { .. } => false,

            // Sparse gather/scatter index data movement; no real-valued
            // transformer is pinned, and `Scatter` / `ScatterElements`
            // are `no_grad`.
            RiscOp::Gather { .. }
            | RiscOp::ScatterAdd { .. }
            | RiscOp::Scatter { .. }
            | RiscOp::ScatterElements { .. } => false,

            // Linearity / lifecycle markers carry no numeric semantics
            // (backends emit nothing for `Drop`); they are transparent to
            // bound propagation but are not themselves targetable ops.
            RiscOp::Copy | RiscOp::Drop | RiscOp::Realize | RiscOp::Store { .. } => false,

            // AD-only adjoint of a windowed reduction; the gradient-graph
            // path is a later Beacon item (WI-B8 verified Greeks), not the
            // pinned forward surface today.
            RiscOp::ReduceWindowGrad { .. } => false,

            // Internal reverse-mode selection node; Beacon targets the
            // forward extrema identity rather than its generated adjoint.
            RiscOp::ExtremaAdjoint { .. } => false,

            // [05-OP-43] identity and its AD-only selector stay explicit so
            // the zero-at-zero convention cannot collapse to MaxElem's tie
            // rule. Beacon has not yet registered dedicated transformers.
            RiscOp::Relu | RiscOp::ReluAdjoint => false,

            // `FusedElem` is a backend specialization that bundles
            // elementwise steps into one kernel; Beacon targets the
            // pre-fusion elementwise ops, not the fused marker, so it is
            // off the pinned surface.
            RiscOp::FusedElem { .. } => false,
        }
    }
}

/// Spec §5.7.1 narrowness rule: `accumulator` must be (a) at least as
/// wide as the operand precision and (b) at least as wide as the
/// documented default for that operand precision. Width is measured by
/// `prim_width_rank` (a per-Prim ordering that reflects bit width and
/// integer-vs-float lane). Exposed for the verifier (`crate::verify`)
/// so the BlasMatmul rule can be checked without duplicating the
/// width-rank table.
pub(crate) fn accumulator_at_least_as_wide(
    operand: Prim,
    accumulator: Prim,
    default: Prim,
) -> bool {
    let op_lane = prim_lane(operand);
    let acc_lane = prim_lane(accumulator);
    // Don't allow crossing lanes (integer accumulator on float operand
    // or vice versa); mixing lanes is not in the spec table.
    if op_lane != acc_lane {
        return false;
    }
    let acc_w = prim_width_rank(accumulator);
    let op_w = prim_width_rank(operand);
    let def_w = prim_width_rank(default);
    acc_w >= op_w && acc_w >= def_w
}

/// "Lane" of a numeric Prim — float vs integer. Used by
/// [`accumulator_at_least_as_wide`] to keep accumulator selection inside
/// the operand's lane.
fn prim_lane(p: Prim) -> u8 {
    if p.is_float() {
        1
    } else if p.is_integer() {
        2
    } else {
        0
    }
}

/// Width ordering for the active dtype set. Larger is wider. Within the
/// float lane: f16 = bf16 < f32 < f64. Within the integer lane:
/// int8 < int16 < int32 < int64. Bool is 0; non-numeric returns 0.
///
/// E2 (WS-A0 RT-1 fixup, sibling sweep): `f8e4m3` is deferred per
/// `spec/04-type-system.md` §1.1.1 and is rejected upstream by
/// `default_reduce_sum_accumulator` and `default_matmul_accumulator`.
/// If a caller ever asks for the width rank of f8e4m3 the upstream
/// rejection has a hole — panic rather than silently returning 0
/// (which would entrench f8e4m3 as "narrowest-float" in width
/// comparisons and let the deferred dtype propagate downstream).
fn prim_width_rank(p: Prim) -> u32 {
    match p {
        Prim::Bool => 0,
        Prim::Int8 => 1,
        Prim::Int16 => 2,
        Prim::Int32 => 4,
        Prim::Int64 => 8,
        Prim::F16 | Prim::Bf16 => 2,
        Prim::F32 => 4,
        Prim::F64 => 8,
        Prim::F8e4m3 => panic!(
            "f8e4m3 is deferred per spec/04-type-system.md §1.1.1 and \
             should have been rejected upstream"
        ),
        Prim::String => 0,
    }
}

/// A single node in the DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    pub id: NodeId,
    pub op: RiscOp,
    pub inputs: Vec<NodeId>,
    pub output_type: TensorType,
    pub reusable_input: Option<NodeId>,
    /// Canonical span ID, populated by lowering from the Deep `Expr`'s
    /// `meta["span"]` value. Threaded through later passes per the rules
    /// in `spec/design/chelis_span_survival.md` §2.3.
    ///
    /// `None` is the normal case for hand-written Chelis or for nodes
    /// synthesized in places where the source-region rule does not apply.
    /// Non-empty synthesized markers (e.g. `__synthesized_grad__`) are
    /// emitted by S3 passes; S2 only populates real source spans.
    ///
    /// `#[serde(default)]` keeps existing serialized DAGs (which lack
    /// this field) deserializable for JSON/YAML callers; bincode is a
    /// positional format that always emits both new fields, so its
    /// on-disk shape is the new shape — old caches will fail to decode
    /// and be regenerated.
    #[serde(default)]
    pub span_id: Option<String>,
    /// Additional spans accumulated when N→1 merge passes (Fusion, CSE,
    /// constant fold, lowering's def-collapses-to-body case) collapse
    /// multiple source nodes into a single result node.
    ///
    /// Backend codegen (S4) will emit one `// span:` line per
    /// `span_id ∪ merged_spans` so the audit invariant holds: every span
    /// ID present on any input Deep node appears on at least one IR node.
    #[serde(default)]
    pub merged_spans: Vec<String>,
    /// chelis#384/#397: nodes this node depends on ONLY for their shape, not
    /// their data. An `expand(b, axis, shape(x, k))` reads the
    /// broadcast extent from `x`'s runtime shape but does not consume `x`'s
    /// data, so `x` is not in `inputs`. Without recording the dependency,
    /// `x`'s `Load` is dead-code-eliminated and the symbolic dim it declares
    /// loses its only source. DCE keeps the shape source live through this
    /// edge; eval and the backends never read it (op/inputs/output_type are
    /// unchanged), so it does not alter the Expand operator's arity or
    /// codegen.
    #[serde(default)]
    pub shape_deps: Vec<NodeId>,
}

/// The RISC DAG — an append-only, topologically-ordered vector of [`DagNode`]s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dag {
    nodes: Vec<DagNode>,
    roots: Vec<NodeId>,
}

impl Dag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a new node and return its [`NodeId`].
    ///
    /// `span_id` is REQUIRED (not defaulted) so the type system enforces
    /// the greppable invariant from `spec/design/chelis_span_survival.md`
    /// S2: zero `add_node(` callsites elide an explicit span argument.
    /// Pass `None` for nodes synthesized by passes that don't have a
    /// natural source region in S2 — S3 will populate spans on those
    /// per pass-specific rules.
    pub fn add_node(
        &mut self,
        op: RiscOp,
        inputs: Vec<NodeId>,
        output_type: TensorType,
        span_id: Option<String>,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(DagNode {
            id,
            op,
            inputs,
            output_type,
            reusable_input: None,
            span_id,
            merged_spans: Vec::new(),
            shape_deps: Vec::new(),
        });
        id
    }

    pub fn set_reusable_input(&mut self, id: NodeId, input: NodeId) {
        if let Some(node) = self.nodes.get_mut(id.0) {
            node.reusable_input = Some(input);
        }
    }

    /// chelis#384/#397: record that `id` depends on `dep` only for its shape
    /// (an `expand(..., shape(x, ...))` extent source). DCE keeps `dep`
    /// live through this edge; eval/backends ignore it. See
    /// [`DagNode::shape_deps`].
    pub fn add_shape_dep(&mut self, id: NodeId, dep: NodeId) {
        if let Some(node) = self.nodes.get_mut(id.0)
            && !node.shape_deps.contains(&dep)
        {
            node.shape_deps.push(dep);
        }
    }

    /// chelis#384/#397: copy `source_deps` (an old node's `shape_deps`) onto
    /// node `new_id` in this DAG, remapping each through `remap`. Every
    /// DAG-rebuild pass (DCE, copy/drop insertion, BLAS specialization,
    /// fusion, vmap, grad, splice) must call this alongside its
    /// `merged_spans` preservation so a shape-derived `expand` extent is not
    /// silently dropped on the rebuild — losing it reintroduces the
    /// wrong-shape regression. Deps that don't survive the rebuild's remap
    /// are dropped (the consuming node was itself eliminated, so the dep is
    /// moot).
    pub fn preserve_shape_deps(
        &mut self,
        new_id: NodeId,
        source_deps: &[NodeId],
        remap: &UnordMap<NodeId, NodeId>,
    ) {
        if source_deps.is_empty() {
            return;
        }
        let mapped: Vec<NodeId> = source_deps
            .iter()
            .filter_map(|old| remap.get(old).copied())
            .collect();
        if let Some(node) = self.nodes.get_mut(new_id.0) {
            node.shape_deps = mapped;
        }
    }

    pub fn get(&self, id: NodeId) -> Option<&DagNode> {
        self.nodes.get(id.0)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn nodes(&self) -> &[DagNode] {
        &self.nodes
    }

    /// Mutable access to a node by id. Used by passes that need to
    /// stamp post-construction metadata on a node (e.g. DCE preserving
    /// `merged_spans` per `spec/design/chelis_span_survival.md` §2.3).
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut DagNode> {
        self.nodes.get_mut(id.0)
    }

    pub fn add_root(&mut self, id: NodeId) {
        if !self.roots.contains(&id) {
            self.roots.push(id);
        }
    }

    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    pub fn set_roots(&mut self, roots: Vec<NodeId>) {
        self.roots = roots;
    }

    pub fn is_root(&self, id: NodeId) -> bool {
        self.roots.contains(&id)
    }

    /// Return nodes in topological order (they already are, since we only append).
    pub fn topological_order(&self) -> Vec<NodeId> {
        (0..self.nodes.len()).map(NodeId).collect()
    }

    /// Replace the node at `id` in-place (used by optimization passes).
    pub fn replace_node(
        &mut self,
        id: NodeId,
        op: RiscOp,
        inputs: Vec<NodeId>,
        output_type: TensorType,
    ) {
        if let Some(node) = self.nodes.get_mut(id.0) {
            node.op = op;
            node.inputs = inputs;
            node.output_type = output_type;
        }
    }
}

pub fn symbolic_occurrences(dag: &Dag) -> Vec<SymbolicDimOccurrence> {
    let mut occurrences = Vec::new();
    let mut seen_inputs = UnordSet::new();
    let mut named_dims_in_loads: UnordSet<String> = UnordSet::new();

    // First pass: collect Load occurrences. These are the canonical
    // sources for symbolic dim values (the C codegen turns each into
    // `int <dim> = inputs[<slot>]->shape[<axis>]`).
    for node in dag.nodes() {
        let RiscOp::Load { name } = &node.op else {
            continue;
        };
        if !seen_inputs.insert(name.as_str().to_string()) {
            continue;
        }
        for (axis, dim) in node.output_type.dims.iter().enumerate() {
            if let DimInfo::Named(symbol, None) = dim {
                named_dims_in_loads.insert(symbol.clone());
                occurrences.push(SymbolicDimOccurrence::load(symbol, name.as_str(), axis));
            }
        }
    }

    // chelis#616 pass: movement ops whose output axis extent is computed at
    // RUN TIME by the op itself (a node-valued bound produces a fresh extent
    // no Load traces to). Record an `OpDeclared` source so the C emitter
    // declares (or equality-guards) the dim inline at the op and the eval
    // lane skips pre-eval binding for it. A symbol that is also Load-carried
    // keeps the Load as its canonical declaration; the op site then becomes
    // a runtime equality guard rather than a redeclaration.
    let mut op_declared: UnordSet<String> = UnordSet::new();
    for node in dag.nodes() {
        for (symbol, axis) in op_declared_output_axes(dag, node) {
            if !named_dims_in_loads.contains(&symbol) {
                let _ = bind_symbol_from_any_load(
                    dag,
                    &symbol,
                    &mut occurrences,
                    &mut named_dims_in_loads,
                );
            }
            occurrences.push(SymbolicDimOccurrence {
                name: symbol.clone(),
                source: SymbolicDimSource::OpDeclared {
                    node: node.id,
                    axis,
                },
            });
            op_declared.insert(symbol);
        }
    }

    // Dominance guard (chelis#616 soundness): an op-declared symbol's C
    // declaration is emitted at the declaring op, so every node that
    // references the symbol (output dims or op-internal fields) must come
    // AFTER the declarer in emission (= node id) order, or the C references
    // an undeclared identifier. A violation is a producing-pass bug; fail
    // loud rather than emit non-compiling (or worse, shadowed) C.
    for symbol in op_declared.to_sorted() {
        if named_dims_in_loads.contains(symbol) {
            // Load-declared in the prologue; every reference is dominated.
            continue;
        }
        let declarer = occurrences
            .iter()
            .find_map(|occurrence| match &occurrence.source {
                SymbolicDimSource::OpDeclared { node, .. } if occurrence.name == *symbol => {
                    Some(*node)
                }
                _ => None,
            })
            .expect("op_declared symbols always have an OpDeclared occurrence");
        for node in dag.nodes() {
            let references = node
                .output_type
                .dims
                .iter()
                .any(|dim| matches!(dim, DimInfo::Named(name, None) if name == symbol))
                || op_internal_symbolic_dims(&node.op)
                    .iter()
                    .any(|name| name == symbol);
            if references && node.id.0 < declarer.0 {
                panic!(
                    "internal compiler error: symbolic dim `{symbol}` is declared at run time \
                     by node {} but referenced by EARLIER node {} (op {:?}); the C declaration \
                     would not dominate the reference. Fix the producing IR pass.",
                    declarer.0, node.id.0, node.op
                );
            }
        }
    }

    // Bucket 4c sibling sweep: a polymorphic dim may be referenced by
    // a non-Load node (e.g. `Const` synthesised by tier2 lowering or
    // the gradient backward pass) without appearing in any Load's
    // type. Without an entry in the occurrences list the C codegen
    // emits `(int[]){ d36 }` against an undeclared `d36`.
    //
    // For each unbound name we try to find a Load whose own dims
    // reference the same symbol (e.g. via op-internal references like
    // `RiscOp::Reshape::new_shape` or `RiscOp::Expand::size`). If a
    // matching Load is found we register a synthetic occurrence so
    // the codegen can declare the dim from that input. If no matching
    // Load exists, the dim is unrecoverable from inputs alone — that
    // is a bug in the producing pass and we surface it loudly via
    // `panic!` rather than silently emitting C that won't compile.
    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Drop) {
            continue;
        }
        for dim in &node.output_type.dims {
            if let DimInfo::Named(symbol, None) = dim
                && !named_dims_in_loads.contains(symbol)
                && !op_declared.contains(symbol)
                // chelis#616: a raw ANONYMOUS (wildcard) output dim is not a
                // referenceable symbol — nothing renders it, and distinct
                // runtime extents share it. The C backend renames every anon
                // dim to a unique `_anon_dim_{id}_{axis}` BEFORE this pass
                // (so a genuinely sourceless anon dim still fails loud
                // there); the eval lane computes shapes from values and
                // never reads a wildcard by name.
                && !symbol.is_empty()
                && symbol != "*"
            {
                // Hunt for any Load whose own type contains the same
                // unbound dim name. We have to widen the search because
                // a Load with a shape-mismatched annotation wouldn't
                // necessarily appear in the first pass (its dim could
                // be `Lit(_)` while the synthesised node carries the
                // polymorphic name).
                let mut bound = bind_symbol_from_any_load(
                    dag,
                    symbol,
                    &mut occurrences,
                    &mut named_dims_in_loads,
                );
                if !bound
                    && let Some(axis) =
                        node.output_type.dims.iter().position(
                            |dim| matches!(dim, DimInfo::Named(name, None) if name == symbol),
                        )
                    && let Some((input_label, input_axis)) =
                        shape_source_for_axis(dag, node.id, axis)
                {
                    occurrences.push(SymbolicDimOccurrence::load(
                        symbol,
                        &input_label,
                        input_axis,
                    ));
                    named_dims_in_loads.insert(symbol.clone());
                    bound = true;
                }
                if !bound {
                    // chelis#616: a node-valued movement op computes a FRESH
                    // runtime output extent with no Load source, and so does a
                    // runtime-`shape()`-derived `reshape` target (the window
                    // count `m`). The C backend can declare a movement output dim
                    // from its bound scalars (`emit_shrink`/`emit_stride`/
                    // `emit_pad`), but a `reshape` target has no node-valued dim
                    // source threaded through yet, so a full runtime-symbolic
                    // program still reaches an undeclarable dim. Until node-valued
                    // Reshape/Const dims land (the runtime-dim-from-Shape-arith
                    // declaration capability), this stays FAIL-CLOSED and LOUD
                    // rather than emit a silently mis-sized allocation.
                    panic!(
                        "internal compiler error: symbolic dim `{symbol}` is referenced by a \
                         non-Load node (id {}, op {:?}, inputs {:?}, type {:?}) but no Load input \
                         declares it. The C codegen would emit an undeclared identifier; fix the \
                         producing IR pass.",
                        node.id.0, node.op, node.inputs, node.output_type
                    );
                }
            }
        }
    }

    // Bucket 4d sweep (chelis#345): op-internal symbolic references.
    // A node whose output dims are fully concrete can still reference an
    // undeclared symbolic dim through an op-internal field —
    // `Expand::size`, `Reshape::new_shape`, or `BlasMatmul`'s
    // `{batch_dims, m, n, k}` (the BLAS dims are rendered verbatim into
    // C by `emit_dim_expr`, and `bind_symbolic_dims` / the IR evaluator
    // resolve Expand sizes by name). The #345 bisect found exactly this
    // mixed state in a grad helper DAG: `Load: Lit(2)` next to
    // `Expand { size: Sym("dN") }`. Bucket 4c never sees those names
    // because it scans output types only, so sweep the op fields too.
    //
    // No `shape_source_for_axis` fallback here: the op-internal name has
    // no output axis to recover from (BlasMatmul's `k` is the
    // contraction dim and appears in no output type at all). Either a
    // Load declares the name or the producing pass is buggy.
    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Drop) {
            continue;
        }
        for symbol in op_internal_symbolic_dims(&node.op) {
            if named_dims_in_loads.contains(&symbol) || op_declared.contains(&symbol) {
                continue;
            }
            // chelis#616: a raw ANONYMOUS op-internal reference (the
            // `lower_if` mask expansion over a wildcard-typed branch) is
            // resolved by the evaluator from the node's shape-dep value. The
            // C lane cannot render it — but the emitted identifier `*` fails
            // C compilation LOUDLY if such a node ever reaches codegen
            // (guarded `if` programs route through the host lane).
            if symbol.is_empty() || symbol == "*" {
                continue;
            }
            if !bind_symbol_from_any_load(dag, &symbol, &mut occurrences, &mut named_dims_in_loads)
            {
                panic!(
                    "internal compiler error: symbolic dim `{symbol}` is referenced by a \
                     non-Load node (id {}, op {:?}, inputs {:?}, type {:?}) through an \
                     op-internal field but no Load input declares it. The C codegen would emit \
                     an undeclared identifier; fix the producing IR pass.",
                    node.id.0, node.op, node.inputs, node.output_type
                );
            }
        }
    }

    occurrences
}

/// chelis#616: the output axes of `node` whose symbolic dim is sized at RUN
/// TIME by the op itself — a movement op axis with a non-identity bound,
/// which [`shape_source_for_axis`] deliberately refuses to trace to a Load
/// (the extent is fresh, not the input axis's runtime dim), or a `Reshape`
/// axis whose target is a node-valued (`RtDim::Node`) extent. Each returned
/// `(symbol, axis)` pair becomes an [`SymbolicDimSource::OpDeclared`]
/// occurrence: the C emitter declares (or equality-guards) the dim inline at
/// the op and the eval lane resolves it from actual values.
fn op_declared_output_axes(dag: &Dag, node: &DagNode) -> Vec<(String, usize)> {
    // An ANONYMOUS (wildcard) dim name is not a stable symbol: distinct
    // runtime extents share it, so binding/declaring it would falsely unify
    // them. The C backend renames every anon dim to a unique
    // `_anon_dim_{id}_{axis}` before its occurrence pass (making them
    // eligible there); the eval lane computes shapes from values and never
    // needs a wildcard bound.
    fn is_anon(name: &str) -> bool {
        name.is_empty() || name == "*"
    }
    match &node.op {
        RiscOp::Shrink { .. } | RiscOp::Stride { .. } | RiscOp::Pad { .. } => node
            .output_type
            .dims
            .iter()
            .enumerate()
            .filter_map(|(axis, dim)| match dim {
                DimInfo::Named(symbol, None)
                    if !is_anon(symbol) && shape_source_for_axis(dag, node.id, axis).is_none() =>
                {
                    Some((symbol.clone(), axis))
                }
                _ => None,
            })
            .collect(),
        RiscOp::Reshape { new_shape } => node
            .output_type
            .dims
            .iter()
            .enumerate()
            .filter_map(|(axis, dim)| match (dim, new_shape.get(axis)) {
                (DimInfo::Named(symbol, None), Some(RtDim::Node(_))) if !is_anon(symbol) => {
                    Some((symbol.clone(), axis))
                }
                _ => None,
            })
            .collect(),
        // A node-valued Expand size computes the inserted/set axis at run
        // time. InputAxis normally traces to a Load, but when its explicit
        // tensor input is itself a runtime-shaped movement result, the
        // Expand site declares the extent directly from that tensor's
        // metadata instead of inventing an undeclared symbolic name.
        RiscOp::Expand { axis, size } => match (node.output_type.dims.get(*axis), size) {
            (Some(DimInfo::Named(symbol, None)), RtDim::Node(_)) if !is_anon(symbol) => {
                vec![(symbol.clone(), *axis)]
            }
            (Some(DimInfo::Named(symbol, None)), RtDim::InputAxis { .. })
                if !is_anon(symbol) && shape_source_for_axis(dag, node.id, *axis).is_none() =>
            {
                vec![(symbol.clone(), *axis)]
            }
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// chelis#632: axes of `node` whose extent the op itself can DECLARE at
/// run time if the axis carries a real symbol — the same op family and
/// per-axis conditions as [`op_declared_output_axes`], evaluated on
/// ELIGIBILITY rather than on the current symbol. Used by the host
/// lane's helper-root retype (`host::remap_tensor_helper_dim_symbols`):
/// painting a declared-return symbol onto a root axis is sound only when
/// the root op will declare that symbol's value; anywhere else the
/// symbol would reach the `symbolic_occurrences` no-declaring-Load ICE.
pub(crate) fn op_declarable_axes(dag: &Dag, node: &DagNode) -> Vec<usize> {
    match &node.op {
        RiscOp::Shrink { .. } | RiscOp::Stride { .. } | RiscOp::Pad { .. } => {
            (0..node.output_type.dims.len())
                .filter(|axis| shape_source_for_axis(dag, node.id, *axis).is_none())
                .collect()
        }
        RiscOp::Reshape { new_shape } => new_shape
            .iter()
            .enumerate()
            .filter_map(|(axis, target)| matches!(target, RtDim::Node(_)).then_some(axis))
            .collect(),
        RiscOp::Expand {
            axis,
            size: RtDim::Node(_) | RtDim::InputAxis { .. },
        } => vec![*axis],
        RiscOp::Expand { .. } => Vec::new(),
        _ => Vec::new(),
    }
}

/// chelis#616: every symbolic dim name that is op-declared somewhere in the
/// DAG (see [`op_declared_output_axes`]). Used by [`bind_symbolic_dims`] to
/// exempt these names from the pre-eval "missing symbolic dimension binding"
/// error — their values do not exist until the owning op evaluates.
pub fn op_declared_dim_names(dag: &Dag) -> UnordSet<String> {
    dag.nodes()
        .iter()
        .flat_map(|node| op_declared_output_axes(dag, node))
        .map(|(symbol, _)| symbol)
        .collect()
}

/// chelis#616: per-node op-declared runtime dims (see
/// [`op_declared_output_axes`]), for the evaluator's mid-evaluation binding:
/// when the declaring node's value is computed, each `(symbol, axis)` binds
/// the symbol to the value's actual extent on that axis.
pub(crate) fn op_declared_axes_by_node(dag: &Dag) -> UnordMap<NodeId, Vec<(String, usize)>> {
    let mut out = UnordMap::new();
    for node in dag.nodes() {
        let axes = op_declared_output_axes(dag, node);
        if !axes.is_empty() {
            out.insert(node.id, axes);
        }
    }
    out
}

/// chelis#616: add a shape-dep from every node that references an
/// op-declared runtime dim (in its output dims or op-internal fields) to the
/// dim's declaring node. DCE and grad's output pruning honor `shape_deps`,
/// so this keeps the declarer — and, transitively, its bound-scalar chain —
/// alive for consumers that need the extent at run time even when the
/// declarer's VALUE is dead (e.g. a backward `Expand` over a runtime reshape
/// extent whose forward result the gradient never reads).
pub fn record_runtime_dim_shape_deps(dag: &mut Dag) {
    let mut declarers: UnordMap<String, NodeId> = UnordMap::new();
    for node in dag.nodes() {
        for (symbol, _) in op_declared_output_axes(dag, node) {
            declarers.entry(symbol).or_insert(node.id);
        }
    }
    if declarers.is_empty() {
        return;
    }
    let mut deps: Vec<(NodeId, NodeId)> = Vec::new();
    for node in dag.nodes() {
        let mut names: Vec<String> = node
            .output_type
            .dims
            .iter()
            .filter_map(|dim| match dim {
                DimInfo::Named(name, None) => Some(name.clone()),
                _ => None,
            })
            .collect();
        names.extend(op_internal_symbolic_dims(&node.op));
        for name in names {
            if let Some(declarer) = declarers.get(&name)
                && *declarer != node.id
            {
                deps.push((node.id, *declarer));
            }
        }
    }
    for (from, to) in deps {
        dag.add_shape_dep(from, to);
    }
}

/// Hunt for any Load whose type carries `symbol` (bound or unbound) and
/// register a synthetic occurrence pointing at it. Returns whether a
/// declaring Load was found. Shared by the Bucket 4c (output-dim) and
/// Bucket 4d (op-internal) sweeps of [`symbolic_occurrences`].
fn bind_symbol_from_any_load(
    dag: &Dag,
    symbol: &str,
    occurrences: &mut Vec<SymbolicDimOccurrence>,
    named_dims_in_loads: &mut UnordSet<String>,
) -> bool {
    for candidate in dag.nodes() {
        let RiscOp::Load { name: load_name } = &candidate.op else {
            continue;
        };
        for (axis, candidate_dim) in candidate.output_type.dims.iter().enumerate() {
            if let DimInfo::Named(candidate_sym, _) = candidate_dim
                && candidate_sym == symbol
            {
                occurrences.push(SymbolicDimOccurrence::load(
                    symbol,
                    load_name.as_str(),
                    axis,
                ));
                named_dims_in_loads.insert(symbol.to_string());
                return true;
            }
        }
    }
    false
}

/// Symbolic dim names referenced by an op's internal fields rather than
/// its output type: `Reshape::new_shape` and
/// `BlasMatmul::{batch_dims, m, n, k}`.
fn op_internal_symbolic_dims(op: &RiscOp) -> Vec<String> {
    fn collect_dim_expr(expr: &DimExpr, out: &mut Vec<String>) {
        match expr {
            DimExpr::Concrete(_) => {}
            DimExpr::Sym(name) => out.push(name.clone()),
            DimExpr::Mul(lhs, rhs) | DimExpr::Div(lhs, rhs) => {
                collect_dim_expr(lhs, out);
                collect_dim_expr(rhs, out);
            }
        }
    }

    let mut out = Vec::new();
    match op {
        RiscOp::Reshape { new_shape } => {
            for dim in new_shape {
                if let RtDim::Sym(name) = dim {
                    out.push(name.clone());
                }
            }
        }
        RiscOp::BlasMatmul {
            batch_dims,
            m,
            n,
            k,
            ..
        } => {
            for dim in batch_dims {
                collect_dim_expr(dim, &mut out);
            }
            collect_dim_expr(m, &mut out);
            collect_dim_expr(n, &mut out);
            collect_dim_expr(k, &mut out);
        }
        _ => {}
    }
    out
}

/// Every symbolic dimension identity already carried by a DAG.
///
/// Keep output-axis and op-internal carriers behind one enumerator so a
/// producer allocating a fresh dimension cannot accidentally reserve only
/// the visible `TensorType` half of the namespace. The op-internal half is
/// exactly whatever [`op_internal_symbolic_dims`] recognizes, so an op that
/// spells a dimension symbol only inside its own payload still participates
/// even when no node output repeats that name.
pub(crate) fn dimension_identity_names(dag: &Dag) -> UnordSet<String> {
    let mut names = UnordSet::new();
    for node in dag.nodes() {
        names.extend(node.output_type.dims.iter().filter_map(|dim| match dim {
            DimInfo::Named(name, _) => Some(name.clone()),
            DimInfo::Lit(_) => None,
        }));
        names.extend(op_internal_symbolic_dims(&node.op));
    }
    names
}

fn shape_source_for_axis(dag: &Dag, id: NodeId, axis: usize) -> Option<(String, usize)> {
    let node = dag.get(id)?;
    match &node.op {
        RiscOp::Load { name } => Some((name.as_str().to_string(), axis)),
        RiscOp::Add
        | RiscOp::Sub
        | RiscOp::Mul
        | RiscOp::CmpLt
        | RiscOp::MaxElem
        | RiscOp::MinElem
        | RiscOp::ExtremaAdjoint { .. }
        | RiscOp::ReluAdjoint => node
            .inputs
            .iter()
            .find_map(|input| shape_source_for_axis(dag, *input, axis)),
        RiscOp::Neg
        | RiscOp::Exp
        | RiscOp::Log
        | RiscOp::Sin
        | RiscOp::Sqrt
        | RiscOp::Cos
        | RiscOp::Tan
        | RiscOp::Atan
        | RiscOp::Abs
        | RiscOp::Floor
        | RiscOp::Ceil
        | RiscOp::Round
        | RiscOp::Relu
        | RiscOp::UniformLike { .. }
        | RiscOp::Dropout { .. } => shape_source_for_axis(dag, *node.inputs.first()?, axis),
        RiscOp::Copy
        | RiscOp::Drop
        | RiscOp::Realize
        | RiscOp::Cast { .. }
        | RiscOp::CastTrunc { .. } => shape_source_for_axis(dag, *node.inputs.first()?, axis),
        // chelis#384/#397: an Expand INSERTS a new axis (rank+1) or SETS an
        // existing size-1 axis (rank unchanged) at `expand_axis`. The newly
        // inserted/set axis's extent comes from the Expand's `size`, NOT from
        // the operand — recursing into the operand at that output index reads
        // the wrong dim (a size-1 axis in the set case, or an out-of-range /
        // shifted index in the insert case). Map the query axis back to the
        // correct operand axis, and return `None` for the inserted/set axis
        // itself so the symbol is declared by the real shape source (the
        // `shape_deps`-kept Load) or surfaces loudly via the sourceless-symbol
        // diagnostic, rather than binding to a fabricated operand source.
        RiscOp::Expand {
            axis: expand_axis, ..
        } => {
            let operand = *node.inputs.first()?;
            let operand_rank = dag.get(operand)?.output_type.dims.len();
            let inserts = node.output_type.dims.len() == operand_rank + 1;
            if axis == *expand_axis {
                return match &node.op {
                    RiscOp::Expand {
                        size:
                            RtDim::InputAxis {
                                tensor,
                                axis: RtAxis::Lit(source_axis),
                            },
                        ..
                    } => {
                        let source_axis = usize::try_from(*source_axis).ok()?;
                        shape_source_for_axis(dag, *node.inputs.get(*tensor)?, source_axis)
                    }
                    _ => None,
                };
            }
            if inserts && axis > *expand_axis {
                shape_source_for_axis(dag, operand, axis - 1)
            } else {
                shape_source_for_axis(dag, operand, axis)
            }
        }
        // chelis#551/#340: a single-axis reduction REMOVES its `axis`, so an
        // output axis `a` maps back to input axis `a` (for `a` before the
        // reduced axis) or `a + 1` (for `a` at/after it). When a reduction
        // operand is a host-lane `concat` (or grad-backward concat cascade),
        // the operand's non-concat axis is a `*` wildcard that the C backend
        // renamed to a fresh `_anon_dim_*`; the reduction output re-anonymizes
        // it to a DIFFERENT `_anon_dim_*`. Both names denote the same runtime
        // dim (the surviving operand axis), so trace the kept output axis back
        // through the reduction to its declaring Load rather than falling to
        // the sourceless-symbol panic.
        RiscOp::Sum {
            axis: reduce_axis, ..
        }
        | RiscOp::MaxReduce { axis: reduce_axis }
        | RiscOp::MinReduce { axis: reduce_axis }
        | RiscOp::ProdReduce { axis: reduce_axis }
        | RiscOp::Argmax { axis: reduce_axis }
        | RiscOp::Argmin { axis: reduce_axis } => {
            let operand = *node.inputs.first()?;
            let input_axis = if axis < *reduce_axis { axis } else { axis + 1 };
            shape_source_for_axis(dag, operand, input_axis)
        }
        RiscOp::Count { axes } => {
            let operand = *node.inputs.first()?;
            let input_axis = (0..dag.get(operand)?.output_type.dims.len())
                .filter(|candidate| !axes.contains(candidate))
                .nth(axis)?;
            shape_source_for_axis(dag, operand, input_axis)
        }
        RiscOp::Reshape { .. } | RiscOp::Permute { .. } | RiscOp::Store { .. } => {
            shape_source_for_axis(dag, *node.inputs.first()?, axis)
        }
        // chelis#616 (soundness): a movement op passes an axis's runtime dim
        // through ONLY when the op is an identity on that axis — a full-axis
        // shrink sentinel `(0, ToEnd)`, a stride step of literal 1, or a
        // zero pad. Every other bound (node-valued OR non-identity literal)
        // produces a FRESH extent that is NOT the input axis's runtime dim,
        // so tracing it to the input's declaring `Load` would bind the
        // symbolic output dim to the wrong extent (e.g. `stride(x, 2)`'s
        // output extent is `ceil(n/2)`, not `n`). Those axes are op-declared
        // instead (see [`op_declared_output_axes`]): the owning op's emitter
        // declares the exact extent formula at run time.
        RiscOp::Shrink { bounds } => match bounds.get(axis) {
            Some((s, e)) if s.as_lit() == Some(0) && matches!(e, RtDim::ToEnd) => {
                shape_source_for_axis(dag, *node.inputs.first()?, axis)
            }
            _ => None,
        },
        RiscOp::Stride { strides } => match strides.get(axis) {
            Some(b) if b.as_lit() == Some(1) => {
                shape_source_for_axis(dag, *node.inputs.first()?, axis)
            }
            _ => None,
        },
        RiscOp::Pad { padding, .. } => match padding.get(axis) {
            Some((b, a)) if b.as_lit() == Some(0) && a.as_lit() == Some(0) => {
                shape_source_for_axis(dag, *node.inputs.first()?, axis)
            }
            _ => None,
        },
        RiscOp::FusedElem { .. } => node
            .inputs
            .iter()
            .find_map(|input| shape_source_for_axis(dag, *input, axis)),
        _ => None,
    }
}

/// The INTERFACE bindings: one per claim with an external witness, canonical
/// first, in assigned ABI input-slot order.
///
/// This is the declaration and entry-guard set, derived from
/// [`crate::axis_sources::derive_dim_witnesses`] rather than recovered by
/// walking for a `Load` that carries a matching string. Consumers that read
/// only interface witnesses - the C and HIP prologues, which declare
/// `int64_t n = inputs[s]->shape[a];` and guard each further witness at
/// entry, and [`symbolic_params`], which reports what a caller must supply -
/// read this.
///
/// The local half, an extent an operation computes at run time, is still
/// [`symbolic_occurrences`]' to report until Slice B's guard commit places
/// those guards at their introducing operations. Each consumer therefore
/// reads one derivation or the other, never a mixture: two derivations that
/// can disagree is the defect this work removes.
///
/// `spec/04-type-system.md` section 4.7 decides the order: "Whatever rule
/// assigns the slots, the guard order follows the assigned slots, and never a
/// separate traversal by binding name, hash iteration, or node identity."
pub fn symbolic_bindings_interface(dag: &Dag) -> Vec<SymbolicDimBinding> {
    crate::axis_sources::derive_dim_witnesses(dag)
        .into_iter()
        .filter_map(|class| {
            let crate::axis_sources::DimClaim::Name(name) = class.claim else {
                return None;
            };
            // Both spellings of "an input tensor's axis" bind here, through
            // the one predicate that answers it. Accepting only the `Load`'s
            // own axis dropped the DECLARATION for a name whose sole
            // interface witness is a folded `shape(t, k)` read - measured on
            // chelis#631's avgpool program, where `main` declares
            // `int64_t _anon_dim_1_0 = chelis_tensor_shape(inputs[0], 1);`
            // beside the Load's own `_anon_dim_0_1` from the same axis, and
            // the emitted C stopped compiling without it.
            let mut occurrences = class
                .members
                .iter()
                .filter_map(|member| {
                    let (load, axis) = crate::axis_sources::member_load_axis(dag, member)?;
                    let RiscOp::Load { name: label } = &dag.get(load)?.op else {
                        return None;
                    };
                    Some(SymbolicDimOccurrence::load(&name, label.as_str(), axis))
                })
                .collect::<Vec<_>>()
                .into_iter();
            let canonical = occurrences.next()?;
            Some(SymbolicDimBinding {
                name,
                canonical,
                others: occurrences.collect(),
            })
        })
        .collect()
}

pub fn symbolic_bindings(dag: &Dag) -> Vec<SymbolicDimBinding> {
    let mut grouped = std::collections::BTreeMap::<String, Vec<SymbolicDimOccurrence>>::new();
    for occurrence in symbolic_occurrences(dag) {
        grouped
            .entry(occurrence.name.clone())
            .or_default()
            .push(occurrence);
    }

    grouped
        .into_iter()
        .map(|(name, mut occurrences)| {
            // chelis#616: a Load source, when one exists, is always the
            // canonical declaration (the prologue declares it; op-declared
            // sites for the same symbol become runtime equality guards).
            let canonical_index = occurrences
                .iter()
                .position(|occurrence| matches!(occurrence.source, SymbolicDimSource::Load { .. }))
                .unwrap_or(0);
            let canonical = occurrences.remove(canonical_index);
            SymbolicDimBinding {
                name,
                canonical,
                others: occurrences,
            }
        })
        .collect()
}

/// The symbolic dims a caller can (and must) supply — those bound from input
/// shape metadata. chelis#616: op-declared dims are computed at run time by
/// their owning op and are deliberately excluded; they are not parameters.
pub fn symbolic_params(dag: &Dag) -> Vec<String> {
    // Deduplicated by NAME, because C2.4's scope split is about which axes are
    // guarded together and not about how many parameters a caller supplies.
    // One binder spelled once in a signature is one parameter however many
    // scopes the derivation finds it in; without this a merged kernel
    // published `["batch", "batch"]`.
    let mut seen = std::collections::BTreeSet::new();
    symbolic_bindings_interface(dag)
        .into_iter()
        .filter(|binding| matches!(binding.canonical.source, SymbolicDimSource::Load { .. }))
        .map(|binding| binding.name)
        .filter(|name| seen.insert(name.clone()))
        .collect()
}

pub fn bind_symbolic_dims(dag: &Dag, bindings: &UnordMap<String, usize>) -> Result<Dag, String> {
    // chelis#616: an op-declared dim (a node-valued movement output extent)
    // has no pre-eval value — the evaluator computes it from actual bound
    // scalars. Leave it unbound instead of raising the loud missing-binding
    // error; the eval movement arms never read the output type for it. The
    // same applies to a raw ANONYMOUS (wildcard) dim: it is not a
    // referenceable symbol, and the evaluator computes the real extent from
    // values.
    let op_declared = op_declared_dim_names(dag);
    let bind_dim = |dim: &DimInfo| -> Result<DimInfo, String> {
        match dim {
            DimInfo::Lit(size) => Ok(DimInfo::Lit(*size)),
            DimInfo::Named(name, Some(size)) => Ok(DimInfo::Named(name.clone(), Some(*size))),
            DimInfo::Named(name, None) => match bindings.get(name) {
                Some(size) => Ok(DimInfo::Named(name.clone(), Some(*size))),
                None if op_declared.contains(name) || name.is_empty() || name == "*" => {
                    Ok(dim.clone())
                }
                None => Err(format!("missing symbolic dimension binding `{name}`")),
            },
        }
    };

    let mut rebound = Dag::new();
    for node in dag.nodes() {
        let output_type = TensorType {
            dims: node
                .output_type
                .dims
                .iter()
                .map(&bind_dim)
                .collect::<Result<_, _>>()?,
            precision: node.output_type.precision,
        };
        let op = match &node.op {
            RiscOp::Expand { axis, size } => RiscOp::Expand {
                axis: *axis,
                size: size.clone(),
            },
            RiscOp::Reshape { new_shape } => RiscOp::Reshape {
                new_shape: new_shape
                    .iter()
                    .map(|dim| match dim {
                        RtDim::Sym(name) => match bindings.get(name) {
                            Some(size) => Ok(RtDim::Lit(*size)),
                            // chelis#616: an op-declared symbol (and an
                            // anonymous wildcard) resolves during evaluation,
                            // not here.
                            None if op_declared.contains(name)
                                || name.is_empty()
                                || name == "*" =>
                            {
                                Ok(dim.clone())
                            }
                            None => Err(format!("missing symbolic dimension binding `{name}`")),
                        },
                        // `Lit` passes through; `Node` (runtime) targets are
                        // resolved by the evaluator from `inputs`, not here.
                        RtDim::Lit(n) => Ok(RtDim::Lit(*n)),
                        RtDim::Node(i) => Ok(RtDim::Node(*i)),
                        RtDim::InputAxis { tensor, axis } => Ok(RtDim::InputAxis {
                            tensor: *tensor,
                            axis: *axis,
                        }),
                        RtDim::ToEnd => {
                            Err("reshape target dim cannot be a shrink-to-end sentinel".to_string())
                        }
                    })
                    .collect::<Result<_, _>>()?,
            },
            // chelis#368: resolve the `SHRINK_TO_END` full-axis sentinel to the
            // axis's now-bound extent (from this node's bound output type). The
            // sentinel is emitted only for a full-axis slice of a symbolic
            // bystander axis (the Pad adjoint's no-pad axes, and since
            // chelis#513 the Stride adjoint's trim and the ProdReduce
            // adjoint's per-element slices), so the resolved `end` is exactly
            // `start` plus the output dim's size.
            RiscOp::Shrink { bounds }
                if bounds.iter().any(|(_, end)| matches!(end, RtDim::ToEnd)) =>
            {
                let resolved = bounds
                    .iter()
                    .zip(output_type.dims.iter())
                    .map(|((start, end), dim)| {
                        if matches!(end, RtDim::ToEnd) {
                            // chelis#1480, `spec/05` section 2.4.1: the
                            // sentinel is well formed only beside a `Lit(0)`
                            // start. This used to accept any literal and
                            // resolve `(Lit(1), ToEnd)` to `(Lit(1), Lit(1 +
                            // size))`, which is a slice the spec says is a
                            // malformed bound rather than a slice at all.
                            let start_lit = match start.as_lit() {
                                Some(0) => 0usize,
                                _ => {
                                    return Err(
                                        "shrink-to-end sentinel requires a literal zero start"
                                            .to_string(),
                                    );
                                }
                            };
                            match dim {
                                DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => {
                                    Ok((RtDim::Lit(start_lit), RtDim::Lit(start_lit + *size)))
                                }
                                // chelis#616: an op-declared (or wildcard)
                                // axis has no pre-eval binding; keep the
                                // sentinel — the evaluator resolves `ToEnd`
                                // to the INPUT's runtime extent, which for
                                // the full-axis identity slice (start 0) is
                                // exactly the sentinel's meaning.
                                DimInfo::Named(name, None)
                                    if start_lit == 0
                                        && (op_declared.contains(name)
                                            || name.is_empty()
                                            || name == "*") =>
                                {
                                    Ok((start.clone(), end.clone()))
                                }
                                DimInfo::Named(name, None) => Err(format!(
                                    "shrink-to-end sentinel left unbound for symbolic \
                                     dimension `{name}`"
                                )),
                            }
                        } else {
                            // `Lit` passes through; `Node` (runtime) bounds are
                            // resolved by the evaluator from `inputs`, not here.
                            Ok((start.clone(), end.clone()))
                        }
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                RiscOp::Shrink { bounds: resolved }
            }
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
                accumulator,
            } => RiscOp::BlasMatmul {
                batch_dims: batch_dims
                    .iter()
                    .map(|dim| dim.bind(bindings))
                    .collect::<Result<_, _>>()?,
                m: m.bind(bindings)?,
                n: n.bind(bindings)?,
                k: k.bind(bindings)?,
                accumulator: *accumulator,
            },
            other => other.clone(),
        };
        let new_id = rebound.add_node(op, node.inputs.clone(), output_type, None);
        // chelis#616: binding is a 1:1 id-preserving rebuild; shape-only
        // deps (runtime-dim declarers, `lower_if` placeholder shape sources)
        // must survive it — the evaluator reads them.
        if let Some(new_node) = rebound.node_mut(new_id) {
            new_node.shape_deps = node.shape_deps.clone();
        }
        if dag.is_root(node.id) {
            rebound.add_root(new_id);
        }
    }
    Ok(rebound)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    #[test]
    fn empty_dag() {
        let dag = Dag::new();
        assert!(dag.is_empty());
        assert_eq!(dag.len(), 0);
        assert!(dag.topological_order().is_empty());
    }

    #[test]
    fn add_const_node() {
        let mut dag = Dag::new();
        let id = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 42.0),
            vec![],
            scalar_f32(),
            None,
        );
        assert_eq!(id, NodeId(0));
        assert_eq!(dag.len(), 1);
        let node = dag.get(id).unwrap();
        assert_eq!(node.op, RiscOp::synth_const(Prim::F32, 42.0));
        assert!(node.inputs.is_empty());
    }

    #[test]
    fn add_binary_op() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        assert_eq!(dag.len(), 3);
        let node = dag.get(c).unwrap();
        assert_eq!(node.inputs, vec![NodeId(0), NodeId(1)]);
    }

    #[test]
    fn topological_order() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(RiscOp::Neg, vec![a], scalar_f32(), None);
        let order = dag.topological_order();
        assert_eq!(order, vec![a, b]);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let dag = Dag::new();
        assert!(dag.get(NodeId(99)).is_none());
    }

    #[test]
    fn roots_can_be_registered() {
        let mut dag = Dag::new();
        let id = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_root(id);
        dag.add_root(id);
        assert_eq!(dag.roots(), &[id]);
        assert!(dag.is_root(id));
    }

    #[test]
    fn symbolic_occurrences_sibling_sweep_picks_up_const_dims() {
        // Bucket 4c: a `Const` node (or any non-Load node) with a
        // polymorphic dim must produce a synthetic occurrence so the
        // C codegen can declare the dim from a Load that carries it.
        let load_ty = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let const_ty = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Load { name: "x".into() }, vec![], load_ty, None);
        dag.add_node(
            RiscOp::synth_const(const_ty.precision, 1.0),
            vec![],
            const_ty,
            None,
        );

        let occurrences = symbolic_occurrences(&dag);
        // The Load is the canonical source. The sibling-sweep pass must
        // not duplicate the Load occurrence for the Const (the Const's
        // dim is already covered).
        assert_eq!(occurrences.len(), 1);
        assert_eq!(occurrences[0].name, "n");
        assert_eq!(
            occurrences[0].source,
            SymbolicDimSource::Load {
                input_label: "x".into(),
                axis: 0,
            }
        );
    }

    #[test]
    #[should_panic(expected = "internal compiler error: symbolic dim `n` is referenced")]
    fn symbolic_occurrences_panics_when_dim_has_no_load_source() {
        // Negative parity for the sibling sweep: if a non-Load node
        // declares a polymorphic dim that no Load carries, the C
        // codegen would emit `(int[]){ n }` against an undeclared
        // identifier. The sweep panics rather than producing
        // un-compilable C.
        let const_ty = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        // Only a Const with a polymorphic dim, no Load. There is no
        // input slot to pull the dim value from.
        dag.add_node(
            RiscOp::synth_const(const_ty.precision, 1.0),
            vec![],
            const_ty,
            None,
        );
        let _ = symbolic_occurrences(&dag);
    }

    #[test]
    fn symbolic_params_and_bindings_follow_input_order() {
        let ty_x = TensorType {
            dims: vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let ty_y = TensorType {
            dims: vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty_x, None);
        dag.add_node(RiscOp::Load { name: "y".into() }, vec![], ty_y, None);

        assert_eq!(symbolic_params(&dag), vec!["batch"]);
        assert_eq!(
            symbolic_occurrences(&dag),
            vec![
                SymbolicDimOccurrence {
                    name: "batch".into(),
                    source: SymbolicDimSource::Load {
                        input_label: "x".into(),
                        axis: 0,
                    },
                },
                SymbolicDimOccurrence {
                    name: "batch".into(),
                    source: SymbolicDimSource::Load {
                        input_label: "y".into(),
                        axis: 0,
                    },
                },
            ]
        );

        let bindings = symbolic_bindings(&dag);
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].canonical.source,
            SymbolicDimSource::Load {
                input_label: "x".into(),
                axis: 0,
            }
        );
        assert_eq!(bindings[0].others.len(), 1);
        assert_eq!(
            bindings[0].others[0].source,
            SymbolicDimSource::Load {
                input_label: "y".into(),
                axis: 0,
            }
        );
    }

    #[test]
    fn bind_symbolic_dims_rewrites_output_types_and_preserves_input_axis_sizes() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("batch".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let h = dag.add_node(
            RiscOp::Load { name: "h".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("hidden".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let y = dag.add_node(
            RiscOp::Expand {
                axis: 1,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![x, h],
            TensorType {
                dims: vec![
                    DimInfo::Named("batch".into(), None),
                    DimInfo::Named("hidden".into(), None),
                ],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(y);

        let rebound = bind_symbolic_dims(
            &dag,
            &UnordMap::from([
                ("batch".to_string(), 3usize),
                ("hidden".to_string(), 8usize),
            ]),
        )
        .expect("bindings should apply");
        let node = rebound.get(y).unwrap();
        assert_eq!(
            node.output_type.dims,
            vec![
                DimInfo::Named("batch".into(), Some(3)),
                DimInfo::Named("hidden".into(), Some(8)),
            ]
        );
        assert_eq!(
            node.op,
            RiscOp::Expand {
                axis: 1,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            }
        );
        assert_eq!(rebound.roots(), &[y]);
    }

    /// chelis#368: `bind_symbolic_dims` resolves a `Shrink`'s `SHRINK_TO_END`
    /// full-axis sentinel to the axis's now-bound extent (read from the bound
    /// output type), leaving an explicitly-padded axis untouched. The
    /// sentinel is the Pad-adjoint's no-pad-axis identity for a symbolic dim.
    #[test]
    fn bind_symbolic_dims_resolves_shrink_to_end_sentinel() {
        let mut dag = Dag::new();
        let g = dag.add_node(
            RiscOp::Load { name: "g".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Named("m".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        // Adjoint of a Pad that padded axis 0 (concat axis, before=1) and
        // left axis 1 (`m`, symbolic) unpadded: `(1, 2)` on axis 0 (concrete),
        // `(0, SHRINK_TO_END)` full-axis identity on axis 1.
        let shrunk = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![
                    (RtDim::Lit(1), RtDim::Lit(2)),
                    (RtDim::Lit(0), RtDim::ToEnd),
                ],
            },
            vec![g],
            TensorType {
                dims: vec![DimInfo::Lit(1), DimInfo::Named("m".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(shrunk);

        let rebound = bind_symbolic_dims(&dag, &UnordMap::from([("m".to_string(), 3usize)]))
            .expect("bindings should apply");
        let node = rebound.get(shrunk).unwrap();
        // The concrete axis-0 bound is untouched; the sentinel axis-1 bound
        // resolves to `(0, 0 + 3)` = the full bound extent.
        assert_eq!(
            node.op,
            RiscOp::Shrink {
                bounds: vec![
                    (RtDim::Lit(1), RtDim::Lit(2)),
                    (RtDim::Lit(0), RtDim::Lit(3))
                ],
            },
            "SHRINK_TO_END must resolve to (start, start + bound extent)",
        );
    }

    /// chelis#368: an UNBOUND symbolic dim under a `SHRINK_TO_END` sentinel is
    /// a hard error, never a silent miscompile (no binding to read the extent
    /// from).
    #[test]
    fn bind_symbolic_dims_rejects_unbound_shrink_to_end() {
        let mut dag = Dag::new();
        let g = dag.add_node(
            RiscOp::Load { name: "g".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("m".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let shrunk = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::ToEnd)],
            },
            vec![g],
            TensorType {
                dims: vec![DimInfo::Named("m".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(shrunk);
        // No binding for `m`: bind_dim fails first on the output type, but
        // even a partial binding map must not silently drop the sentinel.
        let err = bind_symbolic_dims(&dag, &UnordMap::new());
        assert!(err.is_err(), "unbound symbolic dim must fail closed");
    }

    // --- chelis#345: op-internal symbolic references (Bucket 4d) ---
    //
    // The Bucket 4c sibling sweep scans only `output_type.dims`, so a
    // node whose OUTPUT dims are fully concrete can still smuggle an
    // undeclared symbolic dim through an op-internal field:
    // `Expand { size: Sym(_) }`, `Reshape { new_shape }`, and
    // `BlasMatmul { batch_dims, m, n, k }` (the latter is rendered
    // verbatim into C by `emit_dim_expr`). chelis#345's bisect found
    // exactly this mixed state in a grad helper DAG. These tests pin
    // that the sweep covers op-internal references with the same
    // panic-don't-emit contract as output dims.

    #[test]
    fn symbolic_occurrences_input_axis_uses_only_its_structural_load_source() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("n".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let value = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![value, x],
            TensorType {
                dims: vec![DimInfo::Named("n".into(), None)],
                precision: Prim::F32,
            },
            None,
        );

        let occurrences = symbolic_occurrences(&dag);
        assert_eq!(occurrences.len(), 1);
        assert_eq!(occurrences[0].name, "n");
        assert_eq!(
            occurrences[0].source,
            SymbolicDimSource::Load {
                input_label: "x".into(),
                axis: 0,
            }
        );
    }

    #[test]
    fn symbolic_occurrences_input_axis_from_runtime_movement_is_op_declared() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let runtime_source = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::Lit(3))],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Named("m".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let value = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        let expanded = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![value, runtime_source],
            TensorType {
                dims: vec![DimInfo::Named("k".into(), None)],
                precision: Prim::F32,
            },
            None,
        );

        let occurrences = symbolic_occurrences(&dag);
        assert!(occurrences.contains(&SymbolicDimOccurrence {
            name: "k".into(),
            source: SymbolicDimSource::OpDeclared {
                node: expanded,
                axis: 0,
            },
        }));
    }

    #[test]
    fn verifier_rejects_expand_size_sym_without_consulting_the_symbol_walk() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            RiscOp::Expand {
                axis: 1,
                size: RtDim::Sym("d7".into()),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        let errors = crate::verify::verify(&dag);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("expand") && error.contains("Sym")),
            "{errors:?}"
        );
    }

    #[test]
    #[should_panic(expected = "internal compiler error: symbolic dim `d9` is referenced")]
    fn symbolic_occurrences_panics_on_reshape_shape_sym_without_load() {
        // Negative: `Reshape::new_shape` carries an unbound Named dim
        // while the node's own output dims are concrete.
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Sym("d9".into()), RtDim::Lit(2)],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        let _ = symbolic_occurrences(&dag);
    }

    #[test]
    #[should_panic(expected = "internal compiler error: symbolic dim `d11` is referenced")]
    fn symbolic_occurrences_panics_on_blas_matmul_dim_sym_without_load() {
        // Negative: `BlasMatmul::{m,n,k}` are emitted verbatim into C
        // (`emit_dim_expr`), so an unbound sym there is exactly the
        // undeclared-identifier hazard the guard exists for. `k` is the
        // contraction dim and never appears in the output type at all.
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
                precision: Prim::F32,
            },
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            RiscOp::matmul_default(
                vec![],
                DimExpr::Concrete(2),
                DimExpr::Concrete(4),
                DimExpr::Sym("d11".into()),
                Prim::F32,
            )
            .expect("f32 matmul"),
            vec![a, b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let _ = symbolic_occurrences(&dag);
    }

    #[test]
    fn symbolic_occurrences_concrete_op_internals_stay_quiet() {
        // Negative parity for the sweep itself: fully concrete
        // op-internal fields must not invent occurrences or panic.
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            RiscOp::Expand {
                axis: 1,
                size: RtDim::Lit(3),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
                precision: Prim::F32,
            },
            None,
        );
        assert!(symbolic_occurrences(&dag).is_empty());
    }

    #[test]
    fn symbolic_occurrences_traces_reduction_kept_axis_to_load() {
        // chelis#551/#340: a host-lane / grad-backward reduction whose
        // operand is a `concat` output carries a `*`-wildcard non-concat
        // axis. The C backend renames the operand's wildcard and the
        // reduction's surviving wildcard to DIFFERENT `_anon_dim_*` names,
        // so the reduction output's dim symbol does not appear in any Load.
        // The reduction arm of `shape_source_for_axis` must map the kept
        // output axis back through the removed reduce axis to the declaring
        // Load rather than tripping the sourceless-symbol panic.
        let mut dag = Dag::new();
        // Load "c": [batch, d1] (d1 == the concat wildcard, renamed).
        let c = dag.add_node(
            RiscOp::Load { name: "c".into() },
            vec![],
            TensorType {
                dims: vec![
                    DimInfo::Named("batch".into(), None),
                    DimInfo::Named("d1".into(), None),
                ],
                precision: Prim::F32,
            },
            None,
        );
        // Sum over axis 0 keeps axis 1, re-anonymised to a DIFFERENT symbol.
        dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            vec![c],
            TensorType {
                dims: vec![DimInfo::Named("d2".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let occurrences = symbolic_occurrences(&dag);
        let d2 = occurrences
            .iter()
            .find(|o| o.name == "d2")
            .expect("reduction kept-axis symbol `d2` must be declared");
        // The kept output axis 0 maps back to input axis 1 of Load "c".
        assert_eq!(
            d2.source,
            SymbolicDimSource::Load {
                input_label: "c".into(),
                axis: 1,
            }
        );
    }

    #[test]
    #[should_panic(expected = "internal compiler error: symbolic dim `d2` is referenced")]
    fn symbolic_occurrences_reduction_arm_still_fails_loud_without_load() {
        // Negative parity for the reduction arm (chelis#551): the arm must
        // NOT launder a genuinely-unbound symbolic dim green. When the
        // reduction operand traces to a non-Load with no declaring Load
        // (here a `Const`), the kept-axis symbol is unrecoverable and the
        // guard must still panic — do not weaken fail-loud.
        let mut dag = Dag::new();
        // A CONCRETE-dim Const operand (so its own dims do not trip the
        // guard) whose reduction output nonetheless carries an unbound
        // symbol. The reduction arm recurses into the Const and bottoms out
        // (`Const` is not a Load and has no shape source), so the kept-axis
        // symbol is unrecoverable and the guard must panic.
        let src = dag.add_node(
            RiscOp::synth_const(
                TensorType {
                    dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
                    precision: Prim::F32,
                }
                .precision,
                0.0,
            ),
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            vec![src],
            TensorType {
                dims: vec![DimInfo::Named("d2".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let _ = symbolic_occurrences(&dag);
    }

    /// One instance of every `RiscOp` variant. The
    /// `is_verifier_targetable` classifier (WI-2) is a wildcard-free
    /// exhaustive match, so adding a variant to the enum is a compile
    /// error there; this sample is the runtime companion guard. If a new
    /// variant is added but not appended here, the count assertion in
    /// `every_risc_op_is_classified_for_verifier_subset` fails, so a new
    /// op cannot silently escape classification by either route.
    fn one_of_every_risc_op() -> Vec<RiscOp> {
        vec![
            RiscOp::Add,
            RiscOp::Mul,
            RiscOp::Div,
            RiscOp::FloorDiv,
            RiscOp::TruncDiv,
            RiscOp::Mod,
            RiscOp::CmpLt,
            RiscOp::MaxElem,
            RiscOp::Neg,
            RiscOp::Exp,
            RiscOp::Log,
            RiscOp::Sin,
            RiscOp::Sqrt,
            RiscOp::Cos,
            RiscOp::Tan,
            RiscOp::Atan,
            RiscOp::Abs,
            RiscOp::Floor,
            RiscOp::Ceil,
            RiscOp::Round,
            RiscOp::Recip,
            RiscOp::UniformLike {
                low: 0.0,
                high: 1.0,
                seed: 7,
            },
            RiscOp::Dropout { rate: 0.5, seed: 7 },
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            RiscOp::MaxReduce { axis: 0 },
            RiscOp::MinReduce { axis: 0 },
            RiscOp::ProdReduce { axis: 0 },
            RiscOp::ReduceWindow {
                reducer: ReduceWindowKind::Max,
                window_shape: vec![2],
                strides: vec![1],
            },
            RiscOp::ReduceWindowGrad {
                reducer: ReduceWindowKind::Max,
                window_shape: vec![2],
                strides: vec![1],
            },
            RiscOp::Argmax { axis: 0 },
            RiscOp::Argmin { axis: 0 },
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(4)],
            },
            RiscOp::Permute { axes: vec![0] },
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Lit(4),
            },
            RiscOp::OneHot { vocab: 8 },
            RiscOp::zero_pad(Prim::F32, vec![(RtDim::Lit(0), RtDim::Lit(0))]),
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::Lit(1))],
            },
            RiscOp::Stride {
                strides: vec![RtDim::Lit(1)],
            },
            RiscOp::Shape { axis: 0 },
            RiscOp::synth_const(Prim::F32, 1.0),
            RiscOp::synth_const_tensor(Prim::F32, vec![1.0, 2.0]),
            RiscOp::Load {
                name: LoadStoreName::must("x"),
            },
            RiscOp::Store {
                name: LoadStoreName::must("y"),
            },
            RiscOp::Copy,
            RiscOp::Drop,
            RiscOp::Realize,
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            RiscOp::CastTrunc {
                new_precision: Prim::Int32,
            },
            RiscOp::FusedElem { ops: vec![] },
            RiscOp::BlasMatmul {
                batch_dims: vec![],
                m: DimExpr::Concrete(2),
                n: DimExpr::Concrete(2),
                k: DimExpr::Concrete(2),
                accumulator: Prim::F32,
            },
            RiscOp::Gather { axis: 0 },
            RiscOp::ScatterAdd { axis: 0 },
            RiscOp::Scatter { axis: 0 },
            RiscOp::ScatterElements { axis: 0 },
            RiscOp::Relu,
            RiscOp::ReluAdjoint,
        ]
    }

    /// Exhaustiveness guard for the WI-2 verifier/Beacon op subset: every
    /// `RiscOp` variant must be classified, and the in/out partition must
    /// match the documented `beacon_plan.md` §3.1 corpus. A future new op
    /// added to the enum (and to `one_of_every_risc_op`) must land in one
    /// side of this partition explicitly; if it is added to the enum but
    /// not to the partition lists here, the count assertions fail.
    #[test]
    fn every_risc_op_is_classified_for_verifier_subset() {
        let all = one_of_every_risc_op();
        // Explicit targetability samples include both dedicated ReLU
        // identities so they cannot inherit a verifier disposition.
        assert_eq!(
            all.len(),
            56,
            "one_of_every_risc_op must list all 56 classified samples"
        );

        // The classifier returns a definite bool for every variant (no
        // panic, no escape); count both sides and pin the partition.
        let targetable = all.iter().filter(|op| op.is_verifier_targetable()).count();
        let excluded = all.len() - targetable;

        // Pinned partition per beacon_plan.md §3.1: the elementwise math
        // (5 binary/cmp + 13 unary, including `round`), 5 reductions, 6
        // movement, 4 memory/blas value nodes (Const, ConstTensor, Load,
        // BlasMatmul), and Cast are targetable (34); stochastic (2),
        // arg-reductions (2), integer floor/trunc division and remainder (3),
        // `cast_trunc` (1, chelis#759), one_hot (1), the `Shape` metadata read
        // (1), sparse gather/scatter (4, including element-wise
        // `ScatterElements`), linearity/lifecycle markers + store (4),
        // reduce-window-grad (1), fused-elem (1), and the dedicated ReLU
        // identity/adjoint (2) are excluded (22) until Beacon registers their
        // own transformers.
        assert_eq!(
            targetable, 34,
            "targetable op count drifted from the pinned WI-2 subset"
        );
        assert_eq!(
            excluded, 22,
            "excluded op count drifted from the pinned WI-2 subset"
        );

        // Spot-check a representative op on each side so a wrong
        // reclassification (not just a count drift) is caught.
        assert!(RiscOp::Add.is_verifier_targetable());
        assert!(RiscOp::Exp.is_verifier_targetable());
        assert!(
            RiscOp::CmpLt.is_verifier_targetable(),
            "CmpLt drives erf64 branch-and-bound; must be targetable"
        );
        assert!(
            RiscOp::Cast {
                new_precision: Prim::F32
            }
            .is_verifier_targetable(),
            "Cast is real-valued-first targetable (beacon_plan.md §6)"
        );
        assert!(
            !RiscOp::Dropout { rate: 0.5, seed: 0 }.is_verifier_targetable(),
            "stochastic ops have no deterministic envelope to bound"
        );
        assert!(
            !RiscOp::Argmax { axis: 0 }.is_verifier_targetable(),
            "argmax returns discrete indices, not a real envelope"
        );
        assert!(
            !RiscOp::CastTrunc {
                new_precision: Prim::Int32
            }
            .is_verifier_targetable(),
            "cast_trunc is piecewise constant with an integer output; it has \
             no real-valued envelope, unlike the checked `cast`"
        );
        assert!(
            !RiscOp::Mod.is_verifier_targetable(),
            "integer remainder is discrete and has no real-valued envelope"
        );
    }

    #[test]
    fn every_risc_op_has_an_exact_pre_phase4c_atom_disposition() {
        use std::collections::BTreeSet;

        let all = one_of_every_risc_op();
        let mut discovery_cases = all.clone();
        discovery_cases.extend([
            RiscOp::ExtentWitness {
                site: crate::dag::ExtentWitnessSite::Caller,
                parameter: "x".into(),
                axis: RtAxis::Lit(0),
                requirements: vec![chelis_types::scalar_from_i64("load", Prim::Int64, 4).unwrap()],
                claims: Vec::new(),
            },
            RiscOp::Sub,
            RiscOp::MinElem,
            RiscOp::Count { axes: vec![0] },
            RiscOp::ExtremaAdjoint {
                kind: ExtremaKind::Max,
                operand: ExtremaOperand::Left,
            },
        ]);
        discovery_cases.extend(
            [
                ReduceWindowKind::Min,
                ReduceWindowKind::Sum,
                ReduceWindowKind::Mean,
            ]
            .map(|reducer| RiscOp::ReduceWindow {
                reducer,
                window_shape: vec![2],
                strides: vec![1],
            }),
        );
        let semantic: BTreeSet<_> = discovery_cases
            .iter()
            .filter_map(|op| match op.atom_disposition() {
                RiscAtomDisposition::Semantic(identity) => Some(identity),
                RiscAtomDisposition::Structural => None,
            })
            .collect();
        let expected: BTreeSet<_> = RiscAtomIdentity::ALL.iter().copied().collect();

        assert_eq!(
            semantic, expected,
            "the exhaustive RiscOp disposition and canonical semantic identity universe drifted"
        );
        for op in discovery_cases {
            let structural = matches!(
                op,
                RiscOp::ExtentWitness { .. }
                    | RiscOp::OneHot { .. }
                    | RiscOp::Const { .. }
                    | RiscOp::ConstTensor { .. }
                    | RiscOp::Load { .. }
                    | RiscOp::Store { .. }
                    | RiscOp::Copy
                    | RiscOp::Drop
                    | RiscOp::Realize
                    | RiscOp::FusedElem { .. }
            );
            assert_eq!(
                matches!(op.atom_disposition(), RiscAtomDisposition::Structural),
                structural,
                "only the explicit compiler/lifetime representation variants are structural: {op:?}"
            );
        }
    }
}
