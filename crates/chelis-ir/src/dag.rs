//! RISC DAG data structure.
//!
//! The DAG is a flat vector of nodes, each referencing earlier nodes by [`NodeId`].
//! Nodes are always in topological order (an invariant maintained by append-only construction).

use chelis_unord::{UnordMap, UnordSet};
use std::fmt;

pub use chelis_deep::NamedCastMode;
use chelis_types::types::Prim;
use serde::{Deserialize, Serialize};

use crate::load_store_name::LoadStoreName;

/// Index into the DAG node array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub usize);

/// Index into a DAG's [`Dag::declarations`]: the declaration a node belongs to.
///
/// Every node carries one ([`DagNode::decl`]), supplied at its construction
/// ([`Dag::add_node`]): there is no default declaration, so a node built
/// without one does not compile, and a wrong one is visible at its site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DeclId(pub u32);

/// A node's owner (spec/10 section 3.2): the declaration it belongs to and the
/// activation it runs under.
///
/// The activation is the scalar or per-row Bool node that holds exactly when
/// the source position of this node is entered: the conjunction of the
/// enclosing runtime `if` arms' predicates (and of a `grad` or `vmap` body's
/// call-site activation), or `None` where every execution of the graph
/// enters the node. A node whose activation is false is still computed, since
/// a `Where` may read its value, but checks nothing: no trap fires, no draw
/// validates, no count is read.
///
/// Every node carries one, supplied at construction ([`Dag::add_node`]). The
/// activation is a dependency like [`DagNode::shape_deps`]: a pass that
/// rebuilds a graph carries it through its node map ([`Owner::remap`]), which
/// panics on an activation the map does not cover rather than dropping it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Owner {
    /// The declaration this node belongs to, an index into
    /// [`Dag::declarations`].
    pub decl: DeclId,
    /// The Bool node under which this node runs, or `None` when it runs on
    /// every execution of its declaration.
    pub activation: Option<NodeId>,
}

impl Owner {
    /// An owner in `decl` under `activation`.
    pub const fn new(decl: DeclId, activation: Option<NodeId>) -> Self {
        Self { decl, activation }
    }

    /// An owner in `decl` that every execution of `decl` enters.
    pub const fn unconditional(decl: DeclId) -> Self {
        Self {
            decl,
            activation: None,
        }
    }

    /// This owner in a rebuilt graph, its activation carried through `map`
    /// (old node to new node). An activation `map` does not cover is a defect
    /// of the rebuilding pass, never a reason to drop the activation: it
    /// panics.
    pub fn remap_with(self, map: impl FnOnce(NodeId) -> Option<NodeId>) -> Self {
        self.try_remap_with(map)
            .unwrap_or_else(|message| panic!("{message}"))
    }

    /// [`Self::remap_with`] through a node map.
    pub fn remap(self, map: &UnordMap<NodeId, NodeId>) -> Self {
        self.remap_with(|old| map.get(&old).copied())
    }

    /// [`Self::remap_with`], reporting an unmapped activation instead of
    /// panicking, for a pass with a recoverable error channel.
    pub fn try_remap_with(
        self,
        map: impl FnOnce(NodeId) -> Option<NodeId>,
    ) -> Result<Self, String> {
        let activation = match self.activation {
            Some(old) => Some(map(old).ok_or_else(|| {
                format!("node activation {old:?} has no node in the rebuilt graph")
            })?),
            None => None,
        };
        Ok(Self {
            decl: self.decl,
            activation,
        })
    }
}

/// A bare declaration owns its nodes unconditionally: a graph built outside
/// program lowering (a backend helper kernel, a runtime transform, a test)
/// has no runtime branch to activate them under.
impl From<DeclId> for Owner {
    fn from(decl: DeclId) -> Self {
        Self::unconditional(decl)
    }
}

/// One declaration of a graph (chelis#2476, #2413).
///
/// A lowered program holds every top-level declaration's activation in one
/// graph, whether or not anything calls it, and a function's parameters are
/// `Load`s built exactly like an entry's inputs. Selecting roots is a scoping
/// decision, so the graph records which declaration owns each node: a seed (an
/// abort, or a draw that can trap) runs only when a selected root belongs to
/// its declaration ([`Dag::entered_declarations`]), and a parameter is its
/// declaration and its name, never its name alone. A graph built outside
/// program lowering (a backend helper kernel, a runtime transform, a test)
/// registers its own named declaration.
///
/// Another declaration runs a declaration's work only inlined into its own
/// nodes: a function's body where it calls it, and a value's initializer
/// where it reads the value when that initializer may trap. It reads a
/// value declaration's own nodes only when none of them can trap, which the
/// verifier checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Declaration {
    /// The declaration's name; empty for an unnamed top-level expression.
    pub name: String,
    /// Whether it declares a value rather than a function.
    pub value: bool,
}

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
        self.map_symbols(&mut |name| {
            bindings
                .get(name)
                .copied()
                .map(Some)
                .ok_or_else(|| format!("missing symbolic dimension binding `{name}`"))
        })
    }

    fn map_symbols<E>(
        &self,
        resolve: &mut impl FnMut(&str) -> Result<Option<usize>, E>,
    ) -> Result<Self, E> {
        match self {
            Self::Concrete(value) => Ok(Self::Concrete(*value)),
            Self::Sym(name) => Ok(resolve(name)?
                .map(Self::Concrete)
                .unwrap_or_else(|| self.clone())),
            Self::Mul(lhs, rhs) => Ok(Self::Mul(
                Box::new(lhs.map_symbols(resolve)?),
                Box::new(rhs.map_symbols(resolve)?),
            )),
            Self::Div(lhs, rhs) => Ok(Self::Div(
                Box::new(lhs.map_symbols(resolve)?),
                Box::new(rhs.map_symbols(resolve)?),
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
///
/// One variant, and deliberately so since chelis#665. An occurrence exists
/// to say "the function entry supplies this extent from that input's axis",
/// which is what the interface bindings the HIP prologue reads are for. A
/// locally produced extent is not an interface value, so it has no spelling
/// here; [`crate::axis_sources::ExtentOrigin`] is the total answer to where
/// an extent comes from, and a lane that needs a locally produced one reads
/// that instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolicDimSource {
    /// Declared from an input tensor's shape: the C prologue emits
    /// `int name = inputs[slot]->shape[axis];` and the eval lane binds the
    /// value from the corresponding input before evaluation.
    Load { input_label: String, axis: usize },
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
    Neg,
    Recip,
    Exp,
    Log,
    Sin,
    Sqrt,
    Cos,
    Tan,
    Atan,
    Tanh,
    Abs,
    Floor,
    Ceil,
    Round,
}

/// Identity-preserving numeric comparison operation.
///
/// `CmpLt` and `Lt` intentionally remain distinct source identities even
/// though both use the same ordered comparison kernel.
impl FusedStepOp {
    /// [05-OP-46]'s transcendental operations, which no device lane may
    /// compute until it has correctly rounded kernels of its own.
    pub const fn transcendental_name(self) -> Option<&'static str> {
        match self {
            Self::Exp => Some("exp"),
            Self::Log => Some("log"),
            Self::Sin => Some("sin"),
            Self::Cos => Some("cos"),
            Self::Tan => Some("tan"),
            Self::Atan => Some("atan"),
            Self::Tanh => Some("tanh"),
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::FloorDiv
            | Self::TruncDiv
            | Self::MaxElem
            | Self::MinElem
            | Self::Neg
            | Self::Recip
            | Self::Sqrt
            | Self::Abs
            | Self::Floor
            | Self::Ceil
            | Self::Round => None,
        }
    }

    /// The [05-OP-46] operations a device lane may not compute: the
    /// transcendentals, and `sqrt`, whose correct rounding neither device
    /// lane establishes (Metal compiles with fast math, and the HIP runtime
    /// compile does not pin a correctly rounded square root).
    pub const fn device_fenced_name(self) -> Option<&'static str> {
        match self {
            Self::Sqrt => Some("sqrt"),
            _ => self.transcendental_name(),
        }
    }
}

/// chelis#2957 GPU fence (`spec/design/correctly_rounded_math.md` §4.3): a
/// device lane has no correctly rounded kernels for [05-OP-46]'s
/// transcendentals or `sqrt`, so a DAG that computes one, directly or inside
/// a fused chain, is rejected through [05-UNS-1] rather than computed with a
/// vendor library.
pub fn reject_device_correctly_rounded_ops(
    nodes: &[DagNode],
    target: &'static str,
) -> Result<(), chelis_types::unsupported::Unsupported> {
    for node in nodes {
        let name = match &node.op {
            RiscOp::Exp => Some("exp"),
            RiscOp::Log => Some("log"),
            RiscOp::Sin => Some("sin"),
            RiscOp::Cos => Some("cos"),
            RiscOp::Tan => Some("tan"),
            RiscOp::Atan => Some("atan"),
            RiscOp::Tanh => Some("tanh"),
            RiscOp::Sqrt => Some("sqrt"),
            RiscOp::FusedElem { ops } => ops.iter().find_map(|step| step.op.device_fenced_name()),
            _ => None,
        };
        if let Some(name) = name {
            return Err(chelis_types::unsupported::Unsupported::new(
                chelis_types::unsupported::UnsupportedKind::Op(name.to_string()),
                format!(
                    "`{name}` must be correctly rounded and the {target} device lane has no correctly rounded kernel for it; build for `--target c`"
                ),
                chelis_types::unsupported::Stage::Codegen(target),
                chelis_types::deliberate_rejection!(
                    "[05-OP-46]",
                    "device transcendentals and sqrt are fenced until the device lane has correctly rounded kernels"
                ),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonKind {
    CmpLt,
    Lt,
    Eq,
    Neq,
    Gt,
    Gte,
    Lte,
}

impl ComparisonKind {
    pub const fn surf_name(self) -> &'static str {
        match self {
            Self::CmpLt => "cmplt",
            Self::Lt => "lt",
            Self::Eq => "eq",
            Self::Neq => "neq",
            Self::Gt => "gt",
            Self::Gte => "gte",
            Self::Lte => "lte",
        }
    }
}

/// Bool-only eager logical operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalKind {
    And,
    Or,
    Not,
}

impl LogicalKind {
    pub const fn surf_name(self) -> &'static str {
        match self {
            Self::And => "and",
            Self::Or => "or",
            Self::Not => "not",
        }
    }

    pub const fn arity(self) -> usize {
        match self {
            Self::And | Self::Or => 2,
            Self::Not => 1,
        }
    }
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

/// Which `[05-OP-8]` bound a [`RiscOp::UniformBoundAdjoint`] materializes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UniformBound {
    Low,
    High,
}

/// Which half of `[05-OP-70]`'s pair a [`RiscOp::Split`] produces: `Left` is
/// `derive(k, 0)` and `Right` is `derive(k, 1)` of `[05-RNG-2]`. `split_key`
/// is two nodes because an IR node has one output (LaCaDiLE's
/// `KeyPath.left/right`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyBranch {
    Left,
    Right,
}

impl KeyBranch {
    /// The kernel half this branch computes.
    pub const fn half(self) -> chelis_types::dtype_semantics::KeyHalf {
        match self {
            Self::Left => chelis_types::dtype_semantics::KeyHalf::Left,
            Self::Right => chelis_types::dtype_semantics::KeyHalf::Right,
        }
    }
}

/// How a key-operand random primitive's inputs relate to its key batch
/// (spec/10 §3.2, rule V5). Every lane checks their runtime extents in this
/// order before it reads one: the data's leading axes against the key's
/// shape, then each per-row input's axes against the key's leading ones,
/// then the node's own activation's ([`Owner::activation`]), which is shaped
/// like a leading part of the key's shape too. The DAG evaluator and the C
/// lane both read this one table, so they check the same inputs in the same
/// order and report the same line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawBatchLayout {
    /// The operation its traps name.
    pub op: &'static str,
    /// The key's input slot.
    pub key: usize,
    /// The input whose leading axes are the key's shape: the data, the
    /// template, or a bound adjoint's cotangent.
    pub data_input: usize,
    /// The controls' slots, each shaped like a leading part of the key's
    /// shape. The activation is the node's owner's, not an input.
    pub per_row: &'static [usize],
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtentWitnessSite {
    /// One authored literal result obligation, stored as a tagged scalar.
    /// The producing node retains this token through an ordered shape dependency.
    LiteralResultClaim,
    Caller,
    LocalExpand,
    /// A declaring extent retained for a result obligation. The witness's
    /// value is the required extent; its node identity distinguishes calls.
    /// A producer's shape dependency on this witness owns the comparison at
    /// this output axis. The label is diagnostic, never a dimension binding.
    ResultClaim {
        claim: String,
        axis: RtAxis,
    },
    /// One checker-retained authored local tensor-ascription obligation.
    /// The stable checker identity and binding name are provenance; `claim`
    /// is the authored axis spelling used by diagnostics. The enclosing
    /// producer's `shape_deps` edge owns execution at the initializer op.
    LocalAscriptionClaim {
        ascription_id: u64,
        binding: String,
        claim: String,
        axis: RtAxis,
    },
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
    /// [05-OP-54]: a tensor carrier for the half-open i64 `range(start, end)`.
    /// Inputs are two exact rank-zero i64 values; output is rank-one i64.
    /// The runtime length is max(end-start, 0), checked in mathematical
    /// integers before allocation. Integer endpoints have no cotangent.
    Iota,
    /// A scalar lexical read in a runtime List map. The first capture is the
    /// invocation identity and reads its rank-one carrier in input 1. Later
    /// captures name that first capture in input 1. CSE preserves identities.
    /// Input 1's first axis supplies the result length in either case.
    /// Unlike authored Expand, AD retains each consumer edge until it reaches
    /// input 0, interleaving edges by invocation row before accumulation.
    ListMapCapture {
        first: bool,
    },
    /// spec/06 §2.4's positive-zero-prefixed, own-dtype adjacent-pair tree.
    /// Each group consumes this many consecutive inputs. Rank-one inputs in
    /// one group are interleaved row first, then input order; a scalar group
    /// has exactly one input. Groups follow canonical forward order.
    OrderedAdjointSum {
        groups: Vec<usize>,
    },
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
    /// `Std.Decimal` limb arithmetic relies on. Non-differentiable;
    /// `grad` rejects it. See `spec/05-risc-primitives.md` §2.1.
    TruncDiv,
    /// Exact signed remainder, with DivZero traps at the stored width
    /// and dividend-sign semantics under [05-OP-64].
    Mod,
    /// Exact signed-width [05-OP-47] operation.
    Bitwise(chelis_types::BitwiseKind),
    /// Identity-preserving comparison with Bool output ([05-OP-36]).
    Compare(ComparisonKind),
    /// Bool-only eager logical operation ([05-OP-26..28]).
    Logical(LogicalKind),
    /// Eager stored-bit conditional selection ([05-OP-53]).
    Where,
    /// Guarded abort ([05-OP-68]). Inputs are `(condition, fallback)`.
    /// When `condition` is true the program aborts with `message`; otherwise
    /// the result is `fallback`'s stored bits unchanged.
    ///
    /// chelis#1464: a scalar `if` whose branch is `fail(...)` cannot lower to
    /// [`RiscOp::Where`], because `Where` selects between branch VALUES and a
    /// trap has none. Lowering that branch to a placeholder value instead
    /// made a TAKEN `fail` return the placeholder with exit 0, discarding a
    /// user-authored abort and violating `spec/06-transformations.md` §2.10.1
    /// and §5.2. This identity keeps the trap in the graph, so its occurrence
    /// survives differentiation, batching, optimization and code generation.
    GuardedFail {
        /// The authored abort message. Part of the operation's identity, so
        /// the DAG needs no string value vocabulary to carry it.
        message: String,
        /// Whether the abort fires when the condition is true. A `fail` in
        /// the `else` branch lowers with this false rather than synthesizing
        /// a separate negation node.
        trap_on_true: bool,
    },
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
    /// [05-OP-48] identity retained through AD; one float tensor input.
    Softmax {
        axis: usize,
    },
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
    Tanh,
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
    /// `[05-OP-8]` with operand controls. Inputs are exactly `[template, low,
    /// high, key]`. The template supplies only the shape and dtype `p`; `low`
    /// and `high` are floats of dtype `p`, or f32 while the checker's bound
    /// signature is f32 (chelis#1295); `key` is this draw's `Prim::Key`,
    /// consumed here. Under spec/10 §3.2's rule V5 the key's shape is the
    /// template's leading axes and the bounds and the node's activation
    /// ([`Owner::activation`]) are shaped like leading parts of the key's
    /// shape. An inactive draw validates nothing and produces positive
    /// zeros.
    UniformLike,
    /// `[05-OP-37]` with an operand rate. Inputs are exactly `[x, rate,
    /// key]`, shaped as for `UniformLike`; `rate` is a value of `x`'s dtype
    /// and `key` is consumed here. An inactive draw validates nothing and
    /// produces positive zeros.
    Dropout,
    /// AD-only `[05-OP-37]` pathwise input adjoint. Inputs are exactly `[g,
    /// rate, key]`, and its owner is its forward draw's, activation included.
    /// It reads its forward `Dropout`'s key and rate without consuming the
    /// key, and applies the same saved mask and finalized sub/div to the
    /// cotangent.
    DropoutReplay,
    /// AD-only `[05-OP-8]` bound adjoint. Inputs are exactly `[template, g,
    /// key]`, and its owner is its forward draw's, activation included; the
    /// result is a value of the template's dtype shaped like a leading part
    /// of the key's shape (rule V5). It reads its forward `UniformLike`'s key
    /// without consuming it.
    UniformBoundAdjoint {
        bound: UniformBound,
    },
    /// `[05-OP-69]` `key_from_seed`: input `[seed: tensor[D, i64]]`, output
    /// the `tensor[D, key]` of each seed's two's-complement bits. Pure and
    /// never constant-folded, so an exported key stays symbolic.
    KeyFromSeed,
    /// One half of `[05-OP-70]` `split_key`: input exactly `[k: tensor[D,
    /// key]]`, output the `tensor[D, key]` of `derive(k, 0)` (`Left`) or
    /// `derive(k, 1)` (`Right`). The node's activation ([`Owner::activation`])
    /// is shaped like a leading part of `D`. A parent feeds at most one
    /// `Split` of each branch, and nothing else, unless rule V3 admits the
    /// sharing through exclusive activations (spec/10 §3.2). The activation
    /// changes no key.
    Split {
        branch: KeyBranch,
    },
    /// `[05-OP-72]` `fold_in`: inputs exactly `[k: tensor[D, key], n:
    /// tensor[D, i64]]` of equal shape; output `derive(derive(k, 2), n)`
    /// element-wise. The node's activation is shaped like a leading part of
    /// `D` and changes no key.
    FoldIn,
    /// `[05-OP-71]` `split_keys`: input `[k: tensor[D, key]]`, then the
    /// rank-0 exact i64 count node when `count` is `RtDim::Node(1)`, and
    /// nothing else; the node's activation is shaped like a leading part of
    /// `D`. The output is `tensor[D ++ [count], key]`, the new axis last; row
    /// `j` is `derive(derive(k, 2), j)`. A negative runtime count traps before
    /// allocation, as a negative movement bound does. Where the activation
    /// holds in no row the count is not read: the count axis takes the extent
    /// the output type declares where another node or a literal fixes it, and
    /// zero where the split itself declares it, so an unselected arm's split
    /// neither traps nor allocates on its count.
    SplitN {
        count: RtDim,
    },
    /// Rule S's join (spec/10 §3.2): the key of a runtime `if` whose value is
    /// a key. Inputs are `[then_key, else_key, then_active, else_active]`:
    /// two keys of the result's exact type, then the two arms' Bool
    /// activations, each shaped like a leading part of the key's shape.
    /// Input 0 is consumed under `then_active` and input 1 under
    /// `else_active`. The two are the join's own activation
    /// ([`Owner::activation`], the enclosing one) conjoined with the branch's
    /// condition and with its negation, so where the join's activation holds
    /// exactly one of them does. Element `i` is `then_key[i]` where
    /// `then_active` holds for its row, and `else_key[i]` elsewhere. The
    /// result is a fresh key under the join's activation; the join derives
    /// nothing and changes no key.
    KeySelect,

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
    /// strictly descending order. The result precision is always i64.
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
    /// the type-system result is canonically `tensor[..., i64]`
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
    /// `i32`, while the hydronnx ONNX translator constructs it as
    /// `i64` per chelis#558).
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
    /// A call's shape-only witness. Requirements are tagged i64 literals,
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
    /// input and the result is scalar i64. The nonempty `claims` labels
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
    /// A named lossy cast of the chelis#759 ladder: `cast_trunc`
    /// ([05-OP-6]). A separate op rather than a mode flag on `Cast` so
    /// every backend, evaluator, and adjoint site is forced by exhaustive
    /// matching to state its disposition instead of inheriting the checked
    /// default's, and the rung is an enum so each site states it per rung.
    NamedCast {
        mode: NamedCastMode,
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
    /// (operand precision in {i8, i16, i32, i64}) is NOT
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
        /// Leading axes paired positionwise between values and indices.
        /// They occur once in the result, never once per operand.
        batch_rank: usize,
    },

    /// Sparse scatter-add used by gather's adjoint. Inputs are
    /// `target, indices, updates`; duplicate indices accumulate.
    ScatterAdd {
        axis: usize,
        batch_rank: usize,
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
        batch_rank: usize,
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
    Softmax,
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
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    CmpLt,
    Lt,
    Eq,
    Neq,
    Gt,
    Gte,
    Lte,
    And,
    Or,
    Not,
    Where,
    GuardedFail,
    MaxElem,
    Neg,
    Exp,
    Log,
    Sin,
    Sqrt,
    Cos,
    Tan,
    Atan,
    Tanh,
    Abs,
    Floor,
    Ceil,
    Round,
    Recip,
    UniformLike,
    Dropout,
    DropoutReplay,
    UniformBoundAdjoint,
    KeyFromSeed,
    SplitKey,
    SplitKeys,
    FoldIn,
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
        Self::Softmax,
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
        Self::BitAnd,
        Self::BitOr,
        Self::BitXor,
        Self::ShiftLeft,
        Self::ShiftRight,
        Self::CmpLt,
        Self::Lt,
        Self::Eq,
        Self::Neq,
        Self::Gt,
        Self::Gte,
        Self::Lte,
        Self::And,
        Self::Or,
        Self::Not,
        Self::Where,
        Self::GuardedFail,
        Self::MaxElem,
        Self::Neg,
        Self::Exp,
        Self::Log,
        Self::Sin,
        Self::Sqrt,
        Self::Cos,
        Self::Tan,
        Self::Atan,
        Self::Tanh,
        Self::Abs,
        Self::Floor,
        Self::Ceil,
        Self::Round,
        Self::Recip,
        Self::UniformLike,
        Self::Dropout,
        Self::DropoutReplay,
        Self::UniformBoundAdjoint,
        Self::KeyFromSeed,
        Self::SplitKey,
        Self::SplitKeys,
        Self::FoldIn,
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
            Self::Softmax => "softmax",
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
            Self::BitAnd => "bitand",
            Self::BitOr => "bitor",
            Self::BitXor => "bitxor",
            Self::ShiftLeft => "shl",
            Self::ShiftRight => "shr",
            Self::CmpLt => "cmplt",
            Self::Lt => "lt",
            Self::Eq => "eq",
            Self::Neq => "neq",
            Self::Gt => "gt",
            Self::Gte => "gte",
            Self::Lte => "lte",
            Self::And => "and",
            Self::Or => "or",
            Self::Not => "not",
            Self::Where => "where",
            Self::GuardedFail => "guarded_fail",
            Self::MaxElem => "max_elem",
            Self::Neg => "neg",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Sqrt => "sqrt",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Atan => "atan",
            Self::Tanh => "tanh",
            Self::Abs => "abs",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Recip => "recip",
            Self::UniformLike => "uniform_like",
            Self::Dropout => "dropout",
            Self::DropoutReplay => "DropoutReplay",
            Self::UniformBoundAdjoint => "UniformBoundAdjoint",
            Self::KeyFromSeed => "key_from_seed",
            Self::SplitKey => "split_key",
            Self::SplitKeys => "split_keys",
            Self::FoldIn => "fold_in",
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
    /// The exact number of inputs a key-operand random primitive, its replay
    /// or bound adjoint, a key operation or a join reads; `None` for any
    /// other operation. None of them reads an activation from an input: a
    /// draw's and a key operation's activation is its own
    /// ([`Owner::activation`]), and a join's two slot activations are its
    /// last two inputs. So a stale trailing activation operand is an arity
    /// error (spec/10 §3.2).
    pub fn key_operand_arity(&self) -> Option<usize> {
        match self {
            Self::KeyFromSeed | Self::Split { .. } => Some(1),
            Self::FoldIn => Some(2),
            // A runtime count names the input it reads, by value or by axis.
            Self::SplitN {
                count: RtDim::Node(slot) | RtDim::InputAxis { tensor: slot, .. },
            } => Some(slot + 1),
            Self::SplitN { .. } => Some(1),
            Self::Dropout | Self::DropoutReplay | Self::UniformBoundAdjoint { .. } => Some(3),
            Self::UniformLike | Self::KeySelect => Some(4),
            _ => None,
        }
    }

    /// The batch layout of a key-operand random primitive, or `None` for any
    /// other operation.
    pub fn draw_batch_layout(&self) -> Option<DrawBatchLayout> {
        let (op, key, data_input, per_row): (_, _, _, &'static [usize]) = match self {
            Self::Dropout | Self::DropoutReplay => ("dropout", 2, 0, &[1]),
            Self::UniformLike => ("uniform_like", 3, 0, &[1, 2]),
            Self::UniformBoundAdjoint { .. } => ("uniform_like", 2, 1, &[]),
            _ => return None,
        };
        Some(DrawBatchLayout {
            op,
            key,
            data_input,
            per_row,
        })
    }

    /// chelis#2368 / [05-OP-68]: operations that must execute because of
    /// what they DO, not because something consumes their result.
    ///
    /// Every liveness computation in the compiler derives "live" from value
    /// reachability, and an abort has no consumer by design, so each one
    /// needs this seed. `Store` is deliberately NOT here: it carries its own
    /// `implicit_observations` gating, because a projected execution slice
    /// legitimately excludes an unrelated store.
    pub const fn is_unconditional_effect(&self) -> bool {
        matches!(self, Self::GuardedFail { .. })
    }

    pub const fn atom_disposition(&self) -> RiscAtomDisposition {
        use RiscAtomDisposition::{Semantic, Structural};
        use RiscAtomIdentity as Id;

        match self {
            Self::ReluAdjoint { .. } => Semantic(Id::ReluAdjoint),
            Self::Relu => Semantic(Id::Relu),
            Self::Softmax { .. } => Semantic(Id::Softmax),
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
            Self::Bitwise(kind) => Semantic(match kind {
                chelis_types::BitwiseKind::And => Id::BitAnd,
                chelis_types::BitwiseKind::Or => Id::BitOr,
                chelis_types::BitwiseKind::Xor => Id::BitXor,
                chelis_types::BitwiseKind::ShiftLeft => Id::ShiftLeft,
                chelis_types::BitwiseKind::ShiftRight => Id::ShiftRight,
            }),
            Self::Compare(kind) => Semantic(match kind {
                ComparisonKind::CmpLt => Id::CmpLt,
                ComparisonKind::Lt => Id::Lt,
                ComparisonKind::Eq => Id::Eq,
                ComparisonKind::Neq => Id::Neq,
                ComparisonKind::Gt => Id::Gt,
                ComparisonKind::Gte => Id::Gte,
                ComparisonKind::Lte => Id::Lte,
            }),
            Self::Logical(kind) => Semantic(match kind {
                LogicalKind::And => Id::And,
                LogicalKind::Or => Id::Or,
                LogicalKind::Not => Id::Not,
            }),
            Self::Where => Semantic(Id::Where),
            Self::GuardedFail { .. } => Semantic(Id::GuardedFail),
            Self::MaxElem => Semantic(Id::MaxElem),
            Self::Neg => Semantic(Id::Neg),
            Self::Exp => Semantic(Id::Exp),
            Self::Log => Semantic(Id::Log),
            Self::Sin => Semantic(Id::Sin),
            Self::Sqrt => Semantic(Id::Sqrt),
            Self::Cos => Semantic(Id::Cos),
            Self::Tan => Semantic(Id::Tan),
            Self::Atan => Semantic(Id::Atan),
            Self::Tanh => Semantic(Id::Tanh),
            Self::Abs => Semantic(Id::Abs),
            Self::Floor => Semantic(Id::Floor),
            Self::Ceil => Semantic(Id::Ceil),
            Self::Round => Semantic(Id::Round),
            Self::Recip => Semantic(Id::Recip),
            Self::UniformLike => Semantic(Id::UniformLike),
            Self::Dropout => Semantic(Id::Dropout),
            Self::DropoutReplay => Semantic(Id::DropoutReplay),
            Self::UniformBoundAdjoint { .. } => Semantic(Id::UniformBoundAdjoint),
            Self::KeyFromSeed => Semantic(Id::KeyFromSeed),
            // Both halves are one identity: [05-OP-70] returns the pair.
            Self::Split { .. } => Semantic(Id::SplitKey),
            Self::SplitN { .. } => Semantic(Id::SplitKeys),
            Self::FoldIn => Semantic(Id::FoldIn),
            // The join is how a runtime `if` over keys is represented, not a
            // callable Table-A operation: it selects one of two existing keys.
            Self::KeySelect => Structural,
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
            Self::NamedCast { mode, .. } => Semantic(match mode {
                NamedCastMode::Trunc => Id::CastTrunc,
            }),
            Self::BlasMatmul { .. } => Semantic(Id::Matmul),
            Self::Gather { .. } => Semantic(Id::Gather),
            Self::ScatterAdd { .. } => Semantic(Id::Scatter),
            Self::Scatter { .. } => Semantic(Id::ScatterReplace),
            Self::ScatterElements { .. } => Semantic(Id::ScatterElements),
            // Extent witnesses and checks carry the compiler's operation
            // preconditions under [04-NUM-9], not callable Table-A operations.
            // Its tagged requirements and shape-only dependency are checked
            // by the IR verifier and the runtime-extent oracle.
            Self::ListMapCapture { .. }
            | Self::OrderedAdjointSum { .. }
            | Self::Iota
            | Self::ExtentWitness { .. }
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
            Prim::Bool | Prim::String | Prim::Key => {
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
            // precondition), and direct comparisons (the branch predicates that
            // drive branch-and-bound on piecewise definitions such as
            // the `erf64` sign/small-x folds) all have sound interval /
            // linear-relaxation transformers (beacon_plan.md §3.1, §3.3).
            RiscOp::Add
            | RiscOp::Sub
            | RiscOp::Mul
            | RiscOp::Div
            | RiscOp::Compare(_)
            | RiscOp::MaxElem
            | RiscOp::MinElem => true,

            // --- Guarded abort ---
            // chelis#1464: excluded. An envelope transformer would have to
            // represent "this path aborts", which is a control effect rather
            // than an output range, and a relaxation that simply passed the
            // fallback's envelope through would silently drop the abort — the
            // exact substitution [05-OP-68] exists to prevent.
            RiscOp::GuardedFail { .. } => false,

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
            | RiscOp::Tanh
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
            // Logical/selection nodes are discrete control operators. Beacon
            // targets the numeric regions around them, not these nodes.
            RiscOp::Logical(_) | RiscOp::Where => false,

            // Stochastic ops have no deterministic value to bound.
            RiscOp::UniformLike
            | RiscOp::Dropout
            | RiscOp::DropoutReplay
            | RiscOp::UniformBoundAdjoint { .. } => false,

            // Key derivations produce opaque keys, not a numeric envelope.
            RiscOp::KeyFromSeed
            | RiscOp::Split { .. }
            | RiscOp::FoldIn
            | RiscOp::SplitN { .. }
            | RiscOp::KeySelect => false,

            // Argmax/argmin return discrete indices, not a numeric
            // envelope over the reals; outside the forward-bound story.
            RiscOp::Argmax { .. } | RiscOp::Argmin { .. } | RiscOp::Count { .. } => false,

            // Floor/truncating integer division (chelis#178) are
            // piecewise-constant, non-differentiable rounding ops; like
            // `Floor`/`Ceil`/`Round` they have a step-function envelope,
            // but the integer-quotient semantics are not part of the
            // pinned real-valued forward-bound surface today.
            RiscOp::FloorDiv | RiscOp::TruncDiv | RiscOp::Mod | RiscOp::Bitwise(_) => false,

            // [05-OP-6] `cast_trunc` is the same shape as the integer
            // quotients above: piecewise constant with an integer output,
            // so it has no real-valued envelope to bound. The CHECKED
            // `cast` stays targetable because its float-to-float leg is
            // real-valued and its integer leg only admits exact values.
            RiscOp::NamedCast { .. } => false,

            // `OneHot` produces a discrete 0/1 indicator from an integer
            // index; it is an internal lowering marker (dag.rs) consumed
            // before backend emission and is not part of the numeric
            // forward-bound surface.
            RiscOp::OneHot { .. }
            | RiscOp::Iota
            | RiscOp::ListMapCapture { .. }
            | RiscOp::OrderedAdjointSum { .. } => false,

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

            // [05-OP-48] remains a retained composition identity. Beacon
            // has no dedicated transformer for an undecomposed Softmax.
            RiscOp::Softmax { .. } => false,

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
/// i8 < i16 < i32 < i64. Bool is 0; non-numeric returns 0.
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
        Prim::String | Prim::Key => 0,
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
    /// Producer-owned declared-result obligations.
    ///
    /// These are execution dependencies, but they are neither value inputs nor
    /// shape sources. Keeping them in a distinct lane prevents ownership and
    /// copy insertion from treating a result check as tensor fanout. The
    /// referenced witness still executes before this node and remains live
    /// until this producer discharges the obligation.
    #[serde(default)]
    pub result_claim_deps: Vec<NodeId>,
    /// The node's owner: its declaration and its activation ([`Owner`]).
    /// Required: lowering supplies the declaration it is lowering and the
    /// path activation of the position it lowers, a rebuilding pass the
    /// source node's owner carried through its node map, and a node a pass
    /// synthesizes the owner of the node it derives from.
    pub owner: Owner,
}

impl DagNode {
    /// Every node this node reads, in every dependency lane: its value
    /// inputs, its shape-only and result-claim dependencies, and its
    /// activation ([`Owner::activation`]). A liveness walk or a partition
    /// that follows these keeps everything the node needs to run.
    pub fn dependencies(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.inputs
            .iter()
            .chain(&self.shape_deps)
            .chain(&self.result_claim_deps)
            .copied()
            .chain(self.owner.activation)
    }

    /// The value operand `slot` takes where this node's activation is false
    /// ([`Owner::activation`]): one no check of the operation rejects, so the
    /// node computes a value and reports nothing (spec/10 section 3.2).
    /// `None` for an operation that checks nothing of its operands' values
    /// ([`RuntimeCheck::OperandValues`], [`RuntimeCheck::MeanDivisor`] and
    /// [`RuntimeCheck::Abort`] are the ones that do). The evaluator and the C
    /// lane substitute exactly these values.
    pub fn inactive_operand(&self, slot: usize) -> Option<i64> {
        match self.runtime_check() {
            RuntimeCheck::OperandValues | RuntimeCheck::MeanDivisor => Some(match &self.op {
                // A zero divisor, an integer `MIN / -1`, and an empty
                // `mean`'s count: zero divided by one rejects none of them.
                RiscOp::Div | RiscOp::FloorDiv | RiscOp::TruncDiv | RiscOp::Mod => {
                    i64::from(slot == 1)
                }
                // Zero converts to every dtype, no sum or product of zeros
                // overflows, and zero is in every integer range.
                _ => 0,
            }),
            // The condition takes the value that does not fire. The fallback
            // is checked by nothing, so it is read unchanged.
            RuntimeCheck::Abort => match &self.op {
                RiscOp::GuardedFail { trap_on_true, .. } if slot == 0 => {
                    Some(i64::from(!trap_on_true))
                }
                _ => None,
            },
            RuntimeCheck::Nothing
            | RuntimeCheck::EmptyAxis
            | RuntimeCheck::MovementBounds
            | RuntimeCheck::ExtentClaims
            | RuntimeCheck::Random
            | RuntimeCheck::SparseIndex
            | RuntimeCheck::Ungated => None,
        }
    }

    /// What this node checks at run time ([`RuntimeCheck`]): the one
    /// exhaustive declaration, with no wildcard arm, from which the trap
    /// seed ([`TrapSeeds::is_observable_root`]) and the false-activation behaviour
    /// ([`Self::inactive_operand`], [`TrapSeeds::is_activation_gated`]) are both
    /// read. A new operation does not compile until it states which class
    /// it is in.
    ///
    /// Inventory (the evaluator's `eval.rs` and the C emitter), by class:
    ///
    /// | class | operations | what traps |
    /// |---|---|---|
    /// | `OperandValues` | integer `Add` `Sub` `Mul` `Neg` `Abs` | overflow |
    /// | | `FloorDiv` `TruncDiv` `Mod`, integer `Div` | division by zero, `MIN / -1` |
    /// | | `Cast` `NamedCast` into an integer or bool width | domain, overflow |
    /// | | integer `Sum` `ProdReduce`, integer `ReduceWindow` | overflow |
    /// | | integer `FusedElem` | its steps' overflow and division |
    /// | `MeanDivisor` | float `Div` | a lowered `mean`'s empty count |
    /// | `EmptyAxis` | `MaxReduce` `MinReduce` `Argmax` `Argmin` | an empty reduced axis |
    /// | `MovementBounds` | `Shrink` `Stride` `Pad` | a runtime bound out of domain |
    /// | `ExtentClaims` | `ExtentWitness` (checking sites), `CheckedReshapeExtent` | a claimed extent |
    /// | `Random` | `Dropout` `DropoutReplay` `UniformLike` `UniformBoundAdjoint` `SplitN` `FoldIn` `KeySelect` | controls, key extents, a negative count |
    /// | `Abort` | `GuardedFail` | its authored condition |
    /// | `Ungated` | `Reshape` `Expand` | a runtime target extent |
    /// | `SparseIndex` | `Gather` `ScatterAdd` `Scatter` `ScatterElements` `OneHot` | an index out of range |
    /// | `Nothing` | every other operation, and float arithmetic and reductions | |
    ///
    /// Three checks are not an operation kind's and are listed here for
    /// completeness. A node whose declared extent rests on a claim checked
    /// under its activation ([`TrapSeeds::is_claim_sized`]) is gated whatever
    /// its class, and where its activation is false it produces zeros of its
    /// declared type. Every same-shape producer's operand agreement (the
    /// evaluator's "tensor shapes must match" and the C lane's
    /// `emit_elementwise_operand_guard`) is a memory-safety precondition of
    /// the kernel, not a gated check: a false activation leaves it in place,
    /// except at a claim-sized node, which then reads no operand. A result's
    /// element count and byte size are admitted at every allocation
    /// ([05-OP-33]) whatever the activation.
    pub fn runtime_check(&self) -> RuntimeCheck {
        let integer = self.output_type.precision.is_integer();
        let value_check = |checks: bool| {
            if checks {
                RuntimeCheck::OperandValues
            } else {
                RuntimeCheck::Nothing
            }
        };
        match &self.op {
            RiscOp::Iota => RuntimeCheck::OperandValues,
            RiscOp::ListMapCapture { .. } | RiscOp::OrderedAdjointSum { .. } => {
                RuntimeCheck::Ungated
            }
            RiscOp::Add | RiscOp::Sub | RiscOp::Mul | RiscOp::Neg | RiscOp::Abs => {
                value_check(integer)
            }
            RiscOp::FloorDiv | RiscOp::TruncDiv | RiscOp::Mod => value_check(integer),
            RiscOp::Bitwise(kind) => value_check(kind.is_shift()),
            // Float-only since chelis#178; its one float check is a lowered
            // `mean`'s count, which the node alone cannot tell apart.
            RiscOp::Div if integer => RuntimeCheck::OperandValues,
            RiscOp::Div => RuntimeCheck::MeanDivisor,
            // Read the cast's OWN target, not the node's output type: if a
            // lowering ever let them drift, deriving the class from the
            // output type would silently switch the check off.
            RiscOp::Cast { new_precision } | RiscOp::NamedCast { new_precision, .. } => {
                value_check(new_precision.is_integer() || *new_precision == Prim::Bool)
            }
            RiscOp::Sum { .. } | RiscOp::ProdReduce { .. } => value_check(integer),
            // An integer window sum overflows; a window is never empty, and
            // max and min select without arithmetic.
            RiscOp::ReduceWindow { reducer, .. } => value_check(
                integer && matches!(reducer, ReduceWindowKind::Sum | ReduceWindowKind::Mean),
            ),
            RiscOp::FusedElem { .. } => value_check(integer),
            RiscOp::Softmax { .. }
            | RiscOp::MaxReduce { .. }
            | RiscOp::MinReduce { .. }
            | RiscOp::Argmax { .. }
            | RiscOp::Argmin { .. } => RuntimeCheck::EmptyAxis,
            RiscOp::Shrink { .. } | RiscOp::Stride { .. } | RiscOp::Pad { .. } => {
                RuntimeCheck::MovementBounds
            }
            RiscOp::ExtentWitness {
                site: ExtentWitnessSite::LiteralResultClaim,
                ..
            } => RuntimeCheck::Nothing,
            RiscOp::ExtentWitness {
                site: ExtentWitnessSite::LocalAscriptionClaim { .. },
                requirements,
                ..
            } if !requirements.is_empty() => RuntimeCheck::Nothing,
            RiscOp::ExtentWitness { .. } | RiscOp::CheckedReshapeExtent { .. } => {
                RuntimeCheck::ExtentClaims
            }
            RiscOp::Dropout
            | RiscOp::DropoutReplay
            | RiscOp::UniformLike
            | RiscOp::UniformBoundAdjoint { .. }
            | RiscOp::SplitN { .. }
            | RiscOp::FoldIn
            | RiscOp::KeySelect => RuntimeCheck::Random,
            RiscOp::GuardedFail { .. } => RuntimeCheck::Abort,
            // [05-OP-52]: "out-of-bounds indices fail loudly". The index
            // is data, so no static fact rules the failure out. The checker
            // rejects an out-of-range LITERAL index only where the base is
            // also a literal, so this class does not try to prove a literal
            // index safe: it may fail is the conservative answer.
            RiscOp::Gather { .. }
            | RiscOp::ScatterAdd { .. }
            | RiscOp::Scatter { .. }
            | RiscOp::ScatterElements { .. }
            | RiscOp::OneHot { .. } => RuntimeCheck::SparseIndex,
            RiscOp::Reshape { .. } | RiscOp::Expand { .. } => RuntimeCheck::Ungated,
            RiscOp::Compare(_)
            | RiscOp::Logical(_)
            | RiscOp::Where
            | RiscOp::MaxElem
            | RiscOp::MinElem
            | RiscOp::ExtremaAdjoint { .. }
            | RiscOp::Relu
            | RiscOp::ReluAdjoint
            | RiscOp::Exp
            | RiscOp::Log
            | RiscOp::Sin
            | RiscOp::Sqrt
            | RiscOp::Cos
            | RiscOp::Tan
            | RiscOp::Atan
            | RiscOp::Tanh
            | RiscOp::Floor
            | RiscOp::Ceil
            | RiscOp::Round
            | RiscOp::Recip
            | RiscOp::KeyFromSeed
            | RiscOp::Split { .. }
            | RiscOp::Count { .. }
            | RiscOp::ReduceWindowGrad { .. }
            | RiscOp::Permute { .. }
            | RiscOp::Shape { .. }
            | RiscOp::CheckedUnitAxis { .. }
            | RiscOp::Const { .. }
            | RiscOp::ConstTensor { .. }
            | RiscOp::Load { .. }
            | RiscOp::Store { .. }
            | RiscOp::Copy
            | RiscOp::Drop
            | RiscOp::Realize
            | RiscOp::BlasMatmul { .. } => RuntimeCheck::Nothing,
        }
    }
}

/// What an operation checks at run time (the classes of
/// [`DagNode::runtime_check`]'s inventory), and so what a node of it does
/// where its activation is false (spec/10 section 3.2: it is computed, since
/// a `Where` may read its value, and checks nothing) and whether it is a
/// trap seed ([`TrapSeeds::is_observable_root`], spec/06 section 5.2).
///
/// Under a per-row activation (a `vmap`ped `if`) a check of an operand's
/// VALUES decides row by row, and a check of an EXTENT decides for every
/// row at once, since the rows of one tensor share their extents: it runs
/// when the activation holds in some row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeCheck {
    /// Checks nothing a well-typed operand can fail. Never a seed.
    Nothing,
    /// Checks its operands' element values: integer overflow, integer
    /// division, a cast's range. Where the activation is false each operand
    /// slot reads [`DagNode::inactive_operand`], a value no check rejects.
    /// A seed.
    OperandValues,
    /// A float `Div`, which checks a lowered `mean`'s count for zero and is
    /// gated like [`Self::OperandValues`]. Not a seed: the node alone does
    /// not say it is a `mean`.
    MeanDivisor,
    /// Checks that its reduced axis is not empty. Where the activation is
    /// false an empty axis reduces to zeros. A seed unless its operand's
    /// reduced axis is a nonzero literal.
    EmptyAxis,
    /// Checks its runtime bounds (a `shrink` range, a `stride` step, a `pad`
    /// width) against its operand's extents. Where the activation is false
    /// it reads no bound and produces zeros of its declared type, each axis
    /// it declares itself taking its operand's extent. A seed unless every
    /// bound is statically in range ([`Dag::movement_bounds_may_fail`]).
    MovementBounds,
    /// Compares an extent a contract claims (a call's, a result's, a local
    /// ascription's, a reshape target's). Where the activation is false it
    /// compares nothing; its value is unchanged. A seed.
    ExtentClaims,
    /// A draw or key operation, gated by its owner's activation in its own
    /// emitter (it draws or validates nothing where it is false) and
    /// seeded by the per-guard rule ([`Dag::random_node_may_trap`]).
    Random,
    /// An authored abort ([05-OP-68]): gated like [`Self::OperandValues`],
    /// its condition reading the value that does not fire. Always a seed.
    Abort,
    /// Checks a sparse index against its base axis: gather and every
    /// scatter mode ([05-OP-52]), and the internal `one_hot` marker
    /// ([05-SPARSE-2]). The index is data, so the check can always fail.
    ///
    /// A seed only where the node has no activation (chelis#2440). It is
    /// the only SEEDED class the lanes do not gate -- [`Self::Ungated`] is
    /// ungated too, but never seeds -- so its emitters read no activation
    /// and a node of it checks even where its activation is false. Seeding one that has an activation would
    /// therefore trap where spec/06 5.2 says it "checks nothing", so an
    /// activated sparse node is left to ordinary value reachability. Gating
    /// the class needs an inactive-value contract in the evaluator and the
    /// C emitter, which chelis#2440 tracks.
    SparseIndex,
    /// Can trap, and neither checks nothing under a false activation nor
    /// is a seed (chelis#2440's remaining kinds).
    Ungated,
}

/// The trap seed ([`Self::is_observable_root`]), the check-may-fail fact it
/// and the activation gate share ([`Self::check_may_fail`]), and the gate
/// itself ([`Self::is_activation_gated`]), over one graph
/// ([`Dag::trap_seeds`]).
///
/// A literal result claim observed at a call's parameter witness makes that
/// witness a check ([`Self::literal_result_witness_requirements`]), and
/// which witness observes a claim is a whole-graph derivation. The queries
/// therefore live on this value rather than on [`Dag`]: a pass takes one
/// before it walks the nodes and the derivation runs at most once, where a
/// per-node query on the graph repeated it for every witness, quadratic in
/// the graph in dead-code elimination, the evaluator's seeds and the
/// verifier.
pub struct TrapSeeds<'dag> {
    dag: &'dag Dag,
    literal_result_witness_requirements:
        std::cell::OnceCell<std::collections::BTreeMap<NodeId, Vec<chelis_types::ScalarValue>>>,
    claim_sized_nodes: std::cell::OnceCell<Result<std::collections::BTreeSet<NodeId>, String>>,
}

impl TrapSeeds<'_> {
    /// Whether `node` is an observable root (`spec/06-transformations.md`
    /// §5.2): it must execute because of what it does, not because a value
    /// reaches it. "Potentially effectful or trapping nodes are observable
    /// roots; purity alone does not make a possible trap dead."
    ///
    /// The members: an unconditional effect (chelis#2368, [05-OP-68]); a
    /// numeric node that can trap (chelis#2440); an UNACTIVATED sparse
    /// operation whose index the base axis can reject
    /// ([`RuntimeCheck::SparseIndex`], chelis#2440); and a random node that can
    /// trap by itself ([`Dag::random_node_may_trap`], chelis#2413). A
    /// backward-synthesized adjoint is not a numeric member: its trap
    /// obligation belongs to the forward node it was derived from, and it is
    /// scaffolding for a gradient that may not be requested (seeding one
    /// resurrects integer adjoint machinery that fails verification as
    /// non-differentiable; `issue_1306_direct_arithmetic` pins it).
    ///
    /// This is the one seed predicate. The evaluator, dead-code elimination,
    /// `grad`'s pruner, the verifier's dangling rule and the host transform
    /// runner all read it; the evaluator and dead-code elimination then keep
    /// only the seeds whose declaration the evaluation enters
    /// ([`Dag::outside_selection`]), and a node whose activation is false
    /// checks nothing when it runs.
    pub fn is_observable_root(&self, node: &DagNode) -> bool {
        let synthesized_adjoint = node.span_id.as_deref() == Some(crate::grad::GRAD_SYNTH_MARKER);
        match node.runtime_check() {
            RuntimeCheck::Abort => true,
            RuntimeCheck::OperandValues
            | RuntimeCheck::EmptyAxis
            | RuntimeCheck::MovementBounds
            | RuntimeCheck::ExtentClaims => !synthesized_adjoint && self.check_may_fail(node),
            RuntimeCheck::Random => self.check_may_fail(node),
            // Unlike every other seeded class, this one has no activation
            // gate in any lane, so a seeded node under a false activation
            // would check and trap where 5.2 says it "checks nothing". Seed
            // only an unconditional one until the class is gated
            // (chelis#2440); a discarded sparse node under an activation is
            // therefore still eliminated, trap and all.
            RuntimeCheck::SparseIndex => {
                !synthesized_adjoint && node.owner.activation.is_none() && self.check_may_fail(node)
            }
            RuntimeCheck::Nothing | RuntimeCheck::MeanDivisor | RuntimeCheck::Ungated => false,
        }
    }

    /// Whether `node`'s run-time check ([`DagNode::runtime_check`]) can
    /// fail for some input: its class checks something, and no static fact
    /// rules the failure out. The facts are per class: a reduced axis of
    /// nonzero literal extent is not empty
    /// ([`Dag::reduced_axis_may_be_empty`]), movement bounds statically in
    /// range are in range ([`Dag::movement_bounds_may_fail`]), and a random
    /// node's literal in-range controls pass ([`Dag::random_node_may_trap`]).
    /// The trap seed ([`Self::is_observable_root`]) and the activation gate
    /// ([`Self::is_activation_gated`]) both read it.
    pub fn check_may_fail(&self, node: &DagNode) -> bool {
        match node.runtime_check() {
            RuntimeCheck::Nothing => false,
            RuntimeCheck::OperandValues
            | RuntimeCheck::MeanDivisor
            | RuntimeCheck::Abort
            | RuntimeCheck::SparseIndex
            | RuntimeCheck::Ungated => true,
            RuntimeCheck::ExtentClaims => self.extent_claims_may_fail(node),
            RuntimeCheck::EmptyAxis => self.dag.reduced_axis_may_be_empty(node),
            RuntimeCheck::MovementBounds => self.dag.movement_bounds_may_fail(node),
            RuntimeCheck::Random => self.dag.random_node_may_trap(node),
        }
    }

    /// Whether an [`RuntimeCheck::ExtentClaims`] node compares anything. A
    /// `CheckedReshapeExtent` always carries a claim. An `ExtentWitness`
    /// compares its axis against its literal requirements, its named claims
    /// and the literal result claims observed at it
    /// ([`Self::literal_result_witness_requirements`], the evaluator's and
    /// the C lane's full list); a witness with none of the three only
    /// reports the extent it reads, which no input can fail. Lowering places
    /// such a witness at every call entry, so seeding it would keep a
    /// parameter's `Load` that nothing else reads and make that parameter a
    /// required input.
    fn extent_claims_may_fail(&self, node: &DagNode) -> bool {
        match &node.op {
            RiscOp::ExtentWitness {
                requirements,
                claims,
                ..
            } => {
                !requirements.is_empty()
                    || !claims.is_empty()
                    || !self.literal_result_witness_requirements(node.id).is_empty()
            }
            _ => true,
        }
    }

    /// The literal result claims checked at `witness`
    /// ([`crate::axis_sources::literal_result_witness_requirements`]), in
    /// claim order. The whole graph's are derived on the first call and
    /// shared by every later one.
    pub fn literal_result_witness_requirements(
        &self,
        witness: NodeId,
    ) -> &[chelis_types::ScalarValue] {
        self.literal_result_witness_requirements
            .get_or_init(|| crate::axis_sources::literal_result_witness_requirements(self.dag))
            .get(&witness)
            .map_or(&[], Vec::as_slice)
    }

    /// Whether `node` checks nothing where its activation is false (spec/10
    /// section 3.2): it has an activation, and either its declared extent
    /// rests on a claim checked under it ([`Self::is_claim_sized`]), or its
    /// check can fail
    /// ([`Self::check_may_fail`]; a node whose check no input fails needs
    /// no gate and computes as usual) and its class is one the lanes gate,
    /// by one of the mechanisms [`RuntimeCheck`] names: every class but
    /// [`RuntimeCheck::Nothing`], [`RuntimeCheck::Random`], whose draw and
    /// key-operation emitters read the owner's activation themselves, and
    /// [`RuntimeCheck::Ungated`].
    ///
    /// The one gate declaration every lane reads: the evaluator and the C
    /// emitter gate exactly these nodes, fusion keeps each in its own
    /// kernel, and the HIP emitter refuses one whose kernel does not take
    /// the gate.
    pub fn is_activation_gated(&self, node: &DagNode) -> bool {
        node.owner.activation.is_some()
            && ((self.check_may_fail(node)
                && match node.runtime_check() {
                    RuntimeCheck::OperandValues
                    | RuntimeCheck::MeanDivisor
                    | RuntimeCheck::EmptyAxis
                    | RuntimeCheck::MovementBounds
                    | RuntimeCheck::ExtentClaims
                    | RuntimeCheck::Abort => true,
                    RuntimeCheck::Nothing
                    | RuntimeCheck::Random
                    | RuntimeCheck::SparseIndex
                    | RuntimeCheck::Ungated => false,
                })
                || self.is_claim_sized(node))
    }

    /// Whether `node`'s declared extent rests on a claim checked under its
    /// activation ([`crate::axis_sources::claim_sized_nodes`]: a result
    /// claim, a local ascription, an extent an operation computes or a
    /// carrier sets, a restamp (chelis#2512), a unit claim, a
    /// [`RiscOp::CheckedUnitAxis`]), whatever its operation's class. Where
    /// its activation holds in no row the claim is not checked, so it checks
    /// nothing and produces zeros of its declared type, each claimed axis
    /// taking the claim's extent, as [`RuntimeCheck::MovementBounds`] does:
    /// its operand's extent need not be the one it declares there, so it
    /// reads no operand, and its operands' agreement, which only its reads
    /// need, is not checked either. Where some row is active it computes as
    /// usual.
    ///
    /// A whole-graph derivation, run on the first call and shared by every
    /// later one; [`Self::is_activation_gated`] asks it only of a node with
    /// an activation. A graph whose guard sites cannot be derived has every
    /// such node gated: each lane refuses that graph when it derives the
    /// sites itself, and until then gating keeps it out of fusion.
    pub fn is_claim_sized(&self, node: &DagNode) -> bool {
        match self
            .claim_sized_nodes
            .get_or_init(|| crate::axis_sources::claim_sized_nodes(self.dag))
        {
            Ok(nodes) => nodes.contains(&node.id),
            Err(_) => true,
        }
    }
}

/// The RISC DAG — an append-only, topologically-ordered vector of [`DagNode`]s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dag {
    nodes: Vec<DagNode>,
    roots: Vec<NodeId>,
    /// The declarations this graph's nodes belong to ([`DagNode::decl`]).
    declarations: Vec<Declaration>,
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
    ///
    /// `owner` is required for the same reason: every node belongs to a
    /// declaration this graph registered ([`Self::declare`]) and runs under
    /// an activation, an earlier node of this graph or none. A bare
    /// [`DeclId`] is an unconditional owner.
    pub fn add_node(
        &mut self,
        owner: impl Into<Owner>,
        op: RiscOp,
        inputs: Vec<NodeId>,
        output_type: TensorType,
        span_id: Option<String>,
    ) -> NodeId {
        // A uniform anonymous branch has no intrinsic extent source. Where's
        // same-shape contract supplies one from an earlier peer branch or the
        // condition, so retain that producer explicitly for verification,
        // eval, and codegen.
        let inferred_where_shape_deps = if matches!(op, RiscOp::Where) && inputs.len() == 3 {
            [
                (inputs[1], [inputs[2], inputs[0]]),
                (inputs[2], [inputs[1], inputs[0]]),
            ]
            .into_iter()
            .filter_map(|(target_id, sources)| {
                let target = self.get(target_id)?;
                if !matches!(target.op, RiscOp::Const { .. })
                    || !target.inputs.is_empty()
                    || !target.shape_deps.is_empty()
                    || !target.output_type.dims.iter().any(|dim| {
                        matches!(
                            dim,
                            DimInfo::Named(name, None) if name.is_empty() || name == "*"
                        )
                    })
                {
                    return None;
                }
                sources
                    .into_iter()
                    .find(|source_id| {
                        self.get(*source_id).is_some_and(|source| {
                            source.id.0 < target.id.0
                                && source.output_type.dims.len() == target.output_type.dims.len()
                        })
                    })
                    .map(|source| (target_id, source))
            })
            .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let owner = owner.into();
        // One fact, one carrier (spec/10 section 3.2): a draw's or key
        // operation's activation is its owner's and never an input, so a
        // pass that still appends one is a defect caught here, at the site
        // that built it.
        if let Some(arity) = op.key_operand_arity() {
            assert!(
                inputs.len() == arity,
                "a {op:?} node reads exactly {arity} inputs, not {}; its activation is its owner's",
                inputs.len()
            );
        }
        let decl = owner.decl;
        assert!(
            (decl.0 as usize) < self.declarations.len(),
            "node declaration {decl:?} is not registered in this graph ({} declarations)",
            self.declarations.len()
        );
        let id = NodeId(self.nodes.len());
        // The verifier also requires the activation to be a Bool; a graph
        // lowered without type checking may hold another scalar there.
        if let Some(activation) = owner.activation {
            assert!(
                activation.0 < id.0,
                "node {id:?}'s activation {activation:?} is not an earlier node of this graph"
            );
        }
        self.nodes.push(DagNode {
            id,
            op,
            inputs,
            output_type,
            reusable_input: None,
            span_id,
            merged_spans: Vec::new(),
            shape_deps: Vec::new(),
            result_claim_deps: Vec::new(),
            owner,
        });
        for (target, source) in inferred_where_shape_deps {
            self.add_shape_dep(target, source);
        }
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

    /// Attach one producer-owned declared-result obligation without making it
    /// an ordinary shape dependency or value consumer.
    pub fn add_result_claim_dep(&mut self, id: NodeId, dep: NodeId) {
        if let Some(node) = self.nodes.get_mut(id.0)
            && !node.result_claim_deps.contains(&dep)
        {
            node.result_claim_deps.push(dep);
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

    /// Strict splice/rebuild variant of [`Self::preserve_shape_deps`].
    ///
    /// A pass that promises a complete node correspondence must not inherit
    /// the permissive DCE behavior above: silently filtering one live
    /// shape-only edge can turn a rejected runtime extent into an undeclared
    /// rendered identifier. The destination is updated only after every
    /// source dependency has a valid remapped node.
    pub fn preserve_shape_deps_strict(
        &mut self,
        new_id: NodeId,
        source_deps: &[NodeId],
        remap: &UnordMap<NodeId, NodeId>,
    ) -> Result<(), String> {
        let mut mapped = Vec::with_capacity(source_deps.len());
        for old in source_deps {
            let Some(new) = remap.get(old).copied() else {
                return Err(format!(
                    "shape dependency {old:?} has no remapped node for {new_id:?}"
                ));
            };
            if self.get(new).is_none() {
                return Err(format!(
                    "shape dependency {old:?} maps to invalid node {new:?} for {new_id:?}"
                ));
            }
            mapped.push(new);
        }
        let Some(node) = self.nodes.get_mut(new_id.0) else {
            return Err(format!(
                "shape dependency owner {new_id:?} is not present in the rebuilt DAG"
            ));
        };
        node.shape_deps = mapped;
        Ok(())
    }

    /// Preserve producer-owned result claims across a complete rebuild.
    ///
    /// A result claim may never disappear because a rebuilding pass omitted
    /// one of its dependencies. Passes without a recoverable error channel
    /// fail loudly here; fallible composition uses
    /// [`Self::preserve_result_claim_deps_strict`] directly.
    pub fn preserve_result_claim_deps(
        &mut self,
        new_id: NodeId,
        source_deps: &[NodeId],
        remap: &UnordMap<NodeId, NodeId>,
    ) {
        self.preserve_result_claim_deps_strict(new_id, source_deps, remap)
            .unwrap_or_else(|message| panic!("{message}"));
    }

    /// Preserve every producer-owned result claim across a complete rebuild.
    pub fn preserve_result_claim_deps_strict(
        &mut self,
        new_id: NodeId,
        source_deps: &[NodeId],
        remap: &UnordMap<NodeId, NodeId>,
    ) -> Result<(), String> {
        let mut mapped = Vec::with_capacity(source_deps.len());
        for old in source_deps {
            let Some(new) = remap.get(old).copied() else {
                return Err(format!(
                    "result claim dependency {old:?} has no remapped node for {new_id:?}"
                ));
            };
            if self.get(new).is_none() {
                return Err(format!(
                    "result claim dependency {old:?} maps to invalid node {new:?} for {new_id:?}"
                ));
            }
            mapped.push(new);
        }
        let Some(node) = self.nodes.get_mut(new_id.0) else {
            return Err(format!(
                "result claim owner {new_id:?} is not present in the rebuilt DAG"
            ));
        };
        node.result_claim_deps = mapped;
        Ok(())
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

    /// The trap seed and activation-gate queries over this graph
    /// ([`TrapSeeds`]). A pass takes one and asks it about every node, so the
    /// whole-graph facts they read are derived once per pass.
    pub fn trap_seeds(&self) -> TrapSeeds<'_> {
        TrapSeeds {
            dag: self,
            literal_result_witness_requirements: std::cell::OnceCell::new(),
            claim_sized_nodes: std::cell::OnceCell::new(),
        }
    }

    /// Whether the reduced axis of an [`RuntimeCheck::EmptyAxis`] node can
    /// be empty at run time: anything but a nonzero literal extent.
    fn reduced_axis_may_be_empty(&self, node: &DagNode) -> bool {
        let axis = match &node.op {
            RiscOp::Softmax { axis }
            | RiscOp::MaxReduce { axis }
            | RiscOp::MinReduce { axis }
            | RiscOp::Argmax { axis }
            | RiscOp::Argmin { axis } => *axis,
            _ => return true,
        };
        !node
            .inputs
            .first()
            .and_then(|input| self.get(*input))
            .and_then(|input| input.output_type.dims.get(axis))
            .is_some_and(|dim| matches!(dim, DimInfo::Lit(extent) if *extent > 0))
    }

    /// Whether a [`RuntimeCheck::MovementBounds`] node's bounds can fail at
    /// run time. A bound read at run time (a node, another tensor's axis, a
    /// symbol) may; a literal bound is checked here against its operand's
    /// axis, and exempt only where that axis is a literal extent too and the
    /// bound is in range for every lane: a `shrink` range nonempty and
    /// inside the axis (the evaluator rejects an empty one), a `stride` step
    /// positive. A literal `pad` width cannot be negative.
    pub fn movement_bounds_may_fail(&self, node: &DagNode) -> bool {
        let Some(operand) = node.inputs.first().and_then(|input| self.get(*input)) else {
            return true;
        };
        let extent = |axis: usize| match operand.output_type.dims.get(axis) {
            Some(DimInfo::Lit(extent)) => Some(*extent),
            _ => None,
        };
        match &node.op {
            RiscOp::Shrink { bounds } => bounds.iter().enumerate().any(|(axis, (start, end))| {
                let end = match end {
                    RtDim::ToEnd => extent(axis),
                    end => end.as_lit(),
                };
                !matches!(
                    (start.as_lit(), end, extent(axis)),
                    (Some(start), Some(end), Some(extent)) if start < end && end <= extent
                )
            }),
            RiscOp::Stride { strides } => strides
                .iter()
                .any(|step| !step.as_lit().is_some_and(|step| step > 0)),
            RiscOp::Pad { padding, .. } => padding
                .iter()
                .any(|(before, after)| before.as_lit().is_none() || after.as_lit().is_none()),
            _ => true,
        }
    }

    /// chelis#2413: whether a random node can trap by itself, the random
    /// member of [`TrapSeeds::is_observable_root`].
    ///
    /// A draw validates its own controls and key batch ([05-OP-37]/[05-OP-8])
    /// and cannot trap only when every guard is statically satisfied:
    /// [`Self::draw_controls_are_literal_and_in_range`] and
    /// [`Self::draw_key_batch_is_literal`]. A `SplitN` traps on a negative
    /// runtime count ([05-OP-71]); a literal count cannot be negative. The
    /// other key operations are total.
    pub fn random_node_may_trap(&self, node: &DagNode) -> bool {
        match &node.op {
            RiscOp::Dropout | RiscOp::UniformLike => {
                !(self.draw_controls_are_literal_and_in_range(node)
                    && self.draw_key_batch_is_literal(node))
            }
            RiscOp::SplitN { count } => count.as_lit().is_none(),
            _ => false,
        }
    }

    /// Whether every control of the draw `node` (a dropout rate, a
    /// `uniform_like` bound pair) is a `Const` its own atom accepts at the
    /// draw's dtype: the same validation the draw performs before drawing,
    /// applied to the literal. A runtime control, or a literal out of range,
    /// may trap.
    fn draw_controls_are_literal_and_in_range(&self, node: &DagNode) -> bool {
        let literal = |slot: usize| {
            node.inputs
                .get(slot)
                .and_then(|input| self.get(*input))
                .and_then(|input| match &input.op {
                    RiscOp::Const { value } => Some(*value),
                    _ => None,
                })
        };
        let prim = node.output_type.precision;
        match &node.op {
            RiscOp::Dropout => literal(1).is_some_and(|rate| {
                chelis_types::dtype_semantics::DropoutParameters::new(prim, rate).is_ok()
            }),
            RiscOp::UniformLike => literal(1).zip(literal(2)).is_some_and(|(low, high)| {
                chelis_types::dtype_semantics::UniformLikeParameters::new(prim, low, high).is_ok()
            }),
            _ => false,
        }
    }

    /// Whether the draw `node`'s key batch statically indexes its operands:
    /// a rank-0 key, or a key whose dims are all literal and equal to the
    /// literal leading dims of its data, of every per-row operand (its
    /// controls) and of its activation, the static form of the evaluator's
    /// extent check. A symbolic extent on either side may disagree at run
    /// time.
    fn draw_key_batch_is_literal(&self, node: &DagNode) -> bool {
        let Some(layout) = node.op.draw_batch_layout() else {
            return false;
        };
        let dims_of = |slot: usize| {
            node.inputs
                .get(slot)
                .and_then(|input| self.get(*input))
                .map(|input| input.output_type.dims.as_slice())
        };
        let Some(key) = dims_of(layout.key) else {
            return false;
        };
        if key.is_empty() {
            return true;
        }
        let literal_prefix = |dims: &[DimInfo], axes: usize| {
            dims.len() >= axes
                && dims[..axes].iter().zip(key).all(
                    |(dim, key)| matches!((dim, key), (DimInfo::Lit(a), DimInfo::Lit(b)) if a == b),
                )
        };
        let Some(data) = dims_of(layout.data_input) else {
            return false;
        };
        let activation = node
            .owner
            .activation
            .and_then(|activation| self.get(activation))
            .map(|activation| activation.output_type.dims.as_slice());
        literal_prefix(data, key.len())
            && layout
                .per_row
                .iter()
                .map(|slot| dims_of(*slot))
                .chain(std::iter::once(activation))
                .all(|dims| match dims {
                    Some(dims) => dims.len() <= key.len() && literal_prefix(dims, dims.len()),
                    None => true,
                })
    }

    pub fn set_roots(&mut self, roots: Vec<NodeId>) {
        self.roots = roots;
    }

    pub fn is_root(&self, id: NodeId) -> bool {
        self.roots.contains(&id)
    }

    /// The declarations this graph's nodes belong to.
    pub fn declarations(&self) -> &[Declaration] {
        &self.declarations
    }

    /// The declaration `decl` names.
    pub fn declaration(&self, decl: DeclId) -> &Declaration {
        &self.declarations[decl.0 as usize]
    }

    /// Register a function declaration named `name`: its standalone nodes run
    /// only when a selected root belongs to it. A graph built outside program
    /// lowering registers one of these for its nodes.
    pub fn declare(&mut self, name: impl Into<String>) -> DeclId {
        self.push_declaration(name.into(), false)
    }

    /// Register a value declaration named `name`: its own nodes run only
    /// when a selected root belongs to it, and another declaration reads
    /// them only when none of them can trap.
    pub fn declare_value(&mut self, name: impl Into<String>) -> DeclId {
        self.push_declaration(name.into(), true)
    }

    fn push_declaration(&mut self, name: String, value: bool) -> DeclId {
        let id = u32::try_from(self.declarations.len())
            .expect("a graph has fewer than 2^32 declarations");
        self.declarations.push(Declaration { name, value });
        DeclId(id)
    }

    /// Take `source`'s declarations, for a pass that rebuilds `source` node
    /// by node and supplies each node's `decl`. Call it before the first
    /// node is added.
    pub fn inherit_declarations(&mut self, source: &Dag) {
        debug_assert!(
            self.nodes.is_empty(),
            "a rebuild inherits its source's declarations before adding nodes"
        );
        self.declarations.clone_from(&source.declarations);
    }

    /// A diagnostic name for `id`: its declaration, and for a `Load` the
    /// parameter it reads, as "parameter `x` of `f`"; otherwise "node N of
    /// `f`". An unnamed declaration is "the top-level expression".
    pub fn describe_node(&self, id: NodeId) -> String {
        let Some(node) = self.get(id) else {
            return format!("node {}", id.0);
        };
        let owner = self.describe_declaration(node.owner.decl);
        match &node.op {
            RiscOp::Load { name } => format!("parameter `{}` of {owner}", name.as_str()),
            _ => format!("node {} of {owner}", id.0),
        }
    }

    /// A diagnostic name for `decl`: "`f`", or "the top-level expression"
    /// for an unnamed one.
    pub fn describe_declaration(&self, decl: DeclId) -> String {
        match self.declarations.get(decl.0 as usize) {
            Some(declaration) if !declaration.name.is_empty() => {
                format!("`{}`", declaration.name)
            }
            Some(_) => "the top-level expression".to_owned(),
            None => format!("unregistered declaration {}", decl.0),
        }
    }

    /// The declarations an activation of the `selected` roots enters, as a
    /// mask over [`Self::declarations`]: each selected root's declaration.
    /// Another declaration's work runs only inlined into an entered one's
    /// own nodes ([`Declaration`]): a call runs the function's body in the
    /// caller, and a reference to a value whose initializer may trap runs
    /// that initializer where the reference is.
    pub fn entered_declarations(&self, selected: &[NodeId]) -> Vec<bool> {
        let mut entered = vec![false; self.declarations.len()];
        for root in selected {
            entered[self.nodes[root.0].owner.decl.0 as usize] = true;
        }
        entered
    }

    /// The nodes whose seeds (an abort, or a random node that can trap by
    /// itself) an evaluation of the `selected` roots does not run
    /// (chelis#2476, `spec/06-transformations.md` §5.2).
    ///
    /// Selecting roots is a scoping decision, not merely a request for
    /// certain outputs: a lowered program holds every declaration's
    /// activation, called or not, and a seed marks nodes the selected roots
    /// do not reach, which is the whole point of a seed. So a seed runs only
    /// when its node belongs to a declaration the selection enters
    /// ([`Self::entered_declarations`]), whether it sits in a root's value
    /// graph, in a discarded value's terminal, or in a library declaration's
    /// own body. With nothing selected nothing is out of scope. Reachability
    /// from the selection makes a node live whatever this says, so scoping
    /// only ever drops work no selected root needs.
    ///
    /// The evaluator's seeds and dead-code elimination's seeds both read
    /// this one predicate.
    pub fn outside_selection(&self, selected: &[NodeId]) -> Vec<bool> {
        if selected.is_empty() {
            return vec![false; self.len()];
        }
        let entered = self.entered_declarations(selected);
        self.nodes
            .iter()
            .map(|node| !entered[node.owner.decl.0 as usize])
            .collect()
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

/// chelis#616: the output axes of `node` whose symbolic dim is sized at RUN
/// TIME by the op itself — a movement op axis with a non-identity bound,
/// which [`shape_source_for_axis`] deliberately refuses to trace to a Load
/// (the extent is fresh, not the input axis's runtime dim), or a `Reshape`
/// axis whose target is a node-valued (`RtDim::Node`) extent. Each returned
/// `(symbol, axis)` pair names an extent that exists only once its operation
/// has run, so [`bind_symbolic_dims`] leaves it unbound and the eval lane
/// resolves it from actual values.
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
        RiscOp::ListMapCapture { .. } => match node.output_type.dims.first() {
            Some(DimInfo::Named(symbol, None))
                if !is_anon(symbol) && shape_source_for_axis(dag, node.id, 0).is_none() =>
            {
                vec![(symbol.clone(), 0)]
            }
            _ => Vec::new(),
        },
        RiscOp::Iota => match node.output_type.dims.first() {
            Some(DimInfo::Named(symbol, None)) if !is_anon(symbol) => vec![(symbol.clone(), 0)],
            _ => Vec::new(),
        },
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
        // A key split's node-valued count computes its appended last axis.
        RiscOp::SplitN {
            count: RtDim::Node(_),
        } => match node.output_type.dims.last() {
            Some(DimInfo::Named(symbol, None)) if !is_anon(symbol) => {
                vec![(symbol.clone(), node.output_type.dims.len() - 1)]
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
/// the root op will declare that symbol's value; anywhere else the symbol
/// reaches an emission boundary with no extent source, which
/// `axis_sources::check_rendered_dim_origins` refuses.
pub(crate) fn op_declarable_axes(dag: &Dag, node: &DagNode) -> Vec<usize> {
    match &node.op {
        RiscOp::Iota | RiscOp::ListMapCapture { .. } => vec![0],
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
        RiscOp::SplitN {
            count: RtDim::Node(_),
        } => node
            .output_type
            .dims
            .len()
            .checked_sub(1)
            .into_iter()
            .collect(),
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
/// dim's declaring node, when the declarer is earlier: a dependency names an
/// earlier node, and a node before the declarer that names the dim has it
/// from elsewhere (a parameter's shape). The declarer may be a key split: the
/// edge reads its extent, which is not key material ([04-LIN-9]). DCE and
/// grad's output pruning honor `shape_deps`, so this keeps the declarer — and, transitively, its bound-scalar chain —
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
                && *declarer < node.id
            {
                deps.push((node.id, *declarer));
            }
        }
    }
    for (from, to) in deps {
        dag.add_shape_dep(from, to);
    }
}

/// Symbolic dim names referenced by an op's internal fields rather than
/// its output type: `Reshape::new_shape` and
/// `BlasMatmul::{batch_dims, m, n, k}`.
pub(crate) fn op_internal_symbolic_dims(op: &RiscOp) -> Vec<String> {
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

/// Whether an op's internal payload still references `name` as a symbolic
/// dimension, as [`op_internal_symbolic_dims`] recognizes payloads.
///
/// A consumer that RENAMES a dimension identity calls this after rewriting an
/// op, so that an op payload the enumerator learns about later, and the rename
/// does not, fails the rename closed instead of producing a graph that declares
/// one symbol and reads another.
pub fn op_references_symbol(op: &RiscOp, name: &str) -> bool {
    op_internal_symbolic_dims(op)
        .iter()
        .any(|carried| carried == name)
}

/// Every symbolic dimension identity already carried by a DAG.
///
/// Keep output-axis and op-internal carriers behind one enumerator so a
/// producer allocating a fresh dimension cannot accidentally reserve only
/// the visible `TensorType` half of the namespace. The op-internal half is
/// exactly whatever [`op_internal_symbolic_dims`] recognizes, so an op that
/// spells a dimension symbol only inside its own payload still participates
/// even when no node output repeats that name.
pub fn dimension_identity_names(dag: &Dag) -> UnordSet<String> {
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
        | RiscOp::Compare(_)
        | RiscOp::Logical(_)
        | RiscOp::Where
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
        | RiscOp::Tanh
        | RiscOp::Abs
        | RiscOp::Floor
        | RiscOp::Ceil
        | RiscOp::Round
        | RiscOp::Relu
        | RiscOp::Softmax { .. }
        | RiscOp::UniformLike
        | RiscOp::Dropout
        | RiscOp::DropoutReplay => shape_source_for_axis(dag, *node.inputs.first()?, axis),
        RiscOp::Copy
        | RiscOp::Drop
        | RiscOp::Realize
        | RiscOp::Cast { .. }
        | RiscOp::NamedCast { .. }
        | RiscOp::KeyFromSeed
        | RiscOp::Split { .. }
        | RiscOp::FoldIn
        | RiscOp::KeySelect => shape_source_for_axis(dag, *node.inputs.first()?, axis),
        // [05-OP-71]: the key's axes pass through; the appended count axis
        // comes from the count, not from the key.
        RiscOp::SplitN { .. } => {
            let key = *node.inputs.first()?;
            if axis < dag.get(key)?.output_type.dims.len() {
                shape_source_for_axis(dag, key, axis)
            } else {
                None
            }
        }
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
/// The local half, an extent an operation computes at run time, is reported
/// by [`crate::axis_sources::dim_extent_origins`], which names the operation
/// that produces it. Each consumer reads one derivation or the other, never a
/// mixture: two derivations that can disagree is the defect this work
/// removes.
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

/// Names the scoped derivation finds in more than one equality class.
///
/// C2.4 scopes a claim by the results it reaches, so one spelling in two
/// independent signatures is two claims. This is NOT what
/// [`bind_symbolic_dims`] tolerates being unbound - that set is the caller's
/// `unbound`, the names whose scopes actually disagreed - and the difference
/// is the point: a multi-scope name whose scopes AGREE has one extent and
/// must be supplied. The predicate exists so a test can assert that a fixture
/// really does put a name in two scopes before asserting what follows.
#[cfg(test)]
fn multi_scope_dim_names(dag: &Dag) -> UnordSet<String> {
    let mut seen: Vec<String> = Vec::new();
    let mut repeated = UnordSet::new();
    for class in crate::axis_sources::derive_dim_witnesses(dag) {
        let crate::axis_sources::DimClaim::Name(name) = class.claim else {
            continue;
        };
        if seen.contains(&name) {
            repeated.insert(name);
        } else {
            seen.push(name);
        }
    }
    repeated
}

/// Resolve every named dimension the bindings map supplies, and refuse any
/// the graph still needs.
///
/// `unbound` is the set the CALLER deliberately left out, which is
/// information only the caller has: chelis#1566's rule is that a name whose
/// two scopes resolve to DIFFERENT extents binds to nothing, and whether two
/// scopes disagree is a property of the supplied values rather than of the
/// graph. Passing an empty set is therefore the strict reading, and every
/// name the graph needs must be present. An agreeing multi-scope name a
/// caller merely omits is an omitted binding like any other and is refused.
pub fn bind_symbolic_dims(
    dag: &Dag,
    bindings: &UnordMap<String, usize>,
    unbound: &UnordSet<String>,
) -> Result<Dag, String> {
    let op_declared = op_declared_dim_names(dag);
    let mut resolve = |name: &str, usage: SymbolBindingUse| {
        if let Some(size) = bindings.get(name) {
            Ok(Some(*size))
        } else if usage.allows_unbound(name, &op_declared, unbound) {
            Ok(None)
        } else {
            Err(usage.missing(name))
        }
    };
    let mut rebound = Dag::new();
    rebound.inherit_declarations(dag);
    for node in dag.nodes() {
        let (output_type, op) =
            map_node_symbolic_bindings(node, &mut resolve, &mut |message| Err(message.to_owned()))?;
        // Binding keeps every node id, so the owner's activation stands.
        let new_id = rebound.add_node(
            node.owner,
            op.unwrap_or_else(|| node.op.clone()),
            node.inputs.clone(),
            output_type,
            None,
        );
        // Binding is a 1:1 rebuild: retain shape-only dependencies and roots.
        if let Some(new_node) = rebound.node_mut(new_id) {
            new_node.shape_deps = node.shape_deps.clone();
            new_node.result_claim_deps = node.result_claim_deps.clone();
        }
        if dag.is_root(node.id) {
            rebound.add_root(new_id);
        }
    }
    Ok(rebound)
}

// The use, not just the symbol, decides whether evaluation can defer its value.
// ToEnd shares the runtime-extent policy but retains its existing diagnostic.
#[derive(Clone, Copy)]
enum SymbolBindingUse {
    Type,
    RuntimeExtent,
    ToEnd,
    Strict,
}

impl SymbolBindingUse {
    fn allows_unbound(
        self,
        name: &str,
        op_declared: &UnordSet<String>,
        unbound: &UnordSet<String>,
    ) -> bool {
        match self {
            Self::Type => {
                op_declared.contains(name)
                    || unbound.contains(name)
                    || name.is_empty()
                    || name == "*"
            }
            Self::RuntimeExtent | Self::ToEnd => {
                op_declared.contains(name) || name.is_empty() || name == "*"
            }
            Self::Strict => false,
        }
    }

    fn missing(self, name: &str) -> String {
        match self {
            Self::ToEnd => {
                format!("shrink-to-end sentinel left unbound for symbolic dimension `{name}`")
            }
            _ => format!("missing symbolic dimension binding `{name}`"),
        }
    }
}

/// One traversal owns both actual rewriting and observation of binding uses.
/// Unchanged ops remain borrowed (None), so observation never clones constants.
/// The observer's infallible refusal callback defers structural diagnostics to
/// the real binder; it does not validate or execute a trial graph.
fn map_node_symbolic_bindings<E>(
    node: &DagNode,
    resolve: &mut impl FnMut(&str, SymbolBindingUse) -> Result<Option<usize>, E>,
    refuse: &mut impl FnMut(&str) -> Result<(), E>,
) -> Result<(TensorType, Option<RiscOp>), E> {
    let output_type = TensorType {
        dims: node
            .output_type
            .dims
            .iter()
            .map(|dim| match dim {
                DimInfo::Named(name, None) => Ok(DimInfo::Named(
                    name.clone(),
                    resolve(name, SymbolBindingUse::Type)?,
                )),
                other => Ok(other.clone()),
            })
            .collect::<Result<_, E>>()?,
        precision: node.output_type.precision,
    };
    let op = match &node.op {
        RiscOp::Reshape { new_shape } => Some(RiscOp::Reshape {
            new_shape: new_shape
                .iter()
                .map(|dim| match dim {
                    RtDim::Sym(name) => Ok(resolve(name, SymbolBindingUse::RuntimeExtent)?
                        .map(RtDim::Lit)
                        .unwrap_or_else(|| dim.clone())),
                    RtDim::ToEnd => {
                        refuse("reshape target dim cannot be a shrink-to-end sentinel")?;
                        Ok(dim.clone())
                    }
                    other => Ok(other.clone()),
                })
                .collect::<Result<_, E>>()?,
        }),
        RiscOp::Shrink { bounds } if bounds.iter().any(|(_, end)| matches!(end, RtDim::ToEnd)) => {
            let resolved = bounds
                .iter()
                .zip(&output_type.dims)
                .map(|((start, end), dim)| {
                    if !matches!(end, RtDim::ToEnd) {
                        return Ok((start.clone(), end.clone()));
                    }
                    if start.as_lit() != Some(0) {
                        refuse("shrink-to-end sentinel requires a literal zero start")?;
                        return Ok((start.clone(), end.clone()));
                    }
                    let size = match dim {
                        DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => Some(*size),
                        DimInfo::Named(name, None) => resolve(name, SymbolBindingUse::ToEnd)?,
                    };
                    Ok(match size {
                        Some(size) => (RtDim::Lit(0), RtDim::Lit(size)),
                        None => (start.clone(), end.clone()),
                    })
                })
                .collect::<Result<_, E>>()?;
            Some(RiscOp::Shrink { bounds: resolved })
        }
        RiscOp::BlasMatmul {
            batch_dims,
            m,
            n,
            k,
            accumulator,
        } => {
            let mut strict = |name: &str| resolve(name, SymbolBindingUse::Strict);
            Some(RiscOp::BlasMatmul {
                batch_dims: batch_dims
                    .iter()
                    .map(|dim| dim.map_symbols(&mut strict))
                    .collect::<Result<_, E>>()?,
                m: m.map_symbols(&mut strict)?,
                n: n.map_symbols(&mut strict)?,
                k: k.map_symbols(&mut strict)?,
                accumulator: *accumulator,
            })
        }
        _ => None,
    };
    Ok((output_type, op))
}

/// Whole-DAG prebinding obligations for names no witness class answered.
/// A dynamic-unbound exemption cannot be guessed here: only inference knows
/// whether supplied values disagree, and such names are already class-claimed.
pub(crate) fn required_prebinding_symbols(dag: &Dag) -> UnordSet<String> {
    let op_declared = op_declared_dim_names(dag);
    let unbound = UnordSet::new();
    let mut required = UnordSet::new();
    for node in dag.nodes() {
        let observed = map_node_symbolic_bindings(
            node,
            &mut |name, usage| {
                if !usage.allows_unbound(name, &op_declared, &unbound) {
                    required.insert(name.to_owned());
                }
                Ok::<_, std::convert::Infallible>(None)
            },
            &mut |_| Ok(()),
        );
        match observed {
            Ok(_) => {}
            Err(never) => match never {},
        }
    }
    required
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    /// chelis#2413: a draw cannot trap only when every guard is statically
    /// satisfied ([`Dag::random_node_may_trap`]): its controls are literals
    /// in range at its dtype, and its key is rank 0 or its literal dims equal
    /// its data's literal leading dims. One negative per guard.
    mod draw_trap_guards {
        use super::*;
        use chelis_types::types::Prim;

        fn ty(dims: &[DimInfo], precision: Prim) -> TensorType {
            TensorType {
                dims: dims.to_vec(),
                precision,
            }
        }

        fn lit(dims: &[usize]) -> Vec<DimInfo> {
            dims.iter().copied().map(DimInfo::Lit).collect()
        }

        /// A key of `key_dims` (`None` for `key_from_seed(7)`), data of
        /// `data_dims`, and the draw `op` over them with `controls`, each a
        /// literal or, for `None`, a runtime `Load`.
        fn draw(
            op: RiscOp,
            controls: &[Option<f64>],
            key_dims: Option<Vec<DimInfo>>,
            data_dims: Vec<DimInfo>,
        ) -> (Dag, NodeId) {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let data = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty(&data_dims, Prim::F32),
                None,
            );
            let mut inputs = vec![data];
            for (slot, control) in controls.iter().enumerate() {
                let op = match control {
                    Some(value) => RiscOp::synth_const(Prim::F32, *value),
                    None => RiscOp::Load {
                        name: format!("control{slot}").as_str().into(),
                    },
                };
                inputs.push(dag.add_node(decl, op, vec![], scalar_f32(), None));
            }
            let key = match key_dims {
                None => {
                    let seed = dag.add_node(
                        decl,
                        RiscOp::Const {
                            value: chelis_types::scalar_from_i64("test", Prim::Int64, 7).unwrap(),
                        },
                        vec![],
                        ty(&[], Prim::Int64),
                        None,
                    );
                    dag.add_node(
                        decl,
                        RiscOp::KeyFromSeed,
                        vec![seed],
                        ty(&[], Prim::Key),
                        None,
                    )
                }
                Some(dims) => dag.add_node(
                    decl,
                    RiscOp::Load { name: "k".into() },
                    vec![],
                    ty(&dims, Prim::Key),
                    None,
                ),
            };
            inputs.push(key);
            let drawn = dag.add_node(decl, op, inputs, ty(&data_dims, Prim::F32), None);
            (dag, drawn)
        }

        fn may_trap((dag, drawn): &(Dag, NodeId)) -> bool {
            dag.random_node_may_trap(dag.get(*drawn).unwrap())
        }

        /// Evidentiary status: REGRESSION TEST. At 727e74b41 every draw was
        /// a seed, so a discarded draw with an in-range literal rate kept its
        /// data live (`dead_draw_input_is_not_required_when_not_data_live_at_the_selected_root`).
        #[test]
        fn a_draw_whose_guards_all_hold_statically_cannot_trap() {
            for graph in [
                draw(RiscOp::Dropout, &[Some(0.0)], None, lit(&[4])),
                draw(RiscOp::Dropout, &[Some(0.5)], Some(lit(&[2])), lit(&[2, 4])),
                draw(
                    RiscOp::UniformLike,
                    &[Some(-1.0), Some(1.0)],
                    None,
                    lit(&[4]),
                ),
            ] {
                assert!(!may_trap(&graph));
                // Dead, it is eliminated with its data.
                let (mut dag, _) = graph;
                let root = dag.add_node(
                    DeclId(0),
                    RiscOp::synth_const(Prim::F32, 1.0),
                    vec![],
                    scalar_f32(),
                    None,
                );
                dag.add_root(root);
                let pruned = crate::optimize::dead_code_eliminate(&dag);
                assert_eq!(pruned.len(), 1, "{:?}", pruned.nodes());
            }
        }

        /// Evidentiary status: disposition lock (every draw was a seed at
        /// 727e74b41).
        #[test]
        fn an_out_of_range_literal_control_may_trap() {
            assert!(may_trap(&draw(
                RiscOp::Dropout,
                &[Some(1.0)],
                None,
                lit(&[4])
            )));
            assert!(may_trap(&draw(
                RiscOp::Dropout,
                &[Some(-0.5)],
                None,
                lit(&[4])
            )));
            assert!(may_trap(&draw(
                RiscOp::UniformLike,
                &[Some(1.0), Some(-1.0)],
                None,
                lit(&[4])
            )));
        }

        /// Evidentiary status: disposition lock (every draw was a seed at
        /// 727e74b41).
        #[test]
        fn a_symbolic_key_extent_against_a_literal_one_may_trap() {
            let symbolic = vec![DimInfo::Named("n".into(), None)];
            assert!(may_trap(&draw(
                RiscOp::Dropout,
                &[Some(0.5)],
                Some(symbolic.clone()),
                lit(&[2, 4])
            )));
            let mut data = symbolic;
            data.push(DimInfo::Lit(4));
            assert!(may_trap(&draw(
                RiscOp::Dropout,
                &[Some(0.5)],
                Some(lit(&[2])),
                data
            )));
            assert!(may_trap(&draw(
                RiscOp::Dropout,
                &[Some(0.5)],
                Some(lit(&[3])),
                lit(&[2, 4])
            )));
        }

        /// Evidentiary status: disposition lock (every draw was a seed at
        /// 727e74b41).
        #[test]
        fn a_runtime_control_may_trap() {
            assert!(may_trap(&draw(RiscOp::Dropout, &[None], None, lit(&[4]))));
            assert!(may_trap(&draw(
                RiscOp::UniformLike,
                &[Some(0.0), None],
                None,
                lit(&[4])
            )));
        }
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
        let decl = dag.declare("test");
        let id = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        assert_eq!(dag.len(), 3);
        let node = dag.get(c).unwrap();
        assert_eq!(node.inputs, vec![NodeId(0), NodeId(1)]);
    }

    #[test]
    fn strict_shape_dependency_remap_rejects_missing_correspondence() {
        let mut source = Dag::new();
        let source_decl = source.declare("test");
        let dep = source.add_node(
            source_decl,
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let owner = source.add_node(source_decl, RiscOp::Neg, vec![dep], scalar_f32(), None);
        source.add_shape_dep(owner, dep);

        let mut rebuilt = Dag::new();
        let rebuilt_decl = rebuilt.declare("test");
        let rebuilt_owner = rebuilt.add_node(
            rebuilt_decl,
            RiscOp::synth_const(Prim::F32, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        let error = rebuilt
            .preserve_shape_deps_strict(
                rebuilt_owner,
                &source.get(owner).unwrap().shape_deps,
                &UnordMap::new(),
            )
            .unwrap_err();
        assert!(
            error.contains("shape dependency") && error.contains("no remapped node"),
            "{error}"
        );
        assert!(rebuilt.get(rebuilt_owner).unwrap().shape_deps.is_empty());
    }

    #[test]
    fn topological_order() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(decl, RiscOp::Neg, vec![a], scalar_f32(), None);
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
        let decl = dag.declare("test");
        let id = dag.add_node(
            decl,
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

    /// The interface bindings and the parameter list follow ABI input-slot
    /// order, and the canonical member is the first slot's axis.
    ///
    /// This used to assert the legacy occurrence walk's own output. The walk
    /// is gone (chelis#665): declarations come from the axis SOURCE now, so
    /// what is left to pin here is the derived interface the HIP prologue and
    /// `symbolic_params` read.
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
        let decl = dag.declare("test");
        dag.add_node(decl, RiscOp::Load { name: "x".into() }, vec![], ty_x, None);
        dag.add_node(decl, RiscOp::Load { name: "y".into() }, vec![], ty_y, None);

        assert_eq!(symbolic_params(&dag), vec!["batch"]);

        let bindings = symbolic_bindings_interface(&dag);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].name, "batch");
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

    /// Both `Load` axes name one extent, and the declaration for it comes
    /// from the first input slot rather than from a search for a matching
    /// string (chelis#665).
    #[test]
    fn a_shared_binder_declares_from_the_first_input_slots_axis() {
        let ty = |dims: Vec<DimInfo>| TensorType {
            dims,
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(4)]),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            ty(vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(2)]),
            None,
        );

        assert_eq!(
            crate::axis_sources::dim_extent_origins(&dag),
            vec![(
                "batch".to_string(),
                crate::axis_sources::ExtentOrigin::ExternalAxis { load: x, axis: 0 }
            )],
        );
        assert!(crate::axis_sources::unresolved_dim_names(&dag).is_empty());
    }

    #[test]
    fn bind_symbolic_dims_rewrites_output_types_and_preserves_input_axis_sizes() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("batch".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let h = dag.add_node(
            decl,
            RiscOp::Load { name: "h".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("hidden".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let y = dag.add_node(
            decl,
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
            &UnordSet::new(),
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
        let decl = dag.declare("test");
        let g = dag.add_node(
            decl,
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
            decl,
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

        let rebound = bind_symbolic_dims(
            &dag,
            &UnordMap::from([("m".to_string(), 3usize)]),
            &UnordSet::new(),
        )
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
        let decl = dag.declare("test");
        let g = dag.add_node(
            decl,
            RiscOp::Load { name: "g".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("m".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let shrunk = dag.add_node(
            decl,
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
        let err = bind_symbolic_dims(&dag, &UnordMap::new(), &UnordSet::new());
        assert!(err.is_err(), "unbound symbolic dim must fail closed");
    }

    /// chelis#1566's tolerance is for DISAGREEING scopes, and only those.
    ///
    /// A name the scope split finds in two scopes has no single pre-eval
    /// extent when those scopes resolve differently, and the inference then
    /// binds nothing for it. It does NOT follow that every multi-scope name
    /// may go unbound: two scopes that agree have one extent, a caller that
    /// omits it has omitted a required binding, and the answer is the same
    /// loud refusal any other missing binding gets.
    ///
    /// EVIDENTIARY STATUS: regression test, watched failing on `bd84d2619`,
    /// where keying the tolerance on multi-scope membership alone returned
    /// `Ok` with the dims still `Named("seq", None)`.
    #[test]
    fn an_agreeing_multi_scope_name_still_requires_its_binding() {
        let ty = |dims: Vec<DimInfo>| TensorType {
            dims,
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Named("seq".into(), None)]),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            ty(vec![
                DimInfo::Named("batch".into(), None),
                DimInfo::Named("seq".into(), None),
            ]),
            None,
        );
        let from_x = dag.add_node(
            decl,
            RiscOp::Neg,
            vec![x],
            ty(vec![DimInfo::Named("seq".into(), None)]),
            None,
        );
        let from_y = dag.add_node(
            decl,
            RiscOp::Neg,
            vec![y],
            ty(vec![
                DimInfo::Named("batch".into(), None),
                DimInfo::Named("seq".into(), None),
            ]),
            None,
        );
        dag.add_root(from_x);
        dag.add_root(from_y);
        assert_eq!(
            multi_scope_dim_names(&dag).to_sorted(),
            vec![&"seq".to_string()],
            "the fixture must actually put `seq` in two scopes",
        );

        let error = bind_symbolic_dims(&dag, &UnordMap::new(), &UnordSet::new())
            .expect_err("an omitted binding is an omitted binding, multi-scope or not");
        assert!(
            error.contains("missing symbolic dimension binding `seq`"),
            "the refusal names the missing binder: {error}",
        );
    }

    /// The other side of the same rule: a name the CALLER deliberately left
    /// unbound, because its scopes disagreed, is tolerated in a type.
    ///
    /// EVIDENTIARY STATUS: regression test for the repair's own mechanism.
    /// Without it chelis#1566's witness cannot evaluate, which is the
    /// measurement the witness carries.
    #[test]
    fn a_deliberately_unbound_name_is_tolerated_in_a_type() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("seq".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(x);

        let unbound = UnordSet::from([("seq".to_string())]);
        let bound = bind_symbolic_dims(&dag, &UnordMap::new(), &unbound)
            .expect("a deliberately unbound name leaves its axis to be computed from values");
        assert_eq!(
            bound.get(x).expect("the load").output_type.dims,
            vec![DimInfo::Named("seq".into(), None)],
            "the axis keeps its unresolved claim rather than taking a guessed extent",
        );

        // The tolerance stops at a by-value read: a `Reshape` target that
        // SPELLS the name needs a number and there is none to give it.
        let mut reading = Dag::new();
        let reading_decl = reading.declare("test");
        let y = reading.add_node(
            reading_decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("seq".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        let reshaped = reading.add_node(
            reading_decl,
            RiscOp::Reshape {
                new_shape: vec![RtDim::Sym("seq".into())],
            },
            vec![y],
            TensorType {
                dims: vec![DimInfo::Named("seq".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        reading.add_root(reshaped);
        let error = bind_symbolic_dims(&reading, &UnordMap::new(), &unbound)
            .expect_err("a by-value read of an unbound name has no answer");
        assert!(
            error.contains("missing symbolic dimension binding `seq`"),
            "the refusal names the binder it cannot resolve: {error}",
        );
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
    fn verifier_rejects_expand_size_sym_without_consulting_the_symbol_walk() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            decl,
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
            RiscOp::Bitwise(chelis_types::BitwiseKind::And),
            RiscOp::Compare(ComparisonKind::Eq),
            RiscOp::Logical(LogicalKind::And),
            RiscOp::Where,
            RiscOp::GuardedFail {
                message: "sample".to_string(),
                trap_on_true: true,
            },
            RiscOp::MaxElem,
            RiscOp::Neg,
            RiscOp::Exp,
            RiscOp::Log,
            RiscOp::Sin,
            RiscOp::Sqrt,
            RiscOp::Cos,
            RiscOp::Tan,
            RiscOp::Atan,
            RiscOp::Tanh,
            RiscOp::Abs,
            RiscOp::Floor,
            RiscOp::Ceil,
            RiscOp::Round,
            RiscOp::Recip,
            RiscOp::UniformLike,
            RiscOp::Dropout,
            RiscOp::DropoutReplay,
            RiscOp::UniformBoundAdjoint {
                bound: UniformBound::High,
            },
            RiscOp::KeyFromSeed,
            RiscOp::Split {
                branch: KeyBranch::Left,
            },
            RiscOp::FoldIn,
            RiscOp::SplitN {
                count: RtDim::Lit(3),
            },
            RiscOp::KeySelect,
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
            RiscOp::NamedCast {
                mode: NamedCastMode::Trunc,
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
            RiscOp::Gather {
                axis: 0,
                batch_rank: 0,
            },
            RiscOp::ScatterAdd {
                axis: 0,
                batch_rank: 0,
            },
            RiscOp::Scatter {
                axis: 0,
                batch_rank: 0,
            },
            RiscOp::ScatterElements { axis: 0 },
            RiscOp::Relu,
            RiscOp::Softmax { axis: 0 },
            RiscOp::ReluAdjoint,
        ]
    }

    /// chelis#2413: the verifier's key rule V4 reads the key allow-list
    /// (`chelis_types::key_admission`). Every graph operation admits a key at
    /// exactly the input slots of its role's admission, and the admissions
    /// graph operations reach are exactly the list's graph admissions, the
    /// same list the linearity checker reads.
    ///
    /// Each slot's admission is its read's ([`crate::verify::slot_read`]): an
    /// extent slot is an extent observation whatever the operation.
    ///
    /// Evidentiary status: REGRESSION TEST for `Drop`: at `f4eeca363` a key
    /// reaching a `Drop` broke V4, so `def f(k: key) = drop(k)` checked and
    /// then failed the evaluator's key rules. A lock for every other op.
    #[test]
    fn every_risc_op_admits_a_key_exactly_where_the_allow_list_does() {
        use crate::verify::{KeyGraph, verify_key_rules};
        use chelis_types::key_admission::KeyAdmission;
        use std::collections::BTreeSet;
        let ty = |precision| TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision,
        };
        // An operation with no fixed key arity reads up to four inputs here.
        const DATA_SLOTS: usize = 4;
        let mut reached = BTreeSet::new();
        let mut failures = Vec::new();
        for op in one_of_every_risc_op() {
            let arity = op.key_operand_arity().unwrap_or(DATA_SLOTS);
            for slot in 0..arity {
                let mut dag = Dag::new();
                let decl = dag.declare("f");
                let load = |dag: &mut Dag, name: &str, precision| {
                    dag.add_node(
                        decl,
                        RiscOp::Load { name: name.into() },
                        vec![],
                        ty(precision),
                        None,
                    )
                };
                let key = load(&mut dag, "k", Prim::Key);
                let mut inputs: Vec<NodeId> = ["x0", "x1", "x2", "x3"][..arity]
                    .iter()
                    .map(|name| load(&mut dag, name, Prim::F32))
                    .collect();
                inputs[slot] = key;
                let node = dag.add_node(decl, op.clone(), inputs, ty(Prim::F32), None);
                dag.add_root(node);
                let role = KeyGraph::role(&dag, node.0);
                let mut errors = Vec::new();
                verify_key_rules(&dag, &mut errors);
                let refused = errors
                    .iter()
                    .any(|error| error.contains(&format!("reaches input {slot} of")));
                let admission = KeyGraph::slot_read(&dag, node.0, slot).admission(role, slot);
                if refused == admission.is_some() {
                    failures.push(format!("{op:?} slot {slot}: {admission:?}, {errors:?}"));
                }
                reached.extend(admission);
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
        let listed: BTreeSet<KeyAdmission> = KeyAdmission::ALL
            .into_iter()
            .filter(|admission| admission.in_graph())
            .collect();
        assert_eq!(reached, listed);
    }

    /// `op` with every runtime bound read from input 1's axis 0, the form in
    /// which each bound is an extent slot; `None` for an operation without
    /// bounds.
    fn with_extent_bounds(op: &RiscOp) -> Option<RiscOp> {
        let read = || RtDim::InputAxis {
            tensor: 1,
            axis: RtAxis::Lit(0),
        };
        Some(match op {
            RiscOp::Expand { axis, .. } => RiscOp::Expand {
                axis: *axis,
                size: read(),
            },
            RiscOp::Reshape { new_shape } => RiscOp::Reshape {
                new_shape: new_shape.iter().map(|_| read()).collect(),
            },
            RiscOp::Pad { padding, fill } => RiscOp::Pad {
                padding: padding.iter().map(|_| (read(), read())).collect(),
                fill: *fill,
            },
            RiscOp::Shrink { bounds } => RiscOp::Shrink {
                bounds: bounds.iter().map(|_| (read(), read())).collect(),
            },
            RiscOp::Stride { strides } => RiscOp::Stride {
                strides: strides.iter().map(|_| read()).collect(),
            },
            RiscOp::SplitN { .. } => RiscOp::SplitN { count: read() },
            _ => return None,
        })
    }

    /// chelis#2413 (spec/10 §3.2, [04-LIN-9]): a key tensor's extent is not
    /// key material. Every extent slot the table declares
    /// ([`crate::verify::slot_read`]), found by sweeping every operation and
    /// its bound-reading form rather than listed here, observes a key
    /// without using it: the key's one `Drop` is still its one use, and a
    /// second `Drop` is still refused. Every kind of extent slot is reached.
    ///
    /// Evidentiary status: REGRESSION TEST for the bound slots. At
    /// `83f9781fe` a key at an `InputAxis` bound (`expand(s, 0i32, shape(ks,
    /// 0i32))` folds to one) was refused ("reaches input 1").
    #[test]
    fn every_extent_slot_observes_a_key_without_using_it() {
        use crate::verify::{ExtentSlot, SlotRead, slot_read, verify_key_rules};
        use std::collections::BTreeSet;
        let ty = |precision| TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision,
        };
        let witness = RiscOp::ExtentWitness {
            site: ExtentWitnessSite::Caller,
            parameter: "ks".into(),
            axis: RtAxis::Lit(0),
            requirements: vec![],
            claims: vec![],
        };
        let ops = one_of_every_risc_op()
            .iter()
            .flat_map(|op| [Some(op.clone()), with_extent_bounds(op)])
            .flatten()
            .chain([witness])
            .collect::<Vec<_>>();
        let mut reached = BTreeSet::new();
        let mut failures = Vec::new();
        for op in &ops {
            for slot in 0..4 {
                let SlotRead::Extent(kind) = slot_read(op, slot) else {
                    continue;
                };
                reached.insert(kind);
                for drops in [1, 2] {
                    let mut dag = Dag::new();
                    let decl = dag.declare("f");
                    let key = dag.add_node(
                        decl,
                        RiscOp::Load { name: "ks".into() },
                        vec![],
                        ty(Prim::Key),
                        None,
                    );
                    let mut inputs = (0..=slot)
                        .map(|input| {
                            dag.add_node(
                                decl,
                                RiscOp::Load {
                                    name: format!("x{input}").as_str().into(),
                                },
                                vec![],
                                ty(Prim::F32),
                                None,
                            )
                        })
                        .collect::<Vec<_>>();
                    inputs[slot] = key;
                    // A split produces keys whatever its count reads.
                    let produces = match op {
                        RiscOp::SplitN { .. } => Prim::Key,
                        _ => Prim::F32,
                    };
                    let node = dag.add_node(decl, op.clone(), inputs, ty(produces), None);
                    dag.add_root(node);
                    for _ in 0..drops {
                        dag.add_node(decl, RiscOp::Drop, vec![key], ty(Prim::Key), None);
                    }
                    let mut errors = Vec::new();
                    verify_key_rules(&dag, &mut errors);
                    let verdict = match drops {
                        1 => errors.is_empty(),
                        _ => {
                            errors.len() == 1
                                && errors[0].starts_with("key `ks` of `f` is consumed twice")
                        }
                    };
                    if !verdict {
                        failures.push(format!(
                            "{op:?} slot {slot} ({kind:?}), {drops} drops: {errors:?}"
                        ));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
        assert_eq!(reached, BTreeSet::from(ExtentSlot::ALL));
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
            69,
            "one_of_every_risc_op must list all 69 classified samples"
        );

        // The classifier returns a definite bool for every variant (no
        // panic, no escape); count both sides and pin the partition.
        let targetable = all.iter().filter(|op| op.is_verifier_targetable()).count();
        let excluded = all.len() - targetable;

        // Pinned partition per beacon_plan.md §3.1: the elementwise math
        // (5 binary/cmp + 14 unary, including `round` and the chelis#2957
        // `tanh` primitive), 5 reductions, 6
        // movement, 4 memory/blas value nodes (Const, ConstTensor, Load,
        // BlasMatmul), and Cast are targetable (35); stochastic (the two
        // key-operand draws and their two AD replays: 4),
        // arg-reductions (2), integer floor/trunc division and remainder (3),
        // `cast_trunc` (1, chelis#759), one_hot (1), the `Shape` metadata read
        // (1), sparse gather/scatter (4, including element-wise
        // `ScatterElements`), linearity/lifecycle markers + store (4),
        // reduce-window-grad (1), fused-elem (1), and the dedicated ReLU
        // identity/adjoint (2) are excluded (22) until Beacon registers their
        // own transformers. chelis#1464 adds the [05-OP-68] guarded abort to
        // the excluded side (+1 = 25): an abort is a control effect, not an
        // output envelope, and relaxing it to its fallback's envelope would
        // drop the trap. The chelis#2413 key-operand IR replaces the two
        // baked draws with the two key-operand draws and adds their two
        // AD replays (+2 = 27). The four explicit key derivations produce
        // opaque keys, not numeric envelopes (+4 = 31), and so does a
        // branch's key join (+1 = 32). The internal extrema adjoint
        // remains excluded (+1 = 33); the retained [05-OP-48] Softmax
        // composition has no dedicated transformer (+1 = 34).
        assert_eq!(
            targetable, 35,
            "targetable op count drifted from the pinned WI-2 subset"
        );
        assert_eq!(
            excluded, 34,
            "excluded op count drifted from the pinned WI-2 subset"
        );

        // Spot-check a representative op on each side so a wrong
        // reclassification (not just a count drift) is caught.
        assert!(RiscOp::Add.is_verifier_targetable());
        assert!(RiscOp::Exp.is_verifier_targetable());
        assert!(
            !RiscOp::Softmax { axis: 0 }.is_verifier_targetable(),
            "retained Softmax requires its own transformer before verifier admission"
        );
        assert!(
            RiscOp::Compare(ComparisonKind::CmpLt).is_verifier_targetable(),
            "Compare(CmpLt) drives erf64 branch-and-bound; must be targetable"
        );
        assert!(
            RiscOp::Cast {
                new_precision: Prim::F32
            }
            .is_verifier_targetable(),
            "Cast is real-valued-first targetable (beacon_plan.md §6)"
        );
        assert!(
            !RiscOp::Dropout.is_verifier_targetable(),
            "stochastic ops have no deterministic envelope to bound"
        );
        assert!(
            !RiscOp::Argmax { axis: 0 }.is_verifier_targetable(),
            "argmax returns discrete indices, not a real envelope"
        );
        assert!(
            !RiscOp::NamedCast {
                mode: NamedCastMode::Trunc,
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
            RiscOp::Compare(ComparisonKind::CmpLt),
            RiscOp::Compare(ComparisonKind::Lt),
            RiscOp::Compare(ComparisonKind::Neq),
            RiscOp::Compare(ComparisonKind::Gt),
            RiscOp::Compare(ComparisonKind::Gte),
            RiscOp::Compare(ComparisonKind::Lte),
            RiscOp::Logical(LogicalKind::Or),
            RiscOp::Logical(LogicalKind::Not),
            RiscOp::Bitwise(chelis_types::BitwiseKind::Or),
            RiscOp::Bitwise(chelis_types::BitwiseKind::Xor),
            RiscOp::Bitwise(chelis_types::BitwiseKind::ShiftLeft),
            RiscOp::Bitwise(chelis_types::BitwiseKind::ShiftRight),
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
                    | RiscOp::KeySelect
            );
            assert_eq!(
                matches!(op.atom_disposition(), RiscAtomDisposition::Structural),
                structural,
                "only the explicit compiler/lifetime representation variants are structural: {op:?}"
            );
        }
    }
}
