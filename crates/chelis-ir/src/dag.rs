//! RISC DAG data structure.
//!
//! The DAG is a flat vector of nodes, each referencing earlier nodes by [`NodeId`].
//! Nodes are always in topological order (an invariant maintained by append-only construction).

use std::collections::{HashMap, HashSet};
use std::fmt;

use chelis_types::types::Prim;
use serde::{Deserialize, Serialize};

use crate::load_store_name::LoadStoreName;

/// Index into the DAG node array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// A single movement-op (`Pad` / `Shrink` / `Stride`) bound value.
///
/// chelis#616: movement bounds were compile-time `usize` only, so a runtime
/// (`shape()`-derived) `shrink`/`stride` bound was dropped to empty at lowering
/// and the windowed axis degraded to a `Named("*")` wildcard. A `RtDim` can now
/// be a compile-time literal, the full-axis sentinel, or a *runtime* value read
/// from a rank-0 integer node.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RtDim {
    /// A compile-time-constant bound.
    Lit(usize),
    /// The runtime end of a (symbolic) axis; `Shrink` `end` only.
    ToEnd,
    /// A runtime bound read from `inputs[i]` (a rank-0 integer node).
    Node(usize),
}

impl RtDim {
    /// The compile-time value, if this bound is a literal.
    pub fn as_lit(&self) -> Option<usize> {
        match self {
            RtDim::Lit(n) => Some(*n),
            _ => None,
        }
    }

    /// Whether this bound is only known at runtime (`Node` or `ToEnd`).
    pub fn is_runtime(&self) -> bool {
        matches!(self, RtDim::Node(_) | RtDim::ToEnd)
    }

    /// The `inputs` slot index if this bound is node-valued.
    pub fn node_input(&self) -> Option<usize> {
        match self {
            RtDim::Node(i) => Some(*i),
            _ => None,
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

/// Canonical key for sound dimension-expression equality.
///
/// The key is mathematically equivalent for positive-integer-valued dim
/// expressions (tensor axis sizes are always positive integers): multiplication
/// is flattened, factor-sorted, and constants are folded; division flattens
/// nested quotients, cancels common factors between numerator and denominator,
/// and GCD-reduces the concrete portions of both. Division stays structural
/// only when no cancellation applies (e.g. `(n * 3) / 2`, where 3 is not a
/// multiple of 2 and the symbolic factor `n` is not known to be divisible by
/// 2 either).
///
/// Symbols compare by their stored names. `DimExpr` currently carries no binder
/// identity or property scope, so this key does not alpha-rename symbolic dims.
/// A future scoped alpha-renaming path must take explicit same-binder aliases as
/// input instead of inferring equivalence from expression shape alone.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DimExprKey {
    Concrete(usize),
    Sym(String),
    Mul(Vec<DimExprKey>),
    Div(Box<DimExprKey>, Box<DimExprKey>),
}

impl DimExpr {
    pub fn evaluate(&self, bindings: &HashMap<String, usize>) -> Result<usize, String> {
        match self {
            Self::Concrete(value) => Ok(*value),
            Self::Sym(name) => bindings
                .get(name)
                .copied()
                .ok_or_else(|| format!("missing symbolic dimension binding `{name}`")),
            Self::Mul(lhs, rhs) => Ok(lhs.evaluate(bindings)? * rhs.evaluate(bindings)?),
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
            Self::Mul(lhs, rhs) => Some(lhs.as_concrete()? * rhs.as_concrete()?),
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

    pub fn symbolic_names(&self) -> HashSet<String> {
        let mut names = HashSet::new();
        self.collect_symbolic_names(&mut names);
        names
    }

    fn collect_symbolic_names(&self, names: &mut HashSet<String>) {
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

    pub fn bind(&self, bindings: &HashMap<String, usize>) -> Result<Self, String> {
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

    pub fn normalized_key(&self) -> DimExprKey {
        match self {
            Self::Concrete(value) => DimExprKey::Concrete(*value),
            Self::Sym(name) => DimExprKey::Sym(name.clone()),
            Self::Mul(lhs, rhs) => {
                normalize_dim_product([lhs.normalized_key(), rhs.normalized_key()])
            }
            Self::Div(lhs, rhs) => {
                normalize_dim_quotient(lhs.normalized_key(), rhs.normalized_key())
            }
        }
    }
}

/// Greatest common divisor for usize. Euclidean algorithm. `gcd(0, x) = x`.
fn usize_gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Split a product key into (concrete factor, symbolic-atom multiset).
///
/// Symbolic atoms are non-`Mul` and non-`Concrete` keys — i.e. `Sym` or
/// residual `Div` that could not be cancelled.
fn flatten_product(key: DimExprKey) -> (usize, Vec<DimExprKey>) {
    let mut concrete = 1usize;
    let mut atoms = Vec::new();
    push_product_atoms(key, &mut concrete, &mut atoms);
    (concrete, atoms)
}

fn push_product_atoms(key: DimExprKey, concrete: &mut usize, atoms: &mut Vec<DimExprKey>) {
    match key {
        DimExprKey::Concrete(value) => *concrete = concrete.saturating_mul(value),
        DimExprKey::Mul(nested) => {
            for factor in nested {
                push_product_atoms(factor, concrete, atoms);
            }
        }
        atom @ (DimExprKey::Sym(_) | DimExprKey::Div(_, _)) => atoms.push(atom),
    }
}

/// Reassemble a canonical key from a concrete factor and a sorted atom multiset.
///
/// - if both are empty (concrete == 1 and no atoms), returns `Concrete(1)`
/// - if concrete is 0, returns `Concrete(0)` (the multiset is irrelevant)
/// - if there is exactly one factor (concrete = 1 with one atom, or no atoms
///   with concrete > 1), returns that factor directly
/// - otherwise returns a `Mul` of the sorted factors with the concrete (if !=1)
///   appended last so canonical ordering keeps `Concrete` after `Sym`/`Div`.
fn assemble_product(concrete: usize, mut atoms: Vec<DimExprKey>) -> DimExprKey {
    if concrete == 0 {
        return DimExprKey::Concrete(0);
    }
    if concrete != 1 {
        atoms.push(DimExprKey::Concrete(concrete));
    }
    atoms.sort();
    match atoms.len() {
        0 => DimExprKey::Concrete(1),
        1 => atoms.pop().expect("one factor"),
        _ => DimExprKey::Mul(atoms),
    }
}

fn normalize_dim_product(factors: impl IntoIterator<Item = DimExprKey>) -> DimExprKey {
    let mut num_concrete = 1usize;
    let mut num_atoms: Vec<DimExprKey> = Vec::new();
    let mut denom_concrete = 1usize;
    let mut denom_atoms: Vec<DimExprKey> = Vec::new();
    let mut saw_div = false;

    // Collapse a single factor into the running numerator / denominator.
    // Calling `take_factor` recursively handles `Mul` and `Div`; `Concrete(0)`
    // is caught at the top of the loop below.
    fn take_factor(
        factor: DimExprKey,
        num_concrete: &mut usize,
        num_atoms: &mut Vec<DimExprKey>,
        denom_concrete: &mut usize,
        denom_atoms: &mut Vec<DimExprKey>,
        saw_div: &mut bool,
    ) {
        match factor {
            DimExprKey::Concrete(value) => *num_concrete = num_concrete.saturating_mul(value),
            DimExprKey::Sym(_) => num_atoms.push(factor),
            DimExprKey::Mul(nested) => {
                for nested_factor in nested {
                    take_factor(
                        nested_factor,
                        num_concrete,
                        num_atoms,
                        denom_concrete,
                        denom_atoms,
                        saw_div,
                    );
                }
            }
            DimExprKey::Div(num, denom) => {
                *saw_div = true;
                let (c_num, atoms_num) = flatten_product(*num);
                let (c_denom, atoms_denom) = flatten_product(*denom);
                *num_concrete = num_concrete.saturating_mul(c_num);
                num_atoms.extend(atoms_num);
                *denom_concrete = denom_concrete.saturating_mul(c_denom);
                denom_atoms.extend(atoms_denom);
            }
        }
    }

    for factor in factors {
        if let DimExprKey::Concrete(0) = factor {
            return DimExprKey::Concrete(0);
        }
        take_factor(
            factor,
            &mut num_concrete,
            &mut num_atoms,
            &mut denom_concrete,
            &mut denom_atoms,
            &mut saw_div,
        );
    }

    if !saw_div {
        return assemble_product(num_concrete, num_atoms);
    }

    normalize_quotient_parts(num_concrete, num_atoms, denom_concrete, denom_atoms)
}

fn normalize_dim_quotient(num: DimExprKey, denom: DimExprKey) -> DimExprKey {
    // Handle a nested numerator quotient: `(a / b) / c = a / (b * c)`. We
    // lift the inner denominator into the outer denominator and recurse on
    // the cleaned-up form.
    if let DimExprKey::Div(inner_num, inner_denom) = num {
        let combined_denom = normalize_dim_product([*inner_denom, denom]);
        return normalize_dim_quotient(*inner_num, combined_denom);
    }

    // Handle nested division on the denominator: `a / (b / c) = (a * c) / b`.
    if let DimExprKey::Div(inner_num, inner_denom) = denom {
        let combined_num = normalize_dim_product([num, *inner_denom]);
        return normalize_dim_quotient(combined_num, *inner_num);
    }

    let (num_concrete, num_atoms) = flatten_product(num);
    let (denom_concrete, denom_atoms) = flatten_product(denom);
    normalize_quotient_parts(num_concrete, num_atoms, denom_concrete, denom_atoms)
}

/// Cancel common atom factors and GCD-reduce the concrete portions of a
/// numerator / denominator pair, then reassemble the canonical key.
///
/// This relies on tensor axis sizes being positive integers: `n / n = 1` is
/// only sound when `n > 0`. Chelis dimensions are always positive (zero-axis
/// tensors are degenerate and not used as slot keys), so atom-level
/// cancellation is sound. Constants are GCD-reduced exactly; division
/// remains structural when nothing more cancels.
fn normalize_quotient_parts(
    mut num_concrete: usize,
    mut num_atoms: Vec<DimExprKey>,
    mut denom_concrete: usize,
    mut denom_atoms: Vec<DimExprKey>,
) -> DimExprKey {
    if num_concrete == 0 {
        // 0 / x is 0; division by zero is rejected by `evaluate`, but at
        // the key level a structural `0 / x` collapses to `Concrete(0)`.
        return DimExprKey::Concrete(0);
    }
    if denom_concrete == 0 {
        // Division by an all-zero denominator would be invalid; keep
        // structural so the runtime evaluator can surface the error.
        return DimExprKey::Div(
            Box::new(assemble_product(num_concrete, num_atoms)),
            Box::new(DimExprKey::Concrete(0)),
        );
    }

    // Cancel matching atoms between numerator and denominator.
    num_atoms.sort();
    denom_atoms.sort();
    let mut cancelled_num: Vec<DimExprKey> = Vec::with_capacity(num_atoms.len());
    let mut remaining_denom: Vec<DimExprKey> = Vec::with_capacity(denom_atoms.len());
    let mut denom_iter = denom_atoms.into_iter().peekable();
    for atom in num_atoms {
        // Advance denom_iter past atoms strictly less than `atom`.
        while let Some(d) = denom_iter.peek() {
            if *d < atom {
                remaining_denom.push(denom_iter.next().expect("peeked"));
            } else {
                break;
            }
        }
        if denom_iter.peek() == Some(&atom) {
            // Cancel one copy.
            denom_iter.next();
        } else {
            cancelled_num.push(atom);
        }
    }
    remaining_denom.extend(denom_iter);

    // GCD-reduce the concrete factors.
    let g = usize_gcd(num_concrete, denom_concrete);
    if g > 1 {
        num_concrete /= g;
        denom_concrete /= g;
    }

    // If the denominator collapses to 1 with no remaining atoms, the
    // result is a pure product.
    if denom_concrete == 1 && remaining_denom.is_empty() {
        return assemble_product(num_concrete, cancelled_num);
    }

    // Otherwise keep a structural Div. Both sides are themselves
    // canonical products.
    let numerator = assemble_product(num_concrete, cancelled_num);
    let denominator = assemble_product(denom_concrete, remaining_denom);
    DimExprKey::Div(Box::new(numerator), Box::new(denominator))
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicDimOccurrence {
    pub name: String,
    pub input_label: String,
    pub axis: usize,
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
    Mul,
    Div,
    /// Floor division step (chelis#178); see [`RiscOp::FloorDiv`].
    FloorDiv,
    /// Truncating (round-toward-zero) division step, integer-only
    /// (chelis#178); see [`RiscOp::TruncDiv`].
    TruncDiv,
    MaxElem,
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

/// A RISC primitive operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiscOp {
    // --- Binary elementwise ---
    Add,
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
    CmpLt,
    MaxElem,

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
    Reshape {
        new_shape: Vec<DimInfo>,
    },
    Permute {
        axes: Vec<usize>,
    },
    Expand {
        axis: usize,
        size: DimExpr,
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
        fill: f64,
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

    // --- Memory ---
    Const {
        value: f64,
    },
    /// A multi-element constant tensor stored as a flat row-major data
    /// vector. Replaces the Const+Pad+Add tree for literal `to_tensor`
    /// calls with non-uniform data.
    ConstTensor {
        data: Vec<f64>,
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

impl RiscOp {
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
            RiscOp::Add | RiscOp::Mul | RiscOp::Div | RiscOp::CmpLt | RiscOp::MaxElem => true,

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
            RiscOp::Argmax { .. } | RiscOp::Argmin { .. } => false,

            // Floor/truncating integer division (chelis#178) are
            // piecewise-constant, non-differentiable rounding ops; like
            // `Floor`/`Ceil`/`Round` they have a step-function envelope,
            // but the integer-quotient semantics are not part of the
            // pinned real-valued forward-bound surface today.
            RiscOp::FloorDiv | RiscOp::TruncDiv => false,

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
            RiscOp::Shape { .. } => false,

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
    /// their data. A Form-3 `expand(b, axis, shape(x, k))` reads the
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
    /// (a Form-3 `expand(..., shape(x, ...))` extent source). DCE keeps `dep`
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
    /// `merged_spans` preservation so a Form-3 `expand` extent source is not
    /// silently dropped on the rebuild — losing it reintroduces the
    /// wrong-shape regression. Deps that don't survive the rebuild's remap
    /// are dropped (the consuming node was itself eliminated, so the dep is
    /// moot).
    pub fn preserve_shape_deps(
        &mut self,
        new_id: NodeId,
        source_deps: &[NodeId],
        remap: &HashMap<NodeId, NodeId>,
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
    let mut seen_inputs = HashSet::new();
    let mut named_dims_in_loads: HashSet<String> = HashSet::new();

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
                occurrences.push(SymbolicDimOccurrence {
                    name: symbol.clone(),
                    input_label: name.as_str().to_string(),
                    axis,
                });
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
                    occurrences.push(SymbolicDimOccurrence {
                        name: symbol.clone(),
                        input_label,
                        axis: input_axis,
                    });
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
            if named_dims_in_loads.contains(&symbol) {
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

/// Hunt for any Load whose type carries `symbol` (bound or unbound) and
/// register a synthetic occurrence pointing at it. Returns whether a
/// declaring Load was found. Shared by the Bucket 4c (output-dim) and
/// Bucket 4d (op-internal) sweeps of [`symbolic_occurrences`].
fn bind_symbol_from_any_load(
    dag: &Dag,
    symbol: &str,
    occurrences: &mut Vec<SymbolicDimOccurrence>,
    named_dims_in_loads: &mut HashSet<String>,
) -> bool {
    for candidate in dag.nodes() {
        let RiscOp::Load { name: load_name } = &candidate.op else {
            continue;
        };
        for (axis, candidate_dim) in candidate.output_type.dims.iter().enumerate() {
            if let DimInfo::Named(candidate_sym, _) = candidate_dim
                && candidate_sym == symbol
            {
                occurrences.push(SymbolicDimOccurrence {
                    name: symbol.to_string(),
                    input_label: load_name.as_str().to_string(),
                    axis,
                });
                named_dims_in_loads.insert(symbol.to_string());
                return true;
            }
        }
    }
    false
}

/// Symbolic dim names referenced by an op's internal fields rather than
/// its output type: `Expand::size`, `Reshape::new_shape`, and
/// `BlasMatmul::{batch_dims, m, n, k}`. These are the only `RiscOp`
/// fields that carry `DimExpr` / unbound `DimInfo` payloads.
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
        RiscOp::Expand { size, .. } => collect_dim_expr(size, &mut out),
        RiscOp::Reshape { new_shape } => {
            for dim in new_shape {
                if let DimInfo::Named(name, None) = dim {
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

fn shape_source_for_axis(dag: &Dag, id: NodeId, axis: usize) -> Option<(String, usize)> {
    let node = dag.get(id)?;
    match &node.op {
        RiscOp::Load { name } => Some((name.as_str().to_string(), axis)),
        RiscOp::Add | RiscOp::Mul | RiscOp::CmpLt | RiscOp::MaxElem => node
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
        | RiscOp::UniformLike { .. }
        | RiscOp::Dropout { .. } => shape_source_for_axis(dag, *node.inputs.first()?, axis),
        RiscOp::Copy | RiscOp::Drop | RiscOp::Realize | RiscOp::Cast { .. } => {
            shape_source_for_axis(dag, *node.inputs.first()?, axis)
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
                return None;
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
        RiscOp::Reshape { .. } | RiscOp::Permute { .. } | RiscOp::Store { .. } => {
            shape_source_for_axis(dag, *node.inputs.first()?, axis)
        }
        // chelis#616 (soundness): a node-valued (runtime) movement bound produces
        // a FRESH runtime extent that is not the input axis's runtime dim, so it
        // must not be traced to the input's declaring `Load` (that would bind a
        // symbolic movement-output dim to the wrong input extent). `Lit` / `ToEnd`
        // axes keep the pre-existing pass-through: `ToEnd` is a full-axis identity
        // (same runtime dim as the input), `Lit` is concrete/non-symbolic.
        RiscOp::Shrink { bounds } => match bounds.get(axis) {
            Some((s, e)) if s.node_input().is_some() || e.node_input().is_some() => None,
            _ => shape_source_for_axis(dag, *node.inputs.first()?, axis),
        },
        RiscOp::Stride { strides } => match strides.get(axis) {
            Some(b) if b.node_input().is_some() => None,
            _ => shape_source_for_axis(dag, *node.inputs.first()?, axis),
        },
        RiscOp::Pad { padding, .. } => match padding.get(axis) {
            Some((b, a)) if b.node_input().is_some() || a.node_input().is_some() => None,
            _ => shape_source_for_axis(dag, *node.inputs.first()?, axis),
        },
        RiscOp::FusedElem { .. } => node
            .inputs
            .iter()
            .find_map(|input| shape_source_for_axis(dag, *input, axis)),
        _ => None,
    }
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
            let canonical = occurrences.remove(0);
            SymbolicDimBinding {
                name,
                canonical,
                others: occurrences,
            }
        })
        .collect()
}

pub fn symbolic_params(dag: &Dag) -> Vec<String> {
    symbolic_bindings(dag)
        .into_iter()
        .map(|binding| binding.name)
        .collect()
}

pub fn bind_symbolic_dims(dag: &Dag, bindings: &HashMap<String, usize>) -> Result<Dag, String> {
    fn bind_dim(dim: &DimInfo, bindings: &HashMap<String, usize>) -> Result<DimInfo, String> {
        match dim {
            DimInfo::Lit(size) => Ok(DimInfo::Lit(*size)),
            DimInfo::Named(name, Some(size)) => Ok(DimInfo::Named(name.clone(), Some(*size))),
            DimInfo::Named(name, None) => Ok(DimInfo::Named(
                name.clone(),
                Some(
                    bindings
                        .get(name)
                        .copied()
                        .ok_or_else(|| format!("missing symbolic dimension binding `{name}`"))?,
                ),
            )),
        }
    }

    let mut rebound = Dag::new();
    for node in dag.nodes() {
        let output_type = TensorType {
            dims: node
                .output_type
                .dims
                .iter()
                .map(|dim| bind_dim(dim, bindings))
                .collect::<Result<_, _>>()?,
            precision: node.output_type.precision,
        };
        let op = match &node.op {
            RiscOp::Expand { axis, size } => RiscOp::Expand {
                axis: *axis,
                size: size.bind(bindings)?,
            },
            RiscOp::Reshape { new_shape } => RiscOp::Reshape {
                new_shape: new_shape
                    .iter()
                    .map(|dim| bind_dim(dim, bindings))
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
                            // The sentinel is always emitted as `(Lit(0), ToEnd)`;
                            // a non-literal start is malformed.
                            let start_lit = start.as_lit().ok_or_else(|| {
                                "shrink-to-end sentinel with non-literal start".to_string()
                            })?;
                            match dim {
                                DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => {
                                    Ok((RtDim::Lit(start_lit), RtDim::Lit(start_lit + *size)))
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
        let id = dag.add_node(RiscOp::Const { value: 42.0 }, vec![], scalar_f32(), None);
        assert_eq!(id, NodeId(0));
        assert_eq!(dag.len(), 1);
        let node = dag.get(id).unwrap();
        assert_eq!(node.op, RiscOp::Const { value: 42.0 });
        assert!(node.inputs.is_empty());
    }

    #[test]
    fn add_binary_op() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        assert_eq!(dag.len(), 3);
        let node = dag.get(c).unwrap();
        assert_eq!(node.inputs, vec![NodeId(0), NodeId(1)]);
    }

    #[test]
    fn dim_expr_normalized_key_canonicalizes_mul_order_and_associativity() {
        let lhs = DimExpr::Mul(
            Box::new(DimExpr::Sym("batch".into())),
            Box::new(DimExpr::Mul(
                Box::new(DimExpr::Concrete(4)),
                Box::new(DimExpr::Sym("hidden".into())),
            )),
        );
        let rhs = DimExpr::Mul(
            Box::new(DimExpr::Sym("hidden".into())),
            Box::new(DimExpr::Mul(
                Box::new(DimExpr::Sym("batch".into())),
                Box::new(DimExpr::Concrete(4)),
            )),
        );

        assert_eq!(lhs.normalized_key(), rhs.normalized_key());
    }

    #[test]
    fn dim_expr_normalized_key_folds_mul_constants_and_identity() {
        let expr = DimExpr::Mul(
            Box::new(DimExpr::Concrete(2)),
            Box::new(DimExpr::Mul(
                Box::new(DimExpr::Sym("n".into())),
                Box::new(DimExpr::Concrete(1)),
            )),
        );
        let equivalent = DimExpr::Mul(
            Box::new(DimExpr::Sym("n".into())),
            Box::new(DimExpr::Concrete(2)),
        );

        assert_eq!(expr.normalized_key(), equivalent.normalized_key());
    }

    #[test]
    fn dim_expr_normalized_key_does_not_alpha_rename_unrelated_symbols() {
        assert_ne!(
            DimExpr::Sym("n".into()).normalized_key(),
            DimExpr::Sym("m".into()).normalized_key()
        );

        let first = DimExpr::Mul(
            Box::new(DimExpr::Sym("m".into())),
            Box::new(DimExpr::Sym("n".into())),
        );
        let alpha_renamed_shape = DimExpr::Mul(
            Box::new(DimExpr::Sym("x".into())),
            Box::new(DimExpr::Sym("y".into())),
        );

        assert_ne!(
            first.normalized_key(),
            alpha_renamed_shape.normalized_key(),
            "plain DimExpr symbols have no binder identity, so same-shaped \
             symbolic products are not equivalent under alpha-renaming"
        );
    }

    #[test]
    fn dim_expr_normalized_key_handles_div_exact_and_identity() {
        let exact = DimExpr::Div(
            Box::new(DimExpr::Concrete(12)),
            Box::new(DimExpr::Concrete(3)),
        );
        assert_eq!(exact.normalized_key(), DimExprKey::Concrete(4));

        let identity = DimExpr::Div(
            Box::new(DimExpr::Sym("n".into())),
            Box::new(DimExpr::Concrete(1)),
        );
        assert_eq!(identity.normalized_key(), DimExprKey::Sym("n".into()));
    }

    #[test]
    fn dim_expr_normalized_key_constant_factor_div_canonicalizes() {
        // Phase Perf-F2(a) broadens v1's structural-only Div handling.
        // `(n * 4) / 2 = n * 2` is mathematically exact for any positive
        // integer `n` (because 4 is exactly divisible by 2). v1 left this
        // structural; v2 GCD-reduces the concrete factor so the slot
        // planner can equate the two shapes and reuse a single slot.
        let quotient = DimExpr::Div(
            Box::new(DimExpr::Mul(
                Box::new(DimExpr::Sym("n".into())),
                Box::new(DimExpr::Concrete(4)),
            )),
            Box::new(DimExpr::Concrete(2)),
        );
        let product = DimExpr::Mul(
            Box::new(DimExpr::Sym("n".into())),
            Box::new(DimExpr::Concrete(2)),
        );
        assert_eq!(quotient.normalized_key(), product.normalized_key());
    }

    #[test]
    fn topological_order() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
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
        let id = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
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
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], const_ty, None);

        let occurrences = symbolic_occurrences(&dag);
        // The Load is the canonical source. The sibling-sweep pass must
        // not duplicate the Load occurrence for the Const (the Const's
        // dim is already covered).
        assert_eq!(occurrences.len(), 1);
        assert_eq!(occurrences[0].name, "n");
        assert_eq!(occurrences[0].input_label, "x");
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
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], const_ty, None);
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
                    input_label: "x".into(),
                    axis: 0,
                },
                SymbolicDimOccurrence {
                    name: "batch".into(),
                    input_label: "y".into(),
                    axis: 0,
                },
            ]
        );

        let bindings = symbolic_bindings(&dag);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].canonical.input_label, "x");
        assert_eq!(bindings[0].others.len(), 1);
        assert_eq!(bindings[0].others[0].input_label, "y");
    }

    #[test]
    fn bind_symbolic_dims_rewrites_output_types_and_expand_sizes() {
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
        let y = dag.add_node(
            RiscOp::Expand {
                axis: 1,
                size: DimExpr::Sym("hidden".into()),
            },
            vec![x],
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
            &HashMap::from([
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
                size: DimExpr::Concrete(8),
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

        let rebound = bind_symbolic_dims(&dag, &HashMap::from([("m".to_string(), 3usize)]))
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
        let err = bind_symbolic_dims(&dag, &HashMap::new());
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
    fn symbolic_occurrences_expand_size_sym_binds_from_load() {
        // Positive: the op-internal `Expand::size` sym is declared by a
        // Load, so the sweep stays quiet and the Load occurrence is the
        // canonical (and only) source — no duplicate occurrences.
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
        dag.add_node(
            RiscOp::Expand {
                axis: 1,
                size: DimExpr::Sym("n".into()),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );

        let occurrences = symbolic_occurrences(&dag);
        assert_eq!(occurrences.len(), 1);
        assert_eq!(occurrences[0].name, "n");
        assert_eq!(occurrences[0].input_label, "x");
    }

    #[test]
    #[should_panic(expected = "internal compiler error: symbolic dim `d7` is referenced")]
    fn symbolic_occurrences_panics_on_expand_size_sym_without_load() {
        // Negative: concrete output dims everywhere, but the Expand's
        // op-internal `size` references `d7`, which no Load declares.
        // This is the chelis#345 mixed state; the sweep must panic
        // instead of letting an undeclared identifier reach a backend.
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
                size: DimExpr::Sym("d7".into()),
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
                new_shape: vec![DimInfo::Named("d9".into(), None), DimInfo::Lit(2)],
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
                size: DimExpr::Concrete(3),
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
        assert_eq!(d2.input_label, "c");
        assert_eq!(d2.axis, 1);
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
            RiscOp::Const { value: 0.0 },
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
                new_shape: vec![DimInfo::Lit(4)],
            },
            RiscOp::Permute { axes: vec![0] },
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Concrete(4),
            },
            RiscOp::OneHot { vocab: 8 },
            RiscOp::Pad {
                padding: vec![(RtDim::Lit(0), RtDim::Lit(0))],
                fill: 0.0,
            },
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::Lit(1))],
            },
            RiscOp::Stride {
                strides: vec![RtDim::Lit(1)],
            },
            RiscOp::Shape { axis: 0 },
            RiscOp::Const { value: 1.0 },
            RiscOp::ConstTensor {
                data: vec![1.0, 2.0],
            },
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
        // 52-variant closed vocabulary (spec WI-2 / dag.rs RiscOp).
        assert_eq!(
            all.len(),
            52,
            "one_of_every_risc_op must list all 52 RiscOp variants"
        );

        // The classifier returns a definite bool for every variant (no
        // panic, no escape); count both sides and pin the partition.
        let targetable = all.iter().filter(|op| op.is_verifier_targetable()).count();
        let excluded = all.len() - targetable;

        // Pinned partition per beacon_plan.md §3.1: the elementwise math
        // (5 binary/cmp + 13 unary, including `round`), 5 reductions, 6
        // movement, 4 memory/blas value nodes (Const, ConstTensor, Load,
        // BlasMatmul), and Cast are targetable (34); stochastic (2),
        // arg-reductions (2), integer floor/trunc division (2), one_hot (1),
        // the `Shape` metadata read (1), sparse gather/scatter (4, including
        // element-wise `ScatterElements`), linearity/lifecycle markers + store
        // (4), reduce-window-grad (1), and fused-elem (1) are excluded (18).
        assert_eq!(
            targetable, 34,
            "targetable op count drifted from the pinned WI-2 subset"
        );
        assert_eq!(
            excluded, 18,
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
    }
}
