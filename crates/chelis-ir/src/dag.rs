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

/// Canonical key for conservative dimension-expression equality.
///
/// The key is intentionally weaker than algebraic simplification: multiplication
/// is flattened and sorted, constants are folded, and division stays structural
/// unless it can be evaluated exactly or the denominator is one.
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
                let lhs = lhs.normalized_key();
                let rhs = rhs.normalized_key();
                match (&lhs, &rhs) {
                    (DimExprKey::Concrete(lhs), DimExprKey::Concrete(rhs))
                        if *rhs != 0 && lhs % rhs == 0 =>
                    {
                        DimExprKey::Concrete(lhs / rhs)
                    }
                    (_, DimExprKey::Concrete(1)) => lhs,
                    _ => DimExprKey::Div(Box::new(lhs), Box::new(rhs)),
                }
            }
        }
    }
}

fn normalize_dim_product(factors: impl IntoIterator<Item = DimExprKey>) -> DimExprKey {
    let mut concrete = 1usize;
    let mut symbolic = Vec::new();
    for factor in factors {
        match factor {
            DimExprKey::Concrete(0) => return DimExprKey::Concrete(0),
            DimExprKey::Concrete(value) => concrete *= value,
            DimExprKey::Mul(nested) => {
                for nested_factor in nested {
                    match nested_factor {
                        DimExprKey::Concrete(0) => return DimExprKey::Concrete(0),
                        DimExprKey::Concrete(value) => concrete *= value,
                        other => symbolic.push(other),
                    }
                }
            }
            other => symbolic.push(other),
        }
    }

    if concrete != 1 {
        symbolic.push(DimExprKey::Concrete(concrete));
    }
    symbolic.sort();

    match symbolic.len() {
        0 => DimExprKey::Concrete(1),
        1 => symbolic.pop().expect("one symbolic factor"),
        _ => DimExprKey::Mul(symbolic),
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
    MaxElem,
    CmpLt,
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
    Sum {
        axis: usize,
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
    /// Index of maximum element along `axis`.
    ///
    /// Note: per Phase 3j-pre, argmax/argmin logically return integer indices.
    /// The evaluator stores them as f64 integer-valued floats and the type
    /// system carries whatever precision the caller assigns (typically F32,
    /// since the C backend only supports F32/Bool tensors). The Std wrapper
    /// layer (Batch 2) is responsible for casting/annotating as needed.
    /// TODO(phase3j): widen backend runtime to carry Int64 tensors natively.
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
    Pad {
        padding: Vec<(usize, usize)>,
        fill: f64,
    },
    Shrink {
        bounds: Vec<(usize, usize)>,
    },
    Stride {
        strides: Vec<usize>,
    },

    // --- Memory ---
    Const {
        value: f64,
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
    BlasMatmul {
        batch_dims: Vec<DimExpr>,
        m: DimExpr,
        n: DimExpr,
        k: DimExpr,
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
        });
        id
    }

    pub fn set_reusable_input(&mut self, id: NodeId, input: NodeId) {
        if let Some(node) = self.nodes.get_mut(id.0) {
            node.reusable_input = Some(input);
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
                let mut bound = false;
                for candidate in dag.nodes() {
                    let RiscOp::Load { name: load_name } = &candidate.op else {
                        continue;
                    };
                    for (axis, candidate_dim) in candidate.output_type.dims.iter().enumerate() {
                        if let DimInfo::Named(candidate_sym, _) = candidate_dim
                            && candidate_sym == symbol
                        {
                            occurrences.push(SymbolicDimOccurrence {
                                name: symbol.clone(),
                                input_label: load_name.as_str().to_string(),
                                axis,
                            });
                            named_dims_in_loads.insert(symbol.clone());
                            bound = true;
                            break;
                        }
                    }
                    if bound {
                        break;
                    }
                }
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

    occurrences
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
        | RiscOp::UniformLike { .. }
        | RiscOp::Dropout { .. } => shape_source_for_axis(dag, *node.inputs.first()?, axis),
        RiscOp::Copy | RiscOp::Drop | RiscOp::Realize | RiscOp::Cast { .. } => {
            shape_source_for_axis(dag, *node.inputs.first()?, axis)
        }
        RiscOp::Reshape { .. }
        | RiscOp::Permute { .. }
        | RiscOp::Expand { .. }
        | RiscOp::Pad { .. }
        | RiscOp::Shrink { .. }
        | RiscOp::Stride { .. }
        | RiscOp::Store { .. } => shape_source_for_axis(dag, *node.inputs.first()?, axis),
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
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
            } => RiscOp::BlasMatmul {
                batch_dims: batch_dims
                    .iter()
                    .map(|dim| dim.bind(bindings))
                    .collect::<Result<_, _>>()?,
                m: m.bind(bindings)?,
                n: n.bind(bindings)?,
                k: k.bind(bindings)?,
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
    fn dim_expr_normalized_key_handles_div_conservatively() {
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
        assert_ne!(quotient.normalized_key(), product.normalized_key());
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
}
