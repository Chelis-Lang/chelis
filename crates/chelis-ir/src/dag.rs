//! RISC DAG data structure.
//!
//! The DAG is a flat vector of nodes, each referencing earlier nodes by [`NodeId`].
//! Nodes are always in topological order (an invariant maintained by append-only construction).

use std::collections::{HashMap, HashSet};
use std::fmt;

use chelis_types::types::Prim;

/// Index into the DAG node array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Tensor type carried on each DAG node.
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimInfo {
    /// A named dimension with an optional known size.
    Named(String, Option<usize>),
    /// A fixed literal size.
    Lit(usize),
}

/// Runtime-capable dimension expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimExpr {
    Concrete(usize),
    Sym(String),
    Mul(Box<DimExpr>, Box<DimExpr>),
    Div(Box<DimExpr>, Box<DimExpr>),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicDimOccurrence {
    pub name: String,
    pub input_label: String,
    pub axis: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicDimBinding {
    pub name: String,
    pub canonical: SymbolicDimOccurrence,
    pub others: Vec<SymbolicDimOccurrence>,
}

/// One step in a fused elementwise chain.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedStep {
    pub op: FusedStepOp,
    /// Indices into the fused node's inputs: either an external input index
    /// or a previous step's output index (offset by external input count).
    pub input_indices: Vec<FusedInput>,
}

/// The operation performed by a single fusion step.
#[derive(Debug, Clone, Copy, PartialEq)]
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
}

/// Input reference within a fused chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusedInput {
    /// Index into the FusedElem node's `inputs` vec (external inputs from the DAG).
    External(usize),
    /// Output of a previous step in the chain (index into the `ops` vec).
    PreviousStep(usize),
}

/// A RISC primitive operation.
#[derive(Debug, Clone, PartialEq)]
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
        name: String,
    },
    Store {
        name: String,
    },
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
}

/// A single node in the DAG.
#[derive(Debug, Clone)]
pub struct DagNode {
    pub id: NodeId,
    pub op: RiscOp,
    pub inputs: Vec<NodeId>,
    pub output_type: TensorType,
    pub reusable_input: Option<NodeId>,
}

/// The RISC DAG — an append-only, topologically-ordered vector of [`DagNode`]s.
#[derive(Debug, Clone, Default)]
pub struct Dag {
    nodes: Vec<DagNode>,
    roots: Vec<NodeId>,
}

impl Dag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a new node and return its [`NodeId`].
    pub fn add_node(&mut self, op: RiscOp, inputs: Vec<NodeId>, output_type: TensorType) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(DagNode {
            id,
            op,
            inputs,
            output_type,
            reusable_input: None,
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

    for node in dag.nodes() {
        let RiscOp::Load { name } = &node.op else {
            continue;
        };
        if !seen_inputs.insert(name.clone()) {
            continue;
        }
        for (axis, dim) in node.output_type.dims.iter().enumerate() {
            if let DimInfo::Named(symbol, None) = dim {
                occurrences.push(SymbolicDimOccurrence {
                    name: symbol.clone(),
                    input_label: name.clone(),
                    axis,
                });
            }
        }
    }

    occurrences
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
            other => other.clone(),
        };
        let new_id = rebound.add_node(op, node.inputs.clone(), output_type);
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
        let id = dag.add_node(RiscOp::Const { value: 42.0 }, vec![], scalar_f32());
        assert_eq!(id, NodeId(0));
        assert_eq!(dag.len(), 1);
        let node = dag.get(id).unwrap();
        assert_eq!(node.op, RiscOp::Const { value: 42.0 });
        assert!(node.inputs.is_empty());
    }

    #[test]
    fn add_binary_op() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        assert_eq!(dag.len(), 3);
        let node = dag.get(c).unwrap();
        assert_eq!(node.inputs, vec![NodeId(0), NodeId(1)]);
    }

    #[test]
    fn topological_order() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Neg, vec![a], scalar_f32());
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
        let id = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_root(id);
        dag.add_root(id);
        assert_eq!(dag.roots(), &[id]);
        assert!(dag.is_root(id));
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
        dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty_x);
        dag.add_node(RiscOp::Load { name: "y".into() }, vec![], ty_y);

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
