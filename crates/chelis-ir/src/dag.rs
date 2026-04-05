//! RISC DAG data structure.
//!
//! The DAG is a flat vector of nodes, each referencing earlier nodes by [`NodeId`].
//! Nodes are always in topological order (an invariant maintained by append-only construction).

use chelis_types::types::Prim;

/// Index into the DAG node array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Tensor type carried on each DAG node.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub enum DimInfo {
    /// A named dimension with an optional known size.
    Named(String, Option<usize>),
    /// A fixed literal size.
    Lit(usize),
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
        size: usize,
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

    // --- Cast ---
    Cast {
        new_precision: Prim,
    },
}

/// A single node in the DAG.
#[derive(Debug, Clone)]
pub struct DagNode {
    pub id: NodeId,
    pub op: RiscOp,
    pub inputs: Vec<NodeId>,
    pub output_type: TensorType,
}

/// The RISC DAG — an append-only, topologically-ordered vector of [`DagNode`]s.
#[derive(Debug, Clone, Default)]
pub struct Dag {
    nodes: Vec<DagNode>,
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
        });
        id
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
}
