use crate::dag::{Bound, Dag, DimInfo, RiscOp, TensorType};

pub fn vectorize_axis0(dag: &Dag, batch_dim: DimInfo) -> Result<Dag, String> {
    let mut out = Dag::new();
    let concrete_batch = match &batch_dim {
        DimInfo::Lit(size) => Some(*size),
        DimInfo::Named(_, Some(size)) => Some(*size),
        DimInfo::Named(_, None) => None,
    };

    for node in dag.nodes() {
        let output_type = prepend_batch_type(&node.output_type, &batch_dim);
        let op = match &node.op {
            RiscOp::Sum { axis, accumulator } => RiscOp::Sum {
                axis: axis + 1,
                accumulator: *accumulator,
            },
            RiscOp::MaxReduce { axis } => RiscOp::MaxReduce { axis: axis + 1 },
            RiscOp::MinReduce { axis } => RiscOp::MinReduce { axis: axis + 1 },
            RiscOp::ProdReduce { axis } => RiscOp::ProdReduce { axis: axis + 1 },
            RiscOp::Argmax { axis } => RiscOp::Argmax { axis: axis + 1 },
            RiscOp::Argmin { axis } => RiscOp::Argmin { axis: axis + 1 },
            RiscOp::Reshape { new_shape } => RiscOp::Reshape {
                new_shape: prepend_batch_dims(new_shape, &batch_dim),
            },
            RiscOp::Permute { axes } => RiscOp::Permute {
                axes: std::iter::once(0)
                    .chain(axes.iter().map(|axis| axis + 1))
                    .collect(),
            },
            RiscOp::Expand { axis, size } => RiscOp::Expand {
                axis: axis + 1,
                size: size.clone(),
            },
            RiscOp::Pad { padding, fill } => RiscOp::Pad {
                padding: std::iter::once((Bound::Lit(0), Bound::Lit(0)))
                    .chain(padding.iter().copied())
                    .collect(),
                fill: *fill,
            },
            RiscOp::Shrink { bounds } => {
                let Some(batch) = concrete_batch else {
                    return Err(
                        "vmap over shrink requires a concrete batch size for the preserved batch axis"
                            .to_string(),
                    );
                };
                RiscOp::Shrink {
                    bounds: std::iter::once((Bound::Lit(0), Bound::Lit(batch)))
                        .chain(bounds.iter().copied())
                        .collect(),
                }
            }
            RiscOp::Stride { strides } => RiscOp::Stride {
                strides: std::iter::once(Bound::Lit(1))
                    .chain(strides.iter().copied())
                    .collect(),
            },
            RiscOp::Load { name } => RiscOp::Load { name: name.clone() },
            other => other.clone(),
        };

        // Vmap is a pure clone of the per-node operator (with axis
        // shifts) onto a new DAG. Per spec/design/chelis_span_survival.md
        // §2.3 vmap row, span_id and merged_spans are cloned unchanged
        // — every input span survives the pass.
        let new_id = out.add_node(op, node.inputs.clone(), output_type, node.span_id.clone());
        if let Some(new_node) = out.node_mut(new_id) {
            if !node.merged_spans.is_empty() {
                new_node.merged_spans = node.merged_spans.clone();
            }
            // chelis#384/#397: vmap is a 1:1 id-preserving clone, so a
            // Form-3 `expand` shape-dep maps to the same id verbatim.
            new_node.shape_deps = node.shape_deps.clone();
        }
        if let Some(reusable_input) = node.reusable_input {
            out.set_reusable_input(new_id, reusable_input);
        }
        if dag.is_root(node.id) {
            out.add_root(new_id);
        }
    }

    Ok(out)
}

fn prepend_batch_type(ty: &TensorType, batch_dim: &DimInfo) -> TensorType {
    TensorType {
        dims: prepend_batch_dims(&ty.dims, batch_dim),
        precision: ty.precision,
    }
}

fn prepend_batch_dims(dims: &[DimInfo], batch_dim: &DimInfo) -> Vec<DimInfo> {
    let mut out = Vec::with_capacity(dims.len() + 1);
    out.push(batch_dim.clone());
    out.extend(dims.iter().cloned());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{DimExpr, NodeId, RiscOp, TensorType};
    use crate::eval::{TensorValue, eval_tensor_roots_with_strict};
    use chelis_types::types::Prim;
    use std::collections::HashMap;

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn mat_f32(m: usize, n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(m), DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn batched_mat_f32(batch: usize, m: usize, n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(batch), DimInfo::Lit(m), DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn eval_root(dag: &Dag, root: NodeId, inputs: &HashMap<String, TensorValue>) -> TensorValue {
        let values = eval_tensor_roots_with_strict(dag, &[root], |name| inputs.get(name).cloned())
            .expect("evaluation should succeed");
        values[&root].clone()
    }

    #[test]
    fn elementwise_vmap_prepends_batch_axis() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3), None);
        let y = dag.add_node(RiscOp::Neg, vec![x], vec_f32(3), None);
        dag.add_root(y);

        let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
        let root = vmapped.roots()[0];
        let actual = eval_root(
            &vmapped,
            root,
            &HashMap::from([(
                "x".to_string(),
                TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, -5.0, 6.0]),
            )]),
        );
        assert_eq!(actual.shape, vec![2, 3]);
        assert_eq!(actual.data, vec![-1.0, -2.0, -3.0, -4.0, 5.0, -6.0]);
    }

    #[test]
    fn reduction_vmap_shifts_axis() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            mat_f32(2, 3),
            None,
        );
        let y = dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![x],
            vec_f32(2),
            None,
        );
        dag.add_root(y);

        let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
        let root = vmapped.roots()[0];
        let actual = eval_root(
            &vmapped,
            root,
            &HashMap::from([(
                "x".to_string(),
                TensorValue::from_vec(
                    vec![2, 2, 3],
                    vec![
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 8.0, 9.0,
                    ],
                ),
            )]),
        );
        assert_eq!(actual.shape, vec![2, 2]);
        assert_eq!(actual.data, vec![6.0, 15.0, 60.0, 24.0]);
    }

    #[test]
    fn nested_vmap_adds_two_batch_axes() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        dag.add_root(x);

        let inner = vectorize_axis0(&dag, DimInfo::Lit(3)).expect("inner vmap should succeed");
        let outer = vectorize_axis0(&inner, DimInfo::Lit(2)).expect("outer vmap should succeed");
        let root = outer.roots()[0];
        let node = outer.get(root).expect("root node");
        assert_eq!(
            node.output_type,
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            }
        );
    }

    #[test]
    fn batched_matmul_shape_matches_expand_mul_sum_pattern() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            mat_f32(3, 4),
            None,
        );
        let a_exp = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: DimExpr::Concrete(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let b_exp = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Concrete(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let prod = dag.add_node(
            RiscOp::Mul,
            vec![a_exp, b_exp],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![prod],
            mat_f32(2, 4),
            None,
        );
        dag.add_root(out);

        let vmapped = vectorize_axis0(&dag, DimInfo::Lit(5)).expect("vmap should succeed");
        let root = vmapped.roots()[0];
        let node = vmapped.get(root).expect("root");
        assert_eq!(node.output_type, batched_mat_f32(5, 2, 4));
        assert!(
            vmapped
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Expand { axis: 3, .. })),
            "expected vmap to preserve the batched matmul decomposition shape"
        );
    }
}
