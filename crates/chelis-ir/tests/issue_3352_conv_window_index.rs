//! chelis#3352 / [05-OP-51]: `lower_conv`'s window gather.
//!
//! The defining graph gathers each output window from the flattened padded
//! input in `(input-channel, kernel-axis-0, ..)` row order. These tests pin the
//! gather's index matrix against an independent im2col computation, check the
//! whole convolution against a direct cross-correlation, and bound the size of
//! the compile-time constants the lowering emits, which must not scale with
//! the product of contraction rows and output columns.

use chelis_unord::UnordMap;

use chelis_ir::dag::{Dag, DimInfo, NodeId, Owner, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::tier2::lower_conv;
use chelis_ir::verify;
use chelis_types::types::Prim;

struct Case {
    input: Vec<usize>,
    kernel: Vec<usize>,
    strides: Vec<usize>,
    padding: Vec<(usize, usize)>,
}

fn case(input: &[usize], kernel: &[usize], strides: &[usize], padding: &[(usize, usize)]) -> Case {
    Case {
        input: input.to_vec(),
        kernel: kernel.to_vec(),
        strides: strides.to_vec(),
        padding: padding.to_vec(),
    }
}

fn ty(dims: &[usize]) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    }
}

impl Case {
    fn rank(&self) -> usize {
        self.input.len() - 2
    }

    fn padded(&self) -> Vec<usize> {
        let mut padded = self.input.clone();
        for axis in 0..self.rank() {
            padded[axis + 2] += self.padding[axis].0 + self.padding[axis].1;
        }
        padded
    }

    fn output(&self) -> Vec<usize> {
        let padded = self.padded();
        let mut out = vec![self.input[0], self.kernel[0]];
        for axis in 0..self.rank() {
            out.push((padded[axis + 2] - self.kernel[axis + 2]) / self.strides[axis] + 1);
        }
        out
    }

    fn lower(&self) -> (Dag, NodeId) {
        let mut dag = Dag::new();
        let owner = Owner::from(dag.declare("test"));
        let x = dag.add_node(
            owner,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(&self.input),
            None,
        );
        let k = dag.add_node(
            owner,
            RiscOp::Load { name: "k".into() },
            vec![],
            ty(&self.kernel),
            None,
        );
        let root = lower_conv(
            owner,
            &mut dag,
            x,
            k,
            &ty(&self.input),
            &ty(&self.kernel),
            &ty(&self.output()),
            &self.strides,
            &self.padding,
            None,
        );
        dag.add_root(root);
        (dag, root)
    }

    /// The im2col index matrix computed directly, row-major
    /// `[C * prod(kernel), N * prod(out)]`.
    fn expected_indices(&self) -> Vec<f64> {
        let rank = self.rank();
        let padded = self.padded();
        let out = self.output();
        let kernel_volume: usize = self.kernel[2..].iter().product();
        let output_volume: usize = out[2..].iter().product();
        let mut indices = Vec::new();
        for row in 0..self.input[1] * kernel_volume {
            let channel = row / kernel_volume;
            let mut rest = row % kernel_volume;
            let mut offsets = vec![0; rank];
            for axis in (0..rank).rev() {
                offsets[axis] = rest % self.kernel[axis + 2];
                rest /= self.kernel[axis + 2];
            }
            for column in 0..self.input[0] * output_volume {
                let batch = column / output_volume;
                let mut rest = column % output_volume;
                let mut coords = vec![0; rank];
                for axis in (0..rank).rev() {
                    coords[axis] = rest % out[axis + 2];
                    rest /= out[axis + 2];
                }
                let mut index = batch * self.input[1] + channel;
                for axis in 0..rank {
                    index = index * padded[axis + 2]
                        + coords[axis] * self.strides[axis]
                        + offsets[axis];
                }
                indices.push(index as f64);
            }
        }
        indices
    }

    fn input_values(&self) -> Vec<f64> {
        let n: usize = self.input.iter().product();
        (0..n).map(|i| ((i * 7) % 11) as f64 - 5.0).collect()
    }

    fn kernel_values(&self) -> Vec<f64> {
        let n: usize = self.kernel.iter().product();
        (0..n).map(|i| ((i * 3) % 5) as f64 - 2.0).collect()
    }

    /// Direct cross-correlation over the zero-padded input. Integer-valued
    /// operands keep every partial sum exact in f32, so the result does not
    /// depend on accumulation order.
    fn expected_output(&self) -> Vec<f64> {
        let rank = self.rank();
        let padded = self.padded();
        let out = self.output();
        let x = self.input_values();
        let k = self.kernel_values();
        let in_strides = row_major_strides(&self.input);
        let k_strides = row_major_strides(&self.kernel);
        let kernel_spatial: Vec<usize> = self.kernel[2..].to_vec();
        let mut result = Vec::new();
        for n in 0..out[0] {
            for o in 0..out[1] {
                for position in multi_index(&out[2..]) {
                    let mut acc = 0.0;
                    for c in 0..self.input[1] {
                        for offset in multi_index(&kernel_spatial) {
                            let mut flat = n * in_strides[0] + c * in_strides[1];
                            let mut inside = true;
                            for axis in 0..rank {
                                let p = position[axis] * self.strides[axis] + offset[axis];
                                let low = self.padding[axis].0;
                                if p < low || p - low >= self.input[axis + 2] {
                                    inside = false;
                                    break;
                                }
                                flat += (p - low) * in_strides[axis + 2];
                            }
                            debug_assert!(padded.len() == self.input.len());
                            if inside {
                                let mut kf = o * k_strides[0] + c * k_strides[1];
                                for axis in 0..rank {
                                    kf += offset[axis] * k_strides[axis + 2];
                                }
                                acc += x[flat] * k[kf];
                            }
                        }
                    }
                    result.push(acc);
                }
            }
        }
        result
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn multi_index(extents: &[usize]) -> Vec<Vec<usize>> {
    let mut all = vec![vec![]];
    for &extent in extents {
        all = all
            .into_iter()
            .flat_map(|prefix| {
                (0..extent).map(move |i| {
                    let mut next = prefix.clone();
                    next.push(i);
                    next
                })
            })
            .collect();
    }
    all
}

fn grid() -> Vec<Case> {
    vec![
        case(&[1, 1, 4], &[1, 1, 2], &[1], &[(0, 0)]),
        case(&[2, 3, 7], &[2, 3, 3], &[2], &[(1, 2)]),
        case(&[1, 1, 5], &[1, 1, 2], &[3], &[(0, 0)]),
        case(&[1, 3, 32, 32], &[8, 3, 3, 3], &[1, 1], &[(0, 0), (0, 0)]),
        case(&[1, 2, 5, 6], &[3, 2, 3, 2], &[2, 1], &[(1, 0), (0, 2)]),
        case(&[2, 2, 4, 4], &[2, 2, 4, 4], &[1, 1], &[(0, 0), (0, 0)]),
        case(&[1, 3, 7, 7], &[4, 3, 3, 3], &[3, 2], &[(1, 1), (2, 0)]),
        case(
            &[1, 2, 3, 4, 3],
            &[2, 2, 2, 1, 2],
            &[1, 2, 1],
            &[(0, 1), (1, 0), (0, 0)],
        ),
    ]
}

fn gather_index_input(dag: &Dag) -> NodeId {
    let gathers: Vec<_> = dag
        .nodes()
        .iter()
        .filter(|node| matches!(node.op, RiscOp::Gather { .. }))
        .collect();
    assert_eq!(gathers.len(), 1, "one window gather");
    gathers[0].inputs[1]
}

fn const_tensor_elements(dag: &Dag) -> usize {
    dag.nodes()
        .iter()
        .map(|node| match &node.op {
            RiscOp::ConstTensor { data } => data.len(),
            _ => 0,
        })
        .sum()
}

fn evaluate(case: &Case, dag: &Dag) -> UnordMap<NodeId, TensorValue> {
    let x =
        TensorValue::finalize_from_wide("test", Prim::F32, case.input.clone(), case.input_values())
            .expect("input");
    let k = TensorValue::finalize_from_wide(
        "test",
        Prim::F32,
        case.kernel.clone(),
        case.kernel_values(),
    )
    .expect("kernel");
    eval_tensor(dag, &UnordMap::from([("x".into(), x), ("k".into(), k)])).expect("conv eval")
}

#[test]
fn window_gather_indices_equal_the_direct_im2col_table() {
    for case in grid() {
        let (dag, _) = case.lower();
        assert!(
            verify::verify(&dag).is_empty(),
            "{:?}",
            verify::verify(&dag)
        );
        let values = evaluate(&case, &dag);
        let index = &values[&gather_index_input(&dag)];
        let expected = case.expected_indices();
        let kernel_volume: usize = case.kernel[2..].iter().product();
        let output_volume: usize = case.output()[2..].iter().product();
        assert_eq!(
            index.shape,
            vec![case.input[1] * kernel_volume, case.input[0] * output_volume],
            "index shape for input {:?} kernel {:?}",
            case.input,
            case.kernel
        );
        assert_eq!(index.prim(), Prim::Int64);
        assert_eq!(
            index.to_f64_lossy_vec(),
            expected,
            "indices for input {:?} kernel {:?} strides {:?} padding {:?}",
            case.input,
            case.kernel,
            case.strides,
            case.padding
        );
    }
}

#[test]
fn convolution_equals_direct_cross_correlation() {
    for case in grid() {
        let (dag, root) = case.lower();
        let values = evaluate(&case, &dag);
        let result = &values[&root];
        assert_eq!(result.shape, case.output());
        assert_eq!(
            result.to_f64_lossy_vec(),
            case.expected_output(),
            "input {:?} kernel {:?} strides {:?} padding {:?}",
            case.input,
            case.kernel,
            case.strides,
            case.padding
        );
    }
}

#[test]
fn index_constants_scale_with_rows_plus_columns_not_their_product() {
    // The chelis#3352 shapes: the 3->8 3x3 probe over 32x32, and a
    // ResNet-scale 64->64 3x3 layer over 56x56 with unit padding. The direct
    // table has rows * columns elements (24,300 and 1,806,336).
    for case in [
        case(&[1, 3, 32, 32], &[8, 3, 3, 3], &[1, 1], &[(0, 0), (0, 0)]),
        case(
            &[1, 64, 56, 56],
            &[64, 64, 3, 3],
            &[1, 1],
            &[(1, 1), (1, 1)],
        ),
    ] {
        let (dag, _) = case.lower();
        let kernel_volume: usize = case.kernel[2..].iter().product();
        let output_volume: usize = case.output()[2..].iter().product();
        let rows = case.input[1] * kernel_volume;
        let columns = case.input[0] * output_volume;
        let constants = const_tensor_elements(&dag);
        assert!(
            constants <= rows + columns,
            "conv over {:?} emits {constants} constant elements; bound is rows + columns = {}",
            case.input,
            rows + columns
        );
    }
}

#[test]
fn empty_contraction_and_empty_batch_keep_their_shapes() {
    for case in [
        case(&[1, 0, 4, 4], &[2, 0, 3, 3], &[1, 1], &[(0, 0), (0, 0)]),
        case(&[0, 2, 4, 4], &[2, 2, 3, 3], &[1, 1], &[(0, 0), (0, 0)]),
    ] {
        let (dag, root) = case.lower();
        assert!(
            verify::verify(&dag).is_empty(),
            "{:?}",
            verify::verify(&dag)
        );
        let values = evaluate(&case, &dag);
        let result = &values[&root];
        assert_eq!(result.shape, case.output());
        assert_eq!(result.to_f64_lossy_vec(), case.expected_output());
    }
}
