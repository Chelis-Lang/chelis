use chelis_abi::metadata::ShapeMetadata;
use chelis_dag_core::dag::*;
use chelis_dag_core::fuse::reduction_inlined_products;
use chelis_dag_core::tier2_ordered::*;
use chelis_dag_core::verify;
use chelis_ir::ordered_layout::einsum_contraction;
use chelis_numeric_core::prim::Prim;

const LARGE: usize = 1 << 21;

fn ty(dims: &[DimInfo], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.to_vec(),
        precision,
    }
}

fn lit(dims: &[usize]) -> Vec<DimInfo> {
    dims.iter().map(|&n| DimInfo::Lit(n)).collect()
}

fn runtime(name: &str) -> DimInfo {
    DimInfo::Named(name.to_string(), None)
}

fn plan(equation: &str, a: &[DimInfo], b: &[DimInfo], p: Prim) -> Option<EinsumContraction> {
    einsum_contraction(equation, &ty(a, p), &ty(b, p), None)
}

/// [05-OP-33]'s one-reduction-label contractions whose output is
/// `[batch.., rows.., m, n]`, with the row operand's labels flattening
/// to one axis, at every float dtype and every accumulator it converts
/// to exactly.
#[test]
fn matmul_shaped_float_contractions_have_the_contraction_form() {
    let f32 = Prim::F32;
    for (equation, a, b) in [
        ("mk,kn->mn", lit(&[3, 37]), lit(&[37, 5])),
        ("amk,kn->amn", lit(&[2, 3, 37]), lit(&[37, 5])),
        ("mk,nk->mn", lit(&[3, 37]), lit(&[5, 37])),
        ("km,kn->mn", lit(&[37, 3]), lit(&[37, 5])),
        ("nk,mk->mn", lit(&[5, 7]), lit(&[3, 7])),
        ("bmk,bkn->bmn", lit(&[2, 3, 17]), lit(&[2, 17, 4])),
        ("bamk,bkn->bamn", lit(&[2, 5, 3, 17]), lit(&[2, 17, 4])),
        ("mak,kn->amn", lit(&[3, 2, 7]), lit(&[7, 4])),
        ("ij,jk->ki", lit(&[3, 7]), lit(&[7, 5])),
        (
            "mk,kn->mn",
            vec![runtime("m"), DimInfo::Lit(8)],
            lit(&[8, 5]),
        ),
        (
            "amk,kn->amn",
            vec![
                DimInfo::Lit(1),
                runtime("m"),
                DimInfo::Named("k".into(), Some(8)),
            ],
            vec![DimInfo::Lit(8), runtime("n")],
        ),
    ] {
        assert!(plan(equation, &a, &b, f32).is_some(), "`{equation}`");
    }
    // Reduced floats at their default f32 accumulator and every float at
    // an explicit f64 one (chelis#3783).
    for (p, accumulator) in [
        (Prim::F16, None),
        (Prim::Bf16, None),
        (Prim::F16, Some(Prim::F32)),
        (Prim::F16, Some(Prim::F64)),
        (Prim::Bf16, Some(Prim::F64)),
        (f32, Some(Prim::F64)),
        (Prim::F64, Some(Prim::F64)),
    ] {
        let contraction = einsum_contraction(
            "mk,kn->mn",
            &ty(&lit(&[3, 7]), p),
            &ty(&lit(&[7, 5]), p),
            accumulator,
        );
        assert!(contraction.is_some(), "{p:?} at {accumulator:?}");
    }
    assert!(plan("mk,kn->mn", &lit(&[3, 7]), &lit(&[7, 5]), Prim::F64).is_some());
    let explicit = einsum_contraction(
        "mk,kn->mn",
        &ty(&lit(&[3, 7]), f32),
        &ty(&lit(&[7, 5]), f32),
        Some(f32),
    );
    assert!(
        explicit.is_some(),
        "an explicit accumulator equal to the operand dtype"
    );
}

/// Everything else keeps the runtime kernel: integers, an accumulator
/// the operand dtype does not convert to exactly, diagonals, no or
/// several reduction labels, an output that does not end in a row and a
/// column, a shared label without one static extent, a lead label the
/// contraction could only reach by copying an operand, and a planned
/// shape i64 cannot hold.
#[test]
fn every_other_einsum_has_no_contraction_form() {
    let f32 = Prim::F32;
    for p in [Prim::Int32, Prim::Int64, Prim::Int8] {
        assert!(
            plan("mk,kn->mn", &lit(&[3, 7]), &lit(&[7, 5]), p).is_none(),
            "{p:?}"
        );
    }
    for (p, accumulator) in [
        (f32, Prim::F16),
        (f32, Prim::Bf16),
        (Prim::F64, f32),
        (Prim::F16, Prim::F16),
        (Prim::F16, Prim::Bf16),
        (Prim::Bf16, Prim::F16),
        (f32, Prim::Int64),
    ] {
        let narrowed = einsum_contraction(
            "mk,kn->mn",
            &ty(&lit(&[3, 7]), p),
            &ty(&lit(&[7, 5]), p),
            Some(accumulator),
        );
        assert!(narrowed.is_none(), "{p:?} at {accumulator:?}");
    }
    assert!(
        einsum_contraction(
            "mk,kn->mn",
            &ty(&lit(&[3, 7]), f32),
            &ty(&lit(&[7, 5]), Prim::F64),
            None
        )
        .is_none()
    );
    for (equation, a, b) in [
        ("ii,i->i", lit(&[4, 4]), lit(&[4])),
        ("mkk,kn->mn", lit(&[3, 4, 4]), lit(&[4, 5])),
        ("i,j->ij", lit(&[3]), lit(&[5])),
        ("ij,ij->", lit(&[3, 7]), lit(&[3, 7])),
        ("ijk,jk->i", lit(&[3, 4, 5]), lit(&[4, 5])),
        ("mjk,kn->mn", lit(&[3, 2, 7]), lit(&[7, 5])),
        ("ij,j->i", lit(&[5, 7]), lit(&[7])),
        ("bmk,bkn->mbn", lit(&[2, 3, 7]), lit(&[2, 7, 4])),
        ("mk,kn->m", lit(&[3, 7]), lit(&[7, 5])),
        (
            "mk,kn->mn",
            vec![DimInfo::Lit(3), runtime("k")],
            vec![runtime("k"), DimInfo::Lit(5)],
        ),
        (
            "bmk,bkn->bmn",
            vec![runtime("b"), DimInfo::Lit(3), DimInfo::Lit(7)],
            vec![runtime("b"), DimInfo::Lit(7), DimInfo::Lit(4)],
        ),
        ("mk,kn->mn", lit(&[3, 7]), lit(&[6, 5])),
        ("mk,kn->mn", lit(&[3, 7, 1]), lit(&[7, 5])),
        ("mk,kn", lit(&[3, 7]), lit(&[7, 5])),
        ("mK,Kn->mn", lit(&[3, 7]), lit(&[7, 5])),
        ("mk,kn->mnn", lit(&[3, 7]), lit(&[7, 5])),
        ("mk,kn->mx", lit(&[3, 7]), lit(&[7, 5])),
        // A lead label only the column operand carries.
        ("mk,bkn->bmn", lit(&[3, 9]), lit(&[2, 9, 4])),
        // A row-only lead label before a batch label.
        ("abmk,bkn->abmn", lit(&[2, 3, 4, 5]), lit(&[3, 5, 6])),
        // Rows whose static product i64 cannot hold.
        (
            "amk,kn->amn",
            lit(&[usize::MAX / 2 + 1, 2, 8]),
            lit(&[8, 5]),
        ),
        // A runtime row beside a static row above one, whose product
        // the route would compute in a consumer helper that keeps its
        // operand alive past the runtime kernel's release (chelis#3781),
        // and two runtime rows, which also bound nothing when either is
        // zero.
        (
            "amk,kn->amn",
            vec![runtime("a"), DimInfo::Lit(3), DimInfo::Lit(8)],
            lit(&[8, 5]),
        ),
        (
            "amk,kn->amn",
            vec![DimInfo::Lit(2), runtime("m"), DimInfo::Lit(8)],
            lit(&[8, 5]),
        ),
        (
            "amk,kn->amn",
            vec![runtime("a"), runtime("m"), DimInfo::Lit(8)],
            lit(&[8, 5]),
        ),
        (
            "abmk,kn->abmn",
            vec![runtime("a"), DimInfo::Lit(3), runtime("m"), DimInfo::Lit(8)],
            lit(&[8, 5]),
        ),
        // A runtime row beside a static zero, which bounds nothing.
        (
            "amk,kn->amn",
            vec![runtime("a"), DimInfo::Lit(3), DimInfo::Lit(0)],
            lit(&[0, 5]),
        ),
        // A static zero batch beside a runtime row extent.
        (
            "bmk,bkn->bmn",
            vec![DimInfo::Lit(0), runtime("m"), DimInfo::Lit(4_194_304)],
            lit(&[0, 4_194_304, 4_194_304]),
        ),
        // An empty batch whose expansion strides exceed i64.
        (
            "bmk,bkn->bmn",
            lit(&[0, 4_194_304, 4_194_304]),
            lit(&[0, 4_194_304, 4_194_304]),
        ),
    ] {
        assert!(plan(equation, &a, &b, f32).is_none(), "`{equation}`");
    }
}

fn lowered(
    equation: &str,
    a: &[DimInfo],
    b: &[DimInfo],
    p: Prim,
    plan: &EinsumContraction,
) -> (Dag, NodeId) {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let a_ty = ty(a, p);
    let b_ty = ty(b, p);
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        a_ty.clone(),
        None,
    );
    let y = dag.add_node(
        owner,
        RiscOp::Load { name: "y".into() },
        vec![],
        b_ty.clone(),
        None,
    );
    let root = lower_einsum_contraction(owner, &mut dag, plan, [(x, &a_ty), (y, &b_ty)], None)
        .expect("lowered");
    dag.set_roots(vec![root]);
    assert!(
        verify::verify(&dag).is_empty(),
        "`{equation}`: {:?}",
        verify::verify(&dag)
    );
    (dag, root)
}

/// The evaluator reads the contraction's expansions and widenings in
/// place and computes the bits of [05-OP-33]'s own graph, which
/// [`lower_einsum`] builds by storing each conversion, at every dtype and
/// accumulator, over ties, signed zeros, infinities and NaN.
#[test]
fn the_contraction_evaluates_to_the_einsum_graph_bits() {
    let (f16, bf16, f32, f64) = (Prim::F16, Prim::Bf16, Prim::F32, Prim::F64);
    let special = [f64::INFINITY, f64::NAN, -0.0, 65504.0, 1.0 + 1.0 / 2048.0];
    let values = |len: usize, salt: usize| -> Vec<f64> {
        (0..len)
            .map(|i| match (i * 7 + salt) % 23 {
                0..=4 if salt % 2 == 1 => special[(i + salt) % special.len()],
                k => (k as f64 - 11.0) / 8.0 + (i % 5) as f64 / 4096.0,
            })
            .collect()
    };
    let mut compared = 0usize;
    for (p, accumulator) in [
        (f32, None),
        (f64, None),
        (f16, None),
        (bf16, None),
        (f16, Some(f64)),
        (bf16, Some(f64)),
        (f32, Some(f64)),
    ] {
        for (salt, (equation, a, b)) in [
            ("amk,kn->amn", lit(&[2, 3, 37]), lit(&[37, 5])),
            ("nk,mk->mn", lit(&[5, 9]), lit(&[3, 9])),
            ("bmk,bkn->bmn", lit(&[2, 3, 17]), lit(&[2, 17, 4])),
            ("mak,kn->amn", lit(&[3, 2, 300]), lit(&[300, 2])),
        ]
        .into_iter()
        .enumerate()
        {
            let contraction = einsum_contraction(equation, &ty(&a, p), &ty(&b, p), accumulator)
                .expect("contraction form");
            let count = |dims: &[DimInfo]| dims.iter().filter_map(static_extent).product();
            let input = |dims: &[DimInfo], salt: usize| {
                let storage = chelis_numeric_core::dtype_semantics::finalize_tensor(
                    "input",
                    p,
                    chelis_numeric_core::dtype_semantics::RawTensor::Float(values(
                        count(dims),
                        salt,
                    )),
                )
                .expect("finalized input");
                chelis_ir::eval::TensorValue::from_storage(
                    dims.iter().filter_map(static_extent).collect(),
                    storage,
                )
            };
            let inputs = [("x", input(&a, salt)), ("y", input(&b, salt + 1))];
            let load = |name: &str| {
                inputs
                    .iter()
                    .find(|(own, _)| *own == name)
                    .map(|(_, value)| value.clone())
            };
            let (routed, root) = lowered(equation, &a, &b, p, &contraction);
            let routed = chelis_ir::eval::eval_tensor_roots_exact(&routed, &[root], load)
                .expect("routed evaluation")[&root]
                .clone();
            let mut dag = Dag::new();
            let owner = Owner::from(dag.declare("reference"));
            let x = dag.add_node(
                owner,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty(&a, p),
                None,
            );
            let y = dag.add_node(
                owner,
                RiscOp::Load { name: "y".into() },
                vec![],
                ty(&b, p),
                None,
            );
            let reference = lower_einsum(
                owner,
                &mut dag,
                equation,
                [(x, &ty(&a, p)), (y, &ty(&b, p))],
                accumulator,
                None,
            )
            .ok()
            .expect("reference graph");
            dag.set_roots(vec![reference]);
            let reference = chelis_ir::eval::eval_tensor_roots_exact(&dag, &[reference], load)
                .expect("reference evaluation")[&reference]
                .clone();
            assert_eq!(routed.shape, reference.shape, "`{equation}` {p:?}");
            for i in 0..reference.len() {
                assert_eq!(
                    format!("{:?}", routed.storage().scalar_at(i)),
                    format!("{:?}", reference.storage().scalar_at(i)),
                    "`{equation}` {p:?} at {accumulator:?}, element {i}"
                );
                compared += 1;
            }
        }
    }
    assert!(compared > 500, "{compared} elements compared");
}

/// Every two-operand equation over up to four labels, each operand of
/// rank one to three.
fn equations() -> Vec<String> {
    let letters = *b"abcd";
    let mut sequences = vec![Vec::new()];
    for _ in 0..3 {
        let mut longer = Vec::new();
        for sequence in &sequences {
            for &letter in &letters {
                if !sequence.contains(&letter) {
                    let mut next: Vec<u8> = sequence.clone();
                    next.push(letter);
                    longer.push(next);
                }
            }
        }
        sequences.extend(
            longer
                .iter()
                .filter(|s| !sequences.contains(s))
                .cloned()
                .collect::<Vec<_>>(),
        );
    }
    sequences.sort();
    sequences.dedup();
    let operands: Vec<&Vec<u8>> = sequences.iter().filter(|s| !s.is_empty()).collect();
    let mut out = Vec::new();
    for a in &operands {
        for b in &operands {
            let union: Vec<u8> = letters
                .iter()
                .copied()
                .filter(|l| a.contains(l) || b.contains(l))
                .collect();
            for output in sequences
                .iter()
                .filter(|s| s.iter().all(|l| union.contains(l)))
            {
                out.push(format!(
                    "{},{}->{}",
                    String::from_utf8_lossy(a),
                    String::from_utf8_lossy(b),
                    String::from_utf8_lossy(output)
                ));
            }
        }
    }
    out
}

/// Every node's actual shape, and every rank-zero integer's value, with
/// each runtime extent at `binding` of its name: a flattened row axis then
/// has the product of its factors, whatever its declared dimension says.
/// Magnitudes are `f64`, exact for the extents the oracles draw.
fn concrete_shapes(dag: &Dag, binding: &dyn Fn(&str) -> f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    concrete_node_shapes(dag.nodes(), binding)
}

fn concrete_node_shapes(
    nodes: &[chelis_dag_core::dag::DagNode],
    binding: &dyn Fn(&str) -> f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut shapes: Vec<Vec<f64>> = Vec::with_capacity(nodes.len());
    let mut scalars: Vec<Option<f64>> = Vec::with_capacity(nodes.len());
    let extent = |dim: &DimInfo| match dim {
        DimInfo::Named(name, None) => binding(name),
        other => static_extent(other).expect("static extent") as f64,
    };
    for node in nodes {
        let input = |slot: usize| node.inputs[slot].0;
        let bound = |dim: &RtDim, shapes: &[Vec<f64>], scalars: &[Option<f64>]| match dim {
            RtDim::Lit(n) => *n as f64,
            RtDim::Sym(name) => binding(name),
            RtDim::Node(slot) => scalars[input(*slot)].expect("scalar bound"),
            RtDim::InputAxis {
                tensor,
                axis: RtAxis::Lit(axis),
            } => shapes[input(*tensor)][*axis as usize],
            RtDim::ToEnd => panic!("no contraction bound reads to the end"),
        };
        let (shape, scalar) = match &node.op {
            RiscOp::Load { .. } => (node.output_type.dims.iter().map(extent).collect(), None),
            RiscOp::Permute { axes } => (
                axes.iter().map(|&axis| shapes[input(0)][axis]).collect(),
                None,
            ),
            RiscOp::Reshape { new_shape } => (
                new_shape
                    .iter()
                    .map(|dim| bound(dim, &shapes, &scalars))
                    .collect(),
                None,
            ),
            RiscOp::Expand { axis, size } => {
                let mut shape = shapes[input(0)].clone();
                shape.insert(*axis, bound(size, &shapes, &scalars));
                (shape, None)
            }
            RiscOp::Sum { axis, .. } => {
                let mut shape = shapes[input(0)].clone();
                shape.remove(*axis);
                (shape, None)
            }
            RiscOp::Shape { axis } => (Vec::new(), Some(shapes[input(0)][*axis])),
            RiscOp::Mul if node.output_type.dims.is_empty() => (
                Vec::new(),
                Some(scalars[input(0)].unwrap() * scalars[input(1)].unwrap()),
            ),
            RiscOp::Mul | RiscOp::Cast { .. } | RiscOp::Drop => (shapes[input(0)].clone(), None),
            other => panic!("the contraction graph has no {other:?}"),
        };
        shapes.push(shape);
        scalars.push(scalar);
    }
    let scalars = scalars.into_iter().flatten().collect();
    (shapes, scalars)
}

/// The bytes the C lane's verified storage plan holds for the lowered
/// contraction, every slot at its largest owner, with each runtime
/// extent at `binding`, and the bound the runtime kernel sets: the result,
/// plus each operand the graph must permute (chelis#3802).
/// `dag` with its root `root` read by `consumer`: a sum over the last
/// axis, a product with a loaded tensor then that sum, an inserted axis
/// then a sum over it, or a flattening reshape then a sum. Each keeps the
/// contraction's output a private input of another node, which an inlined
/// product may claim.
fn with_consumer(dag: &Dag, root: NodeId, consumer: &str) -> Dag {
    let mut dag = dag.clone();
    let owner = dag.get(root).unwrap().owner;
    let out = dag.get(root).unwrap().output_type.clone();
    let rank = out.dims.len();
    let accumulator = out.precision.default_reduce_sum_accumulator().unwrap();
    let sum = |dag: &mut Dag, input: NodeId, axis: usize| {
        let mut dims = dag.get(input).unwrap().output_type.dims.clone();
        dims.remove(axis);
        dag.add_node(
            owner,
            RiscOp::Sum { axis, accumulator },
            vec![input],
            ty(&dims, accumulator),
            None,
        )
    };
    let top = match consumer {
        "sum" => sum(&mut dag, root, rank - 1),
        "mul-sum" => {
            let v = dag.add_node(
                owner,
                RiscOp::Load { name: "v".into() },
                vec![],
                out.clone(),
                None,
            );
            let product = dag.add_node(owner, RiscOp::Mul, vec![root, v], out.clone(), None);
            sum(&mut dag, product, rank - 1)
        }
        "insert-sum" => {
            let mut dims = vec![DimInfo::Lit(2)];
            dims.extend(out.dims.clone());
            let inserted = dag.add_node(
                owner,
                RiscOp::Expand {
                    axis: 0,
                    size: RtDim::Lit(2),
                },
                vec![root],
                ty(&dims, out.precision),
                None,
            );
            sum(&mut dag, inserted, 0)
        }
        _ => {
            let flat = DimInfo::Named("flat".into(), None);
            let reshaped = dag.add_node(
                owner,
                RiscOp::Reshape {
                    new_shape: vec![RtDim::Sym("flat".into())],
                },
                vec![root],
                ty(&[flat], out.precision),
                None,
            );
            sum(&mut dag, reshaped, 0)
        }
    };
    dag.set_roots(vec![top]);
    dag
}

/// The C storage plan of `dag` builds, so every node an inlined product
/// claims has exactly one storage decision (chelis#3802).
fn assert_c_storage_plans(dag: &Dag, context: &str) {
    let program = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(dag.clone()).expect("consumer lowers"),
    )
    .expect("consumer ownership verifies");
    if let Err(error) = chelis_ir::ownership::plan_c_storage(program) {
        panic!("{context}: {error}");
    }
}

fn c_storage_bytes(dag: &Dag, binding: f64) -> StorageBytes {
    let program = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(dag.clone()).expect("contraction lowers"),
    )
    .expect("contraction ownership verifies");
    let plan = chelis_ir::ownership::plan_c_storage(program).expect("C storage plan");
    let nodes = plan.emission().nodes();
    let (shapes, _) = concrete_node_shapes(nodes, &|_| binding);
    let bytes = |node: &chelis_dag_core::dag::DagNode| -> f64 {
        let width = node
            .output_type
            .precision
            .runtime_dtype()
            .unwrap()
            .byte_width();
        shapes[node.id.0].iter().product::<f64>() * width as f64
    };
    let mut slots = std::collections::BTreeMap::new();
    // A computed row extent's rank-zero scalars are metadata, not data.
    for node in nodes
        .iter()
        .filter(|node| !node.output_type.dims.is_empty())
    {
        if let Some(slot) = plan.slot_for_node(node.id) {
            let held = slots.entry(slot).or_insert(0f64);
            *held = held.max(bytes(node));
        }
    }
    let root = nodes
        .iter()
        .find(|node| plan.emission().roots().contains(&node.id))
        .expect("root");
    // `chelis_tensor_einsum` allocates its result and, for a nonempty
    // result, one accumulator per product of an output's reduction.
    let sum = nodes
        .iter()
        .find(|node| matches!(node.op, RiscOp::Sum { .. }))
        .expect("product sum");
    let RiscOp::Sum { axis, .. } = sum.op else {
        unreachable!()
    };
    let reduction = shapes[sum.inputs[0].0][axis];
    let accumulator = sum
        .output_type
        .precision
        .runtime_dtype()
        .unwrap()
        .byte_width() as f64;
    let result = bytes(root);
    let kernel = if result > 0.0 {
        result + reduction * accumulator
    } else {
        0.0
    };
    let permuted: f64 = nodes
        .iter()
        .filter(|node| matches!(node.op, RiscOp::Permute { .. }))
        .map(bytes)
        .sum();
    StorageBytes {
        held: slots.values().sum(),
        kernel,
        permuted,
        computed_rows: nodes
            .iter()
            .any(|node| matches!(node.op, RiscOp::Shape { .. })),
    }
}

/// What the C lane holds for a lowered contraction, and what the runtime
/// kernel would hold for the same case.
struct StorageBytes {
    held: f64,
    kernel: f64,
    permuted: f64,
    computed_rows: bool,
}

impl StorageBytes {
    /// The route holds no more than the runtime kernel. A permuted
    /// operand's copy is admitted only on forms routed before chelis#3781
    /// (static rows or one runtime row beside extents of one), and a
    /// computed row product permutes nothing (chelis#3832 round 1).
    fn check(&self, context: &str) {
        assert!(
            !(self.computed_rows && self.permuted > 0.0),
            "{context}: a computed row product permutes {} bytes",
            self.permuted
        );
        let bound = self.kernel
            + if self.computed_rows {
                0.0
            } else {
                self.permuted
            };
        assert!(
            self.held <= bound,
            "{context}: the C lane holds {} bytes, over the runtime kernel's {} (+{} permuted)",
            self.held,
            self.kernel,
            self.permuted
        );
    }
}

/// Whether the runtime would allocate `shape` at `precision`: its
/// strides, element count and byte size fit i64.
fn allocatable(shape: &[f64], precision: Prim) -> bool {
    let limit = i64::MAX as f64;
    let extents: Option<Vec<i64>> = shape
        .iter()
        .map(|&n| (n <= limit).then_some(n as i64))
        .collect();
    extents.is_some_and(|extents| {
        ShapeMetadata::contiguous(&extents, precision.runtime_dtype().unwrap()).is_ok()
    })
}

/// For every assignment of 0, 3 and 2^62 to `a` and `b`'s runtime extents
/// under which both operands and the einsum's result are allocatable, the
/// lowered graph traps nowhere the runtime kernel does not: every computed
/// extent fits i64 and every planned shape is allocatable. Returns the
/// number of assignments checked.
fn assert_existing_operands_plan_no_trap(
    equation: &str,
    a: &[DimInfo],
    b: &[DimInfo],
    p: Prim,
    accumulator: Option<Prim>,
    dag: &Dag,
    root: NodeId,
) -> usize {
    let mut names: Vec<&str> = a
        .iter()
        .chain(b)
        .filter_map(|dim| match dim {
            DimInfo::Named(name, None) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    names.sort_unstable();
    names.dedup();
    let values = [0.0, 3.0, (1u64 << 62) as f64];
    let result = p.sum_result_precision(accumulator.unwrap_or_else(|| {
        p.default_reduce_sum_accumulator()
            .expect("float accumulator")
    }));
    let mut checked = 0;
    for choice in 0..values.len().pow(names.len() as u32) {
        let value = |name: &str| {
            let at = names
                .iter()
                .position(|own| *own == name)
                .expect("bound name");
            values[choice / values.len().pow(at as u32) % values.len()]
        };
        let (shapes, scalars) = concrete_shapes(dag, &value);
        let load = |dims: &[DimInfo]| -> Vec<f64> {
            dims.iter()
                .map(|dim| match dim {
                    DimInfo::Named(name, None) => value(name),
                    other => static_extent(other).unwrap() as f64,
                })
                .collect()
        };
        if !allocatable(&load(a), p)
            || !allocatable(&load(b), p)
            || !allocatable(&shapes[root.0], result)
        {
            continue;
        }
        checked += 1;
        for scalar in &scalars {
            assert!(
                *scalar <= i64::MAX as f64,
                "`{equation}` {a:?} {b:?} at {choice}: a computed row extent of {scalar} overflows i64"
            );
        }
        for node in dag.nodes() {
            if !node.output_type.dims.is_empty() {
                assert!(
                    allocatable(&shapes[node.id.0], node.output_type.precision),
                    "`{equation}` {a:?} {b:?} at {choice}: {:?} plans unallocatable {:?}",
                    node.op,
                    shapes[node.id.0]
                );
            }
        }
    }
    checked
}

/// The flattened-row class over five labels, which [`equations`] cannot
/// reach: up to three row factors, each drawn from zero, one, three, a
/// runtime extent and a large one, beside a static or runtime column.
/// Every accepted case plans no trap for operands that exist and holds no
/// more C storage than the runtime kernel, and rows with two runtime
/// factors keep the runtime kernel.
#[test]
fn every_flattened_row_plans_no_trap_for_operands_that_exist() {
    let extents = [
        DimInfo::Lit(0),
        DimInfo::Lit(1),
        DimInfo::Lit(3),
        runtime("r"),
        DimInfo::Lit(LARGE),
    ];
    let (mut accepted, mut assignments, mut refused) = (0usize, 0usize, 0usize);
    let mut consumed = 0usize;
    for equation in [
        "abmk,kn->abmn",
        "bamk,kn->abmn",
        "abkm,kn->abmn",
        "mabk,kn->abmn",
        "abmk,nk->abmn",
        "kn,abmk->abmn",
    ] {
        let (inputs, _) = equation.split_once("->").unwrap();
        let (la, lb) = inputs.split_once(',').unwrap();
        for choice in 0..extents.len().pow(3) * 2 {
            let pick = |label: u8| -> DimInfo {
                let dim = match label {
                    b'a' => extents[choice % 5].clone(),
                    b'b' => extents[choice / 5 % 5].clone(),
                    b'm' => extents[choice / 25 % 5].clone(),
                    b'n' if choice / 125 == 1 => runtime("r"),
                    b'n' => DimInfo::Lit(5),
                    _ => DimInfo::Lit(8),
                };
                match dim {
                    DimInfo::Named(_, None) => runtime(&format!("r{}", char::from(label))),
                    dim => dim,
                }
            };
            let a: Vec<DimInfo> = la.bytes().map(pick).collect();
            let b: Vec<DimInfo> = lb.bytes().map(pick).collect();
            let runtime_rows = b"abm"
                .iter()
                .filter(|&&l| static_extent(&pick(l)).is_none())
                .count();
            let static_rows: usize = b"abm"
                .iter()
                .filter_map(|&l| static_extent(&pick(l)))
                .product();
            let computed = runtime_rows >= 2 || (runtime_rows == 1 && static_rows != 1);
            for p in [Prim::F32, Prim::F16] {
                let Some(contraction) = einsum_contraction(equation, &ty(&a, p), &ty(&b, p), None)
                else {
                    refused += usize::from(computed);
                    continue;
                };
                assert!(
                    !computed,
                    "`{equation}` {a:?} {b:?} routes a computed row product"
                );
                accepted += 1;
                let (dag, root) = lowered(equation, &a, &b, p, &contraction);
                assignments +=
                    assert_existing_operands_plan_no_trap(equation, &a, &b, p, None, &dag, root);
                for consumer in ["sum", "mul-sum", "insert-sum", "reshape-sum"] {
                    assert_c_storage_plans(
                        &with_consumer(&dag, root, consumer),
                        &format!("`{equation}` {a:?} {b:?} {p:?} under {consumer}"),
                    );
                    consumed += 1;
                }
                for binding in [0.0f64, 7.0] {
                    let shape = |dims: &[DimInfo]| -> Vec<f64> {
                        dims.iter()
                            .map(|dim| static_extent(dim).map_or(binding, |n| n as f64))
                            .collect()
                    };
                    if allocatable(&shape(&a), p) && allocatable(&shape(&b), p) {
                        c_storage_bytes(&dag, binding)
                            .check(&format!("`{equation}` {a:?} {b:?} {p:?} at {binding}"));
                    }
                }
            }
        }
    }
    assert!(
        accepted > 1000 && assignments > 1000 && refused > 1000 && consumed > 4000,
        "{accepted} accepted, {assignments} assignments, {refused} computed-row refusals"
    );
}

/// The class oracle over the classifier's own acceptances: for every
/// generated equation, operand dtype and accumulator, and per-label
/// extents drawn from zero, one, three, a runtime extent and a large one,
/// an accepted contraction lowers to one product sum whose planned shapes
/// fit i64, which has the einsum's output shape, and which stores no node
/// larger in bytes than an operand or the result. Sizes are each node's
/// actual shape with a runtime extent at 0 and at 7, and beside a static
/// zero also near i64's limit. Its runtime extents are also bound
/// together to every mix of 0, 3 and 2^62 under which the operands
/// exist, and the graph must then plan no trap.
#[test]
fn every_accepted_contraction_stores_no_more_than_its_operands_or_result() {
    let precisions = [
        (Prim::F16, None),
        (Prim::Bf16, None),
        (Prim::F32, None),
        (Prim::F64, None),
        (Prim::F32, Some(Prim::F64)),
        (Prim::F16, Some(Prim::F64)),
        (Prim::Bf16, Some(Prim::F64)),
        (Prim::F16, Some(Prim::F32)),
        (Prim::F64, Some(Prim::F64)),
    ];
    let extents = [
        DimInfo::Lit(0),
        DimInfo::Lit(1),
        DimInfo::Lit(3),
        runtime("r"),
        DimInfo::Lit(LARGE),
    ];
    let mut state = 0x3777_u64;
    let mut accepted = 0usize;
    let mut converted = 0usize;
    let mut peak_checked = 0usize;
    for equation in equations() {
        let (inputs, output) = equation.split_once("->").unwrap();
        let (la, lb) = inputs.split_once(',').unwrap();
        for _ in 0..4 {
            let mut extent = std::collections::BTreeMap::new();
            for label in la.bytes().chain(lb.bytes()) {
                extent.entry(label).or_insert_with(|| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let mut dim = extents[(state >> 33) as usize % extents.len()].clone();
                    if let DimInfo::Named(_, None) = dim {
                        dim = runtime(&format!("r{}", char::from(label)));
                    }
                    dim
                });
            }
            let a: Vec<DimInfo> = la.bytes().map(|l| extent[&l].clone()).collect();
            let b: Vec<DimInfo> = lb.bytes().map(|l| extent[&l].clone()).collect();
            for &(p, accumulator) in &precisions {
                let Some(contraction) =
                    einsum_contraction(&equation, &ty(&a, p), &ty(&b, p), accumulator)
                else {
                    continue;
                };
                accepted += 1;
                let (dag, root) = lowered(&equation, &a, &b, p, &contraction);
                let inlined = reduction_inlined_products(&dag);
                assert_eq!(inlined.len(), 1, "`{equation}` {a:?} {b:?} {p:?}");
                let skipped = inlined[0].skipped();
                converted += usize::from(inlined[0].casts.iter().any(Option::is_some));
                let product = inlined[0].product;
                for binding in [0.0f64, 7.0] {
                    let (shapes, _) = concrete_shapes(&dag, &|_| binding);
                    let bytes = |id: NodeId| -> f64 {
                        let node = dag.get(id).unwrap();
                        let width = node
                            .output_type
                            .precision
                            .runtime_dtype()
                            .unwrap()
                            .byte_width();
                        shapes[id.0].iter().product::<f64>() * width as f64
                    };
                    let operand = |dims: &[DimInfo]| -> f64 {
                        let width = p.runtime_dtype().unwrap().byte_width() as f64;
                        dims.iter()
                            .map(|dim| static_extent(dim).map_or(binding, |n| n as f64))
                            .product::<f64>()
                            * width
                    };
                    let expected: Vec<f64> = output
                        .bytes()
                        .map(|l| static_extent(&extent[&l]).map_or(binding, |n| n as f64))
                        .collect();
                    assert_eq!(
                        shapes[root.0], expected,
                        "`{equation}` {a:?} {b:?} output shape"
                    );
                    let bound = operand(&a).max(operand(&b)).max(bytes(root));
                    for node in dag.nodes() {
                        if skipped.contains(&node.id) || node.output_type.dims.is_empty() {
                            continue;
                        }
                        assert!(
                            bytes(node.id) <= bound,
                            "`{equation}` {a:?} {b:?} {p:?}/{accumulator:?} stores {:?} of {} bytes, over {bound}",
                            node.op,
                            bytes(node.id)
                        );
                    }
                    // Every stride and the byte size of the planned
                    // expansion fit i64, a runtime extent counting as
                    // at least one.
                    let mut suffix = 1f64;
                    let (strides, _) = concrete_shapes(&dag, &|_| binding.max(1.0));
                    for &extent in strides[product.0].iter().rev() {
                        suffix *= extent;
                        assert!(
                            suffix <= i64::MAX as f64,
                            "`{equation}` {a:?} {b:?} expansion stride"
                        );
                    }
                    assert!(
                        bytes(product) <= i64::MAX as f64,
                        "`{equation}` {a:?} {b:?} expansion bytes"
                    );
                }
                assert_existing_operands_plan_no_trap(
                    &equation,
                    &a,
                    &b,
                    p,
                    accumulator,
                    &dag,
                    root,
                );
                // The C lane holds no more than the runtime kernel: the
                // row operand's flattening reshape is a view and the
                // output reshape holds the total directly.
                let exists = |dims: &[DimInfo], binding: f64| {
                    let shape: Vec<f64> = dims
                        .iter()
                        .map(|dim| static_extent(dim).map_or(binding, |n| n as f64))
                        .collect();
                    allocatable(&shape, p)
                };
                // Storage depends on dtype only through widths, so one
                // reduced, one native and one widened pair suffice.
                let representative = matches!(
                    (p, accumulator),
                    (Prim::F16, None) | (Prim::F32, None) | (Prim::F32, Some(Prim::F64))
                );
                for binding in [0.0f64, 7.0] {
                    if !representative || !exists(&a, binding) || !exists(&b, binding) {
                        continue;
                    }
                    peak_checked += 1;
                    c_storage_bytes(&dag, binding).check(&format!(
                        "`{equation}` {a:?} {b:?} {p:?}/{accumulator:?} at {binding}"
                    ));
                }
                // Without a static zero, the operands' existence in memory
                // bounds every runtime extent. A static zero removes that
                // bound, so then every planned shape's strides must fit
                // with runtime extents near i64's limit.
                if a.iter().chain(&b).any(|dim| static_extent(dim) == Some(0)) {
                    let near_limit = (1u64 << 62) as f64;
                    for node in dag.nodes() {
                        let mut suffix = 1f64;
                        for dim in node.output_type.dims.iter().rev() {
                            suffix *= static_extent(dim).map_or(near_limit, |n| n as f64);
                            assert!(
                                suffix <= i64::MAX as f64,
                                "`{equation}` {a:?} {b:?} {:?} stride beside a static zero",
                                node.op
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        converted > 10_000 && peak_checked > 10_000,
        "the oracle reached {converted} conversions and {peak_checked} storage plans"
    );
    assert!(
        accepted > 1000,
        "the oracle reached {accepted} accepted cases"
    );
}

fn static_extent(dim: &DimInfo) -> Option<usize> {
    match dim {
        DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => Some(*n),
        DimInfo::Named(_, None) => None,
    }
}
