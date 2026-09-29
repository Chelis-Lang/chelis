//! chelis#2491, [05-OP-33]: the DAG evaluator admits every result it sizes
//! from extents before allocating it. An unrepresentable count, byte size, or
//! stride product is the operation's typed `Overflow` trap, rendered as the C
//! runtime's operation plans render it; a representable result whose
//! evaluator scratch cannot be allocated is the C runtime's allocation
//! failure. None of them is a `capacity overflow` or arithmetic panic.
//!
//! Every test below is a regression test: on the base it panicked (or, where
//! noted, returned a value the compiled lane refuses). The two `*_control`
//! tests are disposition locks that already passed there; they show the
//! admission leaves ordinary extents alone.

use chelis_ir::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, RtDim, TensorType, UniformBound};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with, eval_tensor_roots_with_strict};
use chelis_types::dtype_semantics::{RawTensor, finalize_tensor, scalar_from_i64};
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

const HUGE: usize = 1 << 62;

fn named(name: &str) -> DimInfo {
    DimInfo::Named(name.into(), None)
}

fn lit(extents: &[usize]) -> Vec<DimInfo> {
    extents.iter().map(|extent| DimInfo::Lit(*extent)).collect()
}

fn add(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    op: RiscOp,
    inputs: Vec<NodeId>,
    dims: Vec<DimInfo>,
    prim: Prim,
) -> NodeId {
    dag.add_node(
        decl,
        op,
        inputs,
        TensorType {
            dims,
            precision: prim,
        },
        None,
    )
}

fn load(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    name: &str,
    dims: Vec<DimInfo>,
    prim: Prim,
) -> NodeId {
    add(
        dag,
        decl,
        RiscOp::Load { name: name.into() },
        vec![],
        dims,
        prim,
    )
}

fn i64_const(dag: &mut Dag, decl: chelis_ir::dag::DeclId, value: i64) -> NodeId {
    let value = scalar_from_i64("test", Prim::Int64, value).unwrap();
    add(
        dag,
        decl,
        RiscOp::Const { value },
        vec![],
        vec![],
        Prim::Int64,
    )
}

fn floats(prim: Prim, shape: &[usize], data: Vec<f64>) -> TensorValue {
    let storage = finalize_tensor("test", prim, RawTensor::Float(data)).unwrap();
    TensorValue::from_storage(shape.to_vec(), storage)
}

fn ints(prim: Prim, shape: &[usize], data: Vec<i64>) -> TensorValue {
    let storage = finalize_tensor("test", prim, RawTensor::Int(data)).unwrap();
    TensorValue::from_storage(shape.to_vec(), storage)
}

fn count(value: i64) -> TensorValue {
    ints(Prim::Int64, &[], vec![value])
}

/// Evaluate `dag`'s roots over `inputs`; `strict` refuses a missing input.
fn run(
    dag: &Dag,
    inputs: Vec<(&str, TensorValue)>,
    strict: bool,
) -> Result<Vec<TensorValue>, String> {
    let inputs = inputs.into_iter().collect::<UnordMap<_, _>>();
    let load = |name: &str| inputs.get(name).cloned();
    let out = if strict {
        eval_tensor_roots_with_strict(dag, dag.roots(), load)?
    } else {
        eval_tensor_roots_with(dag, dag.roots(), load)?
    };
    Ok(dag.roots().iter().map(|root| out[root].clone()).collect())
}

fn trap(op: &str, reason: &str) -> String {
    format!("Overflow: {reason}\nnumeric trap: overflow in {op} at i64")
}

const ALLOCATION_FAILED: &str = "Domain: chelis_alloc tensor allocation failed";

/// `split_keys(key(7), n)`, the count read at run time.
fn runtime_split() -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let seed = i64_const(&mut dag, decl, 7);
    let key = add(
        &mut dag,
        decl,
        RiscOp::KeyFromSeed,
        vec![seed],
        vec![],
        Prim::Key,
    );
    let n = load(&mut dag, decl, "n", vec![], Prim::Int64);
    let rows = add(
        &mut dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Node(1),
        },
        vec![key, n],
        vec![named("keys")],
        Prim::Key,
    );
    dag.add_root(rows);
    dag
}

#[test]
fn a_runtime_split_count_past_the_key_byte_domain_traps_overflow() {
    // 2^60 keys already need 2^63 bytes; i64::MAX keys fit the count and not
    // the bytes.
    for n in [1i64 << 60, 1 << 61, i64::MAX] {
        assert_eq!(
            run(&runtime_split(), vec![("n", count(n))], true).unwrap_err(),
            trap("split_keys", "byte size exceeds i64"),
            "n = {n}"
        );
    }
}

#[test]
fn a_literal_split_count_past_the_key_byte_domain_traps_overflow() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let seed = i64_const(&mut dag, decl, 7);
    let key = add(
        &mut dag,
        decl,
        RiscOp::KeyFromSeed,
        vec![seed],
        vec![],
        Prim::Key,
    );
    let rows = add(
        &mut dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Lit(1 << 61),
        },
        vec![key],
        lit(&[1 << 61]),
        Prim::Key,
    );
    dag.add_root(rows);
    assert_eq!(
        run(&dag, vec![], true).unwrap_err(),
        trap("split_keys", "byte size exceeds i64")
    );
}

/// `split_keys(key_from_seed(s), n)` over a key batch shaped like `s`.
fn batched_split(batch: Vec<DimInfo>) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let seeds = load(&mut dag, decl, "s", batch.clone(), Prim::Int64);
    let keys = add(
        &mut dag,
        decl,
        RiscOp::KeyFromSeed,
        vec![seeds],
        batch.clone(),
        Prim::Key,
    );
    let n = load(&mut dag, decl, "n", vec![], Prim::Int64);
    let mut dims = batch;
    dims.push(named("keys"));
    let rows = add(
        &mut dag,
        decl,
        RiscOp::SplitN {
            count: RtDim::Node(1),
        },
        vec![keys, n],
        dims,
        Prim::Key,
    );
    dag.add_root(rows);
    dag
}

#[test]
fn a_key_batch_times_its_split_count_past_i64_traps_overflow() {
    let dag = batched_split(lit(&[4]));
    let inputs = vec![
        ("s", ints(Prim::Int64, &[4], vec![1, 2, 3, 4])),
        ("n", count(1 << 62)),
    ];
    assert_eq!(
        run(&dag, inputs, true).unwrap_err(),
        trap("split_keys", "extent product exceeds i64")
    );
}

/// An empty key batch splits into no keys, but `[0, 2^40, 2^40]` has a stride
/// product past i64, which the compiled lane refuses to allocate. On the base
/// the evaluator returned the empty tensor.
#[test]
fn an_empty_key_batch_split_past_the_stride_domain_traps_overflow() {
    let dag = batched_split(vec![DimInfo::Lit(0), named("q")]);
    let inputs = vec![
        ("s", ints(Prim::Int64, &[0, 1 << 40], vec![])),
        ("n", count(1 << 40)),
    ];
    assert_eq!(
        run(&dag, inputs, true).unwrap_err(),
        trap("split_keys", "stride product exceeds i64")
    );
}

/// `expand` of `x` along axis 0 to the run-time size `n`; `insert` when the
/// result has one more axis than `x`.
fn runtime_expand(x_dims: Vec<DimInfo>, out_dims: Vec<DimInfo>, prim: Prim) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", x_dims, prim);
    let n = load(&mut dag, decl, "n", vec![], Prim::Int64);
    let expanded = add(
        &mut dag,
        decl,
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![x, n],
        out_dims,
        prim,
    );
    dag.add_root(expanded);
    dag
}

#[test]
fn an_expand_past_the_byte_domain_traps_overflow() {
    let dag = runtime_expand(lit(&[1]), vec![named("m")], Prim::F32);
    let inputs = vec![
        ("x", floats(Prim::F32, &[1], vec![1.0])),
        ("n", count(1 << 62)),
    ];
    assert_eq!(
        run(&dag, inputs, true).unwrap_err(),
        trap("expand", "byte size exceeds i64")
    );
}

#[test]
fn an_insert_past_the_count_domain_traps_overflow() {
    let dag = runtime_expand(lit(&[2]), vec![named("m"), DimInfo::Lit(2)], Prim::F32);
    let inputs = vec![
        ("x", floats(Prim::F32, &[2], vec![1.0, 2.0])),
        ("n", count(1 << 62)),
    ];
    assert_eq!(
        run(&dag, inputs, true).unwrap_err(),
        trap("insert", "extent product exceeds i64")
    );
}

/// 2^60 f32 elements are 2^62 bytes and 2^61 bools are 2^61 bytes: both fit
/// the byte domain, so both admit, and both need more index scratch than the
/// evaluator can allocate. C's allocation of them fails the same way.
#[test]
fn an_expand_the_evaluator_cannot_hold_is_the_allocation_failure() {
    let dag = runtime_expand(lit(&[1]), vec![named("m")], Prim::F32);
    let inputs = vec![
        ("x", floats(Prim::F32, &[1], vec![1.0])),
        ("n", count(1 << 60)),
    ];
    assert_eq!(run(&dag, inputs, true).unwrap_err(), ALLOCATION_FAILED);
    let dag = runtime_expand(lit(&[1]), vec![named("m")], Prim::Bool);
    let inputs = vec![
        ("x", ints(Prim::Bool, &[1], vec![1])),
        ("n", count(1 << 61)),
    ];
    assert_eq!(run(&dag, inputs, true).unwrap_err(), ALLOCATION_FAILED);
}

#[test]
fn an_expand_control_still_evaluates() {
    // Disposition lock: passed on the base.
    let dag = runtime_expand(lit(&[1]), vec![named("m")], Prim::F32);
    let inputs = vec![("x", floats(Prim::F32, &[1], vec![2.5])), ("n", count(3))];
    let out = run(&dag, inputs, true).unwrap();
    assert_eq!(out[0], floats(Prim::F32, &[3], vec![2.5; 3]));
}

fn runtime_pad(before: RtDim, after: RtDim) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", lit(&[1]), Prim::F32);
    let n = load(&mut dag, decl, "n", vec![], Prim::Int64);
    let padded = add(
        &mut dag,
        decl,
        RiscOp::zero_pad(Prim::F32, vec![(before, after)]),
        vec![x, n],
        vec![named("m")],
        Prim::F32,
    );
    dag.add_root(padded);
    dag
}

#[test]
fn a_pad_past_the_byte_or_extent_domain_traps_overflow() {
    let x = || floats(Prim::F32, &[1], vec![1.0]);
    let dag = runtime_pad(RtDim::Lit(0), RtDim::Node(1));
    assert_eq!(
        run(&dag, vec![("x", x()), ("n", count(1 << 62))], true).unwrap_err(),
        trap("pad", "byte size exceeds i64")
    );
    // 1 + i64::MAX + i64::MAX is past i64 though not past usize, so on the
    // base the unchecked host sum reached the allocation.
    let dag = runtime_pad(RtDim::Node(1), RtDim::Node(1));
    assert_eq!(
        run(&dag, vec![("x", x()), ("n", count(i64::MAX))], true).unwrap_err(),
        trap("pad", "padded extent exceeds i64")
    );
}

/// `x: tensor[n, 0]` with `n = 2^62`: an empty operand whose reduction over
/// axis 1 has 2^62 elements.
fn empty_rows(prim: Prim) -> TensorValue {
    match prim {
        Prim::Bool => ints(Prim::Bool, &[HUGE, 0], vec![]),
        _ => floats(prim, &[HUGE, 0], vec![]),
    }
}

fn reduction(op: RiscOp, input: Prim, result: Prim) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(
        &mut dag,
        decl,
        "x",
        vec![named("n"), DimInfo::Lit(0)],
        input,
    );
    let reduced = add(&mut dag, decl, op, vec![x], vec![named("n")], result);
    dag.add_root(reduced);
    dag
}

#[test]
fn a_reduction_of_an_empty_axis_past_the_byte_domain_traps_overflow() {
    let f32 = Prim::F32;
    let cases = [
        (
            RiscOp::Sum {
                axis: 1,
                accumulator: f32,
            },
            f32,
            f32,
            "sum",
        ),
        (RiscOp::MaxReduce { axis: 1 }, f32, f32, "max_reduce"),
        (RiscOp::MinReduce { axis: 1 }, f32, f32, "min_reduce"),
        (RiscOp::ProdReduce { axis: 1 }, f32, f32, "prod_reduce"),
        (
            RiscOp::Argmax { axis: 1 },
            f32,
            Prim::Int64,
            "argmax_reduce",
        ),
        (
            RiscOp::Argmin { axis: 1 },
            f32,
            Prim::Int64,
            "argmin_reduce",
        ),
        (
            RiscOp::Count { axes: vec![1] },
            Prim::Bool,
            Prim::Int64,
            "count",
        ),
    ];
    for (op, input, result, name) in cases {
        let dag = reduction(op, input, result);
        assert_eq!(
            run(&dag, vec![("x", empty_rows(input))], true).unwrap_err(),
            trap(name, "byte size exceeds i64"),
            "{name}"
        );
    }
}

#[test]
fn a_constant_shaped_past_the_byte_domain_traps_overflow() {
    // Shaped by a literal extent.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let c = add(
        &mut dag,
        decl,
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        lit(&[HUGE]),
        Prim::F32,
    );
    dag.add_root(c);
    assert_eq!(
        run(&dag, vec![], true).unwrap_err(),
        trap("const", "byte size exceeds i64")
    );
    // Shaped by an extent bound from an (empty) input.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(
        &mut dag,
        decl,
        "x",
        vec![named("n"), DimInfo::Lit(0)],
        Prim::F32,
    );
    let c = add(
        &mut dag,
        decl,
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        vec![named("n")],
        Prim::F32,
    );
    let s = add(
        &mut dag,
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![x],
        vec![named("n")],
        Prim::F32,
    );
    let total = add(
        &mut dag,
        decl,
        RiscOp::Add,
        vec![c, s],
        vec![named("n")],
        Prim::F32,
    );
    dag.add_root(total);
    assert_eq!(
        run(&dag, vec![("x", empty_rows(Prim::F32))], true).unwrap_err(),
        trap("const", "byte size exceeds i64")
    );
    // A tensor literal whose declared extents have no representable count.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let c = add(
        &mut dag,
        decl,
        RiscOp::synth_const_tensor(Prim::F32, vec![]),
        vec![],
        lit(&[1 << 32, 1 << 32]),
        Prim::F32,
    );
    dag.add_root(c);
    assert_eq!(
        run(&dag, vec![], true).unwrap_err(),
        trap("const", "extent product exceeds i64")
    );
}

#[test]
fn a_missing_input_defaulted_past_the_byte_domain_traps_overflow() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", lit(&[HUGE]), Prim::F32);
    dag.add_root(x);
    assert_eq!(
        run(&dag, vec![], false).unwrap_err(),
        trap("load", "byte size exceeds i64")
    );
}

fn runtime_reshape(x: &[usize], target: Vec<RtDim>) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", lit(x), Prim::F32);
    let n = load(&mut dag, decl, "n", vec![], Prim::Int64);
    let dims = (0..target.len())
        .map(|axis| named(&format!("r{axis}")))
        .collect();
    let reshaped = add(
        &mut dag,
        decl,
        RiscOp::Reshape { new_shape: target },
        vec![x, n],
        dims,
        Prim::F32,
    );
    dag.add_root(reshaped);
    dag
}

#[test]
fn a_reshape_target_past_the_count_domain_traps_overflow() {
    let dag = runtime_reshape(&[4], vec![RtDim::Node(1), RtDim::Node(1)]);
    let inputs = vec![
        ("x", floats(Prim::F32, &[4], vec![1.0; 4])),
        ("n", count(1 << 62)),
    ];
    assert_eq!(
        run(&dag, inputs, true).unwrap_err(),
        trap("reshape", "extent product exceeds i64")
    );
}

/// `[2^62, 2^62, 0]` has zero elements and representable strides, so the
/// compiled lane reshapes an empty tensor into it; the base folded the
/// product before the zero and panicked.
#[test]
fn an_empty_reshape_target_with_huge_extents_evaluates() {
    let dag = runtime_reshape(&[0], vec![RtDim::Node(1), RtDim::Node(1), RtDim::Lit(0)]);
    let inputs = vec![
        ("x", floats(Prim::F32, &[0], vec![])),
        ("n", count(1 << 62)),
    ];
    let out = run(&dag, inputs, true).unwrap();
    assert_eq!(out[0].shape, vec![HUGE, HUGE, 0]);
    assert!(out[0].is_empty());
}

/// Permuting an empty `[2^62, 2^62, 0]` to `[0, 2^62, 2^62]` keeps the count
/// and loses the stride domain, which the compiled lane's permutation plan
/// refuses. On the base the evaluator returned the empty tensor.
#[test]
fn a_permutation_past_the_stride_domain_traps_overflow() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(
        &mut dag,
        decl,
        "x",
        vec![named("a"), named("b"), DimInfo::Lit(0)],
        Prim::F32,
    );
    let permuted = add(
        &mut dag,
        decl,
        RiscOp::Permute {
            axes: vec![2, 0, 1],
        },
        vec![x],
        vec![DimInfo::Lit(0), named("a"), named("b")],
        Prim::F32,
    );
    dag.add_root(permuted);
    let inputs = vec![("x", floats(Prim::F32, &[HUGE, HUGE, 0], vec![]))];
    assert_eq!(
        run(&dag, inputs, true).unwrap_err(),
        trap("permute", "stride product exceeds i64")
    );
}

#[test]
fn a_gather_past_the_count_domain_traps_overflow() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let values = load(
        &mut dag,
        decl,
        "v",
        vec![DimInfo::Lit(0), named("w")],
        Prim::F32,
    );
    let indices = load(&mut dag, decl, "i", lit(&[3]), Prim::Int64);
    let gathered = add(
        &mut dag,
        decl,
        RiscOp::Gather {
            axis: 0,
            batch_rank: 0,
        },
        vec![values, indices],
        vec![DimInfo::Lit(3), named("w")],
        Prim::F32,
    );
    dag.add_root(gathered);
    let inputs = vec![
        ("v", floats(Prim::F32, &[0, HUGE], vec![])),
        ("i", ints(Prim::Int64, &[3], vec![0, 0, 0])),
    ];
    assert_eq!(
        run(&dag, inputs, true).unwrap_err(),
        trap("gather", "extent product exceeds i64")
    );
}

/// `lhs @ rhs` over the batch extents `batch` and the matrix extents `m`,
/// `n` and `k`.
fn matmul(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    [lhs, rhs]: [NodeId; 2],
    batch: Vec<DimExpr>,
    [m, n, k]: [DimExpr; 3],
    dims: Vec<DimInfo>,
) -> NodeId {
    add(
        dag,
        decl,
        RiscOp::BlasMatmul {
            batch_dims: batch,
            m,
            n,
            k,
            accumulator: Prim::F32,
        },
        vec![lhs, rhs],
        dims,
        Prim::F32,
    )
}

#[test]
fn a_matmul_over_an_empty_inner_axis_past_the_count_domain_traps_overflow() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = load(
        &mut dag,
        decl,
        "a",
        vec![named("m"), DimInfo::Lit(0)],
        Prim::F32,
    );
    let b = load(&mut dag, decl, "b", lit(&[0, 4]), Prim::F32);
    let extents = [
        DimExpr::Sym("m".into()),
        DimExpr::Concrete(4),
        DimExpr::Concrete(0),
    ];
    let product = matmul(
        &mut dag,
        decl,
        [a, b],
        vec![],
        extents,
        vec![named("m"), DimInfo::Lit(4)],
    );
    dag.add_root(product);
    let inputs = vec![
        ("a", floats(Prim::F32, &[HUGE, 0], vec![])),
        ("b", floats(Prim::F32, &[0, 4], vec![])),
    ];
    assert_eq!(
        run(&dag, inputs, true).unwrap_err(),
        trap("matmul", "extent product exceeds i64")
    );
}

/// An empty batched product whose batch extents alone overflow: the result
/// admits, and the base panicked folding the batch count.
#[test]
fn an_empty_batched_matmul_with_huge_batch_extents_evaluates() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let dims = || vec![named("p"), named("q"), DimInfo::Lit(0), DimInfo::Lit(0)];
    let a = load(&mut dag, decl, "a", dims(), Prim::F32);
    let b = load(&mut dag, decl, "b", dims(), Prim::F32);
    let batch = vec![DimExpr::Sym("p".into()), DimExpr::Sym("q".into())];
    let extents = [0, 0, 0].map(DimExpr::Concrete);
    let product = matmul(&mut dag, decl, [a, b], batch, extents, dims());
    dag.add_root(product);
    let empty = || floats(Prim::F32, &[HUGE, HUGE, 0, 0], vec![]);
    let out = run(&dag, vec![("a", empty()), ("b", empty())], true).unwrap();
    assert_eq!(out[0].shape, vec![HUGE, HUGE, 0, 0]);
    assert!(out[0].is_empty());
}

#[test]
fn a_one_hot_past_the_count_domain_traps_overflow() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let i = load(&mut dag, decl, "i", lit(&[2]), Prim::Int64);
    let hot = add(
        &mut dag,
        decl,
        RiscOp::OneHot { vocab: HUGE },
        vec![i],
        lit(&[2, HUGE]),
        Prim::F32,
    );
    dag.add_root(hot);
    let inputs = vec![("i", ints(Prim::Int64, &[2], vec![0, 1]))];
    assert_eq!(
        run(&dag, inputs, true).unwrap_err(),
        trap("one_hot", "extent product exceeds i64")
    );
}

/// A bound adjoint over an empty key batch `[2^62, 0]` has one group per
/// element of its `[2^62]` result.
#[test]
fn a_bound_adjoint_over_an_empty_key_batch_traps_overflow() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let batch = || vec![named("n"), DimInfo::Lit(0)];
    let seeds = load(&mut dag, decl, "s", batch(), Prim::Int64);
    let keys = add(
        &mut dag,
        decl,
        RiscOp::KeyFromSeed,
        vec![seeds],
        batch(),
        Prim::Key,
    );
    let template = load(&mut dag, decl, "t", batch(), Prim::F32);
    let g = load(&mut dag, decl, "g", batch(), Prim::F32);
    let adjoint = add(
        &mut dag,
        decl,
        RiscOp::UniformBoundAdjoint {
            bound: UniformBound::Low,
        },
        vec![template, g, keys],
        vec![named("n")],
        Prim::F32,
    );
    dag.add_root(adjoint);
    let inputs = vec![
        ("s", ints(Prim::Int64, &[HUGE, 0], vec![])),
        ("t", empty_rows(Prim::F32)),
        ("g", empty_rows(Prim::F32)),
    ];
    assert_eq!(
        run(&dag, inputs, true).unwrap_err(),
        trap("uniform_like", "byte size exceeds i64")
    );
}

/// A batched dropout over data `[2, 2^62, 2^62, 0]`: each row is empty, and
/// the base folded the row extents' product before reaching the zero.
#[test]
fn a_batched_draw_over_empty_rows_with_huge_extents_evaluates() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let seeds = load(&mut dag, decl, "s", lit(&[2]), Prim::Int64);
    let keys = add(
        &mut dag,
        decl,
        RiscOp::KeyFromSeed,
        vec![seeds],
        lit(&[2]),
        Prim::Key,
    );
    let dims = || vec![DimInfo::Lit(2), named("p"), named("q"), DimInfo::Lit(0)];
    let data = load(&mut dag, decl, "d", dims(), Prim::F32);
    let rate = add(
        &mut dag,
        decl,
        RiscOp::synth_const(Prim::F32, 0.5),
        vec![],
        vec![],
        Prim::F32,
    );
    let dropped = add(
        &mut dag,
        decl,
        RiscOp::Dropout,
        vec![data, rate, keys],
        dims(),
        Prim::F32,
    );
    dag.add_root(dropped);
    let inputs = vec![
        ("s", ints(Prim::Int64, &[2], vec![1, 2])),
        ("d", floats(Prim::F32, &[2, HUGE, HUGE, 0], vec![])),
    ];
    let out = run(&dag, inputs, true).unwrap();
    assert_eq!(out[0].shape, vec![2, HUGE, HUGE, 0]);
    assert!(out[0].is_empty());
}

#[test]
fn a_split_control_still_evaluates() {
    // Disposition lock: passed on the base.
    let out = run(&runtime_split(), vec![("n", count(3))], true).unwrap();
    assert_eq!(out[0].shape, vec![3]);
}
