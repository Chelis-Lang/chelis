//! chelis#2413 B3: `Split`, `FoldIn` and `SplitN` take an optional path
//! activation, so rule V3 (spec/10 §3.2) admits a key shared by exclusive
//! consumers of every kind, not only by two draws, and confines the keys a key
//! operation derives under that sharing to its activation. On hand-built
//! graphs: the verifier's operand and key rules, the evaluator's inactive
//! `SplitN`, and `vmap` and `grad` carrying the activation.
//!
//! Every expected key and draw bit pattern comes from `key_ref.py` and
//! `slice2_ref.py` (independent transcriptions of [05-RNG-2] and [05-OP-8]),
//! through `briefs/keys-b-b3-probes/b3_ref.py`, never from an evaluator.

use chelis_ir::dag::{Dag, DimInfo, KeyBranch, LogicalKind, NodeId, RiscOp, RtDim, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_exact};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::verify::verify;
use chelis_ir::vmap::vectorize_axis0;
use chelis_types::dtype_semantics::{RawTensor, StorageView, TensorStorage, finalize_tensor};
use chelis_types::types::Prim;
use chelis_types::{RandomKey, scalar_from_i64};
use chelis_unord::UnordMap;

// ---- b3_ref.py ----

/// `split(key(7))`.
const LEFT_7: u64 = 0xaa38_9617_2f9a_3213;
const RIGHT_7: u64 = 0x8fd0_6b2e_7bad_8630;
/// `fold_in(split(key(7)).0, 3)`.
const FOLD_LEFT_7_3: u64 = 0xad64_f3fd_0721_99cf;
/// `split_n(split(key(7)).1, 2)`.
const SPLIT_RIGHT_7_2: [u64; 2] = [0xfa58_b3a4_0bda_b437, 0x9392_5f39_938c_c5c7];
/// `split_n(key(7), 4)`.
const SPLIT_7_4: [u64; 4] = [
    0x25ea_33e6_1c10_576f,
    0x7071_24fb_ecd5_f054,
    0x8239_3615_3a56_5205,
    0x53c6_f7e8_3810_b049,
];
/// f32 `uniform_like` over two elements in `[0, 1)`: under `split(key(7)).0`.
const UNIFORM_LEFT_7: [u64; 2] = [0x3f25_1727, 0x3f2c_5b5b];
/// `vmap_arm.ch`'s rows: row 0 draws with `split_n(key(1), 2)[0]`; row 1's
/// two halves `a`, `b` of `split(split_n(key(1), 2)[1])` draw separately.
const VMAP_ROW0: [u64; 2] = [0x3df7_8d2f, 0x3f21_f02c];
const VMAP_ROW1_LEFT: [u64; 2] = [0x3e4c_8b14, 0x3edc_3c1b];
const VMAP_ROW1_RIGHT: [u64; 2] = [0x3eaf_fdc8, 0x3ec2_f914];
/// `split_n(key(1), 2)`.
const SPLIT_1_2: [u64; 2] = [0xf2e0_ed7d_61bc_7ab1, 0x1c3b_e871_ed9d_079c];
/// `grad_arm.ch`: `uniform(a) + uniform(b)` for `(a, b) = split(key(1))`, and
/// `uniform(key(1))` for the other arm.
const GRAD_THEN: [u64; 2] = [0x3f73_6d46, 0x3fbd_b202];
const GRAD_ELSE: [u64; 2] = [0x3e35_1a96, 0x3f15_def3];
/// f32 `uniform_like` over two elements under `key(7)`.
const UNIFORM_7: [u64; 2] = [0x3e01_9516, 0x3f56_526f];

fn ty(dims: &[usize], prim: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().map(|extent| DimInfo::Lit(*extent)).collect(),
        precision: prim,
    }
}

fn node(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    op: RiscOp,
    inputs: Vec<NodeId>,
    dims: &[usize],
    prim: Prim,
) -> NodeId {
    dag.add_node(decl, op, inputs, ty(dims, prim), None)
}

fn i64_const(dag: &mut Dag, decl: chelis_ir::dag::DeclId, value: i64) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::Const {
            value: scalar_from_i64("test", Prim::Int64, value).unwrap(),
        },
        vec![],
        &[],
        Prim::Int64,
    )
}

fn bool_const(dag: &mut Dag, decl: chelis_ir::dag::DeclId, value: bool) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::Const {
            value: scalar_from_i64("test", Prim::Bool, i64::from(value)).unwrap(),
        },
        vec![],
        &[],
        Prim::Bool,
    )
}

fn f32_const(dag: &mut Dag, decl: chelis_ir::dag::DeclId, value: f64) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::synth_const(Prim::F32, value),
        vec![],
        &[],
        Prim::F32,
    )
}

fn load(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    name: &str,
    dims: &[usize],
    prim: Prim,
) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::Load { name: name.into() },
        vec![],
        dims,
        prim,
    )
}

fn key7(dag: &mut Dag, decl: chelis_ir::dag::DeclId) -> NodeId {
    let seed = i64_const(dag, decl, 7);
    node(dag, decl, RiscOp::KeyFromSeed, vec![seed], &[], Prim::Key)
}

fn not(dag: &mut Dag, decl: chelis_ir::dag::DeclId, value: NodeId) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::Logical(LogicalKind::Not),
        vec![value],
        &[],
        Prim::Bool,
    )
}

fn and(dag: &mut Dag, decl: chelis_ir::dag::DeclId, left: NodeId, right: NodeId) -> NodeId {
    node(
        dag,
        decl,
        RiscOp::Logical(LogicalKind::And),
        vec![left, right],
        &[],
        Prim::Bool,
    )
}

/// A rank-0 key operation over `key`, with `active` appended when present.
fn key_op(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    op: RiscOp,
    key: NodeId,
    active: Option<NodeId>,
) -> NodeId {
    let mut inputs = vec![key];
    let dims: &[usize] = match &op {
        RiscOp::FoldIn => {
            inputs.push(i64_const(dag, decl, 3));
            &[]
        }
        RiscOp::SplitN {
            count: RtDim::Lit(count),
        } => match count {
            2 => &[2],
            _ => unreachable!("the tests split in two"),
        },
        _ => &[],
    };
    inputs.extend(active);
    node(dag, decl, op, inputs, dims, Prim::Key)
}

fn left() -> RiscOp {
    RiscOp::Split {
        branch: KeyBranch::Left,
    }
}

fn right() -> RiscOp {
    RiscOp::Split {
        branch: KeyBranch::Right,
    }
}

fn split_two() -> RiscOp {
    RiscOp::SplitN {
        count: RtDim::Lit(2),
    }
}

/// An f32 uniform draw over a template shaped like `key`'s leading axes plus
/// two elements, under `active`.
fn draw(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    key: NodeId,
    active: Option<NodeId>,
) -> NodeId {
    let mut dims = dag.get(key).unwrap().output_type.dims.clone();
    dims.push(DimInfo::Lit(2));
    let data = TensorType {
        dims,
        precision: Prim::F32,
    };
    let template = dag.add_node(
        decl,
        RiscOp::Load {
            name: format!("t{}", data.dims.len()).as_str().into(),
        },
        vec![],
        data.clone(),
        None,
    );
    let low = f32_const(dag, decl, 0.0);
    let high = f32_const(dag, decl, 1.0);
    let inputs = [template, low, high, key]
        .into_iter()
        .chain(active)
        .collect();
    dag.add_node(decl, RiscOp::UniformLike, inputs, data, None)
}

fn assert_accepted(dag: &Dag) {
    assert_eq!(verify(dag), Vec::<String>::new());
}

fn assert_rejected(dag: &Dag, needle: &str) {
    let errors = verify(dag);
    assert!(
        errors.iter().any(|error| error.contains(needle)),
        "expected `{needle}` in {errors:?}"
    );
}

fn bool_value(shape: Vec<usize>, values: &[bool]) -> TensorValue {
    TensorValue::from_storage(
        shape,
        finalize_tensor(
            "test",
            Prim::Bool,
            RawTensor::Int(values.iter().map(|value| i64::from(*value)).collect()),
        )
        .unwrap(),
    )
}

fn i64_value(value: i64) -> TensorValue {
    TensorValue::from_storage(
        vec![],
        finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![value])).unwrap(),
    )
}

fn f32_value(shape: Vec<usize>, data: Vec<f64>) -> TensorValue {
    TensorValue::from_storage(
        shape,
        finalize_tensor("test", Prim::F32, RawTensor::Float(data)).unwrap(),
    )
}

fn stored_bits(value: &TensorValue) -> Vec<u64> {
    match value.storage().view() {
        StorageView::F32(values) => values.iter().map(|v| u64::from(v.to_bits())).collect(),
        other => panic!("not an f32 draw: {other:?}"),
    }
}

fn key_bits(value: &TensorValue) -> Vec<u64> {
    value
        .storage()
        .keys()
        .expect("a key value")
        .iter()
        .map(|key| key.bits())
        .collect()
}

/// Evaluate `dag`'s roots with the templates `t1`/`t2` of ones and the other
/// named inputs from `inputs`.
fn eval(
    dag: &Dag,
    inputs: &[(&str, TensorValue)],
) -> Result<UnordMap<NodeId, TensorValue>, String> {
    eval_tensor_roots_exact(dag, dag.roots(), |name| match name {
        "t1" => Some(f32_value(vec![2], vec![1.0, 1.0])),
        "t2" => Some(f32_value(vec![2, 2], vec![1.0; 4])),
        _ => inputs
            .iter()
            .find(|(input, _)| *input == name)
            .map(|(_, value)| value.clone()),
    })
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Draw,
    SplitPair,
    Left,
    FoldIn,
    SplitN,
}

/// A consumer of `key` of `kind` under `active`, and a draw under `active`
/// of every key it derives; returns the nodes to root.
fn consume(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    kind: Kind,
    key: NodeId,
    active: NodeId,
) -> Vec<NodeId> {
    let derived = match kind {
        Kind::Draw => return vec![draw(dag, decl, key, Some(active))],
        Kind::SplitPair => vec![
            key_op(dag, decl, left(), key, Some(active)),
            key_op(dag, decl, right(), key, Some(active)),
        ],
        Kind::Left => vec![key_op(dag, decl, left(), key, Some(active))],
        Kind::FoldIn => vec![key_op(dag, decl, RiscOp::FoldIn, key, Some(active))],
        Kind::SplitN => vec![key_op(dag, decl, split_two(), key, Some(active))],
    };
    derived
        .into_iter()
        .map(|derived| draw(dag, decl, derived, Some(active)))
        .collect()
}

/// Rule V3 for every pair of consumer kinds: a key shared by two exclusive
/// arms, each consuming it once, is one use per selected arm. Regression
/// test: the base verifier refuses every activated key operation by arity.
#[test]
fn exclusive_consumers_of_every_kind_share_one_key() {
    let kinds = [
        Kind::Draw,
        Kind::SplitPair,
        Kind::Left,
        Kind::FoldIn,
        Kind::SplitN,
    ];
    for first in kinds {
        for second in kinds {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let key = key7(&mut dag, decl);
            let c = load(&mut dag, decl, "c", &[], Prim::Bool);
            let parent = load(&mut dag, decl, "p", &[], Prim::Bool);
            let then_arm = and(&mut dag, decl, parent, c);
            let not_c = not(&mut dag, decl, c);
            let else_arm = and(&mut dag, decl, parent, not_c);
            let mut roots = consume(&mut dag, decl, first, key, then_arm);
            roots.extend(consume(&mut dag, decl, second, key, else_arm));
            dag.set_roots(roots);
            assert_eq!(verify(&dag), Vec::<String>::new(), "{first:?} / {second:?}");
        }
    }
}

/// The selected arm draws the reference bits and the other draws nothing:
/// `split_key` in the then arm, a draw of the same key in the else arm.
#[test]
fn a_split_arm_and_a_draw_arm_draw_only_the_selected_arms_bits() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = key7(&mut dag, decl);
    let c = load(&mut dag, decl, "c", &[], Prim::Bool);
    let not_c = not(&mut dag, decl, c);
    let then_draws = consume(&mut dag, decl, Kind::SplitPair, key, c);
    let else_draw = draw(&mut dag, decl, key, Some(not_c));
    dag.set_roots(vec![then_draws[0], then_draws[1], else_draw]);
    assert_accepted(&dag);
    for selected in [true, false] {
        let out = eval(&dag, &[("c", bool_value(vec![], &[selected]))]).unwrap();
        let then_bits = stored_bits(&out[&then_draws[0]]);
        let else_bits = stored_bits(&out[&else_draw]);
        let zeros = vec![0; 2];
        if selected {
            assert_eq!(then_bits, UNIFORM_LEFT_7);
            assert_eq!(else_bits, zeros);
        } else {
            assert_eq!(then_bits, zeros);
            assert_eq!(else_bits, UNIFORM_7);
        }
    }
}

/// Negative parity: a split and a draw of one key whose activations overlap,
/// or of which one has none, are two uses. Regression test for the messages:
/// the base verifier rejects every activated key operation by arity instead.
#[test]
fn a_split_and_a_draw_of_one_key_without_exclusive_activations_are_rejected() {
    let build = |split_active: bool, draw_active: Option<bool>| {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let key = key7(&mut dag, decl);
        let c = load(&mut dag, decl, "c", &[], Prim::Bool);
        let not_c = not(&mut dag, decl, c);
        let half = key_op(&mut dag, decl, left(), key, split_active.then_some(c));
        let drawn_half = draw(&mut dag, decl, half, split_active.then_some(c));
        let active = draw_active.map(|same| if same { c } else { not_c });
        let drawn = draw(&mut dag, decl, key, active);
        dag.set_roots(vec![drawn_half, drawn]);
        dag
    };
    assert_accepted(&build(true, Some(false)));
    assert_rejected(
        &build(true, Some(true)),
        "whose activations are not exclusive",
    );
    assert_rejected(&build(true, None), "is consumed twice");
    assert_rejected(&build(false, Some(false)), "is consumed twice");
    // Two lefts of one key, one without an activation.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = key7(&mut dag, decl);
    let c = load(&mut dag, decl, "c", &[], Prim::Bool);
    let first = key_op(&mut dag, decl, left(), key, Some(c));
    let second = key_op(&mut dag, decl, left(), key, None);
    let a = draw(&mut dag, decl, first, Some(c));
    let b = draw(&mut dag, decl, second, None);
    dag.set_roots(vec![a, b]);
    assert_rejected(&dag, "is split twice for the Left branch");
    // A fold and a split-n under unrelated conditions.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = key7(&mut dag, decl);
    let c = load(&mut dag, decl, "c", &[], Prim::Bool);
    let d = load(&mut dag, decl, "d", &[], Prim::Bool);
    let not_d = not(&mut dag, decl, d);
    let mut roots = consume(&mut dag, decl, Kind::FoldIn, key, c);
    roots.extend(consume(&mut dag, decl, Kind::SplitN, key, not_d));
    dag.set_roots(roots);
    assert_rejected(&dag, "whose activations are not exclusive");
}

/// Two exclusive `Split{Left}`s of one key derive one key. Graph of the
/// hazard: each half drawn outside its arm draws the same bits.
fn escaping_lefts(escape: Escape) -> (Dag, [NodeId; 2]) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = key7(&mut dag, decl);
    let c = load(&mut dag, decl, "c", &[], Prim::Bool);
    let not_c = not(&mut dag, decl, c);
    let first = key_op(&mut dag, decl, left(), key, Some(c));
    let second = key_op(&mut dag, decl, left(), key, Some(not_c));
    let kept = draw(&mut dag, decl, second, Some(not_c));
    let escaped = match escape {
        Escape::Unconditional => draw(&mut dag, decl, first, None),
        Escape::OtherArm => draw(&mut dag, decl, first, Some(not_c)),
        Escape::ThroughAFold => {
            let folded = key_op(&mut dag, decl, RiscOp::FoldIn, first, Some(c));
            draw(&mut dag, decl, folded, None)
        }
        Escape::FoldOutside => {
            let folded = key_op(&mut dag, decl, RiscOp::FoldIn, first, None);
            draw(&mut dag, decl, folded, Some(c))
        }
        Escape::Root => first,
        Escape::Nested => {
            let y = load(&mut dag, decl, "y", &[], Prim::Bool);
            let nested = and(&mut dag, decl, c, y);
            draw(&mut dag, decl, first, Some(nested))
        }
        Escape::NeverRuns => {
            let off = bool_const(&mut dag, decl, false);
            draw(&mut dag, decl, first, Some(off))
        }
    };
    dag.set_roots(vec![escaped, kept]);
    (dag, [escaped, kept])
}

#[derive(Clone, Copy)]
enum Escape {
    Unconditional,
    OtherArm,
    ThroughAFold,
    FoldOutside,
    Root,
    Nested,
    NeverRuns,
}

/// Rule V3's confinement: a key derived under exclusive sharing, and every
/// key derived from it, is used only under its operation's activation.
/// Regression test: the base verifier refuses the activated splits by arity,
/// so no graph here reaches the confinement rule there.
#[test]
fn a_key_derived_under_exclusive_sharing_is_used_only_under_that_activation() {
    let outside = "uses it outside that activation";
    for (escape, needle) in [
        (Escape::Unconditional, outside),
        (Escape::OtherArm, outside),
        (Escape::ThroughAFold, outside),
        (Escape::FoldOutside, outside),
        (Escape::Root, "and is a graph root"),
    ] {
        assert_rejected(&escaping_lefts(escape).0, needle);
    }
    // A nested arm implies its parent's activation, and a `false` one never
    // runs.
    assert_accepted(&escaping_lefts(Escape::Nested).0);
    assert_accepted(&escaping_lefts(Escape::NeverRuns).0);
    // Why: evaluated anyway, the escaped half draws the other arm's bits.
    let (dag, [escaped, kept]) = escaping_lefts(Escape::Unconditional);
    let out = eval(&dag, &[("c", bool_value(vec![], &[false]))]).unwrap();
    assert_eq!(stored_bits(&out[&escaped]), UNIFORM_LEFT_7);
    assert_eq!(stored_bits(&out[&kept]), UNIFORM_LEFT_7);
}

/// A key operation whose key no other consumer shares needs no confinement,
/// and its activation changes no key it derives: the activation of an
/// operation outside any branch is inert. Regression test: the base verifier
/// refuses the activated operations by arity.
#[test]
fn an_activation_on_an_unshared_key_changes_no_key() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = key7(&mut dag, decl);
    let active = load(&mut dag, decl, "c", &[], Prim::Bool);
    let l = key_op(&mut dag, decl, left(), key, Some(active));
    let r = key_op(&mut dag, decl, right(), key, Some(active));
    let folded = key_op(&mut dag, decl, RiscOp::FoldIn, l, Some(active));
    let rows = key_op(&mut dag, decl, split_two(), r, Some(active));
    dag.set_roots(vec![folded, rows]);
    assert_accepted(&dag);
    for value in [true, false] {
        let out = eval(&dag, &[("c", bool_value(vec![], &[value]))]).unwrap();
        assert_eq!(key_bits(&out[&folded]), [FOLD_LEFT_7_3]);
        assert_eq!(key_bits(&out[&rows]), SPLIT_RIGHT_7_2);
    }
    // The halves themselves, with the activation false.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = key7(&mut dag, decl);
    let off = bool_const(&mut dag, decl, false);
    let l = key_op(&mut dag, decl, left(), key, Some(off));
    let r = key_op(&mut dag, decl, right(), key, Some(off));
    dag.set_roots(vec![l, r]);
    assert_accepted(&dag);
    let out = eval(&dag, &[]).unwrap();
    assert_eq!(key_bits(&out[&l]), [LEFT_7]);
    assert_eq!(key_bits(&out[&r]), [RIGHT_7]);
}

/// A key operation ends with at most one Bool activation shaped like a
/// leading part of its key's shape. Negative parity for the operand rule.
#[test]
fn a_key_operation_activation_is_one_bool_shaped_like_its_keys_leading_axes() {
    let reject = |active_dims: &[usize], prim: Prim, extra: bool, needle: &str| {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let seeds = load(&mut dag, decl, "s", &[2], Prim::Int64);
        let keys = node(
            &mut dag,
            decl,
            RiscOp::KeyFromSeed,
            vec![seeds],
            &[2],
            Prim::Key,
        );
        let active = load(&mut dag, decl, "c", active_dims, prim);
        let mut inputs = vec![keys, active];
        if extra {
            inputs.push(active);
        }
        let half = node(&mut dag, decl, left(), inputs, &[2], Prim::Key);
        dag.add_root(half);
        assert_rejected(&dag, needle);
    };
    let shape = "exactly one Bool activation, shaped like a leading part";
    reject(&[], Prim::Int64, false, shape);
    reject(&[3], Prim::Bool, false, shape);
    reject(&[2, 1], Prim::Bool, false, shape);
    reject(&[], Prim::Bool, true, "wrong number of inputs");
    // A key from a seed consumes no key and takes no activation.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let seed = i64_const(&mut dag, decl, 7);
    let active = load(&mut dag, decl, "c", &[], Prim::Bool);
    let key = node(
        &mut dag,
        decl,
        RiscOp::KeyFromSeed,
        vec![seed, active],
        &[],
        Prim::Key,
    );
    dag.add_root(key);
    assert_rejected(&dag, "wrong number of inputs");
    // Accepted: rank 0 and the key's own shape.
    for dims in [&[][..], &[2]] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let seeds = load(&mut dag, decl, "s", &[2], Prim::Int64);
        let keys = node(
            &mut dag,
            decl,
            RiscOp::KeyFromSeed,
            vec![seeds],
            &[2],
            Prim::Key,
        );
        let active = load(&mut dag, decl, "c", dims, Prim::Bool);
        let half = node(&mut dag, decl, left(), vec![keys, active], &[2], Prim::Key);
        dag.add_root(half);
        assert_accepted(&dag);
    }
}

/// `split_keys(key(7), n)` over a runtime count under the activation `c`,
/// its count axis declared `declared`; with `d` binding `n` when `bind`.
fn gated_split(declared: DimInfo, active_dims: &[usize]) -> (Dag, NodeId) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let seeds = load(&mut dag, decl, "s", active_dims, Prim::Int64);
    let keys = node(
        &mut dag,
        decl,
        RiscOp::KeyFromSeed,
        vec![seeds],
        active_dims,
        Prim::Key,
    );
    let n = load(&mut dag, decl, "n", &[], Prim::Int64);
    let active = load(&mut dag, decl, "c", active_dims, Prim::Bool);
    let mut dims = ty(active_dims, Prim::Key).dims;
    dims.push(declared);
    let rows = dag.add_node(
        decl,
        RiscOp::SplitN {
            count: RtDim::Node(1),
        },
        vec![keys, n, active],
        TensorType {
            dims,
            precision: Prim::Key,
        },
        None,
    );
    dag.add_root(rows);
    (dag, rows)
}

/// An unselected `split_keys` reads no count: a negative or impossible count
/// neither traps nor allocates, and its count axis is empty where the split
/// declares it or the declared extent where another node binds it. The same
/// counts trap once any row is active. Regression test: at the base the
/// verifier refuses the activation, and without one the split traps.
#[test]
fn an_inactive_runtime_split_reads_no_count() {
    let seven = |shape: Vec<usize>| {
        let len = shape.iter().product::<usize>();
        TensorValue::from_storage(
            shape,
            finalize_tensor("test", Prim::Int64, RawTensor::Int(vec![7; len])).unwrap(),
        )
    };
    let own = || DimInfo::Named("keys".into(), None);
    for count in [-3, i64::MAX, 1 << 61] {
        let (dag, rows) = gated_split(own(), &[]);
        assert_accepted(&dag);
        let inputs = |active: bool| {
            vec![
                ("s", seven(vec![])),
                ("n", i64_value(count)),
                ("c", bool_value(vec![], &[active])),
            ]
        };
        let out = eval(&dag, &inputs(false)).unwrap();
        assert_eq!(out[&rows].shape, vec![0], "count {count}");
        let trap = eval(&dag, &inputs(true)).unwrap_err();
        assert!(trap.contains("numeric trap"), "count {count}: {trap}");
        // A batch of two keys whose rows are all inactive reads no count; one
        // active row reads it.
        let (dag, rows) = gated_split(own(), &[2]);
        assert_accepted(&dag);
        let inputs = |active: [bool; 2]| {
            vec![
                ("s", seven(vec![2])),
                ("n", i64_value(count)),
                ("c", bool_value(vec![2], &active)),
            ]
        };
        let out = eval(&dag, &inputs([false, false])).unwrap();
        assert_eq!(out[&rows].shape, vec![2, 0], "count {count}");
        assert!(eval(&dag, &inputs([false, true])).is_err(), "count {count}");
    }
    // The count axis declared by a literal takes that extent, and the keys
    // are the split's own: an activation changes no key.
    let (dag, rows) = gated_split(DimInfo::Lit(4), &[]);
    assert_accepted(&dag);
    let out = eval(
        &dag,
        &[
            ("s", seven(vec![])),
            ("n", i64_value(-3)),
            ("c", bool_value(vec![], &[false])),
        ],
    )
    .unwrap();
    assert_eq!(key_bits(&out[&rows]), SPLIT_7_4);
}

/// `vmap_arm.ch`'s body as a graph: `split_key(k)` under `c` beside a draw of
/// `k` under `Not(c)`. `vmap` batches each key operation's activation as it
/// batches a draw's, and each row draws only its selected arm. Regression
/// test: the base verifier refuses the activated splits.
#[test]
fn vmap_batches_a_key_operations_activation() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let key = load(&mut dag, decl, "k", &[], Prim::Key);
    let c = load(&mut dag, decl, "c", &[], Prim::Bool);
    let not_c = not(&mut dag, decl, c);
    let then_draws = consume(&mut dag, decl, Kind::SplitPair, key, c);
    let else_draw = draw(&mut dag, decl, key, Some(not_c));
    dag.set_roots(vec![then_draws[0], then_draws[1], else_draw]);
    assert_accepted(&dag);
    let batched = vectorize_axis0(&dag, DimInfo::Lit(2)).unwrap();
    assert_accepted(&batched);
    for split in batched
        .nodes()
        .iter()
        .filter(|node| matches!(node.op, RiscOp::Split { .. }))
    {
        let active = batched.get(split.inputs[1]).unwrap();
        assert_eq!(active.output_type, ty(&[2], Prim::Bool));
    }
    // A key's bits are its seed's two's-complement bits ([05-OP-69]).
    let keys = TensorValue::from_storage(
        vec![2],
        TensorStorage::from_keys(
            SPLIT_1_2
                .iter()
                .map(|bits| {
                    let seed = i64::from_ne_bytes(bits.to_ne_bytes());
                    RandomKey::from_seed(scalar_from_i64("test", Prim::Int64, seed).unwrap())
                        .unwrap()
                })
                .collect(),
        ),
    );
    let out = eval_tensor_roots_exact(&batched, batched.roots(), |name| match name {
        "k" => Some(keys.clone()),
        "c" => Some(bool_value(vec![2], &[false, true])),
        _ => Some(f32_value(vec![2, 2], vec![1.0, 2.0, 2.0, 2.0])),
    })
    .unwrap();
    let roots = batched.roots();
    // Row 0 selects the else arm, row 1 the then arm.
    assert_eq!(
        stored_bits(&out[&roots[2]]),
        [VMAP_ROW0[0], VMAP_ROW0[1], 0, 0]
    );
    let halves = [stored_bits(&out[&roots[0]]), stored_bits(&out[&roots[1]])];
    assert_eq!(halves[0][..2], [0, 0]);
    assert_eq!(halves[1][..2], [0, 0]);
    assert_eq!(
        [halves[0][2..].to_vec(), halves[1][2..].to_vec()],
        [VMAP_ROW1_LEFT.to_vec(), VMAP_ROW1_RIGHT.to_vec()]
    );
}

/// `grad_arm.ch` as a graph: `grad` keeps each key operation's activation,
/// so the backward graph verifies and each selected arm's pathwise gradient
/// is the reference. Regression test: at the base the forward graph's splits
/// carry no activation and share the key with the other arm's draw.
#[test]
fn grad_carries_a_key_operations_activation() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let seed = i64_const(&mut dag, decl, 1);
    let key = node(
        &mut dag,
        decl,
        RiscOp::KeyFromSeed,
        vec![seed],
        &[],
        Prim::Key,
    );
    let x = load(&mut dag, decl, "x", &[2], Prim::F32);
    let c = load(&mut dag, decl, "c", &[], Prim::Bool);
    let not_c = not(&mut dag, decl, c);
    let low = f32_const(&mut dag, decl, 0.0);
    let high = f32_const(&mut dag, decl, 1.0);
    let uniform = |dag: &mut Dag, key: NodeId, active: NodeId| {
        dag.add_node(
            decl,
            RiscOp::UniformLike,
            vec![x, low, high, key, active],
            ty(&[2], Prim::F32),
            None,
        )
    };
    let a = key_op(&mut dag, decl, left(), key, Some(c));
    let b = key_op(&mut dag, decl, right(), key, Some(c));
    let ua = uniform(&mut dag, a, c);
    let ub = uniform(&mut dag, b, c);
    let u = uniform(&mut dag, key, not_c);
    let mul = |dag: &mut Dag, u: NodeId| node(dag, decl, RiscOp::Mul, vec![u, x], &[2], Prim::F32);
    let ma = mul(&mut dag, ua);
    let mb = mul(&mut dag, ub);
    let then_value = node(&mut dag, decl, RiscOp::Add, vec![ma, mb], &[2], Prim::F32);
    let else_value = mul(&mut dag, u);
    let c2 = node(
        &mut dag,
        decl,
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(2),
        },
        vec![c],
        &[2],
        Prim::Bool,
    );
    let selected = node(
        &mut dag,
        decl,
        RiscOp::Where,
        vec![c2, then_value, else_value],
        &[2],
        Prim::F32,
    );
    let loss = node(
        &mut dag,
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![selected],
        &[],
        Prim::F32,
    );
    dag.add_root(loss);
    assert_accepted(&dag);
    let grad = grad_dag_checked(&dag, loss, &[x]).unwrap();
    assert_accepted(&grad.dag);
    assert!(
        grad.dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Split { .. }))
            .all(|split| split.inputs.len() == 2),
        "every split keeps its activation"
    );
    let gradient = grad.grad_nodes[&x];
    for (selected, expected) in [(true, GRAD_THEN), (false, GRAD_ELSE)] {
        let out = eval_tensor_roots_exact(&grad.dag, &[gradient], |name| match name {
            "x" => Some(f32_value(vec![2], vec![1.0, 2.0])),
            "c" => Some(bool_value(vec![], &[selected])),
            _ => None,
        })
        .unwrap();
        assert_eq!(stored_bits(&out[&gradient]), expected, "c = {selected}");
    }
}
