//! chelis#1775: a runtime extent's type-level dim name is keyed by the
//! scalar node that PRODUCES the extent, never by the reshape that consumes
//! it.
//!
//! #1693 (`22b193cf7`) wraps every computed reshape target in a per-reshape
//! `Copy` and named the resulting axis after that wrapper. Two reshapes sized
//! from one runtime scalar therefore ended up with two different names for one
//! extent. `concat` requires every non-concat axis to agree across its
//! elements; the disagreement historically sent it to a fallback that
//! emitted a rank-0 `Load { name: "concat" }` placeholder, and `grad` over that
//! placeholder built a backward DAG that failed verification with a rank-zero
//! collapse and `0 extent source(s) for 1 output axis(es)`.
//!
//! The end-to-end receipt is
//! `chelis-cli`'s `issue_368_runtime_symbolic_window_grad_is_half_everywhere`.
//! These fixtures pin the lowering property that test depends on, directly.

use chelis_deep::Expr;
use chelis_ir::axis_sources::check_axis_sources;
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::lower::lower_subexpr_program;
use chelis_types::types::Prim;
use chelis_types::unsupported::Stage;
use chelis_unord::UnordMap;

fn parse_one(src: &str) -> Expr {
    let mut exprs = chelis_deep::parser::parse_str(src).expect("deep parse");
    assert_eq!(exprs.len(), 1, "expected exactly one top-level expr");
    exprs.pop().unwrap()
}

/// `reshape(x, [1, <extent>])`.
fn row(extent: &str) -> String {
    format!(
        "(app {{}} (var {{}} reshape) (var {{}} x) \
          (app {{}} (var {{}} Cons) (cast {{}} (lit {{}} 1) (t-prim {{}} int64)) \
            (app {{}} (var {{}} Cons) {extent} (var {{}} Nil))))"
    )
}

/// `let m = cast(shape(x, 0), int64) in let k = cast(shape(x, 0), int64) in
///  concat([reshape(x, [1, a]), reshape(x, [1, b])], 0)`.
///
/// `m` and `k` are separate bindings over separate `shape` reads, so naming
/// `a`/`b` chooses whether the two rows share one producing scalar node.
fn concat_src(first: &str, second: &str) -> String {
    format!(
        "(let {{}} (bind {{}} m (cast {{}} (app {{}} (var {{}} shape) (var {{}} x) \
           (cast {{}} (lit {{}} 0) (t-prim {{}} int32))) (t-prim {{}} int64))) \
         (let {{}} (bind {{}} k (cast {{}} (app {{}} (var {{}} shape) (var {{}} x) \
           (cast {{}} (lit {{}} 0) (t-prim {{}} int32))) (t-prim {{}} int64))) \
         (app {{}} (var {{}} concat) \
           (app {{}} (var {{}} Cons) {r0} (app {{}} (var {{}} Cons) {r1} (var {{}} Nil))) \
           (cast {{}} (lit {{}} 0) (t-prim {{}} int32)))))",
        r0 = row(first),
        r1 = row(second)
    )
}

fn concat_of_two_rows(first: &str, second: &str) -> Expr {
    parse_one(&concat_src(first, second))
}

fn lower_over_symbolic_x(expr: &Expr) -> Dag {
    let mut scoped = UnordMap::new();
    scoped.insert(
        "x".to_string(),
        TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        },
    );
    lower_subexpr_program(expr, scoped, UnordMap::new(), UnordMap::new())
}

fn reshape_axis_dims(dag: &Dag) -> Vec<DimInfo> {
    dag.nodes()
        .iter()
        .filter(|node| matches!(node.op, RiscOp::Reshape { .. }))
        .map(|node| node.output_type.dims[1].clone())
        .collect()
}

fn has_host_concat_placeholder(dag: &Dag) -> bool {
    dag.nodes()
        .iter()
        .any(|node| matches!(&node.op, RiscOp::Load { name } if name.as_str() == "concat"))
}

fn pad_count(dag: &Dag) -> usize {
    dag.nodes()
        .iter()
        .filter(|node| matches!(node.op, RiscOp::Pad { .. }))
        .count()
}

/// REGRESSION TEST (measured red before the repair: with the axis named after
/// the consuming reshape's `Copy` wrapper the two rows disagree, `concat`
/// bails, and the pruned DAG is a rank-0 `Load { name: "concat" }` placeholder
/// holding no `Reshape` at all).
///
/// Two rows sized from ONE runtime scalar carry ONE name for that extent.
#[test]
fn two_reshapes_from_one_producer_share_the_extent_name() {
    let dag = lower_over_symbolic_x(&concat_of_two_rows("(var {} m)", "(var {} m)"));

    let dims = reshape_axis_dims(&dag);
    assert_eq!(dims.len(), 2, "expected two Reshape nodes, got {dims:?}");
    assert!(
        matches!(&dims[0], DimInfo::Named(name, None) if !name.is_empty() && name != "*"),
        "the runtime extent must carry a real generated name, got {:?}",
        dims[0]
    );
    assert_eq!(
        dims[0], dims[1],
        "both rows are sized from the same scalar node, so they name one extent"
    );
}

/// REGRESSION TEST (same red measurement as above): the shared name is what
/// lets `concat` stay on its differentiable Pad+Add cascade instead of the
/// rank-0 host placeholder that #368 exists to avoid.
#[test]
fn shared_producer_concat_lowers_to_pad_add_not_the_host_placeholder() {
    let dag = lower_over_symbolic_x(&concat_of_two_rows("(var {} m)", "(var {} m)"));

    assert!(
        !has_host_concat_placeholder(&dag),
        "concat must not fall back to the rank-0 host placeholder: {dag:?}"
    );
    assert_eq!(pad_count(&dag), 2, "one Pad per concat element: {dag:?}");

    // The concat result is the summed pair, widened on the concat axis to 2
    // and unchanged on the shared runtime axis.
    let extent = reshape_axis_dims(&dag)[0].clone();
    let sum = dag
        .nodes()
        .iter()
        .rev()
        .find(|node| matches!(node.op, RiscOp::Add))
        .expect("the Pad cascade ends in an Add");
    assert_eq!(
        sum.output_type.dims,
        vec![DimInfo::Lit(2), extent],
        "concat widens only the concat axis"
    );
}

/// NEGATIVE TWIN: distinct producer identities must not be fused.
///
/// Two rows sized from two DISTINCT scalar producers do NOT collapse onto one
/// extent, even though the two `shape(x, 0)` reads agree at run time. Keying
/// the name by the producer is what keeps them apart; lowering declines to
/// fuse extents it cannot prove equal. #1906 replaces the historical fake
/// scalar Load disposition with a structured forced-DAG rejection. Neither
/// disposition is evidence of actual Host execution.
#[test]
fn two_reshapes_from_distinct_producers_reject_forced_dag() {
    let error = chelis_ir::lower::try_lower_subexpr_program(
        &concat_of_two_rows("(var {} m)", "(var {} k)"),
        UnordMap::from([(
            "x".to_string(),
            TensorType {
                dims: vec![DimInfo::Named("n".into(), None)],
                precision: Prim::F32,
            },
        )]),
        UnordMap::new(),
        UnordMap::new(),
    )
    .expect_err("distinct producers cannot prove the non-concat axes equal");
    assert_eq!(
        error.message,
        "tensor concat cannot be represented by the static tensor DAG; use its host execution path (chelis#1906)"
    );
}

/// REGRESSION TEST (measured red before the repair: the summed stack does not
/// even reach `grad_dag_checked`, because lowering the reduction over the
/// rank-0 placeholder aborts first).
///
/// `grad` over the shared-extent window stack builds a backward DAG in which
/// every realized output axis has exactly one extent source.
#[test]
fn grad_over_a_shared_extent_window_stack_gives_every_axis_one_source() {
    let src = format!(
        "(app {{}} (var {{}} sum) (app {{}} (var {{}} sum) {stack} \
           (cast {{}} (lit {{}} 0) (t-prim {{}} int32))) \
           (cast {{}} (lit {{}} 0) (t-prim {{}} int32)))",
        stack = concat_src("(var {} m)", "(var {} m)")
    );
    let dag = lower_over_symbolic_x(&parse_one(&src));
    let x = dag
        .nodes()
        .iter()
        .find(|node| matches!(&node.op, RiscOp::Load { name } if name.as_str() == "x"))
        .expect("the parameter load")
        .id;
    let output = *dag.roots().first().expect("one root");

    let result = grad_dag_checked(&dag, output, &[x]).expect("backward DAG must construct");
    check_axis_sources(&result.dag, Stage::Lowering)
        .expect("every realized output axis has exactly one extent source");
}

/// NEGATIVE TWIN, DISPOSITION LOCK: a DAG whose axis genuinely has no extent
/// source is still refused, so the test above cannot pass by the guard having
/// been weakened.
#[test]
fn an_axis_with_no_extent_source_is_still_refused() {
    let mut dag = Dag::new();
    let sourceless: NodeId = dag.add_node(
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        TensorType {
            dims: vec![DimInfo::Named(String::new(), None)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(sourceless);

    let err = check_axis_sources(&dag, Stage::Lowering)
        .expect_err("an anonymous unsized axis has no extent source");
    let rendered = err.to_string();
    assert!(
        rendered.contains("0 extent source(s) for 1 output axis(es)"),
        "the receipt must name the exact cardinality: {rendered}"
    );
    assert!(
        rendered.contains("chelis#1482"),
        "the receipt must cite its authority: {rendered}"
    );
}
