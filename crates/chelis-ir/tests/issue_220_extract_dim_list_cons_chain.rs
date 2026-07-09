//! Issue Chelis-Lang/chelis#220: `extract_dim_list` in
//! `crates/chelis-ir/src/lower.rs` walked `list.elements` directly and
//! matched any `Atom::Symbol` as a dim name. Surface syntax like
//! `[cast(2, int64), cast(3, int64)]` desugars to a Cons-chain
//! `(app (var Cons) (cast ...) (app (var Cons) (cast ...) (var Nil)))`.
//! Element 0 of that outer `(app ...)` is the literal tag symbol
//! `"app"`, so the broken walker emitted
//! `DimInfo::Named("app", None)` from the tag symbol — synthesizing a
//! bogus symbolic dim that downstream passes (e.g. `symbolic_occurrences`)
//! would treat as a real dim variable.
//!
//! Reachability on current `main`: this defect is unreachable from
//! Surf because `expr_requires_host_runtime` flags any expression
//! containing an uppercase `(var Cons)` / `(var Nil)` for host eval,
//! diverting Cons-chain reshape calls away from the IR `reshape` arm
//! that invokes `extract_dim_list`. The patch nevertheless lands as
//! defense in depth: the IR `reshape` arm is still public surface
//! reachable by tooling, by hand-written Deep, and by future lowering
//! paths that bypass the host-runtime classifier. The acceptance
//! oracle uses `lower_subexpr_program`, which lowers Deep directly via
//! `LowerCtx` without consulting the host-runtime filter, exposing the
//! defect.
//!
//! Fixture acceptance:
//!   1. A Deep `reshape(x, Cons(cast(2, int64), Cons(cast(3, int64), Nil)))`
//!      lowered through `lower_subexpr_program` does NOT synthesize a
//!      `DimInfo::Named("app", _)` (or any other Deep-tag-as-symbol)
//!      anywhere in the resulting DAG.
//!   2. The `Reshape` node's `new_shape` is exactly the integer head
//!      sequence (`[Lit(2), Lit(3)]`), not a single-element symbolic
//!      list derived from misreading the structural tag.
//!
//! Both assertions fail on `main` before the fix and pass after.

use std::collections::HashMap;

use chelis_deep::Expr;
use chelis_ir::dag::{DimInfo, RiscOp, RtDim, TensorType};
use chelis_ir::lower::lower_subexpr_program;
use chelis_types::types::Prim;

/// Hand-built Deep: `reshape(x, Cons(cast(2, int64), Cons(cast(3, int64), Nil)))`.
///
/// The outer `(app ...)` carries no `type:` meta entry, so `lower_app`
/// initializes the reshape's `ty` to `default_type()` and the broken
/// `extract_dim_list` is the only path that can set the
/// `RiscOp::Reshape`'s `new_shape`. The hand-built Cons chain mirrors
/// the exact shape `chelis-surf::desugar` produces for the surface
/// list literal `[cast(2, int64), cast(3, int64)]`.
fn cons_chain_reshape_expr() -> Expr {
    let src = r#"
        (app {}
             (var {} reshape)
             (var {} x)
             (app {}
               (var {} Cons)
               (cast {} (lit {} 2) (t-prim {} int64))
               (app {}
                 (var {} Cons)
                 (cast {} (lit {} 3) (t-prim {} int64))
                 (var {} Nil))))
    "#;
    let mut exprs = chelis_deep::parser::parse_str(src).expect("deep parse");
    assert_eq!(exprs.len(), 1, "expected exactly one top-level expr");
    exprs.pop().unwrap()
}

fn lower_with_x_bound() -> chelis_ir::dag::Dag {
    let mut scoped = HashMap::new();
    scoped.insert(
        "x".to_string(),
        TensorType {
            dims: vec![DimInfo::Lit(6)],
            precision: Prim::F32,
        },
    );
    lower_subexpr_program(
        &cons_chain_reshape_expr(),
        scoped,
        HashMap::new(),
        HashMap::new(),
    )
}

#[test]
fn reshape_cons_chain_does_not_synthesize_tag_named_dim() {
    let dag = lower_with_x_bound();

    // No node anywhere should carry a `Named("app", _)` (or any other
    // Deep tag) as a tensor dim. That's the root-cause invariant: a
    // structural Deep tag must not appear in lowered shape metadata.
    let forbidden_tags = [
        "app", "var", "lit", "cast", "list", "Cons", "Nil", "def", "fn",
    ];
    for node in dag.nodes() {
        for dim in &node.output_type.dims {
            if let DimInfo::Named(name, _) = dim {
                assert!(
                    !forbidden_tags.contains(&name.as_str()),
                    "node id {} ({:?}) carries a Deep-tag-named dim `{}` -- \
                     `extract_dim_list` misread a structural tag as a dim name",
                    node.id.0,
                    node.op,
                    name,
                );
            }
        }
    }
}

#[test]
fn reshape_cons_chain_extracts_integer_dim_list() {
    let dag = lower_with_x_bound();

    // Find the Reshape node and check its `new_shape` matches the
    // Cons-chain head literals exactly. There must be exactly one
    // Reshape node in this program.
    let reshapes: Vec<_> = dag
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, RiscOp::Reshape { .. }))
        .collect();
    assert_eq!(
        reshapes.len(),
        1,
        "expected exactly one Reshape node, got {} (ops: {:?})",
        reshapes.len(),
        dag.nodes().iter().map(|n| &n.op).collect::<Vec<_>>(),
    );
    let reshape = reshapes[0];
    let RiscOp::Reshape { new_shape } = &reshape.op else {
        unreachable!("filtered above");
    };
    assert_eq!(
        new_shape,
        &vec![RtDim::Lit(2), RtDim::Lit(3)],
        "reshape new_shape did not match the integer Cons-chain head \
         literals; got {new_shape:?}",
    );

    // The Reshape node's output type dims should mirror new_shape.
    assert_eq!(
        reshape.output_type.dims,
        vec![DimInfo::Lit(2), DimInfo::Lit(3)],
        "reshape output_type.dims did not match new_shape; got {:?}",
        reshape.output_type.dims,
    );
}
