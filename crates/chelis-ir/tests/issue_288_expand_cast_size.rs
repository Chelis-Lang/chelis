//! Issue #288 root cause: `expand`'s axis/size arguments wrapped in
//! `cast(n, i32)` were not extracted during lowering.
//!
//! `extract_usize_value` recognized bare ints and `(lit {} n)` but not
//! the `(cast {} <inner> ty)` wrapper that Surf emits for
//! `cast(n, i32)`. So `expand(x, cast(0, i32), cast(2, i64))`
//! silently fell back to the defaults (axis 0, size 1) and lowered the
//! constant-broadcast idiom `expand(scalar_to_tensor(c), 0, 2)` to a
//! `tensor[1]` instead of `tensor[2]`. The downstream `mul(x, k)` then
//! mixed `tensor[2]` with `tensor[1]`, which `chelis check` accepts via
//! type-level size-1 broadcasting but the IR (no implicit broadcasting)
//! does not — the malformed `Mul` surfaced only when `grad` verified
//! the backward DAG (`binary op ... mismatched dimension at axis 0:
//! Lit(2) vs Lit(1)`), which is why the issue looked like an autodiff
//! bug.
//!
//! The fix routes `extract_usize_value` through `extract_int_for_dim`,
//! which already unwraps `lit` and `cast`. These tests pin that the
//! `expand` node carries the requested size and that the full
//! constant-broadcast idiom lowers to a verify-clean DAG.

use chelis_ir::dag::{DimInfo, RiscOp};
use chelis_ir::lower::lower_program;
use chelis_ir::verify;
use chelis_types::check_ir_program;

fn lower_surf(src: &str) -> Result<chelis_ir::dag::Dag, String> {
    let decls = chelis_surf::parser::parse_str(src).map_err(|e| format!("parse: {e:?}"))?;
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .map_err(|e| format!("expand: {e:?}"))?
    .into_exprs();
    let checked = check_ir_program(&exprs).map_err(|r| {
        format!(
            "check: {:?}",
            r.errors
                .iter()
                .map(|e| e.message.clone())
                .collect::<Vec<_>>()
        )
    })?;
    let checked = chelis_effects::check_program(&checked).map_err(|e| format!("effects: {e:?}"))?;
    let checked =
        chelis_types::check_linearity(&checked).map_err(|e| format!("linearity: {e:?}"))?;
    Ok(lower_program(&checked))
}

/// The `expand` node must carry the cast-wrapped size `2`, producing a
/// rank-1 `tensor[2]` output — not the default size-1 `tensor[1]`.
#[test]
fn issue_288_expand_with_cast_size_lowers_to_requested_size() {
    let src = r#"
sig run: tensor[2, f32] -> tensor[2, f32]
def run(x) = mul(x, insert(scalar_to_tensor(cast(2.5, f32)), cast(0, i32), cast(2, i64)))
"#;
    let dag = lower_surf(src).expect("expand-with-cast-size must lower");
    assert!(
        verify::verify(&dag).is_empty(),
        "lowered DAG must verify clean; got {:?}",
        verify::verify(&dag),
    );
    let expand = dag
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            RiscOp::Expand { axis, size } => Some((*axis, size.clone(), n.output_type.clone())),
            _ => None,
        })
        .expect("an Expand node must be present");
    assert_eq!(expand.0, 0, "expand axis must be the cast-wrapped 0");
    assert_eq!(
        expand.1,
        chelis_ir::dag::RtDim::Lit(2),
        "expand size must be the cast-wrapped 2, not the default 1",
    );
    assert_eq!(
        expand.2.dims,
        vec![DimInfo::Lit(2)],
        "expand output must be tensor[2], not tensor[1]",
    );
}

/// Negative-parity / regression lock: a plain (non-cast) integer size
/// must still extract correctly, so the cast-unwrapping did not regress
/// the literal path.
#[test]
fn issue_288_expand_with_plain_size_still_lowers() {
    let src = r#"
sig run: tensor[2, f32] -> tensor[2, f32]
def run(x) = mul(x, insert(scalar_to_tensor(cast(2.5, f32)), 0, 2i64))
"#;
    let dag = lower_surf(src).expect("expand-with-plain-size must lower");
    assert!(
        verify::verify(&dag).is_empty(),
        "lowered DAG must verify clean"
    );
    let size = dag.nodes().iter().find_map(|n| match &n.op {
        RiscOp::Expand { size, .. } => Some(size.clone()),
        _ => None,
    });
    assert_eq!(
        size,
        Some(chelis_ir::dag::RtDim::Lit(2)),
        "plain size 2 must still extract"
    );
}

/// The whole issue #288 constant-broadcast forward `f(x) = sum(x *
/// expand(scalar_to_tensor(2.5), 0, 2))` must lower to a verify-clean
/// DAG. Before the fix the forward `Mul` mixed `tensor[2]` and
/// `tensor[1]` and the DAG was malformed.
#[test]
fn issue_288_constant_broadcast_forward_lowers_clean() {
    let src = r#"
sig f: tensor[2, f32] -> f32
def f(x) = tensor_to_scalar(sum(mul(x, insert(scalar_to_tensor(cast(2.5, f32)), cast(0, i32), cast(2, i64))), cast(0, i32)))
"#;
    let dag = lower_surf(src).expect("issue #288 forward must lower");
    let errors = verify::verify(&dag);
    assert!(
        errors.is_empty(),
        "issue #288 constant-broadcast forward must verify clean; got {errors:?}",
    );
}

// ---------------------------------------------------------------------------
// chelis#530 (the #288 cast-wrapped sibling, residual inline-form bypass): a
// `expand` size that is a tuple projection (`t.0`), or a `cast`/arithmetic
// CONTAINING one, used to reach lowering as a swallowed `Type::Error` and
// lower to a hardcoded extent-1 `Expand` (eval `[3, 2]` vs compiled C
// `[1, 2]`). #530 rejected it at check. chelis#469 admits it
// (spec/04-type-system.md section 4.7.2 forbids rejecting an extent for its
// provenance), so the property these rows keep is the one #530 existed for:
// the size lowers to a runtime extent node, never to the extent-1 default.
//
// The arithmetic spelling does not reach the static DAG at all: its def routes
// to the host lane, where a tuple-typed parameter of this dim-polymorphic def
// is a separate, loud lowering gap on both lanes. Its row therefore asserts
// only the half #530 owns, that no extent-1 `Expand` is produced.
// ---------------------------------------------------------------------------

fn assert_runtime_size_lowers_to_a_node(src: &str, label: &str) {
    assert_no_default_extent(src, label, true);
}

fn assert_no_default_extent(src: &str, label: &str, reaches_the_dag: bool) {
    let dag = lower_surf(src).unwrap_or_else(|err| panic!("{label}: must lower, got: {err}"));
    let sizes: Vec<_> = dag
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            RiscOp::Expand { size, .. } => Some(size.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        !sizes.is_empty(),
        reaches_the_dag,
        "{label}: whether the insert lowers to a static-DAG Expand node; got {sizes:?}"
    );
    assert!(
        sizes
            .iter()
            .all(|size| matches!(size, chelis_ir::dag::RtDim::Node(_))),
        "{label}: the size is a runtime extent node, never the extent-1 default; got {sizes:?}"
    );
    assert!(
        verify::verify(&dag).is_empty(),
        "{label}: the lowered DAG must verify clean"
    );
}

#[test]
fn issue_530_tuple_get_size_lowers_to_a_runtime_extent() {
    assert_runtime_size_lowers_to_a_node(
        "def g[a, n](b: tensor[n, f32], t: (i64, i64)) -> tensor[a, n, f32] = insert(b, 0, t.0)\n",
        "tuple-get size",
    );
}

#[test]
fn issue_530_cast_wrapped_tuple_get_size_lowers_to_a_runtime_extent() {
    assert_runtime_size_lowers_to_a_node(
        "def g[a, n](b: tensor[n, f32], t: (i64, i64)) -> tensor[a, n, f32] = insert(b, 0, cast(t.0, i64))\n",
        "cast(tuple-get) size",
    );
}

#[test]
fn issue_530_arith_over_tuple_get_size_never_lowers_to_the_default_extent() {
    assert_no_default_extent(
        "def g[a, n](b: tensor[n, f32], t: (i64, i64)) -> tensor[a, n, f32] = insert(b, 0, add(t.0, cast(0, i64)))\n",
        "add(tuple-get, ...) size",
        false,
    );
}

/// FAIL-CLOSED must not become reject-everything: a genuinely
/// shape-sourced Form-3 size (`shape(c, 0)` of an in-scope tensor) must
/// still lower to a verify-clean DAG carrying the symbolic extent — not be
/// over-rejected by the #530 gate.
#[test]
fn issue_530_shape_sourced_size_still_lowers_clean() {
    let src = "def g[n, a](b: &tensor[n, f32], c: &tensor[a, f32]) -> tensor[a, n, f32] = insert(b, 0, shape(c, cast(0, i32)))\n\
         xs = to_tensor([1.0, 2.0], f32)\n\
         cs = to_tensor([10.0, 20.0, 30.0], f32)\n\
         out = g(&xs, &cs)\n";
    let dag = lower_surf(src).expect("a shape-sourced Form-3 expand size must still lower");
    let errors = verify::verify(&dag);
    assert!(
        errors.is_empty(),
        "shape-sourced Form-3 expand must verify clean; got {errors:?}",
    );
}
