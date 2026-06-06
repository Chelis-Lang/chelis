//! Issue #318 root cause: the SHAPE-DERIVED `expand` size argument
//! `shape(&x, cast(0, int32))` was not recovered during lowering.
//!
//! `extract_dim_expr_value` reads bare ints, `(lit {} n)`, `cast`-wrapped
//! ints (issue #288), and symbolic dim *variables* — but NOT a host-lane
//! `shape(...)` application. So `expand(scalar_to_tensor(c), 0,
//! shape(&x, 0))` fell back to the default size `1` and
//! `fallback_expand_type` lowered the constant-broadcast idiom to a
//! `tensor[1]` instead of `tensor[n]`. The downstream `mul(x, k)` then
//! mixed `tensor[n]` with `tensor[1]`, which `chelis check` accepts via
//! type-level size-1 broadcasting but the IR (no implicit broadcasting)
//! does not — the malformed `Mul` surfaced only when `grad` verified the
//! cloned forward inside the backward DAG (`binary op ... mismatched
//! dimension at axis 0: Lit(2) vs Lit(1)`), which is why the issue
//! looked like an autodiff bug (and is exactly the #288 failure mode,
//! one size-encoding later).
//!
//! The fix recovers the broadcast extent from the type-checker's output
//! type at the broadcast axis when the size argument is not statically
//! extractable, so the shape-derived form lowers the same as the literal
//! form. These tests pin that the `expand` node carries the requested
//! (symbolic) extent and that the full constant-broadcast idiom lowers
//! to a verify-clean DAG.

use chelis_ir::dag::{DimInfo, RiscOp};
use chelis_ir::lower::lower_program;
use chelis_ir::verify;
use chelis_types::check_ir_program;

fn lower_surf(src: &str) -> Result<chelis_ir::dag::Dag, String> {
    let decls = chelis_surf::parser::parse_str(src).map_err(|e| format!("parse: {e:?}"))?;
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls),
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

/// The `expand` node must carry the shape-derived extent (the runtime
/// dimension `n`), producing a `tensor[n]` output — NOT the default
/// size-1 `tensor[1]`. Before the fix the unreadable `shape(&x, 0)` size
/// silently defaulted the extent to `1`.
#[test]
fn issue_318_expand_with_shape_size_lowers_to_requested_extent() {
    let src = r#"
sig run: tensor[n, f32] -> tensor[n, f32]
def run(x) = mul(x, expand(scalar_to_tensor(cast(2.5, f32)), cast(0, int32), shape(&x, cast(0, int32))))
"#;
    let dag = lower_surf(src).expect("expand-with-shape-size must lower");
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
    // The expand output extent must be the runtime dimension `n`, not the
    // collapsed `Lit(1)` the default-size fallback produced.
    assert_eq!(
        expand.2.dims.len(),
        1,
        "expand output must be rank 1 (tensor[n]); got {:?}",
        expand.2.dims,
    );
    assert_ne!(
        expand.2.dims[0],
        DimInfo::Lit(1),
        "expand output extent must NOT be the default-size Lit(1); got {:?}",
        expand.2.dims[0],
    );
    // The size DimExpr must be non-concrete (the runtime extent), not the
    // collapsed Concrete(1).
    assert!(
        !expand.1.is_concrete(),
        "expand size must be the runtime/shape-derived extent, not a \
         concrete default; got {:?}",
        expand.1,
    );
}

/// The whole issue #318 constant-broadcast forward `f(x) = sum(x *
/// expand(scalar_to_tensor(3.0), 0, shape(&x, 0)))` must lower to a
/// verify-clean DAG. Before the fix the forward `Mul` mixed `tensor[n]`
/// and `tensor[1]` and the DAG was malformed — the malformation only
/// surfaced under `grad`'s backward verification.
#[test]
fn issue_318_constant_broadcast_forward_lowers_clean() {
    let src = r#"
sig f: tensor[n, f32] -> f32
def f(x) = tensor_to_scalar(sum(mul(x, expand(scalar_to_tensor(cast(3.0, f32)), cast(0, int32), shape(&x, cast(0, int32)))), cast(0, int32)))
"#;
    let dag = lower_surf(src).expect("issue #318 forward must lower");
    let errors = verify::verify(&dag);
    assert!(
        errors.is_empty(),
        "issue #318 constant-broadcast forward must verify clean; got {errors:?}",
    );
}

/// Negative-parity / regression lock: the LITERAL-size form (the #288
/// fix) must still lower to the requested concrete extent, so the
/// shape-derived recovery did not regress the literal path.
#[test]
fn issue_318_literal_size_still_lowers_to_concrete_extent() {
    let src = r#"
sig run: tensor[2, f32] -> tensor[2, f32]
def run(x) = mul(x, expand(scalar_to_tensor(cast(2.5, f32)), cast(0, int32), cast(2, int32)))
"#;
    let dag = lower_surf(src).expect("expand-with-literal-size must lower");
    assert!(
        verify::verify(&dag).is_empty(),
        "lowered DAG must verify clean"
    );
    let expand = dag
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            RiscOp::Expand { size, .. } => Some((size.clone(), n.output_type.clone())),
            _ => None,
        })
        .expect("an Expand node must be present");
    assert_eq!(
        expand.0,
        chelis_ir::dag::DimExpr::Concrete(2),
        "literal size 2 must still extract to Concrete(2)",
    );
    assert_eq!(
        expand.1.dims,
        vec![DimInfo::Lit(2)],
        "literal expand output must be tensor[2]",
    );
}
