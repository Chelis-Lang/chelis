//! chelis#632: checker movement typing must mint a FRESH extent for every
//! non-identity stride/pad axis instead of passing the input's symbolic
//! dim through unchanged.
//!
//! The pass-through was an annotation-level lie (`stride(&x, 2i64, 2i64)` on
//! `tensor[batch, 4, f32]` stamped `batch` on an axis whose true extent
//! is `ceil(batch/2)`) and falsely tripped the §4.4.1 return-dim rigidity
//! rule on `sig f[n, u]: tensor[n, f32] -> tensor[u, f32]` over `stride(x, 2i64)`
//! (the pass-through unified `u := n`). The identity cases — stride step
//! 1, zero pad — MUST keep passing the symbol through (the
//! `issue_513_symbolic_axis_adjoints` contract), mirroring the IR-side
//! identity-only rule in `chelis_ir::dag::shape_source_for_axis`.
//! `shrink` never passed symbols through (literal bounds produce
//! `Lit(end-start)`, runtime bounds a wildcard), so it is unaffected.
//!
//! Spec: `spec/04-type-system.md` §4.7 (identity-only symbolic
//! pass-through note), `spec/05-risc-primitives.md` §2.4.1.

use chelis_deep::ast::Atom;
use chelis_deep::{Expr, printer::print_canonical};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn expect_clean(src: &str, what: &str) -> Vec<Expr> {
    let deep = surf_to_deep(src);
    match check_ir_program(&deep) {
        Ok(checked) => checked.annotated_exprs().to_vec(),
        Err(rep) => {
            for err in &rep.errors {
                eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
            }
            panic!(
                "expected clean check for {what}, got {} error(s)",
                rep.errors.len()
            );
        }
    }
}

fn var_name(expr: &Expr) -> Option<&str> {
    // Decode-once: the spelling comes from the decoded tag, never a raw
    // head string.
    let Expr::Node(node, _) = expr else {
        return None;
    };
    if node.tag().as_str() != "var" {
        return None;
    }
    match node.children_slice().first() {
        Some(Expr::Atom(Atom::Name(name), _)) => Some(name.as_str()),
        _ => None,
    }
}

/// The canonical-printed `type` metadata of the first `(app ... (var ..
/// <builtin>) ...)` node in the annotated program, e.g.
/// `(t-tensor {} (d-name {} *) (d-lit {} 2) (t-prim {} f32))`.
fn stamped_app_type(exprs: &[Expr], builtin: &str) -> Option<String> {
    fn walk(expr: &Expr, builtin: &str) -> Option<String> {
        let children = match expr {
            Expr::Node(node, _) => {
                if node.tag().as_str() == "app"
                    && node.children_slice().first().and_then(var_name) == Some(builtin)
                    && let Some(ty) = node.meta().ty().map(|ty| ty.expression())
                {
                    return Some(print_canonical(std::slice::from_ref(ty)));
                }
                node.children_slice()
            }
            Expr::BareList(elements, _) => elements.as_slice(),
            _ => return None,
        };
        children.iter().find_map(|kid| walk(kid, builtin))
    }
    exprs.iter().find_map(|expr| walk(expr, builtin))
}

// ---------------------------------------------------------------------------
// Non-identity axes mint fresh extents.
// ---------------------------------------------------------------------------

/// `stride(&x, 2i64, 2i64)` on `tensor[batch, 4, f32]`: axis 0's true extent is
/// `ceil(batch/2)`, not `batch`. The stamped stride type must carry a
/// wildcard on axis 0 (fresh, runtime-guarded extent), not the input's
/// symbol. Pre-chelis#632 the pass-through arm stamped `batch` — the
/// latent lie this test pins away.
#[test]
fn issue632_stride_nonidentity_step_mints_fresh_extent() {
    let annotated = expect_clean(
        r#"
def f(x: tensor[batch, 4, f32]) -> tensor[*, 2, f32] = stride(&x, 2i64, 2i64)
"#,
        "non-identity stride on a symbolic axis",
    );
    let ty = stamped_app_type(&annotated, "stride").expect("stride app has a stamped type");
    assert!(
        !ty.contains("batch"),
        "a stride-2 axis must not keep the input symbol `batch`: {ty}"
    );
    assert!(
        ty.contains('*'),
        "the non-identity stride axis must stamp a wildcard: {ty}"
    );
}

/// `pad(&x, [[1i64, 0i64]], 0.0)` on `tensor[batch, f32]`: the padded extent is
/// `batch + 1`, so the input symbol must not pass through.
#[test]
fn issue632_pad_nonzero_padding_mints_fresh_extent() {
    let annotated = expect_clean(
        r#"
def p(x: tensor[batch, f32]) -> tensor[*, f32] = pad(&x, [[1i64, 0i64]], 0.0)
"#,
        "non-zero pad on a symbolic axis",
    );
    let ty = stamped_app_type(&annotated, "pad").expect("pad app has a stamped type");
    assert!(
        !ty.contains("batch"),
        "a non-zero-padded axis must not keep the input symbol `batch`: {ty}"
    );
    assert!(
        ty.contains('*'),
        "the non-zero-padded axis must stamp a wildcard: {ty}"
    );
}

/// The chelis#632 false-rejection reproducer: a direct-return literal
/// stride under distinct sig symbols. The pass-through unified `u := n`
/// and the §4.4.1 return-dim rigidity guard rejected a well-formed
/// program. With a fresh extent, `u` stays unbound (wildcard unify) and
/// the program checks clean.
#[test]
fn issue632_sig_symbol_stride_no_false_rigidity_rejection() {
    expect_clean(
        r#"
module Repro.SigStride
sig f[n, u]: tensor[n, f32] -> tensor[u, f32]
def f(x) = stride(x, cast(2, i64))
"#,
        "sig-symbol direct-return stride",
    );
}

// ---------------------------------------------------------------------------
// Identity axes keep the symbol (the issue_513 contract, checker level).
// ---------------------------------------------------------------------------

/// Stride step 1 is identity: symbolic `batch` passes through, so the
/// declared `tensor[batch, 2, f32]` return stays honest. Pins the
/// checker-level contract behind
/// `issue_513_symbolic_axis_adjoints::issue_513_stride_symbolic_batch_grad_linear_matches_fd`.
#[test]
fn issue632_stride_identity_step_keeps_symbolic_dim() {
    let annotated = expect_clean(
        r#"
def f(x: tensor[batch, 4, f32]) -> tensor[batch, 2, f32] = stride(&x, 1i64, 2i64)
"#,
        "identity stride on a symbolic axis",
    );
    let ty = stamped_app_type(&annotated, "stride").expect("stride app has a stamped type");
    assert!(
        ty.contains("batch"),
        "a stride-1 (identity) axis must keep the input symbol `batch`: {ty}"
    );
}

/// Zero pad is identity: symbolic `batch` passes through.
#[test]
fn issue632_pad_zero_padding_keeps_symbolic_dim() {
    let annotated = expect_clean(
        r#"
def p(x: tensor[batch, f32]) -> tensor[batch, f32] = pad(&x, [[0i64, 0i64]], 0.0)
"#,
        "zero pad on a symbolic axis",
    );
    let ty = stamped_app_type(&annotated, "pad").expect("pad app has a stamped type");
    assert!(
        ty.contains("batch"),
        "a zero-pad (identity) axis must keep the input symbol `batch`: {ty}"
    );
}

// ---------------------------------------------------------------------------
// Literal axes are untouched (the issue_187 arithmetic stays exact).
// ---------------------------------------------------------------------------

/// Literal extents keep exact arithmetic: stride 2 on `Lit(4)` is
/// `Lit(2)`, non-zero pad on `Lit(2)` is `Lit(3)` — a fresh-extent fix
/// must not widen literal axes to wildcards.
#[test]
fn issue632_literal_axes_keep_exact_arithmetic() {
    expect_clean(
        r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 2, f32] = stride(&x, 1i64, 2i64)
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[0i64, 1i64], [0i64, 1i64]], 0.0)
"#,
        "literal stride/pad arithmetic",
    );
}
