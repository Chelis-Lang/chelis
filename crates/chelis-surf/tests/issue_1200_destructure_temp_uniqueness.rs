//! Chelis-Lang/chelis#1200 — synthesized destructure temps must be unique
//! across a whole desugared program, not per block.
//!
//! `desugar_let_bindings` used to keep its `__chelis_tmpN` counter in a
//! local that restarted at 0 on every call, and it is called once per
//! block. Two destructures in nested blocks therefore both minted
//! `__chelis_tmp0`, `__chelis_tmp1`, ... and the inner names SHADOWED the
//! outer ones in the linearity checker's scope stack.
//!
//! That matters because a destructured component is desugared as an alias
//! of its temp (`a = (var __chelis_tmp2)`), so every consume of `a`
//! resolves through the temp's NAME. With the names colliding, an outer
//! component's consume landed on the inner block's temp entry instead:
//! either blaming the wrong binding, or — when the inner temp was still
//! `Live` — leaving the outer carrier unconsumed, so a genuine double
//! consume was silently accepted. The end-to-end proof of that silent miss
//! lives in
//! `crates/chelis-cli/tests/issue_1200_destructure_scope_lane_parity.rs`.
//!
//! The names are compiler-internal, so the fix is simply never to reuse
//! one within a `DesugarCtx`. These tests are the structural guard.

use std::collections::HashMap;

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;

/// Every `__chelis_tmpN` occurrence in the printed program, counted.
fn temp_occurrences(source: &str) -> HashMap<String, usize> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let printed = deep
        .iter()
        .map(|expr: &Expr| format!("{expr:?}"))
        .collect::<Vec<_>>()
        .join("\n");

    let mut counts: HashMap<String, usize> = HashMap::new();
    let marker = "__chelis_tmp";
    let mut rest = printed.as_str();
    while let Some(start) = rest.find(marker) {
        let tail = &rest[start + marker.len()..];
        let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
        if !digits.is_empty() {
            *counts.entry(format!("{marker}{digits}")).or_default() += 1;
        }
        rest = &tail[digits.len()..];
    }
    counts
}

/// A temp bind and the `(var ...)` references to it are the only
/// occurrences a name should have. Two DISTINCT destructures may never
/// share a name, no matter how the blocks nest.
///
/// Each `(x, y) = t` mints three temps (the tuple root plus one per
/// component) and each is referenced by the component bind that follows
/// it, so a correctly-uniquified program has every name appearing a small
/// bounded number of times — but crucially, the two destructures'
/// name SETS are disjoint. The collision showed up as the SAME name
/// carrying two independent bind sites.
#[test]
fn nested_block_destructures_do_not_reuse_temp_names() {
    let counts = temp_occurrences(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def f(v: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] = {
  (a, b) = two(v)
  r: tensor[2, f32] = {
    (c, d) = two(w)
    add(c, d)
  }
  add(add(a, b), r)
}
"#,
    );

    // Two destructures of a 2-tuple mint three temps each: six distinct
    // names. Before the fix there were only three, each minted twice.
    assert_eq!(
        counts.len(),
        6,
        "expected six distinct destructure temps across the two nested destructures, got {counts:?}"
    );
}

/// Sibling blocks (neither nested inside the other) collided for the same
/// reason: the counter restarted for each. Their live ranges do not
/// overlap, so this one never mis-verdicted on its own — but it is the
/// same defect and the same guard.
#[test]
fn sibling_block_destructures_do_not_reuse_temp_names() {
    let counts = temp_occurrences(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def f(v: tensor[2, f32]) -> tensor[2, f32] = {
  (a, b) = two(v)
  add(a, b)
}
def g(w: tensor[2, f32]) -> tensor[2, f32] = {
  (c, d) = two(w)
  add(c, d)
}
"#,
    );

    assert_eq!(
        counts.len(),
        6,
        "expected six distinct destructure temps across the two sibling destructures, got {counts:?}"
    );
}

/// The uniquifier still refuses to mint a name the authored source
/// already mentions. `fresh_destructure_temp` skips any candidate the
/// program uses, and moving the counter onto the context must not have
/// dropped that guard.
#[test]
fn a_user_written_temp_name_is_never_reused_by_the_desugarer() {
    let counts = temp_occurrences(
        r#"
def two[n](t: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (t, t)
def f(v: tensor[2, f32]) -> tensor[2, f32] = {
  __chelis_tmp0: tensor[2, f32] = v
  (a, b) = two(__chelis_tmp0)
  add(a, b)
}
"#,
    );

    // `__chelis_tmp0` is the user's own binding, so the desugarer must
    // have skipped it when minting: it appears, but only from the user's
    // own two mentions (the bind and the call argument).
    assert_eq!(
        counts.get("__chelis_tmp0").copied(),
        Some(2),
        "the user's own `__chelis_tmp0` must not be reused as a synthesized temp; got {counts:?}"
    );
}
