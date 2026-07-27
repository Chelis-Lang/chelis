//! Issue Chelis-Lang/chelis#319 (checker half) — a `def` declared with a
//! SEPARATE `sig` (no inline parameter annotations) must have its BODY
//! annotated with the resolved tensor types implied by that sig, exactly
//! as an inline-annotated `def` already is.
//!
//! Before the fix `annotate_fn_children` inferred the bare `fn` literal's
//! body against fresh unconstrained parameter tvars (the params carry no
//! inline annotation — their types live only in the separate `sig`), so a
//! shape-sensitive body op such as `permute`/`matmul` was stamped with a
//! bare `(t-var …)` type rather than a `(t-tensor …)`. IR lowering then
//! read a rank-0 `default_type()` off that annotation and
//! `tier2::lower_matmul` panicked with `expects rank >= 2`. The fix seeds
//! the body-inference scope with the declared sig parameter types.
//!
//! Spec authority: spec/04-type-system.md §5.7 / §5.8.
//!
//! The assertions traverse the checked Deep AST STRUCTURALLY (locating
//! the `app` node for a named builtin and reading its `type:` metadata
//! tag) rather than string-windowing the canonical print, so they pin the
//! actual annotation rather than its surface formatting.

use chelis_deep::{Atom, Expr};
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::check_ir_program;

// ─── structural Deep AST helpers ─────────────────────────────────────────

fn list_tag(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    // Decode-once: the spelling comes from the decoded tag, never a raw
    // element-0 string.
    list.tag().map(|tag| tag.as_str())
}

/// The `type:` metadata value of a node whose element 1 is a meta map.
fn node_type_meta(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let Some(Expr::Map(meta, _)) = list.elements.get(1) else {
        return None;
    };
    meta.entries
        .iter()
        .find(|(key, _)| key == "type")
        .map(|(_, value)| value)
}

/// The builtin callee name of an `(app {…} (var {…} <name>) …)` node, or
/// `None` when `expr` is not such an app.
fn app_callee_name(expr: &Expr) -> Option<&str> {
    if list_tag(expr) != Some("app") {
        return None;
    }
    let Expr::List(list, _) = expr else {
        return None;
    };
    let callee = list.elements.get(2)?;
    if list_tag(callee) != Some("var") {
        return None;
    }
    let Expr::List(var_list, _) = callee else {
        return None;
    };
    match var_list.elements.get(2) {
        Some(Expr::Atom(Atom::Symbol(name), _)) => Some(name.as_str()),
        _ => None,
    }
}

/// Depth-first visit of every Deep node.
fn visit<'a>(expr: &'a Expr, f: &mut dyn FnMut(&'a Expr)) {
    f(expr);
    match expr {
        Expr::List(list, _) => {
            for child in &list.elements {
                visit(child, f);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                visit(value, f);
            }
        }
        Expr::MetaExpr(meta, _) => {
            for (_, value) in &meta.entries {
                visit(value, f);
            }
            visit(&meta.expr, f);
        }
        Expr::Atom(_, _) => {}
    }
}

fn is_named_def(expr: &Expr, def_name: &str) -> bool {
    if list_tag(expr) != Some("def") {
        return false;
    }
    let Expr::List(list, _) = expr else {
        return false;
    };
    matches!(list.elements.get(2), Some(Expr::Atom(Atom::Symbol(name), _)) if name == def_name)
}

fn checked_def(src: &str, def_name: &str) -> Expr {
    let decls = parse_str(src).expect("surf parse");
    let deep = desugar_program(&decls);
    let checked = check_ir_program(&deep).expect("check");
    checked
        .exprs()
        .iter()
        .find(|e| is_named_def(e, def_name))
        .cloned()
        .unwrap_or_else(|| panic!("def `{def_name}` not found in checked program"))
}

/// Collect the `type:` metadata tag of every body `app` node calling
/// `callee_name`. Returns the tags (e.g. `"t-tensor"`, `"t-var"`).
fn app_type_tags(def: &Expr, callee_name: &str) -> Vec<String> {
    let mut tags = Vec::new();
    visit(def, &mut |node| {
        if app_callee_name(node) == Some(callee_name)
            && let Some(ty) = node_type_meta(node)
            && let Some(tag) = list_tag(ty)
        {
            tags.push(tag.to_string());
        }
    });
    tags
}

/// True when any body `app` node's whole `type:` annotation is a bare
/// `(t-var …)` (an unresolved result type — the #319 rank-collapse
/// symptom).
fn any_app_type_is_bare_tvar(def: &Expr) -> bool {
    let mut found = false;
    visit(def, &mut |node| {
        if app_callee_name(node).is_some()
            && let Some(ty) = node_type_meta(node)
            && list_tag(ty) == Some("t-var")
        {
            found = true;
        }
    });
    found
}

const SEPARATE_SIG_SDPA: &str = "\
sig sdpa: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, s, p] -> tensor[s, d, p]
def sdpa(q, k, v, scale) = {
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  weights = softmax(mul(scores, scale), -1)
  matmul(weights, v)
}
";

// The inline reimplementation the issue points at: identical body, but
// concrete `f32` parameter annotations written inline on the `def` (no
// separate `sig`). The ordinary annotation path already resolves its body
// to tensor types.
const INLINE_TYPED_SDPA: &str = "\
def sdpa(q: tensor[s, d, f32], k: tensor[s, d, f32], v: tensor[s, d, f32], scale: tensor[s, s, f32]) -> tensor[s, d, f32] = {
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  weights = softmax(mul(scores, scale), -1)
  matmul(weights, v)
}
";

#[test]
fn issue_319_separate_sig_body_permute_is_a_tensor_not_bare_tvar() {
    let def = checked_def(SEPARATE_SIG_SDPA, "sdpa");
    let tags = app_type_tags(&def, "permute");
    assert!(
        !tags.is_empty(),
        "issue #319: expected a `permute` app node carrying a type annotation",
    );
    assert!(
        tags.iter().all(|tag| tag == "t-tensor"),
        "issue #319: separate-sig body `permute` must annotate as a tensor, \
         not a bare type variable; got tags {tags:?}",
    );
}

#[test]
fn issue_319_separate_sig_body_carries_no_bare_tvar_app_type() {
    // Negative parity: no body `app` node may be left with a bare
    // `(t-var …)` whole result type — every shape-bearing op resolves to a
    // concrete tensor type when the sig is present.
    let def = checked_def(SEPARATE_SIG_SDPA, "sdpa");
    assert!(
        !any_app_type_is_bare_tvar(&def),
        "issue #319: a separate-sig verb body must not leave an app result \
         type as a bare type variable",
    );
}

#[test]
fn issue_319_inline_typed_control_also_has_tensor_body_types() {
    // Control: the inline-typed form already resolves its body to tensor
    // types. If this regresses, the fix broke the ordinary annotation path
    // rather than the separate-sig one.
    let def = checked_def(INLINE_TYPED_SDPA, "sdpa");
    let tags = app_type_tags(&def, "permute");
    assert!(
        !tags.is_empty() && tags.iter().all(|tag| tag == "t-tensor"),
        "issue #319 control: inline-typed verb body `permute` must resolve to a \
         tensor type; got tags {tags:?}",
    );
    assert!(
        !any_app_type_is_bare_tvar(&def),
        "issue #319 control: inline-typed verb body must not carry a bare-tvar app type",
    );
}
