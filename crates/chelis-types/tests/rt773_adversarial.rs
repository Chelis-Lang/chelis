//! RED-TEAM adversarial suite for chelis#773 (PR #778) — de-poison of the
//! `infer_app` arg-Error short-circuit. Fresh fixtures written by the red
//! team (NOT the PR author's tests). Each asserts a de-poison invariant the
//! PR must hold. Run against the PR-head branch.

use chelis_deep::{Atom, Expr};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::{InferResult, check_ir_program};

fn surf_to_deep_macro(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn messages(rep: &InferResult) -> Vec<String> {
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

fn reject_messages(source: &str, label: &str) -> Vec<String> {
    let deep = surf_to_deep_macro(source);
    let rep = check_ir_program(&deep)
        .err()
        .unwrap_or_else(|| panic!("{label}: expected a check rejection, but it passed"));
    messages(&rep)
}

// ─── structural Deep-AST helpers (mirrors issue_319 detector) ─────────────

fn node_tag(expr: &Expr) -> Option<&str> {
    // Decode-once: the spelling comes from the decoded tag, never a raw
    // head string.
    expr.tag().map(|tag| tag.as_str())
}

fn node_type_meta(expr: &Expr) -> Option<&Expr> {
    let Expr::Node(node, _) = expr else {
        return None;
    };
    node.meta().ty().map(|ty| ty.expression())
}

fn app_callee_name(expr: &Expr) -> Option<&str> {
    if node_tag(expr) != Some("app") {
        return None;
    }
    let Expr::Node(app, _) = expr else {
        return None;
    };
    let callee = app.children_slice().first()?;
    if node_tag(callee) != Some("var") {
        return None;
    }
    let Expr::Node(var, _) = callee else {
        return None;
    };
    match var.children_slice().first() {
        Some(Expr::Atom(Atom::Name(name), _)) => Some(name.as_str()),
        _ => None,
    }
}

fn visit<'a>(expr: &'a Expr, f: &mut dyn FnMut(&'a Expr)) {
    f(expr);
    match expr {
        Expr::Map(map, _) => {
            map.visit_expressions(&mut |value, _| visit(value, f));
        }
        Expr::MetaExpr(meta, _) => {
            meta.metadata
                .visit_expressions(&mut |value, _| visit(value, f));
            visit(&meta.expr, f);
        }
        Expr::Node(node, _) => {
            node.meta()
                .visit_expressions(&mut |value, _| visit(value, f));
            for child in node.children_iter() {
                match child {
                    chelis_deep::node::ChildRef::Expr(expr)
                    | chelis_deep::node::ChildRef::Syntax(expr)
                    | chelis_deep::node::ChildRef::Type(expr)
                    | chelis_deep::node::ChildRef::EffectHandler(expr)
                    | chelis_deep::node::ChildRef::Bypass(expr) => visit(expr, f),
                    chelis_deep::node::ChildRef::Binder(_)
                    | chelis_deep::node::ChildRef::Selector(_) => {}
                }
            }
        }
        Expr::BareList(elements, _) => {
            for child in elements {
                visit(child, f);
            }
        }
        Expr::UnknownForm(data) => {
            data.meta.visit_expressions(&mut |value, _| visit(value, f));
            for child in &data.children {
                visit(child, f);
            }
        }
        Expr::Atom(_, _) => {}
    }
}

fn is_named_def(expr: &Expr, def_name: &str) -> bool {
    if node_tag(expr) != Some("def") {
        return false;
    }
    let Expr::Node(def, _) = expr else {
        return false;
    };
    matches!(def.children_slice().first(), Some(Expr::Atom(Atom::Name(name), _)) if name == def_name)
}

fn checked_def(src: &str, def_name: &str) -> Expr {
    let decls = parse_surf(src).expect("surf parse");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_ir_program(&deep).expect("check");
    checked
        .exprs()
        .iter()
        .find(|e| is_named_def(e, def_name))
        .cloned()
        .unwrap_or_else(|| panic!("def `{def_name}` not found in checked program"))
}

fn any_app_type_is_bare_tvar(def: &Expr) -> bool {
    let mut found = false;
    visit(def, &mut |node| {
        if app_callee_name(node).is_some()
            && let Some(ty) = node_type_meta(node)
            && node_tag(ty) == Some("t-var")
        {
            found = true;
        }
    });
    found
}

fn app_type_tags(def: &Expr, callee_name: &str) -> Vec<String> {
    let mut tags = Vec::new();
    visit(def, &mut |node| {
        if app_callee_name(node) == Some(callee_name)
            && let Some(ty) = node_type_meta(node)
            && let Some(tag) = node_tag(ty)
        {
            tags.push(tag.to_string());
        }
    });
    tags
}

// ─── (3a) THREE Error args + one wrong sibling: no flood ──────────────────

const PRELUDE4: &str = "\
type P =
  | P { v: tensor[2, f32] }
type V =
  | V { v: tensor[2, f32] }
def take_four(a: P, b: P, c: P, d: V) -> f32 = cast(0.0, f32)
";

#[test]
fn three_error_args_plus_one_wrong_sibling_no_flood() {
    // slots a,b,c get three DISTINCT unbound names (each an Error arg); slot
    // d gets a `P` where a `V` is required (a genuine sibling mismatch). The
    // de-poison must (1) still surface the mismatch, and (2) report the
    // wrong sibling EXACTLY ONCE — no flood off the three Error siblings.
    let src = format!(
        "{PRELUDE4}\
def driver() -> f32 = {{
  pp = P {{ v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }}
  sink = take_four(miss_a, miss_b, miss_c, pp)
  cast(0.0, f32)
}}
"
    );
    let msgs = reject_messages(&src, "3a: three errors + wrong sibling");
    let unbound: Vec<_> = msgs
        .iter()
        .filter(|m| m.contains("unbound variable"))
        .collect();
    let mismatch: Vec<_> = msgs
        .iter()
        .filter(|m| m.contains("mismatch") && m.contains('P') && m.contains('V'))
        .collect();
    assert_eq!(
        unbound.len(),
        3,
        "3a: expected exactly three unbound-var diagnostics, got {msgs:?}",
    );
    assert_eq!(
        mismatch.len(),
        1,
        "3a: the wrong sibling must be reported EXACTLY ONCE (no flood), got {msgs:?}",
    );
    assert_eq!(
        msgs.len(),
        4,
        "3a: total should be 3 unbound + 1 mismatch = 4, no extras, got {msgs:?}",
    );
}

// ─── (3b) fresh separate-sig def, shape-computed builtins, no bare tvar ───

const FRESH_SEPARATE_SIG: &str = "\
sig ffn[n, d, h, p: Float]: tensor[n, d, p] -> tensor[d, h, p] -> tensor[h, d, p] -> tensor[n, d, p] -> tensor[n, d, p]
def ffn(x, w1, w2, r) = {
  hidden = matmul(x, w1)
  proj = matmul(hidden, w2)
  rt = permute(r, 1, 0)
  back = permute(rt, 1, 0)
  mul(proj, back)
}
";

#[test]
fn fresh_separate_sig_two_builtins_checks_clean() {
    // 4 params sharing precision var `p` and dim vars n/d/h; let-bound
    // intermediates; three distinct builtins (matmul, permute, mul). Must
    // check clean — the var-ID-collision fix must not spuriously error.
    let deep = desugar_program(&parse_surf(FRESH_SEPARATE_SIG).expect("surf parse"))
        .expect("Surf fixture must desugar");
    let rep = check_ir_program(&deep);
    assert!(
        rep.is_ok(),
        "3b: fresh separate-sig def must check clean; got {:?}",
        rep.err().map(|r| messages(&r)),
    );
}

#[test]
fn fresh_separate_sig_no_bare_tvar_app_types() {
    // The whole point of the annotation-pass var-ID fix: no shape-computed
    // body `app` may be left as a bare `(t-var …)` result type.
    let def = checked_def(FRESH_SEPARATE_SIG, "ffn");
    assert!(
        !any_app_type_is_bare_tvar(&def),
        "3b: no body app result type may be a bare t-var",
    );
    let matmul_tags = app_type_tags(&def, "matmul");
    assert!(
        !matmul_tags.is_empty() && matmul_tags.iter().all(|t| t == "t-tensor"),
        "3b: every matmul app must annotate as t-tensor, got {matmul_tags:?}",
    );
    let permute_tags = app_type_tags(&def, "permute");
    assert!(
        !permute_tags.is_empty() && permute_tags.iter().all(|t| t == "t-tensor"),
        "3b: every permute app must annotate as t-tensor, got {permute_tags:?}",
    );
}

// ─── (3c) #530 expand gate still fires on an Error-typed inline size ───────

#[test]
fn expand_error_inline_size_still_fires_form3_gate_once() {
    // An inline expand size that infers to Error via `add(t.1, ...)` (a
    // tuple projection distinct from the PR author's `t.0` cases). With the
    // arg-Error short-circuit gone, the size still reaches the per-form
    // Form-3 gate: the #469 sourceless-size reject must fire EXACTLY ONCE
    // (not skipped, not doubled), and never an ICE.
    let msgs = reject_messages(
        "def g[a, n](b: tensor[n, f32], t: (i64, i64)) -> tensor[a, n, f32] = \
         insert(b, 0, add(t.1, cast(1, i64)))\n",
        "3c: expand Error size",
    );
    let form3: Vec<_> = msgs
        .iter()
        .filter(|m| {
            m.contains("insert")
                && m.contains("no tensor in scope carries it")
                && m.contains("chelis#469")
        })
        .collect();
    assert_eq!(
        form3.len(),
        1,
        "3c: the Form-3 sourceless-size reject must fire exactly once, got {msgs:?}",
    );
    assert!(
        !msgs.iter().any(|m| m.contains("internal compiler error")),
        "3c: reject must be a clean diagnostic, never an ICE; got {msgs:?}",
    );
}

// ─── (3d) 3-deep call chain, one reported error: single diagnostic ────────

#[test]
fn three_deep_call_chain_single_diagnostic() {
    // An unbound name flows into a 3-level pass-through call chain. Each
    // level returns its resolved type; no re-poison, no flood through
    // returns. Exactly ONE diagnostic (the unbound var) end-to-end.
    let msgs = reject_messages(
        "\
type P =
  | P { v: tensor[2, f32] }
def wrap1(x: P) -> P = x
def wrap2(x: P) -> P = wrap1(x)
def wrap3(x: P) -> P = wrap2(x)
def driver() -> f32 = {
  sink = wrap3(missing_p)
  cast(0.0, f32)
}
",
        "3d: 3-deep chain",
    );
    assert_eq!(
        msgs.len(),
        1,
        "3d: exactly one diagnostic (unbound var) end-to-end, got {msgs:?}",
    );
    assert!(
        msgs[0].contains("unbound variable") && msgs[0].contains("missing_p"),
        "3d: the single diagnostic must be the unbound-var error, got {msgs:?}",
    );
}
