//! RT-784 adversarial probes (fresh red team, NOT a keeper unless promoted).
//!
//! Two properties of the #784 hotfix guard (`shape_override_operand_error`):
//!
//!  A. GUARD BREADTH — the annotation-writeback clobber the conv2d test pins
//!     must not reproduce for the *other* guarded shape-computed builtins.
//!     Same shape as the conv2d fixture (three top-level defs; the operands
//!     are separate defs and thus unbound → `Error` in the consuming def's
//!     body-annotation scope), for `matmul`, `sum`, and 3-arg `expand`: the
//!     app node's written-back `type:` must stay the concrete `t-tensor`,
//!     never a bare `t-var` (which the writeback degrades to a rank-0 default
//!     and ICEs IR lowering).
//!
//!  B. NO SILENT ACCEPT / NO FLOOD — the guard returns `Type::Error` only
//!     when an operand is ALREADY `Error` (already-reported). So a shape
//!     builtin fed one unbound (reported-error) operand in normal position
//!     must STILL be rejected (never check clean) with EXACTLY ONE diagnostic
//!     (the unbound var), and never ICE. Covers all five signature checkers:
//!     matmul, conv2d, a reduction (sum), expand, layer_norm.

use chelis_deep::parser::parse_str;
use chelis_deep::{Atom, Expr};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::{InferResult, check_ir_program};

// ── Part A helpers (Deep source, mirror issue_778 clobber test) ───────────

fn list_tag(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    match list.elements.first() {
        Some(Expr::Atom(Atom::Symbol(tag), _)) => Some(tag.as_str()),
        _ => None,
    }
}

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

/// After `check_ir_program`, collect the written-back `type:` tag of every
/// app node whose callee is `callee`.
fn writeback_type_tags(src: &str, callee: &str) -> Vec<String> {
    let exprs = parse_str(src).expect("parse deep");
    let checked = check_ir_program(&exprs).expect("program should check clean");
    let mut tags = Vec::new();
    for expr in checked.exprs() {
        visit(expr, &mut |node| {
            if app_callee_name(node) == Some(callee)
                && let Some(ty) = node_type_meta(node)
            {
                tags.push(list_tag(ty).unwrap_or("<none>").to_string());
            }
        });
    }
    tags
}

const MATMUL_SRC: &str = r#"
    (def {} a (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} a))
    (def {} b (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 4) (t-prim {} f32))} b))
    (def {} z
      (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 4) (t-prim {} f32))}
           (var {} matmul) (var {} a) (var {} b)))
"#;

const SUM_SRC: &str = r#"
    (def {} x (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x))
    (def {} s
      (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
           (var {} sum) (var {} x) (lit {} 1)))
"#;

const EXPAND_SRC: &str = r#"
    (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
    (def {} e
      (app {type: (t-tensor {} (d-lit {} 4) (d-lit {} 3) (t-prim {} f32))}
           (var {} expand) (var {} x) (lit {} 0) (lit {} 4)))
"#;

#[test]
fn matmul_error_operand_does_not_clobber_concrete_annotation() {
    let tags = writeback_type_tags(MATMUL_SRC, "matmul");
    assert!(!tags.is_empty(), "expected a matmul app node type");
    assert!(
        tags.iter().all(|t| t == "t-tensor"),
        "matmul written-back type must stay concrete t-tensor, got {tags:?}",
    );
}

#[test]
fn sum_error_operand_does_not_clobber_concrete_annotation() {
    let tags = writeback_type_tags(SUM_SRC, "sum");
    assert!(!tags.is_empty(), "expected a sum app node type");
    assert!(
        tags.iter().all(|t| t == "t-tensor"),
        "sum written-back type must stay concrete t-tensor, got {tags:?}",
    );
}

#[test]
fn expand_error_operand_does_not_clobber_concrete_annotation() {
    let tags = writeback_type_tags(EXPAND_SRC, "expand");
    assert!(!tags.is_empty(), "expected an expand app node type");
    assert!(
        tags.iter().all(|t| t == "t-tensor"),
        "expand written-back type must stay concrete t-tensor, got {tags:?}",
    );
}

// ── Part B helpers (Surf source) ─────────────────────────────────────────

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn messages(rep: &InferResult) -> Vec<String> {
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

/// The program MUST be rejected; returns its diagnostic messages.
fn reject_messages(source: &str, label: &str) -> Vec<String> {
    let deep = surf_to_deep(source);
    let rep = check_ir_program(&deep).err().unwrap_or_else(|| {
        panic!("{label}: expected a check rejection, but it PASSED (silent accept)")
    });
    messages(&rep)
}

fn assert_single_unbound(source: &str, missing: &str, label: &str) {
    let msgs = reject_messages(source, label);
    assert_eq!(
        msgs.len(),
        1,
        "{label}: expected exactly one diagnostic (the unbound var), got {msgs:?}",
    );
    assert!(
        msgs[0].contains("unbound variable") && msgs[0].contains(missing),
        "{label}: the single diagnostic must be the unbound-var error for {missing}; got {msgs:?}",
    );
    assert!(
        !msgs
            .iter()
            .any(|m| m.to_lowercase().contains("internal compiler error")),
        "{label}: must be a clean diagnostic, never an ICE; got {msgs:?}",
    );
}

#[test]
fn matmul_unbound_operand_single_diagnostic_no_accept() {
    assert_single_unbound(
        "def driver(b: tensor[3, 4, f32]) -> f32 = {\n  m = matmul(missing_a, b)\n  cast(0.0, f32)\n}\n",
        "missing_a",
        "matmul main-pass",
    );
}

#[test]
fn conv2d_unbound_operand_rejected_no_accept_no_ice() {
    // conv2d additionally carries a pre-existing IR-fitness validator
    // (`validate_conv2d`, issue #186) that fires on non-concrete argument
    // metadata, independent of and untouched by the #784 guard. So the
    // correct rejection here is TWO diagnostics: the unbound var plus the
    // conv2d metadata validator. Not a flood, not a silent accept, no ICE.
    let msgs = reject_messages(
        "def driver(k: tensor[1, 1, 1, 1, f32]) -> f32 = {\n  y = conv2d(missing_x, k, 1, 0)\n  cast(0.0, f32)\n}\n",
        "conv2d main-pass",
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("unbound variable") && m.contains("missing_x")),
        "conv2d main-pass: must report the unbound var; got {msgs:?}",
    );
    let non_unbound: Vec<_> = msgs
        .iter()
        .filter(|m| !m.contains("unbound variable"))
        .collect();
    assert!(
        non_unbound
            .iter()
            .all(|m| m.contains("conv2d") && m.contains("concrete tensor argument metadata")),
        "conv2d main-pass: the only non-unbound diagnostic must be the #186 metadata \
         validator (no flood, no leaked bare-var cascade); got {msgs:?}",
    );
    assert!(
        !msgs
            .iter()
            .any(|m| m.to_lowercase().contains("internal compiler error")),
        "conv2d main-pass: never an ICE; got {msgs:?}",
    );
}

#[test]
fn sum_unbound_operand_single_diagnostic_no_accept() {
    assert_single_unbound(
        "def driver() -> f32 = {\n  s = sum(missing_x, 1)\n  cast(0.0, f32)\n}\n",
        "missing_x",
        "sum main-pass",
    );
}

#[test]
fn expand_unbound_operand_single_diagnostic_no_accept() {
    assert_single_unbound(
        "def driver() -> f32 = {\n  e = expand(missing_x, 0, 4)\n  cast(0.0, f32)\n}\n",
        "missing_x",
        "expand main-pass",
    );
}

#[test]
fn expand_error_size_operand_single_diagnostic_no_accept() {
    // Genuinely-Error size (arithmetic over an unbound var — NOT a bare
    // identifier, which the size slot would read as a symbolic dim NAME per
    // §4.7.2 Form-2, not an error). Through the guarded generic (3-arg) path
    // the size resolves to `Error`, the guard fires, and the call returns
    // `Error`; the unbound-var diagnostic still fires. Exactly ONE diagnostic
    // — no silent accept, no ICE.
    assert_single_unbound(
        "def driver(x: tensor[3, f32]) -> f32 = {\n  e = expand(x, 0, add(missing_v, cast(1, int32)))\n  cast(0.0, f32)\n}\n",
        "missing_v",
        "expand error-size main-pass",
    );
}

#[test]
fn layer_norm_unbound_operand_single_diagnostic_no_accept() {
    assert_single_unbound(
        "def driver(g: tensor[4, f32], b: tensor[4, f32]) -> f32 = {\n  y = layer_norm(missing_x, g, b)\n  cast(0.0, f32)\n}\n",
        "missing_x",
        "layer_norm main-pass",
    );
}
