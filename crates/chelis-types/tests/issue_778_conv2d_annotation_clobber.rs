//! chelis#778 follow-up regression (hotfix A). A shape-computed builtin whose
//! operands are unbound in the post-inference type-annotation/writeback scope
//! infer to `Error` in that pass; #778's removal of `infer_app`'s arg-`Error`
//! short-circuit then let the shape override return `subst.apply(ret_tv)` — an
//! unbound `Var` — which the writeback degraded to a rank-0 default
//! `(t-tensor {} (t-prim {} f32))` (no spatial dims), CLOBBERING the node's
//! explicit concrete annotation. The IR lowering then ICEd with "conv2d output
//! height axis requires a statically known axis". The fix makes the override
//! return `Type::Error` when an operand is `Error`, so the writeback preserves
//! the concrete annotation.
//!
//! This test pins the checker-level mechanism directly (independent of
//! lowering): after `check_ir_program`, the `conv2d` app node's written-back
//! `type:` must still be the concrete rank-4 `t-tensor`, never a rank-0
//! `t-tensor` or a bare `t-var`.

use chelis_deep::parser::parse_str;
use chelis_deep::{Atom, Expr};
use chelis_types::check_ir_program;

/// conv2d 1x1: three top-level defs. `x`/`k` are separate defs, so in the
/// `def y` body-annotation scope the `(var x)`/`(var k)` operands are unbound
/// and infer to `Error` — the exact trigger for the writeback clobber.
const CONV2D_SRC: &str = r#"
    (def {} x (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))} x))
    (def {} k (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 1) (d-lit {} 1) (t-prim {} f32))} k))
    (def {} y
      (app {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
           (var {} conv2d) (var {} x) (var {} k) (lit {} 1) (lit {} 0)))
"#;

fn list_tag(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    // Decode-once: the spelling comes from the decoded tag, never a raw
    // element-0 string.
    list.tag().map(|tag| tag.as_str())
}

fn node_type_meta(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let Some(Expr::Map(meta, _)) = list.elements.get(1) else {
        return None;
    };
    meta.ty().map(|ty| ty.expression())
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
        Some(Expr::Atom(Atom::Name(name), _)) => Some(name.as_str()),
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

/// Count the dimension children of a `t-tensor` type node (everything after
/// the tag+meta except the trailing `t-prim` precision node).
fn tensor_type_dim_count(ty: &Expr) -> Option<usize> {
    if list_tag(ty) != Some("t-tensor") {
        return None;
    }
    let Expr::List(list, _) = ty else {
        return None;
    };
    let dims = list
        .elements
        .iter()
        .skip(2)
        .filter(|child| list_tag(child) != Some("t-prim"))
        .count();
    Some(dims)
}

#[test]
fn conv2d_error_operand_does_not_clobber_concrete_annotation() {
    let exprs = parse_str(CONV2D_SRC).expect("parse deep");
    // Program is well-typed: the clobber was a SILENT writeback degradation,
    // not a surfaced type error, so `check_ir_program` must succeed.
    let checked = check_ir_program(&exprs).expect("conv2d program should check clean");

    let mut conv2d_types: Vec<Expr> = Vec::new();
    for expr in checked.exprs() {
        visit(expr, &mut |node| {
            if app_callee_name(node) == Some("conv2d")
                && let Some(ty) = node_type_meta(node)
            {
                conv2d_types.push(ty.clone());
            }
        });
    }

    assert!(
        !conv2d_types.is_empty(),
        "expected a conv2d app node with a written-back type annotation"
    );
    for ty in &conv2d_types {
        assert_eq!(
            list_tag(ty),
            Some("t-tensor"),
            "conv2d written-back type must stay a concrete t-tensor (not a bare t-var), got {ty:?}"
        );
        assert_eq!(
            tensor_type_dim_count(ty),
            Some(4),
            "conv2d output type must keep its 4 concrete dims; a rank-0 (dims=[]) \
             writeback is the #778 clobber that ICEs IR lowering; got {ty:?}"
        );
    }
}
