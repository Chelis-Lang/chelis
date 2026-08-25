//! chelis#1023 §B row 1 / chelis#885: structural-position decode fidelity.
//!
//! `spec/03-deep-syntax.md` §7.2 [03-ROLE-3] fixes the disambiguator for a
//! list at a Binder/Syntax/Selector position: it is a vocabulary node exactly
//! when its head is a closed-vocabulary tag AND element 1 is a metadata map.
//! Both fall-through rows are deliberate acceptance of real programs, not
//! missed decode, and this suite locks each row in both polarities:
//!
//! - tag head + metadata map decodes to a typed `Node` at any depth inside a
//!   structural region (the §B row 1 headline);
//! - tag-word head without a metadata map stays a structural `BareList`
//!   (import name lists such as `(copy fill)` are real programs);
//! - ordinary-name head with a metadata map stays a structural `BareList`
//!   (annotated parameters such as `(x {type: ...})` are real programs);
//! - a bare identifier at an expression slot is the [03-ROLE-2] ingress
//!   rejection, naming the identifier and the `(var {} ...)` remediation.

use chelis_deep::parser::parse_and_stamp_file;
use chelis_deep::{Atom, DeepTag, Expr, StampErrorKind, StampOrParseError};

/// Stamp a `.dp` program, panicking with the error on failure.
fn stamp(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source).unwrap_or_else(|e| panic!("fixture must stamp: {e}"))
}

/// Descend into a stamped `Node`, returning the child at `index`.
fn node_child(expr: &Expr, expected: DeepTag, index: usize) -> &Expr {
    let Expr::Node(node, _) = expr else {
        panic!("expected a Node({expected:?}), got {expr:?}");
    };
    assert_eq!(node.tag(), expected, "wrong node tag");
    node.children_slice()
        .get(index)
        .unwrap_or_else(|| panic!("Node({expected:?}) has no child {index}"))
}

#[test]
fn tag_head_with_meta_decodes_at_binder_syntax_selector() {
    // Binder position: `fn` child 0. `(params {} w b x)` carries the
    // [03-ROLE-3] conjunction, so it decodes to Node(Params), not BareList.
    let exprs = stamp(
        "(def {} f (fn {} (params {} w b x) (var {} w)))",
    );
    let body = node_child(&exprs[0], DeepTag::Def, 1);
    let params = node_child(body, DeepTag::Fn, 0);
    let Expr::Node(params_node, _) = params else {
        panic!("params with tag head and metadata map must decode to a Node, got {params:?}");
    };
    assert_eq!(params_node.tag(), DeepTag::Params);
    assert_eq!(params_node.children_slice().len(), 3);

    // Nested depth inside the structural region: an annotated parameter's
    // `type:` metadata value is itself stamped, so the `(t-prim {} f32)`
    // inside it decodes to a Node too.
    let exprs = stamp(
        "(def {} g (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))",
    );
    let body = node_child(&exprs[0], DeepTag::Def, 1);
    let params = node_child(body, DeepTag::Fn, 0);
    let Expr::Node(params_node, _) = params else {
        panic!("expected Node(Params), got {params:?}");
    };
    let Some(Expr::BareList(param, _)) = params_node.children_slice().first() else {
        panic!(
            "the annotated parameter stays a structural BareList: {:?}",
            params_node.children_slice()
        );
    };
    let Some(Expr::Map(map, _)) = param.get(1) else {
        panic!("annotated parameter carries its type map: {param:?}");
    };
    let Some((_, type_value)) = map.entries.iter().find(|(key, _)| key == "type") else {
        panic!("type entry present: {map:?}");
    };
    assert!(
        matches!(type_value, Expr::Node(node, _) if node.tag() == DeepTag::TPrim),
        "a t-prim nested inside a param type map decodes to a typed Node, got {type_value:?}"
    );
}

#[test]
fn tag_head_without_meta_stays_bare_list() {
    // `copy` spells a vocabulary tag, but an import name list carries no
    // metadata map at element 1, so [03-ROLE-3] keeps it a structural
    // BareList — reinterpreting it by its head alone would reject a real
    // program.
    let exprs = stamp("(import {} std.linalg (copy fill))");
    let names = node_child(&exprs[0], DeepTag::Import, 1);
    let Expr::BareList(elements, _) = names else {
        panic!("an import name list stays a structural BareList, got {names:?}");
    };
    assert!(
        matches!(
            elements.first(),
            Some(Expr::Atom(Atom::Name(head), _)) if head == "copy"
        ),
        "the tag-word spelling stays an ordinary name, got {elements:?}"
    );
    assert!(
        matches!(
            elements.get(1),
            Some(Expr::Atom(Atom::Name(second), _)) if second == "fill"
        ),
        "got {elements:?}"
    );
}

#[test]
fn unknown_head_with_meta_stays_bare_list_with_name_head() {
    // `(x {type: ...})` carries a metadata map at element 1, but `x` is not
    // a vocabulary tag, so [03-ROLE-3] keeps it a structural BareList whose
    // head is an ordinary `Atom::Name` — the annotated-parameter shape.
    let exprs = stamp(
        "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))",
    );
    let body = node_child(&exprs[0], DeepTag::Def, 1);
    let params = node_child(body, DeepTag::Fn, 0);
    let Expr::Node(params_node, _) = params else {
        panic!("expected Node(Params), got {params:?}");
    };
    let Some(param) = params_node.children_slice().first() else {
        panic!("one parameter expected");
    };
    let Expr::BareList(elements, _) = param else {
        panic!("an annotated parameter stays a structural BareList, got {param:?}");
    };
    assert!(
        matches!(
            elements.first(),
            Some(Expr::Atom(Atom::Name(head), _)) if head == "x"
        ),
        "the parameter name stays an ordinary Atom::Name head, got {elements:?}"
    );
}

#[test]
fn bare_name_at_expr_slot_rejects_citing_role_rule() {
    // [03-ROLE-2]: a bare identifier at an expression position is an
    // ingress rejection that names the identifier and the `(var {} ...)`
    // remediation spelling.
    let err = parse_and_stamp_file("(def {} f oops)")
        .expect_err("a bare name at a RuntimeExpr slot must reject at ingress");
    let StampOrParseError::Stamp(stamp_err) = err else {
        panic!("the rejection is a stamp error, got {err:?}");
    };
    assert!(
        matches!(&stamp_err.kind, StampErrorKind::NameAtExprSlot { name } if name == "oops"),
        "got {stamp_err:?}"
    );
    let rendered = stamp_err.to_string();
    assert!(
        rendered.contains("bare name `oops`"),
        "identifies the offending identifier: {rendered}"
    );
    assert!(
        rendered.contains("use `(var {} oops)`"),
        "names the remediation spelling: {rendered}"
    );
}
