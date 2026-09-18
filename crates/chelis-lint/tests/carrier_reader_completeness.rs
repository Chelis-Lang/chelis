use chelis_lint::rules::carrier_reader_completeness::CarrierReaderCompleteness;
use chelis_lint::{Context, Rule, Surface};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn check(rel: &str, source: &str) -> Vec<chelis_lint::Violation> {
    let root = Path::new("/repo");
    let path = root.join(rel);
    CarrierReaderCompleteness.check(&Context {
        root,
        path: &path,
        source: Some(source),
        surface: Surface::RustSource,
    })
}

#[test]
fn guarded_expr_list_arm_is_rejected() {
    let source = r#"
use chelis_deep::Expr;

fn read(expr: &Expr) -> bool {
    match expr {
        Expr::List(list, _) if list.elements.len() == 3 => true,
        _ => false,
    }
}
"#;
    let violations = check("crates/chelis-types/src/infer/planted.rs", source);
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert_eq!(violations[0].rule_id, "carrier-reader-completeness");
}

#[test]
fn if_let_and_let_else_expr_list_readers_are_rejected() {
    let source = r#"
use chelis_deep::Expr as DeepExpr;

fn read_if(expr: &DeepExpr) -> bool {
    if let DeepExpr::List(list, _) = expr {
        return list.elements.is_empty();
    }
    false
}

fn read_let(expr: &DeepExpr) -> bool {
    let DeepExpr::List(list, _) = expr else {
        return false;
    };
    list.elements.is_empty()
}
"#;
    let violations = check("crates/chelis-types/src/infer/planted.rs", source);
    assert_eq!(violations.len(), 2, "{violations:?}");
}

#[test]
fn fully_qualified_and_crate_alias_readers_are_rejected() {
    let source = r#"
use chelis_deep::{self as deep};

fn read_direct(expr: &chelis_deep::Expr) -> bool {
    matches!(expr, chelis_deep::Expr::List(_, _))
}

fn read_alias(expr: &deep::Expr) -> bool {
    if let deep::Expr::List(_, _) = expr {
        return true;
    }
    false
}
"#;
    let violations = check("crates/chelis-types/src/infer/planted.rs", source);
    assert_eq!(violations.len(), 2, "{violations:?}");
}

#[test]
fn directly_imported_expr_list_reader_is_rejected() {
    let source = r#"
use chelis_deep::Expr::List;

fn read(expr: &chelis_deep::Expr) -> bool {
    if let List(list, _) = expr {
        return list.elements.is_empty();
    }
    false
}
"#;
    let violations = check("crates/chelis-types/src/infer/planted.rs", source);
    assert_eq!(violations.len(), 1, "{violations:?}");
}

#[test]
fn renamed_expr_list_import_is_rejected() {
    let source = r#"
use chelis_deep::Expr::List as LegacyList;

fn read(expr: &chelis_deep::Expr) -> bool {
    matches!(expr, LegacyList(_, _))
}
"#;
    let violations = check("crates/chelis-types/src/infer/planted.rs", source);
    assert_eq!(violations.len(), 1, "{violations:?}");
}

#[test]
fn expr_variant_glob_reader_is_rejected() {
    let source = r#"
use chelis_deep::Expr::*;

fn read(expr: &chelis_deep::Expr) -> bool {
    matches!(expr, List(list, _) if list.elements.is_empty())
}
"#;
    let violations = check("crates/chelis-types/src/infer/planted.rs", source);
    assert_eq!(violations.len(), 1, "{violations:?}");
}

#[test]
fn node_to_list_method_and_ufcs_alias_are_rejected() {
    let source = r#"
use chelis_deep::{Node, Node as DeepNode, Span};

fn bridge(node: &Node, span: Span) {
    let _ = node.to_list(span);
}

fn bridge_ufcs(node: DeepNode, span: Span) {
    let _ = DeepNode::to_list(node, span);
}
"#;
    let violations = check("crates/chelis-ir/src/planted.rs", source);
    assert_eq!(violations.len(), 2, "{violations:?}");
    assert!(
        violations
            .iter()
            .all(|violation| violation.message.contains("Node::to_list")),
        "{violations:?}"
    );
}

#[test]
fn constructors_and_prose_do_not_look_like_readers() {
    let source = r#"
use chelis_deep::{Expr, List, Span};

const NOTE: &str = "Expr::List(list, _) and node.to_list(span)";

fn produce(list: List, span: Span) -> Expr {
    // Expr::List(list, _) is reader syntax only in this comment.
    Expr::List(list, span)
}
"#;
    assert!(check("crates/chelis-types/src/infer/planted.rs", source).is_empty());
}

#[test]
fn carrier_accessor_match_is_allowed_without_a_named_helper() {
    let source = r#"
use chelis_deep::{DeepTag, Expr, ExprCarrier};

fn read(expr: &Expr) -> usize {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Tuple, _, children) => children.len(),
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => 0,
    }
}
"#;
    assert!(check("crates/chelis-types/src/infer/planted.rs", source).is_empty());
}

#[test]
fn direct_exhaustive_expr_match_is_carrier_complete() {
    let source = r#"
use chelis_deep::Expr;

fn read(expr: &Expr) -> usize {
    match expr {
        Expr::List(list, _) => list.elements.len(),
        Expr::Node(node, _) => node.children_slice().len(),
        Expr::BareList(items, _) => items.len(),
        Expr::UnknownForm(data) => data.children.len(),
        Expr::Atom(_, _) | Expr::Map(_, _) | Expr::MetaExpr(_, _) => 0,
    }
}
"#;
    assert!(check("crates/chelis-types/src/infer/planted.rs", source).is_empty());
}

#[test]
fn directly_imported_variants_form_a_carrier_complete_match() {
    let source = r#"
use chelis_deep::Expr;
use chelis_deep::Expr::{
    Atom as ExprAtom,
    BareList as ExprBareList,
    List as ExprList,
    Map as ExprMap,
    MetaExpr as ExprMeta,
    Node as ExprNode,
    UnknownForm as ExprUnknown,
};

fn read(expr: &Expr) -> usize {
    match expr {
        ExprList(list, _) => list.elements.len(),
        ExprNode(node, _) => node.children_slice().len(),
        ExprBareList(items, _) => items.len(),
        ExprUnknown(data) => data.children.len(),
        ExprAtom(_, _) | ExprMap(_, _) | ExprMeta(_, _) => 0,
    }
}
"#;
    assert!(check("crates/chelis-types/src/infer/planted.rs", source).is_empty());
}

#[test]
fn inline_escape_requires_a_site_level_justification() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("crates/chelis-types/src/infer");
    fs::create_dir_all(&path).expect("create fixture source root");
    let source_path = path.join("reader.rs");
    fs::write(
        &source_path,
        r#"
use chelis_deep::Expr;

fn read(expr: &Expr) -> bool {
    // chelis-lint: allow carrier-reader-completeness
    matches!(expr, Expr::List(list, _) if list.elements.is_empty())
}
"#,
    )
    .expect("write unjustified fixture");
    let rules: Vec<Box<dyn Rule>> = vec![Box::new(CarrierReaderCompleteness)];
    let violations = chelis_lint::lint(dir.path(), &rules).expect("lint fixture");
    assert_eq!(violations.len(), 1, "{violations:?}");

    fs::write(
        &source_path,
        r#"
use chelis_deep::Expr;

fn read(expr: &Expr) -> bool {
    // chelis-lint: allow carrier-reader-completeness -- source is normalized legacy output only; chelis#1125
    matches!(expr, Expr::List(list, _) if list.elements.is_empty())
}
"#,
    )
    .expect("write justified fixture");
    let violations = chelis_lint::lint(dir.path(), &rules).expect("lint justified fixture");
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn chelis_deep_owns_raw_carrier_representation() {
    let source = r#"
use crate::Expr;

fn read(expr: &Expr) -> bool {
    matches!(expr, Expr::List(_, _))
}
"#;
    assert!(check("crates/chelis-deep/src/ast.rs", source).is_empty());
}
