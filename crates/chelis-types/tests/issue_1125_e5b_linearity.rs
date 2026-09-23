//! chelis#1125 PP7/E5b: the linearity reader uses the shared total carrier
//! view without changing binding-generation or diagnostic semantics.

use chelis_deep::{Atom, Expr, List, Metadata};

use chelis_types::{CheckedProgram, check_ir_program, check_linearity, check_typed_program};

fn legacy_metadata(metadata: &Metadata) -> Metadata {
    metadata
        .map_expressions(&mut |value, _| legacy_expr(value))
        .expect("legacy fixture preserves metadata payloads")
}

fn legacy_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Atom(_, _) => expr.clone(),
        Expr::List(list, span) => Expr::List(
            List {
                elements: list.elements.iter().map(legacy_expr).collect(),
            },
            *span,
        ),
        Expr::Map(metadata, span) => Expr::Map(legacy_metadata(metadata), *span),
        Expr::MetaExpr(metadata_expr, span) => Expr::MetaExpr(
            chelis_deep::MetaExpr {
                metadata: legacy_metadata(&metadata_expr.metadata),
                expr: Box::new(legacy_expr(&metadata_expr.expr)),
            },
            *span,
        ),
        Expr::Node(node, span) => {
            let mut elements = Vec::with_capacity(node.child_count() + 2);
            elements.push(Expr::Atom(Atom::Tag(node.tag()), *span));
            elements.push(Expr::Map(legacy_metadata(node.meta()), *span));
            elements.extend(node.children_slice().iter().map(legacy_expr));
            Expr::List(List { elements }, *span)
        }
        Expr::BareList(elements, span) => {
            Expr::BareList(elements.iter().map(legacy_expr).collect(), *span)
        }
        Expr::UnknownForm(data) => Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
            head: data.head.clone(),
            meta: legacy_metadata(&data.meta),
            children: data.children.iter().map(legacy_expr).collect(),
            span: data.span,
        })),
    }
}

fn checked_program(source: &str, legacy: bool) -> CheckedProgram {
    let successor: Vec<chelis_deep::Expr> =
        chelis_deep::parse_and_stamp_file(source).expect("valid core fixture");
    if legacy {
        let legacy = successor.iter().map(legacy_expr).collect::<Vec<_>>();
        check_ir_program(&legacy).expect("legacy fixture type-checks")
    } else {
        check_typed_program(&successor).expect("successor fixture type-checks")
    }
}

fn ordered_linearity_diagnostics(program: &CheckedProgram) -> Vec<String> {
    match check_linearity(program) {
        Ok(_) => Vec::new(),
        Err(errors) => errors
            .into_iter()
            .map(|error| format!("[{:?}] {}", error.kind, error.message))
            .collect(),
    }
}

#[test]
fn nested_destructure_alias_single_consume_matches_successor_and_legacy() {
    let source = include_str!(
        "fixtures/core/tests/issue_1125_e5b_linearity/nested_destructure_alias_single_consume_matches_successor_and_legacy.0.deep"
    );

    let successor = ordered_linearity_diagnostics(&checked_program(source, false));
    let legacy = ordered_linearity_diagnostics(&checked_program(source, true));
    assert!(successor.is_empty(), "positive control: {successor:?}");
    assert_eq!(
        successor, legacy,
        "carrier choice must not change acceptance"
    );
}

#[test]
fn nested_destructure_alias_reuse_preserves_diagnostic_order_across_carriers() {
    let source = include_str!(
        "fixtures/core/tests/issue_1125_e5b_linearity/nested_destructure_alias_reuse_preserves_diagnostic_order_across_carriers.0.deep"
    );

    let successor = ordered_linearity_diagnostics(&checked_program(source, false));
    let legacy = ordered_linearity_diagnostics(&checked_program(source, true));
    assert_eq!(
        successor, legacy,
        "binding identity, diagnostic ownership, and order must match"
    );
    assert!(
        successor.iter().any(|message| {
            message.contains("[UseAfterConsume]")
                && message.contains("variable `left`")
                && message.contains("consumed by realize")
        }),
        "negative control must retain the component/alias consume diagnostic: {successor:?}"
    );
}
