//! chelis#1125 PP7/E5b: the linearity reader uses the shared total carrier
//! view without changing binding-generation or diagnostic semantics.

use chelis_deep::{Atom, Expr, List, Metadata};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
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
    let declarations = parse_str(source).expect("Surf fixture parses");
    let successor = desugar_program(&declarations);
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
    let source = r#"
def ok(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (left, right) = pair
    alias: tensor[4, f32] = left
    consumed_left: tensor[4, f32] = realize(alias)
    consumed_right: tensor[4, f32] = realize(right)
    add(consumed_left, consumed_right)
  }
"#;

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
    let source = r#"
def bad(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (left, right) = pair
    alias: tensor[4, f32] = left
    consumed_left: tensor[4, f32] = realize(alias)
    consumed_right: tensor[4, f32] = realize(right)
    add(left, consumed_left)
  }
"#;

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

#[test]
fn linearity_reader_has_no_private_optional_adapter_or_node_bridge() {
    let source = include_str!("../src/linearity.rs");
    let production = source
        .split("#[cfg(test)]")
        .next()
        .expect("linearity source has a production prefix");
    let adapter_definition = ["fn stamped_", "parts"].concat();
    let adapter_call = ["stamped_", "parts("].concat();

    assert!(
        !production.contains(&adapter_definition) && !production.contains(&adapter_call),
        "E5b requires the measured reads to use Expr::carrier directly"
    );
    assert!(
        !production.contains(".to_list("),
        "E5b forbids Node-to-List bridges in semantic linearity readers"
    );
    assert!(
        !production.contains("Expr::List("),
        "linearity readers must disposition legacy lists through Expr::carrier"
    );
    for disposition in [
        "ExprCarrier::DecodedNode",
        "ExprCarrier::StructuralList",
        "ExprCarrier::UndecodableHead",
        "ExprCarrier::Atom",
        "ExprCarrier::MetadataMap",
        "ExprCarrier::MetadataExpression",
        "ExprCarrier::MalformedLegacyList",
    ] {
        assert!(
            production.contains(disposition),
            "missing explicit carrier disposition `{disposition}`"
        );
    }
}
