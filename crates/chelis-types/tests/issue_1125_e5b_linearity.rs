//! chelis#1125 PP7/E5b: the linearity reader uses the shared total carrier
//! view without changing binding-generation or diagnostic semantics.

use chelis_deep::{Atom, Expr, List, Metadata};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::{CheckedProgram, check_ir_program, check_linearity, check_typed_program};
use std::collections::BTreeSet;
use syn::visit::{self, Visit};

const CARRIER_ROLES: [&str; 7] = [
    "DecodedNode",
    "StructuralList",
    "UndecodableHead",
    "Atom",
    "MetadataMap",
    "MetadataExpression",
    "MalformedLegacyList",
];

fn direct_carrier_call(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::MethodCall(call) => call.method == "carrier",
        syn::Expr::Group(group) => direct_carrier_call(&group.expr),
        syn::Expr::Paren(paren) => direct_carrier_call(&paren.expr),
        _ => false,
    }
}

fn carrier_roles(pattern: &syn::Pat) -> BTreeSet<String> {
    #[derive(Default)]
    struct RoleScan {
        roles: BTreeSet<String>,
    }

    impl<'ast> Visit<'ast> for RoleScan {
        fn visit_path(&mut self, path: &'ast syn::Path) {
            if let Some(role) = path
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
                && CARRIER_ROLES.contains(&role.as_str())
            {
                self.roles.insert(role);
            }
            visit::visit_path(self, path);
        }
    }

    let mut scan = RoleScan::default();
    scan.visit_pat(pattern);
    scan.roles
}

fn macro_mentions_carrier(mac: &syn::Macro) -> bool {
    let tokens = mac.tokens.to_string();
    tokens.contains(". carrier (") || tokens.contains(":: carrier (")
}

fn carrier_totality_findings(source: &str) -> Vec<String> {
    let file = syn::parse_file(source).expect("linearity source parses as Rust");

    #[derive(Default)]
    struct CarrierTotalityScan {
        carrier_calls: usize,
        direct_matches: usize,
        findings: Vec<String>,
    }

    impl<'ast> Visit<'ast> for CarrierTotalityScan {
        fn visit_expr_match(&mut self, match_expr: &'ast syn::ExprMatch) {
            if direct_carrier_call(&match_expr.expr) {
                self.direct_matches += 1;
                let present = match_expr
                    .arms
                    .iter()
                    .flat_map(|arm| carrier_roles(&arm.pat))
                    .collect::<BTreeSet<_>>();
                let missing = CARRIER_ROLES
                    .iter()
                    .filter(|role| !present.contains(**role))
                    .copied()
                    .collect::<Vec<_>>();
                if !missing.is_empty() {
                    self.findings
                        .push(format!("carrier match missing {}", missing.join(", ")));
                }
            }
            visit::visit_expr_match(self, match_expr);
        }

        fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
            if call.method == "carrier" {
                self.carrier_calls += 1;
            }
            visit::visit_expr_method_call(self, call);
        }

        fn visit_macro(&mut self, mac: &'ast syn::Macro) {
            if macro_mentions_carrier(mac) {
                self.findings
                    .push("carrier access hidden inside a macro".to_string());
            }
            visit::visit_macro(self, mac);
        }
    }

    let mut scan = CarrierTotalityScan::default();
    scan.visit_file(&file);
    if scan.carrier_calls != scan.direct_matches {
        scan.findings.push(format!(
            "{} carrier call(s) but {} direct carrier match(es)",
            scan.carrier_calls, scan.direct_matches
        ));
    }
    scan.findings
}

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
fn linearity_reader_has_only_role_total_carrier_matches_and_no_node_bridge() {
    let source = include_str!("../src/linearity.rs");
    let production = source
        .split("#[cfg(test)]")
        .next()
        .expect("linearity source has a production prefix");

    let carrier_findings = carrier_totality_findings(production);
    assert!(
        carrier_findings.is_empty(),
        "E5b requires every carrier access to be a direct, role-total match: {carrier_findings:?}"
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

#[test]
fn carrier_totality_ratchet_is_signature_and_visibility_independent() {
    let incomplete_adapters = [
        r#"
fn decoded_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        _ => None,
    }
}
"#,
        r#"
type DecodedParts<'a> = Option<(DeepTag, &'a Metadata, &'a [Expr])>;
fn decoded_parts(expr: &Expr) -> DecodedParts<'_> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        _ => None,
    }
}
"#,
        r#"
pub(crate) fn decoded_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        _ => None,
    }
}
"#,
        r#"
fn decoded_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    let carrier = expr.carrier();
    match carrier {
        ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        _ => None,
    }
}
"#,
    ];
    for adapter in incomplete_adapters {
        assert!(
            !carrier_totality_findings(adapter).is_empty(),
            "signature, alias, visibility, or local delegation must not bypass totality: {adapter}"
        );
    }

    let unrelated_option = r#"
fn maybe_name(enabled: bool) -> Option<&'static str> {
    enabled.then_some("name")
}
"#;
    assert!(
        carrier_totality_findings(unrelated_option).is_empty(),
        "ordinary Option helpers without carrier access are outside the totality ratchet"
    );
}
