//! chelis#1125 PP7/E5b: the linearity reader uses the shared total carrier
//! view without changing binding-generation or diagnostic semantics.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::{CheckedProgram, check_ir_program, check_linearity, check_typed_program};
use proc_macro2::{TokenStream, TokenTree};
use std::collections::BTreeSet;
use syn::ext::IdentExt;
use syn::visit::{self, Visit};

const CARRIER_ROLES: [&str; 6] = [
    "DecodedNode",
    "StructuralList",
    "UndecodableHead",
    "Atom",
    "MetadataMap",
    "MetadataExpression",
];

fn ident_is(ident: &syn::Ident, expected: &str) -> bool {
    ident.unraw() == expected
}

fn path_ends_with(path: &syn::Path, expected: &str) -> bool {
    path.segments
        .last()
        .is_some_and(|segment| ident_is(&segment.ident, expected))
}

fn direct_carrier_call(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::MethodCall(call) => ident_is(&call.method, "carrier"),
        syn::Expr::Call(call) => {
            matches!(&*call.func, syn::Expr::Path(path) if path_ends_with(&path.path, "carrier"))
        }
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

fn token_spellings(tokens: TokenStream, spellings: &mut Vec<String>) {
    for token in tokens {
        match token {
            TokenTree::Ident(ident) => {
                spellings.push(ident.to_string().trim_start_matches("r#").to_string());
            }
            TokenTree::Punct(punct) => spellings.push(punct.as_char().to_string()),
            TokenTree::Group(group) => token_spellings(group.stream(), spellings),
            TokenTree::Literal(_) => {}
        }
    }
}

fn macro_findings(mac: &syn::Macro) -> Vec<String> {
    let mut spellings = Vec::new();
    token_spellings(mac.tokens.clone(), &mut spellings);
    let mut findings = Vec::new();
    if spellings.iter().any(|spelling| spelling == "carrier") {
        findings.push("carrier access hidden inside a macro".to_string());
    }
    findings
}

fn cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && attr
                .parse_args::<syn::Path>()
                .is_ok_and(|path| path.is_ident("test"))
    })
}

fn carrier_totality_findings(source: &str) -> Vec<String> {
    let file = syn::parse_file(source).expect("linearity source parses as Rust");

    #[derive(Default)]
    struct CarrierTotalityScan {
        carrier_references: usize,
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
            if ident_is(&call.method, "carrier") {
                self.carrier_references += 1;
            }
            visit::visit_expr_method_call(self, call);
        }

        fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
            if (path.qself.is_some() || path.path.segments.len() > 1)
                && path_ends_with(&path.path, "carrier")
            {
                self.carrier_references += 1;
            }
            visit::visit_expr_path(self, path);
        }

        fn visit_macro(&mut self, mac: &'ast syn::Macro) {
            self.findings.extend(macro_findings(mac));
            visit::visit_macro(self, mac);
        }

        fn visit_item_mod(&mut self, item_mod: &'ast syn::ItemMod) {
            if !cfg_test(&item_mod.attrs) {
                visit::visit_item_mod(self, item_mod);
            }
        }
    }

    let mut scan = CarrierTotalityScan::default();
    scan.visit_file(&file);
    if scan.carrier_references != scan.direct_matches {
        scan.findings.push(format!(
            "{} carrier reference(s) but {} direct carrier match(es)",
            scan.carrier_references, scan.direct_matches
        ));
    }
    scan.findings
}

#[derive(Clone, Copy)]
enum Entry {
    Ir,
    Typed,
}

fn checked_program(source: &str, entry: Entry) -> CheckedProgram {
    let declarations = parse_str(source).expect("Surf fixture parses");
    let program = desugar_program(&declarations).expect("Surf fixture must desugar");
    match entry {
        Entry::Ir => check_ir_program(&program).expect("IR entry type-checks the fixture"),
        Entry::Typed => check_typed_program(&program).expect("typed entry type-checks the fixture"),
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
fn nested_destructure_alias_single_consume_matches_across_entries() {
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

    let typed = ordered_linearity_diagnostics(&checked_program(source, Entry::Typed));
    let ir = ordered_linearity_diagnostics(&checked_program(source, Entry::Ir));
    assert!(typed.is_empty(), "positive control: {typed:?}");
    assert_eq!(typed, ir, "checker entry choice must not change acceptance");
}

#[test]
fn nested_destructure_alias_reuse_preserves_diagnostic_order_across_entries() {
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

    let typed = ordered_linearity_diagnostics(&checked_program(source, Entry::Typed));
    let ir = ordered_linearity_diagnostics(&checked_program(source, Entry::Ir));
    assert_eq!(
        typed, ir,
        "binding identity, diagnostic ownership, and order must match"
    );
    assert!(
        typed.iter().any(|message| {
            message.contains("[UseAfterConsume]")
                && message.contains("variable `left`")
                && message.contains("consumed by realize")
        }),
        "negative control must retain the component/alias consume diagnostic: {typed:?}"
    );
}

#[test]
fn linearity_reader_has_only_role_total_carrier_matches() {
    let source = include_str!("../src/linearity.rs");

    let carrier_findings = carrier_totality_findings(source);
    assert!(
        carrier_findings.is_empty(),
        "E5b requires every carrier access to be a direct, role-total match: {carrier_findings:?}"
    );
    for disposition in [
        "ExprCarrier::DecodedNode",
        "ExprCarrier::StructuralList",
        "ExprCarrier::UndecodableHead",
        "ExprCarrier::Atom",
        "ExprCarrier::MetadataMap",
        "ExprCarrier::MetadataExpression",
    ] {
        assert!(
            source.contains(disposition),
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
        r#"
fn decoded_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match Expr::carrier(expr) {
        ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        _ => None,
    }
}
"#,
        r#"
macro_rules! invoke {
    ($value:expr, $method:ident) => {
        $value.$method()
    };
}
fn decoded_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    invoke!(expr, carrier)
}
"#,
        r#"
fn decoded_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    let classify = Expr::carrier;
    match classify(expr) {
        ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        _ => None,
    }
}
"#,
        r#"
// #[cfg(test)] is documentation here, not an item attribute.
fn decoded_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    let carrier = expr.carrier();
    match carrier {
        ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        _ => None,
    }
}
"#,
        r#"
#[cfg(not(test))]
mod production {
    fn decoded_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
        let carrier = expr.carrier();
        match carrier {
            ExprCarrier::DecodedNode(tag, metadata, children) => {
                Some((tag, metadata, children))
            }
            _ => None,
        }
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

    let test_only_carrier = r#"
#[cfg(test)]
mod tests {
    fn test_helper(expr: &Expr) {
        let _ = expr.carrier();
    }
}
"#;
    assert!(
        carrier_totality_findings(test_only_carrier).is_empty(),
        "an actual cfg(test) module is outside the production ratchet"
    );
}
