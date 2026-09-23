//! Unit tests for the invariant-predicate grammar and amenability
//! classifier (RFC D-PRED / D-WF). Predicate fn nodes are parsed from
//! Deep source so the tests are pinned to the real encoding the
//! desugarer produces (verified against the surf desugar probe in W2
//! pre-flight).

use super::*;
use chelis_deep::Span;
use chelis_deep::ast::{Metadata, UnknownFormData};
use chelis_deep::parser::parse_str;
use std::fs;
use std::path::{Path, PathBuf};
use syn::ext::IdentExt;
use syn::visit::Visit;

/// Parse a single Deep expression (the predicate fn node).
fn fnnode(src: &str) -> Expr {
    let exprs = parse_str(src).expect("predicate fn parses");
    assert_eq!(exprs.len(), 1, "expected one top-level expr");
    exprs.into_iter().next().unwrap()
}

// ---------------------------------------------------------------------------
// Representative predicates (one per amenability class) in the canonical
// `(fn {} (params {} p) <body>)` schema, matching the desugar probe.
// ---------------------------------------------------------------------------

/// Linear: `p.value >= 0.0 and p.value <= 1.0`.
const LINEAR: &str = "(fn {} (params {} p) \
    (app {} (var {} and) \
        (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0)) \
        (app {} (var {} lte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 1.0))))";

/// Polynomial: `p.value * p.value <= 1.0`.
const POLYNOMIAL: &str = "(fn {} (params {} p) \
    (app {} (var {} lte) \
        (app {} (var {} mul) (access {} (var {} p) value) (access {} (var {} p) value)) \
        (lit {type: (t-prim {} f32)} 1.0)))";

/// Transcendental: `exp(p.value) <= 3.0`.
const TRANSCENDENTAL: &str = "(fn {} (params {} p) \
    (app {} (var {} lte) \
        (app {} (var {} exp) (access {} (var {} p) value)) \
        (lit {type: (t-prim {} f32)} 3.0)))";

/// Simplex tolerance band (W6 flagship):
/// `sum(p.weights) >= 1.0 - eps and sum(p.weights) <= 1.0 + eps`.
const SIMPLEX: &str = "(fn {} (params {} p) \
    (app {} (var {} and) \
        (app {} (var {} gte) (app {} (var {} sum) (access {} (var {} p) weights)) \
            (app {} (var {} sub) (lit {type: (t-prim {} f32)} 1.0) (var {} eps))) \
        (app {} (var {} lte) (app {} (var {} sum) (access {} (var {} p) weights)) \
            (app {} (var {} add) (lit {type: (t-prim {} f32)} 1.0) (var {} eps)))))";

// ===========================================================================
// PredAmenability as_str / from_str round-trip
// ===========================================================================

#[test]
fn amenability_as_str_canonical() {
    assert_eq!(PredAmenability::Linear.as_str(), "linear");
    assert_eq!(PredAmenability::Polynomial.as_str(), "polynomial");
    assert_eq!(PredAmenability::Transcendental.as_str(), "transcendental");
    assert_eq!(PredAmenability::Opaque.as_str(), "opaque");
}

#[test]
fn amenability_from_str_round_trip() {
    for a in [
        PredAmenability::Linear,
        PredAmenability::Polynomial,
        PredAmenability::Transcendental,
        PredAmenability::Opaque,
    ] {
        assert_eq!(PredAmenability::from_str(a.as_str()), Some(a));
    }
}

#[test]
fn amenability_from_str_rejects_unknown() {
    assert_eq!(PredAmenability::from_str("Linear"), None);
    assert_eq!(PredAmenability::from_str(""), None);
    assert_eq!(PredAmenability::from_str("nonlinear"), None);
}

// ===========================================================================
// classify_predicate: one per class
// ===========================================================================

#[test]
fn classify_linear() {
    assert_eq!(classify_predicate(&fnnode(LINEAR)), PredAmenability::Linear);
}

#[test]
fn classify_polynomial() {
    assert_eq!(
        classify_predicate(&fnnode(POLYNOMIAL)),
        PredAmenability::Polynomial
    );
}

#[test]
fn classify_transcendental() {
    assert_eq!(
        classify_predicate(&fnnode(TRANSCENDENTAL)),
        PredAmenability::Transcendental
    );
}

#[test]
fn classify_simplex_is_linear() {
    // The simplex tolerance band is affine: `sum` of a field is a linear
    // combination, the band bounds are affine in the module constant eps.
    assert_eq!(
        classify_predicate(&fnnode(SIMPLEX)),
        PredAmenability::Linear
    );
}

// ===========================================================================
// Misclassification probes (negative parity for the classifier)
// ===========================================================================

#[test]
fn polynomial_disguised_as_linear() {
    // `p.value * p.value` LOOKS like two field accesses; it must NOT be
    // classified Linear. A product of two non-constant subterms is
    // Polynomial.
    assert_eq!(
        classify_predicate(&fnnode(POLYNOMIAL)),
        PredAmenability::Polynomial
    );
}

#[test]
fn scaling_by_constant_stays_linear() {
    // `p.value * 2.0 <= 1.0`: a product where one operand is a literal
    // is affine, NOT polynomial.
    let src = "(fn {} (params {} p) \
        (app {} (var {} lte) \
            (app {} (var {} mul) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 2.0)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(classify_predicate(&fnnode(src)), PredAmenability::Linear);
}

#[test]
fn scaling_by_module_constant_stays_linear() {
    // `p.value * scale <= 1.0` where `scale` is an in-module constant
    // (a bare var that is not the binder) is affine.
    let src = "(fn {} (params {} p) \
        (app {} (var {} lte) \
            (app {} (var {} mul) (access {} (var {} p) value) (var {} scale)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(classify_predicate(&fnnode(src)), PredAmenability::Linear);
}

#[test]
fn transcendental_dominates_polynomial() {
    // `exp(p.value * p.value) <= 3.0`: a polynomial argument under a
    // transcendental call classifies Transcendental.
    let src = "(fn {} (params {} p) \
        (app {} (var {} lte) \
            (app {} (var {} exp) \
                (app {} (var {} mul) (access {} (var {} p) value) (access {} (var {} p) value))) \
            (lit {type: (t-prim {} f32)} 3.0)))";
    assert_eq!(
        classify_predicate(&fnnode(src)),
        PredAmenability::Transcendental
    );
}

#[test]
fn nested_polynomial_under_if_lifts() {
    // Polynomial buried inside an `if` branch still lifts the class.
    let src = "(fn {} (params {} p) \
        (if {} (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0)) \
            (app {} (var {} lte) \
                (app {} (var {} mul) (access {} (var {} p) value) (access {} (var {} p) value)) \
                (lit {type: (t-prim {} f32)} 1.0)) \
            (lit {type: (t-prim {} bool)} false)))";
    assert_eq!(
        classify_predicate(&fnnode(src)),
        PredAmenability::Polynomial
    );
}

#[test]
fn out_of_grammar_classifies_opaque() {
    // A general function call (`my_helper`) is out of grammar.
    let src = "(fn {} (params {} p) \
        (app {} (var {} lte) \
            (app {} (var {} my_helper) (access {} (var {} p) value)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(classify_predicate(&fnnode(src)), PredAmenability::Opaque);
}

#[test]
fn malformed_fn_classifies_opaque() {
    // Not a fn node at all.
    let src = "(app {} (var {} and) (lit {type: (t-prim {} bool)} true))";
    assert_eq!(classify_predicate(&fnnode(src)), PredAmenability::Opaque);
}

// ===========================================================================
// predicate_in_grammar: accept / reject table
// ===========================================================================

#[test]
fn grammar_accepts_representative_predicates() {
    assert!(predicate_in_grammar(&fnnode(LINEAR)).is_ok());
    assert!(predicate_in_grammar(&fnnode(POLYNOMIAL)).is_ok());
    assert!(predicate_in_grammar(&fnnode(TRANSCENDENTAL)).is_ok());
    assert!(predicate_in_grammar(&fnnode(SIMPLEX)).is_ok());
}

#[test]
fn grammar_accepts_all_intrinsics() {
    for f in INTRINSIC_WHITELIST {
        let src = format!(
            "(fn {{}} (params {{}} p) \
                (app {{}} (var {{}} lte) \
                    (app {{}} (var {{}} {f}) (access {{}} (var {{}} p) value)) \
                    (lit {{type: (t-prim {{}} f32)}} 1.0)))"
        );
        assert!(
            predicate_in_grammar(&fnnode(&src)).is_ok(),
            "intrinsic `{f}` should be in grammar"
        );
    }
}

#[test]
fn grammar_accepts_if_and_division() {
    // `if` and `/` are admitted (division is partial — totality not
    // guaranteed, per D-WF).
    let src = "(fn {} (params {} p) \
        (if {} (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0)) \
            (app {} (var {} lte) \
                (app {} (var {} div) (lit {type: (t-prim {} f32)} 1.0) (access {} (var {} p) value)) \
                (lit {type: (t-prim {} f32)} 1.0)) \
            (lit {type: (t-prim {} bool)} false)))";
    assert!(predicate_in_grammar(&fnnode(src)).is_ok());
}

#[test]
fn grammar_rejects_general_call() {
    let src = "(fn {} (params {} p) \
        (app {} (var {} my_helper) (access {} (var {} p) value)))";
    assert_eq!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::DisallowedCall("my_helper".to_string()))
    );
}

#[test]
fn grammar_rejects_match() {
    let src = "(fn {} (params {} p) \
        (match {} (access {} (var {} p) value) \
            (arm {} (pat-wild {}) () (lit {type: (t-prim {} bool)} true))))";
    assert!(matches!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::DisallowedNode(_))
    ));
}

#[test]
fn grammar_rejection_names_the_unknown_form_head() {
    let src = "(fn {} (params {} p) (future_form {} true))";
    assert_eq!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::DisallowedNode(
            "unknown form `future_form`".to_string()
        )),
        "[04-TOT-3] requires the diagnostic to identify the malformed tag"
    );
}

#[test]
fn grammar_rejects_lambda() {
    let src = "(fn {} (params {} p) \
        (fn {} (params {} q) (access {} (var {} q) value)))";
    assert!(matches!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::DisallowedNode(_))
    ));
}

#[test]
fn grammar_rejects_record_construction() {
    let src = "(fn {} (params {} p) \
        (record {} Foo (kv {} x (lit {type: (t-prim {} f32)} 1.0))))";
    assert!(matches!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::DisallowedNode(_))
    ));
}

#[test]
fn grammar_rejects_sum_over_non_field() {
    // `sum` applied to a literal (not a binder field projection).
    let src = "(fn {} (params {} p) \
        (app {} (var {} gte) \
            (app {} (var {} sum) (lit {type: (t-prim {} f32)} 1.0)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::BadSum)
    );
}

#[test]
fn grammar_rejects_non_fn_top() {
    let src = "(app {} (var {} and) (lit {type: (t-prim {} bool)} true))";
    assert!(matches!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::NotAPredicateFn(_))
    ));
}

#[test]
fn grammar_reads_constructed_and_parsed_decoded_nodes_identically() {
    let span = Span::new(3, 9);
    let constructed = Expr::node(
        DeepTag::Var,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name("value".to_string()), span)],
        span,
    );
    let parsed = fnnode("(var {} value)");

    assert_eq!(check_in_grammar(&constructed), Ok(()));
    assert_eq!(check_in_grammar(&parsed), Ok(()));
}

#[test]
fn grammar_rejects_each_nonexpression_carrier_with_its_exact_role() {
    let span = Span::new(3, 9);
    let cases = [
        (
            Expr::BareList(vec![Expr::Atom(Atom::Name("item".to_string()), span)], span),
            "bare list",
        ),
        (
            Expr::UnknownForm(Box::new(UnknownFormData {
                head: "future-form".to_string(),
                meta: Metadata::default(),
                children: vec![],
                span,
            })),
            "unknown form `future-form`",
        ),
        (Expr::Map(Metadata::default(), span), "map"),
        (
            Expr::MetaExpr(
                chelis_deep::MetaExpr {
                    metadata: Metadata::default(),
                    expr: Box::new(Expr::Atom(Atom::Bool(true), span)),
                },
                span,
            ),
            "meta-expr",
        ),
    ];

    for (expr, expected) in cases {
        assert_eq!(
            check_in_grammar(&expr),
            Err(PredGrammarError::DisallowedNode(expected.to_string()))
        );
    }
}

#[test]
fn predicate_reader_has_no_node_to_list_bridge() {
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let sources = production_rust_sources(&source_root);
    assert!(
        sources.iter().any(|path| path.ends_with("lib.rs")),
        "production audit must include the crate root"
    );

    let calls = sources
        .into_iter()
        .flat_map(|path| {
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            let syntax = syn::parse_file(&source)
                .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
            audit_syntax(&path.display().to_string(), &syntax)
        })
        .collect::<Vec<_>>();
    assert!(
        calls.is_empty(),
        "predicate grammar reserves method, qualified, imported, and macro `to_list` spellings \
         so it cannot reconstruct a Node list; found {:?}",
        calls
    );
}

fn production_rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut sources = Vec::new();
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        {
            let entry = entry.expect("production source directory entry");
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs")
                && path.file_name().is_none_or(|name| name != "tests.rs")
            {
                sources.push(path);
            }
        }
    }
    sources.sort();
    sources
}

fn ident_is(ident: &proc_macro2::Ident, expected: &str) -> bool {
    ident.unraw() == expected
}

fn use_tree_mentions_to_list(tree: &syn::UseTree) -> bool {
    match tree {
        syn::UseTree::Path(path) => {
            ident_is(&path.ident, "to_list") || use_tree_mentions_to_list(&path.tree)
        }
        syn::UseTree::Name(name) => ident_is(&name.ident, "to_list"),
        syn::UseTree::Rename(rename) => {
            ident_is(&rename.ident, "to_list") || ident_is(&rename.rename, "to_list")
        }
        syn::UseTree::Group(group) => group.items.iter().any(use_tree_mentions_to_list),
        syn::UseTree::Glob(_) => false,
    }
}

fn tokens_mention_to_list(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(ident) => ident_is(&ident, "to_list"),
        proc_macro2::TokenTree::Group(group) => tokens_mention_to_list(group.stream()),
        proc_macro2::TokenTree::Punct(_) | proc_macro2::TokenTree::Literal(_) => false,
    })
}

fn macro_mentions_to_list(mac: &syn::Macro) -> bool {
    mac.path
        .segments
        .last()
        .is_some_and(|segment| ident_is(&segment.ident, "to_list"))
        || tokens_mention_to_list(mac.tokens.clone())
}

fn attribute_mentions_to_list(attribute: &syn::Attribute) -> bool {
    if attribute
        .path()
        .segments
        .last()
        .is_some_and(|segment| ident_is(&segment.ident, "to_list"))
    {
        return true;
    }
    match &attribute.meta {
        syn::Meta::List(list) => tokens_mention_to_list(list.tokens.clone()),
        syn::Meta::NameValue(name_value) => {
            struct ToListPathScan(bool);
            impl<'ast> Visit<'ast> for ToListPathScan {
                fn visit_path(&mut self, path: &'ast syn::Path) {
                    self.0 |= path
                        .segments
                        .last()
                        .is_some_and(|segment| ident_is(&segment.ident, "to_list"));
                    syn::visit::visit_path(self, path);
                }
            }
            let mut scan = ToListPathScan(false);
            scan.visit_expr(&name_value.value);
            scan.0
        }
        syn::Meta::Path(_) => false,
    }
}

fn audit_source(source: &str) -> Vec<String> {
    let syntax = syn::parse_file(source).expect("audit fixture parses");
    audit_syntax("fixture", &syntax)
}

fn audit_syntax(label: &str, syntax: &syn::File) -> Vec<String> {
    let mut audit = NodeToListSpellingAudit {
        label,
        findings: Vec::new(),
    };
    audit.visit_file(syntax);
    audit.findings
}

struct NodeToListSpellingAudit<'a> {
    label: &'a str,
    findings: Vec<String>,
}

impl NodeToListSpellingAudit<'_> {
    fn record(&mut self, kind: &str) {
        self.findings.push(format!("{}: {kind}", self.label));
    }
}

impl<'ast> Visit<'ast> for NodeToListSpellingAudit<'_> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if ident_is(&call.method, "to_list") {
            self.record("reserved method spelling");
        }
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path
            .path
            .segments
            .last()
            .is_some_and(|segment| ident_is(&segment.ident, "to_list"))
            && (path.qself.is_some() || path.path.segments.len() > 1)
        {
            self.record("reserved qualified path spelling");
        }
        syn::visit::visit_expr_path(self, path);
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if use_tree_mentions_to_list(&item.tree) {
            self.record("reserved imported path spelling");
        }
        syn::visit::visit_item_use(self, item);
    }

    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        if attribute_mentions_to_list(attribute) {
            self.record("reserved attribute identifier");
        }
        syn::visit::visit_attribute(self, attribute);
    }

    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        if item
            .ident
            .as_ref()
            .is_some_and(|ident| ident_is(ident, "to_list"))
        {
            self.record("reserved macro definition");
        }
        syn::visit::visit_item_macro(self, item);
    }

    fn visit_item_extern_crate(&mut self, item: &'ast syn::ItemExternCrate) {
        if ident_is(&item.ident, "to_list")
            || item
                .rename
                .as_ref()
                .is_some_and(|(_, rename)| ident_is(rename, "to_list"))
        {
            self.record("reserved extern-crate identifier");
        }
        syn::visit::visit_item_extern_crate(self, item);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if macro_mentions_to_list(mac) {
            self.record("reserved macro identifier");
        }
        syn::visit::visit_macro(self, mac);
    }
}

#[test]
fn node_to_list_audit_rejects_method_and_aliased_associated_calls() {
    for source in [
        "use chelis_deep::node::Node; fn bridge(node: Node) { let _ = node.to_list(span); }",
        "use chelis_deep::node::Node; fn bridge() { let _ = Node::to_list(node, span); }",
        "use chelis_deep::node::Node; type DeepNode = Node; fn bridge() { let _ = DeepNode::to_list(node, span); }",
        "fn bridge() { let adapter = chelis_deep::node::Node::to_list; let _ = adapter(node, span); }",
        "type DeepNode = chelis_deep::node::Node; fn bridge() { let adapter = <DeepNode>::to_list; let _ = adapter(node, span); }",
    ] {
        let calls = audit_source(source);
        assert!(
            !calls.is_empty(),
            "bridge spelling escaped the predicate source audit: {source}"
        );
    }
}

#[test]
fn node_to_list_audit_allows_unrelated_same_named_functions() {
    let source = "
        fn to_list(value: u32) -> u32 { value }
        fn unrelated_conversion(value: u32) {
            let _ = to_list(value);
        }
    ";
    let calls = audit_source(source);
    assert!(
        calls.is_empty(),
        "the structural ratchet reserves receiver and qualified spellings, not an unqualified \
         free function: {:?}",
        calls
    );
}

#[test]
fn node_to_list_audit_reserves_all_receiver_spellings_without_type_inference() {
    for source in [
        "use chelis_deep::node; fn bridge() { let _ = node::Node::to_list(value, span); }",
        "use chelis_deep::node as deep_node; fn bridge() { let _ = deep_node::Node::to_list(value, span); }",
        "use chelis_deep::node::*; fn bridge() { let _ = Node::to_list(value, span); }",
        "extern crate chelis_deep as deep; fn bridge() { let _ = deep::node::Node::to_list(value, span); }",
        "fn bridge() { use chelis_deep::node::Node as LocalNode; let _ = LocalNode::to_list(value, span); }",
        "fn bridge() { type LocalNode = chelis_deep::node::Node; let _ = LocalNode::to_list(value, span); }",
        "use chelis_deep::node::Node; fn bridge() { let f = |node: &Node| node.to_list(span); }",
        "fn make_node() -> chelis_deep::node::Node { todo!() } fn bridge() { let node = make_node(); node.to_list(span); }",
        "fn bridge(node: Box<chelis_deep::node::Node>) { node.to_list(span); }",
        "fn bridge(node: chelis_deep::node::Node) { let f = |node: Unrelated| node.to_list(span); let _ = node; }",
        "use chelis_deep::node::Node::to_list as bridge; fn call() { bridge(node, span); }",
        "use helper::convert as to_list; fn call() { to_list(node); }",
        "fn bridge() { quote::quote! { node.to_list(span) }; }",
        "fn bridge() { to_list!(); }",
        "fn bridge() { helper::to_list!(); }",
        "macro_rules! to_list { () => {} }",
        "#[to_list] fn bridge() {}",
        "fn bridge(node: &chelis_deep::node::Node, span: chelis_deep::Span) { let _ = node.r#to_list(span); }",
        "fn bridge() { helper::r#to_list!(); }",
        "macro_rules! r#to_list { () => {} }",
        "#[r#to_list] fn bridge() {}",
        "extern crate r#to_list;",
        "extern crate helper as r#to_list;",
    ] {
        let calls = audit_source(source);
        assert!(
            !calls.is_empty(),
            "reserved bridge spelling escaped the predicate source audit: {source}"
        );
    }

    assert!(
        audit_source(r#"fn bridge() { format_args!("to_list"); }"#).is_empty(),
        "literal content is not an identifier use of the reserved spelling"
    );
}

#[test]
fn unknown_form_parameter_head_does_not_enter_binder_scope() {
    let span = Span::new(3, 9);
    let parameter = Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future_parameter".to_string(),
        meta: Metadata::default(),
        children: vec![],
        span,
    }));
    assert_eq!(
        binder_name(&parameter),
        None,
        "an UnknownForm head is not an authored parameter binder"
    );

    let predicate = Expr::node(
        DeepTag::Fn,
        Metadata::default(),
        vec![
            Expr::node(DeepTag::Params, Metadata::default(), vec![parameter], span),
            Expr::node(
                DeepTag::Var,
                Metadata::default(),
                vec![Expr::Atom(Atom::Name("future_parameter".to_string()), span)],
                span,
            ),
        ],
        span,
    );
    assert!(fn_parts(&predicate).is_none());
    assert!(matches!(
        predicate_in_grammar(&predicate),
        Err(PredGrammarError::NotAPredicateFn(_))
    ));
}

#[test]
fn annotated_bare_list_parameter_enters_binder_scope() {
    let span = Span::new(4, 10);
    let parameter = Expr::BareList(
        vec![
            Expr::Atom(Atom::Name("structural_parameter".to_string()), span),
            Expr::Map(Metadata::default(), span),
        ],
        span,
    );
    assert_eq!(
        binder_name(&parameter),
        Some("structural_parameter".to_string()),
        "the stamped annotated-parameter carrier is a legal structural binder"
    );

    let predicate = Expr::node(
        DeepTag::Fn,
        Metadata::default(),
        vec![
            Expr::node(DeepTag::Params, Metadata::default(), vec![parameter], span),
            Expr::node(
                DeepTag::Var,
                Metadata::default(),
                vec![Expr::Atom(
                    Atom::Name("structural_parameter".to_string()),
                    span,
                )],
                span,
            ),
        ],
        span,
    );
    assert_eq!(
        fn_parts(&predicate).map(|(binder, _)| binder),
        Some("structural_parameter".to_string())
    );
    assert_eq!(predicate_in_grammar(&predicate), Ok(()));
    assert!(predicate_free_vars(&predicate).is_empty());
    assert_eq!(classify_predicate(&predicate), PredAmenability::Linear);
}

#[test]
fn malformed_parameter_carriers_never_mint_binder_scope() {
    let span = Span::new(5, 11);
    // A wrong-arity `var` parameter has no in-memory spelling: `Node`
    // construction rejects it, so the structural list is the one malformed
    // parameter carrier left.
    let malformed = [Expr::BareList(
        vec![
            Expr::Atom(Atom::Name("structural_parameter".to_string()), span),
            Expr::Map(Metadata::default(), span),
            Expr::Atom(Atom::Name("extra".to_string()), span),
        ],
        span,
    )];

    for parameter in malformed {
        assert_eq!(binder_name(&parameter), None);
        let predicate = Expr::node(
            DeepTag::Fn,
            Metadata::default(),
            vec![
                Expr::node(DeepTag::Params, Metadata::default(), vec![parameter], span),
                Expr::node(
                    DeepTag::Var,
                    Metadata::default(),
                    vec![Expr::Atom(Atom::Name("external".to_string()), span)],
                    span,
                ),
            ],
            span,
        );
        assert!(fn_parts(&predicate).is_none());
        assert!(matches!(
            predicate_in_grammar(&predicate),
            Err(PredGrammarError::NotAPredicateFn(_))
        ));
        assert!(predicate_free_vars(&predicate).is_empty());
        assert_eq!(classify_predicate(&predicate), PredAmenability::Opaque);
    }
}

#[test]
fn grammar_accepts_sum_over_field() {
    let src = "(fn {} (params {} p) \
        (app {} (var {} gte) \
            (app {} (var {} sum) (access {} (var {} p) weights)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert!(predicate_in_grammar(&fnnode(src)).is_ok());
}

// ===========================================================================
// predicate_free_vars
// ===========================================================================

#[test]
fn free_vars_binder_only_is_empty() {
    // LINEAR references only the binder `p` and its field; no free vars.
    assert!(predicate_free_vars(&fnnode(LINEAR)).is_empty());
}

#[test]
fn free_vars_picks_up_module_constant() {
    // SIMPLEX references `eps`, an in-module constant, plus the binder.
    assert_eq!(
        predicate_free_vars(&fnnode(SIMPLEX)),
        vec!["eps".to_string()]
    );
}

#[test]
fn free_vars_excludes_field_selectors() {
    // The field names `value` are selectors, never variables.
    let vars = predicate_free_vars(&fnnode(POLYNOMIAL));
    assert!(!vars.iter().any(|v| v == "value"));
    assert!(vars.is_empty());
}

#[test]
fn free_vars_do_not_read_unknown_form_children() {
    let span = Span::new(3, 9);
    let free_var = || {
        Expr::node(
            DeepTag::Var,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("external".to_string()), span)],
            span,
        )
    };
    let predicate = |body| {
        Expr::node(
            DeepTag::Fn,
            Metadata::default(),
            vec![
                Expr::node(
                    DeepTag::Params,
                    Metadata::default(),
                    vec![Expr::Atom(Atom::Name("p".to_string()), span)],
                    span,
                ),
                body,
            ],
            span,
        )
    };

    let successor_unknown = predicate(Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future-form".to_string(),
        meta: Metadata::default(),
        children: vec![free_var()],
        span,
    })));
    assert!(
        predicate_free_vars(&successor_unknown).is_empty(),
        "UnknownForm children are not predicate free-variable scope"
    );
    // Negative control: the same reference as a direct body is free.
    assert_eq!(
        predicate_free_vars(&predicate(free_var())),
        vec!["external".to_string()]
    );
}

#[test]
fn free_vars_dedup_in_order() {
    // `p.value * a + b * a` references `a` (twice) and `b`; dedup keeps
    // first-seen order.
    let src = "(fn {} (params {} p) \
        (app {} (var {} gte) \
            (app {} (var {} add) \
                (app {} (var {} mul) (access {} (var {} p) value) (var {} a)) \
                (app {} (var {} mul) (var {} b) (var {} a))) \
            (lit {type: (t-prim {} f32)} 0.0)))";
    assert_eq!(
        predicate_free_vars(&fnnode(src)),
        vec!["a".to_string(), "b".to_string()]
    );
}

// ===========================================================================
// sum-expansion classification (D-PRED / D-WF)
// ===========================================================================

#[test]
fn sum_over_field_classifies_linear() {
    // `sum(p.weights) >= 1.0`: a sum of scalar terms is affine.
    let src = "(fn {} (params {} p) \
        (app {} (var {} gte) \
            (app {} (var {} sum) (access {} (var {} p) weights)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(classify_predicate(&fnnode(src)), PredAmenability::Linear);
}

#[test]
fn sum_squared_is_polynomial() {
    // `sum(p.weights) * sum(p.weights) <= 1.0`: a product of two
    // non-constant sums is polynomial.
    let src = "(fn {} (params {} p) \
        (app {} (var {} lte) \
            (app {} (var {} mul) \
                (app {} (var {} sum) (access {} (var {} p) weights)) \
                (app {} (var {} sum) (access {} (var {} p) weights))) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(
        classify_predicate(&fnnode(src)),
        PredAmenability::Polynomial
    );
}

#[test]
fn intrinsic_whitelist_is_eight() {
    assert_eq!(INTRINSIC_WHITELIST.len(), 8);
    for f in ["abs", "min", "max", "sqrt", "exp", "log", "sin", "cos"] {
        assert!(INTRINSIC_WHITELIST.contains(&f), "missing {f}");
    }
}
