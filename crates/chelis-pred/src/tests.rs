//! Unit tests for the invariant-predicate grammar and amenability
//! classifier (RFC D-PRED / D-WF). Predicate fn nodes are parsed from
//! Deep source so the tests are pinned to the real encoding the
//! desugarer produces (verified against the surf desugar probe in W2
//! pre-flight).

use super::*;
use chelis_deep::Span;
use chelis_deep::ast::{List, Metadata, UnknownFormData};
use chelis_deep::parser::parse_str;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
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
fn grammar_reads_successor_and_legacy_decoded_nodes_identically() {
    let span = Span::new(3, 9);
    let successor = Expr::node(
        DeepTag::Var,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name("value".to_string()), span)],
        span,
    );
    let legacy = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::Var), span),
                Expr::Map(Metadata::default(), span),
                Expr::Atom(Atom::Name("value".to_string()), span),
            ],
        },
        span,
    );

    assert_eq!(check_in_grammar(&successor), Ok(()));
    assert_eq!(check_in_grammar(&legacy), Ok(()));
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
            "unknown form",
        ),
        (
            Expr::List(
                List {
                    elements: vec![
                        Expr::Atom(Atom::Name("future-legacy".to_string()), span),
                        Expr::Map(Metadata::default(), span),
                    ],
                },
                span,
            ),
            "malformed list",
        ),
        (
            Expr::List(
                List {
                    elements: vec![
                        Expr::Atom(Atom::Tag(DeepTag::Var), span),
                        Expr::Atom(Atom::Name("not-metadata".to_string()), span),
                    ],
                },
                span,
            ),
            "malformed list",
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

    let parsed = sources
        .into_iter()
        .map(|path| {
            let module = production_module_path(&source_root, &path);
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            let syntax = syn::parse_file(&source)
                .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
            AuditSource {
                label: path.display().to_string(),
                module,
                syntax,
            }
        })
        .collect::<Vec<_>>();
    let calls = audit_sources(&parsed);
    assert!(
        calls.is_empty(),
        "predicate grammar must consume ExprCarrier directly; found {:?}",
        calls
    );
}

fn production_module_path(root: &Path, path: &Path) -> Vec<String> {
    let relative = path.strip_prefix(root).unwrap_or_else(|error| {
        panic!("{} is outside {}: {error}", path.display(), root.display())
    });
    let mut components = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let file = components
        .pop()
        .expect("production Rust source has a file name");
    if !matches!(file.as_str(), "lib.rs" | "main.rs" | "mod.rs") {
        components.push(
            file.strip_suffix(".rs")
                .expect("production Rust source has .rs suffix")
                .to_string(),
        );
    }
    components
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

struct AuditSource {
    label: String,
    module: Vec<String>,
    syntax: syn::File,
}

#[derive(Default)]
struct ModuleSymbols {
    imports: BTreeMap<String, Vec<String>>,
    aliases: BTreeMap<String, syn::Type>,
    globs: Vec<Vec<String>>,
}

fn collect_use_bindings(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    imports: &mut BTreeMap<String, Vec<String>>,
    globs: &mut Vec<Vec<String>>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_use_bindings(&path.tree, prefix, imports, globs);
            prefix.pop();
        }
        syn::UseTree::Name(name) if name.ident == "self" => {
            let local = prefix
                .last()
                .expect("a self import has a path prefix")
                .clone();
            imports.insert(local, prefix.clone());
        }
        syn::UseTree::Name(name) => {
            let mut target = prefix.clone();
            target.push(name.ident.to_string());
            imports.insert(name.ident.to_string(), target);
        }
        syn::UseTree::Rename(rename) => {
            let mut target = prefix.clone();
            if rename.ident != "self" {
                target.push(rename.ident.to_string());
            }
            imports.insert(rename.rename.to_string(), target);
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_use_bindings(item, prefix, imports, globs);
            }
        }
        syn::UseTree::Glob(_) => globs.push(prefix.clone()),
    }
}

fn collect_module_symbols(
    module: &[String],
    items: &[syn::Item],
    modules: &mut BTreeMap<Vec<String>, ModuleSymbols>,
) {
    modules.entry(module.to_vec()).or_default();
    for item in items {
        collect_item_symbols(
            item,
            modules
                .get_mut(module)
                .expect("current module was enrolled"),
        );
        if let syn::Item::Mod(item) = item
            && let Some((_, items)) = &item.content
        {
            let mut child = module.to_vec();
            child.push(item.ident.to_string());
            collect_module_symbols(&child, items, modules);
        }
    }
}

fn collect_item_symbols(item: &syn::Item, symbols: &mut ModuleSymbols) {
    match item {
        syn::Item::Use(item) => collect_use_bindings(
            &item.tree,
            &mut Vec::new(),
            &mut symbols.imports,
            &mut symbols.globs,
        ),
        syn::Item::Type(item) => {
            symbols
                .aliases
                .insert(item.ident.to_string(), (*item.ty).clone());
        }
        syn::Item::ExternCrate(item) => {
            let local = item
                .rename
                .as_ref()
                .map_or_else(|| item.ident.to_string(), |(_, rename)| rename.to_string());
            symbols.imports.insert(local, vec![item.ident.to_string()]);
        }
        _ => {}
    }
}

fn audit_sources(sources: &[AuditSource]) -> Vec<String> {
    let mut modules = BTreeMap::new();
    for source in sources {
        collect_module_symbols(&source.module, &source.syntax.items, &mut modules);
    }

    let mut calls = Vec::new();
    for source in sources {
        let mut audit = CrateNodeToListAudit::new(&modules, source.module.clone(), &source.label);
        audit.visit_file(&source.syntax);
        calls.extend(audit.calls);
    }
    calls
}

fn audit_virtual_sources(sources: &[(&str, &str)]) -> Vec<String> {
    let parsed = sources
        .iter()
        .map(|(module, source)| AuditSource {
            label: if module.is_empty() {
                "crate".to_string()
            } else {
                module.to_string()
            },
            module: module
                .split("::")
                .filter(|segment| !segment.is_empty())
                .map(str::to_string)
                .collect(),
            syntax: syn::parse_file(source).expect("audit fixture parses"),
        })
        .collect::<Vec<_>>();
    audit_sources(&parsed)
}

fn audit_source(source: &str) -> Vec<String> {
    audit_virtual_sources(&[("", source)])
}

struct CrateNodeToListAudit<'a> {
    modules: &'a BTreeMap<Vec<String>, ModuleSymbols>,
    current_module: Vec<String>,
    label: &'a str,
    calls: Vec<String>,
    lexical_symbols: Vec<ModuleSymbols>,
    value_scopes: Vec<BTreeMap<String, bool>>,
}

impl<'a> CrateNodeToListAudit<'a> {
    fn new(
        modules: &'a BTreeMap<Vec<String>, ModuleSymbols>,
        current_module: Vec<String>,
        label: &'a str,
    ) -> Self {
        Self {
            modules,
            current_module,
            label,
            calls: Vec::new(),
            lexical_symbols: Vec::new(),
            value_scopes: Vec::new(),
        }
    }

    fn record(&mut self, kind: &str) {
        self.calls.push(format!("{}: {kind}", self.label));
    }

    fn path_targets_node_to_list(&self, path: &syn::ExprPath) -> bool {
        if !path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "to_list")
        {
            return false;
        }
        if path
            .qself
            .as_ref()
            .is_some_and(|qself| self.type_is_node(&qself.ty))
        {
            return true;
        }
        let mut owner = path.path.clone();
        owner.segments.pop();
        self.path_is_node_type(&owner)
    }

    fn path_is_node_type(&self, path: &syn::Path) -> bool {
        let segments = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>();
        self.resolve_path(&self.current_module, &segments, &mut BTreeSet::new())
    }

    fn type_is_node(&self, ty: &syn::Type) -> bool {
        self.type_is_node_from(&self.current_module, ty, &mut BTreeSet::new())
    }

    fn type_is_node_from(
        &self,
        module: &[String],
        ty: &syn::Type,
        seen: &mut BTreeSet<String>,
    ) -> bool {
        match ty {
            syn::Type::Path(path) if path.qself.is_none() => {
                let segments = path
                    .path
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string())
                    .collect::<Vec<_>>();
                self.resolve_path(module, &segments, seen)
            }
            syn::Type::Reference(reference) => {
                self.type_is_node_from(module, &reference.elem, seen)
            }
            syn::Type::Paren(paren) => self.type_is_node_from(module, &paren.elem, seen),
            syn::Type::Group(group) => self.type_is_node_from(module, &group.elem, seen),
            _ => false,
        }
    }

    fn resolve_path(
        &self,
        module: &[String],
        segments: &[String],
        seen: &mut BTreeSet<String>,
    ) -> bool {
        const CANONICAL_NODE: [&str; 3] = ["chelis_deep", "node", "Node"];
        if segments.iter().map(String::as_str).eq(CANONICAL_NODE) {
            return true;
        }
        let Some(first) = segments.first().map(String::as_str) else {
            return false;
        };
        match first {
            "crate" => self.resolve_absolute(&segments[1..], seen),
            "self" => self.resolve_in_module(module, &segments[1..], seen),
            "super" => {
                let mut owner = module.to_vec();
                let mut offset = 0;
                while segments
                    .get(offset)
                    .is_some_and(|segment| segment == "super")
                {
                    if owner.pop().is_none() {
                        return false;
                    }
                    offset += 1;
                }
                self.resolve_in_module(&owner, &segments[offset..], seen)
            }
            _ => self.resolve_in_module(module, segments, seen),
        }
    }

    fn resolve_absolute(&self, segments: &[String], seen: &mut BTreeSet<String>) -> bool {
        for split in (0..=segments.len()).rev() {
            let module = segments[..split].to_vec();
            if self.modules.contains_key(&module)
                && self.resolve_in_module(&module, &segments[split..], seen)
            {
                return true;
            }
        }
        false
    }

    fn resolve_in_module(
        &self,
        module: &[String],
        segments: &[String],
        seen: &mut BTreeSet<String>,
    ) -> bool {
        const CANONICAL_NODE: [&str; 3] = ["chelis_deep", "node", "Node"];
        if segments.iter().map(String::as_str).eq(CANONICAL_NODE) {
            return true;
        }
        let Some(first) = segments.first() else {
            return false;
        };
        let key = format!("{}|{}", module.join("::"), segments.join("::"));
        if !seen.insert(key) {
            return false;
        }

        if module == self.current_module {
            for symbols in self.lexical_symbols.iter().rev() {
                if let Some(result) = self.resolve_from_symbols(module, segments, symbols, seen) {
                    return result;
                }
            }
        }
        if let Some(symbols) = self.modules.get(module)
            && let Some(result) = self.resolve_from_symbols(module, segments, symbols, seen)
        {
            return result;
        }
        if segments.len() > 1 {
            let mut child = module.to_vec();
            child.push(first.clone());
            if self.modules.contains_key(&child)
                && self.resolve_in_module(&child, &segments[1..], seen)
            {
                return true;
            }
        }
        false
    }

    fn resolve_from_symbols(
        &self,
        module: &[String],
        segments: &[String],
        symbols: &ModuleSymbols,
        seen: &mut BTreeSet<String>,
    ) -> Option<bool> {
        let first = segments.first()?;
        if let Some(target) = symbols.imports.get(first) {
            let mut expanded = target.clone();
            expanded.extend_from_slice(&segments[1..]);
            return Some(self.resolve_path(module, &expanded, seen));
        }
        if segments.len() == 1
            && let Some(alias) = symbols.aliases.get(first)
        {
            return Some(self.type_is_node_from(module, alias, seen));
        }
        for glob in &symbols.globs {
            let mut expanded = glob.clone();
            expanded.extend_from_slice(segments);
            if self.resolve_path(module, &expanded, seen) {
                return Some(true);
            }
        }
        None
    }

    fn value_is_node(&self, expr: &syn::Expr) -> bool {
        match expr {
            syn::Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
                let name = path.path.segments[0].ident.to_string();
                self.value_scopes
                    .iter()
                    .rev()
                    .find_map(|scope| scope.get(&name))
                    .copied()
                    .unwrap_or(false)
            }
            syn::Expr::Reference(reference) => self.value_is_node(&reference.expr),
            syn::Expr::Paren(paren) => self.value_is_node(&paren.expr),
            syn::Expr::Group(group) => self.value_is_node(&group.expr),
            syn::Expr::Call(call) => match call.func.as_ref() {
                syn::Expr::Path(path) => {
                    let mut owner = path.path.clone();
                    owner.segments.pop();
                    self.path_is_node_type(&owner)
                }
                _ => false,
            },
            syn::Expr::Struct(struct_expr) => self.path_is_node_type(&struct_expr.path),
            _ => false,
        }
    }

    fn bind_value(&mut self, pat: &syn::Pat, is_node: bool) {
        let name = match pat {
            syn::Pat::Ident(ident) => Some(ident.ident.to_string()),
            syn::Pat::Type(typed) => {
                self.bind_value(&typed.pat, is_node);
                None
            }
            syn::Pat::Reference(reference) => {
                self.bind_value(&reference.pat, is_node);
                None
            }
            syn::Pat::Paren(paren) => {
                self.bind_value(&paren.pat, is_node);
                None
            }
            _ => None,
        };
        if let Some(name) = name
            && let Some(scope) = self.value_scopes.last_mut()
        {
            scope.insert(name, is_node);
        }
    }

    fn bind_signature(&mut self, signature: &syn::Signature) {
        for input in &signature.inputs {
            if let syn::FnArg::Typed(typed) = input {
                self.bind_value(&typed.pat, self.type_is_node(&typed.ty));
            }
        }
    }
}

impl<'ast> Visit<'ast> for CrateNodeToListAudit<'_> {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let Some((_, items)) = &item.content else {
            return;
        };
        let parent_lexical_symbols = std::mem::take(&mut self.lexical_symbols);
        self.current_module.push(item.ident.to_string());
        for item in items {
            self.visit_item(item);
        }
        self.current_module.pop();
        self.lexical_symbols = parent_lexical_symbols;
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.value_scopes.push(BTreeMap::new());
        self.bind_signature(&item.sig);
        syn::visit::visit_item_fn(self, item);
        self.value_scopes.pop();
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.value_scopes.push(BTreeMap::new());
        self.bind_signature(&item.sig);
        syn::visit::visit_impl_item_fn(self, item);
        self.value_scopes.pop();
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        self.value_scopes.push(BTreeMap::new());
        self.bind_signature(&item.sig);
        syn::visit::visit_trait_item_fn(self, item);
        self.value_scopes.pop();
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        self.value_scopes.push(BTreeMap::new());
        for input in &closure.inputs {
            if let syn::Pat::Type(typed) = input {
                self.bind_value(&typed.pat, self.type_is_node(&typed.ty));
            }
        }
        syn::visit::visit_expr_closure(self, closure);
        self.value_scopes.pop();
    }

    fn visit_block(&mut self, block: &'ast syn::Block) {
        let mut symbols = ModuleSymbols::default();
        for statement in &block.stmts {
            if let syn::Stmt::Item(item) = statement {
                collect_item_symbols(item, &mut symbols);
            }
        }
        self.lexical_symbols.push(symbols);
        self.value_scopes.push(BTreeMap::new());
        syn::visit::visit_block(self, block);
        self.value_scopes.pop();
        self.lexical_symbols.pop();
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        syn::visit::visit_local(self, local);
        let is_node = match &local.pat {
            syn::Pat::Type(typed) => self.type_is_node(&typed.ty),
            _ => local
                .init
                .as_ref()
                .is_some_and(|init| self.value_is_node(&init.expr)),
        };
        self.bind_value(&local.pat, is_node);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "to_list" && self.value_is_node(&call.receiver) {
            self.record("method call");
        }
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if self.path_targets_node_to_list(path) {
            self.record("path reference");
        }
        syn::visit::visit_expr_path(self, path);
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
        "the Node bridge ratchet must not ban unrelated symbols named to_list: {:?}",
        calls
    );
}

#[test]
fn node_to_list_audit_resolves_module_imports_and_typed_closures() {
    for source in [
        "use chelis_deep::node; fn bridge() { let _ = node::Node::to_list(value, span); }",
        "use chelis_deep::node as deep_node; fn bridge() { let _ = deep_node::Node::to_list(value, span); }",
        "use chelis_deep::node::*; fn bridge() { let _ = Node::to_list(value, span); }",
        "extern crate chelis_deep as deep; fn bridge() { let _ = deep::node::Node::to_list(value, span); }",
        "fn bridge() { use chelis_deep::node::Node as LocalNode; let _ = LocalNode::to_list(value, span); }",
        "fn bridge() { type LocalNode = chelis_deep::node::Node; let _ = LocalNode::to_list(value, span); }",
        "use chelis_deep::node::Node; fn bridge() { let f = |node: &Node| node.to_list(span); }",
    ] {
        let calls = audit_source(source);
        assert!(
            !calls.is_empty(),
            "module or closure spelling escaped the predicate source audit: {source}"
        );
    }
}

#[test]
fn node_to_list_audit_keeps_sibling_module_types_separate() {
    let source = "
        mod chelis_owner {
            use chelis_deep::node::Node as Shared;
            fn bridge(value: Shared) { let _ = value.to_list(span); }
        }
        mod unrelated_owner {
            struct Shared;
            impl Shared { fn to_list(&self, _: u32) {} }
            fn convert(value: Shared) { value.to_list(0); }
        }
    ";
    let calls = audit_source(source);
    assert_eq!(
        calls.len(),
        1,
        "only the Chelis Node alias in its own module is in scope"
    );
}

#[test]
fn node_to_list_audit_resolves_root_aliases_across_source_files() {
    let calls = audit_virtual_sources(&[
        (
            "",
            "type RedteamDeepNode = chelis_deep::node::Node; mod child;",
        ),
        (
            "child",
            "fn bridge() { let _ = crate::RedteamDeepNode::to_list(value, span); }",
        ),
    ]);
    assert!(
        !calls.is_empty(),
        "a crate-root Node alias used by a child source file must be rejected"
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
    let malformed = [
        Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Name("legacy_parameter".to_string()), span),
                    Expr::Map(Metadata::default(), span),
                    Expr::Atom(Atom::Name("extra".to_string()), span),
                ],
            },
            span,
        ),
        Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Tag(DeepTag::Var), span),
                    Expr::Map(Metadata::default(), span),
                    Expr::Atom(Atom::Name("decoded_parameter".to_string()), span),
                    Expr::Atom(Atom::Name("extra".to_string()), span),
                ],
            },
            span,
        ),
    ];

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
fn free_vars_distinguish_legacy_unknown_lists_from_unknown_forms() {
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

    let legacy = predicate(Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Name("future-form".to_string()), span),
                Expr::Map(Metadata::default(), span),
                free_var(),
            ],
        },
        span,
    ));
    assert_eq!(
        predicate_free_vars(&legacy),
        vec!["external".to_string()],
        "legacy List traversal must retain its pre-carrier child walk"
    );

    let successor_unknown = predicate(Expr::UnknownForm(Box::new(UnknownFormData {
        head: "future-form".to_string(),
        meta: Metadata::default(),
        children: vec![free_var()],
        span,
    })));
    assert!(
        predicate_free_vars(&successor_unknown).is_empty(),
        "UnknownForm children were not predicate free-variable scope before this slice"
    );

    let malformed_legacy = predicate(Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(DeepTag::App), span),
                Expr::Atom(Atom::Name("not-metadata".to_string()), span),
                free_var(),
            ],
        },
        span,
    ));
    assert_eq!(
        predicate_free_vars(&malformed_legacy),
        vec!["external".to_string()],
        "malformed legacy Lists retained the old skip-two child traversal"
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
