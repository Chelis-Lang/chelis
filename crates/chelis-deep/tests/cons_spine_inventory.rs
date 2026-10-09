//! Structural guard for canonical list readers and recognition entry points.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use syn::parse::Parser;
use syn::visit::Visit;

// The reviewed boundary contains constructors, builtin dispatch, and the two
// ConsSpineNode adapters. A new recognition site must be reviewed rather than
// being hidden in a helper that a recursive reader calls by any Rust spelling.
// Counts also catch a second recognizer added to an existing function.
const REVIEWED_CONS_LITERALS: &[(&str, &str, usize)] = &[
    ("crates/chelis-cli/src/prove/mod.rs", "deep_cons_list", 1),
    (
        "crates/chelis-compiler-api/src/runtime/eval.rs",
        "eval_app_under_result_claim",
        2,
    ),
    (
        "crates/chelis-compiler-api/src/runtime/host_ops.rs",
        "pattern_matches_with_result_producer",
        1,
    ),
    (
        "crates/chelis-compiler-api/src/runtime/transforms.rs",
        "make_list_construction_expr",
        1,
    ),
    ("crates/chelis-deep/src/cons_spine.rs", "cons_parts", 1),
    (
        "crates/chelis-deep/src/cons_spine.rs",
        "cons_parts_with_terminal_name",
        1,
    ),
    ("crates/chelis-ir/src/host.rs", "plan_host_pattern", 1),
    ("crates/chelis-ir/src/host.rs", "lower_app_host_expr", 1),
    (
        "crates/chelis-ir/src/lower.rs",
        "expr_requires_host_runtime_with_ctx",
        1,
    ),
    ("crates/chelis-ir/src/lower.rs", "cons_two_int_pair", 2),
    ("crates/chelis-ir/src/lower.rs", "cons_parts", 1),
    ("crates/chelis-ir/src/lower.rs", "rebuild_cons_chain", 1),
    ("crates/chelis-ir/src/lower.rs", "lower_app", 2),
    ("crates/chelis-prove/src/opaque.rs", "deep_cons_list", 1),
    (
        "crates/chelis-prove/src/property_runner.rs",
        "deep_cons_list",
        1,
    ),
    (
        "crates/chelis-surf/src/desugar.rs",
        "desugar_list_literal",
        1,
    ),
    (
        "crates/chelis-surf/src/parser.rs",
        "qualified_constructor_application_parses",
        1,
    ),
    (
        "crates/chelis-surf/src/resugar.rs",
        "resugar_declared_tensor_value",
        1,
    ),
    (
        "crates/chelis-types/src/builtins.rs",
        "register_prelude_adts",
        2,
    ),
    ("crates/chelis-types/src/infer/app.rs", "infer_app_inner", 1),
    (
        "crates/chelis-types/src/infer/common.rs",
        "is_bracket_literal",
        1,
    ),
    (
        "crates/chelis-types/src/infer/validate.rs",
        "validate_static_cons_spine",
        1,
    ),
    (
        "crates/chelis-types/src/infer/validate.rs",
        "validate_ir_expr",
        2,
    ),
];

const REVIEWED_CELL_ACCESS: &[(&str, &str, usize)] = &[
    (
        "crates/chelis-deep/src/cons_spine.rs",
        "cons_parts_with_terminal_name",
        1,
    ),
    ("crates/chelis-deep/src/cons_spine.rs", "next", 2),
];

const SHARED_READERS: &[&str] = &[
    "eval_cons_spine",
    "is_bracket_literal",
    "collect_cons_chain_for_shape",
    "untyped_chain_items",
    "resugar_finite_list",
    "cons_chain_int_pairs",
    "cons_chain_two_ints",
    "collect_shape_list_elements",
    "cons_chain_pair_list",
    "collect_cons_chain",
    "adt_cons_chain_values",
    "static_list_spine_items",
    "untyped_to_tensor_element",
    "lower_list_literal_items",
];

// Generic whole-expression evaluators recurse on every expression kind. They
// recognize Cons as one application, but do not walk a Cons tail specially.
// Pattern walkers likewise recurse over user-authored pattern trees rather
// than the compiler-authored expression/value spine guarded here.
const GENERIC_RECURSION: &[&str] = &[
    "validate_ir_expr",
    "pattern_matches_with_result_producer",
    "plan_host_pattern",
    // This whole-expression host-eligibility walk stops as soon as it sees
    // a Cons call; its recursive calls are for other expression shapes.
    "expr_requires_host_runtime_with_ctx",
];

#[derive(Default)]
struct BodyScan<'a> {
    function: &'a str,
    cons_literal: bool,
    cons_literal_count: usize,
    direct_cons_cell_access_count: usize,
    self_call: bool,
    shared_spine: bool,
}

fn static_string_expr(expr: &syn::Expr) -> Option<String> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(lit),
            ..
        }) => Some(lit.value()),
        syn::Expr::Group(group) => static_string_expr(&group.expr),
        syn::Expr::Paren(paren) => static_string_expr(&paren.expr),
        syn::Expr::Macro(expr) => static_macro_string(&expr.mac),
        _ => None,
    }
}

fn static_macro_string(mac: &syn::Macro) -> Option<String> {
    static_macro_string_named(&mac.path.segments.last()?.ident.to_string(), &mac.tokens)
}

fn static_macro_string_named(name: &str, tokens: &proc_macro2::TokenStream) -> Option<String> {
    match name {
        "stringify" => Some(tokens.to_string()),
        "concat" => {
            let parts = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
                .parse2(tokens.clone())
                .ok()?;
            parts.iter().map(static_string_expr).collect()
        }
        _ => None,
    }
}

fn record_cons_macro(mac: &syn::Macro, scan: &mut BodyScan<'_>) {
    if static_macro_string(mac).as_deref() == Some("Cons") {
        scan.cons_literal = true;
        scan.cons_literal_count += 1;
    }
}

fn scan_macro_tokens(tokens: proc_macro2::TokenStream, scan: &mut BodyScan<'_>) {
    let trees: Vec<_> = tokens.into_iter().collect();
    for (index, token) in trees.iter().enumerate() {
        if let (
            proc_macro2::TokenTree::Ident(name),
            Some(proc_macro2::TokenTree::Punct(bang)),
            Some(proc_macro2::TokenTree::Group(group)),
        ) = (token, trees.get(index + 1), trees.get(index + 2))
            && bang.as_char() == '!'
            && static_macro_string_named(&name.to_string(), &group.stream()).as_deref()
                == Some("Cons")
        {
            scan.cons_literal = true;
            scan.cons_literal_count += 1;
        }
        match token {
            proc_macro2::TokenTree::Group(group) => scan_macro_tokens(group.stream(), scan),
            proc_macro2::TokenTree::Literal(literal) => {
                if let Ok(text) = syn::parse_str::<syn::LitStr>(&literal.to_string())
                    && text.value() == "Cons"
                {
                    scan.cons_literal = true;
                    scan.cons_literal_count += 1;
                }
            }
            proc_macro2::TokenTree::Ident(ident)
                if ident == "cons_parts" || ident == "cons_parts_with_terminal_name" =>
            {
                scan.direct_cons_cell_access_count += 1;
            }
            _ => {}
        }
    }
}

fn is_cons_cell_method(name: &syn::Ident) -> bool {
    name == "cons_parts" || name == "cons_parts_with_terminal_name"
}

fn use_mentions_cons_cell_method(tree: &syn::UseTree) -> bool {
    match tree {
        syn::UseTree::Path(path) => {
            is_cons_cell_method(&path.ident) || use_mentions_cons_cell_method(&path.tree)
        }
        syn::UseTree::Name(name) => is_cons_cell_method(&name.ident),
        syn::UseTree::Rename(rename) => is_cons_cell_method(&rename.ident),
        syn::UseTree::Group(group) => group.items.iter().any(use_mentions_cons_cell_method),
        syn::UseTree::Glob(_) => false,
    }
}

impl<'ast> Visit<'ast> for BodyScan<'_> {
    fn visit_lit_str(&mut self, lit: &'ast syn::LitStr) {
        if lit.value() == "Cons" {
            self.cons_literal = true;
            self.cons_literal_count += 1;
        }
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        record_cons_macro(mac, self);
        scan_macro_tokens(mac.tokens.clone(), self);
    }

    fn visit_item_fn(&mut self, _item: &'ast syn::ItemFn) {
        // The inventory visits local functions separately with their own name.
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        self.direct_cons_cell_access_count +=
            usize::from(use_mentions_cons_cell_method(&item.tree));
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref() {
            self.self_call |= path
                .path
                .get_ident()
                .is_some_and(|callee| callee == self.function);
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.direct_cons_cell_access_count += usize::from(is_cons_cell_method(&call.method));
        self.self_call |= call.method == self.function
            && matches!(call.receiver.as_ref(), syn::Expr::Path(path) if path.path.is_ident("self"));
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.shared_spine |= path.path.segments.iter().any(|s| s.ident == "ConsSpine");
        self.direct_cons_cell_access_count += usize::from(
            path.path
                .segments
                .last()
                .is_some_and(|segment| is_cons_cell_method(&segment.ident)),
        );
        syn::visit::visit_expr_path(self, path);
    }
}

fn inspect<'a>(name: &'a str, body: &syn::Block) -> BodyScan<'a> {
    let mut scan = BodyScan {
        function: name,
        ..BodyScan::default()
    };
    scan.visit_block(body);
    scan
}

#[derive(Default)]
struct Inventory {
    seen_shared: BTreeSet<String>,
    seen_cons_literals: BTreeMap<(String, String), usize>,
    seen_cell_access: BTreeMap<(String, String), usize>,
    problems: Vec<String>,
    source: String,
    function_depth: usize,
    module_expr_depth: usize,
}

impl Inventory {
    fn record_module_scan(&mut self, scan: &BodyScan<'_>) {
        if scan.cons_literal_count != 0 {
            self.problems.push(format!(
                "{}:<module> has unreviewed Cons recognition",
                self.source
            ));
        }
        if scan.direct_cons_cell_access_count != 0 {
            self.problems.push(format!(
                "{}:<module> has direct Cons cell access outside the shared iterator",
                self.source
            ));
        }
    }

    fn record(&mut self, name: &str, body: &syn::Block) {
        let scan = inspect(name, body);
        if SHARED_READERS.contains(&name) {
            self.seen_shared.insert(name.to_string());
            if !scan.shared_spine {
                self.problems
                    .push(format!("{}:{name} must use ConsSpine", self.source));
            }
        }
        if scan.cons_literal && scan.self_call && !GENERIC_RECURSION.contains(&name) {
            self.problems
                .push(format!("{}:{name} recursively walks Cons", self.source));
        }
        if scan.cons_literal_count != 0 {
            let key = (self.source.clone(), name.to_string());
            *self.seen_cons_literals.entry(key.clone()).or_default() += scan.cons_literal_count;
            let reviewed = REVIEWED_CONS_LITERALS
                .iter()
                .find(|(source, function, _)| *source == key.0 && *function == key.1);
            if reviewed.is_none() {
                self.problems.push(format!(
                    "{}:{name} has unreviewed Cons recognition",
                    self.source
                ));
            }
        }
        if scan.direct_cons_cell_access_count != 0 {
            let key = (self.source.clone(), name.to_string());
            *self.seen_cell_access.entry(key.clone()).or_default() +=
                scan.direct_cons_cell_access_count;
            if !REVIEWED_CELL_ACCESS
                .iter()
                .any(|(source, function, _)| *source == key.0 && *function == key.1)
            {
                self.problems.push(format!(
                    "{}:{name} has direct Cons cell access outside the shared iterator",
                    self.source
                ));
            }
        }
    }
}

impl<'ast> Visit<'ast> for Inventory {
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if self.function_depth == 0 && self.module_expr_depth == 0 {
            let mut scan = BodyScan::default();
            record_cons_macro(mac, &mut scan);
            scan_macro_tokens(mac.tokens.clone(), &mut scan);
            self.record_module_scan(&scan);
        }
        syn::visit::visit_macro(self, mac);
    }

    fn visit_expr(&mut self, expr: &'ast syn::Expr) {
        if self.function_depth == 0 && self.module_expr_depth == 0 {
            let mut scan = BodyScan::default();
            scan.visit_expr(expr);
            self.record_module_scan(&scan);
        }
        self.module_expr_depth += 1;
        syn::visit::visit_expr(self, expr);
        self.module_expr_depth -= 1;
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if use_mentions_cons_cell_method(&item.tree) {
            self.problems.push(format!(
                "{}:use has direct Cons cell access outside the shared iterator",
                self.source
            ));
        }
        syn::visit::visit_item_use(self, item);
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.record(&item.sig.ident.to_string(), &item.block);
        self.function_depth += 1;
        syn::visit::visit_item_fn(self, item);
        self.function_depth -= 1;
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.record(&item.sig.ident.to_string(), &item.block);
        self.function_depth += 1;
        syn::visit::visit_impl_item_fn(self, item);
        self.function_depth -= 1;
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        if let Some(body) = &item.default {
            self.record(&item.sig.ident.to_string(), body);
        }
        self.function_depth += 1;
        syn::visit::visit_trait_item_fn(self, item);
        self.function_depth -= 1;
    }
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read source directory") {
        let path = entry.expect("source entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn canonical_cons_readers_share_the_iterator_and_no_direct_literal_reader_appears() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let mut inventory = Inventory::default();
    for crate_dir in fs::read_dir(root.join("crates")).expect("workspace crates") {
        let src = crate_dir.expect("crate entry").path().join("src");
        if !src.is_dir() {
            continue;
        }
        let mut files = Vec::new();
        rust_sources(&src, &mut files);
        for file in files {
            inventory.source = file.strip_prefix(root).unwrap().display().to_string();
            let contents = fs::read_to_string(&file).expect("read Rust source");
            let parsed = syn::parse_file(&contents).expect("parse Rust source");
            inventory.visit_file(&parsed);
        }
    }
    for reader in SHARED_READERS {
        if !inventory.seen_shared.contains(*reader) {
            inventory
                .problems
                .push(format!("missing shared reader: {reader}"));
        }
    }
    let reviewed: BTreeMap<_, _> = REVIEWED_CONS_LITERALS
        .iter()
        .map(|(source, function, count)| ((source.to_string(), function.to_string()), *count))
        .collect();
    assert_eq!(
        inventory.seen_cons_literals, reviewed,
        "canonical Cons recognition entry points drifted"
    );
    let reviewed_cell_access: BTreeMap<_, _> = REVIEWED_CELL_ACCESS
        .iter()
        .map(|(source, function, count)| ((source.to_string(), function.to_string()), *count))
        .collect();
    assert_eq!(
        inventory.seen_cell_access, reviewed_cell_access,
        "direct Cons cell adapter access drifted"
    );
    assert!(
        inventory.problems.is_empty(),
        "{}",
        inventory.problems.join("\n")
    );
}

#[test]
fn inventory_detects_a_new_recursive_cons_reader() {
    let parsed = syn::parse_file("fn new_reader(x: i32) { let _ = \"Cons\"; new_reader(x); }")
        .expect("valid Rust probe");
    let syn::Item::Fn(function) = &parsed.items[0] else {
        unreachable!()
    };
    let scan = inspect("new_reader", &function.block);
    assert!(scan.cons_literal && scan.self_call && !scan.shared_spine);
}

#[test]
fn inventory_does_not_call_another_types_same_named_method_self_recursion() {
    let parsed =
        syn::parse_file("fn new_reader(x: i32) { let _ = \"Cons\"; Other::new_reader(x); }")
            .expect("valid Rust probe");
    let syn::Item::Fn(function) = &parsed.items[0] else {
        unreachable!()
    };
    let scan = inspect("new_reader", &function.block);
    assert!(scan.cons_literal && !scan.self_call);
}

#[test]
fn inventory_rejects_cons_recognition_split_from_recursive_readers() {
    for source in [
        "fn is_cons(name: &str) -> bool { name == \"Cons\" } fn read(name: &str) { if is_cons(name) { read(name); } }",
        "mod helper { pub fn is_cons(name: &str) -> bool { name == \"Cons\" } } fn read(name: &str) { if helper::is_cons(name) { read(name); } }",
        "mod helper { pub fn is_cons(name: &str) -> bool { name == \"Cons\" } } use helper::is_cons as check; fn read(name: &str) { if check(name) { read(name); } }",
        "fn read(name: &str) { fn is_cons(name: &str) -> bool { name == \"Cons\" } if is_cons(name) { read(name); } }",
        "fn is_cons(name: &str) -> bool { matches!(name, \"Cons\") } fn read(name: &str) { if is_cons(name) { read(name); } }",
        "const CONS: &str = \"Cons\"; fn is_cons(name: &str) -> bool { name == CONS } fn read(name: &str) { if is_cons(name) { read(name); } }",
        "macro_rules! is_cons { ($name:expr) => { $name == \"Cons\" }; } fn read(name: &str) { if is_cons!(name) { read(name); } }",
        "fn is_cons(name: &str) -> bool { name == stringify!(Cons) } fn read(name: &str) { if is_cons(name) { read(name); } }",
        "fn is_cons(name: &str) -> bool { name == concat!(\"Con\", \"s\") } fn read(name: &str) { if is_cons(name) { read(name); } }",
        "fn is_cons(name: &str) -> bool { name == concat!(\"Con\", stringify!(s)) } fn read(name: &str) { if is_cons(name) { read(name); } }",
        "const CONS: &str = concat!(\"Con\", \"s\"); fn read(name: &str) { if name == CONS { read(name); } }",
        "macro_rules! is_cons { ($name:expr) => { $name == stringify!(Cons) }; } fn read(name: &str) { if is_cons!(name) { read(name); } }",
    ] {
        let parsed = syn::parse_file(source).expect("valid helper-mediated reader probe");
        let mut inventory = Inventory {
            source: "crates/chelis-deep/src/unreviewed_reader.rs".to_string(),
            ..Inventory::default()
        };
        inventory.visit_file(&parsed);
        assert!(
            inventory
                .problems
                .iter()
                .any(|problem| problem.contains("unreviewed Cons recognition")),
            "unreviewed helper-mediated reader escaped the inventory: {source}"
        );
    }
}

#[test]
fn inventory_rejects_direct_cons_cell_access_outside_the_shared_iterator() {
    for source in [
        "fn read<N: ConsSpineNode>(node: &N) { if let Some((_, tail)) = node.cons_parts() { read(tail); } }",
        "fn read<N: ConsSpineNode>(node: &N) { if matches!(node.cons_parts(), Some(_)) { read(node); } }",
        "fn read<N: ConsSpineNode>(node: &N) { let parts = N::cons_parts; if let Some((_, tail)) = parts(node) { read(tail); } }",
        "use ConsSpineNode::cons_parts as parts; fn read<N: ConsSpineNode>(node: &N) { if let Some((_, tail)) = parts(node) { read(tail); } }",
    ] {
        let parsed = syn::parse_file(source).expect("valid adapter-bypass probe");
        let mut inventory = Inventory {
            source: "crates/chelis-deep/src/unreviewed_reader.rs".to_string(),
            ..Inventory::default()
        };
        inventory.visit_file(&parsed);
        assert!(
            inventory
                .problems
                .iter()
                .any(|problem| problem.contains("direct Cons cell access")),
            "direct adapter access escaped the inventory: {source}"
        );
    }
}

#[test]
fn inventory_rejects_new_cell_access_in_the_iterator_source() {
    for source in [
        "fn added_reader<N: ConsSpineNode>(node: &N) { if let Some((_, tail)) = node.cons_parts() { added_reader(tail); } }",
        "const WALK: fn(&Node) -> Option<&Node> = |node| node.cons_parts();",
    ] {
        let parsed = syn::parse_file(source).expect("valid same-file adapter-bypass probe");
        let mut inventory = Inventory {
            source: "crates/chelis-deep/src/cons_spine.rs".to_string(),
            ..Inventory::default()
        };
        inventory.visit_file(&parsed);
        assert!(
            inventory
                .problems
                .iter()
                .any(|problem| problem.contains("direct Cons cell access")),
            "same-file adapter access escaped the inventory: {source}"
        );
    }
}
