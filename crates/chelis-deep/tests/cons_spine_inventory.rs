//! Structural guard for named canonical list readers and direct literal recursion.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::Visit;

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
];

#[derive(Default)]
struct BodyScan<'a> {
    function: &'a str,
    cons_literal: bool,
    self_call: bool,
    shared_spine: bool,
}

impl<'ast> Visit<'ast> for BodyScan<'_> {
    fn visit_lit_str(&mut self, lit: &'ast syn::LitStr) {
        self.cons_literal |= lit.value() == "Cons";
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
        self.self_call |= call.method == self.function
            && matches!(call.receiver.as_ref(), syn::Expr::Path(path) if path.path.is_ident("self"));
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.shared_spine |= path.path.segments.iter().any(|s| s.ident == "ConsSpine");
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
    problems: Vec<String>,
    source: String,
}

impl Inventory {
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
    }
}

impl<'ast> Visit<'ast> for Inventory {
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.record(&item.sig.ident.to_string(), &item.block);
        syn::visit::visit_item_fn(self, item);
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.record(&item.sig.ident.to_string(), &item.block);
        syn::visit::visit_impl_item_fn(self, item);
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
