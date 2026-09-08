//! The typed AST API owns annotation interpretation; consumers must not add
//! private first/last-key policies over reconstructed generic metadata.
use std::{fs, path::Path};
use syn::visit::{self, Visit};

#[derive(Default)]
struct RawKeyReaders {
    findings: Vec<String>,
}
impl<'ast> Visit<'ast> for RawKeyReaders {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        if let syn::Item::Mod(module) = item
            && module
                .attrs
                .iter()
                .any(|a| a.path().is_ident("cfg") && quote::quote!(#a).to_string().contains("test"))
        {
            return;
        }
        if let syn::Item::Fn(function) = item
            && function.attrs.iter().any(|a| a.path().is_ident("test"))
        {
            return;
        }
        visit::visit_item(self, item);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if matches!(
            call.method.to_string().as_str(),
            "find" | "find_map" | "filter" | "any" | "all" | "position" | "rposition"
        ) {
            for arg in &call.args {
                if let syn::Expr::Closure(closure) = arg {
                    let mut comparisons = KeyComparisons::default();
                    comparisons.visit_expr(&closure.body);
                    self.findings.extend(comparisons.keys);
                }
            }
        }
        visit::visit_expr_method_call(self, call);
    }
}
#[derive(Default)]
struct KeyComparisons {
    keys: Vec<String>,
}
impl<'ast> Visit<'ast> for KeyComparisons {
    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        if matches!(expr.op, syn::BinOp::Eq(_) | syn::BinOp::Ne(_)) {
            for side in [&*expr.left, &*expr.right] {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(value),
                    ..
                }) = side
                    && chelis_deep::metadata::REGISTERED_METADATA_KEYS
                        .contains(&value.value().as_str())
                {
                    self.keys.push(value.value());
                }
            }
        }
        visit::visit_expr_binary(self, expr);
    }
}
fn readers(source: &str) -> Vec<String> {
    let mut scan = RawKeyReaders::default();
    scan.visit_file(&syn::parse_file(source).expect("Rust source parses"));
    scan.findings
}
fn scan_tree(path: &Path, failures: &mut Vec<String>) {
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            scan_tree(&path, failures);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let keys = readers(&fs::read_to_string(&path).unwrap());
            if !keys.is_empty() {
                failures.push(format!("{}: {keys:?}", path.display()));
            }
        }
    }
}
#[test]
fn compiler_consumers_do_not_resolve_registered_annotation_keys_as_strings() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut failures = Vec::new();
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.file_name().is_some_and(|name| name == "chelis-deep") {
            continue;
        }
        let source = path.join("src");
        if source.is_dir() {
            scan_tree(&source, &mut failures);
        }
    }
    assert!(
        failures.is_empty(),
        "Use typed metadata getters; raw iterator key policies found:\n{}",
        failures.join("\n")
    );
}
#[test]
fn raw_reader_census_has_mutation_and_overapplication_controls() {
    for method in ["find", "filter", "any", "rposition"] {
        let source = format!(
            "fn reader(m: &Map) {{ m.entries.iter().{method}(|(key, _)| key == \"type\"); }}"
        );
        assert_eq!(readers(&source), ["type"]);
    }
    assert!(
        readers("fn reader(m: &Metadata) { m.ty(); m.extensions().get(\"tool_note\"); }")
            .is_empty()
    );
    assert!(
        readers("fn reader(m: &Json) { m.get(\"type\"); }").is_empty(),
        "a JSON field is not an AST annotation reader"
    );
}
