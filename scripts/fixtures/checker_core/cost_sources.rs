use quote::ToTokens;
use syn::visit_mut::{self, VisitMut};

struct DataOwner;
impl VisitMut for DataOwner {
    fn visit_path_mut(&mut self, path: &mut syn::Path) {
        if path.segments.len() > 1 && path.segments[0].ident == "chelis_checker_core"
            && matches!(path.segments[1].ident.to_string().as_str(), "env" | "errors") {
            path.segments[0].ident = syn::Ident::new("chelis_types", path.segments[0].ident.span());
        }
        visit_mut::visit_path_mut(self, path);
    }
    fn visit_use_tree_mut(&mut self, tree: &mut syn::UseTree) {
        if let syn::UseTree::Path(owner) = tree {
            if owner.ident == "chelis_checker_core" {
                if let syn::UseTree::Path(module) = owner.tree.as_ref() {
                    if module.ident == "env" || module.ident == "errors" {
                        owner.ident = syn::Ident::new("chelis_types", owner.ident.span());
                    }
                }
            }
        }
        visit_mut::visit_use_tree_mut(self, tree);
    }
}

fn syntax(source: &str) -> String {
    let mut file = syn::parse_file(source).unwrap();
    DataOwner.visit_file_mut(&mut file);
    let mut imports = Vec::new();
    let mut body = Vec::new();
    for item in file.items {
        if matches!(item, syn::Item::Use(_)) {
            imports.push(item.into_token_stream().to_string());
        } else {
            body.push(item.into_token_stream().to_string());
        }
    }
    imports.sort();
    format!("{:?}\n{:?}", imports, body)
}

struct ChangeMeasurement(bool);
impl VisitMut for ChangeMeasurement {
    fn visit_lit_int_mut(&mut self, literal: &mut syn::LitInt) {
        if !self.0 {
            let value = literal.base10_parse::<u64>().unwrap();
            *literal = syn::LitInt::new(&format!("{}{}", value + 1, literal.suffix()), literal.span());
            self.0 = true;
        }
    }
}

fn main() {
    let records: serde_json::Value = serde_json::from_str(SOURCES_JSON).unwrap();
    assert!(!records.as_array().unwrap().is_empty());
    for record in records.as_array().unwrap() {
        assert_eq!(syntax(record["baseline"].as_str().unwrap()),
            syntax(record["candidate"].as_str().unwrap()),
            "The full-host cost fixture changes beyond its canonical data imports: {}", record["path"]);
        let baseline = record["baseline"].as_str().unwrap();
        let mut changed = syn::parse_file(baseline).unwrap();
        let mut mutation = ChangeMeasurement(false);
        mutation.visit_file_mut(&mut changed);
        assert!(mutation.0, "The measurement control must change a real integer.");
        assert_ne!(syntax(baseline), syntax(&changed.into_token_stream().to_string()),
            "The syntax detector must reject a measurement mutation.");
    }
    println!("CHECKER_COST_SOURCE_CONTRACTS accepted");
}
