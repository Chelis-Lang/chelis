use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use syn::visit::{self, Visit};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Finding {
    path: String,
    function: String,
    stages: Vec<&'static str>,
}

#[derive(Default)]
struct AliasCollector {
    aliases: BTreeMap<String, &'static str>,
}

impl AliasCollector {
    fn collect_tree(&mut self, tree: &syn::UseTree) {
        match tree {
            syn::UseTree::Rename(rename) => {
                if let Some(stage) = stage_for_call(&rename.ident.to_string()) {
                    self.aliases.insert(rename.rename.to_string(), stage);
                }
            }
            syn::UseTree::Path(path) => self.collect_tree(&path.tree),
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.collect_tree(item);
                }
            }
            syn::UseTree::Name(_) | syn::UseTree::Glob(_) => {}
        }
    }
}

impl<'ast> Visit<'ast> for AliasCollector {
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        self.collect_tree(&item.tree);
    }
}

#[derive(Default)]
struct CallCollector {
    aliases: BTreeMap<String, &'static str>,
    stages: BTreeSet<&'static str>,
}

impl<'ast> Visit<'ast> for CallCollector {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref()
            && let Some(segment) = path.path.segments.last()
        {
            let name = segment.ident.to_string();
            if let Some(stage) = stage_for_call(&name).or_else(|| self.aliases.get(&name).copied())
            {
                self.stages.insert(stage);
            }
        }
        visit::visit_expr_call(self, call);
    }
}

#[derive(Default)]
struct FunctionCollector {
    aliases: BTreeMap<String, &'static str>,
    findings: Vec<(String, Vec<&'static str>)>,
}

impl FunctionCollector {
    fn inspect_block(&mut self, name: String, block: &syn::Block) {
        let mut calls = CallCollector {
            aliases: self.aliases.clone(),
            ..CallCollector::default()
        };
        calls.visit_block(block);
        if calls.stages.len() >= 2 {
            self.findings
                .push((name, calls.stages.into_iter().collect()));
        }
    }
}

impl<'ast> Visit<'ast> for FunctionCollector {
    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        if is_test_only(&function.attrs) {
            return;
        }
        self.inspect_block(function.sig.ident.to_string(), &function.block);
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        if is_test_only(&function.attrs) {
            return;
        }
        self.inspect_block(function.sig.ident.to_string(), &function.block);
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        if !is_test_only(&module.attrs) {
            visit::visit_item_mod(self, module);
        }
    }
}

fn is_test_only(attributes: &[syn::Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("test")
            || (attribute.path().is_ident("cfg")
                && matches!(
                    &attribute.meta,
                    syn::Meta::List(list) if list.tokens.to_string() == "test"
                ))
    })
}

fn stage_for_call(name: &str) -> Option<&'static str> {
    match name {
        "analyze_ir_program"
        | "check_ir_fitness"
        | "check_ir_program"
        | "check_typed_program"
        | "check_ir_with_context"
        | "check_ir_with_signature_context"
        | "build_compiled_library_context"
        | "build_compiled_library_context_with_base" => Some("type"),
        "check_program" | "check_effects_with_context" => Some("effects"),
        "check_linearity" | "check_linearity_with_context" => Some("linearity"),
        "try_lower_program" | "try_lower_program_with_context" | "try_lower_program_to_library" => {
            Some("lower")
        }
        _ => None,
    }
}

fn inspect_source(path: &str, source: &str) -> Result<Vec<Finding>, syn::Error> {
    let file = syn::parse_file(source)?;
    let mut aliases = AliasCollector::default();
    aliases.visit_file(&file);
    let mut collector = FunctionCollector {
        aliases: aliases.aliases,
        findings: Vec::new(),
    };
    collector.visit_file(&file);
    Ok(collector
        .findings
        .into_iter()
        .map(|(function, stages)| Finding {
            path: path.to_string(),
            function,
            stages,
        })
        .collect())
}

fn guarded_sources(workspace: &Path) -> Vec<PathBuf> {
    let roots = [
        workspace.join("crates/chelis-compiler-api/src"),
        workspace.join("crates/chelis-cli/src"),
        workspace.join("crates/chelis-e2e/src"),
    ];
    let mut files = Vec::new();
    for root in roots {
        collect_rust_files(&root, &mut files);
    }
    files.sort();
    files
}

fn collect_rust_files(directory: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "tests") {
                continue;
            }
            collect_rust_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs")
            && !path
                .file_name()
                .is_some_and(|name| name == "tests.rs" || name == "source_arch.rs")
            && !path.ends_with(Path::new("chelis-compiler-api/src/pipeline.rs"))
        {
            files.push(path);
        }
    }
}

fn actual_workspace_findings(workspace: &Path) -> Vec<Finding> {
    guarded_sources(workspace)
        .into_iter()
        .flat_map(|path| {
            let relative = path.strip_prefix(workspace).unwrap_or(&path);
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            inspect_source(&relative.display().to_string(), &source)
                .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
        })
        .collect()
}

#[test]
fn focused_single_stage_helpers_are_allowed() {
    let source = r#"
        fn annotate(exprs: &[Expr]) { chelis_types::check_ir_program(exprs); }
        fn lower(checked: &CheckedProgram) { chelis_ir::lower::try_lower_program(checked); }
    "#;
    assert!(inspect_source("positive.rs", source).unwrap().is_empty());
}

#[test]
fn production_only_cfg_is_not_excluded() {
    let source = r#"
        #[cfg(not(test))]
        fn duplicate(exprs: &[Expr]) {
            chelis_types::analyze_ir_program(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    let findings = inspect_source("production.rs", source).unwrap();
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].stages, vec!["effects", "type"]);
}

#[test]
fn planted_duplicate_pipeline_is_rejected_with_path_and_stages() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            let checked = chelis_types::check_ir_program(exprs).unwrap();
            let checked = chelis_effects::check_program(&checked).unwrap();
            chelis_types::check_linearity(&checked).unwrap();
        }
    "#;
    let findings = inspect_source("planted.rs", source).unwrap();
    assert_eq!(
        findings,
        vec![Finding {
            path: "planted.rs".to_string(),
            function: "duplicate".to_string(),
            stages: vec!["effects", "linearity", "type"],
        }]
    );
}

#[test]
fn imported_stage_aliases_cannot_bypass_the_guard() {
    let source = r#"
        use chelis_types::check_ir_program as type_check;
        use chelis_effects::check_program as effect_check;
        fn duplicate(exprs: &[Expr]) {
            let checked = type_check(exprs).unwrap();
            effect_check(&checked).unwrap();
        }
    "#;
    let findings = inspect_source("aliases.rs", source).unwrap();
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].stages, vec!["effects", "type"]);
}

#[test]
fn owner_exclusion_does_not_hide_an_e2e_pipeline_file() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let owner = workspace
        .path()
        .join("crates/chelis-compiler-api/src/pipeline.rs");
    let e2e = workspace.path().join("crates/chelis-e2e/src/pipeline.rs");
    for path in [&owner, &e2e] {
        fs::create_dir_all(path.parent().expect("fixture parent")).expect("fixture directory");
        fs::write(
            path,
            "fn duplicate(x: &[Expr]) { check_ir_program(x); check_program(x); }",
        )
        .expect("fixture source");
    }

    let findings = actual_workspace_findings(workspace.path());
    assert_eq!(findings.len(), 1, "only the canonical owner is excluded");
    assert_eq!(findings[0].path, "crates/chelis-e2e/src/pipeline.rs");
}

#[test]
fn production_consumers_do_not_recreate_the_semantic_pipeline() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("compiler API crate must live under the workspace crates directory");
    let findings = actual_workspace_findings(workspace);
    assert!(
        findings.is_empty(),
        "production semantic pipeline duplicates remain: {findings:#?}"
    );
}
