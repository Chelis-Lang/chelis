//! The chelis#730 Phase 3 reviewed `reject_*` source inventory.
//!
//! The inventory is intentionally syntactic and exact. Every `reject_*`
//! function anywhere in either public build crate's Rust source tree,
//! including nested modules, nested free functions, and `impl` methods, must
//! update this reviewed manifest. A name in both crates is always a failure
//! because it creates two editable policies. This test does not claim to recognize a
//! semantic reimplementation hidden under an unrelated name; that remains an
//! architectural-review concern rather than a property an AST name census can
//! prove.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use syn::Visibility;
use syn::visit::{self, Visit};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("compiler-api lives under <workspace>/crates")
        .to_path_buf()
}

#[derive(Default)]
struct RejectFunctionVisitor {
    functions: BTreeSet<(String, bool)>,
}

impl<'ast> Visit<'ast> for RejectFunctionVisitor {
    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        let name = function.sig.ident.to_string();
        if name.starts_with("reject_") {
            self.functions
                .insert((name, matches!(function.vis, Visibility::Public(_))));
        }
        visit::visit_item_fn(self, function);
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        let name = function.sig.ident.to_string();
        if name.starts_with("reject_") {
            self.functions
                .insert((name, matches!(function.vis, Visibility::Public(_))));
        }
        visit::visit_impl_item_fn(self, function);
    }
}

fn reject_functions(path: &Path) -> BTreeSet<(String, bool)> {
    let source =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let file = syn::parse_file(&source)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
    let mut visitor = RejectFunctionVisitor::default();
    visitor.visit_file(&file);
    visitor.functions
}

fn rust_sources_under(root: &Path) -> Vec<PathBuf> {
    fn visit_directory(directory: &Path, sources: &mut Vec<PathBuf>) {
        let mut entries = fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("read directory {}: {error}", directory.display()))
            .map(|entry| entry.expect("read directory entry").path())
            .collect::<Vec<_>>();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                visit_directory(&path, sources);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                sources.push(path);
            }
        }
    }

    let mut sources = Vec::new();
    visit_directory(root, &mut sources);
    sources
}

fn reject_functions_under(root: &Path) -> BTreeSet<(String, String, bool)> {
    rust_sources_under(root)
        .into_iter()
        .flat_map(|path| {
            let relative = path
                .strip_prefix(root)
                .expect("source must be below root")
                .to_string_lossy()
                .replace('\\', "/");
            reject_functions(&path)
                .into_iter()
                .map(move |(name, public)| (relative.clone(), name, public))
        })
        .collect()
}

#[test]
fn reject_inventory_discovers_nested_function_definitions() {
    let directory = tempfile::tempdir().expect("tempdir");
    let source = directory.path().join("nested.rs");
    fs::write(
        &source,
        r#"
mod nested {
    pub fn reject_shadow_policy() -> Result<(), ()> {
        Ok(())
    }
}
"#,
    )
    .expect("write nested source");

    assert_eq!(
        reject_functions(&source),
        BTreeSet::from([("reject_shadow_policy".to_string(), true)]),
        "nested gate definitions are part of the recurrence surface"
    );
}

#[test]
fn reject_inventory_discovers_impl_method_definitions() {
    let directory = tempfile::tempdir().expect("tempdir");
    let source = directory.path().join("impl_method.rs");
    fs::write(
        &source,
        r#"
struct ReviewProbeGates;

impl ReviewProbeGates {
    pub fn reject_shadow_policy() -> Result<(), ()> {
        Ok(())
    }
}
"#,
    )
    .expect("write impl-method source");

    assert_eq!(
        reject_functions(&source),
        BTreeSet::from([("reject_shadow_policy".to_string(), true)]),
        "moving a gate into an impl block must not escape the inventory"
    );
}

#[test]
fn reject_inventory_discovers_every_rust_file_under_a_crate_source_tree() {
    let directory = tempfile::tempdir().expect("tempdir");
    let nested = directory.path().join("nested");
    fs::create_dir(&nested).expect("create nested directory");
    fs::write(
        nested.join("policy.rs"),
        "pub(crate) fn reject_moved_policy() -> Result<(), ()> { Ok(()) }\n",
    )
    .expect("write nested source file");

    assert_eq!(
        reject_functions_under(directory.path()),
        BTreeSet::from([(
            "nested/policy.rs".to_string(),
            "reject_moved_policy".to_string(),
            false,
        )]),
        "moving a gate out of the historical main files must not escape the inventory"
    );
}

#[test]
fn phase3_reject_function_inventory_matches_the_reviewed_manifest() {
    let root = workspace_root();
    let cli = reject_functions_under(&root.join("crates/chelis-cli/src"));
    let compiler = reject_functions_under(&root.join("crates/chelis-compiler-api/src"));

    let expected_cli = BTreeSet::new();
    let expected_compiler = BTreeSet::from([
        (
            "compiler.rs".to_string(),
            "reject_eval_only_builtins".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_host_only_builtins".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_host_only_builtins_before_host_lowering".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_symbolic_windowed_reduce".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_unsized_named_dims".to_string(),
            false,
        ),
        (
            "compiler.rs".to_string(),
            "reject_unsupported_hip_ops".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_unsupported_hip_ops_in_host_program".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_unsupported_effect_ops".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_unsupported_effect_ops_in_host_execution_plan".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_unsupported_effect_ops_in_host_program".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_unsupported_metal_ops".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_unsupported_metal_ops_in_host_program".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_unsupported_reduce_window_precision".to_string(),
            true,
        ),
        (
            "compiler.rs".to_string(),
            "reject_unsupported_windowed_reductions_in_host_program".to_string(),
            true,
        ),
    ]);

    assert_eq!(cli, expected_cli, "review the CLI gate inventory change");
    assert_eq!(
        compiler, expected_compiler,
        "review the shared compiler gate inventory change"
    );

    let cli_names = cli.iter().map(|(_, name, _)| name).collect::<BTreeSet<_>>();
    let compiler_names = compiler
        .iter()
        .map(|(_, name, _)| name)
        .collect::<BTreeSet<_>>();
    let duplicates = cli_names
        .intersection(&compiler_names)
        .copied()
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        duplicates.is_empty(),
        "a gate decision exists in both public build paths: {duplicates:?}"
    );
}
