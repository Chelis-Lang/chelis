//! The chelis#730 Phase 3 no-duplicate-gates tripwire.
//!
//! The inventory is intentionally exact. A new top-level `reject_*` decision
//! in either public build path must update this reviewed manifest; a name in
//! both files is always a failure because it creates two editable policies.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use syn::{Item, Visibility};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("compiler-api lives under <workspace>/crates")
        .to_path_buf()
}

fn reject_functions(path: &Path) -> BTreeSet<(String, bool)> {
    let source =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let file = syn::parse_file(&source)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
    file.items
        .into_iter()
        .filter_map(|item| {
            let Item::Fn(function) = item else {
                return None;
            };
            let name = function.sig.ident.to_string();
            name.starts_with("reject_")
                .then_some((name, matches!(function.vis, Visibility::Public(_))))
        })
        .collect()
}

#[test]
fn phase3_gate_inventory_has_one_definition_per_decision() {
    let root = workspace_root();
    let cli = reject_functions(&root.join("crates/chelis-cli/src/main.rs"));
    let compiler = reject_functions(&root.join("crates/chelis-compiler-api/src/compiler.rs"));

    let expected_cli = BTreeSet::from([
        ("reject_unsupported_effect_ops".to_string(), false),
        ("reject_unsupported_metal_ops".to_string(), false),
        ("reject_with_seed_for_build_target".to_string(), false),
    ]);
    let expected_compiler = BTreeSet::from([
        ("reject_eval_only_builtins".to_string(), true),
        ("reject_host_only_builtins".to_string(), true),
        (
            "reject_host_only_builtins_before_host_lowering".to_string(),
            true,
        ),
        ("reject_symbolic_windowed_reduce".to_string(), true),
        ("reject_unsized_named_dims".to_string(), false),
        ("reject_unsupported_hip_ops".to_string(), true),
        (
            "reject_unsupported_reduce_window_precision".to_string(),
            true,
        ),
    ]);

    assert_eq!(cli, expected_cli, "review the CLI gate inventory change");
    assert_eq!(
        compiler, expected_compiler,
        "review the shared compiler gate inventory change"
    );

    let cli_names = cli.iter().map(|(name, _)| name).collect::<BTreeSet<_>>();
    let compiler_names = compiler
        .iter()
        .map(|(name, _)| name)
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
