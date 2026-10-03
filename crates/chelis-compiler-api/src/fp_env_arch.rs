//! Every public operation of this crate runs under the pinned floating-point
//! environment (spec/design/correctly_rounded_math.md section 6, chelis#2964).
//!
//! Literal finalization, constant folding, evaluation, and differentiation all
//! use host float arithmetic, and a host thread may carry any rounding mode or
//! flush-to-zero state. Rather than tracking which public functions reach that
//! arithmetic, every `pub fn` of the public surface opens with
//! [`chelis_runtime::FpEnvGuard`], which nests, so the rule needs no judgement
//! and a new entry point that skips the guard fails this test. The surface is
//! read from `lib.rs`: its `pub mod`s and the private modules it re-exports
//! from. A function re-exported from another crate cannot carry the guard, so
//! it must be wrapped, unless [`EXTERNAL_REEXPORTS_WITHOUT_ARITHMETIC`] names it.

use std::fs;
use std::path::{Path, PathBuf};

use syn::{ImplItem, Item, UseTree, Visibility};

const GUARD: &str = "chelis_runtime::FpEnvGuard::enter()";

/// Functions re-exported from other crates that do no float arithmetic, with
/// why. Anything else re-exported from another crate is wrapped.
const EXTERNAL_REEXPORTS_WITHOUT_ARITHMETIC: &[(&str, &str)] = &[
    ("find_package_root_for_dir", "filesystem walk for reef.toml"),
    ("find_package_root_for_input", "filesystem walk for reef.toml"),
    ("current_cancel_token", "thread-local read"),
    ("install_cancel_token", "thread-local write"),
    ("is_cancellation", "string search"),
    ("install_linked_program_guard", "thread-local write"),
];

fn is_cfg_test(attributes: &[syn::Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("cfg")
            && attribute
                .parse_args::<syn::Meta>()
                .is_ok_and(|meta| meta.path().is_ident("test"))
    })
}

fn opens_with_guard(block: &syn::Block) -> bool {
    let Some(syn::Stmt::Local(local)) = block.stmts.first() else {
        return false;
    };
    let syn::Pat::Ident(binding) = &local.pat else {
        return false;
    };
    binding.ident == "_fp_env"
        && local.init.as_ref().is_some_and(|init| {
            let syn::Expr::Call(call) = &*init.expr else {
                return false;
            };
            let syn::Expr::Path(function) = &*call.func else {
                return false;
            };
            let segments: Vec<String> = function
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect();
            call.args.is_empty() && segments == ["chelis_runtime", "FpEnvGuard", "enter"]
        })
}

fn check_function(
    path: &str,
    owner: Option<&str>,
    vis: &Visibility,
    sig: &syn::Signature,
    block: &syn::Block,
    checked: &mut Vec<String>,
    unguarded: &mut Vec<String>,
) {
    if !matches!(vis, Visibility::Public(_)) || sig.constness.is_some() {
        return;
    }
    let name = match owner {
        Some(owner) => format!("{path}: {owner}::{}", sig.ident),
        None => format!("{path}: {}", sig.ident),
    };
    if !opens_with_guard(block) {
        unguarded.push(name.clone());
    }
    checked.push(name);
}

fn impl_owner(self_ty: &syn::Type) -> String {
    match self_ty {
        syn::Type::Path(path) => path
            .path
            .segments
            .last()
            .map_or_else(String::new, |segment| segment.ident.to_string()),
        _ => String::new(),
    }
}

/// The public functions in `items` and whether each opens with the guard.
fn scan_items(
    path: &str,
    items: &[Item],
    checked: &mut Vec<String>,
    unguarded: &mut Vec<String>,
) {
    for item in items {
        match item {
            Item::Fn(function) if !is_cfg_test(&function.attrs) => check_function(
                path,
                None,
                &function.vis,
                &function.sig,
                &function.block,
                checked,
                unguarded,
            ),
            Item::Impl(block) if block.trait_.is_none() && !is_cfg_test(&block.attrs) => {
                let owner = impl_owner(&block.self_ty);
                for item in &block.items {
                    if let ImplItem::Fn(method) = item
                        && !is_cfg_test(&method.attrs)
                    {
                        check_function(
                            path,
                            Some(&owner),
                            &method.vis,
                            &method.sig,
                            &method.block,
                            checked,
                            unguarded,
                        );
                    }
                }
            }
            Item::Mod(module) if !is_cfg_test(&module.attrs) => {
                if let Some((_, items)) = &module.content {
                    scan_items(path, items, checked, unguarded);
                }
            }
            _ => {}
        }
    }
}

/// Lowercase leaves of a `use` tree: functions (or modules) by Rust naming.
fn use_leaves(tree: &UseTree, prefix: &mut Vec<String>, leaves: &mut Vec<(Vec<String>, String)>) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            use_leaves(&path.tree, prefix, leaves);
            prefix.pop();
        }
        UseTree::Name(name) => leaves.push((prefix.clone(), name.ident.to_string())),
        UseTree::Rename(rename) => leaves.push((prefix.clone(), rename.ident.to_string())),
        UseTree::Group(group) => {
            for tree in &group.items {
                use_leaves(tree, prefix, leaves);
            }
        }
        UseTree::Glob(_) => leaves.push((prefix.clone(), "*".to_string())),
    }
}

/// Function re-exports from another crate in `items` that are neither wrapped
/// nor named in [`EXTERNAL_REEXPORTS_WITHOUT_ARITHMETIC`].
fn unwrapped_external_reexports(path: &str, items: &[Item], crate_modules: &[String]) -> Vec<String> {
    let mut local = crate_modules.to_vec();
    local.extend(items.iter().filter_map(|item| match item {
        Item::Mod(module) => Some(module.ident.to_string()),
        _ => None,
    }));
    let mut found = Vec::new();
    for item in items {
        let Item::Use(item_use) = item else { continue };
        if !matches!(item_use.vis, Visibility::Public(_)) || is_cfg_test(&item_use.attrs) {
            continue;
        }
        let mut leaves = Vec::new();
        use_leaves(&item_use.tree, &mut Vec::new(), &mut leaves);
        for (prefix, leaf) in leaves {
            let root = prefix.first().map(String::as_str).unwrap_or_default();
            let is_local = matches!(root, "crate" | "self" | "super") || local.iter().any(|m| m == root);
            let is_function = leaf == "*" || leaf.starts_with(|c: char| c.is_ascii_lowercase());
            if !is_local
                && is_function
                && !EXTERNAL_REEXPORTS_WITHOUT_ARITHMETIC
                    .iter()
                    .any(|(name, _)| *name == leaf)
            {
                found.push(format!("{path}: {}::{leaf}", prefix.join("::")));
            }
        }
    }
    found
}

/// Every module `lib.rs` declares, and the files of those on the public surface.
fn surface_files(src: &Path, lib: &syn::File) -> (Vec<String>, Vec<PathBuf>) {
    let mut declared = Vec::new();
    let mut public = Vec::new();
    for item in &lib.items {
        if let Item::Mod(module) = item
            && !is_cfg_test(&module.attrs)
        {
            let name = module.ident.to_string();
            if matches!(module.vis, Visibility::Public(_)) {
                public.push(name.clone());
            }
            declared.push(name);
        }
    }
    for item in &lib.items {
        if let Item::Use(item_use) = item
            && matches!(item_use.vis, Visibility::Public(_))
        {
            let mut leaves = Vec::new();
            use_leaves(&item_use.tree, &mut Vec::new(), &mut leaves);
            // A private module joins the surface through a re-exported
            // function. A re-exported type's methods are not scanned; the one
            // such type, `RuntimeValue`, is the evaluator's value carrier.
            for (prefix, leaf) in leaves {
                if let Some(root) = prefix.first()
                    && leaf.starts_with(|c: char| c.is_ascii_lowercase())
                    && declared.contains(root)
                    && !public.contains(root)
                {
                    public.push(root.clone());
                }
            }
        }
    }
    let mut files = vec![src.join("lib.rs")];
    for module in &public {
        let file = src.join(format!("{module}.rs"));
        assert!(file.is_file(), "module file {} is missing", file.display());
        files.push(file);
        collect_rust_files(&src.join(module), &mut files);
    }
    (declared, files)
}

fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = entries.map(|entry| entry.unwrap().path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

fn parse(path: &Path) -> syn::File {
    syn::parse_file(&fs::read_to_string(path).unwrap())
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

#[test]
fn every_public_operation_pins_the_floating_point_environment() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let lib = parse(&src.join("lib.rs"));
    let (declared, files) = surface_files(&src, &lib);
    let mut checked = Vec::new();
    let mut unguarded = Vec::new();
    let mut reexports = Vec::new();
    for file in &files {
        let relative = file.strip_prefix(&src).unwrap().display().to_string();
        let parsed = parse(file);
        scan_items(&relative, &parsed.items, &mut checked, &mut unguarded);
        reexports.extend(unwrapped_external_reexports(&relative, &parsed.items, &declared));
    }
    assert!(
        unguarded.is_empty(),
        "public functions must open with `let _fp_env = {GUARD};`: {unguarded:#?}"
    );
    assert!(
        reexports.is_empty(),
        "functions re-exported from another crate must be wrapped under the guard: {reexports:#?}"
    );
    // The scan saw the operations this rule exists for.
    for operation in [
        "compiler.rs: compile",
        "compiler.rs: check",
        "compiler.rs: lower",
        "compiler.rs: eval",
        "compiler.rs: grad",
        "pipeline.rs: analyze_prepared",
        "context.rs: compile_reef_context",
    ] {
        assert!(
            checked.iter().any(|name| name == operation),
            "{operation} not scanned: {checked:#?}"
        );
    }
}

#[test]
fn an_unguarded_public_function_or_external_reexport_is_reported() {
    let source: syn::File = syn::parse_str(
        "pub fn guarded() { let _fp_env = chelis_runtime::FpEnvGuard::enter(); }
         pub fn unguarded() -> f32 { 0.1 + 0.2 }
         pub const fn constant() -> u32 { 1 }
         fn private() {}
         pub struct S;
         impl S { pub fn method(&self) {} }
         impl Clone for S { fn clone(&self) -> S { S } }
         #[cfg(test)] mod tests { pub fn helper() {} }
         pub use other_crate::{Type, folded_literal};
         pub use other_crate::is_cancellation;
         pub use local::anything;",
    )
    .unwrap();
    let mut checked = Vec::new();
    let mut unguarded = Vec::new();
    scan_items("planted.rs", &source.items, &mut checked, &mut unguarded);
    assert_eq!(
        unguarded,
        ["planted.rs: unguarded", "planted.rs: S::method"],
        "{checked:?}"
    );
    assert_eq!(
        unwrapped_external_reexports("planted.rs", &source.items, &["local".to_string()]),
        ["planted.rs: other_crate::folded_literal"]
    );
}
