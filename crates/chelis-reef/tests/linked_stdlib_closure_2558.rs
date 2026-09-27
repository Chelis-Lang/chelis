//! chelis#2558: a prepared package graph links only the chelis-std modules
//! that its packages and its caller's entries import, transitively. The
//! linker resolves a name from another module only through an `import`, so
//! that closure is everything linking can reach.

use chelis_reef::{
    EntryImports, PreparedReefGraph, SourceDigest, compile_with_reef_graph,
    compiler_bundled_chelis_std_version, prepare_reef_graph, rewrite_entry_decls_with_reef_graph,
};
use chelis_surf::ast::Decl;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

const NO_IMPORT_MODULE: &str = "module Probe.Main
export (answer)
def answer() -> i64 = match Some(cast(1, i64)) with {
  | Some(value) => value
  | None => cast(0, i64)
}
";

const JSON_MODULE: &str = "module Probe.Main
import Std.Io.Json (try_parse_json)
export (parses)
def parses() -> bool = match try_parse_json(\"{}\") with {
  | Some(_) => true
  | None => false
}
";

const JSON_SNIPPET: &str = "import Std.Io.Json (try_parse_json)
bench = match try_parse_json(\"{}\") with {
  | Some(_) => cast(1, i64)
  | None => cast(0, i64)
}
";

const TEXT_THROUGH_DEPENDENCY: &str = "module Probe.Main
import Mylib.Words (joined)
export (answer)
def answer() -> string = joined()
";

const DEPENDENCY_MODULE: &str = "module Mylib.Words
import Std.Text (join)
export (joined)
def joined() -> string = join([\"a\", \"b\"], \",\")
";

fn write(path: &Path, contents: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, contents).expect("write fixture");
}

fn manifest(name: &str, prefix: &str, extra_dependencies: &str) -> String {
    format!(
        "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"{prefix}\"\n\n[dependencies]\nchelis-std = {{ version = \"{}\" }}\n{extra_dependencies}",
        env!("CARGO_PKG_VERSION"),
        compiler_bundled_chelis_std_version(),
    )
}

/// A package whose manifest names the bundled runtime, so chelis-std is in
/// its graph with or without a lock, and whose one module is `main`.
fn package(main: &str) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("probe");
    write(&root.join("reef.toml"), &manifest("probe", "Probe", ""));
    write(&root.join("src/main.ch"), main);
    (dir, root)
}

fn prepared(root: &Path) -> PreparedReefGraph {
    prepare_reef_graph(root).expect("prepare graph")
}

fn modules(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| name.to_string()).collect()
}

fn decl_name(decl: &Decl) -> Option<&str> {
    match decl {
        Decl::FunDef { name, .. }
        | Decl::LetDef { name, .. }
        | Decl::Sig { name, .. }
        | Decl::TypeDef { name, .. }
        | Decl::TypeAlias { name, .. }
        | Decl::MacroDef { name, .. }
        | Decl::Property { name, .. } => Some(name),
        Decl::Import { .. } | Decl::Export { .. } | Decl::Module { .. } | Decl::Dim { .. } => None,
    }
}

/// The chelis-std modules whose declarations appear in `decls`, read from
/// the linker's internal names rather than from the graph's own record.
fn modules_named_by(decls: &[Decl], candidates: &[&str]) -> BTreeSet<String> {
    let names = decls.iter().filter_map(decl_name).collect::<Vec<_>>();
    for name in &names {
        assert!(
            candidates.iter().any(|module| {
                let stem = format!("__chelis__std__{}__", module.replace('.', "__"));
                name.starts_with(&format!("pkg{stem}")) || name.starts_with(&format!("Pkg{stem}"))
            }),
            "linked chelis-std declaration `{name}` belongs to none of {candidates:?}"
        );
    }
    candidates
        .iter()
        .filter(|module| {
            let stem = format!("__chelis__std__{}__", module.replace('.', "__"));
            names
                .iter()
                .any(|name| name.starts_with(&format!("pkg{stem}")))
        })
        .map(|module| module.to_string())
        .collect()
}

fn closure_row(digests: &[SourceDigest]) -> &SourceDigest {
    let rows = digests
        .iter()
        .filter(|digest| digest.module_name == "<linked::stdlib-modules>")
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 1, "exactly one linked-module row: {rows:?}");
    rows[0]
}

#[test]
fn a_package_that_imports_nothing_links_no_stdlib_module() {
    let (_dir, root) = package(NO_IMPORT_MODULE);
    let graph = prepared(&root);
    // chelis-std is in the graph; it is only not linked.
    closure_row(&graph.source_digests().expect("digests"));
    assert_eq!(graph.linked_stdlib_modules(), BTreeSet::new());
    assert!(
        graph.linked_stdlib_decls.is_empty(),
        "{} chelis-std declarations linked into a package that imports none",
        graph.linked_stdlib_decls.len()
    );
    assert!(
        !graph.linked_library_decls.is_empty(),
        "the package's own module is still linked"
    );
}

#[test]
fn an_imported_stdlib_module_links_with_its_transitive_imports_only() {
    let (_dir, root) = package(JSON_MODULE);
    let graph = prepared(&root);
    // `Std.Io.Json` imports `Std.Text`; nothing else is reachable.
    let expected = modules(&["Std.Io.Json", "Std.Text"]);
    assert_eq!(graph.linked_stdlib_modules(), expected);
    assert_eq!(
        modules_named_by(&graph.linked_stdlib_decls, &["Std.Io.Json", "Std.Text"]),
        expected
    );
}

#[test]
fn a_dependency_import_of_the_stdlib_is_linked_for_a_root_that_imports_none() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("probe");
    write(
        &root.join("reef.toml"),
        &manifest("probe", "Probe", "mylib = { path = \"./mylib\" }\n"),
    );
    write(&root.join("src/main.ch"), TEXT_THROUGH_DEPENDENCY);
    write(
        &root.join("mylib/reef.toml"),
        &manifest("mylib", "Mylib", ""),
    );
    write(&root.join("mylib/src/words.ch"), DEPENDENCY_MODULE);
    let graph = prepared(&root);
    assert_eq!(graph.linked_stdlib_modules(), modules(&["Std.Text"]));
    assert_eq!(
        modules_named_by(&graph.linked_stdlib_decls, &["Std.Text"]),
        modules(&["Std.Text"])
    );
}

#[test]
fn the_stdlib_as_root_package_links_every_module() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packages/chelis-std");
    let graph = prepared(&root);
    let mut declared = BTreeSet::new();
    for entry in walkdir(&root.join("src")) {
        let source = fs::read_to_string(&entry).expect("read std module");
        let module = source
            .lines()
            .find_map(|line| line.strip_prefix("module "))
            .expect("std source declares its module");
        declared.insert(module.trim().to_string());
    }
    assert!(declared.contains("Std.Test"), "{declared:?}");
    assert_eq!(graph.linked_stdlib_modules(), declared);
}

fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).expect("read dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            files.extend(walkdir(&path));
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("ch") {
            files.push(path);
        }
    }
    files
}

#[test]
fn an_entry_extends_the_linked_modules_and_the_digests_name_the_set() {
    let (_dir, root) = package(NO_IMPORT_MODULE);
    let graph = prepared(&root);
    let snippet = chelis_surf::parser::parse_str(JSON_SNIPPET).expect("parse snippet");

    // Entries the package already covers leave the graph as it is.
    let unchanged = graph.covering(&EntryImports::none()).expect("cover none");
    assert!(matches!(unchanged, std::borrow::Cow::Borrowed(_)));

    let covered = graph
        .covering(&EntryImports::from_decls(&snippet))
        .expect("cover the snippet")
        .into_owned();
    assert_eq!(
        covered.linked_stdlib_modules(),
        modules(&["Std.Io.Json", "Std.Text"])
    );
    assert_eq!(
        modules_named_by(&covered.linked_stdlib_decls, &["Std.Io.Json", "Std.Text"]),
        modules(&["Std.Io.Json", "Std.Text"])
    );

    // The same sources with a different linked set must key differently.
    let before = graph.source_digests().expect("digests");
    let after = covered.source_digests().expect("digests");
    assert_ne!(closure_row(&before).sha256, closure_row(&after).sha256);
    let without_row = |digests: &[SourceDigest]| {
        digests
            .iter()
            .filter(|digest| digest.module_name != "<linked::stdlib-modules>")
            .cloned()
            .collect::<Vec<_>>()
    };
    assert_eq!(without_row(&before), without_row(&after));
    assert_ne!(graph.stdlib_source_digest(), covered.stdlib_source_digest());
}

#[test]
fn compiling_an_entry_links_the_stdlib_modules_it_imports() {
    let (_dir, root) = package(NO_IMPORT_MODULE);
    let graph = prepared(&root);
    let snippet = chelis_surf::parser::parse_str(JSON_SNIPPET).expect("parse snippet");
    let program = compile_with_reef_graph(&graph, &snippet).expect("compile snippet");
    assert_eq!(
        modules_named_by(&program.stdlib_decls, &["Std.Io.Json", "Std.Text"]),
        modules(&["Std.Io.Json", "Std.Text"])
    );
}

/// Negative parity: an entry rewritten against a graph prepared without its
/// imports is refused by name, never linked to declarations that are absent.
#[test]
fn rewriting_an_entry_against_a_graph_without_its_imports_is_refused() {
    let (_dir, root) = package(NO_IMPORT_MODULE);
    let graph = prepared(&root);
    let snippet = chelis_surf::parser::parse_str(JSON_SNIPPET).expect("parse snippet");
    let error = rewrite_entry_decls_with_reef_graph(&graph, &snippet)
        .expect_err("the graph does not link Std.Io.Json");
    assert!(
        error.contains("standard-library module `Std.Io.Json` is not linked"),
        "{error}"
    );
}
