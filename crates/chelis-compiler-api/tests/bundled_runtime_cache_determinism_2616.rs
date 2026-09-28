//! chelis#2616: the bundled chelis-std runtime carries no per-process
//! filesystem location.
//!
//! The package here has a `reef.lock`, so its graph is reconstructed from the
//! lock and contains the bundled runtime. Fresh worker processes, each with
//! its own `CHELIS_REEF_HOME` and its own temporary directory, prepare the
//! graph and compile the package context. The persisted prepared-graph cache,
//! the prepared-graph encoding and the compiled-context encoding must be the
//! same bytes in every process, and no process may leave an extraction
//! directory behind.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
use chelis_reef::{prepare_program_for_file, prepare_reef_graph_cached};
use tempfile::TempDir;

const WORKER_MODE: &str = "CHELIS_2616_WORKER_MODE";
const WORKER_PACKAGE_ROOT: &str = "CHELIS_2616_PACKAGE_ROOT";
const WORKER_RESULT_DIR: &str = "CHELIS_2616_RESULT_DIR";
const WORKERS: usize = 3;
const ARTIFACTS: [&str; 3] = [
    "prepared_graph_cache.bin",
    "prepared_graph_encode.bin",
    "compiled_encode.bin",
];

fn make_locked_package(parent: &Path) -> PathBuf {
    let root = parent.join("bundled-runtime-2616");
    fs::create_dir_all(root.join("src")).expect("create package source directory");
    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"bundled-runtime-2616\"\nversion = \"0.1.0\"\n\
             compiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"BundledRuntime\"\n\n\
             [dependencies]\nchelis-std = {{ version = \"{}\" }}\n",
            chelis_reef::compiler_bundled_chelis_std_version()
        ),
    )
    .expect("write reef.toml");
    fs::write(
        root.join("src/main.ch"),
        format!(
            "module BundledRuntime.Main\n{}\ndef main_value() -> i32 = cast(1, i32)\n",
            every_stdlib_module_import()
        ),
    )
    .expect("write source");
    fs::canonicalize(root).expect("canonicalize package root")
}

/// A qualified `import` of every chelis-std module (chelis#2558). A package
/// links only the chelis-std modules it imports, so a fixture that stands
/// for a package linking the whole standard library imports all of them.
fn every_stdlib_module_import() -> String {
    fn collect(dir: &Path, modules: &mut Vec<String>) {
        for entry in fs::read_dir(dir).expect("read chelis-std sources") {
            let path = entry.expect("chelis-std source entry").path();
            if path.is_dir() {
                collect(&path, modules);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("ch") {
                let source = fs::read_to_string(&path).expect("read chelis-std module");
                let module = source
                    .lines()
                    .find_map(|line| line.strip_prefix("module "))
                    .expect("a chelis-std source declares its module");
                modules.push(module.trim().to_string());
            }
        }
    }
    let mut modules = Vec::new();
    collect(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packages/chelis-std/src"),
        &mut modules,
    );
    modules.sort();
    modules
        .iter()
        .map(|module| format!("import {module}\n"))
        .collect()
}

/// Runs one fresh worker process and returns its result directory and the
/// bundled-runtime extraction directories it left in its temporary directory.
fn run_worker(
    mode: &str,
    package_root: &Path,
    scratch: &Path,
    label: &str,
) -> (PathBuf, Vec<OsString>) {
    let reef_home = scratch.join(format!("reef-home-{label}"));
    let tmp = scratch.join(format!("tmp-{label}"));
    let result_dir = scratch.join(format!("result-{label}"));
    fs::create_dir_all(&tmp).expect("create worker temporary directory");
    let output = Command::new(std::env::current_exe().expect("resolve test binary"))
        .args(["--exact", "bundled_runtime_worker", "--nocapture"])
        .env(WORKER_MODE, mode)
        .env(WORKER_PACKAGE_ROOT, package_root)
        .env(WORKER_RESULT_DIR, &result_dir)
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("TMPDIR", &tmp)
        .env_remove("CHELIS_STDLIB_CACHE_DISABLE")
        .output()
        .expect("run fresh worker process");
    assert!(
        output.status.success(),
        "worker {label} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let leaked = fs::read_dir(&tmp)
        .expect("read worker temporary directory")
        .map(|entry| entry.expect("read temporary entry").file_name())
        .filter(|name| name.to_string_lossy().starts_with("chelis-std-bundle-"))
        .collect::<Vec<_>>();
    (result_dir, leaked)
}

fn prepared_graph_cache_file(reef_home: &Path) -> PathBuf {
    let cache_dir = reef_home.join(".cache/prepared-graphs");
    let mut matches = fs::read_dir(&cache_dir)
        .expect("read prepared-graph cache directory")
        .map(|entry| entry.expect("read prepared-graph cache entry").path())
        .filter(|path| path.extension() == Some(OsStr::new("graph")))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "expected one prepared graph: {matches:?}");
    matches.pop().expect("one prepared graph")
}

#[test]
fn bundled_runtime_worker() {
    let Some(mode) = std::env::var_os(WORKER_MODE) else {
        return;
    };
    let package_root = PathBuf::from(std::env::var_os(WORKER_PACKAGE_ROOT).expect("package root"));
    let result_dir = PathBuf::from(std::env::var_os(WORKER_RESULT_DIR).expect("result dir"));
    let reef_home = PathBuf::from(std::env::var_os("CHELIS_REEF_HOME").expect("reef home"));
    fs::create_dir_all(&result_dir).expect("create result directory");
    if mode == "lock" {
        prepare_program_for_file(&package_root.join("src/main.ch"))
            .expect("prepare the entry program")
            .expect("the entry file is inside a package");
        return;
    }
    let prepared = prepare_reef_graph_cached(&package_root).expect("prepare package graph");
    assert!(
        !prepared.linked_stdlib_decls.is_empty(),
        "the locked package graph must contain the bundled runtime"
    );
    fs::copy(
        prepared_graph_cache_file(&reef_home),
        result_dir.join(ARTIFACTS[0]),
    )
    .expect("copy prepared-graph cache");
    fs::write(
        result_dir.join(ARTIFACTS[1]),
        prepared.encode().expect("encode prepared graph"),
    )
    .expect("write prepared-graph encoding");
    let context = compile_reef_context(&reef_home, &package_root).expect("compile package context");
    fs::write(
        result_dir.join(ARTIFACTS[2]),
        context.encode().expect("encode compiled context"),
    )
    .expect("write compiled-context encoding");
}

#[test]
fn bundled_runtime_cache_bytes_match_across_fresh_processes() {
    let package_dir = TempDir::new().expect("package tempdir");
    let scratch = TempDir::new().expect("scratch tempdir");
    let package_root = make_locked_package(package_dir.path());
    let _ = run_worker("lock", &package_root, scratch.path(), "lock");
    assert!(
        package_root.join("reef.lock").is_file(),
        "the lock worker must write reef.lock"
    );

    let mut reference: Option<Vec<Vec<u8>>> = None;
    for worker in 0..WORKERS {
        let (result_dir, _) =
            run_worker("graph", &package_root, scratch.path(), &worker.to_string());
        let observed = ARTIFACTS
            .iter()
            .map(|name| fs::read(result_dir.join(name)).expect("read worker artifact"))
            .collect::<Vec<_>>();
        match &reference {
            Some(expected) => {
                for ((name, expected), observed) in ARTIFACTS.iter().zip(expected).zip(&observed) {
                    assert!(
                        expected == observed,
                        "{name} changed in fresh process {worker}"
                    );
                }
            }
            None => reference = Some(observed),
        }
    }
}

#[test]
fn bundled_runtime_leaves_no_extraction_directory() {
    let package_dir = TempDir::new().expect("package tempdir");
    let scratch = TempDir::new().expect("scratch tempdir");
    let package_root = make_locked_package(package_dir.path());
    for mode in ["lock", "graph"] {
        let (_, leaked) = run_worker(mode, &package_root, scratch.path(), mode);
        assert!(
            leaked.is_empty(),
            "the {mode} worker left bundled-runtime extraction directories behind: {leaked:?}"
        );
    }
}
