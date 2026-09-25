//! chelis#2549: modules of one package that each read the same exported
//! library tensor value must evaluate, build, and test.
//!
//! The package linker gives every importing module the one library value,
//! and each importing `def main() = sampled` is a function declaration whose
//! free reference keeps its declaration scope ([04-INF-7], [04-LIN-4]). The
//! checker used to treat the first such declaration as a closure that
//! consumed the value, so the second importer failed every lane with
//! "already consumed by closure capture". A declaration may be called after
//! every initializer of the linked program, so the negative controls keep a
//! declaration that reads a value some initializer consumes rejected, in the
//! library or in a later-sorted module, as well as a consume-then-reuse inside
//! one declaration body.

mod common;

use assert_cmd::Command;
use common::{gcc_available, link_generated, parse_tensor_data};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

/// `app` with a path dependency on `drawlib`, which exports the tensor value
/// `sampled`. `drawlib` also carries a second module that re-reads `sampled`,
/// so the library leg has two readers of its own value.
fn shared_value_package(importers: &[(&str, &str)]) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("app");
    let version = env!("CARGO_PKG_VERSION");
    write_file(
        &root.join("reef.toml"),
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\ncompiler = \"={version}\"\n\
             module_prefix = \"App\"\n\n[dependencies]\ndrawlib = {{ path = \"./drawlib\" }}\n"
        ),
    );
    write_file(
        &root.join("reef.lock"),
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\n\n[[dependencies]]\n\
             name = \"drawlib\"\nversion = \"0.1.0\"\ncompiler = \"={version}\"\n\
             archive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\n\
             kind = \"path\"\npath = \"./drawlib\"\n"
        ),
    );
    write_file(
        &root.join("drawlib/reef.toml"),
        &format!(
            "[package]\nname = \"drawlib\"\nversion = \"0.1.0\"\ncompiler = \"={version}\"\n\
             module_prefix = \"Drawlib\"\n"
        ),
    );
    write_file(
        &root.join("drawlib/src/draw.ch"),
        "module Drawlib.Draw\nexport (sampled)\nsampled = to_tensor([1.0f32, 1.0f32])\n",
    );
    write_file(
        &root.join("drawlib/src/extra.ch"),
        "module Drawlib.Extra\nimport Drawlib.Draw (sampled)\nexport (again)\n\
         def again() -> tensor[2, f32] = sampled\n",
    );
    for (file, source) in importers {
        write_file(&root.join(file), source);
    }
    (dir, root)
}

fn importer(module: &str) -> String {
    format!(
        "module App.{module}\nimport Drawlib.Draw (sampled)\n\
         def main() -> tensor[2, f32] = sampled\n"
    )
}

fn chelis(root: &Path, cache: &Path, args: &[&str]) -> (bool, String, String) {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("XDG_CACHE_HOME", cache)
        .args(args)
        .output()
        .expect("run chelis");
    (
        output.status.success(),
        String::from_utf8(output.stdout).expect("stdout UTF-8"),
        String::from_utf8(output.stderr).expect("stderr UTF-8"),
    )
}

const CONSUMED_BY_CAPTURE: &str = "already consumed by closure capture";

#[test]
fn two_importing_modules_evaluate_the_shared_library_value() {
    let a = importer("A");
    let b = importer("B");
    let (dir, root) = shared_value_package(&[("src/a.ch", &a), ("src/b.ch", &b)]);
    let cache = dir.path().join("cache");
    for entry in ["src/a.ch", "src/b.ch"] {
        let (ok, stdout, stderr) = chelis(&root, &cache, &["eval", "--file", entry]);
        assert!(ok, "eval {entry} must succeed; stderr: {stderr}");
        assert!(!stderr.contains(CONSUMED_BY_CAPTURE), "{stderr}");
        assert_eq!(
            stdout, "main = tensor(shape=[2], data=[1.0, 1.0])\n",
            "eval {entry}"
        );
    }
}

#[test]
fn three_importers_and_a_second_import_path_evaluate_and_build() {
    let a = importer("A");
    let b = importer("B");
    let c = importer("C");
    let d = "module App.D\nimport Drawlib.Draw (sampled)\nimport Drawlib.Extra (again)\n\
             def main() -> tensor[2, f32] = add(again(), sampled)\n";
    let (dir, root) = shared_value_package(&[
        ("src/a.ch", &a),
        ("src/b.ch", &b),
        ("src/c.ch", &c),
        ("src/d.ch", d),
    ]);
    let cache = dir.path().join("cache");
    let (ok, stdout, stderr) = chelis(&root, &cache, &["eval", "--file", "src/d.ch"]);
    assert!(ok, "eval d must succeed; stderr: {stderr}");
    assert_eq!(stdout, "main = tensor(shape=[2], data=[2.0, 2.0])\n");

    let out = dir.path().join("out");
    let out_arg = out.to_str().expect("UTF-8 out dir");
    let (ok, _stdout, stderr) = chelis(&root, &cache, &["build", "src/d.ch", "-o", out_arg]);
    assert!(ok, "build d must succeed; stderr: {stderr}");
    assert!(
        out.join("d.c").is_file(),
        "build must emit C; stderr: {stderr}"
    );
    if !gcc_available() {
        eprintln!("skipped the native run of d.c: no host C compiler");
        return;
    }
    let status = link_generated(&out, "d.c", "d");
    assert!(status.success(), "link failed: {status}");
    let run = std::process::Command::new(out.join("d"))
        .output()
        .expect("compiled binary should run");
    let native = String::from_utf8(run.stdout).expect("UTF-8 stdout");
    assert!(run.status.success(), "compiled d failed: {native}");
    assert_eq!(
        parse_tensor_data(&native, "main"),
        parse_tensor_data(&stdout, "main"),
        "compiled C must print eval's `main`; native stdout: {native}"
    );
}

#[test]
fn two_test_files_reading_the_shared_library_value_pass() {
    let a = importer("A");
    let first = "module App.Tests.First\nimport Drawlib.Draw (sampled)\n\
                 def expected() -> tensor[2, f32] = to_tensor([1.0f32, 1.0f32])\n\
                 def test_first() -> unit ! { Test } = \
                 test_assert_eq_tensor(sampled, expected(), \"first\")\n";
    let second = "module App.Tests.Second\nimport Drawlib.Draw (sampled)\n\
                  def owned() -> tensor[2, f32] = sampled\n\
                  def test_second() -> unit ! { Test } = \
                  test_assert_eq_tensor(owned(), sampled, \"second\")\n";
    let (dir, root) = shared_value_package(&[
        ("src/a.ch", &a),
        ("tests/first.ch", first),
        ("tests/second.ch", second),
    ]);
    let cache = dir.path().join("cache");
    let (ok, stdout, stderr) = chelis(&root, &cache, &["test", "tests/"]);
    assert!(
        ok,
        "chelis test must pass; stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains("2 passed, 0 failed"), "stdout: {stdout}");
}

#[test]
fn consume_then_reuse_inside_one_importing_declaration_is_still_rejected() {
    let a = importer("A");
    let bad = "module App.Bad\nimport Drawlib.Draw (sampled)\n\
               def main() -> tensor[2, f32] = {\n  y = realize(sampled)\n  add(sampled, y)\n}\n";
    let (dir, root) = shared_value_package(&[("src/a.ch", &a), ("src/bad.ch", bad)]);
    let cache = dir.path().join("cache");
    let (ok, _stdout, stderr) = chelis(&root, &cache, &["eval", "--file", "src/a.ch"]);
    assert!(!ok, "a consume-then-reuse in App.Bad must fail the package");
    assert!(
        stderr.contains("sampled`")
            && stderr.contains("already consumed by realize")
            && !stderr.contains(CONSUMED_BY_CAPTURE),
        "the only rejection must be the realize consume of `sampled`; stderr: {stderr}"
    );
}

#[test]
fn importer_declaration_reading_a_value_the_library_consumes_is_rejected() {
    let a = importer("A");
    let (dir, root) = shared_value_package(&[("src/a.ch", &a)]);
    write_file(
        &root.join("drawlib/src/draw.ch"),
        "module Drawlib.Draw\nexport (sampled)\nsampled = to_tensor([1.0f32, 1.0f32])\n\
         spent = realize(sampled)\n",
    );
    write_file(
        &root.join("drawlib/src/extra.ch"),
        "module Drawlib.Extra\nexport (again)\n\
         def again() -> tensor[2, f32] = to_tensor([1.0f32, 1.0f32])\n",
    );
    let cache = dir.path().join("cache");
    let (ok, _stdout, stderr) = chelis(&root, &cache, &["eval", "--file", "src/a.ch"]);
    assert!(!ok, "App.A reads `sampled` after the library consumes it");
    assert!(
        stderr.contains("sampled`") && stderr.contains("already consumed by realize"),
        "the rejection must be the library's realize of `sampled`; stderr: {stderr}"
    );
}

#[test]
fn declaration_before_a_later_module_consume_is_rejected() {
    // App.A's declaration sorts before App.Z's consuming initializer, and
    // App.Z's own `main` is called after it, so App.A's text position cannot
    // make the read valid.
    let a = importer("A");
    let z = "module App.Z\nimport Drawlib.Draw (sampled)\nspent = realize(sampled)\n";
    let (dir, root) = shared_value_package(&[("src/a.ch", &a), ("src/z.ch", z)]);
    let cache = dir.path().join("cache");
    let (ok, _stdout, stderr) = chelis(&root, &cache, &["eval", "--file", "src/a.ch"]);
    assert!(!ok, "App.A reads `sampled` that App.Z consumes");
    assert!(
        stderr.contains("sampled`") && stderr.contains("already consumed by realize"),
        "the rejection must be App.Z's realize of `sampled`; stderr: {stderr}"
    );
}
