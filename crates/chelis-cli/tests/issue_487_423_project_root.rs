//! chelis#487 + chelis#423: a `--project-root` package-proof / package-eval
//! mode, and the unification of `prove` and `eval` import resolution.
//!
//! ## #487 — prove a standalone file against a package
//!
//! `chelis prove` already resolves imports for a file INSIDE a reef package
//! (it walks up to the `reef.toml`). A standalone property file living
//! OUTSIDE any package could not be proved against the package's exported
//! functions: the import did not resolve, the property's call to an exported
//! function was unbound, and the property fell to `unsupported`/error.
//! `--project-root <dir>` resolves the file's imports against the package
//! rooted there, so the property proves against the REAL dependency graph.
//!
//! ## #423 — eval a standalone file / inline expression against a package
//!
//! `chelis eval --file` and `chelis eval <expr>` gain the same
//! `--project-root`. Both `prove` and `eval` now resolve imports through the
//! ONE `chelis_reef::prepare_program_for_eval_source` path (the unification
//! the issue asks for), so a standalone file that imports an exported
//! package function evaluates to the same value the package would produce.
//!
//! ## Acceptance oracle (negative parity)
//!
//! An UNRESOLVABLE import under `--project-root` yields an honest
//! `import resolution failed` error (exit non-zero, never a false pass).
//! Without `--project-root`, a standalone file importing a package function
//! is never reported as proved/passed.

use assert_cmd::Command;
#[cfg(feature = "smt")]
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

/// A reef package `demopkg` (module prefix `Demo`) exporting `square`.
/// Returned `TempDir` keeps the package alive; the `PathBuf` is its root.
fn demo_package() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("demopkg");
    write_file(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "demopkg"
version = "0.1.0"
compiler = "={}"
module_prefix = "Demo"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &root.join("src/model.ch"),
        "module Demo.Model\nexport (square)\ndef square(x: f32) -> f32 = x * x\n",
    );
    (dir, root)
}

/// A standalone file OUTSIDE the package importing `Demo.Model.square`.
/// Placed in its own tempdir so no `reef.toml` is discoverable from it.
fn standalone(contents: &str) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("standalone.ch");
    fs::write(&file, contents).expect("write standalone");
    (dir, file)
}

#[cfg(feature = "smt")]
fn property_records(output: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(output)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .filter(|record| record.get("kind").and_then(Value::as_str) == Some("property"))
        .collect()
}

// ---------------------------------------------------------------------------
// #487 — prove a standalone file against a package via --project-root.
// ---------------------------------------------------------------------------

/// With `--project-root`, a standalone property file outside the package
/// resolves its import and PROVES at the SMT tier against the real exported
/// function.
#[cfg(feature = "smt")]
#[test]
fn prove_standalone_file_with_project_root_proves_at_smt() {
    let (_pkg, root) = demo_package();
    let (_dir, file) = standalone(
        "import Demo.Model (square)\n@property sq_nn forall(x: f32):\n  (square(x) >= 0.0)\n",
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            file.to_str().unwrap(),
            "--json",
            "--tier",
            "smt-only",
            "--project-root",
            root.to_str().unwrap(),
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "sq_nn");
    assert_eq!(props[0]["status"], "passed");
    assert_eq!(props[0]["proof_tier"], "smt");
    // The goal carries the RESOLVED, linker-internal name — proof that the
    // import was resolved against the package, not left dangling.
    let goal = props[0]["goal"].as_str().unwrap_or_default();
    assert!(
        goal.contains("pkg__demopkg__Demo__Model__square"),
        "the goal must reference the resolved package symbol, got {goal}"
    );
}

/// NEGATIVE twin: the SAME standalone file WITHOUT `--project-root` is never
/// reported as proved — the import does not resolve, so the property is
/// unsupported/error and the summary has zero passes.
#[cfg(feature = "smt")]
#[test]
fn prove_standalone_file_without_project_root_is_never_a_pass() {
    let (_pkg, _root) = demo_package();
    let (_dir, file) = standalone(
        "import Demo.Model (square)\n@property sq_nn forall(x: f32):\n  (square(x) >= 0.0)\n",
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            file.to_str().unwrap(),
            "--json",
            "--tier",
            "smt-only",
        ])
        .output()
        .expect("run prove");
    // Whatever the exit code, the property must never be reported passed.
    let props = property_records(&output.stdout);
    for record in &props {
        assert_ne!(
            record["status"], "passed",
            "an unresolved import must never prove: {record:?}"
        );
    }
    let combined = String::from_utf8_lossy(&output.stdout);
    assert!(
        !combined.contains("\"status\":\"passed\""),
        "an unresolved import must never prove; stdout={combined}"
    );
}

/// NEGATIVE: an UNRESOLVABLE import under `--project-root` (a module the
/// package does not provide) is an honest `import resolution failed` error,
/// never a false pass. This is the #487 acceptance oracle's negative side.
#[cfg(feature = "smt")]
#[test]
fn prove_standalone_unresolvable_import_is_honest_error_not_false_pass() {
    let (_pkg, root) = demo_package();
    let (_dir, file) = standalone(
        "import Demo.Nonexistent (square)\n@property sq_nn forall(x: f32):\n  (square(x) >= 0.0)\n",
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            file.to_str().unwrap(),
            "--json",
            "--tier",
            "smt-only",
            "--project-root",
            root.to_str().unwrap(),
        ])
        .output()
        .expect("run prove");
    assert_ne!(
        output.status.code(),
        Some(0),
        "an unresolvable import must not exit success; stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("import resolution failed"),
        "expected an honest import-resolution error, got {stdout}"
    );
    assert!(
        !stdout.contains("\"status\":\"passed\""),
        "an unresolvable import must never prove; stdout={stdout}"
    );
}

// ---------------------------------------------------------------------------
// #423 — eval a standalone file / inline expression against a package.
// ---------------------------------------------------------------------------

/// `chelis eval --file <standalone.ch> --project-root <pkg>` resolves the
/// import and evaluates the call to the exported function.
#[test]
fn eval_standalone_file_with_project_root_resolves_import() {
    let (_pkg, root) = demo_package();
    let (_dir, file) = standalone("import Demo.Model (square)\nanswer = square(4.0)\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "eval",
            "--file",
            file.to_str().unwrap(),
            "--project-root",
            root.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("16"));
}

/// The unified resolution: `prove` and `eval` resolve the SAME standalone
/// import against the SAME package root. Eval produces the numeric value;
/// prove proves the property — both succeed where neither did before
/// `--project-root`.
#[cfg(feature = "smt")]
#[test]
fn eval_and_prove_share_one_import_resolution_path() {
    let (_pkg, root) = demo_package();
    // A property file (prove) and a value file (eval) that import the same
    // exported function from the same package.
    let (_pdir, prop_file) = standalone(
        "import Demo.Model (square)\n@property nn forall(x: f32):\n  (square(x) >= 0.0)\n",
    );
    let eval_dir = tempdir().expect("tempdir");
    let eval_file = eval_dir.path().join("value.ch");
    fs::write(
        &eval_file,
        "import Demo.Model (square)\nresult = square(5.0)\n",
    )
    .expect("write eval file");

    // prove resolves the import and proves.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            prop_file.to_str().unwrap(),
            "--tier",
            "smt-only",
            "--project-root",
            root.to_str().unwrap(),
        ])
        .assert()
        .success();

    // eval resolves the SAME import and produces the value (5*5 = 25).
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "eval",
            "--file",
            eval_file.to_str().unwrap(),
            "--project-root",
            root.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("25"));
}

/// An inline expression evaluated against a package context (chelis#423's
/// "eval mode that loads a package context and evaluates an inline
/// expression"). A bare arithmetic / builtin expression resolves against the
/// package — it does NOT regress to "no context". (A package EXPORT still
/// needs an `import`, which a one-liner cannot express; that's an inherent
/// naming constraint, exercised by the `--file` cases above.)
#[test]
fn eval_inline_expr_with_project_root_loads_package_context() {
    let (_pkg, root) = demo_package();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "eval",
            "mul(6.0, 7.0)",
            "--project-root",
            root.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("42"));
}

/// Regression: an inline expression with NO package in scope still evaluates
/// as a standalone snippet (the legacy behavior is preserved).
#[test]
fn eval_inline_expr_without_package_is_unchanged() {
    let dir = tempdir().expect("tempdir");
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .args(["eval", "2.0 + 3.0"])
        .assert()
        .success()
        .stdout(predicates::str::contains("5"));
}
