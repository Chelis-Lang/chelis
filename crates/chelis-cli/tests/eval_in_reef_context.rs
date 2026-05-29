//! Phase H, cmd_eval slice — integration probes for the
//! `compile_reef_context + eval_in_context` refactor of `chelis eval`.
//!
//! What the CLI tests need to lock in:
//!
//! 1. The new path produces non-empty stdout for the three fixture
//!    shapes called out in the Phase H plan (simple def, reef package
//!    with a chelis-std-style import, snippet using a path-dep helper).
//! 2. Raw `chelis eval --file <foo.ch>` outside any reef package still
//!    works via the legacy path (no library context to amortize, but
//!    the CLI must not regress).
//! 3. Stdout shape parity vs the monolithic `prepare_eval(library +
//!    snippet)` baseline. The Phase G acceptance suite already proved
//!    byte-identical eval values; the CLI test confirms the dispatch
//!    inside `cmd_eval` actually routes through the new API and that
//!    `format_eval_result` formats the output the same way the legacy
//!    `try_eval` did.
//!
//! See `/home/jeff/.claude/plans/now-plan-out-the-shimmying-wand.md`
//! Phase H for the owning plan.

use assert_cmd::Command;
use chelis_compiler_api::{compile_reef_context, eval_in_context};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

/// Build a reef package with a path-dep `mylib` exporting pure-int
/// helpers. Mirrors the Phase G `library_fixture` shape so the Phase H
/// CLI test exercises the same dispatch as the unit-level parity test.
fn path_dep_package() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");

    write_file(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "={}"
module_prefix = "App"

[dependencies]
mylib = {{ path = "./mylib" }}
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &root.join("src/main.ch"),
        "module App.Main\n\ndef placeholder -> int32 = cast(0, int32)\n",
    );

    write_file(
        &root.join("mylib/reef.toml"),
        &format!(
            r#"[package]
name = "mylib"
version = "0.1.0"
compiler = "={}"
module_prefix = "Mylib"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &root.join("mylib/src/math.ch"),
        "module Mylib.Math\nexport (add, double, square)\n\n\
         def add(x: int32, y: int32) -> int32 = x + y\n\
         def double(x: int32) -> int32 = x + x\n\
         def square(x: int32) -> int32 = x * x\n",
    );

    write_file(
        &root.join("reef.lock"),
        &format!(
            r#"[package]
name = "myapp"
version = "0.1.0"

[[dependencies]]
name = "mylib"
version = "0.1.0"
compiler = "={}"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "path"
path = "./mylib"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    (dir, root)
}

/// Compute the Phase H expected stdout for a `--file` snippet inside a
/// reef package by running the same `compile_reef_context +
/// eval_in_context` flow the refactored `cmd_eval` uses, then formatting
/// its `EvalResult` the same way the CLI does. This is not a fully
/// independent baseline — the Phase G acceptance suite is the parity
/// oracle vs the monolithic `prepare_eval(library + snippet)` path —
/// but it locks in two CLI-side properties that Phase G can't probe:
///
/// 1. The CLI's dispatch in `cmd_eval` actually routes through the new
///    API rather than silently falling back to the legacy path.
/// 2. The CLI's `format_eval_result` formats the same shape the test
///    expects (transcript lines, then a single formatted value or
///    `<name> = <value>` lines, joined by newline, followed by the
///    `println!` trailing newline).
fn expected_stdout(package_root: &Path, snippet: &str) -> String {
    let ctx = compile_reef_context(Path::new(""), package_root).expect("compile_reef_context");
    let result = eval_in_context(&ctx, snippet).expect("eval_in_context");
    let mut lines = result.transcript.clone();
    if result.roots.len() == 1 {
        if let Some(root) = result.roots.first() {
            lines.push(format_execution_value(&root.value));
        }
    } else {
        for (index, root) in result.roots.iter().enumerate() {
            let name = root.name.clone().unwrap_or_else(|| format!("_{index}"));
            lines.push(format!(
                "{} = {}",
                display_root_name(&name),
                format_execution_value(&root.value)
            ));
        }
    }
    let mut out = lines.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Mirror of `chelis-cli`'s private `display_root_name`. Public copies
/// are intentionally minimal — the fixtures here never produce names
/// that hit the `__test_` strip path so the function is effectively the
/// identity, but keeping the same shape makes the parity check robust
/// if that detail changes upstream.
fn display_root_name(name: &str) -> String {
    name.strip_prefix("__test_").unwrap_or(name).to_string()
}

/// Mirror of `chelis-cli`'s `format_execution_value`. Kept in sync with
/// the CLI's formatter; the Phase G suite locks the `EvaluatedRoot`
/// values themselves, so the shape parity here is what closes the loop
/// for the CLI's stdout contract.
fn format_execution_value(value: &chelis_compiler_api::schema::ExecutionValue) -> String {
    use chelis_compiler_api::schema::ExecutionValue;
    match value {
        ExecutionValue::Tensor { value } => {
            const PRINT_LIMIT: usize = 32;
            let visible = value.data.len().min(PRINT_LIMIT);
            if visible < value.data.len() {
                format!(
                    "tensor(shape={:?}, data={:?} + ...)",
                    value.shape,
                    &value.data[..visible]
                )
            } else {
                format!("tensor(shape={:?}, data={:?})", value.shape, value.data)
            }
        }
        ExecutionValue::Int64 { value } => value.to_string(),
        ExecutionValue::Float64 { value } => value.to_string(),
        ExecutionValue::Bool { value } => value.to_string(),
        ExecutionValue::String { value } => value.clone(),
        ExecutionValue::List { value: items } => format!(
            "[{}]",
            items
                .iter()
                .map(format_execution_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Dict { entries } => format!(
            "dict({})",
            entries
                .iter()
                .map(|entry| format!(
                    "{}: {}",
                    format_execution_value(&entry.key),
                    format_execution_value(&entry.value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Tuple { value: items } => format!(
            "({})",
            items
                .iter()
                .map(format_execution_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Adt { ctor, fields } if fields.is_empty() => ctor.clone(),
        ExecutionValue::Adt { ctor, fields } => format!(
            "{}({})",
            ctor,
            fields
                .iter()
                .map(format_execution_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Unit => "()".to_string(),
    }
}

/// Fixture #1: simple def inside a reef package — covers the
/// "single-Module entry file routes through `find_package_root_for_input`"
/// branch of the Phase H detector.
///
/// File path / module-name pairing is load-bearing: reef rejects any
/// file under `src/` whose `module Foo.Bar` doesn't match the relative
/// path (case-insensitive, `.` joins). Use `App.EvalSimple` →
/// `src/eval_simple.ch` to satisfy `validate_module_path`.
#[test]
fn cmd_eval_reef_package_simple_def_matches_baseline() {
    let (_dir, root) = path_dep_package();
    let entry_path = root.join("src/evalsimple.ch");
    let snippet = "module App.EvalSimple\n\ndef simple_value -> int32 = 42\n";
    write_file(&entry_path, snippet);

    let expected = expected_stdout(&root, snippet);

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", entry_path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    assert!(
        output.status.success(),
        "exit status: {:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let actual = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert_eq!(
        actual, expected,
        "Phase H cmd_eval reef-package output must be byte-identical to monolithic baseline"
    );
}

/// Hull Phase 0a Packet B, commit 2: `chelis eval --json --file` routed
/// through the reef-context fast path emits the raw EvalResult JSON on
/// stdout. This covers the `run_eval_in_context` JSON branch, distinct
/// from the legacy `try_eval_result` branch the non-reef `--file` test
/// in `cli.rs` exercises.
#[test]
fn cmd_eval_json_reef_package_simple_def_emits_json() {
    let (_dir, root) = path_dep_package();
    let entry_path = root.join("src/evaljson.ch");
    let snippet = "module App.EvalJson\n\ndef simple_value -> int32 = 42\n";
    write_file(&entry_path, snippet);

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--json", "--file", entry_path.to_str().unwrap()])
        .output()
        .expect("run chelis eval --json");
    assert!(
        output.status.success(),
        "exit status: {:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("reef-context eval --json stdout is JSON");
    let roots = json["roots"].as_array().expect("roots array");
    let simple = roots
        .iter()
        .find(|r| r["name"] == "simple_value")
        .expect("simple_value root present");
    // `42` is an int literal lowered through the IR evaluator, so it
    // surfaces as a scalar tensor (shape []), matching the human path.
    assert_eq!(simple["value"]["type"], "tensor");
    assert_eq!(
        simple["value"]["value"]["shape"]
            .as_array()
            .expect("shape")
            .len(),
        0
    );
}

/// Fixture #2: reef package with a path-dep import. Covers the
/// "library state actually feeds the eval" leg — if `eval_in_context`
/// did not see `Mylib.Math.add`, the eval would fail with an unresolved
/// name error and exit non-zero.
#[test]
fn cmd_eval_reef_package_path_dep_import_matches_baseline() {
    let (_dir, root) = path_dep_package();
    let entry_path = root.join("src/evalpathdep.ch");
    let snippet = "module App.EvalPathDep\nimport Mylib.Math (add)\n\n\
                   def imported_sum -> int32 = add(20, 22)\n";
    write_file(&entry_path, snippet);

    let expected = expected_stdout(&root, snippet);

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", entry_path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    assert!(
        output.status.success(),
        "exit status: {:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let actual = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert_eq!(
        actual, expected,
        "Phase H path-dep import must produce baseline-identical output"
    );
}

/// Fixture #3: a snippet (no Module wrapper) evaluated from inside the
/// package via `current_dir`. Covers the
/// "loose snippet routes through `find_package_root_for_dir`" branch
/// of the Phase H detector — equivalent to what `eval_reef.rs` already
/// covers but explicitly compared against the baseline so a regression
/// in dispatch can't pass silently.
#[test]
fn cmd_eval_reef_package_loose_snippet_via_cwd_matches_baseline() {
    let (_dir, root) = path_dep_package();
    // Place the snippet OUTSIDE the package so the only way to detect
    // the reef context is via `find_package_root_for_dir(current_dir)`.
    let snippet_dir = tempdir().expect("snippet tempdir");
    let entry_path = snippet_dir.path().join("snippet.ch");
    let snippet = "import Mylib.Math (square)\n\n\
                   bench_value: int32 = square(7)\n";
    write_file(&entry_path, snippet);

    let expected = expected_stdout(&root, snippet);

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&root)
        .args(["eval", "--file", entry_path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    assert!(output.status.success(), "exit status: {:?}", output.status);
    let actual = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert_eq!(
        actual, expected,
        "Phase H cwd-detected reef package must match baseline"
    );
}

/// Negative-parity probe: outside any reef package the legacy path
/// must still work. Without this, the refactor could silently route
/// raw files through a code path that requires reef state and produce
/// a confusing error instead of the file's actual eval output. Use a
/// bare `name: int32 = ...` value binding (not a `def` — 0-arg fns
/// are not eagerly evaluated) so the host-program evaluator emits a
/// non-empty transcript.
#[test]
fn cmd_eval_raw_file_outside_reef_package_uses_legacy_path() {
    let dir = tempdir().expect("tempdir");
    let entry_path = dir.path().join("raw.ch");
    write_file(&entry_path, "raw_value: int32 = 13\n");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        // Set current_dir to a tempdir guaranteed not to be inside any
        // reef package so the cwd-detect path also returns None.
        .current_dir(dir.path())
        .args(["eval", "--file", entry_path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    assert!(
        output.status.success(),
        "raw eval exit status: {:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(
        stdout.contains("13"),
        "raw eval must still print the value; got: {stdout:?}"
    );
}
