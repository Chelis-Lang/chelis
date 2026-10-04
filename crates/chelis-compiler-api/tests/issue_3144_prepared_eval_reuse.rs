//! chelis#3144: evaluating many roots of one prepared program derives the
//! program's facts once, not once per root, and each root's evaluation keeps
//! its own state.
//!
//! `chelis test` prepares a test file once and evaluates each test against
//! it. Every evaluation used to rebuild the facts that depend on the program
//! alone: the root manifest and its realizability inputs, the DAG's key-rule
//! and sharing verdicts, and the host-lowering facts such as every
//! definition's effect row. A file's test time therefore grew with the
//! square of its test count. These tests count that work rather than time
//! it. The counters are process-wide, and nextest runs each test in its own
//! process.
#![allow(deprecated)] // `prepare_eval` is the no-library path `chelis test` still takes

use std::fs;
use std::path::{Path, PathBuf};

use chelis_compiler_api::compiler::{eval_program_fact_derivations, prepare_eval};
use chelis_compiler_api::schema::{EvalRequest, EvalResult, SourceKind};
use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context, prepare_eval_in_context};
use chelis_ir::host::def_effect_row_derivations;
use std::collections::BTreeMap;
use tempfile::TempDir;

/// A program whose `roots` values each call one effectful helper, so every
/// evaluation reaches the host lowering of a definition and reads its effect
/// row.
fn source(roots: usize) -> String {
    let mut source = String::from(
        "module App.Eval\n\
         def step(x: i64) -> i64 ! { IO } = {\n  _ = print(to_string(x))\n  add(x, 1i64)\n}\n",
    );
    for root in 0..roots {
        source.push_str(&format!("r{root} = step({root}i64)\n"));
    }
    source
}

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

/// The work counters' growth while `evaluate` runs.
fn work_during(evaluate: impl FnOnce()) -> (u64, u64) {
    let (facts, rows) = (
        eval_program_fact_derivations(),
        def_effect_row_derivations(),
    );
    evaluate();
    (
        eval_program_fact_derivations() - facts,
        def_effect_row_derivations() - rows,
    )
}

/// One root's printed value and transcript.
fn observed(result: &EvalResult, root: &str) -> (Option<String>, Vec<String>) {
    let display = result
        .roots
        .iter()
        .find(|evaluated| evaluated.name.as_deref() == Some(root))
        .and_then(|evaluated| evaluated.display.clone());
    (display, result.transcript.clone())
}

#[test]
fn a_prepared_program_derives_its_facts_once_for_all_roots() {
    for roots in [1, 8] {
        let prepared = prepare_eval(request(&source(roots))).expect("prepare");
        let (facts, rows) = work_during(|| {
            for root in 0..roots {
                prepared
                    .eval_root(BTreeMap::new(), &format!("r{root}"))
                    .expect("evaluate");
            }
        });
        assert_eq!(
            (facts, rows),
            (1, 1),
            "{roots} roots: the evaluation facts and the effect rows are each derived once"
        );
    }
}

/// The twin: sharing the facts shares no evaluation state. Each root of one
/// prepared program reports exactly what a fresh compile of the same program
/// reports for it, including a transcript holding only its own output.
#[test]
fn roots_of_one_prepared_program_keep_independent_state() {
    let source = source(3);
    let prepared = prepare_eval(request(&source)).expect("prepare");
    for root in 0..3 {
        let name = format!("r{root}");
        let shared = prepared
            .eval_root(BTreeMap::new(), &name)
            .expect("evaluate");
        let fresh = prepare_eval(request(&source))
            .expect("prepare")
            .eval_root(BTreeMap::new(), &name)
            .expect("evaluate");
        assert_eq!(observed(&shared, &name), observed(&fresh, &name), "{name}");
        assert_eq!(
            observed(&shared, &name),
            (Some((root + 1).to_string()), vec![root.to_string()]),
            "{name} prints only its own argument and returns its successor"
        );
    }
}

/// The library path `chelis test` takes inside a reef package: the prepared
/// program composes the library with the test file once.
#[test]
fn a_program_prepared_in_context_derives_its_facts_once_for_all_roots() {
    let (_dir, root) = library_fixture();
    let context = compile_reef_context(
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("context");
    for roots in [1, 8] {
        let source = source(roots).replace(
            "module App.Eval\n",
            "module App.Eval\nimport Mylib.Math (double)\n",
        ) + "doubled = double(21i64)\n";
        let prepared = prepare_eval_in_context(&context, &source).expect("prepare");
        let (facts, rows) = work_during(|| {
            for root in 0..roots {
                let name = format!("r{root}");
                let result = prepared
                    .eval_root(BTreeMap::new(), &name)
                    .expect("evaluate");
                assert_eq!(
                    observed(&result, &name),
                    (Some((root + 1).to_string()), vec![root.to_string()]),
                    "{name} keeps its own state"
                );
            }
            let doubled = prepared
                .eval_root(BTreeMap::new(), "doubled")
                .expect("evaluate the library call");
            assert_eq!(observed(&doubled, "doubled").0.as_deref(), Some("42"));
        });
        assert_eq!(
            (facts, rows),
            (1, 1),
            "{roots} roots: the evaluation facts and the effect rows are each derived once"
        );
    }
}

/// A reef package `myapp` with one path dependency `mylib`.
fn library_fixture() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");
    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n"
        ),
    )
    .expect("write app reef.toml");
    fs::write(
        root.join("src/main.ch"),
        "module App.Main\n\ndef placeholder() -> i32 = cast(0, i32)\n",
    )
    .expect("write main.ch");
    fs::write(
        root.join("mylib/reef.toml"),
        format!(
            "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Mylib\"\n"
        ),
    )
    .expect("write mylib reef.toml");
    fs::write(
        root.join("mylib/src/math.ch"),
        "module Mylib.Math\nexport (double)\n\ndef double(x: i64) -> i64 = add(x, x)\n",
    )
    .expect("write math.ch");
    fs::write(
        root.join("reef.lock"),
        format!(
            "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n"
        ),
    )
    .expect("write reef.lock");
    (dir, root)
}
