//! [04-INF-9]: a lexical key-builtin alias shadows a def of the same
//! spelling, so evaluation and the compiled C lane call the alias, both as a
//! direct call and as a named `map` callback (chelis#3484). The C lane
//! previously called the def and printed its result without any diagnostic.
//!
//! These are default-feature unit tests, so they run on every pull request,
//! not only with the ownership-ledger targets.

use crate::compiler::{compile, eval_selected};
use crate::schema::{CompileRequest, CompileTarget, EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::fs;

const SHADOWING: &str = "def mix(k: key, n: i64) -> key = k\n\
def seed(x: i64) -> key = key_from_seed(0i64)\n\
def main() = {\n  \
  mix: key -> i64 -> key = fold_in\n  \
  seed: i64 -> key = key_from_seed\n  \
  (mix(seed(7i64), -1i64), map(seed, [1i64, -1i64]))\n\
}\n";

/// `main.N = <display>` for each root of `main`, as evaluation prints it.
fn evaluated_lines(source: &str) -> Vec<String> {
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            bindings: BTreeMap::new(),
        },
        &["main".into()],
    )
    .unwrap_or_else(|error| panic!("evaluate: {error:?}"));
    result
        .roots
        .iter()
        .enumerate()
        .map(|(index, root)| {
            format!(
                "main.{index} = {}",
                root.display.as_deref().expect("evaluated root display")
            )
        })
        .collect()
}

/// The standard output of the compiled C program for `source`.
fn compiled_stdout(source: &str) -> String {
    let artifact = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some("agreement".into()),
    })
    .unwrap_or_else(|error| panic!("compile: {error:?}"));
    let dir = tempfile::tempdir().expect("temporary build directory");
    for file in &artifact.files {
        fs::write(dir.path().join(&file.path), &file.contents).expect("write generated file");
    }
    let staged = chelis_runtime_bundle::stage(dir.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    let source_path = dir.path().join("agreement.c");
    let binary = dir.path().join("agreement");
    let toolchain = chelis_backend_c::toolchain::test_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements::default(),
    );
    let built = chelis_backend_c::toolchain::tool_command(&toolchain.compiler)
        .arg("-I")
        .arg(dir.path())
        .args(chelis_backend_c::toolchain::link_args(
            &artifact.compile_flags,
            &[source_path.as_os_str(), staged.archive.as_os_str()],
            &artifact.link_flags,
            binary.as_os_str(),
        ))
        .output()
        .expect("run the C compiler");
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = std::process::Command::new(&binary)
        .output()
        .expect("run the compiled program");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).expect("utf-8 program output")
}

#[test]
fn key_alias_shadowing_a_def_agrees_in_eval_and_compiled_c() {
    let evaluated = evaluated_lines(SHADOWING);
    assert_eq!(
        evaluated,
        [
            "main.0 = key(45c80b557fb94ddb)",
            "main.1 = [key(0000000000000001), key(ffffffffffffffff)]",
        ],
        "evaluation calls the aliases: fold_in(7, -1) and key_from_seed of each item"
    );
    let compiled = compiled_stdout(SHADOWING);
    for line in &evaluated {
        assert!(
            compiled.lines().any(|printed| printed == line),
            "compiled C must print `{line}`:\n{compiled}"
        );
    }
}
