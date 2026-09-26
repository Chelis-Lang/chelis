//! chelis#2413 round 1: a key-carrying root that is only observed behaves
//! the same in every lane. [04-LIN-6] makes the observation the key's one
//! use; the DAG evaluator prints the key and compiled C prints the same text
//! ([05-OBS-2]) through the runtime's `chelis_string_from_key`, never
//! building a program that traps when it prints.
//!
//! A tuple of keys built directly by `split_key`, as a top-level value or as
//! a function's result, is a valid key producer for the lowering and the
//! verifier, so it never breaks evaluation of the rest of the file.
//!
//! Every expected key is computed by `briefs/switch-design-probes/key_ref.py`
//! and every draw by `briefs/keys-b-slice2-probes/slice2_ref.py`; no expected
//! value is computed by compiler code.
#![allow(deprecated)]
mod ownership_support;

use chelis_compiler_api::compiler::{eval, eval_selected};
use chelis_compiler_api::context::CompiledContext;
use chelis_compiler_api::schema::{EvalRequest, EvalResult, SourceKind};
use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context, eval_in_context};
use std::collections::BTreeMap;

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

fn lines(result: &EvalResult) -> Vec<String> {
    result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}",
                root.name.as_deref().expect("a named root"),
                root.display.as_deref().expect("an in-process display")
            )
        })
        .collect()
}

/// Every root `chelis eval` reports, in order.
fn eval_lines(source: &str) -> Vec<String> {
    lines(&eval(request(source)).unwrap_or_else(|error| panic!("eval: {error:?}\n{source}")))
}

/// The lines the native C program prints for its roots, with the ownership
/// ledger balanced.
fn c_lines(source: &str, label: &str) -> Vec<String> {
    let generated = ownership_support::emit(source, label);
    let (summary, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
    stdout.lines().map(str::to_string).collect()
}

fn both_lanes(source: &str, label: &str, expected: &[&str]) {
    assert_eq!(eval_lines(source), expected, "eval: {label}\n{source}");
    assert_eq!(c_lines(source, label), expected, "C: {label}\n{source}");
}

/// Evidentiary status: REGRESSION TEST. At `b47fdd7d3` C refused the scalar
/// roots at build ("unresolved host type `Key`"), the key tensor and tuple
/// roots built and aborted in `chelis_scalar_from_bits`, and eval refused
/// the directly built pairs ("produces a key, but only a key operation, a
/// join or a Load produces one").
#[test]
fn an_observed_key_root_prints_the_same_key_in_eval_and_c() {
    let holder = "type Holder =\n  | Holder { k: key, n: i64 }\n";
    let cases: [(&str, String, Vec<&str>); 7] = [
        (
            "scalar_value",
            "first = fold_in(key_from_seed(1i64), 1i64)\n".to_string(),
            vec!["first = key(1c3be871ed9d079c)"],
        ),
        (
            "scalar_result",
            "def main() -> key = key_from_seed(1i64)\n".to_string(),
            vec!["main = key(0000000000000001)"],
        ),
        (
            "pair_value",
            "pair = split_key(key_from_seed(1i64))\n".to_string(),
            vec![
                "pair.0 = key(187d9c3f80640697)",
                "pair.1 = key(de40f57be8762242)",
            ],
        ),
        (
            "pair_result",
            "def main() -> (key, key) = split_key(key_from_seed(4i64))\n".to_string(),
            vec![
                "main.0 = key(ebc2c220945ec8df)",
                "main.1 = key(5283e7f23f901300)",
            ],
        ),
        (
            "key_tensor",
            "ks = split_keys(key_from_seed(1i64), 2i64)\n".to_string(),
            vec!["ks = tensor(shape=[2], data=[key(f2e0ed7d61bc7ab1), key(1c3be871ed9d079c)])"],
        ),
        (
            "key_list",
            "ks = [key_from_seed(1i64), key_from_seed(2i64)]\n".to_string(),
            vec!["ks = [key(0000000000000001), key(0000000000000002)]"],
        ),
        (
            "record_field",
            format!("{holder}h = Holder {{ k: key_from_seed(5i64), n: 2i64 }}\n"),
            vec!["h.k = key(0000000000000005)", "h.n = 2"],
        ),
    ];
    for (label, source, expected) in cases {
        both_lanes(&source, label, &expected);
    }
}

const PAIR: &str = "def pair(k: key) -> (key, key) = split_key(k)\n";

/// 1b's P1-3: a function returning `split_key`'s pair directly lowers to a
/// graph the verifier accepts, uncalled or called, in eval, in a selection
/// of an unrelated root, and in C.
///
/// Evidentiary status: REGRESSION TEST. At `b47fdd7d3` every eval row
/// failed with "node 3 of `pair` produces a key, but only a key operation,
/// a join or a Load produces one"; C accepted it.
#[test]
fn a_function_returning_a_split_pair_never_breaks_the_file() {
    let uncalled = format!("{PAIR}out = to_tensor([1.0f32])\n");
    both_lanes(
        &uncalled,
        "pair_uncalled",
        &["out = tensor(shape=[1], data=[1.0])"],
    );
    let selected = eval_selected(request(&uncalled), &["out".into()])
        .unwrap_or_else(|error| panic!("selection: {error:?}"));
    assert_eq!(lines(&selected), ["out = tensor(shape=[1], data=[1.0])"]);
    // `dropout(split(key(4)).1, [1, 2, 3, 4], 0.5)`.
    let called = format!(
        "{PAIR}def main() -> tensor[4, f32] = {{\n  (a, b) = pair(key_from_seed(4i64))\n  \
         dropout(b, to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), 0.5f32)\n}}\n"
    );
    both_lanes(
        &called,
        "pair_called",
        &["main = tensor(shape=[4], data=[2.0, 0.0, 0.0, 8.0])"],
    );
}

/// A reef package whose dependency `mylib` exports `pair`, compiled as a
/// context, with its decoded copy: the library graph crosses the wire codec,
/// whose key rules read the same verifier.
fn pair_contexts() -> [CompiledContext; 2] {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("app");
    let write = |path: &str, contents: &str| {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    };
    write(
        "reef.toml",
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\n\
             module_prefix = \"App\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n"
        ),
    );
    write(
        "src/main.ch",
        "module App.Main\n\ndef placeholder() -> i32 = 0i32\n",
    );
    write(
        "mylib/reef.toml",
        &format!(
            "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\n\
             module_prefix = \"Mylib\"\n"
        ),
    );
    write(
        "mylib/src/keys.ch",
        &format!("module Mylib.Keys\nexport (pair)\n\n{PAIR}"),
    );
    write(
        "reef.lock",
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\n\n[[dependencies]]\n\
             name = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\n\
             archive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\n\
             kind = \"path\"\npath = \"./mylib\"\n"
        ),
    );
    let context = compile_reef_context(directory.path(), &root)
        .unwrap_or_else(|error| panic!("context: {error:?}"));
    let decoded = CompiledContext::decode(&context.encode().unwrap()).unwrap();
    [context, decoded]
}

/// Evidentiary status: REGRESSION TEST for the context build, which lowered
/// the library's `pair` into a graph the key verifier refused.
#[test]
fn a_library_function_returning_a_split_pair_serves_its_callers() {
    let client = "module App.Eval\nimport Mylib.Keys (pair)\n\ndef main() -> tensor[4, f32] = {\n  \
                  (a, b) = pair(key_from_seed(4i64))\n  dropout(b, to_tensor([1.0f32, 2.0f32, \
                  3.0f32, 4.0f32]), 0.5f32)\n}\n";
    for context in pair_contexts() {
        let result = eval_in_context(&context, client)
            .unwrap_or_else(|error| panic!("in context: {error:?}"));
        assert_eq!(
            lines(&result),
            ["main = tensor(shape=[4], data=[2.0, 0.0, 0.0, 8.0])"]
        );
    }
}
