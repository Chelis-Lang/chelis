//! chelis#2588: an inlined body reads its free names in the scope it was
//! written in. Here that body is a library's, evaluated through the compiler
//! API for new code composed with the library.
//!
//! The library's `total` calls its private `helper`, which reads its private
//! `scale`, so `total(x)` is `2x` wherever it is called. New code that binds a
//! local, a parameter or its own top-level declaration under one of those
//! names must not change that: the library's names are the library's.
//! These are disposition locks, not regression tests: each case already gave
//! `2x` at the pre-fix base `d029224fd`; this file does not establish why.
//! They keep the scoping change from letting a library body resolve new
//! code's names.
use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{check_in_context, prepare_eval_in_context};
use chelis_compiler_api::schema::EvalResult;
use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
use serde_json::json;

const LIBRARY: &str = "scale = to_tensor([2.0f32])\n\
def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, scale)\n\
def total(x: tensor[1, f32]) -> tensor[1, f32] = helper(x)\n";

fn eval_root(client: &str, root: &str) -> EvalResult {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("reef.toml"),
        format!(
            "[package]\nname = \"library-scope\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
        ),
    )
    .unwrap();
    std::fs::write(
        directory.path().join("src/values.ch"),
        format!("module Probe.Values\nexport (total)\n{LIBRARY}"),
    )
    .unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let client = format!("module Probe.Client\nimport Probe.Values (total)\n{client}");
    let checked = check_in_context(&context, &client).unwrap();
    assert_eq!(checked.score.get(), 1.0, "{checked:?}");
    assert!(checked.errors.is_empty(), "{checked:?}");
    prepare_eval_in_context(&context, &client)
        .unwrap()
        .eval_root(BTreeMap::new(), root)
        .unwrap()
}

fn assert_f32(result: &EvalResult, root: &str, bits: &str) {
    let value = result
        .roots
        .iter()
        .find(|entry| entry.name.as_deref() == Some(root))
        .unwrap_or_else(|| panic!("no `{root}` in {result:?}"));
    assert_eq!(
        serde_json::to_value(&value.value).unwrap(),
        json!({"type":"tensor","value":{"shape":[1],"data":{"dtype":"f32","bits":[bits]}}}),
        "{root}"
    );
}

/// 2.0f32, the library's `total(1)`.
const TWO: &str = "40000000";

#[test]
fn a_new_code_local_does_not_capture_a_library_value() {
    let result = eval_root(
        "def main() -> tensor[1, f32] = {\n  scale = to_tensor([10.0f32])\n  total(to_tensor([1.0f32]))\n}\n",
        "main",
    );
    assert_f32(&result, "main", TWO);
}

#[test]
fn a_new_code_parameter_does_not_capture_a_library_value() {
    // total(100) = 200.0f32: the parameter reaches the library only as the
    // argument, never as the library's `scale`.
    let result = eval_root(
        "def main(scale: tensor[1, f32]) -> tensor[1, f32] = total(scale)\nout = main(to_tensor([100.0f32]))\n",
        "out",
    );
    assert_f32(&result, "out", "43480000");
}

#[test]
fn a_new_code_local_function_does_not_capture_a_library_function() {
    // total(1) + ident(0) = 2.0f32.
    let result = eval_root(
        "def ident(x: tensor[1, f32]) -> tensor[1, f32] = x\n\
def main() -> tensor[1, f32] = {\n  helper = ident\n  add(total(to_tensor([1.0f32])), helper(to_tensor([0.0f32])))\n}\n",
        "main",
    );
    assert_f32(&result, "main", TWO);
}

#[test]
fn new_code_declarations_do_not_capture_library_names() {
    let result = eval_root(
        "scale = to_tensor([100.0f32])\n\
def helper(x: tensor[1, f32]) -> tensor[1, f32] = x\n\
def main() -> tensor[1, f32] = total(to_tensor([1.0f32]))\n",
        "main",
    );
    assert_f32(&result, "main", TWO);
}
