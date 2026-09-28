//! Section 12's declaration-sharing rule (`verify_declaration_sharing`)
//! against `vmap` sub-contexts (#2413 round 1).
//!
//! A new-code def `g` whose body holds a potentially trapping `add`, mapped
//! with `vmap(g)(x)` at top level or in a taken runtime `if` arm, traps when
//! a row overflows and returns the doubled rows otherwise, in the in-context
//! DAG evaluator (`eval_in_context`, against a reef dependency context and
//! its decoded copy) and in the whole-program evaluator (`compiler::eval`).
//! These are the verdicts origin/main `7807ca4ff` gives, where the rule does
//! not exist. Every row is collected before the assertion, so one red row
//! never hides a sibling.
#![allow(deprecated)]

use chelis_compiler_api::compiler::{CompilerError, eval};
use chelis_compiler_api::context::CompiledContext;
use chelis_compiler_api::schema::{EvalRequest, EvalResult, ExecutionValue, SourceKind};
use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context, eval_in_context};
use std::collections::BTreeMap;

const ADD_OVERFLOW: &str = "numeric trap: overflow in add at i32";

const G: &str = "def g(v: tensor[i32]) -> tensor[i32] = add(v, v)\n";

/// A reef package `app` with a dependency `mylib`, compiled as a context,
/// with its decoded copy.
fn contexts() -> [CompiledContext; 2] {
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
        "mylib/src/math.ch",
        "module Mylib.Math\nexport (g)\n\ndef g(v: tensor[4, f32]) -> tensor[4, f32] = {\n  dead = \
         copy(v)\n  v\n}\n",
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

/// `out` maps `g` over `input`, directly or in the taken arm of a runtime
/// `if` inside `selected`.
fn program(input: &str, in_arm: bool) -> String {
    if in_arm {
        format!(
            "{G}def selected(x: tensor[4, i32]) -> tensor[4, i32] = {{\n  s = \
             tensor_to_scalar(sum(copy(x), 0i32))\n  if lt(0i32, s) then vmap(g)(x) else x\n}}\n\
             out = selected(to_tensor({input}))\n"
        )
    } else {
        format!("{G}out = vmap(g)(to_tensor({input}))\n")
    }
}

fn out_values(result: &EvalResult) -> Option<Vec<f64>> {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("out"))?;
    match &root.value {
        ExecutionValue::Tensor { value } => Some(
            (0..value.data.len())
                .map(|index| value.data.element_f64_lossy(index))
                .collect(),
        ),
        _ => None,
    }
}

#[derive(Default)]
struct Rows(Vec<String>);

impl Rows {
    fn check(
        &mut self,
        row: &str,
        outcome: Result<EvalResult, CompilerError>,
        expected: Result<&[f64], &str>,
    ) {
        match (outcome, expected) {
            (Ok(result), Ok(values)) => {
                if out_values(&result).as_deref() != Some(values) {
                    self.0
                        .push(format!("{row}: `out` = {:?}", out_values(&result)));
                }
            }
            (Err(error), Err(trap)) if error.errors.iter().any(|error| error.message == trap) => {}
            (Ok(result), Err(trap)) => self.0.push(format!(
                "{row}: expected `{trap}`, returned {:?}",
                out_values(&result)
            )),
            (Err(error), _) => self.0.push(format!(
                "{row}: {:?}",
                error.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
            )),
        }
    }
}

#[test]
fn a_vmapped_def_with_a_trapping_node_runs_in_context_and_whole_program() {
    // The arm's guard sums the row, so its overflowing input keeps the sum
    // in range and overflows only the doubling.
    let overflow = |in_arm: bool| {
        if in_arm {
            "[1073741824i32, 1073741823i32, 0i32, 0i32]"
        } else {
            "[1i32, 2147483647i32, 3i32, 4i32]"
        }
    };
    let fits = "[1i32, 2i32, 3i32, 4i32]";
    let doubled = [2.0, 4.0, 6.0, 8.0];
    let contexts = contexts();
    let mut rows = Rows::default();
    for in_arm in [false, true] {
        for (input, expected) in [
            (overflow(in_arm), Err(ADD_OVERFLOW)),
            (fits, Ok(&doubled[..])),
        ] {
            let source = program(input, in_arm);
            let shape = if in_arm { "arm" } else { "top level" };
            let verdict = if expected.is_ok() {
                "fits"
            } else {
                "overflows"
            };
            for (index, context) in contexts.iter().enumerate() {
                rows.check(
                    &format!("in context {index}, {shape}, {verdict}"),
                    eval_in_context(context, &source),
                    expected,
                );
            }
            rows.check(
                &format!("whole program, {shape}, {verdict}"),
                eval(EvalRequest {
                    source_kind: SourceKind::Surf,
                    source,
                    bindings: BTreeMap::new(),
                }),
                expected,
            );
        }
    }
    assert!(rows.0.is_empty(), "\n{}", rows.0.join("\n"));
}
