//! A recursive user call must fail through the eval diagnostic channel before
//! it exhausts the native stack, preserving effects that completed before it.

use std::collections::BTreeMap;
use std::process::Command;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use chelis_types::{
    reset_grow_segment_bytes_for_test, run_on_grown_stack, set_grow_segment_bytes_for_test,
};

const CHILD_ENV: &str = "CHELIS_2471_STACK_BUDGET_CHILD";
const SMALL_STACK_BYTES: usize = 2 * 1024 * 1024;

fn source(depth: usize) -> String {
    format!(
        "def count_down(k: i64) -> i64 = if eq(k, 0i64) then 0i64 else add(1i64, count_down(sub(k, 1i64)))\n\
         out = {{ _ = print(\"before\")\n count_down({depth}i64) }}\n"
    )
}

fn source_with_bindings(depth: usize, binding_count: usize) -> String {
    let mut bindings = String::from("  x0 = add(k, 0i64)\n");
    for index in 1..binding_count {
        bindings.push_str(&format!("  x{index} = add(x{}, 0i64)\n", index - 1));
    }
    format!(
        "def count_down(k: i64) -> i64 = {{\n{bindings}  if eq(x{last}, 0i64) then 0i64 else add(1i64, count_down(sub(x{last}, 1i64)))\n\
         }}\nout = {{ _ = print(\"before\")\n count_down({depth}i64) }}\n",
        last = binding_count - 1
    )
}

fn evaluate(depth: usize) -> chelis_compiler_api::schema::EvalResult {
    set_grow_segment_bytes_for_test(SMALL_STACK_BYTES);
    let result = run_on_grown_stack(|| {
        eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source(depth),
            bindings: BTreeMap::new(),
        })
    });
    reset_grow_segment_bytes_for_test();
    result.expect("shallow recursion must evaluate")
}

#[test]
fn shallow_user_recursion_succeeds_on_the_same_stack() {
    let result = evaluate(2);
    assert_eq!(result.transcript, ["before"]);
    assert_eq!(result.roots.len(), 1);
    assert_eq!(result.roots[0].name.as_deref(), Some("out"));
    assert_eq!(result.roots[0].display.as_deref(), Some("2"));
}

#[test]
fn original_1000_call_case_still_evaluates_on_the_cli_stack() {
    let result = run_on_grown_stack(|| {
        eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source(1_000),
            bindings: BTreeMap::new(),
        })
    })
    .expect("the reported 1,000-call case fits the ordinary CLI stack");
    assert_eq!(result.transcript, ["before"]);
    assert_eq!(result.roots[0].display.as_deref(), Some("1000"));
}

#[test]
fn wide_200_call_iteration_budget_evaluates_on_the_cli_stack() {
    let result = run_on_grown_stack(|| {
        eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source_with_bindings(200, 16),
            bindings: BTreeMap::new(),
        })
    })
    .expect("a 16-binding, 200-call iteration fits the ordinary CLI stack");
    assert_eq!(result.transcript, ["before"]);
    assert_eq!(result.roots[0].display.as_deref(), Some("200"));
}

#[test]
fn deep_recursion_child() {
    let Some(case) = std::env::var_os(CHILD_ENV) else {
        return;
    };
    set_grow_segment_bytes_for_test(SMALL_STACK_BYTES);
    let error = run_on_grown_stack(|| {
        eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: match case.to_str().expect("case") {
                "wide" => source_with_bindings(10_000, 16),
                "very_wide" => source_with_bindings(10_000, 128),
                _ => source(10_000),
            },
            bindings: BTreeMap::new(),
        })
    })
    .expect_err("recursive evaluation must report stack exhaustion");
    reset_grow_segment_bytes_for_test();
    assert_eq!(error.transcript, ["before"]);
    assert!(
        error.errors.iter().any(|diagnostic| {
            diagnostic.kind().as_str() == "eval_error"
                && diagnostic.message.contains("stack budget exhausted")
        }),
        "{error:?}"
    );
}

fn assert_deep_recursion_reports_diagnostic(case: &str) {
    let output = Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", "deep_recursion_child", "--nocapture"])
        .env(CHILD_ENV, case)
        .output()
        .expect("run recursion in a child process");
    assert!(
        output.status.success(),
        "recursive evaluation aborted ({:?}): stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn deep_user_recursion_returns_a_diagnostic_instead_of_aborting() {
    assert_deep_recursion_reports_diagnostic("simple");
}

#[test]
fn wide_recursive_body_returns_a_diagnostic_instead_of_aborting() {
    assert_deep_recursion_reports_diagnostic("wide");
}

#[test]
fn very_wide_recursive_body_returns_a_diagnostic_instead_of_aborting() {
    assert_deep_recursion_reports_diagnostic("very_wide");
}
