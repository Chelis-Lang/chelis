//! A recursive user call must fail through the eval diagnostic channel before
//! it exhausts the native stack, preserving effects that completed before it.
//! The depth it reaches first is the same whichever lane evaluates it.

#![allow(deprecated)] // `prepare_eval` is the no-library path `chelis test` still takes

use std::collections::BTreeMap;
use std::process::Command;

use chelis_compiler_api::compiler::{eval, prepare_eval};
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

/// The segment both lanes evaluate on in the chelis#3742 receipt, and the
/// native stack of the thread standing in for a `chelis test` worker. The
/// worker thread is far smaller than the segment, so a lane that evaluated on
/// its caller's thread would reach a much shallower depth.
const LANE_SEGMENT_BYTES: usize = 4 * 1024 * 1024;
const WORKER_THREAD_BYTES: usize = 1024 * 1024;

fn request(depth: usize) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source(depth),
        bindings: BTreeMap::new(),
    }
}

/// `chelis eval`'s shape: the whole command runs on the grown segment.
fn eval_lane_passes(depth: usize) -> bool {
    set_grow_segment_bytes_for_test(LANE_SEGMENT_BYTES);
    let passed = run_on_grown_stack(|| eval(request(depth))).is_ok();
    reset_grow_segment_bytes_for_test();
    passed
}

/// `chelis test`'s shape: the program is prepared once, and each root is
/// evaluated on a worker thread of its own.
fn worker_lane_passes(depth: usize) -> bool {
    let prepared = run_on_grown_stack(|| prepare_eval(request(depth))).expect("prepare");
    std::thread::Builder::new()
        .name("issue-3742-worker".to_string())
        .stack_size(WORKER_THREAD_BYTES)
        .spawn(move || {
            set_grow_segment_bytes_for_test(LANE_SEGMENT_BYTES);
            let passed = prepared.eval_root(BTreeMap::new(), "out").is_ok();
            reset_grow_segment_bytes_for_test();
            passed
        })
        .expect("worker thread starts")
        .join()
        .expect("worker thread returns")
}

/// The deepest recursion `passes` accepts: doubling to the first failure,
/// then bisecting. Every probe is an independent evaluation.
fn deepest_passing(passes: impl Fn(usize) -> bool) -> usize {
    assert!(passes(1), "a single call must evaluate");
    let mut low = 1;
    let mut high = 2;
    while passes(high) {
        low = high;
        high *= 2;
        assert!(high <= 1 << 20, "recursion never exhausted the segment");
    }
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if passes(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

/// chelis#3742: the depth a recursion reaches is a property of the evaluator,
/// not of the thread its caller runs on. `chelis eval` evaluated inside the
/// CLI's grown segment while a `chelis test` worker evaluated on its own
/// 32 MiB thread, so the test lane held about a tenth of eval's recursion and
/// a Nautilus test that passed under eval failed under test.
///
/// The receipt is the deepest passing depth of each lane on the same segment
/// size, from a worker thread a quarter of that size. Evaluation enters a
/// fresh segment in both lanes, so the two depths are equal exactly.
///
/// Evidentiary status: REGRESSION TEST, proven failing first. Without the
/// segment at the evaluator's entry the worker lane evaluated on its 1 MiB
/// thread and reached a much shallower depth than the eval lane.
#[test]
fn issue_3742_recursion_depth_is_the_same_in_the_eval_and_worker_lanes() {
    let eval_depth = deepest_passing(eval_lane_passes);
    let worker_depth = deepest_passing(worker_lane_passes);
    eprintln!("#3742 receipt: eval lane {eval_depth}, worker lane {worker_depth}");
    assert!(
        eval_depth >= 8,
        "#3742: a {LANE_SEGMENT_BYTES}-byte segment must hold more than a few calls, or \
         this receipt compares nothing; it held {eval_depth}"
    );
    assert_eq!(
        worker_depth, eval_depth,
        "#3742: the worker lane must reach the eval lane's recursion depth"
    );
}
