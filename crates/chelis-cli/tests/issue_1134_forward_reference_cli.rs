//! chelis#1134 / [04-INF-4]: public commands reject eager top-level value
//! forward references before evaluation or lowering.

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn check_report(name: &str, extension: &str, source: &str) -> (bool, serde_json::Value) {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join(format!("{name}.{extension}"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("UTF-8 fixture path")])
        .output()
        .expect("chelis check must run");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "chelis check must emit JSON: {error}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status.success(), report)
}

fn assert_clean_report(name: &str, extension: &str, source: &str) {
    let (success, report) = check_report(name, extension, source);
    assert!(success, "{name}: expected check success: {report:#}");
    assert_eq!(report["score"].as_f64(), Some(1.0), "{name}: {report:#}");
    assert_eq!(report["errors"].as_array().map(Vec::len), Some(0));
    assert_eq!(report["unresolved_names"].as_array().map(Vec::len), Some(0));
}

fn assert_failed_report(name: &str, extension: &str, source: &str, expected_kind: &str) {
    let (success, report) = check_report(name, extension, source);
    assert!(!success, "{name}: invalid program exited successfully");
    assert!(
        report["score"].as_f64().is_some_and(|score| score < 1.0),
        "{name}: invalid program reported perfect fitness: {report:#}"
    );
    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        errors.iter().any(|error| error["kind"] == expected_kind),
        "{name}: expected {expected_kind}: {report:#}"
    );
}

fn assert_only_cycle_report(name: &str, source: &str) {
    let (success, report) = check_report(name, "ch", source);
    assert!(!success, "{name}: initialization cycle exited successfully");
    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        errors.iter().any(|error| error["kind"] == "CycleDetected"),
        "{name}: expected CycleDetected: {report:#}"
    );
    assert!(
        errors.iter().all(|error| error["kind"] == "CycleDetected"),
        "{name}: component co-inference leaked a non-cycle diagnostic: {report:#}"
    );
}

fn assert_check_eval_build_cycle(name: &str, source: &str) {
    assert_only_cycle_report(name, source);
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join(format!("{name}.ch"));
    write_file(&path, source);
    let path = path.to_str().expect("UTF-8 fixture path");
    for (command, arguments) in [
        ("eval", vec!["eval", "--file", path]),
        ("build", vec!["build", path, "--target", "c"]),
    ] {
        let output = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(arguments)
            .output()
            .unwrap_or_else(|error| panic!("chelis {command} must run: {error}"));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success() && stderr.contains("binding cycle:"),
            "{name}: {command} must reject before execution/lowering; stdout={} stderr={stderr}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert!(
            !stderr.contains("unbound variable"),
            "{name}: {command} leaked scheduling-order UnboundVariable: {stderr}"
        );
    }
}

#[test]
fn check_rejects_forward_values_on_deep_and_surf_surfaces() {
    assert_failed_report(
        "forward_value",
        "dp",
        "(def {} use_base (var {} base))\n\n\
         (def {} base (lit {type: (t-prim {} i32)} 7))\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "forward_value_surf",
        "ch",
        "use_base = base\n\
         base = 7\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "later_external_input",
        "ch",
        "module ExternalCli\n\
         def capture() -> i32 = x\n\
         x = (x : i32)\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "later_declared_external_input",
        "ch",
        "module DeclaredExternalCli\n\
         def capture() -> i32 = x\n\
         x: i32 = x\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "later_declared_value",
        "ch",
        "module DeclaredValueCli\n\
         def capture() -> i32 = value\n\
         value: i32 = 7\n",
        "UnboundVariable",
    );
}

#[test]
fn check_accepts_backward_values_module_helpers_and_external_inputs() {
    assert_clean_report(
        "backward_value",
        "dp",
        "(def {} base (lit {type: (t-prim {} i32)} 7))\n\n\
         (def {} use_base (var {} base))\n",
    );
    assert_clean_report(
        "forward_helper_module",
        "ch",
        "module ForwardCli\n\
         def use(x: f32) -> f32 = identity(x)\n\
         def identity(x) = x\n",
    );
    assert_clean_report("external_input", "ch", "x = (x : tensor[4, f32])\n");
    assert_clean_report("declared_external_input", "ch", "x: tensor[4, f32] = x\n");
}

#[test]
fn check_rejects_missing_local_forward_and_cycle_controls_honestly() {
    assert_failed_report(
        "missing_name",
        "dp",
        "(def {} use_missing (var {} missing))\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "local_forward",
        "dp",
        "(def {} local_forward\n\
           (let {}\n\
             (bind {} x (var {} y) y (lit {type: (t-prim {} i32)} 1))\n\
             (var {} x)))\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "value_cycle",
        "dp",
        "(def {} first (var {} second))\n\n\
         (def {} second (var {} first))\n",
        "CycleDetected",
    );
}

#[test]
fn eval_and_build_reject_forward_values_before_execution_or_lowering() {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("forward_value.ch");
    write_file(
        &path,
        "def capture() -> List[i64] = later\n\
         later = [1i64, 2i64]\n",
    );
    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 fixture path")])
        .output()
        .expect("chelis eval must run");
    let eval_stderr = String::from_utf8_lossy(&eval.stderr);
    assert!(
        !eval.status.success() && eval_stderr.contains("unbound variable: later"),
        "eval must reject the forward value as unbound; stdout={} stderr={eval_stderr}",
        String::from_utf8_lossy(&eval.stdout)
    );

    let build = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().expect("UTF-8 fixture path"),
            "--target",
            "c",
        ])
        .output()
        .expect("chelis build must run");
    let build_stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        !build.status.success() && build_stderr.contains("unbound variable: later"),
        "C build must reject the forward value as unbound; stdout={} stderr={build_stderr}",
        String::from_utf8_lossy(&build.stdout)
    );
}

#[test]
fn check_eval_and_build_reject_higher_order_lambda_cycles() {
    for (name, source) in [
        (
            "returned_lambda_annotated",
            "module ReturnedLambdaAnnotatedCli\n\n\
             result: i32 = (make_reader())(1)\n\n\
             def make_reader() = fn (x: i32) -> read_result(x)\n\n\
             def read_result(n: i32) -> i32 = add(n, result)\n",
        ),
        (
            "returned_lambda_unannotated",
            "module ReturnedLambdaInferredCli\n\n\
             result = (make_reader())(1)\n\n\
             def make_reader() = fn (x: i32) -> read_result(x)\n\n\
             def read_result(n: i32) -> i32 = add(n, result)\n",
        ),
        (
            "stored_lambda_annotated",
            "module StoredLambdaAnnotatedCli\n\n\
             stored = fn (x: i32) -> read_result(x)\n\n\
             result: i32 = stored(1)\n\n\
             def read_result(n: i32) -> i32 = add(n, result)\n",
        ),
        (
            "stored_lambda_unannotated",
            "module StoredLambdaInferredCli\n\n\
             stored = fn (x: i32) -> read_result(x)\n\n\
             result = stored(1)\n\n\
             def read_result(n: i32) -> i32 = add(n, result)\n",
        ),
    ] {
        assert_check_eval_build_cycle(name, source);
    }
}

#[test]
fn eval_and_build_reject_a_later_external_input_before_lowering() {
    let directory = tempdir().expect("tempdir");
    for (name, source) in [
        (
            "later_ascribed_external_input",
            "module ExternalCli\n\
             def capture() -> i32 = x\n\
             x = (x : i32)\n",
        ),
        (
            "later_declared_external_input",
            "module DeclaredExternalCli\n\
             def capture() -> i32 = x\n\
             x: i32 = x\n",
        ),
    ] {
        let path = directory.path().join(format!("{name}.ch"));
        write_file(&path, source);
        for command in [
            vec!["eval", "--file", path.to_str().expect("UTF-8 fixture path")],
            vec![
                "build",
                path.to_str().expect("UTF-8 fixture path"),
                "--target",
                "c",
            ],
        ] {
            let output = Command::cargo_bin("chelis")
                .expect("chelis binary")
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(command)
                .output()
                .expect("chelis command must run");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !output.status.success() && stderr.contains("unbound variable: x"),
                "command must reject the later external input; stdout={} stderr={stderr}",
                String::from_utf8_lossy(&output.stdout)
            );
        }
    }
}

/// The public-surface half of the [04-INF-4] ordering matrix.
///
/// A module whose first declaration is a function makes the body-inference
/// schedule hoist every module function toward that ordinal. A later function
/// then reads a value declared between them, which is a backward reference the
/// atom requires to resolve, so the schedule must defer that reader past the
/// value. `chelis check` is the surface a user meets, so it carries the same
/// interleaving cases as the library oracle: both value spellings, the
/// header-less spelling, and the forward control that must still reject with
/// the reader sitting in the same hoisted position.
#[test]
fn check_accepts_a_backward_value_read_across_a_hoisted_module_function() {
    assert_clean_report(
        "interleaved_backward_declared",
        "ch",
        "module InterleavedBackwardDeclared\n\n\
         def anchor() -> i32 = 1\n\n\
         carried: i32 = 7\n\n\
         def reader() -> i32 = carried\n",
    );
    assert_clean_report(
        "interleaved_backward_unannotated",
        "ch",
        "module InterleavedBackwardUnannotated\n\n\
         def anchor() -> i32 = 1\n\n\
         carried = 7\n\n\
         def reader() -> i32 = carried\n",
    );
    assert_clean_report(
        "interleaved_backward_computed",
        "ch",
        "module InterleavedBackwardComputed\n\n\
         def seed() -> i32 = 3\n\n\
         def anchor() -> i32 = 1\n\n\
         carried = seed()\n\n\
         def reader() -> i32 = carried\n",
    );
    assert_clean_report(
        "interleaved_backward_through_helper",
        "ch",
        "module InterleavedBackwardHelper\n\n\
         def caller() -> i32 = helper()\n\n\
         carried = 5\n\n\
         def helper() -> i32 = carried\n",
    );
    assert_failed_report(
        "interleaved_forward_declared",
        "ch",
        "module InterleavedForwardDeclared\n\n\
         def anchor() -> i32 = 1\n\n\
         def reader() -> i32 = carried\n\n\
         carried: i32 = 7\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "interleaved_forward_unannotated",
        "ch",
        "module InterleavedForwardUnannotated\n\n\
         def anchor() -> i32 = 1\n\n\
         def reader() -> i32 = carried\n\n\
         carried = 7\n",
        "UnboundVariable",
    );
}

/// chelis#1485 / [04-INF-7] at the public surface. Full-reference components
/// reject identically regardless of whether the reader has a complete header,
/// is reached through a lambda, or has no `defsig`. Component co-inference may
/// use provisional bindings, but no `UnboundVariable` may escape.
#[test]
fn a_value_that_names_a_function_reading_it_back_is_a_recorded_stall() {
    assert_only_cycle_report(
        "mirror_escape_signed",
        "module MirrorEscape\n\n\
         def anchor() -> i32 = 1\n\n\
         carried = wrap(f)\n\n\
         def wrap(g) = g\n\n\
         def f(n: i32) -> i32 = if (n <= 0) then 0 else carried((n - 1))\n",
    );
    assert_only_cycle_report(
        "mirror_escape_lambda",
        "module PickEscape\n\n\
         def anchor() -> i32 = 1\n\n\
         carried = pick(fn (x: i32) -> f(x))\n\n\
         def pick(g) = 5\n\n\
         def f(n: i32) -> i32 = add(n, carried)\n",
    );
    assert_only_cycle_report(
        "mirror_escape_defsig_less",
        "module WrapEscape\n\n\
         def anchor() -> i32 = 1\n\n\
         carried = wrap(g)\n\n\
         def wrap(h) = h\n\n\
         def g(n) = if (n <= 0) then 0 else carried((n - 1))\n",
    );
}

/// The reader-first layout of the partial-header program: `r` sits BELOW the
/// hoist floor, where the mirror edge never reached it.
const READER_FIRST_PARTIAL_HEADER: &str = "module PartialHeaderReaderFirst\n\n\
     r: f32 = f(2)\n\n\
     v: i32 = 1\n\n\
     def anchor() -> i32 = 1\n\n\
     def f(n: i32) = add(v, n)\n";

/// A partial header must not be instantiated before its body narrows it
/// (chelis#1486 / [04-INF-5]): the reader's mismatched ascription must reject
/// at the public surface in both layouts. The first was already deferred by
/// the mirror edge; the second sits below the hoist floor, where no mirror
/// edge applies and only the hole edge defers it.
///
/// The first row is a disposition lock and was green before this change. The
/// second is a regression test: the reader-first layout scored 1.0 and
/// compiled a `f32` binding out of an `i32` body.
#[test]
fn check_rejects_a_mismatched_read_of_a_partial_header_deferred_by_a_barrier() {
    assert_failed_report(
        "partial_header",
        "ch",
        "module PartialHeader\n\n\
         def anchor() -> i32 = 1\n\n\
         r: f32 = f(2)\n\n\
         v: i32 = 1\n\n\
         def f(n: i32) = add(v, n)\n",
        "TypeMismatch",
    );
    assert_failed_report(
        "partial_header_reader_first",
        "ch",
        READER_FIRST_PARTIAL_HEADER,
        "TypeMismatch",
    );
}

/// The same reader-first program at the two lanes that run code. `chelis
/// check` reports a score; `eval` and `build` must refuse to execute or lower
/// it, which is the half that makes chelis#1486 a compiled wrong answer rather
/// than a fitness inaccuracy.
///
/// Regression test: before this change both commands succeeded, `eval` printed
/// `r = 3` and the emitted C printed `3.0` for a binding declared `f32`.
#[test]
fn eval_and_build_reject_a_reader_first_partial_header() {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("partial_header_reader_first.ch");
    write_file(&path, READER_FIRST_PARTIAL_HEADER);
    let fixture = path.to_str().expect("UTF-8 fixture path");
    for command in [
        vec!["eval", "--file", fixture],
        vec!["build", fixture, "--target", "c"],
    ] {
        let label = command[0];
        let output = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(&command)
            .output()
            .expect("chelis command must run");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success()
                && stderr.contains("def 'r' body doesn't match declared signature"),
            "{label} must reject the reader-first partial header; stdout={} stderr={stderr}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

/// The public-surface half of the recursive-component ordering matrix.
///
/// A mutually recursive pair is inferred as one unit, so the schedule moves
/// the whole component past a value one member reads. The stamped spelling
/// would pass at `chelis check` even without that, because the serialized-IR
/// ingress can read the value's type off the value's own body; the header-less
/// spelling below is the one that proves the component moved.
#[test]
fn check_accepts_a_stamped_backward_value_read_by_a_recursive_component() {
    assert_clean_report(
        "scc_straddle_stamped_wrapped",
        "ch",
        "module SccStraddleStampedWrapped\n\n\
         def ping(n: i32) -> i32 = if (n <= 0) then 0 else pong((n - 1))\n\n\
         carried = 7\n\n\
         def pong(n: i32) -> i32 = if (n <= 0) then carried else ping((n - 1))\n",
    );
    assert_clean_report(
        "scc_straddle_stamped_bare",
        "ch",
        "def ping(n: i32) -> i32 = if (n <= 0) then 0 else pong((n - 1))\n\n\
         carried = 7\n\n\
         def pong(n: i32) -> i32 = if (n <= 0) then carried else ping((n - 1))\n",
    );
    // The forward control must still reject with the reader inside the
    // component, or the acceptance above proves nothing about ordering.
    assert_failed_report(
        "scc_straddle_forward",
        "ch",
        "module SccStraddleForward\n\n\
         def ping(n: i32) -> i32 = if (n <= 0) then 0 else pong((n - 1))\n\n\
         def pong(n: i32) -> i32 = if (n <= 0) then carried else ping((n - 1))\n\n\
         carried = 7\n",
        "UnboundVariable",
    );
}

/// A header-less value read by a recursive component. No ingress can read
/// `carried`'s type off a header, so this passes only if the schedule really
/// infers the value before the whole component, wrapped and bare alike.
#[test]
fn check_accepts_a_header_less_value_read_by_a_recursive_component() {
    assert_clean_report(
        "scc_straddle_computed_wrapped",
        "ch",
        "module SccStraddleComputedWrapped\n\n\
         def seed() -> i32 = 3\n\n\
         def ping(n: i32) -> i32 = if (n <= 0) then 0 else pong((n - 1))\n\n\
         carried = seed()\n\n\
         def pong(n: i32) -> i32 = if (n <= 0) then carried else ping((n - 1))\n",
    );
    assert_clean_report(
        "scc_straddle_computed_bare",
        "ch",
        "def seed() -> i32 = 3\n\n\
         def ping(n: i32) -> i32 = if (n <= 0) then 0 else pong((n - 1))\n\n\
         carried = seed()\n\n\
         def pong(n: i32) -> i32 = if (n <= 0) then carried else ping((n - 1))\n",
    );
}
