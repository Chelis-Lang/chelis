//! chelis#1334: genuine checker rejections retain their authored call site and
//! directional operands through both the JSON and textual CLI surfaces.

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn run(command: &str, source: &str) -> (bool, String, String) {
    let dir = tempdir().expect("temporary source directory");
    let source_path = dir.path().join("probe.ch");
    std::fs::write(&source_path, source).expect("write fixture");
    let output_path = dir.path().join("output");
    let mut process = Command::cargo_bin("chelis").expect("chelis CLI");
    process
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg(command)
        .arg(&source_path);
    if command == "build" {
        process.args(["--target", "c", "-o"]).arg(output_path);
    }
    let output = process.output().expect("execute chelis CLI");
    (
        output.status.success(),
        String::from_utf8(output.stdout).expect("UTF-8 stdout"),
        String::from_utf8(output.stderr).expect("UTF-8 stderr"),
    )
}

fn checked_error(source: &str, kind: &str, call: &str, expected: &str, got: &str) -> Value {
    let (success, stdout, _) = run("check", source);
    let report: Value = serde_json::from_str(&stdout).expect("check emits JSON");
    assert!(
        !success,
        "erroneous input must reject: {source:?}: {report}"
    );
    let errors = report["errors"].as_array().expect("diagnostics array");
    let callee = call.split_once('(').expect("call prefix").0;
    let error = errors
        .iter()
        .find(|error| {
            error["kind"] == kind
                && error["message"]
                    .as_str()
                    .is_some_and(|text| text.contains(callee))
        })
        .unwrap_or_else(|| panic!("expected {kind} for {callee} in {errors:?}"));
    let at = source.find(call).expect("authored call occurs in source");
    assert_eq!(
        error["span"]["offset"], at,
        "location must point to the call, not its enclosing def: {error}"
    );
    assert!(
        error["span_id"]
            .as_str()
            .is_some_and(|id| id.starts_with(&format!("surf:{at}.."))),
        "call identity: {error}"
    );
    assert_eq!(
        error["expected"], expected,
        "directional expectation: {error}"
    );
    assert_eq!(error["got"], got, "directional actual: {error}");
    let message = error["message"].as_str().unwrap();
    assert!(
        message.contains(expected) && message.contains(got),
        "human and machine operands disagree: {error}"
    );
    error.clone()
}

#[test]
fn specialized_application_arity_rejections_retain_source_and_actual_count() {
    for (source, callee, expected) in [
        ("out = pad(to_tensor([1.0f32]))\n", "pad(", "3 argument(s)"),
        (
            "out = stride(to_tensor([1.0f32]))\n",
            "stride(",
            "at least 2 arguments",
        ),
        (
            "out = reduce_window_sum(to_tensor([1.0f32]))\n",
            "reduce_window_sum(",
            "3 argument(s)",
        ),
    ] {
        let error = checked_error(source, "ArityMismatch", callee, expected, "1 argument(s)");
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .contains(callee.trim_end_matches('('))
        );
    }

    for source in [
        "out = pad(to_tensor([1.0f32]), [[0i64, 0i64]], 0.0f32)\n",
        "out = stride(to_tensor([1.0f32]), 1i64)\n",
        "out = reduce_window_sum(to_tensor([1.0f32]), [1i64], [1i64])\n",
    ] {
        let (ok, stdout, stderr) = run("check", source);
        let report: Value = serde_json::from_str(&stdout).expect("valid arity JSON");
        assert!(ok, "{source:?}: {stderr}: {report}");
        assert_eq!(report["errors"], serde_json::json!([]));
    }
}

#[test]
fn nested_callback_arity_names_the_first_argument_and_direction() {
    let bad = "out = fold(fn (acc: i64) -> acc, 0i64, [1i64])\n";
    let error = checked_error(bad, "ArityMismatch", "fold(", "2 parameters", "1 parameter");
    assert!(
        error["message"].as_str().unwrap().contains("argument 1"),
        "{error}"
    );

    let good = "out = fold(fn (acc: i64, item: i64) -> add(acc, item), 0i64, [1i64])\n";
    let (ok, stdout, stderr) = run("check", good);
    let report: Value = serde_json::from_str(&stdout).expect("valid callback JSON");
    assert!(ok, "{stderr}: {report}");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn reduction_rejections_name_the_axis_or_bool_tensor_requirement() {
    let source = "out = sum(to_tensor([1.0f32]), 2i32)\n";
    let error = checked_error(source, "DimensionMismatch", "sum(", "axis in -1..1", "2");
    assert!(
        error["message"].as_str().unwrap().contains("argument 2")
            && error["message"].as_str().unwrap().contains("axis 2"),
        "the rejected axis must be attributable: {error}"
    );
    let source = "out = count(to_tensor([1.0f32]), 0i32)\n";
    let error = checked_error(source, "PrecisionMismatch", "count(", "bool tensor", "f32");
    assert!(
        error["message"].as_str().unwrap().contains("argument 1"),
        "the rejected tensor must be attributable: {error}"
    );

    for source in [
        "out = sum(to_tensor([1.0f32]), 0i32)\n",
        "out = count(to_tensor([true, false]), 0i32)\n",
    ] {
        let (ok, stdout, stderr) = run("check", source);
        let report: Value = serde_json::from_str(&stdout).expect("valid reduction JSON");
        assert!(ok, "{source:?}: {stderr}: {report}");
        assert_eq!(report["errors"], serde_json::json!([]));
    }
}

#[test]
fn diagonal_rejects_a_repeated_axis_at_the_call_and_keeps_the_allowed_axis() {
    let bad = "out = diagonal(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]), 0i32, 0i32)\n";
    let error = checked_error(
        bad,
        "TypeMismatch",
        "diagonal(",
        "axis distinct from 0",
        "axis 0",
    );
    assert!(
        error["message"].as_str().unwrap().contains("argument 3"),
        "{error}"
    );

    let good = "out = diagonal(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]), 0i32, 1i32)\n";
    let (ok, stdout, stderr) = run("check", good);
    let report: Value = serde_json::from_str(&stdout).expect("valid diagonal JSON");
    assert!(ok, "{stderr}: {report}");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn rejected_tensor_conversion_and_copy_calls_locate_the_genuine_operand() {
    for (source, kind, call, expected, got) in [
        (
            "out = to_list(to_tensor([[1.0f32]]))\n",
            "TypeMismatch",
            "to_list(",
            "rank-1 tensor",
            "rank-2 tensor",
        ),
        (
            "out = to_list(sum(to_tensor([1.0f32, 2.0f32]), 0i32))\n",
            "TypeMismatch",
            "to_list(",
            "rank-1 tensor",
            "rank-0 tensor",
        ),
        (
            "out = tensor_to_scalar(to_tensor([1.0f32]))\n",
            "TypeMismatch",
            "tensor_to_scalar(",
            "rank-0 tensor",
            "rank-1 tensor",
        ),
        (
            "out = copy([1.0f32])\n",
            "TypeMismatch",
            "copy(",
            "tensor",
            "List f32",
        ),
        (
            "def bad(x: List[f32]) -> f64 = cast(x, f64)\n",
            "CastNonTensor",
            "cast(",
            "tensor or numeric/bool scalar",
            "List f32",
        ),
    ] {
        let error = checked_error(source, kind, call, expected, got);
        assert!(
            error["message"].as_str().unwrap().contains("argument 1"),
            "operand position: {error}"
        );
    }
}

#[test]
fn shape_and_arity_failures_name_the_axis_or_argument_without_reversing_values() {
    let source = "def bad(a: tensor[2, f32], b: tensor[3, f32]) -> tensor[2, f32] = add(a, b)\n";
    let error = checked_error(source, "DimensionMismatch", "add(", "2", "3");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("argument 2, axis 0"),
        "shape context: {error}"
    );

    let source = "def f(x: f32) -> f32 = x\nout = f(1.0f32, 2.0f32)\n";
    let error = checked_error(
        source,
        "ArityMismatch",
        "f(1.0",
        "1 argument",
        "2 arguments",
    );
    assert!(
        error["message"].as_str().unwrap().contains("`f`"),
        "callee: {error}"
    );

    let source = "out = shrink(to_tensor([1.0f32]), [])\n";
    let error = checked_error(
        source,
        "ArityMismatch",
        "shrink(",
        "1 bounds pair",
        "0 bounds pairs",
    );
    assert!(
        error["message"].as_str().unwrap().contains("rank 1"),
        "affected rank: {error}"
    );
}

#[test]
fn expand_rejections_keep_the_call_and_directional_rank_or_unit_extent() {
    let wrong_rank =
        "def bad(x: tensor[1, 4, f32]) -> tensor[8, 1, 4, f32] = expand(x, 0i32, 8i64)\n";
    let error = checked_error(wrong_rank, "DimensionMismatch", "expand(", "2", "3");
    assert!(
        error["message"].as_str().unwrap().contains("rank"),
        "{error}"
    );

    let non_unit = "def bad(x: tensor[2, 4, f32]) -> tensor[8, 4, f32] = expand(x, 0i32, 8i64)\n";
    let error = checked_error(non_unit, "DimensionMismatch", "expand(", "1", "2");
    assert!(
        error["message"].as_str().unwrap().contains("axis 0"),
        "non-unit axis: {error}"
    );

    let (ok, stdout, stderr) = run(
        "check",
        "def good(x: tensor[1, 4, f32]) -> tensor[8, 4, f32] = expand(x, 0i32, 8i64)\n",
    );
    let report: Value = serde_json::from_str(&stdout).expect("valid expand JSON");
    assert!(ok, "{stderr}: {report}");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn expand_axis_and_size_rejections_keep_the_call_and_direction() {
    for (source, expected, got, argument) in [
        (
            "out = expand(to_tensor([1.0f32]), 2i32, 3i64)\n",
            "axis in 0..1",
            "2",
            "argument 2",
        ),
        (
            "out = expand(to_tensor([1.0f32]), -1i32, 3i64)\n",
            "non-negative axis",
            "-1",
            "argument 2",
        ),
        (
            "out = expand(to_tensor([1.0f32]), 0i32, -3i64)\n",
            "non-negative size",
            "-3",
            "argument 3",
        ),
    ] {
        let error = checked_error(source, "DimensionMismatch", "expand(", expected, got);
        assert!(
            error["message"].as_str().unwrap().contains(argument),
            "{error}"
        );
    }
}

#[test]
fn expand_rejects_non_tensor_or_runtime_axis_but_admits_runtime_size() {
    for (source, expected, got, argument, kind) in [
        (
            "out = expand([1.0f32], 0i32, 3i64)\n",
            "tensor",
            "List f32",
            "argument 1",
            "TypeMismatch",
        ),
        (
            "def f(axis: i32) = expand(to_tensor([1.0f32]), axis, 3i64)\n",
            "compile-time constant i32 axis",
            "a runtime value `axis`",
            "argument 2",
            "DimensionMismatch",
        ),
    ] {
        let error = checked_error(source, kind, "expand(", expected, got);
        assert!(
            error["message"].as_str().unwrap().contains(argument),
            "{error}"
        );
    }
    let (ok, stdout, stderr) = run(
        "check",
        "def f(n: i64) = expand(to_tensor([1.0f32]), 0i32, n)\n",
    );
    let report: Value = serde_json::from_str(&stdout).expect("runtime-size JSON");
    assert!(ok, "{stderr}: {report}");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn reshape_rejections_locate_the_call_and_keep_input_to_target_count_direction() {
    let source = "def f(x: tensor[2, 3, f32]) = reshape(x, [7i64])\n";
    let error = checked_error(
        source,
        "DimensionMismatch",
        "reshape(",
        "6 elements",
        "7 elements",
    );
    assert!(
        error["message"].as_str().unwrap().contains("argument 2"),
        "{error}"
    );

    let (ok, stdout, stderr) = run(
        "check",
        "def f(x: tensor[2, 3, f32]) -> tensor[6, f32] = reshape(x, [6i64])\n",
    );
    assert!(ok, "{stderr}: {stdout}");
}

#[test]
fn reshape_operand_shape_and_extent_errors_keep_the_call_and_argument_direction() {
    for (source, kind, expected, got, argument) in [
        (
            "out = reshape([1.0f32], [1i64])\n",
            "TypeMismatch",
            "tensor or data-element scalar",
            "List f32",
            "argument 1",
        ),
        (
            "out = reshape(to_tensor([1.0f32]), [1i32])\n",
            "PrecisionMismatch",
            "List i64",
            "List i32",
            "argument 2",
        ),
        (
            "out = reshape(to_tensor([1.0f32]), [-1i64])\n",
            "DimensionMismatch",
            "non-negative target extent",
            "-1",
            "argument 2",
        ),
        (
            "out = reshape(to_tensor([1.0f32]), [1i64], [1i64])\n",
            "ArityMismatch",
            "1 or 2 arguments",
            "3 arguments",
            "reshape",
        ),
    ] {
        let error = checked_error(source, kind, "reshape(", expected, got);
        assert!(
            error["message"].as_str().unwrap().contains(argument),
            "{error}"
        );
    }
}

#[test]
fn deferred_reshape_rejects_at_the_original_call_after_the_operand_settles() {
    let source = "def main() -> i32 = {\n  g = fn (y) -> reshape(y, [3i64])\n  _ = g([1i32, 2i32, 3i32])\n  0i32\n}\n";
    let error = checked_error(
        source,
        "TypeMismatch",
        "reshape(",
        "tensor or data-element scalar",
        "List i32",
    );
    assert!(
        error["message"].as_str().unwrap().contains("argument 1"),
        "{error}"
    );

    let valid = "def main() -> tensor[3, f32] = {\n  g = fn (y) -> reshape(y, [3i64])\n  g(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n}\n";
    let (ok, stdout, stderr) = run("check", valid);
    let report: Value = serde_json::from_str(&stdout).expect("valid deferred reshape JSON");
    assert!(ok, "{stderr}: {report}");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn erroneous_axis_dtype_reports_the_axis_argument_not_a_tensor_operand() {
    let source = "out = expand(to_tensor([1.0f32]), 0i64, 3i64)\n";
    let error = checked_error(source, "TypeMismatch", "expand(", "i32", "i64");
    assert!(
        error["message"].as_str().unwrap().contains("argument 2"),
        "axis operand: {error}"
    );
}

#[test]
fn erroneous_expand_size_dtype_reports_the_third_argument_at_the_call() {
    let source = "out = expand(to_tensor([1.0f32]), 0i32, 3i32)\n";
    let error = checked_error(source, "TypeMismatch", "expand(", "i64", "i32");
    assert!(
        error["message"].as_str().unwrap().contains("argument 3"),
        "{error}"
    );
}

#[test]
fn a_valid_counterpart_has_no_errors_and_a_build_preserves_the_same_diagnostic() {
    let source = "out = to_list(to_tensor([1.0f32, 2.0f32]))\n";
    let (ok, stdout, _) = run("check", source);
    assert!(ok, "valid rank-one conversion: {stdout}");
    let report: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["errors"], serde_json::json!([]));

    let source = "out = to_list(to_tensor([[1.0f32]]))\n";
    let error = checked_error(
        source,
        "TypeMismatch",
        "to_list(",
        "rank-1 tensor",
        "rank-2 tensor",
    );
    let (ok, _, stderr) = run("build", source);
    assert!(!ok, "bad rank must fail build");
    assert!(
        stderr.contains(error["message"].as_str().unwrap()),
        "build renders the same diagnostic: {stderr}"
    );
    assert!(
        stderr.contains(" at byte "),
        "build renders the source location: {stderr}"
    );
}

#[test]
fn concrete_positive_counterparts_are_admitted_without_diagnostics() {
    for source in [
        "out = tensor_to_scalar(sum(to_tensor([1.0f32]), 0i32))\n",
        "out = copy(to_tensor([1.0f32]))\n",
        "def good(x: f32) -> f64 = cast(x, f64)\n",
        "out = expand(to_tensor([1.0f32]), 0i32, 3i64)\n",
        "out = shrink(to_tensor([1.0f32]), [[0i64, 1i64]])\n",
        "def good(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = add(a, b)\n",
        "def f(x: f32) -> f32 = x\nout = f(1.0f32)\n",
    ] {
        let (ok, stdout, stderr) = run("check", source);
        let report: Value = serde_json::from_str(&stdout).expect("check JSON");
        assert!(ok, "{source:?} must check cleanly: {report}; {stderr}");
        assert_eq!(report["errors"], serde_json::json!([]), "{source:?}");
    }
}

#[test]
fn other_concrete_operand_failures_retain_their_directions() {
    checked_error(
        "def bad(x: List[f32]) -> List[f32] = to_list(x)\n",
        "TypeMismatch",
        "to_list(",
        "tensor",
        "List f32",
    );
    let source = "out = expand(to_tensor([1.0f32]), 0i32, 3i32)\n";
    let error = checked_error(source, "TypeMismatch", "expand(", "i64", "i32");
    assert!(
        error["message"].as_str().unwrap().contains("argument 3"),
        "size slot: {error}"
    );
}

#[test]
fn swapping_a_callers_actual_extent_keeps_the_declared_axis_direction() {
    for (declared, actual) in [(2, 3), (3, 2)] {
        let source = format!(
            "def f(x: tensor[{declared}, f32]) -> tensor[{declared}, f32] = x\n\
             out = f(to_tensor([{}]))\n",
            (0..actual).map(|_| "1.0f32").collect::<Vec<_>>().join(", ")
        );
        let error = checked_error(
            &source,
            "DimensionMismatch",
            "f(to_tensor(",
            &declared.to_string(),
            &actual.to_string(),
        );
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .contains("argument 1, axis 0"),
            "direction belongs to parameter: {error}"
        );
    }
}

#[test]
fn nested_tensor_parameter_error_identifies_the_component_and_declared_axis() {
    let source = "def f(xs: List[tensor[2, f32]]) -> i32 = 0i32\nout = f([to_tensor([1.0f32, 2.0f32, 3.0f32])])\n";
    let error = checked_error(source, "DimensionMismatch", "f([", "2", "3");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("argument 1, List element, axis 0"),
        "nested component path: {error}"
    );
    let valid =
        "def f(xs: List[tensor[2, f32]]) -> i32 = 0i32\nout = f([to_tensor([1.0f32, 2.0f32])])\n";
    let (ok, stdout, stderr) = run("check", valid);
    assert!(ok, "valid nested tensor extent: {stdout}; {stderr}");
    let report: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn settled_deferred_copy_and_cast_rejections_keep_the_original_call_site() {
    for (source, kind, call, expected, got) in [
        (
            "def bad(x: List[f32]) -> List[f32] = { k = fn (v) -> copy(v)\n k(x) }\n",
            "TypeMismatch",
            "copy(",
            "tensor",
            "List f32",
        ),
        (
            "def bad(x: List[f32]) -> f64 = { k = fn (v) -> cast(v, f64)\n k(x) }\n",
            "CastNonTensor",
            "cast(",
            "tensor or numeric/bool scalar",
            "List f32",
        ),
    ] {
        let error = checked_error(source, kind, call, expected, got);
        assert!(
            error["message"].as_str().unwrap().contains("argument 1"),
            "deferred operand: {error}"
        );
    }
}

#[test]
fn matmul_names_the_conflicting_contraction_axes_and_the_input_it_rejects() {
    let source =
        "out = matmul(to_tensor([[1.0f32, 2.0f32]]), to_tensor([[1.0f32], [2.0f32], [3.0f32]]))\n";
    let error = checked_error(source, "DimensionMismatch", "matmul(", "2", "3");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("argument 2, axis 0"),
        "{error}"
    );
    let (ok, stdout, stderr) = run(
        "check",
        "out = matmul(to_tensor([[1.0f32, 2.0f32]]), to_tensor([[1.0f32], [2.0f32]]))\n",
    );
    let report: Value = serde_json::from_str(&stdout).expect("valid matmul JSON");
    assert!(ok, "{stderr}: {report}");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn invalid_shrink_bounds_name_argument_two_and_axis_without_rejecting_a_valid_slice() {
    let source = "out = shrink(to_tensor([1.0f32, 2.0f32]), [[0i64, 3i64]])\n";
    let error = checked_error(source, "DimensionMismatch", "shrink(", "end <= 2", "end 3");
    assert!(
        error["message"].as_str().unwrap().contains("argument 2")
            && error["message"].as_str().unwrap().contains("axis 0"),
        "{error}"
    );
    let (ok, stdout, stderr) = run(
        "check",
        "out = shrink(to_tensor([1.0f32, 2.0f32]), [[0i64, 2i64]])\n",
    );
    assert!(ok, "{stderr}: {stdout}");
    let report: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn shrink_rejects_a_non_tensor_operand_at_its_own_call() {
    let source = "out = shrink([1.0f32], [[0i64, 1i64]])\n";
    let error = checked_error(source, "TypeMismatch", "shrink(", "tensor", "List f32");
    assert!(
        error["message"].as_str().unwrap().contains("argument 1"),
        "the source operand, not bounds, is invalid: {error}"
    );
}

#[test]
fn layer_norm_and_conv_name_the_conflicting_second_argument_axis() {
    let cases = [
        (
            "def bad(x: tensor[2, 3, f32], g: tensor[4, f32], b: tensor[3, f32], e: f32) = layer_norm(x, g, b, e)\n",
            "layer_norm(",
            "3",
            "4",
            "argument 2, axis 0",
        ),
        (
            "def bad(x: tensor[1, 2, 5, f32], k: tensor[3, 3, 3, f32]) = conv(x, k, [1i64], [(0i64, 0i64)])\n",
            "conv(",
            "2",
            "3",
            "argument 2, axis 1",
        ),
    ];
    for (source, call, expected, got, position) in cases {
        let error = checked_error(source, "DimensionMismatch", call, expected, got);
        assert!(
            error["message"].as_str().unwrap().contains(position),
            "{error}"
        );
    }
}

#[test]
fn permute_axis_dtype_identifies_the_second_argument_and_a_valid_permutation_passes() {
    let source = "out = permute(to_tensor([[1.0f32, 2.0f32]]), 1i64, 0i32)\n";
    let error = checked_error(source, "TypeMismatch", "permute(", "i32", "i64");
    assert!(
        error["message"].as_str().unwrap().contains("argument 2"),
        "{error}"
    );
    let (ok, stdout, stderr) = run(
        "check",
        "out = permute(to_tensor([[1.0f32, 2.0f32]]), 1i32, 0i32)\n",
    );
    let report: Value = serde_json::from_str(&stdout).unwrap();
    assert!(ok, "{stderr}: {report}");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn duplicate_definition_points_to_the_redeclaration_not_the_first_definition() {
    let source = "def f() -> i32 = 1i32\ndef f() -> i32 = 2i32\n";
    let (ok, stdout, stderr) = run("check", source);
    let report: Value = serde_json::from_str(&stdout).expect("check JSON");
    assert!(!ok, "{stderr}: {report}");
    let duplicate = report["errors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|error| error["kind"] == "DuplicateDefinition")
        .unwrap_or_else(|| panic!("missing duplicate declaration: {report}"));
    let at = source.find("def f() -> i32 = 2i32").unwrap();
    assert_eq!(
        duplicate["span"]["offset"], at,
        "second declaration: {duplicate}"
    );
    let (ok, stdout, stderr) = run("check", "def f() -> i32 = 1i32\ndef g() -> i32 = 2i32\n");
    let report: Value = serde_json::from_str(&stdout).expect("check JSON");
    assert!(ok, "{stderr}: {report}");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn authored_argument_dtypes_are_directional_but_inferred_binary_operands_are_not() {
    let cases = [
        (
            "def bad(x: tensor[1, 1, 3, 3, f32], k: tensor[1, 1, 1, 1, f32]) -> tensor[1, 1, 3, 3, f32] = conv(x, k, [1.0f64, 1.0f64], [(0i64, 0i64), (0i64, 0i64)])\n",
            "conv(",
            "i64",
            "f64",
            "argument 3",
        ),
        (
            "def accept(x: f32) -> f32 = x\nout = accept(1.0f64)\n",
            "accept(1.0",
            "f32",
            "f64",
            "argument 1",
        ),
    ];
    for (source, call, expected, got, position) in cases {
        let error = checked_error(source, "PrecisionMismatch", call, expected, got);
        let message = error["message"].as_str().unwrap();
        assert!(message.contains(position), "{error}");
        assert!(
            message.contains("expected") && message.contains("got"),
            "{error}"
        );
    }
    let (ok, stdout, _) = run("check", "out = add(1.0f32, 2.0f64)\n");
    let report: Value = serde_json::from_str(&stdout).expect("check JSON");
    assert!(!ok, "{report}");
    let conflict = report["errors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|error| {
            error["kind"] == "PrecisionMismatch"
                && error["message"].as_str().is_some_and(|message| {
                    message.contains("add") && message.contains("f32") && message.contains("f64")
                })
        })
        .unwrap_or_else(|| panic!("mixed binary precisions must reject: {report}"));
    assert!(
        conflict["expected"].is_null() && conflict["got"].is_null(),
        "{conflict}"
    );
}

#[test]
fn declared_return_and_ascriptions_name_their_own_direction() {
    for (source, kind, expected, got) in [
        (
            "def bad() -> f32 = 1.0f64\n",
            "TypeMismatch",
            "() -> f32",
            "() -> f64",
        ),
        (
            "def bad(x: f64) -> f32 = { value: f32 = x\n value }\n",
            "PrecisionMismatch",
            "f32",
            "f64",
        ),
        (
            "def bad(x: f64) -> f32 = (x : f32)\n",
            "PrecisionMismatch",
            "f32",
            "f64",
        ),
    ] {
        let (ok, stdout, stderr) = run("check", source);
        let report: Value = serde_json::from_str(&stdout).expect("check JSON");
        assert!(!ok, "{stderr}: {report}");
        let error = report["errors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|error| {
                error["kind"] == kind && error["expected"] == expected && error["got"] == got
            })
            .unwrap_or_else(|| panic!("missing declared-vs-inferred direction: {report}"));
        let message = error["message"].as_str().unwrap();
        assert!(
            message.contains("expected") && message.contains("got"),
            "{error}"
        );
        assert!(
            message.contains(expected) && message.contains(got),
            "{error}"
        );
        assert!(error["span"]["offset"].as_u64().is_some(), "{error}");
    }
}
