// chelis#2123: a float literal whose value is finite as written but rounds to
// infinity at the dtype it binds at scored 1.0 and evaluated to `inf`:
// `70000.0f16`, `1e40f32`, and an unsuffixed `1e40` at the f32 default. Only
// the integer-bodied Surf spelling (`65520f16`) was rejected, by the parser.
//
// spec/04-type-system.md [04-LIT-2] states the rule for every literal: it
// binds at its dtype (suffix, default, contextual position, each instantiation
// of an adopted dtype binder, or a pattern's scrutinee) and is rejected when
// its value there is non-finite. These tests drive the real CLI through
// `check` and `eval --file`, for Surf and Deep ingress, and pin the cases that
// must stay legal beside each rejection: a finite value at the boundary,
// rounding to zero, an explicit cast of a finite wider value, and arithmetic
// overflow, which are operations rather than literals.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_fixture(dir: &Path, name: &str, source: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    fs::write(&path, source).expect("write fixture");
    path
}

/// `chelis check` JSON for `path`. The exit status is not asserted here:
/// `check` exits non-zero when `errors` is non-empty, and both outcomes are
/// asserted from the report by the callers.
fn check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "check output for {} is not JSON ({error}): stdout={} stderr={}",
            path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
    })
}

fn messages(report: &Value) -> Vec<String> {
    report["errors"]
        .as_array()
        .expect("errors is an array")
        .iter()
        .map(|error| error["message"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn eval(path: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval --file")
}

/// The program is rejected by `check` with a type error that cites
/// [04-LIT-2] and names every `needles` string, and `eval` refuses it rather
/// than printing an infinity.
fn assert_rejected(source: &str, file: &str, needles: &[&str]) {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), file, source);

    let report = check(&path);
    let messages = messages(&report);
    assert!(
        report["score"].as_f64().expect("score") < 1.0,
        "{source:?} must not score 1.0: {report}",
    );
    assert!(
        messages.iter().any(|message| {
            message.contains("[04-LIT-2]") && needles.iter().all(|needle| message.contains(needle))
        }),
        "{source:?}: expected a [04-LIT-2] diagnostic naming {needles:?}, got {messages:?}",
    );

    let output = eval(&path);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success(),
        "eval accepted {source:?}: stdout={stdout}",
    );
    assert!(
        !stdout.contains("inf"),
        "eval printed an infinity for {source:?}: {stdout}",
    );
}

/// The program checks clean and `eval` prints exactly `expected`.
fn assert_evaluates(source: &str, file: &str, expected: &str) {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), file, source);

    let report = check(&path);
    assert_eq!(
        report["score"].as_f64(),
        Some(1.0),
        "{source:?} must check clean: {report}",
    );
    assert!(messages(&report).is_empty(), "{source:?}: {report}");

    let output = eval(&path);
    assert!(
        output.status.success(),
        "eval failed for {source:?}: {}",
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        expected,
        "{source:?}"
    );
}

#[test]
fn a_decimal_bodied_suffixed_literal_is_held_to_its_suffix_width() {
    // The issue's headline case and the exact f16 overflow threshold, which
    // ties to infinity under round-to-nearest-even.
    assert_rejected("def y() -> f16 = 70000.0f16\n", "f16_70000.ch", &["f16"]);
    assert_rejected("def y() -> f16 = 65520.0f16\n", "f16_65520.ch", &["f16"]);
    assert_rejected("def z() -> f32 = 1e40f32\n", "f32_1e40.ch", &["f32"]);
    assert_rejected(
        "def z() -> f32 = 3.4028236e38f32\n",
        "f32_edge.ch",
        &["f32"],
    );
    assert_rejected("def b() -> bf16 = 3.4e38bf16\n", "bf16_edge.ch", &["bf16"]);
}

#[test]
fn a_finite_value_at_the_suffix_width_still_binds() {
    // 65504 is f16's largest value, and 65519.0 rounds down to it.
    assert_evaluates(
        "def y() -> f16 = 65504.0f16\n",
        "f16_max.ch",
        "y = 65500.0\n",
    );
    assert_evaluates(
        "def y() -> f16 = 65519.0f16\n",
        "f16_below.ch",
        "y = 65500.0\n",
    );
    assert_evaluates(
        "def z() -> f32 = 3.4028235e38f32\n",
        "f32_max.ch",
        "z = 3.4028235e38\n",
    );
    assert_evaluates(
        "def b() -> bf16 = 3.39e38bf16\n",
        "bf16_max.ch",
        "b = 3.39e38\n",
    );
    assert_evaluates("def w() -> f64 = 1e300f64\n", "f64_big.ch", "w = 1e300\n");
}

#[test]
fn an_unsuffixed_literal_is_held_to_the_f32_default() {
    assert_rejected("def z() -> f32 = 1e40\n", "default_1e40.ch", &["f32"]);
}

#[test]
fn rounding_to_zero_is_ordinary_rounding_not_a_rejection() {
    assert_evaluates("def u() -> f32 = 1e-50\n", "underflow_f32.ch", "u = 0.0\n");
    assert_evaluates(
        "def u() -> f16 = 1e-10f16\n",
        "underflow_f16.ch",
        "u = 0.0\n",
    );
}

#[test]
fn every_contextual_position_holds_the_literal_to_the_adopted_width() {
    // §5.6 positions 1 and 3, a scalar in position 4, and an integer element
    // bound at f16.
    assert_rejected(
        "def t() -> tensor[2, f16] = {\n  xs: tensor[2, f16] = [1.0, 70000.0]\n  xs\n}\n",
        "let_binding.ch",
        &["f16"],
    );
    assert_rejected(
        "def t() -> tensor[2, f16] = [1.0, 70000.0]\n",
        "return_body.ch",
        &["f16"],
    );
    assert_rejected(
        "def c() -> f16 = cast(70000.0, f16)\n",
        "cast_operand.ch",
        &["f16"],
    );
    assert_rejected(
        "def t() -> tensor[2, f16] = [1, 70000]\n",
        "integer_element.ch",
        &["f16"],
    );
}

#[test]
fn operations_that_overflow_still_produce_infinity() {
    // A cast of an already-bound finite value is [04-NUM-14]'s total float
    // cast, and arithmetic overflow is [04-NUM-2]'s finalization; neither is
    // a literal.
    assert_evaluates(
        "def c() -> f16 = cast(70000.0f32, f16)\n",
        "cast_of_f32.ch",
        "c = inf\n",
    );
    assert_evaluates(
        "def m() -> f16 = (65504.0f16 * 2.0f16)\n",
        "mul_overflow.ch",
        "m = inf\n",
    );
}

#[test]
fn a_literal_adopting_a_float_binder_is_held_to_every_family_member() {
    // [04-INF-6]: the body must check at every admissible instantiation, so
    // the literal is rejected at the declaration even though this program
    // only instantiates `p` at f32.
    assert_rejected(
        "def big[p: Float](x: p) -> p = (x * cast(70000.0, p))\n\
         def at32() -> f32 = big(1.0f32)\n",
        "binder_float.ch",
        &["p: Float", "f16"],
    );
    assert_rejected(
        "def big[p: Float](x: p) -> p = (x * cast(70000, p))\n\
         def at32() -> f32 = big(1.0f32)\n",
        "binder_integer.ch",
        &["p: Float", "f16"],
    );
    assert_rejected(
        "def big[p: Numeric](x: p) -> p = (x * cast(70000.0, p))\n\
         def at32() -> f32 = big(1.0f32)\n",
        "binder_numeric.ch",
        &["p: Numeric", "f16"],
    );
    assert_evaluates(
        "def half[p: Float](x: p) -> p = (x * cast(0.5, p))\n\
         def at16() -> f16 = half(3.0f16)\n",
        "binder_ok.ch",
        "at16 = 1.5\n",
    );
}

#[test]
fn deep_ingress_is_held_to_the_declared_primitive() {
    // Hand-written Deep reaches the checker without the Surf parser, so the
    // integer-marked form's parse-time rejection does not cover it.
    assert_rejected(
        "(defsig {} y (t-fn {} (t-prim {} f16)))\n\n\
         (def {} y (fn {} (params {}) (lit {type: (t-prim {} f16)} 70000.0)))\n",
        "float_atom.dp",
        &["f16"],
    );
    assert_rejected(
        "(defsig {} y (t-fn {} (t-prim {} f16)))\n\n\
         (def {}\n  y\n  (fn {}\n    (params {})\n    \
         (lit {literal_source: integer, type: (t-prim {} f16)} 65520)))\n",
        "integer_atom.dp",
        &["f16"],
    );
    // A `lit` with no `type` metadata takes the §5.3 f32 default.
    assert_rejected(
        "(defsig {} y (t-fn {} (t-prim {} f32)))\n\n\
         (def {}\n  y\n  (fn {}\n    (params {})\n    \
         (lit {} 10000000000000000000000000000000000000000.0)))\n",
        "untyped_atom.dp",
        &["f32"],
    );
}

#[test]
fn a_float_literal_pattern_is_held_to_the_scrutinee_width() {
    assert_rejected(
        "def classify(x: f16) -> i32 =\n  match x with {\n    | 70000.0 => 1\n    | _ => 0\n  }\n\
         def big() -> i32 = classify(cast(70000.0f32, f16))\n",
        "pattern_overflow.ch",
        &["[04-PAT-1]", "f16"],
    );
    assert_evaluates(
        "def classify(x: f16) -> i32 =\n  match x with {\n    | 0.5 => 1\n    | _ => 0\n  }\n\
         def half() -> i32 = classify(0.5f16)\n",
        "pattern_ok.ch",
        "half = 1\n",
    );
}
