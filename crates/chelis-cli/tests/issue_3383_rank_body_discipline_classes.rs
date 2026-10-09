//! chelis#3383: the `..r` Body-Discipline admits by property, not by a list.
//!
//! `spec/04-type-system.md` section 4.5.3 admits a builtin in a
//! rank-polymorphic body exactly when it is name-trackable: shape-identity,
//! named-axis, ordered-prefix, or inert (it produces no new tensor). `fail`,
//! `drop`, `dropout`, and `layer_norm` meet that test but were rejected as
//! "shape-rewriting" because the class was a hand-kept allowlist whose
//! default was rejection. Each positive row runs on eval and on compiled C;
//! the rank-polymorphic results are compared with the same computation at
//! concrete rank. The negative rows pin that an untracked builtin, a
//! `layer_norm` whose trailing axis is inside a spread, and an inert call
//! whose operands disagree are still rejected.

#[path = "common/host_effect_parity.rs"]
mod parity;

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

fn check_json(source: &str) -> Value {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    fs::write(&path, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output is json")
}

fn error_messages(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("errors should be a json array, got {json}"))
        .iter()
        .filter_map(|error| error["message"].as_str().map(str::to_owned))
        .collect()
}

fn assert_checks_clean(source: &str, label: &str) {
    let json = check_json(source);
    assert!(
        error_messages(&json).is_empty() && json["score"].as_f64() == Some(1.0),
        "{label}: expected a clean check, got {json}"
    );
}

fn assert_rejected_with(source: &str, needle: &str, label: &str) -> Vec<String> {
    let messages = error_messages(&check_json(source));
    assert!(
        messages.iter().any(|message| message.contains(needle)),
        "{label}: expected a rejection containing {needle:?}, got {messages:?}"
    );
    messages
}

const FAIL_GUARD: &str = "def checked[r](x: tensor[..r, f32], eps: f32) -> tensor[..r, f32] = if lte(eps, 0.0f32) then fail(\"eps must be positive\") else x\n";

#[test]
fn a_fail_guard_in_a_rank_polymorphic_body_returns_its_operand() {
    let source = format!(
        "{FAIL_GUARD}def main() -> tensor[2, 2, f32] = checked(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]), 0.5f32)\n"
    );
    let run = parity::assert_lanes_agree(&source, "fail_guard_passes");
    assert_eq!(run.status, Some(0), "{run:?}");
    assert_eq!(
        run.stdout,
        "main = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])\n"
    );
}

#[test]
fn a_fail_guard_in_a_rank_polymorphic_body_aborts_identically() {
    let source = format!(
        "{FAIL_GUARD}def main() -> tensor[2, f32] = checked(to_tensor([1.0f32, 2.0f32]), 0.0f32)\n"
    );
    let run = parity::assert_lanes_agree(&source, "fail_guard_aborts");
    assert_eq!(run.status, Some(1), "{run:?}");
    assert_eq!(run.failure, "eps must be positive");
}

const LAYER_NORM_ARGS: &str = "to_tensor([[1.0f32, 2.0f32, 4.0f32], [3.0f32, 5.0f32, 6.0f32]]), to_tensor([1.0f32, 2.0f32, 1.0f32]), to_tensor([0.0f32, 1.0f32, 0.0f32])";

#[test]
fn layer_norm_over_a_named_trailing_axis_matches_concrete_rank() {
    let rank_polymorphic = format!(
        "def norm[r](x: tensor[..r, hidden, f32], g: tensor[hidden, f32], b: tensor[hidden, f32]) -> tensor[..r, hidden, f32] = layer_norm(x, g, b, 0.00001f32)\n\
         def rows(x: tensor[rows, hidden, f32], g: tensor[hidden, f32], b: tensor[hidden, f32]) -> tensor[rows, hidden, f32] = norm(x, g, b)\n\
         def main() -> tensor[2, 3, f32] = rows({LAYER_NORM_ARGS})\n"
    );
    let concrete = format!(
        "def norm(x: tensor[2, 3, f32], g: tensor[3, f32], b: tensor[3, f32]) -> tensor[2, 3, f32] = layer_norm(x, g, b, 0.00001f32)\n\
         def main() -> tensor[2, 3, f32] = norm({LAYER_NORM_ARGS})\n"
    );
    let run = parity::assert_lanes_agree(&rank_polymorphic, "layer_norm_rank_polymorphic");
    let oracle = parity::eval_lane(&concrete, "layer_norm_concrete");
    assert_eq!(run.status, Some(0), "{run:?}");
    assert_eq!(oracle.status, Some(0), "{oracle:?}");
    assert_eq!(run.stdout, oracle.stdout);
}

#[test]
fn layer_norm_whose_trailing_axis_is_inside_a_spread_is_rejected() {
    for gamma in ["h", "hidden", "4"] {
        let binders = if gamma == "h" { "r, h" } else { "r" };
        let source = format!(
            "def norm[{binders}](x: tensor[..r, f32], g: tensor[{gamma}, f32], b: tensor[{gamma}, f32]) -> tensor[..r, f32] = layer_norm(x, g, b, 0.00001f32)\n"
        );
        assert_rejected_with(
            &source,
            "expected a named trailing axis at symbolic rank",
            &format!("gamma tensor[{gamma}]"),
        );
    }
}

#[test]
fn dropout_in_a_rank_polymorphic_body_matches_the_concrete_rank_draw() {
    let input = "to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]])";
    let rank_polymorphic = format!(
        "def drop_any[r](k: key, x: tensor[..r, f32], rate: f32) -> tensor[..r, f32] = dropout(k, &x, rate)\n\
         def main() -> tensor[2, 3, f32] = drop_any(key_from_seed(7i64), {input}, 0.5f32)\n"
    );
    let concrete = format!(
        "def drop_two(k: key, x: tensor[2, 3, f32], rate: f32) -> tensor[2, 3, f32] = dropout(k, &x, rate)\n\
         def main() -> tensor[2, 3, f32] = drop_two(key_from_seed(7i64), {input}, 0.5f32)\n"
    );
    let run = parity::assert_lanes_agree(&rank_polymorphic, "dropout_rank_polymorphic");
    let oracle = parity::eval_lane(&concrete, "dropout_concrete");
    assert_eq!(run.status, Some(0), "{run:?}");
    assert_eq!(run.stdout, oracle.stdout);
    assert!(
        run.stdout.contains("0.0") && run.stdout.contains("4.0"),
        "the draw both drops and scales: {}",
        run.stdout
    );
}

#[test]
fn drop_in_a_rank_polymorphic_body_discards_an_owned_operand() {
    let source = "def first[r](x: tensor[..r, f32], y: tensor[..r, f32]) -> tensor[..r, f32] = {\n  _ = drop(y)\n  x\n}\n\
                  def main() -> tensor[2, f32] = first(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32]))\n";
    let run = parity::assert_lanes_agree(source, "drop_operand");
    assert_eq!(run.status, Some(0), "{run:?}");
    assert_eq!(run.stdout, "main = tensor(shape=[2], data=[1.0, 2.0])\n");
}

#[test]
fn observation_and_assertion_builtins_run_in_a_rank_polymorphic_body() {
    let observed = "def seen[r](x: tensor[..r, f32]) -> tensor[..r, f32] ! { IO } = {\n  _ = print(\"seen\")\n  debug(x)\n}\n\
                    out = seen(to_tensor([1.0f32, 2.0f32]))\n";
    let run = parity::assert_lanes_agree(observed, "print_debug");
    assert_eq!(
        run.stdout,
        "seen\ntensor(shape=[2], data=[1.0, 2.0])\nout = tensor(shape=[2], data=[1.0, 2.0])\n"
    );

    let written = "def save[r](x: tensor[..r, f32], path: string) -> tensor[..r, f32] ! { IO } = {\n  _ = write_file(path, \"saved\")\n  x\n}\n\
                   out = save(to_tensor([1.0f32, 2.0f32]), \"saved.txt\")\n";
    let run = parity::assert_lanes_agree(written, "write_file");
    assert_eq!(run.stdout, "out = tensor(shape=[2], data=[1.0, 2.0])\n");

    let asserted = "def same[r](a: tensor[..r, f32], b: tensor[..r, f32]) -> unit ! { Test } = {\n  _ = test_assert_close_tensor(a, b, 0.0f32, \"close\")\n  _ = test_assert_eq_tensor(a, b, \"equal\")\n  _ = test_assert_eq(1i64, 1i64, \"scalar\")\n  test_assert(true, \"holds\")\n}\n\
                    checked = same(to_tensor([[1.0f32, 2.0f32]]), to_tensor([[1.0f32, 2.0f32]]))\n";
    let run = parity::assert_lanes_agree(asserted, "test_asserts");
    assert_eq!(run.stdout, "checked = ()\n");
}

#[test]
fn an_untracked_builtin_is_still_rejected_and_named_by_the_property() {
    for (call, op) in [
        ("permute(x, 1, 0)", "permute"),
        ("reshape(x, [2i64, 3i64])", "reshape"),
        ("cumsum(x, 0i32)", "cumsum"),
    ] {
        let source = format!("def bad[r](x: tensor[..r, f32]) -> tensor[..r, f32] = {call}\n");
        let messages = assert_rejected_with(
            &source,
            &format!("may not call builtin `{op}`: it is not name-trackable at symbolic rank"),
            op,
        );
        assert!(
            messages
                .iter()
                .all(|message| !message.contains("shape-rewriting")),
            "{op}: the rejection names the property, not a shape claim: {messages:?}"
        );
    }
    assert_rejected_with(
        "def bad[r](x: tensor[..r, f32]) -> i64 = string_len(\"abc\")\n",
        "may not call builtin `string_len`: it is not name-trackable",
        "string_len",
    );
}

#[test]
fn an_inert_or_shape_identity_call_still_unifies_its_operands() {
    assert_rejected_with(
        "def bad[r, s](a: tensor[..r, f32], b: tensor[..s, f32]) -> unit ! { Test } = test_assert_close_tensor(a, b, 0.0f32, \"c\")\n",
        "distinct declared rank parameters `r` and `s`",
        "test_assert_close_tensor over two ranks",
    );
    assert_rejected_with(
        "def bad[r](k: key, x: tensor[..r, f32], rate: f32) -> tensor[..r, hidden, f32] = dropout(k, &x, rate)\n",
        "body doesn't match declared signature",
        "dropout claiming an extra axis",
    );
}

#[test]
fn the_issue_definitions_check_clean() {
    assert_checks_clean(FAIL_GUARD, "fail guard");
    assert_checks_clean(
        "def norm[r](x: tensor[..r, hidden, f32], g: tensor[hidden, f32], b: tensor[hidden, f32]) -> tensor[..r, hidden, f32] = layer_norm(x, g, b, 0.00001f32)\n",
        "layer_norm",
    );
    assert_checks_clean(
        "def drop_any[r](k: key, x: tensor[..r, f32], rate: f32) -> tensor[..r, f32] = dropout(k, &x, rate)\n",
        "dropout",
    );
    assert_checks_clean(
        "def first[r](x: tensor[..r, f32], y: tensor[..r, f32]) -> tensor[..r, f32] = {\n  _ = drop(y)\n  x\n}\n",
        "drop",
    );
}
