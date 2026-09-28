//! chelis#2413: a destructured `split_key` pair in a program whose tensor
//! graph stages host-produced sizes.
//!
//! A runtime `reshape` extent such as `numel(first)` makes the lowering stage
//! host sources into the tensor graph. `split_key` has host type `(key,
//! key)`, which is not a tensor, so the staging treated the pair as an opaque
//! host value; the `tuple-get` of the destructuring `let` then found no tuple
//! and lowering failed with "tuple-get index 0 out of bounds during
//! lowering", in eval and in C alike. The graph carries each key as a key
//! node and the pair as a lowered tuple ([05-OP-70]), so a key-only tuple is
//! never staged.
//!
//! Expected bits come from the [05-RNG-2]/[05-OP-37] reference
//! (`key_ref.py` extended by the slice-5 ledger script), not from a run of the
//! implementation: `split_key(key_from_seed(42))` is `(a, b)`; `dropout(a,
//! [1, 2, 3, 4], 0.5)` keeps elements 0 and 3 (`[2, 0, 0, 8]`), and
//! `dropout(b, ., 0.5)` keeps elements 0 and 1, so the result is `[4, 0, 0,
//! 0]`, f32 bits `40800000 00000000 00000000 00000000`.

mod common;

use assert_cmd::Command;

const STAGED_PAIR: &str = r#"def pair(k: key, x: tensor[4, f32]) -> tensor[2, 2, f32] = {
  (k1, k2) = split_key(k)
  first = dropout(k1, x, 0.5f32)
  dropout(k2, reshape(first, [2i64, floor_div(numel(first), 2i64)]), 0.5f32)
}
def main() = pair(key_from_seed(42i64), to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))
"#;

const UNUSED_PAIR: &str = r#"def main() = {
  (k1, k2) = split_key(key_from_seed(42i64))
  x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  reshape(x, [numel(x), 1i64])
}
"#;

const REUSED_HALF: &str = r#"def pair(k: key, x: tensor[4, f32]) -> tensor[2, 2, f32] = {
  (k1, k2) = split_key(k)
  first = dropout(k1, x, 0.5f32)
  dropout(k1, reshape(first, [2i64, floor_div(numel(first), 2i64)]), 0.5f32)
}
"#;

// [04] tuple projection and [04-LIN-9] state transport through a selected
// ADT arm: the wrapper must preserve the direct helper's tuple shape.
const STATE_WRAPPER: &str = include_str!("../../../examples/keyed_state_wrapper.ch");

fn f32_line(name: &str, shape: &str, bits: &[u32]) -> String {
    let data = bits
        .iter()
        .map(|bits| format!("{:?}", f32::from_bits(*bits)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{name} = tensor(shape={shape}, data=[{data}])\n")
}

/// Eval stdout and compiled-C stdout of one standalone program.
fn eval_and_c(stem: &str, source: &str) -> (String, String) {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join(format!("{stem}.ch"));
    std::fs::write(&file, source).unwrap();
    let path = file.to_str().unwrap();
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path])
        .output()
        .unwrap();
    assert!(
        eval.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let out = dir.path().join("out");
    let build = Command::cargo_bin("chelis")
        .unwrap()
        .args(["build", path, "--target", "c", "--output"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(common::link_generated(&out, &format!("{stem}.c"), stem).success());
    let run = std::process::Command::new(out.join(stem))
        .current_dir(&out)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "compiled program failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    (
        String::from_utf8(eval.stdout).unwrap(),
        String::from_utf8(run.stdout).unwrap(),
    )
}

#[test]
fn a_split_pair_feeds_draws_beside_a_host_produced_extent_in_eval_and_c() {
    let (eval, native) = eval_and_c("staged_pair", STAGED_PAIR);
    assert_eq!(
        eval,
        f32_line("main", "[2, 2]", &[0x4080_0000, 0, 0, 0]),
        "eval"
    );
    assert_eq!(native, eval, "eval and C disagree");
}

#[test]
fn an_unused_split_pair_beside_a_host_produced_extent_lowers_in_eval_and_c() {
    let (eval, native) = eval_and_c("unused_pair", UNUSED_PAIR);
    assert_eq!(eval, f32_line("main", "[4, 1]", &[0x3f80_0000; 4]), "eval");
    assert_eq!(native, eval, "eval and C disagree");
}

#[test]
fn a_half_of_the_pair_is_still_used_at_most_once() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("reused.ch");
    std::fs::write(&file, REUSED_HALF).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["check", file.to_str().unwrap()])
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        errors
            .iter()
            .any(|error| error["message"].as_str().is_some_and(|m| m.contains("k1"))),
        "a second consuming use of `k1` must be rejected: {report}"
    );
}

#[test]
fn a_keyed_tuple_result_survives_an_adt_state_match_in_eval_and_c() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("state_wrapper.ch");
    std::fs::write(&file, STATE_WRAPPER).unwrap();
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .args(["check", file.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(checked.status.success());
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"], 1.0, "{report}");
    assert_eq!(report["errors"], serde_json::json!([]), "{report}");
    let (eval, native) = eval_and_c("state_wrapper", STATE_WRAPPER);
    assert_eq!(eval, f32_line("main", "[2, 1, 1, 1]", &[0x4000_0000, 0]));
    assert_eq!(native, eval);
}

#[test]
fn the_direct_helper_stays_a_positive_control_for_state_tuple_transport() {
    let direct = STATE_WRAPPER.replace(
        "wrapped(x, 0.5f32, State { seed: 0i64, next_draw: 0i64 })",
        "direct(x, 0.5f32, 0i64, 0i64)",
    );
    assert_ne!(direct, STATE_WRAPPER, "direct control must change the call");
    let (eval, native) = eval_and_c("direct_state", &direct);
    assert_eq!(eval, f32_line("main", "[2, 1, 1, 1]", &[0x4000_0000, 0]));
    assert_eq!(native, eval);
}

#[test]
fn a_projection_past_the_state_tuple_end_is_a_type_error() {
    let invalid = STATE_WRAPPER.replace("(out, _) = wrapped", "(out, _, extra) = wrapped");
    assert_ne!(
        invalid, STATE_WRAPPER,
        "negative control must change the tuple"
    );
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("state_bad_arity.ch");
    std::fs::write(&file, invalid).unwrap();
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .args(["check", file.to_str().unwrap()])
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        errors
            .iter()
            .any(|error| error["kind"] == "TupleIndexOutOfBounds"),
        "{report}"
    );
}
