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
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .args(["check", path])
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert!(checked.status.success(), "{report}");
    assert_eq!(report["score"], 1.0, "{report}");
    assert_eq!(report["errors"], serde_json::json!([]), "{report}");
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

// [04-PAT-2] selects the tuple once; projections preserve its scalar values.
const SCALAR_STATE: &str = r#"type State =
  | State { size: i64 }
def sizes(state: State) -> (i64, i64) =
  match state with {
    | State { size } => (size, 1i64)
  }
def main() -> tensor[2, 1, f32] = {
  (a, b) = sizes(State { size: 2i64 })
  reshape(to_tensor([1.0f32, 2.0f32]), [a, b])
}
"#;

#[test]
fn scalar_state_tuple_projections_feed_reshape_in_eval_and_c() {
    let (eval, native) = eval_and_c("scalar_state", SCALAR_STATE);
    assert_eq!(
        eval,
        f32_line("main", "[2, 1]", &[0x3f80_0000, 0x4000_0000])
    );
    assert_eq!(native, eval);
}

#[test]
fn scalar_state_tuple_at_host_root_in_eval_and_c() {
    let source = SCALAR_STATE.split("def main()").next().unwrap().to_owned()
        + "def main() = sizes(State { size: 2i64 })\n";
    let (eval, native) = eval_and_c("scalar_host", &source);
    assert_eq!(eval, "main.0 = 2\nmain.1 = 1\n");
    assert_eq!(native, eval);
}

#[test]
fn scalar_state_tuple_invalid_projection_is_rejected() {
    let source = SCALAR_STATE.replace("(a, b) =", "(a, b, extra) =");
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("scalar_invalid.ch");
    std::fs::write(&file, source).unwrap();
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .arg("check")
        .arg(&file)
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert!(
        report["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["kind"] == "TupleIndexOutOfBounds"),
        "{report}"
    );
}

#[test]
fn runtime_selected_nested_host_tuple_preserves_both_branches() {
    for size in [1, 2] {
        let source = format!(
            r#"type State =
  | State {{ size: i64 }}
def sizes(state: State) -> ((i64, i64), string) =
  match state with {{
    | State {{ size }} => if eq(size, 2i64) then ((2i64, 1i64), "two") else ((1i64, 2i64), "one")
  }}
def main() = {{
  x = to_tensor([{values}])
  (dims, label) = sizes(State {{ size: numel(x) }})
  (a, b) = dims
  (reshape(to_tensor([1.0f32, 2.0f32]), [a, b]), label)
}}
"#,
            values = if size == 2 {
                "1.0f32, 2.0f32"
            } else {
                "1.0f32"
            }
        );
        let (eval, native) = eval_and_c("runtime_host_tuple", &source);
        let shape = if size == 2 { "[2, 1]" } else { "[1, 2]" };
        assert!(
            eval.contains(&f32_line("main.0", shape, &[0x3f80_0000, 0x4000_0000])),
            "{eval}"
        );
        assert_eq!(native, eval);
    }
}

fn state_wrapper_with_dynamic_extents(dims: [usize; 4]) -> String {
    let values = vec!["1.0f32"; dims.iter().product()].join(", ");
    let extents = dims.map(|dim| format!("extent({dim}i64)")).join(", ");
    let source = "def extent(n: i64) -> i64 = bitand(n, 7i64)\n".to_owned()
        + &STATE_WRAPPER.replace(
            "to_tensor([1.0f32, 2.0f32]) |> reshape([2i64, 1i64, 1i64, 1i64])",
            &format!("to_tensor([{values}])\n    |> reshape([{extents}])"),
        );
    assert!(source.contains("reshape([extent("));
    source
}

#[test]
fn state_tuple_result_claim_accepts_matching_dynamic_extents_in_both_lanes() {
    let (eval, native) = eval_and_c(
        "state_dynamic",
        &state_wrapper_with_dynamic_extents([2, 1, 1, 1]),
    );
    assert_eq!(eval, f32_line("main", "[2, 1, 1, 1]", &[0x4000_0000, 0]));
    assert_eq!(native, eval);
}

// The executable example's four host result claims retain the `mul`
// producer through the tuple. Each claim must reject in Eval and native C.
#[test]
fn state_tuple_result_claim_rejects_each_wrong_axis_in_both_lanes() {
    for axis in 0..4 {
        let mut dims = [2, 1, 1, 1];
        let claimed = dims[axis];
        dims[axis] += 1;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("wrong_axis.ch");
        std::fs::write(&file, state_wrapper_with_dynamic_extents(dims)).unwrap();
        let checked = Command::cargo_bin("chelis")
            .unwrap()
            .arg("check")
            .arg(&file)
            .output()
            .unwrap();
        let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert!(checked.status.success(), "{report}");
        assert_eq!(report["errors"], serde_json::json!([]), "{report}");
        let eval = Command::cargo_bin("chelis")
            .unwrap()
            .args(["eval", "--file"])
            .arg(&file)
            .output()
            .unwrap();
        let out = dir.path().join("out");
        let build = Command::cargo_bin("chelis")
            .unwrap()
            .arg("build")
            .arg(&file)
            .args(["--target", "c", "--output"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        assert!(common::link_generated(&out, "wrong_axis.c", "wrong_axis").success());
        let native = std::process::Command::new(out.join("wrong_axis"))
            .current_dir(&out)
            .output()
            .unwrap();
        for (lane, output) in [("eval", eval), ("C", native)] {
            assert!(
                !output.status.success(),
                "{lane} accepted wrong axis {axis}"
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains(&format!(
                    "claimed = {claimed}, mul axis {axis} = {}",
                    dims[axis]
                )),
                "{lane}: {stderr}"
            );
            assert!(
                stderr.contains("numeric trap: domain in mul at i64"),
                "{lane}: {stderr}"
            );
        }
    }
}
