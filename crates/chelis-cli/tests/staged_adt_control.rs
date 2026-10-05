//! [04-PAT-2]: ordinary forward matches preserve their selected host arm
//! beside tensor staging, including initializer-returned ADTs (chelis#2724).
mod common;
use assert_cmd::Command;

const INITIALIZED: &str = include_str!("../../../examples/staged_adt_control.ch");

fn write_source(file: &std::path::Path, source: &str) {
    std::fs::write(file, source).unwrap();
    let formatted = Command::cargo_bin("chelis")
        .unwrap()
        .args(["fmt", "--inplace"])
        .arg(file)
        .output()
        .unwrap();
    assert!(
        formatted.status.success(),
        "{}",
        String::from_utf8_lossy(&formatted.stderr)
    );
}

/// Eval stdout and compiled-C stdout of one standalone program.
fn eval_and_c(stem: &str, source: &str) -> (String, String) {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join(format!("{stem}.ch"));
    write_source(&file, source);
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
        .args(["build", "--emit-c", path, "--target", "c", "--output"])
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
fn initialized_state_matches_beside_reshape_in_eval_and_c() {
    let (eval, native) = eval_and_c("initialized", INITIALIZED);
    assert_eq!(eval, "main = tensor(shape=[2, 1], data=[1.0, 2.0])\n");
    assert_eq!(native, eval);
}

#[test]
fn literal_state_and_no_reshape_controls_still_execute() {
    let literal = INITIALIZED.replace("init(0i64)", "State { seed: 0i64 }");
    let (eval, native) = eval_and_c("literal", &literal);
    assert_eq!(eval, "main = tensor(shape=[2, 1], data=[1.0, 2.0])\n");
    assert_eq!(native, eval);
    let reshaped = "[1.0f32, 2.0f32] |> to_tensor |> reshape([2i64, 1i64])";
    assert!(
        INITIALIZED.contains(reshaped),
        "fixture no longer has the reshape control"
    );
    let plain = INITIALIZED
        .replace("tensor[2, 1, f32]", "tensor[2, f32]")
        .replace(reshaped, "[1.0f32, 2.0f32] |> to_tensor");
    assert!(
        !plain.contains("|> reshape("),
        "no-reshape control still reshapes"
    );
    let (eval, native) = eval_and_c("plain", &plain);
    assert_eq!(eval, "main = tensor(shape=[2], data=[1.0, 2.0])\n");
    assert_eq!(native, eval);
}

// [04-PAT-2], [05-OP-68]: a host-selected constructor chooses precisely one
// arm. A trap in the other arm must not be hoisted into a staged source.
fn mode_source(selected: bool) -> String {
    format!(
        r#"type Mode = | Keep | Reject
def mode(n: i64) -> Mode = if eq(n, 2i64) then Keep else Reject
def apply(x: tensor[2, 1, f32], m: Mode) -> tensor[2, 1, f32] =
  match m with {{
    | Keep => x
    | Reject => fail("selected mode rejected")
  }}
def main() -> tensor[2, 1, f32] = {{
  x = reshape(to_tensor([1.0f32, 2.0f32]), [2i64, 1i64])
  selected = mode({selector})
  apply(x, selected)
}}
"#,
        selector = if selected {
            "numel(x)"
        } else {
            "sub(numel(x), 1i64)"
        }
    )
}

#[test]
fn initialized_mode_does_not_evaluate_the_untaken_trapping_arm() {
    let (eval, native) = eval_and_c("keep", &mode_source(true));
    assert_eq!(eval, "main = tensor(shape=[2, 1], data=[1.0, 2.0])\n");
    assert_eq!(native, eval);
}

fn assert_failure_in_both_lanes(source: &str, expected: &str) {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("reject.ch");
    write_source(&file, source);
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .arg("check")
        .arg(&file)
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert!(checked.status.success(), "{report}");
    assert_eq!(report["score"], 1.0);
    assert_eq!(report["errors"], serde_json::json!([]));
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
        .arg("--emit-c")
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
    assert!(common::link_generated(&out, "reject.c", "reject").success());
    let native = std::process::Command::new(out.join("reject"))
        .current_dir(&out)
        .output()
        .unwrap();
    for output in [eval, native] {
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn initialized_mode_preserves_the_selected_arm_trap_in_both_lanes() {
    assert_failure_in_both_lanes(&mode_source(false), "selected mode rejected");
}

#[test]
fn returned_state_can_be_matched_again_without_losing_progression() {
    let source = INITIALIZED.replace("(x, State { seed })", "(x, State { seed: add(seed, 1i64) })")
        .replace("  pair.0", "  next = wrapped(pair.0, pair.1)\n  match next.1 with { | State { seed } => mul(next.0, insert(insert(scalar_to_tensor(cast(seed, f32)), 0i32, 2i64), 1i32, 1i64)) }");
    let (eval, native) = eval_and_c("resume", &source);
    assert_eq!(eval, "main = tensor(shape=[2, 1], data=[2.0, 4.0])\n");
    assert_eq!(native, eval);
}

#[test]
fn invalid_projection_stays_a_type_error_before_either_execution_lane() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("bad_projection.ch");
    write_source(&file, &INITIALIZED.replace("pair.0", "pair.2"));
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .arg("check")
        .arg(&file)
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert!(!checked.status.success());
    assert!(
        report["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|error| error["kind"] == "TupleIndexOutOfBounds"),
        "{report}"
    );
    for args in [vec!["eval", "--file"], vec!["build"]] {
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .args(args)
            .arg(&file)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("TupleIndexOutOfBounds"));
    }
}

// A static constructor does not make its runtime guard static. Both guard
// outcomes must retain declaration order instead of lowering the first body.
#[test]
fn guarded_static_constructor_keeps_both_runtime_selections() {
    for size in [1, 2] {
        let source = format!(
            r#"type Mode = | Keep
def apply(x: tensor[2, 1, f32], n: i64) -> tensor[2, 1, f32] =
  match Keep with {{
    | Keep if eq(n, 2i64) => x
    | Keep => add(x, x)
  }}
def main() -> tensor[2, 1, f32] = {{
  x = reshape(to_tensor([1.0f32, 2.0f32]), [2i64, 1i64])
  n = sub(numel(x), {subtract}i64)
  apply(x, n)
}}
"#,
            subtract = 2 - size
        );
        let (eval, native) = eval_and_c("guarded", &source);
        let values = if size == 2 { "1.0, 2.0" } else { "2.0, 4.0" };
        assert_eq!(
            eval,
            format!("main = tensor(shape=[2, 1], data=[{values}])\n")
        );
        assert_eq!(native, eval);
    }
}

// [04] result claims remain enforced after host-control admission. These
// are the two axes recorded for the shipped initialized-State example.
fn claimed_state_source(a: usize, b: usize) -> String {
    let declarations = r#"type State =
  | State { seed: i64 }
def init(seed: i64) -> State = State { seed }
def wrapped[a, b](x: tensor[a, b, f32], state: State) -> (tensor[a, b, f32], State) =
  match state with {
    | State { seed } => (x, State { seed })
  }
def claimed[a, b](input: tensor[a, b, f32]) -> tensor[2, 1, f32] = {
  x = reshape(input, [add(shape(input, 0i32), 0i64), add(shape(input, 1i32), 0i64)])
  pair = wrapped(x, init(0i64))
  pair.0
}
"#;
    let values = vec!["1.0f32"; a * b].join(", ");
    format!(
        "{declarations}def main() = claimed(reshape(to_tensor([{values}]), [{a}i64, {b}i64]))\n"
    )
}

#[test]
fn initialized_state_accepts_agreeing_runtime_result_claims() {
    let (eval, native) = eval_and_c("matching_claim", &claimed_state_source(2, 1));
    assert_eq!(eval, "main = tensor(shape=[2, 1], data=[1.0, 1.0])\n");
    assert_eq!(native, eval);
}

#[test]
fn initialized_state_rejects_each_false_runtime_result_axis() {
    for (a, b, message) in [
        (3, 1, "claimed = 2, load axis 0 = 3"),
        (2, 2, "claimed = 1, load axis 1 = 2"),
    ] {
        assert_failure_in_both_lanes(&claimed_state_source(a, b), message);
    }
}
