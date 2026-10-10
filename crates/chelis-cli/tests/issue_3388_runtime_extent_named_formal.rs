//! chelis#3388: a runtime extent reaching a named tensor formal on the C lane.
//!
//! `spec/04-type-system.md` section 4.1 lets a concrete literal satisfy a
//! concrete-but-named dimension slot at a call site, and section 4.7 makes a
//! runtime `reshape` target a runtime extent, which unification admits
//! wherever a literal is admitted. The checker and `chelis eval` accepted
//! these programs while `chelis build --target c` refused them in ownership
//! lowering with an internal call-argument type mismatch. Each positive row
//! asserts both lanes print the same result; the negative rows pin that a
//! disagreeing runtime extent still traps identically on both lanes and that
//! a distinct concrete axis name is still a check-time error.

#[path = "common/host_effect_parity.rs"]
mod parity;

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

/// The issue's reproducer: a generic def returns a free-dimension ADT, and
/// its tensor argument's extents are runtime values.
const FREE_DIMENSION_ADT: &str = "module Wild3
type Holder =
  | Holder { features: tensor[n, d, f32], rows: i64 }
def hold[n, d](features: tensor[n, d, f32], rows: i64) -> Holder = Holder { features, rows }
def total(h: Holder) -> f32 =
  match h with {
    | Holder { features, rows } => {
    _ = rows
    tensor_to_scalar(sum(sum(features, 0i32), 0i32))
  }
  }
def main() -> f32 = {
  rows = 2i64
  cols = 2i64
  x = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), [rows, cols])
  total(hold(x, rows))
}
";

/// The same defect without an ADT or a generic: a parameter whose axes are
/// concrete names receives a runtime-extent argument.
const CONCRETE_NAMED_PARAMETER: &str = "def total(features: tensor[batch, feat, f32]) -> f32 = tensor_to_scalar(sum(sum(features, 0i32), 0i32))
def main() -> f32 = {
  rows = 2i64
  cols = 2i64
  x = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), [rows, cols])
  total(x)
}
";

/// Two runtime extents that claim the one named axis `batch` disagree.
const DISAGREEING_EXTENTS: &str = "def both(a: tensor[batch, f32], b: tensor[batch, f32]) -> f32 = tensor_to_scalar(sum(add(a, b), 0i32))
def main() -> f32 = {
  two = 2i64
  three = 3i64
  x = reshape(to_tensor([1.0f32, 2.0f32]), [two])
  y = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32]), [three])
  both(x, y)
}
";

#[test]
fn a_generic_def_returning_a_free_dimension_adt_builds_from_a_runtime_extent() {
    let run = parity::assert_lanes_agree(FREE_DIMENSION_ADT, "wild3");
    assert_eq!(run.status, Some(0), "{run:?}");
    assert_eq!(run.stdout, "main = 10.0\n");
}

#[test]
fn a_concrete_named_parameter_admits_a_runtime_extent_on_both_lanes() {
    let run = parity::assert_lanes_agree(CONCRETE_NAMED_PARAMETER, "named_parameter");
    assert_eq!(run.status, Some(0), "{run:?}");
    assert_eq!(run.stdout, "main = 10.0\n");
}

#[test]
fn disagreeing_runtime_extents_for_one_named_axis_trap_identically() {
    let run = parity::assert_lanes_agree(DISAGREEING_EXTENTS, "disagreeing");
    assert_eq!(run.status, Some(1), "{run:?}");
    assert_eq!(run.failure, "numeric trap: domain in load at i64");
    assert_eq!(run.context, "extent `batch`: a axis 0 = 2, b axis 0 = 3");
}

#[test]
fn a_distinct_concrete_axis_name_is_still_a_check_error() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("distinct.ch");
    fs::write(
        &path,
        "def total(features: tensor[batch, feat, f32]) -> f32 = tensor_to_scalar(sum(sum(features, 0i32), 0i32))\n\
         def caller(y: tensor[rows, feat, f32]) -> f32 = total(y)\n",
    )
    .expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    let json: Value = serde_json::from_slice(&output.stdout).expect("check output is json");
    let errors = json["errors"].as_array().expect("errors array");
    assert!(
        errors.iter().any(|error| {
            error["kind"] == "DimensionMismatch"
                && error["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("expected batch, got rows"))
        }),
        "a named axis still rejects a different name: {errors:?}"
    );
}
