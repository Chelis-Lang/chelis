//! chelis#1948: declared-result guards on same-shape operations belong to the
//! operation that produced the returned value.
//!
//! `spec/04-type-system.md` section 4.7 fixes both ownership and order. For a
//! returned `add`, `add` is the primitive that produced the value, so its
//! declared-result guard runs at that operation after the independent
//! positive-rank operand-shape agreement obligation. It is not attributed to
//! whichever operand `shape_preserving` or `op_computed_axis_origin` happens
//! to encounter first. Rank-0 scalar operands remain outside shape agreement.
//!
//! These tests are deliberately separate from `runtime_extent_slice_b`: the
//! active stride stack owns that target while this preparation branch remains
//! based on main. They are failing test-first rows until the #1948 production
//! stack can touch lowering, axis-source derivation, Eval and C emission.

mod common;

use assert_cmd::Command;
use common::{gcc_available, link_generated, write_file};
use std::process::Command as StdCommand;
use tempfile::TempDir;

#[derive(Debug)]
struct LaneResult {
    success: bool,
    text: String,
}

fn combined(output: &std::process::Output) -> String {
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    text
}

fn eval_result(dir: &TempDir, stem: &str, source: &str) -> LaneResult {
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--allow-style-violations",
            "--file",
            path.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("run eval");
    LaneResult {
        success: output.status.success(),
        text: combined(&output),
    }
}

fn c_result(dir: &TempDir, stem: &str, source: &str) -> LaneResult {
    assert!(
        gcc_available(),
        "the #1948 acceptance target requires an executed C lane"
    );
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join(format!("{stem}-out"));
    write_file(&path, source);
    let built = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            "--allow-style-violations",
            path.to_str().expect("UTF-8 path"),
            "--target",
            "c",
            "-o",
            out_dir.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("build generated C");
    assert!(
        built.status.success(),
        "{stem}: build must reach the executable lane: {}",
        combined(&built)
    );
    let linked = link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(
        linked.success(),
        "{stem}: generated C must compile and link"
    );
    let output = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run generated binary");
    LaneResult {
        success: output.status.success(),
        text: combined(&output),
    }
}

fn both_lanes(dir: &TempDir, stem: &str, source: &str) -> [(&'static str, LaneResult); 2] {
    [
        ("eval", eval_result(dir, &format!("{stem}_eval"), source)),
        ("c", c_result(dir, &format!("{stem}_c"), source)),
    ]
}

fn assert_add_result_claim(dir: &TempDir, stem: &str, source: &str) {
    let context = "extent `2`: claimed = 2, add axis 0 = 3";
    let trap = "numeric trap: domain in add at i64";
    for (lane, result) in both_lanes(dir, stem, source) {
        assert!(
            !result.success,
            "{lane}: a declared extent 2 over add's extent 3 must trap: {}",
            result.text
        );
        assert!(
            result.text.contains(context),
            "{lane}: the returned-value producer owns the context `{context}`: {}",
            result.text
        );
        assert!(
            result.text.lines().any(|line| line == trap),
            "{lane}: [04-NUM-9]'s exact line is `{trap}`: {}",
            result.text
        );
        assert!(
            !result
                .text
                .contains("numeric trap: domain in shrink at i64"),
            "{lane}: operand order or origin traversal must not rename add's guard: {}",
            result.text
        );
        assert!(
            !result.text.contains("shape=[3]"),
            "{lane}: the undeclared result must not escape: {}",
            result.text
        );
    }
}

fn operand_one_source(declared: usize, reverse: bool) -> String {
    let body = if reverse {
        "add(shrink(x, [[1i64, shape(x, 0i32)]]), w)"
    } else {
        "add(w, shrink(x, [[1i64, shape(x, 0i32)]]))"
    };
    format!(
        "def f[n](w: tensor[*, f32], x: tensor[n, f32]) -> tensor[{declared}, f32] =\n  \
         {body}\n\
         out = f(\n  \
         to_tensor([10.0f32, 20.0f32, 30.0f32]),\n  \
         to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]),\n\
         )\n"
    )
}

/// Exact issue witness: the only op-computed path is operand 1, while operand
/// 0 is an unconstrained same-shape tensor. The claim must not disappear.
#[test]
fn the_operand_one_witness_is_guarded_by_add() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_add_result_claim(&dir, "operand_one", &operand_one_source(2, false));
}

/// Reversed twin: finding an op-computed path first must not rename the
/// returned `add` to that operand's `shrink`.
#[test]
fn reversing_operands_does_not_rename_adds_result_claim() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_add_result_claim(&dir, "operand_zero", &operand_one_source(2, true));
}

/// Positive control: the same runtime output under an agreeing declaration
/// executes and preserves identical bytes on Eval and generated C.
#[test]
fn an_agreeing_same_shape_result_claim_executes_exactly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = operand_one_source(3, false);
    let [(_, eval), (_, compiled)] = both_lanes(&dir, "agreeing", &source);
    assert!(eval.success, "eval: {}", eval.text);
    assert!(compiled.success, "c: {}", compiled.text);
    assert_eq!(
        eval.text, compiled.text,
        "the agreeing result is byte-identical on both lanes"
    );
    assert_eq!(
        eval.text,
        "out = tensor(shape=[3], data=[12.0, 23.0, 34.0])\n"
    );
}

/// The elementwise operation has no result until its positive-rank operands
/// agree. Their independent shape guard therefore wins before the declared
/// result claim; #1948 must not mask it with an `add` extent trap.
#[test]
fn runtime_operand_disagreement_precedes_the_result_claim() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f[n](w: tensor[*, f32], x: tensor[n, f32]) -> tensor[2, f32] =\n  \
                  add(w, shrink(x, [[2i64, shape(x, 0i32)]]))\n\
                  out = f(\n  \
                  to_tensor([10.0f32, 20.0f32, 30.0f32]),\n  \
                  to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]),\n\
                  )\n";
    for (lane, result) in both_lanes(&dir, "operand_disagreement", source) {
        assert!(!result.success, "{lane}: shape disagreement must fail");
        let expected = match lane {
            "eval" => "tensor shapes must match for elementwise op, got [3] vs [2]",
            "c" => "elementwise operand shape mismatch",
            _ => unreachable!(),
        };
        assert!(
            result.text.contains(expected),
            "{lane}: expected the operand guard `{expected}`: {}",
            result.text
        );
        assert!(
            !result.text.contains("extent `2`")
                && !result.text.contains("numeric trap: domain in add at i64"),
            "{lane}: the later result claim must not pre-empt operand agreement: {}",
            result.text
        );
    }
}

/// Two operand edges reaching the same computed path still create one result
/// claim owned by the returned `add`, not one claim per traversal.
#[test]
fn repeated_paths_to_one_candidate_produce_one_add_claim() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f[n](x: tensor[n, f32]) -> tensor[2, f32] = {\n  \
                  y = shrink(x, [[1i64, shape(x, 0i32)]])\n  \
                  add(y, y)\n\
                  }\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\n";
    assert_add_result_claim(&dir, "repeated_path", source);
}

/// Distinct op-computed operand paths are all members of `add`'s agreement
/// relation. Neither path is selected as the declared-result owner.
#[test]
fn distinct_candidate_paths_do_not_create_source_order_attribution() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[2, f32] =\n  \
                  add(\n    \
                  shrink(x, [[1i64, shape(x, 0i32)]]),\n    \
                  shrink(y, [[1i64, shape(y, 0i32)]]),\n  \
                  )\n\
                  out = f(\n  \
                  to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]),\n  \
                  to_tensor([5.0f32, 6.0f32, 7.0f32, 8.0f32]),\n\
                  )\n";
    assert_add_result_claim(&dir, "distinct_paths", source);
}

/// A same-shape result does not need an op-computed operand origin. Its own
/// agreement relation supplies the output extent, so two wildcard inputs
/// cannot make the claim disappear.
#[test]
fn no_claim_capable_operand_origin_does_not_drop_the_result_claim() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, f32], y: tensor[*, f32]) -> tensor[2, f32] = add(x, y)\n\
                  out = f(\n  \
                  to_tensor([1.0f32, 2.0f32, 3.0f32]),\n  \
                  to_tensor([4.0f32, 5.0f32, 6.0f32]),\n\
                  )\n";
    assert_add_result_claim(&dir, "no_origin", source);
}

/// Static contradiction remains a checker disposition; the runtime mechanism
/// must not turn a known-false declaration into a dynamic guard.
#[test]
fn a_statically_refuted_same_shape_result_is_a_dimension_mismatch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("static_refutation.ch");
    write_file(
        &path,
        "def f(w: tensor[3, f32], x: tensor[4, f32]) -> tensor[2, f32] =\n  \
         add(w, shrink(x, [[1i64, 4i64]]))\n\
         out = f(\n  \
         to_tensor([10.0f32, 20.0f32, 30.0f32]),\n  \
         to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]),\n\
         )\n",
    );
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "check",
            "--allow-style-violations",
            path.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("run check");
    let text = combined(&output);
    assert_eq!(
        output.status.code(),
        Some(2),
        "a graph-fixed result disagreement is a type error: {text}"
    );
    assert!(
        text.contains("\"kind\":\"DimensionMismatch\""),
        "the checker reports the owning diagnostic kind: {text}"
    );
}
