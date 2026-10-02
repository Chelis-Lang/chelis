//! chelis#469: an `expand`/`insert` size of any provenance executes, and
//! `chelis eval` agrees with compiled C.
//!
//! `spec/04-type-system.md` section 4.7.2 admits any `i64` size and forbids
//! rejecting an extent "because of its provenance"; section 4.7.4 says eval
//! and C "execute the same graph and checks". Each positive row runs one
//! program through `chelis check`, `chelis eval --file`, and `chelis build
//! --target c` plus the produced executable, and requires the two lanes to
//! print the same bytes. Before chelis#469 the checker rejected every
//! scalar-parameter spelling, and lowering rejected a bare name with no tensor
//! source even after the checker had accepted it (a top-level binding).
//!
//! The negative rows are the section 4.7.2 failures that must keep failing,
//! now reachable from these spellings: a runtime negative size traps `Domain`,
//! a runtime size disagreeing with a literal result claim traps `Domain`, a
//! claim the call's constant argument refutes is rejected when the activation
//! is lowered, and two different runtime extents meeting elementwise trap.
//! Each is required to render identically on both lanes.
//!
//! The checker-side rows are
//! `crates/chelis-types/tests/issue_469_runtime_scalar_extent_check.rs`.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::{Command as StdCommand, Output};
use tempfile::{TempDir, tempdir};

/// A runtime `i64` the compiler cannot fold: the sum of a tensor's elements.
/// `RUNTIME_FOUR` is 4, and every row that needs an extent the lowering cannot
/// see uses it, so a claim is checked at run time rather than refuted early.
const RUNTIME_FOUR: &str = "tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0))";

/// The operand and size spellings, once per operation. `expand` broadcasts an
/// existing unit axis, so its operand is `[1, 2]`; `insert` adds one, so its
/// operand is `[2]`. Both print `[3, 2]` for a size of 3.
const OPERATIONS: [(&str, &str); 2] = [
    ("expand", "to_tensor([[1i64, 2i64]])"),
    ("insert", "to_tensor([1i64, 2i64])"),
];

const THREE_BY_TWO: &str = "tensor(shape=[3, 2], data=[1, 2, 1, 2, 1, 2])";

fn cli() -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

fn combined(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn write_fixture(dir: &TempDir, stem: &str, source: &str) -> std::path::PathBuf {
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write fixture");
    path
}

fn check(path: &Path) -> Output {
    cli()
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check")
}

fn eval(path: &Path) -> Output {
    cli()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval")
}

/// Build natively for C and run the executable. A build failure is returned
/// as the build's own output, so a row can tell a lowering rejection from a
/// run-time trap.
fn build_and_run(dir: &TempDir, stem: &str, path: &Path) -> Output {
    let out_dir = dir.path().join(format!("{stem}-out"));
    let build = cli()
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    if !build.status.success() {
        return build;
    }
    StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run the compiled executable")
}

/// Check, evaluate and build `source`; require both lanes to succeed, print
/// the same bytes, and print `expected` among them.
fn assert_lanes_agree(stem: &str, source: &str, expected: &str) {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(&dir, stem, source);
    let checked = check(&path);
    assert!(
        checked.status.success(),
        "{stem}: check must accept a runtime i64 size: {}",
        combined(&checked)
    );
    let evaluated = eval(&path);
    assert!(
        evaluated.status.success(),
        "{stem}: eval must execute: {}",
        combined(&evaluated)
    );
    let compiled = build_and_run(&dir, stem, &path);
    assert!(
        compiled.status.success(),
        "{stem}: the C build must build and run: {}",
        combined(&compiled)
    );
    let eval_out = String::from_utf8_lossy(&evaluated.stdout).to_string();
    let c_out = String::from_utf8_lossy(&compiled.stdout).to_string();
    assert_eq!(eval_out, c_out, "{stem}: eval and C print the same bytes");
    assert!(
        eval_out.contains(expected),
        "{stem}: expected `{expected}` in {eval_out}"
    );
}

/// Check accepts `source`; both lanes then fail at run time with the same
/// rendering, which contains every line of `lines`.
fn assert_lanes_trap_identically(stem: &str, source: &str, lines: &[&str]) {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(&dir, stem, source);
    let checked = check(&path);
    assert!(
        checked.status.success(),
        "{stem}: the failure is a run-time one, so check accepts: {}",
        combined(&checked)
    );
    let evaluated = eval(&path);
    let compiled = build_and_run(&dir, stem, &path);
    let eval_out = combined(&evaluated);
    let c_out = combined(&compiled);
    for (lane, output, text) in [("eval", &evaluated, &eval_out), ("c", &compiled, &c_out)] {
        assert!(!output.status.success(), "{stem}: {lane} must trap: {text}");
        assert!(
            !text.contains("out = "),
            "{stem}: {lane} must not print the trapped binding: {text}"
        );
        for line in lines {
            assert!(
                text.contains(line),
                "{stem}: {lane} must print `{line}`: {text}"
            );
        }
    }
    assert_eq!(
        eval_out.trim_end(),
        c_out.trim_end(),
        "{stem}: the two lanes render the trap identically"
    );
}

fn domain_trap_line(op: &str) -> String {
    format!("numeric trap: domain in {op} at i64")
}

#[test]
fn a_bare_i64_parameter_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_param"),
            &format!("def f(k: i64) = {op}({operand}, 0, k)\nout = f(3i64)\n"),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_cast_i32_parameter_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_cast"),
            &format!("def f(j: i32) = {op}({operand}, 0, cast(j, i64))\nout = f(3i32)\n"),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_let_bound_parameter_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_let"),
            &format!("def f(k: i64) = {{\n  m = k\n  {op}({operand}, 0, m)\n}}\nout = f(3i64)\n"),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn an_arithmetic_size_over_a_parameter_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_arith"),
            &format!("def f(k: i64) = {op}({operand}, 0, add(k, 1i64))\nout = f(2i64)\n"),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_user_function_result_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_userfn"),
            &format!(
                "def g(k: i64) -> i64 = add(k, 1i64)\n\
                 def f(k: i64) = {op}({operand}, 0, g(k))\n\
                 out = f(2i64)\n"
            ),
            THREE_BY_TWO,
        );
    }
}

/// A top-level binding whose value the checker folds. Before chelis#469 check
/// and eval accepted it and only the C build rejected it, because lowering
/// found no tensor source for the global's name.
#[test]
fn a_top_level_static_binding_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_toplevel_static"),
            &format!("h = add(1i64, 2i64)\nout = {op}({operand}, 0, h)\n"),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_top_level_runtime_binding_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_toplevel_runtime"),
            &format!(
                "def g(k: i64) -> i64 = add(k, 1i64)\n\
                 h = g(2i64)\n\
                 out = {op}({operand}, 0, h)\n"
            ),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_tuple_projection_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_tuple"),
            &format!("def f(t: (i64, i64)) = {op}({operand}, 0, t.0)\nout = f((3i64, 9i64))\n"),
            THREE_BY_TWO,
        );
    }
}

/// `expand.sourceless_size.pipe_position` and its lane twin in
/// `scripts/runtime_extent_oracle.py`: the size reaches `expand` through a
/// pipe stage, which chelis#1791 made the same tree as the direct spelling.
#[test]
fn a_pipe_position_runtime_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_pipe"),
            &format!(
                "def f(j: i32) = {{\n  m = j |> cast(i64)\n  {operand} |> {op}(0i32, m)\n}}\n\
                 out = f(3i32)\n"
            ),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_zero_runtime_size_executes_on_both_lanes() {
    assert_lanes_agree(
        "insert_zero",
        "def f(k: i64) = insert(to_tensor([1i64, 2i64]), 0, k)\nout = f(0i64)\n",
        "tensor(shape=[0, 2], data=[])",
    );
}

/// The Std.Datetime shape that motivated the work: a table whose only extent
/// source is a scalar horizon, scanned and reduced.
#[test]
fn a_horizon_sized_table_executes_on_both_lanes() {
    assert_lanes_agree(
        "horizon_table",
        "def table(horizon: i64) = cumsum(expand(to_tensor([1i64]), 0, horizon), 0)\n\
         def total(horizon: i64) -> i64 = \
         tensor_to_scalar(sum(expand(to_tensor([1i64]), 0, horizon), 0))\n\
         t = table(5i64)\n\
         s = total(5i64)\n",
        "t = tensor(shape=[5], data=[1, 2, 3, 4, 5])\ns = 5",
    );
}

/// One runtime extent reaching two operands, then a callee generic over it.
#[test]
fn a_runtime_extent_composes_on_both_lanes() {
    assert_lanes_agree(
        "compose",
        "def g[n](t: &tensor[n, 2, i64]) -> tensor[n, 2, i64] = mul(t, t)\n\
         def f(k: i64) = g(add(insert(to_tensor([1i64, 2i64]), 0, k), \
         insert(to_tensor([1i64, 2i64]), 0, k)))\n\
         out = f(3i64)\n",
        "tensor(shape=[3, 2], data=[4, 16, 4, 16, 4, 16])",
    );
}

#[test]
fn a_runtime_negative_size_traps_domain_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_trap_identically(
            &format!("{op}_negative"),
            &format!("def f(k: i64) = {op}({operand}, 0, k)\nout = f(sub({RUNTIME_FOUR}, 5i64))\n"),
            &[
                "Domain: expansion axis or extent outside domain",
                &domain_trap_line(op),
            ],
        );
    }
}

#[test]
fn a_runtime_size_disagreeing_with_a_literal_claim_traps_domain_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_trap_identically(
            &format!("{op}_claim"),
            &format!(
                "def f(k: i64) -> tensor[3, 2, i64] = {op}({operand}, 0, k)\n\
                 out = f({RUNTIME_FOUR})\n"
            ),
            &["claimed = 3", &domain_trap_line(op)],
        );
    }
}

/// The agreeing control for the claim row: the same declaration over a runtime
/// size equal to its claim executes, so the trap above is the claim's.
#[test]
fn a_runtime_size_agreeing_with_a_literal_claim_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_claim_ok"),
            &format!(
                "def f(k: i64) -> tensor[4, 2, i64] = {op}({operand}, 0, k)\n\
                 out = f({RUNTIME_FOUR})\n"
            ),
            "tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        );
    }
}

/// A constant argument makes the inlined body's extent a literal, so the
/// declared claim is refuted when the activation is lowered: section 4.7's
/// "A violation proven from literals is a type error", identical on both
/// lanes and carrying no run-time trap.
#[test]
fn a_claim_a_constant_argument_refutes_is_rejected_identically_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        let dir = tempdir().expect("tempdir");
        let path = write_fixture(
            &dir,
            &format!("{op}_refuted"),
            &format!("def f(k: i64) -> tensor[3, 2, i64] = {op}({operand}, 0, k)\nout = f(4i64)\n"),
        );
        let evaluated = eval(&path);
        let compiled = build_and_run(&dir, &format!("{op}_refuted"), &path);
        let eval_out = combined(&evaluated);
        let c_out = combined(&compiled);
        for (lane, output, text) in [("eval", &evaluated, &eval_out), ("c", &compiled, &c_out)] {
            assert_eq!(output.status.code(), Some(1), "{op}: {lane}: {text}");
            assert!(
                text.contains(
                    "declares extent 3 at result axis 0, but the inlined body produces 4"
                ),
                "{op}: {lane}: {text}"
            );
            assert!(!text.contains("numeric trap"), "{op}: {lane}: {text}");
        }
        assert_eq!(eval_out, c_out, "{op}: one rendering on both lanes");
    }
}

#[test]
fn two_different_runtime_extents_meeting_elementwise_trap_on_both_lanes() {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(
        &dir,
        "mismatch",
        &format!(
            "def f(k: i64, j: i64) = add(insert(to_tensor([1i64, 2i64]), 0, k), \
             insert(to_tensor([1i64, 2i64]), 0, j))\n\
             out = f(3i64, {RUNTIME_FOUR})\n"
        ),
    );
    let checked = check(&path);
    assert!(
        checked.status.success(),
        "the disagreement is a run-time one, so check accepts: {}",
        combined(&checked)
    );
    let evaluated = eval(&path);
    let compiled = build_and_run(&dir, "mismatch", &path);
    for (lane, output) in [("eval", &evaluated), ("c", &compiled)] {
        let text = combined(output);
        assert!(!output.status.success(), "{lane} must trap: {text}");
        assert!(!text.contains("out = "), "{lane}: {text}");
    }
}
