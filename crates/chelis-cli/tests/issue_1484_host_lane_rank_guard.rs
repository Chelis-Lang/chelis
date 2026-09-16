//! chelis#1484: the HOST-VALUE lane's elementwise operand rank guard.
//!
//! chelis#664 (PR #663) and chelis#668 (PR #1463) added operand-agreement
//! guards to the tensor-DAG emitter, `crates/chelis-backend-c/src/emit.rs`.
//! The host-value emitter, `crates/chelis-backend-c/src/host_emit.rs`, is a
//! different lane and had no operand comparison at all: it allocated the
//! result at the LHS rank and then fed the TARGET's index vector into each
//! operand's strides. An elementwise call one of whose operands carries an
//! IO effect routes there, so `add(debug(e), s)` for a rank-2 `e` and a
//! rank-1 `s` compiled and printed an invented `[2, 6]` broadcast while
//! `chelis eval` rejected the same program.
//!
//! The three programs below are the exact instances recorded in chelis#1484.
//! Each one is asserted in both lanes: `chelis eval` must reject with the
//! shape-mismatch diagnostic, and the compiled C must abort with the guard
//! text rather than exit 0 with values or die on a signal. The fourth test
//! is the matching-rank control that must keep running.
//!
//! Spec authority. `spec/05-risc-primitives.md` §1.2: "All operands must have
//! matching dimensions. No implicit rank extension, no implicit size
//! expansion. Use `expand` explicitly. This is a hard rule." §2.1 states it in
//! the signatures themselves: every binary elementwise primitive, `add` and
//! `max_elem` among them, is `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]`,
//! one `D` for both operands. §2.4 already requires the runtime backstop of
//! the other emitter lane: "same-shape elementwise ops guard operand-shape
//! agreement at equal rank whenever a non-static extent is involved
//! (chelis#664; rank-0 scalar operands are the backend's broadcast idiom and
//! are exempt)." This file adds no semantic rule; it pins the same loud
//! failure in the lane that did not have one.

use std::fs;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::tempdir;

mod common;

/// The six-element f32 input the chelis#1484 reproducers use.
fn six_f32() -> String {
    (1..=6)
        .map(|v| format!("cast({v}.0, f32)"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The chelis#1484 program shape, parameterized on its final expression.
///
/// `s` is a runtime-strided rank-1 `[3]` view and `e` a rank-2 `[2, 6]`
/// expansion of the same rank-1 input. The parameter annotation (rather than
/// a `sig`) is the exact spelling recorded in the issue.
fn program(expression: &str) -> String {
    format!(
        "module Repro.HostRank\n\
         \n\
         def f(x: tensor[n, f32]) = {{\n\
         \x20 s = stride(x, cast(2, i64))\n\
         \x20 e = insert(x, cast(0, i32), cast(2, i64))\n\
         \x20 {expression}\n\
         }}\n\
         \n\
         out = f(to_tensor([{}]))\n",
        six_f32()
    )
}

fn run_eval(source: &str, stem: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write .ch source");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval")
}

/// `chelis build --target c`, link the emitted translation unit with the
/// host C compiler, run it, and return the process output. `None` when the
/// machine has no host C compiler, so the build/run leg skips cleanly.
fn build_link_run(source: &str, stem: &str) -> Option<std::process::Output> {
    if !common::gcc_available() {
        eprintln!("skipping the build lane for `{stem}`: no host C compiler on PATH");
        return None;
    }
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join(format!("{stem}-out"));
    common::write_file(&path, source);

    let built = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build C");
    if !built.status.success() {
        // The checker refused before emission, which is chelis#1484 satisfied
        // rather than skipped: there is no compiled lane to guard because the
        // program never reaches one. Since chelis#1277 fixed `insert`'s rank at
        // the call, that is where a provable rank disagreement is caught.
        let stderr = String::from_utf8_lossy(&built.stderr);
        assert!(
            stderr.contains("tensor rank mismatch")
                || stderr.contains("tensor shapes must match for elementwise op"),
            "`{stem}`: the build refused for a reason other than the rank \
             disagreement this row is about: {stderr}"
        );
        return None;
    }

    let status = common::link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(status.success(), "link failed for `{stem}`: {status}");
    Some(
        StdCommand::new(out_dir.join(stem))
            .output()
            .expect("compiled binary should run"),
    )
}

/// Both lanes must reject: `chelis eval` with its shape diagnostic, the
/// compiled binary with a non-zero exit carrying `guard_needle` on stderr.
/// Exiting 0 with values is the chelis#1484 defect; dying on a signal is the
/// same defect reaching the indexing it was supposed to be stopped before.
fn assert_both_lanes_reject(source: &str, stem: &str, guard_needle: &str) {
    let eval_out = run_eval(source, stem);
    assert!(
        !eval_out.status.success(),
        "{stem}: eval must reject; stdout={}",
        String::from_utf8_lossy(&eval_out.stdout)
    );
    let eval_err = String::from_utf8_lossy(&eval_out.stderr).into_owned();
    // Either diagnostic satisfies chelis#1484, whose defect is exiting 0 with
    // values rather than which stage refuses. Since chelis#1277 gave `insert` a
    // single form, the operand's rank is fixed at the call, so the checker can
    // prove the disagreement and rejects before the elementwise op is reached.
    // `spec/04` \u{00a7}4.7 requires exactly that: a rank disagreement the checker can
    // prove is a check error, not a deferred one. The elementwise message is
    // still accepted for the operands whose rank is not fixed until then.
    assert!(
        eval_err.contains("tensor shapes must match for elementwise op")
            || eval_err.contains("tensor rank mismatch"),
        "{stem}: eval must name the shape or rank mismatch; stderr={eval_err}"
    );

    let Some(run) = build_link_run(source, stem) else {
        return;
    };
    assert!(
        !run.status.success(),
        "{stem}: the compiled binary must abort, never compute over rank-divergent \
         operands; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains(guard_needle),
        "{stem}: the abort must name the guard ({guard_needle}); stderr={stderr}"
    );
    assert_ne!(
        run.status.code(),
        Some(139),
        "{stem}: the guard must stop the program before the out-of-bounds read, \
         not let it segfault; stderr={stderr}"
    );
}

/// chelis#1484 REGRESSION TEST (headline instance): red before, green after.
/// On the pre-fix tree this program exited 0 and printed
/// `tensor(shape=[2, 6], data=[2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9])`, an
/// invented broadcast of a `[2, 6]` operand against a `[3]` one.
#[test]
fn issue_1484_add_host_lane_rank_mismatch_aborts() {
    assert_both_lanes_reject(
        &program("add(debug(e), s)"),
        "hostrank_add",
        "elementwise operand rank mismatch",
    );
}

/// chelis#1484 REGRESSION TEST (second instance): red before, green after.
/// `max_elem` is emitted by a different host function than `add`, so it
/// needed its own guard call. Pre-fix this exited 0 with a wrong value.
#[test]
fn issue_1484_max_elem_host_lane_rank_mismatch_aborts() {
    assert_both_lanes_reject(
        &program("max_elem(debug(e), s)"),
        "hostrank_max_elem",
        "elementwise operand rank mismatch",
    );
}

/// chelis#1484 REGRESSION TEST (third instance): red before, green after.
/// With the operands the other way round the result allocates at the rank-1
/// operand, so the rank-2 operand's stride lookup read an uninitialized
/// index slot: this spelling exited 139 (SIGSEGV) before the fix. The
/// `assert_ne!` on 139 inside the helper is what pins that half.
#[test]
fn issue_1484_reversed_operand_order_host_lane_rank_mismatch_aborts() {
    assert_both_lanes_reject(
        &program("add(s, debug(e))"),
        "hostrank_add_reversed",
        "elementwise operand rank mismatch",
    );
}

/// chelis#1484 DISPOSITION LOCK: green before the fix and green after. Its
/// job is to pin that the guard does not false-abort the agreeing host-lane
/// case and does not change its values. `s` is the same runtime-strided
/// rank-1 `[3]` view, so this exercises the guard's non-static-extent path
/// rather than a fully static one.
#[test]
fn issue_1484_matching_rank_host_lane_still_runs() {
    let source = program("add(debug(s), s)");
    let stem = "hostrank_match";

    let eval_out = run_eval(&source, stem);
    assert!(
        eval_out.status.success(),
        "{stem}: eval must accept the matching-rank program; stderr={}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = common::parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout), "out");
    assert_eq!(
        eval_values,
        vec![2.0, 6.0, 10.0],
        "{stem}: eval returned the wrong values"
    );

    let Some(run) = build_link_run(&source, stem) else {
        return;
    };
    assert!(
        run.status.success(),
        "{stem}: the guard must not abort agreeing operands; stderr={}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let c_values = common::parse_tensor_data(&stdout, "out");
    assert_eq!(
        c_values, eval_values,
        "{stem}: the C lane disagreed with eval; stdout={stdout}"
    );
}
