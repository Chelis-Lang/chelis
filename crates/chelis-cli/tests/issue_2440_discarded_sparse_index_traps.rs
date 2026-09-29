//! A discarded sparse index is still checked (chelis#2440).
//!
//! `spec/06-transformations.md` §5.2 makes a potentially trapping node an
//! observable root: "purity alone does not make a possible trap dead". The
//! index of a `gather`, of every scatter mode, and of `one_hot` is data, so
//! [05-OP-52]'s "out-of-bounds indices fail loudly" cannot be ruled out
//! statically for a non-literal one, and the checker rejects a literal one
//! before lowering. Those five operations were nonetheless outside the trap
//! seed, so an out-of-bounds index that nothing consumed was eliminated and
//! its failure did not occur — in BOTH lanes.
//!
//! Each family below is one `g` whose out-of-bounds index sits in a
//! discarded `let`, beside the same `g` with that index consumed. The two
//! must agree: whatever the consumed program does with a bad index, the
//! discarded program does too, and an in-range index returns a value in
//! both.
//!
//! ## Why this does not assert one message across both lanes
//!
//! Compiled C reports `numeric trap: domain in gather at i64`. The
//! evaluator instead panics out of a bare `assert!`, so it produces no
//! `numeric trap:` line at all. That divergence is **chelis#1636**, not this
//! issue, and it predates the seed: the CONSUMED program panics in exactly
//! the same way. Asserting a shared message here would fail for a reason
//! chelis#2440 does not own, and weakening the compiled lane's assertion to
//! match the evaluator's panic would stop pinning the message C actually
//! owes. So each lane is pinned to what it owes, and the cross-lane claim is
//! the one this issue is about: discarded behaves as consumed. When
//! chelis#1636 lands, the evaluator rows tighten without this file changing
//! shape.
mod common;

use assert_cmd::Command;
use std::path::Path;

/// The trap the COMPILED lane owes an out-of-bounds sparse index.
const C_TRAP: &str = "numeric trap: domain in gather at i64";

/// Out of bounds for a 3-element axis, and in range.
const BAD: &str = "9i64";
const GOOD: &str = "2i64";

/// The index is only ever read by a discarded binding.
const DISCARDED: &str = "def g(v: tensor[3, f32], idx: tensor[2, i64]) -> tensor[3, f32] = {
  dead = gather(v, idx, 0)
  v
}
out = g(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([0i64, INDEX]))
";

/// The same index, consumed. This is the control: it fixes what the
/// discarded program must do.
const CONSUMED: &str = "def g(v: tensor[3, f32], idx: tensor[2, i64]) -> tensor[2, f32] = {
  live = gather(v, idx, 0)
  live
}
out = g(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([0i64, INDEX]))
";

/// What one lane did with a program: it stopped, or it printed a root.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// Stopped. Carries the `numeric trap:` line when there is one, so the
    /// compiled lane's message is still pinned, and `None` for a stop
    /// without one (the evaluator's chelis#1636 panic).
    Stopped(Option<String>),
    Value(String),
}

fn chelis(directory: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(directory)
        .args(args)
        .output()
        .unwrap()
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn verdict(output: &std::process::Output) -> Verdict {
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let last = stdout
            .lines()
            .last()
            .unwrap_or_else(|| panic!("no root was printed:\n{}", text(output)));
        let (_, value) = last
            .split_once(" = ")
            .unwrap_or_else(|| panic!("`{last}` is not a printed root"));
        Verdict::Value(value.to_string())
    } else {
        let all = text(output);
        Verdict::Stopped(
            all.lines()
                .find(|line| line.starts_with("numeric trap:"))
                .map(str::to_string),
        )
    }
}

/// The file is canonical and lint-clean, so the style gate cannot be what
/// rejects it.
fn assert_canonical(directory: &Path, path: &str, source: &str) {
    let formatted = chelis(directory, &["fmt", "--inplace", path]);
    assert!(formatted.status.success(), "{}", text(&formatted));
    assert_eq!(
        std::fs::read_to_string(directory.join(path)).unwrap(),
        source,
        "the witness must already be canonical Surf"
    );
    let linted = chelis(directory, &["lint", "--check", path]);
    assert!(linted.status.success(), "{}", text(&linted));
}

fn c_verdict(directory: &Path, path: &str, stem: &str) -> Verdict {
    let out = directory.join(format!("{stem}-out"));
    let built = chelis(
        directory,
        &[
            "build",
            path,
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ],
    );
    assert!(built.status.success(), "{}", text(&built));
    assert!(common::link_generated(&out, &format!("{stem}.c"), stem).success());
    verdict(&std::process::Command::new(out.join(stem)).output().unwrap())
}

/// The eval and C verdicts of `template` with `index` in place of `INDEX`.
fn lane_verdicts(stem: &str, template: &str, index: &str) -> (Verdict, Verdict) {
    let source = template.replace("INDEX", index);
    let directory = tempfile::tempdir().unwrap();
    let path = format!("{stem}.ch");
    std::fs::write(directory.path().join(&path), &source).unwrap();
    assert_canonical(directory.path(), &path, &source);
    (
        verdict(&chelis(directory.path(), &["eval", "--file", &path])),
        c_verdict(directory.path(), &path, stem),
    )
}

#[test]
fn a_discarded_out_of_bounds_gather_stops_exactly_where_a_consumed_one_does() {
    let (discarded_eval, discarded_c) = lane_verdicts("sparse_discarded_bad", DISCARDED, BAD);
    let (consumed_eval, consumed_c) = lane_verdicts("sparse_consumed_bad", CONSUMED, BAD);

    // The control. If this stops returning a value, the rest proves nothing.
    assert!(
        matches!(consumed_eval, Verdict::Stopped(_)),
        "control: a CONSUMED out-of-bounds index must stop the evaluator, got {consumed_eval:?}"
    );

    // chelis#2440 itself: liveness must not decide whether the check runs.
    assert_eq!(
        discarded_eval, consumed_eval,
        "eval: a discarded out-of-bounds index must do what a consumed one does"
    );
    assert_eq!(
        discarded_c, consumed_c,
        "C: a discarded out-of-bounds index must do what a consumed one does"
    );

    // The compiled lane owes the exact trap line; the evaluator owes only a
    // stop until chelis#1636 gives it the same line.
    assert_eq!(
        discarded_c,
        Verdict::Stopped(Some(C_TRAP.to_string())),
        "C must report the sparse-index domain trap by name"
    );
}

#[test]
fn an_in_range_index_still_returns_a_value_in_both_lanes() {
    // The negative control for the seed: retaining the node must not turn a
    // valid program into a trap, and must not change its result.
    let (discarded_eval, discarded_c) = lane_verdicts("sparse_discarded_good", DISCARDED, GOOD);
    let want = Verdict::Value("tensor(shape=[3], data=[1.0, 2.0, 3.0])".to_string());
    assert_eq!(discarded_eval, want, "eval, in-range index");
    assert_eq!(discarded_c, want, "C, in-range index");

    let (consumed_eval, consumed_c) = lane_verdicts("sparse_consumed_good", CONSUMED, GOOD);
    let want = Verdict::Value("tensor(shape=[2], data=[1.0, 3.0])".to_string());
    assert_eq!(consumed_eval, want, "eval, in-range index, consumed");
    assert_eq!(consumed_c, want, "C, in-range index, consumed");
}
