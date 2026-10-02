//! A discarded sparse index is still checked (chelis#2440).
//!
//! `spec/06-transformations.md` §5.2 makes a potentially trapping node an
//! observable root: "purity alone does not make a possible trap dead". The
//! index of a `gather`, of every scatter mode, and of `one_hot` is data, so
//! [05-OP-52]'s "out-of-bounds indices fail loudly" cannot be ruled out
//! statically. Those five operations were nonetheless outside the trap
//! seed, so an out-of-bounds index that nothing consumed was eliminated and
//! its failure did not occur — in BOTH lanes.
//!
//! The seed covers a sparse node under no activation. §5.2 also says a node
//! whose activation is false checks nothing, and this class has no
//! activation gate in any lane, so seeding an activated one would abort a
//! correct program; `an_out_of_bounds_index_under_a_false_activation_…`
//! below is the guard for that.
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
    /// Stopped, carrying WHY. The `numeric trap:` line when there is one,
    /// so the compiled lane's message stays pinned; otherwise the panic's
    /// own message line, so the evaluator's chelis#1636 stop is compared by
    /// reason too. Carrying `None` for every trapless stop would have made
    /// a type error, an unsupported diagnostic and a lowering panic all
    /// compare equal to the stop this test means.
    Stopped(String),
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
        if let Some(trap) = all.lines().find(|line| line.starts_with("numeric trap:")) {
            return Verdict::Stopped(trap.to_string());
        }
        // A panic prints `thread '<name>' (<id>) panicked at <file>:<line>:`
        // and its message on the NEXT line. The header carries a varying
        // thread id and a source line number, so the message alone is what
        // is stable enough to compare.
        let mut lines = all.lines();
        while let Some(line) = lines.next() {
            if !line.contains("panicked at") {
                continue;
            }
            if let Some(message) = lines.next() {
                return Verdict::Stopped(message.trim().to_string());
            }
        }
        Verdict::Stopped(
            all.lines()
                .find(|line| !line.trim().is_empty())
                .unwrap_or("stopped with no output")
                .trim()
                .to_string(),
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
    // ...and it stops for the reason this test is about, not some other
    // failure that would also satisfy the equality below.
    assert_eq!(
        consumed_eval,
        Verdict::Stopped("gather index 9 out of bounds for axis 0".to_string()),
        "control: the evaluator's stop must be the chelis#1636 bounds panic"
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
        Verdict::Stopped(C_TRAP.to_string()),
        "C must report the sparse-index domain trap by name"
    );
}

/// The index sits in an `if` arm the program does not take, so its
/// activation is false at run time. `spec/06-transformations.md` §5.2: "A
/// potentially trapping node traps only within its activation (spec/10 §3):
/// where its activation is false it computes a value and **checks nothing**."
///
/// This is the regression guard for the seed's first shape, which seeded
/// every sparse node and so turned this correct program into an abort in
/// both lanes. `SparseIndex` is the only checking class with no activation
/// gate in any lane, so seeding an activated node makes it check where the
/// spec says it must not. The activation is computed from an input rather
/// than a literal, so nothing can fold the arm away before lowering.
const FALSE_ACTIVATION: &str =
    "def g(v: tensor[3, f32], idx: tensor[2, i64], flags: tensor[3, f32]) -> tensor[3, f32] = {
  take = gt(tensor_to_scalar(sum(flags, cast(0, i32))), 100.0f32)
  dead = if take then gather(copy(v), idx, 0) else to_tensor([0.0f32, 0.0f32])
  v
}
out = g(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([0i64, INDEX]), to_tensor([1.0f32, 1.0f32, 1.0f32]))
";

#[test]
fn an_out_of_bounds_index_under_a_false_activation_checks_nothing_in_either_lane() {
    let (eval, c) = lane_verdicts("sparse_false_activation", FALSE_ACTIVATION, BAD);
    let want = Verdict::Value("tensor(shape=[3], data=[1.0, 2.0, 3.0])".to_string());
    assert_eq!(
        eval, want,
        "eval: an out-of-bounds index in an untaken arm must check nothing"
    );
    assert_eq!(
        c, want,
        "C: an out-of-bounds index in an untaken arm must check nothing"
    );
}

#[test]
fn an_in_range_index_still_returns_a_value_in_both_lanes() {
    // The negative control for the seed: retaining the node must not turn a
    // valid program into a trap, and must not change its result.
    //
    // Scoped to these two lanes deliberately. Retention is not free
    // everywhere: a target that cannot lower the retained op now refuses a
    // program whose sparse op is dead, where before it only refused one
    // whose sparse op was live. `--target metal` does exactly that
    // (chelis#1383), and a `vmap`ped sparse op fails to lower at all
    // (chelis#2772). Both are lane limits meeting a retention spec/06 5.2
    // requires, not something this seed can decide away.
    let (discarded_eval, discarded_c) = lane_verdicts("sparse_discarded_good", DISCARDED, GOOD);
    let want = Verdict::Value("tensor(shape=[3], data=[1.0, 2.0, 3.0])".to_string());
    assert_eq!(discarded_eval, want, "eval, in-range index");
    assert_eq!(discarded_c, want, "C, in-range index");

    let (consumed_eval, consumed_c) = lane_verdicts("sparse_consumed_good", CONSUMED, GOOD);
    let want = Verdict::Value("tensor(shape=[2], data=[1.0, 3.0])".to_string());
    assert_eq!(consumed_eval, want, "eval, in-range index, consumed");
    assert_eq!(consumed_c, want, "C, in-range index, consumed");
}
