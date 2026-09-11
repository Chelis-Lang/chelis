//! chelis#1739 runtime half: a host-lane function whose declared result
//! carries a literal extent traps when the produced tensor disagrees.
//!
//! chelis#1739's check-time half bounds a `diagonal` result by its literal
//! selected axis, so `def d(x: tensor[n, 4, f32]) -> tensor[3, f32]` is
//! ACCEPTED: `min(n, 4) = 3` whenever `n = 3`. Which side of the minimum wins
//! is a runtime fact, so the declaration is a claim the runtime owes a verdict
//! on. Before this change both lanes silently returned a two-element tensor
//! under that three-element declaration.
//!
//! `diagonal` is a host-lane builtin (`crates/chelis-ir/src/host.rs`'s
//! `HOST_ONLY_BUILTINS`), so `expr_is_dag_lowerable` is false for the witness
//! and the DAG lane's `ExtentWitness` claim never reaches it. The guard is
//! therefore at the host function's RETURN boundary, in both host lanes: the
//! interpreter's closure-return path and the C host emitter's function
//! epilogue. It is not a per-builtin check inside a `tensor_*_host` kernel.
//!
//! Rendering: `[04-NUM-9]` freezes the trap line and requires, on separate
//! accompanying lines, the disagreeing source names, the axis and each observed
//! value. `spec/04-type-system.md` section 4.7 puts "the canonical name of the
//! operation that introduces the guarded extent" in the `<op>` slot, and the
//! observed side here is the RESULT of `diagonal`, an op-produced extent rather
//! than an interface value, so the slot is `diagonal`. That is the same shape
//! the sibling local guards print (`domain in reshape`, `domain in shrink`).
//!
//! Because the slot names an operation, the guard fires only where that
//! operation is known: when the function's return expression IS a direct
//! builtin application. A block-bodied or call-bodied return is deliberately
//! left unguarded here and tracked as chelis#1771, which
//! `a_block_bodied_return_is_not_guarded_and_is_residual` and its C twin lock.
//!
//! # Evidentiary status
//!
//! Per assertion. Every `traps` row is a REGRESSION TEST: measured on the base
//! sha, both lanes printed a `shape=[2]` result and exited zero. Every
//! `executes` row is a DISPOSITION LOCK: green before and after, proving the
//! guard fires only on disagreement.

mod common;

use assert_cmd::Command;
use common::{gcc_available, link_generated};
use std::fs;
use std::process::Command as StdCommand;
use tempfile::{TempDir, tempdir};

/// A two-row operand: `min(2, 4) = 2`.
const TWO_BY_FOUR: &str = "[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]";
/// A three-row operand: `min(3, 4) = 3`.
const THREE_BY_FOUR: &str = "[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]";
/// A five-row operand: `min(5, 4) = 4`, the bound reached from above.
const FIVE_BY_FOUR: &str = "[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], \
                            [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0], \
                            [17.0, 18.0, 19.0, 20.0]]";

/// Root form: the exported `def` is applied from a top-level value binding.
fn binding_root(declared: usize, operand: &str) -> String {
    format!(
        "def d(x: tensor[n, 4, f32]) -> tensor[{declared}, f32] = diagonal(x, 0, 1)\n\
         m = to_tensor({operand})\n\
         out = d(m)\n"
    )
}

/// Root form: the exported `def` is applied inside a nullary `main`, whose own
/// declared result carries the same literal extent.
fn inlined_main_root(declared: usize, operand: &str) -> String {
    format!(
        "def d(x: tensor[n, 4, f32]) -> tensor[{declared}, f32] = diagonal(x, 0, 1)\n\
         def main() -> tensor[{declared}, f32] = d(to_tensor({operand}))\n"
    )
}

/// Root form: the literal result extent spelled through a TYPE ALIAS. The
/// checker resolves the alias, the C emitter reads the resolved ABI type, and
/// round 1 found the interpreter reading the syntactic annotation instead, so
/// the two lanes disagreed on exactly this spelling.
fn alias_root(declared: usize, operand: &str) -> String {
    format!(
        "type Row = tensor[{declared}, f32]\n\
         def d(x: tensor[n, 4, f32]) -> Row = diagonal(x, 0, 1)\n\
         m = to_tensor({operand})\n\
         out = d(m)\n"
    )
}

/// Root form: the same declaration on a function whose return expression is a
/// BLOCK rather than a direct builtin application. The residual witness.
fn block_bodied_root(declared: usize, operand: &str) -> String {
    format!(
        "def d(x: tensor[n, 4, f32]) -> tensor[{declared}, f32] = {{\n  \
         y = diagonal(x, 0, 1)\n  y\n}}\n\
         m = to_tensor({operand})\n\
         out = d(m)\n"
    )
}

fn eval_source(dir: &TempDir, stem: &str, source: &str) -> (bool, String, String) {
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write fixture");
    let out = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["eval", "--file", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run chelis eval");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn check_scores_one(dir: &TempDir, stem: &str, source: &str) {
    let path = dir.path().join(format!("{stem}-check.ch"));
    fs::write(&path, source).expect("write fixture");
    let out = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["check", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run chelis check");
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("chelis check must emit JSON");
    assert_eq!(
        json["score"].as_f64(),
        Some(1.0),
        "the runtime witness must be ACCEPTED at check time, got {json}"
    );
}

fn build_link_run(dir: &TempDir, stem: &str, source: &str) -> (bool, String) {
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write fixture");
    let out_dir = dir.path().join(format!("{stem}-out"));
    let build = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            path.to_str().expect("UTF-8 path"),
            "--target",
            "c",
            "-o",
            out_dir.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("run chelis build");
    assert!(
        build.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let linked = link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(linked.success(), "link failed: {linked}");
    let run = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run the linked binary");
    let mut combined = String::from_utf8_lossy(&run.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&run.stderr));
    (run.status.success(), combined)
}

/// The frozen `[04-NUM-9]` line plus the section 4.7 context content.
fn assert_bound_trap(output: &str, claimed: usize, axis: usize, observed: usize, lane: &str) {
    assert!(
        output.contains("numeric trap: domain in diagonal at int64"),
        "{lane} must render the frozen [04-NUM-9] line, got {output}"
    );
    assert!(
        output.contains(&format!(
            "extent `{claimed}`: claimed = {claimed}, diagonal axis {axis} = {observed}"
        )),
        "{lane} must name the claim, the axis and the observed extent on its own \
         accompanying line, got {output}"
    );
    assert!(
        !output.contains("numeric trap: domain in load"),
        "the interface-value slot belongs to a caller witness, not to an \
         op-produced result, got {output}"
    );
}

// ---------------------------------------------------------------------------
// REGRESSION TESTS: the declaration the runtime cannot satisfy.
// ---------------------------------------------------------------------------

/// REGRESSION TEST. `n = 2` under a declared `tensor[3, f32]`: the real extent
/// is `min(2, 4) = 2`. Measured on the base sha, eval printed
/// `out = tensor(shape=[2], data=[1.0, 6.0])` and exited zero.
#[test]
fn eval_traps_when_the_runtime_extent_is_below_the_declared_literal() {
    let dir = tempdir().expect("tempdir");
    let source = binding_root(3, TWO_BY_FOUR);
    check_scores_one(&dir, "below-binding", &source);
    let (ok, stdout, stderr) = eval_source(&dir, "below-binding", &source);
    assert!(!ok, "eval must trap; stdout was {stdout}");
    assert!(
        !stdout.contains("out = tensor(shape=[2]"),
        "eval must not print a result whose extent the declaration denies, got {stdout}"
    );
    assert_bound_trap(&stderr, 3, 0, 2, "eval");
}

/// REGRESSION TEST. The same disagreement reached through a nullary `main`
/// whose own declared result carries the literal extent.
#[test]
fn eval_traps_on_the_inlined_main_root() {
    let dir = tempdir().expect("tempdir");
    let source = inlined_main_root(3, TWO_BY_FOUR);
    check_scores_one(&dir, "below-main", &source);
    let (ok, stdout, stderr) = eval_source(&dir, "below-main", &source);
    assert!(!ok, "eval must trap; stdout was {stdout}");
    assert_bound_trap(&stderr, 3, 0, 2, "eval");
}

/// REGRESSION TEST (round 1 P1). An aliased declaration traps on eval exactly
/// as the spelled-out one does. Before the fold, this row returned
/// `shape=[2]` and exited zero while the C lane trapped: a parity break the
/// pull request itself introduced, on a four-line program.
#[test]
fn eval_traps_when_the_declared_literal_is_spelled_through_an_alias() {
    let dir = tempdir().expect("tempdir");
    let source = alias_root(3, TWO_BY_FOUR);
    check_scores_one(&dir, "alias-below", &source);
    let (ok, stdout, stderr) = eval_source(&dir, "alias-below", &source);
    assert!(!ok, "eval must trap through the alias; stdout was {stdout}");
    assert_bound_trap(&stderr, 3, 0, 2, "eval");
}

/// DISPOSITION LOCK, C lane, and measured as one: this row was green on both
/// sides of the P1 fold, because the C emitter already read the resolved ABI
/// type. It is here so the pair proves AGREEMENT rather than one lane's
/// behaviour, which is the property P1 broke.
#[test]
fn the_c_lane_traps_when_the_declared_literal_is_spelled_through_an_alias() {
    if !gcc_available() {
        return;
    }
    let dir = tempdir().expect("tempdir");
    let (ran, output) = build_link_run(&dir, "c_alias", &alias_root(3, TWO_BY_FOUR));
    assert!(!ran, "the C lane must abort through the alias: {output}");
    assert_bound_trap(&output, 3, 0, 2, "the C lane");
}

/// DISPOSITION LOCK. The aliased declaration the runtime DOES satisfy executes
/// exactly, on both lanes, so the alias rows prove a guard rather than a
/// blanket refusal of aliased results.
#[test]
fn both_lanes_execute_exactly_through_an_alias_when_the_extents_agree() {
    let dir = tempdir().expect("tempdir");
    let source = alias_root(3, THREE_BY_FOUR);
    let (ok, stdout, stderr) = eval_source(&dir, "alias-equal", &source);
    assert!(ok, "eval must execute; stderr was {stderr}");
    assert!(
        stdout.contains("out = tensor(shape=[3], data=[1.0, 6.0, 11.0])"),
        "the satisfied aliased claim must produce its exact result, got {stdout}"
    );
    if !gcc_available() {
        return;
    }
    let (ran, output) = build_link_run(&dir, "c_alias_equal", &source);
    assert!(ran, "the C lane must execute: {output}");
    assert!(
        output.contains("shape=[3], data=[1.0, 6.0, 11.0]"),
        "the C lane must agree with the evaluator, got {output}"
    );
}

/// RESIDUAL LOCK, not a regression test, and a PASS HERE IS NOT A GUARANTEE OF
/// CORRECTNESS. `[04-NUM-9]`'s `<op>` slot needs the operation that introduces
/// the guarded extent, and a block-bodied return does not name one, so this
/// pull request emits no guard for it: the same disagreement that traps through
/// a direct builtin return still executes silently here. The row exists so the
/// gap is measured rather than assumed, and so that closing it turns this test
/// red on purpose. Tracked as chelis#1771.
#[test]
fn a_block_bodied_return_is_not_guarded_and_is_residual() {
    let dir = tempdir().expect("tempdir");
    let source = block_bodied_root(3, TWO_BY_FOUR);
    check_scores_one(&dir, "residual-block", &source);
    let (ok, stdout, stderr) = eval_source(&dir, "residual-block", &source);
    assert!(ok, "the residual path still executes; stderr was {stderr}");
    assert!(
        stdout.contains("out = tensor(shape=[2], data=[1.0, 6.0])"),
        "the unguarded residual returns the produced extent under a declared 3, \
         got {stdout}"
    );
}

/// RESIDUAL LOCK, C lane, chelis#1771, and A PASS HERE IS NOT EVIDENCE OF
/// CORRECTNESS EITHER. The same gap, lane-symmetric: neither lane guards a
/// block-bodied return, so the two lanes still agree on the wrong answer.
#[test]
fn the_c_lane_leaves_a_block_bodied_return_unguarded_too() {
    if !gcc_available() {
        return;
    }
    let dir = tempdir().expect("tempdir");
    let (ran, output) = build_link_run(&dir, "c_residual", &block_bodied_root(3, TWO_BY_FOUR));
    assert!(ran, "the residual path still executes on C: {output}");
    assert!(
        output.contains("shape=[2], data=[1.0, 6.0]"),
        "both lanes agree on the unguarded residual, got {output}"
    );
}

/// REGRESSION TEST, C lane. The same program built to C, linked and run must
/// abort with the same frozen line. Measured on the base sha it printed
/// `main = tensor(shape=[2], data=[1.0, 6.0])` and exited zero.
#[test]
fn the_c_lane_traps_when_the_runtime_extent_is_below_the_declared_literal() {
    if !gcc_available() {
        return;
    }
    let dir = tempdir().expect("tempdir");
    let (ran, output) = build_link_run(&dir, "c_below", &binding_root(3, TWO_BY_FOUR));
    assert!(!ran, "the C lane must abort: {output}");
    assert_bound_trap(&output, 3, 0, 2, "the C lane");
}

/// REGRESSION TEST, C lane, inlined `main` root.
#[test]
fn the_c_lane_traps_on_the_inlined_main_root() {
    if !gcc_available() {
        return;
    }
    let dir = tempdir().expect("tempdir");
    let (ran, output) = build_link_run(&dir, "c_below_main", &inlined_main_root(3, TWO_BY_FOUR));
    assert!(!ran, "the C lane must abort: {output}");
    assert_bound_trap(&output, 3, 0, 2, "the C lane");
}

// ---------------------------------------------------------------------------
// DISPOSITION LOCKS: the declarations the runtime does satisfy keep running.
// ---------------------------------------------------------------------------

/// DISPOSITION LOCK. `n = 3` under a declared `tensor[3, f32]`: the extents
/// agree and both lanes execute exactly. Green before and after.
#[test]
fn eval_executes_exactly_when_the_runtime_extent_equals_the_declared_literal() {
    let dir = tempdir().expect("tempdir");
    let (ok, stdout, stderr) = eval_source(&dir, "equal", &binding_root(3, THREE_BY_FOUR));
    assert!(ok, "eval must execute; stderr was {stderr}");
    assert!(
        stdout.contains("out = tensor(shape=[3], data=[1.0, 6.0, 11.0])"),
        "the satisfied claim must produce its exact result, got {stdout}"
    );
}

/// DISPOSITION LOCK. The bound reached from above: `n = 5` under a declared
/// `tensor[4, f32]` is `min(5, 4) = 4`, so the declaration holds and the guard
/// must stay quiet. This is the control that the guard compares the PRODUCED
/// extent and not the operand's.
#[test]
fn eval_executes_exactly_when_the_literal_axis_wins_the_minimum() {
    let dir = tempdir().expect("tempdir");
    let (ok, stdout, stderr) = eval_source(&dir, "wins", &binding_root(4, FIVE_BY_FOUR));
    assert!(ok, "eval must execute; stderr was {stderr}");
    assert!(
        stdout.contains("out = tensor(shape=[4], data=[1.0, 6.0, 11.0, 16.0])"),
        "the satisfied claim must produce its exact result, got {stdout}"
    );
}

/// DISPOSITION LOCK, C lane, equality.
#[test]
fn the_c_lane_executes_exactly_when_the_extents_agree() {
    if !gcc_available() {
        return;
    }
    let dir = tempdir().expect("tempdir");
    let (ran, output) = build_link_run(&dir, "c_equal", &binding_root(3, THREE_BY_FOUR));
    assert!(ran, "the C lane must execute: {output}");
    assert!(
        output.contains("shape=[3], data=[1.0, 6.0, 11.0]"),
        "the C lane must agree with the evaluator, got {output}"
    );
}

/// DISPOSITION LOCK, C lane, the bound reached from above.
#[test]
fn the_c_lane_executes_exactly_when_the_literal_axis_wins_the_minimum() {
    if !gcc_available() {
        return;
    }
    let dir = tempdir().expect("tempdir");
    let (ran, output) = build_link_run(&dir, "c_wins", &binding_root(4, FIVE_BY_FOUR));
    assert!(ran, "the C lane must execute: {output}");
    assert!(
        output.contains("shape=[4], data=[1.0, 6.0, 11.0, 16.0]"),
        "the C lane must agree with the evaluator, got {output}"
    );
}

/// DISPOSITION LOCK. A host-lane function with a fully symbolic declared
/// result carries no literal claim, so no guard is emitted and the program runs
/// whatever extent the minimum produces. The control that the guard keys on a
/// LITERAL declared extent and not on every host-lane return.
#[test]
fn a_symbolic_declared_result_is_not_guarded() {
    let dir = tempdir().expect("tempdir");
    let source = "def d(x: tensor[n, 4, f32]) -> tensor[k, f32] = diagonal(x, 0, 1)\n\
                  m = to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])\n\
                  out = d(m)\n";
    let (ok, stdout, stderr) = eval_source(&dir, "symbolic", source);
    assert!(ok, "eval must execute; stderr was {stderr}");
    assert!(
        stdout.contains("out = tensor(shape=[2], data=[1.0, 6.0])"),
        "an unclaimed extent runs as produced, got {stdout}"
    );
}

// ---------------------------------------------------------------------------
// The census. Which shipped code gains a return-boundary guard?
// ---------------------------------------------------------------------------

/// Every emitted return-boundary guard in one C source, as
/// `(enclosing function, axis, required extent)`.
///
/// Read back out of the SHIPPED artifact rather than recomputed from the
/// declaration, so the census cannot drift from the emitter's own decision.
fn emitted_guards(c_source: &str) -> Vec<(String, String, String)> {
    let mut found = Vec::new();
    let mut enclosing = String::from("<none>");
    for line in c_source.lines() {
        if let Some(head) = line.split("__chelis_owned_body(").next()
            && line.contains("__chelis_owned_body(")
            && line.trim_end().ends_with('{')
        {
            enclosing = head
                .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .unwrap_or("<none>")
                .to_string();
        }
        let Some(rest) = line
            .trim()
            .strip_prefix("if (chelis_tensor_shape(__result, ")
        else {
            continue;
        };
        let Some((axis, tail)) = rest.split_once(") != ") else {
            continue;
        };
        let Some(required) = tail.split(')').next() else {
            continue;
        };
        found.push((
            enclosing.clone(),
            axis.to_string(),
            required.trim().to_string(),
        ));
    }
    found
}

fn emit_c(path: &str, out_subdir: &std::path::Path) -> String {
    let build = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            path,
            "--target",
            "c",
            "-o",
            out_subdir.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("run chelis build");
    assert!(
        build.status.success(),
        "chelis build failed for {path}: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let stem = std::path::Path::new(path)
        .file_stem()
        .expect("file stem")
        .to_str()
        .expect("UTF-8 stem");
    fs::read_to_string(out_subdir.join(format!("{stem}.c"))).expect("read generated C")
}

/// POSITIVE CONTROL for the census below. Without this row, "no example gains a
/// guard" could equally mean "the reader never finds a guard", which is how a
/// census stops measuring while staying green.
#[test]
fn the_census_reader_finds_a_guard_that_is_there() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("witness.ch");
    fs::write(&path, binding_root(3, TWO_BY_FOUR)).expect("write fixture");
    let source = emit_c(
        path.to_str().expect("UTF-8 path"),
        &dir.path().join("witness-out"),
    );
    assert_eq!(
        emitted_guards(&source),
        vec![("d".to_string(), "0".to_string(), "3".to_string())],
        "the witness emits exactly one guard, on axis 0 of `d`, claiming 3"
    );
}

/// THE CENSUS. Every executable Phase 0 example is built to C and every
/// return-boundary guard the emitter produced is collected. The guard fires
/// only where a host-lane function's return expression is a direct builtin
/// application AND its declared tensor result carries a literal extent, which
/// no shipped example does today, so this change adds no guard to the corpus
/// and surfaces no pre-existing declaration divergence.
///
/// A function the checker already proved statically (two literal selected axes)
/// still gets a compare it cannot fail. That is kept rather than special-cased:
/// the guard reads the declared ABI type, and teaching it which declarations the
/// checker had already settled would make it depend on checker state the ABI
/// does not carry.
///
/// `examples/illustrative/` is excluded by the repository's example-corpus
/// policy: it is deliberately not on the executable Phase 0 path.
///
/// If a future example gains a guard this test goes red with its name, which is
/// the point. Add the row here after checking, on both lanes, that the
/// declaration the guard now enforces is the one the body really produces.
#[test]
fn no_shipped_example_gains_a_return_boundary_guard() {
    let dir = tempdir().expect("tempdir");
    let mut examples: Vec<std::path::PathBuf> = fs::read_dir("../../examples")
        .expect("read examples")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "ch"))
        .collect();
    examples.sort();
    // The EXACT count, not a floor (round 1 P3). A floor lets the corpus shrink
    // by a third while the census keeps reporting nothing, which is the same
    // failure shape as a reader that stops finding guards. There is no example
    // manifest to read this from, so the number is pinned here: change it in
    // the same commit that adds or removes an executable example, and read the
    // census result before you do.
    // chelis#1787: nothing derives this number, so it drifts, and it drifted
    // again between this branch's two rebases. It reads 31 on `main` against a
    // corpus of 36 there; `record_input_broadcast.ch` makes 37. The pin's own
    // instruction is to change it in the commit that adds an example, so this
    // one sets the true count. The cause chelis#1787 tracks is unchanged, and
    // a textual merge cannot see this conflict: the line does not conflict,
    // so whichever side is replayed last silently wins.
    const EXECUTABLE_PHASE_0_EXAMPLES: usize = 37;
    assert_eq!(
        examples.len(),
        EXECUTABLE_PHASE_0_EXAMPLES,
        "the census must cover every executable Phase 0 example; found {:?}",
        examples
            .iter()
            .filter_map(|path| path.file_name())
            .collect::<Vec<_>>()
    );

    let mut census: Vec<String> = Vec::new();
    for example in &examples {
        let stem = example.file_stem().expect("stem").to_str().expect("UTF-8");
        let source = emit_c(
            example.to_str().expect("UTF-8 path"),
            &dir.path().join(stem),
        );
        for (function, axis, required) in emitted_guards(&source) {
            census.push(format!("{stem}: {function} axis {axis} claims {required}"));
        }
    }

    assert_eq!(
        census,
        Vec::<String>::new(),
        "these shipped defs gained a return-boundary guard and each one needs a \
         both-lane check before it lands"
    );
}
