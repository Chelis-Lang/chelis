//! chelis#1739 runtime half: a host-lane function whose declared result
//! carries a literal extent traps when the produced tensor disagrees.
//!
//! chelis#1739's check-time half bounds a `diagonal` result by its literal
//! selected axis, so `def d[n](x: tensor[n, 4, f32]) -> tensor[3, f32]` is
//! ACCEPTED: `min(n, 4) = 3` whenever `n = 3`. Which side of the minimum wins
//! is a runtime fact, so the declaration is a claim the runtime owes a verdict
//! on. Before this change both lanes silently returned a two-element tensor
//! under that three-element declaration.
//!
//! `diagonal` is a host-lane builtin (`crates/chelis-ir/src/host.rs`'s
//! `HOST_ONLY_BUILTINS`), so `expr_is_dag_lowerable` is false for the witness
//! and the DAG lane's `ExtentWitness` claim never reaches it. The guard is
//! therefore a host-lane guard, in both host lanes: the interpreter's closure
//! application path and the C host emitter. It is not a per-builtin check
//! inside a `tensor_*_host` kernel.
//!
//! Rendering: `[04-NUM-9]` freezes the trap line and requires, on separate
//! accompanying lines, the disagreeing source names, the axis and each observed
//! value. `spec/04-type-system.md` section 4.7 puts "the canonical name of the
//! operation that introduces the guarded extent" in the `<op>` slot, and the
//! observed side here is the RESULT of `diagonal`, an op-produced extent rather
//! than an interface value, so the slot is `diagonal`. That is the same shape
//! the sibling local guards print (`domain in reshape`, `domain in shrink`).
//!
//! chelis#1771 is the widening of that slot. `[04-NUM-9]` requires the lowered
//! primitive name and forbids renaming it to the composed source operation, so
//! the name resolves THROUGH a block tail, a let binding and a callee body:
//! `{ y = diagonal(x, 0, 1); y }` and `def d(x) -> tensor[3, f32] = inner(x)`
//! name `diagonal` exactly as the direct application does. Section 4.7 also
//! fixes WHERE the guard runs, at the source position of the operation that
//! introduces the extent, so an effect bound after the producing binding is
//! observed only when the guard passes.
//!
//! Claims follow the selected branch and cross direct host calls through the
//! private invocation context. The producer checks them before following
//! effects in its own body. Shared callees receive each caller's obligations
//! separately; an untaken branch receives none. Named host result obligations
//! remain separately tracked by chelis#1900.
//!
//! # Evidentiary status
//!
//! Per assertion. Every `traps` row is a REGRESSION TEST: measured on the base
//! sha, both lanes printed a `shape=[2]` result and exited zero. Every
//! `executes` row is a DISPOSITION LOCK: green before and after, proving the
//! guard fires only on disagreement.

#[path = "../src/c_source_name.rs"]
mod c_source_name;
mod common;
#[path = "common/unsupported_wording.rs"]
mod unsupported_wording;

use assert_cmd::Command;
use common::{gcc_available, link_generated};
use std::fs;
use std::process::Command as StdCommand;
use tempfile::{TempDir, tempdir};

/// The C ABI namespace is intentionally independent of authored Chelis names.
/// Census rows stay source-facing so the checker and emitted artifact can be
/// compared without reintroducing source spellings into C.
fn source_name_for_c_symbol(symbol: &str) -> String {
    if symbol.ends_with("__main") {
        return "main".to_string();
    }
    let Some(hex) = symbol.strip_prefix("chelis_fn_") else {
        return symbol.to_string();
    };
    let (pairs, remainder) = hex.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return symbol.to_string();
    }
    let bytes = pairs
        .iter()
        .map(|pair| {
            std::str::from_utf8(pair)
                .ok()
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
        })
        .collect::<Option<Vec<_>>>();
    bytes
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or_else(|| symbol.to_string())
}

/// Examples the C backend refuses at a capability gate, with the
/// diagnostic that refusal must carry. The census asserts the refusal rather
/// than skipping the file: a bare skip stops measuring quietly, and deleting
/// the assertion would make the census's claim smaller than its name.
///
/// Dropout, staged or fixed-control, is compiled and joins the ordinary guard
/// census (chelis#2413). Annotated concat/softmax executes on Eval, but C
/// host emission refuses softmax;
/// `parity_annotated_concat_softmax_eval_and_c_rejection` owns the full-value
/// positive and exact C diagnostic. No C guard artifact exists. When its
/// build succeeds, its row fails and must join the guard census.
fn refused_by_a_capability_gate() -> [(&'static str, &'static str); 1] {
    [(
        "annotated_concat_softmax.ch",
        unsupported_wording::stderr("annotated_concat_softmax_c"),
    )]
}

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
        "def d[n](x: tensor[n, 4, f32]) -> tensor[{declared}, f32] = diagonal(x, 0, 1)\n\
         m = to_tensor({operand})\n\
         out = d(m)\n"
    )
}

/// Root form: the exported `def` is applied inside a nullary `main`, whose own
/// declared result carries the same literal extent.
fn inlined_main_root(declared: usize, operand: &str) -> String {
    format!(
        "def d[n](x: tensor[n, 4, f32]) -> tensor[{declared}, f32] = diagonal(x, 0, 1)\n\
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
         def d[n](x: tensor[n, 4, f32]) -> Row = diagonal(x, 0, 1)\n\
         m = to_tensor({operand})\n\
         out = d(m)\n"
    )
}

/// Root form: the same declaration on a function whose return expression is a
/// BLOCK rather than a direct builtin application. chelis#1771's first witness.
fn block_bodied_root(declared: usize, operand: &str) -> String {
    format!(
        "def d[n](x: tensor[n, 4, f32]) -> tensor[{declared}, f32] = {{\n  \
         y = diagonal(x, 0, 1)\n  y\n}}\n\
         m = to_tensor({operand})\n\
         out = d(m)\n"
    )
}

/// Root form: the declaration sits on a function whose body is a CALL to
/// another def, so the producing primitive is one body further in.
/// chelis#1771's second witness. `inner`'s own result is symbolic, so it
/// carries no claim of its own and the only verdict owed is `d`'s.
fn call_bodied_root(declared: usize, operand: &str) -> String {
    format!(
        "def inner[n, k](x: tensor[n, 4, f32]) -> tensor[k, f32] = diagonal(x, 0, 1)\n\
         def d[n](x: tensor[n, 4, f32]) -> tensor[{declared}, f32] = inner(x)\n\
         m = to_tensor({operand})\n\
         out = d(m)\n"
    )
}

/// Root form: an independent effect sits BETWEEN the producing binding and the
/// block tail. Section 4.7 makes it observable only when the guard passes, so
/// `after` must not reach stdout on either lane.
fn effect_order_root(declared: usize, operand: &str) -> String {
    format!(
        "def d[n](x: tensor[n, 4, f32]) -> tensor[{declared}, f32] ! {{ IO }} = {{\n  \
         y = diagonal(x, 0, 1)\n  _ = print(\"after\")\n  y\n}}\n\
         m = to_tensor({operand})\n\
         out = d(m)\n"
    )
}

/// Root form: the producing operation is inside the CALLEE, and an independent
/// effect sits between it and the callee's own tail. `inner`'s result is
/// symbolic, so the only claim owed is `d`'s, and `d`'s body is the call.
/// chelis#1945's witness.
fn callee_effect_root(declared: usize, operand: &str) -> String {
    format!(
        "def inner[n, k](x: tensor[n, 4, f32]) -> tensor[k, f32] ! {{ IO }} = {{\n  \
         z = diagonal(x, 0, 1)\n  _ = print(\"inside-after\")\n  z\n}}\n\
         def d[n](x: tensor[n, 4, f32]) -> tensor[{declared}, f32] ! {{ IO }} = inner(x)\n\
         m = to_tensor({operand})\n\
         out = d(m)\n"
    )
}

/// Root form: an if selects one producing application at runtime.
fn branchy_root(declared: usize, operand: &str) -> String {
    format!(
        "def d[n](x: tensor[n, 4, f32], flag: bool) -> tensor[{declared}, f32] = \
         if flag then diagonal(x, 0, 1) else diagonal(x, 0, 1)\n\
         m = to_tensor({operand})\n\
         out = d(m, true)\n"
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
        output.contains("numeric trap: domain in diagonal at i64"),
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
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
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
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
    let (ran, output) = build_link_run(&dir, "c_alias_equal", &source);
    assert!(ran, "the C lane must execute: {output}");
    assert!(
        output.contains("shape=[3], data=[1.0, 6.0, 11.0]"),
        "the C lane must agree with the evaluator, got {output}"
    );
}

/// REGRESSION TEST, chelis#1771. Measured on `0820ee28e`, this row printed
/// `out = tensor(shape=[2], data=[1.0, 6.0])` and exited zero: the declaration
/// the direct form traps on executed silently through a block tail. It was the
/// residual `a_block_bodied_return_is_not_guarded_and_is_residual` locked.
#[test]
fn eval_traps_on_a_block_bodied_return() {
    let dir = tempdir().expect("tempdir");
    let source = block_bodied_root(3, TWO_BY_FOUR);
    check_scores_one(&dir, "block-below", &source);
    let (ok, stdout, stderr) = eval_source(&dir, "block-below", &source);
    assert!(
        !ok,
        "eval must trap through the block tail; stdout was {stdout}"
    );
    assert!(
        !stdout.contains("out = tensor(shape=[2]"),
        "eval must not print a result whose extent the declaration denies, got {stdout}"
    );
    assert_bound_trap(&stderr, 3, 0, 2, "eval");
}

/// REGRESSION TEST, C lane, chelis#1771. The lane-symmetric half: measured on
/// `0820ee28e` the linked binary printed `shape=[2], data=[1.0, 6.0]` and
/// exited zero, and the two lanes agreed on the wrong answer.
#[test]
fn the_c_lane_traps_on_a_block_bodied_return() {
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
    let dir = tempdir().expect("tempdir");
    let (ran, output) = build_link_run(&dir, "c_block", &block_bodied_root(3, TWO_BY_FOUR));
    assert!(
        !ran,
        "the C lane must abort through the block tail: {output}"
    );
    assert_bound_trap(&output, 3, 0, 2, "the C lane");
}

/// DISPOSITION LOCK, both lanes. The negative twin of the two rows above: the
/// block-bodied declaration the runtime DOES satisfy executes exactly, so the
/// widened guard is a verdict on disagreement rather than a refusal of block
/// tails. Green on `0820ee28e` and after.
#[test]
fn both_lanes_execute_exactly_on_an_agreeing_block_bodied_return() {
    let dir = tempdir().expect("tempdir");
    let source = block_bodied_root(3, THREE_BY_FOUR);
    let (ok, stdout, stderr) = eval_source(&dir, "block-equal", &source);
    assert!(ok, "eval must execute; stderr was {stderr}");
    assert!(
        stdout.contains("out = tensor(shape=[3], data=[1.0, 6.0, 11.0])"),
        "the satisfied block-bodied claim must produce its exact result, got {stdout}"
    );
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
    let (ran, output) = build_link_run(&dir, "c_block_equal", &source);
    assert!(ran, "the C lane must execute: {output}");
    assert!(
        output.contains("shape=[3], data=[1.0, 6.0, 11.0]"),
        "the C lane must agree with the evaluator, got {output}"
    );
}

/// REGRESSION TEST, chelis#1771. The producing primitive is one callee body
/// further in. Measured on `0820ee28e`, eval printed
/// `out = tensor(shape=[2], data=[1.0, 6.0])` and exited zero.
#[test]
fn eval_traps_on_a_call_bodied_return() {
    let dir = tempdir().expect("tempdir");
    let source = call_bodied_root(3, TWO_BY_FOUR);
    check_scores_one(&dir, "call-below", &source);
    let (ok, stdout, stderr) = eval_source(&dir, "call-below", &source);
    assert!(
        !ok,
        "eval must trap through the callee; stdout was {stdout}"
    );
    assert!(
        !stdout.contains("out = tensor(shape=[2]"),
        "eval must not print a result whose extent the declaration denies, got {stdout}"
    );
    assert_bound_trap(&stderr, 3, 0, 2, "eval");
}

/// REGRESSION TEST, C lane, chelis#1771. Measured on `0820ee28e` the linked
/// binary printed `shape=[2], data=[1.0, 6.0]` and exited zero. `<op>` must be
/// the lowered `diagonal` and not the composed callee name, which is
/// `assert_bound_trap`'s own assertion.
#[test]
fn the_c_lane_traps_on_a_call_bodied_return() {
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
    let dir = tempdir().expect("tempdir");
    let (ran, output) = build_link_run(&dir, "c_call", &call_bodied_root(3, TWO_BY_FOUR));
    assert!(!ran, "the C lane must abort through the callee: {output}");
    assert_bound_trap(&output, 3, 0, 2, "the C lane");
}

/// DISPOSITION LOCK, both lanes. The call-bodied declaration the runtime does
/// satisfy executes exactly. Green on `0820ee28e` and after.
#[test]
fn both_lanes_execute_exactly_on_an_agreeing_call_bodied_return() {
    let dir = tempdir().expect("tempdir");
    let source = call_bodied_root(3, THREE_BY_FOUR);
    let (ok, stdout, stderr) = eval_source(&dir, "call-equal", &source);
    assert!(ok, "eval must execute; stderr was {stderr}");
    assert!(
        stdout.contains("out = tensor(shape=[3], data=[1.0, 6.0, 11.0])"),
        "the satisfied call-bodied claim must produce its exact result, got {stdout}"
    );
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
    let (ran, output) = build_link_run(&dir, "c_call_equal", &source);
    assert!(ran, "the C lane must execute: {output}");
    assert!(
        output.contains("shape=[3], data=[1.0, 6.0, 11.0]"),
        "the C lane must agree with the evaluator, got {output}"
    );
}

/// REGRESSION TEST, chelis#1771, and the row that decides guard PLACEMENT
/// rather than the `<op>` slot. Section 4.7: an independent effect that follows
/// the producing operation in source order is observed only if the guard
/// passes. Measured on `0820ee28e`, eval printed `after` and exited zero, which
/// a guard sited at the return boundary would still do.
#[test]
fn eval_traps_before_an_effect_that_follows_the_producing_operation() {
    let dir = tempdir().expect("tempdir");
    let source = effect_order_root(3, TWO_BY_FOUR);
    check_scores_one(&dir, "effect-below", &source);
    let (ok, stdout, stderr) = eval_source(&dir, "effect-below", &source);
    assert!(!ok, "eval must trap; stdout was {stdout}");
    assert!(
        !stdout.contains("after"),
        "the effect bound after the producing operation must not be observed, got {stdout}"
    );
    assert_bound_trap(&stderr, 3, 0, 2, "eval");
}

/// REGRESSION TEST, C lane, chelis#1771 placement. Measured on `0820ee28e` the
/// linked binary printed `after` and exited zero.
#[test]
fn the_c_lane_traps_before_an_effect_that_follows_the_producing_operation() {
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
    let dir = tempdir().expect("tempdir");
    let (ran, output) = build_link_run(&dir, "c_effect", &effect_order_root(3, TWO_BY_FOUR));
    assert!(!ran, "the C lane must abort: {output}");
    assert!(
        !output.contains("after"),
        "the effect bound after the producing operation must not be observed, got {output}"
    );
    assert_bound_trap(&output, 3, 0, 2, "the C lane");
}

/// DISPOSITION LOCK, both lanes, and the positive control for the row above:
/// when the claim holds, the effect DOES run and the result prints. Without it,
/// "no `after`" could equally mean the fixture never printed at all.
#[test]
fn both_lanes_observe_the_following_effect_when_the_guard_passes() {
    let dir = tempdir().expect("tempdir");
    let source = effect_order_root(3, THREE_BY_FOUR);
    let (ok, stdout, stderr) = eval_source(&dir, "effect-equal", &source);
    assert!(ok, "eval must execute; stderr was {stderr}");
    assert!(
        stdout.contains("after")
            && stdout.contains("out = tensor(shape=[3], data=[1.0, 6.0, 11.0])"),
        "a passing guard observes the following effect and returns its result, got {stdout}"
    );
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
    let (ran, output) = build_link_run(&dir, "c_effect_equal", &source);
    assert!(ran, "the C lane must execute: {output}");
    assert!(
        output.contains("after") && output.contains("shape=[3], data=[1.0, 6.0, 11.0]"),
        "the C lane must agree with the evaluator, got {output}"
    );
}

/// Returning an alias does not move the producing operation or its guard.
/// Sequential shadowing resolves each alias in the scope of its initializer.
#[test]
fn value_aliases_keep_the_guard_at_the_original_producer() {
    for aliases in [
        "z = y\n _ = print(\"after\")\n z",
        "z = y\n w = z\n _ = print(\"after\")\n w",
        "z = y\n y = to_tensor([9.0f32])\n _ = print(\"after\")\n z",
        "y = y\n _ = print(\"after\")\n y",
    ] {
        for (operand, agrees) in [(TWO_BY_FOUR, false), (THREE_BY_FOUR, true)] {
            let dir = tempdir().expect("tempdir");
            let source = format!(
                "def d[n](x: tensor[n, 4, f32]) -> tensor[3, f32] ! {{ IO }} = {{\n \
                 _ = print(\"before\")\n y = diagonal(x, 0, 1)\n {aliases}\n}}\n\
                 out = d(to_tensor({operand}))\n"
            );
            let input = dir.path().join("alias-input.ch");
            fs::write(&input, &source).expect("write source");
            let formatted = Command::cargo_bin("chelis")
                .expect("chelis")
                .arg("fmt")
                .arg(&input)
                .output()
                .expect("fmt");
            assert!(formatted.status.success(), "{formatted:?}");
            let source = String::from_utf8(formatted.stdout).expect("formatted source");
            check_scores_one(&dir, "alias-producer", &source);
            let (ok, stdout, stderr) = eval_source(&dir, "alias-producer", &source);
            assert_eq!(ok, agrees, "{source}\n{stdout}\n{stderr}");
            assert!(stdout.contains("before"), "{stdout}");
            assert_eq!(stdout.contains("after"), agrees, "{stdout}");
            if agrees {
                assert!(
                    stdout.contains("shape=[3], data=[1.0, 6.0, 11.0]"),
                    "{stdout}"
                );
            } else {
                assert_bound_trap(&stderr, 3, 0, 2, "eval");
            }
            {
                assert!(
                    gcc_available(),
                    "the required compiled C receipt must execute"
                );
                let (ok, output) = build_link_run(&dir, "alias-producer-c", &source);
                assert_eq!(ok, agrees, "{source}\n{output}");
                assert!(output.contains("before"), "{output}");
                assert_eq!(output.contains("after"), agrees, "{output}");
                if agrees {
                    assert!(
                        output.contains("shape=[3], data=[1.0, 6.0, 11.0]"),
                        "{output}"
                    );
                } else {
                    assert_bound_trap(&output, 3, 0, 2, "C");
                }
            }
        }
    }
}

/// The caller's literal must reach the callee's producer before its following
/// effect. This was a measured nonconforming rejection: both lanes trapped,
/// but only after printing `inside-after` (chelis#1945).
#[test]
fn an_effect_inside_the_callee_after_the_producing_operation_is_suppressed() {
    let dir = tempdir().expect("tempdir");
    let source = callee_effect_root(3, TWO_BY_FOUR);
    check_scores_one(&dir, "residual-callee-effect", &source);
    let (ok, stdout, stderr) = eval_source(&dir, "residual-callee-effect", &source);
    assert!(
        !ok,
        "the refuted claim is still guarded through the callee; stdout was {stdout}"
    );
    assert_bound_trap(&stderr, 3, 0, 2, "eval");
    assert!(
        !stdout.contains("inside-after"),
        "the callee's following effect must be suppressed, got {stdout}"
    );
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
    let (ran, output) = build_link_run(&dir, "c_residual_callee_effect", &source);
    assert!(!ran, "the C lane traps through the callee too: {output}");
    assert_bound_trap(&output, 3, 0, 2, "the C lane");
    assert!(
        !output.contains("inside-after"),
        "the callee's following effect must be suppressed on C, got {output}"
    );
}

/// An if tail carries the claim to its selected producing expression. Before
/// #1771's branch repair both lanes returned the denying extent silently.
#[test]
fn a_branchy_tail_checks_its_selected_producer() {
    let dir = tempdir().expect("tempdir");
    let source = branchy_root(3, TWO_BY_FOUR);
    check_scores_one(&dir, "residual-branchy", &source);
    let (ok, stdout, stderr) = eval_source(&dir, "residual-branchy", &source);
    assert!(!ok, "the selected branch must trap, got {stdout}");
    assert_bound_trap(&stderr, 3, 0, 2, "eval");
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
    let (ran, output) = build_link_run(&dir, "c_residual_branchy", &source);
    assert!(!ran, "the selected C branch must trap: {output}");
    assert_bound_trap(&output, 3, 0, 2, "C");
}

/// Every branch owns its selected producer, even through a binding and alias.
/// The other branch produces an invalid extent but must remain unexecuted.
fn selected_branch_root(form: &str, second: bool, agrees: bool) -> String {
    let branch = |op: &str, input: &str| {
        let producer = if op == "cumsum" {
            format!("cumsum(diagonal({input}, 0, 1), 0)")
        } else {
            format!("diagonal({input}, 0, 1)")
        };
        format!(
            "{{\n _ = print(\"{op}-before\")\n y = {producer}\n z = y\n y = to_tensor([99.0f32])\n _ = print(\"{op}-after\")\n z\n}}"
        )
    };
    let first = branch("diagonal", "x");
    let other = branch("cumsum", "y");
    let (preamble, param, selection, actual) = match form {
        "if" => (
            "",
            "flag: bool",
            format!("if flag then {first} else {other}"),
            (!second).to_string(),
        ),
        "option" => (
            "",
            "flag: Option[bool]",
            format!("match flag with {{\n | Some(value) => {first}\n | None => {other}\n}}"),
            if second { "None" } else { "Some(true)" }.into(),
        ),
        "adt" => (
            "type Choice = | First | Second\n",
            "flag: Choice",
            format!("match flag with {{\n | First => {first}\n | Second => {other}\n}}"),
            if second { "Second" } else { "First" }.into(),
        ),
        "default" => (
            "type Choice = | First | Second\n",
            "flag: Choice",
            format!("match flag with {{\n | First => {first}\n | _ => {other}\n}}"),
            if second { "Second" } else { "First" }.into(),
        ),
        _ => panic!("unknown branch fixture"),
    };
    let selected = if agrees { THREE_BY_FOUR } else { TWO_BY_FOUR };
    let (x, y) = if second {
        (TWO_BY_FOUR, selected)
    } else {
        (selected, TWO_BY_FOUR)
    };
    format!(
        "{preamble}def d[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32], {param}) -> tensor[3, f32] ! {{ IO }} = {{\n result = {selection}\n alias = result\n _ = print(\"outer-after\")\n alias\n}}\nout = d(to_tensor({x}), to_tensor({y}), {actual})\n"
    )
}

fn canonical_fixture(dir: &TempDir, source: &str) -> String {
    let input = dir.path().join("fixture.ch");
    fs::write(&input, source).expect("write fixture");
    let formatted = Command::cargo_bin("chelis")
        .expect("chelis")
        .arg("fmt")
        .arg(input)
        .output()
        .expect("format fixture");
    assert!(formatted.status.success(), "{formatted:?}");
    String::from_utf8(formatted.stdout).expect("formatted fixture")
}

fn assert_selected_branch(form: &str, c_lane: bool) {
    assert!(
        !c_lane || gcc_available(),
        "the required compiled C receipt must execute"
    );
    for second in [false, true] {
        for agrees in [false, true] {
            let dir = tempdir().expect("tempdir");
            let source = canonical_fixture(&dir, &selected_branch_root(form, second, agrees));
            check_scores_one(&dir, "selected", &source);
            let (ok, output) = if c_lane {
                build_link_run(&dir, "selected", &source)
            } else {
                let (ok, stdout, stderr) = eval_source(&dir, "selected", &source);
                (ok, format!("{stdout}{stderr}"))
            };
            let (op, untaken) = if second {
                ("cumsum", "diagonal")
            } else {
                ("diagonal", "cumsum")
            };
            assert_eq!(ok, agrees, "{source}\n{output}");
            assert!(output.contains(&format!("{op}-before")), "{output}");
            assert_eq!(output.contains(&format!("{op}-after")), agrees, "{output}");
            assert_eq!(output.contains("outer-after"), agrees, "{output}");
            assert!(!output.contains(&format!("{untaken}-before")), "{output}");
            if agrees {
                assert!(
                    output.contains(if second {
                        "shape=[3], data=[1.0, 7.0, 18.0]"
                    } else {
                        "shape=[3], data=[1.0, 6.0, 11.0]"
                    }),
                    "{output}"
                );
                assert!(!output.contains("numeric trap:"), "{output}");
            } else {
                assert!(
                    output
                        .lines()
                        .any(|line| line == format!("numeric trap: domain in {op} at i64")),
                    "{output}"
                );
                assert!(
                    output.contains(&format!("extent `3`: claimed = 3, {op} axis 0 = 2")),
                    "{output}"
                );
                assert!(!output.contains("shape=["), "{output}");
            }
        }
    }
}

#[test]
fn eval_selected_if_result_claims() {
    assert_selected_branch("if", false);
}

#[test]
fn c_selected_if_result_claims() {
    assert_selected_branch("if", true);
}

#[test]
fn eval_selected_match_result_claims() {
    for form in ["option", "adt", "default"] {
        assert_selected_branch(form, false);
    }
}

#[test]
fn c_selected_match_result_claims() {
    for form in ["option", "adt", "default"] {
        assert_selected_branch(form, true);
    }
}

fn assert_shared_callee_claims(c_lane: bool) {
    assert!(
        !c_lane || gcc_available(),
        "the required compiled C receipt must execute"
    );
    for agrees in [false, true] {
        let dir = tempdir().expect("tempdir");
        let last = if agrees { THREE_BY_FOUR } else { TWO_BY_FOUR };
        let source = format!(
            "def inner[n, k](x: tensor[n, 4, f32]) -> tensor[k, f32] ! {{ IO }} = {{\n _ = print(\"inside-before\")\n z = diagonal(x, 0, 1)\n _ = print(\"inside-after\")\n z\n}}\ndef middle[n, k](x: tensor[n, 4, f32]) -> tensor[k, f32] ! {{ IO }} = {{\n y = inner(x)\n _ = print(\"middle-after\")\n y\n}}\ndef two[n](x: tensor[n, 4, f32]) -> tensor[2, f32] ! {{ IO }} = middle(x)\ndef three[n](x: tensor[n, 4, f32]) -> tensor[3, f32] ! {{ IO }} = middle(x)\na = two(to_tensor({TWO_BY_FOUR}))\nb = three(to_tensor({THREE_BY_FOUR}))\nc = three(to_tensor({last}))\n"
        );
        let source = canonical_fixture(&dir, &source);
        check_scores_one(&dir, "shared", &source);
        let (ok, output) = if c_lane {
            build_link_run(&dir, "shared", &source)
        } else {
            let (ok, stdout, stderr) = eval_source(&dir, "shared", &source);
            (ok, format!("{stdout}{stderr}"))
        };
        assert_eq!(ok, agrees, "{source}\n{output}");
        assert_eq!(output.matches("inside-before").count(), 3, "{output}");
        assert_eq!(
            output.matches("inside-after").count(),
            if agrees { 3 } else { 2 },
            "{output}"
        );
        assert_eq!(
            output.matches("middle-after").count(),
            if agrees { 3 } else { 2 },
            "{output}"
        );
        if agrees {
            assert_eq!(
                output.matches("shape=[2], data=[1.0, 6.0]").count(),
                1,
                "{output}"
            );
            assert_eq!(
                output.matches("shape=[3], data=[1.0, 6.0, 11.0]").count(),
                2,
                "{output}"
            );
        } else {
            assert_bound_trap(&output, 3, 0, 2, if c_lane { "C" } else { "eval" });
        }
    }
}

#[test]
fn eval_shared_callee_result_claims_are_invocation_scoped() {
    assert_shared_callee_claims(false);
}

#[test]
fn c_shared_callee_result_claims_are_invocation_scoped() {
    assert_shared_callee_claims(true);
}

/// REGRESSION TEST, C lane. The same program built to C, linked and run must
/// abort with the same frozen line. Measured on the base sha it printed
/// `main = tensor(shape=[2], data=[1.0, 6.0])` and exited zero.
#[test]
fn the_c_lane_traps_when_the_runtime_extent_is_below_the_declared_literal() {
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
    let dir = tempdir().expect("tempdir");
    let (ran, output) = build_link_run(&dir, "c_below", &binding_root(3, TWO_BY_FOUR));
    assert!(!ran, "the C lane must abort: {output}");
    assert_bound_trap(&output, 3, 0, 2, "the C lane");
}

/// REGRESSION TEST, C lane, inlined `main` root.
#[test]
fn the_c_lane_traps_on_the_inlined_main_root() {
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
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
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
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
    assert!(
        gcc_available(),
        "the required compiled C receipt must execute"
    );
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
    let source = "def d[n, k](x: tensor[n, 4, f32]) -> tensor[k, f32] = diagonal(x, 0, 1)\n\
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
// The census. Which shipped code gains a declared-result guard?
// ---------------------------------------------------------------------------

/// A complete emitted host shape guard. Diagnostics alone are not guards:
/// the comparison, diagnostic and canonical trap must share the same block.
struct EmittedShapeGuard<'a> {
    function: String,
    condition: &'a str,
    diagnostic: &'a str,
    trap: &'a str,
}

fn emitted_shape_guard_blocks(c_source: &str) -> Vec<EmittedShapeGuard<'_>> {
    let lines: Vec<_> = c_source.lines().collect();
    let mut found = Vec::new();
    let mut enclosing: Option<String> = None;
    for (index, line) in lines.iter().enumerate() {
        if let Some(head) = line.split("__chelis_owned_body(").next()
            && line.contains("__chelis_owned_body(")
            && line.trim_end().ends_with('{')
        {
            enclosing = Some(source_name_for_c_symbol(
                head.rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("<none>"),
            ));
            continue;
        }
        if *line == "}" {
            // Helpers have their own guard owner and are outside this census.
            enclosing = None;
            continue;
        }
        let Some(function) = enclosing.as_ref() else {
            continue;
        };
        let condition = line.trim();
        if !condition.starts_with("if (")
            || !condition.ends_with('{')
            || !condition.contains("chelis_tensor_shape(")
            || !condition.contains(" != ")
        {
            continue;
        }
        let Some(body) = lines.get(index + 1..index + 4) else {
            continue;
        };
        let diagnostic = body[0].trim();
        let trap = body[1].trim();
        if !diagnostic.starts_with("fprintf(stderr, ")
            || !trap.starts_with("chelis_numeric_trap(\"numeric trap: domain in ")
            || !trap.ends_with(" at i64\");")
            || body[2].trim() != "}"
            || line.len() - line.trim_start().len() != body[2].len() - body[2].trim_start().len()
        {
            continue;
        }
        found.push(EmittedShapeGuard {
            function: function.clone(),
            condition,
            diagnostic,
            trap,
        });
    }
    found
}

/// Host result frames with an executable producer check or forwarded call.
fn emitted_guards(c_source: &str) -> Vec<(String, String, String, String)> {
    let mut found = Vec::new();
    let mut enclosing: Option<String> = None;
    let mut axes = Vec::<(String, String)>::new();
    let mut reading_axes = false;
    for line in c_source.lines() {
        if let Some(head) = line.split("__chelis_owned_body(").next()
            && line.contains("__chelis_owned_body(")
            && line.trim_end().ends_with('{')
        {
            enclosing = Some(source_name_for_c_symbol(
                head.rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("<none>"),
            ));
            axes.clear();
            reading_axes = false;
            continue;
        }
        if line == "}" {
            enclosing = None;
            continue;
        }
        let Some(function) = enclosing.as_ref() else {
            continue;
        };
        let line = line.trim();
        if line == "const __chelis_host_result_axis __chelis_result_axes[] = {" {
            reading_axes = true;
            continue;
        }
        if reading_axes {
            if line == "};" {
                reading_axes = false;
            } else {
                // `{ axis, required, claim, source, source_axis }`: a literal
                // row has NULL labels; a named row reads its witnessing
                // parameter axis and names the binder.
                let row = line
                    .strip_prefix("{ ")
                    .and_then(|s| s.strip_suffix(" },"))
                    .expect("a declared-result row is one braced initializer");
                let (axis, rest) = row.split_once(", ").expect("axis and requirement");
                axis.parse::<usize>().expect("literal axis");
                let required = match rest.strip_suffix(", NULL, NULL, 0") {
                    Some(literal) => {
                        literal.parse::<usize>().expect("literal requirement");
                        literal.to_string()
                    }
                    None => {
                        let (_, labels) = rest
                            .split_once("), ")
                            .expect("a named row reads its witnessing parameter axis");
                        let (claim, _) = labels.split_once(", ").expect("binder and source");
                        claim.trim_matches('"').to_string()
                    }
                };
                axes.push((axis.to_string(), required));
            }
            continue;
        }
        // Local checks and forwarded call obligations are both executable
        // uses of this declaration's frame. A dormant context parameter, or a
        // DAG helper's independent guard, supplies no declaration evidence.
        let target = line
            .strip_prefix("__chelis_check_host_result_claims(__chelis_result_claims, ")
            .and_then(|rest| rest.split_once(", ").map(|(target, _)| target))
            .or_else(|| {
                (line.contains(", __chelis_result_claims, &") && line.ends_with(");"))
                    .then(|| line.split_once(" = "))
                    .flatten()
                    .and_then(|(lhs, _)| lhs.split_whitespace().last())
            });
        if let Some(target) = target {
            for (axis, required) in &axes {
                found.push((
                    function.clone(),
                    target.to_string(),
                    axis.clone(),
                    required.clone(),
                ));
            }
        }
    }
    found
}

/// Every emitted signature-entry guard, as `(enclosing function, claim)`.
/// Named comparisons retain their binder label; literals retain the input,
/// axis and expected extent. Rank/null metadata checks are separate and do not
/// count. Requiring the load trap and entry diagnostic distinguishes these
/// comparisons from declared-result claims, even when both compare literals.
fn emitted_entry_guards(c_source: &str) -> Vec<(String, String)> {
    emitted_entry_inventory(c_source)
        .into_iter()
        .flat_map(|(function, guards)| {
            guards
                .into_iter()
                .map(move |guard| (function.clone(), guard.label.unwrap_or(guard.context)))
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EntryGuardReceipt {
    comparison: String,
    context: String,
    // Dim::Var preserves equality identity but the public checker record may
    // lack its authored spelling. Never recover expected authority from C.
    label: Option<String>,
}

type EntryInventory = std::collections::BTreeMap<String, Vec<EntryGuardReceipt>>;

fn entry_inventories_match(expected: &[EntryGuardReceipt], actual: &[EntryGuardReceipt]) -> bool {
    expected.len() == actual.len()
        && expected.iter().zip(actual).all(|(expected, actual)| {
            expected.comparison == actual.comparison
                && expected.context == actual.context
                && expected
                    .label
                    .as_ref()
                    .is_none_or(|label| actual.label.as_ref() == Some(label))
        })
}

fn emitted_entry_inventory(c_source: &str) -> EntryInventory {
    let mut found = EntryInventory::new();
    for guard in emitted_shape_guard_blocks(c_source) {
        if guard.trap != "chelis_numeric_trap(\"numeric trap: domain in load at i64\");" {
            continue;
        }
        if let Some(rest) = guard.diagnostic.strip_prefix("fprintf(stderr, \"input ")
            && let Some((claim, _)) = rest.split_once(", got %lld")
            && claim.contains(" axis ")
            && claim.contains(" expected ")
        {
            found
                .entry(guard.function)
                .or_default()
                .push(EntryGuardReceipt {
                    comparison: guard.condition.into(),
                    context: format!("input {claim}"),
                    label: None,
                });
        } else if let Some(rest) = guard.diagnostic.strip_prefix("fprintf(stderr, \"extent `")
            && let Some((claim, detail)) = rest.split_once('`')
            && !detail.starts_with(": claimed = ")
            && let Some(detail) = detail.strip_prefix(": ")
            && let Some((context, _)) = detail.split_once("\\n")
        {
            found
                .entry(guard.function)
                .or_default()
                .push(EntryGuardReceipt {
                    comparison: guard.condition.into(),
                    context: context.to_string(),
                    label: Some(claim.to_string()),
                });
        }
    }
    found
}

fn census_checked_program(source: &str) -> chelis_types::CheckedProgram {
    use chelis_compiler_api::pipeline::{
        PipelineGoal, PipelineOutcome, PipelineRequest, run_source,
    };
    let outcome = run_source(PipelineRequest {
        source_kind: chelis_compiler_api::schema::SourceKind::Surf,
        source,
        entry: None,
        goal: PipelineGoal::FullCheck,
    })
    .expect("the emitted example must pass the independent front-end check");
    let PipelineOutcome::Checked(checked) = outcome else {
        unreachable!("requested FullCheck")
    };
    checked.into_parts().2
}

/// Independent traversal of the checker contract, not the production entry
/// plan or host IR. Inference supplies a contract only when none was authored.
fn signature_entry_inventory(
    checked: &chelis_types::CheckedProgram,
    signature: &chelis_types::infer::FunctionSignatureInference,
    c_parameters: &[String],
) -> Vec<EntryGuardReceipt> {
    use chelis_types::types::{Dim, Type};
    let ty = signature
        .authored_signature_type
        .as_ref()
        .unwrap_or(&signature.checked_signature);
    let ty = checked.adt_registry().expand_aliases(ty);
    let mut parameters = Vec::new();
    let mut remainder = &ty;
    while parameters.len() < signature.params.len() {
        let Type::Fn(args, result) = remainder else {
            panic!("missing function parameters for {}", signature.name)
        };
        parameters.extend(args);
        remainder = result;
    }
    assert_eq!(parameters.len(), signature.params.len());
    assert_eq!(parameters.len(), c_parameters.len());
    let mut first = Vec::<(Dim, (String, String))>::new();
    let mut guards = Vec::new();
    for ((parameter, ty), c_name) in signature.params.iter().zip(parameters).zip(c_parameters) {
        let ty = match ty {
            Type::Ref(inner) => inner.as_ref(),
            ty => ty,
        };
        let Type::Tensor(dimensions, _) = ty else {
            continue;
        };
        for (axis, dimension) in dimensions.iter().enumerate() {
            let observed = format!("chelis_tensor_shape({c_name}, {axis})");
            let witness = format!("{} axis {axis} = %lld", parameter.name);
            match dimension {
                Dim::Lit(required) => guards.push(EntryGuardReceipt {
                    comparison: format!("if ({observed} != {required}) {{"),
                    context: format!("input `{}` axis {axis} expected {required}", parameter.name),
                    label: None,
                }),
                Dim::Name(name) if name == "*" => {}
                Dim::Name(_) | Dim::Var(_) => {
                    if let Some((_, (canonical, canonical_witness))) =
                        first.iter().find(|(key, _)| key == dimension)
                    {
                        guards.push(EntryGuardReceipt {
                            comparison: format!("if ({observed} != {canonical}) {{"),
                            context: format!("{canonical_witness}, {witness}"),
                            label: match dimension {
                                Dim::Name(name) => Some(name.clone()),
                                _ => None,
                            },
                        });
                    } else {
                        first.push((dimension.clone(), (observed, witness)));
                    }
                }
                Dim::Wildcard | Dim::Rank(_) => {}
            }
        }
    }
    guards
}

/// Enumerate owned definitions independently of whether a guard was found.
/// This makes deleting every guard from one function a comparison failure.
fn assert_signature_entry_inventory(stem: &str, source: &str, emitted: &str) {
    let checked = census_checked_program(source);
    let mut actual = emitted_entry_inventory(emitted);
    let c_name = c_source_name::CSourceName::from_path(std::path::Path::new(&format!("{stem}.ch")));
    for line in emitted
        .lines()
        .filter(|line| line.trim_end().ends_with('{'))
    {
        let Some((head, parameters)) = line.split_once("__chelis_owned_body(") else {
            continue;
        };
        let function = head
            .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
            .next()
            .expect("function name");
        let source_name = if function == format!("{}__main", c_name.symbol()) {
            "main".to_string()
        } else {
            source_name_for_c_symbol(function)
        };
        let signatures = &checked.signature_inference().functions;
        let signature = signatures
            .get(&source_name)
            .or_else(|| {
                source_name
                    .strip_prefix("chelis_user__")
                    .and_then(|name| signatures.get(name))
            })
            .unwrap_or_else(|| {
                panic!("{stem}: emitted owned function {function} has no checked signature")
            });
        // Only C binding spelling comes from the header. Types, dimensions,
        // declaration order and source labels come from the checked program.
        let tokens: Vec<_> = parameters
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .collect();
        let c_parameters = signature
            .params
            .iter()
            .map(|param| {
                let escaped = format!("chelis_user__{}", param.name);
                if tokens.contains(&escaped.as_str()) {
                    escaped
                } else {
                    assert!(
                        tokens.contains(&param.name.as_str()),
                        "{stem}: missing C parameter {}",
                        param.name
                    );
                    param.name.clone()
                }
            })
            .collect::<Vec<_>>();
        let expected = signature_entry_inventory(&checked, signature, &c_parameters);
        let actual = actual.remove(&source_name).unwrap_or_default();
        assert!(
            entry_inventories_match(&expected, &actual),
            "{stem}: ordered signature entry for {function}: expected {expected:?}, actual {actual:?}"
        );
    }
    assert!(
        actual.is_empty(),
        "{stem}: entry guards outside checked owned functions: {actual:?}"
    );
}

/// POSITIVE CONTROL for the entry-guard reader. Without it, "no example gains
/// an entry guard" could equally mean the reader never finds one.
#[test]
fn the_census_reader_finds_an_entry_guard_that_is_there() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("entry_witness.ch");
    let source = "def both(x: tensor[seq, f32], y: tensor[batch, seq, f32]) -> \
                  (tensor[seq, f32], tensor[batch, seq, f32]) = (neg(x), neg(y))\n\
                  a = to_tensor([1.0, 2.0])\n\
                  b = to_tensor([[1.0, 2.0], [3.0, 4.0]])\n\
                  out = both(a, b)\n";
    fs::write(&path, source).expect("write fixture");
    let emitted = emit_c(
        path.to_str().expect("UTF-8 path"),
        &dir.path().join("entry-witness-out"),
    );
    assert_eq!(
        emitted_entry_guards(&emitted),
        vec![("both".to_string(), "seq".to_string())],
        "the witness emits exactly one entry guard, in `both`, on the binder `seq`"
    );
    assert!(
        emitted_guards(&emitted).is_empty(),
        "and the declared-result reader must not also claim it: the two guard \
         models are counted separately"
    );
}

#[test]
fn the_census_reader_separates_mixed_entry_and_result_guards() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mixed_guard_witness.ch");
    let source = format!(
        "def d[n](x: tensor[n, 4, f32], y: tensor[n, f32]) -> tensor[3, f32] = diagonal(x, 0, 1)\n\
         out = d(to_tensor({TWO_BY_FOUR}), to_tensor([1.0, 2.0]))\n"
    );
    fs::write(&path, source).expect("write fixture");
    let emitted = emit_c(
        path.to_str().expect("UTF-8 path"),
        &dir.path().join("mixed-guard-out"),
    );
    assert_eq!(
        emitted_entry_guards(&emitted),
        vec![
            ("d".to_string(), "input `x` axis 1 expected 4".to_string()),
            ("d".to_string(), "n".to_string()),
        ],
        "literal and repeated-binder entry guards stay in signature order"
    );
    assert_eq!(
        emitted_guards(&emitted),
        vec![(
            "d".to_string(),
            "__result".to_string(),
            "0".to_string(),
            "3".to_string(),
        )],
        "the result reader must not absorb a literal entry comparison"
    );
}

#[test]
fn the_census_reader_ignores_diagnostics_without_executable_guards() {
    let source = r#"
chelis_tensor *d__chelis_owned_body(chelis_tensor *x, chelis_tensor *y) {
    fprintf(stderr, "input `x` axis 0 expected 3, got %lld\n", (long long)chelis_tensor_shape(x, 0));
    fprintf(stderr, "extent `n`: x axis 0 = %lld, y axis 0 = %lld\n", 2LL, 3LL);
    fprintf(stderr, "extent `3`: claimed = 3, diagonal axis 0 = %lld\n", 2LL);
    if (chelis_tensor_shape(x, 0) != 3) {
        fprintf(stderr, "extent `3`: claimed = 3, diagonal axis 0 = %lld\n", (long long)chelis_tensor_shape(x, 0));
    }
    if (x == NULL || chelis_tensor_rank(x) != 1 || chelis_tensor_shape(x, 0) != chelis_tensor_shape(y, 0)) {
        fprintf(stderr, "extent `n`: x axis 0 = %lld, y axis 0 = %lld\n", 2LL, 3LL);
    }
    if (unrelated) {
        fprintf(stderr, "extent `3`: claimed = 3, diagonal axis 0 = %lld\n", 2LL);
        chelis_numeric_trap("numeric trap: domain in diagonal at i64");
    }
}
"#;
    assert!(emitted_guards(source).is_empty());
    assert!(emitted_entry_guards(source).is_empty());
}

#[test]
fn the_census_signature_oracle_preserves_aliases_and_independent_scopes() {
    let source = "type Row = tensor[2, f32]\n\
        def d(x: Row, y: tensor[seq, f32], z: tensor[seq, f32]) -> (Row, tensor[seq, f32], tensor[seq, f32]) = (x, y, z)\n\
        def partial[n](x: Row, y: tensor[n, f32], z: tensor[n, f32]) = (x, y, z)\n\
        def independent[n, k](x: tensor[n, f32], y: tensor[k, f32]) = (x, y)\n\
        def inferred(x) = x\n";
    let checked = census_checked_program(source);
    let functions = &checked.signature_inference().functions;
    let d = &functions["d"];
    assert!(d.authored_signature_type.is_some());
    assert_eq!(
        signature_entry_inventory(&checked, d, &["x".into(), "y".into(), "z".into()]),
        vec![
            EntryGuardReceipt {
                comparison: "if (chelis_tensor_shape(x, 0) != 2) {".into(),
                context: "input `x` axis 0 expected 2".into(),
                label: None,
            },
            EntryGuardReceipt {
                comparison: "if (chelis_tensor_shape(z, 0) != chelis_tensor_shape(y, 0)) {".into(),
                context: "y axis 0 = %lld, z axis 0 = %lld".into(),
                label: Some("seq".into()),
            },
        ]
    );
    let partial = signature_entry_inventory(
        &checked,
        &functions["partial"],
        &["x".into(), "y".into(), "z".into()],
    );
    let complete = signature_entry_inventory(&checked, d, &["x".into(), "y".into(), "z".into()]);
    assert_eq!(partial.len(), 2);
    assert!(partial[1].label.is_none());
    assert!(entry_inventories_match(&partial, &complete));
    let mut wrong_witness = complete.clone();
    wrong_witness[1].context = "x axis 0 = %lld, z axis 0 = %lld".into();
    assert!(!entry_inventories_match(&partial, &wrong_witness));
    let mut wrong_label = complete.clone();
    wrong_label[1].label = Some("wrong".into());
    assert!(!entry_inventories_match(&complete, &wrong_label));
    assert!(
        signature_entry_inventory(
            &checked,
            &functions["independent"],
            &["x".into(), "y".into()]
        )
        .is_empty()
    );
    assert!(functions["inferred"].authored_signature_type.is_none());
    assert!(signature_entry_inventory(&checked, &functions["inferred"], &["x".into()]).is_empty());
}

#[test]
fn the_census_signature_oracle_rejects_missing_duplicate_reordered_and_wrong_axis_guards() {
    let source = "def d[n](x: tensor[2, n, f32], y: tensor[n, f32]) = (x, y)\n";
    let checked = census_checked_program(source);
    let expected = signature_entry_inventory(
        &checked,
        &checked.signature_inference().functions["d"],
        &["x".into(), "y".into()],
    );
    let literal = r#"    if (chelis_tensor_shape(x, 0) != 2) {
        fprintf(stderr, "input `x` axis 0 expected 2, got %lld\n", (long long)chelis_tensor_shape(x, 0));
        chelis_numeric_trap("numeric trap: domain in load at i64");
    }
"#;
    let named = r#"    if (chelis_tensor_shape(y, 0) != chelis_tensor_shape(x, 1)) {
        fprintf(stderr, "extent `n`: x axis 1 = %lld, y axis 0 = %lld\n", (long long)chelis_tensor_shape(x, 1), (long long)chelis_tensor_shape(y, 0));
        chelis_numeric_trap("numeric trap: domain in load at i64");
    }
"#;
    let read = |body: String| {
        let c =
            format!("void d__chelis_owned_body(chelis_tensor *x, chelis_tensor *y) {{\n{body}}}\n");
        emitted_entry_inventory(&c).remove("d").unwrap_or_default()
    };
    assert!(entry_inventories_match(
        &expected,
        &read(format!("{literal}{named}"))
    ));
    for mutated in [
        named.to_string(),
        format!("{literal}{literal}{named}"),
        format!("{named}{literal}"),
        format!("{literal}{named}")
            .replace("chelis_tensor_shape(x, 1)", "chelis_tensor_shape(x, 0)"),
        format!("{literal}{named}").replace("x axis 1 = %lld", "x axis 0 = %lld"),
    ] {
        assert!(!entry_inventories_match(&expected, &read(mutated)));
    }
}

/// Every executable Phase 0 example, sorted: the `.ch` files directly under
/// `examples/`, which the repository's example-corpus policy defines as the
/// executable set. `examples/illustrative/` is a subdirectory and is therefore
/// not reached, which is the policy's intent.
fn executable_examples() -> Vec<std::path::PathBuf> {
    let mut paths: Vec<std::path::PathBuf> = fs::read_dir("../../examples")
        .expect("read examples")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .filter(|path| path.extension().is_some_and(|ext| ext == "ch"))
        .collect();
    paths.sort();
    paths
}

fn build_c(path: &str, out_subdir: &std::path::Path) -> std::process::Output {
    Command::cargo_bin("chelis")
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
        .expect("run chelis build")
}

fn emit_c(path: &str, out_subdir: &std::path::Path) -> String {
    let build = build_c(path, out_subdir);
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
        vec![(
            "d".to_string(),
            "__result".to_string(),
            "0".to_string(),
            "3".to_string()
        )],
        "the witness emits exactly one guard, on axis 0 of `d`'s result, claiming 3"
    );
}

/// POSITIVE CONTROL for the chelis#1771 half of the census. A block-bodied
/// return still emits exactly one guard, and it names the let temporary the
/// producing operation wrote rather than `__result`: that is the placement
/// section 4.7 requires, read back out of the shipped artifact.
#[test]
fn the_census_reader_finds_the_guard_a_block_tail_moved() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("block_witness.ch");
    fs::write(&path, block_bodied_root(3, TWO_BY_FOUR)).expect("write fixture");
    let source = emit_c(
        path.to_str().expect("UTF-8 path"),
        &dir.path().join("block-witness-out"),
    );
    let guards = emitted_guards(&source);
    assert_eq!(
        guards.len(),
        1,
        "the witness emits exactly one guard, got {guards:?}"
    );
    let (enclosing, target, axis, required) = &guards[0];
    assert_eq!(
        (enclosing.as_str(), axis.as_str(), required.as_str()),
        ("d", "0", "3"),
        "the guard sits in `d`, on axis 0, claiming 3, got {guards:?}"
    );
    assert_ne!(
        target, "__result",
        "a block-bodied return guards the producing binding, not the return boundary"
    );

    let call_path = dir.path().join("call_witness.ch");
    fs::write(&call_path, call_bodied_root(3, TWO_BY_FOUR)).expect("write call fixture");
    let call_source = emit_c(
        call_path.to_str().expect("UTF-8 path"),
        &dir.path().join("call-witness-out"),
    );
    assert_eq!(
        emitted_guards(&call_source),
        vec![("d".into(), "__result".into(), "0".into(), "3".into())],
        "the caller's declaration remains visible when its frame is forwarded"
    );
    let no_forwarding = call_source.replace(", __chelis_result_claims, &", ", NULL, &");
    assert!(
        emitted_guards(&no_forwarding).is_empty(),
        "an unused frame declaration must not certify a forwarded guard"
    );
}

/// THE CENSUS. Every executable Phase 0 example is offered to C. Recorded
/// capability refusals are asserted; local and forwarded declared-result
/// obligations are read back from the generated program. The keyed State
/// wrapper retains four checked result axes through its tuple; the positive
/// and per-axis negative Eval/C witnesses live in `key_pair_host_staging`.
///
/// Entry guards have a different contract: compare their emitted conditions
/// and labels with an independent parameter-axis traversal of checked source
/// signatures, including functions where the reader finds no guards. Existing
/// example parity supplies execution receipts; the signature oracle's mutation
/// controls make missing, duplicate, reordered and wrong-axis guards fail.
///
/// # Coverage
///
/// It censuses whatever `executable_examples()` finds, with no count written
/// here. chelis#1787 is what the hand-written count cost: it was pinned at 31,
/// an unrelated merge added a 32nd example without touching this line, and
/// `main` went red on a number rather than on a finding. A count derived from
/// the same `read_dir` would be worse, since it agrees with a reader that has
/// stopped enumerating.
///
/// Whether the directory still holds the corpus the project decided to ship is
/// a different question with an owner: `parity.rs`'s `parity_corpus_is_complete`
/// holds the roster and fails naming the drift in either direction. It runs per
/// pull request, and `faithful_observation_phase3_oracle.py` freezes its
/// definition digest, deleting a line from that very definition in four
/// mutation controls, so the roster cannot be edited quietly. This census does
/// not repeat that comparison, and it must not restate the roster here: a
/// roster compared against the directory it was read from measures nothing.
///
/// What stands in for that comparison is the ENUMERATION WITNESS below. Every
/// entry of `REFUSED_BY_A_CAPABILITY_GATE` must be reached, so an enumeration
/// that returned nothing, or that stopped seeing files it used to see, fails
/// here naming the example it lost. That is an independent fact about the
/// directory, which a count derived from the same `read_dir` would not be.
#[test]
fn no_shipped_example_gains_a_host_lane_guard() {
    let dir = tempdir().expect("tempdir");
    let examples = executable_examples();

    let mut census: Vec<String> = Vec::new();
    let mut refused: Vec<String> = Vec::new();
    for example in &examples {
        let stem = example.file_stem().expect("stem").to_str().expect("UTF-8");
        let name = example.file_name().expect("name").to_str().expect("UTF-8");
        let path = example.to_str().expect("UTF-8 path");
        let out = dir.path().join(stem);
        if let Some((_, diagnostic)) = refused_by_a_capability_gate()
            .iter()
            .find(|(refused_name, _)| *refused_name == name)
        {
            let build = build_c(path, &out);
            let stderr = String::from_utf8_lossy(&build.stderr).to_string();
            assert!(
                build.status.code() == Some(1) && stderr.contains(diagnostic),
                "{name} is recorded as refused by a capability gate, but the build \
                 succeeded or failed for another reason: {stderr}"
            );
            refused.push(name.to_string());
            continue;
        }
        let source = emit_c(path, &out);
        for (function, target, axis, required) in emitted_guards(&source) {
            census.push(format!(
                "{stem}: {function} guards {target} axis {axis} claiming {required}"
            ));
        }
        assert_signature_entry_inventory(
            stem,
            &fs::read_to_string(example).expect("read example"),
            &source,
        );
    }

    // THE ENUMERATION WITNESS. Both sides are sorted, so this asks whether
    // every recorded refusal was reached and nothing else was; the order the
    // constant happens to be declared in is not part of the contract.
    let mut expected_refusals: Vec<String> = refused_by_a_capability_gate()
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect();
    expected_refusals.sort();
    refused.sort();
    assert_eq!(
        refused, expected_refusals,
        "every recorded refusal must be reached: a name that no longer matches an \
         example, or an enumeration that returned nothing, silently shrinks the census"
    );

    assert_eq!(
        census,
        vec![
            "keyed_state_wrapper: main guards __let_19 axis 0 claiming 2",
            "keyed_state_wrapper: main guards __let_19 axis 1 claiming 1",
            "keyed_state_wrapper: main guards __let_19 axis 2 claiming 1",
            "keyed_state_wrapper: main guards __let_19 axis 3 claiming 1",
        ],
        "these shipped defs changed their host-lane result guards; each change \
         needs positive and negative both-lane checks before it lands"
    );
}
