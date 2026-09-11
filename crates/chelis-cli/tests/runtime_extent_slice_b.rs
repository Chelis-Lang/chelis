//! Public CLI acceptance rows for runtime-extent Slice B (chelis#1277).
//!
//! `crates/chelis-ir/tests/runtime_extent_slice_b_classes.rs` locks the
//! DERIVATION: which output axes form one equality class, and in what order.
//! This file locks the CLI-observable consequence: that a class's guard is
//! ordered against an independent trap the way `spec/04-type-system.md`
//! section 4.7 requires, and that chelis#1482's missing extent source is a
//! typed receipt rather than an ICE.
//!
//! ## Which guard rows belong here, and which do not
//!
//! The distinction is the ROOT FORM, not the CLI. A `def main() = f(...)`
//! root inlines `f`, and the emitter gives the root its own kernel carrying
//! no entry guard, so the program runs unguarded on both lanes: measured at
//! `1a585ba1b` for chelis#1377's shape, where `chelis eval --file` and the
//! linked binary both print `shape=[5]` under a declared `tensor[4, f32]`
//! and both exit zero. A top-level VALUE BINDING `out = f(...)` applies the
//! exported kernel instead, so `f`'s entry guards run at the call on eval
//! (B2h's routing) and in the linked binary on C. Every guard row in this
//! file is therefore a value binding, and the rows that need a hand-built
//! DAG with inputs bound at the API boundary, rather than a `.ch` fixture,
//! are the ones with no source spelling: those are the DRIVEN rows in
//! `crates/chelis-backend-c/tests/exec_compile.rs` and the API rows in
//! `crates/chelis-ir/tests/runtime_extent_slice_b_eval_guard.rs`.
//!
//! An earlier version of this paragraph said no guard row could live here at
//! all. That was true of the inlined-root form it measured and over-broad for
//! the value-binding form, which B2h's rows already used.
//!
//! The paragraph above is now history rather than current behaviour, and the
//! chelis#1782 rows at the end of this file are the reason it has to say so.
//! PR #1773 gave the inlined root the callee's own declared result claim and
//! PR #1790 gave it an op-computed one, so a `def main() = f(...)` root is
//! guarded on both lanes today: chelis#1377's shape, which that paragraph
//! measured running unguarded at `1a585ba1b`, traps at `d861a6c6f` with
//! ``extent `4`: claimed = 4, x axis 0 = 5``, which
//! `an_independent_root_literal_claim_still_names_its_claimed_extent`
//! asserts. The root rows in this file are therefore guard rows, and they
//! are marked as such where they sit.
//!
//! The order controls belong here for a second reason: ordering a guard
//! against an in-body trap needs a caller, and a called function's entry sits
//! exactly where its call sits in the caller's source order.
//!
//! chelis#1375's two rows belong here for a third reason, recorded at their
//! own section below: the inlining that erases a class is `def main() =
//! f(...)`'s, and a top-level BINDING `out = f(...)` does not inline, so the
//! def is lowered standalone with its declared extent still symbolic and the
//! guard is reachable from the CLI on both lanes.
//!
//! ## Why the guard-order controls are shaped the way they are
//!
//! A test that only asks "does a mismatching extent trap" cannot tell a guard
//! placed correctly from one hoisted to entry or deferred to the end of the
//! computation. Section 4.7 fixes the position exactly:
//!
//! > A guard that compares a locally computed value ... takes the source
//! > position of the operation that introduces the guarded extent: an
//! > independent effect or trap that precedes that operation in source order
//! > is observed first, and one that follows it is observed only if the guard
//! > passes.
//!
//! So each control puts an INDEPENDENT trap on one side of a mismatching
//! extent and asserts which of the two failures the lane reports. A guard
//! hoisted too early fails the "trap before" row; one deferred too late fails
//! the "trap after" row. Neither direction is detectable without the other,
//! and correct placement is the only placement satisfying both.
//!
//! The independent trap divides by a RUNTIME zero rather than a literal
//! `0i64`, so constant folding cannot turn it into a check-time rejection and
//! remove the control's teeth. The extent under test is locally computed
//! (chelis#1379's `mul(shape(x, 0), 2i64)`) rather than interface-valued: an
//! all-interface class runs at ENTRY, before any other operation of the
//! function, so ordering one against an in-body trap would make both controls
//! pass wherever the local guards went.
//!
//! ## Trap rendering
//!
//! Section 4.7 makes a runtime extent guard a typed operation-precondition
//! guard under [04-NUM-9], so the complete user-facing line is
//! `numeric trap: domain in <op> at int64`, with no prefix and no suffix. The
//! `<op>` slot names the operation introducing the guarded extent, which for
//! an all-interface class is "the `load` primitive of the later witness in
//! signature order". Section 4.7 also requires each lane to convey the
//! disagreeing source names, the axis, and each observed value on separate
//! accompanying lines, but binds "the information conveyed and not the bytes
//! rendered", so the assertions below check the trap LINE byte-exactly and the
//! context by content, never against an invented cross-lane format.

mod common;

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::TempDir;

use common::{gcc_available, link_generated};

/// Chelis#1482's remaining reproducer: a runtime-bound `shrink` under a
/// symbolic signature, consumed by a composite elementwise lowering that
/// still synthesizes anonymous-extent constants.
const RUNTIME_BOUND_SHRINK_SIGMOID: &str = "module Repro.M1\n\
sig f: tensor[n, f32] -> tensor[u, f32]\n\
def f(x) = {\n  \
k = shape(x, cast(0, int32))\n  \
z = cast((k - k), int64)\n  \
s = shrink(x, [[z, k]])\n  \
sigmoid(s)\n\
}\n\
out = f(to_tensor([cast(1.0, f32), cast(-2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";

/// The original #1482 spelling. Chelis#1313's structural ReLU identity no
/// longer synthesizes a tensor zero, so this exact program now executes.
const RUNTIME_BOUND_SHRINK_RELU: &str = "module Repro.M1\n\
sig f: tensor[n, f32] -> tensor[u, f32]\n\
def f(x) = {\n  \
k = shape(x, cast(0, int32))\n  \
z = cast((k - k), int64)\n  \
s = shrink(x, [[z, k]])\n  \
relu(s)\n\
}\n\
out = f(to_tensor([cast(1.0, f32), cast(-2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";

/// The same elementwise shape over a plain symbolic tensor. This builds on
/// `main` and must keep building: its zero fill carries the declared
/// dimension `n`, not an anonymous extent.
const SYMBOLIC_RELU: &str = "module Probe.SymRelu\n\
sig f: tensor[n, f32] -> tensor[n, f32]\n\
def f(x) = relu(x)\n\
out = f(to_tensor([cast(1.0, f32), cast(-2.0, f32)]))\n";

/// The chelis#597 reproducer: a positional `expand` over a unit axis.
///
/// `spec/05-risc-primitives.md` §2.4 gives `expand` the size-1 broadcast that
/// leaves the rank alone. Before chelis#1277's S2b, lowering rewrote every
/// surface `expand` into the rank-increasing form and overwrote the type the
/// checker had stamped, so no program could reach the same-rank node the
/// verifier, both evaluators and the C emitter all implement.
const SAME_RANK_BROADCAST: &str = "module Repro.Expand597\n\
def broadcast(x: tensor[2, 1, f32]) -> tensor[2, 3, f32] = expand(&x, 1, 3i64)\n\
out = broadcast(to_tensor([[cast(1.0, f32)], [cast(2.0, f32)]]))\n";

/// A zero broadcast extent. `spec/04-type-system.md` §4.7.2 makes a static
/// NEGATIVE size a type error and says nothing against zero, so zero is legal
/// and declares an empty axis.
const ZERO_BROADCAST: &str = "module Repro.ExpandZero\n\
def empty(x: tensor[2, 1, f32]) -> tensor[2, 0, f32] = expand(&x, 1, 0i64)\n\
out = empty(to_tensor([[cast(1.0, f32)], [cast(2.0, f32)]]))\n";

/// An operand whose extent is computed INSIDE the function, so section 4.7
/// places the claim's guard locally rather than at entry.
///
/// `shrink` with a runtime end produces an op-declared extent: no input
/// carries it, so the entry guard cannot see it and the claim has to be
/// checked at the operation that makes it.
const LOCAL_NON_UNIT_SOURCE: &str = "module Repro.ExpandLocalRefuted\n\
sig f: tensor[n, f32] -> tensor[3, f32]\n\
def f(x) = expand(shrink(&x, [[0i64, sub(shape(&x, 0), 1i64)]]), 0, 3i64)\n\
out = f(to_tensor([cast(7.0, f32), cast(9.0, f32), cast(11.0, f32)]))\n";

/// The same shape over an operand the shrink leaves at extent 1.
const LOCAL_UNIT_SOURCE: &str = "module Repro.ExpandLocalSatisfied\n\
sig f: tensor[n, f32] -> tensor[3, f32]\n\
def f(x) = expand(shrink(&x, [[0i64, sub(shape(&x, 0), 2i64)]]), 0, 3i64)\n\
out = f(to_tensor([cast(7.0, f32), cast(9.0, f32), cast(11.0, f32)]))\n";

/// One locally computed unit axis feeding TWO `expand` nodes.
///
/// Both claims land on the same (operand node, axis) key with the same claim,
/// canonical and operation, so one emitted guard satisfies both.
const TWO_EXPANDS_OVER_ONE_OPERAND: &str = "module Repro.TwoExpands\n\
sig f: tensor[n, f32] -> tensor[f32]\n\
def f(x) = {\n  \
s = shrink(x, [[0i64, sub(shape(x, 0), 2i64)]])\n  \
a = expand(s, 0, 5i64)\n  \
b = expand(s, 0, 4i64)\n  \
add(sum(a, 0), sum(b, 0))\n\
}\n\
out = f(to_tensor([7.0f32, 9.0f32, 11.0f32]))\n";

/// The same program shrunk to two elements, which refutes the shared claim.
const TWO_EXPANDS_REFUTED: &str = "module Repro.TwoExpandsRefuted\n\
sig f: tensor[n, f32] -> tensor[f32]\n\
def f(x) = {\n  \
s = shrink(x, [[0i64, sub(shape(x, 0), 1i64)]])\n  \
a = expand(s, 0, 5i64)\n  \
b = expand(s, 0, 4i64)\n  \
add(sum(a, 0), sum(b, 0))\n\
}\n\
out = f(to_tensor([7.0f32, 9.0f32, 11.0f32]))\n";

/// A literal operand extent that refutes the claim statically.
const STATIC_NON_UNIT_SOURCE: &str = "module Repro.ExpandStaticNonUnit\n\
def bad(x: tensor[2, 4, f32]) -> tensor[2, 3, f32] = expand(&x, 1, 3i64)\n";

/// A symbolic operand extent that refutes the claim at run time. The checker
/// admits it, because `spec/05-risc-primitives.md` §2.4.1 makes only a LITERAL
/// non-unit extent a type error and sends every other spelling to the runtime
/// extent guard.
const RUNTIME_NON_UNIT_SOURCE: &str = "module Repro.ExpandRuntimeNonUnit\n\
sig broadcast: tensor[n, f32] -> tensor[3, f32]\n\
def broadcast(x) = expand(&x, 0, 3i64)\n\
out = broadcast(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\n";

/// The same program over an operand whose extent DOES satisfy the claim. It is
/// the control that says the guard fires on the disagreement rather than on
/// the symbolic spelling.
const RUNTIME_UNIT_SOURCE: &str = "module Repro.ExpandRuntimeUnit\n\
sig broadcast: tensor[n, f32] -> tensor[3, f32]\n\
def broadcast(x) = expand(&x, 0, 3i64)\n\
out = broadcast(to_tensor([cast(5.0, f32)]))\n";

fn fixture(dir: &TempDir, name: &str, source: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, source).expect("fixture");
    path
}

fn build_c(path: &Path, out_dir: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build")
}

fn eval(path: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--allow-style-violations",
            "--file",
            path.to_str().unwrap(),
        ])
        .output()
        .expect("eval")
}

/// Chelis#1482 / `runtime_extents.md` C4.3: sigmoid's synthesized constants
/// still have a missing extent source and therefore produce the registered
/// typed receipt, not the occurrence pass's internal compiler error or an
/// input extent silently substituted for the real one.
///
/// Eval and the plain-symbolic control remain negative parity: the gap is the
/// source-less synthesized constant under a runtime-bound result, not sigmoid
/// or symbolic elementwise execution generally.
#[test]
fn runtime_bound_shrink_consumed_elementwise_reports_a_typed_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "repro.ch", RUNTIME_BOUND_SHRINK_SIGMOID);

    let build = build_c(&path, &dir.path().join("repro-out"));
    let stderr = String::from_utf8_lossy(&build.stderr).to_string();
    assert!(
        !build.status.success(),
        "the build must refuse rather than emit undeclared C: {stderr}"
    );
    assert!(
        !stderr.contains("internal compiler error"),
        "the ICE must be replaced by the receipt, not accompanied by it: {stderr}"
    );
    assert_eq!(
        build.status.code(),
        Some(1),
        "a typed refusal exits 1, where the panic exited 101: {stderr}"
    );
    assert!(
        stderr.contains("unsupported: "),
        "the receipt carries the section C2 brand: {stderr}"
    );
    assert!(
        stderr.contains("unimplemented chelis#1482"),
        "the receipt cites the issue that owns the gap: {stderr}"
    );
    assert!(
        stderr.contains("extent source(s) for"),
        "the receipt names the axis that has no source: {stderr}"
    );
    assert!(
        stderr.contains("(codegen:c)"),
        "the receipt names the lane that refused: {stderr}"
    );

    // The eval lane refuses with the same typed receipt as C, exit 1, no
    // panic. Before chelis#1277 B2h the host interpreter evaluated this
    // program (it computes shapes from values and never sees the anonymous
    // extent); B2h applies `f` through the kernel the C lane emits for it,
    // so the DAG evaluator's cardinality check refuses exactly where C's
    // does. That is B2h's eval-lane capability regression on chelis#1482,
    // recorded there beside S2a's compiled-lane one; both lanes recover
    // together when the const gains its extent source.
    let evaluated = eval(&path);
    let stderr = String::from_utf8_lossy(&evaluated.stderr).to_string();
    assert!(
        !evaluated.status.success(),
        "eval refuses as C does through the routed kernel: {}",
        String::from_utf8_lossy(&evaluated.stdout)
    );
    assert_eq!(
        evaluated.status.code(),
        Some(1),
        "a typed refusal, not a panic: {stderr}"
    );
    assert!(!stderr.contains("internal compiler error"), "{stderr}");
    assert!(stderr.contains("unsupported: "), "{stderr}");
    assert!(stderr.contains("unimplemented chelis#1482"), "{stderr}");
    assert!(stderr.contains("extent source(s) for"), "{stderr}");
    assert!(
        stderr.contains("(runtime)"),
        "the receipt names the lane that refused: {stderr}"
    );

    // The control that decides how narrow the rule had to be: an
    // elementwise fill under a DECLARED dimension is not a sourceless axis.
    let symbolic = fixture(&dir, "sym_relu.ch", SYMBOLIC_RELU);
    let build = build_c(&symbolic, &dir.path().join("sym-out"));
    assert!(
        build.status.success(),
        "a declared dimension on an elementwise fill still builds: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let evaluated = eval(&symbolic);
    assert!(
        String::from_utf8_lossy(&evaluated.stdout).contains("[1.0, 0.0]"),
        "{}",
        String::from_utf8_lossy(&evaluated.stdout)
    );
}

/// Part of chelis#1482 via chelis#1313: the original ReLU spelling is now a
/// positive build-link-run/eval regression. The dedicated identity removes
/// exactly the anonymous tensor-zero operand that triggered the source walk;
/// this does not claim the broader synthesized-constant class is closed.
#[test]
fn runtime_bound_shrink_relu_builds_and_matches_eval_exactly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "relu_repro.ch", RUNTIME_BOUND_SHRINK_RELU);
    let out_dir = dir.path().join("relu-repro-out");

    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "dedicated ReLU must build without a synthesized Const receipt: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let linked = common::link_generated(&out_dir, "relu_repro.c", "relu_repro");
    assert!(linked.success(), "generated ReLU program must link");
    let compiled = std::process::Command::new(out_dir.join("relu_repro"))
        .output()
        .expect("compiled ReLU program");
    assert!(
        compiled.status.success(),
        "compiled ReLU program failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );

    let evaluated = eval(&path);
    assert!(
        evaluated.status.success(),
        "eval ReLU program failed: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let compiled_stdout = String::from_utf8_lossy(&compiled.stdout);
    let evaluated_stdout = String::from_utf8_lossy(&evaluated.stdout);
    assert_eq!(
        compiled_stdout, evaluated_stdout,
        "compiled C and eval must agree byte-for-byte"
    );
    assert!(
        compiled_stdout.contains("shape=[4]")
            && compiled_stdout.contains("data=[1.0, 0.0, 3.0, 4.0]"),
        "ReLU must retain the exact original result: {compiled_stdout}"
    );
}

// ===========================================================================
// b2.1: guard placement, guard order, and the row moves Slice B owns.
//
// Everything below reuses `fixture`, `build_c`, and `eval` above rather than
// re-deriving them, so every row here passes the same `--allow-style-
// violations` and `CHELIS_STYLE_GATE_DISABLE` handling as the row B1 landed.
// ===========================================================================

/// The exact [04-NUM-9] line an extent guard renders. `<op>` varies with the
/// operation introducing the guarded extent; `<prim>` is always `int64`,
/// because the guard finalizes an extent under [05-DIM-1] rather than a
/// tensor element.
fn domain_trap_line(op: &str) -> String {
    format!("numeric trap: domain in {op} at int64")
}

const DIV_ZERO_TRAP: &str = "numeric trap: division by zero in floor_div at int64";

/// Build to C, link, run, and return whether the binary exited zero plus its
/// combined output. A build or link failure panics: those are defects in the
/// fixture or the emitter, never the behaviour under test.
fn c_run_result(dir: &TempDir, stem: &str, source: &str) -> (bool, String) {
    let out_dir = dir.path().join(format!("{stem}-out"));
    let build = build_c(&fixture(dir, &format!("{stem}.ch"), source), &out_dir);
    assert!(
        build.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let status = link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(status.success(), "link failed: {status}");
    let run = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run compiled binary");
    let mut text = String::from_utf8_lossy(&run.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&run.stderr));
    (run.status.success(), text)
}

// ---------------------------------------------------------------------------
// Guard-order controls.
// ---------------------------------------------------------------------------

/// A LITERAL claim of 4 over a read that yields 5, chelis#1377's shape.
///
/// `insert` since S2a: `b` is rank 0, so this is the rank-RAISING form, which
/// that change gave its own name. The IR node and every guard row here are
/// unchanged by the spelling.
///
/// The extent this guards is an interface value - an input tensor's axis,
/// read by the `expand` size - so `spec/04-type-system.md` section 4.7 places
/// the guard at the ENTRY of the function that reads it. That is `f`, not the
/// root, and a called function's entry sits exactly where its call sits in
/// the caller's source order, so the control still discriminates: the guard
/// is observed after anything the caller evaluates before `widened = f(...)`
/// and before anything it evaluates after. An earlier draft of this comment
/// said the placement was Local at the `expand`, which was the belief before
/// the third placement refinement; the emitted C says otherwise and the
/// emitted C is the oracle.
///
/// A class whose witnesses are two `Load` axes would NOT work here: its guard
/// runs in `f`'s prologue either way, so both controls would pass wherever
/// the guards went. chelis#1379's arithmetic form would not work either -
/// the C lane rejects `mul(shape(x, 0), 2i64)` outright under chelis#469, and
/// a control that fails at build time is not measuring order.
fn guard_order_source(claim: u32, trap_first: bool) -> String {
    let trap = "boom = floor_div(1i64, sub(shape(xb, 0), shape(xb, 0)))";
    let widen = "widened = f(seed, x)";
    let (first, second) = if trap_first {
        (trap, widen)
    } else {
        (widen, trap)
    };
    format!(
        "def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[{claim}, f32] = insert(b, 0, shape(x, 0))\n\
         x = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32])\n\
         xb = to_tensor([1.0f32, 2.0f32])\n\
         seed = sum(to_tensor([1.0f32]), 0)\n\
         {first}\n{second}\n"
    )
}

/// The claim that disagrees with the read (4 against an extent of 5).
const MISMATCHED: u32 = 4;

/// EVIDENTIARY STATUS: disposition lock, as its eval twin.
#[test]
fn c_independent_trap_before_a_mismatch_wins() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "before_c", &guard_order_source(MISMATCHED, true));
    assert!(!ok, "the binary must fail: {out}");
    assert!(
        out.contains(DIV_ZERO_TRAP),
        "the earlier independent trap must be observed first: {out}"
    );
    assert!(
        !out.contains("numeric trap: domain in"),
        "the extent guard must not have run yet: {out}"
    );
}

/// EVIDENTIARY STATUS: regression test, but it fails for a DIFFERENT reason
/// from its eval twin, and the difference is chelis#1377's `lane_divergent`
/// recording. `main`'s C lane already reports something first here, the input
/// shape preamble's static-dim check (`emit.rs`, the `known_dim_size` arm):
/// `f__tensor_0: input `x` axis 0 expected 4, got 5`, followed by `abort()`.
/// So the C lane gets the ORDER right today by a mechanism that is not a
/// runtime extent guard, in a rendering [04-NUM-9] does not permit, while
/// eval has no check at all. That check is the identical comparison to this
/// class's guard, and b2.4 narrows it away for exactly that overlap, so what
/// runs here now is the guard. Asserting the exact trap line is what makes
/// this row fail on `main` rather than pass on the neighbouring behaviour.
#[test]
fn c_independent_trap_after_a_mismatch_loses() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "after_c", &guard_order_source(MISMATCHED, false));
    assert!(!ok, "the binary must fail: {out}");
    assert!(
        out.contains(&domain_trap_line("load")),
        "the extent guard introduces the extent and fires first: {out}"
    );
    assert!(
        out.contains("extent `4`: claimed = 4, x axis 0 = 5"),
        "with section 4.7's context on its own line: {out}"
    );
    assert!(
        !out.contains(DIV_ZERO_TRAP),
        "the later independent trap must not be reached: {out}"
    );
}

// ---------------------------------------------------------------------------
// HIP prologue.
// ---------------------------------------------------------------------------

/// No environment variable turns the section 4.7.2 guard off.
///
/// Two conditions on `std::env::var_os` once gated the two claim-minting sites,
/// live in a release binary, named nowhere else in the repository. With
/// `CHELIS_NO_KERNEL_CLAIM=1` this exact fixture printed
/// `out = tensor(shape=[3], data=[7.0, 7.0, 7.0])`, which is verbatim the
/// silent wrong shape chelis#1374 exists to remove, and the emitted C guard
/// count went from one to zero.
///
/// The names are gone. This row keeps them gone: it sets both, plus the
/// spelling a rename would reach for, and requires the trap anyway. A test seam
/// for this obligation belongs behind `#[cfg(test)]`, never behind an
/// environment read a user can make.
///
/// EVIDENTIARY STATUS: regression test. Measured red at `34d3037dc` with either
/// variable set; the red-team round that found it recorded the same output.
#[test]
fn no_environment_variable_disables_the_named_claim_guard() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "no_switch.ch", &named_claim_cross_tensor_source(2, 3));
    for name in [
        "CHELIS_NO_KERNEL_CLAIM",
        "CHELIS_NO_STAGED_CLAIM",
        "CHELIS_TEST_NO_KERNEL_CLAIM",
        "CHELIS_TEST_NO_STAGED_CLAIM",
    ] {
        let eval = Command::cargo_bin("chelis")
            .expect("chelis")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env(name, "1")
            .args([
                "eval",
                "--allow-style-violations",
                "--file",
                path.to_str().unwrap(),
            ])
            .output()
            .expect("eval");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&eval.stdout),
            String::from_utf8_lossy(&eval.stderr)
        );
        assert!(
            text.contains("extent `rows`: x axis 0 = 2, y axis 0 = 3"),
            "`{name}` must not silence the guard: {text}"
        );
        assert!(
            text.contains(&domain_trap_line("load")),
            "`{name}` must not silence the trap: {text}"
        );
        assert!(
            !text.contains("out = tensor(shape=[3]"),
            "`{name}` must not restore the wrong shape: {text}"
        );
    }

    let out_dir = dir.path().join("no-switch-out");
    let build = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_NO_KERNEL_CLAIM", "1")
        .env("CHELIS_NO_STAGED_CLAIM", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build");
    assert!(
        build.status.success(),
        "build: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted = fs::read_to_string(out_dir.join("no_switch.c")).expect("C source");
    assert_eq!(
        emitted.matches("extent `rows`").count(),
        1,
        "the emitted guard count stays at one with both variables set:\n{emitted}"
    );
}

/// A repeated binder is checked when BOTH occurrences are tensor parameters.
/// One inside a container type is not, and this row measures that rather than
/// leaving it implied.
///
/// `prepare_parameter_witnesses` walks `&[TensorType]`, so a binder reached
/// only through `List[tensor[extent, f32]]` mints no witness and nothing
/// compares it. The claim sentence in this pull request is qualified to
/// tensor-typed parameters for that reason. Extending the walk into container
/// types is a mechanism this change does not carry; chelis#1266 owns the
/// record-projection half of the same shape.
///
/// EVIDENTIARY STATUS: disposition lock naming a gap, NOT a regression test.
/// The `_pending` suffix says the recorded behaviour is the one to change.
#[test]
fn a_container_nested_binder_is_unchecked_pending_the_container_walk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, output) = c_run_result(
        &dir,
        "container_binder",
        "def f(xs: List[tensor[extent, f32]], p: tensor[extent, f32]) -> tensor[f32] = sum(&p, 0i32)\n\
         out = f([to_tensor([1.0f32, 2.0f32])], to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
    assert!(ok, "the program runs to completion today: {output}");
    assert!(
        !output.contains("extent `extent`"),
        "and nothing compares the list element's extent with `p`'s: {output}"
    );
    assert!(
        output.contains("out = 6"),
        "the sum of `p` is what it returns: {output}"
    );
}

/// A synthesized multi-root kernel takes its parameters from captured root
/// bindings, and two of them spelling the same binder is a coincidence, not a
/// claim.
///
/// `id2` and `total` are unrelated definitions that each named an axis `seq`.
/// Their results become top-level roots, the compiler synthesizes one kernel
/// over those roots, and the roots' types carry the spelling in. Reading that
/// as a signature assertion compares `y`'s axis 1 (extent 2) with `total`'s
/// argument axis 0 (extent 3) and aborts a correct program.
///
/// Every line earns its place: the `grad` root is what routes these roots
/// through the staged host partition, and without it the synthesized kernel
/// never forms. `out_tl` gives the collision a second `batch` occurrence, so a
/// repair that only suppressed the disagreeing pair would still fail here.
///
/// EVIDENTIARY STATUS: regression test. Measured red at `01c6e33a1`, where the
/// emitted host source carried one `extent `batch`` and one `extent `seq``
/// guard and the linked binary exited on
/// "extent `seq`: __host_tensor_arg_1 axis 0 = 3, y axis 1 = 2".
#[test]
fn a_synthesized_kernel_parameter_collision_emits_no_guard() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(
        &dir,
        "collision.ch",
        "def id2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x)\n\
         def total(x: &tensor[seq, f32]) -> f32 = tensor_to_scalar(sum(x, seq))\n\
         y = id2(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n\
         out_tl = sum(y, seq)\n\
         outt = total(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n\
         gr = grad(total)(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
    let out_dir = dir.path().join("collision-out");
    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "the fixture builds: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted = fs::read_to_string(out_dir.join("collision.c")).expect("C source is written");
    assert!(
        !emitted.contains("extent `seq`"),
        "no author related the two `seq` axes:\n{emitted}"
    );
    assert!(
        !emitted.contains("extent `batch`"),
        "nor the `batch` ones, which happen to agree and would hide the defect:\n{emitted}"
    );
    let status = link_generated(&out_dir, "collision.c", "collision");
    assert!(status.success(), "link failed: {status}");
    let run = StdCommand::new(out_dir.join("collision"))
        .output()
        .expect("run compiled binary");
    let mut output = String::from_utf8_lossy(&run.stdout).to_string();
    output.push_str(&String::from_utf8_lossy(&run.stderr));
    assert!(
        run.status.success(),
        "and the program runs to completion: {output}"
    );
    assert!(
        output.contains("outt = 6"),
        "with its own result intact: {output}"
    );
}

/// An `ExtentWitness` retained ONLY as a section 4.7 entry obligation reaches
/// the HIP device lane and the program emits.
///
/// chelis#1374's repeated-binder claim keeps a declared-but-unread parameter's
/// interface witness alive. So does every ordinary signature that repeats a
/// binder, `def f(a: tensor[batch, in_dim], b: tensor[in_dim, out_dim])`
/// included. Before the entry-obligation split the HIP gate read that witness
/// as the runtime `shape` value read [05-SHAPE-1] excludes and refused the
/// whole build, which would have made this pull request a breaking change to
/// the HIP target for most programs.
///
/// This is an EMITTED-SOURCE receipt. HIP hardware execution is blocked on this
/// workstation (`docs/local_hip_environment.md` covers the manual gates that
/// are runnable; running a compiled kernel is not among what this row can do),
/// so the assertion is on the text `chelis build --target hip` writes.
///
/// The guard text this row asserts is the LEGACY one. `witness_entry_obligations`
/// omits a claim the entry schedule already owns, and on the HIP lane that
/// schedule is `symbolic_bindings_interface`, which renders
/// ``symbolic dim `x` mismatch`` and `abort()` rather than [04-NUM-9] and
/// `chelis_numeric_trap`. Both lanes derive the comparison from the same
/// `derive_dim_witnesses` classes, so the equality is checked either way and no
/// program runs unguarded; only the spelling diverges. chelis#1786 owns that
/// divergence, and this row locks the legacy form until it lands rather than
/// asserting the [04-NUM-9] form HIP does not yet produce. (chelis#1112 is the
/// HIP runtime's `chelis_gpu_alloc` int-extent signature, an extent-WIDTH issue
/// that changes no diagnostic text; it cannot close this row.)
///
/// EVIDENTIARY STATUS: regression test for the BUILD, disposition lock for the
/// spelling. Measured red at `01c6e33a1`, where the build failed with
/// "`chelis build --target hip` does not support the runtime `shape` value
/// read; lowered node 2 requires it."
#[test]
fn an_entry_obligation_witness_emits_the_legacy_hip_guard_pending_1786() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(
        &dir,
        "unread_hip.ch",
        "def f(x: tensor[extent, f32], p: tensor[extent, f32]) -> tensor[f32] = sum(x, 0i32)\n",
    );
    let out_dir = dir.path().join("unread-hip-out");
    let build = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("hip build");
    assert!(
        build.status.success(),
        "the retained witness is an entry obligation, not a device shape read: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted =
        fs::read_to_string(out_dir.join("unread_hip_hip.cpp")).expect("HIP source is written");
    assert!(
        emitted.contains("input `p` at slot 1"),
        "the declared-but-unread parameter keeps its ABI slot:\n{emitted}"
    );
    assert!(
        emitted.contains("symbolic dim `extent` mismatch"),
        "and its equality is checked at entry, in this lane's rendering:\n{emitted}"
    );
    assert!(
        !emitted.contains("chelis_gpu_tensor *d_t2 ="),
        "the witness itself emits no device value:\n{emitted}"
    );
}

/// chelis#616 left the HIP prologue walking the legacy occurrence list and
/// asserting a `Load` source for every occurrence, with a `panic!` backstop
/// (`require_load_source`) for the op-declared case on the reasoning that
/// `reject_unsupported_hip_ops` had already refused it. It had not:
/// `examples/transformer_block.ch` reaches that panic on `main`, so the whole
/// program emits nothing on HIP.
///
/// The derived interface witnesses contain only `Load` sources by
/// construction - `symbolic_bindings_interface` maps `ExternalAxis` members
/// and nothing else - so the prologue never meets an op-declared occurrence
/// and the backstop is unreachable rather than merely unhit. This row asserts
/// what the user gets: the program emits.
///
/// EVIDENTIARY STATUS: regression test, measured red by restoring the HIP
/// emitter's two call sites to the legacy name-keyed grouping, which
/// reproduced the panic on this exact program. That reproduction is no
/// longer performable: chelis#665 deleted the grouping and the `panic!` arm
/// with it, so what this row pins now is that the program still emits.
#[test]
fn an_op_declared_witness_reaches_the_hip_prologue_without_panicking() {
    let example = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("examples/transformer_block.ch");
    let dir = tempfile::tempdir().expect("tempdir");
    let out_dir = dir.path().join("hip-out");
    let build = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            example.to_str().unwrap(),
            "--target",
            "hip",
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build");
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        !stderr.contains("op-declared runtime dim"),
        "the prologue must not reach chelis#616's backstop: {stderr}"
    );
    assert!(
        build.status.success(),
        "the HIP build must succeed: {stderr}"
    );
    let emitted = fs::read_to_string(out_dir.join("transformer_block_hip.cpp"))
        .expect("HIP host source is written");
    assert!(
        emitted.contains("int64_t seq = chelis_tensor_shape("),
        "and the interface binding is declared from its Load axis: {}",
        &emitted[..emitted.len().min(400)]
    );
}

/// Every local folded read of `seq` is guarded at its operation. The
/// current example's canonical contractions and explicit epsilon produce
/// seventeen such reads: nine in projections/attention, three in each
/// normalization block, and two in the feed-forward projections. The emitted nodes/axes below pin that independent
/// derivation, so a missing or duplicated guard cannot preserve the count.
#[test]
fn every_local_member_of_one_class_is_guarded_at_its_operation_on_c() {
    if !gcc_available() {
        return;
    }
    let example = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("examples/transformer_block.ch");
    let dir = tempfile::tempdir().expect("tempdir");
    let out_dir = dir.path().join("c-out");
    let build = build_c(&example, &out_dir);
    assert!(
        build.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted =
        fs::read_to_string(out_dir.join("transformer_block.c")).expect("C source is written");
    assert!(
        !emitted.contains("chelis: runtime dim `seq` mismatch"),
        "no local guard keeps the legacy rendering"
    );
    // Seventeen sites, pinned by operation and axis. The list was
    // `[(12, 0), (16, 0), (20, 0), (24, 2), (25, 0), (29, 1), (32, 1),
    // (35, 0), (39, 0), (55, 0), (56, 0), (57, 0), (60, 0), (65, 0), (81, 0),
    // (82, 0), (83, 0)]` while the context named a node id, and this multiset
    // is that list with the node component dropped: fourteen on axis 0, two on
    // axis 1, one on axis 2, every one of them an `insert`. Section 4.7 binds
    // the information conveyed rather than the bytes, and a node id is neither
    // a source name nor stable under `spec/06` section 5.2-5.4's renumbering
    // passes, so the rendering names the operation. What the multiset still
    // proves is what this row exists for: the COUNT of guarded local members
    // and the axis each one reads.
    let guarded_sites = [(0usize, 14usize), (1, 2), (2, 1)];
    assert_eq!(
        emitted.matches("extent `seq`: claimed = ").count(),
        guarded_sites.iter().map(|(_, count)| count).sum::<usize>()
    );
    for (axis, count) in guarded_sites {
        assert_eq!(
            emitted
                .matches(&format!(
                    "extent `seq`: claimed = %lld, insert axis {axis} = %lld"
                ))
                .count(),
            count,
            "axis {axis} keeps its guarded members"
        );
    }
    assert_eq!(
        emitted
            .matches(&format!("{}\");", domain_trap_line("insert")))
            .count(),
        guarded_sites.iter().map(|(_, count)| count).sum::<usize>(),
        "each local guard renders [04-NUM-9] naming its operation"
    );
}

// ---------------------------------------------------------------------------
// The eval lane (chelis#1277 B2h). A value binding applying the def is the
// eval twin of the DRIVEN C rows: `chelis eval` interprets the binding and
// applies `f` through the kernel the C lane emits for it, so `f`'s entry
// guards run at its call. A `def main() = f(..)` root is inlined on both
// lanes and its extents become literals, so no such form appears here.
// ---------------------------------------------------------------------------

/// Evaluate a fixture and return whether it succeeded plus its combined
/// stdout and stderr.
fn eval_result(dir: &TempDir, name: &str, source: &str) -> (bool, String) {
    let run = eval(&fixture(dir, name, source));
    let mut text = String::from_utf8_lossy(&run.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&run.stderr));
    (run.status.success(), text)
}

/// The guard-order fixture's claim that agrees with the read.
const AGREEING: u32 = 5;

/// expand.literal_claim.exported_kernel.eval: chelis#1377's shape through the
/// value-binding form. The literal claim propagated onto `x` through
/// inference, so the routed kernel's `Load x` is declared `[4]` and no class
/// exists for the derived guard; what fires is the DAG evaluator's check of a
/// declared literal input extent at entry, the eval analogue of the C ABI
/// preamble (B2a's "complement"), rendered per [04-NUM-9] with section 4.7's
/// context line.
///
/// EVIDENTIARY STATUS: regression test, watched failing on the tree without
/// the evaluator's literal-extent check (eval printed `shape=[5]`).
#[test]
fn a_literal_claim_over_a_runtime_read_traps_at_entry_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = guard_order_source(MISMATCHED, false).replace(
        "boom = floor_div(1i64, sub(shape(xb, 0), shape(xb, 0)))\n",
        "",
    );
    let (ok, out) = eval_result(&dir, "lit_claim.ch", &source);
    assert!(
        !ok,
        "a declared tensor[4] over a read of 5 must not execute: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `4`: claimed = 4, x axis 0 = 5"),
        "section 4.7's context line names the claim, the input and the observed extent: {out}"
    );
}

/// The positive twin: an agreeing literal executes and keeps the declared
/// shape, and is the byte-identity witness for this row (the value is the one
/// the interpreter produced before the routing).
#[test]
fn a_literal_claim_over_an_agreeing_runtime_read_executes_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = guard_order_source(AGREEING, false).replace(
        "boom = floor_div(1i64, sub(shape(xb, 0), shape(xb, 0)))\n",
        "",
    );
    let (ok, out) = eval_result(&dir, "lit_claim_ok.ch", &source);
    assert!(
        ok,
        "a declared tensor[5] over a read of 5 must execute: {out}"
    );
    assert!(out.contains("widened = tensor(shape=[5]"), "{out}");
}

/// guard_order.trap_before.eval: section 4.7, "an independent effect or trap
/// that precedes that operation in source order is observed first".
///
/// EVIDENTIARY STATUS: disposition lock. The host interpreter already reported
/// the division first because it had no extent guard at all; the row defends
/// that the routed kernel's entry guard is not hoisted ahead of the call.
#[test]
fn an_earlier_trap_preempts_the_extent_guard_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "before.ch", &guard_order_source(MISMATCHED, true));
    assert!(!ok, "the program must fail: {out}");
    assert!(
        out.contains(DIV_ZERO_TRAP),
        "the earlier independent trap is observed first: {out}"
    );
    assert!(
        !out.contains("numeric trap: domain in"),
        "the extent guard must not have run yet: {out}"
    );
}

/// guard_order.trap_after.eval: a trap that "follows it is observed only if
/// the guard passes".
///
/// EVIDENTIARY STATUS: regression test, watched failing on `main` (the
/// interpreter reported the division, the wrong answer).
#[test]
fn a_later_trap_is_preempted_by_the_extent_guard_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "after.ch", &guard_order_source(MISMATCHED, false));
    assert!(!ok, "the program must fail: {out}");
    assert!(
        out.contains(&domain_trap_line("load")),
        "the entry guard of `f` fires at its call, before the later trap: {out}"
    );
    assert!(
        !out.contains(DIV_ZERO_TRAP),
        "the later independent trap must not be reached: {out}"
    );
}

/// The two rows above prove nothing unless the SAME program with an AGREEING
/// claim runs past the guard and reaches the later trap.
///
/// EVIDENTIARY STATUS: disposition lock.
#[test]
fn the_guard_order_fixture_reaches_its_later_trap_when_the_claim_agrees_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "agree.ch", &guard_order_source(AGREEING, false));
    assert!(!ok, "the independent trap still fails the program: {out}");
    assert!(
        out.contains(DIV_ZERO_TRAP),
        "with an agreeing claim the later trap is reached: {out}"
    );
    assert!(
        !out.contains("numeric trap: domain in"),
        "an agreeing claim owes no trap: {out}"
    );
}

/// The effect-order fixture: the same claim, applied from an `IO` body that
/// prints before or after the call. The `IO` body is host on both lanes
/// (chelis#1528: the shared kernel decision keeps an effect the DAG cannot
/// carry in host code, so the C host program prints it too), and `f` is a
/// kernel on both, so the print and `f`'s entry guard are ordered by the
/// body's source order on both.
fn effect_order_source(claim: u32, effect_first: bool) -> String {
    let effect = "_ = print(\"effect\")";
    let widen = "widened = f(seed, x)";
    let (first, second) = if effect_first {
        (effect, widen)
    } else {
        (widen, effect)
    };
    format!(
        "def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[{claim}, f32] = insert(b, 0, shape(x, 0))\n\
         def run(x: tensor[n, f32]) -> tensor[{claim}, f32] ! {{ IO }} = {{\n\
         \x20 seed = sum(to_tensor([1.0f32]), 0)\n\
         \x20 {first}\n\
         \x20 {second}\n\
         \x20 widened\n\
         }}\n\
         out = run(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]))\n"
    )
}

/// guard_order.effect_before.eval: assert the effect's actual output before
/// the guard's failure, rather than only the trap (#1585).
#[test]
fn an_effect_before_the_guard_runs_when_the_guard_traps_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "effect_before.ch",
        &effect_order_source(MISMATCHED, true),
    );
    assert!(!ok, "the program must fail: {out}");
    assert_eq!(
        out.lines().filter(|line| *line == "effect").count(),
        1,
        "{out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `4`: claimed = 4, x axis 0 = 5"),
        "{out}"
    );
}

/// guard_order.effect_after.eval: an effect after the trapping guard does
/// not run; the preceding-effect test supplies its non-vacuity control.
#[test]
fn an_effect_after_the_guard_does_not_run_when_the_guard_traps_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "effect_after.ch",
        &effect_order_source(MISMATCHED, false),
    );
    assert!(!ok, "the program must fail: {out}");
    assert!(!out.lines().any(|line| line == "effect"), "{out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
}

/// Build, link and run a fixture, returning whether it exited zero, its
/// combined output, and the emitted C.
fn c_run_result_with_source(dir: &TempDir, stem: &str, source: &str) -> (bool, String, String) {
    let (ok, out) = c_run_result(dir, stem, source);
    let emitted = fs::read_to_string(
        dir.path()
            .join(format!("{stem}-out"))
            .join(format!("{stem}.c")),
    )
    .expect("emitted C");
    (ok, out, emitted)
}

/// Supplement the executed output checks with the emitted statement order:
/// inside `run`'s host body the `print` statement and the call into the
/// kernel C extracts for `f(seed, x)` (`run__tensor_N`, with `f` inlined and
/// the same entry guard `f__tensor_0` carries), in the order `effect_first`
/// names; inside that kernel the entry guard before its first allocation.
/// One host body's DEFINITION, located by NAME rather than by its exact
/// parameter list.
///
/// The emitted file carries a forward declaration and a definition for the
/// same symbol, so the name alone is ambiguous; the definition is the
/// occurrence whose parameter list is followed by `{` instead of `;`.
///
/// Matching the full signature instead is what chelis#1808 was: chelis#1799
/// gave every host body a `chelis_rng_state *__chelis_rng` parameter and
/// renamed the extracted kernels to `..._with_rng`, and a literal match on the
/// old spelling then failed with "the IO body is emitted as host code". That
/// read as the emission having MOVED off the host lane, which would be a real
/// regression in what these rows assert. It had not moved: the body is still
/// host code, the print is still inline in it, and the ordering property these
/// rows exist for was intact throughout. A precondition that cannot tell a
/// changed signature from a changed lane is worse than no precondition, so
/// this one keys on the structure it actually needs.
fn host_body_definition<'a>(emitted: &'a str, name: &str) -> &'a str {
    let needle = format!("{name}(");
    let mut at = 0;
    while let Some(found) = emitted[at..].find(&needle) {
        let start = at + found;
        let rest = &emitted[start..];
        // No parameter of a host body carries a nested parenthesis, so the
        // first `)` closes the list.
        if let Some(close) = rest.find(')')
            && rest[close + 1..].trim_start().starts_with('{')
        {
            return rest;
        }
        at = start + needle.len();
    }
    panic!(
        "no definition of `{name}` in the emitted C: the IO body is expected on the host lane, \
         and a forward declaration alone means it moved off it"
    );
}

fn assert_effect_order_in_emitted_c(emitted: &str, effect_first: bool) {
    // The kernel C extracts for `f(seed, x)` is the `run__tensor_N` whose
    // body carries the entry guard (`seed`'s own sub-expression is another
    // `run__tensor_M`, called before the print in both variants).
    let trap = "chelis_numeric_trap(\"numeric trap: domain in load at int64\")";
    let (kernel_name, kernel) = emitted
        .match_indices("static void run__tensor_")
        .map(|(at, _)| {
            let rest = &emitted[at..];
            let name_end = rest.find('(').expect("kernel signature");
            let name = &rest["static void ".len()..name_end];
            let body_end = rest[1..]
                .find("\nstatic ")
                .map_or(rest.len(), |end| end + 1);
            (name, &rest[..body_end])
        })
        .find(|(_, kernel)| kernel.contains(trap))
        .expect("one extracted kernel carries the entry guard");
    let guard_at = kernel.find(trap).expect("the guard");
    let alloc_at = kernel.find("chelis_alloc(").expect("the kernel allocates");
    assert!(
        guard_at < alloc_at,
        "the entry guard precedes the kernel's first allocation"
    );
    let body = host_body_definition(emitted, "run__chelis_owned_body");
    let print_at = body
        .find("chelis_string_from_cstr(\"effect\")")
        .expect("the print is emitted in the host body (chelis#1528)");
    let call_at = body
        .find(&format!("{kernel_name}("))
        .expect("the host body calls the guarded kernel");
    if effect_first {
        assert!(
            print_at < call_at,
            "the print precedes the call into the guarded kernel"
        );
    } else {
        assert!(
            call_at < print_at,
            "the call into the guarded kernel precedes the print"
        );
    }
}

/// guard_order.effect_before.c: preceding output survives the guard trap,
/// including fully buffered stdout on Linux (#1591).
#[test]
fn an_effect_before_the_guard_runs_when_the_guard_traps_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out, emitted) = c_run_result_with_source(
        &dir,
        "effect_before_c",
        &effect_order_source(MISMATCHED, true),
    );
    assert!(!ok, "the binary must fail: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `4`: claimed = 4, x axis 0 = 5"),
        "{out}"
    );
    assert_eq!(
        out.lines().filter(|line| *line == "effect").count(),
        1,
        "{out}"
    );
    assert_effect_order_in_emitted_c(&emitted, true);
}

/// guard_order.effect_after.c: a trapped guard prevents the later print.
#[test]
fn an_effect_after_the_guard_does_not_run_when_the_guard_traps_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out, emitted) = c_run_result_with_source(
        &dir,
        "effect_after_c",
        &effect_order_source(MISMATCHED, false),
    );
    assert!(!ok, "the binary must fail: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(!out.lines().any(|line| line == "effect"), "{out}");
    assert_effect_order_in_emitted_c(&emitted, false);
}

/// class.load_load.eval: an all-interface class of three witnesses on one
/// binder; the def reads all three, so each is a kernel input and each later
/// witness is guarded against the canonical one at entry. `f` is a kernel on
/// both lanes; a bare-variable body would be host code on both and would
/// carry no guard on either. The signature's first parameter `zz` is the
/// canonical witness on both lanes, as required by section 4.7. Helper
/// extraction must preserve that order rather than sorting by binding name.
///
/// EVIDENTIARY STATUS: regression test for the rendering. On the tree
/// without B2h's guard reordering eval reported the symbolic-binding
/// inference's own wording (`symbolic dimension `zdim` mismatch: canonical
/// p[0] = 2, but r[0] = 3`), never [04-NUM-9]'s line; on `main` the
/// interpreter reports chelis#1382's binder receipt.
#[test]
fn load_load_named_class_guards_every_non_canonical_member_on_eval() {
    const SOURCE: &str = "def f(zz: tensor[zdim, f32], p: tensor[zdim, f32], r: tensor[zdim, f32]) -> tensor[zdim, f32] = add(add(zz, p), r)\n";
    let dir = tempfile::tempdir().expect("tempdir");
    let r_disagrees = format!(
        "{SOURCE}out = f(to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
    );
    let (ok, out) = eval_result(&dir, "r_disagrees.ch", &r_disagrees);
    assert!(!ok, "the witness `r` disagrees: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `zdim`: zz axis 0 = 2, r axis 0 = 3"),
        "the later witness is compared against the canonical one, as the compiled kernel renders it: {out}"
    );
    let zz_disagrees = format!(
        "{SOURCE}out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32]))\n"
    );
    let (ok, out) = eval_result(&dir, "zz_disagrees.ch", &zz_disagrees);
    assert!(!ok, "the witness `zz` disagrees: {out}");
    assert!(
        out.contains("extent `zdim`: zz axis 0 = 3, p axis 0 = 2"),
        "every non-canonical member is its own guard: {out}"
    );
    let agree = format!(
        "{SOURCE}out = f(to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32]))\n"
    );
    let (ok, out) = eval_result(&dir, "agree_class.ch", &agree);
    assert!(ok, "agreeing witnesses execute: {out}");
    assert!(
        out.contains("out = tensor(shape=[2], data=[3.0, 6.0])"),
        "{out}"
    );
}

// ===========================================================================
// B2b-0: the two class-shape rows that need a caller, on both lanes.
//
// Both are value-binding rows for the reason the `load_load` row above is
// one: `out = f(...)` applies the exported kernel, so `f`'s entry guards run
// at the call on eval and in the linked binary on C.
//
// The two lanes then render the same line, and the reason is worth stating
// precisely because the short version of it is false. They do NOT call one
// function. Eval reads `derive_runtime_dim_classes`, the C prologue reads
// `derive_dim_witnesses`, and those are two sibling groupings in
// `axis_sources.rs` that differ in their guard filter over one shared
// primitive, `output_axis_sources`, which answers where an axis's extent comes
// from. That shared primitive is what C2.5's one-derivation property is about.
// Byte identity of the rendered line is therefore something these receipts
// MEASURE, by asserting the same literal string on each lane, and not
// something a single shared call already guarantees.
//
// The DIMENSION NAMES here are deliberately multi-letter. A single lowercase
// letter in a dimension position desugars to `d-var`, a polymorphic dimension
// variable (`chelis-surf/src/desugar.rs`'s implicit single-letter rule), which
// inference instantiates per call site and unifies against the literal
// argument extents, so `f(to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32,
// 2.0f32, 3.0f32]))` under `tensor[q, f32]` is `dimension mismatch: Lit(2) vs
// Lit(3)` at check and never reaches a guard. A multi-letter name desugars to
// `d-name`, a concrete symbolic axis, which is what survives into `DimInfo::
// Named` and forms a class. Measured at `1a585ba1b` across `n`, `m`, `k`, `q`
// (all rejected) and `qq`, `nn`, `n2`, `zdim`, `batch`, `seq` (all accepted).
// ===========================================================================

/// The fixture for `class.no_movement_consumer`: two witnesses of one claim
/// with NO movement primitive anywhere in the program and a result type that
/// carries no symbolic dimension.
///
/// That is what separates this row from `class.load_load` above, whose result
/// is `tensor[zdim, f32]`. `runtime_extents.md` C2.4 states the property the
/// row exists for: Load/Load, Load/op-output and op-output/op-output
/// equalities "exist even when no movement bound owns them". Here nothing
/// owns `zdim`. No `expand`, `insert`, `reshape`, `shrink`, `stride` or `pad`
/// appears, so no movement operation binds it, and the declared result is a
/// scalar, so the signature does not force it either. The claim is held by
/// the two parameter declarations alone, and section 4.7 still requires the
/// entry guard.
fn no_movement_consumer_source(second: &str) -> String {
    format!(
        "module Repro.NoMovementConsumer\n\
         def f(zz: tensor[zdim, f32], p: tensor[zdim, f32]) -> tensor[f32] = sum(mul(zz, p), 0)\n\
         out = f(to_tensor([1.0f32, 2.0f32]), to_tensor({second}))\n"
    )
}

/// Two witnesses at 2 and 3, so the claim `zdim` is false.
const NO_MOVEMENT_DISAGREES: &str = "[1.0f32, 2.0f32, 3.0f32]";

/// Two witnesses at 2, so it holds and the reduction runs.
const NO_MOVEMENT_AGREES: &str = "[3.0f32, 4.0f32]";

/// class.no_movement_consumer.eval.
///
/// EVIDENTIARY STATUS: disposition lock. B2a derives the class and B2h routes
/// the host lane through the kernel, so this shape already traps at
/// `1a585ba1b`; the row records the state those two changes left it in. The
/// assertions are not a tautology: the context line names both witnesses with
/// their observed extents, and the agreeing twin pins the scalar VALUE, so a
/// guard that trapped on every class or a context line replaced by a literal
/// would fail one of them.
#[test]
fn a_class_with_no_movement_bound_consumer_still_guards_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "no_move_bad.ch",
        &no_movement_consumer_source(NO_MOVEMENT_DISAGREES),
    );
    assert!(!ok, "the witnesses of `zdim` disagree at 2 and 3: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `zdim`: zz axis 0 = 2, p axis 0 = 3"),
        "section 4.7's context line names the claim, both witnesses and each \
         observed extent: {out}"
    );

    let (ok, out) = eval_result(
        &dir,
        "no_move_ok.ch",
        &no_movement_consumer_source(NO_MOVEMENT_AGREES),
    );
    assert!(ok, "agreeing witnesses must execute: {out}");
    assert!(
        out.contains("out = 11.0"),
        "and produce sum(zz * p), whose shape does not carry the guarded \
         extent at all: {out}"
    );
}

/// class.no_movement_consumer.c: the same program compiled and run.
///
/// EVIDENTIARY STATUS: disposition lock, as its eval twin. The row's own
/// content is the LANE AGREEMENT: the compiled binary must render the
/// identical context line, which C2.5 requires because both lanes read one
/// derivation.
#[test]
fn a_class_with_no_movement_bound_consumer_still_guards_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(
        &dir,
        "no_move_bad_c",
        &no_movement_consumer_source(NO_MOVEMENT_DISAGREES),
    );
    assert!(!ok, "the binary must fail: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `zdim`: zz axis 0 = 2, p axis 0 = 3"),
        "byte-identical to the eval twin's line: {out}"
    );

    let (ok, out) = c_run_result(
        &dir,
        "no_move_ok_c",
        &no_movement_consumer_source(NO_MOVEMENT_AGREES),
    );
    assert!(ok, "agreeing witnesses must execute: {out}");
    assert!(out.contains("out = 11.0"), "{out}");
}

/// The fixture for `class.shared_member_node`: one rank-2 parameter whose two
/// axes are members of two DIFFERENT claims.
///
/// C2.4: "two classes may share a node, each with its own guard, and
/// derivation yields one member per `(node, axis)`". Both `zz` and `p` are
/// member nodes of the `rows` class and of the `cols` class at once, so a
/// derivation keyed by node rather than by `(node, axis)` would collapse the
/// two into one and lose a guard. Driving each axis to disagree on its own
/// shows the two guards are separate and that each names its own claim.
fn shared_member_node_source(second: &str) -> String {
    format!(
        "module Repro.SharedMemberNode\n\
         def f(zz: tensor[rows, cols, f32], p: tensor[rows, cols, f32]) -> tensor[rows, cols, f32] = add(zz, p)\n\
         out = f(to_tensor([[1.0f32, 2.0f32]]), to_tensor({second}))\n"
    )
}

/// `rows` disagrees at 1 against 2, `cols` agrees at 2.
const SHARED_ROWS_DISAGREE: &str = "[[1.0f32, 2.0f32], [3.0f32, 4.0f32]]";

/// `cols` disagrees at 2 against 3, `rows` agrees at 1.
const SHARED_COLS_DISAGREE: &str = "[[1.0f32, 2.0f32, 5.0f32]]";

/// Both agree.
const SHARED_AGREES: &str = "[[3.0f32, 4.0f32]]";

/// class.shared_member_node.eval.
///
/// EVIDENTIARY STATUS: disposition lock, for the same reason as
/// `no_movement_consumer` above. What the row pins is that the two classes
/// stay separate: each mismatch reports ITS OWN claim and ITS OWN axis, so a
/// collapsed derivation reporting one guard for the node, or reporting `rows`
/// where `cols` disagreed, fails.
///
/// The axis-1 case is where the teeth are. A derivation keyed by node rather
/// than by `(node, axis)` would compare `zz` against `p` once, on whichever
/// axis it kept, so the `cols`-only disagreement would slip through and that
/// program would RUN. `assert!(!ok)` there is not satisfiable by any guard
/// that has collapsed the two classes into one.
#[test]
fn two_classes_sharing_one_node_keep_separate_guards_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "shared_rows.ch",
        &shared_member_node_source(SHARED_ROWS_DISAGREE),
    );
    assert!(!ok, "the `rows` witnesses disagree at 1 and 2: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `rows`: zz axis 0 = 1, p axis 0 = 2"),
        "the axis-0 class reports axis 0 of both members: {out}"
    );
    assert!(
        !out.contains("extent `cols`"),
        "and the axis-1 class, which agrees, contributes no guard failure: {out}"
    );

    let (ok, out) = eval_result(
        &dir,
        "shared_cols.ch",
        &shared_member_node_source(SHARED_COLS_DISAGREE),
    );
    assert!(!ok, "the `cols` witnesses disagree at 2 and 3: {out}");
    assert!(
        out.contains("extent `cols`: zz axis 1 = 2, p axis 1 = 3"),
        "the axis-1 class is its own guard, naming axis 1 of the same two \
         member nodes: {out}"
    );

    let (ok, out) = eval_result(
        &dir,
        "shared_ok.ch",
        &shared_member_node_source(SHARED_AGREES),
    );
    assert!(ok, "both classes hold: {out}");
    assert!(
        out.contains("out = tensor(shape=[1, 2], data=[4.0, 6.0])"),
        "{out}"
    );
}

/// class.shared_member_node.c: the same three programs compiled and run.
///
/// EVIDENTIARY STATUS: disposition lock, as its eval twin, and the lane
/// agreement is the row's content.
#[test]
fn two_classes_sharing_one_node_keep_separate_guards_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(
        &dir,
        "shared_rows_c",
        &shared_member_node_source(SHARED_ROWS_DISAGREE),
    );
    assert!(!ok, "the binary must fail: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `rows`: zz axis 0 = 1, p axis 0 = 2"),
        "byte-identical to the eval twin's line: {out}"
    );
    assert!(!out.contains("extent `cols`"), "{out}");

    let (ok, out) = c_run_result(
        &dir,
        "shared_cols_c",
        &shared_member_node_source(SHARED_COLS_DISAGREE),
    );
    assert!(!ok, "the binary must fail: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `cols`: zz axis 1 = 2, p axis 1 = 3"),
        "byte-identical to the eval twin's line: {out}"
    );

    let (ok, out) = c_run_result(
        &dir,
        "shared_ok_c",
        &shared_member_node_source(SHARED_AGREES),
    );
    assert!(ok, "both classes hold: {out}");
    assert!(
        out.contains("out = tensor(shape=[1, 2], data=[4.0, 6.0])"),
        "{out}"
    );
}

// ===========================================================================
// S2b: `expand` as the same-rank broadcast, and the unit-extent claim.
//
// These rows reuse the fixtures and helpers above, `domain_trap_line`
// included, so the [04-NUM-9] spelling this slice asserts cannot drift from
// the one Slice B's own rows assert.
// ===========================================================================

fn check(path: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations", path.to_str().unwrap()])
        .output()
        .expect("check")
}

/// Oracle row `expand.positional.replacement.c` (chelis#597).
///
/// The compiled kernel takes the C emitter's same-rank arm, which reads index
/// 0 at the broadcast axis, and produces the same values the evaluator does.
/// Before chelis#1277's S2b no surface program could reach that arm: lowering
/// rewrote every `expand` into the rank-increasing form and overwrote the
/// checker's stamped type, so the arm compiled into every binary and executed
/// for nothing a user could write.
///
/// This is a regression row, and the base is the whole of that history: on
/// `main` the same source lowers to a rank-3 node and the C output has three
/// dimensions.
#[test]
fn issue_597_positional_same_rank_replacement_executes_on_c() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "broadcast.ch", SAME_RANK_BROADCAST);
    let out_dir = dir.path().join("broadcast-out");

    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "a same-rank broadcast must build: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let linked = common::link_generated(&out_dir, "broadcast.c", "broadcast");
    assert!(linked.success(), "the generated broadcast must link");
    let compiled = std::process::Command::new(out_dir.join("broadcast"))
        .output()
        .expect("compiled broadcast");
    assert!(
        compiled.status.success(),
        "the compiled broadcast failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );

    let evaluated = eval(&path);
    assert!(
        evaluated.status.success(),
        "eval of the broadcast failed: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let compiled_stdout = String::from_utf8_lossy(&compiled.stdout);
    assert_eq!(
        compiled_stdout,
        String::from_utf8_lossy(&evaluated.stdout),
        "compiled C and eval must agree byte for byte"
    );
    assert!(
        compiled_stdout.contains("shape=[2, 3]"),
        "the rank is unchanged and the unit axis carries the broadcast width: \
         {compiled_stdout}"
    );
    assert!(
        compiled_stdout.contains("data=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]"),
        "each row repeats across the broadcast axis: {compiled_stdout}"
    );
}

/// Oracle row `expand.positional.replacement.eval`.
///
/// The eval half of the row above, kept separate because the two lanes are
/// separate corpus rows and because this one names the shape the insertion
/// form would have produced, so a silent return to it fails here rather than
/// only in the byte comparison.
#[test]
fn a_positional_expand_replaces_a_unit_axis_instead_of_inserting_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "broadcast.ch", SAME_RANK_BROADCAST);
    let evaluated = eval(&path);
    let stdout = String::from_utf8_lossy(&evaluated.stdout).to_string();
    assert!(
        evaluated.status.success(),
        "eval of the broadcast failed: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert!(
        stdout.contains("shape=[2, 3]"),
        "the broadcast sets the unit axis and leaves the rank alone: {stdout}"
    );
    assert!(
        !stdout.contains("shape=[2, 3, 1]") && !stdout.contains("shape=[2, 1, 3]"),
        "no rank-3 shape may appear: the insertion form is `insert`'s and this \
         program does not use it: {stdout}"
    );
    let values = common::parse_tensor_data(&stdout, "out");
    assert_eq!(values, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], "{stdout}");
}

/// Oracle row `expand.positional.replacement.non_unit_source_static`.
///
/// `spec/05-risc-primitives.md` §2.4.1: "A literal operand extent at `axis`
/// other than 1 is a type error." The checker did not enforce the unit extent
/// at all before S2b; the IR verifier did, so the checker was the lenient
/// side and a program refuted by the language's own rule reached lowering.
#[test]
fn a_static_non_unit_source_under_a_same_rank_claim_is_a_type_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "static_non_unit.ch", STATIC_NON_UNIT_SOURCE);
    let checked = check(&path);
    let stdout = String::from_utf8_lossy(&checked.stdout).to_string();
    assert!(
        !checked.status.success(),
        "a literal non-unit operand extent must be refused at check: {stdout}"
    );
    assert!(
        stdout.contains("DimensionMismatch"),
        "the refusal is a dimension error: {stdout}"
    );
    assert!(
        stdout.contains("extent at axis 1 to be 1") && stdout.contains("got 4"),
        "the diagnostic names the axis and the extent observed there, not only \
         that something disagreed: {stdout}"
    );

    // The control that fixes how narrow the rule is: the same shapes with the
    // operation that ADDS an axis are accepted, so the rejection is about the
    // claim rather than about the extents.
    let ok = fixture(
        &dir,
        "insert_non_unit.ch",
        "module Repro.InsertNonUnit\n\
         def fine(x: tensor[2, 4, f32]) -> tensor[3, 2, 4, f32] = insert(&x, 0, 3i64)\n",
    );
    assert!(
        check(&ok).status.success(),
        "`insert` over the same operand is unaffected: {}",
        String::from_utf8_lossy(&check(&ok).stdout)
    );
}

/// Oracle row `expand.positional.replacement_zero.eval`.
///
/// C1.5 of `spec/design/runtime_extents.md`: zero is legal, a static negative
/// is a type error. A zero broadcast declares an empty axis rather than
/// refusing or silently producing one element.
#[test]
fn a_zero_positional_replacement_declares_an_empty_axis_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "zero.ch", ZERO_BROADCAST);
    let evaluated = eval(&path);
    let stdout = String::from_utf8_lossy(&evaluated.stdout).to_string();
    assert!(
        evaluated.status.success(),
        "a zero broadcast extent is legal: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert!(
        stdout.contains("shape=[2, 0]"),
        "the broadcast axis is empty and the rank is unchanged: {stdout}"
    );
    assert!(
        stdout.contains("data=[]"),
        "an empty axis carries no elements: {stdout}"
    );
}

/// Oracle row `expand.positional.replacement_zero.c`.
#[test]
fn a_zero_positional_replacement_declares_an_empty_axis_on_c() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "zero.ch", ZERO_BROADCAST);
    let out_dir = dir.path().join("zero-out");

    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "a zero broadcast extent must build: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let linked = common::link_generated(&out_dir, "zero.c", "zero");
    assert!(linked.success(), "the generated zero program must link");
    let compiled = std::process::Command::new(out_dir.join("zero"))
        .output()
        .expect("compiled zero program");
    assert!(
        compiled.status.success(),
        "the compiled zero program failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let compiled_stdout = String::from_utf8_lossy(&compiled.stdout);
    assert_eq!(
        compiled_stdout,
        String::from_utf8_lossy(&eval(&path).stdout),
        "compiled C and eval must agree byte for byte on an empty axis"
    );
    assert!(
        compiled_stdout.contains("shape=[2, 0]"),
        "the compiled kernel declares the empty axis too: {compiled_stdout}"
    );
}

/// Oracle row `expand.positional.replacement.non_unit_source_traps.eval`.
///
/// The claim's runtime extent guard on the lane a `chelis eval` of a user
/// program actually reaches. That lane is the HOST interpreter, not the DAG
/// evaluator: `eval_compiled` finds zero `Lane::Tensor` roots for a program of
/// this shape and every root falls to the host, which is why the host
/// interpreter checks the claim itself rather than relying on the derived
/// equality classes the compiled lanes read.
///
/// The trap renders as `spec/04-type-system.md` [04-NUM-9] requires, at
/// `int64` because the guarded result is an extent under [05-DIM-1] and not a
/// tensor element.
///
/// `<op>` is `load`, not `expand`, and which lane answers moved under B2h
/// (#1531). Before it, `chelis eval` reached only the host interpreter for a
/// program of this shape, so the trap came from `tensor_expand_host`'s own
/// check and named `expand`. B2h applies a host-lane def through the kernel C
/// emits for it, so the DAG evaluator's ENTRY guard now answers first, and
/// section 4.7 fixes its slot: "for a guard whose operands are all interface
/// values, the `load` primitive of the later witness in signature order". The
/// operand here is an input tensor's axis, so `load` is the correct rendering
/// and the previous one was correct for the lane that used to answer.
///
/// The host interpreter's check remains and is unchanged; it is simply no
/// longer the first guard this program meets.
#[test]
fn a_runtime_non_unit_source_under_a_same_rank_claim_traps_at_entry_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");

    // The control first: the guard must not fire on the symbolic SPELLING.
    let ok = fixture(&dir, "runtime_unit.ch", RUNTIME_UNIT_SOURCE);
    let accepted = eval(&ok);
    let ok_stdout = String::from_utf8_lossy(&accepted.stdout).to_string();
    assert!(
        accepted.status.success(),
        "a symbolic operand extent that satisfies the claim executes: {}",
        String::from_utf8_lossy(&accepted.stderr)
    );
    assert!(
        ok_stdout.contains("shape=[3]") && ok_stdout.contains("data=[5.0, 5.0, 5.0]"),
        "the satisfied claim broadcasts exactly: {ok_stdout}"
    );

    let path = fixture(&dir, "runtime_non_unit.ch", RUNTIME_NON_UNIT_SOURCE);
    let evaluated = eval(&path);
    let stderr = String::from_utf8_lossy(&evaluated.stderr).to_string();
    assert!(
        !evaluated.status.success(),
        "a runtime operand extent of 2 refutes the claim and must trap: {}",
        String::from_utf8_lossy(&evaluated.stdout)
    );
    assert!(
        stderr.contains(&domain_trap_line("load")),
        "the trap line is the [04-NUM-9] rendering verbatim, at the extent's \
         own dtype, with the slot section 4.7 gives an all-interface guard: \
         {stderr}"
    );
    assert!(
        stderr.contains("claimed = 1") && stderr.contains("axis 0 = 2"),
        "the accompanying context names the axis and the value observed, which \
         §4.7 requires on separate lines from the trap: {stderr}"
    );
}

/// A LOCALLY placed unit-extent claim is guarded on C, at its operation.
///
/// `spec/04-type-system.md` section 4.7 splits placement by operand class: a
/// guard whose operands are all interface values runs at entry, and one that
/// "compares a locally computed value (checked integer arithmetic, a
/// user-function result, or an extent an operation computes) is evaluated
/// after its producers and takes the source position of the operation that
/// introduces the guarded extent". The `<op>` slot follows the same rule, so
/// this one renders `expand` where the entry rows render `load`.
///
/// EVIDENTIARY STATUS: regression test, and it is the reason this row exists.
/// The first cut of chelis#1277 S2b derived the Local case and then wired only
/// the Entry consumers, so the compiled kernel for this program emitted ZERO
/// `chelis_numeric_trap` and printed `[7.0, 7.0, 7.0]` at exit 0: element 0 of
/// a two-element axis, broadcast in silence. That is the wrong answer the flip
/// exists to remove, arriving on the lane that matters most.
#[test]
fn a_local_unit_extent_claim_traps_at_its_operation_on_c() {
    let dir = tempfile::tempdir().expect("tempdir");

    // The control first: the same shape with an operand the shrink leaves at
    // extent 1 executes, so the guard fires on the disagreement rather than on
    // the local placement.
    let ok_path = fixture(&dir, "local_unit.ch", LOCAL_UNIT_SOURCE);
    let ok_out = dir.path().join("local-unit-out");
    let build = build_c(&ok_path, &ok_out);
    assert!(
        build.status.success(),
        "a satisfied local claim must build: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(
        common::link_generated(&ok_out, "local_unit.c", "local_unit").success(),
        "the satisfied program must link"
    );
    let ran = std::process::Command::new(ok_out.join("local_unit"))
        .output()
        .expect("compiled satisfied program");
    let ok_stdout = String::from_utf8_lossy(&ran.stdout);
    assert!(
        ran.status.success() && ok_stdout.contains("shape=[3]"),
        "a satisfied local claim broadcasts: {ok_stdout}{}",
        String::from_utf8_lossy(&ran.stderr)
    );

    let path = fixture(&dir, "local_refuted.ch", LOCAL_NON_UNIT_SOURCE);
    let out_dir = dir.path().join("local-refuted-out");
    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "a refuted claim is a RUNTIME failure, so the build still succeeds: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted =
        std::fs::read_to_string(out_dir.join("local_refuted.c")).expect("C source is written");
    assert!(
        emitted.contains(&format!("{}\");", domain_trap_line("expand"))),
        "the emitted kernel carries the claim's guard, rendering [04-NUM-9] \
         with `expand` as the introducing operation:\n{emitted}"
    );
    assert!(
        common::link_generated(&out_dir, "local_refuted.c", "local_refuted").success(),
        "the refuted program must still link"
    );
    let ran = std::process::Command::new(out_dir.join("local_refuted"))
        .output()
        .expect("compiled refuted program");
    let stderr = String::from_utf8_lossy(&ran.stderr);
    let stdout = String::from_utf8_lossy(&ran.stdout);
    assert!(
        !ran.status.success(),
        "an operand extent of 2 under a unit claim must not produce a value: \
         {stdout}"
    );
    assert!(
        !stdout.contains("data=[7.0, 7.0, 7.0]"),
        "and must not broadcast element 0 of a two-element axis: {stdout}"
    );
    assert!(
        stderr.contains(&domain_trap_line("expand")),
        "the trap line names the operation that introduces the claim: {stderr}"
    );
    assert!(
        stderr.contains("claimed = 1") && stderr.contains("axis 0 = 2"),
        "with section 4.7's context on its own line: {stderr}"
    );
}

/// The same locally placed unit-extent claim on the EVAL lane.
///
/// S2b guarded this claim on eval from a check inside the DAG evaluator's
/// `Expand` arm, and shipped no row for it: its two rows are the C lane and the
/// HIP host lowering. B2r's unification deletes that arm, so the claim is now
/// carried by the one consumer that reads every local site's own read
/// instruction. A moved mechanism with no row is a mechanism that can be
/// deleted silently, which is why this row exists.
///
/// It also pins a lane agreement the two implementations did not have. S2b's
/// arm reported the `expand`'s node id and the C emitter reports the site's,
/// which is the OPERAND's, so the same claim on the same program named `node 5`
/// on eval and `node 4` on C. Reading the site rather than the operation makes
/// both lanes name the site, so the whole two-line diagnostic is now identical
/// text on both lanes.
///
/// EVIDENTIARY STATUS: regression test for the unification. Under the unified
/// consumer replaced by a no-op this fails; before the unification it passed
/// with a different node id, which is why the assertion is byte-exact rather
/// than a `contains` on the trap line alone.
#[test]
fn a_local_unit_extent_claim_traps_at_its_operation_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");

    // The control first, so the row proves a guard and not a broken lane: the
    // same shape over an operand the shrink leaves at extent 1 executes.
    let ok_path = fixture(&dir, "local_unit.ch", LOCAL_UNIT_SOURCE);
    let accepted = eval(&ok_path);
    let ok_stdout = String::from_utf8_lossy(&accepted.stdout).to_string();
    assert!(
        accepted.status.success(),
        "a satisfied local claim executes on eval: {}",
        String::from_utf8_lossy(&accepted.stderr)
    );
    assert!(
        ok_stdout.contains("shape=[3]") && ok_stdout.contains("data=[7.0, 7.0, 7.0]"),
        "the satisfied claim broadcasts exactly: {ok_stdout}"
    );

    let path = fixture(&dir, "local_refuted.ch", LOCAL_NON_UNIT_SOURCE);
    let evaluated = eval(&path);
    let stdout = String::from_utf8_lossy(&evaluated.stdout).to_string();
    let stderr = String::from_utf8_lossy(&evaluated.stderr).to_string();
    assert!(
        !evaluated.status.success(),
        "an operand extent of 2 under a unit claim must not produce a value: {stdout}"
    );
    assert!(
        !stdout.contains("data=[7.0, 7.0, 7.0]"),
        "and must not broadcast element 0 of a two-element axis: {stdout}"
    );
    assert!(
        stderr.contains(&domain_trap_line("expand")),
        "the trap line names the operation that introduces the claim: {stderr}"
    );
    // Byte-exact, and on the operand's node id: this is the site's key, the
    // same one the C emitter renders. This row's failure comes from the
    // `ExtentWitness` LocalExpand transport (chelis#1686), whose context
    // spelling is that mechanism's and is unchanged by B2b-0b's local-guard
    // rendering.
    assert!(
        stderr.contains("extent `1`: claimed = 1, node 4 axis 0 = 2"),
        "section 4.7's context names the SITE, so the two lanes agree text for \
         text: {stderr}"
    );
}

/// The HIP lane carries the locally placed claim too, through the shared host
/// lowering rather than through a HIP-specific guard.
///
/// This is the honest disposition for that lane, and it is not the one the
/// plan assumed. The HIP emitter contains no local guard emission of its own:
/// `local_dim_guard_sites` has no reader there. Nor does HIP DEVICE codegen
/// ever see this shape, because `reject_unsupported_hip_ops` refuses a
/// node-valued movement bound before codegen (chelis#616), and calling the HIP
/// emitter directly on such a graph panics on that backstop. What the user
/// gets from `build --target hip` is the C emitter's host lowering, which S2b's
/// change reaches, so the claim is guarded on this lane by construction.
///
/// The row exists because "by construction" is exactly the kind of claim that
/// stops being true silently. If the host sharing ever ends, this fails.
///
/// EVIDENTIARY STATUS: regression test. Before the Local arm had consumers the
/// emitted HIP host source carried no comparison against 1 at all.
#[test]
fn a_local_unit_extent_claim_is_guarded_on_the_hip_host_lowering() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "local_refuted_hip.ch", LOCAL_NON_UNIT_SOURCE);
    let out_dir = dir.path().join("local-refuted-hip-out");

    let build = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("hip build");
    assert!(
        build.status.success(),
        "the HIP build routes this program to the host lane and succeeds: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted = std::fs::read_to_string(out_dir.join("local_refuted_hip_hip.cpp"))
        .expect("HIP host source is written");
    assert!(
        emitted.contains(&format!("{}\");", domain_trap_line("expand"))),
        "the emitted host source carries the claim's guard:\n{emitted}"
    );
    assert!(
        emitted.contains("extent `1`: claimed = %lld"),
        "with section 4.7's context on its own line:\n{emitted}"
    );
}

/// Two `expand` nodes over one locally computed unit axis share a guard site,
/// and that is a coalesce rather than a collision.
///
/// EVIDENTIARY STATUS: regression test, for a defect this pull request created
/// and round 2 found. The duplicate-key check added after round 1 refused any
/// second site on a key, so this program, which checks at score 1.0 and
/// evaluates to `63.0`, could not be built at all. The two sites carry the same
/// claim, the same canonical and the same operation, so one emitted comparison
/// discharges both; only a DISAGREEING pair loses an obligation, and only that
/// is refused now.
///
/// Both halves are asserted because either alone would mislead. Without the
/// trapping twin, coalescing could have been implemented by dropping the guard
/// entirely and this row would still pass.
#[test]
fn two_expands_over_one_operand_axis_share_one_guard() {
    let dir = tempfile::tempdir().expect("tempdir");

    let path = fixture(&dir, "two_expands.ch", TWO_EXPANDS_OVER_ONE_OPERAND);
    let out_dir = dir.path().join("two-expands-out");
    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "two agreeing claims on one axis must not refuse codegen: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted =
        std::fs::read_to_string(out_dir.join("two_expands.c")).expect("C source is written");
    assert_eq!(
        emitted.matches(&domain_trap_line("expand")).count(),
        1,
        "the two sites coalesce to one guard rather than emitting it twice:\n{emitted}"
    );
    assert!(
        common::link_generated(&out_dir, "two_expands.c", "two_expands").success(),
        "the agreeing program must link"
    );
    let ran = std::process::Command::new(out_dir.join("two_expands"))
        .output()
        .expect("compiled agreeing program");
    let stdout = String::from_utf8_lossy(&ran.stdout);
    assert!(
        ran.status.success() && stdout.contains("63"),
        "and must produce the value eval produces: {stdout}{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    assert_eq!(
        stdout,
        String::from_utf8_lossy(&eval(&path).stdout),
        "compiled C and eval agree byte for byte"
    );

    // The coalesced guard is still a guard.
    let bad = fixture(&dir, "two_expands_refuted.ch", TWO_EXPANDS_REFUTED);
    let bad_dir = dir.path().join("two-expands-refuted-out");
    assert!(
        build_c(&bad, &bad_dir).status.success(),
        "a refuted claim is a runtime failure, so the build still succeeds"
    );
    assert!(
        common::link_generated(&bad_dir, "two_expands_refuted.c", "refuted").success(),
        "the refuted program must link"
    );
    let ran = std::process::Command::new(bad_dir.join("refuted"))
        .output()
        .expect("compiled refuted program");
    let stderr = String::from_utf8_lossy(&ran.stderr);
    assert!(
        !ran.status.success() && stderr.contains(&domain_trap_line("expand")),
        "the one coalesced guard still refuses a non-unit operand: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// chelis#1375: a node-valued reshape target under a named claim.
//
// `spec/04-type-system.md` section 4.7.3 makes an arithmetic reshape target a
// FRESH extent, which a surrounding signature may give a name "only by
// imposing an execution-time equality guard". Section 4.7.2 traps `Domain`
// when a claimed named extent is not statically proven equal. So a signature
// claiming the operand's own binder over a computed target owes a guard on
// every lane, and chelis#1375 recorded both lanes executing without one.
//
// These are CLI rows rather than driven rows, unlike the guard rows this file
// keeps out. The header's reason is exact and still holds: `def main() =
// f(...)` inlines `f` into the root, so the target folds to a literal, the
// class disappears, and both lanes print the wrong shape whatever the guards
// do (measured: one kernel, `chelis_alloc(2, {2, 2})` and a `memcpy`). A
// top-level BINDING does not inline. `out = f(...)` lowers `f` standalone
// with its declared `n` symbolic, so the target keeps its `RtDim::Node`
// carrier, the class forms with the `Load`'s axis, and the guard is reachable
// from the CLI on both lanes.
//
// What "the same guard on both lanes" means exactly, because the rows assert
// with `contains` and would not see the difference: the [04-NUM-9] line is
// byte-identical, and section 4.7's context line is the same text on both
// lanes, but on eval the CLI then frames the whole thing as an error, so what
// a user reads there carries an `error: ` prefix and on C it does not. The
// atom's own line takes no prefix on either lane, which is what [04-NUM-9]
// fixes.
// ---------------------------------------------------------------------------

/// chelis#1375's reproducer, with `claim` naming the declared first result
/// extent and `factor` the divisor of the computed target.
///
/// With `claim = "n"` and `factor = 2`, the signature claims the input's own
/// extent for an axis whose runtime extent is half of it. With `claim = "n"`
/// and `factor = 1` the same mechanism produces an extent that AGREES, which
/// is what separates a guard from a lane that traps on every computed target.
fn node_target_source(claim: &str, factor: u32) -> String {
    let second = if factor == 2 { "2i64" } else { "1i64" };
    let second_ty = if factor == 2 { "2" } else { "1" };
    format!(
        "def f(x: tensor[n, f32]) -> tensor[{claim}, {second_ty}, f32] = \
         reshape(x, [floor_div(shape(x, 0), {factor}i64), {second}])\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\n"
    )
}

/// reshape.named_claim.node_target.c
///
/// EVIDENTIARY STATUS: regression test. Watched failing on `f47ce5775` (the
/// commit before `reshape` left the kernel keep-list), where the binary
/// printed `out = tensor(shape=[2, 2], ...)` and exited 0.
#[test]
fn a_node_valued_reshape_target_under_a_named_claim_is_guarded_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "node_target_c", &node_target_source("n", 2));
    assert!(
        !ok,
        "the claimed extent disagrees, so the binary must fail: {out}"
    );
    assert!(
        out.contains(&domain_trap_line("reshape")),
        "the guard takes the position of the reshape that introduces the extent: {out}"
    );
    assert!(
        out.contains("extent `n`: claimed = 4"),
        "section 4.7's context line carries the claim and each observed value: {out}"
    );
    assert!(
        !out.contains("numel mismatch"),
        "the extent guard is observed before the chelis#616 numel abort: {out}"
    );
}

/// The discriminating twin: the same mechanism with a target that AGREES with
/// the claim executes. Without it, a lane that trapped on every node-valued
/// target would pass the row above.
///
/// EVIDENTIARY STATUS: disposition lock.
#[test]
fn a_node_valued_reshape_target_that_agrees_with_its_claim_executes_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "node_target_ok_c", &node_target_source("n", 1));
    assert!(
        ok,
        "the claimed extent agrees, so the binary must run: {out}"
    );
    assert!(
        out.contains("shape=[4, 1]"),
        "the agreeing program produces its declared shape: {out}"
    );
}

/// reshape.named_claim.node_target.eval
///
/// The eval lane reaches this guard because B2h routes a host-lane def through
/// the kernel the C lane emits for it, so once `reshape` is off the keep-list
/// the same DAG, the same class and the same site serve both lanes.
///
/// EVIDENTIARY STATUS: regression test. Watched failing on the tree with
/// `reshape` already off the keep-list and no local guard in the DAG
/// evaluator, where eval printed `out = tensor(shape=[2, 2], ...)` and exited
/// 0 while the compiled binary trapped.
#[test]
fn a_node_valued_reshape_target_under_a_named_claim_is_guarded_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "node_target_eval.ch", &node_target_source("n", 2));
    assert!(
        !ok,
        "the claimed extent disagrees, so eval must fail: {out}"
    );
    assert!(
        out.contains(&domain_trap_line("reshape")),
        "eval renders [04-NUM-9]'s line for the same guard the C lane emits: {out}"
    );
    assert!(
        out.contains("extent `n`: claimed = 4"),
        "section 4.7's context line carries the claim and each observed value: {out}"
    );
    assert!(
        !out.contains("reshape expects"),
        "the extent guard is observed before the evaluator's own numel check: {out}"
    );
}

/// The discriminating twin on eval, for the same reason as its C sibling.
///
/// EVIDENTIARY STATUS: disposition lock.
#[test]
fn a_node_valued_reshape_target_that_agrees_with_its_claim_executes_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "node_target_ok_eval.ch", &node_target_source("n", 1));
    assert!(ok, "the claimed extent agrees, so eval must run: {out}");
    assert!(
        out.contains("shape=[4, 1]"),
        "the agreeing program produces its declared shape: {out}"
    );
}

/// The correct spelling of chelis#1375's program: the computed extent gets a
/// binder of its own rather than restating the operand's. No claim is shared,
/// so no class forms, no guard exists, and both lanes produce the real shape.
///
/// This is the row that says the repair rejects a WRONG claim rather than a
/// computed target, which is the failure mode a guard placed on the carrier
/// instead of on the class would have.
///
/// EVIDENTIARY STATUS: disposition lock.
#[test]
fn a_fresh_binder_over_a_node_valued_reshape_target_executes_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = node_target_source("m", 2);
    let (ok, out) = eval_result(&dir, "fresh_binder_eval.ch", &source);
    assert!(ok, "a fresh binder claims nothing to disagree with: {out}");
    assert!(
        out.contains("shape=[2, 2]"),
        "eval produces the real shape: {out}"
    );
    if !gcc_available() {
        return;
    }
    let (c_ok, c_out) = c_run_result(&dir, "fresh_binder_c", &source);
    assert!(c_ok, "the binary must run: {c_out}");
    assert!(
        c_out.contains("shape=[2, 2]"),
        "the compiled binary produces the same real shape: {c_out}"
    );
}

// ===========================================================================
// B2b-1: a declared result claim survives the call boundary (chelis#1374,
// chelis#1376).
//
// Section 4.7.2 makes a declared result that names an extent not statically
// proven equal to the produced one an execution-time check that traps Domain.
// Section 4.7.3 adds that a signature may name a fresh extent only by imposing
// an execution-time equality guard. The three rows below are the three shapes
// that claim reaches the boundary in:
//
//   * the LITERAL control, `-> tensor[4, f32]` over a read of a second
//     tensor's axis. Its requirement is a constant, so it is the row that says
//     the cross-tensor READ is not what the repair adds;
//   * chelis#1374, the same read under a NAMED result claim `-> tensor[rows,
//     f32]` whose binder is declared by a THIRD tensor. Nothing statically
//     relates `rows` to `cols`, so §4.7.2's execution-time check is the only
//     thing that can reject the disagreement;
//   * chelis#1376, a FOREIGN claim over a SAME-tensor read: the set axis of
//     `insert(x, 1i32, shape(x, 0i32))` is declared `cols`, a binder `x` does
//     not carry. §4.7.3's fresh-extent rule is exactly this case.
//
// All three are VALUE BINDINGS (`out = f(...)`), for the reason this file's
// header gives: a `def main() = f(...)` root inlines `f` into a kernel that
// carries no entry guard, while `out = f(...)` applies `f` and its entry
// obligations run at the call on both lanes.
//
// The dimension names are deliberately multi-letter. A single-letter name
// desugars to `d-var`, a polymorphic dimension variable that inference
// instantiates against each literal argument extent, so the disagreement
// becomes a check-time `dimension mismatch` and never reaches a guard.
// `rows`/`cols` desugar to `d-name`, the concrete symbolic axis that survives
// into `DimInfo::Named` and is what a signature witness carries.
//
// Rendering: §4.7 makes this a typed operation-precondition guard under
// [04-NUM-9], so the complete line is `numeric trap: domain in load at int64`
// and the accompanying context names each disagreeing source, its axis and its
// observed value. The assertions below check the trap line byte-exactly and
// the context by content, never by an invented cross-lane format.
// ===========================================================================

/// The literal control's fixture: `-> tensor[4, f32]` over `shape(y, 0i32)`,
/// with a third parameter `x` present so the program is the same shape as
/// chelis#1374's and differs from it only in the result claim.
fn literal_claim_cross_tensor_source(y_extent: usize) -> String {
    let values = (1..=y_extent)
        .map(|v| format!("{v}.0f32"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "module Repro.LiteralCrossTensor\n\
         def f(b: tensor[f32], x: tensor[rows, f32], y: tensor[cols, f32]) -> tensor[4, f32] = insert(b, 0i32, shape(y, 0i32))\n\
         out = f(scalar_to_tensor(7.0f32), to_tensor([1.0f32, 2.0f32]), to_tensor([{values}]))\n"
    )
}

/// chelis#1374's fixture: the same cross-tensor read under a NAMED result
/// claim. `rows` is declared by `x`; the produced extent is `y`'s axis 0.
fn named_claim_cross_tensor_source(x_extent: usize, y_extent: usize) -> String {
    let list = |n: usize| {
        (1..=n)
            .map(|v| format!("{v}.0f32"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "module Repro.NamedCrossTensor\n\
         def f(b: tensor[f32], x: tensor[rows, f32], y: tensor[cols, f32]) -> tensor[rows, f32] = insert(b, 0i32, shape(y, 0i32))\n\
         out = f(scalar_to_tensor(7.0f32), to_tensor([{}]), to_tensor([{}]))\n",
        list(x_extent),
        list(y_extent)
    )
}

/// chelis#1376's fixture: a FOREIGN claim over a SAME-tensor read. The set
/// axis of `insert(x, 1i32, shape(x, 0i32))` is declared `cols`, which only
/// `y` carries, so the signature names a fresh extent for that axis.
fn foreign_claim_same_tensor_source(y_extent: usize) -> String {
    let values = (1..=y_extent)
        .map(|v| format!("{v}.0f32"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "module Repro.ForeignSameTensor\n\
         def f(x: tensor[rows, f32], y: tensor[cols, f32]) -> tensor[rows, cols, f32] = insert(x, 1i32, shape(x, 0i32))\n\
         out = f(to_tensor([1.0f32, 2.0f32]), to_tensor([{values}]))\n"
    )
}

/// expand.literal_claim.cross_tensor_read.eval
///
/// EVIDENTIARY STATUS of the mismatch assertions: regression test. Recorded
/// red at `e813415d0` with the measured output in the pull request.
/// EVIDENTIARY STATUS of the agreeing twin: disposition lock; it is the row
/// that says the guard fires on the disagreement and not on the cross-tensor
/// spelling.
#[test]
fn a_literal_claim_over_a_cross_tensor_read_traps_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "lit_cross.ch", &literal_claim_cross_tensor_source(3));
    assert!(
        !ok,
        "a declared tensor[4, f32] over a read of 3 must not execute: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `4`: claimed = 4, y axis 0 = 3"),
        "the context names the constant requirement and the read it refutes: {out}"
    );

    let (ok, out) = eval_result(
        &dir,
        "lit_cross_ok.ch",
        &literal_claim_cross_tensor_source(4),
    );
    assert!(ok, "the agreeing extent must execute: {out}");
    assert!(
        out.contains("out = tensor(shape=[4], data=[7.0, 7.0, 7.0, 7.0])"),
        "and produce the declared shape: {out}"
    );
}

/// expand.literal_claim.cross_tensor_read.c
///
/// EVIDENTIARY STATUS: as its eval twin. The two lanes assert the same literal
/// context string, so their agreement is measured here rather than assumed
/// from a shared call.
#[test]
fn a_literal_claim_over_a_cross_tensor_read_traps_on_c() {
    assert!(
        gcc_available(),
        "this row executes a linked program; no lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "lit_cross_c", &literal_claim_cross_tensor_source(3));
    assert!(
        !ok,
        "the linked binary must trap rather than print a shape: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `4`: claimed = 4, y axis 0 = 3"),
        "the C lane renders the same context as eval: {out}"
    );

    let (ok, out) = c_run_result(
        &dir,
        "lit_cross_c_ok",
        &literal_claim_cross_tensor_source(4),
    );
    assert!(ok, "the agreeing extent must execute: {out}");
    assert!(
        out.contains("shape=[4]"),
        "and produce the declared shape: {out}"
    );
}

/// expand.named_claim.cross_tensor_read.eval (chelis#1374)
///
/// EVIDENTIARY STATUS of the mismatch assertions: regression test. Recorded
/// red at `e813415d0`, where eval printed `out = tensor(shape=[3], data=[7.0,
/// 7.0, 7.0])` and exited 0 under a declared `tensor[rows, f32]` whose binder
/// was witnessed at extent 2.
/// EVIDENTIARY STATUS of the agreeing twin: disposition lock.
#[test]
fn a_cross_tensor_read_under_a_named_claim_is_guarded_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "named_cross.ch",
        &named_claim_cross_tensor_source(2, 3),
    );
    assert!(
        !ok,
        "`rows` is witnessed at 2 and the result is produced at 3: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `rows`: x axis 0 = 2, y axis 0 = 3"),
        "the context names the declaring witness and the produced extent: {out}"
    );

    let (ok, out) = eval_result(
        &dir,
        "named_cross_ok.ch",
        &named_claim_cross_tensor_source(3, 3),
    );
    assert!(ok, "an agreeing pair of witnesses must execute: {out}");
    assert!(
        out.contains("out = tensor(shape=[3], data=[7.0, 7.0, 7.0])"),
        "and produce the claimed shape: {out}"
    );
}

/// expand.named_claim.cross_tensor_read.c (chelis#1374)
///
/// EVIDENTIARY STATUS: as its eval twin; recorded red at `e813415d0` with the
/// linked binary printing `out = tensor(shape=[3], data=[7, 7, 7])`.
#[test]
fn issue_1374_cross_tensor_read_under_a_named_claim_is_guarded_on_c() {
    assert!(
        gcc_available(),
        "this row executes a linked program; no lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(
        &dir,
        "named_cross_c",
        &named_claim_cross_tensor_source(2, 3),
    );
    assert!(
        !ok,
        "the linked binary must trap rather than print a shape: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `rows`: x axis 0 = 2, y axis 0 = 3"),
        "the C lane renders the same context as eval: {out}"
    );

    let (ok, out) = c_run_result(
        &dir,
        "named_cross_c_ok",
        &named_claim_cross_tensor_source(3, 3),
    );
    assert!(ok, "an agreeing pair of witnesses must execute: {out}");
    assert!(
        out.contains("shape=[3]"),
        "and produce the claimed shape: {out}"
    );
}

/// expand.foreign_claim.same_tensor_set_axis.eval (chelis#1376)
///
/// EVIDENTIARY STATUS of the mismatch assertions: regression test. Recorded
/// red at `e813415d0`, where eval printed `out = tensor(shape=[2, 2], data=
/// [1.0, 1.0, 2.0, 2.0])` and exited 0 under a declared `tensor[rows, cols,
/// f32]` whose `cols` was witnessed at extent 3.
/// EVIDENTIARY STATUS of the agreeing twin: disposition lock.
#[test]
fn a_same_tensor_read_under_a_foreign_claim_is_guarded_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "foreign_same.ch",
        &foreign_claim_same_tensor_source(3),
    );
    assert!(
        !ok,
        "`cols` is witnessed at 3 and the set axis is produced at 2: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `cols`: y axis 0 = 3, x axis 0 = 2"),
        "the context names the declaring witness and the produced extent: {out}"
    );

    let (ok, out) = eval_result(
        &dir,
        "foreign_same_ok.ch",
        &foreign_claim_same_tensor_source(2),
    );
    assert!(ok, "an agreeing pair of witnesses must execute: {out}");
    assert!(
        out.contains("out = tensor(shape=[2, 2], data=[1.0, 1.0, 2.0, 2.0])"),
        "and produce the claimed shape: {out}"
    );
}

/// expand.foreign_claim.same_tensor_set_axis.c (chelis#1376)
///
/// EVIDENTIARY STATUS: as its eval twin; recorded red at `e813415d0` with the
/// linked binary printing `out = tensor(shape=[2, 2], data=[1, 1, 2, 2])`.
#[test]
fn issue_1376_same_tensor_read_under_a_foreign_claim_is_guarded_on_c() {
    assert!(
        gcc_available(),
        "this row executes a linked program; no lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "foreign_same_c", &foreign_claim_same_tensor_source(3));
    assert!(
        !ok,
        "the linked binary must trap rather than print a shape: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `cols`: y axis 0 = 3, x axis 0 = 2"),
        "the C lane renders the same context as eval: {out}"
    );

    let (ok, out) = c_run_result(
        &dir,
        "foreign_same_c_ok",
        &foreign_claim_same_tensor_source(2),
    );
    assert!(ok, "an agreeing pair of witnesses must execute: {out}");
    assert!(
        out.contains("shape=[2, 2]"),
        "and produce the claimed shape: {out}"
    );
}

// chelis#665 / chelis#1556: a kept output axis whose extent an operation
// computes.
//
// `insert(stride(x, 2i64), 0i32, shape(x, 0i32))`. The `Stride` output axis 0
// carries a fresh runtime extent that no `Load` declares, and the `insert`'s
// KEPT axis 1 inherits that extent under a different spelling (the lowerer's
// `_anon_dim_2_1`). `crates/chelis-ir/tests/runtime_extent_slice_b_sources.rs`
// already pins the derivation's answer for that axis: `InputAxis { input: 0,
// axis: Lit(0) }` into the stride, whose own axis is `OpComputed`. Declaring
// the kept name from that resolved source is what these rows measure.
// ===========================================================================

/// chelis#665's reproducer in current Surf. The issue's text predates S2a, so
/// the rank-RAISING form is `insert`, the axis carrier is `0i32` and the size
/// carrier is int64.
const EXPAND_OVER_STRIDE: &str = "module Repro.ExpandOverStride\n\
     sig f: tensor[n, f32] -> tensor[m, u, f32]\n\
     def f(x) = insert(stride(x, 2i64), 0i32, shape(x, 0i32))\n\
     out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))\n";

/// The same family reached through a rank-RAISING `reshape` rather than an
/// `insert`, which is the variant chelis#665's own comment records: the
/// reshape's axis 0 is the strided extent under a second spelling.
const RESHAPE_OVER_STRIDE: &str = "module Repro.ReshapeOverStride\n\
     sig f: tensor[n, f32] -> tensor[m, u, f32]\n\
     def f(x) = reshape(stride(x, 2i64), [shape(stride(x, 2i64), 0i32), 1i64])\n\
     out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))\n";

/// chelis#665's third spelling: a RUNTIME-bounded `shrink` under the same
/// kept-axis `insert`. A statically bounded `shrink` is not in the family
/// (its output axis is a literal), which is what the negative twin below
/// holds fixed.
const EXPAND_OVER_RUNTIME_SHRINK: &str = "module Repro.ExpandOverRuntimeShrink\n\
     sig f: tensor[n, f32] -> tensor[m, u, f32]\n\
     def f(x) = insert(shrink(x, [[0i64, sub(shape(x, 0i32), 2i64)]]), 0i32, shape(x, 0i32))\n\
     out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))\n";

/// chelis#665's C row (`expand.kept_axis.op_declared_source.c`).
///
/// The kept axis's name is declared from the extent its source produces, so
/// the emitted C compiles, links and runs, and its result equals the eval
/// lane's byte for byte.
///
/// EVIDENTIARY STATUS: regression test. Measured RED on `3dc3f54f6`, where
/// `chelis build --target c` exits 101 with `internal compiler error:
/// symbolic dim `_anon_dim_2_1` is referenced by a non-Load node (id 2, op
/// Expand { axis: 0, size: InputAxis { tensor: 1, axis: Lit(0) } }) but no
/// Load input declares it` from `crates/chelis-ir/src/dag.rs`.
#[test]
fn issue_665_expand_over_stride_builds_and_runs() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "expand_over_stride", EXPAND_OVER_STRIDE);
    assert!(ok, "the compiled kept-axis program must run: {out}");
    assert!(
        out.contains("shape=[6, 3]"),
        "the kept axis is the strided extent 3 under the inserted 6: {out}"
    );
    assert!(
        out.contains(
            "data=[1.0, 3.0, 5.0, 1.0, 3.0, 5.0, 1.0, 3.0, 5.0, 1.0, 3.0, 5.0, \
             1.0, 3.0, 5.0, 1.0, 3.0, 5.0]"
        ),
        "every inserted row is the strided `1, 3, 5`: {out}"
    );
    let evaluated = eval(&fixture(
        &dir,
        "expand_over_stride_eval.ch",
        EXPAND_OVER_STRIDE,
    ));
    assert!(
        evaluated.status.success(),
        "eval must agree: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert_eq!(
        out,
        String::from_utf8_lossy(&evaluated.stdout),
        "the compiled binary and eval must agree byte for byte"
    );
}

/// The same kept-axis source reached through a rank-raising `reshape`.
///
/// EVIDENTIARY STATUS: regression test. Measured RED on `3dc3f54f6` with the
/// same ICE on `_anon_dim_3_0` at the `Reshape` node.
#[test]
fn a_kept_axis_over_a_rank_raising_reshape_of_a_strided_input_builds_and_runs() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "reshape_over_stride", RESHAPE_OVER_STRIDE);
    assert!(ok, "the compiled reshape variant must run: {out}");
    assert!(
        out.contains("shape=[3, 1]") && out.contains("data=[1.0, 3.0, 5.0]"),
        "the reshape target's axis 0 is the strided extent: {out}"
    );
}

/// The same kept-axis source over a RUNTIME-bounded `shrink`.
///
/// EVIDENTIARY STATUS: regression test. Measured RED on `3dc3f54f6` with the
/// same ICE on `_anon_dim_5_1` at the `Expand` node.
#[test]
fn a_kept_axis_over_a_runtime_bounded_shrink_builds_and_runs() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "expand_over_shrink", EXPAND_OVER_RUNTIME_SHRINK);
    assert!(ok, "the compiled runtime-shrink variant must run: {out}");
    assert!(
        out.contains("shape=[6, 4]"),
        "the kept axis is the shrunk extent 4 under the inserted 6: {out}"
    );
}

/// chelis#665's eval row (`expand.kept_axis.op_declared_source.eval`).
///
/// EVIDENTIARY STATUS: disposition lock, NOT a regression test. Measured
/// GREEN on `3dc3f54f6`: `chelis eval --file` already prints
/// `shape=[6, 3]` with `data=[1.0, 3.0, 5.0, ...]` for this program. The row
/// exists because the C lane's declaration change must not move the eval
/// lane, and because the byte-for-byte parity assertion in the C row above
/// has no meaning unless this side is pinned independently.
#[test]
fn an_op_declared_axis_on_an_expand_input_flows_through_the_kept_output_axis_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "kept_axis_eval.ch", EXPAND_OVER_STRIDE);
    assert!(ok, "the kept-axis program must evaluate: {out}");
    assert!(
        out.contains("shape=[6, 3]"),
        "the kept axis is the strided extent 3 under the inserted 6: {out}"
    );
    assert!(
        out.contains(
            "data=[1.0, 3.0, 5.0, 1.0, 3.0, 5.0, 1.0, 3.0, 5.0, 1.0, 3.0, 5.0, \
             1.0, 3.0, 5.0, 1.0, 3.0, 5.0]"
        ),
        "every inserted row is the strided `1, 3, 5`: {out}"
    );
}

/// The negative twin of the three kept-axis rows: a STATICALLY bounded
/// `shrink` under the same `insert` has a literal kept extent, so it is not
/// in the family at all and must keep executing on both lanes with the
/// literal shape. A repair that declared every kept axis from a computed
/// source would show up here as a changed extent.
///
/// EVIDENTIARY STATUS: disposition lock. Measured GREEN on `3dc3f54f6`.
#[test]
fn a_kept_axis_over_a_statically_bounded_shrink_keeps_its_literal_extent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "module Repro.ExpandOverStaticShrink\n\
         sig f: tensor[n, f32] -> tensor[m, u, f32]\n\
         def f(x) = insert(shrink(x, [[0i64, 4i64]]), 0i32, shape(x, 0i32))\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))\n";
    let (ok, out) = eval_result(&dir, "static_shrink_eval.ch", source);
    assert!(ok, "a static shrink bound is not a runtime extent: {out}");
    assert!(
        out.contains("shape=[6, 4]"),
        "the kept axis keeps the literal 4: {out}"
    );
    if !gcc_available() {
        return;
    }
    let (c_ok, c_out) = c_run_result(&dir, "static_shrink_c", source);
    assert!(
        c_ok,
        "the static-bound program must still build and run: {c_out}"
    );
    assert!(
        c_out.contains("shape=[6, 4]"),
        "the compiled binary keeps the same literal extent: {c_out}"
    );
}

// ===========================================================================
// chelis#1556: `uniform_like` over a symbolic parameter inlined into a
// nullary kernel.
//
// These rows are DISPOSITION LOCKS, not regression tests, and the difference
// is worth recording rather than glossing. The issue's program cannot be
// written verbatim any more: it spells the rank-raising form `expand(
// scalar_to_tensor(..), 0, 3i64)`, and since S2a a rank-0 operand makes that
// a check-time type error naming `insert`. Measured on `3dc3f54f6` in its
// faithful modern spelling, and in three further nullary shapes, the ICE does
// not reproduce: every one builds, links and runs, and no synthesized
// `d<N>`-style name reaches the emitted C. No bisect was run for which
// earlier change closed it, and the issue's own fix hypothesis - substitute
// the argument's static extent for the inlined parameter's dim at inlining -
// was never needed.
//
// What the rows are for is the other direction. This change moves every
// declaration onto the axis source, and a nullary kernel with no `Load` at
// all is the shape with the least to recover a name from, so it is exactly
// where a declaration regression would surface first.
// ===========================================================================

/// The issue's program in current Surf.
const UNIFORM_OVER_INLINED_PARAMETER: &str = "module Repro.UniformInline\n\
     def noise(x: tensor[n, f32]) -> tensor[n, f32] = add(x, uniform_like(x, 0.0f32, 1.0f32))\n\
     def main() = with seed(42i64) { add(noise(insert(scalar_to_tensor(1.0f32), 0i32, 3i64)), \
     noise(insert(scalar_to_tensor(2.0f32), 0i32, 3i64))) }\n";

/// The same inlined parameter over an operand whose extent an OPERATION
/// computes rather than a literal, still in a nullary kernel.
const UNIFORM_OVER_INLINED_STRIDE: &str = "module Repro.UniformInlineStride\n\
     def noise(x: tensor[n, f32]) -> tensor[n, f32] = add(x, uniform_like(x, 0.0f32, 1.0f32))\n\
     def main() = with seed(42i64) { noise(stride(insert(scalar_to_tensor(1.0f32), 0i32, 6i64), \
     2i64)) }\n";

/// The same shape reached through an exported def and a value binding, so the
/// kernel has a `Load` while the inlined parameter's dim still does not.
const UNIFORM_OVER_EXPORTED_STRIDE: &str = "module Repro.UniformExportedStride\n\
     sig g: tensor[n, f32] -> tensor[m, f32]\n\
     def g(x) = stride(x, 2i64)\n\
     def noise(x: tensor[k, f32]) -> tensor[k, f32] = add(x, uniform_like(x, 0.0f32, 1.0f32))\n\
     out = with seed(42i64) { noise(g(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))) }\n";

/// Build, link, run and evaluate one nullary-kernel program, and assert that
/// both lanes print `expected` and that no synthesized `d<N>` name survives
/// into the emitted C.
fn assert_nullary_kernel_lanes_agree(stem: &str, source: &str, expected: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, &format!("{stem}_eval.ch"), source);
    assert!(ok, "the nullary kernel must evaluate: {out}");
    assert!(out.contains(expected), "eval prints {expected}: {out}");
    if !gcc_available() {
        return;
    }
    let (c_ok, c_out, emitted) = c_run_result_with_source(&dir, stem, source);
    assert!(c_ok, "the compiled nullary kernel must run: {c_out}");
    assert_eq!(
        c_out, out,
        "the compiled binary and eval must agree byte for byte"
    );
    let synthesized = emitted
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .find(|token| {
            token.len() > 1
                && token.starts_with('d')
                && token[1..].chars().all(|c| c.is_ascii_digit())
        });
    assert_eq!(
        synthesized, None,
        "no synthesized `d<N>` dimension name reaches the emitted C"
    );
}

/// chelis#1556's own program, in its current spelling.
///
/// EVIDENTIARY STATUS: disposition lock. Measured GREEN on `3dc3f54f6`.
#[test]
fn issue_1556_uniform_over_an_inlined_parameter_builds_and_runs() {
    assert_nullary_kernel_lanes_agree(
        "uniform_inline",
        UNIFORM_OVER_INLINED_PARAMETER,
        "data=[4.3952804, 4.3952804, 4.5689626]",
    );
}

/// EVIDENTIARY STATUS: disposition lock. Measured GREEN on `3dc3f54f6`.
#[test]
fn a_nullary_kernel_whose_inlined_parameter_is_sized_by_a_stride_builds_and_runs() {
    assert_nullary_kernel_lanes_agree(
        "uniform_inline_stride",
        UNIFORM_OVER_INLINED_STRIDE,
        "data=[1.6537157, 1.7415649, 1.849176]",
    );
}

/// EVIDENTIARY STATUS: disposition lock. Measured GREEN on `3dc3f54f6`.
#[test]
fn an_exported_stride_under_an_inlined_uniform_parameter_builds_and_runs() {
    assert_nullary_kernel_lanes_agree(
        "uniform_exported_stride",
        UNIFORM_OVER_EXPORTED_STRIDE,
        "data=[1.6537157, 3.7415648, 5.849176]",
    );
}

// ---------------------------------------------------------------------------
// chelis#1397 and the op-computed local-guard admission (B2b-0b).
//
// `spec/04-type-system.md` section 4.7 places a guard that compares "an extent
// an operation computes" at the source position of that operation, "after its
// producers and before the first allocation or element access whose shape
// depends on the guarded extent". Its movement paragraph makes `shrink` the
// owner whose symbolic axes "always mint fresh extents", so a declared result
// over a runtime-bound `shrink` claims a value no input carries and no carrier
// states: it has to be computed from the operation's own bounds and compared
// before the operation allocates.
//
// The fixtures are value bindings for the reason the file header gives: only a
// binding applies the exported kernel. The `shrink.{export,binding,root}` rows
// in `runtime_extent_claim_preparation.rs` cover all three call forms.
// ---------------------------------------------------------------------------

/// A rank-1 f32 tensor literal holding `1.0 .. len`.
fn vector_literal(len: usize) -> String {
    let values = (1..=len)
        .map(|v| format!("{v}.0f32"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("to_tensor([{values}])")
}

/// A `side` x `side` f32 tensor literal holding `1.0 .. side*side`.
fn square_literal(side: usize) -> String {
    let rows = (0..side)
        .map(|r| {
            let row = (0..side)
                .map(|c| format!("{}.0f32", r * side + c + 1))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{row}]")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("to_tensor([{rows}])")
}

/// A `side` x `side` f32 tensor literal of ones.
fn ones_literal(side: usize) -> String {
    let rows = (0..side)
        .map(|_| {
            let row = vec!["1.0f32"; side].join(", ");
            format!("[{row}]")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("to_tensor([{rows}])")
}

/// `f(x: tensor[rows]) -> tensor[2]` over a runtime-bound shrink: chelis#1397's
/// declaration half. The shrink produces `len - 1` against a declared 2.
fn shape_derived_declared_result_source(len: usize) -> String {
    format!(
        "def f(x: tensor[rows, f32]) -> tensor[2, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
         out = f({})\n",
        vector_literal(len)
    )
}

/// expand.shape_derived.declared_result_survives.eval.
///
/// EVIDENTIARY STATUS: regression test for both mismatch assertions. Measured
/// on the base `fd6fc6f5d`, the mismatching program printed
/// `out = tensor(shape=[3], data=[2.0, 3.0, 4.0])` and exited zero under a
/// declared `tensor[2, f32]`: the actualized result carried the synthesized
/// `_rt_shrink_dim_N_0` instead of the declared literal, so no class and no
/// guard existed. The satisfied row is the non-vacuity control and passed on
/// the base.
#[test]
fn a_shape_derived_bound_keeps_its_declared_result_dimension_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "shape_derived_ok.ch",
        &shape_derived_declared_result_source(3),
    );
    assert!(ok, "a shrink that produces the declared 2 executes: {out}");
    assert!(
        out.contains("shape=[2]") && out.contains("data=[2.0, 3.0]"),
        "and produces exactly the declared extent: {out}"
    );

    let (ok, out) = eval_result(
        &dir,
        "shape_derived_bad.ch",
        &shape_derived_declared_result_source(4),
    );
    assert!(
        !ok,
        "a shrink that produces 3 under a declared 2 must not return a value: {out}"
    );
    assert!(
        !out.contains("shape=[3]"),
        "and must not print the undeclared extent: {out}"
    );
    assert!(out.contains(&domain_trap_line("shrink")), "{out}");
    assert!(
        out.contains("extent `2`: claimed = 2, shrink axis 0 = 3"),
        "section 4.7's context names the claim, the operation, the axis and \
         the observed value: {out}"
    );
}

/// expand.shape_derived.declared_result_survives.c.
///
/// EVIDENTIARY STATUS: regression test for the mismatch assertions, with the
/// same base measurement as its eval twin (`shape=[3]`, exit zero). The
/// emitted-C ordering assertion is a regression assertion too: no comparison
/// against the declared 2 existed in the emitted source at all.
#[test]
fn a_shape_derived_bound_keeps_its_declared_result_dimension_on_c() {
    assert!(
        gcc_available(),
        "this row executes a linked program; no lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(
        &dir,
        "shape_derived_ok_c",
        &shape_derived_declared_result_source(3),
    );
    assert!(ok, "a shrink that produces the declared 2 executes: {out}");
    assert!(
        out.contains("shape=[2]") && out.contains("data=[2.0, 3.0]"),
        "and produces exactly the declared extent: {out}"
    );

    let (ok, out, emitted) = c_run_result_with_source(
        &dir,
        "shape_derived_bad_c",
        &shape_derived_declared_result_source(4),
    );
    assert!(!ok, "the linked binary must trap: {out}");
    assert!(out.contains(&domain_trap_line("shrink")), "{out}");
    assert!(
        out.contains("extent `2`: claimed = 2, shrink axis 0 = 3"),
        "the C lane renders the same context as eval: {out}"
    );
    assert!(
        !out.contains("movement target shape mismatch"),
        "and PREEMPTS the legacy movement failure rather than following it: {out}"
    );
    let guard_at = emitted
        .find("extent `2`: claimed = %lld, shrink axis 0 = %lld")
        .expect("the emitted guard");
    let check_at = emitted
        .find("chelis_movement_check_target")
        .expect("the emitted legacy target check");
    let alloc_at = emitted
        .find("chelis_alloc(1,")
        .expect("the emitted result allocation");
    assert!(
        guard_at < check_at && guard_at < alloc_at,
        "C1.3 places the guard between the computed extent and the allocation \
         that depends on it: {emitted}"
    );
}

/// A class holding two interface witnesses for `n` and one op-computed witness
/// for the same claim. Two parameters declare `n` so its witnesses survive as
/// an entry obligation (chelis#1374); the third witness is the shrink result,
/// which no input carries.
fn load_and_op_output_class_source(len: usize) -> String {
    format!(
        "def f(x: tensor[n, f32], p: tensor[n, f32], y: tensor[k, f32]) -> tensor[n, f32] = \
         shrink(y, [[1i64, shape(y, 0i32)]])\n\
         out = f(to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32]), {})\n",
        vector_literal(len)
    )
}

/// class.load_op_output.eval: an interface witness and an op-computed witness
/// share a class, and the op-computed member is the one guarded.
///
/// EVIDENTIARY STATUS: regression test for the mismatch assertions. On the base
/// the mismatching program printed `out = tensor(shape=[3], data=[2.0, 3.0,
/// 4.0])` and exited zero under a declared `tensor[n, f32]` with `n = 2`,
/// because the result axis carried a synthesized name rather than `n` and so
/// joined no class. The satisfied row passed on the base.
#[test]
fn load_and_op_output_members_share_one_guarded_class_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "load_opout_ok.ch",
        &load_and_op_output_class_source(3),
    );
    assert!(ok, "a shrink that produces `n` = 2 executes: {out}");
    assert!(
        out.contains("shape=[2]") && out.contains("data=[2.0, 3.0]"),
        "and produces the canonical member's extent: {out}"
    );

    let (ok, out) = eval_result(
        &dir,
        "load_opout_bad.ch",
        &load_and_op_output_class_source(4),
    );
    assert!(!ok, "a shrink that produces 3 under `n` = 2 traps: {out}");
    assert!(out.contains(&domain_trap_line("shrink")), "{out}");
    assert!(
        out.contains("extent `n`: claimed = 2, shrink axis 0 = 3"),
        "the interface witness supplies the claimed value and the op-computed \
         member supplies the observed one: {out}"
    );
}

/// Two op-computed axes of ONE shrink in one class, against an interface
/// member as the canonical witness. `x`'s two axes both declare `n`.
fn two_op_output_class_source(n: usize, side: usize) -> String {
    format!(
        "def f(x: tensor[n, n, f32], y: tensor[a, b, f32]) -> tensor[n, n, f32] = \
         shrink(y, [[1i64, shape(y, 0i32)], [1i64, shape(y, 1i32)]])\n\
         out = f({}, {})\n",
        ones_literal(n),
        square_literal(side)
    )
}

/// class.op_output_op_output.eval.
///
/// EVIDENTIARY STATUS: regression test for the mismatch assertions. On the base
/// the mismatching program printed `out = tensor(shape=[3, 3], ...)` and exited
/// zero under a declared `tensor[n, n, f32]` with `n = 2`. The
/// declaration-order assertion is a regression assertion against the reversed
/// member order the untied sort key produced: two axes of one node sorted
/// `[axis 1, axis 0]`, so axis 1 was canonical and reported first.
#[test]
fn two_op_output_members_guard_against_the_canonical_member_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "opout2_ok.ch", &two_op_output_class_source(2, 3));
    assert!(ok, "a shrink that produces 2x2 executes: {out}");
    assert!(
        out.contains("shape=[2, 2]") && out.contains("data=[5.0, 6.0, 8.0, 9.0]"),
        "and produces exactly the declared shape: {out}"
    );

    let (ok, out) = eval_result(&dir, "opout2_bad.ch", &two_op_output_class_source(2, 4));
    assert!(!ok, "a shrink that produces 3x3 under `n` = 2 traps: {out}");
    assert!(out.contains(&domain_trap_line("shrink")), "{out}");
    assert!(
        out.contains("extent `n`: claimed = 2, shrink axis 0 = 3"),
        "both op-computed members are guarded against the interface member, \
         and section 4.7's declaration order reports axis 0 first: {out}"
    );
    assert!(
        !out.contains("shrink axis 1 = 3\nextent"),
        "the first failing guard stops the evaluation: {out}"
    );
}

/// class.op_output_op_output.c: the same row on the compiled lane.
///
/// EVIDENTIARY STATUS: regression test, same base measurement as its eval twin
/// (`shape=[3, 3]`, exit zero). The emitted-guard count is a regression
/// assertion for the per-axis site key.
#[test]
fn two_op_output_members_guard_against_the_canonical_member_on_c() {
    assert!(
        gcc_available(),
        "this row executes a linked program; no lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "opout2_ok_c", &two_op_output_class_source(2, 3));
    assert!(ok, "a shrink that produces 2x2 executes: {out}");
    assert!(
        out.contains("shape=[2, 2]") && out.contains("data=[5.0, 6.0, 8.0, 9.0]"),
        "and produces exactly the declared shape: {out}"
    );

    let (ok, out, emitted) =
        c_run_result_with_source(&dir, "opout2_bad_c", &two_op_output_class_source(2, 4));
    assert!(!ok, "the linked binary must trap: {out}");
    assert!(out.contains(&domain_trap_line("shrink")), "{out}");
    assert!(
        out.contains("extent `n`: claimed = 2, shrink axis 0 = 3"),
        "the C lane renders the same context as eval: {out}"
    );
    assert_eq!(
        emitted.matches("shrink axis 0 = %lld").count(),
        1,
        "one guard for axis 0: {emitted}"
    );
    assert_eq!(
        emitted.matches("shrink axis 1 = %lld").count(),
        1,
        "and one for axis 1, so a member is neither duplicated nor dropped: {emitted}"
    );
}

/// `f(a, a)`: `splice_dag` maps both parameters of `f(n, n)` to one `NodeId`.
/// `start` is 0 for the satisfied form and 1 for the refuted one, so the same
/// spliced call shape produces both verdicts.
fn splice_f_of_n_n_source(start: usize, side: usize) -> String {
    format!(
        "def f(x: tensor[n, n, f32], y: tensor[n, n, f32]) -> tensor[n, n, f32] = \
         shrink(add(x, y), [[{start}i64, shape(x, 0i32)], [{start}i64, shape(x, 1i32)]])\n\
         a = {}\n\
         out = f(a, a)\n",
        square_literal(side)
    )
}

/// class.splice_f_of_n_n.eval: one member per `(node, axis)`, not one per
/// referencing parameter, and the members of one node stay in axis order.
///
/// EVIDENTIARY STATUS: regression test for the mismatch assertions. On the base
/// the refuted program printed `out = tensor(shape=[2, 2], ...)` and exited
/// zero under a declared `tensor[n, n, f32]` with `n = 3`. The single-trap-line
/// assertion is a regression assertion against the duplicate a per-parameter
/// member would produce.
#[test]
fn splicing_f_of_n_n_yields_one_member_per_output_axis_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "splice_ok.ch", &splice_f_of_n_n_source(0, 3));
    assert!(ok, "a full-axis span agrees with `n` on both axes: {out}");
    assert!(
        out.contains("out = tensor(shape=[3, 3]")
            && out.contains("data=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]"),
        "and the spliced parameter is added to itself exactly once: {out}"
    );

    let (ok, out) = eval_result(&dir, "splice_bad.ch", &splice_f_of_n_n_source(1, 3));
    assert!(!ok, "a 3x3 actual shrunk to 2x2 under `n` = 3 traps: {out}");
    assert!(
        out.contains("extent `n`: claimed = 3, shrink axis 0 = 2"),
        "axis 0 is reported first, and the spliced parameter contributes one \
         member rather than one per occurrence: {out}"
    );
    assert_eq!(
        out.lines()
            .filter(|line| *line == domain_trap_line("shrink"))
            .count(),
        1,
        "exactly one trap line: {out}"
    );
}

/// class.splice_f_of_n_n.c: the same row on the compiled lane.
///
/// EVIDENTIARY STATUS: regression test, same base measurement as its eval twin
/// (`shape=[2, 2]`, exit zero).
#[test]
fn splicing_f_of_n_n_yields_one_member_per_output_axis_on_c() {
    assert!(
        gcc_available(),
        "this row executes a linked program; no lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "splice_ok_c", &splice_f_of_n_n_source(0, 3));
    assert!(ok, "a full-axis span agrees with `n` on both axes: {out}");
    assert!(
        out.contains("out = tensor(shape=[3, 3]"),
        "and produces the declared shape: {out}"
    );

    let (ok, out, emitted) =
        c_run_result_with_source(&dir, "splice_bad_c", &splice_f_of_n_n_source(1, 3));
    assert!(!ok, "a 3x3 actual shrunk to 2x2 under `n` = 3 traps: {out}");
    assert!(
        out.contains("extent `n`: claimed = 3, shrink axis 0 = 2"),
        "the C lane renders the same context as eval: {out}"
    );
    assert_eq!(
        emitted.matches("shrink axis 0 = %lld").count(),
        1,
        "one emitted guard for the spliced axis, not one per parameter: {emitted}"
    );
    assert_eq!(
        emitted.matches("shrink axis 1 = %lld").count(),
        1,
        "and one for the second output axis: {emitted}"
    );
}

/// Two op-computed axes of one shrink under a binder declared by an UNREAD
/// parameter, so the class is guarded per axis in declaration order.
fn unbound_binder_class_source(rows: usize, cols: usize) -> String {
    let values = (0..rows)
        .map(|r| {
            let row = (0..cols)
                .map(|c| format!("{}.0f32", r * cols + c + 1))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{row}]")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "def f(x: tensor[n, f32], y: tensor[a, b, f32]) -> tensor[n, n, f32] = \
         shrink(y, [[1i64, shape(y, 0i32)], [1i64, shape(y, 1i32)]])\n\
         out = f(to_tensor([1.0f32, 2.0f32]), to_tensor([{values}]))\n"
    )
}

/// Two op-computed members of one class are guarded per axis against the
/// declaring parameter's extent, in declaration order, identically on both
/// lanes. The mismatch is placed on axis 1 so the row proves the SECOND axis
/// is guarded and not only the first.
///
/// This row was written for a different property. Before the claim recorded
/// its declaring parameter as a dependency, `x` was eliminated, `n` was bound
/// by nothing, and the class's first site declared the binder from its own
/// extent while the second guarded against it. The retention makes `n` the
/// caller's value HERE, so this spelling no longer reaches that rule.
///
/// It is still reachable, by a REPEATED FREE result name: `-> tensor[n, n]`
/// where no parameter declares `n`. The checker keeps such a name because it
/// appears twice (a name appearing once is published as `*`), no parameter
/// witness exists to retain, and the two op-computed axes form the class
/// alone. `a_repeated_free_result_name_declares_from_its_first_site` is that
/// row; an earlier version of this comment claimed the rule had become
/// unreachable, which was wrong and would have left it untested.
///
/// EVIDENTIARY STATUS: regression test for the mismatch assertions. On
/// `33cc78e84` both lanes printed `out = tensor(shape=[2, 3], ...)` and exited
/// zero under a declared `tensor[n, n, f32]`. The satisfied row is the
/// non-vacuity control.
#[test]
fn an_op_computed_class_with_an_unbound_binder_agrees_across_lanes() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let square = unbound_binder_class_source(3, 3);
    let (eval_ok, eval_out) = eval_result(&dir, "unbound_ok.ch", &square);
    let (c_ok, c_out) = c_run_result(&dir, "unbound_ok_c", &square);
    assert!(
        eval_ok,
        "a 2x2 span agrees with `n` on both axes: {eval_out}"
    );
    assert!(c_ok, "a 2x2 span agrees with `n` on both axes: {c_out}");
    for (lane, out) in [("eval", &eval_out), ("c", &c_out)] {
        assert!(
            out.contains("shape=[2, 2]"),
            "{lane}: the declared square shape: {out}"
        );
    }

    // Axis 0 agrees and axis 1 does not, so the reported failure is the
    // second axis: both members are guarded, in declaration order.
    let oblong = unbound_binder_class_source(3, 4);
    let (eval_ok, eval_out) = eval_result(&dir, "unbound_bad.ch", &oblong);
    let (c_ok, c_out) = c_run_result(&dir, "unbound_bad_c", &oblong);
    assert!(!eval_ok, "a 3x4 actual shrinks to 2x3: {eval_out}");
    assert!(!c_ok, "a 3x4 actual shrinks to 2x3: {c_out}");
    let context = "extent `n`: claimed = 2, shrink axis 1 = 3";
    for (lane, out) in [("eval", &eval_out), ("c", &c_out)] {
        assert!(
            out.contains(&domain_trap_line("shrink")),
            "{lane}: [04-NUM-9]'s line: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
    }
}

/// A result name declared by no parameter and repeated across two result axes.
fn repeated_free_result_name_source(rows: usize, cols: usize) -> String {
    let values = (0..rows)
        .map(|r| {
            let row = (0..cols)
                .map(|c| format!("{}.0f32", r * cols + c + 1))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{row}]")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "def f(x: tensor[r, c, f32]) -> tensor[n, n, f32] = \
         shrink(x, [[1i64, shape(x, 0i32)], [1i64, shape(x, 1i32)]])\n\
         out = f(to_tensor([{values}]))\n"
    )
}

/// A class whose binder NO input declares: the first site in derivation order
/// declares it from its own observed extent and every later site guards
/// against that value, identically on both lanes.
///
/// This is C2.4's canonical-member rule reaching the local sites, and it is
/// the rule the C emitter already applied through `runtime_dim_sites`'
/// declare-then-guard split. It is reachable from source only through a
/// repeated FREE result name: a name appearing once in the result is published
/// as `*` by the checker and forms no class, and a name a parameter declares is
/// retained and supplies the caller's value instead. Two axes of one result
/// naming the same undeclared binder is what is left, and it asserts exactly
/// what `tensor[n, n]` says: that the two axes agree with each other.
///
/// EVIDENTIARY STATUS: regression test for the mismatch assertions. On
/// `33cc78e84` both lanes printed `out = tensor(shape=[3, 2], ...)` and exited
/// zero under the declared `tensor[n, n, f32]`. The satisfied row is the
/// non-vacuity control. Round 1 of chelis#1397 found this spelling untested
/// after a sibling receipt was retargeted onto the retained-parameter case.
#[test]
fn a_repeated_free_result_name_declares_from_its_first_site() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let square = repeated_free_result_name_source(3, 3);
    let (eval_ok, eval_out) = eval_result(&dir, "free_name_ok.ch", &square);
    let (c_ok, c_out) = c_run_result(&dir, "free_name_ok_c", &square);
    assert!(eval_ok, "a 2x2 span agrees with itself: {eval_out}");
    assert!(c_ok, "a 2x2 span agrees with itself: {c_out}");
    for (lane, out) in [("eval", &eval_out), ("c", &c_out)] {
        assert!(
            out.contains("shape=[2, 2]") && out.contains("data=[5.0, 6.0, 8.0, 9.0]"),
            "{lane}: and produces the square the binder asserts: {out}"
        );
    }

    // Axis 0 declares `n` as 3 from its own extent; axis 1 produces 2 and is
    // guarded against it. No parameter supplies the 3.
    let oblong = repeated_free_result_name_source(4, 3);
    let (eval_ok, eval_out) = eval_result(&dir, "free_name_bad.ch", &oblong);
    let (c_ok, c_out) = c_run_result(&dir, "free_name_bad_c", &oblong);
    assert!(!eval_ok, "a 4x3 actual shrinks to 3x2: {eval_out}");
    assert!(!c_ok, "a 4x3 actual shrinks to 3x2: {c_out}");
    let context = "extent `n`: claimed = 3, shrink axis 1 = 2";
    for (lane, out) in [("eval", &eval_out), ("c", &c_out)] {
        assert!(
            out.contains(&domain_trap_line("shrink")),
            "{lane}: [04-NUM-9]'s line: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
    }
}

/// A helper whose parameter-bound named result is consumed inside ANOTHER
/// def's body. `len` is the outer actual's extent; the inner `n` is 2.
fn nested_named_result_source(len: usize, declared: &str) -> String {
    let values = (1..=len)
        .map(|v| format!("{v}.0f32"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "def f(w: tensor[n, f32], x: tensor[r, f32]) -> tensor[{declared}, f32] = \
         shrink(x, [[1i64, shape(x, 0i32)]])\n\
         def g(y: tensor[s, f32]) -> tensor[k, f32] = f(to_tensor([1.0f32, 2.0f32]), y)\n\
         out = g(to_tensor([{values}]))\n"
    )
}

/// The boundary of this slice's declaration half, stated rather than implied.
///
/// A NAMED result claim is enforced at the outermost activation: an exported
/// def, a top-level value binding, an inlined root. It is NOT enforced when the
/// declaring def is called from inside another def's body, because the
/// enclosing signature's own result name is written over the axis the inner
/// activation stamped, so the class keyed by the inner binder has one member
/// and C2.4 does not make that a class. Read off the emitted C: `f`'s own
/// kernel carries `int64_t n = chelis_tensor_shape(inputs[0], 0)` and the
/// comparison, while `g`'s kernel, with `f` inlined, carries
/// `int64_t k = chelis_movement_extent(...)` and none.
///
/// The LITERAL half survives the same nesting, which is why this row asserts
/// both: the two halves of the declared-result contract diverge exactly here,
/// and a lock that pinned only the unguarded side would not show that.
///
/// EVIDENTIARY STATUS: disposition lock on both lanes for the named half, not
/// a regression test: the same program exits zero on `8ae55787f` and this
/// slice neither introduces nor worsens it. Regression test for the literal
/// half, which this slice does deliver through the same nesting. Tracked by
/// chelis#1800, which must update this row when it closes.
#[test]
fn a_nested_named_result_claim_is_not_enforced_by_this_slice() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    // The named half: unguarded, and the outer result takes the extent the
    // operation computed rather than the one `n` declares.
    let named = nested_named_result_source(4, "n");
    let (eval_ok, eval_out) = eval_result(&dir, "nested_named.ch", &named);
    let (c_ok, c_out) = c_run_result(&dir, "nested_named_c", &named);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(
            ok,
            "{lane}: the nested named claim is unenforced (chelis#1800): {out}"
        );
        assert!(
            out.contains("shape=[3]") && out.contains("data=[2.0, 3.0, 4.0]"),
            "{lane}: and the extent the shrink computed is returned: {out}"
        );
        assert!(
            !out.contains("extent `n`"),
            "{lane}: no guard claims to have checked it: {out}"
        );
    }

    // The literal half through the SAME nesting: guarded.
    let literal = nested_named_result_source(4, "2");
    let (eval_ok, eval_out) = eval_result(&dir, "nested_lit.ch", &literal);
    let (c_ok, c_out) = c_run_result(&dir, "nested_lit_c", &literal);
    assert!(!eval_ok, "a literal claim survives the nesting: {eval_out}");
    assert!(!c_ok, "a literal claim survives the nesting: {c_out}");
    let context = "extent `2`: claimed = 2, shrink axis 0 = 3";
    for (lane, out) in [("eval", &eval_out), ("c", &c_out)] {
        assert!(
            out.contains(&domain_trap_line("shrink")),
            "{lane}: [04-NUM-9]'s line: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
    }

    // And the named half's agreeing control still executes exactly, so the
    // first block above is pinning an unguarded claim and not a broken lane.
    let agreeing = nested_named_result_source(3, "n");
    let (eval_ok, eval_out) = eval_result(&dir, "nested_named_ok.ch", &agreeing);
    assert!(eval_ok, "an agreeing nested claim executes: {eval_out}");
    assert!(
        eval_out.contains("shape=[2]") && eval_out.contains("data=[2.0, 3.0]"),
        "with the declared extent: {eval_out}"
    );
}

/// The op-computed guard-order pair: a declared 2 over a shrink that produces
/// `len - 1`, with the independent effect on one side of the guarded call.
fn op_computed_effect_order_source(len: usize, effect_first: bool) -> String {
    let effect = "_ = print(\"effect\")";
    let shrunk = "narrowed = f(x)";
    let (first, second) = if effect_first {
        (effect, shrunk)
    } else {
        (shrunk, effect)
    };
    format!(
        "def f(x: tensor[rows, f32]) -> tensor[2, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
         def run(x: tensor[rows, f32]) -> tensor[2, f32] ! {{ IO }} = {{\n\
         \x20 {first}\n\
         \x20 {second}\n\
         \x20 narrowed\n\
         }}\n\
         out = run({})\n",
        vector_literal(len)
    )
}

/// guard_order.op_computed.effect_before.c: an effect that precedes the guarded
/// operation in source order is observed even though the guard traps.
///
/// EVIDENTIARY STATUS: regression test. On the base neither variant trapped at
/// all, so neither direction of the order was observable.
#[test]
fn an_effect_before_an_op_computed_guard_runs_when_the_guard_traps_on_c() {
    assert!(
        gcc_available(),
        "this row executes a linked program; no lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(
        &dir,
        "opc_effect_before",
        &op_computed_effect_order_source(4, true),
    );
    assert!(!ok, "the binary must fail: {out}");
    assert_eq!(
        out.lines().filter(|line| *line == "effect").count(),
        1,
        "the preceding effect is observed: {out}"
    );
    assert!(out.contains(&domain_trap_line("shrink")), "{out}");
}

/// guard_order.op_computed.effect_after.c: an effect that follows the guarded
/// operation is observed only if the guard passes.
///
/// EVIDENTIARY STATUS: regression test, with the preceding-effect row as one
/// non-vacuity control and the agreeing variant below as the other.
#[test]
fn an_effect_after_an_op_computed_guard_does_not_run_when_the_guard_traps_on_c() {
    assert!(
        gcc_available(),
        "this row executes a linked program; no lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(
        &dir,
        "opc_effect_after",
        &op_computed_effect_order_source(4, false),
    );
    assert!(!ok, "the binary must fail: {out}");
    assert!(
        !out.lines().any(|line| line == "effect"),
        "the later effect is not observed: {out}"
    );
    assert!(out.contains(&domain_trap_line("shrink")), "{out}");

    let (ok, out) = c_run_result(
        &dir,
        "opc_effect_after_ok",
        &op_computed_effect_order_source(3, false),
    );
    assert!(ok, "an agreeing claim runs the later effect: {out}");
    assert_eq!(
        out.lines().filter(|line| *line == "effect").count(),
        1,
        "{out}"
    );
}

/// A result name declared NOWHERE else is not a claim this change can stamp:
/// the checker publishes it as `*` under section 4.7.2's fresh-extent rule, so
/// it reaches neither the declaration arm nor a class, and the program runs.
///
/// EVIDENTIARY STATUS: disposition lock, not a regression test. This is the
/// base behaviour and this change preserves it deliberately: the wildcard
/// result is chelis#1397's OTHER half (a wildcard-returning root), and keeping
/// it unguarded is what leaves chelis#1378's `vmap` witness executable.
#[test]
fn a_result_name_no_parameter_declares_is_not_stamped_as_a_claim() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(y: tensor[k, f32]) -> tensor[m, f32] = shrink(y, [[1i64, shape(y, 0i32)]])\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\n";
    let path = fixture(&dir, "free_result_name.ch", source);
    let checked = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "check",
            "--show-inferred",
            "--allow-style-violations",
            path.to_str().unwrap(),
        ])
        .output()
        .expect("check");
    let report = String::from_utf8_lossy(&checked.stdout).to_string();
    assert!(
        report.contains("(tensor[d0, f32]) -> tensor[*, f32]"),
        "the checker still publishes the free result name as a wildcard: {report}"
    );
    let (ok, out) = eval_result(&dir, "free_result_name_run.ch", source);
    assert!(ok, "and the program executes unguarded: {out}");
    assert!(
        out.contains("shape=[3]") && out.contains("data=[2.0, 3.0, 4.0]"),
        "producing the extent the operation computed: {out}"
    );
}

/// The unadmitted op-computed owners, stated as a lock rather than left to be
/// discovered. `spec/04-type-system.md` section 4.7 makes a non-unit `stride`
/// step and non-zero `pad` mint fresh extents exactly as `shrink` does, and
/// this change admits only `shrink`: those two remain unguarded, unchanged
/// rather than newly silent.
///
/// EVIDENTIARY STATUS: disposition lock, not a regression test. The behaviour
/// asserted here is the base behaviour and this change does not alter it; the
/// row exists so that admitting either owner has to update this test.
#[test]
fn pad_and_stride_op_computed_extents_remain_unadmitted() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, source) in [
        (
            "stride_unadmitted.ch",
            "def f(x: tensor[rows, f32]) -> tensor[2, f32] = stride(x, 2i64)\n\
             out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))\n",
        ),
        (
            "pad_unadmitted.ch",
            "def f(x: tensor[rows, f32]) -> tensor[2, f32] = pad(x, [[1i64, 1i64]], 0.0f32)\n\
             out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
        ),
    ] {
        let (ok, out) = eval_result(&dir, name, source);
        assert!(
            ok,
            "{name}: an unadmitted op-computed owner is unchanged by this \
             change, not newly trapping: {out}"
        );
    }
}

// ---------------------------------------------------------------------------
// Round 1's two confirmed findings, pinned.
// ---------------------------------------------------------------------------

/// The binder's only declaring parameter is unread, so nothing enforces the
/// claim the signature makes about the result.
fn unread_declaring_parameter_source(len: usize) -> String {
    format!(
        "def f(w: tensor[n, f32], x: tensor[r, f32]) -> tensor[n, f32] = \
         shrink(x, [[1i64, shape(x, 0i32)]])\n\
         out = f(to_tensor([1.0f32, 2.0f32]), {})\n",
        vector_literal(len)
    )
}

/// A named result claim is enforced even when the binder's only declaring
/// parameter is UNREAD. `n` is 2 at the call and the shrink produces 3.
///
/// The claim makes the declaring parameter a dependency of the result, so `w`
/// becomes a kernel input and the canonical value is the caller's rather than
/// the operation's own. Without that dependency PR #1773's witness for a
/// binder declared by a SINGLE parameter carries neither a requirement nor a
/// claim, `retain_invocation_witnesses` drops it, elimination takes `w`'s
/// `Load`, and the class has one member: C2.4 does not make that a class and
/// nothing compares anything. `scope.unread` in the preparation corpus is the
/// same shape for a REPEATED binder, where PR #1773's own comparison already
/// retains the witness.
///
/// EVIDENTIARY STATUS: regression test for the mismatch assertions. Measured
/// on `cab086ef4`, where the admission, the stamp and the ordering key were
/// all already in place, this program printed
/// `out = tensor(shape=[3], data=[2.0, 3.0, 4.0])` and exited ZERO on both
/// lanes under a declared `tensor[n, f32]` with `n` = 2. The satisfied row is
/// the non-vacuity control and passed there.
#[test]
fn an_unread_declaring_parameter_still_guards_its_named_result_claim() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let good = unread_declaring_parameter_source(3);
    let (eval_ok, eval_out) = eval_result(&dir, "unread_declarer_ok.ch", &good);
    let (c_ok, c_out) = c_run_result(&dir, "unread_declarer_ok_c", &good);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: a shrink that produces `n` = 2 executes: {out}");
        assert!(
            out.contains("shape=[2]") && out.contains("data=[2.0, 3.0]"),
            "{lane}: and produces the declaring parameter's extent: {out}"
        );
    }

    let bad = unread_declaring_parameter_source(4);
    let (eval_ok, eval_out) = eval_result(&dir, "unread_declarer_bad.ch", &bad);
    let (c_ok, c_out) = c_run_result(&dir, "unread_declarer_bad_c", &bad);
    assert!(
        !eval_ok,
        "a shrink that produces 3 under `n` = 2 traps: {eval_out}"
    );
    assert!(
        !c_ok,
        "a shrink that produces 3 under `n` = 2 traps: {c_out}"
    );
    // `claimed = 2` rather than `w axis 0 = 2`: the canonical side of every
    // local guard renders that way, and naming the declaring source is a
    // `LocalGuardClaim` representation change tracked by chelis#1794.
    let context = "extent `n`: claimed = 2, shrink axis 0 = 3";
    for (lane, out) in [("eval", &eval_out), ("c", &c_out)] {
        assert!(
            out.contains(&domain_trap_line("shrink")),
            "{lane}: [04-NUM-9]'s line names the operation: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
        assert!(
            !out.contains("shape=[3]"),
            "{lane}: and no undeclared extent is returned: {out}"
        );
    }
}

/// A `shrink` whose span selects nothing, under a declared literal.
fn empty_span_source(len: usize) -> String {
    format!(
        "def f(x: tensor[rows, f32]) -> tensor[2, f32] = \
         shrink(x, [[shape(x, 0i32), shape(x, 0i32)]])\n\
         out = f({})\n",
        vector_literal(len)
    )
}

/// An empty span: the two lanes disagree about WHICH failure it is, and the
/// numbered spec says the compiled lane is right.
///
/// `spec/05-risc-primitives.md` section 2.4.1 lists the runtime-bound errors
/// as a negative bound, a shrink range overshoot, a non-positive stride step
/// and the two reshape errors. An empty span is not among them, and
/// `spec/04-type-system.md` section 4.7.2 makes only a NEGATIVE size an error.
/// So `start == end` should produce an extent-0 result, and an extent-0 result
/// under a declared `tensor[2, f32]` is a claim mismatch: C's
/// `claimed = 2, shrink axis 0 = 0` is the conforming diagnostic. The
/// evaluator instead rejects the span under chelis#616's operation-level
/// admission rule, which the numbered spec does not require.
///
/// Round 1 read this the other way round and asked for C to be reordered
/// behind the evaluator. That change was made, then the atom was checked, and
/// it was taken out: it would have moved the conforming lane onto the
/// non-conforming one. The divergence is an operation-level admission rule
/// against section 2.4.1's closed list; it predates this slice, and mere
/// discovery during review does not bring it into scope. Tracked by
/// chelis#1795, and chelis#1481 asks for the opposite direction on a premise
/// chelis#1795 questions.
///
/// EVIDENTIARY STATUS: disposition lock on both lanes, not a regression test.
/// Both assertions describe the behaviour on the reviewed head `cab086ef4` and
/// on this one. The row exists so the divergence is recorded with its spec
/// citation rather than rediscovered, and so that changing either lane has to
/// update it.
#[test]
fn an_empty_shrink_span_diverges_across_lanes_under_the_616_admission_rule() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let source = empty_span_source(4);
    let (eval_ok, eval_out) = eval_result(&dir, "empty_span.ch", &source);
    let (c_ok, c_out) = c_run_result(&dir, "empty_span_c", &source);
    assert!(
        !eval_ok,
        "an empty span does not produce a value: {eval_out}"
    );
    assert!(!c_ok, "an empty span does not produce a value: {c_out}");
    assert!(
        eval_out.contains("shrink axis 0 bound [4, 4] is empty or inverted"),
        "eval rejects the span itself (chelis#616): {eval_out}"
    );
    assert!(
        c_out.contains("extent `2`: claimed = 2, shrink axis 0 = 0")
            && c_out.contains(&domain_trap_line("shrink")),
        "C reports the claim the extent-0 result refutes, which is what \
         section 2.4.1's closed error list implies: {c_out}"
    );
}

/// A `shrink` whose end exceeds the operand's extent, under a declared literal.
fn end_beyond_operand_source(len: usize) -> String {
    format!(
        "def f(x: tensor[rows, f32]) -> tensor[2, f32] = \
         shrink(x, [[1i64, add(shape(x, 0i32), 3i64)]])\n\
         out = f({})\n",
        vector_literal(len)
    )
}

/// The limit of this slice's cross-lane claim, pinned rather than implied.
///
/// `spec/05-risc-primitives.md` section 2.4.1 makes "a shrink range overshoot"
/// an error every execution mode reports with matching language, and the two
/// lanes do not match here: C's movement plan rejects the span before the guard
/// site, while the evaluator's guard runs first and reports the claim. The
/// evaluator cannot simply decline, because its own answer to an overshooting
/// span is the `assert!` in `eval::shrink`, which PANICS rather than returning
/// a typed error, so declining the guard would trade a diagnostic naming the
/// wrong reason for a panic.
///
/// This slice therefore bounds its cross-lane claim to IN-DOMAIN spans and
/// pins both dispositions here. Closing the divergence belongs to chelis#1797.
/// chelis#523, which the assertion's own message cites, is CLOSED and is that
/// issue's predecessor, not its owner.
///
/// EVIDENTIARY STATUS: disposition lock for both lanes, not a regression test.
/// Neither assertion describes an improvement; both describe a limit, and the
/// test exists so that fixing chelis#1797 has to update it.
#[test]
fn an_overshooting_shrink_span_is_outside_this_slices_cross_lane_claim() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let source = end_beyond_operand_source(4);
    let (eval_ok, eval_out) = eval_result(&dir, "end_beyond.ch", &source);
    let (c_ok, c_out) = c_run_result(&dir, "end_beyond_c", &source);
    assert!(
        !eval_ok,
        "an overshoot does not produce a value: {eval_out}"
    );
    assert!(!c_ok, "an overshoot does not produce a value: {c_out}");
    assert!(
        eval_out.contains("extent `2`: claimed = 2, shrink axis 0 = 6"),
        "eval's guard runs first and names the claim (chelis#1797): {eval_out}"
    );
    assert!(
        c_out.contains("shrink bounds outside input extent"),
        "C's movement plan rejects the span first: {c_out}"
    );
}

// ---------------------------------------------------------------------------
// chelis#1782: a root's restated literal claim defers to the callee's named
// guard.
//
// At `def main() = f(...)` the checker infers the root's result type by
// instantiating `f`'s dimension binder against the argument it was bound
// from, so the root's result dimension is the LITERAL that binder
// monomorphized to. Lowering then recorded that literal as a second
// obligation on the same produced extent, and because it is a requirement on
// the witness rather than a claim between two witnesses, it rendered first:
// the user saw ``extent `2`: claimed = 2, y axis 0 = 3`` where
// `spec/04-type-system.md` section 4.7's [04-NUM-9] asks for both disagreeing
// sources, which the callee's own named guard already names.
//
// The two obligations are one comparison. The named claim asserts that the
// produced witness and the declaring witness observe the same extent; the
// declaring witness here observes a tensor whose extent the lowered graph
// fixes, and that extent IS the literal. A produced extent disagreeing with
// the literal therefore disagrees with the declaring witness too, so the
// named guard fires on exactly the inputs the literal guard would have.
// Section 4.7 evaluates each guard once, so the one that survives is the one
// naming both sources.
//
// These rows are inlined-root rows, which the file header's older paragraph
// said could not be guarded at all. PR #1773 and PR #1790 changed that; the
// header is corrected where it makes the claim.
// ---------------------------------------------------------------------------

/// chelis#1374's reproducer with the issue's original polymorphic binder, in
/// the inlined-root form. `n` is a single letter, so it desugars to a
/// polymorphic dimension variable that inference instantiates against `x`.
fn polymorphic_named_root_source(x_extent: usize, y_extent: usize) -> String {
    let list = |n: usize| {
        (1..=n)
            .map(|v| format!("{v}.0f32"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "module Repro.PolymorphicNamedRoot\n\
         def f(b: tensor[f32], x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = insert(b, 0i32, shape(y, 0i32))\n\
         def main() = f(scalar_to_tensor(7.0f32), to_tensor([{}]), to_tensor([{}]))\n",
        list(x_extent),
        list(y_extent)
    )
}

/// chelis#1376's reproducer with the issue's original polymorphic binders, in
/// the inlined-root form. The result's two axes resolve to DIFFERENT
/// witnesses, so a per-witness reorder would repair the row above and not
/// this one.
fn polymorphic_foreign_root_source(x_extent: usize, y_extent: usize) -> String {
    let list = |n: usize| {
        (1..=n)
            .map(|v| format!("{v}.0f32"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "module Repro.PolymorphicForeignRoot\n\
         def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, m, f32] = insert(x, 1i32, shape(x, 0i32))\n\
         def main() = f(to_tensor([{}]), to_tensor([{}]))\n",
        list(x_extent),
        list(y_extent)
    )
}

/// A root literal claim with NO named claim over its produced extent: the
/// callee declares a literal result of its own, so nothing relates two
/// witnesses and the literal is the only check there is.
fn independent_literal_root_source(x_extent: usize) -> String {
    let list = (1..=x_extent)
        .map(|v| format!("{v}.0f32"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "module Repro.IndependentLiteralRoot\n\
         def f(b: tensor[f32], x: tensor[rows, f32]) -> tensor[4, f32] = insert(b, 0i32, shape(x, 0i32))\n\
         def main() = f(scalar_to_tensor(7.0f32), to_tensor([{list}]))\n"
    )
}

/// A binding whose entailing extent arrives through the ABI: `g`'s binder `n`
/// is declared by `f`'s parameter `a`, and `a`'s extent is an input promise
/// the entry guard checks rather than one the lowered graph fixes.
///
/// `a_prim` types `f`'s first parameter and `a_arg` spells what `f` hands
/// `g`, so one helper produces the direct spelling and the two that put a
/// node between the parameter and the witness.
fn abi_promised_entailment_source(a_prim: &str, a_arg: &str, b_extent: usize) -> String {
    let list = |n: usize, prim: &str| {
        (1..=n)
            .map(|v| format!("{v}.0{prim}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "module Repro.AbiPromisedEntailment\n\
         def g(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = insert(scalar_to_tensor(7.0f32), 0i32, shape(y, 0i32))\n\
         def f(a: tensor[4, {a_prim}], b: tensor[rows, f32]) -> tensor[4, f32] = g({a_arg}, b)\n\
         out = f(to_tensor([{}]), to_tensor([{}]))\n",
        list(4, a_prim),
        list(b_extent, "f32")
    )
}

/// polymorphic.named.root.mismatch (chelis#1782, chelis#1374)
///
/// EVIDENTIARY STATUS: regression test. Recorded red at `d861a6c6f`, where
/// both lanes print ``extent `2`: claimed = 2, y axis 0 = 3`` and the named
/// guard that names both disagreeing sources never runs. The satisfied half
/// is a parity control: suppressing the restatement must not stop the
/// agreeing program from producing its value.
#[test]
fn issue_1782_a_root_literal_restating_a_binder_defers_to_the_named_guard_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "poly_named_root.ch",
        &polymorphic_named_root_source(2, 3),
    );
    assert!(
        !ok,
        "`n` is witnessed at 2 and the result is produced at 3: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "the guard the user sees names both disagreeing sources: {out}"
    );
    assert!(
        !out.contains("claimed = 2"),
        "the root's restatement of `n` records no second guard: {out}"
    );

    let (ok, out) = eval_result(
        &dir,
        "poly_named_root_ok.ch",
        &polymorphic_named_root_source(3, 3),
    );
    assert!(ok, "an agreeing pair of witnesses must execute: {out}");
    assert!(
        out.contains("tensor(shape=[3], data=[7.0, 7.0, 7.0])"),
        "and produce the claimed shape: {out}"
    );
}

/// polymorphic.named.root.mismatch on the C lane (chelis#1782, chelis#1374)
///
/// EVIDENTIARY STATUS: as its eval twin; recorded red at `d861a6c6f` with the
/// linked binary printing ``extent `2`: claimed = 2, y axis 0 = 3``. The
/// context line is asserted byte-for-byte against the eval row's.
#[test]
fn issue_1782_a_root_literal_restating_a_binder_defers_to_the_named_guard_on_c() {
    assert!(
        gcc_available(),
        "this row executes a linked program; no lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(
        &dir,
        "poly_named_root_c",
        &polymorphic_named_root_source(2, 3),
    );
    assert!(
        !ok,
        "`n` is witnessed at 2 and the result is produced at 3: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "the guard the user sees names both disagreeing sources: {out}"
    );
    assert!(
        !out.contains("claimed = 2"),
        "the root's restatement of `n` records no second guard: {out}"
    );

    let (ok, out) = c_run_result(
        &dir,
        "poly_named_root_c_ok",
        &polymorphic_named_root_source(3, 3),
    );
    assert!(ok, "an agreeing pair of witnesses must execute: {out}");
    assert!(
        out.contains("tensor(shape=[3], data=[7.0, 7.0, 7.0])"),
        "and produce the claimed shape: {out}"
    );
}

/// polymorphic.foreign.root.mismatch (chelis#1782, chelis#1376)
///
/// EVIDENTIARY STATUS: regression test. Recorded red at `d861a6c6f`, where
/// both lanes print ``extent `3`: claimed = 3, x axis 0 = 2``. This row's
/// literal and named obligations sit on DIFFERENT witnesses, with the
/// literal's earlier in the schedule, so it fails for a reorder that repairs
/// the same-witness row above.
#[test]
fn issue_1782_a_root_literal_restating_a_foreign_binder_defers_to_the_named_guard_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "poly_foreign_root.ch",
        &polymorphic_foreign_root_source(2, 3),
    );
    assert!(
        !ok,
        "`m` is witnessed at 3 and the result is produced at 2: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `m`: y axis 0 = 3, x axis 0 = 2"),
        "the guard the user sees names both disagreeing sources: {out}"
    );
    assert!(
        !out.contains("claimed = 3"),
        "the root's restatement of `m` records no second guard: {out}"
    );

    let (ok, out) = eval_result(
        &dir,
        "poly_foreign_root_ok.ch",
        &polymorphic_foreign_root_source(2, 2),
    );
    assert!(ok, "an agreeing pair of witnesses must execute: {out}");
    assert!(
        out.contains("tensor(shape=[2, 2], data=[1.0, 1.0, 2.0, 2.0])"),
        "and produce the claimed shape: {out}"
    );
}

/// polymorphic.foreign.root.mismatch on the C lane (chelis#1782, chelis#1376)
///
/// EVIDENTIARY STATUS: as its eval twin; recorded red at `d861a6c6f`.
#[test]
fn issue_1782_a_root_literal_restating_a_foreign_binder_defers_to_the_named_guard_on_c() {
    assert!(
        gcc_available(),
        "this row executes a linked program; no lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(
        &dir,
        "poly_foreign_root_c",
        &polymorphic_foreign_root_source(2, 3),
    );
    assert!(
        !ok,
        "`m` is witnessed at 3 and the result is produced at 2: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `m`: y axis 0 = 3, x axis 0 = 2"),
        "the guard the user sees names both disagreeing sources: {out}"
    );
    assert!(
        !out.contains("claimed = 3"),
        "the root's restatement of `m` records no second guard: {out}"
    );

    let (ok, out) = c_run_result(
        &dir,
        "poly_foreign_root_c_ok",
        &polymorphic_foreign_root_source(2, 2),
    );
    assert!(ok, "an agreeing pair of witnesses must execute: {out}");
    assert!(
        out.contains("tensor(shape=[2, 2], data=[1.0, 1.0, 2.0, 2.0])"),
        "and produce the claimed shape: {out}"
    );
}

/// The negative twin, on both lanes in one row so the two renderings are
/// compared against one another.
///
/// A root literal claim is suppressed only when a named claim already makes
/// the same comparison. Here the callee declares its own literal result, no
/// named claim relates two witnesses over the produced axis, and the literal
/// guard is the only check there is: it stays, and it still names the claimed
/// extent.
///
/// The sharper twin a reader might expect, a root that DECLARES a literal
/// disagreeing with the binder's monomorphization, does not exist: the
/// checker rejects a root whose declared result disagrees with the type it
/// infers, so a root literal it accepts always equals the instantiation. That
/// is measured, not assumed; it is why the suppression's premise holds
/// whenever the restatement is recognised.
///
/// EVIDENTIARY STATUS: disposition lock. Both assertions pass at `d861a6c6f`
/// and describe behaviour this change must not alter.
#[test]
fn an_independent_root_literal_claim_still_names_its_claimed_extent() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let source = independent_literal_root_source(5);
    let (eval_ok, eval_out) = eval_result(&dir, "independent_literal_root.ch", &source);
    let (c_ok, c_out) = c_run_result(&dir, "independent_literal_root_c", &source);
    assert!(
        !eval_ok,
        "a literal claim of 4 over a read of 5 traps: {eval_out}"
    );
    assert!(
        !c_ok,
        "a literal claim of 4 over a read of 5 traps: {c_out}"
    );
    for out in [&eval_out, &c_out] {
        assert!(out.contains(&domain_trap_line("load")), "{out}");
        assert!(
            out.contains("extent `4`: claimed = 4, x axis 0 = 5"),
            "an independent literal keeps its claimed-extent rendering: {out}"
        );
    }
}

/// The rule's FIRST bound, and the negatives that fix it: an entailing extent
/// that arrives through the ABI keeps its literal guard, however many
/// pass-through hops separate the parameter from the witness.
///
/// The suppression requires the other witness of the named claim to observe an
/// axis whose extent the lowered GRAPH fixes. Here `g`'s binder `n` is
/// declared by `f`'s parameter `a`, whose extent reaches the program as an ABI
/// input spelled `tensor[4, ...]`: a promise the entry guard checks, not a
/// fact of this graph. `f`'s own literal claim is therefore recorded, and all
/// three spellings still render `claimed = 4`.
///
/// The three spellings are the finding. An earlier version of this row tested
/// only `g(a, b)` and the code decided "graph-fixed" by matching the observed
/// node's op against `RiscOp::Load`, so ONE node between the parameter and the
/// witness defeated the bound: red team round 1 measured `g(mul(a, a), b)` and
/// `g(cast(a, f32), b)` rendering ``extent `n`: x axis 0 = 4, y axis 0 = 5``
/// at `a1b54dbc1`, against ``extent `4`: claimed = 4, y axis 0 = 5`` at
/// `d861a6c6f`. The repair resolves the observed axis to its ORIGIN with
/// `axis_sources::resolve_axis_extent`, so an `ExternalAxis` origin is
/// recognised through any number of pass-through hops.
///
/// That bound is deliberate rather than a statement that the obligation is
/// independent here. The entry guard does pin `a axis 0` to 4, so the named
/// claim plus that guard entail the literal exactly as they do at an inlined
/// root; declining only on a graph-fixed extent keeps this change inside the
/// form chelis#1782 reports and leaves the ABI-promised form untouched.
///
/// EVIDENTIARY STATUS: mixed, per row. The `mul` and `cast` spellings are
/// REGRESSION tests, recorded red at `a1b54dbc1` with the named rendering. The
/// direct spelling is a DISPOSITION LOCK that passes at `d861a6c6f` and at
/// `a1b54dbc1`. The entry-guard assertion is a lock on why this is a
/// rendering finding and not a lost check.
#[test]
fn an_abi_promised_entailing_extent_keeps_its_literal_guard() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    for (label, a_prim, a_arg) in [
        ("direct", "f32", "a"),
        ("through_mul", "f32", "mul(a, a)"),
        ("through_cast", "f64", "cast(a, f32)"),
    ] {
        let source = abi_promised_entailment_source(a_prim, a_arg, 5);
        let (eval_ok, eval_out) = eval_result(&dir, &format!("abi_{label}.ch"), &source);
        let (c_ok, c_out, emitted) =
            c_run_result_with_source(&dir, &format!("abi_{label}_c"), &source);
        assert!(
            !eval_ok,
            "{label}: a literal claim of 4 over a read of 5 traps: {eval_out}"
        );
        assert!(
            !c_ok,
            "{label}: a literal claim of 4 over a read of 5 traps: {c_out}"
        );
        for out in [&eval_out, &c_out] {
            assert!(out.contains(&domain_trap_line("load")), "{label}: {out}");
            assert!(
                out.contains("extent `4`: claimed = 4, y axis 0 = 5"),
                "{label}: an ABI-promised entailing extent keeps the literal guard: {out}"
            );
        }
        assert!(
            emitted.contains("chelis_tensor_shape(inputs[0], 0) != 4"),
            "{label}: the kernel still checks `a`'s promised extent at entry, which is \
             why the declined suppression would cost no check: {emitted}"
        );
    }
}

/// The rule's SECOND bound: the two call forms that never inline the callee
/// into a root render exactly as they did.
///
/// A value binding applies the exported kernel, so `f`'s parameters are
/// `Load`s and no enclosing root contributes an inferred literal. The four
/// exported-kernel cells of the same two programs,
/// `polymorphic.{named,foreign}.export.mismatch` on both lanes, are locked by
/// the preparation baseline rather than repeated here.
///
/// EVIDENTIARY STATUS: disposition lock. All four assertions pass at
/// `d861a6c6f` and describe behaviour this change must not alter.
#[test]
fn the_value_binding_form_of_a_polymorphic_binder_is_unchanged() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let named = polymorphic_named_root_source(2, 3).replace("def main() = f(", "out = f(");
    for (ok, out) in [
        eval_result(&dir, "poly_named_binding.ch", &named),
        c_run_result(&dir, "poly_named_binding_c", &named),
    ] {
        assert!(!ok, "the exported kernel's entry guard still fires: {out}");
        assert!(
            out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
            "the value-binding form's rendering is unchanged: {out}"
        );
    }

    let foreign = polymorphic_foreign_root_source(2, 3).replace("def main() = f(", "out = f(");
    for (ok, out) in [
        eval_result(&dir, "poly_foreign_binding.ch", &foreign),
        c_run_result(&dir, "poly_foreign_binding_c", &foreign),
    ] {
        assert!(!ok, "the exported kernel's entry guard still fires: {out}");
        assert!(
            out.contains("extent `m`: y axis 0 = 3, x axis 0 = 2"),
            "the value-binding form's rendering is unchanged: {out}"
        );
    }
}

// ===========================================================================
// Record-field and pipe shape sources (chelis#1266, chelis#569).
//
// Rows `expand.record_projection.size` and `expand.piped_shape_read.lint_fix`.
//
// Both issues are one question asked twice: does a `shape(...)` read reach
// the size slot when it is SPELLED differently. A record field and a pipe
// stage are both spellings a person and a tool actually produce -- every
// hydronnx model with more than one input takes its arguments as a record,
// and `chelis lint --fix` rewrites a nested call chain into a pipe -- and
// `spec/04-type-system.md` section 4.7.2 admits an extent by what supplies
// it, never by how it is written: "no stage may reject an extent because of
// its provenance".
//
// The claim these rows carry is admissibility plus execution on both lanes.
// It is NOT that a declared result NAME survives: `f(inp: Inputs)` under
// `sig f: Inputs -> tensor[batch, f32]` still checks to `tensor[*, f32]`,
// because `batch` appears nowhere in the parameter list for unification to
// bind it to. That erasure is chelis#1397's checker half. The C lane does
// declare the extent by name from the field's axis (`int64_t batch =
// chelis_tensor_shape(inputs[0], 0)`), which is what makes the compiled
// program agree with eval below.
// ===========================================================================

/// chelis#1266's own reproducer: the projection read inline in the size slot.
const RECORD_PROJECTION_DIRECT: &str = "module Repro.RecordDirect\n\
     type Inputs = | Inputs { q: tensor[batch, 4, f32] }\n\
     sig f: Inputs -> tensor[batch, f32]\n\
     def f(inp: Inputs) = expand(to_tensor([0.25f32]), 0i32, shape(inp.q, 0i32))\n\
     out = f(Inputs { q: to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], \
     [5.0f32, 6.0f32, 7.0f32, 8.0f32]]) })\n";

/// The same read through a local, the rewrite chelis#1266 reports downstream
/// applying by hand. It is the control: identical semantics, and the whole
/// difference is the spelling.
const RECORD_PROJECTION_ALIAS: &str = "module Repro.RecordAlias\n\
     type Inputs = | Inputs { q: tensor[batch, 4, f32] }\n\
     sig f: Inputs -> tensor[batch, f32]\n\
     def f(inp: Inputs) = { q = inp.q\n\
     expand(to_tensor([0.25f32]), 0i32, shape(q, 0i32)) }\n\
     out = f(Inputs { q: to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], \
     [5.0f32, 6.0f32, 7.0f32, 8.0f32]]) })\n";

/// A field of a field. The projection resolves through the whole chain, and
/// each link binds one host local.
const RECORD_PROJECTION_NESTED: &str = "module Repro.RecordNested\n\
     type Inner = | Inner { q: tensor[batch, 4, f32] }\n\
     type Outer = | Outer { inner: Inner }\n\
     sig f: Outer -> tensor[batch, f32]\n\
     def f(o: Outer) = expand(to_tensor([0.25f32]), 0i32, shape(o.inner.q, 0i32))\n\
     out = f(Outer { inner: Inner { q: to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], \
     [5.0f32, 6.0f32, 7.0f32, 8.0f32]]) } })\n";

/// The shape chelis#1266 was filed for: `forward(inputs: ForwardInputs)` with
/// two tensor fields, one sizing a broadcast against the other.
const RECORD_PROJECTION_FORWARD: &str = "module Repro.RecordForward\n\
     type ForwardInputs = | ForwardInputs { a: tensor[batch, f32], b: tensor[1, f32] }\n\
     sig forward: ForwardInputs -> tensor[batch, f32]\n\
     def forward(inputs: ForwardInputs) = \
     add(expand(inputs.b, 0i32, shape(inputs.a, 0i32)), inputs.a)\n\
     out = forward(ForwardInputs { a: to_tensor([1.0f32, 2.0f32, 3.0f32]), \
     b: to_tensor([0.25f32]) })\n";

/// The record reaches the projecting def as a PARAMETER of another def, so
/// the base is a runtime value at two removes from the construction.
const RECORD_PROJECTION_THROUGH_A_DEF: &str = "module Repro.RecordThroughDef\n\
     type Inputs = | Inputs { q: tensor[batch, 4, f32] }\n\
     sig inner: Inputs -> tensor[batch, f32]\n\
     def inner(inp: Inputs) = expand(to_tensor([0.25f32]), 0i32, shape(inp.q, 0i32))\n\
     sig outer: Inputs -> tensor[batch, f32]\n\
     def outer(inp: Inputs) = inner(inp)\n\
     out = outer(Inputs { q: to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], \
     [5.0f32, 6.0f32, 7.0f32, 8.0f32]]) })\n";

/// A record built at compile time and projected inside the same body. The
/// constructor is a compile-time fact, so `lower_access` projects it and the
/// def keeps its DAG route: no `chelis_adt_get_field` reaches the emitted C.
const COMPILE_TIME_RECORD_PROJECTION: &str = "module Repro.StaticRecord\n\
     type Pair = | Pair { lhs: tensor[2, f32], rhs: tensor[2, f32] }\n\
     sig f: tensor[2, f32] -> tensor[2, f32]\n\
     def f(x: tensor[2, f32]) = { p = Pair { lhs: x, rhs: to_tensor([10.0f32, 20.0f32]) }\n\
     add(p.lhs, p.rhs) }\n\
     out = f(to_tensor([1.0f32, 2.0f32]))\n";

/// chelis#569's `with_pipe_cast` in today's canonical spelling. The v0.18
/// lambda the issue quotes (`|> fn (p) -> cast(p, int64)`) is now a parse
/// error; `|> cast(int64)` is what the parser and the formatter produce.
const PIPED_SHAPE_READ: &str = "module Repro.PipedShapeRead\n\
     sig broadcast_rows: tensor[a, f32] -> tensor[1, f32] -> tensor[a, f32]\n\
     def broadcast_rows(x: tensor[a, f32], b: tensor[1, f32]) = \
     { a_dim = x |> shape(cast(0, int32)) |> cast(int64)\n\
     expand(b, cast(0, int32), a_dim) }\n\
     out = broadcast_rows(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([0.25f32]))\n";

/// chelis#569's `with_direct_cast`: the form that builds, and the one
/// `prefer-pipe-operator` rewrites. The inputs are bound to names so the only
/// replacement the fix makes is the shape read under test; a call written
/// inline would also be piped, and a bare-name pipe stage at a call site
/// still loses its shape source at lowering (chelis#1791, out of scope here).
const DIRECT_SHAPE_READ_FOR_LINT_FIX: &str = "sig broadcast_rows: tensor[a, f32] -> tensor[1, f32] -> tensor[a, f32]\n\
     def broadcast_rows(x: tensor[a, f32], b: tensor[1, f32]) = {\n  \
     a_dim = cast(shape(x, cast(0, int32)), int64)\n  \
     expand(b, cast(0, int32), a_dim)\n\
     }\n\
     xs = to_tensor([1.0f32, 2.0f32, 3.0f32])\n\
     bias = to_tensor([0.25f32])\n\
     out = broadcast_rows(xs, bias)\n";

/// A piped read of a value that is not a tensor. The stage parameter carries
/// the upstream value's class, so a runtime scalar stays sourceless through
/// however many stages.
const PIPED_SOURCELESS_SCALAR: &str = "module Repro.PipedSourceless\n\
     sig f: tensor[a, f32] -> int32 -> tensor[a, f32]\n\
     def f(x: tensor[a, f32], k: int32) = { a_dim = k |> cast(int64)\n\
     expand(to_tensor([0.25f32]), 0i32, a_dim) }\n";

/// The projection with an axis the field does not have.
const RECORD_PROJECTION_BAD_AXIS: &str = "module Repro.RecordBadAxis\n\
     type Inputs = | Inputs { q: tensor[batch, 4, f32] }\n\
     sig f: Inputs -> tensor[batch, f32]\n\
     def f(inp: Inputs) = expand(to_tensor([0.25f32]), 0i32, shape(inp.q, 2i32))\n";

/// Run a fixture on both lanes and assert they agree byte for byte on the
/// expected rendering.
fn assert_both_lanes_render(stem: &str, source: &str, expected: &str) {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, stem, source);
    assert!(ok, "the compiled program must run: {out}");
    assert!(
        out.contains(expected),
        "the compiled binary must render {expected}: {out}"
    );
    let evaluated = eval(&fixture(&dir, &format!("{stem}_eval.ch"), source));
    assert!(
        evaluated.status.success(),
        "eval must succeed: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert_eq!(
        out,
        String::from_utf8_lossy(&evaluated.stdout),
        "the compiled binary and eval must agree byte for byte"
    );
}

/// Oracle row `expand.record_projection.size` (chelis#1266).
///
/// Both spellings of the same read are admissible sizes that check, evaluate
/// and build, and the two agree with each other as well as across lanes.
///
/// EVIDENTIARY STATUS: regression test, per assertion.
///   * the direct spelling on CHECK: measured RED on `33cc78e84`, where
///     `chelis check` scores 0.976 and reports "`expand` size resolves to a
///     runtime scalar, but no tensor in scope carries it".
///   * the direct spelling on C: measured RED on `33cc78e84` once the check
///     arm admitted it, with `unsupported: builtin `expand` on `chelis build`
///     host emission (codegen:c)`.
///   * the alias spelling on EVAL: measured RED on `33cc78e84` with "`access`
///     on a runtime value is not supported by IR lowering".
///   * the alias spelling on C: measured GREEN on `33cc78e84`; that assertion
///     is a disposition lock, and it is here because the two spellings must
///     not diverge again.
#[test]
fn a_record_projection_is_an_admissible_expand_size() {
    assert_both_lanes_render(
        "record_direct",
        RECORD_PROJECTION_DIRECT,
        "shape=[2], data=[0.25, 0.25]",
    );
    assert_both_lanes_render(
        "record_alias",
        RECORD_PROJECTION_ALIAS,
        "shape=[2], data=[0.25, 0.25]",
    );
    assert_both_lanes_render(
        "record_nested",
        RECORD_PROJECTION_NESTED,
        "shape=[2], data=[0.25, 0.25]",
    );
}

/// The record reaches the projecting def through a second def's parameter.
///
/// EVIDENTIARY STATUS: regression test. Measured RED on `33cc78e84` with the
/// same check-lane rejection as the direct spelling.
#[test]
fn a_record_field_reached_through_a_def_parameter_is_an_admissible_expand_size() {
    assert_both_lanes_render(
        "record_through_def",
        RECORD_PROJECTION_THROUGH_A_DEF,
        "shape=[2], data=[0.25, 0.25]",
    );
}

/// The shape chelis#1266 names as the reason it matters: a multi-input
/// forward whose broadcast is sized from a sibling field.
///
/// EVIDENTIARY STATUS: regression test. Measured RED on `33cc78e84` with the
/// check-lane sourceless rejection.
#[test]
fn a_multi_input_record_forward_sizes_its_broadcast_from_a_sibling_field() {
    assert_both_lanes_render(
        "record_forward",
        RECORD_PROJECTION_FORWARD,
        "shape=[3], data=[1.25, 2.25, 3.25]",
    );
}

/// The projection becomes the helper's own tensor input, which is what makes
/// the direct spelling reach the same lowering as the local-alias spelling
/// rather than a second mechanism.
///
/// EVIDENTIARY STATUS: regression test. Measured RED on `33cc78e84`: the
/// direct spelling emitted no tensor helper at all, because the host lane
/// rejected `expand` outright.
#[test]
fn a_record_projection_becomes_the_tensor_helper_s_own_input() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out_dir = dir.path().join("record_direct_c-out");
    let build = build_c(
        &fixture(&dir, "record_direct_c.ch", RECORD_PROJECTION_DIRECT),
        &out_dir,
    );
    assert!(
        build.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted = fs::read_to_string(out_dir.join("record_direct_c.c")).expect("generated C");
    assert!(
        emitted.contains("__host_record_field_inp_q"),
        "the projected field is bound to a host local: {emitted}"
    );
    assert!(
        emitted.contains("input `__host_record_field_inp_q` at slot 0"),
        "and that local is the helper's slot-0 tensor input: {emitted}"
    );
    assert!(
        emitted.contains("int64_t batch = chelis_tensor_shape(inputs[0], 0);"),
        "the kept extent is declared from the field's own axis: {emitted}"
    );
}

/// Negative parity for the decision above: a record whose constructor IS a
/// compile-time fact keeps the DAG route `lower_access` already serves.
///
/// EVIDENTIARY STATUS: regression test for this pull request's own risk.
/// Measured RED against an intermediate head of this branch, where the new
/// uncarriable arm fired on a `let`-bound `(record ..)` literal and pushed
/// the whole def onto the host lane.
#[test]
fn a_compile_time_record_projection_keeps_its_dag_route() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out_dir = dir.path().join("static_record-out");
    let build = build_c(
        &fixture(&dir, "static_record.ch", COMPILE_TIME_RECORD_PROJECTION),
        &out_dir,
    );
    assert!(
        build.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted = fs::read_to_string(out_dir.join("static_record.c")).expect("generated C");
    assert!(
        !emitted.contains("chelis_adt_get_field"),
        "a compile-time record is projected at lowering, never through the \
         runtime ADT accessor: {emitted}"
    );
    assert!(
        !emitted.contains("__host_record_field_"),
        "and it needs no hoisted host local: {emitted}"
    );
    let evaluated = eval(&fixture(
        &dir,
        "static_record_eval.ch",
        COMPILE_TIME_RECORD_PROJECTION,
    ));
    assert!(
        String::from_utf8_lossy(&evaluated.stdout).contains("data=[11.0, 22.0]"),
        "and it still evaluates: {}",
        String::from_utf8_lossy(&evaluated.stdout)
    );
}

/// Negative parity for the admissible projection: an axis the field does not
/// have is still rejected, and now for the RIGHT reason.
///
/// EVIDENTIARY STATUS: regression test on the reason, disposition lock on the
/// rejection. Measured on `33cc78e84`: the program was already rejected, but
/// with the sourceless-size diagnostic, because the walk never resolved the
/// operand far enough to check its rank. The rejection must stay, and the
/// diagnostic must now name the axis.
#[test]
fn a_record_projection_with_an_out_of_range_axis_is_still_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let checked = check(&fixture(
        &dir,
        "record_bad_axis.ch",
        RECORD_PROJECTION_BAD_AXIS,
    ));
    let report = String::from_utf8_lossy(&checked.stdout).to_string();
    assert!(
        report.contains("shape axis 2 is out of bounds for rank 2 tensor"),
        "the axis bound is what rejects it: {report}"
    );
    assert!(
        !report.contains("no tensor in scope carries it"),
        "and it is no longer reported as having no shape source: {report}"
    );
}

/// Oracle row `expand.piped_shape_read.lint_fix` (chelis#569), positive half.
///
/// EVIDENTIARY STATUS: regression test, per assertion.
///   * CHECK: measured RED on `33cc78e84`, where the pipe node fell to the
///     classifier's fail-closed default and `chelis check` reported "`expand`
///     size resolves to the symbolic dimension `a_dim`, but no tensor in
///     scope carries it".
///   * EVAL and C: measured RED on `33cc78e84` after the check arm admitted
///     it, with "`expand` size resolves to `a_dim`, but no in-scope tensor
///     axis supplies that extent" from the lowerer -- the check-versus-build
///     disagreement chelis#469 exists to prevent.
#[test]
fn the_canonical_piped_shape_read_checks_evaluates_and_builds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let checked = check(&fixture(&dir, "piped_shape_read.ch", PIPED_SHAPE_READ));
    assert!(
        checked.status.success(),
        "the piped read must check: {}",
        String::from_utf8_lossy(&checked.stdout)
    );
    assert_both_lanes_render(
        "piped_shape_read",
        PIPED_SHAPE_READ,
        "shape=[3], data=[0.25, 0.25, 0.25]",
    );
}

/// The row's own name: what `chelis lint --fix` produces must still work.
///
/// This runs the real style path -- no `--allow-style-violations`, no
/// `CHELIS_STYLE_GATE_DISABLE` -- because the defect chelis#569 reports is
/// that following the style tool breaks a building program.
///
/// EVIDENTIARY STATUS: regression test. Measured RED on `33cc78e84`: the
/// fixture evaluates before the fix, `chelis lint --fix` rewrites the shape
/// read into the pipe form, and `chelis check` then rejects it.
#[test]
fn a_lint_fix_of_a_direct_shape_read_still_checks_evaluates_and_builds() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "lint_fix.ch", DIRECT_SHAPE_READ_FOR_LINT_FIX);
    let styled = |args: &[&str]| {
        Command::cargo_bin("chelis")
            .expect("chelis")
            .args(args)
            .output()
            .expect("styled chelis invocation")
    };
    let before = styled(&["eval", "--file", path.to_str().unwrap()]);
    assert!(
        String::from_utf8_lossy(&before.stdout).contains("data=[0.25, 0.25, 0.25]"),
        "the direct spelling works before the fix: {}",
        String::from_utf8_lossy(&before.stderr)
    );

    let fixed = styled(&["lint", "--fix", path.to_str().unwrap()]);
    assert!(
        String::from_utf8_lossy(&fixed.stdout).contains("fixed 1 replacement"),
        "the fix rewrites exactly the shape read: {}{}",
        String::from_utf8_lossy(&fixed.stdout),
        String::from_utf8_lossy(&fixed.stderr)
    );
    let formatted = styled(&["fmt", "--inplace", path.to_str().unwrap()]);
    assert!(formatted.status.success(), "fmt must succeed");
    let rewritten = fs::read_to_string(&path).expect("rewritten fixture");
    assert!(
        rewritten.contains("a_dim = x |> shape(cast(0, int32)) |> cast(int64)"),
        "the pipe form is what the tools produce: {rewritten}"
    );

    let relinted = styled(&["lint", "--check", path.to_str().unwrap()]);
    assert!(
        relinted.status.success(),
        "and the fixed program is lint-clean: {}",
        String::from_utf8_lossy(&relinted.stderr)
    );
    let checked = styled(&["check", path.to_str().unwrap()]);
    assert!(
        checked.status.success(),
        "the fixed program must still check: {}",
        String::from_utf8_lossy(&checked.stdout)
    );
    let evaluated = styled(&["eval", "--file", path.to_str().unwrap()]);
    assert!(
        String::from_utf8_lossy(&evaluated.stdout).contains("data=[0.25, 0.25, 0.25]"),
        "and evaluate to the same tensor: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let out_dir = dir.path().join("lint_fix-out");
    let built = styled(&[
        "build",
        path.to_str().unwrap(),
        "--target",
        "c",
        "-o",
        out_dir.to_str().unwrap(),
    ]);
    assert!(
        built.status.success(),
        "and build: {}",
        String::from_utf8_lossy(&built.stderr)
    );
    let status = link_generated(&out_dir, "lint_fix.c", "lint_fix");
    assert!(status.success(), "link failed: {status}");
    let run = StdCommand::new(out_dir.join("lint_fix"))
        .output()
        .expect("run compiled binary");
    assert!(
        String::from_utf8_lossy(&run.stdout).contains("data=[0.25, 0.25, 0.25]"),
        "and run: {}",
        String::from_utf8_lossy(&run.stdout)
    );
}

/// Negative parity for the pipe arm: a piped read of a non-tensor stays
/// sourceless, with the section 4.7.2 diagnostic unchanged.
///
/// EVIDENTIARY STATUS: disposition lock. Measured GREEN on `33cc78e84` and it
/// must stay green: admitting a pipe stage must not admit the bare runtime
/// scalar the pipe carries.
#[test]
fn a_piped_shape_read_of_a_non_tensor_is_still_sourceless() {
    let dir = tempfile::tempdir().expect("tempdir");
    let checked = check(&fixture(
        &dir,
        "piped_sourceless.ch",
        PIPED_SOURCELESS_SCALAR,
    ));
    let report = String::from_utf8_lossy(&checked.stdout).to_string();
    assert!(
        report.contains(
            "`expand` size resolves to the symbolic dimension `a_dim`, but no tensor in \
             scope carries it"
        ),
        "the piped runtime scalar keeps the section 4.7.2 rejection: {report}"
    );
}
