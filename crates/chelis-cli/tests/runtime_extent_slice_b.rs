//! Public CLI acceptance rows for runtime-extent Slice B (chelis#1277).
//!
//! `crates/chelis-ir/tests/runtime_extent_slice_b_classes.rs` locks the
//! DERIVATION: which output axes form one equality class, and in what order.
//! This file locks the OBSERVABLE consequence: that each class's guard runs,
//! in the position `spec/04-type-system.md` section 4.7 requires, rendering
//! the trap [04-NUM-9] requires, on both the eval and the compiled C lane.
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

    // The eval lane accepts this program on `main` and still does: it
    // computes shapes from values and never sees the anonymous extent.
    let evaluated = eval(&path);
    let stdout = String::from_utf8_lossy(&evaluated.stdout).to_string();
    assert!(
        evaluated.status.success(),
        "eval acceptance is unchanged: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let values = common::parse_tensor_data(&stdout, "out");
    let expected = [
        0.731_058_6_f64,
        0.119_202_92_f64,
        0.952_574_13_f64,
        0.982_013_76_f64,
    ];
    assert_eq!(values.len(), expected.len(), "{stdout}");
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
    }

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

/// Combined output of `chelis build --target c`, plus whether it exited zero.
/// Used where the expected outcome is a REJECTION rather than a run: section
/// 4.7 makes a violation proven from literals a type error, so the row that
/// checks one must observe the build, not a binary.
fn build_result(dir: &TempDir, name: &str, source: &str) -> (bool, String) {
    let out = build_c(
        &fixture(dir, &format!("{name}.ch"), source),
        &dir.path().join(format!("{name}-out")),
    );
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

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

/// A LITERAL claim of 4 over a read that yields 5. The class has one
/// `InputAxis`-sourced member, so `spec/04-type-system.md` section 4.7 places
/// its guard LOCALLY, at the `expand` that introduces the extent, which is
/// what makes it orderable against an in-body trap. chelis#1377's shape.
///
/// An all-interface class would not work here: section 4.7 runs those at
/// entry, "before any other operation of the function", so both order
/// controls would pass wherever the local guards went. chelis#1379's
/// arithmetic form would not work either, because the C lane rejects it at
/// lowering until b2.3 removes that rejection, and a control that fails at
/// build time is not measuring order.
fn guard_order_source(claim: u32, trap_first: bool) -> String {
    let trap = "boom = floor_div(1i64, sub(shape(xb, 0), shape(xb, 0)))";
    let widen = "widened = f(seed, x)";
    let (first, second) = if trap_first {
        (trap, widen)
    } else {
        (widen, trap)
    };
    format!(
        "def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[{claim}, f32] = expand(b, 0, shape(x, 0))\n\
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
/// runtime extent guard, at ENTRY rather than at the `expand`, in a rendering
/// [04-NUM-9] does not permit, while eval has no check at all. Asserting the
/// exact trap line is what makes this row fail on `main` rather than pass on
/// the neighbouring behaviour.
#[test]
fn c_independent_trap_after_a_mismatch_loses() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "after_c", &guard_order_source(MISMATCHED, false));
    assert!(!ok, "the binary must fail: {out}");
    assert!(
        out.contains(&domain_trap_line("expand")),
        "the extent guard introduces the extent and fires first: {out}"
    );
    assert!(
        !out.contains(DIV_ZERO_TRAP),
        "the later independent trap must not be reached: {out}"
    );
}

// ---------------------------------------------------------------------------
// Entry-guard order.
//
// Section 4.7: interface guards run "at function entry, in declared signature
// order, before any other operation of the function", and never "by binding
// name, hash iteration, or node identity".
// ---------------------------------------------------------------------------

/// Two interface classes both mismatch. The class whose declaring witness sits
/// in the earlier assigned slot must trap first. The claim names are chosen so
/// alphabetical order is the OPPOSITE of slot order: today's
/// `symbolic_bindings` groups in a `BTreeMap<String, _>` and would report the
/// other one.
///
/// Both rows below assert the exact [04-NUM-9] line, and that is not
/// decoration. Without it they PASS on `main` for a reason that has nothing
/// to do with guard order: the nullary-root path already refuses this program
/// with `unsupported: [05-UNS-1] unavailable root `main` ... dimension binder
/// `zdim` has inconsistent runtime witnesses: 2 and 3 ... unimplemented
/// chelis#912`, whose text satisfies "fails", "mentions zdim" and "does not
/// mention adim" all at once. Asserting the trap line is what makes these
/// rows measure the guard rather than the neighbouring receipt.
///
/// EVIDENTIARY STATUS: regression tests. Watched passing-for-the-wrong-reason
/// at b2.1, then failing once the trap-line assertion was added.
const TWO_ENTRY_CLASSES: &str = "def f(zz: tensor[zdim, f32], aa: tensor[adim, f32], p: tensor[zdim, f32], q: tensor[adim, f32]) -> tensor[zdim, f32] = add(zz, p)\n\
def main() = f(to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([1.0f32]))\n";

#[test]
fn c_entry_guards_run_in_slot_order_not_name_order() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "entry_c", TWO_ENTRY_CLASSES);
    assert!(!ok, "both classes mismatch, so the binary must fail: {out}");
    assert!(
        out.contains(&domain_trap_line("load")),
        "an all-interface class renders [04-NUM-9]'s line, not chelis#912's \
         root receipt: {out}"
    );
    assert!(
        out.contains("zdim"),
        "`zdim` occupies the earlier slot and is reported first: {out}"
    );
    assert!(
        !out.contains("adim"),
        "`adim` sorts first by name but its guard runs second: {out}"
    );
}

// ---------------------------------------------------------------------------
// Row moves. Each fixture is one verified by execution against a current
// reference binary, and its recorded `main` baseline is named beside it so a
// reader can tell a regression test from a disposition lock.
// ---------------------------------------------------------------------------

/// chelis#1374, baseline `silent_unguarded`: the emitted kernel binds `m` from
/// `y`, allocates at `{ m }`, and never compares it to `n`.
const REPRO_1374: &str = "def f(b: tensor[f32], x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = expand(b, 0, shape(y, 0))\n\
def main() = f(sum(to_tensor([1.0f32]), 0), to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";

#[test]
fn issue_1374_cross_tensor_read_traps_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "r1374_c", REPRO_1374);
    assert!(!ok, "n = 2 and m = 3 must not execute silently: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
}

/// chelis#1376, baseline `silent_unguarded`: the inserted axis is claimed as
/// `m` but its runtime extent is `shape(x, 0)`.
const REPRO_1376: &str = "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, m, f32] = expand(x, 1, shape(x, 0))\n\
def main() = f(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32, 5.0f32]))\n";

#[test]
fn issue_1376_same_tensor_read_under_a_foreign_claim_traps_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "r1376_c", REPRO_1376);
    assert!(!ok, "m = 3 claimed for an axis of extent 2: {out}");
    assert!(out.contains(&domain_trap_line("expand")), "{out}");
}

/// chelis#1377, baseline `lane_divergent` and silent on the ROOTED path: a
/// literal claim of 4 over a read yielding 5. The class has one
/// `InputAxis`-sourced member, so its guard is local at the `Expand` site,
/// which the inlined kernel emits: this row fails if the guard reaches only
/// the exported kernel.
const REPRO_1377: &str = "def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[4, f32] = expand(b, 0, shape(x, 0))\n\
def main() = f(sum(to_tensor([1.0f32]), 0), to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]))\n";

#[test]
fn issue_1377_literal_claim_traps_at_the_inlined_root_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "r1377_c", REPRO_1377);
    assert!(!ok, "a declared tensor[4] over a read of 5: {out}");
    assert!(out.contains(&domain_trap_line("expand")), "{out}");
}

/// chelis#665, baseline `ice` in this modern spelling. The reproducer on the
/// issue predates the [05-DIM] int64 extent migration and no longer type
/// checks, so a reader re-running the filed one sees a type error and could
/// wrongly conclude the ICE is fixed. Here an op-declared axis on the `stride`
/// input must flow through the `expand`'s KEPT output axis with an index
/// shift, which is C4.2's case exactly.
const REPRO_665: &str = "module Repro.ExpandOverStride\n\
sig f: tensor[n, f32] -> tensor[m, u, f32]\n\
def f(x) = expand(stride(x, 2i64), cast(0, int32), shape(x, cast(0, int32)))\n\
out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))\n";

#[test]
fn issue_665_expand_over_stride_builds_and_runs() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "r665_c", REPRO_665);
    assert!(ok, "the kept axis has a source and must emit: {out}");
}

/// chelis#597. The checker is right and lowering is wrong:
/// `fallback_expand_type` has no replacement branch at all, so a checked
/// same-rank replacement always lowers as an insertion. The mixed-rank
/// comparison makes the divergence observable, because the checker stamps
/// `a : tensor[3, 2]` by INSERTION and `b : tensor[3, 2]` by REPLACEMENT,
/// one shape as [05-OP-36] requires, while lowering turns `b` into [3, 1, 2].
const REPRO_597: &str = "x1 = to_tensor([1.0f32, 2.0f32])\n\
y2 = to_tensor([[1.0f32, 2.0f32]])\n\
a = expand(x1, 0, 3i64)\n\
b = expand(y2, 0, 3i64)\n\
out = cmplt(a, b)\n";

#[test]
fn issue_597_positional_same_rank_replacement_executes_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "r597_c", REPRO_597);
    assert!(
        ok,
        "lowering must not insert an axis the checker replaced: {out}"
    );
}

/// The same-rank set form is well formed only over a UNIT source extent.
///
/// Selecting the same-rank form IS a claim that the operand's extent at `axis`
/// is 1, so it is an equality guard on a claimed extent and section 4.7's
/// existing rule places and names it - it is NOT a guard "under `expand`".
/// The operand's axis extent and the literal 1 are both interface values when
/// the operand is an input, so the guard runs at entry with `<op>` = `load`;
/// an op-produced operand extent gets a local guard at its producer instead
/// (`expand(stride(x, 2i64), 0i32, 3i64)` traps `domain in stride`). It is
/// implemented as a `DimClaim::Literal(1)` witness on the operand's axis fed
/// into the SAME class derivation, never a separate check at the `Expand`
/// site, so there is one derivation point per C2.7.
///
/// Section 4.7 splits this into TWO rows, and the split is not cosmetic:
/// "A violation proven from literals is a type error. A constraint that
/// depends on runtime values is checked before allocation or element access
/// and traps `Domain`" (`spec/04-type-system.md:1404-1407`). A
/// literal-extent operand disproves `extent == 1` at compile time, so it is a
/// TYPE ERROR and never reaches a guard; only a runtime extent traps.
///
/// MEASURED BASELINE for all three rows: `check` reports 0 errors and both
/// eval and compiled C produce `shape=[2, 6]` under a declared rank-1
/// `tensor[2, f32]`. Silent on both lanes, for both the literal-operand and
/// the symbolic-operand spelling.

/// A RUNTIME non-unit source: the operand's extent is symbolic, so the claim
/// is checked at entry and traps.
const REPRO_RUNTIME_NON_UNIT: &str = "module P.SymSrc\nsig f: tensor[n, f32] -> tensor[2, f32]\ndef f(x) = expand(x, 0, 2i64)\nout = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))\n";

/// A STATIC non-unit source: the operand's extent is the literal 6, so
/// `extent == 1` is disproved from literals and the program is rejected.
const REPRO_STATIC_NON_UNIT: &str = "module P.NonUnit\nsig f: tensor[6, f32] -> tensor[2, f32]\ndef f(x) = expand(x, 0, 2i64)\nout = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))\n";

/// A unit source extent is the well-formed case and must keep executing.
const REPRO_UNIT_SOURCE: &str = "module P.Unit\nsig f: tensor[1, f32] -> tensor[4, f32]\ndef f(x) = expand(x, 0, 4i64)\nout = f(to_tensor([7.0f32]))\n";

#[test]
fn a_runtime_non_unit_source_under_a_same_rank_claim_traps_at_entry_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "rt_nonunit_c", REPRO_RUNTIME_NON_UNIT);
    assert!(
        !ok,
        "a runtime extent of 6 cannot satisfy the unit claim: {out}"
    );
    assert!(
        out.contains(&domain_trap_line("load")),
        "the operand axis and the literal 1 are both interface values, so \
         section 4.7 runs this at entry and names the later witness's `load`: {out}"
    );
}

/// The static twin. Not a trap: section 4.7 makes a violation proven from
/// literals a TYPE ERROR, so this must never reach a guard at all.
#[test]
fn a_static_non_unit_source_under_a_same_rank_claim_is_a_type_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = build_result(&dir, "static_nonunit", REPRO_STATIC_NON_UNIT);
    assert!(!ok, "a literal extent of 6 disproves the unit claim: {out}");
    assert!(
        !out.contains("numeric trap"),
        "a literal violation is rejected, not trapped: {out}"
    );
}

/// The positive twin: without it, a guard that rejected every same-rank set
/// form would satisfy both rows above and delete the feature b2.5 exists to
/// make executable.
#[test]
fn a_unit_source_under_a_same_rank_claim_executes_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "unit_c", REPRO_UNIT_SOURCE);
    assert!(ok, "a unit source is the well-formed set form: {out}");
    assert!(out.contains("shape=[4]"), "{out}");
}
