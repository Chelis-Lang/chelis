//! Public CLI acceptance rows for runtime-extent Slice B (chelis#1277).
//!
//! `crates/chelis-ir/tests/runtime_extent_slice_b_classes.rs` locks the
//! DERIVATION: which output axes form one equality class, and in what order.
//! This file locks the CLI-observable consequence: that a class's guard is
//! ordered against an independent trap the way `spec/04-type-system.md`
//! section 4.7 requires, and that chelis#1482's missing extent source is a
//! typed receipt rather than an ICE.
//!
//! ## Why the guard rows themselves are not here
//!
//! A CLI-rooted program is not the exported kernel. `def main() = f(...)`
//! over literal tensors inlines `f` into the root, so every extent becomes a
//! literal, the classes disappear, and a violation is a static type error
//! nobody raises rather than a runtime guard anything can observe; a
//! top-level binding whose def computes no tensor emits no kernel at all.
//! The guard placement and rendering rows are therefore DRIVEN rows in
//! `crates/chelis-backend-c/tests/exec_compile.rs`, which compiles the
//! exported kernel and calls it with runtime inputs. That is the same shape
//! as the eval-lane finding recorded in `spec/design/runtime_extents.md`:
//! the lane that can observe the guard is not the lane a `.ch` fixture
//! reaches.
//!
//! The two rows that DO belong here are the order controls, because ordering
//! a guard against an in-body trap needs a caller, and a called function's
//! entry sits exactly where its call sits in the caller's source order.
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

/// chelis#616 left the HIP prologue walking `symbolic_occurrences` and
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
/// EVIDENTIARY STATUS: regression test. Measured red by restoring the HIP
/// emitter's two call sites to `symbolic_bindings()`, which reproduces the
/// panic on this exact program.
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

/// The same example on the C lane, for the LOCAL half of section 4.7.
///
/// `seq` is one class with sixteen members: `x`'s own axis, seven extents
/// operations compute, and eight folded reads of computed tensors. The last
/// group is the guard set, and on `main` only FOUR of the eight are guarded,
/// because chelis#616's `runtime_dim_sites` finds sites by walking
/// occurrences for an op-declared name rather than by asking which members a
/// class has. The derivation finds all eight, and each renders [04-NUM-9]
/// instead of `chelis: runtime dim `seq` mismatch at node N axis A` followed
/// by `abort()`.
///
/// The eight is this example's measured count, not a property of the rule; an
/// edit to the example is expected to change it, and a reader updating it
/// should re-derive rather than relax the assertion, because the count is the
/// only thing here that distinguishes finding every member from finding the
/// four the walk already found.
///
/// EVIDENTIARY STATUS: regression test on both halves - four guards and the
/// legacy rendering on `main`, eight and [04-NUM-9] here.
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
    assert_eq!(
        emitted.matches("extent `seq`: claimed = ").count(),
        8,
        "every folded read of a computed tensor under this claim is guarded"
    );
    assert_eq!(
        emitted
            .matches(&format!("{}\");", domain_trap_line("expand")))
            .count(),
        8,
        "and each renders [04-NUM-9] naming the operation"
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

/// guard_order.effect_before.eval: an effect that precedes the call in source
/// order is observed before the guard traps. NOT provable on eval today, and
/// the row stays at its baseline: `chelis eval` emits a program's printed
/// output only when the evaluation succeeds, so an effect that precedes any
/// failure is discarded (`_ = print("hello")` followed by a division trap
/// prints nothing, with no runtime-extent guard involved; chelis#1585), while
/// the compiled program prints it and then traps. What this test locks is the half the
/// eval lane can show: the guard fires from inside an `IO` body, which the
/// shared kernel decision keeps in host code on both lanes.
///
/// EVIDENTIARY STATUS: regression test for the trap (silent `shape=[5]` on
/// the tree without the evaluator's literal-extent check); the effect's
/// order is unobservable on this lane and is not asserted.
#[test]
fn an_effect_before_the_guard_runs_when_the_guard_traps_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "effect_before.ch",
        &effect_order_source(MISMATCHED, true),
    );
    assert!(!ok, "the program must fail: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `4`: claimed = 4, x axis 0 = 5"),
        "{out}"
    );
}

/// guard_order.effect_after.eval: an effect that follows the call is not
/// observed when the guard traps. On eval the absence of "effect" cannot be
/// read as ORDER (see the row above: nothing printed before a failure is
/// emitted either, chelis#1585), so the row stays at its baseline and this test locks
/// the trap alone.
///
/// EVIDENTIARY STATUS: regression test for the trap (on the tree without the
/// evaluator's literal-extent check the program succeeded and printed
/// "effect"); the effect's absence is not evidence of order on this lane.
#[test]
fn an_effect_after_the_guard_does_not_run_when_the_guard_traps_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(
        &dir,
        "effect_after.ch",
        &effect_order_source(MISMATCHED, false),
    );
    assert!(!ok, "the program must fail: {out}");
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

/// The statement order the two C effect rows assert, read from the emitted C
/// because the bytes a `print` writes before a trap do not survive
/// `chelis_numeric_trap`'s `abort()` on a buffered stdout (chelis#1591):
/// inside `run`'s host body the `print` statement and the call into the
/// kernel C extracts for `f(seed, x)` (`run__tensor_N`, with `f` inlined and
/// the same entry guard `f__tensor_0` carries), in the order `effect_first`
/// names; inside that kernel the entry guard before its first allocation.
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
    let body_start = emitted
        .find("run__chelis_owned_body(chelis_tensor* x) {")
        .expect("the IO body is emitted as host code");
    let body = &emitted[body_start..];
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

/// guard_order.effect_before.c: the row that returns with chelis#1528. On
/// `main` the C lane dropped the `IO` effect of a kernel-lowered body, so the
/// string never appeared in the emitted program and no row could order it.
/// The order is asserted from the emitted C's statement order (chelis#1591:
/// the printed bytes are lost on a buffered stdout when the trap aborts, so
/// they cannot carry the assertion on Linux), and the trap by execution.
///
/// EVIDENTIARY STATUS: regression test against `main` for the print's
/// presence and position (absent there); disposition lock for the guard's
/// position and the trap.
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
    assert_effect_order_in_emitted_c(&emitted, true);
}

/// guard_order.effect_after.c: the print follows the call into `f` in the
/// emitted host body, so the trap in `f`'s prologue precedes it; asserted
/// from the emitted C's statement order for the reason above (chelis#1591),
/// and the trap by execution.
///
/// EVIDENTIARY STATUS: regression test against `main` for the print's
/// presence and position (absent there); disposition lock for the trap.
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
    assert_effect_order_in_emitted_c(&emitted, false);
}

/// class.load_load.eval: an all-interface class of three witnesses on one
/// binder; the def reads all three, so each is a kernel input and each later
/// witness is guarded against the canonical one at entry. `f` is a kernel on
/// both lanes; a bare-variable body would be host code on both and would
/// carry no guard on either. The canonical member is the derivation's, not
/// the signature's first parameter: the compiled kernel for this program
/// renders `p` as canonical, and the eval lane renders the identical lines
/// because it reads the same derivation (C2.7).
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
        out.contains("extent `zdim`: p axis 0 = 2, r axis 0 = 3"),
        "the later witness is compared against the canonical one, as the compiled kernel renders it: {out}"
    );
    let zz_disagrees = format!(
        "{SOURCE}out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32]))\n"
    );
    let (ok, out) = eval_result(&dir, "zz_disagrees.ch", &zz_disagrees);
    assert!(!ok, "the witness `zz` disagrees: {out}");
    assert!(
        out.contains("extent `zdim`: p axis 0 = 2, zz axis 0 = 3"),
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
/// tensor element. The [05-UNS-1] envelope around it is the host lane's
/// existing root-realization wrapper (chelis#912) and is not this row's
/// subject.
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
        stderr.contains(&domain_trap_line("expand")),
        "the trap line is the [04-NUM-9] rendering verbatim, at the extent's \
         own dtype: {stderr}"
    );
    assert!(
        stderr.contains("axis 0 is 1") && stderr.contains("observed 2"),
        "the accompanying context names the axis and the value observed, which \
         §4.7 requires on separate lines from the trap: {stderr}"
    );
}
