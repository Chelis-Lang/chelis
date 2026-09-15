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
//! remove the control's teeth. The extent under test is an input tensor's
//! axis, so its guard runs at `f`'s ENTRY, and the control still discriminates
//! because a called function's entry sits exactly where its call sits in the
//! caller's source order. `guard_order_source`'s own comment records the
//! measurement behind that placement; the emitted C is its oracle. An earlier
//! version of this paragraph called the extent locally computed and cited
//! chelis#1379's arithmetic form as the example, which described neither the
//! fixture nor the placement.
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

/// Build to C, link, run, and return the binary's exit STATUS plus its combined
/// output. A build or link failure panics: those are defects in the fixture or
/// the emitter, never the behaviour under test.
///
/// The status rather than a boolean, because the runtime's two failure shapes
/// are different exits and the difference is observable: a `Domain` rejection
/// from `affine_result` exits 1, while `chelis_numeric_trap` reached through a
/// guard aborts and exits 134.
fn c_run_status(dir: &TempDir, stem: &str, source: &str) -> (std::process::ExitStatus, String) {
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
    (run.status, text)
}

/// The same run reduced to whether the binary exited zero, which is what every
/// row that only needs "it failed" reads.
fn c_run_result(dir: &TempDir, stem: &str, source: &str) -> (bool, String) {
    let (status, text) = c_run_status(dir, stem, source);
    (status.success(), text)
}

/// The `span_id` a fatal lowering rejection carries for a root spelled
/// `def main() -> T = <call>`: the call expression, which is the last thing
/// such a source says.
///
/// Computed from the fixture rather than pasted in, so editing a program
/// cannot leave a stale byte range asserted somewhere that still passes.
/// Only valid for the single-call-root shape; a rejection raised from a
/// NESTED activation carries that inner call's span instead.
fn root_call_span(source: &str) -> String {
    let body = source.trim_end();
    let start = body.rfind("= ").expect("a root body follows `= `") + 2;
    format!("surf:{start}..{}", body.len())
}

/// Both host lanes must reject `source` when the activation is LOWERED,
/// before anything executes, printing exactly `line` and nothing else.
///
/// Four properties, and each one is a way the rejection could be wrong rather
/// than a restatement of the others. Exit 1 rather than 101 separates a
/// deliberate diagnostic from a panic escaping the lowering catch. The exact
/// equality, rather than a `contains`, is what pins the rendering. The absent
/// [04-NUM-9] line is `spec/04-type-system.md` section 4.7's own division: a
/// claim proven wrong is a type error and owes no runtime trap, so a trap
/// beside this diagnostic would mean the guard had also been recorded. And
/// the two lanes are compared to EACH OTHER, so a future change that moves
/// one rendering cannot pass by moving the asserted constant with it.
fn both_lanes_reject_at_lowering(dir: &TempDir, stem: &str, source: &str, line: &str) {
    let eval_run = eval(&fixture(dir, &format!("{stem}.ch"), source));
    let eval_out = format!(
        "{}{}",
        String::from_utf8_lossy(&eval_run.stdout),
        String::from_utf8_lossy(&eval_run.stderr)
    );
    let build = build_c(
        &fixture(dir, &format!("{stem}_c.ch"), source),
        &dir.path().join(format!("{stem}_c-out")),
    );
    let c_out = format!(
        "{}{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    for (lane, code, out) in [
        ("eval", eval_run.status.code(), &eval_out),
        ("c", build.status.code(), &c_out),
    ] {
        assert_eq!(
            code,
            Some(1),
            "{lane}: a rejected claim exits 1, where an escaping panic exits 101: {out}"
        );
        assert_eq!(
            out.trim_end(),
            line,
            "{lane}: the exact rendering, with nothing beside it"
        );
        assert!(
            !out.contains("numeric trap"),
            "{lane}: a claim proven wrong owes no [04-NUM-9] trap: {out}"
        );
    }
    assert_eq!(
        eval_out, c_out,
        "the two lanes render this rejection byte-identically"
    );
}

/// The agreeing control every rejection row owes: the same program with a
/// claim the body satisfies executes on both lanes and prints `printed`.
///
/// Without it a row pins "this program fails" rather than "this CLAIM fails",
/// and a lowering that refused the whole shape would satisfy it.
fn both_lanes_execute(dir: &TempDir, stem: &str, source: &str, printed: &str) {
    let (eval_ok, eval_out) = eval_result(dir, &format!("{stem}.ch"), source);
    let (c_ok, c_out) = c_run_result(dir, &format!("{stem}_c"), source);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: an agreeing claim executes: {out}");
        assert!(
            out.contains(printed),
            "{lane}: expected `{printed}` in: {out}"
        );
    }
}

#[test]
fn vmap_ordinary_shape_value_executes_on_both_lanes() {
    let dir = TempDir::new().unwrap();
    for (name, result_type, body, printed) in [
        (
            "read",
            "int64",
            "shape(x, 0)",
            "tensor(shape=[2], data=[3, 3])",
        ),
        (
            "cast",
            "f32",
            "x |> shape(0) |> cast(f32)",
            "tensor(shape=[2], data=[3.0, 3.0])",
        ),
    ] {
        for (axis, data) in [
            (0, "[[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]"),
            (1, "[[1.0f32, 4.0f32], [2.0f32, 5.0f32], [3.0f32, 6.0f32]]"),
        ] {
            let axis_argument = if axis == 0 { "" } else { ", axis=1" };
            let source = format!(
                "def width(x: tensor[3, f32]) -> {result_type} = {body}\nout = vmap(width{axis_argument})(to_tensor({data}))\n"
            );
            both_lanes_execute(
                &dir,
                &format!("shape_value_{name}_{axis}"),
                &source,
                printed,
            );
        }
    }
}

#[test]
fn vmap_ordinary_shape_root_transport_preserves_repeated_result_leaves() {
    let dir = TempDir::new().unwrap();
    let source = "def pair(x: tensor[3, f32]) -> (tensor[3, f32], tensor[3, f32], tensor[3, f32]) = {\n  y = x\n  z = neg(y)\n  (y, z, y)\n}\nout = vmap(pair)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n";
    let path = fixture(&dir, "repeated_vmap_roots.ch", source);
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let roots = result["roots"].as_array().unwrap();
    assert_eq!(roots.len(), 3);
    for (index, bits) in [
        [
            "3f800000", "40000000", "40400000", "40800000", "40a00000", "40c00000",
        ],
        [
            "bf800000", "c0000000", "c0400000", "c0800000", "c0a00000", "c0c00000",
        ],
        [
            "3f800000", "40000000", "40400000", "40800000", "40a00000", "40c00000",
        ],
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(roots[index]["name"], format!("out.{index}"));
        assert_eq!(
            roots[index]["value"]["value"],
            serde_json::json!({
                "shape": [2, 3], "data": {"dtype": "f32", "bits": bits}
            })
        );
    }
}

#[test]
fn vmap_ordinary_shape_preserves_the_nested_gradient_claim() {
    let dir = TempDir::new().unwrap();
    for extent in [2, 3] {
        let row = if extent == 2 {
            "[2.0f32, 7.0f32]"
        } else {
            "[2.0f32, 7.0f32, 11.0f32]"
        };
        let source = format!(
            "def claim(x: tensor[n, f32]) -> tensor[2, f32] = shrink(x, [[0i64, shape(x, 0)]])\ndef loss(x: tensor[{extent}, f32]) -> f32 = cast(shape(claim(x), 0), f32)\nout = vmap(grad(loss))(to_tensor([{row}, {row}]))\n"
        );
        let path = fixture(&dir, &format!("nested_shape_{extent}.ch"), &source);
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .env_remove("CHELIS_STYLE_GATE_DISABLE")
            .args([
                "eval",
                "--file",
                path.to_str().unwrap(),
                "--json",
                "--timeout",
                "20",
            ])
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if extent == 2 {
            assert!(output.status.success(), "{stderr}");
            let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(result["roots"].as_array().unwrap().len(), 1);
            assert_eq!(
                result["roots"][0]["value"]["value"],
                serde_json::json!({
                    "shape": [2, 2],
                    "data": {"dtype": "f32", "bits": ["00000000", "00000000", "00000000", "00000000"]}
                })
            );
        } else {
            assert!(!output.status.success());
            assert!(stderr.contains("claimed = 2"), "{stderr}");
            assert!(stderr.contains("shrink axis 1 = 3"), "{stderr}");
            assert!(
                stderr.contains("numeric trap: domain in shrink at int64"),
                "{stderr}"
            );
            assert!(!stderr.contains("extent source(s)"), "{stderr}");
        }
    }
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
/// the guards went. chelis#1379's arithmetic form was unusable here for a
/// different reason, that the C lane refused to build `mul(shape(x, 0), 2i64)`
/// at all under chelis#469, and a control that fails at build time is not
/// measuring order. That rejection is gone and the form now lands a local
/// guard at its own operation, so it would serve; this fixture keeps the
/// shape-read spelling because that is the form its measurements were taken
/// on.
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
        emitted.contains("chelis_device_metadata seq = chelis_tensor_shape("),
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

/// The byte index, in `text`, of the `)` that closes the first `(` in it.
///
/// Counting depth rather than taking the first `)` is what lets
/// `host_body_definition` read a parameter list whose spelling nests
/// parentheses. Parentheses are ASCII, so the returned byte index is always a
/// char boundary.
fn closing_paren(text: &str) -> Option<usize> {
    let open = text.find('(')?;
    let mut depth = 0usize;
    for (offset, byte) in text.bytes().enumerate().skip(open) {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(offset);
                }
            }
            _ => {}
        }
    }
    None
}

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
///
/// Two ways it could still misread the emission, both closed here after the
/// #1792 round reported them. A match with no left word boundary accepts
/// `g_run__chelis_owned_body(` as `run__chelis_owned_body`, and the wrong
/// function's body then satisfies the order assertions for the wrong reason.
/// And taking the first `)` as the end of the parameter list mistakes a
/// nested parenthesis for the end of the signature, so the `{` test fails and
/// this helper reports the body as having left the host lane, which is
/// #1808's own misdiagnosis in a new spelling. `the_host_body_locator_*`
/// tests hold both.
fn host_body_definition<'a>(emitted: &'a str, name: &str) -> &'a str {
    let needle = format!("{name}(");
    let mut at = 0;
    while let Some(found) = emitted[at..].find(&needle) {
        let start = at + found;
        at = start + needle.len();
        let preceded_by_identifier = emitted[..start]
            .chars()
            .next_back()
            .is_some_and(|previous| previous.is_alphanumeric() || previous == '_');
        if preceded_by_identifier {
            continue;
        }
        let rest = &emitted[start..];
        if let Some(close) = closing_paren(rest)
            && rest[close + 1..].trim_start().starts_with('{')
        {
            return rest;
        }
    }
    panic!(
        "no definition of `{name}` in the emitted C: the IO body is expected on the host lane, \
         and a forward declaration alone means it moved off it"
    );
}

/// Disposition lock for the locator above, on synthetic C rather than on an
/// emission: a suffix match and a nested parenthesis are the two ways it can
/// name the wrong body, and neither is reachable from today's emitter, so an
/// emitted-code test could not hold them. Measured red on the pre-fold
/// helper: the first case returned the decoy's body and the second panicked
/// with "it moved off it".
#[test]
fn the_host_body_locator_reads_the_definition_and_not_a_look_alike() {
    let decoy = concat!(
        "void g_run__chelis_owned_body(int a) { decoy; }\n",
        "void run__chelis_owned_body(int a);\n",
        "void run__chelis_owned_body(int a) { real; }\n",
    );
    assert!(
        host_body_definition(decoy, "run__chelis_owned_body")
            .starts_with("run__chelis_owned_body(int a) { real;"),
        "a symbol ENDING in the wanted name is a different function"
    );

    let nested = "void run__chelis_owned_body(void (*cb)(int), int a) { real; }\n";
    assert!(
        host_body_definition(nested, "run__chelis_owned_body").ends_with("{ real; }\n"),
        "a parameter list that nests parentheses is still a definition"
    );
}

/// Negative parity for the locator: a forward declaration with no definition
/// is the emission actually leaving the host lane, and must still panic.
///
/// Caught rather than declared `#[should_panic]`, because the oracle's
/// `cli_slice_b` row selects this file whole and compares libtest's printed
/// names against the reviewed manifest. libtest prints a `should_panic` test
/// as `<name> - should panic`, which the manifest cannot carry without
/// disagreeing with `runtime_extent_target_manifest.rs`, the reader that
/// takes the same names out of this source.
#[test]
fn the_host_body_locator_refuses_a_forward_declaration_alone() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(|| {
        host_body_definition(
            "void run__chelis_owned_body(int a);\n",
            "run__chelis_owned_body",
        )
    });
    std::panic::set_hook(previous);
    let payload = outcome.expect_err("a declaration with no definition must panic");
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .expect("panic payload is a string");
    assert!(
        message.contains("no definition of `run__chelis_owned_body`")
            && message.contains("means it moved off it"),
        "the panic must name the symbol and the lane it left: {message}"
    );
}

/// Supplement the executed output checks with the emitted statement order:
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

/// claim.named.nested_fresh_result.{eval,c}: a NAMED result claim is enforced
/// when the declaring def is called from inside another def's body.
///
/// Two sites lost it, and the repair needed both. `host::remap_tensor_helper_
/// dim_symbols` pairs the root's output type with the enclosing declared
/// result as an extra formal, and the graph-wide substitution that pairing
/// produced renamed the inner activation's `n` to `g`'s fresh `k`, orphaning
/// the class. Even unrenamed, the class keyed by `n` had ONE member: the
/// declaring axis is the const `to_tensor([1.0f32, 2.0f32])`, which carries
/// `Lit(2)` and so belongs to the `Literal(2)` class instead, and C2.4 did not
/// make a one-member `Name` class a class. So the pairing now substitutes only
/// compiler-minted names, and lowering stamps the inner claim RESOLVED, as
/// `Named("n", Some(2))`, which `derive_runtime_dim_classes` admits as a
/// complete single-member class.
///
/// Read off the emitted C before the repair: `f`'s own kernel carried
/// `int64_t n = chelis_tensor_shape(inputs[0], 0)` and the comparison, while
/// `g`'s kernel, with `f` inlined, carried
/// `int64_t k = chelis_movement_extent(...)` and none.
///
/// The LITERAL half survives the same nesting and did so before this change,
/// which is why this row still asserts both: the two halves of the
/// declared-result contract diverged exactly here, and a receipt that dropped
/// the literal side would stop showing that they now agree.
///
/// This row was `a_nested_named_result_claim_is_not_enforced_by_this_slice`, a
/// disposition lock, until chelis#1800 closed it.
///
/// EVIDENTIARY STATUS: regression test for the named block. Measured at
/// `a5fee66b9`, this branch's base, this program printed
/// `out = tensor(shape=[3], data=[2.0, 3.0, 4.0])` and exited ZERO on eval and
/// on the linked C binary under a declared `tensor[n, f32]` with `n` = 2.
/// Regression test for the literal block too, which this slice delivered
/// through the same nesting earlier. Disposition lock for the agreeing
/// control, the non-vacuity check.
#[test]
fn a_nested_named_result_claim_is_enforced_through_its_resolved_binder() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    // The named half: guarded, reporting the inner binder and the number the
    // graph-fixed declaring argument resolved it to.
    let named = nested_named_result_source(4, "n");
    let (eval_ok, eval_out) = eval_result(&dir, "nested_named.ch", &named);
    let (c_ok, c_out) = c_run_result(&dir, "nested_named_c", &named);
    let context = "extent `n`: claimed = 2, shrink axis 0 = 3";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: the nested named claim traps: {out}");
        assert!(
            out.contains(&domain_trap_line("shrink")),
            "{lane}: [04-NUM-9]'s line: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
        assert!(
            !out.contains("shape=[3]"),
            "{lane}: and the extent the shrink computed is not returned: {out}"
        );
        assert!(
            !out.contains("extent `k`"),
            "{lane}: the enclosing fresh result name does not replace the \
             inner binder: {out}"
        );
    }

    // The literal half through the SAME nesting: guarded, as before.
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

    // And the named half's agreeing control still executes exactly, on both
    // lanes, so the first block is pinning a guard and not a broken lane.
    let agreeing = nested_named_result_source(3, "n");
    let (eval_ok, eval_out) = eval_result(&dir, "nested_named_ok.ch", &agreeing);
    let (c_ok, c_out) = c_run_result(&dir, "nested_named_ok_c", &agreeing);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: an agreeing nested claim executes: {out}");
        assert!(
            out.contains("shape=[2]") && out.contains("data=[2.0, 3.0]"),
            "{lane}: with the declared extent: {out}"
        );
    }
}

/// claim.named.nested_runtime_declarer: a nested named claim whose declaring
/// argument carries a RUNTIME extent is a checker rejection, which is why the
/// resolved stamp alone closes chelis#1800.
///
/// The stamp resolves a claim only when its declaring witness observes a
/// graph-fixed extent. An UNRESOLVED inner claim would still be renamed by the
/// root pairing in `host::remap_tensor_helper_dim_symbols`, because
/// `tensor_dim_substitutions` keys on `DimInfo::Named(name, None)` and a
/// resolved `Named(name, Some(v))` is not a key at all. So the question is
/// whether an unresolved nested claim is reachable, and it is not: passing
/// `g`'s own parameter as the argument that declares `n` unifies two rigid
/// dim parameters, which §4.4.1 refuses.
///
/// Without this row the repair looks incomplete, and a reader would reach for
/// the pairing change the design proposed. That change is measurably wrong:
/// masking user-spelled names out of the pairing removed the guard from seven
/// existing receipts in this file, among them
/// `a_literal_claim_over_a_runtime_read_traps_at_entry_on_eval`, where a
/// declared `tensor[4, f32]` over a read of 5 executed and printed
/// `widened = tensor(shape=[5], ...)`. The pairing carries legitimate
/// user-spelled declarations as well as synthesized ones.
///
/// EVIDENTIARY STATUS: disposition lock. Existing checker behaviour that this
/// change neither introduces nor alters; the row exists so the resolved
/// stamp's sufficiency is recorded rather than assumed.
#[test]
fn a_nested_claim_whose_declarer_is_a_runtime_extent_is_refused_by_the_checker() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(w: tensor[n, f32], x: tensor[r, f32]) -> tensor[n, f32] = \
                  shrink(x, [[1i64, shape(x, 0i32)]])\n\
                  def g(y: tensor[s, f32], z: tensor[t, f32]) -> tensor[k, f32] = f(z, y)\n\
                  out = g(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), \
                  to_tensor([1.0f32, 2.0f32]))\n";
    let (ok, out) = eval_result(&dir, "nested_runtime_declarer.ch", source);
    assert!(!ok, "two rigid dim parameters do not unify: {out}");
    assert!(
        out.contains("declared dim parameters are rigid and must remain distinct"),
        "and §4.4.1 is what refuses it: {out}"
    );
    assert!(
        !out.contains(&domain_trap_line("shrink")),
        "so no [04-NUM-9] runtime trap is rendered for it: {out}"
    );
}

/// claim.named.nested_param_result.{eval,c}: the spelling that binds the
/// enclosing result to the enclosing PARAMETER is a checker rejection, not a
/// guard.
///
/// chelis#1800's issue text gestures at `-> tensor[m]` beside the fresh `k`
/// above, and the two are not the same case. A dim parameter the signature
/// declares must stay polymorphic under `spec/04-type-system.md` §4.4.1, and
/// this body pins it: `f`'s declared `n` comes from a const argument, so `m`
/// is forced to 2. The checker refuses that before any lane runs, which is why
/// the row above uses a FRESH enclosing name; a fresh name is the only
/// spelling where the residual existed.
///
/// EVIDENTIARY STATUS: disposition lock, and a correction to the design that
/// preceded this change, which recorded this spelling as already trapping
/// ``extent `m` `` at run time. Measured at `a5fee66b9`, it is a
/// `DimensionMismatch` from the checker, with or without the repair: this
/// change touches `crates/chelis-ir` only, so it cannot reach the verdict.
#[test]
fn a_nested_param_bound_result_claim_is_refused_by_the_checker() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(w: tensor[n, f32], x: tensor[r, f32]) -> tensor[n, f32] = \
                  shrink(x, [[1i64, shape(x, 0i32)]])\n\
                  def g(y: tensor[m, f32]) -> tensor[m, f32] = \
                  f(to_tensor([1.0f32, 2.0f32]), y)\n\
                  out = g(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\n";
    let (ok, out) = eval_result(&dir, "nested_param_result.ch", source);
    assert!(!ok, "a pinned dim parameter does not execute: {out}");
    assert!(
        out.contains("polymorphic dim parameter `m` forced to concrete Lit(2)"),
        "and §4.4.1's rigid-parameter rule is what refuses it: {out}"
    );
    assert!(
        !out.contains(&domain_trap_line("shrink")),
        "so no [04-NUM-9] runtime trap is rendered for it: {out}"
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

/// The still-unadmitted op-computed owner, stated as a lock rather than left
/// to be discovered. `spec/04-type-system.md` section 4.7 makes a non-unit
/// `stride` step mint a fresh extent exactly as `shrink` does, and it remains
/// unguarded: unchanged rather than newly silent. chelis#1379 owns the
/// arithmetic-sized forms.
///
/// This row was `pad_and_stride_op_computed_extents_remain_unadmitted` until
/// chelis#1837 admitted `pad`. Its pad half moved to
/// `a_non_zero_pad_extent_is_guarded_on_both_lanes` below, as a receipt rather
/// than a lock; the two must not share one test, because a lock and a receipt
/// make opposite claims about whether the behaviour may change.
///
/// The disposition is LANE-DIVERGENT, and this row records both halves rather
/// than the evaluator's alone. An unadmitted owner has no guard site, so the
/// claim is dropped; the evaluator then returns the undeclared extent at exit
/// zero, while the C binary aborts at `chelis_movement_check_target`, the
/// movement plan's own target comparison, which names the allocation rather
/// than the claim and carries no [04-NUM-9] context line. Tracked by
/// chelis#1931, which must update this row when `stride` is admitted.
///
/// An earlier version of this row called `eval_result` only and asserted `ok`
/// under "an unadmitted op-computed owner is unchanged by this change, not
/// newly trapping", which reads as "both lanes execute" when one rejects. The
/// pad row beside it says the combined lock it was split from "ran eval only";
/// this is the other half of that same observation.
///
/// EVIDENTIARY STATUS: disposition lock on both lanes, not a regression test.
/// Measured at `f45a7848a`: eval exit 0 with
/// `out = tensor(shape=[3], data=[1.0, 3.0, 5.0])` under a declared
/// `tensor[2, f32]`, and the linked binary exit 1 with
/// `Domain: movement target shape mismatch` before
/// `numeric trap: domain in stride at int64`. This change alters neither.
#[test]
fn a_non_unit_stride_op_computed_extent_remains_unadmitted() {
    assert!(
        gcc_available(),
        "this row records a divergence, so neither lane may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[rows, f32]) -> tensor[2, f32] = stride(x, 2i64)\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))\n";

    // The evaluator half: silently wrong, which is the defect chelis#1931
    // tracks rather than a behaviour to preserve.
    let (ok, out) = eval_result(&dir, "stride_unadmitted.ch", source);
    assert!(ok, "eval returns the undeclared extent at exit zero: {out}");
    assert!(
        out.contains("out = tensor(shape=[3], data=[1.0, 3.0, 5.0])"),
        "eval: and the extent is the stride's, not the declared 2: {out}"
    );
    assert!(
        !out.contains("extent `2`"),
        "eval: with no guard claiming to have checked it: {out}"
    );

    // The C half: it rejects, but at the movement plan's target check rather
    // than the claim's guard, so it names neither extent.
    let (ok, out) = c_run_result(&dir, "stride_unadmitted_c", source);
    assert!(!ok, "the C binary rejects: {out}");
    assert!(
        out.contains("Domain: movement target shape mismatch")
            && out.contains(&domain_trap_line("stride")),
        "c: reporting the allocation rather than the claim: {out}"
    );
    assert!(
        !out.contains("extent `2`: claimed = 2, stride axis 0 = 3"),
        "c: an admitted owner's rendering would replace this one: {out}"
    );
}

/// A declared literal over a non-zero `pad` with LITERAL bounds is guarded at
/// the `pad`, on both lanes and with the same rendering.
///
/// This is the direct spelling of the extent chelis#1837 reaches through
/// `concat`'s lowered cascade, and it is here because admitting `pad` has to
/// be visible in the owner's own spelling rather than only through a composed
/// operation. It also closes a lane divergence the old lock did not measure,
/// because the lock ran eval only.
///
/// This row is the LITERAL-BOUND control. Its `[[1i64, 1i64]]` bounds are
/// compile-time constants and the extent is non-constant only because the
/// OPERAND is symbolic, so it does not measure a runtime padding bound. The
/// three rows below do, and round 1 of this pull request's review found that
/// distinction hiding a defect: the claim's precondition and its evidence were
/// measuring different things.
///
/// EVIDENTIARY STATUS: regression test on BOTH lanes, for different base
/// behaviour on each. Measured at `a5fee66b9`: eval printed
/// `out = tensor(shape=[5], data=[0.0, 1.0, 2.0, 3.0, 0.0])` and exited ZERO
/// under a declared `tensor[2, f32]`, while the C binary exited 1 reporting
/// the generic `Domain: movement target shape mismatch` before
/// `numeric trap: domain in pad at int64` - a trap, but one naming the
/// allocation rather than the claim, and arriving at the movement plan's
/// target check rather than at the claim's guard.
#[test]
fn a_non_zero_pad_extent_is_guarded_on_both_lanes() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let source = |claim: u32| {
        format!(
            "def f(x: tensor[rows, f32]) -> tensor[{claim}, f32] = \
             pad(x, [[1i64, 1i64]], 0.0f32)\n\
             out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
        )
    };

    let mismatched = source(2);
    let (eval_ok, eval_out) = eval_result(&dir, "pad_admitted.ch", &mismatched);
    let (c_ok, c_out) = c_run_result(&dir, "pad_admitted_c", &mismatched);
    let context = "extent `2`: claimed = 2, pad axis 0 = 5";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: a claim of 2 over a pad to 5 traps: {out}");
        assert!(
            out.contains(&domain_trap_line("pad")),
            "{lane}: [04-NUM-9]'s line names the `pad`: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
        assert!(
            !out.contains("Domain: movement target shape mismatch"),
            "{lane}: and the claim's guard precedes the plan's target check: {out}"
        );
    }

    // The agreeing control: three elements padded by one on each side is five.
    let agreeing = source(5);
    let (eval_ok, eval_out) = eval_result(&dir, "pad_admitted_ok.ch", &agreeing);
    let (c_ok, c_out) = c_run_result(&dir, "pad_admitted_ok_c", &agreeing);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: an agreeing pad claim executes: {out}");
        assert!(
            out.contains("out = tensor(shape=[5], data=[0.0, 1.0, 2.0, 3.0, 0.0])"),
            "{lane}: with the padded extent and the fill: {out}"
        );
    }

    // A pad extent of ZERO, which is why `ComputedAxisExtent::PadSpan` carries
    // no `> 0` filter where `ShrinkSpan` does. A shrink span of zero computes
    // no extent and the operation's own domain rejection owns it; a pad extent
    // of zero is a real extent, so a claim of another number over it is a
    // mismatch the guard still owes. Round 1's verification measured this
    // reachable, correcting a comment that had called it unreachable, and a
    // filter added for parity would silence exactly this row.
    let zero = "def f(x: tensor[rows, f32], y: tensor[s, f32]) -> tensor[2, f32] = \
                pad(x, [[sub(shape(y, 0i32), shape(y, 0i32)), \
                sub(shape(y, 0i32), shape(y, 0i32))]], 0.0f32)\n\
                out = f(to_tensor([]), to_tensor([1.0f32, 2.0f32]))\n";
    let (eval_ok, eval_out) = eval_result(&dir, "pad_zero_extent.ch", zero);
    let (c_ok, c_out) = c_run_result(&dir, "pad_zero_extent_c", zero);
    let context = "extent `2`: claimed = 2, pad axis 0 = 0";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: a claim of 2 over a zero extent traps: {out}");
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
    }

    // And the agreeing zero, so the row above pins a guard rather than a
    // lowering that cannot produce an empty axis at all.
    let zero_ok = zero.replace("-> tensor[2, f32]", "-> tensor[0, f32]");
    let (eval_ok, eval_out) = eval_result(&dir, "pad_zero_ok.ch", &zero_ok);
    let (c_ok, c_out) = c_run_result(&dir, "pad_zero_ok_c", &zero_ok);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: an agreeing zero claim executes: {out}");
        assert!(
            out.contains("out = tensor(shape=[0], data=[])"),
            "{lane}: producing the empty axis: {out}"
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

/// chelis#1797's reproducer: a `shrink` whose runtime end overshoots the
/// operand's extent, under each result-claim spelling the issue's witnesses
/// take.
///
/// The operand holds four elements and the bound is
/// `[1, shape(x, 0) + 3) = [1, 7)`, so the end overshoots by three and the
/// span's arithmetic width is six in every spelling. `claim` is the declared
/// result extent, and the three values separate the three ways a claim can sit
/// over a span that is out of domain: `2` DISAGREES with the width, `6` AGREES
/// with it, and `k` is a free dim that claims nothing. Before the repair those
/// three took three different exits, which is why one of them is not enough.
fn overshooting_shrink_source(claim: &str) -> String {
    format!(
        "def f(x: tensor[rows, f32]) -> tensor[{claim}, f32] = \
         shrink(x, [[1i64, add(shape(x, 0i32), 3i64)]])\n\
         out = f({})\n",
        vector_literal(4)
    )
}

/// The three result-claim spellings of the overshoot reproducer.
const OVERSHOOT_CLAIMS: [&str; 3] = ["2", "6", "k"];

/// The two lines a lane prints for an out-of-domain `shrink` bound, and the
/// whole of what it prints.
///
/// `ShapeMetadata::shrunk` (`crates/chelis-abi/src/metadata.rs`) answers a
/// negative start, an inverted pair, or an end past the operand extent with one
/// `Domain` message; `affine_result` (`crates/chelis-runtime/src/lib.rs`)
/// prints it, then [04-NUM-9]'s trap line, then exits 1. `SHRINK_DOMAIN_TRAP`
/// in `crates/chelis-ir/src/eval.rs` is the evaluator's own spelling of the
/// same two lines.
///
/// The message therefore exists in two independent literals, and this constant
/// is a third. Sharing one is not available: `chelis-ir` depends on neither
/// `chelis-abi` nor `chelis-runtime`, and adding that edge to carry one string
/// would cost more than the drift it prevents. The mitigation is the pair of
/// receipts below, which run BOTH lanes as processes and compare each one's
/// whole output against this constant, so a change to either spelling fails a
/// test instead of quietly separating the lanes. A shared literal would be
/// weaker at exactly that: it would let both lanes drift together with nothing
/// left to notice.
const OVERSHOOT_RENDERING: &str =
    "Domain: shrink bounds outside input extent\nnumeric trap: domain in shrink at int64\n";

/// The same two lines under the eval lane's reporter, which prefixes `error: `
/// to the first line of every diagnostic it raises. That prefix is the only
/// difference between the lanes, and it is the same prefix the extent guard's
/// [04-NUM-9] line already carries on this lane.
fn overshoot_eval_rendering() -> String {
    format!("error: {OVERSHOOT_RENDERING}")
}

/// Run a fixture under `chelis eval` and return its exit code with its combined
/// output.
///
/// [`eval_result`]'s boolean cannot separate the two outcomes chelis#1797 is
/// about: a typed diagnostic exits 1 and a panicking evaluator exits 101, and
/// both are "not success".
fn eval_code(dir: &TempDir, name: &str, source: &str) -> (Option<i32>, String) {
    let run = eval(&fixture(dir, name, source));
    let mut text = String::from_utf8_lossy(&run.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&run.stderr));
    (run.status.code(), text)
}

/// shrink.runtime_bound.overshoot.eval: chelis#1797. An overshooting runtime
/// `shrink` end is reported as `spec/05-risc-primitives.md` section 2.4.1's
/// overshoot error, in the compiled lane's exact words, under every
/// result-claim spelling.
///
/// Two things had to change together. `eval::shrink` answers an out-of-domain
/// bound with a typed `Err` carrying the compiled lane's two lines instead of
/// an `assert!`; and `computed_axis_extent_value` DECLINES a span whose end
/// runs past the operand, so the extent guard yields and the operation reports
/// the defect that is actually present. Declining alone would have traded a
/// diagnostic naming the wrong reason for a panic, which is why chelis#1790
/// bounded its cross-lane claim to in-domain spans rather than repairing this.
///
/// The in-domain guard is untouched, and
/// [`a_local_unit_extent_claim_traps_at_its_operation_on_eval`] and the
/// claim rows around it are the controls that say so: a span the operand
/// contains still compares against its claim and still reports a mismatch as
/// [04-NUM-9]'s line.
///
/// EVIDENTIARY STATUS: regression test, all three spellings. Measured on the
/// base `6abca2406`: `claim = "2"` printed
/// ``error: extent `2`: claimed = 2, shrink axis 0 = 6`` with the trap line and
/// exit 1, naming a claim that was not the defect; `claim = "6"` and
/// `claim = "k"` both PANICKED at `crates/chelis-ir/src/eval.rs:1496` with
/// exit 101 and no diagnostic at all.
#[test]
fn an_overshooting_shrink_span_reports_the_domain_error_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    for claim in OVERSHOOT_CLAIMS {
        let (code, out) = eval_code(
            &dir,
            &format!("overshoot_{claim}.ch"),
            &overshooting_shrink_source(claim),
        );
        assert_eq!(
            out,
            overshoot_eval_rendering(),
            "claim `{claim}` reports the overshoot and nothing else"
        );
        assert_eq!(
            code,
            Some(1),
            "claim `{claim}` exits 1 as a diagnostic, not 101 as a panic: {out}"
        );
    }
}

/// shrink.runtime_bound.overshoot.c: the compiled twin, and the half that makes
/// the row above a cross-lane claim rather than one lane's wording.
///
/// C reaches the rejection before the guard site: `chelis_tensor_affine_plan`
/// builds the movement plan first, so `ShapeMetadata::shrunk` refuses the
/// bounds and the emitted claim comparison never executes. That order is the
/// one the eval lane now takes too, which is what lets both lanes print
/// [`OVERSHOOT_RENDERING`] verbatim.
///
/// EVIDENTIARY STATUS: disposition lock. This lane is unchanged by chelis#1797
/// and printed exactly these bytes on the base `6abca2406` for all three
/// spellings; the row is here because the eval receipt above is required to
/// match it, and because a later change to either lane has to move both.
#[test]
fn an_overshooting_shrink_span_reports_the_domain_error_on_c() {
    assert!(
        gcc_available(),
        "this row is one half of a cross-lane claim; it may not skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    for claim in OVERSHOOT_CLAIMS {
        let (status, out) = c_run_status(
            &dir,
            &format!("overshoot_{claim}_c"),
            &overshooting_shrink_source(claim),
        );
        assert_eq!(
            out, OVERSHOOT_RENDERING,
            "claim `{claim}` reports the overshoot and nothing else"
        );
        assert_eq!(
            status.code(),
            Some(1),
            "claim `{claim}` exits 1, the runtime's `Domain` exit: {out}"
        );
    }
}

/// Negative parity for the pair above, and the boundary of what chelis#1797
/// repairs: a span that overshoots AND selects nothing is still refused by
/// chelis#616's operation-level admission rule, which runs before the operand's
/// shape is consulted.
///
/// `[shape(x, 0) + 3, shape(x, 0) + 3)` is `[7, 7)` over an operand of four, so
/// it is out of domain by the same three elements as the rows above and empty
/// as well. The `RiscOp::Shrink` arm rejects `start >= end` first and never
/// reaches `eval::shrink`, so this spelling keeps the admission rule's wording
/// while the compiled lane keeps saying `Domain: shrink bounds outside input
/// extent`. That divergence is chelis#1795's, whose reproducer is the same
/// empty span without the overshoot, and repairing it would change text this
/// change does not own.
///
/// EVIDENTIARY STATUS: disposition lock, eval lane. Unchanged from the base
/// `6abca2406`. The row exists so the overshoot claim above is read as bounded
/// to spans that select something, rather than as covering every out-of-domain
/// bound.
#[test]
fn an_overshooting_empty_span_keeps_the_616_admission_rule_wording_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = format!(
        "def f(x: tensor[rows, f32]) -> tensor[2, f32] = \
         shrink(x, [[add(shape(x, 0i32), 3i64), add(shape(x, 0i32), 3i64)]])\n\
         out = f({})\n",
        vector_literal(4)
    );
    let (code, out) = eval_code(&dir, "overshoot_empty.ch", &source);
    assert_eq!(
        out, "error: shrink axis 0 bound [7, 7] is empty or inverted (start >= end)\n",
        "the admission rule reports the span, not the overshoot (chelis#1795)"
    );
    assert_eq!(code, Some(1), "and does so as a diagnostic: {out}");
}

/// The other non-selecting shape, for the same reason: a span that overshoots
/// AND is INVERTED. Without this row the bound above is pinned for one of the
/// two ways a span can select nothing, which would read as if the inverted one
/// had been repaired.
///
/// `[shape(x, 0) + 3, 1)` is `[7, 1)` over an operand of four. Both of
/// chelis#616's conditions hold, and its `start >= end` rejection is the one
/// that fires, so this lane keeps the admission rule's wording where the
/// compiled lane says `Domain: shrink bounds outside input extent`. The
/// evaluator's own `SHRINK_DOMAIN_TRAP` branch would give the compiled lane's
/// words for an inverted bound, and it is unreachable from source precisely
/// because this rejection precedes it.
///
/// Cited to chelis#1795 like the empty row, and for the same reason: the
/// divergence is that operation-level admission rule, which the numbered spec
/// does not require and which chelis#1797 does not own.
///
/// EVIDENTIARY STATUS: disposition lock, eval lane. Unchanged from the base
/// `6abca2406`, where this spelling took the same rejection before reaching the
/// former `assert!`.
#[test]
fn an_overshooting_inverted_span_keeps_the_616_admission_rule_wording_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = format!(
        "def f(x: tensor[rows, f32]) -> tensor[2, f32] = \
         shrink(x, [[add(shape(x, 0i32), 3i64), 1i64]])\n\
         out = f({})\n",
        vector_literal(4)
    );
    let (code, out) = eval_code(&dir, "overshoot_inverted.ch", &source);
    assert_eq!(
        out, "error: shrink axis 0 bound [7, 1] is empty or inverted (start >= end)\n",
        "the admission rule reports the span, not the overshoot (chelis#1795)"
    );
    assert_eq!(code, Some(1), "and does so as a diagnostic: {out}");
}

// ---------------------------------------------------------------------------
// chelis#1379: a checked integer-arithmetic `expand`/`insert` size.
//
// `spec/04-type-system.md` §4.7.2 puts a "checked integer-arithmetic
// expression" in the same admissibility list as a parameter, a binding, a cast
// and a user-function result, and says no stage may reject an extent because
// of its provenance. §4.7.4 says such an extent is lowered as ordinary typed
// integer dataflow and that eval, C, HIP and Metal execute the same graph.
//
// Before this change the checker admitted `mul(shape(x, 0), 2i64)` as
// shape-sourced and no lane could run it. Measured on `6dbbbf2bc`, by reverting
// both source trees and rebuilding: under the value-binding root below, eval and
// the C lane BOTH exit 1 with the chelis#469 lowering rejection; under a
// `def main()` root, which inlines and takes the host path, eval exits 101 with
// empty stdout and stderr, a suppressed `UnrepresentableDag` raise escaping as a
// silent panic. chelis#1379 records eval printing `shape=[6]` unguarded, and
// that is no longer what either root does. The rows below pin the repair: the
// size lowers as an ordinary scalar-input extent, and the claim it sits under is
// checked at execution on each lane.
// ---------------------------------------------------------------------------

/// chelis#1379's reproducer in the value-binding root form, with `claim`
/// naming the declared result extent and `factor` the multiplier applied to
/// the input's own extent.
///
/// With `claim = "n"` and `factor = 2` the computed extent is `2n` under a
/// result claimed as `n`. With `factor = 1` the same mechanism produces an
/// extent that AGREES, which is what separates a guard from a lane that traps
/// on every computed size.
fn arith_size_source(claim: &str, factor: u32) -> String {
    format!(
        "def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[{claim}, f32] = \
         insert(b, 0, mul(shape(x, 0), {factor}i64))\n\
         seed = sum(to_tensor([1.0f32]), 0)\n\
         xs = to_tensor([1.0f32, 2.0f32, 3.0f32])\n\
         out = f(seed, xs)\n"
    )
}

/// expand.arith_size.named_claim.c
///
/// EVIDENTIARY STATUS: regression test. Watched failing on the base of this
/// change, where `chelis build --target c` refused the program outright with
/// the chelis#469 "cannot materialize as an extent" lowering rejection, so the
/// fixture could not reach a linked binary at all.
#[test]
fn checked_arithmetic_expand_size_under_a_named_claim_agrees_on_every_lane_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "arith_size_c", &arith_size_source("n", 2));
    assert!(
        !ok,
        "the computed extent is 2n under a claim of n, so the binary must fail: {out}"
    );
    assert!(
        out.contains(&domain_trap_line("insert")),
        "the guard takes the position of the insert that introduces the extent: {out}"
    );
    assert!(
        out.contains("extent `n`: claimed = 3, insert axis 0 = 6"),
        "section 4.7's context line names both disagreeing sides: {out}"
    );
}

/// The discriminating twin: the same arithmetic with a factor that AGREES with
/// the claim executes and produces the declared shape. Without it, a lane that
/// rejected or trapped on every arithmetic size would pass the row above.
///
/// EVIDENTIARY STATUS: disposition lock.
#[test]
fn a_checked_arithmetic_expand_size_that_agrees_with_its_claim_executes_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "arith_size_ok_c", &arith_size_source("n", 1));
    assert!(
        ok,
        "the computed extent agrees, so the binary must run: {out}"
    );
    assert!(
        out.contains("shape=[3]") && out.contains("data=[1.0, 1.0, 1.0]"),
        "the agreeing program produces its declared shape exactly: {out}"
    );
}

/// expand.arith_size.named_claim.eval
///
/// EVIDENTIARY STATUS: regression test. Watched failing on the base of this
/// change, where `chelis eval --file` exited 1 with the chelis#469 lowering
/// rejection, the same refusal the compiled lane gave. It did NOT print
/// `shape=[6]`: this root is a value binding, so eval routes it through the
/// same lowering the C lane uses. Only the `def main()` root reaches the host
/// path, and on the base that one exits 101 with empty output rather than
/// printing an unguarded result.
#[test]
fn checked_arithmetic_expand_size_under_a_named_claim_agrees_on_every_lane_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "arith_size_eval.ch", &arith_size_source("n", 2));
    assert!(
        !ok,
        "the computed extent is 2n under a claim of n, so eval must fail: {out}"
    );
    assert!(
        !out.contains("shape=[6]"),
        "and must not produce the unguarded 2n result chelis#1379 reported: {out}"
    );
    assert!(
        out.contains(&domain_trap_line("insert")),
        "eval renders [04-NUM-9]'s line for the same guard the C lane emits: {out}"
    );
    assert!(
        out.contains("extent `n`: claimed = 3, insert axis 0 = 6"),
        "section 4.7's context line names both disagreeing sides, text for text \
         with the C lane: {out}"
    );
}

/// The discriminating twin on eval, for the same reason as its C sibling.
///
/// EVIDENTIARY STATUS: disposition lock.
#[test]
fn a_checked_arithmetic_expand_size_that_agrees_with_its_claim_executes_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "arith_size_ok_eval.ch", &arith_size_source("n", 1));
    assert!(ok, "the computed extent agrees, so eval must run: {out}");
    assert!(
        out.contains("shape=[3]") && out.contains("data=[1.0, 1.0, 1.0]"),
        "the agreeing program produces its declared shape exactly: {out}"
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
    // The leading index is the injective part of the name (round 1, P1-1);
    // the path tail is there so a reader can find the field.
    assert!(
        emitted.contains("__host_record_field_0_inp_q"),
        "the projected field is bound to a host local: {emitted}"
    );
    assert!(
        emitted.contains("input `__host_record_field_0_inp_q` at slot 0"),
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

// ---------------------------------------------------------------------------
// Round-1 repairs (chelis#1266). Both are about the hoist's IDENTITY and its
// COVERAGE rather than about whether a projection is admissible at all, so
// they sit beside the rows above rather than in them.
// ---------------------------------------------------------------------------

/// Two projection paths whose segments join to the same string. `_` separates
/// segments and also occurs inside field names, so `inputs.features.mask` and
/// `inputs.features_mask` are the natural collision, not a contrived one: a
/// nested `features` record beside a sibling `features_mask` is ordinary
/// model-input shape.
const COLLIDING_PROJECTION_PATHS: &str = "module Repro.CollidingPaths\n\
     type Features = | Features { mask: tensor[2, f32] }\n\
     type ForwardInputs = | ForwardInputs { features: Features, \
     features_mask: tensor[2, f32] }\n\
     sig forward: ForwardInputs -> tensor[2, f32]\n\
     def forward(inputs: ForwardInputs) = add(inputs.features.mask, inputs.features_mask)\n\
     out = forward(ForwardInputs { features: Features { mask: to_tensor([1.0f32, 2.0f32]) }, \
     features_mask: to_tensor([10.0f32, 20.0f32]) })\n";

/// The spelling the §4.7.2 diagnostic's own suggestion text asks for: "bind
/// that read to a `let`".
const LET_BOUND_RECORD_SHAPE_READ: &str = "module Repro.LetBoundRecordRead\n\
     type Inputs = | Inputs { q: tensor[batch, f32] }\n\
     sig f: Inputs -> tensor[batch, f32]\n\
     def f(inp: Inputs) = { a_dim = cast(shape(inp.q, cast(0, int32)), int64)\n\
     expand(to_tensor([0.25f32]), 0i32, a_dim) }\n\
     out = f(Inputs { q: to_tensor([1.0f32, 2.0f32, 3.0f32]) })\n";

/// The same read piped, which is what `chelis lint --fix` makes of it.
const LET_BOUND_PIPED_RECORD_SHAPE_READ: &str = "module Repro.LetBoundPipedRecordRead\n\
     type Inputs = | Inputs { q: tensor[batch, f32] }\n\
     sig f: Inputs -> tensor[batch, f32]\n\
     def f(inp: Inputs) = \
     { a_dim = inp.q |> shape(cast(0, int32)) |> cast(int64)\n\
     expand(to_tensor([0.25f32]), 0i32, a_dim) }\n\
     out = f(Inputs { q: to_tensor([1.0f32, 2.0f32, 3.0f32]) })\n";

/// A whole `expand` inside a `let` value, whose projection sits two levels
/// down. This is the spelling that reached the [04-TOT-2] wall the C4
/// paragraph says the hoist exists to prevent.
const LET_VALUE_RECORD_EXPAND: &str = "module Repro.LetValueRecordExpand\n\
     type Inputs = | Inputs { q: tensor[batch, f32], b: tensor[1, f32] }\n\
     sig f: Inputs -> tensor[batch, f32]\n\
     def f(inp: Inputs) = \
     { y = expand(inp.b, cast(0, int32), shape(inp.q, cast(0, int32)))\n\
     y }\n\
     out = f(Inputs { q: to_tensor([1.0f32, 2.0f32, 3.0f32]), b: to_tensor([0.5f32]) })\n";

/// Round 1, P1-1. Two distinct projection paths get two distinct locals.
///
/// The hoist's identity is the path, never the rendered name, and the name
/// carries a per-def index so it is injective whatever the segments spell.
///
/// EVIDENTIARY STATUS: regression test. Measured RED at `612199377`, where the
/// dedup keyed on a joined string: both projections rewrote to one local, the
/// compiled binary printed `[2.0, 4.0]` and exited 0 while eval printed
/// `[11.0, 22.0]`. On `33cc78e84` this program is correct on C and loudly
/// refused on eval, so the defect was a regression in both directions at once.
#[test]
fn two_record_paths_that_join_to_one_string_get_two_locals() {
    assert_both_lanes_render(
        "colliding_paths",
        COLLIDING_PROJECTION_PATHS,
        "shape=[2], data=[11.0, 22.0]",
    );
}

/// Round 1, P1-2. A projection nested inside a `let` binding's value is
/// hoisted, and the body that reads it stays whole.
///
/// The three spellings are one finding: the exemption skipped the whole bind
/// VALUE slot instead of only the bare projection, so the checker admitted
/// sizes the C lane refused.
///
/// EVIDENTIARY STATUS: regression test. All three measured RED at
/// `612199377`: check clean, eval correct, and C rejecting with either
/// "`expand` size resolves to `a_dim`, but no in-scope tensor axis supplies
/// that extent" (the first two) or the [04-TOT-2] host-emission wall (the
/// third). On `33cc78e84` all three were rejected at CHECK, so none of them
/// is a pre-existing divergence.
#[test]
fn a_record_projection_nested_in_a_let_value_is_hoisted() {
    assert_both_lanes_render(
        "let_bound_record_read",
        LET_BOUND_RECORD_SHAPE_READ,
        "shape=[3], data=[0.25, 0.25, 0.25]",
    );
    assert_both_lanes_render(
        "let_bound_piped_record_read",
        LET_BOUND_PIPED_RECORD_SHAPE_READ,
        "shape=[3], data=[0.25, 0.25, 0.25]",
    );
    assert_both_lanes_render(
        "let_value_record_expand",
        LET_VALUE_RECORD_EXPAND,
        "shape=[3], data=[0.5, 0.5, 0.5]",
    );
}

/// Negative parity for that widening: the alias spelling, whose bind value IS
/// the bare projection, is still exempt and still emits exactly the C it
/// emitted before this pull request.
///
/// EVIDENTIARY STATUS: disposition lock, and the control on P1-2's repair.
/// Widening the exemption's inverse too far would add a second alias to a
/// spelling that already lowers; this pins the bytes against `33cc78e84`.
#[test]
fn the_local_alias_spelling_emits_the_same_c_it_always_did() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out_dir = dir.path().join("alias_bytes-out");
    let build = build_c(
        &fixture(&dir, "alias_bytes.ch", RECORD_PROJECTION_ALIAS),
        &out_dir,
    );
    assert!(
        build.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted = fs::read_to_string(out_dir.join("alias_bytes.c")).expect("generated C");
    assert!(
        !emitted.contains("__host_record_field_"),
        "the alias spelling binds its own local and needs no hoisted one: {emitted}"
    );
    assert!(
        emitted.contains("chelis_adt_get_field"),
        "it still reads the field on the host lane: {emitted}"
    );
}

/// Round 1, P1-3. A lambda whose TYPED parameter shadows the record base.
/// Deep renders a typed parameter as an `UnknownForm` whose head is the name,
/// which the hand-written shadow reader did not recognize, so the inner
/// projection was rewritten to the OUTER record's local.
const TYPED_LAMBDA_SHADOWS_THE_RECORD_BASE: &str = "module Repro.TypedShadow\n\
     type Inputs = | Inputs { q: tensor[2, f32] }\n\
     def g(t: tensor[2, f32]) -> tensor[2, f32] = add(t, to_tensor([100.0f32, 100.0f32]))\n\
     sig f: Inputs -> tensor[2, f32]\n\
     def f(inp: Inputs) = add(inp.q, \
     (fn (inp: Inputs) -> g(inp.q))(Inputs { q: to_tensor([7.0f32, 8.0f32]) }))\n\
     out = f(Inputs { q: to_tensor([1.0f32, 2.0f32]) })\n";

/// The same shadow inside a `let` value, which the round-1 slot-wide exemption
/// had masked and the narrowed exemption exposed.
const TYPED_LAMBDA_SHADOW_IN_A_LET_VALUE: &str = "module Repro.TypedShadowLet\n\
     type Inputs = | Inputs { q: tensor[2, f32] }\n\
     def g(t: tensor[2, f32]) -> tensor[2, f32] = add(t, to_tensor([100.0f32, 100.0f32]))\n\
     sig f: Inputs -> tensor[2, f32]\n\
     def f(inp: Inputs) = \
     { y = (fn (inp: Inputs) -> g(inp.q))(Inputs { q: to_tensor([7.0f32, 8.0f32]) })\n\
     add(inp.q, y) }\n\
     out = f(Inputs { q: to_tensor([1.0f32, 2.0f32]) })\n";

/// The UNTYPED twin. Its binder is a bare name, the spelling the old reader
/// did handle, so it was correct before and must stay correct.
const UNTYPED_LAMBDA_SHADOWS_THE_RECORD_BASE: &str = "module Repro.UntypedShadow\n\
     type Inputs = | Inputs { q: tensor[2, f32] }\n\
     def g(t: tensor[2, f32]) -> tensor[2, f32] = add(t, to_tensor([100.0f32, 100.0f32]))\n\
     sig f: Inputs -> tensor[2, f32]\n\
     def f(inp: Inputs) = add(inp.q, \
     (fn (inp) -> g(inp.q))(Inputs { q: to_tensor([7.0f32, 8.0f32]) }))\n\
     out = f(Inputs { q: to_tensor([1.0f32, 2.0f32]) })\n";

/// Round 1, P1-3. A rebinding of the record base suppresses the hoist, in
/// every binder spelling.
///
/// The hoist no longer reconstructs lexical scope from tag shapes. It asks one
/// question -- is this name bound anywhere under the body -- and answers it
/// from the closed vocabulary's own `Binder` child role, failing closed on a
/// spelling it cannot decode. `crates/chelis-ir/src/host.rs`'s
/// `every_binder_position_in_the_closed_vocabulary_yields_a_name` is the
/// oracle over that vocabulary; these are the surface programs.
///
/// EVIDENTIARY STATUS: regression test for the two typed rows, disposition
/// lock for the untyped one. Measured at `eccdcb7e2`: the typed lambda gave
/// eval `[108.0, 110.0]` against compiled C `[102.0, 104.0]`, exit 0 and no
/// diagnostic, in both the body and the let-value position. The untyped twin
/// was correct there and on `33cc78e84`.
#[test]
fn a_binder_that_shadows_the_record_base_suppresses_the_hoist() {
    assert_both_lanes_render(
        "typed_shadow",
        TYPED_LAMBDA_SHADOWS_THE_RECORD_BASE,
        "shape=[2], data=[108.0, 110.0]",
    );
    assert_both_lanes_render(
        "typed_shadow_let",
        TYPED_LAMBDA_SHADOW_IN_A_LET_VALUE,
        "shape=[2], data=[108.0, 110.0]",
    );
    assert_both_lanes_render(
        "untyped_shadow",
        UNTYPED_LAMBDA_SHADOWS_THE_RECORD_BASE,
        "shape=[2], data=[108.0, 110.0]",
    );
}

/// Round 2, P1-1. A TYPED lambda parameter that does NOT shadow the record
/// base. Deep writes it `(t {type: ..})`, an unstamped form whose head is the
/// name, and the hoist must read it rather than refuse.
const TYPED_LAMBDA_BESIDE_A_PROJECTION: &str = "module Repro.TypedLambdaBeside\n\
     type Inputs = | Inputs { q: tensor[batch, f32], b: tensor[1, f32] }\n\
     def scale(t: tensor[1, f32]) -> tensor[1, f32] = mul(t, t)\n\
     sig f: Inputs -> tensor[batch, f32]\n\
     def f(inp: Inputs) = expand((fn (t: tensor[1, f32]) -> scale(t))(inp.b), 0i32, \
     shape(inp.q, cast(0, int32)))\n\
     out = f(Inputs { q: to_tensor([1.0f32, 2.0f32, 3.0f32]), b: to_tensor([0.25f32]) })\n";

/// The untyped twin, whose binder the earlier readers did handle.
const UNTYPED_LAMBDA_BESIDE_A_PROJECTION: &str = "module Repro.UntypedLambdaBeside\n\
     type Inputs = | Inputs { q: tensor[batch, f32], b: tensor[1, f32] }\n\
     def scale(t: tensor[1, f32]) -> tensor[1, f32] = mul(t, t)\n\
     sig f: Inputs -> tensor[batch, f32]\n\
     def f(inp: Inputs) = expand((fn (t) -> scale(t))(inp.b), 0i32, \
     shape(inp.q, cast(0, int32)))\n\
     out = f(Inputs { q: to_tensor([1.0f32, 2.0f32, 3.0f32]), b: to_tensor([0.25f32]) })\n";

/// Round 2, P1-1. A typed binder that does not shadow the record base no
/// longer stops the hoist.
///
/// EVIDENTIARY STATUS: regression test. Measured RED at `2570da8d1`: check
/// clean, eval `[0.0625, 0.0625, 0.0625]`, and the C lane refusing with
/// `unsupported: builtin expand on chelis build host emission ... [04-TOT-2]`,
/// because the reader could not decode `(t {type: ..})` and the hoist was
/// abandoned for the whole definition. The untyped twin built there, which is
/// what identified the binder spelling as the cause. On `33cc78e84` the typed
/// program was rejected at check, so neither lane had it working.
#[test]
fn a_typed_binder_beside_a_projection_does_not_stop_the_hoist() {
    assert_both_lanes_render(
        "typed_lambda_beside",
        TYPED_LAMBDA_BESIDE_A_PROJECTION,
        "shape=[3], data=[0.0625, 0.0625, 0.0625]",
    );
    assert_both_lanes_render(
        "untyped_lambda_beside",
        UNTYPED_LAMBDA_BESIDE_A_PROJECTION,
        "shape=[3], data=[0.0625, 0.0625, 0.0625]",
    );
}

/// Round 2, P1-1. The two outcomes are told apart in the emitted C, not
/// inferred from the values.
///
/// A definition whose typed binder is merely NEARBY hoists: its C carries the
/// hoisted local. A definition whose typed binder REBINDS the record base
/// suppresses that base's projections: its C carries none, and the host lane
/// emits the read itself. Without this pair a suppressed shadow and an
/// abandoned hoist look identical from the values alone, which is exactly how
/// round 1's shadow receipt passed while the reader was broken.
///
/// EVIDENTIARY STATUS: regression test on the first assertion, disposition
/// lock on the second. At `2570da8d1` BOTH emitted zero hoisted locals,
/// because the reader refused on either program.
#[test]
fn a_typed_binder_suppresses_only_the_base_it_rebinds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let emitted = |stem: &str, source: &str| -> String {
        let out_dir = dir.path().join(format!("{stem}-out"));
        let build = build_c(&fixture(&dir, &format!("{stem}.ch"), source), &out_dir);
        assert!(
            build.status.success(),
            "{stem} must build: {}",
            String::from_utf8_lossy(&build.stderr)
        );
        fs::read_to_string(out_dir.join(format!("{stem}.c"))).expect("generated C")
    };
    let beside = emitted("typed_beside_c", TYPED_LAMBDA_BESIDE_A_PROJECTION);
    assert!(
        beside.contains("__host_record_field_"),
        "a typed binder beside the projection leaves the hoist alone: {beside}"
    );
    let shadowing = emitted("typed_shadow_c", TYPED_LAMBDA_SHADOWS_THE_RECORD_BASE);
    assert!(
        !shadowing.contains("__host_record_field_"),
        "a typed binder that rebinds the base suppresses that base's \
         projections: {shadowing}"
    );
    assert!(
        shadowing.contains("chelis_adt_get_field"),
        "and the host lane reads the field itself instead: {shadowing}"
    );
}
/// The same arithmetic in the `expand` position rather than the `insert` one.
/// `expand` takes its result dim from the CHECKER's stamped type while `insert`
/// derives it from the size, so the two reach the guard by different routes and
/// a repair that served only one would leave the other unguarded.
///
/// EVIDENTIARY STATUS: regression test. Watched failing on `fc5b6aa99`, where
/// the same chelis#469 rejection refused the build.
#[test]
fn a_checked_arithmetic_expand_size_is_guarded_in_the_expand_position_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(b: tensor[1, f32], x: tensor[n, f32]) -> tensor[n, f32] = \
                  expand(b, 0, mul(shape(x, 0), 2i64))\n\
                  xs = to_tensor([1.0f32, 2.0f32, 3.0f32])\n\
                  out = f(to_tensor([7.0f32]), xs)\n";
    let (ok, out) = c_run_result(&dir, "arith_size_expand_c", source);
    assert!(!ok, "2n under a claim of n must not produce a value: {out}");
    assert!(
        out.contains(&domain_trap_line("expand"))
            && out.contains("extent `n`: claimed = 3, expand axis 0 = 6"),
        "the guard names `expand`, the operation that introduces the extent: {out}"
    );
}

/// The `expand` position on eval, byte-identical to its C sibling.
///
/// EVIDENTIARY STATUS: regression test, watched failing on `fc5b6aa99`.
#[test]
fn a_checked_arithmetic_expand_size_is_guarded_in_the_expand_position_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(b: tensor[1, f32], x: tensor[n, f32]) -> tensor[n, f32] = \
                  expand(b, 0, mul(shape(x, 0), 2i64))\n\
                  xs = to_tensor([1.0f32, 2.0f32, 3.0f32])\n\
                  out = f(to_tensor([7.0f32]), xs)\n";
    let (ok, out) = eval_result(&dir, "arith_size_expand_eval.ch", source);
    assert!(!ok, "2n under a claim of n must not produce a value: {out}");
    assert!(
        out.contains(&domain_trap_line("expand"))
            && out.contains("extent `n`: claimed = 3, expand axis 0 = 6"),
        "eval renders the same line the C lane emits: {out}"
    );
}

/// `sub`, `floor_div`, `trunc_div`, `mod`, `neg` and nested combinations reach
/// the same admission and the same guard as `mul`. The walk combines operand
/// classes rather than matching a spelling, so a repair keyed to one operator
/// would leave the rest rejecting. `div` is absent deliberately:
/// `spec/05-risc-primitives.md` section 2.1 makes it float-only, so it cannot
/// carry an `int64` extent and a row for it would fail on precision before
/// reaching the size class.
///
/// Each row states the arithmetic and the extent it computes from `n = 4`.
///
/// EVIDENTIARY STATUS: regression test. Every row was refused at lowering on
/// `fc5b6aa99` with the chelis#469 rejection.
#[test]
fn every_checked_arithmetic_operator_is_an_admissible_expand_size_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    // (size expression, extent it computes, agrees with the claim `n` = 4)
    let rows = [
        ("sub(shape(x, 0), 1i64)", 3, false),
        ("floor_div(shape(x, 0), 2i64)", 2, false),
        ("trunc_div(shape(x, 0), 4i64)", 1, false),
        ("mod(shape(x, 0), 3i64)", 1, false),
        ("neg(neg(shape(x, 0)))", 4, true),
        ("add(sub(shape(x, 0), 1i64), 1i64)", 4, true),
        ("mul(floor_div(shape(x, 0), 2i64), 2i64)", 4, true),
    ];
    for (index, (size, extent, agrees)) in rows.iter().enumerate() {
        let source = format!(
            "def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[n, f32] = insert(b, 0, {size})\n\
             seed = sum(to_tensor([1.0f32]), 0)\n\
             xs = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])\n\
             out = f(seed, xs)\n"
        );
        let (ok, out) = eval_result(&dir, &format!("arith_op_{index}.ch"), &source);
        assert_eq!(
            ok, *agrees,
            "`{size}` computes {extent} under a claim of 4: {out}"
        );
        if *agrees {
            assert!(
                out.contains("shape=[4]"),
                "`{size}` agrees, so it produces the declared shape: {out}"
            );
        } else {
            assert!(
                out.contains(&domain_trap_line("insert"))
                    && out.contains(&format!(
                        "extent `n`: claimed = 4, insert axis 0 = {extent}"
                    )),
                "`{size}` disagrees, so its guard names both sides: {out}"
            );
        }
    }
}

/// Arithmetic that combines TWO tensors' axes under a claim matching one of
/// them is guarded against the value the arithmetic computes, not against
/// either operand's own axis. `n = 3` and `m = 2` give `3 + 2 = 5` under a
/// claim of 3, so the line must report 5.
///
/// EVIDENTIARY STATUS: regression test. Refused at lowering on `fc5b6aa99`.
#[test]
fn arithmetic_over_two_tensors_is_guarded_against_the_value_it_computes_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(b: tensor[f32], x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = \
                  insert(b, 0, add(shape(x, 0), shape(y, 0)))\n\
                  seed = sum(to_tensor([1.0f32]), 0)\n\
                  xs = to_tensor([1.0f32, 2.0f32, 3.0f32])\n\
                  ys = to_tensor([1.0f32, 2.0f32])\n\
                  out = f(seed, xs, ys)\n";
    let (ok, out) = eval_result(&dir, "arith_two_tensors.ch", source);
    assert!(
        !ok,
        "3 + 2 under a claim of 3 must not produce a value: {out}"
    );
    assert!(
        out.contains("extent `n`: claimed = 3, insert axis 0 = 5"),
        "the observed side is the sum, not either operand's own axis: {out}"
    );
    assert!(
        !out.contains("insert axis 0 = 2") && !out.contains("insert axis 0 = 3"),
        "and neither operand's extent is reported in its place: {out}"
    );
}

/// A shape read mixed with a runtime scalar PARAMETER is admissible, and the
/// scalar's value reaches the guard. This is the row the checker change exists
/// for: `add(shape(x, 0), k)` has one operand with a real shape source and one
/// with none, and the arithmetic walk used to let the sourceless operand poison
/// the whole expression.
///
/// EVIDENTIARY STATUS: regression test. On `fc5b6aa99` both spellings were
/// refused at CHECK with "`insert` size resolves to a runtime scalar", so
/// neither lane ever saw the program.
#[test]
fn a_shape_read_mixed_with_a_runtime_scalar_is_an_admissible_expand_size_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = |k: &str| {
        format!(
            "def f(b: tensor[f32], x: tensor[n, f32], k: int64) -> tensor[n, f32] = \
             insert(b, 0, add(shape(x, 0), k))\n\
             seed = sum(to_tensor([1.0f32]), 0)\n\
             xs = to_tensor([1.0f32, 2.0f32, 3.0f32])\n\
             out = f(seed, xs, {k})\n"
        )
    };
    // The control first, so the row proves a guard and not a broken lane.
    let (ok, out) = eval_result(&dir, "scalar_mix_ok.ch", &source("0i64"));
    assert!(ok, "k = 0 gives n + 0 = n, which agrees: {out}");
    assert!(
        out.contains("shape=[3]") && out.contains("data=[1.0, 1.0, 1.0]"),
        "the agreeing program produces its declared shape exactly: {out}"
    );

    let (ok, out) = eval_result(&dir, "scalar_mix_trap.ch", &source("2i64"));
    assert!(!ok, "k = 2 gives n + 2 = 5 under a claim of 3: {out}");
    assert!(
        out.contains("extent `n`: claimed = 3, insert axis 0 = 5"),
        "the scalar parameter's value reaches the guard: {out}"
    );
}

/// A def-returned scalar mixed with a shape read is admissible for the same
/// reason. A user `def` is opaque to the checker's walk, so it classifies as
/// sourceless; the shape read beside it is what makes the expression
/// admissible.
///
/// EVIDENTIARY STATUS: regression test. Refused at CHECK on `fc5b6aa99`.
#[test]
fn a_def_returned_scalar_mixed_with_a_shape_read_is_an_admissible_expand_size_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def g(x: tensor[n, f32]) -> int64 = shape(x, 0)\n\
                  def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[n, f32] = \
                  insert(b, 0, add(shape(x, 0), g(x)))\n\
                  seed = sum(to_tensor([1.0f32]), 0)\n\
                  xs = to_tensor([1.0f32, 2.0f32, 3.0f32])\n\
                  out = f(seed, xs)\n";
    let (ok, out) = eval_result(&dir, "def_returned_mix.ch", source);
    assert!(!ok, "n + n = 6 under a claim of 3 must not run: {out}");
    assert!(
        out.contains("extent `n`: claimed = 3, insert axis 0 = 6"),
        "the def's returned extent reaches the guard: {out}"
    );
}

/// The negative that bounds the checker change: arithmetic with NO admissible
/// operand stays sourceless, with its diagnostic unchanged. Without this row a
/// rule that simply stopped rejecting arithmetic would pass every positive
/// above while admitting a size no lane can source.
///
/// EVIDENTIARY STATUS: disposition lock. Measured identical on `fc5b6aa99` and
/// on this head; the behaviour is deliberately unchanged.
#[test]
fn arithmetic_over_a_scalar_with_no_tensor_source_is_still_sourceless() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, size) in [
        ("bare_scalar_arith", "add(k, 1i64)"),
        ("bare_scalar_mul", "mul(k, 2i64)"),
        ("bare_scalar_nested", "add(mul(k, 2i64), 1i64)"),
    ] {
        let source = format!(
            "def f(b: tensor[f32], k: int64) -> tensor[m, f32] = insert(b, 0, {size})\n\
             seed = sum(to_tensor([1.0f32]), 0)\n\
             out = f(seed, 3i64)\n"
        );
        let (ok, out) = eval_result(&dir, &format!("{name}.ch"), &source);
        assert!(
            !ok,
            "`{size}` has no shape source and must be rejected: {out}"
        );
        assert!(
            out.contains("`insert` size resolves to a runtime scalar, but no tensor in scope")
                && out.contains("chelis#469"),
            "`{size}` keeps the section 4.7.2 sourceless diagnostic verbatim: {out}"
        );
    }
}

/// A bare runtime scalar is still rejected too. `add(k, 1i64)` above and a bare
/// `k` are the same class, and a change that admitted arithmetic by weakening
/// the leaf rule rather than the combination rule would separate them.
///
/// EVIDENTIARY STATUS: disposition lock. Unchanged from `fc5b6aa99`.
#[test]
fn a_bare_runtime_scalar_expand_size_is_still_sourceless() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(b: tensor[f32], k: int64) -> tensor[m, f32] = insert(b, 0, k)\n\
                  seed = sum(to_tensor([1.0f32]), 0)\n\
                  out = f(seed, 3i64)\n";
    let (ok, out) = eval_result(&dir, "bare_scalar.ch", source);
    assert!(!ok, "a bare runtime scalar size must be rejected: {out}");
    assert!(
        out.contains("`insert` size resolves to the symbolic dimension `k`")
            && out.contains("chelis#469"),
        "with the bare-symbol wording of the same diagnostic: {out}"
    );
}

/// A computed size that goes NEGATIVE traps before allocation on BOTH lanes,
/// which is what `spec/04-type-system.md` section 4.7.2 requires of a runtime
/// negative size. What the two lanes do NOT share is the rendering, and this
/// row and its C twin below exist as a pair so that divergence is visible
/// rather than implied by one lane's text.
///
/// Eval reports through the `RtDim::Node` carrier it shares with `stride` and
/// `slice` bounds, so it says "movement bound" about an `insert` size and never
/// renders [04-NUM-9]'s form. C renders [04-NUM-9]'s form with section 4.7's
/// context line and never renders the movement-bound message. Under a free
/// result dim C says "Domain: expansion axis or extent outside domain" instead
/// of the claim line, while eval's text does not change at all. Newly reachable
/// through chelis#1379's admission, because the only node-valued sizes before
/// it came from extent witnesses, which carry real axis extents and are never
/// negative. Tracked by chelis#1802; repairing it would change text shared with
/// rows this change does not own, so both lanes are locked here and that issue
/// has to update both locks.
///
/// EVIDENTIARY STATUS: disposition lock on the rendering, regression test on
/// the trap. On `fc5b6aa99` the program was refused at lowering, so no lane
/// reached a negative extent at all.
#[test]
fn a_computed_expand_size_that_goes_negative_traps_before_allocation_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[n, f32] = \
                  insert(b, 0, sub(shape(x, 0), 5i64))\n\
                  seed = sum(to_tensor([1.0f32]), 0)\n\
                  xs = to_tensor([1.0f32, 2.0f32])\n\
                  out = f(seed, xs)\n";
    let (ok, out) = eval_result(&dir, "arith_negative.ch", source);
    assert!(!ok, "an extent of -3 must not produce a value: {out}");
    assert!(
        out.contains("must be a non-negative integer, got -3"),
        "the negative extent is reported with the value it computed: {out}"
    );
    assert!(
        !out.contains("shape=["),
        "and nothing is allocated before the trap: {out}"
    );
}

/// The C twin of the negative-extent row, and the half that makes chelis#1802 a
/// LANE DIVERGENCE rather than a wording problem: this lane renders
/// [04-NUM-9]'s form with section 4.7's context, and the words "movement bound"
/// appear nowhere in it.
///
/// EVIDENTIARY STATUS: disposition lock on the rendering, regression test on
/// the trap. Measured on this head; on `6dbbbf2bc` the program did not build.
#[test]
fn a_computed_expand_size_that_goes_negative_traps_before_allocation_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[n, f32] = \
                  insert(b, 0, sub(shape(x, 0), 5i64))\n\
                  seed = sum(to_tensor([1.0f32]), 0)\n\
                  xs = to_tensor([1.0f32, 2.0f32])\n\
                  out = f(seed, xs)\n";
    let (ok, out) = c_run_result(&dir, "arith_negative_c", source);
    assert!(!ok, "an extent of -3 must not produce a value: {out}");
    assert!(
        out.contains(&domain_trap_line("insert"))
            && out.contains("extent `n`: claimed = 2, insert axis 0 = -3"),
        "C renders [04-NUM-9]'s form with section 4.7's context: {out}"
    );
    assert!(
        !out.contains("movement bound"),
        "and never the movement-bound message eval uses for the same program \
         (chelis#1802 is a lane divergence, not one wording): {out}"
    );
    assert!(
        !out.contains("shape=["),
        "nothing is allocated before the trap: {out}"
    );
}

/// The same negative extent under a FREE result dim, where C loses the claim
/// line and falls back to the operation's own domain message while eval's text
/// is unchanged. Without this row the divergence above could be read as "C
/// always names the claim", which is not what either lane does.
///
/// EVIDENTIARY STATUS: disposition lock, both lanes. Measured on this head.
#[test]
fn a_negative_computed_size_under_a_free_result_dim_diverges_by_lane() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[m, f32] = \
                  insert(b, 0, sub(shape(x, 0), 5i64))\n\
                  seed = sum(to_tensor([1.0f32]), 0)\n\
                  xs = to_tensor([1.0f32, 2.0f32])\n\
                  out = f(seed, xs)\n";
    let (c_ok, c_out) = c_run_result(&dir, "arith_negative_free_c", source);
    let (eval_ok, eval_out) = eval_result(&dir, "arith_negative_free_eval.ch", source);
    assert!(
        !c_ok && !eval_ok,
        "neither lane produces a value: {c_out} {eval_out}"
    );
    assert!(
        c_out.contains("Domain: expansion axis or extent outside domain")
            && c_out.contains(&domain_trap_line("insert")),
        "with no claim to name, C reports the operation's own domain: {c_out}"
    );
    assert!(
        eval_out.contains("must be a non-negative integer, got -3"),
        "eval's rendering does not change with the claim shape: {eval_out}"
    );
}

/// The C twin of the operator row. The eval row above covers the whole family;
/// this one links and runs a binary for a disagreeing and an agreeing member of
/// it, which is what lets the claim say the family "executes on eval and C"
/// rather than "on eval, and on C for `mul`".
///
/// Two members rather than six: each row here compiles and links a binary, and
/// the property under test is that the operator reaches the same admission and
/// the same guard on this lane, not that every spelling has its own C process.
///
/// EVIDENTIARY STATUS: regression test. Both were refused at lowering on
/// `6dbbbf2bc` with the chelis#469 rejection.
#[test]
fn checked_arithmetic_operators_reach_the_same_guard_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let source = |size: &str| {
        format!(
            "def f(b: tensor[f32], x: tensor[n, f32]) -> tensor[n, f32] = insert(b, 0, {size})\n\
             seed = sum(to_tensor([1.0f32]), 0)\n\
             xs = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])\n\
             out = f(seed, xs)\n"
        )
    };

    // `floor_div` is the operator the shared static folder already walked and
    // the provenance walk did not, so it is the one whose admission this change
    // added; 4 / 2 = 2 disagrees with the claim of 4.
    let (ok, out) = c_run_result(
        &dir,
        "arith_op_floor_div_c",
        &source("floor_div(shape(x, 0), 2i64)"),
    );
    assert!(!ok, "2 under a claim of 4 must not produce a value: {out}");
    assert!(
        out.contains(&domain_trap_line("insert"))
            && out.contains("extent `n`: claimed = 4, insert axis 0 = 2"),
        "`floor_div` reaches the same guard on C as on eval: {out}"
    );

    // The agreeing control, nested so the walk has to combine three operands.
    let (ok, out) = c_run_result(
        &dir,
        "arith_op_nested_ok_c",
        &source("mul(floor_div(shape(x, 0), 2i64), 2i64)"),
    );
    assert!(ok, "the nested form agrees, so the binary must run: {out}");
    assert!(
        out.contains("shape=[4]"),
        "and produces the declared shape exactly: {out}"
    );
}

// ---------------------------------------------------------------------------
// chelis#1822: an `expand` whose kept axis carries a signature binder.
//
// `spec/05-risc-primitives.md` section 2.4 replaces the size-1 axis with the
// size argument's extent, so the expanded axis is the size and never the
// operand's. The C preparation's `Expand` arm resolves each anonymous output
// axis from its own source, but its guard required that NO axis carry a real
// name. With a named bystander (`batch`) the arm was skipped, and the
// pass-through arm copied the operand's PRE-EXPAND dims onto the output.
//
// The consumer form then failed `spec/04-type-system.md` section 4.7.2's
// check/build agreement at ownership verification. The no-consumer form had
// nothing to verify, so the wrong type reached codegen and the binary trapped
// while eval returned the right answer, which is a lane divergence rather than
// an internal error.
//
// The issue's "axis 0 is unaffected" sentence is false and the axis-0 row
// below is why: the variable is a named bystander plus a shape-sourced size,
// not the position of the expanded axis.
// ---------------------------------------------------------------------------

/// The issue's reproducer: `batch` spans both parameters and the broadcast size
/// is read from the other operand's axis 1.
const NAMED_BYSTANDER_AXIS_ONE: &str = "module Repro.Ax1\n\
sig f: tensor[batch, 4, f32] -> tensor[batch, 1, f32] -> tensor[batch, 4, f32]\n\
def f(features: tensor[batch, 4, f32], mask: tensor[batch, 1, f32]) = \
mul(features, expand(mask, 1i32, cast(shape(features, cast(1, int32)), int64)))\n\
out = f(to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]]), \
to_tensor([[1.0f32], [2.0f32]]))\n";

/// The same shape with the expanded axis at position 0 and `batch` at 1.
const NAMED_BYSTANDER_AXIS_ZERO: &str = "module Repro.Ax0\n\
sig f: tensor[4, batch, f32] -> tensor[1, batch, f32] -> tensor[4, batch, f32]\n\
def f(features: tensor[4, batch, f32], mask: tensor[1, batch, f32]) = \
mul(features, expand(mask, 0i32, cast(shape(features, cast(0, int32)), int64)))\n\
out = f(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32], [7.0f32, 8.0f32]]), \
to_tensor([[1.0f32, 2.0f32]]))\n";

/// The same `expand` with no binary consumer, so nothing verifies the type it
/// produced.
const NAMED_BYSTANDER_NO_CONSUMER: &str = "module Repro.NoConsumer\n\
sig f: tensor[batch, 4, f32] -> tensor[batch, 1, f32] -> tensor[batch, 4, f32]\n\
def f(features: tensor[batch, 4, f32], mask: tensor[batch, 1, f32]) = \
expand(mask, 1i32, cast(shape(features, cast(1, int32)), int64))\n\
out = f(to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]]), \
to_tensor([[1.0f32], [2.0f32]]))\n";

/// The literal-size spelling, which never reached the broken arm.
const NAMED_BYSTANDER_LITERAL_SIZE: &str = "module Repro.LiteralSize\n\
sig f: tensor[batch, 4, f32] -> tensor[batch, 1, f32] -> tensor[batch, 4, f32]\n\
def f(features: tensor[batch, 4, f32], mask: tensor[batch, 1, f32]) = \
mul(features, expand(mask, 1i32, 4i64))\n\
out = f(to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]]), \
to_tensor([[1.0f32], [2.0f32]]))\n";

/// The rank-1 spelling, whose only axis is the expanded one, so no bystander
/// existed to trip the guard.
const NAMED_BYSTANDER_RANK_ONE: &str = "module Repro.Rank1\n\
sig f: tensor[4, f32] -> tensor[1, f32] -> tensor[4, f32]\n\
def f(features: tensor[4, f32], mask: tensor[1, f32]) = \
mul(features, expand(mask, 0i32, cast(shape(features, cast(0, int32)), int64)))\n\
out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), to_tensor([1.0f32]))\n";

/// expand.named_bystander.consumer.c, with its eval lane as the byte-identity
/// twin.
///
/// EVIDENTIARY STATUS: regression test on the C lane. On `6abca2406` the build
/// failed with ``ownership lowering invariant failed in `dag`: binary op at
/// node 5 has mismatched dimension at axis 1: Lit(4) vs Lit(1)``. The eval
/// assertion is a disposition lock: eval already printed this exact line, and
/// it is what the C lane now has to match.
#[test]
fn a_named_bystander_axis_keeps_its_size_extent_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let expected = "out = tensor(shape=[2, 4], data=[1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0])";
    let (ok, out) = c_run_result(&dir, "bystander_ax1_c", NAMED_BYSTANDER_AXIS_ONE);
    assert!(ok, "a check-clean program must build and run: {out}");
    assert!(out.contains(expected), "{out}");
    let (eval_ok, eval_out) = eval_result(&dir, "bystander_ax1_eval.ch", NAMED_BYSTANDER_AXIS_ONE);
    assert!(eval_ok, "{eval_out}");
    assert!(
        eval_out.contains(expected),
        "the lanes agree byte for byte: {eval_out}"
    );
}

/// expand.named_bystander.axis_zero.c: the issue's "axis 0 is unaffected" sentence
/// is wrong, and this row is the correction.
///
/// EVIDENTIARY STATUS: regression test. On `6abca2406` the build failed with the
/// same invariant naming axis 0 instead of axis 1, so the position of the
/// expanded axis was never the variable.
#[test]
fn a_named_bystander_axis_keeps_its_size_extent_on_axis_zero_too() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let expected = "out = tensor(shape=[4, 2], data=[1.0, 4.0, 3.0, 8.0, 5.0, 12.0, 7.0, 16.0])";
    let (ok, out) = c_run_result(&dir, "bystander_ax0_c", NAMED_BYSTANDER_AXIS_ZERO);
    assert!(ok, "{out}");
    assert!(out.contains(expected), "{out}");
    let (eval_ok, eval_out) = eval_result(&dir, "bystander_ax0_eval.ch", NAMED_BYSTANDER_AXIS_ZERO);
    assert!(eval_ok, "{eval_out}");
    assert!(eval_out.contains(expected), "{eval_out}");
}

/// expand.named_bystander.no_consumer.{c,eval}: the lane divergence.
///
/// EVIDENTIARY STATUS: regression test on the C lane, disposition lock on eval.
/// On `6abca2406` the C binary built and then aborted with ``extent `1`:
/// claimed = 1, features axis 1 = 4`` and `numeric trap: domain in load at
/// int64`, exit 134, while eval printed the correct value. With no binary
/// consumer nothing verified the prepared type, so the defect surfaced as a
/// divergence rather than as the build error the two rows above recorded.
#[test]
fn a_named_bystander_expand_with_no_consumer_agrees_across_lanes() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let expected = "out = tensor(shape=[2, 4], data=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])";
    let (ok, out) = c_run_result(&dir, "bystander_nocons_c", NAMED_BYSTANDER_NO_CONSUMER);
    assert!(ok, "the binary must not trap on its own broadcast: {out}");
    assert!(out.contains(expected), "{out}");
    assert!(
        !out.contains("numeric trap"),
        "and the trap the prepared type used to provoke is gone: {out}"
    );
    let (eval_ok, eval_out) = eval_result(
        &dir,
        "bystander_nocons_eval.ch",
        NAMED_BYSTANDER_NO_CONSUMER,
    );
    assert!(eval_ok, "{eval_out}");
    assert!(
        eval_out.contains(expected),
        "the lane that was already right is unchanged: {eval_out}"
    );
}

/// A reduction consumer and a declared FREE result dim on the expanded axis:
/// the two further lane divergences round 1 measured.
///
/// They vary the two things the rows above hold fixed. One replaces the
/// elementwise consumer with a `sum`, so the operand's wrong prepared extent
/// reaches a reduction rather than a binary op; the other declares the result
/// dim free instead of literal, so nothing downstream constrains the expanded
/// axis at all. Both had a correct interpreter answer and a trapping binary on
/// the base, which is the divergence the no-consumer row records for a third
/// shape.
///
/// EVIDENTIARY STATUS: regression tests on the C lane, disposition locks on
/// eval. Reverting `emit.rs` to `origin/main` and rerunning, both binaries
/// abort with ``extent `1`: claimed = 1, features axis 1 = 4`` and
/// `numeric trap: domain in load at int64` while eval prints these exact
/// values.
#[test]
fn a_reduction_consumer_and_a_free_result_dim_also_agree_across_lanes() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let cases = [
        (
            "bystander_reduction",
            "sig f: tensor[batch, 4, f32] -> tensor[batch, 1, f32] -> tensor[batch, f32]\n",
            "sum(expand(mask, 1i32, cast(shape(features, cast(1, int32)), int64)), 1)",
            "out = tensor(shape=[2], data=[4.0, 8.0])",
        ),
        (
            "bystander_free_result",
            "sig f: tensor[batch, 4, f32] -> tensor[batch, 1, f32] -> tensor[batch, m, f32]\n",
            "expand(mask, 1i32, cast(shape(features, cast(1, int32)), int64))",
            "out = tensor(shape=[2, 4], data=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])",
        ),
    ];
    for (stem, signature, body, expected) in cases {
        let source = format!(
            "{signature}\
             def f(features: tensor[batch, 4, f32], mask: tensor[batch, 1, f32]) = {body}\n\
             out = f(to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], \
             [5.0f32, 6.0f32, 7.0f32, 8.0f32]]), to_tensor([[1.0f32], [2.0f32]]))\n"
        );
        let (ok, out) = c_run_result(&dir, stem, &source);
        assert!(ok, "{stem} must not trap on its own broadcast: {out}");
        assert!(out.contains(expected), "{stem} on C: {out}");
        let (eval_ok, eval_out) = eval_result(&dir, &format!("{stem}.ch"), &source);
        assert!(
            eval_ok && eval_out.contains(expected),
            "{stem} on eval: {eval_out}"
        );
    }
}

/// The two spellings that never reached the broken arm, so the narrowed guard
/// must leave them exactly where they were.
///
/// EVIDENTIARY STATUS: disposition locks. Both built and ran correctly on
/// `6abca2406`; the literal size resolves without an axis source at all, and
/// the rank-1 form has no bystander axis to trip the guard.
#[test]
fn the_spellings_that_never_tripped_the_bystander_guard_stay_green() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "bystander_lit_c", NAMED_BYSTANDER_LITERAL_SIZE);
    assert!(ok, "{out}");
    assert!(
        out.contains(
            "out = tensor(shape=[2, 4], data=[1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0])"
        ),
        "{out}"
    );
    let (ok, out) = c_run_result(&dir, "bystander_rank1_c", NAMED_BYSTANDER_RANK_ONE);
    assert!(ok, "{out}");
    assert!(
        out.contains("out = tensor(shape=[4], data=[1.0, 2.0, 3.0, 4.0])"),
        "{out}"
    );
}

// ---------------------------------------------------------------------------
// chelis#1798: a declared result axis that PASSES THROUGH an op-computed
// extent.
//
// `preserve_declared_result` dispatched on the RESULT node's own axis source,
// so a claim was stamped only when the declared axis WAS the op-computed one.
// An `add` over two `shrink`s forwards its operand's axis
// (`AxisSource::InputAxis`), the literal and named witness arms walk those
// hops looking for an `ExtentWitness` an op-computed origin never has, and the
// op-computed arm refused the forwarded source outright: the claim was dropped
// and both lanes returned the extent the operation computed at exit zero.
//
// The repair resolves the declared axis through
// `axis_sources::op_computed_axis_origin`, which returns the `(node, axis)` of
// the operation that INTRODUCES the extent, and runs the op-computed arm
// against that origin. `runtime_extent_slice_b_sources.rs`'s origin-walk row
// names the two resolvers where they are tested and pins the difference
// between them; this header names only its own.
// `spec/04-type-system.md` section 4.7 puts the guard at "the source position
// of the operation that introduces the guarded extent", which is that origin,
// so the claim is stamped there as well as on the result and [04-NUM-9]'s
// `<op>` slot names `shrink`.
//
// Both operands of the `add` shrink the SAME extent in every fixture here.
// Only input 0's origin carries the stamp, and a second operand of a
// different extent would trap through the `add`'s own elementwise check with
// a different message, which would measure that check rather than this claim.
// ---------------------------------------------------------------------------

/// chelis#1798's reproducer, with `declared` naming the result extent and
/// `w_len` the extent of the parameter that declares `n`.
///
/// `x` and `y` both hold four elements and each `shrink` drops the first, so
/// the body produces 3 on every spelling below; `w_len` is what the claim
/// says it will be.
fn pass_through_op_computed_source(w_len: usize, declared: &str) -> String {
    format!(
        "def f(w: tensor[n, f32], x: tensor[r, f32], y: tensor[s, f32]) -> \
         tensor[{declared}, f32] = \
         add(shrink(x, [[1i64, shape(x, 0i32)]]), shrink(y, [[1i64, shape(y, 0i32)]]))\n\
         out = f({}, {}, {})\n",
        vector_literal(w_len),
        vector_literal(4),
        vector_literal(4)
    )
}

/// claim.named.pass_through.{eval,c}: the NAMED claim over a forwarded
/// op-computed extent is guarded at the `shrink` that introduces it.
///
/// One test for both lanes because the rendering is the same rendering: the
/// context line and [04-NUM-9]'s trap line are asserted on each lane
/// separately, and a lane that placed the guard elsewhere or named a
/// different operation would fail here rather than in a twin nobody compared
/// against.
///
/// EVIDENTIARY STATUS: regression test for the two mismatch blocks. Measured
/// at `6abca2406`, this exact program printed
/// `out = tensor(shape=[3], data=[4.0, 6.0, 8.0])` and exited ZERO on eval and
/// on the linked C binary, under a declared `tensor[n, f32]` with `n` = 2.
/// Disposition lock for the agreeing block, which exited zero there too and
/// must keep doing so: it is the non-vacuity control.
#[test]
fn a_pass_through_named_claim_is_guarded_at_its_op_computed_origin() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    let mismatched = pass_through_op_computed_source(2, "n");
    let (eval_ok, eval_out) = eval_result(&dir, "pass_through_named.ch", &mismatched);
    let (c_ok, c_out) = c_run_result(&dir, "pass_through_named_c", &mismatched);
    let context = "extent `n`: claimed = 2, shrink axis 0 = 3";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: a claim of 2 over a shrink of 3 traps: {out}");
        assert!(
            out.contains(&domain_trap_line("shrink")),
            "{lane}: [04-NUM-9]'s line names the origin operation: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
        assert!(
            !out.contains("shape=[3]"),
            "{lane}: and no undeclared extent is produced: {out}"
        );
    }

    // The agreeing control: `n` is 3 and the shrinks produce 3.
    let agreeing = pass_through_op_computed_source(3, "n");
    let (eval_ok, eval_out) = eval_result(&dir, "pass_through_named_ok.ch", &agreeing);
    let (c_ok, c_out) = c_run_result(&dir, "pass_through_named_ok_c", &agreeing);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: an agreeing claim executes: {out}");
        assert!(
            out.contains("out = tensor(shape=[3], data=[4.0, 6.0, 8.0])"),
            "{lane}: with the declared extent and the body's values: {out}"
        );
    }
}

/// claim.literal.pass_through.{eval,c}: the LITERAL half of the same class.
///
/// The issue reported the named spelling; the literal one is silent through
/// the same forwarded source for the same reason, and it reaches the guard
/// through the same origin resolution. Its claim renders as the number rather
/// than a binder, which is the only difference between the two rows.
///
/// EVIDENTIARY STATUS: regression test. Measured at `6abca2406`, this program
/// printed `out = tensor(shape=[3], data=[4.0, 6.0, 8.0])` and exited ZERO on
/// both lanes under a declared `tensor[2, f32]`.
#[test]
fn a_pass_through_literal_claim_is_guarded_at_its_op_computed_origin() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    let mismatched = pass_through_op_computed_source(2, "2");
    let (eval_ok, eval_out) = eval_result(&dir, "pass_through_lit.ch", &mismatched);
    let (c_ok, c_out) = c_run_result(&dir, "pass_through_lit_c", &mismatched);
    let context = "extent `2`: claimed = 2, shrink axis 0 = 3";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: a literal claim of 2 over 3 traps: {out}");
        assert!(
            out.contains(&domain_trap_line("shrink")),
            "{lane}: [04-NUM-9]'s line names the origin operation: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
    }

    // The agreeing literal control, so the row above pins a guard rather than
    // a lane that stopped executing this shape at all.
    let agreeing = pass_through_op_computed_source(2, "3");
    let (eval_ok, eval_out) = eval_result(&dir, "pass_through_lit_ok.ch", &agreeing);
    let (c_ok, c_out) = c_run_result(&dir, "pass_through_lit_ok_c", &agreeing);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: a literal claim of 3 over 3 executes: {out}");
        assert!(
            out.contains("out = tensor(shape=[3], data=[4.0, 6.0, 8.0])"),
            "{lane}: with the declared extent and the body's values: {out}"
        );
    }
}

/// claim.named.pass_through.inlined_root.{eval,c}: the same forwarded origin
/// reached through an INLINED root rather than a value binding.
///
/// `def main() -> tensor[2, f32] = f(...)` inlines `f`, so two claims reach
/// one origin: the callee's `n`, and the root's own literal 2. The claim
/// label is the BINDER, because chelis#1800's resolved stamp gives the inner
/// activation's `n` a number from the const argument that declares it, and
/// chelis#1782's rule then makes the root's restated literal entailed by that
/// named claim and records no second requirement. The origin, the axis and
/// the observed extent are the ones the value-binding rows report.
///
/// This row asserted ``extent `2` `` when the origin resolution landed on its
/// own, which is what the head produced before the resolved stamp: the const
/// argument minted no usable signature witness for the inlined activation, so
/// `f`'s named claim declined and the root's literal was the only one left.
/// The rendering moved with the stamp, in the same pull request, and the
/// measured value is what it asserts.
///
/// This row exists because the root form is where chelis#1782 and PR #1790
/// put the inlined claim, and a repair that only reached the exported-kernel
/// form would leave the root spelling of the same defect silent.
///
/// EVIDENTIARY STATUS: regression test. Measured at `6abca2406`, this program
/// printed `main = tensor(shape=[3], data=[4.0, 6.0, 8.0])` and exited ZERO on
/// both lanes.
#[test]
fn an_inlined_root_pass_through_claim_is_guarded_on_both_lanes() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    // `main` always claims 2, because the callee's own `n` is forced to 2 by
    // the const argument that declares it and the checker unifies the root's
    // declared result against that: `-> tensor[3, f32]` is a
    // `DimensionMismatch` rejection rather than a disagreeing root. The
    // agreeing control therefore moves the BODY instead, shrinking three
    // elements to two.
    let source = |operand_len: usize| {
        format!(
            "def f(w: tensor[n, f32], x: tensor[r, f32], y: tensor[s, f32]) -> \
             tensor[n, f32] = \
             add(shrink(x, [[1i64, shape(x, 0i32)]]), shrink(y, [[1i64, shape(y, 0i32)]]))\n\
             def main() -> tensor[2, f32] = f({}, {}, {})\n",
            vector_literal(2),
            vector_literal(operand_len),
            vector_literal(operand_len)
        )
    };

    let mismatched = source(4);
    let (eval_ok, eval_out) = eval_result(&dir, "pass_through_root.ch", &mismatched);
    let (c_ok, c_out) = c_run_result(&dir, "pass_through_root_c", &mismatched);
    let context = "extent `n`: claimed = 2, shrink axis 0 = 3";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: the root's claim of 2 over 3 traps: {out}");
        assert!(
            out.contains(&domain_trap_line("shrink")),
            "{lane}: [04-NUM-9]'s line names the origin operation: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
        assert!(
            !out.contains("extent `2`"),
            "{lane}: and the root's restated literal records no second \
             requirement (chelis#1782): {out}"
        );
    }

    // The agreeing root, so the row above pins a guard rather than a root
    // form that stopped producing a value.
    let agreeing = source(3);
    let (eval_ok, eval_out) = eval_result(&dir, "pass_through_root_ok.ch", &agreeing);
    let (c_ok, c_out) = c_run_result(&dir, "pass_through_root_ok_c", &agreeing);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: an agreeing root claim executes: {out}");
        assert!(
            out.contains("main = tensor(shape=[2], data=[4.0, 6.0])"),
            "{lane}: with the declared extent and the body's values: {out}"
        );
    }
}

// ---------------------------------------------------------------------------
// chelis#1837: a declared `concat`-axis extent over a symbolic operand.
//
// `tensor_concat_result_type` encodes a concat-axis extent it cannot compute
// as `Dim::Wildcard`, which `spec/04-type-system.md` §4.5.4 rule 3 makes the
// correct TYPE, and a surrounding declaration then narrows that wildcard to
// whatever it claims. §4.7.6 supplies the missing half - "A surrounding
// literal or named-dimension claim adds the runtime equality guard" - and
// [05-OP-62] states the rule the silence broke: "A joined unknown extent
// never licenses retaining the first element's extent without a guard". So
// the wildcard stays and the guard ends the admission; no spec amendment is
// owed.
//
// `concat` has no `RiscOp`. `tensor_concat_from_nodes` lowers a Pad+Add
// cascade, and [04-NUM-9] fixes which name the trap carries: "When a composed
// source operation lowers to that primitive, the trap SHALL retain the lowered
// primitive name; it SHALL NOT be renamed to the composed operation." The DAG
// path therefore names `pad`, the first Pad of the cascade being the add's
// axis-0 origin, and the host path names `concat`, which is the guard the
// return boundary already places. Each lane pair agrees per activation form,
// which is what the two rows below assert separately.
// ---------------------------------------------------------------------------

/// chelis#1837's reproducer in the inlined-root form, with `claim` naming the
/// declared concat-axis extent. Four rows joined to themselves produce eight.
fn concat_symbolic_root_source(claim: &str) -> String {
    let row = "[1.0f32, 2.0f32, 3.0f32]";
    let rows = [row; 4].join(", ");
    format!(
        "def probe(v: tensor[n, 3, f32]) -> tensor[{claim}, 3, f32] = concat([v, v], 0i32)\n\
         def main() -> tensor[{claim}, 3, f32] = probe(to_tensor([{rows}]))\n"
    )
}

/// concat.literal_claim.inlined_root.{eval,c}: chelis#1837's reproducer is
/// REJECTED when the activation is lowered, on both lanes, byte-identically.
///
/// `concat` has no `RiscOp`. `tensor_concat_from_nodes` lowers a Pad+Add
/// cascade, and it lowers one only when it can compute the joined extent from
/// concrete element extents: it writes `Lit(total)` on each `Pad`, so on this
/// path the cascade's padding bounds are always literal and the joined extent
/// is always a compile-time constant. An inlined root supplies concrete
/// arguments by construction, so this is the only shape the DAG path has.
///
/// A compile-time constant is exactly what `spec/04-type-system.md` §4.7.2's
/// guard does NOT cover: it conditions the check on a claim "that is not
/// statically proven equal to `size`". §4.7's first sentence covers it
/// instead, "A violation proven from literals is a type error", and
/// `the_checker_refuses_a_static_pad_extent_a_declaration_refutes` below shows
/// the checker already reporting that verdict wherever it can see the extent.
/// What the checker cannot see here is that `n` is 4: §3.2 types `probe(lit)`
/// by the signature, and `tensor_concat_result_type` publishes the joined
/// extent as `Dim::Wildcard` (§4.5.4 rule 3, the correct TYPE), which the
/// declaration then narrows. Lowering is the first stage holding both the
/// claim and the literals, so the rejection is raised there; chelis#526's
/// checker-tier repair for `n + n` is unchanged by it.
///
/// Why a rejection rather than the stamped claim the earlier disposition
/// declined to write. Stamping it produced, on the C lane,
/// `pad at node 1: output axis 0 has size 100, expected 8` from `verify`'s
/// per-owner static size check, refusing the build, while the DAG evaluator,
/// which does not run the verifier, trapped at run time. `verify` computes
/// the same arithmetic the claim contradicts, so the graph cannot STATE a
/// refuted claim at all. The repair is therefore to report it, not to record
/// it.
///
/// EVIDENTIARY STATUS: regression test on BOTH lanes. Measured at
/// `0820ee28e`, before this change, this program printed
/// `main = tensor(shape=[8, 3], ...)` and exited ZERO on eval and on the
/// linked C binary under a declared `tensor[100, 3, f32]`. This test replaces
/// the lock that pinned that disposition,
/// `a_declared_concat_axis_extent_on_the_dag_path_is_not_guarded_by_this_slice`.
#[test]
fn a_declared_concat_axis_extent_an_inlined_root_refutes_is_rejected_on_both_lanes() {
    assert!(
        gcc_available(),
        "this row compares two lanes and links its control; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    let claimed = concat_symbolic_root_source("100");
    both_lanes_reject_at_lowering(
        &dir,
        "concat_root",
        &claimed,
        &format!(
            "error: dimension mismatch: `probe` declares extent 100 at result axis 0, \
             but the inlined body produces 8 at source span `{}`",
            root_call_span(&claimed)
        ),
    );

    // The agreeing control: the true joined extent still executes exactly, so
    // the block above pins a refuted CLAIM rather than a refused shape.
    both_lanes_execute(
        &dir,
        "concat_root_ok",
        &concat_symbolic_root_source("8"),
        "shape=[8, 3]",
    );
}

/// chelis#1930's reproducer, with `claim` naming the declared extent of the
/// axis the `pad` leaves alone. Two rows of three, widened only on axis 1.
fn zero_padded_identity_axis_source(claim: &str) -> String {
    format!(
        "def f(x: tensor[rows, cols, f32], y: tensor[s, f32]) -> tensor[{claim}, 6, f32] = \
         pad(x, [[0i64, 0i64], [shape(y, 0i32), 1i64]], 0.0f32)\n\
         def main() -> tensor[{claim}, 6, f32] = \
         f(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]), \
         to_tensor([1.0f32, 2.0f32]))\n"
    )
}

/// pad.identity_axis.literal_claim.inlined_root.{eval,c}: chelis#1930.
///
/// The declared result has two axes and the operation widens only one. Axis 1
/// takes a RUNTIME padding bound (`shape(y, 0i32)`), so it remains §4.7.2's
/// guard case and nothing here touches it. Axis 0 is padded by
/// `[[0i64, 0i64]]`, which fixes it to the operand's own extent, so a
/// declaration claiming 9 over an operand of 2 is proven wrong. The rejection
/// names axis 0, and that is the row's point: an axis is not exempt from its
/// declaration merely because the operation left it alone.
///
/// EVIDENTIARY STATUS: regression test, watched failing differently on each
/// lane, which is why #1930 reads as two defects. Measured at `0820ee28e`:
/// eval printed `main = tensor(shape=[2, 6], ...)` at exit ZERO, while the C
/// build failed with ``ownership lowering invariant failed in `dag`: pad at
/// node 3: output axis 0 has size 9, expected 2`` - the IR verifier catching
/// the stamped claim and reporting it as an internal invariant rather than as
/// the user's declaration.
#[test]
fn a_zero_padded_identity_axis_a_declaration_refutes_is_rejected_on_both_lanes() {
    assert!(
        gcc_available(),
        "this row compares two lanes and links its control; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    let claimed = zero_padded_identity_axis_source("9");
    both_lanes_reject_at_lowering(
        &dir,
        "pad_identity_axis",
        &claimed,
        &format!(
            "error: dimension mismatch: `f` declares extent 9 at result axis 0, \
             but the inlined body produces 2 at source span `{}`",
            root_call_span(&claimed)
        ),
    );

    // The agreeing control. Axis 1's runtime bound still widens three to six,
    // so the executing half proves the rejection is about axis 0's claim and
    // not about the shape of the program.
    both_lanes_execute(
        &dir,
        "pad_identity_axis_ok",
        &zero_padded_identity_axis_source("2"),
        "shape=[2, 6]",
    );
}

/// A body that is its own argument, declared at `claim`. Two elements in.
fn identity_root_source(claim: &str) -> String {
    format!(
        "def f(x: tensor[rows, f32]) -> tensor[{claim}, f32] = x\n\
         def main() -> tensor[{claim}, f32] = f(to_tensor([1.0f32, 2.0f32]))\n"
    )
}

/// claim.literal.identity_root.{eval,c}: no operation at all, and the claim
/// is still refuted.
///
/// The body is the parameter. Its axis resolves straight through to the
/// caller's literal argument, so `ExtentOrigin::Literal` rather than
/// `OpComputed` is what proves the declaration wrong, and the row exists
/// because the two origins are separate arms of
/// `LowerCtx::graph_fixed_axis_extent`. It is also the smallest member of the
/// class: with no operation to attach a guard to, §4.7.2 had nothing to place
/// even in principle, which is why both lanes returned the undeclared extent
/// in silence.
///
/// EVIDENTIARY STATUS: regression test on BOTH lanes. Measured at
/// `0820ee28e`: `main = tensor(shape=[2], data=[1.0, 2.0])` at exit ZERO on
/// eval and on the linked C binary, under a declared `tensor[9, f32]`.
#[test]
fn an_identity_body_that_refutes_its_declared_extent_is_rejected_on_both_lanes() {
    assert!(
        gcc_available(),
        "this row compares two lanes and links its control; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    let claimed = identity_root_source("9");
    both_lanes_reject_at_lowering(
        &dir,
        "identity_root",
        &claimed,
        &format!(
            "error: dimension mismatch: `f` declares extent 9 at result axis 0, \
             but the inlined body produces 2 at source span `{}`",
            root_call_span(&claimed)
        ),
    );

    both_lanes_execute(
        &dir,
        "identity_root_ok",
        &identity_root_source("2"),
        "shape=[2]",
    );
}

/// A `pad` widening three elements by one, declared at `claim`, reached
/// through an inlined root rather than a value binding.
fn pad_inlined_root_source(claim: &str) -> String {
    format!(
        "def f(x: tensor[n, f32]) -> tensor[{claim}, f32] = pad(x, [[1i64, 0i64]], 0.0f32)\n\
         def main() -> tensor[{claim}, f32] = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
    )
}

/// pad.literal_claim.inlined_root.{eval,c}: the lane divergence chelis#1911
/// recorded as residual, closed.
///
/// This is `a_non_zero_pad_extent_is_guarded_on_both_lanes`'s program in the
/// INLINED-ROOT activation form rather than the value-binding one, and the
/// form is the whole difference. A value binding stages its argument across
/// the host boundary, so the `pad`'s operand axis is an `ExternalAxis` and the
/// extent is a runtime value: §4.7.2's guard owns it, and that row still
/// renders ``extent `2`: claimed = 2, pad axis 0 = 5`` on both lanes.
/// An inlined root hands the operation a literal, so the same declaration
/// becomes provably wrong and changes tiers.
///
/// EVIDENTIARY STATUS: regression test, watched failing differently on each
/// lane. Measured at `0820ee28e`: eval trapped with ``extent `2`: claimed =
/// 2, pad axis 0 = 4`` - a guard, but for a comparison that could never hold -
/// while the C build failed with ``ownership lowering invariant failed in
/// `dag`: pad at node 1: output axis 0 has size 2, expected 4``. #1911's
/// "every activation form" sentence names this divergence; this row is its
/// receipt.
#[test]
fn a_literal_pad_claim_an_inlined_root_refutes_is_rejected_on_both_lanes() {
    assert!(
        gcc_available(),
        "this row compares two lanes and links its control; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    let claimed = pad_inlined_root_source("2");
    both_lanes_reject_at_lowering(
        &dir,
        "pad_inlined_root",
        &claimed,
        &format!(
            "error: dimension mismatch: `f` declares extent 2 at result axis 0, \
             but the inlined body produces 4 at source span `{}`",
            root_call_span(&claimed)
        ),
    );

    both_lanes_execute(
        &dir,
        "pad_inlined_root_ok",
        &pad_inlined_root_source("4"),
        "shape=[4]",
    );
}

/// A result claimed by the BINDER `n`, which `w`'s literal argument resolves.
/// `x` is joined to itself, so the body produces six.
fn resolved_named_claim_source(w: &str, root: &str) -> String {
    format!(
        "def g(w: tensor[n, f32], x: tensor[rows, f32]) -> tensor[n, f32] = concat([x, x], 0i32)\n\
         def main() -> tensor[{root}, f32] = \
         g(to_tensor([{w}]), to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
    )
}

/// claim.named.resolved.inlined_root.{eval,c}: a NAMED claim is refutable too,
/// once chelis#1800 has resolved its declaring witness to a number.
///
/// `g` claims its result is `n`, and nothing in the graph relates `n` to the
/// concat, which is precisely why §4.7.2 makes a named claim an execution-time
/// equality. An inlined root changes what is known rather than what is
/// claimed: `w`'s argument fixes `n` to 4, the cascade fixes the result to 6,
/// and `4 == 6` is now a comparison the graph has already decided. The
/// diagnostic reports the binder beside its resolved value, because a reader
/// told only "extent 4" would go looking for a 4 the signature never spells.
///
/// The unresolved form is the control below and must stay a guard: it is the
/// ordinary case, where `n`'s declaring argument is a runtime value and §4.7.2
/// owns the comparison.
///
/// EVIDENTIARY STATUS: regression test on BOTH lanes. Measured at
/// `0820ee28e`: `main = tensor(shape=[6], data=[1.0, 2.0, 3.0, 1.0, 2.0,
/// 3.0])` at exit ZERO on eval and on the linked C binary, under a declared
/// `tensor[n, f32]` with `n` fixed to 4.
#[test]
fn a_resolved_named_claim_the_inlined_body_refutes_is_rejected_on_both_lanes() {
    assert!(
        gcc_available(),
        "this row compares two lanes and links its control; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    let claimed = resolved_named_claim_source("1.0f32, 2.0f32, 3.0f32, 4.0f32", "4");
    both_lanes_reject_at_lowering(
        &dir,
        "named_resolved",
        &claimed,
        &format!(
            "error: dimension mismatch: `g` declares extent `n` = 4 at result axis 0, \
             but the inlined body produces 6 at source span `{}`",
            root_call_span(&claimed)
        ),
    );

    // The agreeing control: `n` resolved to the extent the body does produce.
    both_lanes_execute(
        &dir,
        "named_resolved_ok",
        &resolved_named_claim_source("1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32", "6"),
        "shape=[6]",
    );
}

/// The same claim reached through a PIPE. Its activation carried no name
/// until chelis#1923 folded the stage into the application it denotes.
fn piped_root_source(claim: &str) -> String {
    format!(
        "def f(x: tensor[rows, f32]) -> tensor[{claim}, f32] = x\n\
         def main() -> tensor[{claim}, f32] = to_tensor([1.0f32, 2.0f32]) |> f\n"
    )
}

/// The direct spelling the piped one folds to, byte for byte the same
/// program to the checker and the lowerer.
fn direct_root_source(claim: &str) -> String {
    format!(
        "def f(x: tensor[rows, f32]) -> tensor[{claim}, f32] = x\n\
         def main() -> tensor[{claim}, f32] = f(to_tensor([1.0f32, 2.0f32]))\n"
    )
}

/// The `span_id` a folded pipe stage carries: the stage name itself, which is
/// the text the author wrote for that activation.
///
/// Computed from the fixture for the same reason `root_call_span` is, so
/// editing the program cannot leave a stale byte range asserted.
fn pipe_stage_span(source: &str) -> String {
    let body = source.trim_end();
    let start = body.rfind("|> ").expect("a pipe stage follows `|> `") + 3;
    format!("surf:{start}..{}", body.len())
}

/// The same claim on a root whose body is a `vmap`, which lowers through the
/// resolved-body boundary rather than through a named call.
fn vmapped_root_source(claim: &str) -> String {
    format!(
        "def f(x: tensor[rows, f32]) -> tensor[9, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n\
         def main() -> tensor[2, {claim}, f32] = \
         vmap(f)(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n"
    )
}

/// claim.literal.nameless_activation.{eval,c}: the rejection still reports
/// when the lowerer has no name for the activation that made the claim.
///
/// Only `lower_plain_callable_app` holds a callee name (`inlining_name`, and
/// even there it is `Option`). An AD, vectorization or host-list boundary
/// reaches `lower_resolved_body` with a `ResolvedFunction` that carries a
/// signature and no name, and a root applied host-side has none either. The
/// diagnostic says `the signature` in that slot rather than omitting the
/// clause, so the sentence does not change shape with the spelling that
/// reached it. The `vmap` half below is that witness.
///
/// chelis#1923 took the PIPE STAGE off that list, which is why the piped half
/// reads differently here than it did when this row landed. A pipe stage is
/// folded into the application it denotes before the checker runs, so it now
/// reaches `lower_plain_callable_app` with the callee's name in hand and the
/// rejection says `f` rather than `the signature`. That is the better
/// diagnostic and it is the one the DIRECT spelling of the same program has
/// always given, so the piped half is now an agreement lock between the two
/// spellings: same sentence, each pointing at its own source text. The span
/// differs because the folded application carries the STAGE's span, which is
/// the `f` the author wrote, where the direct spelling carries the whole
/// call.
///
/// The `vmap` half also pins a residual rather than a feature: its rejection
/// carries NO source span, because `current_span_id` is unset at that
/// boundary. That is pre-existing `LowerDiagnostic` behaviour and is asserted
/// here so the gap is visible instead of being discovered as a surprise.
///
/// EVIDENTIARY STATUS: regression test on BOTH lanes, for both spellings.
/// Measured at `0820ee28e`: the piped root printed
/// `main = tensor(shape=[2], data=[1.0, 2.0])` and the vmapped root
/// `main = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])`, each at exit
/// ZERO on eval and on the linked C binary, under a declared extent of 9.
/// The piped half's rendering was re-measured at this head, where it names
/// `f`; on `23729c638` it said `the signature` and carried the whole call's
/// span.
#[test]
fn a_nameless_activation_that_refutes_its_own_claim_is_rejected_on_both_lanes() {
    assert!(
        gcc_available(),
        "this row compares two lanes and links its controls; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    let piped = piped_root_source("9");
    both_lanes_reject_at_lowering(
        &dir,
        "piped_root",
        &piped,
        &format!(
            "error: dimension mismatch: `f` declares extent 9 at result axis 0, \
             but the inlined body produces 2 at source span `{}`",
            pipe_stage_span(&piped)
        ),
    );

    // The direct spelling of the same program, which is what the piped one
    // now folds to. Same sentence, its own span: the agreement chelis#1923
    // exists to produce, asserted here rather than assumed.
    let direct = direct_root_source("9");
    both_lanes_reject_at_lowering(
        &dir,
        "direct_root",
        &direct,
        &format!(
            "error: dimension mismatch: `f` declares extent 9 at result axis 0, \
             but the inlined body produces 2 at source span `{}`",
            root_call_span(&direct)
        ),
    );

    // The `vmap` boundary, which has no span to report.
    both_lanes_reject_at_lowering(
        &dir,
        "vmapped_root",
        &vmapped_root_source("9"),
        "error: dimension mismatch: the signature declares extent 9 at result axis 1, \
         but the inlined body produces 2",
    );

    both_lanes_execute(&dir, "piped_root_ok", &piped_root_source("2"), "shape=[2]");
    both_lanes_execute(
        &dir,
        "vmapped_root_ok",
        &vmapped_root_source("2"),
        "shape=[2, 2]",
    );
}

/// claim.literal.kernel_entry.checker: a LITERAL parameter extent keeps the
/// checker's verdict, and lowering never sees the program.
///
/// This is the control for where the rejection does NOT belong. A standalone
/// or exported kernel entry reads its parameters from `Load` nodes, whose axes
/// resolve to `ExtentOrigin::ExternalAxis` and fix nothing. Declare those
/// extents literally and the operands become compile-time constants, which
/// would make the kernel entry refutable too - except that the checker then
/// computes the body's own result type from the same literals and refuses the
/// signature first. `tensor_concat_result_type`'s wildcard, the reason the
/// checker cannot do this for the rows above, does not survive concrete
/// element extents.
///
/// So the two tiers partition the class rather than overlapping on it, and
/// that is what this row asserts: the verdict here names the DEF and its two
/// function types, which is the checker's rendering, and never the lowering
/// diagnostic's.
///
/// EVIDENTIARY STATUS: disposition lock. Existing checker behaviour that this
/// change neither introduces nor alters, recorded so a later reader can see
/// that the kernel-entry call sites were measured rather than assumed.
#[test]
fn a_literal_parameter_extent_keeps_the_checkers_verdict_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let row = "[1.0f32, 2.0f32, 3.0f32]";
    let rows = [row; 4].join(", ");
    let source = format!(
        "def probe(v: tensor[4, 3, f32]) -> tensor[100, 3, f32] = concat([v, v], 0i32)\n\
         out = probe(to_tensor([{rows}]))\n"
    );

    let (eval_ok, eval_out) = eval_result(&dir, "kernel_lit.ch", &source);
    let build = build_c(
        &fixture(&dir, "kernel_lit_c.ch", &source),
        &dir.path().join("kernel_lit_c-out"),
    );
    let c_out = String::from_utf8_lossy(&build.stderr).to_string();
    for (lane, ok, out) in [
        ("eval", eval_ok, &eval_out),
        ("c", build.status.success(), &c_out),
    ] {
        assert!(
            !ok,
            "{lane}: a literal-parameter refutation is refused: {out}"
        );
        assert!(
            out.contains("DimensionMismatch")
                && out.contains(
                    "def 'probe' body doesn't match declared signature: body has type \
                     `(tensor[4, 3, f32]) -> tensor[8, 3, f32]`, declared type is \
                     `(tensor[4, 3, f32]) -> tensor[100, 3, f32]`"
                ),
            "{lane}: and the CHECKER reports it, naming both function types: {out}"
        );
        assert!(
            !out.contains("the inlined body produces"),
            "{lane}: lowering is never reached, so its diagnostic must not appear: {out}"
        );
    }
}

/// The tier boundary the row above turns on: a STATIC extent a declaration
/// refutes is a check-time dimension mismatch, not a runtime guard.
///
/// This is the same `pad` owner and the same disagreement as
/// `a_non_zero_pad_extent_is_guarded_on_both_lanes`, with the operand's extent
/// declared literally instead of symbolically. `spec/04-type-system.md`
/// §4.7.2 conditions its guard on a claim "that is not statically proven equal
/// to `size`", and this one is statically refuted, so the verdict belongs to
/// the tier that can report one. Without this row the concat lock reads as an
/// unexplained gap rather than as a wildcard hiding an extent the checker
/// would otherwise refuse.
///
/// EVIDENTIARY STATUS: disposition lock. This is existing checker behaviour
/// that this change neither introduces nor alters; the row exists so that the
/// static and runtime halves of §4.7.2 stay visibly separate.
#[test]
fn the_checker_refuses_a_static_pad_extent_a_declaration_refutes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[4, f32]) -> tensor[2, f32] = pad(x, [[1i64, 1i64]], 0.0f32)\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\n";
    let (ok, out) = eval_result(&dir, "pad_static_refuted.ch", source);
    assert!(!ok, "a statically refuted claim does not execute: {out}");
    assert!(
        out.contains("DimensionMismatch")
            && out.contains("body has type `(tensor[4, f32]) -> tensor[6, f32]`")
            && out.contains("declared type is `(tensor[4, f32]) -> tensor[2, f32]`"),
        "and the checker, not a runtime guard, reports it: {out}"
    );
    assert!(
        !out.contains(&domain_trap_line("pad")),
        "so no [04-NUM-9] trap is rendered for it: {out}"
    );
}

/// concat.literal_claim.host.{eval,c}: the host path keeps naming `concat`.
///
/// [04-NUM-9]'s composed-source rule is about the operation the TRAP names,
/// and the two activation forms reach two different guards: the value binding
/// applies the exported kernel, whose chelis#1739 return guard compares the
/// declared literal at the boundary and names the composed operation the user
/// wrote. This row pins that rendering so the row above cannot be read as
/// having renamed every `concat` trap to `pad`.
///
/// EVIDENTIARY STATUS: disposition lock on both lanes. Measured at
/// `a5fee66b9`, this program already printed
/// ``extent `100`: claimed = 100, concat axis 0 = 8`` and
/// `numeric trap: domain in concat at int64`, exiting 1 on eval and 134 on the
/// linked binary. This change neither introduces nor alters it.
#[test]
fn the_host_path_concat_claim_still_names_concat_on_both_lanes() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let row = "[1.0f32, 2.0f32, 3.0f32]";
    let rows = [row; 4].join(", ");
    let source = format!(
        "def probe(v: tensor[n, 3, f32]) -> tensor[100, 3, f32] = concat([v, v], 0i32)\n\
         m = to_tensor([{rows}])\n\
         out = probe(m)\n"
    );
    let (eval_ok, eval_out) = eval_result(&dir, "concat_host.ch", &source);
    let (c_ok, c_out) = c_run_result(&dir, "concat_host_c", &source);
    let context = "extent `100`: claimed = 100, concat axis 0 = 8";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: the host path traps: {out}");
        assert!(
            out.contains(&domain_trap_line("concat")),
            "{lane}: naming the composed operation the user wrote: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
    }
}

/// A caller-established result label and an op-computed origin stamp coexist.
///
/// This row carries no corpus id: the behaviour is already correct on this
/// branch's base, so it has no start state to move from. It is registered in
/// the target manifest as a lock, beside the other four this pull request
/// adds.
///
/// chelis#1889's repair (PR #1895) transports a checked caller-side axis name
/// onto an already-lowered helper result, and its rule is that "the name is
/// never interpreted in the callee binder namespace". The resolved stamp this
/// slice adds writes into the same result type from the callee side, so the
/// two could have raced: if the relabel renamed the claim the stamp had just
/// made, the class keyed by the callee binder would lose its declaring witness
/// and the guard with it.
///
/// It does not, and the reason is that the two act on different axes. The
/// callee binder here is declared by a parameter whose extent is the CALLER's,
/// so it does not resolve to a graph-fixed value and the stamp leaves it
/// unresolved; the relabel then renames the result axis to the caller's
/// spelling, the declaring witness is the caller's own parameter, and the
/// class forms under that spelling. The trap reports the caller's name, which
/// is the name the user wrote at the boundary that failed.
///
/// This row exists because the interaction is invisible in either change
/// alone. None of PR #1895's own `compiled_context` programs reaches an
/// op-computed result: its helpers are `mul`, `copy` and
/// `insert(sum(gain, fixed), ..)` bodies, and its mismatch control is a `copy`
/// of a literal-extent tensor, which takes the witness arm rather than the
/// op-computed one.
///
/// EVIDENTIARY STATUS: disposition lock on both lanes. Measured at
/// `a5fee66b9`, this branch's base, which already carries PR #1895: this
/// program printed ``extent `fixed`: claimed = 2, shrink axis 0 = 3`` and
/// trapped there, and prints the same at this head. The row pins that the
/// resolved stamp did not move the reported name, and it fails on a tree where
/// the stamp overwrites a transported label.
#[test]
fn a_caller_transported_result_label_survives_the_op_computed_stamp() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let source = |gain: usize| {
        format!(
            "def narrow[d](x: tensor[r, f32], gain: tensor[d, f32]) -> tensor[d, f32] = \
             shrink(x, [[1i64, shape(x, 0i32)]])\n\
             def caller(g: tensor[fixed, f32], y: tensor[r, f32]) -> tensor[fixed, f32] = \
             narrow(y, g)\n\
             out = caller({}, {})\n",
            vector_literal(gain),
            vector_literal(4)
        )
    };

    let mismatched = source(2);
    let (eval_ok, eval_out) = eval_result(&dir, "checked_label.ch", &mismatched);
    let (c_ok, c_out) = c_run_result(&dir, "checked_label_c", &mismatched);
    let context = "extent `fixed`: claimed = 2, shrink axis 0 = 3";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(
            !ok,
            "{lane}: the transported claim of 2 over 3 traps: {out}"
        );
        assert!(
            out.contains(&domain_trap_line("shrink")),
            "{lane}: [04-NUM-9]'s line names the origin operation: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
        assert!(
            !out.contains("extent `d`"),
            "{lane}: and the callee binder is not interpreted in the caller's \
             namespace (chelis#1889): {out}"
        );
    }

    // The agreeing control: the caller's gain is 3 and the shrink produces 3.
    let agreeing = source(3);
    let (eval_ok, eval_out) = eval_result(&dir, "checked_label_ok.ch", &agreeing);
    let (c_ok, c_out) = c_run_result(&dir, "checked_label_ok_c", &agreeing);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: an agreeing transported claim executes: {out}");
        assert!(
            out.contains("out = tensor(shape=[3], data=[2.0, 3.0, 4.0])"),
            "{lane}: with the produced extent: {out}"
        );
    }
}

// ---------------------------------------------------------------------------
// Round 1's P1: a runtime padding bound.
//
// Admitting `pad` in `op_computed_axis_extent` made lowering stamp the claim,
// but `host::rank_preserving_movement_type`'s Pad arm then discarded it. Its
// runtime case minted a fresh `_rt_pad_dim_N_A` unconditionally, where the
// sibling Shrink arm asks `op_computed_axis_extent` first and routes through
// `unresolved_axis`, which keeps the declared dim from the fallback so a class
// can form. The stamp survived the inlined root, whose root type is not
// actualized through that function, and died in the exported and
// value-binding forms.
//
// So the rows below are keyed on the BOUND rather than on the operand:
// `spec/04-type-system.md` section 4.7 names "a non-identity literal step or
// padding, or any runtime bound" as the movement axes that type a fresh extent
// the owning operation "declares and guards at run time", and a runtime bound
// is the half this pull request admitted without guarding.
// ---------------------------------------------------------------------------

/// A `pad` whose BEFORE bound is read at run time, with `claim` naming the
/// declared result extent. Three elements padded by two then one is six.
fn runtime_pad_before_source(claim: &str) -> String {
    format!(
        "def f(x: tensor[rows, f32], y: tensor[s, f32]) -> tensor[{claim}, f32] = \
         pad(x, [[shape(y, 0i32), 1i64]], 0.0f32)\n\
         out = f({}, {})\n",
        vector_literal(3),
        vector_literal(2)
    )
}

/// pad.runtime_bound_claim.{eval,c}: a runtime padding bound is guarded in the
/// value-binding form as well as the inlined root.
///
/// The two activation forms are asserted in one test because the defect was
/// exactly their DIVERGENCE: the root trapped while the value binding did not,
/// so a row that measured either form alone would have passed.
///
/// EVIDENTIARY STATUS: regression test. Measured at `779c46626`, this pull
/// request's own round-1 head, the value-binding form printed
/// `out = tensor(shape=[6], data=[0.0, 0.0, 1.0, 2.0, 3.0, 0.0])` and exited
/// ZERO on eval and on the linked C binary under a declared `tensor[2, f32]`,
/// while the inlined root trapped. The emitted C carried
/// `int64_t _rt_pad_dim_3_0 = chelis_movement_extent(...)` and no comparison.
/// Disposition lock for the agreeing block.
#[test]
fn a_runtime_bound_pad_claim_is_guarded_in_every_activation_form() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let context = "extent `2`: claimed = 2, pad axis 0 = 6";

    // The value-binding form, which applies the exported kernel.
    let binding = runtime_pad_before_source("2");
    let (eval_ok, eval_out) = eval_result(&dir, "rt_pad_binding.ch", &binding);
    let (c_ok, c_out) = c_run_result(&dir, "rt_pad_binding_c", &binding);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: a claim of 2 over a pad to 6 traps: {out}");
        assert!(
            out.contains(&domain_trap_line("pad")),
            "{lane}: [04-NUM-9]'s line names the `pad`: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
        assert!(
            !out.contains("shape=[6]"),
            "{lane}: and no undeclared extent is produced: {out}"
        );
    }

    // The inlined root over the same body, which trapped even before the
    // repair. Both forms must agree, which is the property this row adds.
    let root = binding.replace("out = f(", "def main() -> tensor[2, f32] = f(");
    let (eval_ok, eval_out) = eval_result(&dir, "rt_pad_root.ch", &root);
    let (c_ok, c_out) = c_run_result(&dir, "rt_pad_root_c", &root);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: the inlined root traps too: {out}");
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
    }

    // The agreeing control: the true padded extent still executes exactly, so
    // the blocks above pin a guard rather than a lowering that stopped working.
    let agreeing = runtime_pad_before_source("6");
    let (eval_ok, eval_out) = eval_result(&dir, "rt_pad_ok.ch", &agreeing);
    let (c_ok, c_out) = c_run_result(&dir, "rt_pad_ok_c", &agreeing);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(
            ok,
            "{lane}: an agreeing runtime-bound claim executes: {out}"
        );
        assert!(
            out.contains("out = tensor(shape=[6], data=[0.0, 0.0, 1.0, 2.0, 3.0, 0.0])"),
            "{lane}: with the padded extent and the fill: {out}"
        );
    }
}

/// pad.runtime_after_and_named.{eval,c}: the AFTER bound and a NAMED claim
/// reach the same guard.
///
/// Two shapes in one row because they vary one factor each against the row
/// above: which side of the padding is read at run time, and whether the claim
/// is a literal or a binder. Both were silent before the repair, and a repair
/// that only reached the `before` bound or only the literal claim would fail
/// here.
///
/// EVIDENTIARY STATUS: regression test for both blocks. Measured at
/// `779c46626`, each printed `shape=[6]` and exited ZERO on both lanes.
#[test]
fn a_runtime_after_bound_and_a_named_pad_claim_reach_the_same_guard() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");

    // The runtime bound on the other side of the padding.
    let after = format!(
        "def f(x: tensor[rows, f32], y: tensor[s, f32]) -> tensor[2, f32] = \
         pad(x, [[1i64, shape(y, 0i32)]], 0.0f32)\n\
         out = f({}, {})\n",
        vector_literal(3),
        vector_literal(2)
    );
    let (eval_ok, eval_out) = eval_result(&dir, "rt_pad_after.ch", &after);
    let (c_ok, c_out) = c_run_result(&dir, "rt_pad_after_c", &after);
    let context = "extent `2`: claimed = 2, pad axis 0 = 6";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: a runtime `after` bound is guarded too: {out}");
        assert!(
            out.contains(&domain_trap_line("pad")),
            "{lane}: [04-NUM-9]'s line names the `pad`: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
    }

    // The NAMED claim over the same runtime-bound pad, declared by a parameter
    // the body never reads.
    let named = format!(
        "def f(w: tensor[n, f32], x: tensor[rows, f32], y: tensor[s, f32]) -> \
         tensor[n, f32] = pad(x, [[shape(y, 0i32), 1i64]], 0.0f32)\n\
         out = f({}, {}, {})\n",
        vector_literal(2),
        vector_literal(3),
        vector_literal(2)
    );
    let (eval_ok, eval_out) = eval_result(&dir, "rt_pad_named.ch", &named);
    let (c_ok, c_out) = c_run_result(&dir, "rt_pad_named_c", &named);
    let context = "extent `n`: claimed = 2, pad axis 0 = 6";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: a named runtime-bound claim is guarded: {out}");
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
    }
}

/// pad.runtime_bound_axis1.{eval,c}: a rank-2 `pad` guards the axis it widens
/// and reports that axis, not axis 0.
///
/// The axis index travels from `ComputedAxisExtent::PadSpan`'s `operand_axis`
/// through the site key to the rendering, and a repair that hard-coded axis 0
/// or keyed the site on the wrong axis would pass every rank-1 row above. The
/// kept axis is zero-padded, so it stays a pass-through and forms no site of
/// its own.
///
/// EVIDENTIARY STATUS: regression test. Measured at `779c46626`, this program
/// printed `out = tensor(shape=[2, 6], ...)` and exited ZERO on both lanes
/// under a declared `tensor[rows, 2, f32]`.
#[test]
fn a_rank_two_pad_guards_and_reports_the_runtime_axis_it_widens() {
    assert!(
        gcc_available(),
        "this row compares two executed lanes; neither may skip"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let source = |claim: &str| {
        format!(
            "def f(x: tensor[rows, 3, f32], y: tensor[s, f32]) -> \
             tensor[rows, {claim}, f32] = \
             pad(x, [[0i64, 0i64], [shape(y, 0i32), 1i64]], 0.0f32)\n\
             out = f(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]), {})\n",
            vector_literal(2)
        )
    };

    let mismatched = source("2");
    let (eval_ok, eval_out) = eval_result(&dir, "rt_pad_axis1.ch", &mismatched);
    let (c_ok, c_out) = c_run_result(&dir, "rt_pad_axis1_c", &mismatched);
    let context = "extent `2`: claimed = 2, pad axis 1 = 6";
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: the widened axis is guarded: {out}");
        assert!(
            out.contains(&domain_trap_line("pad")),
            "{lane}: [04-NUM-9]'s line names the `pad`: {out}"
        );
        assert!(out.contains(context), "{lane}: expected {context}: {out}");
        assert!(
            !out.contains("pad axis 0"),
            "{lane}: and the guard names the axis it widens, not axis 0: {out}"
        );
    }

    let agreeing = source("6");
    let (eval_ok, eval_out) = eval_result(&dir, "rt_pad_axis1_ok.ch", &agreeing);
    let (c_ok, c_out) = c_run_result(&dir, "rt_pad_axis1_ok_c", &agreeing);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: the agreeing rank-2 claim executes: {out}");
        assert!(
            out.contains("shape=[2, 6]"),
            "{lane}: keeping the zero-padded axis and widening the other: {out}"
        );
    }
}

// ---------------------------------------------------------------------------
// chelis#1801: a root whose extent arrives through a nested helper's claim.
// ---------------------------------------------------------------------------

/// `g` resolves its declared result extent only at run time, and `h` claims
/// one extent for its parameter and its result. On the base sha `h`'s
/// instantiation variable met `g`'s runtime extent, `unify_dim` left it free,
/// and def-level generalization quantified it, so `main` checked as `() ->
/// tensor[d0, f32]`: a root with no ABI, which both lanes dropped in silence.
/// `spec/04-type-system.md` section 3.2 now makes that variable denote the
/// extent it met.
const NESTED_HELPER_ROOT: &str = "def g(x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
     def h(y: tensor[k, f32]) -> tensor[k, f32] = add(y, y)\n\
     def main() = h(g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n";

/// The rendering both lanes owe. `g` keeps the last two of three elements and
/// `h` doubles them.
const NESTED_HELPER_ROOT_RENDERING: &str = "main = tensor(shape=[2], data=[4.0, 6.0])";

/// Receipt for corpus row `root.dim_variable.nested_helper.eval`.
///
/// Regression test. On the base sha `ae9260727` this program evaluated to
/// nothing: eval printed the `input contains only def declarations; nothing to
/// evaluate` warning on stderr, nothing on stdout, and exited 0.
#[test]
fn a_nested_helper_claim_sizes_a_root_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "nested_helper_root.ch", NESTED_HELPER_ROOT);

    let evaluated = eval(&path);
    let stderr = String::from_utf8_lossy(&evaluated.stderr).to_string();
    assert!(evaluated.status.success(), "{stderr}");
    assert!(
        !stderr.contains("nothing to evaluate"),
        "the root is realizable and must not be dropped: {stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&evaluated.stdout).trim_end(),
        NESTED_HELPER_ROOT_RENDERING,
        "the root is sized from the extent `h`'s claim absorbed"
    );
}

/// Receipt for corpus row `root.dim_variable.nested_helper.c`.
///
/// Regression test. On the base sha the emitted translation unit contained no
/// `int main(`, so the program linked to an object with no entry point and
/// the lane reported success having run nothing. The C rendering is asserted
/// byte-identical to the eval one above.
#[test]
fn a_nested_helper_claim_sizes_a_root_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out, emitted) =
        c_run_result_with_source(&dir, "nested_helper_root_c", NESTED_HELPER_ROOT);
    assert!(
        emitted.contains("int main("),
        "a realizable root owes a C entry point:\n{emitted}"
    );
    assert!(ok, "the sized root must build, link and run: {out}");
    assert_eq!(
        out.trim_end(),
        NESTED_HELPER_ROOT_RENDERING,
        "C renders the root exactly as eval does"
    );
}

/// The same root reached through a POLYMORPHIC named def passed as an
/// argument, which is chelis#1925's round-1 P1.
///
/// `apply1` mints its own dimension variable for the shared extent and `h`
/// mints one too, while the ARGUMENT is inferred rather than the callee, so
/// unification makes `h`'s the alias class's root. Absorbing the variable the
/// application itself minted left that root free, the result kept a quantified
/// dimension, and the root was dropped exactly as in the row above. The
/// absorption reaches the class rather than the one variable.
const POLYMORPHIC_HELPER_ROOT: &str = "def g(x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
     def h(y: tensor[k, f32]) -> tensor[k, f32] = add(y, y)\n\
     def apply1(f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = f(v)\n\
     def main() = apply1(h, g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n";

/// Receipt for corpus row `root.dim_variable.polymorphic_argument.eval`.
///
/// Regression test. On the base sha AND on this change's first head
/// `f2238550d` this evaluated to nothing: `main :: () -> tensor[d0, f32]`,
/// the `nothing to evaluate` warning on stderr, empty stdout, exit 0.
#[test]
fn a_polymorphic_argument_claim_sizes_a_root_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "polymorphic_helper_root.ch", POLYMORPHIC_HELPER_ROOT);

    let evaluated = eval(&path);
    let stderr = String::from_utf8_lossy(&evaluated.stderr).to_string();
    assert!(evaluated.status.success(), "{stderr}");
    assert!(
        !stderr.contains("nothing to evaluate"),
        "the root is realizable and must not be dropped: {stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&evaluated.stdout).trim_end(),
        NESTED_HELPER_ROOT_RENDERING,
        "the root is sized from the extent the alias class absorbed"
    );
}

/// Receipt for corpus row `root.dim_variable.polymorphic_argument.c`.
///
/// Regression test. On the base sha and at `f2238550d` the emitted translation
/// unit contained no `int main(`, so the lane reported success having run
/// nothing. The rendering is byte-identical to the eval one above and to the
/// simpler row's.
#[test]
fn a_polymorphic_argument_claim_sizes_a_root_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out, emitted) =
        c_run_result_with_source(&dir, "polymorphic_helper_root_c", POLYMORPHIC_HELPER_ROOT);
    assert!(
        emitted.contains("int main("),
        "a realizable root owes a C entry point:\n{emitted}"
    );
    assert!(ok, "the sized root must build, link and run: {out}");
    assert_eq!(
        out.trim_end(),
        NESTED_HELPER_ROOT_RENDERING,
        "C renders the root exactly as eval does"
    );
}

/// The same root whose extent a RESULT-ONLY binder names: `outer` declares
/// `-> tensor[seq, f32]` and no parameter binds `seq`, so nothing supplies a
/// value for that name. chelis#1925's round-1 verification found the LANE
/// DIVERGENCE this left on `main` `0820ee28e`: the C lane emitted an entry
/// point and printed the correct line, while eval refused the same program
/// with `error: missing symbolic dimension binding \`seq\``. The absorption
/// binds the alias class before the declared result unifies with it, and
/// `Dim::Name` unifies permissively with the `*` already there, so the
/// published signature reads `tensor[*, f32]` and both lanes agree.
///
/// `spec/04-type-system.md` section 4.7 requires that agreement ("Every
/// execution mode observes the same values and traps"), section 4.7.4 repeats
/// it for the lanes by name, and section 4.7.3 forbids a verdict that turns on
/// a function boundary. `main` itself already absorbed the one-call-shallower
/// `def bare(t: tensor[3, f32]) -> tensor[seq, f32] = g(t)` to
/// `tensor[*, f32]` on both lanes, so retaining the name here would make the
/// answer depend on how many calls the extent crossed.
const RESULT_ONLY_BINDER_ROOT: &str = "def g(x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
     def h(y: tensor[k, f32]) -> tensor[k, f32] = add(y, y)\n\
     def apply1(f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = f(v)\n\
     def outer(t: tensor[3, f32]) -> tensor[seq, f32] = apply1(h, g(t))\n\
     def main() = outer(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";

/// Receipt for corpus row `root.dim_variable.result_only_binder.eval`.
///
/// Regression test. On `0820ee28e` this printed nothing on stdout and
/// `error: missing symbolic dimension binding \`seq\`` on stderr at exit 1,
/// while the C half below already printed the value: the divergence is the
/// defect this row records.
#[test]
fn a_result_only_binder_claim_sizes_a_root_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "result_only_binder_root.ch", RESULT_ONLY_BINDER_ROOT);

    let evaluated = eval(&path);
    let stderr = String::from_utf8_lossy(&evaluated.stderr).to_string();
    assert!(evaluated.status.success(), "{stderr}");
    assert!(
        !stderr.contains("missing symbolic dimension binding"),
        "a binder no parameter binds denotes the extent it met: {stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&evaluated.stdout).trim_end(),
        NESTED_HELPER_ROOT_RENDERING,
        "eval renders the root exactly as the C half does"
    );
}

/// Receipt for corpus row `root.dim_variable.result_only_binder.c`.
///
/// Disposition lock, unlike its eval twin: this half already built, linked and
/// printed this line on `0820ee28e`, and the repair must keep it. The two
/// assertions together are what makes the row a lane-agreement receipt rather
/// than a one-lane improvement.
#[test]
fn a_result_only_binder_claim_sizes_a_root_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out, emitted) =
        c_run_result_with_source(&dir, "result_only_binder_root_c", RESULT_ONLY_BINDER_ROOT);
    assert!(
        emitted.contains("int main("),
        "a realizable root owes a C entry point:\n{emitted}"
    );
    assert!(ok, "the sized root must build, link and run: {out}");
    assert_eq!(
        out.trim_end(),
        NESTED_HELPER_ROOT_RENDERING,
        "C renders the root exactly as eval does"
    );
}

/// The same root reached through an alias class with THREE dimension-variable
/// members, which is chelis#1925's round-2 P1. `apply3` takes two polymorphic
/// function arguments beside the data one, so all three of its instantiation's
/// variables land in one class, and where the runtime-extent argument sits
/// decides which member roots the class when the meeting is recorded.
///
/// `unify_dim` resolves through `constraint_dim` before it matches, so the
/// meeting is recorded on whatever rooted the class at that moment, and a
/// later `bind_dvar` in the same call re-roots the class over it. Asking two
/// ends of the class then missed a meeting recorded on the middle member, so
/// the middle ordering alone stayed dropped while the other two absorbed: an
/// argument-order disagreement `main` did not have. `spec/04-type-system.md`
/// section 4.7.3 forbids a verdict that turns on the spelling, so the receipt
/// asserts the three orderings render IDENTICALLY rather than asserting each
/// one separately.
const THREE_MEMBER_HELPERS: &str = "def g(x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
     def h(y: tensor[k, f32]) -> tensor[k, f32] = add(y, y)\n\
     def h2(z: tensor[k, f32]) -> tensor[k, f32] = add(z, z)\n";

/// The runtime-extent argument first, in the middle, and last. Every other
/// token is identical, so the three differ only in argument order.
fn three_member_orderings() -> [(&'static str, String); 3] {
    [
        (
            "first",
            format!(
                "{THREE_MEMBER_HELPERS}\
                 def apply3(v: tensor[p, f32], f: (tensor[p, f32]) -> tensor[p, f32], q: (tensor[p, f32]) -> tensor[p, f32]) -> tensor[p, f32] = q(f(v))\n\
                 def main() = apply3(g(to_tensor([1.0f32, 2.0f32, 3.0f32])), h, h2)\n"
            ),
        ),
        (
            "middle",
            format!(
                "{THREE_MEMBER_HELPERS}\
                 def apply3(f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32], q: (tensor[p, f32]) -> tensor[p, f32]) -> tensor[p, f32] = q(f(v))\n\
                 def main() = apply3(h, g(to_tensor([1.0f32, 2.0f32, 3.0f32])), h2)\n"
            ),
        ),
        (
            "last",
            format!(
                "{THREE_MEMBER_HELPERS}\
                 def apply3(f: (tensor[p, f32]) -> tensor[p, f32], q: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = q(f(v))\n\
                 def main() = apply3(h, h2, g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n"
            ),
        ),
    ]
}

/// The rendering all three orderings owe: `g` keeps the last two of three
/// elements and each of `h` and `h2` doubles them.
const THREE_MEMBER_RENDERING: &str = "main = tensor(shape=[2], data=[8.0, 12.0])";

/// Receipt for corpus row `root.dim_variable.argument_order.eval`.
///
/// Regression test. On `main` all three orderings printed nothing and exited
/// 0; at `dd0f92fd8` the middle one still did while the other two printed
/// this line, which is the order dependence this row exists to forbid.
#[test]
fn a_three_member_alias_class_sizes_a_root_in_every_argument_order_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut rendered = Vec::new();
    for (name, source) in three_member_orderings() {
        let path = fixture(&dir, &format!("three_member_{name}.ch"), &source);
        let evaluated = eval(&path);
        let stderr = String::from_utf8_lossy(&evaluated.stderr).to_string();
        assert!(evaluated.status.success(), "{name}: {stderr}");
        assert!(
            !stderr.contains("nothing to evaluate"),
            "{name}: the root is realizable and must not be dropped: {stderr}"
        );
        rendered.push(
            String::from_utf8_lossy(&evaluated.stdout)
                .trim_end()
                .to_string(),
        );
    }
    assert_eq!(rendered[0], THREE_MEMBER_RENDERING, "first");
    assert_eq!(
        rendered[1], rendered[0],
        "the middle ordering must render exactly as the first"
    );
    assert_eq!(
        rendered[2], rendered[0],
        "and so must the last: argument order may not change the verdict"
    );
}

/// Receipt for corpus row `root.dim_variable.argument_order.c`.
///
/// Regression test, and the same three orderings on the compiled lane. Each
/// owes a `int main(` and the rendering eval printed, byte for byte.
#[test]
fn a_three_member_alias_class_sizes_a_root_in_every_argument_order_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let mut rendered = Vec::new();
    for (name, source) in three_member_orderings() {
        let (ok, out, emitted) =
            c_run_result_with_source(&dir, &format!("three_member_{name}_c"), &source);
        assert!(
            emitted.contains("int main("),
            "{name}: a realizable root owes a C entry point:\n{emitted}"
        );
        assert!(ok, "{name}: the sized root must build, link and run: {out}");
        rendered.push(out.trim_end().to_string());
    }
    assert_eq!(rendered[0], THREE_MEMBER_RENDERING, "first");
    assert_eq!(rendered[1], rendered[0], "middle renders as first on C too");
    assert_eq!(rendered[2], rendered[0], "and last");
}

// ---------------------------------------------------------------------------
// chelis#1923 and chelis#1791 half A: a pipe stage denotes first-argument
// insertion, in the checker and in the lowerer.
//
// `spec/02-surf-syntax.md` §0.1 fixes the meaning: `x |> f(y)` means
// `f(x, y)`. Every consumer that met a `Pipe` node used to reconstruct that
// application for itself, and they did not all reconstruct it the same way.
// The checker typed a bare-name `to_tensor` stage from the callee's FUNCTION
// type rather than as the application, so the stage's result type stayed an
// unresolved variable and every rule that reads an application's arguments
// was lost downstream of it: `expand`'s size, `sum`'s axis. The lowerer had
// the matching defect on its own side.
//
// `chelis_deep::pipe::fold_pipe` states the sentence once, and the checker
// folds its input before inferring anything, so every later pass sees the
// application. A pipe reaching lowering is now a fail-closed error rather
// than a second derivation.
// ---------------------------------------------------------------------------

/// The size is read from an in-scope tensor, so it IS materializable: this
/// program is the one a bare-name stage broke for no reason.
const PIPED_SHAPE_SOURCED_EXPAND: &str = "module Repro.PipeShape\n\
sig f: tensor[a, f32] -> tensor[a, f32]\n\
def f(x: tensor[a, f32]) = {\n  \
a_dim = cast(shape(x, cast(0, int32)), int64)\n  \
[0.25f32] |> to_tensor |> expand(0i32, a_dim)\n\
}\n\
out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";

/// The same program with the `expand` applied instead of piped, which always
/// worked. It is the control that says the defect was the notation.
const DIRECT_SHAPE_SOURCED_EXPAND: &str = "module Repro.DirectShape\n\
sig f: tensor[a, f32] -> tensor[a, f32]\n\
def f(x: tensor[a, f32]) = {\n  \
a_dim = cast(shape(x, cast(0, int32)), int64)\n  \
expand(to_tensor([0.25f32]), 0i32, a_dim)\n\
}\n\
out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";

/// `sum` downstream of the same stage: the second witness that made this a
/// class rather than one builtin's bug.
const PIPED_SUM_AFTER_A_BARE_STAGE: &str = "module Repro.PipeSum\n\
sig f: tensor[a, f32] -> tensor[f32]\n\
def f(x: tensor[a, f32]) = [0.25f32] |> to_tensor |> sum(0)\n\
out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";

/// pipe.bare_name_stage.to_tensor.expand.{eval,c}
///
/// EVIDENTIARY STATUS: regression test on both lanes. On `f45a7848a` this
/// program was REJECTED at check, score 0.9647, with "unresolved `expand`
/// shape obligation at declaration boundary", while the applied spelling
/// below checked clean at score 1. Neither lane could run it.
#[test]
fn a_bare_name_pipe_stage_upstream_of_expand_checks_and_runs() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let expected = "out = tensor(shape=[3], data=[0.25, 0.25, 0.25])";
    let (ok, out) = c_run_result(&dir, "pipe_bare_expand_c", PIPED_SHAPE_SOURCED_EXPAND);
    assert!(
        ok,
        "a materializable piped expand must build and run: {out}"
    );
    assert!(out.contains(expected), "{out}");
    let (eval_ok, eval_out) =
        eval_result(&dir, "pipe_bare_expand_eval.ch", PIPED_SHAPE_SOURCED_EXPAND);
    assert!(eval_ok, "{eval_out}");
    assert!(
        eval_out.contains(expected),
        "the lanes agree byte for byte: {eval_out}"
    );
}

/// The applied control, which says the repair is about the notation rather
/// than about `expand` or `to_tensor`.
///
/// EVIDENTIARY STATUS: disposition lock. Clean and correct on `f45a7848a`.
#[test]
fn the_applied_spelling_of_the_same_expand_is_unchanged() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let expected = "out = tensor(shape=[3], data=[0.25, 0.25, 0.25])";
    let (ok, out) = c_run_result(&dir, "pipe_applied_expand_c", DIRECT_SHAPE_SOURCED_EXPAND);
    assert!(ok, "{out}");
    assert!(out.contains(expected), "{out}");
    let (eval_ok, eval_out) = eval_result(
        &dir,
        "pipe_applied_expand_eval.ch",
        DIRECT_SHAPE_SOURCED_EXPAND,
    );
    assert!(eval_ok && eval_out.contains(expected), "{eval_out}");
}

/// pipe.bare_name_stage.to_tensor.sum.{eval,c}: the second witness.
///
/// EVIDENTIARY STATUS: regression test on both lanes. On `f45a7848a` this
/// was rejected at check with the same declaration-boundary message, score
/// 0.94, which is what made the defect a class over every rule that reads an
/// application's arguments rather than one builtin's bug.
#[test]
fn a_bare_name_pipe_stage_upstream_of_sum_checks_and_runs() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "pipe_bare_sum_c", PIPED_SUM_AFTER_A_BARE_STAGE);
    assert!(ok, "{out}");
    assert!(out.contains("out = 0.25"), "{out}");
    let (eval_ok, eval_out) =
        eval_result(&dir, "pipe_bare_sum_eval.ch", PIPED_SUM_AFTER_A_BARE_STAGE);
    assert!(eval_ok, "{eval_out}");
    assert!(eval_out.contains("out = 0.25"), "{eval_out}");
}

// ---------------------------------------------------------------------------
// chelis#1791 half A: a bare-name stage at a CALL SITE breaks the callee's
// expand shape source.
//
// Same class as the rows above, met at the lowerer instead of the checker.
// `[...] |> to_tensor |> g(b)` checked clean and evaluated correctly, and the
// C lane refused to build it: the old `lower_pipe` synthesized
// `(app (var g) (var __acc))` with `__acc` bound to an already-lowered value,
// so `g`'s `expand` size resolved to a name no in-scope tensor axis supplied
// and chelis#469's lowering error fired on a program the checker had
// accepted. After the fold the lowerer receives `g(to_tensor([...]), b)` and
// resolves the source exactly as it does for the applied spelling.
// ---------------------------------------------------------------------------

/// The issue's reproducer A. `g`'s `expand` reads its size from `g`'s own
/// parameter, so the extent is materializable in either spelling.
const BARE_STAGE_AT_A_CALL_SITE: &str = "module Repro.PipeCallSite\n\
sig g: tensor[a, f32] -> tensor[1, f32] -> tensor[a, f32]\n\
def g(x: tensor[a, f32], b: tensor[1, f32]) = {\n  \
a_dim = cast(shape(x, cast(0, int32)), int64)\n  \
expand(b, cast(0, int32), a_dim)\n\
}\n\
out = [1.0f32, 2.0f32, 3.0f32] |> to_tensor |> g(to_tensor([0.25f32]))\n";

/// The same call written with `to_tensor` applied, which always built. The
/// bare-name stage is the whole difference.
const APPLIED_STAGE_AT_A_CALL_SITE: &str = "module Repro.AppliedCallSite\n\
sig g: tensor[a, f32] -> tensor[1, f32] -> tensor[a, f32]\n\
def g(x: tensor[a, f32], b: tensor[1, f32]) = {\n  \
a_dim = cast(shape(x, cast(0, int32)), int64)\n  \
expand(b, cast(0, int32), a_dim)\n\
}\n\
out = to_tensor([1.0f32, 2.0f32, 3.0f32]) |> g(to_tensor([0.25f32]))\n";

/// The direct spelling `chelis lint --fix` rewrites INTO the pipe form.
const DIRECT_CALL_FOR_LINT_FIX: &str = "module Repro.LintFixCallSite\n\
sig g: tensor[a, f32] -> tensor[1, f32] -> tensor[a, f32]\n\
def g(x: tensor[a, f32], b: tensor[1, f32]) = {\n  \
a_dim = cast(shape(x, cast(0, int32)), int64)\n  \
expand(b, cast(0, int32), a_dim)\n\
}\n\
out = g(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([0.25f32]))\n";

/// pipe.bare_name_stage.expand_source.{eval,c}
///
/// EVIDENTIARY STATUS: regression test on the C lane, disposition lock on
/// eval. On `08e46ebe6` this checked clean at score 1 and eval printed the
/// tensor below, while `chelis build --target c` exited 1 with "`expand` size
/// resolves to `a_dim`, but no in-scope tensor axis supplies that extent".
/// Both lanes carry a row because the pair diverged, which is the property
/// section 4.7 forbids; the eval half of that pair was the correct one.
#[test]
fn a_bare_name_stage_at_a_call_site_keeps_the_callees_expand_source() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let expected = "out = tensor(shape=[3], data=[0.25, 0.25, 0.25])";
    let (ok, out) = c_run_result(&dir, "pipe_call_site_c", BARE_STAGE_AT_A_CALL_SITE);
    assert!(ok, "the C lane must build and run this program: {out}");
    assert!(out.contains(expected), "{out}");
    let (eval_ok, eval_out) =
        eval_result(&dir, "pipe_call_site_eval.ch", BARE_STAGE_AT_A_CALL_SITE);
    assert!(eval_ok, "{eval_out}");
    assert!(
        eval_out.contains(expected),
        "and the lanes agree byte for byte: {eval_out}"
    );
}

/// The applied control for the same call site.
///
/// EVIDENTIARY STATUS: disposition lock. Built and printed the same line on
/// `08e46ebe6`.
#[test]
fn the_applied_stage_at_the_same_call_site_is_unchanged() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let expected = "out = tensor(shape=[3], data=[0.25, 0.25, 0.25])";
    let (ok, out) = c_run_result(&dir, "applied_call_site_c", APPLIED_STAGE_AT_A_CALL_SITE);
    assert!(ok, "{out}");
    assert!(out.contains(expected), "{out}");
    let (eval_ok, eval_out) = eval_result(
        &dir,
        "applied_call_site_eval.ch",
        APPLIED_STAGE_AT_A_CALL_SITE,
    );
    assert!(eval_ok && eval_out.contains(expected), "{eval_out}");
}

/// pipe.bare_name_stage.lint_fix.c: the row's own name. Following the style
/// tool must not break a building program.
///
/// This runs the real style path, with no `--allow-style-violations` and no
/// `CHELIS_STYLE_GATE_DISABLE`, because that is the defect: `lint --fix`'s
/// `prefer-pipe-operator` rewrites the direct call into the bare-name stage
/// spelling.
///
/// EVIDENTIARY STATUS: regression test. Measured on `08e46ebe6`: the direct
/// program builds and runs, `chelis lint --fix` makes two replacements, the
/// rewritten program still checks at score 1 and still evaluates, and
/// `chelis build --target c` then exits 1 with chelis#469's lowering error.
#[test]
fn a_lint_fix_of_a_direct_call_still_checks_evaluates_and_builds() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "lint_fix_call.ch", DIRECT_CALL_FOR_LINT_FIX);
    let styled = |args: &[&str]| {
        Command::cargo_bin("chelis")
            .expect("chelis")
            .args(args)
            .output()
            .expect("styled chelis invocation")
    };
    let expected = "data=[0.25, 0.25, 0.25]";
    let before = styled(&["eval", "--file", path.to_str().unwrap()]);
    assert!(
        String::from_utf8_lossy(&before.stdout).contains(expected),
        "the direct spelling works before the fix: {}",
        String::from_utf8_lossy(&before.stderr)
    );

    let formatted = styled(&["fmt", "--inplace", path.to_str().unwrap()]);
    assert!(formatted.status.success(), "fmt must succeed");
    let fixed = styled(&["lint", "--fix", path.to_str().unwrap()]);
    assert!(
        String::from_utf8_lossy(&fixed.stdout).contains("replacement"),
        "the fix rewrites the call into the pipe form: {}{}",
        String::from_utf8_lossy(&fixed.stdout),
        String::from_utf8_lossy(&fixed.stderr)
    );
    let rewritten = fs::read_to_string(&path).expect("rewritten fixture");
    assert!(
        rewritten.contains("|> to_tensor |> g(to_tensor([0.25f32]))"),
        "the bare-name stage is what the tool produces: {rewritten}"
    );

    let checked = styled(&["check", path.to_str().unwrap()]);
    assert!(
        checked.status.success(),
        "the fixed program must still check: {}",
        String::from_utf8_lossy(&checked.stdout)
    );
    let out_dir = dir.path().join("lint_fix_call-out");
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
        "and must still build: {}",
        String::from_utf8_lossy(&built.stderr)
    );
    let status = link_generated(&out_dir, "lint_fix_call.c", "lint_fix_call");
    assert!(status.success(), "link failed: {status}");
    let run = StdCommand::new(out_dir.join("lint_fix_call"))
        .output()
        .expect("run compiled binary");
    let text = String::from_utf8_lossy(&run.stdout).to_string();
    assert!(
        run.status.success() && text.contains(expected),
        "and print the same tensor: {text}"
    );
}

// ---------------------------------------------------------------------------
// chelis#1791 half B, through the CLI: `chelis check` must reject a sourceless
// `expand` size written as a pipe stage exactly as it rejects the direct
// spelling.
//
// `crates/chelis-types/tests/issue_530_expand_inline_size_gate.rs` holds the
// checker-level rows and the byte comparison. This row exists because the
// verdict a user sees is `chelis check`'s.
// ---------------------------------------------------------------------------

/// The issue's reproducer B, whose size is a cast over a bare `int32`
/// parameter and so has no tensor shape source.
const SOURCELESS_PIPE_STAGE: &str = "module Repro.BPipe\n\
sig f: tensor[a, f32] -> int32 -> tensor[a, f32]\n\
def f(x: tensor[a, f32], k: int32) = {\n  \
a_dim = k |> cast(int64)\n  \
[0.25f32] |> to_tensor |> expand(0i32, a_dim)\n\
}\n";

/// The same program with the `expand` written directly.
const SOURCELESS_DIRECT: &str = "module Repro.BDirect\n\
sig f: tensor[a, f32] -> int32 -> tensor[a, f32]\n\
def f(x: tensor[a, f32], k: int32) = {\n  \
a_dim = k |> cast(int64)\n  \
expand(to_tensor([0.25f32]), 0i32, a_dim)\n\
}\n";

/// expand.sourceless_size.pipe_position: the user-visible verdict.
///
/// EVIDENTIARY STATUS: regression test on the pipe spelling, disposition lock
/// on the direct one. On `08e46ebe6` the pipe spelling was rejected, but with
/// chelis#1909's declaration-boundary obligation message rather than section
/// 4.7.2's, so the substring assertion below failed. Before chelis#1909, on
/// `6abca2406`, it scored a clean 1.0 with an empty error list and the
/// sourceless size reached the lowerer instead.
#[test]
fn a_sourceless_expand_size_is_rejected_in_pipe_position_by_the_cli() {
    let dir = tempfile::tempdir().expect("tempdir");
    let needle = "but no tensor in scope carries it";
    let piped = check(&fixture(&dir, "sourceless_pipe.ch", SOURCELESS_PIPE_STAGE));
    let piped_out = String::from_utf8_lossy(&piped.stdout).to_string();
    assert!(
        piped_out.contains(needle) && piped_out.contains("chelis#469"),
        "the pipe stage must carry the section 4.7.2 sourceless-size diagnostic: {piped_out}"
    );
    assert!(
        !piped_out.contains("\"score\": 1,"),
        "and must not score a clean 1.0: {piped_out}"
    );
    let direct = check(&fixture(&dir, "sourceless_direct.ch", SOURCELESS_DIRECT));
    let direct_out = String::from_utf8_lossy(&direct.stdout).to_string();
    assert!(
        direct_out.contains(needle) && direct_out.contains("chelis#469"),
        "the direct spelling keeps its diagnostic: {direct_out}"
    );
}

// ---------------------------------------------------------------------------
// chelis#1779: a runtime-shaped `to_tensor` routes to the host lane under
// staging.
//
// A `to_tensor` whose operand is not a literal cons chain lowers to a
// deliberate rank-0 `Load { name: "to_tensor" }` placeholder whose documented
// contract is to be refused so the definition routes to the host lane. The
// staged host-source partition added by chelis#1693 runs ahead of the
// decision that reads that signal and cannot carry the marker, so it rejected
// the whole definition after a clean check.
// ---------------------------------------------------------------------------

/// The Shoals pricer's `const_col` column, reduced. `const_col` is the staged
/// candidate: it names `reshape` and reaches host-only builtins.
const STAGED_RUNTIME_SHAPED_COLUMN: &str = "module Repro.StagedColumn\n\
def const_col[n](spots: tensor[n, f32], v: f64) -> tensor[n, 1, f64] = {\n  \
nn = cast(shape(copy(spots), cast(0, int32)), int64)\n  \
reshape(to_tensor(map(fn (i: int64) -> v, range(cast(0, int64), nn))), [nn, cast(1, int64)])\n\
}\n\
def prices[n](spots: tensor[n, f32], k: f32) -> tensor[n, f32] = {\n  \
kc = const_col(spots, cast(k, f64))\n  \
p64 = vmap(fn (ka: tensor[1, f64]) -> tensor_to_scalar(sum(ka, 0)))(kc)\n  \
cast(p64, f32)\n\
}\n\
out = prices(to_tensor([1.0f32, 2.0f32, 3.0f32]), 5.0f32)\n";

/// staged.dynamic_to_tensor.vmap_column.{eval,c}
///
/// EVIDENTIARY STATUS: regression test on both lanes. On `08e46ebe6`
/// `chelis check` scored this 1.0 with an empty error list and BOTH lanes
/// then exited 1 with "every staged graph input needs exactly one declared or
/// host producer". On chelis#1693's parent `062c29c19` both lanes printed the
/// line below, through the host lane, and the emitted C carried the same 18
/// `const_col__tensor_` helper spellings it carries here.
#[test]
fn a_runtime_shaped_to_tensor_column_routes_to_the_host_lane_on_both_lanes() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let expected = "out = tensor(shape=[3], data=[5.0, 5.0, 5.0])";
    let (ok, out) = c_run_result(&dir, "staged_column_c", STAGED_RUNTIME_SHAPED_COLUMN);
    assert!(ok, "the C lane must build and run the column: {out}");
    assert!(out.contains(expected), "{out}");
    let (eval_ok, eval_out) =
        eval_result(&dir, "staged_column_eval.ch", STAGED_RUNTIME_SHAPED_COLUMN);
    assert!(eval_ok, "{eval_out}");
    assert!(
        eval_out.contains(expected),
        "and the lanes agree byte for byte: {eval_out}"
    );

    // Declining staging must retain the host path's reshape-size check.
    let invalid = STAGED_RUNTIME_SHAPED_COLUMN
        .replace("tensor[n, 1, f64]", "tensor[n, 2, f64]")
        .replace("[nn, cast(1, int64)]", "[nn, cast(2, int64)]")
        .replace("tensor[1, f64]", "tensor[2, f64]");
    let (eval_ok, eval_error) = eval_result(&dir, "invalid_column.ch", &invalid);
    assert!(
        !eval_ok && eval_error.contains("reshape expects 6 elements but tensor has 3"),
        "{eval_error}"
    );
    let (c_ok, c_error) = c_run_result(&dir, "invalid_column_c", &invalid);
    assert!(
        !c_ok && c_error.contains("Domain: chelis_tensor_check_reshape reshape numel mismatch: target 6 but tensor has 3 elements"),
        "{c_error}"
    );
}

// ---------------------------------------------------------------------------
// chelis#1788: the entry obligation a host-bodied signature carries.
// ---------------------------------------------------------------------------

/// A tuple root whose ONE signature spells `seq` on two parameters.
///
/// A tuple-bodied def is host-bodied, and each tensor leaf is lowered from its
/// own subexpression into its own kernel helper, so `both__tensor_0` receives
/// `x` alone and `both__tensor_1` receives `y` alone. No DAG on this lane ever
/// saw `seq` twice, which is why nothing guarded the signature's own
/// obligation.
fn split_kernel_tuple_root(x: &str, y: &str) -> String {
    format!(
        "def both(x: tensor[seq, f32], y: tensor[batch, seq, f32]) -> \
         (tensor[seq, f32], tensor[batch, seq, f32]) = (neg(x), neg(y))\n\
         a = to_tensor({x})\n\
         b = to_tensor({y})\n\
         out = both(a, b)\n"
    )
}

/// The same signature over ONE kernel. `add(x, sum(y, 0i32))` joins both
/// witnesses inside a single helper, so the DAG lane's entry guards already own
/// the obligation, and this is the rendering the split form owes.
fn one_kernel_repeated_binder_root(x: &str, y: &str) -> String {
    format!(
        "def joined(x: tensor[seq, f32], y: tensor[batch, seq, f32]) -> tensor[seq, f32] = \
         add(x, sum(y, 0i32))\n\
         a = to_tensor({x})\n\
         b = to_tensor({y})\n\
         out = joined(a, b)\n"
    )
}

/// The `[04-NUM-9]` pair an all-interface `seq` disagreement owes: section 4.7
/// puts the `load` primitive of the LATER witness in the `<op>` slot, and the
/// declaring witness is rendered first because the guard runs in declared
/// signature order.
const REPEATED_BINDER_CONTEXT: &str = "extent `seq`: x axis 0 = 3, y axis 1 = 2";

const DISAGREEING_X: &str = "[1.0, 2.0, 3.0]";
const AGREEING_X: &str = "[1.0, 2.0]";
const TWO_BY_TWO_Y: &str = "[[1.0, 2.0], [3.0, 4.0]]";

/// entry.host_tuple.repeated_binder.eval. REGRESSION TEST on the RENDERING.
/// Measured on `0820ee28e`, eval refused with the private sentence
/// ``dimension binder `seq` has inconsistent runtime witnesses: 3 and 2``,
/// which conveys neither the parameters nor the axes [04-NUM-9] requires.
#[test]
fn a_split_kernel_tuple_root_guards_its_repeated_binder_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = split_kernel_tuple_root(DISAGREEING_X, TWO_BY_TWO_Y);
    let (ok, out) = eval_result(&dir, "entry_tuple_binder.ch", &source);
    assert!(!ok, "eval must refuse the disagreeing binder: {out}");
    assert!(
        out.contains(&domain_trap_line("load")),
        "[04-NUM-9]'s line names the `load` of the later witness: {out}"
    );
    assert!(
        out.contains(REPEATED_BINDER_CONTEXT),
        "expected {REPEATED_BINDER_CONTEXT}: {out}"
    );
}

/// entry.host_tuple.repeated_binder.c. REGRESSION TEST. Measured on
/// `0820ee28e` the linked binary exited ZERO and printed both outputs at
/// `shape=[3]` and `shape=[2, 2]` under one signature that spells `seq` on
/// both, while eval refused: a lane divergence on a four-line program.
#[test]
fn a_split_kernel_tuple_root_guards_its_repeated_binder_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let source = split_kernel_tuple_root(DISAGREEING_X, TWO_BY_TWO_Y);
    let (ok, out) = c_run_result(&dir, "entry_tuple_binder_c", &source);
    assert!(!ok, "the C lane must abort: {out}");
    assert!(
        !out.contains("out.0 = "),
        "and must not print an output it computed under a refuted signature: {out}"
    );
    assert!(
        out.contains(&domain_trap_line("load")),
        "[04-NUM-9]'s line names the `load` of the later witness: {out}"
    );
    assert!(
        out.contains(REPEATED_BINDER_CONTEXT),
        "expected {REPEATED_BINDER_CONTEXT}: {out}"
    );
}

/// DISPOSITION LOCK, both lanes. The agreeing call executes exactly, so the
/// entry guard is a verdict on disagreement rather than a refusal of every
/// repeated binder. Green on `0820ee28e` and after.
#[test]
fn a_split_kernel_tuple_root_executes_when_its_repeated_binder_agrees() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = split_kernel_tuple_root(AGREEING_X, TWO_BY_TWO_Y);
    let (eval_ok, eval_out) = eval_result(&dir, "entry_tuple_binder_ok.ch", &source);
    let (c_ok, c_out) = c_run_result(&dir, "entry_tuple_binder_ok_c", &source);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(ok, "{lane}: the agreeing signature executes: {out}");
        assert!(
            out.contains("out.0 = tensor(shape=[2], data=[-1.0, -2.0])"),
            "{lane}: the first leaf's exact result: {out}"
        );
        assert!(
            out.contains("out.1 = tensor(shape=[2, 2], data=[-1.0, -2.0, -3.0, -4.0])"),
            "{lane}: the second leaf's exact result: {out}"
        );
    }
}

/// entry.kernel.repeated_binder.{eval,c}. DISPOSITION LOCK: the one-kernel
/// twin already trapped with this exact pair on both lanes before the change,
/// and it is what the split form now reproduces. It also pins that the fix did
/// not double-guard the form the DAG lane already owns: the wrapper emits no
/// entry guard where a helper can see the binder twice, because that helper's
/// prologue also owns the slot ORDER section 4.7 requires.
#[test]
fn a_one_kernel_root_keeps_its_repeated_binder_guard_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = one_kernel_repeated_binder_root(DISAGREEING_X, TWO_BY_TWO_Y);
    let (eval_ok, eval_out) = eval_result(&dir, "entry_kernel_binder.ch", &source);
    let (c_ok, c_out) = c_run_result(&dir, "entry_kernel_binder_c", &source);
    for (lane, ok, out) in [("eval", eval_ok, &eval_out), ("c", c_ok, &c_out)] {
        assert!(!ok, "{lane}: the disagreeing binder is refused: {out}");
        assert!(
            out.contains(&domain_trap_line("load")),
            "{lane}: [04-NUM-9]'s line names the `load`: {out}"
        );
        assert!(
            out.contains(REPEATED_BINDER_CONTEXT),
            "{lane}: expected {REPEATED_BINDER_CONTEXT}: {out}"
        );
        assert_eq!(
            out.matches(&domain_trap_line("load")).count(),
            1,
            "{lane}: and exactly one guard reports it, not two: {out}"
        );
    }
}

/// A type alias denotes the resolved parameter type, including its runtime
/// binder witnesses. Rejection happens before any body effect on both lanes.
#[test]
fn aliased_parameter_types_keep_the_entry_binder_guard() {
    for aliases in [
        "type Row = tensor[seq, f32]\ntype Batch = tensor[batch, seq, f32]\n",
        "type R = tensor[seq, f32]\ntype B = tensor[batch, seq, f32]\ntype Row = R\ntype Batch = B\n",
    ] {
        for (x, agrees) in [(DISAGREEING_X, false), (AGREEING_X, true)] {
            let dir = tempfile::tempdir().expect("tempdir");
            let source = format!(
                "{aliases}def both(x: Row, y: Batch) -> (Row, Batch) ! {{ IO }} = {{\n \
                 _ = print(\"body\")\n (neg(x), neg(y))\n}}\n\
                 out = both(to_tensor({x}), to_tensor({TWO_BY_TWO_Y}))\n"
            );
            let path = dir.path().join("entry-alias.ch");
            fs::write(&path, &source).expect("source");
            let checked = check(&path);
            let checked: serde_json::Value =
                serde_json::from_slice(&checked.stdout).expect("check JSON");
            assert_eq!(checked["score"], 1, "{checked}");
            let eval = eval_result(&dir, "entry-alias.ch", &source);
            let compiled = c_run_result(&dir, "entry-alias-c", &source);
            for (lane, (ok, out)) in [("eval", eval), ("C", compiled)] {
                assert_eq!(ok, agrees, "{lane}: {source}\n{out}");
                assert_eq!(out.contains("body"), agrees, "{lane}: {out}");
                if agrees {
                    assert!(
                        out.contains("out.0 = tensor(shape=[2], data=[-1.0, -2.0])"),
                        "{lane}: {out}"
                    );
                    assert!(
                        out.contains("out.1 = tensor(shape=[2, 2], data=[-1.0, -2.0, -3.0, -4.0])"),
                        "{lane}: {out}"
                    );
                } else {
                    assert!(out.contains(REPEATED_BINDER_CONTEXT), "{lane}: {out}");
                    assert!(out.contains(&domain_trap_line("load")), "{lane}: {out}");
                }
            }
        }
    }
}

/// Entry witnesses remain live through the guard even when the body does not
/// use their tensor. Both the declaring and later parameter can be unused.
#[test]
fn unused_parameters_keep_their_entry_witnesses_until_after_the_guard() {
    for used in ["x", "y"] {
        for (x, agrees) in [(DISAGREEING_X, false), (AGREEING_X, true)] {
            let dir = tempfile::tempdir().expect("tempdir");
            let source = format!(
                "def both(x: tensor[seq, f32], y: tensor[seq, f32]) -> \
                 (tensor[seq, f32], tensor[seq, f32]) ! {{ IO }} = {{\n \
                 _ = print(\"body\")\n (neg({used}), neg({used}))\n}}\n\
                 out = both(to_tensor({x}), to_tensor([1.0, 2.0]))\n"
            );
            let evaluated = eval_result(&dir, "unused-entry.ch", &source);
            let compiled = c_run_result(&dir, "unused-entry-c", &source);
            for (lane, (ok, out)) in [("eval", evaluated), ("C", compiled)] {
                assert_eq!(ok, agrees, "{lane}, used={used}: {out}");
                assert_eq!(out.contains("body"), agrees, "{lane}: {out}");
                if agrees {
                    for field in [0, 1] {
                        assert!(
                            out.contains(&format!(
                                "out.{field} = tensor(shape=[2], data=[-1.0, -2.0])"
                            )),
                            "{lane}: {out}"
                        );
                    }
                } else {
                    assert!(
                        out.contains("extent `seq`: x axis 0 = 3, y axis 0 = 2"),
                        "{lane}: {out}"
                    );
                    assert!(out.contains(&domain_trap_line("load")), "{lane}: {out}");
                }
            }
        }
    }
}

/// A wrapper guard must not preempt an earlier literal input-axis obligation
/// that an existing tensor helper owns. Drive the exported ABI directly so
/// compile-time checking does not reject the malformed foreign input first.
#[test]
fn literal_parameter_obligations_keep_the_existing_helper_order() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def both(x: tensor[2, seq, f32], y: tensor[batch, seq, f32]) -> \
                  (tensor[2, seq, f32], tensor[batch, seq, f32]) = (neg(x), neg(y))\n\
                  out = both(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), \
                  to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";
    let path = fixture(&dir, "literal-entry.ch", source);
    let out_dir = dir.path().join("c");
    let built = build_c(&path, &out_dir);
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    for (name, x0, y1, agrees) in [
        ("both_refuted", 3, 2, false),
        ("literal_refuted", 3, 3, false),
        ("agreeing", 2, 3, true),
    ] {
        let harness = format!(
            "#define main generated_main\n#include \"literal-entry.c\"\n#undef main\n\
             int main(void) {{\n\
             chelis_tensor *x = chelis_alloc(2, (int64_t[]){{{x0}, 3}}, CHELIS_DTYPE_F32);\n\
             chelis_tensor *y = chelis_alloc(2, (int64_t[]){{2, {y1}}}, CHELIS_DTYPE_F32);\n\
             chelis_tuple *result = both(x, y);\n\
             chelis_tuple_release(result);\n\
             chelis_tensor_release(x); chelis_tensor_release(y);\n\
             puts(\"completed\"); return 0;\n}}\n"
        );
        fs::write(out_dir.join("harness.c"), harness).expect("harness");
        assert!(link_generated(&out_dir, "harness.c", name).success());
        let run = std::process::Command::new(out_dir.join(name))
            .output()
            .expect("run");
        let out = format!(
            "{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(run.status.success(), agrees, "{name}: {out}");
        assert_eq!(out.contains("completed"), agrees, "{name}: {out}");
        if !agrees {
            assert!(
                out.contains("`x` axis 0 expected 2, got 3"),
                "{name}: {out}"
            );
            assert!(!out.contains("extent `seq`"), "{name}: {out}");
        }
    }
}

/// Invalid ABI ranks retain the helper's rank diagnostic before comparing
/// extents, whether the malformed tensor is the first or later witness.
#[test]
fn malformed_parameter_ranks_keep_the_existing_helper_diagnostic() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def both(x: tensor[seq, f32], y: tensor[seq, f32]) -> \
                  (tensor[seq, f32], tensor[seq, f32]) = (neg(x), neg(y))\n\
                  out = both(to_tensor([1.0, 2.0]), to_tensor([1.0, 2.0]))\n";
    let path = fixture(&dir, "rank-entry.ch", source);
    let out_dir = dir.path().join("c");
    let built = build_c(&path, &out_dir);
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    for (name, xr, x0, yr, y0, expected) in [
        ("agreeing", 1, 2, 1, 2, "completed"),
        (
            "extent_refuted",
            1,
            3,
            1,
            2,
            "numeric trap: domain in load at int64",
        ),
        ("first_extra_equal", 2, 2, 1, 2, "expected rank 1, got 2"),
        ("first_extra_mismatch", 2, 3, 1, 2, "expected rank 1, got 2"),
        ("later_extra_equal", 1, 2, 2, 2, "expected rank 1, got 2"),
        ("later_extra_mismatch", 1, 2, 2, 3, "expected rank 1, got 2"),
    ] {
        let harness = format!(
            "#define main generated_main\n#include \"rank-entry.c\"\n#undef main\n\
             int main(void) {{\n\
             chelis_tensor *x = chelis_alloc({xr}, (int64_t[]){{{x0}, 1}}, CHELIS_DTYPE_F32);\n\
             chelis_tensor *y = chelis_alloc({yr}, (int64_t[]){{{y0}, 1}}, CHELIS_DTYPE_F32);\n\
             chelis_tuple *result = both(x, y);\n\
             chelis_tuple_release(result);\n\
             chelis_tensor_release(x); chelis_tensor_release(y);\n\
             puts(\"completed\"); return 0;\n}}\n"
        );
        fs::write(out_dir.join("harness.c"), harness).expect("harness");
        assert!(link_generated(&out_dir, "harness.c", name).success());
        let run = std::process::Command::new(out_dir.join(name))
            .output()
            .expect("run");
        let out = format!(
            "{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(run.status.success(), name == "agreeing", "{name}: {out}");
        assert!(out.contains(expected), "{name}: {out}");
        if xr != 1 || yr != 1 {
            assert!(!out.contains("extent `seq`"), "{name}: {out}");
        }
    }
}

// ---------------------------------------------------------------------------
// chelis#1821: the forward activation's extent obligations under `grad`.
//
// Differentiation is an execution mode of the same activation, and section 4.7
// requires every execution mode to observe the same values and traps. Before
// this change only the per-wrt cotangents were rooted in the parent DAG after
// the gradient splice, so the entry-point dead-code elimination deleted the
// spliced forward output along with the carrier holding the activation's
// witness claims: `grad` of a program the forward call rejects returned zeros
// and exited zero on both lanes.
//
// The two disagreeing rows below are therefore regression tests, measured
// silent on `6abca2406` by reverting `crates/chelis-ir/src/{lower,vmap}.rs` to
// that head and rebuilding `chelis`. Their agreeing twins and the op-computed
// and `vmap(grad(...))` rows are disposition locks: those already behaved and
// the new shape dependency must not move them.
// ---------------------------------------------------------------------------

/// chelis#1821's reproducer as a builder. `f`'s binder `n` is witnessed by `x`
/// and its declared result extent is produced from `y`, which is three long, so
/// the claim disagrees for every width but three.
fn grad_named_claim_source(width: usize) -> String {
    let operand = (1..=width)
        .map(|value| format!("{value}.0f32"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = \
         insert(scalar_to_tensor(7.0f32), 0i32, shape(y, 0i32))\n\
         def h(x: tensor[{width}, f32]) -> tensor[f32] = \
         sum(f(x, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32)\n\
         def main() = grad(h)(to_tensor([{operand}]))\n"
    )
}

/// The same program with an `exp` between `f` and the reduction, so the
/// backward READS a forward value and the forward chain cannot be dead.
///
/// This separates two ways the activation can be lost. The claim lives on a
/// carrier node that no cotangent reads, so it dies even when every forward
/// VALUE survives; a repair that only kept the forward values reachable would
/// pass the row above and fail this one.
fn grad_live_forward_source(width: usize) -> String {
    let operand = (1..=width)
        .map(|value| format!("{value}.0f32"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = \
         insert(scalar_to_tensor(7.0f32), 0i32, shape(y, 0i32))\n\
         def h(x: tensor[{width}, f32]) -> tensor[f32] = \
         sum(exp(f(x, to_tensor([1.0f32, 2.0f32, 3.0f32]))), 0i32)\n\
         def main() = grad(h)(to_tensor([{operand}]))\n"
    )
}

/// A LITERAL result claim of 2 over a `shrink` whose extent is computed inside
/// the function, so section 4.7 places the guard locally at the `shrink` rather
/// than at entry. `width` of 4 shrinks to 3 and refutes the claim; 3 shrinks to
/// 2 and satisfies it.
fn grad_op_computed_source(width: usize) -> String {
    let operand = (1..=width)
        .map(|value| format!("{value}.0f32"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "sig f: tensor[n, f32] -> tensor[2, f32]\n\
         def f(x) = shrink(&x, [[1i64, shape(&x, 0)]])\n\
         def h(x: tensor[{width}, f32]) -> tensor[f32] = sum(f(x), 0i32)\n\
         def main() = grad(h)(to_tensor([{operand}]))\n"
    )
}

/// grad.wrt_tensor.single.dead_forward.eval.
///
/// EVIDENTIARY STATUS: regression test. On `6abca2406` this printed
/// `main = tensor(shape=[2], data=[0.0, 0.0])` and exited 0.
#[test]
fn grad_over_a_disagreeing_named_claim_traps_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "grad_named_bad.ch", &grad_named_claim_source(2));
    assert!(
        !ok,
        "grad of a rejected activation must not produce a value: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "the derivative reports the forward call's own context line: {out}"
    );
    assert!(
        !out.contains("shape=[2]"),
        "and no cotangent is printed before the trap: {out}"
    );
}

/// grad.wrt_tensor.single.dead_forward.c: the same two lines on the linked binary.
///
/// EVIDENTIARY STATUS: regression test. On `6abca2406` the binary printed
/// `main = tensor(shape=[2], data=[0.0, 0.0])` and exited 0.
#[test]
fn grad_over_a_disagreeing_named_claim_traps_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "grad_named_bad_c", &grad_named_claim_source(2));
    assert!(
        !ok,
        "grad of a rejected activation must not produce a value: {out}"
    );
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "byte-identical to the eval row's context line: {out}"
    );
}

/// The agreeing twin on eval: a width of three satisfies the claim, so the
/// derivative is computed and printed exactly.
///
/// EVIDENTIARY STATUS: disposition lock. Identical on `6abca2406`; the row
/// says the new shape dependency does not turn an agreeing activation into a
/// trap or change the cotangent.
#[test]
fn grad_over_an_agreeing_named_claim_executes_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "grad_named_ok.ch", &grad_named_claim_source(3));
    assert!(ok, "an agreeing activation differentiates: {out}");
    assert!(
        out.contains("main = tensor(shape=[3], data=[0.0, 0.0, 0.0])"),
        "the cotangent is unchanged: {out}"
    );
}

/// The agreeing twin on C.
///
/// EVIDENTIARY STATUS: disposition lock. Identical on `6abca2406`.
#[test]
fn grad_over_an_agreeing_named_claim_executes_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "grad_named_ok_c", &grad_named_claim_source(3));
    assert!(ok, "an agreeing activation differentiates on C too: {out}");
    assert!(
        out.contains("main = tensor(shape=[3], data=[0.0, 0.0, 0.0])"),
        "byte-identical to the eval twin: {out}"
    );
}

/// grad.wrt_tensor.single.live_forward.eval: the claim's carrier dies even when the
/// forward values live.
///
/// EVIDENTIARY STATUS: regression test. On `6abca2406` this printed
/// `main = tensor(shape=[2], data=[0.0, 0.0])` and exited 0, which is what
/// makes the carrier rather than the forward values the thing that was lost.
#[test]
fn grad_keeps_the_entry_carrier_when_the_backward_reads_the_forward_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "grad_live_bad.ch", &grad_live_forward_source(2));
    assert!(!ok, "a live forward does not excuse the lost claim: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "{out}"
    );
}

/// grad.wrt_tensor.single.live_forward.c.
///
/// EVIDENTIARY STATUS: regression test. On `6abca2406` the binary printed
/// `main = tensor(shape=[2], data=[0.0, 0.0])` and exited 0.
#[test]
fn grad_keeps_the_entry_carrier_when_the_backward_reads_the_forward_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "grad_live_bad_c", &grad_live_forward_source(2));
    assert!(!ok, "a live forward does not excuse the lost claim: {out}");
    assert!(out.contains(&domain_trap_line("load")), "{out}");
    assert!(
        out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "{out}"
    );
}

/// An op-computed LOCAL claim under `grad`, refuted.
///
/// EVIDENTIARY STATUS: disposition lock, both directions measured on
/// `6abca2406`. The local guard sits on a `CheckedShrinkExtent` inside the
/// value chain the cotangent reads, so it already survived AD; the row says the
/// new shape dependency neither duplicates it nor moves its rendering. The
/// NAMED spelling of this shape is not a row here: it is unguarded on the
/// forward lane too, so there is no obligation for `grad` to inherit.
#[test]
fn grad_keeps_an_op_computed_local_guard_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = eval_result(&dir, "grad_local_bad.ch", &grad_op_computed_source(4));
    assert!(!ok, "{out}");
    assert!(
        out.contains(&domain_trap_line("shrink"))
            && out.contains("extent `2`: claimed = 2, shrink axis 0 = 3"),
        "the local guard names the claim and the shrink's own extent: {out}"
    );

    let (ok, out) = eval_result(&dir, "grad_local_ok.ch", &grad_op_computed_source(3));
    assert!(ok, "the agreeing width differentiates: {out}");
    assert!(
        out.contains("main = tensor(shape=[3], data=[0.0, 1.0, 1.0])"),
        "and the cotangent is the shrink adjoint's pad: {out}"
    );
}

/// grad.op_computed.local_guard on C.
///
/// EVIDENTIARY STATUS: disposition lock. Identical on `6abca2406`.
#[test]
fn grad_keeps_an_op_computed_local_guard_on_c() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (ok, out) = c_run_result(&dir, "grad_local_bad_c", &grad_op_computed_source(4));
    assert!(!ok, "{out}");
    assert!(
        out.contains(&domain_trap_line("shrink"))
            && out.contains("extent `2`: claimed = 2, shrink axis 0 = 3"),
        "byte-identical to the eval row: {out}"
    );
}

/// `vmap(grad(...))` carries the same dependency, so its batched path needs the
/// same two controls: a plain cotangent still computes, and a refuted local
/// claim still traps with the batched axis in the context line.
///
/// EVIDENTIARY STATUS: disposition locks, both measured identical on
/// `6abca2406`. No regression row exists on this path: the named-claim
/// carrier's reproducer does not lower under `vmap(grad(...))` at that head
/// either, failing with "`vmap(...)` lowering produced no roots", which is a
/// separate defect this change does not touch.
#[test]
fn vmap_grad_keeps_its_batched_cotangent_and_local_guard_on_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plain = "def h(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, x), 0i32)\n\
                 def main() = vmap(grad(h))(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n";
    let (ok, out) = eval_result(&dir, "vmap_grad_plain.ch", plain);
    assert!(ok, "{out}");
    assert!(
        out.contains("main = tensor(shape=[2, 2], data=[2.0, 4.0, 6.0, 8.0])"),
        "the batched cotangent is unchanged: {out}"
    );

    let refuted = "sig f: tensor[n, f32] -> tensor[2, f32]\n\
                   def f(x) = shrink(&x, [[1i64, shape(&x, 0)]])\n\
                   def h(x: tensor[4, f32]) -> tensor[f32] = sum(f(x), 0i32)\n\
                   def main() = vmap(grad(h))(to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], \
                   [5.0f32, 6.0f32, 7.0f32, 8.0f32]]))\n";
    let (ok, out) = eval_result(&dir, "vmap_grad_local.ch", refuted);
    assert!(!ok, "{out}");
    assert!(
        out.contains(&domain_trap_line("shrink"))
            && out.contains("extent `2`: claimed = 2, shrink axis 1 = 3"),
        "the batched guard names axis 1: {out}"
    );
}

/// Both lanes retain the authored result claim under multi-target grad.
/// The historical test name remains the corpus receipt identity. Inferred
/// result binders must not replace the signature's `n` at the eval boundary.
#[test]
fn a_multi_target_grad_over_the_same_claim_is_still_lane_divergent() {
    if !gcc_available() {
        return;
    }
    let source = "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = \
                  insert(scalar_to_tensor(7.0f32), 0i32, shape(y, 0i32))\n\
                  def h(x: tensor[2, f32], z: tensor[2, f32]) -> tensor[f32] = \
                  sum(f(x, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32)\n\
                  def main() = grad(h, wrt=(x, z))(to_tensor([1.0f32, 2.0f32]), \
                  to_tensor([4.0f32, 5.0f32]))\n";
    let dir = tempfile::tempdir().expect("tempdir");

    let (c_ok, c_out) = c_run_result(&dir, "grad_multiwrt_c", source);
    assert!(!c_ok, "the C lane observes the claim: {c_out}");
    assert!(
        c_out.contains(&domain_trap_line("load"))
            && c_out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "with the same rendering the single-target row asserts: {c_out}"
    );

    let (eval_ok, eval_out) = eval_result(&dir, "grad_multiwrt_eval.ch", source);
    assert!(
        !eval_ok,
        "the rejected activation has no gradient: {eval_out}"
    );
    assert!(
        eval_out.contains(&domain_trap_line("load"))
            && eval_out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "the evaluator keeps the authored claim: {eval_out}"
    );
    let agreeing = source.replace("[1.0f32, 2.0f32, 3.0f32]", "[1.0f32, 2.0f32]");
    for (ok, out) in [
        eval_result(&dir, "grad_multiwrt_agree.ch", &agreeing),
        c_run_result(&dir, "grad_multiwrt_agree_c", &agreeing),
    ] {
        assert!(ok, "{out}");
        assert!(
            out.contains("main.0 = tensor(shape=[2], data=[0.0, 0.0])"),
            "{out}"
        );
        assert!(
            out.contains("main.1 = tensor(shape=[2], data=[0.0, 0.0])"),
            "{out}"
        );
    }
}

fn independent_grad_claim_source(
    distinct: bool,
    renamed: bool,
    width: usize,
    sizes: (usize, usize),
    zero: bool,
    computed: bool,
) -> String {
    let (n, m) = if renamed { ("p", "q") } else { ("n", "m") };
    let size = if computed {
        "add(shape(y, 0i32), 0i64)"
    } else {
        "shape(y, 0i32)"
    };
    let declaration = if distinct {
        format!(
            "def g(x: tensor[{n}, f32], y: tensor[{m}, f32]) -> tensor[{n}, f32] = insert(scalar_to_tensor(11.0f32), 0i32, {size})\n"
        )
    } else {
        String::new()
    };
    let callee = if distinct { "g" } else { "f" };
    let left = format!("f(copy(a), {})", vector_literal(sizes.0));
    let right = format!("{callee}(copy(b), {})", vector_literal(sizes.1));
    let (left, right) = if zero {
        (left, right)
    } else {
        (format!("mul({left}, a)"), format!("mul({right}, b)"))
    };
    format!(
        "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = insert(scalar_to_tensor(7.0f32), 0i32, {size})\n{declaration}def h(a: tensor[2, f32], b: tensor[{width}, f32]) -> tensor[f32] = add(sum({left}, 0i32), sum({right}, 0i32))\ndef main() = grad(h, wrt=(b, a))({}, {})\n",
        vector_literal(2),
        vector_literal(width)
    )
}

fn independent_grad_claim_outputs(dir: &TempDir, source: &str) -> [(bool, String); 2] {
    [
        eval_result(dir, "independent_grad.ch", source),
        c_run_result(dir, "independent_grad_c", source),
    ]
}

#[test]
fn independent_grad_entry_claims_agree_on_eval_and_c() {
    assert!(gcc_available(), "both lanes must execute");
    let dir = tempfile::tempdir().unwrap();
    for (distinct, renamed) in [(false, false), (true, false), (true, true)] {
        for width in [2, 3] {
            for zero in [false, true] {
                let source = independent_grad_claim_source(
                    distinct,
                    renamed,
                    width,
                    (2, width),
                    zero,
                    false,
                );
                let first = if zero {
                    0.0
                } else if distinct {
                    11.0
                } else {
                    7.0
                };
                let second = if zero { 0.0 } else { 7.0 };
                for (ok, out) in independent_grad_claim_outputs(&dir, &source) {
                    assert!(ok, "{source}\n{out}");
                    assert!(
                        out.contains(&format!(
                            "main.0 = tensor(shape=[{width}], data={:?})",
                            vec![first; width]
                        )),
                        "{out}"
                    );
                    assert!(
                        out.contains(&format!(
                            "main.1 = tensor(shape=[2], data={:?})",
                            vec![second; 2]
                        )),
                        "{out}"
                    );
                }
            }
        }
    }
}

#[test]
fn independent_grad_entry_failures_are_ordered_on_eval_and_c() {
    assert!(gcc_available(), "both lanes must execute");
    let dir = tempfile::tempdir().unwrap();
    for (distinct, renamed) in [(false, false), (true, false), (true, true)] {
        for zero in [false, true] {
            for (sizes, label, required, actual) in [
                ((4, 5), "n", 2, 4),
                ((2, 5), if renamed { "p" } else { "n" }, 3, 5),
            ] {
                let source =
                    independent_grad_claim_source(distinct, renamed, 3, sizes, zero, false);
                for (ok, out) in independent_grad_claim_outputs(&dir, &source) {
                    assert!(!ok, "{source}\n{out}");
                    assert!(
                        out.contains(&format!(
                            "extent `{label}`: x axis 0 = {required}, y axis 0 = {actual}"
                        )),
                        "{source}\n{out}"
                    );
                    assert!(out.contains(&domain_trap_line("load")), "{out}");
                }
            }
        }
    }
}

#[test]
fn independent_grad_computed_claims_keep_producer_on_eval_and_c() {
    assert!(gcc_available(), "both lanes must execute");
    let dir = tempfile::tempdir().unwrap();
    for (distinct, renamed) in [(false, false), (true, false), (true, true)] {
        for zero in [false, true] {
            for mismatch in [false, true] {
                let source = independent_grad_claim_source(
                    distinct,
                    renamed,
                    3,
                    (2, if mismatch { 5 } else { 3 }),
                    zero,
                    true,
                );
                for (ok, out) in independent_grad_claim_outputs(&dir, &source) {
                    assert_eq!(ok, !mismatch, "{source}\n{out}");
                    if mismatch {
                        let label = if renamed { "p" } else { "n" };
                        assert!(
                            out.contains(&format!(
                                "extent `{label}`: claimed = 3, insert axis 0 = 5"
                            )),
                            "{source}\n{out}"
                        );
                        assert!(out.contains(&domain_trap_line("insert")), "{out}");
                    } else {
                        let first = if zero {
                            0.0
                        } else if distinct {
                            11.0
                        } else {
                            7.0
                        };
                        let second = if zero { 0.0 } else { 7.0 };
                        assert!(
                            out.contains(&format!(
                                "main.0 = tensor(shape=[3], data={:?})",
                                vec![first; 3]
                            )),
                            "{out}"
                        );
                        assert!(
                            out.contains(&format!(
                                "main.1 = tensor(shape=[2], data={:?})",
                                vec![second; 2]
                            )),
                            "{out}"
                        );
                    }
                }
            }
        }
    }
}

/// Aggregate single/multiple targets preserve the same activation claim.
/// Both the refuted call and an agreeing exact-zero control execute on eval
/// and compiled C; the historical name remains the corpus receipt identity.
#[test]
fn an_aggregate_typed_wrt_is_still_lane_divergent() {
    if !gcc_available() {
        return;
    }
    let callee = "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = \
                  insert(scalar_to_tensor(7.0f32), 0i32, shape(y, 0i32))\n";
    let record = "type Params =\n  | Params { w: tensor[2, f32], b: tensor[2, f32] }\n";
    let cases = [
        (
            "grad_agg_record",
            format!(
                "{record}{callee}\
                 def h(p: Params) -> tensor[f32] = \
                 sum(f(p.w, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32)\n\
                 def main() = grad(h)(Params {{ w: to_tensor([1.0f32, 2.0f32]), \
                 b: to_tensor([3.0f32, 4.0f32]) }})\n"
            ),
            "main = Params(tensor(shape=[2], data=[0.0, 0.0]), tensor(shape=[2], data=[0.0, 0.0]))",
        ),
        (
            "grad_agg_tuple",
            format!(
                "{callee}\
                 def h(p: (tensor[2, f32], tensor[2, f32])) -> tensor[f32] = \
                 sum(f(p.0, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32)\n\
                 def main() = grad(h)((to_tensor([1.0f32, 2.0f32]), \
                 to_tensor([3.0f32, 4.0f32])))\n"
            ),
            "main.0 = tensor(shape=[2], data=[0.0, 0.0])",
        ),
        (
            "grad_agg_multi",
            format!(
                "{record}{callee}\
                 def h(p: Params, z: tensor[2, f32]) -> tensor[f32] = \
                 sum(f(p.w, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32)\n\
                 def main() = grad(h, wrt=(p, z))(Params {{ w: to_tensor([1.0f32, 2.0f32]), \
                 b: to_tensor([3.0f32, 4.0f32]) }}, to_tensor([5.0f32, 6.0f32]))\n"
            ),
            "main.1 = tensor(shape=[2], data=[0.0, 0.0])",
        ),
    ];
    let dir = tempfile::tempdir().expect("tempdir");
    for (stem, source, zeros) in cases {
        let (c_ok, c_out) = c_run_result(&dir, stem, &source);
        assert!(!c_ok, "{stem}: the C lane observes the claim: {c_out}");
        assert!(
            c_out.contains(&domain_trap_line("load"))
                && c_out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
            "{stem}: with the rendering the tensor rows assert: {c_out}"
        );
        let (eval_ok, eval_out) = eval_result(&dir, &format!("{stem}.ch"), &source);
        assert!(
            !eval_ok,
            "{stem}: the rejected activation has no gradient: {eval_out}"
        );
        assert!(
            eval_out.contains(&domain_trap_line("load"))
                && eval_out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
            "{stem}: the evaluator keeps the authored claim: {eval_out}"
        );
        let agreeing = source.replace("[1.0f32, 2.0f32, 3.0f32]", "[1.0f32, 2.0f32]");
        for (ok, out) in [
            eval_result(&dir, &format!("{stem}_agree.ch"), &agreeing),
            c_run_result(&dir, &format!("{stem}_agree_c"), &agreeing),
        ] {
            assert!(ok, "{stem}: {out}");
            assert!(out.contains(zeros), "{stem}: {out}");
        }
    }
}

/// Primitive scalar eval retains the authored claim for single/multiple
/// targets. Native admission remains a separate #1934 repair; its build
/// refusal is still recorded here until a generated binary executes.
#[test]
fn a_prim_scalar_wrt_is_still_silent_on_eval_and_refused_on_c() {
    let source = "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = \
                  insert(scalar_to_tensor(7.0f32), 0i32, shape(y, 0i32))\n\
                  def h(s: f32, x: tensor[2, f32]) -> tensor[f32] = \
                  mul(sum(f(x, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32), \
                  scalar_to_tensor(s))\n\
                  def main() = grad(h, wrt=s)(3.0f32, to_tensor([1.0f32, 2.0f32]))\n";
    let dir = tempfile::tempdir().expect("tempdir");

    // The eval half rejects the same activation as the forward call.
    let (eval_ok, eval_out) = eval_result(&dir, "grad_prim_scalar.ch", source);
    assert!(
        !eval_ok,
        "the rejected activation has no gradient: {eval_out}"
    );
    assert!(
        eval_out.contains(&domain_trap_line("load"))
            && eval_out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "{eval_out}"
    );

    // The undifferentiated control, which is what makes the silence a defect
    // rather than a program with no obligation.
    let forward = source.replace("grad(h, wrt=s)(3.0f32,", "h(3.0f32,");
    let (fwd_ok, fwd_out) = eval_result(&dir, "grad_prim_scalar_fwd.ch", &forward);
    assert!(!fwd_ok, "the forward call rejects: {fwd_out}");
    assert!(
        fwd_out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3")
            && fwd_out.contains(&domain_trap_line("load")),
        "with the rendering the tensor rows assert: {fwd_out}"
    );

    // The rank-0 tensor control: same body, `wrt` typed `tensor[f32]`, traps.
    // This is what says the variable is the `wrt`'s type and not its rank.
    let rank0 = "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = \
                 insert(scalar_to_tensor(7.0f32), 0i32, shape(y, 0i32))\n\
                 def h(s: tensor[f32], x: tensor[2, f32]) -> tensor[f32] = \
                 mul(sum(f(x, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32), s)\n\
                 def main() = grad(h, wrt=s)(scalar_to_tensor(3.0f32), \
                 to_tensor([1.0f32, 2.0f32]))\n";
    let (rank0_ok, rank0_out) = eval_result(&dir, "grad_rank0_tensor.ch", rank0);
    assert!(!rank0_ok, "a rank-0 tensor target traps: {rank0_out}");
    assert!(
        rank0_out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "{rank0_out}"
    );

    // The C half: a build REFUSAL, not a trap, and not what the rank-0
    // variant hits.
    if !gcc_available() {
        return;
    }
    let out_dir = dir.path().join("grad_prim_scalar-out");
    let build = build_c(&fixture(&dir, "grad_prim_scalar_c.ch", source), &out_dir);
    assert!(
        !build.status.success(),
        "the C lane reaches no exit state for this kind"
    );
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        stderr.contains("can't lower these defs")
            && stderr.contains("in a position the host lane can't resolve"),
        "and it is the host-lane transform-position refusal: {stderr}"
    );
    // The prim scalar crossed with a MULTI target, so the axis's sixth cell
    // preserves the claim on eval too; C still refuses before execution.
    let multi = source.replace("grad(h, wrt=s)(", "grad(h, wrt=(s, x))(");
    let (multi_ok, multi_out) = eval_result(&dir, "grad_prim_multi.ch", &multi);
    assert!(!multi_ok, "{multi_out}");
    assert!(
        multi_out.contains(&domain_trap_line("load"))
            && multi_out.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "the multi-target call keeps the same claim: {multi_out}"
    );
    let multi_build = build_c(
        &fixture(&dir, "grad_prim_multi_c.ch", &multi),
        &dir.path().join("grad_prim_multi-out"),
    );
    assert!(
        !multi_build.status.success()
            && String::from_utf8_lossy(&multi_build.stderr).contains("can't lower these defs"),
        "and the same refusal on C: {}",
        String::from_utf8_lossy(&multi_build.stderr)
    );

    let rank0_build = build_c(
        &fixture(&dir, "grad_rank0_tensor_c.ch", rank0),
        &dir.path().join("grad_rank0_tensor-out"),
    );
    assert!(
        rank0_build.status.success(),
        "while the rank-0 tensor variant of the same inline `grad` builds, so \
         the refusal is specific to the prim-scalar kind: {}",
        String::from_utf8_lossy(&rank0_build.stderr)
    );
}
