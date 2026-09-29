//! chelis#2371 / chelis#2383: a direct `fail(...)` whose message cannot become
//! an [05-OP-68] abort identity must not produce a value under a transform,
//! and must be named by the reason it is unguardable.
//!
//! `spec/05-risc-primitives.md` [05-OP-68] states that the abort message "is
//! part of the operation's identity, not a runtime operand, so the operation
//! takes no string value". A guard therefore exists only for a message that is
//! a string literal written at the `fail` itself. Every other spelling of a
//! direct `fail` is unguardable — and before this change the unguardable ones
//! were silently replaced by a zero placeholder that the transform consumed as
//! an answer:
//!
//! | shape, all under `grad`/`vmap` | before |
//! |---|---|
//! | `sum(add(x, fail(string_concat("bad: ", tag()))), 0i32)` | `1.0`, exit 0 |
//! | the same as the whole body | `[0.0]`, exit 0 |
//! | the same under `vmap` | inputs echoed, abort gone |
//! | a `let`-bound literal, `m = "bad"` then `fail(m)` | `1.0`, exit 0 |
//!
//! ## Where the decision is taken, and why not at the `fail`
//!
//! The emit site knows the `fail` has no guard. It does **not** know whether
//! anything will consume the placeholder, and that distinction is the whole
//! problem: an untransformed `fail` reaches the same site, its placeholder is
//! never read, and the host lane delivers the abort at run time. Rejecting at
//! the emit site was implemented and **measured to break**
//! `loud_unsupported_census_canaries::canary_dynamic_fail_aborts_loudly`
//! (`def boom(msg: string) -> tensor[2, f32] = fail(msg)`, no transform
//! anywhere). `host_program` does not separate the two cases either — it is
//! `None` for that canary too.
//!
//! So the lowerer RECORDS the unguardable `fail`, and each transform checks
//! the record after lowering its body, because the transform is the consumer.
//! The untransformed half of this file is not decoration: it is the evidence
//! that the fence is bounded to consumers.
//!
//! chelis#2383 is the diagnostic half. A direct `fail` branch that
//! `fail_message_of` declined used to fall through and collect the
//! *indirect*-`fail` diagnostic, which told the author to "write `fail(...)`
//! directly as the branch" — which is what they had written. One classifier
//! now returns either a usable message or a typed reason, so a direct `fail`
//! always gets a direct diagnostic, and a genuinely indirect one still gets
//! the indirect wording.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// Evaluate a program; first stdout line on success, stderr on failure.
fn eval_program(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

const TAG: &str = "def tag() -> string = \"t\"\n";

fn assert_names_the_unguardable_reason(stderr: &str, what: &str) {
    assert!(
        stderr.contains("must be a string literal written at the `fail` itself"),
        "the rejection must name the REASON the message is unguardable for {what}: {stderr}"
    );
    assert!(
        stderr.contains("05-OP-68"),
        "the rejection must cite the atom that decides it for {what}: {stderr}"
    );
    assert!(
        !stderr.contains("as part of an `if` branch rather than as the branch itself"),
        "chelis#2383: a DIRECT `fail` must never collect the indirect-`fail` \
         diagnostic for {what}: {stderr}"
    );
}

// ===========================================================================
// Under a transform: no value, ever
// ===========================================================================

/// The issue's reproducer. `1.0` before — the fabricated zero was an operand
/// of the arithmetic, so it did not merely survive, it was added to `x`.
#[test]
fn a_non_literal_message_as_an_operand_rejects_under_grad() {
    let program = format!(
        "module M.Main\n{TAG}\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         sum(add(x, fail(string_concat(\"bad: \", tag()))), cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n"
    );
    let stderr = eval_program(&program).expect_err("[05-OP-68]: this must not produce a value");
    assert_names_the_unguardable_reason(&stderr, "an operand under grad");
    assert!(
        stderr.contains("grad"),
        "the rejection must name the consuming transform: {stderr}"
    );
}

/// `vmap` dropped the abort entirely and echoed its inputs.
#[test]
fn a_non_literal_message_rejects_under_vmap() {
    let program = format!(
        "module M.Main\n{TAG}\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         sum(add(x, fail(string_concat(\"bad: \", tag()))), cast(0, i32))\n\
         out = vmap(loss)(to_tensor([[cast(3.0, f32)], [cast(4.0, f32)]]))\n"
    );
    let stderr = eval_program(&program).expect_err("[05-OP-68]: this must not produce a value");
    assert_names_the_unguardable_reason(&stderr, "vmap");
    assert!(
        stderr.contains("vmap"),
        "the rejection must name the consuming transform: {stderr}"
    );
}

/// The whole transformed body is the `fail`.
#[test]
fn a_non_literal_message_as_the_whole_body_rejects_under_grad() {
    let program = format!(
        "module M.Main\n{TAG}\
         def loss(x: tensor[1, f32]) -> tensor[f32] = fail(string_concat(\"bad: \", tag()))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n"
    );
    let stderr = eval_program(&program).expect_err("[05-OP-68]: this must not produce a value");
    assert_names_the_unguardable_reason(&stderr, "a whole body under grad");
}

/// A `let`-bound message is a DIFFERENT SPELLING of the same reason, not a new
/// reason: the message is a literal, but not at the `fail`, so [05-OP-68] has
/// no identity to carry. This is the shape that would have needed a fourth
/// one-off arm under the old fall-through (chelis#2383 acceptance 4).
#[test]
fn a_let_bound_literal_message_rejects_under_grad() {
    let program = "module M.Main\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = {\n\
           m = \"bad\"\n\
           sum(add(x, fail(m)), cast(0, i32))\n\
         }\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    let stderr = eval_program(program).expect_err("[05-OP-68]: this must not produce a value");
    assert_names_the_unguardable_reason(&stderr, "a let-bound message");
}

// ===========================================================================
// chelis#2383: a direct `fail` never collects the indirect diagnostic
// ===========================================================================

/// The `if`-branch spelling. Before, this reported the indirect-`fail`
/// diagnostic and instructed the author to write `fail(...)` directly as the
/// branch — which is exactly what is written here.
#[test]
fn a_direct_non_literal_fail_branch_is_not_called_indirect() {
    let program = format!(
        "module M.Main\n{TAG}\
         def loss(x: tensor[1, f32]) -> tensor[f32] =\n\
         \x20 if gt(tensor_to_scalar(sum(&x, cast(0, i32))), cast(0.5, f32)) \
         then fail(string_concat(\"bad: \", tag())) \
         else sum(mul(&x, &x), cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n"
    );
    let stderr = eval_program(&program).expect_err("an unguardable branch must be rejected");
    assert_names_the_unguardable_reason(&stderr, "a direct if-branch");
}

/// The negative half: a GENUINELY indirect `fail` — reached through a helper,
/// so it is part of the branch rather than the branch itself — still gets the
/// indirect wording, which is correct for it. Without this the previous test
/// could be satisfied by deleting the indirect diagnostic outright.
#[test]
fn a_genuinely_indirect_fail_still_reports_the_indirect_diagnostic() {
    let program = "module M.Main\n\
         def boom(x: tensor[1, f32]) -> tensor[f32] = fail(\"indirect boom\")\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] =\n\
         \x20 if gt(tensor_to_scalar(sum(&x, cast(0, i32))), cast(0.5, f32)) \
         then add(boom(copy(x)), sum(&x, cast(0, i32))) \
         else sum(mul(&x, &x), cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    let stderr = eval_program(program).expect_err("an indirect fail in a branch must be rejected");
    assert!(
        stderr.contains("as part of an `if` branch rather than as the branch itself"),
        "a genuinely indirect `fail` keeps the indirect diagnostic: {stderr}"
    );
}

/// `fail("")` keeps its own reason. An empty message is a type error under
/// [05-OP-68] whatever position it is written in, and it must not be folded
/// into the not-a-literal reason.
#[test]
fn an_empty_message_keeps_its_own_diagnostic() {
    let program = "module M.Main\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = sum(add(x, fail(\"\")), cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    let stderr = eval_program(program).expect_err("`fail(\"\")` must be rejected");
    assert!(
        stderr.contains("has no message to report"),
        "an empty message keeps its dedicated diagnostic: {stderr}"
    );
}

// ===========================================================================
// NEGATIVE PARITY: the fence is bounded to consumers
// ===========================================================================

/// Untransformed, the abort is the host lane's and must still fire at run
/// time with its dynamic message. These four rows are what make the fence
/// honest; rejecting at the `fail` site passed every test above and broke
/// these.
#[test]
fn untransformed_non_literal_fails_still_abort_with_their_message() {
    let cases = [
        (
            "plain operand",
            format!(
                "module M.Main\n{TAG}\
                 def loss(x: tensor[1, f32]) -> tensor[f32] = \
                 sum(add(x, fail(string_concat(\"bad: \", tag()))), cast(0, i32))\n\
                 out = loss(to_tensor([cast(3.0, f32)]))\n"
            ),
            "bad: t",
        ),
        (
            "whole body",
            format!(
                "module M.Main\n{TAG}\
                 def loss(x: tensor[1, f32]) -> tensor[f32] = \
                 fail(string_concat(\"bad: \", tag()))\n\
                 out = loss(to_tensor([cast(3.0, f32)]))\n"
            ),
            "bad: t",
        ),
        (
            "if branch",
            format!(
                "module M.Main\n{TAG}\
                 def loss(x: tensor[1, f32]) -> tensor[f32] =\n\
                 \x20 if gt(tensor_to_scalar(sum(&x, cast(0, i32))), cast(0.5, f32)) \
                 then fail(string_concat(\"bad: \", tag())) \
                 else sum(mul(&x, &x), cast(0, i32))\n\
                 out = loss(to_tensor([cast(3.0, f32)]))\n"
            ),
            "bad: t",
        ),
        (
            "let-bound message",
            "module M.Main\n\
             def loss(x: tensor[1, f32]) -> tensor[f32] = {\n\
               m = \"bad\"\n\
               sum(add(x, fail(m)), cast(0, i32))\n\
             }\n\
             out = loss(to_tensor([cast(3.0, f32)]))\n"
                .to_string(),
            "bad",
        ),
        (
            // The census canary's shape: the message is a PARAMETER. This is
            // the row that refuted rejecting at the emit site.
            "message through a string parameter",
            "module M.Main\n\
             def boom(msg: string) -> tensor[2, f32] = fail(msg)\n\
             def run() -> tensor[2, f32] = boom(string_concat(\"dynamic \", \"message\"))\n\
             out = print(run())\n"
                .to_string(),
            "dynamic message",
        ),
    ];
    for (what, program, expected) in cases {
        let stderr = eval_program(&program).unwrap_or_else(|e| e).to_string();
        assert!(
            stderr.contains(expected),
            "untransformed `{what}` must abort with its message ({expected:?}): {stderr}"
        );
        assert!(
            !stderr.contains("05-OP-68"),
            "untransformed `{what}` must not be fenced at lowering: {stderr}"
        );
    }
}

/// The positive control [05-OP-68] exists for: a LITERAL message under a
/// transform still builds its guard and still aborts with that message. If
/// this regressed, the fence would be indistinguishable from "reject every
/// `fail` under a transform".
#[test]
fn a_literal_message_under_grad_still_aborts_through_its_guard() {
    let program = "module M.Main\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         sum(add(x, fail(\"literal boom\")), cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    let stderr = eval_program(program).expect_err("a guarded abort must fire");
    assert!(
        stderr.contains("literal boom"),
        "a literal message keeps reaching the user through its guard: {stderr}"
    );
    assert!(
        !stderr.contains("05-OP-68"),
        "a literal message is guardable and must not be fenced: {stderr}"
    );
}

/// An untaken guard with a literal message still differentiates. The fence
/// must not have turned every `fail` in a transformed body into a rejection.
#[test]
fn an_untaken_literal_guard_still_differentiates() {
    let program = "module M.Main\n\
         def loss(x: tensor[2, f32]) -> tensor[f32] = \
         if gt(cast(2, i64), cast(shape(x, cast(0, i32)), i64)) \
         then fail(\"too short\") \
         else sum(x, cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32), cast(4.0, f32)]))\n";
    let line = eval_program(program).expect("an untaken guard must still differentiate");
    assert!(
        line.contains("1.0, 1.0"),
        "d(sum(x))/dx = ones through an untaken guard; got {line}"
    );
}
