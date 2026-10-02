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
//!
//! The untransformed rows below are two different kinds of evidence, and
//! conflating them has now been wrong three times, so they are labelled:
//!
//! * **Behaviour preservation** (rows 1-4). These pin that an untransformed
//!   `fail` with a computed message still aborts at run time with its message.
//!   That is worth pinning on its own — but it does NOT discriminate *where*
//!   the rejection lives, because these programs never reach the placeholder
//!   emit site at all: their message is evaluated and aborted by the host
//!   interpreter first.
//! * **Placement** (row 5, the `string`-parameter canary shape). This is the
//!   only row that reaches the emit site, and therefore the only one that reds
//!   if the rejection is moved there.
//!
//! **In an untransformed program**, the measured discriminator for reaching the
//! emit site is the message arriving as a `string` PARAMETER of the def
//! containing the `fail`. The scope clause is load-bearing: a TRANSFORMED body
//! with a purely local `string_concat(..)` message reaches the site too — that
//! is how the record gets set at all, and it is this file's central case.
//! Return rank is irrelevant — an earlier revision of this comment claimed a rank-0
//! body never reaches DAG lowering and that a tensor-returning one does; both
//! are false. A rank-0 body with a `string` parameter reaches it; a
//! tensor-returning body with a local message does not.
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

/// `chelis build --target c`: the rejection fires during lowering, so the build
/// fails and its stderr is the artifact. Returns `Ok(())` when a build
/// unexpectedly succeeded.
fn c_build_stderr(program: &str, name: &str) -> Result<(), String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join("out").to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    if built.status.success() {
        return Ok(());
    }
    Err(String::from_utf8_lossy(&built.stderr).into_owned())
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
    // Rows 1-4 are BEHAVIOUR-PRESERVATION rows: they pin that the abort still
    // fires, and they do not reach the emit site, so they do not discriminate
    // where the rejection lives. Row 5 is the PLACEMENT row and is the only one
    // that reds if the rejection is moved to the emit site. Do not "strengthen"
    // rows 1-4 by copying row 5's shape into them -- that would be four copies
    // of one witness, which is what #2743 rounds 1 and 2 both pushed back on.
    let cases = [
        (
            "plain operand",
            format!(
                "module M.Main\n{TAG}\
                 def loss(x: tensor[1, f32]) -> tensor[1, f32] = \
                 add(x, fail(string_concat(\"bad: \", tag())))\n\
                 out = print(loss(to_tensor([cast(3.0, f32)])))\n"
            ),
            "bad: t",
        ),
        (
            "whole body",
            format!(
                "module M.Main\n{TAG}\
                 def loss(x: tensor[1, f32]) -> tensor[1, f32] = \
                 fail(string_concat(\"bad: \", tag()))\n\
                 out = print(loss(to_tensor([cast(3.0, f32)])))\n"
            ),
            "bad: t",
        ),
        (
            "if branch",
            format!(
                "module M.Main\n{TAG}\
                 def loss(x: tensor[1, f32]) -> tensor[1, f32] =\n\
                 \x20 if gt(tensor_to_scalar(sum(&x, cast(0, i32))), cast(0.5, f32)) \
                 then fail(string_concat(\"bad: \", tag())) \
                 else mul(&x, &x)\n\
                 out = print(loss(to_tensor([cast(3.0, f32)])))\n"
            ),
            "bad: t",
        ),
        (
            "let-bound message",
            "module M.Main\n\
             def loss(x: tensor[1, f32]) -> tensor[1, f32] = {\n\
               m = \"bad\"\n\
               add(x, fail(m))\n\
             }\n\
             out = print(loss(to_tensor([cast(3.0, f32)])))\n"
                .to_string(),
            "bad",
        ),
        (
            // The census canary's shape: the message is a PARAMETER, and the
            // def is nullary, so this row reached the emit site even before
            // rows 1-4 were made tensor-returning.
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
        let outcome = eval_program(&program);
        let stderr = match outcome {
            Ok(stdout) => panic!(
                "untransformed `{what}` must ABORT, not produce a value; got stdout {stdout:?}"
            ),
            Err(stderr) => stderr,
        };
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

// ===========================================================================
// The third wired transform site, and cross-lane agreement
// ===========================================================================

/// `vmap(grad(..))` is a THIRD lowering site with its own check, and it had no
/// test: disabling all three checks reddened four tests, none of them this
/// shape, so deleting this site's call alone would have gone unnoticed
/// (#2743 round 1, P2-1).
#[test]
fn a_non_literal_message_rejects_under_vmap_of_grad() {
    let program = format!(
        "module M.Main\n{TAG}\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         sum(add(x, fail(string_concat(\"bad: \", tag()))), cast(0, i32))\n\
         out = vmap(grad(loss))(to_tensor([[cast(3.0, f32)], [cast(4.0, f32)]]))\n"
    );
    let stderr = eval_program(&program).expect_err("[05-OP-68]: this must not produce a value");
    assert_names_the_unguardable_reason(&stderr, "vmap(grad(...))");
    assert!(
        stderr.contains("vmap(grad(...))"),
        "the rejection must name the composed transform, not just `vmap` or `grad`: {stderr}"
    );
}

/// A statically-selected `fail` branch whose message is ALSO unguardable
/// carries two facts, and the static-selection one is the more useful: the `if`
/// has no value on any path. Reporting only the message defect sent the author
/// to fix the message and re-run to discover that (#2743 round 1, P3-1).
#[test]
fn a_statically_taken_unguardable_fail_reports_the_selection_first() {
    let program = format!(
        "module M.Main\n{TAG}\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         if gt(cast(2, i64), cast(1, i64)) \
         then fail(string_concat(\"bad: \", tag())) \
         else sum(mul(&x, &x), cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n"
    );
    let stderr = eval_program(&program).expect_err("a statically-taken abort must be rejected");
    let selection = stderr
        .find("always selects its `fail(...)` branch")
        .unwrap_or_else(|| panic!("the static-selection fact must be reported: {stderr}"));
    let defect = stderr
        .find("must be a string literal written at the `fail` itself")
        .unwrap_or_else(|| panic!("the message defect must be reported too: {stderr}"));
    assert!(
        selection < defect,
        "the selection fact must come FIRST -- it is the one that cannot be fixed by \
         editing the message: {stderr}"
    );
}

/// The same, for the `fail("")` spelling. This is the shape whose diagnostic
/// regressed in round 1: it used to report the static selection and briefly
/// reported only the empty-message rule.
#[test]
fn a_statically_taken_empty_fail_reports_the_selection_first() {
    let program = "module M.Main\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         if gt(cast(2, i64), cast(1, i64)) then fail(\"\") \
         else sum(mul(&x, &x), cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    let stderr = eval_program(program).expect_err("a statically-taken abort must be rejected");
    let selection = stderr
        .find("always selects its `fail(...)` branch")
        .unwrap_or_else(|| panic!("the static-selection fact must be reported: {stderr}"));
    let defect = stderr
        .find("has no message to report")
        .unwrap_or_else(|| panic!("the empty-message rule must be reported too: {stderr}"));
    assert!(
        selection < defect,
        "the selection fact must come FIRST: {stderr}"
    );
}

/// chelis#2371 acceptance 4, **for the fatal record path only**.
///
/// `reject_recorded_unguardable_fail` is fatal, so the host fallback cannot
/// absorb it and the C target reports the same reason as `eval`. The three
/// non-fatal raisers (`reject_non_literal_fail_message`,
/// `reject_fail_message_defect`, `reject_static_taken_unguardable_fail`) ARE
/// absorbed, and the C lane reports a generic "can't lower these defs" instead.
/// That absorption is pre-existing and identical on the base — it is not
/// introduced here — but it means cross-lane agreement holds for this path and
/// not yet for those, so this test's name says which
/// (chelis#2743 round 2, NEW-2).
#[test]
fn the_c_target_agrees_with_the_evaluator_on_the_fatal_record_path() {
    let transformed = format!(
        "module M.Main\n{TAG}\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         sum(add(x, fail(string_concat(\"bad: \", tag()))), cast(0, i32))\n\
         def d(x: tensor[1, f32]) -> tensor[1, f32] = grad(loss, wrt = x)(x)\n\
         out = d(to_tensor([cast(3.0, f32)]))\n"
    );
    let eval_err = eval_program(&transformed).expect_err("eval must reject the transformed shape");
    assert_names_the_unguardable_reason(&eval_err, "eval lane");
    let c_err = c_build_stderr(&transformed, "fenced")
        .expect_err("the C target must reject the transformed shape too");
    assert_names_the_unguardable_reason(&c_err, "c lane");

    // The untransformed twin must still BUILD; its abort is a runtime abort.
    // Asserted as a build SUCCESS, not merely as "did not fail with 05-OP-68":
    // the weaker form also passed when the build failed for any other reason
    // (chelis#2743 round 2, NEW-6).
    let untransformed = format!(
        "module M.Main\n{TAG}\
         def loss(x: tensor[1, f32]) -> tensor[1, f32] = \
         add(x, fail(string_concat(\"bad: \", tag())))\n\
         out = print(loss(to_tensor([cast(3.0, f32)])))\n"
    );
    if let Err(c_err) = c_build_stderr(&untransformed, "unfenced") {
        panic!(
            "the untransformed twin must still build; its abort is a runtime abort, so a \
             build failure means the fence leaked into it: {c_err}"
        );
    }
}

/// chelis#2743 round 2, NEW-7b: an UNCALLED def whose body applies a transform
/// over an unguardable `fail` now fails `chelis build --target c`, where the
/// base exited 0.
///
/// Pinned deliberately rather than left uncharacterised. The rejection is a
/// lowering-time decision and lowering is not DCE-sensitive, so a def that
/// would abort if called is refused whether or not anything calls it — and the
/// base was not shipping a wrong artifact either, because DCE omitted the def
/// entirely. No value is produced on either head; only the exit status moved.
///
/// The control below establishes one fact and no more: the same unguardable
/// `fail` in an uncalled def with NO transform still builds, on base and on
/// this head. It is NOT a leak detector — reapplying the refuted emit-site
/// design does not red it either, because this shape does not reach the emit
/// site in the build lane (chelis#2743, NEW-8b). An earlier version of this
/// control did not build on EITHER head, for an unrelated rank-0 `expand` type
/// error, so it satisfied its own weak assertion trivially while its comment
/// claimed the opposite.
#[test]
fn an_uncalled_def_applying_a_transform_over_an_unguardable_fail_is_refused() {
    let with_transform = format!(
        "module M.Main\n{TAG}\
         def bfail() -> f32 = fail(string_concat(\"bad: \", tag()))\n\
         def loss(x: tensor[1, f32]) -> f32 = {{\n\
           u = bfail()\n\
           cast(1.0, f32)\n\
         }}\n\
         def never_called(x: tensor[1, f32]) -> tensor[1, f32] = grad(loss)(x)\n\
         def used(y: tensor[2, f32]) -> tensor[f32] = sum(y, cast(0, i32))\n\
         out = used(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\n"
    );
    let err = c_build_stderr(&with_transform, "uncalled_transform")
        .expect_err("an uncalled transform over an unguardable `fail` is refused at lowering");
    assert_names_the_unguardable_reason(&err, "an uncalled transform def");

    // Control: no transform, so the fence does not reach it. Asserted as a
    // build SUCCESS -- the weak `if let Err { assert!(!contains(..)) }` form is
    // what let the previous control pass while failing to build at all.
    let without_transform = format!(
        "module M.Main\n{TAG}\
         def never_called(x: tensor[1, f32]) -> tensor[1, f32] = \
         add(x, fail(string_concat(\"bad: \", tag())))\n\
         def used(y: tensor[2, f32]) -> tensor[f32] = sum(y, cast(0, i32))\n\
         out = used(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\n"
    );
    if let Err(c_err) = c_build_stderr(&without_transform, "uncalled_plain") {
        panic!(
            "an uncalled def with an unguardable `fail` and NO transform must still \
             build, as it does on base; a failure here means the fence reached past \
             the transform sites: {c_err}"
        );
    }
}
