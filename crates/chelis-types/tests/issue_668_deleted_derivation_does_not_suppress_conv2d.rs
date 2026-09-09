//! chelis#668 round-1 P0: deleting a derivation must not reclassify its
//! operations as FAILED derivations, because a failed one silences a
//! downstream validator.
//!
//! # The mechanism, and why nothing caught it
//!
//! `record_let_binding_shape_fact` marks a `let` name in `failed_let_names`
//! when its right-hand side is structurally a shape-sensitive form whose output
//! type this validator could not derive. `conv`'s validator then returns
//! early for any input naming a marked binding, so one root cause reports one
//! diagnostic instead of one per consumer (RT-205 round-2 F3).
//!
//! PR A deleted the `stride`, `expand`, and `insert` arms of
//! `derive_ir_builtin_output_type` but left those three names in
//! `is_ir_shape_sensitive_builtin`, which is what the marker was keyed on. The
//! three became permanent failed derivations, so
//! `y = expand(...); conv(&y, ...)` switched the ENTIRE `conv` validator
//! off: input concreteness, kernel concreteness, rank-4, stride positivity, and
//! padding non-negativity all stopped running. An invalid `stride = 0` then
//! passed `chelis check` and panicked in codegen (`chelis-ir/src/dag.rs`),
//! which is the internal-compiler-error class the guard's own comment cites
//! from chelis#186 F1.
//!
//! Nothing caught it because the conv guard's regression test calls `conv`
//! directly, with no intervening `let`. The suppression path was unmeasured.
//!
//! The repair keys the marker on `ir_builtin_has_output_type_derivation`, the
//! table that owes the type, rather than on `is_ir_shape_sensitive_builtin`,
//! which answers a different question. Both halves are needed: the second alone
//! would mark every `y = relu(x)`, because the Identity fallthrough has an arm
//! for `relu`.
//!
//! # Evidentiary status, per assertion
//!
//! REGRESSION for the four rows that assert a `conv` diagnostic: each is red
//! at `948ed5736`, the pushed head that carried the defect, where all of these
//! programs scored 1 with an empty error list. Each is green on the base
//! `12c04c66a` and on the repaired head, which measured byte-identical on all
//! seven of the reviewer's probes, scores included.
//!
//! DISPOSITION LOCK for the direct-call control and the cascade row.
//!
//! # Which ingress these rows speak for, and why it is only one
//!
//! `check_typed_program` never calls `validate_ir_program`, so no row here has
//! a `check_typed_program` verdict to assert: that ingress accepts every
//! program below, including a bare `conv` with `stride = 0`. That asymmetry
//! is PRE-EXISTING and untouched by this pull request. `infer/program.rs`, which
//! decides it, is byte-identical to the base, and `checker_totality.md` D1
//! already records that the ingress `chelis prove` uses runs no post-inference
//! validator. `the_post_inference_validator_runs_only_on_the_ir_ingress` states
//! it once, so the silence in the other rows is recorded rather than implied.
//! It is a real gap, and it belongs to whoever closes it, not to this PR.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn deep(source: &str) -> Vec<chelis_deep::Expr> {
    desugar_program(&parse_surf(source).unwrap_or_else(|error| {
        panic!("fixture must parse as Surf: {error:?}\n{source}");
    }))
}

fn messages(errors: &[CheckError]) -> Vec<String> {
    let mut out: Vec<String> = errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect();
    out.sort();
    out
}

fn ir_diagnostics(source: &str) -> Vec<String> {
    match check_ir_program(&deep(source)) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    }
}

fn typed_diagnostics(source: &str) -> Vec<String> {
    match check_typed_program(&deep(source)) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    }
}

fn assert_conv_still_validates(source: &str, needle: &str, label: &str) {
    let diagnostics = ir_diagnostics(source);
    assert!(
        diagnostics.iter().any(|m| m.contains(needle)),
        "{label}: a let-bound movement operand must not switch the `conv` \
         validator off; expected a diagnostic containing {needle:?}, got \
         {diagnostics:?}"
    );
}

const CONV2D_STRIDE_ZERO_DIRECT: &str = "module Repro.Conv2dStrideZeroDirect\n\
     def f(x: tensor[2, 3, 8, 8, f32], k: tensor[4, 3, 3, 3, f32]) = \
     conv(&x, &k, [0i64, 0i64], [(0i64, 0i64), (0i64, 0i64)])\n";

/// `stride = 0` is undefined: the output spatial-dim formula divides by it.
/// Routing the input through each of the three operations whose derivation
/// chelis#668 deleted must not stop the guard firing.
///
/// REGRESSION. At `948ed5736` all three scored 1 with no errors, and the first
/// then panicked in codegen instead of failing to compile.
///
/// The needle is the callee rather than the stride sentence on purpose: the
/// concreteness check runs first and returns, so the diagnostic these programs
/// reach is the metadata one. What the row asserts is that the validator RUNS.
/// `an_invalid_stride_without_a_let_binding_names_the_stride` is the row that
/// pins the stride check's own text.
#[test]
fn an_invalid_stride_is_refused_through_each_deleted_derivation() {
    for (binding, input, label) in [
        (
            "y = expand(x, 0i32, 2i64)",
            "tensor[1, 3, 8, 8, f32]",
            "let-bound expand",
        ),
        (
            "y = insert(x, 0i32, 2i64)",
            "tensor[3, 8, 8, f32]",
            "let-bound insert",
        ),
        (
            "y = stride(x, 1i64, 1i64, 1i64, 1i64)",
            "tensor[2, 3, 8, 8, f32]",
            "let-bound stride",
        ),
    ] {
        let source = format!(
            "module Repro.Conv2dStrideZero\n\
             def f(x: {input}, k: tensor[4, 3, 3, 3, f32]) = {{\n\
               {binding}\n\
               conv(&y, &k, [0i64, 0i64], [(0i64, 0i64), (0i64, 0i64)])\n\
             }}\n"
        );
        assert_conv_still_validates(&source, "IR builtin `conv`", label);
    }
}

/// The control for the row above: the same invalid stride with no intervening
/// binding reaches the stride check itself and names it.
///
/// DISPOSITION LOCK, green in all three states. It is what proves the rows
/// above measure the SUPPRESSION rather than conv validation in general.
#[test]
fn an_invalid_stride_without_a_let_binding_names_the_stride() {
    assert_conv_still_validates(
        CONV2D_STRIDE_ZERO_DIRECT,
        "requires a positive stride, got 0",
        "conv with stride 0 and no let-binding",
    );
}

/// A symbolic spatial dimension cannot be lowered, and the guard says so. A
/// let-bound movement operand must not hide that either.
///
/// REGRESSION: at `948ed5736` this scored 1 with no errors.
#[test]
fn a_symbolic_spatial_dimension_is_refused_through_a_let_bound_expand() {
    assert_conv_still_validates(
        "module Repro.Conv2dSymbolicSpatial\n\
         def f(x: tensor[1, 3, h, 8, f32], k: tensor[4, 3, 3, 3, f32]) = {\n\
           y = expand(x, 0i32, 2i64)\n\
           conv(&y, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n\
         }\n",
        "requires concrete tensor argument metadata",
        "symbolic spatial dimension behind a let-bound expand",
    );
}

/// A well-typed `conv` behind a let-bound `expand`.
///
/// REGRESSION for the diagnostic COUNT, and a LOCK ON A FALSE REJECTION for its
/// content. At `948ed5736` this scored 1 with no errors, which looked like an
/// improvement and was not: the validator was off, and the rows above were
/// passing invalid programs through the same hole.
///
/// The program is well typed. `expand(x, 0i32, 2i64)` over
/// `tensor[1, 3, 8, 8, f32]` satisfies the unit-extent claim at axis 0 and
/// yields `tensor[2, 3, 8, 8, f32]`, so the declared `tensor[2, 4, 6, 6, f32]`
/// is right and inference accepts it. The `conv` validator refuses it anyway,
/// because its own environment holds no entry for `y`.
///
/// That false rejection is PRE-EXISTING, not introduced here: the base refuses
/// this program with the same message and the same score, 0.9571428571428572.
/// It is locked rather than fixed because fixing it means giving the validator
/// a type for `y`, which is either a new derivation arm -- the mechanism this
/// pull request exists to delete -- or the redesign that has the validator read
/// unification's stamped type instead of a private derivation. The second is
/// the right answer, is larger than this pull request, and is owned by
/// chelis#1612. If that issue closes, this row should go red and be updated to
/// assert acceptance.
#[test]
fn a_well_typed_conv_through_a_let_bound_expand_is_still_refused() {
    let diagnostics = ir_diagnostics(
        "module Repro.Conv2dWellTyped\n\
         def f(\n\
           x: tensor[1, 3, 8, 8, f32],\n\
           k: tensor[4, 3, 3, 3, f32],\n\
         ) -> tensor[2, 4, 6, 6, f32] = {\n\
           y = expand(x, 0i32, 2i64)\n\
           conv(&y, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n\
         }\n",
    );
    assert_eq!(
        diagnostics.len(),
        1,
        "exactly one diagnostic: the validator runs, and reports only its \
         missing-metadata refusal: {diagnostics:?}"
    );
    assert!(
        diagnostics[0].contains("requires concrete tensor argument metadata"),
        "the surviving diagnostic is the pre-existing false rejection, not a \
         new one: {diagnostics:?}"
    );
}

/// Negative parity for the marker repair: a genuinely failed derivation must
/// STILL suppress its cascade. Without this, "stop marking `expand`" could be
/// over-applied to "stop marking anything" and nothing would notice.
///
/// A `conv` over a symbolic spatial dimension has no derivable output type,
/// so `y` is a real failed derivation. The consumer must stay silent and let
/// the inner call report once.
///
/// DISPOSITION LOCK: the chained program reports exactly what the single call
/// reports. The single-call control is what makes "one" meaningful; without it
/// the row would also pass against a compiler that reported nothing at all.
#[test]
fn a_genuinely_failed_derivation_still_suppresses_its_cascade() {
    let chained = ir_diagnostics(
        "module Repro.Conv2dCascade\n\
         def f(\n\
           x: tensor[2, 3, h, 8, f32],\n\
           k1: tensor[4, 3, 3, 3, f32],\n\
           k2: tensor[4, 4, 3, 3, f32],\n\
         ) = {\n\
           y = conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n\
           conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n\
         }\n",
    );
    let single = ir_diagnostics(
        "module Repro.Conv2dCascadeControl\n\
         def f(x: tensor[2, 3, h, 8, f32], k1: tensor[4, 3, 3, 3, f32]) = \
         conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n",
    );
    assert_eq!(
        single.len(),
        1,
        "the control must report the inner cause once: {single:?}"
    );
    assert_eq!(
        chained.len(),
        1,
        "chaining a second `conv` onto a failed derivation must not add a \
         second diagnostic; the cascade stays suppressed (RT-205 round-2 F3): \
         {chained:?}"
    );
}

/// The post-inference validator runs on `check_ir_program` only.
///
/// DISPOSITION LOCK on a PRE-EXISTING gap, stated once so the other rows'
/// silence about the second ingress is recorded rather than implied. This pull
/// request does not touch it: `infer/program.rs`, which decides which ingress
/// calls `validate_ir_program`, is byte-identical to the base.
///
/// The bare `conv` with `stride = 0` is the sharpest witness available: the
/// normalizing ingress refuses it and the stamped ingress, which `chelis prove`
/// uses, accepts it. Recorded, not endorsed.
#[test]
fn the_post_inference_validator_runs_only_on_the_ir_ingress() {
    assert!(
        ir_diagnostics(CONV2D_STRIDE_ZERO_DIRECT)
            .iter()
            .any(|m| m.contains("requires a positive stride, got 0")),
        "the normalizing ingress runs the validator"
    );
    assert!(
        typed_diagnostics(CONV2D_STRIDE_ZERO_DIRECT).is_empty(),
        "the stamped ingress runs no post-inference validator, so it accepts \
         a `conv` the other ingress refuses. Pre-existing and out of scope \
         for chelis#668; if this row goes red because the validator was wired \
         into `check_typed_program`, that is an improvement and the row should \
         be updated deliberately"
    );
}
