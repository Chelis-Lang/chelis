//! The chelis#218 `Cons` join's widening of mismatched literal element dims,
//! pinned as a DISPOSITION LOCK.
//!
//! `Cons`'s per-axis join keeps equal literals and equal names, and widens a
//! genuinely mismatched pair of concrete literals to `Dim::Wildcard`. That is
//! deliberate: chelis#218 added it so a bare `concat([a, b], axis)` accepts
//! ragged concrete axes, and chelis#272 tightened it so a pair involving a
//! rigid dimension VARIABLE unifies instead of widening. The rule is stated at
//! the join itself, in `crates/chelis-types/src/infer/app.rs`.
//!
//! # Why this file exists
//!
//! The widening used to have a witness inside
//! `slice_c_composite_carriers.rs`, which asserted that a PENDING list element
//! must not reach the widening branch: a wildcard satisfies either candidate
//! shape, so a pending element that reached it would have been accepted with
//! the choice silently unmade. `spec/04-type-system.md` §4.7.2 gives `expand`
//! and `insert` one result shape each, so no element is ever pending, that
//! witness has no instance, and it went with the file (chelis#1277 S2b).
//!
//! What survives is the widening itself, which is reachable with no deferral
//! at all and which nothing else in the tree pins. This file records what it
//! does TODAY. It is not a regression test and it is not a claim that the
//! behaviour below is right: it asserts the current disposition so that a
//! change to it has to be deliberate, and so that the reader of chelis#218 or
//! chelis#272 has an executable statement of the rule to point at.
//!
//! Every row here passes on the base commit as well, because S2b changes
//! nothing about `Cons`.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn check(source: &str) -> Result<(), Vec<CheckError>> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    match check_typed_program(&deep) {
        Ok(_) => Ok(()),
        Err(report) => Err(report.errors),
    }
}

fn summary(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Mismatched concrete literals at the same rank widen to a wildcard, and a
/// wildcard satisfies a concrete element type the list never contained.
///
/// This is the widening's observable consequence, and it needs no deferral to
/// reach: both elements are fully resolved function parameters.
#[test]
fn mismatched_literal_element_dims_widen_and_satisfy_an_unrelated_extent() {
    let accepted = check(
        "def sink(xs: List[tensor[9, f32]]) -> int32 = 0\n\
         def f(a: tensor[2, f32], b: tensor[3, f32]) -> int32 = sink([a, b])\n",
    );
    assert!(
        accepted.is_ok(),
        "chelis#218's join widens `2` against `3` to a wildcard, and the \
         wildcard then satisfies the declared `tensor[9, f32]` element. If this \
         starts rejecting, the widening has been narrowed and #218/#272 should \
         say so: {}",
        accepted.err().map(|e| summary(&e)).unwrap_or_default()
    );
}

/// The same list with no consumer is also accepted, which locates the
/// acceptance in the join rather than in the parameter's type.
#[test]
fn a_mismatched_literal_element_list_is_accepted_on_its_own() {
    let accepted = check(
        "def f(a: tensor[2, f32], b: tensor[3, f32]) -> int32 = {\n  \
         xs = [a, b]\n  \
         0\n\
         }\n",
    );
    assert!(
        accepted.is_ok(),
        "the join accepts the list itself: {}",
        accepted.err().map(|e| summary(&e)).unwrap_or_default()
    );
}

/// The boundary the widening does NOT cross: a rank disagreement is refused
/// with its own diagnostic, so the wildcard is per-axis and not a blanket
/// element wildcard.
#[test]
fn a_mismatched_element_rank_is_still_rejected() {
    let errors = check(
        "def sink(xs: List[tensor[9, f32]]) -> int32 = 0\n\
         def f(a: tensor[2, f32], b: tensor[3, 4, f32]) -> int32 = sink([a, b])\n",
    )
    .expect_err("rank-uniform elements are required");
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("list element rank mismatch")),
        "the rank rule is separate from the per-axis join and still applies: {}",
        summary(&errors)
    );
}
