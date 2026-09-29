//! chelis#2413 (#2586 round 3), spec/10 §3.2 with [05-OP-53]: at a runtime
//! `if` join every lane returns what the taken arm computes when that value
//! satisfies the join's declared type, or traps with the typed extent message
//! when the taken arm's claim is false. An untaken arm's claim never sizes
//! the join. A join whose arms' extents lowering does not prove equal (one
//! origin, or one C identity) is refused with the typed chelis#2583 refusal
//! instead of being sized from either arm; it is never a silent value.
//!
//! The matrix is generated, not sampled: the claimed arm in the `then` or
//! `else` position; its claim from a callee's result type, a local ascription,
//! or an `expand`'s unit claim; the claimed extent 0, a literal equal to the
//! extent the arm computes, a different literal, or a runtime extent equal to
//! or different from (0) that extent; the claimed arm taken or untaken; the
//! other arm unclaimed and either independent of the claimed one (`x[0..3]`)
//! or computed from the same value outside the join, so that the two arms'
//! extents have one origin (for a result claim and an ascription). Each
//! cell's expectation is computed from its parameters, never read from a
//! lane: an independent other arm's extent is not proven equal to the
//! claimed arm's, so those cells expect the typed refusal in every lane
//! (the host interpreter included: it runs `selected` as the kernel C
//! emits, and a kernel's lowering failure is its error, never a
//! fall-through to the interpreter).
//!
//! Lanes: the host interpreter (`chelis eval --file` on `main`), the DAG
//! evaluator (`selected` lowered as a tensor entry, as `exec_compile` runs
//! it; where it declines to lower, the same body's kernel lowering states
//! why), and the whole program's C (`chelis build`, linked and run). A cell
//! C built at 096daea8c (the 40 shared-origin cells) must still build; a C
//! run must agree. Every cell is collected before the assertion, so one red
//! cell never hides a sibling.
//!
//! Evidentiary status: REGRESSION TEST for the three silent values the
//! round-2b reviewer found at dfaefd9a2 ([`the_reviewers_unproven_joins_are_the_taken_arm_or_refused`]:
//! a nested join returned an empty tensor in the host interpreter and the
//! DAG evaluator, a `vmap` row join `[0, 0, 0]` in the DAG evaluator), and
//! for the three shared-origin untaken `else` cells whose claim is 0, which
//! returned an empty tensor at 44c9b23e7; DISPOSITION LOCK for every other
//! cell, for the typed refusal of the 60 independent-arm cells, and for the
//! C build rule (C built the same 40 cells at 096daea8c, 592dc55ce and
//! dfaefd9a2).
#[path = "common/mod.rs"]
mod common;

include!("support/claimed_join.rs");

/// Every cell of the matrix, in every lane: the taken arm's value, or its
/// claim's typed trap; never another value.
#[test]
fn every_lane_returns_the_taken_arm_or_its_claims_trap_at_a_claimed_join() {
    let cells = cells();
    assert_eq!(cells.len(), 100);
    let failures = disagreements(&cells);
    assert!(
        failures.is_empty(),
        "{} of {} cell lanes disagree:\n{}",
        failures.len(),
        cells.len() * 3,
        failures.join("\n")
    );
}
