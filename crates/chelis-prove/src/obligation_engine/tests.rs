//! Obligation-engine tests for the review-3 unifications.
//!
//! U1: ONE produced-value validation chokepoint. A produced opaque value
//! with ANY non-finite representation leaf (NaN OR Inf, in a scalar field,
//! a tensor element, or a nested-record field) must FAIL the obligation
//! fail-closed, on BOTH the scalar and the tensor/record path, under an
//! inequality invariant AND a `!=`/`not(==)` invariant. The historical bug:
//! the all-scalar branch evaluated the predicate through the host runtime,
//! which never reached a finiteness guard, so a scalar NaN under a `!=`
//! invariant shipped as a PASSING obligation (`NaN != C == true`).

use super::*;

/// Run a single-module Surf source through the obligation engine at the
/// given tier and return the outcomes.
fn run(surf: &str, tier: &str) -> Vec<ObligationOutcome> {
    let opts = ObligationRunOptions {
        seed: 0,
        samples: 16,
        smt_timeout_ms: 2000,
        tier: tier.to_string(),
        only: None,
        invariant_min_rate: 0.01,
    };
    match run_surf_source_obligations(surf, &opts).expect("engine run") {
        ObligationRunResult::Ran(o) => o,
        other => panic!("expected a clean module to run, got {other:?}"),
    }
}

/// The single producer obligation of a one-obligation module.
fn only_outcome(outcomes: &[ObligationOutcome]) -> &ObligationOutcome {
    assert_eq!(
        outcomes.len(),
        1,
        "expected exactly one obligation, got {outcomes:?}"
    );
    &outcomes[0]
}

fn synthetic_obligation() -> ObligationProperty {
    ObligationProperty {
        name: "invariant:Probability:make".to_string(),
        source_type: "Probability".to_string(),
        producer: "make".to_string(),
        is_constant: false,
        position: ProducedPosition::Direct,
        meta: ObligationMeta {
            obligation_kind: "invariant_producer".to_string(),
            source_type: "Probability".to_string(),
            producer: "make".to_string(),
        },
    }
}

#[test]
fn failed_obligation_without_counterexample_is_unsupported_not_failed() {
    let ob = synthetic_obligation();
    let out = outcome(
        &ob,
        ObligationStatus::Failed,
        ObligationTier::Smt,
        0,
        0,
        None,
        None,
    );
    assert_eq!(out.status, ObligationStatus::Unsupported);
    assert_eq!(out.composite_verdict, CompositeVerdict::Unsupported);
    assert!(out.counterexample.is_none());
    assert!(
        out.reason
            .as_deref()
            .is_some_and(|reason| reason.contains("missing counterexample"))
    );
}

// --- scalar field, +Inf, inequality invariant ---

const SCALAR_INF_INEQ: &str = "module M.Prob
export (make)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def make(x: f32) -> Probability = Probability { value: 1.0 / 0.0 }
";

#[test]
fn u1_scalar_inf_under_inequality_fails_closed_fuzz() {
    // A producer whose scalar field is +Inf must FAIL the obligation: +Inf
    // is not in [0, 1], and more fundamentally not a finite inhabitant.
    let outcomes = run(SCALAR_INF_INEQ, "fuzz-only");
    assert_eq!(
        only_outcome(&outcomes).status,
        ObligationStatus::Failed,
        "scalar +Inf under an inequality invariant must fail-closed (fuzz)"
    );
}

#[test]
fn u1_scalar_inf_under_inequality_fails_closed_auto() {
    let outcomes = run(SCALAR_INF_INEQ, "auto");
    assert_ne!(
        only_outcome(&outcomes).status,
        ObligationStatus::Passed,
        "scalar +Inf under an inequality invariant must not pass (auto)"
    );
}

// --- scalar field, NaN, `!=` (negation-shaped) invariant ---
//
// The flagship of the historical fail-open: `NaN != C` is true under
// strict IEEE, so a NaN scalar under a `!=` invariant slipped through the
// host-runtime predicate eval that lacked a finiteness guard.

const SCALAR_NAN_NEQ: &str = "module M.Prob
export (make)
@opaque
@invariant(p) p.value != 0.5
type Probability =
  | Probability { value: f32 }
def make(x: f32) -> Probability = Probability { value: 0.0 / 0.0 }
";

#[test]
fn u1_scalar_nan_under_neq_fails_closed_fuzz() {
    let outcomes = run(SCALAR_NAN_NEQ, "fuzz-only");
    assert_eq!(
        only_outcome(&outcomes).status,
        ObligationStatus::Failed,
        "scalar NaN under a `!=` invariant must fail-closed, not pass via `NaN != C`"
    );
}

#[test]
fn u1_scalar_nan_under_neq_fails_closed_auto() {
    let outcomes = run(SCALAR_NAN_NEQ, "auto");
    assert_ne!(
        only_outcome(&outcomes).status,
        ObligationStatus::Passed,
        "scalar NaN under a `!=` invariant must not pass (auto)"
    );
}

// --- negative parity: a clean scalar producer still passes ---

const SCALAR_CLEAN: &str = "module M.Prob
export (make)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def make(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
";

#[test]
fn u1_clean_scalar_producer_still_passes() {
    // The finiteness guard must not break a legitimately-passing producer.
    let outcomes = run(SCALAR_CLEAN, "fuzz-only");
    assert_eq!(
        only_outcome(&outcomes).status,
        ObligationStatus::Passed,
        "a clean scalar producer still passes under the unified chokepoint"
    );
}

// --- tensor field, non-finite element, `sum` band invariant ---
//
// The tensor/record path already had the guard, but the unified chokepoint
// must keep walking EVERY tensor element. A producer whose tensor field
// carries a non-finite element must fail-closed.

const TENSOR_NONFINITE: &str = "module M.Simplex
export (make)
@opaque
@invariant(p) sum(p.weights) >= 0.0 && sum(p.weights) <= 100.0
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def make(x: f32) -> Simplex =
  Simplex { weights: to_tensor([1.0 / 0.0, 0.0, 0.0]) }
";

#[test]
fn u1_tensor_nonfinite_element_fails_closed_fuzz() {
    let outcomes = run(TENSOR_NONFINITE, "fuzz-only");
    assert_eq!(
        only_outcome(&outcomes).status,
        ObligationStatus::Failed,
        "a non-finite tensor element must fail-closed (every element is walked)"
    );
}

// --- there is exactly ONE produced-value finiteness chokepoint ---
//
// This is a structural assertion enforced by the source-level grep in the
// final report, mirrored here as a documentation anchor: the scalar and
// tensor/record validation share `validate_produced_env`, and the
// generator's `validate_env` shares the same `any_non_finite` helper. The
// runtime predicate-eval branch (`eval_obligation_predicate`) is removed.
// --- there is exactly ONE produced-value finiteness chokepoint ---
//
// The scalar and tensor/record validation share `validate_produced_env`,
// and the generator's `validate_env` shares the same `any_non_finite`
// helper. The runtime predicate-eval branch (`eval_obligation_predicate`)
// is removed. This compile-time reference keeps the chokepoint name from
// silently drifting.
#[test]
fn u1_chokepoint_signature_is_stable() {
    let _ = validate_produced_env
        as fn(&ExecutionValue, &OpaqueInvariant, &crate::solver::SmtExpr) -> Result<bool, String>;
}

// ===================================================================
// Review 5: int-width consistency on the SAMPLING axis. Every integer
// width (int8/int16/int32/int64) samples as a width-clamped INTEGER at
// Tier C (not a float), via the single-source `int_sample_bounds`, so an
// int-field/param obligation runs and passes at the fuzz tier.
// ===================================================================

/// A producer obligation over an int-width field+param runs at Tier C
/// (fuzz-only, no SMT) with integer-valued samples and PASSES, looped over
/// every integer width. The producer guards `x >= 0` so the constructed
/// value satisfies `c.n >= 0` for every accepted sample; a float-sampled
/// int param (the pre-fix bug for int8/int16) would mis-type the producer
/// call and error rather than pass cleanly.
#[test]
fn w5_int_width_field_param_samples_as_integer_at_tier_c() {
    for width in ["int8", "int16", "int32", "int64"] {
        let surf = format!(
            "module M
export (make)
@opaque
@invariant(c) c.n >= ({lo} : {width})
type Counter =
  | Counter {{ n: {width} }}
def make(x: {width}) -> Option[Counter] =
  if x >= ({lo} : {width}) then Some(Counter {{ n: x }}) else None
",
            lo = 0,
            width = width
        );
        let outcomes = run(&surf, "fuzz-only");
        let o = only_outcome(&outcomes);
        assert_eq!(
            o.status,
            ObligationStatus::Passed,
            "the `{width}` obligation passes at Tier C with integer samples: {o:?}"
        );
        assert_eq!(
            o.proof_tier,
            ObligationTier::Fuzz,
            "fuzz-only stays at Tier C for {width}"
        );
        assert!(o.samples > 0, "collected fuzz samples for {width}");
    }
}

#[test]
fn tier_c_obligation_counterexample_records_accepted_shrink_steps() {
    let surf = "module M
export (make)
@opaque
@invariant(p) p.value > 5.0
type Probability =
  | Probability { value: f32 }
def make(x: f32) -> Option[Probability] =
  if x >= 0.0 then Some(Probability { value: x }) else None
";
    let outcomes = run(surf, "fuzz-only");
    let o = only_outcome(&outcomes);
    assert_eq!(o.status, ObligationStatus::Failed, "{o:?}");
    assert!(
        o.shrink_steps > 0,
        "failed fuzz obligation should record accepted shrink steps: {o:?}"
    );
    assert_eq!(
        o.counterexample.as_ref().and_then(|cx| cx["x"].as_f64()),
        Some(0.0),
        "zero is still a failing producer input for p.value > 5.0: {o:?}"
    );
}

/// The fuzz-sampling bounds are single-source and width-clamped: an int8
/// samples in [-128, 127], an int16 in [-1000, 1000] (the convenience
/// window, since [-32768, 32767] exceeds it), etc. -- proving the sampling
/// axis routes through `int_sample_bounds` / `Prim::integer_fuzz_bounds`,
/// not an independent [-1000, 1000] for every width.
#[test]
fn w5_int_sample_bounds_are_width_clamped_single_source() {
    assert_eq!(
        crate::opaque::int_sample_bounds("int8"),
        Some((-128, 127)),
        "int8 sampling is clamped to its representable range"
    );
    assert_eq!(
        crate::opaque::int_sample_bounds("int16"),
        Some((-1000, 1000)),
        "int16 keeps the convenience window (within its range)"
    );
    assert_eq!(
        crate::opaque::int_sample_bounds("int32"),
        Some((-1000, 1000))
    );
    assert_eq!(
        crate::opaque::int_sample_bounds("int64"),
        Some((-1000, 1000))
    );
    assert_eq!(
        crate::opaque::int_sample_bounds("f32"),
        None,
        "a non-integer prim has no integer sample bounds"
    );
}
