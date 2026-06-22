//! WI-10 AD-rail SHAPE tests.
//!
//! No in-tree engine discharges `BoxRange` yet (cvc5 fits only SMT; Beacon is
//! out-of-tree), so the gradient goals are verified by SHAPE + no-fit honesty,
//! NOT by a real bound proof. Every positive case has a negative twin:
//!
//! - a real multi-input gradient program fans out to N box/range goals, one per
//!   gradient target, each with a populated `IrHandle` (gradient-DAG sha256 +
//!   that target's root index), a name-keyed deterministically-sorted input
//!   box, and a single scalar output; all N share ONE gradient-DAG hash;
//! - each gradient goal dispatched in-tree is NO-FIT -> the canonical
//!   Untrusted + empty -> `CompositeVerdict::Unsupported`, never green;
//! - a test-only mock `BoxRange` engine proves the AD -> producer -> dispatch
//!   -> engine path end-to-end WITHOUT Beacon;
//! - an unknown gradient target -> a typed error (no partial / wrong-root goal);
//! - a non-finite gradient DAG / an inverted output range -> rejected at the
//!   WI-3 producer boundary.

use std::collections::BTreeMap;

use chelis_compiler_api::schema::{
    GradResult, SourceKind, WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagNode, WireDimInfo, WireRiscOp,
    WireTensorType,
};

use super::*;
use crate::composition::{
    AssumptionDischarge, AssumptionRecord, CompositeVerdict, DischargeMethod, NonVacuityRecord,
    rollup_composite,
};
use crate::discharge::{
    Discharge, DischargeEngine, Goal, GoalShape, IntervalBox, Qualifier, QualifierSet, Soundness,
};
use crate::engine_registry::DischargeRegistry;
use crate::tier_b::TierBResult;

/// A scalar loss differentiated w.r.t. TWO named inputs `a` and `b`. `grad`
/// returns one gradient root per wrt name (probed: `{"a": _, "b": _}`), so this
/// is the canonical multi-target fan-out fixture.
const TWO_INPUT_LOSS: &str = "a = (a : tensor[4, f32])\n\
                              b = (b : tensor[4, f32])\n\
                              loss = (mean(add(mul(a, a), b), 0) : tensor[f32])\n";

fn input_box(dims: &[(&str, f64, f64)]) -> IntervalBox {
    IntervalBox {
        dims: dims
            .iter()
            .map(|(n, lo, hi)| (n.to_string(), *lo, *hi))
            .collect(),
    }
}

fn target_range(target: &str, lo: f64, hi: f64) -> GradTargetRange {
    GradTargetRange {
        target: target.to_string(),
        lo,
        hi,
    }
}

/// The canonical multi-target request over [`TWO_INPUT_LOSS`]: bound both
/// gradients `d loss / d a` and `d loss / d b` over the same input box.
fn two_target_request() -> AdRailRequest {
    AdRailRequest {
        source: TWO_INPUT_LOSS.to_string(),
        source_kind: SourceKind::Surf,
        output_name: "loss".to_string(),
        wrt_names: vec!["a".to_string(), "b".to_string()],
        input_box: input_box(&[("a", -1.0, 1.0), ("b", -1.0, 1.0)]),
        target_ranges: vec![
            target_range("a", -10.0, 10.0),
            target_range("b", -10.0, 10.0),
        ],
    }
}

// ===========================================================================
// Positive: a real multi-input gradient fans out to N per-target goals,
// each a single scalar output addressed by its own root, sharing one hash.
// ===========================================================================

#[test]
fn multi_target_gradient_fans_out_to_per_target_goals() {
    let goals =
        grad_goals_from_request(&two_target_request()).expect("a two-target gradient fans out");

    // One goal per requested gradient target, in request order.
    assert_eq!(goals.len(), 2, "one goal per gradient target");
    assert_eq!(
        goals.iter().map(|g| g.target.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"],
        "goals are returned in request order, named by gradient target"
    );

    for goal in &goals {
        // Each goal is a BoxRange goal, not an SMT goal.
        assert!(
            matches!(goal.extracted.goal.shape, GoalShape::BoxRange { .. }),
            "each gradient goal is a BoxRange goal"
        );
        // The IrHandle is populated: gradient-DAG hash + a root index.
        assert!(
            goal.extracted.goal.ir.is_populated(),
            "each gradient goal carries a populated IR handle"
        );
        // Single scalar output, named by the gradient target.
        let GoalShape::BoxRange { inputs, output } = &goal.extracted.goal.shape else {
            unreachable!("checked above");
        };
        assert_eq!(
            output.output, goal.target,
            "the goal's single output is named by its gradient target"
        );
        // The input box is name-keyed and deterministically name-sorted.
        let names: Vec<&str> = inputs.dims.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(names, vec!["a", "b"], "input box dims are sorted by name");
        // The handle's hash is exactly the sha256 of the carried gradient bytes.
        let hash = goal
            .extracted
            .goal
            .ir
            .dag_hash()
            .expect("populated handle carries a dag hash");
        assert_eq!(goal.extracted.dag_hash, hash);
        assert_eq!(hash.len(), 64, "sha256 is 64 lowercase-hex chars");
        assert!(
            hash.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "hash must be lowercase hex, got {hash}"
        );
    }
}

#[test]
fn fan_out_targets_have_distinct_roots_sharing_one_gradient_dag_hash() {
    let goals =
        grad_goals_from_request(&two_target_request()).expect("a two-target gradient fans out");

    // DISTINCT root indices: d loss / d a and d loss / d b are different
    // gradient roots.
    let a_root = goals[0]
        .extracted
        .goal
        .ir
        .root_index()
        .expect("a root index");
    let b_root = goals[1]
        .extracted
        .goal
        .ir
        .root_index()
        .expect("b root index");
    assert_ne!(
        a_root, b_root,
        "each gradient target is addressed by its OWN root index"
    );

    // SHARED gradient-DAG hash + bytes: the gradient is lowered ONCE.
    assert_eq!(
        goals[0].extracted.dag_hash, goals[1].extracted.dag_hash,
        "all fan-out goals address the one gradient-DAG artifact (lowered once)"
    );
    assert_eq!(
        goals[0].extracted.wire_dag_bytes, goals[1].extracted.wire_dag_bytes,
        "the gradient artifact bytes are identical across the fan-out"
    );

    // The serialized gradient artifact round-trips as a v1 WireDag, and each
    // goal's root index addresses a real root of it.
    let parsed: WireDag = serde_json::from_slice(&goals[0].extracted.wire_dag_bytes)
        .expect("the gradient bytes parse back as a WireDag");
    parsed
        .validate_schema_version()
        .expect("the gradient artifact is a supported version");
    for goal in &goals {
        let root = goal.extracted.goal.ir.root_index().expect("root index") as usize;
        assert!(
            parsed.roots.contains(&root),
            "each goal's root index is a real root of the gradient DAG"
        );
    }
}

#[test]
fn single_target_gradient_is_one_goal() {
    // A gradient w.r.t. ONE input is still a fan-out of one: one scalar-output
    // goal, no packed vector.
    let request = AdRailRequest {
        source: TWO_INPUT_LOSS.to_string(),
        source_kind: SourceKind::Surf,
        output_name: "loss".to_string(),
        wrt_names: vec!["a".to_string()],
        input_box: input_box(&[("a", -1.0, 1.0)]),
        target_ranges: vec![target_range("a", -5.0, 5.0)],
    };
    let goals = grad_goals_from_request(&request).expect("a one-target gradient fans out to one");
    assert_eq!(goals.len(), 1);
    assert_eq!(goals[0].target, "a");
    assert!(matches!(
        goals[0].extracted.goal.shape,
        GoalShape::BoxRange { .. }
    ));
}

// ===========================================================================
// No-fit: each in-tree-dispatched gradient goal is Unsupported, never green.
// ===========================================================================

#[test]
fn each_gradient_goal_dispatched_in_tree_is_no_fit_unsupported() {
    let goals = grad_goals_from_request(&two_target_request()).expect("fan-out");
    // The in-tree builtin registry has no BoxRange engine (cvc5 fits only SMT;
    // Beacon is out-of-tree), so every gradient goal is NO-FIT.
    let registry = DischargeRegistry::with_builtin_engines();
    let dispatched = dispatch_grad_goals(&registry, &goals, 1_000);

    assert_eq!(dispatched.len(), 2, "one discharge per gradient goal");
    for (target, discharge) in &dispatched {
        assert!(
            ["a", "b"].contains(&target.as_str()),
            "discharge is paired with its gradient target"
        );
        assert_no_fit_lattice_membership(discharge);
    }
}

/// Pin the EXACT no-fit lattice membership the brief requires:
/// `Soundness::Untrusted` + empty `QualifierSet` + `TierBResult::Error`,
/// projecting to `CompositeVerdict::Unsupported` (never `Proven`, never a
/// silent pass). Mirrors the WI-9 dispatcher's own assertion.
fn assert_no_fit_lattice_membership(discharge: &Discharge) {
    assert_eq!(
        discharge.soundness(),
        Soundness::Untrusted,
        "no-fit must be untrusted (bottom of the lattice)"
    );
    assert!(
        discharge.qualifier_set().is_empty(),
        "no-fit must carry no qualifier (no laundered badge)"
    );
    assert!(
        matches!(discharge.result(), TierBResult::Error(_)),
        "no-fit must be an Error result, never Proved/Disproved"
    );

    // Project through the WI-6 verdict algebra: even composed onto a PROVEN
    // base, a no-fit gradient discharge degrades the composite to Unsupported
    // and never launders into Proven.
    let status = discharge_status(discharge);
    assert_eq!(
        status, "unsupported",
        "a no-fit (untrusted/empty/Error) discharge maps to the `unsupported` status"
    );
    let record = AssumptionRecord::new(
        "no_fit_grad_dependency",
        Some(AssumptionDischarge::new(
            DischargeMethod::Smt,
            serde_json::json!({ "status": status }),
        )),
        Some(NonVacuityRecord::established(
            serde_json::json!({ "result": "sat" }),
        )),
    );
    let verdict = rollup_composite(CompositeVerdict::Proven, &[record]);
    assert_eq!(
        verdict,
        CompositeVerdict::Unsupported,
        "the no-fit gradient discharge must render Unsupported"
    );
    assert_ne!(
        verdict,
        CompositeVerdict::Proven,
        "the no-fit gradient discharge must never render Proven, even over a proven base"
    );
}

/// The prove-flow status a discharge maps to. Exact + green is `proved`;
/// everything else (the no-fit case) is `unsupported`.
fn discharge_status(discharge: &Discharge) -> &'static str {
    match (discharge.soundness(), discharge.result()) {
        (Soundness::Exact, TierBResult::Proved) => "proved",
        (Soundness::Exact, TierBResult::Disproved(_)) => "failed",
        _ => "unsupported",
    }
}

// ===========================================================================
// The AD -> producer -> dispatch -> engine path works WITHOUT Beacon (mock).
// ===========================================================================

/// A test-only sound-over-approximation `BoxRange` engine, standing in for
/// Beacon's interval engine so the full rail can be exercised in-tree without
/// the out-of-tree shell. Mirrors WI-9's mock.
#[derive(Debug)]
struct MockBoxRangeEngine;

impl DischargeEngine for MockBoxRangeEngine {
    fn name(&self) -> &'static str {
        "mock_interval"
    }

    fn fitness(&self, goal: &Goal) -> bool {
        matches!(goal.shape, GoalShape::BoxRange { .. })
    }

    fn discharge(&self, _goal: &Goal, _timeout_ms: u64) -> Discharge {
        Discharge::new(
            Soundness::SoundApproximate,
            QualifierSet::from_iter_kinds([Qualifier::SoundOverApproximation]),
            TierBResult::Proved,
            serde_json::json!({ "engine": "mock_interval" }),
        )
        .expect("mock discharge satisfies the integrity invariant")
    }
}

#[test]
fn gradient_goal_routes_to_a_registered_box_range_engine_without_beacon() {
    let goals = grad_goals_from_request(&two_target_request()).expect("fan-out");

    // Register the mock BoxRange engine on top of the builtin engines: now the
    // box/range lane has a fit, proving the AD -> producer -> dispatch -> engine
    // path end-to-end without Beacon.
    let mut registry = DischargeRegistry::with_builtin_engines();
    registry.register(Box::new(MockBoxRangeEngine));

    // The mock is selected for each gradient goal.
    for goal in &goals {
        assert_eq!(
            registry.selected_engine_name(&goal.extracted.goal),
            Some("mock_interval"),
            "the registered BoxRange engine is selected for a gradient goal"
        );
    }

    let dispatched = dispatch_grad_goals(&registry, &goals, 1_000);
    assert_eq!(dispatched.len(), 2);
    for (target, discharge) in &dispatched {
        assert!(["a", "b"].contains(&target.as_str()));
        assert_eq!(
            discharge.soundness(),
            Soundness::SoundApproximate,
            "the mock engine's discharge flows back through the rail"
        );
        assert!(
            discharge
                .qualifier_set()
                .contains(Qualifier::SoundOverApproximation)
        );
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(
            discharge.evidence().get("engine").and_then(|v| v.as_str()),
            Some("mock_interval"),
            "the discharge came from the mock interval engine"
        );
    }
}

// ===========================================================================
// Negative: an unknown gradient target fails the whole fan-out.
// ===========================================================================

#[test]
fn unknown_gradient_target_fails_the_whole_fan_out() {
    // `c` is not a differentiated input, so no gradient root addresses it. The
    // whole fan-out must fail with the AD-rail-specific diagnostic, never a
    // partial goal set or a goal addressed by a wrong root.
    let request = AdRailRequest {
        source: TWO_INPUT_LOSS.to_string(),
        source_kind: SourceKind::Surf,
        output_name: "loss".to_string(),
        wrt_names: vec!["a".to_string(), "b".to_string()],
        input_box: input_box(&[("a", -1.0, 1.0), ("b", -1.0, 1.0)]),
        // First target is real, second is bogus: proves the failure is the
        // whole fan-out, not a per-goal silent drop.
        target_ranges: vec![target_range("a", -10.0, 10.0), target_range("c", -1.0, 1.0)],
    };
    let err = grad_goals_from_request(&request)
        .expect_err("an unknown gradient target must fail the fan-out");
    match err {
        AdRailError::UnknownGradTarget { target, available } => {
            assert_eq!(target, "c");
            assert!(
                available.contains(&"a".to_string()) && available.contains(&"b".to_string()),
                "available targets list the real gradient targets, got {available:?}"
            );
            let mut sorted = available.clone();
            sorted.sort();
            assert_eq!(available, sorted, "available targets are sorted");
        }
        other => panic!("expected UnknownGradTarget, got {other:?}"),
    }
}

#[test]
fn grad_failure_surfaces_a_typed_grad_error() {
    // A non-scalar output cannot be differentiated; the checked grad path
    // rejects it, and the rail surfaces GradFailed rather than a fabricated
    // goal.
    let request = AdRailRequest {
        source: "x = (x : tensor[4, f32])\n\
                 out = (add(copy(x), x) : tensor[4, f32])\n"
            .to_string(),
        source_kind: SourceKind::Surf,
        output_name: "out".to_string(),
        wrt_names: vec!["x".to_string()],
        input_box: input_box(&[("x", -1.0, 1.0)]),
        target_ranges: vec![target_range("x", -1.0, 1.0)],
    };
    let err = grad_goals_from_request(&request)
        .expect_err("a non-scalar output cannot be differentiated");
    assert!(
        matches!(err, AdRailError::GradFailed(_)),
        "expected GradFailed, got {err:?}"
    );
}

#[test]
fn unknown_wrt_name_surfaces_a_typed_grad_error() {
    // A wrt name that is not a forward node fails inside `grad`, surfacing
    // GradFailed at the rail boundary.
    let request = AdRailRequest {
        source: TWO_INPUT_LOSS.to_string(),
        source_kind: SourceKind::Surf,
        output_name: "loss".to_string(),
        wrt_names: vec!["nope".to_string()],
        input_box: input_box(&[("nope", -1.0, 1.0)]),
        target_ranges: vec![target_range("nope", -1.0, 1.0)],
    };
    let err = grad_goals_from_request(&request).expect_err("an unknown wrt name must fail in grad");
    assert!(
        matches!(err, AdRailError::GradFailed(_)),
        "expected GradFailed, got {err:?}"
    );
}

// ===========================================================================
// Negative: the WI-3 producer boundary checks apply to the gradient DAG.
// These exercise the pure fan-out core over a constructed GradResult so the
// gradient DAG's contents are controlled directly.
// ===========================================================================

/// A single-node gradient `GradResult` carrying `op`, rooted at node 0, whose
/// sole gradient target `t` maps to root 0.
fn single_op_grad_result(op: WireRiscOp) -> GradResult {
    let mut grad_nodes_by_name = BTreeMap::new();
    grad_nodes_by_name.insert("t".to_string(), 0usize);
    GradResult {
        dag: WireDag {
            schema_version: WIRE_DAG_SCHEMA_VERSION,
            nodes: vec![WireDagNode {
                id: 0,
                op,
                inputs: vec![],
                output_type: WireTensorType {
                    dims: vec![WireDimInfo::Lit { size: 1 }],
                    precision: "f32".to_string(),
                },
            }],
            roots: vec![0],
        },
        output_node: 0,
        grad_nodes_by_name,
        forward_nodes_by_name: BTreeMap::new(),
    }
}

#[test]
fn non_finite_in_gradient_dag_is_rejected_at_the_producer_boundary() {
    // A non-finite float in the gradient DAG must be rejected before hashing
    // (the WI-3 finite-float guard applies via box_range_goal_from_wire_dag),
    // surfacing as GoalConstruction(NonFiniteValue), not a corrupted artifact.
    let grad = single_op_grad_result(WireRiscOp::Const {
        value: f64::INFINITY,
    });
    let err = fan_out_grad_goals(
        &grad,
        &input_box(&[("x", -1.0, 1.0)]),
        &[target_range("t", 0.0, 1.0)],
    )
    .expect_err("a non-finite gradient DAG must be rejected");
    assert!(
        matches!(
            err,
            AdRailError::GoalConstruction(GraphExtractError::NonFiniteValue { .. })
        ),
        "expected GoalConstruction(NonFiniteValue), got {err:?}"
    );
}

#[test]
fn finite_gradient_dag_passes_the_producer_boundary() {
    // The positive twin: a finite gradient DAG produces a populated goal.
    let grad = single_op_grad_result(WireRiscOp::Const { value: 1.0 });
    let goals = fan_out_grad_goals(
        &grad,
        &input_box(&[("x", -1.0, 1.0)]),
        &[target_range("t", 0.0, 1.0)],
    )
    .expect("a finite gradient DAG passes the boundary");
    assert_eq!(goals.len(), 1);
    assert!(goals[0].extracted.goal.ir.is_populated());
    assert_eq!(goals[0].extracted.goal.ir.root_index(), Some(0));
}

#[test]
fn inverted_output_range_on_a_gradient_target_is_rejected_as_ill_formed() {
    // An inverted (lo > hi) range on a gradient target must surface as an
    // ill-formed goal, not a goal asserting an empty region.
    let grad = single_op_grad_result(WireRiscOp::Const { value: 1.0 });
    let err = fan_out_grad_goals(
        &grad,
        &input_box(&[("x", -1.0, 1.0)]),
        &[target_range("t", 10.0, 1.0)],
    )
    .expect_err("an inverted gradient range must be rejected");
    assert!(
        matches!(
            err,
            AdRailError::GoalConstruction(GraphExtractError::IllFormedGoal(_))
        ),
        "expected GoalConstruction(IllFormedGoal), got {err:?}"
    );
}

#[test]
fn non_v1_gradient_dag_is_rejected_at_the_producer_boundary() {
    // A future-version gradient DAG must be rejected, not silently hashed.
    let mut grad = single_op_grad_result(WireRiscOp::Const { value: 1.0 });
    grad.dag.schema_version = WIRE_DAG_SCHEMA_VERSION + 1;
    let err = fan_out_grad_goals(
        &grad,
        &input_box(&[("x", -1.0, 1.0)]),
        &[target_range("t", 0.0, 1.0)],
    )
    .expect_err("a future-version gradient DAG must be rejected");
    assert!(
        matches!(
            err,
            AdRailError::GoalConstruction(GraphExtractError::SchemaRejected(_))
        ),
        "expected GoalConstruction(SchemaRejected), got {err:?}"
    );
}

#[test]
fn unknown_target_on_constructed_grad_result_fails_the_fan_out() {
    // The unknown-target guard is in the pure fan-out core, independent of
    // lowering: a target absent from grad_nodes_by_name fails with
    // UnknownGradTarget.
    let grad = single_op_grad_result(WireRiscOp::Const { value: 1.0 });
    let err = fan_out_grad_goals(
        &grad,
        &input_box(&[("x", -1.0, 1.0)]),
        &[target_range("not_a_target", 0.0, 1.0)],
    )
    .expect_err("a target absent from grad_nodes_by_name must fail");
    match err {
        AdRailError::UnknownGradTarget { target, available } => {
            assert_eq!(target, "not_a_target");
            assert_eq!(available, vec!["t".to_string()]);
        }
        other => panic!("expected UnknownGradTarget, got {other:?}"),
    }
}
