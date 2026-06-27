//! End-to-end integration test: chelis → BeaconShim → real chelis-beacon binary.
//!
//! Requires CHELIS_BEACON_BIN to point at the real beacon binary.
//! Run with: CHELIS_BEACON_BIN=/path/to/chelis-beacon \
//!           cargo nextest run -p chelis-prove --test beacon_e2e -- --ignored

use chelis_compiler_api::schema::SourceKind;
use chelis_prove::discharge::{IntervalBox, OutputRange, Qualifier, Soundness};
use chelis_prove::engine_registry::DischargeRegistry;
use chelis_prove::graph_extract::box_range_goal_from_source;
use chelis_prove::{BeaconShim, DischargeEngine, WireDagByteStore};

fn beacon_bin() -> Option<String> {
    std::env::var("CHELIS_BEACON_BIN").ok()
}

fn input_box(dims: &[(&str, f64, f64)]) -> IntervalBox {
    IntervalBox {
        dims: dims
            .iter()
            .map(|(n, lo, hi)| (n.to_string(), *lo, *hi))
            .collect(),
    }
}

fn output_range(name: &str, lo: f64, hi: f64) -> OutputRange {
    OutputRange {
        output: name.to_string(),
        lo,
        hi,
    }
}

/// Build a simple forward DAG: f(x) = x * x (scalar), then extract a BoxRange goal.
/// x in [0,1], output in [0,1] but Beacon's interval arithmetic over-approximates
/// so we use wider claimed ranges for provability.
fn simple_square_goal(
    output_lo: f64,
    output_hi: f64,
) -> chelis_prove::ExtractedGoal {
    let source = "x = (x : tensor[f32])\nout = (mul(x, x) : tensor[f32])\n";
    box_range_goal_from_source(
        source,
        SourceKind::Surf,
        input_box(&[("x", 0.0, 1.0)]),
        output_range("out", output_lo, output_hi),
    )
    .expect("extraction must succeed for a trivial square program")
}

#[test]
#[ignore] // requires CHELIS_BEACON_BIN
fn beacon_e2e_forward_goal_sound_green() {
    let bin = beacon_bin().expect("CHELIS_BEACON_BIN must be set");
    let store = WireDagByteStore::new();
    // x in [0,1], f(x) = x^2 in [0,1]; use wider claimed range [-1,2] to
    // accommodate Beacon's interval over-approximation.
    let extracted = simple_square_goal(-1.0, 2.0);
    store.insert_extracted(&extracted);

    let shim = BeaconShim::new(&bin, store);
    assert!(
        shim.fitness(&extracted.goal),
        "BeaconShim must fit a BoxRange goal"
    );

    let discharge = shim.discharge(&extracted.goal, 10_000);
    assert_eq!(
        discharge.soundness(),
        Soundness::SoundApproximate,
        "Beacon proved goal must be SoundApproximate, got {:?}",
        discharge
    );
    assert!(
        discharge
            .qualifier_set()
            .contains(Qualifier::SoundOverApproximation),
        "discharge must carry SoundOverApproximation qualifier"
    );
}

#[test]
#[ignore] // requires CHELIS_BEACON_BIN
fn beacon_e2e_forward_goal_wrong_range_not_green() {
    let bin = beacon_bin().expect("CHELIS_BEACON_BIN must be set");
    let store = WireDagByteStore::new();
    // x in [0,1], f(x) = x^2 is in [0,1], NOT [10,20] — Beacon must reject
    let extracted = simple_square_goal(10.0, 20.0);
    store.insert_extracted(&extracted);

    let shim = BeaconShim::new(&bin, store);
    let discharge = shim.discharge(&extracted.goal, 10_000);
    // Must NOT be a Proved verdict — Beacon should refute or fail, never prove
    // a wrong range. A verified refutation is still SoundApproximate (it's a
    // sound over-approximation that disproves containment), so check the result.
    assert!(
        !matches!(discharge.result(), chelis_prove::tier_b::TierBResult::Proved),
        "a wrong-range goal must NOT be Proved by Beacon, got {:?}",
        discharge
    );
}

#[test]
#[ignore] // requires CHELIS_BEACON_BIN
fn beacon_e2e_registry_dispatch_routes_box_range() {
    let bin = beacon_bin().expect("CHELIS_BEACON_BIN must be set");
    let store = WireDagByteStore::new();
    let extracted = simple_square_goal(-1.0, 2.0);
    store.insert_extracted(&extracted);

    // Build registry with beacon registered
    let mut registry = DischargeRegistry::with_builtin_engines();
    registry.register(Box::new(BeaconShim::new(&bin, store)));

    let discharge = registry.dispatch(&extracted.goal, 10_000);
    assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
}

// ===========================================================================
// WI-10: Verified Greeks via the wired Beacon (gradient / AD rail)
// ===========================================================================

#[test]
#[ignore] // requires CHELIS_BEACON_BIN
fn beacon_e2e_gradient_goal_verified_greek() {
    use chelis_compiler_api::compiler;
    use chelis_compiler_api::schema::GradRequest;
    use chelis_prove::graph_extract::box_range_goal_from_wire_dag;

    let bin = beacon_bin().expect("CHELIS_BEACON_BIN must be set");

    // f(x) = mul(x, x) = x^2, scalar. x in [1,2], df/dx = 2x, so df/dx in [2,4].
    // Use fuse: false to avoid fused_elem ops that Beacon's v1 surface doesn't
    // support (the AD rail hardcodes fuse: true; here we call compiler::grad
    // directly for Beacon compatibility).
    let grad_result = compiler::grad(GradRequest {
        source_kind: SourceKind::Surf,
        source: "x = (x : tensor[f32])\nloss = (mul(x, x) : tensor[f32])\n".to_string(),
        output_name: "loss".to_string(),
        wrt_names: vec!["x".to_string()],
        fuse: false,
    })
    .expect("gradient lowering must succeed");

    // Build the goal from the gradient DAG directly
    let extracted = box_range_goal_from_wire_dag(
        &grad_result.dag,
        &grad_result.grad_nodes_by_name,
        input_box(&[("x", 1.0, 2.0)]),
        // df/dx = 2x, x in [1,2] → df/dx in [2,4]. Use wider range for
        // Beacon's over-approximation headroom.
        output_range("x", -10.0, 10.0),
    )
    .expect("goal construction must succeed");

    // Populate the store with the gradient DAG bytes
    let store = WireDagByteStore::new();
    store.insert_extracted(&extracted);

    // Build registry with beacon
    let mut registry = DischargeRegistry::with_builtin_engines();
    registry.register(Box::new(BeaconShim::new(&bin, store)));

    let discharge = registry.dispatch(&extracted.goal, 10_000);
    assert_eq!(
        discharge.soundness(),
        Soundness::SoundApproximate,
        "gradient goal must be SoundApproximate (verified Greek), got {:?}",
        discharge
    );
    assert!(
        discharge
            .qualifier_set()
            .contains(Qualifier::SoundOverApproximation),
        "gradient discharge must carry SoundOverApproximation"
    );
}
