//! End-to-end integration test: chelis → BeaconShim → real chelis-beacon binary.
//!
//! Requires CHELIS_BEACON_BIN to point at the real beacon binary.
//! Run with:
//!
//! ```text
//! CHELIS_BEACON_BIN=/path/to/chelis-beacon \
//!   cargo test -p chelis-prove --test beacon_e2e -- --ignored --nocapture
//! ```

use std::process::Command;

use chelis_compiler_api::schema::SourceKind;
use chelis_prove::discharge::{IntervalBox, OutputRange, Qualifier, Soundness};
use chelis_prove::engine_registry::DischargeRegistry;
use chelis_prove::graph_extract::{box_range_goal_from_source, box_range_goal_from_source_entry};
use chelis_prove::tier_b::TierBResult;
use chelis_prove::{BeaconOracleMode, BeaconShim, DischargeEngine, WireDagByteStore};
use sha2::{Digest, Sha256};

const BS_VEC_SOURCE: &str = include_str!("fixtures/bs_vec.ch");

fn beacon_bin() -> Option<String> {
    std::env::var("CHELIS_BEACON_BIN").ok()
}

fn require_arb_enabled_beacon() -> String {
    let bin = beacon_bin().expect("CHELIS_BEACON_BIN must be set");
    let output = Command::new(&bin)
        .arg("protocol")
        .arg("--json")
        .output()
        .expect("must be able to run chelis-beacon protocol --json");
    assert!(
        output.status.success(),
        "chelis-beacon protocol --json failed: status={:?}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let protocol: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("protocol output must be JSON");
    assert_eq!(
        protocol["capabilities"]["dispatch_zonotope_verified"],
        serde_json::Value::Bool(true),
        "live BeaconShim gate requires an arb-oracle Beacon binary: {protocol:#}"
    );
    assert_eq!(
        protocol["coordination"]["arb_oracle_feature_enabled"],
        serde_json::Value::Bool(true),
        "verified dispatch must not silently degrade to a non-Arb binary: {protocol:#}"
    );
    bin
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(hex, "{byte:02x}").expect("writing to a String never fails");
    }
    hex
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

fn bs_atm_singleton_input_box() -> IntervalBox {
    input_box(&[
        ("s", 100.0, 100.0),
        ("k", 100.0, 100.0),
        ("r", 0.05, 0.05),
        ("sigma", 0.2, 0.2),
        ("t", 1.0, 1.0),
        ("half", 0.5, 0.5),
        (
            "inv_sqrt2",
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ),
        ("a1", 0.254829592, 0.254829592),
        ("a2", -0.284496736, -0.284496736),
        ("a3", 1.421413741, 1.421413741),
        ("a4", -1.453152027, -1.453152027),
        ("a5", 1.061405429, 1.061405429),
        ("p", 0.3275911, 0.3275911),
        (
            "twosqrtpi",
            std::f64::consts::FRAC_2_SQRT_PI,
            std::f64::consts::FRAC_2_SQRT_PI,
        ),
        ("small_thresh", 0.00001, 0.00001),
    ])
}

/// Build a simple forward DAG from Chelis source, then extract a BoxRange goal.
/// The resulting `wire_dag_bytes` are produced by Chelis WI-3, not by Beacon.
fn simple_square_goal(output_lo: f64, output_hi: f64) -> chelis_prove::ExtractedGoal {
    let source = "x = (x : tensor[f32])\nout = (mul(x, x) : tensor[f32])\n";
    box_range_goal_from_source(
        source,
        SourceKind::Surf,
        input_box(&[("x", 0.0, 1.0)]),
        output_range("out", output_lo, output_hi),
    )
    .expect("extraction must succeed for a trivial square program")
}

fn bs_call_vec_atm_goal(output_lo: f64, output_hi: f64) -> chelis_prove::ExtractedGoal {
    box_range_goal_from_source_entry(
        BS_VEC_SOURCE,
        SourceKind::Surf,
        "bs_call_vec",
        bs_atm_singleton_input_box(),
        output_range("bs_call_vec", output_lo, output_hi),
    )
    .expect("bs_call_vec must lower to a Chelis-produced WireDag root")
}

fn verified_zonotope_shim(bin: &str, store: WireDagByteStore) -> BeaconShim {
    BeaconShim::new(bin, store).with_oracle_mode(BeaconOracleMode::VerifiedZonotope)
}

#[test]
#[ignore] // requires CHELIS_BEACON_BIN
fn beacon_e2e_live_shim_to_real_binary_sound_green() {
    let bin = require_arb_enabled_beacon();
    let store = WireDagByteStore::new();
    // x in [0,1], f(x) = x^2 in [0,1]; use wider claimed range [-1,2] to
    // accommodate Beacon's interval over-approximation.
    let extracted = simple_square_goal(-1.0, 2.0);
    assert_eq!(
        extracted.dag_hash,
        sha256_hex(&extracted.wire_dag_bytes),
        "WI-3 dag_hash must address the exact bytes transported to Beacon"
    );
    store.insert_extracted(&extracted);

    let shim = verified_zonotope_shim(&bin, store);
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
    let bin = require_arb_enabled_beacon();
    let store = WireDagByteStore::new();
    // x in [0,1], f(x) = x^2 is in [0,1], NOT [10,20] — Beacon must reject
    let extracted = simple_square_goal(10.0, 20.0);
    store.insert_extracted(&extracted);

    let shim = verified_zonotope_shim(&bin, store);
    let discharge = shim.discharge(&extracted.goal, 10_000);
    // Must NOT be a Proved verdict — Beacon should refute or fail, never prove
    // a wrong range. A verified refutation is still SoundApproximate (it's a
    // sound over-approximation that disproves containment), so check the result.
    assert!(
        !matches!(
            discharge.result(),
            chelis_prove::tier_b::TierBResult::Proved
        ),
        "a wrong-range goal must NOT be Proved by Beacon, got {:?}",
        discharge
    );
}

#[test]
#[ignore] // requires CHELIS_BEACON_BIN
fn beacon_e2e_corrupt_store_hash_mismatch_fails_before_laundering() {
    let bin = require_arb_enabled_beacon();
    let extracted = simple_square_goal(-1.0, 2.0);
    let mut corrupt_bytes = extracted.wire_dag_bytes.clone();
    corrupt_bytes.push(b'\n');

    let store = WireDagByteStore::new();
    store.insert(extracted.dag_hash.clone(), corrupt_bytes);

    let discharge = verified_zonotope_shim(&bin, store).discharge(&extracted.goal, 10_000);
    assert!(
        matches!(discharge.result(), TierBResult::Error(_)),
        "hash mismatch must fail closed as an error: {discharge:?}"
    );
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(
        discharge.qualifier_set().is_empty(),
        "hash mismatch must not carry a proof qualifier: {discharge:?}"
    );
    assert!(
        discharge
            .evidence()
            .get("error")
            .and_then(|v| v.as_str())
            .is_some_and(|error| error.contains("wire_dag_hash_mismatch")),
        "hash mismatch evidence should identify the exact-byte seam failure: {discharge:?}"
    );
}

#[test]
#[ignore] // requires CHELIS_BEACON_BIN
fn beacon_e2e_bs_call_vec_live_shim_round_trip_uses_chelis_wi3_bytes() {
    let bin = require_arb_enabled_beacon();
    let extracted = bs_call_vec_atm_goal(10.0, 11.0);
    assert_eq!(
        extracted.dag_hash,
        sha256_hex(&extracted.wire_dag_bytes),
        "bs_call_vec goal must be content-addressed by exact Chelis WI-3 bytes"
    );

    let store = WireDagByteStore::new();
    store.insert_extracted(&extracted);

    let discharge = BeaconShim::new(&bin, store).discharge(&extracted.goal, 10_000);
    assert_eq!(*discharge.result(), TierBResult::Proved);
    assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
    assert!(
        discharge
            .qualifier_set()
            .contains(Qualifier::SoundOverApproximation),
        "trusted BS proof must carry SoundOverApproximation"
    );

    let beacon_evidence = discharge
        .evidence()
        .get("beacon_evidence")
        .expect("successful live shim mapping preserves Beacon-side evidence");
    assert_eq!(
        beacon_evidence["expected_dag_sha256"].as_str(),
        Some(extracted.dag_hash.as_str())
    );
    assert_eq!(
        beacon_evidence["dag_sha256"].as_str(),
        Some(extracted.dag_hash.as_str())
    );
    assert_eq!(
        beacon_evidence["dispatch_hash_identity"].as_str(),
        Some("exact_serialized_bytes")
    );
    assert_eq!(
        beacon_evidence["dispatch_root_index"].as_u64(),
        extracted.goal.ir.root_index()
    );
    assert_eq!(beacon_evidence["output_name"].as_str(), Some("bs_call_vec"));
    assert_eq!(beacon_evidence["schema_version"].as_u64(), Some(2));
}

#[test]
#[ignore] // requires CHELIS_BEACON_BIN
fn beacon_e2e_registry_dispatch_routes_box_range() {
    let bin = require_arb_enabled_beacon();
    let store = WireDagByteStore::new();
    let extracted = simple_square_goal(-1.0, 2.0);
    store.insert_extracted(&extracted);

    // Build registry with beacon registered
    let mut registry = DischargeRegistry::with_builtin_engines();
    registry.register(Box::new(verified_zonotope_shim(&bin, store)));

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

    let bin = require_arb_enabled_beacon();

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
    registry.register(Box::new(verified_zonotope_shim(&bin, store)));

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
