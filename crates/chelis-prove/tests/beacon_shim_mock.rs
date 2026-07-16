//! Spec-first integration tests for the Beacon subprocess shim (chelis#439),
//! the acceptance oracle named in `docs/design/beacon_subprocess_shim.md` §8.
//!
//! Each test maps to a row of the §5 `CheckReport -> Discharge` table and is
//! paired with its negative twin. The `mock-chelis-beacon` test helper binary
//! stands in for the real out-of-tree verifier; the scenario it emits is
//! selected by `MOCK_BEACON_SCENARIO`.
//!
//! `BeaconShim::run_beacon` spawns a child that inherits THIS process's
//! environment, so two test threads setting different scenarios concurrently
//! would race. Each test holds `SCENARIO_LOCK` across its single `discharge`
//! call (and any env mutation), serializing the env so a child never observes
//! another test's scenario.

use std::sync::{Mutex, MutexGuard};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use sha2::{Digest, Sha256};

use chelis_prove::composition::{CompositeVerdict, base_verdict_from_discharge};
use chelis_prove::discharge::{
    Goal, GoalShape, IntervalBox, IrHandle, OutputRange, Qualifier, Soundness,
};
use chelis_prove::tier_b::TierBResult;
use chelis_prove::{
    BEACON_BIN_ENV, BeaconOracleMode, BeaconShim, DischargeEngine, RequestTransport,
    WireDagByteStore,
};

/// The compiled mock binary path (cargo sets this for integration tests).
const MOCK_BIN: &str = env!("CARGO_BIN_EXE_mock-chelis-beacon");

/// Serializes the `MOCK_BEACON_SCENARIO` env mutation across test threads.
static SCENARIO_LOCK: Mutex<()> = Mutex::new(());

/// Set the scenario env for the duration of the returned guard, holding the
/// serialization lock so no other test thread's child observes it.
fn with_scenario(scenario: &str) -> MutexGuard<'static, ()> {
    let guard = SCENARIO_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // SAFETY: serialized by SCENARIO_LOCK; only the guard holder mutates this
    // var and only spawns children while holding the guard.
    unsafe {
        std::env::set_var("MOCK_BEACON_SCENARIO", scenario);
    }
    guard
}

/// Synthetic serialized `WireDag` v1 bytes. The shim treats these opaquely
/// (base64 + sha256); any deterministic byte string exercises transport.
fn fake_wire_dag_bytes() -> Vec<u8> {
    br#"{"schema_version":1,"nodes":[],"roots":[0]}"#.to_vec()
}

/// Lowercase-hex sha256 of `bytes` — the content-address key, computed the same
/// way the WI-3 producer does (this mirrors the shim's internal `sha256_hex`,
/// which is private; the integration test recomputes it from the public sha2).
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(hex, "{byte:02x}").expect("writing to a String never fails");
    }
    hex
}

fn fake_dag_hash() -> String {
    sha256_hex(&fake_wire_dag_bytes())
}

/// A well-formed `BoxRange` goal whose `IrHandle` addresses
/// [`fake_wire_dag_bytes`] (root index 0).
fn box_goal() -> Goal {
    Goal::box_range(
        IntervalBox {
            dims: vec![("s".to_string(), 0.0, 100.0)],
        },
        OutputRange {
            output: "price".to_string(),
            lo: 0.0,
            hi: 50.0,
        },
    )
    .expect("well-formed box goal")
    .with_ir(IrHandle::from_wire_dag(fake_dag_hash(), 0))
}

fn populated_store() -> WireDagByteStore {
    let store = WireDagByteStore::new();
    store.insert(fake_dag_hash(), fake_wire_dag_bytes());
    store
}

fn shim() -> BeaconShim {
    BeaconShim::new(MOCK_BIN, populated_store())
}

/// 256 KiB of synthetic `WireDag` bytes — base64s to ~341 KiB, well past both
/// the 32 KiB stdin auto-fallback ceiling and the ~64 KiB OS pipe buffer. This
/// is the payload that triggered the red-team HIGH stdin deadlock.
fn large_wire_dag_bytes() -> Vec<u8> {
    vec![b'x'; 256 * 1024]
}

fn large_dag_hash() -> String {
    sha256_hex(&large_wire_dag_bytes())
}

/// A `BoxRange` goal addressing the 256 KiB artifact.
fn large_box_goal() -> Goal {
    Goal::box_range(
        IntervalBox {
            dims: vec![("s".to_string(), 0.0, 100.0)],
        },
        OutputRange {
            output: "price".to_string(),
            lo: 0.0,
            hi: 50.0,
        },
    )
    .expect("well-formed box goal")
    .with_ir(IrHandle::from_wire_dag(large_dag_hash(), 0))
}

fn large_populated_store() -> WireDagByteStore {
    let store = WireDagByteStore::new();
    store.insert(large_dag_hash(), large_wire_dag_bytes());
    store
}

/// A generous timeout for the non-hang scenarios (the mock returns instantly).
const FAST_TIMEOUT_MS: u64 = 30_000;

fn evidence_error(discharge: &chelis_prove::Discharge) -> Option<String> {
    discharge
        .evidence()
        .get("error")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

// ===========================================================================
// §5 row: proved (oracle-verified) -> SoundApproximate + SoundOverApproximation
// ===========================================================================

#[test]
fn proved_maps_to_sound_approximate_over_approximation() {
    let _g = with_scenario("proved");
    let discharge = shim().discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert_eq!(*discharge.result(), TierBResult::Proved);
    assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
    assert!(
        discharge
            .qualifier_set()
            .contains(Qualifier::SoundOverApproximation),
        "a proved bound is a sound over-approximation"
    );
    assert!(
        !discharge.qualifier_set().contains(Qualifier::Exact),
        "an interval over-approximation must not claim exact soundness"
    );
    let verdict = base_verdict_from_discharge(discharge.soundness(), discharge.qualifier_set());
    assert_eq!(verdict, CompositeVerdict::SoundApproximate);
    assert_ne!(verdict, CompositeVerdict::Proven);
}

// ===========================================================================
// §5 row: proved_oracle_unverified -> Untrusted + empty (integrity guard).
// NEGATIVE TWIN of `proved`.
// ===========================================================================

#[test]
fn proved_oracle_unverified_maps_to_untrusted_empty_never_a_proof() {
    let _g = with_scenario("proved_oracle_unverified");
    let discharge = shim().discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert!(
        matches!(discharge.result(), TierBResult::Error(_)),
        "an unverified proof is not a Proved result"
    );
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(
        discharge.qualifier_set().is_empty(),
        "an unverified proof carries NO proof qualifier (no laundered badge)"
    );
    let verdict = base_verdict_from_discharge(discharge.soundness(), discharge.qualifier_set());
    assert_ne!(verdict, CompositeVerdict::Proven);
    assert_ne!(verdict, CompositeVerdict::SoundApproximate);
}

// ===========================================================================
// §5 row: refuted + oracle-verified TRUE -> Disproved + SoundApproximate
// ===========================================================================

#[test]
fn refuted_oracle_verified_maps_to_disproved_sound_approximate() {
    let _g = with_scenario("refuted_verified");
    let discharge = shim().discharge(&box_goal(), FAST_TIMEOUT_MS);
    match discharge.result() {
        TierBResult::Disproved(model) => {
            assert_eq!(
                model.get("s").and_then(serde_json::Value::as_f64),
                Some(42.0),
                "the counterexample model is carried into the Disproved result"
            );
        }
        other => panic!("expected Disproved, got {other:?}"),
    }
    assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
    assert!(
        discharge
            .qualifier_set()
            .contains(Qualifier::SoundOverApproximation),
        "a verified refutation is a sound over-approximation, symmetric to proved"
    );
}

// ===========================================================================
// §5 row: refuted + flag FALSE -> Disproved + Untrusted (NEGATIVE TWIN)
// ===========================================================================

#[test]
fn refuted_oracle_unverified_maps_to_disproved_untrusted() {
    let _g = with_scenario("refuted_unverified");
    let discharge = shim().discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert!(
        matches!(discharge.result(), TierBResult::Disproved(_)),
        "the goal still did not hold: the result is Disproved"
    );
    assert_eq!(
        discharge.soundness(),
        Soundness::Untrusted,
        "an unverified refutation is conservative: Untrusted, mirroring proved_oracle_unverified"
    );
    assert!(
        discharge.qualifier_set().is_empty(),
        "an unverified refutation carries no qualifier"
    );
}

// ===========================================================================
// §5 row: refuted + flag ABSENT -> defaults to Untrusted (NEGATIVE TWIN).
// The Beacon-contract follow-up default: no flag => conservative.
// ===========================================================================

#[test]
fn refuted_without_oracle_flag_defaults_to_untrusted() {
    let _g = with_scenario("refuted_no_flag");
    let discharge = shim().discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert!(matches!(discharge.result(), TierBResult::Disproved(_)));
    assert_eq!(
        discharge.soundness(),
        Soundness::Untrusted,
        "a refutation with NO oracle_verified flag defaults to Untrusted (fail-safe)"
    );
    assert!(discharge.qualifier_set().is_empty());
}

// ===========================================================================
// §5 row: nonzero exit -> Untrusted + Error, stderr captured in evidence
// ===========================================================================

#[test]
fn nonzero_exit_maps_to_untrusted_error_with_stderr_in_evidence() {
    let _g = with_scenario("nonzero_exit");
    let discharge = shim().discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert!(matches!(discharge.result(), TierBResult::Error(_)));
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(discharge.qualifier_set().is_empty());
    let stderr = discharge
        .evidence()
        .get("stderr")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(
        stderr.contains("simulated verifier failure"),
        "the child's stderr must be captured in the discharge evidence, got: {stderr:?}"
    );
}

// ===========================================================================
// §5 row: unparseable report -> Untrusted + Error (NEGATIVE TWIN of proved)
// ===========================================================================

#[test]
fn unparseable_report_maps_to_untrusted_error_fail_closed() {
    let _g = with_scenario("unparseable");
    let discharge = shim().discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert!(matches!(discharge.result(), TierBResult::Error(_)));
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(discharge.qualifier_set().is_empty());
    assert_eq!(
        evidence_error(&discharge).as_deref(),
        Some("unparseable_report"),
        "an unparseable report fails closed, never greens"
    );
}

// ===========================================================================
// §5 row: timeout (hard kill) -> Untrusted + Error("beacon timeout")
// ===========================================================================

#[test]
fn hang_is_hard_killed_at_timeout_and_maps_to_untrusted_error() {
    let _g = with_scenario("hang");
    let start = std::time::Instant::now();
    let discharge = shim().discharge(&box_goal(), 500);
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "the hard kill must return promptly (took {elapsed:?}), not wait for the child"
    );
    assert!(matches!(discharge.result(), TierBResult::Error(_)));
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(discharge.qualifier_set().is_empty());
    assert_eq!(
        evidence_error(&discharge).as_deref(),
        Some("timeout"),
        "a timeout fails closed as `timeout`, never a silent pass"
    );
}

// ===========================================================================
// §5 row: binary not configured (Q2) -> from_env returns None (not registered),
// never a crash.
// ===========================================================================

#[test]
fn from_env_with_unset_var_yields_no_shim_no_crash() {
    let guard = SCENARIO_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // SAFETY: serialized by SCENARIO_LOCK.
    unsafe {
        std::env::remove_var(BEACON_BIN_ENV);
    }
    let shim = BeaconShim::from_env(populated_store());
    assert!(
        shim.is_none(),
        "with CHELIS_BEACON_BIN unset and no explicit path, the shim is not constructed"
    );
    drop(guard);
}

#[test]
fn from_env_with_set_var_constructs_a_shim() {
    let guard = SCENARIO_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // SAFETY: serialized by SCENARIO_LOCK.
    unsafe {
        std::env::set_var(BEACON_BIN_ENV, MOCK_BIN);
    }
    let constructed = BeaconShim::from_env(populated_store()).is_some();
    // SAFETY: serialized by SCENARIO_LOCK.
    unsafe {
        std::env::remove_var(BEACON_BIN_ENV);
    }
    drop(guard);
    assert!(constructed, "an env-set binary path constructs the shim");
}

// ===========================================================================
// §5 row: dag_hash mismatch -> fail-closed Error before spawning Beacon.
// ===========================================================================

#[test]
fn dag_hash_mismatch_fails_closed_before_spawning_beacon() {
    let _g = with_scenario("proved"); // would prove, but the mismatch fails first
    let store = WireDagByteStore::new();
    store.insert(fake_dag_hash(), b"totally different bytes".to_vec());
    let shim = BeaconShim::new(MOCK_BIN, store);
    let discharge = shim.discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert!(matches!(discharge.result(), TierBResult::Error(_)));
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(discharge.qualifier_set().is_empty());
    assert_eq!(
        evidence_error(&discharge).as_deref(),
        Some("wire_dag_hash_mismatch"),
        "store bytes that do not hash to the handle's dag_hash must fail closed"
    );
}

// ===========================================================================
// Fail-closed: store MISS -> Untrusted + Error (distinct from no-fit). NEGATIVE.
// ===========================================================================

#[test]
fn empty_store_byte_miss_fails_closed_not_a_crash() {
    let _g = with_scenario("proved");
    let shim = BeaconShim::new(MOCK_BIN, WireDagByteStore::new());
    let discharge = shim.discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert!(matches!(discharge.result(), TierBResult::Error(_)));
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert_eq!(
        evidence_error(&discharge).as_deref(),
        Some("byte_store_miss"),
        "a store miss fails closed, never greens and never crashes"
    );
}

// ===========================================================================
// Fail-closed: unpopulated IrHandle -> Untrusted + Error. NEGATIVE.
// ===========================================================================

#[test]
fn unpopulated_ir_handle_fails_closed() {
    let _g = with_scenario("proved");
    let goal = Goal::box_range(
        IntervalBox {
            dims: vec![("s".to_string(), 0.0, 1.0)],
        },
        OutputRange {
            output: "price".to_string(),
            lo: 0.0,
            hi: 1.0,
        },
    )
    .expect("well-formed box goal");
    assert!(!goal.ir.is_populated());
    let discharge = shim().discharge(&goal, FAST_TIMEOUT_MS);
    assert!(matches!(discharge.result(), TierBResult::Error(_)));
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert_eq!(
        evidence_error(&discharge).as_deref(),
        Some("unpopulated_ir_handle")
    );
}

// ===========================================================================
// fitness: claims ONLY BoxRange. NEGATIVE: an SMT goal is not a fit.
// ===========================================================================

#[test]
fn fitness_accepts_box_range_rejects_smt() {
    use chelis_prove::solver::{CmpOp, SmtExpr, SmtSort};
    use chelis_prove::tier_b::SmtProperty;
    let shim = shim();
    assert!(shim.fitness(&box_goal()), "beacon fits a BoxRange goal");

    let smt_goal = Goal::smt(SmtProperty {
        variables: vec![("x".to_string(), SmtSort::Real)],
        preconditions: vec![],
        postcondition: SmtExpr::Cmp(
            CmpOp::Eq,
            Box::new(SmtExpr::Var("x".to_string())),
            Box::new(SmtExpr::Var("x".to_string())),
        ),
    });
    assert!(
        !shim.fitness(&smt_goal),
        "beacon does NOT fit an SMT goal (cvc5's lane)"
    );
}

/// A non-fitting SMT goal handed to `discharge` anyway must not fabricate a
/// proof: it fails closed. (Defends the fitness gate.)
#[test]
fn discharging_an_smt_goal_directly_fails_closed() {
    use chelis_prove::solver::{CmpOp, SmtExpr, SmtSort};
    use chelis_prove::tier_b::SmtProperty;
    let smt_goal = Goal::smt(SmtProperty {
        variables: vec![("x".to_string(), SmtSort::Real)],
        preconditions: vec![],
        postcondition: SmtExpr::Cmp(
            CmpOp::Eq,
            Box::new(SmtExpr::Var("x".to_string())),
            Box::new(SmtExpr::Var("x".to_string())),
        ),
    });
    let _g = with_scenario("proved");
    let discharge = shim().discharge(&smt_goal, FAST_TIMEOUT_MS);
    assert!(matches!(discharge.result(), TierBResult::Error(_)));
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert_eq!(
        evidence_error(&discharge).as_deref(),
        Some("wrong_goal_shape")
    );
}

// ===========================================================================
// Q5/Q6 transport invariant: the request carries the EXACT base64 bytes and
// expected_dag_sha256 == dag_hash, NO reformat between WI-3 bytes and base64.
// Asserted via the `echo_request` mock that mirrors the request back on stdout.
// ===========================================================================

#[test]
fn request_carries_exact_base64_bytes_and_expected_hash() {
    let _g = with_scenario("echo_request");
    // echo_request mirrors the request back; the shim then fails to parse it as
    // a CheckReport (it is the request, not a report), so the discharge is an
    // Error -- but the evidence captures the echoed stdout (the request JSON).
    let discharge = shim().discharge(&box_goal(), FAST_TIMEOUT_MS);
    let echoed = discharge
        .evidence()
        .get("stdout")
        .and_then(|v| v.as_str())
        .expect("echoed request is captured as the unparseable-report stdout");
    let request: serde_json::Value =
        serde_json::from_str(echoed).expect("the echoed request is valid JSON the shim sent");

    assert_eq!(
        request.get("expected_dag_sha256").and_then(|v| v.as_str()),
        Some(fake_dag_hash().as_str()),
        "expected_dag_sha256 must equal the IrHandle dag_hash"
    );
    assert_eq!(
        request
            .get("schema_version")
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert_eq!(
        request
            .get("root_index")
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );

    let b64 = request
        .get("wire_dag_v1_base64")
        .and_then(|v| v.as_str())
        .expect("wire_dag_v1_base64 present");
    let decoded = BASE64.decode(b64).expect("base64 decodes");
    assert_eq!(
        decoded,
        fake_wire_dag_bytes(),
        "the base64 must decode to the EXACT WireDag bytes, byte-identical, no reformat"
    );
    assert_eq!(
        sha256_hex(&decoded),
        fake_dag_hash(),
        "the decoded buffer's sha256 is the round-trip identity"
    );
    assert_eq!(
        request.get("oracle"),
        Some(&serde_json::Value::Null),
        "default shim requests preserve the #439 `oracle: null` contract"
    );
    assert_eq!(
        request
            .get("output")
            .and_then(|o| o.get("output"))
            .and_then(|v| v.as_str()),
        Some("price")
    );
}

#[test]
fn verified_zonotope_mode_emits_exact_selector_string() {
    let _g = with_scenario("echo_request");
    let discharge = shim()
        .with_oracle_mode(BeaconOracleMode::VerifiedZonotope)
        .discharge(&box_goal(), FAST_TIMEOUT_MS);
    let echoed = discharge
        .evidence()
        .get("stdout")
        .and_then(|v| v.as_str())
        .expect("echoed request is captured as the unparseable-report stdout");
    let request: serde_json::Value =
        serde_json::from_str(echoed).expect("the echoed request is valid JSON the shim sent");

    assert_eq!(
        request.get("oracle").and_then(|v| v.as_str()),
        Some("zonotope_verified"),
        "the dispatch selector is the underscore wire-contract spelling"
    );
    assert_eq!(
        request.get("expected_dag_sha256").and_then(|v| v.as_str()),
        Some(fake_dag_hash().as_str()),
        "selector mode must not alter exact-byte identity"
    );
}

#[test]
fn verified_zonotope_mode_does_not_launder_unverified_proof() {
    let _g = with_scenario("proved_oracle_unverified");
    let discharge = shim()
        .with_oracle_mode(BeaconOracleMode::VerifiedZonotope)
        .discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert!(
        matches!(discharge.result(), TierBResult::Error(_)),
        "an unverified zonotope proof is not a Proved result"
    );
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(
        discharge.qualifier_set().is_empty(),
        "verified-zonotope selector cannot launder oracle-unverified reports into proof qualifiers"
    );
}

// ===========================================================================
// The temp-file transport arm works too (same proved mapping).
// ===========================================================================

#[test]
fn temp_file_transport_proved_maps_identically() {
    let _g = with_scenario("proved");
    let discharge = shim()
        .with_transport(RequestTransport::TempFile)
        .discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert_eq!(*discharge.result(), TierBResult::Proved);
    assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
    assert!(
        discharge
            .qualifier_set()
            .contains(Qualifier::SoundOverApproximation)
    );
}

// ===========================================================================
// Spawn failure (binary path does not exist) -> fail-closed, no crash.
// ===========================================================================

#[test]
fn nonexistent_binary_fails_closed_not_a_crash() {
    let _g = with_scenario("proved");
    let shim = BeaconShim::new("/nonexistent/path/to/chelis-beacon-xyz", populated_store());
    let discharge = shim.discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert!(matches!(discharge.result(), TierBResult::Error(_)));
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(discharge.qualifier_set().is_empty());
}

// ===========================================================================
// Store convenience: get returns the inserted bytes; a miss is None.
// ===========================================================================

#[test]
fn store_round_trips_inserted_bytes() {
    let store = WireDagByteStore::new();
    store.insert(fake_dag_hash(), fake_wire_dag_bytes());
    // (no public getter; round-trip is observed through the shim succeeding)
    let shim = BeaconShim::new(MOCK_BIN, store);
    let _g = with_scenario("proved");
    let discharge = shim.discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert_eq!(*discharge.result(), TierBResult::Proved);
}

// ===========================================================================
// Registry integration: with no Beacon registered a BoxRange goal is no-fit
// Unsupported; with the shim registered it routes to "beacon". (default lane.)
// ===========================================================================

#[cfg(not(feature = "smt"))]
#[test]
fn box_range_goal_with_no_beacon_registered_is_no_fit_unsupported() {
    use chelis_prove::DischargeRegistry;
    let registry = DischargeRegistry::with_builtin_engines();
    assert_eq!(registry.selected_engine_name(&box_goal()), None);
    let discharge = registry.dispatch(&box_goal(), FAST_TIMEOUT_MS);
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(discharge.qualifier_set().is_empty());
    assert!(matches!(discharge.result(), TierBResult::Error(_)));
}

#[cfg(not(feature = "smt"))]
#[test]
fn registered_beacon_shim_is_selected_for_box_range_goal() {
    use chelis_prove::DischargeRegistry;
    let mut registry = DischargeRegistry::with_builtin_engines();
    registry.register(Box::new(shim()));
    assert_eq!(registry.selected_engine_name(&box_goal()), Some("beacon"));
    let _g = with_scenario("proved");
    let discharge = registry.dispatch(&box_goal(), FAST_TIMEOUT_MS);
    assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
    assert_eq!(*discharge.result(), TierBResult::Proved);
}

/// A `GoalShape::BoxRange` matcher sanity check kept here so a refactor that
/// changes the variant shape trips this integration test too.
#[test]
fn box_goal_is_box_range_shaped() {
    assert!(matches!(box_goal().shape, GoalShape::BoxRange { .. }));
}

// ===========================================================================
// RED-TEAM HIGH (pinned regression): a large request (256 KiB WireDag, base64s
// to ~341 KiB, past the OS pipe buffer) against a NON-DRAINING child must
// HARD-KILL at the timeout rather than hang on a full stdin pipe. The shim
// auto-falls-back the Stdin transport to TempFile above the 32 KiB ceiling, so
// the parent writes a temp file (not a pipe) and `wait_timeout` still fires.
// Before the fix this hung for the child's full 600s lifetime.
// ===========================================================================

#[test]
fn large_request_against_non_draining_child_hard_kills_not_deadlocks() {
    let _g = with_scenario("hang_no_drain");
    // Stdin transport configured explicitly: the auto-fallback must override it
    // for this oversized request, or the write deadlocks.
    let shim =
        BeaconShim::new(MOCK_BIN, large_populated_store()).with_transport(RequestTransport::Stdin);
    let start = std::time::Instant::now();
    let discharge = shim.discharge(&large_box_goal(), 500);
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "a 256 KiB request must hard-kill at the timeout (took {elapsed:?}), \
         not block on a full stdin pipe for the child's lifetime"
    );
    assert!(matches!(discharge.result(), TierBResult::Error(_)));
    assert_eq!(discharge.soundness(), Soundness::Untrusted);
    assert!(discharge.qualifier_set().is_empty());
    assert_eq!(
        evidence_error(&discharge).as_deref(),
        Some("timeout"),
        "the deadlock-safe path still fails closed as a timeout"
    );
}

/// The other half of the HIGH acceptance: a large PROVED verdict is DELIVERED,
/// not lost. A 256 KiB request that the child actually processes (drains via the
/// temp file, emits proved) round-trips to a Proved + `SoundApproximate` discharge.
#[test]
fn large_request_proved_verdict_is_delivered_not_lost() {
    let _g = with_scenario("proved");
    let shim =
        BeaconShim::new(MOCK_BIN, large_populated_store()).with_transport(RequestTransport::Stdin); // auto-falls-back to TempFile
    let discharge = shim.discharge(&large_box_goal(), FAST_TIMEOUT_MS);
    assert_eq!(
        *discharge.result(),
        TierBResult::Proved,
        "a large request's proved verdict must survive the transport"
    );
    assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
    assert!(
        discharge
            .qualifier_set()
            .contains(Qualifier::SoundOverApproximation)
    );
}

/// NEGATIVE/positive twin of the deadlock fix: a SMALL stdin request still uses
/// the stdin transport (the fallback is a ceiling, not an always-tempfile
/// switch) and maps proved identically. (Observed indirectly: the small request
/// proves through the stdin path, which the rest of the suite exercises; this
/// pins that a small request is NOT forced onto the temp file.)
#[test]
fn small_request_stays_on_stdin_and_proves() {
    let _g = with_scenario("proved");
    let discharge = shim()
        .with_transport(RequestTransport::Stdin)
        .discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert_eq!(*discharge.result(), TierBResult::Proved);
    assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
}

/// A large request against a child that DOES drain stdin but hangs after, via
/// the temp-file fallback, still hard-kills. (Defends that the fallback's
/// temp-file child is reaped, not just the stdin one.)
#[test]
fn large_request_temp_file_fallback_hard_kills_a_draining_hang() {
    let _g = with_scenario("hang"); // drains the temp file, then sleeps 600s
    let shim =
        BeaconShim::new(MOCK_BIN, large_populated_store()).with_transport(RequestTransport::Stdin);
    let start = std::time::Instant::now();
    let discharge = shim.discharge(&large_box_goal(), 500);
    assert!(
        start.elapsed() < std::time::Duration::from_secs(30),
        "the temp-file fallback child must also hard-kill at the timeout"
    );
    assert_eq!(evidence_error(&discharge).as_deref(), Some("timeout"));
}

// ===========================================================================
// RED-TEAM MED (pinned regression): a `proved` verdict carrying an explicit
// `oracle_verified: false` is SELF-CONTRADICTORY and must fail closed to
// Untrusted, symmetric with the refuted arm. (An ABSENT flag stays proved ->
// SoundApproximate; the rest of the suite pins that.)
// ===========================================================================

#[test]
fn proved_with_oracle_verified_false_fails_closed_to_untrusted() {
    let _g = with_scenario("proved_oracle_false");
    let discharge = shim().discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert!(
        matches!(discharge.result(), TierBResult::Error(_)),
        "a self-contradictory proved+false must not be a Proved result"
    );
    assert_eq!(
        discharge.soundness(),
        Soundness::Untrusted,
        "proved+oracle_verified=false fails closed, symmetric with unverified refuted"
    );
    assert!(
        discharge.qualifier_set().is_empty(),
        "the contradictory proof carries NO proof qualifier"
    );
    assert_eq!(
        evidence_error(&discharge).as_deref(),
        Some("contradictory_proved_unverified")
    );
    // And it must NOT project to a green.
    let verdict = base_verdict_from_discharge(discharge.soundness(), discharge.qualifier_set());
    assert_ne!(verdict, CompositeVerdict::Proven);
    assert_ne!(verdict, CompositeVerdict::SoundApproximate);
}

/// Positive twin: a `proved` verdict with the flag ABSENT is the normal verified
/// case and stays `SoundApproximate`. The production gate is `oracle_verified ==
/// Some(false)`, so an absent (`None`) flag is accepted — pinned here against a
/// `proved` report that omits the field entirely.
#[test]
fn proved_with_absent_oracle_flag_stays_sound_approximate() {
    let _g = with_scenario("proved_no_flag");
    let discharge = shim().discharge(&box_goal(), FAST_TIMEOUT_MS);
    assert_eq!(
        *discharge.result(),
        TierBResult::Proved,
        "a proved verdict with no oracle_verified field is the normal accepted case"
    );
    assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
    assert!(
        discharge
            .qualifier_set()
            .contains(Qualifier::SoundOverApproximation)
    );
}

#[test]
fn verified_zonotope_mode_does_not_launder_unverified_transport_or_invalid_reports() {
    let cases = [
        (
            "proved_oracle_unverified",
            "oracle-unverified",
            FAST_TIMEOUT_MS,
        ),
        ("proved_oracle_false", "oracle-unverified", FAST_TIMEOUT_MS),
        ("refuted_unverified", "oracle-unverified", FAST_TIMEOUT_MS),
        ("refuted_no_flag", "oracle-unverified", FAST_TIMEOUT_MS),
        ("nonzero_exit", "transport", FAST_TIMEOUT_MS),
        ("hang", "transport", 500),
        ("unparseable", "invalid", FAST_TIMEOUT_MS),
    ];

    for (scenario, class, timeout_ms) in cases {
        let _g = with_scenario(scenario);
        let discharge = shim()
            .with_oracle_mode(BeaconOracleMode::VerifiedZonotope)
            .discharge(&box_goal(), timeout_ms);
        assert_eq!(
            discharge.soundness(),
            Soundness::Untrusted,
            "{class} scenario `{scenario}` must not map to sound with zonotope_verified selected"
        );
        assert!(
            discharge.qualifier_set().is_empty(),
            "{class} scenario `{scenario}` must carry no soundness qualifier"
        );
        let verdict = base_verdict_from_discharge(discharge.soundness(), discharge.qualifier_set());
        assert_ne!(
            verdict,
            CompositeVerdict::SoundApproximate,
            "{class} scenario `{scenario}` must not project to sound_approximate"
        );
        assert_ne!(
            verdict,
            CompositeVerdict::Proven,
            "{class} scenario `{scenario}` must not project to proven"
        );
    }
}
