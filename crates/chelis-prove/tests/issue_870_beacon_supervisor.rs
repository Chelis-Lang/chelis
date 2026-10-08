//! LU1: a configured Beacon subprocess is supervised on the contract route.

use std::path::Path;
use std::sync::Mutex;

use chelis_prove::BeaconContractProver;

const MOCK_BIN: &str = env!("CARGO_BIN_EXE_mock-chelis-beacon");
static SCENARIO_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn beacon_callers_cannot_use_direct_wait_timeout_or_child_lifecycle_calls() {
    fn scan(path: &Path) {
        for entry in std::fs::read_dir(path).expect("source directory") {
            let path = entry.expect("source entry").path();
            if path.is_dir() {
                scan(&path);
            } else if path.extension().is_some_and(|extension| extension == "rs")
                && path
                    .file_name()
                    .is_none_or(|name| name != "beacon_supervisor.rs")
            {
                let source = std::fs::read_to_string(&path).expect("Rust source");
                let forbidden = ["wait", "_timeout"].concat();
                assert!(
                    !source.contains(&forbidden),
                    "direct wait-timeout use in {}",
                    path.display()
                );
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    scan(&root.join("src"));
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("crate manifest");
    assert!(
        !manifest.contains("wait-timeout"),
        "Beacon crate depends on wait-timeout"
    );

    for caller in ["beacon_contract_prover.rs", "beacon_shim.rs"] {
        let source = std::fs::read_to_string(root.join("src").join(caller)).expect("Beacon caller");
        for forbidden in ["Command::new", ".spawn(", ".try_wait(", ".wait(", ".kill("] {
            assert!(
                !source.contains(forbidden),
                "{caller} bypasses supervisor via {forbidden}"
            );
        }
    }
}

#[test]
fn contract_completion_returns_certified_evidence() {
    let _guard = SCENARIO_LOCK.lock().unwrap();
    unsafe { std::env::set_var("MOCK_BEACON_SCENARIO", "contract_proved") };
    let discharge = BeaconContractProver::new(MOCK_BIN)
        .prove_contract("std.normal_cdf.reflection")
        .expect("supervisor completed")
        .expect("contract proved");
    assert_eq!(discharge.evidence["status"], "proved");
    unsafe { std::env::remove_var("MOCK_BEACON_SCENARIO") };
}

#[test]
fn missing_binary_is_a_branded_degradation() {
    let directory = tempfile::tempdir().unwrap();
    let err = BeaconContractProver::new(directory.path().join("missing-beacon"))
        .prove_contract("std.normal_cdf.reflection")
        .expect_err("configured Beacon could not spawn");
    assert!(err.to_string().starts_with("unsupported: "), "{err}");
    assert!(err.to_string().contains("chelis#730"), "{err}");
}

#[test]
fn crashed_child_is_a_branded_degradation() {
    let _guard = SCENARIO_LOCK.lock().unwrap();
    unsafe { std::env::set_var("MOCK_BEACON_SCENARIO", "crash") };
    let err = BeaconContractProver::new(MOCK_BIN)
        .prove_contract("std.normal_cdf.reflection")
        .expect_err("configured Beacon crashed");
    assert!(err.to_string().starts_with("unsupported: "), "{err}");
    assert!(err.to_string().contains("chelis#730"), "{err}");
    unsafe { std::env::remove_var("MOCK_BEACON_SCENARIO") };
}
