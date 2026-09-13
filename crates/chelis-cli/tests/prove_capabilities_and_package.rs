use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

// --- Issue #488: --capabilities ---

#[test]
fn prove_capabilities_emits_valid_json() {
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["prove", "--capabilities"])
        .output()
        .unwrap();
    assert!(output.status.success(), "exit 0");
    let caps: Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    assert_eq!(caps["schema_version"], 3);
    assert_eq!(caps["prove_json_schema_version"], 1);
    let tiers = caps["supported_tiers"].as_array().unwrap();
    assert!(tiers.contains(&Value::String("type_system".into())));
    assert!(tiers.contains(&Value::String("smt".into())));
    assert!(tiers.contains(&Value::String("fuzz".into())));
    assert!(tiers.contains(&Value::String("certified_envelope".into())));
    assert!(tiers.contains(&Value::String("beacon_scalar_real".into())));
    // Boolean fields exist
    assert!(caps["smt_available"].is_boolean());
    assert!(caps["beacon_available"].is_boolean());
    assert!(caps["beacon_contract_prover_available"].is_boolean());
    assert!(caps["dispatcher_available"].is_boolean());
    assert!(caps["obligation_engine_available"].is_boolean());
    // reachable_bs_tier is a string
    assert!(caps["reachable_bs_tier"].is_string());
    // supported_flags includes --package
    let flags = caps["supported_flags"].as_array().unwrap();
    assert!(flags.contains(&Value::String("--package".into())));
    assert!(flags.contains(&Value::String("--beacon-budget".into())));
    // engine_registry is an array
    assert!(caps["engine_registry"].is_array());
}

// --- Issue #673: --capabilities must not claim beacon is dispatchable ---

#[test]
fn prove_capabilities_distinguishes_scalar_route_from_contract_upgrade() {
    // A discoverable binary is NOT dispatchability. `with_beacon` has no caller
    // on the production dispatch path, so CHELIS_BEACON_BIN pointing at a real
    // executable must still not make the machine-readable surface say prove can
    // route to beacon -- in any of the three places that claim it.
    let dir = tempdir().unwrap();
    let fake_beacon = dir.path().join("chelis-beacon");
    std::fs::write(&fake_beacon, "#!/bin/sh\nexit 0\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake_beacon, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["prove", "--capabilities"])
        .env("CHELIS_BEACON_BIN", &fake_beacon)
        .output()
        .unwrap();
    assert!(output.status.success(), "exit 0");
    let caps: Value = serde_json::from_slice(&output.stdout).expect("valid JSON");

    assert_eq!(
        caps["beacon_available"],
        Value::Bool(cfg!(feature = "chelis-prove"))
    );
    assert_eq!(
        caps["beacon_scalar_available"],
        Value::Bool(cfg!(feature = "chelis-prove"))
    );
    assert_eq!(
        caps["beacon_wired"],
        Value::Bool(cfg!(feature = "chelis-prove"))
    );
    assert_eq!(caps["beacon_contract_prover_available"], Value::Bool(false));
    let registry = caps["engine_registry"].as_array().unwrap();
    assert!(
        registry.contains(&Value::String("beacon_shim".into())) == cfg!(feature = "chelis-prove"),
        "scalar route availability must agree with engine registry: {registry:?}"
    );
    // ...and the env var IS observed, so none of the above passed vacuously.
    assert_eq!(caps["beacon_binary_present"], Value::Bool(true));
}

#[test]
fn prove_capabilities_reports_beacon_binary_absent_without_env() {
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["prove", "--capabilities"])
        .env_remove("CHELIS_BEACON_BIN")
        .output()
        .unwrap();
    assert!(output.status.success(), "exit 0");
    let caps: Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    assert_eq!(caps["beacon_binary_present"], Value::Bool(false));
    assert_eq!(caps["beacon_available"], Value::Bool(false));
}

#[test]
fn prove_capabilities_does_not_require_input_files() {
    // --capabilities should exit 0 even without any .ch/.dp files present
    let dir = tempdir().unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["prove", "--capabilities"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());
}

// --- Issue #487: --package ---

#[test]
fn prove_package_flag_accepted() {
    // Just verify the flag is accepted (even if the path has no reef.toml,
    // the error should be about missing files, not an unknown flag).
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join("prop.ch"),
        "@property trivial forall(x: int32):\n  x == x\n",
    )
    .unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args([
            "prove",
            "--package",
            dir.path().to_str().unwrap(),
            dir.path().join("prop.ch").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    // Should not fail with "unknown argument"; the prove run itself may
    // pass or error depending on reef resolution, but the flag is accepted.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("unexpected argument"),
        "--package flag should be accepted: {stderr}"
    );
}

#[test]
fn prove_package_auto_detects_reef_toml() {
    // When no --package is passed and the file lives inside a reef package,
    // auto-detection should work (the existing behavior). We just verify
    // that a file with no imports in a non-package dir still proves fine.
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join("prop.ch"),
        "@property always_true forall(x: int32):\n  x == x\n",
    )
    .unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["prove", dir.path().join("prop.ch").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "trivial property should pass: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
