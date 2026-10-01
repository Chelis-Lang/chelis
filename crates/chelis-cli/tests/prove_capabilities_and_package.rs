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
        "@property trivial forall(x: i32):\n  x == x\n",
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
        "@property always_true forall(x: i32):\n  x == x\n",
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

// `examples/beacon_scalar_range.ch` is the documented Beacon entry point. A
// style pass rewrote its goal into pipe form, which the Beacon route then
// rejected before lowering, and a stale declaration table rejected the
// lowered graph after that. Running the example pins both: without an engine
// binary the run must stop only at the missing binary, and with one the
// prover must dispatch to it.
#[cfg(feature = "chelis-prove")]
fn prove_beacon_example(beacon: Option<&std::path::Path>) -> Value {
    let example = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/beacon_scalar_range.ch");
    let mut command = Command::cargo_bin("chelis").unwrap();
    command.args([
        "prove",
        example.to_str().unwrap(),
        "--tier",
        "beacon-only",
        "--json",
    ]);
    match beacon {
        Some(path) => command.env("CHELIS_BEACON_BIN", path),
        None => command.env_remove("CHELIS_BEACON_BIN"),
    };
    let output = command.output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let record: Value = serde_json::from_str(stdout.lines().next().expect("property record"))
        .unwrap_or_else(|error| panic!("{error}: {stdout}"));
    assert_eq!(record["name"], "bounded_neuron", "{stdout}");
    assert_eq!(record["proof_tier"], "beacon", "{stdout}");
    record
}

#[cfg(feature = "chelis-prove")]
#[test]
fn beacon_example_reaches_the_engine_without_a_binary() {
    let record = prove_beacon_example(None);
    assert_eq!(record["status"], "unsupported", "{record}");
    assert_eq!(
        record["reason"], "CHELIS_BEACON_BIN is not configured",
        "{record}"
    );
}

#[cfg(all(feature = "chelis-prove", unix))]
#[test]
fn beacon_example_dispatches_to_the_configured_engine() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempdir().unwrap();
    let invoked = dir.path().join("invoked");
    let fake_beacon = dir.path().join("chelis-beacon");
    std::fs::write(
        &fake_beacon,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nexit 3\n",
            invoked.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&fake_beacon, std::fs::Permissions::from_mode(0o755)).unwrap();
    let record = prove_beacon_example(Some(&fake_beacon));
    let arguments = std::fs::read_to_string(&invoked)
        .unwrap_or_else(|error| panic!("engine was not invoked ({error}): {record}"));
    assert!(
        arguments.lines().any(|line| line == "--request"),
        "{arguments}"
    );
    assert_ne!(record["status"], "passed", "{record}");
}
