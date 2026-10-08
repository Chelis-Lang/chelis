//! Checked package calls reach the in-process native solver only when the
//! optional provider feature is selected.

use assert_cmd::Command;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn package() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/chelis-clarabel")
}

fn example() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/clarabel_qp")
}

fn eval(file: &str) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(package())
        .args(["eval", "--file", file])
        .output()
        .expect("run Chelis evaluator")
}

#[test]
fn checked_unconstrained_and_inequality_calls_solve() {
    for file in ["tests/solve.ch", "tests/nonnegative.ch"] {
        let result = eval(file);
        assert!(
            result.status.success(),
            "{file}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let expected: &[u8] = if file == "tests/solve.ch" {
            b"empty_values = []\nmain = true\n"
        } else {
            b"main = true\n"
        };
        assert_eq!(result.stdout, expected, "{file}");
    }
}

#[test]
fn path_dependency_calls_the_native_solver() {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args(["fmt", "src/main.ch", "--check"])
        .assert()
        .success();
    let check = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args(["check", "src/main.ch"])
        .output()
        .expect("check example");
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&check.stdout).expect("check JSON");
    assert_eq!(report["score"], 1.0);
    assert_eq!(report["errors"].as_array().map(Vec::len), Some(0));
    assert_eq!(report["unresolved_names"].as_array().map(Vec::len), Some(0));
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args(["eval", "--file", "src/main.ch"])
        .output()
        .expect("evaluate example");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.stdout, b"main = true\n");
}

#[test]
fn stopped_status_does_not_carry_solved_fields() {
    let result = eval("tests/stopped.ch");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.stdout, b"main = true\n");
}

#[test]
fn invalid_cone_partition_fails_before_solver() {
    let result = eval("tests/invalid_cone.ch");
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("cone dimensions sum to 2, but b and A have 1 rows")
    );
}

#[test]
fn wrong_tensor_dtype_is_rejected_by_checker() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(package())
        .args(["check", "tests/wrong_dtype.ch"])
        .output()
        .expect("check Chelis source");
    assert!(!result.status.success());
    let report = String::from_utf8_lossy(&result.stdout);
    assert!(report.contains("PrecisionMismatch"), "{report}");
}

#[test]
fn package_with_matching_linked_name_and_changed_source_runs_its_own_body() {
    let directory = tempfile::tempdir().expect("temporary package");
    std::fs::create_dir(directory.path().join("src")).expect("source directory");
    std::fs::create_dir(directory.path().join("tests")).expect("test directory");
    std::fs::copy(
        package().join("reef.toml"),
        directory.path().join("reef.toml"),
    )
    .expect("copy manifest");
    std::fs::copy(
        package().join("tests/solve.ch"),
        directory.path().join("tests/solve.ch"),
    )
    .expect("copy call");
    let source = std::fs::read_to_string(package().join("src/qp.ch")).expect("provider source");
    let changed = source.replace(
        "Clarabel native provider unavailable",
        "lookalike package body executed",
    );
    assert_ne!(changed, source);
    std::fs::write(directory.path().join("src/qp.ch"), changed).expect("write lookalike");
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(directory.path())
        .args(["eval", "--file", "tests/solve.ch"])
        .output()
        .expect("evaluate lookalike");
    assert!(
        !result.status.success(),
        "the lookalike must not invoke Clarabel"
    );
    let report = String::from_utf8_lossy(&result.stderr);
    assert!(
        report.contains("lookalike package body executed"),
        "{report}"
    );
}

#[test]
fn compiled_c_matches_evaluator_for_solved_and_stopped() {
    for (root, file) in [
        (package(), "tests/nonnegative.ch"),
        (package(), "tests/stopped.ch"),
        (example(), "src/main.ch"),
    ] {
        let directory = tempfile::tempdir().expect("build output");
        let result = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .current_dir(root)
            .args([
                "build",
                "--target",
                "c",
                "--output",
                directory.path().to_str().expect("UTF-8 test path"),
                file,
            ])
            .output()
            .expect("build Clarabel caller");
        assert!(
            result.status.success(),
            "{file}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let executable = directory
            .path()
            .join(std::path::Path::new(file).file_stem().expect("source stem"));
        let run = std::process::Command::new(executable)
            .output()
            .expect("run C caller");
        assert!(
            run.status.success(),
            "{file}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(run.stdout, b"main = true\n", "{file}");
    }
}

#[test]
fn c_build_keeps_changed_package_body_as_ordinary_chelis_code() {
    let directory = tempfile::tempdir().expect("temporary package");
    std::fs::create_dir(directory.path().join("src")).expect("source directory");
    std::fs::create_dir(directory.path().join("tests")).expect("test directory");
    std::fs::copy(
        package().join("reef.toml"),
        directory.path().join("reef.toml"),
    )
    .expect("copy manifest");
    std::fs::copy(
        package().join("tests/solve.ch"),
        directory.path().join("tests/solve.ch"),
    )
    .expect("copy call");
    let source = std::fs::read_to_string(package().join("src/qp.ch")).expect("provider source");
    let changed = source.replace(
        "Clarabel native provider unavailable",
        "lookalike package body executed",
    );
    assert_ne!(changed, source);
    std::fs::write(directory.path().join("src/qp.ch"), changed).expect("write lookalike");
    let output = directory.path().join("build");
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(directory.path())
        .args([
            "build",
            "--target",
            "c",
            "--output",
            output.to_str().expect("UTF-8 test path"),
            "tests/solve.ch",
        ])
        .output()
        .expect("build lookalike caller");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = std::process::Command::new(output.join("solve"))
        .output()
        .expect("run changed-source caller");
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("lookalike package body executed"));
}

#[test]
fn compiled_c_rejects_invalid_cone_partition() {
    let directory = tempfile::tempdir().expect("build output");
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(package())
        .args([
            "build",
            "--target",
            "c",
            "--output",
            directory.path().to_str().expect("UTF-8 test path"),
            "tests/invalid_cone.ch",
        ])
        .output()
        .expect("build invalid cone caller");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = std::process::Command::new(directory.path().join("invalid_cone"))
        .output()
        .expect("run invalid cone caller");
    assert!(!run.status.success());
    assert!(
        String::from_utf8_lossy(&run.stderr)
            .contains("cone dimensions sum to 2, but b and A have 1 rows")
    );
}

#[test]
fn compiled_c_solves_the_checked_qp() {
    let directory = tempfile::tempdir().expect("build output");
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(package())
        .args([
            "build",
            "--target",
            "c",
            "--output",
            directory.path().to_str().expect("UTF-8 test path"),
            "tests/solve.ch",
        ])
        .output()
        .expect("build Clarabel caller");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let archive = std::fs::read(directory.path().join("libchelis_runtime.a"))
        .expect("staged native provider archive");
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("chelis_runtime.receipt.json"))
            .expect("staged runtime receipt"),
    )
    .expect("runtime receipt JSON");
    assert_eq!(
        receipt["archive_sha256"],
        format!("{:x}", Sha256::digest(&archive))
    );
    let generated =
        std::fs::read_to_string(directory.path().join("solve.c")).expect("generated C source");
    assert!(generated.contains(&format!(
        "native provider archive sha256: {}",
        receipt["archive_sha256"].as_str().expect("archive digest")
    )));
    let executable = directory.path().join("solve");
    let run = std::process::Command::new(executable)
        .output()
        .expect("run compiled Clarabel caller");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(run.stdout, b"empty_values = []\nmain = true\n");
}

#[cfg(feature = "smt")]
#[test]
fn ideal_property_uses_the_registered_solved_call() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_stationary.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove ideal property");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "passed");
    assert_eq!(row["composite_verdict"], "proven_modulo_asserted_axiom");
    assert!(
        row["qualifiers"]
            .as_array()
            .is_some_and(|qualifiers| qualifiers.iter().any(|value| value == "real_arithmetic"))
    );
    assert!(row["assumptions"].as_array().is_some_and(|assumptions| {
        assumptions
            .iter()
            .any(|value| value["name"] == "clarabel.qp.ideal_optimality")
    }));
    assert!(row["assumptions"].as_array().is_some_and(|assumptions| {
        assumptions.iter().any(|value| {
            value["name"] == "clarabel.qp.ideal_optimality"
                && value["discharge"]["evidence"]["provider_archive_sha256"]
                    .as_str()
                    .is_some_and(|digest| {
                        digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit())
                    })
        })
    }));
    let independent: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .nth(1)
            .expect("independent property row"),
    )
    .expect("JSON independent property row");
    assert_eq!(independent["name"], "independent_identity");
    assert_eq!(independent["status"], "passed");
    assert_eq!(independent["assumptions"], serde_json::json!([]));
    assert_ne!(
        independent["composite_verdict"],
        "proven_modulo_asserted_axiom"
    );
}

#[cfg(feature = "smt")]
#[test]
fn ideal_property_compares_against_a_feasible_baseline() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_baseline.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove constrained baseline property");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "passed", "{row}");
    assert_eq!(row["composite_verdict"], "proven_modulo_asserted_axiom");
}

#[cfg(feature = "smt")]
#[test]
fn ideal_property_proves_symbolic_quality_against_a_runtime_baseline() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_symbolic_quality.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove symbolic quality property");
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "passed", "{row}");
    assert_eq!(row["samples"], 0, "{row}");
    assert_eq!(row["composite_verdict"], "proven_modulo_asserted_axiom");
}

#[cfg(feature = "smt")]
#[test]
fn ideal_property_proves_downstream_quality_for_symbolic_gram_qp() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_symbolic_gram_quality.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove symbolic Gram QP property");
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "passed", "{row}");
    assert_eq!(row["samples"], 0, "{row}");
    assert_eq!(row["composite_verdict"], "proven_modulo_asserted_axiom");
    assert!(
        row["qualifiers"]
            .as_array()
            .is_some_and(|values| values.contains(&serde_json::json!("real_arithmetic")))
    );
    let axiom = row["assumptions"]
        .as_array()
        .and_then(|records| {
            records
                .iter()
                .find(|record| record["name"] == "clarabel.qp.ideal_optimality")
        })
        .expect("ideal call assumption");
    assert_eq!(
        axiom["discharge"]["evidence"]["psd_evidence"],
        "verified_gram"
    );
    assert_eq!(
        axiom["discharge"]["evidence"]["call_fingerprint_sha256"]
            .as_str()
            .map(str::len),
        Some(64)
    );
}

#[cfg(feature = "smt")]
#[test]
fn ideal_property_proves_constrained_quality_only_for_a_feasible_baseline() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_symbolic_constrained_quality.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove symbolic constrained QP property");
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "passed", "{row}");
    assert_eq!(row["samples"], 0, "{row}");
    assert_eq!(row["composite_verdict"], "proven_modulo_asserted_axiom");
}

#[cfg(feature = "smt")]
#[test]
fn ideal_property_does_not_use_the_solver_axiom_for_a_different_q() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_symbolic_wrong_argument.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove changed-q symbolic quality property");
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "failed", "{row}");
    assert_ne!(row["composite_verdict"], "proven_modulo_asserted_axiom");
}

#[cfg(feature = "smt")]
#[test]
fn symbolic_p_requires_a_call_bound_psd_premise() {
    for (file, expected) in [
        ("ideal_symbolic_assumed_psd.ch", "passed"),
        ("ideal_symbolic_missing_psd.ch", "unsupported"),
        ("ideal_symbolic_indefinite_psd.ch", "unsupported"),
    ] {
        let output = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .current_dir(example())
            .args([
                "prove",
                &format!("tests/{file}"),
                "--tier",
                "smt-only",
                "--json",
            ])
            .output()
            .expect("prove symbolic P property");
        let row: serde_json::Value = serde_json::from_slice(
            output
                .stdout
                .split(|byte| *byte == b'\n')
                .next()
                .expect("property row"),
        )
        .expect("JSON property row");
        assert_eq!(row["status"], expected, "{file}: {row}");
        if expected == "passed" {
            let assumptions = row["assumptions"].as_array().expect("assumptions");
            assert!(
                assumptions
                    .iter()
                    .any(|entry| entry["name"] == "clarabel.qp.assume_psd")
            );
            assert!(
                assumptions
                    .iter()
                    .any(|entry| entry["name"] == "clarabel.qp.ideal_optimality")
            );
            let fingerprints = assumptions
                .iter()
                .map(|entry| entry["discharge"]["evidence"]["call_fingerprint_sha256"].as_str())
                .collect::<Vec<_>>();
            assert_eq!(fingerprints.len(), 2);
            assert_eq!(
                fingerprints[0], fingerprints[1],
                "both axioms must bind one call"
            );
            assert_eq!(row["composite_verdict"], "proven_modulo_asserted_axiom");
        }
    }
}

#[cfg(feature = "smt")]
#[test]
fn symbolic_qp_rejects_unproved_baseline_reversed_claim_and_vacuous_premises() {
    for file in [
        "ideal_symbolic_infeasible_baseline.ch",
        "ideal_symbolic_reversed_quality.ch",
        "ideal_symbolic_contradictory_where.ch",
        "ideal_symbolic_unsupported_division.ch",
    ] {
        let output = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .current_dir(example())
            .args([
                "prove",
                &format!("tests/{file}"),
                "--tier",
                "smt-only",
                "--json",
            ])
            .output()
            .expect("prove symbolic QP negative case");
        let row: serde_json::Value = serde_json::from_slice(
            output
                .stdout
                .split(|byte| *byte == b'\n')
                .next()
                .expect("property row"),
        )
        .expect("JSON property row");
        assert!(
            ["failed", "invalid", "unsupported"].contains(&row["status"].as_str().unwrap_or("")),
            "{file}: {row}"
        );
        assert_ne!(
            row["composite_verdict"], "proven_modulo_asserted_axiom",
            "{file}: {row}"
        );
    }
}

#[cfg(feature = "smt")]
#[test]
fn ideal_interior_optimum_is_stationary() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_interior.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove interior stationarity");
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "passed", "{row}");
    assert_eq!(row["composite_verdict"], "proven_modulo_asserted_axiom");
}

#[cfg(feature = "smt")]
#[test]
fn ideal_contract_follows_a_typed_chelis_wrapper() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_wrapper.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove wrapper property");
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "passed", "{row}");
    assert_eq!(row["composite_verdict"], "proven_modulo_asserted_axiom");
}

#[cfg(feature = "smt")]
#[test]
fn ideal_contract_uses_the_actual_q_argument() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_wrong_argument.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove changed-argument property");
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "failed", "{row}");
    assert_ne!(row["composite_verdict"], "proven_modulo_asserted_axiom");
}

#[cfg(feature = "smt")]
#[test]
fn changed_provider_source_cannot_grant_the_ideal_contract() {
    let directory = tempfile::tempdir().expect("temporary package");
    std::fs::create_dir(directory.path().join("src")).expect("example source directory");
    std::fs::create_dir_all(directory.path().join("provider/src"))
        .expect("provider source directory");
    std::fs::create_dir(directory.path().join("tests")).expect("test directory");
    std::fs::copy(
        package().join("reef.toml"),
        directory.path().join("provider/reef.toml"),
    )
    .expect("copy provider manifest");
    let original_manifest =
        std::fs::read_to_string(example().join("reef.toml")).expect("example manifest");
    let manifest = original_manifest.replace("../../packages/chelis-clarabel", "provider");
    assert_ne!(manifest, original_manifest);
    std::fs::write(directory.path().join("reef.toml"), manifest).expect("write example manifest");
    std::fs::copy(
        example().join("src/main.ch"),
        directory.path().join("src/main.ch"),
    )
    .expect("copy example source");
    std::fs::copy(
        example().join("tests/ideal_stationary.ch"),
        directory.path().join("tests/ideal_stationary.ch"),
    )
    .expect("copy property");
    let source = std::fs::read_to_string(package().join("src/qp.ch")).expect("provider source");
    let changed = source.replace(
        "Clarabel native provider unavailable",
        "lookalike package body",
    );
    assert_ne!(changed, source);
    std::fs::write(directory.path().join("provider/src/qp.ch"), changed).expect("write lookalike");
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(directory.path())
        .args([
            "prove",
            "tests/ideal_stationary.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove lookalike property");
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "unsupported", "{row}");
    assert_ne!(row["composite_verdict"], "proven_modulo_asserted_axiom");
}

#[cfg(feature = "smt")]
#[test]
fn unsupported_cone_does_not_grant_an_ideal_contract() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_unsupported_cone.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove unsupported-cone property");
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "unsupported", "{row}");
    assert_ne!(row["composite_verdict"], "proven_modulo_asserted_axiom");
}

#[cfg(feature = "smt")]
#[test]
fn zero_dimension_cones_do_not_grant_an_ideal_contract() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_zero_dimension_cones.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove zero-dimension-cone properties");
    let rows = result
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).expect("JSON property row"))
        .filter(|row| row["kind"] == "property")
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 2, "{rows:?}");
    for row in rows {
        assert_eq!(row["status"], "unsupported", "{row}");
        assert_ne!(row["composite_verdict"], "proven_modulo_asserted_axiom");
    }
}

#[cfg(feature = "smt")]
#[test]
fn stopped_branch_and_indefinite_p_do_not_grant_ideal_optimality() {
    for file in ["ideal_stopped_branch.ch", "ideal_indefinite.ch"] {
        let result = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .current_dir(example())
            .args([
                "prove",
                &format!("tests/{file}"),
                "--tier",
                "smt-only",
                "--json",
            ])
            .output()
            .expect("prove negative property");
        let row: serde_json::Value = serde_json::from_slice(
            result
                .stdout
                .split(|byte| *byte == b'\n')
                .next()
                .expect("property row"),
        )
        .expect("JSON property row");
        assert_eq!(row["status"], "unsupported", "{file}: {row}");
        assert_ne!(row["composite_verdict"], "proven_modulo_asserted_axiom");
    }
}

#[cfg(not(feature = "smt"))]
#[test]
fn ideal_contract_without_smt_is_explicitly_unsupported() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args([
            "prove",
            "tests/ideal_stationary.ch",
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("prove without SMT feature");
    let row: serde_json::Value = serde_json::from_slice(
        result
            .stdout
            .split(|byte| *byte == b'\n')
            .next()
            .expect("property row"),
    )
    .expect("JSON property row");
    assert_eq!(row["status"], "unsupported", "{row}");
    assert_ne!(row["composite_verdict"], "proven_modulo_asserted_axiom");
}
