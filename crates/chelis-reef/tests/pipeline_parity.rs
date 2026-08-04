use chelis_reef::{BuildOptions, build_package_with_options, package_schema, verify_artifact_pair};
use chelis_shell::read_shell;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

const BASELINE_PROVENANCE: &str = include_str!("fixtures/pipeline_parity/BASELINE.md");
const ACCEPTED_MANIFEST: &str = include_str!("fixtures/pipeline_parity/accepted/reef.toml");
const ACCEPTED_SOURCE: &str = include_str!("fixtures/pipeline_parity/accepted/src/main.ch");
const EXPECTED_SHELL: &str = include_str!("fixtures/pipeline_parity/accepted/expected_shell.json");
const EXPECTED_SCHEMA: &str =
    include_str!("fixtures/pipeline_parity/accepted/expected_schema.json");
const EXPECTED_HASHES: &str = include_str!("fixtures/pipeline_parity/accepted/expected_hashes.txt");
const REJECTED_MANIFEST: &str = include_str!("fixtures/pipeline_parity/rejected/reef.toml");
const TYPE_REJECTION_SOURCE: &str = include_str!("fixtures/pipeline_parity/rejected/type.ch");
const EFFECT_REJECTION_SOURCE: &str = include_str!("fixtures/pipeline_parity/rejected/effects.ch");
const LINEARITY_REJECTION_SOURCE: &str =
    include_str!("fixtures/pipeline_parity/rejected/linearity.ch");
const EXPECTED_TYPE_REJECTION: &str =
    include_str!("fixtures/pipeline_parity/rejected/expected_type.txt");
const EXPECTED_EFFECT_REJECTION: &str =
    include_str!("fixtures/pipeline_parity/rejected/expected_effects.txt");
const EXPECTED_LINEARITY_REJECTION: &str =
    include_str!("fixtures/pipeline_parity/rejected/expected_linearity.txt");

fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("fixture parent must be created");
    }
    fs::write(path, contents).expect("fixture file must be written");
}

fn accepted_package_fixture() -> (tempfile::TempDir, PathBuf) {
    let directory = tempdir().expect("fixture directory must be created");
    let root = directory.path().join("pipeline-parity");
    write(&root.join("reef.toml"), ACCEPTED_MANIFEST);
    write(&root.join("src/main.ch"), ACCEPTED_SOURCE);
    (directory, root)
}

fn rejected_package_fixture(source: &str) -> (tempfile::TempDir, PathBuf) {
    let directory = tempdir().expect("fixture directory must be created");
    let root = directory.path().join("pipeline-rejected");
    write(&root.join("reef.toml"), REJECTED_MANIFEST);
    write(&root.join("src/main.ch"), source);
    (directory, root)
}

fn pretty_json(value: &impl Serialize) -> String {
    format!(
        "{}\n",
        serde_json::to_string_pretty(value).expect("fixture output must serialize")
    )
}

#[test]
fn baseline_records_the_pre_migration_revision() {
    assert!(BASELINE_PROVENANCE.contains("261d64260d5a974c06f15245ecd31b671ecc80a4"));
    assert!(BASELINE_PROVENANCE.contains("before the core extraction"));
}

#[test]
fn accepted_package_outputs_match_the_pre_migration_baseline() {
    const CHILD_ENV: &str = "CHELIS_PIPELINE_PARITY_BASELINE_CHILD";
    if std::env::var_os(CHILD_ENV).is_none() {
        let status = std::process::Command::new(
            std::env::current_exe().expect("the test executable path must exist"),
        )
        .args([
            "--exact",
            "accepted_package_outputs_match_the_pre_migration_baseline",
        ])
        .env(CHILD_ENV, "1")
        .env("SOURCE_DATE_EPOCH", "315532800")
        .status()
        .expect("the deterministic baseline child must start");
        assert!(status.success(), "the deterministic baseline child failed");
        return;
    }

    let (_directory, root) = accepted_package_fixture();
    let artifacts = build_package_with_options(&root, &BuildOptions { auto_fetch: false })
        .expect("the accepted package must build");
    let verified = verify_artifact_pair(&artifacts.archive_path, &artifacts.shell_path)
        .expect("the accepted artifact pair must verify");
    let shell = read_shell(&artifacts.shell_path).expect("the accepted shell must decode");
    let schema = package_schema(&root).expect("the accepted package schema must build");

    let shell_json = pretty_json(&shell);
    let schema_json = pretty_json(&schema);
    let hashes = format!(
        "archive_sha256={}\nshell_sha256={}\n",
        artifacts.archive_sha256, artifacts.shell_sha256
    );

    assert_eq!(verified.package, artifacts.package);
    assert_eq!(verified.archive_sha256, artifacts.archive_sha256);
    assert_eq!(verified.shell_sha256, artifacts.shell_sha256);
    assert_eq!(shell_json, EXPECTED_SHELL);
    assert_eq!(schema_json, EXPECTED_SCHEMA);
    assert_eq!(hashes, EXPECTED_HASHES);
}

#[test]
fn rejected_package_errors_match_the_pre_migration_baseline() {
    let cases = [
        ("type", TYPE_REJECTION_SOURCE, EXPECTED_TYPE_REJECTION),
        (
            "effects",
            EFFECT_REJECTION_SOURCE,
            EXPECTED_EFFECT_REJECTION,
        ),
        (
            "linearity",
            LINEARITY_REJECTION_SOURCE,
            EXPECTED_LINEARITY_REJECTION,
        ),
    ];
    let actual = cases
        .iter()
        .map(|(_name, source, _)| {
            let (_directory, root) = rejected_package_fixture(source);
            let error = build_package_with_options(&root, &BuildOptions { auto_fetch: false })
                .expect_err("the rejected package must fail");
            format!("{error}\n")
        })
        .collect::<Vec<_>>();

    for ((name, _, expected), actual) in cases.into_iter().zip(actual) {
        assert_eq!(actual, expected, "{name} rejection changed");
    }
}
