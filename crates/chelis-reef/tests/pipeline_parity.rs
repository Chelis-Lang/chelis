use chelis_reef::{BuildOptions, build_package_with_options, package_schema, verify_artifact_pair};
use chelis_shell::read_shell;
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{check_typed_program, errors::CheckErrorKind};
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
    assert!(BASELINE_PROVENANCE.contains("e1065d94fbd7a41f086e0690929a66c2335accc5"));
    assert!(BASELINE_PROVENANCE.contains("before the core extraction"));
}

// chelis#1198: artifact archive hashes are nondeterministic across runs,
// so the accepted pre-migration snapshot stays ignored. Rejected snapshots
// likewise pinned diagnostic wording and incidental offsets; the active
// rejection test checks the three observable failure boundaries instead.
#[test]
#[ignore = "nondeterministic package archive hash; see chelis#1198"]
fn accepted_package_outputs_match_the_pre_migration_baseline() {
    const CHILD_ENV: &str = "CHELIS_PIPELINE_PARITY_BASELINE_CHILD";
    if std::env::var_os(CHILD_ENV).is_none() {
        // `--include-ignored` is load-bearing: this leg is `#[ignore]`d for
        // chelis#1198, so without it `--exact` selects the leg but libtest
        // skips it, the child reports `0 passed; 1 ignored`, and the process
        // exits 0 -- a vacuous pass that asserts nothing. The output check
        // below makes "zero tests ran" a failure regardless of cause, so a
        // future rename that breaks `--exact` cannot silently re-hollow this.
        let output = std::process::Command::new(
            std::env::current_exe().expect("the test executable path must exist"),
        )
        .args([
            "--exact",
            "--include-ignored",
            "accepted_package_outputs_match_the_pre_migration_baseline",
        ])
        .env(CHILD_ENV, "1")
        .env("SOURCE_DATE_EPOCH", "315532800")
        .output()
        .expect("the deterministic baseline child must start");
        let report = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        assert!(
            report.contains("1 passed") || report.contains("1 failed"),
            "the deterministic baseline child executed no assertions \
             (a vacuous pass); child report:\n{report}"
        );
        assert!(
            output.status.success(),
            "the deterministic baseline child failed; child report:\n{report}"
        );
        return;
    }

    let (_directory, root) = accepted_package_fixture();
    let artifacts = build_package_with_options(
        &root,
        &BuildOptions { auto_fetch: false },
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("the accepted package must build");
    let verified = verify_artifact_pair(&artifacts.archive_path, &artifacts.shell_path)
        .expect("the accepted artifact pair must verify");
    let shell = read_shell(&artifacts.shell_path).expect("the accepted shell must decode");
    let schema = package_schema(&root, &chelis_std_bundle::EMBEDDED_RUNTIME)
        .expect("the accepted package schema must build");

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
fn rejected_packages_keep_type_effect_and_linearity_boundaries() {
    let (_directory, root) = rejected_package_fixture(TYPE_REJECTION_SOURCE);
    build_package_with_options(
        &root,
        &BuildOptions { auto_fetch: false },
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
        .expect_err("both unbound names must reject the package");
    let decls = parse_str(TYPE_REJECTION_SOURCE).expect("rejected package source must parse");
    let exprs = desugar_program(&decls).expect("rejected package source must desugar");
    let report = check_typed_program(&exprs).expect_err("both unbound names must reject at check");
    let mut unbound = report
        .errors
        .iter()
        .filter_map(|error| match &error.kind {
            CheckErrorKind::UnboundVariable { identifier } => Some(identifier.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    unbound.sort_unstable();
    assert_eq!(unbound, ["missing_first", "missing_second"]);

    let (_directory, root) = rejected_package_fixture(EFFECT_REJECTION_SOURCE);
    let effect = build_package_with_options(
        &root,
        &BuildOptions { auto_fetch: false },
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
        .expect_err("undeclared IO must reject");
    assert!(
        effect.contains("effects `{}`") && effect.contains("effects `{IO}`"),
        "the undeclared effect must be identified: {effect}"
    );

    let (_directory, root) = rejected_package_fixture(LINEARITY_REJECTION_SOURCE);
    let linearity = build_package_with_options(
        &root,
        &BuildOptions { auto_fetch: false },
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
        .expect_err("using both consumed tensors must reject");
    assert_eq!(
        linearity.matches("already consumed by realize").count(),
        2,
        "both consumed values must be reported: {linearity}"
    );
}
