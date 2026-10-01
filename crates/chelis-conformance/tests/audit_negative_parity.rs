//! Phase 2 oracle: spec-first negative parity for the conformance audit.
//!
//! Stamp a temp shell with `scaffold` (the core of `conform init`), confirm the
//! audit is green, then mutate it four ways and assert each mutation fails the
//! audit **on the correct row** with a non-empty diagnostic. This is the
//! authoritative Phase 2 completion oracle; the CLI `conform audit` verb is a
//! thin wrapper over `audit::audit`, smoke-tested separately.

use std::path::{Path, PathBuf};

use chelis_conformance::audit::{self, Verdict};
use chelis_conformance::scaffold;

// Must track the crate version: the audit's canonical-body comparison only
// applies to blocks stamped at AUDITOR_VERSION, so a hardcoded scaffold
// version silently skips the forged-block oracle after every release bump.
const VER: &str = env!("CARGO_PKG_VERSION");
// Known historical legacy profile implementation, not a consumer rollout approval.
const CENTRAL_SHA: &str = "4394706b569bdd7d557f6edc7b9818249decc330";
const ARCHIVE_DIGEST: &str =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn stamp(dir: &Path, name: &str) -> PathBuf {
    let root = dir.join(name);
    scaffold::scaffold(&root, name, "Myshell", VER).expect("scaffold");
    if matches!(name, "coral" | "nautilus") {
        if name == "coral" {
            let manifest = root.join("reef.toml");
            let reef = std::fs::read_to_string(&manifest).unwrap();
            std::fs::write(
                manifest,
                format!("{reef}\n[dependencies]\nnautilus = {{ version = \"4.5.6\" }}\n"),
            )
            .unwrap();
        }
        write_digest_lock(&root, ARCHIVE_DIGEST, ARCHIVE_DIGEST);
    }
    root
}

fn write_digest_lock(root: &Path, linux: &str, darwin: &str) {
    let lock = serde_json::json!({
        "schema": "chelis-toolchain-digests/v1",
        "versions": {
            (VER): {"linux-x86_64": linux, "darwin-arm64": darwin}
        }
    });
    std::fs::write(
        root.join(".github/chelis-toolchains.json"),
        lock.to_string(),
    )
    .unwrap();
}

fn verdict_of(report: &audit::AuditReport, key: &str) -> Verdict {
    report
        .rows
        .iter()
        .find(|r| r.key == key)
        .unwrap_or_else(|| panic!("no row with key {key:?}"))
        .verdict
}

fn diagnostic_of(report: &audit::AuditReport, key: &str) -> String {
    report
        .rows
        .iter()
        .find(|r| r.key == key)
        .map(|r| r.diagnostic.clone())
        .unwrap_or_default()
}

#[test]
fn scaffolded_shell_audits_green() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "greenshell");
    let report = audit::audit(&root);

    // No MUST-tier failures.
    let must_fails: Vec<_> = report
        .rows
        .iter()
        .filter(|r| {
            r.verdict == Verdict::Fail
                && !matches!(r.tier, chelis_conformance::manifest::Tier::Should)
        })
        .map(|r| format!("row {} ({}): {}", r.row, r.key, r.diagnostic))
        .collect();
    assert!(
        report.ok() && must_fails.is_empty(),
        "freshly scaffolded shell must audit green, but MUST rows failed:\n  {}",
        must_fails.join("\n  ")
    );
}

#[test]
fn deleting_chelis_surface_fails_row_7() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s7");
    std::fs::remove_file(root.join("docs/CHELIS_SURFACE.md")).unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "chelis-surface"), Verdict::Fail);
    assert!(diagnostic_of(&report, "chelis-surface").contains("CHELIS_SURFACE.md"));
}

/// chelis#2831: every upstream item is filed where it originates, so the
/// contract neither requires nor checks a local folder of issue drafts. A
/// scaffold writes none, no row reports on one, and adding one changes no
/// verdict.
#[test]
fn issue_draft_folder_is_neither_required_nor_checked() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nodrafts");
    assert!(!root.join("docs/issue_drafts").exists());

    let without = audit::audit(&root);
    let fails: Vec<_> = without
        .rows
        .iter()
        .filter(|r| r.verdict == Verdict::Fail)
        .map(|r| format!("row {} ({}): {}", r.row, r.key, r.diagnostic))
        .collect();
    assert!(fails.is_empty(), "no row of any tier may fail: {fails:?}");
    assert!(
        without
            .rows
            .iter()
            .all(|r| !r.key.contains("draft") && !r.diagnostic.contains("issue_drafts")),
        "no row may report on an issue-draft folder"
    );

    std::fs::create_dir_all(root.join("docs/issue_drafts")).unwrap();
    std::fs::write(root.join("docs/issue_drafts/README.md"), "# drafts\n").unwrap();
    let with = audit::audit(&root);
    let verdicts = |r: &audit::AuditReport| {
        r.rows
            .iter()
            .map(|row| (row.key, row.verdict))
            .collect::<Vec<_>>()
    };
    assert_eq!(verdicts(&without), verdicts(&with));
}

#[test]
fn editing_workflow_pin_fails_row_3() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s3");
    let ci = root.join(".github/workflows/ci.yml");
    let text = std::fs::read_to_string(&ci).unwrap();
    std::fs::write(
        &ci,
        text.replace(&format!("CHELIS_VERSION: {VER}"), "CHELIS_VERSION: 0.13.0"),
    )
    .unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
    assert!(diagnostic_of(&report, "workflow-env-pins").contains("0.13.0"));
}

fn central_profile_wrapper(profile: &str, inputs: &[String]) -> String {
    let mut lines = vec![
        "name: central".to_string(),
        "on: workflow_dispatch".to_string(),
        "permissions:".to_string(),
        "  contents: read".to_string(),
        "jobs:".to_string(),
        "  call-central:".to_string(),
        "    name: Call central profile".to_string(),
        format!("    uses: Chelis-Lang/ci/.github/workflows/consumer.yml@{CENTRAL_SHA}"),
        "    with:".to_string(),
        format!("      profile: {profile}"),
    ];
    lines.extend(inputs.iter().map(|input| format!("      {input}")));
    lines.extend([
        "    secrets:".to_string(),
        "      CHELIS_RELEASE_TOKEN: ${{ secrets.CHELIS_RELEASE_TOKEN }}".to_string(),
    ]);
    format!("{}\n", lines.join("\n"))
}

fn central_ci_wrapper(package: &str, version: &str) -> String {
    let profile = match package {
        "coral" => "coral-ci",
        "nautilus" => "nautilus-ci",
        other => panic!("unsupported central wrapper fixture {other}"),
    };
    let mut inputs = vec![
        format!("chelis-tag: v{version}"),
        format!("chelis-version: {version}"),
        format!("chelis-linux-sha256: {ARCHIVE_DIGEST}"),
        format!("chelis-darwin-sha256: {ARCHIVE_DIGEST}"),
    ];
    if package == "coral" {
        inputs.push("nautilus-tag: v4.5.6".to_string());
        inputs.push("package-version: 0.1.0".to_string());
    }
    central_profile_wrapper(profile, &inputs).replace(
        "on: workflow_dispatch",
        "on:\n  push:\n    branches: [main]\n  pull_request:\n  workflow_dispatch:",
    )
}

#[test]
fn immutable_central_ci_wrappers_satisfy_executed_contract_rows() {
    for package in ["coral", "nautilus"] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        std::fs::write(
            root.join(".github/workflows/ci.yml"),
            central_ci_wrapper(package, VER),
        )
        .unwrap();

        let report = audit::audit(&root);
        for row in [
            "workflow-env-pins",
            "pin-consistency-guard",
            "toolchain-installer",
            "tests-neg",
        ] {
            assert_eq!(
                verdict_of(&report, row),
                Verdict::Pass,
                "{package} central wrapper did not satisfy {row}: {}",
                diagnostic_of(&report, row)
            );
        }
        assert!(report.ok(), "{package}: central wrapper must audit green");
    }
}

#[test]
fn central_coral_inputs_must_match_package_and_nautilus_dependency_pins() {
    for (field, expected, replacement) in [
        ("package-version", "0.1.0", "0.1.1"),
        ("nautilus-tag", "v4.5.6", "v4.5.5"),
        ("nautilus-tag", "v4.5.6", "\"\""),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "coral");
        let wrapper = central_ci_wrapper("coral", VER).replace(
            &format!("{field}: {expected}"),
            &format!("{field}: {replacement}"),
        );
        std::fs::write(root.join(".github/workflows/ci.yml"), wrapper).unwrap();

        let report = audit::audit(&root);
        for row in ["workflow-env-pins", "pin-consistency-guard", "tests-neg"] {
            assert_eq!(
                verdict_of(&report, row),
                Verdict::Fail,
                "{field}: {replacement:?} incorrectly certified {row}"
            );
        }
        assert!(
            diagnostic_of(&report, "workflow-env-pins").contains(field),
            "{field} mismatch must be diagnosed"
        );
    }
}

#[test]
fn central_coral_guard_tracks_changed_reef_package_and_dependency_versions() {
    for (before, after, field) in [
        (
            "version = \"0.1.0\"",
            "version = \"0.1.1\"",
            "package-version",
        ),
        ("version = \"4.5.6\"", "version = \"4.5.5\"", "nautilus-tag"),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "coral");
        std::fs::write(
            root.join(".github/workflows/ci.yml"),
            central_ci_wrapper("coral", VER),
        )
        .unwrap();
        let reef = root.join("reef.toml");
        let manifest = std::fs::read_to_string(&reef).unwrap();
        std::fs::write(&reef, manifest.replace(before, after)).unwrap();

        let report = audit::audit(&root);
        assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
        assert!(
            diagnostic_of(&report, "workflow-env-pins").contains(field),
            "changed reef source for {field} must be diagnosed"
        );
        assert_eq!(verdict_of(&report, "pin-consistency-guard"), Verdict::Fail);
        assert_eq!(verdict_of(&report, "tests-neg"), Verdict::Fail);
    }
}

#[test]
fn historical_coral_guard_requires_readable_raw_reef_pins() {
    for (before, after, source) in [
        (
            "nautilus = { version = \"4.5.6\" }",
            "[dependencies.nautilus]\nversion = \"4.5.6\"",
            "nautilus inline version",
        ),
        (
            "version = \"0.1.0\"",
            " version = \"0.1.0\"",
            "package version",
        ),
        (
            "schema = \"3\"\n\n[package]",
            "schema = \"3\"\n\n[metadata]\nversion = \"9.9.9\"\n\n[package]",
            "package version",
        ),
        (
            &format!("compiler = \"={VER}\""),
            &format!(" compiler = \"={VER}\""),
            "compiler pin",
        ),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "coral");
        std::fs::write(
            root.join(".github/workflows/ci.yml"),
            central_ci_wrapper("coral", VER),
        )
        .unwrap();
        let manifest = root.join("reef.toml");
        let reef = std::fs::read_to_string(&manifest).unwrap();
        assert!(reef.contains(before), "{source}: fixture source missing");
        std::fs::write(manifest, reef.replace(before, after)).unwrap();

        let report = audit::audit(&root);
        for row in ["workflow-env-pins", "pin-consistency-guard", "tests-neg"] {
            assert_eq!(
                verdict_of(&report, row),
                Verdict::Fail,
                "{source}: historical guard cannot certify {row} from a TOML-only equivalent"
            );
        }
        assert!(
            diagnostic_of(&report, "workflow-env-pins").contains(source),
            "{source}: failure must identify the raw guard source"
        );
    }
}

#[test]
fn historical_nautilus_guard_requires_anchored_compiler_source() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nautilus");
    std::fs::write(
        root.join(".github/workflows/ci.yml"),
        central_ci_wrapper("nautilus", VER),
    )
    .unwrap();
    std::fs::create_dir_all(root.join("tests_blocked/example")).unwrap();
    std::fs::write(root.join("tests_blocked/example/case.ch"), "1\n").unwrap();
    let manifest = root.join("reef.toml");
    let reef = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(
        manifest,
        reef.replace(
            &format!("compiler = \"={VER}\""),
            &format!(" compiler = \"={VER}\""),
        ),
    )
    .unwrap();
    let report = audit::audit(&root);
    for row in [
        "workflow-env-pins",
        "pin-consistency-guard",
        "tests-neg",
        "tests-blocked",
    ] {
        assert_eq!(verdict_of(&report, row), Verdict::Fail, "{row}");
    }
}

#[test]
fn central_ci_digests_must_match_the_committed_lock_for_both_platforms() {
    let other_digest = format!("sha256:{}", "b".repeat(64));
    for package in ["coral", "nautilus"] {
        for platform in ["linux", "darwin"] {
            let tmp = tempfile::tempdir().unwrap();
            let root = stamp(tmp.path(), package);
            let old_input = format!("chelis-{platform}-sha256: {ARCHIVE_DIGEST}");
            let new_input = format!("chelis-{platform}-sha256: {other_digest}");
            let wrapper = central_ci_wrapper(package, VER).replace(&old_input, &new_input);
            std::fs::write(root.join(".github/workflows/ci.yml"), &wrapper).unwrap();
            if package == "nautilus" {
                std::fs::create_dir_all(root.join("tests_blocked/example")).unwrap();
                std::fs::write(root.join("tests_blocked/example/case.ch"), "1\n").unwrap();
            }

            let report = audit::audit(&root);
            for row in ["workflow-env-pins", "pin-consistency-guard", "tests-neg"] {
                assert_eq!(
                    verdict_of(&report, row),
                    Verdict::Fail,
                    "{package} {platform} digest drift incorrectly certified {row}"
                );
            }
            if package == "nautilus" {
                assert_eq!(verdict_of(&report, "tests-blocked"), Verdict::Fail);
            }
            assert!(
                diagnostic_of(&report, "workflow-env-pins").contains(platform),
                "{package} {platform} digest drift must be diagnosed"
            );

            let (linux, darwin) = if platform == "linux" {
                (other_digest.as_str(), ARCHIVE_DIGEST)
            } else {
                (ARCHIVE_DIGEST, other_digest.as_str())
            };
            write_digest_lock(&root, linux, darwin);
            let matching = audit::audit(&root);
            for row in ["workflow-env-pins", "pin-consistency-guard", "tests-neg"] {
                assert_eq!(
                    verdict_of(&matching, row),
                    Verdict::Pass,
                    "{package} matching {platform} lock should certify {row}"
                );
            }
            assert!(
                matching.ok(),
                "{package} matching {platform} lock must audit green"
            );
        }
    }
}

#[test]
fn central_ci_requires_a_valid_committed_digest_lock() {
    for package in ["coral", "nautilus"] {
        for invalid in ["missing", "wrong schema", "invalid other entry", "symlink"] {
            let tmp = tempfile::tempdir().unwrap();
            let root = stamp(tmp.path(), package);
            std::fs::write(
                root.join(".github/workflows/ci.yml"),
                central_ci_wrapper(package, VER),
            )
            .unwrap();
            let path = root.join(".github/chelis-toolchains.json");
            match invalid {
                "missing" => std::fs::remove_file(&path).unwrap(),
                "wrong schema" => {
                    let body = std::fs::read_to_string(&path).unwrap();
                    std::fs::write(&path, body.replace("chelis-toolchain-digests/v1", "v2"))
                        .unwrap();
                }
                "invalid other entry" => {
                    std::fs::write(
                        &path,
                        serde_json::json!({
                            "schema": "chelis-toolchain-digests/v1",
                            "versions": {
                                (VER): {
                                    "linux-x86_64": ARCHIVE_DIGEST,
                                    "darwin-arm64": ARCHIVE_DIGEST
                                },
                                "0.1.0": {"linux-x86_64": ARCHIVE_DIGEST}
                            }
                        })
                        .to_string(),
                    )
                    .unwrap();
                }
                "symlink" => {
                    let body = std::fs::read_to_string(&path).unwrap();
                    std::fs::write(root.join(".github/digest-real.json"), body).unwrap();
                    std::fs::remove_file(&path).unwrap();
                    std::os::unix::fs::symlink("digest-real.json", &path).unwrap();
                }
                _ => unreachable!(),
            }
            let report = audit::audit(&root);
            assert_eq!(
                verdict_of(&report, "workflow-env-pins"),
                Verdict::Fail,
                "{package}: {invalid} digest lock must fail the pin row"
            );
            assert_eq!(verdict_of(&report, "pin-consistency-guard"), Verdict::Fail);
            assert_eq!(verdict_of(&report, "tests-neg"), Verdict::Fail);
        }
    }
}

#[test]
fn central_ci_inputs_require_the_exact_reef_version_and_tag_shape() {
    for package in ["coral", "nautilus"] {
        for (needle, replacement) in [
            (format!("chelis-tag: v{VER}"), format!("chelis-tag: {VER}")),
            (
                format!("chelis-tag: v{VER}"),
                format!("chelis-tag: vv{VER}"),
            ),
            (
                format!("chelis-version: {VER}"),
                format!("chelis-version: v{VER}"),
            ),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            let root = stamp(tmp.path(), package);
            let wrapper = central_ci_wrapper(package, VER).replace(&needle, &replacement);
            std::fs::write(root.join(".github/workflows/ci.yml"), wrapper).unwrap();
            if package == "nautilus" {
                std::fs::create_dir_all(root.join("tests_blocked/example")).unwrap();
                std::fs::write(root.join("tests_blocked/example/case.ch"), "1\n").unwrap();
            }
            let report = audit::audit(&root);
            assert_eq!(
                verdict_of(&report, "workflow-env-pins"),
                Verdict::Fail,
                "{package}: {replacement} must not match the reef pin"
            );
            if package == "nautilus" {
                assert_eq!(verdict_of(&report, "tests-blocked"), Verdict::Fail);
            }
            assert!(!report.ok(), "{package}: {replacement} must fail the audit");
        }
    }
}

#[test]
fn central_ci_must_run_for_pull_requests_and_main_pushes() {
    for package in ["coral", "nautilus"] {
        let active = central_ci_wrapper(package, VER);
        let events = "on:\n  push:\n    branches: [main]\n  pull_request:\n  workflow_dispatch:";
        for replacement in [
            "on: workflow_dispatch",
            "on:",
            "on: {}",
            "",
            "on:\n  pull_request:\n  workflow_dispatch:",
            "on:\n  push:\n    branches: [main]\n  workflow_dispatch:",
            "on:\n  push:\n    branches: [dev]\n  pull_request:",
            "# on:\n#   push:\n#   pull_request:",
            "env:\n  NOTE: |\n    on:\n      push:\n      pull_request:",
            "on:\n  push:\n    branches: [main]\n    paths: [src/**]\n  pull_request:",
            "on:\n  push:\n    branches: [main, '!main']\n  pull_request:",
            "on:\n  push:\n    branches: [main]\n  pull_request:\n    types: [closed]",
            "on:\n  push:\n  pull_request:\non: workflow_dispatch",
        ] {
            let tmp = tempfile::tempdir().unwrap();
            let root = stamp(tmp.path(), package);
            let wrapper = active.replace(events, replacement);
            std::fs::write(root.join(".github/workflows/ci.yml"), wrapper).unwrap();
            if package == "nautilus" {
                std::fs::create_dir_all(root.join("tests_blocked/example")).unwrap();
                std::fs::write(root.join("tests_blocked/example/case.ch"), "1\n").unwrap();
            }
            let report = audit::audit(&root);
            for row in ["pin-consistency-guard", "tests-neg"] {
                assert_eq!(
                    verdict_of(&report, row),
                    Verdict::Fail,
                    "{package}: {replacement:?} must not certify {row}"
                );
            }
            if package == "nautilus" {
                assert_eq!(verdict_of(&report, "tests-blocked"), Verdict::Fail);
            }
            assert!(!report.ok(), "{package}: {replacement:?} must fail audit");
        }
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        let alternate = active.replace(
            events,
            "on:\n  pull_request:\n    branches:\n      - main\n  push: {}\n  workflow_dispatch:",
        );
        std::fs::write(root.join(".github/workflows/ci.yml"), alternate).unwrap();
        let report = audit::audit(&root);
        for row in ["pin-consistency-guard", "tests-neg"] {
            assert_eq!(
                verdict_of(&report, row),
                Verdict::Pass,
                "{package}: an unfiltered main push and PR targeting main must satisfy {row}"
            );
        }
        assert!(
            report.ok(),
            "{package}: legitimate change triggers must audit green"
        );
    }
}

#[test]
fn quoted_yaml_event_keys_share_the_structural_job_authority() {
    let events = "on:\n  push:\n    branches: [main]\n  pull_request:\n  workflow_dispatch:";
    for package in ["coral", "nautilus"] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        let wrapper = central_ci_wrapper(package, VER).replace(
            events,
            "\"on\":\n  \"push\":\n    branches: [main]\n  'pull_request':\n  workflow_dispatch:",
        );
        std::fs::write(root.join(".github/workflows/ci.yml"), &wrapper).unwrap();
        let report = audit::audit(&root);
        for row in ["workflow-env-pins", "pin-consistency-guard", "tests-neg"] {
            assert_eq!(verdict_of(&report, row), Verdict::Pass, "{package}: {row}");
        }
        assert!(
            report.ok(),
            "{package}: quoted event mapping is a live gate"
        );

        for invalid in [
            "\"on\":\n  \"push\":\n    branches: [main]\n    paths: [src/**]\n  'pull_request':",
            "\"on\":\n  \"push\":\n    branches: [main, '!main']\n  'pull_request':",
            "\"on\":\n  \"push\":\n    branches: [main]\n  'pull_request':\n    types: [closed]",
            "\"on\":\n  \"push\":\n    branches: [main]\n  'pull_request':\non: workflow_dispatch",
        ] {
            std::fs::write(
                root.join(".github/workflows/ci.yml"),
                central_ci_wrapper(package, VER).replace(events, invalid),
            )
            .unwrap();
            let report = audit::audit(&root);
            for row in ["pin-consistency-guard", "tests-neg"] {
                assert_eq!(
                    verdict_of(&report, row),
                    Verdict::Fail,
                    "{package}: {invalid:?} cannot certify {row}"
                );
            }
        }
    }
}

#[test]
fn central_nightly_and_release_profiles_are_real_install_contracts() {
    let cases = [
        (
            "coral",
            "coral-release",
            vec![format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
        ),
        (
            "nautilus",
            "nautilus-nightly",
            vec![
                format!("chelis-tag: v{VER}"),
                format!("chelis-version: {VER}"),
                format!("chelis-linux-sha256: {ARCHIVE_DIGEST}"),
            ],
        ),
        (
            "nautilus",
            "nautilus-release",
            vec![format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
        ),
    ];
    for (package, profile, inputs) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        std::fs::write(
            root.join(".github/workflows/ci.yml"),
            "name: inert\njobs:\n  no-install:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo no\n",
        )
        .unwrap();
        std::fs::remove_file(root.join(".github/workflows/bump-pr.yml")).unwrap();
        std::fs::write(
            root.join(".github/workflows/central.yml"),
            central_profile_wrapper(profile, &inputs),
        )
        .unwrap();

        let report = audit::audit(&root);
        assert_eq!(
            verdict_of(&report, "workflow-env-pins"),
            Verdict::Pass,
            "{profile} was not recognized: {}",
            diagnostic_of(&report, "workflow-env-pins")
        );
        assert_eq!(
            verdict_of(&report, "toolchain-installer"),
            Verdict::Pass,
            "{profile} was not recognized as an installer"
        );
    }
}

#[test]
fn historical_non_ci_profiles_require_their_raw_reef_sources() {
    for (package, profile, inputs, needs_package, needs_nautilus) in [
        (
            "coral",
            "coral-release",
            vec![format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
            true,
            true,
        ),
        (
            "nautilus",
            "nautilus-release",
            vec![format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
            true,
            false,
        ),
        (
            "nautilus",
            "nautilus-nightly",
            vec![
                format!("chelis-tag: v{VER}"),
                format!("chelis-version: {VER}"),
                format!("chelis-linux-sha256: {ARCHIVE_DIGEST}"),
            ],
            false,
            false,
        ),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        std::fs::write(
            root.join(".github/workflows/ci.yml"),
            "name: inert\njobs:\n  no-install:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo no\n",
        )
        .unwrap();
        std::fs::remove_file(root.join(".github/workflows/bump-pr.yml")).unwrap();
        std::fs::write(
            root.join(".github/workflows/central.yml"),
            central_profile_wrapper(profile, &inputs),
        )
        .unwrap();
        let manifest = root.join("reef.toml");
        let valid = std::fs::read_to_string(&manifest).unwrap();
        let mut mutations = vec![(
            format!("compiler = \"={VER}\""),
            format!(" compiler = \"={VER}\""),
        )];
        if needs_package {
            mutations.push((
                "version = \"0.1.0\"".to_string(),
                " version = \"0.1.0\"".to_string(),
            ));
        }
        if needs_nautilus {
            mutations.push((
                "nautilus = { version = \"4.5.6\" }".to_string(),
                "[dependencies.nautilus]\nversion = \"4.5.6\"".to_string(),
            ));
        }
        for (before, after) in mutations {
            assert!(
                valid.contains(&before),
                "{profile}: missing fixture source {before}"
            );
            std::fs::write(&manifest, valid.replace(&before, &after)).unwrap();
            let report = audit::audit(&root);
            assert_eq!(
                verdict_of(&report, "workflow-env-pins"),
                Verdict::Fail,
                "{profile}: unreadable {before} must fail the pin row"
            );
            assert_ne!(
                verdict_of(&report, "toolchain-installer"),
                Verdict::Pass,
                "{profile}: unreadable {before} must not certify an installer"
            );
        }
    }
}

#[test]
fn historical_coral_source_uses_first_grep_readable_nautilus_version() {
    for profile in ["coral-ci", "coral-release"] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "coral");
        std::fs::remove_file(root.join(".github/workflows/bump-pr.yml")).unwrap();
        let (workflow, caller) = if profile == "coral-ci" {
            (
                root.join(".github/workflows/ci.yml"),
                central_ci_wrapper("coral", VER),
            )
        } else {
            std::fs::write(
                root.join(".github/workflows/ci.yml"),
                "name: inert\njobs:\n  no-install:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo no\n",
            )
            .unwrap();
            (
                root.join(".github/workflows/central.yml"),
                central_profile_wrapper(
                    "coral-release",
                    &[format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
                ),
            )
        };
        std::fs::write(workflow, caller).unwrap();
        let manifest = root.join("reef.toml");
        let valid = std::fs::read_to_string(&manifest).unwrap();
        let original = "nautilus = { version = \"4.5.6\" }";
        for inline in [
            "nautilus = { version = \"4.5.6\" } # pin",
            "nautilus = { version = \"4.5.6\", features = [\"x\"] }",
            "nautilus = { features = [\"x\"], version = \"4.5.6\" }",
            "nautilus = { version = \"4.5.6\" }\nnautilus-addons = { version = \"0.1.0\" }",
            "nautilus-addons = { version = \"4.5.6\" }\nnautilus = { version = \"4.5.6\" }",
        ] {
            std::fs::write(&manifest, valid.replace(original, inline)).unwrap();
            let report = audit::audit(&root);
            let rows: &[&str] = if profile == "coral-ci" {
                &[
                    "workflow-env-pins",
                    "toolchain-installer",
                    "pin-consistency-guard",
                    "tests-neg",
                ]
            } else {
                &["workflow-env-pins", "toolchain-installer"]
            };
            for row in rows {
                assert_eq!(
                    verdict_of(&report, row),
                    Verdict::Pass,
                    "{profile}: {inline} must satisfy {row}"
                );
            }
        }
        for unreadable in [
            "[dependencies.nautilus]\nversion = \"4.5.6\"",
            "nautilus = { extra_version = \"9.9.9\", version = \"4.5.6\" }",
            "nautilus = { version = \"4.5.6\"",
            "nautilus-addons = { version = \"0.1.0\" }\nnautilus = { version = \"4.5.6\" }",
        ] {
            std::fs::write(&manifest, valid.replace(original, unreadable)).unwrap();
            let report = audit::audit(&root);
            assert_eq!(
                verdict_of(&report, "workflow-env-pins"),
                Verdict::Fail,
                "{profile}: {unreadable} must not certify the historical raw source"
            );
            assert_ne!(verdict_of(&report, "toolchain-installer"), Verdict::Pass);
            if profile == "coral-ci" {
                for row in ["pin-consistency-guard", "tests-neg"] {
                    assert_eq!(
                        verdict_of(&report, row),
                        Verdict::Fail,
                        "{unreadable}: {row}"
                    );
                }
            }
        }
    }
}

#[test]
fn central_non_ci_profiles_require_the_committed_linux_digest() {
    let other_digest = format!("sha256:{}", "b".repeat(64));
    for (package, profile, inputs) in [
        (
            "coral",
            "coral-release",
            vec![format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
        ),
        (
            "nautilus",
            "nautilus-release",
            vec![format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
        ),
        (
            "nautilus",
            "nautilus-nightly",
            vec![
                format!("chelis-tag: v{VER}"),
                format!("chelis-version: {VER}"),
                format!("chelis-linux-sha256: {ARCHIVE_DIGEST}"),
            ],
        ),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        std::fs::write(
            root.join(".github/workflows/ci.yml"),
            "name: inert\njobs:\n  no-install:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo no\n",
        )
        .unwrap();
        std::fs::remove_file(root.join(".github/workflows/bump-pr.yml")).unwrap();
        let wrapper =
            central_profile_wrapper(profile, &inputs).replace(ARCHIVE_DIGEST, &other_digest);
        std::fs::write(root.join(".github/workflows/central.yml"), wrapper).unwrap();

        let report = audit::audit(&root);
        assert_eq!(
            verdict_of(&report, "workflow-env-pins"),
            Verdict::Fail,
            "{profile}: installer digest must match the committed lock"
        );
        assert_ne!(verdict_of(&report, "toolchain-installer"), Verdict::Pass);
    }
}

#[test]
fn central_wrapper_pin_drift_fails_workflow_pin_row() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nautilus");
    std::fs::write(
        root.join(".github/workflows/ci.yml"),
        central_ci_wrapper("nautilus", "0.0.1"),
    )
    .unwrap();

    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
    assert!(diagnostic_of(&report, "workflow-env-pins").contains("0.0.1"));
}

#[test]
fn central_nautilus_profile_binds_blocked_suite() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nautilus");
    std::fs::write(
        root.join(".github/workflows/ci.yml"),
        central_ci_wrapper("nautilus", VER),
    )
    .unwrap();
    std::fs::create_dir_all(root.join("tests_blocked/example")).unwrap();
    std::fs::write(root.join("tests_blocked/example/case.ch"), "1\n").unwrap();

    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "tests-blocked"), Verdict::Pass);
}

#[test]
fn different_immutable_central_revision_cannot_supply_legacy_audit_authority() {
    for package in ["coral", "nautilus"] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        let wrapper = central_ci_wrapper(package, VER)
            .replace(CENTRAL_SHA, "31ab77c35018d72bccdcae8d7330b25312a38114");
        std::fs::write(root.join(".github/workflows/ci.yml"), wrapper).unwrap();
        std::fs::remove_file(root.join(".github/workflows/bump-pr.yml")).unwrap();
        let report = audit::audit(&root);
        assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
        assert_eq!(verdict_of(&report, "pin-consistency-guard"), Verdict::Fail);
        assert_eq!(verdict_of(&report, "tests-neg"), Verdict::Fail);
        assert_ne!(verdict_of(&report, "toolchain-installer"), Verdict::Pass);
        assert!(
            !report.ok(),
            "{package}: unknown central SHA cannot audit green"
        );
    }
}

#[test]
fn unrecognized_release_wrapper_cannot_hide_behind_known_ci_profile() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nautilus");
    std::fs::write(
        root.join(".github/workflows/ci.yml"),
        central_ci_wrapper("nautilus", VER),
    )
    .unwrap();
    let release = central_profile_wrapper(
        "nautilus-release",
        &[format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
    )
    .replace(CENTRAL_SHA, "31ab77c35018d72bccdcae8d7330b25312a38114");
    std::fs::write(root.join(".github/workflows/release.yml"), release).unwrap();
    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
    assert!(
        !report.ok(),
        "different central revision must not audit green"
    );
}

#[test]
fn mixed_central_repository_paths_and_revisions_cannot_hide_behind_known_ci() {
    for package in ["coral", "nautilus"] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        std::fs::write(
            root.join(".github/workflows/ci.yml"),
            central_ci_wrapper(package, VER),
        )
        .unwrap();
        let release = central_profile_wrapper(
            &format!("{package}-release"),
            &[format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
        );
        let other = root.join(".github/workflows/other.yml");
        std::fs::write(&other, &release).unwrap();
        let accepted = audit::audit(&root);
        assert_eq!(
            verdict_of(&accepted, "workflow-env-pins"),
            Verdict::Pass,
            "{package}: the accepted workflow path and revision must pass"
        );
        assert!(
            accepted.ok(),
            "{package}: valid CI and release callers must audit green"
        );

        for (repository, path, revision) in [
            ("chelis-lang/ci", ".github/workflows/consumer.yml", "main"),
            (
                "CHELIS-LANG/CI",
                ".github/workflows/consumer.yml",
                "31ab77c35018d72bccdcae8d7330b25312a38114",
            ),
            ("Chelis-Lang/CI", ".github/workflows/consumer.yml", "main"),
            (
                "cHeLiS-lAnG/cI",
                ".github/workflows/consumer.yml",
                "31ab77c35018d72bccdcae8d7330b25312a38114",
            ),
            ("Chelis-Lang/ci", ".github/workflows/consumer.yaml", "main"),
            (
                "CHELIS-LANG/CI",
                ".github/workflows/consumer.yaml",
                CENTRAL_SHA,
            ),
            ("chelis-lang/ci", ".github/workflows/other.yml", CENTRAL_SHA),
        ] {
            let unrecognized = release.replace(
                &format!("Chelis-Lang/ci/.github/workflows/consumer.yml@{CENTRAL_SHA}"),
                &format!("{repository}/{path}@{revision}"),
            );
            std::fs::write(&other, unrecognized).unwrap();
            let report = audit::audit(&root);
            assert_eq!(
                verdict_of(&report, "workflow-env-pins"),
                Verdict::Fail,
                "{package}: {repository}/{path}@{revision} is an unaccepted central pointer"
            );
            assert!(
                !report.ok(),
                "{package}: known CI must not conceal {repository}/{path}@{revision}"
            );
        }
    }
}

/// YAML accepts either indentation width. Neither an alternate pointer nor
/// a second accepted caller may hide behind a valid, canonical CI wrapper.
#[test]
fn structurally_mapped_job_refs_reject_unaccepted_and_extra_calls_at_either_indentation() {
    for package in ["coral", "nautilus"] {
        for indent in [2, 4] {
            for revision in [
                "main",
                "31ab77c35018d72bccdcae8d7330b25312a38114",
                CENTRAL_SHA,
            ] {
                let tmp = tempfile::tempdir().unwrap();
                let root = stamp(tmp.path(), package);
                std::fs::write(
                    root.join(".github/workflows/ci.yml"),
                    central_ci_wrapper(package, VER),
                )
                .unwrap();
                let other = format!(
                    "name: alternate\non: workflow_dispatch\njobs:\n{job}alternate:\n{field}uses: CHELIS-LANG/CI/.github/workflows/consumer.yml@{revision}\n",
                    job = " ".repeat(indent),
                    field = " ".repeat(indent * 2),
                );
                std::fs::write(root.join(".github/workflows/other.yml"), other).unwrap();
                let report = audit::audit(&root);
                assert_eq!(
                    verdict_of(&report, "workflow-env-pins"),
                    Verdict::Fail,
                    "{package}: {indent}/{field_indent} job @{revision} must not be laundered behind the accepted CI",
                    field_indent = indent * 2
                );
                assert!(
                    !report.ok(),
                    "{package}: {indent}/{field_indent} @{revision}",
                    field_indent = indent * 2
                );
            }
        }
    }
}

#[test]
fn known_caller_remains_recognized_when_jobs_use_four_eight_space_indentation() {
    for package in ["coral", "nautilus"] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        let canonical = central_ci_wrapper(package, VER);
        let (header, jobs) = canonical.split_once("jobs:\n").unwrap();
        let mut wrapper = format!("{header}jobs:\n");
        for job_line in jobs.lines() {
            let indent = job_line.len() - job_line.trim_start().len();
            wrapper.push_str(&" ".repeat(indent));
            wrapper.push_str(job_line);
            wrapper.push('\n');
        }
        std::fs::write(root.join(".github/workflows/ci.yml"), wrapper).unwrap();
        let report = audit::audit(&root);
        for row in ["workflow-env-pins", "pin-consistency-guard", "tests-neg"] {
            assert_eq!(verdict_of(&report, row), Verdict::Pass, "{package}: {row}");
        }
        assert!(
            report.ok(),
            "{package}: alternate indentation is valid YAML"
        );
    }
}

#[test]
fn quoted_job_keys_and_aliases_do_not_bypass_central_pointer_audit() {
    for package in ["coral", "nautilus"] {
        for other in [
            "name: alternate\non: workflow_dispatch\n\"jobs\":\n    alternate:\n        \"uses\": CHELIS-LANG/CI/.github/workflows/consumer.yml@main\n",
            "name: alternate\non: workflow_dispatch\nenv: &central\n  uses: CHELIS-LANG/CI/.github/workflows/consumer.yml@main\njobs:\n  alternate: *central\n",
            "name: alternate\non: workflow_dispatch\nenv: &central\n  uses: CHELIS-LANG/CI/.github/workflows/consumer.yml@main\njobs:\n  alternate:\n    <<: *central\n",
        ] {
            let tmp = tempfile::tempdir().unwrap();
            let root = stamp(tmp.path(), package);
            std::fs::write(
                root.join(".github/workflows/ci.yml"),
                central_ci_wrapper(package, VER),
            )
            .unwrap();
            std::fs::write(root.join(".github/workflows/other.yml"), other).unwrap();
            let report = audit::audit(&root);
            assert_eq!(
                verdict_of(&report, "workflow-env-pins"),
                Verdict::Fail,
                "{package}: quoted or aliased job-level pointer must fail"
            );
            assert!(!report.ok(), "{package}: quoted or aliased pointer");
        }
    }
}

#[test]
fn accepted_central_revision_keeps_github_repository_identity_case_insensitive() {
    for package in ["coral", "nautilus"] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), package);
        std::fs::write(
            root.join(".github/workflows/ci.yml"),
            central_ci_wrapper(package, VER),
        )
        .unwrap();
        let release = central_profile_wrapper(
            &format!("{package}-release"),
            &[format!("chelis-linux-sha256: {ARCHIVE_DIGEST}")],
        )
        .replace("Chelis-Lang/ci/", "cHeLiS-lAnG/CI/");
        std::fs::write(root.join(".github/workflows/other.yml"), release).unwrap();
        let report = audit::audit(&root);
        assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Pass);
        assert!(report.ok(), "{package}: repository casing is not a new SHA");
    }
}

#[test]
fn comments_run_blocks_nested_steps_and_inert_aliases_are_not_job_calls() {
    for package in ["coral", "nautilus"] {
        for other in [
            "name: logging\non: workflow_dispatch\njobs:\n  logging:\n    runs-on: ubuntu-latest\n    steps:\n      - run: \"echo 'uses: CHELIS-LANG/CI/.github/workflows/consumer.yml@main'\"\n",
            "name: logging\non: workflow_dispatch\n# uses: CHELIS-LANG/CI/.github/workflows/consumer.yml@main\njobs:\n    logging:\n        runs-on: ubuntu-latest\n        steps:\n            - run: |\n                uses: CHELIS-LANG/CI/.github/workflows/consumer.yml@main\n",
            "name: logging\non: workflow_dispatch\nenv:\n  NOTE: &annotation \"echo 'uses: CHELIS-LANG/CI/.github/workflows/consumer.yml@main'\"\njobs:\n  logging:\n    runs-on: ubuntu-latest\n    steps:\n      - run: *annotation\n",
            "name: logging\non: workflow_dispatch\njobs:\n  logging:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n        with:\n          uses: CHELIS-LANG/CI/.github/workflows/consumer.yml@main\n",
        ] {
            let tmp = tempfile::tempdir().unwrap();
            let root = stamp(tmp.path(), package);
            std::fs::write(
                root.join(".github/workflows/ci.yml"),
                central_ci_wrapper(package, VER),
            )
            .unwrap();
            std::fs::write(root.join(".github/workflows/other.yml"), other).unwrap();
            let report = audit::audit(&root);
            assert_eq!(
                verdict_of(&report, "workflow-env-pins"),
                Verdict::Pass,
                "{package}: {}",
                diagnostic_of(&report, "workflow-env-pins")
            );
            assert!(
                report.ok(),
                "{package}: inert YAML cannot become a job-level use"
            );
        }
    }
}

#[test]
fn duplicate_jobs_document_cannot_launder_an_inert_central_call() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nautilus");
    let wrapper = format!(
        "{}jobs:\n  unexpected:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo no\n",
        central_ci_wrapper("nautilus", VER)
    );
    std::fs::write(root.join(".github/workflows/ci.yml"), wrapper).unwrap();
    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "workflow-env-pins"), Verdict::Fail);
    assert_eq!(verdict_of(&report, "pin-consistency-guard"), Verdict::Fail);
    assert_eq!(verdict_of(&report, "tests-neg"), Verdict::Fail);
}

#[test]
fn inert_or_skippable_central_markers_do_not_satisfy_audit() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nautilus");
    let inert = [
        "name: inert".to_string(),
        "jobs:".to_string(),
        "  test:".to_string(),
        "    runs-on: ubuntu-latest".to_string(),
        "    steps:".to_string(),
        "      - run: |".to_string(),
        format!(
            "          echo 'uses: Chelis-Lang/ci/.github/workflows/consumer.yml@{CENTRAL_SHA}'"
        ),
        "          echo 'chelis reef conform audit'".to_string(),
        "          echo 'uses: ./.github/actions/install-chelis'".to_string(),
        "          echo 'chelis test tests_neg --expect neg'".to_string(),
    ]
    .join("\n");
    std::fs::write(root.join(".github/workflows/ci.yml"), inert).unwrap();
    std::fs::remove_file(root.join(".github/workflows/bump-pr.yml")).unwrap();

    let report = audit::audit(&root);
    assert_eq!(verdict_of(&report, "pin-consistency-guard"), Verdict::Fail);
    assert_ne!(verdict_of(&report, "toolchain-installer"), Verdict::Pass);
    assert_eq!(verdict_of(&report, "tests-neg"), Verdict::Fail);
}

#[test]
fn central_call_rejects_mutable_skipped_and_wrong_secret_shapes() {
    let cases = [
        central_ci_wrapper("nautilus", VER).replace(CENTRAL_SHA, "main"),
        central_ci_wrapper("nautilus", VER).replace("    with:\n", "    if: false\n    with:\n"),
        central_ci_wrapper("nautilus", VER)
            .replace("${{ secrets.CHELIS_RELEASE_TOKEN }}", "${{ github.token }}"),
        central_ci_wrapper("nautilus", VER).replace("profile: nautilus-ci", "profile: coral-ci"),
    ];
    for (index, wrapper) in cases.into_iter().enumerate() {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "nautilus");
        std::fs::write(root.join(".github/workflows/ci.yml"), wrapper).unwrap();
        let report = audit::audit(&root);
        assert_eq!(
            verdict_of(&report, "pin-consistency-guard"),
            Verdict::Fail,
            "case {index} must not become central authority"
        );
    }
}

#[test]
fn forking_a_skill_fails_row_14() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s14");
    let skill = root.join("agent-skills/spec-sync/SKILL.md");
    let mut text = std::fs::read_to_string(&skill).unwrap();
    text.push_str("\nlocally forked line\n");
    std::fs::write(&skill, text).unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "vendored-skills"), Verdict::Fail);
    assert!(diagnostic_of(&report, "vendored-skills").contains("spec-sync"));
}

#[test]
fn stale_managed_block_stamp_fails_row_1() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "s1");
    let agents = root.join("AGENTS.md");
    let text = std::fs::read_to_string(&agents).unwrap();
    // Restamp the managed block to an older version without touching the body,
    // simulating a pin bump that skipped `conform sync`.
    std::fs::write(
        &agents,
        text.replace(&format!("chelis@{VER}"), "chelis@0.13.0"),
    )
    .unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "agents-md"), Verdict::Fail);
    let diag = diagnostic_of(&report, "agents-md");
    assert!(
        diag.contains("stamped") && diag.contains("0.13.0"),
        "diag was: {diag}"
    );
}

/// B2: a managed block whose body is forged (edited *and* re-stamped so the
/// fence hash matches) at the current version passes integrity + version but
/// must be caught by the canonical-body comparison.
#[test]
fn self_consistent_forked_managed_block_fails() {
    use chelis_conformance::managed_block;

    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "forge");
    let agents = root.join("AGENTS.md");
    let text = std::fs::read_to_string(&agents).unwrap();

    let block = managed_block::find(&text, "agents-inheritance").unwrap();
    let (start, end) = block.span;
    let forged = managed_block::render(
        "agents-inheritance",
        VER,
        "Forged inheritance text that is not the canonical upstream.",
    );
    // Sanity: the forgery is self-consistent (would fool integrity_ok alone).
    let refound = managed_block::find(&forged, "agents-inheritance").unwrap();
    assert!(refound.integrity_ok() && refound.version == VER);

    std::fs::write(
        &agents,
        format!("{}{forged}{}", &text[..start], &text[end..]),
    )
    .unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok(), "a self-consistent fork must fail the audit");
    assert_eq!(verdict_of(&report, "agents-md"), Verdict::Fail);
    assert!(diagnostic_of(&report, "agents-md").contains("canonical"));
}

/// B1: a shell pinning below the contract baseline must not gate every row out
/// to `Na` and read as conformant.
#[test]
fn below_baseline_pin_does_not_pass_empty_shell() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("ancient");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("reef.toml"), "compiler = \"=0.1.0\"\n").unwrap();

    let report = audit::audit(&root);
    assert!(
        !report.ok(),
        "a bare shell pinning below baseline must not audit conformant"
    );
    assert!(
        report.evaluated_any(),
        "baseline rows must still apply under a floored pin"
    );
}

/// M1: an extra skill dir outside the pinned set is drift (additions, not just
/// edits/deletions, must fail).
#[test]
fn extra_skill_dir_fails_row_14() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "extra");
    std::fs::create_dir_all(root.join("agent-skills/rogue")).unwrap();
    std::fs::write(root.join("agent-skills/rogue/SKILL.md"), "forked\n").unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "vendored-skills"), Verdict::Fail);
    assert!(diagnostic_of(&report, "vendored-skills").contains("rogue"));
}

/// M1: an extra file inside a pinned skill dir is drift too.
#[test]
fn extra_file_in_skill_dir_fails_row_14() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "extraf");
    std::fs::write(root.join("agent-skills/spec-sync/EXTRA.md"), "x\n").unwrap();

    let report = audit::audit(&root);
    assert!(!report.ok());
    assert_eq!(verdict_of(&report, "vendored-skills"), Verdict::Fail);
    assert!(diagnostic_of(&report, "vendored-skills").contains("EXTRA.md"));
}

/// M1: `sync` (materialize) prunes drift so the tree converges back to green.
#[test]
fn sync_prunes_extra_skill_content() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "prune");
    std::fs::create_dir_all(root.join("agent-skills/rogue")).unwrap();
    std::fs::write(root.join("agent-skills/rogue/SKILL.md"), "forked\n").unwrap();
    std::fs::write(root.join("agent-skills/spec-sync/EXTRA.md"), "x\n").unwrap();
    assert!(!audit::audit(&root).ok(), "drift must be present first");

    scaffold::materialize_skills(&root).unwrap();
    assert!(
        !root.join("agent-skills/rogue").exists(),
        "rogue skill dir must be pruned"
    );
    assert!(
        !root.join("agent-skills/spec-sync/EXTRA.md").exists(),
        "extra skill file must be pruned"
    );
    assert!(audit::audit(&root).ok(), "audit green after prune");
}

// Row 8 (§4) cite-by-number has its own adversarial matrix in
// `tests/row8_citation_matrix.rs` (prose-name Fail, chelis#NNN / draft-path
// Pass, per-entry partition, sub-heading + ordered-list entries, nested-item
// handling, the honest-Manual fallback, §Archived exemption, and the
// malformed-heading fail-closed case). Kept there rather than duplicated here.

/// Tripwire: a MANIFEST row added without a `check_row` dispatch arm falls
/// through to the catch-all `Manual`, silently reporting "no check implemented"
/// instead of a real verdict. Auditing a scaffolded shell exercises every row
/// (the audit iterates all of MANIFEST); assert none carries the catch-all
/// sentinel, so adding a row without a check fails the build (chelis#739).
///
/// Boundary (chelis#739 red team, LOW): this is pin-scoped. `check_row` applies
/// `since_version` gating *before* the dispatch match, so a future row whose
/// `since_version` is above the scaffold's pin returns `Na` and never reaches
/// its arm — the tripwire would not exercise it. Today every row's
/// `since_version` is the contract baseline, so all 18 are exercised; a
/// future-dated row would need its own coverage.
#[test]
fn every_manifest_key_hits_a_real_arm() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "arms");
    let report = audit::audit(&root);
    let orphans: Vec<String> = report
        .rows
        .iter()
        .filter(|r| r.diagnostic.starts_with(audit::NO_CHECK_IMPLEMENTED_PREFIX))
        .map(|r| format!("row {} ({})", r.row, r.key))
        .collect();
    assert!(
        orphans.is_empty(),
        "MANIFEST rows with no check_row arm (fell through to the catch-all): {orphans:?}"
    );
}

/// M3: row 17 applicability is driven by the registry's authoritative
/// `links_chelis_crates` flag, not a Cargo.toml guess.
#[test]
fn chelis_src_trigger_comes_from_registry() {
    let tmp = tempfile::tempdir().unwrap();

    // `octant` links chelis crates (registry flag true), so [chelis-src] is
    // required even though the scaffolded tree has no Cargo path deps.
    let octant = stamp(tmp.path(), "octant");
    assert_eq!(
        verdict_of(&audit::audit(&octant), "chelis-src"),
        Verdict::Fail,
        "a registry crate-linking shell without [chelis-src] must fail row 17"
    );

    // `school` is a Reef shell (registry flag false), so it is NA even when its
    // Cargo.toml looks like it links crates — the registry overrides the guess.
    let school = stamp(tmp.path(), "school");
    std::fs::write(
        school.join("Cargo.toml"),
        "[dependencies]\nchelis-ir = { path = \"../chelis/crates/chelis-ir\" }\n",
    )
    .unwrap();
    assert_eq!(
        verdict_of(&audit::audit(&school), "chelis-src"),
        Verdict::Na,
        "the registry's Reef classification must override the Cargo.toml heuristic"
    );
}
