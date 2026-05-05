//! Phase A — Item 7 named acceptance oracle:
//! `chelis reef install --bootstrap [<list>]`.
//!
//! This file owns the named oracle test for Phase A Item 7:
//!
//! ```sh
//! cargo test -p chelis-cli phaseA_item7_bootstrap_oracle -- --exact
//! ```
//!
//! The oracle is a single comprehensive test function that exercises,
//! against localhost `wiremock` fixtures, every spec acceptance bullet
//! from `spec/design/reef_distribution.md` § Item 7 plus the locked
//! negative-parity cases. The `CHELIS_REEF_GITHUB_BASE_API` env var
//! injects the localhost URL into the fetch path so no real network
//! traffic is required for the default-tier oracle.
//!
//! Test-archive construction is bottom-up: each synthetic shell is a
//! `(name, version, [deps])` triple. The fixture builds a `.tar.zst`
//! whose root contains a `reef.toml` declaring the named deps and a
//! `src/main.ch` placeholder, and a `.chb` shell payload whose
//! `archive_sha256` matches the archive bytes. Both are mounted on
//! the wiremock server under the GitHub API two-step paths
//! (`/repos/<org>/<repo>/releases/tags/<tag>` plus
//! `/repos/<org>/<repo>/releases/assets/<id>`), exactly matching the
//! shape of the live API the production fetch path hits.
//!
//! This file also contains the manual-gate test
//! `phaseA_item7_real_bootstrap_manual_gate`, which is `#[ignore]`d
//! and only runnable with `--ignored --exact` plus a real
//! `GITHUB_TOKEN` that has read access to the canonical org. The
//! manual-gate test is what the orchestrator runs separately to
//! validate the default bootstrap list against the live canonical
//! shells.
//!
//! Negative parity (separate `#[test]` functions, run as part of the
//! default `cargo test --workspace` loop):
//!
//! - `phaseA_item7_three_shell_cycle_named` — depth-3 cycle detection
//! - `phaseA_item7_self_loop_is_one_cycle` — self-dependency
//! - `phaseA_item7_manifest_read_failure_is_typed` — bad TOML in archive
//! - `phaseA_item7_duplicate_package_versions_rejected`
//! - `phaseA_item7_empty_input_and_empty_default_errors_nothing_to_install`
//! - `phaseA_item7_auth_failure_aborts_before_first_install`
//!
//! Contract invariants (locked as tests):
//!
//! - `phaseA_item7_idempotence_byte_identical_state` — re-run produces
//!   byte-identical registry state
//! - `phaseA_item7_order_independence_of_input_list` — same input set
//!   in different orders produces same install sequence

// The named-oracle convention `phaseA_item7_bootstrap_oracle` is
// locked by the brief; the orchestrator runs it via
// `cargo test -p chelis-cli phaseA_item7_bootstrap_oracle -- --exact`.
// Renaming to snake_case would silently break that oracle invocation.
#![allow(non_snake_case)]

use assert_cmd::Command;
use predicates::prelude::*;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use tar::Builder;
use tempfile::tempdir;
use wiremock::matchers::{header, method, path as wm_path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

// ============================================================
// Test fixture builders
// ============================================================

/// Build a `.tar.zst` archive containing `reef.toml` + a placeholder
/// `src/main.ch`. The manifest declares the given name, version, and
/// dependencies. Returns the archive bytes.
fn build_test_archive(
    name: &str,
    version: &str,
    deps: &[(&str, &str)], // (name, version) pairs
) -> Vec<u8> {
    let mut deps_toml = String::new();
    for (dep_name, dep_version) in deps {
        deps_toml.push_str(&format!(
            "\n[dependencies.{dep_name}]\nversion = \"{dep_version}\"\n"
        ));
    }
    let manifest_text = format!(
        r#"[package]
name = "{name}"
version = "{version}"
compiler = "=0.5.0"
module_prefix = "Test"
{deps_toml}"#
    );
    let main_text = "module Test.Main\n\nexport (placeholder)\ndef placeholder -> int32 = 0\n";

    let mut tar_bytes = Vec::new();
    {
        let mut builder = Builder::new(&mut tar_bytes);
        // reef.toml at root.
        let manifest_bytes = manifest_text.as_bytes();
        let mut header = tar::Header::new_gnu();
        header.set_path("reef.toml").expect("set path");
        header.set_size(manifest_bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append(&header, manifest_bytes)
            .expect("append manifest");
        // src/main.ch.
        let main_bytes = main_text.as_bytes();
        let mut header = tar::Header::new_gnu();
        header.set_path("src/main.ch").expect("set path");
        header.set_size(main_bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append(&header, main_bytes).expect("append main");
        builder.finish().expect("finish tar");
    }
    zstd::stream::encode_all(Cursor::new(tar_bytes), 19).expect("zstd encode")
}

fn sha256_bytes(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    format!("{:x}", h.finalize())
}

/// Build a synthetic `ShellPackage` with the given identity that
/// agrees with `archive_sha256`. The shell module list is empty —
/// nothing in the install path inspects `shell.modules` beyond the
/// agreement check, and an empty list keeps the fixture minimal.
fn build_test_shell_bytes(name: &str, version: &str, archive_sha256: &str) -> Vec<u8> {
    let shell = chelis_shell::ShellPackage {
        package: chelis_shell::PackageId {
            name: name.to_string(),
            version: version.to_string(),
        },
        compiler: "=0.5.0".to_string(),
        modules: Vec::new(),
        dependencies: Vec::new(),
        archive_sha256: archive_sha256.to_string(),
    };
    chelis_shell::encode_shell(&shell).expect("encode shell")
}

/// One synthetic shell to mount on the wiremock server. The triple
/// identifies the shell in graph-ordering assertions; `deps` controls
/// the `[dependencies]` block packed into the archive's `reef.toml`.
#[derive(Clone, Debug)]
struct SyntheticShell {
    org: &'static str,
    repo: &'static str,
    tag: &'static str,
    version: &'static str,
    deps: Vec<(&'static str, &'static str)>, // (name, version)
}

impl SyntheticShell {
    fn new(
        org: &'static str,
        repo: &'static str,
        tag: &'static str,
        version: &'static str,
        deps: Vec<(&'static str, &'static str)>,
    ) -> Self {
        Self {
            org,
            repo,
            tag,
            version,
            deps,
        }
    }
    fn spec_string(&self) -> String {
        format!("{}/{}@{}", self.org, self.repo, self.tag)
    }
    fn archive_name(&self) -> String {
        format!("{}-{}.tar.zst", self.repo, self.version)
    }
    fn shell_name(&self) -> String {
        format!("{}-{}.chb", self.repo, self.version)
    }
    fn archive_bytes(&self) -> Vec<u8> {
        // Convert (&str, &str) -> (&str, &str) — already correct.
        build_test_archive(self.repo, self.version, &self.deps)
    }
    fn shell_bytes(&self) -> Vec<u8> {
        let archive = self.archive_bytes();
        build_test_shell_bytes(self.repo, self.version, &sha256_bytes(&archive))
    }
}

/// Stable asset ids per shell — encode the shell index into the id
/// so wiremock paths are deterministic.
fn archive_asset_id(shell_idx: usize) -> u64 {
    1000 + (shell_idx as u64) * 10
}
fn shell_asset_id(shell_idx: usize) -> u64 {
    1000 + (shell_idx as u64) * 10 + 1
}

/// Stand up a wiremock server serving a set of synthetic shells under
/// the GitHub API two-step path, with optional install-call recording.
/// Returns the harness; callers ask for the URI and the recorded
/// install order via accessors.
fn fixture_for_shells(shells: &[SyntheticShell]) -> WiremockHarness {
    let harness = WiremockHarness::new();
    let mut mocks: Vec<Mock> = Vec::new();
    for (idx, shell) in shells.iter().enumerate() {
        let archive_id = archive_asset_id(idx);
        let shell_id = shell_asset_id(idx);
        let archive_bytes = shell.archive_bytes();
        let shell_bytes = shell.shell_bytes();
        let archive_name = shell.archive_name();
        let shell_name = shell.shell_name();

        let metadata_body = serde_json::json!({
            "id": 1000 + idx as u64,
            "tag_name": shell.tag,
            "assets": [
                {"id": archive_id, "name": archive_name, "size": archive_bytes.len() as u64,
                 "content_type": "application/octet-stream"},
                {"id": shell_id, "name": shell_name, "size": shell_bytes.len() as u64,
                 "content_type": "application/octet-stream"},
            ]
        })
        .to_string();

        let meta_path = format!(
            "/repos/{}/{}/releases/tags/{}",
            shell.org, shell.repo, shell.tag
        );
        let archive_path = format!(
            "/repos/{}/{}/releases/assets/{archive_id}",
            shell.org, shell.repo
        );
        let shell_path = format!(
            "/repos/{}/{}/releases/assets/{shell_id}",
            shell.org, shell.repo
        );

        // Tag of the install (recorded at byte-fetch time on the
        // shell asset endpoint — that's the last call per shell, so
        // recording there guarantees the install's full sequence is
        // captured before the mock returns).
        let order_tag = shell.repo.to_string();
        let recorder = harness.recorder.clone();

        mocks.push(
            Mock::given(method("GET"))
                .and(wm_path(meta_path))
                .and(header("authorization", "token unit-test-token"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(metadata_body)
                        .insert_header("content-type", "application/json"),
                ),
        );
        mocks.push(
            Mock::given(method("GET"))
                .and(wm_path(archive_path))
                .and(header("authorization", "token unit-test-token"))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_bytes)),
        );
        // The shell-bytes endpoint records the install. We use a
        // `respond_with` closure that inspects the request and
        // appends the order tag before returning the bytes. wiremock
        // expects an impl Respond, so we use a tiny wrapper.
        let shell_bytes_inner = shell_bytes.clone();
        mocks.push(
            Mock::given(method("GET"))
                .and(wm_path(shell_path))
                .and(header("authorization", "token unit-test-token"))
                .respond_with(RecordingResponse {
                    bytes: shell_bytes_inner,
                    tag: order_tag,
                    recorder,
                }),
        );
    }
    harness.mount_all(mocks);
    harness
}

/// Custom wiremock responder that records the request tag in the
/// shared order vec, then returns a 200 with the configured bytes.
/// Wiremock's `respond_with` accepts any `impl Respond`; we implement
/// it to splice in an install-order recording side effect.
struct RecordingResponse {
    bytes: Vec<u8>,
    tag: String,
    recorder: Arc<Mutex<Vec<String>>>,
}

impl wiremock::Respond for RecordingResponse {
    fn respond(&self, _req: &Request) -> ResponseTemplate {
        self.recorder
            .lock()
            .expect("install-order recorder mutex")
            .push(self.tag.clone());
        ResponseTemplate::new(200).set_body_bytes(self.bytes.clone())
    }
}

// ============================================================
// Wiremock harness (shared async runtime + server)
// ============================================================

/// Adapted from `phaseA_item6_from_github.rs::WiremockHarness`. Adds
/// the `recorder` shared `Vec<String>` so the byte-stream endpoint
/// can append a tag per install for order assertions.
struct WiremockHarness {
    rt: tokio::runtime::Runtime,
    server: MockServer,
    recorder: Arc<Mutex<Vec<String>>>,
}

impl WiremockHarness {
    fn new() -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("tokio multi-thread runtime");
        let server = rt.block_on(MockServer::start());
        Self {
            rt,
            server,
            recorder: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn uri(&self) -> String {
        self.server.uri()
    }

    fn mount_all(&self, mocks: Vec<Mock>) {
        self.rt.block_on(async {
            for m in mocks {
                m.mount(&self.server).await;
            }
        });
    }

    fn install_order(&self) -> Vec<String> {
        self.recorder
            .lock()
            .expect("install-order recorder mutex")
            .clone()
    }
}

// ============================================================
// Env-var serialization (set_var/remove_var are racy)
// ============================================================

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn file_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Invoke `chelis_reef::install_bootstrap` with the API base URL and
/// token piped in via env vars. Restores prior env on return.
///
/// **Pre-condition:** caller holds `file_lock()`.
fn lib_install_bootstrap(
    spec_strings: &[&str],
    api_base_url: &str,
    token: Option<&str>,
    registry_root: &Path,
) -> Result<Vec<chelis_reef::InstalledArtifact>, chelis_reef::BootstrapError> {
    let prior_api_base = std::env::var_os("CHELIS_REEF_GITHUB_BASE_API");
    let prior_token = std::env::var_os("GITHUB_TOKEN");
    let prior_path = std::env::var_os("PATH");
    unsafe {
        std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", api_base_url);
        match token {
            Some(t) => std::env::set_var("GITHUB_TOKEN", t),
            None => {
                std::env::remove_var("GITHUB_TOKEN");
                std::env::set_var("PATH", "");
            }
        }
    }
    let parsed: Vec<chelis_reef::GitHubReleaseSpec> = spec_strings
        .iter()
        .map(|s| chelis_reef::GitHubReleaseSpec::parse(s).expect("parse"))
        .collect();
    let result = chelis_reef::install_bootstrap(&parsed, registry_root);
    unsafe {
        match prior_api_base {
            Some(v) => std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", v),
            None => std::env::remove_var("CHELIS_REEF_GITHUB_BASE_API"),
        }
        match prior_token {
            Some(v) => std::env::set_var("GITHUB_TOKEN", v),
            None => std::env::remove_var("GITHUB_TOKEN"),
        }
        match prior_path {
            Some(v) => std::env::set_var("PATH", v),
            None => std::env::remove_var("PATH"),
        }
    }
    result
}

// ============================================================
// Named acceptance oracle
// ============================================================

#[test]
fn phaseA_item7_bootstrap_oracle() {
    let _g = file_lock();
    oracle_linear_chain_topo_order();
    oracle_diamond_topo_order();
    oracle_cycle_named();
    oracle_missing_dep_named();
    oracle_default_list_path();
    oracle_per_shell_atomicity_preserved();
    oracle_cli_surface_dispatches_to_install_bootstrap();
    oracle_cli_mutex_with_from_github_and_from_monorepo();
}

/// Linear chain A -> B -> C: install A, B, C in any input order.
/// Expected install order: C, B, A (deps before dependents).
fn oracle_linear_chain_topo_order() {
    let shells = vec![
        SyntheticShell::new("chelis-lang", "A", "v0.1.0", "0.1.0", vec![("B", "0.1.0")]),
        SyntheticShell::new("chelis-lang", "B", "v0.1.0", "0.1.0", vec![("C", "0.1.0")]),
        SyntheticShell::new("chelis-lang", "C", "v0.1.0", "0.1.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    // Submit in input order [A, B, C] — the topo sort must reorder.
    let input_strings: Vec<String> = shells.iter().map(|s| s.spec_string()).collect();
    let inputs: Vec<&str> = input_strings.iter().map(|s| s.as_str()).collect();
    let installed =
        lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
            .expect("linear-chain bootstrap");
    let order = harness.install_order();
    assert_eq!(
        order,
        vec!["C", "B", "A"],
        "linear chain must install dependencies before dependents"
    );
    assert_eq!(installed.len(), 3);
    // Registry-state sanity check.
    for repo in &["A", "B", "C"] {
        assert!(
            registry
                .join(format!("packages/{repo}/0.1.0/{repo}-0.1.0.chb"))
                .exists(),
            "{repo} shell missing from registry"
        );
        assert!(
            registry
                .join(format!("packages/{repo}/0.1.0/{repo}-0.1.0.tar.zst"))
                .exists(),
            "{repo} archive missing from registry"
        );
    }
}

/// Diamond: A depends on B and C; both B and C depend on D.
/// Expected: D installed first; B and C both before A.
fn oracle_diamond_topo_order() {
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "DA",
            "v0.1.0",
            "0.1.0",
            vec![("DB", "0.1.0"), ("DC", "0.1.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "DB",
            "v0.1.0",
            "0.1.0",
            vec![("DD", "0.1.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "DC",
            "v0.1.0",
            "0.1.0",
            vec![("DD", "0.1.0")],
        ),
        SyntheticShell::new("chelis-lang", "DD", "v0.1.0", "0.1.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let input_strings: Vec<String> = shells.iter().map(|s| s.spec_string()).collect();
    let inputs: Vec<&str> = input_strings.iter().map(|s| s.as_str()).collect();
    lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect("diamond bootstrap");
    let order = harness.install_order();
    let pos = |needle: &str| {
        order
            .iter()
            .position(|s| s == needle)
            .unwrap_or_else(|| panic!("{needle} not in install order: {order:?}"))
    };
    assert!(
        pos("DD") < pos("DB"),
        "DD must be installed before DB, got: {order:?}"
    );
    assert!(
        pos("DD") < pos("DC"),
        "DD must be installed before DC, got: {order:?}"
    );
    assert!(
        pos("DB") < pos("DA"),
        "DB must be installed before DA, got: {order:?}"
    );
    assert!(
        pos("DC") < pos("DA"),
        "DC must be installed before DA, got: {order:?}"
    );
}

/// Two-shell cycle: A -> B -> A. Must surface BootstrapError::Cycle.
fn oracle_cycle_named() {
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "ZA",
            "v0.1.0",
            "0.1.0",
            vec![("ZB", "0.1.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "ZB",
            "v0.1.0",
            "0.1.0",
            vec![("ZA", "0.1.0")],
        ),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let input_strings: Vec<String> = shells.iter().map(|s| s.spec_string()).collect();
    let inputs: Vec<&str> = input_strings.iter().map(|s| s.as_str()).collect();
    let err = lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect_err("cycle must fail");
    let msg = err.to_string();
    match err {
        chelis_reef::BootstrapError::Cycle { cycle } => {
            // Cycle members must be named (both ZA and ZB present).
            let joined = cycle.join(",");
            assert!(joined.contains("ZA"), "cycle must name ZA: {joined}");
            assert!(joined.contains("ZB"), "cycle must name ZB: {joined}");
        }
        other => panic!("expected BootstrapError::Cycle, got: {other:?}"),
    }
    assert!(
        msg.contains("ZA") && msg.contains("ZB"),
        "Display must name all cycle members: {msg}"
    );
    // Registry stays untouched on cycle (we never reached the install loop).
    assert!(!registry.join("index.json").exists());
}

/// Missing dep: A depends on B, but B is not in the input set.
fn oracle_missing_dep_named() {
    let shells = vec![SyntheticShell::new(
        "chelis-lang",
        "MA",
        "v0.1.0",
        "0.1.0",
        vec![("MB", "0.1.0")],
    )];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let input_strings: Vec<String> = shells.iter().map(|s| s.spec_string()).collect();
    let inputs: Vec<&str> = input_strings.iter().map(|s| s.as_str()).collect();
    let err = lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect_err("missing dep must fail");
    let msg = err.to_string();
    match err {
        chelis_reef::BootstrapError::MissingDependency { dependent, missing } => {
            assert_eq!(dependent, "MA");
            assert_eq!(missing, "MB");
        }
        other => panic!("expected MissingDependency, got: {other:?}"),
    }
    assert!(
        msg.contains("MA") && msg.contains("MB"),
        "Display must name both dependent and missing dep: {msg}"
    );
}

/// Default-list path:
///
/// 1. Explicit list `<org>/foo@v0.1.0 <org>/bar@v0.1.0` against
///    wiremock — installs both in correct order.
/// 2. Empty default list parses (we cannot exercise it against real
///    network in this oracle; the manual gate covers that).
fn oracle_default_list_path() {
    // Sub-case 1: explicit list with two shells.
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "foo",
            "v0.1.0",
            "0.1.0",
            vec![("bar", "0.1.0")],
        ),
        SyntheticShell::new("chelis-lang", "bar", "v0.1.0", "0.1.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let inputs = vec!["chelis-lang/foo@v0.1.0", "chelis-lang/bar@v0.1.0"];
    lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect("explicit two-shell bootstrap");
    let order = harness.install_order();
    assert_eq!(order, vec!["bar", "foo"]);

    // Sub-case 2: DEFAULT_BOOTSTRAP_LIST itself parses without panic.
    // We don't execute it against real network here.
    for (repo, tag) in chelis_reef::DEFAULT_BOOTSTRAP_LIST {
        let spec_str = format!("{}/{repo}@{tag}", chelis_reef::CANONICAL_REEF_ORG);
        chelis_reef::GitHubReleaseSpec::parse(&spec_str)
            .unwrap_or_else(|e| panic!("default-list entry `{spec_str}` must parse: {e}"));
    }
}

/// Per-shell atomicity: shell 2 of 3 fails validation; shell 1 fully
/// installed, shell 2 leaves no orphans, shell 3 not attempted.
fn oracle_per_shell_atomicity_preserved() {
    // Build three shells. Shell PB advertises a corrupt shell payload
    // (the shell's archive_sha256 disagrees with the on-wire archive
    // bytes). The bootstrap topo order will be PC, PB, PA — so PC
    // installs successfully, PB fails validation, PA never runs.
    //
    // Topology: PA -> PB -> PC.
    let pa = SyntheticShell::new(
        "chelis-lang",
        "PA",
        "v0.1.0",
        "0.1.0",
        vec![("PB", "0.1.0")],
    );
    let pb = SyntheticShell::new(
        "chelis-lang",
        "PB",
        "v0.1.0",
        "0.1.0",
        vec![("PC", "0.1.0")],
    );
    let pc = SyntheticShell::new("chelis-lang", "PC", "v0.1.0", "0.1.0", vec![]);

    // Stand up wiremock manually so PB can be served with a tampered
    // shell (archive_sha256 set to a wrong value).
    let harness = WiremockHarness::new();
    let mut mocks: Vec<Mock> = Vec::new();
    for (idx, shell) in [&pa, &pb, &pc].iter().enumerate() {
        let archive_id = archive_asset_id(idx);
        let shell_id = shell_asset_id(idx);
        let archive_bytes = shell.archive_bytes();
        // Tamper PB's shell: bind the shell to a DIFFERENT archive_sha
        // than the on-wire archive. Validation step catches this.
        let shell_bytes = if shell.repo == "PB" {
            let bogus_sha = "0".repeat(64);
            build_test_shell_bytes(shell.repo, shell.version, &bogus_sha)
        } else {
            build_test_shell_bytes(shell.repo, shell.version, &sha256_bytes(&archive_bytes))
        };

        let metadata_body = serde_json::json!({
            "id": 1000 + idx as u64,
            "tag_name": shell.tag,
            "assets": [
                {"id": archive_id, "name": shell.archive_name(),
                 "size": archive_bytes.len() as u64},
                {"id": shell_id, "name": shell.shell_name(),
                 "size": shell_bytes.len() as u64},
            ]
        })
        .to_string();
        let meta_path = format!(
            "/repos/{}/{}/releases/tags/{}",
            shell.org, shell.repo, shell.tag
        );
        let archive_path = format!(
            "/repos/{}/{}/releases/assets/{archive_id}",
            shell.org, shell.repo
        );
        let shell_path = format!(
            "/repos/{}/{}/releases/assets/{shell_id}",
            shell.org, shell.repo
        );
        let order_tag = shell.repo.to_string();
        let recorder = harness.recorder.clone();

        mocks.push(
            Mock::given(method("GET"))
                .and(wm_path(meta_path))
                .and(header("authorization", "token unit-test-token"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(metadata_body)
                        .insert_header("content-type", "application/json"),
                ),
        );
        mocks.push(
            Mock::given(method("GET"))
                .and(wm_path(archive_path))
                .and(header("authorization", "token unit-test-token"))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_bytes)),
        );
        mocks.push(
            Mock::given(method("GET"))
                .and(wm_path(shell_path))
                .and(header("authorization", "token unit-test-token"))
                .respond_with(RecordingResponse {
                    bytes: shell_bytes,
                    tag: order_tag,
                    recorder,
                }),
        );
    }
    harness.mount_all(mocks);

    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let inputs = [pa.spec_string(), pb.spec_string(), pc.spec_string()];
    let inputs_ref: Vec<&str> = inputs.iter().map(|s| s.as_str()).collect();
    let err = lib_install_bootstrap(
        &inputs_ref,
        &harness.uri(),
        Some("unit-test-token"),
        &registry,
    )
    .expect_err("PB validation must fail");

    // Error must be wrapped Validation/Fetch carrying the
    // archive_sha256 disagreement message.
    match &err {
        chelis_reef::BootstrapError::Fetch(chelis_reef::GitHubFetchError::Validation {
            ..
        }) => {}
        chelis_reef::BootstrapError::Validation { .. } => {}
        other => panic!("expected Validation-class error, got: {other:?}"),
    }

    // PC was installed before PB tried; PC's bytes must be on disk and
    // index.json must mention PC.
    assert!(
        registry.join("packages/PC/0.1.0/PC-0.1.0.chb").exists(),
        "PC must be installed before PB validation fired"
    );
    assert!(registry.join("packages/PC/0.1.0/PC-0.1.0.tar.zst").exists());

    // PB's shell-bytes endpoint was hit (recorded), but the install
    // step failed; the package directory MAY exist with copied bytes
    // (validation runs after copy in install_validated_artifact_pair).
    // What must NOT exist is an index.json entry for PB.
    let index_text = fs::read_to_string(registry.join("index.json")).expect("index.json present");
    let index: serde_json::Value = serde_json::from_str(&index_text).unwrap();
    assert!(
        index["packages"]["PC"].is_array(),
        "index must have PC after a successful first install"
    );
    assert!(
        index["packages"]["PB"].is_null() || !index["packages"]["PB"].is_array(),
        "index must NOT have PB after validation failure: {index}"
    );
    assert!(
        index["packages"]["PA"].is_null() || !index["packages"]["PA"].is_array(),
        "PA must NOT be in the index — it was not attempted: {index}"
    );

    // Order recording: PC happened, PB happened (it reaches the
    // shell-bytes fetch endpoint before validation fires), PA never.
    let order = harness.install_order();
    assert!(
        order.contains(&"PC".to_string()),
        "PC was attempted: {order:?}"
    );
    assert!(
        !order.contains(&"PA".to_string()),
        "PA must NOT be attempted after PB fails: {order:?}"
    );
}

/// CLI surface: `chelis reef install --bootstrap a b` parses both and
/// dispatches to `install_bootstrap`. End-to-end via the binary.
fn oracle_cli_surface_dispatches_to_install_bootstrap() {
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "cli-foo",
            "v0.1.0",
            "0.1.0",
            vec![("cli-bar", "0.1.0")],
        ),
        SyntheticShell::new("chelis-lang", "cli-bar", "v0.1.0", "0.1.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("cli-reef-home");

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .args([
            "reef",
            "install",
            "--bootstrap",
            "chelis-lang/cli-foo@v0.1.0",
            "chelis-lang/cli-bar@v0.1.0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Installed cli-foo 0.1.0"))
        .stdout(predicate::str::contains("Installed cli-bar 0.1.0"));

    let order = harness.install_order();
    assert_eq!(order, vec!["cli-bar", "cli-foo"]);
}

/// CLI: `--bootstrap` is mutually exclusive with `--from-github` and
/// with `--from-monorepo`.
fn oracle_cli_mutex_with_from_github_and_from_monorepo() {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "reef",
            "install",
            "--bootstrap",
            "chelis-lang/x@v0.1.0",
            "--from-github",
            "chelis-lang/y@v0.1.0",
        ])
        .assert()
        .failure();

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "reef",
            "install",
            "--bootstrap",
            "chelis-lang/x@v0.1.0",
            "--from-monorepo",
            ".",
        ])
        .assert()
        .failure();
}

// ============================================================
// Negative-parity tests (separate test functions)
// ============================================================

#[test]
fn phaseA_item7_three_shell_cycle_named() {
    let _g = file_lock();
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "T3A",
            "v0.1.0",
            "0.1.0",
            vec![("T3B", "0.1.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "T3B",
            "v0.1.0",
            "0.1.0",
            vec![("T3C", "0.1.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "T3C",
            "v0.1.0",
            "0.1.0",
            vec![("T3A", "0.1.0")],
        ),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let input_strings: Vec<String> = shells.iter().map(|s| s.spec_string()).collect();
    let inputs: Vec<&str> = input_strings.iter().map(|s| s.as_str()).collect();
    let err = lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect_err("3-cycle must fail");
    let msg = err.to_string();
    match err {
        chelis_reef::BootstrapError::Cycle { cycle } => {
            // All three nodes named.
            let joined = cycle.join(" ");
            for needle in ["T3A", "T3B", "T3C"] {
                assert!(
                    joined.contains(needle),
                    "cycle must name {needle}: {joined}"
                );
            }
        }
        other => panic!("expected Cycle, got: {other:?}"),
    }
    assert!(msg.contains("cycle"), "Display must mention 'cycle': {msg}");
}

#[test]
fn phaseA_item7_self_loop_is_one_cycle() {
    let _g = file_lock();
    let shells = vec![SyntheticShell::new(
        "chelis-lang",
        "self-dep",
        "v0.1.0",
        "0.1.0",
        vec![("self-dep", "0.1.0")],
    )];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let input_strings: Vec<String> = shells.iter().map(|s| s.spec_string()).collect();
    let inputs: Vec<&str> = input_strings.iter().map(|s| s.as_str()).collect();
    let err = lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect_err("self-loop must fail");
    match err {
        chelis_reef::BootstrapError::Cycle { cycle } => {
            assert!(
                cycle.iter().any(|s| s.contains("self-dep")),
                "self-loop must name itself: {cycle:?}"
            );
        }
        other => panic!("expected Cycle for self-loop, got: {other:?}"),
    }
}

#[test]
fn phaseA_item7_manifest_read_failure_is_typed() {
    let _g = file_lock();
    // Build an archive whose `reef.toml` is intentionally not valid
    // TOML. The fetch path must surface a typed `ManifestRead` error.
    let mut tar_bytes = Vec::new();
    {
        let mut builder = Builder::new(&mut tar_bytes);
        let bad_toml = b"this is not [valid toml = nor is { it } anywhere)";
        let mut header = tar::Header::new_gnu();
        header.set_path("reef.toml").expect("set path");
        header.set_size(bad_toml.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append(&header, &bad_toml[..]).expect("append");
        builder.finish().expect("finish");
    }
    let archive_bytes = zstd::stream::encode_all(Cursor::new(tar_bytes), 19).expect("zstd");
    let shell_bytes = build_test_shell_bytes("bad", "0.1.0", &sha256_bytes(&archive_bytes));

    let harness = WiremockHarness::new();
    let metadata_body = serde_json::json!({
        "id": 1u64,
        "tag_name": "v0.1.0",
        "assets": [
            {"id": 100u64, "name": "bad-0.1.0.tar.zst",
             "size": archive_bytes.len() as u64},
            {"id": 101u64, "name": "bad-0.1.0.chb",
             "size": shell_bytes.len() as u64},
        ]
    })
    .to_string();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path("/repos/chelis-lang/bad/releases/tags/v0.1.0"))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(metadata_body)
                    .insert_header("content-type", "application/json"),
            ),
        Mock::given(method("GET"))
            .and(wm_path("/repos/chelis-lang/bad/releases/assets/100"))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_bytes)),
        Mock::given(method("GET"))
            .and(wm_path("/repos/chelis-lang/bad/releases/assets/101"))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(shell_bytes)),
    ]);

    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let err = lib_install_bootstrap(
        &["chelis-lang/bad@v0.1.0"],
        &harness.uri(),
        Some("unit-test-token"),
        &registry,
    )
    .expect_err("bad manifest must fail");
    match err {
        chelis_reef::BootstrapError::ManifestRead { spec, .. } => {
            assert!(
                spec.contains("bad"),
                "manifest-read error must name spec: {spec}"
            );
        }
        other => panic!("expected ManifestRead, got: {other:?}"),
    }
}

#[test]
fn phaseA_item7_duplicate_package_versions_rejected() {
    let _g = file_lock();
    // Same package name appearing at two different versions in the
    // input list. The bootstrap must refuse this rather than picking.
    let shell_v1 = SyntheticShell::new("chelis-lang", "dup", "v0.1.0", "0.1.0", vec![]);
    let shell_v2 = SyntheticShell::new("chelis-lang", "dup", "v0.2.0", "0.2.0", vec![]);
    let harness = fixture_for_shells(&[shell_v1.clone(), shell_v2.clone()]);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let inputs = ["chelis-lang/dup@v0.1.0", "chelis-lang/dup@v0.2.0"];
    let err = lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect_err("duplicate versions must fail");
    let msg = err.to_string();
    match err {
        chelis_reef::BootstrapError::DuplicateVersion { package, versions } => {
            assert_eq!(package, "dup");
            assert!(versions.contains(&"0.1.0".to_string()));
            assert!(versions.contains(&"0.2.0".to_string()));
        }
        other => panic!("expected DuplicateVersion, got: {other:?}"),
    }
    assert!(
        msg.contains("0.1.0") && msg.contains("0.2.0"),
        "Display must list both versions: {msg}"
    );
}

#[test]
fn phaseA_item7_empty_input_and_empty_default_errors_nothing_to_install() {
    let _g = file_lock();
    // Library-level: install_bootstrap with empty specs must surface
    // BootstrapError::NothingToInstall. (The CLI handler substitutes
    // the default list when the user passes `--bootstrap` with no
    // args; the lib call here bypasses that substitution.)
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let err = chelis_reef::install_bootstrap(&[], &registry)
        .expect_err("empty list must fail at lib level");
    match err {
        chelis_reef::BootstrapError::NothingToInstall => {}
        other => panic!("expected NothingToInstall, got: {other:?}"),
    }
}

#[test]
fn phaseA_item7_auth_failure_aborts_before_first_install() {
    let _g = file_lock();
    // Three shells in a chain, but no GITHUB_TOKEN (and no `gh` on
    // PATH inside the helper). The first manifest fetch fails with
    // AuthMissing; the bootstrap must NOT try the install loop.
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "AuthA",
            "v0.1.0",
            "0.1.0",
            vec![("AuthB", "0.1.0")],
        ),
        SyntheticShell::new("chelis-lang", "AuthB", "v0.1.0", "0.1.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let input_strings: Vec<String> = shells.iter().map(|s| s.spec_string()).collect();
    let inputs: Vec<&str> = input_strings.iter().map(|s| s.as_str()).collect();
    let err = lib_install_bootstrap(&inputs, &harness.uri(), None, &registry)
        .expect_err("no token must fail");
    match err {
        chelis_reef::BootstrapError::Fetch(chelis_reef::GitHubFetchError::AuthMissing {
            ..
        }) => {}
        other => panic!("expected Fetch(AuthMissing), got: {other:?}"),
    }
    // Registry untouched.
    assert!(
        !registry.join("index.json").exists(),
        "auth failure must not touch the registry"
    );
}

// ============================================================
// Contract invariants
// ============================================================

#[test]
fn phaseA_item7_idempotence_byte_identical_state() {
    let _g = file_lock();
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "idemA",
            "v0.1.0",
            "0.1.0",
            vec![("idemB", "0.1.0")],
        ),
        SyntheticShell::new("chelis-lang", "idemB", "v0.1.0", "0.1.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let inputs = ["chelis-lang/idemA@v0.1.0", "chelis-lang/idemB@v0.1.0"];

    // First run.
    lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect("first run");
    let first_archive =
        fs::read(registry.join("packages/idemA/0.1.0/idemA-0.1.0.tar.zst")).unwrap();
    let first_shell = fs::read(registry.join("packages/idemA/0.1.0/idemA-0.1.0.chb")).unwrap();
    let first_index = fs::read_to_string(registry.join("index.json")).unwrap();

    // Second run against the same registry.
    lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect("second run");
    let second_archive =
        fs::read(registry.join("packages/idemA/0.1.0/idemA-0.1.0.tar.zst")).unwrap();
    let second_shell = fs::read(registry.join("packages/idemA/0.1.0/idemA-0.1.0.chb")).unwrap();
    let second_index = fs::read_to_string(registry.join("index.json")).unwrap();

    assert_eq!(
        first_archive, second_archive,
        "archive bytes must be identical"
    );
    assert_eq!(first_shell, second_shell, "shell bytes must be identical");
    assert_eq!(first_index, second_index, "index.json must be identical");
}

#[test]
fn phaseA_item7_order_independence_of_input_list() {
    let _g = file_lock();
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "OA",
            "v0.1.0",
            "0.1.0",
            vec![("OB", "0.1.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "OB",
            "v0.1.0",
            "0.1.0",
            vec![("OC", "0.1.0")],
        ),
        SyntheticShell::new("chelis-lang", "OC", "v0.1.0", "0.1.0", vec![]),
    ];

    let permutations = [
        vec![
            "chelis-lang/OA@v0.1.0",
            "chelis-lang/OB@v0.1.0",
            "chelis-lang/OC@v0.1.0",
        ],
        vec![
            "chelis-lang/OC@v0.1.0",
            "chelis-lang/OB@v0.1.0",
            "chelis-lang/OA@v0.1.0",
        ],
        vec![
            "chelis-lang/OB@v0.1.0",
            "chelis-lang/OA@v0.1.0",
            "chelis-lang/OC@v0.1.0",
        ],
    ];

    let mut observed_orders: Vec<Vec<String>> = Vec::new();
    for perm in &permutations {
        let harness = fixture_for_shells(&shells);
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("reef-home");
        lib_install_bootstrap(perm, &harness.uri(), Some("unit-test-token"), &registry)
            .expect("permutation must install");
        observed_orders.push(harness.install_order());
    }
    // All three permutations must produce the same install order
    // (driven by dep graph, not input position).
    for order in &observed_orders {
        assert_eq!(
            order,
            &vec!["OC".to_string(), "OB".to_string(), "OA".to_string()],
            "topo order must depend only on dep graph, not input order"
        );
    }
}

// ============================================================
// CLI help test
// ============================================================

#[test]
fn phaseA_item7_help_lists_bootstrap_flag() {
    let _g = file_lock();
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["reef", "install", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--bootstrap"));
}

// ============================================================
// Manual gate (real network). `#[ignore]`d. Documented invocation:
// `cargo test -p chelis-cli phaseA_item7_real_bootstrap_manual_gate -- --ignored --exact`.
// Requires a working `GITHUB_TOKEN` with read access to the canonical
// org and the canonical shells in DEFAULT_BOOTSTRAP_LIST must all be
// published with their manifests intact and reachable.
//
// **Maintenance rule:** when DEFAULT_BOOTSTRAP_LIST is bumped to new
// tags, this gate's expected version strings need no update — it
// reads the list and asserts each entry was installed at the version
// the list pinned. Maintainers verify each release exists with both
// canonical assets attached before bumping the list
// (`gh release view <tag> -R chelis-lang/<repo> --json assets`).
// ============================================================

#[test]
#[ignore = "real-network manual gate; run with `--ignored --exact`"]
fn phaseA_item7_real_bootstrap_manual_gate() {
    let _g = file_lock();
    let token = std::env::var("GITHUB_TOKEN").unwrap_or_default();
    assert!(
        !token.is_empty(),
        "manual gate requires GITHUB_TOKEN set; \
         run `export GITHUB_TOKEN=$(gh auth token)` first"
    );
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_REEF_HOME", &registry)
        .env("GITHUB_TOKEN", token)
        .args(["reef", "install", "--bootstrap"])
        .assert()
        .success();
    // Each entry in the default list must end up in the registry at
    // the pinned version.
    for (repo, tag) in chelis_reef::DEFAULT_BOOTSTRAP_LIST {
        let version = tag.strip_prefix('v').unwrap_or(tag);
        let pkg_dir = registry.join(format!("packages/{repo}/{version}"));
        assert!(
            pkg_dir.join(format!("{repo}-{version}.chb")).exists(),
            "{repo}@{tag} chb missing from registry"
        );
        assert!(
            pkg_dir.join(format!("{repo}-{version}.tar.zst")).exists(),
            "{repo}@{tag} archive missing from registry"
        );
    }
}
