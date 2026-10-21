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
//! and runnable separately against an available compatible published graph.
//! Public releases work anonymously; optional GitHub authentication avoids
//! the anonymous API rate limit. Offline fixtures own default discovery's
//! recurrence, joint constraints, determinism, provenance and rejection tests.
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

const CURRENT_COMPILER_PIN: &str = concat!("=", env!("CARGO_PKG_VERSION"));

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
    schema: u32,
    compiler: &str,
) -> Vec<u8> {
    let mut deps_toml = String::new();
    for (dep_name, dep_version) in deps {
        deps_toml.push_str(&format!(
            "\n[dependencies.{dep_name}]\nversion = \"{dep_version}\"\n"
        ));
    }
    let schema_header = if schema == 0 {
        String::new()
    } else {
        format!("schema = \"{schema}\"\n")
    };
    let resolver = if schema >= 2 {
        "resolver = \"2\"\n"
    } else {
        ""
    };
    let manifest_text = format!(
        r#"{schema_header}[package]
name = "{name}"
version = "{version}"
compiler = "{compiler}"
module_prefix = "Test"
{resolver}{deps_toml}"#,
        compiler = compiler,
    );
    let main_text = "module Test.Main\n\nexport (placeholder)\ndef placeholder() -> i32 = 0\n";

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
fn build_test_shell_bytes(
    name: &str,
    version: &str,
    archive_sha256: &str,
    compiler: &str,
) -> Vec<u8> {
    let shell = chelis_shell_core::ShellPackage {
        format_version: chelis_shell_core::SHELL_FORMAT_VERSION,
        package: chelis_shell_core::PackageId {
            name: name.to_string(),
            version: version.to_string(),
        },
        compiler: compiler.to_string(),
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
    deps: Vec<(&'static str, String)>, // (name, requirement)
    schema: u32,
    compiler: &'static str,
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
            deps: deps
                .into_iter()
                .map(|(name, version)| (name, version.to_string()))
                .collect(),
            schema: 0,
            compiler: CURRENT_COMPILER_PIN,
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
        let deps = self
            .deps
            .iter()
            .map(|(name, requirement)| (*name, requirement.as_str()))
            .collect::<Vec<_>>();
        build_test_archive(self.repo, self.version, &deps, self.schema, self.compiler)
    }
    fn shell_bytes(&self) -> Vec<u8> {
        let archive = self.archive_bytes();
        build_test_shell_bytes(
            self.repo,
            self.version,
            &sha256_bytes(&archive),
            self.compiler,
        )
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
    mount_shells(&harness, shells, true);
    harness
}

fn mount_shells(harness: &WiremockHarness, shells: &[SyntheticShell], authenticated: bool) {
    let request = |path: String| {
        let mock = Mock::given(method("GET")).and(wm_path(path));
        if authenticated {
            mock.and(header("authorization", "token unit-test-token"))
        } else {
            mock
        }
    };
    let mut releases = std::collections::BTreeMap::<(&str, &str), Vec<serde_json::Value>>::new();
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
            "draft": false,
            "prerelease": false,
            "assets": [
                {"id": archive_id, "name": archive_name, "size": archive_bytes.len() as u64,
                 "content_type": "application/octet-stream"},
                {"id": shell_id, "name": shell_name, "size": shell_bytes.len() as u64,
                 "content_type": "application/octet-stream"},
            ]
        });
        releases
            .entry((shell.org, shell.repo))
            .or_default()
            .push(metadata_body.clone());

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
            request(meta_path).respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(metadata_body)
                    .insert_header("content-type", "application/json"),
            ),
        );
        mocks.push(
            request(archive_path)
                .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_bytes)),
        );
        // The shell-bytes endpoint records the install. We use a
        // `respond_with` closure that inspects the request and
        // appends the order tag before returning the bytes. wiremock
        // expects an impl Respond, so we use a tiny wrapper.
        let shell_bytes_inner = shell_bytes.clone();
        mocks.push(request(shell_path).respond_with(RecordingResponse {
            bytes: shell_bytes_inner,
            tag: order_tag,
            recorder,
        }));
    }
    for ((org, repo), rows) in releases {
        mocks.push(
            request(format!("/repos/{org}/{repo}/releases"))
                .respond_with(ResponseTemplate::new(200).set_body_json(rows)),
        );
    }
    harness.mount_all(mocks);
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

    fn requests(&self) -> Vec<Request> {
        self.rt
            .block_on(self.server.received_requests())
            .expect("request recording enabled")
    }

    fn reset(&self) {
        self.rt.block_on(self.server.reset());
        self.recorder.lock().expect("recorder").clear();
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
    let result = chelis_reef::install_bootstrap(
        &parsed,
        registry_root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    );
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

// Bootstrap runtime checks use the manifest's normal resolver semantics
// (reef_distribution.md, typed identities and Item 7).
fn runtime_requirement_fixture(schema: u32, requirement: &str) -> SyntheticShell {
    let mut shell = SyntheticShell::new("chelis-lang", "runtime-user", "v0.2.0", "0.2.0", vec![]);
    shell.schema = schema;
    shell.deps.push(("chelis-std", requirement.to_string()));
    shell
}

#[test]
fn bootstrap_runtime_requirement_forms_match_bundled() {
    let _g = file_lock();
    let bundled = chelis_std_bundle::EMBEDDED_RUNTIME.version();
    let cases = [
        (1, bundled.to_string()),
        (2, format!("={bundled}")),
        (2, bundled.to_string()),
        (2, format!("^{bundled}")),
        (2, format!("~{bundled}")),
        (2, format!(">={bundled}, <99.0.0")),
        (2, "*".to_string()),
        (3, format!("={bundled}")),
    ];
    for (schema, requirement) in cases {
        let shell = runtime_requirement_fixture(schema, &requirement);
        let harness = fixture_for_shells(std::slice::from_ref(&shell));
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_REEF_HOME", &registry)
            .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
            .env("GITHUB_TOKEN", "unit-test-token")
            .env("PATH", "")
            .args(["reef", "install", "--bootstrap", &shell.spec_string()])
            .assert()
            .success()
            .stdout(predicate::str::contains("Installed runtime-user 0.2.0"));
        assert_eq!(harness.install_order(), vec!["runtime-user"]);
        let index = fs::read_to_string(registry.join("index.json")).expect("registry index");
        assert!(
            index.contains("runtime-user"),
            "schema {schema}: {requirement}"
        );
        assert!(
            !index.contains("chelis-std"),
            "runtime must remain compiler-bundled"
        );
    }
}

#[test]
fn bootstrap_runtime_requirement_mismatches_rejected_before_install() {
    let _g = file_lock();
    for (schema, requirement) in [(1, "99.0.0"), (2, "=99.0.0"), (2, "^99.0.0"), (2, "<0.0.0")] {
        let shell = runtime_requirement_fixture(schema, requirement);
        let harness = fixture_for_shells(std::slice::from_ref(&shell));
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        let spec = shell.spec_string();
        let error =
            lib_install_bootstrap(&[&spec], &harness.uri(), Some("unit-test-token"), &registry)
                .expect_err("incompatible runtime requirement must fail");
        assert!(matches!(
            error,
            chelis_reef::BootstrapError::Validation { .. }
        ));
        let message = error.to_string();
        assert!(message.contains(requirement), "{message}");
        assert!(
            message.contains(chelis_std_bundle::EMBEDDED_RUNTIME.version()),
            "{message}"
        );
        assert!(message.contains("cannot be substituted"), "{message}");
        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_REEF_HOME", &registry)
            .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
            .env("GITHUB_TOKEN", "unit-test-token")
            .env("PATH", "")
            .args(["reef", "install", "--bootstrap", &spec])
            .assert()
            .failure()
            .stderr(predicate::str::contains("cannot be substituted"));
        assert!(harness.install_order().is_empty());
        assert!(
            !registry.exists(),
            "rejection must precede registry mutation"
        );
    }
}

#[test]
fn bootstrap_invalid_runtime_requirements_use_manifest_validation() {
    let _g = file_lock();
    for (schema, requirement) in [(1, "=0.4.0"), (2, "not-a-version"), (2, "=0.4.0+build")] {
        let shell = runtime_requirement_fixture(schema, requirement);
        let harness = fixture_for_shells(std::slice::from_ref(&shell));
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        let spec = shell.spec_string();
        let error =
            lib_install_bootstrap(&[&spec], &harness.uri(), Some("unit-test-token"), &registry)
                .expect_err("invalid requirement must fail manifest validation");
        assert!(
            matches!(error, chelis_reef::BootstrapError::ManifestRead { .. }),
            "{error}"
        );
        assert!(error.to_string().contains(requirement), "{error}");
        assert!(harness.install_order().is_empty());
        assert!(!registry.exists());
    }
}

fn peer_requirement_fixture(schema: u32, requirement: &str) -> SyntheticShell {
    let mut shell = SyntheticShell::new("chelis-lang", "peer-user", "v0.2.0", "0.2.0", vec![]);
    shell.schema = schema;
    shell.deps.push(("peer", requirement.to_string()));
    shell
}

#[test]
fn bootstrap_peer_requirement_forms_match_selected_version() {
    let _g = file_lock();
    for (schema, requirement) in [
        (1, "0.2.0"),
        (2, "=0.2.0"),
        (2, "^0.2.0"),
        (2, "~0.2"),
        (2, "0.2.*"),
        (2, ">=0.2.0, <0.3.0"),
        (3, "=0.2.0"),
    ] {
        let shell = peer_requirement_fixture(schema, requirement);
        let peer = SyntheticShell::new("chelis-lang", "peer", "v0.2.0", "0.2.0", vec![]);
        let harness = fixture_for_shells(&[shell.clone(), peer.clone()]);
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
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
                &shell.spec_string(),
                &peer.spec_string(),
            ])
            .assert()
            .success()
            .stdout(predicate::str::contains("Installed peer-user 0.2.0"))
            .stdout(predicate::str::contains("Installed peer 0.2.0"));
        assert_eq!(
            harness.install_order(),
            vec!["peer", "peer-user"],
            "schema {schema}: {requirement}"
        );
    }
}

#[test]
fn bootstrap_incompatible_peer_requirement_rejected_before_install() {
    let _g = file_lock();
    for (schema, requirement) in [(1, "0.3.0"), (2, "=0.3.0"), (2, "^0.3.0"), (2, "<0.2.0")] {
        let shell = peer_requirement_fixture(schema, requirement);
        let peer = SyntheticShell::new("chelis-lang", "peer", "v0.2.0", "0.2.0", vec![]);
        let harness = fixture_for_shells(&[shell.clone(), peer.clone()]);
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        let spec = shell.spec_string();
        let peer_spec = peer.spec_string();
        let error = lib_install_bootstrap(
            &[&spec, &peer_spec],
            &harness.uri(),
            Some("unit-test-token"),
            &registry,
        )
        .expect_err("incompatible explicit peer must fail");
        assert!(
            matches!(error, chelis_reef::BootstrapError::Validation { .. }),
            "{error}"
        );
        let message = error.to_string();
        assert!(
            message.contains("peer-user") && message.contains("peer"),
            "{message}"
        );
        assert!(
            message.contains(requirement) && message.contains("0.2.0"),
            "{message}"
        );
        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_REEF_HOME", &registry)
            .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
            .env("GITHUB_TOKEN", "unit-test-token")
            .env("PATH", "")
            .args(["reef", "install", "--bootstrap", &spec, &peer_spec])
            .assert()
            .failure()
            .stderr(predicate::str::contains("selects `0.2.0`"));
        assert!(harness.install_order().is_empty());
        assert!(!registry.exists());
    }
}

#[test]
fn bootstrap_invalid_peer_requirements_use_manifest_validation() {
    let _g = file_lock();
    for (schema, requirement) in [(1, "=0.2.0"), (2, "not-a-version"), (3, "=0.2.0+build")] {
        let shell = peer_requirement_fixture(schema, requirement);
        let harness = fixture_for_shells(std::slice::from_ref(&shell));
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        let spec = shell.spec_string();
        let error =
            lib_install_bootstrap(&[&spec], &harness.uri(), Some("unit-test-token"), &registry)
                .expect_err("malformed requirement must fail before peer selection");
        assert!(
            matches!(error, chelis_reef::BootstrapError::ManifestRead { .. }),
            "{error}"
        );
        assert!(error.to_string().contains(requirement), "{error}");
        assert!(harness.install_order().is_empty());
        assert!(!registry.exists());
    }
}

#[test]
fn bootstrap_typed_peer_edges_preserve_cycle_rejection() {
    let _g = file_lock();
    let shell = peer_requirement_fixture(2, "=0.2.0");
    let mut peer = SyntheticShell::new(
        "chelis-lang",
        "peer",
        "v0.2.0",
        "0.2.0",
        vec![("peer-user", "=0.2.0")],
    );
    peer.schema = 2;
    let harness = fixture_for_shells(&[shell.clone(), peer.clone()]);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("registry");
    let spec = shell.spec_string();
    let peer_spec = peer.spec_string();
    let error = lib_install_bootstrap(
        &[&spec, &peer_spec],
        &harness.uri(),
        Some("unit-test-token"),
        &registry,
    )
    .expect_err("matching requirements still form a cycle");
    assert!(
        matches!(error, chelis_reef::BootstrapError::Cycle { .. }),
        "{error}"
    );
    assert!(harness.install_order().is_empty());
    assert!(!registry.exists());
}

#[test]
fn bootstrap_core_preserves_conflict_and_requirement_precedence() {
    let _g = file_lock();
    let user = peer_requirement_fixture(2, "=0.9.0");
    let peer = SyntheticShell::new("chelis-lang", "peer", "v0.2.0", "0.2.0", vec![]);
    let duplicate = SyntheticShell::new("chelis-lang", "peer", "v0.3.0", "0.3.0", vec![]);
    let harness = fixture_for_shells(&[user.clone(), peer.clone(), duplicate.clone()]);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("registry");
    let inputs = [
        user.spec_string(),
        peer.spec_string(),
        duplicate.spec_string(),
    ];
    let references = inputs.iter().map(String::as_str).collect::<Vec<_>>();
    let error = lib_install_bootstrap(
        &references,
        &harness.uri(),
        Some("unit-test-token"),
        &registry,
    )
    .expect_err("version conflict precedes incompatible requirement");
    assert!(matches!(
        error,
        chelis_reef::BootstrapError::DuplicateVersion { .. }
    ));
    assert!(harness.install_order().is_empty());
    assert!(!registry.exists());

    let mut peer = peer;
    peer.schema = 2;
    peer.deps.push(("peer", "=0.2.0".to_string()));
    let harness = fixture_for_shells(&[user.clone(), peer.clone()]);
    let inputs = [user.spec_string(), peer.spec_string()];
    let references = inputs.iter().map(String::as_str).collect::<Vec<_>>();
    let error = lib_install_bootstrap(
        &references,
        &harness.uri(),
        Some("unit-test-token"),
        &registry,
    )
    .expect_err("incompatible requirement precedes cycle");
    assert_eq!(
        error.to_string(),
        "shell `peer-user` requires `peer` `=0.9.0`, but the bootstrap input list \
         selects `0.2.0`. Select a shell version that satisfies the requirement."
    );
    assert!(matches!(
        error,
        chelis_reef::BootstrapError::Validation { .. }
    ));
    assert!(harness.install_order().is_empty());
    assert!(!registry.exists());
}

// The default-mode contract is owned by reef_distribution.md Item 7.
fn default_cohort() -> Vec<SyntheticShell> {
    let mut shells = vec![
        SyntheticShell::new("chelis-lang", "nautilus", "v0.2.0", "0.2.0", vec![]),
        SyntheticShell::new(
            "chelis-lang",
            "coral",
            "v0.2.0",
            "0.2.0",
            vec![("nautilus", "=0.2.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "shoals",
            "v0.2.0",
            "0.2.0",
            vec![("coral", "=0.2.0"), ("bootstrap-bed", "^1")],
        ),
        SyntheticShell::new("chelis-lang", "bootstrap-bed", "v1.0.0", "1.0.0", vec![]),
    ];
    for shell in &mut shells {
        shell.schema = 3;
    }
    shells[0].deps.push((
        "chelis-std",
        format!("={}", chelis_std_bundle::EMBEDDED_RUNTIME.version()),
    ));
    shells
}

fn default_command(
    harness: &WiremockHarness,
    registry: &Path,
    cwd: &Path,
    authenticated: bool,
) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command
        .current_dir(cwd)
        .env("CHELIS_REEF_HOME", registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env_remove("GITHUB_TOKEN")
        .env_remove("CHELIS_REEF_ALLOW_DEP_COMPILER_DRIFT")
        .env("PATH", "")
        .args(["reef", "install", "--bootstrap"]);
    if authenticated {
        command.env("GITHUB_TOKEN", "unit-test-token");
    }
    command
}

fn assert_default_registry(registry: &Path) -> serde_json::Value {
    let index: serde_json::Value =
        serde_json::from_slice(&fs::read(registry.join("index.json")).expect("index"))
            .expect("JSON");
    let packages = index["packages"].as_object().expect("packages");
    assert_eq!(
        packages.len(),
        4,
        "only roots and selected closure belong in registry"
    );
    for (package, version) in [
        ("nautilus", "0.2.0"),
        ("coral", "0.2.0"),
        ("shoals", "0.2.0"),
        ("bootstrap-bed", "1.0.0"),
    ] {
        let entries = packages[package].as_array().expect("versions");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["version"], version);
        assert_eq!(entries[0]["compiler"], CURRENT_COMPILER_PIN);
        assert_eq!(
            entries[0]["remote_origin"],
            format!("github://chelis-lang/{package}@v{version}")
        );
        let shell = registry.join(format!(
            "packages/{package}/{version}/{package}-{version}.chb"
        ));
        let archive = registry.join(format!(
            "packages/{package}/{version}/{package}-{version}.tar.zst"
        ));
        let verified =
            chelis_reef::verify_artifact_pair(&archive, &shell).expect("authentic selected pair");
        assert_eq!(entries[0]["archive_sha256"], verified.archive_sha256);
        assert_eq!(entries[0]["shell_sha256"], verified.shell_sha256);
    }
    assert!(!packages.contains_key("chelis-std"));
    index
}

#[test]
fn bootstrap_defaults_resolve_joint_graph_without_project_writes() {
    let _g = file_lock();
    oracle_default_compiler_graph();
}

fn oracle_default_compiler_graph() {
    let mut shells = default_cohort();
    let mut newer = SyntheticShell::new("chelis-lang", "nautilus", "v0.3.0", "0.3.0", vec![]);
    newer.schema = 3;
    let mut drift = SyntheticShell::new("chelis-lang", "nautilus", "v0.4.0", "0.4.0", vec![]);
    drift.schema = 3;
    drift.compiler = "=0.0.1";
    shells.extend([newer, drift]);
    let harness = fixture_for_shells(&shells);
    let directory = tempdir().expect("tempdir");
    let registry = directory.path().join("registry");
    let output = default_command(&harness, &registry, directory.path(), true)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).expect("UTF-8");
    assert!(
        stdout.find("Installed nautilus 0.2.0").expect("nautilus")
            < stdout.find("Installed coral 0.2.0").expect("coral")
    );
    assert!(
        stdout.find("Installed coral 0.2.0").expect("coral")
            < stdout.find("Installed shoals 0.2.0").expect("shoals")
    );
    assert!(
        stdout.find("Installed bootstrap-bed 1.0.0").expect("bed")
            < stdout.find("Installed shoals 0.2.0").expect("shoals")
    );
    assert_default_registry(&registry);
    for path in ["reef.toml", "reef.lock", ".reef-write.lock", ".gitignore"] {
        assert!(
            !directory.path().join(path).exists(),
            "hidden project write: {path}"
        );
    }
}

#[test]
fn bootstrap_defaults_same_binary_accepts_later_published_cohort() {
    let _g = file_lock();
    let harness = WiremockHarness::new();
    let mut absent = default_cohort();
    for shell in &mut absent {
        shell.compiler = "=0.0.1";
    }
    mount_shells(&harness, &absent, true);
    let directory = tempdir().expect("tempdir");
    let registry = directory.path().join("registry");
    default_command(&harness, &registry, directory.path(), true)
        .env("CHELIS_REEF_ALLOW_DEP_COMPILER_DRIFT", "1")
        .assert()
        .failure()
        .stderr(predicate::str::contains("default bootstrap"))
        .stderr(predicate::str::contains(env!("CARGO_PKG_VERSION")));
    assert!(!registry.join("index.json").exists());
    harness.reset();
    mount_shells(&harness, &default_cohort(), true);
    default_command(&harness, &registry, directory.path(), true)
        .assert()
        .success();
    assert_default_registry(&registry);
}

#[test]
fn bootstrap_defaults_permutation_and_repeat_preserve_registry_and_output() {
    let _g = file_lock();
    let directory = tempdir().expect("tempdir");
    let mut results = Vec::new();
    let mut shells = default_cohort();
    let mut newer = SyntheticShell::new("chelis-lang", "nautilus", "v0.3.0", "0.3.0", vec![]);
    newer.schema = 3;
    shells.push(newer);
    for index in 0..2 {
        if index == 1 {
            shells.reverse();
        }
        let harness = fixture_for_shells(&shells);
        let registry = directory.path().join(format!("registry-{index}"));
        let first = default_command(&harness, &registry, directory.path(), true)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let state = fs::read(registry.join("index.json")).expect("index");
        let repeated = default_command(&harness, &registry, directory.path(), true)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        assert_eq!(
            state,
            fs::read(registry.join("index.json")).expect("repeat index")
        );
        assert_eq!(first, repeated);
        let report = String::from_utf8(first)
            .expect("UTF-8")
            .lines()
            .filter(|line| line.starts_with("Installed "))
            .map(str::to_string)
            .collect::<Vec<_>>();
        results.push((state, report));
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn bootstrap_defaults_use_canonical_remote_candidates_not_cached_fork() {
    let _g = file_lock();
    let mut shells = default_cohort();
    shells[1].deps.clear();
    let mut fork = SyntheticShell::new("acme", "nautilus", "v0.9.0", "0.9.0", vec![]);
    fork.schema = 3;
    shells.push(fork);
    let harness = fixture_for_shells(&shells);
    let directory = tempdir().expect("tempdir");
    let registry = directory.path().join("registry");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "install", "--from-github", "acme/nautilus@v0.9.0"])
        .assert()
        .success();
    let prior_requests = harness.requests().len();
    default_command(&harness, &registry, directory.path(), true)
        .assert()
        .success()
        .stdout(predicate::str::contains("Installed nautilus 0.2.0"));
    assert!(
        harness
            .requests()
            .iter()
            .skip(prior_requests)
            .all(|request| request.url.path().starts_with("/repos/chelis-lang/"))
    );
    let index: serde_json::Value =
        serde_json::from_slice(&fs::read(registry.join("index.json")).expect("index"))
            .expect("JSON");
    assert!(
        index["packages"]["nautilus"]
            .as_array()
            .expect("versions")
            .iter()
            .any(|entry| entry["version"] == "0.9.0"
                && entry["remote_origin"] == "github://acme/nautilus@v0.9.0")
    );
}

#[test]
fn bootstrap_defaults_public_without_token_or_gh() {
    let _g = file_lock();
    let harness = WiremockHarness::new();
    mount_shells(&harness, &default_cohort(), false);
    let directory = tempdir().expect("tempdir");
    let registry = directory.path().join("registry");
    default_command(&harness, &registry, directory.path(), false)
        .assert()
        .success();
    assert_default_registry(&registry);
    let requests = harness.requests();
    assert!(!requests.is_empty());
    assert!(
        requests
            .iter()
            .all(|request| request.headers.get("authorization").is_none())
    );
}

#[test]
fn bootstrap_defaults_private_errors_remain_actionable() {
    let _g = file_lock();
    for status in [401, 403] {
        let harness = WiremockHarness::new();
        harness.mount_all(vec![
            Mock::given(method("GET")).respond_with(ResponseTemplate::new(status)),
        ]);
        let directory = tempdir().expect("tempdir");
        let registry = directory.path().join("registry");
        default_command(&harness, &registry, directory.path(), false)
            .assert()
            .failure()
            .stderr(predicate::str::contains("GITHUB_TOKEN"))
            .stderr(predicate::str::contains(status.to_string()));
        assert!(!registry.join("index.json").exists());
    }
}

#[test]
fn bootstrap_defaults_reject_invalid_runtime_missing_and_cyclic_graphs() {
    let _g = file_lock();
    for case in ["invalid", "runtime", "missing", "cycle"] {
        let mut shells = default_cohort();
        match case {
            "invalid" => shells[0].deps = vec![("chelis-std", "not-a-version".to_string())],
            "runtime" => shells[0].deps = vec![("chelis-std", "=99.0.0".to_string())],
            "missing" => {
                shells.pop();
            }
            "cycle" => shells[0].deps.push(("coral", "=0.2.0".to_string())),
            _ => unreachable!(),
        }
        let harness = fixture_for_shells(&shells);
        let directory = tempdir().expect("tempdir");
        let registry = directory.path().join("registry");
        let assertion = default_command(&harness, &registry, directory.path(), true)
            .assert()
            .failure();
        if case == "cycle" {
            assertion.stderr(predicate::str::contains("cycle"));
        } else {
            assertion.stderr(predicate::str::contains("default bootstrap"));
        }
        assert!(
            !registry.join("index.json").exists(),
            "planning failure mutated registry: {case}"
        );
        assert!(harness.requests().iter().all(|request| {
            !request
                .url
                .path()
                .starts_with("/repos/chelis-lang/chelis-std/")
        }));
    }
}

#[test]
fn bootstrap_defaults_verify_selected_shells_before_registry_publication() {
    let _g = file_lock();
    let shells = default_cohort();
    let harness = fixture_for_shells(&shells);
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(format!(
                "/repos/chelis-lang/nautilus/releases/assets/{}",
                shell_asset_id(0)
            )))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"corrupt selected CHB"))
            .with_priority(1),
    ]);
    let directory = tempdir().expect("tempdir");
    let registry = directory.path().join("registry");
    default_command(&harness, &registry, directory.path(), true)
        .assert()
        .failure()
        .stderr(predicate::str::contains("default bootstrap"));
    assert!(!registry.join("index.json").exists());
    assert!(!registry.join("packages").exists());
}

#[test]
fn bootstrap_defaults_rebind_identical_cached_bytes_to_observed_canonical_origin() {
    let _g = file_lock();
    let harness = WiremockHarness::new();
    let mut fork = default_cohort();
    fork[0].org = "acme";
    mount_shells(&harness, &fork, true);
    let directory = tempdir().expect("tempdir");
    let registry = directory.path().join("registry");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "install", "--from-github", "acme/nautilus@v0.2.0"])
        .assert()
        .success();
    let before: serde_json::Value =
        serde_json::from_slice(&fs::read(registry.join("index.json")).expect("prior index"))
            .expect("prior index JSON");
    assert_eq!(
        before["packages"]["nautilus"][0]["remote_origin"],
        "github://acme/nautilus@v0.2.0"
    );
    harness.reset();
    mount_shells(&harness, &default_cohort(), true);
    default_command(&harness, &registry, directory.path(), true)
        .assert()
        .success();
    let after: serde_json::Value =
        serde_json::from_slice(&fs::read(registry.join("index.json")).expect("canonical index"))
            .expect("canonical index JSON");
    assert_eq!(
        after["packages"]["nautilus"][0]["archive_sha256"],
        before["packages"]["nautilus"][0]["archive_sha256"]
    );
    assert_eq!(
        after["packages"]["nautilus"][0]["shell_sha256"],
        before["packages"]["nautilus"][0]["shell_sha256"]
    );
    assert_default_registry(&registry);
    assert!(
        harness
            .requests()
            .iter()
            .all(|request| !request.url.path().contains("/acme/"))
    );
}

#[test]
fn bootstrap_defaults_existing_identity_conflicts_preserve_registry_metadata() {
    let _g = file_lock();
    let harness = WiremockHarness::new();
    let mut prior = default_cohort();
    prior[0].schema = 2;
    mount_shells(&harness, &prior, true);
    let directory = tempdir().expect("tempdir");
    let registry = directory.path().join("registry");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args([
            "reef",
            "install",
            "--from-github",
            "chelis-lang/nautilus@v0.2.0",
        ])
        .assert()
        .success();
    let before = fs::read(registry.join("index.json")).expect("prior index");
    harness.reset();
    mount_shells(&harness, &default_cohort(), true);
    default_command(&harness, &registry, directory.path(), true)
        .assert()
        .failure()
        .stderr(predicate::str::contains("differs by bytes"));
    assert_eq!(
        before,
        fs::read(registry.join("index.json")).expect("index after conflict")
    );
}

#[test]
fn bootstrap_defaults_keep_the_shared_release_tag_budget() {
    use wiremock::matchers::query_param;
    let _g = file_lock();
    let harness = fixture_for_shells(&default_cohort());
    for page in 1..=3 {
        let count = if page == 3 { 57 } else { 100 };
        let rows = (0..count).map(|offset| { let version = format!("1.0.{}", (page - 1) * 100 + offset); serde_json::json!({"tag_name":format!("v{version}"),"draft":false,"prerelease":false,"assets":[{"id":10,"name":format!("nautilus-{version}.tar.zst")},{"id":11,"name":format!("nautilus-{version}.chb")}]}) }).collect::<Vec<_>>();
        harness.mount_all(vec![
            Mock::given(method("GET"))
                .and(wm_path("/repos/chelis-lang/nautilus/releases"))
                .and(query_param("page", page.to_string()))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(rows)
                        .insert_header("link", "<http://localhost/next>; rel=\"next\""),
                )
                .with_priority(1),
        ]);
    }
    let directory = tempdir().expect("tempdir");
    let registry = directory.path().join("registry");
    default_command(&harness, &registry, directory.path(), true)
        .assert()
        .failure()
        .stderr(predicate::str::contains("accepted release tags"));
    assert!(!registry.join("index.json").exists());
}

#[test]
fn bootstrap_explicit_selectors_do_not_discover_a_missing_peer() {
    let _g = file_lock();
    let harness = fixture_for_shells(&default_cohort());
    let directory = tempdir().expect("tempdir");
    let registry = directory.path().join("registry");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "install", "--bootstrap", "chelis-lang/coral@v0.2.0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not in the bootstrap input list"));
    assert!(
        harness
            .requests()
            .iter()
            .all(|request| !request.url.path().ends_with("/releases"))
    );
    assert!(!registry.join("index.json").exists());
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

/// Linear chain a -> b -> c: install each package in any input order.
/// Expected install order: c, b, a (dependencies before dependents).
fn oracle_linear_chain_topo_order() {
    let shells = vec![
        SyntheticShell::new("chelis-lang", "a", "v0.2.0", "0.2.0", vec![("b", "0.2.0")]),
        SyntheticShell::new("chelis-lang", "b", "v0.2.0", "0.2.0", vec![("c", "0.2.0")]),
        SyntheticShell::new("chelis-lang", "c", "v0.2.0", "0.2.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    // Submit in input order [a, b, c] — the topo sort must reorder.
    let input_strings: Vec<String> = shells.iter().map(|s| s.spec_string()).collect();
    let inputs: Vec<&str> = input_strings.iter().map(|s| s.as_str()).collect();
    let installed =
        lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
            .expect("linear-chain bootstrap");
    let order = harness.install_order();
    assert_eq!(
        order,
        vec!["c", "b", "a"],
        "linear chain must install dependencies before dependents"
    );
    assert_eq!(installed.len(), 3);
    // Registry-state sanity check.
    for repo in &["a", "b", "c"] {
        assert!(
            registry
                .join(format!("packages/{repo}/0.2.0/{repo}-0.2.0.chb"))
                .exists(),
            "{repo} shell missing from registry"
        );
        assert!(
            registry
                .join(format!("packages/{repo}/0.2.0/{repo}-0.2.0.tar.zst"))
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
            "diamond-a",
            "v0.2.0",
            "0.2.0",
            vec![("diamond-b", "0.2.0"), ("diamond-c", "0.2.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "diamond-b",
            "v0.2.0",
            "0.2.0",
            vec![("diamond-d", "0.2.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "diamond-c",
            "v0.2.0",
            "0.2.0",
            vec![("diamond-d", "0.2.0")],
        ),
        SyntheticShell::new("chelis-lang", "diamond-d", "v0.2.0", "0.2.0", vec![]),
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
        pos("diamond-d") < pos("diamond-b"),
        "diamond-d must be installed before diamond-b, got: {order:?}"
    );
    assert!(
        pos("diamond-d") < pos("diamond-c"),
        "diamond-d must be installed before diamond-c, got: {order:?}"
    );
    assert!(
        pos("diamond-b") < pos("diamond-a"),
        "diamond-b must be installed before diamond-a, got: {order:?}"
    );
    assert!(
        pos("diamond-c") < pos("diamond-a"),
        "diamond-c must be installed before diamond-a, got: {order:?}"
    );
}

/// Two-shell cycle: A -> B -> A. Must surface BootstrapError::Cycle.
fn oracle_cycle_named() {
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "cycle-a",
            "v0.2.0",
            "0.2.0",
            vec![("cycle-b", "0.2.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "cycle-b",
            "v0.2.0",
            "0.2.0",
            vec![("cycle-a", "0.2.0")],
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
            // Cycle members must be named.
            let joined = cycle.join(",");
            assert!(
                joined.contains("cycle-a"),
                "cycle must name cycle-a: {joined}"
            );
            assert!(
                joined.contains("cycle-b"),
                "cycle must name cycle-b: {joined}"
            );
        }
        other => panic!("expected BootstrapError::Cycle, got: {other:?}"),
    }
    assert!(
        msg.contains("cycle-a") && msg.contains("cycle-b"),
        "Display must name all cycle members: {msg}"
    );
    // Registry stays untouched on cycle (we never reached the install loop).
    assert!(!registry.join("index.json").exists());
}

/// Missing dep: A depends on B, but B is not in the input set.
fn oracle_missing_dep_named() {
    let shells = vec![SyntheticShell::new(
        "chelis-lang",
        "missing-a",
        "v0.2.0",
        "0.2.0",
        vec![("missing-b", "0.2.0")],
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
            assert_eq!(dependent, "missing-a");
            assert_eq!(missing, "missing-b");
        }
        other => panic!("expected MissingDependency, got: {other:?}"),
    }
    assert!(
        msg.contains("missing-a") && msg.contains("missing-b"),
        "Display must name both dependent and missing dep: {msg}"
    );
}

/// Default-list path:
///
/// 1. Explicit list `<org>/foo@v0.2.0 <org>/bar@v0.2.0` against
///    wiremock — installs both in correct order.
/// 2. Empty default list parses (we cannot exercise it against real
///    network in this oracle; the manual gate covers that).
fn oracle_default_list_path() {
    // Sub-case 1: explicit list with two shells.
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "foo",
            "v0.2.0",
            "0.2.0",
            vec![("bar", "0.2.0")],
        ),
        SyntheticShell::new("chelis-lang", "bar", "v0.2.0", "0.2.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let inputs = vec!["chelis-lang/foo@v0.2.0", "chelis-lang/bar@v0.2.0"];
    lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect("explicit two-shell bootstrap");
    let order = harness.install_order();
    assert_eq!(order, vec!["bar", "foo"]);

    // Default root identities are versionless; execute the actual no-selector
    // CLI against a compatible multi-candidate graph rather than parsing tags.
    for root in chelis_reef::DEFAULT_BOOTSTRAP_ROOTS {
        root.parse::<chelis_reef::package_versioning::PackageName>()
            .expect("canonical default root");
    }
    oracle_default_compiler_graph();
}

/// Per-shell atomicity: shell 2 of 3 fails validation; shell 1 fully
/// installed, shell 2 leaves no orphans, shell 3 not attempted.
fn oracle_per_shell_atomicity_preserved() {
    // Build three shells. atomic-b advertises a corrupt shell payload
    // (the shell's archive_sha256 disagrees with the on-wire archive
    // bytes). The bootstrap topo order will be atomic-c, atomic-b,
    // atomic-a. The first package installs, the second fails, and the
    // third never runs.
    let pa = SyntheticShell::new(
        "chelis-lang",
        "atomic-a",
        "v0.2.0",
        "0.2.0",
        vec![("atomic-b", "0.2.0")],
    );
    let pb = SyntheticShell::new(
        "chelis-lang",
        "atomic-b",
        "v0.2.0",
        "0.2.0",
        vec![("atomic-c", "0.2.0")],
    );
    let pc = SyntheticShell::new("chelis-lang", "atomic-c", "v0.2.0", "0.2.0", vec![]);

    // Stand up wiremock manually so PB can be served with a tampered
    // shell (archive_sha256 set to a wrong value).
    let harness = WiremockHarness::new();
    let mut mocks: Vec<Mock> = Vec::new();
    for (idx, shell) in [&pa, &pb, &pc].iter().enumerate() {
        let archive_id = archive_asset_id(idx);
        let shell_id = shell_asset_id(idx);
        let archive_bytes = shell.archive_bytes();
        // Tamper atomic-b's shell with a different archive hash.
        // than the on-wire archive. Validation step catches this.
        let shell_bytes = if shell.repo == "atomic-b" {
            let bogus_sha = "0".repeat(64);
            build_test_shell_bytes(shell.repo, shell.version, &bogus_sha, shell.compiler)
        } else {
            build_test_shell_bytes(
                shell.repo,
                shell.version,
                &sha256_bytes(&archive_bytes),
                shell.compiler,
            )
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
    .expect_err("atomic-b validation must fail");

    // Error must be wrapped Validation/Fetch carrying the
    // archive_sha256 disagreement message.
    match &err {
        chelis_reef::BootstrapError::Fetch(chelis_reef::GitHubFetchError::Validation {
            ..
        }) => {}
        chelis_reef::BootstrapError::Validation { .. } => {}
        other => panic!("expected Validation-class error, got: {other:?}"),
    }

    // atomic-c was installed before atomic-b tried.
    assert!(
        registry
            .join("packages/atomic-c/0.2.0/atomic-c-0.2.0.chb")
            .exists(),
        "atomic-c must be installed before atomic-b validation fired"
    );
    assert!(
        registry
            .join("packages/atomic-c/0.2.0/atomic-c-0.2.0.tar.zst")
            .exists()
    );

    // PB's shell-bytes endpoint was hit (recorded), but the install
    // step failed; the package directory MAY exist with copied bytes
    // (validation runs after copy in install_validated_artifact_pair).
    // What must NOT exist is an index.json entry for PB.
    let index_text = fs::read_to_string(registry.join("index.json")).expect("index.json present");
    let index: serde_json::Value = serde_json::from_str(&index_text).unwrap();
    assert!(
        index["packages"]["atomic-c"].is_array(),
        "index must have atomic-c after a successful first install"
    );
    assert!(
        index["packages"]["atomic-b"].is_null() || !index["packages"]["atomic-b"].is_array(),
        "index must not have atomic-b after validation failure: {index}"
    );
    assert!(
        index["packages"]["atomic-a"].is_null() || !index["packages"]["atomic-a"].is_array(),
        "atomic-a must not be in the index because it was not attempted: {index}"
    );

    // Order recording: PC happened, PB happened (it reaches the
    // shell-bytes fetch endpoint before validation fires), PA never.
    let order = harness.install_order();
    assert!(
        order.contains(&"atomic-c".to_string()),
        "atomic-c was attempted: {order:?}"
    );
    assert!(
        !order.contains(&"atomic-a".to_string()),
        "atomic-a must not be attempted after atomic-b fails: {order:?}"
    );
}

/// CLI surface: `chelis reef install --bootstrap a b` parses both and
/// dispatches to `install_bootstrap`. End-to-end via the binary.
fn oracle_cli_surface_dispatches_to_install_bootstrap() {
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "cli-foo",
            "v0.2.0",
            "0.2.0",
            vec![("cli-bar", "0.2.0")],
        ),
        SyntheticShell::new("chelis-lang", "cli-bar", "v0.2.0", "0.2.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("cli-reef-home");

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .args([
            "reef",
            "install",
            "--bootstrap",
            "chelis-lang/cli-foo@v0.2.0",
            "chelis-lang/cli-bar@v0.2.0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Installed cli-foo 0.2.0"))
        .stdout(predicate::str::contains("Installed cli-bar 0.2.0"));

    let order = harness.install_order();
    assert_eq!(order, vec!["cli-bar", "cli-foo"]);
}

/// CLI: `--bootstrap` is mutually exclusive with `--from-github` and
/// with `--from-monorepo`.
fn oracle_cli_mutex_with_from_github_and_from_monorepo() {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "reef",
            "install",
            "--bootstrap",
            "chelis-lang/x@v0.2.0",
            "--from-github",
            "chelis-lang/y@v0.2.0",
        ])
        .assert()
        .failure();

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "reef",
            "install",
            "--bootstrap",
            "chelis-lang/x@v0.2.0",
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
            "t3a",
            "v0.2.0",
            "0.2.0",
            vec![("t3b", "0.2.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "t3b",
            "v0.2.0",
            "0.2.0",
            vec![("t3c", "0.2.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "t3c",
            "v0.2.0",
            "0.2.0",
            vec![("t3a", "0.2.0")],
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
            for needle in ["t3a", "t3b", "t3c"] {
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
        "v0.2.0",
        "0.2.0",
        vec![("self-dep", "0.2.0")],
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
    let shell_bytes = build_test_shell_bytes(
        "bad",
        "0.2.0",
        &sha256_bytes(&archive_bytes),
        CURRENT_COMPILER_PIN,
    );

    let harness = WiremockHarness::new();
    let metadata_body = serde_json::json!({
        "id": 1u64,
        "tag_name": "v0.2.0",
        "assets": [
            {"id": 100u64, "name": "bad-0.2.0.tar.zst",
             "size": archive_bytes.len() as u64},
            {"id": 101u64, "name": "bad-0.2.0.chb",
             "size": shell_bytes.len() as u64},
        ]
    })
    .to_string();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path("/repos/chelis-lang/bad/releases/tags/v0.2.0"))
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
        &["chelis-lang/bad@v0.2.0"],
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
    let shell_v1 = SyntheticShell::new("chelis-lang", "dup", "v0.2.0", "0.2.0", vec![]);
    let shell_v2 = SyntheticShell::new("chelis-lang", "dup", "v0.3.0", "0.3.0", vec![]);
    let harness = fixture_for_shells(&[shell_v1.clone(), shell_v2.clone()]);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let inputs = ["chelis-lang/dup@v0.2.0", "chelis-lang/dup@v0.3.0"];
    let err = lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect_err("duplicate versions must fail");
    let msg = err.to_string();
    match err {
        chelis_reef::BootstrapError::DuplicateVersion { package, versions } => {
            assert_eq!(package, "dup");
            assert!(versions.contains(&"0.2.0".to_string()));
            assert!(versions.contains(&"0.3.0".to_string()));
        }
        other => panic!("expected DuplicateVersion, got: {other:?}"),
    }
    assert!(
        msg.contains("0.2.0") && msg.contains("0.3.0"),
        "Display must list both versions: {msg}"
    );
}

#[test]
fn phaseA_item7_empty_input_and_empty_default_errors_nothing_to_install() {
    let _g = file_lock();
    // Library-level: install_bootstrap with empty specs must surface
    // BootstrapError::NothingToInstall. The CLI uses the separate
    // default-discovery entry point when no selectors are supplied.
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let err = chelis_reef::install_bootstrap(&[], &registry, &chelis_std_bundle::EMBEDDED_RUNTIME)
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
    // PATH inside the helper). The private server rejects the first
    // anonymous manifest fetch; bootstrap must not try the install loop.
    let shells = vec![
        SyntheticShell::new(
            "chelis-lang",
            "auth-a",
            "v0.2.0",
            "0.2.0",
            vec![("auth-b", "0.2.0")],
        ),
        SyntheticShell::new("chelis-lang", "auth-b", "v0.2.0", "0.2.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path("/repos/chelis-lang/auth-a/releases/tags/v0.2.0"))
            .respond_with(ResponseTemplate::new(401)),
    ]);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let input_strings: Vec<String> = shells.iter().map(|s| s.spec_string()).collect();
    let inputs: Vec<&str> = input_strings.iter().map(|s| s.as_str()).collect();
    let err = lib_install_bootstrap(&inputs, &harness.uri(), None, &registry)
        .expect_err("private release must reject an anonymous request");
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
    let requests = harness
        .rt
        .block_on(harness.server.received_requests())
        .unwrap();
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].headers.contains_key("authorization"));
}

#[test]
fn phaseA_item7_public_bootstrap_without_token() {
    let _g = file_lock();
    let shell = SyntheticShell::new("chelis-lang", "public-shell", "v0.2.0", "0.2.0", vec![]);
    let archive_id = archive_asset_id(0);
    let shell_id = shell_asset_id(0);
    let metadata = serde_json::json!({
        "assets": [
            {"id": archive_id, "name": shell.archive_name()},
            {"id": shell_id, "name": shell.shell_name()},
        ]
    })
    .to_string();
    let harness = WiremockHarness::new();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(
                "/repos/chelis-lang/public-shell/releases/tags/v0.2.0",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_string(metadata)),
        Mock::given(method("GET"))
            .and(wm_path(format!(
                "/repos/chelis-lang/public-shell/releases/assets/{archive_id}"
            )))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(shell.archive_bytes())),
        Mock::given(method("GET"))
            .and(wm_path(format!(
                "/repos/chelis-lang/public-shell/releases/assets/{shell_id}"
            )))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(shell.shell_bytes())),
    ]);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let spec = shell.spec_string();
    lib_install_bootstrap(&[&spec], &harness.uri(), None, &registry)
        .expect("public bootstrap must install without gh or GITHUB_TOKEN");
    assert!(
        registry
            .join("packages/public-shell/0.2.0/public-shell-0.2.0.chb")
            .is_file()
    );
    let requests = harness
        .rt
        .block_on(harness.server.received_requests())
        .unwrap();
    assert_eq!(requests.len(), 5);
    assert!(
        requests
            .iter()
            .all(|request| !request.headers.contains_key("authorization"))
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
            "idem-a",
            "v0.2.0",
            "0.2.0",
            vec![("idem-b", "0.2.0")],
        ),
        SyntheticShell::new("chelis-lang", "idem-b", "v0.2.0", "0.2.0", vec![]),
    ];
    let harness = fixture_for_shells(&shells);
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let inputs = ["chelis-lang/idem-a@v0.2.0", "chelis-lang/idem-b@v0.2.0"];

    // First run.
    lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect("first run");
    let first_archive =
        fs::read(registry.join("packages/idem-a/0.2.0/idem-a-0.2.0.tar.zst")).unwrap();
    let first_shell = fs::read(registry.join("packages/idem-a/0.2.0/idem-a-0.2.0.chb")).unwrap();
    let first_index = fs::read_to_string(registry.join("index.json")).unwrap();

    // Second run against the same registry.
    lib_install_bootstrap(&inputs, &harness.uri(), Some("unit-test-token"), &registry)
        .expect("second run");
    let second_archive =
        fs::read(registry.join("packages/idem-a/0.2.0/idem-a-0.2.0.tar.zst")).unwrap();
    let second_shell = fs::read(registry.join("packages/idem-a/0.2.0/idem-a-0.2.0.chb")).unwrap();
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
            "order-a",
            "v0.2.0",
            "0.2.0",
            vec![("order-b", "0.2.0")],
        ),
        SyntheticShell::new(
            "chelis-lang",
            "order-b",
            "v0.2.0",
            "0.2.0",
            vec![("order-c", "0.2.0")],
        ),
        SyntheticShell::new("chelis-lang", "order-c", "v0.2.0", "0.2.0", vec![]),
    ];

    let permutations = [
        vec![
            "chelis-lang/order-a@v0.2.0",
            "chelis-lang/order-b@v0.2.0",
            "chelis-lang/order-c@v0.2.0",
        ],
        vec![
            "chelis-lang/order-c@v0.2.0",
            "chelis-lang/order-b@v0.2.0",
            "chelis-lang/order-a@v0.2.0",
        ],
        vec![
            "chelis-lang/order-b@v0.2.0",
            "chelis-lang/order-a@v0.2.0",
            "chelis-lang/order-c@v0.2.0",
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
            &vec![
                "order-c".to_string(),
                "order-b".to_string(),
                "order-a".to_string(),
            ],
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["reef", "install", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--bootstrap"))
        .stdout(predicate::str::contains("complete dependency closure"))
        .stdout(predicate::str::contains("Explicit"))
        .stdout(predicate::str::contains("closed graph"))
        .stdout(predicate::str::contains("anonymously"));
}

// ============================================================
// Manual gate (real network). `#[ignore]`d. Documented invocation:
// `cargo nextest run -p chelis-cli --test reef_install_bootstrap --run-ignored only
// -E 'test(=phaseA_item7_real_bootstrap_manual_gate)'`.
// A compatible published default-root dependency graph must exist.
// Public releases work anonymously; optional GITHUB_TOKEN/gh auth avoids
// the anonymous API rate limit.
// This ecosystem availability gate does not require shell publication before
// the compiler release. Offline fixtures own the recurrence regression.
// ============================================================

#[test]
#[ignore = "real-network manual gate; run with `--ignored --exact`"]
fn phaseA_item7_real_bootstrap_manual_gate() {
    let _g = file_lock();
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_REEF_HOME", &registry)
        .args(["reef", "install", "--bootstrap"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let index: serde_json::Value =
        serde_json::from_slice(&fs::read(registry.join("index.json")).expect("registry index"))
            .expect("index JSON");
    let mut dependencies = String::new();
    for root in chelis_reef::DEFAULT_BOOTSTRAP_ROOTS {
        let version = index["packages"][root][0]["version"]
            .as_str()
            .expect("root version");
        dependencies.push_str(&format!("{root} = \"={version}\"\n"));
    }
    for (name, versions) in index["packages"].as_object().expect("packages") {
        let entries = versions.as_array().expect("versions");
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        let version = entry["version"].as_str().expect("version");
        assert_eq!(entry["compiler"], CURRENT_COMPILER_PIN);
        let verified = chelis_reef::verify_artifact_pair(
            &registry.join(format!(
                "packages/{name}/{version}/{name}-{version}.tar.zst"
            )),
            &registry.join(format!("packages/{name}/{version}/{name}-{version}.chb")),
        )
        .expect("published artifact pair");
        assert_eq!(entry["archive_sha256"], verified.archive_sha256);
        assert_eq!(entry["shell_sha256"], verified.shell_sha256);
        let origin = entry["remote_origin"].as_str().expect("canonical origin");
        assert!(origin.starts_with(&format!(
            "github://{}/{name}@",
            chelis_reef::CANONICAL_REEF_ORG
        )));
        println!(
            "{name}@{version}: compiler={} origin={origin} archive={} shell={}",
            entry["compiler"], verified.archive_sha256, verified.shell_sha256
        );
    }
    let project = dir.path().join("verify-closure");
    fs::create_dir(&project).expect("verification project");
    fs::write(project.join("reef.toml"), format!("schema = \"3\"\n[package]\nname = \"bootstrap-verification\"\nversion = \"0.0.0\"\ncompiler = \"{CURRENT_COMPILER_PIN}\"\nmodule_prefix = \"Verify\"\nresolver = \"2\"\n[dependencies]\n{dependencies}")).expect("verification manifest");
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&project)
        .env("CHELIS_REEF_HOME", &registry)
        .args(["reef", "update", "--offline"])
        .assert()
        .success();
    let stdout = String::from_utf8(output).expect("UTF-8");
    print!("{stdout}");
    for root in chelis_reef::DEFAULT_BOOTSTRAP_ROOTS {
        assert!(stdout.contains(&format!("Installed {root} ")));
    }
}
