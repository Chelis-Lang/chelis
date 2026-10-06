//! Item 11 (chelis#468) acceptance: `chelis reef install --from-lockfile`
//! for a binary artifact, plus `chelis reef which`.
//!
//! These tests exercise the CLI install path against a localhost
//! `wiremock` fixture (the `CHELIS_REEF_GITHUB_BASE_API` seam injects the
//! mock host, same pattern as `reef_install_from_github.rs`). The fixture
//! serves a per-platform binary release asset (a `.tar.zst` carrying a
//! single binary) plus its release metadata. The lockfile carries a
//! `LockSource::Binary` entry; `--from-lockfile` resolves the host
//! platform, downloads the asset, SHA-256-verifies it fail-closed,
//! extracts the binary, and places it at `$CHELIS_HOME/bin/<name>`.
//!
//! Positive: the binary is placed, executable, byte-correct, and
//! `chelis reef which` prints its path.
//!
//! Negative parity:
//! - a lockfile whose pinned `sha256` does not match the served bytes
//!   aborts fail-closed with nothing placed under `$CHELIS_HOME/bin/`;
//! - `chelis reef which <name>` for an artifact that is not installed
//!   exits non-zero with an actionable message.

use assert_cmd::Command;
use predicates::prelude::*;
use sha2::{Digest, Sha256};
use std::io::Cursor;
use std::path::Path;
use tempfile::tempdir;
use wiremock::matchers::{header, method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const ARTIFACT_NAME: &str = "octant-translator";
const ARTIFACT_VERSION: &str = "1.0.0";
const ORG: &str = "test-org";
const REPO: &str = "test-repo";
const TAG: &str = "v1.0.0";
const ASSET_ID: u64 = 4242;
const MARKER: &[u8] = b"#!/bin/sh\necho octant-translator-marker\n";

/// The binary asset filename. The host slug is baked in so the fixture
/// resembles a real per-platform release asset.
fn asset_name(platform: &str) -> String {
    format!("{ARTIFACT_NAME}-{platform}.tar.zst")
}

fn sha256_bytes(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    format!("{:x}", h.finalize())
}

/// Build a `.tar.zst` archive carrying a single file named `inner_name`
/// with `payload` bytes. Returns the compressed archive bytes (what the
/// release would serve and what the lockfile pins by SHA-256).
fn build_tar_zst(inner_name: &str, payload: &[u8]) -> Vec<u8> {
    let mut tar_buf: Vec<u8> = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut tar_buf);
        let mut header = tar::Header::new_gnu();
        header.set_size(payload.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        builder
            .append_data(&mut header, inner_name, payload)
            .expect("append binary to tar");
        builder.finish().expect("finish tar");
    }
    zstd::stream::encode_all(Cursor::new(tar_buf), 0).expect("zstd encode")
}

fn metadata_path() -> String {
    format!("/repos/{ORG}/{REPO}/releases/tags/{TAG}")
}

fn asset_id_path() -> String {
    format!("/repos/{ORG}/{REPO}/releases/assets/{ASSET_ID}")
}

fn metadata_json(asset: &str) -> String {
    serde_json::json!({
        "id": 1u64,
        "tag_name": TAG,
        "assets": [
            { "id": ASSET_ID, "name": asset, "size": 0, "content_type": "application/octet-stream" },
        ],
    })
    .to_string()
}

/// Write a `reef.lock` into `package_root` carrying a single
/// `LockSource::Binary` dependency for `platform`, pinning `sha256`.
fn write_binary_lockfile(package_root: &Path, platform: &str, asset: &str, sha256: &str) {
    let lock = format!(
        r#"[package]
name = "consumer"
version = "0.1.0"

[[dependencies]]
name = "{ARTIFACT_NAME}"
version = "{ARTIFACT_VERSION}"
compiler = "=0.12.0"
archive_sha256 = "{sha256}"
shell_sha256 = ""

[dependencies.source]
kind = "binary"
remote_origin = "github://{ORG}/{REPO}@{TAG}"
platform = "{platform}"
asset = "{asset}"
sha256 = "{sha256}"
"#
    );
    std::fs::create_dir_all(package_root).expect("create package root");
    std::fs::write(package_root.join("reef.lock"), lock).expect("write reef.lock");
}

/// A wiremock harness owning a long-lived multi-threaded tokio runtime
/// plus the mock server. Mirrors `reef_install_from_github.rs`: setup runs
/// inside `mount_all`'s `block_on`; the test body is plain sync code that
/// drives the `chelis` CLI child process against `harness.uri()`.
struct WiremockHarness {
    rt: tokio::runtime::Runtime,
    server: MockServer,
}

impl WiremockHarness {
    fn new() -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("tokio multi-thread runtime");
        let server = rt.block_on(MockServer::start());
        Self { rt, server }
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
}

/// Serve the metadata + asset-bytes endpoints for a binary release asset,
/// both requiring `Authorization: token unit-test-token`.
fn mount_release(harness: &WiremockHarness, asset: &str, asset_bytes: Vec<u8>) {
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path()))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(metadata_json(asset))
                    .insert_header("content-type", "application/json"),
            ),
        Mock::given(method("GET"))
            .and(wm_path(asset_id_path()))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(asset_bytes)),
    ]);
}

/// Run `chelis reef install --from-lockfile` against the fixture, with
/// `CHELIS_HOME` / `CHELIS_REEF_HOME` isolated to tempdirs.
fn run_install(
    package_root: &Path,
    chelis_home: &Path,
    reef_home: &Path,
    api_base: &str,
) -> assert_cmd::assert::Assert {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_HOME", chelis_home)
        .env("CHELIS_REEF_HOME", reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", api_base)
        .env("GITHUB_TOKEN", "unit-test-token")
        // Drop PATH so the child cannot fall back to `gh auth token`.
        .env("PATH", "")
        .args([
            "reef",
            "install",
            "--from-lockfile",
            "--package-root",
            package_root.to_str().unwrap(),
        ])
        .assert()
}

#[test]
fn binary_install_from_lockfile_places_and_verifies() {
    let platform = match chelis_reef::host_platform_slug() {
        Some(p) => p,
        // No defined slug for this host (e.g. Windows): the install path
        // would have nothing to resolve, so skip rather than assert.
        None => return,
    };
    let asset = asset_name(platform);
    let asset_bytes = build_tar_zst(ARTIFACT_NAME, MARKER);
    let sha = sha256_bytes(&asset_bytes);

    let harness = WiremockHarness::new();
    mount_release(&harness, &asset, asset_bytes);

    let dir = tempdir().expect("tempdir");
    let package_root = dir.path().join("consumer");
    let chelis_home = dir.path().join("chelis-home");
    let reef_home = dir.path().join("reef-home");
    write_binary_lockfile(&package_root, platform, &asset, &sha);

    run_install(&package_root, &chelis_home, &reef_home, &harness.uri())
        .success()
        .stdout(predicate::str::contains(format!(
            "Installed binary {ARTIFACT_NAME} {ARTIFACT_VERSION}"
        )));

    let placed = chelis_home.join("bin").join(ARTIFACT_NAME);
    assert!(
        placed.exists(),
        "binary must be placed at {}",
        placed.display()
    );
    assert_eq!(
        std::fs::read(&placed).expect("read placed binary"),
        MARKER,
        "placed binary bytes must equal the archived payload"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&placed)
            .expect("stat placed binary")
            .permissions()
            .mode();
        assert_eq!(mode & 0o111, 0o111, "placed binary must be executable");
    }

    // `chelis reef which` prints the resolved path.
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_HOME", &chelis_home)
        .args(["reef", "which", ARTIFACT_NAME])
        .assert()
        .success()
        .stdout(predicate::str::contains(placed.to_str().unwrap()));
}

#[test]
fn public_binary_lockfile_installs_without_token() {
    let Some(platform) = chelis_reef::host_platform_slug() else {
        return;
    };
    let asset = asset_name(platform);
    let bytes = build_tar_zst(ARTIFACT_NAME, MARKER);
    let sha = sha256_bytes(&bytes);
    let harness = WiremockHarness::new();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path()))
            .respond_with(ResponseTemplate::new(200).set_body_string(metadata_json(&asset))),
        Mock::given(method("GET"))
            .and(wm_path(asset_id_path()))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes)),
    ]);
    let dir = tempdir().expect("tempdir");
    let package_root = dir.path().join("consumer");
    let chelis_home = dir.path().join("chelis-home");
    let reef_home = dir.path().join("reef-home");
    write_binary_lockfile(&package_root, platform, &asset, &sha);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_HOME", &chelis_home)
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env_remove("GITHUB_TOKEN")
        .env("PATH", "")
        .args([
            "reef",
            "install",
            "--from-lockfile",
            "--package-root",
            package_root.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert_eq!(
        std::fs::read(chelis_home.join("bin").join(ARTIFACT_NAME)).unwrap(),
        MARKER
    );
    let requests = harness
        .rt
        .block_on(harness.server.received_requests())
        .unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| !request.headers.contains_key("authorization"))
    );
}

#[test]
fn binary_install_sha_mismatch_aborts_fail_closed() {
    let platform = match chelis_reef::host_platform_slug() {
        Some(p) => p,
        None => return,
    };
    let asset = asset_name(platform);
    let asset_bytes = build_tar_zst(ARTIFACT_NAME, MARKER);
    // Pin a wrong SHA: the served bytes will not match, so the install
    // must abort before any placement.
    let wrong_sha = "0".repeat(64);
    assert_ne!(wrong_sha, sha256_bytes(&asset_bytes));

    let harness = WiremockHarness::new();
    mount_release(&harness, &asset, asset_bytes);

    let dir = tempdir().expect("tempdir");
    let package_root = dir.path().join("consumer");
    let chelis_home = dir.path().join("chelis-home");
    let reef_home = dir.path().join("reef-home");
    write_binary_lockfile(&package_root, platform, &asset, &wrong_sha);

    run_install(&package_root, &chelis_home, &reef_home, &harness.uri())
        .failure()
        .stderr(predicate::str::contains("SHA-256 mismatch"));

    // Fail-closed: nothing placed under $CHELIS_HOME/bin/.
    let placed = chelis_home.join("bin").join(ARTIFACT_NAME);
    assert!(
        !placed.exists(),
        "no binary must be placed on a SHA mismatch, found {}",
        placed.display()
    );
}

#[test]
fn which_unknown_artifact_fails() {
    let dir = tempdir().expect("tempdir");
    let chelis_home = dir.path().join("chelis-home");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_HOME", &chelis_home)
        .args(["reef", "which", "nonexistent-artifact"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not installed"));
}

#[test]
fn binary_install_skips_foreign_platform_entry() {
    // A lockfile written for a platform that is never the host: the entry
    // is skipped (informational), not installed, and not an error.
    let foreign = "totally-not-a-real-platform";
    let asset = asset_name(foreign);
    let sha = "a".repeat(64);

    // No release is mounted: a skipped foreign-platform entry must not
    // hit the network at all.
    let dir = tempdir().expect("tempdir");
    let package_root = dir.path().join("consumer");
    let chelis_home = dir.path().join("chelis-home");
    let reef_home = dir.path().join("reef-home");
    write_binary_lockfile(&package_root, foreign, &asset, &sha);

    run_install(
        &package_root,
        &chelis_home,
        &reef_home,
        "http://localhost:9",
    )
    .success()
    .stdout(predicate::str::contains("Skipped binary"));

    let placed = chelis_home.join("bin").join(ARTIFACT_NAME);
    assert!(
        !placed.exists(),
        "foreign-platform entry must not place a binary"
    );
}
