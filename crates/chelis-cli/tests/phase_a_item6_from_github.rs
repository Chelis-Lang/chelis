//! Phase A — Item 6 named acceptance oracle: `chelis reef install --from-github`.
//!
//! This file owns the named oracle test for Phase A Item 6:
//!
//! ```sh
//! cargo test -p chelis-cli phaseA_item6_from_github_oracle -- --exact
//! ```
//!
//! The oracle is a single comprehensive test function that exercises,
//! against a localhost `wiremock` fixture, every spec acceptance bullet
//! plus the locked negative-parity cases. The
//! `CHELIS_REEF_GITHUB_BASE_API` env var injects the localhost URL
//! into the fetch path; without that seam the test would have to hit
//! real GitHub.
//!
//! The oracle test lives next to the existing `phase3t_reef_install.rs`
//! `--from-monorepo` regression test so the contract-invariant
//! "source-equivalence" sub-case can re-use the same monorepo-built
//! `chelis-std-0.2.0.{chb,tar.zst}` artifacts that the existing test
//! treats as the byte-exact reference.
//!
//! This file also contains the manual-gate test
//! `phaseA_real_github_manual_gate`, which is `#[ignore]`d and only
//! runnable with `--ignored --exact` plus a real `GITHUB_TOKEN`. The
//! manual-gate test is what the orchestrator runs separately to
//! validate against the canonical `chelis-lang/nautilus` release
//! pinned by `MANUAL_GATE_NAUTILUS_TAG`. The pin must be bumped on
//! each new Nautilus release; see the constant's rustdoc.
//!
//! Negative parity (separate `#[test]` functions, run as part of the
//! default `cargo test --workspace` loop):
//!
//! - `partial_install_does_not_corrupt_index`: simulate a copy failure
//!   between archive and shell placement; assert the index file was
//!   not updated and no orphaned `index.json.tmp` is left behind.
//! - `wrong_asset_name_is_404`: a release that only carries a `.tgz`
//!   asset (the publisher contract names `.tar.zst`) surfaces a typed
//!   404 error; the helper does not silently fall back.
//! - `tag_without_leading_v_is_accepted_and_normalized`: spec lock —
//!   `chelis-lang/nautilus@0.4.0` and `chelis-lang/nautilus@v0.4.0`
//!   both install to `packages/nautilus/0.4.0/` (the `v` is decorative
//!   for the asset URL only).
//! - `parse_failure_messages_name_the_input`: missing `/`, missing
//!   `@`, empty components each surface a parse error that names the
//!   malformed input.
//! - `tempdir_is_removed_on_success_and_failure`: the fetch tempdir
//!   contract — `$TMPDIR` entry count is the same before and after
//!   any return path.

// The named-oracle convention `phaseA_item6_from_github_oracle` is
// locked by the plan and the brief; the orchestrator runs it via
// `cargo test -p chelis-cli phaseA_item6_from_github_oracle -- --exact`.
// Renaming to snake_case would silently break that oracle invocation.
#![allow(non_snake_case)]

use assert_cmd::Command;
use predicates::prelude::*;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;
use wiremock::matchers::{header, method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The chelis monorepo root (two levels up from `crates/chelis-cli`).
fn monorepo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("monorepo root must exist")
}

/// Real prebuilt artifacts shipped in `packages/chelis-std/dist/`. The
/// oracle uses these as the bytes the wiremock fixture serves, so the
/// validation step downstream of fetch sees a real chelis-std shell
/// agreeing with its real archive — exactly what the live GitHub
/// release would carry.
fn chelis_std_dist() -> (PathBuf, PathBuf) {
    let dist = monorepo_root().join("packages/chelis-std/dist");
    let archive = dist.join("chelis-std-0.2.0.tar.zst");
    let shell = dist.join("chelis-std-0.2.0.chb");
    assert!(
        archive.exists() && shell.exists(),
        "prebuilt chelis-std artifacts missing under {} — \
         run `chelis reef build` in packages/chelis-std/ first",
        dist.display()
    );
    (archive, shell)
}

fn sha256_bytes(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    format!("{:x}", h.finalize())
}

/// Wiremock path for the GitHub API release-metadata endpoint.
/// Form: `/repos/<org>/<repo>/releases/tags/<tag>`. Step 1 of the
/// two-step API fetch.
fn metadata_path(org: &str, repo: &str, tag: &str) -> String {
    format!("/repos/{org}/{repo}/releases/tags/{tag}")
}

/// Wiremock path for the GitHub API asset-by-id endpoint. Form:
/// `/repos/<org>/<repo>/releases/assets/<id>`. Step 2 of the two-step
/// API fetch — the byte-stream endpoint.
fn asset_id_path(org: &str, repo: &str, asset_id: u64) -> String {
    format!("/repos/{org}/{repo}/releases/assets/{asset_id}")
}

/// Build a minimal release-metadata JSON document with the given
/// asset list. Mirrors the shape of GitHub's real Release API
/// response — only the fields `parse_release_metadata` reads (`assets[]`,
/// each with `id`, `name`) need to be present, but we add a few extras
/// (`tag_name`, `id`) so the fixture more honestly resembles the live
/// payload.
fn metadata_json(tag: &str, assets: &[(u64, &str)]) -> String {
    let asset_entries: Vec<serde_json::Value> = assets
        .iter()
        .map(|(id, name)| {
            serde_json::json!({
                "id": id,
                "name": name,
                "size": 0,
                "content_type": "application/octet-stream",
            })
        })
        .collect();
    serde_json::json!({
        "id": 1u64,
        "tag_name": tag,
        "assets": asset_entries,
    })
    .to_string()
}

/// Stable asset ids used by the canonical fixture. Real GitHub asset
/// ids are 8+ digit numerics; using 1001/1002 makes the test's
/// `/releases/assets/<id>` URL look plausibly real while still being
/// trivial to reason about.
const ARCHIVE_ASSET_ID: u64 = 1001;
const SHELL_ASSET_ID: u64 = 1002;

/// Build the four mocks needed for a chelis-lang/chelis-std@v0.2.0
/// canonical-API install: metadata GET + archive bytes GET + shell
/// bytes GET, all requiring an `Authorization: token unit-test-token`
/// header. No 401 fallthrough — callers that want unauthenticated
/// behavior should use [`fixture_canonical_release`] (which adds
/// catch-all 401 mounts).
fn canonical_api_mocks(archive_bytes: Vec<u8>, shell_bytes: Vec<u8>) -> Vec<Mock> {
    let meta_path = metadata_path("chelis-lang", "chelis-std", "v0.2.0");
    let archive_url_path = asset_id_path("chelis-lang", "chelis-std", ARCHIVE_ASSET_ID);
    let shell_url_path = asset_id_path("chelis-lang", "chelis-std", SHELL_ASSET_ID);
    let metadata_body = metadata_json(
        "v0.2.0",
        &[
            (ARCHIVE_ASSET_ID, "chelis-std-0.2.0.tar.zst"),
            (SHELL_ASSET_ID, "chelis-std-0.2.0.chb"),
        ],
    );
    vec![
        Mock::given(method("GET"))
            .and(wm_path(meta_path))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(metadata_body)
                    .insert_header("content-type", "application/json"),
            ),
        Mock::given(method("GET"))
            .and(wm_path(archive_url_path))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_bytes)),
        Mock::given(method("GET"))
            .and(wm_path(shell_url_path))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(shell_bytes)),
    ]
}

/// Stand up a [`WiremockHarness`] that serves the chelis-std artifacts
/// as `chelis-lang/chelis-std@v0.2.0` release assets via the GitHub
/// API two-step path. The metadata endpoint requires an
/// `Authorization: token ...` header (returns 401 otherwise); the
/// byte endpoints likewise. Returns the harness and the (archive,
/// shell) bytes.
fn fixture_canonical_release() -> (WiremockHarness, Vec<u8>, Vec<u8>) {
    let (archive_p, shell_p) = chelis_std_dist();
    let archive_bytes = fs::read(&archive_p).expect("read archive");
    let shell_bytes = fs::read(&shell_p).expect("read shell");

    let harness = WiremockHarness::new();
    let meta_path = metadata_path("chelis-lang", "chelis-std", "v0.2.0");
    let archive_url_path = asset_id_path("chelis-lang", "chelis-std", ARCHIVE_ASSET_ID);
    let shell_url_path = asset_id_path("chelis-lang", "chelis-std", SHELL_ASSET_ID);
    let metadata_body = metadata_json(
        "v0.2.0",
        &[
            (ARCHIVE_ASSET_ID, "chelis-std-0.2.0.tar.zst"),
            (SHELL_ASSET_ID, "chelis-std-0.2.0.chb"),
        ],
    );

    harness.mount_all(vec![
        // Step 1 — metadata, authenticated.
        Mock::given(method("GET"))
            .and(wm_path(meta_path.clone()))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(metadata_body)
                    .insert_header("content-type", "application/json"),
            ),
        // Step 2 — bytes-by-id, authenticated.
        Mock::given(method("GET"))
            .and(wm_path(archive_url_path.clone()))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_bytes.clone())),
        Mock::given(method("GET"))
            .and(wm_path(shell_url_path.clone()))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(shell_bytes.clone())),
        // Unauthenticated catch-alls (mounted later so they have
        // lower priority — wiremock falls through when the
        // higher-priority header match fails, exercising the
        // auth-rejected path on every endpoint).
        Mock::given(method("GET"))
            .and(wm_path(meta_path))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized")),
        Mock::given(method("GET"))
            .and(wm_path(archive_url_path))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized")),
        Mock::given(method("GET"))
            .and(wm_path(shell_url_path))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized")),
    ]);
    (harness, archive_bytes, shell_bytes)
}

/// A wiremock harness. Owns a long-lived multi-threaded tokio runtime
/// on which the mock server runs, plus the server handle itself. Test
/// code holds this struct, calls `mount_all` once with a vec of
/// pre-built `Mock` instances, then drops back into pure sync code to
/// call the blocking helper-under-test against `harness.uri()`.
///
/// **Why this shape:** `reqwest::blocking::Client` constructs and
/// drops its own internal current-thread tokio runtime per call. If
/// we call `reqwest::blocking` from inside a running tokio runtime
/// (i.e. inside an outer `rt.block_on(async { ... })`), the inner
/// runtime's drop panics with "Cannot drop a runtime in a context
/// where blocking is not allowed." So:
///
/// 1. Wiremock setup runs inside `harness.mount_all(vec![...])`,
///    which `block_on`s long enough to register the mocks and then
///    returns.
/// 2. Once `mount_all` returns, the test body is plain sync code
///    that calls into `chelis_reef::install_from_github`. The
///    runtime is alive (so the wiremock server keeps listening) but
///    we are not *inside* `block_on` — `reqwest::blocking` is happy.
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

    /// Run async setup on the harness's runtime, then return. The
    /// caller hands in a list of `Mock` builders; `setup` mounts each
    /// one. Async closures over `&MockServer` hit lifetime-elision
    /// limits in stable Rust today; passing pre-built `Mock`s
    /// sidesteps the HRTB problem entirely.
    fn mount_all(&self, mocks: Vec<Mock>) {
        self.rt.block_on(async {
            for m in mocks {
                m.mount(&self.server).await;
            }
        });
    }
}

/// Helper: invoke the lib-level `chelis_reef::install_from_github`
/// directly. The CLI binary path is also exercised in this oracle, but
/// the lib-level path is what most sub-cases discriminate against
/// (typed errors, byte-exact comparisons).
///
/// Sets `CHELIS_REEF_GITHUB_BASE_API` and `GITHUB_TOKEN` for the
/// duration of the call. The env mutations rely on the test having
/// acquired the file-wide [`file_lock`] (every `#[test]` in this
/// file does so at entry). To kill the `gh auth token` fallback path
/// deterministically, when `token` is `None` we also empty `PATH`
/// for the call.
///
/// **Pre-condition:** the caller must already hold [`file_lock`].
/// This helper does not acquire it.
fn lib_install_from_github(
    spec: &str,
    api_base_url: &str,
    token: Option<&str>,
    registry_root: &Path,
) -> Result<chelis_reef::InstalledArtifact, chelis_reef::GitHubFetchError> {
    // SAFETY: tests serialize through `file_lock`; no other thread
    // mutates these env vars while the test's guard is held. The
    // helper restores prior values before returning.
    let prior_api_base = std::env::var_os("CHELIS_REEF_GITHUB_BASE_API");
    let prior_token = std::env::var_os("GITHUB_TOKEN");
    let prior_path = std::env::var_os("PATH");
    unsafe {
        std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", api_base_url);
        match token {
            Some(t) => std::env::set_var("GITHUB_TOKEN", t),
            None => {
                std::env::remove_var("GITHUB_TOKEN");
                // Empty PATH so `gh auth token` fallback is unreachable.
                std::env::set_var("PATH", "");
            }
        }
    }
    let result = chelis_reef::install_from_github(spec, registry_root);
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

// Process-global lock. set_var/remove_var are racy, so any sub-case
// that touches process env serializes through this mutex.
//
// We serialize *every* test in this file through this lock — even
// tests that don't touch env directly — because the
// tempdir-cleanup test (`phaseA_item6_tempdir_is_removed_on_success_and_failure`)
// overrides `TMPDIR` for its duration. Any concurrent test that calls
// `tempfile::tempdir()` during that window would contaminate the
// cleanup test's count by creating its own tempdir under the
// override path. Single file-wide lock keeps the cost: ~10 tests, ~80
// ms total in single-thread mode, runs are still well under the
// 60-second default-gate budget. The lock is also poison-tolerant
// (`unwrap_or_else(|p| p.into_inner())`) so a panicking sub-case
// does not cascade into spurious failures across the whole file.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the file-wide test lock. Every `#[test]` function in this
/// file should call this as its first line. The returned guard must
/// be held for the entire test body (don't `_` it; bind it to a
/// named variable that lives until the test returns).
fn file_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Count entries directly under `$TMPDIR` whose names start with
/// `.tmp` — the prefix `tempfile::tempdir()` uses. Comparing before
/// and after a fetch lets the oracle assert the tempdir was removed.
fn count_tempfile_entries(root: &Path) -> usize {
    fs::read_dir(root)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|s| s.starts_with(".tmp"))
                })
                .count()
        })
        .unwrap_or(0)
}

// ============================================================
// Named acceptance oracle. Single test per spec § Item 6, but
// internally dispatches to focused sub-helpers.
// ============================================================

#[test]
fn phaseA_item6_from_github_oracle() {
    let _g = file_lock();
    oracle_happy_path_via_lib_and_cli();
    oracle_byte_equality_with_from_monorepo();
    oracle_auth_missing_no_gh();
    oracle_auth_rejected_401_distinct_from_missing();
    oracle_auth_rejected_403_also_typed();
    oracle_404_release_asset_not_found();
    oracle_hash_mismatch_on_second_fetch();
    oracle_5xx_server_error_distinct_category();
    oracle_429_rate_limited_includes_retry_after();
    oracle_dns_error_distinct_category();
}

fn oracle_happy_path_via_lib_and_cli() {
    let (harness, archive_bytes, shell_bytes) = fixture_canonical_release();
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");

    // Lib-level install.
    let installed = lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        &harness.uri(),
        Some("unit-test-token"),
        &registry,
    )
    .expect("install_from_github happy path");
    assert_eq!(installed.package.name, "chelis-std");
    assert_eq!(installed.package.version, "0.2.0");
    let placed_archive = registry.join("packages/chelis-std/0.2.0/chelis-std-0.2.0.tar.zst");
    let placed_shell = registry.join("packages/chelis-std/0.2.0/chelis-std-0.2.0.chb");
    assert!(placed_archive.exists() && placed_shell.exists());
    assert_eq!(fs::read(&placed_archive).unwrap(), archive_bytes);
    assert_eq!(fs::read(&placed_shell).unwrap(), shell_bytes);
    assert!(
        registry.join("index.json").exists(),
        "index.json must exist after a happy install"
    );
    // Atomic-rename invariant: no leftover index.json.tmp.
    assert!(!registry.join("index.json.tmp").exists());

    // CLI-level install (fresh registry, exercises the binary).
    let cli_registry = dir.path().join("cli-reef-home");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_REEF_HOME", &cli_registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        // Drop PATH so the binary cannot fall back to `gh auth token`.
        .env("PATH", "")
        .args([
            "reef",
            "install",
            "--from-github",
            "chelis-lang/chelis-std@v0.2.0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Installed chelis-std 0.2.0"));
    assert_eq!(
        fs::read(cli_registry.join("packages/chelis-std/0.2.0/chelis-std-0.2.0.tar.zst")).unwrap(),
        archive_bytes
    );
}

fn oracle_byte_equality_with_from_monorepo() {
    // Source-equivalence contract invariant: --from-monorepo and
    // --from-github produce byte-identical local registry state for
    // the same `(name, version)`.
    let (harness, _arc, _shl) = fixture_canonical_release();
    let dir = tempdir().expect("tempdir");
    let github_reg = dir.path().join("github-reef-home");
    let mono_reg = dir.path().join("mono-reef-home");

    // GitHub install.
    lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        &harness.uri(),
        Some("unit-test-token"),
        &github_reg,
    )
    .expect("github install");

    // Monorepo install (CLI, since install_from_monorepo also reads
    // CHELIS_REEF_HOME via registry_root). No env scope races —
    // assert_cmd spawns a child process.
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_REEF_HOME", &mono_reg)
        .args([
            "reef",
            "install",
            "--from-monorepo",
            monorepo_root().to_str().unwrap(),
            "chelis-std=0.2.0",
        ])
        .assert()
        .success();

    let g_a =
        fs::read(github_reg.join("packages/chelis-std/0.2.0/chelis-std-0.2.0.tar.zst")).unwrap();
    let m_a =
        fs::read(mono_reg.join("packages/chelis-std/0.2.0/chelis-std-0.2.0.tar.zst")).unwrap();
    assert_eq!(
        g_a, m_a,
        "archive bytes diverge between --from-github and --from-monorepo"
    );
    let g_s = fs::read(github_reg.join("packages/chelis-std/0.2.0/chelis-std-0.2.0.chb")).unwrap();
    let m_s = fs::read(mono_reg.join("packages/chelis-std/0.2.0/chelis-std-0.2.0.chb")).unwrap();
    assert_eq!(
        g_s, m_s,
        "shell bytes diverge between --from-github and --from-monorepo"
    );

    // Index entries: archive_sha256, shell_sha256, version, compiler must
    // all agree. The package list includes only chelis-std on both sides.
    //
    // Item 9 adds a `remote_origin` field that is populated by
    // `--from-github` (with `github://<org>/<repo>@<tag>`) and left
    // `None` (omitted from JSON) by `--from-monorepo`. That field is
    // intentionally divergent — it records source provenance, which is
    // exactly the user-facing difference between the two install
    // sources. Strip it before comparing so the spec-locked
    // source-equivalence invariant (archive_sha256, shell_sha256,
    // version, compiler) stays asserted.
    let g_idx: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(github_reg.join("index.json")).unwrap()).unwrap();
    let m_idx: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(mono_reg.join("index.json")).unwrap()).unwrap();
    let strip_origin = |v: &serde_json::Value| -> serde_json::Value {
        let arr = v.as_array().expect("packages must be array");
        arr.iter()
            .map(|entry| {
                let mut obj = entry.as_object().expect("entry must be object").clone();
                obj.remove("remote_origin");
                serde_json::Value::Object(obj)
            })
            .collect::<Vec<_>>()
            .into()
    };
    assert_eq!(
        strip_origin(&g_idx["packages"]["chelis-std"]),
        strip_origin(&m_idx["packages"]["chelis-std"]),
        "index.json package entries (modulo remote_origin) diverge between sources"
    );

    // Item 9 forward-lock: `remote_origin` is asymmetric on purpose.
    // GitHub install records the canonical `github://...` URI; monorepo
    // install does not (no remote source exists). The next assertions
    // pin both sides so a regression that drops the GitHub origin or
    // accidentally populates the monorepo origin would surface here.
    let g_first = &g_idx["packages"]["chelis-std"][0];
    let m_first = &m_idx["packages"]["chelis-std"][0];
    assert_eq!(
        g_first["remote_origin"],
        serde_json::Value::String("github://chelis-lang/chelis-std@v0.2.0".to_string()),
        "GitHub install must stamp remote_origin in registry index entry"
    );
    assert!(
        m_first.get("remote_origin").is_none()
            || m_first["remote_origin"] == serde_json::Value::Null,
        "monorepo install must NOT stamp remote_origin"
    );
}

fn oracle_auth_missing_no_gh() {
    // No GITHUB_TOKEN, no `gh` on PATH (we drop PATH inside the
    // helper). The error must name the env var and the actionable fix.
    let (harness, _, _) = fixture_canonical_release();
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");

    let err = lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        &harness.uri(),
        None,
        &registry,
    )
    .expect_err("must fail without auth");
    let msg = err.to_string();
    assert!(
        matches!(err, chelis_reef::GitHubFetchError::AuthMissing { .. }),
        "expected AuthMissing, got: {err:?}"
    );
    assert!(
        msg.contains("GITHUB_TOKEN"),
        "auth-missing message must name GITHUB_TOKEN: {msg}"
    );
    assert!(
        msg.contains("gh auth token"),
        "auth-missing message must suggest `gh auth token`: {msg}"
    );
}

/// Wrong token (not empty, just wrong) yields a 401 from the GitHub
/// API. Helper must surface `GitHubFetchError::AuthRejected { status:
/// 401, .. }` — distinct from `AuthMissing`. The Display strings for
/// the two variants must differ so users can tell "I have no token"
/// from "my token is invalid / lacks scope" at a glance.
///
/// The fixture mounts an authorization-aware mock plus an
/// unauthenticated catch-all that returns 401. Sending a non-empty
/// but non-matching token misses the higher-priority match and falls
/// through to the catch-all, which is exactly what GitHub does for
/// any non-`token unit-test-token` value.
fn oracle_auth_rejected_401_distinct_from_missing() {
    let (harness, _, _) = fixture_canonical_release();
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");

    let err = lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        &harness.uri(),
        Some("not-the-right-token"),
        &registry,
    )
    .expect_err("wrong token must surface 401");
    let msg = err.to_string();
    assert!(
        matches!(
            err,
            chelis_reef::GitHubFetchError::AuthRejected { status: 401, .. }
        ),
        "expected AuthRejected(401), got: {err:?}"
    );
    assert!(
        msg.contains("401"),
        "AuthRejected message must name the HTTP status: {msg}"
    );
    // The URL must appear so users know which endpoint rejected the
    // token (metadata vs. asset bytes — useful when debugging
    // partial-scope tokens).
    assert!(
        msg.contains(&harness.uri()) || msg.contains("releases/tags"),
        "AuthRejected message must name the URL: {msg}"
    );

    // Display-distinctness contract: AuthMissing and AuthRejected
    // must produce different user-facing messages so the operator
    // can tell "I never set a token" from "my token is wrong."
    let missing_err = chelis_reef::GitHubFetchError::AuthMissing {
        reason: "synthetic for distinctness check".to_string(),
    };
    assert_ne!(
        missing_err.to_string(),
        msg,
        "AuthMissing and AuthRejected Display must be distinct"
    );
}

/// 403 also maps to `AuthRejected`. Real-world GitHub uses 403 for
/// "token is valid but lacks repo scope" and 401 for "token is
/// missing or invalid"; both are auth-rejection from the user's
/// perspective. Our helper collapses both into the same typed
/// variant with the actual HTTP status preserved in the `status`
/// field, so callers can distinguish if needed.
fn oracle_auth_rejected_403_also_typed() {
    let harness = WiremockHarness::new();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path(
                "chelis-lang",
                "chelis-std",
                "v0.2.0",
            )))
            .respond_with(ResponseTemplate::new(403).set_body_string("forbidden")),
    ]);
    let dir = tempdir().expect("tempdir");
    let err = lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        &harness.uri(),
        Some("scope-limited-token"),
        &dir.path().join("reef-home"),
    )
    .expect_err("403 must surface AuthRejected");
    assert!(
        matches!(
            err,
            chelis_reef::GitHubFetchError::AuthRejected { status: 403, .. }
        ),
        "expected AuthRejected(403), got: {err:?}"
    );
}

fn oracle_404_release_asset_not_found() {
    // Tag has no release at all: the metadata endpoint 404s and the
    // helper must surface ReleaseAssetNotFound naming the metadata
    // URL we tried — that is the actionable signal for "wrong tag /
    // release missing." Asset-list-mismatch (release exists but the
    // named asset is absent) is exercised separately by
    // `phaseA_item6_wrong_asset_name_in_release_is_typed_404`, which
    // covers the post-metadata branch of `find_asset_id`.
    let harness = WiremockHarness::new();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path(
                "chelis-lang",
                "missing-shell",
                "v0.2.0",
            )))
            .respond_with(ResponseTemplate::new(404)),
    ]);

    let dir = tempdir().expect("tempdir");
    let err = lib_install_from_github(
        "chelis-lang/missing-shell@v0.2.0",
        &harness.uri(),
        Some("unit-test-token"),
        &dir.path().join("reef-home"),
    )
    .expect_err("must 404");
    assert!(
        matches!(
            err,
            chelis_reef::GitHubFetchError::ReleaseAssetNotFound { .. }
        ),
        "expected ReleaseAssetNotFound, got: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("releases/tags/v0.2.0"),
        "metadata-404 message must name the metadata URL: {msg}"
    );
    assert!(
        msg.contains("HTTP 404"),
        "404 message must name status: {msg}"
    );
}

fn oracle_hash_mismatch_on_second_fetch() {
    // First install puts known-good chelis-std bytes into the registry.
    // Then we serve different bytes (corrupt the archive on the wire)
    // and re-install: the helper computes a fresh sha256 on the bytes
    // it just wrote, but the embedded shell.archive_sha256 will not
    // match — that is the agreement check fired by
    // install_validated_artifact_pair.
    let (archive_p, shell_p) = chelis_std_dist();
    let archive_bytes = fs::read(&archive_p).unwrap();
    let shell_bytes = fs::read(&shell_p).unwrap();

    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");

    // Install round one: serve the genuine archive.
    {
        let harness = WiremockHarness::new();
        harness.mount_all(canonical_api_mocks(
            archive_bytes.clone(),
            shell_bytes.clone(),
        ));
        lib_install_from_github(
            "chelis-lang/chelis-std@v0.2.0",
            &harness.uri(),
            Some("unit-test-token"),
            &registry,
        )
        .expect("first install succeeds");
    }

    // Install round two: serve a tampered archive (single byte flip)
    // alongside the original shell. The shell's embedded
    // archive_sha256 still names the original archive, so the helper
    // detects the disagreement.
    let mut tampered = archive_bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xff;
    assert_ne!(sha256_bytes(&archive_bytes), sha256_bytes(&tampered));

    let harness = WiremockHarness::new();
    harness.mount_all(canonical_api_mocks(tampered, shell_bytes));

    let err = lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        &harness.uri(),
        Some("unit-test-token"),
        &registry,
    )
    .expect_err("hash mismatch must fail");
    let msg = err.to_string();
    assert!(
        matches!(err, chelis_reef::GitHubFetchError::Validation { .. }),
        "expected Validation error category, got: {err:?}"
    );
    assert!(
        msg.contains("archive_sha256"),
        "hash-mismatch message must name the disagreeing field: {msg}"
    );
}

fn oracle_5xx_server_error_distinct_category() {
    // 5xx is mounted on the metadata endpoint — that's the first
    // call the helper makes; the byte-download endpoints are never
    // reached.
    let harness = WiremockHarness::new();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path(
                "chelis-lang",
                "chelis-std",
                "v0.2.0",
            )))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream down")),
    ]);
    let dir = tempdir().expect("tempdir");
    let err = lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        &harness.uri(),
        Some("unit-test-token"),
        &dir.path().join("reef-home"),
    )
    .expect_err("must surface 5xx");
    assert!(
        matches!(
            err,
            chelis_reef::GitHubFetchError::ServerError { status: 503, .. }
        ),
        "expected ServerError(503), got: {err:?}"
    );
}

fn oracle_429_rate_limited_includes_retry_after() {
    let harness = WiremockHarness::new();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path(
                "chelis-lang",
                "chelis-std",
                "v0.2.0",
            )))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("Retry-After", "120")
                    .set_body_string("rate limited"),
            ),
    ]);
    let dir = tempdir().expect("tempdir");
    let err = lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        &harness.uri(),
        Some("unit-test-token"),
        &dir.path().join("reef-home"),
    )
    .expect_err("must surface 429");
    let msg = err.to_string();
    assert!(
        matches!(err, chelis_reef::GitHubFetchError::RateLimited { .. }),
        "expected RateLimited, got: {err:?}"
    );
    assert!(
        msg.contains("120"),
        "rate-limit message must include Retry-After value: {msg}"
    );
}

fn oracle_dns_error_distinct_category() {
    // Point at a host whose name will never resolve. The `.invalid`
    // TLD is reserved (RFC 2606) and guaranteed not to resolve, so we
    // get a real DNS error category.
    let dir = tempdir().expect("tempdir");
    let err = lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        "http://this-host-does-not-exist.invalid",
        Some("unit-test-token"),
        &dir.path().join("reef-home"),
    )
    .expect_err("must surface DNS error");
    assert!(
        matches!(err, chelis_reef::GitHubFetchError::Network { .. }),
        "expected Network category for DNS failure, got: {err:?}"
    );
}

// ============================================================
// Negative parity tests (separate test functions, run by the default
// `cargo test --workspace` loop).
// ============================================================

#[test]
fn phaseA_item6_partial_install_does_not_corrupt_index() {
    let _g = file_lock();
    // The helper verifies the shell *before* writing the index, so
    // any failure inside `install_validated_artifact_pair` (e.g. a
    // corrupted shell) must leave `index.json` untouched. We exercise
    // that path by serving a shell that's not a valid Chelis shell
    // payload — read_shell fails before the index is touched.
    let (archive_p, _) = chelis_std_dist();
    let archive_bytes = fs::read(&archive_p).unwrap();

    // Reuse the canonical-API mock shape but substitute invalid bytes
    // for the shell payload.
    let harness = WiremockHarness::new();
    harness.mount_all(canonical_api_mocks(
        archive_bytes,
        b"not-a-valid-shell".to_vec(),
    ));

    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let err = lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        &harness.uri(),
        Some("unit-test-token"),
        &registry,
    )
    .expect_err("invalid shell must fail validation");
    assert!(matches!(
        err,
        chelis_reef::GitHubFetchError::Validation { .. }
    ));

    // Atomicity: index.json must not exist (we never reached the
    // write step) and no orphan index.json.tmp is left behind.
    assert!(
        !registry.join("index.json").exists(),
        "index.json must NOT exist after validation failure"
    );
    assert!(
        !registry.join("index.json.tmp").exists(),
        "no orphan index.json.tmp must be left behind"
    );
}

#[test]
fn phaseA_item6_wrong_asset_name_in_release_is_typed_404() {
    let _g = file_lock();
    // Lock the asset-naming convention via negative test: a release
    // exists but carries the non-canonical name `.tgz` instead of the
    // locked `.tar.zst`. The metadata lookup succeeds (release is
    // present), then `find_asset_id` fails to match the expected
    // `.tar.zst` name in the assets list and surfaces a typed
    // ReleaseAssetNotFound — no silent fallback to the wrong name.
    // The error message must list the assets actually present so a
    // publisher can spot the misnaming without a second `gh release
    // view` round-trip.
    let harness = WiremockHarness::new();
    let metadata_body = metadata_json(
        "v0.2.0",
        // Note: only the `.tgz` form is attached — the canonical
        // `.tar.zst` is missing. Asset id 9999 is arbitrary; the
        // helper never reaches the byte-fetch step.
        &[(9999, "chelis-std-0.2.0.tgz")],
    );
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path(
                "chelis-lang",
                "chelis-std",
                "v0.2.0",
            )))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(metadata_body)
                    .insert_header("content-type", "application/json"),
            ),
    ]);
    let dir = tempdir().expect("tempdir");
    let err = lib_install_from_github(
        "chelis-lang/chelis-std@v0.2.0",
        &harness.uri(),
        Some("unit-test-token"),
        &dir.path().join("reef-home"),
    )
    .expect_err("wrong asset name must surface ReleaseAssetNotFound");
    assert!(matches!(
        err,
        chelis_reef::GitHubFetchError::ReleaseAssetNotFound { .. }
    ));
    let msg = err.to_string();
    assert!(
        msg.contains(".tar.zst"),
        "404 must name canonical asset: {msg}"
    );
    assert!(
        msg.contains("chelis-std-0.2.0.tgz"),
        "404 must list assets actually present: {msg}"
    );
}

#[test]
fn phaseA_item6_tag_without_leading_v_is_accepted_and_normalized() {
    let _g = file_lock();
    // Spec lock (per brief): both `chelis-lang/<r>@v0.2.0` and
    // `chelis-lang/<r>@0.2.0` install to packages/<r>/0.2.0/. The
    // `v` is decorative: when present, it appears in the metadata
    // URL (`/releases/tags/v0.2.0` vs `/releases/tags/0.2.0`) so we
    // serve metadata under whichever form the caller passed; the
    // version is the tag with one `v` stripped, so the on-disk
    // package directory and the asset names use `0.2.0` either way.
    let (archive_p, shell_p) = chelis_std_dist();
    let archive_bytes = fs::read(&archive_p).unwrap();
    let shell_bytes = fs::read(&shell_p).unwrap();

    // Form: tag = "0.2.0" (no leading v). Metadata path:
    // /repos/.../releases/tags/0.2.0
    let harness = WiremockHarness::new();
    let metadata_body = metadata_json(
        "0.2.0",
        &[
            (ARCHIVE_ASSET_ID, "chelis-std-0.2.0.tar.zst"),
            (SHELL_ASSET_ID, "chelis-std-0.2.0.chb"),
        ],
    );
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path("chelis-lang", "chelis-std", "0.2.0")))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(metadata_body)
                    .insert_header("content-type", "application/json"),
            ),
        Mock::given(method("GET"))
            .and(wm_path(asset_id_path(
                "chelis-lang",
                "chelis-std",
                ARCHIVE_ASSET_ID,
            )))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_bytes)),
        Mock::given(method("GET"))
            .and(wm_path(asset_id_path(
                "chelis-lang",
                "chelis-std",
                SHELL_ASSET_ID,
            )))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(shell_bytes)),
    ]);

    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let installed = lib_install_from_github(
        "chelis-lang/chelis-std@0.2.0",
        &harness.uri(),
        Some("unit-test-token"),
        &registry,
    )
    .expect("no-v form must succeed");
    assert_eq!(installed.package.version, "0.2.0");
    assert!(
        registry
            .join("packages/chelis-std/0.2.0/chelis-std-0.2.0.tar.zst")
            .exists()
    );
}

#[test]
fn phaseA_item6_parse_failure_messages_name_the_input() {
    let _g = file_lock();
    // Each malformed input must surface a typed Parse error whose
    // message includes the original input. This is the contract that
    // makes troubleshooting tractable for users.
    let cases = [
        ("missing-slash@v0.2.0", "missing `/`"),
        ("org/repo-no-at-tag", "missing `@<tag>`"),
        ("/repo@v0.2.0", "empty <org>"),
        ("org/@v0.2.0", "empty <repo>"),
        ("org/repo@", "empty <tag>"),
        ("org/repo@v", "tag is just `v`"),
    ];
    for (input, must_contain_reason) in cases {
        let result = chelis_reef::GitHubReleaseSpec::parse(input);
        let err = result.expect_err(&format!("`{input}` must parse-fail"));
        assert!(
            matches!(err, chelis_reef::GitHubFetchError::Parse { .. }),
            "`{input}` must surface Parse, got: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains(input),
            "parse error for `{input}` must name the input; got: {msg}"
        );
        assert!(
            msg.contains(must_contain_reason),
            "parse error for `{input}` must name reason `{must_contain_reason}`; got: {msg}"
        );
    }
}

#[test]
fn phaseA_item6_parse_accepts_v_and_no_v_forms() {
    let _g = file_lock();
    // Symmetric positive: both `v0.2.0` and `0.2.0` parse; the
    // version field is the tag without one leading v.
    let with_v = chelis_reef::GitHubReleaseSpec::parse("chelis-lang/x@v0.4.0").unwrap();
    let no_v = chelis_reef::GitHubReleaseSpec::parse("chelis-lang/x@0.4.0").unwrap();
    assert_eq!(with_v.tag, "v0.4.0");
    assert_eq!(with_v.version, "0.4.0");
    assert_eq!(no_v.tag, "0.4.0");
    assert_eq!(no_v.version, "0.4.0");
}

#[test]
fn phaseA_item6_tempdir_is_removed_on_success_and_failure() {
    let _g = file_lock();
    // The fetch tempdir lives inside `$TMPDIR` (the system temp). We
    // override TMPDIR to a private root and count entries there
    // before and after each install path. The file-wide lock keeps
    // any other test in this binary from creating tempdirs under the
    // override during this test's window.
    let outer = tempdir().expect("outer tempdir");
    let private_tmp = outer.path().join("private-tmp");
    fs::create_dir_all(&private_tmp).unwrap();
    let registry = outer.path().join("reef-home");

    // Success path.
    {
        let (harness, _, _) = fixture_canonical_release();
        let prior_tmp = std::env::var_os("TMPDIR");
        let prior_base = std::env::var_os("CHELIS_REEF_GITHUB_BASE_API");
        let prior_token = std::env::var_os("GITHUB_TOKEN");
        unsafe {
            std::env::set_var("TMPDIR", &private_tmp);
            std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", harness.uri());
            std::env::set_var("GITHUB_TOKEN", "unit-test-token");
        }
        let before = count_tempfile_entries(&private_tmp);
        chelis_reef::install_from_github("chelis-lang/chelis-std@v0.2.0", &registry)
            .expect("install ok");
        let after = count_tempfile_entries(&private_tmp);
        unsafe {
            match prior_tmp {
                Some(p) => std::env::set_var("TMPDIR", p),
                None => std::env::remove_var("TMPDIR"),
            }
            match prior_base {
                Some(v) => std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", v),
                None => std::env::remove_var("CHELIS_REEF_GITHUB_BASE_API"),
            }
            match prior_token {
                Some(v) => std::env::set_var("GITHUB_TOKEN", v),
                None => std::env::remove_var("GITHUB_TOKEN"),
            }
        }
        assert_eq!(
            before, after,
            "tempdir count must be the same after success"
        );
    }

    // Failure path: connection refused at discard port (RFC 863).
    {
        let prior_tmp = std::env::var_os("TMPDIR");
        let prior_base = std::env::var_os("CHELIS_REEF_GITHUB_BASE_API");
        let prior_token = std::env::var_os("GITHUB_TOKEN");
        unsafe {
            std::env::set_var("TMPDIR", &private_tmp);
            std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", "http://localhost:9");
            std::env::set_var("GITHUB_TOKEN", "unit-test-token");
        }
        let before = count_tempfile_entries(&private_tmp);
        let _ = chelis_reef::install_from_github("chelis-lang/chelis-std@v0.2.0", &registry);
        let after = count_tempfile_entries(&private_tmp);
        unsafe {
            match prior_tmp {
                Some(p) => std::env::set_var("TMPDIR", p),
                None => std::env::remove_var("TMPDIR"),
            }
            match prior_base {
                Some(v) => std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", v),
                None => std::env::remove_var("CHELIS_REEF_GITHUB_BASE_API"),
            }
            match prior_token {
                Some(v) => std::env::set_var("GITHUB_TOKEN", v),
                None => std::env::remove_var("GITHUB_TOKEN"),
            }
        }
        assert_eq!(
            before, after,
            "tempdir count must be the same after failure"
        );
    }
}

/// Io error category — H3 regression. Wave 0 raised
/// `GitHubFetchError::Io` on tempdir/file/client failures but no
/// test asserted on it. Point `TMPDIR` at a path whose parent isn't
/// writable so `tempfile::tempdir()` fails with an OS error; the
/// helper must surface that as `Io`, distinct from `Network` and
/// from `Validation`.
///
/// Strategy: create a regular file `unwritable-dir-stub`, then point
/// `TMPDIR` at a child of it. `tempfile::tempdir()` calls
/// `fs::create_dir_all($TMPDIR)` (or equivalently
/// `fs::create_dir($TMPDIR/<random>)`), which fails with `ENOTDIR`
/// because the parent path is a file. That's a clean OS-level
/// failure independent of permissions, which makes the test
/// portable across filesystems and runner identities.
#[test]
fn phaseA_item6_io_error_category_on_tempdir_failure() {
    let _g = file_lock();
    let outer = tempdir().expect("outer tempdir");
    // `outer/not-a-dir` is a regular file. We then point TMPDIR at
    // a path *inside* that file — there's no way to create a
    // directory under a regular file, so tempfile::tempdir() fails.
    let stub_file = outer.path().join("not-a-dir");
    fs::write(&stub_file, b"placeholder").expect("seed stub file");
    let bad_tmpdir = stub_file.join("inside");

    let prior_tmp = std::env::var_os("TMPDIR");
    let prior_token = std::env::var_os("GITHUB_TOKEN");
    let prior_api_base = std::env::var_os("CHELIS_REEF_GITHUB_BASE_API");
    unsafe {
        std::env::set_var("TMPDIR", &bad_tmpdir);
        std::env::set_var("GITHUB_TOKEN", "unit-test-token");
        // API base does not matter — we never get past tempdir
        // creation. Pointing at the discard port keeps the test
        // hermetic if the bad-TMPDIR somehow doesn't trip first.
        std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", "http://localhost:9");
    }
    let registry = outer.path().join("reef-home");
    let result = chelis_reef::install_from_github("chelis-lang/chelis-std@v0.2.0", &registry);
    unsafe {
        match prior_tmp {
            Some(p) => std::env::set_var("TMPDIR", p),
            None => std::env::remove_var("TMPDIR"),
        }
        match prior_token {
            Some(v) => std::env::set_var("GITHUB_TOKEN", v),
            None => std::env::remove_var("GITHUB_TOKEN"),
        }
        match prior_api_base {
            Some(v) => std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", v),
            None => std::env::remove_var("CHELIS_REEF_GITHUB_BASE_API"),
        }
    }
    let err = result.expect_err("must fail with Io on bad TMPDIR");
    assert!(
        matches!(err, chelis_reef::GitHubFetchError::Io { .. }),
        "expected Io error category for tempdir failure, got: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("tempdir") || msg.contains("I/O"),
        "Io message must name the failed step: {msg}"
    );

    // Type-distinctness: Io must not collide with Network or
    // Validation Display strings on this case.
    let synthetic_network = chelis_reef::GitHubFetchError::Network {
        url: "http://example".to_string(),
        message: "synthetic".to_string(),
    };
    let synthetic_validation = chelis_reef::GitHubFetchError::Validation {
        message: "synthetic".to_string(),
    };
    assert_ne!(synthetic_network.to_string(), msg);
    assert_ne!(synthetic_validation.to_string(), msg);
}

#[test]
fn phaseA_item6_cli_rejects_both_sources_set() {
    let _g = file_lock();
    // CLI surface: --from-monorepo and --from-github are mutually
    // exclusive. The clap derive's `conflicts_with` enforces this at
    // arg-parse time, before any work runs.
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "reef",
            "install",
            "--from-monorepo",
            ".",
            "--from-github",
            "chelis-lang/x@v0.2.0",
        ])
        .assert()
        .failure();
}

#[test]
fn phaseA_item6_cli_rejects_positional_packages_with_from_github() {
    let _g = file_lock();
    // The brief locks: positional package selectors are ignored /
    // rejected with --from-github (the spec is the selector). We
    // surface a clear error rather than silently discard them.
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("GITHUB_TOKEN", "x")
        .env("PATH", "")
        .env("CHELIS_REEF_GITHUB_BASE_API", "http://localhost:9")
        .args([
            "reef",
            "install",
            "--from-github",
            "chelis-lang/x@v0.2.0",
            "extra-positional",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("positional"));
}

#[test]
fn phaseA_item6_help_lists_from_github_flag() {
    let _g = file_lock();
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["reef", "install", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--from-github"));
}

// ============================================================
// Manual gate (real network). Must be `#[ignore]`d. Documented
// invocation: `cargo test -p chelis-cli phaseA_real_github_manual_gate
// -- --ignored --exact`. The runner must have a working
// `GITHUB_TOKEN` with read access to the canonical org and the
// `chelis-lang/nautilus@<PINNED_TAG>` release must still exist with
// both canonical assets attached.
//
// **Maintenance rule:** the pinned tag is intentionally not
// discovered at runtime — pinning is what lets this gate catch
// publication regressions (release deleted, asset deleted, asset
// renamed). When the canonical org publishes a new Nautilus release,
// bump `MANUAL_GATE_NAUTILUS_TAG` below to the new tag and update the
// derived assertions to match the new version string. Verify the
// release exists with both canonical assets attached before
// committing the bump (`gh release view <tag> -R chelis-lang/nautilus
// --json assets --jq '.assets[].name'`).
// ============================================================

/// Tag pinned to the canonical `chelis-lang/nautilus` release this
/// manual gate validates against. Bump on each new Nautilus release.
const MANUAL_GATE_NAUTILUS_TAG: &str = "v0.5.0";
/// Version string derived from the tag (leading `v` stripped). Used
/// for the asset filenames and the on-disk package directory.
const MANUAL_GATE_NAUTILUS_VERSION: &str = "0.6.0";

#[test]
#[ignore = "real-network manual gate; run with `--ignored --exact`"]
fn phaseA_real_github_manual_gate() {
    let _g = file_lock();
    let token = std::env::var("GITHUB_TOKEN").unwrap_or_default();
    assert!(
        !token.is_empty(),
        "manual gate requires GITHUB_TOKEN set; \
         run `export GITHUB_TOKEN=$(gh auth token)` first"
    );
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");
    let spec = format!("chelis-lang/nautilus@{MANUAL_GATE_NAUTILUS_TAG}");
    let expected_stdout = format!("Installed nautilus {MANUAL_GATE_NAUTILUS_VERSION}");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_REEF_HOME", &registry)
        .env("GITHUB_TOKEN", token)
        .args(["reef", "install", "--from-github", spec.as_str()])
        .assert()
        .success()
        .stdout(predicate::str::contains(expected_stdout));
    let pkg_dir = registry
        .join("packages/nautilus")
        .join(MANUAL_GATE_NAUTILUS_VERSION);
    assert!(
        pkg_dir
            .join(format!("nautilus-{MANUAL_GATE_NAUTILUS_VERSION}.chb"))
            .exists()
    );
    assert!(
        pkg_dir
            .join(format!("nautilus-{MANUAL_GATE_NAUTILUS_VERSION}.tar.zst"))
            .exists()
    );
}
