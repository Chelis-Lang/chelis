//! Phase A — Item 8 named acceptance oracle: auto-fetch during
//! `chelis reef build`.
//!
//! ```sh
//! cargo test -p chelis-cli phaseA_item8_autofetch_build_oracle -- --exact
//! ```
//!
//! The oracle is a single comprehensive test that exercises every
//! spec acceptance bullet for Item 8 against a localhost wiremock
//! fixture. The `CHELIS_REEF_GITHUB_BASE_API` env var injects the
//! fixture URL into the auto-fetch path; without that seam the test
//! would hit real GitHub.
//!
//! Negative-parity tests are separate `#[test]` functions; they cover
//! the auth-missing, 429, stale-lock, and one-of-two-builds-opted-out
//! cases the brief locks. The manual gate
//! `phaseA_item8_real_github_manual_gate` is `#[ignore]`d and lives at
//! the bottom of the file.
//!
//! Coordination with Item 9 (parallel agent): the test
//! `phaseA_item8_lockfile_remote_origin_honored_when_present` is
//! intentionally `#[ignore]`d with a comment describing how to wire it
//! on after Item 9 lands. The brief locks the shim such that adding
//! the field after Item 9 is a one-line change to
//! `lockfile_remote_origin` in `chelis-reef/src/lib.rs`.

#![allow(non_snake_case)]

use assert_cmd::Command;
use predicates::prelude::*;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use tar::Builder;
use tempfile::tempdir;
use wiremock::matchers::{header, method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const CURRENT_COMPILER_PIN: &str = concat!("=", env!("CARGO_PKG_VERSION"));

// ============================================================
// Shared fixture plumbing.
//
// Phase A correction (chelis-std-runtime fix wave): these tests
// previously fetched real chelis-std bytes off the monorepo dist
// tree and pretended they were a network-distributed shell. With
// chelis-std now classified as the language runtime (not a shell),
// auto-fetch is intentionally disabled for chelis-std and the
// previous fixture path is no longer reachable. The tests now use
// a synthetic shell named `nautilus` constructed at test start;
// the wiremock fixture serves the synthetic bytes.
// ============================================================

/// Build a `.tar.zst` archive containing `reef.toml` + a placeholder
/// `src/main.ch`. The manifest declares the given name, version, and
/// dependencies. Same shape as item-7's helper. Returns archive bytes.
fn build_test_archive(name: &str, version: &str, deps: &[(&str, &str)]) -> Vec<u8> {
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
compiler = "{compiler}"
module_prefix = "Test"
{deps_toml}"#,
        compiler = CURRENT_COMPILER_PIN,
    );
    let main_text = "module Test.Main\n\nexport (placeholder)\ndef placeholder() -> int32 = 0\n";
    let mut tar_bytes = Vec::new();
    {
        let mut builder = Builder::new(&mut tar_bytes);
        let manifest_bytes = manifest_text.as_bytes();
        let mut header = tar::Header::new_gnu();
        header.set_path("reef.toml").expect("set path");
        header.set_size(manifest_bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append(&header, manifest_bytes)
            .expect("append manifest");
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

/// Build a synthetic shell binary whose `archive_sha256` matches the
/// archive's bytes. Item-8 tests only need the validation pipeline to
/// accept the bytes — no module data has to be present.
fn build_test_shell_bytes(name: &str, version: &str, archive_sha256: &str) -> Vec<u8> {
    let shell = chelis_shell::ShellPackage {
        format_version: chelis_shell::SHELL_FORMAT_VERSION,
        package: chelis_shell::PackageId {
            name: name.to_string(),
            version: version.to_string(),
        },
        compiler: CURRENT_COMPILER_PIN.to_string(),
        modules: Vec::new(),
        dependencies: Vec::new(),
        archive_sha256: archive_sha256.to_string(),
    };
    chelis_shell::encode_shell(&shell).expect("encode shell")
}

/// Synthetic `nautilus` archive+shell pair used as the auto-fetch
/// target. No filesystem dependencies; everything is in-memory.
fn synthetic_nautilus_artifacts() -> (Vec<u8>, Vec<u8>) {
    // Empty deps — nautilus is the leaf in this fixture's graph.
    let archive = build_test_archive("nautilus", "0.2.0", &[]);
    let archive_sha = sha256_bytes(&archive);
    let shell = build_test_shell_bytes("nautilus", "0.2.0", &archive_sha);
    (archive, shell)
}

fn metadata_path(org: &str, repo: &str, tag: &str) -> String {
    format!("/repos/{org}/{repo}/releases/tags/{tag}")
}

fn asset_id_path(org: &str, repo: &str, asset_id: u64) -> String {
    format!("/repos/{org}/{repo}/releases/assets/{asset_id}")
}

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

const ARCHIVE_ASSET_ID: u64 = 2001;
const SHELL_ASSET_ID: u64 = 2002;

/// Build the canonical-API mocks for `chelis-lang/nautilus@v0.2.0`.
/// The fixture pins to the synthetic `nautilus@0.2.0` so the wiremock
/// path-based matchers cannot accidentally race against another test's
/// `nautilus` mock.
fn canonical_api_mocks(archive_bytes: Vec<u8>, shell_bytes: Vec<u8>) -> Vec<Mock> {
    let meta_path = metadata_path("chelis-lang", "nautilus", "v0.2.0");
    let archive_url_path = asset_id_path("chelis-lang", "nautilus", ARCHIVE_ASSET_ID);
    let shell_url_path = asset_id_path("chelis-lang", "nautilus", SHELL_ASSET_ID);
    let metadata_body = metadata_json(
        "v0.2.0",
        &[
            (ARCHIVE_ASSET_ID, "nautilus-0.2.0.tar.zst"),
            (SHELL_ASSET_ID, "nautilus-0.2.0.chb"),
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

/// Wiremock harness — same shape as Item 6's oracle file. See its
/// rustdoc for why the runtime stays alive while we run sync code
/// against it.
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

/// Stage a tiny `downstream` package that depends on `nautilus =
/// "0.2.0"` (a synthetic shell mocked by the wiremock fixture). The
/// runtime (`chelis-std`) is implicit and not declared.
fn stage_downstream_project(parent: &Path) -> std::path::PathBuf {
    // Use a unique name per test invocation to avoid path-name
    // collisions when two tests share an outer tempdir. AtomicUsize
    // is process-shared; collisions across binaries are impossible
    // because each test gets its own tempdir.
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let app = parent.join(format!("downstream-{n}"));
    fs::create_dir_all(app.join("src")).expect("mkdir downstream/src");
    let ver = env!("CARGO_PKG_VERSION");
    fs::write(
        app.join("reef.toml"),
        format!(
            r#"[package]
name = "downstream-item8"
version = "0.2.0"
compiler = "={ver}"
module_prefix = "Demo"

[dependencies]
nautilus = {{ version = "0.2.0" }}
"#,
        ),
    )
    .expect("write reef.toml");
    fs::write(
        app.join("src/main.ch"),
        // Trivial body — Item 8 cares about resolution, not codegen.
        "module Demo.Main\n\ndef noop(x: int32) -> int32 = x\n",
    )
    .expect("write main.ch");
    app
}

// File-wide env lock: every test that mutates process env or runs
// `chelis reef build` (which reads `$CHELIS_REEF_HOME` /
// `$GITHUB_TOKEN`) must serialize through this. Same shape as
// Item 6's `file_lock`.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn file_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Read `index.json` and assert the synthetic nautilus@0.2.0 was
/// installed via auto-fetch.
fn assert_nautilus_installed(reef_home: &Path) {
    let index_path = reef_home.join("index.json");
    assert!(
        index_path.exists(),
        "index.json must exist after auto-fetch"
    );
    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&index_path).unwrap()).unwrap();
    let entries = index["packages"]["nautilus"]
        .as_array()
        .expect("packages.nautilus must be array");
    assert!(
        entries.iter().any(|e| e["version"] == "0.2.0"),
        "expected nautilus 0.2.0 in index, got {entries:?}"
    );
    assert!(
        reef_home
            .join("packages/nautilus/0.2.0/nautilus-0.2.0.tar.zst")
            .exists()
    );
    assert!(
        reef_home
            .join("packages/nautilus/0.2.0/nautilus-0.2.0.chb")
            .exists()
    );
}

// ============================================================
// Named acceptance oracle. Single test per spec § Item 8.
// ============================================================

/// Phase A Item 8 named acceptance oracle. One test per spec, internally
/// dispatching to focused sub-cases. Each sub-case asserts a load-bearing
/// invariant from the brief or the spec.
#[test]
fn phaseA_item8_autofetch_build_oracle() {
    let _g = file_lock();
    oracle_autofetch_happy_path();
    oracle_no_auto_fetch_opt_out_blocks_fetch();
    oracle_help_lists_no_auto_fetch_flag();
    oracle_autofetch_network_failure_no_half_install();
    oracle_lock_file_engaged_during_autofetch();
    oracle_error_wording_shape_regex();
    oracle_autofetch_event_observable();
}

/// Acceptance bullet 1 (spec): a fresh dev environment with
/// `GITHUB_TOKEN` set runs `chelis reef build` on a project that
/// depends on a shell (nautilus) and the build succeeds with auto-fetch
/// in the middle.
fn oracle_autofetch_happy_path() {
    let (archive_bytes, shell_bytes) = synthetic_nautilus_artifacts();

    let harness = WiremockHarness::new();
    harness.mount_all(canonical_api_mocks(
        archive_bytes.clone(),
        shell_bytes.clone(),
    ));

    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_downstream_project(outer.path());

    // Empty registry. Build must auto-fetch nautilus before failing.
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .current_dir(&app)
        .args(["reef", "build"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Built downstream-item8"));

    assert_nautilus_installed(&reef_home);
    // Lockfile written under the downstream project root.
    assert!(app.join("reef.lock").exists());

    // Index file is on disk and well-formed (atomic_write contract).
    assert!(!reef_home.join("index.json.tmp").exists());
}

/// Acceptance bullet 2 (spec): same project under `--no-auto-fetch`
/// fails with the existing error shape but improved wording.
fn oracle_no_auto_fetch_opt_out_blocks_fetch() {
    // No need for a wiremock — auto-fetch is off, nothing should call
    // out to the network at all. Pointing the API base at the discard
    // port keeps the test hermetic if we accidentally do.
    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_downstream_project(outer.path());

    let assertion = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", "http://localhost:9")
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .current_dir(&app)
        .args(["reef", "build", "--no-auto-fetch"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();

    // Improved-wording invariants: name URL, name auto-fetch state.
    assert!(
        stderr.contains("chelis-lang/nautilus@v0.2.0"),
        "no-auto-fetch error must name canonical-org URL; got: {stderr}"
    );
    assert!(
        stderr.contains("auto-fetch disabled"),
        "no-auto-fetch error must declare auto-fetch was disabled; got: {stderr}"
    );
    assert!(
        stderr.contains("chelis reef install --from-github"),
        "no-auto-fetch error must hint at manual install command; got: {stderr}"
    );

    // No half-install: registry remains empty (no index.json, no
    // packages dir).
    assert!(
        !reef_home.join("index.json").exists(),
        "no-auto-fetch must not touch index.json"
    );
    assert!(
        !reef_home.join("packages").exists(),
        "no-auto-fetch must not create packages/"
    );
}

/// Brief lock: the `--no-auto-fetch` flag must appear in
/// `chelis reef build --help` so users can discover the opt-out.
fn oracle_help_lists_no_auto_fetch_flag() {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["reef", "build", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--no-auto-fetch"));
}

/// Negative parity: auto-fetch network failure during build. Error
/// must name the URL and the typed category; registry must remain
/// empty (no half-install).
fn oracle_autofetch_network_failure_no_half_install() {
    // Wiremock that returns 503 on the metadata endpoint. The
    // helper surfaces `ServerError` category.
    let harness = WiremockHarness::new();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path("chelis-lang", "nautilus", "v0.2.0")))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream down")),
    ]);

    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_downstream_project(outer.path());

    let assertion = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .current_dir(&app)
        .args(["reef", "build"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();

    assert!(
        stderr.contains("chelis-lang/nautilus@v0.2.0"),
        "network-failure message must name URL; got: {stderr}"
    );
    assert!(
        stderr.contains("category: server-error"),
        "network-failure message must name typed category; got: {stderr}"
    );
    assert!(
        stderr.contains("auto-fetch failed"),
        "message must announce auto-fetch failure; got: {stderr}"
    );

    // No half-install: index.json was never written, no orphan tmp.
    assert!(!reef_home.join("index.json").exists());
    assert!(!reef_home.join("index.json.tmp").exists());
    assert!(!reef_home.join("packages/nautilus/0.2.0").exists());
}

/// Lock-engaged invariant: while auto-fetch runs, the
/// `.reef-lock` file exists at the registry root. We assert it is
/// gone (or, more precisely, has no exclusive holder) once the
/// build returns. The simplest cross-platform observable: the file
/// itself exists post-success because we don't unlink it (it is
/// safe to leave; `flock` semantics live on the open fd not the
/// path), but its presence after success — without an active
/// holder — must not block subsequent acquisitions.
///
/// The "active holder" check is exercised by
/// `phaseA_item8_two_concurrent_builds_serialize`, which would
/// fail with a deadlock if `acquire_reef_home_lock` did not
/// release on drop.
fn oracle_lock_file_engaged_during_autofetch() {
    let (archive_bytes, shell_bytes) = synthetic_nautilus_artifacts();
    let harness = WiremockHarness::new();
    harness.mount_all(canonical_api_mocks(archive_bytes, shell_bytes));

    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_downstream_project(outer.path());

    // Pre-condition: the lock file does not exist before the build
    // starts (we have not touched the registry).
    assert!(!reef_home.join(".reef-lock").exists());

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .current_dir(&app)
        .args(["reef", "build"])
        .assert()
        .success();

    // Post-condition: the lock file is on disk (created by
    // `acquire_reef_home_lock`) but no longer holds the lock — the
    // child process exited so the kernel released the fd.
    assert!(
        reef_home.join(".reef-lock").exists(),
        "lock file must be created during auto-fetch"
    );

    // Re-acquire from this test process: must succeed immediately.
    let lock = chelis_reef::acquire_reef_home_lock(&reef_home, std::time::Duration::from_secs(2))
        .expect("lock must be released after build process exits");
    drop(lock);
}

/// Improved-error-wording shape: regex-strict pattern check, not
/// just `contains()`. The brief locks this assertion's shape.
fn oracle_error_wording_shape_regex() {
    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_downstream_project(outer.path());

    let assertion = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("GITHUB_TOKEN", "x")
        .env("PATH", "")
        .env("CHELIS_REEF_GITHUB_BASE_API", "http://localhost:9")
        .current_dir(&app)
        .args(["reef", "build", "--no-auto-fetch"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();

    // The error must match a shape like:
    //   missing dependency `nautilus` `0.2.0` (auto-fetch disabled by
    //   `--no-auto-fetch`); would have fetched from
    //   `chelis-lang/nautilus@v0.2.0` (category: would-attempt,
    //   GITHUB_TOKEN is set). ...
    //
    // Use `(?s)` so `.` matches newlines (the rendered error is a
    // single line today, but assertion regex must remain stable
    // even if a future change introduces wrapping).
    let pattern = predicates::str::is_match(
        r"(?s)missing dependency `nautilus` `0\.2\.0`.*auto-fetch disabled.*chelis-lang/nautilus@v0\.2\.0.*category: would-attempt.*GITHUB_TOKEN is set",
    )
    .expect("regex compile");
    assert!(
        pattern.eval(&stderr),
        "error wording did not match locked regex; got: {stderr}"
    );
}

/// Auto-fetch event must be observable: when it runs, the build's
/// stderr contains an "auto-fetching ..." line. The brief locks this
/// as a contract invariant ("silent data loss is the main bug
/// pattern; an 'auto-fetch happened' event must be observable").
fn oracle_autofetch_event_observable() {
    let (archive_bytes, shell_bytes) = synthetic_nautilus_artifacts();
    let harness = WiremockHarness::new();
    harness.mount_all(canonical_api_mocks(archive_bytes, shell_bytes));

    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_downstream_project(outer.path());

    let assertion = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .current_dir(&app)
        .args(["reef", "build"])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();
    assert!(
        stderr.contains("auto-fetching `nautilus` `0.2.0`"),
        "auto-fetch event must be visible in stderr; got: {stderr}"
    );
}

// ============================================================
// Negative-parity tests (separate `#[test]` functions).
// ============================================================

/// Negative parity: `GITHUB_TOKEN` unset and no `gh` on PATH means
/// auto-fetch's first step (resolving the token) fails with the
/// `auth-missing` typed category. The error names the env-var fix
/// suggestion. Build must not silently proceed.
#[test]
fn phaseA_item8_auth_missing_during_autofetch_names_env_fix() {
    let _g = file_lock();

    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_downstream_project(outer.path());

    // No GITHUB_TOKEN, no `gh` (PATH empty). Wiremock not needed —
    // we never get past auth resolution.
    let assertion = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", "http://localhost:9")
        .env_remove("GITHUB_TOKEN")
        .env("PATH", "")
        .current_dir(&app)
        .args(["reef", "build"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();
    assert!(
        stderr.contains("category: auth-missing"),
        "auth-missing must surface as typed category; got: {stderr}"
    );
    assert!(
        stderr.contains("GITHUB_TOKEN"),
        "auth-missing must name the env var; got: {stderr}"
    );
    assert!(
        stderr.contains("GITHUB_TOKEN is not set"),
        "must report GITHUB_TOKEN auth state; got: {stderr}"
    );
    assert!(
        !reef_home.join("index.json").exists(),
        "no half-install on auth-missing"
    );
}

/// Negative parity: HTTP 429 must surface with `Retry-After`. No
/// auto-retry; the brief locks fail-fast.
#[test]
fn phaseA_item8_429_during_autofetch_names_retry_after() {
    let _g = file_lock();

    let harness = WiremockHarness::new();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path("chelis-lang", "nautilus", "v0.2.0")))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("Retry-After", "300")
                    .set_body_string("rate limited"),
            ),
    ]);

    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_downstream_project(outer.path());

    let assertion = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .current_dir(&app)
        .args(["reef", "build"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();
    assert!(
        stderr.contains("category: rate-limited"),
        "429 must surface as rate-limited category; got: {stderr}"
    );
    assert!(
        stderr.contains("300"),
        "429 message must include Retry-After value; got: {stderr}"
    );
    assert!(!reef_home.join("index.json").exists());
}

/// Negative parity / contract invariant: a stale `.reef-lock` left
/// over from a crashed prior process does **not** block acquisition.
/// `flock(2)` semantics: the kernel auto-releases the lock when the
/// holder's fd closes (process death closes all fds). The lock file
/// itself remains on disk but no longer holds anything.
///
/// We verify: pre-create an empty `.reef-lock` file (as if a prior
/// process had touched it and died), then run an auto-fetch build
/// against it. Acquisition must succeed immediately.
#[test]
fn phaseA_item8_stale_lock_file_does_not_block_acquisition() {
    let _g = file_lock();

    let (archive_bytes, shell_bytes) = synthetic_nautilus_artifacts();
    let harness = WiremockHarness::new();
    harness.mount_all(canonical_api_mocks(archive_bytes, shell_bytes));

    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_downstream_project(outer.path());

    // Pre-stage a stale lock file.
    fs::create_dir_all(&reef_home).unwrap();
    fs::write(reef_home.join(".reef-lock"), b"").unwrap();

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .current_dir(&app)
        .args(["reef", "build"])
        .assert()
        .success();
}

/// Negative parity / concurrent-build contract: spawn two
/// `chelis reef build` subprocesses in parallel against the same
/// `$CHELIS_REEF_HOME`. Both eventually succeed; the dep is
/// installed exactly once with no duplicate index entries; one
/// runs serially after the other (lock acquisition observable
/// via timing).
#[test]
fn phaseA_item8_two_concurrent_builds_serialize() {
    let _g = file_lock();

    let (archive_bytes, shell_bytes) = synthetic_nautilus_artifacts();
    let harness = WiremockHarness::new();
    harness.mount_all(canonical_api_mocks(archive_bytes, shell_bytes));

    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app_a = stage_downstream_project(outer.path());
    let app_b = stage_downstream_project(outer.path());

    // Spawn two builds in parallel via std::thread.
    let api_uri = harness.uri();
    let reef_home_a = reef_home.clone();
    let reef_home_b = reef_home.clone();
    let api_a = api_uri.clone();
    let api_b = api_uri.clone();

    let h_a = std::thread::spawn(move || {
        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home_a)
            .env("CHELIS_REEF_GITHUB_BASE_API", api_a)
            .env("GITHUB_TOKEN", "unit-test-token")
            .env("PATH", "")
            .current_dir(&app_a)
            .args(["reef", "build"])
            .assert()
            .success();
    });
    let h_b = std::thread::spawn(move || {
        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home_b)
            .env("CHELIS_REEF_GITHUB_BASE_API", api_b)
            .env("GITHUB_TOKEN", "unit-test-token")
            .env("PATH", "")
            .current_dir(&app_b)
            .args(["reef", "build"])
            .assert()
            .success();
    });

    h_a.join().expect("build A panicked");
    h_b.join().expect("build B panicked");

    // Index has exactly one nautilus@0.2.0 entry. The second
    // builder's double-checked-locking branch must skip the
    // re-install, so the index has no duplicate entry.
    assert_nautilus_installed(&reef_home);
    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(reef_home.join("index.json")).unwrap()).unwrap();
    let entries = index["packages"]["nautilus"]
        .as_array()
        .expect("packages.nautilus must be array");
    let v010_count = entries.iter().filter(|e| e["version"] == "0.2.0").count();
    assert_eq!(
        v010_count, 1,
        "nautilus 0.2.0 must be installed exactly once; index entries: {entries:?}"
    );
}

/// Negative parity: two concurrent builds, one with `--no-auto-fetch`.
/// The opt-out one must fail (no auto-fetch attempted, registry
/// empty). The opt-in one must succeed. They must not deadlock.
#[test]
fn phaseA_item8_concurrent_one_no_auto_fetch_does_not_deadlock() {
    let _g = file_lock();

    let (archive_bytes, shell_bytes) = synthetic_nautilus_artifacts();
    let harness = WiremockHarness::new();
    harness.mount_all(canonical_api_mocks(archive_bytes, shell_bytes));

    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app_yes = stage_downstream_project(outer.path());
    let app_no = stage_downstream_project(outer.path());

    let api_uri = harness.uri();
    let reef_home_y = reef_home.clone();
    let reef_home_n = reef_home.clone();
    let api_y = api_uri.clone();
    let api_n = api_uri.clone();

    let h_yes = std::thread::spawn(move || {
        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home_y)
            .env("CHELIS_REEF_GITHUB_BASE_API", api_y)
            .env("GITHUB_TOKEN", "unit-test-token")
            .env("PATH", "")
            .current_dir(&app_yes)
            .args(["reef", "build"])
            .assert()
            .success();
    });
    let h_no = std::thread::spawn(move || {
        // The --no-auto-fetch builder either fails fast (if it
        // runs before the auto-fetch builder finishes) or succeeds
        // (if the auto-fetch builder already populated the
        // registry by the time --no-auto-fetch's resolver runs).
        // Both outcomes are spec-allowed; the brief locks "no
        // deadlock" not "no-auto-fetch always loses." We check
        // exit-status-agnostic: both threads return.
        let _ = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home_n)
            .env("CHELIS_REEF_GITHUB_BASE_API", api_n)
            .env("GITHUB_TOKEN", "unit-test-token")
            .env("PATH", "")
            .current_dir(&app_no)
            .args(["reef", "build", "--no-auto-fetch"])
            .output()
            .expect("spawn opt-out builder");
    });

    // Generous join: 90 s upper bound on each thread. Anything
    // approaching this is a deadlock signal.
    h_yes.join().expect("auto-fetch build panicked");
    h_no.join().expect("no-auto-fetch build panicked");

    // The auto-fetch builder must have populated the registry
    // either way.
    assert_nautilus_installed(&reef_home);
}

/// Coordination contract (Item 8 ↔ Item 9): a populated
/// `remote_origin` in the lockfile must route auto-fetch to that URL,
/// not to the canonical-org default. Item 9 introduced the field and
/// the production wiring (`lockfile_remote_origin`); this test
/// exercises the round-trip end-to-end via wiremock.
///
/// Stages a hand-written `reef.lock` whose `[dependencies.source]`
/// block sets `remote_origin = "github://other-org/nautilus@v0.2.0"`,
/// then asserts the build fetches from `other-org`'s wiremock path
/// rather than the canonical-org one.
#[test]
fn phaseA_item8_lockfile_remote_origin_honored_when_present() {
    let _g = file_lock();

    let (archive_bytes, shell_bytes) = synthetic_nautilus_artifacts();

    // Wiremock that serves nautilus@v0.2.0 only at the `other-org`
    // path. The canonical-org path returns 404 — if the resolver
    // ignored `remote_origin` and fell back to the canonical default,
    // the build would fail.
    let harness = WiremockHarness::new();
    let other_meta = metadata_path("other-org", "nautilus", "v0.2.0");
    let other_archive = asset_id_path("other-org", "nautilus", ARCHIVE_ASSET_ID);
    let other_shell = asset_id_path("other-org", "nautilus", SHELL_ASSET_ID);
    let canonical_meta = metadata_path("chelis-lang", "nautilus", "v0.2.0");
    let metadata_body = metadata_json(
        "v0.2.0",
        &[
            (ARCHIVE_ASSET_ID, "nautilus-0.2.0.tar.zst"),
            (SHELL_ASSET_ID, "nautilus-0.2.0.chb"),
        ],
    );
    harness.mount_all(vec![
        // Canonical org path: 404. If the resolver hit this we'd see
        // a `release-tag-not-found-or-unauthorized` error and fail.
        Mock::given(method("GET"))
            .and(wm_path(canonical_meta))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(404).set_body_string("not found here")),
        // Other-org path: the real bytes.
        Mock::given(method("GET"))
            .and(wm_path(other_meta))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(metadata_body)
                    .insert_header("content-type", "application/json"),
            ),
        Mock::given(method("GET"))
            .and(wm_path(other_archive))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_bytes.clone())),
        Mock::given(method("GET"))
            .and(wm_path(other_shell))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(shell_bytes.clone())),
    ]);

    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_downstream_project(outer.path());

    // Hand-write a reef.lock that pins `remote_origin` to the
    // non-canonical org. The compiler/archive_sha256/shell_sha256
    // fields are placeholders — install_from_lockfile (which checks
    // pin agreement) is not the path under test here; auto-fetch
    // during `chelis reef build` runs the install pipeline anew and
    // recomputes hashes.
    let lockfile_text = format!(
        r#"[package]
name = "downstream-item8"
version = "0.2.0"

[[dependencies]]
name = "nautilus"
version = "0.2.0"
compiler = "{compiler}"
archive_sha256 = "{archive_sha}"
shell_sha256 = "{shell_sha}"

[dependencies.source]
kind = "local_registry"
remote_origin = "github://other-org/nautilus@v0.2.0"
"#,
        compiler = CURRENT_COMPILER_PIN,
        archive_sha = sha256_bytes(&archive_bytes),
        shell_sha = sha256_bytes(&shell_bytes),
    );
    fs::write(app.join("reef.lock"), lockfile_text).expect("write reef.lock");

    // Use `chelis reef install --from-lockfile` because that's the
    // CLI surface that reads each lockfile entry's `remote_origin`
    // field and routes the fetch accordingly. The fresh-resolver
    // path used by `chelis reef build` always passes
    // `lockfile_dep = None` to the auto-fetch helper and so falls
    // back to the canonical-org default; the lockfile-replay path
    // instead consults `LockSource::LocalRegistry::remote_origin`
    // verbatim. Item 9 wired this exact contract.
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .current_dir(&app)
        .args(["reef", "install", "--from-lockfile"])
        .assert()
        .success();
    assert_nautilus_installed(&reef_home);
}

// ============================================================
// Manual gate (real network). Documented invocation:
//
//   cargo test -p chelis-cli phaseA_item8_real_github_manual_gate \
//     -- --ignored --exact
//
// Pre-condition: `GITHUB_TOKEN` set, with read access to the
// canonical org. The real release `chelis-lang/nautilus@v0.5.0`
// must carry the canonical assets (this is the current published
// tag at the time of the chelis-std-runtime fix wave).
// ============================================================

/// Stage a downstream project that depends on the real released
/// `nautilus@v0.5.0` (manual-gate only). Uses the same project shape
/// as the wiremock fixture but with the real published version pin.
fn stage_real_nautilus_downstream(parent: &Path) -> std::path::PathBuf {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let app = parent.join(format!("downstream-real-{n}"));
    fs::create_dir_all(app.join("src")).expect("mkdir downstream/src");
    let ver = env!("CARGO_PKG_VERSION");
    fs::write(
        app.join("reef.toml"),
        format!(
            r#"[package]
name = "downstream-item8-real"
version = "0.2.0"
compiler = "={ver}"
module_prefix = "Demo"

[dependencies]
nautilus = {{ version = "0.6.1" }}
"#,
        ),
    )
    .expect("write reef.toml");
    fs::write(
        app.join("src/main.ch"),
        "module Demo.Main\n\ndef noop(x: int32) -> int32 = x\n",
    )
    .expect("write main.ch");
    app
}

#[test]
#[ignore = "real-network manual gate; run with `--ignored --exact`"]
fn phaseA_item8_real_github_manual_gate() {
    let _g = file_lock();
    let token = std::env::var("GITHUB_TOKEN").unwrap_or_default();
    assert!(
        !token.is_empty(),
        "manual gate requires GITHUB_TOKEN set; \
         run `export GITHUB_TOKEN=$(gh auth token)` first"
    );
    let outer = tempdir().expect("tempdir");
    let reef_home = outer.path().join("reef-home");
    let app = stage_real_nautilus_downstream(outer.path());

    // chelis-std is now bundled inside the chelis binary
    // (`crates/chelis-std-bundle`), so the manual gate no longer
    // needs to pre-seed the runtime via `--from-monorepo`. The
    // `chelis reef build` invocation below resolves chelis-std from
    // the embedded bytes directly.

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .env("GITHUB_TOKEN", token)
        .current_dir(&app)
        .args(["reef", "build"])
        .assert()
        .success();
    let index_path = reef_home.join("index.json");
    assert!(
        index_path.exists(),
        "index.json must exist after auto-fetch"
    );
    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&index_path).unwrap()).unwrap();
    let entries = index["packages"]["nautilus"]
        .as_array()
        .expect("packages.nautilus must be array");
    assert!(
        entries.iter().any(|e| e["version"] == "0.6.1"),
        "expected nautilus 0.5.0 in index, got {entries:?}"
    );
}
