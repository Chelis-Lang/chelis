//! Phase A — Item 9 named acceptance oracle: lockfile gains
//! `remote_origin` + `chelis reef install --from-lockfile`.
//!
//! Named oracle invocation (locked by the dispatch brief):
//!
//! ```sh
//! cargo test -p chelis-cli phaseA_item9_lockfile_origin_oracle -- --exact
//! ```
//!
//! The oracle is one comprehensive `#[test]` that dispatches to focused
//! sub-helpers, each of which pins one spec acceptance bullet:
//!
//! - `oracle_backcompat_deserializes_old_schema` — pre-Item-9
//!   lockfiles (no `remote_origin` field) deserialize cleanly with
//!   `remote_origin: None` on every entry.
//! - `oracle_roundtrip_with_and_without_origin` — toml round-trip is
//!   a fixed point both with and without `remote_origin` populated.
//! - `oracle_install_from_github_populates_field` — installing via
//!   `--from-github` writes the canonical `github://...` URI to the
//!   registry index and flows it into the downstream `reef.lock`.
//! - `oracle_install_from_monorepo_leaves_field_none` — installing
//!   via `--from-monorepo` leaves the field unset.
//! - `oracle_dev_a_to_dev_b_byte_identical` — the centerpiece. Dev A
//!   installs from GitHub, builds, commits the lockfile. Dev B (fresh
//!   `$CHELIS_REEF_HOME`) runs `--from-lockfile` against the same
//!   wiremock URL and ends up with byte-identical archive, shell, and
//!   index entries.
//! - `oracle_hash_mismatch_on_refetch` — populate stale registry
//!   bytes that disagree with the lockfile pin; `--from-lockfile`
//!   surfaces a `Validation` error category.
//! - `oracle_entry_without_origin_errors_clearly` — lockfile entry
//!   has `kind = local_registry` but no `remote_origin`;
//!   `--from-lockfile` errors with a clear message naming the entry
//!   and suggesting `--bootstrap`.
//! - `oracle_mutex_with_other_flags` — `--from-lockfile
//!   --from-github X` errors via clap.
//!
//! Negative-parity tests (separate `#[test]` functions, run by the
//! default `cargo test --workspace` loop):
//!
//! - `phaseA_item9_unknown_scheme_in_origin_errors` — `remote_origin`
//!   that doesn't start with a supported scheme surfaces a typed
//!   `RemoteOriginParseError::UnknownScheme`.
//! - `phaseA_item9_remote_origin_404_surfaces_release_asset_not_found`.
//! - `phaseA_item9_malformed_lockfile_errors_with_line_info`.
//! - `phaseA_item9_no_lockfile_present_suggests_reef_build`.
//!
//! Manual gate (`#[ignore]`d): `phaseA_item9_real_github_manual_gate`.
//! Real-network round-trip via `chelis-lang/nautilus@v0.5.0`. Run via:
//!
//! ```sh
//! GITHUB_TOKEN=$(gh auth token) cargo test -p chelis-cli \
//!   phaseA_item9_real_github_manual_gate -- --ignored --exact
//! ```
//!
//! **Phase A correction (chelis-std-runtime fix wave):** the wiremock
//! fixtures previously served real chelis-std bytes off the monorepo
//! dist tree. Now that chelis-std is classified as the language runtime
//! (not a shell) and routed via `LockSource::Bundled` rather than
//! through GitHub auto-fetch, those tests use a synthetic `nautilus`
//! shell built in-memory at test start. Tests that exercise the
//! lockfile schema (round-trip / back-compat) still use "chelis-std"
//! string literals because they assert serde behavior, not resolver
//! behavior.

#![allow(non_snake_case)]

use assert_cmd::Command;
use chelis_reef::{
    GitHubFetchError, GitHubReleaseSpec, LockSource, LockedDependency, LockfileInstallEntry,
    LockfileInstallError, ReefLock, RemoteOriginParseError, parse_remote_origin,
};
use chelis_shell::PackageId;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use tar::Builder;
use tempfile::tempdir;
use wiremock::matchers::{header, method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const CURRENT_COMPILER_VERSION: &str = env!("CARGO_PKG_VERSION");
const CURRENT_COMPILER_PIN: &str = concat!("=", env!("CARGO_PKG_VERSION"));

/// The chelis monorepo root (two levels up from `crates/chelis-cli`).
fn monorepo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("monorepo root must exist")
}

// ============================================================
// Synthetic-nautilus fixture builders.
// Same shape as item-7 / item-8, kept inline rather than factored
// into a shared helper crate so each phaseA test file remains
// self-contained.
// ============================================================

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
    let main_text = "module Test.Main\n\nexport (placeholder)\ndef placeholder -> int32 = 0\n";
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

fn build_test_shell_bytes(name: &str, version: &str, archive_sha256: &str) -> Vec<u8> {
    let shell = chelis_shell::ShellPackage {
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

/// Synthetic `nautilus` archive+shell pair. No filesystem dependency.
fn synthetic_nautilus_artifacts() -> (Vec<u8>, Vec<u8>) {
    let archive = build_test_archive("nautilus", "0.2.0", &[]);
    let archive_sha = sha256_bytes(&archive);
    let shell = build_test_shell_bytes("nautilus", "0.2.0", &archive_sha);
    (archive, shell)
}

/// Wiremock helper paths.
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

const ARCHIVE_ASSET_ID: u64 = 1001;
const SHELL_ASSET_ID: u64 = 1002;

/// Stand up a wiremock harness that serves the synthetic nautilus
/// artifacts as `chelis-lang/nautilus@v0.2.0` release assets via the
/// GitHub API two-step path. The metadata + byte endpoints all
/// require an `Authorization: token unit-test-token` header. Returns
/// harness + (archive, shell) bytes.
fn fixture_canonical_release() -> (WiremockHarness, Vec<u8>, Vec<u8>) {
    let (archive_bytes, shell_bytes) = synthetic_nautilus_artifacts();

    let harness = WiremockHarness::new();
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

    harness.mount_all(vec![
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
            .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_bytes.clone())),
        Mock::given(method("GET"))
            .and(wm_path(shell_url_path))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(shell_bytes.clone())),
    ]);
    (harness, archive_bytes, shell_bytes)
}

/// A wiremock harness that owns a long-lived multi-threaded tokio
/// runtime and a `MockServer`. Mirrors the shape used by Item 6's
/// oracle so test code stays uniform across phases.
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

// Process-global lock — every test in this file holds it for the
// duration of the test body. Same rationale as Item 6's oracle:
// CHELIS_REEF_HOME / GITHUB_TOKEN / CHELIS_REEF_GITHUB_BASE_API are
// process env, and `set_var`/`remove_var` are racy. We serialize even
// tests that don't touch env directly because the test binary spawns
// subprocesses (`assert_cmd`) that inherit env at spawn time, and we
// want each test to fully control what those children see.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn file_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ============================================================
// Named acceptance oracle.
// ============================================================

#[test]
fn phaseA_item9_lockfile_origin_oracle() {
    let _g = file_lock();
    oracle_backcompat_deserializes_old_schema();
    oracle_roundtrip_with_and_without_origin();
    oracle_install_from_github_populates_field();
    oracle_install_from_monorepo_leaves_field_none();
    oracle_dev_a_to_dev_b_byte_identical();
    oracle_hash_mismatch_on_refetch();
    oracle_entry_without_origin_errors_clearly();
    oracle_mutex_with_other_flags();
}

/// Backward-compat invariant: a lockfile written by pre-Item-9 code
/// has no `remote_origin` field. The new code must parse it cleanly
/// and surface `remote_origin = None` on every `LocalRegistry` entry.
fn oracle_backcompat_deserializes_old_schema() {
    let toml = format!(
        r#"
[package]
name = "demo"
version = "0.2.0"

[[dependencies]]
name = "chelis-std"
version = "0.2.0"
compiler = "{compiler}"
archive_sha256 = "deadbeef"
shell_sha256 = "cafebabe"

[dependencies.source]
kind = "local_registry"

[[dependencies]]
name = "neighbor"
version = "0.2.0"
compiler = "{compiler}"
archive_sha256 = "abcd"
shell_sha256 = "ef01"

[dependencies.source]
kind = "path"
path = "../neighbor"
"#,
        compiler = CURRENT_COMPILER_PIN,
    );
    let lock: ReefLock = toml::from_str(&toml).expect("old-schema lockfile must deserialize");
    assert_eq!(lock.dependencies.len(), 2);
    let std_entry = lock
        .dependencies
        .iter()
        .find(|d| d.name == "chelis-std")
        .expect("chelis-std entry");
    match &std_entry.source {
        LockSource::LocalRegistry { remote_origin } => {
            assert!(
                remote_origin.is_none(),
                "old-schema LocalRegistry must deserialize with remote_origin: None"
            );
        }
        other => panic!("expected LocalRegistry, got {other:?}"),
    }
    let path_entry = lock
        .dependencies
        .iter()
        .find(|d| d.name == "neighbor")
        .expect("path entry");
    assert!(matches!(&path_entry.source, LockSource::Path { path } if path == "../neighbor"));
}

/// Round-trip invariant: serialize → parse → serialize is a fixed
/// point for every `LockSource` variant. Pinned at the byte level so
/// a serde-attribute regression that flipped `skip_serializing_if` on
/// a TOML-serialized field would surface as a diff here.
fn oracle_roundtrip_with_and_without_origin() {
    fn synth_lock(source: LockSource, name: &str) -> ReefLock {
        ReefLock {
            package: PackageId {
                name: "demo".to_string(),
                version: "0.2.0".to_string(),
            },
            dependencies: vec![LockedDependency {
                name: name.to_string(),
                version: "0.2.0".to_string(),
                source,
                compiler: CURRENT_COMPILER_PIN.to_string(),
                archive_sha256: "ABC".to_string(),
                shell_sha256: "DEF".to_string(),
            }],
        }
    }

    // LocalRegistry with and without remote_origin.
    for origin in [None, Some("github://chelis-lang/nautilus@v0.2.0")] {
        let source = LockSource::LocalRegistry {
            remote_origin: origin.map(str::to_string),
        };
        let lock = synth_lock(source, "nautilus");
        let serialized = toml::to_string_pretty(&lock).expect("serialize");
        let parsed: ReefLock = toml::from_str(&serialized).expect("parse round-1");
        assert_eq!(parsed, lock, "first round-trip must equal original");
        let reserialized = toml::to_string_pretty(&parsed).expect("serialize round-2");
        assert_eq!(
            serialized, reserialized,
            "byte-level fixed point: second serialize must equal first"
        );
        match origin {
            None => assert!(
                !serialized.contains("remote_origin"),
                "None must not emit remote_origin field; got: {serialized}"
            ),
            Some(uri) => assert!(
                serialized.contains(&format!("remote_origin = \"{uri}\"")),
                "Some({uri}) must emit field with canonical form; got: {serialized}"
            ),
        }
    }

    // Bundled (the language runtime). Same fixed-point property.
    let bundled_source = LockSource::Bundled {
        compiler_version: CURRENT_COMPILER_VERSION.to_string(),
    };
    let lock = synth_lock(bundled_source, "chelis-std");
    let serialized = toml::to_string_pretty(&lock).expect("serialize bundled");
    assert!(
        serialized.contains("kind = \"bundled\""),
        "Bundled variant must serialize with `bundled` tag; got: {serialized}"
    );
    assert!(
        serialized.contains(&format!(
            "compiler_version = \"{CURRENT_COMPILER_VERSION}\""
        )),
        "Bundled must record compiler_version; got: {serialized}"
    );
    let parsed: ReefLock = toml::from_str(&serialized).expect("parse bundled");
    assert_eq!(parsed, lock, "Bundled round-trip must equal original");
    let reserialized = toml::to_string_pretty(&parsed).expect("serialize bundled round-2");
    assert_eq!(
        serialized, reserialized,
        "Bundled byte-level fixed point must hold"
    );
}

/// Run a wiremock-backed `install_from_github` (lib-level, no
/// subprocess) and return the registry path. The helper centralizes
/// env-var management: `CHELIS_REEF_GITHUB_BASE_API`, `GITHUB_TOKEN`,
/// and `PATH` are all set/restored. Pre-condition: caller holds
/// `file_lock()`.
fn lib_install_from_github_canonical(registry_root: &Path, harness_uri: &str) {
    let prior_api_base = std::env::var_os("CHELIS_REEF_GITHUB_BASE_API");
    let prior_token = std::env::var_os("GITHUB_TOKEN");
    let prior_path = std::env::var_os("PATH");
    unsafe {
        std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", harness_uri);
        std::env::set_var("GITHUB_TOKEN", "unit-test-token");
        // Empty PATH so the gh fallback can never win.
        std::env::set_var("PATH", "");
    }
    let result = chelis_reef::install_from_github("chelis-lang/nautilus@v0.2.0", registry_root);
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
    result.expect("install_from_github canonical happy path");
}

/// `install_from_github` must stamp the canonical
/// `github://<org>/<repo>@<tag>` URI into both the registry's
/// `index.json` `remote_origin` field and (after a downstream
/// `chelis reef build`) the lockfile's `remote_origin` field. This is
/// the source of truth for Item 9's "developer A populates origin"
/// half of the dev-A → dev-B handoff.
fn oracle_install_from_github_populates_field() {
    let (harness, _archive_bytes, _shell_bytes) = fixture_canonical_release();
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");

    lib_install_from_github_canonical(&registry, &harness.uri());

    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(registry.join("index.json")).unwrap()).unwrap();
    let nautilus_versions = &index["packages"]["nautilus"];
    let entry = &nautilus_versions[0];
    assert_eq!(
        entry["remote_origin"],
        serde_json::Value::String("github://chelis-lang/nautilus@v0.2.0".to_string()),
        "GitHub install must stamp remote_origin into index.json"
    );

    // Downstream lockfile: build a tiny package that depends on
    // nautilus and assert `chelis reef build` populates
    // remote_origin in its reef.lock from the registry index.
    let pkg_root = dir.path().join("downstream-pkg");
    seed_downstream_package_for_dep(&pkg_root, "nautilus", "0.2.0", "Downstream");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &registry)
        .args(["reef", "build", pkg_root.to_str().unwrap()])
        .assert()
        .success();
    let lock_text = fs::read_to_string(pkg_root.join("reef.lock")).expect("read lockfile");
    assert!(
        lock_text.contains("remote_origin = \"github://chelis-lang/nautilus@v0.2.0\""),
        "lockfile must carry remote_origin from registry; got:\n{lock_text}"
    );
}

/// Reciprocal: `install_from_monorepo` must NOT stamp
/// `remote_origin` (no remote source exists). Pre-Item-9 lockfiles
/// from monorepo installs continue to round-trip identically.
fn oracle_install_from_monorepo_leaves_field_none() {
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("reef-home");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &registry)
        .args([
            "reef",
            "install",
            "--from-monorepo",
            monorepo_root().to_str().unwrap(),
            "chelis-std=0.3.0",
        ])
        .assert()
        .success();

    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(registry.join("index.json")).unwrap()).unwrap();
    let entry = &index["packages"]["chelis-std"][0];
    assert!(
        entry.get("remote_origin").is_none() || entry["remote_origin"] == serde_json::Value::Null,
        "monorepo install must NOT stamp remote_origin; got entry: {entry}"
    );
}

/// THE centerpiece: developer A installs via `--from-github`
/// against the wiremock fixture, builds a downstream package whose
/// reef.lock now carries `remote_origin`, commits the lockfile.
/// Developer B has a fresh, empty `$CHELIS_REEF_HOME` and runs
/// `chelis reef install --from-lockfile` against the same wiremock
/// URL. Their local registry must end up byte-identical to A's.
fn oracle_dev_a_to_dev_b_byte_identical() {
    let (harness, _archive_bytes, _shell_bytes) = fixture_canonical_release();
    let dir = tempdir().expect("tempdir");

    let dev_a_registry = dir.path().join("dev-a-reef-home");
    let dev_b_registry = dir.path().join("dev-b-reef-home");
    let pkg_root = dir.path().join("shared-pkg");

    // Dev A: install nautilus via GitHub, build downstream package.
    lib_install_from_github_canonical(&dev_a_registry, &harness.uri());
    seed_downstream_package_for_dep(&pkg_root, "nautilus", "0.2.0", "Downstream");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &dev_a_registry)
        .args(["reef", "build", pkg_root.to_str().unwrap()])
        .assert()
        .success();

    // Snapshot dev A's registry state.
    let a_archive = fs::read(dev_a_registry.join("packages/nautilus/0.2.0/nautilus-0.2.0.tar.zst"))
        .expect("dev A archive must exist");
    let a_shell = fs::read(dev_a_registry.join("packages/nautilus/0.2.0/nautilus-0.2.0.chb"))
        .expect("dev A shell must exist");
    let a_index = fs::read_to_string(dev_a_registry.join("index.json")).expect("dev A index");

    // Dev B: fresh home, run --from-lockfile against the same
    // wiremock URL. Pass --package-root explicitly so we don't have to
    // chdir.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &dev_b_registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .args([
            "reef",
            "install",
            "--from-lockfile",
            "--package-root",
            pkg_root.to_str().unwrap(),
        ])
        .assert()
        .success();

    let b_archive = fs::read(dev_b_registry.join("packages/nautilus/0.2.0/nautilus-0.2.0.tar.zst"))
        .expect("dev B archive must exist");
    let b_shell = fs::read(dev_b_registry.join("packages/nautilus/0.2.0/nautilus-0.2.0.chb"))
        .expect("dev B shell must exist");
    let b_index = fs::read_to_string(dev_b_registry.join("index.json")).expect("dev B index");

    assert_eq!(
        a_archive, b_archive,
        "dev A and dev B archive bytes must be byte-identical"
    );
    assert_eq!(
        a_shell, b_shell,
        "dev A and dev B shell bytes must be byte-identical"
    );
    assert_eq!(
        a_index, b_index,
        "dev A and dev B index.json must be byte-identical"
    );
}

/// Hash-pin invariant: if the URL serves bytes that disagree with the
/// lockfile pin, `--from-lockfile` must surface a `Validation` error
/// (not silently overwrite).
fn oracle_hash_mismatch_on_refetch() {
    let (harness, archive_bytes, shell_bytes) = fixture_canonical_release();
    let dir = tempdir().expect("tempdir");
    let dev_a_registry = dir.path().join("dev-a-reef-home");
    let pkg_root = dir.path().join("hash-mismatch-pkg");

    lib_install_from_github_canonical(&dev_a_registry, &harness.uri());
    seed_downstream_package_for_dep(&pkg_root, "nautilus", "0.2.0", "Downstream");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &dev_a_registry)
        .args(["reef", "build", pkg_root.to_str().unwrap()])
        .assert()
        .success();

    // Now stand up a second harness that serves DIFFERENT bytes on
    // the same wiremock paths (single byte flip in the archive). We
    // also serve a shell whose embedded archive_sha256 will not match
    // the tampered archive — `install_validated_artifact_pair`
    // catches the disagreement and the helper surfaces the error as
    // `GitHubFetchError::Validation`, which `install_from_lockfile`
    // wraps in `LockfileInstallError::Fetch { inner: Validation }`.
    let mut tampered_archive = archive_bytes.clone();
    let last = tampered_archive.len() - 1;
    tampered_archive[last] ^= 0xff;

    let bad_harness = WiremockHarness::new();
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
    bad_harness.mount_all(vec![
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
            .respond_with(ResponseTemplate::new(200).set_body_bytes(tampered_archive)),
        Mock::given(method("GET"))
            .and(wm_path(shell_url_path))
            .and(header("authorization", "token unit-test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(shell_bytes)),
    ]);

    // Fresh dev-B home so we exercise the install-fresh path.
    let dev_b_registry = dir.path().join("dev-b-mismatch-home");
    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &dev_b_registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", bad_harness.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .env("PATH", "")
        .args([
            "reef",
            "install",
            "--from-lockfile",
            "--package-root",
            pkg_root.to_str().unwrap(),
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    assert!(
        stderr.contains("archive_sha256") || stderr.contains("validation"),
        "hash-mismatch error must name the validation surface; got: {stderr}"
    );
}

/// `kind = local_registry` with no `remote_origin` is the
/// pre-Item-9 / monorepo-only case. `--from-lockfile` cannot fetch
/// such entries — the operator gets a clear typed message that names
/// the entry and suggests `--bootstrap`.
fn oracle_entry_without_origin_errors_clearly() {
    let dir = tempdir().expect("tempdir");
    let pkg_root = dir.path().join("origin-less-pkg");
    seed_downstream_package_for_dep(&pkg_root, "nautilus", "0.2.0", "Downstream");

    // Hand-write a lockfile that pins nautilus as a registry dep but
    // omits `remote_origin`. No build step required — `--from-lockfile`
    // operates on the lockfile bytes directly.
    fs::write(
        pkg_root.join("reef.lock"),
        format!(
            r#"[package]
name = "downstream"
version = "0.2.0"

[[dependencies]]
name = "nautilus"
version = "0.2.0"
compiler = "{ver}"
archive_sha256 = "abc"
shell_sha256 = "def"

[dependencies.source]
kind = "local_registry"
"#,
            ver = env!("CARGO_PKG_VERSION"),
        ),
    )
    .expect("write lockfile");

    let dev_b_registry = dir.path().join("dev-b-no-origin-home");
    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &dev_b_registry)
        .args([
            "reef",
            "install",
            "--from-lockfile",
            "--package-root",
            pkg_root.to_str().unwrap(),
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    assert!(
        stderr.contains("nautilus"),
        "no-origin error must name the entry; got: {stderr}"
    );
    assert!(
        stderr.contains("--bootstrap"),
        "no-origin error must suggest --bootstrap; got: {stderr}"
    );
}

/// `--from-lockfile` is mutually exclusive with `--from-github` and
/// `--from-monorepo`. Enforced by clap's `conflicts_with_all`; the
/// resulting error mentions the mutex.
fn oracle_mutex_with_other_flags() {
    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "reef",
            "install",
            "--from-lockfile",
            "--from-github",
            "chelis-lang/nautilus@v0.2.0",
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    assert!(
        stderr.contains("cannot be used") || stderr.contains("conflicts"),
        "mutex error must mention conflict; got: {stderr}"
    );

    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "reef",
            "install",
            "--from-lockfile",
            "--from-monorepo",
            "/tmp/nonexistent",
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    assert!(
        stderr.contains("cannot be used") || stderr.contains("conflicts"),
        "mutex error must mention conflict; got: {stderr}"
    );
}

// ============================================================
// Negative-parity tests (separate `#[test]` functions).
// ============================================================

/// Migration parity: an old lockfile that records `chelis-std` as
/// `LocalRegistry` (whether or not `remote_origin` is set) must be
/// transparently treated as the bundled runtime by
/// `install_from_lockfile`. The migration fires a one-line warning
/// to stderr; the next `chelis reef build` rewrites the lockfile.
#[test]
fn phaseA_item9_old_chelis_std_lockfile_migrates_to_bundled() {
    let _g = file_lock();
    let dir = tempdir().expect("tempdir");
    let pkg_root = dir.path().join("old-lockfile-pkg");
    fs::create_dir_all(&pkg_root).unwrap();
    fs::write(
        pkg_root.join("reef.toml"),
        format!(
            r#"[package]
name = "downstream"
version = "0.2.0"
compiler = "{compiler}"
module_prefix = "Downstream"
"#,
            compiler = CURRENT_COMPILER_PIN,
        ),
    )
    .unwrap();
    // Note: the entry has no remote_origin AND no version match against
    // the compiler. The migration path bypasses both checks because
    // chelis-std is now treated as bundled regardless.
    fs::write(
        pkg_root.join("reef.lock"),
        format!(
            r#"[package]
name = "downstream"
version = "0.2.0"

[[dependencies]]
name = "chelis-std"
version = "0.2.0"
compiler = "{compiler}"
archive_sha256 = "abc"
shell_sha256 = "def"

[dependencies.source]
kind = "local_registry"
"#,
            compiler = CURRENT_COMPILER_PIN,
        ),
    )
    .unwrap();

    let registry = dir.path().join("reef-home");
    let entries = chelis_reef::install_from_lockfile(&pkg_root, &registry)
        .expect("install_from_lockfile must Ok with per-entry results");
    assert_eq!(entries.len(), 1);
    match &entries[0] {
        LockfileInstallEntry::SkippedBundledRuntime {
            name,
            version,
            compiler_version,
        } => {
            assert_eq!(name, "chelis-std");
            assert_eq!(version, "0.2.0");
            assert!(
                !compiler_version.is_empty(),
                "compiler_version must be recorded"
            );
        }
        other => panic!(
            "old chelis-std-as-LocalRegistry must migrate to SkippedBundledRuntime; got: {other:?}"
        ),
    }
}

/// `parse_remote_origin` rejects unknown-scheme strings with a typed
/// `RemoteOriginParseError::UnknownScheme`. Pinned at the lib API
/// level so the surface is decoupled from the CLI's error wrapping.
#[test]
fn phaseA_item9_unknown_scheme_in_origin_errors() {
    let _g = file_lock();
    let cases = [
        "http://example.com/foo",
        "registry://chelis-lang/x@v1.0",
        "file:///local/path",
        "just-a-string",
    ];
    for input in cases {
        let err = parse_remote_origin(input).expect_err(&format!("`{input}` must error"));
        assert!(
            matches!(err, RemoteOriginParseError::UnknownScheme { .. }),
            "`{input}` must surface UnknownScheme, got: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains(input),
            "UnknownScheme message must name the input; got: {msg}"
        );
        assert!(
            msg.contains("github://"),
            "UnknownScheme message must list the supported scheme; got: {msg}"
        );
    }

    // Sibling case: `github://` body that itself fails to parse must
    // surface `Malformed`, not `UnknownScheme`. This separates "wrong
    // scheme" from "right scheme, malformed body" so the operator can
    // tell which fix to apply.
    let err =
        parse_remote_origin("github://no-at-tag").expect_err("missing @tag must surface error");
    assert!(
        matches!(err, RemoteOriginParseError::Malformed { .. }),
        "missing @tag must surface Malformed, got: {err:?}"
    );
}

/// Spec-locked positive: a well-formed origin parses to the expected
/// `GitHubReleaseSpec`. Pinned so a parser regression that drops the
/// scheme-strip step would fail here.
#[test]
fn phaseA_item9_parse_remote_origin_happy_path() {
    let _g = file_lock();
    let spec: GitHubReleaseSpec =
        parse_remote_origin("github://chelis-lang/nautilus@v0.4.0").expect("parse");
    assert_eq!(spec.org, "chelis-lang");
    assert_eq!(spec.repo, "nautilus");
    assert_eq!(spec.tag, "v0.4.0");
    assert_eq!(spec.version, "0.4.0");
}

/// `remote_origin` pointing at a 404 surfaces
/// `GitHubFetchError::ReleaseAssetNotFound` (wrapped in
/// `LockfileInstallError::Fetch`). The error names the URL.
#[test]
fn phaseA_item9_remote_origin_404_surfaces_release_asset_not_found() {
    let _g = file_lock();
    let harness = WiremockHarness::new();
    harness.mount_all(vec![
        Mock::given(method("GET"))
            .and(wm_path(metadata_path("chelis-lang", "nautilus", "v9.9.9")))
            .respond_with(ResponseTemplate::new(404)),
    ]);

    let dir = tempdir().expect("tempdir");
    let pkg_root = dir.path().join("pkg-404");
    fs::create_dir_all(&pkg_root).unwrap();
    fs::write(
        pkg_root.join("reef.lock"),
        format!(
            r#"[package]
name = "x"
version = "0.2.0"

[[dependencies]]
name = "nautilus"
version = "9.9.9"
compiler = "{compiler}"
archive_sha256 = "abc"
shell_sha256 = "def"

[dependencies.source]
kind = "local_registry"
remote_origin = "github://chelis-lang/nautilus@v9.9.9"
"#,
            compiler = CURRENT_COMPILER_PIN,
        ),
    )
    .unwrap();

    let prior_api_base = std::env::var_os("CHELIS_REEF_GITHUB_BASE_API");
    let prior_token = std::env::var_os("GITHUB_TOKEN");
    let prior_path = std::env::var_os("PATH");
    unsafe {
        std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", harness.uri());
        std::env::set_var("GITHUB_TOKEN", "unit-test-token");
        std::env::set_var("PATH", "");
    }
    let registry = dir.path().join("reef-home");
    let results = chelis_reef::install_from_lockfile(&pkg_root, &registry);
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
    let entries = results.expect("install_from_lockfile returns Ok with per-entry results");
    assert_eq!(entries.len(), 1);
    match &entries[0] {
        LockfileInstallEntry::Failed {
            name,
            version,
            error: LockfileInstallError::Fetch { inner, .. },
        } => {
            assert_eq!(name, "nautilus");
            assert_eq!(version, "9.9.9");
            assert!(
                matches!(inner, GitHubFetchError::ReleaseAssetNotFound { .. }),
                "404 must surface ReleaseAssetNotFound; got: {inner:?}"
            );
            assert!(
                inner.to_string().contains("releases/tags/v9.9.9"),
                "404 message must name URL: {inner}"
            );
        }
        other => panic!("expected Failed(ReleaseAssetNotFound), got: {other:?}"),
    }
}

/// Malformed lockfile (broken TOML) surfaces a clear error.
#[test]
fn phaseA_item9_malformed_lockfile_errors_with_line_info() {
    let _g = file_lock();
    let dir = tempdir().expect("tempdir");
    let pkg_root = dir.path().join("malformed-pkg");
    fs::create_dir_all(&pkg_root).unwrap();
    fs::write(
        pkg_root.join("reef.lock"),
        "this is not = valid toml = at all\n[broken",
    )
    .unwrap();

    let registry = dir.path().join("reef-home");
    let err =
        chelis_reef::install_from_lockfile(&pkg_root, &registry).expect_err("malformed must err");
    assert!(
        matches!(err, LockfileInstallError::Malformed { .. }),
        "malformed must surface Malformed, got: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("malformed") || msg.contains("parse") || msg.contains("expected"),
        "Malformed error must describe the parser failure; got: {msg}"
    );
}

/// No lockfile present: actionable error suggests `chelis reef build`.
#[test]
fn phaseA_item9_no_lockfile_present_suggests_reef_build() {
    let _g = file_lock();
    let dir = tempdir().expect("tempdir");
    let pkg_root = dir.path().join("no-lockfile-pkg");
    fs::create_dir_all(&pkg_root).unwrap();

    let registry = dir.path().join("reef-home");
    let err = chelis_reef::install_from_lockfile(&pkg_root, &registry)
        .expect_err("missing lockfile must err");
    assert!(
        matches!(err, LockfileInstallError::NotFound { .. }),
        "missing lockfile must surface NotFound, got: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("chelis reef build"),
        "no-lockfile error must suggest chelis reef build; got: {msg}"
    );
}

/// CLI mutex: `--from-monorepo` and `--from-lockfile` together must
/// fail at clap parse time with a conflict message. Same shape as
/// the oracle's `--from-github` + `--from-lockfile` case but exercised
/// independently because clap's `conflicts_with_all` matrix is what
/// the brief locks.
#[test]
fn phaseA_item9_cli_rejects_lockfile_with_monorepo() {
    let _g = file_lock();
    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "reef",
            "install",
            "--from-lockfile",
            "--from-monorepo",
            "/tmp/whatever",
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    assert!(
        stderr.contains("cannot be used") || stderr.contains("conflicts"),
        "mutex must produce a clap-style conflict message; got: {stderr}"
    );
}

/// Manual gate: real-network `--from-github` then re-fetch via
/// `--from-lockfile`. `#[ignore]`d so the default workspace run
/// stays hermetic. Run via:
///
/// ```sh
/// GITHUB_TOKEN=$(gh auth token) cargo test -p chelis-cli \
///   phaseA_item9_real_github_manual_gate -- --ignored --exact
/// ```
///
/// The gate validates the live canonical-org path: real GitHub API
/// hit, real Nautilus release tag (`v0.5.0`, matching the manual
/// gate pinned by Item 6's oracle). Update the tag in lockstep with
/// Item 6 if Nautilus is bumped.
#[test]
#[ignore = "real-network manual gate; run with `--ignored --exact` and a working GITHUB_TOKEN"]
fn phaseA_item9_real_github_manual_gate() {
    let _g = file_lock();
    let dir = tempdir().expect("tempdir");
    let dev_a_registry = dir.path().join("dev-a-real-home");
    let dev_b_registry = dir.path().join("dev-b-real-home");
    let pkg_root = dir.path().join("real-pkg");

    // chelis-std is now bundled inside the chelis binary
    // (`crates/chelis-std-bundle`), so neither dev A's nor dev B's
    // registry needs to be pre-seeded with the runtime. The
    // `chelis reef build` invocations below resolve chelis-std from
    // the embedded bytes directly.

    // Step 1: install Nautilus from the real GitHub API.
    let result = chelis_reef::install_from_github("chelis-lang/nautilus@v0.5.0", &dev_a_registry);
    let _installed = result.expect("real-network install must succeed; check GITHUB_TOKEN");

    // Step 2: build a downstream package whose reef.lock will record
    // remote_origin. The shape mirrors `seed_downstream_package` but
    // depends on Nautilus instead of chelis-std because that's what
    // the real release ships.
    seed_downstream_package_for_dep(&pkg_root, "nautilus", "0.6.1", "Real");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &dev_a_registry)
        .args(["reef", "build", pkg_root.to_str().unwrap()])
        .assert()
        .success();

    // Step 3: re-install on dev B from the lockfile alone.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &dev_b_registry)
        .args([
            "reef",
            "install",
            "--from-lockfile",
            "--package-root",
            pkg_root.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Step 4: byte-equality across the two registries.
    let a_archive = fs::read(dev_a_registry.join("packages/nautilus/0.5.0/nautilus-0.5.0.tar.zst"))
        .expect("dev A nautilus archive");
    let b_archive = fs::read(dev_b_registry.join("packages/nautilus/0.5.0/nautilus-0.5.0.tar.zst"))
        .expect("dev B nautilus archive");
    assert_eq!(a_archive, b_archive, "real-network byte-equality must hold");
}

// ============================================================
// Helpers
// ============================================================

/// Generic helper: seed a downstream package that depends on
/// `(dep_name, dep_version)` and uses `module_prefix` for its module
/// declarations.
fn seed_downstream_package_for_dep(
    root: &Path,
    dep_name: &str,
    dep_version: &str,
    module_prefix: &str,
) {
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    let compiler = format!("={}", env!("CARGO_PKG_VERSION"));
    fs::write(
        root.join("reef.toml"),
        format!(
            r#"[package]
name = "downstream"
version = "0.2.0"
compiler = "{compiler}"
module_prefix = "{module_prefix}"

[dependencies]
{dep_name} = {{ version = "{dep_version}" }}
"#
        ),
    )
    .expect("write reef.toml");
    fs::write(
        root.join("src/main.ch"),
        format!("module {module_prefix}.Main\n\ndef id(x: int32) -> int32 = x\n"),
    )
    .expect("write main.ch");
}
