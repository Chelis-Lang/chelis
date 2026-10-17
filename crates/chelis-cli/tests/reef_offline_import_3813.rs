//! chelis#3813: `chelis reef fetch-plan` prints the download URL and the
//! pinned SHA-256 of each locked artifact. `chelis reef install
//! --from-lockfile --artifact-dir <DIR>` imports those files without
//! network access. It checks each file against `reef.lock` and keeps the
//! locked origin in `index.json`.
//!
//! Every command here runs with a network trap. The GitHub API base points
//! at a local mock server that must receive no request, and every proxy
//! variable points at a closed port. `PATH` is empty, so no `gh` helper can
//! run. A command that tries the network fails, or the mock server records
//! the request and the test fails.

use assert_cmd::Command;
use chelis_reef::{LocalRegistryIndex, host_platform_slug};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use tar::Builder;
use tempfile::{TempDir, tempdir};
use wiremock::MockServer;

const CURRENT_COMPILER_PIN: &str = concat!("=", env!("CARGO_PKG_VERSION"));
const ORG: &str = "test-org";
const TAG: &str = "v0.2.0";
const ORIGIN: &str = "github://test-org/nautilus@v0.2.0";
const BINARY_NAME: &str = "octant-translator";
const BINARY_ORIGIN: &str = "github://test-org/octant@v1.0.0";
const BINARY_PAYLOAD: &[u8] = b"#!/bin/sh\necho octant-translator-marker\n";
const CLOSED_PROXY: &str = "http://127.0.0.1:9";

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// A `nautilus` 0.2.0 source archive with a manifest and one module.
fn nautilus_archive() -> Vec<u8> {
    let manifest = format!(
        "[package]\nname = \"nautilus\"\nversion = \"0.2.0\"\ncompiler = \"{CURRENT_COMPILER_PIN}\"\nmodule_prefix = \"Test\"\n"
    );
    let main = "module Test.Main\n\nexport (placeholder)\ndef placeholder() -> i32 = 0\n";
    let mut tar_bytes = Vec::new();
    {
        let mut builder = Builder::new(&mut tar_bytes);
        for (path, bytes) in [
            ("reef.toml", manifest.as_bytes()),
            ("src/main.ch", main.as_bytes()),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_path(path).expect("set path");
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, bytes).expect("append member");
        }
        builder.finish().expect("finish tar");
    }
    zstd::stream::encode_all(Cursor::new(tar_bytes), 19).expect("zstd encode")
}

/// The `nautilus` 0.2.0 shell that pairs with `archive`.
fn nautilus_shell(archive: &[u8]) -> Vec<u8> {
    chelis_shell::encode_shell(&chelis_shell::ShellPackage {
        format_version: chelis_shell::SHELL_FORMAT_VERSION,
        package: chelis_shell::PackageId {
            name: "nautilus".to_string(),
            version: "0.2.0".to_string(),
        },
        compiler: CURRENT_COMPILER_PIN.to_string(),
        modules: Vec::new(),
        dependencies: Vec::new(),
        archive_sha256: sha256_bytes(archive),
    })
    .expect("encode shell")
}

/// A `.tar.zst` binary asset that carries one file, `BINARY_NAME`.
fn binary_asset() -> Vec<u8> {
    let mut tar_bytes = Vec::new();
    {
        let mut builder = Builder::new(&mut tar_bytes);
        let mut header = tar::Header::new_gnu();
        header.set_size(BINARY_PAYLOAD.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        builder
            .append_data(&mut header, BINARY_NAME, BINARY_PAYLOAD)
            .expect("append binary");
        builder.finish().expect("finish tar");
    }
    zstd::stream::encode_all(Cursor::new(tar_bytes), 0).expect("zstd encode")
}

fn host() -> &'static str {
    host_platform_slug().expect("the test host has a platform slug")
}

fn binary_asset_name() -> String {
    format!("{BINARY_NAME}-{}.tar.zst", host())
}

/// The files of one locked release, as a fetcher outside Reef gets them.
struct Release {
    archive: Vec<u8>,
    shell: Vec<u8>,
    binary: Vec<u8>,
}

impl Release {
    fn new() -> Self {
        let archive = nautilus_archive();
        let shell = nautilus_shell(&archive);
        Self {
            archive,
            shell,
            binary: binary_asset(),
        }
    }
}

/// A package whose `reef.lock` locks `nautilus` from a GitHub release, a
/// binary for this host, a path dependency, and the bundled runtime.
/// `nautilus_origin` is the `remote_origin` line of the `nautilus` entry,
/// or nothing.
fn locked_package(dir: &Path, release: &Release, nautilus_origin: &str) -> PathBuf {
    let root = dir.join("consumer");
    fs::create_dir_all(&root).expect("create the package root");
    let lock = format!(
        r#"[package]
name = "consumer"
version = "0.1.0"

[[dependencies]]
name = "nautilus"
version = "0.2.0"
compiler = "{CURRENT_COMPILER_PIN}"
archive_sha256 = "{archive_sha}"
shell_sha256 = "{shell_sha}"

[dependencies.source]
kind = "local_registry"
{nautilus_origin}

[[dependencies]]
name = "{BINARY_NAME}"
version = "1.0.0"
compiler = "{CURRENT_COMPILER_PIN}"
archive_sha256 = "{binary_sha}"
shell_sha256 = ""

[dependencies.source]
kind = "binary"
remote_origin = "{BINARY_ORIGIN}"
platform = "{platform}"
asset = "{asset}"
sha256 = "{binary_sha}"

[[dependencies]]
name = "chelis-std"
version = "{std_version}"
compiler = "{CURRENT_COMPILER_PIN}"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "bundled"
compiler_version = "{compiler}"
"#,
        archive_sha = sha256_bytes(&release.archive),
        shell_sha = sha256_bytes(&release.shell),
        binary_sha = sha256_bytes(&release.binary),
        platform = host(),
        asset = binary_asset_name(),
        std_version = chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION,
        compiler = env!("CARGO_PKG_VERSION"),
    );
    fs::write(root.join("reef.lock"), lock).expect("write reef.lock");
    root
}

/// A mock GitHub API that must see no request, and the homes that the
/// commands write.
struct Sandbox {
    _dir: TempDir,
    dir: PathBuf,
    runtime: tokio::runtime::Runtime,
    server: MockServer,
}

impl Sandbox {
    fn new() -> Self {
        let dir = tempdir().expect("tempdir");
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("tokio runtime");
        let server = runtime.block_on(MockServer::start());
        Self {
            dir: dir.path().to_path_buf(),
            _dir: dir,
            runtime,
            server,
        }
    }

    fn reef_home(&self) -> PathBuf {
        self.dir.join("reef-home")
    }

    fn chelis_home(&self) -> PathBuf {
        self.dir.join("chelis-home")
    }

    fn chelis(&self) -> Command {
        let mut command = Command::cargo_bin("chelis").expect("binary");
        command
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", self.reef_home())
            .env("CHELIS_HOME", self.chelis_home())
            .env("CHELIS_REEF_GITHUB_BASE_API", self.server.uri())
            .env("GITHUB_TOKEN", "unit-test-token")
            .env("PATH", "");
        for proxy in [
            "HTTPS_PROXY",
            "https_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "ALL_PROXY",
            "all_proxy",
        ] {
            command.env(proxy, CLOSED_PROXY);
        }
        // The mock API on the loopback host stays reachable, so it can
        // record any request. Every other host goes to the closed proxy.
        command
            .env("NO_PROXY", "127.0.0.1,localhost")
            .env("no_proxy", "127.0.0.1,localhost");
        command
    }

    fn assert_no_network(&self) {
        let requests = self
            .runtime
            .block_on(self.server.received_requests())
            .expect("the mock server records requests");
        assert!(
            requests.is_empty(),
            "the command tried the network: {requests:?}"
        );
    }

    fn fetch_plan(&self, root: &Path) -> serde_json::Value {
        let output = self
            .chelis()
            .args(["reef", "fetch-plan", "--package-root"])
            .arg(root)
            .output()
            .expect("run fetch-plan");
        assert!(
            output.status.success(),
            "fetch-plan failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).expect("fetch-plan prints JSON")
    }

    fn import(&self, root: &Path, artifacts: &Path) -> std::process::Output {
        self.chelis()
            .args(["reef", "install", "--from-lockfile", "--package-root"])
            .arg(root)
            .arg("--artifact-dir")
            .arg(artifacts)
            .output()
            .expect("run the import")
    }
}

/// Write each file that `plan` names into `artifacts`, with the bytes of
/// `release`. `skip` names a file to leave out.
fn fetch_into(plan: &serde_json::Value, release: &Release, artifacts: &Path, skip: Option<&str>) {
    fs::create_dir_all(artifacts).expect("create the artifact directory");
    for element in plan["artifacts"].as_array().expect("artifacts array") {
        let file = element["file"].as_str().expect("file name");
        if Some(file) == skip {
            continue;
        }
        let bytes = match element["kind"].as_str().expect("kind") {
            "archive" => &release.archive,
            "shell" => &release.shell,
            "binary" => &release.binary,
            other => panic!("unexpected kind {other}"),
        };
        assert_eq!(
            element["sha256"].as_str().expect("sha256"),
            sha256_bytes(bytes),
            "the plan pins the bytes of {file}"
        );
        fs::write(artifacts.join(file), bytes).expect("write an artifact file");
    }
}

fn read_index(reef_home: &Path) -> LocalRegistryIndex {
    serde_json::from_str(&fs::read_to_string(reef_home.join("index.json")).expect("read index"))
        .expect("parse index")
}

#[test]
fn fetch_plan_lists_each_locked_file_with_url_and_pin() {
    let sandbox = Sandbox::new();
    let release = Release::new();
    let root = locked_package(
        &sandbox.dir,
        &release,
        &format!("remote_origin = \"{ORIGIN}\""),
    );

    let plan = sandbox.fetch_plan(&root);
    sandbox.assert_no_network();

    let expected = serde_json::json!({
        "format_version": 1,
        "package": { "name": "consumer", "version": "0.1.0" },
        "artifacts": [
            {
                "name": "nautilus",
                "version": "0.2.0",
                "kind": "archive",
                "file": "nautilus-0.2.0.tar.zst",
                "url": format!("https://github.com/{ORG}/nautilus/releases/download/{TAG}/nautilus-0.2.0.tar.zst"),
                "sha256": sha256_bytes(&release.archive),
                "remote_origin": ORIGIN,
            },
            {
                "name": "nautilus",
                "version": "0.2.0",
                "kind": "shell",
                "file": "nautilus-0.2.0.chb",
                "url": format!("https://github.com/{ORG}/nautilus/releases/download/{TAG}/nautilus-0.2.0.chb"),
                "sha256": sha256_bytes(&release.shell),
                "remote_origin": ORIGIN,
            },
            {
                "name": BINARY_NAME,
                "version": "1.0.0",
                "kind": "binary",
                "file": binary_asset_name(),
                "url": format!("https://github.com/{ORG}/octant/releases/download/v1.0.0/{}", binary_asset_name()),
                "sha256": sha256_bytes(&release.binary),
                "remote_origin": BINARY_ORIGIN,
                "platform": host(),
            },
        ],
    });
    assert_eq!(plan, expected);
    assert!(
        !sandbox.reef_home().exists(),
        "fetch-plan writes no registry state"
    );
}

/// Negative parity: an entry with no origin has no download URL, so the
/// plan fails and names the entry.
#[test]
fn fetch_plan_rejects_an_entry_without_origin() {
    let sandbox = Sandbox::new();
    let root = locked_package(&sandbox.dir, &Release::new(), "");

    let output = sandbox
        .chelis()
        .args(["reef", "fetch-plan", "--package-root"])
        .arg(&root)
        .output()
        .expect("run fetch-plan");
    sandbox.assert_no_network();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "fetch-plan must fail: {stderr}");
    assert!(
        stderr.contains("`nautilus` v0.2.0 has no `remote_origin`"),
        "{stderr}"
    );
    assert!(output.stdout.is_empty(), "no partial plan on stdout");
}

/// The import places the fetched pair in the registry with the locked
/// origin, and the binary under `$CHELIS_HOME/bin`. The mock server sees
/// no request.
#[test]
fn import_installs_fetched_files_with_their_origins_without_network() {
    let sandbox = Sandbox::new();
    let release = Release::new();
    let root = locked_package(
        &sandbox.dir,
        &release,
        &format!("remote_origin = \"{ORIGIN}\""),
    );
    let artifacts = sandbox.dir.join("fetched");
    fetch_into(&sandbox.fetch_plan(&root), &release, &artifacts, None);

    let output = sandbox.import(&root, &artifacts);
    sandbox.assert_no_network();
    assert!(
        output.status.success(),
        "import failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let package = sandbox.reef_home().join("packages/nautilus/0.2.0");
    assert_eq!(
        fs::read(package.join("nautilus-0.2.0.tar.zst")).expect("read the archive"),
        release.archive
    );
    assert_eq!(
        fs::read(package.join("nautilus-0.2.0.chb")).expect("read the shell"),
        release.shell
    );
    let index = read_index(&sandbox.reef_home());
    let versions = &index.packages["nautilus"];
    assert_eq!(versions.len(), 1);
    assert_eq!(versions[0].version, "0.2.0");
    assert_eq!(versions[0].archive_sha256, sha256_bytes(&release.archive));
    assert_eq!(versions[0].shell_sha256, sha256_bytes(&release.shell));
    assert_eq!(
        versions[0].remote_origin.as_deref(),
        Some(ORIGIN),
        "the registry keeps the locked origin"
    );
    assert_eq!(
        fs::read(sandbox.chelis_home().join("bin").join(BINARY_NAME)).expect("read the binary"),
        BINARY_PAYLOAD
    );
}

/// A file whose bytes differ from the lock pin fails the import before any
/// registry write for that entry.
#[test]
fn import_rejects_a_checksum_mismatch() {
    let sandbox = Sandbox::new();
    let release = Release::new();
    let root = locked_package(
        &sandbox.dir,
        &release,
        &format!("remote_origin = \"{ORIGIN}\""),
    );
    let artifacts = sandbox.dir.join("fetched");
    fetch_into(&sandbox.fetch_plan(&root), &release, &artifacts, None);
    let tampered = Release {
        archive: nautilus_archive().into_iter().chain(*b"tampered").collect(),
        shell: release.shell.clone(),
        binary: release.binary.clone(),
    };
    fs::write(artifacts.join("nautilus-0.2.0.tar.zst"), &tampered.archive)
        .expect("tamper with the archive");

    let output = sandbox.import(&root, &artifacts);
    sandbox.assert_no_network();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "the import must fail: {stderr}");
    assert!(
        stderr.contains(&format!(
            "SHA-256 mismatch for {}: reef.lock pins {}, but the file hashes to {}",
            artifacts.join("nautilus-0.2.0.tar.zst").display(),
            sha256_bytes(&release.archive),
            sha256_bytes(&tampered.archive)
        )),
        "{stderr}"
    );
    assert!(
        !sandbox.reef_home().join("packages/nautilus").exists(),
        "a mismatched file never enters the registry"
    );
    assert!(!sandbox.reef_home().join("index.json").exists());
}

/// A missing file fails the import and names the file.
#[test]
fn import_rejects_a_missing_file() {
    let sandbox = Sandbox::new();
    let release = Release::new();
    let root = locked_package(
        &sandbox.dir,
        &release,
        &format!("remote_origin = \"{ORIGIN}\""),
    );
    let artifacts = sandbox.dir.join("fetched");
    fetch_into(
        &sandbox.fetch_plan(&root),
        &release,
        &artifacts,
        Some("nautilus-0.2.0.chb"),
    );

    let output = sandbox.import(&root, &artifacts);
    sandbox.assert_no_network();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "the import must fail: {stderr}");
    assert!(
        stderr.contains(&format!(
            "missing artifact file {}",
            artifacts.join("nautilus-0.2.0.chb").display()
        )),
        "{stderr}"
    );
    assert!(!sandbox.reef_home().join("packages/nautilus").exists());
    assert!(!sandbox.reef_home().join("index.json").exists());
}

/// Negative parity for the network trap: the same lock without
/// `--artifact-dir` tries the network, and the trap catches it.
#[test]
fn network_install_trips_the_trap() {
    let sandbox = Sandbox::new();
    let root = locked_package(
        &sandbox.dir,
        &Release::new(),
        &format!("remote_origin = \"{ORIGIN}\""),
    );

    let output = sandbox
        .chelis()
        .args(["reef", "install", "--from-lockfile", "--package-root"])
        .arg(&root)
        .output()
        .expect("run the network install");

    assert!(!output.status.success());
    let requests = sandbox
        .runtime
        .block_on(sandbox.server.received_requests())
        .expect("the mock server records requests");
    assert!(
        !requests.is_empty(),
        "the network install must reach the mock API"
    );
}
