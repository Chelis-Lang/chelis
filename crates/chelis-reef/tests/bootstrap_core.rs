//! Host bootstrap fixtures use local HTTP services and real archive installation.

use chelis_reef::{BootstrapError, GitHubReleaseSpec};

fn install_bootstrap(
    specs: &[GitHubReleaseSpec],
    root: &Path,
) -> Result<Vec<chelis_reef::InstalledArtifact>, BootstrapError> {
    chelis_reef::install_bootstrap(specs, root, &chelis_std_bundle::EMBEDDED_RUNTIME)
}
use sha2::{Digest, Sha256};
use std::io::Cursor;
use std::path::Path;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

static ENV: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn manifest(name: &str, version: &str, deps: &str) -> String {
    format!(
        "[package]\nname = \"{name}\"\nversion = \"{version}\"\ncompiler = \"={}\"\nmodule_prefix = \"Test\"\n{deps}",
        env!("CARGO_PKG_VERSION")
    )
}

fn archive(text: &str) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for (name, bytes) in [
        ("reef.toml", text.as_bytes()),
        ("src/main.ch", b"module Test.Main\n".as_slice()),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, name, bytes).unwrap();
    }
    zstd::stream::encode_all(Cursor::new(builder.into_inner().unwrap()), 1).unwrap()
}

struct Host {
    runtime: tokio::runtime::Runtime,
    server: MockServer,
    prior_api: Option<std::ffi::OsString>,
    prior_token: Option<std::ffi::OsString>,
}

impl Host {
    fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(MockServer::start());
        let prior_api = std::env::var_os("CHELIS_REEF_GITHUB_BASE_API");
        let prior_token = std::env::var_os("GITHUB_TOKEN");
        unsafe {
            std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", server.uri());
            std::env::set_var("GITHUB_TOKEN", "local-fixture");
        }
        Self {
            runtime,
            server,
            prior_api,
            prior_token,
        }
    }

    fn release(
        &self,
        org: &str,
        name: &str,
        version: &str,
        text: &str,
        id: u64,
        corrupt: bool,
    ) -> GitHubReleaseSpec {
        let bytes = archive(text);
        let hash = if corrupt {
            "0".repeat(64)
        } else {
            format!("{:x}", Sha256::digest(&bytes))
        };
        let shell = chelis_shell::ShellPackage {
            format_version: chelis_shell::SHELL_FORMAT_VERSION,
            package: chelis_shell::PackageId {
                name: name.into(),
                version: version.into(),
            },
            compiler: format!("={}", env!("CARGO_PKG_VERSION")),
            modules: vec![],
            dependencies: vec![],
            archive_sha256: hash,
        };
        let shell_bytes = chelis_shell::encode_shell(&shell).unwrap();
        let metadata = serde_json::json!({"id": id, "tag_name": format!("v{version}"), "assets": [
            {"id": id, "name": format!("{name}-{version}.tar.zst"), "size": bytes.len()},
            {"id": id + 1, "name": format!("{name}-{version}.chb"), "size": shell_bytes.len()}
        ]});
        let routes = [
            (
                format!("/repos/{org}/{name}/releases/tags/v{version}"),
                ResponseTemplate::new(200).set_body_json(metadata),
            ),
            (
                format!("/repos/{org}/{name}/releases/assets/{id}"),
                ResponseTemplate::new(200).set_body_bytes(bytes),
            ),
            (
                format!("/repos/{org}/{name}/releases/assets/{}", id + 1),
                ResponseTemplate::new(200).set_body_bytes(shell_bytes),
            ),
        ];
        self.runtime.block_on(async {
            for (route, response) in routes {
                Mock::given(method("GET"))
                    .and(path(route))
                    .respond_with(response)
                    .mount(&self.server)
                    .await;
            }
        });
        GitHubReleaseSpec::parse(&format!("{org}/{name}@v{version}")).unwrap()
    }

    fn requests(&self, suffix: &str) -> usize {
        self.runtime
            .block_on(self.server.received_requests())
            .unwrap()
            .iter()
            .filter(|r| r.url.path().ends_with(suffix))
            .count()
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        unsafe {
            match &self.prior_api {
                Some(value) => std::env::set_var("CHELIS_REEF_GITHUB_BASE_API", value),
                None => std::env::remove_var("CHELIS_REEF_GITHUB_BASE_API"),
            }
            match &self.prior_token {
                Some(value) => std::env::set_var("GITHUB_TOKEN", value),
                None => std::env::remove_var("GITHUB_TOKEN"),
            }
        }
    }
}

fn no_install(root: &Path, host: &Host, ids: &[u64]) {
    assert!(!root.join("index.json").exists());
    for id in ids {
        assert_eq!(host.requests(&format!("/assets/{}", id + 1)), 0);
    }
}

#[test]
fn exact_graph_diagnostics_and_no_install() {
    let _guard = ENV.lock();
    let host = Host::new();
    let a = host.release(
        "test",
        "a",
        "1.0.0",
        &manifest("a", "1.0.0", "[dependencies.b]\nversion = \"2.0.0\"\n"),
        10,
        false,
    );
    let dir = tempfile::tempdir().unwrap();
    let err = install_bootstrap(&[a], dir.path()).unwrap_err();
    assert!(matches!(err, BootstrapError::MissingDependency { .. }));
    assert_eq!(
        err.to_string(),
        "shell `a` depends on `b` but `b` is not in the bootstrap input list: add an explicit `<org>/b@<tag>` entry to the bootstrap arguments"
    );
    no_install(dir.path(), &host, &[10]);

    let c = host.release(
        "test",
        "c",
        "1.0.0",
        &manifest("c", "1.0.0", "[dependencies.c]\nversion = \"1.0.0\"\n"),
        20,
        false,
    );
    let err = install_bootstrap(&[c.clone(), c], dir.path()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "dependency cycle detected among bootstrap shells: c@1.0.0 -> c@1.0.0 -> c@1.0.0"
    );
    no_install(dir.path(), &host, &[20]);

    let d2 = host.release("test", "d", "2.0.0", &manifest("d", "2.0.0", ""), 30, false);
    let d1 = host.release("test", "d", "1.0.0", &manifest("d", "1.0.0", ""), 40, false);
    let err = install_bootstrap(&[d2.clone(), d1, d2], dir.path()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "package `d` appears at multiple versions in the bootstrap input list: [1.0.0, 2.0.0]. Pick one"
    );
    no_install(dir.path(), &host, &[30, 40]);
}

#[test]
fn repeated_installs_select_original_releases_and_preserve_fetch_counts() {
    let _guard = ENV.lock();
    let host = Host::new();
    let first = host.release(
        "first",
        "a",
        "1.0.0",
        &manifest("a", "1.0.0", ""),
        10,
        false,
    );
    let second = host.release(
        "second",
        "a",
        "1.0.0",
        &manifest("a", "1.0.0", ""),
        20,
        false,
    );
    let dir = tempfile::tempdir().unwrap();
    let specs = [first, second];
    assert_eq!(install_bootstrap(&specs, dir.path()).unwrap().len(), 2);
    assert_eq!(host.requests("/assets/10"), 2);
    assert_eq!(host.requests("/assets/11"), 1);
    assert_eq!(host.requests("/assets/20"), 2);
    assert_eq!(host.requests("/assets/21"), 1);
    let before = std::fs::read(dir.path().join("index.json")).unwrap();
    let shell_path = dir.path().join("packages/a/1.0.0/a-1.0.0.chb");
    let shell_before = std::fs::read(&shell_path).unwrap();
    assert_eq!(install_bootstrap(&specs, dir.path()).unwrap().len(), 2);
    assert_eq!(
        std::fs::read(dir.path().join("index.json")).unwrap(),
        before
    );
    assert_eq!(std::fs::read(shell_path).unwrap(), shell_before);
    assert_eq!(host.requests("/assets/10"), 4);
    assert_eq!(host.requests("/assets/11"), 2);
    assert_eq!(host.requests("/assets/20"), 4);
    assert_eq!(host.requests("/assets/21"), 2);
}

#[test]
fn host_projection_keeps_runtime_path_and_manifest_rules() {
    let _guard = ENV.lock();
    let host = Host::new();
    let dir = tempfile::tempdir().unwrap();
    let runtime = GitHubReleaseSpec::parse("test/chelis-std@v1.0.0").unwrap();
    assert!(matches!(
        install_bootstrap(&[runtime], dir.path()),
        Err(BootstrapError::RuntimeNotABootstrapTarget { .. })
    ));
    assert_eq!(host.requests("v1.0.0"), 0);
    let deps = format!(
        "[dependencies.local]\npath = \"../local\"\n[dependencies.chelis-std]\nversion = \"{}\"\n",
        chelis_std_bundle::EMBEDDED_RUNTIME.version()
    );
    let valid = host.release(
        "test",
        "valid",
        "1.0.0",
        &manifest("valid", "1.0.0", &deps),
        10,
        false,
    );
    assert_eq!(install_bootstrap(&[valid], dir.path()).unwrap().len(), 1);
    for (name, text, manifest_error) in [
        (
            "mismatch",
            manifest(
                "mismatch",
                "1.0.0",
                "[dependencies.chelis-std]\nversion = \"99.0.0\"\n",
            ),
            false,
        ),
        (
            "versionless",
            manifest("versionless", "1.0.0", "[dependencies.other]\n"),
            true,
        ),
        ("wrongname", manifest("different", "1.0.0", ""), false),
        ("wrongversion", manifest("wrongversion", "2.0.0", ""), false),
        ("unreadable", "not TOML [".into(), true),
    ] {
        let spec = host.release("test", name, "1.0.0", &text, 100, false);
        let empty = tempfile::tempdir().unwrap();
        let err = install_bootstrap(&[spec], empty.path()).unwrap_err();
        if manifest_error {
            assert!(matches!(err, BootstrapError::ManifestRead { .. }));
        } else {
            assert!(matches!(err, BootstrapError::Validation { .. }));
        }
        no_install(empty.path(), &host, &[100]);
    }
}

#[test]
fn artifact_failure_preserves_prior_install_and_stops_later_installs() {
    let _guard = ENV.lock();
    let host = Host::new();
    let first = host.release(
        "test",
        "first",
        "1.0.0",
        &manifest("first", "1.0.0", ""),
        10,
        false,
    );
    let broken = host.release(
        "test",
        "broken",
        "1.0.0",
        &manifest("broken", "1.0.0", ""),
        20,
        true,
    );
    let last = host.release(
        "test",
        "last",
        "1.0.0",
        &manifest("last", "1.0.0", ""),
        30,
        false,
    );
    let dir = tempfile::tempdir().unwrap();
    let err = install_bootstrap(&[first, broken, last], dir.path()).unwrap_err();
    assert!(matches!(
        err,
        BootstrapError::Fetch(chelis_reef::GitHubFetchError::Validation { .. })
    ));
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("index.json")).unwrap()).unwrap();
    assert!(index["packages"]["first"].is_array());
    assert!(index["packages"]["broken"].is_null());
    assert!(index["packages"]["last"].is_null());
    assert!(
        dir.path()
            .join("packages/first/1.0.0/first-1.0.0.chb")
            .exists()
    );
    assert_eq!(host.requests("/assets/11"), 1);
    assert_eq!(host.requests("/assets/21"), 1);
    assert_eq!(host.requests("/assets/31"), 0);
}
