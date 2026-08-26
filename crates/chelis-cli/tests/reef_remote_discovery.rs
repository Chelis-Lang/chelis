//! Authoritative acceptance oracle for bounded Reef remote discovery.

use assert_cmd::Command;
use chelis_reef::{
    BudgetDimension, DiscoveryMode, ResolutionBudget, SourceLocator,
    inspect_candidate_manifest_archive,
};
use predicates::prelude::*;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;
use tar::{Builder, EntryType, Header};
use tempfile::tempdir;
use wiremock::matchers::{method, path as wire_path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

const COMPILER_PIN: &str = concat!("=", env!("CARGO_PKG_VERSION"));

fn chelis(root: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.current_dir(root);
    command
}

fn schema_one_manifest(name: &str, dependencies: &str) -> String {
    format!(
        "schema = \"1\"\n\n[package]\nname = \"{name}\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\n\n[dependencies]\n{dependencies}"
    )
}

fn build_release_with_dependencies(
    parent: &Path,
    name: &str,
    version: &str,
    dependencies: &str,
    registry: &Path,
) -> (PathBuf, PathBuf) {
    let root = parent.join(format!("{name}-{version}"));
    fs::create_dir_all(root.join("src")).unwrap();
    let manifest = if dependencies.is_empty() {
        format!(
            "schema = \"1\"\n\n[package]\nname = \"{name}\"\nversion = \"{version}\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\n"
        )
    } else {
        format!(
            "schema = \"2\"\n\n[package]\nname = \"{name}\"\nversion = \"{version}\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\n{dependencies}"
        )
    };
    fs::write(root.join("reef.toml"), manifest).unwrap();
    fs::write(
        root.join("src/main.ch"),
        "module Remote.Main\n\ndef value() -> int32 = 1\n",
    )
    .unwrap();
    chelis(&root)
        .env("CHELIS_REEF_HOME", registry)
        .args(["reef", "build", "--no-auto-fetch"])
        .assert()
        .success();
    (
        root.join("dist").join(format!("{name}-{version}.tar.zst")),
        root.join("dist").join(format!("{name}-{version}.chb")),
    )
}

fn build_release_artifacts(parent: &Path, name: &str, version: &str) -> (PathBuf, PathBuf) {
    build_release_with_dependencies(parent, name, version, "", &parent.join("build-registry"))
}

fn build_release_with_body(
    parent: &Path,
    name: &str,
    version: &str,
    body_value: i32,
) -> (PathBuf, PathBuf) {
    let root = parent.join(format!("{name}-{version}-body-{body_value}"));
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"1\"\n\n[package]\nname = \"{name}\"\nversion = \"{version}\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\n"
        ),
    )
    .unwrap();
    fs::write(
        root.join("src/main.ch"),
        format!("module Remote.Main\n\ndef value() -> int32 = {body_value}\n"),
    )
    .unwrap();
    chelis(&root)
        .env("CHELIS_REEF_HOME", parent.join("body-build-registry"))
        .args(["reef", "build", "--no-auto-fetch"])
        .assert()
        .success();
    (
        root.join("dist").join(format!("{name}-{version}.tar.zst")),
        root.join("dist").join(format!("{name}-{version}.chb")),
    )
}

fn build_release_with_compiler(
    parent: &Path,
    name: &str,
    version: &str,
    compiler: &str,
) -> (PathBuf, PathBuf) {
    let root = parent.join(format!("{name}-{version}-{compiler}"));
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"1\"\n\n[package]\nname = \"{name}\"\nversion = \"{version}\"\ncompiler = \"{compiler}\"\nmodule_prefix = \"Remote\"\n"
        ),
    )
    .unwrap();
    fs::write(
        root.join("src/main.ch"),
        "module Remote.Main\n\ndef value() -> int32 = 1\n",
    )
    .unwrap();
    chelis(&root)
        .env("CHELIS_REEF_HOME", parent.join("compiler-build-registry"))
        .env("CHELIS_REEF_ALLOW_DEP_COMPILER_DRIFT", "1")
        .args(["reef", "build", "--no-auto-fetch"])
        .assert()
        .success();
    (
        root.join("dist").join(format!("{name}-{version}.tar.zst")),
        root.join("dist").join(format!("{name}-{version}.chb")),
    )
}

fn release_json(name: &str, versions: &[(&str, u64, u64)]) -> serde_json::Value {
    serde_json::Value::Array(
        versions
            .iter()
            .map(|(version, archive_id, shell_id)| {
                serde_json::json!({
                    "tag_name": format!("v{version}"),
                    "draft": false,
                    "prerelease": false,
                    "assets": [
                        {"id": archive_id, "name": format!("{name}-{version}.tar.zst")},
                        {"id": shell_id, "name": format!("{name}-{version}.chb")}
                    ]
                })
            })
            .collect(),
    )
}

async fn mount_single_release(
    server: &MockServer,
    name: &str,
    version: &str,
    archive_id: u64,
    shell_id: u64,
    archive: Vec<u8>,
    shell: Vec<u8>,
) {
    Mock::given(method("GET"))
        .and(wire_path(format!("/repos/chelis-lang/{name}/releases")))
        .and(query_param("per_page", "100"))
        .and(query_param("page", "1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(release_json(name, &[(version, archive_id, shell_id)])),
        )
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(wire_path(format!(
            "/repos/chelis-lang/{name}/releases/assets/{archive_id}"
        )))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(archive))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(wire_path(format!(
            "/repos/chelis-lang/{name}/releases/assets/{shell_id}"
        )))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(shell))
        .mount(server)
        .await;
}

fn stage_update_project(
    parent: &Path,
    case_name: &str,
    archive: &Path,
    shell: &Path,
) -> (PathBuf, PathBuf, Vec<u8>) {
    let registry = parent.join(format!("{case_name}-registry"));
    chelis_reef::install_validated_artifact_pair(
        archive,
        shell,
        "nautilus",
        "1.0.0",
        &registry,
        Some("github://chelis-lang/nautilus@v1.0.0"),
    )
    .unwrap();
    let root = parent.join(format!("{case_name}-app"));
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"{case_name}-app\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nnautilus = \"^1\"\n"
        ),
    )
    .unwrap();
    fs::write(
        root.join("src/main.ch"),
        "module Remote.Main\n\ndef main() -> int32 = 1\n",
    )
    .unwrap();
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .args(["reef", "update", "--offline"])
        .assert()
        .success();
    let lock = fs::read(root.join("reef.lock")).unwrap();
    (root, registry, lock)
}

fn append_tar_entry(
    builder: &mut Builder<Vec<u8>>,
    path: &str,
    entry_type: EntryType,
    body: &[u8],
) {
    let mut header = Header::new_gnu();
    header.set_entry_type(entry_type);
    header.set_mode(0o644);
    header.set_size(body.len() as u64);
    header.set_cksum();
    builder
        .append_data(&mut header, path, Cursor::new(body))
        .expect("append tar entry");
}

fn candidate_archive(entries: &[(&str, EntryType, &[u8])]) -> Vec<u8> {
    let mut builder = Builder::new(Vec::new());
    for (path, kind, body) in entries {
        append_tar_entry(&mut builder, path, *kind, body);
    }
    let tar = builder.into_inner().expect("finish tar");
    zstd::stream::encode_all(Cursor::new(tar), 3).expect("compress tar")
}

fn raw_path_candidate_archive(path: &[u8], entry_type: EntryType, body: &[u8]) -> Vec<u8> {
    assert!(path.len() <= 100);
    let mut builder = Builder::new(Vec::new());
    let mut header = Header::new_gnu();
    header.set_entry_type(entry_type);
    header.set_mode(0o644);
    header.set_size(body.len() as u64);
    header.as_mut_bytes()[..100].fill(0);
    header.as_mut_bytes()[..path.len()].copy_from_slice(path);
    header.set_cksum();
    builder.append(&header, Cursor::new(body)).unwrap();
    let tar = builder.into_inner().unwrap();
    zstd::stream::encode_all(Cursor::new(tar), 3).unwrap()
}

fn write_archive(root: &Path, bytes: &[u8]) -> PathBuf {
    let path = root.join("candidate.tar.zst");
    fs::write(&path, bytes).expect("write candidate archive");
    path
}

#[test]
fn discovery_modes_are_closed_and_parse_at_the_boundary() {
    assert_eq!(
        DiscoveryMode::from_str("locked").unwrap(),
        DiscoveryMode::Locked
    );
    assert_eq!(
        DiscoveryMode::from_str("resolve").unwrap(),
        DiscoveryMode::Resolve
    );
    assert_eq!(
        DiscoveryMode::from_str("refresh").unwrap(),
        DiscoveryMode::Refresh
    );
    assert_eq!(
        DiscoveryMode::from_str("inspect").unwrap(),
        DiscoveryMode::Inspect
    );
    assert!(DiscoveryMode::from_str("fallback").is_err());
}

#[test]
fn source_locators_are_typed_and_fail_closed() {
    assert_eq!(
        SourceLocator::from_str("github://chelis-lang/nautilus")
            .unwrap()
            .to_string(),
        "github://chelis-lang/nautilus"
    );
    assert!(SourceLocator::from_str("https://github.com/chelis-lang/nautilus").is_err());
    assert!(SourceLocator::from_str("mirror://chelis-lang/nautilus").is_err());
    assert!(SourceLocator::from_str("github://Chelis-Lang/nautilus").is_err());
}

#[test]
fn every_production_budget_dimension_accepts_its_limit_and_rejects_the_next_unit() {
    let cases = [
        (BudgetDimension::ReleasePages, 10),
        (BudgetDimension::HttpRequests, 2_048),
        (BudgetDimension::AcceptedTags, 256),
        (BudgetDimension::CandidateManifests, 64),
        (BudgetDimension::CompressedCandidateBytes, 64 * 1024 * 1024),
        (BudgetDimension::ScannedCandidateBytes, 256 * 1024 * 1024),
        (BudgetDimension::ManifestBytes, 1024 * 1024),
        (BudgetDimension::TotalDownloadBytes, 1024 * 1024 * 1024),
    ];
    for (dimension, limit) in cases {
        let mut budget = ResolutionBudget::production();
        budget
            .charge(dimension, limit, "nautilus", "control")
            .unwrap();
        let error = budget
            .charge(dimension, 1, "nautilus", "control")
            .unwrap_err();
        assert_eq!(error.dimension(), dimension);
        assert_eq!(error.limit(), limit);
        assert_eq!(error.observed(), limit + 1);
        assert_eq!(error.package(), "nautilus");
    }

    let budget = ResolutionBudget::production();
    budget
        .check_request_elapsed(Duration::from_secs(60), "nautilus", "release list")
        .unwrap();
    let error = budget
        .check_request_elapsed(Duration::from_secs(61), "nautilus", "release list")
        .unwrap_err();
    assert_eq!(error.dimension(), BudgetDimension::RequestElapsedTime);
}

#[test]
fn budget_counter_overflow_is_a_typed_failure() {
    let mut budget = ResolutionBudget::production();
    budget
        .charge(BudgetDimension::HttpRequests, 1, "nautilus", "seed")
        .unwrap();
    let error = budget
        .charge(
            BudgetDimension::HttpRequests,
            u64::MAX,
            "nautilus",
            "overflow probe",
        )
        .unwrap_err();
    assert_eq!(error.dimension(), BudgetDimension::HttpRequests);
    assert_eq!(error.observed(), u64::MAX);
}

#[test]
fn candidate_inspection_accepts_one_root_regular_manifest_without_extracting_files() {
    let directory = tempdir().unwrap();
    let manifest = schema_one_manifest("nautilus", "");
    let archive = candidate_archive(&[
        ("reef.toml", EntryType::Regular, manifest.as_bytes()),
        ("src/main.ch", EntryType::Regular, b"module Remote.Main\n"),
    ]);
    let path = write_archive(directory.path(), &archive);
    let mut budget = ResolutionBudget::production();

    let inspected = inspect_candidate_manifest_archive(&path, "nautilus", &mut budget).unwrap();
    assert_eq!(inspected.package.name, "nautilus");
    assert!(!directory.path().join("src").exists());
}

#[test]
fn candidate_inspection_rejects_duplicate_and_nonregular_manifests() {
    let directory = tempdir().unwrap();
    let manifest = schema_one_manifest("nautilus", "");
    let duplicate = candidate_archive(&[
        ("reef.toml", EntryType::Regular, manifest.as_bytes()),
        ("reef.toml", EntryType::Regular, manifest.as_bytes()),
    ]);
    let duplicate_path = write_archive(directory.path(), &duplicate);
    let mut budget = ResolutionBudget::production();
    assert!(
        inspect_candidate_manifest_archive(&duplicate_path, "nautilus", &mut budget)
            .unwrap_err()
            .to_string()
            .contains("exactly one")
    );

    let linked = candidate_archive(&[("reef.toml", EntryType::Symlink, b"")]);
    let linked_path = write_archive(directory.path(), &linked);
    let mut budget = ResolutionBudget::production();
    assert!(
        inspect_candidate_manifest_archive(&linked_path, "nautilus", &mut budget)
            .unwrap_err()
            .to_string()
            .contains("regular file")
    );
}

#[test]
fn candidate_inspection_rejects_malformed_compressed_tar_data() {
    let directory = tempdir().unwrap();
    let malformed = zstd::stream::encode_all(Cursor::new(b"not a tar archive"), 3).unwrap();
    let path = write_archive(directory.path(), &malformed);
    let mut budget = ResolutionBudget::production();
    assert!(inspect_candidate_manifest_archive(&path, "nautilus", &mut budget).is_err());
}

#[test]
fn candidate_inspection_rejects_unsafe_paths_devices_missing_manifests_and_future_schemas() {
    let directory = tempdir().unwrap();
    let manifest = schema_one_manifest("nautilus", "");
    for path in [
        b"../reef.toml".as_slice(),
        b"/reef.toml".as_slice(),
        b"dir\\reef.toml".as_slice(),
    ] {
        let archive = raw_path_candidate_archive(path, EntryType::Regular, manifest.as_bytes());
        let archive_path = write_archive(directory.path(), &archive);
        let mut budget = ResolutionBudget::production();
        let error = inspect_candidate_manifest_archive(&archive_path, "nautilus", &mut budget)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsafe tar path"), "{error}");
    }

    let device = raw_path_candidate_archive(b"device", EntryType::Block, b"");
    let device_path = write_archive(directory.path(), &device);
    let mut budget = ResolutionBudget::production();
    assert!(
        inspect_candidate_manifest_archive(&device_path, "nautilus", &mut budget)
            .unwrap_err()
            .to_string()
            .contains("regular file or directory")
    );

    let missing = candidate_archive(&[("src/main.ch", EntryType::Regular, b"module X\n")]);
    let missing_path = write_archive(directory.path(), &missing);
    let mut budget = ResolutionBudget::production();
    assert!(
        inspect_candidate_manifest_archive(&missing_path, "nautilus", &mut budget)
            .unwrap_err()
            .to_string()
            .contains("exactly one")
    );

    let future = manifest.replace("schema = \"1\"", "schema = \"999\"");
    let future_archive = candidate_archive(&[("reef.toml", EntryType::Regular, future.as_bytes())]);
    let future_path = write_archive(directory.path(), &future_archive);
    let mut budget = ResolutionBudget::production();
    assert!(
        inspect_candidate_manifest_archive(&future_path, "nautilus", &mut budget)
            .unwrap_err()
            .to_string()
            .contains("unsupported manifest schema 999")
    );
}

#[test]
fn candidate_inspection_enforces_the_manifest_byte_limit_during_io() {
    let directory = tempdir().unwrap();
    let oversized = vec![b'x'; 1024 * 1024 + 1];
    let archive = candidate_archive(&[("reef.toml", EntryType::Regular, &oversized)]);
    let path = write_archive(directory.path(), &archive);
    let mut budget = ResolutionBudget::production();
    let error = inspect_candidate_manifest_archive(&path, "nautilus", &mut budget).unwrap_err();
    assert!(error.to_string().contains("parsed reef.toml bytes"));
}

#[test]
fn schema_one_to_two_migration_preserves_text_and_exact_selection() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("schema-two");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        schema_one_manifest(
            "schema-two",
            "# dependency comment\nnautilus = { version = \"1.2.3\" }\nlocal = { path = \"./local\" }\nboth = { version = \"2.3.4\", path = \"./both\" }\n",
        ),
    )
    .unwrap();
    fs::write(
        root.join("src/main.ch"),
        "module Remote.Main\n\ndef main() -> int32 = 1\n",
    )
    .unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace", "--manifest-to", "2"])
        .assert()
        .success();

    let migrated = fs::read_to_string(root.join("reef.toml")).unwrap();
    assert!(migrated.starts_with("schema = \"2\"\n"), "{migrated}");
    assert!(migrated.contains("resolver = \"2\""), "{migrated}");
    assert!(migrated.contains("# dependency comment"), "{migrated}");
    assert!(migrated.contains("version = \"=1.2.3\""), "{migrated}");
    assert!(
        migrated.contains("local = { path = \"./local\" }"),
        "{migrated}"
    );
    assert!(migrated.contains("version = \"=2.3.4\""), "{migrated}");
}

#[test]
fn migration_supports_root_inline_tables_without_widening_exact_versions() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("inline-migration");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"1\"\npackage = {{ name = \"inline-migration\", version = \"0.1.0\", compiler = \"{COMPILER_PIN}\", module_prefix = \"Remote\" }}\ndependencies = {{ nautilus = {{ version = \"1.2.3\" }}, local = {{ path = \"./local\" }} }}\n"
        ),
    )
    .unwrap();
    chelis(&root)
        .args(["reef", "upgrade", "--inplace", "--manifest-to", "2"])
        .assert()
        .success();
    let migrated = fs::read_to_string(root.join("reef.toml")).unwrap();
    assert!(migrated.contains("resolver = \"2\""), "{migrated}");
    assert!(migrated.contains("version = \"=1.2.3\""), "{migrated}");
    assert!(migrated.contains("path = \"./local\""), "{migrated}");
}

#[test]
fn migration_rejects_fields_that_are_invalid_in_the_source_schema() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("invalid-source-schema");
    fs::create_dir_all(&root).unwrap();
    let manifest = format!(
        "schema = \"1\"\n[package]\nname = \"invalid-source-schema\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"1\"\n"
    );
    fs::write(root.join("reef.toml"), &manifest).unwrap();
    chelis(&root)
        .args(["reef", "upgrade", "--inplace", "--manifest-to", "2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("resolver"));
    assert_eq!(
        fs::read_to_string(root.join("reef.toml")).unwrap(),
        manifest
    );
}

#[test]
fn migration_preserves_trailing_schema_and_dependency_comments() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("comment-migration");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"1\" # schema policy\n\n[package]\nname = \"comment-migration\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\n\n[dependencies.nautilus]\nversion = \"1.2.3\" # security pin\n"
        ),
    )
    .unwrap();
    chelis(&root)
        .args(["reef", "upgrade", "--inplace", "--manifest-to", "2"])
        .assert()
        .success();
    let migrated = fs::read_to_string(root.join("reef.toml")).unwrap();
    assert!(
        migrated.contains("schema = \"2\" # schema policy"),
        "{migrated}"
    );
    assert!(
        migrated.contains("version = \"=1.2.3\" # security pin"),
        "{migrated}"
    );
}

#[test]
fn invalid_schema_one_exact_dependency_blocks_migration_without_writes() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("invalid-migration");
    fs::create_dir_all(&root).unwrap();
    let manifest =
        schema_one_manifest("invalid-migration", "nautilus = { version = \"^1.2.3\" }\n");
    fs::write(root.join("reef.toml"), &manifest).unwrap();

    chelis(&root)
        .args(["reef", "upgrade", "--inplace", "--manifest-to", "2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("nautilus"));
    assert_eq!(
        fs::read_to_string(root.join("reef.toml")).unwrap(),
        manifest
    );
}

#[test]
fn schema_two_requires_resolver_two_and_accepts_cargo_requirements() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("schema-two-parse");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/main.ch"),
        "module Remote.Main\n\ndef main() -> int32 = 1\n",
    )
    .unwrap();
    let base = format!(
        "schema = \"2\"\n\n[package]\nname = \"schema-two-parse\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nnautilus = \"^1.2\"\n"
    );
    fs::write(root.join("reef.toml"), &base).unwrap();
    let parsed = chelis_reef::read_manifest_for_src(&root).expect("schema 2 parses");
    assert_eq!(parsed.package.name, "schema-two-parse");

    fs::write(
        root.join("reef.toml"),
        base.replace("resolver = \"2\"\n", ""),
    )
    .unwrap();
    let error = chelis_reef::read_manifest_for_src(&root).unwrap_err();
    assert!(error.contains("resolver"), "{error}");

    for invalid in ["resolver = \"1\"", "resolver = \"3\"", "resolver = 2"] {
        fs::write(
            root.join("reef.toml"),
            base.replace("resolver = \"2\"", invalid),
        )
        .unwrap();
        let error = chelis_reef::read_manifest_for_src(&root).unwrap_err();
        assert!(error.contains("resolver"), "{invalid}: {error}");
    }
}

#[test]
fn explicit_bundled_runtime_resolves_without_registry_or_network_access() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("explicit-bundled-runtime");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"explicit-bundled-runtime\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nchelis-std = \"={}\"\n",
            chelis_reef::compiler_bundled_chelis_std_version()
        ),
    )
    .unwrap();
    fs::write(
        root.join("src/main.ch"),
        "module Remote.Main\n\ndef main() -> int32 = 1\n",
    )
    .unwrap();

    let registry = directory.path().join("empty-registry");
    let output = directory.path().join("source-build");
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", "http://127.0.0.1:9")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "src/main.ch", "-o", output.to_str().unwrap()])
        .assert()
        .success();
    assert!(output.join("main.c").exists());
    let source_lock = fs::read_to_string(root.join("reef.lock")).unwrap();
    assert!(source_lock.contains("kind = \"bundled\""), "{source_lock}");
    let warm_output = directory.path().join("warm-source-build");
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", "http://127.0.0.1:9")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "src/main.ch", "-o", warm_output.to_str().unwrap()])
        .assert()
        .success();
    assert!(warm_output.join("main.c").exists());
    fs::remove_file(root.join("reef.lock")).unwrap();

    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", "http://127.0.0.1:9")
        .args(["reef", "build"])
        .assert()
        .success();

    let lock = fs::read_to_string(root.join("reef.lock")).unwrap();
    assert!(lock.contains("name = \"chelis-std\""), "{lock}");
    assert!(lock.contains("kind = \"bundled\""), "{lock}");
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", "http://127.0.0.1:9")
        .args(["reef", "update", "chelis-std"])
        .assert()
        .success();
    assert_eq!(fs::read_to_string(root.join("reef.lock")).unwrap(), lock);

    let manifest = fs::read_to_string(root.join("reef.toml")).unwrap();
    fs::write(
        root.join("reef.toml"),
        manifest.replace(
            &format!(
                "chelis-std = \"={}\"",
                chelis_reef::compiler_bundled_chelis_std_version()
            ),
            "chelis-std = \"=999.0.0\"",
        ),
    )
    .unwrap();
    fs::remove_file(root.join("reef.lock")).unwrap();
    let output = chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", "http://127.0.0.1:9")
        .args(["reef", "build"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("chelis-std"), "{stderr}");
    assert!(
        !stderr.contains("remote discovery is unavailable"),
        "{stderr}"
    );
}

#[test]
fn cyclic_path_dependencies_fail_before_lock_publication() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("path-cycle-root");
    let alpha = directory.path().join("alpha");
    let beta = directory.path().join("beta");
    for package_root in [&root, &alpha, &beta] {
        fs::create_dir_all(package_root.join("src")).unwrap();
        fs::write(package_root.join("src/main.ch"), "module Remote.Main\n").unwrap();
    }
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"path-cycle-root\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nalpha = {{ path = \"../alpha\" }}\n"
        ),
    )
    .unwrap();
    fs::write(
        alpha.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"alpha\"\nversion = \"1.0.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nbeta = {{ path = \"../beta\" }}\n"
        ),
    )
    .unwrap();
    fs::write(
        beta.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"beta\"\nversion = \"1.0.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nalpha = {{ path = \"../alpha\" }}\n"
        ),
    )
    .unwrap();

    chelis(&root)
        .env(
            "CHELIS_REEF_HOME",
            directory.path().join("path-cycle-registry"),
        )
        .args(["reef", "update", "--offline"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "path dependency cycle detected: alpha -> beta -> alpha",
        ));
    assert!(!root.join("reef.lock").exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn remote_candidate_path_dependencies_cannot_escape_temporary_storage() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("remote-path-app");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"remote-path-app\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nnautilus = \"^1\"\n"
        ),
    )
    .unwrap();
    fs::write(root.join("src/main.ch"), "module Remote.Main\n").unwrap();
    let remote_manifest = format!(
        "schema = \"2\"\n\n[package]\nname = \"nautilus\"\nversion = \"1.0.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nvictim = {{ path = \"../../../../victim\" }}\n"
    );
    let archive =
        candidate_archive(&[("reef.toml", EntryType::Regular, remote_manifest.as_bytes())]);
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/nautilus/releases"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(release_json("nautilus", &[("1.0.0", 710, 711)])),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/nautilus/releases/assets/710"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(archive))
        .mount(&server)
        .await;
    chelis(&root)
        .env(
            "CHELIS_REEF_HOME",
            directory.path().join("remote-path-registry"),
        )
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("path dependency `victim`")
                .and(predicate::str::contains("../../../../victim")),
        );
    assert!(!root.join("reef.lock").exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_and_inconsistent_provider_results_are_excluded_with_context() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("provider-error-app");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"provider-error-app\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nnautilus = \"^1\"\n"
        ),
    )
    .unwrap();
    fs::write(root.join("src/main.ch"), "module Remote.Main\n").unwrap();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/nautilus/releases"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
            {
                "tag_name": "v1.0.0+portable",
                "draft": false,
                "prerelease": false,
                "assets": []
            },
            {
                "tag_name": "v1.1.0",
                "draft": false,
                "prerelease": true,
                "assets": []
            }
        ])))
        .mount(&server)
        .await;
    let registry = directory.path().join("provider-error-registry");
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("excluded provider results")
                .and(predicate::str::contains("build metadata"))
                .and(predicate::str::contains("GitHub prerelease")),
        );
    assert!(!root.join("reef.lock").exists());
    assert!(!registry.join("index.json").exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn release_pagination_stops_before_an_eleventh_request() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("page-limit-app");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"page-limit-app\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nnautilus = \"^1\"\n"
        ),
    )
    .unwrap();
    fs::write(root.join("src/main.ch"), "module Remote.Main\n").unwrap();
    let releases = serde_json::Value::Array(
        (0..100)
            .map(|index| {
                serde_json::json!({
                    "tag_name": format!("v1.0.{index}"),
                    "draft": true,
                    "prerelease": false,
                    "assets": []
                })
            })
            .collect(),
    );
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/nautilus/releases"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("link", format!("<{}>; rel=\"next\"", server.uri()))
                .set_body_json(releases),
        )
        .mount(&server)
        .await;
    let registry = directory.path().join("page-limit-registry");
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "GitHub release pages per package limit 10",
        ));
    assert_eq!(server.received_requests().await.unwrap().len(), 10);
    assert!(!root.join("reef.lock").exists());
    assert!(!registry.join("index.json").exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn refresh_queries_remote_publishes_verified_bytes_and_replaces_the_lock_last() {
    let directory = tempdir().unwrap();
    let artifacts = directory.path().join("artifacts");
    fs::create_dir_all(&artifacts).unwrap();
    let (archive_100, shell_100) = build_release_artifacts(&artifacts, "nautilus", "1.0.0");
    let (archive_110, shell_110) = build_release_artifacts(&artifacts, "nautilus", "1.1.0");
    let archive_100_bytes = fs::read(&archive_100).unwrap();
    let archive_110_bytes = fs::read(&archive_110).unwrap();
    let shell_110_bytes = fs::read(&shell_110).unwrap();

    let registry = directory.path().join("registry");
    chelis_reef::install_validated_artifact_pair(
        &archive_100,
        &shell_100,
        "nautilus",
        "1.0.0",
        &registry,
        Some("github://chelis-lang/nautilus@v1.0.0"),
    )
    .unwrap();

    let root = directory.path().join("app");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"remote-app\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nnautilus = \"^1\"\n"
        ),
    )
    .unwrap();
    fs::write(
        root.join("src/main.ch"),
        "module Remote.Main\n\ndef main() -> int32 = 1\n",
    )
    .unwrap();

    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .args(["reef", "update", "--offline"])
        .assert()
        .success()
        .stdout(predicate::str::contains("nautilus").and(predicate::str::contains("1.0.0")));
    let initial_lock = fs::read_to_string(root.join("reef.lock")).unwrap();
    assert!(
        initial_lock.contains("version = \"1.0.0\""),
        "{initial_lock}"
    );

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/nautilus/releases"))
        .and(query_param("per_page", "100"))
        .and(query_param("page", "1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(release_json(
            "nautilus",
            &[("1.0.0", 100, 101), ("1.1.0", 110, 111)],
        )))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/nautilus/releases/assets/110"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_110_bytes))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/nautilus/releases/assets/111"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(shell_110_bytes))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/nautilus/releases/assets/100"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(archive_100_bytes))
        .mount(&server)
        .await;

    fs::remove_file(root.join("reef.lock")).unwrap();
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "schema"])
        .assert()
        .success();
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        0,
        "a complete local graph used a provider"
    );
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update", "--offline"])
        .assert()
        .success();
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        0,
        "offline refresh used a provider"
    );

    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("Updated nautilus 1.0.0 -> 1.1.0")
                .and(predicate::str::contains("1.0.0 -> 1.1.0")),
        );
    let updated_lock = fs::read_to_string(root.join("reef.lock")).unwrap();
    assert!(
        updated_lock.contains("version = \"1.1.0\""),
        "{updated_lock}"
    );
    assert!(updated_lock.contains("github://chelis-lang/nautilus@v1.1.0"));
    let package_dir = registry.join("packages/nautilus/1.1.0");
    assert!(package_dir.join("nautilus-1.1.0.tar.zst").is_file());
    assert!(package_dir.join("nautilus-1.1.0.chb").is_file());

    let index_before = fs::read(registry.join("index.json")).unwrap();
    let lock_before = fs::read(root.join("reef.lock")).unwrap();
    let package_before = fs::read_dir(&package_dir).unwrap().count();
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "outdated", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"newest_compatible\": \"1.1.0\""));
    assert_eq!(fs::read(registry.join("index.json")).unwrap(), index_before);
    assert_eq!(fs::read(root.join("reef.lock")).unwrap(), lock_before);
    assert_eq!(fs::read_dir(&package_dir).unwrap().count(), package_before);

    let requests_before = server.received_requests().await.unwrap().len();
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "schema"])
        .assert()
        .success();
    let requests_after = server.received_requests().await.unwrap().len();
    assert_eq!(
        requests_after, requests_before,
        "a valid lock listed releases"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn targeted_refresh_keeps_unrelated_locks_and_allows_required_transitive_changes() {
    let directory = tempdir().unwrap();
    let artifacts = directory.path().join("targeted-artifacts");
    let build_registry = directory.path().join("targeted-build-registry");
    fs::create_dir_all(&artifacts).unwrap();

    let (alpha_100_archive, alpha_100_shell) =
        build_release_artifacts(&artifacts, "alpha", "1.0.0");
    let (gamma_100_archive, gamma_100_shell) =
        build_release_artifacts(&artifacts, "gamma", "1.0.0");
    let (beta_200_archive, beta_200_shell) = build_release_artifacts(&artifacts, "beta", "2.0.0");
    chelis_reef::install_validated_artifact_pair(
        &beta_200_archive,
        &beta_200_shell,
        "beta",
        "2.0.0",
        &build_registry,
        Some("github://chelis-lang/beta@v2.0.0"),
    )
    .unwrap();
    let (alpha_110_archive, alpha_110_shell) = build_release_with_dependencies(
        &artifacts,
        "alpha",
        "1.1.0",
        "beta = \"^2\"\n",
        &build_registry,
    );

    let registry = directory.path().join("targeted-registry");
    for (name, archive, shell) in [
        ("alpha", &alpha_100_archive, &alpha_100_shell),
        ("gamma", &gamma_100_archive, &gamma_100_shell),
    ] {
        chelis_reef::install_validated_artifact_pair(
            archive,
            shell,
            name,
            "1.0.0",
            &registry,
            Some(&format!("github://chelis-lang/{name}@v1.0.0")),
        )
        .unwrap();
    }

    let root = directory.path().join("targeted-app");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"targeted-app\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nalpha = \"^1\"\ngamma = \"^1\"\n"
        ),
    )
    .unwrap();
    fs::write(
        root.join("src/main.ch"),
        "module Remote.Main\n\ndef main() -> int32 = 1\n",
    )
    .unwrap();
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .args(["reef", "update", "--offline"])
        .assert()
        .success();

    let server = MockServer::start().await;
    let dead_end_manifest = format!(
        "schema = \"2\"\n[package]\nname = \"alpha\"\nversion = \"1.2.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n[dependencies]\nc = \"^2\"\n"
    );
    let dead_end_archive = candidate_archive(&[(
        "reef.toml",
        EntryType::Regular,
        dead_end_manifest.as_bytes(),
    )]);
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/alpha/releases"))
        .respond_with(ResponseTemplate::new(200).set_body_json(release_json(
            "alpha",
            &[("1.2.0", 212, 213), ("1.1.0", 210, 211)],
        )))
        .mount(&server)
        .await;
    for (id, bytes) in [
        (212, dead_end_archive),
        (210, fs::read(&alpha_110_archive).unwrap()),
        (211, fs::read(&alpha_110_shell).unwrap()),
    ] {
        Mock::given(method("GET"))
            .and(wire_path(format!(
                "/repos/chelis-lang/alpha/releases/assets/{id}"
            )))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes))
            .mount(&server)
            .await;
    }
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/c/releases"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
        .mount(&server)
        .await;
    mount_single_release(
        &server,
        "beta",
        "2.0.0",
        220,
        221,
        fs::read(&beta_200_archive).unwrap(),
        fs::read(&beta_200_shell).unwrap(),
    )
    .await;

    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update", "alpha"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("Updated alpha 1.0.0 -> 1.1.0")
                .and(predicate::str::contains("Updated beta <none> -> 2.0.0")),
        );

    let lock = fs::read_to_string(root.join("reef.lock")).unwrap();
    assert!(
        lock.contains("name = \"alpha\"\nversion = \"1.1.0\""),
        "{lock}"
    );
    assert!(
        lock.contains("name = \"beta\"\nversion = \"2.0.0\""),
        "{lock}"
    );
    assert!(
        lock.contains("name = \"gamma\"\nversion = \"1.0.0\""),
        "{lock}"
    );
    let requests = server.received_requests().await.unwrap();
    assert!(
        requests
            .iter()
            .all(|request| !request.url.path().contains("gamma"))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn exact_requirements_report_incompatible_remote_versions_without_changes() {
    let directory = tempdir().unwrap();
    let artifacts = directory.path().join("exact-artifacts");
    fs::create_dir_all(&artifacts).unwrap();
    let (archive_100, shell_100) = build_release_artifacts(&artifacts, "nautilus", "1.0.0");
    let (archive_110, shell_110) = build_release_artifacts(&artifacts, "nautilus", "1.1.0");
    let registry = directory.path().join("exact-registry");
    chelis_reef::install_validated_artifact_pair(
        &archive_100,
        &shell_100,
        "nautilus",
        "1.0.0",
        &registry,
        Some("github://chelis-lang/nautilus@v1.0.0"),
    )
    .unwrap();
    let root = directory.path().join("exact-app");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"2\"\n\n[package]\nname = \"exact-app\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Remote\"\nresolver = \"2\"\n\n[dependencies]\nnautilus = \"=1.0.0\"\n"
        ),
    )
    .unwrap();
    fs::write(
        root.join("src/main.ch"),
        "module Remote.Main\n\ndef main() -> int32 = 1\n",
    )
    .unwrap();
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .args(["reef", "update", "--offline"])
        .assert()
        .success();

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wire_path("/repos/chelis-lang/nautilus/releases"))
        .respond_with(ResponseTemplate::new(200).set_body_json(release_json(
            "nautilus",
            &[("1.1.0", 410, 411), ("1.0.0", 400, 401)],
        )))
        .mount(&server)
        .await;
    for (id, bytes) in [
        (410, fs::read(&archive_110).unwrap()),
        (400, fs::read(&archive_100).unwrap()),
    ] {
        Mock::given(method("GET"))
            .and(wire_path(format!(
                "/repos/chelis-lang/nautilus/releases/assets/{id}"
            )))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes))
            .mount(&server)
            .await;
    }
    let lock_before = fs::read(root.join("reef.lock")).unwrap();
    let index_before = fs::read(registry.join("index.json")).unwrap();
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "outdated", "--json"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("\"current\": \"1.0.0\"")
                .and(predicate::str::contains("\"newest_compatible\": \"1.0.0\""))
                .and(predicate::str::contains(
                    "\"newest_incompatible\": \"1.1.0\"",
                ))
                .and(predicate::str::contains("exact requirement")),
        );
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "outdated"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "current=1.0.0, compatible=1.0.0, incompatible=1.1.0",
        ));
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .success()
        .stdout(predicate::str::contains("All Reef packages are current."));
    assert_eq!(fs::read(root.join("reef.lock")).unwrap(), lock_before);
    assert_eq!(fs::read(registry.join("index.json")).unwrap(), index_before);
    let _ = shell_110;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_missing_local_directory_refreshes_from_the_locked_repository() {
    let directory = tempdir().unwrap();
    let artifacts = directory.path().join("locked-source-artifacts");
    fs::create_dir_all(&artifacts).unwrap();
    let (archive_100, shell_100) = build_release_artifacts(&artifacts, "nautilus", "1.0.0");
    let (archive_110, shell_110) = build_release_artifacts(&artifacts, "nautilus", "1.1.0");
    let (root, registry, _) =
        stage_update_project(directory.path(), "locked-source", &archive_100, &shell_100);
    let lock_path = root.join("reef.lock");
    let lock = fs::read_to_string(&lock_path)
        .unwrap()
        .replace("github://chelis-lang/nautilus", "github://acme/nautilus");
    fs::write(&lock_path, lock).unwrap();
    fs::remove_dir_all(registry.join("packages/nautilus/1.0.0")).unwrap();

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wire_path("/repos/acme/nautilus/releases"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(release_json("nautilus", &[("1.1.0", 610, 611)])),
        )
        .mount(&server)
        .await;
    for (id, bytes) in [
        (610, fs::read(&archive_110).unwrap()),
        (611, fs::read(&shell_110).unwrap()),
    ] {
        Mock::given(method("GET"))
            .and(wire_path(format!(
                "/repos/acme/nautilus/releases/assets/{id}"
            )))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes))
            .mount(&server)
            .await;
    }
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .success();
    let updated_lock = fs::read_to_string(root.join("reef.lock")).unwrap();
    assert!(updated_lock.contains("github://acme/nautilus@v1.1.0"));
    assert!(
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|request| request.url.path().contains("/repos/acme/"))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn remote_candidate_compiler_drift_is_excluded_before_publication() {
    let directory = tempdir().unwrap();
    let artifacts = directory.path().join("compiler-drift-artifacts");
    fs::create_dir_all(&artifacts).unwrap();
    let (archive_100, shell_100) = build_release_artifacts(&artifacts, "nautilus", "1.0.0");
    let (archive_110, shell_110) =
        build_release_with_compiler(&artifacts, "nautilus", "1.1.0", "=0.0.1");
    let (root, registry, lock_before) =
        stage_update_project(directory.path(), "compiler-drift", &archive_100, &shell_100);
    let index_before = fs::read(registry.join("index.json")).unwrap();
    let server = MockServer::start().await;
    mount_single_release(
        &server,
        "nautilus",
        "1.1.0",
        510,
        511,
        fs::read(archive_110).unwrap(),
        fs::read(shell_110).unwrap(),
    )
    .await;
    chelis(&root)
        .env("CHELIS_REEF_HOME", &registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .success()
        .stdout(predicate::str::contains("All Reef packages are current."));
    assert_eq!(fs::read(root.join("reef.lock")).unwrap(), lock_before);
    assert_eq!(fs::read(registry.join("index.json")).unwrap(), index_before);
    assert!(!registry.join("packages/nautilus/1.1.0").exists());
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn publication_failures_preserve_the_prior_lock_and_only_leave_complete_cache_entries() {
    use std::os::unix::fs::symlink;

    let directory = tempdir().unwrap();
    let artifacts = directory.path().join("failure-artifacts");
    fs::create_dir_all(&artifacts).unwrap();
    let (archive_100, shell_100) = build_release_artifacts(&artifacts, "nautilus", "1.0.0");
    let (archive_110, shell_110) = build_release_artifacts(&artifacts, "nautilus", "1.1.0");
    let archive_110_bytes = fs::read(&archive_110).unwrap();
    let shell_110_bytes = fs::read(&shell_110).unwrap();
    let (different_archive_110, different_shell_110) =
        build_release_with_body(&artifacts, "nautilus", "1.1.0", 2);

    let bad_server = MockServer::start().await;
    mount_single_release(
        &bad_server,
        "nautilus",
        "1.1.0",
        310,
        311,
        archive_110_bytes.clone(),
        b"invalid shell".to_vec(),
    )
    .await;
    let (bad_root, bad_registry, bad_lock) =
        stage_update_project(directory.path(), "bad-pair", &archive_100, &shell_100);
    let bad_index = fs::read(bad_registry.join("index.json")).unwrap();
    chelis(&bad_root)
        .env("CHELIS_REEF_HOME", &bad_registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", bad_server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .failure();
    assert_eq!(fs::read(bad_root.join("reef.lock")).unwrap(), bad_lock);
    assert_eq!(
        fs::read(bad_registry.join("index.json")).unwrap(),
        bad_index
    );
    assert!(!bad_registry.join("packages/nautilus/1.1.0").exists());

    let good_server = MockServer::start().await;
    mount_single_release(
        &good_server,
        "nautilus",
        "1.1.0",
        320,
        321,
        archive_110_bytes.clone(),
        shell_110_bytes.clone(),
    )
    .await;
    let (index_root, index_registry, index_lock) =
        stage_update_project(directory.path(), "index-failure", &archive_100, &shell_100);
    let index_before = fs::read(index_registry.join("index.json")).unwrap();
    fs::create_dir(index_registry.join("index.json.tmp")).unwrap();
    chelis(&index_root)
        .env("CHELIS_REEF_HOME", &index_registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", good_server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("index"));
    assert_eq!(fs::read(index_root.join("reef.lock")).unwrap(), index_lock);
    assert_eq!(
        fs::read(index_registry.join("index.json")).unwrap(),
        index_before
    );
    let index_package = index_registry.join("packages/nautilus/1.1.0");
    assert!(index_package.join("nautilus-1.1.0.tar.zst").is_file());
    assert!(index_package.join("nautilus-1.1.0.chb").is_file());

    let (conflict_root, conflict_registry, conflict_lock) =
        stage_update_project(directory.path(), "byte-conflict", &archive_100, &shell_100);
    let conflict_package = conflict_registry.join("packages/nautilus/1.1.0");
    fs::create_dir_all(&conflict_package).unwrap();
    fs::copy(
        &different_archive_110,
        conflict_package.join("nautilus-1.1.0.tar.zst"),
    )
    .unwrap();
    fs::copy(
        &different_shell_110,
        conflict_package.join("nautilus-1.1.0.chb"),
    )
    .unwrap();
    chelis(&conflict_root)
        .env("CHELIS_REEF_HOME", &conflict_registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", good_server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("differs by bytes"));
    assert_eq!(
        fs::read(conflict_root.join("reef.lock")).unwrap(),
        conflict_lock
    );
    assert_eq!(
        fs::read(conflict_package.join("nautilus-1.1.0.tar.zst")).unwrap(),
        fs::read(different_archive_110).unwrap()
    );

    let (identical_root, identical_registry, _) =
        stage_update_project(directory.path(), "identical-race", &archive_100, &shell_100);
    let identical_package = identical_registry.join("packages/nautilus/1.1.0");
    fs::create_dir_all(&identical_package).unwrap();
    fs::copy(
        &archive_110,
        identical_package.join("nautilus-1.1.0.tar.zst"),
    )
    .unwrap();
    fs::copy(&shell_110, identical_package.join("nautilus-1.1.0.chb")).unwrap();
    chelis(&identical_root)
        .env("CHELIS_REEF_HOME", &identical_registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", good_server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .success();
    let identical_lock = fs::read_to_string(identical_root.join("reef.lock")).unwrap();
    assert!(identical_lock.contains("version = \"1.1.0\""));

    let (lock_root, lock_registry, lock_before) =
        stage_update_project(directory.path(), "lock-failure", &archive_100, &shell_100);
    let lock_path = lock_root.join("reef.lock");
    let lock_target = lock_root.join("retained-lock.toml");
    fs::write(&lock_target, &lock_before).unwrap();
    fs::remove_file(&lock_path).unwrap();
    symlink(&lock_target, &lock_path).unwrap();
    chelis(&lock_root)
        .env("CHELIS_REEF_HOME", &lock_registry)
        .env("CHELIS_REEF_GITHUB_BASE_API", good_server.uri())
        .env("GITHUB_TOKEN", "unit-test-token")
        .args(["reef", "update"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("symbolic-link"));
    assert_eq!(fs::read(&lock_target).unwrap(), lock_before);
    let lock_package = lock_registry.join("packages/nautilus/1.1.0");
    assert!(lock_package.join("nautilus-1.1.0.tar.zst").is_file());
    assert!(lock_package.join("nautilus-1.1.0.chb").is_file());
    let index = fs::read_to_string(lock_registry.join("index.json")).unwrap();
    assert!(index.contains("1.1.0"), "{index}");
}
