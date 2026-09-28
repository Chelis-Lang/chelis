use assert_cmd::Command;
use chelis_reef::{
    METADATA_FILE_MAX_BYTES, METADATA_TOTAL_MAX_BYTES, PackageDescription, PackageMetadataError,
    PackageUrl, PortablePackagePath, SpdxLicense, build_package, manifest_schema_v3_json,
    read_manifest_for_src, snapshot_declared_metadata_file,
};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Read};
use std::path::Path;
use std::str::FromStr;
use tar::Archive;
use tempfile::tempdir;

const COMPILER_PIN: &str = concat!("=", env!("CARGO_PKG_VERSION"));

fn chelis(root: &Path) -> Command {
    let mut command = Command::new(assert_cmd::cargo::cargo_bin!("chelis"));
    command.current_dir(root);
    command
}

fn schema_three_manifest(name: &str, metadata: &str) -> String {
    format!(
        "#:schema https://raw.githubusercontent.com/Chelis-Lang/chelis/main/docs/schemas/reef/manifest-v3.schema.json\n\
         schema = \"3\"\n\n\
         [package]\n\
         name = \"{name}\"\n\
         version = \"0.1.0\"\n\
         compiler = \"{COMPILER_PIN}\"\n\
         module_prefix = \"Metadata\"\n\
         resolver = \"2\"\n\
         {metadata}\n"
    )
}

fn stage_manifest(root: &Path, manifest: &str) {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("reef.toml"), manifest).unwrap();
}

fn stage_buildable_package(root: &Path, metadata: &str) {
    stage_manifest(root, &schema_three_manifest("metadata-oracle", metadata));
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/main.ch"),
        "module Metadata.Main\n\nexport (answer)\ndef answer() -> i32 = 42\n",
    )
    .unwrap();
}

fn archive_entry_paths(path: &Path) -> Vec<String> {
    let bytes = fs::read(path).unwrap();
    let decoded = zstd::stream::decode_all(Cursor::new(bytes)).unwrap();
    let mut archive = Archive::new(Cursor::new(decoded));
    archive
        .entries()
        .unwrap()
        .map(|entry| {
            entry
                .unwrap()
                .path()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

fn archive_members(path: &Path) -> BTreeMap<String, (Vec<u8>, u32, u64, u64, u64)> {
    let bytes = fs::read(path).unwrap();
    let decoded = zstd::stream::decode_all(Cursor::new(bytes)).unwrap();
    let mut archive = Archive::new(Cursor::new(decoded));
    let mut members = BTreeMap::new();
    for entry in archive.entries().unwrap() {
        let mut entry = entry.unwrap();
        let path = entry.path().unwrap().to_string_lossy().into_owned();
        let header = entry.header().clone();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        members.insert(
            path,
            (
                bytes,
                header.mode().unwrap(),
                header.uid().unwrap(),
                header.gid().unwrap(),
                header.mtime().unwrap(),
            ),
        );
    }
    members
}

#[test]
fn schema_three_parses_typed_metadata_and_schema_two_rejects_it() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    stage_manifest(
        root,
        &schema_three_manifest(
            "typed-metadata",
            "description = \"  Tensor primitives for Chelis  \"\n\
             license = \"MIT OR Apache-2.0\"\n\
             repository = \"https://example.com/source\"\n\
             documentation = \"https://docs.example.com/pkg\"\n\
             homepage = \"https://example.com/pkg\"\n\
             readme = \"docs/README.md\"",
        ),
    );

    let manifest = read_manifest_for_src(root).unwrap();
    let metadata = manifest.package.metadata;
    assert_eq!(
        metadata.description().unwrap().as_str(),
        "Tensor primitives for Chelis"
    );
    assert_eq!(metadata.license().unwrap().as_str(), "MIT OR Apache-2.0");
    assert_eq!(
        metadata.repository().unwrap().as_str(),
        "https://example.com/source"
    );
    assert_eq!(metadata.readme().unwrap().as_str(), "docs/README.md");

    let schema_two = fs::read_to_string(root.join("reef.toml"))
        .unwrap()
        .replace("manifest-v3", "manifest-v2")
        .replace("schema = \"3\"", "schema = \"2\"");
    fs::write(root.join("reef.toml"), schema_two).unwrap();
    let error = read_manifest_for_src(root).unwrap_err();
    assert!(
        error.contains("description") && error.contains("[package]"),
        "{error}"
    );
}

#[test]
fn legacy_schema_cannot_bypass_the_flat_schema_three_metadata_boundary() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    stage_manifest(
        root,
        &format!(
            "[package]\nname = \"legacy-metadata\"\nversion = \"0.1.0\"\ncompiler = \"{COMPILER_PIN}\"\nmodule_prefix = \"Legacy\"\n\n[package.metadata]\nreadme = \"README.md\"\n"
        ),
    );
    let error = read_manifest_for_src(root).unwrap_err();
    assert!(
        error.contains("schema 0") && error.contains("upgrade to schema 3"),
        "{error}"
    );
}

#[test]
fn descriptions_enforce_trimmed_byte_and_character_rules() {
    assert_eq!(
        PackageDescription::from_str("  useful package  ")
            .unwrap()
            .as_str(),
        "useful package"
    );
    for invalid in [
        "",
        "   ",
        "two\nlines",
        "has\u{7f}control",
        "line\u{2028}separator",
        "bidi\u{202e}override",
        "arabic\u{0600}format",
    ] {
        assert!(
            PackageDescription::from_str(invalid).is_err(),
            "accepted {invalid:?}"
        );
    }
    assert!(PackageDescription::from_str(&"é".repeat(256)).is_ok());
    assert!(PackageDescription::from_str(&"é".repeat(257)).is_err());
}

#[test]
fn licenses_use_spdx_and_the_two_forms_are_exclusive() {
    assert!(SpdxLicense::from_str("MIT OR Apache-2.0").is_ok());
    assert!(SpdxLicense::from_str("some permissive license").is_err());
    let overlong = "MIT OR ".repeat(200);
    let error = SpdxLicense::from_str(&overlong).unwrap_err().to_string();
    assert!(error.contains("1024"), "{error}");

    let directory = tempdir().unwrap();
    stage_manifest(
        directory.path(),
        &schema_three_manifest(
            "duplicate-license",
            "license = \"MIT\"\nlicense-file = \"LICENSE.custom\"",
        ),
    );
    let error = read_manifest_for_src(directory.path()).unwrap_err();
    assert!(
        error.contains("package.license") && error.contains("package.license-file"),
        "{error}"
    );
}

#[test]
fn metadata_urls_require_absolute_credential_free_https() {
    assert_eq!(
        PackageUrl::from_str("https://example.com/docs")
            .unwrap()
            .as_str(),
        "https://example.com/docs"
    );
    assert_eq!(
        PackageUrl::from_str("HTTPS://EXAMPLE.COM")
            .unwrap()
            .as_str(),
        "https://example.com/"
    );
    let overlong = format!("https://example.com/{}", "a".repeat(2048));
    let error = PackageUrl::from_str(&overlong).unwrap_err().to_string();
    assert!(error.contains("2048"), "{error}");
    let normalized_overlong = format!("https://example.com/{}", "é".repeat(400));
    assert!(normalized_overlong.len() < 2048);
    let error = PackageUrl::from_str(&normalized_overlong)
        .unwrap_err()
        .to_string();
    assert!(error.contains("after URL normalization"), "{error}");
    for invalid in [
        "http://example.com",
        "https:///missing-host",
        "docs/index.html",
        "https://user@example.com",
        "https://user:secret@example.com",
    ] {
        assert!(PackageUrl::from_str(invalid).is_err(), "accepted {invalid}");
    }

    let directory = tempdir().unwrap();
    stage_manifest(
        directory.path(),
        &schema_three_manifest("invalid-url", "repository = \"http://example.com\""),
    );
    let error = read_manifest_for_src(directory.path()).unwrap_err();
    assert!(
        error.contains("package.repository") && error.contains("HTTPS"),
        "{error}"
    );
}

#[test]
fn portable_paths_enforce_every_host_independent_rule() {
    let path = PortablePackagePath::from_str("docs/README.md").unwrap();
    assert_eq!(path.as_str(), "docs/README.md");
    assert_eq!(path.segments(), &["docs", "README.md"]);
    assert!(PortablePackagePath::from_str(&"a".repeat(255)).is_ok());
    let exact_path = format!(
        "{}/{}/{}/{}/tail",
        "a".repeat(254),
        "b".repeat(254),
        "c".repeat(254),
        "d".repeat(254)
    );
    assert_eq!(exact_path.len(), 1024);
    assert!(PortablePackagePath::from_str(&exact_path).is_ok());

    let decomposed = "docs/e\u{301}.md";
    let too_long_path = format!("{}/b", "a".repeat(1023));
    let too_long_segment = format!("docs/{}", "a".repeat(256));
    let invalid = [
        "",
        "../README.md",
        "./README.md",
        "/README.md",
        "docs//README.md",
        "docs\\README.md",
        "docs/read\u{7f}me",
        "docs/read\u{85}me",
        "docs/bidi\u{202e}name",
        "docs/read:me",
        "docs/readme ",
        "docs/readme.",
        "docs/CON.txt",
        "docs/COM1 .txt",
        "docs/lPt9.any",
        "reef.toml",
        "REEF.LOCK",
        decomposed,
        too_long_path.as_str(),
        too_long_segment.as_str(),
    ];
    for value in invalid {
        assert!(
            PortablePackagePath::from_str(value).is_err(),
            "accepted {value:?}"
        );
    }

    let directory = tempdir().unwrap();
    stage_manifest(
        directory.path(),
        &schema_three_manifest("invalid-readme", "readme = \"docs/CON.txt\""),
    );
    let error = read_manifest_for_src(directory.path()).unwrap_err();
    assert!(
        error.contains("package.readme")
            && error.contains("CON.txt")
            && error.contains("Windows device name"),
        "{error}"
    );
}

#[test]
fn declared_file_snapshots_are_bounded_stable_and_handle_relative() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fs::create_dir_all(root.join("docs")).unwrap();
    fs::write(root.join("docs/README.md"), b"stable bytes").unwrap();
    let path = PortablePackagePath::from_str("docs/README.md").unwrap();

    let snapshot = snapshot_declared_metadata_file(root, &path).unwrap();
    assert_eq!(snapshot.path(), &path);
    assert_eq!(snapshot.bytes(), b"stable bytes");

    fs::write(root.join("docs/README.md"), b"replacement bytes").unwrap();
    assert_eq!(snapshot.bytes(), b"stable bytes");

    let directory_path = PortablePackagePath::from_str("docs").unwrap();
    assert!(snapshot_declared_metadata_file(root, &directory_path).is_err());
    let missing = PortablePackagePath::from_str("docs/missing.md").unwrap();
    assert!(snapshot_declared_metadata_file(root, &missing).is_err());
}

#[test]
fn snapshot_size_limit_fails_closed() {
    assert_eq!(METADATA_TOTAL_MAX_BYTES, 2 * METADATA_FILE_MAX_BYTES);
    let directory = tempdir().unwrap();
    let root = directory.path();
    let path = PortablePackagePath::from_str("README.md").unwrap();
    fs::write(root.join("README.md"), vec![b'x'; METADATA_FILE_MAX_BYTES]).unwrap();
    assert_eq!(
        snapshot_declared_metadata_file(root, &path)
            .unwrap()
            .bytes()
            .len(),
        METADATA_FILE_MAX_BYTES
    );
    fs::write(
        root.join("README.md"),
        vec![b'x'; METADATA_FILE_MAX_BYTES + 1],
    )
    .unwrap();
    let error = snapshot_declared_metadata_file(root, &path)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("4194304") && error.contains("README.md"),
        "{error}"
    );
}

#[cfg(unix)]
#[test]
fn no_follow_walk_rejects_final_and_parent_symbolic_links() {
    use std::os::unix::fs::symlink;

    let directory = tempdir().unwrap();
    let root = directory.path();
    fs::create_dir_all(root.join("real")).unwrap();
    fs::write(root.join("real/README.md"), b"secret target").unwrap();

    symlink(root.join("real/README.md"), root.join("README.md")).unwrap();
    let final_link = PortablePackagePath::from_str("README.md").unwrap();
    let error = snapshot_declared_metadata_file(root, &final_link).unwrap_err();
    assert!(
        matches!(error, PackageMetadataError::SymbolicLink { .. }),
        "{error}"
    );

    symlink(root.join("real"), root.join("docs")).unwrap();
    let parent_link = PortablePackagePath::from_str("docs/README.md").unwrap();
    let error = snapshot_declared_metadata_file(root, &parent_link).unwrap_err();
    assert!(
        matches!(
            error,
            PackageMetadataError::SymbolicLink { .. } | PackageMetadataError::Open { .. }
        ),
        "{error}"
    );
}

#[cfg(unix)]
#[test]
fn nonregular_declared_files_fail_without_a_blocking_open() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let directory = tempdir().unwrap();
    let fifo = directory.path().join("README.pipe");
    let fifo_c = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    let result = unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) };
    assert_eq!(result, 0);
    let path = PortablePackagePath::from_str("README.pipe").unwrap();
    let error = snapshot_declared_metadata_file(directory.path(), &path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("not one regular file"), "{error}");
}

#[test]
fn source_archive_contains_only_declared_metadata_with_canonical_headers() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    stage_buildable_package(
        root,
        "description = \"Archive metadata\"\n\
         license-file = \"LEGAL.txt\"\n\
         readme = \"docs/README.md\"",
    );
    fs::create_dir_all(root.join("docs")).unwrap();
    fs::write(root.join("docs/README.md"), b"readme bytes").unwrap();
    fs::write(root.join("LEGAL.txt"), b"license bytes").unwrap();
    fs::write(root.join("README-undiscovered.md"), b"must stay out").unwrap();

    let first = build_package(root).unwrap();
    let first_bytes = fs::read(&first.archive_path).unwrap();
    let second = build_package(root).unwrap();
    let second_bytes = fs::read(&second.archive_path).unwrap();
    assert_eq!(first_bytes, second_bytes);

    let paths = archive_entry_paths(&second.archive_path);
    let mut sorted_paths = paths.clone();
    sorted_paths.sort();
    assert_eq!(paths, sorted_paths, "archive member order is not canonical");
    let members = archive_members(&second.archive_path);
    assert_eq!(members["docs/README.md"].0, b"readme bytes");
    assert_eq!(members["LEGAL.txt"].0, b"license bytes");
    assert!(!members.contains_key("README-undiscovered.md"));
    let expected_mtime = members.values().next().unwrap().4;
    for (mode, uid, gid, mtime) in members
        .values()
        .map(|entry| (entry.1, entry.2, entry.3, entry.4))
    {
        assert_eq!((mode, uid, gid, mtime), (0o644, 0, 0, expected_mtime));
    }

    fs::write(root.join("docs/README.md"), b"changed readme bytes").unwrap();
    let changed_manifest = fs::read_to_string(root.join("reef.toml"))
        .unwrap()
        .replace("Archive metadata", "Changed archive metadata");
    fs::write(root.join("reef.toml"), changed_manifest).unwrap();
    let changed = build_package(root).unwrap();
    assert_eq!(changed.package, second.package);
    assert_ne!(fs::read(changed.archive_path).unwrap(), second_bytes);
}

#[test]
fn duplicate_declared_and_source_paths_produce_one_member_each() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    stage_buildable_package(
        root,
        "license-file = \"src/NOTICE.md\"\nreadme = \"src/NOTICE.md\"",
    );
    fs::write(root.join("src/NOTICE.md"), b"one snapshot").unwrap();
    let build = build_package(root).unwrap();
    let members = archive_members(&build.archive_path);
    assert_eq!(
        members
            .keys()
            .filter(|path| *path == "src/NOTICE.md")
            .count(),
        1
    );
    assert_eq!(members["src/NOTICE.md"].0, b"one snapshot");
}

#[test]
fn metadata_paths_do_not_widen_the_existing_source_root_grammar() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    stage_buildable_package(root, "");
    fs::write(root.join("src/CON.txt"), b"existing source member").unwrap();
    let build = build_package(root).unwrap();
    let members = archive_members(&build.archive_path);
    assert_eq!(members["src/CON.txt"].0, b"existing source member");
}

#[test]
fn portable_spelling_collisions_fail_instead_of_creating_duplicate_members() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    stage_buildable_package(root, "readme = \"src/README.md\"");
    fs::write(root.join("src/README.md"), b"declared bytes").unwrap();
    fs::write(root.join("src/Readme.md"), b"case collision").unwrap();
    if cfg!(target_os = "macos") {
        let build = build_package(root).unwrap();
        let members = archive_members(&build.archive_path);
        assert_eq!(
            members
                .keys()
                .filter(|path| path.eq_ignore_ascii_case("src/README.md"))
                .count(),
            1
        );
    } else {
        let error = build_package(root).unwrap_err();
        assert!(error.contains("portable spelling collision"), "{error}");
    }
}

#[cfg(unix)]
#[test]
fn distinct_declared_hard_link_paths_remain_distinct_members() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    stage_buildable_package(root, "license-file = \"LICENSE\"\nreadme = \"README.md\"");
    fs::write(root.join("LICENSE"), b"shared inode bytes").unwrap();
    fs::hard_link(root.join("LICENSE"), root.join("README.md")).unwrap();
    let build = build_package(root).unwrap();
    let members = archive_members(&build.archive_path);
    assert_eq!(members["LICENSE"].0, b"shared inode bytes");
    assert_eq!(members["README.md"].0, b"shared inode bytes");
}

#[cfg(unix)]
#[test]
fn concurrent_build_waits_for_the_package_root_project_lock() {
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;
    use std::process::Stdio;
    use std::time::Duration;

    let directory = tempdir().unwrap();
    let root = directory.path();
    stage_buildable_package(root, "readme = \"README.md\"");
    fs::write(root.join("README.md"), b"locked snapshot").unwrap();
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(root.join(".reef-write.lock"))
        .unwrap();
    assert_eq!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) }, 0);

    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("chelis"))
        .current_dir(root)
        .args(["reef", "build"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        child.try_wait().unwrap().is_none(),
        "build ignored project lock"
    );
    assert!(!root.join("dist/metadata-oracle-0.1.0.tar.zst").exists());

    assert_eq!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_UN) }, 0);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn metadata_failure_preserves_the_previous_archive() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    stage_buildable_package(root, "readme = \"README.md\"");
    fs::write(root.join("README.md"), b"valid readme").unwrap();
    let build = build_package(root).unwrap();
    let before = fs::read(&build.archive_path).unwrap();
    let unowned = root.join("dist/.metadata-oracle-0.1.0.tar.zst.reef-tmp-unowned");
    fs::write(&unowned, b"foreign sibling").unwrap();

    fs::write(
        root.join("README.md"),
        vec![b'x'; METADATA_FILE_MAX_BYTES + 1],
    )
    .unwrap();
    let error = build_package(root).unwrap_err();
    assert!(
        error.contains("README.md") && error.contains("4194304"),
        "{error}"
    );
    assert_eq!(fs::read(&build.archive_path).unwrap(), before);
    assert_eq!(fs::read(&unowned).unwrap(), b"foreign sibling");
    assert!(fs::read_dir(root.join("dist")).unwrap().all(|entry| {
        let path = entry.unwrap().path();
        path == unowned
            || !path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("reef-tmp")
    }));
}

#[test]
fn schema_two_upgrades_to_three_without_invented_metadata() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let manifest = format!(
        "#:schema https://example.invalid/manifest-v2.schema.json # keep\n\
         schema = \"2\" # schema comment\n\n\
         [package]\n\
         name = \"upgrade-metadata\"\n\
         version = \"0.1.0\"\n\
         compiler = \"{COMPILER_PIN}\"\n\
         module_prefix = \"Metadata\"\n\
         resolver = \"2\" # resolver comment\n"
    );
    stage_manifest(root, &manifest);

    chelis(root)
        .args(["reef", "upgrade", "--inplace", "--manifest-to", "3"])
        .assert()
        .success();
    let upgraded = fs::read_to_string(root.join("reef.toml")).unwrap();
    assert!(
        upgraded.contains("manifest-v3.schema.json # keep"),
        "{upgraded}"
    );
    assert!(
        upgraded.contains("schema = \"3\" # schema comment"),
        "{upgraded}"
    );
    assert!(
        upgraded.contains("resolver = \"2\" # resolver comment"),
        "{upgraded}"
    );
    for field in ["description", "license", "readme", "homepage"] {
        assert!(!upgraded.contains(field), "invented {field}: {upgraded}");
    }
}

#[test]
fn schema_three_artifact_matches_the_wire_fields_and_init_uses_schema_three() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let committed =
        fs::read_to_string(repository.join("docs/schemas/reef/manifest-v3.schema.json")).unwrap();
    assert_eq!(committed, manifest_schema_v3_json());
    let schema: serde_json::Value = serde_json::from_str(&manifest_schema_v3_json()).unwrap();
    let package = &schema["definitions"]["ManifestPackageWireV3"]["properties"];
    for field in [
        "name",
        "version",
        "compiler",
        "module_prefix",
        "additional_sources",
        "resolver",
        "description",
        "license",
        "license-file",
        "repository",
        "documentation",
        "homepage",
        "readme",
    ] {
        assert!(package.get(field).is_some(), "missing {field}");
    }

    let directory = tempdir().unwrap();
    let root = directory.path().join("fresh");
    chelis(directory.path())
        .args([
            "reef",
            "init",
            "fresh-meta",
            "--module-prefix",
            "FreshMeta",
            "--output",
            root.to_str().unwrap(),
        ])
        .assert()
        .success();
    let manifest = fs::read_to_string(root.join("reef.toml")).unwrap();
    assert!(manifest.contains("schema = \"3\""), "{manifest}");
    assert!(manifest.contains("resolver = \"2\""), "{manifest}");
    assert!(!manifest.contains("description ="), "{manifest}");
}

#[test]
fn metadata_does_not_enter_lock_or_source_selection() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let metadata = "description = \"Not an identity field\"\n\
         repository = \"https://example.com/not-the-source\"\n\
         documentation = \"https://docs.example.com/package\"\n\
         homepage = \"https://example.com/package\"\n\
         license = \"MIT\"\n\
         readme = \"README.md\"";
    stage_buildable_package(root, metadata);
    fs::write(root.join("README.md"), b"compatibility metadata").unwrap();
    let dependency = root.join("dep");
    stage_manifest(
        &dependency,
        &schema_three_manifest("metadata-dep", "").replace(
            "module_prefix = \"Metadata\"",
            "module_prefix = \"MetadataDep\"",
        ),
    );
    fs::create_dir_all(dependency.join("src")).unwrap();
    fs::write(
        dependency.join("src/main.ch"),
        "module MetadataDep.Main\n\nexport (dep)\ndef dep() -> i32 = 1\n",
    )
    .unwrap();
    let mut root_manifest = fs::read_to_string(root.join("reef.toml")).unwrap();
    root_manifest.push_str("\n[dependencies]\nmetadata-dep = { path = \"dep\" }\n");
    fs::write(root.join("reef.toml"), root_manifest).unwrap();
    let build = build_package(root).unwrap();
    let lock = fs::read_to_string(root.join("reef.lock")).unwrap();
    assert!(lock.contains("kind = \"path\"") && lock.contains("path = \"dep\""));
    for field in [
        "description",
        "repository",
        "documentation",
        "homepage",
        "license",
        "readme",
    ] {
        assert!(!lock.contains(field), "metadata entered lock: {lock}");
    }
    let shell = fs::read(&build.shell_path).unwrap();
    for value in [
        "Not an identity field",
        "https://example.com/not-the-source",
        "https://docs.example.com/package",
        "https://example.com/package",
        "README.md",
    ] {
        assert!(
            !shell
                .windows(value.len())
                .any(|bytes| bytes == value.as_bytes()),
            "metadata entered the shell: {value}"
        );
    }

    let archive_before = fs::read(&build.archive_path).unwrap();
    fs::write(root.join("README.md"), b"changed metadata bytes").unwrap();
    let rebuilt = build_package(root).unwrap();
    assert!(
        fs::read(&rebuilt.archive_path).unwrap() != archive_before,
        "declared metadata must change the source archive"
    );
    assert!(
        fs::read(&rebuilt.shell_path).unwrap() != shell,
        "the shell must carry the changed source archive hash"
    );

    let reef_home = directory.path().join("registry");
    let published = directory.path().join("published");
    stage_buildable_package(&published, metadata);
    fs::write(published.join("README.md"), b"published metadata").unwrap();
    chelis(&published)
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish"])
        .assert()
        .success();
    let index = fs::read_to_string(reef_home.join("index.json")).unwrap();
    for field in [
        "description",
        "repository",
        "documentation",
        "homepage",
        "license",
        "readme",
    ] {
        assert!(!index.contains(field), "metadata entered index: {index}");
    }
}
