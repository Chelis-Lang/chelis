use chelis_reef::{install_validated_artifact_pair, verify_artifact_pair};
use chelis_shell::{
    PackageId, SHELL_FORMAT_VERSION, SHELL_MAGIC, ShellModule, ShellPackage, encode_shell,
};
use sha2::{Digest, Sha256};
use std::fs;
use tempfile::tempdir;

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fixture_shell(archive_sha256: String) -> ShellPackage {
    ShellPackage {
        format_version: SHELL_FORMAT_VERSION,
        package: PackageId {
            name: "demo".to_string(),
            version: "1.2.3".to_string(),
        },
        compiler: "=0.17.4".to_string(),
        modules: Vec::new(),
        dependencies: Vec::new(),
        archive_sha256,
    }
}

fn write_pair(
    shell: &ShellPackage,
    archive: &[u8],
) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let archive_path = dir.path().join("demo-1.2.3.tar.zst");
    let shell_path = dir.path().join("demo-1.2.3.chb");
    fs::write(&archive_path, archive).expect("write archive");
    fs::write(&shell_path, encode_shell(shell).expect("encode shell")).expect("write shell");
    (dir, archive_path, shell_path)
}

fn write_unchecked_pair(
    shell: &ShellPackage,
    archive: &[u8],
) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let archive_path = dir.path().join("demo-1.2.3.tar.zst");
    let shell_path = dir.path().join("demo-1.2.3.chb");
    fs::write(&archive_path, archive).expect("write archive");
    let payload = bincode::serialize(shell).expect("encode unchecked payload");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(SHELL_MAGIC);
    bytes.extend_from_slice(&SHELL_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&payload);
    fs::write(&shell_path, bytes).expect("write unchecked shell");
    (dir, archive_path, shell_path)
}

#[test]
fn valid_artifact_pair_verifies_without_writing() {
    let archive = b"opaque archive bytes";
    let shell = fixture_shell(sha256(archive));
    let (dir, archive_path, shell_path) = write_pair(&shell, archive);
    let before = fs::read_dir(dir.path()).expect("read dir").count();

    let verified = verify_artifact_pair(&archive_path, &shell_path).expect("valid pair");

    assert_eq!(verified.package, shell.package);
    assert_eq!(verified.compiler, shell.compiler);
    assert_eq!(verified.archive_sha256, shell.archive_sha256);
    assert_eq!(
        verified.shell_sha256,
        sha256(&fs::read(shell_path).unwrap())
    );
    assert_eq!(fs::read_dir(dir.path()).expect("read dir").count(), before);
}

#[test]
fn artifact_pair_rejects_mismatched_archive_sha() {
    let shell = fixture_shell(sha256(b"expected archive"));
    let (_dir, archive_path, shell_path) = write_pair(&shell, b"different archive");

    let error = verify_artifact_pair(&archive_path, &shell_path).expect_err("mismatch");

    assert!(
        error.contains("archive_sha256"),
        "unexpected error: {error}"
    );
}

#[test]
fn artifact_pair_rejects_trailing_and_truncated_chb() {
    let archive = b"opaque archive bytes";
    let shell = fixture_shell(sha256(archive));
    let (_dir, archive_path, shell_path) = write_unchecked_pair(&shell, archive);
    let canonical = fs::read(&shell_path).expect("read shell");

    let mut trailing = canonical.clone();
    trailing.extend_from_slice(b"junk");
    fs::write(&shell_path, trailing).expect("write trailing shell");
    let error = verify_artifact_pair(&archive_path, &shell_path).expect_err("trailing bytes");
    assert!(error.contains("decode"), "unexpected error: {error}");

    fs::write(&shell_path, &canonical[..canonical.len() - 1]).expect("write truncated shell");
    let error = verify_artifact_pair(&archive_path, &shell_path).expect_err("truncated");
    assert!(error.contains("decode"), "unexpected error: {error}");
}

#[test]
fn artifact_pair_rejects_decodable_mutation_in_unconsumed_metadata() {
    let archive = b"opaque archive bytes";
    let shell = fixture_shell(sha256(archive));
    let (_dir, archive_path, shell_path) = write_pair(&shell, archive);
    let mut bytes = fs::read(&shell_path).expect("read shell");
    let pin = shell.compiler.as_bytes();
    let offset = bytes
        .windows(pin.len())
        .position(|window| window == pin)
        .expect("compiler pin bytes");
    bytes[offset] = b'?';
    fs::write(&shell_path, bytes).expect("write mutated shell");

    let error = verify_artifact_pair(&archive_path, &shell_path).expect_err("invalid compiler");

    assert!(error.contains("compiler"), "unexpected error: {error}");
}

#[test]
fn artifact_pair_rejects_noncanonical_metadata_order() {
    let archive = b"opaque archive bytes";
    let mut shell = fixture_shell(sha256(archive));
    shell.modules = vec![
        ShellModule {
            module: "Demo.Zed".to_string(),
            exports: Vec::new(),
        },
        ShellModule {
            module: "Demo.Alpha".to_string(),
            exports: Vec::new(),
        },
    ];
    let (_dir, archive_path, shell_path) = write_unchecked_pair(&shell, archive);

    let error = verify_artifact_pair(&archive_path, &shell_path).expect_err("module order");

    assert!(
        error.contains("strictly sorted"),
        "unexpected error: {error}"
    );
}

#[test]
fn installation_rejects_trailing_chb_before_registry_mutation() {
    let archive = b"opaque archive bytes";
    let shell = fixture_shell(sha256(archive));
    let (dir, archive_path, shell_path) = write_pair(&shell, archive);
    let mut bytes = fs::read(&shell_path).expect("read shell");
    bytes.extend_from_slice(b"junk");
    fs::write(&shell_path, bytes).expect("append junk");
    let registry = dir.path().join("registry");

    let error = install_validated_artifact_pair(
        &archive_path,
        &shell_path,
        "demo",
        "1.2.3",
        &registry,
        None,
    )
    .expect_err("install must reject trailing CHB bytes");

    assert!(error.contains("decode"), "unexpected error: {error}");
    assert!(
        !registry.join("index.json").exists(),
        "failed verification must not update the registry index"
    );
    assert!(
        !registry.exists(),
        "failed verification must not create registry state"
    );
}
