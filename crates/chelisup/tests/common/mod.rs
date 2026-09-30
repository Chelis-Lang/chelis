//! Shared helpers for the chelisup integration tests. Unix-only (the
//! shim `exec`s and tests set a custom `argv[0]`).
#![cfg(unix)]
#![allow(dead_code)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The built chelisup binary under test.
pub fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_chelisup")
}

/// A fake `chelis` toolchain binary: a python script that self-reports
/// the version from its own path (`.../toolchains/<ver>/bin/chelis`) and
/// echoes the forwarded args. Lets a test assert both which version ran
/// and that `+ver` was stripped from the args.
pub const FAKE_CHELIS: &str = "#!/usr/bin/env python3\n\
import sys, pathlib\n\
p = pathlib.Path(sys.argv[0]).resolve()\n\
ver = p.parent.parent.name\n\
sys.stdout.write('FAKE-CHELIS ' + ver + ' ' + ' '.join(sys.argv[1:]))\n";

/// True iff `python3` runs here (the fake toolchain needs it).
pub fn python3_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn set_exec(path: &Path) {
    let mut perms = fs::metadata(path).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms).unwrap();
}

/// Write a fake installed toolchain directly into the store (no install
/// flow). Used by the resolution tests.
pub fn write_fake_toolchain(home: &Path, version: &str) {
    let bin_dir = home.join("toolchains").join(version).join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    let chelis = bin_dir.join("chelis");
    fs::write(&chelis, FAKE_CHELIS).unwrap();
    set_exec(&chelis);
}

/// Record the default version in the store.
pub fn write_default(home: &Path, version: &str) {
    fs::create_dir_all(home).unwrap();
    fs::write(home.join("default"), format!("{version}\n")).unwrap();
}

/// Build a release-tarball fixture `chelis-v<ver>-<build>.tar.gz` under
/// `release_dir`, whose top-level `chelis-v<ver>-<build>/bin/chelis` is
/// the fake toolchain. This is what `CHELISUP_RELEASE_BASE` serves.
pub fn build_fixture_tarball(release_dir: &Path, version: &str, build: &str) -> PathBuf {
    let stage = tempfile::tempdir().unwrap();
    let root = stage.path().join(format!("chelis-v{version}-{build}"));
    let bin_dir = root.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    let chelis = bin_dir.join("chelis");
    fs::write(&chelis, FAKE_CHELIS).unwrap();
    set_exec(&chelis);
    write_tarball(release_dir, &root)
}

/// The runtime files a fixture release ships and what its fake
/// `chelis runtime export` independently writes and reports.
pub struct ReleaseRuntime {
    /// `lib/libchelis_runtime.a` in the tarball.
    pub shipped_archive: Vec<u8>,
    /// Omit the archive entirely from the release tarball when false.
    pub ship_archive: bool,
    /// Ship `lib/libchelis_runtime.a` as a symlink to a sibling holding
    /// `shipped_archive`.
    pub archive_is_symlink: bool,
    /// Ship `lib` as a symlink to a sibling `lib.real` directory.
    pub lib_is_symlink: bool,
    /// Ship the top-level `chelis-v*` entry as a link to an absolute path
    /// outside the tarball, where the release's files are.
    pub root_is_symlink: bool,
    /// `include/<name>` files in the tarball.
    pub shipped_headers: Vec<(String, Vec<u8>)>,
    /// The fake compiler's separately carried runtime archive. `None`
    /// makes `runtime export` omit the archive.
    pub exported_archive: Option<Vec<u8>>,
    /// Make the fake compiler export the archive as a symlink.
    pub exported_archive_is_symlink: bool,
    /// Header files independently written by the fake compiler's export.
    pub exported_headers: Vec<(String, Vec<u8>)>,
    /// The staging receipt the fake export writes.
    pub receipt: serde_json::Value,
    /// Omit the fake compiler's export receipt when false.
    pub export_receipt: bool,
    /// The fake export's exit status; a failed export writes nothing.
    pub export_status: i32,
    /// The interpreter the fake `chelis` names. A missing one makes the
    /// binary impossible to start, like a glibc build on a musl system.
    pub chelis_interpreter: &'static str,
}

impl ReleaseRuntime {
    /// A sealed release of `version` whose shipped runtime files are the
    /// ones its independent export reports.
    pub fn matching(version: &str) -> Self {
        let archive = b"carried runtime archive".to_vec();
        let headers = vec![
            ("chelis_runtime.h".to_owned(), b"/* runtime */\n".to_vec()),
            (
                "chelis_runtime_views.h".to_owned(),
                b"/* views */\n".to_vec(),
            ),
            (
                "chelis_runtime_dtype.h".to_owned(),
                b"/* dtype */\n".to_vec(),
            ),
            ("chelis_blas.h".to_owned(), b"/* blas */\n".to_vec()),
            ("chelis_simd.h".to_owned(), b"/* simd */\n".to_vec()),
            ("chelis_math.h".to_owned(), b"/* math */\n".to_vec()),
        ];
        let digests: serde_json::Map<String, serde_json::Value> = headers
            .iter()
            .map(|(name, bytes)| (name.clone(), sha256(bytes).into()))
            .collect();
        let receipt = serde_json::json!({
            "schema": "chelis-runtime-staging/1",
            "archive": "libchelis_runtime.a",
            "archive_sha256": sha256(&archive),
            "headers": digests,
            "mode": "sealed",
            "chelis_version": version,
        });
        Self {
            shipped_archive: archive.clone(),
            ship_archive: true,
            archive_is_symlink: false,
            lib_is_symlink: false,
            root_is_symlink: false,
            shipped_headers: headers.clone(),
            exported_archive: Some(archive),
            exported_archive_is_symlink: false,
            exported_headers: headers,
            receipt,
            export_receipt: true,
            export_status: 0,
            chelis_interpreter: "/bin/sh",
        }
    }
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Build a release tarball that ships `runtime`'s files, with a fake
/// `chelis` whose `runtime export <dir>` copies independent runtime payloads
/// and writes `runtime.receipt` into `<dir>` (and, like the real export,
/// refuses a set `CHELIS_RUNTIME_DIR`). Other invocations echo their args.
pub fn build_release_tarball(
    release_dir: &Path,
    version: &str,
    build: &str,
    runtime: &ReleaseRuntime,
) -> PathBuf {
    let stage = tempfile::tempdir().unwrap();
    let root = stage.path().join(format!("chelis-v{version}-{build}"));
    for directory in ["bin", "lib", "include"] {
        fs::create_dir_all(root.join(directory)).unwrap();
    }
    let chelis = root.join("bin/chelis");
    fs::write(
        &chelis,
        format!(
            "#!{interpreter}\n\
             if [ \"$1\" = runtime ] && [ \"$2\" = export ]; then\n\
             \x20 if [ -n \"${{CHELIS_RUNTIME_DIR+set}}\" ]; then\n\
             \x20   echo 'error: CHELIS_RUNTIME_DIR is set' >&2; exit 1\n\
             \x20 fi\n\
             \x20 if [ {status} -ne 0 ]; then echo 'error: export refused' >&2; exit {status}; fi\n\
             \x20 mkdir -p \"$3\" || exit 1\n\
             \x20 cp -R -P \"$(dirname \"$0\")/.runtime-export/.\" \"$3/\" || exit 1\n\
             \x20 if [ {export_receipt} = true ]; then\n\
             \x20 cat > \"$3/chelis_runtime.receipt.json\" <<'RECEIPT'\n\
             {receipt}\n\
             RECEIPT\n\
             \x20 fi\n\
             \x20 exit 0\n\
             fi\n\
             printf 'FAKE-CHELIS %s' \"$*\"\n",
            interpreter = runtime.chelis_interpreter,
            status = runtime.export_status,
            receipt = runtime.receipt,
            export_receipt = runtime.export_receipt,
        ),
    )
    .unwrap();
    set_exec(&chelis);
    let runtime_export = root.join("bin/.runtime-export");
    fs::create_dir_all(&runtime_export).unwrap();
    for (name, bytes) in &runtime.exported_headers {
        fs::write(runtime_export.join(name), bytes).unwrap();
    }
    let exported_archive = runtime_export.join("libchelis_runtime.a");
    if runtime.exported_archive_is_symlink {
        std::os::unix::fs::symlink("chelis_runtime.h", exported_archive).unwrap();
    } else if let Some(archive) = &runtime.exported_archive {
        fs::write(exported_archive, archive).unwrap();
    }
    let archive = root.join("lib/libchelis_runtime.a");
    if runtime.archive_is_symlink {
        fs::write(
            root.join("lib/libchelis_runtime.real"),
            &runtime.shipped_archive,
        )
        .unwrap();
        std::os::unix::fs::symlink("libchelis_runtime.real", &archive).unwrap();
    } else if runtime.ship_archive {
        fs::write(&archive, &runtime.shipped_archive).unwrap();
    }
    for (name, bytes) in &runtime.shipped_headers {
        fs::write(root.join("include").join(name), bytes).unwrap();
    }
    if runtime.lib_is_symlink {
        fs::rename(root.join("lib"), root.join("lib.real")).unwrap();
        std::os::unix::fs::symlink("lib.real", root.join("lib")).unwrap();
    }
    if runtime.root_is_symlink {
        let top = root.file_name().unwrap().to_str().unwrap().to_owned();
        let outside = release_dir.join(format!("outside-{top}"));
        fs::create_dir_all(release_dir).unwrap();
        fs::rename(&root, &outside).unwrap();
        return write_link_tarball(release_dir, &top, &outside);
    }
    write_tarball(release_dir, &root)
}

/// Write `root` (a staged `chelis-v*` directory) as `<release_dir>/<name>.tar.gz`,
/// keeping real file modes and symlinks.
fn write_tarball(release_dir: &Path, root: &Path) -> PathBuf {
    use flate2::Compression;
    use flate2::write::GzEncoder;

    fs::create_dir_all(release_dir).unwrap();
    let top = root.file_name().unwrap().to_str().unwrap().to_owned();
    let asset = release_dir.join(format!("{top}.tar.gz"));
    let file = fs::File::create(&asset).unwrap();
    let enc = GzEncoder::new(file, Compression::fast());
    let mut builder = tar::Builder::new(enc);
    builder.follow_symlinks(false);
    builder.append_dir_all(&top, root).unwrap();
    let enc = builder.into_inner().unwrap();
    enc.finish().unwrap();
    asset
}

/// Write `<release_dir>/<top>.tar.gz` holding only the link `top -> target`.
fn write_link_tarball(release_dir: &Path, top: &str, target: &Path) -> PathBuf {
    use flate2::Compression;
    use flate2::write::GzEncoder;

    let asset = release_dir.join(format!("{top}.tar.gz"));
    let enc = GzEncoder::new(fs::File::create(&asset).unwrap(), Compression::fast());
    let mut builder = tar::Builder::new(enc);
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_size(0);
    builder.append_link(&mut header, top, target).unwrap();
    builder.into_inner().unwrap().finish().unwrap();
    asset
}

/// Spawn the binary as the `chelis` shim (argv[0] = "chelis"), with an
/// isolated `CHELIS_HOME`, the given cwd, and `CHELIS_TOOLCHAIN`
/// explicitly removed unless a test re-adds it via `extra_env`.
pub fn run_shim(
    home: &Path,
    cwd: &Path,
    args: &[&str],
    extra_env: &[(&str, &str)],
) -> std::process::Output {
    use std::os::unix::process::CommandExt;
    let mut cmd = Command::new(bin());
    cmd.arg0("chelis")
        .args(args)
        .current_dir(cwd)
        .env("CHELIS_HOME", home)
        .env_remove("CHELIS_TOOLCHAIN");
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.output().unwrap()
}

/// Spawn the binary as `chelisup` (normal argv[0]) with an isolated
/// `CHELIS_HOME` and the given cwd.
pub fn run_cli(
    home: &Path,
    cwd: &Path,
    args: &[&str],
    extra_env: &[(&str, &str)],
) -> std::process::Output {
    let mut cmd = Command::new(bin());
    cmd.args(args)
        .current_dir(cwd)
        .env("CHELIS_HOME", home)
        .env_remove("CHELIS_TOOLCHAIN");
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.output().unwrap()
}
