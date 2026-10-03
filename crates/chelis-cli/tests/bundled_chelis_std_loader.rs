//! Phase A bundled-chelis-std loader property oracle.
//!
//! This file owns the named test
//! `phaseA_bundled_chelis_std_loader_property_oracle`, which is the
//! single comprehensive oracle for the architectural correction that
//! makes chelis-std's bytes reachable from the chelis binary itself
//! (packed from `packages/chelis-std` by `chelis-std-bundle` while the
//! binary builds) and synthesized into every project's lockfile via
//! `compiler =` pin.
//!
//! Sub-cases (asserted in one test so the oracle is one entrypoint):
//!
//! 1. **Implicit-runtime project builds.** A reef project whose
//!    `[dependencies]` does NOT list chelis-std builds successfully
//!    against an empty `$CHELIS_REEF_HOME`. The lockfile carries a
//!    `chelis-std` entry with `LockSource::Bundled`.
//! 2. **Explicit-runtime project builds.** Same shape, but the
//!    project lists `chelis-std = { version = "<bundled version>" }` explicitly.
//!    The lockfile is structurally identical to the implicit case
//!    (synthesis is idempotent).
//! 3. **Wipe-and-rebuild.** After the first build, the test wipes
//!    `$CHELIS_REEF_HOME/packages/chelis-std/` (simulating "registry
//!    state has no chelis-std at all"). A second `chelis reef build`
//!    must succeed because the bundled bytes are reachable from the
//!    chelis binary, not from the registry.
//! 4. **No registry at all.** A pristine `$CHELIS_REEF_HOME` with no
//!    `index.json` and no `packages/` dir builds successfully. This
//!    is the "fresh dev clone with `chelis reef build` works" case
//!    the bundle is designed to support.
//!
//! Negative parity: `phaseA_item1_negative_parity` (separate test,
//! same file) asserts that a project explicitly listing chelis-std at
//! a non-bundled version fails with the soft-verify error and does
//! NOT silently get papered over by synthesis.
//!
//! No `#[ignore]` here. The test must be green by default. If it
//! cannot be, the bundle/loader/synthesis design is under-implemented.

#![allow(non_snake_case)]

use assert_cmd::Command;
use chelis_reef::{LockSource, ReefLock};
use chelis_std_bundle::{
    BUNDLED_CHELIS_STD_VERSION, CHELIS_STD_ARCHIVE, CHELIS_STD_SHELL, EMBEDDED_RUNTIME,
};
use predicates::prelude::*;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;
use walkdir::WalkDir;

fn std_package_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/chelis-std")
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Copy the runtime inputs of `packages/chelis-std` (the files the embedded
/// runtime is packed from) to `dest`.
fn copy_std_sources(dest: &Path) -> PathBuf {
    let source = std_package_root();
    for relative in chelis_std_bundle::stage::runtime_inputs(&source).expect("runtime inputs") {
        let target = dest.join(&relative);
        fs::create_dir_all(target.parent().expect("input has a parent")).expect("mkdir");
        fs::copy(source.join(&relative), &target).expect("copy runtime input");
    }
    dest.to_path_buf()
}

/// `chelis reef build` of the package at `root`, offline, with the default
/// archive mtime.
fn reef_build(root: &Path, reef_home: &Path) {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .env_remove("SOURCE_DATE_EPOCH")
        .args(["reef", "build", "--no-auto-fetch", root.to_str().unwrap()])
        .assert()
        .success();
}

fn built_pair(root: &Path) -> (Vec<u8>, Vec<u8>) {
    let dist = root.join("dist");
    let stem = format!("chelis-std-{BUNDLED_CHELIS_STD_VERSION}");
    (
        fs::read(dist.join(format!("{stem}.tar.zst"))).expect("read built archive"),
        fs::read(dist.join(format!("{stem}.chb"))).expect("read built shell"),
    )
}

/// The `(archive_sha256, shell_sha256)` the lock at `root` records for the
/// bundled runtime.
fn locked_runtime_hashes(root: &Path) -> (String, String) {
    let lock = read_lockfile(root);
    assert_chelis_std_bundled_entry(&lock);
    let entry = lock
        .dependencies
        .iter()
        .find(|dependency| dependency.name == "chelis-std")
        .expect("the lock records the runtime");
    (entry.archive_sha256.clone(), entry.shell_sha256.clone())
}

/// `reef.toml` and every `.ch` file below `src/` of the package at `root`,
/// found by walking the tree rather than by the build script's own selection.
fn manifest_and_ch_sources(root: &Path) -> Vec<PathBuf> {
    let mut found = vec![PathBuf::from("reef.toml")];
    for entry in WalkDir::new(root.join("src")) {
        let entry = entry.expect("walk the std sources");
        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "ch")
        {
            found.push(
                entry
                    .path()
                    .strip_prefix(root)
                    .expect("a source lies below its package")
                    .to_path_buf(),
            );
        }
    }
    found.sort();
    found
}

/// The embedded archive holds exactly the manifest and the `.ch` sources of
/// `packages/chelis-std`, byte for byte, so a binary never carries a runtime
/// built from other sources.
#[test]
fn bundled_chelis_std_sources_match_the_checked_in_runtime_package() {
    let package_root = std_package_root();
    let inputs = manifest_and_ch_sources(&package_root);
    let embedded = EMBEDDED_RUNTIME
        .archive_files()
        .expect("read the embedded archive");
    assert_eq!(
        embedded.keys().cloned().collect::<Vec<_>>(),
        inputs,
        "embedded chelis-std file inventory differs from packages/chelis-std"
    );
    for relative in &inputs {
        let checked_in = fs::read(package_root.join(relative)).expect("read runtime input");
        assert!(
            embedded.get(relative) == Some(&checked_in),
            "embedded chelis-std source drift at {}; the build of chelis-std-bundle \
             packs packages/chelis-std, so this binary was built from other sources",
            relative.display()
        );
    }
}

/// The fixed point between the compiler and the runtime it embeds:
/// `chelis reef build` of a copy holding only the runtime inputs, with
/// `SOURCE_DATE_EPOCH` unset, writes exactly the embedded archive and shell,
/// and the lock that build writes, like the lock
/// of any package built against this runtime, names the hashes of those
/// freshly built bytes.
#[test]
fn building_the_std_sources_reproduces_the_embedded_runtime() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_copy = copy_std_sources(&dir.path().join("chelis-std"));
    reef_build(&std_copy, &reef_home);
    let (archive, shell) = built_pair(&std_copy);
    assert!(
        archive == CHELIS_STD_ARCHIVE,
        "chelis reef build of the std sources wrote an archive other than the embedded one"
    );
    assert!(
        shell == CHELIS_STD_SHELL,
        "chelis reef build of the std sources wrote a shell other than the embedded one"
    );
    let built_hashes = (sha256_hex(&archive), sha256_hex(&shell));
    assert_eq!(
        locked_runtime_hashes(&std_copy),
        built_hashes,
        "the runtime's own lock must name the hashes of the runtime it builds"
    );

    let downstream = dir.path().join("downstream");
    write_implicit_runtime_project(&downstream, "Downstream");
    reef_build(&downstream, &reef_home);
    assert_eq!(
        locked_runtime_hashes(&downstream),
        built_hashes,
        "a package's lock must name the hashes of the runtime built from the std sources"
    );
}

/// Negative parity for the fixed point: an edited std copy builds a different
/// pair, while every lock this binary writes keeps naming the runtime the
/// binary embeds, so the comparison above can fail.
#[test]
fn an_edited_std_copy_does_not_reproduce_the_embedded_runtime() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_copy = copy_std_sources(&dir.path().join("chelis-std"));
    let edited = std_copy.join("src/text.ch");
    let mut source = fs::read(&edited).expect("read std source");
    source.extend_from_slice(b"\n-- an edit\n");
    fs::write(&edited, source).expect("edit std source");
    reef_build(&std_copy, &reef_home);
    let (archive, shell) = built_pair(&std_copy);
    assert!(
        archive != CHELIS_STD_ARCHIVE,
        "the edit must change the archive"
    );
    assert!(shell != CHELIS_STD_SHELL, "the edit must change the shell");
    let locked = locked_runtime_hashes(&std_copy);
    assert_ne!(locked, (sha256_hex(&archive), sha256_hex(&shell)));
    assert_eq!(
        locked,
        (
            EMBEDDED_RUNTIME.archive_sha256().to_string(),
            EMBEDDED_RUNTIME.shell_sha256().to_string()
        )
    );
}

fn write_implicit_runtime_project(root: &Path, module_prefix: &str) {
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    let compiler = format!("={}", env!("CARGO_PKG_VERSION"));
    fs::write(
        root.join("reef.toml"),
        format!(
            r#"[package]
name = "downstream"
version = "0.4.0"
compiler = "{compiler}"
module_prefix = "{module_prefix}"

[dependencies]
"#
        ),
    )
    .expect("write reef.toml");
    fs::write(
        root.join("src/main.ch"),
        format!("module {module_prefix}.Main\n\ndef id(x: i32) -> i32 = x\n"),
    )
    .expect("write main.ch");
}

fn write_explicit_runtime_project(root: &Path, module_prefix: &str, std_version: &str) {
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    let compiler = format!("={}", env!("CARGO_PKG_VERSION"));
    fs::write(
        root.join("reef.toml"),
        format!(
            r#"[package]
name = "downstream"
version = "0.4.0"
compiler = "{compiler}"
module_prefix = "{module_prefix}"

[dependencies]
chelis-std = {{ version = "{std_version}" }}
"#
        ),
    )
    .expect("write reef.toml");
    fs::write(
        root.join("src/main.ch"),
        format!("module {module_prefix}.Main\n\ndef id(x: i32) -> i32 = x\n"),
    )
    .expect("write main.ch");
}

fn read_lockfile(pkg_root: &Path) -> ReefLock {
    let text = fs::read_to_string(pkg_root.join("reef.lock")).expect("read reef.lock");
    toml::from_str(&text).expect("parse reef.lock")
}

fn assert_chelis_std_bundled_entry(lock: &ReefLock) {
    let entry = lock
        .dependencies
        .iter()
        .find(|d| d.name == "chelis-std")
        .unwrap_or_else(|| {
            panic!(
                "lockfile must include chelis-std; got entries: {:?}",
                lock.dependencies
                    .iter()
                    .map(|d| (&d.name, &d.version))
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!(
        entry.version, BUNDLED_CHELIS_STD_VERSION,
        "bundled chelis-std version"
    );
    match &entry.source {
        LockSource::Bundled { compiler_version } => {
            // The compiler_version field records the version of the
            // compiler that supplied the bundled bytes — by
            // construction, the running test binary's
            // CARGO_PKG_VERSION.
            assert_eq!(compiler_version, env!("CARGO_PKG_VERSION"));
        }
        other => panic!("chelis-std must use LockSource::Bundled, got {other:?}"),
    }
    assert!(
        !entry.archive_sha256.is_empty(),
        "Bundled chelis-std must carry archive_sha256 from embedded bytes"
    );
    assert!(
        !entry.shell_sha256.is_empty(),
        "Bundled chelis-std must carry shell_sha256 from embedded bytes"
    );
}

#[test]
fn phaseA_bundled_chelis_std_loader_property_oracle() {
    // Sub-case 1: implicit-runtime project + empty CHELIS_REEF_HOME.
    {
        let dir = tempdir().expect("tempdir");
        let pkg_root = dir.path().join("implicit-pkg");
        let reef_home = dir.path().join("reef-home");
        write_implicit_runtime_project(&pkg_root, "Implicit");

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();
        let lock = read_lockfile(&pkg_root);
        assert_chelis_std_bundled_entry(&lock);

        // Sub-case 4: a pristine reef_home (no index.json, no
        // packages/) suffices because the bundled bytes are reachable
        // from the chelis binary itself. We've just demonstrated that
        // — pin it explicitly so a regression that re-introduced a
        // registry-write side effect surfaces here.
        // Note: chelis reef build creates index.json on first run for
        // the registry it touches, but it should NOT create a
        // chelis-std entry under packages/.
        let chelis_std_pkg_dir = reef_home.join("packages").join("chelis-std");
        assert!(
            !chelis_std_pkg_dir.exists(),
            "build must not write chelis-std into the local registry; \
             expected absent: {chelis_std_pkg_dir:?}"
        );
    }

    // Sub-case 2: explicit-runtime project produces structurally
    // identical lockfile.
    {
        let dir = tempdir().expect("tempdir");
        let pkg_root = dir.path().join("explicit-pkg");
        let reef_home = dir.path().join("reef-home");
        write_explicit_runtime_project(&pkg_root, "Explicit", BUNDLED_CHELIS_STD_VERSION);

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();
        let lock = read_lockfile(&pkg_root);
        assert_chelis_std_bundled_entry(&lock);
        // Explicit and implicit must produce identical chelis-std
        // entries: synthesis is a no-op when the closure already
        // produced the entry.
        let entry = lock
            .dependencies
            .iter()
            .find(|d| d.name == "chelis-std")
            .expect("chelis-std entry");
        // archive/shell sha256s come from the embedded bundle in
        // both paths so they MUST agree.
        assert_eq!(
            entry.archive_sha256,
            EMBEDDED_RUNTIME.archive_sha256(),
            "explicit-listing path must use bundled archive_sha256"
        );
        assert_eq!(
            entry.shell_sha256,
            EMBEDDED_RUNTIME.shell_sha256(),
            "explicit-listing path must use bundled shell_sha256"
        );
    }

    // Sub-case 3: wipe-then-rebuild. Build once; wipe everything in
    // CHELIS_REEF_HOME (including index.json); rebuild. Must
    // succeed because the bundled bytes don't depend on registry
    // state.
    {
        let dir = tempdir().expect("tempdir");
        let pkg_root = dir.path().join("wipe-pkg");
        let reef_home = dir.path().join("reef-home");
        write_implicit_runtime_project(&pkg_root, "Wipe");

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();

        // Wipe the entire reef home (registry).
        if reef_home.exists() {
            fs::remove_dir_all(&reef_home).expect("wipe reef home");
        }

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();

        // Lockfile is regenerated by the second build; it must
        // again carry the Bundled chelis-std entry.
        let lock = read_lockfile(&pkg_root);
        assert_chelis_std_bundled_entry(&lock);
    }

    // Sub-case 3b: wipe ONLY chelis-std out of the registry, keeping
    // index.json. This exercises the "MissingFromIndex" branch of
    // load_registry_package — but the bundle's special-case at the
    // top of that function returns before consulting index.json, so
    // the build still succeeds.
    {
        let dir = tempdir().expect("tempdir");
        let pkg_root = dir.path().join("wipe-std-only");
        let reef_home = dir.path().join("reef-home");
        write_explicit_runtime_project(&pkg_root, "WipeStdOnly", BUNDLED_CHELIS_STD_VERSION);

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();

        // Wipe only the chelis-std pkg dir (if it was ever written).
        let std_pkg_dir = reef_home.join("packages").join("chelis-std");
        if std_pkg_dir.exists() {
            fs::remove_dir_all(&std_pkg_dir).expect("wipe chelis-std pkg");
        }

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();
        let lock = read_lockfile(&pkg_root);
        assert_chelis_std_bundled_entry(&lock);
    }
}

/// Negative parity for Item 1: a project explicitly listing
/// chelis-std at a non-bundled version must error with the
/// soft-verify message, and the lockfile must NOT be silently
/// rewritten to the bundled version.
#[test]
fn phaseA_item1_negative_parity_explicit_version_mismatch() {
    let dir = tempdir().expect("tempdir");
    let pkg_root = dir.path().join("mismatch-pkg");
    let reef_home = dir.path().join("reef-home");
    write_explicit_runtime_project(&pkg_root, "Mismatch", "9.9.9");

    let assert_out = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "build", pkg_root.to_str().unwrap()])
        .assert()
        .failure();

    let stderr = String::from_utf8_lossy(&assert_out.get_output().stderr).to_string();
    assert!(
        stderr.contains("chelis-std")
            && stderr.contains("9.9.9")
            && stderr.contains(BUNDLED_CHELIS_STD_VERSION),
        "soft-verify error must name both versions; got stderr:\n{stderr}"
    );

    // The lockfile must NOT have been written silently behind a
    // mismatch error.
    assert!(
        !pkg_root.join("reef.lock").exists(),
        "soft-verify mismatch must not write reef.lock; the synthesis \
         path must not paper over the explicit declaration"
    );
}

/// The lock at `root` with the `kind` hash of `package`'s entry replaced by
/// zeros, written back; returns the edited text.
fn poison_locked_hash(root: &Path, package: &str, kind: &str) -> String {
    let lock_path = root.join("reef.lock");
    let text = fs::read_to_string(&lock_path).expect("read reef.lock");
    let entry = read_lockfile(root)
        .dependencies
        .into_iter()
        .find(|dependency| dependency.name == package)
        .unwrap_or_else(|| panic!("the lock records {package}"));
    let hash = match kind {
        "archive" => entry.archive_sha256,
        _ => entry.shell_sha256,
    };
    let field = format!("{kind}_sha256");
    let poisoned = text.replacen(
        &format!("{field} = \"{hash}\""),
        &format!("{field} = \"{}\"", "0".repeat(64)),
        1,
    );
    assert_ne!(poisoned, text, "the {package} {kind} hash must be replaced");
    fs::write(&lock_path, &poisoned).expect("write reef.lock");
    poisoned
}

fn chelis_in(root: &Path, reef_home: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command
        .current_dir(root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .env_remove("SOURCE_DATE_EPOCH");
    command
}

/// A lock written before a std edit names other runtime hashes under the same
/// versions. `chelis check` resolves the runtime again in memory and leaves
/// the lock alone; `chelis reef build` resolves it again and writes the lock
/// it would have written.
#[test]
fn check_and_build_resolve_a_lock_with_old_runtime_hashes_again() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let app = dir.path().join("app");
    write_implicit_runtime_project(&app, "Downstream");
    reef_build(&app, &reef_home);
    let current = fs::read_to_string(app.join("reef.lock")).expect("read reef.lock");
    for kind in ["archive", "shell"] {
        let poisoned = poison_locked_hash(&app, "chelis-std", kind);
        chelis_in(&app, &reef_home)
            .args(["check", "src/main.ch"])
            .assert()
            .success()
            .stderr(predicate::str::contains(format!(
                "{kind} hash differs from the compiler-bundled runtime"
            )));
        assert_eq!(
            fs::read_to_string(app.join("reef.lock")).expect("read reef.lock"),
            poisoned,
            "chelis check never writes the lock"
        );
        reef_build(&app, &reef_home);
        assert_eq!(
            fs::read_to_string(app.join("reef.lock")).expect("read reef.lock"),
            current,
            "chelis reef build rewrites the lock to name the embedded runtime"
        );
    }
}

/// Negative control: a locked registry package whose recorded hash differs
/// from its bytes is an integrity failure for both commands, and the lock is
/// kept. Only the runtime entry, which names the compiler's own bytes, goes
/// stale on a hash mismatch.
#[test]
fn a_registry_package_hash_mismatch_still_fails_hard() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let compiler = format!("={}", env!("CARGO_PKG_VERSION"));
    let helper = dir.path().join("monorepo/packages/helper");
    fs::create_dir_all(helper.join("src")).expect("mkdir helper/src");
    fs::write(
        helper.join("reef.toml"),
        format!(
            "[package]\nname = \"helper\"\nversion = \"0.1.0\"\ncompiler = \"{compiler}\"\nmodule_prefix = \"Helper\"\n\n[dependencies]\n"
        ),
    )
    .expect("write helper reef.toml");
    fs::write(
        helper.join("src/core.ch"),
        "module Helper.Core\nexport (twice)\n\ndef twice(x: i32) -> i32 = x\n",
    )
    .expect("write helper source");
    reef_build(&helper, &reef_home);
    chelis_in(dir.path(), &reef_home)
        .args([
            "reef",
            "install",
            "--from-monorepo",
            dir.path().join("monorepo").to_str().unwrap(),
            "helper=0.1.0",
        ])
        .assert()
        .success();

    let app = dir.path().join("app");
    fs::create_dir_all(app.join("src")).expect("mkdir app/src");
    fs::write(
        app.join("reef.toml"),
        format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\ncompiler = \"{compiler}\"\nmodule_prefix = \"App\"\n\n[dependencies]\nhelper = {{ version = \"0.1.0\" }}\n"
        ),
    )
    .expect("write app reef.toml");
    fs::write(
        app.join("src/main.ch"),
        "module App.Main\nimport Helper.Core (twice)\n\ndef four() -> i32 = twice(4)\n",
    )
    .expect("write app source");
    reef_build(&app, &reef_home);
    let current = fs::read_to_string(app.join("reef.lock")).expect("read reef.lock");

    for kind in ["archive", "shell"] {
        let poisoned = poison_locked_hash(&app, "helper", kind);
        let mismatch = format!("locked {kind} hash mismatch for `helper`");
        chelis_in(&app, &reef_home)
            .args(["check", "src/main.ch"])
            .assert()
            .failure()
            .stdout(predicate::str::contains(&mismatch));
        chelis_in(&app, &reef_home)
            .args(["reef", "build", "--no-auto-fetch", "."])
            .assert()
            .failure()
            .stderr(predicate::str::contains(&mismatch));
        assert_eq!(
            fs::read_to_string(app.join("reef.lock")).expect("read reef.lock"),
            poisoned,
            "an integrity failure keeps the lock"
        );
        fs::write(app.join("reef.lock"), &current).expect("restore reef.lock");
    }
}
