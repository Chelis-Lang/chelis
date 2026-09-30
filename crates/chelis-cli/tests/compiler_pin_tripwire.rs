//! Tripwire test for the `package.compiler` pin in real (non-test-fixture)
//! `.toml` files that ship in this repo.
//!
//! Background: `chelis_compiler_api::COMPILER_VERSION` makes the compiler
//! pin in test fixtures auto-sync with `workspace.package.version` because
//! every fixture is built from `env!("CARGO_PKG_VERSION")` at compile time.
//! Real `.toml` files on disk cannot do that — they have a literal pin
//! string. That means a workspace version bump silently leaves these
//! files behind unless someone remembers to update them.
//!
//! In v0.3.1, that "silently leaves" bit was the failure mode: the
//! `chelis-std` package's `reef.toml` kept the v0.3.0 pin, every
//! `chelis reef install --from-monorepo` test ran the v0.3.1 binary
//! against the old pin, and the chelis-std-importing tests all failed
//! with a confusing `compiler = "=0.3.0"` mismatch error.
//!
//! This file pins the invariant: when the workspace version changes,
//! all real `.toml` files must change in lockstep, and the failure
//! must point the operator at `scripts/bump_compiler_pins.py` so the
//! release flow stays a one-liner.

use std::fs;
use std::path::{Path, PathBuf};

/// Repository root resolved from `crates/chelis-cli`'s `CARGO_MANIFEST_DIR`.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root must exist")
}

/// Real `.toml` files whose `package.compiler` pin must equal
/// `={CARGO_PKG_VERSION}`. Add new entries here when shipping new
/// pinned-compiler real packages; do NOT add test-fixture files —
/// those are auto-synced via `chelis_compiler_api::COMPILER_VERSION`.
fn pinned_real_toml_files() -> Vec<PathBuf> {
    let root = repo_root();
    vec![
        root.join("packages/chelis-std/reef.toml"),
        root.join("crates/chelis-cli/tests/fixtures/pseudo_nautilus/reef.toml"),
        root.join("crates/chelis-cli/tests/fixtures/release_pipe_stage/reef.toml"),
        root.join("examples/illustrative/io_pipeline/reef.toml"),
        root.join("examples/nautilus_quantile_contract/reef.toml"),
        root.join("examples/nautilus_quantile_contract/fixtures/nautilus/reef.toml"),
    ]
}

/// Real `reef.lock` files that record a `chelis-std` bundled dependency.
/// The dependency's `compiler =` pin is synthesized from the project's own
/// `package.compiler` (see `build_lockfile` in `chelis-reef`), so it must
/// equal `={CARGO_PKG_VERSION}` exactly like the `.toml` pins do. Unlike
/// the `.toml` files, these locks are NOT auto-synced from
/// `chelis_compiler_api::COMPILER_VERSION` — they are literal on-disk
/// strings, so a version bump silently leaves them at the prior pin. That
/// is the 0.9.0 release failure: the `.toml` pins were bumped but
/// `release_pipe_stage/reef.lock` still pinned `=0.8.0`, and the
/// release-only fixture step (`chelis test tests`) rejected it. Add new
/// entries here when shipping new committed lockfiles with a chelis-std
/// dependency.
fn pinned_real_lock_files() -> Vec<PathBuf> {
    let root = repo_root();
    vec![
        root.join("packages/chelis-std/reef.lock"),
        root.join("crates/chelis-cli/tests/fixtures/release_pipe_stage/reef.lock"),
        root.join("examples/nautilus_quantile_contract/reef.lock"),
        root.join("examples/nautilus_quantile_contract/fixtures/nautilus/reef.lock"),
    ]
}

/// Extract the value on the right-hand side of a `compiler = "..."` line
/// in a `.toml` file. Returns `Some(pin)` (e.g. `Some("=0.3.1")`) or
/// `None` if the file has no `compiler =` key. The minimal hand-rolled
/// parser intentionally avoids pulling `toml` as a dev-dep just for one
/// tripwire test; the line shape is stable across our fixtures.
fn read_compiler_pin(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("compiler") {
            // Match `compiler = "..."` or `compiler="..."` (whitespace tolerant).
            let rest = rest.trim_start();
            let rest = rest.strip_prefix('=')?.trim_start();
            let rest = rest.strip_prefix('"')?;
            let end = rest.find('"')?;
            return Some(rest[..end].to_string());
        }
    }
    None
}

/// Collect every `compiler = "..."` pin in a `reef.lock`. A lockfile can
/// carry several `[[dependencies]]` entries, each with its own pin; the
/// `.toml` parser above stops at the first hit, which is wrong here. Used
/// to assert that every recorded dependency pin matches the workspace
/// version, since a stale `chelis-std` pin is what the release fixture
/// step rejects.
fn read_lock_compiler_pins(path: &Path) -> Vec<String> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut pins = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("compiler") {
            let rest = rest.trim_start();
            let Some(rest) = rest.strip_prefix('=') else {
                continue;
            };
            let rest = rest.trim_start();
            let Some(rest) = rest.strip_prefix('"') else {
                continue;
            };
            if let Some(end) = rest.find('"') {
                pins.push(rest[..end].to_string());
            }
        }
    }
    pins
}

#[test]
fn real_toml_compiler_pins_match_workspace_version() {
    let workspace_version = env!("CARGO_PKG_VERSION");
    let expected = format!("={workspace_version}");

    let files = pinned_real_toml_files();
    let mut stale: Vec<(PathBuf, String)> = Vec::new();
    for path in &files {
        let pin =
            read_compiler_pin(path).unwrap_or_else(|| panic!("no `compiler =` line in {path:?}"));
        if pin != expected {
            stale.push((path.clone(), pin));
        }
    }

    if !stale.is_empty() {
        let mut msg = String::new();
        msg.push_str(&format!(
            "Compiler-pin tripwire fired: workspace version is {workspace_version} \
             (expected pin string {expected:?}), but {n} real .toml file(s) \
             still pin a different version:\n",
            n = stale.len()
        ));
        for (path, pin) in &stale {
            msg.push_str(&format!("  {} pins {pin:?}\n", path.display()));
        }
        msg.push_str(
            "\n\
             Fix: run `python3 scripts/bump_compiler_pins.py <new-version>` \
             from the repo root. That script updates Cargo.toml, all real \
             `.toml` files listed above, and rebuilds packages/chelis-std/dist/. \
             It is the canonical release-bump entry point. The test \
             auto-sync only covers test fixtures, not these on-disk files.\n",
        );
        panic!("{msg}");
    }
}

#[test]
fn real_lock_compiler_pins_match_workspace_version() {
    let workspace_version = env!("CARGO_PKG_VERSION");
    let expected = format!("={workspace_version}");

    let files = pinned_real_lock_files();
    let mut stale: Vec<(PathBuf, String)> = Vec::new();
    for path in &files {
        let pins = read_lock_compiler_pins(path);
        assert!(
            !pins.is_empty(),
            "no dependency `compiler =` line in {path:?}; the lockfile parser \
             expected at least one `[[dependencies]]` pin to validate"
        );
        for pin in pins {
            if pin != expected {
                stale.push((path.clone(), pin));
            }
        }
    }

    if !stale.is_empty() {
        let mut msg = String::new();
        msg.push_str(&format!(
            "Lockfile compiler-pin tripwire fired: workspace version is \
             {workspace_version} (expected pin string {expected:?}), but \
             {n} committed reef.lock dependency pin(s) still record a \
             different version:\n",
            n = stale.len()
        ));
        for (path, pin) in &stale {
            msg.push_str(&format!("  {} pins {pin:?}\n", path.display()));
        }
        msg.push_str(
            "\n\
             These locks synthesize the `chelis-std` bundled-dependency pin \
             from the project's own `package.compiler` (build_lockfile), so a \
             stale pin makes `chelis test` reject the package with \
             \"package.compiler must be `=X.Y.Z`\". This is the failure that \
             broke the 0.9.0 release fixture step.\n\
             Fix: run `python3 scripts/bump_compiler_pins.py <new-version>` \
             from the repo root. It rebuilds the chelis-std bundle and \
             regenerates these locks in lockstep with the workspace version.\n",
        );
        panic!("{msg}");
    }
}
