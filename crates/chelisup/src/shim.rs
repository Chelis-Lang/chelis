//! The `chelis` shim: resolve the active toolchain at this invocation
//! and hand off to the real compiler with the original arguments.
//!
//! This runs on every `chelis` call, so it does no network work and does
//! not parse through clap. It resolves a version (see [`crate::resolve`]),
//! checks that the version is actually installed, and `exec`s the
//! toolchain binary. A resolved-but-not-installed version is a loud,
//! actionable error naming `chelisup install <ver>` -- never a silent
//! fall-through to a different version, and never an auto-download.

use std::ffi::OsString;
use std::path::Path;

use crate::paths::Store;
use crate::resolve::{Resolution, ResolveInput, resolve};
use crate::version::{is_safe_path_component, validate_install_version};

/// Linked directory identities since the last real compiler entry.
///
/// The compiler removes this internal transport variable before it starts threads.
pub const SHIM_ANCESTRY_ENV: &str = "_CHELISUP_LINKED_ANCESTRY";

/// Run the shim. `argv` is the full process argv (including `argv[0]`).
/// On success this never returns (it replaces the process image); any
/// return value is therefore an error exit code.
pub fn run_shim(argv: &[OsString]) -> i32 {
    let args = argv.get(1..).unwrap_or_default();

    let store = match Store::from_env() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("chelis: {e}");
            return 1;
        }
    };
    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("chelis: could not read the current directory: {e}");
            return 1;
        }
    };
    let env_toolchain = std::env::var("CHELIS_TOOLCHAIN").ok();

    let input = ResolveInput {
        args,
        env_toolchain,
        cwd: &cwd,
        store: &store,
    };

    let resolution = match resolve(&input) {
        Some(r) => r,
        None => {
            eprintln!("{}", no_toolchain_message(&store, &cwd));
            return 1;
        }
    };

    if !is_safe_path_component(&resolution.version) {
        eprintln!(
            "chelis: malformed toolchain version {:?} (resolved from {}); \
             expected a plain version such as 0.12.0",
            resolution.version, resolution.source
        );
        return 1;
    }

    let binary = store.toolchain_bin(&resolution.version);
    if !binary.is_file() {
        eprintln!("{}", not_installed_message(&store, &resolution));
        return 1;
    }

    #[cfg(unix)]
    let ancestry = match linked_dispatch_ancestry(&store, &resolution.version) {
        Ok(ancestry) => ancestry,
        Err(error) => {
            eprintln!("chelis: {error}");
            return 1;
        }
    };
    #[cfg(not(unix))]
    let ancestry: Option<String> = None;

    exec_toolchain(&binary, &resolution.forwarded_args, ancestry.as_deref())
}

#[cfg(unix)]
fn linked_dispatch_ancestry(store: &Store, version: &str) -> Result<Option<String>, String> {
    use std::os::unix::fs::MetadataExt;

    let Some(target) = store.linked_target(version) else {
        return Ok(None);
    };
    let metadata = std::fs::metadata(&target).map_err(|error| {
        format!(
            "cannot inspect linked toolchain {}: {error}",
            target.display()
        )
    })?;
    let mut ancestry = std::env::var(SHIM_ANCESTRY_ENV).unwrap_or_default();
    if !append_linked_ancestor(&mut ancestry, metadata.dev(), metadata.ino()) {
        return Err(format!(
            "shim recursion through linked toolchain {}: bin/chelis calls the shim again before a compiler starts",
            target.display()
        ));
    }
    Ok(Some(ancestry))
}

#[cfg(unix)]
fn append_linked_ancestor(ancestry: &mut String, device: u64, inode: u64) -> bool {
    use std::fmt::Write as _;

    if ancestry.split_whitespace().any(|entry| {
        entry.split_once(':').is_some_and(|(dev, ino)| {
            dev.parse::<u64>() == Ok(device) && ino.parse::<u64>() == Ok(inode)
        })
    }) {
        return false;
    }
    if !ancestry.is_empty() {
        ancestry.push(' ');
    }
    write!(ancestry, "{device}:{inode}").expect("write to a String");
    true
}

/// Loud error for "a version resolved, but it is not installed."
///
/// The remedy names `chelisup install <ver>` only when `chelisup install`
/// accepts `<ver>`. A selector such as `+latest` gets the accepted form
/// instead of a command that would fail in turn. A dangling `chelisup
/// link` names its target and the two remedies for it.
fn not_installed_message(store: &Store, resolution: &Resolution) -> String {
    if let Some(target) = store.linked_target(&resolution.version) {
        return format!(
            "chelis: toolchain {ver} (resolved from {src}) is linked to {target}, \
             which has no bin/chelis.\n  \
             link a toolchain again: chelisup link <dir>\n  \
             or remove the link: chelisup uninstall {ver}",
            ver = resolution.version,
            src = resolution.source,
            target = target.display(),
        );
    }
    let remedy = match validate_install_version(&resolution.version) {
        Ok(()) => format!("install it: chelisup install {}", resolution.version),
        Err(_) => "a toolchain version is an exact X.Y.Z (for example 0.12.0); \
                   install one with: chelisup install X.Y.Z"
            .to_string(),
    };
    format!(
        "chelis: toolchain {ver} (resolved from {src}) is not installed.\n  \
         installed: {installed}\n  {remedy}",
        ver = resolution.version,
        src = resolution.source,
        installed = installed_listing(store),
    )
}

/// Loud error for "nothing resolved at all."
fn no_toolchain_message(store: &Store, cwd: &Path) -> String {
    format!(
        "chelis: no chelis toolchain resolved.\n  \
         no applicable reef.toml compiler selection above {cwd}, \
         no chelis-toolchain file, CHELIS_TOOLCHAIN is unset, and no default is recorded.\n  \
         installed: {installed}\n  \
         pick a default: chelisup default <ver>  (after chelisup install <ver>)",
        cwd = cwd.display(),
        installed = installed_listing(store),
    )
}

fn installed_listing(store: &Store) -> String {
    let versions = store.installed_versions();
    if versions.is_empty() {
        "(none)".to_string()
    } else {
        versions.join(", ")
    }
}

/// Replace this process with the toolchain binary, with argv[0] set to the
/// versioned binary path.
#[cfg(unix)]
fn exec_toolchain(binary: &Path, args: &[OsString], ancestry: Option<&str>) -> i32 {
    use std::os::unix::process::CommandExt;
    let mut cmd = std::process::Command::new(binary);
    cmd.arg0(binary);
    cmd.args(args);
    if let Some(ancestry) = ancestry {
        cmd.env(SHIM_ANCESTRY_ENV, ancestry);
    }
    // `exec` only returns on failure.
    let err = cmd.exec();
    eprintln!("chelis: failed to exec {}: {err}", binary.display());
    127
}

/// Non-unix fallback: spawn and propagate the exit code. The toolchain
/// is unix-only in practice (the slugs are darwin/linux), so this exists
/// to keep the crate buildable, not as a supported path.
#[cfg(not(unix))]
fn exec_toolchain(binary: &Path, args: &[OsString], _ancestry: Option<&str>) -> i32 {
    match std::process::Command::new(binary).args(args).status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("chelis: failed to run {}: {e}", binary.display());
            127
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve::ToolchainSource;

    #[test]
    fn not_installed_message_is_loud_and_actionable() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::at(tmp.path());
        let res = Resolution {
            version: "0.99.0".to_string(),
            source: ToolchainSource::ReefPin("/proj/reef.toml".into()),
            forwarded_args: vec![],
        };
        let msg = not_installed_message(&store, &res);
        assert!(msg.contains("chelisup install 0.99.0"), "{msg}");
        assert!(msg.contains("0.99.0 (resolved from"), "{msg}");
        assert!(msg.contains("installed: (none)"), "{msg}");
    }

    #[test]
    fn not_installed_message_does_not_suggest_an_unparseable_version() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::at(tmp.path());
        let res = Resolution {
            version: "latest".to_string(),
            source: ToolchainSource::PlusArg,
            forwarded_args: vec![],
        };
        let msg = not_installed_message(&store, &res);
        assert!(!msg.contains("chelisup install latest"), "{msg}");
        assert!(msg.contains("chelisup install X.Y.Z"), "{msg}");
        assert!(msg.contains("latest (resolved from"), "{msg}");
    }

    #[test]
    fn no_toolchain_message_names_the_remedy() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::at(tmp.path());
        let msg = no_toolchain_message(&store, Path::new("/work"));
        assert!(msg.contains("no chelis toolchain resolved"), "{msg}");
        assert!(msg.contains("chelisup default"), "{msg}");
        assert!(msg.contains("/work"), "{msg}");
    }

    #[cfg(unix)]
    #[test]
    fn linked_ancestry_rejects_repeated_directory_identity() {
        let mut ancestry = String::new();
        assert!(append_linked_ancestor(&mut ancestry, 1, 2));
        assert!(append_linked_ancestor(&mut ancestry, 1, 3));
        assert!(!append_linked_ancestor(&mut ancestry, 1, 2));
        assert_eq!(ancestry, "1:2 1:3");
    }

    #[cfg(unix)]
    #[test]
    fn linked_ancestry_has_no_depth_limit() {
        let mut ancestry = String::new();
        for inode in 0..256 {
            assert!(append_linked_ancestor(&mut ancestry, 1, inode));
        }
        assert!(!append_linked_ancestor(&mut ancestry, 1, 255));
    }
}
