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

/// Run the shim. `argv` is the full process argv (including `argv[0]`).
/// On success this never returns (it replaces the process image); any
/// return value is therefore an error exit code.
pub fn run_shim(argv: &[OsString]) -> i32 {
    let args: Vec<OsString> = argv.iter().skip(1).cloned().collect();

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
        args: &args,
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

    exec_toolchain(&binary, &resolution.forwarded_args)
}

/// Loud error for "a version resolved, but it is not installed."
///
/// The remedy names `chelisup install <ver>` only when `chelisup install`
/// accepts `<ver>`; a selector such as `+latest` gets the accepted form
/// instead of a command that would fail in turn.
fn not_installed_message(store: &Store, resolution: &Resolution) -> String {
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
fn exec_toolchain(binary: &Path, args: &[OsString]) -> i32 {
    use std::os::unix::process::CommandExt;
    let mut cmd = std::process::Command::new(binary);
    cmd.arg0(binary);
    cmd.args(args);
    // `exec` only returns on failure.
    let err = cmd.exec();
    eprintln!("chelis: failed to exec {}: {err}", binary.display());
    127
}

/// Non-unix fallback: spawn and propagate the exit code. The toolchain
/// is unix-only in practice (the slugs are darwin/linux), so this exists
/// to keep the crate buildable, not as a supported path.
#[cfg(not(unix))]
fn exec_toolchain(binary: &Path, args: &[OsString]) -> i32 {
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
}
