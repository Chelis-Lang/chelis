//! The `chelisup` installer CLI (clap derive).
//!
//! Subcommands: `install`, `default`, `show`, `list-installed`, `which`,
//! `uninstall`, `update` (a documented stub), and `self {uninstall,
//! update}`.

use clap::{Parser, Subcommand};

use crate::install;
use crate::paths::Store;
use crate::resolve::{Resolution, ResolveInput, resolve};
use crate::version::{is_safe_path_component, validate_install_version};

#[derive(Parser)]
#[command(
    name = "chelisup",
    version = env!("CARGO_PKG_VERSION"),
    about = "The Chelis toolchain installer and version manager"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download and install a chelis toolchain
    Install {
        /// The version to install, as X.Y.Z (for example 0.12.0)
        version: String,
    },
    /// Set the recorded default toolchain (used outside any reef package)
    Default {
        /// An installed version, as X.Y.Z
        version: String,
    },
    /// Show the active toolchain resolution and the store layout
    Show,
    /// List installed toolchains
    ListInstalled,
    /// Print the toolchain binary that `chelis` resolves to here
    Which,
    /// Remove an installed toolchain
    Uninstall {
        /// The version to remove, as X.Y.Z
        version: String,
    },
    /// Update the installer itself (not yet implemented; see the message)
    Update,
    /// Manage the chelisup installer itself
    #[command(name = "self")]
    Zelf {
        #[command(subcommand)]
        command: SelfCommand,
    },
}

#[derive(Subcommand)]
enum SelfCommand {
    /// Remove the chelis shim and the chelisup installer copies
    Uninstall,
    /// Update the installer itself (not yet implemented; see the message)
    Update,
}

/// Parse argv and run the installer CLI. Returns the process exit code.
pub fn run_cli() -> i32 {
    let cli = Cli::parse();
    let store = match Store::from_env() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("chelisup: {e}");
            return 1;
        }
    };
    match cli.command {
        Command::Install { version } => cmd_install(&store, &version),
        Command::Default { version } => cmd_default(&store, &version),
        Command::Show => cmd_show(&store),
        Command::ListInstalled => cmd_list_installed(&store),
        Command::Which => cmd_which(&store),
        Command::Uninstall { version } => cmd_uninstall(&store, &version),
        Command::Update => cmd_update(),
        Command::Zelf { command } => match command {
            SelfCommand::Uninstall => cmd_self_uninstall(&store),
            SelfCommand::Update => cmd_update(),
        },
    }
}

fn cmd_install(store: &Store, version: &str) -> i32 {
    match install::install(store, version) {
        Ok(install::InstallOutcome::Installed { version, build }) => {
            println!(
                "installed chelis {version} ({build}) into {}",
                store.toolchain_dir(&version).display()
            );
            println!(
                "the chelis shim is at {} (add {} to your PATH)",
                store.shim_path().display(),
                store.bin_dir().display()
            );
            0
        }
        Ok(install::InstallOutcome::AlreadyInstalled { version }) => {
            println!("chelis {version} is already installed; refreshed the shim");
            0
        }
        Err(e) => {
            eprintln!("chelisup: install failed: {e}");
            1
        }
    }
}

fn cmd_default(store: &Store, version: &str) -> i32 {
    if let Err(e) = validate_install_version(version) {
        eprintln!("chelisup: {e}");
        return 1;
    }
    if !store.is_installed(version) {
        eprintln!(
            "chelisup: cannot set default to {version}: it is not installed.\n  \
             installed: {}\n  install it: chelisup install {version}",
            installed_or_none(store)
        );
        return 1;
    }
    if let Err(e) = store.write_default(version) {
        eprintln!("chelisup: {e}");
        return 1;
    }
    if let Err(e) = install::ensure_shim_installed(store) {
        // Non-fatal: the default was recorded; warn about the shim.
        eprintln!("chelisup: warning: could not refresh the shim: {e}");
    }
    println!("default toolchain set to {version}");
    0
}

fn cmd_show(store: &Store) -> i32 {
    println!("chelis home: {}", store.home().display());
    println!("shim:        {}", store.shim_path().display());
    match store.read_default() {
        Some(d) => println!("default:     {d}"),
        None => println!("default:     (unset)"),
    }
    println!("installed:   {}", installed_or_none(store));
    match resolve_for_cwd(store) {
        Some(r) => {
            let state = if store.is_installed(&r.version) {
                "installed"
            } else {
                "NOT INSTALLED"
            };
            println!("active here: {} (from {}) [{state}]", r.version, r.source);
        }
        None => println!("active here: (none resolved)"),
    }
    0
}

fn cmd_list_installed(store: &Store) -> i32 {
    let versions = store.installed_versions();
    if versions.is_empty() {
        println!("(no toolchains installed)");
    } else {
        let default = store.read_default();
        for v in versions {
            if default.as_deref() == Some(v.as_str()) {
                println!("{v} (default)");
            } else {
                println!("{v}");
            }
        }
    }
    0
}

fn cmd_which(store: &Store) -> i32 {
    let resolution = match resolve_for_cwd(store) {
        Some(r) => r,
        None => {
            eprintln!(
                "chelisup: no toolchain resolved here and no default recorded.\n  \
                 set one with: chelisup default <ver>"
            );
            return 1;
        }
    };
    if !is_safe_path_component(&resolution.version) {
        eprintln!(
            "chelisup: malformed toolchain version {:?} (from {})",
            resolution.version, resolution.source
        );
        return 1;
    }
    if !store.is_installed(&resolution.version) {
        eprintln!(
            "chelisup: toolchain {} (resolved from {}) is not installed.\n  \
             install it: chelisup install {}",
            resolution.version, resolution.source, resolution.version
        );
        return 1;
    }
    println!("{}", store.toolchain_bin(&resolution.version).display());
    0
}

fn cmd_uninstall(store: &Store, version: &str) -> i32 {
    match install::uninstall(store, version) {
        Ok(was_default) => {
            println!("removed chelis {version}");
            if was_default {
                println!(
                    "note: {version} was the default; the default is now unset \
                     (set a new one with: chelisup default <ver>)"
                );
            }
            0
        }
        Err(e) => {
            eprintln!("chelisup: {e}");
            1
        }
    }
}

fn cmd_self_uninstall(store: &Store) -> i32 {
    match install::self_uninstall(store) {
        Ok(removed) => {
            if removed.is_empty() {
                println!("nothing to remove (no shim or installer copies found)");
            } else {
                for p in &removed {
                    println!("removed {}", p.display());
                }
            }
            println!(
                "installed toolchains and the reef/src stores under {} were left in place; \
                 remove the whole directory to purge everything",
                store.home().display()
            );
            0
        }
        Err(e) => {
            eprintln!("chelisup: {e}");
            1
        }
    }
}

fn cmd_update() -> i32 {
    // Documented stub. Self-update is not implemented yet; re-running the
    // bootstrap is the supported upgrade path until it is.
    eprintln!(
        "chelisup: self-update is not implemented yet. To upgrade, re-run the bootstrap \
         installer (the published `chelisup` one-liner), which drops the latest binary at \
         ~/.chelis/bin/chelisup."
    );
    0
}

/// Resolve the active toolchain for the real cwd with no `+ver` args
/// (the CLI's `which`/`show` view of resolution).
fn resolve_for_cwd(store: &Store) -> Option<Resolution> {
    let cwd = std::env::current_dir().ok()?;
    let env_toolchain = std::env::var("CHELIS_TOOLCHAIN").ok();
    let input = ResolveInput {
        args: &[],
        env_toolchain,
        cwd: &cwd,
        store,
    };
    resolve(&input)
}

fn installed_or_none(store: &Store) -> String {
    let v = store.installed_versions();
    if v.is_empty() {
        "(none)".to_string()
    } else {
        v.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn clap_command_tree_is_valid() {
        // Catches derive mistakes (duplicate names, bad nesting) at test
        // time rather than first run.
        Cli::command().debug_assert();
    }

    #[test]
    fn self_subcommand_is_named_self() {
        let cmd = Cli::command();
        let names: Vec<&str> = cmd.get_subcommands().map(clap::Command::get_name).collect();
        assert!(names.contains(&"self"), "subcommands: {names:?}");
        assert!(names.contains(&"list-installed"), "subcommands: {names:?}");
    }
}
