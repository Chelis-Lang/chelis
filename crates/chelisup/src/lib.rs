//! chelisup: a first-party, rustup-style toolchain installer and
//! version-routing shim for Chelis (chelis#164).
//!
//! # Two roles, one binary
//!
//! The published binary is installed at `~/.chelis/bin/chelisup` and a
//! second hard copy at `~/.chelis/bin/chelis`. [`dispatch`] inspects the
//! invocation name (`argv[0]`):
//!
//! - invoked as **`chelis`** it runs [`shim::run_shim`]: resolve the
//!   active toolchain at this invocation and `exec` the real compiler.
//!   This path does **no** network work and does not parse through clap,
//!   so it stays fast (it is on every `chelis` call).
//! - invoked as anything else (i.e. **`chelisup`**) it runs the
//!   installer CLI ([`cli::run_cli`]).
//!
//! This is the rustup proxy model: a single shipped binary, copied to
//! each proxy name. The alternative (a separate tiny `chelis-shim`
//! binary) keeps the shim image smaller but doubles the bootstrap
//! payload and forces `chelisup` to know where a second binary lives in
//! order to install the shim. The single-binary form means `chelisup`
//! installs the shim by copying its own `current_exe()`, which is the
//! simplest correct thing during a `curl | sh` bootstrap.
//!
//! # Store layout (override the root with `$CHELIS_HOME`)
//!
//! ```text
//! ~/.chelis/
//!   bin/chelis        the shim (a copy of the chelisup binary)
//!   bin/chelisup      the installer
//!   toolchains/<ver>/ side-by-side toolchains; <ver>/bin/chelis is real
//!   default           recorded default version (one bare line)
//! ```
//!
//! # Bootstrap
//!
//! `bootstrap/chelisup.sh` (the one shell carve-out in the repo; see
//! AGENTS.md) is the `curl -fsSL .../chelisup.sh | sh` one-liner. It
//! fetches the prebuilt `chelisup` for the host from the latest
//! `Chelis-Lang/chelis` release. The published asset is a bare
//! executable named `chelisup-<slug>` (slug in `darwin-arm64`,
//! `darwin-x86_64`, `linux-x86_64`): no version in the name and no
//! tarball, distinct from the toolchain tarball
//! `chelis-v<ver>-<build>.tar.gz` that `install` downloads (on Linux the
//! glibc-2.31 build, see [`install::release_build`]).

pub mod cli;
pub mod install;
pub mod paths;
pub mod resolve;
pub mod shim;
pub mod version;

use std::ffi::OsString;
use std::path::Path;

/// Inspect `argv[0]` and route to the shim or the installer CLI.
///
/// Returns the process exit code. The shim path never returns on
/// success (it `exec`s the toolchain), so a return from it is always an
/// error code.
pub fn dispatch() -> i32 {
    let argv: Vec<OsString> = std::env::args_os().collect();
    let invoked_as = argv
        .first()
        .map(Path::new)
        .and_then(Path::file_name)
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    // Only the exact name `chelis` is the shim. `chelisup`, the cargo
    // target path (`.../debug/chelisup`), and anything else run the CLI.
    if invoked_as == "chelis" {
        shim::run_shim(&argv)
    } else {
        cli::run_cli()
    }
}
