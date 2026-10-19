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
//!   toolchains/<ver>/ side-by-side toolchains: a real <ver>/bin/chelis,
//!                     or a `chelisup link` to an external directory
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
//! musl build on a musl host, otherwise the static build when the release
//! publishes one, otherwise the glibc-2.31 build; see
//! [`install::release_builds`]).

pub mod cli;
pub mod install;
pub mod link;
pub mod nss;
pub mod paths;
pub mod resolve;
pub mod runtime_check;
pub mod shim;
pub mod version;

use std::borrow::Cow;
use std::ffi::OsString;
use std::path::Path;

/// How to get the latest chelisup while it cannot update itself. `chelisup
/// update` prints it, and so does the refusal of a release newer than the
/// running chelisup. [`UPGRADE_ADVICE_VAR`] replaces it.
pub(crate) const UPGRADE_ADVICE: &str =
    "re-run the bootstrap installer (the published `chelisup` one-liner)";

/// The environment variable that replaces [`UPGRADE_ADVICE`].
/// The Nix launcher sets this phrase because the bootstrap installer replaces it.
/// Unset, blank, or non-UTF-8 values retain the bootstrap advice.
/// The variable changes only the advice text.
pub(crate) const UPGRADE_ADVICE_VAR: &str = "CHELISUP_UPGRADE_ADVICE";

/// The configured upgrade advice, or the bootstrap advice.
pub(crate) fn upgrade_advice() -> Cow<'static, str> {
    std::env::var(UPGRADE_ADVICE_VAR)
        .ok()
        .filter(|advice| !advice.trim().is_empty())
        .map_or(Cow::Borrowed(UPGRADE_ADVICE), Cow::Owned)
}

/// How to use a toolchain that the Chelis flake builds where no release
/// toolchain runs: on a host that no release build supports, and on NixOS
/// without nix-ld for a dynamic release build.
pub(crate) const NIX_LINK_ADVICE: &str = "build chelis with Nix and register it with \
     chelisup link \"$(nix build --no-link --print-out-paths \
     github:Chelis-Lang/chelis/vX.Y.Z#chelis)\", with a release version for X.Y.Z";

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
