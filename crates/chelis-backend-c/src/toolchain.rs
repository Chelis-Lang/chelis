//! Cross-platform toolchain probing for Chelis C-backend tests and harnesses.
//!
//! The project targets Linux/x86_64 with real GCC, where `gcc` is the right
//! default. On macOS, `gcc` is typically a wrapper around Apple clang, which
//! does not accept `-fopenmp`. This module picks a compiler that supports the
//! flags the codegen emits.

use std::process::Command;
use std::sync::OnceLock;

static C_COMPILER: OnceLock<String> = OnceLock::new();

// Ordered preference for a real OpenMP-capable GCC on macOS when `gcc` is
// Apple clang. Newer versions first. Homebrew installs these on PATH.
const MACOS_GCC_FALLBACKS: &[&str] = &["gcc-15", "gcc-14", "gcc-13", "gcc-12"];

/// Return the C compiler binary name to use for tests and harness compiles.
///
/// Resolution order:
/// 1. `CHELIS_TEST_CC` environment variable (explicit override).
/// 2. `gcc` when it is real GCC (Linux default; brew gcc aliased as `gcc`).
/// 3. On macOS, the first available `gcc-NN` from `MACOS_GCC_FALLBACKS`.
/// 4. `gcc` as a last resort so downstream `--version` probes report clearly.
pub fn c_compiler() -> String {
    C_COMPILER.get_or_init(resolve_c_compiler).clone()
}

fn resolve_c_compiler() -> String {
    if let Ok(override_cc) = std::env::var("CHELIS_TEST_CC")
        && !override_cc.is_empty()
    {
        return override_cc;
    }
    if is_real_gcc("gcc") {
        return "gcc".to_string();
    }
    if cfg!(target_os = "macos") {
        for candidate in MACOS_GCC_FALLBACKS {
            if is_real_gcc(candidate) {
                return (*candidate).to_string();
            }
        }
    }
    "gcc".to_string()
}

fn is_real_gcc(bin: &str) -> bool {
    let Ok(output) = Command::new(bin).arg("--version").output() else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    // Apple clang aliases `gcc` and reports "Apple clang version ...".
    // Real GCC reports "gcc (<distro/tag>) N.N.N".
    !text.contains("Apple clang") && !text.contains("clang version")
}
