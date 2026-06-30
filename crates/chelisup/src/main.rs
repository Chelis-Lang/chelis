//! Entry point for the `chelisup` binary.
//!
//! The same binary serves two roles, chosen by `argv[0]` (the rustup
//! proxy model): invoked as `chelisup` it runs the installer CLI;
//! invoked as `chelis` it runs the fast pin-resolving shim. All logic
//! lives in the library so it is unit-testable without a process spawn.

fn main() {
    std::process::exit(chelisup::dispatch());
}
