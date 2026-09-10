//! Explicit runtime selection for native acceptance harnesses.
//! The Phase 1 runner supplies one archive selected from the current Cargo
//! build and checks its digest before and after execution. Outside that runner
//! the existing resolver remains responsible; this does not close #1354.
use std::path::PathBuf;

pub fn explicit() -> Option<PathBuf> {
    std::env::var_os("CHELIS_RUNTIME_DIR").map(|directory| {
        let archive = PathBuf::from(directory).join("libchelis_runtime.a");
        assert!(
            archive.is_file(),
            "explicit runtime archive is missing: {}",
            archive.display()
        );
        archive
    })
}
