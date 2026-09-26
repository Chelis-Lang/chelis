//! Explicit runtime selection for native acceptance harnesses that link the
//! runtime archive themselves. The runtime-representation runner names one
//! archive selected from the current Cargo build as the exact file
//! `CHELIS_RUNTIME_LIB` and checks its digest before and after execution.
//! `chelis build` never takes a runtime from the environment: it stages the
//! archive carried by its own build. Outside that runner each harness's existing
//! fallback remains responsible; this does not close #1354.
use std::path::PathBuf;

pub fn explicit() -> Option<PathBuf> {
    std::env::var_os("CHELIS_RUNTIME_LIB").map(|archive| {
        let archive = PathBuf::from(archive);
        assert!(
            archive.is_file(),
            "explicit runtime archive is missing: {}",
            archive.display()
        );
        archive
    })
}
