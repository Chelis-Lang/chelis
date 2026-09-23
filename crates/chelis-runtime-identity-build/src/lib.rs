//! Effectful boundary for managed runtime identity production.
use chelis_runtime_identity::RecordKind;
use std::{env, error::Error, fmt, fs, path::PathBuf};

#[derive(Debug)]
pub struct BuildError(pub String);
impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for BuildError {}

/// Declare a producer; the managed compiler observer supplies retained record bytes.
/// This hook deliberately performs no Cargo invocation and derives no identity.
pub fn declare_producer(kind: RecordKind) -> Result<(), BuildError> {
    if env::var("CHELIS_IDENTITY_PROTOCOL").as_deref() != Ok("1") {
        return Err(BuildError("runtime identity production requires scripts/runtime_identity_build.py (managed observation protocol 1)".into()));
    }
    for name in [
        "CHELIS_IDENTITY_PROTOCOL",
        "CHELIS_IDENTITY_BACKEND",
        "CHELIS_IDENTITY_WORKSPACE",
        "CHELIS_IDENTITY_STATE",
        "CHELIS_IDENTITY_PROVENANCE",
        "PROFILE",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    // Track source membership without recursively watching Cargo's target tree.
    let manifest = env::var("CARGO_MANIFEST_DIR").map_err(|e| BuildError(e.to_string()))?;
    println!("cargo:rerun-if-changed={manifest}");
    let out = PathBuf::from(
        env::var_os("OUT_DIR").ok_or_else(|| BuildError("OUT_DIR is absent".into()))?,
    );
    let declaration = serde_json::json!({
        "protocol": 1, "kind": kind,
        "profile": env::var("PROFILE").map_err(|e| BuildError(e.to_string()))?,
        "rustflags": env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default(),
    });
    fs::write(
        out.join("chelis-runtime-identity-producer.json"),
        serde_json::to_vec(&declaration).map_err(|e| BuildError(e.to_string()))?,
    )
    .map_err(|e| BuildError(format!("writing producer declaration: {e}")))?;
    Ok(())
}
