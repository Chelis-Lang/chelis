//! Effectful boundary for managed runtime identity production.
use chelis_runtime_identity::{RecordKind, UNSELECTED_COMPONENTS};
use std::{
    env,
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
};

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
    // Watch only entries the identity inventory can select. Tests, benches and
    // examples are never runtime inputs, so editing one must not rerun this
    // hook and rebuild the producer with everything that depends on it.
    let manifest = env::var_os("CARGO_MANIFEST_DIR")
        .ok_or_else(|| BuildError("CARGO_MANIFEST_DIR is absent".into()))?;
    for entry in watched_entries(Path::new(&manifest))? {
        println!("cargo:rerun-if-changed={}", entry.display());
    }
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

fn watched_entries(manifest: &Path) -> Result<Vec<PathBuf>, BuildError> {
    let mut watched = Vec::new();
    for entry in fs::read_dir(manifest)
        .map_err(|e| BuildError(format!("reading {}: {e}", manifest.display())))?
    {
        let entry = entry.map_err(|e| BuildError(e.to_string()))?;
        if !UNSELECTED_COMPONENTS.contains(&entry.file_name().to_string_lossy().as_ref()) {
            watched.push(entry.path());
        }
    }
    watched.sort();
    Ok(watched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn producer_hook_watches_only_selectable_entries() {
        let root = env::temp_dir().join(format!("chelis-producer-watch-{}", std::process::id()));
        for directory in ["src", "tests", "benches", "examples", "target"] {
            fs::create_dir_all(root.join(directory)).unwrap();
        }
        fs::write(root.join("Cargo.toml"), "").unwrap();
        fs::write(root.join("build.rs"), "").unwrap();
        let watched = watched_entries(&root);
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(
            watched.unwrap(),
            ["Cargo.toml", "build.rs", "src"].map(|name| root.join(name))
        );
    }
}
