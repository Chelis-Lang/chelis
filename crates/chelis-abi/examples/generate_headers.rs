//! Render checked-in ABI fragments; `--write` updates, default checks freshness.

use std::path::{Path, PathBuf};

fn update(path: &Path, expected: &str, write: bool) -> Result<(), String> {
    let actual = std::fs::read_to_string(path).ok();
    if actual.as_deref() == Some(expected) {
        return Ok(());
    }
    if !write {
        return Err(format!("stale generated ABI artifact: {}", path.display()));
    }
    std::fs::create_dir_all(path.parent().expect("artifact parent"))
        .map_err(|error| error.to_string())?;
    std::fs::write(path, expected).map_err(|error| error.to_string())
}

fn main() -> Result<(), String> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let write = match arguments.as_slice() {
        [] => false,
        [argument] if argument == "--write" => true,
        _ => return Err("usage: generate_headers [--write]".to_string()),
    };
    let owner = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for (path, expected) in [
        (
            owner.join("../chelis-runtime/include/chelis_runtime_views.h"),
            chelis_abi::render::host_views_header(),
        ),
        (
            owner.join("../chelis-backend-hip/runtime/chelis_device_descriptor.h"),
            chelis_abi::render::device_descriptor_header(),
        ),
    ] {
        update(&path, &expected, write)?;
    }
    Ok(())
}
