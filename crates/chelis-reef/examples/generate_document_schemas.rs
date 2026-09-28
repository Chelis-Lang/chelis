use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = repository.join("docs/schemas/reef");
    fs::create_dir_all(&output)?;
    fs::write(
        output.join("manifest-v1.schema.json"),
        chelis_reef::manifest_schema_v1_json(),
    )?;
    fs::write(
        output.join("manifest-v2.schema.json"),
        chelis_reef::manifest_schema_v2_json(),
    )?;
    fs::write(
        output.join("manifest-v3.schema.json"),
        chelis_reef::manifest_schema_v3_json(),
    )?;
    fs::write(
        output.join("lock-v1.schema.json"),
        chelis_reef::lock_schema_v1_json(),
    )?;
    Ok(())
}
