use bincode::Options;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellPackage {
    pub package: PackageId,
    pub compiler: String,
    pub modules: Vec<ShellModule>,
    pub dependencies: Vec<PackageId>,
    pub archive_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageId {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellModule {
    pub module: String,
    pub exports: Vec<ShellSymbol>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellSymbol {
    pub name: String,
    pub kind: SymbolKind,
    pub type_repr: Option<String>,
    pub effects: Vec<String>,
    pub has_body: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolKind {
    Value,
    Type,
    Macro,
    Dim,
}

pub fn encode_shell(shell: &ShellPackage) -> Result<Vec<u8>, bincode::Error> {
    bincode::serialize(shell)
}

pub fn decode_shell(bytes: &[u8]) -> Result<ShellPackage, bincode::Error> {
    let shell: ShellPackage = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .deserialize(bytes)?;
    validate_shell(&shell)?;
    let canonical = encode_shell(&shell)?;
    if canonical != bytes {
        return Err(validation_error(
            "shell bytes are not the canonical encoding of the decoded payload",
        ));
    }
    Ok(shell)
}

pub fn write_shell(path: &Path, shell: &ShellPackage) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = encode_shell(shell)?;
    fs::write(path, bytes)?;
    Ok(())
}

pub fn read_shell(path: &Path) -> Result<ShellPackage, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    Ok(decode_shell(&bytes)?)
}

/// Validate invariants carried by the complete CHB payload.
///
/// A CHB is read at a package-install trust boundary, so fields that are
/// not needed to locate the paired archive still need validation. This
/// rejects malformed metadata, ambiguous duplicate entries, and ordering
/// that cannot have been emitted by [`encode_shell`] from Reef's canonical
/// shell builder.
pub fn validate_shell(shell: &ShellPackage) -> Result<(), bincode::Error> {
    validate_nonempty_trimmed("package.name", &shell.package.name)?;
    validate_nonempty_trimmed("package.version", &shell.package.version)?;
    validate_compiler_pin(&shell.compiler)?;
    if !is_lowercase_hex_sha256(&shell.archive_sha256) {
        return Err(validation_error(
            "archive_sha256 must be exactly 64 lowercase hexadecimal characters",
        ));
    }

    validate_strict_order(
        "modules",
        shell.modules.iter().map(|module| module.module.as_str()),
    )?;
    for module in &shell.modules {
        validate_nonempty_trimmed("module name", &module.module)?;
        validate_strict_order(
            &format!("exports for module `{}`", module.module),
            module.exports.iter().map(|symbol| symbol.name.as_str()),
        )?;
        for symbol in &module.exports {
            validate_nonempty_trimmed("export name", &symbol.name)?;
            if symbol
                .type_repr
                .as_ref()
                .is_some_and(|repr| repr.is_empty())
            {
                return Err(validation_error(&format!(
                    "export `{}` in module `{}` has an empty type representation",
                    symbol.name, module.module
                )));
            }
            for effect in &symbol.effects {
                validate_nonempty_trimmed("effect name", effect)?;
            }
        }
    }

    validate_strict_order(
        "dependencies",
        shell
            .dependencies
            .iter()
            .map(|dependency| dependency.name.as_str()),
    )?;
    for dependency in &shell.dependencies {
        validate_nonempty_trimmed("dependency name", &dependency.name)?;
        if !dependency.version.is_empty() && dependency.version.trim() != dependency.version {
            return Err(validation_error(&format!(
                "dependency `{}` version must not have leading or trailing whitespace",
                dependency.name
            )));
        }
    }
    Ok(())
}

fn validate_nonempty_trimmed(label: &str, value: &str) -> Result<(), bincode::Error> {
    if value.is_empty() || value.trim() != value {
        return Err(validation_error(&format!(
            "{label} must be non-empty without leading or trailing whitespace"
        )));
    }
    Ok(())
}

fn validate_compiler_pin(pin: &str) -> Result<(), bincode::Error> {
    let Some(version) = pin.strip_prefix('=') else {
        return Err(validation_error(
            "compiler must be an exact pin beginning with `=`",
        ));
    };
    semver::Version::parse(version).map_err(|error| {
        validation_error(&format!(
            "compiler must be an exact `=<semver>` pin: {error}"
        ))
    })?;
    Ok(())
}

fn validate_strict_order<'a>(
    label: &str,
    values: impl Iterator<Item = &'a str>,
) -> Result<(), bincode::Error> {
    let mut previous: Option<&str> = None;
    for value in values {
        if previous.is_some_and(|prior| prior >= value) {
            return Err(validation_error(&format!(
                "{label} must be strictly sorted with no duplicates"
            )));
        }
        previous = Some(value);
    }
    Ok(())
}

fn is_lowercase_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validation_error(message: &str) -> bincode::Error {
    Box::new(bincode::ErrorKind::Custom(format!(
        "invalid shell envelope: {message}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_roundtrips() {
        let shell = fixture_shell();

        let bytes = encode_shell(&shell).expect("encode shell");
        let decoded = decode_shell(&bytes).expect("decode shell");
        assert_eq!(decoded, shell);
    }

    #[test]
    fn shell_decode_rejects_trailing_bytes() {
        let mut bytes = encode_shell(&fixture_shell()).expect("encode shell");
        bytes.extend_from_slice(b"junk");

        assert!(
            decode_shell(&bytes).is_err(),
            "CHB decoding must consume the complete input"
        );
    }

    #[test]
    fn shell_decode_rejects_every_truncation() {
        let bytes = encode_shell(&fixture_shell()).expect("encode shell");

        for end in 0..bytes.len() {
            assert!(
                decode_shell(&bytes[..end]).is_err(),
                "truncation at byte {end} unexpectedly decoded"
            );
        }
    }

    fn fixture_shell() -> ShellPackage {
        ShellPackage {
            package: PackageId {
                name: "chelis-std".to_string(),
                version: "0.1.0".to_string(),
            },
            compiler: "=0.1.21".to_string(),
            modules: vec![ShellModule {
                module: "App.Demo".to_string(),
                exports: vec![ShellSymbol {
                    name: "forward".to_string(),
                    kind: SymbolKind::Value,
                    type_repr: Some("(t-fn {} (t-prim {} f32) (t-prim {} f32))".to_string()),
                    effects: Vec::new(),
                    has_body: true,
                }],
            }],
            dependencies: Vec::new(),
            archive_sha256: "d".repeat(64),
        }
    }
}
